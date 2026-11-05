"""Full-weight GRPO backend, imported only after Unsloth patches the stack."""

import json
import time
from pathlib import Path
import torch

from .full_optim import StreamingFullOptimizer
from .replay import RandomReplay

# PyTorch testing defaults for BF16: rtol=0.016, atol=1e-5. The
# objective is FP32 but its Transformer forward arithmetic is BF16.
BF16_LOSS_RTOL = 0.016
BF16_LOSS_ATOL = 1e-5

def loss_values_agree(before,after):
    return bool(torch.isfinite(before).all() and torch.isfinite(after).all() and
        torch.allclose(before.float(),after.float(),rtol=BF16_LOSS_RTOL,atol=BF16_LOSS_ATOL))



def make_optimizer(model, config, run_dir=None, resume_from=None):
    settings = config['full_finetuning']
    lr = config['training']['learning_rate']
    if settings.get('optimizer') == 'disk_adamw':
        from .disk_optim import DiskAdamW, restore_disk_state
        directory = Path(run_dir)/'optimizer-live'
        if resume_from:
            restore_disk_state(resume_from,directory)
        names = {id(parameter):name for name,parameter in model.named_parameters()}
        factory = lambda parameter: DiskAdamW(parameter,names[id(parameter)],directory,lr=lr,
            chunk_elements=settings['optimizer_chunk_elements'],
            stochastic_round=settings['bf16_stochastic_round'])
    elif settings.get('optimizer') == 'chunked_adafactor':
        from .factored_optim import ChunkedAdafactor
        factory = lambda parameter: ChunkedAdafactor([parameter], lr=lr,
            chunk_elements=settings['optimizer_chunk_elements'],
            stochastic_round=settings['bf16_stochastic_round'])
    else:
        from torchao.optim import AdamW4bit
        factory = lambda parameter: AdamW4bit([parameter], lr=lr, betas=(.9,.999), eps=1e-8,
            weight_decay=0, bf16_stochastic_round=settings['bf16_stochastic_round'])
    return StreamingFullOptimizer(model.named_parameters(),
        factory, lr, settings['global_grad_clip'])


def memory_stage(trainer, stage):
    import psutil
    record = dict(stage=stage, step=trainer.state.global_step, time=time.time(),
        allocated_gib=torch.cuda.memory_allocated()/2**30,
        reserved_gib=torch.cuda.memory_reserved()/2**30,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        system_available_gib=psutil.virtual_memory().available/2**30,
        gradient_checkpointing=bool(getattr(trainer.model,'is_gradient_checkpointing',False)))
    with (Path(trainer.args.output_dir).parent/'memory-stages.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    print('FULL_STAGE',json.dumps(record),flush=True)


def backward_diagnostics(trainer,phase):
    optimizer = trainer.full_optimizer
    record = dict(phase=phase,step=trainer.state.global_step,
        norm_tensors_seen=len(optimizer.norm_seen),update_tensors_seen=len(optimizer.update_seen),
        retained_gradients=[{'name':name,'bytes':p.grad.numel()*p.grad.element_size()}
            for name,p in optimizer.named.items() if p.grad is not None],
        cuda_free_gib=torch.cuda.mem_get_info()[0]/2**30)
    with (Path(trainer.args.output_dir).parent/'backward-diagnostics.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    memory_stage(trainer,phase)



def trainer_class(base):
    class FullGRPOTrainer(base):
        def _init_training_state(self,*args,**kwargs):
            result = super()._init_training_state(*args,**kwargs)
            # Transformers restores old logging/save intervals from TrainerState.
            # A continued run must use the explicitly selected current schedule.
            self.state.compute_steps(self.args,self.state.max_steps)
            return result

        def _load_from_checkpoint(self,resume_from_checkpoint,model=None):
            from .loading import restore_full_weights
            restore_full_weights(self.model if model is None else model,resume_from_checkpoint)

        def log(self, logs, *args, **kwargs):
            # The outer optimizer has no retained gradients. Report the measured
            # norm instead of its otherwise misleading zero in native W&B charts.
            optimizer = getattr(self,'full_optimizer',None)
            if 'grad_norm' in logs and optimizer is not None and optimizer.last_grad_norm is not None:
                logs['grad_norm'] = optimizer.last_grad_norm
            return super().log(logs,*args,**kwargs)

        def _save(self, output_dir=None, state_dict=None):
            from functools import partial
            original = self.model.save_pretrained
            self.model.save_pretrained = partial(original,max_shard_size='1GB')
            try:
                return super()._save(output_dir,state_dict)
            finally:
                self.model.save_pretrained = original

        def compute_loss(self,model,inputs,*args,**kwargs):
            shapes=[]
            def capture(module,positional,keywords):
                ids=keywords.get('input_ids',positional[0] if positional else None)
                shapes.append(list(ids.shape) if ids is not None else None)
            handle=model.register_forward_pre_hook(capture,with_kwargs=True)
            try:
                loss=super().compute_loss(model,inputs,*args,**kwargs)
            finally:
                handle.remove()
                record=dict(step=self.state.global_step,phase=self.full_optimizer.phase,
                    forwards=shapes,completion_shape=list(inputs['completion_ids'].shape),
                    cuda_free_gib=torch.cuda.mem_get_info()[0]/2**30)
                with (Path(self.args.output_dir).parent/'policy-forwards.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
            if len(shapes)!=1 or shapes[0][0]!=self.args.per_device_train_batch_size:
                raise RuntimeError(f'Streaming full optimizer requires one whole-group policy forward, got {shapes}')
            return loss

        def memory_preflight(self,tokenizer,rendered,completion_tokens):
            # Deliberately synthetic: test the longest tensor shapes with no
            # updates, no learned moments, and the frozen reference still resident.
            from unsloth import FastModel
            model=self.model
            # Trainer normally enables this inside train(); preflight bypasses
            # that loop and must enter the same training/checkpointing mode.
            FastModel.for_training(model)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            model.train()
            if not model.is_gradient_checkpointing:
                raise RuntimeError('Preflight requires activation checkpointing')
            batch=self.args.per_device_train_batch_size
            device=next(model.parameters()).device
            prompt=torch.tensor(tokenizer.encode(rendered,add_special_tokens=False),device=device).repeat(batch,1)
            tokens=tokenizer.encode(' table river cloud stone',add_special_tokens=False)
            completion=torch.tensor([tokens[i%len(tokens)] for i in range(completion_tokens)],device=device).repeat(batch,1)
            inputs=dict(prompt_ids=prompt,prompt_mask=torch.ones_like(prompt),
                completion_ids=completion,completion_mask=torch.ones_like(completion),
                advantages=torch.linspace(-1,1,batch,device=device),
                ref_per_token_logps=torch.full((batch,completion_tokens),-5.,device=device))
            versions={name:p._version for name,p in model.named_parameters()}
            optimizer=self.full_optimizer
            optimizer.begin_norm()
            memory_stage(self,'preflight_before_loss')
            with self.compute_loss_context_manager():
                loss=self.compute_loss(model,inputs)
            if not torch.isfinite(loss.detach()).all():
                raise FloatingPointError('Non-finite preflight loss')
            memory_stage(self,'preflight_after_loss')
            try:
                self.accelerator.backward(loss)
            except BaseException:
                backward_diagnostics(self,'preflight_backward_failed')
                raise
            norm=optimizer.finish_norm()
            memory_stage(self,'preflight_after_backward')
            changed=[name for name,p in model.named_parameters() if p._version!=versions[name]]
            if changed or optimizer.num_updates or optimizer.update_seen or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError('Norm-only preflight must not update weights or retain gradients')
            return dict(synthetic=True,optimizer_updates=0,changed_parameters=changed,
                batch=batch,prompt_tokens=prompt.shape[1],completion_tokens=completion_tokens,
                gradient_tensors=len(optimizer.norm_seen),gradient_norm=norm,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                reference_resident=self.ref_model is not None)

        def training_step(self, model, inputs, num_items_in_batch=None):
            if self.args.gradient_accumulation_steps != 1 or self.args.n_gpu != 1:
                raise RuntimeError('The streaming full-weight backend requires one GPU and no accumulation')
            started = time.perf_counter()
            model.train()
            memory_stage(self, 'before_rollouts')
            inputs = self._prepare_inputs(inputs)
            memory_stage(self, 'after_rollouts_and_reference')
            if isinstance(inputs.get('prompt_ids'),torch.Tensor) and isinstance(inputs.get('completion_ids'),torch.Tensor):
                trace = {'optimizer_step_before_update':self.state.global_step,
                    'prompt_shape':list(inputs['prompt_ids'].shape),
                    'completion_shape':list(inputs['completion_ids'].shape),
                    'prompt_tail_ids':inputs['prompt_ids'][0,-32:].detach().cpu().tolist(),
                    'completion_head_ids':inputs['completion_ids'][0,:8].detach().cpu().tolist()}
                with (Path(self.args.output_dir).parent/'policy-inputs.jsonl').open('a') as stream:
                    stream.write(json.dumps(trace)+'\n')
            if not model.is_gradient_checkpointing:
                raise RuntimeError('Full GRPO requires activation checkpointing during the gradient forward')
            optimizer = self.full_optimizer
            optimizer.begin_norm()
            replay = RandomReplay()
            versions = {name:parameter._version for name,parameter in model.named_parameters()}
            memory_stage(self, 'before_norm_loss')
            with self.compute_loss_context_manager():
                norm_loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            memory_stage(self, 'after_norm_loss')
            if not torch.isfinite(norm_loss.detach()).all():
                raise FloatingPointError('Non-finite loss before full-model update')
            # Keep a private value before backward or compiled scratch buffers can
            # invalidate a detached view. Replay RNG to recompute the same function.
            before = norm_loss.detach().float().clone()
            try:
                self.accelerator.backward(norm_loss)
            except BaseException:
                backward_diagnostics(self,'norm_backward_failed')
                raise
            memory_stage(self, 'after_norm_backward')
            optimizer.finish_norm()
            after_backward = float(norm_loss.detach().float())
            del norm_loss
            rng_advanced = replay.advanced()
            replay.restore()
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            changed = [name for name,p in model.named_parameters() if p._version!=versions[name]]
            agreement = dict(step=self.state.global_step,norm_loss=float(before),
                norm_loss_after_backward=after_backward,update_loss=float(loss.detach()),
                rng_advanced_in_norm_pass=rng_advanced,parameters_changed_before_update=changed,
                abs_delta=abs(float(before)-float(loss.detach())),rtol=BF16_LOSS_RTOL,atol=BF16_LOSS_ATOL)
            agreement['allowed_delta'] = BF16_LOSS_ATOL+BF16_LOSS_RTOL*abs(agreement['update_loss'])
            agreement['passed'] = not changed and not optimizer.update_seen and loss_values_agree(before,loss.detach())
            run_dir=Path(self.args.output_dir).parent
            with (run_dir/'loss-agreement.jsonl').open('a') as stream:
                stream.write(json.dumps(agreement)+'\n')
            if not agreement['passed']:
                def cpu(value):
                    if isinstance(value,torch.Tensor):return value.detach().cpu()
                    if isinstance(value,dict):return {key:cpu(item) for key,item in value.items()}
                    if isinstance(value,list):return [cpu(item) for item in value]
                    if isinstance(value,tuple):return tuple(cpu(item) for item in value)
                    return value
                torch.save({'inputs':cpu(inputs),'rng_cpu':replay.cpu,'rng_cuda':replay.cuda,'diagnostics':agreement},
                    run_dir/f'loss-mismatch-step-{self.state.global_step}.pt')
                raise FloatingPointError(f'Two-pass loss mismatch: norm={agreement["norm_loss"]:.10g}, update={agreement["update_loss"]:.10g}, delta={agreement["abs_delta"]:.10g}, tolerance={agreement["allowed_delta"]:.10g}, changed_parameters={len(changed)}')
            self.accelerator.backward(loss)
            memory_stage(self, 'after_update_backward')
            optimizer.finish_backward()
            agreement.update(update_grad_norm=optimizer.last_update_grad_norm,
                gradient_norm_relative_delta=optimizer.last_gradient_norm_relative_delta)
            with (run_dir/'gradient-agreement.jsonl').open('a') as stream:
                stream.write(json.dumps(agreement)+'\n')
            # GRPO's generation buffer uses this counter; one whole generation group is one step.
            self._step += 1
            self._current_train_step_time += time.perf_counter()-started
            self._metrics['train']['step_time'].append(self._current_train_step_time)
            self._current_train_step_time = 0.0
            return loss.detach()
    return FullGRPOTrainer


def probe_indices(numel, count, device):
    if numel <= 0:
        raise ValueError('Cannot sample an empty parameter')
    count = min(count,numel)
    if count == 1:
        return torch.zeros(1,device=device,dtype=torch.long)
    # FP32 linspace rounds n-1 to n for large matrices; keep the arithmetic integral.
    return torch.div(torch.arange(count,device=device,dtype=torch.long)*(numel-1),count-1,rounding_mode='floor')


def parameter_samples(model, prompt_token_ids):
    samples = {}
    for name, parameter in model.named_parameters():
        category = ('embedding' if 'embed_tokens.weight' in name else
                    'lm_head' if name.endswith('lm_head.weight') else
                    'attention' if 'in_proj_qkv.weight' in name or 'q_proj.weight' in name else
                    'mlp' if 'gate_proj.weight' in name else None)
        if category is None or category in samples:
            continue
        if category == 'embedding':
            ids = sorted(set(prompt_token_ids))[:24]
            if not ids or min(ids)<0 or max(ids)>=parameter.shape[0]:
                raise ValueError('Prompt token IDs exceed the embedding vocabulary')
            indices = torch.tensor([row*parameter.shape[1]+col for row in ids for col in range(64)],device=parameter.device)
        else:
            indices = probe_indices(parameter.numel(),2048,parameter.device)
        values = parameter.detach().reshape(-1)[indices].float().cpu().clone()
        samples[category] = (name,parameter,indices,values)
    if set(samples) != {'embedding','lm_head','attention','mlp'}:
        raise RuntimeError('Missing full-weight update probes for a required language-model component')
    return samples


def sample_changes(samples):
    result = {}
    for category,(name,parameter,indices,initial) in samples.items():
        current = parameter.detach().reshape(-1)[indices].float().cpu()
        result[f'full/{category}_sample_changed_fraction'] = float((current!=initial).float().mean())
        result[f'full/{category}_sample_delta_rms'] = float((current-initial).square().mean().sqrt())
    return result
