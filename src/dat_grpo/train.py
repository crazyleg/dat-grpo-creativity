"""Train a QLoRA or full-weight policy with DAT GRPO and W&B monitoring."""

import argparse
import collections
from contextlib import nullcontext
import fcntl
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

from .common import sha256_file, utc_now, write_json
from .checkpoints import archive_adapter, archive_full_model, publish_full_final, full_snapshot_due


def record_generation_state(trainer, directory, phase, step):
    """Record effective generation settings without changing model or RNG state."""
    names = ['do_sample', 'temperature', 'top_p', 'top_k', 'max_new_tokens',
        'max_length', 'eos_token_id', 'pad_token_id', 'bos_token_id', 'use_cache',
        'bad_words_ids', 'suppress_tokens', 'begin_suppress_tokens']
    def fields(config):
        return {name:getattr(config,name,None) for name in names}
    path = Path(directory)/f'generation-state-{phase}-{step}.json'
    write_json(path, {'created_at':utc_now(), 'phase':phase, 'optimizer_step':step,
        'trainer_generation_config':fields(trainer.generation_config),
        'model_generation_config':fields(trainer.model.generation_config),
        'model_training':bool(trainer.model.training),
        'gradient_checkpointing':bool(getattr(trainer.model,'is_gradient_checkpointing',False))})
    return path


def validate_config(config):
    if config.get('adaptation','lora') not in ['lora','full']:
        raise ValueError('adaptation must be lora or full')
    training = config["training"]
    if config.get('memory_preflight') and config.get('adaptation') != 'full':
        raise ValueError('Memory preflight requires the full-weight backend')
    group = training["num_generations"]
    batch = training["per_device_train_batch_size"] * training["gradient_accumulation_steps"]
    if group < 2 or batch % group:
        raise ValueError("Effective single-GPU batch must be divisible by num_generations >= 2")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This initial Spark recipe supports exactly one training process")
    if training["use_vllm"]:
        raise ValueError("The Qwen3.8 Unsloth RL recipe requires use_vllm=False")
    conditional = 'word_classes' in config['task']
    if conditional:
        from .conditional_task import prompt_rows
        prompt_rows(config['task'])
        settings = config['reward']
        penalty = settings['wrong_class_penalty_per_word']
        if not math.isfinite(penalty) or penalty <= 0 or not math.isfinite(settings['invalid_reward']) or settings['invalid_reward'] > -10*penalty:
            raise ValueError('Invalid response reward must not evade class penalties')
        if training.get('remove_unused_columns') is not False:
            raise ValueError('Conditional DAT requires preserved task metadata')
    elif config["task"]["word_class"] not in ("nouns", "adjectives"):
        raise ValueError("word_class must be nouns or adjectives")
    if config["task"]["requested_words"] != 10 or config["task"]["scored_words"] != 7:
        raise ValueError("This scorer requires ten requested words and seven scored words")
    if training["max_completion_length"] >= config["max_seq_length"]:
        raise ValueError("Completion limit must leave room for the prompt")
    if config.get('adaptation', 'lora') == 'full':
        settings = config['full_finetuning']
        if not settings.get('enabled',False):
            raise ValueError(settings.get('disabled_reason','Full training is disabled until its memory plan is validated'))
        if training['gradient_accumulation_steps'] != 1 or training['per_device_train_batch_size'] != group:
            raise ValueError('Full streaming GRPO requires the entire generation group in one batch')
        if not training['bf16'] or training['max_grad_norm'] != 0 or training['lr_scheduler_type'] != 'constant':
            raise ValueError('Full streaming GRPO requires BF16, manual clipping, and constant LR')
        if settings.get('optimizer') in ('chunked_adafactor','disk_adamw'):
            if not 0 < settings['cuda_memory_fraction'] <= .74 or settings['min_available_gib'] < 18:
                raise ValueError('Full-model probe must reserve host memory and cap CUDA allocations')
            if not 0 < settings['optimizer_chunk_elements'] <= 2**23:
                raise ValueError('Factored optimizer chunks must stay bounded')
        elif settings['optimizer_state_bits'] != 4:
            raise ValueError('Unsupported full optimizer recipe')
        if not settings['bf16_stochastic_round'] or not settings['stream_gradients']:
            raise ValueError('Unsupported full optimizer recipe')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lora1500.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument('--restart-initialization', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    full = config.get('adaptation', 'lora') == 'full'
    conditional = 'word_classes' in config['task']
    if full and config['full_finetuning'].get('optimizer') in ('chunked_adafactor','disk_adamw'):
        parent = os.getppid()
        parent_command = Path(f'/proc/{parent}/cmdline').read_bytes()
        if os.environ.get('DAT_GRPO_WATCHDOG_PID') != str(parent) or b'guarded_process.py' not in parent_command:
            raise RuntimeError('This full-model recipe must run under scripts/launch.py and its independent memory watchdog')
    resume_step = 0
    if args.resume_from:
        args.resume_from = args.resume_from.resolve()
        checkpoint_state = json.loads((args.resume_from / "trainer_state.json").read_text())
        resume_step = int(checkpoint_state["global_step"])
        if resume_step >= config["training"]["max_steps"]:
            raise ValueError("The target max_steps must exceed the saved optimizer step")
        model_file = 'model.safetensors.index.json' if full else 'adapter_model.safetensors'
        for name in (model_file, "optimizer.pt", "scheduler.pt", "rng_state.pth"):
            if not (args.resume_from / name).is_file():
                raise ValueError(f"Resume checkpoint is missing {name}")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir/'restart-blocked.json').exists():
        raise ValueError('This failed run is blocked from restart; see restart-blocked.json and the memory incident report')
    if args.restart_initialization:
        if args.resume_from:
            raise ValueError('Initialization restart cannot also resume a checkpoint')
        old_status = json.loads((run_dir/'status.json').read_text())
        if old_status['status'] != 'failed' or old_status['optimizer_step'] != 0 or any((run_dir/'checkpoints').glob('checkpoint-*')):
            raise ValueError('Initialization restart is allowed only for a failed run with zero updates and no checkpoints')
        if json.loads((run_dir/'config.json').read_text()) != config:
            raise ValueError('Initialization restart requires unchanged experiment settings')
    lock = (run_dir / "training.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_lock = (run_dir.parent/'gpu-training.lock').open('w')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = run_dir / "status.json"
    status = {"status": "starting", "pid": os.getpid(), "started_at": utc_now(),
              "optimizer_step": resume_step, "target_optimizer_steps": config["training"]["max_steps"]}
    if args.resume_from:
        status["resumed_from"] = str(args.resume_from)
        if not full:
            archive_adapter(args.resume_from, run_dir / "adapter-snapshots", resume_step)
        elif full_snapshot_due(resume_step,config['full_finetuning']):
            archive_full_model(args.resume_from, run_dir/'model-snapshots', resume_step)
    session_name = status["started_at"].replace(":", "-")
    write_json(run_dir / "sessions" / session_name / "launch.json", {**status, "config": config})
    write_json(status_path, status)
    if (run_dir / "wandb-run.json").exists() and not (args.resume_from or args.restart_initialization):
        raise ValueError("An existing run requires --resume-from; use a new run directory otherwise")
    data_dir = args.data_dir.resolve()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_PROJECT", config["wandb"]["project"])
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    os.environ.setdefault("WANDB_WATCH", "false")
    # Triton 3.5 ships a CUDA 12.8 assembler; GB10 kernels require CUDA 13.
    system_ptxas = Path("/usr/local/cuda/bin/ptxas")
    if system_ptxas.is_file():
        os.environ.setdefault("TRITON_PTXAS_PATH", str(system_ptxas))
    run = None
    trainer = None
    try:
        # Unsloth must patch Transformers before the other training imports.
        import unsloth
        from unsloth import FastModel
        import torch
        import psutil
        import wandb
        from datasets import Dataset
        from transformers import TrainerCallback
        from trl import GRPOConfig, GRPOTrainer
        from .reward import DATReward, FrozenDAT

        if not torch.cuda.is_available():
            raise RuntimeError("A working CUDA GPU is required")
        if full and 'cuda_memory_fraction' in config['full_finetuning']:
            torch.cuda.set_per_process_memory_fraction(config['full_finetuning']['cuda_memory_fraction'], 0)
        frozen = FrozenDAT(data_dir / "dat/glove840b-dat.npz", data_dir / "dat/manifest.json")
        prepared_model = json.loads((data_dir / "model.json").read_text())
        for key in ("id", "revision"):
            if prepared_model[key] != config["model"][key]:
                raise ValueError(f"Prepared model {key} differs from the configuration")
        if conditional:
            from .conditional_task import prompt_rows, task_manifest, class_metrics
            from .word_classes import WordClasses
            from .reward import ConditionalDATReward
            source_rows = prompt_rows(config['task'])
            prompt = source_rows[0]['text']
            verifier = WordClasses(config['reward']['lexicon_file'], config['reward']['lexicon_manifest'])
            if verifier.manifest['lexicon_sha256'] != config['reward']['lexicon_sha256'] or verifier.manifest['source_archive_sha256'] != config['reward']['wordnet_archive_sha256']:
                raise ValueError('Word-class verifier differs from the pinned experiment assets')
            write_json(run_dir/'word-class-manifest.json',verifier.manifest)
            (run_dir/'prompt-templates.json').write_text(Path(config['task']['prompt_templates_file']).read_text())
        else:
            prompt_template = Path(config["task"]["prompt_file"]).read_text()
            prompt = prompt_template.format(word_class=config["task"]["word_class"])
        package_names = ["torch", "torchvision", "unsloth", "unsloth-zoo", "transformers", "trl", "peft", "bitsandbytes", "wandb", "numpy", "scipy", "flash-linear-attention"]
        versions = {name: importlib.metadata.version(name) for name in package_names}
        if full:
            versions['torchao'] = importlib.metadata.version('torchao')
        versions["triton_ptxas_path"] = os.environ.get("TRITON_PTXAS_PATH")
        versions['loader_flags'] = {name:os.environ.get(name) for name in
            ('HF_DEACTIVATE_ASYNC_LOAD','MALLOC_ARENA_MAX','UNSLOTH_SAFETENSORS_PREAD','UNSLOTH_COMPILE_DISABLE')}
        write_json(run_dir / "config.json", config)
        write_json(run_dir / "environment.json", versions)
        write_json(run_dir / "dat-manifest.json", frozen.manifest)
        (run_dir / "prompt.txt").write_text(prompt)
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:
            git_commit = None
        previous = json.loads((run_dir / "wandb-run.json").read_text()) if args.resume_from or args.restart_initialization else {}
        wandb_config = {
            **config, "environment": versions, "dat_artifact": frozen.manifest,
            "prompt_sha256": sha256_file(run_dir / "prompt.txt"),
            "git_commit": git_commit, "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "system_memory_gib": psutil.virtual_memory().total / 2**30,
            "reward_definition": (f"DAT/100 - {config['reward']['wrong_class_penalty_per_word']} per wrong/unverified word; invalid => {config['reward']['invalid_reward']}" if conditional else "official DAT score / 100; malformed output or <7 valid unique words => 0"),
            "word_class_enforced_by": "frozen WordNet noun/verb verifier; all ten output words" if conditional else "prompt only; official DAT dictionary does not enforce POS",
            "prompt_is_representative_only": conditional,
        }
        run = wandb.init(
            project=config["wandb"]["project"], entity=config["wandb"].get("entity"),
            id=previous.get("id"), resume="must" if args.resume_from or args.restart_initialization else None,
            allow_val_change=bool(args.resume_from or args.restart_initialization),
            name=run_dir.name, config=wandb_config, dir=str(run_dir),
            tags=["grpo", "qwen3.8-27b", ('full-bf16-'+config['full_finetuning'].get('optimizer','adam4bit')) if full else "qlora-nf4", "dat-nouns-verbs-conditional" if conditional else "dat-" + config["task"]["word_class"]] + (['memory-preflight'] if config.get('memory_preflight') else []),
            settings=wandb.Settings(init_timeout=90),
        )
        write_json(run_dir / "wandb-run.json", {"id": run.id, "url": run.url, "project": run.project, "entity": run.entity})
        run.define_metric("dat/optimizer_step")
        run.define_metric("dat/*", step_metric="dat/optimizer_step")
        run.define_metric("resources/*", step_metric="dat/optimizer_step")
        run.define_metric("timing/*", step_metric="dat/optimizer_step")
        run.define_metric('full/*', step_metric='dat/optimizer_step')
        if conditional:
            for namespace in ['pos/*','tasks/*','reward/*']:
                run.define_metric(namespace, step_metric='dat/optimizer_step')
        status.update(status="loading_model", wandb_url=run.url)
        write_json(status_path, status)
        print(f"WANDB_RUN_URL={run.url}", flush=True)
        run.summary["target_optimizer_steps"] = config["training"]["max_steps"]
        run.summary['execution_status'] = 'initializing'
        if args.resume_from:
            run.summary["resumed_from_optimizer_step"] = resume_step
            run.summary['completed_optimizer_steps'] = resume_step
            run.summary['latest_checkpoint_step'] = resume_step

        policy_path = Path(config['full_finetuning']['policy_path']).resolve() if full else prepared_model['snapshot']
        if full:
            export = json.loads((policy_path/'export-manifest.json').read_text())
            if any(export['source'][key] != config['model'][key] for key in ['id','revision']):
                raise ValueError('Full policy export does not match the frozen original backbone')
            write_json(run_dir/'initialization.json', export)
            run.config.update({'full_policy_initialization':export,'reference_model_source':prepared_model['snapshot']})
        from .loading import loading_trace
        with loading_trace(run_dir,torch) if full else nullcontext():
            model, processor = FastModel.from_pretrained(
                model_name=str(policy_path), max_seq_length=config["max_seq_length"],
                dtype=torch.bfloat16, load_in_4bit=not full, full_finetuning=full,
                fast_inference=False, offload_embedding=False, trust_remote_code=False,
                use_exact_model_name=True, device_map={"": 0},
                **({'float32_mixed_precision':False,'use_gradient_checkpointing':True} if full else {}),
            )
        if full:
            if getattr(model,'is_loaded_in_4bit',False) or any('lora_' in name for name,_ in model.named_parameters()):
                raise RuntimeError('Full policy must have unquantized trainable weights and no adapters')
            model.requires_grad_(True)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            torch.cuda.empty_cache()
            print('FULL_POLICY_MEMORY',{'allocated_gib':torch.cuda.memory_allocated()/2**30,
                'reserved_gib':torch.cuda.memory_reserved()/2**30,'system_available_gib':psutil.virtual_memory().available/2**30},flush=True)
        else:
            model = FastModel.get_peft_model(
                model, finetune_vision_layers=False, finetune_language_layers=True,
                finetune_attention_modules=True, finetune_mlp_modules=True,
                **config["lora"], bias="none", use_gradient_checkpointing="unsloth",
                random_state=config["training"]["seed"],
            )
        tokenizer = getattr(processor, "tokenizer", processor)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompt_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
        if conditional:
            rendered_cache = {}
            for row in source_rows:
                if row['text'] not in rendered_cache:
                    rendered_cache[row['text']] = tokenizer.apply_chat_template(
                        [{'role':'user','content':row['text']}],tokenize=False,
                        add_generation_prompt=True,enable_thinking=False)
            prompt_tokens = max(len(tokenizer.encode(text,add_special_tokens=False)) for text in rendered_cache.values())
            rows = [{key:row[key] for key in ['word_class','prompt_id','strict_class']}
                    | {'prompt':rendered_cache[row['text']]} for row in source_rows]
            with (run_dir/'training-prompts.jsonl').open('w') as stream:
                for row in rows:
                    stream.write(json.dumps(row)+'\n')
            manifest = task_manifest(config['task'],rows)
            manifest['rendered_dataset_sha256'] = sha256_file(run_dir/'training-prompts.jsonl')
            write_json(run_dir/'training-task-manifest.json',manifest)
            run.config.update({'training_tasks':manifest,'word_class_verifier':verifier.manifest})
        if prompt_tokens + config["training"]["max_completion_length"] > config["max_seq_length"]:
            raise ValueError("Prompt plus completion limit exceeds max_seq_length")
        (run_dir / "rendered-prompt.txt").write_text(rendered)
        if full:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
        else:
            trainable, total = model.get_nb_trainable_parameters()
        if not full and (trainable <= 0 or trainable >= total):
            raise RuntimeError("Expected a trainable LoRA adapter and a frozen base model")
        if full and trainable != total:
            raise RuntimeError('All full-policy parameters must be trainable')
        run.config.update({"trainable_parameters": trainable, "total_parameters": total, "prompt_tokens": prompt_tokens})
        if conditional:
            dataset = Dataset.from_list(rows)
            reward = ConditionalDATReward(frozen,run_dir/'rollouts.jsonl',config['training']['num_generations'],
                verifier,config['reward']['wrong_class_penalty_per_word'],config['reward']['invalid_reward'])
        else:
            dataset = Dataset.from_dict({"prompt": [rendered] * config["task"]["dataset_rows"]})
            reward = DATReward(frozen, run_dir / "rollouts.jsonl", config["training"]["num_generations"])
        full_optimizer = None
        samples = None
        if full:
            from .full_backend import make_optimizer, trainer_class, parameter_samples, sample_changes
            full_optimizer = make_optimizer(model,config,run_dir,args.resume_from)
            samples = parameter_samples(model,tokenizer.encode(rendered,add_special_tokens=False))
            run.config.update({'full_parameter_probes':{key:value[0] for key,value in samples.items()}})

        class Diagnostics(TrainerCallback):
            def __init__(self):
                self.last_step_time = time.monotonic()
                self.class_windows = {name:collections.deque(maxlen=100) for name in ['nouns','verbs']}
                self.class_seen = collections.Counter()
                if conditional and args.resume_from:
                    from .conditional_task import restore_class_history
                    self.class_windows, self.class_seen = restore_class_history(
                        run_dir/'rollouts.jsonl', resume_step, config['training']['num_generations'])

            def on_train_begin(self,training_args,state,control,**kwargs):
                # TrainerState restores the old intervals; apply the requested
                # continuation schedule after the checkpoint has been loaded.
                state.compute_steps(training_args, state.max_steps)
                if state.max_steps != config['training']['max_steps'] or state.global_step != resume_step:
                    raise RuntimeError('Trainer did not restore the selected continuation step and target')
                status['save_every_optimizer_steps'] = state.save_steps
                status['restored_optimizer_step'] = state.global_step
                if conditional:
                    status['restored_class_completions'] = dict(self.class_seen)
                if full and conditional:
                    record_generation_state(trainer,run_dir/'sessions'/session_name,'train-begin',state.global_step)
                write_json(status_path,status)

            def on_step_end(self, training_args, state, control, **kwargs):
                status.update(status="training", optimizer_step=state.global_step, updated_at=utc_now())
                run.summary['completed_optimizer_steps'] = state.global_step
                run.summary['execution_status'] = 'training'
                if state.global_step in config.get('early_checkpoint_steps',[]):
                    control.should_save = True
                if full:
                    if full_optimizer.num_updates != state.global_step:
                        raise RuntimeError('Trainer and streaming optimizer step counters diverged')
                    if state.global_step == config['full_finetuning']['first_checkpoint_step'] or full_snapshot_due(state.global_step,config['full_finetuning']):
                        control.should_save = True
                write_json(status_path, status)

            def on_save(self, training_args, state, control, **kwargs):
                checkpoint = Path(training_args.output_dir)/f'checkpoint-{state.global_step}'
                if full and config['full_finetuning'].get('optimizer') == 'disk_adamw':
                    from .disk_optim import snapshot_disk_state
                    snapshot_disk_state(full_optimizer,checkpoint)
                status.update(last_checkpoint_step=state.global_step,last_checkpoint=str(checkpoint))
                if not full:
                    snapshot = archive_adapter(checkpoint, run_dir/'adapter-snapshots',state.global_step)
                    status['last_adapter_snapshot'] = str(snapshot)
                elif full_snapshot_due(state.global_step,config['full_finetuning']):
                    snapshot = archive_full_model(checkpoint,run_dir/'model-snapshots',state.global_step)
                    status['last_model_snapshot'] = str(snapshot)
                write_json(status_path, status)

                run.summary['latest_checkpoint_step'] = state.global_step
                if full and conditional:
                    record_generation_state(trainer,run_dir/'sessions'/session_name,'after-save',state.global_step)

            def on_log(self, training_args, state, control, logs=None, **kwargs):
                for key in ("loss", "grad_norm"):
                    value = (logs or {}).get(key)
                    if isinstance(value, (int, float)) and not math.isfinite(value):
                        raise FloatingPointError(f"Non-finite {key}: stopping this run")
                records = reward.drain()
                now = time.monotonic()
                metrics = {
                    "dat/optimizer_step": state.global_step,
                    "resources/cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                    "resources/cuda_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                    "resources/cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "resources/process_rss_gib": psutil.Process().memory_info().rss / 2**30,
                    "resources/system_available_gib": psutil.virtual_memory().available / 2**30,
                    "timing/seconds_since_last_log": now - self.last_step_time,
                }
                self.last_step_time = now
                if full and full_optimizer.last_grad_norm is not None:
                    metrics.update({'full/gradient_norm':full_optimizer.last_grad_norm,
                        'full/update_gradient_norm':full_optimizer.last_update_grad_norm,
                        'full/gradient_norm_relative_delta':full_optimizer.last_gradient_norm_relative_delta,
                        'full/updated_tensors':len(full_optimizer.update_seen),
                        'full/parameters_with_gradients':sum(full_optimizer.named[name].numel() for name in full_optimizer.update_seen),
                        **sample_changes(samples)})
                if records:
                    valid = [r["score"] for r in records if r["score"] is not None]
                    metrics.update({
                        "dat/valid_fraction": len(valid) / len(records),
                        "dat/reward_mean": statistics.mean(r["reward"] for r in records),
                        "dat/unique_response_fraction": len({tuple(w.lower() for w in r["words"]) for r in records}) / len(records),
                        "dat/unique_selected_words": len({w for r in records for w in r["selected_words"]}),
                    })
                    if valid:
                        metrics.update({"dat/score_mean_valid": statistics.mean(valid), "dat/score_min_valid": min(valid), "dat/score_max_valid": max(valid)})
                    groups = {}
                    for record in records:
                        groups.setdefault((record["reward_call"], record["group"]), []).append(record["reward"])
                    spreads = [statistics.pstdev(g) for g in groups.values()]
                    metrics["dat/group_reward_std_mean"] = statistics.mean(spreads)
                    metrics["dat/zero_variance_group_fraction"] = sum(s == 0 for s in spreads) / len(spreads)
                    if conditional:
                        diagnostic = class_metrics(records)
                        metrics.update({'reward/composite_mean':diagnostic['reward_mean'],
                            'reward/base_dat_mean':diagnostic['base_reward_mean'],
                            'reward/class_penalty_mean':diagnostic['penalty_mean'],
                            **{f'pos/{key}':value for key,value in diagnostic.items() if key not in ['reward_mean','base_reward_mean','penalty_mean']}})
                        for record in records:
                            self.class_windows[record['word_class']].append(record)
                            self.class_seen[record['word_class']] += 1
                        for name, window in self.class_windows.items():
                            if window:
                                metrics[f'tasks/{name}/seen_completions'] = self.class_seen[name]
                                metrics.update({f'tasks/{name}/rolling100/{key}':value for key,value in class_metrics(window).items()})
                    if state.global_step % config["sample_table_interval"] == 0 or state.global_step == 1:
                        if conditional:
                            keys = ['word_class','prompt_id','strict_class','score','reward','class_correct_words',
                                    'class_wrong_words','class_penalty','invalid_penalty','completion']
                            metrics['samples/completions'] = wandb.Table(columns=['optimizer_step',*keys,'word_checks'],
                                data=[[state.global_step,*[r[key] for key in keys],json.dumps(r['word_checks'])] for r in records])
                        else:
                            metrics["samples/completions"] = wandb.Table(
                            columns=["optimizer_step", "completion", "dat_score", "reward", "selected_words", "invalid_reason"],
                            data=[[state.global_step, r["completion"], r["score"], r["reward"], json.dumps(r["selected_words"]), r["reason"]] for r in records],
                            )
                run.log(metrics)
                write_json(run_dir / "last-metrics.json", {"optimizer_step": state.global_step, **(logs or {}), **{k: v for k, v in metrics.items() if k != "samples/completions"}})

        training_args = GRPOConfig(
            output_dir=str(run_dir / "checkpoints"), run_name=run_dir.name,
            **config["training"], logging_first_step=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_pin_memory=False,
            # This argument counts chunks, not rows. Multiple policy forwards
            # retain shared-parameter gradients until all branches accumulate.
            **({'unsloth_grpo_mini_batch':1,'unsloth_logit_chunk_multiplier':4} if full else {}),
        )
        trainer_type = trainer_class(GRPOTrainer) if full else GRPOTrainer
        kwargs = {}
        original_policy_id = model.config._name_or_path
        if full:
            from transformers import BitsAndBytesConfig
            # TRL reloads its frozen reference from this ID. The already-loaded policy stays BF16.
            model.config._name_or_path = prepared_model['snapshot']
            training_args.model_init_kwargs = {'dtype':torch.bfloat16,'device_map':{'':0},'local_files_only':True}
            kwargs = {'optimizers':(full_optimizer,None),'quantization_config':BitsAndBytesConfig(
                load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)}
        try:
            trainer = trainer_type(
                model=model, processing_class=processor, reward_funcs=reward,
                args=training_args, train_dataset=dataset, callbacks=[Diagnostics()], **kwargs,
            )
        finally:
            model.config._name_or_path = original_policy_id
        if full:
            trainer.full_optimizer = full_optimizer
            if trainer.ref_model is None or not getattr(trainer.ref_model,'is_loaded_in_4bit',False):
                raise RuntimeError('Full GRPO requires the frozen NF4 reference to fit this Spark')
            trainer.ref_model.requires_grad_(False)
            run.config.update({'reference_trainable_parameters':sum(p.numel() for p in trainer.ref_model.parameters() if p.requires_grad),
                'reference_loaded_in_4bit':True,'policy_loaded_in_4bit':False,'manual_global_grad_clip':config['full_finetuning']['global_grad_clip']})
        status.update(status="training_initialization", trainable_parameters=trainable)
        write_json(status_path, status)
        if config.get('memory_preflight'):
            if args.resume_from:
                raise ValueError('Memory preflight uses a separate run, not a training continuation')
            status.update(status='memory_preflight')
            write_json(status_path,status)
            longest_prompt = max(rendered_cache.values(),key=lambda text:len(tokenizer.encode(text,add_special_tokens=False))) if conditional else rendered
            result = trainer.memory_preflight(tokenizer,longest_prompt,config['training']['max_completion_length'])
            write_json(run_dir/'memory-preflight.json',result)
            status.update(status='memory_preflight_passed',optimizer_step=0,completed_at=utc_now())
            write_json(status_path,status)
            run.summary['memory_preflight'] = result
            run.summary['execution_status'] = 'memory_preflight_passed'
            run.summary['completed_optimizer_steps'] = 0
            run.finish()
            return
        trainer.train(resume_from_checkpoint=str(args.resume_from) if args.resume_from else None)
        final_directory = run_dir/('model-final' if full else 'adapter-final')
        final_checkpoint = run_dir/'checkpoints'/f'checkpoint-{trainer.state.global_step}'
        if full and (final_checkpoint/'trainer_state.json').is_file():
            publish_full_final(final_checkpoint,final_directory,trainer.state.global_step)
        else:
            trainer.save_model(str(final_directory))
        processor.save_pretrained(str(final_directory))
        status.update(status="completed", optimizer_step=trainer.state.global_step, completed_at=utc_now())
        write_json(status_path, status)
        run.summary["completed_optimizer_steps"] = trainer.state.global_step
        run.summary['execution_status'] = 'completed'
        run.finish()
    except BaseException as exc:
        status.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=f"{type(exc).__name__}: {exc}", updated_at=utc_now())
        write_json(status_path, status)
        if run is not None:
            run.summary["failure"] = status["error"]
            run.summary['execution_status'] = status['status']
            run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
