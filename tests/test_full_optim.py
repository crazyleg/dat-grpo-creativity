import copy
import unittest
import torch
from torch.utils.checkpoint import checkpoint
from dat_grpo.full_optim import StreamingFullOptimizer


class StreamingFullOptimizerTests(unittest.TestCase):
    def test_matches_full_batch_adamw_with_global_clipping_and_checkpointing(self):
        torch.manual_seed(11)
        reference = torch.nn.Sequential(torch.nn.Linear(7, 13), torch.nn.Tanh(), torch.nn.Linear(13, 5))
        model = copy.deepcopy(reference)
        expected = torch.optim.AdamW(reference.parameters(), lr=1e-3, weight_decay=0)
        streamed = StreamingFullOptimizer(model.named_parameters(), lambda p: torch.optim.AdamW([p], lr=1e-3, weight_decay=0), 1e-3, .5)
        inputs, targets = torch.randn(4, 7), torch.randn(4, 5)
        def loss(net):
            return (checkpoint(net, inputs, use_reentrant=False)-targets).square().mean()
        for _ in range(3):
            expected.zero_grad(set_to_none=True)
            loss(reference).backward()
            norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), .5)
            expected.step()
            streamed.begin_norm()
            loss(model).backward()
            actual_norm = streamed.finish_norm()
            self.assertAlmostEqual(actual_norm, float(norm), places=6)
            loss(model).backward()
            streamed.finish_backward()
            streamed.step()
            for before, after in zip(reference.parameters(), model.parameters()):
                torch.testing.assert_close(before, after, atol=1e-7, rtol=1e-6)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
        restored_model = copy.deepcopy(model)
        restored = StreamingFullOptimizer(restored_model.named_parameters(), lambda p: torch.optim.AdamW([p], lr=1e-3, weight_decay=0), 1e-3, .5)
        restored.load_state_dict(copy.deepcopy(streamed.state_dict()))
        self.assertEqual(restored.num_updates, 3)
        for net, optimizer in [(model, streamed), (restored_model, restored)]:
            optimizer.begin_norm(); loss(net).backward(); optimizer.finish_norm()
            loss(net).backward(); optimizer.finish_backward(); optimizer.step()
        for a,b in zip(model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(a,b)
        streamed.close(); restored.close()

    def test_nonfinite_gradient_cannot_update_weights(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = StreamingFullOptimizer([('weight',parameter)], lambda p: torch.optim.SGD([p], lr=.1), .1, .5)
        optimizer.begin_norm()
        (parameter*float('nan')).sum().backward()
        with self.assertRaises(FloatingPointError): optimizer.finish_norm()
        self.assertEqual(parameter.item(), 1.0)
        optimizer.close()

    def test_probe_indices_stay_in_bounds_for_billion_element_weights(self):
        from dat_grpo.full_backend import probe_indices
        for numel in [1, 8192, 67108864, 1271398400]:
            indices = probe_indices(numel,2048,'cpu')
            self.assertEqual(indices.dtype,torch.long)
            self.assertGreaterEqual(int(indices.min()),0)
            self.assertLess(int(indices.max()),numel)
            self.assertEqual(int(indices[-1]),numel-1)

    def test_failed_full_model_recipe_is_disabled_before_model_loading(self):
        import json
        from pathlib import Path
        from dat_grpo.train import validate_config
        config=json.loads((Path(__file__).resolve().parents[1]/'configs/full250.json').read_text())
        config['full_finetuning']['enabled'] = False
        config['full_finetuning']['disabled_reason'] = 'Recipe exhausted shared memory'
        self.assertFalse(config['full_finetuning']['enabled'])
        with self.assertRaisesRegex(ValueError,'exhausted shared memory'):
            validate_config(config)

    def test_full_saves_bound_shards_and_restore_method_after_failure(self):
        from dat_grpo.full_backend import trainer_class
        from unittest.mock import Mock
        class Base:
            def _save(self,output_dir=None,state_dict=None):
                return self.model.save_pretrained(output_dir,state_dict=state_dict)
        trainer=trainer_class(Base)()
        trainer.model=Mock()
        original=trainer.model.save_pretrained
        trainer._save('/checkpoint',{'weight':'tensor'})
        original.assert_called_once_with('/checkpoint',max_shard_size='1GB',state_dict={'weight':'tensor'})
        self.assertIs(trainer.model.save_pretrained,original)
        original.side_effect=OSError('disk error')
        with self.assertRaises(OSError): trainer._save('/checkpoint')
        self.assertIs(trainer.model.save_pretrained,original)

    def test_resume_uses_current_checkpoint_schedule_and_preserves_progress(self):
        from dat_grpo.full_backend import trainer_class
        from transformers.trainer_callback import TrainerState
        from types import SimpleNamespace
        class Base:
            def _init_training_state(self,*args,**kwargs):
                self.state=TrainerState(global_step=2,max_steps=1250,save_steps=1)
                return (0,2)
        trainer=trainer_class(Base)()
        trainer.args=SimpleNamespace(logging_steps=1,eval_steps=None,save_steps=250)
        self.assertEqual(trainer._init_training_state(),(0,2))
        self.assertEqual(trainer.state.global_step,2)
        self.assertEqual(trainer.state.save_steps,250)


class WholeGroupForwardTests(unittest.TestCase):
    def test_rejects_split_forward_before_streaming_backward(self):
        from dat_grpo.full_backend import trainer_class
        from types import SimpleNamespace
        from unittest.mock import patch
        from tempfile import TemporaryDirectory
        from pathlib import Path
        class Base:
            def compute_loss(self,model,inputs):
                return sum(model(input_ids=chunk).square().sum()
                    for chunk in inputs['completion_ids'].chunk(self.chunks))
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.ones(3))
            def forward(self,input_ids):
                return input_ids*self.weight
        trainer=trainer_class(Base)()
        trainer.state=SimpleNamespace(global_step=0)
        trainer.full_optimizer=SimpleNamespace(phase='norm')
        model=Model()
        inputs={'completion_ids':torch.ones(4,3)}
        with TemporaryDirectory() as directory, patch('torch.cuda.mem_get_info',return_value=(100,200)):
            trainer.args=SimpleNamespace(output_dir=str(Path(directory)/'checkpoints'),per_device_train_batch_size=4)
            trainer.chunks=4
            with self.assertRaisesRegex(RuntimeError,'one whole-group policy forward'):
                trainer.compute_loss(model,inputs)
            self.assertIsNone(model.weight.grad)
            trainer.chunks=1
            self.assertTrue(torch.isfinite(trainer.compute_loss(model,inputs)))
            self.assertFalse(model._forward_pre_hooks)
