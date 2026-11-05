import copy
import unittest
import torch
from transformers.optimization import Adafactor
from dat_grpo.factored_optim import ChunkedAdafactor, stochastic_bf16


class FactoredOptimizerTests(unittest.TestCase):
    def test_matches_reference_over_steps_shapes_clipping_and_restore(self):
        torch.manual_seed(29)
        for shape in [(19,), (17,23), (3,9,11)]:
            expected = torch.nn.Parameter(torch.randn(shape))
            actual = torch.nn.Parameter(expected.detach().clone())
            reference = Adafactor([expected],lr=1e-3,relative_step=False,scale_parameter=False)
            chunked = ChunkedAdafactor([actual],lr=1e-3,chunk_elements=31,stochastic_round=False)
            for step in range(4):
                gradient = torch.randn(shape)*(step+1)
                expected.grad, actual.grad = gradient.clone(), gradient.clone()
                reference.step(); chunked.step()
                torch.testing.assert_close(actual,expected,atol=2e-7,rtol=1e-6)
            restored = torch.nn.Parameter(actual.detach().clone())
            optimizer = ChunkedAdafactor([restored],lr=1e-3,chunk_elements=31,stochastic_round=False)
            optimizer.load_state_dict(copy.deepcopy(chunked.state_dict()))
            actual.grad,restored.grad = gradient.clone(),gradient.clone()
            chunked.step(); optimizer.step()
            torch.testing.assert_close(actual,restored)

    def test_bf16_small_updates_are_finite_and_statistically_preserved(self):
        torch.manual_seed(17)
        value = torch.full((200000,),1.0+1e-4)
        rounded = stochastic_bf16(value)
        self.assertTrue(torch.isfinite(rounded).all())
        self.assertGreater(int((rounded!=1).sum()),0)
        self.assertAlmostEqual(float(rounded.float().mean()),float(value.mean()),delta=1e-5)

    def test_bf16_restore_preserves_fp32_statistics_exactly(self):
        parameter = torch.nn.Parameter(torch.ones(7,11,dtype=torch.bfloat16))
        optimizer = ChunkedAdafactor([parameter], stochastic_round=False)
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        saved = copy.deepcopy(optimizer.state_dict())
        restored = ChunkedAdafactor([parameter], stochastic_round=False)
        restored.load_state_dict(saved)
        for name in ('row','col'):
            self.assertEqual(restored.state[parameter][name].dtype,torch.float32)
            torch.testing.assert_close(restored.state[parameter][name], optimizer.state[parameter][name],rtol=0,atol=0)
