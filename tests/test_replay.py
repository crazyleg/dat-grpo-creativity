import copy
import random
import unittest
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from dat_grpo.full_optim import StreamingFullOptimizer
from dat_grpo.replay import RandomReplay


class ReplayTests(unittest.TestCase):
    def test_two_pass_update_matches_one_pass_with_stochastic_forward(self):
        torch.manual_seed(51);random.seed(51);np.random.seed(51)
        expected=torch.nn.Sequential(torch.nn.Linear(7,13),torch.nn.Dropout(.25),torch.nn.Linear(13,3))
        actual=copy.deepcopy(expected)
        reference=torch.optim.AdamW(expected.parameters(),lr=1e-3,weight_decay=0)
        streamed=StreamingFullOptimizer(actual.named_parameters(),lambda p:torch.optim.AdamW([p],lr=1e-3,weight_decay=0),1e-3,.5)
        inputs=torch.randn(4,7)
        def loss(model):return checkpoint(model,inputs,use_reentrant=False).square().mean()*(1+random.random()+np.random.random())
        for step in range(3):
            replay=RandomReplay()
            reference.zero_grad();loss(expected).backward();torch.nn.utils.clip_grad_norm_(expected.parameters(),.5);reference.step()
            expected_rng=torch.get_rng_state().clone()
            replay.restore();repeat=RandomReplay()
            streamed.begin_norm();first=loss(actual);first.backward();streamed.finish_norm()
            self.assertTrue(repeat.advanced()['cpu']);repeat.restore()
            second=loss(actual);torch.testing.assert_close(first,second)
            second.backward();streamed.finish_backward();streamed.step()
            self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng))
            for before,after in zip(expected.parameters(),actual.parameters()):torch.testing.assert_close(before,after,atol=1e-7,rtol=1e-6)
        streamed.close()

    def test_retained_checkpointed_graph_matches_one_pass_and_uses_one_forward(self):
        torch.manual_seed(74)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.layers=torch.nn.Sequential(torch.nn.Linear(7,13),torch.nn.Dropout(.25),torch.nn.Linear(13,3));self.calls=0
            def forward(self,inputs):
                self.calls+=1
                return checkpoint(self.layers,inputs,use_reentrant=False)
        reference_model=Model();model=copy.deepcopy(reference_model)
        reference=torch.optim.AdamW(reference_model.parameters(),lr=1e-3,weight_decay=0)
        streamed=StreamingFullOptimizer(model.named_parameters(),lambda p:torch.optim.AdamW([p],lr=1e-3,weight_decay=0),1e-3,.5)
        inputs=torch.randn(4,7)
        for step in range(3):
            replay=RandomReplay()
            reference.zero_grad();reference_model(inputs).square().mean().backward()
            norm=torch.nn.utils.clip_grad_norm_(reference_model.parameters(),.5);reference.step()
            expected_rng=torch.get_rng_state().clone();replay.restore()
            streamed.begin_norm();loss=model(inputs).square().mean()
            loss.backward(retain_graph=True);actual_norm=streamed.finish_norm()
            self.assertAlmostEqual(float(norm),actual_norm,places=6)
            loss.backward();streamed.finish_backward();streamed.step()
            self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng))
            self.assertEqual(model.calls,step+1)
            for a,b in zip(model.parameters(),reference_model.parameters()):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
        streamed.close()

    def test_measured_bf16_loss_jitter_passes_but_large_changes_and_nonfinite_fail(self):
        from dat_grpo.full_backend import loss_values_agree
        self.assertTrue(loss_values_agree(torch.tensor(.01577088237),torch.tensor(.01574153453)))
        self.assertFalse(loss_values_agree(torch.tensor(.01577088237),torch.tensor(.02)))
        self.assertFalse(loss_values_agree(torch.tensor(.01),torch.tensor(float('nan'))))

    def test_different_gradient_norms_stop_the_uncheckpointed_update(self):
        parameter=torch.nn.Parameter(torch.ones(3))
        optimizer=StreamingFullOptimizer([('weight',parameter)],lambda p:torch.optim.SGD([p],lr=.01),.01,.5)
        optimizer.begin_norm();parameter.sum().backward();optimizer.finish_norm()
        (parameter*2).sum().backward()
        with self.assertRaisesRegex(FloatingPointError,'gradient norms disagree'):optimizer.finish_backward()
        with self.assertRaises(RuntimeError):optimizer.state_dict()
        optimizer.close()
