import copy
import hashlib
from pathlib import Path
import tempfile
import types
import unittest
import torch
from dat_grpo.disk_optim import DiskAdamW,snapshot_disk_state,restore_disk_state


class DiskOptimizerTests(unittest.TestCase):
    def test_adam_matches_reference_and_snapshots_remain_immutable(self):
        torch.manual_seed(91)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            actual=torch.nn.Parameter(torch.randn(11,17))
            expected=torch.nn.Parameter(actual.detach().clone())
            disk=DiskAdamW(actual,'layer.weight',root/'live',lr=1e-3,chunk_elements=31,stochastic_round=False)
            reference=torch.optim.AdamW([expected],lr=1e-3,weight_decay=0,foreach=False,fused=False)
            for step in range(3):
                gradient=torch.randn_like(actual)
                actual.grad,expected.grad=gradient.clone(),gradient.clone()
                disk.step();reference.step()
                torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)
                if step==0:
                    checkpoint=root/'checkpoint-1';checkpoint.mkdir()
                    snapshot_disk_state(types.SimpleNamespace(parameter_optimizers={'layer.weight':disk}),checkpoint)
                    old_file=checkpoint/'nvme-optimizer'/disk.filename
                    digest=hashlib.sha256(old_file.read_bytes()).hexdigest()
                    saved=copy.deepcopy(disk.state_dict());saved_weight=actual.detach().clone()
            self.assertEqual(hashlib.sha256(old_file.read_bytes()).hexdigest(),digest)
            restore_disk_state(checkpoint,root/'restored')
            restored=torch.nn.Parameter(saved_weight.clone())
            resume=DiskAdamW(restored,'layer.weight',root/'restored',lr=1e-3,chunk_elements=31,stochastic_round=False)
            resume.load_state_dict(saved)
            restored.grad=gradient.clone();resume.step()
            initial_reference=torch.nn.Parameter(saved_weight.clone())
            check=torch.optim.AdamW([initial_reference],lr=1e-3,weight_decay=0,foreach=False,fused=False)
            # The saved first-step moments are exact FP32 values in the checkpoint.
            import numpy as np
            moments=np.fromfile(old_file,dtype=np.float32).reshape(2,-1)
            check.state[initial_reference]=dict(step=torch.tensor(1.),exp_avg=torch.from_numpy(moments[0].copy()).reshape(11,17),exp_avg_sq=torch.from_numpy(moments[1].copy()).reshape(11,17))
            initial_reference.grad=gradient.clone();check.step()
            torch.testing.assert_close(restored,initial_reference)
