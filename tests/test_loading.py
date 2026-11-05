import json
from pathlib import Path
import tempfile
import unittest
import torch
from safetensors.torch import save_file
from dat_grpo.loading import restore_full_weights


class LoadingTests(unittest.TestCase):
    def test_direct_restore_preserves_parameter_objects_and_gradient_hooks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            model=torch.nn.Linear(7,5).to(dtype=torch.bfloat16)
            expected={name:torch.randn_like(value) for name,value in model.state_dict().items()}
            save_file(expected,str(root/'weights.safetensors'))
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{name:'weights.safetensors' for name in expected}}))
            identities={name:id(p) for name,p in model.named_parameters()}
            seen=[]
            handle=model.weight.register_post_accumulate_grad_hook(lambda p:seen.append(id(p)))
            restore_full_weights(model,root)
            for name,p in model.named_parameters():
                self.assertEqual(id(p),identities[name]);torch.testing.assert_close(p,expected[name])
            model(torch.ones(2,7,dtype=torch.bfloat16)).float().sum().backward()
            self.assertEqual(seen,[identities['weight']]);handle.remove()
