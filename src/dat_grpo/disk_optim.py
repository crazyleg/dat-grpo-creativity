"""Stream ordinary FP32 AdamW moments from NVMe without retaining them in RAM.

PyTorch AdamW computes each independent parameter slice. Moment files are
replaced atomically after an update, so hard-linked checkpoint states stay
immutable. BF16 policy weights use stochastic rounding after each FP32 update.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np
import torch
from .factored_optim import stochastic_bf16


class DiskAdamW(torch.optim.Optimizer):
    _staging = {}

    def __init__(self, parameter, name, directory, lr=1e-6, chunk_elements=2**23,
                 stochastic_round=True):
        super().__init__([parameter],dict(lr=lr))
        self.parameter, self.name = parameter, name
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.filename = hashlib.sha256(name.encode()).hexdigest()+'.bin'
        self.path = self.directory/self.filename
        self.chunk_elements, self.stochastic_round = chunk_elements, stochastic_round
        self.steps = 0

    def staging_buffers(self):
        key = (str(self.parameter.device),self.chunk_elements)
        if key not in self._staging:
            # Shared across all parameter optimizers, not one pair per tensor.
            self._staging[key] = [torch.empty(self.chunk_elements,dtype=torch.float32,
                device='cpu',pin_memory=True) for _ in range(2)]
        return self._staging[key]

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Closures are unsupported')
        parameter = self.parameter
        if parameter.grad is None:
            return
        if not parameter.is_contiguous() or not parameter.grad.is_contiguous():
            raise ValueError('Disk AdamW requires contiguous tensors')
        count = parameter.numel()
        required = 8*count
        if shutil.disk_usage(self.directory).free < required+10*2**30:
            raise RuntimeError('Insufficient SSD space for a transactional Adam update')
        old = np.memmap(self.path,mode='r',dtype=np.float32,shape=(2,count)) if self.steps else None
        temporary = self.path.with_suffix('.next')
        new = np.memmap(temporary,mode='w+',dtype=np.float32,shape=(2,count))
        flat, grad = parameter.view(-1), parameter.grad.view(-1)
        staging = self.staging_buffers() if parameter.device.type == 'cuda' else None
        try:
            for start in range(0,count,self.chunk_elements):
                stop = min(count,start+self.chunk_elements)
                local = torch.nn.Parameter(flat[start:stop].float().clone())
                local.grad = grad[start:stop].float().clone()
                optimizer = torch.optim.AdamW([local],lr=self.param_groups[0]['lr'],
                    betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False,fused=False)
                if old is None:
                    moments = [torch.zeros_like(local),torch.zeros_like(local)]
                elif staging is not None:
                    for i in range(2):
                        np.copyto(staging[i].numpy()[:stop-start],old[i,start:stop])
                    moments = [buffer[:stop-start].to(parameter.device,non_blocking=True) for buffer in staging]
                else:
                    # Copy is required even on CPU: the old checkpoint mapping is immutable.
                    moments = [torch.from_numpy(np.array(old[i,start:stop],copy=True)).to(
                        device=parameter.device,copy=True) for i in range(2)]
                optimizer.state[local] = dict(step=torch.tensor(float(self.steps)),
                    exp_avg=moments[0],exp_avg_sq=moments[1])
                optimizer.step()
                if staging is not None:
                    for i,moment in enumerate(moments):
                        staging[i][:stop-start].copy_(moment,non_blocking=True)
                    torch.cuda.current_stream(parameter.device).synchronize()
                    for i in range(2):
                        new[i,start:stop] = staging[i].numpy()[:stop-start]
                else:
                    for i,moment in enumerate(moments):
                        new[i,start:stop] = moment.cpu().numpy()
                value = local.detach()
                flat[start:stop].copy_(stochastic_bf16(value) if
                    self.stochastic_round and parameter.dtype == torch.bfloat16 else value)
                del local,optimizer,moments,value
            new.flush()
        finally:
            del new
            if old is not None:
                del old
        os.replace(temporary,self.path)
        # The statistics are safely on SSD; don't retain their pages in the host cache.
        with self.path.open('rb') as stream:
            os.fsync(stream.fileno())
            if hasattr(os,'posix_fadvise'):
                os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        self.steps += 1

    def state_dict(self):
        result = super().state_dict()
        result['disk_adamw'] = dict(version=1,name=self.name,shape=list(self.parameter.shape),
            steps=self.steps,filename=self.filename,chunk_elements=self.chunk_elements,
            stochastic_round=self.stochastic_round)
        return result

    def load_state_dict(self,state_dict):
        metadata = state_dict['disk_adamw']
        expected = self.state_dict()['disk_adamw']
        if any(metadata[k]!=expected[k] for k in ('version','name','shape','filename','chunk_elements','stochastic_round')):
            raise ValueError('NVMe Adam checkpoint recipe mismatch')
        if metadata['steps'] and (not self.path.is_file() or self.path.stat().st_size!=8*self.parameter.numel()):
            raise ValueError('NVMe Adam moment file is missing or incomplete')
        super().load_state_dict({k:v for k,v in state_dict.items() if k!='disk_adamw'})
        self.steps = metadata['steps']


def snapshot_disk_state(streaming_optimizer,checkpoint):
    destination = Path(checkpoint)/'nvme-optimizer'
    destination.mkdir(exist_ok=True)
    records = []
    for name,optimizer in streaming_optimizer.parameter_optimizers.items():
        if not isinstance(optimizer,DiskAdamW) or not optimizer.steps:
            continue
        target = destination/optimizer.filename
        if not target.exists():
            os.link(optimizer.path,target)
        records.append(dict(name=name,filename=optimizer.filename,steps=optimizer.steps,bytes=target.stat().st_size))
    (destination/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')


def restore_disk_state(checkpoint,directory):
    source,destination = Path(checkpoint)/'nvme-optimizer',Path(directory)
    records = json.loads((source/'manifest.json').read_text())
    destination.mkdir(parents=True,exist_ok=True)
    for record in records:
        filename = record['filename']
        if Path(filename).name != filename or not filename.endswith('.bin'):
            raise ValueError('Invalid optimizer state filename')
        path = source/filename
        if path.stat().st_size != record['bytes']:
            raise ValueError('Incomplete optimizer state file')
        temporary = destination/(filename+'.restore')
        if temporary.exists():
            temporary.unlink()
        os.link(path,temporary)
        os.replace(temporary,destination/filename)
