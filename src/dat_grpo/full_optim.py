"""Stream full-parameter gradients through a two-pass, globally clipped update.

The first backward computes the global norm and releases each parameter's grad.
The second recomputes the BF16 loss and updates each parameter after its
gradient has accumulated. Both passes check their gradient norms agree. This requires one complete batch per step, no dropout, no tied
parameters spanning reentrant backward calls, and no gradient accumulation.
"""

import math
import torch


def gradient_squared_norm(gradient, chunk_elements=16*1024*1024):
    flat = gradient.detach().reshape(-1)
    total = torch.zeros((),device=gradient.device,dtype=torch.float64)
    for start in range(0,flat.numel(),chunk_elements):
        norm = torch.linalg.vector_norm(flat[start:start+chunk_elements],dtype=torch.float32)
        total.add_(norm.double().square())
    return total


class StreamingFullOptimizer(torch.optim.Optimizer):
    def __init__(self, named_parameters, factory, lr, max_grad_norm):
        self.named = {name: parameter for name, parameter in named_parameters if parameter.requires_grad}
        if not self.named:
            raise ValueError('No trainable parameters')
        if len({id(parameter) for parameter in self.named.values()}) != len(self.named):
            raise ValueError('Duplicate/shared parameters are not supported by this recipe')
        super().__init__(list(self.named.values()), {'lr': lr})
        self.parameter_optimizers = {name: factory(parameter) for name, parameter in self.named.items()}
        self.max_grad_norm = max_grad_norm
        self.phase = 'idle'
        self.norm_seen, self.update_seen = set(), set()
        device = next(iter(self.named.values())).device
        self.norm_squared = torch.zeros((), device=device, dtype=torch.float64)
        self.update_norm_squared = torch.zeros((),device=device,dtype=torch.float64)
        self.clip_coefficient = torch.ones((), device=device, dtype=torch.float32)
        self.last_grad_norm = None
        self.last_update_grad_norm = None
        self.last_gradient_norm_relative_delta = None
        self.num_updates = 0
        self.handles = [parameter.register_post_accumulate_grad_hook(self._hook(name)) for name, parameter in self.named.items()]

    def _hook(self, name):
        def hook(parameter):
            gradient = parameter.grad
            if gradient is None:
                return
            if self.phase == 'norm':
                if name in self.norm_seen:
                    raise RuntimeError(f'Reentrant/repeated gradient accumulation for {name}')
                # Bound temporary dtype-conversion storage even for the billion-element LM head.
                self.norm_squared.add_(gradient_squared_norm(gradient))
                self.norm_seen.add(name)
            elif self.phase == 'update':
                if name not in self.norm_seen or name in self.update_seen:
                    raise RuntimeError(f'Unexpected gradient during streaming update: {name}')
                self.update_norm_squared.add_(gradient_squared_norm(gradient))
                gradient.mul_(self.clip_coefficient)
                self.parameter_optimizers[name].step()
                self.update_seen.add(name)
            else:
                raise RuntimeError(f'Gradient arrived while optimizer phase was {self.phase}')
            parameter.grad = None
        return hook

    def begin_norm(self):
        if self.phase != 'idle':
            raise RuntimeError('Previous update did not finish')
        if any(parameter.grad is not None for parameter in self.named.values()):
            raise RuntimeError('Accumulated gradients are not supported')
        self.norm_seen.clear()
        self.update_seen.clear()
        self.norm_squared.zero_()
        self.update_norm_squared.zero_()
        self.phase = 'norm'
        lr = float(self.param_groups[0]['lr'])
        for optimizer in self.parameter_optimizers.values():
            for group in optimizer.param_groups:
                if isinstance(group['lr'], torch.Tensor):
                    group['lr'].fill_(lr)
                else:
                    group['lr'] = lr

    def finish_norm(self):
        norm = float(self.norm_squared.sqrt().item())
        if not self.norm_seen or not math.isfinite(norm):
            raise FloatingPointError('Missing or non-finite full-model gradients; no weights updated')
        self.last_grad_norm = norm
        coefficient = min(1.0, self.max_grad_norm / (norm + 1e-6)) if self.max_grad_norm else 1.0
        self.clip_coefficient.fill_(coefficient)
        self.phase = 'update'
        return norm

    def finish_backward(self):
        if self.phase != 'update' or self.norm_seen != self.update_seen:
            raise RuntimeError('The two backward passes did not cover the same parameters')
        if any(parameter.grad is not None for parameter in self.named.values()):
            raise RuntimeError('A full gradient buffer was retained')
        update_norm = float(self.update_norm_squared.sqrt().item())
        self.last_update_grad_norm = update_norm
        self.last_gradient_norm_relative_delta = abs(update_norm-self.last_grad_norm)/max(self.last_grad_norm,1e-12)
        if not math.isfinite(update_norm) or abs(update_norm-self.last_grad_norm)>1e-6+.016*self.last_grad_norm:
            raise FloatingPointError(f'Two-pass gradient norms disagree: norm_pass={self.last_grad_norm}, update_pass={update_norm}; discard this uncheckpointed update')
        self.phase = 'ready'

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None or self.phase != 'ready':
            raise RuntimeError('Streaming optimizer requires its completed two-pass backward')
        self.num_updates += 1
        self.phase = 'idle'

    def state_dict(self):
        if self.phase != 'idle':
            raise RuntimeError('Only complete optimizer steps can be checkpointed')
        result = super().state_dict()
        result['streaming_full'] = {'version': 1, 'num_updates': self.num_updates,
            'max_grad_norm': self.max_grad_norm,
            'parameter_optimizers': {name: optimizer.state_dict() for name, optimizer in self.parameter_optimizers.items()}}
        return result

    def load_state_dict(self, state_dict):
        metadata = state_dict['streaming_full']
        if metadata['version'] != 1 or metadata['max_grad_norm'] != self.max_grad_norm:
            raise ValueError('Optimizer checkpoint recipe mismatch')
        if set(metadata['parameter_optimizers']) != set(self.parameter_optimizers):
            raise ValueError('Optimizer parameter names do not match the model')
        super().load_state_dict({key: value for key, value in state_dict.items() if key != 'streaming_full'})
        for name, state in metadata['parameter_optimizers'].items():
            self.parameter_optimizers[name].load_state_dict(state)
        self.num_updates = metadata['num_updates']
        self.phase = 'idle'

    def close(self):
        for handle in self.handles:
            handle.remove()
