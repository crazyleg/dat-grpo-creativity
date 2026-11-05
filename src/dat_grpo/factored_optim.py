"""Chunked Adafactor with FP32 factored statistics and optional BF16 stochastic rounding.

Fixed external LR, beta1=None, scale_parameter=False, relative_step=False,
weight_decay=0. Matches Transformers Adafactor without materializing full FP32
parameter, gradient, second moment, or update tensors for large matrices.
"""
import math
import torch


def stochastic_bf16(value):
    bits = value.contiguous().view(torch.int32)
    noise = torch.randint(0, 65536, value.shape, device=value.device, dtype=torch.int32)
    rounded = ((bits + noise) & -65536).view(torch.float32)
    return rounded.to(torch.bfloat16)


class ChunkedAdafactor(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-6, chunk_elements=1024*1024,
                 stochastic_round=True, clip_threshold=1.0, decay_rate=-0.8, eps=1e-30):
        if lr <= 0 or chunk_elements < 1 or clip_threshold <= 0:
            raise ValueError('Invalid factored optimizer settings')
        super().__init__(params, dict(lr=lr, chunk_elements=chunk_elements,
            stochastic_round=stochastic_round, clip_threshold=clip_threshold,
            decay_rate=decay_rate, eps=eps))

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Optimizer.load_state_dict normally casts floating state to the BF16
        # parameter dtype; the factored running statistics must stay FP32.
        for saved_group, group in zip(state_dict['param_groups'], self.param_groups):
            for saved_id, parameter in zip(saved_group['params'], group['params']):
                saved = state_dict['state'].get(saved_id, {})
                for name in ('row', 'col', 'variance'):
                    if name in saved:
                        self.state[parameter][name] = saved[name].to(
                            device=parameter.device, dtype=torch.float32, copy=True)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Closures are unsupported')
        for group in self.param_groups:
            for parameter in group['params']:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse or not gradient.is_contiguous() or not parameter.is_contiguous():
                    raise ValueError('Dense contiguous parameters and gradients are required')
                state = self.state[parameter]
                if not state:
                    state['step'] = 0
                    if parameter.ndim >= 2:
                        state['row'] = torch.zeros(parameter.shape[:-1], device=parameter.device, dtype=torch.float32)
                        state['col'] = torch.zeros(parameter.shape[:-2]+parameter.shape[-1:], device=parameter.device, dtype=torch.float32)
                    else:
                        state['variance'] = torch.zeros_like(parameter, dtype=torch.float32)
                state['step'] += 1
                beta = 1 - state['step']**group['decay_rate']
                if parameter.ndim >= 2:
                    rows, cols = parameter.shape[-2:]
                    batch = math.prod(parameter.shape[:-2])
                    grad = gradient.view(batch, rows, cols)
                    weight = parameter.view(batch, rows, cols)
                    row_state, col_state = state['row'].view(batch, rows), state['col'].view(batch, cols)
                    rows_per_chunk = max(1, group['chunk_elements']//cols)
                    for b in range(batch):
                        col_sum = torch.zeros(cols, device=parameter.device, dtype=torch.float32)
                        for start in range(0, rows, rows_per_chunk):
                            stop = min(rows, start+rows_per_chunk)
                            square = grad[b,start:stop].float().square().add_(group['eps'])
                            row_state[b,start:stop].mul_(beta).add_(square.mean(-1), alpha=1-beta)
                            col_sum.add_(square.sum(0))
                        col_state[b].mul_(beta).add_(col_sum/rows, alpha=1-beta)
                    row_scale = (row_state/row_state.mean(-1, keepdim=True)).rsqrt()
                    col_scale = col_state.rsqrt()
                    def chunks():
                        for b in range(batch):
                            for start in range(0, rows, rows_per_chunk):
                                stop = min(rows, start+rows_per_chunk)
                                update = grad[b,start:stop].float()*row_scale[b,start:stop,None]*col_scale[b,None,:]
                                yield weight[b,start:stop], update
                else:
                    variance = state['variance']
                    variance.mul_(beta).add_(gradient.float().square().add_(group['eps']), alpha=1-beta)
                    def chunks():
                        yield parameter, variance.rsqrt()*gradient.float()
                norm_squared = torch.zeros((), device=parameter.device, dtype=torch.float64)
                for _, update in chunks():
                    norm_squared.add_(update.square().sum(dtype=torch.float64))
                rms = (norm_squared/parameter.numel()).sqrt()
                divisor = (rms/group['clip_threshold']).clamp_min(1).float()
                if not torch.isfinite(divisor):
                    raise FloatingPointError('Non-finite Adafactor update')
                for target, update in chunks():
                    value = target.float()-update*(group['lr']/divisor)
                    target.copy_(stochastic_bf16(value) if group['stochastic_round'] and parameter.dtype == torch.bfloat16 else value)
