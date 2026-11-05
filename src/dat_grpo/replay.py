"""Replay random state when recomputing a loss without retaining full gradients."""
import random
import numpy as np
import torch


class RandomReplay:
    def __init__(self):
        self.cpu = torch.get_rng_state().clone()
        self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
        self.python = random.getstate()
        self.numpy = np.random.get_state()

    def advanced(self):
        return dict(cpu=not torch.equal(self.cpu,torch.get_rng_state()),
            cuda=any(not torch.equal(a,b) for a,b in zip(self.cuda,torch.cuda.get_rng_state_all())) if self.cuda else False,
            python=self.python!=random.getstate(),
            numpy=not np.array_equal(self.numpy[1],np.random.get_state()[1]) or self.numpy[2:]!=np.random.get_state()[2:])

    def restore(self):
        torch.set_rng_state(self.cpu)
        if self.cuda:
            torch.cuda.set_rng_state_all(self.cuda)
        random.setstate(self.python)
        np.random.set_state(self.numpy)
