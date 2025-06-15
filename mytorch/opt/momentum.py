from .optimizer import Optimizer
from mytorch.grad import SparseGrad
from mytorch.backend import xp


class Momentum(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.v = [xp.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            g = param.grad

            if isinstance(g, SparseGrad):
                for j, idx in enumerate(g.indices):
                    grad_row = g.values[j]
                    self.v[i][idx] = self.momentum * self.v[i][idx] + grad_row
                    param.data[idx] -= self.lr * self.v[i][idx]
            else:
                self.v[i] = self.momentum * self.v[i] + g
                param.data -= self.lr * self.v[i]
