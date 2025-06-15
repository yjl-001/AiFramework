from .optimizer import Optimizer
from mytorch.grad import SparseGrad
from mytorch.backend import xp


class RMSProp(Optimizer):
    def __init__(self, params, lr=0.001, alpha=0.99, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.alpha = alpha  # 衰减系数
        self.eps = eps
        self.v = [xp.zeros_like(p.data) for p in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            g = param.grad

            if isinstance(g, SparseGrad):
                for j, idx in enumerate(g.indices):
                    grad_row = g.values[j]
                    self.v[i][idx] = self.alpha * self.v[i][idx] + \
                        (1 - self.alpha) * (grad_row * grad_row)
                    param.data[idx] -= self.lr * grad_row / \
                        (xp.sqrt(self.v[i][idx]) + self.eps)
            else:
                self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (g * g)
                param.data -= self.lr * g / (xp.sqrt(self.v[i]) + self.eps)
