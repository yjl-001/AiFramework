from .optimizer import Optimizer
from ..tensor import Tensor


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [Tensor.zeros_like(p) for p in self.params]
        self.v = [Tensor.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            g = param.grad
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * g
            self.v[i] = self.betas[1] * self.v[i] + \
                (1 - self.betas[1]) * (g * g)

            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)

            param.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
