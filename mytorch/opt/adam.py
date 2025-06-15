from .optimizer import Optimizer
from mytorch.grad import SparseGrad
from mytorch.backend import xp


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [xp.zeros_like(p.data) for p in self.params]
        self.v = [xp.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            g = param.grad

            if isinstance(g, SparseGrad):
                # 稀疏更新
                for j, idx in enumerate(g.indices):
                    grad_row = g.values[j]
                    self.m[i][idx] = self.betas[0] * \
                        self.m[i][idx] + (1 - self.betas[0]) * grad_row
                    self.v[i][idx] = self.betas[1] * self.v[i][idx] + \
                        (1 - self.betas[1]) * (grad_row * grad_row)

                    m_hat = self.m[i][idx] / (1 - self.betas[0] ** self.t)
                    v_hat = self.v[i][idx] / (1 - self.betas[1] ** self.t)

                    param.data[idx] -= self.lr * m_hat / \
                        (xp.sqrt(v_hat) + self.eps)

            else:
                # 普通 dense 更新
                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * g
                self.v[i] = self.betas[1] * self.v[i] + \
                    (1 - self.betas[1]) * (g * g)

                m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
                v_hat = self.v[i] / (1 - self.betas[1] ** self.t)

                param.data -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)
