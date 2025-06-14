from .optimizer import Optimizer
from mytorch.grad import SparseGrad
from mytorch.backend import xp


class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            g = param.grad

            if isinstance(g, SparseGrad):
                # 支持稀疏梯度更新
                for j, idx in enumerate(g.indices):
                    grad_row = g.values[j]
                    param.data[idx] -= self.lr * grad_row
            else:
                # 普通 dense 梯度更新
                param.data -= self.lr * g
