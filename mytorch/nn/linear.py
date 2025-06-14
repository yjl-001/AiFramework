from .module import Module
from mytorch.tensor import Tensor
import numpy as np


class Linear(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = Tensor(np.random.randn(in_dim, out_dim) * 0.01)
        self.bias = Tensor(np.zeros((1, out_dim)))

    def forward(self, x):
        return x @ self.weight + self.bias
