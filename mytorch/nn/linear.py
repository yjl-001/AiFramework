from .module import Module
from mytorch.tensor import Tensor
from mytorch.backend import xp


class Linear(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_features = in_dim
        self.out_features = out_dim
        self.weight = Tensor(xp.random.randn(in_dim, out_dim) * 0.01)
        self.bias = Tensor(xp.zeros((1, out_dim)))

    def forward(self, x):
        return x @ self.weight + self.bias
