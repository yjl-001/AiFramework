from mytorch.ops.regular import *
from .module import Module


class L1Regularizer(Module):
    def __init__(self, weight=1e-5):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return L1RegularizerFunction.apply(x, self.weight)


class L2Regularizer(Module):
    def __init__(self, weight=1e-4):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return L2RegularizerFunction.apply(x, self.weight)
