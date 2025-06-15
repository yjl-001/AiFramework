from .module import Module
from mytorch.ops.pooling import *


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(
            kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.padding = padding

    def forward(self, x):
        return MaxPool2dFunction.apply(x, self.kernel_size, self.stride, self.padding)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(
            kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.padding = padding

    def forward(self, x):
        return AvgPool2dFunction.apply(x, self.kernel_size, self.stride, self.padding)


class GlobalAvgPool2d(Module):
    def forward(self, x):
        # x shape: (N, C, H, W)
        return x.mean(axis=(2, 3), keepdims=True)
