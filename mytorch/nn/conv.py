from .module import Module
from mytorch.tensor import Tensor
from mytorch.ops.conv import *

import numpy as np


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 初始化权重和偏置
        weight_shape = (out_channels, in_channels, *kernel_size)
        self.weight = Tensor(np.random.randn(
            *weight_shape) * np.sqrt(2. / (in_channels * np.prod(kernel_size))))
        self.bias = Tensor(np.zeros((out_channels,)))

    def forward(self, x):
        return Conv2dFunction.apply(x, self.weight, self.bias, self.stride, self.padding)


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        if isinstance(stride, int):
            stride = (stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return MaxPool2dFunction.apply(x, self.kernel_size, self.stride)
