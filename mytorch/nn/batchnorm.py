from mytorch.ops.batchnorm import BatchNormFunction
from .module import Module
from mytorch.tensor import Tensor
from mytorch.backend import xp


class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = Tensor(xp.ones((num_features,)))
            self.bias = Tensor(xp.zeros((num_features,)))
        else:
            self.weight = None
            self.bias = None

        if track_running_stats:
            self.running_mean = xp.zeros((num_features,))
            self.running_var = xp.ones((num_features,))
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x):
        return BatchNormFunction.apply(
            x, self.weight, self.bias,
            self.running_mean, self.running_var,
            self.training, self.momentum, self.eps
        )


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = Tensor(xp.ones((num_features,)))
            self.bias = Tensor(xp.zeros((num_features,)))
        else:
            self.weight = None
            self.bias = None

        if track_running_stats:
            self.running_mean = xp.zeros((num_features,))
            self.running_var = xp.ones((num_features,))
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x):
        return BatchNormFunction.apply(
            x, self.weight, self.bias,
            self.running_mean, self.running_var,
            self.training, self.momentum, self.eps
        )


class BatchNorm3d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = Tensor(xp.ones((num_features,)))
            self.bias = Tensor(xp.zeros((num_features,)))
        else:
            self.weight = None
            self.bias = None

        if track_running_stats:
            self.running_mean = xp.zeros((num_features,))
            self.running_var = xp.ones((num_features,))
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x):
        return BatchNormFunction.apply(
            x, self.weight, self.bias,
            self.running_mean, self.running_var,
            self.training, self.momentum, self.eps
        )
