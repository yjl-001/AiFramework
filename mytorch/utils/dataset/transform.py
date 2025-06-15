from mytorch.tensor import Tensor
from mytorch.backend import xp


def to_tensor(x, dtype=xp.float32, frozen=False):
    return Tensor(x, dtype=dtype,  frozen=frozen)


def normalize(x, mean=0.5, std=0.5):
    return (x - mean) / std


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
