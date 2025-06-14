def ensure_tensor(x):
    from .tensor import Tensor
    return x if isinstance(x, Tensor) else Tensor(x)