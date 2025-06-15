from mytorch.backend import xp
from contextlib import contextmanager

_grad_enabled = True


def is_grad_enabled():
    """
    Check if gradient tracking is enabled.
    """
    return _grad_enabled


@contextmanager
def no_grad():
    """
    Context manager to disable gradient tracking.
    """
    global _grad_enabled
    prev_state = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev_state


class SparseGrad:
    def __init__(self, indices, values, shape):
        self.indices = xp.array(indices)
        self.values = xp.array(values)
        self.shape = shape

    def __repr__(self):
        return f"SparseGrad(indices={self.indices}, values={self.values}, shape={self.shape})"

    def to_dense(self):
        dense = xp.zeros(self.shape, dtype=self.values.dtype)
        dense[self.indices] = self.values
        return dense
