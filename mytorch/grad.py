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
        grad = xp.zeros(self.shape, dtype=self.values.dtype)
        xp.add.at(grad, self.indices, self.values)
        return grad

    def __add__(self, other):
        if isinstance(other, SparseGrad):
            assert self.shape == other.shape
            indices = xp.concatenate([self.indices, other.indices], axis=0)
            values = xp.concatenate([self.values, other.values], axis=0)
            return SparseGrad(indices, values, self.shape)
        elif isinstance(other, xp.ndarray):
            return self.to_dense() + other
        else:
            raise TypeError(
                f"Unsupported add between SparseGrad and {type(other)}")

    def __radd__(self, other):
        return self.__add__(other)

    @staticmethod
    def from_dense(dense_grad):
        indices = xp.nonzero(xp.any(dense_grad != 0, axis=1))[0]
        values = dense_grad[indices]
        return SparseGrad(indices, values, dense_grad.shape)
