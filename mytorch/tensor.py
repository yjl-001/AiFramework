import numpy as np
from .ops.function import Function, Context
from .ops.basic import Add, Mul, MatMul, Sum, ReLU
from .utils import ensure_tensor


class Tensor:
    def __init__(self, data, frozen=False):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data) if not frozen else None
        self.frozen = frozen
        self._ctx: Context | None = Context()
        self._backward_fn: type[Function] | None = None
        self.frozen = frozen

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)

        self.grad = grad

        visited = set()
        ordered: list[Tensor] = []

        def build_topo(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for inp in tensor._ctx.inputs:
                    if isinstance(inp, Tensor):
                        build_topo(inp)
                ordered.append(tensor)

        build_topo(self)

        for t in reversed(ordered):
            if t._backward_fn is not None:
                grads = t._backward_fn.backward(t._ctx, t.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                for inp, g in zip(t._ctx.inputs, grads):
                    if isinstance(inp, Tensor):
                        inp.grad += g

    def __add__(self, other):
        return Add.apply(self, ensure_tensor(other))

    def __mul__(self, other):
        return Mul.apply(self, ensure_tensor(other))

    def __matmul__(self, other):
        return MatMul.apply(self, ensure_tensor(other))

    def sum(self, axis=None, keepdims=False):
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def relu(self):
        return ReLU.apply(self)
