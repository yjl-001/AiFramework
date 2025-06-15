from .grad import is_grad_enabled
from .backend import xp
from .ops.function import Function, Context
from .ops.basic import *
from .ops.advanced import *


def ensure_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x)


class Tensor:
    def __init__(self, data, dtype=xp.float32, frozen=False):
        frozen = frozen or not is_grad_enabled()
        self.data = xp.array(data, dtype=dtype)
        self.grad = xp.zeros_like(self.data) if not frozen else None
        self.frozen = frozen
        self._ctx: Context | None = Context()
        self._backward_fn: type[Function] | None = None

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __str__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def item(self):
        if self.data.size != 1:
            raise ValueError("Tensor must be scalar to call item()")
        return self.data.item()

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self):
        return "cpu"

    def astype(self, dtype):
        return Tensor(self.data.astype(dtype), dtype=dtype, frozen=self.frozen)

    def float(self):
        return self.astype(xp.float32)

    def int(self):
        return self.astype(xp.int32)

    def long(self):
        return self.astype(xp.int64)

    def freeze(self):
        if self.grad is not None:
            self.grad.fill(0)
        self.frozen = True

    def unfreeze(self):
        if self.frozen:
            self.grad = xp.zeros_like(self.data)
            self.frozen = False

    def squeeze(self):
        return Tensor(self.data.squeeze(), dtype=self.dtype, frozen=self.frozen)

    def unsqueeze(self, axis):
        if axis < 0:
            axis += len(self.shape) + 1
        return Tensor(xp.expand_dims(self.data, axis), dtype=self.dtype, frozen=self.frozen)

    def to(self, device):
        if device != "cpu":
            raise NotImplementedError("Only 'cpu' device is supported.")
        return self

    def detach(self):
        detached_tensor = Tensor(
            self.data.copy(), dtype=self.dtype, frozen=True)
        detached_tensor.grad = None
        return detached_tensor

    def clone(self):
        cloned_tensor = Tensor(
            self.data.copy(), dtype=self.dtype, frozen=self.frozen)
        if self.grad is not None:
            cloned_tensor.grad = self.grad.copy()
        return cloned_tensor

    def zeros(self, shape=None):
        if shape is None:
            shape = self.data.shape
        return Tensor(xp.zeros(shape, dtype=self.dtype), frozen=self.frozen)

    def numpy(self):
        return self.data.copy()

    def zero_grad(self):
        if self.grad is not None:
            self.grad.fill(0)

    @classmethod
    def zeros_like(cls, other):
        return cls(xp.zeros_like(other.data), frozen=other.frozen)

    def backward(self, grad=None):
        if grad is None:
            grad = xp.ones_like(self.data)

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

        def add_grad(old_grad, new_grad):
            if old_grad is None:
                return new_grad
            if isinstance(old_grad, SparseGrad) and isinstance(new_grad, SparseGrad):
                # 合并两个 SparseGrad
                return old_grad + new_grad
            elif isinstance(old_grad, SparseGrad):
                return old_grad + SparseGrad.from_dense(new_grad)
            elif isinstance(new_grad, SparseGrad):
                return SparseGrad.from_dense(old_grad) + new_grad
            else:
                return old_grad + new_grad

        for t in reversed(ordered):
            if t._backward_fn is not None:
                grads = t._backward_fn.backward(t._ctx, t.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                for inp, g in zip(t._ctx.inputs, grads):
                    if isinstance(inp, Tensor):
                        inp.grad = add_grad(inp.grad, g)

    def __getitem__(self, idx):
        return GetItem.apply(self, idx)

    def __neg__(self):
        return Neg.apply(self)

    def __add__(self, other):
        return Add.apply(self, ensure_tensor(other))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return Sub.apply(self, ensure_tensor(other))

    def __rsub__(self, other):
        return ensure_tensor(other) - self

    def __mul__(self, other):
        return Mul.apply(self, ensure_tensor(other))

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        return MatMul.apply(self, ensure_tensor(other))

    def __truediv__(self, other):
        return Div.apply(self, ensure_tensor(other))

    def __rtruediv__(self, other):
        return ensure_tensor(other) / self

    def __pow__(self, power):
        return Pow.apply(self, ensure_tensor(power))

    def __rpow__(self, base):
        return Pow.apply(ensure_tensor(base), self)

    def __eq__(self, other):
        return Equal.apply(self, ensure_tensor(other))

    def __ne__(self, other):
        return NotEqual.apply(self, ensure_tensor(other))

    def __lt__(self, other):
        return Less.apply(self, ensure_tensor(other))

    def __le__(self, other):
        return LessEqual.apply(self, ensure_tensor(other))

    def __gt__(self, other):
        return Greater.apply(self, ensure_tensor(other))

    def __ge__(self, other):
        return GreaterEqual.apply(self, ensure_tensor(other))

    def exp(self):
        return Exp.apply(self)

    def sqrt(self):
        return Sqrt.apply(self)

    def log(self):
        return Log.apply(self)

    def sum(self, axis=None, keepdims=False):
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def abs(self):
        return Abs.apply(self)

    def max(self, axis=None, keepdims=False):
        return Max.apply(self, axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims=False):
        return Min.apply(self, axis=axis, keepdims=keepdims)

    def argmax(self, axis=None, dim=None):
        if dim is not None:
            axis = dim
        return ArgMax.apply(self, axis)

    def argmin(self, axis=None, dim=None):
        if dim is not None:
            axis = dim
        return ArgMin.apply(self, axis)

    def mean(self):
        return Mean.apply(self)

    def relu(self):
        return ReLU.apply(self)

    def sigmoid(self):
        return Sigmoid.apply(self)

    def log_softmax(self, dim):
        return LogSoftmax.apply(self, dim)

    def tanh(self):
        return Tanh.apply(self)

    def reshape(self, shape):
        return Reshape.apply(self, shape)

    def transpose(self, axes):
        return Transpose.apply(self, axes)

    def flatten(self):
        return Flatten.apply(self)

    def clip(self, min_val, max_val):
        return Clip.apply(self, min_val, max_val)

    def chunk(self, chunks, dim=0):
        return tuple(Chunk.apply(self, index=i, chunks=chunks, dim=dim) for i in range(chunks))

    @staticmethod
    def stack(tensors, dim=0):
        return Stack.apply(tensors, dim)

    @staticmethod
    def where(condition, x, y):
        return Where.apply(ensure_tensor(condition), ensure_tensor(x), ensure_tensor(y))
