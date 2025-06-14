from mytorch.backend import xp
from .function import Function, Context


def unbroadcast(grad, target_shape):
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, x):
        ctx.save_for_backward(x)
        return -x.data

    @staticmethod
    def backward(ctx: Context, grad_output):
        return -grad_output


class Add(Function):
    op = '+'

    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        return a.data + b.data

    @staticmethod
    def backward(ctx: Context, grad_output):
        a, b = ctx.saved_tensors
        grad_a = unbroadcast(grad_output, a.data.shape)
        grad_b = unbroadcast(grad_output, b.data.shape)
        return grad_a, grad_b


class Sub(Function):
    op = '-'

    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        return a.data - b.data

    @staticmethod
    def backward(ctx: Context, grad_output):
        a, b = ctx.saved_tensors
        grad_a = unbroadcast(grad_output, a.data.shape)
        grad_b = unbroadcast(-grad_output, b.data.shape)
        return grad_a, grad_b


class Mul(Function):
    op = '*'

    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        return a.data * b.data

    @staticmethod
    def backward(ctx: Context, grad_output):
        a, b = ctx.saved_tensors
        grad_a = unbroadcast(grad_output * b.data, a.data.shape)
        grad_b = unbroadcast(grad_output * a.data, b.data.shape)
        return grad_a, grad_b


class Div(Function):
    op = '/'

    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        return a.data / b.data

    @staticmethod
    def backward(ctx: Context, grad_output):
        a, b = ctx.saved_tensors
        return grad_output / b.data, -grad_output * a.data / (b.data ** 2)


class MatMul(Function):
    op = '@'

    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        return xp.matmul(a.data, b.data)

    @staticmethod
    def backward(ctx: Context, grad_output):
        a, b = ctx.saved_tensors
        grad_a = xp.matmul(grad_output, xp.swapaxes(b.data, -1, -2))
        grad_b = xp.matmul(xp.swapaxes(a.data, -1, -2), grad_output)
        grad_a = unbroadcast(grad_a, a.data.shape)
        grad_b = unbroadcast(grad_b, b.data.shape)
        return grad_a, grad_b


class Pow(Function):
    op = 'pow'

    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        return a.data ** b.data

    @staticmethod
    def backward(ctx: Context, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output * b.data * (a.data ** (b.data - 1))
        grad_b = grad_output * (a.data ** b.data) * xp.log(a.data + 1e-10)
        grad_a = unbroadcast(grad_a, a.data.shape)
        grad_b = unbroadcast(grad_b, b.data.shape)
        return grad_a, grad_b


class Sqrt(Function):
    op = 'sqrt'

    @staticmethod
    def forward(ctx: Context, a):
        out = xp.sqrt(a.data)
        ctx.save_for_backward(a)
        ctx.out = out
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        (a,) = ctx.saved_tensors
        return grad_output * 0.5 / xp.sqrt(a.data)


class Exp(Function):
    op = 'exp'

    @staticmethod
    def forward(ctx: Context, a):
        out = xp.exp(a.data)
        ctx.save_for_backward(a)
        ctx.out = out
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        return grad_output * ctx.out


class Log(Function):
    op = 'log'

    @staticmethod
    def forward(ctx: Context, a):
        ctx.save_for_backward(a)
        return xp.log(a.data)

    @staticmethod
    def backward(ctx: Context, grad_output):
        (a,) = ctx.saved_tensors
        return grad_output / a.data


class Abs(Function):
    op = 'abs'

    @staticmethod
    def forward(ctx: Context, a):
        ctx.save_for_backward(a)
        return xp.abs(a.data)

    @staticmethod
    def backward(ctx: Context, grad_output):
        (a,) = ctx.saved_tensors
        return grad_output * xp.sign(a.data)


class ReLU(Function):
    op = 'relu'

    @staticmethod
    def forward(ctx: Context, a):
        ctx.save_for_backward(a)
        return xp.maximum(0, a.data)

    @staticmethod
    def backward(ctx: Context, grad_output):
        (a,) = ctx.saved_tensors
        return grad_output * (a.data > 0).astype(a.data.dtype)


class Sigmoid(Function):
    op = 'sigmoid'

    @staticmethod
    def forward(ctx: Context, a):
        out = 1 / (1 + xp.exp(-a.data))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        (out,) = ctx.saved_tensors
        return grad_output * out * (1 - out)


class Tanh(Function):
    op = 'tanh'

    @staticmethod
    def forward(ctx: Context, a):
        out = xp.tanh(a.data)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        (out,) = ctx.saved_tensors
        return grad_output * (1 - out ** 2)


class Sum(Function):
    op = 'sum'

    @staticmethod
    def forward(ctx: Context, a, axis=None, keepdims=False):
        ctx.save_for_backward(a)
        ctx.axis = axis
        ctx.keepdims = keepdims
        return a.data.sum(axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output):
        (a,) = ctx.saved_tensors
        axis = ctx.axis
        keepdims = ctx.keepdims
        if not keepdims and axis is not None:
            grad_output = xp.expand_dims(grad_output, axis)
        return grad_output * xp.ones_like(a.data)


class Mean(Function):
    op = 'mean'

    @staticmethod
    def forward(ctx: Context, a):
        ctx.save_for_backward(a)
        ctx.input_shape = a.data.shape
        return xp.array(a.data.mean(), dtype=xp.float32)

    @staticmethod
    def backward(ctx: Context, grad_output):
        shape = ctx.input_shape
        return grad_output * xp.ones(shape, dtype=xp.float32) / xp.prod(shape)


class Max(Function):
    op = 'max'

    @staticmethod
    def forward(ctx: Context, a, axis=None, keepdims=False):
        ctx.save_for_backward(a)
        ctx.axis = axis
        ctx.keepdims = keepdims
        out = xp.max(a.data, axis=axis, keepdims=keepdims)
        ctx.out = out
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        (a,) = ctx.saved_tensors
        axis = ctx.axis
        keepdims = ctx.keepdims
        out = ctx.out
        if not keepdims and axis is not None:
            grad_output = xp.expand_dims(grad_output, axis)
        mask = (a.data == out)
        count = xp.sum(mask, axis=axis, keepdims=True)
        return grad_output * mask / count


class Min(Function):
    op = 'min'

    @staticmethod
    def forward(ctx: Context, a, axis=None, keepdims=False):
        ctx.save_for_backward(a)
        ctx.axis = axis
        ctx.keepdims = keepdims
        out = xp.min(a.data, axis=axis, keepdims=keepdims)
        ctx.out = out
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        (a,) = ctx.saved_tensors
        axis = ctx.axis
        keepdims = ctx.keepdims
        out = ctx.out
        if not keepdims and axis is not None:
            grad_output = xp.expand_dims(grad_output, axis)
        mask = (a.data == out)
        count = xp.sum(mask, axis=axis, keepdims=True)
        return grad_output * mask / count


class ArgMax(Function):
    op = 'argmax'

    @staticmethod
    def forward(ctx: Context, a, dim=None):
        ctx.save_for_backward()
        return xp.argmax(a.data, axis=dim)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return None, None


class ArgMin(Function):
    op = 'argmin'

    @staticmethod
    def forward(ctx: Context, a, dim=None):
        ctx.save_for_backward()
        return xp.argmin(a.data, axis=dim)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return None, None


class Reshape(Function):
    op = 'reshape'

    @staticmethod
    def forward(ctx: Context, a, shape):
        ctx.save_for_backward(a)
        ctx.original_shape = a.data.shape
        return a.data.reshape(shape)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return grad_output.reshape(ctx.original_shape), None


class Transpose(Function):
    op = 'transpose'

    @staticmethod
    def forward(ctx: Context, a, axes):
        ctx.save_for_backward(a)
        ctx.axes = axes
        return xp.transpose(a.data, axes)

    @staticmethod
    def backward(ctx: Context, grad_output):
        reverse_axes = xp.argsort(ctx.axes)
        return xp.transpose(grad_output, reverse_axes), None


class Flatten(Function):
    op = 'flatten'

    @staticmethod
    def forward(ctx: Context, a):
        ctx.save_for_backward(a)
        ctx.original_shape = a.data.shape
        return a.data.reshape(a.data.shape[0], -1)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return grad_output.reshape(ctx.original_shape)


class Clip(Function):
    op = 'clip'

    @staticmethod
    def forward(ctx: Context, a, min_val, max_val):
        ctx.save_for_backward(a)
        ctx.min_val = min_val
        ctx.max_val = max_val
        return xp.clip(a.data, min_val, max_val)

    @staticmethod
    def backward(ctx: Context, grad_output):
        (a,) = ctx.saved_tensors
        mask = ((a.data >= ctx.min_val) & (
            a.data <= ctx.max_val)).astype(grad_output.dtype)
        return grad_output * mask, None, None


# Comparison ops (no gradient)
class Equal(Function):
    op = '=='

    @staticmethod
    def forward(ctx: Context, a, b):
        return (a.data == b.data).astype(xp.float32)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return None, None


class NotEqual(Function):
    op = '!='

    @staticmethod
    def forward(ctx: Context, a, b):
        return (a.data != b.data).astype(xp.float32)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return None, None


class Less(Function):
    op = '<'

    @staticmethod
    def forward(ctx: Context, a, b):
        return (a.data < b.data).astype(xp.float32)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return None, None


class LessEqual(Function):
    op = '<='

    @staticmethod
    def forward(ctx: Context, a, b):
        return (a.data <= b.data).astype(xp.float32)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return None, None


class Greater(Function):
    op = '>'

    @staticmethod
    def forward(ctx: Context, a, b):
        return (a.data > b.data).astype(xp.float32)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return None, None


class GreaterEqual(Function):
    op = '>='

    @staticmethod
    def forward(ctx: Context, a, b):
        return (a.data >= b.data).astype(xp.float32)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return None, None
