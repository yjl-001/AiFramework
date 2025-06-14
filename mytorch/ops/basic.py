import numpy as np
from mytorch.ops.function import Function, Context


def unbroadcast(grad, target_shape):
    """
    将梯度 grad 按照 target_shape 的维度还原（用于广播维度的反向传播）
    """
    # 先消除多余的维度
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    # 对于原始的维度中为1的维度，沿该维度求和（因为在前向中被广播过）
    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, x):
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


class MatMul(Function):
    op = '@'

    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        return np.matmul(a.data, b.data)

    @staticmethod
    def backward(ctx: Context, grad_output):
        a, b = ctx.saved_tensors
        grad_a = np.matmul(grad_output, np.swapaxes(b.data, -1, -2))
        grad_b = np.matmul(np.swapaxes(a.data, -1, -2), grad_output)

        # 自动处理广播（与 Add/Mul 类似）
        grad_a = unbroadcast(grad_a, a.data.shape)
        grad_b = unbroadcast(grad_b, b.data.shape)

        return grad_a, grad_b


class Div(Function):
    op = '/'

    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a.data, b.data)
        return a.data / b.data

    @staticmethod
    def backward(ctx: Context, grad_output):
        a, b = ctx.saved_tensors
        return grad_output / b, -grad_output * a / (b ** 2)


class Equal(Function):
    op = '=='

    @staticmethod
    def forward(ctx: Context, a, b):
        return (a.data == b.data).astype(np.float32)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return None, None


class NotEqual(Function):
    op = '!='

    @staticmethod
    def forward(ctx: Context, a, b):
        return (a.data != b.data).astype(np.float32)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return None, None


class Less(Function):
    op = '<'

    @staticmethod
    def forward(ctx: Context, a, b):
        return (a.data < b.data).astype(np.float32)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return None, None


class LessEqual(Function):
    op = '<='

    @staticmethod
    def forward(ctx: Context, a, b):
        return (a.data <= b.data).astype(np.float32)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return None, None


class Greater(Function):
    op = '>'

    @staticmethod
    def forward(ctx: Context, a, b):
        return (a.data > b.data).astype(np.float32)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return None, None


class GreaterEqual(Function):
    op = '>='

    @staticmethod
    def forward(ctx: Context, a, b):
        return (a.data >= b.data).astype(np.float32)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return None, None


class Exp(Function):
    op = 'EXP'

    @staticmethod
    def forward(ctx: Context, a):
        out = np.exp(a.data)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        (out,) = ctx.saved_tensors
        return grad_output * out


class Log(Function):
    op = 'LOG'

    @staticmethod
    def forward(ctx: Context, a):
        ctx.save_for_backward(a.data)
        return np.log(a.data)

    @staticmethod
    def backward(ctx: Context, grad_output):
        (a,) = ctx.saved_tensors
        return grad_output / a


class Abs(Function):
    op = 'abs'

    @staticmethod
    def forward(ctx: Context, a):
        ctx.save_for_backward(a)
        return np.abs(a.data)

    @staticmethod
    def backward(ctx: Context, grad_output):
        (a,) = ctx.saved_tensors
        grad = grad_output * np.sign(a.data)
        return grad


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
        grad_b = grad_output * (a.data ** b.data) * \
            np.log(a.data + 1e-10)  # 避免 log(0)
        grad_a = unbroadcast(grad_a, a.data.shape)
        grad_b = unbroadcast(grad_b, b.data.shape)
        return grad_a, grad_b


class Max(Function):
    op = 'max'

    @staticmethod
    def forward(ctx: Context, a, axis=None, keepdims=False):
        ctx.save_for_backward(a)
        ctx.axis = axis
        ctx.keepdims = keepdims
        out = np.max(a.data, axis=axis, keepdims=keepdims)
        ctx.out = out
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        (a,) = ctx.saved_tensors
        axis = ctx.axis
        keepdims = ctx.keepdims
        out = ctx.out

        if not keepdims and axis is not None:
            grad_output = np.expand_dims(grad_output, axis)

        mask = (a.data == out)
        count = np.sum(mask, axis=axis, keepdims=True)
        grad = grad_output * mask / count
        return grad


class Min(Function):
    op = 'min'

    @staticmethod
    def forward(ctx: Context, a, axis=None, keepdims=False):
        ctx.save_for_backward(a)
        ctx.axis = axis
        ctx.keepdims = keepdims
        out = np.min(a.data, axis=axis, keepdims=keepdims)
        ctx.out = out
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        (a,) = ctx.saved_tensors
        axis = ctx.axis
        keepdims = ctx.keepdims
        out = ctx.out

        if not keepdims and axis is not None:
            grad_output = np.expand_dims(grad_output, axis)

        mask = (a.data == out)
        count = np.sum(mask, axis=axis, keepdims=True)
        grad = grad_output * mask / count
        return grad


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
        a, = ctx.saved_tensors
        axis = ctx.axis
        keepdims = ctx.keepdims

        # 如果 keepdims=False，需要还原维度再广播
        if not keepdims and axis is not None:
            grad_output = np.expand_dims(grad_output, axis=axis)

        grad_a = grad_output * np.ones_like(a.data)
        return grad_a


class Mean(Function):
    op = 'mean'

    @staticmethod
    def forward(ctx: Context, x):
        ctx.input_shape = x.data.shape
        return np.array(x.data.mean(), dtype=np.float32)

    @staticmethod
    def backward(ctx: Context, grad_output):
        grad = grad_output * \
            np.ones(ctx.input_shape, dtype=np.float32) / \
            np.prod(ctx.input_shape)
        return grad


class ReLU(Function):
    op = 'relu'

    @staticmethod
    def forward(ctx: Context, a):
        ctx.save_for_backward(a)
        return np.maximum(0, a.data)

    @staticmethod
    def backward(ctx: Context, grad_output):
        a, = ctx.saved_tensors
        relu_grad = (a.data > 0).astype(a.data.dtype)
        return grad_output * relu_grad


class Sigmoid(Function):
    op = 'sigmoid'

    @staticmethod
    def forward(ctx: Context, a):
        out = 1 / (1 + np.exp(-a.data))
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
        out = np.tanh(a.data)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        (out,) = ctx.saved_tensors
        return grad_output * (1 - out ** 2)


class Reshape(Function):
    op = 'reshape'

    @staticmethod
    def forward(ctx: Context, a, shape):
        ctx.save_for_backward(a.data.shape)
        return a.data.reshape(shape)

    @staticmethod
    def backward(ctx: Context, grad_output):
        (original_shape,) = ctx.saved_tensors
        return grad_output.reshape(original_shape), None


class Transpose(Function):
    op = 'transpose'

    @staticmethod
    def forward(ctx: Context, a, axes):
        ctx.save_for_backward(axes)
        return np.transpose(a.data, axes)

    @staticmethod
    def backward(ctx: Context, grad_output):
        (axes,) = ctx.saved_tensors
        reverse_axes = np.argsort(axes)
        return np.transpose(grad_output, reverse_axes), None


class Flatten(Function):
    op = 'flatten'

    @staticmethod
    def forward(ctx: Context, a):
        ctx.save_for_backward(a.data.shape)
        return a.data.reshape(a.data.shape[0], -1)

    @staticmethod
    def backward(ctx: Context, grad_output):
        (original_shape,) = ctx.saved_tensors
        return grad_output.reshape(original_shape)


class Clip(Function):
    op = 'clip'

    @staticmethod
    def forward(ctx: Context, a, min_val, max_val):
        ctx.save_for_backward(a.data, min_val, max_val)
        return np.clip(a.data, min_val, max_val)

    @staticmethod
    def backward(ctx: Context, grad_output):
        a, min_val, max_val = ctx.saved_tensors
        mask = (a >= min_val) & (a <= max_val)
        return grad_output * mask, None, None
