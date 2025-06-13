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


class Add(Function):
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


class Sum(Function):
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


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a):
        ctx.save_for_backward(a)
        return np.maximum(0, a.data)

    @staticmethod
    def backward(ctx: Context, grad_output):
        a, = ctx.saved_tensors
        relu_grad = (a.data > 0).astype(a.data.dtype)
        return grad_output * relu_grad
