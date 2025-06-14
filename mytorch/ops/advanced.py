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


class GetItem(Function):
    @staticmethod
    def forward(ctx, x, idx):
        ctx.idx = idx
        ctx.input_shape = x.data.shape
        return x.data[idx]

    @staticmethod
    def backward(ctx, grad_output):
        grad = np.zeros(ctx.input_shape, dtype=np.float32)
        grad[ctx.idx] = grad_output
        return grad, None


class Where(Function):
    @staticmethod
    def forward(ctx: Context, condition, x, y):
        ctx.save_for_backward(condition, x, y)
        return np.where(condition.data.astype(bool), x.data, y.data)

    @staticmethod
    def backward(ctx: Context, grad_output):
        condition, x, y = ctx.saved_tensors
        cond = condition.data.astype(bool)

        grad_x = grad_output * cond
        grad_y = grad_output * (~cond)

        grad_x = unbroadcast(grad_x, x.data.shape)
        grad_y = unbroadcast(grad_y, y.data.shape)

        return None, grad_x, grad_y  # condition 不可导


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx: Context, x, dim):
        ctx.save_for_backward(x)
        ctx.dim = dim

        x_max = np.max(x.data, axis=dim, keepdims=True)
        shifted = x.data - x_max
        exp = np.exp(shifted)
        sum_exp = np.sum(exp, axis=dim, keepdims=True)
        log_softmax = shifted - np.log(sum_exp)
        ctx.output = np.exp(log_softmax)  # softmax(x)
        return log_softmax

    @staticmethod
    def backward(ctx: Context, grad_output):
        x, = ctx.saved_tensors
        softmax = ctx.output
        dim = ctx.dim

        grad_input = grad_output - \
            np.sum(grad_output * softmax, axis=dim, keepdims=True) * softmax
        return grad_input
