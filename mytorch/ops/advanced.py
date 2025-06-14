import numpy as np
from mytorch.ops.function import Function, Context


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
