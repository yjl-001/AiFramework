from mytorch.backend import xp
from .function import Function, Context


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
        ctx.out = out
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        return grad_output * ctx.out * (1 - ctx.out)


class Tanh(Function):
    op = 'tanh'

    @staticmethod
    def forward(ctx: Context, a):
        out = xp.tanh(a.data)
        ctx.out = out
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        return grad_output * (1 - ctx.out ** 2)


class LeakyReLU(Function):
    op = 'leaky_relu'

    @staticmethod
    def forward(ctx: Context, a, negative_slope=0.01):
        ctx.save_for_backward(a)
        ctx.negative_slope = negative_slope
        return xp.where(a.data > 0, a.data, negative_slope * a.data)

    @staticmethod
    def backward(ctx: Context, grad_output):
        (a,) = ctx.saved_tensors
        grad = xp.ones_like(a.data)
        grad[a.data < 0] = ctx.negative_slope
        return grad_output * grad


class ELU(Function):
    op = 'elu'

    @staticmethod
    def forward(ctx: Context, a, alpha=1.0):
        ctx.save_for_backward(a)
        ctx.alpha = alpha
        out = xp.where(a.data > 0, a.data, alpha * (xp.exp(a.data) - 1))
        ctx.out = out
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        (a,) = ctx.saved_tensors
        grad = xp.where(a.data > 0, 1.0, ctx.out + ctx.alpha)
        return grad_output * grad
