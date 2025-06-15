from .function import Function
from mytorch.backend import xp


class L1RegularizerFunction(Function):
    @staticmethod
    def forward(ctx, x, weight=1.0):
        ctx.save_for_backward(x)
        ctx.weight = weight
        return weight * xp.sum(xp.abs(x.data))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad = ctx.weight * xp.sign(x.data) * grad_output
        return grad, None  # x 有梯度，weight 无梯度


class L2RegularizerFunction(Function):
    @staticmethod
    def forward(ctx, x, weight=1.0):
        ctx.save_for_backward(x)
        ctx.weight = weight
        return 0.5 * weight * xp.sum(x.data ** 2)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad = ctx.weight * x.data * grad_output
        return grad, None
