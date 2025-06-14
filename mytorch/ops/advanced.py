from mytorch.backend import xp
from .function import Function, Context


def unbroadcast(grad, target_shape):
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class GetItem(Function):
    @staticmethod
    def forward(ctx: Context, x, idx):
        from ..tensor import Tensor
        if isinstance(idx, tuple):
            idx = tuple(
                i.data if isinstance(i, Tensor) else
                xp.array(i) if isinstance(i, range) else
                i
                for i in idx
            )
        elif isinstance(idx, Tensor):
            idx = idx.data
        elif isinstance(idx, range):
            idx = xp.array(idx)

        ctx.save_for_backward(x)
        ctx.idx = idx
        return x.data[idx]

    @staticmethod
    def backward(ctx: Context, grad_output):
        (x,) = ctx.saved_tensors
        grad = xp.zeros_like(x.data, dtype=xp.float32)
        grad[ctx.idx] = grad_output
        return grad, None


class Where(Function):
    @staticmethod
    def forward(ctx: Context, condition, x, y):
        ctx.save_for_backward(condition, x, y)
        return xp.where(condition.data.astype(bool), x.data, y.data)

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
        ctx.dim = dim
        x_max = xp.max(x.data, axis=dim, keepdims=True)
        shifted = x.data - x_max
        exp = xp.exp(shifted)
        sum_exp = xp.sum(exp, axis=dim, keepdims=True)
        softmax = exp / sum_exp
        log_softmax = xp.log(softmax + 1e-9)  # 加 epsilon 防止 log(0)

        ctx.softmax = softmax
        return log_softmax

    @staticmethod
    def backward(ctx: Context, grad_output):
        softmax = ctx.softmax
        dim = ctx.dim
        grad_input = grad_output - \
            xp.sum(grad_output * softmax, axis=dim, keepdims=True) * softmax
        return grad_input, None
