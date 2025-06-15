from .function import Function
from mytorch.backend import xp


class DropoutFunction(Function):
    @staticmethod
    def forward(ctx, x, p=0.5, training=True):
        if training:
            # 生成 dropout 掩码（保持概率为 1-p）
            mask = xp.random.rand(*x.shape) > p
            scale = 1.0 / (1.0 - p)
            out = x.data * mask * scale

            # 保存 mask 和 scale 用于反向传播
            ctx.save_for_backward(mask)
            ctx.scale = scale
        else:
            out = x.data
        return out

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        scale = ctx.scale
        grad_input = grad_output * mask * scale
        return grad_input, None, None  # 后两个是 p 和 training，没有梯度
