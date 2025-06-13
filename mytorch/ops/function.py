class Context:
    """
    用于在 forward 中保存中间变量，以便 backward 使用。
    """

    def __init__(self):
        self.inputs = ()
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class Function:
    @classmethod
    def apply(cls, *inputs):
        from ..tensor import Tensor

        ctx = Context()

        raw_inputs = [x.data if isinstance(x, Tensor) else x for x in inputs]

        result = cls.forward(ctx, *raw_inputs)

        # 包装成 Tensor（构建计算图）
        out = Tensor(result)
        out._ctx = ctx
        out._ctx.inputs = inputs  # 保存原始输入 Tensor
        out._backward_fn = cls    # 指定反向传播要用的类
        return out

    @staticmethod
    def forward(ctx: Context, *args):
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output):
        raise NotImplementedError
