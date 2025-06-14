class Context:
    """
    用于在 forward 中保存中间变量，以便 backward 使用。
    """

    def __init__(self):
        self.inputs = ()
        self.saved_tensors = ()

    def save_for_backward(self, *tensors):
        from ..tensor import Tensor
        assert all(
            isinstance(t, Tensor) for t in tensors
        ), "All saved tensors must be instances of Tensor."
        self.saved_tensors = tensors


class Function:
    op = ''

    @classmethod
    def apply(cls, *inputs, **kwargs):
        from ..tensor import Tensor

        ctx = Context()
        ctx.inputs = inputs

        result = cls.forward(ctx, *inputs, **kwargs)

        if isinstance(result, tuple):
            outputs = []
            for res in result:
                t = Tensor(res.data if isinstance(res, Tensor) else res)
                t._ctx = ctx
                t._backward_fn = cls
                outputs.append(t)
            return tuple(outputs)
        else:
            t = Tensor(result.data if isinstance(result, Tensor) else result)
            t._ctx = ctx
            t._backward_fn = cls
            return t

    @staticmethod
    def forward(ctx: Context, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output):
        raise NotImplementedError
