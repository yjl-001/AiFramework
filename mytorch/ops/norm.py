from .function import Function, Context
from mytorch.backend import xp


class BatchNormFunction(Function):
    @staticmethod
    def forward(ctx: Context, x, weight, bias, running_mean, running_var, training, momentum, eps):
        N, C = x.shape[:2]
        axis = (0,) + tuple(range(2, x.ndim))

        if training:
            mean = xp.mean(x.data, axis=axis, keepdims=True)
            var = xp.var(x.data, axis=axis, keepdims=True)

            if running_mean is not None:
                running_mean[...] = (1 - momentum) * \
                    running_mean + momentum * mean.squeeze()
            if running_var is not None:
                running_var[...] = (1 - momentum) * \
                    running_var + momentum * var.squeeze()
        else:
            mean = running_mean.reshape((1, C) + (1,) * (x.ndim - 2))
            var = running_var.reshape((1, C) + (1,) * (x.ndim - 2))

        std = xp.sqrt(var + eps)
        x_hat = (x.data - mean) / std

        out = x_hat
        if weight is not None:
            out = out * weight.data.reshape((1, C) + (1,) * (x.ndim - 2))
        if bias is not None:
            out = out + bias.data.reshape((1, C) + (1,) * (x.ndim - 2))

        # 保存 Tensor 类型
        ctx.save_for_backward(x, weight, bias)
        # 保存非 Tensor 类型
        ctx.mean = mean
        ctx.var = var
        ctx.eps = eps
        ctx.axis = axis
        ctx.x_hat = x_hat

        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        x, weight, bias = ctx.saved_tensors
        mean = ctx.mean
        var = ctx.var
        eps = ctx.eps
        axis = ctx.axis
        x_hat = ctx.x_hat

        std = xp.sqrt(var + eps)
        N = x.data.size / x.data.shape[1]  # 每个通道的样本数

        if weight is not None:
            grad_x_hat = grad_output * \
                weight.data.reshape((1, x.data.shape[1]) + (1,) * (x.ndim - 2))
        else:
            grad_x_hat = grad_output

        grad_var = xp.sum(grad_x_hat * (x.data - mean) * -
                          0.5 * std**-3, axis=axis, keepdims=True)
        grad_mean = xp.sum(grad_x_hat * -1 / std, axis=axis, keepdims=True) + \
            grad_var * xp.mean(-2 * (x.data - mean), axis=axis, keepdims=True)

        grad_input = grad_x_hat / std + grad_var * \
            2 * (x.data - mean) / N + grad_mean / N

        if weight is not None:
            grad_weight = xp.sum(grad_output * x_hat,
                                 axis=axis, keepdims=False)
        else:
            grad_weight = None

        if bias is not None:
            grad_bias = xp.sum(grad_output, axis=axis, keepdims=False)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class LayerNormFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        normalized_shape = weight.data.shape if weight is not None else x.shape[1:]
        axis = tuple(range(-len(normalized_shape), 0))

        mean = xp.mean(x.data, axis=axis, keepdims=True)
        var = xp.var(x.data, axis=axis, keepdims=True)
        std = xp.sqrt(var + eps)

        x_hat = (x.data - mean) / std

        out = x_hat
        if weight is not None:
            out = out * \
                weight.data.reshape(
                    (1,) * (x.ndim - len(normalized_shape)) + normalized_shape)
        if bias is not None:
            out = out + \
                bias.data.reshape(
                    (1,) * (x.ndim - len(normalized_shape)) + normalized_shape)

        ctx.save_for_backward(x, weight, bias)
        ctx.mean = mean
        ctx.var = var
        ctx.eps = eps
        ctx.axis = axis
        ctx.x_hat = x_hat
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        mean = ctx.mean
        var = ctx.var
        eps = ctx.eps
        axis = ctx.axis
        x_hat = ctx.x_hat

        std = xp.sqrt(var + eps)
        N = xp.prod(xp.array([x.shape[ax] for ax in axis]))

        if weight is not None:
            grad_x_hat = grad_output * \
                weight.data.reshape(
                    (1,) * (x.ndim - len(axis)) + weight.data.shape)
        else:
            grad_x_hat = grad_output

        grad_var = xp.sum(grad_x_hat * (x.data - mean) * -
                          0.5 * std**-3, axis=axis, keepdims=True)
        grad_mean = xp.sum(grad_x_hat * -1 / std, axis=axis, keepdims=True) + \
            grad_var * xp.mean(-2 * (x.data - mean), axis=axis, keepdims=True)

        grad_input = grad_x_hat / std + grad_var * \
            2 * (x.data - mean) / N + grad_mean / N

        if weight is not None:
            grad_weight = xp.sum(
                grad_output * x_hat, axis=tuple(i for i in range(x.ndim - len(weight.data.shape))))
        else:
            grad_weight = None

        if bias is not None:
            grad_bias = xp.sum(grad_output, axis=tuple(
                i for i in range(x.ndim - len(bias.data.shape))))
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None


class GroupNormFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, num_groups, eps):
        N, C = x.shape[:2]
        G = num_groups
        assert C % G == 0
        group_shape = (N, G, C // G) + x.shape[2:]

        x_group = x.data.reshape(group_shape)
        axis = (2,) + tuple(range(3, x.ndim + 1))
        mean = xp.mean(x_group, axis=axis, keepdims=True)
        var = xp.var(x_group, axis=axis, keepdims=True)
        std = xp.sqrt(var + eps)

        x_hat = (x_group - mean) / std
        x_hat = x_hat.reshape(x.shape)

        out = x_hat
        if weight is not None:
            out = out * weight.data.reshape((1, C) + (1,) * (x.ndim - 2))
        if bias is not None:
            out = out + bias.data.reshape((1, C) + (1,) * (x.ndim - 2))

        ctx.save_for_backward(x, weight, bias)
        ctx.mean = mean
        ctx.var = var
        ctx.eps = eps
        ctx.num_groups = G
        ctx.x_hat = x_hat
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        mean = ctx.mean
        var = ctx.var
        eps = ctx.eps
        G = ctx.num_groups
        x_hat = ctx.x_hat

        N, C = x.shape[:2]
        group_shape = (N, G, C // G) + x.shape[2:]
        x_group = x.data.reshape(group_shape)
        x_hat_group = x_hat.reshape(group_shape)
        grad_output_group = grad_output.reshape(group_shape)

        axis = (2,) + tuple(range(3, x.ndim + 1))
        M = xp.prod(xp.array([x_group.shape[ax] for ax in axis]))

        if weight is not None:
            grad_x_hat = grad_output * \
                weight.data.reshape((1, C) + (1,) * (x.ndim - 2))
        else:
            grad_x_hat = grad_output

        grad_x_hat = grad_x_hat.reshape(group_shape)

        grad_var = xp.sum(grad_x_hat * (x_group - mean) * -
                          0.5 * (var + eps)**-1.5, axis=axis, keepdims=True)
        grad_mean = xp.sum(grad_x_hat * -1 / xp.sqrt(var + eps), axis=axis, keepdims=True) + \
            grad_var * xp.mean(-2 * (x_group - mean), axis=axis, keepdims=True)

        grad_input = grad_x_hat / xp.sqrt(var + eps) + \
            grad_var * 2 * (x_group - mean) / M + \
            grad_mean / M

        grad_input = grad_input.reshape(x.shape)

        if weight is not None:
            grad_weight = xp.sum(grad_output * x_hat,
                                 axis=tuple(range(0, x.ndim)), keepdims=False)
        else:
            grad_weight = None

        if bias is not None:
            grad_bias = xp.sum(grad_output, axis=tuple(
                range(0, x.ndim)), keepdims=False)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None
