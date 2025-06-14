from .function import Function, Context
from mytorch.backend import xp


from mytorch.backend import xp


def im2col(x, KH, KW, SH, SW, PH, PW):
    N, C, H, W = x.shape
    H_out = (H + 2 * PH - KH) // SH + 1
    W_out = (W + 2 * PW - KW) // SW + 1

    x_padded = xp.pad(x, ((0, 0), (0, 0), (PH, PH), (PW, PW)), mode='constant')
    cols = xp.zeros((N, C, KH, KW, H_out, W_out))

    for y in range(KH):
        y_max = y + SH * H_out
        for x_ in range(KW):
            x_max = x_ + SW * W_out
            cols[:, :, y, x_, :, :] = x_padded[:, :, y:y_max:SH, x_:x_max:SW]

    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1)
    return cols, x_padded, H_out, W_out


def col2im(cols, x_shape, KH, KW, SH, SW, PH, PW, H_out, W_out):
    N, C, H, W = x_shape
    x_padded = xp.zeros((N, C, H + 2 * PH, W + 2 * PW))
    cols_reshaped = cols.reshape(
        N, H_out, W_out, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    for y in range(KH):
        y_max = y + SH * H_out
        for x_ in range(KW):
            x_max = x_ + SW * W_out
            x_padded[:, :, y:y_max:SH,
                     x_:x_max:SW] += cols_reshaped[:, :, y, x_, :, :]

    if PH == 0 and PW == 0:
        return x_padded
    return x_padded[:, :, PH:-PH, PW:-PW]


"""
下面是非常慢的循环版本
"""
# class Conv2dFunction(Function):
#     @staticmethod
#     def forward(ctx: Context, x, weight, bias, stride, padding):
#         ctx.save_for_backward(x, weight, bias)
#         ctx.stride = stride
#         ctx.padding = padding

#         N, C_in, H_in, W_in = x.data.shape
#         C_out, _, KH, KW = weight.data.shape
#         SH, SW = stride
#         PH, PW = padding

#         # Padding
#         x_padded = xp.pad(x.data, ((0, 0), (0, 0), (PH, PH),
#                           (PW, PW)), mode='constant')

#         H_out = (H_in + 2 * PH - KH) // SH + 1
#         W_out = (W_in + 2 * PW - KW) // SW + 1

#         out = xp.zeros((N, C_out, H_out, W_out))

#         for n in range(N):
#             for c_out in range(C_out):
#                 for h in range(H_out):
#                     for w in range(W_out):
#                         h_start = h * SH
#                         w_start = w * SW
#                         h_end = h_start + KH
#                         w_end = w_start + KW

#                         region = x_padded[n, :, h_start:h_end, w_start:w_end]
#                         out[n, c_out, h, w] = xp.sum(
#                             region * weight.data[c_out]) + bias.data[c_out]

#         ctx.x_padded = x_padded
#         return out

#     @staticmethod
#     def backward(ctx: Context, grad_output):
#         x, weight, bias = ctx.saved_tensors
#         stride = ctx.stride
#         padding = ctx.padding

#         N, C_in, H_in, W_in = x.data.shape
#         C_out, _, KH, KW = weight.data.shape
#         SH, SW = stride
#         PH, PW = padding
#         H_out, W_out = grad_output.shape[2:]

#         dx_padded = xp.zeros_like(ctx.x_padded)
#         dw = xp.zeros_like(weight.data)
#         db = xp.zeros_like(bias.data)

#         for n in range(N):
#             for c_out in range(C_out):
#                 for h in range(H_out):
#                     for w in range(W_out):
#                         h_start = h * SH
#                         w_start = w * SW
#                         h_end = h_start + KH
#                         w_end = w_start + KW

#                         region = ctx.x_padded[n, :,
#                                               h_start:h_end, w_start:w_end]
#                         dw[c_out] += grad_output[n, c_out, h, w] * region
#                         dx_padded[n, :, h_start:h_end, w_start:w_end] += grad_output[n,
#                                                                                      c_out, h, w] * weight.data[c_out]
#                 db[c_out] += xp.sum(grad_output[n, c_out])

#         # Remove padding from dx
#         if PH == 0 and PW == 0:
#             dx = dx_padded
#         else:
#             dx = dx_padded[:, :, PH:-PH, PW:-PW]

#         return dx, dw, db, None, None


class Conv2dFunction(Function):
    """
    使用im2col的加速版本
    """
    @staticmethod
    def forward(ctx: Context, x, weight, bias, stride, padding):
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding

        N, C_in, H_in, W_in = x.data.shape
        C_out, _, KH, KW = weight.data.shape
        SH, SW = stride
        PH, PW = padding

        # im2col
        x_col, x_padded, H_out, W_out = im2col(x.data, KH, KW, SH, SW, PH, PW)
        w_col = weight.data.reshape(C_out, -1)  # (C_out, C_in * KH * KW)

        out = x_col @ w_col.T  # (N*H_out*W_out, C_out)
        if bias is not None:
            out += bias.data.reshape(1, -1)

        out = out.reshape(N, H_out, W_out, C_out).transpose(
            0, 3, 1, 2)  # (N, C_out, H_out, W_out)

        # 保存用于反向传播
        ctx.x_col = x_col
        ctx.x_padded_shape = x_padded.shape
        ctx.H_out = H_out
        ctx.W_out = W_out

        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        x, weight, bias = ctx.saved_tensors
        SH, SW = ctx.stride
        PH, PW = ctx.padding
        KH, KW = weight.data.shape[2:]
        N, C_in, H_in, W_in = x.data.shape
        C_out = weight.data.shape[0]
        H_out, W_out = ctx.H_out, ctx.W_out

        grad_output_reshaped = grad_output.transpose(
            0, 2, 3, 1).reshape(-1, C_out)  # (N*H_out*W_out, C_out)

        # dw
        dw = grad_output_reshaped.T @ ctx.x_col  # (C_out, C_in * KH * KW)
        dw = dw.reshape(weight.data.shape)

        # db
        db = grad_output_reshaped.sum(axis=0)

        # dx
        w_col = weight.data.reshape(C_out, -1)
        # (N*H_out*W_out, C_in * KH * KW)
        dx_col = grad_output_reshaped @ w_col
        dx = col2im(dx_col, x.data.shape, KH, KW, SH, SW, PH, PW, H_out, W_out)

        return dx, dw, db, None, None


class MaxPool2dFunction(Function):
    @staticmethod
    def forward(ctx: Context, x, kernel_size, stride):
        ctx.save_for_backward(x)
        ctx.kernel_size = kernel_size
        ctx.stride = stride

        N, C, H, W = x.data.shape
        KH, KW = kernel_size
        SH, SW = stride

        H_out = (H - KH) // SH + 1
        W_out = (W - KW) // SW + 1

        out = xp.zeros((N, C, H_out, W_out))
        ctx.max_indices = xp.zeros_like(x.data, dtype=bool)

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * SH
                        w_start = w * SW
                        h_end = h_start + KH
                        w_end = w_start + KW

                        region = x.data[n, c, h_start:h_end, w_start:w_end]
                        max_val = xp.max(region)
                        out[n, c, h, w] = max_val

                        # 保存最大值位置
                        mask = (region == max_val)
                        ctx.max_indices[n, c, h_start:h_end,
                                        w_start:w_end] += mask

        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        x, = ctx.saved_tensors
        KH, KW = ctx.kernel_size
        SH, SW = ctx.stride
        N, C, H, W = x.data.shape
        H_out, W_out = grad_output.shape[2:]

        dx = xp.zeros_like(x.data)

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * SH
                        w_start = w * SW
                        h_end = h_start + KH
                        w_end = w_start + KW

                        mask = ctx.max_indices[n, c,
                                               h_start:h_end, w_start:w_end]
                        dx[n, c, h_start:h_end,
                            w_start:w_end] += grad_output[n, c, h, w] * mask

        return dx, None, None
