from .function import Function, Context
import numpy as np


class Conv2dFunction(Function):
    @staticmethod
    def forward(ctx: Context, x, weight, bias, stride, padding):
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding

        N, C_in, H_in, W_in = x.data.shape
        C_out, _, KH, KW = weight.data.shape
        SH, SW = stride
        PH, PW = padding

        # Padding
        x_padded = np.pad(x.data, ((0, 0), (0, 0), (PH, PH),
                          (PW, PW)), mode='constant')

        H_out = (H_in + 2 * PH - KH) // SH + 1
        W_out = (W_in + 2 * PW - KW) // SW + 1

        out = np.zeros((N, C_out, H_out, W_out))

        for n in range(N):
            for c_out in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * SH
                        w_start = w * SW
                        h_end = h_start + KH
                        w_end = w_start + KW

                        region = x_padded[n, :, h_start:h_end, w_start:w_end]
                        out[n, c_out, h, w] = np.sum(
                            region * weight.data[c_out]) + bias.data[c_out]

        ctx.x_padded = x_padded
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        N, C_in, H_in, W_in = x.data.shape
        C_out, _, KH, KW = weight.data.shape
        SH, SW = stride
        PH, PW = padding
        H_out, W_out = grad_output.shape[2:]

        dx_padded = np.zeros_like(ctx.x_padded)
        dw = np.zeros_like(weight.data)
        db = np.zeros_like(bias.data)

        for n in range(N):
            for c_out in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * SH
                        w_start = w * SW
                        h_end = h_start + KH
                        w_end = w_start + KW

                        region = ctx.x_padded[n, :,
                                              h_start:h_end, w_start:w_end]
                        dw[c_out] += grad_output[n, c_out, h, w] * region
                        dx_padded[n, :, h_start:h_end, w_start:w_end] += grad_output[n,
                                                                                     c_out, h, w] * weight.data[c_out]
                db[c_out] += np.sum(grad_output[n, c_out])

        # Remove padding from dx
        if PH == 0 and PW == 0:
            dx = dx_padded
        else:
            dx = dx_padded[:, :, PH:-PH, PW:-PW]

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

        out = np.zeros((N, C, H_out, W_out))
        ctx.max_indices = np.zeros_like(x.data, dtype=bool)

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * SH
                        w_start = w * SW
                        h_end = h_start + KH
                        w_end = w_start + KW

                        region = x.data[n, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(region)
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

        dx = np.zeros_like(x.data)

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
