from .function import Function
from mytorch.backend import xp


class MaxPool2dFunction(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride=None, padding=0):
        if stride is None:
            stride = kernel_size

        N, C, H, W = x.shape
        KH, KW = kernel_size
        SH, SW = stride
        PH, PW = padding, padding

        # Padding
        x_padded = xp.pad(x.data, ((0, 0), (0, 0), (PH, PH), (PW, PW)),
                          mode='constant', constant_values=(float('-inf'),))

        # 输出尺寸
        out_h = (H + 2 * PH - KH) // SH + 1
        out_w = (W + 2 * PW - KW) // SW + 1
        out = xp.zeros((N, C, out_h, out_w), dtype=x.data.dtype)

        # 记录最大值索引
        max_indices = xp.zeros_like(out, dtype=xp.int32)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * SH
                w_start = j * SW
                window = x_padded[:, :, h_start:h_start+KH,
                                  w_start:w_start+KW]  # shape: (N, C, KH, KW)
                out[:, :, i, j] = xp.max(window, axis=(2, 3))
                max_indices[:, :, i, j] = xp.argmax(
                    window.reshape(N, C, -1), axis=2)

        ctx.save_for_backward(x, max_indices)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.x_padded_shape = x_padded.shape

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, max_indices = ctx.saved_tensors
        KH, KW = ctx.kernel_size
        SH, SW = ctx.stride
        PH, PW = ctx.padding
        N, C, H, W = x.shape
        x_padded_shape = ctx.x_padded_shape

        grad_input_padded = xp.zeros(x_padded_shape, dtype=grad_output.dtype)

        out_h, out_w = grad_output.shape[2:]

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * SH
                w_start = j * SW

                grad_patch = grad_output[:, :, i, j]
                for n in range(N):
                    for c in range(C):
                        idx = max_indices[n, c, i, j]
                        h_idx = h_start + idx // KW
                        w_idx = w_start + idx % KW
                        grad_input_padded[n, c, h_idx,
                                          w_idx] += grad_patch[n, c]

        # Remove padding
        if PH == 0 and PW == 0:
            return grad_input_padded
        else:
            return grad_input_padded[:, :, PH:PH+H, PW:PW+W]


class AvgPool2dFunction(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride=None, padding=0):
        if stride is None:
            stride = kernel_size

        N, C, H, W = x.shape
        KH, KW = kernel_size
        SH, SW = stride
        PH, PW = padding, padding

        x_padded = xp.pad(x.data, ((0, 0), (0, 0), (PH, PH),
                          (PW, PW)), mode='constant')

        out_h = (H + 2 * PH - KH) // SH + 1
        out_w = (W + 2 * PW - KW) // SW + 1
        out = xp.zeros((N, C, out_h, out_w), dtype=x.data.dtype)

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * SH
                w_start = j * SW
                window = x_padded[:, :, h_start:h_start+KH, w_start:w_start+KW]
                out[:, :, i, j] = xp.mean(window, axis=(2, 3))

        ctx.save_for_backward(x)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.input_shape = x.shape

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        KH, KW = ctx.kernel_size
        SH, SW = ctx.stride
        PH, PW = ctx.padding
        N, C, H, W = ctx.input_shape

        grad_input_padded = xp.zeros(
            (N, C, H + 2 * PH, W + 2 * PW), dtype=grad_output.dtype)
        out_h, out_w = grad_output.shape[2:]

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * SH
                w_start = j * SW
                grad = grad_output[:, :, i, j][:, :, None, None] / (KH * KW)
                grad_input_padded[:, :, h_start:h_start +
                                  KH, w_start:w_start+KW] += grad

        if PH == 0 and PW == 0:
            return grad_input_padded
        else:
            return grad_input_padded[:, :, PH:PH+H, PW:PW+W]
