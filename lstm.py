from mytorch.nn import Module, Linear
from mytorch.tensor import Tensor
from mytorch.backend import xp


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 合并所有门的线性变换：i, f, g, o
        self.x2h = Linear(input_size, 4 * hidden_size)
        self.h2h = Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, h_prev, c_prev):
        # x: (batch, input_size)
        # h_prev, c_prev: (batch, hidden_size)

        gates = self.x2h(x) + self.h2h(h_prev)  # (batch, 4 * hidden_size)

        # 拆分门
        i, f, g, o = gates.chunk(4, dim=1)  # 每个 shape: (batch, hidden_size)

        i = i.sigmoid()
        f = f.sigmoid()
        g = g.tanh()
        o = o.sigmoid()

        c = f * c_prev + i * g
        h = o * c.tanh()

        return h, c


class LSTM(Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x, h_c=None):
        # x: (seq_len, batch_size, input_size)
        seq_len, batch_size, _ = x.shape

        if h_c is None:
            h_t = Tensor.zeros(
                (batch_size, self.hidden_size))
            c_t = Tensor.zeros(
                (batch_size, self.hidden_size))
        else:
            h_t, c_t = h_c

        outputs = []

        for t in range(seq_len):
            x_t = x[t]  # (batch_size, input_size)
            h_t, c_t = self.cell(x_t, h_t, c_t)
            outputs.append(h_t)

        # (seq_len, batch_size, hidden_size)
        output = Tensor.stack(outputs, dim=0)
        return output, (h_t, c_t)


class SimpleLSTM(Module):
    def __init__(self):
        super().__init__()
        self.lstm = LSTM(input_size=28, hidden_size=128)
        self.fc = Linear(128, 10)

    def forward(self, x):
        x = x.squeeze(1)  # (batch, 28, 28)
        x = x.transpose(0, 1)  # (seq_len=28, batch, input_size=28)
        output, (h_n, c_n) = self.lstm(x)
        return self.fc(h_n)  # (batch, 10)
