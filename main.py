from common.tensor import Tensor
from common.module import Module
from common.linear import Linear


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 128)
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # 初始化输入张量
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    w = Tensor([[2.0], [1.0]])

    # 前向传播
    y = x @ w          # shape (2, 1)
    z = y.sum()              # scalar

    print("z =", z)

    # 反向传播
    z.backward()

    print("x.grad =\n", x.grad)
    print("w.grad =\n", w.grad)
