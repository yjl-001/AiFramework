from mytorch.tensor import Tensor
from mytorch.ops.basic import MatMul
from mytorch.backend import xp

x = Tensor([[1., 2., 3., 4.]], dtype=xp.float32)
a, b = x.chunk(2, dim=1)
y = a + b
loss = y.sum()
loss.backward()

print(x.grad)  # 应该是 [[1, 1, 1, 1]]
