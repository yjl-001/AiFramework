from mytorch.tensor import Tensor
from mytorch.ops.basic import MatMul
from mytorch.backend import xp


# 数值梯度检查工具
def numerical_grad(f, x, eps=1e-6):
    grad = xp.zeros_like(x)
    it = xp.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        ix = it.multi_index
        old_value = x[ix]

        x[ix] = old_value + eps
        fx1 = f(x)

        x[ix] = old_value - eps
        fx2 = f(x)

        x[ix] = old_value  # 恢复

        grad[ix] = (fx1 - fx2) / (2 * eps)
        it.iternext()

    return grad


# 测试函数：支持 batched matmul
def test_batched_matmul():
    xp.random.seed(42)

    # 构造 batched 输入张量
    B, N, M, K, P = 2, 3, 4, 5, 6  # batch, num_heads, dim1, dim2, dim3
    a_data = xp.random.randn(B, N, M, K)
    b_data = xp.random.randn(B, N, K, P)

    # 创建 Tensor（支持自动求导）
    a = Tensor(a_data)
    b = Tensor(b_data)

    # 执行 batched matmul
    out = MatMul.apply(a, b)  # shape: (B, N, M, P)

    # 构造损失函数：将所有元素求和（标量）
    loss = out.sum()
    loss.backward()

    # 数值梯度检查
    def f_a(x):
        return xp.matmul(x, b_data).sum()

    def f_b(x):
        return xp.matmul(a_data, x).sum()

    grad_a_numeric = numerical_grad(f_a, a_data)
    grad_b_numeric = numerical_grad(f_b, b_data)

    # 比较误差
    err_a = xp.max(xp.abs(grad_a_numeric - a.grad))
    err_b = xp.max(xp.abs(grad_b_numeric - b.grad))

    print("最大误差（a）：", err_a)
    print("最大误差（b）：", err_b)

    assert err_a < 1e-4, "a 的梯度误差过大"
    assert err_b < 1e-4, "b 的梯度误差过大"
    print("✅ Batched MatMul 前向 & 反向梯度检查通过！")


if __name__ == "__main__":
    test_batched_matmul()
