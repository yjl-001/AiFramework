import numpy as np

# 激活函数基类
class Activation:
    # 前向传播方法，子类必须实现
    def forward(self, Z):
        raise NotImplementedError

    # 反向传播方法，子类必须实现
    def backward(self, Z, dA):
        raise NotImplementedError

# ReLU (Rectified Linear Unit) 激活函数
class ReLU(Activation):
    # 前向传播：max(0, Z)
    def forward(self, Z):
        self.cache = Z  # 缓存输入Z，用于反向传播
        return np.maximum(0, Z)

    # 反向传播：当Z > 0时梯度为1，否则为0
    def backward(self, Z_cache, dA):
        dZ = np.array(dA, copy=True) # 复制上游梯度dA
        dZ[Z_cache <= 0] = 0  # 当Z <= 0时，梯度设为0
        return dZ

# Sigmoid 激活函数
class Sigmoid(Activation):
    # 前向传播：1 / (1 + exp(-Z))
    def forward(self, Z):
        self.cache = 1 / (1 + np.exp(-Z))  # 缓存Sigmoid输出值，用于反向传播
        return self.cache

    # 反向传播：dA * s * (1 - s)，其中s是Sigmoid输出
    def backward(self, Z_cache_is_A, dA):
        s = Z_cache_is_A  # Z_cache在这里实际上是前向传播的输出A
        dZ = dA * s * (1 - s)  # Sigmoid的导数公式
        return dZ

# Softmax 激活函数
class Softmax(Activation):
    # 前向传播：将输入转换为概率分布
    def forward(self, Z):
        # Z的形状预期为 (batch_size, num_classes)
        # 为了数值稳定性，减去Z中的最大值
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        # 计算Softmax输出，即归一化后的概率
        self.cache = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return self.cache  # 这是激活值A

    # 反向传播：通常与交叉熵损失结合使用，简化梯度计算
    def backward(self, A_cache, dL_dA):
        # 当Softmax与交叉熵损失结合时，dL/dZ 的导数通常简化为 A - Y (Y是one-hot真实标签)
        # 因此，这里假设传入的dL_dA已经是简化后的dL/dZ。
        # 如果需要独立的Softmax导数，它会涉及雅可比矩阵，更为复杂。
        # 在大多数神经网络中，Softmax是最后一层并与交叉熵一起使用，
        # 所以梯度 dL/dZ 直接传递给这一层激活前的输出 (Z)。
        return dL_dA  # 占位符，通常与交叉熵损失结合使用

# 线性激活函数（无激活，直接传递）
class Linear(Activation):
    # 前向传播：直接返回输入Z
    def forward(self, Z):
        self.cache = Z  # 缓存输入Z
        return Z

    # 反向传播：直接返回上游梯度dA
    def backward(self, Z_cache, dA):
        return dA


# 根据名称获取激活函数实例的工厂函数
def get_activation(name):
    if name == 'relu':
        return ReLU()
    elif name == 'sigmoid':
        return Sigmoid()
    elif name == 'softmax':
        return Softmax()
    elif name == 'linear':
        return Linear()
    else:
        raise ValueError(f"未知激活函数: {name}")