import numpy as np

class Activation:
    """激活函数的基类"""
    def __init__(self):
        """初始化激活函数实例。

        Attributes:
            input: 存储前向传播时的输入。
            output: 存储前向传播时的输出。
        """
        self.input = None
        self.output = None

    def forward(self, input):
        """执行前向传播。

        Args:
            input (numpy.ndarray): 输入数据。

        Raises:
            NotImplementedError: 如果子类没有实现此方法。
        """
        raise NotImplementedError

    def backward(self, output_gradient):
        """执行反向传播。

        Args:
            output_gradient (numpy.ndarray): 输出的梯度。

        Raises:
            NotImplementedError: 如果子类没有实现此方法。
        """
        raise NotImplementedError

class ReLU(Activation):
    """ReLU (Rectified Linear Unit) 激活函数。"""
    def forward(self, input):
        """ReLU 的前向传播。

        Args:
            input (numpy.ndarray): 输入数据。

        Returns:
            numpy.ndarray: ReLU激活后的输出。
        """
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient):
        """ReLU 的反向传播。

        Args:
            output_gradient (numpy.ndarray): 输出的梯度。

        Returns:
            numpy.ndarray: 输入的梯度。
        """
        return output_gradient * (self.input > 0)

class LeakyReLU(Activation):
    """Leaky ReLU 激活函数。"""
    def __init__(self, alpha=0.01):
        """初始化 LeakyReLU 实例。

        Args:
            alpha (float, optional): 负斜率系数。默认为 0.01。
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        """LeakyReLU 的前向传播。

        Args:
            input (numpy.ndarray): 输入数据。

        Returns:
            numpy.ndarray: LeakyReLU激活后的输出。
        """
        self.input = input
        return np.maximum(self.alpha * input, input)

    def backward(self, output_gradient):
        """LeakyReLU 的反向传播。

        Args:
            output_gradient (numpy.ndarray): 输出的梯度。

        Returns:
            numpy.ndarray: 输入的梯度。
        """
        dx = np.ones_like(self.input)
        dx[self.input < 0] = self.alpha
        return output_gradient * dx

class Sigmoid(Activation):
    """Sigmoid 激活函数。"""
    def forward(self, input):
        """Sigmoid 的前向传播。

        Args:
            input (numpy.ndarray): 输入数据。

        Returns:
            numpy.ndarray: Sigmoid激活后的输出。
        """
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, output_gradient):
        """Sigmoid 的反向传播。

        Args:
            output_gradient (numpy.ndarray): 输出的梯度。

        Returns:
            numpy.ndarray: 输入的梯度。
        """
        return output_gradient * (self.output * (1 - self.output))

class Tanh(Activation):
    """Tanh (Hyperbolic Tangent) 激活函数。"""
    def forward(self, input):
        """Tanh 的前向传播。

        Args:
            input (numpy.ndarray): 输入数据。

        Returns:
            numpy.ndarray: Tanh激活后的输出。
        """
        self.input = input
        self.output = np.tanh(input)
        return self.output

    def backward(self, output_gradient):
        """Tanh 的反向传播。

        Args:
            output_gradient (numpy.ndarray): 输出的梯度。

        Returns:
            numpy.ndarray: 输入的梯度。
        """
        return output_gradient * (1 - np.square(self.output))

class Softmax(Activation):
    """Softmax 激活函数。
    通常用于多分类问题的输出层。
    """
    def forward(self, input):
        """Softmax 的前向传播。

        Args:
            input (numpy.ndarray): 输入数据 (通常是logits)。

        Returns:
            numpy.ndarray: Softmax激活后的概率分布。
        """
        # 减去最大值以保证数值稳定性
        exp_scores = np.exp(input - np.max(input, axis=-1, keepdims=True))
        self.output = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        return self.output

    def backward(self, output_gradient):
        """Softmax 的反向传播。

        注意: Softmax 的反向传播通常与交叉熵损失函数结合处理，
        因为 dL/dz = p - y (其中 p 是softmax输出, y 是真实标签)。
        此处的实现是一个更通用的版本，用于直接传播梯度，
        如果 Softmax 单独使用或作为更复杂损失函数的一部分。
        在典型的分类设置中，这个梯度可能不会直接使用。

        Args:
            output_gradient (numpy.ndarray): 输出的梯度 (dL/dp)。

        Returns:
            numpy.ndarray: 输入的梯度 (dL/dz)。
        """
        # s 是 softmax 的输出 (self.output)
        # jacobian 是 d(softmax_i)/d(input_j) 的雅可比矩阵
        # 对于单个样本，如果 s 是一个列向量：
        # J = diag(s) - s * s.T
        # dL/dz = dL/dp * dp/dz = output_gradient * jacobian
        
        # 此实现假设 output_gradient 的形状与 self.output 相同。
        # 并且它计算的是 dL/dz_i = sum_j (dL/dp_j * dp_j/dz_i)
        # 这需要对每个样本独立计算雅可比矩阵并进行点积。
        
        # 一个简化的处理方式（当与交叉熵损失结合时，梯度更简单）：
        # 如果 output_gradient 是 dL/dp，那么 dL/dz_i = p_i * (dL/dp_i - sum_k(dL/dp_k * p_k))
        # 这里我们提供一个更直接的雅可比矩阵方法，尽管它可能不是最高效的
        # 或者在典型场景下最常用的。

        # 获取批处理大小和类别数
        batch_size = self.output.shape[0]
        num_classes = self.output.shape[1]
        input_gradient = np.zeros_like(self.output)

        for i in range(batch_size):
            # 单个样本的 softmax 输出
            s_i = self.output[i, :].reshape(-1, 1) # (num_classes, 1)
            # 计算单个样本的雅可比矩阵
            jacobian_i = np.diagflat(s_i) - np.dot(s_i, s_i.T) # (num_classes, num_classes)
            # 计算单个样本的输入梯度
            input_gradient[i, :] = np.dot(jacobian_i.T, output_gradient[i, :]) # (num_classes,)
            # 注意：这里是 jacobian_i.T @ output_gradient[i, :].T 如果 output_gradient[i,:] 是行向量
            # 或者 output_gradient[i, :].reshape(-1,1) 然后 jacobian_i @ grad
            # 确保维度匹配。假设 output_gradient[i,:] 是 (num_classes,) 形状
            # 我们需要 dL/dz_k = sum_j (dL/dp_j * dp_j/dz_k)
            # dp_j/dz_k = p_j * (delta_jk - p_k)
            # dL/dz_k = sum_j (output_gradient_j * p_j * (delta_jk - p_k))
            #         = output_gradient_k * p_k * (1 - p_k) - sum_{j!=k} (output_gradient_j * p_j * p_k)
            #         = p_k * (output_gradient_k - sum_j (output_gradient_j * p_j) )
            # 这个公式更常用且高效。
            
            # 使用更高效的公式: dL/dz_i = p_i * (dL/dp_i - sum(dL/dp * p)) 
            #其中 p 是softmax输出, dL/dp 是 output_gradient
            p_i = self.output[i, :] # (num_classes,)
            dL_dp_i = output_gradient[i, :] # (num_classes,)
            input_gradient[i, :] = p_i * (dL_dp_i - np.sum(dL_dp_i * p_i, axis=-1, keepdims=True))
            
        return input_gradient