import numpy as np

# 优化器基类
class Optimizer:
    # 更新权重和偏置的方法，子类必须实现
    def update(self, weights, biases, gradients, learning_rate):
        raise NotImplementedError

# 随机梯度下降 (Stochastic Gradient Descent, SGD) 优化器
class SGD(Optimizer):
    # 更新权重和偏置
    def update(self, weights, biases, gradients, learning_rate):
        updated_weights = []
        updated_biases = []
        for i in range(len(weights)):
            # 权重更新：W = W - learning_rate * dW
            updated_weights.append(weights[i] - learning_rate * gradients[i][0])
            # 偏置更新：b = b - learning_rate * db
            updated_biases.append(biases[i] - learning_rate * gradients[i][1])
        return updated_weights, updated_biases

# 动量 (Momentum) 优化器
class Momentum(Optimizer):
    def __init__(self, beta=0.9):
        self.beta = beta  # 动量参数
        self.v_w = []     # 权重动量
        self.v_b = []     # 偏置动量

    # 更新权重和偏置
    def update(self, weights, biases, gradients, learning_rate):
        # 如果是第一次更新，初始化动量为零
        if not self.v_w:
            self.v_w = [np.zeros_like(w) for w in weights]
            self.v_b = [np.zeros_like(b) for b in biases]

        updated_weights = []
        updated_biases = []
        for i in range(len(weights)):
            # 更新权重动量：v_w = beta * v_w + (1 - beta) * dW
            self.v_w[i] = self.beta * self.v_w[i] + (1 - self.beta) * gradients[i][0]
            # 更新偏置动量：v_b = beta * v_b + (1 - beta) * db
            self.v_b[i] = self.beta * self.v_b[i] + (1 - self.beta) * gradients[i][1]
            # 权重更新：W = W - learning_rate * v_w
            updated_weights.append(weights[i] - learning_rate * self.v_w[i])
            # 偏置更新：b = b - learning_rate * v_b
            updated_biases.append(biases[i] - learning_rate * self.v_b[i])
        return updated_weights, updated_biases

# RMSprop 优化器
class RMSprop(Optimizer):
    def __init__(self, beta=0.999, epsilon=1e-8):
        self.beta = beta        # 平滑常数
        self.epsilon = epsilon  # 避免除以零的小常数
        self.s_w = []           # 权重平方梯度均值
        self.s_b = []           # 偏置平方梯度均值

    # 更新权重和偏置
    def update(self, weights, biases, gradients, learning_rate):
        # 如果是第一次更新，初始化平方梯度均值为零
        if not self.s_w:
            self.s_w = [np.zeros_like(w) for w in weights]
            self.s_b = [np.zeros_like(b) for b in biases]

        updated_weights = []
        updated_biases = []
        for i in range(len(weights)):
            # 更新权重平方梯度均值：s_w = beta * s_w + (1 - beta) * dW^2
            self.s_w[i] = self.beta * self.s_w[i] + (1 - self.beta) * (gradients[i][0]**2)
            # 更新偏置平方梯度均值：s_b = beta * s_b + (1 - beta) * db^2
            self.s_b[i] = self.beta * self.s_b[i] + (1 - self.beta) * (gradients[i][1]**2)
            # 权重更新：W = W - learning_rate * dW / (sqrt(s_w) + epsilon)
            updated_weights.append(weights[i] - learning_rate * gradients[i][0] / (np.sqrt(self.s_w[i]) + self.epsilon))
            # 偏置更新：b = b - learning_rate * db / (sqrt(s_b) + epsilon)
            updated_biases.append(biases[i] - learning_rate * gradients[i][1] / (np.sqrt(self.s_b[i]) + self.epsilon))
        return updated_weights, updated_biases

# Adam 优化器
class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1      # 一阶矩估计的指数衰减率
        self.beta2 = beta2      # 二阶矩估计的指数衰减率
        self.epsilon = epsilon  # 避免除以零的小常数
        self.m_w = []           # 权重的一阶矩估计
        self.v_w = []           # 权重的二阶矩估计
        self.m_b = []           # 偏置的一阶矩估计
        self.v_b = []           # 偏置的二阶矩估计
        self.t = 0              # 时间步

    # 更新权重和偏置
    def update(self, weights, biases, gradients, learning_rate):
        self.t += 1  # 增加时间步
        # 如果是第一次更新，初始化矩估计为零
        if not self.m_w:
            self.m_w = [np.zeros_like(w) for w in weights]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_b = [np.zeros_like(b) for b in biases]

        updated_weights = []
        updated_biases = []
        for i in range(len(weights)):
            # 更新权重的一阶矩估计：m_w = beta1 * m_w + (1 - beta1) * dW
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gradients[i][0]
            # 更新权重的二阶矩估计：v_w = beta2 * v_w + (1 - beta2) * dW^2
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (gradients[i][0]**2)
            # 偏差修正后的一阶矩估计
            m_w_corrected = self.m_w[i] / (1 - self.beta1**self.t)
            # 偏差修正后的二阶矩估计
            v_w_corrected = self.v_w[i] / (1 - self.beta2**self.t)
            # 权重更新：W = W - learning_rate * m_w_corrected / (sqrt(v_w_corrected) + epsilon)
            updated_weights.append(weights[i] - learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon))

            # 更新偏置的一阶矩估计：m_b = beta1 * m_b + (1 - beta1) * db
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gradients[i][1]
            # 更新偏置的二阶矩估计：v_b = beta2 * v_b + (1 - beta2) * db^2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (gradients[i][1]**2)
            # 偏差修正后的一阶矩估计
            m_b_corrected = self.m_b[i] / (1 - self.beta1**self.t)
            # 偏差修正后的二阶矩估计
            v_b_corrected = self.v_b[i] / (1 - self.beta2**self.t)
            # 偏置更新：b = b - learning_rate * m_b_corrected / (sqrt(v_b_corrected) + epsilon)
            updated_biases.append(biases[i] - learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon))
        return updated_weights, updated_biases

# 根据名称获取优化器实例的工厂函数
def get_optimizer(name, **kwargs):
    if name == 'sgd':
        return SGD()
    elif name == 'momentum':
        return Momentum(**kwargs)
    elif name == 'rmsprop':
        return RMSprop(**kwargs)
    elif name == 'adam':
        return Adam(**kwargs)
    else:
        raise ValueError(f"未知优化器: {name}")