import numpy as np

# 权重初始化器基类
class Initializer:
    # 初始化权重的方法，子类必须实现
    def initialize(self, shape):
        raise NotImplementedError

# 全零初始化器
class Zeros(Initializer):
    # 初始化为全零矩阵
    def initialize(self, shape):
        return np.zeros(shape)

# 全一初始化器
class Ones(Initializer):
    # 初始化为全一矩阵
    def initialize(self, shape):
        return np.ones(shape)

# 随机正态分布初始化器
class RandomNormal(Initializer):
    # 使用标准正态分布初始化权重
    def initialize(self, shape):
        return np.random.randn(*shape)

# 随机均匀分布初始化器
class RandomUniform(Initializer):
    # 使用0到1之间的均匀分布初始化权重
    def initialize(self, shape):
        return np.random.rand(*shape)

# Glorot均匀分布初始化器 (Xavier均匀初始化)
class GlorotUniform(Initializer):
    # 根据输入和输出单元的数量计算均匀分布的范围
    def initialize(self, shape):
        fan_in, fan_out = shape
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(low=-limit, high=limit, size=shape)

# Glorot正态分布初始化器 (Xavier正态初始化)
class GlorotNormal(Initializer):
    # 根据输入和输出单元的数量计算正态分布的标准差
    def initialize(self, shape):
        fan_in, fan_out = shape
        std_dev = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(loc=0.0, scale=std_dev, size=shape)

# He均匀分布初始化器
class HeUniform(Initializer):
    # 根据输入单元的数量计算均匀分布的范围
    def initialize(self, shape):
        fan_in, fan_out = shape
        limit = np.sqrt(6 / fan_in)
        return np.random.uniform(low=-limit, high=limit, size=shape)

# He正态分布初始化器
class HeNormal(Initializer):
    # 根据输入单元的数量计算正态分布的标准差
    def initialize(self, shape):
        fan_in, fan_out = shape
        std_dev = np.sqrt(2 / fan_in)
        return np.random.normal(loc=0.0, scale=std_dev, size=shape)

# 根据名称获取初始化器实例的工厂函数
def get_initializer(name):
    if name == 'zeros':
        return Zeros()
    elif name == 'ones':
        return Ones()
    elif name == 'random_normal':
        return RandomNormal()
    elif name == 'random_uniform':
        return RandomUniform()
    elif name == 'glorot_uniform':
        return GlorotUniform()
    elif name == 'glorot_normal':
        return GlorotNormal()
    elif name == 'he_uniform':
        return HeUniform()
    elif name == 'he_normal':
        return HeNormal()
    else:
        raise ValueError(f"未知初始化器: {name}")