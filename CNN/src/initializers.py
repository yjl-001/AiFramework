import numpy as np

class Initializer:
    """权重初始化器的基类。"""
    def initialize(self, shape):
        """根据给定的形状初始化权重。

        Args:
            shape (tuple): 权重的形状。

        Raises:
            NotImplementedError: 如果子类没有实现此方法。

        Returns:
            numpy.ndarray: 初始化后的权重。
        """
        raise NotImplementedError

class Zeros(Initializer):
    """将权重初始化为全零的初始化器。"""
    def initialize(self, shape):
        """将权重初始化为全零。

        Args:
            shape (tuple): 权重的形状。

        Returns:
            numpy.ndarray: 初始化为零的权重。
        """
        return np.zeros(shape)

class Ones(Initializer):
    """将权重初始化为全一的初始化器。"""
    def initialize(self, shape):
        """将权重初始化为全一。

        Args:
            shape (tuple): 权重的形状。

        Returns:
            numpy.ndarray: 初始化为一的权重。
        """
        return np.ones(shape)

class RandomNormal(Initializer):
    """使用正态分布初始化权重的初始化器。"""
    def __init__(self, mean=0.0, stddev=0.05):
        """初始化 RandomNormal 实例。

        Args:
            mean (float, optional): 正态分布的均值。默认为 0.0。
            stddev (float, optional): 正态分布的标准差。默认为 0.05。
        """
        self.mean = mean
        self.stddev = stddev

    def initialize(self, shape):
        """使用正态分布初始化权重。

        Args:
            shape (tuple): 权重的形状。

        Returns:
            numpy.ndarray: 从正态分布采样的权重。
        """
        return np.random.normal(self.mean, self.stddev, size=shape)

class RandomUniform(Initializer):
    """使用均匀分布初始化权重的初始化器。"""
    def __init__(self, minval=-0.05, maxval=0.05):
        """初始化 RandomUniform 实例。

        Args:
            minval (float, optional): 均匀分布的最小值。默认为 -0.05。
            maxval (float, optional): 均匀分布的最大值。默认为 0.05。
        """
        self.minval = minval
        self.maxval = maxval

    def initialize(self, shape):
        """使用均匀分布初始化权重。

        Args:
            shape (tuple): 权重的形状。

        Returns:
            numpy.ndarray: 从均匀分布采样的权重。
        """
        return np.random.uniform(self.minval, self.maxval, size=shape)

class GlorotNormal(Initializer):
    """Glorot (Xavier) 正态初始化器。
    从均值为0，标准差为 sqrt(2 / (fan_in + fan_out)) 的截断正态分布中采样。
    """
    def initialize(self, shape):
        """使用 Glorot 正态分布初始化权重。

        Args:
            shape (tuple): 权重的形状。

        Returns:
            numpy.ndarray: 初始化后的权重。
        """
        fan_in, fan_out = self._compute_fans(shape)
        stddev = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0.0, stddev, size=shape)

    def _compute_fans(self, shape):
        """计算权重的 fan_in 和 fan_out。

        Args:
            shape (tuple): 权重的形状。

        Returns:
            tuple: (fan_in, fan_out)
        """
        if len(shape) == 2:  # 全连接层: (input_dim, output_dim)
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) == 4:  # 卷积层: (num_filters, input_channels, filter_height, filter_width)
            receptive_field_size = np.prod(shape[2:]) # filter_height * filter_width
            fan_in = shape[1] * receptive_field_size  # input_channels * receptive_field_size
            fan_out = shape[0] * receptive_field_size # num_filters * receptive_field_size
        else:
            # 其他形状的回退方案，通常是全连接层或卷积层
            fan_in = np.sqrt(np.prod(shape))
            fan_out = np.sqrt(np.prod(shape))
        return fan_in, fan_out

class GlorotUniform(Initializer):
    """Glorot (Xavier) 均匀初始化器。
    从 [-limit, limit] 范围内的均匀分布中采样，其中 limit = sqrt(6 / (fan_in + fan_out))。
    """
    def initialize(self, shape):
        """使用 Glorot 均匀分布初始化权重。

        Args:
            shape (tuple): 权重的形状。

        Returns:
            numpy.ndarray: 初始化后的权重。
        """
        fan_in, fan_out = self._compute_fans(shape)
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)

    def _compute_fans(self, shape):
        """计算权重的 fan_in 和 fan_out。

        Args:
            shape (tuple): 权重的形状。

        Returns:
            tuple: (fan_in, fan_out)
        """
        if len(shape) == 2: # 全连接层
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) == 4: # 卷积层
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        else:
            fan_in = np.sqrt(np.prod(shape))
            fan_out = np.sqrt(np.prod(shape))
        return fan_in, fan_out

class HeNormal(Initializer):
    """He 正态初始化器。
    从均值为0，标准差为 sqrt(2 / fan_in) 的截断正态分布中采样。
    通常用于 ReLU 激活函数。
    """
    def initialize(self, shape):
        """使用 He 正态分布初始化权重。

        Args:
            shape (tuple): 权重的形状。

        Returns:
            numpy.ndarray: 初始化后的权重。
        """
        fan_in, _ = self._compute_fans(shape)
        stddev = np.sqrt(2.0 / fan_in)
        return np.random.normal(0.0, stddev, size=shape)

    def _compute_fans(self, shape):
        """计算权重的 fan_in 和 fan_out。

        Args:
            shape (tuple): 权重的形状。

        Returns:
            tuple: (fan_in, fan_out)
        """
        if len(shape) == 2: # 全连接层
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) == 4: # 卷积层
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        else:
            fan_in = np.sqrt(np.prod(shape))
            fan_out = np.sqrt(np.prod(shape))
        return fan_in, fan_out

class HeUniform(Initializer):
    """He 均匀初始化器。
    从 [-limit, limit] 范围内的均匀分布中采样，其中 limit = sqrt(6 / fan_in)。
    通常用于 ReLU 激活函数。
    """
    def initialize(self, shape):
        """使用 He 均匀分布初始化权重。

        Args:
            shape (tuple): 权重的形状。

        Returns:
            numpy.ndarray: 初始化后的权重。
        """
        fan_in, _ = self._compute_fans(shape)
        limit = np.sqrt(6.0 / fan_in)
        return np.random.uniform(-limit, limit, size=shape)

    def _compute_fans(self, shape):
        """计算权重的 fan_in 和 fan_out。

        Args:
            shape (tuple): 权重的形状。

        Returns:
            tuple: (fan_in, fan_out)
        """
        if len(shape) == 2: # 全连接层
            fan_in = shape[0]
            fan_out = shape[1]
        elif len(shape) == 4: # 卷积层
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        else:
            fan_in = np.sqrt(np.prod(shape))
            fan_out = np.sqrt(np.prod(shape))
        return fan_in, fan_out