# MLP/src/__init__.py
# 这个文件使得 'src' 目录成为一个Python包。
# 它也定义了当使用 'from MLP.src import *' 时，哪些模块和名称会被导入。

# 从各个模块导入主要的类和函数，以便用户可以直接从 MLP.src 导入它们。
from .api import MLPAPI
from .mlp import MLP
from .activations import get_activation, ReLU, Sigmoid, Softmax, Linear
from .losses import get_loss_function, MeanSquaredError, CrossEntropyLoss
from .optimizers import get_optimizer, SGD, Momentum, RMSprop, Adam
from .initializers import get_initializer, Zeros, Ones, RandomNormal, RandomUniform, GlorotUniform, GlorotNormal, HeUniform, HeNormal
from .regularization import get_regularization, L1, L2, ElasticNet
from .metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from .utils import to_one_hot, train_test_split, get_batches

# __all__ 列表定义了 'from MLP.src import *' 时公开的接口。
# 这有助于控制包的命名空间，并明确哪些部分是供外部使用的。
__all__ = [
    'MLPAPI',  # MLP模型的高级API接口
    'MLP',     # 核心MLP模型类
    # 激活函数
    'get_activation', 'ReLU', 'Sigmoid', 'Softmax', 'Linear',
    # 损失函数
    'get_loss_function', 'MeanSquaredError', 'CrossEntropyLoss',
    # 优化器
    'get_optimizer', 'SGD', 'Momentum', 'RMSprop', 'Adam',
    # 初始化器
    'get_initializer', 'Zeros', 'Ones', 'RandomNormal', 'RandomUniform', 'GlorotUniform', 'GlorotNormal', 'HeUniform', 'HeNormal',
    # 正则化方法
    'get_regularization', 'L1', 'L2', 'ElasticNet',
    # 评估指标
    'confusion_matrix', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score',
    # 工具函数
    'to_one_hot', 'train_test_split', 'get_batches'
]