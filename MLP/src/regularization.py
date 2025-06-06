import numpy as np

# 正则化基类
class Regularization:
    # 计算正则化损失的方法，子类必须实现
    def loss(self, weights):
        raise NotImplementedError

    # 计算正则化梯度的方法，子类必须实现
    def gradient(self, weights):
        raise NotImplementedError

# L1 正则化 (Lasso)
class L1(Regularization):
    def __init__(self, reg_lambda):
        self.reg_lambda = reg_lambda  # 正则化强度参数

    # 计算L1正则化损失
    def loss(self, weights):
        l1_loss = 0
        for w in weights:
            l1_loss += np.sum(np.abs(w))  # L1损失是权重的绝对值之和
        return self.reg_lambda * l1_loss

    # 计算L1正则化梯度
    def gradient(self, weights):
        l1_gradients = []
        for w in weights:
            l1_gradients.append(self.reg_lambda * np.sign(w))  # L1梯度是权重的符号函数
        return l1_gradients

# L2 正则化 (Ridge)
class L2(Regularization):
    def __init__(self, reg_lambda):
        self.reg_lambda = reg_lambda  # 正则化强度参数

    # 计算L2正则化损失
    def loss(self, weights):
        l2_loss = 0
        for w in weights:
            l2_loss += np.sum(w**2)  # L2损失是权重的平方和
        return 0.5 * self.reg_lambda * l2_loss  # 通常乘以0.5方便求导

    # 计算L2正则化梯度
    def gradient(self, weights):
        l2_gradients = []
        for w in weights:
            l2_gradients.append(self.reg_lambda * w)  # L2梯度是权重本身
        return l2_gradients

# Elastic Net 正则化 (L1和L2的组合)
class ElasticNet(Regularization):
    def __init__(self, reg_lambda, l1_ratio):
        self.reg_lambda = reg_lambda      # 总正则化强度
        self.l1_ratio = l1_ratio          # L1正则化的比例 (0到1之间)
        # 创建L1和L2正则化实例，并根据l1_ratio分配正则化强度
        self.l1_reg = L1(reg_lambda * l1_ratio)
        self.l2_reg = L2(reg_lambda * (1 - l1_ratio))

    # 计算Elastic Net正则化损失
    def loss(self, weights):
        return self.l1_reg.loss(weights) + self.l2_reg.loss(weights)  # 损失是L1和L2损失之和

    # 计算Elastic Net正则化梯度
    def gradient(self, weights):
        l1_grad = self.l1_reg.gradient(weights)
        l2_grad = self.l2_reg.gradient(weights)
        elastic_gradients = []
        for i in range(len(weights)):
            elastic_gradients.append(l1_grad[i] + l2_grad[i])  # 梯度是L1和L2梯度之和
        return elastic_gradients

# 根据名称获取正则化实例的工厂函数
def get_regularization(name, reg_lambda, l1_ratio=None):
    if name == 'l1':
        return L1(reg_lambda)
    elif name == 'l2':
        return L2(reg_lambda)
    elif name == 'elastic_net':
        if l1_ratio is None:
            raise ValueError("ElasticNet正则化必须提供l1_ratio。")
        return ElasticNet(reg_lambda, l1_ratio)
    else:
        raise ValueError(f"未知正则化类型: {name}")