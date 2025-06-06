import numpy as np

# 损失函数基类
class Loss:
    # 计算损失的方法，子类必须实现
    def compute_loss(self, y_true, y_pred):
        raise NotImplementedError

    # 计算损失对预测值的梯度的方法，子类必须实现
    def backward(self, y_true, y_pred):
        raise NotImplementedError

# 均方误差 (Mean Squared Error, MSE) 损失函数
class MeanSquaredError(Loss):
    # 计算均方误差损失
    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    # 计算均方误差损失对预测值的梯度
    def backward(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]

# 交叉熵损失 (Cross Entropy Loss) 函数
class CrossEntropyLoss(Loss):
    # 计算交叉熵损失
    def compute_loss(self, y_true, y_pred):
        # 确保预测值y_pred不会是0或1，以避免log(0)或log(1)导致的问题
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        # 如果y_true和y_pred形状相同，则假设y_true是one-hot编码的多分类标签
        if y_true.shape == y_pred.shape:
            # 多分类（y_true为one-hot编码）
            loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        else:
            # 二分类或单标签多分类（y_true是类别索引）
            # 假设y_pred是每个类别的概率
            num_samples = y_true.shape[0]
            # 使用高级索引选择对应真实类别的预测概率的对数
            log_likelihood = -np.log(y_pred[range(num_samples), y_true])
            loss = np.sum(log_likelihood) / num_samples
        return loss

    # 计算交叉熵损失对预测值的梯度
    def backward(self, y_true, y_pred):
        # Softmax + 交叉熵的组合导数通常简化为 y_pred - y_true (当y_true是one-hot编码时)
        # 如果y_true是类别索引，则先将其转换为one-hot编码
        if y_true.shape != y_pred.shape:
            # 将y_true转换为one-hot编码
            num_samples = y_true.shape[0]
            num_classes = y_pred.shape[1]
            y_true_one_hot = np.zeros((num_samples, num_classes))
            y_true_one_hot[np.arange(num_samples), y_true] = 1
            y_true = y_true_one_hot

        return (y_pred - y_true) / y_true.shape[0]

# 根据名称获取损失函数实例的工厂函数
def get_loss_function(name):
    if name == 'mse':
        return MeanSquaredError()
    elif name == 'cross_entropy':
        return CrossEntropyLoss()
    else:
        raise ValueError(f"未知损失函数: {name}")