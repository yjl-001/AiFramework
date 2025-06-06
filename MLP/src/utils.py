import numpy as np

# 将类别标签转换为one-hot编码
def to_one_hot(y, num_classes=None):
    # 如果未指定类别数量，则根据y的最大值自动确定
    if num_classes is None:
        num_classes = np.max(y) + 1
    # 使用np.eye创建one-hot编码矩阵
    return np.eye(num_classes)[y]

# 划分训练集和测试集
def train_test_split(X, y, test_size=0.2, random_state=None):
    # 设置随机种子以保证结果可复现
    if random_state:
        np.random.seed(random_state)
    num_samples = X.shape[0]
    # 随机打乱样本索引
    indices = np.random.permutation(num_samples)
    # 计算划分点
    split_idx = int(num_samples * (1 - test_size))
    # 划分训练集和测试集的索引
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    # 返回划分后的数据集
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# 生成数据批次
def get_batches(X, y, batch_size):
    num_samples = X.shape[0]
    # 遍历数据，按批次返回X和y
    for i in range(0, num_samples, batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

# 简单的数据加载函数示例 (例如，用于CSV文件)
# import pandas as pd
# def load_csv_data(file_path, target_column):
#     data = pd.read_csv(file_path)
#     y = data[target_column].values
#     X = data.drop(columns=[target_column]).values
#     return X, y