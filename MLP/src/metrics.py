import numpy as np

# 计算混淆矩阵
def confusion_matrix(y_true, y_pred):
    # 如果y_true是one-hot编码，则转换为类别标签
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    # 如果y_pred是概率，则转换为类别标签
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # 确定类别数量
    num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    # 初始化混淆矩阵
    cm = np.zeros((num_classes, num_classes), dtype=int)
    # 填充混淆矩阵
    for i in range(len(y_true)):
        cm[y_true[i]][y_pred[i]] += 1
    return cm

# 计算准确率
def accuracy_score(y_true, y_pred):
    # 如果y_true是one-hot编码，则转换为类别标签
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    # 如果y_pred是概率，则转换为类别标签
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    # 计算并返回准确率
    return np.mean(y_true == y_pred)

# 计算精确率
def precision_score(y_true, y_pred, average='weighted'):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes)
    for i in range(num_classes):
        tp = cm[i, i]  # 真正例
        fp = np.sum(cm[:, i]) - tp  # 假正例
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0

    if average == 'weighted':
        # 加权平均精确率
        weights = np.sum(cm, axis=1) / np.sum(cm)
        return np.sum(precision * weights)
    elif average == 'macro':
        # 宏平均精确率
        return np.mean(precision)
    else:
        # 返回每个类别的精确率
        return precision

# 计算召回率
def recall_score(y_true, y_pred, average='weighted'):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    recall = np.zeros(num_classes)
    for i in range(num_classes):
        tp = cm[i, i]  # 真正例
        fn = np.sum(cm[i, :]) - tp  # 假反例
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0

    if average == 'weighted':
        # 加权平均召回率
        weights = np.sum(cm, axis=1) / np.sum(cm)
        return np.sum(recall * weights)
    elif average == 'macro':
        # 宏平均召回率
        return np.mean(recall)
    else:
        # 返回每个类别的召回率
        return recall

# 计算F1分数
def f1_score(y_true, y_pred, average='weighted'):
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = np.zeros_like(precision)
    for i in range(len(f1)):
        if (precision[i] + recall[i]) > 0:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    if average == 'weighted':
        # 加权平均F1分数
        cm = confusion_matrix(y_true, y_pred)
        weights = np.sum(cm, axis=1) / np.sum(cm)
        return np.sum(f1 * weights)
    elif average == 'macro':
        # 宏平均F1分数
        return np.mean(f1)
    else:
        # 返回每个类别的F1分数
        return f1