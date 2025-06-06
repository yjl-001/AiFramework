from .mlp import MLP
from .metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from .utils import to_one_hot, train_test_split
import numpy as np

# MLPAPI 类提供了一个统一的接口来训练、预测、评估、保存和加载MLP模型。
class MLPAPI:
    def __init__(self, layers, activations, weight_initializer='random_normal'):
        # 初始化MLP模型实例
        self.model = MLP(layers, activations, weight_initializer)

    def train(self, X_train, y_train, epochs=100, learning_rate=0.01, loss_function='cross_entropy', optimizer='adam', batch_size=32, stop_criteria=None, regularization=None, reg_lambda=0.01, l1_ratio=None):
        """
        训练MLP模型。

        参数:
            X_train (np.array): 训练特征。
            y_train (np.array): 训练标签。
            epochs (int): 训练轮数。
            learning_rate (float): 优化器的学习率。
            loss_function (str): 损失函数名称 ('mse', 'cross_entropy')。
            optimizer (str): 优化器名称 ('sgd', 'momentum', 'rmsprop', 'adam')。
            batch_size (int): mini-batch的大小。
            stop_criteria (dict): 提前停止的字典，例如 {'patience': 5}。
            regularization (str): 正则化类型 ('l1', 'l2', 'elastic_net')。
            reg_lambda (float): 正则化强度。
            l1_ratio (float): Elastic Net正则化的L1比例 (0到1之间)。

        返回:
            dict: 训练历史 (例如，每轮的损失)。
        """
        print("开始MLP训练...")
        history = self.model.train(
            X_train, y_train, epochs, learning_rate, loss_function,
            optimizer, batch_size, stop_criteria, regularization, reg_lambda, l1_ratio
        )
        print("MLP训练完成。")
        return history

    def predict(self, X):
        """
        使用训练好的MLP模型进行预测。

        参数:
            X (np.array): 用于预测的特征。

        返回:
            np.array: 预测输出 (分类的概率，回归的值)。
        """
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, task_type='classification'):
        """
        评估模型的性能。

        参数:
            X_test (np.array): 测试特征。
            y_test (np.array): 真实测试标签。
            task_type (str): 'classification' (分类) 或 'regression' (回归)。

        返回:
            dict: 评估指标。
        """
        y_pred = self.predict(X_test)

        metrics = {}
        if task_type == 'classification':
            # 假设y_pred是分类的概率
            # 转换为类别标签以计算指标
            y_pred_labels = np.argmax(y_pred, axis=1)
            if y_test.ndim > 1 and y_test.shape[1] > 1:
                y_test_labels = np.argmax(y_test, axis=1)
            else:
                y_test_labels = y_test

            metrics['accuracy'] = accuracy_score(y_test_labels, y_pred_labels)
            metrics['confusion_matrix'] = confusion_matrix(y_test_labels, y_pred_labels).tolist()
            metrics['precision'] = precision_score(y_test_labels, y_pred_labels)
            metrics['recall'] = recall_score(y_test_labels, y_pred_labels)
            metrics['f1_score'] = f1_score(y_test_labels, y_pred_labels)
        elif task_type == 'regression':
            # 对于回归，y_pred已经是预测值
            metrics['mse'] = np.mean((y_test - y_pred)**2)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = np.mean(np.abs(y_test - y_pred))
        else:
            raise ValueError("task_type必须是'classification'或'regression'")

        return metrics

    def save_model(self, path):
        """
        保存模型的权重和偏置。
        """
        model_state = {
            'layers': self.model.layers,
            'activations': [act.__class__.__name__.lower() for act in self.model.activations],
            'weights': [w.tolist() for w in self.model.weights],
            'biases': [b.tolist() for b in self.model.biases]
        }
        np.save(path, model_state)
        print(f"模型已保存到 {path}")

    @classmethod
    def load_model(cls, path):
        """
        从保存的状态加载模型。
        """
        model_state = np.load(path, allow_pickle=True).item()
        # 从名称重建激活函数
        activations = [act_name for act_name in model_state['activations']]
        instance = cls(model_state['layers'], activations)
        instance.model.weights = [np.array(w) for w in model_state['weights']]
        instance.model.biases = [np.array(b) for b in model_state['biases']]
        print(f"模型已从 {path} 加载")
        return instance

# 数据准备的辅助函数 (如果需要，可以移动到utils.py中)
def prepare_data_for_classification(X, y, num_classes=None):
    # 确保y是类别索引的一维数组
    if y.ndim > 1 and y.shape[1] > 1:
        y = np.argmax(y, axis=1)
    
    # 如果损失函数需要 (例如，CrossEntropyLoss的backward期望one-hot)，将y转换为one-hot编码。
    # 然而，我们当前的CrossEntropyLoss在backward中可以处理y_true的one-hot和整数标签。
    # 为了与典型的分类输出 (softmax) 保持一致，最好将y_train设为one-hot。
    if num_classes is None:
        num_classes = np.max(y) + 1
    y_one_hot = to_one_hot(y, num_classes)
    return X, y_one_hot

def prepare_data_for_regression(X, y):
    # 对于回归，确保y是二维的 (样本数, 输出数)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return X, y


# 示例用法 (可以放在单独的example.py或test.py中)
# if __name__ == '__main__':
#     # 1. 分类示例 (例如，XOR问题或简单的合成数据)
#     print("\n--- 分类示例 ---")
#     X_clf = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
#     y_clf = np.array([0, 1, 1, 0], dtype=np.int32) # XOR标签
#     num_classes_clf = 2

#     # 准备数据: 将y转换为one-hot
#     X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.25, random_state=42)
#     X_clf_train, y_clf_train_one_hot = prepare_data_for_classification(X_clf_train, y_clf_train, num_classes_clf)
#     X_clf_test, y_clf_test_one_hot = prepare_data_for_classification(X_clf_test, y_clf_test, num_classes_clf)

#     # 定义分类的MLP结构
#     # 输入: 2个特征, 隐藏层: 4个神经元, 输出: 2个类别
#     mlp_clf = MLPAPI(layers=[X_clf.shape[1], 4, num_classes_clf], activations=['relu', 'softmax'], weight_initializer='he_normal')

#     # 训练模型
#     history_clf = mlp_clf.train(
#         X_clf_train, y_clf_train_one_hot, epochs=1000, learning_rate=0.1,
#         loss_function='cross_entropy', optimizer='adam', batch_size=2,
#         regularization='l2', reg_lambda=0.001
#     )

#     # 评估模型
#     metrics_clf = mlp_clf.evaluate(X_clf_test, y_clf_test, task_type='classification')
#     print("分类指标:", metrics_clf)

#     # 进行预测
#     predictions_clf = mlp_clf.predict(X_clf)
#     print("预测 (概率):\n", predictions_clf)
#     print("预测类别:\n", np.argmax(predictions_clf, axis=1))

#     # 2. 回归示例 (简单的线性关系)
#     print("\n--- 回归示例 ---")
#     X_reg = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
#     y_reg = np.array([[2], [4], [6], [8], [10]], dtype=np.float32) # y = 2x

#     # 准备回归数据
#     X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
#     X_reg_train, y_reg_train = prepare_data_for_regression(X_reg_train, y_reg_train)
#     X_reg_test, y_reg_test = prepare_data_for_regression(X_reg_test, y_reg_test)

#     # 定义回归的MLP结构
#     # 输入: 1个特征, 隐藏层: 5个神经元, 输出: 1个值
#     mlp_reg = MLPAPI(layers=[X_reg.shape[1], 5, y_reg.shape[1]], activations=['relu', 'linear'], weight_initializer='glorot_uniform')

#     # 训练模型
#     history_reg = mlp_reg.train(
#         X_reg_train, y_reg_train, epochs=500, learning_rate=0.01,
#         loss_function='mse', optimizer='adam', batch_size=1
#     )

#     # 评估模型
#     metrics_reg = mlp_reg.evaluate(X_reg_test, y_reg_test, task_type='regression')
#     print("回归指标:", metrics_reg)

#     # 进行预测
#     predictions_reg = mlp_reg.predict(X_reg)
#     print("预测:\n", predictions_reg)

#     # 3. 模型保存/加载示例
#     print("\n--- 模型保存/加载示例 ---")
#     save_path = 'my_mlp_model.npy'
#     mlp_clf.save_model(save_path)

#     loaded_mlp = MLPAPI.load_model(save_path)
#     loaded_predictions = loaded_mlp.predict(X_clf)
#     print("Predictions from loaded model (probabilities):\n", loaded_predictions)
#     print("Predicted classes from loaded model:\n", np.argmax(loaded_predictions, axis=1))