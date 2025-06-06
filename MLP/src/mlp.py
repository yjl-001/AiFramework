import numpy as np
from .activations import get_activation
from .losses import get_loss_function
from .optimizers import get_optimizer
from .initializers import get_initializer
from .regularization import get_regularization
from .metrics import accuracy_score # 导入 accuracy_score

# 定义多层感知机 (MLP) 类
class MLP:
    def __init__(self, layers, activations, weight_initializer='zeros'):
        # layers: 一个列表，定义了每一层的神经元数量，例如 [784, 128, 10] 表示输入层784个神经元，隐藏层128个，输出层10个。
        self.layers = layers
        # activations: 一个列表，定义了每一层（除了输入层）使用的激活函数。
        # 通过 get_activation 函数获取激活函数对象。
        self.activations = [get_activation(act) for act in activations]
        # 存储网络的权重矩阵。
        self.weights = []
        # 存储网络的偏置向量。
        self.biases = []
        # 权重初始化器，默认为 'random_normal' (随机正态分布初始化)。
        self.weight_initializer = get_initializer(weight_initializer)
        # 调用内部方法初始化权重和偏置。
        self._initialize_weights()

    # 初始化权重和偏置的方法
    def _initialize_weights(self):
        # 遍历每一层（除了最后一层，因为权重连接到下一层）
        for i in range(len(self.layers) - 1):
            # 根据当前层和下一层的神经元数量初始化权重矩阵。
            # 使用 weight_initializer 对象进行初始化。
            weight = self.weight_initializer.initialize((self.layers[i], self.layers[i+1]))
            # 偏置通常初始化为零向量。
            bias = np.zeros((1, self.layers[i+1]))
            # 将初始化后的权重和偏置添加到列表中。
            self.weights.append(weight)
            self.biases.append(bias)

    # 前向传播方法
    def forward(self, X):
        # 存储每一层的输入（A_prev），第一个是网络的输入X。
        self.layer_inputs = [X]
        # 存储每一层激活函数之前的输出（Z值）。
        self.layer_outputs_pre_activation = []
        # 当前层的激活输出，初始为输入X。
        A = X
        # 遍历每一层进行前向计算。
        for i in range(len(self.layers) - 1):
            # 计算当前层的加权和 Z = A_prev * W + b。
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            # 存储Z值。
            self.layer_outputs_pre_activation.append(Z)
            # 应用激活函数，得到当前层的激活输出A。
            A = self.activations[i].forward(Z)
            # 存储当前层的激活输出A，作为下一层的输入。
            self.layer_inputs.append(A)
        # 返回最终层的输出。
        return A

    # 反向传播方法
    def backward(self, dL_dA):
        # 存储计算出的梯度。
        gradients = []
        # 初始梯度为损失函数对输出层激活的梯度。
        d_prev = dL_dA
        # 获取网络层数（不包括输入层）。
        num_layers = len(self.layers) - 1

        # 从最后一层开始反向遍历。
        for i in reversed(range(num_layers)):
            # 获取当前层的输入（即前一层的激活输出）。
            A_prev = self.layer_inputs[i]
            # 获取当前层激活函数之前的输出（Z值）。
            Z_curr = self.layer_outputs_pre_activation[i]
            # 获取当前层的激活函数对象。
            current_activation = self.activations[i]

            # 计算当前层Z的梯度 dZ = dL/dA * dA/dZ。
            dZ = current_activation.backward(Z_curr, d_prev)

            # 计算权重dW和偏置db的梯度。
            # dW = A_prev.T * dZ
            dW = np.dot(A_prev.T, dZ)
            # db = sum(dZ) (按行求和)
            db = np.sum(dZ, axis=0, keepdims=True)

            # 计算前一层激活的梯度 d_prev = dZ * W.T。
            d_prev = np.dot(dZ, self.weights[i].T)

            # 将当前层的dW和db插入到梯度列表的开头。
            gradients.insert(0, (dW, db))
        # 返回所有层的梯度。
        return gradients

    # 更新权重和偏置的方法
    def update_weights(self, gradients, learning_rate, optimizer):
        # 使用优化器更新权重和偏置。
        self.weights, self.biases = optimizer.update(self.weights, self.biases, gradients, learning_rate)

    # 训练模型的方法
    def train(self, X_train, y_train, epochs, learning_rate, loss_function_name, optimizer_name='sgd', batch_size=32, stop_criteria=None, regularization=None, reg_lambda=0.01, l1_ratio=None):
        # 获取损失函数对象。
        loss_fn = get_loss_function(loss_function_name)
        # 获取优化器对象。
        optimizer = get_optimizer(optimizer_name)
        # 初始化正则化函数为None。
        reg_fn = None
        # 如果指定了正则化，则获取正则化函数对象。
        if regularization:
            reg_fn = get_regularization(regularization, reg_lambda, l1_ratio)

        # 训练样本数量。
        num_samples = X_train.shape[0]
        # 存储训练历史，例如损失值。
        history = {'loss': [], 'accuracy': []} # 初始化 history 字典，添加 accuracy 列表

        # 遍历每个训练周期 (epoch)。
        for epoch in range(epochs):
            # 每个epoch开始时打乱数据。
            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            # 记录当前epoch的总损失。
            epoch_loss = 0
            # 记录当前epoch的预测和真实标签，用于计算准确率
            all_y_pred_epoch = []
            all_y_true_epoch = []

            # 按批次遍历数据。
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # 前向传播，得到预测值。
                y_pred = self.forward(X_batch)

                # 收集预测和真实标签
                all_y_pred_epoch.append(np.argmax(y_pred, axis=1))
                if y_batch.ndim > 1 and y_batch.shape[1] > 1: # 检查y_batch是否为one-hot编码
                    all_y_true_epoch.append(np.argmax(y_batch, axis=1))
                else:
                    all_y_true_epoch.append(y_batch)

                # 计算损失。
                loss = loss_fn.compute_loss(y_batch, y_pred)
                # 如果存在正则化，将正则化损失加到总损失中。
                if reg_fn:
                    loss += reg_fn.loss(self.weights)
                # 累加当前epoch的损失。
                epoch_loss += loss

                # 反向传播，计算损失对输出层激活的梯度。
                dL_dA = loss_fn.backward(y_batch, y_pred)
                # 计算所有层的梯度。
                gradients = self.backward(dL_dA)

                # 添加正则化梯度。
                if reg_fn:
                    reg_gradients = reg_fn.gradient(self.weights)
                    for j in range(len(gradients)):
                        # 将正则化梯度加到权重梯度上。
                        gradients[j] = (gradients[j][0] + reg_gradients[j], gradients[j][1])

                # 更新权重和偏置。
                self.update_weights(gradients, learning_rate, optimizer)

            # 计算当前epoch的平均损失。
            avg_epoch_loss = epoch_loss / (num_samples / batch_size)
            # 记录平均损失。
            history['loss'].append(avg_epoch_loss)

            # 计算并记录当前epoch的准确率
            y_pred_epoch_flat = np.concatenate(all_y_pred_epoch)
            y_true_epoch_flat = np.concatenate(all_y_true_epoch)
            epoch_accuracy = accuracy_score(y_true_epoch_flat, y_pred_epoch_flat)
            history['accuracy'].append(epoch_accuracy)

            # 打印当前epoch的损失和准确率信息。
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

            # 检查停止条件（基于损失的简单早停）。
            if stop_criteria and len(history['loss']) > stop_criteria['patience']:
                # 如果在耐心期内损失没有改善，则提前停止。
                if all(history['loss'][-stop_criteria['patience']:] > history['loss'][-stop_criteria['patience']-1]):
                    print(f"Early stopping at epoch {epoch+1} due to no improvement in loss.")
                    break
        # 返回训练历史。
        return history

    # 预测方法
    def predict(self, X):
        # 通过前向传播进行预测。
        return self.forward(X)