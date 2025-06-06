# 多层感知机 (MLP) 实现文档

这是一个从零开始实现的简单多层感知机 (MLP) 库，旨在提供一个清晰、模块化的深度学习基础组件。该库包含了构建、训练和评估 MLP 模型所需的核心功能，并支持分类和回归任务。

## 核心组件

### 1. `mlp.py`

- **`MLP` 类**: 定义了多层感知机模型的核心结构。它允许用户指定网络层数、每层的神经元数量、激活函数、权重初始化方法、优化器、损失函数以及正则化方法。
- **前向传播**: 计算给定输入通过网络后的输出。
- **反向传播**: 根据损失函数的梯度计算网络中所有权重和偏置的梯度。
- **参数更新**: 使用选定的优化器更新模型的权重和偏置。

### 2. `activations.py`

包含了常用的激活函数及其导数，例如：
- **ReLU (Rectified Linear Unit)**: `relu`, `relu_derivative`
- **Sigmoid**: `sigmoid`, `sigmoid_derivative`
- **Tanh (Hyperbolic Tangent)**: `tanh`, `tanh_derivative`
- **Softmax**: `softmax` (通常用于输出层，不直接用于隐藏层反向传播)

### 3. `losses.py`

定义了用于训练模型的损失函数及其导数，例如：
- **均方误差 (Mean Squared Error, MSE)**: `mse`, `mse_derivative` (常用于回归任务)
- **交叉熵损失 (Cross-Entropy Loss)**: `cross_entropy`, `cross_entropy_derivative` (常用于分类任务)

### 4. `optimizers.py`

实现了多种优化算法，用于更新模型参数：
- **随机梯度下降 (Stochastic Gradient Descent, SGD)**: `SGD` 类
- **Adam**: `Adam` 类

### 5. `initializers.py`

提供了不同的权重初始化策略，以帮助模型更好地收敛：
- **零初始化**: `zeros_init`
- **随机初始化**: `random_init`
- **Xavier/Glorot 初始化**: `xavier_init`
- **He 初始化**: `he_init`

### 6. `regularization.py`

包含了正则化技术，用于防止过拟合：
- **L1 正则化**: `l1_regularization`, `l1_regularization_derivative`
- **L2 正则化**: `l2_regularization`, `l2_regularization_derivative`

### 7. `metrics.py`

提供了评估模型性能的指标：
- **混淆矩阵 (Confusion Matrix)**: `confusion_matrix`
- **准确率 (Accuracy Score)**: `accuracy_score`
- **精确率 (Precision Score)**: `precision_score`
- **召回率 (Recall Score)**: `recall_score`
- **F1 分数 (F1 Score)**: `f1_score`

### 8. `utils.py`

包含了一些实用工具函数：
- **独热编码 (One-Hot Encoding)**: `to_one_hot`
- **训练集/测试集划分**: `train_test_split`
- **批量数据生成**: `get_batches`

### 9. `api.py`

- **`MLPAPI` 类**: 提供了一个高级 API 接口，简化了 MLP 模型的训练、预测、评估、保存和加载过程。它封装了底层 MLP 类的复杂性，使得用户可以更便捷地使用模型。

### 10. `__init__.py`

- 作为 Python 包的初始化文件，它定义了 `MLP/src` 目录下的模块结构，并使用 `__all__` 列表导出了核心类和函数，方便用户直接从 `MLP.src` 导入。

## 设计原则

- **模块化**: 每个核心功能（如激活函数、损失函数、优化器）都被封装在独立的模块中，提高了代码的可读性和可维护性。
- **可扩展性**: 易于添加新的激活函数、损失函数、优化器或正则化方法。
- **灵活性**: 用户可以根据需求自定义网络结构、选择不同的组件。
- **Numpy 实现**: 核心计算基于 NumPy，避免了对大型深度学习框架的依赖，有助于理解底层机制。

## 训练与评估结果

### 1. MNIST 手写数字分类

- **数据集**: MNIST 手写数字数据集 (60,000 训练样本, 10,000 测试样本, 10 个类别)。
- **模型配置**: 
    - 输入层: 784 (28x28 像素)
    - 隐藏层: 256 个神经元 (ReLU 激活)
    - 输出层: 10 个神经元 (Softmax 激活)
    - 损失函数: 交叉熵
    - 优化器: Adam
    - 训练轮次: 50
    - 批量大小: 64
- **结果**: 
    - 训练损失和准确率随训练轮次的变化曲线已保存为 `mnist_training_history.png`。
    - 随机选择的 MNIST 预测示例已保存为 `mnist_predictions_example.png`。
    - 模型在测试集上的准确率约为 97%。
    - 混淆矩阵展示了模型在各个数字类别上的分类性能。

### 2. California Housing 回归

- **数据集**: California Housing 数据集 (20,640 样本，8 个特征，预测房价中位数)。
- **模型配置**: 
    - 输入层: 8 个特征
    - 隐藏层: 64 个神经元 (ReLU 激活)
    - 输出层: 1 个神经元 (无激活函数)
    - 损失函数: 均方误差 (MSE)
    - 优化器: Adam
    - 训练轮次: 100
    - 批量大小: 32
- **结果**: 
    - 训练损失随训练轮次的变化曲线已保存为 `california_training_history.png`。
    - 真实值与预测值的散点图已保存为 `california_predictions_scatter.png`，显示了模型对房价的预测能力。
    - 模型在测试集上的 MSE 较低，表明回归效果良好。

## 模型保存与加载

- `MLPAPI` 提供了 `save_model` 和 `load_model` 方法，可以将训练好的模型参数保存到 `.pkl` 文件中，并在需要时重新加载，方便模型的部署和复用。

## 关键设计与实现

- **NumPy 向量化操作**: 尽可能利用 NumPy 的向量化特性进行矩阵运算，提高了计算效率。
- **梯度计算**: 严格按照链式法则实现反向传播，确保梯度的正确性。
- **模块化接口**: 通过 `MLPAPI` 提供了统一且易于使用的接口，将底层实现细节封装起来。
- **数据预处理**: 示例中包含了对 MNIST 数据的归一化和独热编码，以及对 California Housing 数据的标准化处理，强调了数据预处理的重要性。
- **可视化**: 集成了 `matplotlib` 用于训练过程的可视化和结果展示，帮助用户直观理解模型性能。

## 如何运行示例

1. **安装依赖**: 确保已安装 `numpy`, `scikit-learn`, `matplotlib`。
   ```bash
   pip install numpy scikit-learn matplotlib
   ```
2. **下载数据集**: 
   - MNIST 数据集：请手动下载并放置到 `MLP/dataset/MINIST` 目录下。
   - California Housing 数据集：`scikit-learn` 会自动下载。
3. **运行 `example.py`**: 
   ```bash
   python MLP/example.py
   ```
   运行后，训练历史图和预测结果图将保存在项目根目录。

## 总结

这个 MLP 实现不仅提供了一个功能完备的神经网络模型，更重要的是，它通过清晰的模块划分和详细的文档，帮助用户深入理解多层感知机的工作原理和实现细节。通过在 MNIST 和 California Housing 数据集上的实践，展示了其在分类和回归任务中的有效性。