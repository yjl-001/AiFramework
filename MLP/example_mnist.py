import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 解决matplotlib中文显示问题
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 指定默认字体
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 从 src 包中导入 MLPAPI 类以及数据准备和划分的辅助函数
from src.api import MLPAPI, prepare_data_for_classification

# 当脚本直接运行时执行以下代码
if __name__ == '__main__':
    # MNIST分类示例
    print("\n--- MNIST分类示例 ---")
    # 定义MNIST数据集路径
    # 注意：请确保数据集路径相对于此脚本是正确的，或者使用绝对路径
    mnist_train_path = 'dataset/MINIST/mnist_train.csv' # 假设脚本在MLP目录下运行
    mnist_test_path = 'dataset/MINIST/mnist_test.csv'   # 假设脚本在MLP目录下运行

    try:
        # 加载MNIST训练数据
        train_data = pd.read_csv(mnist_train_path).values
        X_train_mnist = train_data[:, 1:] / 255.0  # 归一化像素值到0-1范围
        y_train_mnist = train_data[:, 0]

        # 加载MNIST测试数据
        test_data = pd.read_csv(mnist_test_path).values
        X_test_mnist = test_data[:, 1:] / 255.0
        y_test_mnist = test_data[:, 0]

        num_classes_mnist = 10 # MNIST有10个类别 (0-9)

        # 数据准备: 将y转换为one-hot编码
        X_train_mnist, y_train_mnist_one_hot = prepare_data_for_classification(X_train_mnist, y_train_mnist, num_classes_mnist)
        X_test_mnist, y_test_mnist_one_hot = prepare_data_for_classification(X_test_mnist, y_test_mnist, num_classes_mnist)

        # 定义MLP模型结构
        # 输入层: 784个特征 (28x28像素)
        # 隐藏层: 128个神经元
        # 输出层: 10个类别
        mlp_mnist = MLPAPI(layers=[X_train_mnist.shape[1], 128, num_classes_mnist], activations=['relu', 'softmax'], weight_initializer='random_uniform')

        # 训练模型
        print("开始训练MNIST模型...")
        history_mnist = mlp_mnist.train(
            X_train_mnist, y_train_mnist_one_hot, epochs=50, learning_rate=0.01,
            loss_function='cross_entropy', optimizer='adam', batch_size=64,
            regularization='l2', reg_lambda=0.0001
        )
        print("MNIST模型训练完成。")

        # 评估模型
        metrics_mnist = mlp_mnist.evaluate(X_test_mnist, y_test_mnist, task_type='classification')
        print("MNIST分类指标:", metrics_mnist)

        # 绘制训练历史 (损失和准确率)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history_mnist['loss'], label='training loss')
        plt.title('MNIST Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history_mnist['accuracy'], label='training accuracy')
        plt.title('MNIST Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('accuracy')
        plt.legend()

        # 创建结果目录 (如果不存在)
        import os
        output_dir = 'result'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.savefig(os.path.join(output_dir, 'mnist_training_history.png'))
        print(f"MNIST训练历史图已保存到 {os.path.join(output_dir, 'mnist_training_history.png')}")
        # plt.show() # 在脚本中运行时，可以注释掉show，避免阻塞

        # 随机选择一些测试样本进行预测并可视化
        num_samples_to_show = 5
        random_indices = np.random.choice(len(X_test_mnist), num_samples_to_show, replace=False)
        sample_images = X_test_mnist[random_indices].reshape(-1, 28, 28)
        true_labels = y_test_mnist[random_indices]
        predicted_probs = mlp_mnist.predict(X_test_mnist[random_indices])
        predicted_labels = np.argmax(predicted_probs, axis=1)

        plt.figure(figsize=(10, 4))
        for i in range(num_samples_to_show):
            plt.subplot(1, num_samples_to_show, i + 1)
            plt.imshow(sample_images[i], cmap='gray')
            plt.title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}")
            plt.axis('off')
        plt.suptitle('MNIST Pred Result Examples')

        # 确保结果目录存在 (通常在前面已创建)
        output_dir = 'result'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir) # 以防万一

        plt.savefig(os.path.join(output_dir, 'mnist_predictions_example.png'))
        print(f"MNIST预测结果示例图已保存到 {os.path.join(output_dir, 'mnist_predictions_example.png')}")
        # plt.show()

        # 模型保存
        save_path_mnist = 'mnist_mlp_model.npy'
        mlp_mnist.save_model(os.path.join(output_dir, save_path_mnist))
        print(f"MNIST模型已保存到 {save_path_mnist}")

        # 模型加载示例 (可选, 用于验证)
        # loaded_mlp_mnist = MLPAPI.load_model(save_path_mnist)
        # print(f"MNIST模型已从 {save_path_mnist} 加载成功。")
        # sample_index = np.random.randint(0, len(X_test_mnist))
        # sample_image_loaded = X_test_mnist[sample_index].reshape(1, -1)
        # true_label_loaded = y_test_mnist[sample_index]
        # predicted_prob_loaded = loaded_mlp_mnist.predict(sample_image_loaded)
        # predicted_label_loaded = np.argmax(predicted_prob_loaded)
        # print(f"加载模型预测示例: 真实标签: {true_label_loaded}, 预测标签: {predicted_label_loaded}")

    except FileNotFoundError:
        print(f"错误: 未找到MNIST数据集文件。请确保文件位于 {mnist_train_path} 和 {mnist_test_path}，或者检查路径设置。")
    except Exception as e:
        print(f"处理MNIST数据集时发生错误: {e}")