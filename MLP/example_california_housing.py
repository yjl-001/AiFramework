import numpy as np
import matplotlib.pyplot as plt
# 解决matplotlib中文显示问题
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 指定默认字体
# plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
from sklearn.datasets import fetch_california_housing

# 从 src 包中导入 MLPAPI 类以及数据准备和划分的辅助函数
from src.api import MLPAPI, train_test_split # prepare_data_for_regression 通常不需要，因为y已经是数值

# 当脚本直接运行时执行以下代码
if __name__ == '__main__':
    # 加州房价回归示例
    print("\n--- 加州房价回归示例 ---")
    try:
        # 加载加州房价数据集
        california = fetch_california_housing()
        X_california = california.data.astype(np.float32)
        y_california = california.target.astype(np.float32).reshape(-1, 1)

        # 数据准备: 划分训练集和测试集
        X_train_california, X_test_california, y_train_california, y_test_california = train_test_split(X_california, y_california, test_size=0.2, random_state=42)

        # 定义MLP模型结构
        # 输入层: 特征数量 (由数据集决定)
        # 隐藏层: 64个神经元
        # 输出层: 1个值 (房价)
        mlp_california = MLPAPI(layers=[X_train_california.shape[1], 64, y_train_california.shape[1]], activations=['relu', 'linear'], weight_initializer='glorot_uniform')

        # 训练模型
        print("开始训练加州房价模型...")
        history_california = mlp_california.train(
            X_train_california, y_train_california, epochs=200, learning_rate=0.001,
            loss_function='mse', optimizer='adam', batch_size=32,
            regularization='l2', reg_lambda=0.001
        )
        print("加州房价模型训练完成。")

        # 评估模型
        metrics_california = mlp_california.evaluate(X_test_california, y_test_california, task_type='regression')
        print("加州房价回归指标:", metrics_california)

        # 绘制训练历史 (损失)
        plt.figure(figsize=(6, 5))
        plt.plot(history_california['loss'], label='training loss')
        plt.title('House Price Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # 创建结果目录 (如果不存在)
        import os
        output_dir = 'result'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.savefig(os.path.join(output_dir, 'california_training_history.png'))
        print(f"加州房价训练历史图已保存到 {os.path.join(output_dir, 'california_training_history.png')}")
        # plt.show()

        # 预测并绘制真实值与预测值的散点图
        predictions_california = mlp_california.predict(X_test_california)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_california, predictions_california, alpha=0.7)
        plt.xlabel("True")
        plt.ylabel("Pred")
        plt.title("House Price: True vs. Pred")
        plt.plot([y_test_california.min(), y_test_california.max()], [y_test_california.min(), y_test_california.max()], 'k--', lw=2) # 对角线
        plt.grid(True)

        # 确保结果目录存在 (通常在前面已创建)
        output_dir = 'result'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir) # 以防万一

        plt.savefig(os.path.join(output_dir, 'california_predictions_scatter.png'))
        print(f"加州房价预测散点图已保存到 {os.path.join(output_dir, 'california_predictions_scatter.png')}")
        # plt.show()

        # 模型保存
        save_path_california = 'california_mlp_model.npy'
        mlp_california.save_model(os.path.join(output_dir, save_path_california))
        print(f"加州房价模型已保存到 {save_path_california}")

        # 模型加载示例 (可选, 用于验证)
        # loaded_mlp_california = MLPAPI.load_model(save_path_california)
        # print(f"加州房价模型已从 {save_path_california} 加载成功。")
        # sample_idx_cal = np.random.randint(0, len(X_test_california))
        # sample_data_cal = X_test_california[sample_idx_cal].reshape(1, -1)
        # true_value_cal = y_test_california[sample_idx_cal]
        # predicted_value_cal = loaded_mlp_california.predict(sample_data_cal)
        # print(f"加载模型预测示例: 真实值: {true_value_cal[0]:.2f}, 预测值: {predicted_value_cal[0][0]:.2f}")

    except ImportError:
        print("错误: 未安装scikit-learn库。请运行 'pip install scikit-learn matplotlib' 安装。")
    except Exception as e:
        print(f"处理加州房价数据集时发生错误: {e}")