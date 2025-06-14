import matplotlib.pyplot as plt
from mytorch.dataset import MNISTDataset
from mytorch.backend import xp


def check_dataset(dataset, num_samples=10):
    print("🔍 正在检查数据集...")

    # 检查长度
    length = len(dataset)
    print(f"✅ 样本总数: {length}")

    # 检查第一个样本
    try:
        image, label = dataset[0]
        assert isinstance(image, np.ndarray), "图像不是 NumPy 数组"
        assert isinstance(label, (int, np.integer)), "标签不是整数"
        assert image.shape == (1, 28, 28), f"图像 shape 错误: {image.shape}"
        assert 0 <= label <= 9, f"标签超出范围: {label}"
        print("✅ 第一个样本格式正确")
    except Exception as e:
        print("❌ 样本格式错误:", e)
        return

    # 可视化前 num_samples 个样本
    print(f"🖼️ 正在可视化前 {num_samples} 个样本...")
    plt.figure(figsize=(10, 4))  # 可选：调整图像大小

    for i in range(10):
        image, label = dataset[i]
        plt.subplot(2, 5, i + 1)  # 两行五列
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(str(label))
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("✅ 数据集检查完成")


if __name__ == "__main__":
    dataset = MNISTDataset(root="./data/mnist", train=True, from_csv=True)
    check_dataset(dataset, num_samples=10)
