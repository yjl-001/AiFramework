import os
from mytorch.backend import xp
import csv


class MNISTDataset:
    def __init__(self, root, train=True, from_csv=True, use_cache=True):
        self.root = root
        self.train = train
        self.from_csv = from_csv
        self.use_cache = use_cache

        if self.from_csv:
            self._load_data()
        else:
            raise NotImplementedError("目前仅支持 CSV 格式")

    def _load_data(self):
        # 文件名
        csv_filename = "mnist_train.csv" if self.train else "mnist_test.csv"
        csv_path = os.path.join(self.root, csv_filename)

        # 缓存文件名
        cache_filename = "mnist_train.npz" if self.train else "mnist_test.npz"
        cache_path = os.path.join(self.root, cache_filename)

        # 如果启用缓存并且缓存文件存在，直接加载
        if self.use_cache and os.path.exists(cache_path):
            print(f"📦 正在从缓存加载数据: {cache_path}")
            data = xp.load(cache_path)
            self.images = data['images']
            self.labels = data['labels']
            print(f"✅ 加载完成: {len(self.labels)} 个样本")
            return

        # 否则从 CSV 加载
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到文件: {csv_path}")

        print(f"📂 正在从 CSV 加载数据: {csv_path}")
        images = []
        labels = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                label = int(row[0])
                pixels = xp.array(row[1:], dtype=xp.uint8).reshape(1, 28, 28)
                images.append(pixels)
                labels.append(label)

        self.images = xp.stack(images)
        self.labels = xp.array(labels)

        print(f"✅ 加载完成: {len(self.labels)} 个样本")

        # 保存为缓存文件
        if self.use_cache:
            print(f"💾 正在保存缓存文件: {cache_path}")
            xp.savez_compressed(
                cache_path, images=self.images, labels=self.labels)
            print("✅ 缓存保存完成")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
