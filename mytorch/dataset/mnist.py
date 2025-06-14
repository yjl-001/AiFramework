import os
import numpy as np
import csv


class MNISTDataset:
    def __init__(self, root, train=True, from_csv=True):
        self.root = root
        self.train = train
        self.from_csv = from_csv

        if self.from_csv:
            self._load_from_csv()
        else:
            raise NotImplementedError("目前仅支持 CSV 格式")

    def _load_from_csv(self):
        filename = "mnist_train.csv" if self.train else "mnist_test.csv"
        filepath = os.path.join(self.root, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"找不到文件: {filepath}")

        print(f"📂 正在加载 CSV 数据: {filepath}")
        images = []
        labels = []

        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                label = int(row[0])
                pixels = np.array(row[1:], dtype=np.uint8).reshape(1, 28, 28)
                images.append(pixels)
                labels.append(label)

        self.images = np.stack(images)
        self.labels = np.array(labels)
        print(f"✅ 加载完成: {len(self.labels)} 个样本")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
