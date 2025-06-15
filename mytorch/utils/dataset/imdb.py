import os
import csv
from mytorch.backend import xp


class IMDBDataset:
    def __init__(self, root, tokenizer, max_len=200, train=True, from_csv=True, use_cache=True):
        self.root = root
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train = train
        self.from_csv = from_csv
        self.use_cache = use_cache

        if self.from_csv:
            self._load_data()
        else:
            raise NotImplementedError("目前仅支持 CSV 格式")

    def _load_data(self):
        # 文件名
        csv_filename = "imdb_train.csv" if self.train else "imdb_test.csv"
        csv_path = os.path.join(self.root, csv_filename)

        # 缓存文件名
        cache_filename = "imdb_train.npz" if self.train else "imdb_test.npz"
        cache_path = os.path.join(self.root, cache_filename)

        # 如果启用缓存并且缓存文件存在，直接加载
        if self.use_cache and os.path.exists(cache_path):
            print(f"📦 正在从缓存加载数据: {cache_path}")
            data = xp.load(cache_path, allow_pickle=True)
            self.input_ids = data['input_ids']
            self.labels = data['labels']
            print(f"✅ 加载完成: {len(self.labels)} 个样本")
            return

        # 否则从 CSV 加载
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到文件: {csv_path}")

        print(f"📂 正在从 CSV 加载数据: {csv_path}")
        input_ids = []
        labels = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                text = row[2]  # 第3列是 review
                label_str = row[3].strip().lower()  # 第4列是 label
                label = 1 if label_str == 'pos' else 0

                ids = self.tokenizer.encode(text, self.max_len)
                input_ids.append(xp.array(ids, dtype=xp.int32))
                labels.append(label)

        self.input_ids = xp.stack(input_ids)
        self.labels = xp.array(labels, dtype=xp.int32)

        print(f"✅ 加载完成: {len(self.labels)} 个样本")

        # 保存为缓存文件
        if self.use_cache:
            print(f"💾 正在保存缓存文件: {cache_path}")
            xp.savez_compressed(
                cache_path, input_ids=self.input_ids, labels=self.labels)
            print("✅ 缓存保存完成")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]
