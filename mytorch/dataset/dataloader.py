import random
from .dataset import Dataset


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.indices):
            raise StopIteration

        batch_indices = self.indices[self.current_idx:
                                     self.current_idx + self.batch_size]
        batch_data = [self.dataset[i] for i in batch_indices]

        # 拆分数据和标签
        data_batch, label_batch = zip(*batch_data)

        self.current_idx += self.batch_size
        return data_batch, label_batch
