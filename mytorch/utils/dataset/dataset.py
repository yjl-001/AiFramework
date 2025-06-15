from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    @abstractmethod
    def __getitem__(self, index):
        """返回第 index 个样本"""
        pass

    @abstractmethod
    def __len__(self):
        """返回数据集大小"""
        pass
