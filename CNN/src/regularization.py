import numpy as np

class Regularizer:
    def __call__(self, weights):
        raise NotImplementedError

    def gradient(self, weights):
        raise NotImplementedError

class L1(Regularizer):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, weights):
        return self.alpha * np.sum(np.abs(weights))

    def gradient(self, weights):
        return self.alpha * np.sign(weights)

class L2(Regularizer):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, weights):
        return self.alpha * np.sum(np.square(weights))

    def gradient(self, weights):
        return 2 * self.alpha * weights

class ElasticNet(Regularizer):
    def __init__(self, alpha, l1_ratio):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.l1_regularizer = L1(alpha * l1_ratio)
        self.l2_regularizer = L2(alpha * (1 - l1_ratio))

    def __call__(self, weights):
        return self.l1_regularizer(weights) + self.l2_regularizer(weights)

    def gradient(self, weights):
        return self.l1_regularizer.gradient(weights) + self.l2_regularizer.gradient(weights)