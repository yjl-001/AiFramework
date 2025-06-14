from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad
