from .module import Module
from mytorch.ops.dropout import *


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        return DropoutFunction.apply(x, self.p, self.training)
