from ..tensor import Tensor

import numpy as np


def cross_entropy_loss(logits: Tensor, targets):
    log_probs = logits.log_softmax(dim=1)
    batch_size = targets.shape[0]
    loss = -log_probs[np.arange(batch_size), targets].mean()
    return loss
