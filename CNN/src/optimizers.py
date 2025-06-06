import numpy as np

class Optimizer:
    def update(self, params, grads):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.learning_rate * grads[i]
        return params

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = []

    def update(self, params, grads):
        if not self.velocities:
            self.velocities = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            self.velocities[i] = self.momentum * self.velocities[i] + self.learning_rate * grads[i]
            params[i] -= self.velocities[i]
        return params

class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.001, decay_rate=0.99, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = []

    def update(self, params, grads):
        if not self.cache:
            self.cache = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            self.cache[i] = self.decay_rate * self.cache[i] + (1 - self.decay_rate) * (grads[i]**2)
            params[i] -= self.learning_rate * grads[i] / (np.sqrt(self.cache[i]) + self.epsilon)
        return params

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []
        self.v = []
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        if not self.m:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i]**2)

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            params[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params