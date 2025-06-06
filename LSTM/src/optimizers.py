# LSTM/src/optimizers.py
import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        raise NotImplementedError

    def set_learning_rate(self, lr):
        self.learning_rate = lr

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = None # Will be initialized on first update

    def update(self, params, grads):
        if params is None or grads is None:
            return
        
        if not isinstance(params, list):
            params = [params]
            grads = [grads]

        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            if param is None or grad is None: # Skip if a param/grad is None (e.g. for non-trainable parts)
                continue
            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] - self.learning_rate * grad
                param += self.velocities[i]
            else:
                param -= self.learning_rate * grad

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0     # Timestep

    def update(self, params, grads):
        if params is None or grads is None:
            return
            
        if not isinstance(params, list):
            params = [params]
            grads = [grads]

        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1
        for i, (param, grad) in enumerate(zip(params, grads)):
            if param is None or grad is None:
                continue
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-7):
        super().__init__(learning_rate)
        self.rho = rho # Decay rate
        self.epsilon = epsilon
        self.squared_gradients = None # Accumulated squared gradients

    def update(self, params, grads):
        if params is None or grads is None:
            return

        if not isinstance(params, list):
            params = [params]
            grads = [grads]

        if self.squared_gradients is None:
            self.squared_gradients = [np.zeros_like(p) for p in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            if param is None or grad is None:
                continue
            
            # Update accumulated squared gradients
            self.squared_gradients[i] = self.rho * self.squared_gradients[i] + (1 - self.rho) * (grad ** 2)
            
            # Update parameters
            param -= self.learning_rate * grad / (np.sqrt(self.squared_gradients[i]) + self.epsilon)

class Adagrad(Optimizer):
    def __init__(self, learning_rate=0.01, epsilon=1e-7):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.accumulated_squared_gradients = None

    def update(self, params, grads):
        if params is None or grads is None:
            return

        if not isinstance(params, list):
            params = [params]
            grads = [grads]

        if self.accumulated_squared_gradients is None:
            self.accumulated_squared_gradients = [np.zeros_like(p) for p in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            if param is None or grad is None:
                continue
            
            self.accumulated_squared_gradients[i] += grad ** 2
            param -= self.learning_rate * grad / (np.sqrt(self.accumulated_squared_gradients[i]) + self.epsilon)


OPTIMIZERS = {
    'sgd': SGD,
    'adam': Adam,
    'rmsprop': RMSprop,
    'adagrad': Adagrad,
}

def get_optimizer(name, **kwargs):
    optimizer_class = OPTIMIZERS.get(name.lower())
    if optimizer_class is None:
        raise ValueError(f"Optimizer '{name}' not found. Available: {list(OPTIMIZERS.keys())}")
    return optimizer_class(**kwargs)

if __name__ == '__main__':
    # Example Usage
    params_list = [
        np.random.randn(10, 5),
        np.random.randn(5),
        np.random.randn(5, 3),
        np.random.randn(3)
    ]
    # Make copies for each optimizer to test independently
    params_sgd = [p.copy() for p in params_list]
    params_adam = [p.copy() for p in params_list]
    params_rmsprop = [p.copy() for p in params_list]
    params_adagrad = [p.copy() for p in params_list]

    grads_list = [
        np.random.randn(10, 5) * 0.1,
        np.random.randn(5) * 0.1,
        np.random.randn(5, 3) * 0.1,
        np.random.randn(3) * 0.1
    ]

    print("Initial params (first layer, first 2x2):")
    print(params_list[0][:2, :2])

    # Test SGD
    sgd_opt = SGD(learning_rate=0.1, momentum=0.9)
    sgd_opt.update(params_sgd, grads_list)
    print("\nParams after SGD (first layer, first 2x2):")
    print(params_sgd[0][:2, :2])
    # Second update to see momentum effect
    sgd_opt.update(params_sgd, [g*0.5 for g in grads_list]) # Smaller grads for 2nd step
    print("Params after 2nd SGD step (first layer, first 2x2):")
    print(params_sgd[0][:2, :2])

    # Test Adam
    adam_opt = Adam(learning_rate=0.01)
    for _ in range(5): # Adam needs a few steps for t to increase
        adam_opt.update(params_adam, grads_list)
    print("\nParams after Adam (5 steps) (first layer, first 2x2):")
    print(params_adam[0][:2, :2])

    # Test RMSprop
    rmsprop_opt = RMSprop(learning_rate=0.01)
    rmsprop_opt.update(params_rmsprop, grads_list)
    print("\nParams after RMSprop (first layer, first 2x2):")
    print(params_rmsprop[0][:2, :2])

    # Test Adagrad
    adagrad_opt = Adagrad(learning_rate=0.1) # Adagrad often needs higher LR initially
    adagrad_opt.update(params_adagrad, grads_list)
    print("\nParams after Adagrad (first layer, first 2x2):")
    print(params_adagrad[0][:2, :2])
    adagrad_opt.update(params_adagrad, grads_list) # Second update
    print("Params after 2nd Adagrad step (first layer, first 2x2):")
    print(params_adagrad[0][:2, :2])

    try:
        get_optimizer('sgd', learning_rate=0.01)
        print("\nOptimizer getter OK.")
        get_optimizer('unknown')
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Test with single param/grad (not lists)
    print("\nTesting with single param/grad:")
    param_single = np.random.randn(3,2)
    grad_single = np.random.randn(3,2) * 0.1
    param_single_copy = param_single.copy()
    print("Initial single param:\n", param_single_copy)
    sgd_single_opt = SGD(learning_rate=0.1)
    sgd_single_opt.update(param_single_copy, grad_single)
    print("Single param after SGD:\n", param_single_copy)

    # Test with None params/grads in list (should be skipped)
    print("\nTesting with None params/grads in list:")
    params_with_none = [params_list[0].copy(), None, params_list[2].copy()]
    grads_with_none = [grads_list[0].copy(), None, grads_list[2].copy()]
    params_with_none_orig = [p.copy() if p is not None else None for p in params_with_none]
    
    sgd_none_opt = SGD(learning_rate=0.1)
    sgd_none_opt.update(params_with_none, grads_with_none)
    
    print("Param 0 after SGD (should change):\n", params_with_none[0][:2,:2])
    if params_with_none[1] is None:
        print("Param 1 is None (should be unchanged)")
    print("Param 2 after SGD (should change):\n", params_with_none[2][:2,:2])
    
    assert not np.array_equal(params_with_none_orig[0], params_with_none[0]), "Param 0 should have changed"
    assert params_with_none_orig[1] is None and params_with_none[1] is None, "Param 1 should remain None"
    assert not np.array_equal(params_with_none_orig[2], params_with_none[2]), "Param 2 should have changed"
    print("None param/grad handling OK.")