# LSTM/src/activations.py
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def d_leaky_relu(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def d_elu(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True)) # Subtract max for numerical stability
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# Note: The derivative of softmax is more complex as it's a vector function.
# Typically, dL/dz = dL/da * da/dz, where a = softmax(z).
# If L is Cross-Entropy loss, then dL/dz = a - y (where y is one-hot true label).
# So, we usually don't compute d_softmax directly in isolation for backprop in that common case.

ACTIVATIONS = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'leaky_relu': leaky_relu,
    'elu': elu,
    'softmax': softmax
}

DERIVATIVES = {
    'sigmoid': d_sigmoid,
    'tanh': d_tanh,
    'relu': d_relu,
    'leaky_relu': d_leaky_relu,
    'elu': d_elu
    # 'softmax': d_softmax # Not typically used directly
}

def get_activation(name):
    func = ACTIVATIONS.get(name.lower())
    if func is None:
        raise ValueError(f"Activation function '{name}' not found. Available: {list(ACTIVATIONS.keys())}")
    return func

def get_derivative(name):
    func = DERIVATIVES.get(name.lower())
    if func is None:
        raise ValueError(f"Derivative for activation function '{name}' not found. Available: {list(DERIVATIVES.keys())}")
    return func

if __name__ == '__main__':
    x_test = np.array([-2., -1., 0., 1., 2.])
    print(f"x = {x_test}")

    print(f"sigmoid(x) = {sigmoid(x_test)}")
    print(f"d_sigmoid(x) = {d_sigmoid(x_test)}")

    print(f"tanh(x) = {tanh(x_test)}")
    print(f"d_tanh(x) = {d_tanh(x_test)}")

    print(f"relu(x) = {relu(x_test)}")
    print(f"d_relu(x) = {d_relu(x_test)}")

    print(f"leaky_relu(x) = {leaky_relu(x_test)}")
    print(f"d_leaky_relu(x) = {d_leaky_relu(x_test)}")

    print(f"elu(x) = {elu(x_test)}")
    print(f"d_elu(x) = {d_elu(x_test)}")

    x_softmax = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    print(f"softmax({x_softmax}) = \n{softmax(x_softmax)}")
    print(f"softmax({x_softmax}, axis=0) = \n{softmax(x_softmax, axis=0)}")

    try:
        get_activation('sigmoid')
        get_derivative('tanh')
        print("Activation getters OK.")
        get_activation('unknown')
    except ValueError as e:
        print(f"Caught expected error: {e}")