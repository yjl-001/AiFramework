# LSTM/src/losses.py
import numpy as np

class Loss:
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    def gradient(self, y_true, y_pred):
        raise NotImplementedError

    def __call__(self, y_true, y_pred):
        return self.loss(y_true, y_pred)

class MeanSquaredError(Loss):
    def loss(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def gradient(self, y_true, y_pred):
        # dL/dy_pred = 2 * (y_pred - y_true) / N
        # The factor of 2 is sometimes omitted or absorbed into learning rate.
        # Here, using the standard derivative.
        return 2 * (y_pred - y_true) / y_true.shape[0] # Assuming y_true.shape[0] is batch size

class MeanAbsoluteError(Loss):
    def loss(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def gradient(self, y_true, y_pred):
        return np.sign(y_pred - y_true) / y_true.shape[0]

class CategoricalCrossentropy(Loss):
    """
    Categorical Cross-Entropy Loss.
    Assumes y_true is one-hot encoded and y_pred contains probabilities (e.g., from softmax).
    """
    def loss(self, y_true, y_pred):
        # Clip predictions to prevent log(0) errors
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0] # Average over batch

    def gradient(self, y_true, y_pred):
        """
        Computes dL/dy_pred.
        If y_pred is the output of a softmax layer, and L is CCE loss,
        then the gradient dL/dz (where z is input to softmax) is simply y_pred - y_true.
        This gradient here is dL/dy_pred = -y_true / y_pred.
        It's often combined with softmax in backprop.
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        # Gradient is -y_true / y_pred, averaged over batch for consistency if loss is averaged.
        # However, if this gradient is fed directly to a softmax layer's input (z),
        # the combined gradient dL/dz = y_pred - y_true is simpler.
        # For a standalone loss function, dL/dy_pred is appropriate.
        return -y_true / y_pred / y_true.shape[0]

class BinaryCrossentropy(Loss):
    """
    Binary Cross-Entropy Loss.
    Assumes y_true contains binary labels (0 or 1) and y_pred contains probabilities (e.g., from sigmoid).
    """
    def loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def gradient(self, y_true, y_pred):
        """
        Computes dL/dy_pred.
        If y_pred is output of sigmoid, dL/dz = y_pred - y_true (where z is input to sigmoid).
        This gradient is dL/dy_pred = -(y_true/y_pred - (1-y_true)/(1-y_pred)).
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        # Averaged over batch size if loss is averaged.
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]

LOSSES = {
    'mse': MeanSquaredError,
    'mean_squared_error': MeanSquaredError,
    'mae': MeanAbsoluteError,
    'mean_absolute_error': MeanAbsoluteError,
    'categorical_crossentropy': CategoricalCrossentropy,
    'binary_crossentropy': BinaryCrossentropy,
}

def get_loss(name):
    loss_class = LOSSES.get(name.lower())
    if loss_class is None:
        raise ValueError(f"Loss function '{name}' not found. Available: {list(LOSSES.keys())}")
    return loss_class()

if __name__ == '__main__':
    # Test MSE
    y_true_mse = np.array([[1, 2], [3, 4]])
    y_pred_mse = np.array([[1.5, 2.5], [2.5, 3.5]])
    mse = MeanSquaredError()
    print(f"MSE Loss: {mse.loss(y_true_mse, y_pred_mse)}") # Expected: ((0.5^2 + 0.5^2) + (0.5^2 + 0.5^2))/2 = (0.25+0.25+0.25+0.25)/2 = 0.5
                                                        # Or if mean over all elements: 0.25
                                                        # np.mean averages over all elements. (0.25*4)/4 = 0.25
    print(f"MSE Gradient: \n{mse.gradient(y_true_mse, y_pred_mse)}")
    # Expected grad: 2 * (y_pred - y_true) / N. N=batch_size=2
    # 2 * [[0.5, 0.5], [-0.5, -0.5]] / 2 = [[0.5, 0.5], [-0.5, -0.5]]

    # Test MAE
    mae = MeanAbsoluteError()
    print(f"MAE Loss: {mae.loss(y_true_mse, y_pred_mse)}") # Expected: (0.5+0.5+0.5+0.5)/4 = 0.5
    print(f"MAE Gradient: \n{mae.gradient(y_true_mse, y_pred_mse)}")
    # Expected grad: sign(y_pred - y_true) / N. N=batch_size=2
    # [[1,1],[-1,-1]] / 2 = [[0.5,0.5],[-0.5,-0.5]]

    # Test Categorical Crossentropy
    y_true_cce = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) # Batch size 3, 3 classes
    y_pred_cce = np.array([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.2, 0.6, 0.2]]) # Softmax output
    cce = CategoricalCrossentropy()
    print(f"CCE Loss: {cce.loss(y_true_cce, y_pred_cce)}")
    # Expected: (-log(0.7) -log(0.8) -log(0.6)) / 3
    print(f"CCE Gradient: \n{cce.gradient(y_true_cce, y_pred_cce)}")
    # Expected: -y_true / y_pred / N

    # Test Binary Crossentropy
    y_true_bce = np.array([[1], [0], [1], [0]]) # Batch size 4, 1 output (sigmoid)
    y_pred_bce = np.array([[0.9], [0.2], [0.8], [0.1]])
    bce = BinaryCrossentropy()
    print(f"BCE Loss: {bce.loss(y_true_bce, y_pred_bce)}")
    print(f"BCE Gradient: \n{bce.gradient(y_true_bce, y_pred_bce)}")

    try:
        get_loss('mse')
        print("Loss getter OK.")
        get_loss('unknown')
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Note on CCE/BCE gradients:
    # The gradients dL/dy_pred are correct for the loss function itself.
    # When backpropagating through softmax/sigmoid, the combined gradient (dL/dz) is simpler:
    # For CCE with Softmax: y_pred - y_true
    # For BCE with Sigmoid: y_pred - y_true
    # This simplification happens when the derivative of the activation is combined with dL/dy_pred.
    # The Model class will typically handle this combination if it knows about the final activation.