import numpy as np
from .layers import Conv2D, BatchNorm2D, MaxPooling2D, Flatten, FullyConnected, AvgPooling2D
from .activations import Activation, ReLU # Using correct class name MaxPooling2D
from .model import Model
from .losses import TripletLoss



def InceptionResNetV1_Block(input_tensor, scale, block_type, activation='relu'):
    """Helper function to create a single Inception-ResNet-v1 block."""
    # This is a placeholder for the actual Inception-ResNet-v1 block architecture.
    # A full implementation would involve multiple convolutional layers, branches, and concatenation.
    # For simplicity, we'll use a simplified block structure.
    # This needs to be expanded based on the FaceNet paper's Inception-ResNet-v1 details.

    # Example simplified block:
    # Branch 0
    # conv1 = Conv2D(input_tensor.shape[1], 32, 1, padding='same')(input_tensor) # Assuming input_tensor is (N, C, H, W)
    # act1 = ReLU()(conv1)
    # return act1
    pass # Placeholder - full block architecture needed

class FaceNet(Model):
    def __init__(self, input_shape=(3, 160, 160), embedding_size=128):
        super().__init__()
        self.input_shape = input_shape
        self.embedding_size = embedding_size

        # Stem
        self.add(Conv2D(input_shape=input_shape, num_filters=32, kernel_size=3, stride=2, padding=0)) # Output: (N, 32, 79, 79)
        self.add(BatchNorm2D(num_features=32))
        self.add(ReLU())
        self.add(Conv2D(input_shape=(32, 79, 79), num_filters=32, kernel_size=3, stride=1, padding=0)) # Output: (N, 32, 77, 77)
        self.add(BatchNorm2D(num_features=32))
        self.add(ReLU())
        self.add(Conv2D(input_shape=(32, 77, 77), num_filters=64, kernel_size=3, stride=1, padding=1)) # Output: (N, 64, 77, 77)
        self.add(BatchNorm2D(num_features=64))
        self.add(ReLU())

        self.add(MaxPooling2D(pool_size=3, stride=2)) # Output: (N, 64, 38, 38)

        # Simplified Inception-ResNet-v1 style blocks (placeholders)
        # These would be complex blocks in a real FaceNet model.
        # For now, using simpler Conv layers as placeholders for the blocks.
        # Block 1 (example)
        self.add(Conv2D(input_shape=(64, 38, 38), num_filters=80, kernel_size=1, stride=1, padding=0)) # Output: (N, 80, 38, 38)
        self.add(BatchNorm2D(num_features=80))
        self.add(ReLU())
        self.add(Conv2D(input_shape=(80, 38, 38), num_filters=192, kernel_size=3, stride=1, padding=0)) # Output: (N, 192, 36, 36)
        self.add(BatchNorm2D(num_features=192))
        self.add(ReLU())
        self.add(Conv2D(input_shape=(192, 36, 36), num_filters=256, kernel_size=3, stride=2, padding=0)) # Output: (N, 256, 17, 17)
        self.add(BatchNorm2D(num_features=256))
        self.add(ReLU())

        # Further blocks would follow here...
        # For simplicity, we'll go to Flatten and Dense layers

        self.add(Flatten()) # Output: (N, 256*17*17)
        # Calculate flattened size: 256 * 17 * 17 = 73984
        self.add(FullyConnected(input_size=256*17*17, output_size=self.embedding_size))
        # No activation after the last FC layer for embeddings, L2 normalization is applied later.

    def forward(self, input_data, training=True):
        x = input_data
        for layer in self.layers:
            if isinstance(layer, (BatchNorm2D)):
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)
        
        # L2 Normalization of embeddings
        embeddings = x / np.sqrt(np.sum(np.square(x), axis=-1, keepdims=True) + 1e-6)
        return embeddings

    def train_step(self, anchor_batch, positive_batch, negative_batch, learning_rate):
        # Forward pass for anchor, positive, and negative
        anchor_embeddings = self.forward(anchor_batch, training=True)
        positive_embeddings = self.forward(positive_batch, training=True)
        negative_embeddings = self.forward(negative_batch, training=True)

        # Calculate loss
        if not isinstance(self.loss_fn, TripletLoss):
            raise ValueError("Loss function must be TripletLoss for FaceNet training.")
        
        loss_value = self.loss_fn.loss(anchor_embeddings, positive_embeddings, negative_embeddings)

        # Backward pass
        # Get gradients from the loss function
        grad_anchor, grad_positive, grad_negative = self.loss_fn.gradient(anchor_embeddings, positive_embeddings, negative_embeddings)

        # Initialize accumulated gradients for all trainable layers
        for layer in self.layers:
            if hasattr(layer, 'weights_gradient'): # For Conv2D, FullyConnected
                layer.accumulated_weights_gradient = np.zeros_like(layer.weights_gradient)
                layer.accumulated_biases_gradient = np.zeros_like(layer.biases_gradient)
            if hasattr(layer, 'gamma_gradient'): # For BatchNorm2D
                layer.accumulated_gamma_gradient = np.zeros_like(layer.gamma_gradient)
                layer.accumulated_beta_gradient = np.zeros_like(layer.beta_gradient)

        # --- Backward pass for anchor ---
        current_grad_anchor = grad_anchor
        for layer in reversed(self.layers):
            current_grad_anchor = layer.backward(current_grad_anchor)
            if hasattr(layer, 'weights_gradient'):
                layer.accumulated_weights_gradient += layer.weights_gradient
                layer.accumulated_biases_gradient += layer.biases_gradient
            if hasattr(layer, 'gamma_gradient'):
                layer.accumulated_gamma_gradient += layer.gamma_gradient
                layer.accumulated_beta_gradient += layer.beta_gradient

        # --- Backward pass for positive ---
        current_grad_positive = grad_positive
        for layer in reversed(self.layers):
            current_grad_positive = layer.backward(current_grad_positive)
            if hasattr(layer, 'weights_gradient'):
                layer.accumulated_weights_gradient += layer.weights_gradient
                layer.accumulated_biases_gradient += layer.biases_gradient
            if hasattr(layer, 'gamma_gradient'):
                layer.accumulated_gamma_gradient += layer.gamma_gradient
                layer.accumulated_beta_gradient += layer.beta_gradient

        # --- Backward pass for negative ---
        current_grad_negative = grad_negative
        for layer in reversed(self.layers):
            current_grad_negative = layer.backward(current_grad_negative)
            if hasattr(layer, 'weights_gradient'):
                layer.accumulated_weights_gradient += layer.weights_gradient
                layer.accumulated_biases_gradient += layer.biases_gradient
            if hasattr(layer, 'gamma_gradient'):
                layer.accumulated_gamma_gradient += layer.gamma_gradient
                layer.accumulated_beta_gradient += layer.beta_gradient

        # Collect parameters and their accumulated gradients for the optimizer
        params_for_optimizer = []
        grads_for_optimizer = []
        for layer in self.layers:
            if hasattr(layer, 'weights'): # Conv2D, FullyConnected
                params_for_optimizer.append(layer.weights)
                grads_for_optimizer.append(layer.accumulated_weights_gradient)
                params_for_optimizer.append(layer.biases)
                grads_for_optimizer.append(layer.accumulated_biases_gradient)
            if hasattr(layer, 'gamma'): # BatchNorm2D
                params_for_optimizer.append(layer.gamma)
                grads_for_optimizer.append(layer.accumulated_gamma_gradient)
                params_for_optimizer.append(layer.beta)
                grads_for_optimizer.append(layer.accumulated_beta_gradient)

        # Apply optimizer step
        if self.optimizer:
            self.optimizer.update(params_for_optimizer, grads_for_optimizer)

        return loss_value

    def get_embeddings(self, input_data):
        return self.forward(input_data, training=False)