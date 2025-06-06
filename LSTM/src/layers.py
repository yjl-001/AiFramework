# LSTM/src/layers.py
import numpy as np
from .initializers import get_initializer
from .activations import get_activation, get_derivative
# from .regularizers import get_regularizer # If regularization is added

class Layer:
    """Base layer class."""
    def __init__(self):
        self.parameters = [] # List of trainable parameter arrays (e.g., weights, biases)
        self.input_shape = None
        self.is_trainable = True # Most layers are trainable by default

    def forward(self, X, training=True):
        raise NotImplementedError

    def backward(self, d_output, cache):
        # cache is whatever was returned by forward pass for use in backward pass
        raise NotImplementedError

    def get_parameters(self):
        """Returns a flat list of trainable parameter arrays."""
        return [p for p in self.parameters if p is not None and isinstance(p, np.ndarray)]

    def get_gradients(self, grads_dict_or_list):
        """
        Helper to extract gradients for self.parameters from a dict or list.
        This depends on how layer.backward returns gradients for its parameters.
        If backward returns a list of grads corresponding to self.parameters, this is direct.
        If backward returns a dict (e.g. {'dW': ..., 'db': ...}), we need to map.
        For now, assume it returns a list of gradients in the same order as self.parameters.
        """
        if isinstance(grads_dict_or_list, list):
            return grads_dict_or_list
        # Add dict handling if necessary based on layer.backward API
        return []

class Dense(Layer):
    def __init__(self, output_units, input_units=None, activation=None, 
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None):
        super().__init__()
        self.output_units = output_units
        self.input_units = input_units # Can be inferred at build time
        self.activation_name = activation
        self.activation_func = get_activation(activation) if activation else None
        self.activation_derivative_func = get_derivative(activation) if activation else None
        
        self.kernel_initializer_name = kernel_initializer
        self.bias_initializer_name = bias_initializer
        # self.kernel_regularizer = get_regularizer(kernel_regularizer) if kernel_regularizer else None
        # self.bias_regularizer = get_regularizer(bias_regularizer) if bias_regularizer else None

        self.W = None # Weights
        self.b = None # Biases
        self.parameters = [self.W, self.b] # Will be populated in build()

        self.cached_input = None
        self.cached_pre_activation = None

    def build(self, input_shape):
        """Initialize weights based on input shape."""
        if self.input_units is None:
            self.input_units = input_shape[-1]
        
        kernel_init_func = get_initializer(self.kernel_initializer_name)
        bias_init_func = get_initializer(self.bias_initializer_name)

        self.W = kernel_init_func((self.input_units, self.output_units))
        self.b = bias_init_func(self.output_units)
        self.parameters = [self.W, self.b] # Update parameters list with actual arrays
        self.input_shape = input_shape

    def forward(self, X, training=True):
        if self.W is None: # Build layer if not already built
            self.build(X.shape)

        self.cached_input = X
        self.cached_pre_activation = np.dot(X, self.W) + self.b
        
        output = self.cached_pre_activation
        if self.activation_func:
            output = self.activation_func(output)
        
        # Cache for backward: input X, weights W, pre-activation output Z
        # (or just input X if Z can be recomputed, but caching Z is often better)
        return output, (X, self.W, self.cached_pre_activation, self.activation_name)

    def backward(self, d_output, cache):
        """
        d_output: Gradient of loss w.r.t. this layer's output (dL/dA if activation, dL/dZ if no activation)
        cache: (X, W, Z, activation_name) from forward pass
        Returns: [dW, db], dX (gradients for parameters, gradient for input)
        """
        X, W, Z_pre_activation, activation_name = cache
        batch_size = X.shape[0]

        # Backpropagate through activation function (if any)
        # d_output is dL/dA (gradient w.r.t. activated output)
        # We need dL/dZ (gradient w.r.t. pre-activation output)
        if activation_name:
            # activation_derivative_func should take Z_pre_activation as input
            dZ = d_output * self.activation_derivative_func(Z_pre_activation) # dL/dA * dA/dZ = dL/dZ
        else:
            dZ = d_output # If no activation, dL/dZ = dL/dA (where A=Z)

        # Gradients for weights and biases
        # dL/dW = dL/dZ * dZ/dW = X.T * dL/dZ
        # dL/db = dL/dZ * dZ/db = sum(dL/dZ, axis=0)
        dW = np.dot(X.T, dZ) / batch_size # Average over batch
        db = np.sum(dZ, axis=0) / batch_size   # Average over batch
        
        # Gradient w.r.t. input (to pass to previous layer)
        # dL/dX = dL/dZ * dZ/dX = dL/dZ * W.T
        dX = np.dot(dZ, W.T)
        
        # Regularization gradients (if implemented)
        # if self.kernel_regularizer:
        #     dW += self.kernel_regularizer.grad(self.W)
        # if self.bias_regularizer:
        #     db += self.bias_regularizer.grad(self.b)

        param_grads = [dW, db]
        return param_grads, dX

class Embedding(Layer):
    def __init__(self, input_dim, output_dim, embeddings_initializer='random_uniform'):
        """
        input_dim: Size of the vocabulary (max integer index + 1).
        output_dim: Dimension of the dense embedding.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer_name = embeddings_initializer
        
        self.embeddings = None # Embedding matrix
        self.parameters = [self.embeddings] # Will be populated in build()
        self.cached_input_indices = None

    def build(self, input_shape=None): # input_shape not strictly needed if input_dim is given
        initializer_func = get_initializer(self.embeddings_initializer_name)
        self.embeddings = initializer_func((self.input_dim, self.output_dim))
        self.parameters = [self.embeddings] # Update with actual array

    def forward(self, X_indices, training=True):
        """
        X_indices: Batch of input sequences (integer-encoded token indices).
                   Shape: (batch_size, sequence_length)
        """
        if self.embeddings is None:
            self.build()
        
        self.cached_input_indices = X_indices
        # Perform embedding lookup
        # Output shape: (batch_size, sequence_length, output_dim)
        embedded_output = self.embeddings[X_indices]
        return embedded_output, X_indices # Cache input indices for backward pass

    def backward(self, d_output, cache):
        """
        d_output: Gradient of loss w.r.t. the embedding layer's output.
                  Shape: (batch_size, sequence_length, output_dim)
        cache: X_indices (input token indices) from forward pass.
        Returns: [dEmbeddings], None (gradient for embeddings, no gradient for input indices)
        """
        X_indices = cache
        batch_size, seq_len, _ = d_output.shape

        # Initialize gradient for the embedding matrix
        dEmbeddings = np.zeros_like(self.embeddings)

        # Accumulate gradients for each token index that appeared in the input
        # This can be done efficiently using np.add.at or a loop
        # np.add.at(dEmbeddings, X_indices, d_output) # This is the most efficient way
        # X_indices needs to be flat for the first argument of np.add.at if dEmbeddings is 2D
        # and d_output needs to be reshaped accordingly.
        
        # Reshape X_indices to be (batch_size * seq_len,)
        # Reshape d_output to be (batch_size * seq_len, output_dim)
        flat_indices = X_indices.reshape(-1)
        flat_d_output = d_output.reshape(-1, self.output_dim)
        
        np.add.at(dEmbeddings, flat_indices, flat_d_output)
        dEmbeddings /= batch_size # Average over batch if loss is averaged

        # The input to Embedding layer (token indices) is not differentiable, 
        # so gradient w.r.t input is None or not applicable in the same way.
        # The chain rule stops here for the input path.
        param_grads = [dEmbeddings]
        return param_grads, None # No gradient to pass further back along the input index path

class Activation(Layer):
    """Layer that applies an activation function element-wise."""
    def __init__(self, activation_name):
        super().__init__()
        self.activation_name = activation_name
        self.activation_func = get_activation(activation_name)
        self.activation_derivative_func = get_derivative(activation_name)
        self.is_trainable = False # Activation layers typically have no trainable parameters
        self.parameters = []
        self.cached_input = None

    def forward(self, X, training=True):
        self.cached_input = X
        return self.activation_func(X), X # Cache input for backward

    def backward(self, d_output, cache):
        """
        d_output: Gradient of loss w.r.t. this layer's output (dL/dA)
        cache: X (input to activation) from forward pass
        Returns: [], dX (no parameter gradients, gradient for input)
        """
        X = cache
        # dL/dX = dL/dA * dA/dX
        dA_dX = self.activation_derivative_func(X)
        dX = d_output * dA_dX
        return [], dX # No parameters, so empty list for param_grads

# TODO: Add Dropout layer (requires handling of 'training' flag)
# TODO: Add other common layers like Flatten, Reshape, Conv2D, MaxPooling2D if building a full CNN/RNN framework

if __name__ == '__main__':
    # --- Test Dense Layer ---
    print("--- Testing Dense Layer ---")
    dense_layer = Dense(output_units=3, input_units=5, activation='relu')
    # dense_layer.build(input_shape=(None, 5)) # Or build explicitly
    
    X_dense = np.random.randn(2, 5) # batch_size=2, input_features=5
    print("Input X_dense:\n", X_dense)
    
    output_dense, cache_dense = dense_layer.forward(X_dense)
    print("Dense output (after relu):\n", output_dense)
    assert output_dense.shape == (2, 3)
    assert (output_dense >= 0).all() # ReLU property

    # Dummy gradient from next layer
    d_output_dense = np.random.randn(2, 3)
    param_grads_dense, dX_dense = dense_layer.backward(d_output_dense, cache_dense)
    dW_dense, db_dense = param_grads_dense
    print("dW_dense shape:", dW_dense.shape) # Expected (5, 3)
    print("db_dense shape:", db_dense.shape) # Expected (3,)
    print("dX_dense shape:", dX_dense.shape) # Expected (2, 5)
    assert dW_dense.shape == dense_layer.W.shape
    assert db_dense.shape == dense_layer.b.shape
    assert dX_dense.shape == X_dense.shape

    # --- Test Embedding Layer ---
    print("\n--- Testing Embedding Layer ---")
    vocab_size = 10
    embed_dim = 4
    seq_length = 3
    batch_size_embed = 2

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim)
    # embedding_layer.build() # Or build explicitly

    X_indices_embed = np.random.randint(0, vocab_size, size=(batch_size_embed, seq_length))
    print("Input X_indices_embed:\n", X_indices_embed)

    output_embed, cache_embed = embedding_layer.forward(X_indices_embed)
    print("Embedding output shape:", output_embed.shape) # Expected (batch, seq_len, embed_dim)
    assert output_embed.shape == (batch_size_embed, seq_length, embed_dim)

    # Dummy gradient from next layer (e.g., LSTM)
    d_output_embed = np.random.randn(batch_size_embed, seq_length, embed_dim)
    param_grads_embed, dX_embed = embedding_layer.backward(d_output_embed, cache_embed)
    dEmbeddings = param_grads_embed[0]
    print("dEmbeddings shape:", dEmbeddings.shape) # Expected (vocab_size, embed_dim)
    assert dEmbeddings.shape == embedding_layer.embeddings.shape
    assert dX_embed is None # No gradient w.r.t. input indices

    # --- Test Activation Layer ---
    print("\n--- Testing Activation Layer ---")
    activation_layer = Activation('tanh')
    X_activation = np.array([[-1., 0., 1.], [-2., 0.5, 2.]])
    print("Input X_activation:\n", X_activation)
    
    output_activation, cache_activation = activation_layer.forward(X_activation)
    print("Activation output (tanh):\n", output_activation)
    assert np.allclose(output_activation, np.tanh(X_activation))

    d_output_activation = np.ones_like(X_activation) # Dummy gradient
    param_grads_act, dX_activation = activation_layer.backward(d_output_activation, cache_activation)
    print("dX_activation (gradient w.r.t input of tanh):\n", dX_activation)
    # Expected: d_output * (1 - tanh(X_activation)**2)
    expected_dX_activation = 1 * (1 - np.tanh(X_activation)**2)
    assert np.allclose(dX_activation, expected_dX_activation)
    assert not param_grads_act # No parameters for activation layer

    print("\nLayer tests completed.")