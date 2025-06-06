# LSTM/src/gru.py
import numpy as np
from .activations import get_activation, get_derivative
from .initializers import get_initializer
from .layers import Layer

class GRUCell:
    """Non-vectorized GRU cell for conceptual understanding."""
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights and biases
        # Update gate (z)
        self.W_z = get_initializer('glorot_uniform')((input_dim, hidden_dim))
        self.U_z = get_initializer('glorot_uniform')((hidden_dim, hidden_dim))
        self.b_z = np.zeros(hidden_dim)
        
        # Reset gate (r)
        self.W_r = get_initializer('glorot_uniform')((input_dim, hidden_dim))
        self.U_r = get_initializer('glorot_uniform')((hidden_dim, hidden_dim))
        self.b_r = np.zeros(hidden_dim)
        
        # Candidate hidden state (h_tilde)
        self.W_h = get_initializer('glorot_uniform')((input_dim, hidden_dim))
        self.U_h = get_initializer('glorot_uniform')((hidden_dim, hidden_dim))
        self.b_h = np.zeros(hidden_dim)
        
        self.parameters = [self.W_z, self.U_z, self.b_z,
                          self.W_r, self.U_r, self.b_r,
                          self.W_h, self.U_h, self.b_h]

    def forward(self, x_t, h_prev):
        """
        x_t: Input at current time step. Shape: (input_dim,)
        h_prev: Previous hidden state. Shape: (hidden_dim,)
        Returns: New hidden state h_t and cache for backward pass.
        """
        # Update gate
        z_t = np.dot(x_t, self.W_z) + np.dot(h_prev, self.U_z) + self.b_z
        z_t = get_activation('sigmoid')(z_t)
        
        # Reset gate
        r_t = np.dot(x_t, self.W_r) + np.dot(h_prev, self.U_r) + self.b_r
        r_t = get_activation('sigmoid')(r_t)
        
        # Candidate hidden state
        h_tilde = np.dot(x_t, self.W_h) + np.dot(r_t * h_prev, self.U_h) + self.b_h
        h_tilde = get_activation('tanh')(h_tilde)
        
        # New hidden state
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        cache = (x_t, h_prev, z_t, r_t, h_tilde)
        return h_t, cache

    def backward(self, d_h_next, cache):
        """
        d_h_next: Gradient of loss w.r.t. next hidden state.
        cache: Values stored during forward pass.
        Returns: Gradients for input, previous hidden state, and parameters.
        """
        x_t, h_prev, z_t, r_t, h_tilde = cache
        
        # Initialize parameter gradients
        d_W_z, d_U_z, d_b_z = np.zeros_like(self.W_z), np.zeros_like(self.U_z), np.zeros_like(self.b_z)
        d_W_r, d_U_r, d_b_r = np.zeros_like(self.W_r), np.zeros_like(self.U_r), np.zeros_like(self.b_r)
        d_W_h, d_U_h, d_b_h = np.zeros_like(self.W_h), np.zeros_like(self.U_h), np.zeros_like(self.b_h)
        
        # Gradient through h_t = (1 - z_t) * h_prev + z_t * h_tilde
        d_h_tilde = d_h_next * z_t
        d_z_t = d_h_next * (h_tilde - h_prev)
        d_h_prev_1 = d_h_next * (1 - z_t)
        
        # Gradient through h_tilde = tanh(np.dot(x_t, W_h) + np.dot(r_t * h_prev, U_h) + b_h)
        d_h_tilde_pre_tanh = d_h_tilde * get_derivative('tanh')(h_tilde)
        d_W_h = np.outer(x_t, d_h_tilde_pre_tanh)
        d_U_h = np.outer(r_t * h_prev, d_h_tilde_pre_tanh)
        d_b_h = d_h_tilde_pre_tanh
        d_r_h_prev = np.dot(d_h_tilde_pre_tanh, self.U_h.T)
        d_r_t = d_r_h_prev * h_prev
        d_h_prev_2 = d_r_h_prev * r_t
        
        # Gradient through r_t = sigmoid(np.dot(x_t, W_r) + np.dot(h_prev, U_r) + b_r)
        d_r_t_pre_sigmoid = d_r_t * get_derivative('sigmoid')(r_t)
        d_W_r = np.outer(x_t, d_r_t_pre_sigmoid)
        d_U_r = np.outer(h_prev, d_r_t_pre_sigmoid)
        d_b_r = d_r_t_pre_sigmoid
        d_h_prev_3 = np.dot(d_r_t_pre_sigmoid, self.U_r.T)
        
        # Gradient through z_t = sigmoid(np.dot(x_t, W_z) + np.dot(h_prev, U_z) + b_z)
        d_z_t_pre_sigmoid = d_z_t * get_derivative('sigmoid')(z_t)
        d_W_z = np.outer(x_t, d_z_t_pre_sigmoid)
        d_U_z = np.outer(h_prev, d_z_t_pre_sigmoid)
        d_b_z = d_z_t_pre_sigmoid
        d_h_prev_4 = np.dot(d_z_t_pre_sigmoid, self.U_z.T)
        
        # Combine gradients for h_prev
        d_h_prev = d_h_prev_1 + d_h_prev_2 + d_h_prev_3 + d_h_prev_4
        
        # Gradient for input x_t
        d_x_t = (np.dot(d_z_t_pre_sigmoid, self.W_z.T) +
                 np.dot(d_r_t_pre_sigmoid, self.W_r.T) +
                 np.dot(d_h_tilde_pre_tanh, self.W_h.T))
        
        param_grads = [d_W_z, d_U_z, d_b_z,
                      d_W_r, d_U_r, d_b_r,
                      d_W_h, d_U_h, d_b_h]
        
        return d_x_t, d_h_prev, param_grads

class VectorizedGRUCell:
    """Vectorized GRU cell for efficient computation."""
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights and biases
        # Update gate (z)
        self.W_z = get_initializer('glorot_uniform')((input_dim, hidden_dim))
        self.U_z = get_initializer('glorot_uniform')((hidden_dim, hidden_dim))
        self.b_z = np.zeros((1, hidden_dim))
        
        # Reset gate (r)
        self.W_r = get_initializer('glorot_uniform')((input_dim, hidden_dim))
        self.U_r = get_initializer('glorot_uniform')((hidden_dim, hidden_dim))
        self.b_r = np.zeros((1, hidden_dim))
        
        # Candidate hidden state (h_tilde)
        self.W_h = get_initializer('glorot_uniform')((input_dim, hidden_dim))
        self.U_h = get_initializer('glorot_uniform')((hidden_dim, hidden_dim))
        self.b_h = np.zeros((1, hidden_dim))
        
        self.parameters = [self.W_z, self.U_z, self.b_z,
                          self.W_r, self.U_r, self.b_r,
                          self.W_h, self.U_h, self.b_h]

    def forward(self, X_batch, h_prev):
        """
        X_batch: Input batch. Shape: (batch_size, input_dim)
        h_prev: Previous hidden states. Shape: (batch_size, hidden_dim)
        Returns: New hidden states and cache for backward pass.
        """
        batch_size = X_batch.shape[0]
        
        # Update gate
        z_t = np.dot(X_batch, self.W_z) + np.dot(h_prev, self.U_z) + self.b_z
        z_t = get_activation('sigmoid')(z_t)
        
        # Reset gate
        r_t = np.dot(X_batch, self.W_r) + np.dot(h_prev, self.U_r) + self.b_r
        r_t = get_activation('sigmoid')(r_t)
        
        # Candidate hidden state
        h_tilde = np.dot(X_batch, self.W_h) + np.dot(r_t * h_prev, self.U_h) + self.b_h
        h_tilde = get_activation('tanh')(h_tilde)
        
        # New hidden state
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        cache = (X_batch, h_prev, z_t, r_t, h_tilde)
        return h_t, cache

    def backward(self, d_h_next, cache):
        """
        d_h_next: Gradient of loss w.r.t. next hidden states. Shape: (batch_size, hidden_dim)
        cache: Values stored during forward pass.
        Returns: Gradients for inputs, previous hidden states, and parameters.
        """
        X_batch, h_prev, z_t, r_t, h_tilde = cache
        batch_size = X_batch.shape[0]
        
        # Initialize parameter gradients
        d_W_z = np.zeros_like(self.W_z)
        d_U_z = np.zeros_like(self.U_z)
        d_b_z = np.zeros_like(self.b_z)
        d_W_r = np.zeros_like(self.W_r)
        d_U_r = np.zeros_like(self.U_r)
        d_b_r = np.zeros_like(self.b_r)
        d_W_h = np.zeros_like(self.W_h)
        d_U_h = np.zeros_like(self.U_h)
        d_b_h = np.zeros_like(self.b_h)
        
        # Gradient through h_t = (1 - z_t) * h_prev + z_t * h_tilde
        d_h_tilde = d_h_next * z_t
        d_z_t = d_h_next * (h_tilde - h_prev)
        d_h_prev_1 = d_h_next * (1 - z_t)
        
        # Gradient through h_tilde = tanh(np.dot(X_batch, W_h) + np.dot(r_t * h_prev, U_h) + b_h)
        d_h_tilde_pre_tanh = d_h_tilde * get_derivative('tanh')(h_tilde)
        d_W_h = np.dot(X_batch.T, d_h_tilde_pre_tanh)
        d_U_h = np.dot((r_t * h_prev).T, d_h_tilde_pre_tanh)
        d_b_h = np.sum(d_h_tilde_pre_tanh, axis=0, keepdims=True)
        d_r_h_prev = np.dot(d_h_tilde_pre_tanh, self.U_h.T)
        d_r_t = d_r_h_prev * h_prev
        d_h_prev_2 = d_r_h_prev * r_t
        
        # Gradient through r_t = sigmoid(np.dot(X_batch, W_r) + np.dot(h_prev, U_r) + b_r)
        d_r_t_pre_sigmoid = d_r_t * get_derivative('sigmoid')(r_t)
        d_W_r = np.dot(X_batch.T, d_r_t_pre_sigmoid)
        d_U_r = np.dot(h_prev.T, d_r_t_pre_sigmoid)
        d_b_r = np.sum(d_r_t_pre_sigmoid, axis=0, keepdims=True)
        d_h_prev_3 = np.dot(d_r_t_pre_sigmoid, self.U_r.T)
        
        # Gradient through z_t = sigmoid(np.dot(X_batch, W_z) + np.dot(h_prev, U_z) + b_z)
        d_z_t_pre_sigmoid = d_z_t * get_derivative('sigmoid')(z_t)
        d_W_z = np.dot(X_batch.T, d_z_t_pre_sigmoid)
        d_U_z = np.dot(h_prev.T, d_z_t_pre_sigmoid)
        d_b_z = np.sum(d_z_t_pre_sigmoid, axis=0, keepdims=True)
        d_h_prev_4 = np.dot(d_z_t_pre_sigmoid, self.U_z.T)
        
        # Combine gradients for h_prev
        d_h_prev = d_h_prev_1 + d_h_prev_2 + d_h_prev_3 + d_h_prev_4
        
        # Gradient for input X_batch
        d_X = (np.dot(d_z_t_pre_sigmoid, self.W_z.T) +
               np.dot(d_r_t_pre_sigmoid, self.W_r.T) +
               np.dot(d_h_tilde_pre_tanh, self.W_h.T))
        
        param_grads = [d_W_z, d_U_z, d_b_z,
                      d_W_r, d_U_r, d_b_r,
                      d_W_h, d_U_h, d_b_h]
        
        return d_X, d_h_prev, param_grads

class GRULayer(Layer):
    """GRU layer that processes sequences using vectorized GRU cell."""
    def __init__(self, units, input_dim=None, return_sequences=False, return_state=False):
        super().__init__()
        self.units = units
        self.input_dim = input_dim
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.gru_cell = None

    def build(self, input_shape):
        """Initialize the GRU cell if not already done."""
        if self.input_dim is None:
            self.input_dim = input_shape[-1]
        if self.gru_cell is None:
            self.gru_cell = VectorizedGRUCell(self.input_dim, self.units)
            self.parameters = self.gru_cell.parameters

    def forward(self, X, initial_state=None, training=True):
        """
        X: Input sequences. Shape: (batch_size, sequence_length, input_dim)
        initial_state: Optional initial hidden state. Shape: (batch_size, units)
        Returns: Layer output and cache for backward pass.
        """
        if self.gru_cell is None:
            self.build(X.shape)

        batch_size, seq_length, _ = X.shape
        h_prev = initial_state if initial_state is not None else np.zeros((batch_size, self.units))
        
        # Store all hidden states if return_sequences is True
        h_states = np.zeros((batch_size, seq_length, self.units)) if self.return_sequences else None
        # Store cell states and intermediate values for backward pass
        caches = []
        
        for t in range(seq_length):
            h_prev, cache_t = self.gru_cell.forward(X[:, t, :], h_prev)
            if self.return_sequences:
                h_states[:, t, :] = h_prev
            caches.append(cache_t)
        
        if self.return_sequences:
            if self.return_state:
                return h_states, h_prev, caches
            return h_states, caches
        if self.return_state:
            return h_prev, h_prev, caches
        return h_prev, caches

    def backward(self, d_output, caches, d_next_h=None):
        """
        d_output: Gradient of loss w.r.t. layer output.
                 If return_sequences: Shape (batch_size, sequence_length, units)
                 Else: Shape (batch_size, units)
        caches: List of caches from forward pass.
        d_next_h: Gradient of loss w.r.t. final hidden state from subsequent layer.
        Returns: Gradients for input, initial state, and parameters.
        """
        X_batch = caches[0][0] # Get input from first cache
        batch_size, seq_length, _ = X_batch.shape
        
        # Initialize gradients
        d_X = np.zeros_like(X_batch)
        d_h = np.zeros((batch_size, self.units)) if d_next_h is None else d_next_h
        
        # Initialize parameter gradients
        param_grads = [np.zeros_like(p) for p in self.parameters]
        
        # Backpropagate through time
        for t in reversed(range(seq_length)):
            # If return_sequences, add gradient from output at this timestep
            if self.return_sequences:
                d_h += d_output[:, t, :]
            elif t == seq_length - 1: # Add output gradient only at last timestep if not return_sequences
                d_h += d_output
            
            # Backpropagate through GRU cell
            d_X_t, d_h, d_params_t = self.gru_cell.backward(d_h, caches[t])
            
            # Accumulate gradients
            d_X[:, t, :] = d_X_t
            for i, d_p in enumerate(d_params_t):
                param_grads[i] += d_p
        
        return d_X, d_h, param_grads

if __name__ == '__main__':
    # --- Test GRU Cell ---
    print("--- Testing GRU Cell ---")
    input_dim = 5
    hidden_dim = 4
    batch_size = 2
    seq_length = 3

    # Test non-vectorized GRU cell
    print("\nTesting non-vectorized GRU cell...")
    gru_cell = GRUCell(input_dim, hidden_dim)
    x_t = np.random.randn(input_dim)
    h_prev = np.random.randn(hidden_dim)
    h_t, cache = gru_cell.forward(x_t, h_prev)
    print("Single step output shape:", h_t.shape)
    assert h_t.shape == (hidden_dim,)

    d_h_next = np.random.randn(hidden_dim)
    d_x_t, d_h_prev, param_grads = gru_cell.backward(d_h_next, cache)
    print("Single step gradients shapes:")
    print("d_x_t:", d_x_t.shape)
    print("d_h_prev:", d_h_prev.shape)
    print("Number of parameter gradients:", len(param_grads))
    assert d_x_t.shape == (input_dim,)
    assert d_h_prev.shape == (hidden_dim,)
    assert len(param_grads) == len(gru_cell.parameters)

    # Test vectorized GRU cell
    print("\nTesting vectorized GRU cell...")
    vectorized_gru_cell = VectorizedGRUCell(input_dim, hidden_dim)
    X_batch = np.random.randn(batch_size, input_dim)
    h_prev_batch = np.random.randn(batch_size, hidden_dim)
    h_t_batch, cache_batch = vectorized_gru_cell.forward(X_batch, h_prev_batch)
    print("Batch output shape:", h_t_batch.shape)
    assert h_t_batch.shape == (batch_size, hidden_dim)

    d_h_next_batch = np.random.randn(batch_size, hidden_dim)
    d_X_batch, d_h_prev_batch, param_grads_batch = vectorized_gru_cell.backward(d_h_next_batch, cache_batch)
    print("Batch gradients shapes:")
    print("d_X_batch:", d_X_batch.shape)
    print("d_h_prev_batch:", d_h_prev_batch.shape)
    print("Number of parameter gradients:", len(param_grads_batch))
    assert d_X_batch.shape == (batch_size, input_dim)
    assert d_h_prev_batch.shape == (batch_size, hidden_dim)
    assert len(param_grads_batch) == len(vectorized_gru_cell.parameters)

    # Test GRU Layer
    print("\n--- Testing GRU Layer ---")
    # Test with return_sequences=True and return_state=True
    gru_layer = GRULayer(units=hidden_dim, input_dim=input_dim, 
                         return_sequences=True, return_state=True)
    X_seq = np.random.randn(batch_size, seq_length, input_dim)
    initial_state = np.random.randn(batch_size, hidden_dim)

    # Forward pass
    output_seq, final_state, caches = gru_layer.forward(X_seq, initial_state)
    print("Layer output shapes:")
    print("Output sequence:", output_seq.shape)
    print("Final state:", final_state.shape)
    assert output_seq.shape == (batch_size, seq_length, hidden_dim)
    assert final_state.shape == (batch_size, hidden_dim)

    # Backward pass
    d_output_seq = np.random.randn(*output_seq.shape)
    d_next_h = np.random.randn(batch_size, hidden_dim)
    d_X_seq, d_init_state, param_grads_layer = gru_layer.backward(d_output_seq, caches, d_next_h)
    print("\nLayer gradient shapes:")
    print("d_X_seq:", d_X_seq.shape)
    print("d_init_state:", d_init_state.shape)
    print("Number of parameter gradients:", len(param_grads_layer))
    assert d_X_seq.shape == X_seq.shape
    assert d_init_state.shape == initial_state.shape
    assert len(param_grads_layer) == len(gru_layer.parameters)

    # Test with return_sequences=False and return_state=False
    print("\nTesting GRU Layer with return_sequences=False...")
    gru_layer_simple = GRULayer(units=hidden_dim, input_dim=input_dim, 
                                return_sequences=False, return_state=False)
    output_simple, caches_simple = gru_layer_simple.forward(X_seq)
    print("Simple output shape:", output_simple.shape)
    assert output_simple.shape == (batch_size, hidden_dim)

    d_output_simple = np.random.randn(batch_size, hidden_dim)
    d_X_simple, d_init_simple, param_grads_simple = gru_layer_simple.backward(d_output_simple, caches_simple)
    print("Simple gradient shapes:")
    print("d_X_simple:", d_X_simple.shape)
    print("d_init_simple:", d_init_simple.shape)
    assert d_X_simple.shape == X_seq.shape
    assert d_init_simple.shape == (batch_size, hidden_dim)

    print("\nGRU tests completed.")
    print("Note: Gradient checking is essential for validating the implementation.")