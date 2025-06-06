# LSTM/src/attention.py
import numpy as np
from .layers import Dense, Layer # Assuming Dense can be used for layers within attention
from .activations import get_activation # For tanh in Bahdanau

class Attention(Layer):
    """Base class for attention mechanisms."""
    def __init__(self):
        super().__init__()
        self.context_vector = None
        self.attention_weights = None

    def forward(self, query, values):
        """
        query: The query vector (e.g., decoder hidden state at time t).
        values: The values to attend over (e.g., encoder hidden states).
        Returns: context_vector, attention_weights
        """
        raise NotImplementedError

    def backward(self, d_context_vector, cache):
        """
        d_context_vector: Gradient of the loss w.r.t. the context vector.
        cache: Values stored during the forward pass needed for gradient calculation.
        Returns: d_query, d_values, param_grads
        """
        raise NotImplementedError

class BahdanauAttention(Attention):
    """Bahdanau Attention (Additive Attention)."""
    def __init__(self, units):
        """
        units: Dimensionality of the internal projection layer (W1, W2, v).
        """
        super().__init__()
        self.units = units
        # Layers for the alignment score calculation
        # e_ij = v_a^T * tanh(W_a * s_{i-1} + U_a * h_j)
        # query is s_{i-1} (decoder hidden state)
        # values are h_j (encoder hidden states)
        self.W1 = Dense(units, activation=None) # Applied to query (decoder state)
        self.W2 = Dense(units, activation=None) # Applied to values (encoder states)
        self.V = Dense(1, activation=None)      # Applied to the sum after tanh
        
        self.parameters = [] # Will be populated after W1, W2, V are built
        self.is_trainable = True
        self.tanh_activation = get_activation('tanh')

    def _initialize_params(self, query_dim, values_dim):
        """Initialize layers if not already done, and collect parameters."""
        if not self.W1.W: # Check if W1 is built
            self.W1.build(input_shape=(None, query_dim))
        if not self.W2.W: # Check if W2 is built
            # W2 processes each encoder output, so input_dim is features of one encoder output
            self.W2.build(input_shape=(None, values_dim))
        if not self.V.W:
            self.V.build(input_shape=(None, self.units))
        
        self.parameters = self.W1.parameters + self.W2.parameters + self.V.parameters

    def forward(self, query, values):
        """
        query: Decoder hidden state at time t-1. Shape: (batch_size, query_dim)
        values: Encoder hidden states. Shape: (batch_size, seq_len_encoder, values_dim)
        Returns: context_vector, attention_weights
        """
        batch_size, query_dim = query.shape
        _, seq_len_encoder, values_dim = values.shape

        self._initialize_params(query_dim, values_dim)

        # query shape: (batch_size, query_dim)
        # W1(query) shape: (batch_size, units)
        # We need to expand query to be (batch_size, 1, query_dim) for broadcasting with values later
        # or expand its projection for broadcasting with projected values.
        query_proj, _ = self.W1.forward(query) # (batch_size, units)
        query_proj_expanded = np.expand_dims(query_proj, axis=1) # (batch_size, 1, units)

        # values shape: (batch_size, seq_len_encoder, values_dim)
        # W2(values) needs to be applied to each time step of encoder_outputs.
        # Reshape values to (batch_size * seq_len_encoder, values_dim) for Dense layer
        values_reshaped = values.reshape(-1, values_dim)
        values_proj_reshaped, _ = self.W2.forward(values_reshaped) # (batch_size * seq_len_encoder, units)
        values_proj = values_proj_reshaped.reshape(batch_size, seq_len_encoder, self.units) # (batch_size, seq_len_encoder, units)

        # Additive score: V.T * tanh(W1(query_expanded) + W2(values))
        # Sum W1(query) and W2(values). query_proj_expanded broadcasts across seq_len_encoder.
        # score_input shape: (batch_size, seq_len_encoder, units)
        score_input_pre_tanh = query_proj_expanded + values_proj 
        score_input_tanh = self.tanh_activation(score_input_pre_tanh) # (batch_size, seq_len_encoder, units)
        
        # Reshape for V layer
        score_input_tanh_reshaped = score_input_tanh.reshape(-1, self.units)
        # score (energy) shape: (batch_size * seq_len_encoder, 1)
        score_reshaped, _ = self.V.forward(score_input_tanh_reshaped)
        score = score_reshaped.reshape(batch_size, seq_len_encoder, 1) # (batch_size, seq_len_encoder, 1)
        score = np.squeeze(score, axis=2) # (batch_size, seq_len_encoder)

        # Attention weights (alpha_ij)
        self.attention_weights = np.exp(score - np.max(score, axis=1, keepdims=True)) # Softmax for stability
        self.attention_weights /= np.sum(self.attention_weights, axis=1, keepdims=True)
        # self.attention_weights shape: (batch_size, seq_len_encoder)

        # Context vector (c_i)
        # Weighted sum of encoder hidden states (values)
        # self.attention_weights needs to be (batch_size, seq_len_encoder, 1) for broadcasting
        expanded_attention_weights = np.expand_dims(self.attention_weights, axis=2)
        self.context_vector = np.sum(expanded_attention_weights * values, axis=1)
        # self.context_vector shape: (batch_size, values_dim)
        
        # Cache for backward pass
        cache = (query, values, query_proj, values_proj, score_input_pre_tanh, self.attention_weights)
        return self.context_vector, self.attention_weights, cache

    def backward(self, d_context_vector, cache):
        """
        d_context_vector: Gradient of loss w.r.t. context_vector. Shape: (batch_size, values_dim)
        cache: (query, values, query_proj, values_proj, score_input_pre_tanh, attention_weights)
        Returns: d_query, d_values_all_steps, param_grads_dict
        """
        query, values, query_proj, values_proj, score_input_pre_tanh, attention_weights = cache
        batch_size, query_dim = query.shape
        _, seq_len_encoder, values_dim = values.shape

        param_grads_W1 = [np.zeros_like(p) for p in self.W1.parameters]
        param_grads_W2 = [np.zeros_like(p) for p in self.W2.parameters]
        param_grads_V = [np.zeros_like(p) for p in self.V.parameters]

        # 1. Gradient w.r.t. attention_weights and values (from context_vector = sum(weights * values))
        # d_context_vector shape: (batch_size, values_dim)
        # attention_weights shape: (batch_size, seq_len_encoder)
        # values shape: (batch_size, seq_len_encoder, values_dim)
        
        # dL/d_values_k = sum_i (dL/d_context_i * d_context_i / d_values_k)
        # d_context_i / d_values_kj = attention_weights_ij (if i==j for value_dim)
        # So, dL/d_values_kj = dL/d_context_j * attention_weights_kj (element-wise for value_dim)
        d_values = np.expand_dims(attention_weights, axis=2) * np.expand_dims(d_context_vector, axis=1)
        # d_values shape: (batch_size, seq_len_encoder, values_dim)
        
        # dL/d_attention_weights_j = sum_k (dL/d_context_k * d_context_k / d_attention_weights_j)
        # d_context_k / d_attention_weights_j = values_jk
        # So, dL/d_attention_weights_j = sum_k (dL/d_context_k * values_jk) = dot(values_j, d_context_vector)
        d_attention_weights = np.einsum('bvd,bv->bvs', values, d_context_vector) # Sum over values_dim
        # Should be: d_attention_weights = np.sum(values * np.expand_dims(d_context_vector, axis=1), axis=2)
        d_attention_weights = np.sum(values * np.expand_dims(d_context_vector, axis=1), axis=2)
        # d_attention_weights shape: (batch_size, seq_len_encoder)

        # 2. Gradient through softmax (for attention_weights from scores)
        # Let S = scores. A = softmax(S). dL/dS_k = sum_j (dL/dA_j * dA_j/dS_k)
        # dA_j/dS_k = A_j * (delta_jk - A_k)
        # This is complex. Simpler: if y = softmax(x), dy/dx = diag(y) - y.T*y
        # For dL/dx = dL/dy * dy/dx
        # d_score = (d_attention_weights - np.sum(d_attention_weights * attention_weights, axis=1, keepdims=True)) * attention_weights
        # More direct: dL/dS_i = A_i * (dL/dA_i - sum_j(dL/dA_j * A_j))
        sum_dLdA_A = np.sum(d_attention_weights * attention_weights, axis=1, keepdims=True)
        d_score = attention_weights * (d_attention_weights - sum_dLdA_A)
        # d_score shape: (batch_size, seq_len_encoder)
        d_score_expanded = np.expand_dims(d_score, axis=2) # (batch_size, seq_len_encoder, 1)

        # 3. Gradient through V layer
        # score = V(tanh_output). d_score_expanded is dL/d_score.
        # V.backward needs dL/d_output_of_V (which is d_score_expanded) and its cache.
        # Cache for V: (input_to_V which is score_input_tanh_reshaped)
        score_input_tanh = self.tanh_activation(score_input_pre_tanh)
        score_input_tanh_reshaped = score_input_tanh.reshape(-1, self.units)
        V_cache = (score_input_tanh_reshaped, self.V.W, None, None) # (X, W, Z, activation_name=None)
        
        # d_score_expanded needs to be reshaped for V.backward
        d_score_for_V_backward = d_score_expanded.reshape(-1, 1)
        V_param_grads_list, d_score_input_tanh_reshaped = self.V.backward(d_score_for_V_backward, V_cache)
        param_grads_V = V_param_grads_list
        d_score_input_tanh = d_score_input_tanh_reshaped.reshape(batch_size, seq_len_encoder, self.units)
        # d_score_input_tanh shape: (batch_size, seq_len_encoder, self.units)

        # 4. Gradient through tanh activation
        # score_input_tanh = tanh(score_input_pre_tanh)
        # d_score_input_pre_tanh = d_score_input_tanh * (1 - score_input_tanh**2)
        d_score_input_pre_tanh = d_score_input_tanh * (1 - score_input_tanh**2)
        # d_score_input_pre_tanh shape: (batch_size, seq_len_encoder, self.units)

        # 5. Gradient through summation (score_input_pre_tanh = query_proj_expanded + values_proj)
        # Gradient is passed equally to both terms of the sum.
        d_query_proj_expanded = np.sum(d_score_input_pre_tanh, axis=1) # Sum over seq_len_encoder
        # d_query_proj_expanded shape: (batch_size, units)
        d_values_proj = d_score_input_pre_tanh
        # d_values_proj shape: (batch_size, seq_len_encoder, self.units)

        # 6. Gradient through W1 (for query_proj)
        # query_proj = W1(query). d_query_proj_expanded is dL/d_query_proj.
        # W1.backward needs dL/d_output_of_W1 and its cache.
        # Cache for W1: (query, self.W1.W, None, None)
        W1_cache = (query, self.W1.W, None, None) 
        W1_param_grads_list, d_query = self.W1.backward(d_query_proj_expanded, W1_cache)
        param_grads_W1 = W1_param_grads_list
        # d_query shape: (batch_size, query_dim)

        # 7. Gradient through W2 (for values_proj)
        # values_proj = W2(values_reshaped). d_values_proj is dL/d_values_proj.
        # W2.backward needs dL/d_output_of_W2 and its cache.
        # Cache for W2: (values_reshaped, self.W2.W, None, None)
        values_reshaped = values.reshape(-1, values_dim)
        W2_cache = (values_reshaped, self.W2.W, None, None)
        
        d_values_proj_reshaped = d_values_proj.reshape(-1, self.units)
        W2_param_grads_list, d_values_reshaped = self.W2.backward(d_values_proj_reshaped, W2_cache)
        param_grads_W2 = W2_param_grads_list
        d_values_from_W2 = d_values_reshaped.reshape(batch_size, seq_len_encoder, values_dim)
        # d_values_from_W2 shape: (batch_size, seq_len_encoder, values_dim)

        # Accumulate gradients for values from two paths
        d_values_total = d_values + d_values_from_W2

        param_grads_all = param_grads_W1 + param_grads_W2 + param_grads_V
        return d_query, d_values_total, param_grads_all

class LuongAttention(Attention):
    """Luong Attention (Multiplicative Attention)."""
    def __init__(self, method='dot', units=None):
        """
        method: 'dot', 'general', or 'concat'. 
                If 'general', units for W_a must be specified (implicitly values_dim).
                If 'concat', units for W_c must be specified.
        units: Dimensionality for W_a in 'general' or W_c in 'concat'.
        """
        super().__init__()
        self.method = method
        self.units = units # Used for 'general' (as W_a output dim) or 'concat'
        self.W_a = None # For 'general' method: Dense(values_dim)
        self.W_c1 = None # For 'concat' method
        self.W_c2 = None # For 'concat' method
        self.V_c = None  # For 'concat' method
        self.tanh_activation = get_activation('tanh')
        self.parameters = []
        self.is_trainable = True

    def _initialize_params(self, query_dim, values_dim):
        if self.method == 'general':
            if not self.W_a or not self.W_a.W:
                # Score: query.T * W_a * value. W_a projects value to query_dim (or common dim)
                # Let W_a project value to query_dim for simplicity of dot product.
                # Or, W_a projects value to `units`, then query also projected to `units` (more flexible)
                # Standard Luong 'general': score(h_t, h_s) = h_t^T W_a h_s
                # W_a has shape (query_dim, values_dim) if h_t is query_dim, h_s is values_dim
                # Or (query_dim, units) if h_s is projected to units by W_a
                # Let's assume W_a projects values to query_dim for direct dot with query.
                # So W_a input is values_dim, output is query_dim.
                self.W_a = Dense(output_units=query_dim, input_units=values_dim, activation=None)
                self.W_a.build(input_shape=(None, values_dim))
                self.parameters = self.W_a.parameters
        elif self.method == 'concat':
            if not self.W_c1 or not self.W_c1.W:
                # Score: v_c^T * tanh(W_c1 * query + W_c2 * value)
                assert self.units is not None, "'units' must be specified for 'concat' method."
                self.W_c1 = Dense(self.units, activation=None) # projects query
                self.W_c2 = Dense(self.units, activation=None) # projects value
                self.V_c = Dense(1, activation=None) # combines them
                self.W_c1.build(input_shape=(None, query_dim))
                self.W_c2.build(input_shape=(None, values_dim))
                self.V_c.build(input_shape=(None, self.units))
                self.parameters = self.W_c1.parameters + self.W_c2.parameters + self.V_c.parameters
        # 'dot' method has no trainable parameters for score calculation itself.

    def forward(self, query, values):
        """
        query: Decoder hidden state. Shape: (batch_size, query_dim)
        values: Encoder hidden states. Shape: (batch_size, seq_len_encoder, values_dim)
        Returns: context_vector, attention_weights, cache
        """
        batch_size, query_dim = query.shape
        _, seq_len_encoder, values_dim = values.shape

        self._initialize_params(query_dim, values_dim)

        # Calculate alignment scores (energies)
        if self.method == 'dot':
            # query: (batch, query_dim), values: (batch, seq_len, values_dim=query_dim)
            # score = sum(query_expanded * values, axis=2)
            # query_expanded: (batch, 1, query_dim)
            if query_dim != values_dim:
                raise ValueError("For 'dot' attention, query_dim must equal values_dim.")
            query_expanded = np.expand_dims(query, axis=1)
            score = np.sum(query_expanded * values, axis=2) # (batch, seq_len_encoder)
            cache_score_terms = (query, values) # For backward
        
        elif self.method == 'general':
            # score(h_t, h_s) = h_t^T (W_a h_s)
            # W_a projects values: (batch, seq_len, values_dim) -> (batch, seq_len, query_dim)
            values_reshaped = values.reshape(-1, values_dim)
            projected_values_reshaped, _ = self.W_a.forward(values_reshaped)
            projected_values = projected_values_reshaped.reshape(batch_size, seq_len_encoder, query_dim)
            
            query_expanded = np.expand_dims(query, axis=1) # (batch, 1, query_dim)
            score = np.sum(query_expanded * projected_values, axis=2) # (batch, seq_len_encoder)
            cache_score_terms = (query, values, projected_values, self.W_a.W) # For backward

        elif self.method == 'concat':
            # score = V_c^T * tanh(W_c1 * query_exp + W_c2 * values)
            query_proj, _ = self.W_c1.forward(query) # (batch, units)
            query_proj_expanded = np.expand_dims(query_proj, axis=1) # (batch, 1, units)

            values_reshaped = values.reshape(-1, values_dim)
            values_proj_reshaped, _ = self.W_c2.forward(values_reshaped) # (batch*seq_len, units)
            values_proj = values_proj_reshaped.reshape(batch_size, seq_len_encoder, self.units)
            
            sum_projections = query_proj_expanded + values_proj # (batch, seq_len, units)
            activated_sum = self.tanh_activation(sum_projections) # (batch, seq_len, units)
            
            activated_sum_reshaped = activated_sum.reshape(-1, self.units)
            score_reshaped, _ = self.V_c.forward(activated_sum_reshaped) # (batch*seq_len, 1)
            score = score_reshaped.reshape(batch_size, seq_len_encoder) # (batch, seq_len)
            cache_score_terms = (query, values, query_proj, values_proj, sum_projections, activated_sum, 
                                 self.W_c1.W, self.W_c2.W, self.V_c.W) # For backward
        else:
            raise ValueError(f"Unknown Luong attention method: {self.method}")

        # Attention weights (alpha_ij) - Softmax over scores
        self.attention_weights = np.exp(score - np.max(score, axis=1, keepdims=True))
        self.attention_weights /= np.sum(self.attention_weights, axis=1, keepdims=True)
        # self.attention_weights shape: (batch_size, seq_len_encoder)

        # Context vector (c_i)
        expanded_attention_weights = np.expand_dims(self.attention_weights, axis=2)
        self.context_vector = np.sum(expanded_attention_weights * values, axis=1)
        # self.context_vector shape: (batch_size, values_dim)

        cache = (cache_score_terms, self.attention_weights, values, query_dim, values_dim, self.method)
        return self.context_vector, self.attention_weights, cache

    def backward(self, d_context_vector, cache):
        """
        d_context_vector: Gradient of loss w.r.t. context_vector. Shape: (batch_size, values_dim)
        cache: From forward pass.
        Returns: d_query, d_values_all_steps, param_grads (list or dict)
        """
        cache_score_terms, attention_weights, values, query_dim, values_dim, method = cache
        batch_size = d_context_vector.shape[0]
        seq_len_encoder = values.shape[1]

        param_grads_all = [] # Store list of [dW, db] for each Dense layer involved

        # 1. Gradient w.r.t. attention_weights and values (from context_vector calculation)
        # This part is common for all Luong methods as context_vector is sum(weights * values)
        d_values_from_context = np.expand_dims(attention_weights, axis=2) * np.expand_dims(d_context_vector, axis=1)
        d_attention_weights = np.sum(values * np.expand_dims(d_context_vector, axis=1), axis=2)

        # 2. Gradient through softmax (for attention_weights from scores)
        sum_dLdA_A = np.sum(d_attention_weights * attention_weights, axis=1, keepdims=True)
        d_score = attention_weights * (d_attention_weights - sum_dLdA_A)
        # d_score shape: (batch_size, seq_len_encoder)

        # 3. Gradient backpropagated through score calculation (method-specific)
        d_query = np.zeros((batch_size, query_dim))
        d_values_from_score = np.zeros_like(values) # (batch, seq_len, values_dim)

        if method == 'dot':
            query, vals_cache = cache_score_terms # vals_cache is 'values' from forward
            # score = sum(query_expanded * values, axis=2)
            # dL/d_query_k = sum_j (dL/d_score_j * d_score_j/d_query_k)
            # d_score_j / d_query_k = values_jk (element k of j-th value vector)
            # So, dL/d_query = sum_j (d_score_j * values_j) = einsum('bj,bjk->bk', d_score, values)
            d_query += np.einsum('bs,bsv->bv', d_score, vals_cache)
            
            # dL/d_values_jk = dL/d_score_j * d_score_j/d_values_jk
            # d_score_j / d_values_jk = query_k
            # So, dL/d_values_j = d_score_j * query
            d_values_from_score += np.expand_dims(d_score, axis=2) * np.expand_dims(query, axis=1)
            # No parameters for 'dot' score calculation itself

        elif method == 'general':
            query, vals_cache, projected_values, W_a_matrix = cache_score_terms
            # score = sum(query_expanded * projected_values, axis=2)
            # projected_values = W_a * values
            
            # Grad w.r.t query (similar to 'dot' but with projected_values)
            d_query += np.einsum('bs,bsv->bv', d_score, projected_values)
            
            # Grad w.r.t projected_values
            d_projected_values = np.expand_dims(d_score, axis=2) * np.expand_dims(query, axis=1)
            # d_projected_values shape: (batch, seq_len, query_dim) (query_dim is W_a output_units)
            
            # Grad through W_a layer
            # W_a.backward needs dL/d_output_of_W_a (d_projected_values) and its cache
            # Cache for W_a: (values_reshaped, W_a.W, None, None)
            values_reshaped = vals_cache.reshape(-1, values_dim)
            W_a_cache = (values_reshaped, W_a_matrix, None, None)
            d_projected_values_reshaped = d_projected_values.reshape(-1, query_dim)
            
            W_a_param_grads, d_vals_reshaped_from_Wa = self.W_a.backward(d_projected_values_reshaped, W_a_cache)
            param_grads_all.extend(W_a_param_grads)
            d_values_from_score += d_vals_reshaped_from_Wa.reshape(batch_size, seq_len_encoder, values_dim)

        elif method == 'concat':
            query, vals_cache, query_proj, values_proj, \
            sum_projections, activated_sum, W_c1_W, W_c2_W, V_c_W = cache_score_terms
            # score = V_c(activated_sum)
            d_score_for_Vc_backward = d_score.reshape(-1, 1)
            activated_sum_reshaped = activated_sum.reshape(-1, self.units)
            Vc_cache = (activated_sum_reshaped, V_c_W, None, None)
            Vc_param_grads, d_activated_sum_reshaped = self.V_c.backward(d_score_for_Vc_backward, Vc_cache)
            param_grads_all.extend(Vc_param_grads)
            d_activated_sum = d_activated_sum_reshaped.reshape(batch_size, seq_len_encoder, self.units)

            # Grad through tanh
            # activated_sum = tanh(sum_projections)
            d_sum_projections = d_activated_sum * (1 - activated_sum**2)

            # Grad through summation (sum_projections = query_proj_expanded + values_proj)
            d_query_proj_expanded_sum = np.sum(d_sum_projections, axis=1) # (batch, units)
            d_values_proj = d_sum_projections # (batch, seq_len, units)

            # Grad through W_c1 (for query_proj)
            Wc1_cache = (query, W_c1_W, None, None)
            Wc1_param_grads, d_query_from_Wc1 = self.W_c1.backward(d_query_proj_expanded_sum, Wc1_cache)
            param_grads_all.extend(Wc1_param_grads)
            d_query += d_query_from_Wc1

            # Grad through W_c2 (for values_proj)
            values_reshaped = vals_cache.reshape(-1, values_dim)
            Wc2_cache = (values_reshaped, W_c2_W, None, None)
            d_values_proj_reshaped = d_values_proj.reshape(-1, self.units)
            Wc2_param_grads, d_vals_reshaped_from_Wc2 = self.W_c2.backward(d_values_proj_reshaped, Wc2_cache)
            param_grads_all.extend(Wc2_param_grads)
            d_values_from_score += d_vals_reshaped_from_Wc2.reshape(batch_size, seq_len_encoder, values_dim)

        # Total gradient for values is sum from context calculation and score calculation
        d_values_total = d_values_from_context + d_values_from_score
        
        return d_query, d_values_total, param_grads_all

if __name__ == '__main__':
    batch_size = 2
    seq_len_enc = 5
    query_dim_dec = 10 # Decoder LSTM units
    values_dim_enc = 8 # Encoder LSTM units

    # Dummy data
    decoder_hidden_state = np.random.randn(batch_size, query_dim_dec)
    encoder_outputs = np.random.randn(batch_size, seq_len_enc, values_dim_enc)

    # --- Test Bahdanau Attention ---
    print("--- Testing Bahdanau Attention ---")
    bahdanau_attention = BahdanauAttention(units=12) # Internal units for Bahdanau
    
    context_b, weights_b, cache_b = bahdanau_attention.forward(decoder_hidden_state, encoder_outputs)
    print("Bahdanau Context vector shape:", context_b.shape) # (batch, values_dim_enc)
    print("Bahdanau Attention weights shape:", weights_b.shape) # (batch, seq_len_enc)
    assert context_b.shape == (batch_size, values_dim_enc)
    assert weights_b.shape == (batch_size, seq_len_enc)
    assert np.allclose(np.sum(weights_b, axis=1), 1.0)

    # Dummy gradient for context vector
    d_context_b = np.random.randn(*context_b.shape)
    d_query_b, d_values_b, params_grads_b = bahdanau_attention.backward(d_context_b, cache_b)
    print("Bahdanau d_query shape:", d_query_b.shape)
    print("Bahdanau d_values shape:", d_values_b.shape)
    print(f"Bahdanau num param grads: {len(params_grads_b)}")
    assert d_query_b.shape == decoder_hidden_state.shape
    assert d_values_b.shape == encoder_outputs.shape
    assert len(params_grads_b) == len(bahdanau_attention.W1.parameters) + \
                                   len(bahdanau_attention.W2.parameters) + \
                                   len(bahdanau_attention.V.parameters)

    # --- Test Luong Attention --- 
    print("\n--- Testing Luong Attention ('dot' method) ---")
    # For 'dot', query_dim must equal values_dim. Adjusting for test.
    query_dim_luong_dot = 8
    decoder_hidden_state_luong_dot = np.random.randn(batch_size, query_dim_luong_dot)
    encoder_outputs_luong_dot = np.random.randn(batch_size, seq_len_enc, query_dim_luong_dot)
    
    luong_dot = LuongAttention(method='dot')
    context_ld, weights_ld, cache_ld = luong_dot.forward(decoder_hidden_state_luong_dot, encoder_outputs_luong_dot)
    print("Luong 'dot' Context vector shape:", context_ld.shape)
    print("Luong 'dot' Attention weights shape:", weights_ld.shape)
    assert context_ld.shape == (batch_size, query_dim_luong_dot)
    assert weights_ld.shape == (batch_size, seq_len_enc)
    assert np.allclose(np.sum(weights_ld, axis=1), 1.0)

    d_context_ld = np.random.randn(*context_ld.shape)
    d_query_ld, d_values_ld, params_grads_ld = luong_dot.backward(d_context_ld, cache_ld)
    print("Luong 'dot' d_query shape:", d_query_ld.shape)
    print("Luong 'dot' d_values shape:", d_values_ld.shape)
    print(f"Luong 'dot' num param grads: {len(params_grads_ld)}") # Should be 0 for dot
    assert d_query_ld.shape == decoder_hidden_state_luong_dot.shape
    assert d_values_ld.shape == encoder_outputs_luong_dot.shape
    assert len(params_grads_ld) == 0

    print("\n--- Testing Luong Attention ('general' method) ---")
    # query_dim_dec, values_dim_enc can be different
    luong_general = LuongAttention(method='general') # W_a projects values_dim_enc to query_dim_dec
    context_lg, weights_lg, cache_lg = luong_general.forward(decoder_hidden_state, encoder_outputs)
    print("Luong 'general' Context vector shape:", context_lg.shape)
    print("Luong 'general' Attention weights shape:", weights_lg.shape)
    assert context_lg.shape == (batch_size, values_dim_enc)
    assert weights_lg.shape == (batch_size, seq_len_enc)

    d_context_lg = np.random.randn(*context_lg.shape)
    d_query_lg, d_values_lg, params_grads_lg = luong_general.backward(d_context_lg, cache_lg)
    print("Luong 'general' d_query shape:", d_query_lg.shape)
    print("Luong 'general' d_values shape:", d_values_lg.shape)
    print(f"Luong 'general' num param grads: {len(params_grads_lg)}")
    assert len(params_grads_lg) == len(luong_general.W_a.parameters)

    print("\n--- Testing Luong Attention ('concat' method) ---")
    luong_concat = LuongAttention(method='concat', units=15)
    context_lc, weights_lc, cache_lc = luong_concat.forward(decoder_hidden_state, encoder_outputs)
    print("Luong 'concat' Context vector shape:", context_lc.shape)
    print("Luong 'concat' Attention weights shape:", weights_lc.shape)
    assert context_lc.shape == (batch_size, values_dim_enc)
    assert weights_lc.shape == (batch_size, seq_len_enc)

    d_context_lc = np.random.randn(*context_lc.shape)
    d_query_lc, d_values_lc, params_grads_lc = luong_concat.backward(d_context_lc, cache_lc)
    print("Luong 'concat' d_query shape:", d_query_lc.shape)
    print("Luong 'concat' d_values shape:", d_values_lc.shape)
    print(f"Luong 'concat' num param grads: {len(params_grads_lc)}")
    assert len(params_grads_lc) == len(luong_concat.W_c1.parameters) + \
                                     len(luong_concat.W_c2.parameters) + \
                                     len(luong_concat.V_c.parameters)

    print("\nAttention mechanism tests completed.")
    print("Note: Gradient checking is crucial for validating backward passes.")