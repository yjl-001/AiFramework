# LSTM/src/seq2seq.py
import numpy as np
from .layers import Embedding, Dense # Assuming Dense might be used for output projection
from .lstm import VectorizedLSTMLayer # Using the vectorized LSTM
from .model import Model # May inherit or use Model for training loop
from .losses import get_loss
from .optimizers import get_optimizer

class Encoder(Model): # Inheriting Model to reuse add_layer, compile, etc. if needed, or just use as a component
    def __init__(self, vocab_size, embedding_dim, lstm_units, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units

        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.lstm = VectorizedLSTMLayer(units=lstm_units, return_sequences=False, return_state=True)
        
        # Add layers to the internal list for parameter management if using Model's structure
        self.add_layer(self.embedding)
        self.add_layer(self.lstm)

    def forward(self, X_encoder, initial_states=None, training=True):
        """
        X_encoder: Input sequence to the encoder (batch_size, sequence_length)
        initial_states: Optional initial hidden and cell states for the LSTM.
        training: Boolean, for layers like Dropout (if implemented).
        Returns: encoder_outputs (if return_sequences=True for LSTM, not typical for basic encoder), 
                 final_hidden_state, final_cell_state
        """
        # 1. Embedding layer
        embedded_input, _ = self.embedding.forward(X_encoder, training=training)
        
        # 2. LSTM layer
        # The VectorizedLSTMLayer's forward expects initial_h, initial_c if provided
        h_0, c_0 = initial_states if initial_states else (None, None)
        
        # lstm_output will be (final_h, final_c) because return_sequences=False, return_state=True
        # cache_lstm will contain (X_input_lstm, H_all, C_all, F_all, I_all, O_all, C_tilde_all, initial_h, initial_c)
        lstm_output, cache_lstm = self.lstm.forward(embedded_input, initial_h=h_0, initial_c=c_0, training=training)
        
        final_hidden_state, final_cell_state = lstm_output
        
        # Cache for backward pass (simplified for now, will need to align with Model's backward pass)
        # For a standalone encoder, we might not need to cache everything if it's part of a larger Seq2Seq model
        # that handles the full backpropagation.
        self.cache = {
            'embedding': (X_encoder, self.embedding.parameters), # Input to embedding, params
            'lstm': cache_lstm # Cache from LSTM forward pass
        }
        return final_hidden_state, final_cell_state

    def backward(self, d_states):
        """
        d_states: Tuple (d_final_hidden_state, d_final_cell_state) - gradients from the decoder or subsequent part.
        This is a simplified backward pass for a standalone encoder. In a Seq2Seq model,
        the gradients would flow from the decoder through these states.
        """
        d_final_hidden, d_final_cell = d_states
        
        # Backward pass for LSTM
        # The LSTM backward needs d_output (which is d_final_hidden here, as return_sequences=False)
        # and d_final_cell_state. It also needs its cache.
        # The VectorizedLSTMLayer.backward expects d_output, d_next_h, d_next_c
        # If return_sequences=False, d_output is effectively the gradient for the last hidden state.
        # So, d_output_lstm = d_final_hidden, d_next_h_lstm = None (or 0), d_next_c_lstm = d_final_cell
        
        # The current VectorizedLSTMLayer.backward expects d_output (grad for all sequences if return_sequences=True, 
        # or grad for last hidden state if return_sequences=False), d_next_h (grad for final h from next layer/time), 
        # d_next_c (grad for final c from next layer/time).
        # For an encoder, d_output is d_final_hidden. d_next_h and d_next_c are the gradients passed from the decoder's initial state.
        # However, the LSTM's own final state gradients are d_final_hidden and d_final_cell.
        # Let's assume d_final_hidden is the primary gradient for the output, and d_final_cell for the cell state.
        
        lstm_cache = self.cache['lstm']
        # The LSTM backward needs grad w.r.t its output (final_h if not return_sequences) 
        # and grad w.r.t its final cell state.
        # The `d_output` for LSTM backward when `return_sequences=False` is the gradient of the final hidden state.
        # The `d_next_h` and `d_next_c` are gradients for the *final* h and c states passed from *subsequent* computation.
        # In an encoder, these are the gradients propagated from the decoder's initial state. So, d_final_hidden and d_final_cell are these.
        lstm_param_grads, d_embedding_output, _, _ = self.lstm.backward(d_output=d_final_hidden, 
                                                                      d_next_h=None, # No further layer using h beyond this point in encoder
                                                                      d_next_c=d_final_cell, # Grad for cell state
                                                                      cache=lstm_cache)
        
        # Backward pass for Embedding
        embedding_input_indices, _ = self.cache['embedding']
        # The cache for embedding.backward should be just X_indices from its forward pass.
        embedding_param_grads, _ = self.embedding.backward(d_embedding_output, embedding_input_indices)
        
        self.gradients = {
            'embedding': embedding_param_grads,
            'lstm': lstm_param_grads
        }
        # Gradients for parameters are stored in self.gradients, similar to Model class
        # The Model class's optimizer would then use these.
        # No dX to return as this is the start of the network.
        return self.gradients

class Decoder(Model):
    def __init__(self, vocab_size, embedding_dim, lstm_units, output_dim, **kwargs):
        """
        output_dim: Usually same as vocab_size for classification over vocabulary.
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size # Target vocabulary size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units # Should match encoder's lstm_units for state transfer
        self.output_dim = output_dim

        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.lstm = VectorizedLSTMLayer(units=lstm_units, return_sequences=True, return_state=True)
        self.dense = Dense(output_units=output_dim, activation='softmax') # Output layer

        self.add_layer(self.embedding)
        self.add_layer(self.lstm)
        self.add_layer(self.dense)

    def forward(self, X_decoder_token, initial_states, training=True):
        """
        X_decoder_token: Current input token to the decoder (batch_size, 1) - for one step.
        initial_states: Tuple (hidden_state, cell_state) from encoder or previous decoder step.
        training: Boolean.
        Returns: output_token_probs (batch_size, vocab_size), next_hidden_state, next_cell_state
        """
        # 1. Embedding layer for the current token
        # Input X_decoder_token shape: (batch_size,) or (batch_size, 1)
        # Embedding layer expects (batch_size, sequence_length)
        # If X_decoder_token is (batch_size,), reshape to (batch_size, 1)
        if X_decoder_token.ndim == 1:
            X_decoder_token = X_decoder_token.reshape(-1, 1)
            
        embedded_token, cache_embedding = self.embedding.forward(X_decoder_token, training=training)
        # embedded_token shape: (batch_size, 1, embedding_dim)

        # 2. LSTM layer (processes one time step)
        h_prev, c_prev = initial_states
        # LSTM forward for a single time step input: embedded_token shape (batch_size, 1, embedding_dim)
        # lstm_outputs will be (all_hidden_states, final_hidden_state, final_cell_state)
        # all_hidden_states shape: (batch_size, 1, lstm_units)
        # final_hidden_state shape: (batch_size, lstm_units)
        # final_cell_state shape: (batch_size, lstm_units)
        (all_h, next_h, next_c), cache_lstm = self.lstm.forward(embedded_token, 
                                                               initial_h=h_prev, 
                                                               initial_c=c_prev, 
                                                               training=training)
        
        # `all_h` from LSTM has shape (batch_size, 1, lstm_units) as it's for one time step.
        # We need to feed (batch_size, lstm_units) to the Dense layer.
        lstm_output_for_dense = all_h[:, 0, :] # Squeeze the time step dimension

        # 3. Dense layer for output token probabilities
        output_token_logits, cache_dense = self.dense.forward(lstm_output_for_dense, training=training)
        # output_token_probs shape: (batch_size, vocab_size)
        
        # Cache for backward pass (for one step)
        self.cache_step = {
            'embedding_input': X_decoder_token, # Cache for embedding layer
            'embedding_cache': cache_embedding,
            'lstm_cache': cache_lstm,
            'dense_cache': cache_dense,
            'initial_states': initial_states # Need h_prev, c_prev for LSTM backward
        }
        return output_token_logits, next_h, next_c

    def backward_step(self, d_output_token_logits, d_next_h_from_future, d_next_c_from_future):
        """
        Backward pass for a single decoding step.
        d_output_token_logits: Gradient of loss w.r.t. the dense layer's output for this step.
        d_next_h_from_future: Gradient of loss w.r.t. this step's h_out, coming from the *next* time step's h_in.
        d_next_c_from_future: Gradient of loss w.r.t. this step's c_out, coming from the *next* time step's c_in.
        Returns: d_embedding_input_grad (not really, as input is indices), 
                 d_prev_h (grad w.r.t. h_in of this step), 
                 d_prev_c (grad w.r.t. c_in of this step),
                 param_grads (list of gradients for embedding, lstm, dense parameters for this step)
        """
        cache = self.cache_step
        
        # 1. Backward through Dense layer
        dense_param_grads, d_lstm_output_for_dense = self.dense.backward(d_output_token_logits, cache['dense_cache'])
        # d_lstm_output_for_dense shape: (batch_size, lstm_units)
        # This needs to be reshaped to (batch_size, 1, lstm_units) for LSTM backward if LSTM expects seq output grad
        d_lstm_h_output_reshaped = d_lstm_output_for_dense.reshape(d_lstm_output_for_dense.shape[0], 1, d_lstm_output_for_dense.shape[1])

        # 2. Backward through LSTM layer
        # LSTM backward expects: d_output (grad for H_all), d_next_h (grad for final H), d_next_c (grad for final C)
        # For a single step, H_all is just H_current. So d_output is d_lstm_h_output_reshaped.
        # d_next_h and d_next_c are gradients from the *next* time step for *this* step's output states.
        lstm_param_grads, d_embedded_token, d_prev_h, d_prev_c = self.lstm.backward(
            d_output=d_lstm_h_output_reshaped, # Grad w.r.t. h_t of this step
            d_next_h=d_next_h_from_future,    # Grad w.r.t. h_t from h_{t+1}'s input
            d_next_c=d_next_c_from_future,    # Grad w.r.t. c_t from c_{t+1}'s input
            cache=cache['lstm_cache']
        )
        # d_embedded_token shape: (batch_size, 1, embedding_dim)

        # 3. Backward through Embedding layer
        # Embedding backward expects d_output (grad for embedded output) and cache (input indices)
        embedding_param_grads, _ = self.embedding.backward(d_embedded_token, cache['embedding_input'])

        step_param_grads = {
            'embedding': embedding_param_grads,
            'lstm': lstm_param_grads,
            'dense': dense_param_grads
        }
        return step_param_grads, d_prev_h, d_prev_c

class Seq2Seq(Model):
    def __init__(self, encoder, decoder, target_vocab_size, sos_token_id, eos_token_id, max_decoder_steps=50):
        super().__init__() # Initialize Model base class
        self.encoder = encoder
        self.decoder = decoder
        self.target_vocab_size = target_vocab_size
        self.sos_token_id = sos_token_id # Start Of Sequence token ID
        self.eos_token_id = eos_token_id # End Of Sequence token ID
        self.max_decoder_steps = max_decoder_steps

        # The parameters of Seq2Seq are the combined parameters of encoder and decoder
        self.layers = encoder.layers + decoder.layers # For easy access by Model's methods
        # self.parameters will be automatically gathered by Model's get_parameters if layers are added.

    def compile(self, optimizer_name, loss_name, learning_rate):
        self.optimizer = get_optimizer(optimizer_name, learning_rate=learning_rate)
        self.loss_func = get_loss(loss_name) # Typically CategoricalCrossentropy for NMT
        # Initialize optimizer with parameters from both encoder and decoder
        all_params = self.encoder.get_parameters() + self.decoder.get_parameters()
        self.optimizer.initialize(all_params)

    def forward(self, X_encoder, X_decoder_input, training=True):
        """
        X_encoder: Source sequence (batch_size, encoder_seq_len)
        X_decoder_input: Target sequence for teacher forcing (batch_size, decoder_seq_len)
                         The actual tokens fed to decoder one by one.
        training: Boolean flag.
        Returns: all_decoder_outputs (batch_size, decoder_seq_len, target_vocab_size)
                 and a cache for backward pass.
        """
        batch_size = X_encoder.shape[0]
        decoder_seq_len = X_decoder_input.shape[1]
        
        # 1. Encoder forward pass
        encoder_h, encoder_c = self.encoder.forward(X_encoder, training=training)
        
        # 2. Decoder forward pass (with teacher forcing if training)
        decoder_hidden_state = encoder_h
        decoder_cell_state = encoder_c
        
        all_decoder_outputs_logits = [] # Store logits from each time step
        
        # Decoder cache for backward pass (list of step caches)
        decoder_step_caches = [] 

        for t in range(decoder_seq_len):
            # Current input token for the decoder
            # During training with teacher forcing, this is the true target token X_decoder_input[:, t]
            # During inference, this would be the token predicted at step t-1 (or SOS at t=0)
            current_decoder_input_token = X_decoder_input[:, t] # Shape: (batch_size,)
            
            # Decoder forward for one step
            # output_token_logits shape: (batch_size, target_vocab_size)
            # next_h, next_c shapes: (batch_size, decoder_lstm_units)
            output_token_logits, next_h, next_c = self.decoder.forward(
                current_decoder_input_token,
                initial_states=(decoder_hidden_state, decoder_cell_state),
                training=training
            )
            all_decoder_outputs_logits.append(output_token_logits)
            decoder_step_caches.append(self.decoder.cache_step) # Save cache from this step
            
            # Update states for the next step
            decoder_hidden_state = next_h
            decoder_cell_state = next_c

        # Stack all decoder step outputs: (decoder_seq_len, batch_size, target_vocab_size)
        # Then transpose to (batch_size, decoder_seq_len, target_vocab_size)
        all_decoder_outputs_logits = np.stack(all_decoder_outputs_logits, axis=1)
        
        # Cache for Seq2Seq backward pass
        self.cache = {
            'encoder_cache': self.encoder.cache,
            'decoder_step_caches': decoder_step_caches, # List of caches from each decoder step
            'encoder_final_states': (encoder_h, encoder_c) # Needed to link decoder grads to encoder
        }
        return all_decoder_outputs_logits, self.cache

    def backward(self, d_all_decoder_outputs_logits, cache):
        """
        d_all_decoder_outputs_logits: Gradient of total loss w.r.t. all decoder output logits.
                                   Shape: (batch_size, decoder_seq_len, target_vocab_size)
        cache: Cache from the forward pass.
        Returns: Gradients for all parameters in encoder and decoder.
        """
        encoder_cache = cache['encoder_cache']
        decoder_step_caches = cache['decoder_step_caches']
        # encoder_h, encoder_c = cache['encoder_final_states'] # Not directly used here, but was link

        batch_size, decoder_seq_len, _ = d_all_decoder_outputs_logits.shape

        # Initialize gradients for encoder and decoder parameters
        # These will accumulate gradients from all decoder time steps
        # Using dicts to store gradients by layer name, then by param type (dW, db, etc.)
        # This structure needs to align with how the optimizer expects gradients.
        # For now, let's assume optimizer takes a flat list of gradients corresponding to a flat list of params.
        
        # Initialize cumulative parameter gradients for encoder and decoder
        # For simplicity, let's assume layers in encoder/decoder have `parameters` (list of ndarrays)
        # and their backward methods return grads in the same order.
        encoder_param_grads_cumulative = [np.zeros_like(p) for p in self.encoder.get_parameters()]
        decoder_param_grads_cumulative = [np.zeros_like(p) for p in self.decoder.get_parameters()]

        # Initialize gradients for decoder's hidden and cell states (propagated backwards in time)
        d_next_h_from_future = np.zeros_like(self.encoder.lstm.units) # Placeholder, shape should be (batch_size, decoder_lstm_units)
        d_next_c_from_future = np.zeros_like(self.encoder.lstm.units) # Placeholder
        # Correct initialization for d_next_h/c based on decoder's LSTM units
        if self.decoder.lstm.units > 0: # Check if LSTM layer exists and has units
             d_next_h_from_future = np.zeros((batch_size, self.decoder.lstm.units))
             d_next_c_from_future = np.zeros((batch_size, self.decoder.lstm.units))
        else: # Should not happen if decoder has an LSTM
            d_next_h_from_future = np.zeros((batch_size, 1)) # Fallback, but indicates issue
            d_next_c_from_future = np.zeros((batch_size, 1))

        # Loop backwards through decoder time steps
        for t in reversed(range(decoder_seq_len)):
            # Gradient of loss w.r.t. current time step's output logits
            d_current_output_logits = d_all_decoder_outputs_logits[:, t, :]
            
            # Restore cache for this specific decoder step
            self.decoder.cache_step = decoder_step_caches[t]
            
            # Perform backward pass for this decoder step
            # step_param_grads is a dict: {'embedding': [dEmb], 'lstm': [dWh, dWx, db], 'dense': [dW, db]}
            # d_prev_h, d_prev_c are grads w.r.t. input states to this step's LSTM
            step_param_grads_dict, d_prev_h, d_prev_c = self.decoder.backward_step(
                d_current_output_logits, 
                d_next_h_from_future, 
                d_next_c_from_future
            )
            
            # Accumulate parameter gradients for decoder layers
            # This needs careful mapping from step_param_grads_dict to decoder_param_grads_cumulative list
            # Assuming self.decoder.layers are [Embedding, LSTM, Dense]
            # And their backward methods return grads in order of their self.parameters
            # Example accumulation (needs to be robust):
            # Embedding grads
            if 'embedding' in step_param_grads_dict and self.decoder.embedding.parameters:
                for i, grad in enumerate(step_param_grads_dict['embedding']):
                    decoder_param_grads_cumulative[i] += grad # Assumes embedding is first layer
            # LSTM grads
            offset = len(self.decoder.embedding.parameters)
            if 'lstm' in step_param_grads_dict and self.decoder.lstm.parameters:
                for i, grad in enumerate(step_param_grads_dict['lstm']):
                    decoder_param_grads_cumulative[offset + i] += grad
            # Dense grads
            offset += len(self.decoder.lstm.parameters)
            if 'dense' in step_param_grads_dict and self.decoder.dense.parameters:
                for i, grad in enumerate(step_param_grads_dict['dense']):
                    decoder_param_grads_cumulative[offset + i] += grad
            
            # Update gradients for next (previous in time) step's output states
            d_next_h_from_future = d_prev_h
            d_next_c_from_future = d_prev_c

        # Gradients d_prev_h, d_prev_c from the first decoder step (t=0) are the gradients
        # w.r.t. the encoder's final hidden and cell states.
        d_encoder_final_h = d_next_h_from_future
        d_encoder_final_c = d_next_c_from_future
        
        # 2. Encoder backward pass
        # Restore encoder's cache (if it was modified or not passed correctly)
        self.encoder.cache = encoder_cache 
        encoder_grads_dict = self.encoder.backward(d_states=(d_encoder_final_h, d_encoder_final_c))
        
        # Accumulate encoder parameter gradients (similar logic as for decoder)
        # Example accumulation (needs to be robust):
        if 'embedding' in encoder_grads_dict and self.encoder.embedding.parameters:
             for i, grad in enumerate(encoder_grads_dict['embedding']):
                encoder_param_grads_cumulative[i] += grad
        offset = len(self.encoder.embedding.parameters)
        if 'lstm' in encoder_grads_dict and self.encoder.lstm.parameters:
            for i, grad in enumerate(encoder_grads_dict['lstm']):
                encoder_param_grads_cumulative[offset+i] += grad

        # Combine all parameter gradients for the optimizer
        # The order must match self.get_parameters() used in optimizer.initialize()
        all_param_gradients = encoder_param_grads_cumulative + decoder_param_grads_cumulative
        return all_param_gradients

    def predict(self, X_encoder, training=False):
        """
        Generate output sequence given an input sequence.
        X_encoder: Source sequence (batch_size, encoder_seq_len)
        Returns: predicted_sequence (batch_size, max_decoder_steps)
        """
        batch_size = X_encoder.shape[0]
        
        # 1. Encoder forward pass
        encoder_h, encoder_c = self.encoder.forward(X_encoder, training=False)
        
        # 2. Decoder prediction loop
        decoder_hidden_state = encoder_h
        decoder_cell_state = encoder_c
        
        # Initialize decoder input with SOS token for each sequence in batch
        current_decoder_input_token = np.full((batch_size, 1), self.sos_token_id, dtype=int)
        
        predicted_sequence_tokens = []

        for _ in range(self.max_decoder_steps):
            output_token_logits, next_h, next_c = self.decoder.forward(
                current_decoder_input_token,
                initial_states=(decoder_hidden_state, decoder_cell_state),
                training=False
            )
            
            # Get token with highest probability (greedy decoding)
            # output_token_logits shape: (batch_size, target_vocab_size)
            predicted_token_id = np.argmax(output_token_logits, axis=1) # Shape: (batch_size,)
            predicted_sequence_tokens.append(predicted_token_id)
            
            # Update states for the next step
            decoder_hidden_state = next_h
            decoder_cell_state = next_c
            
            # Set predicted token as input for the next step
            current_decoder_input_token = predicted_token_id.reshape(-1, 1)
            
            # Optional: Stop if all sequences in batch have generated EOS token
            # This requires checking `predicted_token_id == self.eos_token_id` for all batch items.

        # Stack predicted tokens: (max_decoder_steps, batch_size)
        # Then transpose to (batch_size, max_decoder_steps)
        predicted_sequence = np.stack(predicted_sequence_tokens, axis=1)
        return predicted_sequence

    # train_step, fit, evaluate methods would be similar to Model class, 
    # but using the Seq2Seq forward/backward logic.
    # The loss would be calculated over all_decoder_outputs_logits and Y_target.

if __name__ == '__main__':
    # --- Test Seq2Seq Model (Conceptual) ---
    print("--- Testing Seq2Seq Model (Conceptual) ---")
    # Hyperparameters
    source_vocab_size = 20
    target_vocab_size = 25
    embedding_dim = 32
    lstm_units = 64
    batch_size = 2
    encoder_seq_len = 5
    decoder_seq_len = 6 # For teacher forcing
    sos_token = 0 # Example SOS token ID
    eos_token = 1 # Example EOS token ID (ensure it's < target_vocab_size)

    # Create Encoder and Decoder
    encoder = Encoder(vocab_size=source_vocab_size, embedding_dim=embedding_dim, lstm_units=lstm_units)
    decoder = Decoder(vocab_size=target_vocab_size, embedding_dim=embedding_dim, 
                      lstm_units=lstm_units, output_dim=target_vocab_size)
    
    # Create Seq2Seq model
    seq2seq_model = Seq2Seq(encoder, decoder, target_vocab_size, 
                            sos_token_id=sos_token, eos_token_id=eos_token, 
                            max_decoder_steps=decoder_seq_len + 2)

    # Compile the model (needed for training)
    seq2seq_model.compile(optimizer_name='adam', loss_name='categorical_crossentropy', learning_rate=0.001)

    # Dummy input data
    X_encoder_dummy = np.random.randint(0, source_vocab_size, size=(batch_size, encoder_seq_len))
    # For teacher forcing, decoder input starts with SOS and has true target tokens
    X_decoder_input_dummy = np.random.randint(1, target_vocab_size, size=(batch_size, decoder_seq_len))
    X_decoder_input_dummy = np.concatenate([
        np.full((batch_size,1), sos_token), 
        X_decoder_input_dummy[:, :-1]], axis=1) # Shift right, prepend SOS
    
    # Target output data (true sequences, one-hot encoded or as indices for sparse loss)
    # For CategoricalCrossentropy, Y should be one-hot or loss should handle sparse labels.
    # Assuming loss function handles integer labels for now.
    Y_target_dummy_indices = np.random.randint(0, target_vocab_size, size=(batch_size, decoder_seq_len))

    print(f"X_encoder shape: {X_encoder_dummy.shape}")
    print(f"X_decoder_input shape: {X_decoder_input_dummy.shape}")
    print(f"Y_target_dummy_indices shape: {Y_target_dummy_indices.shape}")

    # --- Test Forward Pass ---
    print("\n--- Testing Forward Pass ---")
    all_decoder_outputs_logits, cache_seq2seq = seq2seq_model.forward(X_encoder_dummy, X_decoder_input_dummy, training=True)
    print(f"Decoder output logits shape: {all_decoder_outputs_logits.shape}") # (batch, dec_len, target_vocab)
    assert all_decoder_outputs_logits.shape == (batch_size, decoder_seq_len, target_vocab_size)

    # --- Test Loss Calculation (Conceptual) ---
    # loss_value, d_loss_logits = seq2seq_model.loss_func(all_decoder_outputs_logits, Y_target_dummy_indices)
    # print(f"Calculated loss (conceptual): {loss_value}")
    # print(f"Gradient of loss w.r.t. logits shape: {d_loss_logits.shape}")
    # assert d_loss_logits.shape == all_decoder_outputs_logits.shape
    
    # --- Test Backward Pass (Conceptual, assuming d_loss_logits is available) ---
    # For a real test, we'd need a loss function that can handle 3D logits and 2D targets (or 3D one-hot targets)
    # and its gradient.
    # Let's use a dummy d_loss_logits for shape testing.
    dummy_d_loss_logits = np.random.randn(*all_decoder_outputs_logits.shape)
    print("\n--- Testing Backward Pass (Conceptual) ---")
    # This part is complex and needs careful gradient checking in a real scenario.
    # The current backward pass implementation is very basic and needs robust testing.
    try:
        all_param_gradients = seq2seq_model.backward(dummy_d_loss_logits, cache_seq2seq)
        print(f"Number of parameter gradient sets: {len(all_param_gradients)}")
        total_params_in_model = len(seq2seq_model.encoder.get_parameters()) + len(seq2seq_model.decoder.get_parameters())
        assert len(all_param_gradients) == total_params_in_model
        print("Backward pass conceptual check: gradient list length matches parameter list length.")
        # Further checks: shapes of individual gradients match parameter shapes.
        model_params = seq2seq_model.encoder.get_parameters() + seq2seq_model.decoder.get_parameters()
        for i, grad in enumerate(all_param_gradients):
            assert grad.shape == model_params[i].shape, f"Grad shape {grad.shape} mismatch for param {i} shape {model_params[i].shape}"
        print("All parameter gradient shapes match corresponding parameter shapes.")

    except Exception as e:
        print(f"Error during conceptual backward pass: {e}")
        import traceback
        traceback.print_exc()

    # --- Test Prediction (Inference) ---
    print("\n--- Testing Prediction (Inference) ---")
    predicted_sequence = seq2seq_model.predict(X_encoder_dummy, training=False)
    print(f"Predicted sequence shape: {predicted_sequence.shape}") # (batch, max_decoder_steps)
    assert predicted_sequence.shape == (batch_size, seq2seq_model.max_decoder_steps)
    print("Predicted sequence (sample):\n", predicted_sequence[0, :10]) # Print first 10 tokens of first batch item

    print("\nSeq2Seq model conceptual tests completed.")
    print("Note: Full training loop and gradient checking are essential for validation.")