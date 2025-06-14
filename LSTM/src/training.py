# LSTM/src/training.py
import numpy as np
import time
from .utils import batch_iterator, bleu_score # Assuming BLEU score is in utils
from .losses import get_loss # For loss calculation if not handled by model directly

class Trainer:
    def __init__(self, model, optimizer, loss_function, metrics=None):
        """
        model: The model object (e.g., an instance of Model or Seq2Seq).
        optimizer: The optimizer object.
        loss_function: The loss function object or name.
        metrics: List of metric functions or names to evaluate during training/testing.
        """
        self.model = model
        self.optimizer = optimizer
        if isinstance(loss_function, str):
            self.loss_func = get_loss(loss_function)
        else:
            self.loss_func = loss_function
        
        self.metrics = metrics if metrics else []
        # TODO: Add support for named metrics like 'accuracy', 'bleu'
        # self.compiled_metrics = [get_metric(m) if isinstance(m, str) else m for m in self.metrics]

    def train_step(self, X_batch, Y_batch):
        """
        Performs a single training step (forward pass, loss, backward pass, optimizer update).
        For Seq2Seq, X_batch might be (X_encoder_batch, X_decoder_input_batch).
        Y_batch is the target output.
        """
        # Handle different input structures (e.g., for Seq2Seq)
        if isinstance(X_batch, tuple) or isinstance(X_batch, list):
            # Assuming Seq2Seq: X_batch = (encoder_input, decoder_input)
            predictions_logits, cache = self.model.forward(*X_batch, training=True)
        else:
            predictions_logits, cache = self.model.forward(X_batch, training=True)
        
        loss_value = self.loss_func.loss(predictions_logits, Y_batch)
        d_loss_logits = self.loss_func.gradient(predictions_logits, Y_batch)
        
        # Gradients for all trainable parameters
        param_gradients = self.model.backward(d_loss_logits, cache)
        
        # Update parameters using the optimizer
        # The optimizer needs all model parameters and their corresponding gradients
        all_params = self.model.trainable_parameters # Ensure this gets all relevant params
        self.optimizer.update(all_params, param_gradients)
        
        return loss_value, predictions_logits

    def fit(self, X_train, Y_train, epochs, batch_size, 
            X_val=None, Y_val=None, 
            shuffle=True, callbacks=None):
        """
        Trains the model for a fixed number of epochs.
        X_train: Training data. For Seq2Seq, this could be a tuple (X_encoder_train, X_decoder_input_train).
        Y_train: Training labels/targets.
        epochs: Number of epochs to train.
        batch_size: Size of batches for training.
        X_val, Y_val: Validation data and labels.
        shuffle: Whether to shuffle training data before each epoch.
        callbacks: List of callback functions to call during training (e.g., for logging, early stopping).
        """
        num_samples = X_train[0].shape[0] if isinstance(X_train, tuple) else X_train.shape[0]
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_loss = 0
            num_batches = 0

            # Create batch iterator
            # Handle tuple input for X_train (e.g. Seq2Seq)
            if isinstance(X_train, tuple):
                # Assuming X_train = (X_enc, X_dec_in), Y_train = Y_dec_target
                # The batch_iterator needs to handle multiple arrays for X
                # For simplicity, let's assume batch_iterator can take a tuple of arrays for X
                # and a single array for Y.
                # This might require modification in batch_iterator or a specific one for Seq2Seq.
                # A simple way: iterate indices and slice all arrays.
                indices = np.arange(num_samples)
                if shuffle:
                    np.random.shuffle(indices)
                
                for i in range(0, num_samples, batch_size):
                    batch_indices = indices[i:i+batch_size]
                    X_enc_batch = X_train[0][batch_indices]
                    X_dec_in_batch = X_train[1][batch_indices]
                    X_batch_current = (X_enc_batch, X_dec_in_batch)
                    Y_batch_current = Y_train[batch_indices]
                    
                    loss, _ = self.train_step(X_batch_current, Y_batch_current)
                    epoch_loss += loss
                    num_batches += 1
            else: # Standard case
                for X_batch_current, Y_batch_current in batch_iterator((X_train, Y_train), batch_size, shuffle=shuffle):
                    loss, _ = self.train_step(X_batch_current, Y_batch_current)
                    epoch_loss += loss
                    num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            epoch_duration = time.time() - epoch_start_time
            
            log_message = f"Epoch {epoch+1}/{epochs} - loss: {avg_epoch_loss:.4f} - time: {epoch_duration:.2f}s"
            
            # Validation step
            if X_val is not None and Y_val is not None:
                val_loss, val_metrics_results = self.evaluate(X_val, Y_val, batch_size)
                log_message += f" - val_loss: {val_loss:.4f}"
                # for metric_name, metric_value in val_metrics_results.items():
                #     log_message += f" - val_{metric_name}: {metric_value:.4f}"
            print(log_message)
            
            # Callbacks
            if callbacks:
                for callback in callbacks:
                    # Pass logs to callback (e.g., {'loss': avg_epoch_loss, 'val_loss': val_loss})
                    # callback.on_epoch_end(epoch, logs=...)
                    pass # Implement callback logic
        print("Training finished.")

    def evaluate(self, X_test, Y_test, batch_size):
        """
        Evaluates the model on the test data.
        X_test: Test data. For Seq2Seq, tuple (X_encoder_test, X_decoder_input_test for teacher forcing eval, or just X_encoder_test for generation eval).
        Y_test: Test labels/targets.
        batch_size: Batch size for evaluation.
        Returns: test_loss, test_metrics_results (dictionary)
        """
        total_loss = 0
        num_batches = 0
        all_predictions = [] # For metrics like BLEU that need full sequences
        all_targets = []     # For metrics like BLEU

        # Similar batching logic as in fit
        num_samples = X_test[0].shape[0] if isinstance(X_test, tuple) else X_test.shape[0]
        indices = np.arange(num_samples)

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            Y_batch_current = Y_test[batch_indices]
            
            if isinstance(X_test, tuple):
                # Seq2Seq evaluation might differ: 
                # 1. Teacher forcing: X_test = (X_enc, X_dec_in), Y_test = Y_dec_target
                # 2. Generation: X_test = X_enc, Y_test = Y_dec_target (for BLEU)
                # Assuming teacher forcing for loss calculation here.
                X_enc_batch = X_test[0][batch_indices]
                X_dec_in_batch = X_test[1][batch_indices] # Assumes X_test[1] is decoder input
                X_batch_current = (X_enc_batch, X_dec_in_batch)
                predictions_logits, _ = self.model.forward(*X_batch_current, training=False)
            else:
                X_batch_current = X_test[batch_indices]
                predictions_logits, _ = self.model.forward(X_batch_current, training=False)
            
            loss = self.loss_func.loss(predictions_logits, Y_batch_current)
            total_loss += loss
            num_batches += 1
            
            # Store predictions and targets for sequence-level metrics
            # For classification, predictions_logits might be converted to class labels
            # For Seq2Seq, predictions_logits are (batch, seq_len, vocab_size)
            # We need to convert them to token IDs (e.g., by argmax)
            if self.model.__class__.__name__ == 'Seq2Seq': # Check if it's a Seq2Seq model
                # For BLEU, we need predicted token sequences, not logits
                # This requires a predict_on_batch like function or using model.predict
                # For simplicity, let's assume we get token IDs from logits here.
                # This part needs to align with how Seq2Seq predict works.
                # If evaluating with teacher forcing, Y_batch_current are target indices.
                # predictions_logits -> predicted_indices
                predicted_indices_batch = np.argmax(predictions_logits, axis=-1)
                all_predictions.extend(list(predicted_indices_batch)) # List of arrays
                all_targets.extend(list(Y_batch_current)) # List of arrays
            # else: handle other model types for metrics

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        metrics_results = {}

        # Calculate metrics
        # Example for BLEU score if applicable (for Seq2Seq)
        if self.model.__class__.__name__ == 'Seq2Seq' and any('bleu' in str(m).lower() for m in self.metrics):
            # all_predictions and all_targets are lists of token ID sequences
            # calculate_bleu_score expects list of lists of tokens (strings or ints)
            # This might need conversion from token IDs to actual tokens/words if vocab is available
            # For now, assume calculate_bleu_score can work with lists of lists of integer token IDs.
            # The references (all_targets) might need to be a list of lists of lists for multiple references per sample.
            # Assuming single reference for simplicity here.
            # Ensure EOS tokens are handled (e.g., by trimming sequences at EOS)
            
            # Reformat for calculate_bleu_score: list of candidate sentences, list of lists of reference sentences
            # Example: candidates = [[token1, token2], [token_a, token_b]]
            #          references = [[[ref1_sent1_tok1, ...]], [[ref2_sent1_tok1,...]]]
            # This part is highly dependent on the exact format expected by calculate_bleu_score
            # and how sequences are padded/terminated.
            
            # Simplified BLEU calculation (assuming integer sequences are okay and single reference)
            # This is a placeholder. Real BLEU calculation needs careful data prep.
            if all_predictions and all_targets:
                # Convert list of np.arrays to list of lists
                candidates_for_bleu = [list(p) for p in all_predictions]
                references_for_bleu = [[list(t)] for t in all_targets] # List of lists of lists
                try:
                    bleu = calculate_bleu_score(references_for_bleu, candidates_for_bleu)
                    metrics_results['bleu'] = bleu
                except Exception as e:
                    print(f"Warning: Could not calculate BLEU score: {e}")
                    metrics_results['bleu'] = 0.0 # Or handle error appropriately
        
        # TODO: Implement other metrics (accuracy, etc.)
        return avg_loss, metrics_results

    def predict(self, X_input, batch_size=32):
        """
        Generates predictions for the input data.
        X_input: Input data. For Seq2Seq, this is typically X_encoder.
        """
        # This method might be more suitable directly in the Model class (like model.predict)
        # If Trainer.predict is used, it should call model.predict or model.forward in batches.
        if hasattr(self.model, 'predict') and callable(getattr(self.model, 'predict')):
            # Use model's own predict method if available (e.g., for Seq2Seq generation)
            # Batching for model.predict might need to be handled here if model.predict doesn't do it.
            num_samples = X_input.shape[0]
            all_outputs = []
            for i in range(0, num_samples, batch_size):
                X_batch = X_input[i:i+batch_size]
                outputs_batch = self.model.predict(X_batch, training=False) # model.predict should handle its logic
                all_outputs.append(outputs_batch)
            if all_outputs:
                return np.concatenate(all_outputs, axis=0) if isinstance(all_outputs[0], np.ndarray) else all_outputs
            return [] # Or handle empty input
        else:
            # Generic forward pass for models without a specific predict method
            # This would be for classification/regression, not generation like Seq2Seq
            all_predictions_logits = []
            num_samples = X_input.shape[0]
            for i in range(0, num_samples, batch_size):
                X_batch = X_input[i:i+batch_size]
                logits_batch, _ = self.model.forward(X_batch, training=False)
                all_predictions_logits.append(logits_batch)
            if all_predictions_logits:
                return np.concatenate(all_predictions_logits, axis=0)
            return []

# TODO: Implement Callbacks (e.g., EarlyStopping, ModelCheckpoint, LearningRateScheduler)
class Callback:
    def on_epoch_end(self, epoch, logs=None):
        pass
    def on_train_begin(self, logs=None):
        pass
    # etc.

# Example Usage (Conceptual - requires defined model, data, etc.)
if __name__ == '__main__':
    # This is a conceptual guide. Actual usage requires concrete model and data.
    print("Trainer class defined. Conceptual usage below:")

    # 1. Define or load your model (e.g., from model.py or seq2seq.py)
    #    from .model import Model # Or your specific model class
    #    from .layers import Dense, VectorizedLSTMLayer # etc.
    #    my_model = Model()
    #    my_model.add_layer(VectorizedLSTMLayer(units=64, input_dim=10, return_sequences=False))
    #    my_model.add_layer(Dense(units=1, activation='sigmoid'))

    # 2. Define or load your optimizer
    #    from .optimizers import Adam
    #    my_optimizer = Adam(learning_rate=0.001)

    # 3. Define your loss function
    #    my_loss = 'binary_crossentropy' # Or an instance of a loss class

    # 4. Create Trainer instance
    #    trainer = Trainer(model=my_model, optimizer=my_optimizer, loss_function=my_loss, metrics=['accuracy'])

    # 5. Prepare your data (X_train, Y_train, X_val, Y_val)
    #    # Dummy data for illustration
    #    num_samples_train = 100
    #    seq_len = 10
    #    input_features = 10
    #    X_train_dummy = np.random.rand(num_samples_train, seq_len, input_features)
    #    Y_train_dummy = np.random.randint(0, 2, size=(num_samples_train, 1))
    #    X_val_dummy = np.random.rand(20, seq_len, input_features)
    #    Y_val_dummy = np.random.randint(0, 2, size=(20, 1))

    # 6. Train the model
    #    # trainer.fit(X_train_dummy, Y_train_dummy, epochs=10, batch_size=32, 
    #    #             X_val=X_val_dummy, Y_val=Y_val_dummy)

    # 7. Evaluate the model
    #    # test_loss, test_metrics = trainer.evaluate(X_val_dummy, Y_val_dummy, batch_size=32)
    #    # print(f"Test Loss: {test_loss}, Test Metrics: {test_metrics}")

    # 8. Make predictions
    #    # predictions = trainer.predict(X_val_dummy)
    #    # print("Predictions shape:", predictions.shape)

    print("\nConceptual Trainer usage outlined. Implement with actual model and data.")

    # Example for Seq2Seq BLEU score (very conceptual)
    # Assume seq2seq_model is defined and trained
    # Assume X_encoder_test, X_decoder_input_test, Y_target_test are available
    # Y_target_test would be list of lists of token IDs for reference sentences
    
    # trainer_seq2seq = Trainer(model=seq2seq_model, optimizer=..., loss_function=..., metrics=['bleu'])
    # X_test_s2s = (X_encoder_test, X_decoder_input_test) # For teacher-forcing eval loss
    # Y_test_s2s = Y_target_test # Target sequences (indices)
    # test_loss_s2s, test_metrics_s2s = trainer_seq2seq.evaluate(X_test_s2s, Y_test_s2s, batch_size=16)
    # print(f"Seq2Seq Test Loss: {test_loss_s2s}, BLEU: {test_metrics_s2s.get('bleu')}")

    # For BLEU with generated sequences (not teacher-forced during prediction for BLEU):
    # The evaluate method would need to call model.predict (which does generation)
    # and then compare with Y_target_test.
    # This requires the evaluate method to be more flexible or have a specific mode for generative models.