# LSTM/main.py
import numpy as np

# Import components from the src directory
from src.model import Model
from src.layers import Dense, Embedding # Assuming VectorizedLSTMLayer is in lstm.py or layers.py
from src.lstm import VectorizedLSTMLayer # Corrected import
from src.activations import get_activation
from src.losses import get_loss
from src.optimizers import Adam, SGD
from src.training import Trainer
from src.utils import (pad_sequences, build_vocab, sentences_to_sequences, 
                       bleu_score, batch_iterator)
from src.seq2seq import Encoder, Decoder, Seq2Seq
# from src.attention import BahdanauAttention, LuongAttention # If using attention
# from src.gru import GRULayer # If comparing with GRU

def run_sequence_classification_example():
    print("Running Sequence Classification Example...")
    # 1. Generate Dummy Data
    num_samples = 200
    sequence_length = 15
    vocab_size = 20 # Max integer value in sequences + 1 for padding
    embedding_dim = 10
    lstm_units = 32

    # Random integer sequences
    X = np.random.randint(1, vocab_size, size=(num_samples, sequence_length)) # 0 is for padding
    # Pad sequences (though they are already same length here, good practice)
    X = pad_sequences(X, max_len=sequence_length, value=0)
    Y = np.random.randint(0, 2, size=(num_samples, 1)) # Binary classification

    # Split data (simple split)
    split_idx = int(num_samples * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

    # 2. Define Model
    model = Model()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
    model.add(VectorizedLSTMLayer(input_size=embedding_dim, hidden_size=lstm_units, return_sequences=False))
    model.add(Dense(output_units=1, activation='sigmoid')) # Sigmoid for binary classification

    # 3. Compile Model (Set optimizer and loss)
    optimizer = Adam(learning_rate=0.01)
    loss_function = 'binary_crossentropy'
    # model.compile(optimizer=optimizer, loss_function=loss_function) # Model class needs compile method

    # 4. Train Model using Trainer
    trainer = Trainer(model=model, optimizer=optimizer, loss_function=loss_function)
    print("\nStarting training for sequence classification...")
    trainer.fit(X_train, Y_train, epochs=5, batch_size=16, X_val=X_test, Y_val=Y_test)

    # 5. Evaluate Model
    print("\nEvaluating sequence classification model...")
    test_loss, test_metrics = trainer.evaluate(X_test, Y_test, batch_size=16)
    print(f"Test Loss: {test_loss:.4f}")
    # print(f"Test Metrics: {test_metrics}") # If accuracy or other metrics are implemented

    # 6. Make Predictions
    # predictions = trainer.predict(X_test[:5])
    # print("\nSample Predictions (logits/probabilities):")
    # print(predictions)
    # print("Actual Labels:")
    # print(Y_test[:5])

def run_nmt_seq2seq_example():
    print("\nRunning Neural Machine Translation (NMT) Seq2Seq Example...")

    # 1. Dummy NMT Data
    # Source sentences (e.g., English)
    input_texts = [
        "hello world",
        "how are you",
        "good morning",
        "thank you",
        "see you later"
    ] * 20 # More data

    # Target sentences (e.g., French - very simplified)
    target_texts_in = [
        "<sos> bonjour le monde",
        "<sos> comment ca va",
        "<sos> bonjour",
        "<sos> merci",
        "<sos> a bientot"
    ] * 20
    target_texts_out = [
        "bonjour le monde <eos>",
        "comment ca va <eos>",
        "bonjour <eos>",
        "merci <eos>",
        "a bientot <eos>"
    ] * 20

    # 2. Create Vocabularies and Convert to Sequences
    input_vocab, input_token_to_idx, input_idx_to_token = create_vocabulary(input_texts)
    target_vocab, target_token_to_idx, target_idx_to_token = create_vocabulary(target_texts_in + target_texts_out)

    input_vocab_size = len(input_vocab)
    target_vocab_size = len(target_vocab)
    print(f"Input Vocab Size: {input_vocab_size}, Target Vocab Size: {target_vocab_size}")

    encoder_input_seq = sentences_to_sequences(input_texts, input_token_to_idx)
    decoder_input_seq = sentences_to_sequences(target_texts_in, target_token_to_idx)
    decoder_target_seq = sentences_to_sequences(target_texts_out, target_token_to_idx)

    max_encoder_seq_len = max(len(seq) for seq in encoder_input_seq)
    max_decoder_seq_len = max(len(seq) for seq in decoder_input_seq) # also for target

    encoder_input_data = pad_sequences(encoder_input_seq, maxlen=max_encoder_seq_len, padding_value=input_token_to_idx.get('<pad>', 0))
    decoder_input_data = pad_sequences(decoder_input_seq, maxlen=max_decoder_seq_len, padding_value=target_token_to_idx.get('<pad>', 0))
    # Decoder target data needs to be one-hot encoded or use sparse categorical crossentropy
    # For now, assume loss function handles integer targets and logits
    decoder_target_data = pad_sequences(decoder_target_seq, maxlen=max_decoder_seq_len, padding_value=target_token_to_idx.get('<pad>', 0))

    # For Seq2Seq, the target for loss is typically 3D (batch, seq_len, vocab_size) if using categorical CE with one-hot targets
    # Or 2D (batch, seq_len) if using sparse CE. Our current loss expects logits and 2D/3D targets.
    # Let's assume our CategoricalCrossentropy can handle 2D targets (batch, seq_len) of indices and 3D logits (batch, seq_len, vocab_size)

    print(f"Encoder input shape: {encoder_input_data.shape}")
    print(f"Decoder input shape: {decoder_input_data.shape}")
    print(f"Decoder target shape: {decoder_target_data.shape}")

    # Split data
    num_samples = len(encoder_input_data)
    split_idx = int(num_samples * 0.8)
    X_enc_train, X_enc_test = encoder_input_data[:split_idx], encoder_input_data[split_idx:]
    X_dec_in_train, X_dec_in_test = decoder_input_data[:split_idx], decoder_input_data[split_idx:]
    Y_dec_target_train, Y_dec_target_test = decoder_target_data[:split_idx], decoder_target_data[split_idx:]

    # 3. Define Seq2Seq Model
    embedding_dim = 20
    hidden_units = 32

    encoder = Encoder(input_vocab_size, embedding_dim, hidden_units, max_encoder_seq_len)
    decoder = Decoder(target_vocab_size, embedding_dim, hidden_units, max_decoder_seq_len, target_vocab_size)
    seq2seq_model = Seq2Seq(encoder, decoder, target_token_to_idx['<sos>'], target_token_to_idx['<eos>'], max_decoder_seq_len)

    # 4. Trainer for Seq2Seq
    optimizer_s2s = Adam(learning_rate=0.01)
    # Loss for Seq2Seq is typically categorical crossentropy over the vocabulary distribution at each time step.
    loss_function_s2s = get_loss('categorical_crossentropy') # Ensure this can handle (logits_3d, targets_2d_indices)
    
    trainer_s2s = Trainer(model=seq2seq_model, optimizer=optimizer_s2s, loss_function=loss_function_s2s, metrics=['bleu'])

    # Training data for Seq2Seq trainer.fit needs to be (X_encoder, X_decoder_input), Y_decoder_target
    X_train_s2s = (X_enc_train, X_dec_in_train)
    Y_train_s2s = Y_dec_target_train
    X_test_s2s = (X_enc_test, X_dec_in_test)
    Y_test_s2s = Y_dec_target_test

    print("\nStarting training for NMT Seq2Seq model...")
    trainer_s2s.fit(X_train_s2s, Y_train_s2s, epochs=10, batch_size=8, 
                    X_val=X_test_s2s, Y_val=Y_test_s2s)

    # 5. Evaluate (including BLEU score)
    print("\nEvaluating NMT Seq2Seq model...")
    # The evaluate method in Trainer needs to be robust for Seq2Seq, especially for BLEU.
    # It might need to generate sequences using model.predict and then compare.
    # For now, it calculates loss based on teacher forcing if X_dec_in_test is provided.
    test_loss_s2s, test_metrics_s2s = trainer_s2s.evaluate(X_test_s2s, Y_test_s2s, batch_size=8)
    print(f"NMT Test Loss (teacher forcing): {test_loss_s2s:.4f}")
    if 'bleu' in test_metrics_s2s:
        print(f"NMT Test BLEU Score: {test_metrics_s2s['bleu']:.4f}")
    else:
        print("BLEU score not calculated or available in metrics.")

    # 6. Translate some test sentences
    print("\nTranslating sample sentences...")
    for i in range(min(5, len(X_enc_test))):
        input_seq = X_enc_test[i:i+1] # Batch of 1
        # The model.predict for Seq2Seq should handle the generation loop
        predicted_sequence_indices = seq2seq_model.predict(input_seq) # Returns list of lists of token indices
        
        input_sentence = ' '.join([input_idx_to_token.get(idx, '?') for idx in input_seq[0] if idx != input_token_to_idx.get('<pad>', 0)])
        
        # predicted_sequence_indices is expected to be a list (batch) of lists (sequence) of token IDs
        if predicted_sequence_indices and predicted_sequence_indices[0]:
            translated_tokens = [target_idx_to_token.get(idx, '?') for idx in predicted_sequence_indices[0]]
            # Remove SOS/EOS if present for display
            if translated_tokens[0] == '<sos>': translated_tokens = translated_tokens[1:]
            if translated_tokens and translated_tokens[-1] == '<eos>': translated_tokens = translated_tokens[:-1]
            translated_sentence = ' '.join(translated_tokens)
        else:
            translated_sentence = "<empty_prediction>"

        reference_tokens = [target_idx_to_token.get(idx, '?') for idx in Y_dec_target_test[i] if idx != target_token_to_idx.get('<pad>',0) and idx != target_token_to_idx.get('<sos>', -1)]
        if reference_tokens and reference_tokens[-1] == '<eos>': reference_tokens = reference_tokens[:-1]
        reference_sentence = ' '.join(reference_tokens)

        print(f"Input: '{input_sentence}'")
        print(f"Predicted: '{translated_sentence}'")
        print(f"Reference: '{reference_sentence}'")
        # Individual BLEU (not standard, but for illustration)
        # bleu_single = calculate_bleu_score([[reference_tokens]], [translated_tokens])
        # print(f"BLEU (single): {bleu_single:.4f}")
        print("---")

if __name__ == '__main__':
    # You can choose which example to run
    run_sequence_classification_example()
    # run_nmt_seq2seq_example() # This is more complex and might need more debugging

    print("\nScript finished.")
    print("Note: These examples use dummy data and simplified setups.")
    print("The NMT example, in particular, requires careful handling of vocabularies, padding, and the BLEU calculation for meaningful results.")
    print("Further debugging and refinement of Seq2Seq model, Trainer, and BLEU integration are likely needed.")

# TODO:
# - Implement `model.compile` if it simplifies Trainer setup.
# - Refine BLEU score calculation and integration in Trainer.evaluate for Seq2Seq.
# - Ensure `Seq2Seq.predict` correctly generates sequences for translation.
# - Add more sophisticated data loading and preprocessing for NMT.
# - Implement attention mechanisms and GRU comparison as per requirements.
# - Add recurrent dropout to LSTM/GRU layers.
# - Compare with TensorFlow/PyTorch implementations (qualitative or quantitative).
# - Add training optimizations (gradient clipping, learning rate schedules).