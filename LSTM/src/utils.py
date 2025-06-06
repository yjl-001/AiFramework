# LSTM/src/utils.py
import numpy as np
from collections import Counter
import math

# --- Data Preprocessing --- 
def build_vocab(sentences, min_freq=1, reserved_tokens=None):
    """
    Builds a vocabulary from a list of sentences.
    sentences: list of list of tokens (e.g., [['hello', 'world'], ['how', 'are', 'you']])
    min_freq: minimum frequency for a token to be included in the vocabulary.
    reserved_tokens: list of tokens to be added at the beginning of the vocab (e.g., ['<pad>', '<unk>', '<sos>', '<eos>'])
    Returns: word_to_idx (dict), idx_to_word (list)
    """
    token_counts = Counter()
    for sentence in sentences:
        token_counts.update(sentence)

    # Filter tokens by min_freq
    filtered_tokens = [token for token, count in token_counts.items() if count >= min_freq]
    
    # Add reserved tokens if any
    if reserved_tokens is None:
        reserved_tokens = []
    
    # Ensure reserved tokens are unique and at the beginning
    unique_reserved = []
    for rt in reserved_tokens:
        if rt not in unique_reserved:
            unique_reserved.append(rt)
            
    # Combine and create vocab
    idx_to_word = unique_reserved + [token for token in filtered_tokens if token not in unique_reserved]
    word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}
    
    return word_to_idx, idx_to_word

def sentences_to_sequences(sentences, word_to_idx, unk_token='<unk>', max_len=None, padding='post', truncating='post'):
    """
    Converts a list of sentences (list of tokens) to a list of sequences (list of token indices).
    Handles unknown words, padding, and truncating.
    """
    sequences = []
    unk_idx = word_to_idx.get(unk_token)
    if unk_idx is None and '<unk>' in word_to_idx: # Fallback if specific unk_token not found
        unk_idx = word_to_idx['<unk>']

    for sentence in sentences:
        seq = [word_to_idx.get(token, unk_idx) for token in sentence if token in word_to_idx or unk_idx is not None]
        sequences.append(seq)

    if max_len is not None:
        sequences = pad_sequences(sequences, max_len=max_len, padding=padding, truncating=truncating, value=word_to_idx.get('<pad>', 0))
        
    return np.array(sequences)

def pad_sequences(sequences, max_len, padding='post', truncating='post', value=0):
    """
    Pads sequences to the same length.
    sequences: list of lists (sequences of token indices)
    max_len: maximum length of all sequences.
    padding: 'pre' or 'post'.
    truncating: 'pre' or 'post'.
    value: padding value (typically index of <pad> token).
    """
    padded_sequences = []
    for seq in sequences:
        if len(seq) > max_len:
            if truncating == 'pre':
                new_seq = seq[-max_len:]
            else: # 'post'
                new_seq = seq[:max_len]
        else:
            new_seq = seq
        
        pad_width = max_len - len(new_seq)
        if padding == 'pre':
            padded_sequences.append([value] * pad_width + new_seq)
        else: # 'post'
            padded_sequences.append(new_seq + [value] * pad_width)
            
    return np.array(padded_sequences)

# --- BLEU Score Calculation --- 
# (Based on nltk.translate.bleu_score implementation logic)

def ngrams(sequence, n):
    """Generates n-grams from a sequence."""
    return zip(*[sequence[i:] for i in range(n)])

def modified_precision(references, hypothesis, n):
    """
    Calculates modified n-gram precision.
    references: list of reference translations (list of tokens).
    hypothesis: hypothesis translation (list of tokens).
    n: n-gram order.
    """
    hyp_ngrams = Counter(ngrams(hypothesis, n))
    if not hyp_ngrams: # Empty hypothesis n-grams
        return 0

    clipped_counts = 0
    total_hyp_ngram_count = sum(hyp_ngrams.values())

    for ngram, count in hyp_ngrams.items():
        max_ref_count = 0
        for ref in references:
            ref_ngrams = Counter(ngrams(ref, n))
            max_ref_count = max(max_ref_count, ref_ngrams.get(ngram, 0))
        clipped_counts += min(count, max_ref_count)
    
    return clipped_counts / total_hyp_ngram_count if total_hyp_ngram_count > 0 else 0

def brevity_penalty(references, hypothesis):
    """
    Calculates brevity penalty.
    """
    hyp_len = len(hypothesis)
    if hyp_len == 0:
        return 0 # Avoid division by zero if hypothesis is empty

    # Find the reference with length closest to hypothesis length
    closest_ref_len = min((abs(len(ref) - hyp_len), len(ref)) for ref in references)[1]
    
    if hyp_len > closest_ref_len:
        return 1.0
    else:
        return math.exp(1 - closest_ref_len / hyp_len)

def bleu_score(references, hypothesis, max_n=4, weights=None):
    """
    Calculates BLEU score.
    references: list of reference translations (each is a list of tokens).
    hypothesis: a hypothesis translation (list of tokens).
    max_n: maximum n-gram order to consider.
    weights: list of weights for n-grams (e.g., [0.25, 0.25, 0.25, 0.25] for BLEU-4).
    """
    if weights is None:
        weights = [1/max_n] * max_n
    if len(weights) != max_n:
        raise ValueError(f"Length of weights ({len(weights)}) must be equal to max_n ({max_n}).")

    # Ensure references is a list of lists
    if not isinstance(references, list) or not all(isinstance(ref, list) for ref in references):
        # If a single reference string is passed, wrap it
        if isinstance(references, list) and all(isinstance(token, (str, int)) for token in references):
             references = [references]
        else:
            raise TypeError("References should be a list of lists of tokens.")
    if not isinstance(hypothesis, list):
        raise TypeError("Hypothesis should be a list of tokens.")

    # Handle empty hypothesis or references
    if not hypothesis or not any(references):
        return 0.0

    precisions = []
    for i in range(1, max_n + 1):
        p_n = modified_precision(references, hypothesis, i)
        precisions.append(p_n)
    
    # Geometric mean of precisions (log sum to avoid underflow)
    # Add small epsilon to precisions to avoid log(0)
    epsilon = 1e-12 
    sum_log_precisions = 0.0
    for i, p_n in enumerate(precisions):
        if p_n > 0: # Only include if precision is > 0
            sum_log_precisions += weights[i] * math.log(p_n + epsilon)
        # If any precision is 0 for a non-zero weight, the geometric mean will be 0.
        # However, NLTK's smoothing handles this. Here, if p_n is 0, its log term is effectively -inf.
        # A common approach is if any p_n is 0, the score is 0 unless smoothing is applied.
        # For simplicity, if a precision is zero, its contribution to the sum of logs will be very negative.
        # If all precisions are zero, score will be zero.

    # If all precisions for which weights > 0 are zero, then score is 0.
    # Check if any weighted precision is non-zero
    weighted_precisions_are_zero = True
    for i, p_n in enumerate(precisions):
        if weights[i] > 0 and p_n > 0:
            weighted_precisions_are_zero = False
            break
    
    if weighted_precisions_are_zero and sum(weights) > 0:
        # If all relevant precisions are zero, BLEU is 0
        # (unless smoothing is applied, which is not here)
        return 0.0
        
    # If sum_log_precisions is very small (e.g., due to a zero precision), exp will be near zero.
    # If all precisions are zero, sum_log_precisions will be effectively -infinity.
    # math.exp(-inf) is 0.0.
    score = math.exp(sum_log_precisions)
    
    bp = brevity_penalty(references, hypothesis)
    return bp * score

# --- Other Utilities (Example: Batch Iterator) ---
def batch_iterator(data, batch_size, shuffle=True):
    """Generates batches of data."""
    X, y = data
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


if __name__ == '__main__':
    # --- Test Vocab Building ---
    print("--- Testing Vocab Building ---")
    corpus = [
        ['this', 'is', 'a', 'sentence'],
        ['this', 'is', 'another', 'sentence', 'example'],
        ['example', 'of', 'vocab', 'building']
    ]
    reserved = ['<pad>', '<unk>', '<sos>', '<eos>']
    w2i, i2w = build_vocab(corpus, min_freq=1, reserved_tokens=reserved)
    print("Word to Idx:", w2i)
    print("Idx to Word:", i2w)
    assert i2w[0] == '<pad>' and i2w[1] == '<unk>'
    assert w2i['sentence'] > w2i['<eos>'] # Check ordering

    # --- Test Sentence to Sequences & Padding ---
    print("\n--- Testing Sentence to Sequences & Padding ---")
    seqs = sentences_to_sequences(corpus, w2i, unk_token='<unk>', max_len=6, padding='post', truncating='post')
    print("Padded Sequences:\n", seqs)
    assert seqs.shape == (3, 6)
    assert seqs[0, -1] == w2i['<pad>'] # Check padding
    assert seqs[1, 4] == w2i['example'] # Check content

    # --- Test BLEU Score ---
    print("\n--- Testing BLEU Score ---")
    # Example from NLTK documentation
    reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']
    reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which', 'guarantees', 'the', 'military', 'forces', 'always', 'being', 'under', 'the', 'command', 'of', 'the', 'Party']
    reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the', 'army', 'always', 'to', 'heed', 'the', 'directions', 'of', 'the', 'party']
    references_list = [reference1, reference2, reference3]
    
    hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'Party']
    hypothesis2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was', 'interested', 'in', 'world', 'history'] # Completely different

    # NLTK's sentence_bleu with smoothing_function=None (or default) can be a bit different
    # if any precision is 0. This implementation is a basic version.
    # For BLEU-4, weights = [0.25, 0.25, 0.25, 0.25]
    bleu4_weights = [0.25, 0.25, 0.25, 0.25]
    
    score1 = bleu_score(references_list, hypothesis1, max_n=4, weights=bleu4_weights)
    print(f"BLEU-4 for hypothesis1: {score1:.4f}") 
    # NLTK with SmoothingFunction().method1 gives ~0.5045 for this example.
    # Without smoothing, if any P_n is 0, score can be 0. This implementation might yield 0 if a P_n is 0.
    # Let's test individual precisions for hypothesis1
    # p1 = modified_precision(references_list, hypothesis1, 1) # 15/18
    # p2 = modified_precision(references_list, hypothesis1, 2) # 8/17
    # p3 = modified_precision(references_list, hypothesis1, 3) # 2/16
    # p4 = modified_precision(references_list, hypothesis1, 4) # 1/15
    # print(f"P1: {p1}, P2: {p2}, P3: {p3}, P4: {p4}")
    # If any of these are 0, the geometric mean (and thus BLEU) will be 0 without smoothing.
    # The current implementation adds epsilon, so log(0) is avoided, but exp(log(small_num)) is small.

    score2 = bleu_score(references_list, hypothesis2, max_n=4, weights=bleu4_weights)
    print(f"BLEU-4 for hypothesis2: {score2:.4f}") # Expected to be very low or 0

    # Test with a perfect match
    score_perfect = bleu_score([reference1], reference1, max_n=4, weights=bleu4_weights)
    print(f"BLEU-4 for perfect match: {score_perfect:.4f}") # Expected to be 1.0 (or close due to epsilon)

    # Test with empty hypothesis
    score_empty_hyp = bleu_score(references_list, [], max_n=4, weights=bleu4_weights)
    print(f"BLEU-4 for empty hypothesis: {score_empty_hyp:.4f}") # Expected 0.0

    # Test with empty references (should also be 0 or error depending on handling)
    try:
        score_empty_ref = bleu_score([[]], hypothesis1, max_n=4, weights=bleu4_weights)
        print(f"BLEU-4 for empty reference: {score_empty_ref:.4f}") # Expected 0.0
    except Exception as e:
        print(f"Error with empty reference: {e}")

    # Test single reference string (should be auto-wrapped)
    single_ref_test = ['this', 'is', 'a', 'test']
    hyp_test = ['this', 'is', 'test']
    score_single_ref = bleu_score(single_ref_test, hyp_test, max_n=2, weights=[0.5,0.5])
    print(f"BLEU-2 for single ref string input: {score_single_ref:.4f}")

    # --- Test Batch Iterator ---
    print("\n--- Testing Batch Iterator ---")
    X_data = np.arange(20).reshape(10, 2)
    y_data = np.arange(10)
    batch_gen = batch_iterator((X_data, y_data), batch_size=3, shuffle=True)
    print("Batches:")
    for x_b, y_b in batch_gen:
        print("X_batch shape:", x_b.shape, "y_batch shape:", y_b.shape)
        assert x_b.shape[0] == y_b.shape[0]
        assert x_b.shape[0] <= 3

    print("\nUtils testing done.")