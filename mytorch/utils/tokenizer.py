import re
import csv
import os
import json
import hashlib
from collections import Counter


class Tokenizer:
    def __init__(self, max_vocab_size=20000, min_freq=5, pad_token="<PAD>", unk_token="<UNK>"):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.word2idx = {}
        self.idx2word = []

    def clean_text(self, text: str) -> str:
        """
        Perform basic text cleaning: lowercase, strip HTML tags, remove punctuation.
        """
        text = text.lower()
        text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
        text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove punctuation
        return text

    def tokenize(self, text: str) -> list[str]:
        text = self.clean_text(text)
        return text.strip().split()

    def build_vocab(self, texts: list[str]):
        print('Building vocabulary...')
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

        # Add special tokens
        vocab = [self.pad_token, self.unk_token]
        for word, freq in counter.most_common(self.max_vocab_size - len(vocab)):
            if freq >= self.min_freq:
                vocab.append(word)

        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = vocab
        print(f"Vocabulary built with {len(self.word2idx)} words.")

    def encode(self, text: str, max_len: int) -> list[int]:
        tokens = self.tokenize(text)
        ids = [self.word2idx.get(token, self.word2idx[self.unk_token])
               for token in tokens]

        # Padding or truncation
        if len(ids) < max_len:
            ids += [self.word2idx[self.pad_token]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join([self.idx2word[i] for i in ids if i < len(self.idx2word)])

    def vocab_size(self):
        return len(self.word2idx)

    def _get_vocab_cache_path(self, csv_path: str, text_column: int = 2) -> str:
        config = {
            "csv_path": os.path.abspath(csv_path),
            "text_column": text_column,
            "max_vocab_size": self.max_vocab_size,
            "min_freq": self.min_freq,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
        }
        config_str = json.dumps(config, sort_keys=True)
        hash_id = hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]
        filename = f"vocab_{hash_id}.vocab"  # Changed from .json to .vocab
        return os.path.join(os.path.dirname(csv_path), filename)

    def build_vocab_from_csv(self, csv_path, text_column=2, encoding='utf-8'):
        cache_path = self._get_vocab_cache_path(csv_path, text_column)

        # Load from cache if available
        if os.path.exists(cache_path):
            print(f"📦 Loading vocabulary from cache: {cache_path}")
            self.load_vocab(cache_path)
            return

        # Otherwise, build from CSV
        print(f"📂 Building vocabulary from CSV: {csv_path}")
        texts = []
        with open(csv_path, 'r', encoding=encoding) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                texts.append(row[text_column])

        self.build_vocab(texts)

        # Save to cache
        print(f"💾 Saving vocabulary to cache: {cache_path}")
        self.save_vocab(cache_path)

    def print_vocab(self, limit=1000):
        print(
            f"Vocabulary (showing first {limit} words of {self.vocab_size()}):")
        if limit > self.vocab_size():
            limit = self.vocab_size()
        for i, word in enumerate(self.idx2word[:limit]):
            print(f"{i}: {word} (ID: {self.word2idx[word]})")

    def save_vocab(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            for word in self.idx2word:
                f.write(f"{word}\n")
        print(f"✅ Vocabulary saved to {filepath}")

    def load_vocab(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.idx2word = [line.strip() for line in f if line.strip()]
            self.word2idx = {word: idx for idx,
                             word in enumerate(self.idx2word)}
        print(
            f"✅ Vocabulary loaded from {filepath} ({len(self.idx2word)} words)")
