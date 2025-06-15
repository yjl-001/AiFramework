import re
import csv
from collections import Counter


class Tokenizer:
    def __init__(self, max_vocab_size=20000, min_freq=2, pad_token="<PAD>", unk_token="<UNK>"):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.word2idx = {}
        self.idx2word = []

    def clean_text(self, text: str) -> str:
        # 简单清洗：小写、去除 HTML 标签、标点等
        text = text.lower()
        text = re.sub(r"<.*?>", "", text)  # 去除 HTML 标签
        text = re.sub(r"[^a-z0-9\s]", "", text)  # 去除标点
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

        # 添加特殊 token
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

    def build_vocab_from_csv(self, csv_path, text_column=2):
        texts = []
        with open(csv_path, 'r', encoding='ISO-8859-1') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                texts.append(row[text_column])
        self.build_vocab(texts)

    def print_vocab(self, limit=1000):
        print(f"Vocabulary (showing first {limit} words of {self.vocab_size()}):")
        if limit > self.vocab_size():
            limit = self.vocab_size()
        for i, word in enumerate(self.idx2word[:limit]):
            print(f"{i}: {word} (ID: {self.word2idx[word]})")
