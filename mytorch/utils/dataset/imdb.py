import os
import csv
import json
import hashlib
from mytorch.backend import xp


class IMDBDataset:
    def __init__(self, root, tokenizer, max_len=200, train=True, from_csv=True, use_cache=True):
        self.root = root
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train = train
        self.from_csv = from_csv
        self.use_cache = use_cache

        if self.from_csv:
            self._load_data()
        else:
            raise NotImplementedError("Only CSV format is currently supported")

    def _load_data(self):
        csv_filename = "imdb_train.csv" if self.train else "imdb_test.csv"
        csv_path = os.path.join(self.root, csv_filename)

        # ==== ✅ Generate a unique cache filename ====
        config = {
            "train": self.train,
            "max_len": self.max_len,
            "vocab_size": self.tokenizer.max_vocab_size,
            "min_freq": self.tokenizer.min_freq,
            "pad_token": self.tokenizer.pad_token,
            "unk_token": self.tokenizer.unk_token,
        }

        # Generate a unique identifier using a hash
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]

        cache_filename = f"imdb_{'train' if self.train else 'test'}_{config_hash}.npz"
        cache_path = os.path.join(self.root, cache_filename)

        # ==== ✅ Load from cache if available ====
        if self.use_cache and os.path.exists(cache_path):
            print(f"📦 Loading data from cache: {cache_path}")
            data = xp.load(cache_path, allow_pickle=True)
            self.input_ids = data['input_ids']
            self.labels = data['labels']
            print(f"✅ Loaded: {len(self.labels)} samples")
            return

        # ==== ✅ Load data from CSV ====
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")

        print(f"📂 Loading data from CSV: {csv_path}")
        input_ids = []
        labels = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                text = row[2]  # Third column is the review
                label_str = row[3].strip().lower()
                label = 1 if label_str == 'pos' else 0

                ids = self.tokenizer.encode(text, self.max_len)
                input_ids.append(xp.array(ids, dtype=xp.int32))
                labels.append(label)

        self.input_ids = xp.stack(input_ids)
        self.labels = xp.array(labels, dtype=xp.int32)

        print(f"✅ Loaded: {len(self.labels)} samples")

        # ==== ✅ Save to cache ====
        if self.use_cache:
            print(f"💾 Saving cache file: {cache_path}")
            xp.savez_compressed(
                cache_path, input_ids=self.input_ids, labels=self.labels)
            print("✅ Cache saved")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]
