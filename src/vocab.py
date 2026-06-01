import re
import json
from collections import Counter


class Vocab:
    def __init__(self, min_freq=2, max_size=10000):
        self.min_freq = min_freq
        self.max_size = max_size

        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.special_tokens = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
        ]

        self.stoi = {}
        self.itos = []

    def tokenize(self, text):
        text = str(text).lower().strip()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-z0-9#@'!?.,]+", " ", text)
        return text.split()

    def build(self, captions):
        counter = Counter()
        for caption in captions:
            counter.update(self.tokenize(caption))

        words = [
            word for word, freq in counter.most_common()
            if freq >= self.min_freq
        ]

        words = words[: self.max_size - len(self.special_tokens)]

        self.itos = self.special_tokens + words
        self.stoi = {word: idx for idx, word in enumerate(self.itos)}

    def encode(self, text, max_len=40):
        tokens = [self.bos_token] + self.tokenize(text)[: max_len - 2] + [self.eos_token]
        return [self.stoi.get(tok, self.stoi[self.unk_token]) for tok in tokens]

    def decode(self, ids):
        words = []
        for idx in ids:
            word = self.itos[int(idx)]
            if word == self.eos_token:
                break
            if word not in [self.pad_token, self.bos_token]:
                words.append(word)
        return " ".join(words)

    @property
    def pad_id(self):
        return self.stoi[self.pad_token]

    @property
    def bos_id(self):
        return self.stoi[self.bos_token]

    @property
    def eos_id(self):
        return self.stoi[self.eos_token]

    def __len__(self):
        return len(self.itos)

    def save(self, path):
        obj = {
            "min_freq": self.min_freq,
            "max_size": self.max_size,
            "itos": self.itos,
            "stoi": self.stoi,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        vocab = cls(
            min_freq=obj.get("min_freq", 2),
            max_size=obj.get("max_size", 10000),
        )
        vocab.itos = obj["itos"]
        vocab.stoi = obj["stoi"]
        return vocab
