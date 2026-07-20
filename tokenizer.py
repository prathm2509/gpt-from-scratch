"""Character-level tokenizer: the simplest possible stand-in for BPE.

Every unique character in the corpus becomes one token. Start here so the
model works before you complicate tokenization; swap in a real BPE tokenizer
later (you already understand how it's built and applied)."""


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)
