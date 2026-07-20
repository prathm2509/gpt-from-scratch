"""Load the corpus, tokenize it once, and hand out random batches.

A batch is (x, y) where y is x shifted one position: for every position the
target is simply "the next token". That single fact is the whole training
signal for a language model."""
import torch
from tokenizer import CharTokenizer


def load_data(path="input.txt"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return tok, data[:n], data[n:]          # tokenizer, train, val


def get_batch(data, block_size, batch_size, device="cpu"):
    # pick batch_size random starting points
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
