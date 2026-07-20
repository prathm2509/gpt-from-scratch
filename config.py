"""All hyperparameters in one place. Defaults are tiny so it trains on a CPU
in a few minutes. Scale these up (n_layer / n_head / n_embd / block_size) only
after the small model works end to end."""
from dataclasses import dataclass
import torch


@dataclass
class GPTConfig:
    vocab_size: int = 65      # overwritten at runtime from the tokenizer
    block_size: int = 64      # context length = max number of positions
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 3000
    eval_interval: int = 250
    eval_iters: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337
