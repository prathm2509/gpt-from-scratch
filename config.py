"""All hyperparameters in one place. Defaults are tiny so it trains on a CPU
in a few minutes. Scale these up (n_layer / n_head / n_embd / block_size) only
after the small model works end to end."""
from dataclasses import dataclass
import torch


@dataclass
class GPTConfig:
    vocab_size: int = 65      # overwritten at runtime from the tokenizer
    # Scaled-up config (~10.7M params) - the nanoGPT tiny-shakespeare recipe,
    # which reaches ~1.48 val. Intended for a GPU: ~9-16 h on CPU, minutes on a
    # T4. The earlier 4-layer / 128-dim setup trained on CPU; git history has it.
    block_size: int = 256     # context length = max number of positions
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    # Dropout back to 0.2: at ~10.7M params on a 1 MB corpus the binding
    # constraint flips from capacity (where 0.2 hurt the 800K model) to
    # overfitting, so regularization now earns its place.
    dropout: float = 0.2
    # Position encoding: RoPE rotates q/k inside attention (relative position,
    # zero parameters) instead of adding a learned absolute embedding at the
    # bottom. Flip to False to A/B against learned positional embeddings.
    use_rope: bool = True
    rope_theta: float = 10000.0


@dataclass
class TrainConfig:
    batch_size: int = 64      # larger batch: fine on a GPU, stabilizes training
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 40      # cheap on a GPU; a steadier loss estimate
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337
