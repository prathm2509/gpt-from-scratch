import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# MILESTONE 1 - baseline you can run TODAY. No attention: each token predicts
# the next purely from an embedding lookup. Its only job is to prove the whole
# pipeline (data -> loss -> backprop -> sample) works. On tiny-shakespeare the
# loss falls from ~ln(vocab) to ~2.45 and the samples are gibberish. Expected.
# ---------------------------------------------------------------------------
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, vocab_size)
        # Small init so logits start near zero and the initial loss lands on
        # ln(vocab). nn.Embedding's default N(0,1) would start the model
        # "confidently wrong", inflating the loss and blunting sanity check 1.
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        logits = self.token_emb(idx)                      # (B, T, vocab)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -1:])                 # only last token matters
            probs = F.softmax(logits[:, -1, :], dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# ===========================================================================
# YOUR BUILD - the GPT, bottom-up. Implement each class, deleting its
# `raise NotImplementedError` as you go. Run the sanity checks in train.py
# after each rung. The GPT-specific deltas from the 2017 Transformer are
# called out in CAPS in the docstrings - those are the things that are new.
# ===========================================================================

class Head(nn.Module):
    """MILESTONE 2 - one CAUSAL self-attention head.
      - key, query, value: three Linear(n_embd, head_size, bias=False).
      - scores = q @ k.transpose(-2,-1) * head_size**-0.5        # (B,T,T)
      - CAUSAL MASK: scores = scores.masked_fill(tril[:T,:T]==0, float('-inf'))
        so each position attends only to itself and earlier ones. Register
        `tril` as a buffer: self.register_buffer('tril', torch.tril(torch.ones(bs,bs)))
      - softmax(dim=-1) -> dropout -> @ v.
    This causal mask is the single most important difference from the encoder
    self-attention you may have seen before."""
    def __init__(self, cfg, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(cfg.n_embd, head_size, bias = False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias = False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias = False)
        self.dropout = nn.Dropout(cfg.dropout)
        # A buffer, not a Parameter: the mask is a constant and is never trained,
        # but it still needs to follow the model across .to(device).
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2,-1) * self.head_size** -0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float("-inf"))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)
        return wei @ v                                          # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """MILESTONE 3 - n_head heads in parallel, concatenated.
      - head_size = n_embd // n_head.
      - run the heads, concat their outputs on the last dim back to n_embd.
      - output projection: Linear(n_embd, n_embd) then dropout.
    Start with nn.ModuleList of Head; batch them into one matmul later."""
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)      # w_o: mixes across heads
        self.dropout = nn.Dropout(cfg.dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        out = self.dropout(out)
        return out 
        


class FeedForward(nn.Module):
    """MILESTONE 3 - position-wise MLP applied to every token independently.
      Linear(n_embd, 4*n_embd) -> GELU -> Linear(4*n_embd, n_embd) -> dropout.
    DELTA: activation is GELU, not ReLU."""
    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class Block(nn.Module):
    """MILESTONE 3 - one Transformer block, PRE-LN (the depth-stability delta):
          x = x + self.attn(self.ln1(x))
          x = x + self.ffwd(self.ln2(x))
    LayerNorm is applied BEFORE each sublayer, and the residual is added to the
    ORIGINAL x. (The 2017 paper did LN *after* - post-LN - which is harder to
    train deep.) You need: ln1, attn (MultiHeadAttention), ln2, ffwd."""
    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class GPT(nn.Module):
    """MILESTONE 4 - assemble the full model.
      - token_emb : Embedding(vocab_size, n_embd)
      - pos_emb   : Embedding(block_size, n_embd)   # LEARNED positions (delta)
      - blocks    : nn.Sequential of n_layer Block
      - ln_f      : final LayerNorm                 # DELTA: GPT-2 adds this
      - lm_head   : Linear(n_embd, vocab_size, bias=False)
      - WEIGHT TYING: self.lm_head.weight = self.token_emb.weight
      - SCALED INIT (optional but real): init the residual output projections
        (attn out-proj, ffwd 2nd linear) with std = 0.02 / (2*n_layer)**0.5.

    forward(idx, targets=None):
      B, T = idx.shape
      tok = token_emb(idx)                                    # (B,T,C)
      pos = pos_emb(torch.arange(T, device=idx.device))       # (T,C)
      x = tok + pos  -> blocks -> ln_f -> lm_head             # logits (B,T,vocab)
      if targets: cross_entropy(logits.view(B*T,-1), targets.view(B*T))

    generate(idx, max_new_tokens):
      each step CROP context to the last block_size tokens (positions only go
      up to block_size!), forward, softmax last step, multinomial sample, append."""
    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError

    def forward(self, idx, targets=None):
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        raise NotImplementedError
