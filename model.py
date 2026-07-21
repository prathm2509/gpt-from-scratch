import torch
import torch.nn as nn
from torch.nn import functional as F

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

class Head(nn.Module):
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
    """All heads in one batched matmul, instead of a Python loop over Head.

    Mathematically identical to the ModuleList version (same parameter count),
    but the head dimension becomes a tensor axis rather than a for-loop, so the
    whole thing is 4 matmuls instead of 4*3+1. Head is kept above for reference.

    Shapes, with C = n_embd, h = n_head, hs = head_size:
        x            (B, T, C)
        c_attn(x)    (B, T, 3C)          one fused q/k/v projection
        q, k, v      (B, T, C)  each     -> view/transpose -> (B, h, T, hs)
        att          (B, h, T, T)        masked, softmaxed
        att @ v      (B, h, T, hs)       -> transpose/view -> (B, T, C)  = the concat
    """
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_size = cfg.n_embd // cfg.n_head

        # Fused q, k, v in a single matmul (3C wide, split apart in forward).
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)      # w_o: mixes across heads
        self.attn_dropout = nn.Dropout(cfg.dropout)        # on the attention weights
        self.dropout = nn.Dropout(cfg.dropout)             # on the residual output
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)          # 3x (B, T, C)
        # (B, T, C) -> (B, T, h, hs) -> (B, h, T, hs): head becomes a batch axis
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        att = q @ k.transpose(-2, -1) * self.head_size ** -0.5      # (B, h, T, T)
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v                                                 # (B, h, T, hs)
        # transpose back and flatten the head axis - this IS the concatenation
        y = y.transpose(1, 2).contiguous().view(B, T, C)            # (B, T, C)
        return self.dropout(self.proj(y))



class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout)
        )

    def forward(self, x):
        return self.net(x)
    
        


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.attn = MultiHeadAttention(cfg)
        self.ffwd = FeedForward(cfg) 

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias = False)
        self.lm_head.weight = self.token_emb.weight

        # GPT-2 init. Not cosmetic: nn.Embedding defaults to N(0,1), and because
        # of weight tying that same big matrix IS lm_head - so logits start huge
        # and the initial loss lands ~20x above ln(vocab).
        self.apply(self._init_weights)
        # Scaled residual init: every layer adds into the residual stream, so its
        # variance grows with depth. Shrink the projections that write into it by
        # 1/sqrt(2*n_layer) (2 per block: attn out-proj and ffwd's 2nd linear).
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layer) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            nxt = torch.multinomial(probs,num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        
        return idx    
