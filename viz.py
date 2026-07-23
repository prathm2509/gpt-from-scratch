"""Look inside the forward pass on a prompt - turn the black box translucent.

Shows two things the model computes but usually hides:
  1. the top-k next-token predictions at every position (watch it "decide")
  2. the attention pattern for one layer/head - which earlier characters each
     position actually looks at

    python viz.py --prompt "ROMEO:" --layer 5 --head 0
    python viz.py --prompt "To be or not to " --ckpt /content/drive/MyDrive/ckpt_scaled_10k.pt

No change to model.py: attention is recomputed from each module's own weights and
buffers via a forward pre-hook, so what you see is exactly what the model did."""
import argparse
import torch
import torch.nn.functional as F

from data import load_data
from model import GPT, apply_rope
from config import GPTConfig, TrainConfig


def attn_for_layer(attn, x):
    """Recompute one MultiHeadAttention's weights from its input x - mirrors the
    real forward up to the softmax, using the module's actual weights/buffers."""
    B, T, C = x.shape
    q, k, _ = attn.c_attn(x).split(attn.n_embd, dim=2)
    hs, nh = attn.head_size, attn.n_head
    q = q.view(B, T, nh, hs).transpose(1, 2)
    k = k.view(B, T, nh, hs).transpose(1, 2)
    if attn.use_rope:
        q = apply_rope(q, attn.rope_cos[:T], attn.rope_sin[:T])
        k = apply_rope(k, attn.rope_cos[:T], attn.rope_sin[:T])
    att = q @ k.transpose(-2, -1) * hs ** -0.5
    att = att.masked_fill(attn.tril[:T, :T] == 0, float("-inf"))
    return F.softmax(att, dim=-1)                 # (B, nh, T, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--layer", type=int, default=-1, help="which block's attention to show")
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    tcfg = TrainConfig()
    tok, _, _ = load_data("input.txt")
    cfg = GPTConfig(vocab_size=tok.vocab_size)
    model = GPT(cfg).to(tcfg.device)

    try:
        sd = torch.load(args.ckpt, map_location=tcfg.device, weights_only=False)["model"]
        model.load_state_dict(sd)
        print(f"loaded {args.ckpt}  ({cfg.n_layer} layers, {cfg.n_head} heads)")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[no matching checkpoint at {args.ckpt} - UNTRAINED model, patterns "
              f"will be flat/uniform. That itself is worth seeing.]\n   ({type(e).__name__})")
    model.eval()

    def lab(i):                                   # printable label for one token id
        c = tok.decode([i])
        return {"\n": "\\n", " ": "_"}.get(c, c)

    # capture the input to each block's attention (the normed x) via a pre-hook
    captured = {}
    for i, blk in enumerate(model.blocks):
        blk.attn.register_forward_pre_hook(
            lambda mod, inp, idx=i: captured.__setitem__(idx, inp[0].detach()))

    ids = tok.encode(args.prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=tcfg.device)
    with torch.no_grad():
        logits, _ = model(idx)
    chars = [lab(i) for i in ids]

    print(f"\nprompt: {args.prompt!r}  ->  {len(ids)} tokens: {chars}")

    # 1) top-k next-token predictions at each position
    print("\n=== what each position predicts comes NEXT ===")
    probs = F.softmax(logits[0], dim=-1)          # (T, vocab)
    for t in range(len(ids)):
        top = torch.topk(probs[t], args.topk)
        preds = "  ".join(f"{lab(j.item()):>3}={p:.2f}" for p, j in zip(top.values, top.indices))
        print(f"  pos {t:2d} {chars[t]:>3} |  {preds}")

    # 2) attention pattern for the chosen layer/head
    L = args.layer % len(model.blocks)
    att = attn_for_layer(model.blocks[L].attn, captured[L])[0, args.head]   # (T, T)
    print(f"\n=== attention: layer {L}, head {args.head} - where each position LOOKS ===")
    for t in range(len(ids)):
        row = att[t, :t + 1]
        top = torch.topk(row, min(3, t + 1))
        looks = "  ".join(f"{chars[j]}({row[j]:.2f})" for j in top.indices)
        print(f"  pos {t:2d} {chars[t]:>3} attends to:  {looks}")


if __name__ == "__main__":
    main()
