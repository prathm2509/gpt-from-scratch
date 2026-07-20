"""Interactive playground: type a prompt, watch your GPT continue it.

    python prompt.py
    python prompt.py --tokens 500 --temp 0.6
    python prompt.py --top-k 20

Temperature < 1 makes the model more confident/repetitive, > 1 more chaotic.
top-k restricts sampling to the k most likely characters. Blank line to quit."""
import argparse

import torch
import torch.nn.functional as F

from data import load_data
from model import GPT
from config import GPTConfig, TrainConfig


@torch.no_grad()
def sample(model, idx, max_new_tokens, block_size, temperature=1.0, top_k=None):
    """Same loop as GPT.generate, plus temperature and top-k for nicer output."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=300, help="characters to generate")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    tcfg = TrainConfig()
    tok, _, _ = load_data("input.txt")
    cfg = GPTConfig(vocab_size=tok.vocab_size)

    model = GPT(cfg).to(tcfg.device)
    model.load_state_dict(torch.load("ckpt.pt", map_location=tcfg.device)["model"])
    model.eval()

    n = sum(p.numel() for p in model.parameters())
    print(f"loaded ckpt.pt | {n:,} params | device={tcfg.device}")
    print(f"tokens={args.tokens}  temp={args.temp}  top_k={args.top_k}")
    print("Type a prompt and press Enter. Blank line (or 'quit') to exit.\n")

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip().lower() in ("", "quit", "exit"):
            break

        # The char tokenizer only knows the 65 characters in the corpus - anything
        # else would KeyError, so drop it and say so.
        unknown = sorted(set(prompt) - set(tok.stoi))
        if unknown:
            print(f"  [not in vocab, skipped: {''.join(unknown)!r}]")
            prompt = "".join(c for c in prompt if c in tok.stoi)
        if not prompt:
            print("  [nothing left to feed the model]\n")
            continue

        idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=tcfg.device)
        out = sample(model, idx, args.tokens, cfg.block_size, args.temp, args.top_k)

        print("-" * 64)
        print(tok.decode(out[0].tolist()))
        print("-" * 64 + "\n")


if __name__ == "__main__":
    main()
