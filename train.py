"""Training loop + the two correctness checks.

  python train.py            # train (prints the init-loss sanity check first)
  python train.py --overfit  # train on ONE batch; loss must fall to ~0
  python train.py --iters N  # override number of iterations

Flip `model =` from BigramLanguageModel to GPT once you've built the GPT."""
import argparse
import math
import torch

from data import load_data, get_batch
from model import BigramLanguageModel   # , GPT
from config import GPTConfig, TrainConfig


@torch.no_grad()
def estimate_loss(model, splits, tcfg, block_size):
    out = {}
    model.eval()
    for name, data in splits.items():
        losses = torch.zeros(tcfg.eval_iters)
        for k in range(tcfg.eval_iters):
            x, y = get_batch(data, block_size, tcfg.batch_size, tcfg.device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overfit", action="store_true",
                    help="train on one fixed batch; loss should approach 0")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None,
                    help="override learning rate (bigram baseline wants ~1e-2; "
                         "the deep GPT wants the 3e-4 default)")
    args = ap.parse_args()

    tcfg = TrainConfig()
    torch.manual_seed(tcfg.seed)
    tok, train_data, val_data = load_data("input.txt")
    gcfg = GPTConfig(vocab_size=tok.vocab_size)

    model = BigramLanguageModel(tok.vocab_size).to(tcfg.device)
    # model = GPT(gcfg).to(tcfg.device)        # <- switch to this after milestone 4
    print(f"device={tcfg.device}  vocab={tok.vocab_size}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    # --- SANITY CHECK 1: an untrained model should be guessing uniformly, so
    #     the initial loss should be about ln(vocab_size). Big mismatch => a
    #     wiring bug in logits / softmax / weight-tying. ---
    x, y = get_batch(train_data, gcfg.block_size, tcfg.batch_size, tcfg.device)
    _, loss0 = model(x, y)
    print(f"init loss {loss0.item():.3f}  |  expected ~ln(V) = {math.log(tok.vocab_size):.3f}")

    lr = args.lr or tcfg.learning_rate
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    max_iters = args.iters or tcfg.max_iters
    print(f"lr={lr}  iters={max_iters}")

    # --- SANITY CHECK 2: overfit a single batch. A model with context and
    #     enough capacity can simply memorize it, so loss -> ~0. If your GPT
    #     can't, forward/backward has a bug.
    #     NOTE: this test does NOT apply to the bigram baseline - it has no
    #     context, so it can only ever learn P(next|current) and floors at that
    #     conditional entropy (~2.3 here). That is a capacity limit, not a bug. ---
    if args.overfit:
        xb, yb = get_batch(train_data, gcfg.block_size, tcfg.batch_size, tcfg.device)
        for i in range(max_iters):
            _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if i % 100 == 0:
                print(f"iter {i:5d}  loss {loss.item():.4f}")
        print("overfit done.")
        print("  GPT      -> expect ~0.0  (it can memorize one batch; if not, you have a bug)")
        print("  bigram   -> expect ~2.3  (no context, so it cannot memorize. not a bug)")
        return

    splits = {"train": train_data, "val": val_data}
    for i in range(max_iters):
        if i % tcfg.eval_interval == 0:
            l = estimate_loss(model, splits, tcfg, gcfg.block_size)
            print(f"iter {i:5d}  train {l['train']:.4f}  val {l['val']:.4f}")
        x, y = get_batch(train_data, gcfg.block_size, tcfg.batch_size, tcfg.device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    torch.save({"model": model.state_dict(), "stoi": tok.stoi, "itos": tok.itos}, "ckpt.pt")
    print("saved ckpt.pt")


if __name__ == "__main__":
    main()
