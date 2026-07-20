"""Generate text from a trained checkpoint:  python sample.py
Switch the model class to GPT once you've built and trained it."""
import torch
from data import load_data
from model import BigramLanguageModel   # , GPT
from config import GPTConfig, TrainConfig


def main():
    tcfg = TrainConfig()
    tok, _, _ = load_data("input.txt")
    model = BigramLanguageModel(tok.vocab_size).to(tcfg.device)
    # gcfg = GPTConfig(vocab_size=tok.vocab_size); model = GPT(gcfg).to(tcfg.device)

    ckpt = torch.load("ckpt.pt", map_location=tcfg.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ctx = torch.zeros((1, 1), dtype=torch.long, device=tcfg.device)
    out = model.generate(ctx, max_new_tokens=500)[0].tolist()
    print(tok.decode(out))


if __name__ == "__main__":
    main()
