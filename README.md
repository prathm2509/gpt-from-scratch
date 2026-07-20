# GPT from scratch

A small GPT-2-style decoder-only Transformer, built from the ground up — the
same architecture as GPT-3, tiny enough to train on a laptop CPU. The pipeline
(data, tokenizer, training loop, a bigram baseline) is provided so you can focus
your from-scratch effort on the model itself.

## Setup
```
pip install torch          # numpy is already installed
python prepare_data.py     # downloads input.txt (tiny-shakespeare, ~1 MB)
```

## The ladder — build in this order, and RUN at every rung
Start each rung only once the previous one runs. Never add attention before the
plumbing is proven.

- **0 · Pipeline (already works).** `data.py`, `tokenizer.py`, and the
  `BigramLanguageModel` baseline in `model.py`.
  ```
  python train.py --lr 1e-2    # bigram loss falls 4.17 -> ~2.49
  python sample.py             # generates gibberish — expected, it has no context
  ```
  The baseline needs `--lr 1e-2`: it's a shallow model, and the `3e-4` default in
  `config.py` is tuned for the deep GPT you're about to build.
- **1 · Sanity checks.** Build the habit before adding complexity.
  ```
  python train.py --lr 1e-2              # "init loss" should print ~ ln(vocab) = 4.17
  python train.py --overfit --lr 1e-2     # train on ONE batch
  ```
  **Init loss** should land on `ln(vocab)`, because an untrained model guesses
  uniformly. Way off ⇒ a wiring bug in logits/softmax/weight-tying.

  **Overfit** drives one batch to ~0 *for the GPT* — it has the context and
  capacity to memorize, so if it can't, forward/backward is broken. It does **not**
  apply to the bigram: with no context it can only learn `P(next|current)` and
  floors at that conditional entropy (~2.3). That's a capacity limit, not a bug.
- **2 · One causal attention head** — `model.py : Head`.
- **3 · Multi-head + block** — `MultiHeadAttention`, `FeedForward`, `Block`
  (Pre-LN, GELU).
- **4 · Full GPT** — `GPT` (learned positional embeddings, final LayerNorm,
  weight tying, scaled init). Then flip `BigramLanguageModel` → `GPT` in
  `train.py` and `sample.py`.
- **5 · Scale up** — raise `n_layer / n_head / n_embd / block_size` in
  `config.py` and retrain.

## GPT-specific deltas from the 2017 Transformer (spelled out in the stubs)
- decoder-only, **causal** self-attention (the mask)
- **learned** positional embeddings (not sinusoidal) → these cap context length
- **Pre-LN** (LayerNorm before each sublayer)
- **GELU** activation (not ReLU)
- **weight tying** (token embedding == output projection)
- **scaled residual init** (std 0.02 / √(2·n_layer))

## Rough targets (tiny-shakespeare, char-level)
| model | val loss | output |
|---|---|---|
| bigram baseline | ~2.45 | gibberish |
| small GPT (4 layer, 128 dim) | ~1.5 | semi-coherent English |

## Files
| file | role |
|---|---|
| `config.py` | all hyperparameters |
| `tokenizer.py` | char-level encode/decode |
| `data.py` | corpus → batches of (x, next-token) |
| `model.py` | bigram baseline **+ your GPT to build** |
| `train.py` | training loop + the two sanity checks |
| `sample.py` | generate from a checkpoint |
| `prepare_data.py` | fetch the corpus |
