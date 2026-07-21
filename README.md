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

## Beyond GPT-2: RoPE
`cfg.use_rope` (default on) swaps learned absolute positional embeddings for
**rotary position embeddings**, the scheme used by Llama, Mistral and most
models since. Instead of adding a position vector at the bottom of the network,
each attention layer rotates `q` and `k` by an angle proportional to position —
never `v`, since position should decide *who attends to whom*, not *what*
information flows.

Because rotations compose (`Rₘᵀ Rₙ = Rₙ₋ₘ`), the resulting `q·k` score depends
only on the **distance** `m − n`, not on absolute positions. Relative position
falls out of the geometry with zero learned parameters, and there is no lookup
table to run out of, so context length stops being a hard architectural cap.

```
python train.py --rope off --iters 5000 --out ckpt_norope.pt
python train.py --rope on  --iters 5000 --out ckpt_rope.pt
```

## Results (tiny-shakespeare, char-level)

| model | iters | val loss | output |
|---|---|---|---|
| uniform guessing | — | 4.174 | — |
| bigram baseline | 3k | 2.49 | `Whencoughefran` |
| GPT, learned pos emb | 3k | 1.79 | speaker tags, real words, line breaks |
| GPT, learned pos emb | 10k | 1.58 | — |
| **GPT + RoPE** | **5k** | **1.57** | — |

Verified along the way: init loss 4.185 against ln(65) = 4.174; single-batch
overfit reaches 0.09 (the bigram floors at 2.3, having no context); causality
holds across all layers.

### Experiment log

Kept honestly, including what did not work.

**Batched attention** — replacing the `ModuleList` head loop with one fused
qkv projection: identical parameters and outputs, 276.7 → 188.2 ms/step
(1.47×). No effect on loss; it buys iterations per hour.

**Training longer** — 3k → 10k iterations took val 1.786 → 1.581 with no
architectural change. The model had simply been stopped early, not converged.

**block_size 64→128 + dropout 0.1→0.2 + cosine LR** — *regression*, val
1.5811 → 1.5931. Three changes in one run, so attribution is impossible;
dropout is the leading suspect, since the train-val gap was only 0.21 and
the model is capacity-bound at ~800K params. Dropout was reverted. Lesson:
bundle changes only when you are willing to give up knowing which one acted.

**RoPE vs learned positional embeddings** — controlled A/B, 5k iters, same
seed, same data order, same fixed eval batches, one variable changed:

| | train | val | gap |
|---|---|---|---|
| learned pos emb | 1.4801 | 1.6666 | 0.187 |
| RoPE | 1.3594 | **1.5696** | 0.210 |

RoPE wins by **0.097 val**, far more than the 0.02–0.05 typically reported at
scale. Three plausible reasons, all specific to a small model: capacity is the
binding constraint here and RoPE spends none of it learning what position
means; position is re-injected at every layer rather than added once at the
bottom and expected to survive four layers of mixing; and at 5k iterations a
learned table is still being learned while RoPE works from step 0. RoPE also
overfits faster (larger gap), so dropout may earn its place at longer runs.

*Measurement note:* evaluation switched from fresh random batches to a fixed
set drawn once, because random batches carried noise (~0.02) comparable to the
effects being measured. Numbers before and after that change are not strictly
comparable.

## Files
| file | role |
|---|---|
| `config.py` | all hyperparameters |
| `tokenizer.py` | char-level encode/decode |
| `data.py` | corpus → batches of (x, next-token) |
| `model.py` | bigram baseline, Head, MultiHeadAttention, FeedForward, Block, GPT |
| `train.py` | training loop + the two sanity checks |
| `sample.py` | generate from a checkpoint |
| `prompt.py` | interactive: type a prompt, watch the model continue it |
| `prepare_data.py` | fetch the corpus |

## Try it
```
python prompt.py                 # type a prompt at the >>> and see it continue
python prompt.py --temp 0.5      # more conservative sampling
python prompt.py --top-k 10      # only sample from the 10 likeliest characters
```

## Credits
Structure follows Andrej Karpathy's "Let's build GPT from scratch" and
[nanoGPT](https://github.com/karpathy/nanoGPT). Architecture from Vaswani et al.,
*Attention Is All You Need* (2017); Radford et al., GPT-2 (2019); Brown et al.,
*Language Models are Few-Shot Learners* (2020). Corpus is the tiny-shakespeare
dataset from Karpathy's char-rnn.
