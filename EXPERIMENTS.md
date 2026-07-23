# Experiments

A running log of what was tried, what it bought, and what did not work — kept
honestly, because the failures and the negative results taught as much as the
wins. All numbers are char-level val loss (lower is better); e^loss is the
"effective number of characters the model is choosing among."

Evaluation uses a **fixed** set of batches drawn once from a dedicated
generator, so runs are directly comparable *within the same corpus*. Numbers
across different corpora are **not** strictly comparable (different held-out
sets); across corpora, read the *behavior* (the train–val gap, and whether val
rose after its minimum), not the absolute value.

---

## 1 · Batched attention
Replaced the `ModuleList` of `Head` modules with one fused qkv projection and a
head-as-tensor-axis attention. Identical parameters and outputs; **276.7 →
188.2 ms/step (1.47×)** on CPU. No effect on loss — it buys iterations per hour.

## 2 · Training longer
3k → 10k iterations took val **1.786 → 1.581** with no architectural change. The
model had simply been stopped early, not converged.

## 3 · A regression that did not work
`block_size 64→128` + `dropout 0.1→0.2` + cosine LR, all at once: val **1.5811 →
1.5931**. Three changes in one run, so attribution was impossible; dropout was
the leading suspect (the train–val gap was only 0.21, and the ~800K model was
capacity-bound). Dropout was reverted. **Lesson: bundle changes only when you
are willing to give up knowing which one acted.**

## 4 · RoPE vs learned positional embeddings
Controlled A/B — 5k iters, same seed, same data order, same fixed eval batches,
one variable changed:

| position encoding | train | val | gap |
|---|---|---|---|
| learned absolute | 1.4801 | 1.6666 | 0.187 |
| **RoPE** | 1.3594 | **1.5696** | 0.210 |

RoPE won by **0.097 val** — far beyond the 0.02–0.05 usually reported at scale.
Three reasons, all specific to a small model: capacity is the binding constraint
and RoPE spends none of it learning what position means; position is re-injected
at every layer rather than added once at the bottom; and at 5k iters a learned
table is still being learned while RoPE works from step 0. RoPE also overfits
faster (larger gap). RoPE is now the default (`cfg.use_rope`).

## 5 · Mixed precision (fp16)
`torch.autocast` + `GradScaler`, gated on CUDA. No effect on the final loss;
buys a large speedup on the GPU's tensor cores (idle in fp32). Made the scaled
runs below feasible on a free T4.

## 6 · Scaling up: the overfitting cliff
The nanoGPT tiny-shakespeare recipe — 6 layers, 384-dim, block 256, **10.7M
params** — on the **1 MB** corpus, 10k iters, T4:

```
iter     0   train 4.1888   val 4.1963   gap 0.008
iter  2250   train 1.0863   val 1.4754   gap 0.389   <- val minimum (saved)
iter  5000   train 0.7190   val 1.6059   gap 0.887
iter 10000   train 0.3639   val 1.9169   gap 1.553   <- discarded
```

Val bottomed a quarter of the way in, then **climbed for 7,750 iterations**
while train collapsed to 0.36 (perplexity ~1.4 — the corpus is memorized).
Best-val checkpointing kept the iter-2250 model; saving the final model would
have produced something worse than the 800K-param baseline. At 10.7M params
against a 1 MB corpus, the binding constraint is **data**: ~0.09 training chars
per parameter, versus GPT-3's ~1.7 tokens/param (itself later judged
data-starved by Chinchilla). The data-limited regime of the scaling laws,
reproduced on a free GPU.

---

## 7 · Data vs parameters (a 2×2, Chinchilla by hand)

The question: does more data *unlock* more parameters? Test it by running the
same param bump (10.7M → 19M) on both corpus sizes and seeing if the result
flips. All runs 8k iters, T4, fp16, RoPE, fixed eval batches.

| | **1 MB (tiny)** | **5.6 MB (complete works)** |
|---|---|---|
| **10.7M** | val 1.475, overfit @2250, gap → 1.55 | val **1.364**, gap 0.31, still improving |
| **19M** | *(Run C — pending)* | val **1.355**, gap 0.375, still improving |

### The data lever caused a regime change, not a small gain
The **same 10.7M model**, only the corpus size changed:

- on **1 MB**: val bottomed at 1.475 (iter 2250), then **rose to 1.92** —
  catastrophic overfitting, gap → 1.55.
- on **5.6 MB**: val **still falling at iter 8000** (1.364), **never overfit**,
  gap only 0.31.

More data didn't help a little — it moved the model *out of the overfitting
regime entirely*. Overfitting is the model running out of *new* text to learn
from; 5× the data postponed that past the end of the run.

### More params helped — but only barely, and that is the deeper lesson
On the 5.6 MB corpus (directly comparable): 19M (**1.355**) beat 10.7M
(**1.364**) — the *opposite* of what they would do on 1 MB, where 10.7M already
overfit. On enough data, capacity pays off.

But the gain is tiny (**0.009**), and that is itself the result. Chinchilla-
optimal for a 19M model is ~380M tokens; it was fed 5.6M — still **~68× under-
data**. So the bigger model can barely use its extra capacity. Two tells confirm
the bottleneck is still *data*, not capacity: both runs were **still improving at
8k iters** (undertrained), and the bigger model **barely pulled ahead** (if
capacity were the constraint, it would win clearly).

**Takeaway:** the model was data-bound at *both* sizes — acutely at 1 MB
(overfitting), chronically at 5.6 MB (the bigger model can't stretch its legs).
The same param bump flips from harmful (1 MB) to helpful (5.6 MB): that flip is
the data–parameter coupling, derived from the curves rather than read from a
paper.

*Run C (19M on 1 MB) will close the 2×2 — the control that should overfit even
harder than 10.7M did on 1 MB.*
