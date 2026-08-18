# Testing hybrid4:fixed in a real pretrained encoder

Plan only. **Nothing here has been run.** Code is written and validated for shapes,
parameter matching and device support; no training has been started.

Follows from [`PARTIAL_GATING.md`](PARTIAL_GATING.md), whose findings are toy-scale
(4-block MLP, synthetic regression) and cannot say whether partial gating matters in a
transformer.

## The hypothesis, stated so it can fail

**Not** "hybrid4:fixed produces a better model." The toy study found it *tied* with plain
GELU on fit, and there is no reason to expect that to change. Claiming a quality win here
would be fitting the hypothesis to the hoped-for result.

**The claim is about stability.** `hybrid4:fixed` had the lowest gradient ceiling measured
(13.4, against plain GELU 17.2 and GeGLU 23.2), and the multiplicative blowup in GeGLU is
scale-dependent — mild at unit activation scale, 37.8x at 8x. So:

> At a learning rate high enough to destabilise GeGLU, hybrid4:fixed continues to train.

If that is false, the whole line is dead and no further money should be spent. This is
also a failure mode already observed in this repo, not a hypothetical: mmBERT under
sdpa+bf16 went non-finite at step 15 across five loss terms at once.

## Why crammed-BERT is the right baseline

[Geiping & Goldstein, arXiv:2212.14034](https://arxiv.org/abs/2212.14034) (ICML 2023):

| property | value | why it matters |
|---|---|---|
| FFN | `(GELU(xW₁) ⊗ xV)W₂` — **GeGLU** | the baseline IS the thing hybrid4 replaces |
| budget | 24h wall-clock, one GPU | fixed-compute framing: "who gets further on equal budget" |
| reference hardware | 2080 Ti (5 exaFLOP), A4000 (8), A6000 (13) | |
| data | English Wikipedia + BookCorpus, seq 128, **single epoch** | English-only, no data revisited |
| result | **GLUE 78.6** (A6000) vs BERT-base **80.9** | published number to reproduce |
| finetuning | ≤5 epochs, batch 16, lr 4e-5 cosine, global across tasks | explicitly excluded from the budget |

It is an encoder (so it transfers to what this repo builds), and the paper is itself a
component ablation, so the codebase expects pieces to be swapped.

Cheaper alternative: MosaicBERT, arXiv:2312.17482, also GLU, also English, ~$20 to
pretrain. **Avoid modded-nanogpt** despite its excellent variance characterisation — it is
a decoder and its components are co-tuned tightly enough that an activation swap confounds
the result.

## Staged plan, with a stop rule at every gate

### Stage 0 — LR ladder ([`lr_ladder.py`](lr_ladder.py))

Three FFN variants, identical everything else, learning rate escalated until something
breaks. No published baseline needed, no GLUE, no pretraining budget: the comparison is
internal.

```bash
# free local screen, ~22 min on MPS (measured 5.39 steps/s)
uv run python tools/prototypes/lr_ladder.py --device auto

# the scaled version, BERT-base dims, ~$7 on one A10 (14h on MPS -- pay instead)
uv run python tools/prototypes/lr_ladder.py --steps 4000 --seeds 3 \
    --d-model 768 --n-layers 12 --n-heads 12 --d-ff 3072
```

The default 6-layer / `d_model=256` model is a *screen*, not a verdict. Instability
usually needs depth and duration, so a null at that size should be re-run at
`d_model=768, n_layers=12` before it is believed.

**Stop rule:**

| outcome | reading | action |
|---|---|---|
| GeGLU diverges where hybrid4:fixed survives | ceiling is real at depth | proceed to stage 1 |
| nothing diverges (at scaled size) | ceiling does not bite | **stop** |
| hybrid4:fixed diverges first | hypothesis false | **stop** |

A null is a stop, not an inconclusive. Stage 2 measures quality, and quality is already
predicted to be a tie — without a stability finding there is nothing left to motivate it.

### Stage 1 — reproduce the published baseline

One unmodified crammed-BERT run, must land near GLUE 78.6. **Non-negotiable before stage
2.** At an expected effect near zero, a regression and a broken setup are
indistinguishable without one clean reproduction.

Lambda has no A6000; an A10 is roughly 0.8x its throughput, so a 24h A10 run should land
slightly *below* 78.6. That is expected, not a failed reproduction — the ablation only
requires all arms share a budget.

### Stage 2 — three arms, three seeds

`geglu` (the recipe's own) / `gelu` / `hybrid4:fixed`. Plain GELU is what shows whether any
difference is about gating at all, rather than about the specific gate.

Three seeds minimum: the expected quality delta is ~0, so a single seed is noise either
way. This is the same discipline that made the toy study readable — there, plain GELU alone
ranged 0.043-0.074 across runs of one configuration.

## Cost

Lambda `gpu_1x_a10` at $1.29/hr, `gpu_1x_a100_sxm4` at $1.99/hr.

**Stage 0 throughput was measured, not estimated** (Apple MPS, batch 32, seq 128,
`geglu` arm): 5.39 steps/s for the screen model, 0.95 steps/s at BERT-base dims
(108,549,120 params). The A10 column below is extrapolated from that at ~8x and is
worth +/-2x; the pretraining rows come from the paper's own 24h budget definition.

| stage | work | steps | A10 |
|---|---|--:|--:|
| 0a — local screen | 12 configs, d=256 L=6, 600 steps | 7,200 | **$0** (22 min on MPS) |
| 0b — scaled ladder | 12 configs, d=768 L=12, 4000 steps, 3 seeds | 144,000 | ~$7 |
| 1 — baseline reproduction | 24h pretrain + ~5h GLUE | — | ~$37 |
| 2 — full comparison | 9 x (24h + GLUE) | — | ~$335 |
| | **worst case, all stages** | | **~$379** |

An earlier version of this document put stage 0b at ~$31 by assuming ~2h per
configuration. Measured, it is ~13 min per configuration on an A10. **Run 0a locally for
free, then pay for 0b rather than spending 14 wall-clock hours on MPS** — at ~$7 the
three-seed version is cheap enough that a single-seed divergence result is not worth the
ambiguity.

Per arm-seed: 24h pretrain ($30.96) + ~5h GLUE across 8 tasks, MNLI and QQP dominating
($6.45) = ~$37. On A100 the same is ~$58, but the budget is defined as wall-clock, so a
faster card buys a *better* model rather than a cheaper run — it changes what "one day"
means and breaks comparability with the published number.

Arms parallelise for free: 9 runs across 3 instances is 3 days elapsed at the same dollar
cost.

## Design constraints that are easy to get wrong

- **Parameter matching is handled in [`ffn_variants.py`](ffn_variants.py)**, not left to
  the config. `matched_d_ff()` applies Shazeer's two-thirds rule to GeGLU and leaves
  hybrid4 alone (it preserves width). Verified: all three land at exactly 4,718,592 FFN
  parameters at `d_model=768, d_ff=3072`. Print the counts in any run — a silent mismatch
  invalidates everything.
- **Do not randomise the slot assignment.** `Hybrid4FFN` is fixed by construction. The toy
  study's clearest result is that randomising which slots receive the nonlinearity costs
  ~10x, and that it cannot be repaired at eval time.
- **Report the gradient norms, not just the loss.** The ceiling is the hypothesis; final
  loss is the control variable.
- **`d_ff` must be divisible by 8** for the 8-chunk layout. `Hybrid4FFN` raises rather than
  silently reshaping.
