# Partial and stochastic gating in FFN activations

**Status: toy-scale technical note. Not validated at model scale.** Every number here
comes from a 4-block residual MLP on synthetic regression. Nothing in this note has been
run inside a transformer. It is written up because the strongest results here are
negative ones with an identified mechanism (§4, §5, §6), and those are what otherwise get
rediscovered the hard way. The one positive (§7) is a tie, reported as a tie.

Code: [`activation_variants.py`](activation_variants.py). Reproduce with
`uv run python tools/prototypes/activation_variants.py` and `--grad-scan`.

---

## 1. Motivation

mmBERT [6], following ModernBERT [5], uses GeGLU in its feed-forward blocks. GeGLU
computes `y = v ⊙ gelu(g)` from two projections of the same input. That multiplicative
form changes the gradient structure relative to a pointwise activation:

```
pointwise:  dy/dx = gelu'(x)              bounded, always
GeGLU:      dy/dv = gelu(g)               scales with the OTHER branch
            dy/dg = v ⊙ gelu'(g)          scales with the OTHER branch
```

A pointwise activation cannot amplify a gradient: ReLU's derivative is exactly `[0, 1]`
and GELU's is `[-0.1289, +1.1289]`. Neither branch of a GLU is bounded, because each is
scaled by the other. The question this note asks is whether the multiplicative
expressiveness can be kept while restoring a bound on the gradient path.

A correction to a common framing, since it motivated the original line of enquiry: GELU
did not fix exploding gradients relative to ReLU. Both are bounded. What GELU fixed was
dead units — ReLU has a half-line of exactly-zero gradient from which a unit cannot
recover.

## 2. Variants

All defined on the projected hidden vector `h`, split into `n` equal chunks along the
last dimension. Slot positions are preserved in every variant: `fc2` always reads the
same feature in the same place, so nothing here permutes features.

| name | construction | out width |
|---|---|---|
| `gelu` | `gelu(h)` | `1.0` |
| `geglu` | `v ⊙ gelu(g)`, `h` split in two | `0.5` |
| `stoch:k:n` | GELU applied to `k` of `n` chunks, rest identity | `1.0` |
| `hybrid4` | 8 chunks: 6 GELU, one slot holds `a ⊙ b` of the two linear chunks, one holds a passthrough | `1.0` |
| `hybrid5` | 8 chunks: 6 always GELU; the 2 linear slots each gated by a chosen GELU'd chunk | `1.0` |

Each of `stoch`, `hybrid4`, `hybrid5` has a **fixed** form (deterministic assignment) and
a **random** form (assignment redrawn per forward pass).

`hybrid4` and `hybrid5` preserve width exactly, so they need no up-projection and are
parameter-identical to a plain GELU FFN. GeGLU is parameter-matched by the two-thirds
`d_ff` rule from Shazeer [2].

### Eval rules for the random forms

`stoch` uses `p·gelu(x) + (1−p)·x` with `p = k/n`. **This is an approximation**: GELU does
not commute with the expectation, so no scalar correction exists.

`hybrid5` uses `E[c₆ ⊙ g_j] = c₆ ⊙ mean(g)`. **This is exact**, because `c₆` does not
depend on the draw.

A fixed mask must be evaluated with the same fixed mask. Blending it is a train/eval
mismatch, not a calibration — see §6.

## 3. Setup

4-block pre-norm residual MLP, `D=64`, `H=192`, AdamW at `3e-3`, 3000 steps, batch 256,
5 seeds. Target carries multiplicative interactions so that gating has something to earn:

```
y = sin(3x₀)·x₁ + x₂²·tanh(x₃) + x₄·x₅·x₆ − 0.5·cos(2x₇)
```

Parameter counts: 100,481 for every width-preserving variant, 100,737 for GeGLU.

## 4. Finding 1 — partial gating does not partially protect

Max `|∂y/∂x|` through the activation alone, as input scale grows:

| input scale | `gelu` | `geglu` | `stoch` | 1-gated-chunk-of-6 |
|---|--:|--:|--:|--:|
| 1x | 1.13 | 4.42 | 1.13 | 4.36 |
| 2x | 1.13 | 9.80 | 1.13 | 8.69 |
| 4x | 1.13 | 19.48 | 1.13 | 17.41 |
| 8x | 1.13 | 37.77 | 1.13 | 32.68 |

Gating **one chunk in six** leaves only 2.6% of channels above the pointwise bound, yet
the maximum is within 15% of fully-gated GeGLU. Explosion risk is set by the worst
channel, not the mean, so a small gated fraction buys a proportionally small reduction in
*exposure* and almost none in *tail*.

Note also that the blowup is scale-dependent — mild at unit scale, severe at 8x. This is
consistent with GLU-family models depending more heavily on normalisation discipline than
pointwise-FFN models do, since pre-norm holds the input near the regime where the
multiplicative term stays tame.

## 5. Finding 2 — the cost is function-class churn, not randomness

Test MSE, 5 seeds, lower is better:

| layout | fixed | random |
|---|--:|--:|
| `stoch:2:4` | 0.0567 | 0.7267 |
| `stoch:6:12` | 0.0702 | 0.8010 |
| `stoch:8:12` | 0.0492 | 0.6239 |
| `stoch:10:12` | — | 0.4326 |
| `hybrid4` | 0.0655 | 0.6555 |
| `hybrid5` | 0.0601 | **0.0623** |

Every `stoch` and `hybrid4` random arm costs roughly 10x. `hybrid5` randomised costs
nothing. The controlled comparison is the last two rows: both randomise, same harness,
same target, same parameter count.

| randomised variant | what the draw changes | MSE |
|---|---|--:|
| `hybrid4` | whether a slot is GELU or identity — **function class moves** | 0.6555 |
| `hybrid5` | which chunk partners a gated slot — **function class fixed** | 0.0623 |

**A draw that changes what kind of function a slot computes is fatal; a draw inside a
stable function form is free.**

The mechanism: `fc2` reads a fixed slot, and one weight cannot be simultaneously correct
for GELU output and identity output. Dropout [7] escapes this only because it is *linear
in the mask* — `E[mask ⊙ x] = p·x`, so a single scalar corrects the train/eval gap
exactly. Swapping a nonlinearity admits no scalar correction.

Two details support the diagnosis over a "just needs tuning" reading. The failing arms
have the **tightest** seed spread in the study (`hybrid4` random, ±0.0089): they converge
reliably to a bad solution, which is a method-level floor rather than unlucky
initialisation. And finer granularity does not help — per-sample masks (0.7783) and 8
chunks instead of 4 (0.7439) are no better. The random arms improve with the activated
fraction only because that dilutes the randomness; at `p=1` the variant *is* plain GELU.

There is an irony worth recording. GELU is *defined* by Hendrycks and Gimpel [3] as "the
expected transformation of a stochastic regularizer which randomly applies the identity
or zero map to a neuron's input." The `stoch` variants re-stochasticise, at chunk
granularity, a function that was constructed as the deterministic expectation of exactly
that kind of process. The experiment says the deterministic expectation was the right
object.

## 6. Finding 3 — the intuitive eval rule is the worst one

Same trained weights (`stoch:2:4` random), three eval rules:

| eval rule | MSE |
|---|--:|
| expectation blend `p·gelu(x) + (1−p)·x` | 0.8203 |
| sampled (same random rule as training) | 0.8622 |
| **plain GELU on all channels** | **7.2860** |

"Stochastic at train, clean at eval" — the natural transfer of dropout's discipline — is
**9x worse** than the expectation blend. Channels that trained as linear half the time
receive a function they rarely saw; the magnitude shift compounds across residual blocks.

A harness bug found and fixed mid-study, recorded so it is not repeated: applying the
expectation blend at eval to a *deterministic* mask is a train/eval mismatch, not a
calibration. It reported fixed variants at 7.54 and 81.1 MSE before correction; the real
values are 0.0492 and 0.0702.

## 7. Finding 4 — one gated slot in eight is the sweet spot

Whole-network gradient max against fit, all rows statistically tied on MSE:

| variant | gated fraction | grad max | MSE |
|---|---|--:|--:|
| `hybrid4:fixed` | 1 slot of 8 | **13.4** | 0.0655 |
| `gelu` | none | 17.2 | 0.0636 |
| `geglu` | all | 23.2 | 0.0483 |
| `hybrid5:fixed` | 2 slots of 8 | 26.1 | 0.0601 |
| `hybrid5:random` | 2 slots, random partner | 34.0 | 0.0623 |

`hybrid4:fixed` is parameter-identical to plain GELU, preserves width (so it avoids both
GeGLU's +50% parameters and the two-thirds `d_ff` workaround), ties it on fit, and has the
lowest gradient ceiling measured. `hybrid5` is the better *science* — it is what isolates
the churn mechanism — and the worse *design*: it moves the gradient ceiling above GeGLU,
which is the thing the exercise set out to avoid.

## 8. Prior art and attribution

**The mechanism in `hybrid4` is not new.** Its product chunk is exactly the **Bilinear**
variant of Shazeer [2] — a GLU with the nonlinearity omitted, leaving the component-wise
product of two projections. What is not covered in [2] is applying it to a *fraction* of
channels: every variant there gates all hidden units. [2] is also the source of the
two-thirds `d_ff` parameter-matching convention used throughout this note, and of the
finding that GEGLU and SwiGLU are the strongest variants.

- Gated linear units originate with Dauphin et al. [1].
- GELU, and its stochastic-regularizer derivation, with Hendrycks and Gimpel [3].
- Highway Networks [4] are the nearest conceptual relative to partial gating: a learned
  gate mixes a transformed path and a carried path. The difference is that Highway's mix
  is learned and continuous per unit, where `hybrid4`'s is a fixed structural split.
- Bilinear layers have been studied for weight-based mechanistic interpretability [8],
  a different motivation for the same primitive.

A search over several phrasings found no published study of gating only a fraction of FFN
channels with the remainder pointwise. **This is weak evidence and not a novelty claim.**
Web search is not a systematic review; the construction is simple enough to be sitting
unremarked in an ablation appendix; and a negative result like §5 is precisely the kind of
thing that never gets written up.

## 9. Limitations

Binding on every claim above:

- **Toy harness.** 4-block MLP, `D=64`, synthetic regression. No transformer, no
  attention, no real data, no scale.
- **The harness cannot rank the working variants.** Plain GELU alone ranged 0.043–0.074
  across runs of the same configuration. It separates 0.06 from 0.65 reliably and cannot
  separate 0.048 from 0.066 at all. Findings 2 and 3 rest on 10x gaps; finding 4's MSE
  column does not, and is reported as a tie for that reason.
- **Gradient maxima are single-batch measurements** of an unclipped norm, not a
  convergence claim.
- **No interaction with normalisation was tested.** Since finding 1 shows the blowup is
  scale-dependent, and pre-norm controls scale, a normalisation sweep could change the
  practical picture substantially.

## 10. What would make this real

In rough order of cost: swap `hybrid4:fixed` into a small transformer FFN and check it
holds at depth; sweep it against normalisation placement; then, if it survives, a
`gelu` vs `geglu` vs `hybrid4:fixed` arm on an actual encoder pretraining run. The
gradient-ceiling claim is the one worth testing first, since it is the only place
`hybrid4:fixed` measurably beats both baselines.

## References

1. Y. N. Dauphin, A. Fan, M. Auli, D. Grangier. *Language Modeling with Gated
   Convolutional Networks.* arXiv:1612.08083 (2016); ICML 2017.
   https://arxiv.org/abs/1612.08083
2. N. Shazeer. *GLU Variants Improve Transformer.* arXiv:2002.05202 (2020).
   https://arxiv.org/abs/2002.05202
3. D. Hendrycks, K. Gimpel. *Gaussian Error Linear Units (GELUs).* arXiv:1606.08415
   (2016). https://arxiv.org/abs/1606.08415
4. R. K. Srivastava, K. Greff, J. Schmidhuber. *Highway Networks.* arXiv:1505.00387
   (2015). https://arxiv.org/abs/1505.00387
5. B. Warner et al. *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for
   Fast, Memory Efficient, and Long Context Finetuning and Inference* (ModernBERT).
   arXiv:2412.13663 (2024). https://arxiv.org/abs/2412.13663
6. M. Marone, O. Weller, W. Fleshman, E. Yang, D. Lawrie, et al. *mmBERT: A Modern
   Multilingual Encoder with Annealed Language Learning.* arXiv:2509.06888 (2025).
   https://arxiv.org/abs/2509.06888
7. N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov. *Dropout: A
   Simple Way to Prevent Neural Networks from Overfitting.* JMLR 15:1929–1958 (2014).
8. M. T. Pearce, T. Dooms, A. Rigg, J. M. Oramas, L. Sharkey. *Bilinear MLPs enable
   weight-based mechanistic interpretability.* arXiv:2410.08417 (2024).
   https://arxiv.org/abs/2410.08417

Related in-repo: `tools/events_working_papers/TODO.md` item 2d.
