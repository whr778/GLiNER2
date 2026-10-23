# Flipped-gate GEGLU (fg-GEGLU): spec, math, and a toy read

2026-10-23. Status: module built and tested (`gliner2/models/boundary/flipped_gate.py`), **not wired to any head**;
toy evidence only. This note exists so the idea is not re-run from the wrong inference rule and dropped (which is what
the first toy run would have concluded).

## 1. The idea (the user's design)

A stochastic gated activation, zero parameters, used in place of GELU inside an MLP. A biological picture: a nerve
cell in a cluster has on average one bidirectional dendrite, three unidirectional ones, and one that stops propagation.

Take a layer's pre-activation `x` of width `D` and split it along the last axis into `C` equal chunks `x_1..x_C`
(`C` even). On every training forward draw two disjoint random index sets without replacement:

- `A`, the **activated** chunks, `|A| = g` (the gates);
- `S`, the **skipped** chunks, `|S| = s` (the stop dendrite), `g + s <= C`.

The output keeps the width:

```
y_i = prod_{j in A} GELU(x_j)   if i in A        (every activated slot holds the same product)
y_i = x_i                        if i in S        (skipped: linear pass-through)
y_i = GELU(x_i)                  otherwise
```

In the user's original use inference was plain GELU over the whole width. The toy below shows that rule is what fails,
and that the Monte Carlo rule (mean of `K` random patterns at inference) is what makes the idea work. Both rules are in
the module (`eval_mode="gelu" | "mc"`, `mc_samples=K`).

Earlier work (user's, other models): longer training, but a smaller gap between dev and blind test, i.e. it behaved as
a regularizer, "almost like dropout for activations".

## 2. The math

### 2.1 Moments (standard-normal pre-activations, 4e6 Monte Carlo draws, `torch`)

| quantity | mean | std |
|---|---|---|
| `x ~ N(0, 1)` | 0.000 | 1.000 |
| `GELU(x)` | 0.282 (closed form `1 / (2 sqrt(pi))` = 0.2821) | 0.588 |
| `GELU(x) GELU(y)` (two independent gates) | 0.080 | 0.419 |
| `GELU(x) GELU(y) GELU(z)` (three gates) | 0.022 | 0.276 |

The closed form: `E[x Phi(x)] = E[Phi'(x)] = E[phi(x)] = 1 / (2 sqrt(pi))` by integration by parts. An activated slot
therefore carries a signal 3.5x weaker in mean (two gates) or 12.6x weaker (three gates) than a GELU slot, with a
smaller spread. Scale matters: at `x ~ N(0, 0.5^2)` the two-gate product mean is 0.008 vs GELU 0.089 (11x); at
`N(0, 2^2)` it is 0.51 vs 0.71 (1.4x). The gates are "quiet" channels for small pre-activations and ordinary ones for
large.

### 2.2 Expected training output per chunk

Under uniform random patterns, chunk `i` is activated with probability `g / C`, skipped with `s / C`, GELU'd otherwise.
Treating chunks as independent `N(0, 1)` (the init regime):

```
E[y_i] = (g/C) * E[GELU]^g + (s/C) * 0 + (1 - (g+s)/C) * E[GELU]
```

| (C, g, s) | P(activated) | P(skipped) | P(GELU) | E[train output] | ratio to E[GELU] = 0.282 |
|---|---|---|---|---|---|
| (4, 2, 1) | 0.50 | 0.25 | 0.25 | 0.110 | 0.39 |
| (6, 2, 2) | 0.33 | 0.33 | 0.33 | 0.121 | 0.43 |
| (8, 2, 2) | 0.25 | 0.25 | 0.50 | 0.161 | 0.57 |
| (12, 4, 4) | 0.33 | 0.33 | 0.33 | 0.096 | 0.34 |
| (6, 3, 2) | 0.50 | 0.33 | 0.17 | 0.058 | 0.21 |

### 2.3 Why plain GELU at inference fails

The next layer is trained on `y` whose per-chunk expectation is 0.2-0.6 of GELU's, with the product slots quiet and the
skipped slots carrying raw (possibly negative) values. Switching to `GELU(x)` at inference feeds that layer inputs
with a different mean, a different spread and no negative values: a covariate shift of order the entries above. The
toy (section 3) measures it directly as the gap between the training loss in train mode and the same weights in eval
mode: 0.05 vs 2.18 for (4, 2, 1) at 10,000 steps. More steps do not close it; it is a property of the rule, not of
convergence.

### 2.4 Why the Monte Carlo rule works, and what it costs

With `eval_mode="mc"`, inference returns `mean_{k=1..K} y(pattern_k)`, an unbiased estimate of the training-time
expectation `E_pattern[y]`, with variance shrinking as `1/K`. The downstream layer then sees inputs from the
distribution it was trained on. This is the same argument that makes Monte Carlo dropout consistent; the difference
is that dropout's noise has expectation equal to the identity (through its `1 / (1 - p)` rescale), so plain inference
is already the expectation, whereas fg-GEGLU's noise has expectation `E[y] != GELU(x)` and no rescale, so only the
sampled estimate matches.

Cost: training adds `g - 1` elementwise products per forward. Inference costs `K` evaluations of the activation and of
everything downstream of it. For a scorer MLP (one hidden layer after the activation) that is `K` small matmuls;
for anything inside the encoder it is prohibitive, and the encoder is mmBERT, untouched. The toy used `K = 8`.

### 2.5 Relation to dropout and GEGLU

GEGLU: `GELU(xW) * xV`, a fixed gate/value split with two projections. fg-GEGLU: no extra projection, the gate and
value roles are drawn at random per forward, and the "value" is itself a GELU. Dropout 0.1 on GELU outputs: unbiased
multiplicative Bernoulli noise. fg-GEGLU: biased multiplicative noise (products) plus a random linear pass-through.
On the toy, dropout 0.1 beat every fg configuration under plain-GELU inference and was within 0.05-0.2 MSE of the best
fg configuration under MC inference (section 3).

## 3. Toy protocol and results

A 2-hidden-layer MLP (20 -> H -> H -> 1) fits a noisy nonlinear target from 400 training points (label noise sd 0.3);
dev 1,000 and blind 5,000 points without noise. Two budgets: 1,500 full-batch AdamW steps (lr 3e-3, wd 0.01; GELU
overfits here) and 10,000 mini-batch steps (batch 64, cosine LR). Metric MSE; seeds 8 (short) / 4 (long). Scripts:
scratchpad `toy_fg_geglu*.py` (outputs under /Volumes/Development/tmp/relation_fix/toy_fg_geglu*.txt).

### 3.1 Width 64, plain-GELU inference (first and second run)

| budget | GELU | dropout 0.1 | fg (4, 2, 1) | fg (4, 3, 1) | fg (4, 2, 0) |
|---|---|---|---|---|---|
| 1,500 full-batch | 1.82 | 1.17 | 1.76 | 9.46 (one seed 41.9) | 3.73 |
| 10,000 mini-batch | 1.33 | 1.16 | 2.77 | 8.47 | 3.04 |
| train loss, train mode vs eval mode, 10k | 0.00 / 0.00 | 0.07 / 0.00 | 0.07 / 2.20 | 0.09 / 8.20 | 0.02 / 1.59 |

A literal first build had every non-activated chunk linear (one fixed passive chunk, no GELU elsewhere): 3.84 / 4.26
blind. With GELU on the un-drawn chunks (the user's actual design) the short-budget number improves to 1.76, but the
long budget and the eval-mode gap say the same thing: the rule does not match the training distribution.

### 3.2 Width 72, both inference rules

| eval rule | fg (4, 2, 1) short / long | fg (6, 2, 2) short / long | GELU short / long | dropout short / long |
|---|---|---|---|---|
| plain GELU | 1.64 / 2.68 | 1.66 / 3.04 | 1.81 / 1.25 | 1.15 / 1.15 |
| one random pattern | 2.34 / 2.50 | 1.36 / 1.61 | | |
| mean of 8 patterns | 2.04 / 2.44 | **1.20 / 1.35** | | |
| dev-blind gap, mean of 8, short | 0.133 | **0.051** (GELU 0.085, dropout 0.051) | | |

(6, 2, 2) under the MC rule beats GELU at both budgets and sits within 0.05-0.2 of dropout; (4, 2, 1) does not
benefit from the MC rule. The configuration decides, which motivates the sweep.

### 3.3 Parameter sweep (width 72, MC rule with K = 8 and plain GELU side by side)

`toy_fg_geglu_sweep.py`, 17 configurations plus the two baselines; blind MSE (dev-blind gap in parentheses); short =
1,500 full-batch steps, 6 seeds; long = 10,000 mini-batch steps, 4 seeds. Baselines: GELU 1.87 / 1.25, dropout 0.1
1.17 (0.052) / 1.15 (0.047).

| (C, g, s) | plain GELU short / long | **MC (K=8) short / long** |
|---|---|---|
| (4, 2, 1) | 1.70 / 2.68 | 2.14 / 2.44 |
| (4, 2, 2) | 2.29 / 2.96 | 1.96 / 2.04 |
| (6, 2, 1) | 2.43 / 2.31 | 1.20 (0.043) / 1.34 (0.033) |
| (6, 2, 2) | 1.72 / 3.04 | 1.22 (0.070) / 1.35 (0.064) |
| (6, 2, 3) | 1.65 / 2.86 | 1.30 / 1.58 |
| (6, 3, 2) | 3.45 / 3.01 | 2.75 / 2.51 |
| (8, 2, 2) | 1.80 / 2.23 | 1.06 (0.060) / 1.12 (0.072) |
| (8, 2, 4) | 1.71 / 2.43 | 1.08 (0.069) / 1.16 (0.082) |
| (8, 3, 3) | 4.77 / 3.70 | 1.42 / 1.35 |
| (8, 4, 2) | 54.3 / 80.4 | 1.40 / **1.01** (0.060) |
| (8, 4, 4) | 3.75 / 5.25 | 3.01 / one seed diverged (4023) |
| (12, 2, 2) | 1.69 / 1.85 | 1.11 (0.062) / 1.17 (0.014) |
| (12, 2, 4) | 1.62 / 2.08 | 1.04 (0.056) / **1.03 (0.029)** |
| **(12, 3, 3)** | 4.39 / 2.84 | **1.02 (0.040)** / 1.05 (0.039) |
| (12, 4, 2) | 10.1 / 7.8 | 1.05 (0.060) / 1.07 (0.050) |
| (12, 4, 4) | 13.7 / 8.3 | 1.23 / 1.08 (0.089) |
| (12, 6, 4) | 161 / 200 | 1.54 / 1.24 (0.030) |

Reading, in order of confidence:

1. **The inference rule is decisive.** Under plain GELU no configuration beats dropout and every configuration with
   `g >= 4` is catastrophic (54 to 200 MSE): the product of four or more GELUs has mean `0.282^4 = 0.006`, so the
   downstream layer learns gains of order 100, and plain GELU at inference feeds those gains full-size activations.
   Under the MC rule the same configurations are among the best.
2. **More chunks is better.** With `C = 12`, every configuration with `g <= 4` beats dropout at both budgets under
   the MC rule; `C = 8` with `g = 2` does too; `C = 4` never does. Finer chunks make the per-forward perturbation
   smaller and smoother, closer to dropout's regime.
3. **Best stable configurations: (12, 3, 3) and (12, 2, 4)**: top-3 at both budgets, blind 1.02-1.05, dev-blind gap
   0.03-0.04, below dropout's 1.15-1.17 and GELU's 1.25-1.87. (8, 4, 2) is the long-budget winner but 1.40 at the short
   one. The module's defaults are now (12, 3, 3), MC, K = 8.
4. **Skipped chunks matter less than gates**: at `C = 12`, `s` from 2 to 4 moves blind MSE by <= 0.1; `g` from 2 to 6
   moves it from 1.03 to 1.24.
5. One divergence: (8, 4, 4) under MC on one seed (blind 4023), so stability needs watching for `g + s = C`.

## 4. Where it could apply in GLiNER2, and the decision rule

The property measured is a smaller dev-to-blind gap, so the candidate is a head whose VAL-minus-blind gap exceeds its
run-to-run floor. We have no per-head gap table yet; eb22d's blind-test scoring will produce one. The relation head's
problem today is the opposite (a fit failure, code review Issues 8-9), where a regularizer costs before it helps.
Candidate insertion points, all post-activation MLPs where MC inference is cheap: the relation scorer's hidden GELU,
the coref distance MLP, the junction geometry channel's activation arm. Not the encoder.

Decision rule: if a head shows a gap beyond its floor after eb22d, run one one-epoch arm with fg (12, 3, 3), MC inference
K = 8, on that head's MLP against the control, ~$17-23. Width constraint: the layer width must be divisible by C
(768 is; the 64-wide coref distance MLP would take C = 8, e.g. (8, 2, 2)). If no head shows a gap, this note records
a sound idea the programme does not need.

## 5. Open questions

- Why more chunks help (answered by the sweep in direction, not mechanism): C = 12 beats C = 6 beats C = 4 at matched
  fractions; the per-forward perturbation is finer. Whether that holds at width 768 is untested.
- The MC rule's `K`: 8 in the toy; the inference cost scales with it and the variance as `1 / K`.
- Whether the random linear pass-through (`s > 0`) is doing the regularizing or the product is: (4, 2, 0) was worse
  than (4, 2, 1) under plain GELU, and at C = 12 the skipped count moved blind MSE by <= 0.1 while the gate count moved
  it by 0.2: the product is the active ingredient, the pass-through a minor one.
- Whether a learned rescale (one scalar per chunk role) would let plain-GELU inference work, removing the `K` cost.
