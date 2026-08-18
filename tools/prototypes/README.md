# tools/prototypes

Standalone architecture experiments. Nothing here is imported by the training pipeline,
the extractor, or the viewer — these files depend only on `torch` and are safe to change
without touching model behaviour anywhere else.

The bar for this directory is lower than for `tools/train`, but not zero: an experiment
lives here only with its numbers, its harness limits, and its prior art written down.
An unlabelled result is worse than none, because someone will cite it later.

## Contents

| file | what it is |
|---|---|
| [`activation_variants.py`](activation_variants.py) | Runnable comparison of GELU, GeGLU, and chunked partial/stochastic gating variants on a parameter-matched toy MLP. `--grad-scan` for gradient bounds. |
| [`PARTIAL_GATING.md`](PARTIAL_GATING.md) | Write-up of the above, **plus §11: the central premise was tested in a transformer and REFUTED**. GeGLU turns out to be the most stable variant, not the least, and hybrid4's gradient-ceiling advantage does not transfer. |
| [`divergence.py`](divergence.py) | Classifies an MLM loss trace as trained / diverged / flat. Split out and unit-tested (`python divergence.py`) after two hand-rolled in-loop criteria were wrong in opposite directions. |
| [`ffn_variants.py`](ffn_variants.py) | Transformer-shaped `nn.Module` FFNs -- `GeluFFN`, `GeGLUFFN`, `Hybrid4FFN` -- parameter-matched by `matched_d_ff()`. The drop-in artifact for testing the toy result at depth. |
| [`lr_ladder.py`](lr_ladder.py) | Stage 0 of the plan below: escalate the learning rate across the three variants until one destabilises. Runs on CPU, CUDA or MPS. |
| [`CRAMMING_EXPERIMENT.md`](CRAMMING_EXPERIMENT.md) | Staged plan and costing for testing `hybrid4:fixed` in a real encoder. **Stage 0 ran, the stop rule fired, stages 1-2 cancelled** — $0 spent against a ~$379 plan. |

Related in-repo: `tools/events_working_papers/TODO.md` item 2d.

---

## References

Collected 2026-08-18 while writing `PARTIAL_GATING.md`. **Entries marked ✓ were opened
and their title/authors verified; entries marked ○ surfaced in search and are listed as
leads only — do not cite them without reading them first.**

### Gating and activation functions

- ✓ **GLU, original.** Y. N. Dauphin, A. Fan, M. Auli, D. Grangier. *Language Modeling
  with Gated Convolutional Networks.* arXiv:[1612.08083](https://arxiv.org/abs/1612.08083)
  (2016); ICML 2017. Introduces the gated linear unit.
- ✓ **The GLU variant family.** N. Shazeer. *GLU Variants Improve Transformer.*
  arXiv:[2002.05202](https://arxiv.org/abs/2002.05202) (2020). Defines GEGLU, SwiGLU, and
  **Bilinear** (the product of two projections with no nonlinearity). Reports GEGLU and
  SwiGLU as strongest. **Source of the two-thirds `d_ff` parameter-matching convention.**
  This is the direct prior art for `hybrid4` — its product chunk is Bilinear applied to a
  fraction of channels.
- ✓ **GELU.** D. Hendrycks, K. Gimpel. *Gaussian Error Linear Units (GELUs).*
  arXiv:[1606.08415](https://arxiv.org/abs/1606.08415) (2016). Defines GELU as `xΦ(x)` and,
  importantly for §5 of the write-up, derives it as "the expected transformation of a
  stochastic regularizer which randomly applies the identity or zero map to a neuron's
  input."
- ✓ **Highway Networks.** R. K. Srivastava, K. Greff, J. Schmidhuber.
  arXiv:[1505.00387](https://arxiv.org/abs/1505.00387) (2015). Nearest conceptual relative
  to partial gating: a learned gate mixes a transformed path against a carried path.
- ✓ **Bilinear layers for interpretability.** M. T. Pearce, T. Dooms, A. Rigg,
  J. M. Oramas, L. Sharkey. *Bilinear MLPs enable weight-based mechanistic
  interpretability.* arXiv:[2410.08417](https://arxiv.org/abs/2410.08417) (2024). Same
  primitive, different motivation.
- **Dropout.** N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov.
  *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* JMLR 15:1929–1958
  (2014). The contrast case: dropout is linear in its mask, which is exactly why its
  train/eval scalar works and a stochastic *nonlinearity* swap has no equivalent fix.

### Encoder architectures this work is aimed at

- ✓ **ModernBERT.** B. Warner et al. *Smarter, Better, Faster, Longer: A Modern
  Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and
  Inference.* arXiv:[2412.13663](https://arxiv.org/abs/2412.13663) (2024). GeGLU + RoPE +
  FlashAttention, 8192 context.
- ✓ **mmBERT.** M. Marone, O. Weller, W. Fleshman, E. Yang, D. Lawrie, et al. *mmBERT: A
  Modern Multilingual Encoder with Annealed Language Learning.*
  arXiv:[2509.06888](https://arxiv.org/abs/2509.06888) (2025). ModernBERT-derived, 3T
  tokens, 1833 languages. The encoder behind this repo's boundary models, and the reason
  the GeGLU gradient question came up at all.

### Candidate baselines for a real pretraining test

Both are English-only, cheap, and **already use GeGLU**, so `hybrid4:fixed` is a drop-in
swap against a baseline that is exactly the thing it replaces.

- ✓ **Cramming.** J. Geiping, T. Goldstein. *Cramming: Training a Language Model on a
  Single GPU in One Day.* arXiv:[2212.14034](https://arxiv.org/abs/2212.14034) (2022);
  [PMLR v202](https://proceedings.mlr.press/v202/geiping23a/geiping23a.pdf), ICML 2023.
  FFN is `(GELU(xW₁) ⊗ xV)W₂`. Fixed-compute protocol and a component ablation by
  construction — the recommended starting point.
  Weights: [JonasGeiping/crammed-bert](https://huggingface.co/JonasGeiping/crammed-bert).
- ○ **MosaicBERT.** *MosaicBERT: A Bidirectional Encoder Optimized for Fast Pretraining.*
  arXiv:[2312.17482](https://arxiv.org/pdf/2312.17482) (2023);
  [Databricks writeup](https://www.databricks.com/blog/mosaicbert) quotes ~$20 to
  pretrain. Newer and cheaper than Cramming; also GLU.
- ○ **modded-nanogpt.** K. Jordan et al.
  [github.com/KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).
  Target is 3.28 FineWeb val loss; the best-characterised small-scale pretraining
  benchmark that exists. **Listed as a caution, not a recommendation** — it is a decoder,
  and its components are co-tuned tightly enough that swapping one confounds the result.

### Surfaced but unread

Leads only. Titles and IDs are from search listings and have not been checked.

- ○ *Gating Enables Curvature: A Geometric Expressivity Gap in Attention*,
  arXiv:2604.14702 — reportedly ablates multiplicative gating against a plain
  nonlinearity, which is adjacent to the "gating adds capacity, it does not regularise"
  claim.
- ○ *Dependency-Aware Semi-Structured Sparsity of GLU Variants in Large Language Models*,
  arXiv:2405.01943 — structured sparsity within GLU FFNs.
