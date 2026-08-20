# Cross-event muting arm — results, 2026-08-20

TODO item 2. Two arms trained on one Lambda A10 (~$2.35, terminated), 4 epochs each,
differing only in the corpus:

| | corpus | model (private) |
|---|---|---|
| control | `casualty_loc_split` | `whr778/gliner2-base-v1-casualty-loc-split-4ep` |
| treated | `casualty_loc_muted` (`--mute-interference-prob 0.35`) | `whr778/gliner2-base-v1-casualty-loc-muted` |

Matched epochs deliberately: the published clean control ran 8, and this arm runs 4.

## 1. Blind test — the treatment has the shape it was designed to have

| | F1 | precision | recall |
|---|--:|--:|--:|
| control | **0.8151** | 0.8119 | 0.8182 |
| muted | 0.8005 | **0.8273** | 0.7754 |

Precision **+0.015**, recall **−0.043**. That is learned suppression: the muted arm withholds
more. Pre-registered as "a drop in F1 here is the treatment working, not a regression",
because the test split is built at prob 0.0 so every interference snippet *does* carry gold.

## 2. Binding probe — a wash, and NOT the abstention failure that was guarded against

| | C catches | C FP | binds (cross · ok) | loc fill | event fill |
|---|--:|--:|--:|--:|--:|
| control | 1/11 | 1.2% | 5/11 · 10/83 | 96.1% | 6.5% |
| muted | 1/11 | 2.4% | 5/11 · **15**/83 | 94.1% | **32.4%** |

The muted arm passes guard 4 (FP ≤ 3.6%, binds ≥ 12/83, cross ≥ 5/11) and specifically did
**not** abstain — it binds 50% more ordinary observations than its control and fills `event`
five times as often. But cross-event catches are 1/11 for both, the FP gap is one observation
of 83 (inside the drift measured on `fastino`), and the fill gap is inside the 4-point noise
floor. **On binding this is a wash.**

## 3. Helene pipeline — the primary guard, and it needs a caveat before it can be read

Ungated per-place mean nRMSE, all three runs on identical pipeline settings:

| model | dead obs | ungated | gated @2.0 |
|---|--:|--:|--:|
| control 4ep | 205 | **46.844** | 3.336 |
| muted | 109 | **19.822** | 3.729 |

**Muting more than halves the ungated error, 46.844 → 19.822 (2.4x).** The win is almost
entirely one state: **North Carolina 127.504 → 0.640**, which is where Katrina's 1,400 lives
— the single largest cross-event contaminant and exactly what this arm was built to remove.
The muted arm also extracts 47% fewer `dead` observations (205 → 109), which is the
suppression doing what it was trained to do rather than a recall collapse.

Gated, the muted arm is slightly *worse* (3.729 vs 3.336), so the gain does not survive a
mechanism that was already removing large contaminants by magnitude. That is consistent:
both are attacking the same figures.

### THE CAVEAT, and it blocks reading guard 1 as pre-registered

**These pipeline settings do not reproduce the archived baseline.** Running the *production*
`casualty-docee` model through the same invocation gives 88 `dead` observations and an ungated
mean of **378.809**, where the archived `tracked_rollup.json` has 106 observations and 5.247.
The archived run's full invocation was never recorded — only `associate: record` is stored in
the file — so something material differs (chunk size, thresholds, gate/event model, or a
model revision).

Consequences, stated rather than smoothed over:

- The three runs here are mutually comparable — same settings, same feed, same rollup — so
  **the A/B between control and muted is valid**.
- No absolute number here is comparable to any published Helene figure, including the 5.247
  and 0.591 that guard 1 was pre-registered against. **Guard 1 cannot be scored as written.**
- This is the same provenance gap that stopped Türkiye–Syria being re-extracted
  (`EKF_MHT_BUILD_RECORD.md` §25.5). Recovering the archived invocation is a prerequisite for
  reading this arm against the programme's baseline, and it is a separate task.

## Reproduce

```bash
uv run python tools/ekf_showcase/run_pipeline.py \
  --feed datasets/helene2024/_cache/feed.jsonl \
  --out <out>.json --casualty-model <model> \
  --associate record --rollup datasets/helene2024/rollup.json \
  --window long --device cpu

uv run python tools/ekf_showcase/scope_gate_test.py --dataset helene --tracked <out>.json
uv run python tools/ekf_showcase/event_binding_probe.py --model <model> --device cpu
```

Control rows for the earlier models are in `../control_rows/`.
