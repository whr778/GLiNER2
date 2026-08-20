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

### The fresh baseline, all four runs on ONE recorded invocation

The archived reference turned out to be unreproducible (`PROVENANCE.md`), so the production
`casualty-docee` model was run through the *same* settings as both arms. That makes a
common-footing table, which is what guard 1 actually needs:

| model | dead obs | ungated | gated @2.0 |
|---|--:|--:|--:|
| `casualty-docee` (production) | 88 | 378.809 | 378.555 |
| control 4ep | 205 | 46.844 | 3.336 |
| **muted** | **109** | **19.822** | 3.729 |

On one footing **both new models beat the production model by an order of magnitude, and
muting beats its own control 2.4x.** Read the docee row with care: it is destroyed by a
single extraction, a **94,000 in the Tennessee stream** where the archived run's largest
`dead` value anywhere is 3,000. That figure appears on current *and* on 2026-08-10 code, so
it is not a regression introduced since — it is one of the things the archived invocation
evidently avoided and this one does not.

Settings for all four, now written into every output file's `invocation` block:

    --associate record --rollup datasets/helene2024/rollup.json --window long
    --device cpu            (grid-step at its default 6.0)

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
  (`EKF_MHT_BUILD_RECORD.md` §25.5).

**Chased to a conclusion on 2026-08-20: the archived invocation cannot be recovered.** The
`--rollup` flag did not exist in any commit before the artifact was written, and
`rollup.json` was not in the tree either; both were uncommitted working-tree state that
changed before being committed. Full evidence in `PROVENANCE.md`. The response is the
fresh baseline above rather than further archaeology, and `run_pipeline.py` now writes its
complete argument vector and git commit — with a `-dirty` marker — into every output.

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
