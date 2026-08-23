# Gates 3 and 4 -- candidate vs incumbent on the SHARED 137k blind test

Measured 2026-08-23 on one A100 (~$1.83). Both models scored by the SAME command,
the SAME 11 test files (15,456 rows), threshold PINNED to 0.5:

    uv run python tools/train/eval.py \
      --config tools/train/config/joint-boundary-mmbert-137k.yaml \
      --checkpoint <ckpt> --threshold 0.5

candidate = whr778/gliner2-ekf-frontend-mmbert
incumbent = whr778/gliner2-joint-boundary-mmbert-137k-clean

| category (strict micro F1) | candidate | incumbent | delta | support |
|---|--:|--:|--:|--:|
| entity | 0.6358 | 0.6200 | **+0.0158** | 60400 |
| relation | 0.0943 | 0.0350 | **+0.0593** | 5282 |
| classification | 0.6488 | 0.6328 | **+0.0160** | 7255 |
| structure | 0.0851 | 0.0755 | **+0.0096** | 4245 |
| event_type | 0.9542 | 0.9387 | **+0.0155** | 4157 |
| event_trigger | 0.7632 | 0.7487 | **+0.0145** | 6109 |
| event_argument | 0.1046 | 0.0913 | **+0.0133** | 20827 |
| event | 0.3752 | 0.3708 | **+0.0043** | 31093 |

| fair (Ortmann) | candidate | incumbent | delta |
|---|--:|--:|--:|
| entity | 0.6732 | 0.6454 | **+0.0278** |
| event_trigger | 0.7671 | 0.7523 | **+0.0147** |
| event_argument | 0.5508 | 0.4939 | **+0.0570** |

## Gate 4 structure -- SWEPT to the record head's own thresholds

The record head's max object probability is 0.178, so a default-0.5 number measures
a cutoff it cannot reach. Both swept over the same 856 records.

| | best threshold | precision | recall | F1 | TP | FP |
|---|--:|--:|--:|--:|--:|--:|
| candidate | 0.1 | 0.2535 | 0.0773 | **0.1184** | 328 | 966 |
| incumbent | 0.1 | 0.2385 | 0.0742 | **0.1132** | 315 | 1006 |

delta: **+0.0052**

## Verdict

**GATE 3 PASS** -- event_trigger +0.0145 and event_argument +0.0133 strict
(fair +0.0147 / +0.0570). No regression; an improvement.

**GATE 4 PASS** -- entity +0.0158, relation +0.0593, structure +0.0096 at the shared
threshold and +0.0052 swept (against the incumbent re-swept on this same box;
the historical out/record_sweeps figure was 0.1119, this run measured 0.1132 -- a
re-measurement difference, not a discrepancy). Nothing fell below the incumbent.

**The candidate is ahead of the incumbent on EVERY head measured here -- all 8 strict
categories and all 3 fair ones.** Set against gates 1 and 2, which it FAILS while the
incumbent passes gate 1: the mix rebuild improved everything measurable on held-out
corpora while making the behaviour it was actually built for worse.
