# The plausibility ceiling supersedes the muting arm

Added and swept 2026-08-20, following the false-positive audit. A declared per-event ceiling
drops any observation above the largest credible toll for that event, before gating.
Implemented in `scope_gate_test.py` as `plausibility_filter` / `--max-plausible`.

Hurricane Helene killed on the order of 230 people, so a five- or six-figure figure in any of
its streams is not a casualty count. That is prior knowledge, declared per event — the same
kind of statement as the `hierarchy` block already in `rollup.json`.

## Swept, not picked

Ungated per-place mean nRMSE. All runs on the same pipeline settings.

| ceiling | docee | control 4ep | muted |
|---:|--:|--:|--:|
| off | 378.809 | 46.844 | 19.822 |
| 20,000 | **18.287** | 26.961 | 19.822 |
| 5,000 | 18.287 | 22.289 | 19.822 |
| **2,000** | 18.190 | **5.853** | **6.194** |
| 1,000 | 18.190 | 5.057 | 6.194 |
| 500 | 18.190 | 5.057 | 6.194 |
| 250 | 18.190 | 5.019 | 6.194 |

**One observation is worth 20x on the production model**: dropping the single 94,000 takes
docee from 378.809 to 18.287.

## The finding: the arm's advantage does not survive it

At a ceiling of 2,000 — roughly nine times Helene's true toll, so generous rather than tuned:

| | dead obs | ungated | gated @2.0 |
|---|--:|--:|--:|
| control 4ep | 187 | **5.853** | **3.336** |
| muted | 106 | 6.194 | 3.729 |

**The control is better on both.** Muting won the unprotected comparison 46.844 → 19.822, and
that entire gain is recovered — and exceeded — by a threshold that needs no model, no
training and no GPU.

Worse for the arm: **the control scores better while carrying 81 more observations.** The
ceiling removes only junk; muting removed genuine signal alongside it. Per state at the
ceiling, muting still wins North Carolina (0.640 vs 4.727) but loses Georgia (6.805 vs 2.819)
and South Carolina (3.689 vs 2.021).

And the gated column never moves at any ceiling, in any arm — the scope gate was already
removing these values by magnitude. The ceiling matters only where nothing else is defending.

## So guard 1 fails once the cheap baseline exists

Guard 1 was pre-registered as ungated per-place mean, want lower than control. Muting passes
against an *undefended* control and fails against a control with a one-line ceiling. Since
nobody would ship the undefended configuration, **the honest reading is that the muting arm
is superseded**, not that it works.

This does not retract what the arm demonstrably did: it suppressed 15 of 20 large false
positives, and the suppression is real and learned. It says that the cheapest possible
alternative does the same job better, which is the comparison that decides whether to keep it.

## What the ceiling still cannot do, unchanged

Set to 2,000 it leaves Katrina's 1,400, 1,500 troops and 8,000 crews untouched, because all
three are plausible magnitudes. Set low enough to catch Katrina it stops being a plausibility
test and becomes the magnitude gate again — rejecting a cross-event toll for being large
rather than for belonging to another storm. **Cross-event is still unsolved, by this and by
the muting arm alike.**
