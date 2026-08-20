# Cross-event detection, re-measured on the corrected instrument (2026-08-20)

All five models re-run after the label re-audit. The old labels were a `(span, value)`
string match that was **27% correct on its own positive class**; these are per-occurrence
labels assigned by reading the context (`../../helene_audit_labels.json`).

The false-positive denominator is now **genuine Helene casualties only** (82 of 106).
Flagging "1,400 landslides" is not a false positive for cross-event detection.

| model | C catches | C false pos | binds an event (cross · genuine) | unbound (cross · genuine) | `event` fill |
|---|--:|--:|--:|--:|--:|
| **`fastino/gliner2-base-v1`** (not casualty-tuned) | **4/6** | 31.7% | **5**/6 · **61**/82 | 1 · 14 | **80.3%** |
| `…-casualty-docee` | 1/6 | 1.2% | 1/6 · 7/82 | 3 · 40 | 72.0% (mostly junk — 35 rejected) |
| `…-casualty-loc-split-clean` | 2/6 | 2.4% | 2/6 · 13/82 | 4 · 66 | 23.7% |
| `…-casualty-loc-muted` | 2/6 | 2.4% | 2/6 · 18/82 | 4 · 57 | 32.4% |
| `…-casualty-loc-split-4ep` | 1/6 | 1.2% | 1/6 · 13/82 | 5 · 62 | 7.1% |

## The finding: casualty fine-tuning trains the attribution out

**The best cross-event detector in the table is the model that was never fine-tuned on
casualties.** `fastino` catches 4 of 6 and binds an event for 61 of 82 genuine observations.
Every casualty-tuned descendant catches 1 or 2 and binds almost nothing.

The near-zero false-positive rates on the tuned models are **not precision**. They are
abstention, and the coverage columns show it: `loc-split-4ep` leaves 62 of 82 genuine
observations unbound and fills an `event` field 7.1% of the time. A model that never names
an event cannot name the wrong one.

`casualty-docee` is the instructive middle case: it fills `event` 72% of the time, but 35 of
those bindings are **rejected** because the string names no event the schema found — the
number-copying artifact. High fill, no information.

**Why this is the shape it is.** Every casualty corpus carries 8 DocEE event *types* and no
named identities. Fine-tuning on them optimizes a schema in which the event a figure belongs
to is never a supervised field, so the ability to name it — present in the base model —
decays. We have been trying to add attribution to models we systematically trained it out of.

## What it does and does not license

**Does not**: `fastino` at 4/6 with 31.7% false positives is not shippable. Flagging a third
of genuine casualty figures as belonging to another storm would be worse than the disease.

**Does**: the signal exists in a model we already have, and the published verdict that
"C via the structure/record path is dead on every model available" was a verdict on a broken
instrument. The right next question is whether a corpus carrying **named event identities**
preserves the base model's binding through fine-tuning, rather than whether the signal exists
at all.

Two of six is also the ceiling nobody has beaten: Bosnia's 16 and the 1916 hurricanes' 80 are
missed by every model, and both are cases where the competing event is not a named storm in
the window.
