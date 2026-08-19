# Control rows for the cross-event muting arm

Collected 2026-08-19 on CPU with the probe fixed in `63249ed`, before spending any GPU on
`tools/train/config/casualty-loc-muted.yaml`. Raw output kept verbatim beside this file.

    uv run python tools/ekf_showcase/event_binding_probe.py --model <id> --device cpu

| model | C catches | C false pos | binds an event (cross / ok) | rejected | unbound | location fill | event fill |
|---|--:|--:|--:|--:|--:|--:|--:|
| `fastino/gliner2-base-v1` | 3/11 | 31.3% | 10/11 · **55**/83 | 0 · 5 | 1 · 23 | 86.2% | 80.0% |
| `...-casualty-loc-split` (fp16, epoch 6) | 3/11 | **9.6%** | 7/11 · **23**/83 | 1 · 4 | 3 · 56 | 91.5% | 41.9% |
| `...-casualty-loc-split-clean` (bf16, 8 ep) | 1/11 | **3.6%** | 5/11 · **12**/83 | 0 · 1 | 6 · 70 | 83.5% | 23.1% |

"binds an event" is `competitor + ours` — the model named something the event schema also
found. "rejected" is a binding the validator threw out because the string names no event (the
artifact that made a casualty-trained head read 9/11 before `63249ed`), so it is neither a
bind nor an abstention and is counted separately.

## The noise floor, measured — because one number did not reproduce

The published `fastino` row is **3/11 @ 30.1%**. The first run here returned exactly that;
every run since returns **31.3%**. That is 25 against 26 of 83, a single observation. The
run that produced 30.1% was on the probe before field-fill measurement was added, and moving
that measurement to a second pass did not bring 30.1% back, so it is not attributable to the
change — it is one observation of drift.

Four consecutive runs of the clean control:

| run | C | location fill |
|---|--:|--:|
| 1 | 1/11 @ 3.6% | 87.3% |
| 2 | 1/11 @ 3.6% | 83.5% |
| 3 | 1/11 @ 3.6% | 83.5% |
| 4 | 1/11 @ 3.6% | 86.4% |

**Signal C is stable and field fill is not.** C returned identical counts in all four runs;
location fill spans 83.5–87.3%, a **3.8 point spread**, and `fastino`'s event fill moved
80.0 → 84.6% between runs. Any fill guard therefore needs a tolerance of at least 4 points,
and a fill difference smaller than that is not a result. This is why the four runs are kept
here rather than summarized away.

`...-casualty-loc-split` returned 3/11 @ 9.6% on every run, matching its published number.

## Row 3 is the one that matters, and it changes a guard

`casualty-loc-split-clean` is the model the muting arm is read against, and its binding
numbers had never been measured — the published 9.6% belongs to the *crashed fp16* run, not
to the clean model that supersedes it. Measured, it scores **1/11 @ 3.6% FP**, which looks
like the best precision in the table and is nothing of the sort. It binds **12 of 83**
ordinary observations against the fp16 model's 23 and the base model's 55, and leaves **6 of
11** cross-event cases unbound. That rate is **abstention, not precision**.

**Consequence: "binding FP no worse than X" is not a valid guard on its own.** A model that
binds nothing scores 0%. This is the same artifact the coverage table was added to expose —
item 0's boundary arm scored `0/11 @ 0.0% FP`, which the old table rendered as perfect
precision. The muting arm is trained to *suppress*, so it is exactly the kind of arm that
could win on false positives by abstaining. Its pass/fail therefore pairs the rate with a
coverage floor.

## Two side-observations, neither acted on here

**More training made the binder worse.** The clean 8-epoch run has better structure F1
(0.8485) than the fp16 6-epoch run and roughly half its binding coverage. That is consistent
with `casualty-loc-split.yaml`'s own finding that field extraction on this corpus saturates
in one epoch, and suggests the extra epochs bought specialization at the cost of the record
head's willingness to fill an out-of-schema `event` field.

**Location fill survives fine-tuning; event fill does not.** Location holds at 83–92% across
all three models, while event fill falls 80.0% → 41.9% → 23.1% from base to fp16 to clean.
Location is in the training schema and `event` is not, so the models progressively stop
answering a field they were never supervised on. Relevant to the muting arm, which also does
not supervise `event`.

Neither is a claim; each needs its own arm.
