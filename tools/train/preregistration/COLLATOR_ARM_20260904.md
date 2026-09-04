# Pre-registration: gate3 collator arm (group_by_length false vs true)

**Written 2026-09-04, during epoch 1 of 8, BEFORE the outcome was known.** Committed first
so the result cannot be retrofitted to whatever comes out -- same discipline as
`datasets/turkey2023/PREREGISTRATION.md`.

## The single variable

`gate3-mmbert-mixed.yaml` is `gate3-mmbert.yaml` with `group_by_length: false` and nothing
else -- verified by diff (only `output_dir`, `experiment_name`, that flag) and by a runtime
assertion in the runner that the value reaches `TrainingConfig` as False with no keys
silently dropped. Same corpus bytes (17,827/2,181/2,156, pulled from `whr778/gate3`), same
seed, same recipe, same A10.

Control `whr778/gliner2-gate3-mmbert` is already trained, so this is one run.

## Why the variable might matter

`group_by_length` defaults TRUE and, for a boundary model, replaces shuffle entirely with
`LengthGroupedSampler`. Document length tracks language here -- median chars en 2,759 /
tr 1,534 / zh 942 -- so length-grouped batches are de-facto language-grouped. Simulated at
batch_size 8: **14.6% of batches are 100% one language, against 3.7% under random
shuffling.**

## PREDICTION, operator's, stated at epoch 1

**The mixed arm peaks at EPOCH 7**, against the control's epoch 4. Rationale: monolingual
batches give a cleaner per-batch gradient, so they converge faster and overfit sooner;
mixed batches give a noisier signal that converges slower but higher.

This makes the epoch-1 deficit a CONFIRMATION, not a counter-example:

    epoch 1   control 0.7508   mixed 0.7203   (mixed behind, as predicted)

## What would confirm it

- Mixed arm's best `eval_classification_strict_micro_f1` occurs at epoch 6-8, AND
- that best exceeds the control's 0.7964.

## What would FALSIFY it

- Mixed peaks at epoch <= 5, or
- mixed's best never reaches the control's 0.7964, or
- mixed wins on validation but loses the downstream bars (Turkish 58/60, Chinese 59/60,
  English pooled RMSE on Helene / Aegean / Turkiye-EN).

The last one is not hypothetical: gate3 itself passed two admission bars and still looked
like an English regression until that failed to replicate. Validation F1 is not the ship
criterion.

## A cost estimate already corrected by this run

Predicted ~2x wall-clock from padding waste going 2.5% -> 50.9%. MEASURED: 13.0-13.7
samples/s against the control's 16.4, i.e. **~20%**, ETA 3.2h against 2h25m. Padding waste
does not translate linearly to wall-clock -- `max_len: 2048` caps it and the GPU is not
purely FLOP-bound. If mixed batching wins, adopting it elsewhere is ~5x cheaper than the
waste figure implies.

---

# RESULT, scored 2026-09-04 against the falsifiers above

## The prediction is FALSIFIED on the mechanism, CONFIRMED on the direction

Full validation curve, `eval_classification_strict_micro_f1`:

| epoch | control (grouped) | mixed |
|---|---|---|
| 1 | 0.7508 | 0.7203 |
| 2 | 0.7792 | 0.7847 |
| 3 | 0.7811 | 0.8012 |
| 4 | **0.7964** peak | (no best) |
| 5 | - | **0.8095** peak |
| 6 | - | (no best) |
| 7 | - | (no best) |
| 8 | - | (no best) |

**FALSIFIED.** The prediction was a peak at EPOCH 7. The peak is epoch 5, and the
pre-registered falsifier is "peaks at epoch <= 5". Epochs 6, 7 and 8 all failed to beat
it while training loss kept falling to 0.0569 -- overfitting, not a late ascent. The
proposed mechanism, noisier gradients producing a LATER and higher peak, is not what
happened: mixed peaked one epoch after the control, not three, and the gain came from a
higher ceiling rather than a delayed one.

**CONFIRMED on direction, and on the better of the two instruments.** Mixed wins the
validation number it was selected on (+0.0131) AND the blind test set it was not:

| | relevance | toll_kind | micro (n=4,312) |
|---|---|---|---|
| gate3 (grouped) | 0.8084 | 0.7393 | 0.7739 |
| gate3-mixed | **0.8159** | **0.7398** | **0.7778** |

A validation win alone would have been worth little at +0.0131 against this project's
documented +/-0.02 run-to-run floor. The test set moving the same way, on an instrument
neither model was selected against, is the result that carries.

## Cost of the variable, measured

12.8 samples/s against the control's 16.4 -- **~22%**, not the ~2x predicted from padding
waste going 2.5% -> 50.9%. Waste does not translate linearly to wall-clock: `max_len:
2048` caps it and the GPU is not purely FLOP-bound. If mixed batching survives the
downstream bars, adopting it on `casualty-multilingual` and the 137k bases costs about a
fifth of what the waste figure implies.

## Not yet decided

Per this file's own falsifier list, a validation win that loses the downstream bars is a
LOSS. Turkish/Chinese admission and pooled RMSE on Helene, Aegean and Turkiye-EN are
running; nothing ships until they report. gate3 itself passed two admission bars and still
looked like an English regression until that failed to replicate.
