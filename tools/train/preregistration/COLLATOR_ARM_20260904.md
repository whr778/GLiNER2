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
