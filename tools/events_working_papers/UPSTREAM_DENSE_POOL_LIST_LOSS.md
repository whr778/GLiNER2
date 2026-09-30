# Upstream report: dense record loss averages the list-field term over padding

**Status:** FILED 2026-09-29 as https://github.com/fastino-ai/GLiNER2/issues/180,
alongside PR #155 and issue #156. Filed as an ISSUE, not a PR -- see "Why not a PR" below.

**FIXED LOCALLY 2026-09-30 (426a38d)**, upstream still open. `compute_dense_group_loss` now
means the list-field BCE over `group.field_membership` instead of the padded pool width. On
the test fixture (10 real spans of 64) the dense `field_loss` went 0.863781 -> 1.016330,
identical to the sparse path; `test_dense_and_sparse_record_losses_agree_on_identical_input`
lost its strict xfail marker and now passes; boundary suite 392 passed, 6 skipped.
Consequence for past evidence: the 2026-09-21 `candidate_pool` A/B trained its shared arms
WITH this bug, so its event-head deltas are confounded; its entity verdict (-0.0745) is not,
because the entity head does not use the record loss.

**Affects:** `boundary_head.candidate_pool: "shared"` only. The default `"per_query"`
routes to `compute_group_loss` and is unaffected.

---

## Summary

`compute_dense_group_loss` reduces its list-field BCE across the **full padded candidate
pool width** rather than across the real spans, so the list-field loss is scaled by
exactly `n_real / n_padded`. The scalar branch is unaffected.

`gliner2/models/boundary/records.py`, in `compute_dense_group_loss`:

```python
list_nll = F.binary_cross_entropy_with_logits(
    group.assign_logits[:, None, :, 1:].expand(-1, count, -1, -1),
    gold[None].expand(ni, -1, -1, -1).to(group.object_logits.dtype),
    reduction="none",
).mean(-1)          # <-- every pool column, most of them padding
```

The mask that would fix it is already on the group and is used for the *targets* but not
for the *reduction*: `group.field_membership` is `[n_fields, pool]` and counts exactly the
real spans. `instance_mask` is applied correctly to the object term a few lines above, so
the pattern is already established in the same function.

## Why this is visible at all

`forward_group` and `forward_group_dense` take identical arguments, so the same
`(spec, query_states, candidates, sample_index)` can build both representations. That
removes the candidate-set confound and leaves only the loss arithmetic.

## Measurement

Tiny fixture, one natural-mode group, 2 fields (1 scalar, 1 list), 10 real spans in a
64-wide pool (`pool_spans` has 64 rows, 10 non-zero; `instance_mask.sum() == 10`;
`field_membership.sum() == 20 == 10 x 2`).

| term | per_query | dense | note |
|---|---|---|---|
| `object_loss` | 0.000000 | 0.000000 | agree (both zero in natural mode) |
| `field_count` | 2 | 2 | agree |
| **`field_loss`** | **1.016330** | **0.863781** | **disagree** |

Reconciliation, treating the list term as diluted by `10/64`:

```
sparse field_loss     = (1.671063 + 0.361597) / 2          = 1.016330   (observed)
predicted dense list  =  0.361597 * 10/64                  = 0.056500
predicted dense field = (1.671063 + 0.056500) / 2          = 0.863781
observed  dense field                                       = 0.863781
residual                                                    = 2.7e-07
```

The scalar term is identical in both paths (1.671063), which is why only the list branch
is implicated: `log_softmax` / `logsumexp` absorbs the padding columns, a plain `mean`
does not.

Ruled out along the way, each by direct measurement rather than inspection:

- **not the instance matching.** `compute_group_loss` maps anchor -> instance through
  `seed_to_inst` while the dense path assumes `instance_row == candidate_column`; on this
  input `seed_to_inst` IS the identity and both select the same `(gold record 0 ->
  instance row 0)`. (This assumption may still be worth checking separately -- it is not
  obviously guaranteed, it just holds here.)
- **not the field reduction.** Both average over fields: `_instance_field_loss` does
  `total / max(n_fields, 1)`, the dense path does `.mean(-1)`.
- **not the logits.** Field 0's rows agree to 1e-6 over the overlapping prefix.

## Impact

The list branch is where record LIST fields train. With `event_records: true` every event
argument role is a list field, so this term carries all argument supervision. The dilution
scales with `pool_size` (default 384), so on a real document with tens of candidates the
under-weighting is far larger than the 10/64 measured here.

It also means `candidate_pool` is not a pure representation switch: building the head both
ways adds, removes and reshapes no tensors, but the loss VALUE is not preserved, so a warm
start or an A/B across that flag is confounded.

## Reproduction

`tests/models/boundary/test_dense_pool_loss_equivalence.py`, marked
`xfail(strict=True)`. It builds one group through the real per-query path, rebuilds it via
`forward_group_dense` from the same arguments, and compares. Current output:

```
E  AssertionError: dense field_loss is diluted by padding columns
E  assert 0.8637813329696655 == 1.0163298845291138 +/- 1.0e-04
```

A companion test pins the premise (the pool is mostly padding, and `field_membership`
counts exactly the real spans) and passes today.

## Suggested fix

Apply the existing mask to the reduction rather than only to the targets, mirroring what
`instance_mask` already does for the object term -- i.e. sum the masked elementwise BCE and
divide by the per-field count of real candidates instead of taking a plain `.mean(-1)`.

## Why not a PR

1. **It may be intended.** A fixed denominator makes the loss scale independent of how many
   candidates survive per document; masking makes it vary. Which behaviour is wanted is a
   design question only upstream can answer.
2. **The fix changes loss magnitude ~6x on this fixture** (list term 0.0565 -> 0.3616), so
   anyone training on `shared` would need to retune `record_loss_weight`. That is a
   regression risk we cannot price.
3. **We do not run this path.** Our configs are `per_query`, so we cannot validate the fix
   on real data or at real pool sizes. The arithmetic is exact; the training impact is
   unmeasured, and those are different claims.

Happy to turn this into a PR with the one-line change plus the test if the behaviour is
confirmed unintended.
