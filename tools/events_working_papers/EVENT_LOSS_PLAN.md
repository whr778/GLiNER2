# Event Loss Separation Plan

**Goal**: Separate the event-task contribution to `structure_loss` into its own
`event_structure_loss` bucket, with no change to the gradient signal.  All other
losses (`classification_loss`, `structure_loss`, `count_loss`) retain their
current meaning; `total_loss` is numerically identical before and after.

---

## Background

`_compute_sample_loss` in `gliner2/model.py` routes all non-classification
structure tasks through a single `struct_loss` accumulator:

```
task_type == "entities"   → struct_loss
task_type == "relations"  → struct_loss + count_loss
task_type == "events"     → struct_loss + count_loss   ← split this out
task_type == "classifications" → cls_loss
```

After separation:

```
task_type == "entities"   → struct_loss
task_type == "relations"  → struct_loss + count_loss
task_type == "events"     → event_struct_loss + count_loss  (count stays combined)
```

The combined `total_loss` is:

```python
total_loss = cls_loss + struct_loss + event_struct_loss + count_loss
```

This is a **pure bucketing change** — the tensors are the same; only the
grouping changes.  Verify post-implementation: on an event-bearing batch,
`new.structure_loss + new.event_structure_loss == old.structure_loss`.

> **Note on `count_loss`**: Events also contribute to `count_loss` (alongside
> relations). This plan leaves `count_loss` combined. If you want a single
> unified `event_loss = event_structure_loss + event_count_loss`, note that in
> a follow-up and this plan handles just the structural piece.

---

## Files to change

### 1. `gliner2/model.py`

#### A. `_compute_sample_loss` (lines ~424–482)

Initialise a new accumulator alongside the existing three:

```python
cls_loss        = torch.tensor(0.0, device=device)
struct_loss     = torch.tensor(0.0, device=device)
event_struct_loss = torch.tensor(0.0, device=device)   # NEW
count_loss      = torch.tensor(0.0, device=device)
```

In the task loop, route `"events"` into `event_struct_loss` instead of
`struct_loss`:

```python
# Before (line ~460):
if span_info is not None:
    struct_loss = struct_loss + self.compute_struct_loss(...)

# After:
if span_info is not None:
    if task_type == "events":
        event_struct_loss = event_struct_loss + self.compute_struct_loss(...)
    else:
        struct_loss = struct_loss + self.compute_struct_loss(...)
```

Return value (add `"event_structure"` key):

```python
return {
    "classification":    cls_loss,
    "structure":         struct_loss,
    "event_structure":   event_struct_loss,   # NEW
    "count":             count_loss,
}
```

#### B. `forward` — loss collection (lines ~269–344)

Add the new list parallel to `struct_losses`:

```python
cls_losses          = []
struct_losses       = []
event_struct_losses = []   # NEW
count_losses        = []
```

In the try-branch (success path), append:

```python
cls_losses.append(sample_losses["classification"])
struct_losses.append(sample_losses["structure"])
event_struct_losses.append(sample_losses["event_structure"])   # NEW
count_losses.append(sample_losses["count"])
```

In the except-branch (error path, currently lines ~305–311), append a zero:

```python
cls_losses.append(zero)
struct_losses.append(zero)
event_struct_losses.append(zero)   # NEW
count_losses.append(zero)
```

Update `individual_losses` (success path, lines ~292–301):

```python
individual.append({
    "total_loss":            (
        sample_losses["classification"]
        + sample_losses["structure"]
        + sample_losses["event_structure"]   # NEW
        + sample_losses["count"]
    ).item(),
    "classification_loss":   sample_losses["classification"].item(),
    "structure_loss":        sample_losses["structure"].item(),
    "event_structure_loss":  sample_losses["event_structure"].item(),   # NEW
    "count_loss":            sample_losses["count"].item(),
})
```

Update `individual_losses` (error path, lines ~313–319):

```python
individual.append({
    "total_loss":           0.0,
    "classification_loss":  0.0,
    "structure_loss":       0.0,
    "event_structure_loss": 0.0,   # NEW
    "count_loss":           0.0,
    "error":                str(e),
})
```

Aggregate and include in total (lines ~328–343):

```python
total_cls          = torch.stack(cls_losses).sum()
total_struct       = torch.stack(struct_losses).sum()
total_event_struct = torch.stack(event_struct_losses).sum()   # NEW
total_count        = torch.stack(count_losses).sum()
total_loss = total_cls + total_struct + total_event_struct + total_count

result = {
    "total_loss":            total_loss,
    "classification_loss":   total_cls,
    "structure_loss":        total_struct,
    "event_structure_loss":  total_event_struct,   # NEW
    "count_loss":            total_count,
    "batch_size":            valid_samples,
}
```

#### C. `_empty_loss_dict` (lines ~346–362)

Add the new key:

```python
return {
    "total_loss":            torch.tensor(0.0, device=device, requires_grad=True),
    "classification_loss":   torch.tensor(0.0, device=device),
    "structure_loss":        torch.tensor(0.0, device=device),
    "event_structure_loss":  torch.tensor(0.0, device=device),   # NEW
    "count_loss":            torch.tensor(0.0, device=device),
    "batch_size":            0,
}
```

---

### 2. `gliner2/training/trainer.py`

#### A. `TrainingMetrics` dataclass (lines ~427–437)

```python
@dataclass
class TrainingMetrics:
    loss:                   float = 0.0
    classification_loss:    float = 0.0
    structure_loss:         float = 0.0
    event_structure_loss:   float = 0.0   # NEW
    count_loss:             float = 0.0
    learning_rate:          float = 0.0
    epoch:                  float = 0.0
    step:                   int   = 0
    samples_seen:           int   = 0
    throughput:             float = 0.0
```

#### B. Training-loop metrics logging (lines ~1268–1280)

```python
metrics = TrainingMetrics(
    loss=avg_loss,
    classification_loss=outputs.get("classification_loss", torch.tensor(0)).item(),
    structure_loss=outputs.get("structure_loss", torch.tensor(0)).item(),
    event_structure_loss=outputs.get("event_structure_loss", torch.tensor(0)).item(),   # NEW
    count_loss=outputs.get("count_loss", torch.tensor(0)).item(),
    ...
)
```

#### C. `_evaluate` — accumulator and return (lines ~1418–1449)

Add accumulator:

```python
total_event_struct_loss = 0.0   # NEW
```

Collect per batch:

```python
total_event_struct_loss += outputs.get(
    "event_structure_loss", torch.tensor(0)
).detach().cpu().item()   # NEW
```

Add to returned metrics dict:

```python
metrics = {
    "eval_loss":                  self._safe_divide(total_loss, num_batches),
    "eval_classification_loss":   self._safe_divide(total_cls_loss, num_batches),
    "eval_structure_loss":        self._safe_divide(total_struct_loss, num_batches),
    "eval_event_structure_loss":  self._safe_divide(total_event_struct_loss, num_batches),   # NEW
    "eval_count_loss":            self._safe_divide(total_count_loss, num_batches),
    "step":  self.global_step,
    "epoch": self.epoch,
}
```

Also update the empty-eval-loader early-return (lines ~1409–1416):

```python
return {
    "eval_loss":                  0.0,
    "eval_classification_loss":   0.0,
    "eval_structure_loss":        0.0,
    "eval_event_structure_loss":  0.0,   # NEW
    "eval_count_loss":            0.0,
    "step":  self.global_step,
    "epoch": self.epoch,
}
```

---

## Parallel / DataParallel compatibility

`gliner2/training/parallel.py:reduce_loss_dicts` (line ~102) iterates
`outputs[0].keys()` dynamically, so it already handles any new dict key.
No change needed.

---

## Tests to update / add

| File | What to check |
|------|---------------|
| `tests/test_struct_loss_variants.py` | Add assertions that `event_structure_loss` is non-zero on event-bearing batches and zero on entity-only batches. |
| `tests/test_mmbert_integration.py` | Confirm `event_structure_loss` key exists in forward output (or add a simple smoke test). |
| New inline invariant test | On a batch with at least one event sample: `assert abs((result["structure_loss"] + result["event_structure_loss"]).item() - old_structure_loss) < 1e-5`. |

---

## Implementation order

1. [x] `_compute_sample_loss`: add `event_struct_loss` accumulator, route events.
2. [x] `forward`: add list, both append sites (try + except), individual_losses,
   aggregate, result dict.
3. [x] `_empty_loss_dict`: add key.
4. [x] `TrainingMetrics`: add field.
5. [x] Training-loop logging: add field.
6. [x] `_evaluate`: add accumulator, per-batch collect, return dict, empty-loader return.
7. [x] Run tests; confirm invariant on a live batch.
    - 12 existing tests pass (`test_struct_loss_variants`, `test_mmbert_integration`).
    - Smoke test confirms: `event_structure_loss > 0` on event batch, `structure_loss == 0`,
      `total_loss == cls + struct + event_struct + count` (diff = 0.00e+00).

---

## What this does NOT change

- `structure_loss` keeps its name (entities + relations).
- `total_loss` gradient is numerically identical.
- `count_loss` includes both relation and event count contributions (no split).
- No changes to checkpoint format (`event_structure_loss` is a forward-pass
  metric, not a saved parameter).

---

## Follow-on: per-task loss variant config

**Implemented.** `ExtractorConfig` gained two optional fields:

- `event_struct_loss: str = None` — loss variant for event tasks; `None` falls
  back to `struct_loss`.
- `event_struct_pos_weight: float = None` — positive weight for `bce_posweight`
  on events; `None` falls back to `struct_pos_weight`.

### Changes made

**`gliner2/model.py`**

- `ExtractorConfig.__init__`: added `event_struct_loss` and
  `event_struct_pos_weight` parameters and stored them on `self`.
- `compute_struct_loss`: added optional `variant` and `pos_weight` kwargs
  (both `None` → read from config as before); passes them to `_struct_loss_term`.
- `_struct_loss_term`: added optional `variant` and `pos_weight` kwargs;
  `variant=None` reads `config.struct_loss`; `pos_weight=None` reads
  `config.struct_pos_weight` for the `bce_posweight` branch.
- `_compute_sample_loss`: for `task_type == "events"`, resolves
  `event_struct_loss` / `event_struct_pos_weight` from config and passes them
  to `compute_struct_loss`; other task types call `compute_struct_loss` with
  no overrides (unchanged behaviour).

**`tools/train/TRAINING.md`**: added "Per-task loss metrics and event-specific
loss variant" subsection documenting the new YAML keys and the metric breakdown.

### YAML example

```yaml
model:
  encoder: jhu-clsp/mmBERT-small
  struct_loss: bce                    # entities + relations
  event_struct_loss: bce_posweight    # events only
  event_struct_pos_weight: 16.0
```

### Verified

- 12 existing tests pass unchanged.
- Smoke test: with identical weights, `bce_posweight` (pos_weight=50)
  yields `event_structure_loss` ≈ 171 vs plain `bce` ≈ 69; `structure_loss`
  is identical in both runs (entity/relation path unaffected); `total_loss`
  reconstruction diff = 0.0.
