# Merging `origin/main` into `mmbert_training` — Conflict Map & Strategy

Status: read-only scouting (no merge performed). Date: 2026-08-04.
Purpose: map exactly what collides before we merge `origin/main` (the boundary /
joint-IE rewrite) into our feature branch, so the merge is a planned port rather
than a cold conflict fight. Companion decision: the merge is **deferred until the
scaling-curve runs + PAPER are done and the A100 is deprovisioned** (see
`mmbert-head-init-finding` / the scaling-curve work). Regenerate this map with
`git fetch` + `git merge-tree` before executing.

## 1. Divergence (as of this scouting)

| | ref | commits since base | files changed |
|---|---|--:|--:|
| ours | `mmbert_training` @ `5f145f3` | 212 | 251 |
| theirs | `origin/main` @ `a91fd1d` | 23 | 201 |

- **merge-base:** `31c8aba` (2026-07-17).
- **Overlap (both sides touched the same file): 17 files.** Everything else is
  disjoint — our 251 are mostly `tools/data` (converters), `tools/train`
  (configs), `tools/events_working_papers`, `viewer`, `tutorial`, `PAPER.md`;
  their 201 are the core rewrite + new packages + a test-dir move.
- Of the 17 overlap files, **10 textually conflict, 7 auto-merge** (per
  `git merge-tree --write-tree HEAD origin/main`).

## 2. What `origin/main` is (their 23 commits)

A core **architecture expansion**, not bugfixes: `add boundary architecture`
(#1/#2/#3), `add joint ie decoding`, `add classification api`, `add entity
attribute`, `update engine`, `some optimization`. It lands as **four new,
purely-additive packages** (we have zero files in them — **no conflict, comes in
clean**):

| new package | files on main |
|---|--:|
| `gliner2/models/` | 25 |
| `gliner2/joint_ie/` | 16 |
| `gliner2/classification/` | 15 |
| `gliner2/processing/` | 6 |

…plus a **near-total rewrite of the model/inference/trainer core** (see §3) and a
**test-directory reorganization** (`move test dir`; ~105 test files, e.g. new
`tests/training/`). Note `gliner2/old_trainer.py` is *deleted* by main (a base-era
legacy file; we never touched it, so it deletes cleanly — not a rename trap).

## 3. Conflict map (the 17 overlap files)

Line counts are `added/deleted` on each side vs the merge-base. "theirs" with a
huge delete count = they rewrote the file, so our change is a **re-port target**,
not a hunk-by-hunk merge.

| file | ours +/− | theirs +/− | merge | severity |
|---|--:|--:|---|---|
| `gliner2/training/trainer.py` | 665/159 | 809/146 | **CONFLICT** | **CRITICAL** — both rewrote |
| `gliner2/training/metrics.py` | 1184/0 | 148/0 | **CONFLICT (add/add)** | **CRITICAL** — two independent files |
| `gliner2/model.py` | 298/53 | 11/952 | **CONFLICT** | **HIGH** — theirs gutted it (−952) |
| `gliner2/inference/engine.py` | 238/10 | 34/1195 | **CONFLICT** | **HIGH** — theirs gutted it (−1195) |
| `gliner2/inference/schema.py` | 127/0 | 229/9 | **CONFLICT** | **HIGH** — both large |
| `gliner2/api_client.py` | 133/5 | 117/390 | **CONFLICT** | **HIGH** — theirs rewrote (−390) |
| `gliner2/training/__init__.py` | 35/0 | 21/0 | **CONFLICT** | LOW — small export list |
| `gliner2/inference/chunking.py` | 16/1 | 89/15 | **CONFLICT** | LOW — take theirs, reapply our 16 |
| `tests/test_batch_span_mask_trim.py` | 2/0 | 21/43 | **CONFLICT** | LOW — take theirs |
| `.gitignore` | 10/1 | 12/1 | **CONFLICT** | TRIVIAL — union |
| `gliner2/processor.py` | 137/4 | 257/25 | auto-merge | **REVIEW** — big both sides |
| `gliner2/training/data.py` | 209/9 | 62/16 | auto-merge | **REVIEW** — big ours |
| `gliner2/inference/schema_model.py` | 43/2 | 29/0 | auto-merge | ok |
| `pyproject.toml` | 24/1 | 7/0 | auto-merge | ok |
| `README.md` | 31/0 | 21/11 | auto-merge | ok |
| `tests/test_torch_free_import.py` | 9/0 | 4/4 | auto-merge | ok |
| `tests/test_trainer_distributed_integration.py` | 21/17 | 70/0 | auto-merge | ok |

**Auto-merge ≠ safe.** `processor.py` and `training/data.py` auto-merge textually
but both sides changed them heavily — they need a **semantic** review after the
merge (our sliding-window/event data path vs their processing rewrite).

## 4. The crux

The real labor is **four core files** — `trainer.py`, `metrics.py`, `model.py`,
`engine.py` — where main rewrote and we also rewrote. Because main *gutted*
`model.py` (−952) and `engine.py` (−1195), our additions there cannot be resolved
as text hunks; they must be **re-implemented against main's new structure**. Our
feature set to re-port onto the new core:

- **mmBERT long-context**: `max_len 8192`, `sliding_window`, native whole-doc eval.
- **Event / document decoding**: `global_decode`, `chunk_size`/`chunk_overlap`,
  OneIE-style cross-window event assembly.
- **Loss variants**: `struct_loss` ∈ {`bce_posweight`, `focal`, `dice`, …},
  `struct_pos_weight`.
- **Eval machinery** (`metrics.py`, ours = 1184 lines): strict/relaxed micro-F1
  per task (entity/relation/event-trigger/-argument/-type), `threshold_sweep`,
  `metric_sweep` per-epoch checkpoint selection.
- **DDP**: `data_parallel`, the distributed trainer path.

The new packages (`models/joint_ie/classification/processing`) most likely *depend
on* main's rewritten `model.py`/`engine.py`, so cherry-picking them without taking
the core rewrite is not viable — the merge has to take main's core as the baseline.

## 5. Recommended strategy

**Take main's rewritten core as the new baseline and re-port our features onto it**
— i.e. treat this as a port, not a textual conflict resolution.

1. **Merge in stages on a throwaway branch** (`merge/main-<date>`), never directly
   on `mmbert_training`.
2. **Free wins first:** the disjoint 234 files (our tools/config/docs/viewer +
   their 62 additive package files) merge with no conflict — let them.
3. **Trivial conflicts:** `.gitignore` (union), `training/__init__.py`,
   `inference/chunking.py`, `tests/test_batch_span_mask_trim.py` (take theirs) — minutes.
4. **Core port (the work):** for `model.py`, `engine.py`, take **theirs** as the
   file, then re-apply the §4 features guided by our base→HEAD diff of each file
   (`git diff <base> HEAD -- <file>` as the spec of what to re-add).
5. **`metrics.py` (add/add):** keep **ours** (the 1184-line strict/relaxed +
   sweep machinery) and graft whatever their 148-line version adds; verify metric
   names still match the configs (`eval_event_argument_strict_micro_f1`, etc.).
6. **`trainer.py`:** the hardest — reconcile our DDP + event-eval + metric_sweep
   loop against their new trainer. Budget the most time here.
7. **Semantic review:** `processor.py`, `training/data.py` (auto-merged but risky).
8. **Validate:** full `pytest`, then reproduce one cheap config end-to-end
   (e.g. a from-encoder RAMS run) to confirm the training path still yields sane
   numbers before trusting the merged core.

## 6. Risk & sequencing

- **Do not merge until the scaling curve + PAPER are complete and the A100 is
  deprovisioned.** The curve requires all points on identical code; a core rewrite
  mid-experiment invalidates the 10k/40k/100k comparison. (Standing decision.)
- After that: execute §5 on a throwaway branch, keep `mmbert_training` intact until
  the merge branch is green.
- Blast radius is bounded: **4 core files carry ~all the risk**; the other 13
  overlaps are trivial or auto-merge, and 234 files are disjoint.

## 7. Related
- Memory: `main-boundary-rewrite-merge-deferred` (the deferral), `merge-main-ddp-decision`.
- Our core feature surface lives in the base→HEAD diffs of the four §4 files.
- Regenerate: `git fetch origin && git merge-tree --write-tree HEAD origin/main`.
