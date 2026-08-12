# rams_baseword — base-word (lemmatized) duplicate samples

RAMS train with a second copy of each record in which every surface word — text **and**
labels — is reduced to its base form. Both variants stay in the mix (1:1 duplication, not
replacement), so `killed` / `killing` / `kills` stop learning the same trigger-role
association three times under-powered. Built for [[TODO]] item 12.

| | |
|---|--:|
| source | `data/rams.train.jsonl` (7,329 records) |
| `train.jsonl` | **13,291 records** (5,962 augmented, 81.3%) |
| backend | `simplemma` 1.2.0, `--lang en` |
| gate | **PASS** — gold mentions 27,599 against 27,599, zero records changed |

**Train only, on purpose.** `rams.dev.jsonl` and `rams.test.jsonl` are deliberately not
augmented: an augmented eval set is not comparable with the un-augmented baseline or with
any historical number.

The 1,367 refused records are dropped whole rather than emitted partially — partial
augmentation is the silent-supervision-loss failure the pass exists to avoid. The original
is always emitted, so a refusal costs the duplicate, never the record.

## Verify before training on it

```bash
uv run python tools/data/check_augment_alignment.py \
  --original data/rams.train.jsonl --augmented datasets/rams_baseword/train.jsonl
```

Exit 0 on PASS, 1 on FAIL. Two things this gate exists to catch, neither of which raises:

1. **`missing_surface_counts()` cannot serve as the check here.** It increments only for
   `task_type == "entities"`; RAMS supervises `events`, where an unlocatable surface is
   treated as legitimately absent and skipped with no counter at all. It reads 0 no matter
   what breaks. The gate reads the target graph instead.
2. **Gold can be INVENTED, not just lost.** Lemmatization collapses surface forms, so a
   label starts matching positions that were never annotated — gold `guns` occurs once,
   and as `gun` it occurs three times. Before the guard this inflated gold by 1,085
   mentions on 31,773 (3.4%) across 718 records, every one a silent false positive. The
   builder now refuses any record whose label occurrence count changes, counted in the
   tokenization collation actually uses.

## Rebuild

```bash
uv run python tools/data/augment_baseword.py \
  --in data/rams.train.jsonl --out datasets/rams_baseword/train.jsonl \
  --backend simplemma --lang en
```

`simplemma` is not yet a project dependency — it was installed to a scratch directory and
used via `PYTHONPATH` while a long precompute was running out of the same virtualenv, since
`uv add` re-locks and syncs the whole environment. Run `uv add simplemma` once that job has
exited, then this command works as written.

**Language gating.** Lemmatization is a no-op for Chinese and a different operation for
agglutinative languages, so this is opt-in per corpus rather than applied across `data/`.
