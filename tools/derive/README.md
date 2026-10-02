# derive_config — measure a corpus against a base model

Point it at a pre-split corpus and a pretrained GLiNER2 checkpoint; it measures what a
fine-tuning config needs and writes the evidence. One run per (corpus, base model) pair.

**Status (2026-10-02):** stages 1–3 run today (CPU only). Stages 4–7 (GPU calibrations and
the YAML writer) are not built yet — see [Roadmap](#roadmap).

## Requirements

- This repo, `uv sync` done.
- `HF_TOKEN` exported if the model or its tokenizer is a private Hub repo.
- A corpus as `<base>.train.jsonl`, `<base>.val.jsonl`, `<base>.test.jsonl` in GLiNER2 format
  (e.g. `convert_ace2005.py` output). All three files must exist.

## Run

```bash
uv run python tools/derive/derive_config.py \
    --corpus /path/to/ace2005_v3 \
    --model whr778/gliner2-eb18-balanced \
    --out derived/
```

`--model` takes a Hub id (`fastino/gliner2.5-multi-v1`, `fastino/gliner2-base-v1`,
`whr778/gliner2-eb17-best`, …) or a local checkpoint directory (`out/my-run/best`).

Every permutation against every base:

```bash
for corpus in /data/ace/*/ace2005; do
  for model in fastino/gliner2.5-multi-v1 whr778/gliner2-eb18-balanced; do
    uv run python tools/derive/derive_config.py --corpus "$corpus" --model "$model" --out derived/
  done
done
```

Output names are `<corpus-dir-name>__<model-name>`. Give each permutation its own directory
(or base name) so outputs do not collide.

## What it prints

```
[derive] ace2005__gliner2.5-multi-v1: arch=boundary encoder=microsoft/mdeberta-v3-base max_len=4096
[derive] split hygiene: duplicates {...} overlap {...}  clean | *** NOT CLEAN ***
[derive] gold capacity: {'recommended_cap': 384, 'pct_groups_over_32': 2.9, 'largest_group': 415, ...}
```

**`*** NOT CLEAN ***` means stop**: duplicate inputs inside a split, or the same document in
two splits. Fix the corpus before training; a contaminated test set is not a measurement.

## Outputs

### `<name>.calibration.json`

| key | what it is | config value it informs |
|---|---|---|
| `model.architecture` | `span` or `boundary` (legacy span models carry no `architecture` key) | which config keys apply at all |
| `model.encoder`, `model.max_len`, `max_len_source` | encoder and its length limit (from `config.max_len`, else the encoder's position limit) | `training.max_len`, windowing |
| `model.tokenizer_source` | where lengths were tokenized from (the checkpoint, else its encoder) | — |
| `corpus.heads.<split>` | gold per head: entities, relations, events, triggers, arguments, … | `metric_for_best`, which heads to score |
| `corpus.lengths.<split>` | input length p50/p90/p99/max **in the base model's subwords** | `max_len`, `window_stride`, eval `chunk_size` |
| `corpus.uniqueness` | duplicates per split and cross-split overlap | the hygiene gate |
| `corpus.gold_capacity` | largest gold group per (document, label) and the recommended cap | `max_gold_per_query`, `training_candidate_budget` |
| `corpus.labels` / `label_review` | every label with use counts, and its review status | label map additions |

`gold_capacity` counts event triggers/arguments only when the base does **not** use
`event_records` (with it, events go through the record head), so the same corpus gets a
different cap per base. It reads the first 3,000 train documents and counts surface
occurrences, which overcounts slightly — the safe direction.

### `<name>.labels_review.md`

Labels are an input to GLiNER2, so a fine-tune should present the base's own spelling of a
concept it already knows. Each corpus label gets one status:

| status | meaning | action |
|---|---|---|
| `known` | a spelling the base trained on | none |
| `mapped` | the base's own `label_map` rewrites it | none (replayed automatically) |
| `fold_match` | same letters as a known label up to case/punctuation | **confirm by reading the surfaces it tags** — never merge on spelling alone |
| `new` | no counterpart in a category the base lists in full | trains as a new label |
| `unlisted` | no counterpart, but the category is open-vocabulary in the base, which keeps no full inventory | not necessarily new; check |
| `open_vocab` / `open_vocab_code_like` | the base carries no label inventory (e.g. fastino models) | give code-like labels (`PER.Individual`, `Vulnerable_System`) a natural-language name |

## Roadmap

| stage | needs | measures | config value | built |
|---|---|---|---|---|
| 1 model probe | CPU | architecture, encoder, limits, label inventory | — | ✅ |
| 2 corpus probe | CPU | heads, lengths, hygiene, gold capacity | windows, capacity, metric | ✅ |
| 3 label review | CPU | label status per category | label map additions | ✅ |
| 4 reachability | GPU | gold reachable at several `start_top_k` (`probe_candidate_coverage.py`) | `start_top_k` / `end_top_k` | — |
| 5 zero-shot baseline | GPU | the base's own scores on **val** before training | the "did training help" control | — |
| 6 operating points | GPU | span threshold and record-gate sweeps on **val** | eval `threshold`, `record_anchor_threshold` | — |
| 7 YAML writer | — | — | the config, every value annotated with its source | — |

Policy values (exact replay ~30% of the mix, warm-start learning rates, negatives on,
`validate_data: true`, no inline `labels:` block) will be written by stage 7 as marked
defaults, not measurements. Operating points from stage 6 are a **starting** point: re-sweep
on validation after fine-tuning, because calibration moves as a model trains.
