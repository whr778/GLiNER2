# ACE 2005 Inference, Scoring, and Viewer Guide

Concise reference for running `tools/infer.py`, `tools/train/score_predictions.py`, and the viewer's **Predictions vs gold** panel against ACE 2005 checkpoints trained with the configs in this directory.

## What the ACE 2005 configs train

All configs under `tools/train/config/ace2005/` warm-start from `whr778/gliner2-eb23a` with `data.office_ace: true`, `labels_file: labels/unified-full-v4.yaml`, and `training.default_schema_mode: dataset`.

- **Entities:** Active (`output.entities`, populated from `entity_facts` on office-ACE records).
- **Relations:** Active (`enable_relations: true`, `relation_keep_gold_pairs: true`, `relation_argument_proposal_threshold: 0.05`).
- **Events (triggers + arguments):** Active (`enable_records: true`, `expand_argument_coref: false` so argument supervision stays on local annotated spans).
- **Entity coref clusters (`MentionLink`):** Active (`entity_coref_link_weight: 0.3`) on `ace2005-eb23a.yaml`, `ace2005-eb23a-schedule.yaml`, `ace2005-eb23a-events.yaml`, `ace2005-coref-enable.yaml`, and `ace2005-lora-eb23a.yaml`. Intentionally disabled only on `ace2005-coref-ignore.yaml` (`training.coref_clusters: ignore`, the A/B treatment arm).
- **Event-trigger coref (`TriggerLink`):** Disabled (`record_coref_link_weight: 0.0`, `record_merge_coreferent: off`).

## Decode flags to keep off (or default)

- **Do not pass `--merge-coreferent link`.** It merges separate event mentions of the same type into multi-trigger event objects (`TriggerLink`), which is unrelated to entity coref (`coref_clusters`), is not trained on ACE 2005 (`record_coref_link_weight: 0.0`), and breaks strict `event_argument` F1 against single-trigger ACE 2005 gold.
- **Leave `--threshold`, `--argument-coref-match`, `--argument-coref-copy`, and `--argument-coref-min-score` unset** unless overriding deliberately. `tools/infer.py` inherits the checkpoint's calibrated `inference_defaults` (`per_head_thresholds`) and `boundary_settings` (`entity_coref_threshold: 0.5`, `record_argument_coref_copy: none`).
- **Always pass `--include-spans` and `--labels-file tools/train/config/labels/unified-full-v4.yaml`** when you plan to score or inspect predictions in the viewer:
  - `--include-spans` emits character offsets so the viewer can draw predictions and `score_predictions.py` can score span-exact `coref` `candidate_recall` and OneIE criteria.
  - `--labels-file` writes `gold_mapped` in the checkpoint's canonical spellings (`PER -> Person`, `ORG -> Organization`, `LOC -> Location`, `FAC -> Facility`, hyphen-to-underscore subtypes). Without it, predictions use mapped spellings while `gold` stays in raw corpus spellings, causing strict-match misses.

## 1. Check the checkpoint's `default_schema`

Before running `--model-schema`, confirm `entities` is listed in `default_schema` (and not hidden under `open_vocab`):

```bash
uv run python -c '
import json
cfg = json.load(open("out/ace2005-eb23a/best/config.json"))
s = cfg.get("default_schema") or {}
print("keys:", list(s.keys()))
print("open_vocab:", s.get("open_vocab"))
print("entities:", s.get("entities"))
print("relations:", s.get("relations"))
print("events count:", len(s.get("events") or {}))
print("inference_defaults:", cfg.get("inference_defaults"))
'
```

## 2. Run inference (`tools/infer.py`)

Both converted GLiNER2 `.jsonl` files and raw office-ACE `.json`/`.jsonl` files work directly (`tools/infer.py` adapts office side lists automatically). On Apple Silicon, add `--device cpu` (or `--batch-size 1` on MPS) for long files; on CUDA, use `--device cuda --batch-size 8`.

### A. Full-ontology decode (`--model-schema`, matches `eval.model_schema: true`)

```bash
uv run python tools/infer.py \
  --model ./out/ace2005-eb23a/best \
  --input data/ace2005_eng.test.jsonl \
  --model-schema \
  --labels-file tools/train/config/labels/unified-full-v4.yaml \
  --include-spans \
  --include-confidence \
  --output out/predictions/ace2005_eng.test.model_schema.preds.jsonl
```

Verify the startup log line includes `entities=`:

```text
[infer] schema entities=..., relations=..., events=...
```

To narrow `--model-schema` to specific task heads:

```bash
uv run python tools/infer.py \
  --model ./out/ace2005-eb23a/best \
  --input data/ace2005_eng.test.jsonl \
  --model-schema \
  --tasks entities,events \
  --labels-file tools/train/config/labels/unified-full-v4.yaml \
  --include-spans \
  --include-confidence \
  --output out/predictions/ace2005_eng.test.ent_ev.preds.jsonl
```

### B. Checkpoints with `open_vocab: ["entities"]` (zero-shot `eb23a` control or older checkpoints)

If `default_schema` lists `"open_vocab": ["entities"]` with no concrete `"entities"` list, `--model-schema` alone drops `open_vocab` and queries zero entities. Either pass `--entities` explicitly or export the ACE checkpoint's `default_schema` and pass `--schema-json` to both arms so control and treatment run on the exact same menu:

Option 1 — add `--entities` alongside `--model-schema`:

```bash
uv run python tools/infer.py \
  --model ./out/ace2005-eb23a/best \
  --input data/ace2005_eng.test.jsonl \
  --model-schema \
  --entities Person,Organization,GPE,Location,Facility,VEH,WEA \
  --labels-file tools/train/config/labels/unified-full-v4.yaml \
  --include-spans \
  --include-confidence \
  --output out/predictions/ace2005_eng.test.model_schema.preds.jsonl
```

Option 2 — export the ACE checkpoint's `default_schema` and run the zero-shot `eb23a` control arm with `--schema-json`:

```bash
uv run python -c '
import json
cfg = json.load(open("out/ace2005-eb23a/best/config.json"))
open("/Volumes/Development/tmp/ace2005_schema.json", "w", encoding="utf-8").write(
    json.dumps(cfg["default_schema"], indent=2, ensure_ascii=False)
)
'

uv run python tools/infer.py \
  --model whr778/gliner2-eb23a \
  --input data/ace2005_eng.test.jsonl \
  --schema-json /Volumes/Development/tmp/ace2005_schema.json \
  --labels-file tools/train/config/labels/unified-full-v4.yaml \
  --threshold 0.3 \
  --include-spans \
  --include-confidence \
  --output out/predictions/ace2005_eng.test.eb23a_control.preds.jsonl
```

### C. Per-record gold-menu decode (`--gold-schema`)

Offers each document only the label types present in its own gold (mapped through `unified-full-v4.yaml`):

```bash
uv run python tools/infer.py \
  --model ./out/ace2005-eb23a/best \
  --input data/ace2005_eng.test.jsonl \
  --gold-schema \
  --labels-file tools/train/config/labels/unified-full-v4.yaml \
  --include-spans \
  --include-confidence \
  --output out/predictions/ace2005_eng.test.gold_schema.preds.jsonl
```

### D. Multi-language (EN + AR + ZH combined)

Concatenate the three splits and run any of the commands above on the combined file, then pass `--by-language` to `score_predictions.py`:

```bash
cat data/ace2005_eng.test.jsonl data/ace2005_ara.test.jsonl data/ace2005_zho.test.jsonl \
  > /Volumes/Development/tmp/ace2005_all.test.jsonl
```

## 3. Score predictions (`tools/train/score_predictions.py`)

```bash
uv run python tools/train/score_predictions.py \
  --predictions out/predictions/ace2005_eng.test.model_schema.preds.jsonl \
  --by-language \
  --out out/predictions/ace2005_eng.test.model_schema.scored.json \
  --card
```

- Automatically selects `gold_mapped` (`--gold auto`) when `--labels-file` was passed to `infer.py`.
- Reports strict and relaxed micro P/R/F1 for `entity`, `relation`, `event_type`, `event_trigger`, `event_argument`, and `event`, plus OneIE criteria (`Trig-I`, `Trig-C`, `Arg-I`, `Arg-C`).
- Automatically reports entity coref mention-pair exact-span P/R/F1 (`all` and `pronoun`) and `candidate_recall` when `gold_coref` (or office `clusters`) is present.
- On `--model-schema` runs, the `off_menu` warning is expected because the full ACE ontology was offered on every document; predictions outside a single document's gold types are scored as false positives against `gold_mapped`.

## 4. Inspect in the viewer

Files written under `out/predictions/` with `--include-spans` and `--labels-file` appear automatically in the **Predictions vs gold** picker with all four layers (**Entities**, **Events**, **Relations**, **Coref**), gold rims, diffs, and per-record strict micro F1 (plus mention-pair F1 for Coref) in `gold_mapped` spellings.

## Troubleshooting: zero predicted or matched entities

1. Check the first line of the predictions JSONL:
   ```bash
   uv run python -c '
   import json
   r = json.loads(open("PATH_TO_PREDS.jsonl", encoding="utf-8").readline())
   print("output keys:", list(r["output"].keys()))
   print("pred entities:", r["output"].get("entities"))
   print("gold entities:", (r.get("gold_mapped") or r.get("gold") or {}).get("entities"))
   print("has gold_mapped:", "gold_mapped" in r)
   '
   ```
2. **`"entities"` missing from `output`:** The checkpoint's `default_schema` had `open_vocab: ["entities"]` and `--model-schema` was run without `--entities` (or `--schema-json`).
3. **`pred entities` populated under `Person`/`Organization`/`Location`/`Facility` while `gold` has `PER`/`ORG`/`LOC`/`FAC`:** Re-run `infer.py` with `--labels-file tools/train/config/labels/unified-full-v4.yaml`.
4. **Viewer shows every gold entity as missed while F1 is non-zero:** Re-run `infer.py` with `--include-spans`.
5. **`pred entities` has empty lists `[]` for every type:** Check `inference_defaults.per_head_thresholds.entity` in the checkpoint's `config.json` (`eb23a` defaults to `0.9`); test with `--threshold 0.3`.
