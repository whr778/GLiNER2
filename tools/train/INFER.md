# Inference with `tools/infer.py`

`tools/infer.py` runs a checkpoint over text, the same way the viewer does:

- **Schema:** `--model-schema` starts from the schema the checkpoint ships (`config.json` `default_schema`).
- **Decode settings:** any setting you don't pass comes from the checkpoint's `inference_defaults`. These are the settings its eval was measured at.
- **Labels:** your labels go through the checkpoint's `label_map`, so `person` reaches eb18 as the `Person` it trained on.

Results are printed as JSON on stdout. Every run first prints two `[infer]` lines: the label rewrites, and the exact schema sizes and decode settings used.

## Examples

All of these were run on 2026-10-05.

**1. The model's own schema, narrowed.** Entities are open-vocabulary (the schema ships an `open_vocab` marker), so name them yourself:

```bash
uv run python tools/infer.py --model whr778/gliner2-eb18-balanced \
    --input document.txt --model-schema --tasks relations \
    --entities person,organization,location --include-confidence
```
```
[infer] label_map rewrites: {"entities": {"person": "Person", "organization": "Organization", "location": "Location"}}
[infer] schema relations=905, entities=3 | decode {'threshold': 0.3, 'chunk_size': 4096, 'chunk_overlap': 0, 'global_decode': True} ...
```

**2. Your own event menu.** Name the types and roles you want:

```bash
uv run python tools/infer.py --model whr778/gliner2-eb18-balanced --input document.txt \
    --events '{"Cyber.Phishing": ["Place", "Victim", "Time", "Attacker"]}'
```

**3. A batch of documents.** A `.jsonl` file with one `{"input": ...}` per line; a corpus split works as-is. Any explicit flag overrides the checkpoint:

```bash
uv run python tools/infer.py --model whr778/gliner2-eb18-balanced --input docs.jsonl \
    --events '{"Cyber.Phishing": ["Victim", "Place"], "Cyber.Databreach": ["Victim", "Compromised-Data"]}' \
    --threshold 0.5 --include-confidence
```

**4. A full schema file.** It replaces every other schema option:

```bash
uv run python tools/infer.py --model whr778/gliner2-eb18-balanced --input document.txt --schema-json schema.json
```

`--input` also accepts a literal string or a `.txt` file (one document).

## Options

| Option | Meaning |
|---|---|
| `--model-schema` | start from the checkpoint's `default_schema` |
| `--tasks events,relations` | with `--model-schema`, keep only these task types: entities, relations, events, classifications, structures |
| `--entities`, `--events` | add to, or replace, that task in the schema |
| `--schema-json` | a full schema file; overrides all of the above |
| `--no-label-map` | send labels exactly as typed |
| `--threshold`, `--chunk-size`, `--chunk-overlap`, `--global-decode` / `--no-global-decode` | override the checkpoint |
| `--include-confidence`, `--include-spans` | add scores and character offsets |

Each decode setting resolves in this order:
1. the flag;
2. the checkpoint's `inference_defaults` (eb18: 0.3 / 4096 / 0 / global on);
3. 0.5, the model's own window, overlap 0, global off.

Checkpoints from before 63eb320 store no defaults and no label map, so they use step 3 and labels as typed.

## Cautions (measured)

- **Don't send the whole event menu.** On one 743-character CASIE article, the full shipped menu (320 event types) fired **225 types and 773 instances**. The same article with a one-type menu gave 0. Eval only ever offers each document its own gold types, so no reported score covers the full-menu case. Use `--tasks` and `--events` to send the types you mean.
- **Structures need record metadata on boundary models.** Without `mode`/`anchor`, a structure decodes to `{}` with no error. The viewer adds them for you, and `infer.py` does not, so put `"mode": "natural", "anchor": "<first field>"` in your `--schema-json`.
- **The stored threshold is the eval setting, not a sweep result.** If a run's validation sweep chose a different threshold, check that the checkpoint was backfilled (`backfill_model_config.py`).
- **Use `AutoExtractor`, not `GLiNER2`, in your own code.** `GLiNER2` is the span class, and a boundary checkpoint fails to load with `'ExtractorConfig' object has no attribute 'max_width'`. `infer.py` had this defect until 2026-10-05.
