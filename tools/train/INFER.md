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

**5. A blind-test set to a predictions file.** `--output` writes JSONL as it goes, one line per
input record: `{"input", "output": the prediction, "gold": the record's own output}`, in input order,
so a scorer can join prediction to gold line by line. Choose the menu:

```bash
# each document offered only its OWN gold labels -- what eval scores
uv run python tools/infer.py --model whr778/gliner2-eb18-balanced \
    --input data/casie.test.jsonl --gold-schema --output casie_test.preds.jsonl

# ... with gold in the model's spellings, built exactly as eval builds it
uv run python tools/infer.py --model whr778/gliner2-eb18-balanced \
    --input data/docfee.test.jsonl --gold-schema \
    --labels-file tools/train/config/labels/unified-full.yaml --output docfee_test.preds.jsonl

# the checkpoint's FULL shipped schema (see the over-firing caution below)
uv run python tools/infer.py --model whr778/gliner2-eb18-balanced \
    --input data/casie.test.jsonl --model-schema --tasks events --output casie_test.full.preds.jsonl
```

Each line of the predictions file:

| key | contents |
|---|---|
| `input` | the document text |
| `output` | **the model's prediction** |
| `gold` | the input record's own `output` (its gold labels), passed through unchanged |
| `gold_mapped` | only with `--labels-file`: the same gold with its labels transformed exactly as training transforms them, so its spellings match the predictions' (`Company Name` -> `CompanyName`). Score against THIS. |

Note the naming: in a training record `output` IS the gold; here it is the prediction, and the
gold moves to `gold`. A category with no predictions may be absent rather than empty: under
`--gold-schema` a document with no predicted relation has no `relation_extraction` key, so a
scorer should read a missing key as "no predictions".

A record with no gold labels under `--gold-schema` is written with `"output": {}` and not decoded.
`--docs-per-write` (default 64) sets how many documents are decoded per flush. Score a blind test
ONCE: pick settings on the validation split first.

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
| `--gold-schema` | each `.jsonl` record gets the schema of its own gold labels, as eval scores it |
| `--labels-file <unified YAML>` | add `gold_mapped`, and with `--gold-schema` build each schema from the mapped gold (eval's path). Warns if the file's map differs from the checkpoint's `label_map` -- usually the wrong file for that model |
| `--output preds.jsonl` | write `{input, output, gold}` per record as JSONL instead of printing |
| `--merge-coreferent off\|link`, `--coref-threshold` | merge mentions of one event (see below); default: the checkpoint's setting |

Each decode setting resolves in this order:
1. the flag;
2. the checkpoint's `inference_defaults` (eb18: 0.3 / 4096 / 0 / global on);
3. 0.5, the model's own window, overlap 0, global off.

Checkpoints from before 63eb320 store no defaults and no label map, so they use step 3 and labels as typed.

## Coreferent triggers (eb20 and later)

One event is often mentioned several times ("Rebels **attacked** the base ... The **assault**
killed four"). A checkpoint trained with `record_coref_link` (eb20) carries a learned trigger x
trigger link that can merge those mentions into one event. Training picks the merge threshold on
validation at the end (`eval.coref_calibration`, or `tools/train/calibrate_coref_threshold.py --write`
for a checkpoint already trained) and writes it into `best/config.json` ONLY if the merges pass the
gates; the table is in `best/coref_threshold_sweep.json`. `infer.py` with no flag uses that stored
setting: `link` + the picked threshold, or `off` if nothing passed. The flags below override it.

```bash
# by hand, overriding the stored default -- pick on VAL, never on test
uv run tools/infer.py --model out/eb20/best \
    --input data/cc_news_events_sonnet55_v2.val.jsonl --gold-schema \
    --labels-file tools/train/config/labels/unified-full-v2.yaml \
    --merge-coreferent link --coref-threshold 0.7 --output sonnet55_val.t07.jsonl

# then score the blind test ONCE at the picked threshold (same flags, .test.jsonl)
```

- **Off (default):** one event per mention. "attacked" and "assault" come out as two events.
- **`link`:** same-type events whose link score clears `--coref-threshold` become one event, and
  every mention is listed in its `triggers`. The log prints `[infer] coreferent merge link
  (link threshold 0.7, link trained)` -- check it.
- **`link` on a checkpoint without the link raises** (`record_merge_coreferent: link needs
  record_coref_link: true`), rather than silently merging nothing.
- **The merge runs inside each window.** Mentions in different windows of a long document are not
  joined by the link. Global decode keeps merged events (traced 2026-10-06: 45 events either way,
  14 vs 13 multi-trigger).
- **An untrained link over-merges.** At init, threshold 0.5 halved the events (28 -> 14) and fused
  "met" with "told". Do not use `link` until the threshold is picked on val against the gates in
  `tools/events_working_papers/COREFERENT_LINK_SPEC.md` section 6: merge precision >= 0.8, and under 5% of
  distinct events fused.
- **Score arguments with `event_cluster_argument_*`**, not `event_argument_*`. The old key is the
  whole trigger set, so a correct event found through a non-first mention scores as a miss.
  `args` mode (merge on shared arguments) is kept for the record only: it traced NEGATIVE, 0 of 6
  merges right.

## Cautions (measured)

- **Don't send the whole event menu.** On one 743-character CASIE article, the full shipped menu (320 event types) fired **225 types and 773 instances**. The same article with a one-type menu gave 0. Eval only ever offers each document its own gold types, so no reported score covers the full-menu case. Use `--tasks` and `--events` to send the types you mean.
- **Structures need record metadata on boundary models.** Without `mode`/`anchor`, a structure decodes to `{}` with no error. The viewer adds them for you, and `infer.py` does not, so put `"mode": "natural", "anchor": "<first field>"` in your `--schema-json`.
- **The stored threshold is the eval setting, not a sweep result.** If a run's validation sweep chose a different threshold, check that the checkpoint was backfilled (`backfill_model_config.py`).
- **On Apple silicon (MPS), batched results depend on the batch.** The `mps-flash-attn` SDPA patch
  does not apply padding exactly: measured 2026-10-05, 3 of 5 CASIE docs decoded differently alone
  than when batched with longer docs, while CPU and stock SDPA on MPS were identical either way. Until
  that is resolved, produce a blind-test file on CPU or with `--batch-size 1` on MPS.
- **Use `AutoExtractor`, not `GLiNER2`, in your own code.** `GLiNER2` is the span class, and a boundary checkpoint fails to load with `'ExtractorConfig' object has no attribute 'max_width'`. `infer.py` had this defect until 2026-10-05.
