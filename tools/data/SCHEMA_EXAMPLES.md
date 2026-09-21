# Schema examples: evaluation and inference

Two different objects in this repo are both called "the schema", and confusing them has cost
real time. This is the catalogue of both, with every example transcribed from actual corpora,
actual checkpoints and actual code -- not written from memory.

| | evaluation schema | inference schema |
|---|---|---|
| what it is | the GOLD answer for one record | the QUESTION asked of the model |
| where it lives | `output` in a `data/*.jsonl` record | `Schema` builder, or `default_schema` in a checkpoint's `config.json` |
| who reads it | `compute_metrics`, `derive_schema` | `model.extract(text, schema)` |
| labels are | the answers | the MENU |
| built by | the converters in this directory | `model.create_schema()`, or `Schema.from_dict` |

The bridge between them: `derive_schema` (`gliner2/inference/schema.py`) unions the gold
`output` of many records into one inference schema, and that is how a checkpoint gets its
`default_schema`. So the same five dimensions appear in both, in two different shapes.

**The menu is an input.** A label the model is never offered cannot be predicted, and a label
offered under a different spelling is a different label. See `CLAUDE.md`, TRAINING DATA LABELS.

---

## 1. Evaluation schema -- the gold `output` block

One JSONL record is `{"input": "<text>", "output": {...}}`. Converters write it through
`_split.dumps_record`. All five dimensions are optional and independent; a corpus supplies
whichever it annotates.

### 1.1 entities -- label to list of surfaces

```json
"entities": {
  "GPE": ["New Zealand", "Australia"],
  "Time": ["2017"],
  "Device": ["home security cameras"],
  "Person": ["cybercriminals", "the user"]
}
```
From `data/casie_typed.train.jsonl`. Note the shape is label to surfaces, NOT a list of span
objects, and there are no character offsets -- matching is by surface string.

### 1.2 events -- trigger plus typed arguments

```json
"events": [
  {
    "event_type": "Cyber.Ransom",
    "triggers": ["ransomware campaigns"],
    "arguments": [
      {"role": "Place", "entity": "New Zealand"},
      {"role": "Place", "entity": "Australia"}
    ]
  },
  {
    "event_type": "Cyber.Ransom",
    "triggers": ["demanding a ransom payment"],
    "arguments": [
      {"role": "Attacker", "entity": "cybercriminals"},
      {"role": "Victim", "entity": "the user"}
    ]
  }
]
```
From `data/casie_typed.train.jsonl`. **Two instances share `event_type`** -- that is the
common case, not an edge case (64.2% of gold event instances share their type with another
instance in the same document), and it is why `boundary_head.event_records: true` exists. The
mention path compiles one instance per TYPE and cannot express the second one.

### 1.3 relations -- a list of single-key objects

```json
"relations": [
  {"bind": {"head": "Db", "tail": "Ly49C"}},
  {"bind": {"head": "Db", "tail": "Ly49C"}}
]
```
From `data/bio_ner_relations.train.jsonl`. The relation NAME is the key; `head`/`tail` are
surfaces. Duplicates are legal in the data but are **collapsed at scoring**: verified in
`_gold_relation_set` (`gliner2/training/eval_metrics.py:574`), which accumulates
`(name, head, tail)` into a `set`. The two identical `bind` rows above are ONE gold triple,
so a corpus carrying many exact-duplicate relations has a smaller effective support than its
raw count suggests. A relation whose head or tail is empty or whitespace is dropped entirely.

### 1.4 classifications -- the menu travels WITH the answer

```json
"classifications": [
  {"task": "certainty",
   "labels": ["asserted", "hedged", "speculative", "denied"],
   "true_label": ["asserted"],
   "multi_label": false},
  {"task": "sentiment",
   "labels": ["positive", "negative", "neutral"],
   "true_label": ["neutral"],
   "multi_label": false}
]
```
From `data/cc_news_haiku45.train.jsonl`. This is the one dimension whose gold record carries
its own `labels` menu, because a classification is only meaningful against its option set.
`true_label` is always a LIST even when `multi_label` is false.

### 1.5 json_structures -- a named record of fields

```json
"json_structures": [
  {"casualty_report": {
     "missing": "66",
     "location": "Villa St. Louis in the community of Orleans, Ontario"}}
]
```
From `data/casualty_anchorless.train.jsonl`. Same single-key-object shape as relations: the
structure name is the key. Values are surfaces.

### 1.6 entity_types -- a JOIN, not a sixth dimension

```json
"entity_types": {"New Zealand": ["GPE"], "cybercriminals": ["Person"]}
```
Written by `tools/data/merge_entity_types.py` **inside `output`**, never as an `entities`
block. It maps an argument surface to the entity type(s) that surface carries, so the typed
role-constraint work can ask "is this filler's type allowed for this role". Adding it as
`entities` would silently invent entity supervision the corpus never had.

---

## 2. Inference schema -- the question

### 2.1 The builder (the normal path)

```python
schema = (model.create_schema()
    .entities(["person", "organization", "location", "date"])
    .classification("sentiment", ["positive", "negative", "neutral"])
    .relations(["works_for", "located_in"])
    .events({"Attack": ["Attacker", "Target", "Place", "Time"]})
    .structure("contact")
        .field("email", dtype="string")
        .field("phone", dtype="string")
)
results = model.extract(text, schema)
```

`events()` also takes a richer per-type form:

```python
.events({"Attack": {
    "roles": ["Attacker", "Target", "Place"],
    "description": "a deliberate hostile act",
    "role_descriptions": {"Attacker": "who carried it out"},
    "trigger_threshold": 0.4,
    "argument_threshold": 0.3,
    "exclusive_roles": ["Target"]}})
```

`exclusive_roles` is DECODE-TIME only -- a span bound to one instance cannot also bind to
another, resolved by Hungarian assignment across instances of that type. It needs no
retraining to turn on or off, but is only meaningful on a checkpoint trained with
`boundary_head.event_records: true`.

### 2.2 `default_schema` in a checkpoint's `config.json`

Every checkpoint ships the schema it was trained on, so a model is self-describing. Read it:

```bash
uv run python -c "
import json; d = json.load(open('<checkpoint>/config.json'))
print(sorted(d['default_schema']))
print(d['default_schema'].get('open_vocab'))
"
```

Real output from `eb16-eventrecords-tr` -- keys `["classifications", "events",
"open_vocab", "relations"]`, 78 event types, 4 classification tasks, and NO `entities` key:

```json
{"events": {
   "Accident":  ["Date", "Location", "Result", "Subject"],
   "Attack":    ["Attacker", "Date", "Death Count", "Injured Count",
                 "Location", "attack_target"],
   "Collapse":  ["Collapsed Structure", "Date", "Death Count", "Injured Count"]},
 "classifications": [
   {"task": "biotech_event",
    "labels": ["alliance & partnership", "article publication",
               "clinical trial sponsorship", "closing", "company description"]}],
 "open_vocab": ["entities"]}
```

(events and classifications truncated here for reading; the file carries all 78 and all 4.)
Note `attack_target` beside `Attacker` and `Location` -- a casing/spelling inconsistency that
survived label unification, and exactly the kind of thing `default_schema` is useful for
auditing, since it is the menu the model will actually be offered.

**`open_vocab` is the trap.** Past `_OPEN_VOCAB_LIMIT = 1000` distinct labels a dimension is
open-vocabulary: `derive_schema` records the dimension name under `open_vocab` and OMITS its
label list entirely (deliberately -- a truncated list would read as a complete ontology).
Every eb16 event checkpoint carries `open_vocab: ["entities"]` and NO entity list. The
practical consequence, paid for once already: full-menu eval cannot widen entities, so
`eval_fullmenu_entity_*` are GOLD-menu numbers under a full-menu key name.

### 2.3 `Schema.from_dict` -- the same dict, rehydrated

```python
from gliner2.inference.schema import Schema
schema = Schema.from_dict(json.load(open("<checkpoint>/config.json"))["default_schema"])
```
`from_dict` IGNORES `open_vocab`, so a round-trip through it silently drops the
open-vocabulary dimensions. Offer those labels explicitly if you need them scored.

---

## 3. What eval does to the schema: full-menu widening

`compute_metrics(..., full_menu=..., menu_negatives=20)` calls `_widen_with_absent`
(`gliner2/training/eval_metrics.py`) to add labels the document does NOT have:

```bash
uv run python tools/train/eval.py --config <cfg> --split test \
    --threshold 0.3 --full-menu --menu-negatives 20
```

- Gold menu: the model is offered only labels the document actually has, so **the menu cannot
  express a wrong answer**. Precision is inflated by construction -- the incumbent's
  `event_type` precision is 1.0000 under gold and 0.2228 under the full menu.
- Full menu: gold plus up to `menu_negatives` absent labels, sampled deterministically from a
  seed derived from the RECORD INDEX, so two checkpoints see the identical menu for the
  identical document. That determinism is what makes the comparison legal.
- **Only entities, events and relations are widened.** Classification and structure are not in
  `_widen_with_absent` at all -- proven by gold == full for both. Their "full-menu" rows are
  gold-menu numbers. And entities are widened only if the menu carries an entity list, which
  an `open_vocab` checkpoint does not (2.2).

Any precision-targeting intervention -- negatives, the typed margin -- is structurally
invisible under the gold menu. Score it full-menu or do not score it.

---

## 4. Gotchas that have actually bitten

- **`derive_schema` drops trigger-only event types** (no roles) and **single-label
  classification tasks**, matching eval. A type present in the data can be absent from
  `default_schema` for that reason alone.
- **Scan every nesting depth.** Reading `rec["events"]` instead of `rec["output"]["events"]`
  hid 147,456 Chinese labels. Check the label MENU separately from the answers: a corpus can
  present a Chinese menu with English answers.
- **An empty inline `labels:` block in a training config silently OVERRIDES `labels_file`.**
  Delete it, or the wiring is a no-op that looks done.
- **Never compare across a changed test set.** If the eval config's corpus list moved, the
  denominator moved; `compare_runs.py` refuses, and so should you.
- **`eval_provenance`** records the operating point (threshold, menu, split, record count).
  Files written before 2026-09-18 have none, and a `<split>_metrics.json` without it records
  314 numbers and nothing about how they were produced.

## Related

- `tools/data/TRAINING_DATA.md` -- the corpora themselves
- `tools/data/README.md` -- the converters
- `METRICS.md` -- what each metric key means
- `tools/events_working_papers/EXPERIMENT_CATALOG.md` -- every run
