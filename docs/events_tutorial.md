# Events with GLiNER2: a tutorial

Events are the hardest of GLiNER2's tasks and the one with the most ways to be quietly wrong.
This walks through extracting them, reading the numbers honestly, and training a model that
does events well.

Everything here is runnable. Numbers quoted are measured on this project's own models and are
cited to the working paper that recorded them.

---

## 1. The shape of an event

An event is a **typed instance** with a trigger and role-filled arguments:

```python
from gliner2 import AutoExtractor

model = AutoExtractor.from_pretrained("whr778/gliner2-eb16-eventrecords-tr")

text = "On 16 March, Acme Corp acquired Beta Inc for $2.1 billion."

schema = {
    "Acquisition": ["acquirer", "target", "price", "date"],
}
model.extract_events(text, schema, threshold=0.3)
```

```python
{'event_extraction': {'Acquisition': [
    {'triggers': ['acquired'],
     'arguments': [{'role': 'acquirer', 'entity': 'Acme Corp'},
                   {'role': 'target',   'entity': 'Beta Inc'},
                   {'role': 'price',    'entity': '$2.1 billion'},
                   {'role': 'date',     'entity': '16 March'}]}]}}
```

The schema is `{event_type: [role, ...]}` — an **input**, not a fixed vocabulary. The same
model answers a different schema on the same text. That is the whole point of the
architecture, and it is also the source of the first trap.

---

## 2. THE FIRST TRAP: the menu you offer decides the answer

Offer a type that is not there and a model that was never trained to reject one will
cheerfully produce it:

```python
model.extract_events(text, {"Earthquake": ["magnitude", "location"]}, threshold=0.3)
```

Measured on this project's incumbent, given a schema containing **only** event types the
document does not have, it fired on **63 of 100 documents**
([EVENT_ARGUMENT_DIAGNOSIS](../tools/events_working_papers/EVENT_ARGUMENT_DIAGNOSIS.md) §4i).

**Always offer a realistic menu.** A single-type schema asks "is this here?" of a model with
no reason to say no. Offer the types you would offer in production — including ones you expect
to be absent — and threshold accordingly.

```python
# better: the taxonomy you would actually deploy with
model.extract_events(text, model.config.default_schema["events"], threshold=0.3)
```

Every trained checkpoint carries the taxonomy it was trained on in
`config.json → default_schema`, so you rarely need to invent one.

---

## 3. Multiple events of one type: the thing to check first

Documents routinely contain several events of the **same type**. On this project's blind test,
**69.7% of gold event instances share a type with another instance in the same document**
(`tools/data/event_multiplicity.py`).

```python
text = ("The navy tested a new missile on Tuesday. "
        "A second test of the same system followed on Friday.")
out = model.extract_events(text, {"Experiment": ["subject", "date"]}, threshold=0.3)
len(out["event_extraction"]["Experiment"])       # 2 if the model can address two slots
```

A model whose events go through the **mention path** returns **one** instance here, carrying
both triggers with the arguments unioned — measured, 11 of 12 documents
(EVENT_ARGUMENT_DIAGNOSIS §4e). One trained with `event_records: true` routes events through
the **record head**, which allocates a slot per instance and can return two.

If you are choosing a checkpoint for event work, this is the single most useful thing to test.

---

## 4. Reading event metrics without fooling yourself

Four numbers, and they mean very different things.

| metric | requires | what it tells you |
|---|---|---|
| `event_type` | the type is present | whether the type was proposed |
| `event_trigger` | type + trigger span | whether the mention was found |
| `event_argument` **relaxed** | type + role + entity | whether the argument was *found* |
| `event_argument` **strict** | + bound to the right trigger | whether it was found **and attached** |

**Strict and relaxed answer different questions, and they move in opposite directions under
threshold.** On this project's incumbent, sweeping the threshold moved relaxed argument recall
a great deal (never-proposed 48.8% → 29.4%) and strict F1 not at all, because precision paid
for it (§4d). Quote both, always.

**`event_type` F1 is not what it looks like.** If you build the menu from a document's own
gold — which the standard eval does — precision is pinned at 1.0000 *by construction* and
`F1 = 2R/(1+R)` exactly, verified on 12 of 12 readings (§4h). Against a real 8-type menu the
same model scores precision **0.5521**. Report `event_type` as **recall**, and say what menu
you used.

```bash
# the honest pass: score against the model's own taxonomy, not the answer key
uv run python tools/train/eval.py --config <cfg> --checkpoint <ckpt> --split test --full-menu
```

That emits `eval_fullmenu_*` keys beside the originals; the originals keep their historical
meaning.

---

## 5. Training data format

```json
{"input": "The navy tested a new missile on Tuesday.",
 "output": {"events": [
   {"event_type": "Experiment",
    "triggers": ["tested"],
    "arguments": [{"role": "subject", "entity": "the navy"},
                  {"role": "date",    "entity": "Tuesday"}]}]}}
```

Notes that save time:

- **`triggers` is a list and must be non-empty.** The training path skips an event with no
  triggers, so an "empty" event contributes nothing rather than acting as a negative.
- **Surfaces must appear in the text**, verbatim. Alignment is by string match.
- **One entry per instance.** Two events of the same type are two entries, not one entry with
  more arguments — that distinction is exactly what §3 is about.
- **Label spelling is load-bearing.** The schema is an input, so `Attack` and `attack` are two
  different types to the model. Unify in the config with `labels_file`, not by rewriting
  corpora (see the project `CLAUDE.md`).

---

## 6. Training a model that does events properly

```yaml
model:
  encoder: jhu-clsp/mmBERT-base
  architecture: boundary
  boundary_head:
    enable_records: true
    event_records: true        # events through the RECORD head -- see section 3
    max_gold_per_query: 256    # see the warning below
    training_candidate_budget: 384

training:
  on_capacity_exceeded: skip_sample
  # label negatives: the model must be shown types it should REJECT
  negative_pools: tools/train/config/labels/negative_pools.json
  negative_labels_per_dim: {entities: 2, events: 1, relations: 1}
```

**`event_records: true` is the flag most people miss.** Without it, events decode through the
mention path, one instance per type per document, and no amount of training or threshold
tuning lifts that ceiling — it is the decoder's addressing scheme, not a capability.

**`max_gold_per_query: 32` (the default) silently destroys supervision.** With
`on_capacity_exceeded: skip_sample`, one crowded query clears gold for *every* query in that
document, and an empty mask is not "no supervision" — it is positive supervision to **abstain**
(`abstention_loss` targets `~mention_mask.any(-1)`). It fired 3,774 times in a 16-hour run
before being caught. Size it from your own data:

```bash
uv run python tools/train/size_gold_capacity.py --config <your-config> --apply
```

**Label negatives** teach the model to say no. Build the pools first:

```bash
uv run python tools/data/build_negative_pools.py --config <your-config>
```

Pools are **per corpus** on purpose. Corpora annotate different things — one in this project
carries 8 event types and zero entity gold — so a global pool would offer `Organization` to a
document that contains organisations and simply does not label them, supervising a true
positive as absent.

---

## 7. A checklist before you trust an event number

1. What **menu** was it scored against — the document's gold, or a real taxonomy?
2. Is it **strict or relaxed**? If the source does not say, assume it does not know.
3. Does the model emit **two instances of one type** when the document has two?
4. Was `event_records: true` set at training time?
5. Was `max_gold_per_query` sized from the data, or left at 32?
6. Were **label negatives** used, or was every training menu built from the answer key?

---

## See also

- `tools/data/SCHEMA_EXAMPLES.md` -- every schema shape, evaluation and inference, transcribed
  from real corpora and real checkpoints. Start here if you are unsure which object you hold.

- [`EVENT_ARGUMENT_DIAGNOSIS.md`](../tools/events_working_papers/EVENT_ARGUMENT_DIAGNOSIS.md) —
  why event arguments fail, measured
- [`LABEL_NEGATIVES_PLAN.md`](../tools/events_working_papers/LABEL_NEGATIVES_PLAN.md) — the
  negatives work and its evidence
- [`METRICS.md`](../METRICS.md) — metric definitions, including menu identity
