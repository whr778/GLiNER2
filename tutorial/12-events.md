# GLiNER2 Event Extraction Tutorial

Learn how to extract events — ACE-style **triggers** plus their **typed arguments** — from text with GLiNER2.

An *event* is something that happens in the text. Each event has:
- an **event type** (e.g. `Attack`, `Hire`, `Transaction`),
- a **trigger**: the word/phrase that signals it (e.g. *bombed*, *joined*, *sold*),
- zero or more **arguments**: participants in typed **roles** (e.g. `Attacker`, `Target`, `Place`).

## Table of Contents
- [Basic Event Extraction](#basic-event-extraction)
- [Using the Schema Builder](#using-the-schema-builder)
- [Understanding the Output Format](#understanding-the-output-format)
- [Multiple Event Types](#multiple-event-types)
- [Descriptions for Types and Roles](#descriptions-for-types-and-roles)
- [Custom Thresholds](#custom-thresholds)
- [Spans and Confidence](#spans-and-confidence)
- [Batch Processing](#batch-processing)
- [Combining with Other Tasks](#combining-with-other-tasks)
- [Real-World Examples](#real-world-examples)
- [Best Practices](#best-practices)
- [Training Your Own Event Model](#training-your-own-event-model)

## Basic Event Extraction

### Simple Example

Define each event type and the argument roles you want filled, then call `extract_events`:

```python
from gliner2 import GLiNER2

extractor = GLiNER2.from_pretrained("your-model-name")

text = "Rebels bombed the airbase near Aleppo on Tuesday."
results = extractor.extract_events(
    text,
    {"Attack": ["Attacker", "Target", "Place", "Time"]},
)
print(results)
# Output (shape):
# {
#     "event_extraction": {
#         "Attack": [
#             {
#                 "trigger": "bombed",
#                 "arguments": [
#                     {"role": "Attacker", "entity": "Rebels"},
#                     {"role": "Target",   "entity": "the airbase"},
#                     {"role": "Place",    "entity": "Aleppo"},
#                     {"role": "Time",     "entity": "Tuesday"},
#                 ],
#             }
#         ]
#     }
# }
```

The model finds the trigger (`bombed`), recognises it as an `Attack`, and fills the roles you asked for from the surrounding text. Roles it can't find are simply omitted.

## Using the Schema Builder

`extract_events` is a shortcut. The general path builds a schema and calls `extract`, which lets you mix tasks (see [Combining with Other Tasks](#combining-with-other-tasks)):

```python
schema = extractor.create_schema().events(
    {"Attack": ["Attacker", "Target", "Place", "Time"]}
)
results = extractor.extract(text, schema)
```

`event_types` accepts three equivalent forms:

```python
# 1) dict: {event_type: [roles]}
.events({"Attack": ["Attacker", "Target"]})

# 2) rich dict: {event_type: {"roles": [...], ...}}
.events({"Attack": {"roles": ["Attacker", "Target"], "description": "a physical attack"}})

# 3) list of dicts (preserves order; handy when building programmatically)
.events([{"name": "Attack", "roles": ["Attacker", "Target"]}])
```

## Understanding the Output Format

Event results live under the `event_extraction` key. The value is a dict keyed by event type; each type maps to a **list of event mentions** (a type can fire more than once in a text):

```python
{
    "event_extraction": {
        "<event_type>": [
            {
                "trigger": "<trigger surface>",
                "arguments": [
                    {"role": "<role>", "entity": "<argument surface>"},
                    ...
                ],
            },
            ...
        ]
    }
}
```

- An event type with no detected trigger is omitted.
- `arguments` is a list of `{role, entity}` pairs; only filled roles appear.
- Trigger and entity are plain strings by default — see [Spans and Confidence](#spans-and-confidence) to get character offsets and scores instead.

## Multiple Event Types

Pass several types at once; each is detected and scored independently:

```python
text = "Acme acquired Beta Corp for $2B, and CEO Jane Doe resigned the next day."
results = extractor.extract_events(text, {
    "Acquisition": ["Buyer", "Acquired", "Price"],
    "Resignation": ["Person", "Organization", "Time"],
})
# {
#   "event_extraction": {
#     "Acquisition": [{"trigger": "acquired",
#                      "arguments": [{"role": "Buyer", "entity": "Acme"},
#                                    {"role": "Acquired", "entity": "Beta Corp"},
#                                    {"role": "Price", "entity": "$2B"}]}],
#     "Resignation": [{"trigger": "resigned",
#                      "arguments": [{"role": "Person", "entity": "Jane Doe"},
#                                    {"role": "Time", "entity": "the next day"}]}],
#   }
# }
```

## Descriptions for Types and Roles

Short natural-language descriptions help a zero-shot model disambiguate types and roles. Use the rich dict form:

```python
results = extractor.extract_events(text, {
    "Transaction": {
        "roles": ["Buyer", "Seller", "Item", "Price"],
        "description": "a purchase or sale of goods or services",
        "role_descriptions": {
            "Buyer": "the party that pays",
            "Seller": "the party that receives payment",
            "Item": "what is being exchanged",
        },
    }
})
```

Descriptions cost nothing at inference time beyond a slightly longer prompt and often sharpen results on ambiguous or domain-specific schemas.

## Custom Thresholds

Triggers and arguments have separate confidence thresholds. Raise them for precision, lower them for recall. Set defaults for the whole task, or per event type:

```python
# Defaults for every type in this call
schema = extractor.create_schema().events(
    {"Attack": ["Attacker", "Target"]},
    trigger_threshold=0.6,     # how confident before a trigger fires
    argument_threshold=0.4,    # how confident before a role is filled
)

# Per-type override
schema = extractor.create_schema().events({
    "Attack": {"roles": ["Attacker", "Target"],
               "trigger_threshold": 0.7, "argument_threshold": 0.5},
})
```

`extract_events(..., threshold=0.5)` also accepts a single overall `threshold` as a simple knob.

## Spans and Confidence

Ask for character offsets and/or confidence scores; the trigger and each entity then become dicts instead of bare strings:

```python
results = extractor.extract_events(
    text,
    {"Attack": ["Attacker", "Target"]},
    include_spans=True,
    include_confidence=True,
)
# "Attack": [
#   {"trigger": {"text": "bombed", "start": 7, "end": 13, "confidence": 0.97},
#    "arguments": [
#        {"role": "Attacker",
#         "entity": {"text": "Rebels", "start": 0, "end": 6, "confidence": 0.91}},
#    ]}
# ]
```

Spans are character offsets into the original `text`, so you can highlight or post-process exact mentions.

## Batch Processing

Score many documents in one call with `batch_extract_events`:

```python
texts = [
    "Rebels bombed the airbase near Aleppo on Tuesday.",
    "Acme acquired Beta Corp for $2B.",
]
results = extractor.batch_extract_events(
    texts,
    {"Attack": ["Attacker", "Target", "Place"],
     "Acquisition": ["Buyer", "Acquired", "Price"]},
    batch_size=8,
)
# -> list of per-document result dicts, same shape as extract_events
for text, res in zip(texts, results):
    print(res["event_extraction"])
```

## Combining with Other Tasks

Events compose with entities, relations, and classification in a single schema and a single forward pass — the result dict carries one key per task:

```python
schema = (
    extractor.create_schema()
    .entities(["person", "organization", "location"])
    .events({"Attack": ["Attacker", "Target", "Place"]})
)
results = extractor.extract(text, schema)
# results -> {"entities": {...}, "event_extraction": {...}}
```

This is how the training data is laid out too: one record can carry entities, relations, and events together, so the model learns them jointly.

## Real-World Examples

### Security / threat intelligence

```python
text = ("On March 3, the APT group exploited a zero-day in the VPN gateway "
        "and exfiltrated 40GB of customer data.")
extractor.extract_events(text, {
    "CyberAttack": {
        "roles": ["Attacker", "Vulnerability", "Target", "Time"],
        "description": "a malicious intrusion or exploit",
    },
    "DataBreach": {
        "roles": ["Victim", "Data", "Volume"],
        "description": "unauthorized access to or theft of data",
    },
})
```

### Business / financial news

```python
text = "Globex named former Initech CFO Pat Lee as its new chief executive."
extractor.extract_events(text, {
    "PersonnelChange": ["Person", "Organization", "Position"],
})
```

## Best Practices

- **Name roles concretely.** `Attacker`/`Target` beat `Arg1`/`Arg2` — the role name is part of the prompt the model reads.
- **Keep schemas focused.** A handful of relevant types per call works better than dumping every type you can think of.
- **Tune the two thresholds separately.** Trigger detection and argument filling fail differently; precision tuning usually means raising `trigger_threshold`, recall means lowering `argument_threshold`.
- **Add descriptions for domain or ambiguous schemas.** They cost almost nothing and disambiguate look-alike types/roles.
- **Expect per-mention lists.** A type can fire multiple times in one document; always iterate the list under each event type.
- **Use `include_spans=True`** when you need exact offsets for highlighting or downstream linking.

## Training Your Own Event Model

The pretrained models give zero-shot event extraction, but a domain model trained on event corpora (ACE 2005, MAVEN, RAMS, WikiEvents, CASIE, …) is much stronger. GLiNER2 trains entities, relations, and events jointly from one JSONL format:

```json
{"input": "Rebels bombed the airbase near Aleppo on Tuesday.",
 "output": {"events": [
     {"event_type": "Attack", "trigger": "bombed",
      "arguments": [{"role": "Attacker", "entity": "Rebels"},
                    {"role": "Place", "entity": "Aleppo"}]}
 ]}}
```

- **Prepare data:** the converters under `tools/data/` produce this format from the public event corpora — see [TRAINING_DATA.md](../TRAINING_DATA.md).
- **Train:** add the event JSONLs to your config's `event_files` and train — see [TRAINING.md](../TRAINING.md). `compute_metrics` reports `event_type`, `event_trigger`, `event_argument`, and a combined `event` score; the metric definitions are in [METRICS.md](../METRICS.md).
