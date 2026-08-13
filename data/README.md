# Corpora

`/data/*` is gitignored; a few small corpora are re-included in `.gitignore`.

The two `synthetic_haiku45_5k` corpora are NOT committed -- they live in private
HuggingFace dataset repos and the trainer pulls a split on demand when it is
missing locally (`hf_jsonl` in `tools/train/dataset_registry.yaml`, fetched by
`_fetch_if_missing` in `tools/train/train.py`). Needs `HF_TOKEN` in the
environment.

| corpus | repo | records | committed |
|---|---|--:|---|
| synthetic_haiku45_5k | `whr778/synthetic_haiku45_5k` | 4,997 | no |
| synthetic_haiku45_5k_coerced | `whr778/synthetic_haiku45_5k-coerced` | 5,000 | no |
| synthetic_sonnet5_1k | `whr778/synthetic_sonnet5_1k` | 1,882 | yes (predates this layout) |

## synthetic_haiku45_5k

Generated 2026-08-13 with `claude-haiku-4-5-20251001`, batch
`msgbatch_01QLWC7wkhoAyvuk6v6LppgT`, at commit `f46bbd8`. 4018 train / 496 val /
483 test; 3 of 5,000 replies failed to parse.

Entity type purity (share of annotations under their surface's majority type) is
92.6% overall and 91.1% on surfaces seen five or more times. 12 absent entity
types per document are seeded as `{type: []}` negatives, giving 59,581 positive
type slots against 59,964 negatives, with no negative-only documents.

## synthetic_haiku45_5k_coerced

5,000 synthetic multi-task documents (4020 train / 496 val / 484 test), generated
2026-08-13 by `tools/data/synthetic/generate.py --batch` with
`claude-haiku-4-5-20251001`. Entities, relations, events, classifications and
structures; ~67% of asked-about entity types are negatives (`{type: []}`).

**Entity types in this corpus are unreliable. Do not treat them as clean type
supervision.** Each document was shown only 14 of the 125 entity types, so when a
span's correct type was not among them the model filed it under the nearest one
that was.

Measured over all 5,000 records:

| | |
|---|--:|
| entity annotations | 39,455 |
| distinct surfaces | 18,423 |
| surfaces carrying more than one type | 3,134 (17.0%) |
| annotations on those surfaces | 22,256 (**56.4%**) |
| most types on a single surface | 11 |

Examples: `San Francisco` appears as location, geopolitical entity, address,
region, area, landmark, airport, facility and cardinal. `Massachusetts General
Hospital` as facility, organization, landmark, educational institution and
company. `March 15, 2024` as date (219), time (152) and deadline (16). The
confusions cluster in exactly the families that are hardest to tell apart:
org/facility/company, GPE/region/area/address, money/currency/quantity, date/time.

Negatives are affected in the other direction: a type absent from a document's
sampled 14 can be present in its text, so some `{type: []}` entries are wrong.

Spans, relations, events, classifications and structures were not measured for
this defect; only entity *types* are known bad. Every span is verbatim from its
document (validation drops anything that isn't).

The generator was fixed after this run: the prompt now offers the full entity
ontology while the sampled subset still decides what is kept, so the coercion
pressure is gone and the negatives become types the model weighed and rejected.
Corpora generated after that fix do not carry this caveat.
