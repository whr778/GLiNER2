# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_swa | train | 5,271 | 5,271 | 13,922 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_swa | val | 752 | 752 | 1,934 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_swa | test | 1,529 | 1,529 | 4,105 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_swa

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 5,271 | 5,271 | 28 | 46 | 109 | 13,922 | 0 | 0 | 0 | 0 | 0 |
| val | 752 | 752 | 29 | 46 | 73 | 1,934 | 0 | 0 | 0 | 0 | 0 |
| test | 1,529 | 1,529 | 29 | 47 | 75 | 4,105 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| swa | 4,742 | 675 | 1,363 | 6,780 | 89.78% |
| und | 488 | 69 | 146 | 703 | 9.31% |
| epo | 17 | 6 | 6 | 29 | 0.38% |
| eng | 2 | 0 | 7 | 9 | 0.12% |
| eus | 3 | 0 | 3 | 6 | 0.08% |
| cor | 5 | 0 | 1 | 6 | 0.08% |
| que | 3 | 0 | 0 | 3 | 0.04% |
| fin | 2 | 0 | 1 | 3 | 0.04% |
| ind | 1 | 0 | 2 | 3 | 0.04% |
| sun | 1 | 1 | 0 | 2 | 0.03% |
| deu | 1 | 0 | 0 | 1 | 0.01% |
| fil | 1 | 0 | 0 | 1 | 0.01% |
| srp | 1 | 0 | 0 | 1 | 0.01% |
| nld | 1 | 0 | 0 | 1 | 0.01% |
| roh | 1 | 0 | 0 | 1 | 0.01% |
| jbo | 1 | 0 | 0 | 1 | 0.01% |
| pol | 1 | 0 | 0 | 1 | 0.01% |
| jav | 0 | 1 | 0 | 1 | 0.01% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 1,529 |
| test_unique_inputs | 1,529 |
| train_records | 5,271 |
| train_unique_inputs | 5,271 |
| val_records | 752 |
| val_unique_inputs | 752 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0215 | 13,922 / 1,934 | location 41%→39%, person 27%→29%, organization 16%→17%, date 16%→16% |
| entities (per mention) | train / test | 0.0308 | 13,922 / 4,105 | location 41%→43%, date 16%→14%, organization 16%→15%, person 27%→28% |
| entities (per mention) | val / test | 0.0379 | 1,934 / 4,105 | location 39%→43%, organization 17%→15%, date 16%→14%, person 29%→28% |


### Events

No event gold.

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

No event arguments.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

No trigger or argument surfaces.

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

No event instances.

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

No event lists more than one trigger, and no document has two events of one type, so there is no trigger pair.

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

No event arguments.

### Entities

#### train

| label | count | share |
| --- | --- | --- |
| location | 5,639 | 40.50% |
| person | 3,791 | 27.23% |
| date | 2,248 | 16.15% |
| organization | 2,244 | 16.12% |

#### val

| label | count | share |
| --- | --- | --- |
| location | 753 | 38.93% |
| person | 554 | 28.65% |
| organization | 326 | 16.86% |
| date | 301 | 15.56% |

#### test

| label | count | share |
| --- | --- | --- |
| location | 1,754 | 42.73% |
| person | 1,153 | 28.09% |
| organization | 608 | 14.81% |
| date | 590 | 14.37% |

### Relations

None.

### Classifications

None.

### Structures

None.

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | unknown |
| relations | unknown |
| events | unknown |
| classifications | unknown |
| structures | unknown |
