# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| finer_ord | train | 1,305 | 1,305 | 2,690 | 0 | 0 | 0 | 0 | 0 |
| finer_ord | val | 178 | 178 | 420 | 0 | 0 | 0 | 0 | 0 |
| finer_ord | test | 169 | 169 | 377 | 0 | 0 | 0 | 0 | 0 |

## finer_ord

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1,305 | 1,305 | 29 | 47 | 139 | 2,690 | 0 | 0 | 0 | 0 | 0 |
| val | 178 | 178 | 29 | 53 | 78 | 420 | 0 | 0 | 0 | 0 | 0 |
| test | 169 | 169 | 28 | 48 | 90 | 377 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 1,252 | 174 | 164 | 1,590 | 96.25% |
| und | 53 | 4 | 4 | 61 | 3.69% |
| ceb | 0 | 0 | 1 | 1 | 0.06% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 169 |
| test_unique_inputs | 169 |
| train_records | 1,305 |
| train_unique_inputs | 1,305 |
| val_records | 178 |
| val_unique_inputs | 178 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0653 | 2,690 / 420 | ORG 54%→48%, PER 21%→24%, LOC 25%→28% |
| entities (per mention) | train / test | 0.0850 | 2,690 / 377 | ORG 54%→46%, LOC 25%→29%, PER 21%→25% |
| entities (per mention) | val / test | 0.0197 | 420 / 377 | ORG 48%→46%, LOC 28%→29%, PER 24%→25% |


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
| ORG | 1,463 | 54.39% |
| LOC | 663 | 24.65% |
| PER | 564 | 20.97% |

#### val

| label | count | share |
| --- | --- | --- |
| ORG | 201 | 47.86% |
| LOC | 117 | 27.86% |
| PER | 102 | 24.29% |

#### test

| label | count | share |
| --- | --- | --- |
| ORG | 173 | 45.89% |
| LOC | 111 | 29.44% |
| PER | 93 | 24.67% |

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
