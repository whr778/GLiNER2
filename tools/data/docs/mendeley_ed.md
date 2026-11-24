# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mendeley_ed | train | 1,420 | 1,420 | 0 | 0 | 0 | 0 | 4,374 | 0 |
| mendeley_ed | val | 156 | 156 | 0 | 0 | 0 | 0 | 495 | 0 |
| mendeley_ed | test | 156 | 156 | 0 | 0 | 0 | 0 | 410 | 0 |

## mendeley_ed

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1,420 | 1,420 | 37 | 59 | 141 | 0 | 0 | 0 | 0 | 4,374 | 0 |
| val | 156 | 156 | 38 | 58 | 116 | 0 | 0 | 0 | 0 | 495 | 0 |
| test | 156 | 156 | 36 | 56 | 104 | 0 | 0 | 0 | 0 | 410 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 1,411 | 155 | 156 | 1,722 | 99.42% |
| und | 9 | 1 | 0 | 10 | 0.58% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 156 |
| test_unique_inputs | 156 |
| train_records | 1,420 |
| train_unique_inputs | 1,420 |
| val_records | 156 |
| val_unique_inputs | 156 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| event types (per trigger mention) | train / val | 0.0000 | 4,408 / 496 | Event 100%→100% |
| event types (per trigger mention) | train / test | 0.0000 | 4,408 / 414 | Event 100%→100% |
| event types (per trigger mention) | val / test | 0.0000 | 496 / 414 | Event 100%→100% |


### Events

**train.** 1,420 instances in 1,420 records, 0 triggerless. Trigger gold 4,374. Argument gold 0. Arguments per instance, mean 0.00. Trigger length in whitespace tokens, mean 1.00, single-token share 100.0%.
Trigger-only: this split cannot teach or score event arguments.

#### train event types

| label | count | share |
| --- | --- | --- |
| Event | 4,408 | 100.00% |

**val.** 156 instances in 156 records, 0 triggerless. Trigger gold 495. Argument gold 0. Arguments per instance, mean 0.00. Trigger length in whitespace tokens, mean 1.00, single-token share 100.0%.
Trigger-only: this split cannot teach or score event arguments.

#### val event types

| label | count | share |
| --- | --- | --- |
| Event | 496 | 100.00% |

**test.** 156 instances in 156 records, 0 triggerless. Trigger gold 410. Argument gold 0. Arguments per instance, mean 0.00. Trigger length in whitespace tokens, mean 1.00, single-token share 100.0%.
Trigger-only: this split cannot teach or score event arguments.

#### test event types

| label | count | share |
| --- | --- | --- |
| Event | 414 | 100.00% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

No event arguments.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.02 | 2.30% | 0.00 | n/a | 0.50% | 0 | 0 | 0 | 4,408 |
| val | 1.01 | 1.42% | 0.00 | n/a | 0.60% | 0 | 0 | 0 | 496 |
| test | 1.04 | 3.64% | 0.00 | n/a | 0.48% | 0 | 0 | 0 | 414 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1,420 | 80.21% | 0 | n/a | n/a | n/a | n/a | 0.00% |
| val | 156 | 81.41% | 0 | n/a | n/a | n/a | n/a | 0.00% |
| test | 156 | 73.08% | 0 | n/a | n/a | n/a | n/a | 0.00% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 1,139 | 6,647 | 0 |
| val | 127 | 791 | 0 |
| test | 114 | 508 | 0 |

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

No event arguments.

### Entities

None.

### Relations

None.

### Classifications

None.

### Structures

None.

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | human |
| relations | human |
| events | human |
| classifications | human |
| structures | human |
