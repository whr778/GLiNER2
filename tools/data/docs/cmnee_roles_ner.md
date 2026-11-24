# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cmnee_roles_ner | train | 9,281 | 9,281 | 51,958 | 0 | 0 | 0 | 0 | 0 |
| cmnee_roles_ner | val | 1,606 | 1,606 | 8,511 | 0 | 0 | 0 | 0 | 0 |
| cmnee_roles_ner | test | 2,724 | 2,724 | 16,394 | 0 | 0 | 0 | 0 | 0 |

## cmnee_roles_ner

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 9,281 | 9,281 | 302 | 395 | 487 | 51,958 | 0 | 0 | 0 | 0 | 0 |
| val | 1,606 | 1,606 | 300 | 391 | 490 | 8,511 | 0 | 0 | 0 | 0 | 0 |
| test | 2,724 | 2,724 | 289 | 390 | 473 | 16,394 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| zho | 9,281 | 1,606 | 2,723 | 13,610 | 99.99% |
| und | 0 | 0 | 1 | 1 | 0.01% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 2,724 |
| test_unique_inputs | 2,724 |
| train_records | 9,281 |
| train_unique_inputs | 9,281 |
| val_records | 1,606 |
| val_unique_inputs | 1,606 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.1048 | 51,958 / 8,511 | EventObject 11%→6%, EventEquipment 17%→21%, EventMaterials 11%→7%, EventLocation 9%→12%, EventMilitaryforce 5%→8% |
| entities (per mention) | train / test | 0.1860 WARN | 51,958 / 16,394 | EventObject 11%→4%, EventMaterials 11%→4%, EventMilitaryforce 5%→10%, EventDate 16%→20%, EventEquipment 17%→22% |
| entities (per mention) | val / test | 0.0832 | 8,511 / 16,394 | EventDate 16%→20%, EventMaterials 7%→4%, EventSubject 28%→26%, EventObject 6%→4%, EventMilitaryforce 8%→10% |


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
| EventSubject | 14,728 | 28.35% |
| EventEquipment | 8,938 | 17.20% |
| EventDate | 8,065 | 15.52% |
| EventObject | 5,618 | 10.81% |
| EventMaterials | 5,552 | 10.69% |
| EventLocation | 4,854 | 9.34% |
| EventMilitaryforce | 2,389 | 4.60% |
| EventResult | 593 | 1.14% |
| EventContent | 559 | 1.08% |
| EventQuantity | 378 | 0.73% |
| EventArea | 284 | 0.55% |

#### val

| label | count | share |
| --- | --- | --- |
| EventSubject | 2,403 | 28.23% |
| EventEquipment | 1,806 | 21.22% |
| EventDate | 1,347 | 15.83% |
| EventLocation | 1,062 | 12.48% |
| EventMilitaryforce | 649 | 7.63% |
| EventMaterials | 578 | 6.79% |
| EventObject | 519 | 6.10% |
| EventContent | 66 | 0.78% |
| EventResult | 37 | 0.43% |
| EventArea | 30 | 0.35% |
| EventQuantity | 14 | 0.16% |

#### test

| label | count | share |
| --- | --- | --- |
| EventSubject | 4,189 | 25.55% |
| EventEquipment | 3,582 | 21.85% |
| EventDate | 3,343 | 20.39% |
| EventLocation | 2,189 | 13.35% |
| EventMilitaryforce | 1,586 | 9.67% |
| EventMaterials | 605 | 3.69% |
| EventObject | 604 | 3.68% |
| EventContent | 111 | 0.68% |
| EventResult | 94 | 0.57% |
| EventArea | 54 | 0.33% |
| EventQuantity | 37 | 0.23% |

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
