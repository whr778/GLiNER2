# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bionlp11id | train | 1,850 | 1,850 | 5,260 | 0 | 0 | 0 | 0 | 0 |
| bionlp11id | val | 586 | 584 | 1,460 | 0 | 0 | 0 | 0 | 0 |
| bionlp11id | test | 1,389 | 1,389 | 3,557 | 0 | 0 | 0 | 0 | 0 |

## bionlp11id

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1,850 | 1,850 | 33 | 55 | 181 | 5,260 | 0 | 0 | 0 | 0 | 0 |
| val | 586 | 584 | 32 | 54 | 123 | 1,460 | 0 | 0 | 0 | 0 | 0 |
| test | 1,389 | 1,389 | 32 | 53 | 134 | 3,557 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 1,823 | 571 | 1,364 | 3,758 | 98.25% |
| und | 26 | 15 | 25 | 66 | 1.73% |
| ido | 1 | 0 | 0 | 1 | 0.03% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 1,389 |
| test_unique_inputs | 1,389 |
| train_records | 1,850 |
| train_unique_inputs | 1,850 |
| val_records | 586 |
| val_unique_inputs | 584 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0992 | 5,260 / 1,460 | Protein 56%→65%, Organism 33%→26%, Chemical 10%→8%, Regulon-operon 0%→2% |
| entities (per mention) | train / test | 0.0545 | 5,260 / 3,557 | Protein 56%→60%, Chemical 10%→6%, Organism 33%→32%, Regulon-operon 0%→1% |
| entities (per mention) | val / test | 0.0622 | 1,460 / 3,557 | Organism 26%→32%, Protein 65%→60%, Chemical 8%→6%, Regulon-operon 2%→1% |


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
| Protein | 2,952 | 56.12% |
| Organism | 1,755 | 33.37% |
| Chemical | 543 | 10.32% |
| Regulon-operon | 10 | 0.19% |

#### val

| label | count | share |
| --- | --- | --- |
| Protein | 945 | 64.73% |
| Organism | 373 | 25.55% |
| Chemical | 120 | 8.22% |
| Regulon-operon | 22 | 1.51% |

#### test

| label | count | share |
| --- | --- | --- |
| Protein | 2,148 | 60.39% |
| Organism | 1,130 | 31.77% |
| Chemical | 230 | 6.47% |
| Regulon-operon | 49 | 1.38% |

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
