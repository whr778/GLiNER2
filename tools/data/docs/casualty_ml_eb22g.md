# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| casualty_ml_eb22g | train | 53,472 | 53,472 | 0 | 0 | 0 | 72,726 | 0 | 0 |
| casualty_ml_eb22g | val | 4,200 | 4,200 | 0 | 0 | 0 | 5,500 | 0 | 0 |
| casualty_ml_eb22g | test | 4,200 | 4,200 | 0 | 0 | 0 | 5,580 | 0 | 0 |

## casualty_ml_eb22g

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 53,472 | 53,472 | 385 | 1460 | 5994 | 0 | 0 | 0 | 72,726 | 0 | 0 |
| val | 4,200 | 4,200 | 378 | 1488 | 5941 | 0 | 0 | 0 | 5,500 | 0 | 0 |
| test | 4,200 | 4,200 | 380 | 1414 | 5940 | 0 | 0 | 0 | 5,580 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 17,830 | 1,401 | 1,400 | 20,631 | 33.34% |
| tur | 17,824 | 1,400 | 1,400 | 20,624 | 33.33% |
| zho | 17,806 | 1,397 | 1,399 | 20,602 | 33.30% |
| und | 10 | 1 | 1 | 12 | 0.02% |
| jpn | 1 | 0 | 0 | 1 | 0.00% |
| rus | 1 | 0 | 0 | 1 | 0.00% |
| deu | 0 | 1 | 0 | 1 | 0.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 4,200 |
| test_unique_inputs | 4,200 |
| train_records | 53,472 |
| train_unique_inputs | 53,472 |
| val_records | 4,200 |
| val_unique_inputs | 4,200 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

No label marginal has gold on both sides of a split pair.


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

None.

### Relations

None.

### Classifications

None.

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| casualty_report | location | 66,270 | 91.12% |
| casualty_report | dead | 51,713 | 71.11% |
| casualty_report | injured | 24,396 | 33.55% |
| casualty_report | missing | 2,203 | 3.03% |
| casualty_report | record_metadata | natural/dead ×39,716, natural/injured ×8,198, natural/location ×5,265, natural/missing ×293 |  |

**val.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| casualty_report | location | 4,986 | 90.65% |
| casualty_report | dead | 3,939 | 71.62% |
| casualty_report | injured | 1,920 | 34.91% |
| casualty_report | missing | 162 | 2.95% |
| casualty_report | record_metadata | natural/dead ×3,125, natural/injured ×660, natural/location ×394, natural/missing ×21 |  |

**test.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| casualty_report | location | 5,038 | 90.29% |
| casualty_report | dead | 4,059 | 72.74% |
| casualty_report | injured | 1,943 | 34.82% |
| casualty_report | missing | 138 | 2.47% |
| casualty_report | record_metadata | natural/dead ×3,177, natural/injured ×635, natural/location ×371, natural/missing ×17 |  |

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | mixed (adjudicated news) |
| relations | mixed (adjudicated news) |
| events | mixed (adjudicated news) |
| classifications | mixed (adjudicated news) |
| structures | mixed (adjudicated news) |
