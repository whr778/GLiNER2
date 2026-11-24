# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| klue_ner | train | 16,782 | 16,778 | 39,780 | 0 | 0 | 0 | 0 | 0 |
| klue_ner | val | 2,116 | 2,116 | 5,055 | 0 | 0 | 0 | 0 | 0 |
| klue_ner | test | 2,104 | 2,103 | 4,896 | 0 | 0 | 0 | 0 | 0 |

## klue_ner

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 16,782 | 16,778 | 38 | 71 | 120 | 39,780 | 0 | 0 | 0 | 0 | 0 |
| val | 2,116 | 2,116 | 36 | 71 | 116 | 5,055 | 0 | 0 | 0 | 0 | 0 |
| test | 2,104 | 2,103 | 37 | 71 | 112 | 4,896 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| kor | 16,779 | 2,116 | 2,104 | 20,999 | 99.99% |
| und | 2 | 0 | 0 | 2 | 0.01% |
| eng | 1 | 0 | 0 | 1 | 0.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 2,104 |
| test_unique_inputs | 2,103 |
| train_records | 16,782 |
| train_unique_inputs | 16,778 |
| val_records | 2,116 |
| val_unique_inputs | 2,116 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0130 | 39,780 / 5,055 | DT 16%→17%, QT 23%→22%, PS 28%→27%, LC 13%→13%, TI 4%→4% |
| entities (per mention) | train / test | 0.0178 | 39,780 / 4,896 | LC 13%→14%, OG 17%→16%, QT 23%→22%, TI 4%→4%, PS 28%→28% |
| entities (per mention) | val / test | 0.0170 | 5,055 / 4,896 | DT 17%→16%, LC 13%→14%, OG 16%→16%, PS 27%→28%, TI 4%→4% |


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
| PS | 10,993 | 27.63% |
| QT | 9,150 | 23.00% |
| OG | 6,619 | 16.64% |
| DT | 6,350 | 15.96% |
| LC | 5,087 | 12.79% |
| TI | 1,581 | 3.97% |

#### val

| label | count | share |
| --- | --- | --- |
| PS | 1,380 | 27.30% |
| QT | 1,123 | 22.22% |
| DT | 851 | 16.83% |
| OG | 832 | 16.46% |
| LC | 658 | 13.02% |
| TI | 211 | 4.17% |

#### test

| label | count | share |
| --- | --- | --- |
| PS | 1,368 | 27.94% |
| QT | 1,088 | 22.22% |
| DT | 778 | 15.89% |
| OG | 769 | 15.71% |
| LC | 680 | 13.89% |
| TI | 213 | 4.35% |

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
