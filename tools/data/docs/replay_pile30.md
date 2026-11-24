# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replay_pile30 | train | 9,361 | 9,361 | 124,512 | 0 | 0 | 0 | 0 | 0 |
| replay_pile30 | val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| replay_pile30 | test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## replay_pile30

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 9,361 | 9,361 | 233 | 263 | 3378 | 124,512 | 0 | 0 | 0 | 0 | 0 |
| val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 8,651 | 0 | 0 | 8,651 | 92.42% |
| und | 550 | 0 | 0 | 550 | 5.88% |
| por | 21 | 0 | 0 | 21 | 0.22% |
| spa | 19 | 0 | 0 | 19 | 0.20% |
| rus | 15 | 0 | 0 | 15 | 0.16% |
| deu | 15 | 0 | 0 | 15 | 0.16% |
| jpn | 14 | 0 | 0 | 14 | 0.15% |
| fra | 12 | 0 | 0 | 12 | 0.13% |
| ita | 8 | 0 | 0 | 8 | 0.09% |
| nld | 8 | 0 | 0 | 8 | 0.09% |
| swe | 7 | 0 | 0 | 7 | 0.07% |
| zho | 7 | 0 | 0 | 7 | 0.07% |
| pol | 5 | 0 | 0 | 5 | 0.05% |
| ron | 4 | 0 | 0 | 4 | 0.04% |
| hrv | 3 | 0 | 0 | 3 | 0.03% |
| slv | 3 | 0 | 0 | 3 | 0.03% |
| ind | 2 | 0 | 0 | 2 | 0.02% |
| nor | 2 | 0 | 0 | 2 | 0.02% |
| ell | 2 | 0 | 0 | 2 | 0.02% |
| dan | 2 | 0 | 0 | 2 | 0.02% |
| hun | 2 | 0 | 0 | 2 | 0.02% |
| ukr | 2 | 0 | 0 | 2 | 0.02% |
| fin | 2 | 0 | 0 | 2 | 0.02% |
| tha | 1 | 0 | 0 | 1 | 0.01% |
| vie | 1 | 0 | 0 | 1 | 0.01% |
| kor | 1 | 0 | 0 | 1 | 0.01% |
| ara | 1 | 0 | 0 | 1 | 0.01% |
| tur | 1 | 0 | 0 | 1 | 0.01% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 0 |
| test_unique_inputs | 0 |
| train_records | 9,361 |
| train_unique_inputs | 9,361 |
| val_records | 0 |
| val_unique_inputs | 0 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | n/a | 124,512 / 0 |  |
| entities (per mention) | train / test | n/a | 124,512 / 0 |  |


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
| e_0 | 14,689 | 11.80% |
| e_1 | 13,505 | 10.85% |
| e_2 | 12,812 | 10.29% |
| e_3 | 12,178 | 9.78% |
| e_4 | 11,118 | 8.93% |
| e_5 | 9,854 | 7.91% |
| e_6 | 8,823 | 7.09% |
| e_7 | 7,578 | 6.09% |
| e_8 | 6,472 | 5.20% |
| e_9 | 5,436 | 4.37% |
| e_10 | 4,460 | 3.58% |
| e_11 | 3,786 | 3.04% |
| e_12 | 3,090 | 2.48% |
| e_13 | 2,433 | 1.95% |
| e_14 | 1,949 | 1.57% |
| e_15 | 1,501 | 1.21% |
| e_16 | 1,166 | 0.94% |
| e_17 | 911 | 0.73% |
| e_18 | 702 | 0.56% |
| e_19 | 534 | 0.43% |
| e_20 | 406 | 0.33% |
| e_21 | 308 | 0.25% |
| e_22 | 232 | 0.19% |
| e_23 | 161 | 0.13% |
| e_24 | 123 | 0.10% |
| e_25 | 81 | 0.07% |
| e_26 | 54 | 0.04% |
| e_27 | 44 | 0.04% |
| e_28 | 35 | 0.03% |
| e_29 | 27 | 0.02% |
| e_30 | 18 | 0.01% |
| e_31 | 8 | 0.01% |
| e_32 | 7 | 0.01% |
| e_33 | 2 | 0.00% |
| e_34 | 2 | 0.00% |
| e_35 | 2 | 0.00% |
| e_36 | 2 | 0.00% |
| e_37 | 1 | 0.00% |
| e_38 | 1 | 0.00% |
| e_39 | 1 | 0.00% |

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
