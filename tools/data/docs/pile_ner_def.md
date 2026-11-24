# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pile_ner_def | train | 37,992 | 37,992 | 509,901 | 0 | 0 | 0 | 0 | 0 |
| pile_ner_def | val | 4,739 | 4,739 | 63,623 | 0 | 0 | 0 | 0 | 0 |
| pile_ner_def | test | 4,715 | 4,715 | 63,163 | 0 | 0 | 0 | 0 | 0 |
| pile_ner_def | all | 47,503 | 47,452 | 637,102 | 0 | 0 | 0 | 0 | 0 |

## pile_ner_def

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 37,992 | 37,992 | 234 | 264 | 3378 | 509,901 | 0 | 0 | 0 | 0 | 0 |
| val | 4,739 | 4,739 | 234 | 263 | 636 | 63,623 | 0 | 0 | 0 | 0 | 0 |
| test | 4,715 | 4,715 | 236 | 265 | 1525 | 63,163 | 0 | 0 | 0 | 0 | 0 |
| all | 47,503 | 47,452 | 234 | 264 | 3378 | 637,102 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | all | records | share |
| --- | --- | --- | --- | --- | --- | --- |
| eng | 35,161 | 4,420 | 4,371 | 43,995 | 87,947 | 92.63% |
| und | 2,200 | 249 | 259 | 2,721 | 5,429 | 5.72% |
| spa | 88 | 11 | 11 | 110 | 220 | 0.23% |
| por | 85 | 11 | 11 | 107 | 214 | 0.23% |
| rus | 63 | 7 | 9 | 79 | 158 | 0.17% |
| jpn | 55 | 7 | 10 | 72 | 144 | 0.15% |
| deu | 55 | 4 | 9 | 68 | 136 | 0.14% |
| fra | 45 | 10 | 6 | 61 | 122 | 0.13% |
| ita | 32 | 2 | 4 | 38 | 76 | 0.08% |
| nld | 28 | 3 | 2 | 33 | 66 | 0.07% |
| fin | 15 | 1 | 5 | 21 | 42 | 0.04% |
| zho | 19 | 0 | 1 | 20 | 40 | 0.04% |
| ron | 16 | 1 | 2 | 19 | 38 | 0.04% |
| swe | 14 | 1 | 2 | 17 | 34 | 0.04% |
| dan | 13 | 1 | 3 | 17 | 34 | 0.04% |
| nor | 12 | 3 | 1 | 16 | 32 | 0.03% |
| pol | 13 | 1 | 2 | 16 | 32 | 0.03% |
| hrv | 8 | 0 | 0 | 8 | 16 | 0.02% |
| ind | 6 | 1 | 1 | 8 | 16 | 0.02% |
| tur | 7 | 1 | 0 | 8 | 16 | 0.02% |
| hun | 5 | 1 | 0 | 6 | 12 | 0.01% |
| srp | 4 | 0 | 2 | 6 | 12 | 0.01% |
| kor | 6 | 0 | 0 | 6 | 12 | 0.01% |
| ces | 4 | 1 | 0 | 5 | 10 | 0.01% |
| ell | 5 | 0 | 0 | 5 | 10 | 0.01% |
| ukr | 4 | 1 | 0 | 5 | 10 | 0.01% |
| vie | 3 | 0 | 1 | 4 | 8 | 0.01% |
| est | 3 | 0 | 1 | 4 | 8 | 0.01% |
| heb | 1 | 1 | 1 | 3 | 6 | 0.01% |
| ara | 2 | 1 | 0 | 3 | 6 | 0.01% |
| slv | 3 | 0 | 0 | 3 | 6 | 0.01% |
| lit | 3 | 0 | 0 | 3 | 6 | 0.01% |
| cym | 2 | 0 | 0 | 2 | 4 | 0.00% |
| slk | 2 | 0 | 0 | 2 | 4 | 0.00% |
| ceb | 2 | 0 | 0 | 2 | 4 | 0.00% |
| tha | 2 | 0 | 0 | 2 | 4 | 0.00% |
| cat | 0 | 0 | 1 | 2 | 3 | 0.00% |
| ben | 1 | 0 | 0 | 1 | 2 | 0.00% |
| fas | 1 | 0 | 0 | 1 | 2 | 0.00% |
| hye | 1 | 0 | 0 | 1 | 2 | 0.00% |
| lmo | 1 | 0 | 0 | 1 | 2 | 0.00% |
| eus | 1 | 0 | 0 | 1 | 2 | 0.00% |
| pan | 1 | 0 | 0 | 1 | 2 | 0.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| all_records | 47,503 |
| all_unique_inputs | 47,452 |
| test_records | 4,715 |
| test_unique_inputs | 4,715 |
| train_records | 37,992 |
| train_unique_inputs | 37,992 |
| val_records | 4,739 |
| val_unique_inputs | 4,739 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0102 | 509,901 / 63,623 | e_1 11%→11%, e_0 12%→12%, e_6 7%→7%, e_16 1%→1%, e_3 10%→10% |
| entities (per mention) | train / test | 0.0077 | 509,901 / 63,163 | e_1 11%→11%, e_0 12%→12%, e_5 8%→8%, e_4 9%→9%, e_10 4%→4% |
| entities (per mention) | val / test | 0.0088 | 63,623 / 63,163 | e_0 12%→12%, e_4 9%→9%, e_5 8%→8%, e_6 7%→7%, e_16 1%→1% |

Thin support in val, entities (per mention): 4 label(s) under 10 gold. e_32 ×2, e_31 ×3, e_30 ×6, e_29 ×8.
Thin support in test, entities (per mention): 4 label(s) under 10 gold. e_30 ×2, e_31 ×3, e_29 ×5, e_28 ×9.
Thin support in all, entities (per mention): 6 label(s) under 10 gold. e_37 ×1, e_38 ×1, e_39 ×1, e_36 ×4, e_34 ×6, e_35 ×6.

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
| e_0 | 59,340 | 11.64% |
| e_1 | 55,570 | 10.90% |
| e_2 | 52,528 | 10.30% |
| e_3 | 49,294 | 9.67% |
| e_4 | 44,835 | 8.79% |
| e_5 | 40,576 | 7.96% |
| e_6 | 35,833 | 7.03% |
| e_7 | 30,867 | 6.05% |
| e_8 | 26,532 | 5.20% |
| e_9 | 22,340 | 4.38% |
| e_10 | 18,629 | 3.65% |
| e_11 | 15,648 | 3.07% |
| e_12 | 12,742 | 2.50% |
| e_13 | 10,143 | 1.99% |
| e_14 | 8,233 | 1.61% |
| e_15 | 6,437 | 1.26% |
| e_16 | 5,011 | 0.98% |
| e_17 | 3,886 | 0.76% |
| e_18 | 2,950 | 0.58% |
| e_19 | 2,261 | 0.44% |
| e_20 | 1,656 | 0.32% |
| e_21 | 1,278 | 0.25% |
| e_22 | 960 | 0.19% |
| e_23 | 671 | 0.13% |
| e_24 | 506 | 0.10% |
| e_25 | 357 | 0.07% |
| e_26 | 251 | 0.05% |
| e_27 | 179 | 0.04% |
| e_28 | 139 | 0.03% |
| e_29 | 93 | 0.02% |
| e_30 | 60 | 0.01% |
| e_31 | 41 | 0.01% |
| e_32 | 25 | 0.00% |
| e_33 | 11 | 0.00% |
| e_34 | 6 | 0.00% |
| e_35 | 6 | 0.00% |
| e_36 | 4 | 0.00% |
| e_37 | 1 | 0.00% |
| e_38 | 1 | 0.00% |
| e_39 | 1 | 0.00% |

#### val

| label | count | share |
| --- | --- | --- |
| e_0 | 7,676 | 12.06% |
| e_1 | 7,214 | 11.34% |
| e_2 | 6,494 | 10.21% |
| e_3 | 6,075 | 9.55% |
| e_4 | 5,613 | 8.82% |
| e_5 | 5,064 | 7.96% |
| e_6 | 4,388 | 6.90% |
| e_7 | 3,905 | 6.14% |
| e_8 | 3,291 | 5.17% |
| e_9 | 2,772 | 4.36% |
| e_10 | 2,350 | 3.69% |
| e_11 | 1,951 | 3.07% |
| e_12 | 1,590 | 2.50% |
| e_13 | 1,259 | 1.98% |
| e_14 | 983 | 1.55% |
| e_15 | 756 | 1.19% |
| e_16 | 547 | 0.86% |
| e_17 | 441 | 0.69% |
| e_18 | 331 | 0.52% |
| e_19 | 250 | 0.39% |
| e_20 | 188 | 0.30% |
| e_21 | 135 | 0.21% |
| e_22 | 106 | 0.17% |
| e_23 | 74 | 0.12% |
| e_24 | 52 | 0.08% |
| e_25 | 40 | 0.06% |
| e_26 | 28 | 0.04% |
| e_27 | 20 | 0.03% |
| e_28 | 11 | 0.02% |
| e_29 | 8 | 0.01% |
| e_30 | 6 | 0.01% |
| e_31 | 3 | 0.00% |
| e_32 | 2 | 0.00% |

#### test

| label | count | share |
| --- | --- | --- |
| e_0 | 7,264 | 11.50% |
| e_1 | 7,162 | 11.34% |
| e_2 | 6,509 | 10.31% |
| e_3 | 6,094 | 9.65% |
| e_4 | 5,484 | 8.68% |
| e_5 | 5,111 | 8.09% |
| e_6 | 4,436 | 7.02% |
| e_7 | 3,865 | 6.12% |
| e_8 | 3,253 | 5.15% |
| e_9 | 2,789 | 4.42% |
| e_10 | 2,355 | 3.73% |
| e_11 | 1,920 | 3.04% |
| e_12 | 1,544 | 2.44% |
| e_13 | 1,264 | 2.00% |
| e_14 | 1,007 | 1.59% |
| e_15 | 778 | 1.23% |
| e_16 | 607 | 0.96% |
| e_17 | 456 | 0.72% |
| e_18 | 358 | 0.57% |
| e_19 | 247 | 0.39% |
| e_20 | 195 | 0.31% |
| e_21 | 147 | 0.23% |
| e_22 | 107 | 0.17% |
| e_23 | 70 | 0.11% |
| e_24 | 51 | 0.08% |
| e_25 | 33 | 0.05% |
| e_26 | 22 | 0.03% |
| e_27 | 16 | 0.03% |
| e_28 | 9 | 0.01% |
| e_29 | 5 | 0.01% |
| e_31 | 3 | 0.00% |
| e_30 | 2 | 0.00% |

#### all

| label | count | share |
| --- | --- | --- |
| e_0 | 74,352 | 11.67% |
| e_1 | 70,008 | 10.99% |
| e_2 | 65,590 | 10.30% |
| e_3 | 61,511 | 9.65% |
| e_4 | 55,977 | 8.79% |
| e_5 | 50,793 | 7.97% |
| e_6 | 44,682 | 7.01% |
| e_7 | 38,656 | 6.07% |
| e_8 | 33,088 | 5.19% |
| e_9 | 27,910 | 4.38% |
| e_10 | 23,341 | 3.66% |
| e_11 | 19,525 | 3.06% |
| e_12 | 15,881 | 2.49% |
| e_13 | 12,667 | 1.99% |
| e_14 | 10,224 | 1.60% |
| e_15 | 7,972 | 1.25% |
| e_16 | 6,166 | 0.97% |
| e_17 | 4,783 | 0.75% |
| e_18 | 3,639 | 0.57% |
| e_19 | 2,758 | 0.43% |
| e_20 | 2,039 | 0.32% |
| e_21 | 1,560 | 0.24% |
| e_22 | 1,173 | 0.18% |
| e_23 | 815 | 0.13% |
| e_24 | 609 | 0.10% |
| e_25 | 430 | 0.07% |
| e_26 | 301 | 0.05% |
| e_27 | 215 | 0.03% |
| e_28 | 159 | 0.02% |
| e_29 | 106 | 0.02% |
| e_30 | 68 | 0.01% |
| e_31 | 47 | 0.01% |
| e_32 | 27 | 0.00% |
| e_33 | 11 | 0.00% |
| e_34 | 6 | 0.00% |
| e_35 | 6 | 0.00% |
| e_36 | 4 | 0.00% |
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
