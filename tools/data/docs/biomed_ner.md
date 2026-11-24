# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| biomed_ner | train | 3,875 | 3,875 | 120,756 | 0 | 0 | 0 | 0 | 0 |
| biomed_ner | val | 485 | 485 | 15,313 | 0 | 0 | 0 | 0 | 0 |
| biomed_ner | test | 470 | 470 | 14,894 | 0 | 0 | 0 | 0 | 0 |
| biomed_ner | all | 4,840 | 4,837 | 151,208 | 0 | 0 | 0 | 0 | 0 |

## biomed_ner

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3,875 | 3,875 | 212 | 370 | 1433 | 120,756 | 0 | 0 | 0 | 0 | 0 |
| val | 485 | 485 | 205 | 366 | 1343 | 15,313 | 0 | 0 | 0 | 0 | 0 |
| test | 470 | 470 | 224 | 372 | 1540 | 14,894 | 0 | 0 | 0 | 0 | 0 |
| all | 4,840 | 4,837 | 212 | 370 | 1540 | 151,208 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | all | records | share |
| --- | --- | --- | --- | --- | --- | --- |
| eng | 3,844 | 480 | 464 | 4,798 | 9,586 | 99.13% |
| und | 29 | 5 | 6 | 40 | 80 | 0.83% |
| epo | 1 | 0 | 0 | 1 | 2 | 0.02% |
| nld | 1 | 0 | 0 | 1 | 2 | 0.02% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| all_records | 4,840 |
| all_unique_inputs | 4,837 |
| test_records | 470 |
| test_unique_inputs | 470 |
| train_records | 3,875 |
| train_unique_inputs | 3,875 |
| val_records | 485 |
| val_unique_inputs | 485 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0452 | 120,756 / 15,313 | INTELLECTUAL PROPERTY 8%→9%, CHEMICALS 13%→11%, ACTIVITY 13%→14%, ORGANISM 4%→3%, MEDICAL PROCEDURE 4%→3% |
| entities (per mention) | train / test | 0.0544 | 120,756 / 14,894 | ACTIVITY 13%→12%, GENE AND GENE PRODUCTS 4%→5%, FUNCTION 8%→9%, CHEMICALS 13%→13%, ORGANIZATION 3%→2% |
| entities (per mention) | val / test | 0.0838 | 15,313 / 14,894 | ACTIVITY 14%→12%, INTELLECTUAL PROPERTY 9%→7%, CHEMICALS 11%→13%, GENE AND GENE PRODUCTS 4%→5%, ORGANISM 3%→4% |

Thin support in val, entities (per mention): 4 label(s) under 10 gold. LOCATIONS ×2, LANGUAGE ×5, INTELLECTUAL ×7, ORGANIZATIONS ×9.
Thin support in test, entities (per mention): 2 label(s) under 10 gold. LANGUAGE ×2, ORGANIZATIONS ×7.

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
| ACTIVITY | 16,226 | 13.44% |
| CHEMICALS | 15,161 | 12.56% |
| PHENOTYPE | 11,345 | 9.39% |
| FUNCTION | 9,538 | 7.90% |
| INTELLECTUAL PROPERTY | 9,175 | 7.60% |
| DISORDER | 8,524 | 7.06% |
| GROUP | 8,395 | 6.95% |
| GENE AND GENE PRODUCTS | 5,063 | 4.19% |
| PRODUCT | 4,867 | 4.03% |
| MEDICAL PROCEDURE | 4,763 | 3.94% |
| ORGANISM | 4,485 | 3.71% |
| ANATOMICAL STRUCTURE | 4,438 | 3.68% |
| ORGANIZATION | 3,416 | 2.83% |
| CELLS AND THEIR COMPONENTS | 2,598 | 2.15% |
| GEOGRAPHICAL AREAS | 2,206 | 1.83% |
| CLINICAL DRUG | 1,937 | 1.60% |
| SIGNALING MOLECULES | 1,368 | 1.13% |
| PRODUCTS | 1,149 | 0.95% |
| PERSON | 895 | 0.74% |
| BODY SUBSTANCE | 820 | 0.68% |
| LOCATION | 719 | 0.60% |
| GENES | 638 | 0.53% |
| FINDINGS/PHENOTYPES | 583 | 0.48% |
| REGULATION OR LAW | 528 | 0.44% |
| DISORDERS | 513 | 0.42% |
| ORGANISMS | 466 | 0.39% |
| EVENT | 381 | 0.32% |
| MONEY | 220 | 0.18% |
| INTELLECTUAL | 151 | 0.13% |
| ORGANIZATIONS | 78 | 0.06% |
| LOCATIONS | 65 | 0.05% |
| LANGUAGE | 30 | 0.02% |
| EVENTS | 15 | 0.01% |

#### val

| label | count | share |
| --- | --- | --- |
| ACTIVITY | 2,166 | 14.14% |
| CHEMICALS | 1,734 | 11.32% |
| PHENOTYPE | 1,446 | 9.44% |
| INTELLECTUAL PROPERTY | 1,426 | 9.31% |
| FUNCTION | 1,252 | 8.18% |
| GROUP | 1,078 | 7.04% |
| DISORDER | 1,012 | 6.61% |
| PRODUCT | 680 | 4.44% |
| GENE AND GENE PRODUCTS | 602 | 3.93% |
| ANATOMICAL STRUCTURE | 596 | 3.89% |
| MEDICAL PROCEDURE | 530 | 3.46% |
| ORGANIZATION | 486 | 3.17% |
| ORGANISM | 475 | 3.10% |
| CELLS AND THEIR COMPONENTS | 305 | 1.99% |
| CLINICAL DRUG | 279 | 1.82% |
| GEOGRAPHICAL AREAS | 247 | 1.61% |
| SIGNALING MOLECULES | 197 | 1.29% |
| LOCATION | 118 | 0.77% |
| PRODUCTS | 118 | 0.77% |
| PERSON | 110 | 0.72% |
| BODY SUBSTANCE | 89 | 0.58% |
| REGULATION OR LAW | 89 | 0.58% |
| DISORDERS | 62 | 0.40% |
| EVENT | 50 | 0.33% |
| GENES | 44 | 0.29% |
| FINDINGS/PHENOTYPES | 41 | 0.27% |
| ORGANISMS | 33 | 0.22% |
| MONEY | 25 | 0.16% |
| ORGANIZATIONS | 9 | 0.06% |
| INTELLECTUAL | 7 | 0.05% |
| LANGUAGE | 5 | 0.03% |
| LOCATIONS | 2 | 0.01% |

#### test

| label | count | share |
| --- | --- | --- |
| CHEMICALS | 1,986 | 13.33% |
| ACTIVITY | 1,794 | 12.05% |
| FUNCTION | 1,301 | 8.74% |
| PHENOTYPE | 1,294 | 8.69% |
| INTELLECTUAL PROPERTY | 1,083 | 7.27% |
| DISORDER | 1,048 | 7.04% |
| GROUP | 939 | 6.30% |
| GENE AND GENE PRODUCTS | 817 | 5.49% |
| PRODUCT | 654 | 4.39% |
| ORGANISM | 637 | 4.28% |
| MEDICAL PROCEDURE | 619 | 4.16% |
| ANATOMICAL STRUCTURE | 505 | 3.39% |
| CELLS AND THEIR COMPONENTS | 388 | 2.61% |
| ORGANIZATION | 315 | 2.11% |
| CLINICAL DRUG | 235 | 1.58% |
| SIGNALING MOLECULES | 226 | 1.52% |
| GEOGRAPHICAL AREAS | 223 | 1.50% |
| PRODUCTS | 134 | 0.90% |
| GENES | 107 | 0.72% |
| LOCATION | 93 | 0.62% |
| DISORDERS | 78 | 0.52% |
| BODY SUBSTANCE | 74 | 0.50% |
| REGULATION OR LAW | 70 | 0.47% |
| PERSON | 68 | 0.46% |
| EVENT | 47 | 0.32% |
| ORGANISMS | 47 | 0.32% |
| INTELLECTUAL | 45 | 0.30% |
| FINDINGS/PHENOTYPES | 34 | 0.23% |
| LOCATIONS | 13 | 0.09% |
| MONEY | 11 | 0.07% |
| ORGANIZATIONS | 7 | 0.05% |
| LANGUAGE | 2 | 0.01% |

#### all

| label | count | share |
| --- | --- | --- |
| ACTIVITY | 20,190 | 13.35% |
| CHEMICALS | 18,982 | 12.55% |
| PHENOTYPE | 14,085 | 9.31% |
| FUNCTION | 12,124 | 8.02% |
| INTELLECTUAL PROPERTY | 11,684 | 7.73% |
| DISORDER | 10,584 | 7.00% |
| GROUP | 10,412 | 6.89% |
| GENE AND GENE PRODUCTS | 6,482 | 4.29% |
| PRODUCT | 6,297 | 4.16% |
| MEDICAL PROCEDURE | 5,914 | 3.91% |
| ORGANISM | 5,599 | 3.70% |
| ANATOMICAL STRUCTURE | 5,541 | 3.66% |
| ORGANIZATION | 4,217 | 2.79% |
| CELLS AND THEIR COMPONENTS | 3,292 | 2.18% |
| GEOGRAPHICAL AREAS | 2,676 | 1.77% |
| CLINICAL DRUG | 2,451 | 1.62% |
| SIGNALING MOLECULES | 1,791 | 1.18% |
| PRODUCTS | 1,401 | 0.93% |
| PERSON | 1,073 | 0.71% |
| BODY SUBSTANCE | 987 | 0.65% |
| LOCATION | 930 | 0.62% |
| GENES | 789 | 0.52% |
| REGULATION OR LAW | 687 | 0.45% |
| FINDINGS/PHENOTYPES | 658 | 0.44% |
| DISORDERS | 653 | 0.43% |
| ORGANISMS | 546 | 0.36% |
| EVENT | 478 | 0.32% |
| MONEY | 256 | 0.17% |
| INTELLECTUAL | 203 | 0.13% |
| ORGANIZATIONS | 94 | 0.06% |
| LOCATIONS | 80 | 0.05% |
| LANGUAGE | 37 | 0.02% |
| EVENTS | 15 | 0.01% |

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
