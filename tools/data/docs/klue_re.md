# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| klue_re | train | 22,521 | 22,521 | 45,042 | 15,994 | 0 | 0 | 0 | 0 |
| klue_re | val | 3,118 | 3,118 | 6,236 | 2,170 | 0 | 0 | 0 | 0 |
| klue_re | test | 3,164 | 3,164 | 6,328 | 2,244 | 0 | 0 | 0 | 0 |

## klue_re

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 22,521 | 22,521 | 64 | 116 | 322 | 45,042 | 15,994 | 0 | 0 | 0 | 0 |
| val | 3,118 | 3,118 | 64 | 117 | 297 | 6,236 | 2,170 | 0 | 0 | 0 | 0 |
| test | 3,164 | 3,164 | 66 | 118 | 308 | 6,328 | 2,244 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| kor | 22,519 | 3,118 | 3,164 | 28,801 | 99.99% |
| sin | 1 | 0 | 0 | 1 | 0.00% |
| ara | 1 | 0 | 0 | 1 | 0.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 3,164 |
| test_unique_inputs | 3,164 |
| train_records | 22,521 |
| train_unique_inputs | 22,521 |
| val_records | 3,118 |
| val_unique_inputs | 3,118 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0177 | 45,042 / 6,236 | PER 40%→41%, ORG 39%→38%, POH 8%→8%, DAT 7%→7%, NOH 1%→1% |
| entities (per mention) | train / test | 0.0096 | 45,042 / 6,328 | PER 40%→40%, LOC 6%→5%, POH 8%→8%, ORG 39%→39%, DAT 7%→6% |
| entities (per mention) | val / test | 0.0179 | 6,236 / 6,328 | ORG 38%→39%, PER 41%→40%, LOC 6%→5%, POH 8%→8%, DAT 7%→6% |
| relations (per instance) | train / val | 0.0519 | 15,994 / 2,170 | per:employee_of 15%→17%, org:top_members/employees 20%→18%, org:product 2%→1%, org:member_of 8%→8%, per:alternate_names 5%→4% |
| relations (per instance) | train / test | 0.0298 | 15,994 / 2,244 | org:top_members/employees 20%→19%, per:title 9%→10%, per:date_of_death 2%→2%, per:alternate_names 5%→4%, per:children 1%→1% |
| relations (per instance) | val / test | 0.0429 | 2,170 / 2,244 | per:employee_of 17%→15%, org:product 1%→2%, org:member_of 8%→8%, per:title 9%→10%, per:children 1%→1% |

Thin support in val, relations (per instance): 2 label(s) under 10 gold. per:place_of_death ×3, org:number_of_employees/members ×7.
Thin support in test, relations (per instance): 5 label(s) under 10 gold. per:place_of_death ×3, org:number_of_employees/members ×3, org:dissolved ×5, per:religion ×8, per:schools_attended ×9.

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
| PER | 17,926 | 39.80% |
| ORG | 17,709 | 39.32% |
| POH | 3,607 | 8.01% |
| DAT | 2,939 | 6.53% |
| LOC | 2,557 | 5.68% |
| NOH | 304 | 0.67% |

#### val

| label | count | share |
| --- | --- | --- |
| PER | 2,580 | 41.37% |
| ORG | 2,366 | 37.94% |
| POH | 482 | 7.73% |
| DAT | 415 | 6.65% |
| LOC | 358 | 5.74% |
| NOH | 35 | 0.56% |

#### test

| label | count | share |
| --- | --- | --- |
| PER | 2,561 | 40.47% |
| ORG | 2,476 | 39.13% |
| POH | 525 | 8.30% |
| DAT | 401 | 6.34% |
| LOC | 327 | 5.17% |
| NOH | 38 | 0.60% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| org:top_members/employees | 3,154 | 19.72% |
| per:employee_of | 2,400 | 15.01% |
| per:title | 1,454 | 9.09% |
| org:member_of | 1,324 | 8.28% |
| org:alternate_names | 941 | 5.88% |
| org:place_of_headquarters | 862 | 5.39% |
| per:origin | 856 | 5.35% |
| per:date_of_birth | 762 | 4.76% |
| per:alternate_names | 726 | 4.54% |
| per:spouse | 521 | 3.26% |
| per:colleagues | 358 | 2.24% |
| org:founded | 336 | 2.10% |
| per:parents | 319 | 1.99% |
| org:members | 303 | 1.89% |
| org:product | 303 | 1.89% |
| per:date_of_death | 244 | 1.53% |
| per:children | 178 | 1.11% |
| per:place_of_residence | 139 | 0.87% |
| org:founded_by | 113 | 0.71% |
| per:place_of_birth | 113 | 0.71% |
| per:other_family | 110 | 0.69% |
| per:product | 93 | 0.58% |
| per:siblings | 84 | 0.53% |
| org:political/religious_affiliation | 72 | 0.45% |
| per:religion | 67 | 0.42% |
| org:dissolved | 49 | 0.31% |
| per:schools_attended | 47 | 0.29% |
| org:number_of_employees/members | 34 | 0.21% |
| per:place_of_death | 32 | 0.20% |

#### val

| label | count | share |
| --- | --- | --- |
| org:top_members/employees | 397 | 18.29% |
| per:employee_of | 368 | 16.96% |
| per:title | 199 | 9.17% |
| org:member_of | 166 | 7.65% |
| org:alternate_names | 128 | 5.90% |
| per:origin | 122 | 5.62% |
| org:place_of_headquarters | 112 | 5.16% |
| per:date_of_birth | 102 | 4.70% |
| per:alternate_names | 88 | 4.06% |
| per:spouse | 66 | 3.04% |
| per:colleagues | 56 | 2.58% |
| per:parents | 53 | 2.44% |
| per:date_of_death | 41 | 1.89% |
| org:founded | 39 | 1.80% |
| org:members | 31 | 1.43% |
| per:place_of_residence | 21 | 0.97% |
| org:product | 21 | 0.97% |
| per:other_family | 19 | 0.88% |
| per:product | 19 | 0.88% |
| per:children | 19 | 0.88% |
| per:place_of_birth | 19 | 0.88% |
| per:siblings | 16 | 0.74% |
| per:schools_attended | 15 | 0.69% |
| org:founded_by | 12 | 0.55% |
| per:religion | 11 | 0.51% |
| org:political/religious_affiliation | 10 | 0.46% |
| org:dissolved | 10 | 0.46% |
| org:number_of_employees/members | 7 | 0.32% |
| per:place_of_death | 3 | 0.14% |

#### test

| label | count | share |
| --- | --- | --- |
| org:top_members/employees | 417 | 18.58% |
| per:employee_of | 342 | 15.24% |
| per:title | 219 | 9.76% |
| org:member_of | 188 | 8.38% |
| org:alternate_names | 136 | 6.06% |
| per:origin | 118 | 5.26% |
| org:place_of_headquarters | 117 | 5.21% |
| per:date_of_birth | 112 | 4.99% |
| per:alternate_names | 94 | 4.19% |
| per:spouse | 69 | 3.07% |
| per:colleagues | 51 | 2.27% |
| per:parents | 50 | 2.23% |
| per:date_of_death | 44 | 1.96% |
| org:product | 44 | 1.96% |
| org:founded | 41 | 1.83% |
| org:members | 39 | 1.74% |
| per:children | 32 | 1.43% |
| per:product | 18 | 0.80% |
| per:other_family | 17 | 0.76% |
| per:place_of_residence | 16 | 0.71% |
| per:place_of_birth | 15 | 0.67% |
| org:founded_by | 13 | 0.58% |
| per:siblings | 13 | 0.58% |
| org:political/religious_affiliation | 11 | 0.49% |
| per:schools_attended | 9 | 0.40% |
| per:religion | 8 | 0.36% |
| org:dissolved | 5 | 0.22% |
| per:place_of_death | 3 | 0.13% |
| org:number_of_employees/members | 3 | 0.13% |

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
