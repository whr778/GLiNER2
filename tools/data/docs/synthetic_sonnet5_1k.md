# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_sonnet5_1k | train | 1,497 | 1,497 | 46,935 | 12,208 | 5,925 | 2,693 | 6,869 | 23,386 |
| synthetic_sonnet5_1k | val | 191 | 191 | 5,900 | 1,571 | 766 | 352 | 881 | 3,022 |
| synthetic_sonnet5_1k | test | 194 | 194 | 6,010 | 1,512 | 762 | 361 | 881 | 2,954 |

## synthetic_sonnet5_1k

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1,497 | 1,497 | 276 | 300 | 366 | 46,935 | 12,208 | 5,925 | 2,693 | 6,869 | 23,386 |
| val | 191 | 191 | 276 | 298 | 325 | 5,900 | 1,571 | 766 | 352 | 881 | 3,022 |
| test | 194 | 194 | 276 | 300 | 326 | 6,010 | 1,512 | 762 | 361 | 881 | 2,954 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 1,497 | 191 | 194 | 1,882 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 194 |
| test_unique_inputs | 194 |
| train_records | 1,497 |
| train_unique_inputs | 1,497 |
| val_records | 191 |
| val_unique_inputs | 191 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0222 | 46,935 / 5,900 | date 19%→19%, quantity 10%→9%, geopolitical entity 4%→4%, location 9%→10%, organization 18%→18% |
| entities (per mention) | train / test | 0.0206 | 46,935 / 6,010 | organization 18%→18%, ordinal 1%→1%, money 6%→6%, person 11%→11%, date 19%→19% |
| entities (per mention) | val / test | 0.0301 | 5,900 / 6,010 | quantity 9%→10%, organization 18%→18%, geopolitical entity 4%→4%, money 6%→6%, person 11%→11% |
| relations (per instance) | train / val | 0.0340 | 12,208 / 1,571 | competitor_of 10%→9%, member_of 2%→3%, works_for 25%→24%, based_in 3%→4%, founded 10%→9% |
| relations (per instance) | train / test | 0.0384 | 12,208 / 1,512 | works_for 25%→24%, located_in 5%→6%, produced_by 6%→7%, leader_of 9%→9%, member_of 2%→2% |
| relations (per instance) | val / test | 0.0503 | 1,571 / 1,512 | member_of 3%→2%, produced_by 6%→7%, leader_of 9%→9%, founded 9%→10%, competitor_of 9%→9% |
| event types (per trigger mention) | train / val | 0.0590 | 6,967 / 899 | Contact.Communicate 9%→7%, Contact.Meet 6%→8%, Business.MergeOrg 1%→2%, Personnel.StartPosition 16%→17%, Disaster.NaturalDisaster 1%→1% |
| event types (per trigger mention) | train / test | 0.0371 | 6,967 / 892 | Contact.Meet 6%→8%, Life.BeBorn 2%→2%, Personnel.StartPosition 16%→15%, Movement.Transport 3%→3%, Transaction.TransferOwnership 12%→13% |
| event types (per trigger mention) | val / test | 0.0649 | 899 / 892 | Contact.Communicate 7%→9%, Personnel.StartPosition 17%→15%, Business.MergeOrg 2%→1%, Life.BeBorn 3%→2%, Personnel.EndPosition 11%→10% |
| event roles (per argument) | train / val | 0.0314 | 23,396 / 3,024 | Communicator 3%→2%, Organization 13%→14%, Participant 3%→3%, Agent 6%→5%, Person 9%→9% |
| event roles (per argument) | train / test | 0.0251 | 23,396 / 2,955 | Time 23%→24%, Organization 13%→13%, Person 9%→8%, Participant 3%→3%, Artifact 3%→4% |
| event roles (per argument) | val / test | 0.0415 | 3,024 / 2,955 | Organization 14%→13%, Person 9%→8%, Time 23%→24%, Communicator 2%→3%, Artifact 3%→4% |
| classifications (true_label) | train / val | 0.0368 | 5,925 / 766 | topic: technology 5%→4%, topic: health 4%→5%, sentiment: neutral 6%→7%, topic: business 17%→18%, sentiment: negative 4%→3% |
| classifications (true_label) | train / test | 0.0418 | 5,925 / 762 | sentiment: neutral 6%→7%, topic: politics 2%→3%, sentiment: positive 16%→15%, topic: sports 2%→2%, topic: science 4%→4% |
| classifications (true_label) | val / test | 0.0397 | 766 / 762 | topic: sports 3%→2%, topic: science 5%→4%, topic: health 5%→4%, topic: technology 4%→5%, topic: politics 2%→3% |

Thin support in val, entities (per mention): 3 label(s) under 10 gold. weapon ×2, language ×4, vehicle ×9.
Thin support in val, relations (per instance): 3 label(s) under 10 gold. part_of ×5, owns ×7, sibling_of ×8.
Thin support in val, event types (per trigger mention): 6 label(s) under 10 gold. Life.Die ×4, Conflict.Attack ×6, Personnel.Elect ×7, Life.Injure ×7, Justice.Arrest ×8, Business.DeclareBankruptcy ×8.
Thin support in val, event roles (per argument): 9 label(s) under 10 gold. Beneficiary ×1, Instrument ×3, Vehicle ×4, Entity ×5, Affected ×6, Attacker ×6, Target ×6, Origin ×8, Demonstrator ×9.
Thin support in val, classifications (true_label): 1 label(s) under 10 gold. topic: education ×7.
Thin support in test, entities (per mention): 3 label(s) under 10 gold. weapon ×3, language ×4, organism ×5.
Thin support in test, relations (per instance): 3 label(s) under 10 gold. part_of ×7, sibling_of ×8, owns ×9.
Thin support in test, event types (per trigger mention): 6 label(s) under 10 gold. Conflict.Attack ×4, Life.Die ×6, Business.MergeOrg ×7, Disaster.NaturalDisaster ×8, Life.Injure ×9, Business.DeclareBankruptcy ×9.
Thin support in test, event roles (per argument): 7 label(s) under 10 gold. Beneficiary ×1, Instrument ×3, Attacker ×3, Target ×3, Affected ×4, Vehicle ×5, Type ×8.
Thin support in test, classifications (true_label): 2 label(s) under 10 gold. topic: world ×1, topic: education ×3.

### Events

**train.** 6,967 instances in 1,488 records, 0 triggerless. Trigger gold 6,869. Argument gold 23,386. Arguments per instance, mean 3.36. Trigger length in whitespace tokens, mean 1.15, single-token share 86.4%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Personnel.StartPosition | 1,112 | 15.96% |
| Business.StartOrg | 1,043 | 14.97% |
| Transaction.TransferOwnership | 850 | 12.20% |
| Personnel.EndPosition | 703 | 10.09% |
| Contact.Communicate | 649 | 9.32% |
| Transaction.TransferMoney | 605 | 8.68% |
| Contact.Meet | 429 | 6.16% |
| Movement.Transport | 176 | 2.53% |
| Award.ReceiveAward | 172 | 2.47% |
| Life.BeBorn | 171 | 2.45% |
| Justice.TrialHearing | 170 | 2.44% |
| Justice.Sentence | 134 | 1.92% |
| Justice.ChargeIndict | 127 | 1.82% |
| Conflict.Demonstrate | 110 | 1.58% |
| Justice.Arrest | 103 | 1.48% |
| Business.DeclareBankruptcy | 95 | 1.36% |
| Business.MergeOrg | 71 | 1.02% |
| Life.Injure | 67 | 0.96% |
| Disaster.NaturalDisaster | 56 | 0.80% |
| Life.Die | 46 | 0.66% |
| Personnel.Elect | 42 | 0.60% |
| Conflict.Attack | 36 | 0.52% |

**val.** 899 instances in 190 records, 0 triggerless. Trigger gold 881. Argument gold 3,022. Arguments per instance, mean 3.36. Trigger length in whitespace tokens, mean 1.15, single-token share 86.0%.

#### val event types

| label | count | share |
| --- | --- | --- |
| Personnel.StartPosition | 151 | 16.80% |
| Business.StartOrg | 129 | 14.35% |
| Transaction.TransferOwnership | 111 | 12.35% |
| Personnel.EndPosition | 96 | 10.68% |
| Transaction.TransferMoney | 76 | 8.45% |
| Contact.Meet | 70 | 7.79% |
| Contact.Communicate | 66 | 7.34% |
| Award.ReceiveAward | 27 | 3.00% |
| Life.BeBorn | 25 | 2.78% |
| Movement.Transport | 22 | 2.45% |
| Business.MergeOrg | 17 | 1.89% |
| Justice.TrialHearing | 17 | 1.89% |
| Justice.Sentence | 15 | 1.67% |
| Justice.ChargeIndict | 14 | 1.56% |
| Disaster.NaturalDisaster | 13 | 1.45% |
| Conflict.Demonstrate | 10 | 1.11% |
| Justice.Arrest | 8 | 0.89% |
| Business.DeclareBankruptcy | 8 | 0.89% |
| Personnel.Elect | 7 | 0.78% |
| Life.Injure | 7 | 0.78% |
| Conflict.Attack | 6 | 0.67% |
| Life.Die | 4 | 0.44% |

**test.** 892 instances in 193 records, 0 triggerless. Trigger gold 881. Argument gold 2,954. Arguments per instance, mean 3.31. Trigger length in whitespace tokens, mean 1.17, single-token share 84.8%.

#### test event types

| label | count | share |
| --- | --- | --- |
| Personnel.StartPosition | 136 | 15.25% |
| Business.StartOrg | 131 | 14.69% |
| Transaction.TransferOwnership | 114 | 12.78% |
| Personnel.EndPosition | 87 | 9.75% |
| Contact.Communicate | 82 | 9.19% |
| Transaction.TransferMoney | 81 | 9.08% |
| Contact.Meet | 67 | 7.51% |
| Movement.Transport | 28 | 3.14% |
| Justice.TrialHearing | 20 | 2.24% |
| Award.ReceiveAward | 20 | 2.24% |
| Justice.Sentence | 16 | 1.79% |
| Justice.ChargeIndict | 15 | 1.68% |
| Life.BeBorn | 15 | 1.68% |
| Justice.Arrest | 14 | 1.57% |
| Conflict.Demonstrate | 13 | 1.46% |
| Personnel.Elect | 10 | 1.12% |
| Life.Injure | 9 | 1.01% |
| Business.DeclareBankruptcy | 9 | 1.01% |
| Disaster.NaturalDisaster | 8 | 0.90% |
| Business.MergeOrg | 7 | 0.78% |
| Life.Die | 6 | 0.67% |
| Conflict.Attack | 4 | 0.45% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 23,396 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 3,024 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 2,955 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.10 | 7.69% | 1.47 | 24.42% | 0.06% | 0 | 23,396 | 0 | 6,967 |
| val | 1.12 | 9.79% | 1.48 | 24.38% | 0.03% | 0 | 3,024 | 0 | 899 |
| test | 1.09 | 6.95% | 1.50 | 25.19% | 0.05% | 0 | 2,955 | 0 | 892 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 6,967 | 0.00% | 816 | 58.21% | 56.99% | 0.61% | 0.12% | 16.13% |
| val | 899 | 0.00% | 113 | 53.10% | 53.10% | 0.00% | 0.00% | 18.42% |
| test | 892 | 0.00% | 90 | 51.11% | 47.78% | 2.22% | 0.00% | 12.95% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 0 | 0 | 816 |
| val | 0 | 0 | 113 |
| test | 0 | 0 | 90 |

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

| split | mentions the string rule adds | arguments skipped, 2+ clusters | arguments in one multi-mention cluster | other mentions of that cluster | argument spans in 2+ clusters |
| --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | 0 | 0 | 0 |
| val | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 |

### Entities

#### train

| label | count | share |
| --- | --- | --- |
| date | 8,782 | 18.71% |
| organization | 8,484 | 18.08% |
| person | 5,118 | 10.90% |
| quantity | 4,686 | 9.98% |
| location | 4,293 | 9.15% |
| job title | 3,793 | 8.08% |
| money | 2,830 | 6.03% |
| geopolitical entity | 1,864 | 3.97% |
| product | 1,335 | 2.84% |
| percent | 1,306 | 2.78% |
| facility | 1,274 | 2.71% |
| ordinal | 688 | 1.47% |
| disease | 427 | 0.91% |
| chemical | 424 | 0.90% |
| work of art | 383 | 0.82% |
| nationality | 311 | 0.66% |
| event | 307 | 0.65% |
| time | 227 | 0.48% |
| law | 201 | 0.43% |
| language | 70 | 0.15% |
| organism | 64 | 0.14% |
| vehicle | 54 | 0.12% |
| weapon | 14 | 0.03% |

#### val

| label | count | share |
| --- | --- | --- |
| date | 1,148 | 19.46% |
| organization | 1,083 | 18.36% |
| person | 657 | 11.14% |
| location | 561 | 9.51% |
| quantity | 546 | 9.25% |
| job title | 464 | 7.86% |
| money | 341 | 5.78% |
| geopolitical entity | 211 | 3.58% |
| facility | 176 | 2.98% |
| percent | 160 | 2.71% |
| product | 156 | 2.64% |
| ordinal | 87 | 1.47% |
| disease | 54 | 0.92% |
| work of art | 54 | 0.92% |
| chemical | 49 | 0.83% |
| event | 44 | 0.75% |
| nationality | 39 | 0.66% |
| time | 24 | 0.41% |
| law | 18 | 0.31% |
| organism | 13 | 0.22% |
| vehicle | 9 | 0.15% |
| language | 4 | 0.07% |
| weapon | 2 | 0.03% |

#### test

| label | count | share |
| --- | --- | --- |
| date | 1,145 | 19.05% |
| organization | 1,053 | 17.52% |
| person | 634 | 10.55% |
| quantity | 613 | 10.20% |
| location | 562 | 9.35% |
| job title | 475 | 7.90% |
| money | 384 | 6.39% |
| geopolitical entity | 253 | 4.21% |
| facility | 179 | 2.98% |
| product | 179 | 2.98% |
| percent | 170 | 2.83% |
| ordinal | 62 | 1.03% |
| disease | 53 | 0.88% |
| chemical | 46 | 0.77% |
| event | 44 | 0.73% |
| work of art | 44 | 0.73% |
| time | 33 | 0.55% |
| nationality | 31 | 0.52% |
| law | 28 | 0.47% |
| vehicle | 10 | 0.17% |
| organism | 5 | 0.08% |
| language | 4 | 0.07% |
| weapon | 3 | 0.05% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| works_for | 3,101 | 25.40% |
| headquartered_in | 1,553 | 12.72% |
| competitor_of | 1,168 | 9.57% |
| founded | 1,160 | 9.50% |
| leader_of | 1,066 | 8.73% |
| produced_by | 744 | 6.09% |
| acquired | 706 | 5.78% |
| located_in | 622 | 5.10% |
| based_in | 369 | 3.02% |
| supplier_of | 316 | 2.59% |
| member_of | 294 | 2.41% |
| subsidiary_of | 248 | 2.03% |
| citizen_of | 150 | 1.23% |
| parent_of | 146 | 1.20% |
| spouse_of | 137 | 1.12% |
| affiliated_with | 131 | 1.07% |
| employer_of | 116 | 0.95% |
| sibling_of | 84 | 0.69% |
| owns | 56 | 0.46% |
| part_of | 41 | 0.34% |

#### val

| label | count | share |
| --- | --- | --- |
| works_for | 384 | 24.44% |
| headquartered_in | 201 | 12.79% |
| founded | 142 | 9.04% |
| leader_of | 135 | 8.59% |
| competitor_of | 134 | 8.53% |
| acquired | 96 | 6.11% |
| produced_by | 93 | 5.92% |
| located_in | 87 | 5.54% |
| based_in | 55 | 3.50% |
| member_of | 53 | 3.37% |
| supplier_of | 37 | 2.36% |
| subsidiary_of | 34 | 2.16% |
| parent_of | 26 | 1.65% |
| citizen_of | 22 | 1.40% |
| spouse_of | 20 | 1.27% |
| affiliated_with | 20 | 1.27% |
| employer_of | 12 | 0.76% |
| sibling_of | 8 | 0.51% |
| owns | 7 | 0.45% |
| part_of | 5 | 0.32% |

#### test

| label | count | share |
| --- | --- | --- |
| works_for | 364 | 24.07% |
| headquartered_in | 186 | 12.30% |
| founded | 149 | 9.85% |
| leader_of | 143 | 9.46% |
| competitor_of | 140 | 9.26% |
| produced_by | 104 | 6.88% |
| located_in | 94 | 6.22% |
| acquired | 88 | 5.82% |
| based_in | 48 | 3.17% |
| supplier_of | 41 | 2.71% |
| subsidiary_of | 35 | 2.31% |
| member_of | 29 | 1.92% |
| parent_of | 15 | 0.99% |
| employer_of | 14 | 0.93% |
| affiliated_with | 14 | 0.93% |
| citizen_of | 13 | 0.86% |
| spouse_of | 11 | 0.73% |
| owns | 9 | 0.60% |
| sibling_of | 8 | 0.53% |
| part_of | 7 | 0.46% |

### Classifications

**train, formality.** Majority class 87.2% of 1,497 answers. The marginal is `true_label`, not the menu.

#### train formality

| label | count | share |
| --- | --- | --- |
| formal | 1,305 | 87.17% |
| informal | 192 | 12.83% |

**train, sentiment.** Majority class 64.0% of 1,497 answers. The marginal is `true_label`, not the menu.

#### train sentiment

| label | count | share |
| --- | --- | --- |
| positive | 958 | 63.99% |
| neutral | 330 | 22.04% |
| negative | 209 | 13.96% |

**train, topic.** Majority class 35.0% of 2,931 answers. The marginal is `true_label`, not the menu.

#### train topic

| label | count | share |
| --- | --- | --- |
| business | 1,025 | 34.97% |
| finance | 580 | 19.79% |
| technology | 312 | 10.64% |
| science | 263 | 8.97% |
| health | 220 | 7.51% |
| legal | 150 | 5.12% |
| sports | 147 | 5.02% |
| politics | 115 | 3.92% |
| environment | 59 | 2.01% |
| education | 52 | 1.77% |
| world | 7 | 0.24% |
| entertainment | 1 | 0.03% |

**val, formality.** Majority class 88.0% of 191 answers. The marginal is `true_label`, not the menu.

#### val formality

| label | count | share |
| --- | --- | --- |
| formal | 168 | 87.96% |
| informal | 23 | 12.04% |

**val, sentiment.** Majority class 62.3% of 191 answers. The marginal is `true_label`, not the menu.

#### val sentiment

| label | count | share |
| --- | --- | --- |
| positive | 119 | 62.30% |
| neutral | 50 | 26.18% |
| negative | 22 | 11.52% |

**val, topic.** Majority class 35.9% of 384 answers. The marginal is `true_label`, not the menu.

#### val topic

| label | count | share |
| --- | --- | --- |
| business | 138 | 35.94% |
| finance | 73 | 19.01% |
| health | 36 | 9.38% |
| science | 36 | 9.38% |
| technology | 30 | 7.81% |
| sports | 20 | 5.21% |
| legal | 17 | 4.43% |
| politics | 17 | 4.43% |
| environment | 10 | 2.60% |
| education | 7 | 1.82% |

**test, formality.** Majority class 85.6% of 194 answers. The marginal is `true_label`, not the menu.

#### test formality

| label | count | share |
| --- | --- | --- |
| formal | 166 | 85.57% |
| informal | 28 | 14.43% |

**test, sentiment.** Majority class 59.3% of 194 answers. The marginal is `true_label`, not the menu.

#### test sentiment

| label | count | share |
| --- | --- | --- |
| positive | 115 | 59.28% |
| neutral | 55 | 28.35% |
| negative | 24 | 12.37% |

**test, topic.** Majority class 36.4% of 374 answers. The marginal is `true_label`, not the menu.

#### test topic

| label | count | share |
| --- | --- | --- |
| business | 136 | 36.36% |
| finance | 75 | 20.05% |
| technology | 36 | 9.63% |
| health | 29 | 7.75% |
| science | 29 | 7.75% |
| politics | 23 | 6.15% |
| legal | 19 | 5.08% |
| sports | 13 | 3.48% |
| environment | 10 | 2.67% |
| education | 3 | 0.80% |
| world | 1 | 0.27% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| job_posting | title | 3 | 100.00% |
| job_posting | company | 3 | 100.00% |
| job_posting | location | 3 | 100.00% |
| job_posting | employment_type | 3 | 100.00% |
| job_posting | record_metadata | natural/location ×3 |  |
| person_profile | name | 1,599 | 99.94% |
| person_profile | employer | 1,581 | 98.81% |
| person_profile | location | 1,563 | 97.69% |
| person_profile | role | 1,486 | 92.88% |
| person_profile | record_metadata | natural/location ×1,205, natural/name ×37 |  |
| product | name | 277 | 100.00% |
| product | brand | 277 | 100.00% |
| product | condition | 277 | 100.00% |
| product | price | 255 | 92.06% |
| product | record_metadata | natural/name ×276 |  |
| transaction | item | 806 | 99.14% |
| transaction | amount | 804 | 98.89% |
| transaction | buyer | 800 | 98.40% |
| transaction | date | 798 | 98.15% |
| transaction | seller | 778 | 95.69% |
| transaction | record_metadata | natural/item ×806, natural/amount ×7 |  |

**val.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| person_profile | name | 203 | 99.51% |
| person_profile | location | 202 | 99.02% |
| person_profile | employer | 201 | 98.53% |
| person_profile | role | 193 | 94.61% |
| person_profile | record_metadata | natural/location ×157, natural/name ×2 |  |
| product | name | 34 | 100.00% |
| product | brand | 34 | 100.00% |
| product | condition | 34 | 100.00% |
| product | price | 30 | 88.24% |
| product | record_metadata | natural/name ×34 |  |
| transaction | item | 111 | 97.37% |
| transaction | buyer | 111 | 97.37% |
| transaction | date | 111 | 97.37% |
| transaction | seller | 111 | 97.37% |
| transaction | amount | 110 | 96.49% |
| transaction | record_metadata | natural/item ×111, natural/amount ×2, natural/date ×1 |  |

**test.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| person_profile | name | 207 | 100.00% |
| person_profile | employer | 204 | 98.55% |
| person_profile | location | 202 | 97.58% |
| person_profile | role | 199 | 96.14% |
| person_profile | record_metadata | natural/location ×153, natural/name ×5 |  |
| product | name | 39 | 100.00% |
| product | brand | 39 | 100.00% |
| product | condition | 39 | 100.00% |
| product | price | 37 | 94.87% |
| product | record_metadata | natural/name ×39 |  |
| transaction | buyer | 114 | 99.13% |
| transaction | date | 113 | 98.26% |
| transaction | amount | 111 | 96.52% |
| transaction | item | 110 | 95.65% |
| transaction | seller | 107 | 93.04% |
| transaction | record_metadata | natural/item ×110, natural/amount ×4, natural/date ×1 |  |

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | llm_synthetic |
| relations | llm_synthetic |
| events | llm_synthetic |
| classifications | llm_synthetic |
| structures | llm_synthetic |
