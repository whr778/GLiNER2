# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lg_twostage | train | 402 | 402 | 10,771 | 356 | 1,318 | 88 | 85 | 228 |
| lg_twostage | val | 49 | 49 | 1,234 | 41 | 159 | 9 | 9 | 18 |
| lg_twostage | test | 49 | 49 | 1,201 | 34 | 151 | 6 | 9 | 24 |

## lg_twostage

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 402 | 402 | 314 | 360 | 559 | 10,771 | 356 | 1,318 | 88 | 85 | 228 |
| val | 49 | 49 | 312 | 357 | 594 | 1,234 | 41 | 159 | 9 | 9 | 18 |
| test | 49 | 49 | 313 | 361 | 519 | 1,201 | 34 | 151 | 6 | 9 | 24 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 402 | 49 | 49 | 500 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 49 |
| test_unique_inputs | 49 |
| train_records | 402 |
| train_unique_inputs | 402 |
| val_records | 49 |
| val_unique_inputs | 49 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.1078 | 10,771 / 1,234 | date 19%→16%, time 3%→2%, cardinal 12%→13%, percent 5%→6%, medication 1%→1% |
| entities (per mention) | train / test | 0.1308 | 10,771 / 1,201 | cardinal 12%→9%, money 7%→10%, time 3%→5%, duration 3%→4%, percent 5%→4% |
| entities (per mention) | val / test | 0.1777 WARN | 1,234 / 1,201 | cardinal 13%→9%, money 6%→10%, time 2%→5%, date 16%→19%, percent 6%→4% |
| relations (per instance) | train / val | 0.3891 STRONG | 356 / 41 | located_in 13%→24%, headquartered_in 4%→12%, based_in 13%→7%, affiliated_with 4%→0%, works_for 11%→15% |
| relations (per instance) | train / test | 0.5173 STRONG | 356 / 34 | located_in 13%→47%, works_for 11%→6%, operates_in 5%→0%, based_in 13%→9%, affiliated_with 4%→0% |
| relations (per instance) | val / test | 0.4484 STRONG | 41 / 34 | located_in 24%→47%, headquartered_in 12%→0%, works_for 15%→6%, operates_in 7%→0%, colleague_of 5%→0% |
| event types (per trigger mention) | train / val | 0.7412 STRONG | 85 / 9 | Conflict.Attack 4%→33%, Medical.Diagnose 1%→22%, Research.Publish 1%→11%, Disaster.Outbreak 5%→11%, Personnel.StartPosition 6%→11% |
| event types (per trigger mention) | train / test | 0.7176 STRONG | 85 / 9 | Contact.Meet 2%→22%, Life.Injure 4%→22%, Business.Layoff 1%→11%, Transaction.TransferMoney 2%→11%, Disaster.Rescue 4%→11% |
| event types (per trigger mention) | val / test | 0.7778 STRONG | 9 / 9 | Conflict.Attack 33%→0%, Medical.Diagnose 22%→0%, Contact.Meet 0%→22%, Life.Injure 0%→22%, Disaster.Rescue 0%→11% |
| event roles (per argument) | train / val | 0.4327 STRONG | 228 / 18 | Attacker 0%→17%, Diagnosis 0%→11%, Author 0%→6%, Instrument 1%→6%, Place 18%→22% |
| event roles (per argument) | train / test | 0.3969 STRONG | 228 / 24 | Participant 1%→12%, Instrument 1%→8%, Person 5%→0%, Organization 9%→4%, Quantity 0%→4% |
| event roles (per argument) | val / test | 0.4306 STRONG | 18 / 24 | Attacker 17%→0%, Participant 0%→12%, Diagnosis 11%→0%, Time 22%→29%, Person 6%→0% |
| classifications (true_label) | train / val | 0.2224 WARN | 1,318 / 159 | audience: expert 4%→1%, certainty: asserted 8%→5%, urgency: routine 5%→8%, topic: science 1%→3%, sentiment: neutral 4%→2% |
| classifications (true_label) | train / test | 0.2104 WARN | 1,318 / 151 | document_genre: correspondence 1%→4%, sentiment: negative 1%→4%, urgency: routine 5%→2%, audience: expert 4%→1%, formality: formal 6%→9% |
| classifications (true_label) | val / test | 0.2809 WARN | 159 / 151 | urgency: routine 8%→2%, topic: science 3%→0%, temporal_orientation: past 4%→7%, sentiment: neutral 2%→5%, document_genre: correspondence 1%→4% |

Thin support in val, entities (per mention): 46 label(s) under 10 gold. contract ×1, aircraft ×1, court case ×1, postal code ×1, fiscal quarter ×1, body of water ×1, work of art ×1, mountain ×1, regulatory body ×1, frequency ×1, weight ×1, model number ×1, stock ticker ×1, academic degree ×1, law ×1, algorithm ×1, criminal organization ×1, patent ×1, book ×1, pathogen ×2, clinical trial ×2, software ×2, family relationship ×2, film ×2, protein ×2, event ×3, chemical ×3, nationality ×3, military rank ×3, non-profit organization ×3, temperature ×4, occupation ×4, address ×4, medical procedure ×4, dosage ×4, era ×5, military unit ×5, ordinal ×5, age ×6, speed ×6, language ×7, product ×7, symptom ×8, landmark ×8, government agency ×8, educational institution ×9.
Unseen in train, present in val, relations (per instance): compatible_with ×1, sponsored_by ×1.
Thin support in val, relations (per instance): 14 label(s) under 10 gold. leader_of ×1, supplier_of ×1, part_of ×1, compatible_with ×1, sponsored_by ×1, treats ×1, employer_of ×2, colleague_of ×2, competitor_of ×2, member_of ×2, operates_in ×3, based_in ×3, headquartered_in ×5, works_for ×6.
Thin support in val, event types (per trigger mention): 6 label(s) under 10 gold. Disaster.NaturalDisaster ×1, Personnel.StartPosition ×1, Research.Publish ×1, Disaster.Outbreak ×1, Medical.Diagnose ×2, Conflict.Attack ×3.
Unseen in train, present in val, event roles (per argument): Author ×1.
Thin support in val, event roles (per argument): 9 label(s) under 10 gold. Type ×1, Person ×1, Organization ×1, Author ×1, Instrument ×1, Diagnosis ×2, Attacker ×3, Place ×4, Time ×4.
Unseen in train, present in val, classifications (true_label): topic: education ×1.
Thin support in val, classifications (true_label): 37 label(s) under 10 gold. urgency: critical ×1, document_genre: advertisement ×1, document_genre: legal filing ×1, topic: energy ×1, topic: finance ×1, topic: education ×1, topic: politics ×1, audience: regulatory ×1, audience: expert ×1, document_genre: correspondence ×2, document_genre: announcement ×2, topic: environment ×2, topic: sports ×2, audience: internal ×2, risk_level: low ×3, risk_level: none ×3, risk_level: high ×3, urgency: elevated ×3, temporal_orientation: present ×3, formality: informal ×3, document_genre: technical documentation ×3, certainty: hedged ×3, topic: world ×3, topic: technology ×3, topic: health ×3, actionability: requires action ×3, sentiment: neutral ×3, sentiment: negative ×3, risk_level: moderate ×4, document_genre: news report ×4, topic: business ×5, topic: science ×5, audience: general public ×5, temporal_orientation: past ×7, language_register: plain ×7, language_register: technical ×8, certainty: asserted ×8.
Thin support in test, entities (per mention): 38 label(s) under 10 gold. protocol ×1, age ×1, clinical trial ×1, publication ×1, military rank ×1, temperature ×1, media outlet ×1, vehicle ×1, natural disaster ×1, database ×1, stock ticker ×1, honorific ×1, military unit ×1, work of art ×1, court case ×1, hardware component ×1, event ×1, api ×1, team role ×2, fiscal quarter ×2, regulation ×2, patent ×2, nationality ×2, symptom ×2, airport ×2, law ×2, ordinal ×2, postal code ×2, weight ×3, chemical ×3, regulatory body ×3, frequency ×4, occupation ×4, disease ×4, medication ×5, brand ×6, model number ×8, address ×9.
Unseen in train, present in test, relations (per instance): successor_of ×1.
Thin support in test, relations (per instance): 11 label(s) under 10 gold. member_of ×1, successor_of ×1, manufactured_by ×1, developed_by ×1, adjacent_to ×1, leader_of ×1, competitor_of ×2, works_for ×2, part_of ×2, based_in ×3, employer_of ×3.
Thin support in test, event types (per trigger mention): 7 label(s) under 10 gold. Disaster.Rescue ×1, Transaction.TransferMoney ×1, Disaster.NaturalDisaster ×1, Personnel.StartPosition ×1, Business.Layoff ×1, Contact.Meet ×2, Life.Injure ×2.
Thin support in test, event roles (per argument): 11 label(s) under 10 gold. Rescuer ×1, Giver ×1, Recipient ×1, Money ×1, Type ×1, Organization ×1, Quantity ×1, Instrument ×2, Participant ×3, Place ×5, Time ×7.
Thin support in test, classifications (true_label): 38 label(s) under 10 gold. document_genre: advertisement ×1, topic: entertainment ×1, topic: travel ×1, topic: agriculture ×1, topic: legal ×1, topic: sports ×1, urgency: elevated ×1, urgency: critical ×1, subjectivity: subjective ×1, temporal_orientation: present ×1, temporal_orientation: timeless ×1, formality: informal ×1, risk_level: none ×1, document_genre: news report ×2, document_genre: announcement ×2, document_genre: technical documentation ×2, document_genre: legal filing ×2, topic: technology ×2, topic: environment ×2, topic: world ×2, audience: expert ×2, risk_level: high ×2, topic: finance ×3, urgency: routine ×3, audience: regulatory ×3, actionability: requires action ×3, actionability: advisory ×3, risk_level: low ×3, risk_level: moderate ×3, topic: business ×4, language_register: technical ×5, language_register: plain ×5, audience: general public ×5, document_genre: correspondence ×6, sentiment: negative ×6, audience: internal ×6, sentiment: neutral ×7, actionability: informational ×9.

### Events

**train.** 85 instances in 68 records, 0 triggerless. Trigger gold 85. Argument gold 228. Arguments per instance, mean 2.68. Trigger length in whitespace tokens, mean 1.25, single-token share 83.5%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Disaster.NaturalDisaster | 8 | 9.41% |
| Personnel.StartPosition | 5 | 5.88% |
| Contact.Communicate | 4 | 4.71% |
| Life.Evacuate | 4 | 4.71% |
| Tech.Release | 4 | 4.71% |
| Life.Die | 4 | 4.71% |
| Life.BeBorn | 4 | 4.71% |
| Business.MergeOrg | 4 | 4.71% |
| Disaster.Outbreak | 4 | 4.71% |
| Disaster.Accident | 3 | 3.53% |
| Conflict.Attack | 3 | 3.53% |
| Life.Injure | 3 | 3.53% |
| Business.StartOrg | 3 | 3.53% |
| Personnel.EndPosition | 3 | 3.53% |
| Disaster.Rescue | 3 | 3.53% |
| Transaction.TransferMoney | 2 | 2.35% |
| Award.ReceiveAward | 2 | 2.35% |
| Justice.Arrest | 2 | 2.35% |
| Transaction.TransferOwnership | 2 | 2.35% |
| Contact.Meet | 2 | 2.35% |
| Medical.Treat | 2 | 2.35% |
| Movement.Transport | 2 | 2.35% |
| Business.LaunchProduct | 2 | 2.35% |
| Medical.Diagnose | 1 | 1.18% |
| Business.Layoff | 1 | 1.18% |
| Research.Discover | 1 | 1.18% |
| Justice.Sentence | 1 | 1.18% |
| Government.Vote | 1 | 1.18% |
| Justice.Sue | 1 | 1.18% |
| Research.Publish | 1 | 1.18% |
| Business.EndOrg | 1 | 1.18% |
| Conflict.Ceasefire | 1 | 1.18% |
| Contact.Negotiate | 1 | 1.18% |

**val.** 9 instances in 7 records, 0 triggerless. Trigger gold 9. Argument gold 18. Arguments per instance, mean 2.00. Trigger length in whitespace tokens, mean 1.33, single-token share 77.8%.

#### val event types

| label | count | share |
| --- | --- | --- |
| Conflict.Attack | 3 | 33.33% |
| Medical.Diagnose | 2 | 22.22% |
| Disaster.NaturalDisaster | 1 | 11.11% |
| Personnel.StartPosition | 1 | 11.11% |
| Research.Publish | 1 | 11.11% |
| Disaster.Outbreak | 1 | 11.11% |

**test.** 9 instances in 8 records, 0 triggerless. Trigger gold 9. Argument gold 24. Arguments per instance, mean 2.67. Trigger length in whitespace tokens, mean 1.56, single-token share 55.6%.

#### test event types

| label | count | share |
| --- | --- | --- |
| Contact.Meet | 2 | 22.22% |
| Life.Injure | 2 | 22.22% |
| Disaster.Rescue | 1 | 11.11% |
| Transaction.TransferMoney | 1 | 11.11% |
| Disaster.NaturalDisaster | 1 | 11.11% |
| Personnel.StartPosition | 1 | 11.11% |
| Business.Layoff | 1 | 11.11% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 228 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 18 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 24 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.33 | 22.35% | 1.39 | 18.06% | 0.32% | 0 | 228 | 0 | 85 |
| val | 1.44 | 22.22% | 1.44 | 27.78% | 0.00% | 0 | 18 | 0 | 9 |
| test | 1.22 | 22.22% | 1.29 | 16.67% | 0.00% | 0 | 24 | 0 | 9 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 85 | 0.00% | 5 | 40.00% | 20.00% | 0.00% | 0.00% | 2.94% |
| val | 9 | 0.00% | 3 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| test | 9 | 0.00% | 1 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 0 | 0 | 5 |
| val | 0 | 0 | 3 |
| test | 0 | 0 | 1 |

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
| date | 2,039 | 18.93% |
| cardinal | 1,304 | 12.11% |
| location | 1,262 | 11.72% |
| money | 744 | 6.91% |
| person | 580 | 5.38% |
| percent | 540 | 5.01% |
| organization | 452 | 4.20% |
| company | 366 | 3.40% |
| time | 361 | 3.35% |
| measurement | 298 | 2.77% |
| duration | 294 | 2.73% |
| job title | 189 | 1.75% |
| facility | 176 | 1.63% |
| quantity | 163 | 1.51% |
| government agency | 125 | 1.16% |
| geopolitical entity | 120 | 1.11% |
| product | 116 | 1.08% |
| disease | 110 | 1.02% |
| distance | 90 | 0.84% |
| region | 86 | 0.80% |
| educational institution | 78 | 0.72% |
| sports team | 69 | 0.64% |
| occupation | 66 | 0.61% |
| address | 64 | 0.59% |
| landmark | 60 | 0.56% |
| road | 58 | 0.54% |
| medication | 57 | 0.53% |
| speed | 53 | 0.49% |
| age | 50 | 0.46% |
| symptom | 42 | 0.39% |
| software | 37 | 0.34% |
| temperature | 33 | 0.31% |
| era | 31 | 0.29% |
| chemical | 30 | 0.28% |
| ordinal | 30 | 0.28% |
| academic degree | 28 | 0.26% |
| military unit | 27 | 0.25% |
| brand | 26 | 0.24% |
| area | 25 | 0.23% |
| frequency | 25 | 0.23% |
| work of art | 21 | 0.19% |
| nationality | 20 | 0.19% |
| fiscal quarter | 20 | 0.19% |
| model number | 20 | 0.19% |
| dosage | 18 | 0.17% |
| weight | 17 | 0.16% |
| deadline | 17 | 0.16% |
| event | 17 | 0.16% |
| body of water | 16 | 0.15% |
| law | 15 | 0.14% |
| military rank | 15 | 0.14% |
| regulatory body | 15 | 0.14% |
| family relationship | 14 | 0.13% |
| stock ticker | 14 | 0.13% |
| clinical trial | 14 | 0.13% |
| regulation | 13 | 0.12% |
| team role | 13 | 0.12% |
| vehicle | 12 | 0.11% |
| organism | 12 | 0.11% |
| publication | 11 | 0.10% |
| non-profit organization | 11 | 0.10% |
| award | 10 | 0.09% |
| court case | 10 | 0.09% |
| aircraft | 8 | 0.07% |
| media outlet | 8 | 0.07% |
| patent | 8 | 0.07% |
| postal code | 7 | 0.06% |
| airport | 6 | 0.06% |
| natural disaster | 6 | 0.06% |
| book | 5 | 0.05% |
| hardware component | 5 | 0.05% |
| medical procedure | 5 | 0.05% |
| pathogen | 4 | 0.04% |
| contract | 4 | 0.04% |
| ethnic group | 4 | 0.04% |
| film | 4 | 0.04% |
| conference | 4 | 0.04% |
| anatomical structure | 4 | 0.04% |
| criminal organization | 3 | 0.03% |
| sports competition | 3 | 0.03% |
| mountain | 3 | 0.03% |
| language | 3 | 0.03% |
| protein | 2 | 0.02% |
| border | 2 | 0.02% |
| vessel | 2 | 0.02% |
| voltage | 2 | 0.02% |
| crime | 2 | 0.02% |
| honorific | 2 | 0.02% |
| protocol | 2 | 0.02% |
| algorithm | 2 | 0.02% |
| security vulnerability | 2 | 0.02% |
| programming language | 2 | 0.02% |
| currency | 1 | 0.01% |
| political affiliation | 1 | 0.01% |
| pseudonym | 1 | 0.01% |
| holiday | 1 | 0.01% |
| database | 1 | 0.01% |
| trade union | 1 | 0.01% |
| witness | 1 | 0.01% |
| sentence | 1 | 0.01% |
| mineral | 0 | 0.00% |
| file format | 0 | 0.00% |
| war | 0 | 0.00% |
| verdict | 0 | 0.00% |
| weapon | 0 | 0.00% |
| allergen | 0 | 0.00% |
| song | 0 | 0.00% |
| interest rate | 0 | 0.00% |
| market index | 0 | 0.00% |
| api | 0 | 0.00% |
| treaty | 0 | 0.00% |
| budget line | 0 | 0.00% |
| tax | 0 | 0.00% |
| video game | 0 | 0.00% |
| framework | 0 | 0.00% |
| vaccine | 0 | 0.00% |
| religious group | 0 | 0.00% |
| species | 0 | 0.00% |
| financial instrument | 0 | 0.00% |
| operating system | 0 | 0.00% |
| political party | 0 | 0.00% |
| gene | 0 | 0.00% |
| diagnosis | 0 | 0.00% |
| license | 0 | 0.00% |
| credit rating | 0 | 0.00% |

#### val

| label | count | share |
| --- | --- | --- |
| date | 196 | 15.88% |
| cardinal | 163 | 13.21% |
| location | 151 | 12.24% |
| money | 79 | 6.40% |
| person | 75 | 6.08% |
| percent | 75 | 6.08% |
| company | 46 | 3.73% |
| organization | 44 | 3.57% |
| duration | 42 | 3.40% |
| measurement | 31 | 2.51% |
| time | 27 | 2.19% |
| facility | 23 | 1.86% |
| disease | 20 | 1.62% |
| job title | 19 | 1.54% |
| medication | 18 | 1.46% |
| road | 18 | 1.46% |
| quantity | 15 | 1.22% |
| geopolitical entity | 15 | 1.22% |
| region | 13 | 1.05% |
| sports team | 12 | 0.97% |
| distance | 12 | 0.97% |
| educational institution | 9 | 0.73% |
| symptom | 8 | 0.65% |
| landmark | 8 | 0.65% |
| government agency | 8 | 0.65% |
| language | 7 | 0.57% |
| product | 7 | 0.57% |
| age | 6 | 0.49% |
| speed | 6 | 0.49% |
| era | 5 | 0.41% |
| military unit | 5 | 0.41% |
| ordinal | 5 | 0.41% |
| temperature | 4 | 0.32% |
| occupation | 4 | 0.32% |
| address | 4 | 0.32% |
| medical procedure | 4 | 0.32% |
| dosage | 4 | 0.32% |
| event | 3 | 0.24% |
| chemical | 3 | 0.24% |
| nationality | 3 | 0.24% |
| military rank | 3 | 0.24% |
| non-profit organization | 3 | 0.24% |
| pathogen | 2 | 0.16% |
| clinical trial | 2 | 0.16% |
| software | 2 | 0.16% |
| family relationship | 2 | 0.16% |
| film | 2 | 0.16% |
| protein | 2 | 0.16% |
| contract | 1 | 0.08% |
| aircraft | 1 | 0.08% |
| court case | 1 | 0.08% |
| postal code | 1 | 0.08% |
| fiscal quarter | 1 | 0.08% |
| body of water | 1 | 0.08% |
| work of art | 1 | 0.08% |
| mountain | 1 | 0.08% |
| regulatory body | 1 | 0.08% |
| frequency | 1 | 0.08% |
| weight | 1 | 0.08% |
| model number | 1 | 0.08% |
| stock ticker | 1 | 0.08% |
| academic degree | 1 | 0.08% |
| law | 1 | 0.08% |
| algorithm | 1 | 0.08% |
| criminal organization | 1 | 0.08% |
| patent | 1 | 0.08% |
| book | 1 | 0.08% |
| war | 0 | 0.00% |
| file format | 0 | 0.00% |
| deadline | 0 | 0.00% |
| mineral | 0 | 0.00% |
| treaty | 0 | 0.00% |
| witness | 0 | 0.00% |
| conference | 0 | 0.00% |
| species | 0 | 0.00% |
| license | 0 | 0.00% |
| allergen | 0 | 0.00% |
| media outlet | 0 | 0.00% |
| video game | 0 | 0.00% |
| religious group | 0 | 0.00% |
| political affiliation | 0 | 0.00% |
| framework | 0 | 0.00% |
| api | 0 | 0.00% |
| gene | 0 | 0.00% |
| budget line | 0 | 0.00% |
| diagnosis | 0 | 0.00% |
| political party | 0 | 0.00% |
| pseudonym | 0 | 0.00% |
| ethnic group | 0 | 0.00% |
| trade union | 0 | 0.00% |
| sentence | 0 | 0.00% |
| holiday | 0 | 0.00% |
| operating system | 0 | 0.00% |
| anatomical structure | 0 | 0.00% |
| regulation | 0 | 0.00% |
| area | 0 | 0.00% |
| interest rate | 0 | 0.00% |
| sports competition | 0 | 0.00% |
| brand | 0 | 0.00% |
| tax | 0 | 0.00% |
| database | 0 | 0.00% |
| financial instrument | 0 | 0.00% |
| publication | 0 | 0.00% |
| voltage | 0 | 0.00% |
| airport | 0 | 0.00% |
| programming language | 0 | 0.00% |
| protocol | 0 | 0.00% |
| market index | 0 | 0.00% |
| team role | 0 | 0.00% |
| border | 0 | 0.00% |
| weapon | 0 | 0.00% |
| award | 0 | 0.00% |
| security vulnerability | 0 | 0.00% |
| currency | 0 | 0.00% |
| vehicle | 0 | 0.00% |
| honorific | 0 | 0.00% |
| song | 0 | 0.00% |
| vessel | 0 | 0.00% |
| natural disaster | 0 | 0.00% |
| organism | 0 | 0.00% |
| crime | 0 | 0.00% |
| verdict | 0 | 0.00% |
| vaccine | 0 | 0.00% |
| credit rating | 0 | 0.00% |
| hardware component | 0 | 0.00% |

#### test

| label | count | share |
| --- | --- | --- |
| date | 224 | 18.65% |
| location | 136 | 11.32% |
| money | 118 | 9.83% |
| cardinal | 110 | 9.16% |
| person | 79 | 6.58% |
| time | 62 | 5.16% |
| duration | 54 | 4.50% |
| percent | 45 | 3.75% |
| company | 42 | 3.50% |
| organization | 38 | 3.16% |
| measurement | 30 | 2.50% |
| product | 23 | 1.92% |
| government agency | 20 | 1.67% |
| job title | 19 | 1.58% |
| facility | 19 | 1.58% |
| landmark | 14 | 1.17% |
| quantity | 14 | 1.17% |
| region | 13 | 1.08% |
| road | 12 | 1.00% |
| geopolitical entity | 11 | 0.92% |
| sports team | 11 | 0.92% |
| speed | 10 | 0.83% |
| distance | 10 | 0.83% |
| address | 9 | 0.75% |
| model number | 8 | 0.67% |
| brand | 6 | 0.50% |
| medication | 5 | 0.42% |
| frequency | 4 | 0.33% |
| occupation | 4 | 0.33% |
| disease | 4 | 0.33% |
| weight | 3 | 0.25% |
| chemical | 3 | 0.25% |
| regulatory body | 3 | 0.25% |
| team role | 2 | 0.17% |
| fiscal quarter | 2 | 0.17% |
| regulation | 2 | 0.17% |
| patent | 2 | 0.17% |
| nationality | 2 | 0.17% |
| symptom | 2 | 0.17% |
| airport | 2 | 0.17% |
| law | 2 | 0.17% |
| ordinal | 2 | 0.17% |
| postal code | 2 | 0.17% |
| protocol | 1 | 0.08% |
| age | 1 | 0.08% |
| clinical trial | 1 | 0.08% |
| publication | 1 | 0.08% |
| military rank | 1 | 0.08% |
| temperature | 1 | 0.08% |
| media outlet | 1 | 0.08% |
| vehicle | 1 | 0.08% |
| natural disaster | 1 | 0.08% |
| database | 1 | 0.08% |
| stock ticker | 1 | 0.08% |
| honorific | 1 | 0.08% |
| military unit | 1 | 0.08% |
| work of art | 1 | 0.08% |
| court case | 1 | 0.08% |
| hardware component | 1 | 0.08% |
| event | 1 | 0.08% |
| api | 1 | 0.08% |
| mineral | 0 | 0.00% |
| conference | 0 | 0.00% |
| license | 0 | 0.00% |
| academic degree | 0 | 0.00% |
| protein | 0 | 0.00% |
| family relationship | 0 | 0.00% |
| era | 0 | 0.00% |
| deadline | 0 | 0.00% |
| song | 0 | 0.00% |
| diagnosis | 0 | 0.00% |
| verdict | 0 | 0.00% |
| educational institution | 0 | 0.00% |
| credit rating | 0 | 0.00% |
| border | 0 | 0.00% |
| organism | 0 | 0.00% |
| religious group | 0 | 0.00% |
| voltage | 0 | 0.00% |
| framework | 0 | 0.00% |
| interest rate | 0 | 0.00% |
| algorithm | 0 | 0.00% |
| pathogen | 0 | 0.00% |
| sentence | 0 | 0.00% |
| species | 0 | 0.00% |
| contract | 0 | 0.00% |
| ethnic group | 0 | 0.00% |
| political affiliation | 0 | 0.00% |
| operating system | 0 | 0.00% |
| vaccine | 0 | 0.00% |
| holiday | 0 | 0.00% |
| body of water | 0 | 0.00% |
| tax | 0 | 0.00% |
| non-profit organization | 0 | 0.00% |
| film | 0 | 0.00% |
| programming language | 0 | 0.00% |
| treaty | 0 | 0.00% |
| allergen | 0 | 0.00% |
| award | 0 | 0.00% |
| vessel | 0 | 0.00% |
| video game | 0 | 0.00% |
| mountain | 0 | 0.00% |
| security vulnerability | 0 | 0.00% |
| medical procedure | 0 | 0.00% |
| file format | 0 | 0.00% |
| weapon | 0 | 0.00% |
| trade union | 0 | 0.00% |
| gene | 0 | 0.00% |
| anatomical structure | 0 | 0.00% |
| pseudonym | 0 | 0.00% |
| book | 0 | 0.00% |
| dosage | 0 | 0.00% |
| political party | 0 | 0.00% |
| currency | 0 | 0.00% |
| criminal organization | 0 | 0.00% |
| area | 0 | 0.00% |
| financial instrument | 0 | 0.00% |
| sports competition | 0 | 0.00% |
| witness | 0 | 0.00% |
| market index | 0 | 0.00% |
| budget line | 0 | 0.00% |
| crime | 0 | 0.00% |
| software | 0 | 0.00% |
| war | 0 | 0.00% |
| aircraft | 0 | 0.00% |
| language | 0 | 0.00% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| based_in | 47 | 13.20% |
| located_in | 46 | 12.92% |
| works_for | 39 | 10.96% |
| employer_of | 19 | 5.34% |
| operates_in | 18 | 5.06% |
| member_of | 18 | 5.06% |
| affiliated_with | 15 | 4.21% |
| headquartered_in | 15 | 4.21% |
| part_of | 15 | 4.21% |
| reports_to | 13 | 3.65% |
| competitor_of | 10 | 2.81% |
| developed_by | 9 | 2.53% |
| participated_in | 8 | 2.25% |
| supplier_of | 7 | 1.97% |
| colleague_of | 6 | 1.69% |
| regulated_by | 6 | 1.69% |
| subsidiary_of | 5 | 1.40% |
| leader_of | 5 | 1.40% |
| member_state_of | 5 | 1.40% |
| produced_by | 4 | 1.12% |
| child_of | 4 | 1.12% |
| depends_on | 4 | 1.12% |
| acquired | 4 | 1.12% |
| manufactured_by | 4 | 1.12% |
| owns | 3 | 0.84% |
| customer_of | 2 | 0.56% |
| designed_by | 2 | 0.56% |
| treats | 2 | 0.56% |
| supersedes | 2 | 0.56% |
| predecessor_of | 2 | 0.56% |
| spouse_of | 2 | 0.56% |
| invested_in | 2 | 0.56% |
| derived_from | 1 | 0.28% |
| board_member_of | 1 | 0.28% |
| published_by | 1 | 0.28% |
| sibling_of | 1 | 0.28% |
| citizen_of | 1 | 0.28% |
| relative_of | 1 | 0.28% |
| born_in | 1 | 0.28% |
| composed_by | 1 | 0.28% |
| advisor_to | 1 | 0.28% |
| adjacent_to | 1 | 0.28% |
| founded | 1 | 0.28% |
| merged_with | 1 | 0.28% |
| licensed_to | 1 | 0.28% |

#### val

| label | count | share |
| --- | --- | --- |
| located_in | 10 | 24.39% |
| works_for | 6 | 14.63% |
| headquartered_in | 5 | 12.20% |
| operates_in | 3 | 7.32% |
| based_in | 3 | 7.32% |
| employer_of | 2 | 4.88% |
| colleague_of | 2 | 4.88% |
| competitor_of | 2 | 4.88% |
| member_of | 2 | 4.88% |
| leader_of | 1 | 2.44% |
| supplier_of | 1 | 2.44% |
| part_of | 1 | 2.44% |
| compatible_with | 1 | 2.44% |
| sponsored_by | 1 | 2.44% |
| treats | 1 | 2.44% |

#### test

| label | count | share |
| --- | --- | --- |
| located_in | 16 | 47.06% |
| based_in | 3 | 8.82% |
| employer_of | 3 | 8.82% |
| competitor_of | 2 | 5.88% |
| works_for | 2 | 5.88% |
| part_of | 2 | 5.88% |
| member_of | 1 | 2.94% |
| successor_of | 1 | 2.94% |
| manufactured_by | 1 | 2.94% |
| developed_by | 1 | 2.94% |
| adjacent_to | 1 | 2.94% |
| leader_of | 1 | 2.94% |

### Classifications

**train, actionability.** Majority class 67.0% of 97 answers. The marginal is `true_label`, not the menu.

#### train actionability

| label | count | share |
| --- | --- | --- |
| informational | 65 | 67.01% |
| requires action | 18 | 18.56% |
| advisory | 14 | 14.43% |

**train, audience.** Majority class 32.3% of 161 answers. The marginal is `true_label`, not the menu.

#### train audience

| label | count | share |
| --- | --- | --- |
| expert | 52 | 32.30% |
| general public | 44 | 27.33% |
| regulatory | 38 | 23.60% |
| internal | 27 | 16.77% |

**train, certainty.** Majority class 97.2% of 108 answers. The marginal is `true_label`, not the menu.

#### train certainty

| label | count | share |
| --- | --- | --- |
| asserted | 105 | 97.22% |
| hedged | 3 | 2.78% |

**train, document_genre.** Majority class 24.5% of 94 answers. The marginal is `true_label`, not the menu.

#### train document_genre

| label | count | share |
| --- | --- | --- |
| technical documentation | 23 | 24.47% |
| news report | 20 | 21.28% |
| announcement | 16 | 17.02% |
| legal filing | 13 | 13.83% |
| opinion | 9 | 9.57% |
| correspondence | 7 | 7.45% |
| advertisement | 4 | 4.26% |
| review | 2 | 2.13% |

**train, formality.** Majority class 90.2% of 92 answers. The marginal is `true_label`, not the menu.

#### train formality

| label | count | share |
| --- | --- | --- |
| formal | 83 | 90.22% |
| informal | 9 | 9.78% |

**train, language_register.** Majority class 60.4% of 111 answers. The marginal is `true_label`, not the menu.

#### train language_register

| label | count | share |
| --- | --- | --- |
| technical | 67 | 60.36% |
| plain | 37 | 33.33% |
| legal | 7 | 6.31% |

**train, risk_level.** Majority class 28.3% of 99 answers. The marginal is `true_label`, not the menu.

#### train risk_level

| label | count | share |
| --- | --- | --- |
| moderate | 28 | 28.28% |
| none | 25 | 25.25% |
| high | 25 | 25.25% |
| low | 21 | 21.21% |

**train, sentiment.** Majority class 64.7% of 85 answers. The marginal is `true_label`, not the menu.

#### train sentiment

| label | count | share |
| --- | --- | --- |
| neutral | 55 | 64.71% |
| negative | 16 | 18.82% |
| positive | 14 | 16.47% |

**train, subjectivity.** Majority class 92.8% of 97 answers. The marginal is `true_label`, not the menu.

#### train subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 90 | 92.78% |
| subjective | 7 | 7.22% |

**train, temporal_orientation.** Majority class 82.1% of 106 answers. The marginal is `true_label`, not the menu.

#### train temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 87 | 82.08% |
| present | 16 | 15.09% |
| timeless | 2 | 1.89% |
| future | 1 | 0.94% |

**train, topic.** Majority class 26.0% of 169 answers. The marginal is `true_label`, not the menu.

#### train topic

| label | count | share |
| --- | --- | --- |
| business | 44 | 26.04% |
| environment | 18 | 10.65% |
| finance | 16 | 9.47% |
| world | 14 | 8.28% |
| legal | 13 | 7.69% |
| politics | 12 | 7.10% |
| health | 10 | 5.92% |
| technology | 9 | 5.33% |
| science | 9 | 5.33% |
| energy | 6 | 3.55% |
| defense | 6 | 3.55% |
| agriculture | 6 | 3.55% |
| sports | 3 | 1.78% |
| travel | 2 | 1.18% |
| entertainment | 1 | 0.59% |

**train, urgency.** Majority class 62.6% of 99 answers. The marginal is `true_label`, not the menu.

#### train urgency

| label | count | share |
| --- | --- | --- |
| routine | 62 | 62.63% |
| elevated | 25 | 25.25% |
| critical | 12 | 12.12% |

**val, actionability.** Majority class 78.6% of 14 answers. The marginal is `true_label`, not the menu.

#### val actionability

| label | count | share |
| --- | --- | --- |
| informational | 11 | 78.57% |
| requires action | 3 | 21.43% |

**val, audience.** Majority class 55.6% of 9 answers. The marginal is `true_label`, not the menu.

#### val audience

| label | count | share |
| --- | --- | --- |
| general public | 5 | 55.56% |
| internal | 2 | 22.22% |
| regulatory | 1 | 11.11% |
| expert | 1 | 11.11% |

**val, certainty.** Majority class 72.7% of 11 answers. The marginal is `true_label`, not the menu.

#### val certainty

| label | count | share |
| --- | --- | --- |
| asserted | 8 | 72.73% |
| hedged | 3 | 27.27% |

**val, document_genre.** Majority class 30.8% of 13 answers. The marginal is `true_label`, not the menu.

#### val document_genre

| label | count | share |
| --- | --- | --- |
| news report | 4 | 30.77% |
| technical documentation | 3 | 23.08% |
| correspondence | 2 | 15.38% |
| announcement | 2 | 15.38% |
| advertisement | 1 | 7.69% |
| legal filing | 1 | 7.69% |

**val, formality.** Majority class 78.6% of 14 answers. The marginal is `true_label`, not the menu.

#### val formality

| label | count | share |
| --- | --- | --- |
| formal | 11 | 78.57% |
| informal | 3 | 21.43% |

**val, language_register.** Majority class 53.3% of 15 answers. The marginal is `true_label`, not the menu.

#### val language_register

| label | count | share |
| --- | --- | --- |
| technical | 8 | 53.33% |
| plain | 7 | 46.67% |

**val, risk_level.** Majority class 30.8% of 13 answers. The marginal is `true_label`, not the menu.

#### val risk_level

| label | count | share |
| --- | --- | --- |
| moderate | 4 | 30.77% |
| low | 3 | 23.08% |
| none | 3 | 23.08% |
| high | 3 | 23.08% |

**val, sentiment.** Majority class 50.0% of 6 answers. The marginal is `true_label`, not the menu.

#### val sentiment

| label | count | share |
| --- | --- | --- |
| neutral | 3 | 50.00% |
| negative | 3 | 50.00% |

**val, subjectivity.** Majority class 100.0% of 11 answers. The marginal is `true_label`, not the menu.

#### val subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 11 | 100.00% |

**val, temporal_orientation.** Majority class 70.0% of 10 answers. The marginal is `true_label`, not the menu.

#### val temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 7 | 70.00% |
| present | 3 | 30.00% |

**val, topic.** Majority class 18.5% of 27 answers. The marginal is `true_label`, not the menu.

#### val topic

| label | count | share |
| --- | --- | --- |
| business | 5 | 18.52% |
| science | 5 | 18.52% |
| world | 3 | 11.11% |
| technology | 3 | 11.11% |
| health | 3 | 11.11% |
| environment | 2 | 7.41% |
| sports | 2 | 7.41% |
| energy | 1 | 3.70% |
| finance | 1 | 3.70% |
| education | 1 | 3.70% |
| politics | 1 | 3.70% |

**val, urgency.** Majority class 75.0% of 16 answers. The marginal is `true_label`, not the menu.

#### val urgency

| label | count | share |
| --- | --- | --- |
| routine | 12 | 75.00% |
| elevated | 3 | 18.75% |
| critical | 1 | 6.25% |

**test, actionability.** Majority class 60.0% of 15 answers. The marginal is `true_label`, not the menu.

#### test actionability

| label | count | share |
| --- | --- | --- |
| informational | 9 | 60.00% |
| requires action | 3 | 20.00% |
| advisory | 3 | 20.00% |

**test, audience.** Majority class 37.5% of 16 answers. The marginal is `true_label`, not the menu.

#### test audience

| label | count | share |
| --- | --- | --- |
| internal | 6 | 37.50% |
| general public | 5 | 31.25% |
| regulatory | 3 | 18.75% |
| expert | 2 | 12.50% |

**test, certainty.** Majority class 100.0% of 10 answers. The marginal is `true_label`, not the menu.

#### test certainty

| label | count | share |
| --- | --- | --- |
| asserted | 10 | 100.00% |

**test, document_genre.** Majority class 40.0% of 15 answers. The marginal is `true_label`, not the menu.

#### test document_genre

| label | count | share |
| --- | --- | --- |
| correspondence | 6 | 40.00% |
| news report | 2 | 13.33% |
| announcement | 2 | 13.33% |
| technical documentation | 2 | 13.33% |
| legal filing | 2 | 13.33% |
| advertisement | 1 | 6.67% |

**test, formality.** Majority class 92.9% of 14 answers. The marginal is `true_label`, not the menu.

#### test formality

| label | count | share |
| --- | --- | --- |
| formal | 13 | 92.86% |
| informal | 1 | 7.14% |

**test, language_register.** Majority class 50.0% of 10 answers. The marginal is `true_label`, not the menu.

#### test language_register

| label | count | share |
| --- | --- | --- |
| technical | 5 | 50.00% |
| plain | 5 | 50.00% |

**test, risk_level.** Majority class 33.3% of 9 answers. The marginal is `true_label`, not the menu.

#### test risk_level

| label | count | share |
| --- | --- | --- |
| low | 3 | 33.33% |
| moderate | 3 | 33.33% |
| high | 2 | 22.22% |
| none | 1 | 11.11% |

**test, sentiment.** Majority class 53.8% of 13 answers. The marginal is `true_label`, not the menu.

#### test sentiment

| label | count | share |
| --- | --- | --- |
| neutral | 7 | 53.85% |
| negative | 6 | 46.15% |

**test, subjectivity.** Majority class 92.3% of 13 answers. The marginal is `true_label`, not the menu.

#### test subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 12 | 92.31% |
| subjective | 1 | 7.69% |

**test, temporal_orientation.** Majority class 84.6% of 13 answers. The marginal is `true_label`, not the menu.

#### test temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 11 | 84.62% |
| present | 1 | 7.69% |
| timeless | 1 | 7.69% |

**test, topic.** Majority class 22.2% of 18 answers. The marginal is `true_label`, not the menu.

#### test topic

| label | count | share |
| --- | --- | --- |
| business | 4 | 22.22% |
| finance | 3 | 16.67% |
| technology | 2 | 11.11% |
| environment | 2 | 11.11% |
| world | 2 | 11.11% |
| entertainment | 1 | 5.56% |
| travel | 1 | 5.56% |
| agriculture | 1 | 5.56% |
| legal | 1 | 5.56% |
| sports | 1 | 5.56% |

**test, urgency.** Majority class 60.0% of 5 answers. The marginal is `true_label`, not the menu.

#### test urgency

| label | count | share |
| --- | --- | --- |
| routine | 3 | 60.00% |
| elevated | 1 | 20.00% |
| critical | 1 | 20.00% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | outcome | 5 | 100.00% |
| clinical_finding | patient_group | 4 | 80.00% |
| clinical_finding | treatment | 4 | 80.00% |
| clinical_finding | condition | 2 | 40.00% |
| clinical_finding | record_metadata | natural/patient_group ×3, natural/condition ×2 |  |
| company_profile | name | 14 | 93.33% |
| company_profile | size | 12 | 80.00% |
| company_profile | headquarters | 7 | 46.67% |
| company_profile | founded | 2 | 13.33% |
| company_profile | industry | 2 | 13.33% |
| company_profile | record_metadata | natural/name ×14, natural/size ×1 |  |
| event_listing | name | 3 | 100.00% |
| event_listing | venue | 3 | 100.00% |
| event_listing | date | 2 | 66.67% |
| event_listing | organizer | 1 | 33.33% |
| event_listing | record_metadata | natural/name ×3 |  |
| financial_report | organization | 13 | 100.00% |
| financial_report | period | 13 | 100.00% |
| financial_report | revenue | 8 | 61.54% |
| financial_report | net_income | 5 | 38.46% |
| financial_report | record_metadata | natural/organization ×12 |  |
| incident_report | date | 15 | 100.00% |
| incident_report | severity | 15 | 100.00% |
| incident_report | location | 12 | 80.00% |
| incident_report | incident_type | 8 | 53.33% |
| incident_report | casualties | 7 | 46.67% |
| incident_report | record_metadata | natural/incident_type ×8, natural/location ×6, natural/date ×1 |  |
| legal_case | status | 4 | 100.00% |
| legal_case | plaintiff | 2 | 50.00% |
| legal_case | defendant | 2 | 50.00% |
| legal_case | case_name | 1 | 25.00% |
| legal_case | court | 1 | 25.00% |
| legal_case | record_metadata | natural/plaintiff ×1, natural/case_name ×1, natural/status ×1, natural/court ×1 |  |
| person_profile | name | 5 | 83.33% |
| person_profile | role | 5 | 83.33% |
| person_profile | employer | 5 | 83.33% |
| person_profile | location | 5 | 83.33% |
| person_profile | record_metadata | natural/name ×5, natural/role ×1 |  |
| product | name | 8 | 100.00% |
| product | condition | 7 | 87.50% |
| product | brand | 6 | 75.00% |
| product | price | 2 | 25.00% |
| product | record_metadata | natural/name ×8 |  |
| real_estate_listing | address | 1 | 100.00% |
| real_estate_listing | price | 1 | 100.00% |
| real_estate_listing | bedrooms | 1 | 100.00% |
| real_estate_listing | area | 1 | 100.00% |
| real_estate_listing | property_type | 1 | 100.00% |
| real_estate_listing | record_metadata | natural/address ×1 |  |
| research_paper | title | 7 | 100.00% |
| research_paper | venue | 2 | 28.57% |
| research_paper | authors | 1 | 14.29% |
| research_paper | year | 1 | 14.29% |
| research_paper | topic | 1 | 14.29% |
| research_paper | record_metadata | natural/title ×7 |  |
| transaction | amount | 9 | 100.00% |
| transaction | date | 8 | 88.89% |
| transaction | item | 7 | 77.78% |
| transaction | buyer | 5 | 55.56% |
| transaction | seller | 4 | 44.44% |
| transaction | record_metadata | natural/item ×7, natural/amount ×2 |  |
| vehicle_listing | make | 2 | 100.00% |
| vehicle_listing | model | 2 | 100.00% |
| vehicle_listing | year | 2 | 100.00% |
| vehicle_listing | mileage | 1 | 50.00% |
| vehicle_listing | price | 1 | 50.00% |
| vehicle_listing | record_metadata | natural/make ×2 |  |

**val.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | condition | 2 | 100.00% |
| clinical_finding | patient_group | 2 | 100.00% |
| clinical_finding | treatment | 1 | 50.00% |
| clinical_finding | outcome | 1 | 50.00% |
| clinical_finding | record_metadata | natural/condition ×2 |  |
| financial_report | period | 2 | 100.00% |
| financial_report | organization | 1 | 50.00% |
| financial_report | revenue | 1 | 50.00% |
| financial_report | record_metadata | natural/period ×1, natural/organization ×1 |  |
| incident_report | incident_type | 2 | 100.00% |
| incident_report | location | 2 | 100.00% |
| incident_report | date | 2 | 100.00% |
| incident_report | casualties | 2 | 100.00% |
| incident_report | severity | 2 | 100.00% |
| incident_report | record_metadata | natural/incident_type ×2 |  |
| research_paper | title | 1 | 100.00% |
| research_paper | authors | 1 | 100.00% |
| research_paper | venue | 1 | 100.00% |
| research_paper | year | 1 | 100.00% |
| research_paper | record_metadata | natural/title ×1 |  |
| transaction | item | 2 | 100.00% |
| transaction | amount | 1 | 50.00% |
| transaction | date | 1 | 50.00% |
| transaction | record_metadata | natural/item ×2 |  |

**test.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| company_profile | name | 1 | 100.00% |
| company_profile | headquarters | 1 | 100.00% |
| company_profile | size | 1 | 100.00% |
| company_profile | record_metadata | natural/name ×1 |  |
| financial_report | organization | 1 | 100.00% |
| financial_report | period | 1 | 100.00% |
| financial_report | revenue | 1 | 100.00% |
| financial_report | record_metadata | natural/organization ×1 |  |
| incident_report | location | 2 | 100.00% |
| incident_report | date | 2 | 100.00% |
| incident_report | severity | 2 | 100.00% |
| incident_report | incident_type | 1 | 50.00% |
| incident_report | record_metadata | natural/location ×1, natural/incident_type ×1 |  |
| person_profile | name | 1 | 100.00% |
| person_profile | role | 1 | 100.00% |
| person_profile | record_metadata | natural/name ×1 |  |
| vehicle_listing | make | 1 | 100.00% |
| vehicle_listing | model | 1 | 100.00% |
| vehicle_listing | record_metadata | natural/make ×1 |  |

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | unknown |
| relations | unknown |
| events | unknown |
| classifications | unknown |
| structures | unknown |
