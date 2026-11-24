# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cc_news_events_haiku45 | train | 6,562 | 6,562 | 136,582 | 0 | 0 | 0 | 10,314 | 27,066 |
| cc_news_events_haiku45 | val | 803 | 803 | 15,912 | 0 | 0 | 0 | 1,220 | 3,164 |
| cc_news_events_haiku45 | test | 895 | 895 | 19,153 | 0 | 0 | 0 | 1,278 | 3,308 |

## cc_news_events_haiku45

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 6,562 | 6,562 | 405 | 1027 | 2582 | 136,582 | 0 | 0 | 0 | 10,314 | 27,066 |
| val | 803 | 803 | 387 | 965 | 2493 | 15,912 | 0 | 0 | 0 | 1,220 | 3,164 |
| test | 895 | 895 | 426 | 1026 | 2342 | 19,153 | 0 | 0 | 0 | 1,278 | 3,308 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 6,559 | 802 | 895 | 8,256 | 99.95% |
| und | 3 | 1 | 0 | 4 | 0.05% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 895 |
| test_unique_inputs | 895 |
| train_records | 6,562 |
| train_unique_inputs | 6,562 |
| val_records | 803 |
| val_unique_inputs | 803 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0403 | 136,582 / 15,912 | location 13%→12%, person 22%→21%, temperature 0%→1%, company 3%→3%, product 2%→2% |
| entities (per mention) | train / test | 0.0453 | 136,582 / 19,153 | person 22%→20%, cardinal 7%→8%, percent 1%→2%, company 3%→3%, geopolitical entity 3%→2% |
| entities (per mention) | val / test | 0.0481 | 15,912 / 19,153 | cardinal 7%→8%, person 21%→20%, geopolitical entity 3%→2%, media outlet 3%→2%, temperature 1%→0% |
| event types (per trigger mention) | train / val | 0.0894 | 11,365 / 1,335 | Contact.Communicate 8%→10%, Personnel.EndPosition 3%→5%, Conflict.Demonstrate 1%→2%, Sport.Compete 14%→12%, Life.Injure 4%→3% |
| event types (per trigger mention) | train / test | 0.0717 | 11,365 / 1,378 | Business.LaunchProduct 3%→4%, Transaction.Donate 1%→2%, Justice.ChargeIndict 4%→3%, Personnel.EndPosition 3%→2%, Justice.Fine 0%→1% |
| event types (per trigger mention) | val / test | 0.1081 | 1,335 / 1,378 | Personnel.EndPosition 5%→2%, Contact.Communicate 10%→8%, Conflict.Demonstrate 2%→1%, Sport.Compete 12%→13%, Transaction.Donate 1%→2% |
| event roles (per argument) | train / val | 0.0575 | 27,852 / 3,260 | Organization 5%→6%, Communicator 3%→4%, Participant 5%→5%, Person 5%→6%, Demonstrator 1%→1% |
| event roles (per argument) | train / test | 0.0460 | 27,852 / 3,392 | Participant 5%→4%, Organization 5%→6%, Result 2%→2%, Competitor 9%→9%, Crime 2%→2% |
| event roles (per argument) | val / test | 0.0644 | 3,260 / 3,392 | Communicator 4%→3%, Participant 5%→4%, Person 6%→5%, Victim 4%→5%, Demonstrator 1%→1% |

Thin support in val, entities (per mention): 35 label(s) under 10 gold. clinical trial ×1, fiscal quarter ×1, diagnosis ×1, protein ×1, pathogen ×1, mountain ×1, operating system ×1, patent ×2, natural disaster ×2, vessel ×2, dosage ×2, programming language ×2, market index ×3, protocol ×3, frequency ×3, vaccine ×3, war ×3, financial instrument ×3, trade union ×4, speed ×4, symptom ×4, interest rate ×5, witness ×5, postal code ×5, species ×6, stock ticker ×6, model number ×6, regulation ×7, treaty ×7, conference ×8, aircraft ×8, criminal organization ×8, area ×9, sentence ×9, software ×9.
Thin support in val, event types (per trigger mention): 21 label(s) under 10 gold. Business.Layoff ×1, Movement.Deport ×1, Business.Strike ×2, Conflict.Ceasefire ×2, Disaster.Outbreak ×2, Justice.Acquit ×3, Life.Divorce ×4, Research.Fund ×4, Personnel.Elect ×4, Justice.Fine ×4, Tech.Breach ×4, Justice.Appeal ×4, Business.DeclareBankruptcy ×5, Business.EndOrg ×5, Business.MergeOrg ×5, Personnel.Nominate ×7, Medical.Diagnose ×8, Research.Discover ×8, Life.Marry ×8, Government.EnactLaw ×8, Disaster.Rescue ×8.
Unseen in train, present in val, event roles (per argument): Version ×1, Mediator ×1.
Thin support in val, event roles (per argument): 26 label(s) under 10 gold. Version ×1, Quantity ×1, Rescuer ×1, Mediator ×1, Striker ×2, Topic ×2, Disease ×2, Data ×2, Affected ×3, Price ×3, Appellant ×3, Awarder ×4, Funder ×4, Reason ×4, Physician ×5, Researcher ×5, Money ×5, Proposal ×5, Treatment ×6, Prosecutor ×6, Venue ×6, Law ×6, Diagnosis ×7, Finding ×7, Voter ×7, Evacuee ×8.
Thin support in test, entities (per mention): 26 label(s) under 10 gold. model number ×1, clinical trial ×1, protein ×1, hardware component ×1, witness ×1, mountain ×2, algorithm ×2, area ×2, deadline ×2, trade union ×2, natural disaster ×2, sentence ×2, file format ×3, war ×3, medical procedure ×4, diagnosis ×4, contract ×4, financial instrument ×4, criminal organization ×6, language ×7, treaty ×7, operating system ×7, postal code ×7, frequency ×9, court case ×9, conference ×9.
Thin support in test, event types (per trigger mention): 19 label(s) under 10 gold. Business.DeclareBankruptcy ×1, Justice.Acquit ×1, Business.Layoff ×1, Transaction.TransferMoney ×2, Life.Divorce ×2, Justice.Appeal ×3, Tech.Breach ×4, Business.EndOrg ×4, Research.Fund ×4, Business.Strike ×4, Disaster.Outbreak ×4, Contact.Negotiate ×5, Personnel.Nominate ×5, Business.MergeOrg ×8, Life.Evacuate ×8, Movement.Migrate ×9, Disaster.Rescue ×9, Research.Discover ×9, Life.Marry ×9.
Thin support in test, event roles (per argument): 19 label(s) under 10 gold. Prosecutor ×1, Quantity ×1, Venue ×2, Giver ×2, Topic ×3, Appellant ×3, Funder ×3, Affected ×4, Striker ×4, Disease ×4, Awarder ×5, Rescuer ×5, Evacuee ×5, Researcher ×6, Reason ×6, Treatment ×6, Migrant ×8, Finding ×8, Physician ×9.

### Events

**train.** 11,365 instances in 4,343 records, 0 triggerless. Trigger gold 10,314. Argument gold 27,066. Arguments per instance, mean 2.45. Trigger length in whitespace tokens, mean 1.28, single-token share 82.0%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Sport.Compete | 1,540 | 13.55% |
| Contact.Communicate | 932 | 8.20% |
| Life.Die | 683 | 6.01% |
| Contact.Meet | 633 | 5.57% |
| Conflict.Attack | 576 | 5.07% |
| Life.Injure | 477 | 4.20% |
| Personnel.StartPosition | 445 | 3.92% |
| Justice.Arrest | 433 | 3.81% |
| Award.ReceiveAward | 428 | 3.77% |
| Justice.ChargeIndict | 410 | 3.61% |
| Business.LaunchProduct | 386 | 3.40% |
| Personnel.EndPosition | 342 | 3.01% |
| Disaster.Accident | 271 | 2.38% |
| Justice.TrialHearing | 235 | 2.07% |
| Government.Vote | 221 | 1.94% |
| Movement.Arrive | 195 | 1.72% |
| Research.Publish | 173 | 1.52% |
| Medical.Treat | 171 | 1.50% |
| Tech.Release | 167 | 1.47% |
| Business.StartOrg | 166 | 1.46% |
| Justice.Sentence | 166 | 1.46% |
| Transaction.TransferOwnership | 142 | 1.25% |
| Transaction.Donate | 139 | 1.22% |
| Government.EnactLaw | 121 | 1.06% |
| Medical.Diagnose | 117 | 1.03% |
| Life.BeBorn | 115 | 1.01% |
| Movement.Transport | 113 | 0.99% |
| Justice.Sue | 112 | 0.99% |
| Disaster.NaturalDisaster | 112 | 0.99% |
| Conflict.Demonstrate | 111 | 0.98% |
| Government.Protest | 102 | 0.90% |
| Life.Marry | 102 | 0.90% |
| Contact.Negotiate | 96 | 0.84% |
| Research.Discover | 87 | 0.77% |
| Life.Evacuate | 84 | 0.74% |
| Personnel.Nominate | 68 | 0.60% |
| Business.EndOrg | 63 | 0.55% |
| Disaster.Rescue | 60 | 0.53% |
| Movement.Migrate | 58 | 0.51% |
| Life.Divorce | 56 | 0.49% |
| Government.Sanction | 54 | 0.48% |
| Personnel.Elect | 45 | 0.40% |
| Business.MergeOrg | 42 | 0.37% |
| Tech.Breach | 40 | 0.35% |
| Research.Fund | 39 | 0.34% |
| Justice.Appeal | 38 | 0.33% |
| Justice.Fine | 33 | 0.29% |
| Movement.Deport | 32 | 0.28% |
| Transaction.TransferMoney | 30 | 0.26% |
| Business.Strike | 26 | 0.23% |
| Business.DeclareBankruptcy | 19 | 0.17% |
| Business.Layoff | 19 | 0.17% |
| Disaster.Outbreak | 18 | 0.16% |
| Justice.Acquit | 17 | 0.15% |
| Conflict.Ceasefire | 3 | 0.03% |
| Transaction.Lend | 2 | 0.02% |

**val.** 1,335 instances in 537 records, 0 triggerless. Trigger gold 1,220. Argument gold 3,164. Arguments per instance, mean 2.44. Trigger length in whitespace tokens, mean 1.32, single-token share 80.8%.

#### val event types

| label | count | share |
| --- | --- | --- |
| Sport.Compete | 166 | 12.43% |
| Contact.Communicate | 137 | 10.26% |
| Life.Die | 79 | 5.92% |
| Contact.Meet | 72 | 5.39% |
| Personnel.EndPosition | 66 | 4.94% |
| Conflict.Attack | 63 | 4.72% |
| Personnel.StartPosition | 59 | 4.42% |
| Business.LaunchProduct | 51 | 3.82% |
| Award.ReceiveAward | 49 | 3.67% |
| Justice.Arrest | 45 | 3.37% |
| Life.Injure | 44 | 3.30% |
| Justice.ChargeIndict | 40 | 3.00% |
| Disaster.Accident | 31 | 2.32% |
| Conflict.Demonstrate | 28 | 2.10% |
| Justice.TrialHearing | 26 | 1.95% |
| Business.StartOrg | 25 | 1.87% |
| Tech.Release | 20 | 1.50% |
| Transaction.TransferOwnership | 20 | 1.50% |
| Justice.Sentence | 19 | 1.42% |
| Government.Protest | 19 | 1.42% |
| Research.Publish | 18 | 1.35% |
| Movement.Arrive | 18 | 1.35% |
| Justice.Sue | 17 | 1.27% |
| Government.Vote | 16 | 1.20% |
| Transaction.Donate | 15 | 1.12% |
| Medical.Treat | 14 | 1.05% |
| Movement.Transport | 13 | 0.97% |
| Life.Evacuate | 13 | 0.97% |
| Disaster.NaturalDisaster | 11 | 0.82% |
| Government.Sanction | 11 | 0.82% |
| Life.BeBorn | 11 | 0.82% |
| Movement.Migrate | 11 | 0.82% |
| Contact.Negotiate | 11 | 0.82% |
| Medical.Diagnose | 8 | 0.60% |
| Research.Discover | 8 | 0.60% |
| Life.Marry | 8 | 0.60% |
| Government.EnactLaw | 8 | 0.60% |
| Disaster.Rescue | 8 | 0.60% |
| Personnel.Nominate | 7 | 0.52% |
| Business.DeclareBankruptcy | 5 | 0.37% |
| Business.EndOrg | 5 | 0.37% |
| Business.MergeOrg | 5 | 0.37% |
| Life.Divorce | 4 | 0.30% |
| Research.Fund | 4 | 0.30% |
| Personnel.Elect | 4 | 0.30% |
| Justice.Fine | 4 | 0.30% |
| Tech.Breach | 4 | 0.30% |
| Justice.Appeal | 4 | 0.30% |
| Justice.Acquit | 3 | 0.22% |
| Business.Strike | 2 | 0.15% |
| Conflict.Ceasefire | 2 | 0.15% |
| Disaster.Outbreak | 2 | 0.15% |
| Business.Layoff | 1 | 0.07% |
| Movement.Deport | 1 | 0.07% |

**test.** 1,378 instances in 568 records, 0 triggerless. Trigger gold 1,278. Argument gold 3,308. Arguments per instance, mean 2.46. Trigger length in whitespace tokens, mean 1.29, single-token share 82.7%.

#### test event types

| label | count | share |
| --- | --- | --- |
| Sport.Compete | 185 | 13.43% |
| Contact.Communicate | 110 | 7.98% |
| Life.Die | 80 | 5.81% |
| Contact.Meet | 69 | 5.01% |
| Conflict.Attack | 66 | 4.79% |
| Business.LaunchProduct | 62 | 4.50% |
| Personnel.StartPosition | 61 | 4.43% |
| Life.Injure | 55 | 3.99% |
| Award.ReceiveAward | 49 | 3.56% |
| Justice.Arrest | 48 | 3.48% |
| Justice.ChargeIndict | 40 | 2.90% |
| Disaster.Accident | 35 | 2.54% |
| Justice.TrialHearing | 34 | 2.47% |
| Personnel.EndPosition | 32 | 2.32% |
| Movement.Arrive | 30 | 2.18% |
| Transaction.Donate | 28 | 2.03% |
| Government.Vote | 25 | 1.81% |
| Research.Publish | 24 | 1.74% |
| Disaster.NaturalDisaster | 22 | 1.60% |
| Justice.Sentence | 21 | 1.52% |
| Tech.Release | 21 | 1.52% |
| Medical.Treat | 20 | 1.45% |
| Transaction.TransferOwnership | 18 | 1.31% |
| Life.BeBorn | 17 | 1.23% |
| Business.StartOrg | 16 | 1.16% |
| Justice.Sue | 15 | 1.09% |
| Government.Protest | 15 | 1.09% |
| Government.EnactLaw | 15 | 1.09% |
| Medical.Diagnose | 15 | 1.09% |
| Justice.Fine | 13 | 0.94% |
| Personnel.Elect | 13 | 0.94% |
| Movement.Transport | 11 | 0.80% |
| Conflict.Demonstrate | 11 | 0.80% |
| Government.Sanction | 10 | 0.73% |
| Movement.Migrate | 9 | 0.65% |
| Disaster.Rescue | 9 | 0.65% |
| Research.Discover | 9 | 0.65% |
| Life.Marry | 9 | 0.65% |
| Business.MergeOrg | 8 | 0.58% |
| Life.Evacuate | 8 | 0.58% |
| Contact.Negotiate | 5 | 0.36% |
| Personnel.Nominate | 5 | 0.36% |
| Tech.Breach | 4 | 0.29% |
| Business.EndOrg | 4 | 0.29% |
| Research.Fund | 4 | 0.29% |
| Business.Strike | 4 | 0.29% |
| Disaster.Outbreak | 4 | 0.29% |
| Justice.Appeal | 3 | 0.22% |
| Transaction.TransferMoney | 2 | 0.15% |
| Life.Divorce | 2 | 0.15% |
| Business.DeclareBankruptcy | 1 | 0.07% |
| Justice.Acquit | 1 | 0.07% |
| Business.Layoff | 1 | 0.07% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 27,852 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 3,260 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 3,392 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.90 | 36.78% | 2.53 | 41.18% | 0.14% | 0 | 27,852 | 0 | 11,365 |
| val | 1.82 | 36.26% | 2.69 | 42.11% | 0.15% | 0 | 3,260 | 0 | 1,335 |
| test | 1.79 | 35.80% | 2.34 | 40.09% | 0.02% | 0 | 3,392 | 0 | 1,378 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 11,365 | 0.00% | 6,327 | 37.90% | 37.16% | 3.94% | 1.12% | 17.57% |
| val | 1,335 | 0.00% | 627 | 55.18% | 54.55% | 5.74% | 3.67% | 16.57% |
| test | 1,378 | 0.00% | 528 | 51.33% | 50.76% | 4.55% | 2.46% | 16.02% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 0 | 0 | 6,327 |
| val | 0 | 0 | 627 |
| test | 0 | 0 | 528 |

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
| person | 29,421 | 21.54% |
| location | 17,502 | 12.81% |
| date | 12,955 | 9.49% |
| cardinal | 8,991 | 6.58% |
| organization | 7,830 | 5.73% |
| job title | 5,267 | 3.86% |
| sports team | 4,044 | 2.96% |
| money | 3,991 | 2.92% |
| geopolitical entity | 3,793 | 2.78% |
| company | 3,722 | 2.73% |
| media outlet | 3,683 | 2.70% |
| product | 2,518 | 1.84% |
| age | 2,324 | 1.70% |
| facility | 2,183 | 1.60% |
| work of art | 1,950 | 1.43% |
| percent | 1,861 | 1.36% |
| time | 1,682 | 1.23% |
| occupation | 1,654 | 1.21% |
| duration | 1,464 | 1.07% |
| government agency | 1,325 | 0.97% |
| educational institution | 1,151 | 0.84% |
| nationality | 918 | 0.67% |
| event | 736 | 0.54% |
| quantity | 720 | 0.53% |
| family relationship | 718 | 0.53% |
| address | 702 | 0.51% |
| ordinal | 621 | 0.45% |
| crime | 593 | 0.43% |
| brand | 549 | 0.40% |
| disease | 538 | 0.39% |
| sports competition | 510 | 0.37% |
| measurement | 481 | 0.35% |
| film | 402 | 0.29% |
| road | 399 | 0.29% |
| region | 392 | 0.29% |
| award | 390 | 0.29% |
| distance | 387 | 0.28% |
| landmark | 376 | 0.28% |
| military rank | 367 | 0.27% |
| political party | 347 | 0.25% |
| ethnic group | 325 | 0.24% |
| military unit | 308 | 0.23% |
| era | 289 | 0.21% |
| song | 284 | 0.21% |
| political affiliation | 280 | 0.21% |
| law | 280 | 0.21% |
| vehicle | 261 | 0.19% |
| non-profit organization | 253 | 0.19% |
| publication | 233 | 0.17% |
| academic degree | 225 | 0.16% |
| temperature | 210 | 0.15% |
| chemical | 204 | 0.15% |
| medication | 186 | 0.14% |
| team role | 181 | 0.13% |
| pseudonym | 175 | 0.13% |
| religious group | 168 | 0.12% |
| weapon | 164 | 0.12% |
| body of water | 162 | 0.12% |
| book | 155 | 0.11% |
| honorific | 153 | 0.11% |
| holiday | 145 | 0.11% |
| software | 136 | 0.10% |
| court case | 120 | 0.09% |
| regulation | 119 | 0.09% |
| regulatory body | 113 | 0.08% |
| stock ticker | 111 | 0.08% |
| anatomical structure | 107 | 0.08% |
| symptom | 106 | 0.08% |
| market index | 101 | 0.07% |
| currency | 94 | 0.07% |
| aircraft | 94 | 0.07% |
| vessel | 91 | 0.07% |
| frequency | 91 | 0.07% |
| species | 90 | 0.07% |
| video game | 83 | 0.06% |
| speed | 80 | 0.06% |
| organism | 71 | 0.05% |
| airport | 64 | 0.05% |
| weight | 64 | 0.05% |
| criminal organization | 55 | 0.04% |
| language | 53 | 0.04% |
| conference | 52 | 0.04% |
| model number | 51 | 0.04% |
| medical procedure | 51 | 0.04% |
| sentence | 47 | 0.03% |
| trade union | 45 | 0.03% |
| operating system | 42 | 0.03% |
| treaty | 38 | 0.03% |
| area | 32 | 0.02% |
| postal code | 27 | 0.02% |
| hardware component | 27 | 0.02% |
| mountain | 25 | 0.02% |
| natural disaster | 25 | 0.02% |
| witness | 20 | 0.01% |
| interest rate | 18 | 0.01% |
| financial instrument | 18 | 0.01% |
| allergen | 16 | 0.01% |
| fiscal quarter | 15 | 0.01% |
| deadline | 15 | 0.01% |
| protein | 13 | 0.01% |
| border | 10 | 0.01% |
| war | 9 | 0.01% |
| diagnosis | 6 | 0.00% |
| vaccine | 6 | 0.00% |
| contract | 5 | 0.00% |
| pathogen | 5 | 0.00% |
| gene | 4 | 0.00% |
| framework | 4 | 0.00% |
| license | 3 | 0.00% |
| programming language | 2 | 0.00% |
| security vulnerability | 2 | 0.00% |
| credit rating | 2 | 0.00% |
| algorithm | 1 | 0.00% |
| mineral | 1 | 0.00% |
| protocol | 1 | 0.00% |
| patent | 1 | 0.00% |
| clinical trial | 1 | 0.00% |
| api | 1 | 0.00% |
| database | 0 | 0.00% |
| dosage | 0 | 0.00% |
| file format | 0 | 0.00% |
| tax | 0 | 0.00% |
| budget line | 0 | 0.00% |
| voltage | 0 | 0.00% |
| verdict | 0 | 0.00% |

#### val

| label | count | share |
| --- | --- | --- |
| person | 3,326 | 20.90% |
| location | 1,921 | 12.07% |
| date | 1,485 | 9.33% |
| cardinal | 1,041 | 6.54% |
| organization | 901 | 5.66% |
| job title | 659 | 4.14% |
| company | 509 | 3.20% |
| money | 501 | 3.15% |
| media outlet | 469 | 2.95% |
| geopolitical entity | 442 | 2.78% |
| sports team | 441 | 2.77% |
| product | 345 | 2.17% |
| age | 267 | 1.68% |
| percent | 243 | 1.53% |
| facility | 240 | 1.51% |
| time | 195 | 1.23% |
| duration | 194 | 1.22% |
| work of art | 177 | 1.11% |
| occupation | 165 | 1.04% |
| educational institution | 157 | 0.99% |
| government agency | 156 | 0.98% |
| nationality | 106 | 0.67% |
| temperature | 106 | 0.67% |
| quantity | 96 | 0.60% |
| brand | 92 | 0.58% |
| ordinal | 90 | 0.57% |
| address | 80 | 0.50% |
| event | 78 | 0.49% |
| measurement | 76 | 0.48% |
| family relationship | 64 | 0.40% |
| crime | 61 | 0.38% |
| film | 53 | 0.33% |
| song | 51 | 0.32% |
| sports competition | 49 | 0.31% |
| region | 47 | 0.30% |
| political party | 47 | 0.30% |
| road | 45 | 0.28% |
| landmark | 42 | 0.26% |
| era | 40 | 0.25% |
| distance | 40 | 0.25% |
| disease | 37 | 0.23% |
| ethnic group | 36 | 0.23% |
| award | 34 | 0.21% |
| anatomical structure | 33 | 0.21% |
| political affiliation | 33 | 0.21% |
| non-profit organization | 32 | 0.20% |
| publication | 31 | 0.19% |
| military unit | 30 | 0.19% |
| team role | 30 | 0.19% |
| military rank | 30 | 0.19% |
| academic degree | 27 | 0.17% |
| law | 24 | 0.15% |
| book | 21 | 0.13% |
| religious group | 21 | 0.13% |
| chemical | 19 | 0.12% |
| holiday | 19 | 0.12% |
| vehicle | 19 | 0.12% |
| currency | 19 | 0.12% |
| organism | 18 | 0.11% |
| court case | 17 | 0.11% |
| medication | 16 | 0.10% |
| body of water | 16 | 0.10% |
| pseudonym | 16 | 0.10% |
| weapon | 14 | 0.09% |
| honorific | 12 | 0.08% |
| medical procedure | 12 | 0.08% |
| airport | 12 | 0.08% |
| weight | 11 | 0.07% |
| video game | 11 | 0.07% |
| language | 10 | 0.06% |
| regulatory body | 10 | 0.06% |
| area | 9 | 0.06% |
| sentence | 9 | 0.06% |
| software | 9 | 0.06% |
| conference | 8 | 0.05% |
| aircraft | 8 | 0.05% |
| criminal organization | 8 | 0.05% |
| regulation | 7 | 0.04% |
| treaty | 7 | 0.04% |
| species | 6 | 0.04% |
| stock ticker | 6 | 0.04% |
| model number | 6 | 0.04% |
| interest rate | 5 | 0.03% |
| witness | 5 | 0.03% |
| postal code | 5 | 0.03% |
| trade union | 4 | 0.03% |
| speed | 4 | 0.03% |
| symptom | 4 | 0.03% |
| market index | 3 | 0.02% |
| protocol | 3 | 0.02% |
| frequency | 3 | 0.02% |
| vaccine | 3 | 0.02% |
| war | 3 | 0.02% |
| financial instrument | 3 | 0.02% |
| patent | 2 | 0.01% |
| natural disaster | 2 | 0.01% |
| vessel | 2 | 0.01% |
| dosage | 2 | 0.01% |
| programming language | 2 | 0.01% |
| clinical trial | 1 | 0.01% |
| fiscal quarter | 1 | 0.01% |
| diagnosis | 1 | 0.01% |
| protein | 1 | 0.01% |
| pathogen | 1 | 0.01% |
| mountain | 1 | 0.01% |
| operating system | 1 | 0.01% |
| verdict | 0 | 0.00% |
| contract | 0 | 0.00% |
| mineral | 0 | 0.00% |
| security vulnerability | 0 | 0.00% |
| database | 0 | 0.00% |
| budget line | 0 | 0.00% |
| tax | 0 | 0.00% |
| algorithm | 0 | 0.00% |
| framework | 0 | 0.00% |
| file format | 0 | 0.00% |
| gene | 0 | 0.00% |
| credit rating | 0 | 0.00% |
| allergen | 0 | 0.00% |
| border | 0 | 0.00% |
| license | 0 | 0.00% |
| hardware component | 0 | 0.00% |
| deadline | 0 | 0.00% |
| voltage | 0 | 0.00% |
| api | 0 | 0.00% |

#### test

| label | count | share |
| --- | --- | --- |
| person | 3,898 | 20.35% |
| location | 2,375 | 12.40% |
| date | 1,761 | 9.19% |
| cardinal | 1,439 | 7.51% |
| organization | 1,092 | 5.70% |
| job title | 782 | 4.08% |
| money | 651 | 3.40% |
| company | 630 | 3.29% |
| sports team | 585 | 3.05% |
| media outlet | 462 | 2.41% |
| geopolitical entity | 427 | 2.23% |
| product | 381 | 1.99% |
| percent | 379 | 1.98% |
| age | 324 | 1.69% |
| facility | 297 | 1.55% |
| work of art | 296 | 1.55% |
| occupation | 234 | 1.22% |
| time | 222 | 1.16% |
| duration | 191 | 1.00% |
| government agency | 191 | 1.00% |
| educational institution | 151 | 0.79% |
| nationality | 112 | 0.58% |
| brand | 107 | 0.56% |
| ordinal | 107 | 0.56% |
| quantity | 99 | 0.52% |
| family relationship | 87 | 0.45% |
| measurement | 86 | 0.45% |
| address | 80 | 0.42% |
| event | 80 | 0.42% |
| sports competition | 80 | 0.42% |
| distance | 75 | 0.39% |
| region | 72 | 0.38% |
| era | 60 | 0.31% |
| road | 57 | 0.30% |
| award | 53 | 0.28% |
| landmark | 52 | 0.27% |
| disease | 51 | 0.27% |
| military rank | 51 | 0.27% |
| crime | 50 | 0.26% |
| film | 50 | 0.26% |
| non-profit organization | 50 | 0.26% |
| political party | 46 | 0.24% |
| ethnic group | 42 | 0.22% |
| law | 38 | 0.20% |
| publication | 35 | 0.18% |
| vehicle | 35 | 0.18% |
| military unit | 32 | 0.17% |
| chemical | 31 | 0.16% |
| weight | 31 | 0.16% |
| religious group | 30 | 0.16% |
| temperature | 30 | 0.16% |
| medication | 28 | 0.15% |
| anatomical structure | 26 | 0.14% |
| regulation | 26 | 0.14% |
| pseudonym | 25 | 0.13% |
| holiday | 25 | 0.13% |
| political affiliation | 25 | 0.13% |
| software | 24 | 0.13% |
| team role | 23 | 0.12% |
| body of water | 23 | 0.12% |
| weapon | 20 | 0.10% |
| academic degree | 20 | 0.10% |
| song | 19 | 0.10% |
| honorific | 18 | 0.09% |
| currency | 17 | 0.09% |
| market index | 15 | 0.08% |
| airport | 15 | 0.08% |
| symptom | 15 | 0.08% |
| book | 14 | 0.07% |
| vessel | 14 | 0.07% |
| regulatory body | 13 | 0.07% |
| stock ticker | 13 | 0.07% |
| speed | 12 | 0.06% |
| aircraft | 12 | 0.06% |
| organism | 11 | 0.06% |
| species | 11 | 0.06% |
| video game | 10 | 0.05% |
| frequency | 9 | 0.05% |
| court case | 9 | 0.05% |
| conference | 9 | 0.05% |
| language | 7 | 0.04% |
| treaty | 7 | 0.04% |
| operating system | 7 | 0.04% |
| postal code | 7 | 0.04% |
| criminal organization | 6 | 0.03% |
| medical procedure | 4 | 0.02% |
| diagnosis | 4 | 0.02% |
| contract | 4 | 0.02% |
| financial instrument | 4 | 0.02% |
| file format | 3 | 0.02% |
| war | 3 | 0.02% |
| mountain | 2 | 0.01% |
| algorithm | 2 | 0.01% |
| area | 2 | 0.01% |
| deadline | 2 | 0.01% |
| trade union | 2 | 0.01% |
| natural disaster | 2 | 0.01% |
| sentence | 2 | 0.01% |
| model number | 1 | 0.01% |
| clinical trial | 1 | 0.01% |
| protein | 1 | 0.01% |
| hardware component | 1 | 0.01% |
| witness | 1 | 0.01% |
| border | 0 | 0.00% |
| patent | 0 | 0.00% |
| interest rate | 0 | 0.00% |
| api | 0 | 0.00% |
| framework | 0 | 0.00% |
| verdict | 0 | 0.00% |
| credit rating | 0 | 0.00% |
| tax | 0 | 0.00% |
| database | 0 | 0.00% |
| security vulnerability | 0 | 0.00% |
| voltage | 0 | 0.00% |
| budget line | 0 | 0.00% |
| protocol | 0 | 0.00% |
| allergen | 0 | 0.00% |
| pathogen | 0 | 0.00% |
| vaccine | 0 | 0.00% |
| fiscal quarter | 0 | 0.00% |
| programming language | 0 | 0.00% |
| mineral | 0 | 0.00% |
| gene | 0 | 0.00% |
| dosage | 0 | 0.00% |
| license | 0 | 0.00% |

### Relations

None.

### Classifications

None.

### Structures

None.

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | llm_real |
| relations | llm_real |
| events | llm_real |
| classifications | llm_real |
| structures | llm_real |
