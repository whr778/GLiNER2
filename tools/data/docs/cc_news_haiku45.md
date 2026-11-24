# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cc_news_haiku45 | train | 15,839 | 15,839 | 350,042 | 18,359 | 49,774 | 1,279 | 3,854 | 10,148 |
| cc_news_haiku45 | val | 2,075 | 2,075 | 45,639 | 2,546 | 6,487 | 164 | 484 | 1,236 |
| cc_news_haiku45 | test | 2,043 | 2,043 | 44,588 | 2,377 | 6,432 | 210 | 492 | 1,259 |

## cc_news_haiku45

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 15,839 | 15,839 | 408 | 1022 | 3715 | 350,042 | 18,359 | 49,774 | 1,279 | 3,854 | 10,148 |
| val | 2,075 | 2,075 | 406 | 1033 | 4892 | 45,639 | 2,546 | 6,487 | 164 | 484 | 1,236 |
| test | 2,043 | 2,043 | 393 | 1007 | 3717 | 44,588 | 2,377 | 6,432 | 210 | 492 | 1,259 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 15,745 | 2,063 | 2,027 | 19,835 | 99.39% |
| und | 94 | 12 | 16 | 122 | 0.61% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 2,043 |
| test_unique_inputs | 2,043 |
| train_records | 15,839 |
| train_unique_inputs | 15,839 |
| val_records | 2,075 |
| val_unique_inputs | 2,075 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0287 | 350,042 / 45,639 | location 14%→14%, sports team 4%→4%, person 23%→23%, date 9%→9%, address 1%→1% |
| entities (per mention) | train / test | 0.0257 | 350,042 / 44,588 | cardinal 8%→8%, person 23%→23%, company 2%→3%, work of art 1%→1%, film 0%→0% |
| entities (per mention) | val / test | 0.0356 | 45,639 / 44,588 | person 23%→23%, location 14%→14%, sports team 4%→4%, address 1%→0%, company 2%→3% |
| relations (per instance) | train / val | 0.1287 | 18,359 / 2,546 | participated_in 2%→1%, manufactured_by 1%→2%, operates_in 3%→4%, based_in 8%→7%, works_for 13%→12% |
| relations (per instance) | train / test | 0.1374 | 18,359 / 2,377 | works_for 13%→19%, member_of 12%→8%, based_in 8%→7%, affiliated_with 8%→10%, produced_by 2%→0% |
| relations (per instance) | val / test | 0.2020 WARN | 2,546 / 2,377 | works_for 12%→19%, member_of 11%→8%, affiliated_with 7%→10%, located_in 11%→9%, operates_in 4%→3% |
| event types (per trigger mention) | train / val | 0.1577 WARN | 4,127 / 504 | Sport.Compete 12%→8%, Contact.Communicate 7%→4%, Movement.Transport 2%→4%, Justice.TrialHearing 3%→1%, Conflict.Attack 6%→4% |
| event types (per trigger mention) | train / test | 0.1646 WARN | 4,127 / 516 | Sport.Compete 12%→7%, Life.Die 5%→7%, Life.Injure 5%→6%, Government.Protest 1%→3%, Contact.Negotiate 1%→2% |
| event types (per trigger mention) | val / test | 0.2205 WARN | 504 / 516 | Movement.Transport 4%→1%, Justice.TrialHearing 1%→4%, Life.Die 5%→7%, Justice.Sue 4%→1%, Contact.Communicate 4%→6% |
| event roles (per argument) | train / val | 0.0988 | 10,299 / 1,246 | Competitor 9%→6%, Time 17%→19%, Result 2%→1%, Communicator 3%→1%, Origin 1%→2% |
| event roles (per argument) | train / test | 0.1009 | 10,299 / 1,271 | Place 16%→21%, Competitor 9%→6%, Competition 1%→0%, Victim 5%→5%, Destination 1%→1% |
| event roles (per argument) | val / test | 0.1320 | 1,246 / 1,271 | Place 17%→21%, Time 19%→17%, Destination 2%→1%, Origin 2%→0%, Result 1%→2% |
| classifications (true_label) | train / val | 0.0265 | 49,774 / 6,487 | temporal_orientation: past 6%→6%, urgency: routine 6%→6%, actionability: informational 7%→8%, subjectivity: subjective 1%→1%, subjectivity: objective 7%→7% |
| classifications (true_label) | train / test | 0.0300 | 49,774 / 6,432 | language_register: plain 7%→6%, urgency: routine 6%→6%, document_genre: news report 7%→6%, topic: legal 2%→2%, sentiment: positive 2%→2% |
| classifications (true_label) | val / test | 0.0283 | 6,487 / 6,432 | subjectivity: objective 7%→7%, language_register: plain 7%→6%, topic: legal 2%→2%, certainty: asserted 8%→8%, topic: health 0%→1% |

Thin support in val, entities (per mention): 22 label(s) under 10 gold. voltage ×1, contract ×1, verdict ×1, model number ×1, protocol ×1, war ×2, financial instrument ×2, fiscal quarter ×2, gene ×2, framework ×2, mountain ×3, video game ×3, pathogen ×3, dosage ×3, license ×4, programming language ×4, hardware component ×5, area ×7, postal code ×7, conference ×8, operating system ×8, symptom ×9.
Thin support in val, relations (per instance): 30 label(s) under 10 gold. convicted_of ×1, successor_of ×1, borders ×1, sold_to ×1, causes ×1, predecessor_of ×1, derived_from ×1, died_in ×2, won ×2, allied_with ×2, party_to ×2, sued_by ×3, merged_with ×3, distributor_of ×3, invested_in ×3, sibling_of ×4, testified_against ×4, adjacent_to ×5, subsidiary_of ×5, sponsored_by ×6, published_by ×6, directed_by ×6, composed_by ×6, mentor_of ×6, named_after ×6, owns ×7, supplier_of ×7, spouse_of ×9, interacts_with ×9, advisor_to ×9.
Thin support in val, event types (per trigger mention): 34 label(s) under 10 gold. Business.MergeOrg ×1, Justice.Acquit ×1, Tech.Breach ×1, Movement.Migrate ×1, Business.Strike ×1, Research.Fund ×2, Life.Marry ×2, Justice.Appeal ×3, Conflict.Ceasefire ×3, Medical.Treat ×3, Government.Sanction ×3, Transaction.Donate ×4, Justice.Fine ×4, Personnel.Elect ×4, Business.EndOrg ×4, Business.StartOrg ×4, Research.Discover ×4, Life.BeBorn ×4, Medical.Diagnose ×4, Government.EnactLaw ×5, Research.Publish ×5, Movement.Deport ×5, Justice.TrialHearing ×6, Personnel.Nominate ×6, Movement.Arrive ×6, Disaster.Outbreak ×6, Contact.Negotiate ×7, Justice.Sentence ×8, Government.Vote ×8, Life.Evacuate ×8, Transaction.TransferOwnership ×8, Government.Protest ×9, Award.ReceiveAward ×9, Business.LaunchProduct ×9.
Thin support in val, event roles (per argument): 32 label(s) under 10 gold. Prosecutor ×1, Funder ×1, Migrant ×1, Reason ×2, Voter ×2, Diagnosis ×2, Appellant ×3, Party ×3, Law ×3, Rescuer ×3, Proposal ×3, Affected ×3, Disease ×3, Price ×3, Seller ×3, Donor ×4, Money ×4, Topic ×4, Author ×4, Researcher ×4, Finding ×4, Vehicle ×5, Plaintiff ×5, Sentence ×5, Publication ×5, Traveler ×6, Buyer ×6, Patient ×7, Evacuee ×7, Award ×8, Adjudicator ×9, Attacker ×9.
Thin support in val, classifications (true_label): 13 label(s) under 10 gold. document_genre: correspondence ×1, document_genre: advertisement ×1, topic: energy ×1, certainty: speculative ×1, document_genre: review ×4, language_register: legal ×5, document_genre: technical documentation ×5, actionability: requires action ×5, temporal_orientation: timeless ×6, topic: agriculture ×6, language_register: colloquial ×8, urgency: urgent ×8, topic: defense ×9.
Thin support in test, entities (per mention): 18 label(s) under 10 gold. pathogen ×1, programming language ×1, dosage ×2, framework ×2, model number ×3, verdict ×3, allergen ×3, hardware component ×4, deadline ×4, operating system ×4, financial instrument ×4, mountain ×4, trade union ×5, border ×6, natural disaster ×7, war ×8, organism ×8, aircraft ×9.
Thin support in test, relations (per instance): 29 label(s) under 10 gold. depends_on ×1, treats ×1, predecessor_of ×1, allied_with ×1, won ×1, causes ×1, adjacent_to ×2, capital_of ×2, derived_from ×2, successor_of ×2, distributor_of ×2, sold_to ×2, signatory_of ×2, borders ×3, party_to ×3, regulated_by ×3, died_in ×4, charged_with ×4, spouse_of ×5, mentor_of ×5, customer_of ×5, advisor_to ×6, convicted_of ×6, subsidiary_of ×6, acquired ×7, sibling_of ×8, supplier_of ×8, composed_by ×8, produced_by ×9.
Thin support in test, event types (per trigger mention): 34 label(s) under 10 gold. Life.Divorce ×1, Movement.Migrate ×1, Life.Marry ×1, Justice.Acquit ×1, Research.Discover ×1, Conflict.Ceasefire ×1, Movement.Deport ×1, Business.Layoff ×1, Research.Publish ×2, Research.Fund ×2, Business.Strike ×2, Tech.Breach ×2, Government.EnactLaw ×3, Transaction.TransferMoney ×3, Business.MergeOrg ×3, Personnel.Nominate ×3, Justice.Fine ×4, Business.EndOrg ×4, Life.Evacuate ×4, Transaction.Donate ×4, Tech.Release ×5, Personnel.Elect ×5, Justice.Appeal ×5, Movement.Arrive ×5, Justice.Sue ×6, Disaster.Rescue ×6, Movement.Transport ×6, Life.BeBorn ×7, Medical.Treat ×7, Medical.Diagnose ×7, Government.Sanction ×7, Award.ReceiveAward ×7, Government.Vote ×8, Disaster.Outbreak ×9.
Thin support in test, event roles (per argument): 38 label(s) under 10 gold. Prosecutor ×1, Disease ×1, Migrant ×1, Giver ×1, Author ×1, Venue ×1, Striker ×1, Researcher ×1, Finding ×1, Employee ×1, Plaintiff ×2, Law ×2, Publication ×2, Funder ×2, Party ×2, Appellant ×2, Topic ×3, Physician ×3, Treatment ×3, Competition ×3, Rescuer ×3, Traveler ×3, Evacuee ×3, Donor ×3, Artifact ×3, Reason ×4, Award ×4, Sentence ×5, Instrument ×5, Affected ×5, Diagnosis ×5, Voter ×5, Vehicle ×5, Origin ×5, Money ×6, Proposal ×7, Destination ×7, Entity ×8.
Thin support in test, classifications (true_label): 12 label(s) under 10 gold. certainty: speculative ×1, document_genre: advertisement ×1, urgency: urgent ×2, document_genre: correspondence ×2, audience: internal ×5, document_genre: technical documentation ×5, actionability: requires action ×6, topic: agriculture ×7, certainty: hedged ×7, language_register: legal ×7, topic: energy ×8, document_genre: review ×8.

### Events

**train.** 4,127 instances in 2,912 records, 0 triggerless. Trigger gold 3,854. Argument gold 10,148. Arguments per instance, mean 2.50. Trigger length in whitespace tokens, mean 1.44, single-token share 79.4%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Sport.Compete | 494 | 11.97% |
| Contact.Communicate | 274 | 6.64% |
| Conflict.Attack | 243 | 5.89% |
| Justice.Arrest | 212 | 5.14% |
| Contact.Meet | 207 | 5.02% |
| Life.Die | 201 | 4.87% |
| Life.Injure | 188 | 4.56% |
| Disaster.Accident | 151 | 3.66% |
| Justice.ChargeIndict | 140 | 3.39% |
| Justice.TrialHearing | 112 | 2.71% |
| Business.LaunchProduct | 110 | 2.67% |
| Personnel.StartPosition | 109 | 2.64% |
| Award.ReceiveAward | 106 | 2.57% |
| Justice.Sue | 88 | 2.13% |
| Personnel.EndPosition | 85 | 2.06% |
| Disaster.NaturalDisaster | 85 | 2.06% |
| Justice.Sentence | 77 | 1.87% |
| Conflict.Demonstrate | 77 | 1.87% |
| Movement.Transport | 75 | 1.82% |
| Movement.Arrive | 62 | 1.50% |
| Disaster.Rescue | 62 | 1.50% |
| Government.Protest | 59 | 1.43% |
| Government.Vote | 57 | 1.38% |
| Tech.Release | 57 | 1.38% |
| Disaster.Outbreak | 55 | 1.33% |
| Business.StartOrg | 53 | 1.28% |
| Medical.Treat | 47 | 1.14% |
| Life.Evacuate | 45 | 1.09% |
| Life.BeBorn | 42 | 1.02% |
| Government.Sanction | 40 | 0.97% |
| Medical.Diagnose | 35 | 0.85% |
| Contact.Negotiate | 35 | 0.85% |
| Research.Discover | 35 | 0.85% |
| Transaction.TransferOwnership | 34 | 0.82% |
| Research.Publish | 34 | 0.82% |
| Government.EnactLaw | 31 | 0.75% |
| Transaction.Donate | 31 | 0.75% |
| Personnel.Elect | 31 | 0.75% |
| Justice.Fine | 29 | 0.70% |
| Personnel.Nominate | 26 | 0.63% |
| Business.EndOrg | 26 | 0.63% |
| Movement.Migrate | 22 | 0.53% |
| Tech.Breach | 17 | 0.41% |
| Life.Marry | 17 | 0.41% |
| Justice.Appeal | 16 | 0.39% |
| Business.MergeOrg | 16 | 0.39% |
| Research.Fund | 15 | 0.36% |
| Conflict.Ceasefire | 12 | 0.29% |
| Business.Strike | 11 | 0.27% |
| Transaction.TransferMoney | 11 | 0.27% |
| Justice.Acquit | 10 | 0.24% |
| Movement.Deport | 9 | 0.22% |
| Business.Layoff | 7 | 0.17% |
| Transaction.Lend | 2 | 0.05% |
| Business.DeclareBankruptcy | 1 | 0.02% |
| Life.Divorce | 1 | 0.02% |

**val.** 504 instances in 376 records, 0 triggerless. Trigger gold 484. Argument gold 1,236. Arguments per instance, mean 2.47. Trigger length in whitespace tokens, mean 1.29, single-token share 82.1%.

#### val event types

| label | count | share |
| --- | --- | --- |
| Sport.Compete | 42 | 8.33% |
| Contact.Meet | 28 | 5.56% |
| Life.Injure | 27 | 5.36% |
| Life.Die | 23 | 4.56% |
| Justice.Arrest | 23 | 4.56% |
| Conflict.Attack | 22 | 4.37% |
| Movement.Transport | 21 | 4.17% |
| Justice.ChargeIndict | 20 | 3.97% |
| Contact.Communicate | 20 | 3.97% |
| Disaster.Accident | 19 | 3.77% |
| Justice.Sue | 18 | 3.57% |
| Disaster.NaturalDisaster | 17 | 3.37% |
| Personnel.EndPosition | 15 | 2.98% |
| Personnel.StartPosition | 14 | 2.78% |
| Conflict.Demonstrate | 13 | 2.58% |
| Tech.Release | 13 | 2.58% |
| Disaster.Rescue | 11 | 2.18% |
| Government.Protest | 9 | 1.79% |
| Award.ReceiveAward | 9 | 1.79% |
| Business.LaunchProduct | 9 | 1.79% |
| Justice.Sentence | 8 | 1.59% |
| Government.Vote | 8 | 1.59% |
| Life.Evacuate | 8 | 1.59% |
| Transaction.TransferOwnership | 8 | 1.59% |
| Contact.Negotiate | 7 | 1.39% |
| Justice.TrialHearing | 6 | 1.19% |
| Personnel.Nominate | 6 | 1.19% |
| Movement.Arrive | 6 | 1.19% |
| Disaster.Outbreak | 6 | 1.19% |
| Government.EnactLaw | 5 | 0.99% |
| Research.Publish | 5 | 0.99% |
| Movement.Deport | 5 | 0.99% |
| Transaction.Donate | 4 | 0.79% |
| Justice.Fine | 4 | 0.79% |
| Personnel.Elect | 4 | 0.79% |
| Business.EndOrg | 4 | 0.79% |
| Business.StartOrg | 4 | 0.79% |
| Research.Discover | 4 | 0.79% |
| Life.BeBorn | 4 | 0.79% |
| Medical.Diagnose | 4 | 0.79% |
| Justice.Appeal | 3 | 0.60% |
| Conflict.Ceasefire | 3 | 0.60% |
| Medical.Treat | 3 | 0.60% |
| Government.Sanction | 3 | 0.60% |
| Research.Fund | 2 | 0.40% |
| Life.Marry | 2 | 0.40% |
| Business.MergeOrg | 1 | 0.20% |
| Justice.Acquit | 1 | 0.20% |
| Tech.Breach | 1 | 0.20% |
| Movement.Migrate | 1 | 0.20% |
| Business.Strike | 1 | 0.20% |

**test.** 516 instances in 376 records, 0 triggerless. Trigger gold 492. Argument gold 1,259. Arguments per instance, mean 2.46. Trigger length in whitespace tokens, mean 1.40, single-token share 80.2%.

#### test event types

| label | count | share |
| --- | --- | --- |
| Sport.Compete | 38 | 7.36% |
| Life.Die | 36 | 6.98% |
| Life.Injure | 33 | 6.40% |
| Conflict.Attack | 33 | 6.40% |
| Contact.Communicate | 31 | 6.01% |
| Justice.TrialHearing | 20 | 3.88% |
| Justice.Arrest | 20 | 3.88% |
| Contact.Meet | 19 | 3.68% |
| Justice.ChargeIndict | 17 | 3.29% |
| Conflict.Demonstrate | 16 | 3.10% |
| Government.Protest | 16 | 3.10% |
| Business.LaunchProduct | 15 | 2.91% |
| Personnel.StartPosition | 14 | 2.71% |
| Justice.Sentence | 13 | 2.52% |
| Disaster.Accident | 13 | 2.52% |
| Disaster.NaturalDisaster | 13 | 2.52% |
| Contact.Negotiate | 12 | 2.33% |
| Business.StartOrg | 12 | 2.33% |
| Personnel.EndPosition | 11 | 2.13% |
| Disaster.Outbreak | 9 | 1.74% |
| Government.Vote | 8 | 1.55% |
| Life.BeBorn | 7 | 1.36% |
| Medical.Treat | 7 | 1.36% |
| Medical.Diagnose | 7 | 1.36% |
| Government.Sanction | 7 | 1.36% |
| Award.ReceiveAward | 7 | 1.36% |
| Justice.Sue | 6 | 1.16% |
| Disaster.Rescue | 6 | 1.16% |
| Movement.Transport | 6 | 1.16% |
| Tech.Release | 5 | 0.97% |
| Personnel.Elect | 5 | 0.97% |
| Justice.Appeal | 5 | 0.97% |
| Movement.Arrive | 5 | 0.97% |
| Justice.Fine | 4 | 0.78% |
| Business.EndOrg | 4 | 0.78% |
| Life.Evacuate | 4 | 0.78% |
| Transaction.Donate | 4 | 0.78% |
| Government.EnactLaw | 3 | 0.58% |
| Transaction.TransferMoney | 3 | 0.58% |
| Business.MergeOrg | 3 | 0.58% |
| Personnel.Nominate | 3 | 0.58% |
| Research.Publish | 2 | 0.39% |
| Research.Fund | 2 | 0.39% |
| Business.Strike | 2 | 0.39% |
| Tech.Breach | 2 | 0.39% |
| Life.Divorce | 1 | 0.19% |
| Movement.Migrate | 1 | 0.19% |
| Life.Marry | 1 | 0.19% |
| Justice.Acquit | 1 | 0.19% |
| Research.Discover | 1 | 0.19% |
| Conflict.Ceasefire | 1 | 0.19% |
| Movement.Deport | 1 | 0.19% |
| Business.Layoff | 1 | 0.19% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 10,299 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 1,246 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 1,271 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.93 | 39.32% | 2.27 | 39.20% | 0.09% | 0 | 10,299 | 0 | 4,127 |
| val | 1.82 | 38.77% | 2.27 | 36.14% | 0.11% | 0 | 1,246 | 0 | 504 |
| test | 1.86 | 39.34% | 2.27 | 37.95% | 0.06% | 0 | 1,271 | 0 | 516 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4,127 | 0.00% | 3,003 | 26.37% | 25.97% | 1.90% | 0.87% | 7.31% |
| val | 504 | 0.00% | 129 | 25.58% | 24.81% | 3.88% | 1.55% | 6.38% |
| test | 516 | 0.00% | 214 | 21.96% | 21.50% | 2.34% | 0.00% | 6.65% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 0 | 0 | 3,003 |
| val | 0 | 0 | 129 |
| test | 0 | 0 | 214 |

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
| person | 80,298 | 22.94% |
| location | 48,257 | 13.79% |
| date | 32,664 | 9.33% |
| cardinal | 28,537 | 8.15% |
| organization | 19,854 | 5.67% |
| sports team | 15,038 | 4.30% |
| job title | 12,427 | 3.55% |
| money | 9,307 | 2.66% |
| company | 8,650 | 2.47% |
| geopolitical entity | 8,440 | 2.41% |
| media outlet | 7,417 | 2.12% |
| age | 5,460 | 1.56% |
| facility | 5,451 | 1.56% |
| product | 4,737 | 1.35% |
| percent | 4,516 | 1.29% |
| time | 4,261 | 1.22% |
| work of art | 4,117 | 1.18% |
| occupation | 3,551 | 1.01% |
| duration | 3,160 | 0.90% |
| government agency | 3,130 | 0.89% |
| educational institution | 2,570 | 0.73% |
| event | 2,134 | 0.61% |
| nationality | 2,132 | 0.61% |
| address | 1,785 | 0.51% |
| measurement | 1,606 | 0.46% |
| crime | 1,592 | 0.45% |
| sports competition | 1,541 | 0.44% |
| ordinal | 1,464 | 0.42% |
| quantity | 1,344 | 0.38% |
| family relationship | 1,255 | 0.36% |
| disease | 1,134 | 0.32% |
| road | 1,008 | 0.29% |
| landmark | 947 | 0.27% |
| brand | 944 | 0.27% |
| distance | 927 | 0.26% |
| region | 910 | 0.26% |
| era | 794 | 0.23% |
| political affiliation | 767 | 0.22% |
| film | 741 | 0.21% |
| military rank | 730 | 0.21% |
| political party | 706 | 0.20% |
| award | 705 | 0.20% |
| law | 694 | 0.20% |
| ethnic group | 673 | 0.19% |
| non-profit organization | 653 | 0.19% |
| team role | 634 | 0.18% |
| song | 576 | 0.16% |
| temperature | 473 | 0.14% |
| publication | 472 | 0.13% |
| vehicle | 447 | 0.13% |
| military unit | 442 | 0.13% |
| chemical | 408 | 0.12% |
| pseudonym | 392 | 0.11% |
| religious group | 369 | 0.11% |
| medication | 358 | 0.10% |
| holiday | 313 | 0.09% |
| academic degree | 310 | 0.09% |
| book | 291 | 0.08% |
| symptom | 265 | 0.08% |
| honorific | 261 | 0.07% |
| anatomical structure | 240 | 0.07% |
| body of water | 237 | 0.07% |
| regulatory body | 234 | 0.07% |
| currency | 223 | 0.06% |
| court case | 221 | 0.06% |
| regulation | 204 | 0.06% |
| software | 198 | 0.06% |
| market index | 197 | 0.06% |
| frequency | 181 | 0.05% |
| weapon | 176 | 0.05% |
| speed | 167 | 0.05% |
| medical procedure | 164 | 0.05% |
| species | 151 | 0.04% |
| stock ticker | 147 | 0.04% |
| weight | 145 | 0.04% |
| sentence | 144 | 0.04% |
| video game | 131 | 0.04% |
| organism | 125 | 0.04% |
| criminal organization | 122 | 0.03% |
| treaty | 117 | 0.03% |
| aircraft | 116 | 0.03% |
| airport | 109 | 0.03% |
| vessel | 106 | 0.03% |
| language | 105 | 0.03% |
| postal code | 91 | 0.03% |
| conference | 90 | 0.03% |
| area | 77 | 0.02% |
| trade union | 72 | 0.02% |
| witness | 71 | 0.02% |
| model number | 69 | 0.02% |
| war | 66 | 0.02% |
| natural disaster | 64 | 0.02% |
| operating system | 61 | 0.02% |
| deadline | 52 | 0.01% |
| mountain | 44 | 0.01% |
| programming language | 36 | 0.01% |
| hardware component | 32 | 0.01% |
| pathogen | 30 | 0.01% |
| vaccine | 23 | 0.01% |
| protocol | 17 | 0.00% |
| financial instrument | 15 | 0.00% |
| border | 14 | 0.00% |
| allergen | 14 | 0.00% |
| protein | 12 | 0.00% |
| dosage | 10 | 0.00% |
| diagnosis | 10 | 0.00% |
| fiscal quarter | 10 | 0.00% |
| contract | 10 | 0.00% |
| license | 7 | 0.00% |
| verdict | 7 | 0.00% |
| file format | 6 | 0.00% |
| clinical trial | 6 | 0.00% |
| interest rate | 5 | 0.00% |
| algorithm | 5 | 0.00% |
| patent | 5 | 0.00% |
| voltage | 3 | 0.00% |
| security vulnerability | 3 | 0.00% |
| database | 2 | 0.00% |
| mineral | 2 | 0.00% |
| api | 1 | 0.00% |
| framework | 1 | 0.00% |
| credit rating | 0 | 0.00% |
| gene | 0 | 0.00% |
| tax | 0 | 0.00% |
| budget line | 0 | 0.00% |

#### val

| label | count | share |
| --- | --- | --- |
| person | 10,670 | 23.38% |
| location | 6,579 | 14.42% |
| date | 4,064 | 8.90% |
| cardinal | 3,625 | 7.94% |
| organization | 2,552 | 5.59% |
| sports team | 1,735 | 3.80% |
| job title | 1,682 | 3.69% |
| money | 1,243 | 2.72% |
| company | 1,119 | 2.45% |
| geopolitical entity | 1,014 | 2.22% |
| media outlet | 946 | 2.07% |
| age | 734 | 1.61% |
| facility | 717 | 1.57% |
| work of art | 634 | 1.39% |
| percent | 570 | 1.25% |
| product | 564 | 1.24% |
| time | 529 | 1.16% |
| occupation | 485 | 1.06% |
| duration | 405 | 0.89% |
| address | 380 | 0.83% |
| government agency | 380 | 0.83% |
| nationality | 353 | 0.77% |
| educational institution | 297 | 0.65% |
| event | 289 | 0.63% |
| crime | 246 | 0.54% |
| sports competition | 210 | 0.46% |
| ordinal | 185 | 0.41% |
| measurement | 177 | 0.39% |
| quantity | 133 | 0.29% |
| family relationship | 131 | 0.29% |
| landmark | 119 | 0.26% |
| disease | 115 | 0.25% |
| road | 114 | 0.25% |
| military rank | 104 | 0.23% |
| political party | 103 | 0.23% |
| region | 102 | 0.22% |
| distance | 98 | 0.21% |
| award | 97 | 0.21% |
| era | 97 | 0.21% |
| political affiliation | 91 | 0.20% |
| temperature | 89 | 0.20% |
| chemical | 86 | 0.19% |
| law | 86 | 0.19% |
| song | 85 | 0.19% |
| book | 84 | 0.18% |
| film | 80 | 0.18% |
| ethnic group | 79 | 0.17% |
| non-profit organization | 77 | 0.17% |
| vehicle | 73 | 0.16% |
| brand | 72 | 0.16% |
| team role | 70 | 0.15% |
| publication | 64 | 0.14% |
| pseudonym | 59 | 0.13% |
| honorific | 56 | 0.12% |
| military unit | 54 | 0.12% |
| religious group | 54 | 0.12% |
| software | 53 | 0.12% |
| academic degree | 48 | 0.11% |
| medication | 48 | 0.11% |
| court case | 39 | 0.09% |
| holiday | 37 | 0.08% |
| body of water | 34 | 0.07% |
| regulatory body | 31 | 0.07% |
| language | 26 | 0.06% |
| species | 25 | 0.05% |
| regulation | 24 | 0.05% |
| currency | 23 | 0.05% |
| frequency | 23 | 0.05% |
| organism | 23 | 0.05% |
| vessel | 22 | 0.05% |
| sentence | 21 | 0.05% |
| stock ticker | 20 | 0.04% |
| speed | 20 | 0.04% |
| medical procedure | 19 | 0.04% |
| anatomical structure | 18 | 0.04% |
| weapon | 17 | 0.04% |
| market index | 17 | 0.04% |
| weight | 17 | 0.04% |
| treaty | 16 | 0.04% |
| airport | 14 | 0.03% |
| natural disaster | 13 | 0.03% |
| trade union | 13 | 0.03% |
| criminal organization | 12 | 0.03% |
| witness | 11 | 0.02% |
| deadline | 10 | 0.02% |
| aircraft | 10 | 0.02% |
| symptom | 9 | 0.02% |
| conference | 8 | 0.02% |
| operating system | 8 | 0.02% |
| area | 7 | 0.02% |
| postal code | 7 | 0.02% |
| hardware component | 5 | 0.01% |
| license | 4 | 0.01% |
| programming language | 4 | 0.01% |
| mountain | 3 | 0.01% |
| video game | 3 | 0.01% |
| pathogen | 3 | 0.01% |
| dosage | 3 | 0.01% |
| war | 2 | 0.00% |
| financial instrument | 2 | 0.00% |
| fiscal quarter | 2 | 0.00% |
| gene | 2 | 0.00% |
| framework | 2 | 0.00% |
| voltage | 1 | 0.00% |
| contract | 1 | 0.00% |
| verdict | 1 | 0.00% |
| model number | 1 | 0.00% |
| protocol | 1 | 0.00% |
| patent | 0 | 0.00% |
| algorithm | 0 | 0.00% |
| api | 0 | 0.00% |
| border | 0 | 0.00% |
| credit rating | 0 | 0.00% |
| database | 0 | 0.00% |
| tax | 0 | 0.00% |
| file format | 0 | 0.00% |
| vaccine | 0 | 0.00% |
| diagnosis | 0 | 0.00% |
| budget line | 0 | 0.00% |
| mineral | 0 | 0.00% |
| security vulnerability | 0 | 0.00% |
| allergen | 0 | 0.00% |
| interest rate | 0 | 0.00% |
| protein | 0 | 0.00% |
| clinical trial | 0 | 0.00% |

#### test

| label | count | share |
| --- | --- | --- |
| person | 10,080 | 22.61% |
| location | 6,178 | 13.86% |
| date | 4,047 | 9.08% |
| cardinal | 3,430 | 7.69% |
| organization | 2,591 | 5.81% |
| sports team | 1,918 | 4.30% |
| job title | 1,604 | 3.60% |
| money | 1,241 | 2.78% |
| company | 1,235 | 2.77% |
| media outlet | 991 | 2.22% |
| geopolitical entity | 980 | 2.20% |
| facility | 714 | 1.60% |
| work of art | 657 | 1.47% |
| age | 646 | 1.45% |
| time | 601 | 1.35% |
| percent | 547 | 1.23% |
| product | 508 | 1.14% |
| occupation | 504 | 1.13% |
| government agency | 398 | 0.89% |
| duration | 386 | 0.87% |
| educational institution | 374 | 0.84% |
| nationality | 316 | 0.71% |
| event | 282 | 0.63% |
| film | 213 | 0.48% |
| crime | 199 | 0.45% |
| ordinal | 198 | 0.44% |
| sports competition | 198 | 0.44% |
| measurement | 196 | 0.44% |
| family relationship | 190 | 0.43% |
| address | 174 | 0.39% |
| disease | 174 | 0.39% |
| quantity | 172 | 0.39% |
| brand | 137 | 0.31% |
| road | 122 | 0.27% |
| distance | 115 | 0.26% |
| landmark | 113 | 0.25% |
| era | 112 | 0.25% |
| region | 108 | 0.24% |
| political affiliation | 107 | 0.24% |
| award | 105 | 0.24% |
| ethnic group | 97 | 0.22% |
| non-profit organization | 96 | 0.22% |
| temperature | 84 | 0.19% |
| law | 73 | 0.16% |
| military rank | 66 | 0.15% |
| religious group | 65 | 0.15% |
| team role | 61 | 0.14% |
| song | 59 | 0.13% |
| publication | 58 | 0.13% |
| academic degree | 54 | 0.12% |
| political party | 53 | 0.12% |
| vehicle | 49 | 0.11% |
| book | 47 | 0.11% |
| chemical | 46 | 0.10% |
| honorific | 46 | 0.10% |
| pseudonym | 46 | 0.10% |
| holiday | 45 | 0.10% |
| military unit | 43 | 0.10% |
| market index | 40 | 0.09% |
| medication | 40 | 0.09% |
| body of water | 37 | 0.08% |
| software | 27 | 0.06% |
| regulatory body | 25 | 0.06% |
| vessel | 24 | 0.05% |
| postal code | 23 | 0.05% |
| symptom | 23 | 0.05% |
| speed | 22 | 0.05% |
| sentence | 22 | 0.05% |
| regulation | 20 | 0.04% |
| medical procedure | 20 | 0.04% |
| court case | 19 | 0.04% |
| frequency | 19 | 0.04% |
| currency | 18 | 0.04% |
| stock ticker | 18 | 0.04% |
| weapon | 18 | 0.04% |
| witness | 17 | 0.04% |
| conference | 17 | 0.04% |
| weight | 16 | 0.04% |
| airport | 14 | 0.03% |
| anatomical structure | 14 | 0.03% |
| criminal organization | 13 | 0.03% |
| language | 13 | 0.03% |
| video game | 11 | 0.02% |
| area | 11 | 0.02% |
| species | 10 | 0.02% |
| treaty | 10 | 0.02% |
| aircraft | 9 | 0.02% |
| war | 8 | 0.02% |
| organism | 8 | 0.02% |
| natural disaster | 7 | 0.02% |
| border | 6 | 0.01% |
| trade union | 5 | 0.01% |
| hardware component | 4 | 0.01% |
| deadline | 4 | 0.01% |
| operating system | 4 | 0.01% |
| financial instrument | 4 | 0.01% |
| mountain | 4 | 0.01% |
| model number | 3 | 0.01% |
| verdict | 3 | 0.01% |
| allergen | 3 | 0.01% |
| dosage | 2 | 0.00% |
| framework | 2 | 0.00% |
| pathogen | 1 | 0.00% |
| programming language | 1 | 0.00% |
| fiscal quarter | 0 | 0.00% |
| mineral | 0 | 0.00% |
| api | 0 | 0.00% |
| protein | 0 | 0.00% |
| database | 0 | 0.00% |
| file format | 0 | 0.00% |
| diagnosis | 0 | 0.00% |
| budget line | 0 | 0.00% |
| algorithm | 0 | 0.00% |
| gene | 0 | 0.00% |
| license | 0 | 0.00% |
| voltage | 0 | 0.00% |
| interest rate | 0 | 0.00% |
| patent | 0 | 0.00% |
| security vulnerability | 0 | 0.00% |
| credit rating | 0 | 0.00% |
| contract | 0 | 0.00% |
| protocol | 0 | 0.00% |
| tax | 0 | 0.00% |
| vaccine | 0 | 0.00% |
| clinical trial | 0 | 0.00% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| works_for | 2,426 | 13.21% |
| member_of | 2,226 | 12.12% |
| located_in | 1,811 | 9.86% |
| based_in | 1,514 | 8.25% |
| affiliated_with | 1,502 | 8.18% |
| employer_of | 975 | 5.31% |
| part_of | 776 | 4.23% |
| leader_of | 701 | 3.82% |
| colleague_of | 582 | 3.17% |
| operates_in | 492 | 2.68% |
| participated_in | 447 | 2.43% |
| headquartered_in | 445 | 2.42% |
| relative_of | 313 | 1.70% |
| reports_to | 309 | 1.68% |
| produced_by | 284 | 1.55% |
| represents | 266 | 1.45% |
| citizen_of | 265 | 1.44% |
| parent_of | 221 | 1.20% |
| developed_by | 185 | 1.01% |
| competitor_of | 185 | 1.01% |
| child_of | 156 | 0.85% |
| board_member_of | 142 | 0.77% |
| owns | 133 | 0.72% |
| authored_by | 116 | 0.63% |
| born_in | 114 | 0.62% |
| directed_by | 110 | 0.60% |
| founded | 105 | 0.57% |
| manufactured_by | 103 | 0.56% |
| spouse_of | 95 | 0.52% |
| designed_by | 92 | 0.50% |
| interacts_with | 87 | 0.47% |
| subsidiary_of | 85 | 0.46% |
| published_by | 83 | 0.45% |
| member_state_of | 82 | 0.45% |
| sibling_of | 70 | 0.38% |
| advisor_to | 57 | 0.31% |
| regulated_by | 57 | 0.31% |
| composed_by | 56 | 0.31% |
| party_to | 52 | 0.28% |
| charged_with | 52 | 0.28% |
| sponsored_by | 48 | 0.26% |
| opposed_to | 46 | 0.25% |
| supplier_of | 45 | 0.25% |
| named_after | 42 | 0.23% |
| won | 37 | 0.20% |
| customer_of | 34 | 0.19% |
| allied_with | 33 | 0.18% |
| distributor_of | 29 | 0.16% |
| acquired | 24 | 0.13% |
| adjacent_to | 23 | 0.13% |
| testified_against | 23 | 0.13% |
| died_in | 23 | 0.13% |
| invested_in | 21 | 0.11% |
| borders | 20 | 0.11% |
| successor_of | 18 | 0.10% |
| mentor_of | 17 | 0.09% |
| capital_of | 17 | 0.09% |
| convicted_of | 16 | 0.09% |
| derived_from | 13 | 0.07% |
| sold_to | 11 | 0.06% |
| predecessor_of | 9 | 0.05% |
| merged_with | 9 | 0.05% |
| causes | 8 | 0.04% |
| sued_by | 8 | 0.04% |
| licensed_to | 4 | 0.02% |
| signatory_of | 4 | 0.02% |
| depends_on | 2 | 0.01% |
| treats | 2 | 0.01% |
| implements | 1 | 0.01% |

#### val

| label | count | share |
| --- | --- | --- |
| works_for | 302 | 11.86% |
| located_in | 283 | 11.12% |
| member_of | 275 | 10.80% |
| affiliated_with | 187 | 7.34% |
| based_in | 174 | 6.83% |
| employer_of | 161 | 6.32% |
| part_of | 139 | 5.46% |
| operates_in | 105 | 4.12% |
| leader_of | 99 | 3.89% |
| colleague_of | 69 | 2.71% |
| represents | 61 | 2.40% |
| reports_to | 59 | 2.32% |
| developed_by | 54 | 2.12% |
| manufactured_by | 52 | 2.04% |
| competitor_of | 45 | 1.77% |
| headquartered_in | 44 | 1.73% |
| produced_by | 41 | 1.61% |
| citizen_of | 41 | 1.61% |
| authored_by | 40 | 1.57% |
| board_member_of | 27 | 1.06% |
| participated_in | 22 | 0.86% |
| charged_with | 21 | 0.82% |
| relative_of | 19 | 0.75% |
| parent_of | 17 | 0.67% |
| member_state_of | 15 | 0.59% |
| child_of | 15 | 0.59% |
| designed_by | 14 | 0.55% |
| born_in | 12 | 0.47% |
| opposed_to | 11 | 0.43% |
| regulated_by | 10 | 0.39% |
| founded | 10 | 0.39% |
| spouse_of | 9 | 0.35% |
| interacts_with | 9 | 0.35% |
| advisor_to | 9 | 0.35% |
| owns | 7 | 0.27% |
| supplier_of | 7 | 0.27% |
| sponsored_by | 6 | 0.24% |
| published_by | 6 | 0.24% |
| directed_by | 6 | 0.24% |
| composed_by | 6 | 0.24% |
| mentor_of | 6 | 0.24% |
| named_after | 6 | 0.24% |
| adjacent_to | 5 | 0.20% |
| subsidiary_of | 5 | 0.20% |
| sibling_of | 4 | 0.16% |
| testified_against | 4 | 0.16% |
| sued_by | 3 | 0.12% |
| merged_with | 3 | 0.12% |
| distributor_of | 3 | 0.12% |
| invested_in | 3 | 0.12% |
| died_in | 2 | 0.08% |
| won | 2 | 0.08% |
| allied_with | 2 | 0.08% |
| party_to | 2 | 0.08% |
| convicted_of | 1 | 0.04% |
| successor_of | 1 | 0.04% |
| borders | 1 | 0.04% |
| sold_to | 1 | 0.04% |
| causes | 1 | 0.04% |
| predecessor_of | 1 | 0.04% |
| derived_from | 1 | 0.04% |

#### test

| label | count | share |
| --- | --- | --- |
| works_for | 441 | 18.55% |
| affiliated_with | 235 | 9.89% |
| located_in | 207 | 8.71% |
| member_of | 193 | 8.12% |
| based_in | 155 | 6.52% |
| employer_of | 132 | 5.55% |
| part_of | 108 | 4.54% |
| headquartered_in | 78 | 3.28% |
| leader_of | 71 | 2.99% |
| colleague_of | 64 | 2.69% |
| operates_in | 60 | 2.52% |
| participated_in | 57 | 2.40% |
| citizen_of | 55 | 2.31% |
| relative_of | 47 | 1.98% |
| published_by | 34 | 1.43% |
| represents | 33 | 1.39% |
| reports_to | 28 | 1.18% |
| member_state_of | 27 | 1.14% |
| manufactured_by | 21 | 0.88% |
| parent_of | 20 | 0.84% |
| born_in | 19 | 0.80% |
| owns | 16 | 0.67% |
| directed_by | 16 | 0.67% |
| sponsored_by | 16 | 0.67% |
| authored_by | 16 | 0.67% |
| developed_by | 15 | 0.63% |
| designed_by | 14 | 0.59% |
| board_member_of | 14 | 0.59% |
| interacts_with | 14 | 0.59% |
| founded | 14 | 0.59% |
| child_of | 13 | 0.55% |
| competitor_of | 13 | 0.55% |
| named_after | 11 | 0.46% |
| opposed_to | 10 | 0.42% |
| produced_by | 9 | 0.38% |
| sibling_of | 8 | 0.34% |
| supplier_of | 8 | 0.34% |
| composed_by | 8 | 0.34% |
| acquired | 7 | 0.29% |
| advisor_to | 6 | 0.25% |
| convicted_of | 6 | 0.25% |
| subsidiary_of | 6 | 0.25% |
| spouse_of | 5 | 0.21% |
| mentor_of | 5 | 0.21% |
| customer_of | 5 | 0.21% |
| died_in | 4 | 0.17% |
| charged_with | 4 | 0.17% |
| borders | 3 | 0.13% |
| party_to | 3 | 0.13% |
| regulated_by | 3 | 0.13% |
| adjacent_to | 2 | 0.08% |
| capital_of | 2 | 0.08% |
| derived_from | 2 | 0.08% |
| successor_of | 2 | 0.08% |
| distributor_of | 2 | 0.08% |
| sold_to | 2 | 0.08% |
| signatory_of | 2 | 0.08% |
| depends_on | 1 | 0.04% |
| treats | 1 | 0.04% |
| predecessor_of | 1 | 0.04% |
| allied_with | 1 | 0.04% |
| won | 1 | 0.04% |
| causes | 1 | 0.04% |

### Classifications

**train, actionability.** Majority class 93.2% of 3,905 answers. The marginal is `true_label`, not the menu.

#### train actionability

| label | count | share |
| --- | --- | --- |
| informational | 3,639 | 93.19% |
| advisory | 215 | 5.51% |
| requires action | 51 | 1.31% |

**train, audience.** Majority class 88.0% of 4,425 answers. The marginal is `true_label`, not the menu.

#### train audience

| label | count | share |
| --- | --- | --- |
| general public | 3,894 | 88.00% |
| expert | 345 | 7.80% |
| regulatory | 177 | 4.00% |
| internal | 9 | 0.20% |

**train, certainty.** Majority class 97.6% of 4,093 answers. The marginal is `true_label`, not the menu.

#### train certainty

| label | count | share |
| --- | --- | --- |
| asserted | 3,993 | 97.56% |
| hedged | 91 | 2.22% |
| speculative | 8 | 0.20% |
| denied | 1 | 0.02% |

**train, document_genre.** Majority class 83.6% of 3,973 answers. The marginal is `true_label`, not the menu.

#### train document_genre

| label | count | share |
| --- | --- | --- |
| news report | 3,322 | 83.61% |
| announcement | 351 | 8.83% |
| opinion | 164 | 4.13% |
| review | 77 | 1.94% |
| technical documentation | 24 | 0.60% |
| correspondence | 16 | 0.40% |
| advertisement | 15 | 0.38% |
| legal filing | 4 | 0.10% |

**train, formality.** Majority class 81.5% of 4,002 answers. The marginal is `true_label`, not the menu.

#### train formality

| label | count | share |
| --- | --- | --- |
| formal | 3,261 | 81.48% |
| informal | 741 | 18.52% |

**train, language_register.** Majority class 89.6% of 3,967 answers. The marginal is `true_label`, not the menu.

#### train language_register

| label | count | share |
| --- | --- | --- |
| plain | 3,553 | 89.56% |
| technical | 305 | 7.69% |
| colloquial | 69 | 1.74% |
| legal | 40 | 1.01% |

**train, risk_level.** Majority class 47.6% of 3,685 answers. The marginal is `true_label`, not the menu.

#### train risk_level

| label | count | share |
| --- | --- | --- |
| none | 1,754 | 47.60% |
| high | 755 | 20.49% |
| moderate | 623 | 16.91% |
| low | 553 | 15.01% |

**train, sentiment.** Majority class 49.2% of 3,788 answers. The marginal is `true_label`, not the menu.

#### train sentiment

| label | count | share |
| --- | --- | --- |
| neutral | 1,862 | 49.16% |
| negative | 993 | 26.21% |
| positive | 933 | 24.63% |

**train, subjectivity.** Majority class 89.9% of 3,931 answers. The marginal is `true_label`, not the menu.

#### train subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 3,534 | 89.90% |
| subjective | 397 | 10.10% |

**train, temporal_orientation.** Majority class 77.4% of 3,799 answers. The marginal is `true_label`, not the menu.

#### train temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 2,940 | 77.39% |
| present | 564 | 14.85% |
| future | 243 | 6.40% |
| timeless | 52 | 1.37% |

**train, topic.** Majority class 15.8% of 6,619 answers. The marginal is `true_label`, not the menu.

#### train topic

| label | count | share |
| --- | --- | --- |
| politics | 1,046 | 15.80% |
| world | 976 | 14.75% |
| sports | 902 | 13.63% |
| business | 788 | 11.91% |
| legal | 764 | 11.54% |
| entertainment | 648 | 9.79% |
| health | 314 | 4.74% |
| technology | 256 | 3.87% |
| finance | 190 | 2.87% |
| education | 184 | 2.78% |
| environment | 151 | 2.28% |
| defense | 113 | 1.71% |
| science | 98 | 1.48% |
| travel | 90 | 1.36% |
| energy | 53 | 0.80% |
| agriculture | 46 | 0.69% |

**train, urgency.** Majority class 81.2% of 3,587 answers. The marginal is `true_label`, not the menu.

#### train urgency

| label | count | share |
| --- | --- | --- |
| routine | 2,912 | 81.18% |
| elevated | 547 | 15.25% |
| critical | 103 | 2.87% |
| urgent | 25 | 0.70% |

**val, actionability.** Majority class 93.9% of 524 answers. The marginal is `true_label`, not the menu.

#### val actionability

| label | count | share |
| --- | --- | --- |
| informational | 492 | 93.89% |
| advisory | 27 | 5.15% |
| requires action | 5 | 0.95% |

**val, audience.** Majority class 89.2% of 553 answers. The marginal is `true_label`, not the menu.

#### val audience

| label | count | share |
| --- | --- | --- |
| general public | 493 | 89.15% |
| expert | 45 | 8.14% |
| regulatory | 15 | 2.71% |

**val, certainty.** Majority class 97.2% of 528 answers. The marginal is `true_label`, not the menu.

#### val certainty

| label | count | share |
| --- | --- | --- |
| asserted | 513 | 97.16% |
| hedged | 14 | 2.65% |
| speculative | 1 | 0.19% |

**val, document_genre.** Majority class 83.6% of 500 answers. The marginal is `true_label`, not the menu.

#### val document_genre

| label | count | share |
| --- | --- | --- |
| news report | 418 | 83.60% |
| announcement | 47 | 9.40% |
| opinion | 24 | 4.80% |
| technical documentation | 5 | 1.00% |
| review | 4 | 0.80% |
| correspondence | 1 | 0.20% |
| advertisement | 1 | 0.20% |

**val, formality.** Majority class 79.6% of 519 answers. The marginal is `true_label`, not the menu.

#### val formality

| label | count | share |
| --- | --- | --- |
| formal | 413 | 79.58% |
| informal | 106 | 20.42% |

**val, language_register.** Majority class 89.0% of 502 answers. The marginal is `true_label`, not the menu.

#### val language_register

| label | count | share |
| --- | --- | --- |
| plain | 447 | 89.04% |
| technical | 42 | 8.37% |
| colloquial | 8 | 1.59% |
| legal | 5 | 1.00% |

**val, risk_level.** Majority class 47.0% of 494 answers. The marginal is `true_label`, not the menu.

#### val risk_level

| label | count | share |
| --- | --- | --- |
| none | 232 | 46.96% |
| high | 111 | 22.47% |
| moderate | 80 | 16.19% |
| low | 71 | 14.37% |

**val, sentiment.** Majority class 51.6% of 492 answers. The marginal is `true_label`, not the menu.

#### val sentiment

| label | count | share |
| --- | --- | --- |
| neutral | 254 | 51.63% |
| positive | 124 | 25.20% |
| negative | 114 | 23.17% |

**val, subjectivity.** Majority class 93.2% of 512 answers. The marginal is `true_label`, not the menu.

#### val subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 477 | 93.16% |
| subjective | 35 | 6.84% |

**val, temporal_orientation.** Majority class 76.9% of 529 answers. The marginal is `true_label`, not the menu.

#### val temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 407 | 76.94% |
| present | 86 | 16.26% |
| future | 30 | 5.67% |
| timeless | 6 | 1.13% |

**val, topic.** Majority class 16.1% of 839 answers. The marginal is `true_label`, not the menu.

#### val topic

| label | count | share |
| --- | --- | --- |
| politics | 135 | 16.09% |
| sports | 129 | 15.38% |
| world | 120 | 14.30% |
| business | 102 | 12.16% |
| legal | 98 | 11.68% |
| entertainment | 82 | 9.77% |
| technology | 33 | 3.93% |
| finance | 31 | 3.69% |
| education | 27 | 3.22% |
| health | 25 | 2.98% |
| environment | 17 | 2.03% |
| science | 13 | 1.55% |
| travel | 11 | 1.31% |
| defense | 9 | 1.07% |
| agriculture | 6 | 0.72% |
| energy | 1 | 0.12% |

**val, urgency.** Majority class 81.4% of 495 answers. The marginal is `true_label`, not the menu.

#### val urgency

| label | count | share |
| --- | --- | --- |
| routine | 403 | 81.41% |
| elevated | 74 | 14.95% |
| critical | 10 | 2.02% |
| urgent | 8 | 1.62% |

**test, actionability.** Majority class 91.4% of 534 answers. The marginal is `true_label`, not the menu.

#### test actionability

| label | count | share |
| --- | --- | --- |
| informational | 488 | 91.39% |
| advisory | 40 | 7.49% |
| requires action | 6 | 1.12% |

**test, audience.** Majority class 89.6% of 546 answers. The marginal is `true_label`, not the menu.

#### test audience

| label | count | share |
| --- | --- | --- |
| general public | 489 | 89.56% |
| expert | 38 | 6.96% |
| regulatory | 14 | 2.56% |
| internal | 5 | 0.92% |

**test, certainty.** Majority class 98.5% of 536 answers. The marginal is `true_label`, not the menu.

#### test certainty

| label | count | share |
| --- | --- | --- |
| asserted | 528 | 98.51% |
| hedged | 7 | 1.31% |
| speculative | 1 | 0.19% |

**test, document_genre.** Majority class 82.1% of 497 answers. The marginal is `true_label`, not the menu.

#### test document_genre

| label | count | share |
| --- | --- | --- |
| news report | 408 | 82.09% |
| announcement | 46 | 9.26% |
| opinion | 27 | 5.43% |
| review | 8 | 1.61% |
| technical documentation | 5 | 1.01% |
| correspondence | 2 | 0.40% |
| advertisement | 1 | 0.20% |

**test, formality.** Majority class 80.6% of 500 answers. The marginal is `true_label`, not the menu.

#### test formality

| label | count | share |
| --- | --- | --- |
| formal | 403 | 80.60% |
| informal | 97 | 19.40% |

**test, language_register.** Majority class 88.2% of 473 answers. The marginal is `true_label`, not the menu.

#### test language_register

| label | count | share |
| --- | --- | --- |
| plain | 417 | 88.16% |
| technical | 39 | 8.25% |
| colloquial | 10 | 2.11% |
| legal | 7 | 1.48% |

**test, risk_level.** Majority class 49.6% of 464 answers. The marginal is `true_label`, not the menu.

#### test risk_level

| label | count | share |
| --- | --- | --- |
| none | 230 | 49.57% |
| high | 98 | 21.12% |
| moderate | 73 | 15.73% |
| low | 63 | 13.58% |

**test, sentiment.** Majority class 48.2% of 492 answers. The marginal is `true_label`, not the menu.

#### test sentiment

| label | count | share |
| --- | --- | --- |
| neutral | 237 | 48.17% |
| positive | 140 | 28.46% |
| negative | 115 | 23.37% |

**test, subjectivity.** Majority class 89.7% of 497 answers. The marginal is `true_label`, not the menu.

#### test subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 446 | 89.74% |
| subjective | 51 | 10.26% |

**test, temporal_orientation.** Majority class 76.1% of 522 answers. The marginal is `true_label`, not the menu.

#### test temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 397 | 76.05% |
| present | 75 | 14.37% |
| future | 38 | 7.28% |
| timeless | 12 | 2.30% |

**test, topic.** Majority class 14.5% of 887 answers. The marginal is `true_label`, not the menu.

#### test topic

| label | count | share |
| --- | --- | --- |
| sports | 129 | 14.54% |
| world | 127 | 14.32% |
| politics | 126 | 14.21% |
| legal | 119 | 13.42% |
| business | 104 | 11.72% |
| entertainment | 80 | 9.02% |
| health | 43 | 4.85% |
| education | 37 | 4.17% |
| technology | 30 | 3.38% |
| environment | 22 | 2.48% |
| finance | 21 | 2.37% |
| defense | 13 | 1.47% |
| travel | 11 | 1.24% |
| science | 10 | 1.13% |
| energy | 8 | 0.90% |
| agriculture | 7 | 0.79% |

**test, urgency.** Majority class 82.4% of 484 answers. The marginal is `true_label`, not the menu.

#### test urgency

| label | count | share |
| --- | --- | --- |
| routine | 399 | 82.44% |
| elevated | 72 | 14.88% |
| critical | 11 | 2.27% |
| urgent | 2 | 0.41% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | condition | 20 | 86.96% |
| clinical_finding | patient_group | 18 | 78.26% |
| clinical_finding | treatment | 16 | 69.57% |
| clinical_finding | outcome | 14 | 60.87% |
| clinical_finding | record_metadata | natural/condition ×19, natural/patient_group ×2, natural/outcome ×1 |  |
| company_profile | name | 135 | 97.83% |
| company_profile | size | 113 | 81.88% |
| company_profile | industry | 85 | 61.59% |
| company_profile | headquarters | 85 | 61.59% |
| company_profile | founded | 32 | 23.19% |
| company_profile | record_metadata | natural/name ×123, natural/industry ×2, natural/headquarters ×1 |  |
| event_listing | date | 210 | 89.36% |
| event_listing | name | 208 | 88.51% |
| event_listing | venue | 191 | 81.28% |
| event_listing | organizer | 141 | 60.00% |
| event_listing | price | 37 | 15.74% |
| event_listing | record_metadata | natural/name ×159, natural/venue ×18, natural/date ×3 |  |
| financial_report | organization | 12 | 100.00% |
| financial_report | period | 12 | 100.00% |
| financial_report | revenue | 8 | 66.67% |
| financial_report | net_income | 7 | 58.33% |
| financial_report | record_metadata | natural/organization ×12 |  |
| incident_report | severity | 187 | 99.47% |
| incident_report | date | 179 | 95.21% |
| incident_report | location | 160 | 85.11% |
| incident_report | incident_type | 129 | 68.62% |
| incident_report | casualties | 77 | 40.96% |
| incident_report | record_metadata | natural/location ×152, natural/incident_type ×14, natural/date ×13, natural/severity ×1 |  |
| job_posting | title | 7 | 100.00% |
| job_posting | company | 7 | 100.00% |
| job_posting | employment_type | 7 | 100.00% |
| job_posting | location | 6 | 85.71% |
| job_posting | record_metadata | natural/location ×5, natural/title ×1 |  |
| legal_case | status | 75 | 100.00% |
| legal_case | defendant | 69 | 92.00% |
| legal_case | court | 48 | 64.00% |
| legal_case | plaintiff | 42 | 56.00% |
| legal_case | case_name | 37 | 49.33% |
| legal_case | record_metadata | natural/case_name ×37, natural/court ×25, natural/plaintiff ×8, natural/defendant ×3 |  |
| person_profile | name | 356 | 98.34% |
| person_profile | role | 314 | 86.74% |
| person_profile | location | 252 | 69.61% |
| person_profile | employer | 210 | 58.01% |
| person_profile | record_metadata | natural/location ×196, natural/name ×74, natural/role ×2 |  |
| product | name | 139 | 95.86% |
| product | condition | 132 | 91.03% |
| product | brand | 106 | 73.10% |
| product | price | 56 | 38.62% |
| product | record_metadata | natural/name ×95, natural/brand ×3, natural/condition ×1, natural/price ×1 |  |
| real_estate_listing | property_type | 13 | 100.00% |
| real_estate_listing | address | 12 | 92.31% |
| real_estate_listing | price | 9 | 69.23% |
| real_estate_listing | bedrooms | 8 | 61.54% |
| real_estate_listing | area | 4 | 30.77% |
| real_estate_listing | record_metadata | natural/address ×6, natural/area ×1 |  |
| research_paper | venue | 5 | 71.43% |
| research_paper | authors | 3 | 42.86% |
| research_paper | year | 3 | 42.86% |
| research_paper | title | 3 | 42.86% |
| research_paper | topic | 1 | 14.29% |
| research_paper | record_metadata | natural/title ×3, natural/venue ×2, natural/authors ×1, natural/topic ×1 |  |
| software_release | product | 9 | 100.00% |
| software_release | platform | 8 | 88.89% |
| software_release | release_date | 5 | 55.56% |
| software_release | version | 3 | 33.33% |
| software_release | record_metadata | natural/product ×9 |  |
| transaction | amount | 41 | 89.13% |
| transaction | item | 39 | 84.78% |
| transaction | buyer | 35 | 76.09% |
| transaction | seller | 25 | 54.35% |
| transaction | date | 23 | 50.00% |
| transaction | record_metadata | natural/item ×33, natural/amount ×5, natural/buyer ×2 |  |
| vehicle_listing | model | 19 | 100.00% |
| vehicle_listing | make | 18 | 94.74% |
| vehicle_listing | year | 16 | 84.21% |
| vehicle_listing | price | 4 | 21.05% |
| vehicle_listing | mileage | 3 | 15.79% |
| vehicle_listing | record_metadata | natural/make ×13, natural/model ×1 |  |

**val.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | condition | 2 | 66.67% |
| clinical_finding | patient_group | 2 | 66.67% |
| clinical_finding | treatment | 1 | 33.33% |
| clinical_finding | outcome | 1 | 33.33% |
| clinical_finding | record_metadata | natural/condition ×2, natural/treatment ×1 |  |
| company_profile | name | 19 | 100.00% |
| company_profile | size | 16 | 84.21% |
| company_profile | industry | 14 | 73.68% |
| company_profile | headquarters | 12 | 63.16% |
| company_profile | founded | 2 | 10.53% |
| company_profile | record_metadata | natural/name ×18 |  |
| event_listing | date | 30 | 90.91% |
| event_listing | name | 28 | 84.85% |
| event_listing | venue | 28 | 84.85% |
| event_listing | organizer | 12 | 36.36% |
| event_listing | price | 7 | 21.21% |
| event_listing | record_metadata | natural/name ×22, natural/venue ×3, natural/date ×2 |  |
| financial_report | organization | 1 | 100.00% |
| financial_report | period | 1 | 100.00% |
| financial_report | revenue | 1 | 100.00% |
| financial_report | net_income | 1 | 100.00% |
| financial_report | record_metadata | natural/organization ×1 |  |
| incident_report | severity | 19 | 100.00% |
| incident_report | location | 18 | 94.74% |
| incident_report | date | 18 | 94.74% |
| incident_report | incident_type | 10 | 52.63% |
| incident_report | casualties | 8 | 42.11% |
| incident_report | record_metadata | natural/location ×18, natural/date ×1 |  |
| job_posting | title | 2 | 100.00% |
| job_posting | company | 2 | 100.00% |
| job_posting | employment_type | 2 | 100.00% |
| job_posting | location | 1 | 50.00% |
| job_posting | record_metadata | natural/location ×1, natural/title ×1 |  |
| legal_case | status | 13 | 100.00% |
| legal_case | court | 10 | 76.92% |
| legal_case | defendant | 10 | 76.92% |
| legal_case | plaintiff | 7 | 53.85% |
| legal_case | case_name | 6 | 46.15% |
| legal_case | record_metadata | natural/case_name ×6, natural/court ×5, natural/plaintiff ×2 |  |
| person_profile | name | 51 | 98.08% |
| person_profile | role | 42 | 80.77% |
| person_profile | location | 36 | 69.23% |
| person_profile | employer | 25 | 48.08% |
| person_profile | record_metadata | natural/location ×25, natural/name ×11, natural/role ×1 |  |
| product | condition | 14 | 93.33% |
| product | name | 13 | 86.67% |
| product | brand | 13 | 86.67% |
| product | price | 5 | 33.33% |
| product | record_metadata | natural/name ×11, natural/brand ×2 |  |
| real_estate_listing | address | 1 | 100.00% |
| real_estate_listing | price | 1 | 100.00% |
| real_estate_listing | property_type | 1 | 100.00% |
| real_estate_listing | record_metadata | natural/address ×1 |  |
| transaction | amount | 3 | 100.00% |
| transaction | buyer | 3 | 100.00% |
| transaction | seller | 3 | 100.00% |
| transaction | date | 2 | 66.67% |
| transaction | item | 2 | 66.67% |
| transaction | record_metadata | natural/item ×2, natural/amount ×1 |  |
| vehicle_listing | make | 3 | 100.00% |
| vehicle_listing | model | 3 | 100.00% |
| vehicle_listing | year | 3 | 100.00% |
| vehicle_listing | price | 2 | 66.67% |
| vehicle_listing | record_metadata | natural/make ×2 |  |

**test.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| company_profile | name | 24 | 100.00% |
| company_profile | size | 18 | 75.00% |
| company_profile | industry | 12 | 50.00% |
| company_profile | headquarters | 11 | 45.83% |
| company_profile | founded | 3 | 12.50% |
| company_profile | record_metadata | natural/name ×23 |  |
| event_listing | date | 58 | 90.62% |
| event_listing | venue | 53 | 82.81% |
| event_listing | name | 36 | 56.25% |
| event_listing | organizer | 15 | 23.44% |
| event_listing | price | 9 | 14.06% |
| event_listing | record_metadata | natural/name ×22, natural/venue ×5, natural/date ×3 |  |
| financial_report | organization | 4 | 100.00% |
| financial_report | period | 4 | 100.00% |
| financial_report | revenue | 3 | 75.00% |
| financial_report | net_income | 3 | 75.00% |
| financial_report | record_metadata | natural/organization ×4 |  |
| incident_report | severity | 28 | 100.00% |
| incident_report | date | 27 | 96.43% |
| incident_report | location | 22 | 78.57% |
| incident_report | incident_type | 20 | 71.43% |
| incident_report | casualties | 10 | 35.71% |
| incident_report | record_metadata | natural/location ×22, natural/incident_type ×4, natural/date ×2 |  |
| job_posting | title | 11 | 100.00% |
| job_posting | employment_type | 11 | 100.00% |
| job_posting | company | 9 | 81.82% |
| job_posting | location | 4 | 36.36% |
| job_posting | record_metadata | natural/title ×1 |  |
| legal_case | defendant | 8 | 100.00% |
| legal_case | status | 8 | 100.00% |
| legal_case | court | 5 | 62.50% |
| legal_case | plaintiff | 5 | 62.50% |
| legal_case | record_metadata | natural/court ×5, natural/plaintiff ×3 |  |
| person_profile | name | 36 | 100.00% |
| person_profile | role | 32 | 88.89% |
| person_profile | location | 25 | 69.44% |
| person_profile | employer | 20 | 55.56% |
| person_profile | record_metadata | natural/location ×23, natural/name ×8 |  |
| product | name | 14 | 93.33% |
| product | brand | 14 | 93.33% |
| product | condition | 12 | 80.00% |
| product | price | 12 | 80.00% |
| product | record_metadata | natural/name ×6, natural/brand ×1 |  |
| research_paper | authors | 1 | 100.00% |
| research_paper | venue | 1 | 100.00% |
| research_paper | year | 1 | 100.00% |
| research_paper | record_metadata | natural/authors ×1 |  |
| transaction | item | 18 | 94.74% |
| transaction | amount | 17 | 89.47% |
| transaction | buyer | 15 | 78.95% |
| transaction | seller | 12 | 63.16% |
| transaction | date | 5 | 26.32% |
| transaction | record_metadata | natural/item ×10, natural/amount ×1 |  |

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | llm_real |
| relations | llm_real |
| events | llm_real |
| classifications | llm_real |
| structures | llm_real |
