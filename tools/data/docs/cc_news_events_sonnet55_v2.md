# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cc_news_events_sonnet55_v2 | train | 4,597 | 4,597 | 115,250 | 3,391 | 14,650 | 624 | 18,690 | 26,598 |
| cc_news_events_sonnet55_v2 | val | 500 | 500 | 11,742 | 316 | 1,555 | 69 | 1,897 | 2,652 |
| cc_news_events_sonnet55_v2 | test | 500 | 500 | 11,567 | 276 | 1,536 | 66 | 1,865 | 2,701 |

## cc_news_events_sonnet55_v2

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4,597 | 4,597 | 417 | 1024 | 2919 | 115,250 | 3,391 | 14,650 | 624 | 18,690 | 26,598 |
| val | 500 | 500 | 407 | 982 | 2450 | 11,742 | 316 | 1,555 | 69 | 1,897 | 2,652 |
| test | 500 | 500 | 384 | 993 | 2330 | 11,567 | 276 | 1,536 | 66 | 1,865 | 2,701 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 4,595 | 499 | 500 | 5,594 | 99.95% |
| und | 2 | 1 | 0 | 3 | 0.05% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 500 |
| test_unique_inputs | 500 |
| train_records | 4,597 |
| train_unique_inputs | 4,597 |
| val_records | 500 |
| val_unique_inputs | 500 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0490 | 115,250 / 11,742 | organization 5%→4%, age 1%→2%, company 4%→3%, person 21%→21%, percent 1%→1% |
| entities (per mention) | train / test | 0.0513 | 115,250 / 11,567 | sports team 3%→4%, person 21%→22%, location 7%→6%, geopolitical entity 5%→4%, organization 5%→5% |
| entities (per mention) | val / test | 0.0584 | 11,742 / 11,567 | sports team 3%→4%, geopolitical entity 5%→4%, money 2%→3%, age 2%→1%, location 7%→6% |
| relations (per instance) | train / val | 0.1975 WARN | 3,391 / 316 | works_for 17%→21%, affiliated_with 8%→5%, convicted_of 0%→2%, composed_by 0%→2%, participated_in 2%→0% |
| relations (per instance) | train / test | 0.2072 WARN | 3,391 / 276 | works_for 17%→25%, located_in 8%→12%, leader_of 7%→4%, member_of 10%→8%, based_in 3%→5% |
| relations (per instance) | val / test | 0.2301 WARN | 316 / 276 | affiliated_with 5%→9%, works_for 21%→25%, leader_of 8%→4%, based_in 3%→5%, spouse_of 2%→0% |
| event types (per trigger mention) | train / val | 0.0569 | 19,917 / 2,045 | Justice.Arrest 2%→3%, Life.Die 6%→7%, Sport.Compete 7%→6%, Medical.Treat 1%→1%, Government.Vote 1%→0% |
| event types (per trigger mention) | train / test | 0.0677 | 19,917 / 2,004 | Contact.Communicate 52%→53%, Justice.ChargeIndict 2%→3%, Life.Die 6%→5%, Disaster.Accident 2%→1%, Disaster.Rescue 0%→1% |
| event types (per trigger mention) | val / test | 0.0808 | 2,045 / 2,004 | Life.Die 7%→5%, Justice.ChargeIndict 2%→3%, Contact.Communicate 52%→53%, Disaster.Accident 2%→1%, Justice.Arrest 3%→2% |
| event roles (per argument) | train / val | 0.0551 | 26,692 / 2,670 | Victim 4%→5%, Crime 2%→2%, Participant 3%→2%, Time 13%→14%, Organization 3%→2% |
| event roles (per argument) | train / test | 0.0542 | 26,692 / 2,715 | Participant 3%→2%, Crime 2%→3%, Time 13%→14%, Place 8%→7%, Person 4%→4% |
| event roles (per argument) | val / test | 0.0555 | 2,670 / 2,715 | Place 8%→7%, Person 5%→4%, Competitor 5%→6%, Victim 5%→4%, Author 0%→1% |
| classifications (true_label) | train / val | 0.0725 | 14,650 / 1,555 | certainty: asserted 7%→9%, urgency: routine 7%→6%, formality: formal 5%→4%, temporal_orientation: past 4%→5%, sentiment: negative 2%→3% |
| classifications (true_label) | train / test | 0.0734 | 14,650 / 1,536 | actionability: informational 7%→8%, document_genre: news report 6%→7%, urgency: routine 7%→8%, sentiment: neutral 3%→2%, temporal_orientation: past 4%→5% |
| classifications (true_label) | val / test | 0.0922 | 1,555 / 1,536 | certainty: asserted 9%→7%, urgency: routine 6%→8%, actionability: informational 6%→8%, sentiment: negative 3%→2%, formality: formal 4%→5% |

Thin support in val, entities (per mention): 34 label(s) under 10 gold. model number ×1, currency ×1, court case ×1, postal code ×1, voltage ×1, pathogen ×1, vaccine ×1, market index ×1, verdict ×1, database ×1, interest rate ×1, trade union ×2, treaty ×2, protocol ×2, clinical trial ×2, allergen ×2, license ×2, natural disaster ×3, file format ×3, witness ×3, fiscal quarter ×4, stock ticker ×5, airport ×5, mineral ×5, aircraft ×6, temperature ×6, pseudonym ×6, speed ×6, academic degree ×7, conference ×7, war ×8, hardware component ×8, mountain ×8, era ×9.
Unseen in train, present in val, relations (per instance): supersedes ×1.
Thin support in val, relations (per instance): 36 label(s) under 10 gold. predecessor_of ×1, causes ×1, capital_of ×1, supersedes ×1, died_in ×1, represents ×1, colleague_of ×1, borders ×1, born_in ×1, authored_by ×1, designed_by ×1, invested_in ×1, acquired ×1, advisor_to ×1, reports_to ×2, headquartered_in ×2, citizen_of ×2, competitor_of ×2, parent_of ×2, allied_with ×2, owns ×3, successor_of ×3, charged_with ×3, supplier_of ×4, operates_in ×4, founded ×5, manufactured_by ×6, subsidiary_of ×6, child_of ×6, composed_by ×6, relative_of ×6, spouse_of ×7, developed_by ×7, convicted_of ×7, produced_by ×7, based_in ×8.
Thin support in val, event types (per trigger mention): 24 label(s) under 10 gold. Research.Discover ×1, Contact.Negotiate ×1, Business.EndOrg ×1, Conflict.Ceasefire ×1, Government.EnactLaw ×2, Research.Fund ×2, Tech.Breach ×3, Life.Divorce ×3, Justice.Acquit ×3, Transaction.TransferMoney ×3, Business.DeclareBankruptcy ×4, Movement.Migrate ×5, Government.Vote ×5, Government.Sanction ×5, Personnel.Nominate ×5, Justice.Fine ×5, Justice.Appeal ×6, Medical.Diagnose ×7, Life.BeBorn ×7, Business.Layoff ×7, Business.LaunchProduct ×8, Research.Publish ×8, Disaster.NaturalDisaster ×9, Personnel.Elect ×9.
Thin support in val, event roles (per argument): 29 label(s) under 10 gold. Researcher ×1, Finding ×1, Physician ×1, Affected ×1, Funder ×1, Giver ×1, Topic ×1, Proposal ×2, Venue ×2, Mediator ×2, Voter ×3, Awarder ×3, Quantity ×3, Prosecutor ×4, Evacuee ×4, Price ×4, Type ×4, Rescuer ×4, Author ×4, Migrant ×5, Plaintiff ×5, Publication ×5, Donor ×6, Treatment ×6, Seller ×6, Appellant ×6, Diagnosis ×6, Money ×7, Buyer ×8.
Thin support in val, classifications (true_label): 16 label(s) under 10 gold. audience: regulatory ×1, urgency: critical ×1, topic: energy ×3, topic: education ×3, actionability: requires action ×3, document_genre: review ×3, topic: travel ×4, topic: defense ×4, topic: science ×4, language_register: technical ×4, urgency: urgent ×4, risk_level: high ×4, topic: environment ×5, temporal_orientation: timeless ×5, actionability: advisory ×6, temporal_orientation: future ×9.
Thin support in test, entities (per mention): 40 label(s) under 10 gold. mountain ×1, conference ×1, diagnosis ×1, court case ×1, voltage ×1, pathogen ×1, stock ticker ×2, witness ×2, algorithm ×2, file format ×2, license ×2, model number ×2, regulation ×2, protein ×2, financial instrument ×3, trade union ×3, clinical trial ×3, speed ×4, contract ×4, treaty ×4, pseudonym ×4, border ×4, tax ×5, mineral ×5, protocol ×5, airport ×5, vessel ×5, hardware component ×6, operating system ×6, criminal organization ×7, regulatory body ×7, market index ×7, war ×7, currency ×8, sentence ×8, aircraft ×8, temperature ×8, fiscal quarter ×9, landmark ×9, area ×9.
Unseen in train, present in test, relations (per instance): sold_to ×1.
Thin support in test, relations (per instance): 28 label(s) under 10 gold. sold_to ×1, allied_with ×1, designed_by ×1, board_member_of ×1, charged_with ×1, directed_by ×1, authored_by ×1, capital_of ×1, owns ×1, derived_from ×1, advisor_to ×1, predecessor_of ×1, reports_to ×2, operates_in ×2, manufactured_by ×2, produced_by ×2, colleague_of ×2, headquartered_in ×3, opposed_to ×3, published_by ×3, founded ×3, sponsored_by ×3, developed_by ×4, child_of ×4, parent_of ×4, participated_in ×4, subsidiary_of ×5, relative_of ×8.
Thin support in test, event types (per trigger mention): 26 label(s) under 10 gold. Research.Fund ×1, Transaction.Lend ×1, Movement.Deport ×1, Medical.Diagnose ×2, Business.Layoff ×2, Disaster.Outbreak ×2, Movement.Migrate ×3, Justice.Fine ×3, Tech.Breach ×3, Government.EnactLaw ×4, Business.EndOrg ×4, Justice.Acquit ×4, Contact.Negotiate ×5, Life.Evacuate ×5, Justice.Appeal ×5, Personnel.Elect ×5, Government.Sanction ×5, Tech.Release ×6, Business.LaunchProduct ×6, Life.Marry ×7, Transaction.Donate ×7, Government.Vote ×7, Movement.Transport ×8, Personnel.Nominate ×9, Justice.Sue ×9, Life.BeBorn ×9.
Thin support in test, event roles (per argument): 32 label(s) under 10 gold. Evacuee ×1, Quantity ×1, Data ×1, Funder ×1, Lender ×1, Disease ×1, Physician ×2, Affected ×2, Topic ×2, Migrant ×2, Diagnosis ×2, Appellant ×2, Law ×3, Finding ×3, Proposal ×3, Treatment ×4, Origin ×4, Price ×4, Donor ×4, Voter ×4, Venue ×5, Plaintiff ×5, Researcher ×5, Type ×6, Prosecutor ×7, Giver ×7, Awarder ×7, Sentence ×7, Seller ×7, Rescuer ×8, Entity ×8, Demonstrator ×8.
Thin support in test, classifications (true_label): 19 label(s) under 10 gold. audience: regulatory ×1, document_genre: review ×1, temporal_orientation: timeless ×1, topic: energy ×1, urgency: urgent ×1, language_register: legal ×1, language_register: technical ×1, actionability: requires action ×2, document_genre: advertisement ×2, topic: travel ×2, audience: expert ×3, risk_level: high ×3, topic: agriculture ×3, topic: defense ×4, topic: finance ×5, topic: science ×5, topic: environment ×6, document_genre: announcement ×7, topic: education ×8.

### Events

**train.** 13,855 instances in 4,101 records, 0 triggerless. Trigger gold 18,690. Argument gold 26,598. Arguments per instance, mean 1.93. Trigger length in whitespace tokens, mean 1.70, single-token share 69.0%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Contact.Communicate | 10,287 | 51.65% |
| Sport.Compete | 1,324 | 6.65% |
| Life.Die | 1,247 | 6.26% |
| Conflict.Attack | 724 | 3.64% |
| Life.Injure | 516 | 2.59% |
| Personnel.EndPosition | 503 | 2.53% |
| Justice.Arrest | 402 | 2.02% |
| Disaster.Accident | 399 | 2.00% |
| Personnel.StartPosition | 353 | 1.77% |
| Justice.ChargeIndict | 315 | 1.58% |
| Contact.Meet | 297 | 1.49% |
| Award.ReceiveAward | 232 | 1.16% |
| Conflict.Demonstrate | 216 | 1.08% |
| Transaction.TransferOwnership | 209 | 1.05% |
| Justice.TrialHearing | 179 | 0.90% |
| Movement.Arrive | 179 | 0.90% |
| Justice.Sue | 155 | 0.78% |
| Medical.Treat | 142 | 0.71% |
| Government.Vote | 141 | 0.71% |
| Justice.Sentence | 139 | 0.70% |
| Business.StartOrg | 138 | 0.69% |
| Disaster.NaturalDisaster | 115 | 0.58% |
| Movement.Transport | 113 | 0.57% |
| Business.LaunchProduct | 111 | 0.56% |
| Life.Marry | 108 | 0.54% |
| Life.BeBorn | 106 | 0.53% |
| Research.Publish | 100 | 0.50% |
| Transaction.TransferMoney | 91 | 0.46% |
| Personnel.Elect | 90 | 0.45% |
| Contact.Negotiate | 87 | 0.44% |
| Transaction.Donate | 86 | 0.43% |
| Government.EnactLaw | 80 | 0.40% |
| Disaster.Rescue | 64 | 0.32% |
| Life.Evacuate | 62 | 0.31% |
| Tech.Release | 60 | 0.30% |
| Tech.Breach | 57 | 0.29% |
| Justice.Appeal | 55 | 0.28% |
| Personnel.Nominate | 55 | 0.28% |
| Medical.Diagnose | 53 | 0.27% |
| Business.EndOrg | 42 | 0.21% |
| Movement.Migrate | 38 | 0.19% |
| Life.Divorce | 36 | 0.18% |
| Research.Discover | 31 | 0.16% |
| Government.Sanction | 31 | 0.16% |
| Justice.Fine | 30 | 0.15% |
| Justice.Acquit | 20 | 0.10% |
| Business.Strike | 17 | 0.09% |
| Research.Fund | 17 | 0.09% |
| Movement.Deport | 13 | 0.07% |
| Business.Layoff | 12 | 0.06% |
| Transaction.Lend | 11 | 0.06% |
| Disaster.Outbreak | 11 | 0.06% |
| Business.DeclareBankruptcy | 9 | 0.05% |
| Business.MergeOrg | 6 | 0.03% |
| Conflict.Ceasefire | 3 | 0.02% |

**val.** 1,470 instances in 441 records, 0 triggerless. Trigger gold 1,897. Argument gold 2,652. Arguments per instance, mean 1.82. Trigger length in whitespace tokens, mean 1.52, single-token share 74.2%.

#### val event types

| label | count | share |
| --- | --- | --- |
| Contact.Communicate | 1,064 | 52.03% |
| Life.Die | 145 | 7.09% |
| Sport.Compete | 120 | 5.87% |
| Conflict.Attack | 75 | 3.67% |
| Justice.Arrest | 61 | 2.98% |
| Life.Injure | 52 | 2.54% |
| Disaster.Accident | 48 | 2.35% |
| Personnel.EndPosition | 47 | 2.30% |
| Justice.ChargeIndict | 36 | 1.76% |
| Personnel.StartPosition | 33 | 1.61% |
| Medical.Treat | 27 | 1.32% |
| Contact.Meet | 24 | 1.17% |
| Award.ReceiveAward | 21 | 1.03% |
| Justice.Sentence | 20 | 0.98% |
| Justice.TrialHearing | 19 | 0.93% |
| Conflict.Demonstrate | 16 | 0.78% |
| Business.StartOrg | 15 | 0.73% |
| Transaction.TransferOwnership | 15 | 0.73% |
| Disaster.Rescue | 14 | 0.68% |
| Movement.Transport | 13 | 0.64% |
| Movement.Arrive | 13 | 0.64% |
| Life.Marry | 12 | 0.59% |
| Life.Evacuate | 12 | 0.59% |
| Tech.Release | 12 | 0.59% |
| Transaction.Donate | 11 | 0.54% |
| Justice.Sue | 10 | 0.49% |
| Disaster.NaturalDisaster | 9 | 0.44% |
| Personnel.Elect | 9 | 0.44% |
| Business.LaunchProduct | 8 | 0.39% |
| Research.Publish | 8 | 0.39% |
| Medical.Diagnose | 7 | 0.34% |
| Life.BeBorn | 7 | 0.34% |
| Business.Layoff | 7 | 0.34% |
| Justice.Appeal | 6 | 0.29% |
| Movement.Migrate | 5 | 0.24% |
| Government.Vote | 5 | 0.24% |
| Government.Sanction | 5 | 0.24% |
| Personnel.Nominate | 5 | 0.24% |
| Justice.Fine | 5 | 0.24% |
| Business.DeclareBankruptcy | 4 | 0.20% |
| Tech.Breach | 3 | 0.15% |
| Life.Divorce | 3 | 0.15% |
| Justice.Acquit | 3 | 0.15% |
| Transaction.TransferMoney | 3 | 0.15% |
| Government.EnactLaw | 2 | 0.10% |
| Research.Fund | 2 | 0.10% |
| Research.Discover | 1 | 0.05% |
| Contact.Negotiate | 1 | 0.05% |
| Business.EndOrg | 1 | 0.05% |
| Conflict.Ceasefire | 1 | 0.05% |

**test.** 1,474 instances in 446 records, 0 triggerless. Trigger gold 1,865. Argument gold 2,701. Arguments per instance, mean 1.84. Trigger length in whitespace tokens, mean 1.59, single-token share 72.4%.

#### test event types

| label | count | share |
| --- | --- | --- |
| Contact.Communicate | 1,067 | 53.24% |
| Sport.Compete | 131 | 6.54% |
| Life.Die | 100 | 4.99% |
| Conflict.Attack | 67 | 3.34% |
| Justice.ChargeIndict | 60 | 2.99% |
| Life.Injure | 59 | 2.94% |
| Personnel.EndPosition | 50 | 2.50% |
| Justice.Arrest | 41 | 2.05% |
| Personnel.StartPosition | 29 | 1.45% |
| Award.ReceiveAward | 29 | 1.45% |
| Justice.TrialHearing | 27 | 1.35% |
| Contact.Meet | 24 | 1.20% |
| Medical.Treat | 23 | 1.15% |
| Disaster.Accident | 23 | 1.15% |
| Disaster.Rescue | 19 | 0.95% |
| Transaction.TransferOwnership | 19 | 0.95% |
| Movement.Arrive | 18 | 0.90% |
| Research.Publish | 18 | 0.90% |
| Business.StartOrg | 18 | 0.90% |
| Justice.Sentence | 15 | 0.75% |
| Conflict.Demonstrate | 12 | 0.60% |
| Transaction.TransferMoney | 11 | 0.55% |
| Research.Discover | 11 | 0.55% |
| Disaster.NaturalDisaster | 10 | 0.50% |
| Personnel.Nominate | 9 | 0.45% |
| Justice.Sue | 9 | 0.45% |
| Life.BeBorn | 9 | 0.45% |
| Movement.Transport | 8 | 0.40% |
| Life.Marry | 7 | 0.35% |
| Transaction.Donate | 7 | 0.35% |
| Government.Vote | 7 | 0.35% |
| Tech.Release | 6 | 0.30% |
| Business.LaunchProduct | 6 | 0.30% |
| Contact.Negotiate | 5 | 0.25% |
| Life.Evacuate | 5 | 0.25% |
| Justice.Appeal | 5 | 0.25% |
| Personnel.Elect | 5 | 0.25% |
| Government.Sanction | 5 | 0.25% |
| Government.EnactLaw | 4 | 0.20% |
| Business.EndOrg | 4 | 0.20% |
| Justice.Acquit | 4 | 0.20% |
| Movement.Migrate | 3 | 0.15% |
| Justice.Fine | 3 | 0.15% |
| Tech.Breach | 3 | 0.15% |
| Medical.Diagnose | 2 | 0.10% |
| Business.Layoff | 2 | 0.10% |
| Disaster.Outbreak | 2 | 0.10% |
| Research.Fund | 1 | 0.05% |
| Transaction.Lend | 1 | 0.05% |
| Movement.Deport | 1 | 0.05% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 26,692 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 2,670 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 2,715 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2.03 | 33.73% | 2.59 | 41.03% | 0.07% | 0 | 26,692 | 0 | 19,917 |
| val | 2.21 | 36.09% | 2.54 | 41.84% | 0.00% | 0 | 2,670 | 0 | 2,045 |
| test | 2.10 | 35.88% | 2.47 | 41.53% | 0.08% | 0 | 2,715 | 0 | 2,004 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 13,855 | 27.14% | 9,864 | 19.03% | 16.29% | 8.88% | 4.28% | 21.09% |
| val | 1,470 | 24.56% | 929 | 16.68% | 14.75% | 7.32% | 3.34% | 18.59% |
| test | 1,474 | 24.08% | 909 | 12.32% | 10.45% | 5.94% | 1.98% | 17.04% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 3,760 | 9,528 | 15,622 |
| val | 361 | 895 | 1,317 |
| test | 355 | 796 | 1,199 |

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
| person | 23,917 | 20.75% |
| date | 10,702 | 9.29% |
| location | 8,087 | 7.02% |
| organization | 5,779 | 5.01% |
| geopolitical entity | 5,584 | 4.85% |
| job title | 5,238 | 4.54% |
| company | 4,253 | 3.69% |
| media outlet | 4,098 | 3.56% |
| sports team | 3,356 | 2.91% |
| duration | 2,843 | 2.47% |
| money | 2,745 | 2.38% |
| government agency | 2,671 | 2.32% |
| event | 2,448 | 2.12% |
| quantity | 2,388 | 2.07% |
| facility | 2,382 | 2.07% |
| occupation | 1,825 | 1.58% |
| age | 1,666 | 1.45% |
| percent | 1,485 | 1.29% |
| product | 1,432 | 1.24% |
| nationality | 1,258 | 1.09% |
| time | 1,204 | 1.04% |
| educational institution | 1,195 | 1.04% |
| region | 1,170 | 1.02% |
| crime | 985 | 0.85% |
| cardinal | 950 | 0.82% |
| road | 621 | 0.54% |
| disease | 589 | 0.51% |
| family relationship | 586 | 0.51% |
| political party | 577 | 0.50% |
| sports competition | 559 | 0.49% |
| work of art | 524 | 0.45% |
| vehicle | 509 | 0.44% |
| film | 474 | 0.41% |
| law | 435 | 0.38% |
| software | 431 | 0.37% |
| brand | 402 | 0.35% |
| measurement | 401 | 0.35% |
| non-profit organization | 388 | 0.34% |
| song | 386 | 0.33% |
| award | 351 | 0.30% |
| political affiliation | 330 | 0.29% |
| ordinal | 328 | 0.28% |
| distance | 327 | 0.28% |
| ethnic group | 315 | 0.27% |
| weapon | 309 | 0.27% |
| anatomical structure | 280 | 0.24% |
| chemical | 278 | 0.24% |
| address | 274 | 0.24% |
| team role | 256 | 0.22% |
| military unit | 253 | 0.22% |
| military rank | 229 | 0.20% |
| symptom | 225 | 0.20% |
| holiday | 221 | 0.19% |
| publication | 192 | 0.17% |
| landmark | 182 | 0.16% |
| organism | 180 | 0.16% |
| medical procedure | 178 | 0.15% |
| body of water | 167 | 0.14% |
| book | 166 | 0.14% |
| weight | 156 | 0.14% |
| religious group | 145 | 0.13% |
| species | 145 | 0.13% |
| honorific | 134 | 0.12% |
| temperature | 127 | 0.11% |
| frequency | 125 | 0.11% |
| market index | 120 | 0.10% |
| video game | 118 | 0.10% |
| hardware component | 112 | 0.10% |
| sentence | 111 | 0.10% |
| academic degree | 109 | 0.09% |
| criminal organization | 109 | 0.09% |
| language | 109 | 0.09% |
| currency | 108 | 0.09% |
| medication | 104 | 0.09% |
| operating system | 101 | 0.09% |
| speed | 100 | 0.09% |
| area | 94 | 0.08% |
| aircraft | 87 | 0.08% |
| stock ticker | 87 | 0.08% |
| vessel | 86 | 0.07% |
| financial instrument | 81 | 0.07% |
| natural disaster | 81 | 0.07% |
| war | 77 | 0.07% |
| pseudonym | 76 | 0.07% |
| era | 75 | 0.07% |
| tax | 74 | 0.06% |
| regulatory body | 66 | 0.06% |
| treaty | 66 | 0.06% |
| airport | 54 | 0.05% |
| fiscal quarter | 53 | 0.05% |
| conference | 46 | 0.04% |
| protocol | 43 | 0.04% |
| regulation | 42 | 0.04% |
| contract | 41 | 0.04% |
| license | 34 | 0.03% |
| trade union | 33 | 0.03% |
| witness | 32 | 0.03% |
| model number | 29 | 0.03% |
| programming language | 29 | 0.03% |
| court case | 29 | 0.03% |
| mountain | 24 | 0.02% |
| file format | 21 | 0.02% |
| mineral | 18 | 0.02% |
| postal code | 18 | 0.02% |
| pathogen | 15 | 0.01% |
| vaccine | 13 | 0.01% |
| verdict | 12 | 0.01% |
| gene | 10 | 0.01% |
| interest rate | 10 | 0.01% |
| protein | 9 | 0.01% |
| border | 9 | 0.01% |
| api | 8 | 0.01% |
| diagnosis | 8 | 0.01% |
| framework | 7 | 0.01% |
| database | 6 | 0.01% |
| deadline | 6 | 0.01% |
| clinical trial | 6 | 0.01% |
| algorithm | 5 | 0.00% |
| allergen | 3 | 0.00% |
| dosage | 3 | 0.00% |
| budget line | 2 | 0.00% |
| credit rating | 2 | 0.00% |
| patent | 1 | 0.00% |
| voltage | 1 | 0.00% |
| security vulnerability | 1 | 0.00% |

#### val

| label | count | share |
| --- | --- | --- |
| person | 2,490 | 21.21% |
| date | 1,100 | 9.37% |
| location | 796 | 6.78% |
| geopolitical entity | 597 | 5.08% |
| job title | 525 | 4.47% |
| organization | 508 | 4.33% |
| media outlet | 442 | 3.76% |
| company | 380 | 3.24% |
| sports team | 375 | 3.19% |
| duration | 301 | 2.56% |
| government agency | 285 | 2.43% |
| facility | 260 | 2.21% |
| event | 254 | 2.16% |
| quantity | 244 | 2.08% |
| money | 244 | 2.08% |
| age | 231 | 1.97% |
| occupation | 194 | 1.65% |
| product | 170 | 1.45% |
| nationality | 161 | 1.37% |
| crime | 119 | 1.01% |
| time | 117 | 1.00% |
| percent | 113 | 0.96% |
| educational institution | 109 | 0.93% |
| region | 95 | 0.81% |
| road | 89 | 0.76% |
| film | 72 | 0.61% |
| political party | 70 | 0.60% |
| sports competition | 64 | 0.55% |
| family relationship | 62 | 0.53% |
| disease | 62 | 0.53% |
| cardinal | 60 | 0.51% |
| work of art | 52 | 0.44% |
| distance | 52 | 0.44% |
| software | 48 | 0.41% |
| ethnic group | 43 | 0.37% |
| non-profit organization | 41 | 0.35% |
| military rank | 37 | 0.32% |
| award | 34 | 0.29% |
| vehicle | 33 | 0.28% |
| symptom | 29 | 0.25% |
| anatomical structure | 29 | 0.25% |
| measurement | 29 | 0.25% |
| law | 28 | 0.24% |
| video game | 28 | 0.24% |
| ordinal | 28 | 0.24% |
| address | 26 | 0.22% |
| song | 25 | 0.21% |
| holiday | 25 | 0.21% |
| weapon | 25 | 0.21% |
| chemical | 25 | 0.21% |
| species | 25 | 0.21% |
| brand | 24 | 0.20% |
| military unit | 24 | 0.20% |
| criminal organization | 22 | 0.19% |
| medication | 21 | 0.18% |
| book | 19 | 0.16% |
| medical procedure | 18 | 0.15% |
| body of water | 18 | 0.15% |
| language | 17 | 0.14% |
| weight | 17 | 0.14% |
| area | 16 | 0.14% |
| sentence | 16 | 0.14% |
| team role | 16 | 0.14% |
| honorific | 14 | 0.12% |
| tax | 14 | 0.12% |
| landmark | 13 | 0.11% |
| financial instrument | 13 | 0.11% |
| operating system | 12 | 0.10% |
| publication | 12 | 0.10% |
| frequency | 12 | 0.10% |
| regulatory body | 11 | 0.09% |
| political affiliation | 10 | 0.09% |
| religious group | 10 | 0.09% |
| organism | 10 | 0.09% |
| vessel | 10 | 0.09% |
| era | 9 | 0.08% |
| war | 8 | 0.07% |
| hardware component | 8 | 0.07% |
| mountain | 8 | 0.07% |
| academic degree | 7 | 0.06% |
| conference | 7 | 0.06% |
| aircraft | 6 | 0.05% |
| temperature | 6 | 0.05% |
| pseudonym | 6 | 0.05% |
| speed | 6 | 0.05% |
| stock ticker | 5 | 0.04% |
| airport | 5 | 0.04% |
| mineral | 5 | 0.04% |
| fiscal quarter | 4 | 0.03% |
| natural disaster | 3 | 0.03% |
| file format | 3 | 0.03% |
| witness | 3 | 0.03% |
| trade union | 2 | 0.02% |
| treaty | 2 | 0.02% |
| protocol | 2 | 0.02% |
| clinical trial | 2 | 0.02% |
| allergen | 2 | 0.02% |
| license | 2 | 0.02% |
| model number | 1 | 0.01% |
| currency | 1 | 0.01% |
| court case | 1 | 0.01% |
| postal code | 1 | 0.01% |
| voltage | 1 | 0.01% |
| pathogen | 1 | 0.01% |
| vaccine | 1 | 0.01% |
| market index | 1 | 0.01% |
| verdict | 1 | 0.01% |
| database | 1 | 0.01% |
| interest rate | 1 | 0.01% |

#### test

| label | count | share |
| --- | --- | --- |
| person | 2,491 | 21.54% |
| date | 1,075 | 9.29% |
| location | 738 | 6.38% |
| job title | 546 | 4.72% |
| organization | 528 | 4.56% |
| geopolitical entity | 506 | 4.37% |
| sports team | 472 | 4.08% |
| company | 414 | 3.58% |
| media outlet | 408 | 3.53% |
| government agency | 309 | 2.67% |
| duration | 303 | 2.62% |
| money | 301 | 2.60% |
| facility | 277 | 2.39% |
| quantity | 242 | 2.09% |
| event | 224 | 1.94% |
| occupation | 173 | 1.50% |
| age | 170 | 1.47% |
| percent | 129 | 1.12% |
| time | 129 | 1.12% |
| product | 129 | 1.12% |
| crime | 127 | 1.10% |
| nationality | 122 | 1.05% |
| region | 121 | 1.05% |
| educational institution | 102 | 0.88% |
| family relationship | 95 | 0.82% |
| road | 70 | 0.61% |
| political party | 65 | 0.56% |
| cardinal | 63 | 0.54% |
| work of art | 59 | 0.51% |
| sports competition | 57 | 0.49% |
| song | 55 | 0.48% |
| film | 44 | 0.38% |
| award | 41 | 0.35% |
| team role | 39 | 0.34% |
| non-profit organization | 39 | 0.34% |
| vehicle | 39 | 0.34% |
| distance | 38 | 0.33% |
| disease | 36 | 0.31% |
| address | 34 | 0.29% |
| brand | 34 | 0.29% |
| anatomical structure | 30 | 0.26% |
| measurement | 30 | 0.26% |
| holiday | 28 | 0.24% |
| law | 27 | 0.23% |
| language | 26 | 0.22% |
| medical procedure | 23 | 0.20% |
| software | 23 | 0.20% |
| ordinal | 22 | 0.19% |
| military unit | 22 | 0.19% |
| symptom | 21 | 0.18% |
| species | 19 | 0.16% |
| political affiliation | 19 | 0.16% |
| chemical | 19 | 0.16% |
| military rank | 19 | 0.16% |
| ethnic group | 18 | 0.16% |
| religious group | 17 | 0.15% |
| weight | 17 | 0.15% |
| era | 16 | 0.14% |
| book | 16 | 0.14% |
| publication | 16 | 0.14% |
| academic degree | 15 | 0.13% |
| organism | 15 | 0.13% |
| honorific | 15 | 0.13% |
| natural disaster | 13 | 0.11% |
| body of water | 13 | 0.11% |
| frequency | 13 | 0.11% |
| weapon | 13 | 0.11% |
| medication | 12 | 0.10% |
| video game | 11 | 0.10% |
| fiscal quarter | 9 | 0.08% |
| landmark | 9 | 0.08% |
| area | 9 | 0.08% |
| currency | 8 | 0.07% |
| sentence | 8 | 0.07% |
| aircraft | 8 | 0.07% |
| temperature | 8 | 0.07% |
| criminal organization | 7 | 0.06% |
| regulatory body | 7 | 0.06% |
| market index | 7 | 0.06% |
| war | 7 | 0.06% |
| hardware component | 6 | 0.05% |
| operating system | 6 | 0.05% |
| tax | 5 | 0.04% |
| mineral | 5 | 0.04% |
| protocol | 5 | 0.04% |
| airport | 5 | 0.04% |
| vessel | 5 | 0.04% |
| speed | 4 | 0.03% |
| contract | 4 | 0.03% |
| treaty | 4 | 0.03% |
| pseudonym | 4 | 0.03% |
| border | 4 | 0.03% |
| financial instrument | 3 | 0.03% |
| trade union | 3 | 0.03% |
| clinical trial | 3 | 0.03% |
| stock ticker | 2 | 0.02% |
| witness | 2 | 0.02% |
| algorithm | 2 | 0.02% |
| file format | 2 | 0.02% |
| license | 2 | 0.02% |
| model number | 2 | 0.02% |
| regulation | 2 | 0.02% |
| protein | 2 | 0.02% |
| mountain | 1 | 0.01% |
| conference | 1 | 0.01% |
| diagnosis | 1 | 0.01% |
| court case | 1 | 0.01% |
| voltage | 1 | 0.01% |
| pathogen | 1 | 0.01% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| works_for | 590 | 17.40% |
| member_of | 349 | 10.29% |
| affiliated_with | 275 | 8.11% |
| located_in | 271 | 7.99% |
| employer_of | 259 | 7.64% |
| leader_of | 247 | 7.28% |
| part_of | 149 | 4.39% |
| based_in | 114 | 3.36% |
| relative_of | 73 | 2.15% |
| headquartered_in | 68 | 2.01% |
| spouse_of | 58 | 1.71% |
| charged_with | 58 | 1.71% |
| participated_in | 54 | 1.59% |
| child_of | 51 | 1.50% |
| parent_of | 49 | 1.45% |
| represents | 42 | 1.24% |
| authored_by | 41 | 1.21% |
| colleague_of | 40 | 1.18% |
| reports_to | 37 | 1.09% |
| owns | 35 | 1.03% |
| operates_in | 35 | 1.03% |
| subsidiary_of | 34 | 1.00% |
| produced_by | 34 | 1.00% |
| competitor_of | 30 | 0.88% |
| founded | 29 | 0.86% |
| advisor_to | 26 | 0.77% |
| manufactured_by | 24 | 0.71% |
| developed_by | 23 | 0.68% |
| sibling_of | 22 | 0.65% |
| citizen_of | 22 | 0.65% |
| board_member_of | 22 | 0.65% |
| opposed_to | 22 | 0.65% |
| won | 16 | 0.47% |
| published_by | 15 | 0.44% |
| acquired | 15 | 0.44% |
| directed_by | 13 | 0.38% |
| convicted_of | 12 | 0.35% |
| designed_by | 11 | 0.32% |
| sponsored_by | 11 | 0.32% |
| composed_by | 10 | 0.29% |
| born_in | 9 | 0.27% |
| named_after | 9 | 0.27% |
| successor_of | 9 | 0.27% |
| member_state_of | 9 | 0.27% |
| supplier_of | 7 | 0.21% |
| allied_with | 7 | 0.21% |
| party_to | 6 | 0.18% |
| died_in | 5 | 0.15% |
| sued_by | 5 | 0.15% |
| derived_from | 4 | 0.12% |
| customer_of | 4 | 0.12% |
| invested_in | 4 | 0.12% |
| adjacent_to | 3 | 0.09% |
| capital_of | 3 | 0.09% |
| depends_on | 3 | 0.09% |
| causes | 2 | 0.06% |
| mentor_of | 2 | 0.06% |
| regulated_by | 2 | 0.06% |
| measured_in | 2 | 0.06% |
| predecessor_of | 2 | 0.06% |
| signatory_of | 2 | 0.06% |
| compatible_with | 2 | 0.06% |
| testified_against | 1 | 0.03% |
| distributor_of | 1 | 0.03% |
| treats | 1 | 0.03% |
| borders | 1 | 0.03% |

#### val

| label | count | share |
| --- | --- | --- |
| works_for | 67 | 21.20% |
| located_in | 30 | 9.49% |
| member_of | 28 | 8.86% |
| leader_of | 24 | 7.59% |
| employer_of | 22 | 6.96% |
| affiliated_with | 17 | 5.38% |
| part_of | 14 | 4.43% |
| based_in | 8 | 2.53% |
| spouse_of | 7 | 2.22% |
| developed_by | 7 | 2.22% |
| convicted_of | 7 | 2.22% |
| produced_by | 7 | 2.22% |
| manufactured_by | 6 | 1.90% |
| subsidiary_of | 6 | 1.90% |
| child_of | 6 | 1.90% |
| composed_by | 6 | 1.90% |
| relative_of | 6 | 1.90% |
| founded | 5 | 1.58% |
| supplier_of | 4 | 1.27% |
| operates_in | 4 | 1.27% |
| owns | 3 | 0.95% |
| successor_of | 3 | 0.95% |
| charged_with | 3 | 0.95% |
| reports_to | 2 | 0.63% |
| headquartered_in | 2 | 0.63% |
| citizen_of | 2 | 0.63% |
| competitor_of | 2 | 0.63% |
| parent_of | 2 | 0.63% |
| allied_with | 2 | 0.63% |
| predecessor_of | 1 | 0.32% |
| causes | 1 | 0.32% |
| capital_of | 1 | 0.32% |
| supersedes | 1 | 0.32% |
| died_in | 1 | 0.32% |
| represents | 1 | 0.32% |
| colleague_of | 1 | 0.32% |
| borders | 1 | 0.32% |
| born_in | 1 | 0.32% |
| authored_by | 1 | 0.32% |
| designed_by | 1 | 0.32% |
| invested_in | 1 | 0.32% |
| acquired | 1 | 0.32% |
| advisor_to | 1 | 0.32% |

#### test

| label | count | share |
| --- | --- | --- |
| works_for | 68 | 24.64% |
| located_in | 32 | 11.59% |
| affiliated_with | 25 | 9.06% |
| employer_of | 24 | 8.70% |
| member_of | 22 | 7.97% |
| based_in | 15 | 5.43% |
| leader_of | 12 | 4.35% |
| part_of | 12 | 4.35% |
| relative_of | 8 | 2.90% |
| subsidiary_of | 5 | 1.81% |
| developed_by | 4 | 1.45% |
| child_of | 4 | 1.45% |
| parent_of | 4 | 1.45% |
| participated_in | 4 | 1.45% |
| headquartered_in | 3 | 1.09% |
| opposed_to | 3 | 1.09% |
| published_by | 3 | 1.09% |
| founded | 3 | 1.09% |
| sponsored_by | 3 | 1.09% |
| reports_to | 2 | 0.72% |
| operates_in | 2 | 0.72% |
| manufactured_by | 2 | 0.72% |
| produced_by | 2 | 0.72% |
| colleague_of | 2 | 0.72% |
| sold_to | 1 | 0.36% |
| allied_with | 1 | 0.36% |
| designed_by | 1 | 0.36% |
| board_member_of | 1 | 0.36% |
| charged_with | 1 | 0.36% |
| directed_by | 1 | 0.36% |
| authored_by | 1 | 0.36% |
| capital_of | 1 | 0.36% |
| owns | 1 | 0.36% |
| derived_from | 1 | 0.36% |
| advisor_to | 1 | 0.36% |
| predecessor_of | 1 | 0.36% |

### Classifications

**train, actionability.** Majority class 91.5% of 1,096 answers. The marginal is `true_label`, not the menu.

#### train actionability

| label | count | share |
| --- | --- | --- |
| informational | 1,003 | 91.51% |
| advisory | 72 | 6.57% |
| requires action | 21 | 1.92% |

**train, audience.** Majority class 97.4% of 1,207 answers. The marginal is `true_label`, not the menu.

#### train audience

| label | count | share |
| --- | --- | --- |
| general public | 1,176 | 97.43% |
| expert | 27 | 2.24% |
| regulatory | 4 | 0.33% |

**train, certainty.** Majority class 90.3% of 1,212 answers. The marginal is `true_label`, not the menu.

#### train certainty

| label | count | share |
| --- | --- | --- |
| asserted | 1,095 | 90.35% |
| hedged | 109 | 8.99% |
| speculative | 7 | 0.58% |
| denied | 1 | 0.08% |

**train, document_genre.** Majority class 79.5% of 1,158 answers. The marginal is `true_label`, not the menu.

#### train document_genre

| label | count | share |
| --- | --- | --- |
| news report | 921 | 79.53% |
| announcement | 123 | 10.62% |
| opinion | 76 | 6.56% |
| review | 16 | 1.38% |
| advertisement | 12 | 1.04% |
| correspondence | 6 | 0.52% |
| technical documentation | 4 | 0.35% |

**train, formality.** Majority class 66.0% of 1,115 answers. The marginal is `true_label`, not the menu.

#### train formality

| label | count | share |
| --- | --- | --- |
| formal | 736 | 66.01% |
| informal | 379 | 33.99% |

**train, language_register.** Majority class 81.7% of 1,070 answers. The marginal is `true_label`, not the menu.

#### train language_register

| label | count | share |
| --- | --- | --- |
| plain | 874 | 81.68% |
| colloquial | 135 | 12.62% |
| technical | 47 | 4.39% |
| legal | 14 | 1.31% |

**train, risk_level.** Majority class 63.4% of 1,132 answers. The marginal is `true_label`, not the menu.

#### train risk_level

| label | count | share |
| --- | --- | --- |
| none | 718 | 63.43% |
| low | 229 | 20.23% |
| moderate | 137 | 12.10% |
| high | 48 | 4.24% |

**train, sentiment.** Majority class 39.6% of 1,194 answers. The marginal is `true_label`, not the menu.

#### train sentiment

| label | count | share |
| --- | --- | --- |
| neutral | 473 | 39.61% |
| positive | 372 | 31.16% |
| negative | 349 | 29.23% |

**train, subjectivity.** Majority class 78.4% of 975 answers. The marginal is `true_label`, not the menu.

#### train subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 764 | 78.36% |
| subjective | 211 | 21.64% |

**train, temporal_orientation.** Majority class 49.8% of 1,153 answers. The marginal is `true_label`, not the menu.

#### train temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 574 | 49.78% |
| present | 431 | 37.38% |
| future | 122 | 10.58% |
| timeless | 26 | 2.25% |

**train, topic.** Majority class 15.6% of 2,188 answers. The marginal is `true_label`, not the menu.

#### train topic

| label | count | share |
| --- | --- | --- |
| politics | 342 | 15.63% |
| business | 273 | 12.48% |
| legal | 256 | 11.70% |
| entertainment | 238 | 10.88% |
| world | 210 | 9.60% |
| sports | 199 | 9.10% |
| health | 123 | 5.62% |
| finance | 106 | 4.84% |
| technology | 98 | 4.48% |
| environment | 92 | 4.20% |
| education | 72 | 3.29% |
| defense | 53 | 2.42% |
| travel | 43 | 1.97% |
| science | 42 | 1.92% |
| energy | 23 | 1.05% |
| agriculture | 18 | 0.82% |

**train, urgency.** Majority class 84.9% of 1,150 answers. The marginal is `true_label`, not the menu.

#### train urgency

| label | count | share |
| --- | --- | --- |
| routine | 976 | 84.87% |
| elevated | 154 | 13.39% |
| urgent | 16 | 1.39% |
| critical | 4 | 0.35% |

**val, actionability.** Majority class 91.7% of 109 answers. The marginal is `true_label`, not the menu.

#### val actionability

| label | count | share |
| --- | --- | --- |
| informational | 100 | 91.74% |
| advisory | 6 | 5.50% |
| requires action | 3 | 2.75% |

**val, audience.** Majority class 99.2% of 124 answers. The marginal is `true_label`, not the menu.

#### val audience

| label | count | share |
| --- | --- | --- |
| general public | 123 | 99.19% |
| regulatory | 1 | 0.81% |

**val, certainty.** Majority class 92.1% of 151 answers. The marginal is `true_label`, not the menu.

#### val certainty

| label | count | share |
| --- | --- | --- |
| asserted | 139 | 92.05% |
| hedged | 12 | 7.95% |

**val, document_genre.** Majority class 81.7% of 131 answers. The marginal is `true_label`, not the menu.

#### val document_genre

| label | count | share |
| --- | --- | --- |
| news report | 107 | 81.68% |
| announcement | 11 | 8.40% |
| opinion | 10 | 7.63% |
| review | 3 | 2.29% |

**val, formality.** Majority class 57.4% of 108 answers. The marginal is `true_label`, not the menu.

#### val formality

| label | count | share |
| --- | --- | --- |
| formal | 62 | 57.41% |
| informal | 46 | 42.59% |

**val, language_register.** Majority class 87.3% of 118 answers. The marginal is `true_label`, not the menu.

#### val language_register

| label | count | share |
| --- | --- | --- |
| plain | 103 | 87.29% |
| colloquial | 11 | 9.32% |
| technical | 4 | 3.39% |

**val, risk_level.** Majority class 66.7% of 123 answers. The marginal is `true_label`, not the menu.

#### val risk_level

| label | count | share |
| --- | --- | --- |
| none | 82 | 66.67% |
| low | 23 | 18.70% |
| moderate | 14 | 11.38% |
| high | 4 | 3.25% |

**val, sentiment.** Majority class 37.5% of 136 answers. The marginal is `true_label`, not the menu.

#### val sentiment

| label | count | share |
| --- | --- | --- |
| negative | 51 | 37.50% |
| neutral | 44 | 32.35% |
| positive | 41 | 30.15% |

**val, subjectivity.** Majority class 78.1% of 96 answers. The marginal is `true_label`, not the menu.

#### val subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 75 | 78.12% |
| subjective | 21 | 21.88% |

**val, temporal_orientation.** Majority class 60.2% of 128 answers. The marginal is `true_label`, not the menu.

#### val temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 77 | 60.16% |
| present | 37 | 28.91% |
| future | 9 | 7.03% |
| timeless | 5 | 3.91% |

**val, topic.** Majority class 12.9% of 217 answers. The marginal is `true_label`, not the menu.

#### val topic

| label | count | share |
| --- | --- | --- |
| entertainment | 28 | 12.90% |
| business | 28 | 12.90% |
| politics | 28 | 12.90% |
| legal | 25 | 11.52% |
| sports | 21 | 9.68% |
| world | 21 | 9.68% |
| technology | 18 | 8.29% |
| health | 13 | 5.99% |
| finance | 12 | 5.53% |
| environment | 5 | 2.30% |
| travel | 4 | 1.84% |
| defense | 4 | 1.84% |
| science | 4 | 1.84% |
| energy | 3 | 1.38% |
| education | 3 | 1.38% |

**val, urgency.** Majority class 76.3% of 114 answers. The marginal is `true_label`, not the menu.

#### val urgency

| label | count | share |
| --- | --- | --- |
| routine | 87 | 76.32% |
| elevated | 22 | 19.30% |
| urgent | 4 | 3.51% |
| critical | 1 | 0.88% |

**test, actionability.** Majority class 91.0% of 134 answers. The marginal is `true_label`, not the menu.

#### test actionability

| label | count | share |
| --- | --- | --- |
| informational | 122 | 91.04% |
| advisory | 10 | 7.46% |
| requires action | 2 | 1.49% |

**test, audience.** Majority class 96.7% of 121 answers. The marginal is `true_label`, not the menu.

#### test audience

| label | count | share |
| --- | --- | --- |
| general public | 117 | 96.69% |
| expert | 3 | 2.48% |
| regulatory | 1 | 0.83% |

**test, certainty.** Majority class 90.5% of 116 answers. The marginal is `true_label`, not the menu.

#### test certainty

| label | count | share |
| --- | --- | --- |
| asserted | 105 | 90.52% |
| hedged | 11 | 9.48% |

**test, document_genre.** Majority class 83.0% of 135 answers. The marginal is `true_label`, not the menu.

#### test document_genre

| label | count | share |
| --- | --- | --- |
| news report | 112 | 82.96% |
| opinion | 13 | 9.63% |
| announcement | 7 | 5.19% |
| advertisement | 2 | 1.48% |
| review | 1 | 0.74% |

**test, formality.** Majority class 67.3% of 110 answers. The marginal is `true_label`, not the menu.

#### test formality

| label | count | share |
| --- | --- | --- |
| formal | 74 | 67.27% |
| informal | 36 | 32.73% |

**test, language_register.** Majority class 83.5% of 121 answers. The marginal is `true_label`, not the menu.

#### test language_register

| label | count | share |
| --- | --- | --- |
| plain | 101 | 83.47% |
| colloquial | 18 | 14.88% |
| legal | 1 | 0.83% |
| technical | 1 | 0.83% |

**test, risk_level.** Majority class 65.0% of 123 answers. The marginal is `true_label`, not the menu.

#### test risk_level

| label | count | share |
| --- | --- | --- |
| none | 80 | 65.04% |
| low | 24 | 19.51% |
| moderate | 16 | 13.01% |
| high | 3 | 2.44% |

**test, sentiment.** Majority class 41.4% of 116 answers. The marginal is `true_label`, not the menu.

#### test sentiment

| label | count | share |
| --- | --- | --- |
| positive | 48 | 41.38% |
| neutral | 37 | 31.90% |
| negative | 31 | 26.72% |

**test, subjectivity.** Majority class 78.6% of 103 answers. The marginal is `true_label`, not the menu.

#### test subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 81 | 78.64% |
| subjective | 22 | 21.36% |

**test, temporal_orientation.** Majority class 58.1% of 124 answers. The marginal is `true_label`, not the menu.

#### test temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 72 | 58.06% |
| present | 38 | 30.65% |
| future | 13 | 10.48% |
| timeless | 1 | 0.81% |

**test, topic.** Majority class 14.2% of 204 answers. The marginal is `true_label`, not the menu.

#### test topic

| label | count | share |
| --- | --- | --- |
| legal | 29 | 14.22% |
| sports | 27 | 13.24% |
| politics | 26 | 12.75% |
| world | 21 | 10.29% |
| entertainment | 21 | 10.29% |
| business | 17 | 8.33% |
| health | 16 | 7.84% |
| technology | 13 | 6.37% |
| education | 8 | 3.92% |
| environment | 6 | 2.94% |
| finance | 5 | 2.45% |
| science | 5 | 2.45% |
| defense | 4 | 1.96% |
| agriculture | 3 | 1.47% |
| travel | 2 | 0.98% |
| energy | 1 | 0.49% |

**test, urgency.** Majority class 89.9% of 129 answers. The marginal is `true_label`, not the menu.

#### test urgency

| label | count | share |
| --- | --- | --- |
| routine | 116 | 89.92% |
| elevated | 12 | 9.30% |
| urgent | 1 | 0.78% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | condition | 14 | 100.00% |
| clinical_finding | patient_group | 13 | 92.86% |
| clinical_finding | treatment | 12 | 85.71% |
| clinical_finding | outcome | 11 | 78.57% |
| clinical_finding | record_metadata | natural/condition ×14 |  |
| company_profile | name | 33 | 100.00% |
| company_profile | size | 33 | 100.00% |
| company_profile | industry | 29 | 87.88% |
| company_profile | headquarters | 18 | 54.55% |
| company_profile | founded | 6 | 18.18% |
| company_profile | record_metadata | natural/name ×33 |  |
| event_listing | name | 50 | 98.04% |
| event_listing | date | 49 | 96.08% |
| event_listing | venue | 43 | 84.31% |
| event_listing | organizer | 37 | 72.55% |
| event_listing | price | 20 | 39.22% |
| event_listing | record_metadata | natural/name ×47, natural/venue ×1 |  |
| financial_report | organization | 1 | 100.00% |
| financial_report | period | 1 | 100.00% |
| financial_report | revenue | 1 | 100.00% |
| financial_report | net_income | 1 | 100.00% |
| financial_report | record_metadata | natural/organization ×1 |  |
| incident_report | severity | 111 | 100.00% |
| incident_report | location | 109 | 98.20% |
| incident_report | date | 106 | 95.50% |
| incident_report | incident_type | 92 | 82.88% |
| incident_report | casualties | 68 | 61.26% |
| incident_report | record_metadata | natural/incident_type ×92, natural/location ×19 |  |
| legal_case | defendant | 20 | 100.00% |
| legal_case | status | 20 | 100.00% |
| legal_case | court | 16 | 80.00% |
| legal_case | plaintiff | 14 | 70.00% |
| legal_case | case_name | 7 | 35.00% |
| legal_case | record_metadata | natural/court ×11, natural/case_name ×7, natural/plaintiff ×2 |  |
| person_profile | name | 362 | 99.72% |
| person_profile | role | 339 | 93.39% |
| person_profile | employer | 285 | 78.51% |
| person_profile | location | 262 | 72.18% |
| person_profile | record_metadata | natural/name ×320, natural/location ×1 |  |
| product | name | 11 | 100.00% |
| product | brand | 11 | 100.00% |
| product | condition | 11 | 100.00% |
| product | price | 8 | 72.73% |
| product | record_metadata | natural/name ×10 |  |
| real_estate_listing | address | 1 | 100.00% |
| real_estate_listing | price | 1 | 100.00% |
| real_estate_listing | property_type | 1 | 100.00% |
| real_estate_listing | record_metadata | natural/address ×1 |  |
| research_paper | title | 2 | 100.00% |
| research_paper | authors | 2 | 100.00% |
| research_paper | topic | 2 | 100.00% |
| research_paper | venue | 1 | 50.00% |
| research_paper | year | 1 | 50.00% |
| research_paper | record_metadata | natural/title ×2 |  |
| software_release | product | 2 | 100.00% |
| software_release | platform | 2 | 100.00% |
| software_release | release_date | 1 | 50.00% |
| software_release | record_metadata | natural/product ×2 |  |
| transaction | item | 14 | 93.33% |
| transaction | amount | 14 | 93.33% |
| transaction | buyer | 10 | 66.67% |
| transaction | seller | 10 | 66.67% |
| transaction | date | 6 | 40.00% |
| transaction | record_metadata | natural/item ×14, natural/amount ×1 |  |

**val.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | condition | 1 | 100.00% |
| clinical_finding | patient_group | 1 | 100.00% |
| clinical_finding | treatment | 1 | 100.00% |
| clinical_finding | outcome | 1 | 100.00% |
| clinical_finding | record_metadata | natural/condition ×1 |  |
| company_profile | name | 3 | 100.00% |
| company_profile | industry | 3 | 100.00% |
| company_profile | size | 3 | 100.00% |
| company_profile | headquarters | 1 | 33.33% |
| company_profile | founded | 1 | 33.33% |
| company_profile | record_metadata | natural/name ×3 |  |
| event_listing | name | 3 | 100.00% |
| event_listing | venue | 3 | 100.00% |
| event_listing | date | 3 | 100.00% |
| event_listing | organizer | 2 | 66.67% |
| event_listing | price | 1 | 33.33% |
| event_listing | record_metadata | natural/name ×3 |  |
| incident_report | incident_type | 12 | 100.00% |
| incident_report | severity | 12 | 100.00% |
| incident_report | date | 10 | 83.33% |
| incident_report | location | 9 | 75.00% |
| incident_report | casualties | 4 | 33.33% |
| incident_report | record_metadata | natural/incident_type ×12 |  |
| person_profile | name | 43 | 100.00% |
| person_profile | role | 36 | 83.72% |
| person_profile | location | 32 | 74.42% |
| person_profile | employer | 31 | 72.09% |
| person_profile | record_metadata | natural/name ×41 |  |
| product | name | 4 | 100.00% |
| product | brand | 4 | 100.00% |
| product | condition | 4 | 100.00% |
| product | price | 2 | 50.00% |
| product | record_metadata | natural/name ×4 |  |
| transaction | item | 3 | 100.00% |
| transaction | amount | 3 | 100.00% |
| transaction | seller | 2 | 66.67% |
| transaction | buyer | 1 | 33.33% |
| transaction | date | 1 | 33.33% |
| transaction | record_metadata | natural/item ×3 |  |

**test.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| company_profile | name | 5 | 100.00% |
| company_profile | size | 5 | 100.00% |
| company_profile | headquarters | 4 | 80.00% |
| company_profile | industry | 3 | 60.00% |
| company_profile | record_metadata | natural/name ×5 |  |
| event_listing | name | 3 | 100.00% |
| event_listing | venue | 3 | 100.00% |
| event_listing | date | 3 | 100.00% |
| event_listing | organizer | 3 | 100.00% |
| event_listing | price | 1 | 33.33% |
| event_listing | record_metadata | natural/name ×3 |  |
| incident_report | date | 12 | 100.00% |
| incident_report | severity | 12 | 100.00% |
| incident_report | location | 11 | 91.67% |
| incident_report | incident_type | 10 | 83.33% |
| incident_report | casualties | 6 | 50.00% |
| incident_report | record_metadata | natural/incident_type ×10, natural/location ×2 |  |
| person_profile | name | 45 | 100.00% |
| person_profile | role | 37 | 82.22% |
| person_profile | location | 34 | 75.56% |
| person_profile | employer | 30 | 66.67% |
| person_profile | record_metadata | natural/name ×43 |  |
| product | name | 1 | 100.00% |
| product | brand | 1 | 100.00% |
| product | condition | 1 | 100.00% |
| product | record_metadata | natural/name ×1 |  |

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | llm_real |
| relations | llm_real |
| events | llm_real |
| classifications | llm_real |
| structures | llm_real |
