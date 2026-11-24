# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_haiku45_5k_coerced | train | 4,020 | 4,020 | 31,678 | 6,396 | 14,478 | 1,201 | 2,889 | 8,628 |
| synthetic_haiku45_5k_coerced | val | 496 | 496 | 3,954 | 787 | 1,771 | 171 | 342 | 973 |
| synthetic_haiku45_5k_coerced | test | 484 | 484 | 3,823 | 802 | 1,767 | 153 | 390 | 1,119 |

## synthetic_haiku45_5k_coerced

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4,020 | 4,020 | 271 | 305 | 408 | 31,678 | 6,396 | 14,478 | 1,201 | 2,889 | 8,628 |
| val | 496 | 496 | 270 | 306 | 382 | 3,954 | 787 | 1,771 | 171 | 342 | 973 |
| test | 484 | 484 | 268 | 308 | 367 | 3,823 | 802 | 1,767 | 153 | 390 | 1,119 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 4,020 | 496 | 484 | 5,000 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 484 |
| test_unique_inputs | 484 |
| train_records | 4,020 |
| train_unique_inputs | 4,020 |
| val_records | 496 |
| val_unique_inputs | 496 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0940 | 31,678 / 3,954 | quantity 5%→4%, company 3%→4%, region 3%→4%, facility 2%→1%, date 4%→5% |
| entities (per mention) | train / test | 0.0832 | 31,678 / 3,823 | organization 4%→4%, ordinal 1%→1%, government agency 2%→2%, non-profit organization 1%→2%, area 2%→2% |
| entities (per mention) | val / test | 0.1239 | 3,954 / 3,823 | company 4%→3%, quantity 4%→6%, government agency 1%→2%, region 4%→3%, geopolitical entity 4%→3% |
| relations (per instance) | train / val | 0.1256 | 6,396 / 787 | works_for 7%→5%, part_of 2%→4%, leader_of 2%→1%, located_in 11%→12%, distributor_of 1%→2% |
| relations (per instance) | train / test | 0.1087 | 6,396 / 802 | member_of 2%→1%, distributor_of 1%→2%, supplier_of 1%→2%, borders 0%→1%, developed_by 3%→4% |
| relations (per instance) | val / test | 0.1430 | 787 / 802 | works_for 5%→8%, colleague_of 1%→3%, operates_in 7%→6%, developed_by 3%→4%, part_of 4%→2% |
| event types (per trigger mention) | train / val | 0.1906 WARN | 2,906 / 342 | Contact.Meet 4%→8%, Tech.Release 6%→3%, Personnel.EndPosition 3%→1%, Disaster.NaturalDisaster 2%→3%, Contact.Communicate 5%→7% |
| event types (per trigger mention) | train / test | 0.1404 | 2,906 / 391 | Personnel.StartPosition 6%→4%, Medical.Treat 3%→2%, Research.Publish 5%→3%, Contact.Communicate 5%→4%, Movement.Arrive 1%→2% |
| event types (per trigger mention) | val / test | 0.2502 WARN | 342 / 391 | Contact.Meet 8%→5%, Contact.Communicate 7%→4%, Tech.Release 3%→6%, Business.LaunchProduct 4%→6%, Personnel.StartPosition 6%→4% |
| event roles (per argument) | train / val | 0.1009 | 8,644 / 973 | Place 12%→14%, Participant 3%→4%, Product 3%→2%, Researcher 1%→2%, Communicator 2%→2% |
| event roles (per argument) | train / test | 0.0849 | 8,644 / 1,120 | Place 12%→13%, Participant 3%→4%, Recipient 3%→2%, Organization 8%→9%, Author 1%→1% |
| event roles (per argument) | val / test | 0.1204 | 973 / 1,120 | Product 2%→4%, Recipient 4%→2%, Communicator 2%→1%, Researcher 2%→1%, Publication 1%→0% |
| classifications (true_label) | train / val | 0.0753 | 14,478 / 1,771 | risk_level: low 3%→4%, audience: general public 4%→5%, formality: formal 7%→6%, topic: technology 2%→3%, topic: environment 1%→1% |
| classifications (true_label) | train / test | 0.0606 | 14,478 / 1,767 | actionability: informational 3%→4%, certainty: asserted 7%→6%, sentiment: negative 2%→3%, subjectivity: objective 7%→6%, formality: formal 7%→6% |
| classifications (true_label) | val / test | 0.1051 | 1,771 / 1,767 | risk_level: low 4%→2%, certainty: asserted 7%→6%, sentiment: negative 2%→3%, topic: health 2%→3%, audience: general public 5%→4% |

Thin support in val, entities (per mention): 46 label(s) under 10 gold. political party ×1, tax ×2, medical procedure ×2, criminal organization ×2, sentence ×2, treaty ×2, credit rating ×3, trade union ×3, natural disaster ×3, weapon ×3, aircraft ×4, age ×4, sports competition ×4, verdict ×5, military rank ×5, crime ×5, market index ×5, vessel ×5, vehicle ×6, book ×6, interest rate ×6, pathogen ×6, work of art ×6, weight ×6, patent ×6, border ×6, team role ×6, era ×6, allergen ×7, award ×7, stock ticker ×7, license ×7, contract ×7, security vulnerability ×7, mountain ×7, file format ×7, ethnic group ×8, publication ×8, academic degree ×8, witness ×8, species ×9, gene ×9, nationality ×9, law ×9, financial instrument ×9, clinical trial ×9.
Thin support in val, relations (per instance): 40 label(s) under 10 gold. named_after ×1, causes ×1, signatory_of ×1, predecessor_of ×1, composed_by ×1, testified_against ×1, depends_on ×2, derived_from ×2, supersedes ×2, borders ×2, member_state_of ×2, mentor_of ×3, capital_of ×3, implements ×3, citizen_of ×3, won ×3, charged_with ×3, licensed_to ×4, participated_in ×4, sold_to ×4, directed_by ×4, opposed_to ×4, authored_by ×5, published_by ×5, board_member_of ×5, party_to ×5, born_in ×6, represents ×6, merged_with ×6, subsidiary_of ×6, invested_in ×7, owns ×7, compatible_with ×8, sponsored_by ×8, treats ×9, founded ×9, colleague_of ×9, designed_by ×9, allied_with ×9, adjacent_to ×9.
Thin support in val, event types (per trigger mention): 36 label(s) under 10 gold. Personnel.Elect ×1, Business.EndOrg ×1, Justice.Appeal ×1, Justice.Sue ×1, Movement.Deport ×1, Transaction.Lend ×1, Business.DeclareBankruptcy ×1, Government.Vote ×1, Government.EnactLaw ×1, Conflict.Demonstrate ×1, Conflict.Ceasefire ×1, Justice.Fine ×2, Justice.Sentence ×2, Life.Die ×2, Movement.Arrive ×3, Business.Strike ×3, Award.ReceiveAward ×3, Personnel.EndPosition ×3, Movement.Transport ×3, Life.Evacuate ×4, Disaster.Outbreak ×4, Movement.Migrate ×4, Transaction.Donate ×4, Justice.TrialHearing ×4, Personnel.Nominate ×5, Disaster.Accident ×5, Life.Injure ×5, Conflict.Attack ×5, Justice.Arrest ×5, Justice.ChargeIndict ×5, Tech.Breach ×5, Transaction.TransferOwnership ×5, Government.Sanction ×6, Disaster.Rescue ×7, Medical.Treat ×8, Contact.Negotiate ×9.
Unseen in train, present in val, event roles (per argument): Beneficiary ×1.
Thin support in val, event roles (per argument): 39 label(s) under 10 gold. Award ×1, Appellant ×1, Beneficiary ×1, Plaintiff ×1, Attacker ×1, Lender ×1, Borrower ×1, Voter ×1, Proposal ×1, Law ×1, Demonstrator ×1, Version ×2, Affected ×2, Evacuee ×2, Venue ×2, Instrument ×2, Migrant ×2, Sentence ×2, Seller ×2, Traveler ×3, Striker ×3, Disease ×3, Entity ×3, Data ×3, Donor ×3, Reason ×4, Buyer ×4, Victim ×5, Prosecutor ×5, Price ×5, Topic ×6, Rescuer ×6, Treatment ×6, Adjudicator ×7, Artifact ×7, Destination ×8, Funder ×9, Quantity ×9, Physician ×9.
Thin support in val, classifications (true_label): 16 label(s) under 10 gold. language_register: colloquial ×1, topic: agriculture ×1, topic: entertainment ×2, urgency: urgent ×3, topic: energy ×3, formality: informal ×3, document_genre: opinion ×4, topic: education ×4, temporal_orientation: timeless ×5, topic: politics ×5, topic: sports ×6, topic: defense ×7, topic: world ×7, temporal_orientation: future ×8, language_register: legal ×8, actionability: advisory ×9.
Thin support in test, entities (per mention): 40 label(s) under 10 gold. pseudonym ×1, holiday ×1, ethnic group ×1, criminal organization ×1, military rank ×2, trade union ×2, film ×2, military unit ×2, interest rate ×2, award ×2, language ×3, verdict ×3, sports competition ×3, treaty ×3, contract ×3, book ×4, conference ×4, border ×4, work of art ×4, license ×5, vessel ×5, pathogen ×5, tax ×5, court case ×5, credit rating ×5, allergen ×5, weapon ×6, mountain ×6, crime ×6, market index ×6, vehicle ×7, gene ×7, team role ×8, medical procedure ×8, weight ×8, era ×8, frequency ×8, security vulnerability ×9, organism ×9, witness ×9.
Thin support in test, relations (per instance): 44 label(s) under 10 gold. directed_by ×1, won ×1, child_of ×1, signatory_of ×1, parent_of ×1, measured_in ×1, spouse_of ×1, sold_to ×2, successor_of ×2, testified_against ×2, died_in ×2, interacts_with ×2, represents ×2, predecessor_of ×2, depends_on ×2, treats ×2, capital_of ×3, member_state_of ×3, owns ×3, supersedes ×3, authored_by ×3, derived_from ×3, participated_in ×3, mentor_of ×4, citizen_of ×4, board_member_of ×4, sued_by ×5, licensed_to ×5, invested_in ×5, born_in ×5, merged_with ×6, adjacent_to ×6, advisor_to ×6, published_by ×6, implements ×6, designed_by ×7, subsidiary_of ×7, allied_with ×7, borders ×7, charged_with ×7, party_to ×8, produced_by ×8, sponsored_by ×8, acquired ×9.
Thin support in test, event types (per trigger mention): 39 label(s) under 10 gold. Business.DeclareBankruptcy ×1, Business.Strike ×1, Life.Divorce ×1, Movement.Migrate ×1, Government.Vote ×2, Conflict.Demonstrate ×2, Justice.Sentence ×2, Government.Protest ×2, Life.Evacuate ×3, Personnel.Nominate ×3, Conflict.Ceasefire ×3, Transaction.Donate ×3, Disaster.Rescue ×3, Government.EnactLaw ×3, Personnel.Elect ×3, Justice.Arrest ×3, Justice.Sue ×3, Justice.Appeal ×4, Justice.ChargeIndict ×4, Life.Die ×4, Justice.TrialHearing ×4, Life.BeBorn ×4, Government.Sanction ×5, Life.Injure ×5, Sport.Compete ×5, Transaction.TransferOwnership ×5, Movement.Transport ×5, Medical.Treat ×6, Transaction.Lend ×6, Business.Layoff ×7, Disaster.Outbreak ×7, Transaction.TransferMoney ×8, Award.ReceiveAward ×8, Business.MergeOrg ×8, Disaster.NaturalDisaster ×8, Movement.Arrive ×9, Disaster.Accident ×9, Business.EndOrg ×9, Conflict.Attack ×9.
Thin support in test, event roles (per argument): 47 label(s) under 10 gold. Venue ×1, Reason ×1, Version ×1, Voter ×1, Striker ×1, Rescuer ×1, Sentence ×1, Migrant ×1, Vehicle ×2, Proposal ×2, Mediator ×2, Plaintiff ×2, Competition ×3, Awarder ×3, Evacuee ×3, Donor ×3, Law ×3, Publication ×4, Treatment ×4, Instrument ×4, Lender ×4, Appellant ×4, Seller ×4, Price ×4, Party ×4, Demonstrator ×4, Physician ×5, Result ×5, Buyer ×5, Artifact ×5, Quantity ×5, Attacker ×5, Disease ×5, Affected ×5, Borrower ×6, Award ×6, Prosecutor ×6, Patient ×7, Giver ×7, Competitor ×7, Traveler ×7, Origin ×7, Topic ×7, Data ×7, Type ×7, Author ×8, Diagnosis ×9.
Thin support in test, classifications (true_label): 14 label(s) under 10 gold. topic: entertainment ×1, urgency: urgent ×1, temporal_orientation: timeless ×3, certainty: hedged ×3, topic: sports ×4, formality: informal ×4, topic: agriculture ×5, document_genre: opinion ×6, document_genre: advertisement ×7, topic: energy ×7, document_genre: review ×8, temporal_orientation: future ×8, topic: education ×8, topic: travel ×8.

### Events

**train.** 2,906 instances in 2,191 records, 0 triggerless. Trigger gold 2,889. Argument gold 8,628. Arguments per instance, mean 2.97. Trigger length in whitespace tokens, mean 1.28, single-token share 82.3%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Personnel.StartPosition | 178 | 6.13% |
| Tech.Release | 164 | 5.64% |
| Contact.Communicate | 156 | 5.37% |
| Business.LaunchProduct | 151 | 5.20% |
| Research.Fund | 143 | 4.92% |
| Research.Discover | 133 | 4.58% |
| Research.Publish | 133 | 4.58% |
| Contact.Meet | 112 | 3.85% |
| Business.StartOrg | 104 | 3.58% |
| Medical.Treat | 91 | 3.13% |
| Transaction.TransferMoney | 87 | 2.99% |
| Personnel.EndPosition | 82 | 2.82% |
| Medical.Diagnose | 82 | 2.82% |
| Transaction.TransferOwnership | 72 | 2.48% |
| Contact.Negotiate | 67 | 2.31% |
| Disaster.Outbreak | 66 | 2.27% |
| Award.ReceiveAward | 63 | 2.17% |
| Tech.Breach | 60 | 2.06% |
| Business.Layoff | 52 | 1.79% |
| Disaster.Accident | 50 | 1.72% |
| Government.Sanction | 49 | 1.69% |
| Disaster.NaturalDisaster | 45 | 1.55% |
| Life.Injure | 45 | 1.55% |
| Conflict.Attack | 45 | 1.55% |
| Life.Evacuate | 44 | 1.51% |
| Justice.Fine | 42 | 1.45% |
| Business.MergeOrg | 41 | 1.41% |
| Justice.Sue | 39 | 1.34% |
| Business.EndOrg | 39 | 1.34% |
| Justice.ChargeIndict | 35 | 1.20% |
| Justice.TrialHearing | 32 | 1.10% |
| Movement.Arrive | 31 | 1.07% |
| Personnel.Nominate | 30 | 1.03% |
| Disaster.Rescue | 29 | 1.00% |
| Movement.Transport | 26 | 0.89% |
| Life.Die | 26 | 0.89% |
| Transaction.Lend | 26 | 0.89% |
| Government.EnactLaw | 25 | 0.86% |
| Transaction.Donate | 24 | 0.83% |
| Sport.Compete | 21 | 0.72% |
| Conflict.Demonstrate | 20 | 0.69% |
| Justice.Sentence | 16 | 0.55% |
| Justice.Appeal | 15 | 0.52% |
| Life.BeBorn | 15 | 0.52% |
| Conflict.Ceasefire | 15 | 0.52% |
| Movement.Migrate | 13 | 0.45% |
| Justice.Arrest | 13 | 0.45% |
| Personnel.Elect | 12 | 0.41% |
| Government.Vote | 11 | 0.38% |
| Business.Strike | 9 | 0.31% |
| Business.DeclareBankruptcy | 8 | 0.28% |
| Government.Protest | 6 | 0.21% |
| Justice.Acquit | 5 | 0.17% |
| Life.Marry | 4 | 0.14% |
| Life.Divorce | 2 | 0.07% |
| Movement.Deport | 2 | 0.07% |

**val.** 342 instances in 252 records, 0 triggerless. Trigger gold 342. Argument gold 973. Arguments per instance, mean 2.85. Trigger length in whitespace tokens, mean 1.39, single-token share 78.1%.

#### val event types

| label | count | share |
| --- | --- | --- |
| Contact.Meet | 27 | 7.89% |
| Contact.Communicate | 24 | 7.02% |
| Research.Discover | 21 | 6.14% |
| Personnel.StartPosition | 21 | 6.14% |
| Research.Fund | 18 | 5.26% |
| Business.StartOrg | 15 | 4.39% |
| Research.Publish | 14 | 4.09% |
| Business.LaunchProduct | 14 | 4.09% |
| Transaction.TransferMoney | 13 | 3.80% |
| Tech.Release | 11 | 3.22% |
| Disaster.NaturalDisaster | 11 | 3.22% |
| Medical.Diagnose | 11 | 3.22% |
| Business.MergeOrg | 10 | 2.92% |
| Business.Layoff | 10 | 2.92% |
| Contact.Negotiate | 9 | 2.63% |
| Medical.Treat | 8 | 2.34% |
| Disaster.Rescue | 7 | 2.05% |
| Government.Sanction | 6 | 1.75% |
| Personnel.Nominate | 5 | 1.46% |
| Disaster.Accident | 5 | 1.46% |
| Life.Injure | 5 | 1.46% |
| Conflict.Attack | 5 | 1.46% |
| Justice.Arrest | 5 | 1.46% |
| Justice.ChargeIndict | 5 | 1.46% |
| Tech.Breach | 5 | 1.46% |
| Transaction.TransferOwnership | 5 | 1.46% |
| Life.Evacuate | 4 | 1.17% |
| Disaster.Outbreak | 4 | 1.17% |
| Movement.Migrate | 4 | 1.17% |
| Transaction.Donate | 4 | 1.17% |
| Justice.TrialHearing | 4 | 1.17% |
| Movement.Arrive | 3 | 0.88% |
| Business.Strike | 3 | 0.88% |
| Award.ReceiveAward | 3 | 0.88% |
| Personnel.EndPosition | 3 | 0.88% |
| Movement.Transport | 3 | 0.88% |
| Justice.Fine | 2 | 0.58% |
| Justice.Sentence | 2 | 0.58% |
| Life.Die | 2 | 0.58% |
| Personnel.Elect | 1 | 0.29% |
| Business.EndOrg | 1 | 0.29% |
| Justice.Appeal | 1 | 0.29% |
| Justice.Sue | 1 | 0.29% |
| Movement.Deport | 1 | 0.29% |
| Transaction.Lend | 1 | 0.29% |
| Business.DeclareBankruptcy | 1 | 0.29% |
| Government.Vote | 1 | 0.29% |
| Government.EnactLaw | 1 | 0.29% |
| Conflict.Demonstrate | 1 | 0.29% |
| Conflict.Ceasefire | 1 | 0.29% |

**test.** 391 instances in 298 records, 0 triggerless. Trigger gold 390. Argument gold 1,119. Arguments per instance, mean 2.86. Trigger length in whitespace tokens, mean 1.22, single-token share 87.0%.

#### test event types

| label | count | share |
| --- | --- | --- |
| Business.LaunchProduct | 25 | 6.39% |
| Tech.Release | 22 | 5.63% |
| Research.Discover | 20 | 5.12% |
| Contact.Meet | 18 | 4.60% |
| Contact.Communicate | 16 | 4.09% |
| Personnel.StartPosition | 16 | 4.09% |
| Research.Fund | 16 | 4.09% |
| Business.StartOrg | 14 | 3.58% |
| Research.Publish | 12 | 3.07% |
| Medical.Diagnose | 12 | 3.07% |
| Contact.Negotiate | 12 | 3.07% |
| Justice.Fine | 10 | 2.56% |
| Personnel.EndPosition | 10 | 2.56% |
| Tech.Breach | 10 | 2.56% |
| Movement.Arrive | 9 | 2.30% |
| Disaster.Accident | 9 | 2.30% |
| Business.EndOrg | 9 | 2.30% |
| Conflict.Attack | 9 | 2.30% |
| Transaction.TransferMoney | 8 | 2.05% |
| Award.ReceiveAward | 8 | 2.05% |
| Business.MergeOrg | 8 | 2.05% |
| Disaster.NaturalDisaster | 8 | 2.05% |
| Business.Layoff | 7 | 1.79% |
| Disaster.Outbreak | 7 | 1.79% |
| Medical.Treat | 6 | 1.53% |
| Transaction.Lend | 6 | 1.53% |
| Government.Sanction | 5 | 1.28% |
| Life.Injure | 5 | 1.28% |
| Sport.Compete | 5 | 1.28% |
| Transaction.TransferOwnership | 5 | 1.28% |
| Movement.Transport | 5 | 1.28% |
| Justice.Appeal | 4 | 1.02% |
| Justice.ChargeIndict | 4 | 1.02% |
| Life.Die | 4 | 1.02% |
| Justice.TrialHearing | 4 | 1.02% |
| Life.BeBorn | 4 | 1.02% |
| Life.Evacuate | 3 | 0.77% |
| Personnel.Nominate | 3 | 0.77% |
| Conflict.Ceasefire | 3 | 0.77% |
| Transaction.Donate | 3 | 0.77% |
| Disaster.Rescue | 3 | 0.77% |
| Government.EnactLaw | 3 | 0.77% |
| Personnel.Elect | 3 | 0.77% |
| Justice.Arrest | 3 | 0.77% |
| Justice.Sue | 3 | 0.77% |
| Government.Vote | 2 | 0.51% |
| Conflict.Demonstrate | 2 | 0.51% |
| Justice.Sentence | 2 | 0.51% |
| Government.Protest | 2 | 0.51% |
| Business.DeclareBankruptcy | 1 | 0.26% |
| Business.Strike | 1 | 0.26% |
| Life.Divorce | 1 | 0.26% |
| Movement.Migrate | 1 | 0.26% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 8,644 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 973 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 1,120 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.28 | 16.72% | 1.53 | 26.56% | 0.24% | 0 | 8,644 | 0 | 2,906 |
| val | 1.23 | 13.74% | 1.49 | 25.13% | 0.15% | 0 | 973 | 0 | 342 |
| test | 1.36 | 20.46% | 1.52 | 26.70% | 0.00% | 0 | 1,120 | 0 | 391 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2,906 | 0.00% | 68 | 42.65% | 38.24% | 2.94% | 1.47% | 1.14% |
| val | 342 | 0.00% | 5 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| test | 391 | 0.00% | 6 | 16.67% | 16.67% | 0.00% | 0.00% | 0.34% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 0 | 0 | 68 |
| val | 0 | 0 | 5 |
| test | 0 | 0 | 6 |

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
| cardinal | 2,636 | 8.32% |
| quantity | 1,707 | 5.39% |
| location | 1,699 | 5.36% |
| date | 1,407 | 4.44% |
| organization | 1,406 | 4.44% |
| time | 1,342 | 4.24% |
| measurement | 1,178 | 3.72% |
| geopolitical entity | 1,147 | 3.62% |
| region | 1,034 | 3.26% |
| company | 951 | 3.00% |
| person | 872 | 2.75% |
| area | 714 | 2.25% |
| facility | 663 | 2.09% |
| address | 635 | 2.00% |
| money | 631 | 1.99% |
| percent | 620 | 1.96% |
| government agency | 533 | 1.68% |
| landmark | 501 | 1.58% |
| brand | 420 | 1.33% |
| currency | 401 | 1.27% |
| regulatory body | 386 | 1.22% |
| job title | 374 | 1.18% |
| duration | 338 | 1.07% |
| deadline | 336 | 1.06% |
| occupation | 318 | 1.00% |
| non-profit organization | 314 | 0.99% |
| software | 292 | 0.92% |
| educational institution | 283 | 0.89% |
| product | 267 | 0.84% |
| distance | 245 | 0.77% |
| disease | 224 | 0.71% |
| postal code | 218 | 0.69% |
| protocol | 216 | 0.68% |
| event | 214 | 0.68% |
| fiscal quarter | 210 | 0.66% |
| chemical | 201 | 0.63% |
| ordinal | 196 | 0.62% |
| model number | 193 | 0.61% |
| symptom | 187 | 0.59% |
| algorithm | 183 | 0.58% |
| hardware component | 179 | 0.57% |
| protein | 170 | 0.54% |
| budget line | 169 | 0.53% |
| medication | 159 | 0.50% |
| vaccine | 159 | 0.50% |
| regulation | 157 | 0.50% |
| team role | 146 | 0.46% |
| pathogen | 144 | 0.45% |
| dosage | 141 | 0.45% |
| framework | 139 | 0.44% |
| temperature | 134 | 0.42% |
| database | 130 | 0.41% |
| clinical trial | 130 | 0.41% |
| api | 129 | 0.41% |
| operating system | 122 | 0.39% |
| honorific | 121 | 0.38% |
| airport | 121 | 0.38% |
| diagnosis | 118 | 0.37% |
| programming language | 110 | 0.35% |
| publication | 106 | 0.33% |
| media outlet | 106 | 0.33% |
| organism | 104 | 0.33% |
| academic degree | 100 | 0.32% |
| body of water | 99 | 0.31% |
| speed | 97 | 0.31% |
| financial instrument | 97 | 0.31% |
| voltage | 96 | 0.30% |
| frequency | 96 | 0.30% |
| species | 93 | 0.29% |
| witness | 92 | 0.29% |
| age | 90 | 0.28% |
| nationality | 89 | 0.28% |
| anatomical structure | 89 | 0.28% |
| road | 86 | 0.27% |
| law | 84 | 0.27% |
| conference | 78 | 0.25% |
| gene | 77 | 0.24% |
| medical procedure | 77 | 0.24% |
| mineral | 75 | 0.24% |
| vehicle | 74 | 0.23% |
| mountain | 73 | 0.23% |
| weight | 69 | 0.22% |
| interest rate | 67 | 0.21% |
| patent | 64 | 0.20% |
| award | 57 | 0.18% |
| natural disaster | 54 | 0.17% |
| contract | 54 | 0.17% |
| file format | 53 | 0.17% |
| military unit | 49 | 0.15% |
| security vulnerability | 48 | 0.15% |
| trade union | 46 | 0.15% |
| language | 44 | 0.14% |
| allergen | 44 | 0.14% |
| license | 43 | 0.14% |
| crime | 41 | 0.13% |
| stock ticker | 41 | 0.13% |
| border | 41 | 0.13% |
| sports competition | 38 | 0.12% |
| work of art | 37 | 0.12% |
| credit rating | 36 | 0.11% |
| court case | 35 | 0.11% |
| era | 33 | 0.10% |
| aircraft | 33 | 0.10% |
| treaty | 32 | 0.10% |
| weapon | 30 | 0.09% |
| market index | 30 | 0.09% |
| vessel | 28 | 0.09% |
| sports team | 28 | 0.09% |
| tax | 20 | 0.06% |
| military rank | 18 | 0.06% |
| book | 17 | 0.05% |
| family relationship | 15 | 0.05% |
| criminal organization | 14 | 0.04% |
| religious group | 13 | 0.04% |
| verdict | 11 | 0.03% |
| ethnic group | 9 | 0.03% |
| war | 8 | 0.03% |
| holiday | 7 | 0.02% |
| video game | 6 | 0.02% |
| pseudonym | 5 | 0.02% |
| political party | 5 | 0.02% |
| sentence | 3 | 0.01% |
| film | 2 | 0.01% |
| political affiliation | 2 | 0.01% |
| song | 0 | 0.00% |

#### val

| label | count | share |
| --- | --- | --- |
| cardinal | 343 | 8.67% |
| location | 217 | 5.49% |
| date | 202 | 5.11% |
| time | 177 | 4.48% |
| quantity | 171 | 4.32% |
| geopolitical entity | 165 | 4.17% |
| region | 164 | 4.15% |
| company | 160 | 4.05% |
| organization | 159 | 4.02% |
| measurement | 158 | 4.00% |
| person | 118 | 2.98% |
| area | 91 | 2.30% |
| address | 83 | 2.10% |
| money | 79 | 2.00% |
| landmark | 74 | 1.87% |
| percent | 67 | 1.69% |
| brand | 65 | 1.64% |
| facility | 50 | 1.26% |
| currency | 48 | 1.21% |
| job title | 47 | 1.19% |
| product | 46 | 1.16% |
| regulatory body | 43 | 1.09% |
| government agency | 42 | 1.06% |
| duration | 40 | 1.01% |
| hardware component | 38 | 0.96% |
| software | 36 | 0.91% |
| deadline | 35 | 0.89% |
| event | 35 | 0.89% |
| airport | 33 | 0.83% |
| non-profit organization | 28 | 0.71% |
| occupation | 28 | 0.71% |
| disease | 27 | 0.68% |
| chemical | 27 | 0.68% |
| educational institution | 26 | 0.66% |
| budget line | 25 | 0.63% |
| medication | 24 | 0.61% |
| temperature | 24 | 0.61% |
| regulation | 22 | 0.56% |
| protein | 22 | 0.56% |
| fiscal quarter | 22 | 0.56% |
| symptom | 22 | 0.56% |
| protocol | 21 | 0.53% |
| speed | 21 | 0.53% |
| media outlet | 20 | 0.51% |
| distance | 20 | 0.51% |
| mineral | 18 | 0.46% |
| frequency | 17 | 0.43% |
| database | 16 | 0.40% |
| operating system | 16 | 0.40% |
| postal code | 16 | 0.40% |
| military unit | 16 | 0.40% |
| framework | 16 | 0.40% |
| road | 15 | 0.38% |
| model number | 15 | 0.38% |
| honorific | 14 | 0.35% |
| ordinal | 14 | 0.35% |
| dosage | 13 | 0.33% |
| api | 13 | 0.33% |
| anatomical structure | 13 | 0.33% |
| programming language | 13 | 0.33% |
| vaccine | 13 | 0.33% |
| organism | 12 | 0.30% |
| conference | 12 | 0.30% |
| voltage | 12 | 0.30% |
| sports team | 11 | 0.28% |
| diagnosis | 11 | 0.28% |
| algorithm | 11 | 0.28% |
| body of water | 10 | 0.25% |
| language | 10 | 0.25% |
| species | 9 | 0.23% |
| gene | 9 | 0.23% |
| nationality | 9 | 0.23% |
| law | 9 | 0.23% |
| financial instrument | 9 | 0.23% |
| clinical trial | 9 | 0.23% |
| ethnic group | 8 | 0.20% |
| publication | 8 | 0.20% |
| academic degree | 8 | 0.20% |
| witness | 8 | 0.20% |
| allergen | 7 | 0.18% |
| award | 7 | 0.18% |
| stock ticker | 7 | 0.18% |
| license | 7 | 0.18% |
| contract | 7 | 0.18% |
| security vulnerability | 7 | 0.18% |
| mountain | 7 | 0.18% |
| file format | 7 | 0.18% |
| vehicle | 6 | 0.15% |
| book | 6 | 0.15% |
| interest rate | 6 | 0.15% |
| pathogen | 6 | 0.15% |
| work of art | 6 | 0.15% |
| weight | 6 | 0.15% |
| patent | 6 | 0.15% |
| border | 6 | 0.15% |
| team role | 6 | 0.15% |
| era | 6 | 0.15% |
| verdict | 5 | 0.13% |
| military rank | 5 | 0.13% |
| crime | 5 | 0.13% |
| market index | 5 | 0.13% |
| vessel | 5 | 0.13% |
| aircraft | 4 | 0.10% |
| age | 4 | 0.10% |
| sports competition | 4 | 0.10% |
| credit rating | 3 | 0.08% |
| trade union | 3 | 0.08% |
| natural disaster | 3 | 0.08% |
| weapon | 3 | 0.08% |
| tax | 2 | 0.05% |
| medical procedure | 2 | 0.05% |
| criminal organization | 2 | 0.05% |
| sentence | 2 | 0.05% |
| treaty | 2 | 0.05% |
| political party | 1 | 0.03% |
| pseudonym | 0 | 0.00% |
| court case | 0 | 0.00% |
| religious group | 0 | 0.00% |
| holiday | 0 | 0.00% |
| political affiliation | 0 | 0.00% |
| song | 0 | 0.00% |
| family relationship | 0 | 0.00% |
| video game | 0 | 0.00% |
| war | 0 | 0.00% |
| film | 0 | 0.00% |

#### test

| label | count | share |
| --- | --- | --- |
| cardinal | 322 | 8.42% |
| location | 221 | 5.78% |
| quantity | 215 | 5.62% |
| date | 184 | 4.81% |
| measurement | 155 | 4.05% |
| time | 146 | 3.82% |
| organization | 145 | 3.79% |
| geopolitical entity | 123 | 3.22% |
| region | 119 | 3.11% |
| person | 104 | 2.72% |
| company | 104 | 2.72% |
| government agency | 88 | 2.30% |
| money | 84 | 2.20% |
| facility | 78 | 2.04% |
| address | 73 | 1.91% |
| area | 66 | 1.73% |
| non-profit organization | 59 | 1.54% |
| percent | 59 | 1.54% |
| job title | 55 | 1.44% |
| landmark | 51 | 1.33% |
| currency | 49 | 1.28% |
| ordinal | 48 | 1.26% |
| occupation | 48 | 1.26% |
| duration | 48 | 1.26% |
| deadline | 40 | 1.05% |
| software | 38 | 0.99% |
| regulatory body | 37 | 0.97% |
| product | 35 | 0.92% |
| brand | 33 | 0.86% |
| regulation | 31 | 0.81% |
| fiscal quarter | 31 | 0.81% |
| disease | 30 | 0.78% |
| hardware component | 29 | 0.76% |
| postal code | 28 | 0.73% |
| model number | 28 | 0.73% |
| symptom | 24 | 0.63% |
| distance | 24 | 0.63% |
| budget line | 22 | 0.58% |
| framework | 21 | 0.55% |
| protein | 21 | 0.55% |
| honorific | 20 | 0.52% |
| age | 20 | 0.52% |
| mineral | 20 | 0.52% |
| event | 20 | 0.52% |
| publication | 20 | 0.52% |
| protocol | 19 | 0.50% |
| educational institution | 19 | 0.50% |
| chemical | 17 | 0.44% |
| vaccine | 17 | 0.44% |
| voltage | 15 | 0.39% |
| medication | 15 | 0.39% |
| airport | 14 | 0.37% |
| road | 14 | 0.37% |
| algorithm | 14 | 0.37% |
| speed | 13 | 0.34% |
| operating system | 13 | 0.34% |
| anatomical structure | 13 | 0.34% |
| programming language | 13 | 0.34% |
| species | 13 | 0.34% |
| sports team | 13 | 0.34% |
| nationality | 13 | 0.34% |
| academic degree | 13 | 0.34% |
| clinical trial | 12 | 0.31% |
| aircraft | 12 | 0.31% |
| temperature | 12 | 0.31% |
| diagnosis | 12 | 0.31% |
| api | 12 | 0.31% |
| natural disaster | 12 | 0.31% |
| financial instrument | 12 | 0.31% |
| file format | 12 | 0.31% |
| database | 12 | 0.31% |
| media outlet | 11 | 0.29% |
| body of water | 11 | 0.29% |
| law | 11 | 0.29% |
| dosage | 11 | 0.29% |
| stock ticker | 10 | 0.26% |
| patent | 10 | 0.26% |
| security vulnerability | 9 | 0.24% |
| organism | 9 | 0.24% |
| witness | 9 | 0.24% |
| team role | 8 | 0.21% |
| medical procedure | 8 | 0.21% |
| weight | 8 | 0.21% |
| era | 8 | 0.21% |
| frequency | 8 | 0.21% |
| vehicle | 7 | 0.18% |
| gene | 7 | 0.18% |
| weapon | 6 | 0.16% |
| mountain | 6 | 0.16% |
| crime | 6 | 0.16% |
| market index | 6 | 0.16% |
| license | 5 | 0.13% |
| vessel | 5 | 0.13% |
| pathogen | 5 | 0.13% |
| tax | 5 | 0.13% |
| court case | 5 | 0.13% |
| credit rating | 5 | 0.13% |
| allergen | 5 | 0.13% |
| book | 4 | 0.10% |
| conference | 4 | 0.10% |
| border | 4 | 0.10% |
| work of art | 4 | 0.10% |
| language | 3 | 0.08% |
| verdict | 3 | 0.08% |
| sports competition | 3 | 0.08% |
| treaty | 3 | 0.08% |
| contract | 3 | 0.08% |
| military rank | 2 | 0.05% |
| trade union | 2 | 0.05% |
| film | 2 | 0.05% |
| military unit | 2 | 0.05% |
| interest rate | 2 | 0.05% |
| award | 2 | 0.05% |
| pseudonym | 1 | 0.03% |
| holiday | 1 | 0.03% |
| ethnic group | 1 | 0.03% |
| criminal organization | 1 | 0.03% |
| video game | 0 | 0.00% |
| family relationship | 0 | 0.00% |
| political party | 0 | 0.00% |
| religious group | 0 | 0.00% |
| political affiliation | 0 | 0.00% |
| song | 0 | 0.00% |
| sentence | 0 | 0.00% |
| war | 0 | 0.00% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| located_in | 672 | 10.51% |
| based_in | 573 | 8.96% |
| works_for | 477 | 7.46% |
| operates_in | 437 | 6.83% |
| headquartered_in | 436 | 6.82% |
| affiliated_with | 332 | 5.19% |
| employer_of | 291 | 4.55% |
| developed_by | 200 | 3.13% |
| competitor_of | 152 | 2.38% |
| leader_of | 151 | 2.36% |
| member_of | 151 | 2.36% |
| regulated_by | 142 | 2.22% |
| reports_to | 132 | 2.06% |
| part_of | 131 | 2.05% |
| colleague_of | 121 | 1.89% |
| manufactured_by | 113 | 1.77% |
| produced_by | 110 | 1.72% |
| customer_of | 89 | 1.39% |
| designed_by | 82 | 1.28% |
| sponsored_by | 79 | 1.24% |
| advisor_to | 77 | 1.20% |
| supplier_of | 75 | 1.17% |
| invested_in | 74 | 1.16% |
| founded | 72 | 1.13% |
| compatible_with | 69 | 1.08% |
| published_by | 66 | 1.03% |
| board_member_of | 64 | 1.00% |
| subsidiary_of | 63 | 0.98% |
| distributor_of | 58 | 0.91% |
| acquired | 53 | 0.83% |
| authored_by | 50 | 0.78% |
| implements | 50 | 0.78% |
| treats | 48 | 0.75% |
| licensed_to | 44 | 0.69% |
| participated_in | 39 | 0.61% |
| interacts_with | 36 | 0.56% |
| party_to | 34 | 0.53% |
| mentor_of | 31 | 0.48% |
| supersedes | 30 | 0.47% |
| born_in | 28 | 0.44% |
| owns | 27 | 0.42% |
| represents | 27 | 0.42% |
| merged_with | 27 | 0.42% |
| adjacent_to | 26 | 0.41% |
| directed_by | 25 | 0.39% |
| sold_to | 23 | 0.36% |
| citizen_of | 23 | 0.36% |
| depends_on | 23 | 0.36% |
| member_state_of | 19 | 0.30% |
| causes | 18 | 0.28% |
| sued_by | 17 | 0.27% |
| charged_with | 17 | 0.27% |
| capital_of | 17 | 0.27% |
| died_in | 16 | 0.25% |
| opposed_to | 16 | 0.25% |
| allied_with | 16 | 0.25% |
| won | 16 | 0.25% |
| testified_against | 15 | 0.23% |
| signatory_of | 15 | 0.23% |
| parent_of | 14 | 0.22% |
| derived_from | 13 | 0.20% |
| spouse_of | 10 | 0.16% |
| relative_of | 10 | 0.16% |
| successor_of | 10 | 0.16% |
| borders | 8 | 0.13% |
| named_after | 4 | 0.06% |
| predecessor_of | 3 | 0.05% |
| composed_by | 3 | 0.05% |
| measured_in | 2 | 0.03% |
| convicted_of | 2 | 0.03% |
| child_of | 2 | 0.03% |

#### val

| label | count | share |
| --- | --- | --- |
| located_in | 91 | 11.56% |
| based_in | 74 | 9.40% |
| operates_in | 59 | 7.50% |
| headquartered_in | 55 | 6.99% |
| affiliated_with | 39 | 4.96% |
| works_for | 39 | 4.96% |
| employer_of | 29 | 3.68% |
| part_of | 28 | 3.56% |
| developed_by | 20 | 2.54% |
| regulated_by | 18 | 2.29% |
| competitor_of | 16 | 2.03% |
| reports_to | 15 | 1.91% |
| distributor_of | 15 | 1.91% |
| customer_of | 14 | 1.78% |
| supplier_of | 13 | 1.65% |
| produced_by | 13 | 1.65% |
| advisor_to | 13 | 1.65% |
| acquired | 12 | 1.52% |
| member_of | 11 | 1.40% |
| manufactured_by | 11 | 1.40% |
| interacts_with | 10 | 1.27% |
| leader_of | 10 | 1.27% |
| treats | 9 | 1.14% |
| founded | 9 | 1.14% |
| colleague_of | 9 | 1.14% |
| designed_by | 9 | 1.14% |
| allied_with | 9 | 1.14% |
| adjacent_to | 9 | 1.14% |
| compatible_with | 8 | 1.02% |
| sponsored_by | 8 | 1.02% |
| invested_in | 7 | 0.89% |
| owns | 7 | 0.89% |
| born_in | 6 | 0.76% |
| represents | 6 | 0.76% |
| merged_with | 6 | 0.76% |
| subsidiary_of | 6 | 0.76% |
| authored_by | 5 | 0.64% |
| published_by | 5 | 0.64% |
| board_member_of | 5 | 0.64% |
| party_to | 5 | 0.64% |
| licensed_to | 4 | 0.51% |
| participated_in | 4 | 0.51% |
| sold_to | 4 | 0.51% |
| directed_by | 4 | 0.51% |
| opposed_to | 4 | 0.51% |
| mentor_of | 3 | 0.38% |
| capital_of | 3 | 0.38% |
| implements | 3 | 0.38% |
| citizen_of | 3 | 0.38% |
| won | 3 | 0.38% |
| charged_with | 3 | 0.38% |
| depends_on | 2 | 0.25% |
| derived_from | 2 | 0.25% |
| supersedes | 2 | 0.25% |
| borders | 2 | 0.25% |
| member_state_of | 2 | 0.25% |
| named_after | 1 | 0.13% |
| causes | 1 | 0.13% |
| signatory_of | 1 | 0.13% |
| predecessor_of | 1 | 0.13% |
| composed_by | 1 | 0.13% |
| testified_against | 1 | 0.13% |

#### test

| label | count | share |
| --- | --- | --- |
| located_in | 89 | 11.10% |
| based_in | 67 | 8.35% |
| works_for | 62 | 7.73% |
| headquartered_in | 57 | 7.11% |
| operates_in | 49 | 6.11% |
| affiliated_with | 36 | 4.49% |
| employer_of | 36 | 4.49% |
| developed_by | 31 | 3.87% |
| colleague_of | 21 | 2.62% |
| regulated_by | 20 | 2.49% |
| leader_of | 19 | 2.37% |
| competitor_of | 19 | 2.37% |
| part_of | 19 | 2.37% |
| supplier_of | 17 | 2.12% |
| distributor_of | 15 | 1.87% |
| compatible_with | 14 | 1.75% |
| manufactured_by | 13 | 1.62% |
| reports_to | 11 | 1.37% |
| customer_of | 11 | 1.37% |
| member_of | 10 | 1.25% |
| founded | 10 | 1.25% |
| acquired | 9 | 1.12% |
| party_to | 8 | 1.00% |
| produced_by | 8 | 1.00% |
| sponsored_by | 8 | 1.00% |
| designed_by | 7 | 0.87% |
| subsidiary_of | 7 | 0.87% |
| allied_with | 7 | 0.87% |
| borders | 7 | 0.87% |
| charged_with | 7 | 0.87% |
| merged_with | 6 | 0.75% |
| adjacent_to | 6 | 0.75% |
| advisor_to | 6 | 0.75% |
| published_by | 6 | 0.75% |
| implements | 6 | 0.75% |
| sued_by | 5 | 0.62% |
| licensed_to | 5 | 0.62% |
| invested_in | 5 | 0.62% |
| born_in | 5 | 0.62% |
| mentor_of | 4 | 0.50% |
| citizen_of | 4 | 0.50% |
| board_member_of | 4 | 0.50% |
| capital_of | 3 | 0.37% |
| member_state_of | 3 | 0.37% |
| owns | 3 | 0.37% |
| supersedes | 3 | 0.37% |
| authored_by | 3 | 0.37% |
| derived_from | 3 | 0.37% |
| participated_in | 3 | 0.37% |
| sold_to | 2 | 0.25% |
| successor_of | 2 | 0.25% |
| testified_against | 2 | 0.25% |
| died_in | 2 | 0.25% |
| interacts_with | 2 | 0.25% |
| represents | 2 | 0.25% |
| predecessor_of | 2 | 0.25% |
| depends_on | 2 | 0.25% |
| treats | 2 | 0.25% |
| directed_by | 1 | 0.12% |
| won | 1 | 0.12% |
| child_of | 1 | 0.12% |
| signatory_of | 1 | 0.12% |
| parent_of | 1 | 0.12% |
| measured_in | 1 | 0.12% |
| spouse_of | 1 | 0.12% |

### Classifications

**train, actionability.** Majority class 49.2% of 964 answers. The marginal is `true_label`, not the menu.

#### train actionability

| label | count | share |
| --- | --- | --- |
| informational | 474 | 49.17% |
| requires action | 382 | 39.63% |
| advisory | 108 | 11.20% |

**train, audience.** Majority class 37.1% of 1,946 answers. The marginal is `true_label`, not the menu.

#### train audience

| label | count | share |
| --- | --- | --- |
| expert | 722 | 37.10% |
| general public | 547 | 28.11% |
| regulatory | 478 | 24.56% |
| internal | 199 | 10.23% |

**train, certainty.** Majority class 99.7% of 975 answers. The marginal is `true_label`, not the menu.

#### train certainty

| label | count | share |
| --- | --- | --- |
| asserted | 972 | 99.69% |
| hedged | 3 | 0.31% |

**train, document_genre.** Majority class 25.9% of 993 answers. The marginal is `true_label`, not the menu.

#### train document_genre

| label | count | share |
| --- | --- | --- |
| news report | 257 | 25.88% |
| announcement | 252 | 25.38% |
| technical documentation | 154 | 15.51% |
| correspondence | 90 | 9.06% |
| legal filing | 81 | 8.16% |
| advertisement | 66 | 6.65% |
| review | 56 | 5.64% |
| opinion | 37 | 3.73% |

**train, formality.** Majority class 95.7% of 1,047 answers. The marginal is `true_label`, not the menu.

#### train formality

| label | count | share |
| --- | --- | --- |
| formal | 1,002 | 95.70% |
| informal | 45 | 4.30% |

**train, language_register.** Majority class 64.4% of 1,015 answers. The marginal is `true_label`, not the menu.

#### train language_register

| label | count | share |
| --- | --- | --- |
| technical | 654 | 64.43% |
| plain | 238 | 23.45% |
| legal | 115 | 11.33% |
| colloquial | 8 | 0.79% |

**train, risk_level.** Majority class 39.2% of 1,039 answers. The marginal is `true_label`, not the menu.

#### train risk_level

| label | count | share |
| --- | --- | --- |
| low | 407 | 39.17% |
| high | 315 | 30.32% |
| none | 166 | 15.98% |
| moderate | 151 | 14.53% |

**train, sentiment.** Majority class 55.6% of 1,019 answers. The marginal is `true_label`, not the menu.

#### train sentiment

| label | count | share |
| --- | --- | --- |
| positive | 567 | 55.64% |
| negative | 302 | 29.64% |
| neutral | 150 | 14.72% |

**train, subjectivity.** Majority class 91.2% of 1,048 answers. The marginal is `true_label`, not the menu.

#### train subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 956 | 91.22% |
| subjective | 92 | 8.78% |

**train, temporal_orientation.** Majority class 69.5% of 990 answers. The marginal is `true_label`, not the menu.

#### train temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 688 | 69.49% |
| present | 197 | 19.90% |
| future | 73 | 7.37% |
| timeless | 32 | 3.23% |

**train, topic.** Majority class 20.7% of 2,466 answers. The marginal is `true_label`, not the menu.

#### train topic

| label | count | share |
| --- | --- | --- |
| business | 511 | 20.72% |
| technology | 314 | 12.73% |
| health | 313 | 12.69% |
| science | 230 | 9.33% |
| finance | 226 | 9.16% |
| environment | 183 | 7.42% |
| legal | 181 | 7.34% |
| politics | 92 | 3.73% |
| defense | 89 | 3.61% |
| world | 86 | 3.49% |
| travel | 55 | 2.23% |
| sports | 51 | 2.07% |
| energy | 49 | 1.99% |
| education | 41 | 1.66% |
| agriculture | 37 | 1.50% |
| entertainment | 8 | 0.32% |

**train, urgency.** Majority class 53.9% of 976 answers. The marginal is `true_label`, not the menu.

#### train urgency

| label | count | share |
| --- | --- | --- |
| routine | 526 | 53.89% |
| elevated | 260 | 26.64% |
| critical | 173 | 17.73% |
| urgent | 17 | 1.74% |

**val, actionability.** Majority class 57.3% of 117 answers. The marginal is `true_label`, not the menu.

#### val actionability

| label | count | share |
| --- | --- | --- |
| informational | 67 | 57.26% |
| requires action | 41 | 35.04% |
| advisory | 9 | 7.69% |

**val, audience.** Majority class 36.2% of 254 answers. The marginal is `true_label`, not the menu.

#### val audience

| label | count | share |
| --- | --- | --- |
| expert | 92 | 36.22% |
| general public | 84 | 33.07% |
| regulatory | 57 | 22.44% |
| internal | 21 | 8.27% |

**val, certainty.** Majority class 100.0% of 128 answers. The marginal is `true_label`, not the menu.

#### val certainty

| label | count | share |
| --- | --- | --- |
| asserted | 128 | 100.00% |

**val, document_genre.** Majority class 22.0% of 123 answers. The marginal is `true_label`, not the menu.

#### val document_genre

| label | count | share |
| --- | --- | --- |
| announcement | 27 | 21.95% |
| news report | 21 | 17.07% |
| technical documentation | 16 | 13.01% |
| correspondence | 16 | 13.01% |
| legal filing | 15 | 12.20% |
| review | 12 | 9.76% |
| advertisement | 12 | 9.76% |
| opinion | 4 | 3.25% |

**val, formality.** Majority class 97.3% of 112 answers. The marginal is `true_label`, not the menu.

#### val formality

| label | count | share |
| --- | --- | --- |
| formal | 109 | 97.32% |
| informal | 3 | 2.68% |

**val, language_register.** Majority class 67.6% of 111 answers. The marginal is `true_label`, not the menu.

#### val language_register

| label | count | share |
| --- | --- | --- |
| technical | 75 | 67.57% |
| plain | 27 | 24.32% |
| legal | 8 | 7.21% |
| colloquial | 1 | 0.90% |

**val, risk_level.** Majority class 47.3% of 146 answers. The marginal is `true_label`, not the menu.

#### val risk_level

| label | count | share |
| --- | --- | --- |
| low | 69 | 47.26% |
| high | 46 | 31.51% |
| none | 21 | 14.38% |
| moderate | 10 | 6.85% |

**val, sentiment.** Majority class 58.2% of 122 answers. The marginal is `true_label`, not the menu.

#### val sentiment

| label | count | share |
| --- | --- | --- |
| positive | 71 | 58.20% |
| negative | 32 | 26.23% |
| neutral | 19 | 15.57% |

**val, subjectivity.** Majority class 89.2% of 139 answers. The marginal is `true_label`, not the menu.

#### val subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 124 | 89.21% |
| subjective | 15 | 10.79% |

**val, temporal_orientation.** Majority class 65.1% of 126 answers. The marginal is `true_label`, not the menu.

#### val temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 82 | 65.08% |
| present | 31 | 24.60% |
| future | 8 | 6.35% |
| timeless | 5 | 3.97% |

**val, topic.** Majority class 24.8% of 274 answers. The marginal is `true_label`, not the menu.

#### val topic

| label | count | share |
| --- | --- | --- |
| business | 68 | 24.82% |
| technology | 50 | 18.25% |
| finance | 30 | 10.95% |
| health | 27 | 9.85% |
| science | 24 | 8.76% |
| legal | 19 | 6.93% |
| environment | 11 | 4.01% |
| travel | 10 | 3.65% |
| defense | 7 | 2.55% |
| world | 7 | 2.55% |
| sports | 6 | 2.19% |
| politics | 5 | 1.82% |
| education | 4 | 1.46% |
| energy | 3 | 1.09% |
| entertainment | 2 | 0.73% |
| agriculture | 1 | 0.36% |

**val, urgency.** Majority class 51.3% of 119 answers. The marginal is `true_label`, not the menu.

#### val urgency

| label | count | share |
| --- | --- | --- |
| routine | 61 | 51.26% |
| elevated | 34 | 28.57% |
| critical | 21 | 17.65% |
| urgent | 3 | 2.52% |

**test, actionability.** Majority class 54.2% of 144 answers. The marginal is `true_label`, not the menu.

#### test actionability

| label | count | share |
| --- | --- | --- |
| informational | 78 | 54.17% |
| requires action | 54 | 37.50% |
| advisory | 12 | 8.33% |

**test, audience.** Majority class 38.3% of 227 answers. The marginal is `true_label`, not the menu.

#### test audience

| label | count | share |
| --- | --- | --- |
| expert | 87 | 38.33% |
| general public | 66 | 29.07% |
| regulatory | 55 | 24.23% |
| internal | 19 | 8.37% |

**test, certainty.** Majority class 97.1% of 105 answers. The marginal is `true_label`, not the menu.

#### test certainty

| label | count | share |
| --- | --- | --- |
| asserted | 102 | 97.14% |
| hedged | 3 | 2.86% |

**test, document_genre.** Majority class 24.6% of 130 answers. The marginal is `true_label`, not the menu.

#### test document_genre

| label | count | share |
| --- | --- | --- |
| announcement | 32 | 24.62% |
| news report | 30 | 23.08% |
| technical documentation | 18 | 13.85% |
| correspondence | 18 | 13.85% |
| legal filing | 11 | 8.46% |
| review | 8 | 6.15% |
| advertisement | 7 | 5.38% |
| opinion | 6 | 4.62% |

**test, formality.** Majority class 96.6% of 116 answers. The marginal is `true_label`, not the menu.

#### test formality

| label | count | share |
| --- | --- | --- |
| formal | 112 | 96.55% |
| informal | 4 | 3.45% |

**test, language_register.** Majority class 67.8% of 115 answers. The marginal is `true_label`, not the menu.

#### test language_register

| label | count | share |
| --- | --- | --- |
| technical | 78 | 67.83% |
| plain | 20 | 17.39% |
| legal | 17 | 14.78% |

**test, risk_level.** Majority class 36.4% of 110 answers. The marginal is `true_label`, not the menu.

#### test risk_level

| label | count | share |
| --- | --- | --- |
| low | 40 | 36.36% |
| high | 34 | 30.91% |
| moderate | 21 | 19.09% |
| none | 15 | 13.64% |

**test, sentiment.** Majority class 47.3% of 131 answers. The marginal is `true_label`, not the menu.

#### test sentiment

| label | count | share |
| --- | --- | --- |
| positive | 62 | 47.33% |
| negative | 50 | 38.17% |
| neutral | 19 | 14.50% |

**test, subjectivity.** Majority class 91.4% of 116 answers. The marginal is `true_label`, not the menu.

#### test subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 106 | 91.38% |
| subjective | 10 | 8.62% |

**test, temporal_orientation.** Majority class 71.7% of 113 answers. The marginal is `true_label`, not the menu.

#### test temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 81 | 71.68% |
| present | 21 | 18.58% |
| future | 8 | 7.08% |
| timeless | 3 | 2.65% |

**test, topic.** Majority class 20.5% of 336 answers. The marginal is `true_label`, not the menu.

#### test topic

| label | count | share |
| --- | --- | --- |
| business | 69 | 20.54% |
| health | 45 | 13.39% |
| technology | 38 | 11.31% |
| legal | 30 | 8.93% |
| science | 29 | 8.63% |
| finance | 28 | 8.33% |
| environment | 25 | 7.44% |
| politics | 15 | 4.46% |
| world | 14 | 4.17% |
| defense | 10 | 2.98% |
| education | 8 | 2.38% |
| travel | 8 | 2.38% |
| energy | 7 | 2.08% |
| agriculture | 5 | 1.49% |
| sports | 4 | 1.19% |
| entertainment | 1 | 0.30% |

**test, urgency.** Majority class 54.0% of 124 answers. The marginal is `true_label`, not the menu.

#### test urgency

| label | count | share |
| --- | --- | --- |
| routine | 67 | 54.03% |
| elevated | 36 | 29.03% |
| critical | 20 | 16.13% |
| urgent | 1 | 0.81% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | condition | 58 | 96.67% |
| clinical_finding | patient_group | 52 | 86.67% |
| clinical_finding | treatment | 51 | 85.00% |
| clinical_finding | outcome | 48 | 80.00% |
| clinical_finding | record_metadata | natural/condition ×58, natural/patient_group ×1, natural/treatment ×1 |  |
| company_profile | size | 249 | 100.00% |
| company_profile | name | 248 | 99.60% |
| company_profile | headquarters | 228 | 91.57% |
| company_profile | industry | 156 | 62.65% |
| company_profile | founded | 89 | 35.74% |
| company_profile | record_metadata | natural/name ×239, natural/industry ×1 |  |
| event_listing | venue | 56 | 93.33% |
| event_listing | date | 55 | 91.67% |
| event_listing | name | 46 | 76.67% |
| event_listing | organizer | 42 | 70.00% |
| event_listing | price | 24 | 40.00% |
| event_listing | record_metadata | natural/name ×46, natural/venue ×12, natural/date ×2 |  |
| financial_report | organization | 65 | 98.48% |
| financial_report | period | 60 | 90.91% |
| financial_report | revenue | 58 | 87.88% |
| financial_report | net_income | 54 | 81.82% |
| financial_report | record_metadata | natural/organization ×65, natural/period ×1 |  |
| incident_report | severity | 143 | 100.00% |
| incident_report | date | 140 | 97.90% |
| incident_report | location | 135 | 94.41% |
| incident_report | incident_type | 101 | 70.63% |
| incident_report | casualties | 94 | 65.73% |
| incident_report | record_metadata | natural/location ×135, natural/incident_type ×6, natural/date ×2 |  |
| job_posting | title | 11 | 100.00% |
| job_posting | company | 11 | 100.00% |
| job_posting | location | 11 | 100.00% |
| job_posting | employment_type | 11 | 100.00% |
| job_posting | record_metadata | natural/location ×9 |  |
| legal_case | status | 33 | 100.00% |
| legal_case | defendant | 31 | 93.94% |
| legal_case | court | 29 | 87.88% |
| legal_case | plaintiff | 28 | 84.85% |
| legal_case | case_name | 6 | 18.18% |
| legal_case | record_metadata | natural/court ×23, natural/case_name ×6, natural/plaintiff ×3, natural/status ×1 |  |
| person_profile | name | 228 | 99.56% |
| person_profile | role | 214 | 93.45% |
| person_profile | employer | 212 | 92.58% |
| person_profile | location | 177 | 77.29% |
| person_profile | record_metadata | natural/location ×142, natural/name ×42, natural/role ×1 |  |
| product | condition | 107 | 100.00% |
| product | name | 103 | 96.26% |
| product | brand | 103 | 96.26% |
| product | price | 77 | 71.96% |
| product | record_metadata | natural/name ×103, natural/brand ×3, natural/price ×1 |  |
| real_estate_listing | address | 11 | 100.00% |
| real_estate_listing | price | 11 | 100.00% |
| real_estate_listing | area | 11 | 100.00% |
| real_estate_listing | property_type | 11 | 100.00% |
| real_estate_listing | bedrooms | 10 | 90.91% |
| real_estate_listing | record_metadata | natural/address ×11 |  |
| research_paper | year | 53 | 96.36% |
| research_paper | title | 52 | 94.55% |
| research_paper | venue | 44 | 80.00% |
| research_paper | authors | 31 | 56.36% |
| research_paper | topic | 13 | 23.64% |
| research_paper | record_metadata | natural/title ×52, natural/authors ×3 |  |
| software_release | product | 55 | 98.21% |
| software_release | release_date | 54 | 96.43% |
| software_release | version | 52 | 92.86% |
| software_release | platform | 44 | 78.57% |
| software_release | record_metadata | natural/product ×55, natural/version ×1 |  |
| transaction | amount | 111 | 96.52% |
| transaction | date | 95 | 82.61% |
| transaction | item | 93 | 80.87% |
| transaction | seller | 91 | 79.13% |
| transaction | buyer | 87 | 75.65% |
| transaction | record_metadata | natural/item ×92, natural/amount ×19 |  |
| vehicle_listing | make | 6 | 100.00% |
| vehicle_listing | model | 6 | 100.00% |
| vehicle_listing | year | 5 | 83.33% |
| vehicle_listing | price | 4 | 66.67% |
| vehicle_listing | mileage | 2 | 33.33% |
| vehicle_listing | record_metadata | natural/make ×6 |  |

**val.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | condition | 9 | 100.00% |
| clinical_finding | patient_group | 8 | 88.89% |
| clinical_finding | outcome | 6 | 66.67% |
| clinical_finding | treatment | 6 | 66.67% |
| clinical_finding | record_metadata | natural/condition ×9 |  |
| company_profile | name | 38 | 100.00% |
| company_profile | size | 38 | 100.00% |
| company_profile | headquarters | 32 | 84.21% |
| company_profile | industry | 20 | 52.63% |
| company_profile | founded | 14 | 36.84% |
| company_profile | record_metadata | natural/name ×35 |  |
| event_listing | name | 5 | 100.00% |
| event_listing | venue | 5 | 100.00% |
| event_listing | date | 5 | 100.00% |
| event_listing | organizer | 5 | 100.00% |
| event_listing | price | 1 | 20.00% |
| event_listing | record_metadata | natural/name ×5 |  |
| financial_report | organization | 6 | 100.00% |
| financial_report | period | 6 | 100.00% |
| financial_report | net_income | 5 | 83.33% |
| financial_report | revenue | 4 | 66.67% |
| financial_report | record_metadata | natural/organization ×6 |  |
| incident_report | date | 23 | 100.00% |
| incident_report | severity | 23 | 100.00% |
| incident_report | location | 21 | 91.30% |
| incident_report | casualties | 19 | 82.61% |
| incident_report | incident_type | 15 | 65.22% |
| incident_report | record_metadata | natural/location ×21, natural/incident_type ×1, natural/date ×1 |  |
| legal_case | court | 2 | 100.00% |
| legal_case | plaintiff | 2 | 100.00% |
| legal_case | defendant | 2 | 100.00% |
| legal_case | status | 2 | 100.00% |
| legal_case | case_name | 1 | 50.00% |
| legal_case | record_metadata | natural/court ×1, natural/case_name ×1 |  |
| person_profile | name | 36 | 100.00% |
| person_profile | role | 32 | 88.89% |
| person_profile | employer | 28 | 77.78% |
| person_profile | location | 28 | 77.78% |
| person_profile | record_metadata | natural/location ×24, natural/name ×7 |  |
| product | name | 13 | 100.00% |
| product | brand | 13 | 100.00% |
| product | condition | 13 | 100.00% |
| product | price | 10 | 76.92% |
| product | record_metadata | natural/name ×13 |  |
| real_estate_listing | address | 4 | 100.00% |
| real_estate_listing | price | 4 | 100.00% |
| real_estate_listing | bedrooms | 4 | 100.00% |
| real_estate_listing | area | 4 | 100.00% |
| real_estate_listing | property_type | 4 | 100.00% |
| real_estate_listing | record_metadata | natural/address ×4 |  |
| research_paper | title | 7 | 100.00% |
| research_paper | year | 6 | 85.71% |
| research_paper | venue | 5 | 71.43% |
| research_paper | authors | 3 | 42.86% |
| research_paper | record_metadata | natural/title ×7 |  |
| software_release | product | 9 | 100.00% |
| software_release | release_date | 9 | 100.00% |
| software_release | version | 8 | 88.89% |
| software_release | platform | 8 | 88.89% |
| software_release | record_metadata | natural/product ×9 |  |
| transaction | amount | 18 | 100.00% |
| transaction | buyer | 15 | 83.33% |
| transaction | seller | 15 | 83.33% |
| transaction | item | 12 | 66.67% |
| transaction | date | 10 | 55.56% |
| transaction | record_metadata | natural/item ×12, natural/amount ×4 |  |
| vehicle_listing | make | 1 | 100.00% |
| vehicle_listing | model | 1 | 100.00% |
| vehicle_listing | year | 1 | 100.00% |
| vehicle_listing | price | 1 | 100.00% |
| vehicle_listing | record_metadata | natural/make ×1 |  |

**test.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | condition | 5 | 100.00% |
| clinical_finding | patient_group | 5 | 100.00% |
| clinical_finding | treatment | 4 | 80.00% |
| clinical_finding | outcome | 4 | 80.00% |
| clinical_finding | record_metadata | natural/condition ×5 |  |
| company_profile | name | 31 | 100.00% |
| company_profile | size | 31 | 100.00% |
| company_profile | headquarters | 28 | 90.32% |
| company_profile | industry | 19 | 61.29% |
| company_profile | founded | 13 | 41.94% |
| company_profile | record_metadata | natural/name ×28 |  |
| event_listing | venue | 3 | 100.00% |
| event_listing | date | 3 | 100.00% |
| event_listing | organizer | 3 | 100.00% |
| event_listing | name | 2 | 66.67% |
| event_listing | record_metadata | natural/name ×2, natural/venue ×1 |  |
| financial_report | organization | 8 | 100.00% |
| financial_report | period | 8 | 100.00% |
| financial_report | net_income | 6 | 75.00% |
| financial_report | revenue | 5 | 62.50% |
| financial_report | record_metadata | natural/organization ×8 |  |
| incident_report | location | 18 | 100.00% |
| incident_report | date | 18 | 100.00% |
| incident_report | severity | 18 | 100.00% |
| incident_report | incident_type | 10 | 55.56% |
| incident_report | casualties | 10 | 55.56% |
| incident_report | record_metadata | natural/location ×18 |  |
| legal_case | court | 2 | 100.00% |
| legal_case | plaintiff | 2 | 100.00% |
| legal_case | defendant | 2 | 100.00% |
| legal_case | status | 2 | 100.00% |
| legal_case | record_metadata | natural/court ×2 |  |
| person_profile | name | 30 | 100.00% |
| person_profile | role | 27 | 90.00% |
| person_profile | location | 26 | 86.67% |
| person_profile | employer | 24 | 80.00% |
| person_profile | record_metadata | natural/location ×22, natural/name ×3 |  |
| product | name | 12 | 100.00% |
| product | brand | 11 | 91.67% |
| product | condition | 11 | 91.67% |
| product | price | 10 | 83.33% |
| product | record_metadata | natural/name ×12 |  |
| real_estate_listing | address | 1 | 100.00% |
| real_estate_listing | price | 1 | 100.00% |
| real_estate_listing | area | 1 | 100.00% |
| real_estate_listing | property_type | 1 | 100.00% |
| real_estate_listing | record_metadata | natural/address ×1 |  |
| research_paper | venue | 9 | 100.00% |
| research_paper | title | 8 | 88.89% |
| research_paper | authors | 8 | 88.89% |
| research_paper | year | 8 | 88.89% |
| research_paper | topic | 2 | 22.22% |
| research_paper | record_metadata | natural/title ×8, natural/authors ×1 |  |
| software_release | product | 5 | 100.00% |
| software_release | release_date | 5 | 100.00% |
| software_release | version | 4 | 80.00% |
| software_release | platform | 2 | 40.00% |
| software_release | record_metadata | natural/product ×5 |  |
| transaction | amount | 28 | 100.00% |
| transaction | date | 26 | 92.86% |
| transaction | buyer | 24 | 85.71% |
| transaction | item | 24 | 85.71% |
| transaction | seller | 23 | 82.14% |
| transaction | record_metadata | natural/item ×24, natural/amount ×4 |  |
| vehicle_listing | make | 1 | 100.00% |
| vehicle_listing | model | 1 | 100.00% |
| vehicle_listing | year | 1 | 100.00% |
| vehicle_listing | price | 1 | 100.00% |
| vehicle_listing | record_metadata | natural/make ×1 |  |

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | llm_synthetic |
| relations | llm_synthetic |
| events | llm_synthetic |
| classifications | llm_synthetic |
| structures | llm_synthetic |
