# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_haiku45_5k | train | 4,018 | 4,018 | 95,250 | 7,043 | 14,232 | 1,123 | 1,832 | 5,740 |
| synthetic_haiku45_5k | val | 496 | 496 | 11,911 | 883 | 1,745 | 130 | 228 | 736 |
| synthetic_haiku45_5k | test | 483 | 483 | 11,477 | 832 | 1,695 | 133 | 233 | 739 |

## synthetic_haiku45_5k

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4,018 | 4,018 | 270 | 308 | 429 | 95,250 | 7,043 | 14,232 | 1,123 | 1,832 | 5,740 |
| val | 496 | 496 | 271 | 311 | 397 | 11,911 | 883 | 1,745 | 130 | 228 | 736 |
| test | 483 | 483 | 271 | 314 | 424 | 11,477 | 832 | 1,695 | 133 | 233 | 739 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 4,018 | 496 | 483 | 4,997 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 483 |
| test_unique_inputs | 483 |
| train_records | 4,018 |
| train_unique_inputs | 4,018 |
| val_records | 496 |
| val_unique_inputs | 496 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0254 | 95,250 / 11,911 | location 11%→11%, quantity 1%→1%, percent 4%→5%, duration 3%→3%, time 2%→2% |
| entities (per mention) | train / test | 0.0222 | 95,250 / 11,477 | location 11%→11%, cardinal 11%→11%, distance 1%→1%, age 1%→0%, sports team 1%→0% |
| entities (per mention) | val / test | 0.0294 | 11,911 / 11,477 | quantity 1%→1%, percent 5%→4%, distance 1%→1%, region 1%→1%, duration 3%→3% |
| relations (per instance) | train / val | 0.1225 | 7,043 / 883 | member_of 4%→8%, affiliated_with 7%→5%, employer_of 8%→6%, part_of 3%→2%, works_for 11%→12% |
| relations (per instance) | train / test | 0.1193 | 7,043 / 832 | located_in 11%→9%, works_for 11%→14%, operates_in 7%→8%, developed_by 2%→3%, affiliated_with 7%→6% |
| relations (per instance) | val / test | 0.1464 | 883 / 832 | member_of 8%→4%, located_in 11%→9%, part_of 2%→4%, leader_of 3%→4%, works_for 12%→14% |
| event types (per trigger mention) | train / val | 0.1790 WARN | 1,839 / 229 | Personnel.StartPosition 7%→10%, Award.ReceiveAward 2%→4%, Justice.Fine 1%→3%, Tech.Breach 2%→0%, Life.Evacuate 2%→0% |
| event types (per trigger mention) | train / test | 0.1564 WARN | 1,839 / 236 | Business.LaunchProduct 5%→8%, Award.ReceiveAward 2%→4%, Disaster.Rescue 1%→3%, Personnel.StartPosition 7%→8%, Disaster.NaturalDisaster 2%→4% |
| event types (per trigger mention) | val / test | 0.1947 WARN | 229 / 236 | Business.LaunchProduct 5%→8%, Justice.Fine 3%→0%, Life.Evacuate 0%→2%, Business.StartOrg 3%→5%, Personnel.StartPosition 10%→8% |
| event roles (per argument) | train / val | 0.1125 | 5,746 / 738 | Position 3%→4%, Person 4%→5%, Time 25%→24%, Organization 10%→11%, Place 13%→12% |
| event roles (per argument) | train / test | 0.0903 | 5,746 / 745 | Organization 10%→12%, Product 3%→4%, Defendant 2%→1%, Competitor 1%→0%, Agent 2%→3% |
| event roles (per argument) | val / test | 0.1255 | 738 / 745 | Time 24%→26%, Person 5%→4%, Crime 2%→0%, Product 2%→4%, Place 12%→13% |
| classifications (true_label) | train / val | 0.0847 | 14,232 / 1,745 | formality: formal 7%→6%, urgency: routine 4%→5%, subjectivity: objective 6%→7%, audience: expert 4%→5%, topic: business 4%→3% |
| classifications (true_label) | train / test | 0.0644 | 14,232 / 1,695 | certainty: asserted 7%→8%, actionability: requires action 2%→2%, language_register: technical 4%→5%, sentiment: positive 4%→4%, risk_level: none 2%→2% |
| classifications (true_label) | val / test | 0.1006 | 1,745 / 1,695 | language_register: technical 4%→5%, formality: formal 6%→7%, subjectivity: objective 7%→6%, audience: expert 5%→4%, certainty: asserted 7%→8% |

Thin support in val, entities (per mention): 46 label(s) under 10 gold. verdict ×1, political party ×1, holiday ×1, credit rating ×1, video game ×1, gene ×1, religious group ×1, war ×1, diagnosis ×1, interest rate ×1, algorithm ×2, political affiliation ×2, sentence ×2, weapon ×2, api ×2, aircraft ×2, pseudonym ×2, database ×2, medical procedure ×3, ethnic group ×3, security vulnerability ×3, framework ×3, vessel ×3, border ×3, criminal organization ×4, protocol ×4, witness ×4, vaccine ×4, patent ×4, anatomical structure ×4, currency ×4, treaty ×5, organism ×5, vehicle ×5, trade union ×6, natural disaster ×6, contract ×6, license ×6, programming language ×6, sports competition ×6, event ×7, stock ticker ×7, voltage ×7, family relationship ×8, protein ×8, hardware component ×9.
Thin support in val, relations (per instance): 39 label(s) under 10 gold. successor_of ×1, charged_with ×1, authored_by ×1, party_to ×1, owns ×1, composed_by ×1, depends_on ×1, merged_with ×2, acquired ×2, participated_in ×2, represents ×2, died_in ×2, parent_of ×2, borders ×2, sued_by ×2, allied_with ×2, capital_of ×2, published_by ×2, directed_by ×2, treats ×3, invested_in ×3, adjacent_to ×3, born_in ×3, spouse_of ×4, customer_of ×4, opposed_to ×4, distributor_of ×5, mentor_of ×5, interacts_with ×5, sponsored_by ×6, member_state_of ×6, citizen_of ×6, advisor_to ×6, designed_by ×7, manufactured_by ×7, board_member_of ×8, subsidiary_of ×9, produced_by ×9, founded ×9.
Unseen in train, present in val, event types (per trigger mention): Justice.Acquit ×1.
Thin support in val, event types (per trigger mention): 38 label(s) under 10 gold. Justice.Acquit ×1, Tech.Breach ×1, Transaction.Lend ×1, Movement.Migrate ×1, Business.DeclareBankruptcy ×1, Conflict.Ceasefire ×1, Government.EnactLaw ×1, Business.Strike ×2, Transaction.TransferOwnership ×2, Justice.Arrest ×2, Justice.Appeal ×2, Life.BeBorn ×2, Justice.Sue ×2, Disaster.Rescue ×3, Business.EndOrg ×3, Disaster.Accident ×3, Movement.Transport ×3, Contact.Negotiate ×3, Life.Injure ×3, Government.Sanction ×3, Medical.Diagnose ×3, Justice.Sentence ×3, Personnel.Nominate ×4, Justice.TrialHearing ×4, Justice.ChargeIndict ×4, Conflict.Attack ×4, Transaction.TransferMoney ×5, Business.MergeOrg ×6, Disaster.Outbreak ×6, Medical.Treat ×6, Justice.Fine ×7, Contact.Communicate ×7, Disaster.NaturalDisaster ×7, Business.StartOrg ×7, Contact.Meet ×8, Research.Publish ×9, Tech.Release ×9, Personnel.EndPosition ×9.
Thin support in val, event roles (per argument): 37 label(s) under 10 gold. Artifact ×1, Data ×1, Lender ×1, Borrower ×1, Mediator ×1, Law ×1, Striker ×2, Buyer ×2, Seller ×2, Price ×2, Origin ×2, Appellant ×2, Instrument ×2, Sentence ×2, Party ×2, Plaintiff ×2, Publication ×3, Rescuer ×3, Prosecutor ×3, Destination ×3, Disease ×3, Affected ×3, Reason ×3, Attacker ×3, Diagnosis ×3, Awarder ×4, Giver ×4, Vehicle ×4, Patient ×4, Communicator ×5, Treatment ×5, Physician ×5, Venue ×6, Victim ×6, Entity ×7, Type ×7, Target ×7.
Thin support in val, classifications (true_label): 17 label(s) under 10 gold. urgency: urgent ×1, topic: entertainment ×1, language_register: colloquial ×1, temporal_orientation: timeless ×1, document_genre: correspondence ×2, topic: travel ×2, topic: sports ×2, document_genre: opinion ×4, topic: defense ×4, topic: agriculture ×5, topic: politics ×5, topic: energy ×5, subjectivity: subjective ×7, formality: informal ×7, document_genre: review ×9, actionability: advisory ×9, topic: world ×9.
Thin support in test, entities (per mention): 39 label(s) under 10 gold. treaty ×1, trade union ×1, gene ×1, political party ×1, weapon ×1, aircraft ×1, vaccine ×1, diagnosis ×2, border ×2, market index ×2, algorithm ×2, witness ×2, license ×2, vessel ×2, criminal organization ×3, api ×3, hardware component ×3, pseudonym ×3, sentence ×3, medical procedure ×3, voltage ×3, framework ×4, database ×4, programming language ×4, contract ×5, sports competition ×5, natural disaster ×5, mountain ×6, stock ticker ×6, patent ×6, security vulnerability ×6, vehicle ×7, ethnic group ×7, language ×8, currency ×8, protocol ×8, protein ×8, family relationship ×8, pathogen ×9.
Thin support in test, relations (per instance): 40 label(s) under 10 gold. capital_of ×1, sold_to ×1, parent_of ×1, member_state_of ×1, causes ×1, named_after ×1, implements ×1, party_to ×1, derived_from ×1, predecessor_of ×1, measured_in ×1, merged_with ×1, citizen_of ×1, spouse_of ×1, directed_by ×2, adjacent_to ×2, customer_of ×2, sponsored_by ×2, allied_with ×2, charged_with ×2, treats ×3, opposed_to ×3, licensed_to ×3, sued_by ×3, interacts_with ×3, produced_by ×4, authored_by ×4, mentor_of ×4, invested_in ×5, born_in ×5, acquired ×5, owns ×5, distributor_of ×6, subsidiary_of ×6, designed_by ×6, published_by ×6, advisor_to ×6, participated_in ×7, represents ×8, board_member_of ×9.
Thin support in test, event types (per trigger mention): 38 label(s) under 10 gold. Justice.Appeal ×1, Business.DeclareBankruptcy ×1, Government.Protest ×1, Justice.Fine ×1, Personnel.Elect ×1, Justice.ChargeIndict ×1, Justice.TrialHearing ×1, Government.Sanction ×2, Justice.Sentence ×2, Transaction.Donate ×2, Movement.Transport ×2, Government.Vote ×2, Business.EndOrg ×2, Contact.Negotiate ×2, Movement.Arrive ×2, Justice.Sue ×2, Life.Die ×2, Disaster.Accident ×3, Medical.Diagnose ×3, Transaction.TransferOwnership ×4, Disaster.Outbreak ×4, Conflict.Attack ×4, Government.EnactLaw ×4, Personnel.Nominate ×4, Medical.Treat ×4, Life.Injure ×5, Transaction.TransferMoney ×5, Business.MergeOrg ×5, Life.Evacuate ×5, Tech.Breach ×5, Disaster.Rescue ×7, Research.Publish ×8, Contact.Meet ×8, Business.Layoff ×9, Contact.Communicate ×9, Research.Discover ×9, Personnel.EndPosition ×9, Disaster.NaturalDisaster ×9.
Thin support in test, event roles (per argument): 47 label(s) under 10 gold. Awarder ×1, Version ×1, Appellant ×1, Disease ×1, Voter ×1, Proposal ×1, Topic ×1, Demonstrator ×1, Prosecutor ×1, Reason ×2, Sentence ×2, Donor ×2, Instrument ×2, Venue ×2, Evacuee ×2, Traveler ×2, Physician ×2, Diagnosis ×2, Plaintiff ×2, Seller ×3, Artifact ×3, Crime ×3, Attacker ×3, Entity ×3, Law ×3, Affected ×3, Patient ×3, Buyer ×4, Price ×4, Origin ×4, Data ×4, Victim ×5, Giver ×5, Publication ×5, Treatment ×5, Defendant ×6, Adjudicator ×6, Destination ×7, Target ×8, Communicator ×8, Finding ×8, Rescuer ×8, Award ×9, Quantity ×9, Researcher ×9, Author ×9, Type ×9.
Thin support in test, classifications (true_label): 19 label(s) under 10 gold. topic: entertainment ×1, topic: defense ×2, language_register: colloquial ×2, topic: energy ×3, document_genre: opinion ×3, urgency: urgent ×3, formality: informal ×4, topic: sports ×5, topic: travel ×5, topic: politics ×5, subjectivity: subjective ×5, document_genre: review ×6, topic: world ×7, actionability: advisory ×7, temporal_orientation: timeless ×7, temporal_orientation: future ×7, document_genre: advertisement ×8, topic: education ×9, topic: agriculture ×9.

### Events

**train.** 1,839 instances in 1,522 records, 0 triggerless. Trigger gold 1,832. Argument gold 5,740. Arguments per instance, mean 3.12. Trigger length in whitespace tokens, mean 1.24, single-token share 84.3%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Research.Fund | 121 | 6.58% |
| Personnel.StartPosition | 120 | 6.53% |
| Business.LaunchProduct | 98 | 5.33% |
| Business.StartOrg | 86 | 4.68% |
| Contact.Communicate | 86 | 4.68% |
| Research.Discover | 83 | 4.51% |
| Contact.Meet | 70 | 3.81% |
| Tech.Release | 68 | 3.70% |
| Research.Publish | 63 | 3.43% |
| Personnel.EndPosition | 61 | 3.32% |
| Business.Layoff | 52 | 2.83% |
| Disaster.Outbreak | 47 | 2.56% |
| Award.ReceiveAward | 45 | 2.45% |
| Medical.Treat | 44 | 2.39% |
| Disaster.NaturalDisaster | 43 | 2.34% |
| Tech.Breach | 41 | 2.23% |
| Conflict.Attack | 40 | 2.18% |
| Disaster.Accident | 39 | 2.12% |
| Life.Injure | 38 | 2.07% |
| Transaction.TransferMoney | 38 | 2.07% |
| Transaction.TransferOwnership | 35 | 1.90% |
| Business.MergeOrg | 33 | 1.79% |
| Life.Evacuate | 31 | 1.69% |
| Medical.Diagnose | 30 | 1.63% |
| Business.EndOrg | 28 | 1.52% |
| Justice.TrialHearing | 25 | 1.36% |
| Disaster.Rescue | 24 | 1.31% |
| Justice.Sue | 24 | 1.31% |
| Justice.ChargeIndict | 24 | 1.31% |
| Contact.Negotiate | 24 | 1.31% |
| Sport.Compete | 22 | 1.20% |
| Personnel.Nominate | 21 | 1.14% |
| Justice.Fine | 21 | 1.14% |
| Life.Die | 20 | 1.09% |
| Movement.Arrive | 19 | 1.03% |
| Justice.Sentence | 16 | 0.87% |
| Government.Sanction | 16 | 0.87% |
| Movement.Transport | 16 | 0.87% |
| Life.BeBorn | 14 | 0.76% |
| Transaction.Donate | 14 | 0.76% |
| Conflict.Demonstrate | 13 | 0.71% |
| Government.EnactLaw | 12 | 0.65% |
| Transaction.Lend | 12 | 0.65% |
| Conflict.Ceasefire | 12 | 0.65% |
| Business.Strike | 10 | 0.54% |
| Justice.Arrest | 10 | 0.54% |
| Personnel.Elect | 7 | 0.38% |
| Justice.Appeal | 5 | 0.27% |
| Government.Protest | 4 | 0.22% |
| Movement.Migrate | 4 | 0.22% |
| Business.DeclareBankruptcy | 4 | 0.22% |
| Government.Vote | 3 | 0.16% |
| Life.Marry | 2 | 0.11% |
| Life.Divorce | 1 | 0.05% |

**val.** 229 instances in 181 records, 0 triggerless. Trigger gold 228. Argument gold 736. Arguments per instance, mean 3.22. Trigger length in whitespace tokens, mean 1.39, single-token share 75.5%.

#### val event types

| label | count | share |
| --- | --- | --- |
| Personnel.StartPosition | 23 | 10.04% |
| Research.Fund | 16 | 6.99% |
| Research.Discover | 11 | 4.80% |
| Business.LaunchProduct | 11 | 4.80% |
| Business.Layoff | 10 | 4.37% |
| Award.ReceiveAward | 10 | 4.37% |
| Research.Publish | 9 | 3.93% |
| Tech.Release | 9 | 3.93% |
| Personnel.EndPosition | 9 | 3.93% |
| Contact.Meet | 8 | 3.49% |
| Justice.Fine | 7 | 3.06% |
| Contact.Communicate | 7 | 3.06% |
| Disaster.NaturalDisaster | 7 | 3.06% |
| Business.StartOrg | 7 | 3.06% |
| Business.MergeOrg | 6 | 2.62% |
| Disaster.Outbreak | 6 | 2.62% |
| Medical.Treat | 6 | 2.62% |
| Transaction.TransferMoney | 5 | 2.18% |
| Personnel.Nominate | 4 | 1.75% |
| Justice.TrialHearing | 4 | 1.75% |
| Justice.ChargeIndict | 4 | 1.75% |
| Conflict.Attack | 4 | 1.75% |
| Disaster.Rescue | 3 | 1.31% |
| Business.EndOrg | 3 | 1.31% |
| Disaster.Accident | 3 | 1.31% |
| Movement.Transport | 3 | 1.31% |
| Contact.Negotiate | 3 | 1.31% |
| Life.Injure | 3 | 1.31% |
| Government.Sanction | 3 | 1.31% |
| Medical.Diagnose | 3 | 1.31% |
| Justice.Sentence | 3 | 1.31% |
| Business.Strike | 2 | 0.87% |
| Transaction.TransferOwnership | 2 | 0.87% |
| Justice.Arrest | 2 | 0.87% |
| Justice.Appeal | 2 | 0.87% |
| Life.BeBorn | 2 | 0.87% |
| Justice.Sue | 2 | 0.87% |
| Justice.Acquit | 1 | 0.44% |
| Tech.Breach | 1 | 0.44% |
| Transaction.Lend | 1 | 0.44% |
| Movement.Migrate | 1 | 0.44% |
| Business.DeclareBankruptcy | 1 | 0.44% |
| Conflict.Ceasefire | 1 | 0.44% |
| Government.EnactLaw | 1 | 0.44% |

**test.** 236 instances in 193 records, 0 triggerless. Trigger gold 233. Argument gold 739. Arguments per instance, mean 3.16. Trigger length in whitespace tokens, mean 1.32, single-token share 79.2%.

#### test event types

| label | count | share |
| --- | --- | --- |
| Personnel.StartPosition | 19 | 8.05% |
| Business.LaunchProduct | 18 | 7.63% |
| Research.Fund | 17 | 7.20% |
| Business.StartOrg | 12 | 5.08% |
| Award.ReceiveAward | 10 | 4.24% |
| Tech.Release | 10 | 4.24% |
| Business.Layoff | 9 | 3.81% |
| Contact.Communicate | 9 | 3.81% |
| Research.Discover | 9 | 3.81% |
| Personnel.EndPosition | 9 | 3.81% |
| Disaster.NaturalDisaster | 9 | 3.81% |
| Research.Publish | 8 | 3.39% |
| Contact.Meet | 8 | 3.39% |
| Disaster.Rescue | 7 | 2.97% |
| Life.Injure | 5 | 2.12% |
| Transaction.TransferMoney | 5 | 2.12% |
| Business.MergeOrg | 5 | 2.12% |
| Life.Evacuate | 5 | 2.12% |
| Tech.Breach | 5 | 2.12% |
| Transaction.TransferOwnership | 4 | 1.69% |
| Disaster.Outbreak | 4 | 1.69% |
| Conflict.Attack | 4 | 1.69% |
| Government.EnactLaw | 4 | 1.69% |
| Personnel.Nominate | 4 | 1.69% |
| Medical.Treat | 4 | 1.69% |
| Disaster.Accident | 3 | 1.27% |
| Medical.Diagnose | 3 | 1.27% |
| Government.Sanction | 2 | 0.85% |
| Justice.Sentence | 2 | 0.85% |
| Transaction.Donate | 2 | 0.85% |
| Movement.Transport | 2 | 0.85% |
| Government.Vote | 2 | 0.85% |
| Business.EndOrg | 2 | 0.85% |
| Contact.Negotiate | 2 | 0.85% |
| Movement.Arrive | 2 | 0.85% |
| Justice.Sue | 2 | 0.85% |
| Life.Die | 2 | 0.85% |
| Justice.Appeal | 1 | 0.42% |
| Business.DeclareBankruptcy | 1 | 0.42% |
| Government.Protest | 1 | 0.42% |
| Justice.Fine | 1 | 0.42% |
| Personnel.Elect | 1 | 0.42% |
| Justice.ChargeIndict | 1 | 0.42% |
| Justice.TrialHearing | 1 | 0.42% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 5,746 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 738 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 745 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.31 | 19.21% | 1.56 | 28.44% | 0.21% | 0 | 5,746 | 0 | 1,839 |
| val | 1.31 | 17.47% | 1.60 | 31.40% | 1.24% | 0 | 738 | 0 | 229 |
| test | 1.31 | 18.72% | 1.62 | 29.44% | 0.20% | 0 | 745 | 0 | 236 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1,839 | 0.00% | 53 | 60.38% | 60.38% | 0.00% | 0.00% | 1.84% |
| val | 229 | 0.00% | 6 | 100.00% | 100.00% | 0.00% | 0.00% | 2.21% |
| test | 236 | 0.00% | 12 | 100.00% | 100.00% | 0.00% | 0.00% | 2.59% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 0 | 0 | 53 |
| val | 0 | 0 | 6 |
| test | 0 | 0 | 12 |

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
| date | 13,002 | 13.65% |
| cardinal | 10,674 | 11.21% |
| location | 10,074 | 10.58% |
| person | 8,031 | 8.43% |
| money | 5,691 | 5.97% |
| organization | 4,787 | 5.03% |
| percent | 4,137 | 4.34% |
| company | 3,919 | 4.11% |
| job title | 3,849 | 4.04% |
| duration | 3,151 | 3.31% |
| time | 1,949 | 2.05% |
| government agency | 1,645 | 1.73% |
| geopolitical entity | 1,552 | 1.63% |
| measurement | 1,411 | 1.48% |
| educational institution | 1,329 | 1.40% |
| facility | 1,283 | 1.35% |
| product | 1,162 | 1.22% |
| region | 1,087 | 1.14% |
| distance | 874 | 0.92% |
| quantity | 771 | 0.81% |
| academic degree | 717 | 0.75% |
| disease | 701 | 0.74% |
| address | 627 | 0.66% |
| occupation | 584 | 0.61% |
| regulatory body | 571 | 0.60% |
| age | 544 | 0.57% |
| sports team | 501 | 0.53% |
| landmark | 476 | 0.50% |
| fiscal quarter | 473 | 0.50% |
| medication | 446 | 0.47% |
| area | 441 | 0.46% |
| temperature | 390 | 0.41% |
| frequency | 349 | 0.37% |
| nationality | 341 | 0.36% |
| software | 339 | 0.36% |
| road | 333 | 0.35% |
| speed | 327 | 0.34% |
| dosage | 285 | 0.30% |
| military unit | 282 | 0.30% |
| ordinal | 257 | 0.27% |
| regulation | 254 | 0.27% |
| non-profit organization | 249 | 0.26% |
| media outlet | 226 | 0.24% |
| body of water | 225 | 0.24% |
| chemical | 217 | 0.23% |
| award | 215 | 0.23% |
| deadline | 214 | 0.22% |
| publication | 201 | 0.21% |
| model number | 194 | 0.20% |
| clinical trial | 177 | 0.19% |
| airport | 171 | 0.18% |
| team role | 167 | 0.18% |
| law | 167 | 0.18% |
| symptom | 165 | 0.17% |
| court case | 152 | 0.16% |
| brand | 152 | 0.16% |
| weight | 151 | 0.16% |
| operating system | 143 | 0.15% |
| postal code | 140 | 0.15% |
| crime | 135 | 0.14% |
| military rank | 134 | 0.14% |
| era | 129 | 0.14% |
| conference | 118 | 0.12% |
| organism | 111 | 0.12% |
| event | 111 | 0.12% |
| honorific | 96 | 0.10% |
| work of art | 85 | 0.09% |
| language | 73 | 0.08% |
| sports competition | 70 | 0.07% |
| mountain | 70 | 0.07% |
| protein | 65 | 0.07% |
| family relationship | 57 | 0.06% |
| patent | 57 | 0.06% |
| natural disaster | 53 | 0.06% |
| voltage | 52 | 0.05% |
| stock ticker | 51 | 0.05% |
| currency | 50 | 0.05% |
| security vulnerability | 49 | 0.05% |
| programming language | 49 | 0.05% |
| contract | 46 | 0.05% |
| ethnic group | 45 | 0.05% |
| criminal organization | 44 | 0.05% |
| pathogen | 41 | 0.04% |
| medical procedure | 40 | 0.04% |
| hardware component | 38 | 0.04% |
| protocol | 35 | 0.04% |
| trade union | 29 | 0.03% |
| license | 26 | 0.03% |
| vaccine | 25 | 0.03% |
| border | 25 | 0.03% |
| species | 25 | 0.03% |
| aircraft | 24 | 0.03% |
| gene | 21 | 0.02% |
| database | 20 | 0.02% |
| framework | 18 | 0.02% |
| vehicle | 18 | 0.02% |
| algorithm | 18 | 0.02% |
| anatomical structure | 16 | 0.02% |
| pseudonym | 16 | 0.02% |
| treaty | 14 | 0.01% |
| witness | 13 | 0.01% |
| sentence | 12 | 0.01% |
| api | 11 | 0.01% |
| political affiliation | 10 | 0.01% |
| file format | 9 | 0.01% |
| interest rate | 8 | 0.01% |
| credit rating | 6 | 0.01% |
| holiday | 6 | 0.01% |
| diagnosis | 5 | 0.01% |
| market index | 4 | 0.00% |
| political party | 4 | 0.00% |
| religious group | 4 | 0.00% |
| book | 4 | 0.00% |
| war | 4 | 0.00% |
| verdict | 3 | 0.00% |
| weapon | 3 | 0.00% |
| film | 1 | 0.00% |
| video game | 1 | 0.00% |
| vessel | 1 | 0.00% |
| song | 0 | 0.00% |
| allergen | 0 | 0.00% |
| mineral | 0 | 0.00% |
| tax | 0 | 0.00% |
| financial instrument | 0 | 0.00% |
| budget line | 0 | 0.00% |

#### val

| label | count | share |
| --- | --- | --- |
| date | 1,606 | 13.48% |
| cardinal | 1,321 | 11.09% |
| location | 1,300 | 10.91% |
| person | 996 | 8.36% |
| money | 700 | 5.88% |
| organization | 597 | 5.01% |
| percent | 552 | 4.63% |
| job title | 481 | 4.04% |
| company | 474 | 3.98% |
| duration | 369 | 3.10% |
| time | 221 | 1.86% |
| government agency | 211 | 1.77% |
| geopolitical entity | 196 | 1.65% |
| measurement | 186 | 1.56% |
| educational institution | 171 | 1.44% |
| facility | 153 | 1.28% |
| quantity | 133 | 1.12% |
| product | 130 | 1.09% |
| region | 116 | 0.97% |
| distance | 99 | 0.83% |
| academic degree | 99 | 0.83% |
| disease | 93 | 0.78% |
| address | 79 | 0.66% |
| occupation | 77 | 0.65% |
| landmark | 75 | 0.63% |
| regulatory body | 73 | 0.61% |
| temperature | 62 | 0.52% |
| sports team | 59 | 0.50% |
| age | 58 | 0.49% |
| fiscal quarter | 55 | 0.46% |
| medication | 52 | 0.44% |
| area | 48 | 0.40% |
| road | 45 | 0.38% |
| non-profit organization | 42 | 0.35% |
| speed | 40 | 0.34% |
| frequency | 39 | 0.33% |
| body of water | 39 | 0.33% |
| dosage | 39 | 0.33% |
| regulation | 38 | 0.32% |
| software | 38 | 0.32% |
| ordinal | 35 | 0.29% |
| nationality | 34 | 0.29% |
| military unit | 34 | 0.29% |
| symptom | 29 | 0.24% |
| media outlet | 28 | 0.24% |
| award | 26 | 0.22% |
| brand | 26 | 0.22% |
| chemical | 23 | 0.19% |
| model number | 23 | 0.19% |
| postal code | 23 | 0.19% |
| law | 23 | 0.19% |
| court case | 22 | 0.18% |
| weight | 21 | 0.18% |
| airport | 19 | 0.16% |
| clinical trial | 19 | 0.16% |
| publication | 19 | 0.16% |
| military rank | 19 | 0.16% |
| team role | 19 | 0.16% |
| conference | 18 | 0.15% |
| deadline | 17 | 0.14% |
| era | 16 | 0.13% |
| work of art | 15 | 0.13% |
| operating system | 15 | 0.13% |
| honorific | 14 | 0.12% |
| crime | 12 | 0.10% |
| language | 11 | 0.09% |
| mountain | 10 | 0.08% |
| pathogen | 10 | 0.08% |
| hardware component | 9 | 0.08% |
| family relationship | 8 | 0.07% |
| protein | 8 | 0.07% |
| event | 7 | 0.06% |
| stock ticker | 7 | 0.06% |
| voltage | 7 | 0.06% |
| trade union | 6 | 0.05% |
| natural disaster | 6 | 0.05% |
| contract | 6 | 0.05% |
| license | 6 | 0.05% |
| programming language | 6 | 0.05% |
| sports competition | 6 | 0.05% |
| treaty | 5 | 0.04% |
| organism | 5 | 0.04% |
| vehicle | 5 | 0.04% |
| criminal organization | 4 | 0.03% |
| protocol | 4 | 0.03% |
| witness | 4 | 0.03% |
| vaccine | 4 | 0.03% |
| patent | 4 | 0.03% |
| anatomical structure | 4 | 0.03% |
| currency | 4 | 0.03% |
| medical procedure | 3 | 0.03% |
| ethnic group | 3 | 0.03% |
| security vulnerability | 3 | 0.03% |
| framework | 3 | 0.03% |
| vessel | 3 | 0.03% |
| border | 3 | 0.03% |
| algorithm | 2 | 0.02% |
| political affiliation | 2 | 0.02% |
| sentence | 2 | 0.02% |
| weapon | 2 | 0.02% |
| api | 2 | 0.02% |
| aircraft | 2 | 0.02% |
| pseudonym | 2 | 0.02% |
| database | 2 | 0.02% |
| verdict | 1 | 0.01% |
| political party | 1 | 0.01% |
| holiday | 1 | 0.01% |
| credit rating | 1 | 0.01% |
| video game | 1 | 0.01% |
| gene | 1 | 0.01% |
| religious group | 1 | 0.01% |
| war | 1 | 0.01% |
| diagnosis | 1 | 0.01% |
| interest rate | 1 | 0.01% |
| financial instrument | 0 | 0.00% |
| mineral | 0 | 0.00% |
| species | 0 | 0.00% |
| song | 0 | 0.00% |
| file format | 0 | 0.00% |
| tax | 0 | 0.00% |
| allergen | 0 | 0.00% |
| budget line | 0 | 0.00% |
| film | 0 | 0.00% |
| market index | 0 | 0.00% |
| book | 0 | 0.00% |

#### test

| label | count | share |
| --- | --- | --- |
| date | 1,562 | 13.61% |
| location | 1,267 | 11.04% |
| cardinal | 1,251 | 10.90% |
| person | 958 | 8.35% |
| money | 680 | 5.92% |
| organization | 574 | 5.00% |
| percent | 497 | 4.33% |
| company | 471 | 4.10% |
| job title | 469 | 4.09% |
| duration | 384 | 3.35% |
| time | 232 | 2.02% |
| government agency | 203 | 1.77% |
| geopolitical entity | 181 | 1.58% |
| educational institution | 174 | 1.52% |
| measurement | 170 | 1.48% |
| product | 153 | 1.33% |
| facility | 143 | 1.25% |
| region | 141 | 1.23% |
| distance | 129 | 1.12% |
| disease | 90 | 0.78% |
| quantity | 85 | 0.74% |
| academic degree | 78 | 0.68% |
| address | 72 | 0.63% |
| regulatory body | 69 | 0.60% |
| occupation | 66 | 0.58% |
| medication | 59 | 0.51% |
| landmark | 57 | 0.50% |
| temperature | 52 | 0.45% |
| fiscal quarter | 52 | 0.45% |
| age | 50 | 0.44% |
| speed | 46 | 0.40% |
| frequency | 46 | 0.40% |
| software | 45 | 0.39% |
| sports team | 45 | 0.39% |
| nationality | 43 | 0.37% |
| dosage | 43 | 0.37% |
| area | 42 | 0.37% |
| road | 37 | 0.32% |
| regulation | 36 | 0.31% |
| military unit | 36 | 0.31% |
| model number | 29 | 0.25% |
| law | 29 | 0.25% |
| ordinal | 29 | 0.25% |
| symptom | 26 | 0.23% |
| body of water | 26 | 0.23% |
| non-profit organization | 26 | 0.23% |
| award | 25 | 0.22% |
| deadline | 25 | 0.22% |
| postal code | 24 | 0.21% |
| weight | 23 | 0.20% |
| airport | 22 | 0.19% |
| publication | 21 | 0.18% |
| brand | 20 | 0.17% |
| court case | 18 | 0.16% |
| clinical trial | 18 | 0.16% |
| honorific | 17 | 0.15% |
| military rank | 17 | 0.15% |
| chemical | 17 | 0.15% |
| operating system | 17 | 0.15% |
| era | 17 | 0.15% |
| conference | 15 | 0.13% |
| team role | 14 | 0.12% |
| event | 13 | 0.11% |
| media outlet | 12 | 0.10% |
| work of art | 12 | 0.10% |
| organism | 11 | 0.10% |
| crime | 10 | 0.09% |
| pathogen | 9 | 0.08% |
| language | 8 | 0.07% |
| currency | 8 | 0.07% |
| protocol | 8 | 0.07% |
| protein | 8 | 0.07% |
| family relationship | 8 | 0.07% |
| vehicle | 7 | 0.06% |
| ethnic group | 7 | 0.06% |
| mountain | 6 | 0.05% |
| stock ticker | 6 | 0.05% |
| patent | 6 | 0.05% |
| security vulnerability | 6 | 0.05% |
| contract | 5 | 0.04% |
| sports competition | 5 | 0.04% |
| natural disaster | 5 | 0.04% |
| framework | 4 | 0.03% |
| database | 4 | 0.03% |
| programming language | 4 | 0.03% |
| criminal organization | 3 | 0.03% |
| api | 3 | 0.03% |
| hardware component | 3 | 0.03% |
| pseudonym | 3 | 0.03% |
| sentence | 3 | 0.03% |
| medical procedure | 3 | 0.03% |
| voltage | 3 | 0.03% |
| diagnosis | 2 | 0.02% |
| border | 2 | 0.02% |
| market index | 2 | 0.02% |
| algorithm | 2 | 0.02% |
| witness | 2 | 0.02% |
| license | 2 | 0.02% |
| vessel | 2 | 0.02% |
| treaty | 1 | 0.01% |
| trade union | 1 | 0.01% |
| gene | 1 | 0.01% |
| political party | 1 | 0.01% |
| weapon | 1 | 0.01% |
| aircraft | 1 | 0.01% |
| vaccine | 1 | 0.01% |
| mineral | 0 | 0.00% |
| film | 0 | 0.00% |
| anatomical structure | 0 | 0.00% |
| political affiliation | 0 | 0.00% |
| video game | 0 | 0.00% |
| interest rate | 0 | 0.00% |
| allergen | 0 | 0.00% |
| file format | 0 | 0.00% |
| credit rating | 0 | 0.00% |
| verdict | 0 | 0.00% |
| religious group | 0 | 0.00% |
| financial instrument | 0 | 0.00% |
| book | 0 | 0.00% |
| holiday | 0 | 0.00% |
| song | 0 | 0.00% |
| tax | 0 | 0.00% |
| war | 0 | 0.00% |
| budget line | 0 | 0.00% |
| species | 0 | 0.00% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| works_for | 805 | 11.43% |
| located_in | 805 | 11.43% |
| based_in | 606 | 8.60% |
| employer_of | 548 | 7.78% |
| affiliated_with | 508 | 7.21% |
| operates_in | 458 | 6.50% |
| headquartered_in | 434 | 6.16% |
| member_of | 304 | 4.32% |
| leader_of | 245 | 3.48% |
| part_of | 226 | 3.21% |
| reports_to | 201 | 2.85% |
| colleague_of | 188 | 2.67% |
| developed_by | 123 | 1.75% |
| board_member_of | 120 | 1.70% |
| competitor_of | 103 | 1.46% |
| founded | 101 | 1.43% |
| regulated_by | 97 | 1.38% |
| produced_by | 64 | 0.91% |
| invested_in | 62 | 0.88% |
| manufactured_by | 62 | 0.88% |
| supplier_of | 59 | 0.84% |
| designed_by | 57 | 0.81% |
| sponsored_by | 52 | 0.74% |
| customer_of | 51 | 0.72% |
| distributor_of | 50 | 0.71% |
| represents | 43 | 0.61% |
| interacts_with | 43 | 0.61% |
| compatible_with | 39 | 0.55% |
| advisor_to | 38 | 0.54% |
| party_to | 38 | 0.54% |
| subsidiary_of | 35 | 0.50% |
| acquired | 34 | 0.48% |
| owns | 32 | 0.45% |
| participated_in | 30 | 0.43% |
| published_by | 26 | 0.37% |
| born_in | 25 | 0.35% |
| treats | 24 | 0.34% |
| authored_by | 23 | 0.33% |
| adjacent_to | 18 | 0.26% |
| sold_to | 18 | 0.26% |
| testified_against | 18 | 0.26% |
| mentor_of | 18 | 0.26% |
| citizen_of | 17 | 0.24% |
| spouse_of | 15 | 0.21% |
| merged_with | 15 | 0.21% |
| charged_with | 14 | 0.20% |
| allied_with | 13 | 0.18% |
| sued_by | 12 | 0.17% |
| relative_of | 12 | 0.17% |
| derived_from | 9 | 0.13% |
| opposed_to | 9 | 0.13% |
| predecessor_of | 8 | 0.11% |
| named_after | 8 | 0.11% |
| died_in | 7 | 0.10% |
| licensed_to | 7 | 0.10% |
| implements | 7 | 0.10% |
| member_state_of | 7 | 0.10% |
| won | 6 | 0.09% |
| capital_of | 6 | 0.09% |
| supersedes | 6 | 0.09% |
| child_of | 5 | 0.07% |
| borders | 4 | 0.06% |
| directed_by | 4 | 0.06% |
| composed_by | 4 | 0.06% |
| successor_of | 3 | 0.04% |
| signatory_of | 3 | 0.04% |
| parent_of | 3 | 0.04% |
| depends_on | 3 | 0.04% |
| convicted_of | 2 | 0.03% |
| causes | 1 | 0.01% |
| sibling_of | 1 | 0.01% |
| measured_in | 1 | 0.01% |

#### val

| label | count | share |
| --- | --- | --- |
| works_for | 110 | 12.46% |
| located_in | 99 | 11.21% |
| based_in | 82 | 9.29% |
| member_of | 72 | 8.15% |
| operates_in | 66 | 7.47% |
| employer_of | 55 | 6.23% |
| headquartered_in | 52 | 5.89% |
| affiliated_with | 48 | 5.44% |
| leader_of | 25 | 2.83% |
| colleague_of | 25 | 2.83% |
| reports_to | 22 | 2.49% |
| part_of | 19 | 2.15% |
| developed_by | 17 | 1.93% |
| regulated_by | 16 | 1.81% |
| supplier_of | 12 | 1.36% |
| competitor_of | 10 | 1.13% |
| compatible_with | 10 | 1.13% |
| subsidiary_of | 9 | 1.02% |
| produced_by | 9 | 1.02% |
| founded | 9 | 1.02% |
| board_member_of | 8 | 0.91% |
| designed_by | 7 | 0.79% |
| manufactured_by | 7 | 0.79% |
| sponsored_by | 6 | 0.68% |
| member_state_of | 6 | 0.68% |
| citizen_of | 6 | 0.68% |
| advisor_to | 6 | 0.68% |
| distributor_of | 5 | 0.57% |
| mentor_of | 5 | 0.57% |
| interacts_with | 5 | 0.57% |
| spouse_of | 4 | 0.45% |
| customer_of | 4 | 0.45% |
| opposed_to | 4 | 0.45% |
| treats | 3 | 0.34% |
| invested_in | 3 | 0.34% |
| adjacent_to | 3 | 0.34% |
| born_in | 3 | 0.34% |
| merged_with | 2 | 0.23% |
| acquired | 2 | 0.23% |
| participated_in | 2 | 0.23% |
| represents | 2 | 0.23% |
| died_in | 2 | 0.23% |
| parent_of | 2 | 0.23% |
| borders | 2 | 0.23% |
| sued_by | 2 | 0.23% |
| allied_with | 2 | 0.23% |
| capital_of | 2 | 0.23% |
| published_by | 2 | 0.23% |
| directed_by | 2 | 0.23% |
| successor_of | 1 | 0.11% |
| charged_with | 1 | 0.11% |
| authored_by | 1 | 0.11% |
| party_to | 1 | 0.11% |
| owns | 1 | 0.11% |
| composed_by | 1 | 0.11% |
| depends_on | 1 | 0.11% |

#### test

| label | count | share |
| --- | --- | --- |
| works_for | 115 | 13.82% |
| located_in | 75 | 9.01% |
| operates_in | 70 | 8.41% |
| based_in | 66 | 7.93% |
| employer_of | 59 | 7.09% |
| affiliated_with | 52 | 6.25% |
| headquartered_in | 49 | 5.89% |
| leader_of | 36 | 4.33% |
| member_of | 32 | 3.85% |
| part_of | 31 | 3.73% |
| developed_by | 23 | 2.76% |
| colleague_of | 19 | 2.28% |
| reports_to | 18 | 2.16% |
| regulated_by | 14 | 1.68% |
| competitor_of | 14 | 1.68% |
| supplier_of | 11 | 1.32% |
| manufactured_by | 11 | 1.32% |
| founded | 10 | 1.20% |
| board_member_of | 9 | 1.08% |
| represents | 8 | 0.96% |
| participated_in | 7 | 0.84% |
| distributor_of | 6 | 0.72% |
| subsidiary_of | 6 | 0.72% |
| designed_by | 6 | 0.72% |
| published_by | 6 | 0.72% |
| advisor_to | 6 | 0.72% |
| invested_in | 5 | 0.60% |
| born_in | 5 | 0.60% |
| acquired | 5 | 0.60% |
| owns | 5 | 0.60% |
| produced_by | 4 | 0.48% |
| authored_by | 4 | 0.48% |
| mentor_of | 4 | 0.48% |
| treats | 3 | 0.36% |
| opposed_to | 3 | 0.36% |
| licensed_to | 3 | 0.36% |
| sued_by | 3 | 0.36% |
| interacts_with | 3 | 0.36% |
| directed_by | 2 | 0.24% |
| adjacent_to | 2 | 0.24% |
| customer_of | 2 | 0.24% |
| sponsored_by | 2 | 0.24% |
| allied_with | 2 | 0.24% |
| charged_with | 2 | 0.24% |
| capital_of | 1 | 0.12% |
| sold_to | 1 | 0.12% |
| parent_of | 1 | 0.12% |
| member_state_of | 1 | 0.12% |
| causes | 1 | 0.12% |
| named_after | 1 | 0.12% |
| implements | 1 | 0.12% |
| party_to | 1 | 0.12% |
| derived_from | 1 | 0.12% |
| predecessor_of | 1 | 0.12% |
| measured_in | 1 | 0.12% |
| merged_with | 1 | 0.12% |
| citizen_of | 1 | 0.12% |
| spouse_of | 1 | 0.12% |

### Classifications

**train, actionability.** Majority class 53.3% of 1,019 answers. The marginal is `true_label`, not the menu.

#### train actionability

| label | count | share |
| --- | --- | --- |
| informational | 543 | 53.29% |
| requires action | 350 | 34.35% |
| advisory | 126 | 12.37% |

**train, audience.** Majority class 34.0% of 1,805 answers. The marginal is `true_label`, not the menu.

#### train audience

| label | count | share |
| --- | --- | --- |
| expert | 614 | 34.02% |
| general public | 575 | 31.86% |
| regulatory | 436 | 24.16% |
| internal | 180 | 9.97% |

**train, certainty.** Majority class 99.5% of 1,021 answers. The marginal is `true_label`, not the menu.

#### train certainty

| label | count | share |
| --- | --- | --- |
| asserted | 1,016 | 99.51% |
| hedged | 4 | 0.39% |
| speculative | 1 | 0.10% |

**train, document_genre.** Majority class 26.6% of 1,036 answers. The marginal is `true_label`, not the menu.

#### train document_genre

| label | count | share |
| --- | --- | --- |
| news report | 276 | 26.64% |
| announcement | 258 | 24.90% |
| technical documentation | 132 | 12.74% |
| correspondence | 117 | 11.29% |
| legal filing | 90 | 8.69% |
| advertisement | 66 | 6.37% |
| review | 63 | 6.08% |
| opinion | 34 | 3.28% |

**train, formality.** Majority class 95.2% of 1,030 answers. The marginal is `true_label`, not the menu.

#### train formality

| label | count | share |
| --- | --- | --- |
| formal | 981 | 95.24% |
| informal | 49 | 4.76% |

**train, language_register.** Majority class 63.5% of 986 answers. The marginal is `true_label`, not the menu.

#### train language_register

| label | count | share |
| --- | --- | --- |
| technical | 626 | 63.49% |
| plain | 253 | 25.66% |
| legal | 98 | 9.94% |
| colloquial | 9 | 0.91% |

**train, risk_level.** Majority class 33.6% of 985 answers. The marginal is `true_label`, not the menu.

#### train risk_level

| label | count | share |
| --- | --- | --- |
| low | 331 | 33.60% |
| high | 284 | 28.83% |
| none | 232 | 23.55% |
| moderate | 138 | 14.01% |

**train, sentiment.** Majority class 50.9% of 981 answers. The marginal is `true_label`, not the menu.

#### train sentiment

| label | count | share |
| --- | --- | --- |
| positive | 499 | 50.87% |
| negative | 309 | 31.50% |
| neutral | 173 | 17.64% |

**train, subjectivity.** Majority class 93.1% of 979 answers. The marginal is `true_label`, not the menu.

#### train subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 911 | 93.05% |
| subjective | 68 | 6.95% |

**train, temporal_orientation.** Majority class 71.5% of 984 answers. The marginal is `true_label`, not the menu.

#### train temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 704 | 71.54% |
| present | 174 | 17.68% |
| future | 83 | 8.43% |
| timeless | 23 | 2.34% |

**train, topic.** Majority class 23.0% of 2,439 answers. The marginal is `true_label`, not the menu.

#### train topic

| label | count | share |
| --- | --- | --- |
| business | 560 | 22.96% |
| technology | 347 | 14.23% |
| finance | 251 | 10.29% |
| science | 229 | 9.39% |
| health | 209 | 8.57% |
| legal | 187 | 7.67% |
| environment | 179 | 7.34% |
| world | 96 | 3.94% |
| politics | 79 | 3.24% |
| defense | 71 | 2.91% |
| education | 59 | 2.42% |
| travel | 58 | 2.38% |
| sports | 42 | 1.72% |
| agriculture | 40 | 1.64% |
| energy | 25 | 1.03% |
| entertainment | 7 | 0.29% |

**train, urgency.** Majority class 57.5% of 967 answers. The marginal is `true_label`, not the menu.

#### train urgency

| label | count | share |
| --- | --- | --- |
| routine | 556 | 57.50% |
| elevated | 226 | 23.37% |
| critical | 171 | 17.68% |
| urgent | 14 | 1.45% |

**val, actionability.** Majority class 61.3% of 111 answers. The marginal is `true_label`, not the menu.

#### val actionability

| label | count | share |
| --- | --- | --- |
| informational | 68 | 61.26% |
| requires action | 34 | 30.63% |
| advisory | 9 | 8.11% |

**val, audience.** Majority class 36.5% of 249 answers. The marginal is `true_label`, not the menu.

#### val audience

| label | count | share |
| --- | --- | --- |
| expert | 91 | 36.55% |
| general public | 75 | 30.12% |
| regulatory | 63 | 25.30% |
| internal | 20 | 8.03% |

**val, certainty.** Majority class 100.0% of 127 answers. The marginal is `true_label`, not the menu.

#### val certainty

| label | count | share |
| --- | --- | --- |
| asserted | 127 | 100.00% |

**val, document_genre.** Majority class 30.7% of 114 answers. The marginal is `true_label`, not the menu.

#### val document_genre

| label | count | share |
| --- | --- | --- |
| news report | 35 | 30.70% |
| announcement | 24 | 21.05% |
| technical documentation | 17 | 14.91% |
| advertisement | 13 | 11.40% |
| legal filing | 10 | 8.77% |
| review | 9 | 7.89% |
| opinion | 4 | 3.51% |
| correspondence | 2 | 1.75% |

**val, formality.** Majority class 93.4% of 106 answers. The marginal is `true_label`, not the menu.

#### val formality

| label | count | share |
| --- | --- | --- |
| formal | 99 | 93.40% |
| informal | 7 | 6.60% |

**val, language_register.** Majority class 57.6% of 118 answers. The marginal is `true_label`, not the menu.

#### val language_register

| label | count | share |
| --- | --- | --- |
| technical | 68 | 57.63% |
| plain | 35 | 29.66% |
| legal | 14 | 11.86% |
| colloquial | 1 | 0.85% |

**val, risk_level.** Majority class 37.3% of 134 answers. The marginal is `true_label`, not the menu.

#### val risk_level

| label | count | share |
| --- | --- | --- |
| low | 50 | 37.31% |
| high | 43 | 32.09% |
| none | 29 | 21.64% |
| moderate | 12 | 8.96% |

**val, sentiment.** Majority class 56.8% of 118 answers. The marginal is `true_label`, not the menu.

#### val sentiment

| label | count | share |
| --- | --- | --- |
| positive | 67 | 56.78% |
| negative | 33 | 27.97% |
| neutral | 18 | 15.25% |

**val, subjectivity.** Majority class 94.9% of 137 answers. The marginal is `true_label`, not the menu.

#### val subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 130 | 94.89% |
| subjective | 7 | 5.11% |

**val, temporal_orientation.** Majority class 71.4% of 126 answers. The marginal is `true_label`, not the menu.

#### val temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 90 | 71.43% |
| present | 25 | 19.84% |
| future | 10 | 7.94% |
| timeless | 1 | 0.79% |

**val, topic.** Majority class 20.8% of 255 answers. The marginal is `true_label`, not the menu.

#### val topic

| label | count | share |
| --- | --- | --- |
| business | 53 | 20.78% |
| technology | 37 | 14.51% |
| science | 36 | 14.12% |
| health | 30 | 11.76% |
| finance | 24 | 9.41% |
| environment | 17 | 6.67% |
| legal | 15 | 5.88% |
| education | 10 | 3.92% |
| world | 9 | 3.53% |
| agriculture | 5 | 1.96% |
| politics | 5 | 1.96% |
| energy | 5 | 1.96% |
| defense | 4 | 1.57% |
| travel | 2 | 0.78% |
| sports | 2 | 0.78% |
| entertainment | 1 | 0.39% |

**val, urgency.** Majority class 58.0% of 150 answers. The marginal is `true_label`, not the menu.

#### val urgency

| label | count | share |
| --- | --- | --- |
| routine | 87 | 58.00% |
| elevated | 36 | 24.00% |
| critical | 26 | 17.33% |
| urgent | 1 | 0.67% |

**test, actionability.** Majority class 67.3% of 104 answers. The marginal is `true_label`, not the menu.

#### test actionability

| label | count | share |
| --- | --- | --- |
| informational | 70 | 67.31% |
| requires action | 27 | 25.96% |
| advisory | 7 | 6.73% |

**test, audience.** Majority class 34.1% of 205 answers. The marginal is `true_label`, not the menu.

#### test audience

| label | count | share |
| --- | --- | --- |
| expert | 70 | 34.15% |
| general public | 65 | 31.71% |
| regulatory | 49 | 23.90% |
| internal | 21 | 10.24% |

**test, certainty.** Majority class 100.0% of 140 answers. The marginal is `true_label`, not the menu.

#### test certainty

| label | count | share |
| --- | --- | --- |
| asserted | 140 | 100.00% |

**test, document_genre.** Majority class 27.2% of 114 answers. The marginal is `true_label`, not the menu.

#### test document_genre

| label | count | share |
| --- | --- | --- |
| news report | 31 | 27.19% |
| announcement | 31 | 27.19% |
| legal filing | 14 | 12.28% |
| correspondence | 11 | 9.65% |
| technical documentation | 10 | 8.77% |
| advertisement | 8 | 7.02% |
| review | 6 | 5.26% |
| opinion | 3 | 2.63% |

**test, formality.** Majority class 96.7% of 122 answers. The marginal is `true_label`, not the menu.

#### test formality

| label | count | share |
| --- | --- | --- |
| formal | 118 | 96.72% |
| informal | 4 | 3.28% |

**test, language_register.** Majority class 70.6% of 126 answers. The marginal is `true_label`, not the menu.

#### test language_register

| label | count | share |
| --- | --- | --- |
| technical | 89 | 70.63% |
| plain | 22 | 17.46% |
| legal | 13 | 10.32% |
| colloquial | 2 | 1.59% |

**test, risk_level.** Majority class 34.3% of 134 answers. The marginal is `true_label`, not the menu.

#### test risk_level

| label | count | share |
| --- | --- | --- |
| low | 46 | 34.33% |
| none | 40 | 29.85% |
| high | 33 | 24.63% |
| moderate | 15 | 11.19% |

**test, sentiment.** Majority class 60.0% of 120 answers. The marginal is `true_label`, not the menu.

#### test sentiment

| label | count | share |
| --- | --- | --- |
| positive | 72 | 60.00% |
| negative | 31 | 25.83% |
| neutral | 17 | 14.17% |

**test, subjectivity.** Majority class 95.5% of 111 answers. The marginal is `true_label`, not the menu.

#### test subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 106 | 95.50% |
| subjective | 5 | 4.50% |

**test, temporal_orientation.** Majority class 68.5% of 124 answers. The marginal is `true_label`, not the menu.

#### test temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 85 | 68.55% |
| present | 25 | 20.16% |
| timeless | 7 | 5.65% |
| future | 7 | 5.65% |

**test, topic.** Majority class 24.6% of 276 answers. The marginal is `true_label`, not the menu.

#### test topic

| label | count | share |
| --- | --- | --- |
| business | 68 | 24.64% |
| technology | 51 | 18.48% |
| finance | 25 | 9.06% |
| science | 24 | 8.70% |
| health | 24 | 8.70% |
| environment | 21 | 7.61% |
| legal | 17 | 6.16% |
| education | 9 | 3.26% |
| agriculture | 9 | 3.26% |
| world | 7 | 2.54% |
| sports | 5 | 1.81% |
| travel | 5 | 1.81% |
| politics | 5 | 1.81% |
| energy | 3 | 1.09% |
| defense | 2 | 0.72% |
| entertainment | 1 | 0.36% |

**test, urgency.** Majority class 57.1% of 119 answers. The marginal is `true_label`, not the menu.

#### test urgency

| label | count | share |
| --- | --- | --- |
| routine | 68 | 57.14% |
| elevated | 26 | 21.85% |
| critical | 22 | 18.49% |
| urgent | 3 | 2.52% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | condition | 56 | 94.92% |
| clinical_finding | patient_group | 52 | 88.14% |
| clinical_finding | treatment | 46 | 77.97% |
| clinical_finding | outcome | 39 | 66.10% |
| clinical_finding | record_metadata | natural/condition ×56, natural/patient_group ×3 |  |
| company_profile | name | 222 | 100.00% |
| company_profile | size | 220 | 99.10% |
| company_profile | headquarters | 200 | 90.09% |
| company_profile | industry | 142 | 63.96% |
| company_profile | founded | 86 | 38.74% |
| company_profile | record_metadata | natural/name ×214 |  |
| event_listing | date | 53 | 84.13% |
| event_listing | venue | 51 | 80.95% |
| event_listing | name | 49 | 77.78% |
| event_listing | organizer | 49 | 77.78% |
| event_listing | price | 12 | 19.05% |
| event_listing | record_metadata | natural/name ×49, natural/venue ×13, natural/price ×1 |  |
| financial_report | organization | 64 | 100.00% |
| financial_report | period | 59 | 92.19% |
| financial_report | revenue | 55 | 85.94% |
| financial_report | net_income | 43 | 67.19% |
| financial_report | record_metadata | natural/organization ×64 |  |
| incident_report | severity | 132 | 100.00% |
| incident_report | date | 131 | 99.24% |
| incident_report | location | 125 | 94.70% |
| incident_report | incident_type | 85 | 64.39% |
| incident_report | casualties | 71 | 53.79% |
| incident_report | record_metadata | natural/location ×125, natural/incident_type ×4, natural/date ×2, natural/severity ×1 |  |
| job_posting | title | 17 | 100.00% |
| job_posting | company | 17 | 100.00% |
| job_posting | employment_type | 17 | 100.00% |
| job_posting | location | 16 | 94.12% |
| job_posting | record_metadata | natural/location ×11, natural/title ×1 |  |
| legal_case | status | 31 | 100.00% |
| legal_case | court | 30 | 96.77% |
| legal_case | plaintiff | 28 | 90.32% |
| legal_case | defendant | 27 | 87.10% |
| legal_case | case_name | 16 | 51.61% |
| legal_case | record_metadata | natural/case_name ×16, natural/court ×14, natural/status ×1 |  |
| person_profile | name | 182 | 100.00% |
| person_profile | role | 179 | 98.35% |
| person_profile | employer | 174 | 95.60% |
| person_profile | location | 159 | 87.36% |
| person_profile | record_metadata | natural/location ×144, natural/name ×21 |  |
| product | name | 118 | 98.33% |
| product | condition | 115 | 95.83% |
| product | brand | 114 | 95.00% |
| product | price | 79 | 65.83% |
| product | record_metadata | natural/name ×118, natural/brand ×2 |  |
| real_estate_listing | address | 25 | 96.15% |
| real_estate_listing | price | 25 | 96.15% |
| real_estate_listing | bedrooms | 25 | 96.15% |
| real_estate_listing | property_type | 25 | 96.15% |
| real_estate_listing | area | 24 | 92.31% |
| real_estate_listing | record_metadata | natural/address ×25, natural/price ×1 |  |
| research_paper | year | 39 | 97.50% |
| research_paper | title | 37 | 92.50% |
| research_paper | venue | 33 | 82.50% |
| research_paper | authors | 27 | 67.50% |
| research_paper | topic | 10 | 25.00% |
| research_paper | record_metadata | natural/title ×37, natural/authors ×2, natural/venue ×1 |  |
| software_release | product | 42 | 100.00% |
| software_release | release_date | 40 | 95.24% |
| software_release | version | 37 | 88.10% |
| software_release | platform | 26 | 61.90% |
| software_release | record_metadata | natural/product ×41 |  |
| transaction | amount | 109 | 98.20% |
| transaction | date | 96 | 86.49% |
| transaction | item | 90 | 81.08% |
| transaction | seller | 85 | 76.58% |
| transaction | buyer | 81 | 72.97% |
| transaction | record_metadata | natural/item ×89, natural/amount ×19 |  |
| vehicle_listing | model | 14 | 100.00% |
| vehicle_listing | price | 14 | 100.00% |
| vehicle_listing | make | 12 | 85.71% |
| vehicle_listing | year | 12 | 85.71% |
| vehicle_listing | mileage | 4 | 28.57% |
| vehicle_listing | record_metadata | natural/make ×12, natural/model ×2 |  |

**val.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | condition | 6 | 100.00% |
| clinical_finding | treatment | 5 | 83.33% |
| clinical_finding | outcome | 5 | 83.33% |
| clinical_finding | patient_group | 4 | 66.67% |
| clinical_finding | record_metadata | natural/condition ×6 |  |
| company_profile | name | 34 | 100.00% |
| company_profile | size | 33 | 97.06% |
| company_profile | headquarters | 27 | 79.41% |
| company_profile | industry | 18 | 52.94% |
| company_profile | founded | 10 | 29.41% |
| company_profile | record_metadata | natural/name ×34 |  |
| event_listing | venue | 4 | 100.00% |
| event_listing | name | 3 | 75.00% |
| event_listing | date | 3 | 75.00% |
| event_listing | organizer | 2 | 50.00% |
| event_listing | price | 1 | 25.00% |
| event_listing | record_metadata | natural/name ×3, natural/venue ×1 |  |
| financial_report | organization | 9 | 100.00% |
| financial_report | period | 9 | 100.00% |
| financial_report | revenue | 8 | 88.89% |
| financial_report | net_income | 5 | 55.56% |
| financial_report | record_metadata | natural/organization ×9 |  |
| incident_report | date | 9 | 100.00% |
| incident_report | severity | 9 | 100.00% |
| incident_report | location | 8 | 88.89% |
| incident_report | incident_type | 4 | 44.44% |
| incident_report | casualties | 3 | 33.33% |
| incident_report | record_metadata | natural/location ×8, natural/incident_type ×1 |  |
| job_posting | title | 1 | 100.00% |
| job_posting | company | 1 | 100.00% |
| job_posting | location | 1 | 100.00% |
| job_posting | employment_type | 1 | 100.00% |
| job_posting | record_metadata | natural/location ×1 |  |
| legal_case | court | 6 | 100.00% |
| legal_case | plaintiff | 6 | 100.00% |
| legal_case | defendant | 6 | 100.00% |
| legal_case | status | 6 | 100.00% |
| legal_case | case_name | 3 | 50.00% |
| legal_case | record_metadata | natural/court ×3, natural/case_name ×3 |  |
| person_profile | name | 18 | 100.00% |
| person_profile | role | 17 | 94.44% |
| person_profile | employer | 15 | 83.33% |
| person_profile | location | 12 | 66.67% |
| person_profile | record_metadata | natural/location ×9, natural/name ×5 |  |
| product | condition | 9 | 100.00% |
| product | name | 8 | 88.89% |
| product | brand | 7 | 77.78% |
| product | price | 7 | 77.78% |
| product | record_metadata | natural/name ×8, natural/price ×1 |  |
| research_paper | title | 6 | 100.00% |
| research_paper | year | 6 | 100.00% |
| research_paper | authors | 5 | 83.33% |
| research_paper | venue | 5 | 83.33% |
| research_paper | record_metadata | natural/title ×6 |  |
| software_release | product | 7 | 100.00% |
| software_release | release_date | 7 | 100.00% |
| software_release | platform | 4 | 57.14% |
| software_release | version | 4 | 57.14% |
| software_release | record_metadata | natural/product ×7 |  |
| transaction | amount | 19 | 95.00% |
| transaction | item | 14 | 70.00% |
| transaction | seller | 13 | 65.00% |
| transaction | date | 10 | 50.00% |
| transaction | buyer | 9 | 45.00% |
| transaction | record_metadata | natural/item ×13, natural/amount ×4, natural/buyer ×1 |  |
| vehicle_listing | make | 1 | 100.00% |
| vehicle_listing | model | 1 | 100.00% |
| vehicle_listing | year | 1 | 100.00% |
| vehicle_listing | price | 1 | 100.00% |
| vehicle_listing | record_metadata | natural/make ×1 |  |

**test.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | condition | 4 | 100.00% |
| clinical_finding | patient_group | 4 | 100.00% |
| clinical_finding | treatment | 3 | 75.00% |
| clinical_finding | outcome | 1 | 25.00% |
| clinical_finding | record_metadata | natural/condition ×4 |  |
| company_profile | name | 33 | 100.00% |
| company_profile | size | 32 | 96.97% |
| company_profile | headquarters | 30 | 90.91% |
| company_profile | industry | 22 | 66.67% |
| company_profile | founded | 13 | 39.39% |
| company_profile | record_metadata | natural/name ×30 |  |
| event_listing | name | 6 | 85.71% |
| event_listing | date | 6 | 85.71% |
| event_listing | organizer | 6 | 85.71% |
| event_listing | venue | 4 | 57.14% |
| event_listing | price | 3 | 42.86% |
| event_listing | record_metadata | natural/name ×6, natural/date ×1 |  |
| financial_report | organization | 10 | 100.00% |
| financial_report | revenue | 10 | 100.00% |
| financial_report | period | 9 | 90.00% |
| financial_report | net_income | 5 | 50.00% |
| financial_report | record_metadata | natural/organization ×10 |  |
| incident_report | severity | 15 | 100.00% |
| incident_report | location | 14 | 93.33% |
| incident_report | date | 14 | 93.33% |
| incident_report | incident_type | 12 | 80.00% |
| incident_report | casualties | 9 | 60.00% |
| incident_report | record_metadata | natural/location ×14, natural/casualties ×1 |  |
| job_posting | title | 2 | 100.00% |
| job_posting | company | 2 | 100.00% |
| job_posting | location | 2 | 100.00% |
| job_posting | employment_type | 2 | 100.00% |
| job_posting | record_metadata | natural/location ×1 |  |
| legal_case | court | 1 | 100.00% |
| legal_case | plaintiff | 1 | 100.00% |
| legal_case | defendant | 1 | 100.00% |
| legal_case | status | 1 | 100.00% |
| legal_case | record_metadata | natural/court ×1 |  |
| person_profile | name | 18 | 100.00% |
| person_profile | role | 17 | 94.44% |
| person_profile | location | 17 | 94.44% |
| person_profile | employer | 15 | 83.33% |
| person_profile | record_metadata | natural/location ×14, natural/name ×1 |  |
| product | name | 16 | 100.00% |
| product | brand | 16 | 100.00% |
| product | condition | 16 | 100.00% |
| product | price | 12 | 75.00% |
| product | record_metadata | natural/name ×14 |  |
| real_estate_listing | price | 3 | 100.00% |
| real_estate_listing | bedrooms | 3 | 100.00% |
| real_estate_listing | address | 2 | 66.67% |
| real_estate_listing | area | 2 | 66.67% |
| real_estate_listing | property_type | 2 | 66.67% |
| real_estate_listing | record_metadata | natural/address ×2, natural/price ×1 |  |
| research_paper | title | 7 | 100.00% |
| research_paper | venue | 6 | 85.71% |
| research_paper | year | 6 | 85.71% |
| research_paper | authors | 4 | 57.14% |
| research_paper | topic | 1 | 14.29% |
| research_paper | record_metadata | natural/title ×7 |  |
| software_release | product | 4 | 100.00% |
| software_release | version | 4 | 100.00% |
| software_release | release_date | 4 | 100.00% |
| software_release | platform | 4 | 100.00% |
| software_release | record_metadata | natural/product ×4 |  |
| transaction | item | 12 | 100.00% |
| transaction | amount | 11 | 91.67% |
| transaction | seller | 11 | 91.67% |
| transaction | buyer | 9 | 75.00% |
| transaction | date | 9 | 75.00% |
| transaction | record_metadata | natural/item ×12 |  |
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
