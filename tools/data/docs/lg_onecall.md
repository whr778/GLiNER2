# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lg_onecall | train | 389 | 389 | 9,351 | 717 | 1,333 | 133 | 208 | 635 |
| lg_onecall | val | 55 | 55 | 1,335 | 115 | 197 | 15 | 25 | 76 |
| lg_onecall | test | 55 | 55 | 1,352 | 82 | 194 | 15 | 37 | 121 |

## lg_onecall

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 389 | 389 | 264 | 304 | 382 | 9,351 | 717 | 1,333 | 133 | 208 | 635 |
| val | 55 | 55 | 266 | 303 | 334 | 1,335 | 115 | 197 | 15 | 25 | 76 |
| test | 55 | 55 | 265 | 310 | 333 | 1,352 | 82 | 194 | 15 | 37 | 121 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 389 | 55 | 55 | 499 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 55 |
| test_unique_inputs | 55 |
| train_records | 389 |
| train_unique_inputs | 389 |
| val_records | 55 |
| val_unique_inputs | 55 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0846 | 9,351 / 1,335 | money 6%→7%, organization 5%→3%, government agency 2%→3%, measurement 2%→1%, duration 4%→3% |
| entities (per mention) | train / test | 0.1153 | 9,351 / 1,352 | date 13%→12%, person 8%→7%, cardinal 11%→12%, percent 4%→3%, job title 4%→3% |
| entities (per mention) | val / test | 0.1574 WARN | 1,335 / 1,352 | person 9%→7%, date 14%→12%, measurement 1%→3%, government agency 3%→1%, money 7%→6% |
| relations (per instance) | train / val | 0.3649 STRONG | 717 / 115 | located_in 10%→26%, employer_of 10%→1%, member_of 5%→1%, competitor_of 1%→5%, operates_in 8%→4% |
| relations (per instance) | train / test | 0.3091 STRONG | 717 / 82 | operates_in 8%→2%, adjacent_to 0%→5%, manufactured_by 1%→5%, located_in 10%→6%, based_in 9%→12% |
| relations (per instance) | val / test | 0.3595 STRONG | 115 / 82 | located_in 26%→6%, employer_of 1%→9%, based_in 7%→12%, adjacent_to 0%→5%, manufactured_by 2%→5% |
| event types (per trigger mention) | train / val | 0.4650 STRONG | 208 / 25 | Transaction.TransferMoney 2%→12%, Transaction.TransferOwnership 2%→8%, Business.StartOrg 6%→0%, Tech.Release 4%→8%, Transaction.Lend 0%→4% |
| event types (per trigger mention) | train / test | 0.4869 STRONG | 208 / 37 | Conflict.Attack 1%→11%, Contact.Meet 3%→11%, Disaster.Rescue 2%→8%, Personnel.StartPosition 8%→3%, Justice.Fine 0%→5% |
| event types (per trigger mention) | val / test | 0.5968 STRONG | 25 / 37 | Transaction.TransferMoney 12%→0%, Conflict.Attack 0%→11%, Contact.Meet 4%→11%, Personnel.EndPosition 0%→5%, Transaction.TransferOwnership 8%→3% |
| event roles (per argument) | train / val | 0.2843 WARN | 635 / 76 | Place 12%→7%, Money 3%→8%, Time 25%→28%, Product 3%→1%, Person 5%→3% |
| event roles (per argument) | train / test | 0.2001 WARN | 635 / 121 | Recipient 4%→1%, Time 25%→27%, Rescuer 1%→2%, Target 1%→2%, Communicator 2%→0% |
| event roles (per argument) | val / test | 0.3187 STRONG | 76 / 121 | Place 7%→13%, Money 8%→2%, Recipient 5%→1%, Product 1%→4%, Communicator 3%→0% |
| classifications (true_label) | train / val | 0.2374 WARN | 1,333 / 197 | certainty: asserted 8%→5%, subjectivity: objective 7%→5%, temporal_orientation: past 5%→8%, document_genre: announcement 2%→4%, sentiment: negative 2%→4% |
| classifications (true_label) | train / test | 0.1535 WARN | 1,333 / 194 | actionability: informational 4%→6%, urgency: routine 5%→3%, temporal_orientation: past 5%→7%, document_genre: news report 2%→1%, topic: business 3%→5% |
| classifications (true_label) | val / test | 0.2137 WARN | 197 / 194 | document_genre: announcement 4%→1%, subjectivity: objective 5%→7%, actionability: informational 4%→6%, language_register: legal 3%→0%, document_genre: news report 3%→1% |

Thin support in val, entities (per mention): 61 label(s) under 10 gold. honorific ×1, stock ticker ×1, dosage ×1, security vulnerability ×1, work of art ×1, contract ×1, era ×1, criminal organization ×1, interest rate ×1, anatomical structure ×1, brand ×1, political affiliation ×1, hardware component ×1, operating system ×1, protocol ×1, mountain ×1, clinical trial ×1, witness ×2, frequency ×2, award ×2, weight ×2, regulation ×2, ordinal ×2, currency ×2, conference ×2, protein ×2, organism ×2, airport ×2, sports competition ×2, non-profit organization ×3, military rank ×3, publication ×3, natural disaster ×3, symptom ×3, court case ×4, law ×4, postal code ×4, speed ×4, military unit ×4, media outlet ×4, nationality ×5, crime ×5, medication ×5, software ×5, chemical ×5, body of water ×6, temperature ×6, model number ×6, deadline ×6, age ×7, academic degree ×7, landmark ×7, sports team ×7, measurement ×8, road ×8, fiscal quarter ×8, quantity ×8, disease ×9, area ×9, product ×9, regulatory body ×9.
Unseen in train, present in val, relations (per instance): predecessor_of ×1, parent_of ×1, treats ×1.
Thin support in val, relations (per instance): 27 label(s) under 10 gold. owns ×1, acquired ×1, predecessor_of ×1, sold_to ×1, parent_of ×1, treats ×1, sponsored_by ×1, member_of ×1, merged_with ×1, employer_of ×1, developed_by ×1, born_in ×1, regulated_by ×1, invested_in ×2, board_member_of ×2, supplier_of ×2, manufactured_by ×2, charged_with ×2, colleague_of ×3, leader_of ×3, part_of ×4, operates_in ×5, reports_to ×5, headquartered_in ×5, competitor_of ×6, affiliated_with ×7, based_in ×8.
Thin support in val, event types (per trigger mention): 18 label(s) under 10 gold. Research.Fund ×1, Disaster.Outbreak ×1, Disaster.Rescue ×1, Medical.Treat ×1, Life.Evacuate ×1, Contact.Meet ×1, Business.Layoff ×1, Contact.Negotiate ×1, Transaction.Lend ×1, Business.LaunchProduct ×1, Justice.Sentence ×1, Justice.Fine ×1, Contact.Communicate ×2, Research.Publish ×2, Tech.Release ×2, Transaction.TransferOwnership ×2, Personnel.StartPosition ×2, Transaction.TransferMoney ×3.
Thin support in val, event roles (per argument): 29 label(s) under 10 gold. Funder ×1, Rescuer ×1, Treatment ×1, Patient ×1, Author ×1, Product ×1, Evacuee ×1, Agent ×1, Seller ×1, Artifact ×1, Quantity ×1, Lender ×1, Borrower ×1, Defendant ×1, Crime ×1, Entity ×1, Adjudicator ×1, Giver ×2, Communicator ×2, Publication ×2, Buyer ×2, Price ×2, Person ×2, Position ×2, Participant ×3, Recipient ×4, Place ×5, Money ×6, Organization ×6.
Thin support in val, classifications (true_label): 40 label(s) under 10 gold. document_genre: legal filing ×1, document_genre: opinion ×1, topic: education ×1, topic: environment ×1, temporal_orientation: present ×1, formality: informal ×1, certainty: hedged ×1, risk_level: none ×1, risk_level: moderate ×1, urgency: critical ×1, document_genre: correspondence ×2, document_genre: technical documentation ×2, subjectivity: subjective ×2, topic: politics ×2, sentiment: neutral ×2, temporal_orientation: future ×2, risk_level: low ×2, risk_level: high ×2, urgency: elevated ×2, topic: science ×3, topic: technology ×3, sentiment: positive ×3, audience: internal ×3, language_register: plain ×3, actionability: advisory ×4, actionability: requires action ×4, topic: legal ×5, topic: finance ×5, language_register: legal ×5, document_genre: news report ×6, topic: health ×6, audience: regulatory ×6, sentiment: negative ×7, actionability: informational ×7, audience: general public ×7, urgency: routine ×7, document_genre: announcement ×8, audience: expert ×8, subjectivity: objective ×9, topic: business ×9.
Thin support in test, entities (per mention): 55 label(s) under 10 gold. conference ×1, vaccine ×1, vessel ×1, holiday ×1, witness ×1, currency ×1, algorithm ×1, political affiliation ×1, non-profit organization ×1, anatomical structure ×1, honorific ×1, brand ×1, postal code ×1, hardware component ×1, work of art ×1, military unit ×2, pathogen ×2, military rank ×2, era ×2, voltage ×2, mountain ×2, publication ×2, award ×2, clinical trial ×2, deadline ×2, protein ×2, event ×3, weight ×3, ethnic group ×3, airport ×3, law ×3, court case ×3, body of water ×3, symptom ×3, sports competition ×3, crime ×4, regulation ×4, dosage ×4, fiscal quarter ×4, nationality ×5, medication ×5, area ×5, operating system ×5, model number ×5, frequency ×5, sports team ×5, age ×5, media outlet ×6, software ×6, address ×7, temperature ×8, occupation ×8, regulatory body ×9, road ×9, ordinal ×9.
Unseen in train, present in test, relations (per instance): treats ×1.
Thin support in test, relations (per instance): 23 label(s) under 10 gold. developed_by ×1, produced_by ×1, colleague_of ×1, represents ×1, sponsored_by ×1, sued_by ×1, charged_with ×1, treats ×1, leader_of ×1, member_of ×2, regulated_by ×2, operates_in ×2, owns ×2, supplier_of ×3, competitor_of ×3, part_of ×3, manufactured_by ×4, adjacent_to ×4, reports_to ×4, located_in ×5, headquartered_in ×5, affiliated_with ×6, employer_of ×7.
Unseen in train, present in test, event types (per trigger mention): Business.EndOrg ×1.
Thin support in test, event types (per trigger mention): 24 label(s) under 10 gold. Research.Publish ×1, Personnel.Nominate ×1, Business.MergeOrg ×1, Business.EndOrg ×1, Government.Sanction ×1, Business.Layoff ×1, Personnel.StartPosition ×1, Contact.Communicate ×1, Government.EnactLaw ×1, Research.Discover ×1, Transaction.TransferOwnership ×1, Justice.Sentence ×1, Business.StartOrg ×1, Sport.Compete ×1, Justice.ChargeIndict ×1, Research.Fund ×1, Medical.Diagnose ×1, Personnel.EndPosition ×2, Justice.Fine ×2, Tech.Release ×2, Business.LaunchProduct ×3, Disaster.Rescue ×3, Conflict.Attack ×4, Contact.Meet ×4.
Thin support in test, event roles (per argument): 31 label(s) under 10 gold. Author ×1, Publication ×1, Entity ×1, Reason ×1, Quantity ×1, Law ×1, Researcher ×1, Finding ×1, Buyer ×1, Seller ×1, Price ×1, Sentence ×1, Victim ×1, Result ×1, Prosecutor ×1, Funder ×1, Recipient ×1, Diagnosis ×1, Attacker ×2, Defendant ×2, Competitor ×2, Money ×3, Adjudicator ×3, Crime ×3, Target ×3, Rescuer ×3, Person ×4, Position ×4, Agent ×4, Participant ×4, Product ×5.
Thin support in test, classifications (true_label): 37 label(s) under 10 gold. temporal_orientation: timeless ×1, temporal_orientation: present ×1, temporal_orientation: future ×1, topic: world ×1, topic: politics ×1, topic: sports ×1, document_genre: correspondence ×1, document_genre: advertisement ×1, document_genre: review ×1, document_genre: legal filing ×1, document_genre: news report ×1, risk_level: moderate ×1, subjectivity: subjective ×1, document_genre: announcement ×2, risk_level: none ×2, sentiment: neutral ×2, formality: informal ×2, urgency: elevated ×2, topic: health ×3, topic: science ×3, topic: environment ×3, topic: legal ×3, document_genre: technical documentation ×3, audience: internal ×3, language_register: plain ×4, actionability: requires action ×4, risk_level: high ×4, topic: technology ×5, topic: finance ×5, risk_level: low ×5, sentiment: negative ×5, urgency: routine ×5, sentiment: positive ×7, audience: regulatory ×7, topic: business ×9, audience: general public ×9, audience: expert ×9.

### Events

**train.** 208 instances in 152 records, 0 triggerless. Trigger gold 208. Argument gold 635. Arguments per instance, mean 3.05. Trigger length in whitespace tokens, mean 1.22, single-token share 84.6%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Personnel.StartPosition | 17 | 8.17% |
| Business.LaunchProduct | 14 | 6.73% |
| Business.StartOrg | 12 | 5.77% |
| Research.Fund | 11 | 5.29% |
| Contact.Communicate | 11 | 5.29% |
| Research.Publish | 10 | 4.81% |
| Tech.Release | 9 | 4.33% |
| Life.Evacuate | 8 | 3.85% |
| Personnel.EndPosition | 7 | 3.37% |
| Disaster.Outbreak | 7 | 3.37% |
| Research.Discover | 6 | 2.88% |
| Contact.Meet | 6 | 2.88% |
| Award.ReceiveAward | 5 | 2.40% |
| Disaster.Accident | 5 | 2.40% |
| Medical.Diagnose | 5 | 2.40% |
| Transaction.TransferMoney | 5 | 2.40% |
| Life.Injure | 5 | 2.40% |
| Disaster.Rescue | 4 | 1.92% |
| Business.Layoff | 4 | 1.92% |
| Personnel.Nominate | 4 | 1.92% |
| Medical.Treat | 4 | 1.92% |
| Transaction.TransferOwnership | 4 | 1.92% |
| Justice.TrialHearing | 4 | 1.92% |
| Justice.Sue | 3 | 1.44% |
| Life.Die | 3 | 1.44% |
| Sport.Compete | 3 | 1.44% |
| Business.MergeOrg | 3 | 1.44% |
| Movement.Arrive | 3 | 1.44% |
| Life.BeBorn | 3 | 1.44% |
| Conflict.Attack | 3 | 1.44% |
| Conflict.Ceasefire | 2 | 0.96% |
| Justice.Arrest | 2 | 0.96% |
| Tech.Breach | 2 | 0.96% |
| Disaster.NaturalDisaster | 2 | 0.96% |
| Justice.Sentence | 2 | 0.96% |
| Justice.ChargeIndict | 2 | 0.96% |
| Contact.Negotiate | 2 | 0.96% |
| Movement.Transport | 1 | 0.48% |
| Government.EnactLaw | 1 | 0.48% |
| Transaction.Donate | 1 | 0.48% |
| Transaction.Lend | 1 | 0.48% |
| Government.Sanction | 1 | 0.48% |
| Justice.Fine | 1 | 0.48% |

**val.** 25 instances in 22 records, 0 triggerless. Trigger gold 25. Argument gold 76. Arguments per instance, mean 3.04. Trigger length in whitespace tokens, mean 1.24, single-token share 84.0%.

#### val event types

| label | count | share |
| --- | --- | --- |
| Transaction.TransferMoney | 3 | 12.00% |
| Contact.Communicate | 2 | 8.00% |
| Research.Publish | 2 | 8.00% |
| Tech.Release | 2 | 8.00% |
| Transaction.TransferOwnership | 2 | 8.00% |
| Personnel.StartPosition | 2 | 8.00% |
| Research.Fund | 1 | 4.00% |
| Disaster.Outbreak | 1 | 4.00% |
| Disaster.Rescue | 1 | 4.00% |
| Medical.Treat | 1 | 4.00% |
| Life.Evacuate | 1 | 4.00% |
| Contact.Meet | 1 | 4.00% |
| Business.Layoff | 1 | 4.00% |
| Contact.Negotiate | 1 | 4.00% |
| Transaction.Lend | 1 | 4.00% |
| Business.LaunchProduct | 1 | 4.00% |
| Justice.Sentence | 1 | 4.00% |
| Justice.Fine | 1 | 4.00% |

**test.** 37 instances in 31 records, 0 triggerless. Trigger gold 37. Argument gold 121. Arguments per instance, mean 3.27. Trigger length in whitespace tokens, mean 1.24, single-token share 86.5%.

#### test event types

| label | count | share |
| --- | --- | --- |
| Conflict.Attack | 4 | 10.81% |
| Contact.Meet | 4 | 10.81% |
| Business.LaunchProduct | 3 | 8.11% |
| Disaster.Rescue | 3 | 8.11% |
| Personnel.EndPosition | 2 | 5.41% |
| Justice.Fine | 2 | 5.41% |
| Tech.Release | 2 | 5.41% |
| Research.Publish | 1 | 2.70% |
| Personnel.Nominate | 1 | 2.70% |
| Business.MergeOrg | 1 | 2.70% |
| Business.EndOrg | 1 | 2.70% |
| Government.Sanction | 1 | 2.70% |
| Business.Layoff | 1 | 2.70% |
| Personnel.StartPosition | 1 | 2.70% |
| Contact.Communicate | 1 | 2.70% |
| Government.EnactLaw | 1 | 2.70% |
| Research.Discover | 1 | 2.70% |
| Transaction.TransferOwnership | 1 | 2.70% |
| Justice.Sentence | 1 | 2.70% |
| Business.StartOrg | 1 | 2.70% |
| Sport.Compete | 1 | 2.70% |
| Justice.ChargeIndict | 1 | 2.70% |
| Research.Fund | 1 | 2.70% |
| Medical.Diagnose | 1 | 2.70% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 635 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 76 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 121 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.30 | 18.84% | 1.53 | 29.00% | 0.59% | 0 | 635 | 0 | 208 |
| val | 1.20 | 16.00% | 1.70 | 31.58% | 0.00% | 0 | 76 | 0 | 25 |
| test | 1.14 | 10.81% | 1.72 | 29.75% | 0.00% | 0 | 121 | 0 | 37 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 208 | 0.00% | 8 | 62.50% | 62.50% | 0.00% | 0.00% | 1.97% |
| val | 25 | 0.00% | 0 | n/a | n/a | n/a | n/a | 0.00% |
| test | 37 | 0.00% | 1 | 100.00% | 100.00% | 0.00% | 0.00% | 3.23% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 0 | 0 | 8 |
| val | 0 | 0 | 0 |
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
| date | 1,260 | 13.47% |
| location | 1,005 | 10.75% |
| cardinal | 1,005 | 10.75% |
| person | 768 | 8.21% |
| money | 520 | 5.56% |
| organization | 423 | 4.52% |
| percent | 402 | 4.30% |
| company | 386 | 4.13% |
| job title | 378 | 4.04% |
| duration | 331 | 3.54% |
| time | 195 | 2.09% |
| measurement | 160 | 1.71% |
| geopolitical entity | 156 | 1.67% |
| government agency | 155 | 1.66% |
| educational institution | 147 | 1.57% |
| product | 121 | 1.29% |
| region | 119 | 1.27% |
| facility | 117 | 1.25% |
| distance | 107 | 1.14% |
| academic degree | 79 | 0.84% |
| address | 75 | 0.80% |
| disease | 70 | 0.75% |
| quantity | 64 | 0.68% |
| software | 57 | 0.61% |
| occupation | 57 | 0.61% |
| area | 55 | 0.59% |
| landmark | 55 | 0.59% |
| fiscal quarter | 54 | 0.58% |
| sports team | 54 | 0.58% |
| age | 53 | 0.57% |
| regulatory body | 49 | 0.52% |
| medication | 43 | 0.46% |
| temperature | 39 | 0.42% |
| ordinal | 36 | 0.38% |
| nationality | 34 | 0.36% |
| dosage | 34 | 0.36% |
| chemical | 31 | 0.33% |
| frequency | 30 | 0.32% |
| publication | 29 | 0.31% |
| road | 28 | 0.30% |
| postal code | 26 | 0.28% |
| symptom | 25 | 0.27% |
| media outlet | 25 | 0.27% |
| award | 24 | 0.26% |
| speed | 24 | 0.26% |
| model number | 24 | 0.26% |
| non-profit organization | 22 | 0.24% |
| military unit | 20 | 0.21% |
| clinical trial | 20 | 0.21% |
| law | 19 | 0.20% |
| brand | 19 | 0.20% |
| military rank | 19 | 0.20% |
| regulation | 18 | 0.19% |
| conference | 18 | 0.19% |
| weight | 18 | 0.19% |
| court case | 16 | 0.17% |
| body of water | 15 | 0.16% |
| sports competition | 14 | 0.15% |
| deadline | 13 | 0.14% |
| era | 11 | 0.12% |
| pathogen | 11 | 0.12% |
| airport | 10 | 0.11% |
| honorific | 9 | 0.10% |
| protein | 9 | 0.10% |
| crime | 9 | 0.10% |
| patent | 9 | 0.10% |
| operating system | 8 | 0.09% |
| protocol | 6 | 0.06% |
| medical procedure | 6 | 0.06% |
| stock ticker | 5 | 0.05% |
| event | 5 | 0.05% |
| mountain | 5 | 0.05% |
| natural disaster | 5 | 0.05% |
| organism | 5 | 0.05% |
| work of art | 5 | 0.05% |
| treaty | 4 | 0.04% |
| team role | 4 | 0.04% |
| family relationship | 4 | 0.04% |
| hardware component | 4 | 0.04% |
| witness | 4 | 0.04% |
| security vulnerability | 4 | 0.04% |
| license | 4 | 0.04% |
| voltage | 4 | 0.04% |
| criminal organization | 3 | 0.03% |
| currency | 3 | 0.03% |
| anatomical structure | 3 | 0.03% |
| language | 2 | 0.02% |
| pseudonym | 2 | 0.02% |
| book | 2 | 0.02% |
| ethnic group | 2 | 0.02% |
| religious group | 2 | 0.02% |
| contract | 2 | 0.02% |
| border | 2 | 0.02% |
| political affiliation | 2 | 0.02% |
| weapon | 2 | 0.02% |
| framework | 2 | 0.02% |
| aircraft | 2 | 0.02% |
| gene | 2 | 0.02% |
| programming language | 1 | 0.01% |
| vaccine | 1 | 0.01% |
| sentence | 1 | 0.01% |
| database | 1 | 0.01% |
| api | 1 | 0.01% |
| political party | 1 | 0.01% |
| algorithm | 1 | 0.01% |
| market index | 1 | 0.01% |
| vehicle | 0 | 0.00% |
| interest rate | 0 | 0.00% |
| holiday | 0 | 0.00% |
| allergen | 0 | 0.00% |
| film | 0 | 0.00% |
| species | 0 | 0.00% |
| vessel | 0 | 0.00% |
| trade union | 0 | 0.00% |
| mineral | 0 | 0.00% |
| song | 0 | 0.00% |
| verdict | 0 | 0.00% |
| tax | 0 | 0.00% |
| war | 0 | 0.00% |
| budget line | 0 | 0.00% |
| financial instrument | 0 | 0.00% |
| diagnosis | 0 | 0.00% |
| video game | 0 | 0.00% |
| credit rating | 0 | 0.00% |
| file format | 0 | 0.00% |

#### val

| label | count | share |
| --- | --- | --- |
| date | 186 | 13.93% |
| location | 150 | 11.24% |
| cardinal | 147 | 11.01% |
| person | 119 | 8.91% |
| money | 95 | 7.12% |
| company | 56 | 4.19% |
| percent | 56 | 4.19% |
| job title | 55 | 4.12% |
| organization | 42 | 3.15% |
| government agency | 39 | 2.92% |
| duration | 35 | 2.62% |
| geopolitical entity | 23 | 1.72% |
| time | 21 | 1.57% |
| educational institution | 21 | 1.57% |
| facility | 20 | 1.50% |
| region | 13 | 0.97% |
| distance | 12 | 0.90% |
| address | 10 | 0.75% |
| occupation | 10 | 0.75% |
| disease | 9 | 0.67% |
| area | 9 | 0.67% |
| product | 9 | 0.67% |
| regulatory body | 9 | 0.67% |
| measurement | 8 | 0.60% |
| road | 8 | 0.60% |
| fiscal quarter | 8 | 0.60% |
| quantity | 8 | 0.60% |
| age | 7 | 0.52% |
| academic degree | 7 | 0.52% |
| landmark | 7 | 0.52% |
| sports team | 7 | 0.52% |
| body of water | 6 | 0.45% |
| temperature | 6 | 0.45% |
| model number | 6 | 0.45% |
| deadline | 6 | 0.45% |
| nationality | 5 | 0.37% |
| crime | 5 | 0.37% |
| medication | 5 | 0.37% |
| software | 5 | 0.37% |
| chemical | 5 | 0.37% |
| court case | 4 | 0.30% |
| law | 4 | 0.30% |
| postal code | 4 | 0.30% |
| speed | 4 | 0.30% |
| military unit | 4 | 0.30% |
| media outlet | 4 | 0.30% |
| non-profit organization | 3 | 0.22% |
| military rank | 3 | 0.22% |
| publication | 3 | 0.22% |
| natural disaster | 3 | 0.22% |
| symptom | 3 | 0.22% |
| witness | 2 | 0.15% |
| frequency | 2 | 0.15% |
| award | 2 | 0.15% |
| weight | 2 | 0.15% |
| regulation | 2 | 0.15% |
| ordinal | 2 | 0.15% |
| currency | 2 | 0.15% |
| conference | 2 | 0.15% |
| protein | 2 | 0.15% |
| organism | 2 | 0.15% |
| airport | 2 | 0.15% |
| sports competition | 2 | 0.15% |
| honorific | 1 | 0.07% |
| stock ticker | 1 | 0.07% |
| dosage | 1 | 0.07% |
| security vulnerability | 1 | 0.07% |
| work of art | 1 | 0.07% |
| contract | 1 | 0.07% |
| era | 1 | 0.07% |
| criminal organization | 1 | 0.07% |
| interest rate | 1 | 0.07% |
| anatomical structure | 1 | 0.07% |
| brand | 1 | 0.07% |
| political affiliation | 1 | 0.07% |
| hardware component | 1 | 0.07% |
| operating system | 1 | 0.07% |
| protocol | 1 | 0.07% |
| mountain | 1 | 0.07% |
| clinical trial | 1 | 0.07% |
| trade union | 0 | 0.00% |
| political party | 0 | 0.00% |
| event | 0 | 0.00% |
| voltage | 0 | 0.00% |
| tax | 0 | 0.00% |
| patent | 0 | 0.00% |
| verdict | 0 | 0.00% |
| sentence | 0 | 0.00% |
| api | 0 | 0.00% |
| holiday | 0 | 0.00% |
| video game | 0 | 0.00% |
| algorithm | 0 | 0.00% |
| vehicle | 0 | 0.00% |
| allergen | 0 | 0.00% |
| vessel | 0 | 0.00% |
| war | 0 | 0.00% |
| treaty | 0 | 0.00% |
| budget line | 0 | 0.00% |
| religious group | 0 | 0.00% |
| book | 0 | 0.00% |
| medical procedure | 0 | 0.00% |
| aircraft | 0 | 0.00% |
| mineral | 0 | 0.00% |
| file format | 0 | 0.00% |
| pseudonym | 0 | 0.00% |
| ethnic group | 0 | 0.00% |
| species | 0 | 0.00% |
| database | 0 | 0.00% |
| gene | 0 | 0.00% |
| weapon | 0 | 0.00% |
| family relationship | 0 | 0.00% |
| financial instrument | 0 | 0.00% |
| diagnosis | 0 | 0.00% |
| border | 0 | 0.00% |
| credit rating | 0 | 0.00% |
| song | 0 | 0.00% |
| programming language | 0 | 0.00% |
| film | 0 | 0.00% |
| vaccine | 0 | 0.00% |
| pathogen | 0 | 0.00% |
| license | 0 | 0.00% |
| language | 0 | 0.00% |
| market index | 0 | 0.00% |
| team role | 0 | 0.00% |
| framework | 0 | 0.00% |

#### test

| label | count | share |
| --- | --- | --- |
| cardinal | 167 | 12.35% |
| date | 159 | 11.76% |
| location | 146 | 10.80% |
| person | 89 | 6.58% |
| money | 76 | 5.62% |
| organization | 55 | 4.07% |
| company | 52 | 3.85% |
| duration | 50 | 3.70% |
| percent | 43 | 3.18% |
| job title | 40 | 2.96% |
| measurement | 34 | 2.51% |
| facility | 31 | 2.29% |
| geopolitical entity | 30 | 2.22% |
| time | 29 | 2.14% |
| product | 27 | 2.00% |
| distance | 27 | 2.00% |
| region | 26 | 1.92% |
| government agency | 16 | 1.18% |
| landmark | 15 | 1.11% |
| speed | 13 | 0.96% |
| quantity | 13 | 0.96% |
| educational institution | 11 | 0.81% |
| disease | 11 | 0.81% |
| academic degree | 10 | 0.74% |
| regulatory body | 9 | 0.67% |
| road | 9 | 0.67% |
| ordinal | 9 | 0.67% |
| temperature | 8 | 0.59% |
| occupation | 8 | 0.59% |
| address | 7 | 0.52% |
| media outlet | 6 | 0.44% |
| software | 6 | 0.44% |
| nationality | 5 | 0.37% |
| medication | 5 | 0.37% |
| area | 5 | 0.37% |
| operating system | 5 | 0.37% |
| model number | 5 | 0.37% |
| frequency | 5 | 0.37% |
| sports team | 5 | 0.37% |
| age | 5 | 0.37% |
| crime | 4 | 0.30% |
| regulation | 4 | 0.30% |
| dosage | 4 | 0.30% |
| fiscal quarter | 4 | 0.30% |
| event | 3 | 0.22% |
| weight | 3 | 0.22% |
| ethnic group | 3 | 0.22% |
| airport | 3 | 0.22% |
| law | 3 | 0.22% |
| court case | 3 | 0.22% |
| body of water | 3 | 0.22% |
| symptom | 3 | 0.22% |
| sports competition | 3 | 0.22% |
| military unit | 2 | 0.15% |
| pathogen | 2 | 0.15% |
| military rank | 2 | 0.15% |
| era | 2 | 0.15% |
| voltage | 2 | 0.15% |
| mountain | 2 | 0.15% |
| publication | 2 | 0.15% |
| award | 2 | 0.15% |
| clinical trial | 2 | 0.15% |
| deadline | 2 | 0.15% |
| protein | 2 | 0.15% |
| conference | 1 | 0.07% |
| vaccine | 1 | 0.07% |
| vessel | 1 | 0.07% |
| holiday | 1 | 0.07% |
| witness | 1 | 0.07% |
| currency | 1 | 0.07% |
| algorithm | 1 | 0.07% |
| political affiliation | 1 | 0.07% |
| non-profit organization | 1 | 0.07% |
| anatomical structure | 1 | 0.07% |
| honorific | 1 | 0.07% |
| brand | 1 | 0.07% |
| postal code | 1 | 0.07% |
| hardware component | 1 | 0.07% |
| work of art | 1 | 0.07% |
| api | 0 | 0.00% |
| tax | 0 | 0.00% |
| interest rate | 0 | 0.00% |
| financial instrument | 0 | 0.00% |
| security vulnerability | 0 | 0.00% |
| aircraft | 0 | 0.00% |
| sentence | 0 | 0.00% |
| allergen | 0 | 0.00% |
| patent | 0 | 0.00% |
| mineral | 0 | 0.00% |
| team role | 0 | 0.00% |
| weapon | 0 | 0.00% |
| trade union | 0 | 0.00% |
| species | 0 | 0.00% |
| diagnosis | 0 | 0.00% |
| stock ticker | 0 | 0.00% |
| framework | 0 | 0.00% |
| vehicle | 0 | 0.00% |
| pseudonym | 0 | 0.00% |
| film | 0 | 0.00% |
| criminal organization | 0 | 0.00% |
| war | 0 | 0.00% |
| political party | 0 | 0.00% |
| file format | 0 | 0.00% |
| natural disaster | 0 | 0.00% |
| religious group | 0 | 0.00% |
| book | 0 | 0.00% |
| organism | 0 | 0.00% |
| video game | 0 | 0.00% |
| song | 0 | 0.00% |
| budget line | 0 | 0.00% |
| family relationship | 0 | 0.00% |
| language | 0 | 0.00% |
| gene | 0 | 0.00% |
| medical procedure | 0 | 0.00% |
| contract | 0 | 0.00% |
| chemical | 0 | 0.00% |
| border | 0 | 0.00% |
| programming language | 0 | 0.00% |
| license | 0 | 0.00% |
| credit rating | 0 | 0.00% |
| verdict | 0 | 0.00% |
| database | 0 | 0.00% |
| market index | 0 | 0.00% |
| treaty | 0 | 0.00% |
| protocol | 0 | 0.00% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| works_for | 79 | 11.02% |
| located_in | 71 | 9.90% |
| employer_of | 70 | 9.76% |
| based_in | 65 | 9.07% |
| affiliated_with | 59 | 8.23% |
| operates_in | 58 | 8.09% |
| member_of | 37 | 5.16% |
| headquartered_in | 31 | 4.32% |
| reports_to | 25 | 3.49% |
| leader_of | 24 | 3.35% |
| regulated_by | 17 | 2.37% |
| colleague_of | 16 | 2.23% |
| part_of | 15 | 2.09% |
| developed_by | 12 | 1.67% |
| board_member_of | 11 | 1.53% |
| founded | 9 | 1.26% |
| produced_by | 9 | 1.26% |
| published_by | 8 | 1.12% |
| competitor_of | 8 | 1.12% |
| manufactured_by | 7 | 0.98% |
| citizen_of | 7 | 0.98% |
| subsidiary_of | 6 | 0.84% |
| supplier_of | 6 | 0.84% |
| party_to | 5 | 0.70% |
| spouse_of | 4 | 0.56% |
| sponsored_by | 4 | 0.56% |
| opposed_to | 3 | 0.42% |
| advisor_to | 3 | 0.42% |
| mentor_of | 3 | 0.42% |
| owns | 3 | 0.42% |
| compatible_with | 3 | 0.42% |
| charged_with | 3 | 0.42% |
| customer_of | 3 | 0.42% |
| invested_in | 2 | 0.28% |
| implements | 2 | 0.28% |
| authored_by | 2 | 0.28% |
| designed_by | 2 | 0.28% |
| causes | 2 | 0.28% |
| represents | 2 | 0.28% |
| sued_by | 2 | 0.28% |
| merged_with | 2 | 0.28% |
| sold_to | 2 | 0.28% |
| supersedes | 1 | 0.14% |
| relative_of | 1 | 0.14% |
| derived_from | 1 | 0.14% |
| won | 1 | 0.14% |
| testified_against | 1 | 0.14% |
| distributor_of | 1 | 0.14% |
| depends_on | 1 | 0.14% |
| born_in | 1 | 0.14% |
| died_in | 1 | 0.14% |
| participated_in | 1 | 0.14% |
| adjacent_to | 1 | 0.14% |
| acquired | 1 | 0.14% |
| interacts_with | 1 | 0.14% |
| allied_with | 1 | 0.14% |
| successor_of | 1 | 0.14% |

#### val

| label | count | share |
| --- | --- | --- |
| located_in | 30 | 26.09% |
| works_for | 16 | 13.91% |
| based_in | 8 | 6.96% |
| affiliated_with | 7 | 6.09% |
| competitor_of | 6 | 5.22% |
| operates_in | 5 | 4.35% |
| reports_to | 5 | 4.35% |
| headquartered_in | 5 | 4.35% |
| part_of | 4 | 3.48% |
| colleague_of | 3 | 2.61% |
| leader_of | 3 | 2.61% |
| invested_in | 2 | 1.74% |
| board_member_of | 2 | 1.74% |
| supplier_of | 2 | 1.74% |
| manufactured_by | 2 | 1.74% |
| charged_with | 2 | 1.74% |
| owns | 1 | 0.87% |
| acquired | 1 | 0.87% |
| predecessor_of | 1 | 0.87% |
| sold_to | 1 | 0.87% |
| parent_of | 1 | 0.87% |
| treats | 1 | 0.87% |
| sponsored_by | 1 | 0.87% |
| member_of | 1 | 0.87% |
| merged_with | 1 | 0.87% |
| employer_of | 1 | 0.87% |
| developed_by | 1 | 0.87% |
| born_in | 1 | 0.87% |
| regulated_by | 1 | 0.87% |

#### test

| label | count | share |
| --- | --- | --- |
| works_for | 11 | 13.41% |
| based_in | 10 | 12.20% |
| employer_of | 7 | 8.54% |
| affiliated_with | 6 | 7.32% |
| located_in | 5 | 6.10% |
| headquartered_in | 5 | 6.10% |
| manufactured_by | 4 | 4.88% |
| adjacent_to | 4 | 4.88% |
| reports_to | 4 | 4.88% |
| supplier_of | 3 | 3.66% |
| competitor_of | 3 | 3.66% |
| part_of | 3 | 3.66% |
| member_of | 2 | 2.44% |
| regulated_by | 2 | 2.44% |
| operates_in | 2 | 2.44% |
| owns | 2 | 2.44% |
| developed_by | 1 | 1.22% |
| produced_by | 1 | 1.22% |
| colleague_of | 1 | 1.22% |
| represents | 1 | 1.22% |
| sponsored_by | 1 | 1.22% |
| sued_by | 1 | 1.22% |
| charged_with | 1 | 1.22% |
| treats | 1 | 1.22% |
| leader_of | 1 | 1.22% |

### Classifications

**train, actionability.** Majority class 54.7% of 95 answers. The marginal is `true_label`, not the menu.

#### train actionability

| label | count | share |
| --- | --- | --- |
| informational | 52 | 54.74% |
| requires action | 36 | 37.89% |
| advisory | 7 | 7.37% |

**train, audience.** Majority class 35.4% of 181 answers. The marginal is `true_label`, not the menu.

#### train audience

| label | count | share |
| --- | --- | --- |
| general public | 64 | 35.36% |
| expert | 61 | 33.70% |
| regulatory | 44 | 24.31% |
| internal | 12 | 6.63% |

**train, certainty.** Majority class 99.0% of 105 answers. The marginal is `true_label`, not the menu.

#### train certainty

| label | count | share |
| --- | --- | --- |
| asserted | 104 | 99.05% |
| hedged | 1 | 0.95% |

**train, document_genre.** Majority class 32.0% of 97 answers. The marginal is `true_label`, not the menu.

#### train document_genre

| label | count | share |
| --- | --- | --- |
| news report | 31 | 31.96% |
| announcement | 25 | 25.77% |
| technical documentation | 12 | 12.37% |
| legal filing | 11 | 11.34% |
| advertisement | 7 | 7.22% |
| review | 5 | 5.15% |
| correspondence | 4 | 4.12% |
| opinion | 2 | 2.06% |

**train, formality.** Majority class 95.6% of 91 answers. The marginal is `true_label`, not the menu.

#### train formality

| label | count | share |
| --- | --- | --- |
| formal | 87 | 95.60% |
| informal | 4 | 4.40% |

**train, language_register.** Majority class 65.4% of 104 answers. The marginal is `true_label`, not the menu.

#### train language_register

| label | count | share |
| --- | --- | --- |
| technical | 68 | 65.38% |
| plain | 26 | 25.00% |
| legal | 10 | 9.62% |

**train, risk_level.** Majority class 33.3% of 102 answers. The marginal is `true_label`, not the menu.

#### train risk_level

| label | count | share |
| --- | --- | --- |
| low | 34 | 33.33% |
| high | 28 | 27.45% |
| none | 23 | 22.55% |
| moderate | 17 | 16.67% |

**train, sentiment.** Majority class 57.7% of 78 answers. The marginal is `true_label`, not the menu.

#### train sentiment

| label | count | share |
| --- | --- | --- |
| positive | 45 | 57.69% |
| negative | 20 | 25.64% |
| neutral | 13 | 16.67% |

**train, subjectivity.** Majority class 97.9% of 96 answers. The marginal is `true_label`, not the menu.

#### train subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 94 | 97.92% |
| subjective | 2 | 2.08% |

**train, temporal_orientation.** Majority class 71.1% of 97 answers. The marginal is `true_label`, not the menu.

#### train temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 69 | 71.13% |
| present | 19 | 19.59% |
| future | 5 | 5.15% |
| timeless | 4 | 4.12% |

**train, topic.** Majority class 22.3% of 179 answers. The marginal is `true_label`, not the menu.

#### train topic

| label | count | share |
| --- | --- | --- |
| business | 40 | 22.35% |
| technology | 28 | 15.64% |
| science | 17 | 9.50% |
| finance | 15 | 8.38% |
| health | 14 | 7.82% |
| legal | 14 | 7.82% |
| environment | 11 | 6.15% |
| world | 8 | 4.47% |
| education | 6 | 3.35% |
| sports | 6 | 3.35% |
| defense | 6 | 3.35% |
| politics | 4 | 2.23% |
| energy | 4 | 2.23% |
| travel | 3 | 1.68% |
| agriculture | 2 | 1.12% |
| entertainment | 1 | 0.56% |

**train, urgency.** Majority class 59.3% of 108 answers. The marginal is `true_label`, not the menu.

#### train urgency

| label | count | share |
| --- | --- | --- |
| routine | 64 | 59.26% |
| elevated | 23 | 21.30% |
| critical | 19 | 17.59% |
| urgent | 2 | 1.85% |

**val, actionability.** Majority class 46.7% of 15 answers. The marginal is `true_label`, not the menu.

#### val actionability

| label | count | share |
| --- | --- | --- |
| informational | 7 | 46.67% |
| advisory | 4 | 26.67% |
| requires action | 4 | 26.67% |

**val, audience.** Majority class 33.3% of 24 answers. The marginal is `true_label`, not the menu.

#### val audience

| label | count | share |
| --- | --- | --- |
| expert | 8 | 33.33% |
| general public | 7 | 29.17% |
| regulatory | 6 | 25.00% |
| internal | 3 | 12.50% |

**val, certainty.** Majority class 90.9% of 11 answers. The marginal is `true_label`, not the menu.

#### val certainty

| label | count | share |
| --- | --- | --- |
| asserted | 10 | 90.91% |
| hedged | 1 | 9.09% |

**val, document_genre.** Majority class 40.0% of 20 answers. The marginal is `true_label`, not the menu.

#### val document_genre

| label | count | share |
| --- | --- | --- |
| announcement | 8 | 40.00% |
| news report | 6 | 30.00% |
| correspondence | 2 | 10.00% |
| technical documentation | 2 | 10.00% |
| legal filing | 1 | 5.00% |
| opinion | 1 | 5.00% |

**val, formality.** Majority class 93.8% of 16 answers. The marginal is `true_label`, not the menu.

#### val formality

| label | count | share |
| --- | --- | --- |
| formal | 15 | 93.75% |
| informal | 1 | 6.25% |

**val, language_register.** Majority class 57.9% of 19 answers. The marginal is `true_label`, not the menu.

#### val language_register

| label | count | share |
| --- | --- | --- |
| technical | 11 | 57.89% |
| legal | 5 | 26.32% |
| plain | 3 | 15.79% |

**val, risk_level.** Majority class 33.3% of 6 answers. The marginal is `true_label`, not the menu.

#### val risk_level

| label | count | share |
| --- | --- | --- |
| low | 2 | 33.33% |
| high | 2 | 33.33% |
| none | 1 | 16.67% |
| moderate | 1 | 16.67% |

**val, sentiment.** Majority class 58.3% of 12 answers. The marginal is `true_label`, not the menu.

#### val sentiment

| label | count | share |
| --- | --- | --- |
| negative | 7 | 58.33% |
| positive | 3 | 25.00% |
| neutral | 2 | 16.67% |

**val, subjectivity.** Majority class 81.8% of 11 answers. The marginal is `true_label`, not the menu.

#### val subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 9 | 81.82% |
| subjective | 2 | 18.18% |

**val, temporal_orientation.** Majority class 83.3% of 18 answers. The marginal is `true_label`, not the menu.

#### val temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 15 | 83.33% |
| future | 2 | 11.11% |
| present | 1 | 5.56% |

**val, topic.** Majority class 25.7% of 35 answers. The marginal is `true_label`, not the menu.

#### val topic

| label | count | share |
| --- | --- | --- |
| business | 9 | 25.71% |
| health | 6 | 17.14% |
| legal | 5 | 14.29% |
| finance | 5 | 14.29% |
| science | 3 | 8.57% |
| technology | 3 | 8.57% |
| politics | 2 | 5.71% |
| education | 1 | 2.86% |
| environment | 1 | 2.86% |

**val, urgency.** Majority class 70.0% of 10 answers. The marginal is `true_label`, not the menu.

#### val urgency

| label | count | share |
| --- | --- | --- |
| routine | 7 | 70.00% |
| elevated | 2 | 20.00% |
| critical | 1 | 10.00% |

**test, actionability.** Majority class 75.0% of 16 answers. The marginal is `true_label`, not the menu.

#### test actionability

| label | count | share |
| --- | --- | --- |
| informational | 12 | 75.00% |
| requires action | 4 | 25.00% |

**test, audience.** Majority class 32.1% of 28 answers. The marginal is `true_label`, not the menu.

#### test audience

| label | count | share |
| --- | --- | --- |
| general public | 9 | 32.14% |
| expert | 9 | 32.14% |
| regulatory | 7 | 25.00% |
| internal | 3 | 10.71% |

**test, certainty.** Majority class 100.0% of 14 answers. The marginal is `true_label`, not the menu.

#### test certainty

| label | count | share |
| --- | --- | --- |
| asserted | 14 | 100.00% |

**test, document_genre.** Majority class 30.0% of 10 answers. The marginal is `true_label`, not the menu.

#### test document_genre

| label | count | share |
| --- | --- | --- |
| technical documentation | 3 | 30.00% |
| announcement | 2 | 20.00% |
| correspondence | 1 | 10.00% |
| advertisement | 1 | 10.00% |
| review | 1 | 10.00% |
| legal filing | 1 | 10.00% |
| news report | 1 | 10.00% |

**test, formality.** Majority class 84.6% of 13 answers. The marginal is `true_label`, not the menu.

#### test formality

| label | count | share |
| --- | --- | --- |
| formal | 11 | 84.62% |
| informal | 2 | 15.38% |

**test, language_register.** Majority class 71.4% of 14 answers. The marginal is `true_label`, not the menu.

#### test language_register

| label | count | share |
| --- | --- | --- |
| technical | 10 | 71.43% |
| plain | 4 | 28.57% |

**test, risk_level.** Majority class 41.7% of 12 answers. The marginal is `true_label`, not the menu.

#### test risk_level

| label | count | share |
| --- | --- | --- |
| low | 5 | 41.67% |
| high | 4 | 33.33% |
| none | 2 | 16.67% |
| moderate | 1 | 8.33% |

**test, sentiment.** Majority class 50.0% of 14 answers. The marginal is `true_label`, not the menu.

#### test sentiment

| label | count | share |
| --- | --- | --- |
| positive | 7 | 50.00% |
| negative | 5 | 35.71% |
| neutral | 2 | 14.29% |

**test, subjectivity.** Majority class 93.3% of 15 answers. The marginal is `true_label`, not the menu.

#### test subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 14 | 93.33% |
| subjective | 1 | 6.67% |

**test, temporal_orientation.** Majority class 82.4% of 17 answers. The marginal is `true_label`, not the menu.

#### test temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 14 | 82.35% |
| timeless | 1 | 5.88% |
| present | 1 | 5.88% |
| future | 1 | 5.88% |

**test, topic.** Majority class 26.5% of 34 answers. The marginal is `true_label`, not the menu.

#### test topic

| label | count | share |
| --- | --- | --- |
| business | 9 | 26.47% |
| technology | 5 | 14.71% |
| finance | 5 | 14.71% |
| health | 3 | 8.82% |
| science | 3 | 8.82% |
| environment | 3 | 8.82% |
| legal | 3 | 8.82% |
| world | 1 | 2.94% |
| politics | 1 | 2.94% |
| sports | 1 | 2.94% |

**test, urgency.** Majority class 71.4% of 7 answers. The marginal is `true_label`, not the menu.

#### test urgency

| label | count | share |
| --- | --- | --- |
| routine | 5 | 71.43% |
| elevated | 2 | 28.57% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | condition | 5 | 100.00% |
| clinical_finding | patient_group | 5 | 100.00% |
| clinical_finding | treatment | 4 | 80.00% |
| clinical_finding | outcome | 4 | 80.00% |
| clinical_finding | record_metadata | natural/condition ×5 |  |
| company_profile | name | 28 | 100.00% |
| company_profile | headquarters | 28 | 100.00% |
| company_profile | size | 28 | 100.00% |
| company_profile | industry | 18 | 64.29% |
| company_profile | founded | 10 | 35.71% |
| company_profile | record_metadata | natural/name ×25 |  |
| event_listing | name | 4 | 100.00% |
| event_listing | venue | 4 | 100.00% |
| event_listing | date | 4 | 100.00% |
| event_listing | organizer | 2 | 50.00% |
| event_listing | price | 1 | 25.00% |
| event_listing | record_metadata | natural/name ×4 |  |
| financial_report | organization | 17 | 100.00% |
| financial_report | period | 16 | 94.12% |
| financial_report | revenue | 14 | 82.35% |
| financial_report | net_income | 12 | 70.59% |
| financial_report | record_metadata | natural/organization ×17 |  |
| incident_report | date | 12 | 100.00% |
| incident_report | location | 11 | 91.67% |
| incident_report | severity | 11 | 91.67% |
| incident_report | incident_type | 9 | 75.00% |
| incident_report | casualties | 6 | 50.00% |
| incident_report | record_metadata | natural/incident_type ×9, natural/location ×2, natural/date ×1 |  |
| job_posting | company | 2 | 100.00% |
| job_posting | location | 2 | 100.00% |
| job_posting | employment_type | 2 | 100.00% |
| job_posting | title | 1 | 50.00% |
| job_posting | record_metadata | natural/company ×1, natural/title ×1 |  |
| legal_case | court | 6 | 100.00% |
| legal_case | status | 6 | 100.00% |
| legal_case | defendant | 5 | 83.33% |
| legal_case | case_name | 4 | 66.67% |
| legal_case | plaintiff | 4 | 66.67% |
| legal_case | record_metadata | natural/case_name ×4, natural/court ×2 |  |
| person_profile | name | 25 | 100.00% |
| person_profile | role | 25 | 100.00% |
| person_profile | employer | 25 | 100.00% |
| person_profile | location | 16 | 64.00% |
| person_profile | record_metadata | natural/name ×18 |  |
| product | name | 10 | 100.00% |
| product | condition | 10 | 100.00% |
| product | brand | 8 | 80.00% |
| product | price | 7 | 70.00% |
| product | record_metadata | natural/name ×9 |  |
| real_estate_listing | address | 1 | 100.00% |
| real_estate_listing | price | 1 | 100.00% |
| real_estate_listing | bedrooms | 1 | 100.00% |
| real_estate_listing | area | 1 | 100.00% |
| real_estate_listing | property_type | 1 | 100.00% |
| real_estate_listing | record_metadata | natural/address ×1 |  |
| research_paper | title | 8 | 100.00% |
| research_paper | year | 8 | 100.00% |
| research_paper | venue | 5 | 62.50% |
| research_paper | authors | 3 | 37.50% |
| research_paper | topic | 2 | 25.00% |
| research_paper | record_metadata | natural/title ×8 |  |
| software_release | product | 1 | 100.00% |
| software_release | version | 1 | 100.00% |
| software_release | release_date | 1 | 100.00% |
| software_release | platform | 1 | 100.00% |
| software_release | record_metadata | natural/product ×1 |  |
| transaction | amount | 11 | 100.00% |
| transaction | item | 10 | 90.91% |
| transaction | seller | 9 | 81.82% |
| transaction | date | 8 | 72.73% |
| transaction | buyer | 5 | 45.45% |
| transaction | record_metadata | natural/item ×10, natural/amount ×1 |  |
| vehicle_listing | make | 3 | 100.00% |
| vehicle_listing | model | 3 | 100.00% |
| vehicle_listing | price | 3 | 100.00% |
| vehicle_listing | year | 1 | 33.33% |
| vehicle_listing | record_metadata | natural/make ×3 |  |

**val.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | condition | 1 | 100.00% |
| clinical_finding | patient_group | 1 | 100.00% |
| clinical_finding | treatment | 1 | 100.00% |
| clinical_finding | outcome | 1 | 100.00% |
| clinical_finding | record_metadata | natural/condition ×1 |  |
| company_profile | name | 1 | 100.00% |
| company_profile | headquarters | 1 | 100.00% |
| company_profile | size | 1 | 100.00% |
| company_profile | record_metadata | natural/name ×1 |  |
| incident_report | date | 4 | 100.00% |
| incident_report | severity | 4 | 100.00% |
| incident_report | location | 3 | 75.00% |
| incident_report | casualties | 2 | 50.00% |
| incident_report | incident_type | 2 | 50.00% |
| incident_report | record_metadata | natural/incident_type ×2, natural/date ×1, natural/location ×1 |  |
| legal_case | case_name | 1 | 100.00% |
| legal_case | court | 1 | 100.00% |
| legal_case | plaintiff | 1 | 100.00% |
| legal_case | defendant | 1 | 100.00% |
| legal_case | status | 1 | 100.00% |
| legal_case | record_metadata | natural/case_name ×1 |  |
| person_profile | name | 3 | 100.00% |
| person_profile | role | 3 | 100.00% |
| person_profile | employer | 3 | 100.00% |
| person_profile | location | 3 | 100.00% |
| person_profile | record_metadata | natural/name ×3 |  |
| product | name | 1 | 100.00% |
| product | brand | 1 | 100.00% |
| product | condition | 1 | 100.00% |
| product | record_metadata | natural/name ×1 |  |
| research_paper | title | 1 | 100.00% |
| research_paper | venue | 1 | 100.00% |
| research_paper | year | 1 | 100.00% |
| research_paper | topic | 1 | 100.00% |
| research_paper | record_metadata | natural/title ×1 |  |
| transaction | item | 2 | 100.00% |
| transaction | date | 2 | 100.00% |
| transaction | amount | 1 | 50.00% |
| transaction | buyer | 1 | 50.00% |
| transaction | seller | 1 | 50.00% |
| transaction | record_metadata | natural/item ×2 |  |
| vehicle_listing | make | 1 | 100.00% |
| vehicle_listing | model | 1 | 100.00% |
| vehicle_listing | year | 1 | 100.00% |
| vehicle_listing | price | 1 | 100.00% |
| vehicle_listing | record_metadata | natural/make ×1 |  |

**test.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| company_profile | name | 3 | 100.00% |
| company_profile | industry | 3 | 100.00% |
| company_profile | headquarters | 3 | 100.00% |
| company_profile | founded | 3 | 100.00% |
| company_profile | size | 3 | 100.00% |
| company_profile | record_metadata | natural/name ×3 |  |
| event_listing | name | 1 | 100.00% |
| event_listing | venue | 1 | 100.00% |
| event_listing | date | 1 | 100.00% |
| event_listing | record_metadata | natural/name ×1 |  |
| person_profile | name | 3 | 100.00% |
| person_profile | role | 3 | 100.00% |
| person_profile | employer | 3 | 100.00% |
| person_profile | location | 3 | 100.00% |
| person_profile | record_metadata | natural/name ×3 |  |
| product | name | 3 | 100.00% |
| product | condition | 3 | 100.00% |
| product | price | 2 | 66.67% |
| product | brand | 1 | 33.33% |
| product | record_metadata | natural/name ×3 |  |
| transaction | item | 4 | 100.00% |
| transaction | amount | 4 | 100.00% |
| transaction | seller | 4 | 100.00% |
| transaction | date | 4 | 100.00% |
| transaction | buyer | 3 | 75.00% |
| transaction | record_metadata | natural/item ×4 |  |
| vehicle_listing | make | 1 | 100.00% |
| vehicle_listing | model | 1 | 100.00% |
| vehicle_listing | year | 1 | 100.00% |
| vehicle_listing | price | 1 | 100.00% |
| vehicle_listing | record_metadata | natural/make ×1 |  |

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | unknown |
| relations | unknown |
| events | unknown |
| classifications | unknown |
| structures | unknown |
