# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tr_dose0 | train | 18,109 | 18,109 | 22,249 | 1,803 | 2,775 | 24,145 | 3,780 | 4,152 |
| tr_dose0 | val | 1,115 | 1,115 | 0 | 0 | 0 | 1,633 | 0 | 0 |
| tr_dose0 | test | 1,115 | 1,115 | 0 | 0 | 0 | 1,680 | 0 | 0 |

## tr_dose0

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 18,109 | 18,109 | 206 | 527 | 9823 | 22,249 | 1,803 | 2,775 | 24,145 | 3,780 | 4,152 |
| val | 1,115 | 1,115 | 194 | 413 | 1145 | 0 | 0 | 0 | 1,633 | 0 | 0 |
| test | 1,115 | 1,115 | 200 | 437 | 1110 | 0 | 0 | 0 | 1,680 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 15,472 | 322 | 345 | 16,139 | 79.35% |
| zho | 2,469 | 0 | 0 | 2,469 | 12.14% |
| tur | 1 | 793 | 770 | 1,564 | 7.69% |
| und | 115 | 0 | 0 | 115 | 0.57% |
| spa | 6 | 0 | 0 | 6 | 0.03% |
| fra | 6 | 0 | 0 | 6 | 0.03% |
| kor | 5 | 0 | 0 | 5 | 0.02% |
| ita | 5 | 0 | 0 | 5 | 0.02% |
| hin | 5 | 0 | 0 | 5 | 0.02% |
| pol | 4 | 0 | 0 | 4 | 0.02% |
| nld | 4 | 0 | 0 | 4 | 0.02% |
| rus | 3 | 0 | 0 | 3 | 0.01% |
| ara | 3 | 0 | 0 | 3 | 0.01% |
| por | 3 | 0 | 0 | 3 | 0.01% |
| deu | 3 | 0 | 0 | 3 | 0.01% |
| nor | 2 | 0 | 0 | 2 | 0.01% |
| ukr | 1 | 0 | 0 | 1 | 0.00% |
| cym | 1 | 0 | 0 | 1 | 0.00% |
| ilo | 1 | 0 | 0 | 1 | 0.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 1,115 |
| test_unique_inputs | 1,115 |
| train_records | 18,109 |
| train_unique_inputs | 18,109 |
| val_records | 1,115 |
| val_unique_inputs | 1,115 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | n/a | 22,249 / 0 |  |
| entities (per mention) | train / test | n/a | 22,249 / 0 |  |
| relations (per instance) | train / val | n/a | 1,803 / 0 |  |
| relations (per instance) | train / test | n/a | 1,803 / 0 |  |
| event types (per trigger mention) | train / val | n/a | 4,337 / 0 |  |
| event types (per trigger mention) | train / test | n/a | 4,337 / 0 |  |
| event roles (per argument) | train / val | n/a | 4,397 / 0 |  |
| event roles (per argument) | train / test | n/a | 4,397 / 0 |  |
| classifications (true_label) | train / val | n/a | 2,775 / 0 |  |
| classifications (true_label) | train / test | n/a | 2,775 / 0 |  |


### Events

**train.** 4,189 instances in 1,030 records, 0 triggerless. Trigger gold 3,780. Argument gold 4,152. Arguments per instance, mean 1.05. Trigger length in whitespace tokens, mean 1.39, single-token share 64.3%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Experiment | 323 | 7.45% |
| Event | 216 | 4.98% |
| Support | 207 | 4.77% |
| Hostile_encounter | 119 | 2.74% |
| Deploy | 117 | 2.70% |
| Process_start | 99 | 2.28% |
| Catastrophe | 89 | 2.05% |
| Attack | 87 | 2.01% |
| Exhibit | 79 | 1.82% |
| match_result | 74 | 1.71% |
| Causation | 73 | 1.68% |
| Cyber.DiscoverVulnerability | 71 | 1.64% |
| Motion | 67 | 1.54% |
| Competition | 63 | 1.45% |
| Process_end | 60 | 1.38% |
| product_publish | 53 | 1.22% |
| Cyber.Ransom | 52 | 1.20% |
| Conquering | 48 | 1.11% |
| Control | 48 | 1.11% |
| Statement | 45 | 1.04% |
| Killing | 42 | 0.97% |
| Cyber.Databreach | 40 | 0.92% |
| Death | 39 | 0.90% |
| Cyber.PatchVulnerability | 37 | 0.85% |
| Destroying | 36 | 0.83% |
| Accident | 35 | 0.81% |
| Military_operation | 35 | 0.81% |
| Preventing_or_letting | 34 | 0.78% |
| Bodily_harm | 34 | 0.78% |
| Arranging | 32 | 0.74% |
| Coming_to_be | 31 | 0.71% |
| Cause_change_of_position_on_a_scale | 31 | 0.71% |
| Arriving | 31 | 0.71% |
| Manoeuvre | 31 | 0.71% |
| Cause_to_be_included | 31 | 0.71% |
| death | 29 | 0.67% |
| Self_motion | 29 | 0.67% |
| Change | 29 | 0.67% |
| Hold | 28 | 0.65% |
| Social_event | 28 | 0.65% |
| judicial_action_arrest | 27 | 0.62% |
| Getting | 27 | 0.62% |
| Presence | 27 | 0.62% |
| Arrest | 27 | 0.62% |
| Giving | 26 | 0.60% |
| Know | 26 | 0.60% |
| Action | 26 | 0.60% |
| Creating | 26 | 0.60% |
| Traveling | 25 | 0.58% |
| Defending | 25 | 0.58% |
| Removing | 24 | 0.55% |
| Becoming | 24 | 0.55% |
| Supporting | 24 | 0.55% |
| Escaping | 24 | 0.55% |
| Supply | 24 | 0.55% |
| Change_of_leadership | 24 | 0.55% |
| Damaging | 23 | 0.53% |
| resignation | 23 | 0.53% |
| Earnings_and_losses | 23 | 0.53% |
| Injure | 21 | 0.48% |
| Agree_or_refuse_to_act | 21 | 0.48% |
| Cause_change_of_strength | 20 | 0.46% |
| Conflict | 20 | 0.46% |
| Placing | 20 | 0.46% |
| Judgment_communication | 20 | 0.46% |
| Expressing_publicly | 20 | 0.46% |
| Cyber.Phishing | 20 | 0.46% |
| Protest | 20 | 0.46% |
| championship_win | 19 | 0.44% |
| Name_conferral | 18 | 0.42% |
| Request | 18 | 0.42% |
| advancement | 18 | 0.42% |
| Motion_directional | 17 | 0.39% |
| Violence | 17 | 0.39% |
| Sending | 16 | 0.37% |
| Achieve | 16 | 0.37% |
| Besieging | 16 | 0.37% |
| Departing | 16 | 0.37% |
| Use_firearm | 16 | 0.37% |
| Perception_active | 15 | 0.35% |
| Using | 15 | 0.35% |
| product_recall | 15 | 0.35% |
| Expansion | 14 | 0.32% |
| divorce | 14 | 0.32% |
| opening | 14 | 0.32% |
| joined | 13 | 0.30% |
| Influence | 13 | 0.30% |
| Reporting | 13 | 0.30% |
| Deciding | 13 | 0.30% |
| Come_together | 13 | 0.30% |
| Building | 13 | 0.30% |
| Assistance | 13 | 0.30% |
| Communication | 13 | 0.30% |
| Recording | 12 | 0.28% |
| price_limit_up | 12 | 0.28% |
| Choosing | 12 | 0.28% |
| judicial_action_summon | 12 | 0.28% |
| Aiming | 12 | 0.28% |
| Participation | 12 | 0.28% |
| Recovering | 10 | 0.23% |
| Traffic Accident | 10 | 0.23% |
| breakup | 10 | 0.23% |
| Check | 10 | 0.23% |
| Change_event_time | 10 | 0.23% |
| Collaboration | 9 | 0.21% |
| Cause_to_make_progress | 9 | 0.21% |
| marriage | 9 | 0.21% |
| Hindering | 9 | 0.21% |
| Being_in_operation | 9 | 0.21% |
| Receiving | 9 | 0.21% |
| Surrendering | 9 | 0.21% |
| competition_ban | 9 | 0.21% |
| Bringing | 9 | 0.21% |
| Terrorism | 9 | 0.21% |
| Openness | 9 | 0.21% |
| Cause_to_amalgamate | 9 | 0.21% |
| birthday_celebration | 8 | 0.18% |
| Connect | 8 | 0.18% |
| Legal_rulings | 8 | 0.18% |
| Sign_agreement | 8 | 0.18% |
| Quarreling | 8 | 0.18% |
| departure | 7 | 0.16% |
| sale_acquisition | 7 | 0.16% |
| financing | 7 | 0.16% |
| Response | 7 | 0.16% |
| product_delist | 7 | 0.16% |
| Manufacturing | 7 | 0.16% |
| Becoming_a_member | 7 | 0.16% |
| Scrutiny | 6 | 0.14% |
| Preserving | 6 | 0.14% |
| Commitment | 6 | 0.14% |
| Legality | 6 | 0.14% |
| Education_teaching | 6 | 0.14% |
| Dispersal | 6 | 0.14% |
| Publishing | 6 | 0.14% |
| Telling | 6 | 0.14% |
| judicial_action_imprisonment | 6 | 0.14% |
| Commerce_pay | 6 | 0.14% |
| Forming_relationships | 6 | 0.14% |
| Plane Crash | 5 | 0.12% |
| closing | 5 | 0.12% |
| Warning | 5 | 0.12% |
| Scouring | 5 | 0.12% |
| Commerce_sell | 5 | 0.12% |
| lost_contact | 5 | 0.12% |
| Employment | 5 | 0.12% |
| Fire | 5 | 0.12% |
| judicial_action_report | 5 | 0.12% |
| product_release | 5 | 0.12% |
| Committing_crime | 5 | 0.12% |
| Research | 5 | 0.12% |
| wedding | 5 | 0.12% |
| Confronting_problem | 5 | 0.12% |
| Award | 5 | 0.12% |
| judicial_action_fine | 5 | 0.12% |
| product_award | 4 | 0.09% |
| Explosion | 4 | 0.09% |
| Wearing | 4 | 0.09% |
| Expend_resource | 4 | 0.09% |
| competition_withdrawal | 4 | 0.09% |
| Bearing_arms | 4 | 0.09% |
| GetReady | 4 | 0.09% |
| layoff | 4 | 0.09% |
| birth | 4 | 0.09% |
| proposal | 4 | 0.09% |
| Practice | 4 | 0.09% |
| Adducing | 4 | 0.09% |
| Coming_to_believe | 4 | 0.09% |
| Rite | 4 | 0.09% |
| apologize_to | 4 | 0.09% |
| Earthquake | 3 | 0.07% |
| strike | 3 | 0.07% |
| GiveUp | 3 | 0.07% |
| Collapse | 3 | 0.07% |
| Convincing | 3 | 0.07% |
| suspension | 3 | 0.07% |
| Testing | 3 | 0.07% |
| Resolve_problem | 3 | 0.07% |
| engagement | 3 | 0.07% |
| Writing | 3 | 0.07% |
| Surrounding | 3 | 0.07% |
| Criminal_investigation | 3 | 0.07% |
| dissolution | 3 | 0.07% |
| praise | 3 | 0.07% |
| Reforming_a_system | 3 | 0.07% |
| judicial_action_filing | 3 | 0.07% |
| judicial_action_prosecution | 3 | 0.07% |
| contract_termination | 2 | 0.05% |
| Rescuing | 2 | 0.05% |
| Temporary_stay | 2 | 0.05% |
| visit_on_set | 2 | 0.05% |
| retirement | 2 | 0.05% |
| rate_increase | 2 | 0.05% |
| Releasing | 2 | 0.05% |
| Filling | 2 | 0.05% |
| march | 2 | 0.05% |
| Incident | 2 | 0.05% |
| infidelity | 2 | 0.05% |
| Patrolling | 2 | 0.05% |
| price_increase | 2 | 0.05% |
| price_limit_down | 2 | 0.05% |
| Create_artwork | 2 | 0.05% |
| Submitting_documents | 2 | 0.05% |
| thank | 2 | 0.05% |
| judicial_action_hearing | 2 | 0.05% |
| Reveal_secret | 2 | 0.05% |
| Limiting | 2 | 0.05% |
| Imposing_obligation | 2 | 0.05% |
| Institutionalization | 2 | 0.05% |
| Revenge | 2 | 0.05% |
| meet_with | 2 | 0.05% |
| Emptying | 2 | 0.05% |
| Body_movement | 2 | 0.05% |
| Vocalizations | 2 | 0.05% |
| Rewards_and_punishments | 2 | 0.05% |
| Flood | 1 | 0.02% |
| Prison | 1 | 0.02% |
| Change_sentiment | 1 | 0.02% |
| Robbery | 1 | 0.02% |
| Lighting | 1 | 0.02% |
| Carry_goods | 1 | 0.02% |
| pregnancy | 1 | 0.02% |
| Justifying | 1 | 0.02% |
| Commerce_buy | 1 | 0.02% |
| Kidnapping | 1 | 0.02% |
| price_decrease | 1 | 0.02% |
| rate_decrease | 1 | 0.02% |
| dismissal | 1 | 0.02% |
| Suspicion | 1 | 0.02% |
| Theft | 1 | 0.02% |
| Cure | 1 | 0.02% |
| Having_or_lacking_access | 1 | 0.02% |
| Containing | 1 | 0.02% |

**val.** No event instances.

**test.** No event instances.

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 4,397 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.72 | 30.19% | 1.72 | 27.58% | 0.05% | 0 | 4,397 | 0 | 4,337 |
| val | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |
| test | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4,189 | 1.31% | 3,156 | 12.61% | 12.01% | 1.52% | 0.51% | 15.83% |
| val | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |
| test | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 55 | 320 | 3,156 |
| val | 0 | 0 | 0 |
| test | 0 | 0 | 0 |

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
| StartDate | 1,310 | 5.89% |
| EquityHolder | 1,143 | 5.14% |
| StockCode | 1,015 | 4.56% |
| CompanyName | 866 | 3.89% |
| StockAbbr | 858 | 3.86% |
| Location | 820 | 3.69% |
| PledgedShares | 807 | 3.63% |
| Date | 714 | 3.21% |
| Pledgee | 664 | 2.98% |
| TotalHoldingShares | 645 | 2.90% |
| TradedShares | 582 | 2.62% |
| TotalHoldingRatio | 571 | 2.57% |
| Casualties and Losses | 563 | 2.53% |
| Reduction Amount | 508 | 2.28% |
| TotalPledgedShares | 440 | 1.98% |
| EndDate | 407 | 1.83% |
| Reducing Shareholder | 389 | 1.75% |
| ReleasedDate | 372 | 1.67% |
| Cause | 332 | 1.49% |
| Reduction Start Date | 302 | 1.36% |
| Company Name | 242 | 1.09% |
| Announcement Date | 230 | 1.03% |
| LaterHoldingShares | 195 | 0.88% |
| pledge_amount | 164 | 0.74% |
| AveragePrice | 136 | 0.61% |
| Person | 132 | 0.59% |
| Affected Areas | 122 | 0.55% |
| Candidates and their Political Parties | 115 | 0.52% |
| Recipient | 112 | 0.50% |
| pledgor | 109 | 0.49% |
| Organization | 105 | 0.47% |
| Accepting Court | 97 | 0.44% |
| RepurchasedShares | 95 | 0.43% |
| Weapon and Equippment | 91 | 0.41% |
| Belligerents | 90 | 0.40% |
| Accident Investigator | 89 | 0.40% |
| Protest Reason | 89 | 0.40% |
| protein | 87 | 0.39% |
| Loss Amount | 86 | 0.39% |
| Protesters | 83 | 0.37% |
| HighestTradingPrice | 81 | 0.36% |
| Position | 81 | 0.36% |
| Attacker | 78 | 0.35% |
| Vulnerability | 78 | 0.35% |
| DiseaseOrPhenotypicFeature | 76 | 0.34% |
| Frozen Amount | 73 | 0.33% |
| Increase Amount | 69 | 0.31% |
| Charged Crime | 69 | 0.31% |
| Flight No. | 67 | 0.30% |
| ClosingDate | 67 | 0.30% |
| Increasing Shareholder | 67 | 0.30% |
| Accusation | 66 | 0.30% |
| ruling_date | 66 | 0.30% |
| Contest Participant | 65 | 0.29% |
| pledge_start_date | 65 | 0.29% |
| Target | 64 | 0.29% |
| Affected Area | 64 | 0.29% |
| Scheduled Landing Place | 62 | 0.28% |
| Individual_protein | 62 | 0.28% |
| ChemicalEntity | 62 | 0.28% |
| Habitat | 60 | 0.27% |
| Influence People | 59 | 0.27% |
| RepurchaseAmount | 57 | 0.26% |
| frozen_shareholder | 57 | 0.26% |
| Areas affected | 57 | 0.26% |
| LegalInstitution | 56 | 0.25% |
| Husband | 55 | 0.25% |
| Policy Content | 54 | 0.24% |
| Freeze Start Date | 53 | 0.24% |
| Aid Agency | 52 | 0.23% |
| Aircraft Agency | 51 | 0.23% |
| Wife | 50 | 0.22% |
| Solution | 48 | 0.22% |
| Microorganism | 48 | 0.22% |
| The Final Votes and Percentages | 46 | 0.21% |
| pledge_end_date | 45 | 0.20% |
| Suspect | 45 | 0.20% |
| Judgement Result/Prison Term | 44 | 0.20% |
| Crew | 42 | 0.19% |
| Acquiree | 42 | 0.19% |
| FrozeShares | 41 | 0.18% |
| Capabilities | 41 | 0.18% |
| Storm Hit Location | 40 | 0.18% |
| GeneOrGeneProduct | 39 | 0.18% |
| Pollution Source | 39 | 0.18% |
| Anti-pollution People/Organizations | 39 | 0.18% |
| Accused People | 39 | 0.18% |
| Disease | 38 | 0.17% |
| Increase Start Date | 37 | 0.17% |
| PII | 37 | 0.17% |
| Strike Agency | 36 | 0.16% |
| Riot Reason | 35 | 0.16% |
| Precautionary Measure | 35 | 0.16% |
| Strike Outcome | 34 | 0.15% |
| Passengers | 33 | 0.15% |
| System | 33 | 0.15% |
| Aid Supplies/Amount | 32 | 0.14% |
| Resignated Person | 31 | 0.14% |
| Child | 31 | 0.14% |
| Participants | 30 | 0.13% |
| Champions | 30 | 0.13% |
| Location/Hospital | 29 | 0.13% |
| Strikers Status | 29 | 0.13% |
| LowestTradingPrice | 29 | 0.13% |
| Organization Name | 29 | 0.13% |
| Election Name | 29 | 0.13% |
| Time | 29 | 0.13% |
| Appointee | 29 | 0.13% |
| Summit Theme | 28 | 0.13% |
| Spectacle Location | 28 | 0.13% |
| Organization Industry | 27 | 0.12% |
| Area Affected | 27 | 0.12% |
| Survivors | 27 | 0.12% |
| Damaged Facility | 26 | 0.12% |
| Magnitude | 26 | 0.12% |
| Deliberating Agency | 26 | 0.12% |
| Employment Agency | 26 | 0.12% |
| compensation_amount | 25 | 0.11% |
| OrganismTaxon | 25 | 0.11% |
| Game Name | 25 | 0.11% |
| Record-breaking Project | 25 | 0.11% |
| Artifacts and Their Chronology | 25 | 0.11% |
| Marriage Duration | 25 | 0.11% |
| The Agency Who Broke the Agreement | 25 | 0.11% |
| Record Breaker | 24 | 0.11% |
| Prosecutor | 24 | 0.11% |
| Deceased | 23 | 0.10% |
| Court | 23 | 0.10% |
| Countries Withdrawing from the Organization | 23 | 0.10% |
| Freeze End Date | 23 | 0.10% |
| Resign Reason | 23 | 0.10% |
| protein-family | 22 | 0.10% |
| Start Date | 22 | 0.10% |
| Way for Spreading | 22 | 0.10% |
| Data | 22 | 0.10% |
| Person under Investigation | 21 | 0.09% |
| Method | 21 | 0.09% |
| Poisoning Type | 21 | 0.09% |
| Army | 21 | 0.09% |
| compensation_recipient | 20 | 0.09% |
| Weapon | 20 | 0.09% |
| Speaker Status | 20 | 0.09% |
| Grades | 20 | 0.09% |
| Effective Date | 20 | 0.09% |
| Score | 20 | 0.09% |
| Spectacle Start Time | 20 | 0.09% |
| Death Reason | 19 | 0.09% |
| Types of the Spectacle | 19 | 0.09% |
| Previous Record Holder | 19 | 0.09% |
| Discover Location | 19 | 0.09% |
| Confirmed/Infected Cases | 19 | 0.09% |
| Number of Damaged Houses | 19 | 0.09% |
| Strike Reason | 19 | 0.09% |
| Patch | 19 | 0.09% |
| Age | 18 | 0.08% |
| Economic Loss | 18 | 0.08% |
| SequenceVariant | 18 | 0.08% |
| senior_personnel | 18 | 0.08% |
| senior_position | 18 | 0.08% |
| Historical Sites | 18 | 0.08% |
| Device | 18 | 0.08% |
| Time of the Case | 18 | 0.08% |
| Winner | 18 | 0.08% |
| Gene/protein/RNA | 17 | 0.08% |
| Declarer | 17 | 0.08% |
| The Grades of the Previous Record Holder | 17 | 0.08% |
| Country Visited | 17 | 0.08% |
| Taking-off Place | 17 | 0.08% |
| Profession | 16 | 0.07% |
| Storm Name | 16 | 0.07% |
| People/Organization who predicted the disaster | 16 | 0.07% |
| Number of Vehicles Involved in the Crash | 16 | 0.07% |
| Protest Slogan | 16 | 0.07% |
| Outbreak Location | 16 | 0.07% |
| Defense Lawyer | 15 | 0.07% |
| Last Job of the Appointee | 15 | 0.07% |
| Award Reason | 15 | 0.07% |
| Software | 15 | 0.07% |
| Head of Investigation Team | 14 | 0.06% |
| Speaker | 14 | 0.06% |
| Released People | 14 | 0.06% |
| Contracting Parties | 14 | 0.06% |
| Number of Influenced People | 14 | 0.06% |
| Approver | 14 | 0.06% |
| Predecessor | 14 | 0.06% |
| Arrest Time | 14 | 0.06% |
| Award | 14 | 0.06% |
| Version | 14 | 0.06% |
| Participating Countries | 14 | 0.06% |
| Tribe | 14 | 0.06% |
| Strikers | 13 | 0.06% |
| Head of the Merged Organization | 13 | 0.06% |
| Epicenter | 13 | 0.06% |
| Conflict Duration | 13 | 0.06% |
| Death Time | 13 | 0.06% |
| Archaeologist | 13 | 0.06% |
| Pests | 13 | 0.06% |
| Competition Items | 13 | 0.06% |
| Evidence | 13 | 0.06% |
| Malware | 13 | 0.06% |
| Number of Evacuated People | 13 | 0.06% |
| Symptoms | 13 | 0.06% |
| Agreement Name | 13 | 0.06% |
| End Date | 12 | 0.05% |
| Investigative Agency | 12 | 0.05% |
| chemical | 12 | 0.05% |
| Visitor | 12 | 0.05% |
| Areas Affected | 12 | 0.05% |
| The Charged Crime | 12 | 0.05% |
| Train No. | 12 | 0.05% |
| Phenotype | 12 | 0.05% |
| Storm Hit Time | 12 | 0.05% |
| People | 11 | 0.05% |
| Outbreak Date | 11 | 0.05% |
| Arrested | 11 | 0.05% |
| Death Cases | 11 | 0.05% |
| Missings | 11 | 0.05% |
| Court Time | 11 | 0.05% |
| Judge | 11 | 0.05% |
| Work | 11 | 0.05% |
| Response Measures | 10 | 0.04% |
| Electoral System | 10 | 0.04% |
| Website | 10 | 0.04% |
| Maximum Wind Speed | 9 | 0.04% |
| Storm Warning Level | 9 | 0.04% |
| Organization Members | 9 | 0.04% |
| Volcano Name | 9 | 0.04% |
| Release Reason | 9 | 0.04% |
| Boycotted Institutions | 9 | 0.04% |
| Annoncement Date | 9 | 0.04% |
| Merger Terms | 9 | 0.04% |
| Number of Victims | 9 | 0.04% |
| Purpose | 9 | 0.04% |
| Withdraw Reason | 9 | 0.04% |
| The Final Seats Result | 9 | 0.04% |
| cell | 9 | 0.04% |
| Term of Office | 9 | 0.04% |
| Rescue Organizer | 9 | 0.04% |
| News Release Agency | 9 | 0.04% |
| Last Time the Record was Broken | 9 | 0.04% |
| Concept | 9 | 0.04% |
| Spacecraft | 9 | 0.04% |
| Property Division | 9 | 0.04% |
| Mathematical Concept | 9 | 0.04% |
| Achievement | 8 | 0.04% |
| Symptom | 8 | 0.04% |
| Economic loss | 8 | 0.04% |
| Number of Destroyed Building | 8 | 0.04% |
| Archaeologist Organization | 8 | 0.04% |
| Casualities and Losses | 8 | 0.04% |
| Epidemic Data Issuing Agency | 8 | 0.04% |
| UnfrozeDate | 8 | 0.04% |
| Voting method | 8 | 0.04% |
| Other Loss | 8 | 0.04% |
| Head of Institution | 8 | 0.04% |
| End Time | 8 | 0.04% |
| Inauguration Time | 8 | 0.04% |
| Bank Name | 8 | 0.04% |
| Investigating agency | 8 | 0.04% |
| Host | 8 | 0.04% |
| Witness | 8 | 0.04% |
| Attending Hospital | 8 | 0.04% |
| DNA | 8 | 0.04% |
| Mathematical Structure | 8 | 0.04% |
| Acquisition amount | 7 | 0.03% |
| protein-domain | 7 | 0.03% |
| Ways to Watch the Speech | 7 | 0.03% |
| Sentencing Location | 7 | 0.03% |
| Agreement Content | 7 | 0.03% |
| Duration | 7 | 0.03% |
| Party Responsible for Pollution | 7 | 0.03% |
| Successor | 7 | 0.03% |
| File | 7 | 0.03% |
| Age of the Appointee | 7 | 0.03% |
| Money | 7 | 0.03% |
| Casualty Count | 7 | 0.03% |
| Perpetrators | 7 | 0.03% |
| Goal | 7 | 0.03% |
| Influenced Crops and Livelihood | 7 | 0.03% |
| algorithm | 7 | 0.03% |
| Policy Proposals | 7 | 0.03% |
| experimental-construct | 6 | 0.03% |
| Summit Name | 6 | 0.03% |
| Transportation | 6 | 0.03% |
| Election Goal | 6 | 0.03% |
| PaymentMethod | 6 | 0.03% |
| Number | 6 | 0.03% |
| Treatment Method | 6 | 0.03% |
| Victim | 6 | 0.03% |
| Child Custody | 6 | 0.03% |
| Susceptible Population | 6 | 0.03% |
| Forecasting Agency | 6 | 0.03% |
| Spectacle End Time | 6 | 0.03% |
| Employed Institution | 6 | 0.03% |
| Appointment Time | 6 | 0.03% |
| Release Time | 6 | 0.03% |
| Hull Discovery Time | 6 | 0.03% |
| Spacecraft Mission | 6 | 0.03% |
| Prosecution Lawyer | 6 | 0.03% |
| Tear Up Reason | 6 | 0.03% |
| species | 6 | 0.03% |
| protein-complex | 6 | 0.03% |
| Government Reaction | 5 | 0.02% |
| Acquirer | 5 | 0.02% |
| Number of Aftershocks | 5 | 0.02% |
| Magnitude(Tsunami heights) | 5 | 0.02% |
| Company Industry | 5 | 0.02% |
| GPE | 5 | 0.02% |
| Age of the Resignated Person | 5 | 0.02% |
| Spectacle Duration | 5 | 0.02% |
| Illness | 5 | 0.02% |
| Damaged Property | 5 | 0.02% |
| Candidate Scandal(Allegations of fraud, etc.) | 5 | 0.02% |
| Disaster-Stricken Farmland | 5 | 0.02% |
| Jail Time | 5 | 0.02% |
| Investigating Spokesperson | 5 | 0.02% |
| Announce Platform | 5 | 0.02% |
| Launch Date | 5 | 0.02% |
| Scale | 5 | 0.02% |
| Commanders and their Position | 5 | 0.02% |
| Hostage | 5 | 0.02% |
| Field of Study | 5 | 0.02% |
| Literary Work | 5 | 0.02% |
| Amount of Precipitation | 5 | 0.02% |
| Cooperative Agency | 5 | 0.02% |
| Launch Site | 5 | 0.02% |
| Award Field | 5 | 0.02% |
| loss_function | 5 | 0.02% |
| Number of Trapped People | 5 | 0.02% |
| Penalty Amount | 5 | 0.02% |
| Fine Reason | 5 | 0.02% |
| Art Form | 5 | 0.02% |
| Group | 5 | 0.02% |
| Publication | 5 | 0.02% |
| Doctor and Medical Team | 4 | 0.02% |
| Best Way to Shoot | 4 | 0.02% |
| Tsunamis | 4 | 0.02% |
| Number of Rescued People | 4 | 0.02% |
| Head of the Institution | 4 | 0.02% |
| Discover Time | 4 | 0.02% |
| MVP | 4 | 0.02% |
| State Before Death | 4 | 0.02% |
| Temporary Settlement | 4 | 0.02% |
| Ship No. | 4 | 0.02% |
| Turnout(who win who lost) | 4 | 0.02% |
| experiment-tag | 4 | 0.02% |
| Police | 4 | 0.02% |
| Prison Term | 4 | 0.02% |
| Beneficiary | 4 | 0.02% |
| Prison | 4 | 0.02% |
| Rescue Tool or Method | 4 | 0.02% |
| Water Level | 4 | 0.02% |
| target | 4 | 0.02% |
| Train Agency | 4 | 0.02% |
| Last Outbreak Time | 4 | 0.02% |
| The State of the Volcano (Dormant or Active) | 4 | 0.02% |
| Military Exercise | 4 | 0.02% |
| Spokenmen | 4 | 0.02% |
| Research Agency | 4 | 0.02% |
| chemical_compound | 4 | 0.02% |
| Arrest Location | 4 | 0.02% |
| Military Strength | 4 | 0.02% |
| environment | 4 | 0.02% |
| fusion-protein | 4 | 0.02% |
| Regulatory Authority | 4 | 0.02% |
| Fined Agency | 4 | 0.02% |
| Art Movement | 4 | 0.02% |
| reagent | 3 | 0.01% |
| Organization Leader | 3 | 0.01% |
| Weapon Used | 3 | 0.01% |
| Provisional Government | 3 | 0.01% |
| Lasting Time | 3 | 0.01% |
| Affected Industries | 3 | 0.01% |
| Start_Date | 3 | 0.01% |
| Unemployed Rate | 3 | 0.01% |
| Gene | 3 | 0.01% |
| Funding | 3 | 0.01% |
| Aid Reason | 3 | 0.01% |
| Death Age | 3 | 0.01% |
| agent | 3 | 0.01% |
| Spokeswoman/Spokesman | 3 | 0.01% |
| Shipwreck Reason | 3 | 0.01% |
| Policy Name & Abbreviation | 3 | 0.01% |
| Appointer | 3 | 0.01% |
| chemical_group | 3 | 0.01% |
| physical_property | 3 | 0.01% |
| Perpetrator | 3 | 0.01% |
| Compensation | 3 | 0.01% |
| Agreement Members | 3 | 0.01% |
| Carrier Rocket | 3 | 0.01% |
| Disease Subtype | 3 | 0.01% |
| Medical Professional | 3 | 0.01% |
| model | 3 | 0.01% |
| embodiment | 3 | 0.01% |
| Nation | 3 | 0.01% |
| Storm Center Location | 2 | 0.01% |
| Warning Device | 2 | 0.01% |
| Service Years | 2 | 0.01% |
| Fire Warning Level | 2 | 0.01% |
| Country | 2 | 0.01% |
| Postpone Time | 2 | 0.01% |
| protein-motif | 2 | 0.01% |
| Death Rate | 2 | 0.01% |
| Complications | 2 | 0.01% |
| Hull Location | 2 | 0.01% |
| Launch Country | 2 | 0.01% |
| Mission Duration | 2 | 0.01% |
| Other Impact | 2 | 0.01% |
| organelle | 2 | 0.01% |
| molecular_orbital | 2 | 0.01% |
| researcher_group | 2 | 0.01% |
| researcher | 2 | 0.01% |
| institutional_unit | 2 | 0.01% |
| Agreement Validity Period | 2 | 0.01% |
| Bill Drafting Agency | 2 | 0.01% |
| Maximum Rainfall | 2 | 0.01% |
| Host Country | 2 | 0.01% |
| Start Time | 2 | 0.01% |
| architecture_component | 2 | 0.01% |
| concept | 2 | 0.01% |
| process | 2 | 0.01% |
| Special Medicine | 2 | 0.01% |
| Invalid Date | 2 | 0.01% |
| CVE | 2 | 0.01% |
| modality | 2 | 0.01% |
| metric | 2 | 0.01% |
| Field | 2 | 0.01% |
| Storm Movement Speed | 2 | 0.01% |
| Sponsor | 2 | 0.01% |
| Time Period | 2 | 0.01% |
| Political Entity | 2 | 0.01% |
| Writing System | 2 | 0.01% |
| Event | 2 | 0.01% |
| Civilization | 2 | 0.01% |
| Law | 2 | 0.01% |
| Recovered Amount | 2 | 0.01% |
| Alternate Landing Place | 1 | 0.00% |
| Head of the Government | 1 | 0.00% |
| Damaged Crops & Livestock | 1 | 0.00% |
| End_Date | 1 | 0.00% |
| Criminal Evidence | 1 | 0.00% |
| Storm Direction | 1 | 0.00% |
| Narrative | 1 | 0.00% |
| Text | 1 | 0.00% |
| Invention | 1 | 0.00% |
| Detention Start Time | 1 | 0.00% |
| element | 1 | 0.00% |
| computational_method | 1 | 0.00% |
| software_package | 1 | 0.00% |
| Reasons for the Formation of the Historical sites | 1 | 0.00% |
| Protein_complex | 1 | 0.00% |
| protein-isoform | 1 | 0.00% |
| data_format | 1 | 0.00% |
| data_structure | 1 | 0.00% |
| publication_venue | 1 | 0.00% |
| parameter | 1 | 0.00% |
| Invited Person | 1 | 0.00% |
| The Sentence Claimed by the Prosecutor's Lawyer | 1 | 0.00% |
| Protein_family_or_group | 1 | 0.00% |
| State of the Hull | 1 | 0.00% |
| Number of Vaccinated People | 1 | 0.00% |
| Astronauts | 1 | 0.00% |
| Strike Industry | 1 | 0.00% |
| Related Rivers or Lakes | 1 | 0.00% |
| Sequelae | 1 | 0.00% |
| Protein | 1 | 0.00% |
| Role | 1 | 0.00% |
| Medical Test | 1 | 0.00% |
| Treatment | 1 | 0.00% |
| Community Group | 1 | 0.00% |
| Service | 1 | 0.00% |
| Program | 1 | 0.00% |
| dataset | 1 | 0.00% |
| task_category | 1 | 0.00% |
| capability | 1 | 0.00% |
| group | 1 | 0.00% |
| Language | 1 | 0.00% |
| Word | 1 | 0.00% |
| Object | 1 | 0.00% |
| Medium | 1 | 0.00% |
| Wedding Venue | 1 | 0.00% |
| Projects | 1 | 0.00% |
| Film | 1 | 0.00% |
| Doctor | 1 | 0.00% |
| Mathematical Result | 1 | 0.00% |
| Mathematical Theorem | 1 | 0.00% |
| Symbol | 1 | 0.00% |
| Document | 1 | 0.00% |
| Economists who Predicted the Crisis | 1 | 0.00% |
| Stolen Amount | 1 | 0.00% |
| Launch Result | 1 | 0.00% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| bind | 156 | 8.65% |
| Lives_In | 83 | 4.60% |
| Positive_Correlation | 52 | 2.88% |
| PPI | 52 | 2.88% |
| Association | 49 | 2.72% |
| Negative_Correlation | 25 | 1.39% |
| Exhibits | 21 | 1.16% |
| country | 11 | 0.61% |
| parent taxon | 9 | 0.50% |
| member of | 9 | 0.50% |
| conflict | 9 | 0.50% |
| commanded by | 9 | 0.50% |
| original broadcaster | 9 | 0.50% |
| country for sport | 9 | 0.50% |
| record label | 9 | 0.50% |
| heritage designation | 8 | 0.44% |
| discoverer or inventor | 8 | 0.44% |
| platform | 8 | 0.44% |
| present in work | 8 | 0.44% |
| architect | 8 | 0.44% |
| crosses | 8 | 0.44% |
| located in the administrative territorial entity | 8 | 0.44% |
| package management system | 8 | 0.44% |
| movement | 8 | 0.44% |
| director | 8 | 0.44% |
| position held | 8 | 0.44% |
| conferred by | 8 | 0.44% |
| place of detention | 8 | 0.44% |
| historic county | 8 | 0.44% |
| member of sports team | 8 | 0.44% |
| astronaut mission | 8 | 0.44% |
| father | 7 | 0.39% |
| mountain range | 7 | 0.39% |
| basin country | 7 | 0.39% |
| natural product of taxon | 7 | 0.39% |
| founded by | 7 | 0.39% |
| military branch | 7 | 0.39% |
| located in the statistical territorial entity | 7 | 0.39% |
| represented by | 7 | 0.39% |
| licensed to broadcast to | 7 | 0.39% |
| significant event | 7 | 0.39% |
| operator | 7 | 0.39% |
| located in/on physical feature | 7 | 0.39% |
| league system | 7 | 0.39% |
| owned by | 7 | 0.39% |
| parent organization | 6 | 0.33% |
| connecting service | 6 | 0.33% |
| performer | 6 | 0.33% |
| cast member | 6 | 0.33% |
| position played on team / speciality | 6 | 0.33% |
| publisher | 6 | 0.33% |
| dialect of | 6 | 0.33% |
| owner of | 6 | 0.33% |
| composer | 6 | 0.33% |
| has use | 5 | 0.28% |
| crew member(s) | 5 | 0.28% |
| replaced by | 5 | 0.28% |
| head of government | 5 | 0.28% |
| field of work | 5 | 0.28% |
| fictional universe described in | 5 | 0.28% |
| winner | 5 | 0.28% |
| item operated | 5 | 0.28% |
| award received | 5 | 0.28% |
| librettist | 5 | 0.28% |
| surface played on | 5 | 0.28% |
| author | 5 | 0.28% |
| interested in | 5 | 0.28% |
| pole position | 5 | 0.28% |
| commander of (DEPRECATED) | 5 | 0.28% |
| taxonomic type | 5 | 0.28% |
| participant in | 5 | 0.28% |
| cathedral | 5 | 0.28% |
| edition or translation of | 5 | 0.28% |
| educated at | 5 | 0.28% |
| online service | 5 | 0.28% |
| home venue | 5 | 0.28% |
| connects with | 5 | 0.28% |
| sport | 5 | 0.28% |
| secretary general | 5 | 0.28% |
| newspaper format | 5 | 0.28% |
| officeholder | 5 | 0.28% |
| partner in business or sport | 5 | 0.28% |
| this zoological name is coordinate with | 5 | 0.28% |
| genre | 5 | 0.28% |
| employer | 5 | 0.28% |
| location of first performance | 5 | 0.28% |
| language used | 5 | 0.28% |
| location of formation | 5 | 0.28% |
| cause of destruction | 5 | 0.28% |
| presenter | 5 | 0.28% |
| distributed by | 5 | 0.28% |
| legal form | 5 | 0.28% |
| first appearance | 5 | 0.28% |
| constellation | 5 | 0.28% |
| head of state | 5 | 0.28% |
| has immediate cause | 5 | 0.28% |
| stock market index | 5 | 0.28% |
| located on astronomical body | 5 | 0.28% |
| location of discovery | 5 | 0.28% |
| notable work | 5 | 0.28% |
| set in period | 5 | 0.28% |
| location | 4 | 0.22% |
| inflows | 4 | 0.22% |
| authority | 4 | 0.22% |
| named after | 4 | 0.22% |
| replaces | 4 | 0.22% |
| legislated by | 4 | 0.22% |
| main subject | 4 | 0.22% |
| currency | 4 | 0.22% |
| basic form of government | 4 | 0.22% |
| chairperson | 4 | 0.22% |
| child | 4 | 0.22% |
| public holiday | 4 | 0.22% |
| instrument | 4 | 0.22% |
| from narrative universe | 4 | 0.22% |
| narrative location | 4 | 0.22% |
| worshipped by | 4 | 0.22% |
| product or material produced | 4 | 0.22% |
| contributed to creative work | 4 | 0.22% |
| creator | 4 | 0.22% |
| victory | 4 | 0.22% |
| stock exchange | 4 | 0.22% |
| shares border with | 4 | 0.22% |
| has effect | 4 | 0.22% |
| Comparison | 4 | 0.22% |
| has part(s) | 4 | 0.22% |
| ethnic group | 4 | 0.22% |
| dam | 4 | 0.22% |
| operating system | 4 | 0.22% |
| characters | 4 | 0.22% |
| has cause | 4 | 0.22% |
| family | 4 | 0.22% |
| brand | 4 | 0.22% |
| member of political party | 4 | 0.22% |
| IUCN conservation status | 4 | 0.22% |
| genic_interaction | 4 | 0.22% |
| diocese | 4 | 0.22% |
| contains | 4 | 0.22% |
| operating area | 4 | 0.22% |
| after a work by | 4 | 0.22% |
| mouth of the watercourse | 4 | 0.22% |
| coextensive with | 4 | 0.22% |
| located on street | 4 | 0.22% |
| physically interacts with | 4 | 0.22% |
| nominated for | 4 | 0.22% |
| occupation | 4 | 0.22% |
| parent astronomical body | 4 | 0.22% |
| musical conductor | 4 | 0.22% |
| collection | 4 | 0.22% |
| spouse | 4 | 0.22% |
| part of the series | 4 | 0.22% |
| party chief representative | 4 | 0.22% |
| transport network | 4 | 0.22% |
| maintained by | 4 | 0.22% |
| continent | 4 | 0.22% |
| Bind | 4 | 0.22% |
| medical condition | 3 | 0.17% |
| has anatomical branch | 3 | 0.17% |
| noble title | 3 | 0.17% |
| place of publication | 3 | 0.17% |
| separated from | 3 | 0.17% |
| producer | 3 | 0.17% |
| has works in the collection | 3 | 0.17% |
| location of creation | 3 | 0.17% |
| target | 3 | 0.17% |
| has part(s) of the class | 3 | 0.17% |
| legislative body | 3 | 0.17% |
| director / manager | 3 | 0.17% |
| developer | 3 | 0.17% |
| school district | 3 | 0.17% |
| student of | 3 | 0.17% |
| honorific prefix | 3 | 0.17% |
| objective of project or action | 3 | 0.17% |
| uses | 3 | 0.17% |
| facet of | 3 | 0.17% |
| production company | 3 | 0.17% |
| located in the present-day administrative territorial entity | 3 | 0.17% |
| adjacent station | 3 | 0.17% |
| member of the crew of | 3 | 0.17% |
| interchange station | 3 | 0.17% |
| medical condition treated | 3 | 0.17% |
| located in or next to body of water | 3 | 0.17% |
| instance of | 3 | 0.17% |
| time period | 3 | 0.17% |
| place of death | 3 | 0.17% |
| country of origin | 3 | 0.17% |
| mascot | 3 | 0.17% |
| highest judicial authority | 3 | 0.17% |
| chromosome | 3 | 0.17% |
| carries | 3 | 0.17% |
| sponsor | 3 | 0.17% |
| has edition or translation | 3 | 0.17% |
| located in time zone | 3 | 0.17% |
| said to be the same as | 3 | 0.17% |
| terminus | 3 | 0.17% |
| military rank | 3 | 0.17% |
| start point | 3 | 0.17% |
| broadcast by | 3 | 0.17% |
| successful candidate | 3 | 0.17% |
| grants | 3 | 0.17% |
| political alignment | 3 | 0.17% |
| head coach | 3 | 0.17% |
| illustrator | 3 | 0.17% |
| place served by transport hub | 3 | 0.17% |
| has natural reservoir | 3 | 0.17% |
| endemic to | 3 | 0.17% |
| has boundary | 3 | 0.17% |
| space launch vehicle | 3 | 0.17% |
| made from material | 3 | 0.17% |
| native language | 3 | 0.17% |
| league level above | 3 | 0.17% |
| discography | 3 | 0.17% |
| work location | 3 | 0.17% |
| foundational text | 3 | 0.17% |
| anatomical location | 3 | 0.17% |
| participant | 3 | 0.17% |
| reservoir created | 3 | 0.17% |
| commissioned by | 3 | 0.17% |
| language of work or name | 3 | 0.17% |
| contains the administrative territorial entity | 3 | 0.17% |
| consecrator | 3 | 0.17% |
| opposite of | 3 | 0.17% |
| writing language | 3 | 0.17% |
| drug or therapy used for treatment | 3 | 0.17% |
| primary destinations | 3 | 0.17% |
| substitute/deputy/replacement of office/officeholder | 3 | 0.17% |
| overlies | 3 | 0.17% |
| editor | 3 | 0.17% |
| vessel class | 3 | 0.17% |
| terminus location | 3 | 0.17% |
| possible treatment | 3 | 0.17% |
| voice type | 3 | 0.17% |
| coach of sports team | 3 | 0.17% |
| sibling | 2 | 0.11% |
| distribution format | 2 | 0.11% |
| armament | 2 | 0.11% |
| engine configuration | 2 | 0.11% |
| minor planet group | 2 | 0.11% |
| subject has role | 2 | 0.11% |
| form of creative work | 2 | 0.11% |
| lake on watercourse | 2 | 0.11% |
| speaker | 2 | 0.11% |
| filming location | 2 | 0.11% |
| office held by head of the organization | 2 | 0.11% |
| track gauge | 2 | 0.11% |
| archives at | 2 | 0.11% |
| ammunition | 2 | 0.11% |
| published in | 2 | 0.11% |
| canonization status | 2 | 0.11% |
| convicted of | 2 | 0.11% |
| studied by | 2 | 0.11% |
| biological process | 2 | 0.11% |
| defendant | 2 | 0.11% |
| gens | 2 | 0.11% |
| connecting line | 2 | 0.11% |
| religion or worldview | 2 | 0.11% |
| genetic association | 2 | 0.11% |
| recommended unit of measurement | 2 | 0.11% |
| significant place | 2 | 0.11% |
| Gram staining | 2 | 0.11% |
| choreographer | 2 | 0.11% |
| director of photography | 2 | 0.11% |
| participating team | 2 | 0.11% |
| part of | 2 | 0.11% |
| enemy of | 2 | 0.11% |
| source of energy | 2 | 0.11% |
| final event | 2 | 0.11% |
| is the study of | 2 | 0.11% |
| residence | 2 | 0.11% |
| historical region | 2 | 0.11% |
| candidate | 2 | 0.11% |
| radio format | 2 | 0.11% |
| official language | 2 | 0.11% |
| political ideology | 2 | 0.11% |
| represents | 2 | 0.11% |
| follows | 2 | 0.11% |
| business division | 2 | 0.11% |
| executive body | 2 | 0.11% |
| medical examination | 2 | 0.11% |
| original language of film or TV show | 2 | 0.11% |
| twinned administrative body | 2 | 0.11% |
| derivative work | 2 | 0.11% |
| practiced by | 2 | 0.11% |
| director of archaeological fieldwork | 2 | 0.11% |
| powered by | 2 | 0.11% |
| mother | 2 | 0.11% |
| encodes | 2 | 0.11% |
| linguistic typology | 2 | 0.11% |
| social classification | 2 | 0.11% |
| followed by | 2 | 0.11% |
| readable file format | 2 | 0.11% |
| subclass of | 2 | 0.11% |
| performed at | 2 | 0.11% |
| competition class | 2 | 0.11% |
| copyright holder | 2 | 0.11% |
| chief executive officer | 2 | 0.11% |
| official religion | 2 | 0.11% |
| manufacturer | 2 | 0.11% |
| location of landing | 2 | 0.11% |
| designed to carry | 2 | 0.11% |
| measurement scale | 2 | 0.11% |
| has subsidiary | 2 | 0.11% |
| given name version for other gender | 2 | 0.11% |
| youth wing | 2 | 0.11% |
| tributary | 2 | 0.11% |
| board member | 2 | 0.11% |
| incarnation of | 2 | 0.11% |
| takes place in fictional universe | 2 | 0.11% |
| domain of saint or deity | 2 | 0.11% |
| produced by | 2 | 0.11% |
| programming language | 2 | 0.11% |
| media franchise | 2 | 0.11% |
| territory overlaps | 2 | 0.11% |
| tracklist | 2 | 0.11% |
| significant person | 2 | 0.11% |
| religious order | 2 | 0.11% |
| dedicated to | 2 | 0.11% |
| applies to jurisdiction | 2 | 0.11% |
| contributor to the creative work or subject | 2 | 0.11% |
| allegiance | 2 | 0.11% |
| associated electoral district | 2 | 0.11% |
| instruction set | 2 | 0.11% |
| headquarters location | 2 | 0.11% |
| standards body | 2 | 0.11% |
| wheel arrangement | 2 | 0.11% |
| office held by head of state | 2 | 0.11% |
| structure replaced by | 1 | 0.06% |
| aspect ratio (W:H) | 1 | 0.06% |
| product certification | 1 | 0.06% |
| relative | 1 | 0.06% |
| catalog | 1 | 0.06% |
| shape | 1 | 0.06% |
| writable file format | 1 | 0.06% |
| taxon rank | 1 | 0.06% |
| anthem | 1 | 0.06% |
| CPU | 1 | 0.06% |
| flag | 1 | 0.06% |
| afflicts | 1 | 0.06% |
| central bank | 1 | 0.06% |
| Christian liturgical rite | 1 | 0.06% |
| site of astronomical discovery | 1 | 0.06% |
| sexual orientation | 1 | 0.06% |
| place of burial | 1 | 0.06% |
| trophy awarded | 1 | 0.06% |
| screenwriter | 1 | 0.06% |
| organization directed by the office or position | 1 | 0.06% |
| measures | 1 | 0.06% |
| commemorates | 1 | 0.06% |
| next higher rank | 1 | 0.06% |
| office held by head of government | 1 | 0.06% |
| vehicle normally used | 1 | 0.06% |
| merged into | 1 | 0.06% |
| origin of the watercourse | 1 | 0.06% |
| league | 1 | 0.06% |
| permanent resident of | 1 | 0.06% |
| statement describes | 1 | 0.06% |
| lifestyle | 1 | 0.06% |
| parent club | 1 | 0.06% |
| set during recurring event | 1 | 0.06% |
| court | 1 | 0.06% |
| family name identical to this given name | 1 | 0.06% |
| game mode | 1 | 0.06% |
| lyrics by | 1 | 0.06% |
| copyright license | 1 | 0.06% |
| advertises | 1 | 0.06% |
| software engine | 1 | 0.06% |
| military casualty classification | 1 | 0.06% |
| next lower rank | 1 | 0.06% |
| color | 1 | 0.06% |
| architectural style | 1 | 0.06% |
| active ingredient in | 1 | 0.06% |
| diaspora | 1 | 0.06% |
| contributing factor of | 1 | 0.06% |
| programming paradigm | 1 | 0.06% |
| anatomical branch of | 1 | 0.06% |
| dual to | 1 | 0.06% |
| general classification of race participants | 1 | 0.06% |
| depicts | 1 | 0.06% |
| radix | 1 | 0.06% |
| input method | 1 | 0.06% |
| members have occupation | 1 | 0.06% |
| sports season of league or competition | 1 | 0.06% |
| source of income | 1 | 0.06% |
| Drug_Interaction | 1 | 0.06% |
| editor-in-chief | 1 | 0.06% |
| theme music | 1 | 0.06% |
| conjugate acid | 1 | 0.06% |
| parallel street | 1 | 0.06% |
| prime factor | 1 | 0.06% |
| official podcast | 1 | 0.06% |
| patron saint | 1 | 0.06% |
| executive producer | 1 | 0.06% |
| capital of | 1 | 0.06% |
| temporal range start | 1 | 0.06% |
| health specialty | 1 | 0.06% |
| this taxon is source of | 1 | 0.06% |
| organizer | 1 | 0.06% |
| disease transmission process | 1 | 0.06% |
| central bank/issuer | 1 | 0.06% |
| depicted by | 1 | 0.06% |
| copyright status | 1 | 0.06% |
| coat of arms | 1 | 0.06% |
| family name | 1 | 0.06% |
| country of registry | 1 | 0.06% |
| killed by | 1 | 0.06% |
| house publication | 1 | 0.06% |
| penalty | 1 | 0.06% |
| appointed by | 1 | 0.06% |
| has written for | 1 | 0.06% |
| structural engineer | 1 | 0.06% |
| home world | 1 | 0.06% |
| affiliation | 1 | 0.06% |
| sexually homologous with | 1 | 0.06% |
| in opposition to | 1 | 0.06% |
| league level below | 1 | 0.06% |
| sports league level | 1 | 0.06% |
| mod of | 1 | 0.06% |
| voice actor | 1 | 0.06% |
| cause of death | 1 | 0.06% |
| stepparent | 1 | 0.06% |
| route of administration | 1 | 0.06% |
| unmarried partner | 1 | 0.06% |
| mount | 1 | 0.06% |
| does not have part | 1 | 0.06% |
| lake outflow | 1 | 0.06% |
| implementation of | 1 | 0.06% |
| territory claimed by | 1 | 0.06% |
| writing system | 1 | 0.06% |
| doctoral student | 1 | 0.06% |
| economy of topic | 1 | 0.06% |
| indigenous to | 1 | 0.06% |
| business model | 1 | 0.06% |
| student | 1 | 0.06% |
| inspired by | 1 | 0.06% |
| amended by | 1 | 0.06% |
| wing configuration | 1 | 0.06% |
| cuisine | 1 | 0.06% |
| launch contractor | 1 | 0.06% |
| character designer | 1 | 0.06% |
| sports discipline competed in | 1 | 0.06% |
| typically sells | 1 | 0.06% |
| delta of | 1 | 0.06% |
| depends on software | 1 | 0.06% |
| display technology | 1 | 0.06% |
| has active ingredient | 1 | 0.06% |
| less than | 1 | 0.06% |
| main building contractor | 1 | 0.06% |
| substrate of | 1 | 0.06% |
| measured physical quantity | 1 | 0.06% |
| day in year for periodic occurrence | 1 | 0.06% |
| drainage basin | 1 | 0.06% |
| airline alliance | 1 | 0.06% |
| exemplar of | 1 | 0.06% |
| model item | 1 | 0.06% |
| destroyed | 1 | 0.06% |
| cell component | 1 | 0.06% |
| signatory | 1 | 0.06% |
| sex or gender | 1 | 0.06% |
| designated as terrorist by | 1 | 0.06% |
| capital | 1 | 0.06% |
| subdivision of this unit | 1 | 0.06% |
| points classification | 1 | 0.06% |
| sidekick of | 1 | 0.06% |
| parent of this hybrid, breed, or cultivar | 1 | 0.06% |
| found in taxon | 1 | 0.06% |

### Classifications

**train, biotech_event.** Majority class 31.7% of 167 answers. The marginal is `true_label`, not the menu.

#### train biotech_event

| label | count | share |
| --- | --- | --- |
| executive statement | 53 | 31.74% |
| company description | 28 | 16.77% |
| alliance & partnership | 13 | 7.78% |
| product launching & presentation | 10 | 5.99% |
| other | 8 | 4.79% |
| new initiatives or programs | 7 | 4.19% |
| hiring | 7 | 4.19% |
| m&a | 6 | 3.59% |
| funding round | 5 | 2.99% |
| executive appointment | 5 | 2.99% |
| expanding geography | 4 | 2.40% |
| regulatory approval | 3 | 1.80% |
| support & philanthropy | 3 | 1.80% |
| service & product providing | 3 | 1.80% |
| product updates | 3 | 1.80% |
| article publication | 2 | 1.20% |
| new initiatives & programs | 2 | 1.20% |
| clinical trial sponsorship | 2 | 1.20% |
| expanding industry | 1 | 0.60% |
| event organization | 1 | 0.60% |
| participation in an event | 1 | 0.60% |

**train, chfinann_event.** Majority class 50.1% of 1,046 answers. The marginal is `true_label`, not the menu.

#### train chfinann_event

| label | count | share |
| --- | --- | --- |
| EquityPledge | 524 | 50.10% |
| EquityUnderweight | 209 | 19.98% |
| EquityOverweight | 206 | 19.69% |
| EquityRepurchase | 65 | 6.21% |
| EquityFreeze | 42 | 4.02% |

**train, docee_event.** Majority class 6.7% of 883 answers. The marginal is `true_label`, not the menu.

#### train docee_event

| label | count | share |
| --- | --- | --- |
| Armed Conflict | 59 | 6.68% |
| Air Crash | 54 | 6.12% |
| Protest | 44 | 4.98% |
| Election | 41 | 4.64% |
| Fire | 35 | 3.96% |
| Government Policy Changes | 30 | 3.40% |
| CommitCrime - Accuse | 29 | 3.28% |
| CommitCrime - Sentence | 28 | 3.17% |
| Riot | 28 | 3.17% |
| Appoint_Inauguration | 22 | 2.49% |
| Disease Outbreaks | 21 | 2.38% |
| Resignation_Dismissal | 21 | 2.38% |
| Famous Person - Death | 20 | 2.27% |
| Sports Competition | 20 | 2.27% |
| Earthquakes | 20 | 2.27% |
| Storm | 17 | 1.93% |
| Withdraw from an Orgnization | 17 | 1.93% |
| Gas Explosion | 16 | 1.81% |
| Mass Poisoning | 15 | 1.70% |
| Environment Pollution | 14 | 1.59% |
| Organization Merge | 14 | 1.59% |
| Break Historical Records | 13 | 1.47% |
| Floods | 13 | 1.47% |
| Strike | 12 | 1.36% |
| CommitCrime - Investigate | 12 | 1.36% |
| Famous Person - Give a Speech | 12 | 1.36% |
| Shipwreck | 12 | 1.36% |
| Diplomatic Talks | 11 | 1.25% |
| New Wonders in Nature | 11 | 1.25% |
| CommitCrime - Release | 11 | 1.25% |
| New Archeological Discoveries | 11 | 1.25% |
| Famine | 11 | 1.25% |
| Military Excercise | 11 | 1.25% |
| Train Collisions | 11 | 1.25% |
| Insect Disaster | 10 | 1.13% |
| Tear Up Agreement | 10 | 1.13% |
| Famous Person - Sick | 9 | 1.02% |
| Road Crash | 9 | 1.02% |
| Bank Robbery | 9 | 1.02% |
| Mudslides | 9 | 1.02% |
| CommitCrime - Arrest | 9 | 1.02% |
| Awards Ceremony | 9 | 1.02% |
| Volcano Eruption | 8 | 0.91% |
| Mine Collapses | 8 | 0.91% |
| Diplomatic Visit | 8 | 0.91% |
| Tsunamis | 7 | 0.79% |
| Famous Person - Divorce | 7 | 0.79% |
| Organization Closed | 7 | 0.79% |
| Organization Established | 7 | 0.79% |
| New Achievements in Aerospace | 7 | 0.79% |
| Sign Agreement | 6 | 0.68% |
| Famous Person - Marriage | 6 | 0.68% |
| Financial Crisis | 6 | 0.68% |
| Droughts | 5 | 0.57% |
| Famous Person - Recovered | 4 | 0.45% |
| Financial Aid | 3 | 0.34% |
| Regime Change | 2 | 0.23% |
| Organization Fine | 2 | 0.23% |

**train, docfee_event.** Majority class 35.8% of 679 answers. The marginal is `true_label`, not the menu.

#### train docfee_event

| label | count | share |
| --- | --- | --- |
| EquityUnderweight | 243 | 35.79% |
| BankruptcyLiquidation | 108 | 15.91% |
| EquityPledge | 106 | 15.61% |
| MajorAssetLoss | 83 | 12.22% |
| EquityFreeze | 48 | 7.07% |
| EquityOverweight | 36 | 5.30% |
| MajorSafetyAccident | 20 | 2.95% |
| ExecutiveDeath | 18 | 2.65% |
| MajorExternalPayout | 17 | 2.50% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| casualty_report | injured | 14,084 | 58.99% |
| casualty_report | missing | 9,154 | 38.34% |
| casualty_report | dead | 8,413 | 35.24% |
| casualty_report | record_metadata | natural/dead ×6,851, natural/injured ×4,606, natural/missing ×1,234 |  |
| record | status | 39 | 14.39% |
| record | location | 26 | 9.59% |
| record | name | 17 | 6.27% |
| record | address | 17 | 6.27% |
| record | title | 13 | 4.80% |
| record | price | 11 | 4.06% |
| record | rating | 8 | 2.95% |
| record | amount | 8 | 2.95% |
| record | author | 7 | 2.58% |
| record | city | 7 | 2.58% |
| record | description | 6 | 2.21% |
| record | date | 5 | 1.85% |
| record | policy_id | 5 | 1.85% |
| record | review_id | 5 | 1.85% |
| record | funding_amount | 5 | 1.85% |
| record | year | 5 | 1.85% |
| record | department | 5 | 1.85% |
| record | claim_id | 4 | 1.48% |
| record | project_id | 4 | 1.48% |
| record | state | 4 | 1.48% |
| record | employee_id | 4 | 1.48% |
| record | company | 4 | 1.48% |
| record | sender | 4 | 1.48% |
| record | venue | 4 | 1.48% |
| record | event_id | 4 | 1.48% |
| record | patient_name | 4 | 1.48% |
| record | amenities | 3 | 1.11% |
| record | policyholder | 3 | 1.11% |
| record | product_id | 3 | 1.11% |
| record | brand | 3 | 1.11% |
| record | submitter | 3 | 1.11% |
| record | company_name | 3 | 1.11% |
| record | manufacturer | 3 | 1.11% |
| record | temperature: Body temperature in degrees Fahrenheit | 3 | 1.11% |
| record | ticket_id | 3 | 1.11% |
| record | patient_id | 3 | 1.11% |
| record | term_length | 3 | 1.11% |
| record | medication | 3 | 1.11% |
| record | age | 3 | 1.11% |
| record | transaction_id | 3 | 1.11% |
| record | location.region | 2 | 0.74% |
| record | location.country | 2 | 0.74% |
| record | expression_level | 2 | 0.74% |
| record | features | 2 | 0.74% |
| record | asteroid.name | 2 | 0.74% |
| record | product_name | 2 | 0.74% |
| record | reporter | 2 | 0.74% |
| record | timestamp | 2 | 0.74% |
| record | project_name | 2 | 0.74% |
| record | net_profit | 2 | 0.74% |
| record | report_id | 2 | 0.74% |
| record | away_team | 2 | 0.74% |
| record | likes | 2 | 0.74% |
| record | shares | 2 | 0.74% |
| record | device | 2 | 0.74% |
| record | education | 2 | 0.74% |
| record | role | 2 | 0.74% |
| record | full_name | 2 | 0.74% |
| record | clause_id | 2 | 0.74% |
| record | severity | 2 | 0.74% |
| record | gender | 2 | 0.74% |
| record | home_team | 2 | 0.74% |
| record | team | 2 | 0.74% |
| record | points | 2 | 0.74% |
| record | resolution_time | 2 | 0.74% |
| record | policyholder_name | 2 | 0.74% |
| record | patient_id: Unique identifier for the patient | 2 | 0.74% |
| record | heart_rate: Heart rate in beats per minute | 2 | 0.74% |
| record | home_team: Name of the team playing at home | 2 | 0.74% |
| record | funding_source | 2 | 0.74% |
| record | duration | 2 | 0.74% |
| record | investor | 2 | 0.74% |
| record | review_status | 2 | 0.74% |
| record | film_title | 2 | 0.74% |
| record | tail | 2 | 0.74% |
| record | head | 2 | 0.74% |
| record | opponent | 2 | 0.74% |
| record | customer_name | 2 | 0.74% |
| record | user | 2 | 0.74% |
| record | issue_date | 2 | 0.74% |
| record | monthly_premium | 2 | 0.74% |
| record | patient_address | 2 | 0.74% |
| record | funding | 2 | 0.74% |
| record | score | 2 | 0.74% |
| record | product | 2 | 0.74% |
| record | return_id: Unique identifier for the return transaction | 2 | 0.74% |
| record | appointment_id | 2 | 0.74% |
| record | cpt_code | 2 | 0.74% |
| record | status: Current status of the appointment | 2 | 0.74% |
| record | quantity | 2 | 0.74% |
| record | country | 2 | 0.74% |
| record | products | 2 | 0.74% |
| record | dosage | 2 | 0.74% |
| record | beneficiary | 2 | 0.74% |
| record | recipient | 2 | 0.74% |
| record | category | 2 | 0.74% |
| record | coverage_amount | 2 | 0.74% |
| record | hotel_details.star_rating | 1 | 0.37% |
| record | ranking.location | 1 | 0.37% |
| record | hotel_details.location | 1 | 0.37% |
| record | booking_options.partners | 1 | 0.37% |
| record | booking_options.description | 1 | 0.37% |
| record | price_range.currency | 1 | 0.37% |
| record | hotel_details.name | 1 | 0.37% |
| record | transcription_factor | 1 | 0.37% |
| record | gene_id | 1 | 0.37% |
| record | tissue | 1 | 0.37% |
| record | network_id | 1 | 0.37% |
| record | expression_units | 1 | 0.37% |
| record | network_context | 1 | 0.37% |
| record | ingredients | 1 | 0.37% |
| record | cooking_methods | 1 | 0.37% |
| record | boiling_instructions.salt_advice | 1 | 0.37% |
| record | boiling_instructions.cooking_time.min_minutes | 1 | 0.37% |
| record | boiling_instructions.cooking_time.max_minutes | 1 | 0.37% |
| record | grilling_instructions.method | 1 | 0.37% |
| record | grilling_instructions.cooking_time.minutes | 1 | 0.37% |
| record | serving_suggestions | 1 | 0.37% |
| record | personal_experience.raw_corn_experience | 1 | 0.37% |
| record | vehicle | 1 | 0.37% |
| record | vin | 1 | 0.37% |
| record | seller | 1 | 0.37% |
| record | date_added | 1 | 0.37% |
| record | tags | 1 | 0.37% |
| record | thumbnail | 1 | 0.37% |
| record | asteroid.dedication.person.nationality | 1 | 0.37% |
| record | asteroid.location | 1 | 0.37% |
| record | asteroid.dedication.person.occupation | 1 | 0.37% |
| record | key_features | 1 | 0.37% |
| record | issues | 1 | 0.37% |
| record | boulder.start.foot_holds.1.foot | 1 | 0.37% |
| record | boulder.start.foot_holds.0.foot | 1 | 0.37% |
| record | boulder.start.hand_holds.1.description | 1 | 0.37% |
| record | boulder.start.hand_holds.0.description | 1 | 0.37% |
| record | boulder.start.foot_holds.0.description | 1 | 0.37% |
| record | boulder.start.hand_holds.1.hand | 1 | 0.37% |
| record | boulder.location.relative_position | 1 | 0.37% |
| record | boulder.location.distance_unit | 1 | 0.37% |
| record | boulder.location.reference_point | 1 | 0.37% |
| record | office | 1 | 0.37% |
| record | update_notes | 1 | 0.37% |
| record | passenger_name: Name of the passenger on the flight | 1 | 0.37% |
| record | flight_no: Flight number as assigned by the airline | 1 | 0.37% |
| record | delay_minutes: Number of minutes flight is delayed | 1 | 0.37% |
| record | destination: Airport code where flight ends | 1 | 0.37% |
| record | email: Email address of the passenger | 1 | 0.37% |
| record | unique_id: Unique identifier for the delay entry | 1 | 0.37% |
| record | departure_date: Date flight departs from origin | 1 | 0.37% |
| record | renewal_amount | 1 | 0.37% |
| record | population.count | 1 | 0.37% |
| record | geography.coordinates.latitude | 1 | 0.37% |
| record | geography.coordinates.longitude | 1 | 0.37% |
| record | census_inclusion | 1 | 0.37% |
| record | claim_id: Unique identifier for each claim | 1 | 0.37% |
| record | patient_name: Name of the patient filing the claim | 1 | 0.37% |
| record | patient_address: Full address including street, apt, city, state, zip | 1 | 0.37% |
| record | diagnosis_code: Standard code indicating the patient's condition | 1 | 0.37% |
| record | treatment_type: Type of medical treatment or procedure | 1 | 0.37% |
| record | claim_amount: Total monetary value of the claim | 1 | 0.37% |
| record | insurance_provider: Name of the insurance company covering the claim | 1 | 0.37% |
| record | status: Current processing status of the claim | 1 | 0.37% |
| record | payment_method: Method used to process payment for the claim | 1 | 0.37% |
| record | deductiblestatus: Whether patient's deductible has been met | 1 | 0.37% |
| record | co_payment: Patient's required co-payment amount | 1 | 0.37% |
| record | notes: Additional remarks or clinical context about the claim | 1 | 0.37% |
| record | finding_id | 1 | 0.37% |
| record | time | 1 | 0.37% |
| record | access_duration | 1 | 0.37% |
| record | access_type | 1 | 0.37% |
| record | data_code | 1 | 0.37% |
| record | review_type | 1 | 0.37% |
| record | review_score | 1 | 0.37% |
| record | github_link | 1 | 0.37% |
| record | claim_id: Unique identifier for each claim entry | 1 | 0.37% |
| record | policyholder_name: Name of the policyholder filing the claim | 1 | 0.37% |
| record | address: Full address of the policyholder | 1 | 0.37% |
| record | claim_date: Date when the incident occurred | 1 | 0.37% |
| record | amount_requested: Total amount requested for compensation | 1 | 0.37% |
| record | status: Current status of the claim | 1 | 0.37% |
| record | description: Brief description of the incident and damages | 1 | 0.37% |
| record | region_churn | 1 | 0.37% |
| record | call_id | 1 | 0.37% |
| record | product_offered | 1 | 0.37% |
| record | other_youth_programs_with_congressional_charters | 1 | 0.37% |
| record | spoofed_domain | 1 | 0.37% |
| record | target_department | 1 | 0.37% |
| record | risk_classification | 1 | 0.37% |
| record | incident_type | 1 | 0.37% |
| record | disruption_duration | 1 | 0.37% |
| record | weather_conditions | 1 | 0.37% |
| record | league | 1 | 0.37% |
| record | comments | 1 | 0.37% |
| record | creator | 1 | 0.37% |
| record | app_name | 1 | 0.37% |
| record | prior_experience | 1 | 0.37% |
| record | magazine.article.page_breakdown.2.page_number | 1 | 0.37% |
| record | magazine.article.page_breakdown.9.advertising_percentage | 1 | 0.37% |
| record | magazine.article.page_breakdown.12.page_number | 1 | 0.37% |
| record | magazine.article.page_breakdown.2.advertising_percentage | 1 | 0.37% |
| record | magazine.article.page_breakdown.9.page_number | 1 | 0.37% |
| record | magazine.article.page_breakdown.3.advertisers | 1 | 0.37% |
| record | magazine.article.page_breakdown.16.advertising_percentage | 1 | 0.37% |
| record | magazine.article.page_breakdown.13.advertisers | 1 | 0.37% |
| record | magazine.article.page_breakdown.12.advertising_percentage | 1 | 0.37% |
| record | magazine.article.page_breakdown.5.advertising_percentage | 1 | 0.37% |
| record | magazine.article.page_breakdown.5.page_number | 1 | 0.37% |
| record | magazine.article.page_breakdown.14.advertisers | 1 | 0.37% |
| record | magazine.article.page_breakdown.13.advertising_percentage | 1 | 0.37% |
| record | magazine.article.page_breakdown.1.advertising_percentage | 1 | 0.37% |
| record | magazine.article.page_breakdown.8.advertisers | 1 | 0.37% |
| record | magazine.mentioned_article.author | 1 | 0.37% |
| record | magazine.article.page_breakdown.7.page_number | 1 | 0.37% |
| record | magazine.article.page_breakdown.10.page_number | 1 | 0.37% |
| record | magazine.name | 1 | 0.37% |
| record | magazine.article.page_breakdown.6.advertisers | 1 | 0.37% |
| record | magazine.article.page_breakdown.0.page_number | 1 | 0.37% |
| record | magazine.article.page_breakdown.2.advertisers | 1 | 0.37% |
| record | magazine.article.page_breakdown.5.advertisers | 1 | 0.37% |
| record | magazine.article.page_breakdown.15.advertisers | 1 | 0.37% |
| record | magazine.article.page_breakdown.4.page_number | 1 | 0.37% |
| record | magazine.article.page_breakdown.9.advertisers | 1 | 0.37% |
| record | magazine.article.page_breakdown.16.page_number | 1 | 0.37% |
| record | magazine.article.page_breakdown.10.advertisers | 1 | 0.37% |
| record | magazine.article.title | 1 | 0.37% |
| record | magazine.article.content_percentage | 1 | 0.37% |
| record | magazine.mentioned_article.year | 1 | 0.37% |
| record | magazine.article.page_breakdown.13.page_number | 1 | 0.37% |
| record | magazine.article.page_breakdown.3.page_number | 1 | 0.37% |
| record | magazine.article.page_breakdown.8.advertising_percentage | 1 | 0.37% |
| record | magazine.article.page_breakdown.0.advertisers | 1 | 0.37% |
| record | magazine.article.advertising_percentage | 1 | 0.37% |
| record | magazine.article.page_breakdown.14.advertising_percentage | 1 | 0.37% |
| record | magazine.article.page_breakdown.8.page_number | 1 | 0.37% |
| record | keywords | 1 | 0.37% |
| record | publication_time | 1 | 0.37% |
| record | claim_id: Unique identifier for the claim as referenced in correspondence | 1 | 0.37% |
| record | source_office: Clinic or office that submitted the claim | 1 | 0.37% |
| record | procedure_code: CPT code describing the medical service performed | 1 | 0.37% |
| record | template_substitution.prefix | 1 | 0.37% |
| record | advantages | 1 | 0.37% |
| record | template_substitution.examples | 1 | 0.37% |
| record | template_substitution.preview_code_example | 1 | 0.37% |
| record | templates_to_substitute | 1 | 0.37% |
| record | template_types.ordinary_template | 1 | 0.37% |
| record | format | 1 | 0.37% |
| record | issue_description | 1 | 0.37% |
| record | dates | 1 | 0.37% |
| record | registration_cap | 1 | 0.37% |
| record | registration_fee | 1 | 0.37% |
| record | early_bird_deadline | 1 | 0.37% |
| record | panel_topics | 1 | 0.37% |
| record | outcomes_and_findings.prognosis.critical_success_factors | 1 | 0.37% |
| record | study_details.cohort.condition_studied | 1 | 0.37% |
| record | pathophysiology.etiology | 1 | 0.37% |
| record | pathophysiology.mechanism.secondary_event.anatomical_locations | 1 | 0.37% |
| record | outcomes_and_findings.prognosis.visual_recovery_potential | 1 | 0.37% |
| record | shipping_cost: Cost to ship the audio device in US dollars | 1 | 0.37% |
| record | material: Material used in device construction | 1 | 0.37% |
| record | weight: Weight of the device in grams | 1 | 0.37% |
| record | discovery_location | 1 | 0.37% |
| record | asteroid_number | 1 | 0.37% |
| record | discovery_date | 1 | 0.37% |
| record | discoverers | 1 | 0.37% |
| record | physical_characteristics.observations.0.telescope | 1 | 0.37% |
| record | designation | 1 | 0.37% |
| record | orbital_characteristics.classification | 1 | 0.37% |
| record | discovery.discoverer | 1 | 0.37% |
| record | discovery.observatory.location.prefecture | 1 | 0.37% |
| record | physical_characteristics.observations.0.year | 1 | 0.37% |
| record | discovery.observatory.location.country | 1 | 0.37% |
| record | orbital_characteristics.semi_major_axis.unit | 1 | 0.37% |
| record | orbital_characteristics.perihelion.unit | 1 | 0.37% |
| record | unit_id | 1 | 0.37% |
| record | github_url | 1 | 0.37% |
| record | policyholder_id: Unique identifier for the policyholder | 1 | 0.37% |
| record | policyholder_name: Full name of the claimant | 1 | 0.37% |
| record | claim_id: Unique identifier for the submitted claim | 1 | 0.37% |
| record | symbol: Unique ticker identifier for the company | 1 | 0.37% |
| record | price: Closing stock price in dollars | 1 | 0.37% |
| record | sector: Industry sector classification | 1 | 0.37% |
| record | status: Direction of price movement | 1 | 0.37% |
| record | post_date | 1 | 0.37% |
| record | user_handle | 1 | 0.37% |
| record | circular.subject | 1 | 0.37% |
| record | circular.ministry | 1 | 0.37% |
| record | labels | 1 | 0.37% |
| record | species_name | 1 | 0.37% |
| record | family.name | 1 | 0.37% |
| record | described_by.year | 1 | 0.37% |
| record | consultation.practice_areas | 1 | 0.37% |
| record | practice_areas | 1 | 0.37% |
| record | states_served | 1 | 0.37% |
| record | services | 1 | 0.37% |
| record | firm_name | 1 | 0.37% |
| record | client_focus | 1 | 0.37% |
| record | additional_services | 1 | 0.37% |
| record | experience.years | 1 | 0.37% |
| record | experience.description | 1 | 0.37% |
| record | restaurant_name | 1 | 0.37% |
| record | count | 1 | 0.37% |
| record | hotel_experience.location.nearby_landmark | 1 | 0.37% |
| record | hotel_experience.location.city | 1 | 0.37% |
| record | hotel_experience.entertainment.tv.content | 1 | 0.37% |
| record | hotel_experience.additional_benefits | 1 | 0.37% |
| record | hotel_experience.suites.0.workspace | 1 | 0.37% |
| record | hotel_experience.suites.0.suite_type | 1 | 0.37% |
| record | hotel_experience.suites.0.bathroom_features | 1 | 0.37% |
| record | hotel_experience.suites.0.amenities | 1 | 0.37% |
| record | hotel_experience.suites.0.features | 1 | 0.37% |
| record | hotel_experience.entertainment.tv.screen_size | 1 | 0.37% |
| record | headquarters: City and country where company is headquartered | 1 | 0.37% |
| record | status: Current operational or strategic status | 1 | 0.37% |
| record | launch_year: Year the company launched its platform | 1 | 0.37% |
| record | post_conference_event | 1 | 0.37% |
| record | student_rate | 1 | 0.37% |
| record | early_bird_discount_ends | 1 | 0.37% |
| record | current_registration_count | 1 | 0.37% |
| record | keynote_sessions | 1 | 0.37% |
| record | panel_discussions | 1 | 0.37% |
| record | tracks | 1 | 0.37% |
| record | core_values | 1 | 0.37% |
| record | registration_type | 1 | 0.37% |
| record | workshop_date | 1 | 0.37% |
| record | backup_plan | 1 | 0.37% |
| record | pre_conference_workshop | 1 | 0.37% |
| record | venue_capacity | 1 | 0.37% |
| record | event_status | 1 | 0.37% |
| record | lightning_talks | 1 | 0.37% |
| record | expected_revenue | 1 | 0.37% |
| record | senior_rate | 1 | 0.37% |
| record | compliance | 1 | 0.37% |
| record | statue.name.french | 1 | 0.37% |
| record | statue.restoration.0.description | 1 | 0.37% |
| record | statue.restoration.0.funding.amount_raised.currency | 1 | 0.37% |
| record | statue.restoration.0.funding.source | 1 | 0.37% |
| record | statue.restoration.0.funding.contribution_range.currency | 1 | 0.37% |
| record | statue.discovery.discoverer.name | 1 | 0.37% |
| record | statue.discovery.location | 1 | 0.37% |
| record | statue.restoration.0.funding.contribution_range.min | 1 | 0.37% |
| record | statue.restoration.0.budget.currency | 1 | 0.37% |
| record | statue.discovery.date | 1 | 0.37% |
| record | statue.restoration.0.start_date | 1 | 0.37% |
| record | statue.transport_to_france | 1 | 0.37% |
| record | statue.public_display.stairs_completion_date | 1 | 0.37% |
| record | statue.location.museum | 1 | 0.37% |
| record | statue.material | 1 | 0.37% |
| record | statue.public_display.reopening_date | 1 | 0.37% |
| record | statue.location.city | 1 | 0.37% |
| record | statue.discovery.discoverer.title | 1 | 0.37% |
| record | statue.restoration.0.funding.number_of_contributors | 1 | 0.37% |
| record | statue.name.greek | 1 | 0.37% |
| record | statue.dedication | 1 | 0.37% |
| record | sunless_tanning.product_collection.products.0.type | 1 | 0.37% |
| record | sunless_tanning.product_collection.products.1.type | 1 | 0.37% |
| record | sunless_tanning.product_collection.products.0.ingredients | 1 | 0.37% |
| record | sunless_tanning.product_collection.products.0.description | 1 | 0.37% |
| record | sunless_tanning.product_collection.brand | 1 | 0.37% |
| record | sunless_tanning.product_collection.products.0.benefits | 1 | 0.37% |
| record | sunless_tanning.product_collection.benefits | 1 | 0.37% |
| record | sunless_tanning.product_collection.products.0.sizes | 1 | 0.37% |
| record | sunless_tanning.product_collection.products.1.name | 1 | 0.37% |
| record | sunless_tanning.product_collection.products.1.description | 1 | 0.37% |
| record | sunless_tanning.concerns | 1 | 0.37% |
| record | incident_id | 1 | 0.37% |
| record | date_reported | 1 | 0.37% |
| record | responsible_engineer | 1 | 0.37% |
| record | environment | 1 | 0.37% |
| record | region | 1 | 0.37% |
| record | production_location | 1 | 0.37% |
| record | treatment_arm | 1 | 0.37% |
| record | enrollment_date | 1 | 0.37% |
| record | contact_number | 1 | 0.37% |
| record | lastfollowup | 1 | 0.37% |
| record | insurance_provider | 1 | 0.37% |
| record | useoffunds: Purpose or application of raised capital | 1 | 0.37% |
| record | funding_amount: Total amount raised in millions | 1 | 0.37% |
| record | funding_id: Unique identifier assigned to the funding round | 1 | 0.37% |
| record | company: Name of the technology company receiving funds | 1 | 0.37% |
| record | party_name | 1 | 0.37% |
| record | launch_date | 1 | 0.37% |
| record | target_market | 1 | 0.37% |
| record | visitor_team | 1 | 0.37% |
| record | investor_leads | 1 | 0.37% |
| record | funds_use | 1 | 0.37% |
| record | open_date | 1 | 0.37% |
| record | ticket_count | 1 | 0.37% |
| record | customer_id | 1 | 0.37% |
| record | patent_number | 1 | 0.37% |
| record | zip | 1 | 0.37% |
| record | dateofincident | 1 | 0.37% |
| record | search_result_page.search_query | 1 | 0.37% |
| record | search_result_page.website_name | 1 | 0.37% |
| record | initial_text.negations | 1 | 0.37% |
| record | id: Unique identifier for the legislative vote | 1 | 0.37% |
| record | date: Date when the vote occurred | 1 | 0.37% |
| record | chamber: Legislative chamber where the vote took place | 1 | 0.37% |
| record | bill: Bill number and type identifier | 1 | 0.37% |
| record | sponsor: Name of the bill's sponsor | 1 | 0.37% |
| record | outcome: Final result of the vote | 1 | 0.37% |
| record | amount: Financial amount associated with the bill | 1 | 0.37% |
| record | location: Geographic location of the bill's jurisdiction | 1 | 0.37% |
| record | status: Current status of the bill | 1 | 0.37% |
| record | description: Summary of the bill's purpose and scope | 1 | 0.37% |
| record | start_term | 1 | 0.37% |
| record | credits | 1 | 0.37% |
| record | hall | 1 | 0.37% |
| record | course_code | 1 | 0.37% |
| record | room | 1 | 0.37% |
| record | enrollment | 1 | 0.37% |
| record | email_id | 1 | 0.37% |
| record | encounter_id: Unique identifier for the visit | 1 | 0.37% |
| record | name: Full name of the patient | 1 | 0.37% |
| record | address: Patient's residential address | 1 | 0.37% |
| record | visit_date: Date of the medical encounter | 1 | 0.37% |
| record | primary_diagnosis: Main medical condition diagnosed | 1 | 0.37% |
| record | icd_code: ICD-10 code for the primary diagnosis | 1 | 0.37% |
| record | bp_systolic: Systolic blood pressure in mmHg | 1 | 0.37% |
| record | bp_diastolic: Diastolic blood pressure in mmHg | 1 | 0.37% |
| record | medications: List of prescribed medications with dosing | 1 | 0.37% |
| record | lab_results: List of lab test results | 1 | 0.37% |
| record | symptoms: Patient-reported symptoms or complaints | 1 | 0.37% |
| record | guideline_id: Unique identifier for the guideline document | 1 | 0.37% |
| record | author: Name of the medical professional who authored the guideline | 1 | 0.37% |
| record | institution: Hospital or organization where author is affiliated | 1 | 0.37% |
| record | recommendation: Specific treatment or intervention advised | 1 | 0.37% |
| record | issue_year: Year the guideline was published or released | 1 | 0.37% |
| record | locations | 1 | 0.37% |
| record | political_context | 1 | 0.37% |
| record | faq_references | 1 | 0.37% |
| record | navigation | 1 | 0.37% |
| record | footer_links | 1 | 0.37% |
| record | browse_topics | 1 | 0.37% |
| record | status: Current operational status of the device | 1 | 0.37% |
| record | owner: Name of the healthcare professional owning the device | 1 | 0.37% |
| record | location: Department or unit where device is currently located | 1 | 0.37% |
| record | device_id: Unique identifier assigned to the medical device | 1 | 0.37% |
| record | model: Manufacturer model name of the medical device | 1 | 0.37% |
| record | investor_list | 1 | 0.37% |
| record | briefing_event | 1 | 0.37% |
| record | follow_up_interest | 1 | 0.37% |
| record | pilot_hubs | 1 | 0.37% |
| record | context.region | 1 | 0.37% |
| record | context.demographic_challenge | 1 | 0.37% |
| record | context.strategic_approach | 1 | 0.37% |
| record | lead_agency.name | 1 | 0.37% |
| record | lead_agency.abbreviation | 1 | 0.37% |
| record | lead_agency.responsibilities | 1 | 0.37% |
| record | lead_agency.primary_goal | 1 | 0.37% |
| record | project.name | 1 | 0.37% |
| record | project.abbreviation | 1 | 0.37% |
| record | project.financials.budget | 1 | 0.37% |
| record | project.financials.funding_source | 1 | 0.37% |
| record | project.timeline.duration_years | 1 | 0.37% |
| record | project.scope.setting_type | 1 | 0.37% |
| record | project.objectives | 1 | 0.37% |
| record | project.operational_model.key_roles | 1 | 0.37% |
| record | project.operational_model.stakeholders.0.entity_type | 1 | 0.37% |
| record | project.operational_model.stakeholders.0.description | 1 | 0.37% |
| record | project.operational_model.stakeholders.1.entity_type | 1 | 0.37% |
| record | project.operational_model.stakeholders.1.description | 1 | 0.37% |
| record | project.operational_model.stakeholders.2.entity_type | 1 | 0.37% |
| record | project.operational_model.stakeholders.2.description | 1 | 0.37% |
| record | outcomes.monitored_metrics | 1 | 0.37% |
| record | outcomes.preliminary_findings | 1 | 0.37% |
| record | outcomes.qualitative_impact | 1 | 0.37% |
| record | training_modules: List of specific training topics assigned to the employee | 1 | 0.37% |
| record | name: Full name of the new employee | 1 | 0.37% |
| record | employee_id: Unique identifier assigned to each new hire | 1 | 0.37% |
| record | bench_points: Total points scored by bench players | 1 | 0.37% |
| record | key_moment: Descriptive note about a pivotal play or event in the game | 1 | 0.37% |
| record | steals: Total steals recorded during the game | 1 | 0.37% |
| record | game_id: Unique identifier for the game as provided in the passage | 1 | 0.37% |
| record | home_score: Final score of the home team | 1 | 0.37% |
| record | founding_year | 1 | 0.37% |
| record | company_id | 1 | 0.37% |
| record | key_challenge | 1 | 0.37% |
| record | outcome | 1 | 0.37% |
| record | organizer | 1 | 0.37% |
| record | attendance | 1 | 0.37% |
| record | contact_email | 1 | 0.37% |
| record | isbn | 1 | 0.37% |
| record | publication_volume | 1 | 0.37% |
| record | publication_issue | 1 | 0.37% |
| record | name: Product name as listed in inventory records | 1 | 0.37% |
| record | last_updated: Date when inventory was last modified | 1 | 0.37% |
| record | sku_id: Unique identifier for the product in inventory system | 1 | 0.37% |
| record | quantity: Current stock level of the SKU | 1 | 0.37% |
| record | location: Warehouse or hub where the SKU is stored | 1 | 0.37% |
| record | contribution_id: Unique identifier for each contribution entry | 1 | 0.37% |
| record | address: Physical address of the donor as recorded | 1 | 0.37% |
| record | status: Current status of the contribution record | 1 | 0.37% |
| record | campaign_target: Name of the candidate or office being supported | 1 | 0.37% |
| record | asteroid.orbital_characteristics.reference_plane | 1 | 0.37% |
| record | asteroid.designation_number | 1 | 0.37% |
| record | meeting_id | 1 | 0.37% |
| record | province | 1 | 0.37% |
| record | professions | 1 | 0.37% |
| record | performance_rating | 1 | 0.37% |
| record | training_modules | 1 | 0.37% |
| record | room_type | 1 | 0.37% |
| record | nightly_rate | 1 | 0.37% |
| record | notes | 1 | 0.37% |
| record | outlet | 1 | 0.37% |
| record | location: Property address including street, city, and state | 1 | 0.37% |
| record | claim_id: Unique identifier for the insurance claim | 1 | 0.37% |
| record | buyer_name | 1 | 0.37% |
| record | store_reference | 1 | 0.37% |
| record | battery_life | 1 | 0.37% |
| record | sound_quality | 1 | 0.37% |
| record | usability_score | 1 | 0.37% |
| record | audiofidelityscore | 1 | 0.37% |
| record | dispute_status | 1 | 0.37% |
| record | owner | 1 | 0.37% |
| record | owner_location | 1 | 0.37% |
| record | service_impact | 1 | 0.37% |
| record | outage_start | 1 | 0.37% |
| record | outage_end | 1 | 0.37% |
| record | outage_code | 1 | 0.37% |
| record | root_cause | 1 | 0.37% |
| record | defendant_name | 1 | 0.37% |
| record | plaintiff_address | 1 | 0.37% |
| record | plaintiff_name | 1 | 0.37% |
| record | contributor_count | 1 | 0.37% |
| record | project_url | 1 | 0.37% |
| record | lastcommitdate | 1 | 0.37% |
| record | reviewer_name | 1 | 0.37% |
| record | game_id | 1 | 0.37% |
| record | player_name | 1 | 0.37% |
| record | steals | 1 | 0.37% |
| record | assists | 1 | 0.37% |
| record | player_id | 1 | 0.37% |
| record | location: Physical address where course is held | 1 | 0.37% |
| record | duration: Number of weeks the course runs | 1 | 0.37% |
| record | credits: Number of academic credits assigned to the course | 1 | 0.37% |
| record | cost: Total cost of enrollment in US dollars | 1 | 0.37% |
| record | municipality.name | 1 | 0.37% |
| record | features: List of key technological or functional capabilities | 1 | 0.37% |
| record | price: Base price of the product in US dollars | 1 | 0.37% |
| record | regions: Geographic areas where the product is distributed or in use | 1 | 0.37% |
| record | name: Name of the product as introduced | 1 | 0.37% |
| record | utilities_included | 1 | 0.37% |
| record | contact_method | 1 | 0.37% |
| record | preferredtenanttype | 1 | 0.37% |
| record | symptom | 1 | 0.37% |
| record | report_date | 1 | 0.37% |
| record | icd10code | 1 | 0.37% |
| record | line1 | 1 | 0.37% |
| record | salary_range | 1 | 0.37% |
| record | company_size | 1 | 0.37% |
| record | underwriting_model | 1 | 0.37% |
| record | concepts | 1 | 0.37% |
| record | policytype | 1 | 0.37% |
| record | prescription_id | 1 | 0.37% |
| record | report_id: Unique identifier assigned to the phishing report | 1 | 0.37% |
| record | status: Current status of the report | 1 | 0.37% |
| record | sender_domain: Domain of the spoofed email address | 1 | 0.37% |
| record | topics | 1 | 0.37% |
| record | actions.0.target | 1 | 0.37% |
| record | actions.0.location | 1 | 0.37% |
| record | instruction | 1 | 0.37% |
| record | key_achievement | 1 | 0.37% |
| record | review_period | 1 | 0.37% |
| record | distance.calculation_method | 1 | 0.37% |
| record | latin_name | 1 | 0.37% |
| record | distance.parsecs | 1 | 0.37% |
| record | apparent_magnitude.max | 1 | 0.37% |
| record | characteristics.effective_temperature.value | 1 | 0.37% |
| record | characteristics.type | 1 | 0.37% |
| record | distance.light_years | 1 | 0.37% |
| record | constellation | 1 | 0.37% |
| record | apparent_magnitude.unit | 1 | 0.37% |
| record | baths | 1 | 0.37% |
| record | zip_code | 1 | 0.37% |
| record | beds | 1 | 0.37% |
| record | symbols | 1 | 0.37% |
| record | determiners | 1 | 0.37% |
| record | verbs_past_participle | 1 | 0.37% |
| record | verbs_past_tense | 1 | 0.37% |
| record | prepositions | 1 | 0.37% |
| record | punctuation | 1 | 0.37% |
| record | possessive_endings | 1 | 0.37% |
| record | adjectives | 1 | 0.37% |
| record | singular_nouns | 1 | 0.37% |
| record | disease_context | 1 | 0.37% |
| record | budget | 1 | 0.37% |
| record | agency | 1 | 0.37% |
| record | technology | 1 | 0.37% |
| record | game_status | 1 | 0.37% |
| record | event_type | 1 | 0.37% |
| record | performance_notes | 1 | 0.37% |
| record | goals | 1 | 0.37% |
| record | player | 1 | 0.37% |
| record | reviewer | 1 | 0.37% |
| record | away_team: Name of the team playing away | 1 | 0.37% |
| record | away_score: Final score for the away team | 1 | 0.37% |
| record | game_id: Unique identifier for the game as listed in the schedule | 1 | 0.37% |
| record | submission_date | 1 | 0.37% |
| record | research_submission | 1 | 0.37% |
| record | authors | 1 | 0.37% |
| record | copyright | 1 | 0.37% |
| record | RP | 1 | 0.37% |
| record | VBD | 1 | 0.37% |
| record | bracket_path | 1 | 0.37% |
| record | team_name | 1 | 0.37% |
| record | team_id | 1 | 0.37% |
| record | registered_on | 1 | 0.37% |
| record | primary_provider: Name of the attending physician | 1 | 0.37% |
| record | admission_diagnosis: Primary diagnosis at admission | 1 | 0.37% |
| record | discharge_date: Date the patient was discharged | 1 | 0.37% |
| record | medications: List of prescribed medications | 1 | 0.37% |
| record | use_case | 1 | 0.37% |
| record | review_length | 1 | 0.37% |
| record | compatibility | 1 | 0.37% |
| record | release_year | 1 | 0.37% |
| record | charging_time | 1 | 0.37% |
| record | warranty_period | 1 | 0.37% |
| record | product_id: Unique identifier assigned by the manufacturer | 1 | 0.37% |
| record | price: Product cost in US dollars | 1 | 0.37% |
| record | company: Manufacturer or developer of the product | 1 | 0.37% |
| record | launch_year: Year the product was first introduced | 1 | 0.37% |
| record | name: Product model name as branded | 1 | 0.37% |
| record | era_name | 1 | 0.37% |
| record | year_in_era | 1 | 0.37% |
| record | dynasty | 1 | 0.37% |
| record | program_name | 1 | 0.37% |
| record | ticker | 1 | 0.37% |
| record | exchange | 1 | 0.37% |
| record | amount: Monetary value of the returned item | 1 | 0.37% |
| record | product_name: Name of the product returned | 1 | 0.37% |
| record | customer_name: Name of the customer initiating the return | 1 | 0.37% |
| record | order_id: Identifier of the original order associated with the return | 1 | 0.37% |
| record | resolution_timeline | 1 | 0.37% |
| record | places | 1 | 0.37% |
| record | organizations | 1 | 0.37% |
| record | location: Physical address of the product launch | 1 | 0.37% |
| record | product_code: Internal code assigned to the product | 1 | 0.37% |
| record | price: Selling price of the product in dollars | 1 | 0.37% |
| record | product: Name of the skincare or apparel product | 1 | 0.37% |
| record | person.publication_information.available_part | 1 | 0.37% |
| record | person.commentary_based_on | 1 | 0.37% |
| record | person.publication_information.commentary_with | 1 | 0.37% |
| record | person.publication_information.location | 1 | 0.37% |
| record | person.time_period.era | 1 | 0.37% |
| record | person.associated_with | 1 | 0.37% |
| record | first_contact_resolution_rate | 1 | 0.37% |
| record | priority | 1 | 0.37% |
| record | assigned_agent | 1 | 0.37% |
| record | support_code | 1 | 0.37% |
| record | bracket_id | 1 | 0.37% |
| record | result | 1 | 0.37% |
| record | teams | 1 | 0.37% |
| record | company.founding_location.city | 1 | 0.37% |
| record | company.location | 1 | 0.37% |
| record | company.transformation_date | 1 | 0.37% |
| record | company.name | 1 | 0.37% |
| record | company.description | 1 | 0.37% |
| record | company.establishment_date | 1 | 0.37% |
| record | claim_amount | 1 | 0.37% |
| record | country: Nationality of the athlete who won the medal | 1 | 0.37% |
| record | venue: Location where the event took place | 1 | 0.37% |
| record | result: Medal awarded to the athlete | 1 | 0.37% |
| record | discipline: Type of sport or competition such as swimming or skiing | 1 | 0.37% |
| record | athlete_name: Name of the athlete who achieved the result | 1 | 0.37% |
| record | event.person.name | 1 | 0.37% |
| record | source | 1 | 0.37% |
| record | event.new_club.contract_duration | 1 | 0.37% |
| record | event.person.age | 1 | 0.37% |
| record | event.new_club.name | 1 | 0.37% |
| record | event.opponent_club.name | 1 | 0.37% |
| record | event.person.position | 1 | 0.37% |
| record | event.person.role | 1 | 0.37% |
| record | document.location | 1 | 0.37% |
| record | document.counterfeiting_statistics.job_losses | 1 | 0.37% |
| record | document.strategy_elements.3.area | 1 | 0.37% |
| record | document.commissioners.1.portfolio | 1 | 0.37% |
| record | document.strategy_elements.6.area | 1 | 0.37% |
| record | document.key_issues | 1 | 0.37% |
| record | document.strategy_elements.0.area | 1 | 0.37% |
| record | document.commissioners.1.name | 1 | 0.37% |
| record | document.commissioners.0.name | 1 | 0.37% |
| record | document.strategy_elements.4.area | 1 | 0.37% |
| record | document.title | 1 | 0.37% |
| record | document.related_documents | 1 | 0.37% |
| record | document.reference_id | 1 | 0.37% |
| record | document.economic_impact.brand_value | 1 | 0.37% |
| record | document.strategy_elements.1.area | 1 | 0.37% |
| record | death.date_julian | 1 | 0.37% |
| record | birth.date_gregorian | 1 | 0.37% |
| record | works.type | 1 | 0.37% |
| record | birth.date_julian | 1 | 0.37% |
| record | occupation | 1 | 0.37% |
| record | teacher | 1 | 0.37% |
| record | death.location | 1 | 0.37% |
| record | father.occupation | 1 | 0.37% |
| record | birth.location | 1 | 0.37% |
| record | exam_type | 1 | 0.37% |
| record | protocol | 1 | 0.37% |
| record | text | 1 | 0.37% |
| record | sender_name: Name of the sender in the correspondence | 1 | 0.37% |
| record | sku | 1 | 0.37% |
| record | sales_volume | 1 | 0.37% |
| record | guidelines.distinction_results_interpretation | 1 | 0.37% |
| record | optional_information | 1 | 0.37% |
| record | research_framework_items | 1 | 0.37% |
| record | publication_date.city | 1 | 0.37% |
| record | publication_date.month | 1 | 0.37% |
| record | guidelines.adherence | 1 | 0.37% |
| record | organization.name | 1 | 0.37% |
| record | subject_focus: Key subjects or pathways emphasized in the results | 1 | 0.37% |
| record | grade_distribution: Breakdown of grades achieved by students | 1 | 0.37% |
| record | location: Geographic region or city where the institution is located | 1 | 0.37% |
| record | publication_code: Unique identifier assigned to the result publication | 1 | 0.37% |
| record | notes: Additional context about the appointment or background | 1 | 0.37% |
| record | compensation: Annual salary or compensation amount | 1 | 0.37% |
| record | position: Job title or role within the department | 1 | 0.37% |
| record | date: Start date of the appointment | 1 | 0.37% |
| record | claims_count | 1 | 0.37% |
| record | filing_location | 1 | 0.37% |
| record | tech_product | 1 | 0.37% |
| record | content | 1 | 0.37% |
| record | tweet_id | 1 | 0.37% |
| record | author_handle | 1 | 0.37% |
| record | procedure_code | 1 | 0.37% |
| record | population | 1 | 0.37% |
| record | entity_name | 1 | 0.37% |
| record | file_id | 1 | 0.37% |
| record | jurisdiction | 1 | 0.37% |
| record | vendor_address | 1 | 0.37% |
| record | recipient_name | 1 | 0.37% |
| record | recipient_email | 1 | 0.37% |
| record | normal_range | 1 | 0.37% |
| record | test_name | 1 | 0.37% |
| record | value | 1 | 0.37% |
| record | lab_id | 1 | 0.37% |
| record | drug_code | 1 | 0.37% |
| record | medicalrecordid | 1 | 0.37% |
| record | effective_date | 1 | 0.37% |
| record | holder_name | 1 | 0.37% |
| record | categories | 1 | 0.37% |
| record | bathrooms | 1 | 0.37% |
| record | bedrooms | 1 | 0.37% |
| record | status: Current state of the incident | 1 | 0.37% |
| record | customer_name: Name of the affected customer | 1 | 0.37% |
| record | address: Full residential or commercial address of the customer | 1 | 0.37% |
| record | impact: Description of service impact including data loss or business effects | 1 | 0.37% |
| record | serial_number | 1 | 0.37% |
| record | bp | 1 | 0.37% |
| record | doctor | 1 | 0.37% |
| record | rr | 1 | 0.37% |
| record | serial_number: Unique serial number for device tracking | 1 | 0.37% |
| record | pathway_name | 1 | 0.37% |
| record | splicing_regulation.current_research_status | 1 | 0.37% |
| record | clinical_observations.activation_status | 1 | 0.37% |
| record | splicing_regulation.impact_on_pathway_components.0.regulatory_mode | 1 | 0.37% |
| record | splicing_regulation.impact_on_pathway_components.0.component_description | 1 | 0.37% |
| record | articleTitle | 1 | 0.37% |
| record | symbol: Stock ticker symbol as listed in the data | 1 | 0.37% |
| record | change_percent: Percentage change in stock price for the day | 1 | 0.37% |
| record | close_price: Final closing price of the stock | 1 | 0.37% |
| record | volume: Total shares traded on the day | 1 | 0.37% |
| record | market_cap: Market capitalization in billions | 1 | 0.37% |
| record | movement_reason: Brief reason for price movement or performance | 1 | 0.37% |
| record | contact_person | 1 | 0.37% |
| record | paymentduedate | 1 | 0.37% |
| record | vendor_name | 1 | 0.37% |
| record | account_code | 1 | 0.37% |
| record | rma_number | 1 | 0.37% |
| record | failure_reason | 1 | 0.37% |
| record | return_date | 1 | 0.37% |
| record | revenue | 1 | 0.37% |
| record | operating_margin | 1 | 0.37% |
| record | company_code | 1 | 0.37% |
| record | percentage | 1 | 0.37% |
| record | title: Event title describing the nature and context of the incident | 1 | 0.37% |
| record | date: Timestamp when the incident occurred | 1 | 0.37% |
| record | involved_parties: List of organizations or individuals involved in the incident | 1 | 0.37% |
| record | source: Media or organization reporting the incident | 1 | 0.37% |
| record | lead_reporter: Reporter providing coverage of the incident | 1 | 0.37% |
| record | geographic_code: Unique code identifying the incident's geographic region | 1 | 0.37% |
| record | isotope | 1 | 0.37% |
| record | match_type | 1 | 0.37% |
| record | disbursement_plan | 1 | 0.37% |
| record | gpa | 1 | 0.37% |
| record | ranking.location_type | 1 | 0.37% |
| record | location.province | 1 | 0.37% |
| record | booking_options.partnerships | 1 | 0.37% |
| record | price_range.per | 1 | 0.37% |
| record | booking_options.tripadvisor_role | 1 | 0.37% |
| record | user: Name of the customer who provided the review | 1 | 0.37% |
| record | rating: Numerical rating from 1 to 5 given by the user | 1 | 0.37% |
| record | location: City and state or province where purchase occurred | 1 | 0.37% |
| record | position_code: Unique identifier for the appointment | 1 | 0.37% |
| record | department: Government department or agency where the appointment occurs | 1 | 0.37% |
| record | asset | 1 | 0.37% |
| record | place.type | 1 | 0.37% |
| record | place.location.province | 1 | 0.37% |
| record | place.demographics.0.year | 1 | 0.37% |
| record | place.demographics.0.population | 1 | 0.37% |
| record | place.location.department | 1 | 0.37% |
| record | place.name | 1 | 0.37% |
| record | downvotes | 1 | 0.37% |
| record | originalpostlink | 1 | 0.37% |
| record | content_length | 1 | 0.37% |
| record | comment_rank | 1 | 0.37% |
| record | purpose | 1 | 0.37% |
| record | payee | 1 | 0.37% |
| record | account | 1 | 0.37% |
| record | family | 1 | 0.37% |
| record | distribution | 1 | 0.37% |
| record | scientific_name | 1 | 0.37% |
| record | planning | 1 | 0.37% |
| record | startup | 1 | 0.37% |
| record | funding_round | 1 | 0.37% |
| record | release_target | 1 | 0.37% |
| record | attendee_count | 1 | 0.37% |
| record | relationships.0.entity | 1 | 0.37% |
| record | relationships.0.action | 1 | 0.37% |
| record | relationships.0.type | 1 | 0.37% |
| record | activities | 1 | 0.37% |
| record | lead_person | 1 | 0.37% |
| record | target | 1 | 0.37% |
| record | legal_counsel | 1 | 0.37% |
| record | medical condition | 1 | 0.37% |
| record | medical procedure | 1 | 0.37% |
| record | problem_description | 1 | 0.37% |
| record | affected_network | 1 | 0.37% |
| record | access_method | 1 | 0.37% |
| record | affected_site | 1 | 0.37% |
| record | problem_start_time | 1 | 0.37% |
| record | alternative_access | 1 | 0.37% |
| record | visit_count | 1 | 0.37% |
| record | website | 1 | 0.37% |
| record | device model | 1 | 0.37% |
| record | developer | 1 | 0.37% |
| record | place | 1 | 0.37% |
| record | availability_date | 1 | 0.37% |
| record | availability_status | 1 | 0.37% |
| record | model_number | 1 | 0.37% |
| record | rating: Star rating out of 5 given by the reviewer | 1 | 0.37% |
| record | battery_life: Battery duration in hours under normal use | 1 | 0.37% |
| record | location: City and country of the reviewer | 1 | 0.37% |
| record | currency_pair: Pair of currencies represented in the rate | 1 | 0.37% |
| record | id: Unique identifier for each exchange rate record | 1 | 0.37% |
| record | description: Brief explanation of market event influencing the rate | 1 | 0.37% |
| record | founder | 1 | 0.37% |
| record | founded_year | 1 | 0.37% |
| record | date: Date when the reaction was reported | 1 | 0.37% |
| record | reaction: Observed adverse effect or symptom | 1 | 0.37% |
| record | status: Current status of the reaction | 1 | 0.37% |
| record | drug: Name and dosage form of the medication taken | 1 | 0.37% |
| record | duration: Number of days the treatment lasted for the reaction | 1 | 0.37% |
| record | patient_id: Unique identifier for the patient in the record | 1 | 0.37% |
| record | service.target_audience | 1 | 0.37% |
| record | service.responsible_person.affiliation | 1 | 0.37% |
| record | service.opening_hours.0.time | 1 | 0.37% |
| record | service.location.address | 1 | 0.37% |
| record | service.responsible_person.profession | 1 | 0.37% |
| record | service.opening_hours.0.day | 1 | 0.37% |
| record | service.location.building | 1 | 0.37% |
| record | service.interventions | 1 | 0.37% |
| record | service.responsible_person.role | 1 | 0.37% |
| record | service.contact_information.email | 1 | 0.37% |
| record | service.contact_information.phone | 1 | 0.37% |
| record | service.location.postal_code | 1 | 0.37% |
| record | genre | 1 | 0.37% |
| record | match | 1 | 0.37% |
| record | phone | 1 | 0.37% |
| record | university | 1 | 0.37% |
| record | credit_hours | 1 | 0.37% |
| record | enrollment_status | 1 | 0.37% |
| record | meeting_days | 1 | 0.37% |
| record | enrollment_capacity | 1 | 0.37% |
| record | passenger_city | 1 | 0.37% |
| record | delay_minutes | 1 | 0.37% |
| record | origin | 1 | 0.37% |
| record | delay_reason | 1 | 0.37% |
| record | destination | 1 | 0.37% |
| record | flight_id | 1 | 0.37% |
| record | disclosure_id | 1 | 0.37% |
| record | donor_name | 1 | 0.37% |
| record | identifier | 1 | 0.37% |
| record | staffing_change | 1 | 0.37% |
| record | new_website_url | 1 | 0.37% |
| record | spo2: Oxygen saturation percentage | 1 | 0.37% |
| record | blood_pressure: Blood pressure in format systolic/diastolic | 1 | 0.37% |
| record | heart_rate: Number of heartbeats per minute | 1 | 0.37% |
| record | address: Street address including city and zip code | 1 | 0.37% |
| record | closing_price | 1 | 0.37% |
| record | reason | 1 | 0.37% |
| record | prescriber | 1 | 0.37% |
| record | fda_code | 1 | 0.37% |
| record | critic_name | 1 | 0.37% |
| record | review_source | 1 | 0.37% |
| record | office_address | 1 | 0.37% |
| record | critic_location | 1 | 0.37% |
| record | sentiment.opinion | 1 | 0.37% |
| record | sentiment.preference | 1 | 0.37% |
| record | quote.text | 1 | 0.37% |
| record | quote.author | 1 | 0.37% |
| record | favorite_players | 1 | 0.37% |
| record | edit_information.editor | 1 | 0.37% |
| record | edit_information.timestamp | 1 | 0.37% |
| record | insurer | 1 | 0.37% |
| record | premium | 1 | 0.37% |
| record | policy_type | 1 | 0.37% |
| record | agent | 1 | 0.37% |
| record | event.description | 1 | 0.37% |
| record | event.organizer.name | 1 | 0.37% |
| record | organization_info.electricity_source | 1 | 0.37% |
| record | event.organizer.abbreviation | 1 | 0.37% |
| record | organization_info.funding | 1 | 0.37% |
| record | event.location.city | 1 | 0.37% |
| record | organization_info.facilities.generating_facilities | 1 | 0.37% |
| record | event.time.photo_op_end | 1 | 0.37% |
| record | event.activities | 1 | 0.37% |
| record | organization_info.website | 1 | 0.37% |
| record | event.name | 1 | 0.37% |
| record | event.time.photo_op_start | 1 | 0.37% |
| record | donations.recipient.type | 1 | 0.37% |
| record | event.time.end | 1 | 0.37% |
| record | release_date | 1 | 0.37% |
| record | event.location.address | 1 | 0.37% |
| record | beliefs | 1 | 0.37% |
| record | actions | 1 | 0.37% |
| record | affiliation | 1 | 0.37% |
| record | drug_name | 1 | 0.37% |
| record | indication | 1 | 0.37% |
| record | ndc | 1 | 0.37% |
| record | patient_gender | 1 | 0.37% |
| record | adverse_event | 1 | 0.37% |
| record | start_date | 1 | 0.37% |
| record | publication | 1 | 0.37% |
| record | follow_up.average_duration.years | 1 | 0.37% |
| record | population.age_range.max | 1 | 0.37% |
| record | study_metadata.measurement_technology.full_name | 1 | 0.37% |
| record | statistical_results.diagnostic_metrics.sensitivity_at_threshold.t_score_threshold | 1 | 0.37% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.0.t_score | 1 | 0.37% |
| record | population.age_range.unit | 1 | 0.37% |
| record | study_metadata.measurement_technology.abbreviation | 1 | 0.37% |
| record | population.age_range.min | 1 | 0.37% |
| record | population.gender | 1 | 0.37% |
| record | statistical_results.diagnostic_metrics.sensitivity_at_threshold.percentage_of_fractures | 1 | 0.37% |
| record | statistical_results.predictive_model.value | 1 | 0.37% |
| record | conclusions | 1 | 0.37% |
| record | study_metadata.measurement_technology.anatomical_site | 1 | 0.37% |
| record | outcomes.fracture_events.0.count | 1 | 0.37% |
| record | study_metadata.study_type | 1 | 0.37% |
| record | status: Updated clinical status like stable, unstable, critical, or resolved | 1 | 0.37% |
| record | patient_id: Unique identifier for the monitored patient | 1 | 0.37% |
| record | clinical_status: Current clinical condition such as normal or elevated | 1 | 0.37% |
| record | oxygen_saturation: Percentage of oxygen saturation in blood | 1 | 0.37% |
| record | election_type | 1 | 0.37% |
| record | candidate_location | 1 | 0.37% |
| record | candidate_name | 1 | 0.37% |
| record | prescriber_id | 1 | 0.37% |
| record | prescriber_name | 1 | 0.37% |
| record | hotel_name | 1 | 0.37% |
| record | returner: Name of the person initiating the return | 1 | 0.37% |
| record | location: City or address where return was submitted | 1 | 0.37% |
| record | item: Description of the returned product | 1 | 0.37% |
| record | amount: Monetary value of the return in dollars | 1 | 0.37% |
| record | status: Current status of the return approval | 1 | 0.37% |
| record | capacity | 1 | 0.37% |
| record | sponsor | 1 | 0.37% |
| record | status: Current status of the exam result | 1 | 0.37% |
| record | school: Name of the school where the exam was administered | 1 | 0.37% |
| record | score: Percentage score achieved in the exam | 1 | 0.37% |
| record | subject: Subject area of the exam | 1 | 0.37% |
| record | asteroid.type | 1 | 0.37% |
| record | asteroid.discovery_year | 1 | 0.37% |
| record | asteroid.orbital_characteristics.unit | 1 | 0.37% |
| record | asteroid.orbital_characteristics.inclination_unit | 1 | 0.37% |
| record | model | 1 | 0.37% |
| record | color | 1 | 0.37% |
| record | user_rating | 1 | 0.37% |
| record | user_feedback | 1 | 0.37% |
| record | subject | 1 | 0.37% |
| record | reference_id | 1 | 0.37% |
| record | regulatory_status.recognition_type | 1 | 0.37% |
| record | announcement_metadata.source_url | 1 | 0.37% |
| record | announcement_metadata.coverage_provider | 1 | 0.37% |
| record | regulatory_status.review_status | 1 | 0.37% |
| record | announcement_metadata.date | 1 | 0.37% |
| record | announcement_metadata.agency | 1 | 0.37% |
| record | status: Current enrollment or completion status | 1 | 0.37% |
| record | start_date: Date when the course begins | 1 | 0.37% |
| record | department: Academic department offering the course | 1 | 0.37% |
| record | instructor: Name of the faculty member teaching the course | 1 | 0.37% |
| record | course_id: Unique identifier for the course as given in the XML | 1 | 0.37% |
| record | establishment_year | 1 | 0.37% |
| record | departments | 1 | 0.37% |
| record | staff.research_assistant_count | 1 | 0.37% |
| record | staff.teaching_staff_count | 1 | 0.37% |
| record | dean.name | 1 | 0.37% |
| record | laboratories | 1 | 0.37% |
| record | dean.title | 1 | 0.37% |
| record | medications | 1 | 0.37% |
| record | symptoms | 1 | 0.37% |
| record | diagnosis | 1 | 0.37% |
| record | mrn | 1 | 0.37% |
| record | followupclinic_address: Physical address of follow-up clinic | 1 | 0.37% |
| record | followupprovider: Name of provider for follow-up visit | 1 | 0.37% |
| record | patient_mrn: Unique medical record number identifying the patient | 1 | 0.37% |
| record | treatments: Prescribed medications or therapies during stay | 1 | 0.37% |
| record | discharge_instructions: Written guidance given to patient post-discharge | 1 | 0.37% |
| record | id: Unique identifier for each rental listing | 1 | 0.37% |
| record | pet_policy: Pet policy including restrictions or fees | 1 | 0.37% |
| record | property_type: Type of housing unit | 1 | 0.37% |
| record | description: Brief description of unit features or conditions | 1 | 0.37% |
| record | address: Full property address including street and city | 1 | 0.37% |
| record | award | 1 | 0.37% |
| record | owner_name | 1 | 0.37% |
| record | term_years | 1 | 0.37% |
| record | research_project | 1 | 0.37% |
| record | principal_investigator | 1 | 0.37% |
| record | institution | 1 | 0.37% |
| record | project_status | 1 | 0.37% |
| record | support_agent | 1 | 0.37% |
| record | user_name: Name of the creator who posted the trend | 1 | 0.37% |
| record | video_id: Unique identifier for the video content | 1 | 0.37% |
| record | shares: Total number of shares the video received | 1 | 0.37% |
| record | comments: Total number of comments on the video | 1 | 0.37% |
| record | location: City and state where the user is based | 1 | 0.37% |
| record | reasons | 1 | 0.37% |
| record | suggestions | 1 | 0.37% |
| record | word_count | 1 | 0.37% |
| record | due_date | 1 | 0.37% |
| record | product.specifications.display.resolution | 1 | 0.37% |
| record | product.launch_date | 1 | 0.37% |
| record | product.registration_count.count | 1 | 0.37% |
| record | product.specifications.memory.internal_storage | 1 | 0.37% |
| record | product.availability.currently_available | 1 | 0.37% |
| record | product.specifications.connectivity | 1 | 0.37% |
| record | product.price.1.currency | 1 | 0.37% |
| record | product.brand | 1 | 0.37% |
| record | product.specifications.camera.features | 1 | 0.37% |
| record | product.specifications.display.size | 1 | 0.37% |
| record | product.launch_location | 1 | 0.37% |
| record | product.model | 1 | 0.37% |
| record | product.registration_count.time_period | 1 | 0.37% |
| record | product.specifications.camera.rear | 1 | 0.37% |
| record | product.specifications.battery.capacity | 1 | 0.37% |
| record | product.specifications.memory.ram | 1 | 0.37% |
| record | venue: Name of the stadium or venue where the event occurred | 1 | 0.37% |
| record | nation: Country of the athlete who won the medal | 1 | 0.37% |
| record | time: Official race time recorded in the event | 1 | 0.37% |
| record | medal: Type of medal won gold, silver, or bronze | 1 | 0.37% |
| record | note | 1 | 0.37% |
| record | diagnosis_code | 1 | 0.37% |
| record | surface | 1 | 0.37% |
| record | entries | 1 | 0.37% |
| record | tournament_name | 1 | 0.37% |
| record | winner_ranking | 1 | 0.37% |
| record | opponent_name | 1 | 0.37% |
| record | attending_physician | 1 | 0.37% |
| record | bloodpressurediastolic | 1 | 0.37% |
| record | monitoring_method | 1 | 0.37% |
| record | bloodpressuresystolic | 1 | 0.37% |
| record | respiratory_rate | 1 | 0.37% |
| record | acquirer | 1 | 0.37% |
| record | acquired | 1 | 0.37% |
| record | value_original | 1 | 0.37% |
| record | regulatory_status | 1 | 0.37% |
| record | patents | 1 | 0.37% |
| record | employees | 1 | 0.37% |
| record | customer_contracts | 1 | 0.37% |
| record | integration_plan | 1 | 0.37% |
| record | platform_name | 1 | 0.37% |
| record | earnout | 1 | 0.37% |
| record | sustainability_commitment | 1 | 0.37% |
| record | related_conditions | 1 | 0.37% |
| record | record_metadata | natural/location ×8, natural/status ×7, natural/title ×6, natural/policy_id ×4 |  |

**val.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| casualty_report | location | 931 | 57.01% |
| casualty_report | dead | 897 | 54.93% |
| casualty_report | injured | 876 | 53.64% |
| casualty_report | missing | 257 | 15.74% |
| casualty_report | record_metadata | natural/dead ×697, natural/injured ×345, natural/missing ×39, natural/location ×34 |  |

**test.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| casualty_report | dead | 951 | 56.61% |
| casualty_report | location | 900 | 53.57% |
| casualty_report | injured | 881 | 52.44% |
| casualty_report | missing | 290 | 17.26% |
| casualty_report | record_metadata | natural/dead ×739, natural/injured ×311, natural/location ×35, natural/missing ×30 |  |

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | unknown |
| relations | unknown |
| events | unknown |
| classifications | unknown |
| structures | unknown |
