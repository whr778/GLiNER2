# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tr_dose5000 | train | 25,218 | 25,218 | 31,231 | 2,395 | 3,842 | 30,620 | 5,217 | 5,739 |
| tr_dose5000 | val | 1,115 | 1,115 | 0 | 0 | 0 | 1,633 | 0 | 0 |
| tr_dose5000 | test | 1,115 | 1,115 | 0 | 0 | 0 | 1,680 | 0 | 0 |

## tr_dose5000

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 25,218 | 25,218 | 205 | 620 | 9823 | 31,231 | 2,395 | 3,842 | 30,620 | 5,217 | 5,739 |
| val | 1,115 | 1,115 | 194 | 413 | 1145 | 0 | 0 | 0 | 1,633 | 0 | 0 |
| test | 1,115 | 1,115 | 200 | 437 | 1110 | 0 | 0 | 0 | 1,680 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 16,536 | 322 | 345 | 17,203 | 62.67% |
| tur | 5,001 | 793 | 770 | 6,564 | 23.91% |
| zho | 3,441 | 0 | 0 | 3,441 | 12.54% |
| und | 160 | 0 | 0 | 160 | 0.58% |
| rus | 9 | 0 | 0 | 9 | 0.03% |
| ita | 8 | 0 | 0 | 8 | 0.03% |
| fra | 7 | 0 | 0 | 7 | 0.03% |
| kor | 7 | 0 | 0 | 7 | 0.03% |
| spa | 7 | 0 | 0 | 7 | 0.03% |
| deu | 7 | 0 | 0 | 7 | 0.03% |
| hin | 6 | 0 | 0 | 6 | 0.02% |
| pol | 6 | 0 | 0 | 6 | 0.02% |
| nld | 5 | 0 | 0 | 5 | 0.02% |
| por | 5 | 0 | 0 | 5 | 0.02% |
| ara | 5 | 0 | 0 | 5 | 0.02% |
| ukr | 3 | 0 | 0 | 3 | 0.01% |
| nor | 2 | 0 | 0 | 2 | 0.01% |
| ilo | 1 | 0 | 0 | 1 | 0.00% |
| cym | 1 | 0 | 0 | 1 | 0.00% |
| ces | 1 | 0 | 0 | 1 | 0.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 1,115 |
| test_unique_inputs | 1,115 |
| train_records | 25,218 |
| train_unique_inputs | 25,218 |
| val_records | 1,115 |
| val_unique_inputs | 1,115 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | n/a | 31,231 / 0 |  |
| entities (per mention) | train / test | n/a | 31,231 / 0 |  |
| relations (per instance) | train / val | n/a | 2,395 / 0 |  |
| relations (per instance) | train / test | n/a | 2,395 / 0 |  |
| event types (per trigger mention) | train / val | n/a | 5,971 / 0 |  |
| event types (per trigger mention) | train / test | n/a | 5,971 / 0 |  |
| event roles (per argument) | train / val | n/a | 6,088 / 0 |  |
| event roles (per argument) | train / test | n/a | 6,088 / 0 |  |
| classifications (true_label) | train / val | n/a | 3,842 / 0 |  |
| classifications (true_label) | train / test | n/a | 3,842 / 0 |  |


### Events

**train.** 5,795 instances in 1,426 records, 0 triggerless. Trigger gold 5,217. Argument gold 5,739. Arguments per instance, mean 1.05. Trigger length in whitespace tokens, mean 1.41, single-token share 63.9%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Experiment | 433 | 7.25% |
| Support | 308 | 5.16% |
| Event | 260 | 4.35% |
| Hostile_encounter | 164 | 2.75% |
| Deploy | 160 | 2.68% |
| Process_start | 134 | 2.24% |
| Catastrophe | 120 | 2.01% |
| Attack | 118 | 1.98% |
| Exhibit | 110 | 1.84% |
| match_result | 109 | 1.83% |
| Causation | 106 | 1.78% |
| Cyber.DiscoverVulnerability | 105 | 1.76% |
| Motion | 95 | 1.59% |
| Competition | 83 | 1.39% |
| Cyber.Ransom | 81 | 1.36% |
| product_publish | 76 | 1.27% |
| Process_end | 72 | 1.21% |
| Conquering | 69 | 1.16% |
| Control | 64 | 1.07% |
| Killing | 58 | 0.97% |
| Statement | 57 | 0.95% |
| Cyber.PatchVulnerability | 57 | 0.95% |
| Social_event | 56 | 0.94% |
| Bodily_harm | 51 | 0.85% |
| Hold | 50 | 0.84% |
| Coming_to_be | 50 | 0.84% |
| Military_operation | 49 | 0.82% |
| Accident | 48 | 0.80% |
| Destroying | 46 | 0.77% |
| Self_motion | 45 | 0.75% |
| Preventing_or_letting | 45 | 0.75% |
| Arranging | 44 | 0.74% |
| Arriving | 43 | 0.72% |
| Death | 43 | 0.72% |
| Cause_change_of_position_on_a_scale | 43 | 0.72% |
| Getting | 42 | 0.70% |
| Cyber.Databreach | 41 | 0.69% |
| death | 40 | 0.67% |
| Earnings_and_losses | 37 | 0.62% |
| Presence | 37 | 0.62% |
| Creating | 37 | 0.62% |
| Cause_to_be_included | 37 | 0.62% |
| Giving | 37 | 0.62% |
| Manoeuvre | 35 | 0.59% |
| Change | 35 | 0.59% |
| Becoming | 35 | 0.59% |
| Cyber.Phishing | 34 | 0.57% |
| Escaping | 34 | 0.57% |
| resignation | 34 | 0.57% |
| Damaging | 34 | 0.57% |
| judicial_action_arrest | 33 | 0.55% |
| Action | 33 | 0.55% |
| Know | 33 | 0.55% |
| Change_of_leadership | 33 | 0.55% |
| Supporting | 32 | 0.54% |
| Cause_change_of_strength | 32 | 0.54% |
| Removing | 31 | 0.52% |
| Arrest | 31 | 0.52% |
| Supply | 31 | 0.52% |
| Defending | 30 | 0.50% |
| Expressing_publicly | 30 | 0.50% |
| Traveling | 30 | 0.50% |
| Injure | 30 | 0.50% |
| Participation | 30 | 0.50% |
| Placing | 29 | 0.49% |
| Agree_or_refuse_to_act | 29 | 0.49% |
| Request | 27 | 0.45% |
| Sending | 26 | 0.44% |
| Conflict | 26 | 0.44% |
| Name_conferral | 26 | 0.44% |
| Departing | 25 | 0.42% |
| Judgment_communication | 25 | 0.42% |
| Using | 24 | 0.40% |
| championship_win | 24 | 0.40% |
| advancement | 23 | 0.39% |
| Motion_directional | 22 | 0.37% |
| Bringing | 21 | 0.35% |
| Protest | 21 | 0.35% |
| Perception_active | 21 | 0.35% |
| divorce | 20 | 0.33% |
| Choosing | 20 | 0.33% |
| Besieging | 20 | 0.33% |
| Influence | 19 | 0.32% |
| Building | 19 | 0.32% |
| product_recall | 18 | 0.30% |
| opening | 18 | 0.30% |
| Recovering | 18 | 0.30% |
| Violence | 18 | 0.30% |
| joined | 18 | 0.30% |
| Communication | 17 | 0.28% |
| Cause_to_make_progress | 17 | 0.28% |
| Recording | 17 | 0.28% |
| Use_firearm | 17 | 0.28% |
| Come_together | 16 | 0.27% |
| judicial_action_summon | 16 | 0.27% |
| price_limit_up | 16 | 0.27% |
| Aiming | 16 | 0.27% |
| Achieve | 16 | 0.27% |
| Expansion | 15 | 0.25% |
| Being_in_operation | 15 | 0.25% |
| Assistance | 15 | 0.25% |
| Reporting | 15 | 0.25% |
| Deciding | 14 | 0.23% |
| marriage | 14 | 0.23% |
| Surrendering | 14 | 0.23% |
| Receiving | 14 | 0.23% |
| Traffic Accident | 14 | 0.23% |
| Openness | 13 | 0.22% |
| product_release | 13 | 0.22% |
| Becoming_a_member | 12 | 0.20% |
| Manufacturing | 12 | 0.20% |
| Confronting_problem | 12 | 0.20% |
| Hindering | 12 | 0.20% |
| competition_ban | 12 | 0.20% |
| financing | 11 | 0.18% |
| Check | 11 | 0.18% |
| Collaboration | 11 | 0.18% |
| Dispersal | 11 | 0.18% |
| Cause_to_amalgamate | 11 | 0.18% |
| Scrutiny | 11 | 0.18% |
| Committing_crime | 10 | 0.17% |
| Legal_rulings | 10 | 0.17% |
| Sign_agreement | 10 | 0.17% |
| breakup | 10 | 0.17% |
| Quarreling | 10 | 0.17% |
| Change_event_time | 10 | 0.17% |
| birthday_celebration | 9 | 0.15% |
| Response | 9 | 0.15% |
| Publishing | 9 | 0.15% |
| Connect | 9 | 0.15% |
| judicial_action_imprisonment | 9 | 0.15% |
| departure | 9 | 0.15% |
| Terrorism | 9 | 0.15% |
| sale_acquisition | 9 | 0.15% |
| Fire | 8 | 0.13% |
| product_delist | 8 | 0.13% |
| apologize_to | 8 | 0.13% |
| product_award | 8 | 0.13% |
| Education_teaching | 8 | 0.13% |
| Commerce_pay | 7 | 0.12% |
| Commitment | 7 | 0.12% |
| Preserving | 7 | 0.12% |
| lost_contact | 7 | 0.12% |
| Telling | 7 | 0.12% |
| Employment | 7 | 0.12% |
| Legality | 7 | 0.12% |
| layoff | 6 | 0.10% |
| competition_withdrawal | 6 | 0.10% |
| judicial_action_filing | 6 | 0.10% |
| Bearing_arms | 6 | 0.10% |
| Forming_relationships | 6 | 0.10% |
| GiveUp | 6 | 0.10% |
| wedding | 6 | 0.10% |
| closing | 6 | 0.10% |
| Resolve_problem | 6 | 0.10% |
| Rite | 6 | 0.10% |
| judicial_action_report | 6 | 0.10% |
| Plane Crash | 5 | 0.08% |
| Convincing | 5 | 0.08% |
| Criminal_investigation | 5 | 0.08% |
| Coming_to_believe | 5 | 0.08% |
| Emptying | 5 | 0.08% |
| Body_movement | 5 | 0.08% |
| Commerce_sell | 5 | 0.08% |
| Warning | 5 | 0.08% |
| Expend_resource | 5 | 0.08% |
| Earthquake | 5 | 0.08% |
| GetReady | 5 | 0.08% |
| birth | 5 | 0.08% |
| proposal | 5 | 0.08% |
| Surrounding | 5 | 0.08% |
| Scouring | 5 | 0.08% |
| judicial_action_fine | 5 | 0.08% |
| Award | 5 | 0.08% |
| judicial_action_prosecution | 5 | 0.08% |
| dissolution | 5 | 0.08% |
| Explosion | 5 | 0.08% |
| Research | 5 | 0.08% |
| Collapse | 5 | 0.08% |
| rate_decrease | 4 | 0.07% |
| Imposing_obligation | 4 | 0.07% |
| meet_with | 4 | 0.07% |
| Reveal_secret | 4 | 0.07% |
| Temporary_stay | 4 | 0.07% |
| Adducing | 4 | 0.07% |
| suspension | 4 | 0.07% |
| price_limit_down | 4 | 0.07% |
| praise | 4 | 0.07% |
| Writing | 4 | 0.07% |
| Wearing | 4 | 0.07% |
| Prison | 4 | 0.07% |
| Practice | 4 | 0.07% |
| retirement | 4 | 0.07% |
| Reforming_a_system | 3 | 0.05% |
| Testing | 3 | 0.05% |
| Filling | 3 | 0.05% |
| Change_sentiment | 3 | 0.05% |
| Patrolling | 3 | 0.05% |
| Rescuing | 3 | 0.05% |
| Create_artwork | 3 | 0.05% |
| judicial_action_hearing | 3 | 0.05% |
| engagement | 3 | 0.05% |
| Submitting_documents | 3 | 0.05% |
| strike | 3 | 0.05% |
| Limiting | 2 | 0.03% |
| Justifying | 2 | 0.03% |
| Carry_goods | 2 | 0.03% |
| Institutionalization | 2 | 0.03% |
| Incident | 2 | 0.03% |
| Robbery | 2 | 0.03% |
| contract_termination | 2 | 0.03% |
| Vocalizations | 2 | 0.03% |
| thank | 2 | 0.03% |
| infidelity | 2 | 0.03% |
| dismissal | 2 | 0.03% |
| price_decrease | 2 | 0.03% |
| rate_increase | 2 | 0.03% |
| Lighting | 2 | 0.03% |
| price_increase | 2 | 0.03% |
| Cure | 2 | 0.03% |
| Having_or_lacking_access | 2 | 0.03% |
| march | 2 | 0.03% |
| Suspicion | 2 | 0.03% |
| Releasing | 2 | 0.03% |
| Hiding_objects | 2 | 0.03% |
| Revenge | 2 | 0.03% |
| visit_on_set | 2 | 0.03% |
| Rewards_and_punishments | 2 | 0.03% |
| Commerce_buy | 1 | 0.02% |
| Kidnapping | 1 | 0.02% |
| Containing | 1 | 0.02% |
| ipo | 1 | 0.02% |
| Ratification | 1 | 0.02% |
| Theft | 1 | 0.02% |
| pregnancy | 1 | 0.02% |
| Cost | 1 | 0.02% |
| Exchange | 1 | 0.02% |
| Flood | 1 | 0.02% |

**val.** No event instances.

**test.** No event instances.

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 6,088 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.69 | 29.67% | 1.72 | 28.68% | 0.04% | 0 | 6,088 | 0 | 5,971 |
| val | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |
| test | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 5,795 | 1.14% | 4,345 | 12.80% | 12.34% | 1.77% | 0.60% | 16.20% |
| val | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |
| test | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 66 | 378 | 4,345 |
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
| StartDate | 1,798 | 5.76% |
| EquityHolder | 1,586 | 5.08% |
| StockCode | 1,404 | 4.50% |
| CompanyName | 1,201 | 3.85% |
| Location | 1,185 | 3.79% |
| StockAbbr | 1,183 | 3.79% |
| PledgedShares | 1,095 | 3.51% |
| Date | 1,008 | 3.23% |
| Pledgee | 912 | 2.92% |
| TotalHoldingShares | 877 | 2.81% |
| TradedShares | 829 | 2.65% |
| Casualties and Losses | 805 | 2.58% |
| TotalHoldingRatio | 785 | 2.51% |
| Reduction Amount | 712 | 2.28% |
| TotalPledgedShares | 624 | 2.00% |
| EndDate | 547 | 1.75% |
| Reducing Shareholder | 546 | 1.75% |
| ReleasedDate | 508 | 1.63% |
| Cause | 460 | 1.47% |
| Reduction Start Date | 441 | 1.41% |
| Company Name | 339 | 1.09% |
| Announcement Date | 325 | 1.04% |
| LaterHoldingShares | 286 | 0.92% |
| pledge_amount | 227 | 0.73% |
| AveragePrice | 200 | 0.64% |
| Affected Areas | 171 | 0.55% |
| Candidates and their Political Parties | 170 | 0.54% |
| Person | 170 | 0.54% |
| Recipient | 156 | 0.50% |
| pledgor | 155 | 0.50% |
| RepurchasedShares | 137 | 0.44% |
| Weapon and Equippment | 135 | 0.43% |
| Accepting Court | 132 | 0.42% |
| Loss Amount | 130 | 0.42% |
| Contest Participant | 129 | 0.41% |
| Belligerents | 129 | 0.41% |
| Organization | 127 | 0.41% |
| Protesters | 126 | 0.40% |
| Attacker | 122 | 0.39% |
| Accident Investigator | 118 | 0.38% |
| Protest Reason | 118 | 0.38% |
| HighestTradingPrice | 115 | 0.37% |
| protein | 108 | 0.35% |
| Vulnerability | 107 | 0.34% |
| pledge_start_date | 104 | 0.33% |
| Target | 104 | 0.33% |
| Individual_protein | 102 | 0.33% |
| Charged Crime | 99 | 0.32% |
| Frozen Amount | 94 | 0.30% |
| ClosingDate | 93 | 0.30% |
| Flight No. | 93 | 0.30% |
| Increase Amount | 92 | 0.29% |
| Position | 91 | 0.29% |
| Increasing Shareholder | 90 | 0.29% |
| Influence People | 88 | 0.28% |
| RepurchaseAmount | 85 | 0.27% |
| Affected Area | 85 | 0.27% |
| ruling_date | 84 | 0.27% |
| Accusation | 82 | 0.26% |
| DiseaseOrPhenotypicFeature | 82 | 0.26% |
| The Final Votes and Percentages | 79 | 0.25% |
| Strike Reason | 78 | 0.25% |
| Scheduled Landing Place | 77 | 0.25% |
| frozen_shareholder | 77 | 0.25% |
| Areas affected | 75 | 0.24% |
| Policy Content | 74 | 0.24% |
| Aid Agency | 72 | 0.23% |
| Aircraft Agency | 71 | 0.23% |
| Storm Hit Location | 69 | 0.22% |
| Husband | 68 | 0.22% |
| LegalInstitution | 67 | 0.21% |
| Freeze Start Date | 67 | 0.21% |
| Capabilities | 65 | 0.21% |
| Suspect | 63 | 0.20% |
| Wife | 63 | 0.20% |
| Habitat | 63 | 0.20% |
| ChemicalEntity | 62 | 0.20% |
| Solution | 62 | 0.20% |
| Disease | 61 | 0.20% |
| pledge_end_date | 60 | 0.19% |
| Strike Agency | 58 | 0.19% |
| Judgement Result/Prison Term | 55 | 0.18% |
| Microorganism | 53 | 0.17% |
| Anti-pollution People/Organizations | 53 | 0.17% |
| Crew | 52 | 0.17% |
| Accused People | 52 | 0.17% |
| Strike Outcome | 52 | 0.17% |
| FrozeShares | 51 | 0.16% |
| Precautionary Measure | 51 | 0.16% |
| System | 50 | 0.16% |
| Pollution Source | 49 | 0.16% |
| Historical Sites | 48 | 0.15% |
| Time | 48 | 0.15% |
| Acquiree | 48 | 0.15% |
| Policy Proposals | 48 | 0.15% |
| Survivors | 47 | 0.15% |
| Artifacts and Their Chronology | 47 | 0.15% |
| LowestTradingPrice | 46 | 0.15% |
| Riot Reason | 46 | 0.15% |
| Organization Name | 46 | 0.15% |
| Increase Start Date | 46 | 0.15% |
| Country Visited | 45 | 0.14% |
| Start Date | 45 | 0.14% |
| Game Name | 45 | 0.14% |
| Champions | 43 | 0.14% |
| Passengers | 43 | 0.14% |
| Participants | 43 | 0.14% |
| Strikers Status | 43 | 0.14% |
| GeneOrGeneProduct | 42 | 0.13% |
| Election Name | 41 | 0.13% |
| Speaker Status | 41 | 0.13% |
| Aid Supplies/Amount | 41 | 0.13% |
| Summit Theme | 41 | 0.13% |
| Appointee | 39 | 0.12% |
| Damaged Facility | 39 | 0.12% |
| PII | 38 | 0.12% |
| Discover Location | 37 | 0.12% |
| Child | 37 | 0.12% |
| protein-family | 37 | 0.12% |
| Location/Hospital | 36 | 0.12% |
| Magnitude | 36 | 0.12% |
| Economic Loss | 35 | 0.11% |
| Countries Withdrawing from the Organization | 35 | 0.11% |
| Speaker | 35 | 0.11% |
| Confirmed/Infected Cases | 35 | 0.11% |
| Spectacle Location | 35 | 0.11% |
| Area Affected | 34 | 0.11% |
| Patch | 34 | 0.11% |
| Record-breaking Project | 33 | 0.11% |
| Resignated Person | 32 | 0.10% |
| Device | 32 | 0.10% |
| Prosecutor | 32 | 0.10% |
| Perpetrators | 32 | 0.10% |
| Record Breaker | 31 | 0.10% |
| Score | 31 | 0.10% |
| Effective Date | 31 | 0.10% |
| Organization Industry | 30 | 0.10% |
| Weapon | 30 | 0.10% |
| Types of the Spectacle | 30 | 0.10% |
| The Agency Who Broke the Agreement | 30 | 0.10% |
| Deliberating Agency | 30 | 0.10% |
| Employment Agency | 29 | 0.09% |
| Host | 29 | 0.09% |
| Method | 29 | 0.09% |
| Time of the Case | 29 | 0.09% |
| Way for Spreading | 29 | 0.09% |
| Spectacle Start Time | 29 | 0.09% |
| Taking-off Place | 29 | 0.09% |
| Deceased | 28 | 0.09% |
| Grades | 28 | 0.09% |
| Marriage Duration | 28 | 0.09% |
| Freeze End Date | 28 | 0.09% |
| Person under Investigation | 27 | 0.09% |
| Court | 26 | 0.08% |
| OrganismTaxon | 26 | 0.08% |
| Poisoning Type | 26 | 0.08% |
| Investigative Agency | 26 | 0.08% |
| compensation_amount | 26 | 0.08% |
| Previous Record Holder | 25 | 0.08% |
| Death Cases | 25 | 0.08% |
| Strikers | 25 | 0.08% |
| Number of Damaged Houses | 25 | 0.08% |
| Death Reason | 24 | 0.08% |
| Software | 24 | 0.08% |
| Resign Reason | 24 | 0.08% |
| People/Organization who predicted the disaster | 23 | 0.07% |
| compensation_recipient | 23 | 0.07% |
| senior_personnel | 23 | 0.07% |
| senior_position | 23 | 0.07% |
| Boycotted Institutions | 23 | 0.07% |
| Storm Name | 23 | 0.07% |
| Number of Vehicles Involved in the Crash | 23 | 0.07% |
| Army | 23 | 0.07% |
| The Charged Crime | 23 | 0.07% |
| Profession | 22 | 0.07% |
| The Grades of the Previous Record Holder | 22 | 0.07% |
| Archaeologist | 22 | 0.07% |
| Data | 22 | 0.07% |
| Number of Influenced People | 22 | 0.07% |
| Winner | 22 | 0.07% |
| Participating Countries | 22 | 0.07% |
| Maximum Wind Speed | 22 | 0.07% |
| Predecessor | 21 | 0.07% |
| Malware | 21 | 0.07% |
| Gene/protein/RNA | 21 | 0.07% |
| Defense Lawyer | 21 | 0.07% |
| Epidemic Data Issuing Agency | 21 | 0.07% |
| Storm Hit Time | 21 | 0.07% |
| Phenotype | 21 | 0.07% |
| Train No. | 21 | 0.07% |
| Evidence | 21 | 0.07% |
| Age | 20 | 0.06% |
| Award Reason | 20 | 0.06% |
| Ship No. | 19 | 0.06% |
| Last Job of the Appointee | 19 | 0.06% |
| Released People | 19 | 0.06% |
| Visitor | 19 | 0.06% |
| Declarer | 19 | 0.06% |
| Protest Slogan | 19 | 0.06% |
| Arrest Time | 19 | 0.06% |
| Contracting Parties | 19 | 0.06% |
| End Time | 18 | 0.06% |
| Number of Destroyed Building | 18 | 0.06% |
| Pests | 18 | 0.06% |
| SequenceVariant | 18 | 0.06% |
| Award | 18 | 0.06% |
| Agreement Name | 18 | 0.06% |
| People | 17 | 0.05% |
| Competition Items | 17 | 0.05% |
| Conflict Duration | 17 | 0.05% |
| Number of Evacuated People | 17 | 0.05% |
| Storm Warning Level | 17 | 0.05% |
| Spacecraft | 17 | 0.05% |
| Outbreak Location | 17 | 0.05% |
| Investigating agency | 17 | 0.05% |
| Version | 17 | 0.05% |
| Rescue Organizer | 16 | 0.05% |
| Death Time | 16 | 0.05% |
| End Date | 16 | 0.05% |
| Arrested | 16 | 0.05% |
| Areas Affected | 16 | 0.05% |
| Judge | 15 | 0.05% |
| chemical | 15 | 0.05% |
| Epicenter | 15 | 0.05% |
| Other Loss | 15 | 0.05% |
| Head of the Merged Organization | 15 | 0.05% |
| GeneProtein | 15 | 0.05% |
| Treatment Method | 14 | 0.04% |
| Inauguration Time | 14 | 0.04% |
| Spacecraft Mission | 14 | 0.04% |
| Bank Name | 14 | 0.04% |
| Goal | 14 | 0.04% |
| Head of Investigation Team | 14 | 0.04% |
| Duration | 14 | 0.04% |
| Approver | 14 | 0.04% |
| Tribe | 14 | 0.04% |
| Symptoms | 13 | 0.04% |
| Money | 13 | 0.04% |
| Missings | 13 | 0.04% |
| Transportation | 13 | 0.04% |
| Last Time the Record was Broken | 12 | 0.04% |
| Court Time | 12 | 0.04% |
| Release Reason | 12 | 0.04% |
| Discover Time | 12 | 0.04% |
| Website | 12 | 0.04% |
| Attending Hospital | 12 | 0.04% |
| Electoral System | 12 | 0.04% |
| Fine Reason | 12 | 0.04% |
| Number of Victims | 12 | 0.04% |
| cell | 11 | 0.04% |
| Concept | 11 | 0.04% |
| Purpose | 11 | 0.04% |
| Response Measures | 11 | 0.04% |
| Casualty Count | 11 | 0.04% |
| Launch Date | 11 | 0.04% |
| Outbreak Date | 11 | 0.04% |
| Police | 11 | 0.04% |
| Merger Terms | 11 | 0.04% |
| Annoncement Date | 11 | 0.04% |
| News Release Agency | 11 | 0.04% |
| Work | 11 | 0.04% |
| Party Responsible for Pollution | 11 | 0.04% |
| Term of Office | 10 | 0.03% |
| The Final Seats Result | 10 | 0.03% |
| Symptom | 10 | 0.03% |
| Organization Members | 10 | 0.03% |
| Witness | 10 | 0.03% |
| Acquirer | 10 | 0.03% |
| Casualities and Losses | 10 | 0.03% |
| Voting method | 10 | 0.03% |
| Military Strength | 10 | 0.03% |
| Property Division | 10 | 0.03% |
| Appointment Time | 9 | 0.03% |
| Release Time | 9 | 0.03% |
| Election Goal | 9 | 0.03% |
| Acquisition amount | 9 | 0.03% |
| protein-domain | 9 | 0.03% |
| Mathematical Concept | 9 | 0.03% |
| Withdraw Reason | 9 | 0.03% |
| File | 9 | 0.03% |
| Launch Site | 9 | 0.03% |
| PaymentMethod | 9 | 0.03% |
| Sentencing Location | 9 | 0.03% |
| Volcano Name | 9 | 0.03% |
| Scale | 9 | 0.03% |
| Age of the Appointee | 9 | 0.03% |
| UnfrozeDate | 9 | 0.03% |
| Achievement | 9 | 0.03% |
| Susceptible Population | 9 | 0.03% |
| Number of Rescued People | 9 | 0.03% |
| Penalty Amount | 9 | 0.03% |
| DNA | 9 | 0.03% |
| CellType | 9 | 0.03% |
| Influenced Crops and Livelihood | 9 | 0.03% |
| Country | 8 | 0.03% |
| Child Custody | 8 | 0.03% |
| Archaeologist Organization | 8 | 0.03% |
| GPE | 8 | 0.03% |
| Mathematical Structure | 8 | 0.03% |
| Ways to Watch the Speech | 8 | 0.03% |
| Carrier Rocket | 8 | 0.03% |
| Research Agency | 8 | 0.03% |
| Economic loss | 8 | 0.03% |
| Weapon Used | 8 | 0.03% |
| Company Industry | 8 | 0.03% |
| Award Field | 8 | 0.03% |
| Head of Institution | 8 | 0.03% |
| Agreement Content | 8 | 0.03% |
| Stolen Amount | 8 | 0.03% |
| Fined Agency | 8 | 0.03% |
| Spectacle Duration | 8 | 0.03% |
| Criminal Evidence | 8 | 0.03% |
| organization | 8 | 0.03% |
| Candidate Scandal(Allegations of fraud, etc.) | 7 | 0.02% |
| Jail Time | 7 | 0.02% |
| protein-complex | 7 | 0.02% |
| experiment-tag | 7 | 0.02% |
| Number of Trapped People | 7 | 0.02% |
| Water Level | 7 | 0.02% |
| Prosecution Lawyer | 7 | 0.02% |
| Policy Name & Abbreviation | 7 | 0.02% |
| Tear Up Reason | 7 | 0.02% |
| Summit Name | 7 | 0.02% |
| Other Impact | 7 | 0.02% |
| Train Agency | 7 | 0.02% |
| reagent | 7 | 0.02% |
| Arrest Location | 7 | 0.02% |
| Death Rate | 7 | 0.02% |
| Regulatory Authority | 7 | 0.02% |
| Species | 7 | 0.02% |
| CellLine | 7 | 0.02% |
| Beneficiary | 7 | 0.02% |
| Successor | 7 | 0.02% |
| experimental-construct | 7 | 0.02% |
| algorithm | 7 | 0.02% |
| ai_model | 7 | 0.02% |
| Forecasting Agency | 7 | 0.02% |
| Rescue Tool or Method | 6 | 0.02% |
| Magnitude(Tsunami heights) | 6 | 0.02% |
| target | 6 | 0.02% |
| Prison Term | 6 | 0.02% |
| Number | 6 | 0.02% |
| Prison | 6 | 0.02% |
| Sequelae | 6 | 0.02% |
| Victim | 6 | 0.02% |
| Cooperative Agency | 6 | 0.02% |
| Turnout(who win who lost) | 6 | 0.02% |
| Employed Institution | 6 | 0.02% |
| Amount of Precipitation | 6 | 0.02% |
| Maximum Rainfall | 6 | 0.02% |
| Spokeswoman/Spokesman | 6 | 0.02% |
| Illness | 6 | 0.02% |
| Launch Result | 6 | 0.02% |
| State Before Death | 6 | 0.02% |
| Spectacle End Time | 6 | 0.02% |
| Anatomy | 6 | 0.02% |
| Hull Discovery Time | 6 | 0.02% |
| Hostage | 6 | 0.02% |
| Disaster-Stricken Farmland | 6 | 0.02% |
| Investigating Spokesperson | 6 | 0.02% |
| species | 6 | 0.02% |
| Best Way to Shoot | 6 | 0.02% |
| Start Time | 6 | 0.02% |
| Age of the Resignated Person | 5 | 0.02% |
| agent | 5 | 0.02% |
| Government Reaction | 5 | 0.02% |
| Announce Platform | 5 | 0.02% |
| Doctor and Medical Team | 5 | 0.02% |
| Damaged Property | 5 | 0.02% |
| Field of Study | 5 | 0.02% |
| Publication | 5 | 0.02% |
| Start_Date | 5 | 0.02% |
| Unemployed Rate | 5 | 0.02% |
| Affected Industries | 5 | 0.02% |
| Organization Leader | 5 | 0.02% |
| Commanders and their Position | 5 | 0.02% |
| Service Years | 5 | 0.02% |
| metric | 5 | 0.02% |
| Temporary Settlement | 5 | 0.02% |
| Number of Aftershocks | 5 | 0.02% |
| Aid Reason | 5 | 0.02% |
| Military Exercise | 5 | 0.02% |
| Literary Work | 5 | 0.02% |
| Art Form | 5 | 0.02% |
| Group | 5 | 0.02% |
| loss_function | 5 | 0.02% |
| Commanders of the Army | 4 | 0.01% |
| Last Outbreak Time | 4 | 0.01% |
| Storm Movement Speed | 4 | 0.01% |
| MVP | 4 | 0.01% |
| Head of the Institution | 4 | 0.01% |
| Spokenmen | 4 | 0.01% |
| Bill Drafting Agency | 4 | 0.01% |
| Storm Direction | 4 | 0.01% |
| Death Age | 4 | 0.01% |
| Strike Industry | 4 | 0.01% |
| environment | 4 | 0.01% |
| Launch Country | 4 | 0.01% |
| process | 4 | 0.01% |
| chemical_compound | 4 | 0.01% |
| Perpetrator | 4 | 0.01% |
| The State of the Volcano (Dormant or Active) | 4 | 0.01% |
| Funding | 4 | 0.01% |
| Tsunamis | 4 | 0.01% |
| Appointer | 4 | 0.01% |
| Art Movement | 4 | 0.01% |
| Lasting Time | 4 | 0.01% |
| Cured Cases | 4 | 0.01% |
| Number of Vaccinated People | 4 | 0.01% |
| fusion-protein | 4 | 0.01% |
| Postpone Time | 4 | 0.01% |
| Theory | 3 | 0.01% |
| Shipwreck Reason | 3 | 0.01% |
| Alternate Landing Place | 3 | 0.01% |
| Related Rivers or Lakes | 3 | 0.01% |
| Special Medicine | 3 | 0.01% |
| model | 3 | 0.01% |
| embodiment | 3 | 0.01% |
| chemical_group | 3 | 0.01% |
| physical_property | 3 | 0.01% |
| Mission Duration | 3 | 0.01% |
| Compensation | 3 | 0.01% |
| The Sentence Claimed by the Defense Lawyer | 3 | 0.01% |
| Detention Start Time | 3 | 0.01% |
| Invalid Date | 3 | 0.01% |
| Host Country | 3 | 0.01% |
| Disease Subtype | 3 | 0.01% |
| Medical Professional | 3 | 0.01% |
| Gene | 3 | 0.01% |
| Provisional Government | 3 | 0.01% |
| Suspected Cases | 3 | 0.01% |
| Damaged Crops & Livestock | 3 | 0.01% |
| Nation | 3 | 0.01% |
| Agreement Members | 3 | 0.01% |
| Sponsor | 3 | 0.01% |
| Warning Device | 2 | 0.01% |
| Principle | 2 | 0.01% |
| Model | 2 | 0.01% |
| Phenomenon | 2 | 0.01% |
| protein-motif | 2 | 0.01% |
| Complications | 2 | 0.01% |
| Fire Warning Level | 2 | 0.01% |
| CVE | 2 | 0.01% |
| dataset | 2 | 0.01% |
| modality | 2 | 0.01% |
| molecular_orbital | 2 | 0.01% |
| researcher_group | 2 | 0.01% |
| researcher | 2 | 0.01% |
| institutional_unit | 2 | 0.01% |
| Hull Location | 2 | 0.01% |
| Field | 2 | 0.01% |
| organelle | 2 | 0.01% |
| Bankrupt Business | 2 | 0.01% |
| Time Period | 2 | 0.01% |
| Political Entity | 2 | 0.01% |
| Writing System | 2 | 0.01% |
| Event | 2 | 0.01% |
| Number of People Hospitalized | 2 | 0.01% |
| amino-acid | 2 | 0.01% |
| Recovered Amount | 2 | 0.01% |
| Postpone Reason | 2 | 0.01% |
| Reasons for the Formation of the Historical sites | 2 | 0.01% |
| architecture_component | 2 | 0.01% |
| concept | 2 | 0.01% |
| Civilization | 2 | 0.01% |
| Law | 2 | 0.01% |
| technical_method | 2 | 0.01% |
| Agreement Validity Period | 2 | 0.01% |
| Storm Center Location | 2 | 0.01% |
| Protein_family_or_group | 2 | 0.01% |
| State of the Hull | 1 | 0.00% |
| Mathematical Result | 1 | 0.00% |
| Mathematical Theorem | 1 | 0.00% |
| Symbol | 1 | 0.00% |
| End_Date | 1 | 0.00% |
| Protein_complex | 1 | 0.00% |
| Countries Joined the Organization | 1 | 0.00% |
| task_category | 1 | 0.00% |
| capability | 1 | 0.00% |
| group | 1 | 0.00% |
| element | 1 | 0.00% |
| computational_method | 1 | 0.00% |
| software_package | 1 | 0.00% |
| Economists who Predicted the Crisis | 1 | 0.00% |
| parameter | 1 | 0.00% |
| protein-isoform | 1 | 0.00% |
| Language | 1 | 0.00% |
| Word | 1 | 0.00% |
| Object | 1 | 0.00% |
| Medium | 1 | 0.00% |
| The Sentence Claimed by the Prosecutor's Lawyer | 1 | 0.00% |
| Exit Conditions | 1 | 0.00% |
| Narrative | 1 | 0.00% |
| Text | 1 | 0.00% |
| Invention | 1 | 0.00% |
| Film | 1 | 0.00% |
| Protein | 1 | 0.00% |
| Role | 1 | 0.00% |
| Medical Test | 1 | 0.00% |
| Treatment | 1 | 0.00% |
| Community Group | 1 | 0.00% |
| Service | 1 | 0.00% |
| Program | 1 | 0.00% |
| How Many Times Get Married | 1 | 0.00% |
| Trapped Depth | 1 | 0.00% |
| Doctor | 1 | 0.00% |
| data_format | 1 | 0.00% |
| data_structure | 1 | 0.00% |
| publication_venue | 1 | 0.00% |
| Document | 1 | 0.00% |
| Wedding Venue | 1 | 0.00% |
| architectural_component | 1 | 0.00% |
| Head of the Government | 1 | 0.00% |
| Astronauts | 1 | 0.00% |
| Projects | 1 | 0.00% |
| Invited Person | 1 | 0.00% |
| Influenced People | 1 | 0.00% |
| Lawyer | 1 | 0.00% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| bind | 183 | 7.64% |
| Lives_In | 85 | 3.55% |
| PPI | 82 | 3.42% |
| Positive_Correlation | 52 | 2.17% |
| Association | 51 | 2.13% |
| Negative_Correlation | 25 | 1.04% |
| Exhibits | 24 | 1.00% |
| architect | 14 | 0.58% |
| creator | 13 | 0.54% |
| composer | 13 | 0.54% |
| located in the administrative territorial entity | 12 | 0.50% |
| parent taxon | 12 | 0.50% |
| position held | 12 | 0.50% |
| original broadcaster | 12 | 0.50% |
| conflict | 11 | 0.46% |
| historic county | 11 | 0.46% |
| country | 11 | 0.46% |
| performer | 11 | 0.46% |
| commanded by | 11 | 0.46% |
| crosses | 11 | 0.46% |
| member of | 11 | 0.46% |
| record label | 11 | 0.46% |
| publisher | 10 | 0.42% |
| platform | 10 | 0.42% |
| conferred by | 10 | 0.42% |
| astronaut mission | 10 | 0.42% |
| discoverer or inventor | 10 | 0.42% |
| movement | 10 | 0.42% |
| member of sports team | 10 | 0.42% |
| owned by | 10 | 0.42% |
| represented by | 10 | 0.42% |
| package management system | 10 | 0.42% |
| operator | 9 | 0.38% |
| country for sport | 9 | 0.38% |
| basin country | 9 | 0.38% |
| genre | 9 | 0.38% |
| place of detention | 9 | 0.38% |
| father | 9 | 0.38% |
| employer | 9 | 0.38% |
| director | 9 | 0.38% |
| item operated | 9 | 0.38% |
| heritage designation | 9 | 0.38% |
| present in work | 8 | 0.33% |
| participant in | 8 | 0.33% |
| award received | 8 | 0.33% |
| located in/on physical feature | 8 | 0.33% |
| participant | 8 | 0.33% |
| licensed to broadcast to | 8 | 0.33% |
| significant event | 8 | 0.33% |
| position played on team / speciality | 8 | 0.33% |
| owner of | 8 | 0.33% |
| partner in business or sport | 8 | 0.33% |
| league system | 8 | 0.33% |
| connects with | 8 | 0.33% |
| mountain range | 8 | 0.33% |
| founded by | 8 | 0.33% |
| military branch | 8 | 0.33% |
| mouth of the watercourse | 8 | 0.33% |
| production company | 8 | 0.33% |
| dialect of | 7 | 0.29% |
| field of work | 7 | 0.29% |
| educated at | 7 | 0.29% |
| inflows | 7 | 0.29% |
| shares border with | 7 | 0.29% |
| online service | 7 | 0.29% |
| family | 7 | 0.29% |
| cast member | 7 | 0.29% |
| crew member(s) | 7 | 0.29% |
| cathedral | 7 | 0.29% |
| ethnic group | 7 | 0.29% |
| member of political party | 7 | 0.29% |
| natural product of taxon | 7 | 0.29% |
| taxonomic type | 7 | 0.29% |
| connecting service | 7 | 0.29% |
| language used | 7 | 0.29% |
| stock market index | 7 | 0.29% |
| spouse | 7 | 0.29% |
| location of formation | 7 | 0.29% |
| set in period | 7 | 0.29% |
| located in the statistical territorial entity | 7 | 0.29% |
| head of government | 7 | 0.29% |
| characters | 7 | 0.29% |
| winner | 6 | 0.25% |
| constellation | 6 | 0.25% |
| newspaper format | 6 | 0.25% |
| illustrator | 6 | 0.25% |
| part of the series | 6 | 0.25% |
| genic_interaction | 6 | 0.25% |
| has use | 6 | 0.25% |
| maintained by | 6 | 0.25% |
| located on astronomical body | 6 | 0.25% |
| officeholder | 6 | 0.25% |
| parent organization | 6 | 0.25% |
| director / manager | 6 | 0.25% |
| member of the crew of | 6 | 0.25% |
| this zoological name is coordinate with | 6 | 0.25% |
| presenter | 6 | 0.25% |
| home venue | 6 | 0.25% |
| has immediate cause | 6 | 0.25% |
| notable work | 6 | 0.25% |
| pole position | 6 | 0.25% |
| location of first performance | 6 | 0.25% |
| located in the present-day administrative territorial entity | 6 | 0.25% |
| transport network | 6 | 0.25% |
| author | 6 | 0.25% |
| librettist | 6 | 0.25% |
| has part(s) of the class | 6 | 0.25% |
| place of publication | 6 | 0.25% |
| location of discovery | 6 | 0.25% |
| operating area | 6 | 0.25% |
| replaced by | 6 | 0.25% |
| has cause | 6 | 0.25% |
| connecting line | 6 | 0.25% |
| medical condition | 6 | 0.25% |
| form of creative work | 6 | 0.25% |
| diocese | 6 | 0.25% |
| replaces | 6 | 0.25% |
| league | 6 | 0.25% |
| first appearance | 6 | 0.25% |
| sport | 6 | 0.25% |
| has anatomical branch | 6 | 0.25% |
| legislated by | 5 | 0.21% |
| primary destinations | 5 | 0.21% |
| endemic to | 5 | 0.21% |
| grants | 5 | 0.21% |
| physically interacts with | 5 | 0.21% |
| biological process | 5 | 0.21% |
| facet of | 5 | 0.21% |
| edition or translation of | 5 | 0.21% |
| commander of (DEPRECATED) | 5 | 0.21% |
| distributed by | 5 | 0.21% |
| opposite of | 5 | 0.21% |
| located on street | 5 | 0.21% |
| fictional universe described in | 5 | 0.21% |
| time period | 5 | 0.21% |
| has effect | 5 | 0.21% |
| noble title | 5 | 0.21% |
| ammunition | 5 | 0.21% |
| start point | 5 | 0.21% |
| military rank | 5 | 0.21% |
| chromosome | 5 | 0.21% |
| narrative location | 5 | 0.21% |
| stock exchange | 5 | 0.21% |
| merged into | 5 | 0.21% |
| filming location | 5 | 0.21% |
| nominated for | 5 | 0.21% |
| has part(s) | 5 | 0.21% |
| uses | 5 | 0.21% |
| producer | 5 | 0.21% |
| public holiday | 5 | 0.21% |
| secretary general | 5 | 0.21% |
| product or material produced | 5 | 0.21% |
| highest judicial authority | 5 | 0.21% |
| dam | 5 | 0.21% |
| coextensive with | 5 | 0.21% |
| published in | 5 | 0.21% |
| head of state | 5 | 0.21% |
| interested in | 5 | 0.21% |
| child | 5 | 0.21% |
| musical conductor | 5 | 0.21% |
| instrument | 5 | 0.21% |
| worshipped by | 5 | 0.21% |
| anatomical location | 5 | 0.21% |
| surface played on | 5 | 0.21% |
| commemorates | 5 | 0.21% |
| cause of destruction | 5 | 0.21% |
| discography | 5 | 0.21% |
| standards body | 5 | 0.21% |
| currency | 5 | 0.21% |
| continent | 5 | 0.21% |
| location | 5 | 0.21% |
| legal form | 5 | 0.21% |
| occupation | 5 | 0.21% |
| school district | 5 | 0.21% |
| medical condition treated | 4 | 0.17% |
| target | 4 | 0.17% |
| after a work by | 4 | 0.17% |
| language of work or name | 4 | 0.17% |
| genetic association | 4 | 0.17% |
| terminus location | 4 | 0.17% |
| consecrator | 4 | 0.17% |
| objective of project or action | 4 | 0.17% |
| vessel class | 4 | 0.17% |
| commissioned by | 4 | 0.17% |
| overlies | 4 | 0.17% |
| brand | 4 | 0.17% |
| parent astronomical body | 4 | 0.17% |
| victory | 4 | 0.17% |
| reservoir created | 4 | 0.17% |
| has written for | 4 | 0.17% |
| IUCN conservation status | 4 | 0.17% |
| native language | 4 | 0.17% |
| authority | 4 | 0.17% |
| Comparison | 4 | 0.17% |
| sports discipline competed in | 4 | 0.17% |
| contains | 4 | 0.17% |
| voice type | 4 | 0.17% |
| domain of saint or deity | 4 | 0.17% |
| sponsor | 4 | 0.17% |
| carries | 4 | 0.17% |
| chairperson | 4 | 0.17% |
| takes place in fictional universe | 4 | 0.17% |
| director of photography | 4 | 0.17% |
| terminus | 4 | 0.17% |
| significant place | 4 | 0.17% |
| collection | 4 | 0.17% |
| main subject | 4 | 0.17% |
| operating system | 4 | 0.17% |
| country of origin | 4 | 0.17% |
| participating team | 4 | 0.17% |
| located in or next to body of water | 4 | 0.17% |
| writing language | 4 | 0.17% |
| located in time zone | 4 | 0.17% |
| party chief representative | 4 | 0.17% |
| drug or therapy used for treatment | 4 | 0.17% |
| space launch vehicle | 4 | 0.17% |
| editor | 4 | 0.17% |
| from narrative universe | 4 | 0.17% |
| armament | 4 | 0.17% |
| source of energy | 4 | 0.17% |
| Bind | 4 | 0.17% |
| basic form of government | 4 | 0.17% |
| made from material | 4 | 0.17% |
| contributed to creative work | 4 | 0.17% |
| contains the administrative territorial entity | 4 | 0.17% |
| origin of the watercourse | 4 | 0.17% |
| distribution format | 4 | 0.17% |
| named after | 4 | 0.17% |
| part of | 4 | 0.17% |
| track gauge | 3 | 0.13% |
| student of | 3 | 0.13% |
| place of death | 3 | 0.13% |
| developer | 3 | 0.13% |
| legislative body | 3 | 0.13% |
| competition class | 3 | 0.13% |
| office held by head of the organization | 3 | 0.13% |
| applies to jurisdiction | 3 | 0.13% |
| recommended unit of measurement | 3 | 0.13% |
| mother | 3 | 0.13% |
| defendant | 3 | 0.13% |
| readable file format | 3 | 0.13% |
| programming language | 3 | 0.13% |
| honorific prefix | 3 | 0.13% |
| territory overlaps | 3 | 0.13% |
| canonization status | 3 | 0.13% |
| has works in the collection | 3 | 0.13% |
| office held by head of state | 3 | 0.13% |
| official language | 3 | 0.13% |
| court | 3 | 0.13% |
| location of creation | 3 | 0.13% |
| described by source | 3 | 0.13% |
| incarnation of | 3 | 0.13% |
| interchange station | 3 | 0.13% |
| medical examination | 3 | 0.13% |
| qualifies for event | 3 | 0.13% |
| engine configuration | 3 | 0.13% |
| convicted of | 3 | 0.13% |
| performed at | 3 | 0.13% |
| coach of sports team | 3 | 0.13% |
| head coach | 3 | 0.13% |
| powered by | 3 | 0.13% |
| headquarters location | 3 | 0.13% |
| instruction set | 3 | 0.13% |
| software engine | 3 | 0.13% |
| drainage basin | 3 | 0.13% |
| tributary | 3 | 0.13% |
| archives at | 3 | 0.13% |
| practiced by | 3 | 0.13% |
| final event | 3 | 0.13% |
| structural engineer | 3 | 0.13% |
| home world | 3 | 0.13% |
| found in taxon | 3 | 0.13% |
| given name version for other gender | 3 | 0.13% |
| adjacent station | 3 | 0.13% |
| instance of | 3 | 0.13% |
| subject has role | 3 | 0.13% |
| lake on watercourse | 3 | 0.13% |
| candidate | 3 | 0.13% |
| foundational text | 3 | 0.13% |
| board member | 3 | 0.13% |
| mascot | 3 | 0.13% |
| writing system | 3 | 0.13% |
| minor planet group | 3 | 0.13% |
| Gram staining | 3 | 0.13% |
| work location | 3 | 0.13% |
| taxon rank | 3 | 0.13% |
| political alignment | 3 | 0.13% |
| has boundary | 3 | 0.13% |
| followed by | 3 | 0.13% |
| dedicated to | 3 | 0.13% |
| location of landing | 3 | 0.13% |
| league level above | 3 | 0.13% |
| religious order | 3 | 0.13% |
| broadcast by | 3 | 0.13% |
| separated from | 3 | 0.13% |
| said to be the same as | 3 | 0.13% |
| produced by | 3 | 0.13% |
| subclass of | 3 | 0.13% |
| substitute/deputy/replacement of office/officeholder | 3 | 0.13% |
| model item | 3 | 0.13% |
| enemy of | 3 | 0.13% |
| historical region | 3 | 0.13% |
| place served by transport hub | 3 | 0.13% |
| possible treatment | 3 | 0.13% |
| has subsidiary | 3 | 0.13% |
| has natural reservoir | 3 | 0.13% |
| has edition or translation | 3 | 0.13% |
| successful candidate | 3 | 0.13% |
| sibling | 3 | 0.13% |
| is the study of | 3 | 0.13% |
| chief executive officer | 2 | 0.08% |
| partially coincident with | 2 | 0.08% |
| studied by | 2 | 0.08% |
| depicted by | 2 | 0.08% |
| parent of this hybrid, breed, or cultivar | 2 | 0.08% |
| measured physical quantity | 2 | 0.08% |
| represents | 2 | 0.08% |
| linguistic typology | 2 | 0.08% |
| significant person | 2 | 0.08% |
| twinned administrative body | 2 | 0.08% |
| parent club | 2 | 0.08% |
| field of this occupation | 2 | 0.08% |
| vehicle normally used | 2 | 0.08% |
| youth wing | 2 | 0.08% |
| typically sells | 2 | 0.08% |
| media franchise | 2 | 0.08% |
| residence | 2 | 0.08% |
| anatomical branch of | 2 | 0.08% |
| cell component | 2 | 0.08% |
| input method | 2 | 0.08% |
| official religion | 2 | 0.08% |
| religion or worldview | 2 | 0.08% |
| perpetrator | 2 | 0.08% |
| indigenous to | 2 | 0.08% |
| derivative work | 2 | 0.08% |
| business division | 2 | 0.08% |
| designed to carry | 2 | 0.08% |
| depends on software | 2 | 0.08% |
| main regulatory text | 2 | 0.08% |
| lyrics by | 2 | 0.08% |
| co-driver | 2 | 0.08% |
| encodes | 2 | 0.08% |
| social classification | 2 | 0.08% |
| organizer | 2 | 0.08% |
| contributor to the creative work or subject | 2 | 0.08% |
| statement describes | 2 | 0.08% |
| gens | 2 | 0.08% |
| Christian liturgical rite | 2 | 0.08% |
| parallel street | 2 | 0.08% |
| copyright holder | 2 | 0.08% |
| follows | 2 | 0.08% |
| choreographer | 2 | 0.08% |
| speaker | 2 | 0.08% |
| radio format | 2 | 0.08% |
| wheel arrangement | 2 | 0.08% |
| allegiance | 2 | 0.08% |
| capital of | 2 | 0.08% |
| lake outflow | 2 | 0.08% |
| airline alliance | 2 | 0.08% |
| tracklist | 2 | 0.08% |
| manufacturer | 2 | 0.08% |
| cuisine | 2 | 0.08% |
| original language of film or TV show | 2 | 0.08% |
| director of archaeological fieldwork | 2 | 0.08% |
| coat of arms | 2 | 0.08% |
| sports season of league or competition | 2 | 0.08% |
| political ideology | 2 | 0.08% |
| patron saint | 2 | 0.08% |
| measurement scale | 2 | 0.08% |
| immediate cause of | 2 | 0.08% |
| executive body | 2 | 0.08% |
| vehicle | 2 | 0.08% |
| associated electoral district | 2 | 0.08% |
| this taxon is source of | 2 | 0.08% |
| killed by | 2 | 0.08% |
| used by | 1 | 0.04% |
| prime factor | 1 | 0.04% |
| editor-in-chief | 1 | 0.04% |
| mother house | 1 | 0.04% |
| proved by | 1 | 0.04% |
| implementation of | 1 | 0.04% |
| amended by | 1 | 0.04% |
| biological variant of | 1 | 0.04% |
| aspect ratio (W:H) | 1 | 0.04% |
| permanent resident of | 1 | 0.04% |
| flag | 1 | 0.04% |
| affiliation | 1 | 0.04% |
| site of astronomical discovery | 1 | 0.04% |
| programmer | 1 | 0.04% |
| honorific suffix | 1 | 0.04% |
| child astronomical body | 1 | 0.04% |
| copyright status | 1 | 0.04% |
| general classification of race participants | 1 | 0.04% |
| output method | 1 | 0.04% |
| relative | 1 | 0.04% |
| afflicts | 1 | 0.04% |
| substrate of | 1 | 0.04% |
| cause of death | 1 | 0.04% |
| exemplar of | 1 | 0.04% |
| territory claimed by | 1 | 0.04% |
| programming paradigm | 1 | 0.04% |
| in opposition to | 1 | 0.04% |
| screenwriter | 1 | 0.04% |
| place of birth | 1 | 0.04% |
| inspired by | 1 | 0.04% |
| officially opened by | 1 | 0.04% |
| game mode | 1 | 0.04% |
| superhuman feature or ability | 1 | 0.04% |
| route of administration | 1 | 0.04% |
| mod of | 1 | 0.04% |
| season | 1 | 0.04% |
| destroyed | 1 | 0.04% |
| charted in | 1 | 0.04% |
| delta of | 1 | 0.04% |
| via | 1 | 0.04% |
| league level below | 1 | 0.04% |
| capital | 1 | 0.04% |
| place of burial | 1 | 0.04% |
| country of registry | 1 | 0.04% |
| antiparticle | 1 | 0.04% |
| candidacy in election | 1 | 0.04% |
| day in year for periodic occurrence | 1 | 0.04% |
| solved by | 1 | 0.04% |
| does not have part | 1 | 0.04% |
| manifestation of | 1 | 0.04% |
| conjugate acid | 1 | 0.04% |
| sports league level | 1 | 0.04% |
| temporal range start | 1 | 0.04% |
| official podcast | 1 | 0.04% |
| central bank | 1 | 0.04% |
| advertises | 1 | 0.04% |
| health specialty | 1 | 0.04% |
| less than | 1 | 0.04% |
| appointed by | 1 | 0.04% |
| executive producer | 1 | 0.04% |
| dual to | 1 | 0.04% |
| business model | 1 | 0.04% |
| architectural style | 1 | 0.04% |
| members have occupation | 1 | 0.04% |
| office contested | 1 | 0.04% |
| Drug_Interaction | 1 | 0.04% |
| structure replaced by | 1 | 0.04% |
| next lower rank | 1 | 0.04% |
| voice actor | 1 | 0.04% |
| contributing factor of | 1 | 0.04% |
| wing configuration | 1 | 0.04% |
| demographics of topic | 1 | 0.04% |
| sexually homologous with | 1 | 0.04% |
| subdivision of this unit | 1 | 0.04% |
| lifestyle | 1 | 0.04% |
| copyright license | 1 | 0.04% |
| diaspora | 1 | 0.04% |
| active ingredient in | 1 | 0.04% |
| military casualty classification | 1 | 0.04% |
| penalty | 1 | 0.04% |
| underlies | 1 | 0.04% |
| signatory | 1 | 0.04% |
| set during recurring event | 1 | 0.04% |
| family name identical to this given name | 1 | 0.04% |
| radix | 1 | 0.04% |
| next higher rank | 1 | 0.04% |
| industry | 1 | 0.04% |
| product certification | 1 | 0.04% |
| student | 1 | 0.04% |
| ancestral home | 1 | 0.04% |
| source of income | 1 | 0.04% |
| catalog | 1 | 0.04% |
| sidekick of | 1 | 0.04% |
| has active ingredient | 1 | 0.04% |
| sex or gender | 1 | 0.04% |
| official symbol | 1 | 0.04% |
| exhibition history | 1 | 0.04% |
| organization directed by the office or position | 1 | 0.04% |
| points classification | 1 | 0.04% |
| unmarried partner | 1 | 0.04% |
| type of variable star | 1 | 0.04% |
| measures | 1 | 0.04% |
| launch contractor | 1 | 0.04% |
| has pet | 1 | 0.04% |
| anthem | 1 | 0.04% |
| writable file format | 1 | 0.04% |
| prohibits | 1 | 0.04% |
| has quality | 1 | 0.04% |
| theme music | 1 | 0.04% |
| display technology | 1 | 0.04% |
| office held by head of government | 1 | 0.04% |
| color | 1 | 0.04% |
| trophy awarded | 1 | 0.04% |
| house publication | 1 | 0.04% |
| designated as terrorist by | 1 | 0.04% |
| referee | 1 | 0.04% |
| family name | 1 | 0.04% |
| central bank/issuer | 1 | 0.04% |
| culture | 1 | 0.04% |
| economy of topic | 1 | 0.04% |
| mount | 1 | 0.04% |
| CPU | 1 | 0.04% |
| destination point | 1 | 0.04% |
| sexual orientation | 1 | 0.04% |
| statistical leader | 1 | 0.04% |
| has tense | 1 | 0.04% |
| main building contractor | 1 | 0.04% |
| fabrication method | 1 | 0.04% |
| comorbidity | 1 | 0.04% |
| disease transmission process | 1 | 0.04% |
| depicts | 1 | 0.04% |
| has list | 1 | 0.04% |
| character designer | 1 | 0.04% |
| shape | 1 | 0.04% |
| stepparent | 1 | 0.04% |
| GPU | 1 | 0.04% |
| doctoral student | 1 | 0.04% |
| history of topic | 1 | 0.04% |

### Classifications

**train, biotech_event.** Majority class 29.9% of 221 answers. The marginal is `true_label`, not the menu.

#### train biotech_event

| label | count | share |
| --- | --- | --- |
| executive statement | 66 | 29.86% |
| company description | 33 | 14.93% |
| alliance & partnership | 18 | 8.14% |
| other | 14 | 6.33% |
| product launching & presentation | 11 | 4.98% |
| new initiatives or programs | 10 | 4.52% |
| funding round | 9 | 4.07% |
| hiring | 9 | 4.07% |
| executive appointment | 7 | 3.17% |
| service & product providing | 6 | 2.71% |
| m&a | 6 | 2.71% |
| expanding geography | 4 | 1.81% |
| regulatory approval | 4 | 1.81% |
| article publication | 4 | 1.81% |
| support & philanthropy | 4 | 1.81% |
| product updates | 4 | 1.81% |
| new initiatives & programs | 2 | 0.90% |
| event organization | 2 | 0.90% |
| clinical trial sponsorship | 2 | 0.90% |
| foundation | 1 | 0.45% |
| investment in public company | 1 | 0.45% |
| expanding industry | 1 | 0.45% |
| department establishment | 1 | 0.45% |
| participation in an event | 1 | 0.45% |
| closing | 1 | 0.45% |

**train, chfinann_event.** Majority class 49.5% of 1,443 answers. The marginal is `true_label`, not the menu.

#### train chfinann_event

| label | count | share |
| --- | --- | --- |
| EquityPledge | 714 | 49.48% |
| EquityUnderweight | 296 | 20.51% |
| EquityOverweight | 287 | 19.89% |
| EquityRepurchase | 95 | 6.58% |
| EquityFreeze | 51 | 3.53% |

**train, docee_event.** Majority class 7.2% of 1,231 answers. The marginal is `true_label`, not the menu.

#### train docee_event

| label | count | share |
| --- | --- | --- |
| Armed Conflict | 89 | 7.23% |
| Air Crash | 77 | 6.26% |
| Protest | 60 | 4.87% |
| Election | 59 | 4.79% |
| Fire | 47 | 3.82% |
| Government Policy Changes | 41 | 3.33% |
| CommitCrime - Accuse | 40 | 3.25% |
| Riot | 40 | 3.25% |
| CommitCrime - Sentence | 34 | 2.76% |
| Sports Competition | 32 | 2.60% |
| Appoint_Inauguration | 30 | 2.44% |
| Storm | 27 | 2.19% |
| Disease Outbreaks | 27 | 2.19% |
| Famous Person - Death | 25 | 2.03% |
| Earthquakes | 25 | 2.03% |
| Withdraw from an Orgnization | 24 | 1.95% |
| Resignation_Dismissal | 22 | 1.79% |
| Floods | 22 | 1.79% |
| Gas Explosion | 21 | 1.71% |
| Strike | 21 | 1.71% |
| Mass Poisoning | 20 | 1.62% |
| CommitCrime - Investigate | 19 | 1.54% |
| Environment Pollution | 19 | 1.54% |
| Break Historical Records | 18 | 1.46% |
| Diplomatic Talks | 18 | 1.46% |
| New Archeological Discoveries | 17 | 1.38% |
| Organization Merge | 17 | 1.38% |
| Famous Person - Give a Speech | 17 | 1.38% |
| Train Collisions | 17 | 1.38% |
| New Wonders in Nature | 16 | 1.30% |
| CommitCrime - Arrest | 16 | 1.30% |
| CommitCrime - Release | 15 | 1.22% |
| Famine | 15 | 1.22% |
| Bank Robbery | 15 | 1.22% |
| Military Excercise | 15 | 1.22% |
| Insect Disaster | 14 | 1.14% |
| Shipwreck | 13 | 1.06% |
| New Achievements in Aerospace | 13 | 1.06% |
| Awards Ceremony | 13 | 1.06% |
| Famous Person - Sick | 12 | 0.97% |
| Tear Up Agreement | 12 | 0.97% |
| Road Crash | 12 | 0.97% |
| Mine Collapses | 11 | 0.89% |
| Financial Crisis | 11 | 0.89% |
| Tsunamis | 10 | 0.81% |
| Diplomatic Visit | 10 | 0.81% |
| Mudslides | 10 | 0.81% |
| Famous Person - Divorce | 9 | 0.73% |
| Volcano Eruption | 8 | 0.65% |
| Organization Closed | 8 | 0.65% |
| Sign Agreement | 8 | 0.65% |
| Droughts | 7 | 0.57% |
| Organization Established | 7 | 0.57% |
| Famous Person - Marriage | 7 | 0.57% |
| Famous Person - Recovered | 5 | 0.41% |
| Organization Fine | 5 | 0.41% |
| Regime Change | 4 | 0.32% |
| Financial Aid | 4 | 0.32% |
| Join in an Orgnization | 1 | 0.08% |

**train, docfee_event.** Majority class 35.8% of 947 answers. The marginal is `true_label`, not the menu.

#### train docfee_event

| label | count | share |
| --- | --- | --- |
| EquityUnderweight | 339 | 35.80% |
| EquityPledge | 153 | 16.16% |
| BankruptcyLiquidation | 144 | 15.21% |
| MajorAssetLoss | 122 | 12.88% |
| EquityFreeze | 66 | 6.97% |
| EquityOverweight | 45 | 4.75% |
| MajorSafetyAccident | 35 | 3.70% |
| ExecutiveDeath | 23 | 2.43% |
| MajorExternalPayout | 20 | 2.11% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| casualty_report | injured | 17,348 | 57.38% |
| casualty_report | dead | 12,658 | 41.86% |
| casualty_report | missing | 9,270 | 30.66% |
| casualty_report | location | 5,790 | 19.15% |
| casualty_report | record_metadata | natural/dead ×10,174, natural/injured ×5,973, natural/missing ×1,249, natural/location ×295 |  |
| record | status | 52 | 13.54% |
| record | location | 40 | 10.42% |
| record | name | 24 | 6.25% |
| record | address | 20 | 5.21% |
| record | title | 17 | 4.43% |
| record | price | 13 | 3.39% |
| record | description | 11 | 2.86% |
| record | amount | 10 | 2.60% |
| record | rating | 9 | 2.34% |
| record | author | 9 | 2.34% |
| record | city | 8 | 2.08% |
| record | date | 7 | 1.82% |
| record | funding_amount | 6 | 1.56% |
| record | patient_name | 6 | 1.56% |
| record | company | 6 | 1.56% |
| record | product_id | 6 | 1.56% |
| record | employee_id | 6 | 1.56% |
| record | claim_id | 6 | 1.56% |
| record | brand | 5 | 1.30% |
| record | state | 5 | 1.30% |
| record | policy_id | 5 | 1.30% |
| record | year | 5 | 1.30% |
| record | transaction_id | 5 | 1.30% |
| record | venue | 5 | 1.30% |
| record | review_id | 5 | 1.30% |
| record | project_id | 5 | 1.30% |
| record | company_name | 5 | 1.30% |
| record | event_id | 5 | 1.30% |
| record | department | 5 | 1.30% |
| record | medication | 4 | 1.04% |
| record | dosage | 4 | 1.04% |
| record | patient_id | 4 | 1.04% |
| record | policyholder | 4 | 1.04% |
| record | product_name | 4 | 1.04% |
| record | sender | 4 | 1.04% |
| record | country | 4 | 1.04% |
| record | submitter | 3 | 0.78% |
| record | source | 3 | 0.78% |
| record | patient_address | 3 | 0.78% |
| record | region | 3 | 0.78% |
| record | match | 3 | 0.78% |
| record | role | 3 | 0.78% |
| record | term_length | 3 | 0.78% |
| record | tail | 3 | 0.78% |
| record | head | 3 | 0.78% |
| record | features | 3 | 0.78% |
| record | quantity | 3 | 0.78% |
| record | amenities | 3 | 0.78% |
| record | age | 3 | 0.78% |
| record | temperature: Body temperature in degrees Fahrenheit | 3 | 0.78% |
| record | customer_name | 3 | 0.78% |
| record | asteroid.name | 3 | 0.78% |
| record | likes | 3 | 0.78% |
| record | manufacturer | 3 | 0.78% |
| record | ticket_id | 3 | 0.78% |
| record | category | 3 | 0.78% |
| record | home_team | 3 | 0.78% |
| record | opponent | 3 | 0.78% |
| record | points | 3 | 0.78% |
| record | origin | 3 | 0.78% |
| record | funding_source | 3 | 0.78% |
| record | product | 3 | 0.78% |
| record | user | 2 | 0.52% |
| record | investor | 2 | 0.52% |
| record | full_name | 2 | 0.52% |
| record | report_id | 2 | 0.52% |
| record | location.country | 2 | 0.52% |
| record | location.region | 2 | 0.52% |
| record | claim_id: Unique identifier for the insurance claim | 2 | 0.52% |
| record | university | 2 | 0.52% |
| record | severity | 2 | 0.52% |
| record | diagnosis | 2 | 0.52% |
| record | icd10code | 2 | 0.52% |
| record | appointment_id | 2 | 0.52% |
| record | beneficiary | 2 | 0.52% |
| record | sales_trend | 2 | 0.52% |
| record | products | 2 | 0.52% |
| record | recipient | 2 | 0.52% |
| record | reference_id | 2 | 0.52% |
| record | resolution_time | 2 | 0.52% |
| record | sku | 2 | 0.52% |
| record | seller | 2 | 0.52% |
| record | battery_life | 2 | 0.52% |
| record | target_audience | 2 | 0.52% |
| record | user_feedback | 2 | 0.52% |
| record | target | 2 | 0.52% |
| record | issue_description | 2 | 0.52% |
| record | location: City and state where the user is based | 2 | 0.52% |
| record | reporter | 2 | 0.52% |
| record | duration | 2 | 0.52% |
| record | provider | 2 | 0.52% |
| record | home_team: Name of the team playing at home | 2 | 0.52% |
| record | municipality.name | 2 | 0.52% |
| record | email | 2 | 0.52% |
| record | coverage_amount | 2 | 0.52% |
| record | monthly_premium | 2 | 0.52% |
| record | issue_date | 2 | 0.52% |
| record | patient_id: Unique identifier for the patient | 2 | 0.52% |
| record | away_team | 2 | 0.52% |
| record | return_id: Unique identifier for the return transaction | 2 | 0.52% |
| record | cpt_code | 2 | 0.52% |
| record | review_status | 2 | 0.52% |
| record | heart_rate: Heart rate in beats per minute | 2 | 0.52% |
| record | event_type: Category of the event | 2 | 0.52% |
| record | department: Academic department offering the course | 2 | 0.52% |
| record | instructor: Name of the faculty member teaching the course | 2 | 0.52% |
| record | timestamp | 2 | 0.52% |
| record | project_name | 2 | 0.52% |
| record | film_title | 2 | 0.52% |
| record | shares | 2 | 0.52% |
| record | funding | 2 | 0.52% |
| record | owner | 2 | 0.52% |
| record | expression_level | 2 | 0.52% |
| record | status: Current operational status of the device | 2 | 0.52% |
| record | color | 2 | 0.52% |
| record | release_date | 2 | 0.52% |
| record | organization | 2 | 0.52% |
| record | authors | 2 | 0.52% |
| record | content | 2 | 0.52% |
| record | score | 2 | 0.52% |
| record | player | 2 | 0.52% |
| record | organizations | 2 | 0.52% |
| record | comments | 2 | 0.52% |
| record | gender | 2 | 0.52% |
| record | clause_id | 2 | 0.52% |
| record | net_profit | 2 | 0.52% |
| record | delay_minutes | 2 | 0.52% |
| record | delay_reason | 2 | 0.52% |
| record | destination | 2 | 0.52% |
| record | flight_id | 2 | 0.52% |
| record | policyholder_name | 2 | 0.52% |
| record | education | 2 | 0.52% |
| record | current_status | 2 | 0.52% |
| record | mrn | 2 | 0.52% |
| record | team | 2 | 0.52% |
| record | vehicle | 2 | 0.52% |
| record | asteroid.type | 2 | 0.52% |
| record | material | 2 | 0.52% |
| record | device | 2 | 0.52% |
| record | asteroid.location | 2 | 0.52% |
| record | tags | 2 | 0.52% |
| record | status: Current status of the appointment | 2 | 0.52% |
| record | distributor_information.capabilities | 1 | 0.26% |
| record | services.maintenance | 1 | 0.26% |
| record | distributor_information.benefits | 1 | 0.26% |
| record | entity | 1 | 0.26% |
| record | claim_id: Unique identifier for the claim as referenced in correspondence | 1 | 0.26% |
| record | source_office: Clinic or office that submitted the claim | 1 | 0.26% |
| record | procedure_code: CPT code describing the medical service performed | 1 | 0.26% |
| record | downvotes | 1 | 0.26% |
| record | originalpostlink | 1 | 0.26% |
| record | content_length | 1 | 0.26% |
| record | comment_rank | 1 | 0.26% |
| record | judge | 1 | 0.26% |
| record | case_type | 1 | 0.26% |
| record | plaintiff | 1 | 0.26% |
| record | defendant | 1 | 0.26% |
| record | case_id | 1 | 0.26% |
| record | resolution_timeline | 1 | 0.26% |
| record | accuracy | 1 | 0.26% |
| record | election_type | 1 | 0.26% |
| record | candidate_location | 1 | 0.26% |
| record | candidate_name | 1 | 0.26% |
| record | planning | 1 | 0.26% |
| record | startup | 1 | 0.26% |
| record | funding_round | 1 | 0.26% |
| record | release_target | 1 | 0.26% |
| record | cpt_code: CPT code for the medical procedure performed | 1 | 0.26% |
| record | paid_amount: Amount actually received from insurance | 1 | 0.26% |
| record | diagnosis: ICD-10 code representing the medical condition | 1 | 0.26% |
| record | billed_amount: Total amount billed for the service | 1 | 0.26% |
| record | program_name | 1 | 0.26% |
| record | author.role | 1 | 0.26% |
| record | version | 1 | 0.26% |
| record | software | 1 | 0.26% |
| record | text | 1 | 0.26% |
| record | underwriting_model | 1 | 0.26% |
| record | ranking.location_type | 1 | 0.26% |
| record | location.province | 1 | 0.26% |
| record | booking_options.partnerships | 1 | 0.26% |
| record | price_range.per | 1 | 0.26% |
| record | booking_options.tripadvisor_role | 1 | 0.26% |
| record | prescriber_id | 1 | 0.26% |
| record | prescriber_name | 1 | 0.26% |
| record | place.type | 1 | 0.26% |
| record | place.location.province | 1 | 0.26% |
| record | place.demographics.0.year | 1 | 0.26% |
| record | place.demographics.0.population | 1 | 0.26% |
| record | place.location.department | 1 | 0.26% |
| record | place.name | 1 | 0.26% |
| record | budget | 1 | 0.26% |
| record | agency | 1 | 0.26% |
| record | technology | 1 | 0.26% |
| record | prescription_id | 1 | 0.26% |
| record | location: Property address including street, city, and state | 1 | 0.26% |
| record | baths | 1 | 0.26% |
| record | zip_code | 1 | 0.26% |
| record | beds | 1 | 0.26% |
| record | defendant_name | 1 | 0.26% |
| record | plaintiff_address | 1 | 0.26% |
| record | plaintiff_name | 1 | 0.26% |
| record | incident_id | 1 | 0.26% |
| record | date_reported | 1 | 0.26% |
| record | responsible_engineer | 1 | 0.26% |
| record | environment | 1 | 0.26% |
| record | id: Unique identifier assigned to each competitor | 1 | 0.26% |
| record | hq: Headquarters city and state of the company | 1 | 0.26% |
| record | date: Date of the competitor record entry | 1 | 0.26% |
| record | revenue: Annual revenue in millions of dollars | 1 | 0.26% |
| record | market_share: Percentage of market share held by the company | 1 | 0.26% |
| record | status: Current operational or strategic status of the company | 1 | 0.26% |
| record | death.date_julian | 1 | 0.26% |
| record | birth.date_gregorian | 1 | 0.26% |
| record | works.type | 1 | 0.26% |
| record | birth.date_julian | 1 | 0.26% |
| record | occupation | 1 | 0.26% |
| record | teacher | 1 | 0.26% |
| record | death.location | 1 | 0.26% |
| record | father.occupation | 1 | 0.26% |
| record | birth.location | 1 | 0.26% |
| record | contribution_id: Unique identifier for each contribution entry | 1 | 0.26% |
| record | address: Physical address of the donor as recorded | 1 | 0.26% |
| record | status: Current status of the contribution record | 1 | 0.26% |
| record | campaign_target: Name of the candidate or office being supported | 1 | 0.26% |
| record | pathway_name | 1 | 0.26% |
| record | splicing_regulation.current_research_status | 1 | 0.26% |
| record | clinical_observations.activation_status | 1 | 0.26% |
| record | splicing_regulation.impact_on_pathway_components.0.regulatory_mode | 1 | 0.26% |
| record | splicing_regulation.impact_on_pathway_components.0.component_description | 1 | 0.26% |
| record | admit_diagnosis | 1 | 0.26% |
| record | lengthofstay | 1 | 0.26% |
| record | discharge_condition | 1 | 0.26% |
| record | ticket_id: Internal ticket identifier for the RMA | 1 | 0.26% |
| record | serial: Unique identifier for the product unit | 1 | 0.26% |
| record | rma_number: Assigned number for the RMA case | 1 | 0.26% |
| record | cost: Repair cost, if any, as stated in the request | 1 | 0.26% |
| record | customer: Name of the product owner submitting the request | 1 | 0.26% |
| record | articleTitle | 1 | 0.26% |
| record | credit_hours | 1 | 0.26% |
| record | enrollment_status | 1 | 0.26% |
| record | meeting_days | 1 | 0.26% |
| record | enrollment_capacity | 1 | 0.26% |
| record | reasons | 1 | 0.26% |
| record | suggestions | 1 | 0.26% |
| record | attending_physician | 1 | 0.26% |
| record | bloodpressurediastolic | 1 | 0.26% |
| record | monitoring_method | 1 | 0.26% |
| record | bloodpressuresystolic | 1 | 0.26% |
| record | respiratory_rate | 1 | 0.26% |
| record | insurer | 1 | 0.26% |
| record | premium | 1 | 0.26% |
| record | policy_type | 1 | 0.26% |
| record | sentiment.opinion | 1 | 0.26% |
| record | sentiment.preference | 1 | 0.26% |
| record | quote.text | 1 | 0.26% |
| record | quote.author | 1 | 0.26% |
| record | favorite_players | 1 | 0.26% |
| record | edit_information.editor | 1 | 0.26% |
| record | edit_information.timestamp | 1 | 0.26% |
| record | subject | 1 | 0.26% |
| record | open_date | 1 | 0.26% |
| record | ticket_count | 1 | 0.26% |
| record | customer_id | 1 | 0.26% |
| record | filer | 1 | 0.26% |
| record | policy_number | 1 | 0.26% |
| record | problem_description | 1 | 0.26% |
| record | affected_network | 1 | 0.26% |
| record | access_method | 1 | 0.26% |
| record | affected_site | 1 | 0.26% |
| record | problem_start_time | 1 | 0.26% |
| record | alternative_access | 1 | 0.26% |
| record | production_location | 1 | 0.26% |
| record | closing_price | 1 | 0.26% |
| record | reason | 1 | 0.26% |
| record | genre | 1 | 0.26% |
| record | passenger_name: Name of the passenger on the flight | 1 | 0.26% |
| record | flight_no: Flight number as assigned by the airline | 1 | 0.26% |
| record | delay_minutes: Number of minutes flight is delayed | 1 | 0.26% |
| record | destination: Airport code where flight ends | 1 | 0.26% |
| record | email: Email address of the passenger | 1 | 0.26% |
| record | unique_id: Unique identifier for the delay entry | 1 | 0.26% |
| record | departure_date: Date flight departs from origin | 1 | 0.26% |
| record | sales_volume | 1 | 0.26% |
| record | date_added | 1 | 0.26% |
| record | singles | 1 | 0.26% |
| record | guideline_id: Unique identifier for the guideline document | 1 | 0.26% |
| record | author: Name of the medical professional who authored the guideline | 1 | 0.26% |
| record | institution: Hospital or organization where author is affiliated | 1 | 0.26% |
| record | recommendation: Specific treatment or intervention advised | 1 | 0.26% |
| record | issue_year: Year the guideline was published or released | 1 | 0.26% |
| record | prescriber | 1 | 0.26% |
| record | fda_code | 1 | 0.26% |
| record | buyer_name | 1 | 0.26% |
| record | store_reference | 1 | 0.26% |
| record | sound_quality | 1 | 0.26% |
| record | usability_score | 1 | 0.26% |
| record | audiofidelityscore | 1 | 0.26% |
| record | title: Event title describing the nature and context of the incident | 1 | 0.26% |
| record | date: Timestamp when the incident occurred | 1 | 0.26% |
| record | involved_parties: List of organizations or individuals involved in the incident | 1 | 0.26% |
| record | source: Media or organization reporting the incident | 1 | 0.26% |
| record | lead_reporter: Reporter providing coverage of the incident | 1 | 0.26% |
| record | geographic_code: Unique code identifying the incident's geographic region | 1 | 0.26% |
| record | underwriter | 1 | 0.26% |
| record | risk_score | 1 | 0.26% |
| record | sender: Email of the person sending the confirmation or submission | 1 | 0.26% |
| record | recipient: Email of the person receiving the confirmation or submission | 1 | 0.26% |
| record | type | 1 | 0.26% |
| record | force_type | 1 | 0.26% |
| record | material_efficiency | 1 | 0.26% |
| record | promotional_code | 1 | 0.26% |
| record | serial_number: Unique serial number for device tracking | 1 | 0.26% |
| record | serial_number | 1 | 0.26% |
| record | hotel_name | 1 | 0.26% |
| record | development_periods | 1 | 0.26% |
| record | see_also | 1 | 0.26% |
| record | stub_article.call_to_action | 1 | 0.26% |
| record | id: Unique identifier for the legislative vote | 1 | 0.26% |
| record | date: Date when the vote occurred | 1 | 0.26% |
| record | chamber: Legislative chamber where the vote took place | 1 | 0.26% |
| record | bill: Bill number and type identifier | 1 | 0.26% |
| record | sponsor: Name of the bill's sponsor | 1 | 0.26% |
| record | outcome: Final result of the vote | 1 | 0.26% |
| record | amount: Financial amount associated with the bill | 1 | 0.26% |
| record | location: Geographic location of the bill's jurisdiction | 1 | 0.26% |
| record | status: Current status of the bill | 1 | 0.26% |
| record | description: Summary of the bill's purpose and scope | 1 | 0.26% |
| record | relationships.0.entity | 1 | 0.26% |
| record | relationships.0.action | 1 | 0.26% |
| record | relationships.0.type | 1 | 0.26% |
| record | activities | 1 | 0.26% |
| record | bathrooms | 1 | 0.26% |
| record | bedrooms | 1 | 0.26% |
| record | location.distance | 1 | 0.26% |
| record | permit_holder | 1 | 0.26% |
| record | nepa_guidance | 1 | 0.26% |
| record | activity | 1 | 0.26% |
| record | additional_information | 1 | 0.26% |
| record | asset | 1 | 0.26% |
| record | industry_focus | 1 | 0.26% |
| record | user_location | 1 | 0.26% |
| record | internal_code | 1 | 0.26% |
| record | revenue_estimate | 1 | 0.26% |
| record | family | 1 | 0.26% |
| record | distribution | 1 | 0.26% |
| record | scientific_name | 1 | 0.26% |
| record | comment_content: Text of the comment including personal anecdotes and tool usage | 1 | 0.26% |
| record | author: Reddit username of the comment author | 1 | 0.26% |
| record | downvotes: Number of downvotes the comment received | 1 | 0.26% |
| record | tool: Productivity tool recommended in the comment | 1 | 0.26% |
| record | meeting_id | 1 | 0.26% |
| record | population.count | 1 | 0.26% |
| record | geography.coordinates.latitude | 1 | 0.26% |
| record | geography.coordinates.longitude | 1 | 0.26% |
| record | census_inclusion | 1 | 0.26% |
| record | service_type | 1 | 0.26% |
| record | total_amount | 1 | 0.26% |
| record | away_team: Name of the team playing away | 1 | 0.26% |
| record | away_score: Final score for the away team | 1 | 0.26% |
| record | game_id: Unique identifier for the game as listed in the schedule | 1 | 0.26% |
| record | cause: Reason for the damage such as flood or fire | 1 | 0.26% |
| record | municipality.name_local | 1 | 0.26% |
| record | municipality.location.region | 1 | 0.26% |
| record | municipality.location.country | 1 | 0.26% |
| record | demographics.population.0.count | 1 | 0.26% |
| record | demographics.population.0.year | 1 | 0.26% |
| record | concepts | 1 | 0.26% |
| record | location.city | 1 | 0.26% |
| record | hotel.name | 1 | 0.26% |
| record | hotel.star_rating | 1 | 0.26% |
| record | hire_date | 1 | 0.26% |
| record | salary | 1 | 0.26% |
| record | purpose | 1 | 0.26% |
| record | payee | 1 | 0.26% |
| record | account | 1 | 0.26% |
| record | owner_name | 1 | 0.26% |
| record | term_years | 1 | 0.26% |
| record | spo2: Oxygen saturation percentage | 1 | 0.26% |
| record | blood_pressure: Blood pressure in format systolic/diastolic | 1 | 0.26% |
| record | heart_rate: Number of heartbeats per minute | 1 | 0.26% |
| record | address: Street address including city and zip code | 1 | 0.26% |
| record | weather_conditions | 1 | 0.26% |
| record | league | 1 | 0.26% |
| record | location: Headquarters city and country of the company | 1 | 0.26% |
| record | lead_investor: Primary investor or consortium leading the round | 1 | 0.26% |
| record | useoffunds: Purpose or application of raised capital | 1 | 0.26% |
| record | funding_amount: Total amount raised in millions | 1 | 0.26% |
| record | funding_id: Unique identifier assigned to the funding round | 1 | 0.26% |
| record | company: Name of the technology company receiving funds | 1 | 0.26% |
| record | visit_count | 1 | 0.26% |
| record | province | 1 | 0.26% |
| record | error.possible_causes | 1 | 0.26% |
| record | error.message | 1 | 0.26% |
| record | event.person.name | 1 | 0.26% |
| record | event.new_club.contract_duration | 1 | 0.26% |
| record | event.person.age | 1 | 0.26% |
| record | event.new_club.name | 1 | 0.26% |
| record | event.opponent_club.name | 1 | 0.26% |
| record | event.person.position | 1 | 0.26% |
| record | event.person.role | 1 | 0.26% |
| record | first_contact_resolution_rate | 1 | 0.26% |
| record | priority | 1 | 0.26% |
| record | assigned_agent | 1 | 0.26% |
| record | support_code | 1 | 0.26% |
| record | subject_focus: Key subjects or pathways emphasized in the results | 1 | 0.26% |
| record | grade_distribution: Breakdown of grades achieved by students | 1 | 0.26% |
| record | location: Geographic region or city where the institution is located | 1 | 0.26% |
| record | publication_code: Unique identifier assigned to the result publication | 1 | 0.26% |
| record | asteroid.name_latin | 1 | 0.26% |
| record | asteroid.asteroid_type | 1 | 0.26% |
| record | asteroid.named_after.person.name | 1 | 0.26% |
| record | asteroid.named_after.person.name_native | 1 | 0.26% |
| record | asteroid.named_after.person.birth_year | 1 | 0.26% |
| record | asteroid.named_after.person.death_year | 1 | 0.26% |
| record | asteroid.named_after.person.occupation | 1 | 0.26% |
| record | asteroid.named_after.person.awards | 1 | 0.26% |
| record | submission_id: Unique identifier for the paper submission | 1 | 0.26% |
| record | author_name: Name of the paper's author | 1 | 0.26% |
| record | affiliation: Institutional affiliation of the author | 1 | 0.26% |
| record | paper_title: Title of the submitted research paper | 1 | 0.26% |
| record | submission_date: Date when the paper was submitted | 1 | 0.26% |
| record | word_count: Total number of words in the paper | 1 | 0.26% |
| record | fieldofstudy: Academic domain of the research | 1 | 0.26% |
| record | conference_name: Name of the conference where submitted | 1 | 0.26% |
| record | review_date: Date when the paper was reviewed | 1 | 0.26% |
| record | funding_source: Source of research funding | 1 | 0.26% |
| record | page_count: Number of pages in the paper | 1 | 0.26% |
| record | abstract_length: Length of the abstract in words | 1 | 0.26% |
| record | keywords | 1 | 0.26% |
| record | publication_time | 1 | 0.26% |
| record | office | 1 | 0.26% |
| record | update_notes | 1 | 0.26% |
| record | copyright | 1 | 0.26% |
| record | returner: Name of the person initiating the return | 1 | 0.26% |
| record | location: City or address where return was submitted | 1 | 0.26% |
| record | item: Description of the returned product | 1 | 0.26% |
| record | amount: Monetary value of the return in dollars | 1 | 0.26% |
| record | status: Current status of the return approval | 1 | 0.26% |
| record | unit_id | 1 | 0.26% |
| record | categories | 1 | 0.26% |
| record | context.region | 1 | 0.26% |
| record | context.demographic_challenge | 1 | 0.26% |
| record | context.strategic_approach | 1 | 0.26% |
| record | lead_agency.name | 1 | 0.26% |
| record | lead_agency.abbreviation | 1 | 0.26% |
| record | lead_agency.responsibilities | 1 | 0.26% |
| record | lead_agency.primary_goal | 1 | 0.26% |
| record | project.name | 1 | 0.26% |
| record | project.abbreviation | 1 | 0.26% |
| record | project.financials.budget | 1 | 0.26% |
| record | project.financials.funding_source | 1 | 0.26% |
| record | project.timeline.duration_years | 1 | 0.26% |
| record | project.scope.setting_type | 1 | 0.26% |
| record | project.objectives | 1 | 0.26% |
| record | project.operational_model.key_roles | 1 | 0.26% |
| record | project.operational_model.stakeholders.0.entity_type | 1 | 0.26% |
| record | project.operational_model.stakeholders.0.description | 1 | 0.26% |
| record | project.operational_model.stakeholders.1.entity_type | 1 | 0.26% |
| record | project.operational_model.stakeholders.1.description | 1 | 0.26% |
| record | project.operational_model.stakeholders.2.entity_type | 1 | 0.26% |
| record | project.operational_model.stakeholders.2.description | 1 | 0.26% |
| record | outcomes.monitored_metrics | 1 | 0.26% |
| record | outcomes.preliminary_findings | 1 | 0.26% |
| record | outcomes.qualitative_impact | 1 | 0.26% |
| record | exam_type | 1 | 0.26% |
| record | protocol | 1 | 0.26% |
| record | contributor_count | 1 | 0.26% |
| record | project_url | 1 | 0.26% |
| record | lastcommitdate | 1 | 0.26% |
| record | reviewer_name | 1 | 0.26% |
| record | distance.calculation_method | 1 | 0.26% |
| record | latin_name | 1 | 0.26% |
| record | distance.parsecs | 1 | 0.26% |
| record | apparent_magnitude.max | 1 | 0.26% |
| record | characteristics.effective_temperature.value | 1 | 0.26% |
| record | characteristics.type | 1 | 0.26% |
| record | distance.light_years | 1 | 0.26% |
| record | constellation | 1 | 0.26% |
| record | apparent_magnitude.unit | 1 | 0.26% |
| record | procedure_code | 1 | 0.26% |
| record | encounter_id: Unique identifier for the visit | 1 | 0.26% |
| record | name: Full name of the patient | 1 | 0.26% |
| record | address: Patient's residential address | 1 | 0.26% |
| record | visit_date: Date of the medical encounter | 1 | 0.26% |
| record | primary_diagnosis: Main medical condition diagnosed | 1 | 0.26% |
| record | icd_code: ICD-10 code for the primary diagnosis | 1 | 0.26% |
| record | bp_systolic: Systolic blood pressure in mmHg | 1 | 0.26% |
| record | bp_diastolic: Diastolic blood pressure in mmHg | 1 | 0.26% |
| record | medications: List of prescribed medications with dosing | 1 | 0.26% |
| record | lab_results: List of lab test results | 1 | 0.26% |
| record | symptoms: Patient-reported symptoms or complaints | 1 | 0.26% |
| record | organizer: Person or group hosting the event | 1 | 0.26% |
| record | name: Title or name of the event | 1 | 0.26% |
| record | location: Physical address or venue of the event | 1 | 0.26% |
| record | price: Entry cost for attending the event | 1 | 0.26% |
| record | status: Current phase of the event lifecycle | 1 | 0.26% |
| record | event_id: Unique identifier for each event | 1 | 0.26% |
| record | post_date | 1 | 0.26% |
| record | user_handle | 1 | 0.26% |
| record | status: Current enrollment or completion status | 1 | 0.26% |
| record | start_date: Date when the course begins | 1 | 0.26% |
| record | course_id: Unique identifier for the course as given in the XML | 1 | 0.26% |
| record | consultation.practice_areas | 1 | 0.26% |
| record | practice_areas | 1 | 0.26% |
| record | states_served | 1 | 0.26% |
| record | services | 1 | 0.26% |
| record | firm_name | 1 | 0.26% |
| record | client_focus | 1 | 0.26% |
| record | additional_services | 1 | 0.26% |
| record | experience.years | 1 | 0.26% |
| record | experience.description | 1 | 0.26% |
| record | total_funding | 1 | 0.26% |
| record | lead_researcher | 1 | 0.26% |
| record | review_type | 1 | 0.26% |
| record | review_score | 1 | 0.26% |
| record | github_link | 1 | 0.26% |
| record | outlet | 1 | 0.26% |
| record | circular.subject | 1 | 0.26% |
| record | circular.ministry | 1 | 0.26% |
| record | related_conditions | 1 | 0.26% |
| record | follow_up.average_duration.years | 1 | 0.26% |
| record | population.age_range.max | 1 | 0.26% |
| record | study_metadata.measurement_technology.full_name | 1 | 0.26% |
| record | statistical_results.diagnostic_metrics.sensitivity_at_threshold.t_score_threshold | 1 | 0.26% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.0.t_score | 1 | 0.26% |
| record | population.age_range.unit | 1 | 0.26% |
| record | study_metadata.measurement_technology.abbreviation | 1 | 0.26% |
| record | population.age_range.min | 1 | 0.26% |
| record | population.gender | 1 | 0.26% |
| record | statistical_results.diagnostic_metrics.sensitivity_at_threshold.percentage_of_fractures | 1 | 0.26% |
| record | statistical_results.predictive_model.value | 1 | 0.26% |
| record | conclusions | 1 | 0.26% |
| record | study_metadata.measurement_technology.anatomical_site | 1 | 0.26% |
| record | outcomes.fracture_events.0.count | 1 | 0.26% |
| record | study_metadata.study_type | 1 | 0.26% |
| record | symbol: Unique ticker identifier for the company | 1 | 0.26% |
| record | price: Closing stock price in dollars | 1 | 0.26% |
| record | sector: Industry sector classification | 1 | 0.26% |
| record | status: Direction of price movement | 1 | 0.26% |
| record | availability_date | 1 | 0.26% |
| record | availability_status | 1 | 0.26% |
| record | model_number | 1 | 0.26% |
| record | investor_list | 1 | 0.26% |
| record | briefing_event | 1 | 0.26% |
| record | follow_up_interest | 1 | 0.26% |
| record | pilot_hubs | 1 | 0.26% |
| record | identifier | 1 | 0.26% |
| record | staffing_change | 1 | 0.26% |
| record | listing_date | 1 | 0.26% |
| record | property_type | 1 | 0.26% |
| record | transcription_factor | 1 | 0.26% |
| record | gene_id | 1 | 0.26% |
| record | tissue | 1 | 0.26% |
| record | network_id | 1 | 0.26% |
| record | expression_units | 1 | 0.26% |
| record | network_context | 1 | 0.26% |
| record | id: Unique identifier for each rental listing | 1 | 0.26% |
| record | pet_policy: Pet policy including restrictions or fees | 1 | 0.26% |
| record | property_type: Type of housing unit | 1 | 0.26% |
| record | description: Brief description of unit features or conditions | 1 | 0.26% |
| record | address: Full property address including street and city | 1 | 0.26% |
| record | defense_systems | 1 | 0.26% |
| record | time_references | 1 | 0.26% |
| record | weapons | 1 | 0.26% |
| record | cities | 1 | 0.26% |
| record | booth_size | 1 | 0.26% |
| record | exhibitor_id | 1 | 0.26% |
| record | registration_date | 1 | 0.26% |
| record | contact_person | 1 | 0.26% |
| record | paymentduedate | 1 | 0.26% |
| record | vendor_name | 1 | 0.26% |
| record | account_code | 1 | 0.26% |
| record | owner: Name of the healthcare professional owning the device | 1 | 0.26% |
| record | location: Department or unit where device is currently located | 1 | 0.26% |
| record | device_id: Unique identifier assigned to the medical device | 1 | 0.26% |
| record | model: Manufacturer model name of the medical device | 1 | 0.26% |
| record | support_agent | 1 | 0.26% |
| record | ingredients | 1 | 0.26% |
| record | cooking_methods | 1 | 0.26% |
| record | boiling_instructions.salt_advice | 1 | 0.26% |
| record | boiling_instructions.cooking_time.min_minutes | 1 | 0.26% |
| record | boiling_instructions.cooking_time.max_minutes | 1 | 0.26% |
| record | grilling_instructions.method | 1 | 0.26% |
| record | grilling_instructions.cooking_time.minutes | 1 | 0.26% |
| record | serving_suggestions | 1 | 0.26% |
| record | personal_experience.raw_corn_experience | 1 | 0.26% |
| record | hotel_details.star_rating | 1 | 0.26% |
| record | ranking.location | 1 | 0.26% |
| record | hotel_details.location | 1 | 0.26% |
| record | booking_options.partners | 1 | 0.26% |
| record | booking_options.description | 1 | 0.26% |
| record | price_range.currency | 1 | 0.26% |
| record | hotel_details.name | 1 | 0.26% |
| record | population | 1 | 0.26% |
| record | medical condition | 1 | 0.26% |
| record | medical procedure | 1 | 0.26% |
| record | website | 1 | 0.26% |
| record | device model | 1 | 0.26% |
| record | developer | 1 | 0.26% |
| record | place | 1 | 0.26% |
| record | birth_information.birth_year | 1 | 0.26% |
| record | profession | 1 | 0.26% |
| record | biography.education.0.teacher | 1 | 0.26% |
| record | biography.education.4.institution | 1 | 0.26% |
| record | biography.awards.0.award_name | 1 | 0.26% |
| record | biography.concert_tours | 1 | 0.26% |
| record | biography.education.4.teacher | 1 | 0.26% |
| record | biography.collaborations | 1 | 0.26% |
| record | biography.education.3.institution | 1 | 0.26% |
| record | biography.education.2.teacher | 1 | 0.26% |
| record | biography.teaching_position.institution | 1 | 0.26% |
| record | biography.teaching_position.mentor | 1 | 0.26% |
| record | birth_information.birth_place.state | 1 | 0.26% |
| record | birth_information.birth_place.county | 1 | 0.26% |
| record | era_name | 1 | 0.26% |
| record | year_in_era | 1 | 0.26% |
| record | dynasty | 1 | 0.26% |
| record | album_title | 1 | 0.26% |
| record | artist | 1 | 0.26% |
| record | person | 1 | 0.26% |
| record | politician | 1 | 0.26% |
| record | award | 1 | 0.26% |
| record | hotel_experience.location.nearby_landmark | 1 | 0.26% |
| record | hotel_experience.location.city | 1 | 0.26% |
| record | hotel_experience.entertainment.tv.content | 1 | 0.26% |
| record | hotel_experience.additional_benefits | 1 | 0.26% |
| record | hotel_experience.suites.0.workspace | 1 | 0.26% |
| record | hotel_experience.suites.0.suite_type | 1 | 0.26% |
| record | hotel_experience.suites.0.bathroom_features | 1 | 0.26% |
| record | hotel_experience.suites.0.amenities | 1 | 0.26% |
| record | hotel_experience.suites.0.features | 1 | 0.26% |
| record | hotel_experience.entertainment.tv.screen_size | 1 | 0.26% |
| record | professions | 1 | 0.26% |
| record | key_concepts | 1 | 0.26% |
| record | count | 1 | 0.26% |
| record | vendor_address | 1 | 0.26% |
| record | recipient_name | 1 | 0.26% |
| record | recipient_email | 1 | 0.26% |
| record | recharge_amount | 1 | 0.26% |
| record | bonus_amount | 1 | 0.26% |
| record | product.specifications.display.resolution | 1 | 0.26% |
| record | product.launch_date | 1 | 0.26% |
| record | product.registration_count.count | 1 | 0.26% |
| record | product.specifications.memory.internal_storage | 1 | 0.26% |
| record | product.availability.currently_available | 1 | 0.26% |
| record | product.specifications.connectivity | 1 | 0.26% |
| record | product.price.1.currency | 1 | 0.26% |
| record | product.brand | 1 | 0.26% |
| record | product.specifications.camera.features | 1 | 0.26% |
| record | product.specifications.display.size | 1 | 0.26% |
| record | product.launch_location | 1 | 0.26% |
| record | product.model | 1 | 0.26% |
| record | product.registration_count.time_period | 1 | 0.26% |
| record | product.specifications.camera.rear | 1 | 0.26% |
| record | product.specifications.battery.capacity | 1 | 0.26% |
| record | product.specifications.memory.ram | 1 | 0.26% |
| record | tech_product | 1 | 0.26% |
| record | tweet_id | 1 | 0.26% |
| record | author_handle | 1 | 0.26% |
| record | country: Nationality of the athlete who won the medal | 1 | 0.26% |
| record | venue: Location where the event took place | 1 | 0.26% |
| record | result: Medal awarded to the athlete | 1 | 0.26% |
| record | discipline: Type of sport or competition such as swimming or skiing | 1 | 0.26% |
| record | athlete_name: Name of the athlete who achieved the result | 1 | 0.26% |
| record | surface | 1 | 0.26% |
| record | entries | 1 | 0.26% |
| record | tournament_name | 1 | 0.26% |
| record | winner_ranking | 1 | 0.26% |
| record | opponent_name | 1 | 0.26% |
| record | assigned_staff: Healthcare professional assigned to manage the device | 1 | 0.26% |
| record | purchase_date: Date the device was acquired | 1 | 0.26% |
| record | serial_number: Unique device identifier assigned by manufacturer | 1 | 0.26% |
| record | expiration_date: Date after which device is no longer valid | 1 | 0.26% |
| record | location: Physical ward or room where device is located | 1 | 0.26% |
| record | search_result_page.search_query | 1 | 0.26% |
| record | search_result_page.website_name | 1 | 0.26% |
| record | initial_text.negations | 1 | 0.26% |
| record | district | 1 | 0.26% |
| record | contribution_type | 1 | 0.26% |
| record | organizer | 1 | 0.26% |
| record | attendance | 1 | 0.26% |
| record | contact_email | 1 | 0.26% |
| record | report_id: Unique identifier assigned to the phishing report | 1 | 0.26% |
| record | status: Current status of the report | 1 | 0.26% |
| record | sender_domain: Domain of the spoofed email address | 1 | 0.26% |
| record | passengers_affected | 1 | 0.26% |
| record | targetagegroup: Age range of intended audience | 1 | 0.26% |
| record | campaign_id: Unique identifier for each promotional campaign | 1 | 0.26% |
| record | promo_code: Discount or offer code associated with the campaign | 1 | 0.26% |
| record | exam_date | 1 | 0.26% |
| record | findings | 1 | 0.26% |
| record | imaging_type | 1 | 0.26% |
| record | github_url | 1 | 0.26% |
| record | venue: Name of the stadium or venue where the event occurred | 1 | 0.26% |
| record | nation: Country of the athlete who won the medal | 1 | 0.26% |
| record | time: Official race time recorded in the event | 1 | 0.26% |
| record | medal: Type of medal won gold, silver, or bronze | 1 | 0.26% |
| record | recipient_jurisdiction: Jurisdiction such as city, state, or federal level | 1 | 0.26% |
| record | donor_location: City and state of the donor | 1 | 0.26% |
| record | contribution_id: Unique identifier for the contribution record | 1 | 0.26% |
| record | headlines | 1 | 0.26% |
| record | position | 1 | 0.26% |
| record | merchantcategorycode | 1 | 0.26% |
| record | locations | 1 | 0.26% |
| record | political_context | 1 | 0.26% |
| record | faq_references | 1 | 0.26% |
| record | navigation | 1 | 0.26% |
| record | footer_links | 1 | 0.26% |
| record | browse_topics | 1 | 0.26% |
| record | product_code | 1 | 0.26% |
| record | pricing | 1 | 0.26% |
| record | utilities_included | 1 | 0.26% |
| record | contact_method | 1 | 0.26% |
| record | preferredtenanttype | 1 | 0.26% |
| record | places | 1 | 0.26% |
| record | lead_person | 1 | 0.26% |
| record | legal_counsel | 1 | 0.26% |
| record | bracket_path | 1 | 0.26% |
| record | team_name | 1 | 0.26% |
| record | team_id | 1 | 0.26% |
| record | registered_on | 1 | 0.26% |
| record | service_impact | 1 | 0.26% |
| record | outage_start | 1 | 0.26% |
| record | outage_end | 1 | 0.26% |
| record | outage_code | 1 | 0.26% |
| record | root_cause | 1 | 0.26% |
| record | training_modules: List of specific training topics assigned to the employee | 1 | 0.26% |
| record | name: Full name of the new employee | 1 | 0.26% |
| record | employee_id: Unique identifier assigned to each new hire | 1 | 0.26% |
| record | claim_id: Unique identifier for each claim entry | 1 | 0.26% |
| record | policyholder_name: Name of the policyholder filing the claim | 1 | 0.26% |
| record | address: Full address of the policyholder | 1 | 0.26% |
| record | claim_date: Date when the incident occurred | 1 | 0.26% |
| record | amount_requested: Total amount requested for compensation | 1 | 0.26% |
| record | status: Current status of the claim | 1 | 0.26% |
| record | description: Brief description of the incident and damages | 1 | 0.26% |
| record | amount: Monetary value of the returned item | 1 | 0.26% |
| record | product_name: Name of the product returned | 1 | 0.26% |
| record | customer_name: Name of the customer initiating the return | 1 | 0.26% |
| record | order_id: Identifier of the original order associated with the return | 1 | 0.26% |
| record | effective_date | 1 | 0.26% |
| record | holder_name | 1 | 0.26% |
| record | creator | 1 | 0.26% |
| record | office_hours.4.day | 1 | 0.26% |
| record | office_hours.3.start_time | 1 | 0.26% |
| record | address.room | 1 | 0.26% |
| record | office_hours.2.start_time | 1 | 0.26% |
| record | office_hours.4.start_time | 1 | 0.26% |
| record | address.street | 1 | 0.26% |
| record | contact.fax | 1 | 0.26% |
| record | office_hours.4.end_time | 1 | 0.26% |
| record | office_hours.0.end_time | 1 | 0.26% |
| record | office_hours.1.start_time | 1 | 0.26% |
| record | office_hours.2.end_time | 1 | 0.26% |
| record | address.postal_code | 1 | 0.26% |
| record | office_hours.0.start_time | 1 | 0.26% |
| record | address.city | 1 | 0.26% |
| record | office_hours.0.day | 1 | 0.26% |
| record | line1 | 1 | 0.26% |
| record | treatment_arm | 1 | 0.26% |
| record | enrollment_date | 1 | 0.26% |
| record | contact_number | 1 | 0.26% |
| record | lastfollowup | 1 | 0.26% |
| record | insurance_provider | 1 | 0.26% |
| record | region_churn | 1 | 0.26% |
| record | call_id | 1 | 0.26% |
| record | dispute_status | 1 | 0.26% |
| record | owner_location | 1 | 0.26% |
| record | passenger_city | 1 | 0.26% |
| record | drug_name | 1 | 0.26% |
| record | indication | 1 | 0.26% |
| record | ndc | 1 | 0.26% |
| record | patient_gender | 1 | 0.26% |
| record | adverse_event | 1 | 0.26% |
| record | start_date | 1 | 0.26% |
| record | dateofincident | 1 | 0.26% |
| record | revenue | 1 | 0.26% |
| record | operating_margin | 1 | 0.26% |
| record | company_code | 1 | 0.26% |
| record | founding_year | 1 | 0.26% |
| record | company_id | 1 | 0.26% |
| record | key_challenge | 1 | 0.26% |
| record | outcome | 1 | 0.26% |
| record | prior_hospitalization | 1 | 0.26% |
| record | primary_diagnosis | 1 | 0.26% |
| record | icd_code | 1 | 0.26% |
| record | treatment_history | 1 | 0.26% |
| record | user_name: Name of the creator who posted the trend | 1 | 0.26% |
| record | video_id: Unique identifier for the video content | 1 | 0.26% |
| record | shares: Total number of shares the video received | 1 | 0.26% |
| record | comments: Total number of comments on the video | 1 | 0.26% |
| record | visitor_team | 1 | 0.26% |
| record | vaccination_id | 1 | 0.26% |
| record | administration_date | 1 | 0.26% |
| record | dose_number | 1 | 0.26% |
| record | stock | 1 | 0.26% |
| record | units_sold | 1 | 0.26% |
| record | template_substitution.prefix | 1 | 0.26% |
| record | advantages | 1 | 0.26% |
| record | template_substitution.examples | 1 | 0.26% |
| record | template_substitution.preview_code_example | 1 | 0.26% |
| record | templates_to_substitute | 1 | 0.26% |
| record | template_types.ordinary_template | 1 | 0.26% |
| record | outcomes_and_findings.prognosis.critical_success_factors | 1 | 0.26% |
| record | study_details.cohort.condition_studied | 1 | 0.26% |
| record | pathophysiology.etiology | 1 | 0.26% |
| record | pathophysiology.mechanism.secondary_event.anatomical_locations | 1 | 0.26% |
| record | outcomes_and_findings.prognosis.visual_recovery_potential | 1 | 0.26% |
| record | agent | 1 | 0.26% |
| record | game_id | 1 | 0.26% |
| record | player_name | 1 | 0.26% |
| record | steals | 1 | 0.26% |
| record | assists | 1 | 0.26% |
| record | player_id | 1 | 0.26% |
| record | disbursement_plan | 1 | 0.26% |
| record | gpa | 1 | 0.26% |
| record | capacity | 1 | 0.26% |
| record | sponsor | 1 | 0.26% |
| record | game_status | 1 | 0.26% |
| record | event_type | 1 | 0.26% |
| record | performance_notes | 1 | 0.26% |
| record | goals | 1 | 0.26% |
| record | entity_name | 1 | 0.26% |
| record | file_id | 1 | 0.26% |
| record | jurisdiction | 1 | 0.26% |
| record | person.publication_information.available_part | 1 | 0.26% |
| record | person.commentary_based_on | 1 | 0.26% |
| record | person.publication_information.commentary_with | 1 | 0.26% |
| record | person.publication_information.location | 1 | 0.26% |
| record | person.time_period.era | 1 | 0.26% |
| record | person.associated_with | 1 | 0.26% |
| record | statement_id | 1 | 0.26% |
| record | speaker | 1 | 0.26% |
| record | event_date | 1 | 0.26% |
| record | company.founding_location.city | 1 | 0.26% |
| record | company.location | 1 | 0.26% |
| record | company.transformation_date | 1 | 0.26% |
| record | company.name | 1 | 0.26% |
| record | company.description | 1 | 0.26% |
| record | company.establishment_date | 1 | 0.26% |
| record | use_case | 1 | 0.26% |
| record | review_length | 1 | 0.26% |
| record | compatibility | 1 | 0.26% |
| record | release_year | 1 | 0.26% |
| record | charging_time | 1 | 0.26% |
| record | warranty_period | 1 | 0.26% |
| record | post_conference_event | 1 | 0.26% |
| record | student_rate | 1 | 0.26% |
| record | early_bird_discount_ends | 1 | 0.26% |
| record | current_registration_count | 1 | 0.26% |
| record | keynote_sessions | 1 | 0.26% |
| record | panel_discussions | 1 | 0.26% |
| record | tracks | 1 | 0.26% |
| record | core_values | 1 | 0.26% |
| record | registration_type | 1 | 0.26% |
| record | workshop_date | 1 | 0.26% |
| record | backup_plan | 1 | 0.26% |
| record | pre_conference_workshop | 1 | 0.26% |
| record | venue_capacity | 1 | 0.26% |
| record | event_status | 1 | 0.26% |
| record | lightning_talks | 1 | 0.26% |
| record | expected_revenue | 1 | 0.26% |
| record | senior_rate | 1 | 0.26% |
| record | new_website_url | 1 | 0.26% |
| record | contact_name | 1 | 0.26% |
| record | event_name: Title of the event as listed | 1 | 0.26% |
| record | organizer_name: Name of the person or group organizing the event | 1 | 0.26% |
| record | location: Full address including street, city, and state | 1 | 0.26% |
| record | event_date: Date when the event will take place | 1 | 0.26% |
| record | event_status: Current status of the event | 1 | 0.26% |
| record | attendee_count: Number of registered attendees | 1 | 0.26% |
| record | city: City where the event is held | 1 | 0.26% |
| record | event_code: Unique code identifying the event | 1 | 0.26% |
| record | investor_leads | 1 | 0.26% |
| record | funds_use | 1 | 0.26% |
| record | critic_name | 1 | 0.26% |
| record | review_source | 1 | 0.26% |
| record | office_address | 1 | 0.26% |
| record | critic_location | 1 | 0.26% |
| record | beliefs | 1 | 0.26% |
| record | actions | 1 | 0.26% |
| record | affiliation | 1 | 0.26% |
| record | submission_date | 1 | 0.26% |
| record | research_submission | 1 | 0.26% |
| record | disclosure_id | 1 | 0.26% |
| record | donor_name | 1 | 0.26% |
| record | performance_rating | 1 | 0.26% |
| record | prior_experience | 1 | 0.26% |
| record | return_id | 1 | 0.26% |
| record | customer_address | 1 | 0.26% |
| record | originalordernumber | 1 | 0.26% |
| record | product_title | 1 | 0.26% |
| record | refund_amount | 1 | 0.26% |
| record | return_reason | 1 | 0.26% |
| record | shipping_method | 1 | 0.26% |
| record | processing_code | 1 | 0.26% |
| record | status: Updated clinical status like stable, unstable, critical, or resolved | 1 | 0.26% |
| record | patient_id: Unique identifier for the monitored patient | 1 | 0.26% |
| record | clinical_status: Current clinical condition such as normal or elevated | 1 | 0.26% |
| record | oxygen_saturation: Percentage of oxygen saturation in blood | 1 | 0.26% |
| record | reviewer | 1 | 0.26% |
| record | labels | 1 | 0.26% |
| record | research_project | 1 | 0.26% |
| record | principal_investigator | 1 | 0.26% |
| record | institution | 1 | 0.26% |
| record | project_status | 1 | 0.26% |
| record | phone | 1 | 0.26% |
| record | name: Full title of the course including subject and focus | 1 | 0.26% |
| record | credit_hours: Number of credit hours for the course | 1 | 0.26% |
| record | guidelines.distinction_results_interpretation | 1 | 0.26% |
| record | optional_information | 1 | 0.26% |
| record | research_framework_items | 1 | 0.26% |
| record | publication_date.city | 1 | 0.26% |
| record | publication_date.month | 1 | 0.26% |
| record | guidelines.adherence | 1 | 0.26% |
| record | organization.name | 1 | 0.26% |
| record | topics | 1 | 0.26% |
| record | claims_count | 1 | 0.26% |
| record | filing_location | 1 | 0.26% |
| record | physical_characteristics.observations.0.telescope | 1 | 0.26% |
| record | designation | 1 | 0.26% |
| record | orbital_characteristics.classification | 1 | 0.26% |
| record | discovery.discoverer | 1 | 0.26% |
| record | discovery.observatory.location.prefecture | 1 | 0.26% |
| record | physical_characteristics.observations.0.year | 1 | 0.26% |
| record | discovery.observatory.location.country | 1 | 0.26% |
| record | orbital_characteristics.semi_major_axis.unit | 1 | 0.26% |
| record | orbital_characteristics.perihelion.unit | 1 | 0.26% |
| record | regulatory_status.recognition_type | 1 | 0.26% |
| record | announcement_metadata.source_url | 1 | 0.26% |
| record | announcement_metadata.coverage_provider | 1 | 0.26% |
| record | regulatory_status.review_status | 1 | 0.26% |
| record | announcement_metadata.date | 1 | 0.26% |
| record | announcement_metadata.agency | 1 | 0.26% |
| record | boulder.start.foot_holds.1.foot | 1 | 0.26% |
| record | boulder.start.foot_holds.0.foot | 1 | 0.26% |
| record | boulder.start.hand_holds.1.description | 1 | 0.26% |
| record | boulder.start.hand_holds.0.description | 1 | 0.26% |
| record | boulder.start.foot_holds.0.description | 1 | 0.26% |
| record | boulder.start.hand_holds.1.hand | 1 | 0.26% |
| record | boulder.location.relative_position | 1 | 0.26% |
| record | boulder.location.distance_unit | 1 | 0.26% |
| record | boulder.location.reference_point | 1 | 0.26% |
| record | credits.graphic_design.0.copyright_year | 1 | 0.26% |
| record | credits.contributors.0.person | 1 | 0.26% |
| record | credits.contributors.0.role | 1 | 0.26% |
| record | credits.organization | 1 | 0.26% |
| record | credits.content_implementers | 1 | 0.26% |
| record | credits.contributors.1.role | 1 | 0.26% |
| record | credits.graphic_design.0.designer | 1 | 0.26% |
| record | credits.graphic_design.0.area | 1 | 0.26% |
| record | credits.advertising_revenue_use | 1 | 0.26% |
| record | credits.graphic_design.1.copyright_year | 1 | 0.26% |
| record | credits.creation_date | 1 | 0.26% |
| record | disease_context | 1 | 0.26% |
| record | other_youth_programs_with_congressional_charters | 1 | 0.26% |
| record | entity_name: Company name associated with filing | 1 | 0.26% |
| record | filing_id: Unique identifier assigned to the filing | 1 | 0.26% |
| record | revenue: Total reported revenue in millions | 1 | 0.26% |
| record | net_income: Total reported net income in millions | 1 | 0.26% |
| record | address: Physical address of the company | 1 | 0.26% |
| record | service.target_audience | 1 | 0.26% |
| record | service.responsible_person.affiliation | 1 | 0.26% |
| record | service.opening_hours.0.time | 1 | 0.26% |
| record | service.location.address | 1 | 0.26% |
| record | service.responsible_person.profession | 1 | 0.26% |
| record | service.opening_hours.0.day | 1 | 0.26% |
| record | service.location.building | 1 | 0.26% |
| record | service.interventions | 1 | 0.26% |
| record | service.responsible_person.role | 1 | 0.26% |
| record | service.contact_information.email | 1 | 0.26% |
| record | service.contact_information.phone | 1 | 0.26% |
| record | service.location.postal_code | 1 | 0.26% |
| record | chemical_name | 1 | 0.26% |
| record | cas_number | 1 | 0.26% |
| record | emergency_contact | 1 | 0.26% |
| record | establishment_year | 1 | 0.26% |
| record | departments | 1 | 0.26% |
| record | staff.research_assistant_count | 1 | 0.26% |
| record | staff.teaching_staff_count | 1 | 0.26% |
| record | dean.name | 1 | 0.26% |
| record | laboratories | 1 | 0.26% |
| record | dean.title | 1 | 0.26% |
| record | event | 1 | 0.26% |
| record | actions.0.target | 1 | 0.26% |
| record | actions.0.location | 1 | 0.26% |
| record | instruction | 1 | 0.26% |
| record | finding_id | 1 | 0.26% |
| record | time | 1 | 0.26% |
| record | access_duration | 1 | 0.26% |
| record | access_type | 1 | 0.26% |
| record | data_code | 1 | 0.26% |
| record | symbol: Stock ticker symbol as listed in the data | 1 | 0.26% |
| record | change_percent: Percentage change in stock price for the day | 1 | 0.26% |
| record | close_price: Final closing price of the stock | 1 | 0.26% |
| record | volume: Total shares traded on the day | 1 | 0.26% |
| record | market_cap: Market capitalization in billions | 1 | 0.26% |
| record | movement_reason: Brief reason for price movement or performance | 1 | 0.26% |
| record | product_id: Unique identifier assigned by the manufacturer | 1 | 0.26% |
| record | price: Product cost in US dollars | 1 | 0.26% |
| record | company: Manufacturer or developer of the product | 1 | 0.26% |
| record | launch_year: Year the product was first introduced | 1 | 0.26% |
| record | name: Product model name as branded | 1 | 0.26% |
| record | client_name | 1 | 0.26% |
| record | client_address | 1 | 0.26% |
| record | rate | 1 | 0.26% |
| record | location-gpe | 1 | 0.26% |
| record | email_id | 1 | 0.26% |
| record | vin | 1 | 0.26% |
| record | operation_hours | 1 | 0.26% |
| record | asteroid.orbital_characteristics.reference_plane | 1 | 0.26% |
| record | asteroid.designation_number | 1 | 0.26% |
| record | discovery_location | 1 | 0.26% |
| record | asteroid_number | 1 | 0.26% |
| record | discovery_date | 1 | 0.26% |
| record | discoverers | 1 | 0.26% |
| record | note | 1 | 0.26% |
| record | diagnosis_code | 1 | 0.26% |
| record | launch_date | 1 | 0.26% |
| record | target_market | 1 | 0.26% |
| record | RP | 1 | 0.26% |
| record | VBD | 1 | 0.26% |
| record | renewal_amount | 1 | 0.26% |
| record | asteroid.discovery_year | 1 | 0.26% |
| record | asteroid.orbital_characteristics.unit | 1 | 0.26% |
| record | asteroid.orbital_characteristics.inclination_unit | 1 | 0.26% |
| record | location: Physical address where course is held | 1 | 0.26% |
| record | duration: Number of weeks the course runs | 1 | 0.26% |
| record | credits: Number of academic credits assigned to the course | 1 | 0.26% |
| record | cost: Total cost of enrollment in US dollars | 1 | 0.26% |
| record | dates | 1 | 0.26% |
| record | registration_cap | 1 | 0.26% |
| record | registration_fee | 1 | 0.26% |
| record | early_bird_deadline | 1 | 0.26% |
| record | panel_topics | 1 | 0.26% |
| record | people | 1 | 0.26% |
| record | path | 1 | 0.26% |
| record | substances | 1 | 0.26% |
| record | materials | 1 | 0.26% |
| record | door | 1 | 0.26% |
| record | metaphors | 1 | 0.26% |
| record | objects | 1 | 0.26% |
| record | supernatural | 1 | 0.26% |
| record | decor | 1 | 0.26% |
| record | food | 1 | 0.26% |
| record | normal_range | 1 | 0.26% |
| record | test_name | 1 | 0.26% |
| record | value | 1 | 0.26% |
| record | lab_id | 1 | 0.26% |
| record | followupclinic_address: Physical address of follow-up clinic | 1 | 0.26% |
| record | followupprovider: Name of provider for follow-up visit | 1 | 0.26% |
| record | patient_mrn: Unique medical record number identifying the patient | 1 | 0.26% |
| record | treatments: Prescribed medications or therapies during stay | 1 | 0.26% |
| record | discharge_instructions: Written guidance given to patient post-discharge | 1 | 0.26% |
| record | magazine.article.page_breakdown.2.page_number | 1 | 0.26% |
| record | magazine.article.page_breakdown.9.advertising_percentage | 1 | 0.26% |
| record | magazine.article.page_breakdown.12.page_number | 1 | 0.26% |
| record | magazine.article.page_breakdown.2.advertising_percentage | 1 | 0.26% |
| record | magazine.article.page_breakdown.9.page_number | 1 | 0.26% |
| record | magazine.article.page_breakdown.3.advertisers | 1 | 0.26% |
| record | magazine.article.page_breakdown.16.advertising_percentage | 1 | 0.26% |
| record | magazine.article.page_breakdown.13.advertisers | 1 | 0.26% |
| record | magazine.article.page_breakdown.12.advertising_percentage | 1 | 0.26% |
| record | magazine.article.page_breakdown.5.advertising_percentage | 1 | 0.26% |
| record | magazine.article.page_breakdown.5.page_number | 1 | 0.26% |
| record | magazine.article.page_breakdown.14.advertisers | 1 | 0.26% |
| record | magazine.article.page_breakdown.13.advertising_percentage | 1 | 0.26% |
| record | magazine.article.page_breakdown.1.advertising_percentage | 1 | 0.26% |
| record | magazine.article.page_breakdown.8.advertisers | 1 | 0.26% |
| record | magazine.mentioned_article.author | 1 | 0.26% |
| record | magazine.article.page_breakdown.7.page_number | 1 | 0.26% |
| record | magazine.article.page_breakdown.10.page_number | 1 | 0.26% |
| record | magazine.name | 1 | 0.26% |
| record | magazine.article.page_breakdown.6.advertisers | 1 | 0.26% |
| record | magazine.article.page_breakdown.0.page_number | 1 | 0.26% |
| record | magazine.article.page_breakdown.2.advertisers | 1 | 0.26% |
| record | magazine.article.page_breakdown.5.advertisers | 1 | 0.26% |
| record | magazine.article.page_breakdown.15.advertisers | 1 | 0.26% |
| record | magazine.article.page_breakdown.4.page_number | 1 | 0.26% |
| record | magazine.article.page_breakdown.9.advertisers | 1 | 0.26% |
| record | magazine.article.page_breakdown.16.page_number | 1 | 0.26% |
| record | magazine.article.page_breakdown.10.advertisers | 1 | 0.26% |
| record | magazine.article.title | 1 | 0.26% |
| record | magazine.article.content_percentage | 1 | 0.26% |
| record | magazine.mentioned_article.year | 1 | 0.26% |
| record | magazine.article.page_breakdown.13.page_number | 1 | 0.26% |
| record | magazine.article.page_breakdown.3.page_number | 1 | 0.26% |
| record | magazine.article.page_breakdown.8.advertising_percentage | 1 | 0.26% |
| record | magazine.article.page_breakdown.0.advertisers | 1 | 0.26% |
| record | magazine.article.advertising_percentage | 1 | 0.26% |
| record | magazine.article.page_breakdown.14.advertising_percentage | 1 | 0.26% |
| record | magazine.article.page_breakdown.8.page_number | 1 | 0.26% |
| record | key_features | 1 | 0.26% |
| record | issues | 1 | 0.26% |
| record | founder | 1 | 0.26% |
| record | founded_year | 1 | 0.26% |
| record | document.location | 1 | 0.26% |
| record | document.counterfeiting_statistics.job_losses | 1 | 0.26% |
| record | document.strategy_elements.3.area | 1 | 0.26% |
| record | document.commissioners.1.portfolio | 1 | 0.26% |
| record | document.strategy_elements.6.area | 1 | 0.26% |
| record | document.key_issues | 1 | 0.26% |
| record | document.strategy_elements.0.area | 1 | 0.26% |
| record | document.commissioners.1.name | 1 | 0.26% |
| record | document.commissioners.0.name | 1 | 0.26% |
| record | document.strategy_elements.4.area | 1 | 0.26% |
| record | document.title | 1 | 0.26% |
| record | document.related_documents | 1 | 0.26% |
| record | document.reference_id | 1 | 0.26% |
| record | document.economic_impact.brand_value | 1 | 0.26% |
| record | document.strategy_elements.1.area | 1 | 0.26% |
| record | size_and_fit.measurements.2.unit | 1 | 0.26% |
| record | size_and_fit.model_size | 1 | 0.26% |
| record | size_and_fit.measurements.2.type | 1 | 0.26% |
| record | size_and_fit.size_guide | 1 | 0.26% |
| record | size_and_fit.measurements.1.unit | 1 | 0.26% |
| record | size_and_fit.model_height.value | 1 | 0.26% |
| record | size_and_fit.measurements.2.value | 1 | 0.26% |
| record | price.installment_option.currency | 1 | 0.26% |
| record | size_and_fit.length.value | 1 | 0.26% |
| record | size_and_fit.length.unit | 1 | 0.26% |
| record | size_and_fit.details | 1 | 0.26% |
| record | policytype | 1 | 0.26% |
| record | claim_id: Unique identifier for each claim | 1 | 0.26% |
| record | patient_name: Name of the patient filing the claim | 1 | 0.26% |
| record | patient_address: Full address including street, apt, city, state, zip | 1 | 0.26% |
| record | diagnosis_code: Standard code indicating the patient's condition | 1 | 0.26% |
| record | treatment_type: Type of medical treatment or procedure | 1 | 0.26% |
| record | claim_amount: Total monetary value of the claim | 1 | 0.26% |
| record | insurance_provider: Name of the insurance company covering the claim | 1 | 0.26% |
| record | status: Current processing status of the claim | 1 | 0.26% |
| record | payment_method: Method used to process payment for the claim | 1 | 0.26% |
| record | deductiblestatus: Whether patient's deductible has been met | 1 | 0.26% |
| record | co_payment: Patient's required co-payment amount | 1 | 0.26% |
| record | notes: Additional remarks or clinical context about the claim | 1 | 0.26% |
| record | drug_code | 1 | 0.26% |
| record | medicalrecordid | 1 | 0.26% |
| record | primary_provider: Name of the attending physician | 1 | 0.26% |
| record | admission_diagnosis: Primary diagnosis at admission | 1 | 0.26% |
| record | discharge_date: Date the patient was discharged | 1 | 0.26% |
| record | medications: List of prescribed medications | 1 | 0.26% |
| record | species_name | 1 | 0.26% |
| record | family.name | 1 | 0.26% |
| record | described_by.year | 1 | 0.26% |
| record | salary_range | 1 | 0.26% |
| record | company_size | 1 | 0.26% |
| record | bp | 1 | 0.26% |
| record | doctor | 1 | 0.26% |
| record | rr | 1 | 0.26% |
| record | key_achievement | 1 | 0.26% |
| record | review_period | 1 | 0.26% |
| record | service_type: Type of service impacted such as broadband or mobile | 1 | 0.26% |
| record | outage_end: Time when the outage ended in HH:MM AM/PM format | 1 | 0.26% |
| record | service_code: Internal code identifying the service type | 1 | 0.26% |
| record | ticket_number: Unique identifier for the outage ticket | 1 | 0.26% |
| record | outage_start: Time when the outage began in HH:MM AM/PM format | 1 | 0.26% |
| record | statue.name.french | 1 | 0.26% |
| record | statue.restoration.0.description | 1 | 0.26% |
| record | statue.restoration.0.funding.amount_raised.currency | 1 | 0.26% |
| record | statue.restoration.0.funding.source | 1 | 0.26% |
| record | statue.restoration.0.funding.contribution_range.currency | 1 | 0.26% |
| record | statue.discovery.discoverer.name | 1 | 0.26% |
| record | statue.discovery.location | 1 | 0.26% |
| record | statue.restoration.0.funding.contribution_range.min | 1 | 0.26% |
| record | statue.restoration.0.budget.currency | 1 | 0.26% |
| record | statue.discovery.date | 1 | 0.26% |
| record | statue.restoration.0.start_date | 1 | 0.26% |
| record | statue.transport_to_france | 1 | 0.26% |
| record | statue.public_display.stairs_completion_date | 1 | 0.26% |
| record | statue.location.museum | 1 | 0.26% |
| record | statue.material | 1 | 0.26% |
| record | statue.public_display.reopening_date | 1 | 0.26% |
| record | statue.location.city | 1 | 0.26% |
| record | statue.discovery.discoverer.title | 1 | 0.26% |
| record | statue.restoration.0.funding.number_of_contributors | 1 | 0.26% |
| record | statue.name.greek | 1 | 0.26% |
| record | statue.dedication | 1 | 0.26% |
| record | match_type | 1 | 0.26% |
| record | isbn | 1 | 0.26% |
| record | publication_volume | 1 | 0.26% |
| record | publication_issue | 1 | 0.26% |
| record | supply_duration | 1 | 0.26% |
| record | item_description.shipping.ready_to_ship_in.quantity | 1 | 0.26% |
| record | item_description.shipping.rush_details | 1 | 0.26% |
| record | item_description.weight.unit | 1 | 0.26% |
| record | item_description.dimensions.length | 1 | 0.26% |
| record | item_description.setup_charge.per | 1 | 0.26% |
| record | item_description.setup_charge.amount | 1 | 0.26% |
| record | item_description.weight.value | 1 | 0.26% |
| record | ticker | 1 | 0.26% |
| record | exchange | 1 | 0.26% |
| record | genus.period | 1 | 0.26% |
| record | genus.naming.year | 1 | 0.26% |
| record | genus.type_species | 1 | 0.26% |
| record | genus.naming.profession | 1 | 0.26% |
| record | genus.unique_features | 1 | 0.26% |
| record | genus.species.0.reason | 1 | 0.26% |
| record | genus.language | 1 | 0.26% |
| record | genus.naming.named_by | 1 | 0.26% |
| record | rating: Star rating out of 5 given by the reviewer | 1 | 0.26% |
| record | battery_life: Battery duration in hours under normal use | 1 | 0.26% |
| record | location: City and country of the reviewer | 1 | 0.26% |
| record | name: Product name as listed in inventory records | 1 | 0.26% |
| record | last_updated: Date when inventory was last modified | 1 | 0.26% |
| record | sku_id: Unique identifier for the product in inventory system | 1 | 0.26% |
| record | quantity: Current stock level of the SKU | 1 | 0.26% |
| record | location: Warehouse or hub where the SKU is stored | 1 | 0.26% |
| record | user: Name of the customer who provided the review | 1 | 0.26% |
| record | rating: Numerical rating from 1 to 5 given by the user | 1 | 0.26% |
| record | location: City and state or province where purchase occurred | 1 | 0.26% |
| record | shipping_cost: Cost to ship the audio device in US dollars | 1 | 0.26% |
| record | material: Material used in device construction | 1 | 0.26% |
| record | weight: Weight of the device in grams | 1 | 0.26% |
| record | isps | 1 | 0.26% |
| record | software_versions | 1 | 0.26% |
| record | theme | 1 | 0.26% |
| record | comment_id | 1 | 0.26% |
| record | event.description | 1 | 0.26% |
| record | event.organizer.name | 1 | 0.26% |
| record | organization_info.electricity_source | 1 | 0.26% |
| record | event.organizer.abbreviation | 1 | 0.26% |
| record | organization_info.funding | 1 | 0.26% |
| record | event.location.city | 1 | 0.26% |
| record | organization_info.facilities.generating_facilities | 1 | 0.26% |
| record | event.time.photo_op_end | 1 | 0.26% |
| record | event.activities | 1 | 0.26% |
| record | organization_info.website | 1 | 0.26% |
| record | event.name | 1 | 0.26% |
| record | event.time.photo_op_start | 1 | 0.26% |
| record | donations.recipient.type | 1 | 0.26% |
| record | event.time.end | 1 | 0.26% |
| record | event.location.address | 1 | 0.26% |
| record | delivery_method | 1 | 0.26% |
| record | newsletter_name | 1 | 0.26% |
| record | confirmation_code | 1 | 0.26% |
| record | subscription_amount | 1 | 0.26% |
| record | app_name | 1 | 0.26% |
| record | publication | 1 | 0.26% |
| record | policyholder_id: Unique identifier for the policyholder | 1 | 0.26% |
| record | policyholder_name: Full name of the claimant | 1 | 0.26% |
| record | claim_id: Unique identifier for the submitted claim | 1 | 0.26% |
| record | location: City where the treaty was signed | 1 | 0.26% |
| record | signatories: List of officials who co-signed the treaty | 1 | 0.26% |
| record | name: Unique identifier of the treaty agreement | 1 | 0.26% |
| record | format | 1 | 0.26% |
| record | date: Date when the reaction was reported | 1 | 0.26% |
| record | reaction: Observed adverse effect or symptom | 1 | 0.26% |
| record | status: Current status of the reaction | 1 | 0.26% |
| record | drug: Name and dosage form of the medication taken | 1 | 0.26% |
| record | duration: Number of days the treatment lasted for the reaction | 1 | 0.26% |
| record | patient_id: Unique identifier for the patient in the record | 1 | 0.26% |
| record | rma_number | 1 | 0.26% |
| record | failure_reason | 1 | 0.26% |
| record | return_date | 1 | 0.26% |
| record | announcement.divisions.0.star_of_the_week.player_name | 1 | 0.26% |
| record | announcement.presented_by | 1 | 0.26% |
| record | announcement.divisions.0.division_name | 1 | 0.26% |
| record | announcement.divisions.5.honorable_mentions.1.team | 1 | 0.26% |
| record | announcement.divisions.0.star_of_the_week.team | 1 | 0.26% |
| record | announcement.divisions.2.honorable_mentions.0.player_name | 1 | 0.26% |
| record | announcement.divisions.5.star_of_the_week.position | 1 | 0.26% |
| record | announcement.divisions.0.second_star.team | 1 | 0.26% |
| record | announcement.divisions.1.star_of_the_week.team | 1 | 0.26% |
| record | announcement.divisions.0.second_star.player_name | 1 | 0.26% |
| record | announcement.divisions.0.honorable_mentions.3.position | 1 | 0.26% |
| record | announcement.divisions.1.honorable_mentions.0.team | 1 | 0.26% |
| record | announcement.divisions.5.honorable_mentions.2.position | 1 | 0.26% |
| record | announcement.divisions.5.second_star.team | 1 | 0.26% |
| record | announcement.divisions.5.honorable_mentions.0.team | 1 | 0.26% |
| record | announcement.divisions.0.honorable_mentions.7.player_name | 1 | 0.26% |
| record | announcement.divisions.2.honorable_mentions.4.position | 1 | 0.26% |
| record | announcement.divisions.3.second_star.position | 1 | 0.26% |
| record | announcement.divisions.0.honorable_mentions.0.player_name | 1 | 0.26% |
| record | announcement.divisions.0.honorable_mentions.4.position | 1 | 0.26% |
| record | announcement.divisions.1.star_of_the_week.stats | 1 | 0.26% |
| record | announcement.divisions.0.honorable_mentions.5.team | 1 | 0.26% |
| record | announcement.divisions.2.division_name | 1 | 0.26% |
| record | announcement.divisions.4.honorable_mentions.2.team | 1 | 0.26% |
| record | announcement.divisions.0.honorable_mentions.7.position | 1 | 0.26% |
| record | announcement.divisions.0.honorable_mentions.1.player_name | 1 | 0.26% |
| record | announcement.divisions.3.honorable_mentions.1.player_name | 1 | 0.26% |
| record | announcement.divisions.4.honorable_mentions.0.player_name | 1 | 0.26% |
| record | announcement.divisions.4.honorable_mentions.1.player_name | 1 | 0.26% |
| record | announcement.divisions.0.honorable_mentions.1.team | 1 | 0.26% |
| record | announcement.divisions.1.honorable_mentions.0.position | 1 | 0.26% |
| record | announcement.divisions.3.star_of_the_week.position | 1 | 0.26% |
| record | announcement.divisions.5.star_of_the_week.player_name | 1 | 0.26% |
| record | announcement.divisions.2.honorable_mentions.1.position | 1 | 0.26% |
| record | announcement.divisions.0.honorable_mentions.2.player_name | 1 | 0.26% |
| record | announcement.divisions.1.star_of_the_week.position | 1 | 0.26% |
| record | announcement.divisions.3.honorable_mentions.0.player_name | 1 | 0.26% |
| record | announcement.divisions.0.second_star.position | 1 | 0.26% |
| record | announcement.divisions.0.honorable_mentions.0.position | 1 | 0.26% |
| record | announcement.divisions.1.star_of_the_week.player_name | 1 | 0.26% |
| record | announcement.divisions.3.division_name | 1 | 0.26% |
| record | announcement.divisions.5.honorable_mentions.1.position | 1 | 0.26% |
| record | announcement.divisions.4.second_star.team | 1 | 0.26% |
| record | announcement.divisions.2.honorable_mentions.0.team | 1 | 0.26% |
| record | announcement.divisions.3.second_star.team | 1 | 0.26% |
| record | announcement.divisions.4.division_name | 1 | 0.26% |
| record | announcement.divisions.1.honorable_mentions.1.team | 1 | 0.26% |
| record | announcement.divisions.0.honorable_mentions.4.player_name | 1 | 0.26% |
| record | announcement.divisions.2.honorable_mentions.0.position | 1 | 0.26% |
| record | announcement.divisions.0.honorable_mentions.7.team | 1 | 0.26% |
| record | announcement.divisions.0.honorable_mentions.6.player_name | 1 | 0.26% |
| record | announcement.divisions.1.second_star.player_name | 1 | 0.26% |
| record | announcement.divisions.2.honorable_mentions.2.position | 1 | 0.26% |
| record | announcement.week_ending | 1 | 0.26% |
| record | announcement.divisions.2.second_star.position | 1 | 0.26% |
| record | announcement.divisions.5.star_of_the_week.team | 1 | 0.26% |
| record | announcement.divisions.2.honorable_mentions.4.player_name | 1 | 0.26% |
| record | announcement.divisions.1.second_star.position | 1 | 0.26% |
| record | announcement.divisions.2.second_star.player_name | 1 | 0.26% |
| record | announcement.divisions.1.honorable_mentions.0.player_name | 1 | 0.26% |
| record | announcement.divisions.4.honorable_mentions.1.position | 1 | 0.26% |
| record | announcement.divisions.5.division_name | 1 | 0.26% |
| record | announcement.divisions.3.star_of_the_week.team | 1 | 0.26% |
| record | announcement.divisions.2.honorable_mentions.1.team | 1 | 0.26% |
| record | announcement.divisions.4.honorable_mentions.2.position | 1 | 0.26% |
| record | status: Current state of the incident | 1 | 0.26% |
| record | customer_name: Name of the affected customer | 1 | 0.26% |
| record | address: Full residential or commercial address of the customer | 1 | 0.26% |
| record | impact: Description of service impact including data loss or business effects | 1 | 0.26% |
| record | attendee_count | 1 | 0.26% |
| record | service_description | 1 | 0.26% |
| record | taxonomy.describer.affiliation.0.laboratory | 1 | 0.26% |
| record | description.distinguishing_features | 1 | 0.26% |
| record | species.etymology.named_after.0.location.city | 1 | 0.26% |
| record | distribution.habitat | 1 | 0.26% |
| record | taxonomy.describer.affiliation.0.university | 1 | 0.26% |
| record | taxonomy.describer.affiliation.0.location.country | 1 | 0.26% |
| record | description.size | 1 | 0.26% |
| record | description.color | 1 | 0.26% |
| record | description.width | 1 | 0.26% |
| record | taxonomy.subgenus | 1 | 0.26% |
| record | species.etymology.named_after.0.location.country | 1 | 0.26% |
| record | taxonomy.describer.affiliation.0.location.city | 1 | 0.26% |
| record | taxonomy.describer.name | 1 | 0.26% |
| record | taxonomy.year_described | 1 | 0.26% |
| record | taxonomy.type_materials | 1 | 0.26% |
| record | training_modules | 1 | 0.26% |
| record | buyer: Name of the customer who made the purchase | 1 | 0.26% |
| record | location: City and state where buyer resides | 1 | 0.26% |
| record | product: Product model identifier such as EBC-1234 or WAT-5678 | 1 | 0.26% |
| record | price: Price paid for the product in US dollars | 1 | 0.26% |
| record | rating: Star rating from 1 to 5 given by the buyer | 1 | 0.26% |
| record | shipping_method: Type of shipping used | 1 | 0.26% |
| record | tracking_number: Shipping tracking number assigned | 1 | 0.26% |
| record | order_id: Unique order identifier from the retailer | 1 | 0.26% |
| record | return_status: Whether the product was returned or kept | 1 | 0.26% |
| record | feedback: Additional comments or issues noted in review | 1 | 0.26% |
| record | symbols | 1 | 0.26% |
| record | determiners | 1 | 0.26% |
| record | verbs_past_participle | 1 | 0.26% |
| record | verbs_past_tense | 1 | 0.26% |
| record | prepositions | 1 | 0.26% |
| record | punctuation | 1 | 0.26% |
| record | possessive_endings | 1 | 0.26% |
| record | adjectives | 1 | 0.26% |
| record | singular_nouns | 1 | 0.26% |
| record | symptom | 1 | 0.26% |
| record | report_date | 1 | 0.26% |
| record | status: Current status of the exam result | 1 | 0.26% |
| record | school: Name of the school where the exam was administered | 1 | 0.26% |
| record | score: Percentage score achieved in the exam | 1 | 0.26% |
| record | subject: Subject area of the exam | 1 | 0.26% |
| record | compliance | 1 | 0.26% |
| record | claim_amount | 1 | 0.26% |
| record | currency_pair: Pair of currencies represented in the rate | 1 | 0.26% |
| record | id: Unique identifier for each exchange rate record | 1 | 0.26% |
| record | description: Brief explanation of market event influencing the rate | 1 | 0.26% |
| record | abstract_id | 1 | 0.26% |
| record | sample_size | 1 | 0.26% |
| record | bracket_id | 1 | 0.26% |
| record | result | 1 | 0.26% |
| record | teams | 1 | 0.26% |
| record | spoofed_domain | 1 | 0.26% |
| record | target_department | 1 | 0.26% |
| record | risk_classification | 1 | 0.26% |
| record | incident_type | 1 | 0.26% |
| record | disruption_duration | 1 | 0.26% |
| record | acquirer | 1 | 0.26% |
| record | acquired | 1 | 0.26% |
| record | value_original | 1 | 0.26% |
| record | regulatory_status | 1 | 0.26% |
| record | patents | 1 | 0.26% |
| record | employees | 1 | 0.26% |
| record | customer_contracts | 1 | 0.26% |
| record | integration_plan | 1 | 0.26% |
| record | platform_name | 1 | 0.26% |
| record | earnout | 1 | 0.26% |
| record | sustainability_commitment | 1 | 0.26% |
| record | asteroid.dedication.person.nationality | 1 | 0.26% |
| record | asteroid.dedication.person.occupation | 1 | 0.26% |
| record | steps | 1 | 0.26% |
| record | medications | 1 | 0.26% |
| record | symptoms | 1 | 0.26% |
| record | features: List of key technological or functional capabilities | 1 | 0.26% |
| record | price: Base price of the product in US dollars | 1 | 0.26% |
| record | regions: Geographic areas where the product is distributed or in use | 1 | 0.26% |
| record | name: Name of the product as introduced | 1 | 0.26% |
| record | isotope | 1 | 0.26% |
| record | insurance_id | 1 | 0.26% |
| record | amount_paid | 1 | 0.26% |
| record | dateofservice | 1 | 0.26% |
| record | location: Physical address of the product launch | 1 | 0.26% |
| record | product_code: Internal code assigned to the product | 1 | 0.26% |
| record | price: Selling price of the product in dollars | 1 | 0.26% |
| record | product: Name of the skincare or apparel product | 1 | 0.26% |
| record | flight_number | 1 | 0.26% |
| record | booking_reference | 1 | 0.26% |
| record | passenger_name | 1 | 0.26% |
| record | party_name | 1 | 0.26% |
| record | founder_name | 1 | 0.26% |
| record | founding_date | 1 | 0.26% |
| record | initial_funding | 1 | 0.26% |
| record | investment_source | 1 | 0.26% |
| record | industry | 1 | 0.26% |
| record | early_challenges | 1 | 0.26% |
| record | exit_status | 1 | 0.26% |
| record | model | 1 | 0.26% |
| record | user_rating | 1 | 0.26% |
| record | percentage | 1 | 0.26% |
| record | performance_time | 1 | 0.26% |
| record | audience_rating | 1 | 0.26% |
| record | festival_id | 1 | 0.26% |
| record | ticket_price | 1 | 0.26% |
| record | price: Current selling price of the product in USD | 1 | 0.26% |
| record | last_updated: Date when product record was last modified | 1 | 0.26% |
| record | warehouse_location: Geographic warehouse where product is stored | 1 | 0.26% |
| record | product.vendor | 1 | 0.26% |
| record | product.type | 1 | 0.26% |
| record | product.version | 1 | 0.26% |
| record | product.description | 1 | 0.26% |
| record | product.features | 1 | 0.26% |
| record | start_term | 1 | 0.26% |
| record | credits | 1 | 0.26% |
| record | hall | 1 | 0.26% |
| record | course_code | 1 | 0.26% |
| record | room | 1 | 0.26% |
| record | enrollment | 1 | 0.26% |
| record | thumbnail | 1 | 0.26% |
| record | passenger_address | 1 | 0.26% |
| record | departure_original | 1 | 0.26% |
| record | departure_actual | 1 | 0.26% |
| record | patent_number | 1 | 0.26% |
| record | zip | 1 | 0.26% |
| record | sunless_tanning.product_collection.products.0.type | 1 | 0.26% |
| record | sunless_tanning.product_collection.products.1.type | 1 | 0.26% |
| record | sunless_tanning.product_collection.products.0.ingredients | 1 | 0.26% |
| record | sunless_tanning.product_collection.products.0.description | 1 | 0.26% |
| record | sunless_tanning.product_collection.brand | 1 | 0.26% |
| record | sunless_tanning.product_collection.products.0.benefits | 1 | 0.26% |
| record | sunless_tanning.product_collection.benefits | 1 | 0.26% |
| record | sunless_tanning.product_collection.products.0.sizes | 1 | 0.26% |
| record | sunless_tanning.product_collection.products.1.name | 1 | 0.26% |
| record | sunless_tanning.product_collection.products.1.description | 1 | 0.26% |
| record | sunless_tanning.concerns | 1 | 0.26% |
| record | headquarters: City and country where company is headquartered | 1 | 0.26% |
| record | status: Current operational or strategic status | 1 | 0.26% |
| record | launch_year: Year the company launched its platform | 1 | 0.26% |
| record | geography.borders.2.direction | 1 | 0.26% |
| record | geography.borders.0.region | 1 | 0.26% |
| record | geography.borders.1.direction | 1 | 0.26% |
| record | infrastructure.refineries.0.refineryName | 1 | 0.26% |
| record | economy.gdpCurrency | 1 | 0.26% |
| record | infrastructure.highways.2.highwayName | 1 | 0.26% |
| record | infrastructure.roads.0.connections | 1 | 0.26% |
| record | infrastructure.airports.0.location | 1 | 0.26% |
| record | economy.economicBase | 1 | 0.26% |
| record | history.mergerDescription | 1 | 0.26% |
| record | infrastructure.airports.0.trafficVolume | 1 | 0.26% |
| record | infrastructure.airports.0.passengersYear | 1 | 0.26% |
| record | geography.rivers | 1 | 0.26% |
| record | infrastructure.highways.0.highwayName | 1 | 0.26% |
| record | economy.gdpRankAmongRegions | 1 | 0.26% |
| record | regionName | 1 | 0.26% |
| record | lowerLevelDivisions | 1 | 0.26% |
| record | infrastructure.refineries.0.closureDate | 1 | 0.26% |
| record | geography.borders.2.region | 1 | 0.26% |
| record | capital | 1 | 0.26% |
| record | infrastructure.refineries.0.status | 1 | 0.26% |
| record | regionNameBerber | 1 | 0.26% |
| record | history.establishmentDate | 1 | 0.26% |
| record | asteroid.designation | 1 | 0.26% |
| record | asteroid.alternative_designations | 1 | 0.26% |
| record | asteroid.discovery.date | 1 | 0.26% |
| record | asteroid.discovery.discoverer | 1 | 0.26% |
| record | asteroid.discovery.discovery_rank | 1 | 0.26% |
| record | asteroid.description.discovery_details.date | 1 | 0.26% |
| record | asteroid.description.discovery_details.observatory | 1 | 0.26% |
| record | asteroid.description.discovery_details.location.country | 1 | 0.26% |
| record | asteroid.description.discovery_details.location.state | 1 | 0.26% |
| record | asteroid.description.discovery_details.location.county | 1 | 0.26% |
| record | asteroid.description.discovery_details.project | 1 | 0.26% |
| record | asteroid.description.observations.last_observation_date | 1 | 0.26% |
| record | asteroid.description.observations.observation_period_years | 1 | 0.26% |
| record | asteroid.description.orbital_period_years | 1 | 0.26% |
| record | asteroid.orbital_characteristics.semi_major_axis | 1 | 0.26% |
| record | asteroid.orbital_characteristics.perihelion | 1 | 0.26% |
| record | asteroid.orbital_characteristics.eccentricity | 1 | 0.26% |
| record | asteroid.orbital_characteristics.inclination | 1 | 0.26% |
| record | asteroid.orbital_characteristics.classification | 1 | 0.26% |
| record | asteroid.external_links.0.title | 1 | 0.26% |
| record | asteroid.wiki_sky_links | 1 | 0.26% |
| record | restaurant_name | 1 | 0.26% |
| record | user_id | 1 | 0.26% |
| record | position_code: Unique identifier for the appointment | 1 | 0.26% |
| record | department: Government department or agency where the appointment occurs | 1 | 0.26% |
| record | room_type | 1 | 0.26% |
| record | nightly_rate | 1 | 0.26% |
| record | notes | 1 | 0.26% |
| record | notes: Additional context about the appointment or background | 1 | 0.26% |
| record | compensation: Annual salary or compensation amount | 1 | 0.26% |
| record | position: Job title or role within the department | 1 | 0.26% |
| record | date: Start date of the appointment | 1 | 0.26% |
| record | waterway | 1 | 0.26% |
| record | nationality | 1 | 0.26% |
| record | military unit | 1 | 0.26% |
| record | island | 1 | 0.26% |
| record | thumbnail_style | 1 | 0.26% |
| record | video_id | 1 | 0.26% |
| record | channel_name | 1 | 0.26% |
| record | description_length | 1 | 0.26% |
| record | word_count | 1 | 0.26% |
| record | due_date | 1 | 0.26% |
| record | reviewer_name: Full name of the reviewer | 1 | 0.26% |
| record | key_comment: Specific positive or negative observation about the stay | 1 | 0.26% |
| record | location: City and state where the hotel is located | 1 | 0.26% |
| record | review_id: Unique identifier for the review entry | 1 | 0.26% |
| record | product_offered | 1 | 0.26% |
| record | bench_points: Total points scored by bench players | 1 | 0.26% |
| record | key_moment: Descriptive note about a pivotal play or event in the game | 1 | 0.26% |
| record | steals: Total steals recorded during the game | 1 | 0.26% |
| record | game_id: Unique identifier for the game as provided in the passage | 1 | 0.26% |
| record | home_score: Final score of the home team | 1 | 0.26% |
| record | sender_name: Name of the sender in the correspondence | 1 | 0.26% |
| record | resolution_hours | 1 | 0.26% |
| record | resolution_status | 1 | 0.26% |
| record | record_metadata | natural/location ×10, natural/status ×9, natural/title ×8, natural/name ×6 |  |

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
