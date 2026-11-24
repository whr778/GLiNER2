# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replay_137k30 | train | 18,167 | 17,902 | 74,254 | 5,752 | 9,131 | 1,031 | 13,430 | 13,852 |
| replay_137k30 | val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| replay_137k30 | test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## replay_137k30

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 18,167 | 17,902 | 314 | 1312 | 22512 | 74,254 | 5,752 | 9,131 | 1,031 | 13,430 | 13,852 |
| val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 9,423 | 0 | 0 | 9,423 | 51.87% |
| zho | 8,172 | 0 | 0 | 8,172 | 44.98% |
| und | 377 | 0 | 0 | 377 | 2.08% |
| fra | 26 | 0 | 0 | 26 | 0.14% |
| rus | 22 | 0 | 0 | 22 | 0.12% |
| por | 16 | 0 | 0 | 16 | 0.09% |
| kor | 16 | 0 | 0 | 16 | 0.09% |
| hin | 13 | 0 | 0 | 13 | 0.07% |
| ita | 13 | 0 | 0 | 13 | 0.07% |
| spa | 13 | 0 | 0 | 13 | 0.07% |
| pol | 13 | 0 | 0 | 13 | 0.07% |
| nld | 11 | 0 | 0 | 11 | 0.06% |
| deu | 11 | 0 | 0 | 11 | 0.06% |
| tur | 11 | 0 | 0 | 11 | 0.06% |
| ukr | 10 | 0 | 0 | 10 | 0.06% |
| ara | 8 | 0 | 0 | 8 | 0.04% |
| fin | 3 | 0 | 0 | 3 | 0.02% |
| ces | 3 | 0 | 0 | 3 | 0.02% |
| nor | 2 | 0 | 0 | 2 | 0.01% |
| epo | 1 | 0 | 0 | 1 | 0.01% |
| wuu | 1 | 0 | 0 | 1 | 0.01% |
| ilo | 1 | 0 | 0 | 1 | 0.01% |
| cym | 1 | 0 | 0 | 1 | 0.01% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 0 |
| test_unique_inputs | 0 |
| train_records | 18,167 |
| train_unique_inputs | 17,902 |
| val_records | 0 |
| val_unique_inputs | 0 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | n/a | 74,254 / 0 |  |
| entities (per mention) | train / test | n/a | 74,254 / 0 |  |
| relations (per instance) | train / val | n/a | 5,752 / 0 |  |
| relations (per instance) | train / test | n/a | 5,752 / 0 |  |
| event types (per trigger mention) | train / val | n/a | 15,465 / 0 |  |
| event types (per trigger mention) | train / test | n/a | 15,465 / 0 |  |
| event roles (per argument) | train / val | n/a | 14,657 / 0 |  |
| event roles (per argument) | train / test | n/a | 14,657 / 0 |  |
| classifications (true_label) | train / val | n/a | 9,131 / 0 |  |
| classifications (true_label) | train / test | n/a | 9,131 / 0 |  |


### Events

**train.** 15,100 instances in 3,456 records, 0 triggerless. Trigger gold 13,430. Argument gold 13,852. Arguments per instance, mean 0.97. Trigger length in whitespace tokens, mean 1.37, single-token share 66.8%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Experiment | 1,037 | 6.71% |
| Support | 753 | 4.87% |
| Event | 554 | 3.58% |
| Hostile_encounter | 379 | 2.45% |
| Attack | 369 | 2.39% |
| Process_start | 368 | 2.38% |
| Deploy | 349 | 2.26% |
| Catastrophe | 336 | 2.17% |
| Competition | 306 | 1.98% |
| Causation | 302 | 1.95% |
| Cyber.DiscoverVulnerability | 261 | 1.69% |
| match_result | 250 | 1.62% |
| Exhibit | 222 | 1.44% |
| Killing | 218 | 1.41% |
| Motion | 218 | 1.41% |
| Social_event | 210 | 1.36% |
| Conquering | 190 | 1.23% |
| product_publish | 183 | 1.18% |
| Cyber.PatchVulnerability | 180 | 1.16% |
| Statement | 177 | 1.14% |
| Process_end | 176 | 1.14% |
| Cyber.Databreach | 161 | 1.04% |
| Control | 152 | 0.98% |
| Cyber.Ransom | 140 | 0.91% |
| Arriving | 133 | 0.86% |
| Cyber.Phishing | 133 | 0.86% |
| Coming_to_be | 129 | 0.83% |
| Hold | 126 | 0.81% |
| Self_motion | 125 | 0.81% |
| Death | 125 | 0.81% |
| Bodily_harm | 123 | 0.80% |
| Arranging | 123 | 0.80% |
| Creating | 121 | 0.78% |
| Destroying | 115 | 0.74% |
| Military_operation | 113 | 0.73% |
| Earnings_and_losses | 112 | 0.72% |
| death | 106 | 0.69% |
| Cause_change_of_position_on_a_scale | 105 | 0.68% |
| Getting | 105 | 0.68% |
| Becoming | 104 | 0.67% |
| Preventing_or_letting | 101 | 0.65% |
| Traveling | 100 | 0.65% |
| Defending | 99 | 0.64% |
| Cause_to_be_included | 98 | 0.63% |
| Presence | 97 | 0.63% |
| Participation | 95 | 0.61% |
| Know | 94 | 0.61% |
| Name_conferral | 93 | 0.60% |
| Supporting | 92 | 0.59% |
| Damaging | 90 | 0.58% |
| Accident | 88 | 0.57% |
| Escaping | 88 | 0.57% |
| Cause_change_of_strength | 87 | 0.56% |
| Motion_directional | 84 | 0.54% |
| judicial_action_arrest | 83 | 0.54% |
| Change_of_leadership | 83 | 0.54% |
| Change | 81 | 0.52% |
| Placing | 80 | 0.52% |
| Action | 80 | 0.52% |
| Giving | 79 | 0.51% |
| Removing | 78 | 0.50% |
| Manoeuvre | 78 | 0.50% |
| Request | 78 | 0.50% |
| resignation | 77 | 0.50% |
| Expressing_publicly | 77 | 0.50% |
| Judgment_communication | 74 | 0.48% |
| Supply | 70 | 0.45% |
| Choosing | 66 | 0.43% |
| Agree_or_refuse_to_act | 65 | 0.42% |
| Influence | 64 | 0.41% |
| Sending | 60 | 0.39% |
| Violence | 60 | 0.39% |
| Arrest | 57 | 0.37% |
| Recording | 55 | 0.36% |
| Using | 54 | 0.35% |
| Terrorism | 52 | 0.34% |
| Injure | 51 | 0.33% |
| Bringing | 50 | 0.32% |
| advancement | 50 | 0.32% |
| Perception_active | 49 | 0.32% |
| Building | 48 | 0.31% |
| Recovering | 48 | 0.31% |
| Conflict | 46 | 0.30% |
| championship_win | 46 | 0.30% |
| Assistance | 44 | 0.28% |
| Departing | 44 | 0.28% |
| Reporting | 43 | 0.28% |
| divorce | 43 | 0.28% |
| Communication | 42 | 0.27% |
| Aiming | 42 | 0.27% |
| Deciding | 41 | 0.27% |
| Cause_to_make_progress | 41 | 0.27% |
| Hindering | 41 | 0.27% |
| Expansion | 41 | 0.27% |
| judicial_action_summon | 40 | 0.26% |
| marriage | 39 | 0.25% |
| opening | 38 | 0.25% |
| Besieging | 38 | 0.25% |
| Manufacturing | 38 | 0.25% |
| Use_firearm | 37 | 0.24% |
| Come_together | 37 | 0.24% |
| Dispersal | 37 | 0.24% |
| Receiving | 37 | 0.24% |
| product_recall | 36 | 0.23% |
| Scrutiny | 36 | 0.23% |
| product_release | 35 | 0.23% |
| Committing_crime | 35 | 0.23% |
| Surrendering | 35 | 0.23% |
| Achieve | 34 | 0.22% |
| Publishing | 34 | 0.22% |
| Protest | 33 | 0.21% |
| Becoming_a_member | 33 | 0.21% |
| Openness | 33 | 0.21% |
| Traffic Accident | 32 | 0.21% |
| joined | 32 | 0.21% |
| price_limit_up | 32 | 0.21% |
| Legal_rulings | 32 | 0.21% |
| Cause_to_amalgamate | 31 | 0.20% |
| Change_event_time | 31 | 0.20% |
| Being_in_operation | 28 | 0.18% |
| Response | 27 | 0.17% |
| Quarreling | 27 | 0.17% |
| Collaboration | 27 | 0.17% |
| Sign_agreement | 27 | 0.17% |
| Check | 26 | 0.17% |
| departure | 26 | 0.17% |
| Confronting_problem | 26 | 0.17% |
| sale_acquisition | 26 | 0.17% |
| Connect | 25 | 0.16% |
| GiveUp | 25 | 0.16% |
| Coming_to_believe | 24 | 0.16% |
| judicial_action_fine | 24 | 0.16% |
| judicial_action_imprisonment | 23 | 0.15% |
| financing | 22 | 0.14% |
| competition_ban | 22 | 0.14% |
| Commerce_pay | 22 | 0.14% |
| Commitment | 21 | 0.14% |
| birthday_celebration | 21 | 0.14% |
| judicial_action_prosecution | 20 | 0.13% |
| product_delist | 20 | 0.13% |
| layoff | 20 | 0.13% |
| Fire | 19 | 0.12% |
| breakup | 19 | 0.12% |
| Criminal_investigation | 19 | 0.12% |
| apologize_to | 19 | 0.12% |
| Warning | 19 | 0.12% |
| Legality | 19 | 0.12% |
| product_award | 18 | 0.12% |
| Forming_relationships | 18 | 0.12% |
| Convincing | 18 | 0.12% |
| Reforming_a_system | 18 | 0.12% |
| competition_withdrawal | 17 | 0.11% |
| Telling | 17 | 0.11% |
| Bearing_arms | 17 | 0.11% |
| Education_teaching | 16 | 0.10% |
| birth | 16 | 0.10% |
| Preserving | 16 | 0.10% |
| Earthquake | 16 | 0.10% |
| Employment | 16 | 0.10% |
| Writing | 16 | 0.10% |
| Commerce_sell | 15 | 0.10% |
| Reveal_secret | 15 | 0.10% |
| judicial_action_filing | 15 | 0.10% |
| Resolve_problem | 15 | 0.10% |
| Surrounding | 14 | 0.09% |
| dissolution | 14 | 0.09% |
| lost_contact | 13 | 0.08% |
| price_decrease | 13 | 0.08% |
| retirement | 13 | 0.08% |
| suspension | 13 | 0.08% |
| Change_sentiment | 13 | 0.08% |
| Testing | 13 | 0.08% |
| Collapse | 12 | 0.08% |
| Body_movement | 12 | 0.08% |
| Patrolling | 12 | 0.08% |
| Rescuing | 12 | 0.08% |
| Limiting | 12 | 0.08% |
| visit_on_set | 12 | 0.08% |
| praise | 11 | 0.07% |
| price_limit_down | 11 | 0.07% |
| proposal | 11 | 0.07% |
| judicial_action_report | 11 | 0.07% |
| Temporary_stay | 11 | 0.07% |
| closing | 11 | 0.07% |
| Plane Crash | 10 | 0.06% |
| march | 10 | 0.06% |
| judicial_action_hearing | 10 | 0.06% |
| wedding | 10 | 0.06% |
| Filling | 10 | 0.06% |
| strike | 10 | 0.06% |
| Expend_resource | 10 | 0.06% |
| Adducing | 10 | 0.06% |
| thank | 9 | 0.06% |
| Labeling | 9 | 0.06% |
| Emptying | 9 | 0.06% |
| Explosion | 9 | 0.06% |
| meet_with | 9 | 0.06% |
| Award | 9 | 0.06% |
| dismissal | 8 | 0.05% |
| GetReady | 8 | 0.05% |
| Research | 8 | 0.05% |
| Kidnapping | 8 | 0.05% |
| engagement | 8 | 0.05% |
| contract_termination | 8 | 0.05% |
| Create_artwork | 8 | 0.05% |
| Rite | 8 | 0.05% |
| Releasing | 8 | 0.05% |
| Wearing | 8 | 0.05% |
| Submitting_documents | 7 | 0.05% |
| Prison | 7 | 0.05% |
| Scouring | 7 | 0.05% |
| Imposing_obligation | 7 | 0.05% |
| price_increase | 7 | 0.05% |
| Hiding_objects | 7 | 0.05% |
| Containing | 7 | 0.05% |
| Cure | 7 | 0.05% |
| Exchange | 7 | 0.05% |
| Having_or_lacking_access | 7 | 0.05% |
| Carry_goods | 6 | 0.04% |
| Practice | 6 | 0.04% |
| Revenge | 6 | 0.04% |
| infidelity | 5 | 0.03% |
| ipo | 5 | 0.03% |
| Robbery | 5 | 0.03% |
| Justifying | 5 | 0.03% |
| Commerce_buy | 5 | 0.03% |
| Cost | 5 | 0.03% |
| Rewards_and_punishments | 5 | 0.03% |
| rate_decrease | 4 | 0.03% |
| Vocalizations | 4 | 0.03% |
| pregnancy | 3 | 0.02% |
| rate_increase | 3 | 0.02% |
| Suspicion | 3 | 0.02% |
| Lighting | 3 | 0.02% |
| Flood | 2 | 0.01% |
| Institutionalization | 2 | 0.01% |
| Incident | 2 | 0.01% |
| Extradition | 1 | 0.01% |
| Emergency | 1 | 0.01% |
| Ratification | 1 | 0.01% |
| Theft | 1 | 0.01% |
| Risk | 1 | 0.01% |

**val.** No event instances.

**test.** No event instances.

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 14,657 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.71 | 30.27% | 1.76 | 29.13% | 0.04% | 0 | 14,657 | 0 | 15,465 |
| val | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |
| test | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 15,100 | 0.95% | 12,041 | 11.02% | 10.57% | 1.48% | 0.56% | 15.62% |
| val | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |
| test | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 144 | 787 | 12,041 |
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
| StartDate | 4,291 | 5.78% |
| EquityHolder | 3,817 | 5.14% |
| StockCode | 3,354 | 4.52% |
| CompanyName | 2,847 | 3.83% |
| StockAbbr | 2,833 | 3.82% |
| Location | 2,666 | 3.59% |
| PledgedShares | 2,665 | 3.59% |
| Date | 2,422 | 3.26% |
| Pledgee | 2,198 | 2.96% |
| TotalHoldingShares | 2,122 | 2.86% |
| TradedShares | 1,975 | 2.66% |
| TotalHoldingRatio | 1,924 | 2.59% |
| Casualties and Losses | 1,821 | 2.45% |
| Reduction Amount | 1,573 | 2.12% |
| TotalPledgedShares | 1,511 | 2.03% |
| EndDate | 1,305 | 1.76% |
| ReleasedDate | 1,245 | 1.68% |
| Reducing Shareholder | 1,176 | 1.58% |
| Cause | 1,003 | 1.35% |
| Reduction Start Date | 947 | 1.28% |
| Company Name | 835 | 1.12% |
| Announcement Date | 767 | 1.03% |
| LaterHoldingShares | 682 | 0.92% |
| pledge_amount | 509 | 0.69% |
| AveragePrice | 489 | 0.66% |
| Person | 420 | 0.57% |
| Organization | 411 | 0.55% |
| pledgor | 382 | 0.51% |
| Recipient | 381 | 0.51% |
| Candidates and their Political Parties | 379 | 0.51% |
| Affected Areas | 352 | 0.47% |
| Accepting Court | 331 | 0.45% |
| RepurchasedShares | 313 | 0.42% |
| Contest Participant | 298 | 0.40% |
| Weapon and Equippment | 296 | 0.40% |
| Vulnerability | 294 | 0.40% |
| Protest Reason | 293 | 0.39% |
| Attacker | 290 | 0.39% |
| Protesters | 287 | 0.39% |
| Individual_protein | 283 | 0.38% |
| Loss Amount | 280 | 0.38% |
| Target | 269 | 0.36% |
| HighestTradingPrice | 266 | 0.36% |
| Accident Investigator | 255 | 0.34% |
| protein | 253 | 0.34% |
| pledge_start_date | 241 | 0.32% |
| Charged Crime | 241 | 0.32% |
| Affected Area | 239 | 0.32% |
| Belligerents | 227 | 0.31% |
| Frozen Amount | 221 | 0.30% |
| Increase Amount | 221 | 0.30% |
| ruling_date | 213 | 0.29% |
| DiseaseOrPhenotypicFeature | 210 | 0.28% |
| Accusation | 204 | 0.27% |
| ClosingDate | 201 | 0.27% |
| Increasing Shareholder | 200 | 0.27% |
| Flight No. | 196 | 0.26% |
| Scheduled Landing Place | 195 | 0.26% |
| Position | 194 | 0.26% |
| GeneOrGeneProduct | 194 | 0.26% |
| RepurchaseAmount | 189 | 0.25% |
| Areas affected | 186 | 0.25% |
| Policy Content | 183 | 0.25% |
| Husband | 180 | 0.24% |
| frozen_shareholder | 178 | 0.24% |
| Disease | 175 | 0.24% |
| Suspect | 175 | 0.24% |
| Aid Agency | 175 | 0.24% |
| Wife | 168 | 0.23% |
| Capabilities | 168 | 0.23% |
| Crew | 168 | 0.23% |
| LegalInstitution | 166 | 0.22% |
| Strike Reason | 163 | 0.22% |
| System | 161 | 0.22% |
| Game Name | 158 | 0.21% |
| Aircraft Agency | 157 | 0.21% |
| Storm Hit Location | 154 | 0.21% |
| Time | 145 | 0.20% |
| ChemicalEntity | 142 | 0.19% |
| Influence People | 141 | 0.19% |
| The Final Votes and Percentages | 141 | 0.19% |
| Judgement Result/Prison Term | 138 | 0.19% |
| Freeze Start Date | 134 | 0.18% |
| FrozeShares | 133 | 0.18% |
| pledge_end_date | 133 | 0.18% |
| Accused People | 132 | 0.18% |
| Increase Start Date | 125 | 0.17% |
| Strike Outcome | 120 | 0.16% |
| Champions | 119 | 0.16% |
| Acquiree | 118 | 0.16% |
| Anti-pollution People/Organizations | 117 | 0.16% |
| Precautionary Measure | 116 | 0.16% |
| Policy Proposals | 114 | 0.15% |
| LowestTradingPrice | 111 | 0.15% |
| Solution | 110 | 0.15% |
| Organization Name | 110 | 0.15% |
| PII | 110 | 0.15% |
| Strike Agency | 108 | 0.15% |
| Pollution Source | 106 | 0.14% |
| Country Visited | 106 | 0.14% |
| Magnitude | 105 | 0.14% |
| Economic Loss | 103 | 0.14% |
| Participants | 101 | 0.14% |
| Summit Theme | 100 | 0.13% |
| Habitat | 100 | 0.13% |
| Election Name | 98 | 0.13% |
| Score | 97 | 0.13% |
| Riot Reason | 95 | 0.13% |
| Speaker Status | 95 | 0.13% |
| Location/Hospital | 93 | 0.13% |
| Winner | 93 | 0.13% |
| Patch | 93 | 0.13% |
| Start Date | 91 | 0.12% |
| Record-breaking Project | 91 | 0.12% |
| Deliberating Agency | 90 | 0.12% |
| Passengers | 90 | 0.12% |
| Resignated Person | 86 | 0.12% |
| Resign Reason | 85 | 0.11% |
| compensation_amount | 84 | 0.11% |
| Historical Sites | 84 | 0.11% |
| Damaged Facility | 83 | 0.11% |
| Participating Countries | 82 | 0.11% |
| Record Breaker | 82 | 0.11% |
| Survivors | 81 | 0.11% |
| Confirmed/Infected Cases | 81 | 0.11% |
| Microorganism | 81 | 0.11% |
| Epicenter | 80 | 0.11% |
| Award | 80 | 0.11% |
| Device | 80 | 0.11% |
| Grades | 78 | 0.11% |
| Countries Withdrawing from the Organization | 76 | 0.10% |
| GeneProtein | 75 | 0.10% |
| Time of the Case | 74 | 0.10% |
| Court | 73 | 0.10% |
| Sponsor | 73 | 0.10% |
| Effective Date | 72 | 0.10% |
| protein-family | 72 | 0.10% |
| Gene/protein/RNA | 72 | 0.10% |
| Data | 72 | 0.10% |
| Method | 71 | 0.10% |
| Death Cases | 71 | 0.10% |
| Strikers Status | 71 | 0.10% |
| OrganismTaxon | 70 | 0.09% |
| Aid Supplies/Amount | 70 | 0.09% |
| Artifacts and Their Chronology | 70 | 0.09% |
| Software | 70 | 0.09% |
| Marriage Duration | 69 | 0.09% |
| compensation_recipient | 68 | 0.09% |
| Prosecutor | 67 | 0.09% |
| Discover Location | 67 | 0.09% |
| Number of Vehicles Involved in the Crash | 67 | 0.09% |
| Types of the Spectacle | 66 | 0.09% |
| Appointee | 65 | 0.09% |
| Contracting Parties | 65 | 0.09% |
| The Charged Crime | 64 | 0.09% |
| Taking-off Place | 63 | 0.08% |
| Child | 62 | 0.08% |
| Deceased | 61 | 0.08% |
| Weapon | 60 | 0.08% |
| Speaker | 60 | 0.08% |
| Spectacle Location | 60 | 0.08% |
| Area Affected | 60 | 0.08% |
| senior_personnel | 59 | 0.08% |
| Army | 59 | 0.08% |
| Employment Agency | 59 | 0.08% |
| senior_position | 57 | 0.08% |
| Way for Spreading | 57 | 0.08% |
| Competition Items | 57 | 0.08% |
| Profession | 56 | 0.08% |
| Organization Industry | 56 | 0.08% |
| People | 55 | 0.07% |
| Host | 55 | 0.07% |
| Spectacle Start Time | 54 | 0.07% |
| Previous Record Holder | 54 | 0.07% |
| Freeze End Date | 53 | 0.07% |
| Poisoning Type | 53 | 0.07% |
| Version | 53 | 0.07% |
| Award Reason | 52 | 0.07% |
| Number of Damaged Houses | 52 | 0.07% |
| Person under Investigation | 51 | 0.07% |
| The Agency Who Broke the Agreement | 51 | 0.07% |
| Police | 50 | 0.07% |
| Visitor | 50 | 0.07% |
| Investigative Agency | 50 | 0.07% |
| File | 50 | 0.07% |
| Spacecraft | 50 | 0.07% |
| Goal | 49 | 0.07% |
| Evidence | 48 | 0.06% |
| Maximum Wind Speed | 48 | 0.06% |
| Protest Slogan | 48 | 0.06% |
| Storm Name | 47 | 0.06% |
| People/Organization who predicted the disaster | 47 | 0.06% |
| Outbreak Location | 47 | 0.06% |
| Symptom | 46 | 0.06% |
| CVE | 46 | 0.06% |
| Boycotted Institutions | 45 | 0.06% |
| Storm Hit Time | 45 | 0.06% |
| Death Time | 45 | 0.06% |
| Arrested | 45 | 0.06% |
| Purpose | 45 | 0.06% |
| Strikers | 43 | 0.06% |
| Age | 43 | 0.06% |
| Attending Hospital | 43 | 0.06% |
| Declarer | 43 | 0.06% |
| Agreement Name | 43 | 0.06% |
| Conflict Duration | 43 | 0.06% |
| Affected Industries | 43 | 0.06% |
| Arrest Time | 42 | 0.06% |
| Defense Lawyer | 42 | 0.06% |
| Death Reason | 42 | 0.06% |
| Pests | 42 | 0.06% |
| Epidemic Data Issuing Agency | 41 | 0.06% |
| The Grades of the Previous Record Holder | 41 | 0.06% |
| Perpetrators | 40 | 0.05% |
| Fined Agency | 40 | 0.05% |
| Number of Evacuated People | 39 | 0.05% |
| Phenotype | 39 | 0.05% |
| Fine Reason | 39 | 0.05% |
| Released People | 39 | 0.05% |
| CellType | 39 | 0.05% |
| Anatomy | 39 | 0.05% |
| End Time | 38 | 0.05% |
| Last Job of the Appointee | 38 | 0.05% |
| Storm Warning Level | 38 | 0.05% |
| chemical | 37 | 0.05% |
| Malware | 37 | 0.05% |
| Number of Influenced People | 37 | 0.05% |
| Archaeologist | 36 | 0.05% |
| Outbreak Date | 36 | 0.05% |
| Court Time | 36 | 0.05% |
| Judge | 36 | 0.05% |
| Award Field | 35 | 0.05% |
| End Date | 34 | 0.05% |
| Approver | 34 | 0.05% |
| Number of Destroyed Building | 33 | 0.04% |
| Predecessor | 33 | 0.04% |
| Annoncement Date | 33 | 0.04% |
| Agreement Content | 33 | 0.04% |
| Missings | 33 | 0.04% |
| SequenceVariant | 33 | 0.04% |
| Train No. | 33 | 0.04% |
| species | 33 | 0.04% |
| Website | 32 | 0.04% |
| Symptoms | 32 | 0.04% |
| Ship No. | 32 | 0.04% |
| DNA | 31 | 0.04% |
| Party Responsible for Pollution | 31 | 0.04% |
| Launch Date | 31 | 0.04% |
| Investigating agency | 30 | 0.04% |
| Penalty Amount | 30 | 0.04% |
| Spacecraft Mission | 30 | 0.04% |
| The Final Seats Result | 29 | 0.04% |
| Areas Affected | 28 | 0.04% |
| Ways to Watch the Speech | 28 | 0.04% |
| Volcano Name | 28 | 0.04% |
| Money | 28 | 0.04% |
| Number | 28 | 0.04% |
| Other Loss | 27 | 0.04% |
| Military Exercise | 27 | 0.04% |
| Concept | 27 | 0.04% |
| Head of the Merged Organization | 26 | 0.04% |
| Last Time the Record was Broken | 26 | 0.04% |
| Bank Name | 26 | 0.04% |
| Regulatory Authority | 26 | 0.04% |
| Carrier Rocket | 26 | 0.04% |
| Research Agency | 26 | 0.04% |
| Start Time | 25 | 0.03% |
| Arrest Location | 25 | 0.03% |
| Term of Office | 25 | 0.03% |
| protein-complex | 25 | 0.03% |
| Victim | 25 | 0.03% |
| Duration | 24 | 0.03% |
| protein-domain | 24 | 0.03% |
| Military Strength | 24 | 0.03% |
| Rescue Organizer | 24 | 0.03% |
| Election Goal | 24 | 0.03% |
| Electoral System | 23 | 0.03% |
| UnfrozeDate | 23 | 0.03% |
| Casualty Count | 23 | 0.03% |
| Influenced Crops and Livelihood | 23 | 0.03% |
| Scale | 23 | 0.03% |
| Withdraw Reason | 23 | 0.03% |
| Host Country | 23 | 0.03% |
| Release Reason | 23 | 0.03% |
| Achievement | 22 | 0.03% |
| Amount of Precipitation | 22 | 0.03% |
| Company Industry | 22 | 0.03% |
| Lasting Time | 21 | 0.03% |
| News Release Agency | 21 | 0.03% |
| Discover Time | 21 | 0.03% |
| Acquisition amount | 21 | 0.03% |
| Beneficiary | 21 | 0.03% |
| Property Division | 21 | 0.03% |
| Treatment Method | 20 | 0.03% |
| Child Custody | 20 | 0.03% |
| Related Rivers or Lakes | 20 | 0.03% |
| Summit Name | 20 | 0.03% |
| Inauguration Time | 20 | 0.03% |
| Criminal Evidence | 20 | 0.03% |
| Number of Rescued People | 20 | 0.03% |
| Launch Site | 20 | 0.03% |
| Spectacle Duration | 20 | 0.03% |
| Release Time | 20 | 0.03% |
| Witness | 19 | 0.03% |
| Prosecution Lawyer | 19 | 0.03% |
| Announce Platform | 19 | 0.03% |
| Number of Victims | 19 | 0.03% |
| Response Measures | 19 | 0.03% |
| Magnitude(Tsunami heights) | 19 | 0.03% |
| Launch Result | 19 | 0.03% |
| Storm Direction | 19 | 0.03% |
| Acquirer | 18 | 0.02% |
| Archaeologist Organization | 18 | 0.02% |
| CellLine | 18 | 0.02% |
| Group | 18 | 0.02% |
| Head of Investigation Team | 18 | 0.02% |
| Projects | 18 | 0.02% |
| Sentencing Location | 18 | 0.02% |
| Turnout(who win who lost) | 17 | 0.02% |
| Susceptible Population | 17 | 0.02% |
| Appointment Time | 17 | 0.02% |
| GPE | 17 | 0.02% |
| Policy Name & Abbreviation | 17 | 0.02% |
| Number of Trapped People | 17 | 0.02% |
| Species | 17 | 0.02% |
| Doctor and Medical Team | 16 | 0.02% |
| Temporary Settlement | 16 | 0.02% |
| Water Level | 16 | 0.02% |
| Government Reaction | 16 | 0.02% |
| Weapon Used | 16 | 0.02% |
| Voting method | 16 | 0.02% |
| Spokeswoman/Spokesman | 16 | 0.02% |
| Country | 16 | 0.02% |
| Funding | 16 | 0.02% |
| Planet | 16 | 0.02% |
| Age of the Appointee | 15 | 0.02% |
| PaymentMethod | 15 | 0.02% |
| Employed Institution | 15 | 0.02% |
| Transportation | 15 | 0.02% |
| Number of Vaccinated People | 15 | 0.02% |
| Aid Reason | 15 | 0.02% |
| Casualities and Losses | 15 | 0.02% |
| cell | 15 | 0.02% |
| Head of Institution | 15 | 0.02% |
| Merger Terms | 15 | 0.02% |
| RNA | 14 | 0.02% |
| Appointer | 14 | 0.02% |
| MVP | 14 | 0.02% |
| Damaged Property | 14 | 0.02% |
| Tear Up Reason | 14 | 0.02% |
| Event | 14 | 0.02% |
| Candidate Scandal(Allegations of fraud, etc.) | 14 | 0.02% |
| Age of the Resignated Person | 14 | 0.02% |
| Tribe | 14 | 0.02% |
| Phylum | 14 | 0.02% |
| Storm Movement Speed | 13 | 0.02% |
| Protein_family_or_group | 13 | 0.02% |
| Gene | 13 | 0.02% |
| Forecasting Agency | 13 | 0.02% |
| Successor | 13 | 0.02% |
| Start_Date | 13 | 0.02% |
| Prison Term | 13 | 0.02% |
| Commanders and their Position | 13 | 0.02% |
| Organization Members | 13 | 0.02% |
| Train Agency | 12 | 0.02% |
| The State of the Volcano (Dormant or Active) | 12 | 0.02% |
| Economic loss | 12 | 0.02% |
| State Before Death | 12 | 0.02% |
| Last Outbreak Time | 12 | 0.02% |
| Jail Time | 12 | 0.02% |
| Number of People Hospitalized | 12 | 0.02% |
| Grammatical Term | 12 | 0.02% |
| Wedding Venue | 12 | 0.02% |
| Agreement Members | 12 | 0.02% |
| Number of Aftershocks | 11 | 0.01% |
| Death Rate | 11 | 0.01% |
| Organization Leader | 11 | 0.01% |
| experiment-tag | 11 | 0.01% |
| experimental-construct | 11 | 0.01% |
| Launch Country | 11 | 0.01% |
| ai_model | 11 | 0.01% |
| organization | 11 | 0.01% |
| reagent | 11 | 0.01% |
| Bill Drafting Agency | 11 | 0.01% |
| Disaster-Stricken Farmland | 11 | 0.01% |
| algorithm | 11 | 0.01% |
| Work | 11 | 0.01% |
| model | 11 | 0.01% |
| mathematician | 11 | 0.01% |
| Strike Industry | 10 | 0.01% |
| Spokenmen | 10 | 0.01% |
| Maximum Rainfall | 10 | 0.01% |
| Service Years | 10 | 0.01% |
| Other Impact | 10 | 0.01% |
| Perpetrator | 10 | 0.01% |
| target | 10 | 0.01% |
| Prison | 10 | 0.01% |
| Alternate Landing Place | 10 | 0.01% |
| Mission Duration | 10 | 0.01% |
| Cooperative Agency | 10 | 0.01% |
| Shipwreck Reason | 10 | 0.01% |
| Astronauts | 10 | 0.01% |
| Hull Discovery Time | 10 | 0.01% |
| Document | 10 | 0.01% |
| Product | 10 | 0.01% |
| Hostage | 9 | 0.01% |
| Unemployed Rate | 9 | 0.01% |
| Mathematical Concept | 9 | 0.01% |
| Illness | 9 | 0.01% |
| Rescue Tool or Method | 9 | 0.01% |
| Stolen Amount | 9 | 0.01% |
| agent | 8 | 0.01% |
| Investigating Spokesperson | 8 | 0.01% |
| Mathematical Structure | 8 | 0.01% |
| Agreement Validity Period | 8 | 0.01% |
| Military Unit | 8 | 0.01% |
| Countries Joined the Organization | 8 | 0.01% |
| Spectacle End Time | 8 | 0.01% |
| Postpone Reason | 8 | 0.01% |
| Tsunamis | 8 | 0.01% |
| Invited Person | 8 | 0.01% |
| Damaged Crops & Livestock | 7 | 0.01% |
| Compensation | 7 | 0.01% |
| Sequelae | 7 | 0.01% |
| Publication | 7 | 0.01% |
| Cured Cases | 7 | 0.01% |
| dataset | 7 | 0.01% |
| Storm Center Location | 7 | 0.01% |
| protein-motif | 6 | 0.01% |
| Best Way to Shoot | 6 | 0.01% |
| Death Age | 6 | 0.01% |
| Bankrupt Business | 6 | 0.01% |
| Special Medicine | 6 | 0.01% |
| Economists who Predicted the Crisis | 6 | 0.01% |
| Refuge | 6 | 0.01% |
| The Sentence Claimed by the Defense Lawyer | 6 | 0.01% |
| Influenced People | 6 | 0.01% |
| mathematical_construct | 6 | 0.01% |
| Wedding Dress Designer | 6 | 0.01% |
| Adjective | 6 | 0.01% |
| Moon | 6 | 0.01% |
| mathematical_result | 6 | 0.01% |
| Complications | 5 | 0.01% |
| Field of Study | 5 | 0.01% |
| Art Form | 5 | 0.01% |
| process | 5 | 0.01% |
| Trapped Depth | 5 | 0.01% |
| Economy Loss | 5 | 0.01% |
| Recovered Amount | 5 | 0.01% |
| Literary Work | 5 | 0.01% |
| metric | 5 | 0.01% |
| loss_function | 5 | 0.01% |
| Hull Location | 5 | 0.01% |
| Head of the Institution | 5 | 0.01% |
| Postpone Time | 5 | 0.01% |
| Provisional Government | 5 | 0.01% |
| Detention Start Time | 5 | 0.01% |
| CellComponent | 5 | 0.01% |
| People Casting Key Votes | 5 | 0.01% |
| Commanders of the Army | 5 | 0.01% |
| Invalid Date | 4 | 0.01% |
| Reasons for the Formation of the Historical sites | 4 | 0.01% |
| Time Period | 4 | 0.01% |
| Art Movement | 4 | 0.01% |
| End_Date | 4 | 0.01% |
| The Sentence Claimed by the Prosecutor's Lawyer | 4 | 0.01% |
| Word | 4 | 0.01% |
| organelle | 4 | 0.01% |
| mutation | 4 | 0.01% |
| concept | 4 | 0.01% |
| environment | 4 | 0.01% |
| Head of the Government | 4 | 0.01% |
| geometric_curve | 4 | 0.01% |
| group | 4 | 0.01% |
| Object | 4 | 0.01% |
| Substance | 4 | 0.01% |
| amino-acid | 4 | 0.01% |
| Role | 4 | 0.01% |
| Ship Agency | 4 | 0.01% |
| chemical_compound | 4 | 0.01% |
| researcher | 4 | 0.01% |
| fusion-protein | 4 | 0.01% |
| Hypothetical Planet | 4 | 0.01% |
| mathematical_operator | 4 | 0.01% |
| assay | 3 | 0.00% |
| evaluation_metric | 3 | 0.00% |
| Production Cuts | 3 | 0.00% |
| Rescue Start Time | 3 | 0.00% |
| Material | 3 | 0.00% |
| Fire Warning Level | 3 | 0.00% |
| decoder_architecture | 3 | 0.00% |
| peptide | 3 | 0.00% |
| Project | 3 | 0.00% |
| embodiment | 3 | 0.00% |
| Disease Subtype | 3 | 0.00% |
| Medical Professional | 3 | 0.00% |
| Medical Test | 3 | 0.00% |
| Theory | 3 | 0.00% |
| chemical_group | 3 | 0.00% |
| physical_property | 3 | 0.00% |
| Population | 3 | 0.00% |
| Nation | 3 | 0.00% |
| mathematical_concept | 3 | 0.00% |
| Activity | 3 | 0.00% |
| algebraic_structure | 3 | 0.00% |
| Subphylum | 3 | 0.00% |
| Organism | 3 | 0.00% |
| Suspected Cases | 3 | 0.00% |
| Warning Device | 2 | 0.00% |
| Political Entity | 2 | 0.00% |
| Writing System | 2 | 0.00% |
| baseline_method | 2 | 0.00% |
| method_component | 2 | 0.00% |
| The Worst-Hit Area | 2 | 0.00% |
| Protein_complex | 2 | 0.00% |
| Plant Species | 2 | 0.00% |
| Chemical | 2 | 0.00% |
| technical_method | 2 | 0.00% |
| architecture_component | 2 | 0.00% |
| Join Conditions | 2 | 0.00% |
| How Many Times Get Married | 2 | 0.00% |
| error_correcting_code | 2 | 0.00% |
| hardware_element | 2 | 0.00% |
| Field | 2 | 0.00% |
| Lawyer | 2 | 0.00% |
| Pathogen | 2 | 0.00% |
| modality | 2 | 0.00% |
| Service | 2 | 0.00% |
| Principle | 2 | 0.00% |
| Model | 2 | 0.00% |
| Phenomenon | 2 | 0.00% |
| Grammatical Concept | 2 | 0.00% |
| Test | 2 | 0.00% |
| molecular_orbital | 2 | 0.00% |
| researcher_group | 2 | 0.00% |
| institutional_unit | 2 | 0.00% |
| alignment_technique | 2 | 0.00% |
| adversarial_attack | 2 | 0.00% |
| model_component | 2 | 0.00% |
| Enzyme | 2 | 0.00% |
| Body Part | 2 | 0.00% |
| Disease Form | 2 | 0.00% |
| gene | 2 | 0.00% |
| Trapped Days | 2 | 0.00% |
| Civilization | 2 | 0.00% |
| Law | 2 | 0.00% |
| Publication Issue | 2 | 0.00% |
| Celestial System | 2 | 0.00% |
| Religious Text | 2 | 0.00% |
| Deity | 2 | 0.00% |
| Star | 2 | 0.00% |
| Scientific Theory | 2 | 0.00% |
| Celestial Object | 2 | 0.00% |
| protein-region | 2 | 0.00% |
| geometric_object | 2 | 0.00% |
| numerical_function | 2 | 0.00% |
| mathematical_structure | 2 | 0.00% |
| Biological Group | 2 | 0.00% |
| Geological Period | 2 | 0.00% |
| Pest | 2 | 0.00% |
| Equipment | 2 | 0.00% |
| Metric | 2 | 0.00% |
| Damage | 2 | 0.00% |
| Religion | 1 | 0.00% |
| Mathematical Result | 1 | 0.00% |
| Mathematical Theorem | 1 | 0.00% |
| Symbol | 1 | 0.00% |
| Cured Rate | 1 | 0.00% |
| Film | 1 | 0.00% |
| Registered Capital | 1 | 0.00% |
| research_task | 1 | 0.00% |
| framework | 1 | 0.00% |
| input | 1 | 0.00% |
| constraint | 1 | 0.00% |
| intermediate_representation | 1 | 0.00% |
| output | 1 | 0.00% |
| actor | 1 | 0.00% |
| protein-isoform | 1 | 0.00% |
| Narrative | 1 | 0.00% |
| Text | 1 | 0.00% |
| Invention | 1 | 0.00% |
| Measurement | 1 | 0.00% |
| Plant Group | 1 | 0.00% |
| Habitat Feature | 1 | 0.00% |
| Animal | 1 | 0.00% |
| architectural_component | 1 | 0.00% |
| parameter | 1 | 0.00% |
| data_format | 1 | 0.00% |
| data_structure | 1 | 0.00% |
| publication_venue | 1 | 0.00% |
| error_type | 1 | 0.00% |
| Language | 1 | 0.00% |
| Medium | 1 | 0.00% |
| Cost | 1 | 0.00% |
| Regulation | 1 | 0.00% |
| Category | 1 | 0.00% |
| task_category | 1 | 0.00% |
| capability | 1 | 0.00% |
| Protein | 1 | 0.00% |
| Treatment | 1 | 0.00% |
| Community Group | 1 | 0.00% |
| Program | 1 | 0.00% |
| Doctor | 1 | 0.00% |
| Linguistic Concept | 1 | 0.00% |
| Entity | 1 | 0.00% |
| Platform | 1 | 0.00% |
| element | 1 | 0.00% |
| computational_method | 1 | 0.00% |
| software_package | 1 | 0.00% |
| Exit Conditions | 1 | 0.00% |
| evaluation_method | 1 | 0.00% |
| Biological Entity | 1 | 0.00% |
| publication_source | 1 | 0.00% |
| topological_space | 1 | 0.00% |
| mathematical_identity | 1 | 0.00% |
| State of the Hull | 1 | 0.00% |
| Genetic Element | 1 | 0.00% |
| Time for Dignitaries to Resign | 1 | 0.00% |
| Scientific Name | 1 | 0.00% |
| Control Method | 1 | 0.00% |
| Strategy | 1 | 0.00% |
| Natural Enemy | 1 | 0.00% |
| Lure | 1 | 0.00% |
| Lost Contact Time | 1 | 0.00% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| bind | 370 | 6.43% |
| PPI | 262 | 4.55% |
| Association | 193 | 3.36% |
| Lives_In | 136 | 2.36% |
| Positive_Correlation | 103 | 1.79% |
| Negative_Correlation | 59 | 1.03% |
| Exhibits | 38 | 0.66% |
| director | 29 | 0.50% |
| discoverer or inventor | 29 | 0.50% |
| member of sports team | 28 | 0.49% |
| platform | 28 | 0.49% |
| employer | 28 | 0.49% |
| record label | 28 | 0.49% |
| original broadcaster | 28 | 0.49% |
| architect | 27 | 0.47% |
| characters | 26 | 0.45% |
| located in the administrative territorial entity | 26 | 0.45% |
| historic county | 26 | 0.45% |
| movement | 25 | 0.43% |
| conferred by | 25 | 0.43% |
| owned by | 24 | 0.42% |
| country | 24 | 0.42% |
| member of | 24 | 0.42% |
| conflict | 23 | 0.40% |
| operator | 22 | 0.38% |
| father | 22 | 0.38% |
| performer | 22 | 0.38% |
| present in work | 21 | 0.37% |
| creator | 21 | 0.37% |
| licensed to broadcast to | 20 | 0.35% |
| award received | 20 | 0.35% |
| crosses | 20 | 0.35% |
| parent taxon | 20 | 0.35% |
| position held | 20 | 0.35% |
| located in/on physical feature | 19 | 0.33% |
| league | 19 | 0.33% |
| terminus | 19 | 0.33% |
| mountain range | 19 | 0.33% |
| publisher | 19 | 0.33% |
| military branch | 19 | 0.33% |
| package management system | 19 | 0.33% |
| item operated | 19 | 0.33% |
| member of political party | 18 | 0.31% |
| astronaut mission | 18 | 0.31% |
| nominated for | 18 | 0.31% |
| partner in business or sport | 18 | 0.31% |
| founded by | 18 | 0.31% |
| home venue | 18 | 0.31% |
| family | 17 | 0.30% |
| significant event | 17 | 0.30% |
| presenter | 17 | 0.30% |
| dam | 17 | 0.30% |
| located in the present-day administrative territorial entity | 17 | 0.30% |
| terminus location | 17 | 0.30% |
| composer | 17 | 0.30% |
| located on astronomical body | 17 | 0.30% |
| located in or next to body of water | 17 | 0.30% |
| surface played on | 17 | 0.30% |
| stock exchange | 17 | 0.30% |
| location | 16 | 0.28% |
| genre | 16 | 0.28% |
| location of formation | 16 | 0.28% |
| officeholder | 16 | 0.28% |
| author | 16 | 0.28% |
| production company | 16 | 0.28% |
| heritage designation | 16 | 0.28% |
| sports discipline competed in | 16 | 0.28% |
| position played on team / speciality | 16 | 0.28% |
| secretary general | 16 | 0.28% |
| participant in | 16 | 0.28% |
| cast member | 16 | 0.28% |
| sport | 16 | 0.28% |
| basic form of government | 16 | 0.28% |
| stock market index | 15 | 0.26% |
| child | 15 | 0.26% |
| ethnic group | 15 | 0.26% |
| fictional universe described in | 15 | 0.26% |
| medical condition | 15 | 0.26% |
| place served by transport hub | 15 | 0.26% |
| located in the statistical territorial entity | 15 | 0.26% |
| connecting line | 15 | 0.26% |
| basin country | 15 | 0.26% |
| represented by | 15 | 0.26% |
| owner of | 15 | 0.26% |
| occupation | 15 | 0.26% |
| carries | 15 | 0.26% |
| has effect | 15 | 0.26% |
| language used | 15 | 0.26% |
| parent organization | 15 | 0.26% |
| pole position | 15 | 0.26% |
| significant place | 14 | 0.24% |
| field of work | 14 | 0.24% |
| takes place in fictional universe | 14 | 0.24% |
| educated at | 14 | 0.24% |
| mother | 14 | 0.24% |
| uses | 14 | 0.24% |
| connects with | 14 | 0.24% |
| member of the crew of | 14 | 0.24% |
| commanded by | 14 | 0.24% |
| operating area | 14 | 0.24% |
| transport network | 14 | 0.24% |
| place of detention | 14 | 0.24% |
| drug or therapy used for treatment | 14 | 0.24% |
| spouse | 14 | 0.24% |
| participating team | 14 | 0.24% |
| cathedral | 14 | 0.24% |
| dialect of | 14 | 0.24% |
| online service | 14 | 0.24% |
| notable work | 14 | 0.24% |
| mouth of the watercourse | 14 | 0.24% |
| product or material produced | 14 | 0.24% |
| headquarters location | 14 | 0.24% |
| winner | 14 | 0.24% |
| instrument | 14 | 0.24% |
| location of discovery | 14 | 0.24% |
| worshipped by | 13 | 0.23% |
| has use | 13 | 0.23% |
| applies to jurisdiction | 13 | 0.23% |
| model item | 13 | 0.23% |
| work location | 13 | 0.23% |
| has part(s) of the class | 13 | 0.23% |
| legislated by | 13 | 0.23% |
| instance of | 13 | 0.23% |
| after a work by | 13 | 0.23% |
| published in | 13 | 0.23% |
| convicted of | 13 | 0.23% |
| participant | 13 | 0.23% |
| chairperson | 13 | 0.23% |
| replaces | 13 | 0.23% |
| producer | 13 | 0.23% |
| main subject | 13 | 0.23% |
| has cause | 13 | 0.23% |
| head of government | 12 | 0.21% |
| inflows | 12 | 0.21% |
| has works in the collection | 12 | 0.21% |
| located in protected area | 12 | 0.21% |
| currency | 12 | 0.21% |
| replaced by | 12 | 0.21% |
| standards body | 12 | 0.21% |
| genic_interaction | 12 | 0.21% |
| commemorates | 12 | 0.21% |
| historical region | 12 | 0.21% |
| history of topic | 12 | 0.21% |
| diocese | 12 | 0.21% |
| country for sport | 12 | 0.21% |
| time period | 12 | 0.21% |
| court | 12 | 0.21% |
| continent | 12 | 0.21% |
| represents | 11 | 0.19% |
| constellation | 11 | 0.19% |
| Bind | 11 | 0.19% |
| primary destinations | 11 | 0.19% |
| has part(s) | 11 | 0.19% |
| parent club | 11 | 0.19% |
| country of origin | 11 | 0.19% |
| located on street | 11 | 0.19% |
| start point | 11 | 0.19% |
| maintained by | 11 | 0.19% |
| crew member(s) | 11 | 0.19% |
| has immediate cause | 11 | 0.19% |
| writing language | 11 | 0.19% |
| operating system | 11 | 0.19% |
| endemic to | 11 | 0.19% |
| form of creative work | 11 | 0.19% |
| school district | 11 | 0.19% |
| broadcast by | 11 | 0.19% |
| office held by head of the organization | 11 | 0.19% |
| commissioned by | 11 | 0.19% |
| taxon rank | 11 | 0.19% |
| grants | 11 | 0.19% |
| part of the series | 11 | 0.19% |
| distributed by | 11 | 0.19% |
| medical condition treated | 11 | 0.19% |
| ammunition | 11 | 0.19% |
| taxonomic type | 11 | 0.19% |
| named after | 10 | 0.17% |
| has subsidiary | 10 | 0.17% |
| shares border with | 10 | 0.17% |
| first appearance | 10 | 0.17% |
| league system | 10 | 0.17% |
| physically interacts with | 10 | 0.17% |
| this zoological name is coordinate with | 10 | 0.17% |
| manufacturer | 10 | 0.17% |
| culture | 10 | 0.17% |
| natural product of taxon | 10 | 0.17% |
| board member | 10 | 0.17% |
| connecting service | 10 | 0.17% |
| facet of | 10 | 0.17% |
| head of state | 10 | 0.17% |
| biological process | 10 | 0.17% |
| noble title | 10 | 0.17% |
| place of publication | 10 | 0.17% |
| reservoir created | 10 | 0.17% |
| director / manager | 10 | 0.17% |
| opposite of | 10 | 0.17% |
| chromosome | 10 | 0.17% |
| editor | 10 | 0.17% |
| IUCN conservation status | 10 | 0.17% |
| native language | 10 | 0.17% |
| brand | 10 | 0.17% |
| found in taxon | 10 | 0.17% |
| part of | 10 | 0.17% |
| made from material | 10 | 0.17% |
| source of energy | 9 | 0.16% |
| league level above | 9 | 0.16% |
| commander of (DEPRECATED) | 9 | 0.16% |
| developer | 9 | 0.16% |
| merged into | 9 | 0.16% |
| followed by | 9 | 0.16% |
| canonization status | 9 | 0.16% |
| this taxon is source of | 9 | 0.16% |
| defendant | 9 | 0.16% |
| produced by | 9 | 0.16% |
| distribution format | 9 | 0.16% |
| tracklist | 9 | 0.16% |
| military rank | 9 | 0.16% |
| recommended unit of measurement | 9 | 0.16% |
| contributed to creative work | 9 | 0.16% |
| candidate | 9 | 0.16% |
| is the study of | 9 | 0.16% |
| programming language | 9 | 0.16% |
| vehicle | 9 | 0.16% |
| objective of project or action | 9 | 0.16% |
| indigenous to | 9 | 0.16% |
| anatomical location | 9 | 0.16% |
| origin of the watercourse | 9 | 0.16% |
| collection | 9 | 0.16% |
| located in time zone | 9 | 0.16% |
| librettist | 9 | 0.16% |
| newspaper format | 9 | 0.16% |
| programming paradigm | 9 | 0.16% |
| space launch vehicle | 9 | 0.16% |
| edition or translation of | 9 | 0.16% |
| highest judicial authority | 9 | 0.16% |
| residence | 9 | 0.16% |
| house publication | 8 | 0.14% |
| relative | 8 | 0.14% |
| voice type | 8 | 0.14% |
| domain of saint or deity | 8 | 0.14% |
| place of birth | 8 | 0.14% |
| lake outflow | 8 | 0.14% |
| incarnation of | 8 | 0.14% |
| airline alliance | 8 | 0.14% |
| has edition or translation | 8 | 0.14% |
| sponsor | 8 | 0.14% |
| legal form | 8 | 0.14% |
| discography | 8 | 0.14% |
| location of first performance | 8 | 0.14% |
| executive body | 8 | 0.14% |
| contains | 8 | 0.14% |
| victory | 8 | 0.14% |
| track gauge | 8 | 0.14% |
| head coach | 8 | 0.14% |
| subclass of | 8 | 0.14% |
| consecrator | 8 | 0.14% |
| musical conductor | 8 | 0.14% |
| illustrator | 8 | 0.14% |
| adjacent station | 8 | 0.14% |
| cause of destruction | 8 | 0.14% |
| choreographer | 8 | 0.14% |
| vehicle normally used | 8 | 0.14% |
| parent astronomical body | 8 | 0.14% |
| narrative location | 8 | 0.14% |
| territory overlaps | 8 | 0.14% |
| target | 8 | 0.14% |
| dedicated to | 8 | 0.14% |
| anatomical branch of | 7 | 0.12% |
| manifestation of | 7 | 0.12% |
| writing system | 7 | 0.12% |
| set in period | 7 | 0.12% |
| different from | 7 | 0.12% |
| lake on watercourse | 7 | 0.12% |
| said to be the same as | 7 | 0.12% |
| interested in | 7 | 0.12% |
| home world | 7 | 0.12% |
| copyright license | 7 | 0.12% |
| religion or worldview | 7 | 0.12% |
| powered by | 7 | 0.12% |
| occupant | 7 | 0.12% |
| chief executive officer | 7 | 0.12% |
| lyrics by | 7 | 0.12% |
| public holiday | 7 | 0.12% |
| contains the administrative territorial entity | 7 | 0.12% |
| wheel arrangement | 7 | 0.12% |
| filming location | 7 | 0.12% |
| from narrative universe | 7 | 0.12% |
| radio format | 7 | 0.12% |
| voice actor | 7 | 0.12% |
| child astronomical body | 7 | 0.12% |
| has written for | 7 | 0.12% |
| copyright holder | 7 | 0.12% |
| business division | 7 | 0.12% |
| qualifies for event | 7 | 0.12% |
| language of work or name | 7 | 0.12% |
| software engine | 7 | 0.12% |
| director of photography | 7 | 0.12% |
| fabrication method | 7 | 0.12% |
| field of this occupation | 6 | 0.10% |
| competition class | 6 | 0.10% |
| legislative body | 6 | 0.10% |
| substitute/deputy/replacement of office/officeholder | 6 | 0.10% |
| location of landing | 6 | 0.10% |
| sibling | 6 | 0.10% |
| original language of film or TV show | 6 | 0.10% |
| engine configuration | 6 | 0.10% |
| interchange station | 6 | 0.10% |
| religious order | 6 | 0.10% |
| allegiance | 6 | 0.10% |
| mascot | 6 | 0.10% |
| subject has role | 6 | 0.10% |
| described by source | 6 | 0.10% |
| country of citizenship | 6 | 0.10% |
| ancestral home | 6 | 0.10% |
| implementation of | 6 | 0.10% |
| director of archaeological fieldwork | 6 | 0.10% |
| perpetrator | 6 | 0.10% |
| genetic association | 6 | 0.10% |
| parent of this hybrid, breed, or cultivar | 6 | 0.10% |
| immediate cause of | 6 | 0.10% |
| measured physical quantity | 6 | 0.10% |
| overlies | 6 | 0.10% |
| media franchise | 6 | 0.10% |
| killed by | 6 | 0.10% |
| industry | 6 | 0.10% |
| coextensive with | 6 | 0.10% |
| depicted by | 6 | 0.10% |
| has anatomical branch | 6 | 0.10% |
| place of death | 6 | 0.10% |
| office held by head of state | 6 | 0.10% |
| performed at | 6 | 0.10% |
| structural engineer | 6 | 0.10% |
| co-driver | 6 | 0.10% |
| cause of death | 6 | 0.10% |
| location of creation | 6 | 0.10% |
| sex or gender | 6 | 0.10% |
| foundational text | 5 | 0.09% |
| significant person | 5 | 0.09% |
| has natural reservoir | 5 | 0.09% |
| organizer | 5 | 0.09% |
| associated electoral district | 5 | 0.09% |
| armament | 5 | 0.09% |
| linguistic typology | 5 | 0.09% |
| student of | 5 | 0.09% |
| highest point | 5 | 0.09% |
| color | 5 | 0.09% |
| drainage basin | 5 | 0.09% |
| possible treatment | 5 | 0.09% |
| candidacy in election | 5 | 0.09% |
| vessel class | 5 | 0.09% |
| has quality | 5 | 0.09% |
| health specialty | 5 | 0.09% |
| affiliation | 5 | 0.09% |
| studied by | 5 | 0.09% |
| official language | 5 | 0.09% |
| destination point | 5 | 0.09% |
| signatory | 5 | 0.09% |
| next higher rank | 5 | 0.09% |
| separated from | 5 | 0.09% |
| enemy of | 5 | 0.09% |
| authority | 5 | 0.09% |
| encodes | 5 | 0.09% |
| Drug_Interaction | 5 | 0.09% |
| minor planet group | 5 | 0.09% |
| used by | 5 | 0.09% |
| party chief representative | 5 | 0.09% |
| anthem | 5 | 0.09% |
| unmarried partner | 5 | 0.09% |
| cell component | 5 | 0.09% |
| airline hub | 5 | 0.09% |
| coach of sports team | 5 | 0.09% |
| twinned administrative body | 4 | 0.07% |
| general classification of race participants | 4 | 0.07% |
| capital | 4 | 0.07% |
| based on | 4 | 0.07% |
| manner of death | 4 | 0.07% |
| sports season of league or competition | 4 | 0.07% |
| GUI toolkit or framework | 4 | 0.07% |
| readable file format | 4 | 0.07% |
| screenwriter | 4 | 0.07% |
| tributary | 4 | 0.07% |
| official religion | 4 | 0.07% |
| practiced by | 4 | 0.07% |
| designed by | 4 | 0.07% |
| mount | 4 | 0.07% |
| superhuman feature or ability | 4 | 0.07% |
| central bank/issuer | 4 | 0.07% |
| dual to | 4 | 0.07% |
| office held by head of government | 4 | 0.07% |
| less than | 4 | 0.07% |
| measurement scale | 4 | 0.07% |
| depends on software | 4 | 0.07% |
| archives at | 4 | 0.07% |
| solved by | 4 | 0.07% |
| Gram staining | 4 | 0.07% |
| sidekick of | 4 | 0.07% |
| political ideology | 4 | 0.07% |
| successful candidate | 4 | 0.07% |
| final event | 4 | 0.07% |
| family name identical to this given name | 4 | 0.07% |
| approved by | 4 | 0.07% |
| translator | 4 | 0.07% |
| political alignment | 4 | 0.07% |
| Comparison | 4 | 0.07% |
| has boundary | 3 | 0.05% |
| honorific prefix | 3 | 0.05% |
| molecular function | 3 | 0.05% |
| place of burial | 3 | 0.05% |
| has list | 3 | 0.05% |
| drafted by | 3 | 0.05% |
| charted in | 3 | 0.05% |
| statement describes | 3 | 0.05% |
| patron saint | 3 | 0.05% |
| temporal range start | 3 | 0.05% |
| regulated by | 3 | 0.05% |
| shape | 3 | 0.05% |
| mountains classification | 3 | 0.05% |
| given name version for other gender | 3 | 0.05% |
| advertises | 3 | 0.05% |
| amended by | 3 | 0.05% |
| CPU | 3 | 0.05% |
| contributor to the creative work or subject | 3 | 0.05% |
| typically sells | 3 | 0.05% |
| coat of arms | 3 | 0.05% |
| underlies | 3 | 0.05% |
| copyright representative | 3 | 0.05% |
| depicts | 3 | 0.05% |
| youth wing | 3 | 0.05% |
| medical examination | 3 | 0.05% |
| cuisine | 3 | 0.05% |
| day in year for periodic occurrence | 3 | 0.05% |
| architectural style | 3 | 0.05% |
| Christian liturgical rite | 3 | 0.05% |
| instrumentation | 3 | 0.05% |
| contributing factor of | 3 | 0.05% |
| inspired by | 3 | 0.05% |
| social classification | 3 | 0.05% |
| tonality | 3 | 0.05% |
| in opposition to | 3 | 0.05% |
| input method | 3 | 0.05% |
| derivative work | 3 | 0.05% |
| game mechanics | 3 | 0.05% |
| source of material | 3 | 0.05% |
| season | 3 | 0.05% |
| has active ingredient | 3 | 0.05% |
| title of chess person | 3 | 0.05% |
| designed to carry | 3 | 0.05% |
| site of astronomical discovery | 3 | 0.05% |
| partially coincident with | 3 | 0.05% |
| editor-in-chief | 3 | 0.05% |
| prerequisite | 3 | 0.05% |
| main regulatory text | 3 | 0.05% |
| afflicts | 3 | 0.05% |
| members have occupation | 3 | 0.05% |
| via | 3 | 0.05% |
| game mode | 3 | 0.05% |
| student | 3 | 0.05% |
| enclave within | 3 | 0.05% |
| instruction set | 3 | 0.05% |
| sports league level | 2 | 0.03% |
| flag | 2 | 0.03% |
| language regulatory body | 2 | 0.03% |
| family name | 2 | 0.03% |
| mother house | 2 | 0.03% |
| foods traditionally associated | 2 | 0.03% |
| type of variable star | 2 | 0.03% |
| product certification | 2 | 0.03% |
| programmer | 2 | 0.03% |
| economy of topic | 2 | 0.03% |
| appears in the form of | 2 | 0.03% |
| innervated by | 2 | 0.03% |
| character designer | 2 | 0.03% |
| music created for | 2 | 0.03% |
| parallel street | 2 | 0.03% |
| territory claimed by | 2 | 0.03% |
| main food source | 2 | 0.03% |
| honorific suffix | 2 | 0.03% |
| destroyed | 2 | 0.03% |
| office contested | 2 | 0.03% |
| prohibits | 2 | 0.03% |
| active ingredient in | 2 | 0.03% |
| measures | 2 | 0.03% |
| languages spoken, written or signed | 2 | 0.03% |
| official residence | 2 | 0.03% |
| designated as terrorist by | 2 | 0.03% |
| theme music | 2 | 0.03% |
| executive producer | 2 | 0.03% |
| subdivision of this unit | 2 | 0.03% |
| capital of | 2 | 0.03% |
| gens | 2 | 0.03% |
| interaction | 2 | 0.03% |
| expansion of | 2 | 0.03% |
| captain | 2 | 0.03% |
| central bank | 2 | 0.03% |
| league level below | 2 | 0.03% |
| connector | 2 | 0.03% |
| people or cargo transported | 2 | 0.03% |
| country of registry | 2 | 0.03% |
| delta of | 2 | 0.03% |
| charge | 2 | 0.03% |
| official podcast | 2 | 0.03% |
| does not have part | 2 | 0.03% |
| symptoms and signs | 2 | 0.03% |
| recorded at studio or venue | 2 | 0.03% |
| speaker | 2 | 0.03% |
| follows | 2 | 0.03% |
| type of orbit | 2 | 0.03% |
| sexually homologous with | 2 | 0.03% |
| character type | 2 | 0.03% |
| fastest lap | 2 | 0.03% |
| soundtrack release | 2 | 0.03% |
| antiparticle | 2 | 0.03% |
| monomer of | 2 | 0.03% |
| statistical leader | 2 | 0.03% |
| narrative role | 2 | 0.03% |
| theorised by | 1 | 0.02% |
| scheduled service destination | 1 | 0.02% |
| prime factor | 1 | 0.02% |
| distributary | 1 | 0.02% |
| officially opened by | 1 | 0.02% |
| offers view on | 1 | 0.02% |
| next lower rank | 1 | 0.02% |
| Cotreatment | 1 | 0.02% |
| wing configuration | 1 | 0.02% |
| academic degree | 1 | 0.02% |
| stereoisomer of | 1 | 0.02% |
| feast day | 1 | 0.02% |
| referee | 1 | 0.02% |
| expressed in | 1 | 0.02% |
| doctoral student | 1 | 0.02% |
| diaspora | 1 | 0.02% |
| emergency phone number | 1 | 0.02% |
| calculated from | 1 | 0.02% |
| comorbidity | 1 | 0.02% |
| decay mode | 1 | 0.02% |
| adjacent building | 1 | 0.02% |
| trophy awarded | 1 | 0.02% |
| item disputed by | 1 | 0.02% |
| next crossing upstream | 1 | 0.02% |
| has contributing factor | 1 | 0.02% |
| day of week | 1 | 0.02% |
| source of income | 1 | 0.02% |
| style of karate | 1 | 0.02% |
| appointed by | 1 | 0.02% |
| has melody | 1 | 0.02% |
| symbolizes | 1 | 0.02% |
| wears | 1 | 0.02% |
| structure replaced by | 1 | 0.02% |
| train depot | 1 | 0.02% |
| uses capitalization for | 1 | 0.02% |
| demographics of topic | 1 | 0.02% |
| epithet | 1 | 0.02% |
| geography of topic | 1 | 0.02% |
| record held | 1 | 0.02% |
| top-level Internet domain | 1 | 0.02% |
| lifestyle | 1 | 0.02% |
| proved by | 1 | 0.02% |
| tournament format | 1 | 0.02% |
| biological variant of | 1 | 0.02% |
| display technology | 1 | 0.02% |
| named by | 1 | 0.02% |
| official symbol | 1 | 0.02% |
| influenced by | 1 | 0.02% |
| diplomatic relation | 1 | 0.02% |
| has facility | 1 | 0.02% |
| GPU | 1 | 0.02% |
| military casualty classification | 1 | 0.02% |
| madhhab | 1 | 0.02% |
| exemplar of | 1 | 0.02% |
| narrator | 1 | 0.02% |
| points classification | 1 | 0.02% |
| disease transmission process | 1 | 0.02% |
| has tense | 1 | 0.02% |
| record or record progression | 1 | 0.02% |
| radix | 1 | 0.02% |
| home port | 1 | 0.02% |
| exhibition history | 1 | 0.02% |
| mod of | 1 | 0.02% |
| significant drug interaction | 1 | 0.02% |
| typing discipline | 1 | 0.02% |
| output method | 1 | 0.02% |
| series spin-off | 1 | 0.02% |
| penalty | 1 | 0.02% |
| by-product | 1 | 0.02% |
| set during recurring event | 1 | 0.02% |
| permanent resident of | 1 | 0.02% |
| conjugate acid | 1 | 0.02% |
| decays to | 1 | 0.02% |
| route of administration | 1 | 0.02% |
| vaccine for | 1 | 0.02% |
| risk factor | 1 | 0.02% |
| supported sports team | 1 | 0.02% |
| substrate of | 1 | 0.02% |
| chassis | 1 | 0.02% |
| plaintiff | 1 | 0.02% |
| stepparent | 1 | 0.02% |
| has pet | 1 | 0.02% |
| business model | 1 | 0.02% |
| type foundry | 1 | 0.02% |
| doctoral advisor | 1 | 0.02% |
| has phenotype | 1 | 0.02% |
| presynaptic connection | 1 | 0.02% |
| film editor | 1 | 0.02% |
| repeals | 1 | 0.02% |
| writable file format | 1 | 0.02% |
| first aid measures | 1 | 0.02% |
| computes solution to | 1 | 0.02% |
| copyright status | 1 | 0.02% |
| general manager | 1 | 0.02% |
| organization directed by the office or position | 1 | 0.02% |
| political coalition | 1 | 0.02% |
| launch contractor | 1 | 0.02% |
| aspect ratio (W:H) | 1 | 0.02% |
| main building contractor | 1 | 0.02% |
| facial hair | 1 | 0.02% |
| has fruit type | 1 | 0.02% |
| sexual orientation | 1 | 0.02% |
| catalog | 1 | 0.02% |
| host | 1 | 0.02% |

### Classifications

**train, biotech_event.** Majority class 28.5% of 557 answers. The marginal is `true_label`, not the menu.

#### train biotech_event

| label | count | share |
| --- | --- | --- |
| executive statement | 159 | 28.55% |
| company description | 82 | 14.72% |
| alliance & partnership | 43 | 7.72% |
| other | 38 | 6.82% |
| product launching & presentation | 30 | 5.39% |
| new initiatives or programs | 26 | 4.67% |
| m&a | 23 | 4.13% |
| funding round | 22 | 3.95% |
| support & philanthropy | 19 | 3.41% |
| expanding geography | 16 | 2.87% |
| hiring | 15 | 2.69% |
| service & product providing | 15 | 2.69% |
| article publication | 15 | 2.69% |
| executive appointment | 10 | 1.80% |
| regulatory approval | 9 | 1.62% |
| event organization | 7 | 1.26% |
| product updates | 6 | 1.08% |
| participation in an event | 4 | 0.72% |
| new initiatives & programs | 3 | 0.54% |
| ipo exit | 3 | 0.54% |
| closing | 3 | 0.54% |
| department establishment | 2 | 0.36% |
| clinical trial sponsorship | 2 | 0.36% |
| expanding industry | 2 | 0.36% |
| investment in public company | 2 | 0.36% |
| foundation | 1 | 0.18% |

**train, chfinann_event.** Majority class 49.3% of 3,448 answers. The marginal is `true_label`, not the menu.

#### train chfinann_event

| label | count | share |
| --- | --- | --- |
| EquityPledge | 1,701 | 49.33% |
| EquityOverweight | 716 | 20.77% |
| EquityUnderweight | 683 | 19.81% |
| EquityRepurchase | 225 | 6.53% |
| EquityFreeze | 123 | 3.57% |

**train, docee_event.** Majority class 7.7% of 2,901 answers. The marginal is `true_label`, not the menu.

#### train docee_event

| label | count | share |
| --- | --- | --- |
| Armed Conflict | 224 | 7.72% |
| Air Crash | 183 | 6.31% |
| Election | 134 | 4.62% |
| Protest | 127 | 4.38% |
| Sports Competition | 103 | 3.55% |
| Fire | 99 | 3.41% |
| CommitCrime - Accuse | 97 | 3.34% |
| CommitCrime - Sentence | 96 | 3.31% |
| Government Policy Changes | 93 | 3.21% |
| Riot | 75 | 2.59% |
| Earthquakes | 75 | 2.59% |
| Resignation_Dismissal | 64 | 2.21% |
| Disease Outbreaks | 58 | 2.00% |
| Appoint_Inauguration | 56 | 1.93% |
| Famous Person - Death | 54 | 1.86% |
| Break Historical Records | 54 | 1.86% |
| Storm | 53 | 1.83% |
| Withdraw from an Orgnization | 51 | 1.76% |
| Floods | 49 | 1.69% |
| Diplomatic Talks | 48 | 1.65% |
| Strike | 46 | 1.59% |
| Environment Pollution | 45 | 1.55% |
| Famous Person - Give a Speech | 41 | 1.41% |
| CommitCrime - Arrest | 41 | 1.41% |
| Military Excercise | 40 | 1.38% |
| Gas Explosion | 39 | 1.34% |
| Famous Person - Sick | 38 | 1.31% |
| Road Crash | 38 | 1.31% |
| CommitCrime - Investigate | 36 | 1.24% |
| Mass Poisoning | 36 | 1.24% |
| Financial Crisis | 36 | 1.24% |
| Awards Ceremony | 35 | 1.21% |
| Mine Collapses | 35 | 1.21% |
| New Achievements in Aerospace | 34 | 1.17% |
| Diplomatic Visit | 33 | 1.14% |
| Train Collisions | 32 | 1.10% |
| New Archeological Discoveries | 32 | 1.10% |
| Shipwreck | 31 | 1.07% |
| Organization Merge | 31 | 1.07% |
| New Wonders in Nature | 30 | 1.03% |
| Famine | 30 | 1.03% |
| Insect Disaster | 29 | 1.00% |
| CommitCrime - Release | 29 | 1.00% |
| Organization Closed | 28 | 0.97% |
| Sign Agreement | 27 | 0.93% |
| Bank Robbery | 25 | 0.86% |
| Famous Person - Divorce | 24 | 0.83% |
| Organization Fine | 24 | 0.83% |
| Volcano Eruption | 21 | 0.72% |
| Famous Person - Marriage | 21 | 0.72% |
| Tear Up Agreement | 20 | 0.69% |
| Financial Aid | 16 | 0.55% |
| Droughts | 15 | 0.52% |
| Organization Established | 15 | 0.52% |
| Tsunamis | 15 | 0.52% |
| Mudslides | 14 | 0.48% |
| Famous Person - Recovered | 12 | 0.41% |
| Regime Change | 9 | 0.31% |
| Join in an Orgnization | 5 | 0.17% |

**train, docfee_event.** Majority class 34.5% of 2,225 answers. The marginal is `true_label`, not the menu.

#### train docfee_event

| label | count | share |
| --- | --- | --- |
| EquityUnderweight | 767 | 34.47% |
| EquityPledge | 360 | 16.18% |
| BankruptcyLiquidation | 359 | 16.13% |
| MajorAssetLoss | 278 | 12.49% |
| EquityFreeze | 147 | 6.61% |
| EquityOverweight | 117 | 5.26% |
| MajorSafetyAccident | 78 | 3.51% |
| MajorExternalPayout | 60 | 2.70% |
| ExecutiveDeath | 59 | 2.65% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| record | status | 148 | 14.35% |
| record | location | 107 | 10.38% |
| record | name | 66 | 6.40% |
| record | address | 61 | 5.92% |
| record | description | 36 | 3.49% |
| record | title | 34 | 3.30% |
| record | price | 29 | 2.81% |
| record | amount | 26 | 2.52% |
| record | date | 22 | 2.13% |
| record | author | 22 | 2.13% |
| record | city | 22 | 2.13% |
| record | patient_name | 20 | 1.94% |
| record | patient_id | 20 | 1.94% |
| record | rating | 19 | 1.84% |
| record | employee_id | 18 | 1.75% |
| record | claim_id | 17 | 1.65% |
| record | company | 15 | 1.45% |
| record | age | 15 | 1.45% |
| record | transaction_id | 14 | 1.36% |
| record | country | 14 | 1.36% |
| record | state | 14 | 1.36% |
| record | department | 14 | 1.36% |
| record | product_id | 14 | 1.36% |
| record | venue | 13 | 1.26% |
| record | year | 12 | 1.16% |
| record | product_name | 11 | 1.07% |
| record | review_id | 11 | 1.07% |
| record | manufacturer | 10 | 0.97% |
| record | event_id | 10 | 0.97% |
| record | score | 10 | 0.97% |
| record | category | 10 | 0.97% |
| record | head | 10 | 0.97% |
| record | tail | 10 | 0.97% |
| record | sender | 10 | 0.97% |
| record | project_id | 9 | 0.87% |
| record | patient_address | 9 | 0.87% |
| record | dosage | 9 | 0.87% |
| record | region | 9 | 0.87% |
| record | home_team | 8 | 0.78% |
| record | customer_name | 8 | 0.78% |
| record | ticket_id | 8 | 0.78% |
| record | severity | 8 | 0.78% |
| record | brand | 7 | 0.68% |
| record | organization | 7 | 0.68% |
| record | reviewer | 7 | 0.68% |
| record | company_name | 7 | 0.68% |
| record | symptoms | 7 | 0.68% |
| record | asteroid.name | 7 | 0.68% |
| record | notes | 7 | 0.68% |
| record | mrn | 7 | 0.68% |
| record | recipient | 7 | 0.68% |
| record | subject | 7 | 0.68% |
| record | funding_amount | 7 | 0.68% |
| record | email | 6 | 0.58% |
| record | full_name | 6 | 0.58% |
| record | medication | 6 | 0.58% |
| record | role | 6 | 0.58% |
| record | duration | 6 | 0.58% |
| record | policy_id | 6 | 0.58% |
| record | gender | 6 | 0.58% |
| record | location.city | 6 | 0.58% |
| record | authors | 6 | 0.58% |
| record | funding_source | 6 | 0.58% |
| record | sku | 6 | 0.58% |
| record | patient_age | 6 | 0.58% |
| record | provider | 6 | 0.58% |
| record | policyholder | 6 | 0.58% |
| record | points | 5 | 0.48% |
| record | clause_id | 5 | 0.48% |
| record | features | 5 | 0.48% |
| record | orbital_characteristics.classification | 5 | 0.48% |
| record | discovery.observatory.location.country | 5 | 0.48% |
| record | discovery.observatory.name | 5 | 0.48% |
| record | person | 5 | 0.48% |
| record | diagnosis_code | 5 | 0.48% |
| record | likes | 5 | 0.48% |
| record | event | 5 | 0.48% |
| record | target_audience | 5 | 0.48% |
| record | destination | 5 | 0.48% |
| record | origin | 5 | 0.48% |
| record | word_count | 5 | 0.48% |
| record | review_date | 5 | 0.48% |
| record | heart_rate: Heart rate in beats per minute | 5 | 0.48% |
| record | temperature: Body temperature in degrees Fahrenheit | 5 | 0.48% |
| record | report_id | 5 | 0.48% |
| record | serial_number | 5 | 0.48% |
| record | material | 5 | 0.48% |
| record | source | 5 | 0.48% |
| record | orbital_characteristics.semi_major_axis.unit | 5 | 0.48% |
| record | id | 5 | 0.48% |
| record | match | 5 | 0.48% |
| record | resolution_time | 5 | 0.48% |
| record | zip_code | 5 | 0.48% |
| record | player_id | 5 | 0.48% |
| record | color | 5 | 0.48% |
| record | user_id | 5 | 0.48% |
| record | merchant | 4 | 0.39% |
| record | drug_name | 4 | 0.39% |
| record | indication | 4 | 0.39% |
| record | ndc | 4 | 0.39% |
| record | opponent | 4 | 0.39% |
| record | amenities | 4 | 0.39% |
| record | family | 4 | 0.39% |
| record | price_range.per | 4 | 0.39% |
| record | location.country | 4 | 0.39% |
| record | price_range.currency | 4 | 0.39% |
| record | flight_id | 4 | 0.39% |
| record | content | 4 | 0.39% |
| record | hotel_name | 4 | 0.39% |
| record | revenue | 4 | 0.39% |
| record | claim_id: Unique identifier for the insurance claim | 4 | 0.39% |
| record | institution | 4 | 0.39% |
| record | report_date | 4 | 0.39% |
| record | quantity | 4 | 0.39% |
| record | discovery_date | 4 | 0.39% |
| record | icd10code | 4 | 0.39% |
| record | keywords | 4 | 0.39% |
| record | asteroid.discovery.date | 4 | 0.39% |
| record | insurance_provider | 4 | 0.39% |
| record | enrollment_date | 4 | 0.39% |
| record | film_title | 4 | 0.39% |
| record | reason | 4 | 0.39% |
| record | return_id | 4 | 0.39% |
| record | delay_minutes | 4 | 0.39% |
| record | delay_reason | 4 | 0.39% |
| record | findings | 4 | 0.39% |
| record | product | 4 | 0.39% |
| record | review_status | 4 | 0.39% |
| record | entity_name | 4 | 0.39% |
| record | copyright | 4 | 0.39% |
| record | cpt_code | 4 | 0.39% |
| record | policyholder_name | 4 | 0.39% |
| record | term_length | 4 | 0.39% |
| record | job_title | 4 | 0.39% |
| record | salary_range | 4 | 0.39% |
| record | model | 4 | 0.39% |
| record | target | 4 | 0.39% |
| record | locations | 4 | 0.39% |
| record | physical_characteristics.observations.0.telescope | 4 | 0.39% |
| record | assists | 4 | 0.39% |
| record | diagnosis | 4 | 0.39% |
| record | tags | 4 | 0.39% |
| record | location.province | 4 | 0.39% |
| record | normal_range | 4 | 0.39% |
| record | value | 4 | 0.39% |
| record | project_name | 4 | 0.39% |
| record | newsletter_name | 4 | 0.39% |
| record | timestamp | 4 | 0.39% |
| record | type | 4 | 0.39% |
| record | policy_number | 4 | 0.39% |
| record | discovery_year | 4 | 0.39% |
| record | physical_characteristics.observations.0.publication_year | 4 | 0.39% |
| record | provider_name | 4 | 0.39% |
| record | payee | 3 | 0.29% |
| record | bedrooms | 3 | 0.29% |
| record | asteroid.named_after.person.birth_year | 3 | 0.29% |
| record | asteroid.asteroid_type | 3 | 0.29% |
| record | asteroid.named_after.person.name | 3 | 0.29% |
| record | ticker | 3 | 0.29% |
| record | format | 3 | 0.29% |
| record | owner | 3 | 0.29% |
| record | key_player | 3 | 0.29% |
| record | route | 3 | 0.29% |
| record | scientific_name | 3 | 0.29% |
| record | order | 3 | 0.29% |
| record | product_name: Name of the product being used | 3 | 0.29% |
| record | user: Name of the team member or user using the product | 3 | 0.29% |
| record | company: Organization where the product is used | 3 | 0.29% |
| record | location.district | 3 | 0.29% |
| record | dose_number | 3 | 0.29% |
| record | root_cause | 3 | 0.29% |
| record | departure_time | 3 | 0.29% |
| record | airline | 3 | 0.29% |
| record | booking_reference | 3 | 0.29% |
| record | dateofservice | 3 | 0.29% |
| record | insurance_plan | 3 | 0.29% |
| record | claim_status | 3 | 0.29% |
| record | lead_investor | 3 | 0.29% |
| record | teams | 3 | 0.29% |
| record | medications | 3 | 0.29% |
| record | company_id: Unique identifier assigned to each company | 3 | 0.29% |
| record | priority | 3 | 0.29% |
| record | recipient_name | 3 | 0.29% |
| record | lead_person | 3 | 0.29% |
| record | prescription_id | 3 | 0.29% |
| record | home_team: Name of the team playing at home | 3 | 0.29% |
| record | issue_description | 3 | 0.29% |
| record | course_id | 3 | 0.29% |
| record | event_type: Category of the event | 3 | 0.29% |
| record | room | 3 | 0.29% |
| record | blood_pressure | 3 | 0.29% |
| record | time | 3 | 0.29% |
| record | attending_physician | 3 | 0.29% |
| record | discovery_location | 3 | 0.29% |
| record | program_name | 3 | 0.29% |
| record | insurance | 3 | 0.29% |
| record | condition | 3 | 0.29% |
| record | trial_id | 3 | 0.29% |
| record | date: Date when the transaction occurred | 3 | 0.29% |
| record | reporter | 3 | 0.29% |
| record | study_id | 3 | 0.29% |
| record | sample_size | 3 | 0.29% |
| record | product.type | 3 | 0.29% |
| record | product.description | 3 | 0.29% |
| record | reviewer_name | 3 | 0.29% |
| record | patient_name: Name of the patient filing the claim | 3 | 0.29% |
| record | claim_amount: Total cost of the medical service billed | 3 | 0.29% |
| record | patient_address: Full address of the patient | 3 | 0.29% |
| record | submitter | 3 | 0.29% |
| record | device | 3 | 0.29% |
| record | lease_term | 3 | 0.29% |
| record | founding_year | 3 | 0.29% |
| record | education | 3 | 0.29% |
| record | passenger_name | 3 | 0.29% |
| record | battery_life | 3 | 0.29% |
| record | exam_date | 3 | 0.29% |
| record | sponsor | 3 | 0.29% |
| record | district | 3 | 0.29% |
| record | replies | 3 | 0.29% |
| record | organizations | 3 | 0.29% |
| record | asteroid_name | 3 | 0.29% |
| record | acquirer | 3 | 0.29% |
| record | filer | 3 | 0.29% |
| record | name: Full name of the patient | 3 | 0.29% |
| record | patient_id: Unique identifier for the patient | 3 | 0.29% |
| record | discovery.discoverer | 3 | 0.29% |
| record | assigned_agent | 3 | 0.29% |
| record | visit_type | 3 | 0.29% |
| record | payment_terms | 3 | 0.29% |
| record | treatments: Prescribed medications or therapies during stay | 3 | 0.29% |
| record | followupprovider: Name of provider for follow-up visit | 3 | 0.29% |
| record | followupclinic_address: Physical address of follow-up clinic | 3 | 0.29% |
| record | designation | 3 | 0.29% |
| record | discovery.observatory.location.prefecture | 3 | 0.29% |
| record | orbital_characteristics.perihelion.unit | 3 | 0.29% |
| record | bathrooms | 3 | 0.29% |
| record | asteroid.designation | 3 | 0.29% |
| record | registration_fee | 3 | 0.29% |
| record | year_in_era | 3 | 0.29% |
| record | booking_options.partnerships | 3 | 0.29% |
| record | player_name | 3 | 0.29% |
| record | appointment_id | 3 | 0.29% |
| record | salary | 3 | 0.29% |
| record | video_id | 3 | 0.29% |
| record | asset | 3 | 0.29% |
| record | net_profit | 3 | 0.29% |
| record | affected_network | 3 | 0.29% |
| record | problem_description | 3 | 0.29% |
| record | away_team | 3 | 0.29% |
| record | headquarters: City and country where company is headquartered | 3 | 0.29% |
| record | status: Current operational or strategic status | 3 | 0.29% |
| record | incident_id | 3 | 0.29% |
| record | release_date | 3 | 0.29% |
| record | office_address | 3 | 0.29% |
| record | party_name | 3 | 0.29% |
| record | user_feedback | 3 | 0.29% |
| record | administration_date | 3 | 0.29% |
| record | author_handle | 3 | 0.29% |
| record | user_name: Name of the creator who posted the trend | 3 | 0.29% |
| record | video_id: Unique identifier for the video content | 3 | 0.29% |
| record | location: City and state where the user is based | 3 | 0.29% |
| record | user | 3 | 0.29% |
| record | donor_name | 3 | 0.29% |
| record | discovery.date | 3 | 0.29% |
| record | established | 3 | 0.29% |
| record | funding | 3 | 0.29% |
| record | cas_number | 3 | 0.29% |
| record | physical_characteristics.observations.0.telescope.name | 3 | 0.29% |
| record | concepts | 3 | 0.29% |
| record | rate | 3 | 0.29% |
| record | asteroid.type | 3 | 0.29% |
| record | purpose | 2 | 0.19% |
| record | patient_gender | 2 | 0.19% |
| record | start_date | 2 | 0.19% |
| record | subscriber_name | 2 | 0.19% |
| record | subscription_id | 2 | 0.19% |
| record | player | 2 | 0.19% |
| record | final_score | 2 | 0.19% |
| record | asteroid.named_after.person.awards | 2 | 0.19% |
| record | asteroid.named_after.person.occupation | 2 | 0.19% |
| record | lead_researcher | 2 | 0.19% |
| record | movement_reason | 2 | 0.19% |
| record | close_price | 2 | 0.19% |
| record | id: Unique identifier assigned to the startup | 2 | 0.19% |
| record | funding_amount: Total amount raised in the investment round | 2 | 0.19% |
| record | sector: Technology domain such as health tech, AI, clean energy | 2 | 0.19% |
| record | key_features | 2 | 0.19% |
| record | issues | 2 | 0.19% |
| record | award | 2 | 0.19% |
| record | vote_percentage | 2 | 0.19% |
| record | party | 2 | 0.19% |
| record | phone | 2 | 0.19% |
| record | approval_year | 2 | 0.19% |
| record | side_effects | 2 | 0.19% |
| record | approval_status | 2 | 0.19% |
| record | publication.author | 2 | 0.19% |
| record | publication.year | 2 | 0.19% |
| record | performance_impact: Percentage improvement in lead quality or hiring time | 2 | 0.19% |
| record | ranking.rank | 2 | 0.19% |
| record | physical_characteristics.observations.0.publication | 2 | 0.19% |
| record | discovery.observatory.location.county | 2 | 0.19% |
| record | orbital_characteristics.inclination.unit | 2 | 0.19% |
| record | user_id: ID of the user who submitted the snapshot | 2 | 0.19% |
| record | status: Current status of the exchange rate record | 2 | 0.19% |
| record | snapshot_id: Unique identifier for the snapshot in format currency-yearmonthday | 2 | 0.19% |
| record | statement_id | 2 | 0.19% |
| record | duration: Number of weeks the course runs | 2 | 0.19% |
| record | cost: Total cost of enrollment in US dollars | 2 | 0.19% |
| record | vaccine_type | 2 | 0.19% |
| record | outage_code | 2 | 0.19% |
| record | service_impact | 2 | 0.19% |
| record | departure_date | 2 | 0.19% |
| record | arrival_airport | 2 | 0.19% |
| record | rescheduled_event_time | 2 | 0.19% |
| record | departure_airport | 2 | 0.19% |
| record | last_updated: Date when product record was last modified | 2 | 0.19% |
| record | warehouse_location: Geographic warehouse where product is stored | 2 | 0.19% |
| record | provider_address | 2 | 0.19% |
| record | retweets | 2 | 0.19% |
| record | enrollment_code | 2 | 0.19% |
| record | supervisor | 2 | 0.19% |
| record | defendant_name | 2 | 0.19% |
| record | plaintiff_address | 2 | 0.19% |
| record | plaintiff_name | 2 | 0.19% |
| record | location: Geographic headquarters of the company | 2 | 0.19% |
| record | net_profit: Annual net profit in millions of dollars | 2 | 0.19% |
| record | ebitda_margin: Percentage of revenue that represents EBITDA | 2 | 0.19% |
| record | current_ratio: Liquidity ratio measuring short-term obligations | 2 | 0.19% |
| record | operational_status: Current phase of business operations | 2 | 0.19% |
| record | speaker | 2 | 0.19% |
| record | event_date | 2 | 0.19% |
| record | vendor_address | 2 | 0.19% |
| record | recipient_email | 2 | 0.19% |
| record | status: Current operational status of the device | 2 | 0.19% |
| record | legal_counsel | 2 | 0.19% |
| record | company.name | 2 | 0.19% |
| record | supply_duration | 2 | 0.19% |
| record | prescriber | 2 | 0.19% |
| record | game_id: Unique identifier for the game as listed in the schedule | 2 | 0.19% |
| record | away_team: Name of the team playing away | 2 | 0.19% |
| record | away_score: Final score for the away team | 2 | 0.19% |
| record | company_id | 2 | 0.19% |
| record | key_metric | 2 | 0.19% |
| record | last_updated | 2 | 0.19% |
| record | invoice_number | 2 | 0.19% |
| record | founder_name | 2 | 0.19% |
| record | founding_date | 2 | 0.19% |
| record | investment_source | 2 | 0.19% |
| record | industry | 2 | 0.19% |
| record | early_challenges | 2 | 0.19% |
| record | exit_status | 2 | 0.19% |
| record | current_status | 2 | 0.19% |
| record | asteroid.location | 2 | 0.19% |
| record | asteroid.dedication.person.occupation | 2 | 0.19% |
| record | reasons | 2 | 0.19% |
| record | proposer | 2 | 0.19% |
| record | vote_id | 2 | 0.19% |
| record | city: City where the event is held | 2 | 0.19% |
| record | event_status: Current status of the event | 2 | 0.19% |
| record | event_name: Title of the event as listed | 2 | 0.19% |
| record | location: Full address including street, city, and state | 2 | 0.19% |
| record | event_code: Unique code identifying the event | 2 | 0.19% |
| record | event_date: Date when the event will take place | 2 | 0.19% |
| record | promo_code: Discount or offer code associated with the campaign | 2 | 0.19% |
| record | office_hours.1.start_time | 2 | 0.19% |
| record | office_hours.2.end_time | 2 | 0.19% |
| record | office_hours.0.day | 2 | 0.19% |
| record | office_hours.0.start_time | 2 | 0.19% |
| record | office_hours.4.start_time | 2 | 0.19% |
| record | address.room | 2 | 0.19% |
| record | office_hours.2.start_time | 2 | 0.19% |
| record | office_hours.3.start_time | 2 | 0.19% |
| record | office_hours.0.end_time | 2 | 0.19% |
| record | address.street | 2 | 0.19% |
| record | temperature | 2 | 0.19% |
| record | monitoring_method | 2 | 0.19% |
| record | disbursement_plan | 2 | 0.19% |
| record | gpa | 2 | 0.19% |
| record | asteroid_number | 2 | 0.19% |
| record | discoverers | 2 | 0.19% |
| record | patient_case.clinical_presentation.medical_history.0.status | 2 | 0.19% |
| record | patient_case.demographics.gender | 2 | 0.19% |
| record | disease_profile.genetic_mapping.chromosome_location | 2 | 0.19% |
| record | patient_case.clinical_presentation.skin_lesions.characteristics | 2 | 0.19% |
| record | disease_profile.abbreviation | 2 | 0.19% |
| record | disease_profile.name | 2 | 0.19% |
| record | patient_case.clinical_presentation.medical_history.0.condition | 2 | 0.19% |
| record | report_context.geographic_significance | 2 | 0.19% |
| record | patient_case.genetic_analysis.mutation_details.dna_nomenclature | 2 | 0.19% |
| record | patient_case.genetic_analysis.mutation_details.type | 2 | 0.19% |
| record | report_context.epidemiology_notes | 2 | 0.19% |
| record | patient_case.genetic_analysis.mutation_details.affected_protein | 2 | 0.19% |
| record | patient_case.clinical_presentation.skin_lesions.description | 2 | 0.19% |
| record | patient_case.clinical_presentation.skin_lesions.locations | 2 | 0.19% |
| record | patient_case.clinical_presentation.skin_lesions.symptoms | 2 | 0.19% |
| record | patient_case.clinical_presentation.skin_lesions.size_range.min_mm | 2 | 0.19% |
| record | patient_case.genetic_analysis.mutation_details.functional_effect | 2 | 0.19% |
| record | patient_case.genetic_analysis.mutation_details.exon_location | 2 | 0.19% |
| record | genus.type_species | 2 | 0.19% |
| record | effective_date | 2 | 0.19% |
| record | hotel_details.year_built | 2 | 0.19% |
| record | location.airport_travel_time.value | 2 | 0.19% |
| record | location.distance_from_airport.value | 2 | 0.19% |
| record | child_policy.infants.age_range | 2 | 0.19% |
| record | location.nearby_attractions | 2 | 0.19% |
| record | booking_policy.multiple_rooms | 2 | 0.19% |
| record | child_policy.adult_age | 2 | 0.19% |
| record | hotel_details.room_amenities | 2 | 0.19% |
| record | hotel_details.recreation | 2 | 0.19% |
| record | asteroid.designation_number | 2 | 0.19% |
| record | jurisdiction | 2 | 0.19% |
| record | filing_type | 2 | 0.19% |
| record | isbn | 2 | 0.19% |
| record | publication_volume | 2 | 0.19% |
| record | publication_issue | 2 | 0.19% |
| record | consent_status | 2 | 0.19% |
| record | study_site | 2 | 0.19% |
| record | policy_type | 2 | 0.19% |
| record | transaction_id: Unique identifier for the transaction entry | 2 | 0.19% |
| record | status: Current status of the transaction | 2 | 0.19% |
| record | product.brand | 2 | 0.19% |
| record | product.name | 2 | 0.19% |
| record | outage_end: Time when the outage ended in HH:MM AM/PM format | 2 | 0.19% |
| record | service_code: Internal code identifying the service type | 2 | 0.19% |
| record | ticket_number: Unique identifier for the outage ticket | 2 | 0.19% |
| record | outage_start: Time when the outage began in HH:MM AM/PM format | 2 | 0.19% |
| record | return_date | 2 | 0.19% |
| record | year_founded | 2 | 0.19% |
| record | goals | 2 | 0.19% |
| record | disease_links | 2 | 0.19% |
| record | activation_mechanism | 2 | 0.19% |
| record | agent | 2 | 0.19% |
| record | book_title | 2 | 0.19% |
| record | provider: Name of the medical professional or facility providing service | 2 | 0.19% |
| record | reference_id | 2 | 0.19% |
| record | contact_name | 2 | 0.19% |
| record | university | 2 | 0.19% |
| record | funding_amount: Total amount raised in millions | 2 | 0.19% |
| record | publication_date | 2 | 0.19% |
| record | launch_date | 2 | 0.19% |
| record | sentiment.opinion | 2 | 0.19% |
| record | sentiment.preference | 2 | 0.19% |
| record | quote.text | 2 | 0.19% |
| record | favorite_players | 2 | 0.19% |
| record | edit_information.timestamp | 2 | 0.19% |
| record | products | 2 | 0.19% |
| record | abstract_id | 2 | 0.19% |
| record | enrollment_capacity | 2 | 0.19% |
| record | credit_hours | 2 | 0.19% |
| record | instructor | 2 | 0.19% |
| record | meeting_days | 2 | 0.19% |
| record | date: Date when the reaction was reported | 2 | 0.19% |
| record | patient_id: Unique identifier for the patient in the record | 2 | 0.19% |
| record | duration: Number of days the treatment lasted for the reaction | 2 | 0.19% |
| record | status: Current status of the reaction | 2 | 0.19% |
| record | file_id | 2 | 0.19% |
| record | open_rate | 2 | 0.19% |
| record | campaign_id | 2 | 0.19% |
| record | product.version | 2 | 0.19% |
| record | product.features | 2 | 0.19% |
| record | resolution_hours | 2 | 0.19% |
| record | patient_id: Unique identifier for the monitored patient | 2 | 0.19% |
| record | building | 2 | 0.19% |
| record | key_challenge | 2 | 0.19% |
| record | province | 2 | 0.19% |
| record | flight_number | 2 | 0.19% |
| record | passenger_email | 2 | 0.19% |
| record | hotel.star_rating | 2 | 0.19% |
| record | municipality.name | 2 | 0.19% |
| record | exam_type | 2 | 0.19% |
| record | protocol | 2 | 0.19% |
| record | instructor: Name of the faculty member teaching the course | 2 | 0.19% |
| record | department: Academic department offering the course | 2 | 0.19% |
| record | distribution | 2 | 0.19% |
| record | issue_date | 2 | 0.19% |
| record | monthly_premium | 2 | 0.19% |
| record | product_model | 2 | 0.19% |
| record | trial_type | 2 | 0.19% |
| record | arm | 2 | 0.19% |
| record | population | 2 | 0.19% |
| record | filing_date: Date the patent was filed | 2 | 0.19% |
| record | topic | 2 | 0.19% |
| record | personal_experience.raw_corn_experience | 2 | 0.19% |
| record | serving_suggestions | 2 | 0.19% |
| record | boiling_instructions.cooking_time.max_minutes | 2 | 0.19% |
| record | cooking_methods | 2 | 0.19% |
| record | ingredients | 2 | 0.19% |
| record | boiling_instructions.salt_advice | 2 | 0.19% |
| record | grilling_instructions.cooking_time.minutes | 2 | 0.19% |
| record | sunless_tanning.concerns | 2 | 0.19% |
| record | sunless_tanning.product_collection.brand | 2 | 0.19% |
| record | sunless_tanning.product_collection.benefits | 2 | 0.19% |
| record | sunless_tanning.product_collection.products.0.type | 2 | 0.19% |
| record | sunless_tanning.product_collection.products.0.sizes | 2 | 0.19% |
| record | sunless_tanning.product_collection.products.0.description | 2 | 0.19% |
| record | sunless_tanning.product_collection.products.0.ingredients | 2 | 0.19% |
| record | sunless_tanning.product_collection.products.0.benefits | 2 | 0.19% |
| record | sunless_tanning.product_collection.products.1.name | 2 | 0.19% |
| record | sunless_tanning.product_collection.products.1.type | 2 | 0.19% |
| record | sunless_tanning.product_collection.products.1.description | 2 | 0.19% |
| record | recipient: Email of the person receiving the confirmation or submission | 2 | 0.19% |
| record | discovery_observatory | 2 | 0.19% |
| record | pathway_id | 2 | 0.19% |
| record | line1 | 2 | 0.19% |
| record | address: Full street address including suite and city | 2 | 0.19% |
| record | name: Official name of the business unit operating the office | 2 | 0.19% |
| record | model_number | 2 | 0.19% |
| record | ticket_id: Unique identifier for each support request | 2 | 0.19% |
| record | sector | 2 | 0.19% |
| record | employees | 2 | 0.19% |
| record | integration_plan | 2 | 0.19% |
| record | launch_year: Year the product was first introduced | 2 | 0.19% |
| record | company: Manufacturer or developer of the product | 2 | 0.19% |
| record | price: Product cost in US dollars | 2 | 0.19% |
| record | sales_trend | 2 | 0.19% |
| record | product_id: Unique identifier for the product instance | 2 | 0.19% |
| record | affiliation | 2 | 0.19% |
| record | rating: Star rating out of 5 given by the reviewer | 2 | 0.19% |
| record | location: City and country of the reviewer | 2 | 0.19% |
| record | text | 2 | 0.19% |
| record | age: Age of the patient in years | 2 | 0.19% |
| record | enrollment | 2 | 0.19% |
| record | passengers_affected | 2 | 0.19% |
| record | beliefs | 2 | 0.19% |
| record | actions | 2 | 0.19% |
| record | surface | 2 | 0.19% |
| record | entries | 2 | 0.19% |
| record | tournament_name | 2 | 0.19% |
| record | winner_ranking | 2 | 0.19% |
| record | opponent_name | 2 | 0.19% |
| record | physical_characteristics.observations.0.year | 2 | 0.19% |
| record | investor | 2 | 0.19% |
| record | customer_address | 2 | 0.19% |
| record | payer | 2 | 0.19% |
| record | disease.pathogenesis.transmission | 2 | 0.19% |
| record | disease.subspecies.2.name | 2 | 0.19% |
| record | disease.subspecies.0.affected_species | 2 | 0.19% |
| record | disease.subspecies.1.name | 2 | 0.19% |
| record | disease.description | 2 | 0.19% |
| record | disease.subspecies.1.associated_diseases | 2 | 0.19% |
| record | nightly_rate | 2 | 0.19% |
| record | room_type | 2 | 0.19% |
| record | person.name | 2 | 0.19% |
| record | person.occupation | 2 | 0.19% |
| record | schema_name | 2 | 0.19% |
| record | upvotes | 2 | 0.19% |
| record | article.title | 2 | 0.19% |
| record | attendee_count | 2 | 0.19% |
| record | lab_results | 2 | 0.19% |
| record | user_location | 2 | 0.19% |
| record | user: Name of the customer who provided the review | 2 | 0.19% |
| record | rating: Numerical rating from 1 to 5 given by the user | 2 | 0.19% |
| record | location: City and state or province where purchase occurred | 2 | 0.19% |
| record | risk_category | 2 | 0.19% |
| record | patient_mrn: Unique medical record number identifying the patient | 2 | 0.19% |
| record | discharge_instructions: Written guidance given to patient post-discharge | 2 | 0.19% |
| record | asteroid.discovery.discoverer | 2 | 0.19% |
| record | panel_topics | 2 | 0.19% |
| record | registration_cap | 2 | 0.19% |
| record | dates | 2 | 0.19% |
| record | sender_domain: Domain of the spoofed email address | 2 | 0.19% |
| record | status: Current status of the report | 2 | 0.19% |
| record | case_id | 2 | 0.19% |
| record | era_name | 2 | 0.19% |
| record | dynasty | 2 | 0.19% |
| record | project_url | 2 | 0.19% |
| record | contributor_count | 2 | 0.19% |
| record | ranking.location_type | 2 | 0.19% |
| record | response | 2 | 0.19% |
| record | website | 2 | 0.19% |
| record | device model | 2 | 0.19% |
| record | return_status: Whether the product was returned or kept | 2 | 0.19% |
| record | location: City and state where buyer resides | 2 | 0.19% |
| record | product: Product model identifier such as EBC-1234 or WAT-5678 | 2 | 0.19% |
| record | buyer: Name of the customer who made the purchase | 2 | 0.19% |
| record | feedback: Additional comments or issues noted in review | 2 | 0.19% |
| record | shipping_method: Type of shipping used | 2 | 0.19% |
| record | price: Price paid for the product in US dollars | 2 | 0.19% |
| record | stock | 2 | 0.19% |
| record | units_sold | 2 | 0.19% |
| record | seller | 2 | 0.19% |
| record | transfer_id | 2 | 0.19% |
| record | transfer_fee | 2 | 0.19% |
| record | preferredtenanttype | 2 | 0.19% |
| record | utilities_included | 2 | 0.19% |
| record | contact_method | 2 | 0.19% |
| record | additional_notes | 2 | 0.19% |
| record | comments | 2 | 0.19% |
| record | wallet | 2 | 0.19% |
| record | currency | 2 | 0.19% |
| record | claim_id: Unique identifier for each claim entry | 2 | 0.19% |
| record | operating_margin | 2 | 0.19% |
| record | company_code | 2 | 0.19% |
| record | problem_start_time | 2 | 0.19% |
| record | compliance | 2 | 0.19% |
| record | property_address: Full address of the leased property | 2 | 0.19% |
| record | primary_diagnosis: Main medical condition diagnosed | 2 | 0.19% |
| record | lab_id | 2 | 0.19% |
| record | places | 2 | 0.19% |
| record | supplier | 2 | 0.19% |
| record | people | 2 | 0.19% |
| record | path | 2 | 0.19% |
| record | substances | 2 | 0.19% |
| record | decor | 2 | 0.19% |
| record | food | 2 | 0.19% |
| record | affected_site | 2 | 0.19% |
| record | alternative_access | 2 | 0.19% |
| record | steals | 2 | 0.19% |
| record | rebounds | 2 | 0.19% |
| record | steps | 2 | 0.19% |
| record | launch_year: Year the company launched its platform | 2 | 0.19% |
| record | magazine.article.page_breakdown.2.page_number | 2 | 0.19% |
| record | magazine.article.page_breakdown.12.page_number | 2 | 0.19% |
| record | magazine.article.page_breakdown.9.page_number | 2 | 0.19% |
| record | magazine.article.page_breakdown.16.advertising_percentage | 2 | 0.19% |
| record | magazine.article.page_breakdown.13.advertisers | 2 | 0.19% |
| record | magazine.article.page_breakdown.12.advertising_percentage | 2 | 0.19% |
| record | magazine.article.page_breakdown.5.advertising_percentage | 2 | 0.19% |
| record | magazine.article.page_breakdown.13.advertising_percentage | 2 | 0.19% |
| record | magazine.article.page_breakdown.1.advertising_percentage | 2 | 0.19% |
| record | magazine.article.page_breakdown.8.advertisers | 2 | 0.19% |
| record | magazine.name | 2 | 0.19% |
| record | magazine.article.page_breakdown.6.advertisers | 2 | 0.19% |
| record | magazine.article.page_breakdown.0.page_number | 2 | 0.19% |
| record | magazine.article.page_breakdown.15.advertisers | 2 | 0.19% |
| record | magazine.article.page_breakdown.9.advertisers | 2 | 0.19% |
| record | magazine.article.page_breakdown.16.page_number | 2 | 0.19% |
| record | magazine.article.page_breakdown.10.advertisers | 2 | 0.19% |
| record | magazine.article.content_percentage | 2 | 0.19% |
| record | magazine.article.page_breakdown.3.page_number | 2 | 0.19% |
| record | magazine.article.page_breakdown.8.advertising_percentage | 2 | 0.19% |
| record | magazine.article.page_breakdown.0.advertisers | 2 | 0.19% |
| record | magazine.article.page_breakdown.8.page_number | 2 | 0.19% |
| record | description: Brief explanation of market event influencing the rate | 2 | 0.19% |
| record | currency_pair: Pair of currencies represented in the rate | 2 | 0.19% |
| record | hire_date | 2 | 0.19% |
| record | political_context | 2 | 0.19% |
| record | po_reference | 2 | 0.19% |
| record | arguments_for_using_gdz | 2 | 0.19% |
| record | creator | 2 | 0.19% |
| record | thumbnail | 2 | 0.19% |
| record | coverage_amount | 2 | 0.19% |
| record | place.location.province | 2 | 0.19% |
| record | place.demographics.0.year | 2 | 0.19% |
| record | place.location.department | 2 | 0.19% |
| record | place.name | 2 | 0.19% |
| record | hotel_experience.entertainment.tv.screen_size | 2 | 0.19% |
| record | hotel_experience.suites.0.suite_type | 2 | 0.19% |
| record | hotel_experience.suites.0.features | 2 | 0.19% |
| record | hotel_experience.location.city | 2 | 0.19% |
| record | hotel_experience.additional_benefits | 2 | 0.19% |
| record | refund_amount | 2 | 0.19% |
| record | confirmation_code | 2 | 0.19% |
| record | user_rating | 2 | 0.19% |
| record | payment_method | 2 | 0.19% |
| record | album_title | 2 | 0.19% |
| record | description: Brief text describing the content of the post | 2 | 0.19% |
| record | status: Current status of the trend repost | 2 | 0.19% |
| record | user_name: Name of the user who created the repost | 2 | 0.19% |
| record | trend_name: Name of the viral trend being reposted | 2 | 0.19% |
| record | shares: Number of shares the post received | 2 | 0.19% |
| record | vehicle | 2 | 0.19% |
| record | research.experiment_details.quantum_state | 2 | 0.19% |
| record | research.focus | 2 | 0.19% |
| record | research.researcher.affiliation.name | 2 | 0.19% |
| record | research.experiment_details.outcome | 2 | 0.19% |
| record | research.materials | 2 | 0.19% |
| record | research.temperature | 2 | 0.19% |
| record | research.researcher.name | 2 | 0.19% |
| record | labels | 2 | 0.19% |
| record | property_type | 2 | 0.19% |
| record | study_details.cohort.condition_studied | 2 | 0.19% |
| record | pathophysiology.etiology | 2 | 0.19% |
| record | pathophysiology.mechanism.secondary_event.anatomical_locations | 2 | 0.19% |
| record | outcomes_and_findings.prognosis.visual_recovery_potential | 2 | 0.19% |
| record | outcomes_and_findings.prognosis.critical_success_factors | 2 | 0.19% |
| record | study.title | 2 | 0.19% |
| record | study.quotes | 2 | 0.19% |
| record | study.key_findings | 2 | 0.19% |
| record | study.period.end_year | 2 | 0.19% |
| record | study.author.name | 2 | 0.19% |
| record | study.author.affiliation | 2 | 0.19% |
| record | key_concepts | 2 | 0.19% |
| record | contact_person | 2 | 0.19% |
| record | vendor_name | 2 | 0.19% |
| record | donor_id | 2 | 0.19% |
| record | clinical_observations.activation_status | 2 | 0.19% |
| record | splicing_regulation.impact_on_pathway_components.0.regulatory_mode | 2 | 0.19% |
| record | splicing_regulation.impact_on_pathway_components.0.component_description | 2 | 0.19% |
| record | employee_id: Unique identifier assigned to the employee | 2 | 0.19% |
| record | department: Department where the employee works | 2 | 0.19% |
| record | goal_achievement: Percentage of goals met, from 60% to 100% | 2 | 0.19% |
| record | shares | 2 | 0.19% |
| record | risk_score | 2 | 0.19% |
| record | winner | 2 | 0.19% |
| record | symptom | 2 | 0.19% |
| record | activities | 2 | 0.19% |
| record | study_metadata.study_type | 2 | 0.19% |
| record | study_metadata.measurement_technology.abbreviation | 2 | 0.19% |
| record | study_metadata.measurement_technology.full_name | 2 | 0.19% |
| record | study_metadata.measurement_technology.anatomical_site | 2 | 0.19% |
| record | population.gender | 2 | 0.19% |
| record | population.age_range.min | 2 | 0.19% |
| record | population.age_range.max | 2 | 0.19% |
| record | population.age_range.unit | 2 | 0.19% |
| record | follow_up.average_duration.years | 2 | 0.19% |
| record | outcomes.fracture_events.0.count | 2 | 0.19% |
| record | statistical_results.diagnostic_metrics.sensitivity_at_threshold.t_score_threshold | 2 | 0.19% |
| record | statistical_results.diagnostic_metrics.sensitivity_at_threshold.percentage_of_fractures | 2 | 0.19% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.0.t_score | 2 | 0.19% |
| record | statistical_results.predictive_model.value | 2 | 0.19% |
| record | conclusions | 2 | 0.19% |
| record | service.location.address | 2 | 0.19% |
| record | service.opening_hours.0.day | 2 | 0.19% |
| record | service.location.building | 2 | 0.19% |
| record | service.interventions | 2 | 0.19% |
| record | service.responsible_person.role | 2 | 0.19% |
| record | service.contact_information.email | 2 | 0.19% |
| record | product_category | 2 | 0.19% |
| record | location.region | 2 | 0.19% |
| record | election_type | 2 | 0.19% |
| record | candidate_location | 2 | 0.19% |
| record | candidate_name | 2 | 0.19% |
| record | comments: Total number of comments on the video | 2 | 0.19% |
| record | trend_duration: Number of days the trend remained active or trending | 2 | 0.19% |
| record | patent_id: Unique identifier for the patent application | 2 | 0.19% |
| record | dimensions | 2 | 0.19% |
| record | disclosure_id | 2 | 0.19% |
| record | discipline: Type of sport or competition such as swimming or skiing | 2 | 0.19% |
| record | discovery.observatory | 2 | 0.19% |
| record | frequency | 2 | 0.19% |
| record | cost | 2 | 0.19% |
| record | genus.period | 2 | 0.19% |
| record | genus.unique_features | 2 | 0.19% |
| record | genus.naming.named_by | 2 | 0.19% |
| record | bracket_path | 2 | 0.19% |
| record | team_name | 2 | 0.19% |
| record | registered_on | 2 | 0.19% |
| record | beneficiary | 2 | 0.19% |
| record | team | 2 | 0.19% |
| record | distance.parsecs | 2 | 0.19% |
| record | apparent_magnitude.max | 2 | 0.19% |
| record | characteristics.type | 2 | 0.19% |
| record | constellation | 2 | 0.19% |
| record | quantity: Number of tablets or units dispensed | 2 | 0.19% |
| record | status: Current state of prescription fulfillment | 2 | 0.19% |
| record | pharmacy_name: Name of the pharmacy that processed the prescription | 2 | 0.19% |
| record | dosage: Specific amount or form of medication prescribed | 2 | 0.19% |
| record | brands | 2 | 0.19% |
| record | approval_turnaround_days | 2 | 0.19% |
| record | provider_institution | 2 | 0.19% |
| record | customer_id | 2 | 0.19% |
| record | ticket_count | 2 | 0.19% |
| record | open_date | 2 | 0.19% |
| record | consultation.practice_areas | 2 | 0.19% |
| record | practice_areas | 2 | 0.19% |
| record | states_served | 2 | 0.19% |
| record | services | 2 | 0.19% |
| record | firm_name | 2 | 0.19% |
| record | client_focus | 2 | 0.19% |
| record | location: City where the treaty was signed | 2 | 0.19% |
| record | signatories: List of officials who co-signed the treaty | 2 | 0.19% |
| record | name: Unique identifier of the treaty agreement | 2 | 0.19% |
| record | review_id: Unique identifier for the review entry | 2 | 0.19% |
| record | size | 2 | 0.19% |
| record | amount: Monetary value of the returned item | 2 | 0.19% |
| record | product_name: Name of the product returned | 2 | 0.19% |
| record | order_id: Identifier of the original order associated with the return | 2 | 0.19% |
| record | return_id: Unique identifier for the return transaction | 2 | 0.19% |
| record | chemical_name | 2 | 0.19% |
| record | emergency_contact | 2 | 0.19% |
| record | count | 2 | 0.19% |
| record | expiry_date | 2 | 0.19% |
| record | zip | 2 | 0.19% |
| record | staff | 2 | 0.19% |
| record | species.scientific_name | 2 | 0.19% |
| record | species.genus | 2 | 0.19% |
| record | species.description | 2 | 0.19% |
| record | expression_level | 2 | 0.19% |
| record | billed_amount | 2 | 0.19% |
| record | asteroid.discovery_year | 2 | 0.19% |
| record | asteroid.orbital_characteristics.inclination_unit | 2 | 0.19% |
| record | entity | 2 | 0.19% |
| record | change | 2 | 0.19% |
| record | treatment_arm | 2 | 0.19% |
| record | lastfollowup | 2 | 0.19% |
| record | booking_options.tripadvisor_role | 2 | 0.19% |
| record | asteroid.name_latin | 2 | 0.19% |
| record | initial_symptoms | 2 | 0.19% |
| record | treatment | 2 | 0.19% |
| record | claim_amount | 2 | 0.19% |
| record | publication | 2 | 0.19% |
| record | invoice_id | 2 | 0.19% |
| record | date_reported | 2 | 0.19% |
| record | environment | 2 | 0.19% |
| record | listing_id: Unique identifier for each property listing | 2 | 0.19% |
| record | filing_date | 2 | 0.19% |
| record | client_name | 2 | 0.19% |
| record | client_address | 2 | 0.19% |
| record | due_date | 2 | 0.19% |
| record | artist | 2 | 0.19% |
| record | size_and_fit.length.unit | 2 | 0.19% |
| record | size_and_fit.measurements.2.value | 2 | 0.19% |
| record | price.installment_option.currency | 2 | 0.19% |
| record | size_and_fit.measurements.2.unit | 2 | 0.19% |
| record | size_and_fit.measurements.1.unit | 2 | 0.19% |
| record | size_and_fit.details | 2 | 0.19% |
| record | size_and_fit.measurements.2.type | 2 | 0.19% |
| record | credits | 2 | 0.19% |
| record | status: Current status of the appointment | 2 | 0.19% |
| record | patent_number | 2 | 0.19% |
| record | tech_product | 2 | 0.19% |
| record | tweet_id | 2 | 0.19% |
| record | exchange | 2 | 0.19% |
| record | device_name | 2 | 0.19% |
| record | account | 1 | 0.10% |
| record | id: Unique identifier for each rental listing | 1 | 0.10% |
| record | pet_policy: Pet policy including restrictions or fees | 1 | 0.10% |
| record | property_type: Type of housing unit | 1 | 0.10% |
| record | description: Brief description of unit features or conditions | 1 | 0.10% |
| record | address: Full property address including street and city | 1 | 0.10% |
| record | adverse_event | 1 | 0.10% |
| record | monthly_fee | 1 | 0.10% |
| record | document_details.series_title | 1 | 0.10% |
| record | document_details.description | 1 | 0.10% |
| record | document_details.oclc_number | 1 | 0.10% |
| record | match_code | 1 | 0.10% |
| record | total_funding | 1 | 0.10% |
| record | JJ | 1 | 0.10% |
| record | CD | 1 | 0.10% |
| record | DT | 1 | 0.10% |
| record | . | 1 | 0.10% |
| record | NNP | 1 | 0.10% |
| record | IN | 1 | 0.10% |
| record | CC | 1 | 0.10% |
| record | NN | 1 | 0.10% |
| record | symbols | 1 | 0.10% |
| record | determiners | 1 | 0.10% |
| record | verbs_past_participle | 1 | 0.10% |
| record | verbs_past_tense | 1 | 0.10% |
| record | prepositions | 1 | 0.10% |
| record | punctuation | 1 | 0.10% |
| record | possessive_endings | 1 | 0.10% |
| record | adjectives | 1 | 0.10% |
| record | singular_nouns | 1 | 0.10% |
| record | admit_diagnosis | 1 | 0.10% |
| record | lengthofstay | 1 | 0.10% |
| record | discharge_condition | 1 | 0.10% |
| record | participants | 1 | 0.10% |
| record | specificity | 1 | 0.10% |
| record | product_name: Name of the launched skincare or apparel product | 1 | 0.10% |
| record | price: Selling price of the product in US dollars | 1 | 0.10% |
| record | name: Full name of the technology startup | 1 | 0.10% |
| record | valuation: Post-money valuation after the round | 1 | 0.10% |
| record | ticket_id: Internal ticket identifier for the RMA | 1 | 0.10% |
| record | serial: Unique identifier for the product unit | 1 | 0.10% |
| record | rma_number: Assigned number for the RMA case | 1 | 0.10% |
| record | cost: Repair cost, if any, as stated in the request | 1 | 0.10% |
| record | customer: Name of the product owner submitting the request | 1 | 0.10% |
| record | merchant_code | 1 | 0.10% |
| record | dispute_status | 1 | 0.10% |
| record | owner_location | 1 | 0.10% |
| record | incident.suspects.0.age | 1 | 0.10% |
| record | incident.suspects.1.age | 1 | 0.10% |
| record | incident.charges.0.count | 1 | 0.10% |
| record | incident.charges.0.description | 1 | 0.10% |
| record | incident.charges.1.count | 1 | 0.10% |
| record | incident.charges.1.description | 1 | 0.10% |
| record | incident.charges.2.count | 1 | 0.10% |
| record | incident.charges.2.description | 1 | 0.10% |
| record | incident.additional_acts | 1 | 0.10% |
| record | incident.affected_locations | 1 | 0.10% |
| record | incident.number_of_complaints | 1 | 0.10% |
| record | source.newspaper | 1 | 0.10% |
| record | source.author | 1 | 0.10% |
| record | broadcast_network | 1 | 0.10% |
| record | media | 1 | 0.10% |
| record | number | 1 | 0.10% |
| record | production_location | 1 | 0.10% |
| record | ballot_type | 1 | 0.10% |
| record | dosage_range | 1 | 0.10% |
| record | age_range | 1 | 0.10% |
| record | rating: Star rating given by the reviewer out of 5 | 1 | 0.10% |
| record | status: Current status of the review | 1 | 0.10% |
| record | model: Unique identifier or model number of the device | 1 | 0.10% |
| record | word_count: Number of words in the review text | 1 | 0.10% |
| record | investor_name | 1 | 0.10% |
| record | ranking.total | 1 | 0.10% |
| record | diameter.unit | 1 | 0.10% |
| record | payee: Name of the entity receiving payment | 1 | 0.10% |
| record | description: Brief description of the transaction purpose | 1 | 0.10% |
| record | amount: Monetary value of the transaction | 1 | 0.10% |
| record | rate: Exchange rate in USD per foreign currency unit | 1 | 0.10% |
| record | location: Physical address where course is held | 1 | 0.10% |
| record | credits: Number of academic credits assigned to the course | 1 | 0.10% |
| record | lot_number | 1 | 0.10% |
| record | patient | 1 | 0.10% |
| record | platform | 1 | 0.10% |
| record | handle | 1 | 0.10% |
| record | restaurant_name | 1 | 0.10% |
| record | useoffunds | 1 | 0.10% |
| record | geographic_focus | 1 | 0.10% |
| record | bracket_id | 1 | 0.10% |
| record | result | 1 | 0.10% |
| record | hba1c | 1 | 0.10% |
| record | debttoebitda_ratio: Ratio of debt to EBITDA indicating leverage | 1 | 0.10% |
| record | sec_id | 1 | 0.10% |
| record | salary_range: Expected or offered salary range for the role | 1 | 0.10% |
| record | target_role: Job title or position being targeted | 1 | 0.10% |
| record | message_id: Unique identifier for the outreach message | 1 | 0.10% |
| record | target_company: Company where the role is located | 1 | 0.10% |
| record | issue_type | 1 | 0.10% |
| record | cause: Reason for the damage such as flood or fire | 1 | 0.10% |
| record | project_id: Unique identifier for the project in format GH-YYYY-NNN | 1 | 0.10% |
| record | reviewer_name: Name of the reviewer providing feedback | 1 | 0.10% |
| record | forks: Number of forks the repository has received | 1 | 0.10% |
| record | repository_url: GitHub URL hosting the project | 1 | 0.10% |
| record | review_score: Numerical score out of 5 based on user feedback | 1 | 0.10% |
| record | gren_id | 1 | 0.10% |
| record | mapping_period | 1 | 0.10% |
| record | owner: Name of the healthcare professional owning the device | 1 | 0.10% |
| record | location: Department or unit where device is currently located | 1 | 0.10% |
| record | device_id: Unique identifier assigned to the medical device | 1 | 0.10% |
| record | model: Manufacturer model name of the medical device | 1 | 0.10% |
| record | company.financial_results.reporting_period | 1 | 0.10% |
| record | report_metadata.source_url | 1 | 0.10% |
| record | filing_entity | 1 | 0.10% |
| record | checklist_id | 1 | 0.10% |
| record | medical_code | 1 | 0.10% |
| record | home_score: Final score for the home team | 1 | 0.10% |
| record | venue: Name of the arena where the game was played | 1 | 0.10% |
| record | product_focus | 1 | 0.10% |
| record | industry_segment | 1 | 0.10% |
| record | financial_figure | 1 | 0.10% |
| record | lead_analyst | 1 | 0.10% |
| record | location: Physical address of the product launch | 1 | 0.10% |
| record | product_code: Internal code assigned to the product | 1 | 0.10% |
| record | price: Selling price of the product in dollars | 1 | 0.10% |
| record | product: Name of the skincare or apparel product | 1 | 0.10% |
| record | role: Title of the job position being advertised | 1 | 0.10% |
| record | location: City and country where the role is based | 1 | 0.10% |
| record | company: Name of the company issuing the job posting | 1 | 0.10% |
| record | canton_name_french | 1 | 0.10% |
| record | canton_name | 1 | 0.10% |
| record | initial_funding | 1 | 0.10% |
| record | person.publication_information.available_part | 1 | 0.10% |
| record | person.commentary_based_on | 1 | 0.10% |
| record | person.publication_information.commentary_with | 1 | 0.10% |
| record | person.publication_information.location | 1 | 0.10% |
| record | person.time_period.era | 1 | 0.10% |
| record | person.associated_with | 1 | 0.10% |
| record | asteroid.dedication.person.nationality | 1 | 0.10% |
| record | suggestions | 1 | 0.10% |
| record | onboarding_stage | 1 | 0.10% |
| record | booth_size | 1 | 0.10% |
| record | exhibitor_id | 1 | 0.10% |
| record | registration_date | 1 | 0.10% |
| record | bill_number | 1 | 0.10% |
| record | semester | 1 | 0.10% |
| record | legislative_details.initiators | 1 | 0.10% |
| record | targetagegroup: Age range of intended audience | 1 | 0.10% |
| record | campaign_id: Unique identifier for each promotional campaign | 1 | 0.10% |
| record | office_hours.3.end_time | 1 | 0.10% |
| record | office_hours.1.end_time | 1 | 0.10% |
| record | contact.phone | 1 | 0.10% |
| record | office_hours.3.day | 1 | 0.10% |
| record | address.building | 1 | 0.10% |
| record | heart_rate | 1 | 0.10% |
| record | terms_and_conditions.combination_restriction | 1 | 0.10% |
| record | loyalty_program.name | 1 | 0.10% |
| record | discount.description | 1 | 0.10% |
| record | booking.promotional_code | 1 | 0.10% |
| record | hotel_chain | 1 | 0.10% |
| record | terms_and_conditions.availability | 1 | 0.10% |
| record | terms_and_conditions.disclaimer | 1 | 0.10% |
| record | booking.cancellation_policy | 1 | 0.10% |
| record | loyalty_program.membership_url | 1 | 0.10% |
| record | device_id | 1 | 0.10% |
| record | error.possible_causes | 1 | 0.10% |
| record | error.message | 1 | 0.10% |
| record | amount_spent | 1 | 0.10% |
| record | primary_treatment | 1 | 0.10% |
| record | record_id | 1 | 0.10% |
| record | billing_code | 1 | 0.10% |
| record | discharge_diagnosis | 1 | 0.10% |
| record | discharge_instructions | 1 | 0.10% |
| record | medals.silver | 1 | 0.10% |
| record | games | 1 | 0.10% |
| record | medals.total | 1 | 0.10% |
| record | political_entity | 1 | 0.10% |
| record | policyholder_id: Unique identifier for the insurance policyholder | 1 | 0.10% |
| record | claim_type: Type of medical procedure covered | 1 | 0.10% |
| record | diagnosis_code: Medical code associated with the claim condition | 1 | 0.10% |
| record | genus.name_zh | 1 | 0.10% |
| record | genus.name_latin | 1 | 0.10% |
| record | genus.family | 1 | 0.10% |
| record | genus.synonyms | 1 | 0.10% |
| record | genus.subspecies.0.authority | 1 | 0.10% |
| record | genus.subspecies.0.name | 1 | 0.10% |
| record | reviewer_location: City and state where the reviewer is located | 1 | 0.10% |
| record | battery_life: Battery duration noted by the reviewer | 1 | 0.10% |
| record | reviewer_name: Name of the customer who submitted the review | 1 | 0.10% |
| record | product_price: Price of the product in USD | 1 | 0.10% |
| record | address: Full address of the reviewer | 1 | 0.10% |
| record | holder_name | 1 | 0.10% |
| record | actions.0.target | 1 | 0.10% |
| record | actions.0.location | 1 | 0.10% |
| record | instruction | 1 | 0.10% |
| record | medical condition | 1 | 0.10% |
| record | medical procedure | 1 | 0.10% |
| record | hotel_details.year_renovated | 1 | 0.10% |
| record | location.distance_from_airport.unit | 1 | 0.10% |
| record | owner_contact | 1 | 0.10% |
| record | location.distance_from_city_center.unit | 1 | 0.10% |
| record | location.distance_from_city_center.value | 1 | 0.10% |
| record | policyholder_id: Unique identifier for the policyholder | 1 | 0.10% |
| record | policyholder_name: Full name of the claimant | 1 | 0.10% |
| record | claim_id: Unique identifier for the submitted claim | 1 | 0.10% |
| record | therapy_basis | 1 | 0.10% |
| record | treatment_methods.description | 1 | 0.10% |
| record | treatment_methods.selection_criteria | 1 | 0.10% |
| record | diseases_treated.scope | 1 | 0.10% |
| record | diseases_treated.exceptions | 1 | 0.10% |
| record | patient_care.station_assignment | 1 | 0.10% |
| record | patient_care.weekly_plan | 1 | 0.10% |
| record | patient_care.group_therapy.importance | 1 | 0.10% |
| record | patient_care.group_therapy.benefits | 1 | 0.10% |
| record | venue_id | 1 | 0.10% |
| record | players | 1 | 0.10% |
| record | author_information.location.city | 1 | 0.10% |
| record | author_information.family.pets.0.name | 1 | 0.10% |
| record | author_information.occupation | 1 | 0.10% |
| record | author_information.family.pets.0.type | 1 | 0.10% |
| record | author_information.location.state | 1 | 0.10% |
| record | author_information.military_service.status | 1 | 0.10% |
| record | asteroid.discovery.location | 1 | 0.10% |
| record | thread_id: Unique identifier for the support ticket | 1 | 0.10% |
| record | customer_name: Name of the customer filing the issue | 1 | 0.10% |
| record | email_address: Email address associated with the customer | 1 | 0.10% |
| record | date_submitted: Date the issue was submitted to support | 1 | 0.10% |
| record | amount_usd: Monetary amount related to the issue in USD | 1 | 0.10% |
| record | description: Detailed explanation of the issue and customer's concern | 1 | 0.10% |
| record | entity_location | 1 | 0.10% |
| record | main_competitor | 1 | 0.10% |
| record | bracket_size | 1 | 0.10% |
| record | training_modules | 1 | 0.10% |
| record | unit_id | 1 | 0.10% |
| record | route: Origin-destination route string like JFK-LAX | 1 | 0.10% |
| record | delay_reason: Human-readable description of delay cause | 1 | 0.10% |
| record | destination: Airport code where flight arrives | 1 | 0.10% |
| record | passenger_names: List of passenger names affected | 1 | 0.10% |
| record | compensation_offered: Amount offered in dollars per passenger | 1 | 0.10% |
| record | flight_number: Unique identifier for the delayed flight | 1 | 0.10% |
| record | policy_renewal_notice | 1 | 0.10% |
| record | policy_holder | 1 | 0.10% |
| record | merchant: Name of the merchant or store involved | 1 | 0.10% |
| record | other_collaborations | 1 | 0.10% |
| record | office | 1 | 0.10% |
| record | update_notes | 1 | 0.10% |
| record | sustainability.sourced | 1 | 0.10% |
| record | context.region | 1 | 0.10% |
| record | context.demographic_challenge | 1 | 0.10% |
| record | context.strategic_approach | 1 | 0.10% |
| record | lead_agency.name | 1 | 0.10% |
| record | lead_agency.abbreviation | 1 | 0.10% |
| record | lead_agency.responsibilities | 1 | 0.10% |
| record | lead_agency.primary_goal | 1 | 0.10% |
| record | project.name | 1 | 0.10% |
| record | project.abbreviation | 1 | 0.10% |
| record | project.financials.budget | 1 | 0.10% |
| record | project.financials.funding_source | 1 | 0.10% |
| record | project.timeline.duration_years | 1 | 0.10% |
| record | project.scope.setting_type | 1 | 0.10% |
| record | project.objectives | 1 | 0.10% |
| record | project.operational_model.key_roles | 1 | 0.10% |
| record | project.operational_model.stakeholders.0.entity_type | 1 | 0.10% |
| record | project.operational_model.stakeholders.0.description | 1 | 0.10% |
| record | project.operational_model.stakeholders.1.entity_type | 1 | 0.10% |
| record | project.operational_model.stakeholders.1.description | 1 | 0.10% |
| record | project.operational_model.stakeholders.2.entity_type | 1 | 0.10% |
| record | project.operational_model.stakeholders.2.description | 1 | 0.10% |
| record | outcomes.monitored_metrics | 1 | 0.10% |
| record | outcomes.preliminary_findings | 1 | 0.10% |
| record | outcomes.qualitative_impact | 1 | 0.10% |
| record | symptom_date | 1 | 0.10% |
| record | affiliations | 1 | 0.10% |
| record | study_start | 1 | 0.10% |
| record | study_end | 1 | 0.10% |
| record | primary_outcome | 1 | 0.10% |
| record | key_result | 1 | 0.10% |
| record | review.cons | 1 | 0.10% |
| record | review.overall_rating | 1 | 0.10% |
| record | product.size | 1 | 0.10% |
| record | product.ingredients | 1 | 0.10% |
| record | circular.subject | 1 | 0.10% |
| record | circular.ministry | 1 | 0.10% |
| record | outlet | 1 | 0.10% |
| record | service_type: Type of service impacted such as broadband or mobile | 1 | 0.10% |
| record | publicationDate | 1 | 0.10% |
| record | first_office | 1 | 0.10% |
| record | work_hours | 1 | 0.10% |
| record | funding_type | 1 | 0.10% |
| record | id: Unique identifier assigned to each competitor | 1 | 0.10% |
| record | hq: Headquarters city and state of the company | 1 | 0.10% |
| record | date: Date of the competitor record entry | 1 | 0.10% |
| record | revenue: Annual revenue in millions of dollars | 1 | 0.10% |
| record | market_share: Percentage of market share held by the company | 1 | 0.10% |
| record | status: Current operational or strategic status of the company | 1 | 0.10% |
| record | development_periods | 1 | 0.10% |
| record | see_also | 1 | 0.10% |
| record | stub_article.call_to_action | 1 | 0.10% |
| record | features: List of key technological or functional capabilities | 1 | 0.10% |
| record | price: Base price of the product in US dollars | 1 | 0.10% |
| record | regions: Geographic areas where the product is distributed or in use | 1 | 0.10% |
| record | name: Name of the product as introduced | 1 | 0.10% |
| record | potential | 1 | 0.10% |
| record | definition | 1 | 0.10% |
| record | impact | 1 | 0.10% |
| record | measurement.unit | 1 | 0.10% |
| record | measurement.abbreviation | 1 | 0.10% |
| record | measurement.size_comparison | 1 | 0.10% |
| record | examples | 1 | 0.10% |
| record | origin.originator | 1 | 0.10% |
| record | origin.year | 1 | 0.10% |
| record | origin.university | 1 | 0.10% |
| record | applications.industrial | 1 | 0.10% |
| record | applications.medical | 1 | 0.10% |
| record | transcription_factors | 1 | 0.10% |
| record | key_components | 1 | 0.10% |
| record | passenger_address | 1 | 0.10% |
| record | departure_original | 1 | 0.10% |
| record | departure_actual | 1 | 0.10% |
| record | imaging_date | 1 | 0.10% |
| record | followupdate | 1 | 0.10% |
| record | organizer: Person or group hosting the event | 1 | 0.10% |
| record | name: Title or name of the event | 1 | 0.10% |
| record | location: Physical address or venue of the event | 1 | 0.10% |
| record | price: Entry cost for attending the event | 1 | 0.10% |
| record | status: Current phase of the event lifecycle | 1 | 0.10% |
| record | event_id: Unique identifier for each event | 1 | 0.10% |
| record | defense_systems | 1 | 0.10% |
| record | time_references | 1 | 0.10% |
| record | weapons | 1 | 0.10% |
| record | cities | 1 | 0.10% |
| record | location: City and state where the role is based | 1 | 0.10% |
| record | company: Name of the company hiring | 1 | 0.10% |
| record | title: Job title such as Senior Data Analyst or Marketing Manager | 1 | 0.10% |
| record | tools: List of software or platforms used in the role | 1 | 0.10% |
| record | responsibilities: Key duties and tasks associated with the role | 1 | 0.10% |
| record | sales_volume | 1 | 0.10% |
| record | dimensions: Dimensions in cm format | 1 | 0.10% |
| record | name: Name of the product as listed | 1 | 0.10% |
| record | material: Primary materials used in construction | 1 | 0.10% |
| record | weight: Weight of the product in grams or kg | 1 | 0.10% |
| record | time_management | 1 | 0.10% |
| record | feedback_method | 1 | 0.10% |
| record | overall_score | 1 | 0.10% |
| record | diagnosis_code: Medical code assigned to the patient's condition | 1 | 0.10% |
| record | support_agent | 1 | 0.10% |
| record | disease | 1 | 0.10% |
| record | name: Full name of the startup company | 1 | 0.10% |
| record | location: City and state where the company is headquartered | 1 | 0.10% |
| record | achievements | 1 | 0.10% |
| record | career_summary.location | 1 | 0.10% |
| record | career_summary.roles | 1 | 0.10% |
| record | career_summary.commitment | 1 | 0.10% |
| record | future_plans | 1 | 0.10% |
| record | bloodpressurediastolic | 1 | 0.10% |
| record | bloodpressuresystolic | 1 | 0.10% |
| record | respiratory_rate | 1 | 0.10% |
| record | quote.author | 1 | 0.10% |
| record | edit_information.editor | 1 | 0.10% |
| record | investor_list | 1 | 0.10% |
| record | briefing_event | 1 | 0.10% |
| record | follow_up_interest | 1 | 0.10% |
| record | pilot_hubs | 1 | 0.10% |
| record | resolution_timeline | 1 | 0.10% |
| record | app_name | 1 | 0.10% |
| record | return_status | 1 | 0.10% |
| record | shipping_cost | 1 | 0.10% |
| record | severity: Level of reaction intensity | 1 | 0.10% |
| record | address: Complete address of the patient | 1 | 0.10% |
| record | dosage: Prescribed dosage and frequency of the drug | 1 | 0.10% |
| record | tracking_id | 1 | 0.10% |
| record | perpetrator.arrest_details.time | 1 | 0.10% |
| record | victim.relationship_to_perpetrator | 1 | 0.10% |
| record | crime.location.landmark | 1 | 0.10% |
| record | crime.motive | 1 | 0.10% |
| record | crime.weapon.source.stolen_from.name | 1 | 0.10% |
| record | investigation.status | 1 | 0.10% |
| record | victim.injury | 1 | 0.10% |
| record | perpetrator.legal_actions | 1 | 0.10% |
| record | perpetrator.name | 1 | 0.10% |
| record | victim.name | 1 | 0.10% |
| record | crime.weapon.model | 1 | 0.10% |
| record | product.vendor | 1 | 0.10% |
| record | procedure_code | 1 | 0.10% |
| record | payee: Name of the recipient or entity involved | 1 | 0.10% |
| record | assigned_staff: Healthcare professional assigned to manage the device | 1 | 0.10% |
| record | purchase_date: Date the device was acquired | 1 | 0.10% |
| record | serial_number: Unique device identifier assigned by manufacturer | 1 | 0.10% |
| record | expiration_date: Date after which device is no longer valid | 1 | 0.10% |
| record | location: Physical ward or room where device is located | 1 | 0.10% |
| record | purchase_location | 1 | 0.10% |
| record | status: Updated clinical status like stable, unstable, critical, or resolved | 1 | 0.10% |
| record | clinical_status: Current clinical condition such as normal or elevated | 1 | 0.10% |
| record | oxygen_saturation: Percentage of oxygen saturation in blood | 1 | 0.10% |
| record | underwriting_model | 1 | 0.10% |
| record | property | 1 | 0.10% |
| record | document.location | 1 | 0.10% |
| record | document.counterfeiting_statistics.job_losses | 1 | 0.10% |
| record | document.strategy_elements.3.area | 1 | 0.10% |
| record | document.commissioners.1.portfolio | 1 | 0.10% |
| record | document.strategy_elements.6.area | 1 | 0.10% |
| record | document.key_issues | 1 | 0.10% |
| record | document.strategy_elements.0.area | 1 | 0.10% |
| record | document.commissioners.1.name | 1 | 0.10% |
| record | document.commissioners.0.name | 1 | 0.10% |
| record | document.strategy_elements.4.area | 1 | 0.10% |
| record | document.title | 1 | 0.10% |
| record | document.related_documents | 1 | 0.10% |
| record | document.reference_id | 1 | 0.10% |
| record | document.economic_impact.brand_value | 1 | 0.10% |
| record | document.strategy_elements.1.area | 1 | 0.10% |
| record | note | 1 | 0.10% |
| record | review_journal: Name of the publication that reviewed the work | 1 | 0.10% |
| record | author: Name of the author or poet | 1 | 0.10% |
| record | location: Geographic location of the reviewer | 1 | 0.10% |
| record | publisher: Publishing house responsible for the work | 1 | 0.10% |
| record | review_id: Unique identifier assigned to the review | 1 | 0.10% |
| record | outcome | 1 | 0.10% |
| record | referee | 1 | 0.10% |
| record | agency_code | 1 | 0.10% |
| record | departure_city | 1 | 0.10% |
| record | departure_code | 1 | 0.10% |
| record | arrival_city | 1 | 0.10% |
| record | arrival_code | 1 | 0.10% |
| record | arrival_time | 1 | 0.10% |
| record | passenger_type | 1 | 0.10% |
| record | baggage_count | 1 | 0.10% |
| record | seat_class | 1 | 0.10% |
| record | remarks | 1 | 0.10% |
| record | other_youth_programs_with_congressional_charters | 1 | 0.10% |
| record | hotel.transportation.parking.0.name | 1 | 0.10% |
| record | hotel.location.city | 1 | 0.10% |
| record | hotel.transportation.driving_directions | 1 | 0.10% |
| record | hotel.transportation.airport | 1 | 0.10% |
| record | hotel.location.country | 1 | 0.10% |
| record | hotel.transportation.parking.0.walking_time | 1 | 0.10% |
| record | hotel.location.nearby_landmarks | 1 | 0.10% |
| record | hotel.transportation.parking.1.cost | 1 | 0.10% |
| record | hotel.location.postal_code | 1 | 0.10% |
| record | hotel.location.area_description | 1 | 0.10% |
| record | hotel.transportation.parking.1.name | 1 | 0.10% |
| record | hotel.transportation.parking.1.walking_time | 1 | 0.10% |
| record | hotel.contact.phone | 1 | 0.10% |
| record | hotel.transportation.parking.0.cost | 1 | 0.10% |
| record | buyer_name | 1 | 0.10% |
| record | store_reference | 1 | 0.10% |
| record | sound_quality | 1 | 0.10% |
| record | usability_score | 1 | 0.10% |
| record | audiofidelityscore | 1 | 0.10% |
| record | death.date_julian | 1 | 0.10% |
| record | birth.date_gregorian | 1 | 0.10% |
| record | works.type | 1 | 0.10% |
| record | birth.date_julian | 1 | 0.10% |
| record | occupation | 1 | 0.10% |
| record | teacher | 1 | 0.10% |
| record | death.location | 1 | 0.10% |
| record | father.occupation | 1 | 0.10% |
| record | birth.location | 1 | 0.10% |
| record | account_number | 1 | 0.10% |
| record | merchant_name | 1 | 0.10% |
| record | imaging_type | 1 | 0.10% |
| record | description.leaves.terminal_spine.color | 1 | 0.10% |
| record | description.rosette | 1 | 0.10% |
| record | systematics_and_distribution.location.country | 1 | 0.10% |
| record | description.inflorescence.type | 1 | 0.10% |
| record | description.flowers.arrangement | 1 | 0.10% |
| record | description.fruit.width_mm.max | 1 | 0.10% |
| record | description.leaves.color | 1 | 0.10% |
| record | systematics_and_distribution.location.state | 1 | 0.10% |
| record | systematics_and_distribution.similar_species | 1 | 0.10% |
| record | description.leaves.surface | 1 | 0.10% |
| record | systematics_and_distribution.first_description.year | 1 | 0.10% |
| record | systematics_and_distribution.distinguishing_features | 1 | 0.10% |
| record | systematics_and_distribution.location.elevation_m.max | 1 | 0.10% |
| record | description.leaves.length_cm.min | 1 | 0.10% |
| record | description.flowers.length_mm.min | 1 | 0.10% |
| record | description.leaves.width_cm.min | 1 | 0.10% |
| record | description.fruit.type | 1 | 0.10% |
| record | description.fruit.length_mm.min | 1 | 0.10% |
| record | description.leaves.margins | 1 | 0.10% |
| record | description.fruit.shape | 1 | 0.10% |
| record | description.fruit.width_mm.min | 1 | 0.10% |
| record | description.flowering_time | 1 | 0.10% |
| record | opposition_count | 1 | 0.10% |
| record | vote_count | 1 | 0.10% |
| record | funding_id: Unique identifier for each funding round | 1 | 0.10% |
| record | roundtype: Type of funding round | 1 | 0.10% |
| record | status: Current status of the funding round | 1 | 0.10% |
| record | company_name: Name of the startup receiving funding | 1 | 0.10% |
| record | name: Full title of the course including subject and focus | 1 | 0.10% |
| record | credit_hours: Number of credit hours for the course | 1 | 0.10% |
| record | policytype | 1 | 0.10% |
| record | operation_hours | 1 | 0.10% |
| record | renewal_amount | 1 | 0.10% |
| record | description: Technical summary of the patented invention | 1 | 0.10% |
| record | inventor: Name of the patent's inventor | 1 | 0.10% |
| record | address: Full address of the inventor | 1 | 0.10% |
| record | merchantcategorycode | 1 | 0.10% |
| record | closing_price | 1 | 0.10% |
| record | faq_references | 1 | 0.10% |
| record | navigation | 1 | 0.10% |
| record | footer_links | 1 | 0.10% |
| record | browse_topics | 1 | 0.10% |
| record | radius_km | 1 | 0.10% |
| record | sunless_tanning.overview | 1 | 0.10% |
| record | sunless_tanning.product_collection.collection_name | 1 | 0.10% |
| record | sunless_tanning.product_collection.products.0.name | 1 | 0.10% |
| record | sunless_tanning.product_collection.products.1.ingredients | 1 | 0.10% |
| record | sunless_tanning.product_collection.products.1.benefits | 1 | 0.10% |
| record | risks_of_sun_exposure | 1 | 0.10% |
| record | radiologist | 1 | 0.10% |
| record | ordering_physician | 1 | 0.10% |
| record | bp | 1 | 0.10% |
| record | doctor | 1 | 0.10% |
| record | rr | 1 | 0.10% |
| record | risk_score: Numerical value indicating the level of risk exposure | 1 | 0.10% |
| record | risk_category: Categorical classification of risk profile | 1 | 0.10% |
| record | policy_id: Identifier assigned to the insurance policy | 1 | 0.10% |
| record | correctiveactionplan | 1 | 0.10% |
| record | document_id | 1 | 0.10% |
| record | responsible_party | 1 | 0.10% |
| record | regulation_referenced | 1 | 0.10% |
| record | event_id: Unique identifier for the conference or symposium | 1 | 0.10% |
| record | birth_information.birth_year | 1 | 0.10% |
| record | profession | 1 | 0.10% |
| record | biography.education.0.teacher | 1 | 0.10% |
| record | biography.education.4.institution | 1 | 0.10% |
| record | biography.awards.0.award_name | 1 | 0.10% |
| record | biography.concert_tours | 1 | 0.10% |
| record | biography.education.4.teacher | 1 | 0.10% |
| record | biography.collaborations | 1 | 0.10% |
| record | biography.education.3.institution | 1 | 0.10% |
| record | biography.education.2.teacher | 1 | 0.10% |
| record | biography.teaching_position.institution | 1 | 0.10% |
| record | biography.teaching_position.mentor | 1 | 0.10% |
| record | birth_information.birth_place.state | 1 | 0.10% |
| record | birth_information.birth_place.county | 1 | 0.10% |
| record | isps | 1 | 0.10% |
| record | software_versions | 1 | 0.10% |
| record | orbital_characteristics.reference_plane | 1 | 0.10% |
| record | tissue_type | 1 | 0.10% |
| record | lineup | 1 | 0.10% |
| record | event_code | 1 | 0.10% |
| record | vip_price | 1 | 0.10% |
| record | state: State where the office is located | 1 | 0.10% |
| record | office_code: Unique internal identifier for the office location | 1 | 0.10% |
| record | city: City where the office is physically located | 1 | 0.10% |
| record | post_conference_event | 1 | 0.10% |
| record | student_rate | 1 | 0.10% |
| record | early_bird_discount_ends | 1 | 0.10% |
| record | current_registration_count | 1 | 0.10% |
| record | keynote_sessions | 1 | 0.10% |
| record | panel_discussions | 1 | 0.10% |
| record | tracks | 1 | 0.10% |
| record | core_values | 1 | 0.10% |
| record | registration_type | 1 | 0.10% |
| record | workshop_date | 1 | 0.10% |
| record | backup_plan | 1 | 0.10% |
| record | pre_conference_workshop | 1 | 0.10% |
| record | venue_capacity | 1 | 0.10% |
| record | event_status | 1 | 0.10% |
| record | lightning_talks | 1 | 0.10% |
| record | expected_revenue | 1 | 0.10% |
| record | senior_rate | 1 | 0.10% |
| record | location.distance | 1 | 0.10% |
| record | permit_holder | 1 | 0.10% |
| record | nepa_guidance | 1 | 0.10% |
| record | activity | 1 | 0.10% |
| record | additional_information | 1 | 0.10% |
| record | submitter: Name of employee who filed the ticket | 1 | 0.10% |
| record | priority: Severity level assigned to the ticket | 1 | 0.10% |
| record | ebitda_margin | 1 | 0.10% |
| record | liquidity_ratio | 1 | 0.10% |
| record | ceo | 1 | 0.10% |
| record | auditstatus | 1 | 0.10% |
| record | dividend_yield | 1 | 0.10% |
| record | acquired | 1 | 0.10% |
| record | value_original | 1 | 0.10% |
| record | regulatory_status | 1 | 0.10% |
| record | patents | 1 | 0.10% |
| record | customer_contracts | 1 | 0.10% |
| record | platform_name | 1 | 0.10% |
| record | earnout | 1 | 0.10% |
| record | sustainability_commitment | 1 | 0.10% |
| record | location: Property address including street, city, and state | 1 | 0.10% |
| record | question.text | 1 | 0.10% |
| record | question.keywords | 1 | 0.10% |
| record | question.legal_implications | 1 | 0.10% |
| record | details.date_posted | 1 | 0.10% |
| record | details.days_ago | 1 | 0.10% |
| record | output.unit | 1 | 0.10% |
| record | period | 1 | 0.10% |
| record | status: Current market availability status | 1 | 0.10% |
| record | target_market: Intended user segment or environment | 1 | 0.10% |
| record | name: Full name of the enrolled individual | 1 | 0.10% |
| record | enrollment_date: Date when enrollment began | 1 | 0.10% |
| record | program: Training or program name enrolled in | 1 | 0.10% |
| record | id: Unique identifier for the enrollment record | 1 | 0.10% |
| record | location: Geographic location of the enrollment site | 1 | 0.10% |
| record | methodology.population.total_department_nsclc_patients | 1 | 0.10% |
| record | patient_demographics.age_stats.min | 1 | 0.10% |
| record | patient_demographics.gender_distribution.men.count | 1 | 0.10% |
| record | patient_demographics.age_stats.average | 1 | 0.10% |
| record | methodology.study_period.end_year | 1 | 0.10% |
| record | pathological_findings.nodal_involvement_distribution.2.stage | 1 | 0.10% |
| record | treatment_details.pulmonary_resection_types.2.procedure | 1 | 0.10% |
| record | treatment_details.surgical_timeline.interval_brain_to_lung_resection_days.range_max | 1 | 0.10% |
| record | outcomes.mortality_by_nodal_involvement.1.survival_limit_months | 1 | 0.10% |
| record | methodology.population.inclusion_criteria | 1 | 0.10% |
| record | methodology.study_period.start_year | 1 | 0.10% |
| record | study_overview.aim | 1 | 0.10% |
| record | pathological_findings.statistical_factors.1.p_value | 1 | 0.10% |
| record | treatment_details.radiotherapy.regimen | 1 | 0.10% |
| record | methodology.population.study_group_percentage | 1 | 0.10% |
| record | treatment_details.pulmonary_resection_types.0.procedure | 1 | 0.10% |
| record | outcomes.survival_rates.two_year_percentage | 1 | 0.10% |
| record | outcomes.overall_survival_months.average | 1 | 0.10% |
| record | methodology.population.study_group_size | 1 | 0.10% |
| record | outcomes.disease_free_survival_months_average | 1 | 0.10% |
| record | pathological_findings.nodal_involvement_distribution.1.patient_count | 1 | 0.10% |
| record | outcomes.survival_rates.one_year_percentage | 1 | 0.10% |
| record | pathological_findings.statistical_factors.0.factor | 1 | 0.10% |
| record | pathological_findings.nodal_involvement_distribution.2.patient_count | 1 | 0.10% |
| record | outcomes.survival_rates.five_year_percentage | 1 | 0.10% |
| record | pathological_findings.nodal_involvement_distribution.0.stage | 1 | 0.10% |
| record | treatment_details.pulmonary_resection_types.1.count | 1 | 0.10% |
| record | pathological_findings.statistical_factors.0.p_value | 1 | 0.10% |
| record | treatment_details.surgical_timeline.interval_brain_to_lung_resection_days.range_min | 1 | 0.10% |
| record | treatment_details.pulmonary_resection_types.2.count | 1 | 0.10% |
| record | study_overview.disease | 1 | 0.10% |
| record | outcomes.mortality_by_nodal_involvement.0.survival_limit_months | 1 | 0.10% |
| record | patient_demographics.gender_distribution.women.percentage | 1 | 0.10% |
| record | application_id | 1 | 0.10% |
| record | technical_focus | 1 | 0.10% |
| record | monthly_fee: Monthly cost of the product subscription | 1 | 0.10% |
| record | target_population | 1 | 0.10% |
| record | visit_date | 1 | 0.10% |
| record | traveler_name | 1 | 0.10% |
| record | hotel_address | 1 | 0.10% |
| record | statue.name.french | 1 | 0.10% |
| record | statue.restoration.0.description | 1 | 0.10% |
| record | statue.restoration.0.funding.amount_raised.currency | 1 | 0.10% |
| record | statue.restoration.0.funding.source | 1 | 0.10% |
| record | statue.restoration.0.funding.contribution_range.currency | 1 | 0.10% |
| record | statue.discovery.discoverer.name | 1 | 0.10% |
| record | statue.discovery.location | 1 | 0.10% |
| record | statue.restoration.0.funding.contribution_range.min | 1 | 0.10% |
| record | statue.restoration.0.budget.currency | 1 | 0.10% |
| record | statue.discovery.date | 1 | 0.10% |
| record | statue.restoration.0.start_date | 1 | 0.10% |
| record | statue.transport_to_france | 1 | 0.10% |
| record | statue.public_display.stairs_completion_date | 1 | 0.10% |
| record | statue.location.museum | 1 | 0.10% |
| record | statue.material | 1 | 0.10% |
| record | statue.public_display.reopening_date | 1 | 0.10% |
| record | statue.location.city | 1 | 0.10% |
| record | statue.discovery.discoverer.title | 1 | 0.10% |
| record | statue.restoration.0.funding.number_of_contributors | 1 | 0.10% |
| record | statue.name.greek | 1 | 0.10% |
| record | statue.dedication | 1 | 0.10% |
| record | primates | 1 | 0.10% |
| record | course | 1 | 0.10% |
| record | copyright_note | 1 | 0.10% |
| record | contributor | 1 | 0.10% |
| record | entity_name: Company name associated with filing | 1 | 0.10% |
| record | filing_id: Unique identifier assigned to the filing | 1 | 0.10% |
| record | revenue: Total reported revenue in millions | 1 | 0.10% |
| record | net_income: Total reported net income in millions | 1 | 0.10% |
| record | address: Physical address of the company | 1 | 0.10% |
| record | product.language | 1 | 0.10% |
| record | customer_name: Name of the ticket owner reporting the issue | 1 | 0.10% |
| record | resolving_agent: Name of agent who resolved or handled the ticket | 1 | 0.10% |
| record | ticket_id: Unique identifier for the support ticket | 1 | 0.10% |
| record | customer_email: Email address associated with the ticket | 1 | 0.10% |
| record | status: Current state of the ticket after resolution | 1 | 0.10% |
| record | battery_life: Battery duration in hours under normal use | 1 | 0.10% |
| record | status: Current clinical status of the patient | 1 | 0.10% |
| record | address: Street address of the patient | 1 | 0.10% |
| record | template_substitution.prefix | 1 | 0.10% |
| record | advantages | 1 | 0.10% |
| record | template_substitution.examples | 1 | 0.10% |
| record | template_substitution.preview_code_example | 1 | 0.10% |
| record | templates_to_substitute | 1 | 0.10% |
| record | template_types.ordinary_template | 1 | 0.10% |
| record | investigators | 1 | 0.10% |
| record | funding_sources | 1 | 0.10% |
| record | secondary_endpoints | 1 | 0.10% |
| record | median_age | 1 | 0.10% |
| record | affiliation.institution_name | 1 | 0.10% |
| record | research_background.animal_models_used | 1 | 0.10% |
| record | affiliation.department | 1 | 0.10% |
| record | grant_id | 1 | 0.10% |
| record | message_id | 1 | 0.10% |
| record | outreach_status | 1 | 0.10% |
| record | outreach_code | 1 | 0.10% |
| record | boulder.start.foot_holds.1.foot | 1 | 0.10% |
| record | boulder.start.foot_holds.0.foot | 1 | 0.10% |
| record | boulder.start.hand_holds.1.description | 1 | 0.10% |
| record | boulder.start.hand_holds.0.description | 1 | 0.10% |
| record | boulder.start.foot_holds.0.description | 1 | 0.10% |
| record | boulder.start.hand_holds.1.hand | 1 | 0.10% |
| record | boulder.location.relative_position | 1 | 0.10% |
| record | boulder.location.distance_unit | 1 | 0.10% |
| record | boulder.location.reference_point | 1 | 0.10% |
| record | prior_hospitalization | 1 | 0.10% |
| record | primary_diagnosis | 1 | 0.10% |
| record | icd_code | 1 | 0.10% |
| record | treatment_history | 1 | 0.10% |
| record | technologies_mentioned | 1 | 0.10% |
| record | companies_mentioned | 1 | 0.10% |
| record | organizer | 1 | 0.10% |
| record | attendance | 1 | 0.10% |
| record | contact_email | 1 | 0.10% |
| record | test_code: Code assigned to the specific test | 1 | 0.10% |
| record | result_value: Measured value from the test | 1 | 0.10% |
| record | normal_range: Reference range for the measured value | 1 | 0.10% |
| record | test_date: Date the test was conducted | 1 | 0.10% |
| record | prescriber_id | 1 | 0.10% |
| record | prescriber_name | 1 | 0.10% |
| record | performance_time | 1 | 0.10% |
| record | audience_rating | 1 | 0.10% |
| record | festival_id | 1 | 0.10% |
| record | ticket_price | 1 | 0.10% |
| record | sponsorship | 1 | 0.10% |
| record | distributor_information.capabilities | 1 | 0.10% |
| record | services.maintenance | 1 | 0.10% |
| record | distributor_information.benefits | 1 | 0.10% |
| record | useoffunds: Purpose or application of raised capital | 1 | 0.10% |
| record | funding_id: Unique identifier assigned to the funding round | 1 | 0.10% |
| record | company: Name of the technology company receiving funds | 1 | 0.10% |
| record | meeting_id | 1 | 0.10% |
| record | resolution_summary | 1 | 0.10% |
| record | performance_rating | 1 | 0.10% |
| record | title_english | 1 | 0.10% |
| record | disease.subspecies.0.name | 1 | 0.10% |
| record | person.birth_date.year | 1 | 0.10% |
| record | person.birth_date.location | 1 | 0.10% |
| record | person.death_date.year | 1 | 0.10% |
| record | person.death_date.month | 1 | 0.10% |
| record | person.death_date.day | 1 | 0.10% |
| record | person.death_date.location | 1 | 0.10% |
| record | person.biography.early_life.family.status | 1 | 0.10% |
| record | person.biography.early_life.family.siblings.brothers | 1 | 0.10% |
| record | person.biography.early_life.family.siblings.sisters | 1 | 0.10% |
| record | person.biography.early_life.early_career | 1 | 0.10% |
| record | person.biography.marriage.spouse | 1 | 0.10% |
| record | person.biography.marriage.spouse_father | 1 | 0.10% |
| record | person.biography.marriage.spouse_father_occupation | 1 | 0.10% |
| record | person.biography.political_involvement.start_year | 1 | 0.10% |
| record | person.biography.political_involvement.description | 1 | 0.10% |
| record | person.biography.exile.year | 1 | 0.10% |
| record | person.biography.exile.reason | 1 | 0.10% |
| record | person.works.0.type | 1 | 0.10% |
| record | person.works.0.year | 1 | 0.10% |
| record | person.works.0.title | 1 | 0.10% |
| record | person.works.0.description | 1 | 0.10% |
| record | person.awards | 1 | 0.10% |
| record | claim_id: Unique identifier for each claim | 1 | 0.10% |
| record | patient_address: Full address including street, apt, city, state, zip | 1 | 0.10% |
| record | diagnosis_code: Standard code indicating the patient's condition | 1 | 0.10% |
| record | treatment_type: Type of medical treatment or procedure | 1 | 0.10% |
| record | claim_amount: Total monetary value of the claim | 1 | 0.10% |
| record | insurance_provider: Name of the insurance company covering the claim | 1 | 0.10% |
| record | status: Current processing status of the claim | 1 | 0.10% |
| record | payment_method: Method used to process payment for the claim | 1 | 0.10% |
| record | deductiblestatus: Whether patient's deductible has been met | 1 | 0.10% |
| record | co_payment: Patient's required co-payment amount | 1 | 0.10% |
| record | notes: Additional remarks or clinical context about the claim | 1 | 0.10% |
| record | target_audience: Demographic or user segment targeted by the campaign | 1 | 0.10% |
| record | product_name: Name of the product being promoted | 1 | 0.10% |
| record | post_id | 1 | 0.10% |
| record | subreddit | 1 | 0.10% |
| record | journal.title | 1 | 0.10% |
| record | journal.issn | 1 | 0.10% |
| record | article.author.name | 1 | 0.10% |
| record | article.author.givenName | 1 | 0.10% |
| record | article.author.familyName | 1 | 0.10% |
| record | article.publicationDetails.year | 1 | 0.10% |
| record | article.publicationDetails.volume | 1 | 0.10% |
| record | article.publicationDetails.issue | 1 | 0.10% |
| record | article.publicationDetails.pages | 1 | 0.10% |
| record | article.keywords | 1 | 0.10% |
| record | session_id | 1 | 0.10% |
| record | streaming_platform | 1 | 0.10% |
| record | end_time | 1 | 0.10% |
| record | id: Unique identifier for the legislative vote | 1 | 0.10% |
| record | date: Date when the vote occurred | 1 | 0.10% |
| record | chamber: Legislative chamber where the vote took place | 1 | 0.10% |
| record | bill: Bill number and type identifier | 1 | 0.10% |
| record | sponsor: Name of the bill's sponsor | 1 | 0.10% |
| record | outcome: Final result of the vote | 1 | 0.10% |
| record | amount: Financial amount associated with the bill | 1 | 0.10% |
| record | location: Geographic location of the bill's jurisdiction | 1 | 0.10% |
| record | status: Current status of the bill | 1 | 0.10% |
| record | description: Summary of the bill's purpose and scope | 1 | 0.10% |
| record | first_contact_resolution_rate | 1 | 0.10% |
| record | support_code | 1 | 0.10% |
| record | industry_focus | 1 | 0.10% |
| record | internal_code | 1 | 0.10% |
| record | revenue_estimate | 1 | 0.10% |
| record | removal_instructions.browser_specific_instructions.0.steps | 1 | 0.10% |
| record | infection_method | 1 | 0.10% |
| record | risks | 1 | 0.10% |
| record | removal_instructions.browser_specific_instructions.1.steps | 1 | 0.10% |
| record | affected_browsers | 1 | 0.10% |
| record | malware_name | 1 | 0.10% |
| record | removal_instructions.browser_specific_instructions.0.browser | 1 | 0.10% |
| record | removal_instructions.tools | 1 | 0.10% |
| record | prevention_tips | 1 | 0.10% |
| record | removal_instructions.browser_specific_instructions.2.browser | 1 | 0.10% |
| record | spoofed_domain | 1 | 0.10% |
| record | target_department | 1 | 0.10% |
| record | risk_classification | 1 | 0.10% |
| record | incident_type | 1 | 0.10% |
| record | disruption_duration | 1 | 0.10% |
| record | passenger_name: Name of the passenger on the flight | 1 | 0.10% |
| record | flight_no: Flight number as assigned by the airline | 1 | 0.10% |
| record | delay_minutes: Number of minutes flight is delayed | 1 | 0.10% |
| record | destination: Airport code where flight ends | 1 | 0.10% |
| record | email: Email address of the passenger | 1 | 0.10% |
| record | unique_id: Unique identifier for the delay entry | 1 | 0.10% |
| record | departure_date: Date flight departs from origin | 1 | 0.10% |
| record | clinic_name | 1 | 0.10% |
| record | product_id: Unique identifier for the product reviewed | 1 | 0.10% |
| record | purchase_date: Date when the product was purchased | 1 | 0.10% |
| record | price: Formatted price including dollar sign and cents | 1 | 0.10% |
| record | review: Text content of the customer's feedback on the product | 1 | 0.10% |
| record | upload_date | 1 | 0.10% |
| record | rating: Numerical score out of 5 given by the reviewer | 1 | 0.10% |
| record | publication: Name of the media outlet publishing the review | 1 | 0.10% |
| record | system_code | 1 | 0.10% |
| record | coverage_limit | 1 | 0.10% |
| record | property_address | 1 | 0.10% |
| record | user_name: Name of the customer involved in the support request | 1 | 0.10% |
| record | cause: Reason for the reported issue or system failure | 1 | 0.10% |
| record | resolution_method: Action taken to resolve the issue | 1 | 0.10% |
| record | support_agent: Name of the agent assigned to handle the case | 1 | 0.10% |
| record | primary_diagnosis: Clinical diagnosis code and description | 1 | 0.10% |
| record | contribution_type | 1 | 0.10% |
| record | mls_number | 1 | 0.10% |
| record | research_project | 1 | 0.10% |
| record | principal_investigator | 1 | 0.10% |
| record | project_status | 1 | 0.10% |
| record | protecting_group.targets.1.abbreviation | 1 | 0.10% |
| record | protecting_group.targets.1.amino_acid_name | 1 | 0.10% |
| record | protecting_group.targets.1.derivative_form | 1 | 0.10% |
| record | protecting_group.targets.0.amino_acid_name | 1 | 0.10% |
| record | protecting_group.abbreviation | 1 | 0.10% |
| record | protecting_group.targets.1.protected_functional_group | 1 | 0.10% |
| record | deprotection_process.reagents.0.concentration_level | 1 | 0.10% |
| record | protecting_group.name | 1 | 0.10% |
| record | synthesis_context.intermediates_synthesized.0.components | 1 | 0.10% |
| record | protecting_group.targets.0.abbreviation | 1 | 0.10% |
| record | synthesis_context.methodology | 1 | 0.10% |
| record | synthesis_context.carboxyl_protection_requirement.reasoning | 1 | 0.10% |
| record | synthesis_context.intermediates_synthesized.0.type | 1 | 0.10% |
| record | protecting_group.targets.0.protected_functional_group | 1 | 0.10% |
| record | asteroid.provisional_designation | 1 | 0.10% |
| record | asteroid.discovery.observatory.name | 1 | 0.10% |
| record | asteroid.discovery.observatory.location.description | 1 | 0.10% |
| record | asteroid.discovery.observatory.location.mountains | 1 | 0.10% |
| record | asteroid.discovery.observatory.location.city | 1 | 0.10% |
| record | asteroid.discovery.observatory.location.state | 1 | 0.10% |
| record | asteroid.orbital_characteristics.jpl_classification | 1 | 0.10% |
| record | asteroid.physical_characteristics.observations.0.publication.year | 1 | 0.10% |
| record | asteroid.physical_characteristics.observations.0.publication.description | 1 | 0.10% |
| record | asteroid.physical_characteristics.observations.0.telescope.name | 1 | 0.10% |
| record | asteroid.physical_characteristics.observations.0.telescope.acronym | 1 | 0.10% |
| record | asteroid.physical_characteristics.observations.0.telescope.type | 1 | 0.10% |
| record | asteroid.physical_characteristics.observations.0.telescope.year_launched | 1 | 0.10% |
| record | asteroid.physical_characteristics.observations.0.telescope.wavelength | 1 | 0.10% |
| record | judge | 1 | 0.10% |
| record | case_type | 1 | 0.10% |
| record | plaintiff | 1 | 0.10% |
| record | defendant | 1 | 0.10% |
| record | project_type | 1 | 0.10% |
| record | dosage_form: Physical form of the drug such as tablet or injection | 1 | 0.10% |
| record | routeofadministration: Method by which the drug is administered | 1 | 0.10% |
| record | drug: Name of the medication as listed in the database | 1 | 0.10% |
| record | approved_date: Date the drug received regulatory approval | 1 | 0.10% |
| record | patient_population: Group of patients for whom the drug is intended | 1 | 0.10% |
| record | investors | 1 | 0.10% |
| record | funding_id | 1 | 0.10% |
| record | cpt_code: CPT code for the medical procedure performed | 1 | 0.10% |
| record | paid_amount: Amount actually received from insurance | 1 | 0.10% |
| record | diagnosis: ICD-10 code representing the medical condition | 1 | 0.10% |
| record | billed_amount: Total amount billed for the service | 1 | 0.10% |
| record | leasing_offer | 1 | 0.10% |
| record | tenant_name | 1 | 0.10% |
| record | drug_id | 1 | 0.10% |
| record | service_description | 1 | 0.10% |
| record | reaction | 1 | 0.10% |
| record | drug_involved | 1 | 0.10% |
| record | inr_value | 1 | 0.10% |
| record | developer | 1 | 0.10% |
| record | place | 1 | 0.10% |
| record | from_club | 1 | 0.10% |
| record | to_club | 1 | 0.10% |
| record | transfer_code | 1 | 0.10% |
| record | transfer_portal | 1 | 0.10% |
| record | temperature: Body temperature in degrees Celsius | 1 | 0.10% |
| record | respiratory_rate: Number of breaths per minute | 1 | 0.10% |
| record | monitoring_time: Exact time of monitoring in 24-hour format | 1 | 0.10% |
| record | device_model: Model of the monitoring device used | 1 | 0.10% |
| record | percentage | 1 | 0.10% |
| record | founder_id | 1 | 0.10% |
| record | boiling_instructions.cooking_time.min_minutes | 1 | 0.10% |
| record | grilling_instructions.method | 1 | 0.10% |
| record | winner_team | 1 | 0.10% |
| record | assigned_judge | 1 | 0.10% |
| record | tv_program.production.location.name | 1 | 0.10% |
| record | tv_program.production.production_area | 1 | 0.10% |
| record | tv_program.production.location.institution | 1 | 0.10% |
| record | tv_program.setting.year | 1 | 0.10% |
| record | tv_program.setting.location | 1 | 0.10% |
| record | tv_program.year | 1 | 0.10% |
| record | tv_program.previous_program | 1 | 0.10% |
| record | hotel.name | 1 | 0.10% |
| record | appointment_type | 1 | 0.10% |
| record | posted_on | 1 | 0.10% |
| record | property_id | 1 | 0.10% |
| record | thumbnail_style | 1 | 0.10% |
| record | channel_name | 1 | 0.10% |
| record | description_length | 1 | 0.10% |
| record | tx_id | 1 | 0.10% |
| record | policyholder: Name of the vehicle owner filing the claim | 1 | 0.10% |
| record | address: Full street address of the policyholder | 1 | 0.10% |
| record | vehicle: Make and model of the vehicle involved | 1 | 0.10% |
| record | collision_type: Type of collision or damage event | 1 | 0.10% |
| record | damage_type: Specific parts damaged in the incident | 1 | 0.10% |
| record | adjuster: Name of the insurance adjuster assigned | 1 | 0.10% |
| record | location: Geographic location where the incident occurred | 1 | 0.10% |
| record | subject_line | 1 | 0.10% |
| record | office: Regional office location where employee is based | 1 | 0.10% |
| record | manager: Name of the employee's manager | 1 | 0.10% |
| record | name: Full name of the new hire | 1 | 0.10% |
| record | assigned_coordinator: Name of the onboarding coordinator assigned to the employee | 1 | 0.10% |
| record | medium | 1 | 0.10% |
| record | artistic_intent.meaning_zero_point | 1 | 0.10% |
| record | artistic_intent.style | 1 | 0.10% |
| record | bill_name | 1 | 0.10% |
| record | participation | 1 | 0.10% |
| record | founder_names | 1 | 0.10% |
| record | sentiment | 1 | 0.10% |
| record | investor_group | 1 | 0.10% |
| record | statement_reference: Identifier code for the statement document | 1 | 0.10% |
| record | amount: Financial value associated with the policy or proposal | 1 | 0.10% |
| record | location: City and state where the event occurred | 1 | 0.10% |
| record | topic: Subject matter discussed in the statement | 1 | 0.10% |
| record | speaker: Name of the person delivering the statement | 1 | 0.10% |
| record | offer_id: Unique identifier for each leasing offer | 1 | 0.10% |
| record | tenant_name: Name of the leasing company or business | 1 | 0.10% |
| record | special_features: List of unique space features or amenities | 1 | 0.10% |
| record | encounter_id: Unique identifier for the visit | 1 | 0.10% |
| record | address: Patient's residential address | 1 | 0.10% |
| record | visit_date: Date of the medical encounter | 1 | 0.10% |
| record | icd_code: ICD-10 code for the primary diagnosis | 1 | 0.10% |
| record | bp_systolic: Systolic blood pressure in mmHg | 1 | 0.10% |
| record | bp_diastolic: Diastolic blood pressure in mmHg | 1 | 0.10% |
| record | medications: List of prescribed medications with dosing | 1 | 0.10% |
| record | lab_results: List of lab test results | 1 | 0.10% |
| record | symptoms: Patient-reported symptoms or complaints | 1 | 0.10% |
| record | test_date | 1 | 0.10% |
| record | product.technology | 1 | 0.10% |
| record | context.problem | 1 | 0.10% |
| record | context.solution | 1 | 0.10% |
| record | product.intended_use | 1 | 0.10% |
| record | product.browser_compatibility | 1 | 0.10% |
| record | product.download_link | 1 | 0.10% |
| record | winner_name | 1 | 0.10% |
| record | winner_nationality | 1 | 0.10% |
| record | tournament_type | 1 | 0.10% |
| record | lastcommitdate | 1 | 0.10% |
| record | tehsil.tehsil_name | 1 | 0.10% |
| record | tehsil.district | 1 | 0.10% |
| record | responsible_staff | 1 | 0.10% |
| record | causeofloss | 1 | 0.10% |
| record | damage_amount | 1 | 0.10% |
| record | appearances: Total games played in last season | 1 | 0.10% |
| record | contract_length: Duration of contract in years | 1 | 0.10% |
| record | position: Playing position such as midfielder, forward, defender | 1 | 0.10% |
| record | club_from: Previous club before transfer | 1 | 0.10% |
| record | club_to: New club after transfer | 1 | 0.10% |
| record | status: Current status of transfer | 1 | 0.10% |
| record | player_id: Unique identifier assigned to the player transfer record | 1 | 0.10% |
| record | age: Player's age at time of transfer | 1 | 0.10% |
| record | natural_elements | 1 | 0.10% |
| record | lighting | 1 | 0.10% |
| record | sensory | 1 | 0.10% |
| record | rooms | 1 | 0.10% |
| record | hotel.location.nearby_attractions | 1 | 0.10% |
| record | hotel.hotel_group | 1 | 0.10% |
| record | hotel.brand | 1 | 0.10% |
| record | hotel.location.distance_from_center | 1 | 0.10% |
| record | hotel.room_features | 1 | 0.10% |
| record | access_method | 1 | 0.10% |
| record | accessInstructions | 1 | 0.10% |
| record | surveyFocus | 1 | 0.10% |
| record | surveyPurpose | 1 | 0.10% |
| record | expectedOutcome | 1 | 0.10% |
| record | game_id | 1 | 0.10% |
| record | location: City and state where company is headquartered | 1 | 0.10% |
| record | investors: List of venture capital firms involved | 1 | 0.10% |
| record | company: Name of the company raising capital | 1 | 0.10% |
| record | round: Series or type of funding round | 1 | 0.10% |
| record | amount: Total amount raised in millions | 1 | 0.10% |
| record | early_bird_deadline | 1 | 0.10% |
| record | politician | 1 | 0.10% |
| record | label | 1 | 0.10% |
| record | recipient: Name of the recipient professional contacted | 1 | 0.10% |
| record | message_length: Word count of the outreach message | 1 | 0.10% |
| record | subject: Subject line of the outreach email | 1 | 0.10% |
| record | email_id: Unique identifier for the outreach email | 1 | 0.10% |
| record | sender: Name of the recruiter initiating the outreach | 1 | 0.10% |
| record | foods_mentioned | 1 | 0.10% |
| record | service | 1 | 0.10% |
| record | resolution_team | 1 | 0.10% |
| record | product_id: Unique identifier assigned by the manufacturer | 1 | 0.10% |
| record | name: Product model name as branded | 1 | 0.10% |
| record | medal | 1 | 0.10% |
| record | birth_place | 1 | 0.10% |
| record | competition_year | 1 | 0.10% |
| record | result_rank | 1 | 0.10% |
| record | event_purpose | 1 | 0.10% |
| record | rescheduled_event_date | 1 | 0.10% |
| record | event_rescheduled_to | 1 | 0.10% |
| record | magazine.article.page_breakdown.9.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.2.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.3.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.5.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.14.advertisers | 1 | 0.10% |
| record | magazine.mentioned_article.author | 1 | 0.10% |
| record | magazine.article.page_breakdown.7.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.10.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.2.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.5.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.4.page_number | 1 | 0.10% |
| record | magazine.article.title | 1 | 0.10% |
| record | magazine.mentioned_article.year | 1 | 0.10% |
| record | magazine.article.page_breakdown.13.page_number | 1 | 0.10% |
| record | magazine.article.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.14.advertising_percentage | 1 | 0.10% |
| record | key_achievement | 1 | 0.10% |
| record | review_period | 1 | 0.10% |
| record | event.description | 1 | 0.10% |
| record | event.organizer.name | 1 | 0.10% |
| record | organization_info.electricity_source | 1 | 0.10% |
| record | event.organizer.abbreviation | 1 | 0.10% |
| record | organization_info.funding | 1 | 0.10% |
| record | event.location.city | 1 | 0.10% |
| record | organization_info.facilities.generating_facilities | 1 | 0.10% |
| record | event.time.photo_op_end | 1 | 0.10% |
| record | event.activities | 1 | 0.10% |
| record | organization_info.website | 1 | 0.10% |
| record | event.name | 1 | 0.10% |
| record | event.time.photo_op_start | 1 | 0.10% |
| record | donations.recipient.type | 1 | 0.10% |
| record | event.time.end | 1 | 0.10% |
| record | event.location.address | 1 | 0.10% |
| record | outage_start | 1 | 0.10% |
| record | outage_end | 1 | 0.10% |
| record | event.person.name | 1 | 0.10% |
| record | event.new_club.contract_duration | 1 | 0.10% |
| record | event.person.age | 1 | 0.10% |
| record | event.new_club.name | 1 | 0.10% |
| record | event.opponent_club.name | 1 | 0.10% |
| record | event.person.position | 1 | 0.10% |
| record | event.person.role | 1 | 0.10% |
| record | amount: Numerical value of the exchange rate | 1 | 0.10% |
| record | budget_remaining | 1 | 0.10% |
| record | actual_spend | 1 | 0.10% |
| record | remaining_budget | 1 | 0.10% |
| record | onboarding_sprint_completion | 1 | 0.10% |
| record | risk_code | 1 | 0.10% |
| record | backend_integration_status | 1 | 0.10% |
| record | sku: unique product identifier from supply chain database | 1 | 0.10% |
| record | product_name: name of the physical item such as earbuds or lantern | 1 | 0.10% |
| record | warehouse_id: unique identifier for the storage facility | 1 | 0.10% |
| record | warehouse_location: location name of the warehouse facility | 1 | 0.10% |
| record | quantity: number of units currently in stock | 1 | 0.10% |
| record | disease_profile.inheritance_pattern | 1 | 0.10% |
| record | disease_profile.classification | 1 | 0.10% |
| record | disease_profile.typical_manifestations | 1 | 0.10% |
| record | disease_profile.genetic_mapping.associated_gene | 1 | 0.10% |
| record | disease_profile.genetic_mapping.gene_aliases | 1 | 0.10% |
| record | patient_case.demographics.age | 1 | 0.10% |
| record | patient_case.demographics.ethnicity | 1 | 0.10% |
| record | patient_case.clinical_presentation.skin_lesions.size_range.max_mm | 1 | 0.10% |
| record | patient_case.genetic_analysis.gene_tested | 1 | 0.10% |
| record | patient_case.genetic_analysis.mutation_details.protein_nomenclature | 1 | 0.10% |
| record | report_context.clinical_recommendations | 1 | 0.10% |
| record | county | 1 | 0.10% |
| record | division | 1 | 0.10% |
| record | kpivendorcompliance: Percentage of vendor contract adherence met | 1 | 0.10% |
| record | manager_name: Name of the employee's direct manager | 1 | 0.10% |
| record | location: Physical address of the employee's workplace | 1 | 0.10% |
| record | customer_satisfaction: Score from customer feedback surveys | 1 | 0.10% |
| record | kpideliveryaccuracy: Percentage of on-time deliveries achieved | 1 | 0.10% |
| record | performance_rating: Numerical score on a 5-point scale reflecting performance | 1 | 0.10% |
| record | organizer_name: Name of the person or group organizing the event | 1 | 0.10% |
| record | attendee_count: Number of registered attendees | 1 | 0.10% |
| record | grade | 1 | 0.10% |
| record | status: Current state of the incident | 1 | 0.10% |
| record | customer_name: Name of the affected customer | 1 | 0.10% |
| record | address: Full residential or commercial address of the customer | 1 | 0.10% |
| record | impact: Description of service impact including data loss or business effects | 1 | 0.10% |
| record | asteroid.orbital_characteristics.reference_plane | 1 | 0.10% |
| record | market_cap | 1 | 0.10% |
| record | place.type | 1 | 0.10% |
| record | place.demographics.0.population | 1 | 0.10% |
| record | shipping_cost: Cost to ship the audio device in US dollars | 1 | 0.10% |
| record | material: Material used in device construction | 1 | 0.10% |
| record | weight: Weight of the device in grams | 1 | 0.10% |
| record | hotel_experience.description | 1 | 0.10% |
| record | hotel_experience.suites.0.suite_count | 1 | 0.10% |
| record | hotel_experience.location.postal_code_accessibility | 1 | 0.10% |
| record | status: Current status of the exam result | 1 | 0.10% |
| record | school: Name of the school where the exam was administered | 1 | 0.10% |
| record | score: Percentage score achieved in the exam | 1 | 0.10% |
| record | subject: Subject area of the exam | 1 | 0.10% |
| record | reportTitle | 1 | 0.10% |
| record | credits.graphic_design.0.copyright_year | 1 | 0.10% |
| record | credits.contributors.0.person | 1 | 0.10% |
| record | credits.contributors.0.role | 1 | 0.10% |
| record | credits.organization | 1 | 0.10% |
| record | credits.content_implementers | 1 | 0.10% |
| record | credits.contributors.1.role | 1 | 0.10% |
| record | credits.graphic_design.0.designer | 1 | 0.10% |
| record | credits.graphic_design.0.area | 1 | 0.10% |
| record | credits.advertising_revenue_use | 1 | 0.10% |
| record | credits.graphic_design.1.copyright_year | 1 | 0.10% |
| record | credits.creation_date | 1 | 0.10% |
| record | github_url | 1 | 0.10% |
| record | claim_id: Unique identifier for the claim as referenced in correspondence | 1 | 0.10% |
| record | source_office: Clinic or office that submitted the claim | 1 | 0.10% |
| record | procedure_code: CPT code describing the medical service performed | 1 | 0.10% |
| record | event-attack/battle/war/militaryconflict | 1 | 0.10% |
| record | originalordernumber | 1 | 0.10% |
| record | product_title | 1 | 0.10% |
| record | return_reason | 1 | 0.10% |
| record | shipping_method | 1 | 0.10% |
| record | processing_code | 1 | 0.10% |
| record | delivery_method | 1 | 0.10% |
| record | subscription_amount | 1 | 0.10% |
| record | receiver | 1 | 0.10% |
| record | street | 1 | 0.10% |
| record | resolution_status | 1 | 0.10% |
| record | destination_club | 1 | 0.10% |
| record | origin_club | 1 | 0.10% |
| record | player_age | 1 | 0.10% |
| record | appearances | 1 | 0.10% |
| record | playerbirthcity | 1 | 0.10% |
| record | rma_number | 1 | 0.10% |
| record | failure_reason | 1 | 0.10% |
| record | case_report.clinical_presentation.lesion_locations | 1 | 0.10% |
| record | disease_profile.prognosis.expected_outcome | 1 | 0.10% |
| record | case_report.diagnostics.trichoscopy.findings.1.feature | 1 | 0.10% |
| record | case_report.patient.recent_medical_events.0.duration_days | 1 | 0.10% |
| record | disease_profile.risk_factors | 1 | 0.10% |
| record | case_report.clinical_presentation.lesion_description.1.features | 1 | 0.10% |
| record | case_report.patient.medical_history | 1 | 0.10% |
| record | case_report.clinical_presentation.lesion_description.0.features | 1 | 0.10% |
| record | disease_profile.synonyms | 1 | 0.10% |
| record | case_report.diagnostics.trichoscopy.findings.0.location_on_lesion | 1 | 0.10% |
| record | case_report.diagnostics.trichoscopy.findings.2.feature | 1 | 0.10% |
| record | case_report.clinical_presentation.lesion_description.1.count | 1 | 0.10% |
| record | disease_profile.pathophysiology | 1 | 0.10% |
| record | case_report.diagnostics.histopathology.findings | 1 | 0.10% |
| record | disease_profile.differential_diagnoses.1.overlapping_features | 1 | 0.10% |
| record | disease_profile.differential_diagnoses.1.disease_name | 1 | 0.10% |
| record | disease_profile.differential_diagnoses.0.disease_name | 1 | 0.10% |
| record | disease_profile.differential_diagnoses.0.overlapping_features | 1 | 0.10% |
| record | delivery_frequency | 1 | 0.10% |
| record | headlines | 1 | 0.10% |
| record | awayteampoints | 1 | 0.10% |
| record | blocks | 1 | 0.10% |
| record | planning | 1 | 0.10% |
| record | startup | 1 | 0.10% |
| record | funding_round | 1 | 0.10% |
| record | release_target | 1 | 0.10% |
| record | album_series | 1 | 0.10% |
| record | post_date: Date when the trend repost was published | 1 | 0.10% |
| record | sku: Stock keeping unit assigned to each product | 1 | 0.10% |
| record | seller_location: Full address of the seller including city, state, and zip code | 1 | 0.10% |
| record | unit_price: Price per unit of the product | 1 | 0.10% |
| record | primary_provider: Name of the attending physician | 1 | 0.10% |
| record | admission_diagnosis: Primary diagnosis at admission | 1 | 0.10% |
| record | discharge_date: Date the patient was discharged | 1 | 0.10% |
| record | medications: List of prescribed medications | 1 | 0.10% |
| record | population.count | 1 | 0.10% |
| record | geography.coordinates.latitude | 1 | 0.10% |
| record | geography.coordinates.longitude | 1 | 0.10% |
| record | census_inclusion | 1 | 0.10% |
| record | urgency | 1 | 0.10% |
| record | code | 1 | 0.10% |
| record | issue: Description of the reported hardware failure | 1 | 0.10% |
| record | filed_date: Date when the ticket was submitted | 1 | 0.10% |
| record | address: Full physical address of the customer | 1 | 0.10% |
| record | location: City and state where the customer resides | 1 | 0.10% |
| record | devices | 1 | 0.10% |
| record | physiological_measures | 1 | 0.10% |
| record | applications | 1 | 0.10% |
| record | waterway | 1 | 0.10% |
| record | nationality | 1 | 0.10% |
| record | military unit | 1 | 0.10% |
| record | island | 1 | 0.10% |
| record | recipient_jurisdiction: Jurisdiction such as city, state, or federal level | 1 | 0.10% |
| record | donor_location: City and state of the donor | 1 | 0.10% |
| record | contribution_id: Unique identifier for the contribution record | 1 | 0.10% |
| record | acquired_company: Name of the company being acquired | 1 | 0.10% |
| record | sector: Industry or market sector involved in the acquisition | 1 | 0.10% |
| record | acquirer_ceo: CEO of the acquiring company | 1 | 0.10% |
| record | deal_id: Unique identifier for the acquisition transaction | 1 | 0.10% |
| record | research.experiment_details.setup | 1 | 0.10% |
| record | research.objective | 1 | 0.10% |
| record | research.researcher.affiliation.abbreviation | 1 | 0.10% |
| record | research.experiment_details.temperature_details | 1 | 0.10% |
| record | risk_rating | 1 | 0.10% |
| record | study_details.design | 1 | 0.10% |
| record | pathophysiology.mechanism.primary_event | 1 | 0.10% |
| record | pathophysiology.mechanism.secondary_event.description | 1 | 0.10% |
| record | pathophysiology.mechanism.secondary_event.dislocated_structure | 1 | 0.10% |
| record | clinical_analysis_scope | 1 | 0.10% |
| record | co_investors | 1 | 0.10% |
| record | ram | 1 | 0.10% |
| record | holiday_look.collar.brand | 1 | 0.10% |
| record | holiday_look.heels.brand | 1 | 0.10% |
| record | comparison.similar_look | 1 | 0.10% |
| record | comparison.brand_with_similar_look | 1 | 0.10% |
| record | holiday_look.dress.designer | 1 | 0.10% |
| record | eps | 1 | 0.10% |
| record | net_margin | 1 | 0.10% |
| record | call_duration | 1 | 0.10% |
| record | singles | 1 | 0.10% |
| record | service_type | 1 | 0.10% |
| record | total_amount | 1 | 0.10% |
| record | marketing_budget | 1 | 0.10% |
| record | iporevisedvaluation | 1 | 0.10% |
| record | originalheadcountapex | 1 | 0.10% |
| record | preserved_specimens | 1 | 0.10% |
| record | paymentduedate | 1 | 0.10% |
| record | account_code | 1 | 0.10% |
| record | region: Geographic region where product is available | 1 | 0.10% |
| record | launch_date: Date the product was launched | 1 | 0.10% |
| record | product_id: Unique identifier for the LinkedIn product | 1 | 0.10% |
| record | target_audience: Primary user group for the product | 1 | 0.10% |
| record | keynote_time | 1 | 0.10% |
| record | force_type | 1 | 0.10% |
| record | material_efficiency | 1 | 0.10% |
| record | id: Unique identifier for each exchange rate record | 1 | 0.10% |
| record | bench_points: Total points scored by bench players | 1 | 0.10% |
| record | key_moment: Descriptive note about a pivotal play or event in the game | 1 | 0.10% |
| record | steals: Total steals recorded during the game | 1 | 0.10% |
| record | game_id: Unique identifier for the game as provided in the passage | 1 | 0.10% |
| record | home_score: Final score of the home team | 1 | 0.10% |
| record | pathway_name | 1 | 0.10% |
| record | splicing_regulation.current_research_status | 1 | 0.10% |
| record | score: Numerical performance score from 3.0 to 4.1 | 1 | 0.10% |
| record | discharge_destination | 1 | 0.10% |
| record | search_result_page.search_query | 1 | 0.10% |
| record | search_result_page.website_name | 1 | 0.10% |
| record | initial_text.negations | 1 | 0.10% |
| record | visit_count | 1 | 0.10% |
| record | compensation_type | 1 | 0.10% |
| record | product.specifications.display.resolution | 1 | 0.10% |
| record | product.launch_date | 1 | 0.10% |
| record | product.registration_count.count | 1 | 0.10% |
| record | product.specifications.memory.internal_storage | 1 | 0.10% |
| record | product.availability.currently_available | 1 | 0.10% |
| record | product.specifications.connectivity | 1 | 0.10% |
| record | product.price.1.currency | 1 | 0.10% |
| record | product.specifications.camera.features | 1 | 0.10% |
| record | product.specifications.display.size | 1 | 0.10% |
| record | product.launch_location | 1 | 0.10% |
| record | product.model | 1 | 0.10% |
| record | product.registration_count.time_period | 1 | 0.10% |
| record | product.specifications.camera.rear | 1 | 0.10% |
| record | product.specifications.battery.capacity | 1 | 0.10% |
| record | product.specifications.memory.ram | 1 | 0.10% |
| record | discovery.observatory.location.city | 1 | 0.10% |
| record | headliner | 1 | 0.10% |
| record | sender_name: Name of the sender in the correspondence | 1 | 0.10% |
| record | underwriter | 1 | 0.10% |
| record | description: Brief text describing the service or product delivered | 1 | 0.10% |
| record | invoice_number: Unique identifier for the invoice as assigned | 1 | 0.10% |
| record | status: Current financial status of the invoice | 1 | 0.10% |
| record | company_name: Name of the technology company raising funds | 1 | 0.10% |
| record | sector: Industry domain such as logistics, healthcare, fintech | 1 | 0.10% |
| record | investors: List of firms or individuals providing capital | 1 | 0.10% |
| record | date_added | 1 | 0.10% |
| record | tournament_code | 1 | 0.10% |
| record | loser | 1 | 0.10% |
| record | aces | 1 | 0.10% |
| record | break_points | 1 | 0.10% |
| record | fda_code | 1 | 0.10% |
| record | relationships.0.entity | 1 | 0.10% |
| record | relationships.0.action | 1 | 0.10% |
| record | relationships.0.type | 1 | 0.10% |
| record | amount: Quantity of crypto asset bought or sold | 1 | 0.10% |
| record | wallet_address: Unique wallet address used for the transaction | 1 | 0.10% |
| record | attendees | 1 | 0.10% |
| record | professions | 1 | 0.10% |
| record | user_id: Unique identifier for the user on TikTok | 1 | 0.10% |
| record | study_metadata.measurement_technology.device_name | 1 | 0.10% |
| record | follow_up.average_duration.months | 1 | 0.10% |
| record | outcomes.data_source | 1 | 0.10% |
| record | outcomes.fracture_events.0.type | 1 | 0.10% |
| record | statistical_results.hazard_ratios.0.adjustment_factors | 1 | 0.10% |
| record | statistical_results.hazard_ratios.0.value | 1 | 0.10% |
| record | statistical_results.hazard_ratios.0.comparison | 1 | 0.10% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.0.rate_percentage | 1 | 0.10% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.1.t_score | 1 | 0.10% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.1.rate_percentage | 1 | 0.10% |
| record | statistical_results.predictive_model.variables_included | 1 | 0.10% |
| record | election_district | 1 | 0.10% |
| record | service.target_audience | 1 | 0.10% |
| record | service.responsible_person.affiliation | 1 | 0.10% |
| record | service.opening_hours.0.time | 1 | 0.10% |
| record | service.responsible_person.profession | 1 | 0.10% |
| record | service.contact_information.phone | 1 | 0.10% |
| record | service.location.postal_code | 1 | 0.10% |
| record | blood_pressure: Blood pressure in mmHg format as systolic/diastolic | 1 | 0.10% |
| record | respiratory_rate: Breaths per minute respiratory rate | 1 | 0.10% |
| record | title: Name of the course such as Introduction to English Composition | 1 | 0.10% |
| record | location: Office location where the visit occurred | 1 | 0.10% |
| record | medication: Specific medication prescribed with dosage | 1 | 0.10% |
| record | visit_date: Date of the patient visit | 1 | 0.10% |
| record | lab_order: List of lab tests ordered for the patient | 1 | 0.10% |
| record | genre | 1 | 0.10% |
| record | status: Current state of the transaction processing | 1 | 0.10% |
| record | processing_location: Office or location where the transaction was processed | 1 | 0.10% |
| record | source: Origin of the exchange rate data feed | 1 | 0.10% |
| record | transaction_type: Type of transaction such as shipment, purchase, or invoice | 1 | 0.10% |
| record | species_name | 1 | 0.10% |
| record | family.name | 1 | 0.10% |
| record | described_by.year | 1 | 0.10% |
| record | name: Full name of the employee as listed in the review | 1 | 0.10% |
| record | promotional_campaign | 1 | 0.10% |
| record | product_list_id | 1 | 0.10% |
| record | discount_amount | 1 | 0.10% |
| record | discount_condition | 1 | 0.10% |
| record | promotion_channels | 1 | 0.10% |
| record | engagement_rate | 1 | 0.10% |
| record | performance_code | 1 | 0.10% |
| record | sales_growth | 1 | 0.10% |
| record | audience_age_range | 1 | 0.10% |
| record | product_count | 1 | 0.10% |
| record | literary_movements | 1 | 0.10% |
| record | recharge_amount | 1 | 0.10% |
| record | bonus_amount | 1 | 0.10% |
| record | leasetermlength: Duration of the lease in years | 1 | 0.10% |
| record | explanation | 1 | 0.10% |
| record | ability | 1 | 0.10% |
| record | hotel_details.star_rating | 1 | 0.10% |
| record | ranking.location | 1 | 0.10% |
| record | hotel_details.location | 1 | 0.10% |
| record | booking_options.partners | 1 | 0.10% |
| record | booking_options.description | 1 | 0.10% |
| record | hotel_details.name | 1 | 0.10% |
| record | serial_number: Unique serial number for device tracking | 1 | 0.10% |
| record | target_market | 1 | 0.10% |
| record | description: Brief explanation of the transaction purpose | 1 | 0.10% |
| record | inventors: List of inventor names involved in the filing | 1 | 0.10% |
| record | assignee: Entity owning the patent application | 1 | 0.10% |
| record | abstract: Brief summary of the invention's technical approach and purpose | 1 | 0.10% |
| record | classification_codes: List of classification codes assigned to the patent | 1 | 0.10% |
| record | filing_fee: Amount of filing fee paid in local currency | 1 | 0.10% |
| record | required_skills | 1 | 0.10% |
| record | character_name | 1 | 0.10% |
| record | actor_name | 1 | 0.10% |
| record | accuracy | 1 | 0.10% |
| record | shares: Total number of shares the video received | 1 | 0.10% |
| record | advertiser_information.contact_information.phone_number | 1 | 0.10% |
| record | public_information.brochure_name | 1 | 0.10% |
| record | distribution_information.trail_details.trail_name | 1 | 0.10% |
| record | public_information.download_share | 1 | 0.10% |
| record | advertiser_information.new_brochure_planned | 1 | 0.10% |
| record | distribution_information.distribution_locations | 1 | 0.10% |
| record | magazine.article.page_breakdown.6.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.14.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.3.advertising_percentage | 1 | 0.10% |
| record | magazine.issue | 1 | 0.10% |
| record | magazine.article.page_breakdown.11.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.7.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.1.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.12.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.4.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.0.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.10.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.6.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.11.page_number | 1 | 0.10% |
| record | downvotes | 1 | 0.10% |
| record | originalpostlink | 1 | 0.10% |
| record | content_length | 1 | 0.10% |
| record | comment_rank | 1 | 0.10% |
| record | price: Current selling price of the product in USD | 1 | 0.10% |
| record | allergies: List of known allergies with reactions | 1 | 0.10% |
| record | currentmedicationlist: List of active medications prescribed | 1 | 0.10% |
| record | comorbidities: List of additional health conditions present | 1 | 0.10% |
| record | notes: Additional clinical observations or patient-reported symptoms | 1 | 0.10% |
| record | current_status: Current clinical status of the patient's condition | 1 | 0.10% |
| record | diagnosis_date: Date when primary diagnosis was established | 1 | 0.10% |
| record | full_name: Complete name of the patient as recorded | 1 | 0.10% |
| record | insurance_provider: Name of the patient's health insurance provider | 1 | 0.10% |
| record | dateofbirth: Patient's birth date in YYYY-MM-DD format | 1 | 0.10% |
| record | shipping | 1 | 0.10% |
| record | stock_status | 1 | 0.10% |
| record | weight | 1 | 0.10% |
| record | study_duration | 1 | 0.10% |
| record | donor_address | 1 | 0.10% |
| record | race | 1 | 0.10% |
| record | age: Age of the athlete at the time of the event | 1 | 0.10% |
| record | year: Year the event was held | 1 | 0.10% |
| record | event_code: Unique identifier for the event as listed in records | 1 | 0.10% |
| record | passenger_phone | 1 | 0.10% |
| record | establishment_year | 1 | 0.10% |
| record | departments | 1 | 0.10% |
| record | staff.research_assistant_count | 1 | 0.10% |
| record | staff.teaching_staff_count | 1 | 0.10% |
| record | dean.name | 1 | 0.10% |
| record | laboratories | 1 | 0.10% |
| record | dean.title | 1 | 0.10% |
| record | naming.named_after.profession | 1 | 0.10% |
| record | naming.named_after.name | 1 | 0.10% |
| record | scorer | 1 | 0.10% |
| record | research.experiment_details.speckle_details | 1 | 0.10% |
| record | research.title | 1 | 0.10% |
| record | submission_id | 1 | 0.10% |
| record | geography.borders.2.direction | 1 | 0.10% |
| record | geography.borders.0.region | 1 | 0.10% |
| record | geography.borders.1.direction | 1 | 0.10% |
| record | infrastructure.refineries.0.refineryName | 1 | 0.10% |
| record | economy.gdpCurrency | 1 | 0.10% |
| record | infrastructure.highways.2.highwayName | 1 | 0.10% |
| record | infrastructure.roads.0.connections | 1 | 0.10% |
| record | infrastructure.airports.0.location | 1 | 0.10% |
| record | economy.economicBase | 1 | 0.10% |
| record | history.mergerDescription | 1 | 0.10% |
| record | infrastructure.airports.0.trafficVolume | 1 | 0.10% |
| record | infrastructure.airports.0.passengersYear | 1 | 0.10% |
| record | geography.rivers | 1 | 0.10% |
| record | infrastructure.highways.0.highwayName | 1 | 0.10% |
| record | economy.gdpRankAmongRegions | 1 | 0.10% |
| record | regionName | 1 | 0.10% |
| record | lowerLevelDivisions | 1 | 0.10% |
| record | infrastructure.refineries.0.closureDate | 1 | 0.10% |
| record | geography.borders.2.region | 1 | 0.10% |
| record | capital | 1 | 0.10% |
| record | infrastructure.refineries.0.status | 1 | 0.10% |
| record | regionNameBerber | 1 | 0.10% |
| record | history.establishmentDate | 1 | 0.10% |
| record | customer_retention | 1 | 0.10% |
| record | net_income | 1 | 0.10% |
| record | ebit_margin | 1 | 0.10% |
| record | medication_name | 1 | 0.10% |
| record | pharmacy | 1 | 0.10% |
| record | genus.naming.year | 1 | 0.10% |
| record | genus.naming.profession | 1 | 0.10% |
| record | genus.species.0.reason | 1 | 0.10% |
| record | genus.language | 1 | 0.10% |
| record | theme | 1 | 0.10% |
| record | comment_id | 1 | 0.10% |
| record | assessment_name | 1 | 0.10% |
| record | publication_code | 1 | 0.10% |
| record | game | 1 | 0.10% |
| record | vin | 1 | 0.10% |
| record | attendance: Percentage of days attended by participant | 1 | 0.10% |
| record | participant_id: Unique employee identifier from enrollment data | 1 | 0.10% |
| record | status: Current status of the enrollment | 1 | 0.10% |
| record | program_name: Title of the training program enrolled in | 1 | 0.10% |
| record | delivery_mode: Mode of training delivery | 1 | 0.10% |
| record | owner_name | 1 | 0.10% |
| record | term_years | 1 | 0.10% |
| record | dateofincident | 1 | 0.10% |
| record | video_title | 1 | 0.10% |
| record | guideline_id: Unique identifier for the guideline document | 1 | 0.10% |
| record | author: Name of the medical professional who authored the guideline | 1 | 0.10% |
| record | institution: Hospital or organization where author is affiliated | 1 | 0.10% |
| record | recommendation: Specific treatment or intervention advised | 1 | 0.10% |
| record | issue_year: Year the guideline was published or released | 1 | 0.10% |
| record | article.publication_details.source | 1 | 0.10% |
| record | article.publication_details.author | 1 | 0.10% |
| record | article.publication_details.publication_date | 1 | 0.10% |
| record | article.summary | 1 | 0.10% |
| record | article.entities.organizations | 1 | 0.10% |
| record | article.entities.people.0.organization | 1 | 0.10% |
| record | article.entities.people.0.title | 1 | 0.10% |
| record | article.entities.people.0.name | 1 | 0.10% |
| record | article.entities.people.1.organization | 1 | 0.10% |
| record | article.entities.people.1.title | 1 | 0.10% |
| record | article.entities.people.1.name | 1 | 0.10% |
| record | article.copyright_details.copyright_request_period | 1 | 0.10% |
| record | article.copyright_details.copyright_transparency_report_url | 1 | 0.10% |
| record | patient_name: Name of the affected individual as reported | 1 | 0.10% |
| record | patient_age: Age of the patient at time of reaction | 1 | 0.10% |
| record | patient_gender: Gender of the patient | 1 | 0.10% |
| record | patient_address: Full residential address of the patient | 1 | 0.10% |
| record | drug_name: Name of the medication involved in the reaction | 1 | 0.10% |
| record | drug_dosage: Dosage and frequency of the drug exposure | 1 | 0.10% |
| record | report_id: Unique identifier assigned to the adverse event report | 1 | 0.10% |
| record | carrier | 1 | 0.10% |
| record | manifest_id | 1 | 0.10% |
| record | visitor_team | 1 | 0.10% |
| record | distance.calculation_method | 1 | 0.10% |
| record | latin_name | 1 | 0.10% |
| record | characteristics.effective_temperature.value | 1 | 0.10% |
| record | distance.light_years | 1 | 0.10% |
| record | apparent_magnitude.unit | 1 | 0.10% |
| record | splicing_regulation.pathological_outcomes | 1 | 0.10% |
| record | siblings.0.name | 1 | 0.10% |
| record | siblings_in_law.0.relationship | 1 | 0.10% |
| record | employment.2.company | 1 | 0.10% |
| record | nieces | 1 | 0.10% |
| record | siblings_in_law.3.relationship | 1 | 0.10% |
| record | predeceased.2.relationship | 1 | 0.10% |
| record | predeceased.5.relationship | 1 | 0.10% |
| record | siblings_in_law.1.relationship | 1 | 0.10% |
| record | employment.1.company | 1 | 0.10% |
| record | siblings.0.location | 1 | 0.10% |
| record | nephews | 1 | 0.10% |
| record | grandchildren | 1 | 0.10% |
| record | siblings_in_law.3.name | 1 | 0.10% |
| record | siblings_in_law.1.name | 1 | 0.10% |
| record | predeceased.7.name | 1 | 0.10% |
| record | residence | 1 | 0.10% |
| record | employment.3.company | 1 | 0.10% |
| record | predeceased.3.relationship | 1 | 0.10% |
| record | siblings_in_law.1.location | 1 | 0.10% |
| record | siblings_in_law.0.name | 1 | 0.10% |
| record | parents.0.name | 1 | 0.10% |
| record | predeceased.3.name | 1 | 0.10% |
| record | employment.4.company | 1 | 0.10% |
| record | predeceased.4.name | 1 | 0.10% |
| record | children.1.spouse | 1 | 0.10% |
| record | marriage_duration | 1 | 0.10% |
| record | siblings_in_law.0.location | 1 | 0.10% |
| record | predeceased.0.name | 1 | 0.10% |
| record | predeceased.7.relationship | 1 | 0.10% |
| record | deployment_status | 1 | 0.10% |
| record | target_demographic | 1 | 0.10% |
| record | distribution_channels | 1 | 0.10% |
| record | customer_satisfaction | 1 | 0.10% |
| record | prescription_id: Unique identifier for the prescription entry | 1 | 0.10% |
| record | medication_name: Name of the prescribed medication | 1 | 0.10% |
| record | physical_exam | 1 | 0.10% |
| record | orbital_characteristics.database | 1 | 0.10% |
| record | asteroid_designation | 1 | 0.10% |
| record | provisional_designation | 1 | 0.10% |
| record | asteroid_type | 1 | 0.10% |
| record | physical_characteristics.observations.0.telescope_type | 1 | 0.10% |
| record | physical_characteristics.observations.0.year_launched | 1 | 0.10% |
| record | discovery.discoverers | 1 | 0.10% |
| record | consideration_structure | 1 | 0.10% |
| record | acquirer_ceo | 1 | 0.10% |
| record | weather_conditions | 1 | 0.10% |
| record | league | 1 | 0.10% |
| record | characteristics.spectral_class | 1 | 0.10% |
| record | reported_by | 1 | 0.10% |
| record | drug | 1 | 0.10% |
| record | item | 1 | 0.10% |
| record | company_size | 1 | 0.10% |
| record | place.location.country | 1 | 0.10% |
| record | procedure | 1 | 0.10% |
| record | rating: Star rating from 1 to 5 given by the buyer | 1 | 0.10% |
| record | tracking_number: Shipping tracking number assigned | 1 | 0.10% |
| record | order_id: Unique order identifier from the retailer | 1 | 0.10% |
| record | sponsor: Individual legislator who introduced the bill | 1 | 0.10% |
| record | vote_id: Unique identifier for the legislative vote | 1 | 0.10% |
| record | yes_votes: Number of members voting in favor | 1 | 0.10% |
| record | status: Current legislative status of the bill | 1 | 0.10% |
| record | scope: Broad category or focus area of the bill | 1 | 0.10% |
| record | additional_services | 1 | 0.10% |
| record | experience.years | 1 | 0.10% |
| record | experience.description | 1 | 0.10% |
| record | vaccine | 1 | 0.10% |
| record | dose | 1 | 0.10% |
| record | facility: Facility name and address where device is located | 1 | 0.10% |
| record | device_id: Unique identifier for the medical device | 1 | 0.10% |
| record | publication_time | 1 | 0.10% |
| record | position | 1 | 0.10% |
| record | comment_content: Text of the comment including personal anecdotes and tool usage | 1 | 0.10% |
| record | author: Reddit username of the comment author | 1 | 0.10% |
| record | downvotes: Number of downvotes the comment received | 1 | 0.10% |
| record | tool: Productivity tool recommended in the comment | 1 | 0.10% |
| record | donation_date | 1 | 0.10% |
| record | reviewer_name: Full name of the reviewer | 1 | 0.10% |
| record | key_comment: Specific positive or negative observation about the stay | 1 | 0.10% |
| record | location: City and state where the hotel is located | 1 | 0.10% |
| record | summary | 1 | 0.10% |
| record | submission_date | 1 | 0.10% |
| record | research_submission | 1 | 0.10% |
| record | customer_name: Name of the customer initiating the return | 1 | 0.10% |
| record | critic_name | 1 | 0.10% |
| record | review_source | 1 | 0.10% |
| record | critic_location | 1 | 0.10% |
| record | physical_characteristics.observations.0.telescope.origin | 1 | 0.10% |
| record | asteroid_belt_classification | 1 | 0.10% |
| record | reviewer: Name of the person who wrote the review | 1 | 0.10% |
| record | location: City where the reviewer is based | 1 | 0.10% |
| record | film_title: Title of the film being reviewed | 1 | 0.10% |
| record | rating: Numerical score given to the film | 1 | 0.10% |
| record | review_text: Full textual critique of the film | 1 | 0.10% |
| record | asteroid.dedication.person.name | 1 | 0.10% |
| record | description: Detailed account of the issue reported | 1 | 0.10% |
| record | submitter: Name of the employee who submitted the ticket | 1 | 0.10% |
| record | affected_system: Software or hardware system experiencing the problem | 1 | 0.10% |
| record | priority: Urgency level of the ticket | 1 | 0.10% |
| record | status: Current state of the ticket | 1 | 0.10% |
| record | ppe_required | 1 | 0.10% |
| record | hazard_class | 1 | 0.10% |
| record | handling_notes | 1 | 0.10% |
| record | physician.name | 1 | 0.10% |
| record | physician.degree | 1 | 0.10% |
| record | physician.address.street | 1 | 0.10% |
| record | physician.address.city | 1 | 0.10% |
| record | physician.address.state | 1 | 0.10% |
| record | physician.address.zip | 1 | 0.10% |
| record | physician.education.fellowship.0.institution | 1 | 0.10% |
| record | physician.education.residency.0.institution | 1 | 0.10% |
| record | physician.education.medical_school.0.institution | 1 | 0.10% |
| record | item_description.shipping.ready_to_ship_in.quantity | 1 | 0.10% |
| record | item_description.shipping.rush_details | 1 | 0.10% |
| record | item_description.weight.unit | 1 | 0.10% |
| record | item_description.dimensions.length | 1 | 0.10% |
| record | item_description.setup_charge.per | 1 | 0.10% |
| record | item_description.setup_charge.amount | 1 | 0.10% |
| record | item_description.weight.value | 1 | 0.10% |
| record | establishment_approval | 1 | 0.10% |
| record | abolished_entity | 1 | 0.10% |
| record | administrative_level | 1 | 0.10% |
| record | user_email | 1 | 0.10% |
| record | user_name | 1 | 0.10% |
| record | ties | 1 | 0.10% |
| record | final_standing | 1 | 0.10% |
| record | games_played | 1 | 0.10% |
| record | coach | 1 | 0.10% |
| record | games_behind | 1 | 0.10% |
| record | season | 1 | 0.10% |
| record | region_churn | 1 | 0.10% |
| record | call_id | 1 | 0.10% |
| record | transcription_factor | 1 | 0.10% |
| record | gene_id | 1 | 0.10% |
| record | tissue | 1 | 0.10% |
| record | network_id | 1 | 0.10% |
| record | expression_units | 1 | 0.10% |
| record | network_context | 1 | 0.10% |
| record | name: Product name as listed in inventory records | 1 | 0.10% |
| record | last_updated: Date when inventory was last modified | 1 | 0.10% |
| record | sku_id: Unique identifier for the product in inventory system | 1 | 0.10% |
| record | quantity: Current stock level of the SKU | 1 | 0.10% |
| record | location: Warehouse or hub where the SKU is stored | 1 | 0.10% |
| record | location: Headquarters city and country of the company | 1 | 0.10% |
| record | lead_investor: Primary investor or consortium leading the round | 1 | 0.10% |
| record | company_name: Name of the startup founded by the founder | 1 | 0.10% |
| record | equity_stake: Founder's ownership percentage in the company | 1 | 0.10% |
| record | location: City and state where the startup is based | 1 | 0.10% |
| record | founding_year: Year the startup was established | 1 | 0.10% |
| record | study_period | 1 | 0.10% |
| record | trial_phase | 1 | 0.10% |
| record | title: Event title describing the nature and context of the incident | 1 | 0.10% |
| record | date: Timestamp when the incident occurred | 1 | 0.10% |
| record | involved_parties: List of organizations or individuals involved in the incident | 1 | 0.10% |
| record | source: Media or organization reporting the incident | 1 | 0.10% |
| record | lead_reporter: Reporter providing coverage of the incident | 1 | 0.10% |
| record | geographic_code: Unique code identifying the incident's geographic region | 1 | 0.10% |
| record | policyholder_name: Name of the policyholder filing the claim | 1 | 0.10% |
| record | address: Full address of the policyholder | 1 | 0.10% |
| record | claim_date: Date when the incident occurred | 1 | 0.10% |
| record | amount_requested: Total amount requested for compensation | 1 | 0.10% |
| record | status: Current status of the claim | 1 | 0.10% |
| record | description: Brief description of the incident and damages | 1 | 0.10% |
| record | date: Date the match was scheduled or played | 1 | 0.10% |
| record | score: List of two integers representing home and away scores | 1 | 0.10% |
| record | team1: First team participating in the match | 1 | 0.10% |
| record | location: Venue where the match took place | 1 | 0.10% |
| record | taxonomy.describer.affiliation.0.laboratory | 1 | 0.10% |
| record | description.distinguishing_features | 1 | 0.10% |
| record | species.etymology.named_after.0.location.city | 1 | 0.10% |
| record | distribution.habitat | 1 | 0.10% |
| record | taxonomy.describer.affiliation.0.university | 1 | 0.10% |
| record | taxonomy.describer.affiliation.0.location.country | 1 | 0.10% |
| record | description.size | 1 | 0.10% |
| record | description.color | 1 | 0.10% |
| record | description.width | 1 | 0.10% |
| record | taxonomy.subgenus | 1 | 0.10% |
| record | species.etymology.named_after.0.location.country | 1 | 0.10% |
| record | taxonomy.describer.affiliation.0.location.city | 1 | 0.10% |
| record | taxonomy.describer.name | 1 | 0.10% |
| record | taxonomy.year_described | 1 | 0.10% |
| record | taxonomy.type_materials | 1 | 0.10% |
| record | review_code | 1 | 0.10% |
| record | audit_id | 1 | 0.10% |
| record | funding_type: Description of funding commitment such as operational or adaptation | 1 | 0.10% |
| record | affected_people | 1 | 0.10% |
| record | available_units | 1 | 0.10% |
| record | colors | 1 | 0.10% |
| record | performance_reason | 1 | 0.10% |
| record | physical_characteristics.observations.0.telescope.country | 1 | 0.10% |
| record | discovery.observatory.location.state | 1 | 0.10% |
| record | discovery.project | 1 | 0.10% |
| record | founder: Name of the individual who founded the company | 1 | 0.10% |
| record | funding_source: Type of funding received by the company | 1 | 0.10% |
| record | current_status: Current operational phase of the company | 1 | 0.10% |
| record | clientbasesize: Number of clients served across regions | 1 | 0.10% |
| record | business_focus: Core service or product area of the company | 1 | 0.10% |
| record | identifier | 1 | 0.10% |
| record | staffing_change | 1 | 0.10% |
| record | funding_grant | 1 | 0.10% |
| record | outcome_measure | 1 | 0.10% |
| record | intervention | 1 | 0.10% |
| record | capacity | 1 | 0.10% |
| record | return_date: Date when the return was submitted | 1 | 0.10% |
| record | award_name | 1 | 0.10% |
| record | fieldofstudy | 1 | 0.10% |
| record | patient_id: Unique patient identifier from the medical record | 1 | 0.10% |
| record | modality: Type of imaging modality used | 1 | 0.10% |
| record | findings: Key clinical observations or abnormalities detected | 1 | 0.10% |
| record | report_id: Unique identifier for the imaging report | 1 | 0.10% |
| record | status: Current status of the report workflow | 1 | 0.10% |
| record | availability_date | 1 | 0.10% |
| record | availability_status | 1 | 0.10% |
| record | asteroid.named_after.person.name_native | 1 | 0.10% |
| record | asteroid.named_after.person.death_year | 1 | 0.10% |
| record | genus.family.name | 1 | 0.10% |
| record | genus.location | 1 | 0.10% |
| record | genus.species.0.name | 1 | 0.10% |
| record | genus.description | 1 | 0.10% |
| record | genus.species.0.year | 1 | 0.10% |
| record | availability | 1 | 0.10% |
| record | stock_level | 1 | 0.10% |
| record | listing_date | 1 | 0.10% |
| record | study.period.start_year | 1 | 0.10% |
| record | study.focus | 1 | 0.10% |
| record | study.comparison_countries | 1 | 0.10% |
| record | symbol: Stock ticker symbol as listed in the data | 1 | 0.10% |
| record | change_percent: Percentage change in stock price for the day | 1 | 0.10% |
| record | close_price: Final closing price of the stock | 1 | 0.10% |
| record | volume: Total shares traded on the day | 1 | 0.10% |
| record | market_cap: Market capitalization in billions | 1 | 0.10% |
| record | movement_reason: Brief reason for price movement or performance | 1 | 0.10% |
| record | support_ticket | 1 | 0.10% |
| record | impact_percentage | 1 | 0.10% |
| record | materials | 1 | 0.10% |
| record | door | 1 | 0.10% |
| record | metaphors | 1 | 0.10% |
| record | objects | 1 | 0.10% |
| record | supernatural | 1 | 0.10% |
| record | lead_investor: List of lead investors or funds involved | 1 | 0.10% |
| record | post_valuation: Post-money valuation in millions | 1 | 0.10% |
| record | focus: Primary technology or business area of the company | 1 | 0.10% |
| record | amount: Total capital raised in millions | 1 | 0.10% |
| record | reaction: Observed adverse effect or symptom | 1 | 0.10% |
| record | drug: Name and dosage form of the medication taken | 1 | 0.10% |
| record | dedication.description | 1 | 0.10% |
| record | currency: Currency pair such as JPY, EUR, CAD | 1 | 0.10% |
| record | chamber | 1 | 0.10% |
| record | opposing_members | 1 | 0.10% |
| record | major | 1 | 0.10% |
| record | followupinterval | 1 | 0.10% |
| record | domain.registry.name | 1 | 0.10% |
| record | domain.requirements.minimum_length | 1 | 0.10% |
| record | domain.third_level_domains | 1 | 0.10% |
| record | domain.territory | 1 | 0.10% |
| record | budget | 1 | 0.10% |
| record | agency | 1 | 0.10% |
| record | technology | 1 | 0.10% |
| record | announcement.divisions.0.star_of_the_week.player_name | 1 | 0.10% |
| record | announcement.presented_by | 1 | 0.10% |
| record | announcement.divisions.0.division_name | 1 | 0.10% |
| record | announcement.divisions.5.honorable_mentions.1.team | 1 | 0.10% |
| record | announcement.divisions.0.star_of_the_week.team | 1 | 0.10% |
| record | announcement.divisions.2.honorable_mentions.0.player_name | 1 | 0.10% |
| record | announcement.divisions.5.star_of_the_week.position | 1 | 0.10% |
| record | announcement.divisions.0.second_star.team | 1 | 0.10% |
| record | announcement.divisions.1.star_of_the_week.team | 1 | 0.10% |
| record | announcement.divisions.0.second_star.player_name | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.3.position | 1 | 0.10% |
| record | announcement.divisions.1.honorable_mentions.0.team | 1 | 0.10% |
| record | announcement.divisions.5.honorable_mentions.2.position | 1 | 0.10% |
| record | announcement.divisions.5.second_star.team | 1 | 0.10% |
| record | announcement.divisions.5.honorable_mentions.0.team | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.7.player_name | 1 | 0.10% |
| record | announcement.divisions.2.honorable_mentions.4.position | 1 | 0.10% |
| record | announcement.divisions.3.second_star.position | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.0.player_name | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.4.position | 1 | 0.10% |
| record | announcement.divisions.1.star_of_the_week.stats | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.5.team | 1 | 0.10% |
| record | announcement.divisions.2.division_name | 1 | 0.10% |
| record | announcement.divisions.4.honorable_mentions.2.team | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.7.position | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.1.player_name | 1 | 0.10% |
| record | announcement.divisions.3.honorable_mentions.1.player_name | 1 | 0.10% |
| record | announcement.divisions.4.honorable_mentions.0.player_name | 1 | 0.10% |
| record | announcement.divisions.4.honorable_mentions.1.player_name | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.1.team | 1 | 0.10% |
| record | announcement.divisions.1.honorable_mentions.0.position | 1 | 0.10% |
| record | announcement.divisions.3.star_of_the_week.position | 1 | 0.10% |
| record | announcement.divisions.5.star_of_the_week.player_name | 1 | 0.10% |
| record | announcement.divisions.2.honorable_mentions.1.position | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.2.player_name | 1 | 0.10% |
| record | announcement.divisions.1.star_of_the_week.position | 1 | 0.10% |
| record | announcement.divisions.3.honorable_mentions.0.player_name | 1 | 0.10% |
| record | announcement.divisions.0.second_star.position | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.0.position | 1 | 0.10% |
| record | announcement.divisions.1.star_of_the_week.player_name | 1 | 0.10% |
| record | announcement.divisions.3.division_name | 1 | 0.10% |
| record | announcement.divisions.5.honorable_mentions.1.position | 1 | 0.10% |
| record | announcement.divisions.4.second_star.team | 1 | 0.10% |
| record | announcement.divisions.2.honorable_mentions.0.team | 1 | 0.10% |
| record | announcement.divisions.3.second_star.team | 1 | 0.10% |
| record | announcement.divisions.4.division_name | 1 | 0.10% |
| record | announcement.divisions.1.honorable_mentions.1.team | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.4.player_name | 1 | 0.10% |
| record | announcement.divisions.2.honorable_mentions.0.position | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.7.team | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.6.player_name | 1 | 0.10% |
| record | announcement.divisions.1.second_star.player_name | 1 | 0.10% |
| record | announcement.divisions.2.honorable_mentions.2.position | 1 | 0.10% |
| record | announcement.week_ending | 1 | 0.10% |
| record | announcement.divisions.2.second_star.position | 1 | 0.10% |
| record | announcement.divisions.5.star_of_the_week.team | 1 | 0.10% |
| record | announcement.divisions.2.honorable_mentions.4.player_name | 1 | 0.10% |
| record | announcement.divisions.1.second_star.position | 1 | 0.10% |
| record | announcement.divisions.2.second_star.player_name | 1 | 0.10% |
| record | announcement.divisions.1.honorable_mentions.0.player_name | 1 | 0.10% |
| record | announcement.divisions.4.honorable_mentions.1.position | 1 | 0.10% |
| record | announcement.divisions.5.division_name | 1 | 0.10% |
| record | announcement.divisions.3.star_of_the_week.team | 1 | 0.10% |
| record | announcement.divisions.2.honorable_mentions.1.team | 1 | 0.10% |
| record | announcement.divisions.4.honorable_mentions.2.position | 1 | 0.10% |
| record | snapshot_id | 1 | 0.10% |
| record | source_currency | 1 | 0.10% |
| record | target_currency | 1 | 0.10% |
| record | symbol: Unique ticker identifier for the company | 1 | 0.10% |
| record | price: Closing stock price in dollars | 1 | 0.10% |
| record | sector: Industry sector classification | 1 | 0.10% |
| record | status: Direction of price movement | 1 | 0.10% |
| record | baths | 1 | 0.10% |
| record | beds | 1 | 0.10% |
| record | document_status | 1 | 0.10% |
| record | stock_symbol | 1 | 0.10% |
| record | passenger_city | 1 | 0.10% |
| record | email_id | 1 | 0.10% |
| record | match_type | 1 | 0.10% |
| record | office_hours.4.day | 1 | 0.10% |
| record | contact.fax | 1 | 0.10% |
| record | office_hours.4.end_time | 1 | 0.10% |
| record | address.postal_code | 1 | 0.10% |
| record | address.city | 1 | 0.10% |
| record | finding_id | 1 | 0.10% |
| record | access_duration | 1 | 0.10% |
| record | access_type | 1 | 0.10% |
| record | data_code | 1 | 0.10% |
| record | RP | 1 | 0.10% |
| record | VBD | 1 | 0.10% |
| record | episode_number | 1 | 0.10% |
| record | patient_cohort | 1 | 0.10% |
| record | status: Current legal status of the patent | 1 | 0.10% |
| record | country: Country code from patent number prefix | 1 | 0.10% |
| record | patient_province | 1 | 0.10% |
| record | reporting_doctor | 1 | 0.10% |
| record | bedrooms: Number of bedrooms in the property | 1 | 0.10% |
| record | bathrooms: Number of bathrooms in the property | 1 | 0.10% |
| record | address: Full street address of the property | 1 | 0.10% |
| record | price: Listed sale price in dollars | 1 | 0.10% |
| record | lot_size: Size of the property in acres | 1 | 0.10% |
| record | build_year: Year the property was constructed | 1 | 0.10% |
| record | agent: Name of the real estate agent representing the seller | 1 | 0.10% |
| record | responsibility | 1 | 0.10% |
| record | crisis_resilience | 1 | 0.10% |
| record | benefits | 1 | 0.10% |
| record | business_sustainability.elements | 1 | 0.10% |
| record | industry_example.actions | 1 | 0.10% |
| record | industry_example.challenges | 1 | 0.10% |
| record | industry_example.status | 1 | 0.10% |
| record | business_sustainability.product_lifecycle_analysis | 1 | 0.10% |
| record | three_pillars | 1 | 0.10% |
| record | industry_example.industry | 1 | 0.10% |
| record | aspects | 1 | 0.10% |
| record | financial_aspect.relevance_in_crisis | 1 | 0.10% |
| record | personal_sustainability | 1 | 0.10% |
| record | sale_amount | 1 | 0.10% |
| record | sales_period | 1 | 0.10% |
| record | company.founding_location.city | 1 | 0.10% |
| record | company.location | 1 | 0.10% |
| record | company.transformation_date | 1 | 0.10% |
| record | company.description | 1 | 0.10% |
| record | company.establishment_date | 1 | 0.10% |
| record | open_price | 1 | 0.10% |
| record | external_links.0.title | 1 | 0.10% |
| record | classification.source | 1 | 0.10% |
| record | observations.observation_period.years | 1 | 0.10% |
| record | wiki_sky_links | 1 | 0.10% |
| record | alternative_designations | 1 | 0.10% |
| record | orbital_characteristics.perihelion | 1 | 0.10% |
| record | discovery.location.observatory | 1 | 0.10% |
| record | observations.last_observation_date | 1 | 0.10% |
| record | classification.type | 1 | 0.10% |
| record | orbital_characteristics.orbital_period.years | 1 | 0.10% |
| record | discovery.location.place | 1 | 0.10% |
| record | trademark_name: Name of the registered trademark in dispute | 1 | 0.10% |
| record | registrant: Company that owns the registered trademark | 1 | 0.10% |
| record | registration_number: Unique number assigned to the trademark registration | 1 | 0.10% |
| record | opponent: Entity filing the dispute or opposition | 1 | 0.10% |
| record | dispute_date: Date the dispute or opposition was filed | 1 | 0.10% |
| record | status: Current legal status of the dispute | 1 | 0.10% |
| record | dispute_type: Type of legal action such as cancellation or infringement | 1 | 0.10% |
| record | product_offered | 1 | 0.10% |
| record | date_sold | 1 | 0.10% |
| record | submission_method | 1 | 0.10% |
| record | filing_amount | 1 | 0.10% |
| record | person.birth_location.country | 1 | 0.10% |
| record | person.birth_location.country_extended | 1 | 0.10% |
| record | person.death_date | 1 | 0.10% |
| record | person.death_location.region | 1 | 0.10% |
| record | technology_focus: Key technology or application area of the patent | 1 | 0.10% |
| record | inventors: List of individuals who filed the patent | 1 | 0.10% |
| record | claims: Number of claims covered by the patent | 1 | 0.10% |
| record | status: Current status of the patent | 1 | 0.10% |
| record | punct | 1 | 0.10% |
| record | dobj | 1 | 0.10% |
| record | nmod | 1 | 0.10% |
| record | cc | 1 | 0.10% |
| record | preconj | 1 | 0.10% |
| record | nsubj | 1 | 0.10% |
| record | case | 1 | 0.10% |
| record | ROOT | 1 | 0.10% |
| record | advmod | 1 | 0.10% |
| record | npadvmod | 1 | 0.10% |
| record | prep | 1 | 0.10% |
| record | intj | 1 | 0.10% |
| record | nummod | 1 | 0.10% |
| record | status: Current enrollment or completion status | 1 | 0.10% |
| record | start_date: Date when the course begins | 1 | 0.10% |
| record | course_id: Unique identifier for the course as given in the XML | 1 | 0.10% |
| record | charge_amount | 1 | 0.10% |
| record | original_plan | 1 | 0.10% |
| record | training_modules: List of specific training topics assigned to the employee | 1 | 0.10% |
| record | name: Full name of the new employee | 1 | 0.10% |
| record | employee_id: Unique identifier assigned to each new hire | 1 | 0.10% |
| record | enrollment_status | 1 | 0.10% |
| record | country: Nationality of the athlete who won the medal | 1 | 0.10% |
| record | venue: Location where the event took place | 1 | 0.10% |
| record | result: Medal awarded to the athlete | 1 | 0.10% |
| record | athlete_name: Name of the athlete who achieved the result | 1 | 0.10% |
| record | trial_title | 1 | 0.10% |
| record | isotope | 1 | 0.10% |
| record | symbol | 1 | 0.10% |
| record | volume | 1 | 0.10% |
| record | asteroid_belt | 1 | 0.10% |
| record | discovery_details.observatory.name | 1 | 0.10% |
| record | discovery_details.observatory.location.county | 1 | 0.10% |
| record | discovery_details.observatory.location.state | 1 | 0.10% |
| record | discovery_details.observatory.location.country | 1 | 0.10% |
| record | discovery_details.discoverer | 1 | 0.10% |
| record | orbital_characteristics.jpl_classification | 1 | 0.10% |
| record | physical_characteristics.observations.0.publication_details | 1 | 0.10% |
| record | physical_characteristics.observations.0.telescope.type | 1 | 0.10% |
| record | physical_characteristics.observations.0.telescope.location | 1 | 0.10% |
| record | physical_characteristics.observations.0.telescope.year_launched | 1 | 0.10% |
| record | invoice_date | 1 | 0.10% |
| record | address: Patient's residential address including street, unit, and city | 1 | 0.10% |
| record | service.problems_addressed | 1 | 0.10% |
| record | service.location.city | 1 | 0.10% |
| record | service.responsible_person.experience | 1 | 0.10% |
| record | booking_options.assurance_message | 1 | 0.10% |
| record | post_date | 1 | 0.10% |
| record | user_handle | 1 | 0.10% |
| record | section | 1 | 0.10% |
| record | founder | 1 | 0.10% |
| record | founded_year | 1 | 0.10% |
| record | funding: Total funding amount in millions | 1 | 0.10% |
| record | project_name: Name of the technology project | 1 | 0.10% |
| record | location: Geographic location of the project | 1 | 0.10% |
| record | project_id: Unique identifier assigned to the project | 1 | 0.10% |
| record | album | 1 | 0.10% |
| record | record_label | 1 | 0.10% |
| record | articleTitle | 1 | 0.10% |
| record | holiday | 1 | 0.10% |
| record | filing_id | 1 | 0.10% |
| record | submission_id: Unique identifier for the paper submission | 1 | 0.10% |
| record | author_name: Name of the paper's author | 1 | 0.10% |
| record | affiliation: Institutional affiliation of the author | 1 | 0.10% |
| record | paper_title: Title of the submitted research paper | 1 | 0.10% |
| record | submission_date: Date when the paper was submitted | 1 | 0.10% |
| record | word_count: Total number of words in the paper | 1 | 0.10% |
| record | fieldofstudy: Academic domain of the research | 1 | 0.10% |
| record | conference_name: Name of the conference where submitted | 1 | 0.10% |
| record | review_date: Date when the paper was reviewed | 1 | 0.10% |
| record | funding_source: Source of research funding | 1 | 0.10% |
| record | page_count: Number of pages in the paper | 1 | 0.10% |
| record | abstract_length: Length of the abstract in words | 1 | 0.10% |
| record | electoral_code | 1 | 0.10% |
| record | price.installment_option.installment_amount | 1 | 0.10% |
| record | care_instructions | 1 | 0.10% |
| record | size_and_fit.measurements.1.type | 1 | 0.10% |
| record | size_and_fit.length.size | 1 | 0.10% |
| record | price.installment_option.installments | 1 | 0.10% |
| record | start_term | 1 | 0.10% |
| record | hall | 1 | 0.10% |
| record | course_code | 1 | 0.10% |
| record | article_issues | 1 | 0.10% |
| record | claimant_name | 1 | 0.10% |
| record | claim_date | 1 | 0.10% |
| record | hotel_id | 1 | 0.10% |
| record | review_type | 1 | 0.10% |
| record | review_score | 1 | 0.10% |
| record | github_link | 1 | 0.10% |
| record | municipality.name_local | 1 | 0.10% |
| record | municipality.location.region | 1 | 0.10% |
| record | municipality.location.country | 1 | 0.10% |
| record | demographics.population.0.count | 1 | 0.10% |
| record | demographics.population.0.year | 1 | 0.10% |
| record | product_code | 1 | 0.10% |
| record | pricing | 1 | 0.10% |
| record | contribution_id: Unique identifier for each contribution entry | 1 | 0.10% |
| record | address: Physical address of the donor as recorded | 1 | 0.10% |
| record | status: Current status of the contribution record | 1 | 0.10% |
| record | campaign_target: Name of the candidate or office being supported | 1 | 0.10% |
| record | notes: Additional context about the appointment or background | 1 | 0.10% |
| record | compensation: Annual salary or compensation amount | 1 | 0.10% |
| record | position: Job title or role within the department | 1 | 0.10% |
| record | date: Start date of the appointment | 1 | 0.10% |
| record | insurance_id | 1 | 0.10% |
| record | amount_paid | 1 | 0.10% |
| record | inventor | 1 | 0.10% |
| record | technology_area | 1 | 0.10% |
| record | author.role | 1 | 0.10% |
| record | version | 1 | 0.10% |
| record | software | 1 | 0.10% |
| record | concept | 1 | 0.10% |
| record | definitions | 1 | 0.10% |
| record | journals | 1 | 0.10% |
| record | movements | 1 | 0.10% |
| record | related_terms | 1 | 0.10% |
| record | event_id: Conference or symposium identifier for the accepted abstract | 1 | 0.10% |
| record | recipient: Name of the person receiving the confirmation | 1 | 0.10% |
| record | submission_id: Unique identifier assigned to the abstract submission | 1 | 0.10% |
| record | sender: Name of the person sending the confirmation | 1 | 0.10% |
| record | regulatory_status.recognition_type | 1 | 0.10% |
| record | announcement_metadata.source_url | 1 | 0.10% |
| record | announcement_metadata.coverage_provider | 1 | 0.10% |
| record | regulatory_status.review_status | 1 | 0.10% |
| record | announcement_metadata.date | 1 | 0.10% |
| record | announcement_metadata.agency | 1 | 0.10% |
| record | report_id: Unique identifier assigned to the phishing report | 1 | 0.10% |
| record | dateofbirth | 1 | 0.10% |
| record | location: Office or room where issue occurred | 1 | 0.10% |
| record | issue: Description of the technical problem reported | 1 | 0.10% |
| record | size_and_fit.model_size | 1 | 0.10% |
| record | size_and_fit.size_guide | 1 | 0.10% |
| record | size_and_fit.model_height.value | 1 | 0.10% |
| record | size_and_fit.length.value | 1 | 0.10% |
| record | transaction_type | 1 | 0.10% |
| record | settlement_time | 1 | 0.10% |
| record | company_name: Name of the startup founded by the individual | 1 | 0.10% |
| record | founder_name: Full name of the founder as stated in passage | 1 | 0.10% |
| record | motivation: Personal reason driving the company's founding | 1 | 0.10% |
| record | focus_area: Primary domain or mission of the company | 1 | 0.10% |
| record | location: City and state where the company was founded | 1 | 0.10% |
| record | actor | 1 | 0.10% |
| record | technique_name | 1 | 0.10% |
| record | abbreviation | 1 | 0.10% |
| record | cohort_size | 1 | 0.10% |
| record | specific_outcomes.postoperative_cholangiography_rate | 1 | 0.10% |
| record | trader | 1 | 0.10% |
| record | expert_insights | 1 | 0.10% |
| record | priority_registration.submission_methods | 1 | 0.10% |
| record | priority_registration.documents_required | 1 | 0.10% |
| record | on_site_registration.documents_required | 1 | 0.10% |
| record | important_notice | 1 | 0.10% |
| record | on_site_registration.description | 1 | 0.10% |
| record | priority_registration.instructions | 1 | 0.10% |
| record | on_site_registration.note | 1 | 0.10% |
| record | on_site_registration.location.address | 1 | 0.10% |
| record | online_priority_registration.deadline | 1 | 0.10% |
| record | related_conditions | 1 | 0.10% |
| record | recipient: Name of the student receiving the award | 1 | 0.10% |
| record | address: Full street address of the recipient | 1 | 0.10% |
| record | city: City where the recipient resides | 1 | 0.10% |
| record | state: State or province where the recipient resides | 1 | 0.10% |
| record | country: Country where the recipient resides | 1 | 0.10% |
| record | award_name: Title of the scholarship or grant awarded | 1 | 0.10% |
| record | award_code: Unique identifier code for the award | 1 | 0.10% |
| record | status: Current status of the award | 1 | 0.10% |
| record | fieldofstudy: Academic discipline or program the recipient is in | 1 | 0.10% |
| record | topics | 1 | 0.10% |
| record | contact_number | 1 | 0.10% |
| record | metabolite | 1 | 0.10% |
| record | themes | 1 | 0.10% |
| record | setting.political_context | 1 | 0.10% |
| record | main_character.affiliation | 1 | 0.10% |
| record | main_character.name | 1 | 0.10% |
| record | main_character.appears_in_other_work | 1 | 0.10% |
| record | publication_details.first_publication.location | 1 | 0.10% |
| record | publication_details.first_publication_czechoslovakia | 1 | 0.10% |
| record | polish_titles | 1 | 0.10% |
| record | setting.time_period | 1 | 0.10% |
| record | main_character.rank | 1 | 0.10% |
| record | asteroid.alternative_designations | 1 | 0.10% |
| record | asteroid.discovery.discovery_rank | 1 | 0.10% |
| record | asteroid.description.discovery_details.date | 1 | 0.10% |
| record | asteroid.description.discovery_details.observatory | 1 | 0.10% |
| record | asteroid.description.discovery_details.location.country | 1 | 0.10% |
| record | asteroid.description.discovery_details.location.state | 1 | 0.10% |
| record | asteroid.description.discovery_details.location.county | 1 | 0.10% |
| record | asteroid.description.discovery_details.project | 1 | 0.10% |
| record | asteroid.description.observations.last_observation_date | 1 | 0.10% |
| record | asteroid.description.observations.observation_period_years | 1 | 0.10% |
| record | asteroid.description.orbital_period_years | 1 | 0.10% |
| record | asteroid.orbital_characteristics.semi_major_axis | 1 | 0.10% |
| record | asteroid.orbital_characteristics.perihelion | 1 | 0.10% |
| record | asteroid.orbital_characteristics.eccentricity | 1 | 0.10% |
| record | asteroid.orbital_characteristics.inclination | 1 | 0.10% |
| record | asteroid.orbital_characteristics.classification | 1 | 0.10% |
| record | asteroid.external_links.0.title | 1 | 0.10% |
| record | asteroid.wiki_sky_links | 1 | 0.10% |
| record | store_name | 1 | 0.10% |
| record | seller_id | 1 | 0.10% |
| record | asteroid.named_after.location.country | 1 | 0.10% |
| record | asteroid.named_after.location.province | 1 | 0.10% |
| record | discovery.astronomer.nationality | 1 | 0.10% |
| record | requisition_id | 1 | 0.10% |
| record | startup_name | 1 | 0.10% |
| record | new_website_url | 1 | 0.10% |
| record | issuer | 1 | 0.10% |
| record | storm_characteristics_at_stage | 1 | 0.10% |
| record | direction_of_travel | 1 | 0.10% |
| record | vaccination_id | 1 | 0.10% |
| record | hotel_experience.location.nearby_landmark | 1 | 0.10% |
| record | hotel_experience.entertainment.tv.content | 1 | 0.10% |
| record | hotel_experience.suites.0.workspace | 1 | 0.10% |
| record | hotel_experience.suites.0.bathroom_features | 1 | 0.10% |
| record | hotel_experience.suites.0.amenities | 1 | 0.10% |
| record | asteroid.discovery.year | 1 | 0.10% |
| record | location-gpe | 1 | 0.10% |
| record | url | 1 | 0.10% |
| record | agreement_id | 1 | 0.10% |
| record | promotional_code | 1 | 0.10% |
| record | expected_close_date | 1 | 0.10% |
| record | geographic_region | 1 | 0.10% |
| record | creation.event | 1 | 0.10% |
| record | hanja_name | 1 | 0.10% |
| record | creation.year | 1 | 0.10% |
| record | details.uniform.belt_type | 1 | 0.10% |
| record | rank_level | 1 | 0.10% |
| record | hangul_name | 1 | 0.10% |
| record | details.uniform.belt_color | 1 | 0.10% |
| record | creation.king | 1 | 0.10% |
| record | test_name | 1 | 0.10% |
| record | returner: Name of the person initiating the return | 1 | 0.10% |
| record | location: City or address where return was submitted | 1 | 0.10% |
| record | item: Description of the returned product | 1 | 0.10% |
| record | amount: Monetary value of the return in dollars | 1 | 0.10% |
| record | status: Current status of the return approval | 1 | 0.10% |
| record | limitations_of_midwives | 1 | 0.10% |
| record | resources | 1 | 0.10% |
| record | conference | 1 | 0.10% |
| record | co_authors | 1 | 0.10% |
| record | prior_experience | 1 | 0.10% |
| record | use_case | 1 | 0.10% |
| record | review_length | 1 | 0.10% |
| record | compatibility | 1 | 0.10% |
| record | release_year | 1 | 0.10% |
| record | charging_time | 1 | 0.10% |
| record | warranty_period | 1 | 0.10% |
| record | disease.subspecies.2.associated_diseases | 1 | 0.10% |
| record | disease.name | 1 | 0.10% |
| record | spo2: Oxygen saturation percentage | 1 | 0.10% |
| record | blood_pressure: Blood pressure in format systolic/diastolic | 1 | 0.10% |
| record | heart_rate: Number of heartbeats per minute | 1 | 0.10% |
| record | address: Street address including city and zip code | 1 | 0.10% |
| record | defendant: Name of the individual or entity being sued | 1 | 0.10% |
| record | case_number: Unique identifier for the court case | 1 | 0.10% |
| record | motion_status: Status of any filed motions such as denied or pending | 1 | 0.10% |
| record | case_type: Category of legal matter such as contract or personal injury | 1 | 0.10% |
| record | amountindispute: Monetary value of the claim or settlement | 1 | 0.10% |
| record | zip_code: Postal code associated with the office address | 1 | 0.10% |
| record | staff_count: Number of employees currently working at the office | 1 | 0.10% |
| record | Food | 1 | 0.10% |
| record | Attribute | 1 | 0.10% |
| record | position_code: Unique identifier for the appointment | 1 | 0.10% |
| record | department: Government department or agency where the appointment occurs | 1 | 0.10% |
| record | tags: List of hashtags used in the video post | 1 | 0.10% |
| record | town.population.density_per_sq_km | 1 | 0.10% |
| record | town.name | 1 | 0.10% |
| record | town.area_sq_km | 1 | 0.10% |
| record | town.population.year | 1 | 0.10% |
| record | town.location | 1 | 0.10% |
| record | disease_context | 1 | 0.10% |
| record | dedication.name | 1 | 0.10% |
| record | categories | 1 | 0.10% |
| record | responsible_engineer | 1 | 0.10% |
| record | acquisition_date | 1 | 0.10% |
| record | days | 1 | 0.10% |
| record | injury_count | 1 | 0.10% |
| record | app_version | 1 | 0.10% |
| record | functionality_comment | 1 | 0.10% |
| record | app_package | 1 | 0.10% |
| record | product_name: Name of the electronic device being reviewed | 1 | 0.10% |
| record | treatment_type | 1 | 0.10% |
| record | insurer | 1 | 0.10% |
| record | premium | 1 | 0.10% |
| record | team_id | 1 | 0.10% |
| record | hardware_support.gpu.amd.version | 1 | 0.10% |
| record | hardware_support.gpu.nvidia.api | 1 | 0.10% |
| record | operating_systems | 1 | 0.10% |
| record | miner_name | 1 | 0.10% |
| record | developer_fee | 1 | 0.10% |
| record | supported_cryptocurrencies | 1 | 0.10% |
| record | beds: Number of bedrooms | 1 | 0.10% |
| record | state: State where the property is located | 1 | 0.10% |
| record | city: City where the property is located | 1 | 0.10% |
| record | price: Listed price in dollars | 1 | 0.10% |
| record | drug_code | 1 | 0.10% |
| record | medicalrecordid | 1 | 0.10% |
| record | guidelines.distinction_results_interpretation | 1 | 0.10% |
| record | optional_information | 1 | 0.10% |
| record | research_framework_items | 1 | 0.10% |
| record | publication_date.city | 1 | 0.10% |
| record | publication_date.month | 1 | 0.10% |
| record | guidelines.adherence | 1 | 0.10% |
| record | organization.name | 1 | 0.10% |
| record | sender_company | 1 | 0.10% |
| record | recipient_location | 1 | 0.10% |
| record | asteroid.orbital_characteristics.unit | 1 | 0.10% |
| record | subject_focus: Key subjects or pathways emphasized in the results | 1 | 0.10% |
| record | grade_distribution: Breakdown of grades achieved by students | 1 | 0.10% |
| record | location: Geographic region or city where the institution is located | 1 | 0.10% |
| record | publication_code: Unique identifier assigned to the result publication | 1 | 0.10% |
| record | item_description | 1 | 0.10% |
| record | recommendations | 1 | 0.10% |
| record | investor_leads | 1 | 0.10% |
| record | funds_use | 1 | 0.10% |
| record | description: Brief description of the transaction activity | 1 | 0.10% |
| record | creditcardlast_four: Last four digits of the linked credit card | 1 | 0.10% |
| record | merchant_name: Name of the merchant or store | 1 | 0.10% |
| record | transaction_amount: Amount of the individual transaction | 1 | 0.10% |
| record | late_fee: Fee assessed for missing payment deadline | 1 | 0.10% |
| record | status: Transaction approval or rejection status | 1 | 0.10% |
| record | finance_charge: Fee due to unpaid balance from previous cycle | 1 | 0.10% |
| record | customer | 1 | 0.10% |
| record | order_id | 1 | 0.10% |
| record | return_method | 1 | 0.10% |
| record | tracking_number | 1 | 0.10% |
| record | creative work | 1 | 0.10% |
| record | venue: Name of the stadium or venue where the event occurred | 1 | 0.10% |
| record | nation: Country of the athlete who won the medal | 1 | 0.10% |
| record | time: Official race time recorded in the event | 1 | 0.10% |
| record | medal: Type of medal won gold, silver, or bronze | 1 | 0.10% |
| record | acquisition_year | 1 | 0.10% |
| record | startup_capital | 1 | 0.10% |
| record | hba1c: Hemoglobin A1c percentage value | 1 | 0.10% |
| record | address: Street address including number and street name | 1 | 0.10% |
| record | cholesterol: Cholesterol level in mg/dL if elevated | 1 | 0.10% |
| record | visit_date: Date of last medical check-up | 1 | 0.10% |
| record | state: State abbreviation of patient's residence | 1 | 0.10% |
| record | diagnosis: Medical condition such as hypertension or diabetes | 1 | 0.10% |
| record | age: Age in years as stated in the record | 1 | 0.10% |
| record | zip_code: Postal code of patient's address | 1 | 0.10% |
| record | hotel_details.reception_open_until | 1 | 0.10% |
| record | hotel_details.check_out | 1 | 0.10% |
| record | hotel_details.number_of_rooms | 1 | 0.10% |
| record | child_policy.children.age_range | 1 | 0.10% |
| record | hotel_details.check_in | 1 | 0.10% |
| record | donation_id: Unique identifier for each donation entry | 1 | 0.10% |
| record | status: Current status of the donation transaction | 1 | 0.10% |
| record | address: Full street address including city, state, and zip | 1 | 0.10% |
| record | donor: Name of the individual making the donation | 1 | 0.10% |
| record | claims_count | 1 | 0.10% |
| record | filing_location | 1 | 0.10% |
| record | game_status | 1 | 0.10% |
| record | event_type | 1 | 0.10% |
| record | performance_notes | 1 | 0.10% |
| record | test_id | 1 | 0.10% |
| record | reference_number: Unique identifier for the invoice document | 1 | 0.10% |
| record | recipient_name: Name of the recipient organization | 1 | 0.10% |
| record | status: Current status of the outreach interaction | 1 | 0.10% |
| record | sender: Name of the person sending the outreach | 1 | 0.10% |
| record | outreach_date: Date when the outreach email was sent | 1 | 0.10% |
| record | amount: Monetary value associated with the outreach | 1 | 0.10% |
| record | model_id | 1 | 0.10% |
| record | output_power | 1 | 0.10% |
| record | incident | 1 | 0.10% |
| record | claimant | 1 | 0.10% |
| record | vehicle_make | 1 | 0.10% |
| record | damage_type | 1 | 0.10% |
| record | provider_npi | 1 | 0.10% |
| record | species.family | 1 | 0.10% |
| record | first_client | 1 | 0.10% |
| record | treatment_plan | 1 | 0.10% |
| record | treatment_status | 1 | 0.10% |
| record | tournament | 1 | 0.10% |
| record | strategic_impact | 1 | 0.10% |
| record | regulatory_notes | 1 | 0.10% |
| record | deal_date | 1 | 0.10% |
| record | antitrust_review | 1 | 0.10% |
| record | focus_area | 1 | 0.10% |
| record | acquired_company | 1 | 0.10% |
| record | acquisition_id | 1 | 0.10% |
| record | sender: Email of the person sending the confirmation or submission | 1 | 0.10% |
| record | population_density | 1 | 0.10% |
| record | record_metadata | natural/location ×27, natural/status ×22, natural/name ×17, natural/description ×12 |  |

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | unknown |
| relations | unknown |
| events | unknown |
| classifications | unknown |
| structures | unknown |
