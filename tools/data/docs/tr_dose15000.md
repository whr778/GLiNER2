# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tr_dose15000 | train | 39,431 | 39,431 | 48,423 | 3,745 | 5,924 | 43,671 | 8,494 | 9,112 |
| tr_dose15000 | val | 1,115 | 1,115 | 0 | 0 | 0 | 1,633 | 0 | 0 |
| tr_dose15000 | test | 1,115 | 1,115 | 0 | 0 | 0 | 1,680 | 0 | 0 |

## tr_dose15000

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 39,431 | 39,431 | 203 | 669 | 22512 | 48,423 | 3,745 | 5,924 | 43,671 | 8,494 | 9,112 |
| val | 1,115 | 1,115 | 194 | 413 | 1145 | 0 | 0 | 0 | 1,633 | 0 | 0 |
| test | 1,115 | 1,115 | 200 | 437 | 1110 | 0 | 0 | 0 | 1,680 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 18,757 | 322 | 345 | 19,424 | 46.62% |
| tur | 15,002 | 793 | 770 | 16,565 | 39.76% |
| zho | 5,326 | 0 | 0 | 5,326 | 12.78% |
| und | 239 | 0 | 0 | 239 | 0.57% |
| fra | 16 | 0 | 0 | 16 | 0.04% |
| rus | 12 | 0 | 0 | 12 | 0.03% |
| por | 9 | 0 | 0 | 9 | 0.02% |
| kor | 9 | 0 | 0 | 9 | 0.02% |
| deu | 8 | 0 | 0 | 8 | 0.02% |
| ita | 8 | 0 | 0 | 8 | 0.02% |
| spa | 8 | 0 | 0 | 8 | 0.02% |
| hin | 6 | 0 | 0 | 6 | 0.01% |
| pol | 6 | 0 | 0 | 6 | 0.01% |
| nld | 5 | 0 | 0 | 5 | 0.01% |
| ukr | 5 | 0 | 0 | 5 | 0.01% |
| ara | 5 | 0 | 0 | 5 | 0.01% |
| ces | 2 | 0 | 0 | 2 | 0.00% |
| nor | 2 | 0 | 0 | 2 | 0.00% |
| fin | 2 | 0 | 0 | 2 | 0.00% |
| ilo | 1 | 0 | 0 | 1 | 0.00% |
| wuu | 1 | 0 | 0 | 1 | 0.00% |
| cym | 1 | 0 | 0 | 1 | 0.00% |
| epo | 1 | 0 | 0 | 1 | 0.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 1,115 |
| test_unique_inputs | 1,115 |
| train_records | 39,431 |
| train_unique_inputs | 39,431 |
| val_records | 1,115 |
| val_unique_inputs | 1,115 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | n/a | 48,423 / 0 |  |
| entities (per mention) | train / test | n/a | 48,423 / 0 |  |
| relations (per instance) | train / val | n/a | 3,745 / 0 |  |
| relations (per instance) | train / test | n/a | 3,745 / 0 |  |
| event types (per trigger mention) | train / val | n/a | 9,722 / 0 |  |
| event types (per trigger mention) | train / test | n/a | 9,722 / 0 |  |
| event roles (per argument) | train / val | n/a | 9,659 / 0 |  |
| event roles (per argument) | train / test | n/a | 9,659 / 0 |  |
| classifications (true_label) | train / val | n/a | 5,924 / 0 |  |
| classifications (true_label) | train / test | n/a | 5,924 / 0 |  |


### Events

**train.** 9,472 instances in 2,245 records, 0 triggerless. Trigger gold 8,494. Argument gold 9,112. Arguments per instance, mean 1.02. Trigger length in whitespace tokens, mean 1.39, single-token share 65.3%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Experiment | 695 | 7.15% |
| Support | 479 | 4.93% |
| Event | 377 | 3.88% |
| Hostile_encounter | 237 | 2.44% |
| Deploy | 236 | 2.43% |
| Process_start | 227 | 2.33% |
| Attack | 223 | 2.29% |
| Catastrophe | 189 | 1.94% |
| Causation | 182 | 1.87% |
| Competition | 180 | 1.85% |
| match_result | 169 | 1.74% |
| Exhibit | 160 | 1.65% |
| Cyber.DiscoverVulnerability | 160 | 1.65% |
| Motion | 142 | 1.46% |
| Social_event | 124 | 1.28% |
| product_publish | 123 | 1.27% |
| Killing | 121 | 1.24% |
| Cyber.Ransom | 113 | 1.16% |
| Cyber.Databreach | 108 | 1.11% |
| Conquering | 104 | 1.07% |
| Statement | 102 | 1.05% |
| Process_end | 102 | 1.05% |
| Cyber.Phishing | 95 | 0.98% |
| Control | 93 | 0.96% |
| Coming_to_be | 87 | 0.89% |
| Hold | 85 | 0.87% |
| Cyber.PatchVulnerability | 85 | 0.87% |
| Self_motion | 79 | 0.81% |
| Arriving | 78 | 0.80% |
| Death | 77 | 0.79% |
| Creating | 75 | 0.77% |
| Bodily_harm | 75 | 0.77% |
| Arranging | 72 | 0.74% |
| Earnings_and_losses | 72 | 0.74% |
| Cause_change_of_position_on_a_scale | 71 | 0.73% |
| Military_operation | 69 | 0.71% |
| Destroying | 68 | 0.70% |
| Preventing_or_letting | 67 | 0.69% |
| Cause_to_be_included | 66 | 0.68% |
| Getting | 65 | 0.67% |
| Accident | 65 | 0.67% |
| Becoming | 64 | 0.66% |
| death | 63 | 0.65% |
| Participation | 62 | 0.64% |
| Know | 59 | 0.61% |
| Presence | 55 | 0.57% |
| Supporting | 54 | 0.56% |
| Manoeuvre | 53 | 0.55% |
| Change_of_leadership | 52 | 0.53% |
| Supply | 52 | 0.53% |
| resignation | 51 | 0.52% |
| Traveling | 51 | 0.52% |
| Action | 51 | 0.52% |
| Damaging | 50 | 0.51% |
| Defending | 50 | 0.51% |
| Escaping | 50 | 0.51% |
| judicial_action_arrest | 49 | 0.50% |
| Expressing_publicly | 49 | 0.50% |
| Giving | 49 | 0.50% |
| Name_conferral | 49 | 0.50% |
| Change | 48 | 0.49% |
| Cause_change_of_strength | 48 | 0.49% |
| Request | 47 | 0.48% |
| Removing | 45 | 0.46% |
| Judgment_communication | 45 | 0.46% |
| Placing | 44 | 0.45% |
| Motion_directional | 44 | 0.45% |
| Agree_or_refuse_to_act | 41 | 0.42% |
| Arrest | 41 | 0.42% |
| Using | 40 | 0.41% |
| Violence | 40 | 0.41% |
| Choosing | 39 | 0.40% |
| Sending | 38 | 0.39% |
| Conflict | 37 | 0.38% |
| Injure | 37 | 0.38% |
| Influence | 34 | 0.35% |
| Bringing | 34 | 0.35% |
| championship_win | 33 | 0.34% |
| Recording | 32 | 0.33% |
| Perception_active | 31 | 0.32% |
| Reporting | 30 | 0.31% |
| Departing | 30 | 0.31% |
| advancement | 30 | 0.31% |
| Terrorism | 30 | 0.31% |
| Recovering | 29 | 0.30% |
| divorce | 29 | 0.30% |
| Deciding | 28 | 0.29% |
| Communication | 27 | 0.28% |
| Protest | 27 | 0.28% |
| Come_together | 27 | 0.28% |
| opening | 26 | 0.27% |
| Assistance | 26 | 0.27% |
| Building | 26 | 0.27% |
| Manufacturing | 26 | 0.27% |
| Aiming | 26 | 0.27% |
| Dispersal | 25 | 0.26% |
| Besieging | 25 | 0.26% |
| Hindering | 25 | 0.26% |
| product_recall | 25 | 0.26% |
| joined | 24 | 0.25% |
| Cause_to_make_progress | 24 | 0.25% |
| Scrutiny | 24 | 0.25% |
| Receiving | 23 | 0.24% |
| marriage | 23 | 0.24% |
| product_release | 22 | 0.23% |
| Expansion | 22 | 0.23% |
| price_limit_up | 22 | 0.23% |
| judicial_action_summon | 22 | 0.23% |
| Traffic Accident | 21 | 0.22% |
| Legal_rulings | 21 | 0.22% |
| Surrendering | 21 | 0.22% |
| Achieve | 21 | 0.22% |
| Change_event_time | 20 | 0.21% |
| Openness | 20 | 0.21% |
| Use_firearm | 19 | 0.20% |
| Response | 19 | 0.20% |
| Being_in_operation | 19 | 0.20% |
| Quarreling | 18 | 0.19% |
| Cause_to_amalgamate | 18 | 0.19% |
| Check | 18 | 0.19% |
| Connect | 18 | 0.19% |
| Collaboration | 18 | 0.19% |
| financing | 18 | 0.19% |
| Becoming_a_member | 17 | 0.17% |
| sale_acquisition | 17 | 0.17% |
| judicial_action_fine | 16 | 0.16% |
| departure | 16 | 0.16% |
| Sign_agreement | 16 | 0.16% |
| Publishing | 16 | 0.16% |
| competition_ban | 16 | 0.16% |
| Committing_crime | 16 | 0.16% |
| Confronting_problem | 15 | 0.15% |
| Commitment | 15 | 0.15% |
| Convincing | 15 | 0.15% |
| Legality | 14 | 0.14% |
| Commerce_pay | 14 | 0.14% |
| Criminal_investigation | 14 | 0.14% |
| judicial_action_imprisonment | 13 | 0.13% |
| product_award | 13 | 0.13% |
| Fire | 13 | 0.13% |
| Warning | 13 | 0.13% |
| layoff | 13 | 0.13% |
| Preserving | 13 | 0.13% |
| breakup | 13 | 0.13% |
| Coming_to_believe | 13 | 0.13% |
| product_delist | 12 | 0.12% |
| GiveUp | 12 | 0.12% |
| Telling | 12 | 0.12% |
| competition_withdrawal | 12 | 0.12% |
| Writing | 12 | 0.12% |
| Employment | 11 | 0.11% |
| Resolve_problem | 11 | 0.11% |
| Education_teaching | 11 | 0.11% |
| judicial_action_filing | 10 | 0.10% |
| Reforming_a_system | 10 | 0.10% |
| Patrolling | 10 | 0.10% |
| birthday_celebration | 10 | 0.10% |
| Bearing_arms | 10 | 0.10% |
| Earthquake | 10 | 0.10% |
| apologize_to | 10 | 0.10% |
| lost_contact | 10 | 0.10% |
| retirement | 10 | 0.10% |
| Change_sentiment | 9 | 0.09% |
| dissolution | 9 | 0.09% |
| Collapse | 9 | 0.09% |
| Reveal_secret | 9 | 0.09% |
| Commerce_sell | 8 | 0.08% |
| Emptying | 8 | 0.08% |
| wedding | 8 | 0.08% |
| judicial_action_report | 8 | 0.08% |
| price_limit_down | 8 | 0.08% |
| judicial_action_prosecution | 8 | 0.08% |
| birth | 8 | 0.08% |
| suspension | 8 | 0.08% |
| price_decrease | 8 | 0.08% |
| closing | 8 | 0.08% |
| proposal | 7 | 0.07% |
| Limiting | 7 | 0.07% |
| Body_movement | 7 | 0.07% |
| Kidnapping | 7 | 0.07% |
| Forming_relationships | 7 | 0.07% |
| Rite | 7 | 0.07% |
| Surrounding | 7 | 0.07% |
| Scouring | 7 | 0.07% |
| Temporary_stay | 7 | 0.07% |
| Adducing | 7 | 0.07% |
| praise | 6 | 0.06% |
| dismissal | 6 | 0.06% |
| Plane Crash | 6 | 0.06% |
| meet_with | 6 | 0.06% |
| Explosion | 6 | 0.06% |
| visit_on_set | 6 | 0.06% |
| Expend_resource | 6 | 0.06% |
| Cure | 6 | 0.06% |
| engagement | 5 | 0.05% |
| Wearing | 5 | 0.05% |
| Submitting_documents | 5 | 0.05% |
| Testing | 5 | 0.05% |
| Imposing_obligation | 5 | 0.05% |
| judicial_action_hearing | 5 | 0.05% |
| Prison | 5 | 0.05% |
| Award | 5 | 0.05% |
| march | 5 | 0.05% |
| GetReady | 5 | 0.05% |
| contract_termination | 5 | 0.05% |
| Rescuing | 5 | 0.05% |
| Labeling | 5 | 0.05% |
| Filling | 5 | 0.05% |
| Create_artwork | 5 | 0.05% |
| strike | 5 | 0.05% |
| Research | 5 | 0.05% |
| Releasing | 4 | 0.04% |
| Exchange | 4 | 0.04% |
| Commerce_buy | 4 | 0.04% |
| rate_decrease | 4 | 0.04% |
| Cost | 4 | 0.04% |
| price_increase | 4 | 0.04% |
| Hiding_objects | 4 | 0.04% |
| Practice | 4 | 0.04% |
| Revenge | 3 | 0.03% |
| ipo | 3 | 0.03% |
| Containing | 3 | 0.03% |
| Justifying | 3 | 0.03% |
| Lighting | 3 | 0.03% |
| thank | 3 | 0.03% |
| rate_increase | 3 | 0.03% |
| Rewards_and_punishments | 3 | 0.03% |
| Institutionalization | 2 | 0.02% |
| infidelity | 2 | 0.02% |
| Carry_goods | 2 | 0.02% |
| Flood | 2 | 0.02% |
| Suspicion | 2 | 0.02% |
| Incident | 2 | 0.02% |
| Robbery | 2 | 0.02% |
| Vocalizations | 2 | 0.02% |
| Having_or_lacking_access | 2 | 0.02% |
| Emergency | 1 | 0.01% |
| Ratification | 1 | 0.01% |
| Extradition | 1 | 0.01% |
| pregnancy | 1 | 0.01% |
| Theft | 1 | 0.01% |

**val.** No event instances.

**test.** No event instances.

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 9,659 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.69 | 29.67% | 1.71 | 28.63% | 0.04% | 0 | 9,659 | 0 | 9,722 |
| val | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |
| test | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 9,472 | 1.02% | 7,153 | 12.39% | 11.93% | 1.82% | 0.73% | 16.26% |
| val | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |
| test | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 97 | 532 | 7,153 |
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
| StartDate | 2,778 | 5.74% |
| EquityHolder | 2,484 | 5.13% |
| StockCode | 2,185 | 4.51% |
| CompanyName | 1,848 | 3.82% |
| StockAbbr | 1,845 | 3.81% |
| Location | 1,794 | 3.70% |
| PledgedShares | 1,695 | 3.50% |
| Date | 1,557 | 3.22% |
| Pledgee | 1,420 | 2.93% |
| TotalHoldingShares | 1,374 | 2.84% |
| TradedShares | 1,304 | 2.69% |
| TotalHoldingRatio | 1,231 | 2.54% |
| Casualties and Losses | 1,213 | 2.51% |
| Reduction Amount | 1,048 | 2.16% |
| TotalPledgedShares | 972 | 2.01% |
| EndDate | 877 | 1.81% |
| Reducing Shareholder | 798 | 1.65% |
| ReleasedDate | 788 | 1.63% |
| Cause | 689 | 1.42% |
| Reduction Start Date | 634 | 1.31% |
| Company Name | 538 | 1.11% |
| Announcement Date | 505 | 1.04% |
| LaterHoldingShares | 447 | 0.92% |
| pledge_amount | 327 | 0.68% |
| AveragePrice | 318 | 0.66% |
| Person | 299 | 0.62% |
| Organization | 257 | 0.53% |
| Candidates and their Political Parties | 255 | 0.53% |
| Recipient | 233 | 0.48% |
| pledgor | 233 | 0.48% |
| Affected Areas | 218 | 0.45% |
| Contest Participant | 212 | 0.44% |
| RepurchasedShares | 210 | 0.43% |
| Accepting Court | 202 | 0.42% |
| Weapon and Equippment | 197 | 0.41% |
| Loss Amount | 191 | 0.39% |
| Individual_protein | 189 | 0.39% |
| Attacker | 188 | 0.39% |
| Protesters | 186 | 0.38% |
| Accident Investigator | 179 | 0.37% |
| HighestTradingPrice | 178 | 0.37% |
| Protest Reason | 177 | 0.37% |
| Vulnerability | 172 | 0.36% |
| Target | 169 | 0.35% |
| Belligerents | 168 | 0.35% |
| protein | 167 | 0.34% |
| pledge_start_date | 156 | 0.32% |
| Affected Area | 149 | 0.31% |
| Charged Crime | 149 | 0.31% |
| Accusation | 146 | 0.30% |
| Increase Amount | 142 | 0.29% |
| ClosingDate | 140 | 0.29% |
| Flight No. | 140 | 0.29% |
| Frozen Amount | 140 | 0.29% |
| Increasing Shareholder | 134 | 0.28% |
| RepurchaseAmount | 129 | 0.27% |
| ruling_date | 129 | 0.27% |
| Position | 127 | 0.26% |
| Areas affected | 127 | 0.26% |
| DiseaseOrPhenotypicFeature | 122 | 0.25% |
| Policy Content | 120 | 0.25% |
| Husband | 117 | 0.24% |
| Scheduled Landing Place | 117 | 0.24% |
| Disease | 113 | 0.23% |
| Capabilities | 112 | 0.23% |
| frozen_shareholder | 110 | 0.23% |
| Influence People | 109 | 0.23% |
| Wife | 107 | 0.22% |
| The Final Votes and Percentages | 106 | 0.22% |
| Aid Agency | 104 | 0.21% |
| Crew | 104 | 0.21% |
| System | 104 | 0.21% |
| Strike Reason | 104 | 0.21% |
| Aircraft Agency | 100 | 0.21% |
| Game Name | 100 | 0.21% |
| Suspect | 99 | 0.20% |
| LegalInstitution | 98 | 0.20% |
| Storm Hit Location | 97 | 0.20% |
| Judgement Result/Prison Term | 96 | 0.20% |
| Freeze Start Date | 92 | 0.19% |
| ChemicalEntity | 90 | 0.19% |
| pledge_end_date | 88 | 0.18% |
| Policy Proposals | 85 | 0.18% |
| Time | 85 | 0.18% |
| Increase Start Date | 84 | 0.17% |
| PII | 84 | 0.17% |
| Accused People | 83 | 0.17% |
| Anti-pollution People/Organizations | 82 | 0.17% |
| Strike Outcome | 81 | 0.17% |
| Riot Reason | 79 | 0.16% |
| Solution | 79 | 0.16% |
| Champions | 78 | 0.16% |
| GeneOrGeneProduct | 78 | 0.16% |
| FrozeShares | 76 | 0.16% |
| Strike Agency | 74 | 0.15% |
| Country Visited | 72 | 0.15% |
| Habitat | 71 | 0.15% |
| Organization Name | 69 | 0.14% |
| LowestTradingPrice | 68 | 0.14% |
| Score | 67 | 0.14% |
| Participants | 67 | 0.14% |
| Pollution Source | 67 | 0.14% |
| Start Date | 64 | 0.13% |
| Location/Hospital | 62 | 0.13% |
| Passengers | 62 | 0.13% |
| Historical Sites | 62 | 0.13% |
| Microorganism | 62 | 0.13% |
| Election Name | 61 | 0.13% |
| Precautionary Measure | 61 | 0.13% |
| Summit Theme | 60 | 0.12% |
| Economic Loss | 60 | 0.12% |
| Sponsor | 59 | 0.12% |
| Magnitude | 59 | 0.12% |
| Acquiree | 57 | 0.12% |
| Survivors | 56 | 0.12% |
| Strikers Status | 56 | 0.12% |
| Resignated Person | 56 | 0.12% |
| Speaker Status | 56 | 0.12% |
| Artifacts and Their Chronology | 56 | 0.12% |
| Confirmed/Infected Cases | 55 | 0.11% |
| Participating Countries | 55 | 0.11% |
| compensation_amount | 54 | 0.11% |
| Deliberating Agency | 54 | 0.11% |
| protein-family | 54 | 0.11% |
| Countries Withdrawing from the Organization | 53 | 0.11% |
| Court | 53 | 0.11% |
| Death Cases | 53 | 0.11% |
| Record-breaking Project | 52 | 0.11% |
| Epicenter | 52 | 0.11% |
| Area Affected | 50 | 0.10% |
| Winner | 50 | 0.10% |
| Aid Supplies/Amount | 50 | 0.10% |
| Effective Date | 49 | 0.10% |
| Data | 49 | 0.10% |
| Marriage Duration | 49 | 0.10% |
| Damaged Facility | 48 | 0.10% |
| Time of the Case | 47 | 0.10% |
| Child | 47 | 0.10% |
| Discover Location | 47 | 0.10% |
| Record Breaker | 46 | 0.09% |
| Appointee | 46 | 0.09% |
| Device | 45 | 0.09% |
| Method | 44 | 0.09% |
| Patch | 44 | 0.09% |
| Resign Reason | 43 | 0.09% |
| Grades | 43 | 0.09% |
| Award | 43 | 0.09% |
| compensation_recipient | 42 | 0.09% |
| Weapon | 42 | 0.09% |
| Speaker | 41 | 0.08% |
| Contracting Parties | 41 | 0.08% |
| Poisoning Type | 41 | 0.08% |
| Number of Vehicles Involved in the Crash | 41 | 0.08% |
| Army | 40 | 0.08% |
| Taking-off Place | 40 | 0.08% |
| Prosecutor | 40 | 0.08% |
| Software | 40 | 0.08% |
| Gene/protein/RNA | 40 | 0.08% |
| Person under Investigation | 39 | 0.08% |
| senior_personnel | 39 | 0.08% |
| senior_position | 39 | 0.08% |
| Employment Agency | 39 | 0.08% |
| Host | 38 | 0.08% |
| Investigative Agency | 38 | 0.08% |
| Spectacle Location | 38 | 0.08% |
| Freeze End Date | 38 | 0.08% |
| Previous Record Holder | 37 | 0.08% |
| Way for Spreading | 37 | 0.08% |
| Deceased | 37 | 0.08% |
| OrganismTaxon | 36 | 0.07% |
| CellType | 36 | 0.07% |
| Perpetrators | 36 | 0.07% |
| Spacecraft | 35 | 0.07% |
| People | 35 | 0.07% |
| Visitor | 35 | 0.07% |
| Purpose | 35 | 0.07% |
| Types of the Spectacle | 35 | 0.07% |
| The Agency Who Broke the Agreement | 35 | 0.07% |
| The Charged Crime | 34 | 0.07% |
| Profession | 34 | 0.07% |
| Epidemic Data Issuing Agency | 33 | 0.07% |
| Strikers | 33 | 0.07% |
| Award Reason | 33 | 0.07% |
| Spectacle Start Time | 33 | 0.07% |
| Defense Lawyer | 32 | 0.07% |
| Protest Slogan | 32 | 0.07% |
| Number of Damaged Houses | 32 | 0.07% |
| End Time | 31 | 0.06% |
| Organization Industry | 31 | 0.06% |
| Outbreak Location | 31 | 0.06% |
| Symptom | 31 | 0.06% |
| Version | 31 | 0.06% |
| Storm Name | 31 | 0.06% |
| The Grades of the Previous Record Holder | 30 | 0.06% |
| File | 30 | 0.06% |
| Death Reason | 30 | 0.06% |
| SequenceVariant | 29 | 0.06% |
| Malware | 29 | 0.06% |
| Number of Influenced People | 29 | 0.06% |
| People/Organization who predicted the disaster | 29 | 0.06% |
| Storm Hit Time | 29 | 0.06% |
| Declarer | 28 | 0.06% |
| Boycotted Institutions | 28 | 0.06% |
| Maximum Wind Speed | 28 | 0.06% |
| Archaeologist | 28 | 0.06% |
| Fine Reason | 28 | 0.06% |
| Death Time | 27 | 0.06% |
| Storm Warning Level | 27 | 0.06% |
| Evidence | 27 | 0.06% |
| Goal | 27 | 0.06% |
| Competition Items | 26 | 0.05% |
| Conflict Duration | 26 | 0.05% |
| Released People | 26 | 0.05% |
| Age | 26 | 0.05% |
| Number of Destroyed Building | 26 | 0.05% |
| Agreement Name | 26 | 0.05% |
| Arrested | 26 | 0.05% |
| Train No. | 26 | 0.05% |
| Judge | 25 | 0.05% |
| Anatomy | 25 | 0.05% |
| GeneProtein | 25 | 0.05% |
| Number of Evacuated People | 25 | 0.05% |
| CVE | 25 | 0.05% |
| chemical | 25 | 0.05% |
| Spacecraft Mission | 24 | 0.05% |
| Pests | 24 | 0.05% |
| Predecessor | 24 | 0.05% |
| Fined Agency | 24 | 0.05% |
| Affected Industries | 23 | 0.05% |
| End Date | 23 | 0.05% |
| Agreement Content | 23 | 0.05% |
| The Final Seats Result | 23 | 0.05% |
| Arrest Time | 22 | 0.05% |
| Last Job of the Appointee | 22 | 0.05% |
| Website | 22 | 0.05% |
| Launch Date | 22 | 0.05% |
| Money | 22 | 0.05% |
| Attending Hospital | 22 | 0.05% |
| Ship No. | 22 | 0.05% |
| Phenotype | 22 | 0.05% |
| Investigating agency | 21 | 0.04% |
| Annoncement Date | 21 | 0.04% |
| Rescue Organizer | 21 | 0.04% |
| Court Time | 21 | 0.04% |
| Approver | 21 | 0.04% |
| Party Responsible for Pollution | 21 | 0.04% |
| Areas Affected | 21 | 0.04% |
| Police | 21 | 0.04% |
| Number | 20 | 0.04% |
| Missings | 19 | 0.04% |
| Duration | 19 | 0.04% |
| Penalty Amount | 19 | 0.04% |
| Concept | 19 | 0.04% |
| Carrier Rocket | 19 | 0.04% |
| Military Strength | 19 | 0.04% |
| Other Loss | 19 | 0.04% |
| Start Time | 19 | 0.04% |
| Release Reason | 18 | 0.04% |
| Award Field | 18 | 0.04% |
| Regulatory Authority | 18 | 0.04% |
| Military Exercise | 18 | 0.04% |
| Launch Site | 17 | 0.04% |
| Bank Name | 17 | 0.04% |
| Inauguration Time | 17 | 0.04% |
| Symptoms | 17 | 0.04% |
| Discover Time | 17 | 0.04% |
| UnfrozeDate | 17 | 0.04% |
| Magnitude(Tsunami heights) | 17 | 0.04% |
| Launch Result | 17 | 0.04% |
| Victim | 17 | 0.04% |
| Head of Investigation Team | 17 | 0.04% |
| Last Time the Record was Broken | 16 | 0.03% |
| Child Custody | 16 | 0.03% |
| Head of the Merged Organization | 16 | 0.03% |
| Company Industry | 16 | 0.03% |
| Election Goal | 16 | 0.03% |
| Treatment Method | 16 | 0.03% |
| Casualty Count | 16 | 0.03% |
| protein-domain | 16 | 0.03% |
| Witness | 16 | 0.03% |
| Research Agency | 16 | 0.03% |
| Number of Victims | 16 | 0.03% |
| Scale | 16 | 0.03% |
| Amount of Precipitation | 16 | 0.03% |
| protein-complex | 15 | 0.03% |
| Term of Office | 15 | 0.03% |
| News Release Agency | 15 | 0.03% |
| Group | 15 | 0.03% |
| Outbreak Date | 15 | 0.03% |
| Electoral System | 15 | 0.03% |
| Policy Name & Abbreviation | 14 | 0.03% |
| DNA | 14 | 0.03% |
| Beneficiary | 14 | 0.03% |
| Susceptible Population | 14 | 0.03% |
| Transportation | 14 | 0.03% |
| Phylum | 14 | 0.03% |
| Volcano Name | 14 | 0.03% |
| Host Country | 14 | 0.03% |
| Tribe | 14 | 0.03% |
| Lasting Time | 13 | 0.03% |
| Summit Name | 13 | 0.03% |
| Turnout(who win who lost) | 13 | 0.03% |
| Casualities and Losses | 13 | 0.03% |
| Weapon Used | 13 | 0.03% |
| Response Measures | 13 | 0.03% |
| Achievement | 13 | 0.03% |
| Candidate Scandal(Allegations of fraud, etc.) | 13 | 0.03% |
| Property Division | 13 | 0.03% |
| GPE | 13 | 0.03% |
| CellLine | 13 | 0.03% |
| Country | 12 | 0.02% |
| Release Time | 12 | 0.02% |
| Forecasting Agency | 12 | 0.02% |
| Government Reaction | 12 | 0.02% |
| Age of the Appointee | 12 | 0.02% |
| Withdraw Reason | 12 | 0.02% |
| Appointment Time | 12 | 0.02% |
| Voting method | 12 | 0.02% |
| cell | 12 | 0.02% |
| Projects | 12 | 0.02% |
| species | 11 | 0.02% |
| MVP | 11 | 0.02% |
| Ways to Watch the Speech | 11 | 0.02% |
| Species | 11 | 0.02% |
| Number of Rescued People | 11 | 0.02% |
| Merger Terms | 11 | 0.02% |
| Sentencing Location | 11 | 0.02% |
| Aid Reason | 11 | 0.02% |
| Criminal Evidence | 11 | 0.02% |
| Spokeswoman/Spokesman | 11 | 0.02% |
| Organization Members | 11 | 0.02% |
| Number of Trapped People | 11 | 0.02% |
| algorithm | 11 | 0.02% |
| mathematician | 11 | 0.02% |
| Influenced Crops and Livelihood | 11 | 0.02% |
| Work | 11 | 0.02% |
| model | 11 | 0.02% |
| Doctor and Medical Team | 10 | 0.02% |
| Acquisition amount | 10 | 0.02% |
| Announce Platform | 10 | 0.02% |
| Funding | 10 | 0.02% |
| Death Rate | 10 | 0.02% |
| Archaeologist Organization | 10 | 0.02% |
| experimental-construct | 10 | 0.02% |
| Astronauts | 10 | 0.02% |
| PaymentMethod | 10 | 0.02% |
| Prosecution Lawyer | 10 | 0.02% |
| Acquirer | 10 | 0.02% |
| reagent | 10 | 0.02% |
| Product | 10 | 0.02% |
| organization | 10 | 0.02% |
| Cooperative Agency | 9 | 0.02% |
| Stolen Amount | 9 | 0.02% |
| Spectacle Duration | 9 | 0.02% |
| experiment-tag | 9 | 0.02% |
| The State of the Volcano (Dormant or Active) | 9 | 0.02% |
| Successor | 9 | 0.02% |
| Agreement Members | 9 | 0.02% |
| Tear Up Reason | 9 | 0.02% |
| Protein_family_or_group | 9 | 0.02% |
| Train Agency | 9 | 0.02% |
| Mathematical Concept | 9 | 0.02% |
| Launch Country | 8 | 0.02% |
| Economic loss | 8 | 0.02% |
| Investigating Spokesperson | 8 | 0.02% |
| Other Impact | 8 | 0.02% |
| Damaged Property | 8 | 0.02% |
| Water Level | 8 | 0.02% |
| Arrest Location | 8 | 0.02% |
| Jail Time | 8 | 0.02% |
| Temporary Settlement | 8 | 0.02% |
| Head of Institution | 8 | 0.02% |
| Number of Aftershocks | 8 | 0.02% |
| Hull Discovery Time | 8 | 0.02% |
| Wedding Venue | 8 | 0.02% |
| Mathematical Structure | 8 | 0.02% |
| Planet | 8 | 0.02% |
| Hostage | 7 | 0.01% |
| Shipwreck Reason | 7 | 0.01% |
| Prison Term | 7 | 0.01% |
| Prison | 7 | 0.01% |
| Mission Duration | 7 | 0.01% |
| Illness | 7 | 0.01% |
| Spokenmen | 7 | 0.01% |
| Last Outbreak Time | 7 | 0.01% |
| Employed Institution | 7 | 0.01% |
| Service Years | 7 | 0.01% |
| Start_Date | 7 | 0.01% |
| Gene | 7 | 0.01% |
| ai_model | 7 | 0.01% |
| Organization Leader | 6 | 0.01% |
| Tsunamis | 6 | 0.01% |
| Wedding Dress Designer | 6 | 0.01% |
| Unemployed Rate | 6 | 0.01% |
| Disaster-Stricken Farmland | 6 | 0.01% |
| Cured Cases | 6 | 0.01% |
| Number of Vaccinated People | 6 | 0.01% |
| Spectacle End Time | 6 | 0.01% |
| Perpetrator | 6 | 0.01% |
| State Before Death | 6 | 0.01% |
| Invited Person | 6 | 0.01% |
| mathematical_construct | 6 | 0.01% |
| Bill Drafting Agency | 6 | 0.01% |
| mathematical_result | 6 | 0.01% |
| Related Rivers or Lakes | 6 | 0.01% |
| Maximum Rainfall | 6 | 0.01% |
| Rescue Tool or Method | 6 | 0.01% |
| target | 6 | 0.01% |
| Sequelae | 6 | 0.01% |
| Best Way to Shoot | 6 | 0.01% |
| Adjective | 6 | 0.01% |
| dataset | 6 | 0.01% |
| Publication | 6 | 0.01% |
| Special Medicine | 5 | 0.01% |
| Number of People Hospitalized | 5 | 0.01% |
| Age of the Resignated Person | 5 | 0.01% |
| Hull Location | 5 | 0.01% |
| process | 5 | 0.01% |
| agent | 5 | 0.01% |
| Commanders and their Position | 5 | 0.01% |
| Art Form | 5 | 0.01% |
| Field of Study | 5 | 0.01% |
| Literary Work | 5 | 0.01% |
| metric | 5 | 0.01% |
| Appointer | 5 | 0.01% |
| loss_function | 5 | 0.01% |
| Provisional Government | 4 | 0.01% |
| Postpone Time | 4 | 0.01% |
| Postpone Reason | 4 | 0.01% |
| Storm Direction | 4 | 0.01% |
| Storm Movement Speed | 4 | 0.01% |
| fusion-protein | 4 | 0.01% |
| Commanders of the Army | 4 | 0.01% |
| protein-motif | 4 | 0.01% |
| Alternate Landing Place | 4 | 0.01% |
| chemical_compound | 4 | 0.01% |
| researcher | 4 | 0.01% |
| geometric_curve | 4 | 0.01% |
| group | 4 | 0.01% |
| Compensation | 4 | 0.01% |
| Object | 4 | 0.01% |
| Damaged Crops & Livestock | 4 | 0.01% |
| Art Movement | 4 | 0.01% |
| Role | 4 | 0.01% |
| concept | 4 | 0.01% |
| mathematical_operator | 4 | 0.01% |
| Head of the Institution | 4 | 0.01% |
| Invalid Date | 4 | 0.01% |
| Death Age | 4 | 0.01% |
| Strike Industry | 4 | 0.01% |
| environment | 4 | 0.01% |
| Economists who Predicted the Crisis | 3 | 0.01% |
| Subphylum | 3 | 0.01% |
| Organism | 3 | 0.01% |
| Population | 3 | 0.01% |
| Medical Test | 3 | 0.01% |
| Theory | 3 | 0.01% |
| Trapped Depth | 3 | 0.01% |
| Complications | 3 | 0.01% |
| chemical_group | 3 | 0.01% |
| physical_property | 3 | 0.01% |
| Agreement Validity Period | 3 | 0.01% |
| Recovered Amount | 3 | 0.01% |
| decoder_architecture | 3 | 0.01% |
| Fire Warning Level | 3 | 0.01% |
| The Sentence Claimed by the Defense Lawyer | 3 | 0.01% |
| Detention Start Time | 3 | 0.01% |
| mathematical_concept | 3 | 0.01% |
| organelle | 3 | 0.01% |
| Reasons for the Formation of the Historical sites | 3 | 0.01% |
| Nation | 3 | 0.01% |
| Event | 3 | 0.01% |
| RNA | 3 | 0.01% |
| Influenced People | 3 | 0.01% |
| End_Date | 3 | 0.01% |
| Time Period | 3 | 0.01% |
| Material | 3 | 0.01% |
| algebraic_structure | 3 | 0.01% |
| People Casting Key Votes | 3 | 0.01% |
| Suspected Cases | 3 | 0.01% |
| Disease Subtype | 3 | 0.01% |
| Medical Professional | 3 | 0.01% |
| Moon | 3 | 0.01% |
| embodiment | 3 | 0.01% |
| Warning Device | 2 | 0.00% |
| CellComponent | 2 | 0.00% |
| Biological Group | 2 | 0.00% |
| Geological Period | 2 | 0.00% |
| Enzyme | 2 | 0.00% |
| Body Part | 2 | 0.00% |
| Disease Form | 2 | 0.00% |
| Service | 2 | 0.00% |
| peptide | 2 | 0.00% |
| Principle | 2 | 0.00% |
| Model | 2 | 0.00% |
| Phenomenon | 2 | 0.00% |
| molecular_orbital | 2 | 0.00% |
| researcher_group | 2 | 0.00% |
| institutional_unit | 2 | 0.00% |
| Lawyer | 2 | 0.00% |
| error_correcting_code | 2 | 0.00% |
| hardware_element | 2 | 0.00% |
| geometric_object | 2 | 0.00% |
| numerical_function | 2 | 0.00% |
| mathematical_structure | 2 | 0.00% |
| Civilization | 2 | 0.00% |
| Law | 2 | 0.00% |
| protein-region | 2 | 0.00% |
| Storm Center Location | 2 | 0.00% |
| Countries Joined the Organization | 2 | 0.00% |
| Field | 2 | 0.00% |
| Pest | 2 | 0.00% |
| Equipment | 2 | 0.00% |
| Metric | 2 | 0.00% |
| Damage | 2 | 0.00% |
| Economy Loss | 2 | 0.00% |
| Political Entity | 2 | 0.00% |
| Writing System | 2 | 0.00% |
| Plant Species | 2 | 0.00% |
| Grammatical Concept | 2 | 0.00% |
| Test | 2 | 0.00% |
| alignment_technique | 2 | 0.00% |
| adversarial_attack | 2 | 0.00% |
| model_component | 2 | 0.00% |
| Bankrupt Business | 2 | 0.00% |
| technical_method | 2 | 0.00% |
| The Sentence Claimed by the Prosecutor's Lawyer | 2 | 0.00% |
| amino-acid | 2 | 0.00% |
| Hypothetical Planet | 2 | 0.00% |
| modality | 2 | 0.00% |
| architecture_component | 2 | 0.00% |
| Exit Conditions | 1 | 0.00% |
| Head of the Government | 1 | 0.00% |
| How Many Times Get Married | 1 | 0.00% |
| Religion | 1 | 0.00% |
| Genetic Element | 1 | 0.00% |
| Substance | 1 | 0.00% |
| Biological Entity | 1 | 0.00% |
| parameter | 1 | 0.00% |
| element | 1 | 0.00% |
| computational_method | 1 | 0.00% |
| software_package | 1 | 0.00% |
| error_type | 1 | 0.00% |
| publication_source | 1 | 0.00% |
| Document | 1 | 0.00% |
| Refuge | 1 | 0.00% |
| Registered Capital | 1 | 0.00% |
| Protein_complex | 1 | 0.00% |
| gene | 1 | 0.00% |
| Language | 1 | 0.00% |
| Word | 1 | 0.00% |
| Medium | 1 | 0.00% |
| Scientific Name | 1 | 0.00% |
| Control Method | 1 | 0.00% |
| Strategy | 1 | 0.00% |
| Natural Enemy | 1 | 0.00% |
| Lure | 1 | 0.00% |
| Lost Contact Time | 1 | 0.00% |
| Cured Rate | 1 | 0.00% |
| Film | 1 | 0.00% |
| Measurement | 1 | 0.00% |
| Plant Group | 1 | 0.00% |
| Habitat Feature | 1 | 0.00% |
| Chemical | 1 | 0.00% |
| Animal | 1 | 0.00% |
| Linguistic Concept | 1 | 0.00% |
| Entity | 1 | 0.00% |
| Platform | 1 | 0.00% |
| evaluation_metric | 1 | 0.00% |
| evaluation_method | 1 | 0.00% |
| Narrative | 1 | 0.00% |
| Text | 1 | 0.00% |
| Invention | 1 | 0.00% |
| topological_space | 1 | 0.00% |
| mathematical_identity | 1 | 0.00% |
| Doctor | 1 | 0.00% |
| architectural_component | 1 | 0.00% |
| Production Cuts | 1 | 0.00% |
| Mathematical Result | 1 | 0.00% |
| Mathematical Theorem | 1 | 0.00% |
| Symbol | 1 | 0.00% |
| Cost | 1 | 0.00% |
| protein-isoform | 1 | 0.00% |
| Protein | 1 | 0.00% |
| Treatment | 1 | 0.00% |
| Community Group | 1 | 0.00% |
| Program | 1 | 0.00% |
| Rescue Start Time | 1 | 0.00% |
| Publication Issue | 1 | 0.00% |
| Celestial System | 1 | 0.00% |
| Religious Text | 1 | 0.00% |
| Deity | 1 | 0.00% |
| Star | 1 | 0.00% |
| Scientific Theory | 1 | 0.00% |
| Celestial Object | 1 | 0.00% |
| task_category | 1 | 0.00% |
| capability | 1 | 0.00% |
| data_format | 1 | 0.00% |
| data_structure | 1 | 0.00% |
| publication_venue | 1 | 0.00% |
| State of the Hull | 1 | 0.00% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| bind | 250 | 6.68% |
| PPI | 170 | 4.54% |
| Lives_In | 90 | 2.40% |
| Association | 87 | 2.32% |
| Positive_Correlation | 78 | 2.08% |
| Negative_Correlation | 37 | 0.99% |
| Exhibits | 25 | 0.67% |
| record label | 23 | 0.61% |
| located in the administrative territorial entity | 20 | 0.53% |
| architect | 20 | 0.53% |
| conflict | 19 | 0.51% |
| package management system | 18 | 0.48% |
| original broadcaster | 18 | 0.48% |
| member of | 18 | 0.48% |
| owned by | 17 | 0.45% |
| employer | 17 | 0.45% |
| historic county | 17 | 0.45% |
| country | 17 | 0.45% |
| performer | 17 | 0.45% |
| movement | 17 | 0.45% |
| discoverer or inventor | 16 | 0.43% |
| conferred by | 16 | 0.43% |
| creator | 16 | 0.43% |
| characters | 16 | 0.43% |
| parent taxon | 16 | 0.43% |
| platform | 16 | 0.43% |
| crosses | 15 | 0.40% |
| position held | 14 | 0.37% |
| licensed to broadcast to | 14 | 0.37% |
| represented by | 14 | 0.37% |
| located in/on physical feature | 13 | 0.35% |
| present in work | 13 | 0.35% |
| father | 13 | 0.35% |
| award received | 13 | 0.35% |
| has effect | 13 | 0.35% |
| member of sports team | 13 | 0.35% |
| dam | 13 | 0.35% |
| military branch | 13 | 0.35% |
| composer | 13 | 0.35% |
| astronaut mission | 13 | 0.35% |
| item operated | 13 | 0.35% |
| commanded by | 12 | 0.32% |
| founded by | 12 | 0.32% |
| ethnic group | 12 | 0.32% |
| participant in | 12 | 0.32% |
| director | 12 | 0.32% |
| league | 12 | 0.32% |
| significant event | 12 | 0.32% |
| place of detention | 12 | 0.32% |
| genre | 12 | 0.32% |
| operator | 12 | 0.32% |
| connects with | 12 | 0.32% |
| partner in business or sport | 12 | 0.32% |
| cast member | 12 | 0.32% |
| located on astronomical body | 12 | 0.32% |
| position played on team / speciality | 12 | 0.32% |
| publisher | 12 | 0.32% |
| mouth of the watercourse | 11 | 0.29% |
| owner of | 11 | 0.29% |
| location of discovery | 11 | 0.29% |
| connecting line | 11 | 0.29% |
| has use | 11 | 0.29% |
| spouse | 11 | 0.29% |
| occupation | 11 | 0.29% |
| basin country | 11 | 0.29% |
| family | 11 | 0.29% |
| winner | 11 | 0.29% |
| heritage designation | 11 | 0.29% |
| mountain range | 11 | 0.29% |
| product or material produced | 11 | 0.29% |
| member of political party | 11 | 0.29% |
| uses | 11 | 0.29% |
| parent organization | 11 | 0.29% |
| worshipped by | 10 | 0.27% |
| sports discipline competed in | 10 | 0.27% |
| author | 10 | 0.27% |
| after a work by | 10 | 0.27% |
| language used | 10 | 0.27% |
| transport network | 10 | 0.27% |
| child | 10 | 0.27% |
| applies to jurisdiction | 10 | 0.27% |
| convicted of | 10 | 0.27% |
| surface played on | 10 | 0.27% |
| stock market index | 10 | 0.27% |
| biological process | 10 | 0.27% |
| significant place | 10 | 0.27% |
| crew member(s) | 10 | 0.27% |
| location of formation | 10 | 0.27% |
| terminus | 10 | 0.27% |
| secretary general | 10 | 0.27% |
| country for sport | 10 | 0.27% |
| mother | 10 | 0.27% |
| chromosome | 10 | 0.27% |
| located in the present-day administrative territorial entity | 10 | 0.27% |
| ammunition | 10 | 0.27% |
| production company | 10 | 0.27% |
| head of government | 9 | 0.24% |
| has part(s) of the class | 9 | 0.24% |
| notable work | 9 | 0.24% |
| legislated by | 9 | 0.24% |
| operating area | 9 | 0.24% |
| stock exchange | 9 | 0.24% |
| field of work | 9 | 0.24% |
| taxonomic type | 9 | 0.24% |
| nominated for | 9 | 0.24% |
| maintained by | 9 | 0.24% |
| presenter | 9 | 0.24% |
| medical condition | 9 | 0.24% |
| inflows | 9 | 0.24% |
| officeholder | 9 | 0.24% |
| educated at | 9 | 0.24% |
| basic form of government | 9 | 0.24% |
| broadcast by | 9 | 0.24% |
| located on street | 9 | 0.24% |
| constellation | 9 | 0.24% |
| part of the series | 9 | 0.24% |
| has subsidiary | 9 | 0.24% |
| located in the statistical territorial entity | 9 | 0.24% |
| sport | 9 | 0.24% |
| shares border with | 9 | 0.24% |
| collection | 9 | 0.24% |
| diocese | 9 | 0.24% |
| opposite of | 9 | 0.24% |
| home venue | 9 | 0.24% |
| online service | 9 | 0.24% |
| pole position | 9 | 0.24% |
| connecting service | 8 | 0.21% |
| work location | 8 | 0.21% |
| editor | 8 | 0.21% |
| illustrator | 8 | 0.21% |
| this zoological name is coordinate with | 8 | 0.21% |
| published in | 8 | 0.21% |
| cause of destruction | 8 | 0.21% |
| currency | 8 | 0.21% |
| part of | 8 | 0.21% |
| terminus location | 8 | 0.21% |
| reservoir created | 8 | 0.21% |
| commemorates | 8 | 0.21% |
| this taxon is source of | 8 | 0.21% |
| replaced by | 8 | 0.21% |
| location | 8 | 0.21% |
| league system | 8 | 0.21% |
| distributed by | 8 | 0.21% |
| member of the crew of | 8 | 0.21% |
| natural product of taxon | 8 | 0.21% |
| fictional universe described in | 8 | 0.21% |
| followed by | 8 | 0.21% |
| writing language | 8 | 0.21% |
| instrument | 8 | 0.21% |
| anatomical location | 8 | 0.21% |
| cathedral | 8 | 0.21% |
| replaces | 8 | 0.21% |
| producer | 8 | 0.21% |
| primary destinations | 8 | 0.21% |
| main subject | 8 | 0.21% |
| place served by transport hub | 8 | 0.21% |
| contributed to creative work | 8 | 0.21% |
| noble title | 8 | 0.21% |
| participant | 8 | 0.21% |
| takes place in fictional universe | 8 | 0.21% |
| has works in the collection | 7 | 0.19% |
| edition or translation of | 7 | 0.19% |
| medical condition treated | 7 | 0.19% |
| has cause | 7 | 0.19% |
| start point | 7 | 0.19% |
| newspaper format | 7 | 0.19% |
| brand | 7 | 0.19% |
| instance of | 7 | 0.19% |
| native language | 7 | 0.19% |
| headquarters location | 7 | 0.19% |
| origin of the watercourse | 7 | 0.19% |
| highest judicial authority | 7 | 0.19% |
| standards body | 7 | 0.19% |
| legal form | 7 | 0.19% |
| programming language | 7 | 0.19% |
| culture | 7 | 0.19% |
| librettist | 7 | 0.19% |
| head coach | 7 | 0.19% |
| vehicle | 7 | 0.19% |
| has immediate cause | 7 | 0.19% |
| commissioned by | 7 | 0.19% |
| discography | 7 | 0.19% |
| merged into | 7 | 0.19% |
| located in or next to body of water | 7 | 0.19% |
| place of publication | 7 | 0.19% |
| continent | 7 | 0.19% |
| set in period | 7 | 0.19% |
| grants | 7 | 0.19% |
| recommended unit of measurement | 7 | 0.19% |
| target | 7 | 0.19% |
| dialect of | 7 | 0.19% |
| facet of | 7 | 0.19% |
| historical region | 7 | 0.19% |
| musical conductor | 7 | 0.19% |
| director / manager | 7 | 0.19% |
| made from material | 7 | 0.19% |
| operating system | 7 | 0.19% |
| consecrator | 7 | 0.19% |
| has part(s) | 7 | 0.19% |
| voice type | 7 | 0.19% |
| IUCN conservation status | 7 | 0.19% |
| head of state | 6 | 0.16% |
| child astronomical body | 6 | 0.16% |
| airline alliance | 6 | 0.16% |
| adjacent station | 6 | 0.16% |
| space launch vehicle | 6 | 0.16% |
| Bind | 6 | 0.16% |
| military rank | 6 | 0.16% |
| objective of project or action | 6 | 0.16% |
| produced by | 6 | 0.16% |
| endemic to | 6 | 0.16% |
| commander of (DEPRECATED) | 6 | 0.16% |
| parent astronomical body | 6 | 0.16% |
| form of creative work | 6 | 0.16% |
| physically interacts with | 6 | 0.16% |
| anatomical branch of | 6 | 0.16% |
| track gauge | 6 | 0.16% |
| has anatomical branch | 6 | 0.16% |
| participating team | 6 | 0.16% |
| board member | 6 | 0.16% |
| contains | 6 | 0.16% |
| dedicated to | 6 | 0.16% |
| court | 6 | 0.16% |
| model item | 6 | 0.16% |
| first appearance | 6 | 0.16% |
| candidate | 6 | 0.16% |
| carries | 6 | 0.16% |
| narrative location | 6 | 0.16% |
| genic_interaction | 6 | 0.16% |
| canonization status | 6 | 0.16% |
| location of first performance | 6 | 0.16% |
| vehicle normally used | 5 | 0.13% |
| powered by | 5 | 0.13% |
| parent of this hybrid, breed, or cultivar | 5 | 0.13% |
| lake on watercourse | 5 | 0.13% |
| authority | 5 | 0.13% |
| coextensive with | 5 | 0.13% |
| developer | 5 | 0.13% |
| contains the administrative territorial entity | 5 | 0.13% |
| measured physical quantity | 5 | 0.13% |
| found in taxon | 5 | 0.13% |
| time period | 5 | 0.13% |
| office held by head of the organization | 5 | 0.13% |
| source of energy | 5 | 0.13% |
| has written for | 5 | 0.13% |
| has edition or translation | 5 | 0.13% |
| described by source | 5 | 0.13% |
| named after | 5 | 0.13% |
| director of photography | 5 | 0.13% |
| overlies | 5 | 0.13% |
| tracklist | 5 | 0.13% |
| distribution format | 5 | 0.13% |
| qualifies for event | 5 | 0.13% |
| defendant | 5 | 0.13% |
| home world | 5 | 0.13% |
| filming location | 5 | 0.13% |
| incarnation of | 5 | 0.13% |
| competition class | 5 | 0.13% |
| armament | 5 | 0.13% |
| implementation of | 5 | 0.13% |
| public holiday | 5 | 0.13% |
| is the study of | 5 | 0.13% |
| office held by head of state | 5 | 0.13% |
| country of origin | 5 | 0.13% |
| language of work or name | 5 | 0.13% |
| performed at | 5 | 0.13% |
| foundational text | 5 | 0.13% |
| drug or therapy used for treatment | 5 | 0.13% |
| director of archaeological fieldwork | 5 | 0.13% |
| represents | 5 | 0.13% |
| school district | 5 | 0.13% |
| immediate cause of | 5 | 0.13% |
| fabrication method | 5 | 0.13% |
| executive body | 5 | 0.13% |
| religious order | 5 | 0.13% |
| located in time zone | 5 | 0.13% |
| interested in | 5 | 0.13% |
| territory overlaps | 5 | 0.13% |
| business division | 5 | 0.13% |
| radio format | 5 | 0.13% |
| parent club | 5 | 0.13% |
| subject has role | 4 | 0.11% |
| manufacturer | 4 | 0.11% |
| political ideology | 4 | 0.11% |
| vessel class | 4 | 0.11% |
| successful candidate | 4 | 0.11% |
| victory | 4 | 0.11% |
| possible treatment | 4 | 0.11% |
| indigenous to | 4 | 0.11% |
| said to be the same as | 4 | 0.11% |
| media franchise | 4 | 0.11% |
| Gram staining | 4 | 0.11% |
| official language | 4 | 0.11% |
| lake outflow | 4 | 0.11% |
| mascot | 4 | 0.11% |
| writing system | 4 | 0.11% |
| from narrative universe | 4 | 0.11% |
| copyright license | 4 | 0.11% |
| programming paradigm | 4 | 0.11% |
| studied by | 4 | 0.11% |
| organizer | 4 | 0.11% |
| religion or worldview | 4 | 0.11% |
| next higher rank | 4 | 0.11% |
| final event | 4 | 0.11% |
| substitute/deputy/replacement of office/officeholder | 4 | 0.11% |
| domain of saint or deity | 4 | 0.11% |
| engine configuration | 4 | 0.11% |
| taxon rank | 4 | 0.11% |
| history of topic | 4 | 0.11% |
| wheel arrangement | 4 | 0.11% |
| lyrics by | 4 | 0.11% |
| Comparison | 4 | 0.11% |
| party chief representative | 4 | 0.11% |
| sponsor | 4 | 0.11% |
| chairperson | 4 | 0.11% |
| genetic association | 4 | 0.11% |
| location of creation | 4 | 0.11% |
| sex or gender | 4 | 0.11% |
| drainage basin | 4 | 0.11% |
| readable file format | 4 | 0.11% |
| sibling | 4 | 0.11% |
| subclass of | 4 | 0.11% |
| interchange station | 4 | 0.11% |
| place of death | 4 | 0.11% |
| original language of film or TV show | 4 | 0.11% |
| located in protected area | 4 | 0.11% |
| choreographer | 4 | 0.11% |
| has natural reservoir | 3 | 0.08% |
| instruction set | 3 | 0.08% |
| signatory | 3 | 0.08% |
| significant person | 3 | 0.08% |
| residence | 3 | 0.08% |
| place of birth | 3 | 0.08% |
| game mode | 3 | 0.08% |
| legislative body | 3 | 0.08% |
| location of landing | 3 | 0.08% |
| destination point | 3 | 0.08% |
| coach of sports team | 3 | 0.08% |
| cuisine | 3 | 0.08% |
| day in year for periodic occurrence | 3 | 0.08% |
| underlies | 3 | 0.08% |
| field of this occupation | 3 | 0.08% |
| killed by | 3 | 0.08% |
| encodes | 3 | 0.08% |
| practiced by | 3 | 0.08% |
| depicted by | 3 | 0.08% |
| industry | 3 | 0.08% |
| software engine | 3 | 0.08% |
| linguistic typology | 3 | 0.08% |
| general classification of race participants | 3 | 0.08% |
| structural engineer | 3 | 0.08% |
| color | 3 | 0.08% |
| enemy of | 3 | 0.08% |
| student of | 3 | 0.08% |
| dual to | 3 | 0.08% |
| capital | 3 | 0.08% |
| less than | 3 | 0.08% |
| has list | 3 | 0.08% |
| advertises | 3 | 0.08% |
| copyright holder | 3 | 0.08% |
| minor planet group | 3 | 0.08% |
| sports season of league or competition | 3 | 0.08% |
| league level above | 3 | 0.08% |
| statement describes | 3 | 0.08% |
| contributor to the creative work or subject | 3 | 0.08% |
| allegiance | 3 | 0.08% |
| co-driver | 3 | 0.08% |
| social classification | 3 | 0.08% |
| honorific prefix | 3 | 0.08% |
| medical examination | 3 | 0.08% |
| archives at | 3 | 0.08% |
| patron saint | 3 | 0.08% |
| given name version for other gender | 3 | 0.08% |
| partially coincident with | 3 | 0.08% |
| has boundary | 3 | 0.08% |
| anthem | 3 | 0.08% |
| youth wing | 3 | 0.08% |
| tributary | 3 | 0.08% |
| separated from | 3 | 0.08% |
| ancestral home | 3 | 0.08% |
| official religion | 3 | 0.08% |
| associated electoral district | 3 | 0.08% |
| central bank/issuer | 3 | 0.08% |
| political alignment | 3 | 0.08% |
| main regulatory text | 3 | 0.08% |
| shape | 2 | 0.05% |
| superhuman feature or ability | 2 | 0.05% |
| used by | 2 | 0.05% |
| derivative work | 2 | 0.05% |
| depicts | 2 | 0.05% |
| site of astronomical discovery | 2 | 0.05% |
| tonality | 2 | 0.05% |
| antiparticle | 2 | 0.05% |
| sidekick of | 2 | 0.05% |
| measures | 2 | 0.05% |
| Christian liturgical rite | 2 | 0.05% |
| input method | 2 | 0.05% |
| has active ingredient | 2 | 0.05% |
| based on | 2 | 0.05% |
| place of burial | 2 | 0.05% |
| charted in | 2 | 0.05% |
| programmer | 2 | 0.05% |
| central bank | 2 | 0.05% |
| economy of topic | 2 | 0.05% |
| unmarried partner | 2 | 0.05% |
| family name identical to this given name | 2 | 0.05% |
| GUI toolkit or framework | 2 | 0.05% |
| fastest lap | 2 | 0.05% |
| product certification | 2 | 0.05% |
| league level below | 2 | 0.05% |
| destroyed | 2 | 0.05% |
| different from | 2 | 0.05% |
| instrumentation | 2 | 0.05% |
| typically sells | 2 | 0.05% |
| follows | 2 | 0.05% |
| relative | 2 | 0.05% |
| inspired by | 2 | 0.05% |
| capital of | 2 | 0.05% |
| type of variable star | 2 | 0.05% |
| airline hub | 2 | 0.05% |
| depends on software | 2 | 0.05% |
| house publication | 2 | 0.05% |
| contributing factor of | 2 | 0.05% |
| affiliation | 2 | 0.05% |
| twinned administrative body | 2 | 0.05% |
| designed to carry | 2 | 0.05% |
| has quality | 2 | 0.05% |
| speaker | 2 | 0.05% |
| does not have part | 2 | 0.05% |
| sexually homologous with | 2 | 0.05% |
| copyright representative | 2 | 0.05% |
| sports league level | 2 | 0.05% |
| official residence | 2 | 0.05% |
| character designer | 2 | 0.05% |
| screenwriter | 2 | 0.05% |
| manner of death | 2 | 0.05% |
| designated as terrorist by | 2 | 0.05% |
| gens | 2 | 0.05% |
| manifestation of | 2 | 0.05% |
| chief executive officer | 2 | 0.05% |
| via | 2 | 0.05% |
| student | 2 | 0.05% |
| Drug_Interaction | 2 | 0.05% |
| occupant | 2 | 0.05% |
| country of registry | 2 | 0.05% |
| solved by | 2 | 0.05% |
| amended by | 2 | 0.05% |
| family name | 2 | 0.05% |
| cell component | 2 | 0.05% |
| perpetrator | 2 | 0.05% |
| coat of arms | 2 | 0.05% |
| country of citizenship | 2 | 0.05% |
| subdivision of this unit | 2 | 0.05% |
| flag | 2 | 0.05% |
| title of chess person | 2 | 0.05% |
| honorific suffix | 2 | 0.05% |
| measurement scale | 2 | 0.05% |
| drafted by | 2 | 0.05% |
| parallel street | 2 | 0.05% |
| health specialty | 2 | 0.05% |
| cause of death | 2 | 0.05% |
| voice actor | 2 | 0.05% |
| emergency phone number | 1 | 0.03% |
| referee | 1 | 0.03% |
| delta of | 1 | 0.03% |
| radix | 1 | 0.03% |
| enclave within | 1 | 0.03% |
| in opposition to | 1 | 0.03% |
| lifestyle | 1 | 0.03% |
| appointed by | 1 | 0.03% |
| significant drug interaction | 1 | 0.03% |
| copyright status | 1 | 0.03% |
| active ingredient in | 1 | 0.03% |
| display technology | 1 | 0.03% |
| doctoral advisor | 1 | 0.03% |
| has pet | 1 | 0.03% |
| by-product | 1 | 0.03% |
| wears | 1 | 0.03% |
| top-level Internet domain | 1 | 0.03% |
| afflicts | 1 | 0.03% |
| business model | 1 | 0.03% |
| temporal range start | 1 | 0.03% |
| symbolizes | 1 | 0.03% |
| editor-in-chief | 1 | 0.03% |
| structure replaced by | 1 | 0.03% |
| CPU | 1 | 0.03% |
| stepparent | 1 | 0.03% |
| narrative role | 1 | 0.03% |
| members have occupation | 1 | 0.03% |
| recorded at studio or venue | 1 | 0.03% |
| expressed in | 1 | 0.03% |
| has tense | 1 | 0.03% |
| highest point | 1 | 0.03% |
| exemplar of | 1 | 0.03% |
| mountains classification | 1 | 0.03% |
| epithet | 1 | 0.03% |
| GPU | 1 | 0.03% |
| catalog | 1 | 0.03% |
| conjugate acid | 1 | 0.03% |
| permanent resident of | 1 | 0.03% |
| expansion of | 1 | 0.03% |
| trophy awarded | 1 | 0.03% |
| geography of topic | 1 | 0.03% |
| music created for | 1 | 0.03% |
| launch contractor | 1 | 0.03% |
| candidacy in election | 1 | 0.03% |
| theme music | 1 | 0.03% |
| political coalition | 1 | 0.03% |
| interaction | 1 | 0.03% |
| day of week | 1 | 0.03% |
| disease transmission process | 1 | 0.03% |
| approved by | 1 | 0.03% |
| route of administration | 1 | 0.03% |
| points classification | 1 | 0.03% |
| monomer of | 1 | 0.03% |
| writable file format | 1 | 0.03% |
| mother house | 1 | 0.03% |
| main building contractor | 1 | 0.03% |
| regulated by | 1 | 0.03% |
| statistical leader | 1 | 0.03% |
| comorbidity | 1 | 0.03% |
| wing configuration | 1 | 0.03% |
| office contested | 1 | 0.03% |
| diaspora | 1 | 0.03% |
| source of income | 1 | 0.03% |
| biological variant of | 1 | 0.03% |
| penalty | 1 | 0.03% |
| charge | 1 | 0.03% |
| diplomatic relation | 1 | 0.03% |
| foods traditionally associated | 1 | 0.03% |
| official podcast | 1 | 0.03% |
| prime factor | 1 | 0.03% |
| territory claimed by | 1 | 0.03% |
| influenced by | 1 | 0.03% |
| official symbol | 1 | 0.03% |
| has melody | 1 | 0.03% |
| organization directed by the office or position | 1 | 0.03% |
| season | 1 | 0.03% |
| exhibition history | 1 | 0.03% |
| aspect ratio (W:H) | 1 | 0.03% |
| appears in the form of | 1 | 0.03% |
| proved by | 1 | 0.03% |
| office held by head of government | 1 | 0.03% |
| repeals | 1 | 0.03% |
| demographics of topic | 1 | 0.03% |
| next lower rank | 1 | 0.03% |
| Cotreatment | 1 | 0.03% |
| set during recurring event | 1 | 0.03% |
| offers view on | 1 | 0.03% |
| architectural style | 1 | 0.03% |
| executive producer | 1 | 0.03% |
| mount | 1 | 0.03% |
| substrate of | 1 | 0.03% |
| decays to | 1 | 0.03% |
| doctoral student | 1 | 0.03% |
| academic degree | 1 | 0.03% |
| named by | 1 | 0.03% |
| output method | 1 | 0.03% |
| military casualty classification | 1 | 0.03% |
| facial hair | 1 | 0.03% |
| prohibits | 1 | 0.03% |
| prerequisite | 1 | 0.03% |
| character type | 1 | 0.03% |
| officially opened by | 1 | 0.03% |
| sexual orientation | 1 | 0.03% |
| mod of | 1 | 0.03% |
| type of orbit | 1 | 0.03% |

### Classifications

**train, biotech_event.** Majority class 29.7% of 350 answers. The marginal is `true_label`, not the menu.

#### train biotech_event

| label | count | share |
| --- | --- | --- |
| executive statement | 104 | 29.71% |
| company description | 50 | 14.29% |
| alliance & partnership | 27 | 7.71% |
| product launching & presentation | 22 | 6.29% |
| other | 19 | 5.43% |
| new initiatives or programs | 16 | 4.57% |
| funding round | 14 | 4.00% |
| hiring | 14 | 4.00% |
| m&a | 11 | 3.14% |
| support & philanthropy | 10 | 2.86% |
| service & product providing | 9 | 2.57% |
| expanding geography | 8 | 2.29% |
| article publication | 8 | 2.29% |
| executive appointment | 7 | 2.00% |
| regulatory approval | 6 | 1.71% |
| product updates | 5 | 1.43% |
| participation in an event | 4 | 1.14% |
| event organization | 4 | 1.14% |
| new initiatives & programs | 3 | 0.86% |
| closing | 2 | 0.57% |
| department establishment | 2 | 0.57% |
| clinical trial sponsorship | 2 | 0.57% |
| foundation | 1 | 0.29% |
| expanding industry | 1 | 0.29% |
| investment in public company | 1 | 0.29% |

**train, chfinann_event.** Majority class 49.3% of 2,238 answers. The marginal is `true_label`, not the menu.

#### train chfinann_event

| label | count | share |
| --- | --- | --- |
| EquityPledge | 1,103 | 49.29% |
| EquityOverweight | 470 | 21.00% |
| EquityUnderweight | 443 | 19.79% |
| EquityRepurchase | 144 | 6.43% |
| EquityFreeze | 78 | 3.49% |

**train, docee_event.** Majority class 7.6% of 1,886 answers. The marginal is `true_label`, not the menu.

#### train docee_event

| label | count | share |
| --- | --- | --- |
| Armed Conflict | 144 | 7.64% |
| Air Crash | 116 | 6.15% |
| Protest | 89 | 4.72% |
| Election | 89 | 4.72% |
| Sports Competition | 67 | 3.55% |
| CommitCrime - Sentence | 64 | 3.39% |
| Government Policy Changes | 63 | 3.34% |
| Fire | 60 | 3.18% |
| Riot | 59 | 3.13% |
| CommitCrime - Accuse | 59 | 3.13% |
| Disease Outbreaks | 44 | 2.33% |
| Earthquakes | 44 | 2.33% |
| Resignation_Dismissal | 37 | 1.96% |
| Appoint_Inauguration | 37 | 1.96% |
| Storm | 36 | 1.91% |
| Withdraw from an Orgnization | 35 | 1.86% |
| Famous Person - Death | 33 | 1.75% |
| Break Historical Records | 32 | 1.70% |
| Floods | 31 | 1.64% |
| Strike | 30 | 1.59% |
| Environment Pollution | 30 | 1.59% |
| Gas Explosion | 29 | 1.54% |
| Diplomatic Talks | 29 | 1.54% |
| CommitCrime - Investigate | 27 | 1.43% |
| Mass Poisoning | 27 | 1.43% |
| Military Excercise | 26 | 1.38% |
| New Achievements in Aerospace | 25 | 1.33% |
| Famous Person - Sick | 25 | 1.33% |
| Famous Person - Give a Speech | 23 | 1.22% |
| Road Crash | 23 | 1.22% |
| Train Collisions | 23 | 1.22% |
| Financial Crisis | 22 | 1.17% |
| CommitCrime - Arrest | 22 | 1.17% |
| Diplomatic Visit | 21 | 1.11% |
| Awards Ceremony | 21 | 1.11% |
| Famine | 21 | 1.11% |
| New Archeological Discoveries | 21 | 1.11% |
| Bank Robbery | 20 | 1.06% |
| Mine Collapses | 20 | 1.06% |
| New Wonders in Nature | 20 | 1.06% |
| Shipwreck | 19 | 1.01% |
| Organization Merge | 19 | 1.01% |
| CommitCrime - Release | 19 | 1.01% |
| Insect Disaster | 18 | 0.95% |
| Sign Agreement | 18 | 0.95% |
| Famous Person - Divorce | 17 | 0.90% |
| Organization Fine | 14 | 0.74% |
| Tear Up Agreement | 14 | 0.74% |
| Organization Closed | 13 | 0.69% |
| Volcano Eruption | 13 | 0.69% |
| Famous Person - Marriage | 12 | 0.64% |
| Tsunamis | 11 | 0.58% |
| Mudslides | 11 | 0.58% |
| Financial Aid | 10 | 0.53% |
| Droughts | 10 | 0.53% |
| Organization Established | 9 | 0.48% |
| Famous Person - Recovered | 7 | 0.37% |
| Regime Change | 6 | 0.32% |
| Join in an Orgnization | 2 | 0.11% |

**train, docfee_event.** Majority class 35.2% of 1,450 answers. The marginal is `true_label`, not the menu.

#### train docfee_event

| label | count | share |
| --- | --- | --- |
| EquityUnderweight | 511 | 35.24% |
| EquityPledge | 228 | 15.72% |
| BankruptcyLiquidation | 226 | 15.59% |
| MajorAssetLoss | 183 | 12.62% |
| EquityFreeze | 97 | 6.69% |
| EquityOverweight | 75 | 5.17% |
| MajorSafetyAccident | 55 | 3.79% |
| ExecutiveDeath | 39 | 2.69% |
| MajorExternalPayout | 36 | 2.48% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| casualty_report | injured | 23,784 | 55.20% |
| casualty_report | dead | 21,276 | 49.38% |
| casualty_report | location | 17,479 | 40.56% |
| casualty_report | missing | 9,515 | 22.08% |
| casualty_report | record_metadata | natural/dead ×16,870, natural/injured ×8,677, natural/missing ×1,278, natural/location ×866 |  |
| record | status | 85 | 14.60% |
| record | location | 58 | 9.97% |
| record | address | 33 | 5.67% |
| record | name | 33 | 5.67% |
| record | title | 20 | 3.44% |
| record | description | 19 | 3.26% |
| record | price | 17 | 2.92% |
| record | amount | 17 | 2.92% |
| record | city | 15 | 2.58% |
| record | rating | 14 | 2.41% |
| record | author | 14 | 2.41% |
| record | patient_id | 11 | 1.89% |
| record | claim_id | 10 | 1.72% |
| record | state | 10 | 1.72% |
| record | date | 10 | 1.72% |
| record | company | 9 | 1.55% |
| record | review_id | 8 | 1.37% |
| record | patient_name | 8 | 1.37% |
| record | transaction_id | 8 | 1.37% |
| record | department | 8 | 1.37% |
| record | product_name | 7 | 1.20% |
| record | venue | 7 | 1.20% |
| record | event_id | 7 | 1.20% |
| record | employee_id | 7 | 1.20% |
| record | product_id | 7 | 1.20% |
| record | country | 7 | 1.20% |
| record | year | 7 | 1.20% |
| record | brand | 6 | 1.03% |
| record | funding_amount | 6 | 1.03% |
| record | company_name | 6 | 1.03% |
| record | dosage | 6 | 1.03% |
| record | tail | 6 | 1.03% |
| record | head | 6 | 1.03% |
| record | sender | 6 | 1.03% |
| record | age | 6 | 1.03% |
| record | project_id | 6 | 1.03% |
| record | manufacturer | 5 | 0.86% |
| record | medication | 5 | 0.86% |
| record | report_id | 5 | 0.86% |
| record | score | 5 | 0.86% |
| record | home_team | 5 | 0.86% |
| record | notes | 5 | 0.86% |
| record | policy_id | 5 | 0.86% |
| record | provider | 5 | 0.86% |
| record | funding_source | 5 | 0.86% |
| record | policyholder | 5 | 0.86% |
| record | likes | 4 | 0.69% |
| record | product | 4 | 0.69% |
| record | patient_address | 4 | 0.69% |
| record | diagnosis_code | 4 | 0.69% |
| record | mrn | 4 | 0.69% |
| record | severity | 4 | 0.69% |
| record | ticket_id | 4 | 0.69% |
| record | category | 4 | 0.69% |
| record | review_status | 4 | 0.69% |
| record | authors | 4 | 0.69% |
| record | discovery.observatory.location.country | 4 | 0.69% |
| record | duration | 4 | 0.69% |
| record | source | 4 | 0.69% |
| record | region | 4 | 0.69% |
| record | tags | 3 | 0.52% |
| record | room | 3 | 0.52% |
| record | sku | 3 | 0.52% |
| record | film_title | 3 | 0.52% |
| record | word_count | 3 | 0.52% |
| record | review_date | 3 | 0.52% |
| record | return_id | 3 | 0.52% |
| record | reason | 3 | 0.52% |
| record | role | 3 | 0.52% |
| record | full_name | 3 | 0.52% |
| record | quantity | 3 | 0.52% |
| record | recipient_name | 3 | 0.52% |
| record | term_length | 3 | 0.52% |
| record | symptoms | 3 | 0.52% |
| record | policy_number | 3 | 0.52% |
| record | date: Date when the transaction occurred | 3 | 0.52% |
| record | heart_rate: Heart rate in beats per minute | 3 | 0.52% |
| record | temperature: Body temperature in degrees Fahrenheit | 3 | 0.52% |
| record | email | 3 | 0.52% |
| record | opponent | 3 | 0.52% |
| record | points | 3 | 0.52% |
| record | battery_life | 3 | 0.52% |
| record | content | 3 | 0.52% |
| record | recipient | 3 | 0.52% |
| record | submitter | 3 | 0.52% |
| record | project_name | 3 | 0.52% |
| record | match | 3 | 0.52% |
| record | amenities | 3 | 0.52% |
| record | organization | 3 | 0.52% |
| record | person | 3 | 0.52% |
| record | job_title | 3 | 0.52% |
| record | salary_range | 3 | 0.52% |
| record | entity_name | 3 | 0.52% |
| record | features | 3 | 0.52% |
| record | discovery.observatory.name | 3 | 0.52% |
| record | reviewer_name | 3 | 0.52% |
| record | claim_id: Unique identifier for the insurance claim | 3 | 0.52% |
| record | revenue | 3 | 0.52% |
| record | target | 3 | 0.52% |
| record | origin | 3 | 0.52% |
| record | funding | 3 | 0.52% |
| record | customer_name | 3 | 0.52% |
| record | subject | 3 | 0.52% |
| record | user | 3 | 0.52% |
| record | asteroid.name | 3 | 0.52% |
| record | player_id | 3 | 0.52% |
| record | player_name | 3 | 0.52% |
| record | keywords | 3 | 0.52% |
| record | orbital_characteristics.semi_major_axis.unit | 3 | 0.52% |
| record | destination | 3 | 0.52% |
| record | delay_reason | 3 | 0.52% |
| record | delay_minutes | 3 | 0.52% |
| record | comments | 2 | 0.34% |
| record | cpt_code | 2 | 0.34% |
| record | blood_pressure | 2 | 0.34% |
| record | expression_level | 2 | 0.34% |
| record | launch_date | 2 | 0.34% |
| record | refund_amount | 2 | 0.34% |
| record | education | 2 | 0.34% |
| record | newsletter_name | 2 | 0.34% |
| record | investor | 2 | 0.34% |
| record | return_id: Unique identifier for the return transaction | 2 | 0.34% |
| record | user_id | 2 | 0.34% |
| record | policyholder_name | 2 | 0.34% |
| record | net_profit | 2 | 0.34% |
| record | compliance | 2 | 0.34% |
| record | payee | 2 | 0.34% |
| record | product.features | 2 | 0.34% |
| record | issue_description | 2 | 0.34% |
| record | dateofservice | 2 | 0.34% |
| record | institution | 2 | 0.34% |
| record | final_score | 2 | 0.34% |
| record | acquirer | 2 | 0.34% |
| record | employees | 2 | 0.34% |
| record | integration_plan | 2 | 0.34% |
| record | issue_date | 2 | 0.34% |
| record | monthly_premium | 2 | 0.34% |
| record | insurance | 2 | 0.34% |
| record | treatment | 2 | 0.34% |
| record | transaction_id: Unique identifier for the transaction entry | 2 | 0.34% |
| record | reporter | 2 | 0.34% |
| record | hire_date | 2 | 0.34% |
| record | status: Current status of the appointment | 2 | 0.34% |
| record | case_id | 2 | 0.34% |
| record | shares | 2 | 0.34% |
| record | player | 2 | 0.34% |
| record | primary_diagnosis: Main medical condition diagnosed | 2 | 0.34% |
| record | current_status | 2 | 0.34% |
| record | color | 2 | 0.34% |
| record | release_date | 2 | 0.34% |
| record | vehicle | 2 | 0.34% |
| record | user_location | 2 | 0.34% |
| record | findings | 2 | 0.34% |
| record | products | 2 | 0.34% |
| record | clause_id | 2 | 0.34% |
| record | affiliation | 2 | 0.34% |
| record | locations | 2 | 0.34% |
| record | concepts | 2 | 0.34% |
| record | location.region | 2 | 0.34% |
| record | location.country | 2 | 0.34% |
| record | funding_amount: Total amount raised in millions | 2 | 0.34% |
| record | filing_date | 2 | 0.34% |
| record | serial_number | 2 | 0.34% |
| record | device_name | 2 | 0.34% |
| record | key_concepts | 2 | 0.34% |
| record | seller | 2 | 0.34% |
| record | company.name | 2 | 0.34% |
| record | orbital_characteristics.classification | 2 | 0.34% |
| record | effective_date | 2 | 0.34% |
| record | normal_range | 2 | 0.34% |
| record | value | 2 | 0.34% |
| record | coverage_amount | 2 | 0.34% |
| record | beneficiary | 2 | 0.34% |
| record | municipality.name | 2 | 0.34% |
| record | university | 2 | 0.34% |
| record | reviewer | 2 | 0.34% |
| record | team | 2 | 0.34% |
| record | model_number | 2 | 0.34% |
| record | target_audience | 2 | 0.34% |
| record | user_feedback | 2 | 0.34% |
| record | device | 2 | 0.34% |
| record | time | 2 | 0.34% |
| record | flight_number | 2 | 0.34% |
| record | passenger_name | 2 | 0.34% |
| record | course_id | 2 | 0.34% |
| record | claim_id: Unique identifier for each claim entry | 2 | 0.34% |
| record | discovery_date | 2 | 0.34% |
| record | status: Current operational status of the device | 2 | 0.34% |
| record | contact_name | 2 | 0.34% |
| record | founding_year | 2 | 0.34% |
| record | reference_id | 2 | 0.34% |
| record | chemical_name | 2 | 0.34% |
| record | location: City and state where the user is based | 2 | 0.34% |
| record | asteroid.designation | 2 | 0.34% |
| record | asteroid.discovery.date | 2 | 0.34% |
| record | filer | 2 | 0.34% |
| record | appointment_id | 2 | 0.34% |
| record | asteroid.type | 2 | 0.34% |
| record | patient_id: Unique identifier for the patient | 2 | 0.34% |
| record | name: Full name of the patient | 2 | 0.34% |
| record | credits | 2 | 0.34% |
| record | owner | 2 | 0.34% |
| record | department: Academic department offering the course | 2 | 0.34% |
| record | instructor: Name of the faculty member teaching the course | 2 | 0.34% |
| record | insurance_provider | 2 | 0.34% |
| record | transfer_id | 2 | 0.34% |
| record | bathrooms | 2 | 0.34% |
| record | bedrooms | 2 | 0.34% |
| record | enrollment | 2 | 0.34% |
| record | claim_status | 2 | 0.34% |
| record | insurance_plan | 2 | 0.34% |
| record | medications | 2 | 0.34% |
| record | book_title | 2 | 0.34% |
| record | patent_number | 2 | 0.34% |
| record | id | 2 | 0.34% |
| record | enrollment_date | 2 | 0.34% |
| record | risk_score | 2 | 0.34% |
| record | indication | 2 | 0.34% |
| record | ndc | 2 | 0.34% |
| record | event | 2 | 0.34% |
| record | zip_code | 2 | 0.34% |
| record | designation | 2 | 0.34% |
| record | discovery.discoverer | 2 | 0.34% |
| record | discovery.observatory.location.prefecture | 2 | 0.34% |
| record | orbital_characteristics.perihelion.unit | 2 | 0.34% |
| record | event_type: Category of the event | 2 | 0.34% |
| record | property_type | 2 | 0.34% |
| record | asteroid.location | 2 | 0.34% |
| record | diagnosis | 2 | 0.34% |
| record | timestamp | 2 | 0.34% |
| record | away_team | 2 | 0.34% |
| record | entity | 2 | 0.34% |
| record | physical_characteristics.observations.0.telescope.name | 2 | 0.34% |
| record | physical_characteristics.observations.0.publication_year | 2 | 0.34% |
| record | home_team: Name of the team playing at home | 2 | 0.34% |
| record | patient_name: Name of the patient filing the claim | 2 | 0.34% |
| record | registration_fee | 2 | 0.34% |
| record | sample_size | 2 | 0.34% |
| record | abstract_id | 2 | 0.34% |
| record | organizations | 2 | 0.34% |
| record | gender | 2 | 0.34% |
| record | icd10code | 2 | 0.34% |
| record | material | 2 | 0.34% |
| record | activities | 2 | 0.34% |
| record | copyright | 2 | 0.34% |
| record | flight_id | 2 | 0.34% |
| record | sales_trend | 2 | 0.34% |
| record | resolution_time | 2 | 0.34% |
| record | thumbnail_style | 1 | 0.17% |
| record | video_id | 1 | 0.17% |
| record | channel_name | 1 | 0.17% |
| record | description_length | 1 | 0.17% |
| record | article.title | 1 | 0.17% |
| record | article.publication_details.source | 1 | 0.17% |
| record | article.publication_details.author | 1 | 0.17% |
| record | article.publication_details.publication_date | 1 | 0.17% |
| record | article.summary | 1 | 0.17% |
| record | article.entities.organizations | 1 | 0.17% |
| record | article.entities.people.0.organization | 1 | 0.17% |
| record | article.entities.people.0.title | 1 | 0.17% |
| record | article.entities.people.0.name | 1 | 0.17% |
| record | article.entities.people.1.organization | 1 | 0.17% |
| record | article.entities.people.1.title | 1 | 0.17% |
| record | article.entities.people.1.name | 1 | 0.17% |
| record | article.copyright_details.copyright_request_period | 1 | 0.17% |
| record | article.copyright_details.copyright_transparency_report_url | 1 | 0.17% |
| record | features: List of key technological or functional capabilities | 1 | 0.17% |
| record | price: Base price of the product in US dollars | 1 | 0.17% |
| record | regions: Geographic areas where the product is distributed or in use | 1 | 0.17% |
| record | name: Name of the product as introduced | 1 | 0.17% |
| record | distributor_information.capabilities | 1 | 0.17% |
| record | services.maintenance | 1 | 0.17% |
| record | distributor_information.benefits | 1 | 0.17% |
| record | cpt_code: CPT code for the medical procedure performed | 1 | 0.17% |
| record | paid_amount: Amount actually received from insurance | 1 | 0.17% |
| record | diagnosis: ICD-10 code representing the medical condition | 1 | 0.17% |
| record | billed_amount: Total amount billed for the service | 1 | 0.17% |
| record | product.specifications.display.resolution | 1 | 0.17% |
| record | product.launch_date | 1 | 0.17% |
| record | product.registration_count.count | 1 | 0.17% |
| record | product.specifications.memory.internal_storage | 1 | 0.17% |
| record | product.availability.currently_available | 1 | 0.17% |
| record | product.specifications.connectivity | 1 | 0.17% |
| record | product.price.1.currency | 1 | 0.17% |
| record | product.brand | 1 | 0.17% |
| record | product.specifications.camera.features | 1 | 0.17% |
| record | product.specifications.display.size | 1 | 0.17% |
| record | product.launch_location | 1 | 0.17% |
| record | product.model | 1 | 0.17% |
| record | product.registration_count.time_period | 1 | 0.17% |
| record | product.specifications.camera.rear | 1 | 0.17% |
| record | product.specifications.battery.capacity | 1 | 0.17% |
| record | product.specifications.memory.ram | 1 | 0.17% |
| record | claim_amount | 1 | 0.17% |
| record | heart_rate | 1 | 0.17% |
| record | disease_context | 1 | 0.17% |
| record | target_market | 1 | 0.17% |
| record | sales_volume | 1 | 0.17% |
| record | customer | 1 | 0.17% |
| record | product_model | 1 | 0.17% |
| record | order_id | 1 | 0.17% |
| record | return_method | 1 | 0.17% |
| record | tracking_number | 1 | 0.17% |
| record | budget | 1 | 0.17% |
| record | agency | 1 | 0.17% |
| record | technology | 1 | 0.17% |
| record | prescription_id | 1 | 0.17% |
| record | claims_count | 1 | 0.17% |
| record | filing_location | 1 | 0.17% |
| record | prior_experience | 1 | 0.17% |
| record | delivery_method | 1 | 0.17% |
| record | confirmation_code | 1 | 0.17% |
| record | subscription_amount | 1 | 0.17% |
| record | investor_leads | 1 | 0.17% |
| record | funds_use | 1 | 0.17% |
| record | location-gpe | 1 | 0.17% |
| record | planning | 1 | 0.17% |
| record | startup | 1 | 0.17% |
| record | funding_round | 1 | 0.17% |
| record | release_target | 1 | 0.17% |
| record | underwriting_model | 1 | 0.17% |
| record | supply_duration | 1 | 0.17% |
| record | returner: Name of the person initiating the return | 1 | 0.17% |
| record | location: City or address where return was submitted | 1 | 0.17% |
| record | item: Description of the returned product | 1 | 0.17% |
| record | amount: Monetary value of the return in dollars | 1 | 0.17% |
| record | status: Current status of the return approval | 1 | 0.17% |
| record | people | 1 | 0.17% |
| record | path | 1 | 0.17% |
| record | substances | 1 | 0.17% |
| record | materials | 1 | 0.17% |
| record | door | 1 | 0.17% |
| record | metaphors | 1 | 0.17% |
| record | objects | 1 | 0.17% |
| record | supernatural | 1 | 0.17% |
| record | decor | 1 | 0.17% |
| record | food | 1 | 0.17% |
| record | label | 1 | 0.17% |
| record | surface | 1 | 0.17% |
| record | entries | 1 | 0.17% |
| record | tournament_name | 1 | 0.17% |
| record | winner_ranking | 1 | 0.17% |
| record | opponent_name | 1 | 0.17% |
| record | headlines | 1 | 0.17% |
| record | development_periods | 1 | 0.17% |
| record | see_also | 1 | 0.17% |
| record | stub_article.call_to_action | 1 | 0.17% |
| record | closing_price | 1 | 0.17% |
| record | dateofincident | 1 | 0.17% |
| record | region_churn | 1 | 0.17% |
| record | call_id | 1 | 0.17% |
| record | recipient: Name of the recipient professional contacted | 1 | 0.17% |
| record | message_length: Word count of the outreach message | 1 | 0.17% |
| record | subject: Subject line of the outreach email | 1 | 0.17% |
| record | email_id: Unique identifier for the outreach email | 1 | 0.17% |
| record | sender: Name of the recruiter initiating the outreach | 1 | 0.17% |
| record | name: Product name as listed in inventory records | 1 | 0.17% |
| record | last_updated: Date when inventory was last modified | 1 | 0.17% |
| record | sku_id: Unique identifier for the product in inventory system | 1 | 0.17% |
| record | quantity: Current stock level of the SKU | 1 | 0.17% |
| record | location: Warehouse or hub where the SKU is stored | 1 | 0.17% |
| record | location.city | 1 | 0.17% |
| record | hotel.name | 1 | 0.17% |
| record | hotel.star_rating | 1 | 0.17% |
| record | purpose | 1 | 0.17% |
| record | account | 1 | 0.17% |
| record | product.vendor | 1 | 0.17% |
| record | product.type | 1 | 0.17% |
| record | product.version | 1 | 0.17% |
| record | product.description | 1 | 0.17% |
| record | insurance_id | 1 | 0.17% |
| record | amount_paid | 1 | 0.17% |
| record | founder: Name of the individual who founded the company | 1 | 0.17% |
| record | funding_source: Type of funding received by the company | 1 | 0.17% |
| record | current_status: Current operational phase of the company | 1 | 0.17% |
| record | clientbasesize: Number of clients served across regions | 1 | 0.17% |
| record | business_focus: Core service or product area of the company | 1 | 0.17% |
| record | position | 1 | 0.17% |
| record | policyholder_id: Unique identifier for the policyholder | 1 | 0.17% |
| record | policyholder_name: Full name of the claimant | 1 | 0.17% |
| record | claim_id: Unique identifier for the submitted claim | 1 | 0.17% |
| record | award_name | 1 | 0.17% |
| record | fieldofstudy | 1 | 0.17% |
| record | description: Brief text describing the content of the post | 1 | 0.17% |
| record | status: Current status of the trend repost | 1 | 0.17% |
| record | user_name: Name of the user who created the repost | 1 | 0.17% |
| record | trend_name: Name of the viral trend being reposted | 1 | 0.17% |
| record | post_date: Date when the trend repost was published | 1 | 0.17% |
| record | shares: Number of shares the post received | 1 | 0.17% |
| record | offer_id: Unique identifier for each leasing offer | 1 | 0.17% |
| record | property_address: Full address of the leased property | 1 | 0.17% |
| record | tenant_name: Name of the leasing company or business | 1 | 0.17% |
| record | special_features: List of unique space features or amenities | 1 | 0.17% |
| record | referee | 1 | 0.17% |
| record | client_name | 1 | 0.17% |
| record | client_address | 1 | 0.17% |
| record | rate | 1 | 0.17% |
| record | acquired | 1 | 0.17% |
| record | value_original | 1 | 0.17% |
| record | regulatory_status | 1 | 0.17% |
| record | patents | 1 | 0.17% |
| record | customer_contracts | 1 | 0.17% |
| record | platform_name | 1 | 0.17% |
| record | earnout | 1 | 0.17% |
| record | sustainability_commitment | 1 | 0.17% |
| record | product_code | 1 | 0.17% |
| record | pricing | 1 | 0.17% |
| record | policytype | 1 | 0.17% |
| record | visit_type | 1 | 0.17% |
| record | payee: Name of the entity receiving payment | 1 | 0.17% |
| record | description: Brief description of the transaction purpose | 1 | 0.17% |
| record | amount: Monetary value of the transaction | 1 | 0.17% |
| record | document.location | 1 | 0.17% |
| record | document.counterfeiting_statistics.job_losses | 1 | 0.17% |
| record | document.strategy_elements.3.area | 1 | 0.17% |
| record | document.commissioners.1.portfolio | 1 | 0.17% |
| record | document.strategy_elements.6.area | 1 | 0.17% |
| record | document.key_issues | 1 | 0.17% |
| record | document.strategy_elements.0.area | 1 | 0.17% |
| record | document.commissioners.1.name | 1 | 0.17% |
| record | document.commissioners.0.name | 1 | 0.17% |
| record | document.strategy_elements.4.area | 1 | 0.17% |
| record | document.title | 1 | 0.17% |
| record | document.related_documents | 1 | 0.17% |
| record | document.reference_id | 1 | 0.17% |
| record | document.economic_impact.brand_value | 1 | 0.17% |
| record | document.strategy_elements.1.area | 1 | 0.17% |
| record | match_type | 1 | 0.17% |
| record | genus.period | 1 | 0.17% |
| record | genus.naming.year | 1 | 0.17% |
| record | genus.type_species | 1 | 0.17% |
| record | genus.naming.profession | 1 | 0.17% |
| record | genus.unique_features | 1 | 0.17% |
| record | genus.species.0.reason | 1 | 0.17% |
| record | genus.language | 1 | 0.17% |
| record | genus.naming.named_by | 1 | 0.17% |
| record | ticker | 1 | 0.17% |
| record | exchange | 1 | 0.17% |
| record | asteroid.orbital_characteristics.reference_plane | 1 | 0.17% |
| record | asteroid.designation_number | 1 | 0.17% |
| record | payee: Name of the recipient or entity involved | 1 | 0.17% |
| record | accuracy | 1 | 0.17% |
| record | product_offered | 1 | 0.17% |
| record | hba1c: Hemoglobin A1c percentage value | 1 | 0.17% |
| record | address: Street address including number and street name | 1 | 0.17% |
| record | cholesterol: Cholesterol level in mg/dL if elevated | 1 | 0.17% |
| record | visit_date: Date of last medical check-up | 1 | 0.17% |
| record | state: State abbreviation of patient's residence | 1 | 0.17% |
| record | diagnosis: Medical condition such as hypertension or diabetes | 1 | 0.17% |
| record | age: Age in years as stated in the record | 1 | 0.17% |
| record | zip_code: Postal code of patient's address | 1 | 0.17% |
| record | status: Updated clinical status like stable, unstable, critical, or resolved | 1 | 0.17% |
| record | patient_id: Unique identifier for the monitored patient | 1 | 0.17% |
| record | clinical_status: Current clinical condition such as normal or elevated | 1 | 0.17% |
| record | oxygen_saturation: Percentage of oxygen saturation in blood | 1 | 0.17% |
| record | credits.graphic_design.0.copyright_year | 1 | 0.17% |
| record | credits.contributors.0.person | 1 | 0.17% |
| record | credits.contributors.0.role | 1 | 0.17% |
| record | credits.organization | 1 | 0.17% |
| record | credits.content_implementers | 1 | 0.17% |
| record | credits.contributors.1.role | 1 | 0.17% |
| record | credits.graphic_design.0.designer | 1 | 0.17% |
| record | credits.graphic_design.0.area | 1 | 0.17% |
| record | credits.advertising_revenue_use | 1 | 0.17% |
| record | credits.graphic_design.1.copyright_year | 1 | 0.17% |
| record | credits.creation_date | 1 | 0.17% |
| record | context.region | 1 | 0.17% |
| record | context.demographic_challenge | 1 | 0.17% |
| record | context.strategic_approach | 1 | 0.17% |
| record | lead_agency.name | 1 | 0.17% |
| record | lead_agency.abbreviation | 1 | 0.17% |
| record | lead_agency.responsibilities | 1 | 0.17% |
| record | lead_agency.primary_goal | 1 | 0.17% |
| record | project.name | 1 | 0.17% |
| record | project.abbreviation | 1 | 0.17% |
| record | project.financials.budget | 1 | 0.17% |
| record | project.financials.funding_source | 1 | 0.17% |
| record | project.timeline.duration_years | 1 | 0.17% |
| record | project.scope.setting_type | 1 | 0.17% |
| record | project.objectives | 1 | 0.17% |
| record | project.operational_model.key_roles | 1 | 0.17% |
| record | project.operational_model.stakeholders.0.entity_type | 1 | 0.17% |
| record | project.operational_model.stakeholders.0.description | 1 | 0.17% |
| record | project.operational_model.stakeholders.1.entity_type | 1 | 0.17% |
| record | project.operational_model.stakeholders.1.description | 1 | 0.17% |
| record | project.operational_model.stakeholders.2.entity_type | 1 | 0.17% |
| record | project.operational_model.stakeholders.2.description | 1 | 0.17% |
| record | outcomes.monitored_metrics | 1 | 0.17% |
| record | outcomes.preliminary_findings | 1 | 0.17% |
| record | outcomes.qualitative_impact | 1 | 0.17% |
| record | place.type | 1 | 0.17% |
| record | place.location.province | 1 | 0.17% |
| record | place.demographics.0.year | 1 | 0.17% |
| record | place.demographics.0.population | 1 | 0.17% |
| record | place.location.department | 1 | 0.17% |
| record | place.name | 1 | 0.17% |
| record | salary | 1 | 0.17% |
| record | position_code: Unique identifier for the appointment | 1 | 0.17% |
| record | department: Government department or agency where the appointment occurs | 1 | 0.17% |
| record | judge | 1 | 0.17% |
| record | case_type | 1 | 0.17% |
| record | plaintiff | 1 | 0.17% |
| record | defendant | 1 | 0.17% |
| record | creator | 1 | 0.17% |
| record | operation_hours | 1 | 0.17% |
| record | contribution_id: Unique identifier for each contribution entry | 1 | 0.17% |
| record | address: Physical address of the donor as recorded | 1 | 0.17% |
| record | status: Current status of the contribution record | 1 | 0.17% |
| record | campaign_target: Name of the candidate or office being supported | 1 | 0.17% |
| record | claim_id: Unique identifier for the claim as referenced in correspondence | 1 | 0.17% |
| record | source_office: Clinic or office that submitted the claim | 1 | 0.17% |
| record | procedure_code: CPT code describing the medical service performed | 1 | 0.17% |
| record | allergies: List of known allergies with reactions | 1 | 0.17% |
| record | currentmedicationlist: List of active medications prescribed | 1 | 0.17% |
| record | comorbidities: List of additional health conditions present | 1 | 0.17% |
| record | notes: Additional clinical observations or patient-reported symptoms | 1 | 0.17% |
| record | current_status: Current clinical status of the patient's condition | 1 | 0.17% |
| record | diagnosis_date: Date when primary diagnosis was established | 1 | 0.17% |
| record | full_name: Complete name of the patient as recorded | 1 | 0.17% |
| record | insurance_provider: Name of the patient's health insurance provider | 1 | 0.17% |
| record | dateofbirth: Patient's birth date in YYYY-MM-DD format | 1 | 0.17% |
| record | buyer_name | 1 | 0.17% |
| record | store_reference | 1 | 0.17% |
| record | sound_quality | 1 | 0.17% |
| record | usability_score | 1 | 0.17% |
| record | audiofidelityscore | 1 | 0.17% |
| record | prior_hospitalization | 1 | 0.17% |
| record | primary_diagnosis | 1 | 0.17% |
| record | icd_code | 1 | 0.17% |
| record | treatment_history | 1 | 0.17% |
| record | role: Title of the job position being advertised | 1 | 0.17% |
| record | location: City and country where the role is based | 1 | 0.17% |
| record | company: Name of the company issuing the job posting | 1 | 0.17% |
| record | injury_count | 1 | 0.17% |
| record | transcription_factor | 1 | 0.17% |
| record | gene_id | 1 | 0.17% |
| record | tissue | 1 | 0.17% |
| record | network_id | 1 | 0.17% |
| record | expression_units | 1 | 0.17% |
| record | network_context | 1 | 0.17% |
| record | consultation.practice_areas | 1 | 0.17% |
| record | practice_areas | 1 | 0.17% |
| record | states_served | 1 | 0.17% |
| record | services | 1 | 0.17% |
| record | firm_name | 1 | 0.17% |
| record | client_focus | 1 | 0.17% |
| record | additional_services | 1 | 0.17% |
| record | experience.years | 1 | 0.17% |
| record | experience.description | 1 | 0.17% |
| record | building | 1 | 0.17% |
| record | property | 1 | 0.17% |
| record | lease_term | 1 | 0.17% |
| record | currency_pair: Pair of currencies represented in the rate | 1 | 0.17% |
| record | id: Unique identifier for each exchange rate record | 1 | 0.17% |
| record | description: Brief explanation of market event influencing the rate | 1 | 0.17% |
| record | disbursement_plan | 1 | 0.17% |
| record | gpa | 1 | 0.17% |
| record | platform | 1 | 0.17% |
| record | handle | 1 | 0.17% |
| record | retweets | 1 | 0.17% |
| record | training_modules | 1 | 0.17% |
| record | person.birth_location.country | 1 | 0.17% |
| record | person.birth_location.country_extended | 1 | 0.17% |
| record | person.death_date | 1 | 0.17% |
| record | person.death_location.region | 1 | 0.17% |
| record | person.occupation | 1 | 0.17% |
| record | person.name | 1 | 0.17% |
| record | waterway | 1 | 0.17% |
| record | nationality | 1 | 0.17% |
| record | military unit | 1 | 0.17% |
| record | island | 1 | 0.17% |
| record | tech_product | 1 | 0.17% |
| record | tweet_id | 1 | 0.17% |
| record | author_handle | 1 | 0.17% |
| record | address: Patient's residential address including street, unit, and city | 1 | 0.17% |
| record | status: Current state of prescription fulfillment | 1 | 0.17% |
| record | pharmacy_name: Name of the pharmacy that processed the prescription | 1 | 0.17% |
| record | quantity: Number of tablets or units dispensed | 1 | 0.17% |
| record | dosage: Specific amount or form of medication prescribed | 1 | 0.17% |
| record | categories | 1 | 0.17% |
| record | agent | 1 | 0.17% |
| record | expected_close_date | 1 | 0.17% |
| record | user_email | 1 | 0.17% |
| record | user_name | 1 | 0.17% |
| record | initial_symptoms | 1 | 0.17% |
| record | followupinterval | 1 | 0.17% |
| record | procedure_code | 1 | 0.17% |
| record | imaging_date | 1 | 0.17% |
| record | followupdate | 1 | 0.17% |
| record | trademark_name: Name of the registered trademark in dispute | 1 | 0.17% |
| record | registrant: Company that owns the registered trademark | 1 | 0.17% |
| record | registration_number: Unique number assigned to the trademark registration | 1 | 0.17% |
| record | opponent: Entity filing the dispute or opposition | 1 | 0.17% |
| record | dispute_date: Date the dispute or opposition was filed | 1 | 0.17% |
| record | status: Current legal status of the dispute | 1 | 0.17% |
| record | dispute_type: Type of legal action such as cancellation or infringement | 1 | 0.17% |
| record | sentiment.opinion | 1 | 0.17% |
| record | sentiment.preference | 1 | 0.17% |
| record | quote.text | 1 | 0.17% |
| record | quote.author | 1 | 0.17% |
| record | favorite_players | 1 | 0.17% |
| record | edit_information.editor | 1 | 0.17% |
| record | edit_information.timestamp | 1 | 0.17% |
| record | office_hours.4.day | 1 | 0.17% |
| record | office_hours.3.start_time | 1 | 0.17% |
| record | address.room | 1 | 0.17% |
| record | office_hours.2.start_time | 1 | 0.17% |
| record | office_hours.4.start_time | 1 | 0.17% |
| record | address.street | 1 | 0.17% |
| record | contact.fax | 1 | 0.17% |
| record | office_hours.4.end_time | 1 | 0.17% |
| record | office_hours.0.end_time | 1 | 0.17% |
| record | office_hours.1.start_time | 1 | 0.17% |
| record | office_hours.2.end_time | 1 | 0.17% |
| record | address.postal_code | 1 | 0.17% |
| record | office_hours.0.start_time | 1 | 0.17% |
| record | address.city | 1 | 0.17% |
| record | office_hours.0.day | 1 | 0.17% |
| record | account_number | 1 | 0.17% |
| record | merchant_name | 1 | 0.17% |
| record | github_url | 1 | 0.17% |
| record | merchant: Name of the merchant or store involved | 1 | 0.17% |
| record | status: Current status of the transaction | 1 | 0.17% |
| record | subject_line | 1 | 0.17% |
| record | performance_rating | 1 | 0.17% |
| record | target_population | 1 | 0.17% |
| record | publication_date | 1 | 0.17% |
| record | organizer | 1 | 0.17% |
| record | attendance | 1 | 0.17% |
| record | contact_email | 1 | 0.17% |
| record | explanation | 1 | 0.17% |
| record | ability | 1 | 0.17% |
| record | user_id: ID of the user who submitted the snapshot | 1 | 0.17% |
| record | status: Current status of the exchange rate record | 1 | 0.17% |
| record | snapshot_id: Unique identifier for the snapshot in format currency-yearmonthday | 1 | 0.17% |
| record | rate: Exchange rate in USD per foreign currency unit | 1 | 0.17% |
| record | acquired_company: Name of the company being acquired | 1 | 0.17% |
| record | sector: Industry or market sector involved in the acquisition | 1 | 0.17% |
| record | acquirer_ceo: CEO of the acquiring company | 1 | 0.17% |
| record | deal_id: Unique identifier for the acquisition transaction | 1 | 0.17% |
| record | targetagegroup: Age range of intended audience | 1 | 0.17% |
| record | campaign_id: Unique identifier for each promotional campaign | 1 | 0.17% |
| record | promo_code: Discount or offer code associated with the campaign | 1 | 0.17% |
| record | protecting_group.targets.1.abbreviation | 1 | 0.17% |
| record | protecting_group.targets.1.amino_acid_name | 1 | 0.17% |
| record | protecting_group.targets.1.derivative_form | 1 | 0.17% |
| record | protecting_group.targets.0.amino_acid_name | 1 | 0.17% |
| record | protecting_group.abbreviation | 1 | 0.17% |
| record | protecting_group.targets.1.protected_functional_group | 1 | 0.17% |
| record | deprotection_process.reagents.0.concentration_level | 1 | 0.17% |
| record | protecting_group.name | 1 | 0.17% |
| record | synthesis_context.intermediates_synthesized.0.components | 1 | 0.17% |
| record | protecting_group.targets.0.abbreviation | 1 | 0.17% |
| record | synthesis_context.methodology | 1 | 0.17% |
| record | synthesis_context.carboxyl_protection_requirement.reasoning | 1 | 0.17% |
| record | synthesis_context.intermediates_synthesized.0.type | 1 | 0.17% |
| record | protecting_group.targets.0.protected_functional_group | 1 | 0.17% |
| record | unit_id | 1 | 0.17% |
| record | award | 1 | 0.17% |
| record | title: Event title describing the nature and context of the incident | 1 | 0.17% |
| record | date: Timestamp when the incident occurred | 1 | 0.17% |
| record | involved_parties: List of organizations or individuals involved in the incident | 1 | 0.17% |
| record | source: Media or organization reporting the incident | 1 | 0.17% |
| record | lead_reporter: Reporter providing coverage of the incident | 1 | 0.17% |
| record | geographic_code: Unique code identifying the incident's geographic region | 1 | 0.17% |
| record | politician | 1 | 0.17% |
| record | message_id | 1 | 0.17% |
| record | outreach_status | 1 | 0.17% |
| record | outreach_code | 1 | 0.17% |
| record | hotel_details.star_rating | 1 | 0.17% |
| record | ranking.location | 1 | 0.17% |
| record | hotel_details.location | 1 | 0.17% |
| record | booking_options.partners | 1 | 0.17% |
| record | booking_options.description | 1 | 0.17% |
| record | price_range.currency | 1 | 0.17% |
| record | hotel_details.name | 1 | 0.17% |
| record | useoffunds: Purpose or application of raised capital | 1 | 0.17% |
| record | funding_id: Unique identifier assigned to the funding round | 1 | 0.17% |
| record | company: Name of the technology company receiving funds | 1 | 0.17% |
| record | submission_method | 1 | 0.17% |
| record | filing_amount | 1 | 0.17% |
| record | acquisition_date | 1 | 0.17% |
| record | cost | 1 | 0.17% |
| record | entity_name: Company name associated with filing | 1 | 0.17% |
| record | filing_id: Unique identifier assigned to the filing | 1 | 0.17% |
| record | revenue: Total reported revenue in millions | 1 | 0.17% |
| record | net_income: Total reported net income in millions | 1 | 0.17% |
| record | address: Physical address of the company | 1 | 0.17% |
| record | date_added | 1 | 0.17% |
| record | company.founding_location.city | 1 | 0.17% |
| record | company.location | 1 | 0.17% |
| record | company.transformation_date | 1 | 0.17% |
| record | company.description | 1 | 0.17% |
| record | company.establishment_date | 1 | 0.17% |
| record | service.target_audience | 1 | 0.17% |
| record | service.responsible_person.affiliation | 1 | 0.17% |
| record | service.opening_hours.0.time | 1 | 0.17% |
| record | service.location.address | 1 | 0.17% |
| record | service.responsible_person.profession | 1 | 0.17% |
| record | service.opening_hours.0.day | 1 | 0.17% |
| record | service.location.building | 1 | 0.17% |
| record | service.interventions | 1 | 0.17% |
| record | service.responsible_person.role | 1 | 0.17% |
| record | service.contact_information.email | 1 | 0.17% |
| record | service.contact_information.phone | 1 | 0.17% |
| record | service.location.postal_code | 1 | 0.17% |
| record | amount: Monetary value of the returned item | 1 | 0.17% |
| record | product_name: Name of the product returned | 1 | 0.17% |
| record | customer_name: Name of the customer initiating the return | 1 | 0.17% |
| record | order_id: Identifier of the original order associated with the return | 1 | 0.17% |
| record | dimensions: Dimensions in cm format | 1 | 0.17% |
| record | name: Name of the product as listed | 1 | 0.17% |
| record | material: Primary materials used in construction | 1 | 0.17% |
| record | weight: Weight of the product in grams or kg | 1 | 0.17% |
| record | physical_characteristics.observations.0.publication | 1 | 0.17% |
| record | discovery.observatory.location.county | 1 | 0.17% |
| record | diameter.unit | 1 | 0.17% |
| record | orbital_characteristics.inclination.unit | 1 | 0.17% |
| record | agreement_id | 1 | 0.17% |
| record | test_name | 1 | 0.17% |
| record | lab_id | 1 | 0.17% |
| record | owner_name | 1 | 0.17% |
| record | term_years | 1 | 0.17% |
| record | sector: Technology domain such as health tech, AI, clean energy | 1 | 0.17% |
| record | funding_amount: Total amount raised in the investment round | 1 | 0.17% |
| record | id: Unique identifier assigned to the startup | 1 | 0.17% |
| record | date: Date the match was scheduled or played | 1 | 0.17% |
| record | score: List of two integers representing home and away scores | 1 | 0.17% |
| record | team1: First team participating in the match | 1 | 0.17% |
| record | location: Venue where the match took place | 1 | 0.17% |
| record | statue.name.french | 1 | 0.17% |
| record | statue.restoration.0.description | 1 | 0.17% |
| record | statue.restoration.0.funding.amount_raised.currency | 1 | 0.17% |
| record | statue.restoration.0.funding.source | 1 | 0.17% |
| record | statue.restoration.0.funding.contribution_range.currency | 1 | 0.17% |
| record | statue.discovery.discoverer.name | 1 | 0.17% |
| record | statue.discovery.location | 1 | 0.17% |
| record | statue.restoration.0.funding.contribution_range.min | 1 | 0.17% |
| record | statue.restoration.0.budget.currency | 1 | 0.17% |
| record | statue.discovery.date | 1 | 0.17% |
| record | statue.restoration.0.start_date | 1 | 0.17% |
| record | statue.transport_to_france | 1 | 0.17% |
| record | statue.public_display.stairs_completion_date | 1 | 0.17% |
| record | statue.location.museum | 1 | 0.17% |
| record | statue.material | 1 | 0.17% |
| record | statue.public_display.reopening_date | 1 | 0.17% |
| record | statue.location.city | 1 | 0.17% |
| record | statue.discovery.discoverer.title | 1 | 0.17% |
| record | statue.restoration.0.funding.number_of_contributors | 1 | 0.17% |
| record | statue.name.greek | 1 | 0.17% |
| record | statue.dedication | 1 | 0.17% |
| record | holiday_look.collar.brand | 1 | 0.17% |
| record | holiday_look.heels.brand | 1 | 0.17% |
| record | comparison.similar_look | 1 | 0.17% |
| record | comparison.brand_with_similar_look | 1 | 0.17% |
| record | holiday_look.dress.designer | 1 | 0.17% |
| record | id: Unique identifier for the legislative vote | 1 | 0.17% |
| record | date: Date when the vote occurred | 1 | 0.17% |
| record | chamber: Legislative chamber where the vote took place | 1 | 0.17% |
| record | bill: Bill number and type identifier | 1 | 0.17% |
| record | sponsor: Name of the bill's sponsor | 1 | 0.17% |
| record | outcome: Final result of the vote | 1 | 0.17% |
| record | amount: Financial amount associated with the bill | 1 | 0.17% |
| record | location: Geographic location of the bill's jurisdiction | 1 | 0.17% |
| record | status: Current status of the bill | 1 | 0.17% |
| record | description: Summary of the bill's purpose and scope | 1 | 0.17% |
| record | beliefs | 1 | 0.17% |
| record | actions | 1 | 0.17% |
| record | temperature: Body temperature in degrees Celsius | 1 | 0.17% |
| record | respiratory_rate: Number of breaths per minute | 1 | 0.17% |
| record | age: Age of the patient in years | 1 | 0.17% |
| record | monitoring_time: Exact time of monitoring in 24-hour format | 1 | 0.17% |
| record | device_model: Model of the monitoring device used | 1 | 0.17% |
| record | status: Current status of the exam result | 1 | 0.17% |
| record | school: Name of the school where the exam was administered | 1 | 0.17% |
| record | score: Percentage score achieved in the exam | 1 | 0.17% |
| record | subject: Subject area of the exam | 1 | 0.17% |
| record | serial_number: Unique serial number for device tracking | 1 | 0.17% |
| record | visitor_team | 1 | 0.17% |
| record | geography.borders.2.direction | 1 | 0.17% |
| record | geography.borders.0.region | 1 | 0.17% |
| record | geography.borders.1.direction | 1 | 0.17% |
| record | infrastructure.refineries.0.refineryName | 1 | 0.17% |
| record | economy.gdpCurrency | 1 | 0.17% |
| record | infrastructure.highways.2.highwayName | 1 | 0.17% |
| record | infrastructure.roads.0.connections | 1 | 0.17% |
| record | infrastructure.airports.0.location | 1 | 0.17% |
| record | economy.economicBase | 1 | 0.17% |
| record | history.mergerDescription | 1 | 0.17% |
| record | infrastructure.airports.0.trafficVolume | 1 | 0.17% |
| record | infrastructure.airports.0.passengersYear | 1 | 0.17% |
| record | geography.rivers | 1 | 0.17% |
| record | infrastructure.highways.0.highwayName | 1 | 0.17% |
| record | economy.gdpRankAmongRegions | 1 | 0.17% |
| record | regionName | 1 | 0.17% |
| record | lowerLevelDivisions | 1 | 0.17% |
| record | infrastructure.refineries.0.closureDate | 1 | 0.17% |
| record | geography.borders.2.region | 1 | 0.17% |
| record | capital | 1 | 0.17% |
| record | infrastructure.refineries.0.status | 1 | 0.17% |
| record | regionNameBerber | 1 | 0.17% |
| record | history.establishmentDate | 1 | 0.17% |
| record | topics | 1 | 0.17% |
| record | required_skills | 1 | 0.17% |
| record | credit_hours | 1 | 0.17% |
| record | enrollment_status | 1 | 0.17% |
| record | meeting_days | 1 | 0.17% |
| record | enrollment_capacity | 1 | 0.17% |
| record | schema_name | 1 | 0.17% |
| record | post_id | 1 | 0.17% |
| record | upvotes | 1 | 0.17% |
| record | subreddit | 1 | 0.17% |
| record | location: Headquarters city and country of the company | 1 | 0.17% |
| record | lead_investor: Primary investor or consortium leading the round | 1 | 0.17% |
| record | condition | 1 | 0.17% |
| record | phone | 1 | 0.17% |
| record | promotional_code | 1 | 0.17% |
| record | party_name | 1 | 0.17% |
| record | tournament_code | 1 | 0.17% |
| record | winner | 1 | 0.17% |
| record | loser | 1 | 0.17% |
| record | aces | 1 | 0.17% |
| record | break_points | 1 | 0.17% |
| record | format | 1 | 0.17% |
| record | app_name | 1 | 0.17% |
| record | report_id: Unique identifier assigned to the phishing report | 1 | 0.17% |
| record | status: Current status of the report | 1 | 0.17% |
| record | sender_domain: Domain of the spoofed email address | 1 | 0.17% |
| record | due_date | 1 | 0.17% |
| record | tv_program.production.location.name | 1 | 0.17% |
| record | tv_program.production.production_area | 1 | 0.17% |
| record | tv_program.production.location.institution | 1 | 0.17% |
| record | tv_program.setting.year | 1 | 0.17% |
| record | tv_program.setting.location | 1 | 0.17% |
| record | tv_program.year | 1 | 0.17% |
| record | tv_program.previous_program | 1 | 0.17% |
| record | population.count | 1 | 0.17% |
| record | geography.coordinates.latitude | 1 | 0.17% |
| record | geography.coordinates.longitude | 1 | 0.17% |
| record | census_inclusion | 1 | 0.17% |
| record | technologies_mentioned | 1 | 0.17% |
| record | companies_mentioned | 1 | 0.17% |
| record | shipping_cost: Cost to ship the audio device in US dollars | 1 | 0.17% |
| record | material: Material used in device construction | 1 | 0.17% |
| record | weight: Weight of the device in grams | 1 | 0.17% |
| record | passenger_name: Name of the passenger on the flight | 1 | 0.17% |
| record | flight_no: Flight number as assigned by the airline | 1 | 0.17% |
| record | delay_minutes: Number of minutes flight is delayed | 1 | 0.17% |
| record | destination: Airport code where flight ends | 1 | 0.17% |
| record | email: Email address of the passenger | 1 | 0.17% |
| record | unique_id: Unique identifier for the delay entry | 1 | 0.17% |
| record | departure_date: Date flight departs from origin | 1 | 0.17% |
| record | medium | 1 | 0.17% |
| record | artistic_intent.meaning_zero_point | 1 | 0.17% |
| record | artistic_intent.style | 1 | 0.17% |
| record | location: Property address including street, city, and state | 1 | 0.17% |
| record | country: Nationality of the athlete who won the medal | 1 | 0.17% |
| record | venue: Location where the event took place | 1 | 0.17% |
| record | result: Medal awarded to the athlete | 1 | 0.17% |
| record | discipline: Type of sport or competition such as swimming or skiing | 1 | 0.17% |
| record | athlete_name: Name of the athlete who achieved the result | 1 | 0.17% |
| record | operating_margin | 1 | 0.17% |
| record | company_code | 1 | 0.17% |
| record | incident.suspects.0.age | 1 | 0.17% |
| record | incident.suspects.1.age | 1 | 0.17% |
| record | incident.charges.0.count | 1 | 0.17% |
| record | incident.charges.0.description | 1 | 0.17% |
| record | incident.charges.1.count | 1 | 0.17% |
| record | incident.charges.1.description | 1 | 0.17% |
| record | incident.charges.2.count | 1 | 0.17% |
| record | incident.charges.2.description | 1 | 0.17% |
| record | incident.additional_acts | 1 | 0.17% |
| record | incident.affected_locations | 1 | 0.17% |
| record | incident.number_of_complaints | 1 | 0.17% |
| record | source.newspaper | 1 | 0.17% |
| record | source.author | 1 | 0.17% |
| record | consideration_structure | 1 | 0.17% |
| record | acquirer_ceo | 1 | 0.17% |
| record | event-attack/battle/war/militaryconflict | 1 | 0.17% |
| record | finding_id | 1 | 0.17% |
| record | access_duration | 1 | 0.17% |
| record | access_type | 1 | 0.17% |
| record | data_code | 1 | 0.17% |
| record | actions.0.target | 1 | 0.17% |
| record | actions.0.location | 1 | 0.17% |
| record | instruction | 1 | 0.17% |
| record | discovery_year | 1 | 0.17% |
| record | route: Origin-destination route string like JFK-LAX | 1 | 0.17% |
| record | delay_reason: Human-readable description of delay cause | 1 | 0.17% |
| record | destination: Airport code where flight arrives | 1 | 0.17% |
| record | passenger_names: List of passenger names affected | 1 | 0.17% |
| record | compensation_offered: Amount offered in dollars per passenger | 1 | 0.17% |
| record | flight_number: Unique identifier for the delayed flight | 1 | 0.17% |
| record | filing_id | 1 | 0.17% |
| record | filing_type | 1 | 0.17% |
| record | booking_reference | 1 | 0.17% |
| record | disease_links | 1 | 0.17% |
| record | transcription_factors | 1 | 0.17% |
| record | key_components | 1 | 0.17% |
| record | activation_mechanism | 1 | 0.17% |
| record | time_management | 1 | 0.17% |
| record | feedback_method | 1 | 0.17% |
| record | overall_score | 1 | 0.17% |
| record | era_name | 1 | 0.17% |
| record | year_in_era | 1 | 0.17% |
| record | dynasty | 1 | 0.17% |
| record | symbol: Stock ticker symbol as listed in the data | 1 | 0.17% |
| record | change_percent: Percentage change in stock price for the day | 1 | 0.17% |
| record | close_price: Final closing price of the stock | 1 | 0.17% |
| record | volume: Total shares traded on the day | 1 | 0.17% |
| record | market_cap: Market capitalization in billions | 1 | 0.17% |
| record | movement_reason: Brief reason for price movement or performance | 1 | 0.17% |
| record | conference | 1 | 0.17% |
| record | co_authors | 1 | 0.17% |
| record | id: Unique identifier for each rental listing | 1 | 0.17% |
| record | pet_policy: Pet policy including restrictions or fees | 1 | 0.17% |
| record | property_type: Type of housing unit | 1 | 0.17% |
| record | description: Brief description of unit features or conditions | 1 | 0.17% |
| record | address: Full property address including street and city | 1 | 0.17% |
| record | strategic_impact | 1 | 0.17% |
| record | regulatory_notes | 1 | 0.17% |
| record | deal_date | 1 | 0.17% |
| record | antitrust_review | 1 | 0.17% |
| record | focus_area | 1 | 0.17% |
| record | acquired_company | 1 | 0.17% |
| record | acquisition_id | 1 | 0.17% |
| record | email_id | 1 | 0.17% |
| record | causeofloss | 1 | 0.17% |
| record | damage_amount | 1 | 0.17% |
| record | semester | 1 | 0.17% |
| record | merchant_code | 1 | 0.17% |
| record | game_status | 1 | 0.17% |
| record | event_type | 1 | 0.17% |
| record | performance_notes | 1 | 0.17% |
| record | goals | 1 | 0.17% |
| record | policyholder: Name of the vehicle owner filing the claim | 1 | 0.17% |
| record | address: Full street address of the policyholder | 1 | 0.17% |
| record | vehicle: Make and model of the vehicle involved | 1 | 0.17% |
| record | collision_type: Type of collision or damage event | 1 | 0.17% |
| record | damage_type: Specific parts damaged in the incident | 1 | 0.17% |
| record | adjuster: Name of the insurance adjuster assigned | 1 | 0.17% |
| record | location: Geographic location where the incident occurred | 1 | 0.17% |
| record | use_case | 1 | 0.17% |
| record | review_length | 1 | 0.17% |
| record | compatibility | 1 | 0.17% |
| record | release_year | 1 | 0.17% |
| record | charging_time | 1 | 0.17% |
| record | warranty_period | 1 | 0.17% |
| record | payer | 1 | 0.17% |
| record | post_date | 1 | 0.17% |
| record | user_handle | 1 | 0.17% |
| record | ranking.location_type | 1 | 0.17% |
| record | location.province | 1 | 0.17% |
| record | booking_options.partnerships | 1 | 0.17% |
| record | price_range.per | 1 | 0.17% |
| record | booking_options.tripadvisor_role | 1 | 0.17% |
| record | discovery_location | 1 | 0.17% |
| record | asteroid_number | 1 | 0.17% |
| record | discoverers | 1 | 0.17% |
| record | resolution_timeline | 1 | 0.17% |
| record | industry_focus | 1 | 0.17% |
| record | internal_code | 1 | 0.17% |
| record | revenue_estimate | 1 | 0.17% |
| record | customer_address | 1 | 0.17% |
| record | originalordernumber | 1 | 0.17% |
| record | product_title | 1 | 0.17% |
| record | return_reason | 1 | 0.17% |
| record | shipping_method | 1 | 0.17% |
| record | processing_code | 1 | 0.17% |
| record | symptom_date | 1 | 0.17% |
| record | assigned_staff: Healthcare professional assigned to manage the device | 1 | 0.17% |
| record | purchase_date: Date the device was acquired | 1 | 0.17% |
| record | serial_number: Unique device identifier assigned by manufacturer | 1 | 0.17% |
| record | expiration_date: Date after which device is no longer valid | 1 | 0.17% |
| record | location: Physical ward or room where device is located | 1 | 0.17% |
| record | invoice_id | 1 | 0.17% |
| record | lead_investor: List of lead investors or funds involved | 1 | 0.17% |
| record | post_valuation: Post-money valuation in millions | 1 | 0.17% |
| record | focus: Primary technology or business area of the company | 1 | 0.17% |
| record | amount: Total capital raised in millions | 1 | 0.17% |
| record | ingredients | 1 | 0.17% |
| record | cooking_methods | 1 | 0.17% |
| record | boiling_instructions.salt_advice | 1 | 0.17% |
| record | boiling_instructions.cooking_time.min_minutes | 1 | 0.17% |
| record | boiling_instructions.cooking_time.max_minutes | 1 | 0.17% |
| record | grilling_instructions.method | 1 | 0.17% |
| record | grilling_instructions.cooking_time.minutes | 1 | 0.17% |
| record | serving_suggestions | 1 | 0.17% |
| record | personal_experience.raw_corn_experience | 1 | 0.17% |
| record | bracket_path | 1 | 0.17% |
| record | team_name | 1 | 0.17% |
| record | team_id | 1 | 0.17% |
| record | registered_on | 1 | 0.17% |
| record | reasons | 1 | 0.17% |
| record | suggestions | 1 | 0.17% |
| record | faq_references | 1 | 0.17% |
| record | navigation | 1 | 0.17% |
| record | footer_links | 1 | 0.17% |
| record | browse_topics | 1 | 0.17% |
| record | headquarters: City and country where company is headquartered | 1 | 0.17% |
| record | status: Current operational or strategic status | 1 | 0.17% |
| record | launch_year: Year the company launched its platform | 1 | 0.17% |
| record | founder | 1 | 0.17% |
| record | founded_year | 1 | 0.17% |
| record | location: Physical address of the product launch | 1 | 0.17% |
| record | product_code: Internal code assigned to the product | 1 | 0.17% |
| record | price: Selling price of the product in dollars | 1 | 0.17% |
| record | product: Name of the skincare or apparel product | 1 | 0.17% |
| record | upload_date | 1 | 0.17% |
| record | death.date_julian | 1 | 0.17% |
| record | birth.date_gregorian | 1 | 0.17% |
| record | works.type | 1 | 0.17% |
| record | birth.date_julian | 1 | 0.17% |
| record | occupation | 1 | 0.17% |
| record | teacher | 1 | 0.17% |
| record | death.location | 1 | 0.17% |
| record | father.occupation | 1 | 0.17% |
| record | birth.location | 1 | 0.17% |
| record | expiry_date | 1 | 0.17% |
| record | ppe_required | 1 | 0.17% |
| record | hazard_class | 1 | 0.17% |
| record | handling_notes | 1 | 0.17% |
| record | comment_content: Text of the comment including personal anecdotes and tool usage | 1 | 0.17% |
| record | author: Reddit username of the comment author | 1 | 0.17% |
| record | downvotes: Number of downvotes the comment received | 1 | 0.17% |
| record | tool: Productivity tool recommended in the comment | 1 | 0.17% |
| record | buyer: Name of the customer who made the purchase | 1 | 0.17% |
| record | location: City and state where buyer resides | 1 | 0.17% |
| record | product: Product model identifier such as EBC-1234 or WAT-5678 | 1 | 0.17% |
| record | price: Price paid for the product in US dollars | 1 | 0.17% |
| record | rating: Star rating from 1 to 5 given by the buyer | 1 | 0.17% |
| record | shipping_method: Type of shipping used | 1 | 0.17% |
| record | tracking_number: Shipping tracking number assigned | 1 | 0.17% |
| record | order_id: Unique order identifier from the retailer | 1 | 0.17% |
| record | return_status: Whether the product was returned or kept | 1 | 0.17% |
| record | feedback: Additional comments or issues noted in review | 1 | 0.17% |
| record | owner: Name of the healthcare professional owning the device | 1 | 0.17% |
| record | location: Department or unit where device is currently located | 1 | 0.17% |
| record | device_id: Unique identifier assigned to the medical device | 1 | 0.17% |
| record | model: Manufacturer model name of the medical device | 1 | 0.17% |
| record | availability_date | 1 | 0.17% |
| record | availability_status | 1 | 0.17% |
| record | asteroid.discovery.year | 1 | 0.17% |
| record | sku: Stock keeping unit assigned to each product | 1 | 0.17% |
| record | seller_location: Full address of the seller including city, state, and zip code | 1 | 0.17% |
| record | unit_price: Price per unit of the product | 1 | 0.17% |
| record | contributor_count | 1 | 0.17% |
| record | project_url | 1 | 0.17% |
| record | lastcommitdate | 1 | 0.17% |
| record | tissue_type | 1 | 0.17% |
| record | pathway_id | 1 | 0.17% |
| record | downvotes | 1 | 0.17% |
| record | originalpostlink | 1 | 0.17% |
| record | content_length | 1 | 0.17% |
| record | comment_rank | 1 | 0.17% |
| record | asteroid.discovery_year | 1 | 0.17% |
| record | asteroid.orbital_characteristics.unit | 1 | 0.17% |
| record | asteroid.orbital_characteristics.inclination_unit | 1 | 0.17% |
| record | policyholder_name: Name of the policyholder filing the claim | 1 | 0.17% |
| record | address: Full address of the policyholder | 1 | 0.17% |
| record | claim_date: Date when the incident occurred | 1 | 0.17% |
| record | amount_requested: Total amount requested for compensation | 1 | 0.17% |
| record | status: Current status of the claim | 1 | 0.17% |
| record | description: Brief description of the incident and damages | 1 | 0.17% |
| record | attending_physician | 1 | 0.17% |
| record | bloodpressurediastolic | 1 | 0.17% |
| record | monitoring_method | 1 | 0.17% |
| record | bloodpressuresystolic | 1 | 0.17% |
| record | respiratory_rate | 1 | 0.17% |
| record | website | 1 | 0.17% |
| record | device model | 1 | 0.17% |
| record | developer | 1 | 0.17% |
| record | place | 1 | 0.17% |
| record | prescriber_id | 1 | 0.17% |
| record | prescriber_name | 1 | 0.17% |
| record | meeting_id | 1 | 0.17% |
| record | species_name | 1 | 0.17% |
| record | family.name | 1 | 0.17% |
| record | described_by.year | 1 | 0.17% |
| record | posted_on | 1 | 0.17% |
| record | thumbnail | 1 | 0.17% |
| record | vaccination_id | 1 | 0.17% |
| record | administration_date | 1 | 0.17% |
| record | dose_number | 1 | 0.17% |
| record | encounter_id: Unique identifier for the visit | 1 | 0.17% |
| record | address: Patient's residential address | 1 | 0.17% |
| record | visit_date: Date of the medical encounter | 1 | 0.17% |
| record | icd_code: ICD-10 code for the primary diagnosis | 1 | 0.17% |
| record | bp_systolic: Systolic blood pressure in mmHg | 1 | 0.17% |
| record | bp_diastolic: Diastolic blood pressure in mmHg | 1 | 0.17% |
| record | medications: List of prescribed medications with dosing | 1 | 0.17% |
| record | lab_results: List of lab test results | 1 | 0.17% |
| record | symptoms: Patient-reported symptoms or complaints | 1 | 0.17% |
| record | description.leaves.terminal_spine.color | 1 | 0.17% |
| record | description.rosette | 1 | 0.17% |
| record | systematics_and_distribution.location.country | 1 | 0.17% |
| record | description.inflorescence.type | 1 | 0.17% |
| record | description.flowers.arrangement | 1 | 0.17% |
| record | description.fruit.width_mm.max | 1 | 0.17% |
| record | description.leaves.color | 1 | 0.17% |
| record | systematics_and_distribution.location.state | 1 | 0.17% |
| record | systematics_and_distribution.similar_species | 1 | 0.17% |
| record | description.leaves.surface | 1 | 0.17% |
| record | systematics_and_distribution.first_description.year | 1 | 0.17% |
| record | systematics_and_distribution.distinguishing_features | 1 | 0.17% |
| record | systematics_and_distribution.location.elevation_m.max | 1 | 0.17% |
| record | description.leaves.length_cm.min | 1 | 0.17% |
| record | description.flowers.length_mm.min | 1 | 0.17% |
| record | description.leaves.width_cm.min | 1 | 0.17% |
| record | description.fruit.type | 1 | 0.17% |
| record | description.fruit.length_mm.min | 1 | 0.17% |
| record | description.leaves.margins | 1 | 0.17% |
| record | description.fruit.shape | 1 | 0.17% |
| record | description.fruit.width_mm.min | 1 | 0.17% |
| record | description.flowering_time | 1 | 0.17% |
| record | instructor | 1 | 0.17% |
| record | days | 1 | 0.17% |
| record | match_code | 1 | 0.17% |
| record | vin | 1 | 0.17% |
| record | dispute_status | 1 | 0.17% |
| record | owner_location | 1 | 0.17% |
| record | status: Current enrollment or completion status | 1 | 0.17% |
| record | start_date: Date when the course begins | 1 | 0.17% |
| record | course_id: Unique identifier for the course as given in the XML | 1 | 0.17% |
| record | election_type | 1 | 0.17% |
| record | candidate_location | 1 | 0.17% |
| record | candidate_name | 1 | 0.17% |
| record | discharge_destination | 1 | 0.17% |
| record | rma_number | 1 | 0.17% |
| record | failure_reason | 1 | 0.17% |
| record | return_date | 1 | 0.17% |
| record | count | 1 | 0.17% |
| record | notes: Additional context about the appointment or background | 1 | 0.17% |
| record | compensation: Annual salary or compensation amount | 1 | 0.17% |
| record | position: Job title or role within the department | 1 | 0.17% |
| record | date: Start date of the appointment | 1 | 0.17% |
| record | from_club | 1 | 0.17% |
| record | to_club | 1 | 0.17% |
| record | transfer_fee | 1 | 0.17% |
| record | transfer_code | 1 | 0.17% |
| record | transfer_portal | 1 | 0.17% |
| record | founder_names | 1 | 0.17% |
| record | sentiment | 1 | 0.17% |
| record | investor_group | 1 | 0.17% |
| record | author.role | 1 | 0.17% |
| record | version | 1 | 0.17% |
| record | software | 1 | 0.17% |
| record | statement_reference: Identifier code for the statement document | 1 | 0.17% |
| record | amount: Financial value associated with the policy or proposal | 1 | 0.17% |
| record | location: City and state where the event occurred | 1 | 0.17% |
| record | topic: Subject matter discussed in the statement | 1 | 0.17% |
| record | speaker: Name of the person delivering the statement | 1 | 0.17% |
| record | sunless_tanning.product_collection.products.0.type | 1 | 0.17% |
| record | sunless_tanning.product_collection.products.1.type | 1 | 0.17% |
| record | sunless_tanning.product_collection.products.0.ingredients | 1 | 0.17% |
| record | sunless_tanning.product_collection.products.0.description | 1 | 0.17% |
| record | sunless_tanning.product_collection.brand | 1 | 0.17% |
| record | sunless_tanning.product_collection.products.0.benefits | 1 | 0.17% |
| record | sunless_tanning.product_collection.benefits | 1 | 0.17% |
| record | sunless_tanning.product_collection.products.0.sizes | 1 | 0.17% |
| record | sunless_tanning.product_collection.products.1.name | 1 | 0.17% |
| record | sunless_tanning.product_collection.products.1.description | 1 | 0.17% |
| record | sunless_tanning.concerns | 1 | 0.17% |
| record | investigators | 1 | 0.17% |
| record | funding_sources | 1 | 0.17% |
| record | secondary_endpoints | 1 | 0.17% |
| record | median_age | 1 | 0.17% |
| record | exam_type | 1 | 0.17% |
| record | protocol | 1 | 0.17% |
| record | provider_name | 1 | 0.17% |
| record | billed_amount | 1 | 0.17% |
| record | visit_count | 1 | 0.17% |
| record | taxonomy.describer.affiliation.0.laboratory | 1 | 0.17% |
| record | description.distinguishing_features | 1 | 0.17% |
| record | species.etymology.named_after.0.location.city | 1 | 0.17% |
| record | distribution.habitat | 1 | 0.17% |
| record | taxonomy.describer.affiliation.0.university | 1 | 0.17% |
| record | taxonomy.describer.affiliation.0.location.country | 1 | 0.17% |
| record | description.size | 1 | 0.17% |
| record | description.color | 1 | 0.17% |
| record | description.width | 1 | 0.17% |
| record | taxonomy.subgenus | 1 | 0.17% |
| record | species.etymology.named_after.0.location.country | 1 | 0.17% |
| record | taxonomy.describer.affiliation.0.location.city | 1 | 0.17% |
| record | taxonomy.describer.name | 1 | 0.17% |
| record | taxonomy.year_described | 1 | 0.17% |
| record | taxonomy.type_materials | 1 | 0.17% |
| record | hotel_experience.location.nearby_landmark | 1 | 0.17% |
| record | hotel_experience.location.city | 1 | 0.17% |
| record | hotel_experience.entertainment.tv.content | 1 | 0.17% |
| record | hotel_experience.additional_benefits | 1 | 0.17% |
| record | hotel_experience.suites.0.workspace | 1 | 0.17% |
| record | hotel_experience.suites.0.suite_type | 1 | 0.17% |
| record | hotel_experience.suites.0.bathroom_features | 1 | 0.17% |
| record | hotel_experience.suites.0.amenities | 1 | 0.17% |
| record | hotel_experience.suites.0.features | 1 | 0.17% |
| record | hotel_experience.entertainment.tv.screen_size | 1 | 0.17% |
| record | hba1c | 1 | 0.17% |
| record | publication | 1 | 0.17% |
| record | JJ | 1 | 0.17% |
| record | CD | 1 | 0.17% |
| record | DT | 1 | 0.17% |
| record | . | 1 | 0.17% |
| record | NNP | 1 | 0.17% |
| record | IN | 1 | 0.17% |
| record | CC | 1 | 0.17% |
| record | NN | 1 | 0.17% |
| record | cause: Reason for the damage such as flood or fire | 1 | 0.17% |
| record | disease.pathogenesis.transmission | 1 | 0.17% |
| record | disease.subspecies.2.name | 1 | 0.17% |
| record | disease.subspecies.0.affected_species | 1 | 0.17% |
| record | disease.subspecies.1.name | 1 | 0.17% |
| record | disease.subspecies.0.name | 1 | 0.17% |
| record | disease.description | 1 | 0.17% |
| record | disease.subspecies.1.associated_diseases | 1 | 0.17% |
| record | location: Office location where the visit occurred | 1 | 0.17% |
| record | medication: Specific medication prescribed with dosage | 1 | 0.17% |
| record | visit_date: Date of the patient visit | 1 | 0.17% |
| record | lab_order: List of lab tests ordered for the patient | 1 | 0.17% |
| record | episode_number | 1 | 0.17% |
| record | user: Name of the team member or user using the product | 1 | 0.17% |
| record | product_id: Unique identifier for the product instance | 1 | 0.17% |
| record | monthly_fee: Monthly cost of the product subscription | 1 | 0.17% |
| record | performance_impact: Percentage improvement in lead quality or hiring time | 1 | 0.17% |
| record | product_name: Name of the product being used | 1 | 0.17% |
| record | company: Organization where the product is used | 1 | 0.17% |
| record | zip | 1 | 0.17% |
| record | RP | 1 | 0.17% |
| record | VBD | 1 | 0.17% |
| record | primary_provider: Name of the attending physician | 1 | 0.17% |
| record | admission_diagnosis: Primary diagnosis at admission | 1 | 0.17% |
| record | discharge_date: Date the patient was discharged | 1 | 0.17% |
| record | medications: List of prescribed medications | 1 | 0.17% |
| record | status: Current state of the incident | 1 | 0.17% |
| record | customer_name: Name of the affected customer | 1 | 0.17% |
| record | address: Full residential or commercial address of the customer | 1 | 0.17% |
| record | impact: Description of service impact including data loss or business effects | 1 | 0.17% |
| record | start_term | 1 | 0.17% |
| record | hall | 1 | 0.17% |
| record | course_code | 1 | 0.17% |
| record | singles | 1 | 0.17% |
| record | domain.registry.name | 1 | 0.17% |
| record | domain.requirements.minimum_length | 1 | 0.17% |
| record | domain.third_level_domains | 1 | 0.17% |
| record | domain.territory | 1 | 0.17% |
| record | establishment_year | 1 | 0.17% |
| record | departments | 1 | 0.17% |
| record | staff.research_assistant_count | 1 | 0.17% |
| record | staff.teaching_staff_count | 1 | 0.17% |
| record | dean.name | 1 | 0.17% |
| record | laboratories | 1 | 0.17% |
| record | dean.title | 1 | 0.17% |
| record | identifier | 1 | 0.17% |
| record | staffing_change | 1 | 0.17% |
| record | program_name | 1 | 0.17% |
| record | underwriter | 1 | 0.17% |
| record | funding_id: Unique identifier for each funding round | 1 | 0.17% |
| record | roundtype: Type of funding round | 1 | 0.17% |
| record | status: Current status of the funding round | 1 | 0.17% |
| record | company_name: Name of the startup receiving funding | 1 | 0.17% |
| record | url | 1 | 0.17% |
| record | drug_name | 1 | 0.17% |
| record | patient_gender | 1 | 0.17% |
| record | adverse_event | 1 | 0.17% |
| record | start_date | 1 | 0.17% |
| record | boulder.start.foot_holds.1.foot | 1 | 0.17% |
| record | boulder.start.foot_holds.0.foot | 1 | 0.17% |
| record | boulder.start.hand_holds.1.description | 1 | 0.17% |
| record | boulder.start.hand_holds.0.description | 1 | 0.17% |
| record | boulder.start.foot_holds.0.description | 1 | 0.17% |
| record | boulder.start.hand_holds.1.hand | 1 | 0.17% |
| record | boulder.location.relative_position | 1 | 0.17% |
| record | boulder.location.distance_unit | 1 | 0.17% |
| record | boulder.location.reference_point | 1 | 0.17% |
| record | distance.calculation_method | 1 | 0.17% |
| record | latin_name | 1 | 0.17% |
| record | distance.parsecs | 1 | 0.17% |
| record | apparent_magnitude.max | 1 | 0.17% |
| record | characteristics.effective_temperature.value | 1 | 0.17% |
| record | characteristics.type | 1 | 0.17% |
| record | distance.light_years | 1 | 0.17% |
| record | constellation | 1 | 0.17% |
| record | apparent_magnitude.unit | 1 | 0.17% |
| record | contact_person | 1 | 0.17% |
| record | paymentduedate | 1 | 0.17% |
| record | vendor_name | 1 | 0.17% |
| record | account_code | 1 | 0.17% |
| record | approval_year | 1 | 0.17% |
| record | route | 1 | 0.17% |
| record | dosage_range | 1 | 0.17% |
| record | age_range | 1 | 0.17% |
| record | side_effects | 1 | 0.17% |
| record | approval_status | 1 | 0.17% |
| record | attendee_count | 1 | 0.17% |
| record | publication_time | 1 | 0.17% |
| record | vendor_address | 1 | 0.17% |
| record | recipient_email | 1 | 0.17% |
| record | person.publication_information.available_part | 1 | 0.17% |
| record | person.commentary_based_on | 1 | 0.17% |
| record | person.publication_information.commentary_with | 1 | 0.17% |
| record | person.publication_information.location | 1 | 0.17% |
| record | person.time_period.era | 1 | 0.17% |
| record | person.associated_with | 1 | 0.17% |
| record | inventor | 1 | 0.17% |
| record | technology_area | 1 | 0.17% |
| record | physical_characteristics.observations.0.telescope | 1 | 0.17% |
| record | physical_characteristics.observations.0.year | 1 | 0.17% |
| record | medals.silver | 1 | 0.17% |
| record | games | 1 | 0.17% |
| record | medals.total | 1 | 0.17% |
| record | political_entity | 1 | 0.17% |
| record | advertiser_information.contact_information.phone_number | 1 | 0.17% |
| record | public_information.brochure_name | 1 | 0.17% |
| record | distribution_information.trail_details.trail_name | 1 | 0.17% |
| record | public_information.download_share | 1 | 0.17% |
| record | advertiser_information.new_brochure_planned | 1 | 0.17% |
| record | distribution_information.distribution_locations | 1 | 0.17% |
| record | isotope | 1 | 0.17% |
| record | creation.event | 1 | 0.17% |
| record | hanja_name | 1 | 0.17% |
| record | creation.year | 1 | 0.17% |
| record | details.uniform.belt_type | 1 | 0.17% |
| record | rank_level | 1 | 0.17% |
| record | hangul_name | 1 | 0.17% |
| record | details.uniform.belt_color | 1 | 0.17% |
| record | creation.king | 1 | 0.17% |
| record | template_substitution.prefix | 1 | 0.17% |
| record | advantages | 1 | 0.17% |
| record | template_substitution.examples | 1 | 0.17% |
| record | template_substitution.preview_code_example | 1 | 0.17% |
| record | templates_to_substitute | 1 | 0.17% |
| record | template_types.ordinary_template | 1 | 0.17% |
| record | organizer: Person or group hosting the event | 1 | 0.17% |
| record | name: Title or name of the event | 1 | 0.17% |
| record | location: Physical address or venue of the event | 1 | 0.17% |
| record | price: Entry cost for attending the event | 1 | 0.17% |
| record | status: Current phase of the event lifecycle | 1 | 0.17% |
| record | event_id: Unique identifier for each event | 1 | 0.17% |
| record | review_journal: Name of the publication that reviewed the work | 1 | 0.17% |
| record | author: Name of the author or poet | 1 | 0.17% |
| record | location: Geographic location of the reviewer | 1 | 0.17% |
| record | publisher: Publishing house responsible for the work | 1 | 0.17% |
| record | review_id: Unique identifier assigned to the review | 1 | 0.17% |
| record | statement_id | 1 | 0.17% |
| record | speaker | 1 | 0.17% |
| record | event_date | 1 | 0.17% |
| record | receiver | 1 | 0.17% |
| record | street | 1 | 0.17% |
| record | campaign_id | 1 | 0.17% |
| record | open_rate | 1 | 0.17% |
| record | utilities_included | 1 | 0.17% |
| record | contact_method | 1 | 0.17% |
| record | preferredtenanttype | 1 | 0.17% |
| record | description: Brief description of the transaction activity | 1 | 0.17% |
| record | creditcardlast_four: Last four digits of the linked credit card | 1 | 0.17% |
| record | merchant_name: Name of the merchant or store | 1 | 0.17% |
| record | transaction_amount: Amount of the individual transaction | 1 | 0.17% |
| record | late_fee: Fee assessed for missing payment deadline | 1 | 0.17% |
| record | status: Transaction approval or rejection status | 1 | 0.17% |
| record | finance_charge: Fee due to unpaid balance from previous cycle | 1 | 0.17% |
| record | punct | 1 | 0.17% |
| record | dobj | 1 | 0.17% |
| record | nmod | 1 | 0.17% |
| record | cc | 1 | 0.17% |
| record | preconj | 1 | 0.17% |
| record | nsubj | 1 | 0.17% |
| record | case | 1 | 0.17% |
| record | ROOT | 1 | 0.17% |
| record | advmod | 1 | 0.17% |
| record | npadvmod | 1 | 0.17% |
| record | prep | 1 | 0.17% |
| record | intj | 1 | 0.17% |
| record | nummod | 1 | 0.17% |
| record | renewal_amount | 1 | 0.17% |
| record | rating: Star rating out of 5 given by the reviewer | 1 | 0.17% |
| record | battery_life: Battery duration in hours under normal use | 1 | 0.17% |
| record | location: City and country of the reviewer | 1 | 0.17% |
| record | defendant_name | 1 | 0.17% |
| record | plaintiff_address | 1 | 0.17% |
| record | plaintiff_name | 1 | 0.17% |
| record | metabolite | 1 | 0.17% |
| record | established | 1 | 0.17% |
| record | size | 1 | 0.17% |
| record | asset | 1 | 0.17% |
| record | destination_club | 1 | 0.17% |
| record | origin_club | 1 | 0.17% |
| record | player_age | 1 | 0.17% |
| record | appearances | 1 | 0.17% |
| record | playerbirthcity | 1 | 0.17% |
| record | company.financial_results.reporting_period | 1 | 0.17% |
| record | report_metadata.source_url | 1 | 0.17% |
| record | listing_date | 1 | 0.17% |
| record | prescriber | 1 | 0.17% |
| record | fda_code | 1 | 0.17% |
| record | id: Unique identifier assigned to each competitor | 1 | 0.17% |
| record | hq: Headquarters city and state of the company | 1 | 0.17% |
| record | date: Date of the competitor record entry | 1 | 0.17% |
| record | revenue: Annual revenue in millions of dollars | 1 | 0.17% |
| record | market_share: Percentage of market share held by the company | 1 | 0.17% |
| record | status: Current operational or strategic status of the company | 1 | 0.17% |
| record | asteroid.alternative_designations | 1 | 0.17% |
| record | asteroid.discovery.discoverer | 1 | 0.17% |
| record | asteroid.discovery.discovery_rank | 1 | 0.17% |
| record | asteroid.description.discovery_details.date | 1 | 0.17% |
| record | asteroid.description.discovery_details.observatory | 1 | 0.17% |
| record | asteroid.description.discovery_details.location.country | 1 | 0.17% |
| record | asteroid.description.discovery_details.location.state | 1 | 0.17% |
| record | asteroid.description.discovery_details.location.county | 1 | 0.17% |
| record | asteroid.description.discovery_details.project | 1 | 0.17% |
| record | asteroid.description.observations.last_observation_date | 1 | 0.17% |
| record | asteroid.description.observations.observation_period_years | 1 | 0.17% |
| record | asteroid.description.orbital_period_years | 1 | 0.17% |
| record | asteroid.orbital_characteristics.semi_major_axis | 1 | 0.17% |
| record | asteroid.orbital_characteristics.perihelion | 1 | 0.17% |
| record | asteroid.orbital_characteristics.eccentricity | 1 | 0.17% |
| record | asteroid.orbital_characteristics.inclination | 1 | 0.17% |
| record | asteroid.orbital_characteristics.classification | 1 | 0.17% |
| record | asteroid.external_links.0.title | 1 | 0.17% |
| record | asteroid.wiki_sky_links | 1 | 0.17% |
| record | description: Brief explanation of the transaction purpose | 1 | 0.17% |
| record | bp | 1 | 0.17% |
| record | doctor | 1 | 0.17% |
| record | rr | 1 | 0.17% |
| record | item_description.shipping.ready_to_ship_in.quantity | 1 | 0.17% |
| record | item_description.shipping.rush_details | 1 | 0.17% |
| record | item_description.weight.unit | 1 | 0.17% |
| record | item_description.dimensions.length | 1 | 0.17% |
| record | item_description.setup_charge.per | 1 | 0.17% |
| record | item_description.setup_charge.amount | 1 | 0.17% |
| record | item_description.weight.value | 1 | 0.17% |
| record | province | 1 | 0.17% |
| record | percentage | 1 | 0.17% |
| record | passengers_affected | 1 | 0.17% |
| record | related_conditions | 1 | 0.17% |
| record | incident | 1 | 0.17% |
| record | claimant | 1 | 0.17% |
| record | vehicle_make | 1 | 0.17% |
| record | damage_type | 1 | 0.17% |
| record | circular.subject | 1 | 0.17% |
| record | circular.ministry | 1 | 0.17% |
| record | game_id | 1 | 0.17% |
| record | steals | 1 | 0.17% |
| record | assists | 1 | 0.17% |
| record | devices | 1 | 0.17% |
| record | physiological_measures | 1 | 0.17% |
| record | applications | 1 | 0.17% |
| record | name: Full title of the course including subject and focus | 1 | 0.17% |
| record | credit_hours: Number of credit hours for the course | 1 | 0.17% |
| record | political_context | 1 | 0.17% |
| record | audit_id | 1 | 0.17% |
| record | first_office | 1 | 0.17% |
| record | work_hours | 1 | 0.17% |
| record | funding_type | 1 | 0.17% |
| record | year_founded | 1 | 0.17% |
| record | theme | 1 | 0.17% |
| record | comment_id | 1 | 0.17% |
| record | resolution_hours | 1 | 0.17% |
| record | resolution_status | 1 | 0.17% |
| record | question.text | 1 | 0.17% |
| record | question.keywords | 1 | 0.17% |
| record | question.legal_implications | 1 | 0.17% |
| record | details.date_posted | 1 | 0.17% |
| record | details.days_ago | 1 | 0.17% |
| record | district | 1 | 0.17% |
| record | contribution_type | 1 | 0.17% |
| record | salary_range: Expected or offered salary range for the role | 1 | 0.17% |
| record | target_role: Job title or position being targeted | 1 | 0.17% |
| record | message_id: Unique identifier for the outreach message | 1 | 0.17% |
| record | target_company: Company where the role is located | 1 | 0.17% |
| record | risk_category | 1 | 0.17% |
| record | geographic_region | 1 | 0.17% |
| record | rebounds | 1 | 0.17% |
| record | exam_date | 1 | 0.17% |
| record | imaging_type | 1 | 0.17% |
| record | review_code | 1 | 0.17% |
| record | research_project | 1 | 0.17% |
| record | principal_investigator | 1 | 0.17% |
| record | project_status | 1 | 0.17% |
| record | ticket_id: Internal ticket identifier for the RMA | 1 | 0.17% |
| record | serial: Unique identifier for the product unit | 1 | 0.17% |
| record | rma_number: Assigned number for the RMA case | 1 | 0.17% |
| record | cost: Repair cost, if any, as stated in the request | 1 | 0.17% |
| record | customer: Name of the product owner submitting the request | 1 | 0.17% |
| record | population | 1 | 0.17% |
| record | user_name: Name of the customer involved in the support request | 1 | 0.17% |
| record | cause: Reason for the reported issue or system failure | 1 | 0.17% |
| record | resolution_method: Action taken to resolve the issue | 1 | 0.17% |
| record | support_agent: Name of the agent assigned to handle the case | 1 | 0.17% |
| record | product_id: Unique identifier assigned by the manufacturer | 1 | 0.17% |
| record | price: Product cost in US dollars | 1 | 0.17% |
| record | company: Manufacturer or developer of the product | 1 | 0.17% |
| record | launch_year: Year the product was first introduced | 1 | 0.17% |
| record | name: Product model name as branded | 1 | 0.17% |
| record | subject_focus: Key subjects or pathways emphasized in the results | 1 | 0.17% |
| record | grade_distribution: Breakdown of grades achieved by students | 1 | 0.17% |
| record | location: Geographic region or city where the institution is located | 1 | 0.17% |
| record | publication_code: Unique identifier assigned to the result publication | 1 | 0.17% |
| record | physical_characteristics.observations.0.telescope.country | 1 | 0.17% |
| record | discovery.observatory.location.state | 1 | 0.17% |
| record | discovery.project | 1 | 0.17% |
| record | bench_points: Total points scored by bench players | 1 | 0.17% |
| record | key_moment: Descriptive note about a pivotal play or event in the game | 1 | 0.17% |
| record | steals: Total steals recorded during the game | 1 | 0.17% |
| record | game_id: Unique identifier for the game as provided in the passage | 1 | 0.17% |
| record | home_score: Final score of the home team | 1 | 0.17% |
| record | booth_size | 1 | 0.17% |
| record | exhibitor_id | 1 | 0.17% |
| record | registration_date | 1 | 0.17% |
| record | price: Current selling price of the product in USD | 1 | 0.17% |
| record | last_updated: Date when product record was last modified | 1 | 0.17% |
| record | warehouse_location: Geographic warehouse where product is stored | 1 | 0.17% |
| record | room_type | 1 | 0.17% |
| record | nightly_rate | 1 | 0.17% |
| record | symbol: Unique ticker identifier for the company | 1 | 0.17% |
| record | price: Closing stock price in dollars | 1 | 0.17% |
| record | sector: Industry sector classification | 1 | 0.17% |
| record | status: Direction of price movement | 1 | 0.17% |
| record | airline | 1 | 0.17% |
| record | departure_city | 1 | 0.17% |
| record | departure_code | 1 | 0.17% |
| record | arrival_city | 1 | 0.17% |
| record | arrival_code | 1 | 0.17% |
| record | departure_time | 1 | 0.17% |
| record | arrival_time | 1 | 0.17% |
| record | passenger_email | 1 | 0.17% |
| record | passenger_type | 1 | 0.17% |
| record | baggage_count | 1 | 0.17% |
| record | seat_class | 1 | 0.17% |
| record | remarks | 1 | 0.17% |
| record | review_type | 1 | 0.17% |
| record | review_score | 1 | 0.17% |
| record | github_link | 1 | 0.17% |
| record | post_conference_event | 1 | 0.17% |
| record | student_rate | 1 | 0.17% |
| record | early_bird_discount_ends | 1 | 0.17% |
| record | current_registration_count | 1 | 0.17% |
| record | keynote_sessions | 1 | 0.17% |
| record | panel_discussions | 1 | 0.17% |
| record | tracks | 1 | 0.17% |
| record | core_values | 1 | 0.17% |
| record | registration_type | 1 | 0.17% |
| record | workshop_date | 1 | 0.17% |
| record | backup_plan | 1 | 0.17% |
| record | pre_conference_workshop | 1 | 0.17% |
| record | venue_capacity | 1 | 0.17% |
| record | event_status | 1 | 0.17% |
| record | lightning_talks | 1 | 0.17% |
| record | expected_revenue | 1 | 0.17% |
| record | senior_rate | 1 | 0.17% |
| record | recipient_jurisdiction: Jurisdiction such as city, state, or federal level | 1 | 0.17% |
| record | donor_location: City and state of the donor | 1 | 0.17% |
| record | contribution_id: Unique identifier for the contribution record | 1 | 0.17% |
| record | goal_achievement: Percentage of goals met, from 60% to 100% | 1 | 0.17% |
| record | name: Full name of the employee as listed in the review | 1 | 0.17% |
| record | employee_id: Unique identifier assigned to the employee | 1 | 0.17% |
| record | department: Department where the employee works | 1 | 0.17% |
| record | text | 1 | 0.17% |
| record | genre | 1 | 0.17% |
| record | name: Full name of the enrolled individual | 1 | 0.17% |
| record | enrollment_date: Date when enrollment began | 1 | 0.17% |
| record | program: Training or program name enrolled in | 1 | 0.17% |
| record | id: Unique identifier for the enrollment record | 1 | 0.17% |
| record | location: Geographic location of the enrollment site | 1 | 0.17% |
| record | new_website_url | 1 | 0.17% |
| record | claim_amount: Total cost of the medical service billed | 1 | 0.17% |
| record | patient_address: Full address of the patient | 1 | 0.17% |
| record | key_features | 1 | 0.17% |
| record | issues | 1 | 0.17% |
| record | followupclinic_address: Physical address of follow-up clinic | 1 | 0.17% |
| record | followupprovider: Name of provider for follow-up visit | 1 | 0.17% |
| record | patient_mrn: Unique medical record number identifying the patient | 1 | 0.17% |
| record | treatments: Prescribed medications or therapies during stay | 1 | 0.17% |
| record | discharge_instructions: Written guidance given to patient post-discharge | 1 | 0.17% |
| record | dates | 1 | 0.17% |
| record | registration_cap | 1 | 0.17% |
| record | early_bird_deadline | 1 | 0.17% |
| record | panel_topics | 1 | 0.17% |
| record | performance_reason | 1 | 0.17% |
| record | change | 1 | 0.17% |
| record | stock_symbol | 1 | 0.17% |
| record | admit_diagnosis | 1 | 0.17% |
| record | lengthofstay | 1 | 0.17% |
| record | discharge_condition | 1 | 0.17% |
| record | recommendations | 1 | 0.17% |
| record | outlet | 1 | 0.17% |
| record | symbols | 1 | 0.17% |
| record | determiners | 1 | 0.17% |
| record | verbs_past_participle | 1 | 0.17% |
| record | verbs_past_tense | 1 | 0.17% |
| record | prepositions | 1 | 0.17% |
| record | punctuation | 1 | 0.17% |
| record | possessive_endings | 1 | 0.17% |
| record | adjectives | 1 | 0.17% |
| record | singular_nouns | 1 | 0.17% |
| record | performance_time | 1 | 0.17% |
| record | audience_rating | 1 | 0.17% |
| record | festival_id | 1 | 0.17% |
| record | ticket_price | 1 | 0.17% |
| record | spoofed_domain | 1 | 0.17% |
| record | target_department | 1 | 0.17% |
| record | risk_classification | 1 | 0.17% |
| record | incident_type | 1 | 0.17% |
| record | disruption_duration | 1 | 0.17% |
| record | funding_grant | 1 | 0.17% |
| record | outcome_measure | 1 | 0.17% |
| record | intervention | 1 | 0.17% |
| record | office | 1 | 0.17% |
| record | update_notes | 1 | 0.17% |
| record | founder_name | 1 | 0.17% |
| record | founding_date | 1 | 0.17% |
| record | initial_funding | 1 | 0.17% |
| record | investment_source | 1 | 0.17% |
| record | industry | 1 | 0.17% |
| record | early_challenges | 1 | 0.17% |
| record | exit_status | 1 | 0.17% |
| record | dosage_form: Physical form of the drug such as tablet or injection | 1 | 0.17% |
| record | routeofadministration: Method by which the drug is administered | 1 | 0.17% |
| record | drug: Name of the medication as listed in the database | 1 | 0.17% |
| record | approved_date: Date the drug received regulatory approval | 1 | 0.17% |
| record | patient_population: Group of patients for whom the drug is intended | 1 | 0.17% |
| record | baths | 1 | 0.17% |
| record | beds | 1 | 0.17% |
| record | note | 1 | 0.17% |
| record | isps | 1 | 0.17% |
| record | software_versions | 1 | 0.17% |
| record | treatment_plan | 1 | 0.17% |
| record | treatment_status | 1 | 0.17% |
| record | investor_name | 1 | 0.17% |
| record | asteroid.name_latin | 1 | 0.17% |
| record | asteroid.asteroid_type | 1 | 0.17% |
| record | asteroid.named_after.person.name | 1 | 0.17% |
| record | asteroid.named_after.person.name_native | 1 | 0.17% |
| record | asteroid.named_after.person.birth_year | 1 | 0.17% |
| record | asteroid.named_after.person.death_year | 1 | 0.17% |
| record | asteroid.named_after.person.occupation | 1 | 0.17% |
| record | asteroid.named_after.person.awards | 1 | 0.17% |
| record | magazine.article.page_breakdown.2.page_number | 1 | 0.17% |
| record | magazine.article.page_breakdown.9.advertising_percentage | 1 | 0.17% |
| record | magazine.article.page_breakdown.12.page_number | 1 | 0.17% |
| record | magazine.article.page_breakdown.2.advertising_percentage | 1 | 0.17% |
| record | magazine.article.page_breakdown.9.page_number | 1 | 0.17% |
| record | magazine.article.page_breakdown.3.advertisers | 1 | 0.17% |
| record | magazine.article.page_breakdown.16.advertising_percentage | 1 | 0.17% |
| record | magazine.article.page_breakdown.13.advertisers | 1 | 0.17% |
| record | magazine.article.page_breakdown.12.advertising_percentage | 1 | 0.17% |
| record | magazine.article.page_breakdown.5.advertising_percentage | 1 | 0.17% |
| record | magazine.article.page_breakdown.5.page_number | 1 | 0.17% |
| record | magazine.article.page_breakdown.14.advertisers | 1 | 0.17% |
| record | magazine.article.page_breakdown.13.advertising_percentage | 1 | 0.17% |
| record | magazine.article.page_breakdown.1.advertising_percentage | 1 | 0.17% |
| record | magazine.article.page_breakdown.8.advertisers | 1 | 0.17% |
| record | magazine.mentioned_article.author | 1 | 0.17% |
| record | magazine.article.page_breakdown.7.page_number | 1 | 0.17% |
| record | magazine.article.page_breakdown.10.page_number | 1 | 0.17% |
| record | magazine.name | 1 | 0.17% |
| record | magazine.article.page_breakdown.6.advertisers | 1 | 0.17% |
| record | magazine.article.page_breakdown.0.page_number | 1 | 0.17% |
| record | magazine.article.page_breakdown.2.advertisers | 1 | 0.17% |
| record | magazine.article.page_breakdown.5.advertisers | 1 | 0.17% |
| record | magazine.article.page_breakdown.15.advertisers | 1 | 0.17% |
| record | magazine.article.page_breakdown.4.page_number | 1 | 0.17% |
| record | magazine.article.page_breakdown.9.advertisers | 1 | 0.17% |
| record | magazine.article.page_breakdown.16.page_number | 1 | 0.17% |
| record | magazine.article.page_breakdown.10.advertisers | 1 | 0.17% |
| record | magazine.article.title | 1 | 0.17% |
| record | magazine.article.content_percentage | 1 | 0.17% |
| record | magazine.mentioned_article.year | 1 | 0.17% |
| record | magazine.article.page_breakdown.13.page_number | 1 | 0.17% |
| record | magazine.article.page_breakdown.3.page_number | 1 | 0.17% |
| record | magazine.article.page_breakdown.8.advertising_percentage | 1 | 0.17% |
| record | magazine.article.page_breakdown.0.advertisers | 1 | 0.17% |
| record | magazine.article.advertising_percentage | 1 | 0.17% |
| record | magazine.article.page_breakdown.14.advertising_percentage | 1 | 0.17% |
| record | magazine.article.page_breakdown.8.page_number | 1 | 0.17% |
| record | sec_id | 1 | 0.17% |
| record | study.title | 1 | 0.17% |
| record | study.quotes | 1 | 0.17% |
| record | study.key_findings | 1 | 0.17% |
| record | study.period.end_year | 1 | 0.17% |
| record | study.author.name | 1 | 0.17% |
| record | study.author.affiliation | 1 | 0.17% |
| record | search_result_page.search_query | 1 | 0.17% |
| record | search_result_page.website_name | 1 | 0.17% |
| record | initial_text.negations | 1 | 0.17% |
| record | model | 1 | 0.17% |
| record | user_rating | 1 | 0.17% |
| record | album_title | 1 | 0.17% |
| record | artist | 1 | 0.17% |
| record | birth_information.birth_year | 1 | 0.17% |
| record | profession | 1 | 0.17% |
| record | biography.education.0.teacher | 1 | 0.17% |
| record | biography.education.4.institution | 1 | 0.17% |
| record | biography.awards.0.award_name | 1 | 0.17% |
| record | biography.concert_tours | 1 | 0.17% |
| record | biography.education.4.teacher | 1 | 0.17% |
| record | biography.collaborations | 1 | 0.17% |
| record | biography.education.3.institution | 1 | 0.17% |
| record | biography.education.2.teacher | 1 | 0.17% |
| record | biography.teaching_position.institution | 1 | 0.17% |
| record | biography.teaching_position.mentor | 1 | 0.17% |
| record | birth_information.birth_place.state | 1 | 0.17% |
| record | birth_information.birth_place.county | 1 | 0.17% |
| record | symptom | 1 | 0.17% |
| record | report_date | 1 | 0.17% |
| record | type | 1 | 0.17% |
| record | force_type | 1 | 0.17% |
| record | material_efficiency | 1 | 0.17% |
| record | file_id | 1 | 0.17% |
| record | jurisdiction | 1 | 0.17% |
| record | insurer | 1 | 0.17% |
| record | premium | 1 | 0.17% |
| record | policy_type | 1 | 0.17% |
| record | submission_date | 1 | 0.17% |
| record | research_submission | 1 | 0.17% |
| record | assessment_name | 1 | 0.17% |
| record | publication_code | 1 | 0.17% |
| record | patient_case.clinical_presentation.medical_history.0.status | 1 | 0.17% |
| record | patient_case.demographics.gender | 1 | 0.17% |
| record | disease_profile.genetic_mapping.chromosome_location | 1 | 0.17% |
| record | patient_case.clinical_presentation.skin_lesions.characteristics | 1 | 0.17% |
| record | disease_profile.abbreviation | 1 | 0.17% |
| record | disease_profile.name | 1 | 0.17% |
| record | patient_case.clinical_presentation.medical_history.0.condition | 1 | 0.17% |
| record | report_context.geographic_significance | 1 | 0.17% |
| record | patient_case.genetic_analysis.mutation_details.dna_nomenclature | 1 | 0.17% |
| record | patient_case.genetic_analysis.mutation_details.type | 1 | 0.17% |
| record | report_context.epidemiology_notes | 1 | 0.17% |
| record | patient_case.genetic_analysis.mutation_details.affected_protein | 1 | 0.17% |
| record | patient_case.clinical_presentation.skin_lesions.description | 1 | 0.17% |
| record | patient_case.clinical_presentation.skin_lesions.locations | 1 | 0.17% |
| record | patient_case.clinical_presentation.skin_lesions.symptoms | 1 | 0.17% |
| record | patient_case.clinical_presentation.skin_lesions.size_range.min_mm | 1 | 0.17% |
| record | patient_case.genetic_analysis.mutation_details.functional_effect | 1 | 0.17% |
| record | patient_case.genetic_analysis.mutation_details.exon_location | 1 | 0.17% |
| record | production_location | 1 | 0.17% |
| record | attendees | 1 | 0.17% |
| record | lead_person | 1 | 0.17% |
| record | legal_counsel | 1 | 0.17% |
| record | size_and_fit.measurements.2.unit | 1 | 0.17% |
| record | size_and_fit.model_size | 1 | 0.17% |
| record | size_and_fit.measurements.2.type | 1 | 0.17% |
| record | size_and_fit.size_guide | 1 | 0.17% |
| record | size_and_fit.measurements.1.unit | 1 | 0.17% |
| record | size_and_fit.model_height.value | 1 | 0.17% |
| record | size_and_fit.measurements.2.value | 1 | 0.17% |
| record | price.installment_option.currency | 1 | 0.17% |
| record | size_and_fit.length.value | 1 | 0.17% |
| record | size_and_fit.length.unit | 1 | 0.17% |
| record | size_and_fit.details | 1 | 0.17% |
| record | product.technology | 1 | 0.17% |
| record | context.problem | 1 | 0.17% |
| record | context.solution | 1 | 0.17% |
| record | product.intended_use | 1 | 0.17% |
| record | product.browser_compatibility | 1 | 0.17% |
| record | product.download_link | 1 | 0.17% |
| record | total_funding | 1 | 0.17% |
| record | lead_researcher | 1 | 0.17% |
| record | venue: Name of the stadium or venue where the event occurred | 1 | 0.17% |
| record | nation: Country of the athlete who won the medal | 1 | 0.17% |
| record | time: Official race time recorded in the event | 1 | 0.17% |
| record | medal: Type of medal won gold, silver, or bronze | 1 | 0.17% |
| record | relationships.0.entity | 1 | 0.17% |
| record | relationships.0.action | 1 | 0.17% |
| record | relationships.0.type | 1 | 0.17% |
| record | date: Date when the reaction was reported | 1 | 0.17% |
| record | reaction: Observed adverse effect or symptom | 1 | 0.17% |
| record | status: Current status of the reaction | 1 | 0.17% |
| record | drug: Name and dosage form of the medication taken | 1 | 0.17% |
| record | duration: Number of days the treatment lasted for the reaction | 1 | 0.17% |
| record | patient_id: Unique identifier for the patient in the record | 1 | 0.17% |
| record | restaurant_name | 1 | 0.17% |
| record | capacity | 1 | 0.17% |
| record | sponsor | 1 | 0.17% |
| record | cas_number | 1 | 0.17% |
| record | emergency_contact | 1 | 0.17% |
| record | merchantcategorycode | 1 | 0.17% |
| record | eps | 1 | 0.17% |
| record | net_margin | 1 | 0.17% |
| record | call_duration | 1 | 0.17% |
| record | investor_list | 1 | 0.17% |
| record | briefing_event | 1 | 0.17% |
| record | follow_up_interest | 1 | 0.17% |
| record | pilot_hubs | 1 | 0.17% |
| record | claim_id: Unique identifier for each claim | 1 | 0.17% |
| record | patient_address: Full address including street, apt, city, state, zip | 1 | 0.17% |
| record | diagnosis_code: Standard code indicating the patient's condition | 1 | 0.17% |
| record | treatment_type: Type of medical treatment or procedure | 1 | 0.17% |
| record | claim_amount: Total monetary value of the claim | 1 | 0.17% |
| record | insurance_provider: Name of the insurance company covering the claim | 1 | 0.17% |
| record | status: Current processing status of the claim | 1 | 0.17% |
| record | payment_method: Method used to process payment for the claim | 1 | 0.17% |
| record | deductiblestatus: Whether patient's deductible has been met | 1 | 0.17% |
| record | co_payment: Patient's required co-payment amount | 1 | 0.17% |
| record | notes: Additional remarks or clinical context about the claim | 1 | 0.17% |
| record | outcomes_and_findings.prognosis.critical_success_factors | 1 | 0.17% |
| record | study_details.cohort.condition_studied | 1 | 0.17% |
| record | pathophysiology.etiology | 1 | 0.17% |
| record | pathophysiology.mechanism.secondary_event.anatomical_locations | 1 | 0.17% |
| record | outcomes_and_findings.prognosis.visual_recovery_potential | 1 | 0.17% |
| record | hotel_name | 1 | 0.17% |
| record | passenger_city | 1 | 0.17% |
| record | incident_id | 1 | 0.17% |
| record | date_reported | 1 | 0.17% |
| record | responsible_engineer | 1 | 0.17% |
| record | environment | 1 | 0.17% |
| record | budget_remaining | 1 | 0.17% |
| record | actual_spend | 1 | 0.17% |
| record | remaining_budget | 1 | 0.17% |
| record | onboarding_sprint_completion | 1 | 0.17% |
| record | risk_code | 1 | 0.17% |
| record | backend_integration_status | 1 | 0.17% |
| record | isbn | 1 | 0.17% |
| record | publication_volume | 1 | 0.17% |
| record | publication_issue | 1 | 0.17% |
| record | family | 1 | 0.17% |
| record | distribution | 1 | 0.17% |
| record | scientific_name | 1 | 0.17% |
| record | key_metric | 1 | 0.17% |
| record | last_updated | 1 | 0.17% |
| record | event.person.name | 1 | 0.17% |
| record | event.new_club.contract_duration | 1 | 0.17% |
| record | event.person.age | 1 | 0.17% |
| record | event.new_club.name | 1 | 0.17% |
| record | event.opponent_club.name | 1 | 0.17% |
| record | event.person.position | 1 | 0.17% |
| record | event.person.role | 1 | 0.17% |
| record | name: Full name of the startup company | 1 | 0.17% |
| record | location: City and state where the company is headquartered | 1 | 0.17% |
| record | places | 1 | 0.17% |
| record | risk_rating | 1 | 0.17% |
| record | grade | 1 | 0.17% |
| record | arguments_for_using_gdz | 1 | 0.17% |
| record | model_id | 1 | 0.17% |
| record | output_power | 1 | 0.17% |
| record | funding: Total funding amount in millions | 1 | 0.17% |
| record | project_name: Name of the technology project | 1 | 0.17% |
| record | location: Geographic location of the project | 1 | 0.17% |
| record | project_id: Unique identifier assigned to the project | 1 | 0.17% |
| record | item | 1 | 0.17% |
| record | service_type | 1 | 0.17% |
| record | total_amount | 1 | 0.17% |
| record | company_size | 1 | 0.17% |
| record | bracket_id | 1 | 0.17% |
| record | result | 1 | 0.17% |
| record | teams | 1 | 0.17% |
| record | service_description | 1 | 0.17% |
| record | sender: Email of the person sending the confirmation or submission | 1 | 0.17% |
| record | recipient: Email of the person receiving the confirmation or submission | 1 | 0.17% |
| record | regulatory_status.recognition_type | 1 | 0.17% |
| record | announcement_metadata.source_url | 1 | 0.17% |
| record | announcement_metadata.coverage_provider | 1 | 0.17% |
| record | regulatory_status.review_status | 1 | 0.17% |
| record | announcement_metadata.date | 1 | 0.17% |
| record | announcement_metadata.agency | 1 | 0.17% |
| record | location: City and state where the role is based | 1 | 0.17% |
| record | company: Name of the company hiring | 1 | 0.17% |
| record | title: Job title such as Senior Data Analyst or Marketing Manager | 1 | 0.17% |
| record | tools: List of software or platforms used in the role | 1 | 0.17% |
| record | responsibilities: Key duties and tasks associated with the role | 1 | 0.17% |
| record | output.unit | 1 | 0.17% |
| record | period | 1 | 0.17% |
| record | location.distance | 1 | 0.17% |
| record | permit_holder | 1 | 0.17% |
| record | nepa_guidance | 1 | 0.17% |
| record | activity | 1 | 0.17% |
| record | additional_information | 1 | 0.17% |
| record | weather_conditions | 1 | 0.17% |
| record | league | 1 | 0.17% |
| record | location: City where the treaty was signed | 1 | 0.17% |
| record | signatories: List of officials who co-signed the treaty | 1 | 0.17% |
| record | name: Unique identifier of the treaty agreement | 1 | 0.17% |
| record | test_id | 1 | 0.17% |
| record | service_impact | 1 | 0.17% |
| record | outage_start | 1 | 0.17% |
| record | outage_end | 1 | 0.17% |
| record | outage_code | 1 | 0.17% |
| record | root_cause | 1 | 0.17% |
| record | document_details.series_title | 1 | 0.17% |
| record | document_details.description | 1 | 0.17% |
| record | document_details.oclc_number | 1 | 0.17% |
| record | submission_id: Unique identifier for the paper submission | 1 | 0.17% |
| record | author_name: Name of the paper's author | 1 | 0.17% |
| record | affiliation: Institutional affiliation of the author | 1 | 0.17% |
| record | paper_title: Title of the submitted research paper | 1 | 0.17% |
| record | submission_date: Date when the paper was submitted | 1 | 0.17% |
| record | word_count: Total number of words in the paper | 1 | 0.17% |
| record | fieldofstudy: Academic domain of the research | 1 | 0.17% |
| record | conference_name: Name of the conference where submitted | 1 | 0.17% |
| record | review_date: Date when the paper was reviewed | 1 | 0.17% |
| record | funding_source: Source of research funding | 1 | 0.17% |
| record | page_count: Number of pages in the paper | 1 | 0.17% |
| record | abstract_length: Length of the abstract in words | 1 | 0.17% |
| record | charge_amount | 1 | 0.17% |
| record | original_plan | 1 | 0.17% |
| record | problem_description | 1 | 0.17% |
| record | affected_network | 1 | 0.17% |
| record | access_method | 1 | 0.17% |
| record | affected_site | 1 | 0.17% |
| record | problem_start_time | 1 | 0.17% |
| record | alternative_access | 1 | 0.17% |
| record | event.description | 1 | 0.17% |
| record | event.organizer.name | 1 | 0.17% |
| record | organization_info.electricity_source | 1 | 0.17% |
| record | event.organizer.abbreviation | 1 | 0.17% |
| record | organization_info.funding | 1 | 0.17% |
| record | event.location.city | 1 | 0.17% |
| record | organization_info.facilities.generating_facilities | 1 | 0.17% |
| record | event.time.photo_op_end | 1 | 0.17% |
| record | event.activities | 1 | 0.17% |
| record | organization_info.website | 1 | 0.17% |
| record | event.name | 1 | 0.17% |
| record | event.time.photo_op_start | 1 | 0.17% |
| record | donations.recipient.type | 1 | 0.17% |
| record | event.time.end | 1 | 0.17% |
| record | event.location.address | 1 | 0.17% |
| record | articleTitle | 1 | 0.17% |
| record | useoffunds | 1 | 0.17% |
| record | geographic_focus | 1 | 0.17% |
| record | lead_investor | 1 | 0.17% |
| record | research.materials | 1 | 0.17% |
| record | research.experiment_details.speckle_details | 1 | 0.17% |
| record | research.experiment_details.quantum_state | 1 | 0.17% |
| record | research.experiment_details.outcome | 1 | 0.17% |
| record | research.temperature | 1 | 0.17% |
| record | research.researcher.name | 1 | 0.17% |
| record | research.focus | 1 | 0.17% |
| record | research.researcher.affiliation.name | 1 | 0.17% |
| record | research.title | 1 | 0.17% |
| record | po_reference | 1 | 0.17% |
| record | event_name: Title of the event as listed | 1 | 0.17% |
| record | organizer_name: Name of the person or group organizing the event | 1 | 0.17% |
| record | location: Full address including street, city, and state | 1 | 0.17% |
| record | event_date: Date when the event will take place | 1 | 0.17% |
| record | event_status: Current status of the event | 1 | 0.17% |
| record | attendee_count: Number of registered attendees | 1 | 0.17% |
| record | city: City where the event is held | 1 | 0.17% |
| record | event_code: Unique code identifying the event | 1 | 0.17% |
| record | ram | 1 | 0.17% |
| record | follow_up.average_duration.years | 1 | 0.17% |
| record | population.age_range.max | 1 | 0.17% |
| record | study_metadata.measurement_technology.full_name | 1 | 0.17% |
| record | statistical_results.diagnostic_metrics.sensitivity_at_threshold.t_score_threshold | 1 | 0.17% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.0.t_score | 1 | 0.17% |
| record | population.age_range.unit | 1 | 0.17% |
| record | study_metadata.measurement_technology.abbreviation | 1 | 0.17% |
| record | population.age_range.min | 1 | 0.17% |
| record | population.gender | 1 | 0.17% |
| record | statistical_results.diagnostic_metrics.sensitivity_at_threshold.percentage_of_fractures | 1 | 0.17% |
| record | statistical_results.predictive_model.value | 1 | 0.17% |
| record | conclusions | 1 | 0.17% |
| record | study_metadata.measurement_technology.anatomical_site | 1 | 0.17% |
| record | outcomes.fracture_events.0.count | 1 | 0.17% |
| record | study_metadata.study_type | 1 | 0.17% |
| record | reviewer_name: Full name of the reviewer | 1 | 0.17% |
| record | key_comment: Specific positive or negative observation about the stay | 1 | 0.17% |
| record | location: City and state where the hotel is located | 1 | 0.17% |
| record | review_id: Unique identifier for the review entry | 1 | 0.17% |
| record | holder_name | 1 | 0.17% |
| record | critic_name | 1 | 0.17% |
| record | review_source | 1 | 0.17% |
| record | office_address | 1 | 0.17% |
| record | critic_location | 1 | 0.17% |
| record | vaccine | 1 | 0.17% |
| record | dose | 1 | 0.17% |
| record | pathway_name | 1 | 0.17% |
| record | splicing_regulation.current_research_status | 1 | 0.17% |
| record | clinical_observations.activation_status | 1 | 0.17% |
| record | splicing_regulation.impact_on_pathway_components.0.regulatory_mode | 1 | 0.17% |
| record | splicing_regulation.impact_on_pathway_components.0.component_description | 1 | 0.17% |
| record | user_name: Name of the creator who posted the trend | 1 | 0.17% |
| record | video_id: Unique identifier for the video content | 1 | 0.17% |
| record | shares: Total number of shares the video received | 1 | 0.17% |
| record | comments: Total number of comments on the video | 1 | 0.17% |
| record | away_team: Name of the team playing away | 1 | 0.17% |
| record | away_score: Final score for the away team | 1 | 0.17% |
| record | game_id: Unique identifier for the game as listed in the schedule | 1 | 0.17% |
| record | guidelines.distinction_results_interpretation | 1 | 0.17% |
| record | optional_information | 1 | 0.17% |
| record | research_framework_items | 1 | 0.17% |
| record | publication_date.city | 1 | 0.17% |
| record | publication_date.month | 1 | 0.17% |
| record | guidelines.adherence | 1 | 0.17% |
| record | organization.name | 1 | 0.17% |
| record | other_youth_programs_with_congressional_charters | 1 | 0.17% |
| record | first_contact_resolution_rate | 1 | 0.17% |
| record | priority | 1 | 0.17% |
| record | assigned_agent | 1 | 0.17% |
| record | support_code | 1 | 0.17% |
| record | sender_name: Name of the sender in the correspondence | 1 | 0.17% |
| record | stock | 1 | 0.17% |
| record | units_sold | 1 | 0.17% |
| record | professions | 1 | 0.17% |
| record | location: Physical address where course is held | 1 | 0.17% |
| record | duration: Number of weeks the course runs | 1 | 0.17% |
| record | credits: Number of academic credits assigned to the course | 1 | 0.17% |
| record | cost: Total cost of enrollment in US dollars | 1 | 0.17% |
| record | passenger_address | 1 | 0.17% |
| record | departure_original | 1 | 0.17% |
| record | departure_actual | 1 | 0.17% |
| record | training_modules: List of specific training topics assigned to the employee | 1 | 0.17% |
| record | name: Full name of the new employee | 1 | 0.17% |
| record | employee_id: Unique identifier assigned to each new hire | 1 | 0.17% |
| record | physical_characteristics.observations.0.telescope.origin | 1 | 0.17% |
| record | discovery.date | 1 | 0.17% |
| record | asteroid_belt_classification | 1 | 0.17% |
| record | region: Geographic region where product is available | 1 | 0.17% |
| record | launch_date: Date the product was launched | 1 | 0.17% |
| record | product_id: Unique identifier for the LinkedIn product | 1 | 0.17% |
| record | target_audience: Primary user group for the product | 1 | 0.17% |
| record | support_agent | 1 | 0.17% |
| record | key_achievement | 1 | 0.17% |
| record | review_period | 1 | 0.17% |
| record | patient | 1 | 0.17% |
| record | provider_address | 1 | 0.17% |
| record | error.possible_causes | 1 | 0.17% |
| record | error.message | 1 | 0.17% |
| record | line1 | 1 | 0.17% |
| record | company_id | 1 | 0.17% |
| record | key_challenge | 1 | 0.17% |
| record | outcome | 1 | 0.17% |
| record | payment_method | 1 | 0.17% |
| record | municipality.name_local | 1 | 0.17% |
| record | municipality.location.region | 1 | 0.17% |
| record | municipality.location.country | 1 | 0.17% |
| record | demographics.population.0.count | 1 | 0.17% |
| record | demographics.population.0.year | 1 | 0.17% |
| record | guideline_id: Unique identifier for the guideline document | 1 | 0.17% |
| record | author: Name of the medical professional who authored the guideline | 1 | 0.17% |
| record | institution: Hospital or organization where author is affiliated | 1 | 0.17% |
| record | recommendation: Specific treatment or intervention advised | 1 | 0.17% |
| record | issue_year: Year the guideline was published or released | 1 | 0.17% |
| record | steps | 1 | 0.17% |
| record | drug_code | 1 | 0.17% |
| record | medicalrecordid | 1 | 0.17% |
| record | defense_systems | 1 | 0.17% |
| record | time_references | 1 | 0.17% |
| record | weapons | 1 | 0.17% |
| record | cities | 1 | 0.17% |
| record | service_type: Type of service impacted such as broadband or mobile | 1 | 0.17% |
| record | outage_end: Time when the outage ended in HH:MM AM/PM format | 1 | 0.17% |
| record | service_code: Internal code identifying the service type | 1 | 0.17% |
| record | ticket_number: Unique identifier for the outage ticket | 1 | 0.17% |
| record | outage_start: Time when the outage began in HH:MM AM/PM format | 1 | 0.17% |
| record | treatment_arm | 1 | 0.17% |
| record | contact_number | 1 | 0.17% |
| record | lastfollowup | 1 | 0.17% |
| record | traveler_name | 1 | 0.17% |
| record | hotel_address | 1 | 0.17% |
| record | recharge_amount | 1 | 0.17% |
| record | bonus_amount | 1 | 0.17% |
| record | open_date | 1 | 0.17% |
| record | ticket_count | 1 | 0.17% |
| record | customer_id | 1 | 0.17% |
| record | announcement.divisions.0.star_of_the_week.player_name | 1 | 0.17% |
| record | announcement.presented_by | 1 | 0.17% |
| record | announcement.divisions.0.division_name | 1 | 0.17% |
| record | announcement.divisions.5.honorable_mentions.1.team | 1 | 0.17% |
| record | announcement.divisions.0.star_of_the_week.team | 1 | 0.17% |
| record | announcement.divisions.2.honorable_mentions.0.player_name | 1 | 0.17% |
| record | announcement.divisions.5.star_of_the_week.position | 1 | 0.17% |
| record | announcement.divisions.0.second_star.team | 1 | 0.17% |
| record | announcement.divisions.1.star_of_the_week.team | 1 | 0.17% |
| record | announcement.divisions.0.second_star.player_name | 1 | 0.17% |
| record | announcement.divisions.0.honorable_mentions.3.position | 1 | 0.17% |
| record | announcement.divisions.1.honorable_mentions.0.team | 1 | 0.17% |
| record | announcement.divisions.5.honorable_mentions.2.position | 1 | 0.17% |
| record | announcement.divisions.5.second_star.team | 1 | 0.17% |
| record | announcement.divisions.5.honorable_mentions.0.team | 1 | 0.17% |
| record | announcement.divisions.0.honorable_mentions.7.player_name | 1 | 0.17% |
| record | announcement.divisions.2.honorable_mentions.4.position | 1 | 0.17% |
| record | announcement.divisions.3.second_star.position | 1 | 0.17% |
| record | announcement.divisions.0.honorable_mentions.0.player_name | 1 | 0.17% |
| record | announcement.divisions.0.honorable_mentions.4.position | 1 | 0.17% |
| record | announcement.divisions.1.star_of_the_week.stats | 1 | 0.17% |
| record | announcement.divisions.0.honorable_mentions.5.team | 1 | 0.17% |
| record | announcement.divisions.2.division_name | 1 | 0.17% |
| record | announcement.divisions.4.honorable_mentions.2.team | 1 | 0.17% |
| record | announcement.divisions.0.honorable_mentions.7.position | 1 | 0.17% |
| record | announcement.divisions.0.honorable_mentions.1.player_name | 1 | 0.17% |
| record | announcement.divisions.3.honorable_mentions.1.player_name | 1 | 0.17% |
| record | announcement.divisions.4.honorable_mentions.0.player_name | 1 | 0.17% |
| record | announcement.divisions.4.honorable_mentions.1.player_name | 1 | 0.17% |
| record | announcement.divisions.0.honorable_mentions.1.team | 1 | 0.17% |
| record | announcement.divisions.1.honorable_mentions.0.position | 1 | 0.17% |
| record | announcement.divisions.3.star_of_the_week.position | 1 | 0.17% |
| record | announcement.divisions.5.star_of_the_week.player_name | 1 | 0.17% |
| record | announcement.divisions.2.honorable_mentions.1.position | 1 | 0.17% |
| record | announcement.divisions.0.honorable_mentions.2.player_name | 1 | 0.17% |
| record | announcement.divisions.1.star_of_the_week.position | 1 | 0.17% |
| record | announcement.divisions.3.honorable_mentions.0.player_name | 1 | 0.17% |
| record | announcement.divisions.0.second_star.position | 1 | 0.17% |
| record | announcement.divisions.0.honorable_mentions.0.position | 1 | 0.17% |
| record | announcement.divisions.1.star_of_the_week.player_name | 1 | 0.17% |
| record | announcement.divisions.3.division_name | 1 | 0.17% |
| record | announcement.divisions.5.honorable_mentions.1.position | 1 | 0.17% |
| record | announcement.divisions.4.second_star.team | 1 | 0.17% |
| record | announcement.divisions.2.honorable_mentions.0.team | 1 | 0.17% |
| record | announcement.divisions.3.second_star.team | 1 | 0.17% |
| record | announcement.divisions.4.division_name | 1 | 0.17% |
| record | announcement.divisions.1.honorable_mentions.1.team | 1 | 0.17% |
| record | announcement.divisions.0.honorable_mentions.4.player_name | 1 | 0.17% |
| record | announcement.divisions.2.honorable_mentions.0.position | 1 | 0.17% |
| record | announcement.divisions.0.honorable_mentions.7.team | 1 | 0.17% |
| record | announcement.divisions.0.honorable_mentions.6.player_name | 1 | 0.17% |
| record | announcement.divisions.1.second_star.player_name | 1 | 0.17% |
| record | announcement.divisions.2.honorable_mentions.2.position | 1 | 0.17% |
| record | announcement.week_ending | 1 | 0.17% |
| record | announcement.divisions.2.second_star.position | 1 | 0.17% |
| record | announcement.divisions.5.star_of_the_week.team | 1 | 0.17% |
| record | announcement.divisions.2.honorable_mentions.4.player_name | 1 | 0.17% |
| record | announcement.divisions.1.second_star.position | 1 | 0.17% |
| record | announcement.divisions.2.second_star.player_name | 1 | 0.17% |
| record | announcement.divisions.1.honorable_mentions.0.player_name | 1 | 0.17% |
| record | announcement.divisions.4.honorable_mentions.1.position | 1 | 0.17% |
| record | announcement.divisions.5.division_name | 1 | 0.17% |
| record | announcement.divisions.3.star_of_the_week.team | 1 | 0.17% |
| record | announcement.divisions.2.honorable_mentions.1.team | 1 | 0.17% |
| record | announcement.divisions.4.honorable_mentions.2.position | 1 | 0.17% |
| record | user: Name of the customer who provided the review | 1 | 0.17% |
| record | rating: Numerical rating from 1 to 5 given by the user | 1 | 0.17% |
| record | location: City and state or province where purchase occurred | 1 | 0.17% |
| record | disclosure_id | 1 | 0.17% |
| record | donor_name | 1 | 0.17% |
| record | facility: Facility name and address where device is located | 1 | 0.17% |
| record | device_id: Unique identifier for the medical device | 1 | 0.17% |
| record | customer_retention | 1 | 0.17% |
| record | net_income | 1 | 0.17% |
| record | ebit_margin | 1 | 0.17% |
| record | asteroid.dedication.person.nationality | 1 | 0.17% |
| record | asteroid.dedication.person.occupation | 1 | 0.17% |
| record | creative work | 1 | 0.17% |
| record | medical condition | 1 | 0.17% |
| record | medical procedure | 1 | 0.17% |
| record | subscriber_name | 1 | 0.17% |
| record | subscription_id | 1 | 0.17% |
| record | monthly_fee | 1 | 0.17% |
| record | labels | 1 | 0.17% |
| record | submission_id | 1 | 0.17% |
| record | spo2: Oxygen saturation percentage | 1 | 0.17% |
| record | blood_pressure: Blood pressure in format systolic/diastolic | 1 | 0.17% |
| record | heart_rate: Number of heartbeats per minute | 1 | 0.17% |
| record | address: Street address including city and zip code | 1 | 0.17% |
| record | company_id: Unique identifier assigned to each company | 1 | 0.17% |
| record | location: Geographic headquarters of the company | 1 | 0.17% |
| record | net_profit: Annual net profit in millions of dollars | 1 | 0.17% |
| record | ebitda_margin: Percentage of revenue that represents EBITDA | 1 | 0.17% |
| record | debttoebitda_ratio: Ratio of debt to EBITDA indicating leverage | 1 | 0.17% |
| record | current_ratio: Liquidity ratio measuring short-term obligations | 1 | 0.17% |
| record | operational_status: Current phase of business operations | 1 | 0.17% |
| record | record_metadata | natural/location ×13, natural/status ×12, natural/title ×9, natural/name ×9 |  |

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
