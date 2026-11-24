# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replay_eb21_rolesym | train | 29,276 | 29,020 | 203,325 | 14,551 | 25,456 | 1,648 | 17,077 | 22,095 |
| replay_eb21_rolesym | val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| replay_eb21_rolesym | test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## replay_eb21_rolesym

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 29,276 | 29,020 | 342 | 1223 | 20924 | 203,325 | 14,551 | 25,456 | 1,648 | 17,077 | 22,095 |
| val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 14,511 | 0 | 0 | 14,511 | 49.57% |
| zho | 11,161 | 0 | 0 | 11,161 | 38.12% |
| tur | 3,046 | 0 | 0 | 3,046 | 10.40% |
| und | 378 | 0 | 0 | 378 | 1.29% |
| fra | 25 | 0 | 0 | 25 | 0.09% |
| rus | 24 | 0 | 0 | 24 | 0.08% |
| ita | 17 | 0 | 0 | 17 | 0.06% |
| ces | 14 | 0 | 0 | 14 | 0.05% |
| por | 14 | 0 | 0 | 14 | 0.05% |
| hin | 12 | 0 | 0 | 12 | 0.04% |
| pol | 11 | 0 | 0 | 11 | 0.04% |
| kor | 11 | 0 | 0 | 11 | 0.04% |
| deu | 10 | 0 | 0 | 10 | 0.03% |
| nld | 9 | 0 | 0 | 9 | 0.03% |
| ara | 9 | 0 | 0 | 9 | 0.03% |
| ukr | 6 | 0 | 0 | 6 | 0.02% |
| spa | 6 | 0 | 0 | 6 | 0.02% |
| epo | 2 | 0 | 0 | 2 | 0.01% |
| pms | 1 | 0 | 0 | 1 | 0.00% |
| bel | 1 | 0 | 0 | 1 | 0.00% |
| bcl | 1 | 0 | 0 | 1 | 0.00% |
| wuu | 1 | 0 | 0 | 1 | 0.00% |
| nor | 1 | 0 | 0 | 1 | 0.00% |
| bre | 1 | 0 | 0 | 1 | 0.00% |
| cym | 1 | 0 | 0 | 1 | 0.00% |
| ilo | 1 | 0 | 0 | 1 | 0.00% |
| ind | 1 | 0 | 0 | 1 | 0.00% |
| fin | 1 | 0 | 0 | 1 | 0.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 0 |
| test_unique_inputs | 0 |
| train_records | 29,276 |
| train_unique_inputs | 29,020 |
| val_records | 0 |
| val_unique_inputs | 0 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | n/a | 203,325 / 0 |  |
| entities (per mention) | train / test | n/a | 203,325 / 0 |  |
| relations (per instance) | train / val | n/a | 14,551 / 0 |  |
| relations (per instance) | train / test | n/a | 14,551 / 0 |  |
| event types (per trigger mention) | train / val | n/a | 19,439 / 0 |  |
| event types (per trigger mention) | train / test | n/a | 19,439 / 0 |  |
| event roles (per argument) | train / val | n/a | 22,968 / 0 |  |
| event roles (per argument) | train / test | n/a | 22,968 / 0 |  |
| classifications (true_label) | train / val | n/a | 25,456 / 0 |  |
| classifications (true_label) | train / test | n/a | 25,456 / 0 |  |


### Events

**train.** 18,667 instances in 4,754 records, 0 triggerless. Trigger gold 17,077. Argument gold 22,095. Arguments per instance, mean 1.23. Trigger length in whitespace tokens, mean 1.38, single-token share 70.9%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Contact.Communicate | 1,579 | 8.12% |
| Experiment | 1,005 | 5.17% |
| Support | 719 | 3.70% |
| Sport.Compete | 430 | 2.21% |
| Catastrophe | 421 | 2.17% |
| Hostile_encounter | 383 | 1.97% |
| Attack | 368 | 1.89% |
| Competition | 365 | 1.88% |
| Causation | 340 | 1.75% |
| Process_start | 320 | 1.65% |
| Deploy | 299 | 1.54% |
| Life.Die | 294 | 1.51% |
| Personnel.StartPosition | 255 | 1.31% |
| Motion | 249 | 1.28% |
| match_result | 236 | 1.21% |
| Personnel.EndPosition | 221 | 1.14% |
| Killing | 210 | 1.08% |
| Conflict.Attack | 200 | 1.03% |
| Social_event | 199 | 1.02% |
| Process_end | 181 | 0.93% |
| Exhibit | 181 | 0.93% |
| Conquering | 180 | 0.93% |
| Statement | 179 | 0.92% |
| Contact.Meet | 172 | 0.88% |
| product_publish | 165 | 0.85% |
| Business.StartOrg | 161 | 0.83% |
| Arriving | 158 | 0.81% |
| Life.Injure | 158 | 0.81% |
| Transaction.TransferOwnership | 148 | 0.76% |
| Justice.Arrest | 142 | 0.73% |
| Coming_to_be | 141 | 0.73% |
| Self_motion | 140 | 0.72% |
| Bodily_harm | 138 | 0.71% |
| Destroying | 137 | 0.70% |
| Award.ReceiveAward | 136 | 0.70% |
| Death | 131 | 0.67% |
| Control | 130 | 0.67% |
| Hold | 128 | 0.66% |
| Damaging | 128 | 0.66% |
| Cause_change_of_position_on_a_scale | 128 | 0.66% |
| Cause_change_of_strength | 127 | 0.65% |
| Justice.ChargeIndict | 124 | 0.64% |
| Earnings_and_losses | 124 | 0.64% |
| Military_operation | 121 | 0.62% |
| Getting | 119 | 0.61% |
| Presence | 117 | 0.60% |
| Disaster.Accident | 115 | 0.59% |
| Becoming | 112 | 0.58% |
| Accident | 105 | 0.54% |
| Business.LaunchProduct | 105 | 0.54% |
| Transaction.TransferMoney | 104 | 0.54% |
| Creating | 102 | 0.52% |
| death | 100 | 0.51% |
| Preventing_or_letting | 98 | 0.50% |
| Know | 96 | 0.49% |
| Justice.TrialHearing | 91 | 0.47% |
| Escaping | 90 | 0.46% |
| Traveling | 89 | 0.46% |
| Motion_directional | 89 | 0.46% |
| Influence | 88 | 0.45% |
| Placing | 87 | 0.45% |
| resignation | 87 | 0.45% |
| Expressing_publicly | 87 | 0.45% |
| Request | 84 | 0.43% |
| Change | 84 | 0.43% |
| Change_of_leadership | 83 | 0.43% |
| Arranging | 83 | 0.43% |
| judicial_action_arrest | 80 | 0.41% |
| Participation | 80 | 0.41% |
| Action | 80 | 0.41% |
| Cause_to_be_included | 80 | 0.41% |
| Defending | 79 | 0.41% |
| Name_conferral | 77 | 0.40% |
| Communication | 76 | 0.39% |
| Life.BeBorn | 75 | 0.39% |
| Using | 74 | 0.38% |
| Recording | 72 | 0.37% |
| Giving | 71 | 0.37% |
| Judgment_communication | 69 | 0.35% |
| Conflict.Demonstrate | 68 | 0.35% |
| Supporting | 68 | 0.35% |
| Agree_or_refuse_to_act | 66 | 0.34% |
| Reporting | 64 | 0.33% |
| Injure | 61 | 0.31% |
| Movement.Arrive | 60 | 0.31% |
| Choosing | 59 | 0.30% |
| Justice.Sentence | 59 | 0.30% |
| Removing | 58 | 0.30% |
| Manoeuvre | 58 | 0.30% |
| Bringing | 53 | 0.27% |
| Supply | 53 | 0.27% |
| Research.Publish | 53 | 0.27% |
| Sending | 52 | 0.27% |
| championship_win | 51 | 0.26% |
| Government.Vote | 51 | 0.26% |
| Departing | 50 | 0.26% |
| Perception_active | 50 | 0.26% |
| product_release | 49 | 0.25% |
| Conflict | 49 | 0.25% |
| Disaster.NaturalDisaster | 49 | 0.25% |
| Medical.Treat | 48 | 0.25% |
| Use_firearm | 47 | 0.24% |
| Manufacturing | 47 | 0.24% |
| Tech.Release | 47 | 0.24% |
| Cause_to_make_progress | 47 | 0.24% |
| Dispersal | 47 | 0.24% |
| Justice.Sue | 46 | 0.24% |
| Movement.Transport | 46 | 0.24% |
| Response | 42 | 0.22% |
| Aiming | 42 | 0.22% |
| advancement | 42 | 0.22% |
| Come_together | 41 | 0.21% |
| Receiving | 41 | 0.21% |
| Building | 39 | 0.20% |
| marriage | 39 | 0.20% |
| Cause_to_amalgamate | 38 | 0.20% |
| product_recall | 38 | 0.20% |
| Assistance | 37 | 0.19% |
| Hindering | 37 | 0.19% |
| Research.Discover | 37 | 0.19% |
| Contact.Negotiate | 35 | 0.18% |
| Achieve | 35 | 0.18% |
| divorce | 35 | 0.18% |
| Deciding | 34 | 0.17% |
| price_limit_up | 34 | 0.17% |
| Fire | 32 | 0.16% |
| judicial_action_summon | 32 | 0.16% |
| Check | 32 | 0.16% |
| Transaction.Donate | 32 | 0.16% |
| Coming_to_believe | 30 | 0.15% |
| Criminal_investigation | 30 | 0.15% |
| Terrorism | 30 | 0.15% |
| product_delist | 29 | 0.15% |
| joined | 29 | 0.15% |
| Publishing | 29 | 0.15% |
| Besieging | 28 | 0.14% |
| Legal_rulings | 28 | 0.14% |
| judicial_action_fine | 28 | 0.14% |
| Arrest | 28 | 0.14% |
| Change_event_time | 27 | 0.14% |
| Becoming_a_member | 27 | 0.14% |
| Recovering | 27 | 0.14% |
| Collaboration | 27 | 0.14% |
| Scrutiny | 27 | 0.14% |
| Protest | 27 | 0.14% |
| Expansion | 27 | 0.14% |
| opening | 26 | 0.13% |
| Violence | 26 | 0.13% |
| Medical.Diagnose | 26 | 0.13% |
| Commerce_sell | 25 | 0.13% |
| Business.EndOrg | 25 | 0.13% |
| Government.EnactLaw | 25 | 0.13% |
| Convincing | 25 | 0.13% |
| Confronting_problem | 24 | 0.12% |
| Preserving | 24 | 0.12% |
| Quarreling | 24 | 0.12% |
| Disaster.Rescue | 24 | 0.12% |
| departure | 23 | 0.12% |
| Government.Protest | 23 | 0.12% |
| Openness | 23 | 0.12% |
| Life.Marry | 22 | 0.11% |
| Committing_crime | 22 | 0.11% |
| Warning | 21 | 0.11% |
| GiveUp | 21 | 0.11% |
| layoff | 21 | 0.11% |
| sale_acquisition | 21 | 0.11% |
| Connect | 21 | 0.11% |
| judicial_action_imprisonment | 21 | 0.11% |
| Sign_agreement | 21 | 0.11% |
| Bearing_arms | 20 | 0.10% |
| Being_in_operation | 20 | 0.10% |
| Research.Fund | 20 | 0.10% |
| Commitment | 19 | 0.10% |
| Legality | 19 | 0.10% |
| Personnel.Elect | 19 | 0.10% |
| Body_movement | 19 | 0.10% |
| Emptying | 18 | 0.09% |
| Testing | 18 | 0.09% |
| Telling | 18 | 0.09% |
| Life.Evacuate | 18 | 0.09% |
| Change_sentiment | 18 | 0.09% |
| Reforming_a_system | 17 | 0.09% |
| Personnel.Nominate | 16 | 0.08% |
| Reveal_secret | 16 | 0.08% |
| Filling | 16 | 0.08% |
| Traffic Accident | 16 | 0.08% |
| Business.DeclareBankruptcy | 15 | 0.08% |
| birthday_celebration | 15 | 0.08% |
| Expend_resource | 15 | 0.08% |
| Justice.Appeal | 15 | 0.08% |
| competition_ban | 15 | 0.08% |
| Surrounding | 15 | 0.08% |
| judicial_action_report | 15 | 0.08% |
| competition_withdrawal | 15 | 0.08% |
| Business.MergeOrg | 15 | 0.08% |
| Rewards_and_punishments | 15 | 0.08% |
| Writing | 15 | 0.08% |
| Temporary_stay | 14 | 0.07% |
| apologize_to | 14 | 0.07% |
| breakup | 14 | 0.07% |
| product_award | 14 | 0.07% |
| judicial_action_hearing | 14 | 0.07% |
| financing | 14 | 0.07% |
| Research | 14 | 0.07% |
| Surrendering | 13 | 0.07% |
| dissolution | 13 | 0.07% |
| Tech.Breach | 13 | 0.07% |
| birth | 13 | 0.07% |
| Disaster.Outbreak | 13 | 0.07% |
| judicial_action_prosecution | 13 | 0.07% |
| Revenge | 13 | 0.07% |
| praise | 13 | 0.07% |
| Commerce_pay | 13 | 0.07% |
| Government.Sanction | 12 | 0.06% |
| price_increase | 12 | 0.06% |
| Life.Divorce | 12 | 0.06% |
| dismissal | 12 | 0.06% |
| Movement.Migrate | 12 | 0.06% |
| proposal | 11 | 0.06% |
| Wearing | 11 | 0.06% |
| Create_artwork | 11 | 0.06% |
| Rescuing | 11 | 0.06% |
| Resolve_problem | 11 | 0.06% |
| Patrolling | 11 | 0.06% |
| Collapse | 11 | 0.06% |
| retirement | 10 | 0.05% |
| Earthquake | 10 | 0.05% |
| Employment | 10 | 0.05% |
| closing | 10 | 0.05% |
| Forming_relationships | 10 | 0.05% |
| Rite | 10 | 0.05% |
| Commerce_buy | 10 | 0.05% |
| suspension | 9 | 0.05% |
| strike | 9 | 0.05% |
| Cure | 9 | 0.05% |
| Adducing | 9 | 0.05% |
| Award | 9 | 0.05% |
| Cost | 9 | 0.05% |
| price_decrease | 9 | 0.05% |
| thank | 8 | 0.04% |
| judicial_action_filing | 8 | 0.04% |
| Plane Crash | 8 | 0.04% |
| Hiding_objects | 8 | 0.04% |
| Submitting_documents | 8 | 0.04% |
| ipo | 8 | 0.04% |
| Exchange | 8 | 0.04% |
| Business.Strike | 8 | 0.04% |
| Imposing_obligation | 7 | 0.04% |
| Explosion | 7 | 0.04% |
| Lighting | 7 | 0.04% |
| engagement | 7 | 0.04% |
| Flood | 7 | 0.04% |
| Business.Layoff | 7 | 0.04% |
| GetReady | 7 | 0.04% |
| Robbery | 7 | 0.04% |
| wedding | 7 | 0.04% |
| visit_on_set | 6 | 0.03% |
| Having_or_lacking_access | 6 | 0.03% |
| Releasing | 6 | 0.03% |
| Kidnapping | 6 | 0.03% |
| Justice.Fine | 6 | 0.03% |
| Justice.Acquit | 6 | 0.03% |
| Prison | 5 | 0.03% |
| pregnancy | 5 | 0.03% |
| Containing | 5 | 0.03% |
| Labeling | 5 | 0.03% |
| Education_teaching | 5 | 0.03% |
| rate_increase | 5 | 0.03% |
| Limiting | 5 | 0.03% |
| Scouring | 5 | 0.03% |
| meet_with | 5 | 0.03% |
| contract_termination | 5 | 0.03% |
| price_limit_down | 5 | 0.03% |
| lost_contact | 4 | 0.02% |
| Ratification | 4 | 0.02% |
| march | 4 | 0.02% |
| Carry_goods | 4 | 0.02% |
| infidelity | 4 | 0.02% |
| Incident | 3 | 0.02% |
| Transaction.Lend | 3 | 0.02% |
| Movement.Deport | 3 | 0.02% |
| Justifying | 3 | 0.02% |
| Conflict.Ceasefire | 3 | 0.02% |
| rate_decrease | 3 | 0.02% |
| Renting | 2 | 0.01% |
| Practice | 1 | 0.01% |
| Breathing | 1 | 0.01% |
| Suspicion | 1 | 0.01% |

**val.** No event instances.

**test.** No event instances.

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 22,968 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.83 | 31.76% | 1.93 | 32.40% | 0.03% | 0 | 22,968 | 0 | 19,439 |
| val | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |
| test | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 18,667 | 2.64% | 13,642 | 13.38% | 12.89% | 2.24% | 1.18% | 14.47% |
| val | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |
| test | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 493 | 1,215 | 14,442 |
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
| person | 18,103 | 8.90% |
| location | 11,134 | 5.48% |
| date | 9,301 | 4.57% |
| Location | 9,049 | 4.45% |
| Date | 7,936 | 3.90% |
| cardinal | 6,436 | 3.17% |
| organization | 5,787 | 2.85% |
| Organization | 4,644 | 2.28% |
| StartDate | 4,045 | 1.99% |
| Product | 3,812 | 1.87% |
| job title | 3,752 | 1.85% |
| EquityHolder | 3,560 | 1.75% |
| StockCode | 3,198 | 1.57% |
| sports team | 3,163 | 1.56% |
| Casualties and Losses | 3,018 | 1.48% |
| money | 2,822 | 1.39% |
| StockAbbr | 2,675 | 1.32% |
| CompanyName | 2,671 | 1.31% |
| geopolitical entity | 2,645 | 1.30% |
| company | 2,516 | 1.24% |
| PledgedShares | 2,516 | 1.24% |
| Cause | 2,235 | 1.10% |
| Pledgee | 2,061 | 1.01% |
| TotalHoldingShares | 2,056 | 1.01% |
| TradedShares | 1,879 | 0.92% |
| media outlet | 1,872 | 0.92% |
| TotalHoldingRatio | 1,833 | 0.90% |
| Quantity | 1,815 | 0.89% |
| Person | 1,639 | 0.81% |
| percent | 1,500 | 0.74% |
| facility | 1,483 | 0.73% |
| Reduction Amount | 1,453 | 0.71% |
| TotalPledgedShares | 1,415 | 0.70% |
| duration | 1,322 | 0.65% |
| product | 1,301 | 0.64% |
| age | 1,246 | 0.61% |
| quantity | 1,219 | 0.60% |
| Measurement | 1,216 | 0.60% |
| EndDate | 1,197 | 0.59% |
| ReleasedDate | 1,185 | 0.58% |
| Reducing Shareholder | 1,125 | 0.55% |
| time | 1,084 | 0.53% |
| government agency | 1,017 | 0.50% |
| occupation | 954 | 0.47% |
| Announcement Date | 891 | 0.44% |
| Company Name | 853 | 0.42% |
| work of art | 811 | 0.40% |
| Reduction Start Date | 787 | 0.39% |
| event | 766 | 0.38% |
| educational institution | 751 | 0.37% |
| Policy Content | 743 | 0.37% |
| Contest Participant | 639 | 0.31% |
| Money | 632 | 0.31% |
| Nationality | 604 | 0.30% |
| LaterHoldingShares | 599 | 0.29% |
| nationality | 593 | 0.29% |
| Event | 583 | 0.29% |
| Affected Areas | 521 | 0.26% |
| AveragePrice | 519 | 0.26% |
| Facility | 512 | 0.25% |
| Participants | 510 | 0.25% |
| pledge_amount | 497 | 0.24% |
| measurement | 486 | 0.24% |
| Weapon and Equippment | 480 | 0.24% |
| Areas affected | 467 | 0.23% |
| Attacker | 460 | 0.23% |
| Candidates and their Political Parties | 457 | 0.22% |
| Suspect | 450 | 0.22% |
| address | 433 | 0.21% |
| Affected Area | 426 | 0.21% |
| ordinal | 415 | 0.20% |
| Target | 407 | 0.20% |
| disease | 395 | 0.19% |
| crime | 389 | 0.19% |
| region | 386 | 0.19% |
| Protesters | 376 | 0.18% |
| Occupation | 372 | 0.18% |
| pledgor | 369 | 0.18% |
| Protest Reason | 368 | 0.18% |
| Deliberating Agency | 354 | 0.17% |
| Recipient | 337 | 0.17% |
| Accepting Court | 326 | 0.16% |
| Charged Crime | 322 | 0.16% |
| Disease | 318 | 0.16% |
| RepurchasedShares | 318 | 0.16% |
| Loss Amount | 317 | 0.16% |
| family relationship | 309 | 0.15% |
| sports competition | 306 | 0.15% |
| Game Name | 306 | 0.15% |
| Position | 302 | 0.15% |
| Method | 294 | 0.14% |
| Precautionary Measure | 292 | 0.14% |
| distance | 290 | 0.14% |
| Accident Investigator | 289 | 0.14% |
| HighestTradingPrice | 272 | 0.13% |
| Aid Agency | 261 | 0.13% |
| Storm Hit Location | 261 | 0.13% |
| road | 260 | 0.13% |
| protein | 258 | 0.13% |
| Investigative Agency | 255 | 0.13% |
| Person under Investigation | 253 | 0.12% |
| Individual_protein | 250 | 0.12% |
| Champions | 241 | 0.12% |
| Award | 238 | 0.12% |
| Belligerents | 234 | 0.12% |
| Scheduled Landing Place | 231 | 0.11% |
| film | 222 | 0.11% |
| The Charged Crime | 222 | 0.11% |
| law | 221 | 0.11% |
| Strike Reason | 220 | 0.11% |
| chemical | 219 | 0.11% |
| pledge_start_date | 216 | 0.11% |
| RepurchaseAmount | 216 | 0.11% |
| landmark | 215 | 0.11% |
| Summit Theme | 215 | 0.11% |
| Accused People | 210 | 0.10% |
| Flight No. | 204 | 0.10% |
| DiseaseOrPhenotypicFeature | 204 | 0.10% |
| Contracting Parties | 203 | 0.10% |
| Criminal Evidence | 201 | 0.10% |
| political party | 199 | 0.10% |
| Score | 197 | 0.10% |
| Confirmed/Infected Cases | 195 | 0.10% |
| ruling_date | 193 | 0.09% |
| political affiliation | 192 | 0.09% |
| brand | 191 | 0.09% |
| Accusation | 189 | 0.09% |
| award | 188 | 0.09% |
| Frozen Amount | 188 | 0.09% |
| GeneOrGeneProduct | 188 | 0.09% |
| vehicle | 187 | 0.09% |
| ClosingDate | 187 | 0.09% |
| Speaker | 185 | 0.09% |
| Winner | 184 | 0.09% |
| Aircraft Agency | 184 | 0.09% |
| Location/Hospital | 183 | 0.09% |
| Increase Amount | 183 | 0.09% |
| Anti-pollution People/Organizations | 181 | 0.09% |
| ethnic group | 179 | 0.09% |
| military rank | 175 | 0.09% |
| non-profit organization | 174 | 0.09% |
| Host | 174 | 0.09% |
| song | 172 | 0.08% |
| Arrest Location | 172 | 0.08% |
| Influence People | 172 | 0.08% |
| Time | 171 | 0.08% |
| Increasing Shareholder | 171 | 0.08% |
| era | 170 | 0.08% |
| Crew | 166 | 0.08% |
| Deceased | 160 | 0.08% |
| Police | 160 | 0.08% |
| Policy Name & Abbreviation | 160 | 0.08% |
| Death Cases | 156 | 0.08% |
| Number of Vehicles Involved in the Crash | 155 | 0.08% |
| academic degree | 153 | 0.08% |
| LegalInstitution | 153 | 0.08% |
| The Final Votes and Percentages | 153 | 0.08% |
| Judgement Result/Prison Term | 152 | 0.07% |
| Magnitude | 151 | 0.07% |
| Profession | 150 | 0.07% |
| Speaker Status | 148 | 0.07% |
| frozen_shareholder | 146 | 0.07% |
| team role | 143 | 0.07% |
| pledge_end_date | 142 | 0.07% |
| Strike Outcome | 142 | 0.07% |
| Outbreak Location | 140 | 0.07% |
| Economic Loss | 139 | 0.07% |
| military unit | 139 | 0.07% |
| Election Name | 139 | 0.07% |
| Resignated Person | 138 | 0.07% |
| Competition Items | 138 | 0.07% |
| Effective Date | 137 | 0.07% |
| ChemicalEntity | 136 | 0.07% |
| publication | 135 | 0.07% |
| OtherScientificTerm | 133 | 0.07% |
| Attending Hospital | 132 | 0.06% |
| FrozeShares | 131 | 0.06% |
| Appointee | 131 | 0.06% |
| software | 128 | 0.06% |
| Spectacle Location | 128 | 0.06% |
| Grades | 126 | 0.06% |
| Epicenter | 124 | 0.06% |
| medication | 123 | 0.06% |
| Survivors | 123 | 0.06% |
| Epidemic Data Issuing Agency | 122 | 0.06% |
| Acquiree | 121 | 0.06% |
| Time of the Case | 121 | 0.06% |
| Award Field | 119 | 0.06% |
| Damaged Facility | 119 | 0.06% |
| Pollution Source | 118 | 0.06% |
| Record-breaking Project | 116 | 0.06% |
| Resign Reason | 116 | 0.06% |
| Solution | 116 | 0.06% |
| Strike Agency | 115 | 0.06% |
| Agreement Content | 114 | 0.06% |
| Record Breaker | 112 | 0.06% |
| Artifacts and Their Chronology | 111 | 0.05% |
| temperature | 109 | 0.05% |
| Discover Location | 108 | 0.05% |
| Freeze Start Date | 107 | 0.05% |
| Policy Proposals | 106 | 0.05% |
| LowestTradingPrice | 104 | 0.05% |
| Spectacle Start Time | 103 | 0.05% |
| Increase Start Date | 103 | 0.05% |
| Types of the Spectacle | 101 | 0.05% |
| People | 100 | 0.05% |
| body of water | 100 | 0.05% |
| Court | 99 | 0.05% |
| News Release Agency | 99 | 0.05% |
| Achievement | 99 | 0.05% |
| name | 99 | 0.05% |
| Visitor | 98 | 0.05% |
| Prosecutor | 98 | 0.05% |
| Beneficiary | 98 | 0.05% |
| Death Reason | 97 | 0.05% |
| Husband | 97 | 0.05% |
| Spacecraft | 96 | 0.05% |
| Aid Supplies/Amount | 95 | 0.05% |
| Riot Reason | 95 | 0.05% |
| book | 94 | 0.05% |
| Historical Sites | 94 | 0.05% |
| regulatory body | 93 | 0.05% |
| holiday | 93 | 0.05% |
| Start Date | 93 | 0.05% |
| Sponsor | 93 | 0.05% |
| frequency | 92 | 0.05% |
| Employment Agency | 92 | 0.05% |
| Country Visited | 91 | 0.04% |
| Evidence | 91 | 0.04% |
| Passengers | 90 | 0.04% |
| Wife | 89 | 0.04% |
| pseudonym | 88 | 0.04% |
| Host Country | 88 | 0.04% |
| species | 86 | 0.04% |
| religious group | 86 | 0.04% |
| Strikers Status | 85 | 0.04% |
| Protest Slogan | 84 | 0.04% |
| Habitat | 84 | 0.04% |
| Affected Industries | 83 | 0.04% |
| Age | 82 | 0.04% |
| Symptoms | 81 | 0.04% |
| Agreement Name | 81 | 0.04% |
| Way for Spreading | 80 | 0.04% |
| weapon | 78 | 0.04% |
| Award Reason | 77 | 0.04% |
| Weapon | 77 | 0.04% |
| Government Reaction | 77 | 0.04% |
| People/Organization who predicted the disaster | 77 | 0.04% |
| weight | 76 | 0.04% |
| Number of Damaged Houses | 76 | 0.04% |
| Task | 75 | 0.04% |
| GeneProtein | 75 | 0.04% |
| Susceptible Population | 74 | 0.04% |
| symptom | 73 | 0.04% |
| organism | 72 | 0.04% |
| speed | 72 | 0.04% |
| Gene/protein/RNA | 72 | 0.04% |
| anatomical structure | 71 | 0.03% |
| Released People | 71 | 0.03% |
| Generic | 71 | 0.03% |
| Start Time | 71 | 0.03% |
| Microorganism | 71 | 0.03% |
| Arrest Time | 70 | 0.03% |
| Storm Hit Time | 70 | 0.03% |
| protein-family | 69 | 0.03% |
| Army | 69 | 0.03% |
| OrganismTaxon | 68 | 0.03% |
| regulation | 67 | 0.03% |
| honorific | 67 | 0.03% |
| End Time | 67 | 0.03% |
| Taking-off Place | 67 | 0.03% |
| Conflict Duration | 67 | 0.03% |
| Arrested | 67 | 0.03% |
| market index | 64 | 0.03% |
| compensation_amount | 64 | 0.03% |
| Funding | 64 | 0.03% |
| Material | 63 | 0.03% |
| Number of Evacuated People | 62 | 0.03% |
| Spokenmen | 62 | 0.03% |
| language | 61 | 0.03% |
| Head of Institution | 61 | 0.03% |
| Victim | 60 | 0.03% |
| area | 60 | 0.03% |
| Storm Name | 60 | 0.03% |
| Participating Countries | 60 | 0.03% |
| Aid Reason | 60 | 0.03% |
| Previous Record Holder | 59 | 0.03% |
| Last Job of the Appointee | 59 | 0.03% |
| Strikers | 59 | 0.03% |
| Party Responsible for Pollution | 58 | 0.03% |
| Research Agency | 58 | 0.03% |
| Launch Date | 57 | 0.03% |
| compensation_recipient | 56 | 0.03% |
| court case | 54 | 0.03% |
| video game | 54 | 0.03% |
| Organization Name | 54 | 0.03% |
| stock ticker | 53 | 0.03% |
| The Grades of the Previous Record Holder | 53 | 0.03% |
| Summit Name | 53 | 0.03% |
| Spacecraft Mission | 53 | 0.03% |
| Discover Time | 52 | 0.03% |
| medical procedure | 52 | 0.03% |
| Organization Industry | 52 | 0.03% |
| Related Rivers or Lakes | 52 | 0.03% |
| Witness | 51 | 0.03% |
| Archaeologist Organization | 51 | 0.03% |
| Boycotted Institutions | 51 | 0.03% |
| Defense Lawyer | 50 | 0.02% |
| fiscal quarter | 50 | 0.02% |
| Archaeologist | 50 | 0.02% |
| conference | 50 | 0.02% |
| Military Strength | 50 | 0.02% |
| senior_personnel | 49 | 0.02% |
| Number of Influenced People | 49 | 0.02% |
| senior_position | 48 | 0.02% |
| Candidate Scandal(Allegations of fraud, etc.) | 47 | 0.02% |
| Storm Warning Level | 47 | 0.02% |
| vessel | 46 | 0.02% |
| currency | 46 | 0.02% |
| criminal organization | 46 | 0.02% |
| Approver | 46 | 0.02% |
| Fine Reason | 45 | 0.02% |
| MVP | 45 | 0.02% |
| Maximum Wind Speed | 45 | 0.02% |
| Employed Institution | 44 | 0.02% |
| Election Goal | 44 | 0.02% |
| Spectacle Duration | 43 | 0.02% |
| Predecessor | 43 | 0.02% |
| Area Affected | 43 | 0.02% |
| End Date | 43 | 0.02% |
| Appointment Time | 41 | 0.02% |
| Start_Date | 41 | 0.02% |
| Poisoning Type | 41 | 0.02% |
| Symptom | 40 | 0.02% |
| Missings | 40 | 0.02% |
| Anatomy | 40 | 0.02% |
| Freeze End Date | 40 | 0.02% |
| The Agency Who Broke the Agreement | 40 | 0.02% |
| The Final Seats Result | 39 | 0.02% |
| Maximum Rainfall | 39 | 0.02% |
| Head of Investigation Team | 39 | 0.02% |
| CellType | 39 | 0.02% |
| description | 39 | 0.02% |
| Release Reason | 38 | 0.02% |
| Other Loss | 38 | 0.02% |
| Outbreak Date | 38 | 0.02% |
| Organization Members | 38 | 0.02% |
| sentence | 37 | 0.02% |
| Appointer | 37 | 0.02% |
| Areas Affected | 36 | 0.02% |
| Spokeswoman/Spokesman | 36 | 0.02% |
| Term of Office | 36 | 0.02% |
| Number of Destroyed Building | 36 | 0.02% |
| Goal | 36 | 0.02% |
| Volcano Name | 36 | 0.02% |
| Phenotype | 36 | 0.02% |
| Court Time | 35 | 0.02% |
| Pests | 35 | 0.02% |
| Bill Drafting Agency | 35 | 0.02% |
| dosage | 34 | 0.02% |
| Water Level | 34 | 0.02% |
| aircraft | 34 | 0.02% |
| Temporary Settlement | 34 | 0.02% |
| Carrier Rocket | 34 | 0.02% |
| Successor | 33 | 0.02% |
| Bank Name | 33 | 0.02% |
| Region | 33 | 0.02% |
| Turnout(who win who lost) | 33 | 0.02% |
| SequenceVariant | 33 | 0.02% |
| Cooperative Agency | 32 | 0.02% |
| Number of Rescued People | 32 | 0.02% |
| Regulatory Authority | 31 | 0.02% |
| Amount of Precipitation | 31 | 0.02% |
| Last Time the Record was Broken | 30 | 0.01% |
| airport | 30 | 0.01% |
| treaty | 30 | 0.01% |
| deadline | 30 | 0.01% |
| Illness | 30 | 0.01% |
| Treatment Method | 30 | 0.01% |
| operating system | 29 | 0.01% |
| id | 29 | 0.01% |
| Annoncement Date | 29 | 0.01% |
| DNA | 29 | 0.01% |
| Military Exercise | 29 | 0.01% |
| Death Time | 29 | 0.01% |
| Scale | 29 | 0.01% |
| Prosecution Lawyer | 28 | 0.01% |
| Launch Country | 28 | 0.01% |
| Casualities and Losses | 28 | 0.01% |
| Declarer | 28 | 0.01% |
| hardware component | 27 | 0.01% |
| trade union | 27 | 0.01% |
| Reasons for the Formation of the Historical sites | 27 | 0.01% |
| Casualty Count | 27 | 0.01% |
| Acquisition amount | 27 | 0.01% |
| Ways to Watch the Speech | 27 | 0.01% |
| Rescue Organizer | 27 | 0.01% |
| Judge | 26 | 0.01% |
| Launch Site | 26 | 0.01% |
| Countries Withdrawing from the Organization | 26 | 0.01% |
| Storm Direction | 26 | 0.01% |
| war | 25 | 0.01% |
| Age of the Appointee | 25 | 0.01% |
| Fined Agency | 25 | 0.01% |
| Penalty Amount | 25 | 0.01% |
| Projects | 25 | 0.01% |
| postal code | 24 | 0.01% |
| Perpetrator | 24 | 0.01% |
| Sentencing Location | 24 | 0.01% |
| protein-complex | 24 | 0.01% |
| UnfrozeDate | 24 | 0.01% |
| Acquirer | 24 | 0.01% |
| Fire Warning Level | 24 | 0.01% |
| Unemployed Rate | 24 | 0.01% |
| Country | 23 | 0.01% |
| Duration | 23 | 0.01% |
| Publication | 23 | 0.01% |
| Doctor and Medical Team | 23 | 0.01% |
| State Before Death | 23 | 0.01% |
| protein-domain | 23 | 0.01% |
| Damaged Property | 23 | 0.01% |
| Influenced Crops and Livelihood | 22 | 0.01% |
| Forecasting Agency | 22 | 0.01% |
| Release Time | 22 | 0.01% |
| Jail Time | 22 | 0.01% |
| Age of the Resignated Person | 22 | 0.01% |
| Voting method | 22 | 0.01% |
| Launch Result | 22 | 0.01% |
| Invited Person | 22 | 0.01% |
| Perpetrators | 22 | 0.01% |
| Investigating agency | 22 | 0.01% |
| Number of Trapped People | 22 | 0.01% |
| Death Rate | 22 | 0.01% |
| clinical trial | 21 | 0.01% |
| model number | 21 | 0.01% |
| Company Industry | 21 | 0.01% |
| Rescue Tool or Method | 21 | 0.01% |
| protocol | 20 | 0.01% |
| Lasting Time | 20 | 0.01% |
| Train Agency | 20 | 0.01% |
| Ship No. | 20 | 0.01% |
| Commanders and their Position | 20 | 0.01% |
| Number of Victims | 19 | 0.01% |
| Concept | 19 | 0.01% |
| Magnitude(Tsunami heights) | 19 | 0.01% |
| Spectacle End Time | 18 | 0.01% |
| Agreement Validity Period | 18 | 0.01% |
| CellLine | 18 | 0.01% |
| Economic loss | 18 | 0.01% |
| Prison Term | 18 | 0.01% |
| Group | 18 | 0.01% |
| Metric | 18 | 0.01% |
| The Sentence Claimed by the Prosecutor's Lawyer | 17 | 0.01% |
| Wedding Venue | 17 | 0.01% |
| Species | 17 | 0.01% |
| Withdraw Reason | 17 | 0.01% |
| Bankrupt Business | 17 | 0.01% |
| Disaster-Stricken Farmland | 17 | 0.01% |
| vaccine | 16 | 0.01% |
| algorithm | 16 | 0.01% |
| Electoral System | 16 | 0.01% |
| Transportation | 16 | 0.01% |
| Child | 16 | 0.01% |
| Marriage Duration | 16 | 0.01% |
| Shipwreck Reason | 16 | 0.01% |
| Ethnic Group | 16 | 0.01% |
| natural disaster | 15 | 0.01% |
| cell | 15 | 0.01% |
| Inauguration Time | 15 | 0.01% |
| Stolen Amount | 15 | 0.01% |
| Head of the Merged Organization | 15 | 0.01% |
| Animal | 15 | 0.01% |
| Economists who Predicted the Crisis | 15 | 0.01% |
| Alternate Landing Place | 14 | 0.01% |
| Merger Terms | 14 | 0.01% |
| Tear Up Reason | 14 | 0.01% |
| Storm Movement Speed | 14 | 0.01% |
| Complications | 14 | 0.01% |
| Countries Joined the Organization | 14 | 0.01% |
| Strike Industry | 14 | 0.01% |
| RNA | 14 | 0.01% |
| Tribe | 14 | 0.01% |
| programming language | 13 | 0.01% |
| contract | 13 | 0.01% |
| Best Way to Shoot | 13 | 0.01% |
| Storm Center Location | 13 | 0.01% |
| Head of the Institution | 13 | 0.01% |
| Object | 13 | 0.01% |
| Weapon Used | 13 | 0.01% |
| Gene | 13 | 0.01% |
| Join Conditions | 13 | 0.01% |
| Protein_family_or_group | 12 | 0.01% |
| Property Division | 12 | 0.01% |
| Number of People Hospitalized | 12 | 0.01% |
| Response Measures | 12 | 0.01% |
| Damaged Crops & Livestock | 12 | 0.01% |
| title | 12 | 0.01% |
| Military Unit | 12 | 0.01% |
| environment | 12 | 0.01% |
| Detention Start Time | 11 | 0.01% |
| tax | 11 | 0.01% |
| pathogen | 11 | 0.01% |
| reagent | 11 | 0.01% |
| experimental-construct | 11 | 0.01% |
| mountain | 11 | 0.01% |
| Astronauts | 11 | 0.01% |
| Number of Aftershocks | 11 | 0.01% |
| Substance | 11 | 0.01% |
| Announce Platform | 11 | 0.01% |
| Mission Duration | 11 | 0.01% |
| Service Years | 11 | 0.01% |
| experiment-tag | 11 | 0.01% |
| Organization Leader | 11 | 0.01% |
| itemName | 11 | 0.01% |
| Cured Cases | 10 | 0.00% |
| Other Impact | 10 | 0.00% |
| target | 10 | 0.00% |
| city | 10 | 0.00% |
| email | 10 | 0.00% |
| author | 10 | 0.00% |
| witness | 9 | 0.00% |
| license | 9 | 0.00% |
| Trapped Depth | 9 | 0.00% |
| Commanders of the Army | 9 | 0.00% |
| Process | 9 | 0.00% |
| The Worst-Hit Area | 9 | 0.00% |
| Compensation | 9 | 0.00% |
| state | 9 | 0.00% |
| People Casting Key Votes | 9 | 0.00% |
| Name | 9 | 0.00% |
| itemId | 9 | 0.00% |
| border | 8 | 0.00% |
| End_Date | 8 | 0.00% |
| Investigating Spokesperson | 8 | 0.00% |
| agent | 8 | 0.00% |
| type | 8 | 0.00% |
| Educational Stage | 8 | 0.00% |
| Influenced People | 8 | 0.00% |
| Agreement Members | 8 | 0.00% |
| Document | 8 | 0.00% |
| value | 8 | 0.00% |
| Component | 8 | 0.00% |
| Body Part | 8 | 0.00% |
| role | 8 | 0.00% |
| interest rate | 7 | 0.00% |
| Hostage | 7 | 0.00% |
| Sequelae | 7 | 0.00% |
| Warning Device | 7 | 0.00% |
| Postpone Reason | 7 | 0.00% |
| Recovered Amount | 7 | 0.00% |
| Tsunamis | 7 | 0.00% |
| Last Outbreak Time | 7 | 0.00% |
| phone | 7 | 0.00% |
| Technology | 7 | 0.00% |
| status | 7 | 0.00% |
| Special Medicine | 7 | 0.00% |
| Child Custody | 6 | 0.00% |
| Adjective | 6 | 0.00% |
| tags | 6 | 0.00% |
| Program | 6 | 0.00% |
| Invalid Date | 6 | 0.00% |
| Stage | 6 | 0.00% |
| Prison | 6 | 0.00% |
| zipCode | 6 | 0.00% |
| Taxon | 6 | 0.00% |
| Train No. | 6 | 0.00% |
| problem_class | 6 | 0.00% |
| constraint_property | 6 | 0.00% |
| researcher | 6 | 0.00% |
| Phenomenon | 6 | 0.00% |
| price | 6 | 0.00% |
| range | 6 | 0.00% |
| patent | 5 | 0.00% |
| financial instrument | 5 | 0.00% |
| api | 5 | 0.00% |
| notes | 5 | 0.00% |
| startDate | 5 | 0.00% |
| Death Age | 5 | 0.00% |
| Rescue Start Time | 5 | 0.00% |
| Crop | 5 | 0.00% |
| Provisional Government | 5 | 0.00% |
| Tool | 5 | 0.00% |
| process | 5 | 0.00% |
| topicsCovered | 5 | 0.00% |
| Description | 5 | 0.00% |
| City | 5 | 0.00% |
| Exit Conditions | 5 | 0.00% |
| streetAddress | 5 | 0.00% |
| url | 5 | 0.00% |
| rating | 5 | 0.00% |
| CellComponent | 5 | 0.00% |
| loss_function | 5 | 0.00% |
| supportedDevices | 5 | 0.00% |
| comment | 5 | 0.00% |
| diagnosis | 4 | 0.00% |
| security vulnerability | 4 | 0.00% |
| mineral | 4 | 0.00% |
| file format | 4 | 0.00% |
| verdict | 4 | 0.00% |
| gene | 4 | 0.00% |
| Production Cuts | 4 | 0.00% |
| endDate | 4 | 0.00% |
| Suspected Cases | 4 | 0.00% |
| level | 4 | 0.00% |
| Project | 4 | 0.00% |
| Food | 4 | 0.00% |
| Book | 4 | 0.00% |
| Head of the Government | 4 | 0.00% |
| The State of the Volcano (Dormant or Active) | 4 | 0.00% |
| synopsis | 4 | 0.00% |
| Live Broadcast Platform | 4 | 0.00% |
| height | 4 | 0.00% |
| width | 4 | 0.00% |
| phoneNumber | 4 | 0.00% |
| amino-acid | 4 | 0.00% |
| Trapped Days | 4 | 0.00% |
| Province | 4 | 0.00% |
| benefits | 4 | 0.00% |
| destination | 4 | 0.00% |
| chemical_compound | 4 | 0.00% |
| Wedding Dress Designer | 4 | 0.00% |
| unitOfMeasure | 4 | 0.00% |
| inputMaterialIds | 4 | 0.00% |
| fusion-protein | 4 | 0.00% |
| mutation | 4 | 0.00% |
| model | 4 | 0.00% |
| protein-motif | 4 | 0.00% |
| x | 4 | 0.00% |
| y | 4 | 0.00% |
| credit rating | 3 | 0.00% |
| database | 3 | 0.00% |
| voltage | 3 | 0.00% |
| framework | 3 | 0.00% |
| Ship | 3 | 0.00% |
| Website | 3 | 0.00% |
| Role | 3 | 0.00% |
| Procedure | 3 | 0.00% |
| Plant | 3 | 0.00% |
| Media | 3 | 0.00% |
| organelle | 3 | 0.00% |
| Registered Capital | 3 | 0.00% |
| zip | 3 | 0.00% |
| firstName | 3 | 0.00% |
| lastName | 3 | 0.00% |
| year | 3 | 0.00% |
| episodeNumber | 3 | 0.00% |
| airDate | 3 | 0.00% |
| month | 3 | 0.00% |
| day | 3 | 0.00% |
| materialsProvided | 3 | 0.00% |
| ZipCode | 3 | 0.00% |
| Email | 3 | 0.00% |
| Rock Type | 3 | 0.00% |
| Rock Category | 3 | 0.00% |
| capacity | 3 | 0.00% |
| interests | 3 | 0.00% |
| Rocket | 3 | 0.00% |
| Research Field | 3 | 0.00% |
| Religion | 3 | 0.00% |
| Virus | 3 | 0.00% |
| model_architecture | 3 | 0.00% |
| dataset | 3 | 0.00% |
| pharmaceutical_process | 3 | 0.00% |
| Tsunami Warning Level | 3 | 0.00% |
| unit | 3 | 0.00% |
| peptide | 3 | 0.00% |
| Reinforcement Material | 3 | 0.00% |
| Resin Type | 3 | 0.00% |
| Cost | 3 | 0.00% |
| chemical_group | 3 | 0.00% |
| physical_property | 3 | 0.00% |
| sectionName | 3 | 0.00% |
| deliveryDate | 3 | 0.00% |
| software_library | 3 | 0.00% |
| hardware | 3 | 0.00% |
| stationTypeId | 3 | 0.00% |
| outputMaterialId | 3 | 0.00% |
| Nation | 3 | 0.00% |
| Natural Feature | 3 | 0.00% |
| assay | 3 | 0.00% |
| street | 3 | 0.00% |
| chargerType | 3 | 0.00% |
| wattage | 3 | 0.00% |
| length | 3 | 0.00% |
| outputVoltage | 3 | 0.00% |
| outputCurrent | 3 | 0.00% |
| Tissue | 3 | 0.00% |
| Cell Type | 3 | 0.00% |
| due_date | 3 | 0.00% |
| item_id | 2 | 0.00% |
| allergies | 2 | 0.00% |
| Vaccine Research and Development Organization | 2 | 0.00% |
| Grammatical Concept | 2 | 0.00% |
| Test | 2 | 0.00% |
| durability | 2 | 0.00% |
| Element | 2 | 0.00% |
| min | 2 | 0.00% |
| max | 2 | 0.00% |
| Currency | 2 | 0.00% |
| reportDate | 2 | 0.00% |
| dateCompleted | 2 | 0.00% |
| Therapy | 2 | 0.00% |
| Subject | 2 | 0.00% |
| Level | 2 | 0.00% |
| Activity | 2 | 0.00% |
| Frequency | 2 | 0.00% |
| Economy Loss | 2 | 0.00% |
| releaseDate | 2 | 0.00% |
| amount | 2 | 0.00% |
| prerequisites | 2 | 0.00% |
| StreetAddress | 2 | 0.00% |
| State | 2 | 0.00% |
| TelephoneNumber | 2 | 0.00% |
| LanguageCode | 2 | 0.00% |
| floor | 2 | 0.00% |
| Hull Location | 2 | 0.00% |
| Hull Discovery Time | 2 | 0.00% |
| gender | 2 | 0.00% |
| field | 2 | 0.00% |
| classification | 2 | 0.00% |
| Language | 2 | 0.00% |
| keywords | 2 | 0.00% |
| categories | 2 | 0.00% |
| content | 2 | 0.00% |
| software_system | 2 | 0.00% |
| biological_property | 2 | 0.00% |
| Pathogen | 2 | 0.00% |
| companyName | 2 | 0.00% |
| itemQuantityOnHand | 2 | 0.00% |
| genres | 2 | 0.00% |
| createdAt | 2 | 0.00% |
| updatedAt | 2 | 0.00% |
| mathematical_object | 2 | 0.00% |
| formalism | 2 | 0.00% |
| DocumentDate | 2 | 0.00% |
| DocumentName | 2 | 0.00% |
| DocumentType | 2 | 0.00% |
| DocumentUrl | 2 | 0.00% |
| attachmentName | 2 | 0.00% |
| attachmentType | 2 | 0.00% |
| attachmentUrl | 2 | 0.00% |
| Storm Formation Location | 2 | 0.00% |
| How Many Times Get Married | 2 | 0.00% |
| Protein_complex | 2 | 0.00% |
| callId | 2 | 0.00% |
| callDate | 2 | 0.00% |
| molecular_orbital | 2 | 0.00% |
| researcher_group | 2 | 0.00% |
| institutional_unit | 2 | 0.00% |
| phases | 2 | 0.00% |
| Refuge | 2 | 0.00% |
| action | 2 | 0.00% |
| actionDueDate | 2 | 0.00% |
| actionStatus | 2 | 0.00% |
| actionResponsible | 2 | 0.00% |
| actionNotes | 2 | 0.00% |
| documentName | 2 | 0.00% |
| documentType | 2 | 0.00% |
| documentDate | 2 | 0.00% |
| documentRevision | 2 | 0.00% |
| documentLocation | 2 | 0.00% |
| supplierId | 2 | 0.00% |
| itemNumber | 2 | 0.00% |
| Cured Rate | 2 | 0.00% |
| protein-region | 2 | 0.00% |
| Political Faction | 2 | 0.00% |
| Civilization | 2 | 0.00% |
| Law | 2 | 0.00% |
| materialId | 2 | 0.00% |
| materialDescription | 2 | 0.00% |
| quantityRejected | 2 | 0.00% |
| correctiveAction | 2 | 0.00% |
| architecture_component | 2 | 0.00% |
| concept | 2 | 0.00% |
| comments | 2 | 0.00% |
| numberOfPorts | 2 | 0.00% |
| inputVoltage | 2 | 0.00% |
| additionalFeatures | 2 | 0.00% |
| numerical_method | 2 | 0.00% |
| scientific_framework | 2 | 0.00% |
| physical_quantity | 2 | 0.00% |
| department | 2 | 0.00% |
| allergen | 1 | 0.00% |
| licenseNumber | 1 | 0.00% |
| Linguistic Concept | 1 | 0.00% |
| Entity | 1 | 0.00% |
| Platform | 1 | 0.00% |
| protein-isoform | 1 | 0.00% |
| damage | 1 | 0.00% |
| defense | 1 | 0.00% |
| health | 1 | 0.00% |
| mana | 1 | 0.00% |
| strength | 1 | 0.00% |
| agility | 1 | 0.00% |
| intelligence | 1 | 0.00% |
| Diet | 1 | 0.00% |
| Ingredient | 1 | 0.00% |
| category | 1 | 0.00% |
| Hazard | 1 | 0.00% |
| roomType | 1 | 0.00% |
| numberOfGuests | 1 | 0.00% |
| confirmationNumber | 1 | 0.00% |
| securityCode | 1 | 0.00% |
| factoryName | 1 | 0.00% |
| complianceStatus | 1 | 0.00% |
| permitNumber | 1 | 0.00% |
| inspectionDate | 1 | 0.00% |
| inspectorName | 1 | 0.00% |
| Disorder | 1 | 0.00% |
| Disorder Group | 1 | 0.00% |
| Demographic | 1 | 0.00% |
| Toxin | 1 | 0.00% |
| seasonNumber | 1 | 0.00% |
| numberOfEpisodes | 1 | 0.00% |
| room | 1 | 0.00% |
| bio | 1 | 0.00% |
| expertise | 1 | 0.00% |
| maxAttendees | 1 | 0.00% |
| cancellationPolicy | 1 | 0.00% |
| deviceId | 1 | 0.00% |
| deviceType | 1 | 0.00% |
| deviceName | 1 | 0.00% |
| dataCollectionInterval | 1 | 0.00% |
| sensorId | 1 | 0.00% |
| sensorType | 1 | 0.00% |
| reportingInterval | 1 | 0.00% |
| Ship Agency | 1 | 0.00% |
| PlanType | 1 | 0.00% |
| ResiliencyLevel | 1 | 0.00% |
| ParentName | 1 | 0.00% |
| Natural Disaster | 1 | 0.00% |
| Factor | 1 | 0.00% |
| country | 1 | 0.00% |
| postalCode | 1 | 0.00% |
| depth | 1 | 0.00% |
| Rocket Stage | 1 | 0.00% |
| Disease Category | 1 | 0.00% |
| Clinical Syndrome | 1 | 0.00% |
| Clinical Sign | 1 | 0.00% |
| Drug | 1 | 0.00% |
| Medical Intervention | 1 | 0.00% |
| Dynasty | 1 | 0.00% |
| River Name | 1 | 0.00% |
| River | 1 | 0.00% |
| Empire | 1 | 0.00% |
| Field of Study | 1 | 0.00% |
| datePublished | 1 | 0.00% |
| dateModified | 1 | 0.00% |
| Biological Entity | 1 | 0.00% |
| podcastTitle | 1 | 0.00% |
| episodeTitle | 1 | 0.00% |
| Regulatory Body | 1 | 0.00% |
| Demographic Group | 1 | 0.00% |
| model_component | 1 | 0.00% |
| molecular_representation | 1 | 0.00% |
| representation | 1 | 0.00% |
| Regulation | 1 | 0.00% |
| Category | 1 | 0.00% |
| Chemical | 1 | 0.00% |
| companyAddress | 1 | 0.00% |
| companyCity | 1 | 0.00% |
| companyState | 1 | 0.00% |
| companyZip | 1 | 0.00% |
| numberOfDays | 1 | 0.00% |
| averageInventoryValue | 1 | 0.00% |
| finishedGoodsInventoryTurnover | 1 | 0.00% |
| trailerFormat | 1 | 0.00% |
| trailerQuality | 1 | 0.00% |
| thumbnailUrl | 1 | 0.00% |
| issueNumber | 1 | 0.00% |
| publisher | 1 | 0.00% |
| publicationDate | 1 | 0.00% |
| coverPrice | 1 | 0.00% |
| website | 1 | 0.00% |
| payRate | 1 | 0.00% |
| jobTitle | 1 | 0.00% |
| jobDescription | 1 | 0.00% |
| terminationTerms | 1 | 0.00% |
| additionalInfo | 1 | 0.00% |
| Statistic | 1 | 0.00% |
| Definition | 1 | 0.00% |
| alertId | 1 | 0.00% |
| productId | 1 | 0.00% |
| complexity_class | 1 | 0.00% |
| academic_field | 1 | 0.00% |
| Organization_Military | 1 | 0.00% |
| Address | 1 | 0.00% |
| Title | 1 | 0.00% |
| PhoneNumber | 1 | 0.00% |
| AdditionalNotes | 1 | 0.00% |
| machineId | 1 | 0.00% |
| maintenanceType | 1 | 0.00% |
| maintenanceDate | 1 | 0.00% |
| maintenanceDuration | 1 | 0.00% |
| maintenanceCost | 1 | 0.00% |
| performedBy | 1 | 0.00% |
| direction | 1 | 0.00% |
| formNumber | 1 | 0.00% |
| travelPurpose | 1 | 0.00% |
| passportNumber | 1 | 0.00% |
| totalValue | 1 | 0.00% |
| signature | 1 | 0.00% |
| Material Category | 1 | 0.00% |
| Material Component | 1 | 0.00% |
| Storm Formation Time | 1 | 0.00% |
| activationId | 1 | 0.00% |
| msisdn | 1 | 0.00% |
| customerId | 1 | 0.00% |
| activationDate | 1 | 0.00% |
| activationStatus | 1 | 0.00% |
| dataUsage | 1 | 0.00% |
| Number of Vaccinated People | 1 | 0.00% |
| element | 1 | 0.00% |
| computational_method | 1 | 0.00% |
| software_package | 1 | 0.00% |
| checklistId | 1 | 0.00% |
| checklistName | 1 | 0.00% |
| createDate | 1 | 0.00% |
| updateDate | 1 | 0.00% |
| card_number | 1 | 0.00% |
| expiration_date | 1 | 0.00% |
| cvv | 1 | 0.00% |
| completionDate | 1 | 0.00% |
| supplier | 1 | 0.00% |
| planNumber | 1 | 0.00% |
| planDate | 1 | 0.00% |
| planDueDate | 1 | 0.00% |
| planStatus | 1 | 0.00% |
| scheduleId | 1 | 0.00% |
| purchaseOrderId | 1 | 0.00% |
| scheduleDate | 1 | 0.00% |
| programming_language | 1 | 0.00% |
| metric | 1 | 0.00% |
| product_id | 1 | 0.00% |
| num_reviews | 1 | 0.00% |
| review_sentiment | 1 | 0.00% |
| parameter | 1 | 0.00% |
| changeType | 1 | 0.00% |
| flightNumber | 1 | 0.00% |
| Lawyer | 1 | 0.00% |
| Commodity | 1 | 0.00% |
| productionLineTypeId | 1 | 0.00% |
| reportId | 1 | 0.00% |
| supplierName | 1 | 0.00% |
| receivingDocumentNumber | 1 | 0.00% |
| rejectionReasonCode | 1 | 0.00% |
| rejectionReasonDescription | 1 | 0.00% |
| data_format | 1 | 0.00% |
| data_structure | 1 | 0.00% |
| publication_venue | 1 | 0.00% |
| Practice | 1 | 0.00% |
| Society | 1 | 0.00% |
| equipmentId | 1 | 0.00% |
| equipmentType | 1 | 0.00% |
| manufacturer | 1 | 0.00% |
| serialNumber | 1 | 0.00% |
| equipmentCondition | 1 | 0.00% |
| inspectionFrequency | 1 | 0.00% |
| inspectionLevel | 1 | 0.00% |
| start | 1 | 0.00% |
| end | 1 | 0.00% |
| rejectionLimit | 1 | 0.00% |
| preventiveAction | 1 | 0.00% |
| dateOfBirth | 1 | 0.00% |
| specialty | 1 | 0.00% |
| State of the Hull | 1 | 0.00% |
| reviewText | 1 | 0.00% |
| overall | 1 | 0.00% |
| productQuality | 1 | 0.00% |
| customerService | 1 | 0.00% |
| total | 1 | 0.00% |
| paymentTerms | 1 | 0.00% |
| origin | 1 | 0.00% |
| departure_time | 1 | 0.00% |
| seat_number | 1 | 0.00% |
| Rank | 1 | 0.00% |
| Doctor | 1 | 0.00% |
| listName | 1 | 0.00% |
| Field | 1 | 0.00% |
| software_code | 1 | 0.00% |
| astrophysical_object | 1 | 0.00% |
| Military_Rank | 1 | 0.00% |
| totalScore | 1 | 0.00% |
| Organ | 1 | 0.00% |
| Tissue Classification | 1 | 0.00% |
| Anatomical Space | 1 | 0.00% |
| Body Fluid | 1 | 0.00% |
| dependencies | 1 | 0.00% |
| budget line | 0 | 0.00% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| no relation | 1,431 | 9.83% |
| works_for | 841 | 5.78% |
| located_in | 417 | 2.87% |
| member_of | 388 | 2.67% |
| bind | 362 | 2.49% |
| headquartered_in | 295 | 2.03% |
| based_in | 293 | 2.01% |
| employer_of | 269 | 1.85% |
| leader_of | 262 | 1.80% |
| affiliated_with | 260 | 1.79% |
| PPI | 242 | 1.66% |
| instance of | 196 | 1.35% |
| Association | 192 | 1.32% |
| USED-FOR | 166 | 1.14% |
| competitor_of | 156 | 1.07% |
| founded | 138 | 0.95% |
| part_of | 134 | 0.92% |
| occupation | 133 | 0.91% |
| Lives_In | 119 | 0.82% |
| produced_by | 118 | 0.81% |
| country | 110 | 0.76% |
| shares border with | 107 | 0.74% |
| operates_in | 102 | 0.70% |
| participated_in | 102 | 0.70% |
| Positive_Correlation | 99 | 0.68% |
| colleague_of | 98 | 0.67% |
| family name | 95 | 0.65% |
| country of citizenship | 94 | 0.65% |
| has part(s) | 90 | 0.62% |
| given name | 87 | 0.60% |
| located in the administrative territorial entity | 81 | 0.56% |
| acquired | 78 | 0.54% |
| place of birth | 73 | 0.50% |
| member of | 72 | 0.49% |
| founded by | 60 | 0.41% |
| subsidiary_of | 59 | 0.41% |
| contains the administrative territorial entity | 59 | 0.41% |
| genre | 59 | 0.41% |
| represents | 57 | 0.39% |
| Negative_Correlation | 54 | 0.37% |
| supplier_of | 53 | 0.36% |
| reports_to | 52 | 0.36% |
| citizen_of | 51 | 0.35% |
| participant | 50 | 0.34% |
| platform | 50 | 0.34% |
| developed_by | 49 | 0.34% |
| educated at | 49 | 0.34% |
| parent_of | 49 | 0.34% |
| sibling | 46 | 0.32% |
| director | 45 | 0.31% |
| location | 45 | 0.31% |
| named after | 44 | 0.30% |
| CONJUNCTION | 44 | 0.30% |
| relative_of | 44 | 0.30% |
| owned by | 42 | 0.29% |
| language used | 42 | 0.29% |
| father | 41 | 0.28% |
| followed by | 38 | 0.26% |
| sport | 37 | 0.25% |
| board_member_of | 37 | 0.25% |
| replaces | 36 | 0.25% |
| present in work | 36 | 0.25% |
| HYPONYM-OF | 35 | 0.24% |
| performer | 35 | 0.24% |
| headquarters location | 35 | 0.24% |
| award received | 34 | 0.23% |
| significant event | 34 | 0.23% |
| participant in | 33 | 0.23% |
| different from | 33 | 0.23% |
| characters | 32 | 0.22% |
| employer | 32 | 0.22% |
| part of | 32 | 0.22% |
| owns | 31 | 0.21% |
| EVALUATE-FOR | 31 | 0.21% |
| facet of | 30 | 0.21% |
| spouse | 30 | 0.21% |
| said to be the same as | 29 | 0.20% |
| based on | 29 | 0.20% |
| creator | 29 | 0.20% |
| field of work | 29 | 0.20% |
| spouse_of | 29 | 0.20% |
| sibling_of | 28 | 0.19% |
| mother | 28 | 0.19% |
| continent | 28 | 0.19% |
| member of political party | 28 | 0.19% |
| architect | 28 | 0.19% |
| languages spoken, written or signed | 28 | 0.19% |
| Exhibits | 28 | 0.19% |
| developer | 27 | 0.19% |
| conflict | 27 | 0.19% |
| notable work | 27 | 0.19% |
| place of death | 27 | 0.19% |
| located in/on physical feature | 26 | 0.18% |
| position held | 26 | 0.18% |
| capital | 26 | 0.18% |
| has use | 26 | 0.18% |
| place served by transport hub | 26 | 0.18% |
| heritage designation | 25 | 0.17% |
| uses | 25 | 0.17% |
| position played on team / speciality | 25 | 0.17% |
| connects with | 25 | 0.17% |
| composer | 24 | 0.16% |
| main subject | 24 | 0.16% |
| original broadcaster | 24 | 0.16% |
| cast member | 24 | 0.16% |
| member_state_of | 24 | 0.16% |
| country of origin | 23 | 0.16% |
| movement | 23 | 0.16% |
| discoverer or inventor | 23 | 0.16% |
| nominated for | 23 | 0.16% |
| connecting line | 22 | 0.15% |
| member of sports team | 22 | 0.15% |
| family | 22 | 0.15% |
| noble title | 22 | 0.15% |
| work location | 22 | 0.15% |
| significant person | 22 | 0.15% |
| location of formation | 22 | 0.15% |
| regulated_by | 21 | 0.14% |
| operator | 21 | 0.14% |
| narrative location | 21 | 0.14% |
| head of government | 21 | 0.14% |
| owner of | 21 | 0.14% |
| participating team | 21 | 0.14% |
| instrument | 20 | 0.14% |
| product or material produced | 20 | 0.14% |
| child | 20 | 0.14% |
| record label | 20 | 0.14% |
| author | 20 | 0.14% |
| sports discipline competed in | 20 | 0.14% |
| publisher | 20 | 0.14% |
| parent organization | 20 | 0.14% |
| sponsored_by | 19 | 0.13% |
| has effect | 19 | 0.13% |
| religion or worldview | 19 | 0.13% |
| military branch | 19 | 0.13% |
| country for sport | 19 | 0.13% |
| native language | 19 | 0.13% |
| basin country | 19 | 0.13% |
| manufacturer | 19 | 0.13% |
| conferred by | 19 | 0.13% |
| historic county | 19 | 0.13% |
| authored_by | 18 | 0.12% |
| opposite of | 18 | 0.12% |
| musical conductor | 18 | 0.12% |
| manufactured_by | 18 | 0.12% |
| residence | 18 | 0.12% |
| medical condition | 18 | 0.12% |
| located in protected area | 17 | 0.12% |
| interacts_with | 17 | 0.12% |
| mouth of the watercourse | 17 | 0.12% |
| occupant | 17 | 0.12% |
| home venue | 17 | 0.12% |
| form of creative work | 17 | 0.12% |
| writing language | 17 | 0.12% |
| package management system | 17 | 0.12% |
| designed_by | 17 | 0.12% |
| incarnation of | 17 | 0.12% |
| student of | 17 | 0.12% |
| child_of | 17 | 0.12% |
| participant of | 16 | 0.11% |
| production company | 16 | 0.11% |
| military rank | 16 | 0.11% |
| parent taxon | 16 | 0.11% |
| advisor_to | 16 | 0.11% |
| currency | 16 | 0.11% |
| applies to jurisdiction | 16 | 0.11% |
| depicts | 16 | 0.11% |
| league | 16 | 0.11% |
| born_in | 16 | 0.11% |
| online service | 16 | 0.11% |
| subclass of | 16 | 0.11% |
| located on astronomical body | 16 | 0.11% |
| crosses | 16 | 0.11% |
| officeholder | 15 | 0.10% |
| PART-OF | 15 | 0.10% |
| presenter | 15 | 0.10% |
| student | 15 | 0.10% |
| convicted of | 15 | 0.10% |
| located in or next to body of water | 15 | 0.10% |
| crew member(s) | 15 | 0.10% |
| operating area | 15 | 0.10% |
| place of detention | 15 | 0.10% |
| tributary | 14 | 0.10% |
| FEATURE-OF | 14 | 0.10% |
| licensed to broadcast to | 14 | 0.10% |
| this taxon is source of | 14 | 0.10% |
| biological process | 14 | 0.10% |
| court | 14 | 0.10% |
| place of burial | 14 | 0.10% |
| made from material | 14 | 0.10% |
| surface played on | 14 | 0.10% |
| dialect of | 14 | 0.10% |
| replaced by | 14 | 0.10% |
| terminus location | 14 | 0.10% |
| secretary general | 14 | 0.10% |
| composed_by | 14 | 0.10% |
| language of work or name | 14 | 0.10% |
| chairperson | 14 | 0.10% |
| anatomical location | 14 | 0.10% |
| place of publication | 13 | 0.09% |
| follows | 13 | 0.09% |
| illustrator | 13 | 0.09% |
| legislated by | 13 | 0.09% |
| distributed by | 13 | 0.09% |
| located in the statistical territorial entity | 13 | 0.09% |
| allegiance | 13 | 0.09% |
| item operated | 13 | 0.09% |
| organizer | 13 | 0.09% |
| track gauge | 13 | 0.09% |
| producer | 13 | 0.09% |
| lyrics by | 13 | 0.09% |
| editor | 13 | 0.09% |
| located in the present-day administrative territorial entity | 13 | 0.09% |
| school district | 13 | 0.09% |
| this zoological name is coordinate with | 13 | 0.09% |
| mountain range | 13 | 0.09% |
| has part(s) of the class | 13 | 0.09% |
| location of first performance | 13 | 0.09% |
| taxonomic type | 13 | 0.09% |
| directed_by | 13 | 0.09% |
| member of the crew of | 12 | 0.08% |
| stock market index | 12 | 0.08% |
| vehicle | 12 | 0.08% |
| found in taxon | 12 | 0.08% |
| partially coincident with | 12 | 0.08% |
| indigenous to | 12 | 0.08% |
| commanded by | 12 | 0.08% |
| tracklist | 12 | 0.08% |
| architectural style | 12 | 0.08% |
| grants | 12 | 0.08% |
| ammunition | 12 | 0.08% |
| drug or therapy used for treatment | 12 | 0.08% |
| located on street | 12 | 0.08% |
| pole position | 12 | 0.08% |
| territory overlaps | 12 | 0.08% |
| affiliation | 12 | 0.08% |
| taxon rank | 12 | 0.08% |
| collection | 12 | 0.08% |
| genic_interaction | 12 | 0.08% |
| has works in the collection | 12 | 0.08% |
| copyright holder | 12 | 0.08% |
| winner | 12 | 0.08% |
| distributor_of | 12 | 0.08% |
| vessel class | 11 | 0.08% |
| brand | 11 | 0.08% |
| model item | 11 | 0.08% |
| reservoir created | 11 | 0.08% |
| sex or gender | 11 | 0.08% |
| located on terrain feature | 11 | 0.08% |
| from narrative universe | 11 | 0.08% |
| maintained by | 11 | 0.08% |
| has written for | 11 | 0.08% |
| diocese | 11 | 0.08% |
| worshipped by | 11 | 0.08% |
| physically interacts with | 11 | 0.08% |
| genetic association | 11 | 0.08% |
| start point | 11 | 0.08% |
| original language of film or TV show | 11 | 0.08% |
| co-driver | 11 | 0.08% |
| commemorates | 11 | 0.08% |
| time period | 11 | 0.08% |
| objective of project or action | 11 | 0.08% |
| historical region | 11 | 0.08% |
| dam | 11 | 0.08% |
| astronaut mission | 11 | 0.08% |
| distribution format | 11 | 0.08% |
| Bind | 11 | 0.08% |
| operating system | 11 | 0.08% |
| broadcast by | 11 | 0.08% |
| charged_with | 11 | 0.08% |
| league system | 11 | 0.08% |
| part of the series | 10 | 0.07% |
| stock exchange | 10 | 0.07% |
| chief executive officer | 10 | 0.07% |
| librettist | 10 | 0.07% |
| ethnic group | 10 | 0.07% |
| honorific prefix | 10 | 0.07% |
| IUCN conservation status | 10 | 0.07% |
| cause of death | 10 | 0.07% |
| interested in | 10 | 0.07% |
| capital of | 10 | 0.07% |
| filming location | 10 | 0.07% |
| relative | 10 | 0.07% |
| cathedral | 10 | 0.07% |
| used by | 10 | 0.07% |
| partner in business or sport | 10 | 0.07% |
| customer_of | 10 | 0.07% |
| writing system | 10 | 0.07% |
| published in | 10 | 0.07% |
| represented by | 10 | 0.07% |
| enemy of | 9 | 0.06% |
| merged into | 9 | 0.06% |
| commissioned by | 9 | 0.06% |
| location of creation | 9 | 0.06% |
| recommended unit of measurement | 9 | 0.06% |
| is the study of | 9 | 0.06% |
| culture | 9 | 0.06% |
| transport network | 9 | 0.06% |
| contains | 9 | 0.06% |
| carries | 9 | 0.06% |
| basic form of government | 9 | 0.06% |
| influenced by | 9 | 0.06% |
| opposed_to | 9 | 0.06% |
| mentor_of | 9 | 0.06% |
| has subsidiary | 9 | 0.06% |
| has goal | 9 | 0.06% |
| coach of sports team | 9 | 0.06% |
| office held by head of government | 9 | 0.06% |
| COMPARE | 9 | 0.06% |
| subject has role | 9 | 0.06% |
| field of this occupation | 8 | 0.05% |
| linguistic typology | 8 | 0.05% |
| victory | 8 | 0.05% |
| manner of death | 8 | 0.05% |
| adjacent station | 8 | 0.05% |
| designed by | 8 | 0.05% |
| invested_in | 8 | 0.05% |
| killed by | 8 | 0.05% |
| constellation | 8 | 0.05% |
| sponsor | 8 | 0.05% |
| main regulatory text | 8 | 0.05% |
| competition class | 8 | 0.05% |
| has cause | 8 | 0.05% |
| standards body | 8 | 0.05% |
| superhuman feature or ability | 8 | 0.05% |
| public holiday | 8 | 0.05% |
| commander of (DEPRECATED) | 8 | 0.05% |
| dedicated to | 8 | 0.05% |
| located in time zone | 8 | 0.05% |
| color | 8 | 0.05% |
| target | 8 | 0.05% |
| location of landing | 8 | 0.05% |
| foundational text | 8 | 0.05% |
| official residence | 8 | 0.05% |
| medical condition treated | 8 | 0.05% |
| lake outflow | 8 | 0.05% |
| edition or translation of | 8 | 0.05% |
| consecrator | 8 | 0.05% |
| compatible_with | 8 | 0.05% |
| next higher rank | 8 | 0.05% |
| candidate | 7 | 0.05% |
| unmarried partner | 7 | 0.05% |
| history of topic | 7 | 0.05% |
| substitute/deputy/replacement of office/officeholder | 7 | 0.05% |
| won | 7 | 0.05% |
| contributed to creative work | 7 | 0.05% |
| official language | 7 | 0.05% |
| software engine | 7 | 0.05% |
| qualifies for event | 7 | 0.05% |
| voice actor | 7 | 0.05% |
| parent club | 7 | 0.05% |
| copyright license | 7 | 0.05% |
| testified_against | 7 | 0.05% |
| media franchise | 7 | 0.05% |
| inflows | 7 | 0.05% |
| board member | 7 | 0.05% |
| produced by | 7 | 0.05% |
| highest judicial authority | 7 | 0.05% |
| airline alliance | 7 | 0.05% |
| cause of destruction | 7 | 0.05% |
| chromosome | 7 | 0.05% |
| endemic to | 7 | 0.05% |
| source of energy | 7 | 0.05% |
| connecting service | 7 | 0.05% |
| voice type | 7 | 0.05% |
| contributor to the creative work or subject | 7 | 0.05% |
| parent astronomical body | 7 | 0.05% |
| after a work by | 7 | 0.05% |
| has immediate cause | 7 | 0.05% |
| domain of saint or deity | 7 | 0.05% |
| fictional universe described in | 7 | 0.05% |
| party_to | 7 | 0.05% |
| implementation of | 6 | 0.04% |
| overlies | 6 | 0.04% |
| destination point | 6 | 0.04% |
| house publication | 6 | 0.04% |
| significant place | 6 | 0.04% |
| anthem | 6 | 0.04% |
| wheel arrangement | 6 | 0.04% |
| set in period | 6 | 0.04% |
| has edition or translation | 6 | 0.04% |
| political ideology | 6 | 0.04% |
| cell component | 6 | 0.04% |
| office held by head of the organization | 6 | 0.04% |
| official religion | 6 | 0.04% |
| head of state | 6 | 0.04% |
| has quality | 6 | 0.04% |
| central bank/issuer | 6 | 0.04% |
| business division | 6 | 0.04% |
| measured physical quantity | 6 | 0.04% |
| industry | 6 | 0.04% |
| title of chess person | 6 | 0.04% |
| legal form | 6 | 0.04% |
| archives at | 6 | 0.04% |
| location of discovery | 6 | 0.04% |
| primary destinations | 6 | 0.04% |
| religious order | 6 | 0.04% |
| twinned administrative body | 6 | 0.04% |
| named_after | 6 | 0.04% |
| authority | 6 | 0.04% |
| character type | 6 | 0.04% |
| political alignment | 6 | 0.04% |
| head coach | 6 | 0.04% |
| academic degree | 6 | 0.04% |
| inspired by | 6 | 0.04% |
| powered by | 6 | 0.04% |
| greater than | 5 | 0.03% |
| anatomical branch of | 5 | 0.03% |
| candidacy in election | 5 | 0.03% |
| league level below | 5 | 0.03% |
| encodes | 5 | 0.03% |
| dual to | 5 | 0.03% |
| perpetrator | 5 | 0.03% |
| measures | 5 | 0.03% |
| terminus | 5 | 0.03% |
| newspaper format | 5 | 0.03% |
| league level above | 5 | 0.03% |
| type of variable star | 5 | 0.03% |
| Gram staining | 5 | 0.03% |
| central bank | 5 | 0.03% |
| separated from | 5 | 0.03% |
| fastest lap | 5 | 0.03% |
| takes place in fictional universe | 5 | 0.03% |
| cause | 5 | 0.03% |
| manifestation of | 5 | 0.03% |
| lake on watercourse | 5 | 0.03% |
| described by source | 5 | 0.03% |
| depicted by | 5 | 0.03% |
| first appearance | 5 | 0.03% |
| season | 5 | 0.03% |
| practiced by | 5 | 0.03% |
| natural product of taxon | 5 | 0.03% |
| general classification of race participants | 5 | 0.03% |
| subdivision of this unit | 5 | 0.03% |
| merged_with | 5 | 0.03% |
| allied_with | 5 | 0.03% |
| symptoms and signs | 5 | 0.03% |
| has contributing factor | 5 | 0.03% |
| radio format | 5 | 0.03% |
| armament | 5 | 0.03% |
| Drug_Interaction | 5 | 0.03% |
| writable file format | 5 | 0.03% |
| airline hub | 4 | 0.03% |
| drainage basin | 4 | 0.03% |
| delta of | 4 | 0.03% |
| contains settlement | 4 | 0.03% |
| signatory | 4 | 0.03% |
| editor-in-chief | 4 | 0.03% |
| country of registry | 4 | 0.03% |
| GUI toolkit or framework | 4 | 0.03% |
| mountains classification | 4 | 0.03% |
| director of photography | 4 | 0.03% |
| home port | 4 | 0.03% |
| penalty | 4 | 0.03% |
| music created for | 4 | 0.03% |
| has anatomical branch | 4 | 0.03% |
| derivative work | 4 | 0.03% |
| canonization status | 4 | 0.03% |
| executive body | 4 | 0.03% |
| charge | 4 | 0.03% |
| has seal, badge, or sigil | 4 | 0.03% |
| flag | 4 | 0.03% |
| vehicle normally used | 4 | 0.03% |
| Christian liturgical rite | 4 | 0.03% |
| coextensive with | 4 | 0.03% |
| structural engineer | 4 | 0.03% |
| fabrication method | 4 | 0.03% |
| office held by head of state | 4 | 0.03% |
| sports season of league or competition | 4 | 0.03% |
| film editor | 4 | 0.03% |
| advertises | 4 | 0.03% |
| possible treatment | 4 | 0.03% |
| intended public | 4 | 0.03% |
| territory claimed by | 4 | 0.03% |
| drafted by | 4 | 0.03% |
| victim | 4 | 0.03% |
| type foundry | 4 | 0.03% |
| ancestral home | 4 | 0.03% |
| programming paradigm | 4 | 0.03% |
| sued_by | 4 | 0.03% |
| treats | 4 | 0.03% |
| Comparison | 4 | 0.03% |
| recorded at studio or venue | 3 | 0.02% |
| game mode | 3 | 0.02% |
| cuisine | 3 | 0.02% |
| health specialty | 3 | 0.02% |
| mount | 3 | 0.02% |
| addressee | 3 | 0.02% |
| engine configuration | 3 | 0.02% |
| offers view on | 3 | 0.02% |
| organization directed by the office or position | 3 | 0.02% |
| immediate cause of | 3 | 0.02% |
| predecessor_of | 3 | 0.02% |
| animal breed | 3 | 0.02% |
| update method | 3 | 0.02% |
| instruction set | 3 | 0.02% |
| type locality (geology) | 3 | 0.02% |
| published_by | 3 | 0.02% |
| prohibits | 3 | 0.02% |
| next lower rank | 3 | 0.02% |
| vaccine for | 3 | 0.02% |
| parent of this hybrid, breed, or cultivar | 3 | 0.02% |
| party chief representative | 3 | 0.02% |
| patron saint | 3 | 0.02% |
| polymer of | 3 | 0.02% |
| sold_to | 3 | 0.02% |
| successor_of | 3 | 0.02% |
| mascot | 3 | 0.02% |
| choreographer | 3 | 0.02% |
| nominated by | 3 | 0.02% |
| carries scientific instrument | 3 | 0.02% |
| risk factor | 3 | 0.02% |
| legislative body | 3 | 0.02% |
| appears in the form of | 3 | 0.02% |
| successful candidate | 3 | 0.02% |
| capital_of | 3 | 0.02% |
| business model | 3 | 0.02% |
| geography of topic | 3 | 0.02% |
| input method | 3 | 0.02% |
| in opposition to | 3 | 0.02% |
| diplomatic relation | 3 | 0.02% |
| associated electoral district | 3 | 0.02% |
| named by | 3 | 0.02% |
| director / manager | 3 | 0.02% |
| connector | 3 | 0.02% |
| final event | 3 | 0.02% |
| instrumentation | 3 | 0.02% |
| home world | 3 | 0.02% |
| geomorphological unit | 3 | 0.02% |
| arterial supply | 3 | 0.02% |
| origin of the watercourse | 3 | 0.02% |
| game mechanics | 3 | 0.02% |
| series spin-off | 3 | 0.02% |
| prime factor | 2 | 0.01% |
| approved by | 2 | 0.01% |
| has list | 2 | 0.01% |
| sidekick of | 2 | 0.01% |
| conjugate acid | 2 | 0.01% |
| points/goal scored by | 2 | 0.01% |
| programming language | 2 | 0.01% |
| binding of software library | 2 | 0.01% |
| shape | 2 | 0.01% |
| stepparent | 2 | 0.01% |
| antiparticle | 2 | 0.01% |
| cites work | 2 | 0.01% |
| died_in | 2 | 0.01% |
| space launch vehicle | 2 | 0.01% |
| defendant | 2 | 0.01% |
| performed at | 2 | 0.01% |
| active ingredient in | 2 | 0.01% |
| screenwriter | 2 | 0.01% |
| exemplar of | 2 | 0.01% |
| next crossing upstream | 2 | 0.01% |
| family name identical to this given name | 2 | 0.01% |
| set in environment | 2 | 0.01% |
| designed to carry | 2 | 0.01% |
| foods traditionally associated | 2 | 0.01% |
| referee | 2 | 0.01% |
| convicted_of | 2 | 0.01% |
| minor planet group | 2 | 0.01% |
| licensed_to | 2 | 0.01% |
| narrator | 2 | 0.01% |
| statement describes | 2 | 0.01% |
| depends on software | 2 | 0.01% |
| rector | 2 | 0.01% |
| day of week | 2 | 0.01% |
| coat of arms | 2 | 0.01% |
| type of orbit | 2 | 0.01% |
| members have occupation | 2 | 0.01% |
| soundtrack release | 2 | 0.01% |
| academic thesis | 2 | 0.01% |
| designated as terrorist by | 2 | 0.01% |
| discography | 2 | 0.01% |
| language regulatory body | 2 | 0.01% |
| main food source | 2 | 0.01% |
| scheduled service destination | 2 | 0.01% |
| gens | 2 | 0.01% |
| highest point | 2 | 0.01% |
| child astronomical body | 2 | 0.01% |
| structure replaced by | 2 | 0.01% |
| translator | 2 | 0.01% |
| motto | 2 | 0.01% |
| temporal range start | 2 | 0.01% |
| copyright representative | 2 | 0.01% |
| afflicts | 2 | 0.01% |
| host | 2 | 0.01% |
| sexual orientation | 2 | 0.01% |
| wears | 2 | 0.01% |
| interchange station | 2 | 0.01% |
| less than | 2 | 0.01% |
| amended by | 2 | 0.01% |
| day in year for periodic occurrence | 2 | 0.01% |
| discovery method | 2 | 0.01% |
| honorific suffix | 2 | 0.01% |
| mother house | 2 | 0.01% |
| site of astronomical discovery | 2 | 0.01% |
| monomer of | 2 | 0.01% |
| has boundary | 2 | 0.01% |
| innervates | 2 | 0.01% |
| train depot | 2 | 0.01% |
| source of material | 2 | 0.01% |
| expansion of | 2 | 0.01% |
| enclave within | 1 | 0.01% |
| habitat | 1 | 0.01% |
| output method | 1 | 0.01% |
| website account on | 1 | 0.01% |
| has natural reservoir | 1 | 0.01% |
| has phenotype | 1 | 0.01% |
| season of club or team | 1 | 0.01% |
| donated by | 1 | 0.01% |
| via | 1 | 0.01% |
| elector | 1 | 0.01% |
| parent peak | 1 | 0.01% |
| month of the year | 1 | 0.01% |
| officially opened by | 1 | 0.01% |
| solved by | 1 | 0.01% |
| wing configuration | 1 | 0.01% |
| substrate of | 1 | 0.01% |
| promoted | 1 | 0.01% |
| entry receptor | 1 | 0.01% |
| office contested | 1 | 0.01% |
| electoral district | 1 | 0.01% |
| species kept | 1 | 0.01% |
| executive producer | 1 | 0.01% |
| calculated from | 1 | 0.01% |
| adjacent_to | 1 | 0.01% |
| charted in | 1 | 0.01% |
| theorised by | 1 | 0.01% |
| typing discipline | 1 | 0.01% |
| borders | 1 | 0.01% |
| points classification | 1 | 0.01% |
| depends_on | 1 | 0.01% |
| official symbol | 1 | 0.01% |
| has active ingredient | 1 | 0.01% |
| given name version for other gender | 1 | 0.01% |
| derived_from | 1 | 0.01% |
| undercarriage | 1 | 0.01% |
| stereoisomer of | 1 | 0.01% |
| has facility | 1 | 0.01% |
| category of associated people | 1 | 0.01% |
| worn by | 1 | 0.01% |
| record held | 1 | 0.01% |
| issued by | 1 | 0.01% |
| is metaclass for | 1 | 0.01% |
| peak bagging classification | 1 | 0.01% |
| has grammatical case | 1 | 0.01% |
| has facet polytope | 1 | 0.01% |
| readable file format | 1 | 0.01% |
| prerequisite | 1 | 0.01% |
| programmer | 1 | 0.01% |
| storyboard artist | 1 | 0.01% |
| crystal system | 1 | 0.01% |
| measured_in | 1 | 0.01% |
| speaker | 1 | 0.01% |
| demographics of topic | 1 | 0.01% |
| uses capitalization for | 1 | 0.01% |
| religious occupation | 1 | 0.01% |
| product certification | 1 | 0.01% |
| fictional analog of | 1 | 0.01% |
| adjacent building | 1 | 0.01% |
| may prevent | 1 | 0.01% |
| political coalition | 1 | 0.01% |
| distributary | 1 | 0.01% |
| decays to | 1 | 0.01% |
| director of archaeological fieldwork | 1 | 0.01% |
| adapted by | 1 | 0.01% |
| military casualty classification | 1 | 0.01% |
| destroyed | 1 | 0.01% |
| Cotreatment | 1 | 0.01% |
| has fruit type | 1 | 0.01% |
| production designer | 1 | 0.01% |
| metrically compatible typeface | 1 | 0.01% |
| lifestyle | 1 | 0.01% |
| supported sports team | 1 | 0.01% |
| innervated by | 1 | 0.01% |
| typically sells | 1 | 0.01% |
| order of battle | 1 | 0.01% |
| social classification | 1 | 0.01% |
| microarchitecture | 1 | 0.01% |
| causes | 1 | 0.01% |
| emulates | 1 | 0.01% |
| feast day | 1 | 0.01% |
| economy of topic | 1 | 0.01% |
| exhibition history | 1 | 0.01% |
| sexually homologous with | 1 | 0.01% |
| permanent resident of | 1 | 0.01% |
| youth wing | 1 | 0.01% |
| top-level Internet domain | 1 | 0.01% |
| has pet | 1 | 0.01% |
| captain | 1 | 0.01% |
| underlies | 1 | 0.01% |
| repeals | 1 | 0.01% |
| statistical leader | 1 | 0.01% |
| official podcast | 1 | 0.01% |
| madhhab | 1 | 0.01% |
| election called by | 1 | 0.01% |
| studied by | 1 | 0.01% |
| Fach | 1 | 0.01% |
| contributing factor of | 1 | 0.01% |
| cover art by | 1 | 0.01% |
| research site | 1 | 0.01% |
| art director | 1 | 0.01% |
| supplement to | 1 | 0.01% |
| measurement scale | 1 | 0.01% |
| animator | 1 | 0.01% |
| computes solution to | 1 | 0.01% |
| set during recurring event | 1 | 0.01% |
| OSI Model layer location | 1 | 0.01% |
| trophy awarded | 1 | 0.01% |
| CPU | 1 | 0.01% |
| diaspora | 1 | 0.01% |
| proved by | 1 | 0.01% |

### Classifications

**train, actionability.** Majority class 87.6% of 788 answers. The marginal is `true_label`, not the menu.

#### train actionability

| label | count | share |
| --- | --- | --- |
| informational | 690 | 87.56% |
| advisory | 55 | 6.98% |
| requires action | 43 | 5.46% |

**train, audience.** Majority class 75.7% of 941 answers. The marginal is `true_label`, not the menu.

#### train audience

| label | count | share |
| --- | --- | --- |
| general public | 712 | 75.66% |
| expert | 134 | 14.24% |
| regulatory | 74 | 7.86% |
| internal | 21 | 2.23% |

**train, biotech_event.** Majority class 27.6% of 529 answers. The marginal is `true_label`, not the menu.

#### train biotech_event

| label | count | share |
| --- | --- | --- |
| executive statement | 146 | 27.60% |
| company description | 80 | 15.12% |
| alliance & partnership | 44 | 8.32% |
| other | 39 | 7.37% |
| product launching & presentation | 34 | 6.43% |
| new initiatives or programs | 20 | 3.78% |
| expanding geography | 19 | 3.59% |
| support & philanthropy | 18 | 3.40% |
| m&a | 17 | 3.21% |
| article publication | 15 | 2.84% |
| service & product providing | 13 | 2.46% |
| funding round | 12 | 2.27% |
| executive appointment | 11 | 2.08% |
| regulatory approval | 8 | 1.51% |
| product updates | 8 | 1.51% |
| participation in an event | 7 | 1.32% |
| new initiatives & programs | 6 | 1.13% |
| ipo exit | 5 | 0.95% |
| investment in public company | 5 | 0.95% |
| foundation | 5 | 0.95% |
| hiring | 4 | 0.76% |
| closing | 4 | 0.76% |
| event organization | 4 | 0.76% |
| expanding industry | 2 | 0.38% |
| clinical trial sponsorship | 2 | 0.38% |
| department establishment | 1 | 0.19% |

**train, certainty.** Majority class 96.0% of 775 answers. The marginal is `true_label`, not the menu.

#### train certainty

| label | count | share |
| --- | --- | --- |
| asserted | 744 | 96.00% |
| hedged | 29 | 3.74% |
| speculative | 2 | 0.26% |

**train, chfinann_event.** Majority class 48.9% of 3,282 answers. The marginal is `true_label`, not the menu.

#### train chfinann_event

| label | count | share |
| --- | --- | --- |
| EquityPledge | 1,605 | 48.90% |
| EquityUnderweight | 691 | 21.05% |
| EquityOverweight | 652 | 19.87% |
| EquityRepurchase | 232 | 7.07% |
| EquityFreeze | 102 | 3.11% |

**train, docee_event.** Majority class 7.0% of 9,223 answers. The marginal is `true_label`, not the menu.

#### train docee_event

| label | count | share |
| --- | --- | --- |
| Sports Competition | 641 | 6.95% |
| none | 500 | 5.42% |
| Government Policy Changes | 440 | 4.77% |
| Fire | 395 | 4.28% |
| Organization Fine | 393 | 4.26% |
| CommitCrime - Investigate | 359 | 3.89% |
| CommitCrime - Arrest | 334 | 3.62% |
| Armed Conflict | 314 | 3.40% |
| Air Crash | 284 | 3.08% |
| Appoint_Inauguration | 280 | 3.04% |
| Resignation_Dismissal | 254 | 2.75% |
| Famous Person - Death | 237 | 2.57% |
| Protest | 229 | 2.48% |
| Road Crash | 209 | 2.27% |
| Famous Person - Give a Speech | 203 | 2.20% |
| New Achievements in Aerospace | 200 | 2.17% |
| Election | 197 | 2.14% |
| Earthquakes | 193 | 2.09% |
| Riot | 179 | 1.94% |
| Diplomatic Talks | 178 | 1.93% |
| CommitCrime - Accuse | 172 | 1.86% |
| Break Historical Records | 163 | 1.77% |
| Awards Ceremony | 154 | 1.67% |
| Disease Outbreaks | 154 | 1.67% |
| Mass Poisoning | 138 | 1.50% |
| CommitCrime - Sentence | 137 | 1.49% |
| Sign Agreement | 137 | 1.49% |
| Storm | 134 | 1.45% |
| Floods | 131 | 1.42% |
| Military Excercise | 114 | 1.24% |
| Diplomatic Visit | 97 | 1.05% |
| New Wonders in Nature | 95 | 1.03% |
| Strike | 94 | 1.02% |
| Organization Established | 79 | 0.86% |
| Environment Pollution | 78 | 0.85% |
| New Archeological Discoveries | 76 | 0.82% |
| Shipwreck | 75 | 0.81% |
| Financial Crisis | 74 | 0.80% |
| Organization Closed | 69 | 0.75% |
| Gas Explosion | 69 | 0.75% |
| Train Collisions | 66 | 0.72% |
| Mudslides | 62 | 0.67% |
| Famous Person - Recovered | 62 | 0.67% |
| Financial Aid | 57 | 0.62% |
| Famous Person - Divorce | 57 | 0.62% |
| Bank Robbery | 56 | 0.61% |
| Volcano Eruption | 56 | 0.61% |
| Withdraw from an Orgnization | 54 | 0.59% |
| Mine Collapses | 53 | 0.57% |
| CommitCrime - Release | 52 | 0.56% |
| Famine | 52 | 0.56% |
| Droughts | 51 | 0.55% |
| Organization Merge | 44 | 0.48% |
| Famous Person - Marriage | 44 | 0.48% |
| Famous Person - Sick | 43 | 0.47% |
| Regime Change | 42 | 0.46% |
| Join in an Orgnization | 33 | 0.36% |
| Insect Disaster | 31 | 0.34% |
| Tear Up Agreement | 25 | 0.27% |
| Tsunamis | 24 | 0.26% |

**train, docfee_event.** Majority class 33.6% of 2,110 answers. The marginal is `true_label`, not the menu.

#### train docfee_event

| label | count | share |
| --- | --- | --- |
| EquityUnderweight | 709 | 33.60% |
| BankruptcyLiquidation | 354 | 16.78% |
| EquityPledge | 328 | 15.55% |
| MajorAssetLoss | 301 | 14.27% |
| EquityFreeze | 135 | 6.40% |
| EquityOverweight | 108 | 5.12% |
| MajorSafetyAccident | 74 | 3.51% |
| MajorExternalPayout | 52 | 2.46% |
| ExecutiveDeath | 49 | 2.32% |

**train, document_genre.** Majority class 73.7% of 798 answers. The marginal is `true_label`, not the menu.

#### train document_genre

| label | count | share |
| --- | --- | --- |
| news report | 588 | 73.68% |
| announcement | 101 | 12.66% |
| opinion | 39 | 4.89% |
| review | 20 | 2.51% |
| correspondence | 20 | 2.51% |
| technical documentation | 16 | 2.01% |
| advertisement | 10 | 1.25% |
| legal filing | 4 | 0.50% |

**train, formality.** Majority class 79.8% of 926 answers. The marginal is `true_label`, not the menu.

#### train formality

| label | count | share |
| --- | --- | --- |
| formal | 739 | 79.81% |
| informal | 187 | 20.19% |

**train, language_register.** Majority class 78.0% of 705 answers. The marginal is `true_label`, not the menu.

#### train language_register

| label | count | share |
| --- | --- | --- |
| plain | 550 | 78.01% |
| technical | 105 | 14.89% |
| colloquial | 32 | 4.54% |
| legal | 18 | 2.55% |

**train, risk_level.** Majority class 50.0% of 714 answers. The marginal is `true_label`, not the menu.

#### train risk_level

| label | count | share |
| --- | --- | --- |
| none | 357 | 50.00% |
| low | 134 | 18.77% |
| high | 115 | 16.11% |
| moderate | 108 | 15.13% |

**train, sentiment.** Majority class 36.0% of 925 answers. The marginal is `true_label`, not the menu.

#### train sentiment

| label | count | share |
| --- | --- | --- |
| positive | 333 | 36.00% |
| neutral | 329 | 35.57% |
| negative | 263 | 28.43% |

**train, subjectivity.** Majority class 87.0% of 707 answers. The marginal is `true_label`, not the menu.

#### train subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 615 | 86.99% |
| subjective | 92 | 13.01% |

**train, temporal_orientation.** Majority class 69.9% of 705 answers. The marginal is `true_label`, not the menu.

#### train temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 493 | 69.93% |
| present | 146 | 20.71% |
| future | 60 | 8.51% |
| timeless | 6 | 0.85% |

**train, topic.** Majority class 19.0% of 1,629 answers. The marginal is `true_label`, not the menu.

#### train topic

| label | count | share |
| --- | --- | --- |
| business | 309 | 18.97% |
| politics | 194 | 11.91% |
| world | 167 | 10.25% |
| legal | 152 | 9.33% |
| sports | 151 | 9.27% |
| finance | 116 | 7.12% |
| technology | 114 | 7.00% |
| entertainment | 110 | 6.75% |
| health | 99 | 6.08% |
| science | 72 | 4.42% |
| education | 44 | 2.70% |
| environment | 43 | 2.64% |
| defense | 21 | 1.29% |
| travel | 20 | 1.23% |
| agriculture | 9 | 0.55% |
| energy | 8 | 0.49% |

**train, urgency.** Majority class 80.7% of 699 answers. The marginal is `true_label`, not the menu.

#### train urgency

| label | count | share |
| --- | --- | --- |
| routine | 564 | 80.69% |
| elevated | 103 | 14.74% |
| critical | 28 | 4.01% |
| urgent | 4 | 0.57% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | patient_group | 14 | 100.00% |
| clinical_finding | condition | 13 | 92.86% |
| clinical_finding | treatment | 10 | 71.43% |
| clinical_finding | outcome | 10 | 71.43% |
| clinical_finding | record_metadata | natural/condition ×13, natural/patient_group ×1 |  |
| company_profile | name | 41 | 100.00% |
| company_profile | size | 36 | 87.80% |
| company_profile | headquarters | 32 | 78.05% |
| company_profile | industry | 28 | 68.29% |
| company_profile | founded | 11 | 26.83% |
| company_profile | record_metadata | natural/name ×41 |  |
| event_listing | name | 48 | 94.12% |
| event_listing | venue | 45 | 88.24% |
| event_listing | date | 43 | 84.31% |
| event_listing | organizer | 32 | 62.75% |
| event_listing | price | 10 | 19.61% |
| event_listing | record_metadata | natural/name ×36, natural/venue ×3 |  |
| financial_report | organization | 6 | 100.00% |
| financial_report | period | 4 | 66.67% |
| financial_report | revenue | 4 | 66.67% |
| financial_report | net_income | 2 | 33.33% |
| financial_report | record_metadata | natural/organization ×6 |  |
| incident_report | severity | 59 | 100.00% |
| incident_report | date | 56 | 94.92% |
| incident_report | location | 54 | 91.53% |
| incident_report | incident_type | 46 | 77.97% |
| incident_report | casualties | 28 | 47.46% |
| incident_report | record_metadata | natural/location ×37, natural/incident_type ×16, natural/date ×2 |  |
| legal_case | defendant | 15 | 100.00% |
| legal_case | status | 15 | 100.00% |
| legal_case | plaintiff | 13 | 86.67% |
| legal_case | court | 10 | 66.67% |
| legal_case | case_name | 6 | 40.00% |
| legal_case | record_metadata | natural/case_name ×6, natural/court ×4, natural/plaintiff ×3, natural/defendant ×1 |  |
| person_profile | name | 300 | 99.67% |
| person_profile | role | 277 | 92.03% |
| person_profile | location | 277 | 92.03% |
| person_profile | employer | 261 | 86.71% |
| person_profile | record_metadata | natural/location ×180, natural/name ×66 |  |
| product | condition | 62 | 100.00% |
| product | name | 61 | 98.39% |
| product | brand | 60 | 96.77% |
| product | price | 38 | 61.29% |
| product | record_metadata | natural/name ×55, natural/price ×1 |  |
| real_estate_listing | address | 7 | 100.00% |
| real_estate_listing | price | 7 | 100.00% |
| real_estate_listing | property_type | 7 | 100.00% |
| real_estate_listing | bedrooms | 6 | 85.71% |
| real_estate_listing | area | 3 | 42.86% |
| real_estate_listing | record_metadata | natural/address ×3 |  |
| record | status | 124 | 12.72% |
| record | location | 101 | 10.36% |
| record | name | 65 | 6.67% |
| record | address | 51 | 5.23% |
| record | title | 41 | 4.21% |
| record | date | 32 | 3.28% |
| record | description | 25 | 2.56% |
| record | price | 24 | 2.46% |
| record | city | 24 | 2.46% |
| record | amount | 22 | 2.26% |
| record | state | 21 | 2.15% |
| record | patient_name | 20 | 2.05% |
| record | patient_id | 19 | 1.95% |
| record | age | 18 | 1.85% |
| record | department | 18 | 1.85% |
| record | employee_id | 17 | 1.74% |
| record | rating | 17 | 1.74% |
| record | product_id | 17 | 1.74% |
| record | transaction_id | 13 | 1.33% |
| record | author | 13 | 1.33% |
| record | venue | 13 | 1.33% |
| record | ticket_id | 13 | 1.33% |
| record | year | 12 | 1.23% |
| record | company | 11 | 1.13% |
| record | claim_id | 11 | 1.13% |
| record | score | 11 | 1.13% |
| record | time | 10 | 1.03% |
| record | review_id | 10 | 1.03% |
| record | designation | 9 | 0.92% |
| record | head | 9 | 0.92% |
| record | tail | 9 | 0.92% |
| record | event_id | 9 | 0.92% |
| record | project_id | 9 | 0.92% |
| record | reviewer | 9 | 0.92% |
| record | organizations | 8 | 0.82% |
| record | person.name | 8 | 0.82% |
| record | funding_source | 8 | 0.82% |
| record | category | 8 | 0.82% |
| record | patient_address | 7 | 0.72% |
| record | notes | 7 | 0.72% |
| record | provider | 7 | 0.72% |
| record | discovery.discoverer | 7 | 0.72% |
| record | manufacturer | 7 | 0.72% |
| record | player_name | 7 | 0.72% |
| record | region | 7 | 0.72% |
| record | country | 7 | 0.72% |
| record | features | 7 | 0.72% |
| record | serial_number | 7 | 0.72% |
| record | home_team | 7 | 0.72% |
| record | person | 6 | 0.62% |
| record | review_status | 6 | 0.62% |
| record | product_name | 6 | 0.62% |
| record | orbital_characteristics.semi_major_axis.unit | 6 | 0.62% |
| record | diameter.unit | 6 | 0.62% |
| record | medication | 6 | 0.62% |
| record | room | 6 | 0.62% |
| record | asteroid.name | 6 | 0.62% |
| record | asteroid.type | 6 | 0.62% |
| record | full_name | 6 | 0.62% |
| record | event | 6 | 0.62% |
| record | role | 6 | 0.62% |
| record | product | 6 | 0.62% |
| record | policy_id | 6 | 0.62% |
| record | dosage | 6 | 0.62% |
| record | customer_address | 6 | 0.62% |
| record | lead_researcher | 6 | 0.62% |
| record | discovery.observatory.name | 6 | 0.62% |
| record | discovery.observatory.location.country | 6 | 0.62% |
| record | gender | 6 | 0.62% |
| record | url | 6 | 0.62% |
| record | customer_name | 6 | 0.62% |
| record | filing_id | 5 | 0.51% |
| record | submission_id | 5 | 0.51% |
| record | diagnosis | 5 | 0.51% |
| record | district | 5 | 0.51% |
| record | discovery.date | 5 | 0.51% |
| record | scientific_name | 5 | 0.51% |
| record | orbital_characteristics.perihelion.unit | 5 | 0.51% |
| record | service_type | 5 | 0.51% |
| record | company_name | 5 | 0.51% |
| record | brand | 5 | 0.51% |
| record | topic | 5 | 0.51% |
| record | device | 5 | 0.51% |
| record | blood_pressure | 5 | 0.51% |
| record | goals | 5 | 0.51% |
| record | event.date | 5 | 0.51% |
| record | source | 5 | 0.51% |
| record | keywords | 5 | 0.51% |
| record | policyholder_name | 5 | 0.51% |
| record | policy_number | 5 | 0.51% |
| record | duration | 5 | 0.51% |
| record | recipient | 5 | 0.51% |
| record | comments | 5 | 0.51% |
| record | id | 5 | 0.51% |
| record | target_audience | 5 | 0.51% |
| record | tags | 5 | 0.51% |
| record | route | 5 | 0.51% |
| record | sentiment | 5 | 0.51% |
| record | zip_code | 5 | 0.51% |
| record | sender | 5 | 0.51% |
| record | away_team | 5 | 0.51% |
| record | physical_characteristics.observations.0.telescope | 5 | 0.51% |
| record | beneficiary | 5 | 0.51% |
| record | physical_characteristics.observations.0.publication_year | 5 | 0.51% |
| record | battery_life | 5 | 0.51% |
| record | places | 4 | 0.41% |
| record | species.genus | 4 | 0.41% |
| record | product.description | 4 | 0.41% |
| record | product.name | 4 | 0.41% |
| record | publication.year | 4 | 0.41% |
| record | authors | 4 | 0.41% |
| record | physical_characteristics.observations.0.year | 4 | 0.41% |
| record | orbital_characteristics.inclination.unit | 4 | 0.41% |
| record | issue_date | 4 | 0.41% |
| record | frequency | 4 | 0.41% |
| record | course_id | 4 | 0.41% |
| record | instructor | 4 | 0.41% |
| record | asteroid.discovery_year | 4 | 0.41% |
| record | email | 4 | 0.41% |
| record | review_date | 4 | 0.41% |
| record | reviewer_name | 4 | 0.41% |
| record | user | 4 | 0.41% |
| record | assists | 4 | 0.41% |
| record | origin | 4 | 0.41% |
| record | merchant | 4 | 0.41% |
| record | value | 4 | 0.41% |
| record | normal_range | 4 | 0.41% |
| record | vaccine_type | 4 | 0.41% |
| record | risk_score | 4 | 0.41% |
| record | locations | 4 | 0.41% |
| record | client_name | 4 | 0.41% |
| record | resolution_summary | 4 | 0.41% |
| record | provisional_designation | 4 | 0.41% |
| record | campaign_id: Unique identifier for each promotional campaign | 4 | 0.41% |
| record | diagnosis_code | 4 | 0.41% |
| record | policyholder | 4 | 0.41% |
| record | ROOT | 4 | 0.41% |
| record | orbital_characteristics.classification | 4 | 0.41% |
| record | sponsor | 4 | 0.41% |
| record | attendees | 4 | 0.41% |
| record | contact_name | 4 | 0.41% |
| record | procedure | 4 | 0.41% |
| record | report_id | 4 | 0.41% |
| record | organization | 4 | 0.41% |
| record | effective_date | 4 | 0.41% |
| record | orbital_characteristics.semi_major_axis | 4 | 0.41% |
| record | orbital_characteristics.eccentricity | 4 | 0.41% |
| record | orbital_characteristics.inclination | 4 | 0.41% |
| record | severity | 4 | 0.41% |
| record | transfer_fee | 4 | 0.41% |
| record | salary | 4 | 0.41% |
| record | asteroid.discovery.discoverer | 4 | 0.41% |
| record | appointment_id | 4 | 0.41% |
| record | home_team: Team playing at the home venue | 3 | 0.31% |
| record | attendee_count | 3 | 0.31% |
| record | species.scientific_name | 3 | 0.31% |
| record | reporter | 3 | 0.31% |
| record | name: Full name of the patient | 3 | 0.31% |
| record | location: City where the treaty was signed | 3 | 0.31% |
| record | review_length | 3 | 0.31% |
| record | person.birth_place.city | 3 | 0.31% |
| record | person.profession | 3 | 0.31% |
| record | person.us_senator.start_year | 3 | 0.31% |
| record | person.birth_date | 3 | 0.31% |
| record | filing_date | 3 | 0.31% |
| record | user_name | 3 | 0.31% |
| record | family | 3 | 0.31% |
| record | order | 3 | 0.31% |
| record | sample_size | 3 | 0.31% |
| record | policy_id: Unique identifier for the insurance policy | 3 | 0.31% |
| record | side_effects | 3 | 0.31% |
| record | credits | 3 | 0.31% |
| record | acquirer | 3 | 0.31% |
| record | target | 3 | 0.31% |
| record | purpose | 3 | 0.31% |
| record | asteroid_name | 3 | 0.31% |
| record | discovery_date | 3 | 0.31% |
| record | discovery_location | 3 | 0.31% |
| record | replies | 3 | 0.31% |
| record | subreddit | 3 | 0.31% |
| record | upvotes | 3 | 0.31% |
| record | date: Date when the transaction occurred | 3 | 0.31% |
| record | destination_club | 3 | 0.31% |
| record | employee_id: Unique identifier assigned to the employee | 3 | 0.31% |
| record | agent | 3 | 0.31% |
| record | opponent | 3 | 0.31% |
| record | refund_amount | 3 | 0.31% |
| record | quantity | 3 | 0.31% |
| record | product.features | 3 | 0.31% |
| record | call_id | 3 | 0.31% |
| record | donor_name | 3 | 0.31% |
| record | sku | 3 | 0.31% |
| record | prescription_id | 3 | 0.31% |
| record | program_name | 3 | 0.31% |
| record | lab_id | 3 | 0.31% |
| record | vaccination_id | 3 | 0.31% |
| record | device_model | 3 | 0.31% |
| record | initial_status | 3 | 0.31% |
| record | study_id | 3 | 0.31% |
| record | defendant | 3 | 0.31% |
| record | likes | 3 | 0.31% |
| record | shares | 3 | 0.31% |
| record | discovery.observatory.location.county | 3 | 0.31% |
| record | physical_characteristics.observations.0.telescope.type | 3 | 0.31% |
| record | dates | 3 | 0.31% |
| record | bracket_id | 3 | 0.31% |
| record | teams | 3 | 0.31% |
| record | cost | 3 | 0.31% |
| record | amod | 3 | 0.31% |
| record | aux | 3 | 0.31% |
| record | det | 3 | 0.31% |
| record | compound | 3 | 0.31% |
| record | mrn | 3 | 0.31% |
| record | findings | 3 | 0.31% |
| record | genus | 3 | 0.31% |
| record | customer_name: Name of the affected customer | 3 | 0.31% |
| record | word_count | 3 | 0.31% |
| record | helpful_votes | 3 | 0.31% |
| record | clause_id | 3 | 0.31% |
| record | test_date: Date the test was conducted | 3 | 0.31% |
| record | patient_id: Unique identifier for the patient | 3 | 0.31% |
| record | timestamp | 3 | 0.31% |
| record | temperature | 3 | 0.31% |
| record | retention_rate | 3 | 0.31% |
| record | vote_count | 3 | 0.31% |
| record | conference_id | 3 | 0.31% |
| record | website | 3 | 0.31% |
| record | recommendation | 3 | 0.31% |
| record | status: Clinical interpretation of the result | 3 | 0.31% |
| record | invoice_id | 3 | 0.31% |
| record | event_type: Category of the event | 3 | 0.31% |
| record | match_id | 3 | 0.31% |
| record | final_score | 3 | 0.31% |
| record | passenger_name | 3 | 0.31% |
| record | institution | 3 | 0.31% |
| record | address: Patient's residential address | 3 | 0.31% |
| record | symptoms: List of symptoms reported by the patient | 3 | 0.31% |
| record | cpt_code: CPT code associated with the clinical visit type | 3 | 0.31% |
| record | promo_code: Discount or offer code associated with the campaign | 3 | 0.31% |
| record | merchant_address | 3 | 0.31% |
| record | account_id | 3 | 0.31% |
| record | discrepancy_amount | 3 | 0.31% |
| record | location.city | 3 | 0.31% |
| record | orbital_characteristics.perihelion | 3 | 0.31% |
| record | nsubj | 3 | 0.31% |
| record | advmod | 3 | 0.31% |
| record | sleep_hours | 3 | 0.31% |
| record | icd10_code | 3 | 0.31% |
| record | premium | 3 | 0.31% |
| record | address: Full address of the policyholder | 3 | 0.31% |
| record | claim_id: Unique identifier for each claim entry | 3 | 0.31% |
| record | destination | 3 | 0.31% |
| record | zebras_care_foundation.beneficiary.name | 3 | 0.31% |
| record | event.teams | 3 | 0.31% |
| record | zebras_care_foundation.beneficiary.age | 3 | 0.31% |
| record | zebras_care_foundation.introducer.role | 3 | 0.31% |
| record | condition | 3 | 0.31% |
| record | coverage_amount | 3 | 0.31% |
| record | ticket_price | 3 | 0.31% |
| record | discovery_year | 3 | 0.31% |
| record | model | 3 | 0.31% |
| record | supplier | 3 | 0.31% |
| record | affiliation | 3 | 0.31% |
| record | claim_amount | 3 | 0.31% |
| record | total_funding | 3 | 0.31% |
| record | genre | 3 | 0.31% |
| record | hotel.name | 2 | 0.21% |
| record | away_team: Team playing away from home | 2 | 0.21% |
| record | finalscore: Scoreline in format homescore-away_score | 2 | 0.21% |
| record | match_code: Unique identifier for the match as recorded in the database | 2 | 0.21% |
| record | surface | 2 | 0.21% |
| record | filing_type | 2 | 0.21% |
| record | patent_id: Unique identifier for the patent application | 2 | 0.21% |
| record | dream_experiments | 2 | 0.21% |
| record | person.occupation | 2 | 0.21% |
| record | annual_revenue | 2 | 0.21% |
| record | species.description | 2 | 0.21% |
| record | funding | 2 | 0.21% |
| record | conference | 2 | 0.21% |
| record | patient_id: Unique identifier for each patient | 2 | 0.21% |
| record | name: Unique identifier of the treaty agreement | 2 | 0.21% |
| record | funding_type: Description of funding commitment such as operational or adaptation | 2 | 0.21% |
| record | signatories: List of officials who co-signed the treaty | 2 | 0.21% |
| record | closing_price | 2 | 0.21% |
| record | nf_id: Unique identifier for each funding event | 2 | 0.21% |
| record | date: Month and year of funding occurrence | 2 | 0.21% |
| record | amount: Total amount raised in millions | 2 | 0.21% |
| record | investors: List of venture capital firms involved | 2 | 0.21% |
| record | serial_number: Unique serial number for device tracking | 2 | 0.21% |
| record | reviewer: Name of the film critic who wrote the review | 2 | 0.21% |
| record | film_title: Title of the film being reviewed | 2 | 0.21% |
| record | person.us_senator.years | 2 | 0.21% |
| record | person.religion | 2 | 0.21% |
| record | key_concepts | 2 | 0.21% |
| record | product.language | 2 | 0.21% |
| record | insurance_plan | 2 | 0.21% |
| record | discharge_date | 2 | 0.21% |
| record | medications | 2 | 0.21% |
| record | hospital | 2 | 0.21% |
| record | claim_amount: Total monetary value of the claim | 2 | 0.21% |
| record | merchant_code | 2 | 0.21% |
| record | vote_id: Unique identifier for the legislative vote | 2 | 0.21% |
| record | status: Current legislative status of the bill | 2 | 0.21% |
| record | scope: Broad category or focus area of the bill | 2 | 0.21% |
| record | contribution_type | 2 | 0.21% |
| record | location: City and state or province where purchase occurred | 2 | 0.21% |
| record | scientificName | 2 | 0.21% |
| record | discovery.geologicalFormation | 2 | 0.21% |
| record | discovery.age.geologicalPeriod | 2 | 0.21% |
| record | morphology.wingLengthMm | 2 | 0.21% |
| record | publication.author.name | 2 | 0.21% |
| record | publication.author.deathYear | 2 | 0.21% |
| record | publication.author.affiliation.location | 2 | 0.21% |
| record | publication.author.affiliation.country | 2 | 0.21% |
| record | publication.otherTaxaDescribed | 2 | 0.21% |
| record | affiliations | 2 | 0.21% |
| record | object_type | 2 | 0.21% |
| record | orbit.perihelion.unit | 2 | 0.21% |
| record | orbit.aphelion.unit | 2 | 0.21% |
| record | publication.author | 2 | 0.21% |
| record | time_management | 2 | 0.21% |
| record | overall_score | 2 | 0.21% |
| record | target_market | 2 | 0.21% |
| record | discovery.observatory | 2 | 0.21% |
| record | coverage_type: Type of insurance coverage provided | 2 | 0.21% |
| record | availability | 2 | 0.21% |
| record | storage | 2 | 0.21% |
| record | candidate_name | 2 | 0.21% |
| record | total_amount | 2 | 0.21% |
| record | units_sold: Total units sold by a specific date | 2 | 0.21% |
| record | name: Name of the product as listed in best-sellers | 2 | 0.21% |
| record | region: Primary customer region where product is popular | 2 | 0.21% |
| record | cost_savings | 2 | 0.21% |
| record | office_code | 2 | 0.21% |
| record | discovery_observatory | 2 | 0.21% |
| record | status: Current state of the transaction execution | 2 | 0.21% |
| record | transaction_type: Type of transaction as recorded in the system | 2 | 0.21% |
| record | address: Physical address of the payee or recipient | 2 | 0.21% |
| record | experience_level | 2 | 0.21% |
| record | required_skills | 2 | 0.21% |
| record | salary_range | 2 | 0.21% |
| record | hire_date | 2 | 0.21% |
| record | requisition_id | 2 | 0.21% |
| record | property | 2 | 0.21% |
| record | tenant | 2 | 0.21% |
| record | classification_codes: List of classification codes assigned to the patent | 2 | 0.21% |
| record | inventors: List of inventor names involved in the filing | 2 | 0.21% |
| record | Dariusz Śledź <> coached | 2 | 0.21% |
| record | Sparta Wroclaw <> managed by | 2 | 0.21% |
| record | Stal Rzeszów <> managed by | 2 | 0.21% |
| record | establishment_year | 2 | 0.21% |
| record | transfer_type | 2 | 0.21% |
| record | goals_scored | 2 | 0.21% |
| record | position | 2 | 0.21% |
| record | full_name: Name of the employee as recorded in the review | 2 | 0.21% |
| record | location: Physical address of the employee's workplace | 2 | 0.21% |
| record | kpideliveryaccuracy: Percentage of on-time deliveries achieved | 2 | 0.21% |
| record | department: Department where the employee is based | 2 | 0.21% |
| record | customer_satisfaction: Score from customer feedback surveys | 2 | 0.21% |
| record | team | 2 | 0.21% |
| record | player | 2 | 0.21% |
| record | event_type | 2 | 0.21% |
| record | case_id | 2 | 0.21% |
| record | asteroid.named_after.person.name | 2 | 0.21% |
| record | article_metadata.publish_date | 2 | 0.21% |
| record | article_metadata.source | 2 | 0.21% |
| record | organization.short_name | 2 | 0.21% |
| record | organization.full_name | 2 | 0.21% |
| record | startup_name | 2 | 0.21% |
| record | event.person.position | 2 | 0.21% |
| record | event.person.name | 2 | 0.21% |
| record | event.opponent_club.name | 2 | 0.21% |
| record | event.person.role | 2 | 0.21% |
| record | event.new_club.contract_duration | 2 | 0.21% |
| record | event.new_club.name | 2 | 0.21% |
| record | popular | 2 | 0.21% |
| record | product.categories | 2 | 0.21% |
| record | highlights | 2 | 0.21% |
| record | creative work | 2 | 0.21% |
| record | compliance | 2 | 0.21% |
| record | leasing_offer | 2 | 0.21% |
| record | tenant_name | 2 | 0.21% |
| record | color | 2 | 0.21% |
| record | warranty | 2 | 0.21% |
| record | funding_source: Organization or fund providing financial support | 2 | 0.21% |
| record | deployment_count: Number of farms, hospitals, or states using the system | 2 | 0.21% |
| record | sensor_nodes: Number of data collection nodes used in the system | 2 | 0.21% |
| record | outbreak_event.illness.pathogen.full_name | 2 | 0.21% |
| record | outbreak_event.illness.pathogen.abbreviation | 2 | 0.21% |
| record | outbreak_event.illness.pathogen.classification | 2 | 0.21% |
| record | outbreak_event.temporal_data.start_period | 2 | 0.21% |
| record | outbreak_event.location.country | 2 | 0.21% |
| record | outbreak_event.location.setting | 2 | 0.21% |
| record | outbreak_event.investigation.methods | 2 | 0.21% |
| record | outbreak_event.investigation.specimen_source | 2 | 0.21% |
| record | outbreak_event.investigation.findings.agent_details.full_name | 2 | 0.21% |
| record | outbreak_event.investigation.findings.agent_details.abbreviation | 2 | 0.21% |
| record | outbreak_event.public_health_recommendations.target_populations | 2 | 0.21% |
| record | cities | 2 | 0.21% |
| record | session_id | 2 | 0.21% |
| record | speaker | 2 | 0.21% |
| record | test_type | 2 | 0.21% |
| record | vaccination_date | 2 | 0.21% |
| record | municipality.name | 2 | 0.21% |
| record | funding: Monetary commitment in millions of dollars | 2 | 0.21% |
| record | treaty_id: Unique identifier assigned to the treaty | 2 | 0.21% |
| record | treaty | 2 | 0.21% |
| record | movie.title | 2 | 0.21% |
| record | icd10code | 2 | 0.21% |
| record | exploited_ports | 2 | 0.21% |
| record | software_versions | 2 | 0.21% |
| record | sensitivity | 2 | 0.21% |
| record | plaintiff_address | 2 | 0.21% |
| record | case_type | 2 | 0.21% |
| record | key_player | 2 | 0.21% |
| record | institution: Name of the financial institution publishing the rate | 2 | 0.21% |
| record | rate: Exchange rate value for the currency pair | 2 | 0.21% |
| record | validation_status: Current validation state of the rate data | 2 | 0.21% |
| record | source_type: Type of data source such as interbank feed or real-time trading | 2 | 0.21% |
| record | verse_reference | 2 | 0.21% |
| record | topics | 2 | 0.21% |
| record | program_name: Title of the training program enrolled in | 2 | 0.21% |
| record | product_name: Name of the launched product | 2 | 0.21% |
| record | creator | 2 | 0.21% |
| record | date_posted | 2 | 0.21% |
| record | registration_fee | 2 | 0.21% |
| record | presenter | 2 | 0.21% |
| record | attendance_limit | 2 | 0.21% |
| record | route: Origin-destination route string like JFK-LAX | 2 | 0.21% |
| record | delay_reason: Human-readable description of delay cause | 2 | 0.21% |
| record | destination: Airport code where flight arrives | 2 | 0.21% |
| record | passenger_names: List of passenger names affected | 2 | 0.21% |
| record | compensation_offered: Amount offered in dollars per passenger | 2 | 0.21% |
| record | flight_number: Unique identifier for the delayed flight | 2 | 0.21% |
| record | sec_id | 2 | 0.21% |
| record | revenue | 2 | 0.21% |
| record | discovery.project | 2 | 0.21% |
| record | number | 2 | 0.21% |
| record | disputingpartyb | 2 | 0.21% |
| record | disputingpartya | 2 | 0.21% |
| record | status: Current operational status of the campaign | 2 | 0.21% |
| record | product_id: Unique identifier for each product in the best-sellers list | 2 | 0.21% |
| record | unit_price: Price per unit of the product | 2 | 0.21% |
| record | seller: Company name that sells the product | 2 | 0.21% |
| record | seller_location: Full address of the seller including city, state, and zip code | 2 | 0.21% |
| record | amount_billed | 2 | 0.21% |
| record | video_id | 2 | 0.21% |
| record | upload_date | 2 | 0.21% |
| record | view_count | 2 | 0.21% |
| record | disclaimer | 2 | 0.21% |
| record | product_example.currency | 2 | 0.21% |
| record | availability.locations.retail_outlets | 2 | 0.21% |
| record | distributor.location | 2 | 0.21% |
| record | product_example.name | 2 | 0.21% |
| record | availability.locations.online_stores.0.url | 2 | 0.21% |
| record | visit_date | 2 | 0.21% |
| record | visit_type | 2 | 0.21% |
| record | paymentduedays: Number of days until payment is due after notice | 2 | 0.21% |
| record | enrollment_date | 2 | 0.21% |
| record | booth | 2 | 0.21% |
| record | rr | 2 | 0.21% |
| record | patient_id: Unique identifier assigned to the patient | 2 | 0.21% |
| record | warranty_period | 2 | 0.21% |
| record | playertwoid | 2 | 0.21% |
| record | match_type | 2 | 0.21% |
| record | playeronename | 2 | 0.21% |
| record | match_result | 2 | 0.21% |
| record | rsvp_count | 2 | 0.21% |
| record | conj | 2 | 0.21% |
| record | poss | 2 | 0.21% |
| record | npadvmod | 2 | 0.21% |
| record | imaging_date | 2 | 0.21% |
| record | authority.year | 2 | 0.21% |
| record | subgenus | 2 | 0.21% |
| record | commonName | 2 | 0.21% |
| record | species | 2 | 0.21% |
| record | submitter | 2 | 0.21% |
| record | fee | 2 | 0.21% |
| record | bonus_amount | 2 | 0.21% |
| record | post_id: Unique identifier for the social media post | 2 | 0.21% |
| record | medication_name | 2 | 0.21% |
| record | approval_year | 2 | 0.21% |
| record | pharmacy | 2 | 0.21% |
| record | reason | 2 | 0.21% |
| record | os_version | 2 | 0.21% |
| record | location: Venue where the match was held | 2 | 0.21% |
| record | venue_name: Name of the specific tournament venue | 2 | 0.21% |
| record | reference_number: Unique identifier for the invoice document | 2 | 0.21% |
| record | genus.name | 2 | 0.21% |
| record | genus.publication | 2 | 0.21% |
| record | genus.distribution | 2 | 0.21% |
| record | genus.pollination.mechanism | 2 | 0.21% |
| record | genus.pollination.pollinators | 2 | 0.21% |
| record | genus.etymology.part1.name | 2 | 0.21% |
| record | genus.etymology.part1.origin | 2 | 0.21% |
| record | genus.etymology.part2.meaning | 2 | 0.21% |
| record | genus.characteristics | 2 | 0.21% |
| record | genus.species_list | 2 | 0.21% |
| record | genus.related_genera | 2 | 0.21% |
| record | education | 2 | 0.21% |
| record | result_value: Measured value from the test | 2 | 0.21% |
| record | test_type: Type of laboratory test performed | 2 | 0.21% |
| record | market_cap | 2 | 0.21% |
| record | ticker | 2 | 0.21% |
| record | daily_value_percentage | 2 | 0.21% |
| record | creator_name | 2 | 0.21% |
| record | lead_investor: Primary investor leading the funding round | 2 | 0.21% |
| record | country: Country code from patent number prefix | 2 | 0.21% |
| record | filing_date: Date the patent was filed | 2 | 0.21% |
| record | opposition_count | 2 | 0.21% |
| record | language | 2 | 0.21% |
| record | submission_id: Unique identifier assigned to each abstract submission | 2 | 0.21% |
| record | word_count: Number of words in the abstract | 2 | 0.21% |
| record | funding_code: Code indicating eligibility for funding support | 2 | 0.21% |
| record | name: Full name of the enrolled individual | 2 | 0.21% |
| record | enrollment_date: Date when enrollment began | 2 | 0.21% |
| record | program: Training or program name enrolled in | 2 | 0.21% |
| record | id: Unique identifier for the enrollment record | 2 | 0.21% |
| record | location: Geographic location of the enrollment site | 2 | 0.21% |
| record | finding_description | 2 | 0.21% |
| record | referring_physician | 2 | 0.21% |
| record | imaging_type | 2 | 0.21% |
| record | study_code | 2 | 0.21% |
| record | study_protocol | 2 | 0.21% |
| record | defendant_address | 2 | 0.21% |
| record | submitter: Name of employee or partner who reported the incident | 2 | 0.21% |
| record | sender_domain: Domain used in the fraudulent email | 2 | 0.21% |
| record | collection_site: Name of the clinic or lab where test was collected | 2 | 0.21% |
| record | normal_range: Reference range for the measured value | 2 | 0.21% |
| record | warehouse_location: Geographic warehouse where product is stored | 2 | 0.21% |
| record | last_updated: Date when product record was last modified | 2 | 0.21% |
| record | confirmation_code | 2 | 0.21% |
| record | subscription_id | 2 | 0.21% |
| record | therapy_basis | 2 | 0.21% |
| record | patient_care.group_therapy.benefits | 2 | 0.21% |
| record | patient_care.group_therapy.importance | 2 | 0.21% |
| record | treatment | 2 | 0.21% |
| record | symptoms | 2 | 0.21% |
| record | people | 2 | 0.21% |
| record | distance.parsecs | 2 | 0.21% |
| record | latin_name | 2 | 0.21% |
| record | distance.calculation_method | 2 | 0.21% |
| record | characteristics.effective_temperature.value | 2 | 0.21% |
| record | political_entity | 2 | 0.21% |
| record | medals.total | 2 | 0.21% |
| record | games | 2 | 0.21% |
| record | modality | 2 | 0.21% |
| record | asteroid.designation | 2 | 0.21% |
| record | duration: Duration of outage in hours | 2 | 0.21% |
| record | address: Full street address of the claimant | 2 | 0.21% |
| record | incident_type: Type of incident causing the claim | 2 | 0.21% |
| record | office_location | 2 | 0.21% |
| record | airline | 2 | 0.21% |
| record | passenger_email | 2 | 0.21% |
| record | outcome | 2 | 0.21% |
| record | vehicle | 2 | 0.21% |
| record | date: Date the transaction was initiated | 2 | 0.21% |
| record | hash: Cryptographic hash of the transaction data | 2 | 0.21% |
| record | additional_value: Extra numeric value associated with the transaction | 2 | 0.21% |
| record | owner: Name of the transaction owner | 2 | 0.21% |
| record | currency: Currency type used in the transaction | 2 | 0.21% |
| record | patient_name: Full name of the patient completing the checklist | 2 | 0.21% |
| record | age: Patient's age at time of visit | 2 | 0.21% |
| record | checklist_id: Unique identifier for the symptom checklist form | 2 | 0.21% |
| record | visit_type: Type of medical visit based on urgency | 2 | 0.21% |
| record | status: Current status of the checklist record | 2 | 0.21% |
| record | targetagegroup: Age range of intended audience | 2 | 0.21% |
| record | merchant_name | 2 | 0.21% |
| record | receipt_id | 2 | 0.21% |
| record | transaction_type | 2 | 0.21% |
| record | definition | 2 | 0.21% |
| record | examples | 2 | 0.21% |
| record | organization.name | 2 | 0.21% |
| record | research.experiment_details.speckle_details | 2 | 0.21% |
| record | research.experiment_details.outcome | 2 | 0.21% |
| record | research.researcher.name | 2 | 0.21% |
| record | research.researcher.affiliation.name | 2 | 0.21% |
| record | research.title | 2 | 0.21% |
| record | sales_trend | 2 | 0.21% |
| record | asset | 2 | 0.21% |
| record | product_name: Name of the product being promoted | 2 | 0.21% |
| record | pricepernight | 2 | 0.21% |
| record | monthly_price: Subscription cost per month in USD | 2 | 0.21% |
| record | pricing_tier: Subscription level such as monthly or free | 2 | 0.21% |
| record | discovery.observatory.location.city | 2 | 0.21% |
| record | discovery.observatory.location.prefecture | 2 | 0.21% |
| record | physical_characteristics.absolute_magnitude | 2 | 0.21% |
| record | diameter | 2 | 0.21% |
| record | xcomp | 2 | 0.21% |
| record | punct | 2 | 0.21% |
| record | ccomp | 2 | 0.21% |
| record | acl | 2 | 0.21% |
| record | pobj | 2 | 0.21% |
| record | nsubjpass | 2 | 0.21% |
| record | first_client | 2 | 0.21% |
| record | year_founded | 2 | 0.21% |
| record | symptom_fever | 2 | 0.21% |
| record | schema_name | 2 | 0.21% |
| record | address: Full street address of the property | 2 | 0.21% |
| record | policy_id: Unique identifier assigned to the policy | 2 | 0.21% |
| record | policy_type: Type of insurance policy renewed | 2 | 0.21% |
| record | policyholder_name: Full name of the claimant | 2 | 0.21% |
| record | claim_id: Unique identifier for the submitted claim | 2 | 0.21% |
| record | cause | 2 | 0.21% |
| record | support_ticket | 2 | 0.21% |
| record | resolution_time | 2 | 0.21% |
| record | city: City where patient resides | 2 | 0.21% |
| record | division | 2 | 0.21% |
| record | diagnosis_code: Medical code associated with the claim condition | 2 | 0.21% |
| record | contributor_count | 2 | 0.21% |
| record | project_url | 2 | 0.21% |
| record | transfer_portal | 2 | 0.21% |
| record | consideration_cash | 2 | 0.21% |
| record | consideration_stock | 2 | 0.21% |
| record | policyholder: Name of the vehicle owner filing the claim | 2 | 0.21% |
| record | target_audience: Demographic or user segment targeted by the campaign | 2 | 0.21% |
| record | length | 2 | 0.21% |
| record | drug_name: Name of the pharmaceutical product | 2 | 0.21% |
| record | route: Administration method of the drug | 2 | 0.21% |
| record | target_receptor: Biological receptor the drug acts on | 2 | 0.21% |
| record | mechanismofaction: Biological process inhibited or modulated by the drug | 2 | 0.21% |
| record | molecular_weight: Molecular weight of the drug in daltons | 2 | 0.21% |
| record | adverse_effects: List of side effects reported in clinical use | 2 | 0.21% |
| record | lastreviewdate: Date the drug record was last reviewed | 2 | 0.21% |
| record | condition_studied | 2 | 0.21% |
| record | submission_method | 2 | 0.21% |
| record | filing_amount | 2 | 0.21% |
| record | donation_date | 2 | 0.21% |
| record | articleTitle | 2 | 0.21% |
| record | delay_cause | 2 | 0.21% |
| record | delay_date | 2 | 0.21% |
| record | flight_id | 2 | 0.21% |
| record | appointment_code | 2 | 0.21% |
| record | background | 2 | 0.21% |
| record | zebras_care_foundation.beneficiary.condition | 2 | 0.21% |
| record | zebras_care_foundation.beneficiary.treatment | 2 | 0.21% |
| record | zebras_care_foundation.introducer.name | 2 | 0.21% |
| record | discovery.observatory.location.state | 2 | 0.21% |
| record | physical_characteristics.observations.0.telescope.name | 2 | 0.21% |
| record | external_links.0.title | 2 | 0.21% |
| record | observations.last_observation_date | 2 | 0.21% |
| record | classification.type | 2 | 0.21% |
| record | classification.source | 2 | 0.21% |
| record | discovery.location.place | 2 | 0.21% |
| record | employee_id: Unique identifier assigned to each employee | 2 | 0.21% |
| record | themes | 2 | 0.21% |
| record | person.death | 2 | 0.21% |
| record | content | 2 | 0.21% |
| record | symptom_fatigue | 2 | 0.21% |
| record | weight_loss_kg | 2 | 0.21% |
| record | lab_tests_ordered | 2 | 0.21% |
| record | referee | 2 | 0.21% |
| record | person.designation | 2 | 0.21% |
| record | connectivity | 2 | 0.21% |
| record | market_segment | 2 | 0.21% |
| record | return_id | 2 | 0.21% |
| record | order_id | 2 | 0.21% |
| record | languages | 2 | 0.21% |
| record | asteroid.location | 2 | 0.21% |
| record | transfer_id | 2 | 0.21% |
| record | player_age | 2 | 0.21% |
| record | playerbirthcity | 2 | 0.21% |
| record | origin_club | 2 | 0.21% |
| record | unit: Unit of measurement for the result value | 2 | 0.21% |
| record | reference_id | 2 | 0.21% |
| record | donor_location: City and state of the donor | 2 | 0.21% |
| record | contribution_id: Unique identifier for the contribution record | 2 | 0.21% |
| record | recipient_position: Office or position the contribution supports | 2 | 0.21% |
| record | festival_id | 2 | 0.21% |
| record | parties | 2 | 0.21% |
| record | eps | 2 | 0.21% |
| record | study_metadata.measurement_technology.anatomical_site | 2 | 0.21% |
| record | outcomes.data_source | 2 | 0.21% |
| record | population.gender | 2 | 0.21% |
| record | conclusions | 2 | 0.21% |
| record | statistical_results.predictive_model.value | 2 | 0.21% |
| record | population.age_range.max | 2 | 0.21% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.0.t_score | 2 | 0.21% |
| record | follow_up.average_duration.months | 2 | 0.21% |
| record | statistical_results.hazard_ratios.0.value | 2 | 0.21% |
| record | outcomes.fracture_events.0.count | 2 | 0.21% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.1.rate_percentage | 2 | 0.21% |
| record | statistical_results.hazard_ratios.0.comparison | 2 | 0.21% |
| record | follow_up.average_duration.years | 2 | 0.21% |
| record | nhl_officials | 2 | 0.21% |
| record | sender_location | 2 | 0.21% |
| record | location.region | 2 | 0.21% |
| record | asteroid_number | 2 | 0.21% |
| record | invoice_number | 2 | 0.21% |
| record | organizers | 2 | 0.21% |
| record | edition | 2 | 0.21% |
| record | event_name | 2 | 0.21% |
| record | series | 2 | 0.21% |
| record | subject | 2 | 0.21% |
| record | performance_rating | 2 | 0.21% |
| record | funding_amount | 2 | 0.21% |
| record | delay_reason | 2 | 0.21% |
| record | delay_minutes | 2 | 0.21% |
| record | cpt_code | 2 | 0.21% |
| record | disease.pathogenesis.transmission | 2 | 0.21% |
| record | disease.subspecies.2.name | 2 | 0.21% |
| record | disease.subspecies.0.affected_species | 2 | 0.21% |
| record | disease.subspecies.1.name | 2 | 0.21% |
| record | disease.subspecies.0.name | 2 | 0.21% |
| record | disease.description | 2 | 0.21% |
| record | disease.subspecies.1.associated_diseases | 2 | 0.21% |
| record | department: Government department or agency where appointed | 2 | 0.21% |
| record | reporting_to: Name of superior or reporting authority | 2 | 0.21% |
| record | name: Full name of the appointed official | 2 | 0.21% |
| record | title: Official job title or position | 2 | 0.21% |
| record | status: Current status of appointment | 2 | 0.21% |
| record | acquisition_id | 2 | 0.21% |
| record | deal_value | 2 | 0.21% |
| record | unit_id | 2 | 0.21% |
| record | type | 2 | 0.21% |
| record | resolution_status | 2 | 0.21% |
| record | case_number: Unique identifier for the court case | 2 | 0.21% |
| record | commercial_rollout | 2 | 0.21% |
| record | monthly_premium | 2 | 0.21% |
| record | term_years | 2 | 0.21% |
| record | asteroid.discovery.location | 2 | 0.21% |
| record | rebounds | 2 | 0.21% |
| record | payee | 2 | 0.21% |
| record | product_offered | 2 | 0.21% |
| record | shipping_options | 2 | 0.21% |
| record | lead_investor | 2 | 0.21% |
| record | yield_improvement | 2 | 0.21% |
| record | key_metric | 2 | 0.21% |
| record | points | 2 | 0.21% |
| record | status: Current status of the transaction | 2 | 0.21% |
| record | asteroid.description | 2 | 0.21% |
| record | geographic_scope | 2 | 0.21% |
| record | person.us_senator.state | 2 | 0.21% |
| record | person.ideology | 2 | 0.21% |
| record | person.birth_place.state | 2 | 0.21% |
| record | bedrooms | 2 | 0.21% |
| record | network_id | 2 | 0.21% |
| record | expression_level | 2 | 0.21% |
| record | diagnosis_date | 2 | 0.21% |
| record | allergy | 2 | 0.21% |
| record | bmi | 2 | 0.21% |
| record | hotel_name | 2 | 0.21% |
| record | insurance | 2 | 0.21% |
| record | publication_date | 2 | 0.21% |
| record | facility: Facility name and address where device is located | 2 | 0.21% |
| record | canton_name_french | 2 | 0.21% |
| record | municipalities | 2 | 0.21% |
| record | guideline_id | 2 | 0.21% |
| record | landlord | 2 | 0.21% |
| record | days | 2 | 0.21% |
| record | enrollmentstatus | 2 | 0.21% |
| record | status: Current phase of the event lifecycle | 2 | 0.21% |
| record | attendees: Number of people registered for the event | 2 | 0.21% |
| record | company_id: Unique identifier assigned to each company | 2 | 0.21% |
| record | status: Current operational or strategic status | 2 | 0.21% |
| record | headquarters: City and country where company is headquartered | 2 | 0.21% |
| record | manager | 2 | 0.21% |
| record | sector | 2 | 0.21% |
| record | hotel.location.district | 1 | 0.10% |
| record | hotel.location.nearby_attractions | 1 | 0.10% |
| record | hotel.room_features | 1 | 0.10% |
| record | hotel.amenities | 1 | 0.10% |
| record | hotel.hotel_group | 1 | 0.10% |
| record | funding_amount: Initial capital received in USD | 1 | 0.10% |
| record | focus_area: Primary domain or mission of the company | 1 | 0.10% |
| record | motivation: Personal reason driving the company's founding | 1 | 0.10% |
| record | founder_name: Full name of the founder as stated in passage | 1 | 0.10% |
| record | physical property | 1 | 0.10% |
| record | organic molecule | 1 | 0.10% |
| record | carotenoid | 1 | 0.10% |
| record | biological tissue | 1 | 0.10% |
| record | access_type | 1 | 0.10% |
| record | finding_id | 1 | 0.10% |
| record | return_id: Unique identifier for the return transaction | 1 | 0.10% |
| record | customer_name: Name of the customer initiating the return | 1 | 0.10% |
| record | address: Full physical address of the customer | 1 | 0.10% |
| record | order_id: Identifier of the original order associated with the return | 1 | 0.10% |
| record | return_date: Date when the return was submitted | 1 | 0.10% |
| record | amount: Monetary value of the returned item | 1 | 0.10% |
| record | product_name: Name of the product returned | 1 | 0.10% |
| record | return_reason: Description of why the product was returned | 1 | 0.10% |
| record | address: Full physical address of the leased property | 1 | 0.10% |
| record | tenant: Name of the company leasing the space | 1 | 0.10% |
| record | status: Current status of the lease agreement | 1 | 0.10% |
| record | entity_location | 1 | 0.10% |
| record | founder_id | 1 | 0.10% |
| record | technology_focus: Key technology or application area of the patent | 1 | 0.10% |
| record | inventors: List of individuals who filed the patent | 1 | 0.10% |
| record | claims: Number of claims covered by the patent | 1 | 0.10% |
| record | status: Current status of the patent | 1 | 0.10% |
| record | devices | 1 | 0.10% |
| record | person.birth_location.country | 1 | 0.10% |
| record | person.birth_location.country_extended | 1 | 0.10% |
| record | person.death_date | 1 | 0.10% |
| record | person.death_location.region | 1 | 0.10% |
| record | key_challenge | 1 | 0.10% |
| record | founding_year | 1 | 0.10% |
| record | acquisition_year | 1 | 0.10% |
| record | opening_quarter | 1 | 0.10% |
| record | grant_code | 1 | 0.10% |
| record | stock_ticker | 1 | 0.10% |
| record | species.family | 1 | 0.10% |
| record | age: Age of the patient in years | 1 | 0.10% |
| record | address: Street address of the patient | 1 | 0.10% |
| record | clinicorhospital: Name of the clinic or hospital | 1 | 0.10% |
| record | temperature: Body temperature in degrees Fahrenheit | 1 | 0.10% |
| record | heart_rate: Heart rate in beats per minute | 1 | 0.10% |
| record | blood_pressure: Blood pressure in mmHg format | 1 | 0.10% |
| record | status: Current clinical status of the patient | 1 | 0.10% |
| record | competitor_id | 1 | 0.10% |
| record | product_focus | 1 | 0.10% |
| record | volume | 1 | 0.10% |
| record | device_id: Unique identifier assigned to the medical device | 1 | 0.10% |
| record | usage_count: Total patient admissions or scans performed | 1 | 0.10% |
| record | report_id: Unique identifier for the phishing report | 1 | 0.10% |
| record | submitter: Name of the user who reported the phishing incident | 1 | 0.10% |
| record | location: Geographic office or city where the report originated | 1 | 0.10% |
| record | sender_email: Email address of the phishing sender | 1 | 0.10% |
| record | restaurant_name | 1 | 0.10% |
| record | price_range | 1 | 0.10% |
| record | product.type | 1 | 0.10% |
| record | provider_npi | 1 | 0.10% |
| record | claim_status | 1 | 0.10% |
| record | author: Name of user who posted the content | 1 | 0.10% |
| record | content: Text or description of the post content | 1 | 0.10% |
| record | post_id: Unique identifier assigned to the post | 1 | 0.10% |
| record | rating: Star rating given by the reviewer out of 5 | 1 | 0.10% |
| record | status: Current status of the review | 1 | 0.10% |
| record | model: Unique identifier or model number of the device | 1 | 0.10% |
| record | word_count: Number of words in the review text | 1 | 0.10% |
| record | admission_date | 1 | 0.10% |
| record | vital_signs | 1 | 0.10% |
| record | follow_up | 1 | 0.10% |
| record | diagnosis: Medical condition coded using ICD-10 standard | 1 | 0.10% |
| record | policy_number: Insurance policy under which claim was submitted | 1 | 0.10% |
| record | sponsor: Individual legislator who introduced the bill | 1 | 0.10% |
| record | yes_votes: Number of members voting in favor | 1 | 0.10% |
| record | no_votes: Number of members voting against | 1 | 0.10% |
| record | wallet_address | 1 | 0.10% |
| record | current_scale | 1 | 0.10% |
| record | mission | 1 | 0.10% |
| record | early_challenge | 1 | 0.10% |
| record | tv_program.year | 1 | 0.10% |
| record | tv_program.production.production_area | 1 | 0.10% |
| record | tv_program.production.location.institution | 1 | 0.10% |
| record | tv_program.setting.time_period | 1 | 0.10% |
| record | tv_program.production.channel | 1 | 0.10% |
| record | tv_program.setting.location | 1 | 0.10% |
| record | manifest_id | 1 | 0.10% |
| record | consignee | 1 | 0.10% |
| record | destination_address | 1 | 0.10% |
| record | donor | 1 | 0.10% |
| record | patient_address: Full address of the patient | 1 | 0.10% |
| record | patient_name: Name of the patient filing the claim | 1 | 0.10% |
| record | diagnosis_code: Medical code assigned to the patient's condition | 1 | 0.10% |
| record | claim_amount: Total cost of the medical service billed | 1 | 0.10% |
| record | historical_events | 1 | 0.10% |
| record | metabolite | 1 | 0.10% |
| record | name: Full name of the player involved in transfer | 1 | 0.10% |
| record | nationality: Country of origin of the player | 1 | 0.10% |
| record | position: Playing position such as midfielder, forward, defender | 1 | 0.10% |
| record | age: Player's age at time of transfer | 1 | 0.10% |
| record | transfer_fee: Amount paid in million euros for the transfer | 1 | 0.10% |
| record | contract_length: Duration of contract in years | 1 | 0.10% |
| record | appearances: Total games played in last season | 1 | 0.10% |
| record | goals: Goals scored in last season | 1 | 0.10% |
| record | status: Current status of transfer | 1 | 0.10% |
| record | club_from: Previous club before transfer | 1 | 0.10% |
| record | club_to: New club after transfer | 1 | 0.10% |
| record | player_id: Unique identifier assigned to the player transfer record | 1 | 0.10% |
| record | user: Name of the customer who provided the review | 1 | 0.10% |
| record | rating: Numerical rating from 1 to 5 given by the user | 1 | 0.10% |
| record | discovery.location | 1 | 0.10% |
| record | discovery.age.approximateAgeMillionYears | 1 | 0.10% |
| record | discovery.age.geologicalStage | 1 | 0.10% |
| record | publication.author.birthYear | 1 | 0.10% |
| record | publication.author.affiliation.institution | 1 | 0.10% |
| record | publication.genusAssignment.genus | 1 | 0.10% |
| record | publication.genusAssignment.year | 1 | 0.10% |
| record | study_start | 1 | 0.10% |
| record | physical_properties.diameter.value | 1 | 0.10% |
| record | physical_properties.diameter.unit | 1 | 0.10% |
| record | alternate_designations | 1 | 0.10% |
| record | classification | 1 | 0.10% |
| record | orbit.semi_major_axis.unit | 1 | 0.10% |
| record | program | 1 | 0.10% |
| record | feedback_method | 1 | 0.10% |
| record | posted_on | 1 | 0.10% |
| record | address: Full residential address of the policyholder | 1 | 0.10% |
| record | renewal_reason: Justification for renewal or decline | 1 | 0.10% |
| record | policyholder_name: Full name of the policyholder | 1 | 0.10% |
| record | pharmacy_address | 1 | 0.10% |
| record | pharmacy_name | 1 | 0.10% |
| record | starting_dose | 1 | 0.10% |
| record | contraindications | 1 | 0.10% |
| record | shelf_life | 1 | 0.10% |
| record | drug_interactions | 1 | 0.10% |
| record | election_type | 1 | 0.10% |
| record | candidate_location | 1 | 0.10% |
| record | comment | 1 | 0.10% |
| record | location-gpe | 1 | 0.10% |
| record | price: Product price in USD format | 1 | 0.10% |
| record | asteroid.orbital_characteristics.unit | 1 | 0.10% |
| record | asteroid.orbital_characteristics.inclination_unit | 1 | 0.10% |
| record | regulatory_status.recognition_type | 1 | 0.10% |
| record | announcement_metadata.date | 1 | 0.10% |
| record | tax_incentive.duration.period.end_year | 1 | 0.10% |
| record | regulatory_status.review_status | 1 | 0.10% |
| record | announcement_metadata.source_url | 1 | 0.10% |
| record | tax_incentive.duration.period.start_year | 1 | 0.10% |
| record | revenue_uptick | 1 | 0.10% |
| record | hotel.alternative_names | 1 | 0.10% |
| record | review.travelers.0.relationship | 1 | 0.10% |
| record | review.positive_aspects | 1 | 0.10% |
| record | review.travelers.0.age | 1 | 0.10% |
| record | industry_example.industry | 1 | 0.10% |
| record | definition.source_organization.full_name | 1 | 0.10% |
| record | financial_aspect.necessity | 1 | 0.10% |
| record | definition.document_title | 1 | 0.10% |
| record | definition.text | 1 | 0.10% |
| record | personal_sustainability | 1 | 0.10% |
| record | three_pillars | 1 | 0.10% |
| record | industry_example.status | 1 | 0.10% |
| record | benefits | 1 | 0.10% |
| record | aspects | 1 | 0.10% |
| record | definition.source_organization.name | 1 | 0.10% |
| record | source_name | 1 | 0.10% |
| record | open_date | 1 | 0.10% |
| record | innovation_type | 1 | 0.10% |
| record | enrollment_code | 1 | 0.10% |
| record | supervisor | 1 | 0.10% |
| record | location: Headquarters city and state or country | 1 | 0.10% |
| record | focus: Primary technology or business area of the company | 1 | 0.10% |
| record | amount: Total capital raised in millions | 1 | 0.10% |
| record | post_valuation: Post-money valuation in millions | 1 | 0.10% |
| record | job_title | 1 | 0.10% |
| record | actions.0.location | 1 | 0.10% |
| record | actions.0.target | 1 | 0.10% |
| record | summary | 1 | 0.10% |
| record | launch_date | 1 | 0.10% |
| record | launched_by | 1 | 0.10% |
| record | assignee: Entity owning the patent application | 1 | 0.10% |
| record | Dariusz Śledź <> nationality | 1 | 0.10% |
| record | Dariusz Śledź <> occupation | 1 | 0.10% |
| record | performance_metric | 1 | 0.10% |
| record | weather | 1 | 0.10% |
| record | firmware_version | 1 | 0.10% |
| record | link | 1 | 0.10% |
| record | description: Brief explanation of the transaction purpose | 1 | 0.10% |
| record | planetary_system.planets.0.discovery_year | 1 | 0.10% |
| record | star.constellation | 1 | 0.10% |
| record | star.distance.light_years | 1 | 0.10% |
| record | planetary_system.planets.0.designation | 1 | 0.10% |
| record | star.effective_temperature | 1 | 0.10% |
| record | planetary_system.planets.0.discovery_team | 1 | 0.10% |
| record | patient_mrn | 1 | 0.10% |
| record | diagnosis_icd10 | 1 | 0.10% |
| record | staff.teaching_staff_count | 1 | 0.10% |
| record | faculty_name | 1 | 0.10% |
| record | map_location | 1 | 0.10% |
| record | campus | 1 | 0.10% |
| record | departments | 1 | 0.10% |
| record | dean.name | 1 | 0.10% |
| record | salary_max | 1 | 0.10% |
| record | tickets | 1 | 0.10% |
| record | news_id | 1 | 0.10% |
| record | match_score | 1 | 0.10% |
| record | game_status | 1 | 0.10% |
| record | performance_notes | 1 | 0.10% |
| record | credit_hours | 1 | 0.10% |
| record | enrollment_status | 1 | 0.10% |
| record | meeting_days | 1 | 0.10% |
| record | enrollment_capacity | 1 | 0.10% |
| record | investor_name | 1 | 0.10% |
| record | platform | 1 | 0.10% |
| record | game title | 1 | 0.10% |
| record | target audience | 1 | 0.10% |
| record | sales | 1 | 0.10% |
| record | product.execution_file | 1 | 0.10% |
| record | product.included_software | 1 | 0.10% |
| record | description: Brief text describing the content of the post | 1 | 0.10% |
| record | user_name: Name of the user who created the repost | 1 | 0.10% |
| record | likes: Number of likes the post received | 1 | 0.10% |
| record | status: Current status of the trend repost | 1 | 0.10% |
| record | trend_name: Name of the viral trend being reposted | 1 | 0.10% |
| record | donor_address | 1 | 0.10% |
| record | race | 1 | 0.10% |
| record | favorite_players | 1 | 0.10% |
| record | edit_information.timestamp | 1 | 0.10% |
| record | sentiment.opinion | 1 | 0.10% |
| record | quote.text | 1 | 0.10% |
| record | sentiment.preference | 1 | 0.10% |
| record | arguments_for_using_gdz | 1 | 0.10% |
| record | causeofloss | 1 | 0.10% |
| record | damage_amount | 1 | 0.10% |
| record | size | 1 | 0.10% |
| record | stock_status | 1 | 0.10% |
| record | weight | 1 | 0.10% |
| record | dimensions | 1 | 0.10% |
| record | shipping | 1 | 0.10% |
| record | available_units | 1 | 0.10% |
| record | orbit.orbital_type | 1 | 0.10% |
| record | characteristics.diameter.value | 1 | 0.10% |
| record | orbit.orbital_period.unit | 1 | 0.10% |
| record | project_id: Unique identifier assigned to the technology initiative | 1 | 0.10% |
| record | deployment_location: Geographic region where system is deployed | 1 | 0.10% |
| record | event.person.age | 1 | 0.10% |
| record | technologies_mentioned | 1 | 0.10% |
| record | companies_mentioned | 1 | 0.10% |
| record | applications | 1 | 0.10% |
| record | users | 1 | 0.10% |
| record | outbreak_event.illness.description | 1 | 0.10% |
| record | outbreak_event.temporal_data.duration_context | 1 | 0.10% |
| record | outbreak_event.investigation.findings.agent_details.strain_type | 1 | 0.10% |
| record | outbreak_event.investigation.findings.agent_details.classification | 1 | 0.10% |
| record | outbreak_event.public_health_recommendations.action_required | 1 | 0.10% |
| record | lease_id: Unique identifier for each lease agreement | 1 | 0.10% |
| record | property_location: Full address including street, suite, and city | 1 | 0.10% |
| record | tenant_name: Name of the leasing company or business | 1 | 0.10% |
| record | domain.registry.name | 1 | 0.10% |
| record | domain.territory | 1 | 0.10% |
| record | domain.third_level_domains | 1 | 0.10% |
| record | domain.dispute_resolution.methods | 1 | 0.10% |
| record | system_code | 1 | 0.10% |
| record | risk_category | 1 | 0.10% |
| record | underwriter | 1 | 0.10% |
| record | property_address | 1 | 0.10% |
| record | defense_systems | 1 | 0.10% |
| record | facilities | 1 | 0.10% |
| record | supply_duration | 1 | 0.10% |
| record | streaming_platform | 1 | 0.10% |
| record | end_time | 1 | 0.10% |
| record | model_number | 1 | 0.10% |
| record | collaboration | 1 | 0.10% |
| record | recipient: Name of the recipient professional contacted | 1 | 0.10% |
| record | message_length: Word count of the outreach message | 1 | 0.10% |
| record | subject: Subject line of the outreach email | 1 | 0.10% |
| record | email_id: Unique identifier for the outreach email | 1 | 0.10% |
| record | sender: Name of the recruiter initiating the outreach | 1 | 0.10% |
| record | movie.production_year | 1 | 0.10% |
| record | movie.comments.8.time | 1 | 0.10% |
| record | movie.comments.13.date | 1 | 0.10% |
| record | movie.comments.3.date | 1 | 0.10% |
| record | movie.comments.12.text | 1 | 0.10% |
| record | movie.comments.7.date | 1 | 0.10% |
| record | movie.rating.max_score | 1 | 0.10% |
| record | movie.comments.19.text | 1 | 0.10% |
| record | movie.comments.14.date | 1 | 0.10% |
| record | movie.comments.13.time | 1 | 0.10% |
| record | movie.comments.18.time | 1 | 0.10% |
| record | movie.comments.18.date | 1 | 0.10% |
| record | movie.comments.12.date | 1 | 0.10% |
| record | movie.comments.19.time | 1 | 0.10% |
| record | movie.comments.2.time | 1 | 0.10% |
| record | movie.comments.16.date | 1 | 0.10% |
| record | movie.comments.21.time | 1 | 0.10% |
| record | movie.comments.3.text | 1 | 0.10% |
| record | movie.comments.21.text | 1 | 0.10% |
| record | movie.comments.18.text | 1 | 0.10% |
| record | movie.actors | 1 | 0.10% |
| record | movie.comments.13.text | 1 | 0.10% |
| record | movie.comments.9.time | 1 | 0.10% |
| record | movie.comments.11.date | 1 | 0.10% |
| record | movie.director | 1 | 0.10% |
| record | movie.comments.4.date | 1 | 0.10% |
| record | movie.comments.10.date | 1 | 0.10% |
| record | movie.comments.9.text | 1 | 0.10% |
| record | movie.comments.1.text | 1 | 0.10% |
| record | movie.rating.score | 1 | 0.10% |
| record | movie.comments.9.date | 1 | 0.10% |
| record | movie.comments.8.text | 1 | 0.10% |
| record | movie.comments.17.text | 1 | 0.10% |
| record | movie.comments.10.time | 1 | 0.10% |
| record | movie.comments.17.time | 1 | 0.10% |
| record | movie.comments.4.time | 1 | 0.10% |
| record | movie.comments.19.date | 1 | 0.10% |
| record | movie.comments.5.date | 1 | 0.10% |
| record | movie.comments.14.time | 1 | 0.10% |
| record | movie.comments.10.text | 1 | 0.10% |
| record | movie.comments.15.time | 1 | 0.10% |
| record | movie.production_country | 1 | 0.10% |
| record | movie.rating.votes | 1 | 0.10% |
| record | underwriting_notes | 1 | 0.10% |
| record | copyright | 1 | 0.10% |
| record | last_modified | 1 | 0.10% |
| record | webmaster | 1 | 0.10% |
| record | issue_type: Type of technical problem reported | 1 | 0.10% |
| record | submitted_date: Date the ticket was submitted | 1 | 0.10% |
| record | severity: Priority level of the issue | 1 | 0.10% |
| record | employee_name: Name of employee who reported the issue | 1 | 0.10% |
| record | ticket_id: Unique identifier for each service ticket | 1 | 0.10% |
| record | department: Department where the issue originated | 1 | 0.10% |
| record | passengers_affected | 1 | 0.10% |
| record | key_result | 1 | 0.10% |
| record | battery_life: Battery life in hours as specified | 1 | 0.10% |
| record | location: City and state where product is located | 1 | 0.10% |
| record | material: Composition of the product's physical materials | 1 | 0.10% |
| record | availability_date: Date when product becomes available | 1 | 0.10% |
| record | name: Product display name as listed in the entry | 1 | 0.10% |
| record | isps | 1 | 0.10% |
| record | grant_number | 1 | 0.10% |
| record | chipdesignfocus | 1 | 0.10% |
| record | project_lead | 1 | 0.10% |
| record | judge | 1 | 0.10% |
| record | plaintiff | 1 | 0.10% |
| record | consoles | 1 | 0.10% |
| record | delayed_games | 1 | 0.10% |
| record | upcoming_games | 1 | 0.10% |
| record | attendance | 1 | 0.10% |
| record | organizer_email | 1 | 0.10% |
| record | game_id: Unique identifier for the game as listed in the schedule | 1 | 0.10% |
| record | home_team: Name of the team playing at home | 1 | 0.10% |
| record | home_score: Final score for the home team | 1 | 0.10% |
| record | venue: Name of the arena where the game was played | 1 | 0.10% |
| record | currency_pair: Two-currency pair such as EUR/USD or SGD/USD | 1 | 0.10% |
| record | effective_time: Local time when the rate was effective | 1 | 0.10% |
| record | location: Office address of the institution | 1 | 0.10% |
| record | alternative_title | 1 | 0.10% |
| record | price.currency | 1 | 0.10% |
| record | related_links | 1 | 0.10% |
| record | focus_area | 1 | 0.10% |
| record | mapping_period | 1 | 0.10% |
| record | node_count | 1 | 0.10% |
| record | event.start_time | 1 | 0.10% |
| record | team.vehicle.model | 1 | 0.10% |
| record | team.vehicle.fuel_type | 1 | 0.10% |
| record | event.name | 1 | 0.10% |
| record | drivers.1.nationality | 1 | 0.10% |
| record | team.vehicle.make | 1 | 0.10% |
| record | event.finish_time | 1 | 0.10% |
| record | event.stage_count | 1 | 0.10% |
| record | drivers.0.name | 1 | 0.10% |
| record | status: Current status of the enrollment | 1 | 0.10% |
| record | duration: Number of days the program lasts | 1 | 0.10% |
| record | employee_id: Unique identifier for the employee enrollment | 1 | 0.10% |
| record | brand: Company name responsible for the product launch | 1 | 0.10% |
| record | location: Geographic market or city where product is available | 1 | 0.10% |
| record | price: Price of the product in US dollars | 1 | 0.10% |
| record | vehicle model | 1 | 0.10% |
| record | territory | 1 | 0.10% |
| record | person name | 1 | 0.10% |
| record | address: Patient's residential address including street, unit, and city | 1 | 0.10% |
| record | status: Current state of prescription fulfillment | 1 | 0.10% |
| record | pharmacy_name: Name of the pharmacy that processed the prescription | 1 | 0.10% |
| record | quantity: Number of tablets or units dispensed | 1 | 0.10% |
| record | dosage: Specific amount or form of medication prescribed | 1 | 0.10% |
| record | person.publication_information.commentary_with | 1 | 0.10% |
| record | person.time_period.era | 1 | 0.10% |
| record | person.publication_information.available_part | 1 | 0.10% |
| record | person.publication_information.publisher | 1 | 0.10% |
| record | person.publication_information.location | 1 | 0.10% |
| record | trend_repost_id | 1 | 0.10% |
| record | asteroid.number | 1 | 0.10% |
| record | asteroid.name_latin | 1 | 0.10% |
| record | asteroid.etymology.origin | 1 | 0.10% |
| record | asteroid.name_original | 1 | 0.10% |
| record | asteroid.characteristics.type | 1 | 0.10% |
| record | asteroid.etymology.deity.description | 1 | 0.10% |
| record | asteroid.discovery.observatory | 1 | 0.10% |
| record | location: Office location where the visit occurred | 1 | 0.10% |
| record | medication: Specific medication prescribed with dosage | 1 | 0.10% |
| record | visit_date: Date of the patient visit | 1 | 0.10% |
| record | lab_order: List of lab tests ordered for the patient | 1 | 0.10% |
| record | technique_name | 1 | 0.10% |
| record | cohort_size | 1 | 0.10% |
| record | Phenomenon | 1 | 0.10% |
| record | Food | 1 | 0.10% |
| record | Consumption | 1 | 0.10% |
| record | Attribute | 1 | 0.10% |
| record | city.name | 1 | 0.10% |
| record | city.state | 1 | 0.10% |
| record | city.location | 1 | 0.10% |
| record | city.birthplace_of.0.person.name | 1 | 0.10% |
| record | city.birthplace_of.0.person.occupation | 1 | 0.10% |
| record | company.founding_location.city | 1 | 0.10% |
| record | company.location | 1 | 0.10% |
| record | company.transformation_date | 1 | 0.10% |
| record | company.name | 1 | 0.10% |
| record | company.description | 1 | 0.10% |
| record | company.establishment_date | 1 | 0.10% |
| record | sku: Stock keeping unit assigned to each product | 1 | 0.10% |
| record | name: Full product name as listed in the store catalog | 1 | 0.10% |
| record | channel_name | 1 | 0.10% |
| record | description_length | 1 | 0.10% |
| record | thumbnail_style | 1 | 0.10% |
| record | channel_location | 1 | 0.10% |
| record | content_type | 1 | 0.10% |
| record | mentions | 1 | 0.10% |
| record | product_example.size | 1 | 0.10% |
| record | product_example.color | 1 | 0.10% |
| record | availability.locations.online_stores.0.location | 1 | 0.10% |
| record | product_usage | 1 | 0.10% |
| record | dose_number | 1 | 0.10% |
| record | doctor_name | 1 | 0.10% |
| record | prescription | 1 | 0.10% |
| record | format | 1 | 0.10% |
| record | winner_team | 1 | 0.10% |
| record | result | 1 | 0.10% |
| record | renewal_amount: Annual premium amount for the renewal | 1 | 0.10% |
| record | sponsorship | 1 | 0.10% |
| record | affected_people | 1 | 0.10% |
| record | bp | 1 | 0.10% |
| record | doctor | 1 | 0.10% |
| record | delegation.country | 1 | 0.10% |
| record | paralympic_games.season | 1 | 0.10% |
| record | delegation.medals.gold | 1 | 0.10% |
| record | delegation.medals.bronze | 1 | 0.10% |
| record | paralympic_games.country | 1 | 0.10% |
| record | location: Physical location of the testing facility | 1 | 0.10% |
| record | result_value: Numerical value of the test result | 1 | 0.10% |
| record | test_code: Code identifying the specific laboratory test | 1 | 0.10% |
| record | units: Unit of measurement for the test result | 1 | 0.10% |
| record | owner | 1 | 0.10% |
| record | dispute_id | 1 | 0.10% |
| record | material | 1 | 0.10% |
| record | height | 1 | 0.10% |
| record | capacity | 1 | 0.10% |
| record | store | 1 | 0.10% |
| record | use_case | 1 | 0.10% |
| record | compatibility | 1 | 0.10% |
| record | release_year | 1 | 0.10% |
| record | charging_time | 1 | 0.10% |
| record | product_name: Name of the launched skincare or apparel product | 1 | 0.10% |
| record | product_id: Unique identifier assigned to the product | 1 | 0.10% |
| record | launch_location: City and state where the product was launched | 1 | 0.10% |
| record | price: Selling price of the product in US dollars | 1 | 0.10% |
| record | company: Name of the startup receiving funding | 1 | 0.10% |
| record | amount: Funding amount in millions | 1 | 0.10% |
| record | lead_investor: Primary investor leading the round | 1 | 0.10% |
| record | co_investors: List of additional investors participating | 1 | 0.10% |
| record | status: Current status of the funding round | 1 | 0.10% |
| record | tracking_code | 1 | 0.10% |
| record | organizer | 1 | 0.10% |
| record | nummod | 1 | 0.10% |
| record | cc | 1 | 0.10% |
| record | intj | 1 | 0.10% |
| record | article.title | 1 | 0.10% |
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
| record | return_address | 1 | 0.10% |
| record | name: Full name of the policyholder | 1 | 0.10% |
| record | address: Complete street address including city and state | 1 | 0.10% |
| record | renewal_date: Date when the policy is set to renew | 1 | 0.10% |
| record | renewal_amount: Monetary amount required for policy renewal | 1 | 0.10% |
| record | status: Current status of policy renewal | 1 | 0.10% |
| record | notice_type: Method used to notify policyholder of renewal | 1 | 0.10% |
| record | meddra_code: Standard code for the reaction in MedDRA system | 1 | 0.10% |
| record | drug: Name and dosage form of the medication taken | 1 | 0.10% |
| record | severity: Level of reaction intensity | 1 | 0.10% |
| record | date: Date when the reaction was reported | 1 | 0.10% |
| record | duration: Number of days the treatment lasted for the reaction | 1 | 0.10% |
| record | dosage: Prescribed dosage and frequency of the drug | 1 | 0.10% |
| record | recharge_amount | 1 | 0.10% |
| record | ticket_id: Unique identifier for the outage incident | 1 | 0.10% |
| record | address: Full address of the customer location | 1 | 0.10% |
| record | affected_service: List of services disrupted during the outage | 1 | 0.10% |
| record | summary: Brief description of the root cause or issue | 1 | 0.10% |
| record | contact_number: Primary phone number for customer communication | 1 | 0.10% |
| record | root_cause: Identified reason for the service disruption | 1 | 0.10% |
| record | performance_reason | 1 | 0.10% |
| record | change | 1 | 0.10% |
| record | status: Current campaign status | 1 | 0.10% |
| record | brand: Name of the brand launching the product | 1 | 0.10% |
| record | target_audience: Demographic or interest group targeted | 1 | 0.10% |
| record | campaign_code: Internal code for the campaign | 1 | 0.10% |
| record | date: Date the product was launched | 1 | 0.10% |
| record | review_platform | 1 | 0.10% |
| record | venue_id | 1 | 0.10% |
| record | south_yemen_conflict.accusations_against_governments.1.accusations | 1 | 0.10% |
| record | federalism_definition.confederation_examples | 1 | 0.10% |
| record | federalism_definition.confederation | 1 | 0.10% |
| record | south_yemen_conflict.call_for_independence | 1 | 0.10% |
| record | federalism_definition.federalism_examples | 1 | 0.10% |
| record | south_yemen_conflict.historical_context.actions_after_invasion | 1 | 0.10% |
| record | south_yemen_conflict.historical_context.discrimination_and_exclusion | 1 | 0.10% |
| record | south_yemen_conflict.accusations_against_governments.1.government_name | 1 | 0.10% |
| record | south_yemen_conflict.historical_context.casualties_and_prisoners | 1 | 0.10% |
| record | owner: Name of the healthcare professional owning the device | 1 | 0.10% |
| record | location: Department or unit where device is currently located | 1 | 0.10% |
| record | serial: Unique serial number identifying the device | 1 | 0.10% |
| record | vendor: Company that sold or provided the device | 1 | 0.10% |
| record | boulder.location.relative_position | 1 | 0.10% |
| record | boulder.location.distance | 1 | 0.10% |
| record | boulder.location.distance_unit | 1 | 0.10% |
| record | boulder.location.direction | 1 | 0.10% |
| record | boulder.location.reference_point | 1 | 0.10% |
| record | boulder.start.hand_holds.0.hand | 1 | 0.10% |
| record | boulder.start.hand_holds.0.description | 1 | 0.10% |
| record | boulder.start.hand_holds.1.hand | 1 | 0.10% |
| record | boulder.start.hand_holds.1.description | 1 | 0.10% |
| record | boulder.start.foot_holds.0.foot | 1 | 0.10% |
| record | boulder.start.foot_holds.0.description | 1 | 0.10% |
| record | boulder.start.foot_holds.1.foot | 1 | 0.10% |
| record | boulder.start.foot_holds.1.description | 1 | 0.10% |
| record | boulder.moves | 1 | 0.10% |
| record | submission.submitted_by | 1 | 0.10% |
| record | submission.submission_date | 1 | 0.10% |
| record | route_id | 1 | 0.10% |
| record | team_member | 1 | 0.10% |
| record | completion_date | 1 | 0.10% |
| record | budget_used | 1 | 0.10% |
| record | birth_date | 1 | 0.10% |
| record | other_collaborations | 1 | 0.10% |
| record | roles | 1 | 0.10% |
| record | match_round: Stage of the tournament such as final, quarterfinal, or last 16 | 1 | 0.10% |
| record | loser: Player who lost the match | 1 | 0.10% |
| record | historical_periods | 1 | 0.10% |
| record | phone | 1 | 0.10% |
| record | recipient_name: Name of the recipient organization | 1 | 0.10% |
| record | onset_timing | 1 | 0.10% |
| record | genus.established_year | 1 | 0.10% |
| record | genus.established_by | 1 | 0.10% |
| record | genus.number_of_species.min | 1 | 0.10% |
| record | genus.number_of_species.max | 1 | 0.10% |
| record | genus.etymology.part1.meaning | 1 | 0.10% |
| record | genus.etymology.part1.related_entity | 1 | 0.10% |
| record | genus.etymology.part2.name | 1 | 0.10% |
| record | genus.european_species | 1 | 0.10% |
| record | genus.german_species | 1 | 0.10% |
| record | service.interventions | 1 | 0.10% |
| record | service.opening_hours.0.day | 1 | 0.10% |
| record | service.contact_information.email | 1 | 0.10% |
| record | service.location.building | 1 | 0.10% |
| record | service.responsible_person.role | 1 | 0.10% |
| record | service.problems_addressed | 1 | 0.10% |
| record | service.location.city | 1 | 0.10% |
| record | service.location.address | 1 | 0.10% |
| record | service.responsible_person.experience | 1 | 0.10% |
| record | required_materials | 1 | 0.10% |
| record | techtoolsused | 1 | 0.10% |
| record | siblings.0.name | 1 | 0.10% |
| record | siblings.0.location | 1 | 0.10% |
| record | siblings.0.relationship | 1 | 0.10% |
| record | great_grandnieces_and_nephews | 1 | 0.10% |
| record | employment.0.company | 1 | 0.10% |
| record | employment.1.company | 1 | 0.10% |
| record | employment.2.company | 1 | 0.10% |
| record | employment.3.company | 1 | 0.10% |
| record | employment.4.company | 1 | 0.10% |
| record | nieces | 1 | 0.10% |
| record | grandchildren | 1 | 0.10% |
| record | predeceased.0.name | 1 | 0.10% |
| record | predeceased.0.relationship | 1 | 0.10% |
| record | predeceased.1.relationship | 1 | 0.10% |
| record | predeceased.2.relationship | 1 | 0.10% |
| record | predeceased.3.name | 1 | 0.10% |
| record | predeceased.3.relationship | 1 | 0.10% |
| record | predeceased.4.name | 1 | 0.10% |
| record | predeceased.4.relationship | 1 | 0.10% |
| record | predeceased.5.name | 1 | 0.10% |
| record | predeceased.5.relationship | 1 | 0.10% |
| record | predeceased.6.name | 1 | 0.10% |
| record | predeceased.6.relationship | 1 | 0.10% |
| record | predeceased.7.name | 1 | 0.10% |
| record | predeceased.7.relationship | 1 | 0.10% |
| record | siblings_in_law.0.name | 1 | 0.10% |
| record | siblings_in_law.0.location | 1 | 0.10% |
| record | siblings_in_law.0.relationship | 1 | 0.10% |
| record | siblings_in_law.1.name | 1 | 0.10% |
| record | siblings_in_law.1.location | 1 | 0.10% |
| record | siblings_in_law.1.relationship | 1 | 0.10% |
| record | siblings_in_law.2.name | 1 | 0.10% |
| record | siblings_in_law.2.location | 1 | 0.10% |
| record | siblings_in_law.2.relationship | 1 | 0.10% |
| record | siblings_in_law.3.name | 1 | 0.10% |
| record | siblings_in_law.3.location | 1 | 0.10% |
| record | siblings_in_law.3.relationship | 1 | 0.10% |
| record | death_location | 1 | 0.10% |
| record | marriage_duration | 1 | 0.10% |
| record | grandnieces_and_nephews | 1 | 0.10% |
| record | children.0.name | 1 | 0.10% |
| record | children.0.spouse | 1 | 0.10% |
| record | children.1.name | 1 | 0.10% |
| record | children.1.spouse | 1 | 0.10% |
| record | residence | 1 | 0.10% |
| record | spouse | 1 | 0.10% |
| record | parents.0.name | 1 | 0.10% |
| record | nephews | 1 | 0.10% |
| record | James Holmes <> fired shots | 1 | 0.10% |
| record | Assurant Health <> provider of | 1 | 0.10% |
| record | Project SAM <> opposes | 1 | 0.10% |
| record | Denver Mayor Michael Hancock <> commented on | 1 | 0.10% |
| record | freshman girl <> attacked | 1 | 0.10% |
| record | United Airlines <> dominated | 1 | 0.10% |
| record | Gary Coleman <> former resident of | 1 | 0.10% |
| record | Andrew Bulkeley <> attended | 1 | 0.10% |
| record | Frontier <> local to | 1 | 0.10% |
| record | Ivan Suvanjieff <> founder of | 1 | 0.10% |
| record | consumers <> surprised by | 1 | 0.10% |
| record | Obama administration drug czar <> discussed | 1 | 0.10% |
| record | Chinese farmers <> used | 1 | 0.10% |
| record | Amazon <> introduced | 1 | 0.10% |
| record | record labels <> surprised by | 1 | 0.10% |
| record | numbered_designation | 1 | 0.10% |
| record | discovery.location.country | 1 | 0.10% |
| record | physical_characteristics.observations.0.type | 1 | 0.10% |
| record | physical_characteristics.publication_year | 1 | 0.10% |
| record | discovery.survey | 1 | 0.10% |
| record | snapshot_id | 1 | 0.10% |
| record | rate | 1 | 0.10% |
| record | source_currency | 1 | 0.10% |
| record | target_currency | 1 | 0.10% |
| record | prior_experience | 1 | 0.10% |
| record | deductible: Increased deductible amount if applicable | 1 | 0.10% |
| record | product_code | 1 | 0.10% |
| record | concurrentuserslimit | 1 | 0.10% |
| record | devicebatterylife | 1 | 0.10% |
| record | monthly_price | 1 | 0.10% |
| record | integration_capabilities | 1 | 0.10% |
| record | supportresponsetime | 1 | 0.10% |
| record | video_record | 1 | 0.10% |
| record | co_investors: List of additional investors participating in the round | 1 | 0.10% |
| record | company: Name of the technology startup | 1 | 0.10% |
| record | funding_amount: Total amount raised in millions of dollars | 1 | 0.10% |
| record | consideration_structure | 1 | 0.10% |
| record | enterprise_value | 1 | 0.10% |
| record | deal_type | 1 | 0.10% |
| record | status: Current legal status of the patent | 1 | 0.10% |
| record | bp_diastolic | 1 | 0.10% |
| record | bp_systolic | 1 | 0.10% |
| record | temp_f | 1 | 0.10% |
| record | employee_id: Unique identifier for each new hire entry | 1 | 0.10% |
| record | assigned_coordinator: Name of the onboarding coordinator assigned to the employee | 1 | 0.10% |
| record | department: Department assigned to the employee | 1 | 0.10% |
| record | training_completed: Percentage of training completed out of 100 | 1 | 0.10% |
| record | office: Regional office location where employee is based | 1 | 0.10% |
| record | market_share: Percentage of market share held by the company | 1 | 0.10% |
| record | patient | 1 | 0.10% |
| record | provider_address | 1 | 0.10% |
| record | wallet_address: Wallet address used for the transaction | 1 | 0.10% |
| record | timestamp: Date when the transaction occurred | 1 | 0.10% |
| record | description: Brief note on reason or context of the transaction | 1 | 0.10% |
| record | amount: Quantity of asset traded or held in the transaction | 1 | 0.10% |
| record | album.external_links.2.description | 1 | 0.10% |
| record | album.tracklist.cd.1.track_number | 1 | 0.10% |
| record | album.tracklist.cd.2.composer | 1 | 0.10% |
| record | album.tracklist.cd.9.title | 1 | 0.10% |
| record | album.sources.2.reference_number | 1 | 0.10% |
| record | album.artist | 1 | 0.10% |
| record | album.tracklist.dvd.0.type | 1 | 0.10% |
| record | album.sources.0.access_date | 1 | 0.10% |
| record | album.versions.1.type | 1 | 0.10% |
| record | album.sources.1.source_name | 1 | 0.10% |
| record | album.sources.3.publication_date | 1 | 0.10% |
| record | album.sources.0.url | 1 | 0.10% |
| record | album.tracklist.cd.0.composer | 1 | 0.10% |
| record | album.tracklist.cd.1.lyrics | 1 | 0.10% |
| record | album.tracklist.cd.0.track_number | 1 | 0.10% |
| record | album.tracklist.cd.10.track_number | 1 | 0.10% |
| record | album.tracklist.cd.8.track_number | 1 | 0.10% |
| record | album.tracklist.cd.0.lyrics | 1 | 0.10% |
| record | album.tracklist.dvd.1.type | 1 | 0.10% |
| record | album.tracklist.cd.2.notes | 1 | 0.10% |
| record | album.sources.2.source_name | 1 | 0.10% |
| record | album.tracklist.dvd.1.title | 1 | 0.10% |
| record | album.tracklist.cd.5.notes | 1 | 0.10% |
| record | album.versions.2.contents | 1 | 0.10% |
| record | album.external_links.1.url | 1 | 0.10% |
| record | album.tracklist.cd.4.notes | 1 | 0.10% |
| record | album.tracklist.cd.9.notes | 1 | 0.10% |
| record | album.tracklist.cd.8.notes | 1 | 0.10% |
| record | album.tracklist.cd.1.notes | 1 | 0.10% |
| record | album.tracklist.cd.2.tie_in.0.type | 1 | 0.10% |
| record | album.tracklist.cd.3.title | 1 | 0.10% |
| record | album.tracklist.dvd.2.type | 1 | 0.10% |
| record | album.external_links.0.url | 1 | 0.10% |
| record | album.sources.1.access_date | 1 | 0.10% |
| record | album.sources.3.access_date | 1 | 0.10% |
| record | album.chart_performance.chart_type | 1 | 0.10% |
| record | album.tracklist.cd.0.title | 1 | 0.10% |
| record | album.tracklist.cd.0.notes | 1 | 0.10% |
| record | album.tracklist.cd.3.notes | 1 | 0.10% |
| record | album.tracklist.cd.9.track_number | 1 | 0.10% |
| record | album.tracklist.cd.3.tie_in.0.title | 1 | 0.10% |
| record | album.chart_performance.ranking | 1 | 0.10% |
| record | album.tracklist.cd.9.composer | 1 | 0.10% |
| record | album.versions.1.contents | 1 | 0.10% |
| record | album.tracklist.cd.9.tie_in.0.type | 1 | 0.10% |
| record | album.tracklist.cd.8.tie_in.0.title | 1 | 0.10% |
| record | album.tracklist.dvd.1.track_number | 1 | 0.10% |
| record | album.tracklist.cd.6.title | 1 | 0.10% |
| record | album.tracklist.dvd.0.title | 1 | 0.10% |
| record | album.tracklist.cd.7.track_number | 1 | 0.10% |
| record | album.external_links.1.description | 1 | 0.10% |
| record | album.tracklist.cd.7.composer | 1 | 0.10% |
| record | album.versions.2.type | 1 | 0.10% |
| record | album.tracklist.cd.3.composer | 1 | 0.10% |
| record | album.external_links.0.description | 1 | 0.10% |
| record | album.tracklist.cd.1.title | 1 | 0.10% |
| record | album.tracklist.dvd.2.track_number | 1 | 0.10% |
| record | album.anniversary | 1 | 0.10% |
| record | album.sources.1.reference_number | 1 | 0.10% |
| record | album.tracklist.cd.5.composer | 1 | 0.10% |
| record | album.tracklist.cd.3.lyrics | 1 | 0.10% |
| record | album.versions.0.name | 1 | 0.10% |
| record | album.sources.0.source_name | 1 | 0.10% |
| record | album.tracklist.cd.10.composer | 1 | 0.10% |
| record | album.versions.2.name | 1 | 0.10% |
| record | album.tracklist.cd.8.title | 1 | 0.10% |
| record | album.external_links.2.url | 1 | 0.10% |
| record | album.tracklist.cd.5.title | 1 | 0.10% |
| record | album.tracklist.cd.5.track_number | 1 | 0.10% |
| record | album.chart_performance.record_holder_description | 1 | 0.10% |
| record | album.sources.1.publication_date | 1 | 0.10% |
| record | album.tracklist.cd.4.lyrics | 1 | 0.10% |
| record | album.sources.3.url | 1 | 0.10% |
| record | album.chart_performance.record_holder_rank | 1 | 0.10% |
| record | experience | 1 | 0.10% |
| record | company_size | 1 | 0.10% |
| record | job_id | 1 | 0.10% |
| record | author: Name of the abstract's author | 1 | 0.10% |
| record | department: Academic department of the author | 1 | 0.10% |
| record | title: Title of the abstract work | 1 | 0.10% |
| record | other_youth_programs_with_congressional_charters | 1 | 0.10% |
| record | technology_name: Name of the algorithm, platform, or device | 1 | 0.10% |
| record | agent_name | 1 | 0.10% |
| record | film_title | 1 | 0.10% |
| record | publication | 1 | 0.10% |
| record | useoffunds: Purpose or application of raised capital | 1 | 0.10% |
| record | investors: List of investor names or groups involved | 1 | 0.10% |
| record | funding_amount: Total amount raised in millions | 1 | 0.10% |
| record | funding_id: Unique identifier assigned to the funding round | 1 | 0.10% |
| record | court | 1 | 0.10% |
| record | vaccine_type: Type of vaccine administered | 1 | 0.10% |
| record | status: Current status of vaccination record | 1 | 0.10% |
| record | goal_achievement: Percentage of goals met, from 60% to 100% | 1 | 0.10% |
| record | name: Full name of the employee as listed in the review | 1 | 0.10% |
| record | department: Department where the employee works | 1 | 0.10% |
| record | report_id: Unique identifier for the phishing incident | 1 | 0.10% |
| record | recipient: Email address of the victim or intended recipient | 1 | 0.10% |
| record | risk_level: Assessed risk level of the phishing attempt | 1 | 0.10% |
| record | silver_medalist: Name of athlete who won silver medal | 1 | 0.10% |
| record | bronze_medalist: Name of athlete who won bronze medal | 1 | 0.10% |
| record | event_code: Unique identifier for the Olympic competition | 1 | 0.10% |
| record | uploader | 1 | 0.10% |
| record | clientbasesize: Number of clients served across regions | 1 | 0.10% |
| record | location: City and state or country where the company is based | 1 | 0.10% |
| record | funding_source: Type of funding received by the company | 1 | 0.10% |
| record | business_focus: Core service or product area of the company | 1 | 0.10% |
| record | billing_amount | 1 | 0.10% |
| record | subscription_type | 1 | 0.10% |
| record | email_address | 1 | 0.10% |
| record | start_date | 1 | 0.10% |
| record | tenant_id | 1 | 0.10% |
| record | source_agency | 1 | 0.10% |
| record | diseases_treated.scope | 1 | 0.10% |
| record | patient_care.station_assignment | 1 | 0.10% |
| record | policy_reference | 1 | 0.10% |
| record | drexel_info | 1 | 0.10% |
| record | browse_topics | 1 | 0.10% |
| record | footer_links | 1 | 0.10% |
| record | faq_references | 1 | 0.10% |
| record | origin: Airport code where flight departs | 1 | 0.10% |
| record | delay_minutes: Duration of delay in minutes | 1 | 0.10% |
| record | contact_email: Email address of primary passenger for communication | 1 | 0.10% |
| record | techniques | 1 | 0.10% |
| record | diseases | 1 | 0.10% |
| record | compounds | 1 | 0.10% |
| record | enzymes | 1 | 0.10% |
| record | measurements | 1 | 0.10% |
| record | version | 1 | 0.10% |
| record | officer | 1 | 0.10% |
| record | incidence.rarity | 1 | 0.10% |
| record | description: Technical summary of the patented invention | 1 | 0.10% |
| record | address: Full address of the inventor | 1 | 0.10% |
| record | product_line | 1 | 0.10% |
| record | brief_id | 1 | 0.10% |
| record | symbol: Company ticker symbol as listed in the report | 1 | 0.10% |
| record | price: Current stock price in dollars | 1 | 0.10% |
| record | enrollment | 1 | 0.10% |
| record | city: City where the event is held | 1 | 0.10% |
| record | event_status: Current status of the event | 1 | 0.10% |
| record | event_name: Title of the event as listed | 1 | 0.10% |
| record | location: Full address including street, city, and state | 1 | 0.10% |
| record | event_code: Unique code identifying the event | 1 | 0.10% |
| record | event_date: Date when the event will take place | 1 | 0.10% |
| record | venue: Stadium name where the match was held | 1 | 0.10% |
| record | scorers: List of players who scored goals in the match | 1 | 0.10% |
| record | relationship_to_subject | 1 | 0.10% |
| record | thread_id: Unique identifier for the support thread | 1 | 0.10% |
| record | customer_name: Name of the customer involved | 1 | 0.10% |
| record | customer_address: Full address including street, apt, city, state, zip | 1 | 0.10% |
| record | issue_description: Description of the customer's reported problem | 1 | 0.10% |
| record | support_response: Agent's response to the customer's issue | 1 | 0.10% |
| record | refund_amount: Amount refunded or credited to customer | 1 | 0.10% |
| record | priority: Priority level assigned to the case | 1 | 0.10% |
| record | filer | 1 | 0.10% |
| record | dates_mentioned | 1 | 0.10% |
| record | characteristics.effective_temperature.unit | 1 | 0.10% |
| record | characteristics.subtype | 1 | 0.10% |
| record | ticket_id: Unique identifier for the outage ticket | 1 | 0.10% |
| record | address: Full address of the customer | 1 | 0.10% |
| record | city: City where the customer resides | 1 | 0.10% |
| record | state: State where the customer resides | 1 | 0.10% |
| record | outage_start: Start time of the service disruption | 1 | 0.10% |
| record | outage_end: End time of the service disruption | 1 | 0.10% |
| record | service_impact: Type of service affected, such as internet or mobile | 1 | 0.10% |
| record | root_cause: Identified cause of the outage or under investigation | 1 | 0.10% |
| record | provider_code: Service provider code assigned to the ticket | 1 | 0.10% |
| record | technologies | 1 | 0.10% |
| record | client_email | 1 | 0.10% |
| record | medals.silver | 1 | 0.10% |
| record | date_administered | 1 | 0.10% |
| record | medicalrecordid | 1 | 0.10% |
| record | radiology_code | 1 | 0.10% |
| record | finding_summary | 1 | 0.10% |
| record | asteroid.discovery.discoverer.name | 1 | 0.10% |
| record | asteroid.named_after.description | 1 | 0.10% |
| record | asteroid.named_after.birth_location.country | 1 | 0.10% |
| record | asteroid.alternate_name | 1 | 0.10% |
| record | asteroid.discovery.discoverer.affiliation.observatory | 1 | 0.10% |
| record | ticket_id: Unique identifier for each outage report | 1 | 0.10% |
| record | customer: Name of the affected customer | 1 | 0.10% |
| record | address: Full residential address of the customer | 1 | 0.10% |
| record | revenue: Total revenue in millions | 1 | 0.10% |
| record | year: Year of financial data | 1 | 0.10% |
| record | partner_year: Year of partner's financial data | 1 | 0.10% |
| record | partner_revenue: Partner's revenue in millions | 1 | 0.10% |
| record | company_id: Unique identifier for the company | 1 | 0.10% |
| record | partner_company: Name of the partner company | 1 | 0.10% |
| record | partner_location: City and state of the partner company | 1 | 0.10% |
| record | partner_status: Operational status of the partner company | 1 | 0.10% |
| record | status: Current operational status | 1 | 0.10% |
| record | location: City and state of the company | 1 | 0.10% |
| record | partnergrowthrate: Partner's growth rate percentage | 1 | 0.10% |
| record | policy_id: Policy number associated with the claim | 1 | 0.10% |
| record | hotel_experience.description | 1 | 0.10% |
| record | hotel_experience.location.nearby_landmark | 1 | 0.10% |
| record | hotel_experience.location.city | 1 | 0.10% |
| record | hotel_experience.location.postal_code_accessibility | 1 | 0.10% |
| record | hotel_experience.suites.0.bathroom_features | 1 | 0.10% |
| record | hotel_experience.suites.0.workspace | 1 | 0.10% |
| record | hotel_experience.suites.0.suite_type | 1 | 0.10% |
| record | hotel_experience.suites.0.suite_count | 1 | 0.10% |
| record | hotel_experience.suites.0.features | 1 | 0.10% |
| record | hotel_experience.suites.0.amenities | 1 | 0.10% |
| record | hotel_experience.services | 1 | 0.10% |
| record | hotel_experience.entertainment.tv.screen_size | 1 | 0.10% |
| record | hotel_experience.entertainment.tv.content | 1 | 0.10% |
| record | hotel_experience.additional_benefits | 1 | 0.10% |
| record | volume: Total shares traded on the day | 1 | 0.10% |
| record | market_cap: Market capitalization in billions | 1 | 0.10% |
| record | close_price: Final closing price of the stock | 1 | 0.10% |
| record | movement_reason: Brief reason for price movement or performance | 1 | 0.10% |
| record | winner: Candidate who won the district | 1 | 0.10% |
| record | party: Political party of the winner | 1 | 0.10% |
| record | district_id: Unique identifier for the electoral district | 1 | 0.10% |
| record | grade_distribution: Breakdown of grades achieved by students | 1 | 0.10% |
| record | location: Geographic region or city where the institution is located | 1 | 0.10% |
| record | publication_code: Unique identifier assigned to the result publication | 1 | 0.10% |
| record | institution_name: Name of the school or institute releasing results | 1 | 0.10% |
| record | currentheadcountapex | 1 | 0.10% |
| record | marketing_budget | 1 | 0.10% |
| record | iporevisedvaluation | 1 | 0.10% |
| record | flight_number | 1 | 0.10% |
| record | departure_time | 1 | 0.10% |
| record | baggage_count | 1 | 0.10% |
| record | arrival_code | 1 | 0.10% |
| record | arrival_time | 1 | 0.10% |
| record | passenger_type | 1 | 0.10% |
| record | damage_type | 1 | 0.10% |
| record | host_venue | 1 | 0.10% |
| record | repair_shop | 1 | 0.10% |
| record | beliefs | 1 | 0.10% |
| record | religion | 1 | 0.10% |
| record | scriptures | 1 | 0.10% |
| record | influences | 1 | 0.10% |
| record | offer_id | 1 | 0.10% |
| record | end_date | 1 | 0.10% |
| record | registered_attendees | 1 | 0.10% |
| record | error.message | 1 | 0.10% |
| record | error.possible_causes | 1 | 0.10% |
| record | address: Full physical address of the owner | 1 | 0.10% |
| record | contract_id | 1 | 0.10% |
| record | key_enzymes: List of enzymes critical to pathway activity | 1 | 0.10% |
| record | pathway_id: Unique identifier assigned to the pathway | 1 | 0.10% |
| record | primary_function: Core biological role of the pathway | 1 | 0.10% |
| record | discovery_date: Year the pathway was first identified | 1 | 0.10% |
| record | official_name | 1 | 0.10% |
| record | host_location | 1 | 0.10% |
| record | date_range | 1 | 0.10% |
| record | numberofteams | 1 | 0.10% |
| record | current_status | 1 | 0.10% |
| record | vendor | 1 | 0.10% |
| record | po_reference | 1 | 0.10% |
| record | item_description | 1 | 0.10% |
| record | tax_rate | 1 | 0.10% |
| record | recipient_name | 1 | 0.10% |
| record | holder_name | 1 | 0.10% |
| record | score: Final score of the match in set format | 1 | 0.10% |
| record | tournament_code: Unique identifier for the match event | 1 | 0.10% |
| record | impact | 1 | 0.10% |
| record | measurement.size_comparison | 1 | 0.10% |
| record | applications.industrial | 1 | 0.10% |
| record | origin.year | 1 | 0.10% |
| record | applications.medical | 1 | 0.10% |
| record | potential | 1 | 0.10% |
| record | organization.location_founded.city | 1 | 0.10% |
| record | organization.location_founded.country | 1 | 0.10% |
| record | research.materials | 1 | 0.10% |
| record | research.experiment_details.quantum_state | 1 | 0.10% |
| record | research.temperature | 1 | 0.10% |
| record | research.focus | 1 | 0.10% |
| record | unit: Unit designation such as 4B or 6C | 1 | 0.10% |
| record | status: Current availability status of the unit | 1 | 0.10% |
| record | seller | 1 | 0.10% |
| record | action | 1 | 0.10% |
| record | tx_id | 1 | 0.10% |
| record | geologic_region | 1 | 0.10% |
| record | location.airport_travel_time.value | 1 | 0.10% |
| record | hotel_details.recreation | 1 | 0.10% |
| record | hotel_details.room_amenities | 1 | 0.10% |
| record | booking_policy.multiple_rooms | 1 | 0.10% |
| record | hotel_details.year_built | 1 | 0.10% |
| record | child_policy.infants.age_range | 1 | 0.10% |
| record | child_policy.adult_age | 1 | 0.10% |
| record | location.nearby_attractions | 1 | 0.10% |
| record | hotel_details.reception_open_until | 1 | 0.10% |
| record | hotel_details.check_out | 1 | 0.10% |
| record | location.distance_from_airport.value | 1 | 0.10% |
| record | hotel_details.number_of_rooms | 1 | 0.10% |
| record | child_policy.children.age_range | 1 | 0.10% |
| record | hotel_details.check_in | 1 | 0.10% |
| record | id: Unique identifier for each rental listing | 1 | 0.10% |
| record | address: Full property address including street and city | 1 | 0.10% |
| record | tenant_name: Name of the current tenant or prospective renter | 1 | 0.10% |
| record | status: Current status of the lease | 1 | 0.10% |
| record | property_type: Type of housing unit | 1 | 0.10% |
| record | city: Metropolitan area where the property is located | 1 | 0.10% |
| record | pet_policy: Pet policy including restrictions or fees | 1 | 0.10% |
| record | description: Brief description of unit features or conditions | 1 | 0.10% |
| record | related_conditions | 1 | 0.10% |
| record | transaction_date | 1 | 0.10% |
| record | historical_context | 1 | 0.10% |
| record | lease_id | 1 | 0.10% |
| record | molecular_target: Biological molecule the drug interacts with | 1 | 0.10% |
| record | clinical_indication: Medical condition the drug is used to treat | 1 | 0.10% |
| record | mechanismofaction: Description of how the drug exerts its effect | 1 | 0.10% |
| record | pathway: Biological pathway affected by the drug | 1 | 0.10% |
| record | drug_name: Name of the pharmaceutical compound | 1 | 0.10% |
| record | risk_classification | 1 | 0.10% |
| record | event.action | 1 | 0.10% |
| record | event.target | 1 | 0.10% |
| record | event.location.city | 1 | 0.10% |
| record | event.location.region | 1 | 0.10% |
| record | social_media_post.platform | 1 | 0.10% |
| record | event.location.venue | 1 | 0.10% |
| record | previous_incident.year | 1 | 0.10% |
| record | event.subject | 1 | 0.10% |
| record | source.type | 1 | 0.10% |
| record | findings: Key clinical observations or abnormalities detected | 1 | 0.10% |
| record | patient_id: Unique patient identifier from the medical record | 1 | 0.10% |
| record | status: Current status of the report workflow | 1 | 0.10% |
| record | githubrepourl | 1 | 0.10% |
| record | linkedinposturl | 1 | 0.10% |
| record | release_year: Year of product release extracted from release date | 1 | 0.10% |
| record | model_name: Name of the product as marketed to consumers | 1 | 0.10% |
| record | material: Primary material used in construction | 1 | 0.10% |
| record | shipping_origin: City and state where product is shipped from | 1 | 0.10% |
| record | stock_status: Current availability status such as active, backordered, on-hold, out-of-stock | 1 | 0.10% |
| record | target_audience: Group of users or professionals it serves | 1 | 0.10% |
| record | name: Full name of the LinkedIn product | 1 | 0.10% |
| record | physical_characteristics.albedo | 1 | 0.10% |
| record | physical_characteristics.diameter | 1 | 0.10% |
| record | search_result_page.content_types | 1 | 0.10% |
| record | initial_text.negations | 1 | 0.10% |
| record | search_result_page.related_to | 1 | 0.10% |
| record | search_result_page.website_name | 1 | 0.10% |
| record | distributor.name | 1 | 0.10% |
| record | distributor.products_distributed | 1 | 0.10% |
| record | priority | 1 | 0.10% |
| record | title_english | 1 | 0.10% |
| record | medals.bronze | 1 | 0.10% |
| record | account_number | 1 | 0.10% |
| record | payment_type | 1 | 0.10% |
| record | title_fr | 1 | 0.10% |
| record | referral_specialty | 1 | 0.10% |
| record | mark_name | 1 | 0.10% |
| record | registration_number | 1 | 0.10% |
| record | playeroneid | 1 | 0.10% |
| record | playertwoname | 1 | 0.10% |
| record | listing_id: Unique identifier for each property listing | 1 | 0.10% |
| record | bedrooms: Number of bedrooms in the property | 1 | 0.10% |
| record | bathrooms: Number of bathrooms in the property | 1 | 0.10% |
| record | price: Listed sale price in dollars | 1 | 0.10% |
| record | lot_size: Size of the property in acres | 1 | 0.10% |
| record | build_year: Year the property was constructed | 1 | 0.10% |
| record | agent: Name of the real estate agent representing the seller | 1 | 0.10% |
| record | holder: Name of the policyholder involved in renewal | 1 | 0.10% |
| record | launch_location | 1 | 0.10% |
| record | product_price | 1 | 0.10% |
| record | social_handle | 1 | 0.10% |
| record | institution: Hospital or organization where author is affiliated | 1 | 0.10% |
| record | recommendation: Specific treatment or intervention advised | 1 | 0.10% |
| record | guideline_id: Unique identifier for the guideline document | 1 | 0.10% |
| record | organizer.name | 1 | 0.10% |
| record | intendedAudience | 1 | 0.10% |
| record | surveyPurpose | 1 | 0.10% |
| record | accessInstructions | 1 | 0.10% |
| record | claim_date: Date the claim was filed | 1 | 0.10% |
| record | condition: Medical condition or procedure described | 1 | 0.10% |
| record | opening_year | 1 | 0.10% |
| record | review_code | 1 | 0.10% |
| record | author_information.location.city | 1 | 0.10% |
| record | author_information.family.pets.0.name | 1 | 0.10% |
| record | author_information.occupation | 1 | 0.10% |
| record | author_information.family.pets.0.type | 1 | 0.10% |
| record | author_information.location.state | 1 | 0.10% |
| record | author_information.military_service.status | 1 | 0.10% |
| record | representatives.0.statements | 1 | 0.10% |
| record | representatives.0.role | 1 | 0.10% |
| record | representatives.1.statements | 1 | 0.10% |
| record | representatives.1.role | 1 | 0.10% |
| record | representatives.0.name | 1 | 0.10% |
| record | impact_percentage | 1 | 0.10% |
| record | support_code | 1 | 0.10% |
| record | first_contact_resolution_rate | 1 | 0.10% |
| record | assigned_agent | 1 | 0.10% |
| record | drug_name | 1 | 0.10% |
| record | drug_id | 1 | 0.10% |
| record | form | 1 | 0.10% |
| record | indication | 1 | 0.10% |
| record | response | 1 | 0.10% |
| record | administration_date | 1 | 0.10% |
| record | related_verses | 1 | 0.10% |
| record | patient_id: Unique identifier assigned to each patient | 1 | 0.10% |
| record | age: Age in years as stated in the record | 1 | 0.10% |
| record | birth_date: Date of birth in MM/DD/YYYY format | 1 | 0.10% |
| record | address: Street address including number and street name | 1 | 0.10% |
| record | state: State abbreviation of patient's residence | 1 | 0.10% |
| record | zip_code: Postal code of patient's address | 1 | 0.10% |
| record | diagnosis: Medical condition such as hypertension or diabetes | 1 | 0.10% |
| record | cholesterol: Cholesterol level in mg/dL if elevated | 1 | 0.10% |
| record | hba1c: Hemoglobin A1c percentage value | 1 | 0.10% |
| record | medication: Name and dosage of prescribed medication | 1 | 0.10% |
| record | visit_date: Date of last medical check-up | 1 | 0.10% |
| record | treatment: Type of treatment or management plan | 1 | 0.10% |
| record | village_name | 1 | 0.10% |
| record | policyholder: Name of the insurance claimant | 1 | 0.10% |
| record | policyholder_id: Unique identifier for the insurance policyholder | 1 | 0.10% |
| record | claim_type: Type of medical procedure covered | 1 | 0.10% |
| record | batteryorfeature | 1 | 0.10% |
| record | lastcommitdate | 1 | 0.10% |
| record | from_club | 1 | 0.10% |
| record | to_club | 1 | 0.10% |
| record | transfer_code | 1 | 0.10% |
| record | player_id | 1 | 0.10% |
| record | ownership_structure | 1 | 0.10% |
| record | sector_focus | 1 | 0.10% |
| record | vendor_name | 1 | 0.10% |
| record | contact_address | 1 | 0.10% |
| record | service_description | 1 | 0.10% |
| record | auxpass | 1 | 0.10% |
| record | dobj | 1 | 0.10% |
| record | prep | 1 | 0.10% |
| record | acomp | 1 | 0.10% |
| record | appos | 1 | 0.10% |
| record | term | 1 | 0.10% |
| record | adjuster: Name of the insurance adjuster assigned | 1 | 0.10% |
| record | vehicle: Make and model of the vehicle involved | 1 | 0.10% |
| record | patient_gender: Gender of the patient | 1 | 0.10% |
| record | drug_name: Name of the medication involved in the reaction | 1 | 0.10% |
| record | patient_address: Full residential address of the patient | 1 | 0.10% |
| record | report_id: Unique identifier assigned to the adverse event report | 1 | 0.10% |
| record | drug_dosage: Dosage and frequency of the drug exposure | 1 | 0.10% |
| record | foods_mentioned | 1 | 0.10% |
| record | abstract_id | 1 | 0.10% |
| record | study.title | 1 | 0.10% |
| record | study.author.name | 1 | 0.10% |
| record | study.author.affiliation | 1 | 0.10% |
| record | study.period.start_year | 1 | 0.10% |
| record | study.period.end_year | 1 | 0.10% |
| record | study.focus | 1 | 0.10% |
| record | study.key_findings | 1 | 0.10% |
| record | study.comparison_countries | 1 | 0.10% |
| record | study.quotes | 1 | 0.10% |
| record | app | 1 | 0.10% |
| record | merchantcategorycode | 1 | 0.10% |
| record | location: City and state or country where posted | 1 | 0.10% |
| record | status: Current status of the post based on platform action | 1 | 0.10% |
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
| record | project_type | 1 | 0.10% |
| record | reporting_agency | 1 | 0.10% |
| record | manufacturer: Company that produces the drug | 1 | 0.10% |
| record | clinical_phase: Current stage of clinical development | 1 | 0.10% |
| record | efficacy_rate: Percentage of patients showing therapeutic response | 1 | 0.10% |
| record | hotel.star_rating | 1 | 0.10% |
| record | hotel.location.address | 1 | 0.10% |
| record | hotel.location.street | 1 | 0.10% |
| record | hotel.location.postal_code | 1 | 0.10% |
| record | hotel.location.city | 1 | 0.10% |
| record | hotel.location.region | 1 | 0.10% |
| record | hotel.location.country | 1 | 0.10% |
| record | hotel.location.area_description | 1 | 0.10% |
| record | hotel.location.nearby_landmarks | 1 | 0.10% |
| record | hotel.contact.phone | 1 | 0.10% |
| record | hotel.contact.fax | 1 | 0.10% |
| record | hotel.contact.email | 1 | 0.10% |
| record | hotel.transportation.metro.station | 1 | 0.10% |
| record | hotel.transportation.metro.line | 1 | 0.10% |
| record | hotel.transportation.train_station | 1 | 0.10% |
| record | hotel.transportation.airport | 1 | 0.10% |
| record | hotel.transportation.driving_directions | 1 | 0.10% |
| record | hotel.transportation.parking.0.walking_time | 1 | 0.10% |
| record | hotel.transportation.parking.0.cost | 1 | 0.10% |
| record | hotel.transportation.parking.0.name | 1 | 0.10% |
| record | hotel.transportation.parking.1.walking_time | 1 | 0.10% |
| record | hotel.transportation.parking.1.directions | 1 | 0.10% |
| record | hotel.transportation.parking.1.cost | 1 | 0.10% |
| record | hotel.transportation.parking.1.name | 1 | 0.10% |
| record | enrollment_id | 1 | 0.10% |
| record | contact_phone | 1 | 0.10% |
| record | participant_name | 1 | 0.10% |
| record | publicationDate | 1 | 0.10% |
| record | features: List of included amenities or space features | 1 | 0.10% |
| record | location: Full address of the leased property | 1 | 0.10% |
| record | spo2 | 1 | 0.10% |
| record | temp | 1 | 0.10% |
| record | investor: Name of the investment firm conducting the meeting | 1 | 0.10% |
| record | startup: Name of the startup receiving investment | 1 | 0.10% |
| record | amount: Investment amount in millions of dollars | 1 | 0.10% |
| record | meeting_code: Unique code assigned to the briefing session | 1 | 0.10% |
| record | parties: List of countries involved in the treaty | 1 | 0.10% |
| record | tournament_id: Unique identifier for the tournament event | 1 | 0.10% |
| record | name: Full name of the esports tournament | 1 | 0.10% |
| record | date: Date when the tournament took place | 1 | 0.10% |
| record | location: Geographic location of the tournament | 1 | 0.10% |
| record | prize_pool: Total prize money in US dollars | 1 | 0.10% |
| record | format: Match structure used in the tournament | 1 | 0.10% |
| record | status: Current state of the tournament | 1 | 0.10% |
| record | lastdosedate | 1 | 0.10% |
| record | status: Current state of the incident | 1 | 0.10% |
| record | address: Full residential or commercial address of the customer | 1 | 0.10% |
| record | impact: Description of service impact including data loss or business effects | 1 | 0.10% |
| record | review: Text content of the customer's feedback on the product | 1 | 0.10% |
| record | price: Formatted price including dollar sign and cents | 1 | 0.10% |
| record | product_id: Unique identifier for the product reviewed | 1 | 0.10% |
| record | item_description.shipping.ready_to_ship_in.quantity | 1 | 0.10% |
| record | item_description.shipping.rush_details | 1 | 0.10% |
| record | item_description.weight.unit | 1 | 0.10% |
| record | item_description.dimensions.length | 1 | 0.10% |
| record | item_description.setup_charge.per | 1 | 0.10% |
| record | item_description.setup_charge.amount | 1 | 0.10% |
| record | item_description.weight.value | 1 | 0.10% |
| record | purchase_order: Order number from original purchase | 1 | 0.10% |
| record | submitter: Name of the customer who submitted the ticket | 1 | 0.10% |
| record | device_model: Model name of the returned electronic device | 1 | 0.10% |
| record | serial_number: Unique hardware identifier for the device | 1 | 0.10% |
| record | physical_characteristics.observations.0.telescope.country | 1 | 0.10% |
| record | physical_characteristics.observations.0.telescope.launch_year | 1 | 0.10% |
| record | hardware_support.gpu.amd.version | 1 | 0.10% |
| record | hardware_support.gpu.nvidia.api | 1 | 0.10% |
| record | operating_systems | 1 | 0.10% |
| record | miner_name | 1 | 0.10% |
| record | developer_fee | 1 | 0.10% |
| record | supported_cryptocurrencies | 1 | 0.10% |
| record | user_id | 1 | 0.10% |
| record | orbital_characteristics.inclination.reference | 1 | 0.10% |
| record | orbital_characteristics.asteroid_belt_classification | 1 | 0.10% |
| record | observations.number_of_observations | 1 | 0.10% |
| record | observations.observation_period.years | 1 | 0.10% |
| record | alternative_designations | 1 | 0.10% |
| record | wiki_sky_links | 1 | 0.10% |
| record | orbital_characteristics.orbital_period.years | 1 | 0.10% |
| record | discovery.location.observatory | 1 | 0.10% |
| record | artist | 1 | 0.10% |
| record | record_label | 1 | 0.10% |
| record | album | 1 | 0.10% |
| record | department: Department where the employee is assigned | 1 | 0.10% |
| record | heart_rate | 1 | 0.10% |
| record | social_media_platforms | 1 | 0.10% |
| record | facebook_features | 1 | 0.10% |
| record | features: List of key technological or functional capabilities | 1 | 0.10% |
| record | price: Base price of the product in US dollars | 1 | 0.10% |
| record | regions: Geographic areas where the product is distributed or in use | 1 | 0.10% |
| record | name: Name of the product as introduced | 1 | 0.10% |
| record | headliner: Main performing artist for the festival | 1 | 0.10% |
| record | name: Name of the festival as listed in the schedule | 1 | 0.10% |
| record | dimensions: Dimensions in cm format | 1 | 0.10% |
| record | price: Price of the product in dollars | 1 | 0.10% |
| record | product_id: Unique identifier for the product | 1 | 0.10% |
| record | affected_area | 1 | 0.10% |
| record | weight_loss_start_date | 1 | 0.10% |
| record | symptom_joint_pain | 1 | 0.10% |
| record | political_context | 1 | 0.10% |
| record | risk_rating | 1 | 0.10% |
| record | property_type | 1 | 0.10% |
| record | sprouting_definition | 1 | 0.10% |
| record | affiliation.institution_name | 1 | 0.10% |
| record | affiliation.department | 1 | 0.10% |
| record | affiliation.location | 1 | 0.10% |
| record | research_background.focus_area | 1 | 0.10% |
| record | research_background.animal_models_used | 1 | 0.10% |
| record | statement | 1 | 0.10% |
| record | features: Key functionalities or tools included | 1 | 0.10% |
| record | geographic_region | 1 | 0.10% |
| record | return_type | 1 | 0.10% |
| record | insurer | 1 | 0.10% |
| record | term_length | 1 | 0.10% |
| record | policy_type | 1 | 0.10% |
| record | asteroid.discovery.discovery_rank | 1 | 0.10% |
| record | asteroid.discovery.date | 1 | 0.10% |
| record | asteroid.description.discovery_details.location.state | 1 | 0.10% |
| record | asteroid.orbital_characteristics.classification | 1 | 0.10% |
| record | asteroid.description.observations.observation_period_years | 1 | 0.10% |
| record | asteroid.description.orbital_period_years | 1 | 0.10% |
| record | asteroid.alternative_designations | 1 | 0.10% |
| record | asteroid.external_links.0.title | 1 | 0.10% |
| record | asteroid.wiki_sky_links | 1 | 0.10% |
| record | asteroid.description.observations.last_observation_date | 1 | 0.10% |
| record | asteroid.description.discovery_details.location.county | 1 | 0.10% |
| record | appearances | 1 | 0.10% |
| record | test_name: Name of the laboratory test performed | 1 | 0.10% |
| record | result_value: Measured value of the lab test | 1 | 0.10% |
| record | reference_range: Normal range for the test value | 1 | 0.10% |
| record | test_date: Date the lab test was conducted | 1 | 0.10% |
| record | birth_date: Patient's date of birth | 1 | 0.10% |
| record | clinical_note: Clinical interpretation or patient-reported symptoms and recommendations | 1 | 0.10% |
| record | renewal_amount | 1 | 0.10% |
| record | headliner | 1 | 0.10% |
| record | actions | 1 | 0.10% |
| record | distributor_information.capabilities | 1 | 0.10% |
| record | services.maintenance | 1 | 0.10% |
| record | services.selection_assistance | 1 | 0.10% |
| record | categories | 1 | 0.10% |
| record | alternative_names | 1 | 0.10% |
| record | ticket_count | 1 | 0.10% |
| record | customer_id | 1 | 0.10% |
| record | incident_id | 1 | 0.10% |
| record | date_reported | 1 | 0.10% |
| record | responsible_engineer | 1 | 0.10% |
| record | environment | 1 | 0.10% |
| record | location: Venue where the match took place | 1 | 0.10% |
| record | date: Date the match was scheduled or played | 1 | 0.10% |
| record | score: List of two integers representing home and away scores | 1 | 0.10% |
| record | status: Current match status | 1 | 0.10% |
| record | claim_number: Unique identifier for the health claim | 1 | 0.10% |
| record | patient_name: Full name of the claimant | 1 | 0.10% |
| record | age: Patient's age in years | 1 | 0.10% |
| record | diagnosis: Medical diagnosis code and description | 1 | 0.10% |
| record | amount: Total cost of the medical service | 1 | 0.10% |
| record | owner: Name of the claimant or policyholder | 1 | 0.10% |
| record | waterway | 1 | 0.10% |
| record | nationality | 1 | 0.10% |
| record | military unit | 1 | 0.10% |
| record | island | 1 | 0.10% |
| record | warehouse_location: location name of the warehouse facility | 1 | 0.10% |
| record | sku: unique product identifier from supply chain database | 1 | 0.10% |
| record | quantity: number of units currently in stock | 1 | 0.10% |
| record | funding_request | 1 | 0.10% |
| record | donor_name: Full name of the contributor | 1 | 0.10% |
| record | recipient_jurisdiction: Jurisdiction such as city, state, or federal level | 1 | 0.10% |
| record | product.specifications.camera.features | 1 | 0.10% |
| record | product.registration_count.time_period | 1 | 0.10% |
| record | product.specifications.memory.ram | 1 | 0.10% |
| record | product.specifications.connectivity | 1 | 0.10% |
| record | product.price.1.currency | 1 | 0.10% |
| record | product.model | 1 | 0.10% |
| record | product.launch_date | 1 | 0.10% |
| record | product.specifications.memory.internal_storage | 1 | 0.10% |
| record | product.brand | 1 | 0.10% |
| record | product.availability.future_availability | 1 | 0.10% |
| record | scorer | 1 | 0.10% |
| record | performance_time | 1 | 0.10% |
| record | audience_rating | 1 | 0.10% |
| record | mark | 1 | 0.10% |
| record | tier: Subscription tier level such as basic or premium | 1 | 0.10% |
| record | city: City where the subscriber resides | 1 | 0.10% |
| record | subscription_id: Unique identifier for the subscription confirmation | 1 | 0.10% |
| record | status: Current status of the subscription | 1 | 0.10% |
| record | delivery_frequency: How often the newsletter is delivered | 1 | 0.10% |
| record | assigned_to | 1 | 0.10% |
| record | dedication.name | 1 | 0.10% |
| record | course_code | 1 | 0.10% |
| record | ceo | 1 | 0.10% |
| record | study_metadata.study_type | 1 | 0.10% |
| record | outcomes.fracture_events.0.type | 1 | 0.10% |
| record | population.age_range.min | 1 | 0.10% |
| record | statistical_results.hazard_ratios.0.adjustment_factors | 1 | 0.10% |
| record | statistical_results.diagnostic_metrics.sensitivity_at_threshold.percentage_of_fractures | 1 | 0.10% |
| record | prize_pool | 1 | 0.10% |
| record | zebras_care_foundation.beneficiary.family | 1 | 0.10% |
| record | event.location | 1 | 0.10% |
| record | functional_area | 1 | 0.10% |
| record | actor_name | 1 | 0.10% |
| record | related_content | 1 | 0.10% |
| record | diagnosis_code: Medical code indicating the patient's condition | 1 | 0.10% |
| record | policy_id: Insurance policy number associated with the claim | 1 | 0.10% |
| record | dateofservice: Date when medical service was rendered | 1 | 0.10% |
| record | contributor | 1 | 0.10% |
| record | venue: Name of the stadium or venue where the event occurred | 1 | 0.10% |
| record | nation: Country of the athlete who won the medal | 1 | 0.10% |
| record | time: Official race time recorded in the event | 1 | 0.10% |
| record | medal: Type of medal won gold, silver, or bronze | 1 | 0.10% |
| record | affiliation: Institutional affiliation of the author | 1 | 0.10% |
| record | paper_title: Title of the submitted research paper | 1 | 0.10% |
| record | abstract_length: Length of the abstract in words | 1 | 0.10% |
| record | author_name: Name of the paper's author | 1 | 0.10% |
| record | submission_id: Unique identifier for the paper submission | 1 | 0.10% |
| record | funding_source: Source of research funding | 1 | 0.10% |
| record | fieldofstudy: Academic domain of the research | 1 | 0.10% |
| record | review_date: Date when the paper was reviewed | 1 | 0.10% |
| record | conference_name: Name of the conference where submitted | 1 | 0.10% |
| record | serial_number: Unique device identifier assigned by manufacturer | 1 | 0.10% |
| record | manufacturer: Company that produced the device | 1 | 0.10% |
| record | cost: Total cost of the device in dollars | 1 | 0.10% |
| record | assigned_staff: Healthcare professional assigned to manage the device | 1 | 0.10% |
| record | transaction_id: Unique identifier for each transaction entry | 1 | 0.10% |
| record | amount: Monetary value of the transaction in USD | 1 | 0.10% |
| record | description: Brief description of the transaction purpose | 1 | 0.10% |
| record | payee: Name of the entity receiving or sending funds | 1 | 0.10% |
| record | reference_code: Internal code linking to external system or invoice | 1 | 0.10% |
| record | recipient_company | 1 | 0.10% |
| record | location.country | 1 | 0.10% |
| record | amenities | 1 | 0.10% |
| record | ranking.rank | 1 | 0.10% |
| record | ranking.total_hotels | 1 | 0.10% |
| record | ranking.location | 1 | 0.10% |
| record | price_range.currency | 1 | 0.10% |
| record | hotel_details.star_rating | 1 | 0.10% |
| record | hotel_details.name | 1 | 0.10% |
| record | hotel_details.location | 1 | 0.10% |
| record | booking_options.partners | 1 | 0.10% |
| record | booking_options.description | 1 | 0.10% |
| record | asteroid.dedication.person.occupation | 1 | 0.10% |
| record | asteroid.dedication.person.name | 1 | 0.10% |
| record | sku_id: Unique identifier for the product in inventory system | 1 | 0.10% |
| record | name: Product name as listed in inventory records | 1 | 0.10% |
| record | location: Warehouse or hub where the SKU is stored | 1 | 0.10% |
| record | quantity: Current stock level of the SKU | 1 | 0.10% |
| record | last_updated: Date when inventory was last modified | 1 | 0.10% |
| record | supplier: Company that supplies the product | 1 | 0.10% |
| record | reorder_point: Minimum stock level before triggering reorder | 1 | 0.10% |
| record | price: Current selling price of the SKU in USD | 1 | 0.10% |
| record | discoverers | 1 | 0.10% |
| record | wins | 1 | 0.10% |
| record | games_behind | 1 | 0.10% |
| record | coach | 1 | 0.10% |
| record | win_percentage | 1 | 0.10% |
| record | games_played | 1 | 0.10% |
| record | responsible_staff | 1 | 0.10% |
| record | event-attack/battle/war/militaryconflict | 1 | 0.10% |
| record | claim_id: Unique identifier for the insurance claim | 1 | 0.10% |
| record | damage_amount: Total monetary damage assessed in dollars | 1 | 0.10% |
| record | vehicle: Make, model, and year of the vehicle involved | 1 | 0.10% |
| record | incident_type: Description of the type of incident | 1 | 0.10% |
| record | adjuster: Name of the assigned claims adjuster | 1 | 0.10% |
| record | policy_number: Unique policy number associated with the claim | 1 | 0.10% |
| record | school_name | 1 | 0.10% |
| record | performance_improvement | 1 | 0.10% |
| record | funding_code | 1 | 0.10% |
| record | targetgradelevel | 1 | 0.10% |
| record | case_number | 1 | 0.10% |
| record | test_id | 1 | 0.10% |
| record | match_id: Unique identifier for the match as provided | 1 | 0.10% |
| record | away_team: Team playing at the away venue | 1 | 0.10% |
| record | score: Final score in format home-away | 1 | 0.10% |
| record | venue: Stadium where the match took place | 1 | 0.10% |
| record | competition: League or cup type such as Premier League or friendly | 1 | 0.10% |
| record | management_contact: Name or entity managing the property | 1 | 0.10% |
| record | city: City where the property is located | 1 | 0.10% |
| record | rentalboardstatus: Status as reported on the city’s rental board | 1 | 0.10% |
| record | property_id: Unique identifier assigned to the rental unit | 1 | 0.10% |
| record | injury_count | 1 | 0.10% |
| record | company_name: Name of the technology company receiving funding | 1 | 0.10% |
| record | location: Physical address of the company headquarters | 1 | 0.10% |
| record | office_address | 1 | 0.10% |
| record | basin_name | 1 | 0.10% |
| record | statement_reference: Identifier code for the statement document | 1 | 0.10% |
| record | amount: Financial value associated with the policy or proposal | 1 | 0.10% |
| record | location: City and state where the event occurred | 1 | 0.10% |
| record | topic: Subject matter discussed in the statement | 1 | 0.10% |
| record | speaker: Name of the person delivering the statement | 1 | 0.10% |
| record | steps | 1 | 0.10% |
| record | agency | 1 | 0.10% |
| record | budget | 1 | 0.10% |
| record | orbital_characteristics.reference_plane | 1 | 0.10% |
| record | funds_use | 1 | 0.10% |
| record | investor_leads | 1 | 0.10% |
| record | address: Full street address of the recipient | 1 | 0.10% |
| record | award_code: Unique identifier code for the award | 1 | 0.10% |
| record | city: City where the recipient resides | 1 | 0.10% |
| record | state: State or province where the recipient resides | 1 | 0.10% |
| record | episode_number | 1 | 0.10% |
| record | resources | 1 | 0.10% |
| record | powers_abilities | 1 | 0.10% |
| record | department: Department where the individual is assigned | 1 | 0.10% |
| record | status: Current enrollment status | 1 | 0.10% |
| record | cost: Total cost of the enrollment program | 1 | 0.10% |
| record | era_name | 1 | 0.10% |
| record | year_in_era | 1 | 0.10% |
| record | dynasty | 1 | 0.10% |
| record | person.email_address | 1 | 0.10% |
| record | person.phone_number | 1 | 0.10% |
| record | amount_requested: Total amount requested for compensation | 1 | 0.10% |
| record | description: Brief description of the incident and damages | 1 | 0.10% |
| record | policyholder_name: Name of the policyholder filing the claim | 1 | 0.10% |
| record | status: Current status of the claim | 1 | 0.10% |
| record | employee_name | 1 | 0.10% |
| record | test_name | 1 | 0.10% |
| record | issue | 1 | 0.10% |
| record | venues | 1 | 0.10% |
| record | leagues | 1 | 0.10% |
| record | rainfall | 1 | 0.10% |
| record | billed_amount | 1 | 0.10% |
| record | passenger_phone | 1 | 0.10% |
| record | target_department | 1 | 0.10% |
| record | spoofed_domain | 1 | 0.10% |
| record | report_metadata.publication_date | 1 | 0.10% |
| record | company.financial_results.metrics.0.historical_comparison.comparison_period | 1 | 0.10% |
| record | report_metadata.source_url | 1 | 0.10% |
| record | company.financial_results.currency | 1 | 0.10% |
| record | exam_type | 1 | 0.10% |
| record | protocol | 1 | 0.10% |
| record | description: Detailed account of the outage and impact | 1 | 0.10% |
| record | tier: Data speed tier of the service plan | 1 | 0.10% |
| record | status: Current medical status of the patient | 1 | 0.10% |
| record | symptoms: List of reported symptoms like fever or headache | 1 | 0.10% |
| record | reviewer_address | 1 | 0.10% |
| record | compensation_type | 1 | 0.10% |
| record | nmod | 1 | 0.10% |
| record | preconj | 1 | 0.10% |
| record | companies | 1 | 0.10% |
| record | riskfactorcode | 1 | 0.10% |
| record | applicant | 1 | 0.10% |
| record | lastrevieweddate | 1 | 0.10% |
| record | agent_id | 1 | 0.10% |
| record | office | 1 | 0.10% |
| record | public_information.download_share | 1 | 0.10% |
| record | public_information.brochure_name | 1 | 0.10% |
| record | advertiser_information.new_brochure_planned | 1 | 0.10% |
| record | advertiser_information.contact_information.phone_number | 1 | 0.10% |
| record | advertiser_information.contact_information.additional_information | 1 | 0.10% |
| record | distribution_information.distribution_locations | 1 | 0.10% |
| record | distribution_information.trail_details.trail_name | 1 | 0.10% |
| record | distribution_information.trail_details.trail_length | 1 | 0.10% |
| record | distribution_information.trail_details.trail_endpoints | 1 | 0.10% |
| record | path_design.materials | 1 | 0.10% |
| record | path_design.methods | 1 | 0.10% |
| record | path_design.advantages | 1 | 0.10% |
| record | path_design.disadvantages | 1 | 0.10% |
| record | path_design.tile_based_paths.application_areas | 1 | 0.10% |
| record | path_design.concrete_tile_making.mold_description | 1 | 0.10% |
| record | path_design.concrete_tile_making.steps | 1 | 0.10% |
| record | path_design.concrete_tile_making.tips | 1 | 0.10% |
| record | path_design.concrete_tile_making.mold_size_example | 1 | 0.10% |
| record | path_design.concrete_tile_making.equipment | 1 | 0.10% |
| record | path_design.concrete_tile_making.design_options | 1 | 0.10% |
| record | path_design.skill_level | 1 | 0.10% |
| record | nutritional_aspects.sucrose.daily_limit_percentage | 1 | 0.10% |
| record | nutritional_aspects.carbohydrates.daily_percentage | 1 | 0.10% |
| record | nutritional_aspects.carbohydrates.low_glycemic_index.percentage_with_fiber | 1 | 0.10% |
| record | weight_loss_benefits.improvements | 1 | 0.10% |
| record | nutritional_aspects.carbohydrates.low_glycemic_index.recommendation | 1 | 0.10% |
| record | note | 1 | 0.10% |
| record | email: Contact email address of the ticket submitter | 1 | 0.10% |
| record | related_amount: Financial amount involved if applicable | 1 | 0.10% |
| record | submitter: Name of the user who opened the ticket | 1 | 0.10% |
| record | ticket_id: Unique identifier for the support ticket | 1 | 0.10% |
| record | position: Job title or role within the department | 1 | 0.10% |
| record | appointment_id: Unique identifier for the appointment | 1 | 0.10% |
| record | status: Current status of the appointment | 1 | 0.10% |
| record | compensation: Annual salary or compensation amount | 1 | 0.10% |
| record | election_district | 1 | 0.10% |
| record | issue_description | 1 | 0.10% |
| record | test_code: Code assigned to the specific test | 1 | 0.10% |
| record | plaintiff: Individual or entity initiating the lawsuit | 1 | 0.10% |
| record | defendant: Individual or entity being sued | 1 | 0.10% |
| record | citations.4.description | 1 | 0.10% |
| record | gymnosperm_diversity.diversity_center | 1 | 0.10% |
| record | citations.0.year | 1 | 0.10% |
| record | gymnosperm_diversity.rare_species | 1 | 0.10% |
| record | location.mentioned_states | 1 | 0.10% |
| record | taxonomic_reference | 1 | 0.10% |
| record | citations.3.author | 1 | 0.10% |
| record | gymnosperm_diversity.endemic_species | 1 | 0.10% |
| record | gymnosperm_diversity.pine_species_count | 1 | 0.10% |
| record | gymnosperm_diversity.species_examples.1.characteristic | 1 | 0.10% |
| record | citations.1.year | 1 | 0.10% |
| record | gymnosperm_diversity.endemism_ranking | 1 | 0.10% |
| record | gymnosperm_diversity.species_examples.3.characteristic | 1 | 0.10% |
| record | citations.4.author | 1 | 0.10% |
| record | citations.4.year | 1 | 0.10% |
| record | citations.1.author | 1 | 0.10% |
| record | gymnosperm_diversity.species_examples.0.name | 1 | 0.10% |
| record | citations.2.description | 1 | 0.10% |
| record | gymnosperm_diversity.total_species | 1 | 0.10% |
| record | gymnosperm_diversity.species_examples.1.name | 1 | 0.10% |
| record | gymnosperm_diversity.species_examples.0.characteristic | 1 | 0.10% |
| record | meeting_code | 1 | 0.10% |
| record | performance_rating: Numerical score on a 5-point scale reflecting performance | 1 | 0.10% |
| record | kpivendorcompliance: Percentage of vendor contract adherence met | 1 | 0.10% |
| record | peerfeedbackrating: Percentage of positive peer feedback received | 1 | 0.10% |
| record | manager_name: Name of the employee's direct manager | 1 | 0.10% |
| record | company.products | 1 | 0.10% |
| record | company.contact_information.fax_numbers | 1 | 0.10% |
| record | company.services | 1 | 0.10% |
| record | company.address.street | 1 | 0.10% |
| record | company.locations | 1 | 0.10% |
| record | company.contact_information.phone_numbers | 1 | 0.10% |
| record | company.address.building_number | 1 | 0.10% |
| record | company.address.area | 1 | 0.10% |
| record | appointment_code: Internal code for the appointment decision | 1 | 0.10% |
| record | product_model | 1 | 0.10% |
| record | claimant_name | 1 | 0.10% |
| record | claim_date | 1 | 0.10% |
| record | finding_id: Unique identifier for the audit finding | 1 | 0.10% |
| record | violation_type: Category of compliance issue identified | 1 | 0.10% |
| record | affected_entity: Organization or business impacted by the violation | 1 | 0.10% |
| record | location: Physical address of the affected entity | 1 | 0.10% |
| record | violation_code: Code assigned to the specific violation | 1 | 0.10% |
| record | description: Text explaining the nature and impact of the violation | 1 | 0.10% |
| record | account | 1 | 0.10% |
| record | receiver_name | 1 | 0.10% |
| record | receiver_location | 1 | 0.10% |
| record | description: Purpose or label of the transaction | 1 | 0.10% |
| record | status: Current transaction status | 1 | 0.10% |
| record | comment_id | 1 | 0.10% |
| record | submission_date | 1 | 0.10% |
| record | party_name | 1 | 0.10% |
| record | status: Current status of the prescription | 1 | 0.10% |
| record | patient_name: Full name of the patient receiving the medication | 1 | 0.10% |
| record | supply_quantity: Total number of tablets dispensed | 1 | 0.10% |
| record | dosage: Instructions on how and when to take the medication | 1 | 0.10% |
| record | patient_address: Street address of the patient | 1 | 0.10% |
| record | reviewer_name: Name of the reviewer providing feedback | 1 | 0.10% |
| record | forks: Number of forks the repository has received | 1 | 0.10% |
| record | repository_url: GitHub URL hosting the project | 1 | 0.10% |
| record | review_score: Numerical score out of 5 based on user feedback | 1 | 0.10% |
| record | stars: Number of stars the repository has received | 1 | 0.10% |
| record | product.origin.method_origin | 1 | 0.10% |
| record | product.origin.initial_location | 1 | 0.10% |
| record | product.origin.spread_location | 1 | 0.10% |
| record | product.ingredients | 1 | 0.10% |
| record | product.preparation.peach_selection | 1 | 0.10% |
| record | product.preparation.steps | 1 | 0.10% |
| record | product.preparation.drying_time | 1 | 0.10% |
| record | product.storage.containers | 1 | 0.10% |
| record | product.storage.wrapping | 1 | 0.10% |
| record | product.serving | 1 | 0.10% |
| record | historical_context.cappers_weekly.call_for_articles_year | 1 | 0.10% |
| record | historical_context.cappers_weekly.topic | 1 | 0.10% |
| record | historical_context.my_folks_series.first_book_title | 1 | 0.10% |
| record | historical_context.my_folks_series.first_book_year | 1 | 0.10% |
| record | historical_context.my_folks_series.number_of_books | 1 | 0.10% |
| record | description: Detailed account of the issue reported | 1 | 0.10% |
| record | submitter: Name of the employee who submitted the ticket | 1 | 0.10% |
| record | affected_system: Software or hardware system experiencing the problem | 1 | 0.10% |
| record | priority: Urgency level of the ticket | 1 | 0.10% |
| record | status: Current state of the ticket | 1 | 0.10% |
| record | entity_name | 1 | 0.10% |
| record | cusip | 1 | 0.10% |
| record | wallet | 1 | 0.10% |
| record | exchange | 1 | 0.10% |
| record | initiation_date | 1 | 0.10% |
| record | downstream_genes | 1 | 0.10% |
| record | network_interaction | 1 | 0.10% |
| record | data_collection | 1 | 0.10% |
| record | next_milestone | 1 | 0.10% |
| record | usability_score | 1 | 0.10% |
| record | store_reference | 1 | 0.10% |
| record | buyer_name | 1 | 0.10% |
| record | game_id | 1 | 0.10% |
| record | visitor_team | 1 | 0.10% |
| record | home_score | 1 | 0.10% |
| record | visitor_score | 1 | 0.10% |
| record | storage_location | 1 | 0.10% |
| record | article_metadata.origin_newsroom | 1 | 0.10% |
| record | organization.legal_form | 1 | 0.10% |
| record | organization.personnel_changes.0.individual.name | 1 | 0.10% |
| record | organization.personnel_changes.0.effective_date_description | 1 | 0.10% |
| record | attorney | 1 | 0.10% |
| record | counsel | 1 | 0.10% |
| record | abstract: Brief summary of the invention's technical approach and purpose | 1 | 0.10% |
| record | acquired_company | 1 | 0.10% |
| record | revenue_projection | 1 | 0.10% |
| record | strategic_impact | 1 | 0.10% |
| record | integration_lead | 1 | 0.10% |
| record | patent_number | 1 | 0.10% |
| record | filer_name | 1 | 0.10% |
| record | ticket_number | 1 | 0.10% |
| record | key_feature | 1 | 0.10% |
| record | job_id: Unique identifier assigned to the job posting | 1 | 0.10% |
| record | company: Name of the company issuing the job posting | 1 | 0.10% |
| record | role: Title of the job position being advertised | 1 | 0.10% |
| record | location: City and country where the role is based | 1 | 0.10% |
| record | work_type: Type of work arrangement for the role | 1 | 0.10% |
| record | acquisition_value | 1 | 0.10% |
| record | additional_investors | 1 | 0.10% |
| record | release_date | 1 | 0.10% |
| record | apparent_magnitude.max | 1 | 0.10% |
| record | characteristics.type | 1 | 0.10% |
| record | distance.light_years | 1 | 0.10% |
| record | constellation | 1 | 0.10% |
| record | apparent_magnitude.unit | 1 | 0.10% |
| record | religious_concepts | 1 | 0.10% |
| record | status: Current state of the transaction processing | 1 | 0.10% |
| record | source: Origin of the exchange rate data feed | 1 | 0.10% |
| record | entry_id: Unique identifier for each exchange rate entry | 1 | 0.10% |
| record | transaction_type: Type of transaction such as shipment, purchase, or invoice | 1 | 0.10% |
| record | methodology.proposed_solution | 1 | 0.10% |
| record | outcomes.comparative_advantage.alternative_to | 1 | 0.10% |
| record | research_context.problem_statement.limitation | 1 | 0.10% |
| record | outcomes.comparative_advantage.benefit_type | 1 | 0.10% |
| record | research_context.primary_technique | 1 | 0.10% |
| record | methodology.experimental_setup.biological_model.culture_environment.material_properties | 1 | 0.10% |
| record | research_context.application_goal | 1 | 0.10% |
| record | methodology.experimental_setup.reagents.0.labeling | 1 | 0.10% |
| record | passenger_address | 1 | 0.10% |
| record | countries | 1 | 0.10% |
| record | regions_or_states | 1 | 0.10% |
| record | funding_type | 1 | 0.10% |
| record | first_office | 1 | 0.10% |
| record | policyholder_id: Unique identifier for the policyholder | 1 | 0.10% |
| record | handle | 1 | 0.10% |
| record | currency_pair: Pair of currencies involved in the exchange | 1 | 0.10% |
| record | sender: Name of the individual initiating the transfer | 1 | 0.10% |
| record | Dariusz Śledź <> participated in | 1 | 0.10% |
| record | Dariusz Śledź <> won | 1 | 0.10% |
| record | WTS Atlas <> managed by | 1 | 0.10% |
| record | investors: List of firms or individuals providing capital | 1 | 0.10% |
| record | photo_quality | 1 | 0.10% |
| record | retweets | 1 | 0.10% |
| record | location_tag | 1 | 0.10% |
| record | products: List of items included in the campaign bundle | 1 | 0.10% |
| record | email_id | 1 | 0.10% |
| record | development_periods | 1 | 0.10% |
| record | see_also | 1 | 0.10% |
| record | stub_article.call_to_action | 1 | 0.10% |
| record | charging_type | 1 | 0.10% |
| record | patient_id: Unique identifier for the patient in the record | 1 | 0.10% |
| record | dateofexam: Date when the imaging examination was performed | 1 | 0.10% |
| record | location: Anatomical region examined | 1 | 0.10% |
| record | radiologist: Name of the physician who interpreted the report | 1 | 0.10% |
| record | report_id: Unique identifier assigned to the imaging report | 1 | 0.10% |
| record | findings: Diagnostic observations and conclusions from the imaging study | 1 | 0.10% |
| record | goal | 1 | 0.10% |
| record | consensus_percentage | 1 | 0.10% |
| record | treaty_name | 1 | 0.10% |
| record | signatories | 1 | 0.10% |
| record | treaty_code | 1 | 0.10% |
| record | signing_countries | 1 | 0.10% |
| record | donor_id | 1 | 0.10% |
| record | systematics_and_distribution.location.elevation_m.min | 1 | 0.10% |
| record | description.fruit.width_mm.max | 1 | 0.10% |
| record | systematics_and_distribution.location.elevation_m.max | 1 | 0.10% |
| record | description.flowers.length_mm.max | 1 | 0.10% |
| record | description.fruit.width_mm.min | 1 | 0.10% |
| record | description.flowering_time | 1 | 0.10% |
| record | systematics_and_distribution.location.state | 1 | 0.10% |
| record | description.leaves.surface | 1 | 0.10% |
| record | description.flowers.length_mm.min | 1 | 0.10% |
| record | description.fruit.length_mm.min | 1 | 0.10% |
| record | common_names | 1 | 0.10% |
| record | description.leaves.margins | 1 | 0.10% |
| record | description.rosette | 1 | 0.10% |
| record | description.flowers.color | 1 | 0.10% |
| record | systematics_and_distribution.distinguishing_features | 1 | 0.10% |
| record | description.inflorescence.type | 1 | 0.10% |
| record | description.seeds.color | 1 | 0.10% |
| record | description.flowers.arrangement | 1 | 0.10% |
| record | description.flowers.perianth_tube_length_mm.min | 1 | 0.10% |
| record | systematics_and_distribution.first_description.author | 1 | 0.10% |
| record | systematics_and_distribution.similar_species | 1 | 0.10% |
| record | description.inflorescence.height_m.min | 1 | 0.10% |
| record | systematics_and_distribution.location.country | 1 | 0.10% |
| record | description.leaves.terminal_spine.color | 1 | 0.10% |
| record | description.leaves.width_cm.min | 1 | 0.10% |
| record | tone | 1 | 0.10% |
| record | upvote_count | 1 | 0.10% |
| record | founder_names | 1 | 0.10% |
| record | investor_group | 1 | 0.10% |
| record | ticket_id: Unique identifier for the support case | 1 | 0.10% |
| record | user: Name of the user who filed the ticket | 1 | 0.10% |
| record | location: Geographic location of the user or company | 1 | 0.10% |
| record | priority: Severity level assigned to the ticket | 1 | 0.10% |
| record | strength | 1 | 0.10% |
| record | approval_date | 1 | 0.10% |
| record | return_reason | 1 | 0.10% |
| record | assists_leader | 1 | 0.10% |
| record | score_home | 1 | 0.10% |
| record | exam_centers | 1 | 0.10% |
| record | additional_information_url | 1 | 0.10% |
| record | certification.name | 1 | 0.10% |
| record | certification.description | 1 | 0.10% |
| record | certification.prerequisites | 1 | 0.10% |
| record | certification.validity | 1 | 0.10% |
| record | certification.exams.0.code | 1 | 0.10% |
| record | certification.exams.0.name | 1 | 0.10% |
| record | certification.exams.1.code | 1 | 0.10% |
| record | certification.exams.1.name | 1 | 0.10% |
| record | certification.technologies | 1 | 0.10% |
| record | personnel.0.organization | 1 | 0.10% |
| record | personnel.0.title | 1 | 0.10% |
| record | personnel.0.name | 1 | 0.10% |
| record | organization.website | 1 | 0.10% |
| record | training.providers | 1 | 0.10% |
| record | training.delivery_method | 1 | 0.10% |
| record | accuracyorefficiency | 1 | 0.10% |
| record | terminology.ideogram.english | 1 | 0.10% |
| record | terminology.ideographic_script.english | 1 | 0.10% |
| record | comparison.alphabetic.learning_curve | 1 | 0.10% |
| record | comparison.alphabetic.pronunciation | 1 | 0.10% |
| record | comparison.ideographic.learning_curve | 1 | 0.10% |
| record | comparison.ideographic.cross_lingual_understanding | 1 | 0.10% |
| record | comparison.ideographic.example.character | 1 | 0.10% |
| record | comparison.ideographic.example.meaning | 1 | 0.10% |
| record | comparison.ideographic.example.languages.0.language | 1 | 0.10% |
| record | comparison.ideographic.example.languages.0.pronunciation | 1 | 0.10% |
| record | comparison.ideographic.example.languages.1.language | 1 | 0.10% |
| record | comparison.ideographic.example.languages.1.pronunciation | 1 | 0.10% |
| record | comparison.ideographic.historical_understanding | 1 | 0.10% |
| record | caveat.example.character | 1 | 0.10% |
| record | caveat.example.pronunciation | 1 | 0.10% |
| record | caveat.example.meaning | 1 | 0.10% |
| record | caveat.example.components.0.function | 1 | 0.10% |
| record | caveat.example.components.0.component | 1 | 0.10% |
| record | caveat.example.components.0.meaning | 1 | 0.10% |
| record | caveat.example.components.0.related_characters.0.pronunciation | 1 | 0.10% |
| record | caveat.example.components.0.related_characters.0.character | 1 | 0.10% |
| record | caveat.example.components.0.related_characters.0.meaning | 1 | 0.10% |
| record | caveat.example.components.0.related_characters.1.pronunciation | 1 | 0.10% |
| record | caveat.example.components.0.related_characters.1.character | 1 | 0.10% |
| record | caveat.example.components.0.related_characters.1.meaning | 1 | 0.10% |
| record | caveat.example.components.0.related_characters.2.pronunciation | 1 | 0.10% |
| record | caveat.example.components.0.related_characters.2.character | 1 | 0.10% |
| record | caveat.example.components.0.related_characters.2.meaning | 1 | 0.10% |
| record | caveat.example.components.0.position | 1 | 0.10% |
| record | caveat.example.components.1.function | 1 | 0.10% |
| record | caveat.example.components.1.component | 1 | 0.10% |
| record | caveat.example.components.1.meaning | 1 | 0.10% |
| record | caveat.example.components.1.position | 1 | 0.10% |
| record | artist_name | 1 | 0.10% |
| record | company: Name of the company raising capital | 1 | 0.10% |
| record | round: Series or type of funding round | 1 | 0.10% |
| record | location: City and state where company is headquartered | 1 | 0.10% |
| record | match | 1 | 0.10% |
| record | mls_number | 1 | 0.10% |
| record | bathrooms | 1 | 0.10% |
| record | heroes | 1 | 0.10% |
| record | villains | 1 | 0.10% |
| record | equipment | 1 | 0.10% |
| record | expression_units | 1 | 0.10% |
| record | gene_id | 1 | 0.10% |
| record | deductiblestatus: Whether patient's deductible has been met | 1 | 0.10% |
| record | treatment_type: Type of medical treatment or procedure | 1 | 0.10% |
| record | diagnosis_code: Standard code indicating the patient's condition | 1 | 0.10% |
| record | claim_id: Unique identifier for each claim | 1 | 0.10% |
| record | events.0.participants | 1 | 0.10% |
| record | breastfeeding_situation.causes | 1 | 0.10% |
| record | countries.1.name | 1 | 0.10% |
| record | countries.2.name | 1 | 0.10% |
| record | events.0.type | 1 | 0.10% |
| record | countries.0.breastfeeding_rate | 1 | 0.10% |
| record | organizations.0.name | 1 | 0.10% |
| record | report_details.source | 1 | 0.10% |
| record | countries.1.breastfeeding_rate | 1 | 0.10% |
| record | hotel_address | 1 | 0.10% |
| record | traveler_location | 1 | 0.10% |
| record | isotope | 1 | 0.10% |
| record | treatment_status | 1 | 0.10% |
| record | treatment_plan | 1 | 0.10% |
| record | isbn | 1 | 0.10% |
| record | publication_volume | 1 | 0.10% |
| record | publication_issue | 1 | 0.10% |
| record | content.warning | 1 | 0.10% |
| record | content.elements.name | 1 | 0.10% |
| record | content.core_beliefs.relationship_with_society | 1 | 0.10% |
| record | patient_age | 1 | 0.10% |
| record | reaction | 1 | 0.10% |
| record | drug_involved | 1 | 0.10% |
| record | inr_value | 1 | 0.10% |
| record | donor_location | 1 | 0.10% |
| record | donorfecid | 1 | 0.10% |
| record | recipient_organization | 1 | 0.10% |
| record | symptom | 1 | 0.10% |
| record | report_date | 1 | 0.10% |
| record | research.experiment_details.laser_details | 1 | 0.10% |
| record | research.experiment_details.setup | 1 | 0.10% |
| record | research.experiment_details.atoms_used | 1 | 0.10% |
| record | device_id: Unique identifier for the medical device | 1 | 0.10% |
| record | study_date | 1 | 0.10% |
| record | age_range | 1 | 0.10% |
| record | status: Current status of the contribution record | 1 | 0.10% |
| record | donor_name: Full name of the individual or organization making the donation | 1 | 0.10% |
| record | contribution_id: Unique identifier for each contribution entry | 1 | 0.10% |
| record | group_name | 1 | 0.10% |
| record | weight_range_kg.min_kg | 1 | 0.10% |
| record | masters_of_ceremony | 1 | 0.10% |
| record | canton_name | 1 | 0.10% |
| record | sales_volume | 1 | 0.10% |
| record | restaurant | 1 | 0.10% |
| record | dish | 1 | 0.10% |
| record | drink | 1 | 0.10% |
| record | food item | 1 | 0.10% |
| record | day | 1 | 0.10% |
| record | ingredient | 1 | 0.10% |
| record | kitchen appliance | 1 | 0.10% |
| record | side dish | 1 | 0.10% |
| record | vacancy_rate.rate | 1 | 0.10% |
| record | lease_agreements.number | 1 | 0.10% |
| record | percentage | 1 | 0.10% |
| record | team_name | 1 | 0.10% |
| record | stage | 1 | 0.10% |
| record | team_id | 1 | 0.10% |
| record | newsletter_name | 1 | 0.10% |
| record | asteroid.designation_number | 1 | 0.10% |
| record | project_name | 1 | 0.10% |
| record | family.name | 1 | 0.10% |
| record | authority.name | 1 | 0.10% |
| record | target_audience: Demographic group the product targets | 1 | 0.10% |
| record | product: Name of the skincare or apparel product | 1 | 0.10% |
| record | specialization | 1 | 0.10% |
| record | established_year | 1 | 0.10% |
| record | state_or_province | 1 | 0.10% |
| record | project_status | 1 | 0.10% |
| record | research_project | 1 | 0.10% |
| record | principal_investigator | 1 | 0.10% |
| record | asteroid.discovery.observatory.location.country | 1 | 0.10% |
| record | asteroid.discovery.observatory.location.state | 1 | 0.10% |
| record | asteroid.discovery.observatory.name | 1 | 0.10% |
| record | asteroid.discovery.observatory.location.county | 1 | 0.10% |
| record | status: Current transaction state | 1 | 0.10% |
| record | funding_grant | 1 | 0.10% |
| record | geographic_context | 1 | 0.10% |
| record | study_duration | 1 | 0.10% |
| record | intervention | 1 | 0.10% |
| record | match_code | 1 | 0.10% |
| record | asteroid.named_after.person.profession | 1 | 0.10% |
| record | unit_type | 1 | 0.10% |
| record | pet_policy | 1 | 0.10% |
| record | utilities | 1 | 0.10% |
| record | facility_name | 1 | 0.10% |
| record | facility_address | 1 | 0.10% |
| record | clinical_status | 1 | 0.10% |
| record | exhibitor_id | 1 | 0.10% |
| record | registration_date | 1 | 0.10% |
| record | review_id: Unique identifier for the review entry | 1 | 0.10% |
| record | author: Name of the person who wrote the review | 1 | 0.10% |
| record | book_title: Title of the book being reviewed | 1 | 0.10% |
| record | publisher: Publishing house responsible for the book | 1 | 0.10% |
| record | publication_date: Date the book was released | 1 | 0.10% |
| record | rating: Numerical score given to the book out of five | 1 | 0.10% |
| record | word_count: Total number of words in the book | 1 | 0.10% |
| record | genre: Literary category of the book such as fiction or poetry | 1 | 0.10% |
| record | note: Brief descriptive commentary on the book's content and quality | 1 | 0.10% |
| record | open_issues | 1 | 0.10% |
| record | external_resources | 1 | 0.10% |
| record | website_link | 1 | 0.10% |
| record | user_name: Name of the creator who posted the trend | 1 | 0.10% |
| record | location: City and state where the user is based | 1 | 0.10% |
| record | video_id: Unique identifier for the video content | 1 | 0.10% |
| record | comments: Total number of comments on the video | 1 | 0.10% |
| record | shares: Total number of shares the video received | 1 | 0.10% |
| record | tags: List of hashtags used in the video post | 1 | 0.10% |
| record | trend_duration: Number of days the trend remained active or trending | 1 | 0.10% |
| record | funding_id | 1 | 0.10% |
| record | colors | 1 | 0.10% |
| record | years_mentioned | 1 | 0.10% |
| record | defendant: Name of the individual or entity being sued | 1 | 0.10% |
| record | case_type: Category of legal matter such as contract or personal injury | 1 | 0.10% |
| record | assigned_judge: Name of the judge overseeing the case | 1 | 0.10% |
| record | amountindispute: Monetary value of the claim or settlement | 1 | 0.10% |
| record | owner: Name of the customer who submitted the ticket | 1 | 0.10% |
| record | priority: Severity level of the issue | 1 | 0.10% |
| record | resolution_estimate: Estimated hours to resolve the issue | 1 | 0.10% |
| record | impact_users: Number of users directly affected by the issue | 1 | 0.10% |
| record | genus.unique_features | 1 | 0.10% |
| record | genus.naming.profession | 1 | 0.10% |
| record | genus.location | 1 | 0.10% |
| record | genus.naming.year | 1 | 0.10% |
| record | genus.language | 1 | 0.10% |
| record | genus.species.0.reason | 1 | 0.10% |
| record | genus.type_species | 1 | 0.10% |
| record | genus.description | 1 | 0.10% |
| record | label | 1 | 0.10% |
| record | tissue_context | 1 | 0.10% |
| record | asteroid_designation | 1 | 0.10% |
| record | asteroid_type | 1 | 0.10% |
| record | discovery.discoverers | 1 | 0.10% |
| record | physical_characteristics.observations.0.telescope_type | 1 | 0.10% |
| record | physical_characteristics.observations.0.year_launched | 1 | 0.10% |
| record | name: Full name of the contact person | 1 | 0.10% |
| record | email: Email address of the contact | 1 | 0.10% |
| record | date: Date the outreach was made | 1 | 0.10% |
| record | title: Title of the proposed solution | 1 | 0.10% |
| record | price: Total price of the proposed solution | 1 | 0.10% |
| record | status: Current status of the outreach | 1 | 0.10% |
| record | ref_id: Unique reference identifier for the proposal | 1 | 0.10% |
| record | solution: Description of the proposed solution | 1 | 0.10% |
| record | target: Target audience or department for the proposal | 1 | 0.10% |
| record | admit_diagnosis | 1 | 0.10% |
| record | lengthofstay | 1 | 0.10% |
| record | department_head | 1 | 0.10% |
| record | primate.genus | 1 | 0.10% |
| record | primate.species | 1 | 0.10% |
| record | primate.name | 1 | 0.10% |
| record | primate.fossil_origin.epoch | 1 | 0.10% |
| record | primate.fossil_origin.location | 1 | 0.10% |
| record | primate.dental_formula | 1 | 0.10% |
| record | primate.weight_grams | 1 | 0.10% |
| record | primate.similarities | 1 | 0.10% |
| record | primate.classification | 1 | 0.10% |
| record | primate.related_primates.0.location | 1 | 0.10% |
| record | primate.related_primates.0.genus | 1 | 0.10% |
| record | primate.related_primates.0.species | 1 | 0.10% |
| record | primate.related_primates.0.temporal_context | 1 | 0.10% |
| record | primate.related_primates.0.name | 1 | 0.10% |
| record | primate.relationships | 1 | 0.10% |
| record | primate.morphological_analysis.0.genus | 1 | 0.10% |
| record | primate.morphological_analysis.0.type | 1 | 0.10% |
| record | primate.morphological_analysis.0.finding | 1 | 0.10% |
| record | primate.morphological_analysis.1.genus | 1 | 0.10% |
| record | primate.morphological_analysis.1.type | 1 | 0.10% |
| record | primate.morphological_analysis.1.finding | 1 | 0.10% |
| record | event_id: Unique identifier for each event | 1 | 0.10% |
| record | name: Title or name of the event | 1 | 0.10% |
| record | location: Physical address or venue of the event | 1 | 0.10% |
| record | administrative_level | 1 | 0.10% |
| record | location.province | 1 | 0.10% |
| record | location.county | 1 | 0.10% |
| record | establishment_approval | 1 | 0.10% |
| record | abolished_entity | 1 | 0.10% |
| record | administrative_divisions.community_committees | 1 | 0.10% |
| record | administrative_divisions.village_committees | 1 | 0.10% |
| record | user_email | 1 | 0.10% |
| record | entity | 1 | 0.10% |
| record | filed_by | 1 | 0.10% |
| record | post_id: Unique identifier for the Instagram post | 1 | 0.10% |
| record | price: Price of the product in USD | 1 | 0.10% |
| record | promo_code: Discount code offered with the product | 1 | 0.10% |
| record | shares: Total number of shares received on the post | 1 | 0.10% |
| record | lot_size | 1 | 0.10% |
| record | listing_id | 1 | 0.10% |
| record | kb_article | 1 | 0.10% |
| record | product.vendor | 1 | 0.10% |
| record | app_name | 1 | 0.10% |
| record | lease_term | 1 | 0.10% |
| record | listing_code | 1 | 0.10% |
| record | renewal_period | 1 | 0.10% |
| record | book_title | 1 | 0.10% |
| record | theme | 1 | 0.10% |
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
| record | appointment_type | 1 | 0.10% |
| record | launch_year | 1 | 0.10% |
| record | species_name | 1 | 0.10% |
| record | described_by.author | 1 | 0.10% |
| record | timestamp: Date and time of transaction occurrence | 1 | 0.10% |
| record | transaction_id: Unique identifier for the transaction | 1 | 0.10% |
| record | exchange_rate: Rate used to convert base to target currency | 1 | 0.10% |
| record | location: Geographic location of transaction | 1 | 0.10% |
| record | target_currency: Currency to which amount is converted | 1 | 0.10% |
| record | base_currency: Original currency used in transaction | 1 | 0.10% |
| record | source_provider: Provider handling the exchange rate calculation | 1 | 0.10% |
| record | donation_id: Unique identifier for each donation entry | 1 | 0.10% |
| record | date: Date the donation was made | 1 | 0.10% |
| record | status: Current status of the donation transaction | 1 | 0.10% |
| record | speaker.name | 1 | 0.10% |
| record | employee_name: Full name of the enrolled employee | 1 | 0.10% |
| record | duration: Duration of the training program in weeks | 1 | 0.10% |
| record | cost: Total cost paid for the training program | 1 | 0.10% |
| record | location: City where the training session is held | 1 | 0.10% |
| record | status: Current enrollment or completion status | 1 | 0.10% |
| record | article.id | 1 | 0.10% |
| record | article.products | 1 | 0.10% |
| record | article.symptoms | 1 | 0.10% |
| record | article.cause | 1 | 0.10% |
| record | article.resolution | 1 | 0.10% |
| record | article.references.0.id | 1 | 0.10% |
| record | article.references.0.url | 1 | 0.10% |
| record | article.references.0.description | 1 | 0.10% |
| record | article.last_reviewed | 1 | 0.10% |
| record | article.version | 1 | 0.10% |
| record | article.translation_note.disclaimer | 1 | 0.10% |
| record | article.translation_note.english_version.id | 1 | 0.10% |
| record | article.translation_note.english_version.url | 1 | 0.10% |
| record | article.disclaimer | 1 | 0.10% |
| record | performance_score | 1 | 0.10% |
| record | areasforimprovement | 1 | 0.10% |
| record | city: City where the office is physically located | 1 | 0.10% |
| record | address: Full street address including suite and city | 1 | 0.10% |
| record | compliance_status: Compliance classification with local regulations | 1 | 0.10% |
| record | staff_count: Number of employees currently working at the office | 1 | 0.10% |
| record | name: Official name of the business unit operating the office | 1 | 0.10% |
| record | office_code: Unique internal identifier for the office location | 1 | 0.10% |
| record | municipality.creation_date | 1 | 0.10% |
| record | municipality.population.year | 1 | 0.10% |
| record | municipality.merger_municipalities | 1 | 0.10% |
| record | municipality.former_province | 1 | 0.10% |
| record | home_team: Team playing at home stadium | 1 | 0.10% |
| record | away_team: Team playing away | 1 | 0.10% |
| record | player_id: Unique identifier of a player involved in action | 1 | 0.10% |
| record | player_action: Description of player's contribution such as goal or interception | 1 | 0.10% |
| record | jersey_number: Player's assigned jersey number on field | 1 | 0.10% |
| record | match_location: Stadium or venue where match occurred | 1 | 0.10% |
| record | person.political_party | 1 | 0.10% |
| record | passenger_city | 1 | 0.10% |
| record | expiry_date: Date when the device's warranty or use period ends | 1 | 0.10% |
| record | lead_person | 1 | 0.10% |
| record | payer | 1 | 0.10% |
| record | statistical_results.predictive_model.variables_included | 1 | 0.10% |
| record | population.age_range.unit | 1 | 0.10% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.1.t_score | 1 | 0.10% |
| record | statistical_results.diagnostic_metrics.sensitivity_at_threshold.t_score_threshold | 1 | 0.10% |
| record | movie.year | 1 | 0.10% |
| record | movie.based_on.title | 1 | 0.10% |
| record | review.publication | 1 | 0.10% |
| record | review.reviewer | 1 | 0.10% |
| record | movie.based_on.author | 1 | 0.10% |
| record | actor.name | 1 | 0.10% |
| record | actor.quote | 1 | 0.10% |
| record | review.comment | 1 | 0.10% |
| record | movement_reason | 1 | 0.10% |
| record | close_price | 1 | 0.10% |
| record | reporter: Name of the employee who submitted the report | 1 | 0.10% |
| record | transaction_amount: Amount of money transferred due to the phishing | 1 | 0.10% |
| record | distribution.regions.0.region_name | 1 | 0.10% |
| record | morphology.flower_heads | 1 | 0.10% |
| record | distribution.regions.2.countries | 1 | 0.10% |
| record | morphology.stem | 1 | 0.10% |
| record | distribution.regions.1.countries | 1 | 0.10% |
| record | distribution.regions.1.region_name | 1 | 0.10% |
| record | distribution.habitat | 1 | 0.10% |
| record | morphology.leaves | 1 | 0.10% |
| record | album_series | 1 | 0.10% |
| record | album_title | 1 | 0.10% |
| record | candidate | 1 | 0.10% |
| record | party | 1 | 0.10% |
| record | dateoftest: Date when the laboratory test was conducted | 1 | 0.10% |
| record | test_code: Code identifying the type of laboratory test | 1 | 0.10% |
| record | patient_id: Unique identifier for the patient in the system | 1 | 0.10% |
| record | place.location.country | 1 | 0.10% |
| record | place.location.county | 1 | 0.10% |
| record | place.type | 1 | 0.10% |
| record | person.title | 1 | 0.10% |
| record | issuing_body | 1 | 0.10% |
| record | costperpatient | 1 | 0.10% |
| record | coverage_rate | 1 | 0.10% |
| record | population_coverage | 1 | 0.10% |
| record | comorbidity_action | 1 | 0.10% |
| record | pathophysiology.etiology | 1 | 0.10% |
| record | pathophysiology.mechanism.secondary_event.dislocated_structure | 1 | 0.10% |
| record | study_details.design | 1 | 0.10% |
| record | pathophysiology.mechanism.secondary_event.description | 1 | 0.10% |
| record | outcomes_and_findings.prognosis.critical_success_factors | 1 | 0.10% |
| record | clinical_analysis_scope | 1 | 0.10% |
| record | date: Date when the event takes place | 1 | 0.10% |
| record | description: Brief summary of the event's purpose or activities | 1 | 0.10% |
| record | treatment_methods.selection_criteria | 1 | 0.10% |
| record | diseases_treated.exceptions | 1 | 0.10% |
| record | treatment_methods.description | 1 | 0.10% |
| record | disease_context | 1 | 0.10% |
| record | activation_type | 1 | 0.10% |
| record | consignment_id: Unique identifier for each shipment | 1 | 0.10% |
| record | destination: Final address of the shipment | 1 | 0.10% |
| record | disease.name | 1 | 0.10% |
| record | disease.pathogenesis.symptoms | 1 | 0.10% |
| record | disease.pathogenesis.affected_organs | 1 | 0.10% |
| record | disease.pathogenesis.risk_factors | 1 | 0.10% |
| record | disease.subspecies.2.affected_species | 1 | 0.10% |
| record | disease.subspecies.2.associated_diseases | 1 | 0.10% |
| record | price: Selling price of the product in dollars | 1 | 0.10% |
| record | sales_volume: Total units sold over time | 1 | 0.10% |
| record | name: Name of the product as listed | 1 | 0.10% |
| record | product_id: Unique identifier for each product | 1 | 0.10% |
| record | status: Current product availability status | 1 | 0.10% |
| record | respiratory_rate | 1 | 0.10% |
| record | merchant: Name of the merchant or store involved | 1 | 0.10% |
| record | product: Description of the product or service purchased | 1 | 0.10% |
| record | day_of_year | 1 | 0.10% |
| record | co_authors | 1 | 0.10% |
| record | table_of_contents | 1 | 0.10% |
| record | asteroid_belt | 1 | 0.10% |
| record | discovery_details.observatory.name | 1 | 0.10% |
| record | discovery_details.observatory.location.county | 1 | 0.10% |
| record | discovery_details.observatory.location.state | 1 | 0.10% |
| record | discovery_details.observatory.location.country | 1 | 0.10% |
| record | discovery_details.discoverer | 1 | 0.10% |
| record | orbital_characteristics.jpl_classification | 1 | 0.10% |
| record | physical_characteristics.observations.0.publication_details | 1 | 0.10% |
| record | physical_characteristics.observations.0.telescope.location | 1 | 0.10% |
| record | physical_characteristics.observations.0.telescope.year_launched | 1 | 0.10% |
| record | headquarters | 1 | 0.10% |
| record | entities_involved | 1 | 0.10% |
| record | record_metadata | natural/location ×20, natural/title ×19, natural/status ×19, natural/name ×19 |  |
| research_paper | authors | 3 | 100.00% |
| research_paper | venue | 3 | 100.00% |
| research_paper | year | 3 | 100.00% |
| research_paper | title | 2 | 66.67% |
| research_paper | topic | 1 | 33.33% |
| research_paper | record_metadata | natural/title ×2, natural/authors ×1 |  |
| software_release | product | 9 | 100.00% |
| software_release | release_date | 7 | 77.78% |
| software_release | version | 6 | 66.67% |
| software_release | platform | 5 | 55.56% |
| software_release | record_metadata | natural/product ×9 |  |
| transaction | buyer | 97 | 97.98% |
| transaction | amount | 95 | 95.96% |
| transaction | item | 94 | 94.95% |
| transaction | date | 92 | 92.93% |
| transaction | seller | 90 | 90.91% |
| transaction | record_metadata | natural/item ×94, natural/amount ×5 |  |
| vehicle_listing | make | 6 | 100.00% |
| vehicle_listing | model | 6 | 100.00% |
| vehicle_listing | year | 6 | 100.00% |
| vehicle_listing | price | 5 | 83.33% |
| vehicle_listing | mileage | 3 | 50.00% |
| vehicle_listing | record_metadata | natural/make ×2 |  |

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | mixed (replay) |
| relations | mixed (replay) |
| events | mixed (replay) |
| classifications | mixed (replay) |
| structures | mixed (replay) |
