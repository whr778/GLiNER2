# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replay_eb21_coref30 | train | 1,037 | 1,036 | 6,968 | 487 | 876 | 58 | 590 | 754 |
| replay_eb21_coref30 | val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| replay_eb21_coref30 | test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## replay_eb21_coref30

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1,037 | 1,036 | 303 | 1154 | 8212 | 6,968 | 487 | 876 | 58 | 590 | 754 |
| val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 527 | 0 | 0 | 527 | 50.82% |
| zho | 381 | 0 | 0 | 381 | 36.74% |
| tur | 104 | 0 | 0 | 104 | 10.03% |
| und | 19 | 0 | 0 | 19 | 1.83% |
| spa | 2 | 0 | 0 | 2 | 0.19% |
| ita | 1 | 0 | 0 | 1 | 0.10% |
| fra | 1 | 0 | 0 | 1 | 0.10% |
| ara | 1 | 0 | 0 | 1 | 0.10% |
| por | 1 | 0 | 0 | 1 | 0.10% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 0 |
| test_unique_inputs | 0 |
| train_records | 1,037 |
| train_unique_inputs | 1,036 |
| val_records | 0 |
| val_unique_inputs | 0 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | n/a | 6,968 / 0 |  |
| entities (per mention) | train / test | n/a | 6,968 / 0 |  |
| relations (per instance) | train / val | n/a | 487 / 0 |  |
| relations (per instance) | train / test | n/a | 487 / 0 |  |
| event types (per trigger mention) | train / val | n/a | 696 / 0 |  |
| event types (per trigger mention) | train / test | n/a | 696 / 0 |  |
| event roles (per argument) | train / val | n/a | 774 / 0 |  |
| event roles (per argument) | train / test | n/a | 774 / 0 |  |
| classifications (true_label) | train / val | n/a | 876 / 0 |  |
| classifications (true_label) | train / test | n/a | 876 / 0 |  |


### Events

**train.** 677 instances in 181 records, 0 triggerless. Trigger gold 590. Argument gold 754. Arguments per instance, mean 1.14. Trigger length in whitespace tokens, mean 1.31, single-token share 73.4%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Contact.Communicate | 39 | 5.60% |
| Catastrophe | 31 | 4.45% |
| Experiment | 28 | 4.02% |
| Support | 26 | 3.74% |
| Killing | 20 | 2.87% |
| Attack | 17 | 2.44% |
| Causation | 15 | 2.16% |
| Statement | 14 | 2.01% |
| Sport.Compete | 13 | 1.87% |
| Motion | 12 | 1.72% |
| Action | 11 | 1.58% |
| Cyber.Databreach | 10 | 1.44% |
| Process_start | 10 | 1.44% |
| Damaging | 10 | 1.44% |
| Scrutiny | 10 | 1.44% |
| Contact.Meet | 9 | 1.29% |
| Reporting | 9 | 1.29% |
| Come_together | 8 | 1.15% |
| match_result | 8 | 1.15% |
| Cyber.Ransom | 8 | 1.15% |
| Life.Die | 7 | 1.01% |
| Manufacturing | 7 | 1.01% |
| Death | 7 | 1.01% |
| Arriving | 7 | 1.01% |
| Bodily_harm | 7 | 1.01% |
| Publishing | 7 | 1.01% |
| Know | 6 | 0.86% |
| Placing | 6 | 0.86% |
| judicial_action_arrest | 6 | 0.86% |
| Business.StartOrg | 6 | 0.86% |
| Disaster.Accident | 6 | 0.86% |
| Self_motion | 6 | 0.86% |
| product_publish | 6 | 0.86% |
| Life.Injure | 6 | 0.86% |
| Conflict.Attack | 6 | 0.86% |
| Deploy | 6 | 0.86% |
| Traveling | 6 | 0.86% |
| Choosing | 5 | 0.72% |
| Hold | 5 | 0.72% |
| Supporting | 5 | 0.72% |
| Justice.Arrest | 5 | 0.72% |
| Judgment_communication | 5 | 0.72% |
| layoff | 4 | 0.57% |
| Creating | 4 | 0.57% |
| Perception_active | 4 | 0.57% |
| Name_conferral | 4 | 0.57% |
| Motion_directional | 4 | 0.57% |
| Personnel.StartPosition | 4 | 0.57% |
| Conflict | 4 | 0.57% |
| Exhibit | 4 | 0.57% |
| Cause_change_of_position_on_a_scale | 4 | 0.57% |
| Expressing_publicly | 4 | 0.57% |
| championship_win | 3 | 0.43% |
| Accident | 3 | 0.43% |
| resignation | 3 | 0.43% |
| Having_or_lacking_access | 3 | 0.43% |
| Participation | 3 | 0.43% |
| Cause_change_of_strength | 3 | 0.43% |
| Destroying | 3 | 0.43% |
| Transaction.TransferOwnership | 3 | 0.43% |
| Cause_to_be_included | 3 | 0.43% |
| Conflict.Demonstrate | 3 | 0.43% |
| Business.LaunchProduct | 3 | 0.43% |
| Becoming | 3 | 0.43% |
| Getting | 3 | 0.43% |
| departure | 3 | 0.43% |
| Life.Die.Unspecified | 3 | 0.43% |
| Collapse | 3 | 0.43% |
| Use_firearm | 3 | 0.43% |
| Manoeuvre | 3 | 0.43% |
| judicial_action_prosecution | 2 | 0.29% |
| Injure | 2 | 0.29% |
| Preventing_or_letting | 2 | 0.29% |
| Deciding | 2 | 0.29% |
| Terrorism | 2 | 0.29% |
| Filling | 2 | 0.29% |
| Labeling | 2 | 0.29% |
| Cause_to_amalgamate | 2 | 0.29% |
| Ingestion | 2 | 0.29% |
| life.injure.illnessdegradationhungerthirst | 2 | 0.29% |
| Coming_to_be | 2 | 0.29% |
| contact.discussion.meet | 2 | 0.29% |
| Hostile_encounter | 2 | 0.29% |
| Conquering | 2 | 0.29% |
| product_recall | 2 | 0.29% |
| Justice.ChargeIndict | 2 | 0.29% |
| movement.transportartifact.n/a | 2 | 0.29% |
| Government.Vote | 2 | 0.29% |
| Response | 2 | 0.29% |
| Movement.Arrive | 2 | 0.29% |
| Openness | 2 | 0.29% |
| contact.requestadvise.correspondence | 2 | 0.29% |
| advancement | 2 | 0.29% |
| death | 2 | 0.29% |
| movement.transportperson.preventexit | 2 | 0.29% |
| Body_movement | 2 | 0.29% |
| Bringing | 2 | 0.29% |
| Coming_to_believe | 2 | 0.29% |
| Cost | 2 | 0.29% |
| Defending | 2 | 0.29% |
| Using | 2 | 0.29% |
| Removing | 2 | 0.29% |
| Criminal_investigation | 2 | 0.29% |
| Social_event | 2 | 0.29% |
| Request | 2 | 0.29% |
| Hiding_objects | 2 | 0.29% |
| Cyber.DiscoverVulnerability | 2 | 0.29% |
| government.formation.startgpe | 1 | 0.14% |
| Medical.Diagnose | 1 | 0.14% |
| birth | 1 | 0.14% |
| Supply | 1 | 0.14% |
| Resolve_problem | 1 | 0.14% |
| Contact.Negotiate | 1 | 0.14% |
| retirement | 1 | 0.14% |
| joined | 1 | 0.14% |
| Business.DeclareBankruptcy | 1 | 0.14% |
| Research.Fund | 1 | 0.14% |
| Create_artwork | 1 | 0.14% |
| Change_sentiment | 1 | 0.14% |
| Building | 1 | 0.14% |
| Education_teaching | 1 | 0.14% |
| conflict.attack.selfdirectedbattle | 1 | 0.14% |
| Business.Strike | 1 | 0.14% |
| Transaction.TransferMoney | 1 | 0.14% |
| Life.Divorce | 1 | 0.14% |
| Personnel.Elect | 1 | 0.14% |
| Besieging | 1 | 0.14% |
| Releasing | 1 | 0.14% |
| Control | 1 | 0.14% |
| life.injure.n/a | 1 | 0.14% |
| judicial_action_fine | 1 | 0.14% |
| justice.initiatejudicialprocess.chargeindict | 1 | 0.14% |
| contact.requestadvise.broadcast | 1 | 0.14% |
| Award.ReceiveAward | 1 | 0.14% |
| government.vote.violationspreventvote | 1 | 0.14% |
| Personnel.Nominate | 1 | 0.14% |
| Justice.Sentence | 1 | 0.14% |
| Justice.Appeal | 1 | 0.14% |
| Justice.TrialHearing | 1 | 0.14% |
| visit_on_set | 1 | 0.14% |
| Movement.Transport | 1 | 0.14% |
| Connect | 1 | 0.14% |
| Change | 1 | 0.14% |
| Quarreling | 1 | 0.14% |
| Legal_rulings | 1 | 0.14% |
| artifactexistence.damagedestroy.destroy | 1 | 0.14% |
| ipo | 1 | 0.14% |
| Disaster.Rescue | 1 | 0.14% |
| Temporary_stay | 1 | 0.14% |
| government.formation.mergegpe | 1 | 0.14% |
| conflict.yield.n/a | 1 | 0.14% |
| Life.BeBorn | 1 | 0.14% |
| transaction.transaction.transfercontrol | 1 | 0.14% |
| financing | 1 | 0.14% |
| judicial_action_filing | 1 | 0.14% |
| Personnel.EndPosition | 1 | 0.14% |
| price_limit_up | 1 | 0.14% |
| conflict.attack.n/a | 1 | 0.14% |
| contact.threatencoerce.n/a | 1 | 0.14% |
| transaction.transfermoney.giftgrantprovideaid | 1 | 0.14% |
| transaction.transaction.giftgrantprovideaid | 1 | 0.14% |
| transaction.transferownership.giftgrantprovideaid | 1 | 0.14% |
| contact.commitmentpromiseexpressintent.n/a | 1 | 0.14% |
| Conflict.Attack.Unspecified | 1 | 0.14% |
| Escaping | 1 | 0.14% |
| Assistance | 1 | 0.14% |
| Recovering | 1 | 0.14% |
| Emptying | 1 | 0.14% |
| Telling | 1 | 0.14% |
| Departing | 1 | 0.14% |
| judicial_action_summon | 1 | 0.14% |
| price_increase | 1 | 0.14% |
| Arranging | 1 | 0.14% |
| Check | 1 | 0.14% |
| GetReady | 1 | 0.14% |
| Aiming | 1 | 0.14% |
| Writing | 1 | 0.14% |
| Warning | 1 | 0.14% |
| Receiving | 1 | 0.14% |
| Hindering | 1 | 0.14% |
| Committing_crime | 1 | 0.14% |
| Commitment | 1 | 0.14% |
| Military_operation | 1 | 0.14% |
| Recording | 1 | 0.14% |
| strike | 1 | 0.14% |
| movement.transportperson.hide | 1 | 0.14% |
| marriage | 1 | 0.14% |
| Dispersal | 1 | 0.14% |
| Confronting_problem | 1 | 0.14% |
| Earnings_and_losses | 1 | 0.14% |
| Agree_or_refuse_to_act | 1 | 0.14% |
| personnel.endposition.quitretire | 1 | 0.14% |
| justice.judicialconsequences.convict | 1 | 0.14% |
| Cyber.Phishing | 1 | 0.14% |

**val.** No event instances.

**test.** No event instances.

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 774 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2.04 | 34.24% | 1.92 | 32.34% | 0.14% | 0 | 774 | 0 | 696 |
| val | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |
| test | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 677 | 1.77% | 665 | 6.02% | 5.41% | 0.45% | 0.00% | 12.15% |
| val | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |
| test | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

| split | docs | clusters | mentions | multi-mention | singletons | mention pairs | pronoun mentions | pronoun share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1 | 9 | 36 | 9 | 0 | 72 | 7 | 19.44% |
| val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |
| test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 12 | 27 | 675 |
| val | 0 | 0 | 0 |
| test | 0 | 0 | 0 |

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

| split | mentions the string rule adds | arguments skipped, 2+ clusters | arguments in one multi-mention cluster | other mentions of that cluster | argument spans in 2+ clusters |
| --- | --- | --- | --- | --- | --- |
| train | 4 | 0 | 0 | 0 | 0 |
| val | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 |

### Entities

#### train

| label | count | share |
| --- | --- | --- |
| person | 672 | 9.64% |
| location | 335 | 4.81% |
| Location | 316 | 4.54% |
| date | 289 | 4.15% |
| Date | 280 | 4.02% |
| organization | 193 | 2.77% |
| Organization | 183 | 2.63% |
| cardinal | 172 | 2.47% |
| Product | 120 | 1.72% |
| job title | 120 | 1.72% |
| EquityHolder | 118 | 1.69% |
| StartDate | 116 | 1.66% |
| money | 111 | 1.59% |
| StockCode | 108 | 1.55% |
| CompanyName | 98 | 1.41% |
| StockAbbr | 96 | 1.38% |
| Casualties and Losses | 96 | 1.38% |
| geopolitical entity | 96 | 1.38% |
| company | 92 | 1.32% |
| sports team | 90 | 1.29% |
| Person | 80 | 1.15% |
| PledgedShares | 73 | 1.05% |
| media outlet | 72 | 1.03% |
| Cause | 72 | 1.03% |
| educational institution | 66 | 0.95% |
| TotalHoldingShares | 64 | 0.92% |
| TradedShares | 63 | 0.90% |
| TotalHoldingRatio | 60 | 0.86% |
| duration | 58 | 0.83% |
| age | 58 | 0.83% |
| Pledgee | 55 | 0.79% |
| Reduction Amount | 52 | 0.75% |
| facility | 50 | 0.72% |
| occupation | 46 | 0.66% |
| Quantity | 42 | 0.60% |
| Measurement | 41 | 0.59% |
| EndDate | 39 | 0.56% |
| ReleasedDate | 37 | 0.53% |
| Reducing Shareholder | 37 | 0.53% |
| percent | 37 | 0.53% |
| time | 37 | 0.53% |
| TotalPledgedShares | 35 | 0.50% |
| product | 32 | 0.46% |
| Affected Areas | 32 | 0.46% |
| Reduction Start Date | 31 | 0.44% |
| government agency | 31 | 0.44% |
| Protesters | 30 | 0.43% |
| Money | 29 | 0.42% |
| Company Name | 28 | 0.40% |
| Protest Reason | 28 | 0.40% |
| Announcement Date | 27 | 0.39% |
| Candidates and their Political Parties | 27 | 0.39% |
| quantity | 26 | 0.37% |
| event | 24 | 0.34% |
| Event | 23 | 0.33% |
| brand | 22 | 0.32% |
| LaterHoldingShares | 22 | 0.32% |
| Method | 21 | 0.30% |
| Contest Participant | 21 | 0.30% |
| address | 20 | 0.29% |
| Policy Content | 20 | 0.29% |
| team role | 19 | 0.27% |
| work of art | 19 | 0.27% |
| AveragePrice | 19 | 0.27% |
| Time | 18 | 0.26% |
| crime | 18 | 0.26% |
| pledgor | 18 | 0.26% |
| Storm Hit Location | 18 | 0.26% |
| Accident Investigator | 17 | 0.24% |
| Target | 17 | 0.24% |
| Facility | 17 | 0.24% |
| Individual_protein | 15 | 0.22% |
| region | 14 | 0.20% |
| Recipient | 14 | 0.20% |
| pledge_amount | 14 | 0.20% |
| Attacker | 14 | 0.20% |
| HighestTradingPrice | 14 | 0.20% |
| Crew | 14 | 0.20% |
| PER | 14 | 0.20% |
| GPE | 13 | 0.19% |
| landmark | 13 | 0.19% |
| Nationality | 12 | 0.17% |
| Accepting Court | 12 | 0.17% |
| family relationship | 12 | 0.17% |
| RepurchasedShares | 12 | 0.17% |
| song | 12 | 0.17% |
| Winner | 12 | 0.17% |
| Host | 12 | 0.17% |
| nationality | 11 | 0.16% |
| Increasing Shareholder | 10 | 0.14% |
| body of water | 10 | 0.14% |
| measurement | 10 | 0.14% |
| sports competition | 10 | 0.14% |
| Deliberating Agency | 10 | 0.14% |
| Participants | 10 | 0.14% |
| Occupation | 10 | 0.14% |
| RepurchaseAmount | 10 | 0.14% |
| Election Name | 10 | 0.14% |
| Position | 10 | 0.14% |
| Strike Reason | 10 | 0.14% |
| Strike Outcome | 10 | 0.14% |
| Scheduled Landing Place | 9 | 0.13% |
| Increase Amount | 9 | 0.13% |
| ruling_date | 9 | 0.13% |
| software | 9 | 0.13% |
| military rank | 9 | 0.13% |
| Economic Loss | 9 | 0.13% |
| academic degree | 9 | 0.13% |
| Loss Amount | 9 | 0.13% |
| road | 9 | 0.13% |
| ORG | 9 | 0.13% |
| Affected Area | 8 | 0.11% |
| disease | 8 | 0.11% |
| ClosingDate | 8 | 0.11% |
| Astronauts | 8 | 0.11% |
| law | 8 | 0.11% |
| Speaker Status | 8 | 0.11% |
| Suspect | 8 | 0.11% |
| Summit Theme | 8 | 0.11% |
| Survivors | 8 | 0.11% |
| DiseaseOrPhenotypicFeature | 8 | 0.11% |
| Influence People | 8 | 0.11% |
| Strike Agency | 8 | 0.11% |
| Military Unit | 8 | 0.11% |
| Flight No. | 7 | 0.10% |
| military unit | 7 | 0.10% |
| temperature | 7 | 0.10% |
| Spacecraft | 7 | 0.10% |
| Game Name | 7 | 0.10% |
| Policy Name & Abbreviation | 7 | 0.10% |
| ethnic group | 7 | 0.10% |
| Storm Warning Level | 7 | 0.10% |
| Poisoning Type | 7 | 0.10% |
| Charged Crime | 7 | 0.10% |
| Sponsor | 7 | 0.10% |
| Appointee | 7 | 0.10% |
| Microorganism | 7 | 0.10% |
| Contracting Parties | 7 | 0.10% |
| Anti-pollution People/Organizations | 7 | 0.10% |
| Participating Countries | 7 | 0.10% |
| Aircraft Agency | 6 | 0.09% |
| distance | 6 | 0.09% |
| ordinal | 6 | 0.09% |
| Research Agency | 6 | 0.09% |
| political affiliation | 6 | 0.09% |
| The Charged Crime | 6 | 0.09% |
| currency | 6 | 0.09% |
| vehicle | 6 | 0.09% |
| Frozen Amount | 6 | 0.09% |
| pledge_start_date | 6 | 0.09% |
| Protest Slogan | 6 | 0.09% |
| Aid Agency | 6 | 0.09% |
| Achievement | 6 | 0.09% |
| GeneOrGeneProduct | 6 | 0.09% |
| Champions | 6 | 0.09% |
| Score | 6 | 0.09% |
| Appointer | 6 | 0.09% |
| Precautionary Measure | 6 | 0.09% |
| religious group | 5 | 0.07% |
| System | 5 | 0.07% |
| Start Time | 5 | 0.07% |
| Competition Items | 5 | 0.07% |
| Host Country | 5 | 0.07% |
| Speaker | 5 | 0.07% |
| regulation | 5 | 0.07% |
| Arrest Location | 5 | 0.07% |
| era | 5 | 0.07% |
| Award | 5 | 0.07% |
| frozen_shareholder | 5 | 0.07% |
| Weapon | 5 | 0.07% |
| Economic loss | 5 | 0.07% |
| Visitor | 5 | 0.07% |
| Released People | 5 | 0.07% |
| Criminal Evidence | 5 | 0.07% |
| Habitat | 5 | 0.07% |
| Agreement Content | 5 | 0.07% |
| Area Affected | 5 | 0.07% |
| Boycotted Institutions | 5 | 0.07% |
| Start Date | 5 | 0.07% |
| Pollution Source | 5 | 0.07% |
| Areas affected | 5 | 0.07% |
| Magnitude | 4 | 0.06% |
| Start_Date | 4 | 0.06% |
| medical procedure | 4 | 0.06% |
| protein | 4 | 0.06% |
| pseudonym | 4 | 0.06% |
| court case | 4 | 0.06% |
| Task | 4 | 0.06% |
| PII | 4 | 0.06% |
| LowestTradingPrice | 4 | 0.06% |
| Launch Site | 4 | 0.06% |
| non-profit organization | 4 | 0.06% |
| Missings | 4 | 0.06% |
| Belligerents | 4 | 0.06% |
| benefits | 4 | 0.06% |
| medication | 4 | 0.06% |
| Shipwreck Reason | 4 | 0.06% |
| Location/Hospital | 4 | 0.06% |
| Transportation | 4 | 0.06% |
| award | 4 | 0.06% |
| Prosecutor | 4 | 0.06% |
| Voting method | 4 | 0.06% |
| Country Visited | 4 | 0.06% |
| Release Reason | 4 | 0.06% |
| Accusation | 4 | 0.06% |
| People | 4 | 0.06% |
| Damaged Facility | 4 | 0.06% |
| Last Job of the Appointee | 4 | 0.06% |
| Employment Agency | 4 | 0.06% |
| Strikers Status | 4 | 0.06% |
| Party Responsible for Pollution | 4 | 0.06% |
| WEA | 4 | 0.06% |
| Disease | 4 | 0.06% |
| Protein_family_or_group | 3 | 0.04% |
| Gene | 3 | 0.04% |
| Increase Start Date | 3 | 0.04% |
| Launch Country | 3 | 0.04% |
| Spacecraft Mission | 3 | 0.04% |
| Mission Duration | 3 | 0.04% |
| book | 3 | 0.04% |
| mountain | 3 | 0.04% |
| chemical | 3 | 0.04% |
| Police | 3 | 0.04% |
| dosage | 3 | 0.04% |
| publication | 3 | 0.04% |
| Award Field | 3 | 0.04% |
| Water Level | 3 | 0.04% |
| Freeze Start Date | 3 | 0.04% |
| honorific | 3 | 0.04% |
| Storm Name | 3 | 0.04% |
| description | 3 | 0.04% |
| FrozeShares | 3 | 0.04% |
| Ship No. | 3 | 0.04% |
| Rescue Organizer | 3 | 0.04% |
| Number of Damaged Houses | 3 | 0.04% |
| Fine Reason | 3 | 0.04% |
| Weapon and Equippment | 3 | 0.04% |
| Evidence | 3 | 0.04% |
| ChemicalEntity | 3 | 0.04% |
| OrganismTaxon | 3 | 0.04% |
| LegalInstitution | 3 | 0.04% |
| Maximum Wind Speed | 3 | 0.04% |
| pledge_end_date | 3 | 0.04% |
| Accused People | 3 | 0.04% |
| Funding | 3 | 0.04% |
| Related Rivers or Lakes | 3 | 0.04% |
| Forecasting Agency | 3 | 0.04% |
| Tsunami Warning Level | 3 | 0.04% |
| Taking-off Place | 3 | 0.04% |
| Storm Formation Location | 3 | 0.04% |
| Storm Movement Speed | 3 | 0.04% |
| Goal | 3 | 0.04% |
| Turnout(who win who lost) | 3 | 0.04% |
| Susceptible Population | 3 | 0.04% |
| Confirmed/Infected Cases | 3 | 0.04% |
| frequency | 2 | 0.03% |
| language | 2 | 0.03% |
| Generic | 2 | 0.03% |
| Malware | 2 | 0.03% |
| Data | 2 | 0.03% |
| Number | 2 | 0.03% |
| weight | 2 | 0.03% |
| operating system | 2 | 0.03% |
| Cooperative Agency | 2 | 0.03% |
| trade union | 2 | 0.03% |
| weapon | 2 | 0.03% |
| fiscal quarter | 2 | 0.03% |
| sentence | 2 | 0.03% |
| aircraft | 2 | 0.03% |
| Launch Date | 2 | 0.03% |
| Carrier Rocket | 2 | 0.03% |
| war | 2 | 0.03% |
| treaty | 2 | 0.03% |
| Record Breaker | 2 | 0.03% |
| Record-breaking Project | 2 | 0.03% |
| Grades | 2 | 0.03% |
| DNA | 2 | 0.03% |
| political party | 2 | 0.03% |
| People/Organization who predicted the disaster | 2 | 0.03% |
| Number of Vehicles Involved in the Crash | 2 | 0.03% |
| city | 2 | 0.03% |
| state | 2 | 0.03% |
| zipCode | 2 | 0.03% |
| Summit Name | 2 | 0.03% |
| Outbreak Date | 2 | 0.03% |
| Deceased | 2 | 0.03% |
| Profession | 2 | 0.03% |
| Pests | 2 | 0.03% |
| Response Measures | 2 | 0.03% |
| Perpetrators | 2 | 0.03% |
| Weapon Used | 2 | 0.03% |
| Stolen Amount | 2 | 0.03% |
| regulatory body | 2 | 0.03% |
| Publication | 2 | 0.03% |
| Head of Institution | 2 | 0.03% |
| Investigating agency | 2 | 0.03% |
| Bank Name | 2 | 0.03% |
| Arrest Time | 2 | 0.03% |
| itemId | 2 | 0.03% |
| unitPrice | 2 | 0.03% |
| Storm Center Location | 2 | 0.03% |
| Sentencing Location | 2 | 0.03% |
| Rescue Tool or Method | 2 | 0.03% |
| Court | 2 | 0.03% |
| Judgement Result/Prison Term | 2 | 0.03% |
| Age of the Appointee | 2 | 0.03% |
| Maximum Rainfall | 2 | 0.03% |
| Types of the Spectacle | 2 | 0.03% |
| Bill Drafting Agency | 2 | 0.03% |
| Illness | 2 | 0.03% |
| Tsunamis | 2 | 0.03% |
| Magnitude(Tsunami heights) | 2 | 0.03% |
| Damaged Property | 2 | 0.03% |
| Resignated Person | 2 | 0.03% |
| Areas Affected | 2 | 0.03% |
| Damaged Crops & Livestock | 2 | 0.03% |
| Train Agency | 2 | 0.03% |
| Train No. | 2 | 0.03% |
| Conflict Duration | 2 | 0.03% |
| Number of Rescued People | 2 | 0.03% |
| Symptom | 2 | 0.03% |
| Amount of Precipitation | 2 | 0.03% |
| Strikers | 2 | 0.03% |
| Alternate Landing Place | 2 | 0.03% |
| Military Exercise | 2 | 0.03% |
| Army | 2 | 0.03% |
| Scale | 2 | 0.03% |
| Predecessor | 2 | 0.03% |
| Appointment Time | 2 | 0.03% |
| FAC | 2 | 0.03% |
| Way for Spreading | 2 | 0.03% |
| Unemployed Rate | 2 | 0.03% |
| Time of the Case | 2 | 0.03% |
| Economists who Predicted the Crisis | 2 | 0.03% |
| Outbreak Location | 2 | 0.03% |
| Spokeswoman/Spokesman | 2 | 0.03% |
| Government Reaction | 2 | 0.03% |
| Aid Supplies/Amount | 2 | 0.03% |
| Capabilities | 2 | 0.03% |
| Gene/protein/RNA | 1 | 0.01% |
| Casualty Count | 1 | 0.01% |
| airport | 1 | 0.01% |
| OtherScientificTerm | 1 | 0.01% |
| pathogen | 1 | 0.01% |
| vessel | 1 | 0.01% |
| deadline | 1 | 0.01% |
| Lasting Time | 1 | 0.01% |
| Ways to Watch the Speech | 1 | 0.01% |
| witness | 1 | 0.01% |
| Launch Result | 1 | 0.01% |
| holiday | 1 | 0.01% |
| algorithm | 1 | 0.01% |
| symptom | 1 | 0.01% |
| speed | 1 | 0.01% |
| Effective Date | 1 | 0.01% |
| Invalid Date | 1 | 0.01% |
| Previous Record Holder | 1 | 0.01% |
| The Grades of the Previous Record Holder | 1 | 0.01% |
| Arrested | 1 | 0.01% |
| Registered Capital | 1 | 0.01% |
| postal code | 1 | 0.01% |
| Storm Direction | 1 | 0.01% |
| Riot Reason | 1 | 0.01% |
| area | 1 | 0.01% |
| Attending Hospital | 1 | 0.01% |
| Countries Joined the Organization | 1 | 0.01% |
| Organization Name | 1 | 0.01% |
| firstName | 1 | 0.01% |
| lastName | 1 | 0.01% |
| phoneNumber | 1 | 0.01% |
| emailAddress | 1 | 0.01% |
| name | 1 | 0.01% |
| phone | 1 | 0.01% |
| website | 1 | 0.01% |
| title | 1 | 0.01% |
| equityOffered | 1 | 0.01% |
| film | 1 | 0.01% |
| Last Outbreak Time | 1 | 0.01% |
| The State of the Volcano (Dormant or Active) | 1 | 0.01% |
| Hull Discovery Time | 1 | 0.01% |
| Death Reason | 1 | 0.01% |
| Age | 1 | 0.01% |
| financial instrument | 1 | 0.01% |
| Fined Agency | 1 | 0.01% |
| Penalty Amount | 1 | 0.01% |
| Regulatory Authority | 1 | 0.01% |
| Prosecution Lawyer | 1 | 0.01% |
| PaymentMethod | 1 | 0.01% |
| Recovered Amount | 1 | 0.01% |
| Storm Hit Time | 1 | 0.01% |
| Region | 1 | 0.01% |
| quotationId | 1 | 0.01% |
| supplierId | 1 | 0.01% |
| supplierName | 1 | 0.01% |
| streetAddress | 1 | 0.01% |
| quotationDate | 1 | 0.01% |
| expirationDate | 1 | 0.01% |
| totalCost | 1 | 0.01% |
| termsAndConditions | 1 | 0.01% |
| Defense Lawyer | 1 | 0.01% |
| Investigative Agency | 1 | 0.01% |
| Jail Time | 1 | 0.01% |
| Beneficiary | 1 | 0.01% |
| Aid Reason | 1 | 0.01% |
| Judge | 1 | 0.01% |
| Detention Start Time | 1 | 0.01% |
| compensation_recipient | 1 | 0.01% |
| Organization_Military | 1 | 0.01% |
| Spectacle Location | 1 | 0.01% |
| Spectacle Start Time | 1 | 0.01% |
| Spectacle End Time | 1 | 0.01% |
| Warning Device | 1 | 0.01% |
| Influenced Crops and Livelihood | 1 | 0.01% |
| Resign Reason | 1 | 0.01% |
| Age of the Resignated Person | 1 | 0.01% |
| Influenced People | 1 | 0.01% |
| News Release Agency | 1 | 0.01% |
| Other Impact | 1 | 0.01% |
| End Time | 1 | 0.01% |
| The Final Seats Result | 1 | 0.01% |
| Duration | 1 | 0.01% |
| protein-domain | 1 | 0.01% |
| Award Reason | 1 | 0.01% |
| Employed Institution | 1 | 0.01% |
| Agreement Name | 1 | 0.01% |
| Passengers | 1 | 0.01% |
| LOC | 1 | 0.01% |
| Freeze End Date | 1 | 0.01% |
| UnfrozeDate | 1 | 0.01% |
| Death Time | 1 | 0.01% |
| senior_personnel | 1 | 0.01% |
| senior_position | 1 | 0.01% |
| Symptoms | 1 | 0.01% |
| Spokenmen | 1 | 0.01% |
| Group | 1 | 0.01% |
| Country | 1 | 0.01% |
| Provisional Government | 1 | 0.01% |
| Affected Industries | 1 | 0.01% |
| Bankrupt Business | 1 | 0.01% |
| Policy Proposals | 1 | 0.01% |
| Death Cases | 1 | 0.01% |
| Epidemic Data Issuing Agency | 1 | 0.01% |
| Number of People Hospitalized | 1 | 0.01% |
| End Date | 1 | 0.01% |
| Service Years | 1 | 0.01% |
| Victim | 1 | 0.01% |
| Temporary Settlement | 1 | 0.01% |
| Election Goal | 1 | 0.01% |
| Number of Evacuated People | 1 | 0.01% |
| Disaster-Stricken Farmland | 1 | 0.01% |
| Vulnerability | 1 | 0.01% |
| File | 1 | 0.01% |
| Purpose | 1 | 0.01% |
| tax | 0 | 0.00% |
| credit rating | 0 | 0.00% |
| contract | 0 | 0.00% |
| programming language | 0 | 0.00% |
| gene | 0 | 0.00% |
| organism | 0 | 0.00% |
| allergen | 0 | 0.00% |
| verdict | 0 | 0.00% |
| model number | 0 | 0.00% |
| license | 0 | 0.00% |
| clinical trial | 0 | 0.00% |
| anatomical structure | 0 | 0.00% |
| database | 0 | 0.00% |
| voltage | 0 | 0.00% |
| border | 0 | 0.00% |
| api | 0 | 0.00% |
| vaccine | 0 | 0.00% |
| mineral | 0 | 0.00% |
| budget line | 0 | 0.00% |
| protocol | 0 | 0.00% |
| market index | 0 | 0.00% |
| conference | 0 | 0.00% |
| interest rate | 0 | 0.00% |
| species | 0 | 0.00% |
| security vulnerability | 0 | 0.00% |
| criminal organization | 0 | 0.00% |
| file format | 0 | 0.00% |
| natural disaster | 0 | 0.00% |
| video game | 0 | 0.00% |
| hardware component | 0 | 0.00% |
| diagnosis | 0 | 0.00% |
| patent | 0 | 0.00% |
| stock ticker | 0 | 0.00% |
| framework | 0 | 0.00% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| no relation | 50 | 10.27% |
| member_of | 22 | 4.52% |
| PPI | 21 | 4.31% |
| affiliated_with | 19 | 3.90% |
| works_for | 18 | 3.70% |
| located_in | 13 | 2.67% |
| headquartered_in | 11 | 2.26% |
| Lives_In | 10 | 2.05% |
| USED-FOR | 7 | 1.44% |
| bind | 7 | 1.44% |
| represents | 6 | 1.23% |
| founded by | 6 | 1.23% |
| leader_of | 6 | 1.23% |
| Association | 6 | 1.23% |
| reports_to | 5 | 1.03% |
| employer_of | 5 | 1.03% |
| competitor_of | 5 | 1.03% |
| authored_by | 5 | 1.03% |
| has part(s) | 4 | 0.82% |
| operates_in | 4 | 0.82% |
| based on | 4 | 0.82% |
| produced_by | 4 | 0.82% |
| instance of | 4 | 0.82% |
| family name | 4 | 0.82% |
| architect | 4 | 0.82% |
| occupation | 4 | 0.82% |
| composer | 4 | 0.82% |
| product or material produced | 3 | 0.62% |
| residence | 3 | 0.62% |
| founded | 3 | 0.62% |
| supplier_of | 3 | 0.62% |
| dam | 3 | 0.62% |
| depicted by | 3 | 0.62% |
| league | 3 | 0.62% |
| place of birth | 3 | 0.62% |
| position held | 3 | 0.62% |
| given name | 3 | 0.62% |
| form of creative work | 3 | 0.62% |
| shares border with | 3 | 0.62% |
| parent taxon | 3 | 0.62% |
| HYPONYM-OF | 2 | 0.41% |
| COMPARE | 2 | 0.41% |
| country | 2 | 0.41% |
| located on terrain feature | 2 | 0.41% |
| presenter | 2 | 0.41% |
| librettist | 2 | 0.41% |
| online service | 2 | 0.41% |
| located in protected area | 2 | 0.41% |
| participant in | 2 | 0.41% |
| relative_of | 2 | 0.41% |
| director | 2 | 0.41% |
| headquarters location | 2 | 0.41% |
| acquired | 2 | 0.41% |
| conflict | 2 | 0.41% |
| country of citizenship | 2 | 0.41% |
| partner in business or sport | 2 | 0.41% |
| history of topic | 2 | 0.41% |
| IUCN conservation status | 2 | 0.41% |
| publisher | 2 | 0.41% |
| Negative_Correlation | 2 | 0.41% |
| participating team | 2 | 0.41% |
| cast member | 2 | 0.41% |
| distribution format | 2 | 0.41% |
| platform | 2 | 0.41% |
| performer | 2 | 0.41% |
| has contributing factor | 2 | 0.41% |
| based_in | 2 | 0.41% |
| developed_by | 2 | 0.41% |
| genre | 2 | 0.41% |
| competition class | 2 | 0.41% |
| sibling_of | 2 | 0.41% |
| language used | 2 | 0.41% |
| league system | 2 | 0.41% |
| writing language | 1 | 0.21% |
| PART-OF | 1 | 0.21% |
| CONJUNCTION | 1 | 0.21% |
| significant person | 1 | 0.21% |
| political alignment | 1 | 0.21% |
| located in the present-day administrative territorial entity | 1 | 0.21% |
| member of political party | 1 | 0.21% |
| culture | 1 | 0.21% |
| host | 1 | 0.21% |
| collection creator | 1 | 0.21% |
| terminus location | 1 | 0.21% |
| characters | 1 | 0.21% |
| present in work | 1 | 0.21% |
| country of registry | 1 | 0.21% |
| avionics | 1 | 0.21% |
| musical conductor | 1 | 0.21% |
| distributed by | 1 | 0.21% |
| spouse_of | 1 | 0.21% |
| grants | 1 | 0.21% |
| partially coincident with | 1 | 0.21% |
| head of government | 1 | 0.21% |
| owns | 1 | 0.21% |
| award received | 1 | 0.21% |
| advisor_to | 1 | 0.21% |
| political ideology | 1 | 0.21% |
| conferred by | 1 | 0.21% |
| lyrics by | 1 | 0.21% |
| crew member(s) | 1 | 0.21% |
| member of the crew of | 1 | 0.21% |
| territory overlaps | 1 | 0.21% |
| perpetrator | 1 | 0.21% |
| applies to jurisdiction | 1 | 0.21% |
| has anatomical branch | 1 | 0.21% |
| game mechanics | 1 | 0.21% |
| mentor_of | 1 | 0.21% |
| co-driver | 1 | 0.21% |
| item operated | 1 | 0.21% |
| source of energy | 1 | 0.21% |
| convicted of | 1 | 0.21% |
| colleague_of | 1 | 0.21% |
| title of chess person | 1 | 0.21% |
| location of formation | 1 | 0.21% |
| winner | 1 | 0.21% |
| facet of | 1 | 0.21% |
| choreographer | 1 | 0.21% |
| officeholder | 1 | 0.21% |
| general classification of race participants | 1 | 0.21% |
| ancestral home | 1 | 0.21% |
| office held by head of government | 1 | 0.21% |
| follows | 1 | 0.21% |
| main subject | 1 | 0.21% |
| currency | 1 | 0.21% |
| capital of | 1 | 0.21% |
| place of death | 1 | 0.21% |
| uses | 1 | 0.21% |
| connecting line | 1 | 0.21% |
| located in/on physical feature | 1 | 0.21% |
| noble title | 1 | 0.21% |
| medical condition treated | 1 | 0.21% |
| place of publication | 1 | 0.21% |
| customer_of | 1 | 0.21% |
| order of battle | 1 | 0.21% |
| vehicle normally used | 1 | 0.21% |
| mouth of the watercourse | 1 | 0.21% |
| sport | 1 | 0.21% |
| languages spoken, written or signed | 1 | 0.21% |
| twinned administrative body | 1 | 0.21% |
| filming location | 1 | 0.21% |
| location | 1 | 0.21% |
| fictional universe described in | 1 | 0.21% |
| continent | 1 | 0.21% |
| stepparent | 1 | 0.21% |
| legislative body | 1 | 0.21% |
| head of state | 1 | 0.21% |
| native language | 1 | 0.21% |
| connects with | 1 | 0.21% |
| sex or gender | 1 | 0.21% |
| canonization status | 1 | 0.21% |
| given name version for other gender | 1 | 0.21% |
| vehicle | 1 | 0.21% |
| operating system | 1 | 0.21% |
| archives at | 1 | 0.21% |
| surface played on | 1 | 0.21% |
| parent organization | 1 | 0.21% |
| historic county | 1 | 0.21% |
| stock exchange | 1 | 0.21% |
| located on astronomical body | 1 | 0.21% |
| part of the series | 1 | 0.21% |
| manner of death | 1 | 0.21% |
| encodes | 1 | 0.21% |
| country for sport | 1 | 0.21% |
| country of origin | 1 | 0.21% |
| lake outflow | 1 | 0.21% |
| type foundry | 1 | 0.21% |
| participant | 1 | 0.21% |
| party_to | 1 | 0.21% |
| significant event | 1 | 0.21% |
| architectural style | 1 | 0.21% |
| mother | 1 | 0.21% |
| subject has role | 1 | 0.21% |
| radio format | 1 | 0.21% |
| parent club | 1 | 0.21% |
| spouse | 1 | 0.21% |
| educated at | 1 | 0.21% |
| coextensive with | 1 | 0.21% |
| authority | 1 | 0.21% |
| biological process | 1 | 0.21% |
| contains the administrative territorial entity | 1 | 0.21% |
| regulated_by | 1 | 0.21% |
| inflows | 1 | 0.21% |
| anatomical location | 1 | 0.21% |
| named_after | 1 | 0.21% |
| licensed to broadcast to | 1 | 0.21% |
| developer | 1 | 0.21% |
| diocese | 1 | 0.21% |
| derived_from | 1 | 0.21% |
| mascot | 1 | 0.21% |
| language of work or name | 1 | 0.21% |
| directed_by | 1 | 0.21% |
| record label | 1 | 0.21% |
| set in environment | 1 | 0.21% |
| indigenous to | 1 | 0.21% |
| place of burial | 1 | 0.21% |
| has effect | 1 | 0.21% |

### Classifications

**train, actionability.** Majority class 83.9% of 31 answers. The marginal is `true_label`, not the menu.

#### train actionability

| label | count | share |
| --- | --- | --- |
| informational | 26 | 83.87% |
| requires action | 3 | 9.68% |
| advisory | 2 | 6.45% |

**train, audience.** Majority class 86.2% of 29 answers. The marginal is `true_label`, not the menu.

#### train audience

| label | count | share |
| --- | --- | --- |
| general public | 25 | 86.21% |
| expert | 2 | 6.90% |
| regulatory | 2 | 6.90% |

**train, biotech_event.** Majority class 29.4% of 17 answers. The marginal is `true_label`, not the menu.

#### train biotech_event

| label | count | share |
| --- | --- | --- |
| executive statement | 5 | 29.41% |
| company description | 3 | 17.65% |
| product updates | 2 | 11.76% |
| other | 2 | 11.76% |
| m&a | 2 | 11.76% |
| funding round | 1 | 5.88% |
| ipo exit | 1 | 5.88% |
| executive appointment | 1 | 5.88% |

**train, certainty.** Majority class 100.0% of 22 answers. The marginal is `true_label`, not the menu.

#### train certainty

| label | count | share |
| --- | --- | --- |
| asserted | 22 | 100.00% |

**train, chfinann_event.** Majority class 42.9% of 112 answers. The marginal is `true_label`, not the menu.

#### train chfinann_event

| label | count | share |
| --- | --- | --- |
| EquityPledge | 48 | 42.86% |
| EquityOverweight | 26 | 23.21% |
| EquityUnderweight | 25 | 22.32% |
| EquityRepurchase | 10 | 8.93% |
| EquityFreeze | 3 | 2.68% |

**train, docee_event.** Majority class 7.6% of 317 answers. The marginal is `true_label`, not the menu.

#### train docee_event

| label | count | share |
| --- | --- | --- |
| Sports Competition | 24 | 7.57% |
| none | 16 | 5.05% |
| CommitCrime - Arrest | 16 | 5.05% |
| Fire | 16 | 5.05% |
| Government Policy Changes | 13 | 4.10% |
| Organization Fine | 13 | 4.10% |
| Protest | 13 | 4.10% |
| Appoint_Inauguration | 12 | 3.79% |
| Air Crash | 9 | 2.84% |
| Shipwreck | 9 | 2.84% |
| Floods | 9 | 2.84% |
| Armed Conflict | 9 | 2.84% |
| New Achievements in Aerospace | 8 | 2.52% |
| Awards Ceremony | 8 | 2.52% |
| Election | 8 | 2.52% |
| Sign Agreement | 7 | 2.21% |
| CommitCrime - Investigate | 6 | 1.89% |
| Break Historical Records | 6 | 1.89% |
| Diplomatic Talks | 6 | 1.89% |
| Famous Person - Death | 6 | 1.89% |
| Resignation_Dismissal | 6 | 1.89% |
| Earthquakes | 5 | 1.58% |
| Famous Person - Give a Speech | 5 | 1.58% |
| Riot | 5 | 1.58% |
| Mass Poisoning | 5 | 1.58% |
| Bank Robbery | 5 | 1.58% |
| CommitCrime - Accuse | 5 | 1.58% |
| Diplomatic Visit | 5 | 1.58% |
| Financial Crisis | 4 | 1.26% |
| Storm | 4 | 1.26% |
| Road Crash | 4 | 1.26% |
| Disease Outbreaks | 4 | 1.26% |
| Organization Established | 3 | 0.95% |
| Volcano Eruption | 3 | 0.95% |
| Insect Disaster | 3 | 0.95% |
| CommitCrime - Sentence | 3 | 0.95% |
| Withdraw from an Orgnization | 3 | 0.95% |
| Gas Explosion | 3 | 0.95% |
| Famous Person - Sick | 3 | 0.95% |
| Train Collisions | 3 | 0.95% |
| Famous Person - Recovered | 2 | 0.63% |
| Organization Closed | 2 | 0.63% |
| Regime Change | 2 | 0.63% |
| CommitCrime - Release | 2 | 0.63% |
| Military Excercise | 2 | 0.63% |
| Strike | 2 | 0.63% |
| Environment Pollution | 2 | 0.63% |
| Famine | 1 | 0.32% |
| Mudslides | 1 | 0.32% |
| Join in an Orgnization | 1 | 0.32% |
| Financial Aid | 1 | 0.32% |
| New Wonders in Nature | 1 | 0.32% |
| Tsunamis | 1 | 0.32% |
| Droughts | 1 | 0.32% |
| Famous Person - Divorce | 1 | 0.32% |

**train, docfee_event.** Majority class 31.0% of 71 answers. The marginal is `true_label`, not the menu.

#### train docfee_event

| label | count | share |
| --- | --- | --- |
| EquityUnderweight | 22 | 30.99% |
| BankruptcyLiquidation | 15 | 21.13% |
| EquityPledge | 12 | 16.90% |
| MajorAssetLoss | 9 | 12.68% |
| EquityFreeze | 5 | 7.04% |
| EquityOverweight | 4 | 5.63% |
| MajorSafetyAccident | 2 | 2.82% |
| MajorExternalPayout | 1 | 1.41% |
| ExecutiveDeath | 1 | 1.41% |

**train, document_genre.** Majority class 69.0% of 29 answers. The marginal is `true_label`, not the menu.

#### train document_genre

| label | count | share |
| --- | --- | --- |
| news report | 20 | 68.97% |
| announcement | 4 | 13.79% |
| opinion | 3 | 10.34% |
| technical documentation | 1 | 3.45% |
| review | 1 | 3.45% |

**train, formality.** Majority class 83.3% of 30 answers. The marginal is `true_label`, not the menu.

#### train formality

| label | count | share |
| --- | --- | --- |
| formal | 25 | 83.33% |
| informal | 5 | 16.67% |

**train, language_register.** Majority class 81.5% of 27 answers. The marginal is `true_label`, not the menu.

#### train language_register

| label | count | share |
| --- | --- | --- |
| plain | 22 | 81.48% |
| technical | 3 | 11.11% |
| legal | 1 | 3.70% |
| colloquial | 1 | 3.70% |

**train, risk_level.** Majority class 66.7% of 24 answers. The marginal is `true_label`, not the menu.

#### train risk_level

| label | count | share |
| --- | --- | --- |
| none | 16 | 66.67% |
| high | 3 | 12.50% |
| low | 3 | 12.50% |
| moderate | 2 | 8.33% |

**train, sentiment.** Majority class 35.1% of 37 answers. The marginal is `true_label`, not the menu.

#### train sentiment

| label | count | share |
| --- | --- | --- |
| neutral | 13 | 35.14% |
| positive | 12 | 32.43% |
| negative | 12 | 32.43% |

**train, subjectivity.** Majority class 85.0% of 20 answers. The marginal is `true_label`, not the menu.

#### train subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 17 | 85.00% |
| subjective | 3 | 15.00% |

**train, temporal_orientation.** Majority class 73.3% of 30 answers. The marginal is `true_label`, not the menu.

#### train temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 22 | 73.33% |
| present | 6 | 20.00% |
| future | 2 | 6.67% |

**train, topic.** Majority class 19.4% of 62 answers. The marginal is `true_label`, not the menu.

#### train topic

| label | count | share |
| --- | --- | --- |
| business | 12 | 19.35% |
| entertainment | 9 | 14.52% |
| sports | 6 | 9.68% |
| health | 6 | 9.68% |
| world | 5 | 8.06% |
| politics | 5 | 8.06% |
| finance | 4 | 6.45% |
| legal | 4 | 6.45% |
| technology | 3 | 4.84% |
| science | 2 | 3.23% |
| environment | 2 | 3.23% |
| education | 1 | 1.61% |
| travel | 1 | 1.61% |
| agriculture | 1 | 1.61% |
| defense | 1 | 1.61% |

**train, urgency.** Majority class 72.2% of 18 answers. The marginal is `true_label`, not the menu.

#### train urgency

| label | count | share |
| --- | --- | --- |
| routine | 13 | 72.22% |
| elevated | 3 | 16.67% |
| critical | 1 | 5.56% |
| urgent | 1 | 5.56% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | condition | 1 | 100.00% |
| clinical_finding | patient_group | 1 | 100.00% |
| clinical_finding | treatment | 1 | 100.00% |
| clinical_finding | outcome | 1 | 100.00% |
| clinical_finding | record_metadata | natural/condition ×1 |  |
| company_profile | name | 2 | 100.00% |
| company_profile | industry | 2 | 100.00% |
| company_profile | size | 2 | 100.00% |
| company_profile | headquarters | 1 | 50.00% |
| company_profile | founded | 1 | 50.00% |
| company_profile | record_metadata | natural/name ×2 |  |
| event_listing | name | 1 | 100.00% |
| event_listing | venue | 1 | 100.00% |
| event_listing | date | 1 | 100.00% |
| event_listing | organizer | 1 | 100.00% |
| event_listing | record_metadata | natural/name ×1 |  |
| incident_report | date | 5 | 100.00% |
| incident_report | severity | 5 | 100.00% |
| incident_report | location | 4 | 80.00% |
| incident_report | incident_type | 4 | 80.00% |
| incident_report | casualties | 2 | 40.00% |
| incident_report | record_metadata | natural/location ×3, natural/incident_type ×2 |  |
| legal_case | case_name | 1 | 100.00% |
| legal_case | court | 1 | 100.00% |
| legal_case | plaintiff | 1 | 100.00% |
| legal_case | defendant | 1 | 100.00% |
| legal_case | status | 1 | 100.00% |
| legal_case | record_metadata | natural/case_name ×1 |  |
| person_profile | name | 8 | 100.00% |
| person_profile | employer | 8 | 100.00% |
| person_profile | role | 7 | 87.50% |
| person_profile | location | 7 | 87.50% |
| person_profile | record_metadata | natural/location ×4, natural/name ×3 |  |
| product | name | 3 | 100.00% |
| product | brand | 3 | 100.00% |
| product | price | 3 | 100.00% |
| product | condition | 3 | 100.00% |
| product | record_metadata | natural/name ×3 |  |
| record | status | 8 | 23.53% |
| record | location | 5 | 14.71% |
| record | price | 4 | 11.76% |
| record | address | 4 | 11.76% |
| record | name: Full name of the patient | 2 | 5.88% |
| record | repair_cost | 2 | 5.88% |
| record | ticket_id | 2 | 5.88% |
| record | state | 2 | 5.88% |
| record | name | 2 | 5.88% |
| record | city: City where patient resides | 1 | 2.94% |
| record | symptoms: List of reported symptoms like fever or headache | 1 | 2.94% |
| record | onset_date: Date when symptoms first appeared | 1 | 2.94% |
| record | address: Street address of the patient | 1 | 2.94% |
| record | passenger_phone | 1 | 2.94% |
| record | origin | 1 | 2.94% |
| record | delay_reason | 1 | 2.94% |
| record | delay_minutes | 1 | 2.94% |
| record | passenger_email | 1 | 2.94% |
| record | destination | 1 | 2.94% |
| record | patient_id: Unique identifier for the patient entry | 1 | 2.94% |
| record | status: Current status of vaccination record | 1 | 2.94% |
| record | vaccine_type: Type of vaccine administered | 1 | 2.94% |
| record | status: Current processing or acceptance status | 1 | 2.94% |
| record | author: Name of the submitting researcher or academic | 1 | 2.94% |
| record | institution: University or institution where author is affiliated | 1 | 2.94% |
| record | treatment: Therapeutic approach prescribed | 1 | 2.94% |
| record | provider: Name of the physician who treated the patient | 1 | 2.94% |
| record | diagnosis: Medical condition diagnosed with ICD-10 code | 1 | 2.94% |
| record | patient_id: Unique identifier for the patient | 1 | 2.94% |
| record | serial_number | 1 | 2.94% |
| record | claim_amount: Monetary claim amount if applicable | 1 | 2.94% |
| record | case_id: Unique case identifier in format CV-YYYY-XXXX | 1 | 2.94% |
| record | case_name: Full name of the case in format plaintiff v defendant | 1 | 2.94% |
| record | claim_type: Nature of the legal claim such as breach of contract or noise violation | 1 | 2.94% |
| record | entity_name | 1 | 2.94% |
| record | filing_date | 1 | 2.94% |
| record | submission_method | 1 | 2.94% |
| record | page_count: Number of pages in the paper | 1 | 2.94% |
| record | abstract_length: Length of the abstract in words | 1 | 2.94% |
| record | review_date: Date when the paper was reviewed | 1 | 2.94% |
| record | conference_name: Name of the conference where submitted | 1 | 2.94% |
| record | funding_source: Source of research funding | 1 | 2.94% |
| record | word_count: Total number of words in the paper | 1 | 2.94% |
| record | submission_date: Date when the paper was submitted | 1 | 2.94% |
| record | affiliation: Institutional affiliation of the author | 1 | 2.94% |
| record | fieldofstudy: Academic domain of the research | 1 | 2.94% |
| record | amount: Claimed financial amount in dollars | 1 | 2.94% |
| record | sender: Name of person who submitted the report | 1 | 2.94% |
| record | report_id: Unique identifier for the phishing incident | 1 | 2.94% |
| record | subject: Subject line of the malicious email | 1 | 2.94% |
| record | malicious_content: Description of the phishing element like link or attachment | 1 | 2.94% |
| record | acquirer | 1 | 2.94% |
| record | acquired | 1 | 2.94% |
| record | acquisition_amount | 1 | 2.94% |
| record | valuation_method | 1 | 2.94% |
| record | patient_id | 1 | 2.94% |
| record | medication | 1 | 2.94% |
| record | dosage | 1 | 2.94% |
| record | quantity | 1 | 2.94% |
| record | frequency | 1 | 2.94% |
| record | issue_date | 1 | 2.94% |
| record | pharmacy_name | 1 | 2.94% |
| record | pharmacy_address | 1 | 2.94% |
| record | hb: Hemoglobin level in grams per deciliter | 1 | 2.94% |
| record | test_date: Date the lab test was conducted | 1 | 2.94% |
| record | patient_name: Name of the patient undergoing the test | 1 | 2.94% |
| record | platelets: Platelet count in thousands per microliter | 1 | 2.94% |
| record | issue | 1 | 2.94% |
| record | affected_period | 1 | 2.94% |
| record | reviewer | 1 | 2.94% |
| record | product | 1 | 2.94% |
| record | model | 1 | 2.94% |
| record | review_date | 1 | 2.94% |
| record | rating | 1 | 2.94% |
| record | review_length | 1 | 2.94% |
| record | brand | 1 | 2.94% |
| record | release_year | 1 | 2.94% |
| record | warranty_period | 1 | 2.94% |
| record | compatibility | 1 | 2.94% |
| record | power_consumption | 1 | 2.94% |
| record | charging_time | 1 | 2.94% |
| record | storage_capacity | 1 | 2.94% |
| record | use_case | 1 | 2.94% |
| record | reviewer_name | 1 | 2.94% |
| record | review_id | 1 | 2.94% |
| record | description.stipules.dimensions | 1 | 2.94% |
| record | description.leaf.texture | 1 | 2.94% |
| record | family | 1 | 2.94% |
| record | location.other_locations | 1 | 2.94% |
| record | location.endemism | 1 | 2.94% |
| record | honorary_name.field_of_study | 1 | 2.94% |
| record | description.seed.color | 1 | 2.94% |
| record | honorary_name.person | 1 | 2.94% |
| record | description.leaf.base | 1 | 2.94% |
| record | description.type | 1 | 2.94% |
| record | description.leaf.dimensions | 1 | 2.94% |
| record | description.carpels | 1 | 2.94% |
| record | conservation_status.status | 1 | 2.94% |
| record | description.fruit.dimensions | 1 | 2.94% |
| record | description.floral_buds.dimensions | 1 | 2.94% |
| record | habitat.associated_species | 1 | 2.94% |
| record | scientific_name | 1 | 2.94% |
| record | description.leaf.apex | 1 | 2.94% |
| record | description.seed.dimensions | 1 | 2.94% |
| record | honorary_name.profession | 1 | 2.94% |
| record | sku | 1 | 2.94% |
| record | product_id | 1 | 2.94% |
| record | release_date | 1 | 2.94% |
| record | account | 1 | 2.94% |
| record | amount | 1 | 2.94% |
| record | payee | 1 | 2.94% |
| record | dispute_status | 1 | 2.94% |
| record | owner | 1 | 2.94% |
| record | owner_location | 1 | 2.94% |
| record | opponent | 1 | 2.94% |
| record | implementation_notes | 1 | 2.94% |
| record | publication_date | 1 | 2.94% |
| record | guideline_code | 1 | 2.94% |
| record | author | 1 | 2.94% |
| record | cost_impact | 1 | 2.94% |
| record | match_round: Stage of the tournament such as final, quarterfinal, or last 16 | 1 | 2.94% |
| record | score: Final score of the match in set format | 1 | 2.94% |
| record | venue_name: Name of the specific tournament venue | 1 | 2.94% |
| record | location: Venue where the match was held | 1 | 2.94% |
| record | winner: Player who won the match | 1 | 2.94% |
| record | founder | 1 | 2.94% |
| record | company | 1 | 2.94% |
| record | founding_year | 1 | 2.94% |
| record | funding_source | 1 | 2.94% |
| record | key_challenge | 1 | 2.94% |
| record | discovery.date | 1 | 2.94% |
| record | physical_characteristics.observations.0.telescope.type | 1 | 2.94% |
| record | diameter.unit | 1 | 2.94% |
| record | physical_characteristics.observations.0.telescope.origin | 1 | 2.94% |
| record | asteroid_belt_classification | 1 | 2.94% |
| record | discovery.discoverer | 1 | 2.94% |
| record | discovery.observatory.location.prefecture | 1 | 2.94% |
| record | physical_characteristics.observations.0.publication_year | 1 | 2.94% |
| record | physical_characteristics.observations.0.telescope.name | 1 | 2.94% |
| record | discovery.observatory.location.country | 1 | 2.94% |
| record | clause_id | 1 | 2.94% |
| record | capacity | 1 | 2.94% |
| record | sessions | 1 | 2.94% |
| record | conference_id | 1 | 2.94% |
| record | title | 1 | 2.94% |
| record | priority_level | 1 | 2.94% |
| record | reason | 1 | 2.94% |
| record | crime.location.landmark | 1 | 2.94% |
| record | perpetrator.status | 1 | 2.94% |
| record | crime.location.address | 1 | 2.94% |
| record | perpetrator.arrest_details.agency | 1 | 2.94% |
| record | crime.location.sector | 1 | 2.94% |
| record | crime.motive | 1 | 2.94% |
| record | victim.name | 1 | 2.94% |
| record | investigation.agency | 1 | 2.94% |
| record | crime.weapon.type | 1 | 2.94% |
| record | crime.weapon.caliber | 1 | 2.94% |
| record | perpetrator.arrest_details.time | 1 | 2.94% |
| record | crime.date | 1 | 2.94% |
| record | investigation.medical_examiner.name | 1 | 2.94% |
| record | crime.weapon.source.stolen_from.name | 1 | 2.94% |
| record | investigation.status | 1 | 2.94% |
| record | offer_id: Unique identifier for each leasing offer | 1 | 2.94% |
| record | tenant: Name of the company occupying the leased space | 1 | 2.94% |
| record | location: Full address of the leased property | 1 | 2.94% |
| record | features: List of included amenities or space features | 1 | 2.94% |
| record | municipality.name | 1 | 2.94% |
| record | agency | 1 | 2.94% |
| record | position | 1 | 2.94% |
| record | record_metadata | natural/status ×3, natural/city: City where patient resides ×1, natural/passenger_phone ×1, natural/patient_id: Unique identifier for the patient entry ×1 |  |
| research_paper | title | 1 | 100.00% |
| research_paper | authors | 1 | 100.00% |
| research_paper | venue | 1 | 100.00% |
| research_paper | year | 1 | 100.00% |
| research_paper | record_metadata | natural/title ×1 |  |
| transaction | item | 2 | 100.00% |
| transaction | buyer | 2 | 100.00% |
| transaction | seller | 2 | 100.00% |
| transaction | amount | 1 | 50.00% |
| transaction | date | 1 | 50.00% |
| transaction | record_metadata | natural/item ×2 |  |

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | mixed (replay) |
| relations | mixed (replay) |
| events | mixed (replay) |
| classifications | mixed (replay) |
| structures | mixed (replay) |
