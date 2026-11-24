# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replay_eb22g_gate4 | train | 8,809 | 8,761 | 53,627 | 4,039 | 6,665 | 438 | 4,853 | 6,848 |
| replay_eb22g_gate4 | val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| replay_eb22g_gate4 | test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## replay_eb22g_gate4

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 8,809 | 8,761 | 334 | 1209 | 8409 | 53,627 | 4,039 | 6,665 | 438 | 4,853 | 6,848 |
| val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 4,924 | 0 | 0 | 4,924 | 55.90% |
| zho | 2,926 | 0 | 0 | 2,926 | 33.22% |
| tur | 798 | 0 | 0 | 798 | 9.06% |
| und | 114 | 0 | 0 | 114 | 1.29% |
| rus | 5 | 0 | 0 | 5 | 0.06% |
| spa | 5 | 0 | 0 | 5 | 0.06% |
| ukr | 5 | 0 | 0 | 5 | 0.06% |
| ita | 4 | 0 | 0 | 4 | 0.05% |
| fra | 4 | 0 | 0 | 4 | 0.05% |
| ara | 4 | 0 | 0 | 4 | 0.05% |
| hin | 3 | 0 | 0 | 3 | 0.03% |
| por | 3 | 0 | 0 | 3 | 0.03% |
| kor | 3 | 0 | 0 | 3 | 0.03% |
| deu | 2 | 0 | 0 | 2 | 0.02% |
| ces | 2 | 0 | 0 | 2 | 0.02% |
| nor | 1 | 0 | 0 | 1 | 0.01% |
| pol | 1 | 0 | 0 | 1 | 0.01% |
| mlg | 1 | 0 | 0 | 1 | 0.01% |
| nld | 1 | 0 | 0 | 1 | 0.01% |
| fin | 1 | 0 | 0 | 1 | 0.01% |
| hun | 1 | 0 | 0 | 1 | 0.01% |
| eml | 1 | 0 | 0 | 1 | 0.01% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 0 |
| test_unique_inputs | 0 |
| train_records | 8,809 |
| train_unique_inputs | 8,761 |
| val_records | 0 |
| val_unique_inputs | 0 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | n/a | 53,627 / 0 |  |
| entities (per mention) | train / test | n/a | 53,627 / 0 |  |
| relations (per instance) | train / val | n/a | 4,039 / 0 |  |
| relations (per instance) | train / test | n/a | 4,039 / 0 |  |
| event types (per trigger mention) | train / val | n/a | 4,547 / 0 |  |
| event types (per trigger mention) | train / test | n/a | 4,547 / 0 |  |
| event roles (per argument) | train / val | n/a | 7,070 / 0 |  |
| event roles (per argument) | train / test | n/a | 7,070 / 0 |  |
| classifications (true_label) | train / val | n/a | 6,665 / 0 |  |
| classifications (true_label) | train / test | n/a | 6,665 / 0 |  |


### Events

**train.** 5,247 instances in 1,486 records, 899 triggerless. Trigger gold 4,853. Argument gold 6,848. Arguments per instance, mean 1.35. Trigger length in whitespace tokens, mean 1.26, single-token share 81.5%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Contact.Communicate | 371 | 8.16% |
| Hostile_encounter | 121 | 2.66% |
| Sport.Compete | 119 | 2.62% |
| Catastrophe | 100 | 2.20% |
| Attack | 96 | 2.11% |
| Competition | 96 | 2.11% |
| Personnel.StartPosition | 79 | 1.74% |
| Process_start | 79 | 1.74% |
| Causation | 77 | 1.69% |
| match_result | 70 | 1.54% |
| Life.Die | 60 | 1.32% |
| Personnel.EndPosition | 60 | 1.32% |
| Justice.Arrest | 53 | 1.17% |
| Military_operation | 53 | 1.17% |
| Motion | 52 | 1.14% |
| Conflict.Attack | 51 | 1.12% |
| Social_event | 50 | 1.10% |
| Process_end | 49 | 1.08% |
| Coming_to_be | 49 | 1.08% |
| product_publish | 46 | 1.01% |
| Conquering | 45 | 0.99% |
| Business.StartOrg | 44 | 0.97% |
| Contact.Meet | 43 | 0.95% |
| Killing | 42 | 0.92% |
| Life.Injure | 40 | 0.88% |
| Justice.ChargeIndict | 40 | 0.88% |
| Destroying | 40 | 0.88% |
| Arriving | 40 | 0.88% |
| Creating | 40 | 0.88% |
| Self_motion | 39 | 0.86% |
| Control | 39 | 0.86% |
| Statement | 38 | 0.84% |
| Cause_change_of_position_on_a_scale | 36 | 0.79% |
| Getting | 33 | 0.73% |
| Hold | 33 | 0.73% |
| Escaping | 33 | 0.73% |
| Transaction.TransferOwnership | 32 | 0.70% |
| Bodily_harm | 31 | 0.68% |
| Disaster.Accident | 31 | 0.68% |
| Earnings_and_losses | 30 | 0.66% |
| Becoming | 29 | 0.64% |
| Presence | 29 | 0.64% |
| Cause_change_of_strength | 28 | 0.62% |
| Damaging | 27 | 0.59% |
| Award.ReceiveAward | 25 | 0.55% |
| judicial_action_arrest | 25 | 0.55% |
| Justice.TrialHearing | 25 | 0.55% |
| Death | 25 | 0.55% |
| Change_of_leadership | 25 | 0.55% |
| Agree_or_refuse_to_act | 25 | 0.55% |
| Name_conferral | 24 | 0.53% |
| Arranging | 23 | 0.51% |
| Placing | 23 | 0.51% |
| Movement.Transport | 22 | 0.48% |
| Removing | 22 | 0.48% |
| resignation | 21 | 0.46% |
| death | 21 | 0.46% |
| Change | 21 | 0.46% |
| Defending | 21 | 0.46% |
| Besieging | 21 | 0.46% |
| Supporting | 20 | 0.44% |
| Preventing_or_letting | 20 | 0.44% |
| Business.LaunchProduct | 19 | 0.42% |
| Supply | 19 | 0.42% |
| Know | 19 | 0.42% |
| Justice.Sue | 18 | 0.40% |
| Request | 18 | 0.40% |
| Government.Vote | 18 | 0.40% |
| Life.BeBorn | 17 | 0.37% |
| championship_win | 17 | 0.37% |
| Action | 17 | 0.37% |
| Motion_directional | 17 | 0.37% |
| Recovering | 17 | 0.37% |
| Traveling | 17 | 0.37% |
| Justice.Sentence | 17 | 0.37% |
| Expressing_publicly | 17 | 0.37% |
| Departing | 16 | 0.35% |
| Influence | 15 | 0.33% |
| Choosing | 15 | 0.33% |
| Participation | 15 | 0.33% |
| Cause_to_be_included | 15 | 0.33% |
| Bringing | 15 | 0.33% |
| Assistance | 15 | 0.33% |
| Movement.Arrive | 15 | 0.33% |
| Tech.Release | 15 | 0.33% |
| Transaction.TransferMoney | 14 | 0.31% |
| Becoming_a_member | 14 | 0.31% |
| Sending | 14 | 0.31% |
| Conflict.Demonstrate | 13 | 0.29% |
| Manufacturing | 13 | 0.29% |
| Recording | 13 | 0.29% |
| advancement | 12 | 0.26% |
| Giving | 12 | 0.26% |
| departure | 12 | 0.26% |
| Deciding | 12 | 0.26% |
| Come_together | 12 | 0.26% |
| marriage | 11 | 0.24% |
| Medical.Treat | 11 | 0.24% |
| product_recall | 11 | 0.24% |
| Government.EnactLaw | 11 | 0.24% |
| Reporting | 11 | 0.24% |
| Using | 11 | 0.24% |
| judicial_action_summon | 11 | 0.24% |
| opening | 10 | 0.22% |
| Research.Discover | 10 | 0.22% |
| life.injure.injurycausedbyviolentevents | 10 | 0.22% |
| life.die.deathcausedbyviolentevents | 10 | 0.22% |
| Achieve | 10 | 0.22% |
| Expansion | 10 | 0.22% |
| Cause_to_amalgamate | 10 | 0.22% |
| Disaster.NaturalDisaster | 10 | 0.22% |
| Scrutiny | 10 | 0.22% |
| Research.Fund | 9 | 0.20% |
| joined | 9 | 0.20% |
| Dispersal | 9 | 0.20% |
| Building | 9 | 0.20% |
| conflict.attack.selfdirectedbattle | 9 | 0.20% |
| Response | 9 | 0.20% |
| Life.Marry | 8 | 0.18% |
| product_delist | 8 | 0.18% |
| dismissal | 8 | 0.18% |
| Transaction.Donate | 8 | 0.18% |
| Business.EndOrg | 8 | 0.18% |
| Contact.Negotiate | 8 | 0.18% |
| birth | 8 | 0.18% |
| Movement.Migrate | 8 | 0.18% |
| Quarreling | 8 | 0.18% |
| Judgment_communication | 8 | 0.18% |
| divorce | 8 | 0.18% |
| Resolve_problem | 8 | 0.18% |
| breakup | 7 | 0.15% |
| movement.transportartifact.n/a | 7 | 0.15% |
| financing | 7 | 0.15% |
| judicial_action_fine | 7 | 0.15% |
| Change_event_time | 7 | 0.15% |
| Aiming | 7 | 0.15% |
| Temporary_stay | 7 | 0.15% |
| Receiving | 7 | 0.15% |
| Research.Publish | 7 | 0.15% |
| Communication | 7 | 0.15% |
| Employment | 7 | 0.15% |
| disaster.fireexplosion.fireexplosion | 7 | 0.15% |
| Surrendering | 7 | 0.15% |
| Forming_relationships | 7 | 0.15% |
| Kidnapping | 7 | 0.15% |
| Releasing | 7 | 0.15% |
| Hindering | 7 | 0.15% |
| Collaboration | 7 | 0.15% |
| Sign_agreement | 7 | 0.15% |
| Medical.Diagnose | 6 | 0.13% |
| contact.requestadvise.correspondence | 6 | 0.13% |
| Commitment | 6 | 0.13% |
| Emptying | 6 | 0.13% |
| product_award | 6 | 0.13% |
| Justice.Fine | 6 | 0.13% |
| transaction.transfermoney.n/a | 6 | 0.13% |
| Fire | 6 | 0.13% |
| Disaster.Rescue | 6 | 0.13% |
| Life.Evacuate | 6 | 0.13% |
| Use_firearm | 6 | 0.13% |
| Exchange | 6 | 0.13% |
| judicial_action_prosecution | 6 | 0.13% |
| Publishing | 6 | 0.13% |
| product_release | 6 | 0.13% |
| Award | 6 | 0.13% |
| Personnel.Elect | 6 | 0.13% |
| Convincing | 6 | 0.13% |
| Cost | 6 | 0.13% |
| Personnel.Nominate | 5 | 0.11% |
| layoff | 5 | 0.11% |
| judicial_action_report | 5 | 0.11% |
| wedding | 5 | 0.11% |
| Cause_to_make_progress | 5 | 0.11% |
| Coming_to_believe | 5 | 0.11% |
| Check | 5 | 0.11% |
| Protest | 5 | 0.11% |
| Business.MergeOrg | 5 | 0.11% |
| competition_withdrawal | 5 | 0.11% |
| Reforming_a_system | 5 | 0.11% |
| Hiding_objects | 5 | 0.11% |
| Testing | 5 | 0.11% |
| Perception_active | 5 | 0.11% |
| contact.requestadvise.broadcast | 5 | 0.11% |
| Rescuing | 5 | 0.11% |
| Cure | 5 | 0.11% |
| judicial_action_hearing | 5 | 0.11% |
| birthday_celebration | 4 | 0.09% |
| transaction.transferownership.borrowlend | 4 | 0.09% |
| conflict.attack.n/a | 4 | 0.09% |
| contact.commitmentpromiseexpressintent.meet | 4 | 0.09% |
| Collapse | 4 | 0.09% |
| Being_in_operation | 4 | 0.09% |
| artifactexistence.damagedestroy.damage | 4 | 0.09% |
| contact.publicstatementinperson.broadcast | 4 | 0.09% |
| Expend_resource | 4 | 0.09% |
| Containing | 4 | 0.09% |
| Body_movement | 4 | 0.09% |
| Government.Protest | 4 | 0.09% |
| suspension | 4 | 0.09% |
| contact.negotiate.meet | 4 | 0.09% |
| competition_ban | 4 | 0.09% |
| Research | 4 | 0.09% |
| Openness | 4 | 0.09% |
| Confronting_problem | 4 | 0.09% |
| inspection.sensoryobserve.inspectpeopleorganization | 4 | 0.09% |
| Filling | 4 | 0.09% |
| Preserving | 4 | 0.09% |
| engagement | 4 | 0.09% |
| Committing_crime | 4 | 0.09% |
| rate_decrease | 4 | 0.09% |
| contact.threatencoerce.broadcast | 3 | 0.07% |
| retirement | 3 | 0.07% |
| movement.transportartifact.sendsupplyexport | 3 | 0.07% |
| contact.discussion.meet | 3 | 0.07% |
| government.agreements.acceptagreementcontractceasefire | 3 | 0.07% |
| inspection.sensoryobserve.physicalinvestigateinspect | 3 | 0.07% |
| sale_acquisition | 3 | 0.07% |
| Surrounding | 3 | 0.07% |
| personnel.endposition.quitretire | 3 | 0.07% |
| Disaster.Outbreak | 3 | 0.07% |
| conflict.attack.bombing | 3 | 0.07% |
| artifactexistence.damagedestroy.destroy | 3 | 0.07% |
| justice.judicialconsequences.convict | 3 | 0.07% |
| movement.transportartifact.fall | 3 | 0.07% |
| transaction.transaction.n/a | 3 | 0.07% |
| judicial_action_imprisonment | 3 | 0.07% |
| GiveUp | 3 | 0.07% |
| Ratification | 3 | 0.07% |
| transaction.transfermoney.payforservice | 3 | 0.07% |
| movement.transportartifact.grantentry | 3 | 0.07% |
| Connect | 3 | 0.07% |
| praise | 3 | 0.07% |
| movement.transportperson.preventexit | 3 | 0.07% |
| justice.initiatejudicialprocess.chargeindict | 3 | 0.07% |
| Telling | 3 | 0.07% |
| government.agreements.n/a | 3 | 0.07% |
| price_limit_down | 3 | 0.07% |
| life.die.n/a | 3 | 0.07% |
| Commerce_pay | 3 | 0.07% |
| Life.Divorce | 3 | 0.07% |
| Business.DeclareBankruptcy | 3 | 0.07% |
| transaction.transferownership.purchase | 3 | 0.07% |
| transaction.transfermoney.purchase | 3 | 0.07% |
| Legal_rulings | 3 | 0.07% |
| Warning | 3 | 0.07% |
| lost_contact | 2 | 0.04% |
| judicial_action_filing | 2 | 0.04% |
| conflict.attack.invade | 2 | 0.04% |
| proposal | 2 | 0.04% |
| personnel.elect.winelection | 2 | 0.04% |
| Justice.Acquit | 2 | 0.04% |
| movement.transportperson.selfmotion | 2 | 0.04% |
| Plane Crash | 2 | 0.04% |
| Business.Layoff | 2 | 0.04% |
| price_decrease | 2 | 0.04% |
| meet_with | 2 | 0.04% |
| pregnancy | 2 | 0.04% |
| government.agreements.rejectnullifyagreementcontractceasefire | 2 | 0.04% |
| justice.arrestjaildetain.arrestjaildetain | 2 | 0.04% |
| personnel.startposition.hiring | 2 | 0.04% |
| Violence | 2 | 0.04% |
| Explosion | 2 | 0.04% |
| contact.discussion.n/a | 2 | 0.04% |
| movement.transportperson.smuggleextract | 2 | 0.04% |
| Patrolling | 2 | 0.04% |
| Imposing_obligation | 2 | 0.04% |
| contact.commandorder.correspondence | 2 | 0.04% |
| conflict.attack.airstrikemissilestrike | 2 | 0.04% |
| personnel.endposition.firinglayoff | 2 | 0.04% |
| Change_sentiment | 2 | 0.04% |
| contact.collaborate.meet | 2 | 0.04% |
| thank | 2 | 0.04% |
| Movement.Deport | 2 | 0.04% |
| movement.transportperson.grantentryasylum | 2 | 0.04% |
| apologize_to | 2 | 0.04% |
| Education_teaching | 2 | 0.04% |
| Flood | 2 | 0.04% |
| Terrorism | 2 | 0.04% |
| Legality | 2 | 0.04% |
| contact.funeralvigil.meet | 2 | 0.04% |
| conflict.demonstrate.n/a | 2 | 0.04% |
| Commerce_buy | 2 | 0.04% |
| conflict.attack.firearmattack | 2 | 0.04% |
| Earthquake | 2 | 0.04% |
| transaction.transfermoney.embargosanction | 2 | 0.04% |
| Reveal_secret | 2 | 0.04% |
| contact.requestadvise.n/a | 2 | 0.04% |
| Wearing | 2 | 0.04% |
| Experiment | 1 | 0.02% |
| justice.investigate.n/a | 1 | 0.02% |
| conflict.attack.stealrobhijack | 1 | 0.02% |
| government.legislate.legislate | 1 | 0.02% |
| contact.commitmentpromiseexpressintent.correspondence | 1 | 0.02% |
| conflict.yield.retreat | 1 | 0.02% |
| Limiting | 1 | 0.02% |
| closing | 1 | 0.02% |
| government.vote.castvote | 1 | 0.02% |
| Criminal_investigation | 1 | 0.02% |
| Labeling | 1 | 0.02% |
| price_increase | 1 | 0.02% |
| transaction.transferownership.giftgrantprovideaid | 1 | 0.02% |
| Robbery | 1 | 0.02% |
| Writing | 1 | 0.02% |
| march | 1 | 0.02% |
| manufacture.artifact.build | 1 | 0.02% |
| artifactexistence.damagedestroy.n/a | 1 | 0.02% |
| ipo | 1 | 0.02% |
| movement.transportperson.fall | 1 | 0.02% |
| movement.transportartifact.smuggleextract | 1 | 0.02% |
| government.agreements.violateagreement | 1 | 0.02% |
| Rewards_and_punishments | 1 | 0.02% |
| Ingestion | 1 | 0.02% |
| movement.transportperson.n/a | 1 | 0.02% |
| disaster.accidentcrash.accidentcrash | 1 | 0.02% |
| movement.transportartifact.receiveimport | 1 | 0.02% |
| movement.transportartifact.hide | 1 | 0.02% |
| life.die.nonviolentdeath | 1 | 0.02% |
| contact.prevarication.meet | 1 | 0.02% |
| contact.prevarication.correspondence | 1 | 0.02% |
| contact.negotiate.n/a | 1 | 0.02% |
| price_limit_up | 1 | 0.02% |
| transaction.transferownership.n/a | 1 | 0.02% |
| Renting | 1 | 0.02% |
| contact.commitmentpromiseexpressintent.broadcast | 1 | 0.02% |
| movement.transportartifact.preventexit | 1 | 0.02% |
| justice.judicialconsequences.execute | 1 | 0.02% |
| movement.transportartifact.nonviolentthrowlaunch | 1 | 0.02% |
| Lighting | 1 | 0.02% |
| justice.investigate.investigatecrime | 1 | 0.02% |
| transaction.transferownership.embargosanction | 1 | 0.02% |
| movement.transportartifact.bringcarryunload | 1 | 0.02% |
| Bearing_arms | 1 | 0.02% |
| contact.collaborate.n/a | 1 | 0.02% |
| infidelity | 1 | 0.02% |
| dissolution | 1 | 0.02% |
| Government.Sanction | 1 | 0.02% |
| Submitting_documents | 1 | 0.02% |
| contact.publicstatementinperson.n/a | 1 | 0.02% |
| Revenge | 1 | 0.02% |
| Having_or_lacking_access | 1 | 0.02% |
| contract_termination | 1 | 0.02% |
| life.injure.illnessdegradationphysical | 1 | 0.02% |
| contact.collaborate.correspondence | 1 | 0.02% |
| inspection.sensoryobserve.monitorelection | 1 | 0.02% |
| movement.transportperson.bringcarryunload | 1 | 0.02% |
| transaction.transaction.transfercontrol | 1 | 0.02% |
| Traffic Accident | 1 | 0.02% |
| contact.mediastatement.n/a | 1 | 0.02% |
| Carry_goods | 1 | 0.02% |
| visit_on_set | 1 | 0.02% |
| contact.threatencoerce.correspondence | 1 | 0.02% |
| life.injure.n/a | 1 | 0.02% |
| manufacture.artifact.n/a | 1 | 0.02% |
| Arrest | 1 | 0.02% |
| contact.commitmentpromiseexpressintent.n/a | 1 | 0.02% |
| Theft | 1 | 0.02% |
| contact.prevarication.n/a | 1 | 0.02% |
| personnel.startposition.n/a | 1 | 0.02% |
| contact.requestadvise.meet | 1 | 0.02% |
| Commerce_sell | 1 | 0.02% |
| personnel.endposition.n/a | 1 | 0.02% |
| Rite | 1 | 0.02% |
| Justifying | 1 | 0.02% |
| Create_artwork | 1 | 0.02% |
| Adducing | 1 | 0.02% |
| rate_increase | 1 | 0.02% |
| transaction.transfermoney.giftgrantprovideaid | 1 | 0.02% |
| contact.prevarication.broadcast | 1 | 0.02% |
| Support | 0 | 0.00% |
| Conflict | 0 | 0.00% |
| Injure | 0 | 0.00% |
| Exhibit | 0 | 0.00% |
| Deploy | 0 | 0.00% |
| Accident | 0 | 0.00% |
| Cyber.Ransom | 0 | 0.00% |
| Manoeuvre | 0 | 0.00% |
| Cyber.DiscoverVulnerability | 0 | 0.00% |
| Cyber.PatchVulnerability | 0 | 0.00% |
| Cyber.Phishing | 0 | 0.00% |
| Cyber.Databreach | 0 | 0.00% |
| Conflict.Attack.DetonateExplode | 0 | 0.00% |
| Cognitive.IdentifyCategorize.Unspecified | 0 | 0.00% |
| Justice.InvestigateCrime.Unspecified | 0 | 0.00% |
| Life.Die.Unspecified | 0 | 0.00% |
| Life.Injure.Unspecified | 0 | 0.00% |
| Conflict.Attack.Unspecified | 0 | 0.00% |
| Justice.ChargeIndict.Unspecified | 0 | 0.00% |
| Justice.TrialHearing.Unspecified | 0 | 0.00% |
| Movement.Transportation.Unspecified | 0 | 0.00% |
| Justice.ArrestJailDetain.Unspecified | 0 | 0.00% |
| Transaction.ExchangeBuySell.Unspecified | 0 | 0.00% |
| GenericCrime.GenericCrime.GenericCrime | 0 | 0.00% |
| Contact.Contact.Broadcast | 0 | 0.00% |
| Disaster.Crash.Unspecified | 0 | 0.00% |
| Contact.Contact.Meet | 0 | 0.00% |
| Cognitive.Research.Unspecified | 0 | 0.00% |
| Justice.Convict.Unspecified | 0 | 0.00% |
| Justice.Sentence.Unspecified | 0 | 0.00% |

**val.** No event instances.

**test.** No event instances.

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1 | 2,534 | 0.04% | 0 | 1 | 8 | 0 | 2,525 | 4,536 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

**train by role.**

| role | inside | placed | inside share | unmeasured |
| --- | --- | --- | --- | --- |
| Subject | 0 | 670 | 0.00% | 9 |
| Equipment | 1 | 340 | 0.29% | 3 |
| Date | 0 | 272 | 0.00% | 188 |
| Object | 0 | 204 | 0.00% | 2 |
| Materials | 0 | 194 | 0.00% | 4 |
| Location | 0 | 181 | 0.00% | 42 |
| Victim | 0 | 96 | 0.00% | 85 |
| Militaryforce | 0 | 89 | 0.00% | 0 |
| Vulnerability | 0 | 50 | 0.00% | 0 |
| Vulnerable_System | 0 | 46 | 0.00% | 0 |
| Tool | 0 | 34 | 0.00% | 0 |
| Attacker | 0 | 28 | 0.00% | 19 |
| Compromised-Data | 0 | 25 | 0.00% | 0 |
| Trusted-Entity | 0 | 25 | 0.00% | 0 |
| Attack-Pattern | 0 | 24 | 0.00% | 0 |
| Discoverer | 0 | 24 | 0.00% | 0 |
| Time | 0 | 19 | 0.00% | 495 |
| Content | 0 | 17 | 0.00% | 0 |
| Place | 0 | 17 | 0.00% | 333 |
| Patch | 0 | 14 | 0.00% | 0 |
| Purpose | 0 | 13 | 0.00% | 0 |
| Target | 0 | 13 | 0.00% | 28 |
| Defendant | 0 | 12 | 0.00% | 92 |
| Result | 0 | 11 | 0.00% | 32 |
| Capabilities | 0 | 10 | 0.00% | 0 |
| Price | 0 | 10 | 0.00% | 26 |
| Releaser | 0 | 10 | 0.00% | 0 |
| Quantity | 0 | 9 | 0.00% | 2 |
| Vulnerable_System_Version | 0 | 9 | 0.00% | 0 |
| Area | 0 | 7 | 0.00% | 0 |
| ExplosiveDevice | 0 | 7 | 0.00% | 0 |
| Vulnerable_System_Owner | 0 | 7 | 0.00% | 0 |
| Investigator | 0 | 5 | 0.00% | 0 |
| CVE | 0 | 4 | 0.00% | 0 |
| Participant | 0 | 4 | 0.00% | 92 |
| Prosecutor | 0 | 4 | 0.00% | 8 |
| Researcher | 0 | 4 | 0.00% | 9 |
| Communicator | 0 | 3 | 0.00% | 292 |
| Origin | 0 | 3 | 0.00% | 16 |
| Detainee | 0 | 2 | 0.00% | 0 |
| Number-of-Victim | 0 | 2 | 0.00% | 0 |
| PassengerArtifact | 0 | 2 | 0.00% | 0 |
| Perpetrator | 0 | 2 | 0.00% | 0 |
| Transporter | 0 | 2 | 0.00% | 0 |
| AcquiredEntity | 0 | 1 | 0.00% | 0 |
| CrashObject | 0 | 1 | 0.00% | 0 |
| IdentifiedObject | 0 | 1 | 0.00% | 0 |
| Identifier | 0 | 1 | 0.00% | 0 |
| Instrument | 0 | 1 | 0.00% | 6 |
| Issues-Addressed | 0 | 1 | 0.00% | 0 |
| JudgeCourt | 0 | 1 | 0.00% | 0 |
| Killer | 0 | 1 | 0.00% | 0 |
| Payment-Method | 0 | 1 | 0.00% | 0 |
| Vehicle | 0 | 1 | 0.00% | 16 |

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.78 | 30.04% | 1.90 | 32.59% | 0.02% | 2,536 | 4,534 | 899 | 4,547 |
| val | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |
| test | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 5,247 | 2.55% | 3,805 | 11.09% | 10.54% | 2.31% | 1.13% | 14.13% |
| val | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |
| test | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

| split | docs | clusters | mentions | multi-mention | singletons | mention pairs | pronoun mentions | pronoun share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 906 | 24,885 | 49,090 | 6,798 | 18,087 | 170,991 | 19,151 | 39.01% |
| val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |
| test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 134 | 283 | 3,999 |
| val | 0 | 0 | 0 |
| test | 0 | 0 | 0 |

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

| split | mentions the string rule adds | arguments skipped, 2+ clusters | arguments in one multi-mention cluster | other mentions of that cluster | argument spans in 2+ clusters |
| --- | --- | --- | --- | --- | --- |
| train | 312 | 2 | 74 | 438 | 0 |
| val | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 |

### Entities

#### train

| label | count | share |
| --- | --- | --- |
| person | 4,627 | 8.63% |
| location | 2,837 | 5.29% |
| date | 2,455 | 4.58% |
| Location | 2,378 | 4.43% |
| Date | 2,257 | 4.21% |
| cardinal | 1,478 | 2.76% |
| organization | 1,449 | 2.70% |
| Organization | 1,326 | 2.47% |
| StartDate | 1,063 | 1.98% |
| job title | 976 | 1.82% |
| Product | 972 | 1.81% |
| EquityHolder | 947 | 1.77% |
| StockCode | 836 | 1.56% |
| Casualties and Losses | 781 | 1.46% |
| money | 715 | 1.33% |
| CompanyName | 709 | 1.32% |
| StockAbbr | 700 | 1.31% |
| geopolitical entity | 676 | 1.26% |
| PledgedShares | 636 | 1.19% |
| company | 597 | 1.11% |
| media outlet | 591 | 1.10% |
| Cause | 552 | 1.03% |
| Pledgee | 523 | 0.98% |
| TotalHoldingShares | 513 | 0.96% |
| TradedShares | 512 | 0.95% |
| sports team | 509 | 0.95% |
| Quantity | 494 | 0.92% |
| TotalHoldingRatio | 468 | 0.87% |
| Person | 454 | 0.85% |
| percent | 404 | 0.75% |
| Reduction Amount | 387 | 0.72% |
| TotalPledgedShares | 357 | 0.67% |
| facility | 345 | 0.64% |
| duration | 330 | 0.62% |
| work of art | 312 | 0.58% |
| occupation | 311 | 0.58% |
| Reducing Shareholder | 309 | 0.58% |
| age | 309 | 0.58% |
| product | 309 | 0.58% |
| EndDate | 299 | 0.56% |
| Measurement | 298 | 0.56% |
| time | 296 | 0.55% |
| ReleasedDate | 286 | 0.53% |
| quantity | 279 | 0.52% |
| government agency | 261 | 0.49% |
| Announcement Date | 219 | 0.41% |
| Company Name | 208 | 0.39% |
| Reduction Start Date | 206 | 0.38% |
| Contest Participant | 204 | 0.38% |
| educational institution | 188 | 0.35% |
| Money | 186 | 0.35% |
| Event | 170 | 0.32% |
| LaterHoldingShares | 167 | 0.31% |
| event | 166 | 0.31% |
| nationality | 164 | 0.31% |
| Policy Content | 161 | 0.30% |
| Participants | 155 | 0.29% |
| Facility | 152 | 0.28% |
| PER | 137 | 0.26% |
| Areas affected | 133 | 0.25% |
| Nationality | 130 | 0.24% |
| Candidates and their Political Parties | 129 | 0.24% |
| pledge_amount | 129 | 0.24% |
| Suspect | 127 | 0.24% |
| AveragePrice | 125 | 0.23% |
| Protesters | 121 | 0.23% |
| ordinal | 120 | 0.22% |
| Affected Areas | 118 | 0.22% |
| Attacker | 115 | 0.21% |
| Affected Area | 111 | 0.21% |
| crime | 106 | 0.20% |
| measurement | 103 | 0.19% |
| family relationship | 102 | 0.19% |
| Occupation | 102 | 0.19% |
| pledgor | 102 | 0.19% |
| brand | 101 | 0.19% |
| address | 98 | 0.18% |
| Person under Investigation | 96 | 0.18% |
| region | 94 | 0.18% |
| RepurchasedShares | 94 | 0.18% |
| Weapon and Equippment | 93 | 0.17% |
| Loss Amount | 92 | 0.17% |
| sports competition | 92 | 0.17% |
| Protest Reason | 92 | 0.17% |
| Recipient | 92 | 0.17% |
| Belligerents | 91 | 0.17% |
| Disease | 90 | 0.17% |
| disease | 90 | 0.17% |
| Charged Crime | 89 | 0.17% |
| HighestTradingPrice | 89 | 0.17% |
| Game Name | 85 | 0.16% |
| Accepting Court | 85 | 0.16% |
| distance | 84 | 0.16% |
| The Charged Crime | 84 | 0.16% |
| Individual_protein | 84 | 0.16% |
| Strike Outcome | 84 | 0.16% |
| temperature | 83 | 0.15% |
| Target | 83 | 0.15% |
| Investigative Agency | 79 | 0.15% |
| Deliberating Agency | 78 | 0.15% |
| Criminal Evidence | 71 | 0.13% |
| Position | 71 | 0.13% |
| film | 69 | 0.13% |
| RepurchaseAmount | 69 | 0.13% |
| protein | 66 | 0.12% |
| GeneOrGeneProduct | 66 | 0.12% |
| Method | 65 | 0.12% |
| Frozen Amount | 65 | 0.12% |
| Confirmed/Infected Cases | 64 | 0.12% |
| Precautionary Measure | 64 | 0.12% |
| Champions | 62 | 0.12% |
| political party | 62 | 0.12% |
| Strike Reason | 61 | 0.11% |
| Aid Agency | 61 | 0.11% |
| Score | 60 | 0.11% |
| Police | 60 | 0.11% |
| GPE | 60 | 0.11% |
| ORG | 60 | 0.11% |
| military rank | 59 | 0.11% |
| Influence People | 58 | 0.11% |
| DiseaseOrPhenotypicFeature | 58 | 0.11% |
| law | 57 | 0.11% |
| road | 57 | 0.11% |
| Arrest Location | 57 | 0.11% |
| ruling_date | 57 | 0.11% |
| Scheduled Landing Place | 56 | 0.10% |
| Summit Theme | 55 | 0.10% |
| Time | 55 | 0.10% |
| ClosingDate | 55 | 0.10% |
| Winner | 55 | 0.10% |
| Award | 55 | 0.10% |
| The Final Votes and Percentages | 54 | 0.10% |
| Accusation | 54 | 0.10% |
| Accused People | 53 | 0.10% |
| Flight No. | 51 | 0.10% |
| Accident Investigator | 51 | 0.10% |
| chemical | 50 | 0.09% |
| Storm Hit Location | 50 | 0.09% |
| landmark | 49 | 0.09% |
| Election Name | 48 | 0.09% |
| era | 48 | 0.09% |
| Death Cases | 48 | 0.09% |
| pledge_start_date | 48 | 0.09% |
| Vulnerability | 47 | 0.09% |
| Economic Loss | 46 | 0.09% |
| Visitor | 45 | 0.08% |
| Aircraft Agency | 45 | 0.08% |
| Anti-pollution People/Organizations | 45 | 0.08% |
| Contracting Parties | 45 | 0.08% |
| Location/Hospital | 45 | 0.08% |
| Strikers | 45 | 0.08% |
| academic degree | 44 | 0.08% |
| Host | 44 | 0.08% |
| ethnic group | 42 | 0.08% |
| Resignated Person | 42 | 0.08% |
| Policy Proposals | 42 | 0.08% |
| Historical Sites | 42 | 0.08% |
| Judgement Result/Prison Term | 41 | 0.08% |
| pledge_end_date | 41 | 0.08% |
| OtherScientificTerm | 41 | 0.08% |
| Increase Amount | 41 | 0.08% |
| Microorganism | 41 | 0.08% |
| Country Visited | 40 | 0.07% |
| award | 40 | 0.07% |
| political affiliation | 40 | 0.07% |
| Increasing Shareholder | 40 | 0.07% |
| frozen_shareholder | 40 | 0.07% |
| publication | 39 | 0.07% |
| Survivors | 39 | 0.07% |
| military unit | 38 | 0.07% |
| Epidemic Data Issuing Agency | 37 | 0.07% |
| LegalInstitution | 36 | 0.07% |
| Competition Items | 36 | 0.07% |
| Magnitude | 36 | 0.07% |
| Deceased | 36 | 0.07% |
| vehicle | 35 | 0.07% |
| book | 35 | 0.07% |
| software | 35 | 0.07% |
| Start Date | 35 | 0.07% |
| Capabilities | 35 | 0.07% |
| Damaged Facility | 34 | 0.06% |
| Freeze Start Date | 34 | 0.06% |
| honorific | 33 | 0.06% |
| non-profit organization | 33 | 0.06% |
| team role | 33 | 0.06% |
| Attending Hospital | 33 | 0.06% |
| LowestTradingPrice | 33 | 0.06% |
| Strikers Status | 33 | 0.06% |
| Outbreak Location | 32 | 0.06% |
| pseudonym | 32 | 0.06% |
| Record Breaker | 32 | 0.06% |
| Protest Slogan | 32 | 0.06% |
| Profession | 32 | 0.06% |
| Software | 32 | 0.06% |
| Agreement Content | 31 | 0.06% |
| Weapon | 31 | 0.06% |
| Pollution Source | 30 | 0.06% |
| Grades | 30 | 0.06% |
| Speaker | 30 | 0.06% |
| Number of Evacuated People | 30 | 0.06% |
| Passengers | 30 | 0.06% |
| Organization Name | 30 | 0.06% |
| Award Reason | 30 | 0.06% |
| Number of Damaged Houses | 29 | 0.05% |
| name | 29 | 0.05% |
| FrozeShares | 28 | 0.05% |
| Host Country | 28 | 0.05% |
| Policy Name & Abbreviation | 28 | 0.05% |
| Participating Countries | 28 | 0.05% |
| Habitat | 28 | 0.05% |
| Artifacts and Their Chronology | 28 | 0.05% |
| Susceptible Population | 27 | 0.05% |
| regulatory body | 27 | 0.05% |
| regulation | 27 | 0.05% |
| Riot Reason | 27 | 0.05% |
| ChemicalEntity | 27 | 0.05% |
| CellType | 27 | 0.05% |
| organism | 26 | 0.05% |
| song | 26 | 0.05% |
| Court | 26 | 0.05% |
| Resign Reason | 26 | 0.05% |
| Record-breaking Project | 26 | 0.05% |
| Effective Date | 26 | 0.05% |
| Anatomy | 26 | 0.05% |
| Time of the Case | 25 | 0.05% |
| Arrest Time | 25 | 0.05% |
| Start Time | 25 | 0.05% |
| Husband | 25 | 0.05% |
| Age | 25 | 0.05% |
| Aid Supplies/Amount | 25 | 0.05% |
| Employed Institution | 25 | 0.05% |
| Discover Location | 25 | 0.05% |
| Symptoms | 24 | 0.04% |
| Spacecraft | 24 | 0.04% |
| religious group | 24 | 0.04% |
| video game | 24 | 0.04% |
| Wife | 24 | 0.04% |
| Released People | 24 | 0.04% |
| Solution | 24 | 0.04% |
| Device | 24 | 0.04% |
| Way for Spreading | 23 | 0.04% |
| body of water | 23 | 0.04% |
| Evidence | 23 | 0.04% |
| Employment Agency | 23 | 0.04% |
| Crew | 23 | 0.04% |
| Epicenter | 23 | 0.04% |
| Number of Vehicles Involved in the Crash | 23 | 0.04% |
| OrganismTaxon | 23 | 0.04% |
| medication | 22 | 0.04% |
| frequency | 22 | 0.04% |
| species | 22 | 0.04% |
| Task | 22 | 0.04% |
| Generic | 22 | 0.04% |
| Storm Name | 22 | 0.04% |
| Speaker Status | 22 | 0.04% |
| Death Reason | 22 | 0.04% |
| Appointee | 22 | 0.04% |
| criminal organization | 21 | 0.04% |
| speed | 21 | 0.04% |
| weapon | 21 | 0.04% |
| Freeze End Date | 21 | 0.04% |
| Related Rivers or Lakes | 21 | 0.04% |
| System | 21 | 0.04% |
| File | 21 | 0.04% |
| FAC | 21 | 0.04% |
| People/Organization who predicted the disaster | 21 | 0.04% |
| Research Agency | 20 | 0.04% |
| symptom | 20 | 0.04% |
| holiday | 20 | 0.04% |
| Increase Start Date | 20 | 0.04% |
| Areas Affected | 20 | 0.04% |
| Defense Lawyer | 20 | 0.04% |
| Strike Agency | 20 | 0.04% |
| SequenceVariant | 20 | 0.04% |
| court case | 19 | 0.04% |
| Award Field | 19 | 0.04% |
| Summit Name | 19 | 0.04% |
| weight | 18 | 0.03% |
| Fined Agency | 18 | 0.03% |
| Previous Record Holder | 18 | 0.03% |
| Beneficiary | 18 | 0.03% |
| Maximum Wind Speed | 18 | 0.03% |
| protein-family | 18 | 0.03% |
| Spokenmen | 17 | 0.03% |
| Regulatory Authority | 17 | 0.03% |
| Affected Industries | 17 | 0.03% |
| Acquiree | 17 | 0.03% |
| News Release Agency | 17 | 0.03% |
| Spacecraft Mission | 16 | 0.03% |
| Party Responsible for Pollution | 16 | 0.03% |
| Fine Reason | 16 | 0.03% |
| Launch Date | 16 | 0.03% |
| Army | 16 | 0.03% |
| WEA | 16 | 0.03% |
| Achievement | 15 | 0.03% |
| operating system | 15 | 0.03% |
| anatomical structure | 15 | 0.03% |
| Material | 15 | 0.03% |
| People | 15 | 0.03% |
| The Final Seats Result | 15 | 0.03% |
| End Date | 15 | 0.03% |
| compensation_amount | 15 | 0.03% |
| Prosecutor | 15 | 0.03% |
| Head of Institution | 14 | 0.03% |
| Spokeswoman/Spokesman | 14 | 0.03% |
| Start_Date | 14 | 0.03% |
| Agreement Name | 14 | 0.03% |
| The Grades of the Previous Record Holder | 14 | 0.03% |
| Sponsor | 14 | 0.03% |
| Maximum Rainfall | 14 | 0.03% |
| Perpetrators | 14 | 0.03% |
| Election Goal | 14 | 0.03% |
| End Time | 14 | 0.03% |
| Arrested | 14 | 0.03% |
| PII | 14 | 0.03% |
| Boycotted Institutions | 14 | 0.03% |
| VEH | 14 | 0.03% |
| Patch | 14 | 0.03% |
| Medication | 14 | 0.03% |
| VAL | 14 | 0.03% |
| TTL | 14 | 0.03% |
| Taking-off Place | 13 | 0.02% |
| conference | 13 | 0.02% |
| Ship No. | 13 | 0.02% |
| Storm Hit Time | 13 | 0.02% |
| Purpose | 13 | 0.02% |
| description | 13 | 0.02% |
| medical procedure | 12 | 0.02% |
| area | 12 | 0.02% |
| Number of Influenced People | 12 | 0.02% |
| Funding | 12 | 0.02% |
| Rescue Organizer | 12 | 0.02% |
| Lasting Time | 12 | 0.02% |
| Duration | 12 | 0.02% |
| DNA | 12 | 0.02% |
| Missings | 12 | 0.02% |
| Storm Warning Level | 12 | 0.02% |
| Government Reaction | 12 | 0.02% |
| Discover Time | 12 | 0.02% |
| deadline | 11 | 0.02% |
| market index | 11 | 0.02% |
| fiscal quarter | 11 | 0.02% |
| aircraft | 11 | 0.02% |
| stock ticker | 11 | 0.02% |
| airport | 11 | 0.02% |
| Victim | 11 | 0.02% |
| Court Time | 11 | 0.02% |
| Conflict Duration | 11 | 0.02% |
| Head of Investigation Team | 11 | 0.02% |
| senior_personnel | 11 | 0.02% |
| senior_position | 11 | 0.02% |
| compensation_recipient | 11 | 0.02% |
| Website | 11 | 0.02% |
| Declarer | 11 | 0.02% |
| Archaeologist Organization | 11 | 0.02% |
| Reasons for the Formation of the Historical sites | 11 | 0.02% |
| Astronauts | 11 | 0.02% |
| Archaeologist | 11 | 0.02% |
| treaty | 10 | 0.02% |
| Successor | 10 | 0.02% |
| Military Strength | 10 | 0.02% |
| Bank Name | 10 | 0.02% |
| Region | 10 | 0.02% |
| Treatment Method | 10 | 0.02% |
| Other Loss | 10 | 0.02% |
| Types of the Spectacle | 10 | 0.02% |
| Water Level | 10 | 0.02% |
| Organization Industry | 10 | 0.02% |
| Prosecution Lawyer | 10 | 0.02% |
| Amount of Precipitation | 10 | 0.02% |
| Number of People Hospitalized | 10 | 0.02% |
| The Agency Who Broke the Agreement | 10 | 0.02% |
| GeneProtein | 10 | 0.02% |
| model number | 9 | 0.02% |
| Perpetrator | 9 | 0.02% |
| Launch Country | 9 | 0.02% |
| Scale | 9 | 0.02% |
| Commanders and their Position | 9 | 0.02% |
| Turnout(who win who lost) | 9 | 0.02% |
| Predecessor | 9 | 0.02% |
| Jail Time | 9 | 0.02% |
| Version | 9 | 0.02% |
| Data | 9 | 0.02% |
| Approver | 9 | 0.02% |
| Illness | 9 | 0.02% |
| Group | 9 | 0.02% |
| Symptom | 9 | 0.02% |
| Malware | 9 | 0.02% |
| LOC | 9 | 0.02% |
| Condition | 9 | 0.02% |
| id | 9 | 0.02% |
| Complications | 9 | 0.02% |
| Species | 9 | 0.02% |
| dosage | 8 | 0.01% |
| war | 8 | 0.01% |
| clinical trial | 8 | 0.01% |
| hardware component | 8 | 0.01% |
| Child | 8 | 0.01% |
| Shipwreck Reason | 8 | 0.01% |
| Countries Joined the Organization | 8 | 0.01% |
| Ways to Watch the Speech | 8 | 0.01% |
| Death Time | 8 | 0.01% |
| Number of Destroyed Building | 8 | 0.01% |
| Strike Industry | 8 | 0.01% |
| Last Job of the Appointee | 8 | 0.01% |
| Appointer | 8 | 0.01% |
| Appointment Time | 8 | 0.01% |
| Release Time | 8 | 0.01% |
| Age of the Resignated Person | 8 | 0.01% |
| Bill Drafting Agency | 8 | 0.01% |
| Pests | 8 | 0.01% |
| Carrier Rocket | 8 | 0.01% |
| Release Reason | 8 | 0.01% |
| vessel | 7 | 0.01% |
| currency | 7 | 0.01% |
| Judge | 7 | 0.01% |
| Name | 7 | 0.01% |
| Phenotype | 7 | 0.01% |
| Spectacle Location | 7 | 0.01% |
| Aid Reason | 7 | 0.01% |
| Military Exercise | 7 | 0.01% |
| Last Time the Record was Broken | 7 | 0.01% |
| Concept | 7 | 0.01% |
| Investigating agency | 7 | 0.01% |
| Poisoning Type | 7 | 0.01% |
| Number of Victims | 7 | 0.01% |
| Storm Direction | 7 | 0.01% |
| MVP | 7 | 0.01% |
| Rescue Tool or Method | 7 | 0.01% |
| Response Measures | 7 | 0.01% |
| model | 7 | 0.01% |
| tags | 7 | 0.01% |
| contract | 6 | 0.01% |
| Penalty Amount | 6 | 0.01% |
| Witness | 6 | 0.01% |
| license | 6 | 0.01% |
| Metric | 6 | 0.01% |
| Launch Result | 6 | 0.01% |
| Cooperative Agency | 6 | 0.01% |
| Publication | 6 | 0.01% |
| Disaster-Stricken Farmland | 6 | 0.01% |
| Commanders of the Army | 6 | 0.01% |
| The Sentence Claimed by the Prosecutor's Lawyer | 6 | 0.01% |
| Gene/protein/RNA | 6 | 0.01% |
| People Casting Key Votes | 6 | 0.01% |
| Prison Term | 6 | 0.01% |
| Hostage | 6 | 0.01% |
| CellLine | 6 | 0.01% |
| Countries Withdrawing from the Organization | 6 | 0.01% |
| Doctor and Medical Team | 6 | 0.01% |
| Head of the Merged Organization | 6 | 0.01% |
| Acquirer | 6 | 0.01% |
| Death Rate | 6 | 0.01% |
| Term of Office | 6 | 0.01% |
| Protein_family_or_group | 6 | 0.01% |
| workCenter | 6 | 0.01% |
| Area Affected | 6 | 0.01% |
| Damaged Crops & Livestock | 6 | 0.01% |
| Train No. | 6 | 0.01% |
| Medical Condition | 6 | 0.01% |
| Launch Site | 5 | 0.01% |
| UnfrozeDate | 5 | 0.01% |
| Description | 5 | 0.01% |
| Country | 5 | 0.01% |
| Candidate Scandal(Allegations of fraud, etc.) | 5 | 0.01% |
| Annoncement Date | 5 | 0.01% |
| Acquisition amount | 5 | 0.01% |
| Voting method | 5 | 0.01% |
| Outbreak Date | 5 | 0.01% |
| Volcano Name | 5 | 0.01% |
| protein-complex | 5 | 0.01% |
| State Before Death | 5 | 0.01% |
| Train Agency | 5 | 0.01% |
| Spectacle Start Time | 5 | 0.01% |
| Sentencing Location | 5 | 0.01% |
| Economists who Predicted the Crisis | 5 | 0.01% |
| Casualty Count | 5 | 0.01% |
| city | 5 | 0.01% |
| state | 5 | 0.01% |
| architecture_component | 5 | 0.01% |
| role | 5 | 0.01% |
| Economic loss | 5 | 0.01% |
| Art Form | 5 | 0.01% |
| Number of Vaccinated People | 5 | 0.01% |
| Warning Device | 5 | 0.01% |
| Influenced People | 5 | 0.01% |
| Programming Language | 5 | 0.01% |
| Tic Type | 5 | 0.01% |
| pathogen | 4 | 0.01% |
| Mission Duration | 4 | 0.01% |
| witness | 4 | 0.01% |
| Casualities and Losses | 4 | 0.01% |
| process | 4 | 0.01% |
| Temporary Settlement | 4 | 0.01% |
| Postpone Reason | 4 | 0.01% |
| Goal | 4 | 0.01% |
| Organization Members | 4 | 0.01% |
| mutation | 4 | 0.01% |
| RNA | 4 | 0.01% |
| Alternate Landing Place | 4 | 0.01% |
| protein-domain | 4 | 0.01% |
| Electoral System | 4 | 0.01% |
| CVE | 4 | 0.01% |
| unitOfMeasure | 4 | 0.01% |
| leadTime | 4 | 0.01% |
| Number of Rescued People | 4 | 0.01% |
| Organization Leader | 4 | 0.01% |
| Art Movement | 4 | 0.01% |
| Number of Aftershocks | 4 | 0.01% |
| Production Cuts | 4 | 0.01% |
| Cured Cases | 4 | 0.01% |
| postal code | 3 | 0.01% |
| natural disaster | 3 | 0.01% |
| tax | 3 | 0.01% |
| Marriage Duration | 3 | 0.01% |
| Weapon Used | 3 | 0.01% |
| Detention Start Time | 3 | 0.01% |
| Forecasting Agency | 3 | 0.01% |
| cell | 3 | 0.01% |
| Gene | 3 | 0.01% |
| Head of the Institution | 3 | 0.01% |
| Withdraw Reason | 3 | 0.01% |
| Number of Trapped People | 3 | 0.01% |
| Treatment | 3 | 0.01% |
| operationNumber | 3 | 0.01% |
| standardTime | 3 | 0.01% |
| Inauguration Time | 3 | 0.01% |
| Agreement Validity Period | 3 | 0.01% |
| Object | 3 | 0.01% |
| type | 3 | 0.01% |
| title | 3 | 0.01% |
| Company Industry | 3 | 0.01% |
| Tear Up Reason | 3 | 0.01% |
| zip | 3 | 0.01% |
| Structure | 3 | 0.01% |
| Software Component | 3 | 0.01% |
| Operating System | 3 | 0.01% |
| allergen | 2 | 0.00% |
| trade union | 2 | 0.00% |
| mountain | 2 | 0.00% |
| patent | 2 | 0.00% |
| voltage | 2 | 0.00% |
| security vulnerability | 2 | 0.00% |
| language | 2 | 0.00% |
| Trapped Days | 2 | 0.00% |
| file format | 2 | 0.00% |
| Transportation | 2 | 0.00% |
| StreetAddress | 2 | 0.00% |
| ZipCode | 2 | 0.00% |
| TelephoneNumber | 2 | 0.00% |
| Email | 2 | 0.00% |
| LanguageCode | 2 | 0.00% |
| Stolen Amount | 2 | 0.00% |
| Best Way to Shoot | 2 | 0.00% |
| reagent | 2 | 0.00% |
| experimental-construct | 2 | 0.00% |
| country | 2 | 0.00% |
| Special Medicine | 2 | 0.00% |
| Merger Terms | 2 | 0.00% |
| Bankrupt Business | 2 | 0.00% |
| Provisional Government | 2 | 0.00% |
| Number | 2 | 0.00% |
| Organism | 2 | 0.00% |
| Substance | 2 | 0.00% |
| Complication | 2 | 0.00% |
| Test | 2 | 0.00% |
| Influenced Crops and Livelihood | 2 | 0.00% |
| Service Years | 2 | 0.00% |
| Other Impact | 2 | 0.00% |
| Registered Capital | 2 | 0.00% |
| partNumber | 2 | 0.00% |
| standardCost | 2 | 0.00% |
| listPrice | 2 | 0.00% |
| reorderPoint | 2 | 0.00% |
| safetyStockLevel | 2 | 0.00% |
| locationCode | 2 | 0.00% |
| zipCode | 2 | 0.00% |
| workOrderNumber | 2 | 0.00% |
| dueDate | 2 | 0.00% |
| status | 2 | 0.00% |
| setupTime | 2 | 0.00% |
| evaluation_metric | 2 | 0.00% |
| contactNumber | 2 | 0.00% |
| Play Type | 2 | 0.00% |
| assay | 2 | 0.00% |
| url | 2 | 0.00% |
| Time Period | 2 | 0.00% |
| Political Entity | 2 | 0.00% |
| Writing System | 2 | 0.00% |
| firstName | 2 | 0.00% |
| lastName | 2 | 0.00% |
| startDate | 2 | 0.00% |
| endDate | 2 | 0.00% |
| email | 2 | 0.00% |
| phone | 2 | 0.00% |
| Military_Unit | 2 | 0.00% |
| Head of the Government | 2 | 0.00% |
| Medical Test | 2 | 0.00% |
| CellComponent | 2 | 0.00% |
| supplier | 2 | 0.00% |
| cost | 2 | 0.00% |
| verdict | 1 | 0.00% |
| sentence | 1 | 0.00% |
| mineral | 1 | 0.00% |
| Organization_Military | 1 | 0.00% |
| Damaged Property | 1 | 0.00% |
| vaccine | 1 | 0.00% |
| PaymentMethod | 1 | 0.00% |
| Announce Platform | 1 | 0.00% |
| PlanType | 1 | 0.00% |
| ResiliencyLevel | 1 | 0.00% |
| City | 1 | 0.00% |
| State | 1 | 0.00% |
| ParentName | 1 | 0.00% |
| Year | 1 | 0.00% |
| Projects | 1 | 0.00% |
| Live Broadcast Platform | 1 | 0.00% |
| seconds | 1 | 0.00% |
| userId | 1 | 0.00% |
| latitude | 1 | 0.00% |
| longitude | 1 | 0.00% |
| appId | 1 | 0.00% |
| appName | 1 | 0.00% |
| appVersion | 1 | 0.00% |
| networkType | 1 | 0.00% |
| appLanguage | 1 | 0.00% |
| Spectacle Duration | 1 | 0.00% |
| Statistic | 1 | 0.00% |
| Definition | 1 | 0.00% |
| Language | 1 | 0.00% |
| Technology | 1 | 0.00% |
| Child Custody | 1 | 0.00% |
| Last Outbreak Time | 1 | 0.00% |
| Refuge | 1 | 0.00% |
| organelle | 1 | 0.00% |
| Exit Conditions | 1 | 0.00% |
| Trapped Depth | 1 | 0.00% |
| Storm Formation Time | 1 | 0.00% |
| Storm Movement Speed | 1 | 0.00% |
| Hull Location | 1 | 0.00% |
| Spectacle End Time | 1 | 0.00% |
| Wedding Venue | 1 | 0.00% |
| The Sentence Claimed by the Defense Lawyer | 1 | 0.00% |
| Age of the Appointee | 1 | 0.00% |
| training_method | 1 | 0.00% |
| mathematical_framework | 1 | 0.00% |
| dataset | 1 | 0.00% |
| hardware | 1 | 0.00% |
| data_representation | 1 | 0.00% |
| group | 1 | 0.00% |
| dateOfBirth | 1 | 0.00% |
| qualification | 1 | 0.00% |
| route | 1 | 0.00% |
| dispensingDate | 1 | 0.00% |
| quantityDispensed | 1 | 0.00% |
| refillsAllowed | 1 | 0.00% |
| instructions | 1 | 0.00% |
| signature | 1 | 0.00% |
| Postpone Time | 1 | 0.00% |
| coverArtTitle | 1 | 0.00% |
| coverArtImage | 1 | 0.00% |
| coverArtDescription | 1 | 0.00% |
| series | 1 | 0.00% |
| issueNumber | 1 | 0.00% |
| publicationDate | 1 | 0.00% |
| publisher | 1 | 0.00% |
| Fire Warning Level | 1 | 0.00% |
| Activity | 1 | 0.00% |
| Book | 1 | 0.00% |
| Behavior | 1 | 0.00% |
| Resource | 1 | 0.00% |
| creator | 1 | 0.00% |
| network | 1 | 0.00% |
| season | 1 | 0.00% |
| episode | 1 | 0.00% |
| source | 1 | 0.00% |
| rating | 1 | 0.00% |
| Film | 1 | 0.00% |
| hotelId | 1 | 0.00% |
| roomId | 1 | 0.00% |
| numNights | 1 | 0.00% |
| numGuests | 1 | 0.00% |
| Economy Loss | 1 | 0.00% |
| Tsunamis | 1 | 0.00% |
| Property Division | 1 | 0.00% |
| Magnitude(Tsunami heights) | 1 | 0.00% |
| ABS | 1 | 0.00% |
| COM | 1 | 0.00% |
| INF | 1 | 0.00% |
| reservationId | 1 | 0.00% |
| code | 1 | 0.00% |
| number | 1 | 0.00% |
| make | 1 | 0.00% |
| year | 1 | 0.00% |
| licensePlateNumber | 1 | 0.00% |
| additionalInstructions | 1 | 0.00% |
| cardNumber | 1 | 0.00% |
| expirationDate | 1 | 0.00% |
| cvv | 1 | 0.00% |
| productId | 1 | 0.00% |
| street | 1 | 0.00% |
| lastUpdatedDate | 1 | 0.00% |
| Time for Dignitaries to Resign | 1 | 0.00% |
| Hull Discovery Time | 1 | 0.00% |
| roomNumber | 1 | 0.00% |
| roomType | 1 | 0.00% |
| issuedAt | 1 | 0.00% |
| expiresAt | 1 | 0.00% |
| Crime | 1 | 0.00% |
| Organization/Person | 1 | 0.00% |
| Infrastructure | 1 | 0.00% |
| Term | 1 | 0.00% |
| Person/Group | 1 | 0.00% |
| Sequelae | 1 | 0.00% |
| Standard | 1 | 0.00% |
| Software Version | 1 | 0.00% |
| Database | 1 | 0.00% |
| Unemployed Rate | 1 | 0.00% |
| Classification | 1 | 0.00% |
| Cured Rate | 1 | 0.00% |
| Storm Center Location | 1 | 0.00% |
| financial instrument | 0 | 0.00% |
| programming language | 0 | 0.00% |
| api | 0 | 0.00% |
| algorithm | 0 | 0.00% |
| diagnosis | 0 | 0.00% |
| budget line | 0 | 0.00% |
| database | 0 | 0.00% |
| framework | 0 | 0.00% |
| protocol | 0 | 0.00% |
| interest rate | 0 | 0.00% |
| gene | 0 | 0.00% |
| credit rating | 0 | 0.00% |
| border | 0 | 0.00% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| no relation | 366 | 9.06% |
| works_for | 193 | 4.78% |
| member_of | 155 | 3.84% |
| located_in | 125 | 3.09% |
| Association | 100 | 2.48% |
| PPI | 92 | 2.28% |
| affiliated_with | 91 | 2.25% |
| employer_of | 81 | 2.01% |
| competitor_of | 79 | 1.96% |
| based_in | 75 | 1.86% |
| leader_of | 72 | 1.78% |
| headquartered_in | 63 | 1.56% |
| bind | 62 | 1.54% |
| part_of | 60 | 1.49% |
| Lives_In | 54 | 1.34% |
| shares border with | 51 | 1.26% |
| instance of | 48 | 1.19% |
| USED-FOR | 44 | 1.09% |
| occupation | 39 | 0.97% |
| founded | 38 | 0.94% |
| given name | 37 | 0.92% |
| produced_by | 33 | 0.82% |
| family name | 33 | 0.82% |
| participated_in | 33 | 0.82% |
| operates_in | 33 | 0.82% |
| country | 26 | 0.64% |
| place of birth | 25 | 0.62% |
| acquired | 24 | 0.59% |
| contains the administrative territorial entity | 23 | 0.57% |
| member of | 23 | 0.57% |
| located in the administrative territorial entity | 22 | 0.54% |
| Positive_Correlation | 22 | 0.54% |
| colleague_of | 21 | 0.52% |
| supplier_of | 20 | 0.50% |
| replaces | 18 | 0.45% |
| country of citizenship | 17 | 0.42% |
| genre | 16 | 0.40% |
| has part(s) | 15 | 0.37% |
| director | 15 | 0.37% |
| Negative_Correlation | 15 | 0.37% |
| continent | 14 | 0.35% |
| child | 14 | 0.35% |
| reports_to | 14 | 0.35% |
| educated at | 13 | 0.32% |
| location | 13 | 0.32% |
| parent_of | 13 | 0.32% |
| child_of | 13 | 0.32% |
| founded by | 12 | 0.30% |
| secretary general | 12 | 0.30% |
| developed_by | 12 | 0.30% |
| board_member_of | 12 | 0.30% |
| architect | 11 | 0.27% |
| place of death | 11 | 0.27% |
| notable work | 11 | 0.27% |
| EVALUATE-FOR | 11 | 0.27% |
| relative_of | 11 | 0.27% |
| field of work | 11 | 0.27% |
| father | 11 | 0.27% |
| significant event | 11 | 0.27% |
| composed_by | 10 | 0.25% |
| sibling | 10 | 0.25% |
| located in/on physical feature | 10 | 0.25% |
| participant | 10 | 0.25% |
| represents | 10 | 0.25% |
| authored_by | 10 | 0.25% |
| member of sports team | 10 | 0.25% |
| CONJUNCTION | 10 | 0.25% |
| has use | 9 | 0.22% |
| creator | 9 | 0.22% |
| award received | 9 | 0.22% |
| place of detention | 9 | 0.22% |
| cast member | 9 | 0.22% |
| spouse | 9 | 0.22% |
| FEATURE-OF | 9 | 0.22% |
| languages spoken, written or signed | 9 | 0.22% |
| position held | 9 | 0.22% |
| performer | 9 | 0.22% |
| subsidiary_of | 9 | 0.22% |
| employer | 9 | 0.22% |
| named after | 8 | 0.20% |
| online service | 8 | 0.20% |
| published_by | 8 | 0.20% |
| characters | 8 | 0.20% |
| country of origin | 8 | 0.20% |
| significant person | 8 | 0.20% |
| author | 8 | 0.20% |
| present in work | 8 | 0.20% |
| military branch | 8 | 0.20% |
| partner in business or sport | 8 | 0.20% |
| mountain range | 8 | 0.20% |
| organizer | 8 | 0.20% |
| applies to jurisdiction | 8 | 0.20% |
| sibling_of | 8 | 0.20% |
| member of political party | 7 | 0.17% |
| citizen_of | 7 | 0.17% |
| advisor_to | 7 | 0.17% |
| connecting line | 7 | 0.17% |
| family | 7 | 0.17% |
| conflict | 7 | 0.17% |
| located on astronomical body | 7 | 0.17% |
| mouth of the watercourse | 7 | 0.17% |
| parent organization | 7 | 0.17% |
| said to be the same as | 7 | 0.17% |
| basin country | 7 | 0.17% |
| language used | 7 | 0.17% |
| developer | 7 | 0.17% |
| capital | 7 | 0.17% |
| spouse_of | 7 | 0.17% |
| different from | 7 | 0.17% |
| carries | 6 | 0.15% |
| designed_by | 6 | 0.15% |
| original broadcaster | 6 | 0.15% |
| located in protected area | 6 | 0.15% |
| position played on team / speciality | 6 | 0.15% |
| connects with | 6 | 0.15% |
| directed_by | 6 | 0.15% |
| movement | 6 | 0.15% |
| head of government | 6 | 0.15% |
| nominated for | 6 | 0.15% |
| convicted of | 6 | 0.15% |
| owned by | 6 | 0.15% |
| product or material produced | 6 | 0.15% |
| party_to | 6 | 0.15% |
| native language | 6 | 0.15% |
| production company | 6 | 0.15% |
| has effect | 6 | 0.15% |
| follows | 6 | 0.15% |
| instrument | 6 | 0.15% |
| opposite of | 6 | 0.15% |
| surface played on | 6 | 0.15% |
| headquarters location | 6 | 0.15% |
| licensed to broadcast to | 6 | 0.15% |
| occupant | 6 | 0.15% |
| designed by | 6 | 0.15% |
| has subsidiary | 6 | 0.15% |
| head coach | 5 | 0.12% |
| location of formation | 5 | 0.12% |
| heritage designation | 5 | 0.12% |
| ammunition | 5 | 0.12% |
| part of | 5 | 0.12% |
| facet of | 5 | 0.12% |
| Exhibits | 5 | 0.12% |
| platform | 5 | 0.12% |
| followed by | 5 | 0.12% |
| form of creative work | 5 | 0.12% |
| operating system | 5 | 0.12% |
| has quality | 5 | 0.12% |
| participant of | 5 | 0.12% |
| location of first performance | 5 | 0.12% |
| reservoir created | 5 | 0.12% |
| found in taxon | 5 | 0.12% |
| has works in the collection | 5 | 0.12% |
| uses | 5 | 0.12% |
| language of work or name | 5 | 0.12% |
| located on terrain feature | 5 | 0.12% |
| operator | 5 | 0.12% |
| based on | 5 | 0.12% |
| owns | 5 | 0.12% |
| takes place in fictional universe | 5 | 0.12% |
| located in the present-day administrative territorial entity | 5 | 0.12% |
| work location | 5 | 0.12% |
| illustrator | 5 | 0.12% |
| candidacy in election | 5 | 0.12% |
| regulated_by | 5 | 0.12% |
| victory | 5 | 0.12% |
| item operated | 5 | 0.12% |
| fictional universe described in | 5 | 0.12% |
| winner | 5 | 0.12% |
| located in or next to body of water | 5 | 0.12% |
| country for sport | 5 | 0.12% |
| sex or gender | 4 | 0.10% |
| stock exchange | 4 | 0.10% |
| made from material | 4 | 0.10% |
| PART-OF | 4 | 0.10% |
| COMPARE | 4 | 0.10% |
| participant in | 4 | 0.10% |
| anatomical location | 4 | 0.10% |
| historical region | 4 | 0.10% |
| member of the crew of | 4 | 0.10% |
| home venue | 4 | 0.10% |
| target | 4 | 0.10% |
| stock market index | 4 | 0.10% |
| replaced by | 4 | 0.10% |
| head of state | 4 | 0.10% |
| presenter | 4 | 0.10% |
| voice type | 4 | 0.10% |
| industry | 4 | 0.10% |
| vehicle | 4 | 0.10% |
| editor | 4 | 0.10% |
| league | 4 | 0.10% |
| discoverer or inventor | 4 | 0.10% |
| mother | 4 | 0.10% |
| composer | 4 | 0.10% |
| ancestral home | 4 | 0.10% |
| residence | 4 | 0.10% |
| commander of (DEPRECATED) | 4 | 0.10% |
| relative | 4 | 0.10% |
| member_state_of | 4 | 0.10% |
| main subject | 4 | 0.10% |
| place of burial | 4 | 0.10% |
| subclass of | 4 | 0.10% |
| terminus location | 4 | 0.10% |
| biological process | 4 | 0.10% |
| military rank | 4 | 0.10% |
| located on street | 4 | 0.10% |
| contributor to the creative work or subject | 4 | 0.10% |
| commemorates | 4 | 0.10% |
| cause of death | 4 | 0.10% |
| record label | 4 | 0.10% |
| location of discovery | 4 | 0.10% |
| writing language | 4 | 0.10% |
| significant place | 4 | 0.10% |
| religion or worldview | 4 | 0.10% |
| noble title | 4 | 0.10% |
| student of | 4 | 0.10% |
| named_after | 3 | 0.07% |
| origin of the watercourse | 3 | 0.07% |
| school district | 3 | 0.07% |
| located in time zone | 3 | 0.07% |
| dialect of | 3 | 0.07% |
| space launch vehicle | 3 | 0.07% |
| symptoms and signs | 3 | 0.07% |
| depicts | 3 | 0.07% |
| sponsor | 3 | 0.07% |
| domain of saint or deity | 3 | 0.07% |
| cathedral | 3 | 0.07% |
| publisher | 3 | 0.07% |
| flag | 3 | 0.07% |
| astronaut mission | 3 | 0.07% |
| field of this occupation | 3 | 0.07% |
| has immediate cause | 3 | 0.07% |
| crosses | 3 | 0.07% |
| genetic association | 3 | 0.07% |
| unmarried partner | 3 | 0.07% |
| manufactured_by | 3 | 0.07% |
| published in | 3 | 0.07% |
| choreographer | 3 | 0.07% |
| original language of film or TV show | 3 | 0.07% |
| mascot | 3 | 0.07% |
| structural engineer | 3 | 0.07% |
| dam | 3 | 0.07% |
| partially coincident with | 3 | 0.07% |
| physically interacts with | 3 | 0.07% |
| pole position | 3 | 0.07% |
| legislated by | 3 | 0.07% |
| this taxon is source of | 3 | 0.07% |
| tributary | 3 | 0.07% |
| recommended unit of measurement | 3 | 0.07% |
| IUCN conservation status | 3 | 0.07% |
| place of publication | 3 | 0.07% |
| has part(s) of the class | 3 | 0.07% |
| distributed by | 3 | 0.07% |
| measures | 3 | 0.07% |
| derivative work | 3 | 0.07% |
| inflows | 3 | 0.07% |
| grants | 3 | 0.07% |
| canonization status | 3 | 0.07% |
| producer | 3 | 0.07% |
| medical condition | 3 | 0.07% |
| manufacturer | 3 | 0.07% |
| anthem | 3 | 0.07% |
| cause | 3 | 0.07% |
| vessel class | 3 | 0.07% |
| radio format | 3 | 0.07% |
| brand | 3 | 0.07% |
| collection | 3 | 0.07% |
| sport | 3 | 0.07% |
| librettist | 3 | 0.07% |
| part of the series | 3 | 0.07% |
| engine configuration | 3 | 0.07% |
| medical condition treated | 3 | 0.07% |
| offers view on | 3 | 0.07% |
| terminus | 3 | 0.07% |
| drug or therapy used for treatment | 3 | 0.07% |
| place served by transport hub | 3 | 0.07% |
| copyright holder | 3 | 0.07% |
| endemic to | 3 | 0.07% |
| owner of | 3 | 0.07% |
| HYPONYM-OF | 3 | 0.07% |
| edition or translation of | 2 | 0.05% |
| house publication | 2 | 0.05% |
| business model | 2 | 0.05% |
| cause of destruction | 2 | 0.05% |
| dual to | 2 | 0.05% |
| coextensive with | 2 | 0.05% |
| history of topic | 2 | 0.05% |
| source of energy | 2 | 0.05% |
| country of registry | 2 | 0.05% |
| foundational text | 2 | 0.05% |
| is the study of | 2 | 0.05% |
| candidate | 2 | 0.05% |
| main regulatory text | 2 | 0.05% |
| antiparticle | 2 | 0.05% |
| wheel arrangement | 2 | 0.05% |
| animal breed | 2 | 0.05% |
| officeholder | 2 | 0.05% |
| airline hub | 2 | 0.05% |
| allegiance | 2 | 0.05% |
| parent astronomical body | 2 | 0.05% |
| maintained by | 2 | 0.05% |
| culture | 2 | 0.05% |
| type of variable star | 2 | 0.05% |
| chief executive officer | 2 | 0.05% |
| narrative location | 2 | 0.05% |
| borders | 2 | 0.05% |
| species kept | 2 | 0.05% |
| sports discipline competed in | 2 | 0.05% |
| game mode | 2 | 0.05% |
| adjacent station | 2 | 0.05% |
| archives at | 2 | 0.05% |
| product certification | 2 | 0.05% |
| business division | 2 | 0.05% |
| contributed to creative work | 2 | 0.05% |
| territory overlaps | 2 | 0.05% |
| statistical leader | 2 | 0.05% |
| constellation | 2 | 0.05% |
| practiced by | 2 | 0.05% |
| chairperson | 2 | 0.05% |
| diocese | 2 | 0.05% |
| taxon rank | 2 | 0.05% |
| color | 2 | 0.05% |
| connecting service | 2 | 0.05% |
| distribution format | 2 | 0.05% |
| religious order | 2 | 0.05% |
| substitute/deputy/replacement of office/officeholder | 2 | 0.05% |
| official language | 2 | 0.05% |
| has edition or translation | 2 | 0.05% |
| capital of | 2 | 0.05% |
| standards body | 2 | 0.05% |
| dedicated to | 2 | 0.05% |
| health specialty | 2 | 0.05% |
| commanded by | 2 | 0.05% |
| general classification of race participants | 2 | 0.05% |
| consecrator | 2 | 0.05% |
| musical conductor | 2 | 0.05% |
| statement describes | 2 | 0.05% |
| director of archaeological fieldwork | 2 | 0.05% |
| track gauge | 2 | 0.05% |
| readable file format | 2 | 0.05% |
| manner of death | 2 | 0.05% |
| speaker | 2 | 0.05% |
| home port | 2 | 0.05% |
| input method | 2 | 0.05% |
| voice actor | 2 | 0.05% |
| co-driver | 2 | 0.05% |
| separated from | 2 | 0.05% |
| filming location | 2 | 0.05% |
| sports season of league or competition | 2 | 0.05% |
| child astronomical body | 2 | 0.05% |
| first appearance | 2 | 0.05% |
| airline alliance | 2 | 0.05% |
| day in year for periodic occurrence | 2 | 0.05% |
| interested in | 2 | 0.05% |
| merged into | 2 | 0.05% |
| opposed_to | 2 | 0.05% |
| league level above | 2 | 0.05% |
| court | 2 | 0.05% |
| director of photography | 2 | 0.05% |
| sponsored_by | 2 | 0.05% |
| competition class | 2 | 0.05% |
| tracklist | 2 | 0.05% |
| political ideology | 2 | 0.05% |
| has written for | 2 | 0.05% |
| has fruit type | 2 | 0.05% |
| sold_to | 2 | 0.05% |
| office held by head of government | 2 | 0.05% |
| parent club | 2 | 0.05% |
| adjacent_to | 2 | 0.05% |
| league system | 2 | 0.05% |
| historic county | 2 | 0.05% |
| board member | 1 | 0.02% |
| exemplar of | 1 | 0.02% |
| Christian liturgical rite | 1 | 0.02% |
| Gram staining | 1 | 0.02% |
| donated by | 1 | 0.02% |
| office held by head of the organization | 1 | 0.02% |
| charge | 1 | 0.02% |
| coach of sports team | 1 | 0.02% |
| microstructure | 1 | 0.02% |
| signatory | 1 | 0.02% |
| has boundary | 1 | 0.02% |
| party chief representative | 1 | 0.02% |
| set during recurring event | 1 | 0.02% |
| interaction | 1 | 0.02% |
| title of chess person | 1 | 0.02% |
| amended by | 1 | 0.02% |
| legal form | 1 | 0.02% |
| won | 1 | 0.02% |
| died_in | 1 | 0.02% |
| recorded at studio or venue | 1 | 0.02% |
| contains settlement | 1 | 0.02% |
| regulated by | 1 | 0.02% |
| afflicts | 1 | 0.02% |
| sued_by | 1 | 0.02% |
| sidekick of | 1 | 0.02% |
| enemy of | 1 | 0.02% |
| copyright license | 1 | 0.02% |
| electoral district | 1 | 0.02% |
| has natural reservoir | 1 | 0.02% |
| media franchise | 1 | 0.02% |
| customer_of | 1 | 0.02% |
| narrative role | 1 | 0.02% |
| measurement scale | 1 | 0.02% |
| has pet | 1 | 0.02% |
| manifestation of | 1 | 0.02% |
| distributor_of | 1 | 0.02% |
| perpetrator | 1 | 0.02% |
| depicted by | 1 | 0.02% |
| ethnic group | 1 | 0.02% |
| architectural style | 1 | 0.02% |
| relative to | 1 | 0.02% |
| complies with | 1 | 0.02% |
| director / manager | 1 | 0.02% |
| broadcast by | 1 | 0.02% |
| model item | 1 | 0.02% |
| born_in | 1 | 0.02% |
| chromosome | 1 | 0.02% |
| taxon synonym | 1 | 0.02% |
| diaspora | 1 | 0.02% |
| represented by | 1 | 0.02% |
| penalty | 1 | 0.02% |
| incarnation of | 1 | 0.02% |
| has census | 1 | 0.02% |
| highest point | 1 | 0.02% |
| feast day | 1 | 0.02% |
| basic form of government | 1 | 0.02% |
| drafted by | 1 | 0.02% |
| next crossing upstream | 1 | 0.02% |
| reward program | 1 | 0.02% |
| cell component | 1 | 0.02% |
| wing configuration | 1 | 0.02% |
| league level below | 1 | 0.02% |
| shape | 1 | 0.02% |
| character type | 1 | 0.02% |
| overlies | 1 | 0.02% |
| season of club or team | 1 | 0.02% |
| has cause | 1 | 0.02% |
| supports programming language | 1 | 0.02% |
| executive producer | 1 | 0.02% |
| conjugate acid | 1 | 0.02% |
| emulates | 1 | 0.02% |
| enclave within | 1 | 0.02% |
| legislative body | 1 | 0.02% |
| set in environment | 1 | 0.02% |
| capital_of | 1 | 0.02% |
| natural product of taxon | 1 | 0.02% |
| affiliation | 1 | 0.02% |
| taxonomic type | 1 | 0.02% |
| doctoral advisor | 1 | 0.02% |
| GUI toolkit or framework | 1 | 0.02% |
| has phenotype | 1 | 0.02% |
| style of karate | 1 | 0.02% |
| possible treatment | 1 | 0.02% |
| invested_in | 1 | 0.02% |
| prerequisite | 1 | 0.02% |
| home world | 1 | 0.02% |
| update method | 1 | 0.02% |
| after a work by | 1 | 0.02% |
| connector | 1 | 0.02% |
| Cotreatment | 1 | 0.02% |
| investor | 1 | 0.02% |
| package management system | 1 | 0.02% |
| time period | 1 | 0.02% |
| discovery method | 1 | 0.02% |
| political alignment | 1 | 0.02% |
| indigenous to | 1 | 0.02% |
| operating area | 1 | 0.02% |
| this zoological name is coordinate with | 1 | 0.02% |
| participating team | 1 | 0.02% |
| anatomical branch of | 1 | 0.02% |
| primary destinations | 1 | 0.02% |
| parent taxon | 1 | 0.02% |
| mountains classification | 1 | 0.02% |
| epithet | 1 | 0.02% |
| has contributing factor | 1 | 0.02% |
| subject has role | 1 | 0.02% |
| killed by | 1 | 0.02% |
| mod of | 1 | 0.02% |
| treats | 1 | 0.02% |
| order of battle | 1 | 0.02% |
| twinned administrative body | 1 | 0.02% |
| linguistic typology | 1 | 0.02% |
| conferred by | 1 | 0.02% |
| has goal | 1 | 0.02% |
| next higher rank | 1 | 0.02% |
| solved by | 1 | 0.02% |
| newspaper format | 1 | 0.02% |
| blocker of | 1 | 0.02% |
| objective of project or action | 1 | 0.02% |
| temporal range start | 1 | 0.02% |
| molecular function | 1 | 0.02% |
| location of creation | 1 | 0.02% |
| economy of topic | 1 | 0.02% |
| sexually homologous with | 1 | 0.02% |
| allied_with | 1 | 0.02% |
| proved by | 1 | 0.02% |
| designed to carry | 1 | 0.02% |
| game mechanics | 1 | 0.02% |
| parliamentary group | 1 | 0.02% |
| lake outflow | 1 | 0.02% |
| suggests the existence of | 1 | 0.02% |
| lyrics by | 1 | 0.02% |
| student | 1 | 0.02% |
| writing system | 1 | 0.02% |
| points classification | 1 | 0.02% |
| territory claimed by | 1 | 0.02% |
| measured physical quantity | 1 | 0.02% |
| executive body | 1 | 0.02% |
| has anatomical branch | 1 | 0.02% |
| mentor_of | 1 | 0.02% |
| academic degree | 1 | 0.02% |
| wears | 1 | 0.02% |
| instrumentation | 1 | 0.02% |
| madhhab | 1 | 0.02% |
| qualifies for event | 1 | 0.02% |
| production designer | 1 | 0.02% |
| from narrative universe | 1 | 0.02% |
| vehicle normally used | 1 | 0.02% |
| armament | 1 | 0.02% |
| produced by | 1 | 0.02% |
| lake on watercourse | 1 | 0.02% |
| drainage basin | 1 | 0.02% |
| currency | 1 | 0.02% |

### Classifications

**train, actionability.** Majority class 89.4% of 189 answers. The marginal is `true_label`, not the menu.

#### train actionability

| label | count | share |
| --- | --- | --- |
| informational | 169 | 89.42% |
| advisory | 11 | 5.82% |
| requires action | 9 | 4.76% |

**train, audience.** Majority class 77.9% of 249 answers. The marginal is `true_label`, not the menu.

#### train audience

| label | count | share |
| --- | --- | --- |
| general public | 194 | 77.91% |
| expert | 25 | 10.04% |
| regulatory | 23 | 9.24% |
| internal | 7 | 2.81% |

**train, biotech_event.** Majority class 26.1% of 138 answers. The marginal is `true_label`, not the menu.

#### train biotech_event

| label | count | share |
| --- | --- | --- |
| executive statement | 36 | 26.09% |
| company description | 28 | 20.29% |
| product launching & presentation | 11 | 7.97% |
| alliance & partnership | 10 | 7.25% |
| other | 8 | 5.80% |
| m&a | 7 | 5.07% |
| new initiatives or programs | 7 | 5.07% |
| support & philanthropy | 6 | 4.35% |
| expanding geography | 5 | 3.62% |
| event organization | 4 | 2.90% |
| executive appointment | 4 | 2.90% |
| article publication | 3 | 2.17% |
| participation in an event | 2 | 1.45% |
| hiring | 2 | 1.45% |
| new initiatives & programs | 1 | 0.72% |
| foundation | 1 | 0.72% |
| expanding industry | 1 | 0.72% |
| investment in public company | 1 | 0.72% |
| regulatory approval | 1 | 0.72% |

**train, certainty.** Majority class 95.9% of 219 answers. The marginal is `true_label`, not the menu.

#### train certainty

| label | count | share |
| --- | --- | --- |
| asserted | 210 | 95.89% |
| hedged | 9 | 4.11% |

**train, chfinann_event.** Majority class 46.7% of 854 answers. The marginal is `true_label`, not the menu.

#### train chfinann_event

| label | count | share |
| --- | --- | --- |
| EquityPledge | 399 | 46.72% |
| EquityUnderweight | 189 | 22.13% |
| EquityOverweight | 169 | 19.79% |
| EquityRepurchase | 69 | 8.08% |
| EquityFreeze | 28 | 3.28% |

**train, docee_event.** Majority class 5.9% of 2,416 answers. The marginal is `true_label`, not the menu.

#### train docee_event

| label | count | share |
| --- | --- | --- |
| Sports Competition | 143 | 5.92% |
| none | 138 | 5.71% |
| Fire | 121 | 5.01% |
| CommitCrime - Investigate | 109 | 4.51% |
| Organization Fine | 106 | 4.39% |
| CommitCrime - Arrest | 103 | 4.26% |
| Government Policy Changes | 102 | 4.22% |
| Armed Conflict | 79 | 3.27% |
| Protest | 69 | 2.86% |
| Air Crash | 67 | 2.77% |
| Famous Person - Death | 64 | 2.65% |
| Resignation_Dismissal | 64 | 2.65% |
| Election | 59 | 2.44% |
| Diplomatic Talks | 56 | 2.32% |
| Appoint_Inauguration | 56 | 2.32% |
| New Achievements in Aerospace | 53 | 2.19% |
| Road Crash | 51 | 2.11% |
| Riot | 48 | 1.99% |
| Earthquakes | 48 | 1.99% |
| Break Historical Records | 48 | 1.99% |
| Famous Person - Give a Speech | 46 | 1.90% |
| Disease Outbreaks | 45 | 1.86% |
| CommitCrime - Accuse | 44 | 1.82% |
| CommitCrime - Sentence | 43 | 1.78% |
| Sign Agreement | 40 | 1.66% |
| Storm | 38 | 1.57% |
| Diplomatic Visit | 37 | 1.53% |
| Floods | 37 | 1.53% |
| Mass Poisoning | 32 | 1.32% |
| Awards Ceremony | 31 | 1.28% |
| Military Excercise | 29 | 1.20% |
| Strike | 26 | 1.08% |
| Financial Crisis | 25 | 1.03% |
| New Wonders in Nature | 24 | 0.99% |
| Environment Pollution | 20 | 0.83% |
| Shipwreck | 20 | 0.83% |
| Gas Explosion | 20 | 0.83% |
| Organization Established | 19 | 0.79% |
| CommitCrime - Release | 19 | 0.79% |
| New Archeological Discoveries | 19 | 0.79% |
| Withdraw from an Orgnization | 18 | 0.75% |
| Train Collisions | 17 | 0.70% |
| Mine Collapses | 16 | 0.66% |
| Droughts | 15 | 0.62% |
| Organization Closed | 14 | 0.58% |
| Mudslides | 13 | 0.54% |
| Bank Robbery | 13 | 0.54% |
| Join in an Orgnization | 13 | 0.54% |
| Famous Person - Sick | 13 | 0.54% |
| Financial Aid | 12 | 0.50% |
| Regime Change | 11 | 0.46% |
| Famous Person - Recovered | 10 | 0.41% |
| Organization Merge | 9 | 0.37% |
| Volcano Eruption | 9 | 0.37% |
| Famine | 8 | 0.33% |
| Insect Disaster | 7 | 0.29% |
| Famous Person - Divorce | 6 | 0.25% |
| Famous Person - Marriage | 6 | 0.25% |
| Tsunamis | 5 | 0.21% |
| Tear Up Agreement | 3 | 0.12% |

**train, docfee_event.** Majority class 35.6% of 556 answers. The marginal is `true_label`, not the menu.

#### train docfee_event

| label | count | share |
| --- | --- | --- |
| EquityUnderweight | 198 | 35.61% |
| BankruptcyLiquidation | 89 | 16.01% |
| EquityPledge | 86 | 15.47% |
| MajorAssetLoss | 85 | 15.29% |
| EquityFreeze | 37 | 6.65% |
| EquityOverweight | 27 | 4.86% |
| MajorSafetyAccident | 13 | 2.34% |
| ExecutiveDeath | 11 | 1.98% |
| MajorExternalPayout | 10 | 1.80% |

**train, document_genre.** Majority class 74.0% of 204 answers. The marginal is `true_label`, not the menu.

#### train document_genre

| label | count | share |
| --- | --- | --- |
| news report | 151 | 74.02% |
| announcement | 22 | 10.78% |
| opinion | 11 | 5.39% |
| review | 6 | 2.94% |
| correspondence | 6 | 2.94% |
| advertisement | 4 | 1.96% |
| legal filing | 2 | 0.98% |
| technical documentation | 2 | 0.98% |

**train, formality.** Majority class 81.6% of 234 answers. The marginal is `true_label`, not the menu.

#### train formality

| label | count | share |
| --- | --- | --- |
| formal | 191 | 81.62% |
| informal | 43 | 18.38% |

**train, language_register.** Majority class 83.0% of 206 answers. The marginal is `true_label`, not the menu.

#### train language_register

| label | count | share |
| --- | --- | --- |
| plain | 171 | 83.01% |
| technical | 26 | 12.62% |
| colloquial | 8 | 3.88% |
| legal | 1 | 0.49% |

**train, risk_level.** Majority class 49.7% of 173 answers. The marginal is `true_label`, not the menu.

#### train risk_level

| label | count | share |
| --- | --- | --- |
| none | 86 | 49.71% |
| low | 34 | 19.65% |
| moderate | 27 | 15.61% |
| high | 26 | 15.03% |

**train, sentiment.** Majority class 39.9% of 258 answers. The marginal is `true_label`, not the menu.

#### train sentiment

| label | count | share |
| --- | --- | --- |
| positive | 103 | 39.92% |
| neutral | 97 | 37.60% |
| negative | 58 | 22.48% |

**train, subjectivity.** Majority class 83.6% of 189 answers. The marginal is `true_label`, not the menu.

#### train subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 158 | 83.60% |
| subjective | 31 | 16.40% |

**train, temporal_orientation.** Majority class 67.0% of 185 answers. The marginal is `true_label`, not the menu.

#### train temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 124 | 67.03% |
| present | 42 | 22.70% |
| future | 15 | 8.11% |
| timeless | 4 | 2.16% |

**train, topic.** Majority class 17.5% of 417 answers. The marginal is `true_label`, not the menu.

#### train topic

| label | count | share |
| --- | --- | --- |
| business | 73 | 17.51% |
| politics | 60 | 14.39% |
| world | 46 | 11.03% |
| legal | 41 | 9.83% |
| sports | 40 | 9.59% |
| finance | 36 | 8.63% |
| entertainment | 27 | 6.47% |
| technology | 26 | 6.24% |
| education | 14 | 3.36% |
| health | 13 | 3.12% |
| environment | 13 | 3.12% |
| science | 9 | 2.16% |
| defense | 8 | 1.92% |
| travel | 7 | 1.68% |
| energy | 2 | 0.48% |
| agriculture | 2 | 0.48% |

**train, urgency.** Majority class 77.5% of 178 answers. The marginal is `true_label`, not the menu.

#### train urgency

| label | count | share |
| --- | --- | --- |
| routine | 138 | 77.53% |
| elevated | 31 | 17.42% |
| critical | 7 | 3.93% |
| urgent | 2 | 1.12% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | condition | 1 | 100.00% |
| clinical_finding | patient_group | 1 | 100.00% |
| clinical_finding | treatment | 1 | 100.00% |
| clinical_finding | outcome | 1 | 100.00% |
| clinical_finding | record_metadata | natural/condition ×1 |  |
| company_profile | name | 13 | 100.00% |
| company_profile | size | 12 | 92.31% |
| company_profile | headquarters | 11 | 84.62% |
| company_profile | industry | 8 | 61.54% |
| company_profile | founded | 5 | 38.46% |
| company_profile | record_metadata | natural/name ×13 |  |
| event_listing | name | 6 | 85.71% |
| event_listing | venue | 6 | 85.71% |
| event_listing | date | 6 | 85.71% |
| event_listing | price | 4 | 57.14% |
| event_listing | organizer | 2 | 28.57% |
| event_listing | record_metadata | natural/name ×6, natural/venue ×1 |  |
| financial_report | organization | 3 | 100.00% |
| financial_report | period | 3 | 100.00% |
| financial_report | revenue | 2 | 66.67% |
| financial_report | net_income | 1 | 33.33% |
| financial_report | record_metadata | natural/organization ×3 |  |
| incident_report | incident_type | 9 | 90.00% |
| incident_report | date | 9 | 90.00% |
| incident_report | severity | 9 | 90.00% |
| incident_report | location | 8 | 80.00% |
| incident_report | casualties | 4 | 40.00% |
| incident_report | record_metadata | natural/location ×6, natural/incident_type ×4 |  |
| job_posting | title | 3 | 100.00% |
| job_posting | company | 3 | 100.00% |
| job_posting | location | 3 | 100.00% |
| job_posting | employment_type | 3 | 100.00% |
| job_posting | record_metadata | natural/location ×1 |  |
| legal_case | defendant | 6 | 100.00% |
| legal_case | status | 6 | 100.00% |
| legal_case | court | 5 | 83.33% |
| legal_case | plaintiff | 4 | 66.67% |
| legal_case | case_name | 3 | 50.00% |
| legal_case | record_metadata | natural/court ×3, natural/case_name ×3 |  |
| person_profile | name | 77 | 100.00% |
| person_profile | role | 72 | 93.51% |
| person_profile | employer | 69 | 89.61% |
| person_profile | location | 66 | 85.71% |
| person_profile | record_metadata | natural/location ×38, natural/name ×17 |  |
| product | name | 30 | 100.00% |
| product | condition | 29 | 96.67% |
| product | price | 25 | 83.33% |
| product | brand | 23 | 76.67% |
| product | record_metadata | natural/name ×25 |  |
| record | status | 31 | 12.11% |
| record | name | 21 | 8.20% |
| record | location | 15 | 5.86% |
| record | title | 13 | 5.08% |
| record | description | 11 | 4.30% |
| record | price | 10 | 3.91% |
| record | address | 10 | 3.91% |
| record | city | 6 | 2.34% |
| record | transaction_id | 5 | 1.95% |
| record | review_id | 5 | 1.95% |
| record | rating | 5 | 1.95% |
| record | date | 5 | 1.95% |
| record | patient_id | 5 | 1.95% |
| record | sku | 5 | 1.95% |
| record | patient_name | 4 | 1.56% |
| record | normal_range | 4 | 1.56% |
| record | time | 4 | 1.56% |
| record | state | 4 | 1.56% |
| record | amount | 4 | 1.56% |
| record | product_id | 4 | 1.56% |
| record | appointment_id | 4 | 1.56% |
| record | department | 4 | 1.56% |
| record | acquirer | 4 | 1.56% |
| record | brand | 4 | 1.56% |
| record | event_id | 3 | 1.17% |
| record | id | 3 | 1.17% |
| record | employee_id | 3 | 1.17% |
| record | product_name | 3 | 1.17% |
| record | status: Current state of the transaction | 3 | 1.17% |
| record | region | 3 | 1.17% |
| record | temperature | 3 | 1.17% |
| record | venue | 3 | 1.17% |
| record | findings | 3 | 1.17% |
| record | name: Full name of the patient | 3 | 1.17% |
| record | organizations | 3 | 1.17% |
| record | color | 3 | 1.17% |
| record | category | 3 | 1.17% |
| record | score | 3 | 1.17% |
| record | asteroid.name | 3 | 1.17% |
| record | patient_id: Unique identifier for the patient | 2 | 0.78% |
| record | temperature: Body temperature in degrees Fahrenheit | 2 | 0.78% |
| record | tags | 2 | 0.78% |
| record | album | 2 | 0.78% |
| record | test_id | 2 | 0.78% |
| record | value | 2 | 0.78% |
| record | timestamp: Date and time of transaction occurrence | 2 | 0.78% |
| record | exchange_rate: Rate used to convert base to target currency | 2 | 0.78% |
| record | transaction_amount: Amount being converted | 2 | 0.78% |
| record | source_provider: Provider handling the exchange rate calculation | 2 | 0.78% |
| record | user_id: Unique identifier for the user initiating transaction | 2 | 0.78% |
| record | transaction_id: Unique identifier for the transaction | 2 | 0.78% |
| record | notes: Additional context or explanation for the transaction | 2 | 0.78% |
| record | features | 2 | 0.78% |
| record | author | 2 | 0.78% |
| record | tv_program.previous_program | 2 | 0.78% |
| record | tv_program.production.production_area | 2 | 0.78% |
| record | tv_program.setting.year | 2 | 0.78% |
| record | tv_program.production.location.name | 2 | 0.78% |
| record | source | 2 | 0.78% |
| record | position | 2 | 0.78% |
| record | purpose | 2 | 0.78% |
| record | source_agency | 2 | 0.78% |
| record | comparative_authors | 2 | 0.78% |
| record | main_competitor | 2 | 0.78% |
| record | format | 2 | 0.78% |
| record | head | 2 | 0.78% |
| record | tail | 2 | 0.78% |
| record | topic | 2 | 0.78% |
| record | respiratory_rate | 2 | 0.78% |
| record | notes | 2 | 0.78% |
| record | refund_amount | 2 | 0.78% |
| record | authors | 2 | 0.78% |
| record | year | 2 | 0.78% |
| record | company | 2 | 0.78% |
| record | home_team | 2 | 0.78% |
| record | team | 2 | 0.78% |
| record | medication | 2 | 0.78% |
| record | scholarship_id | 2 | 0.78% |
| record | disbursement_plan | 2 | 0.78% |
| record | zip | 2 | 0.78% |
| record | imaging_date | 2 | 0.78% |
| record | locations | 2 | 0.78% |
| record | reviewer | 2 | 0.78% |
| record | reporter | 2 | 0.78% |
| record | organization.name | 2 | 0.78% |
| record | organization.location_founded.country | 2 | 0.78% |
| record | away_team | 2 | 0.78% |
| record | policyholder_name | 2 | 0.78% |
| record | policy_id | 2 | 0.78% |
| record | policytype | 2 | 0.78% |
| record | dimensions | 2 | 0.78% |
| record | origin | 2 | 0.78% |
| record | target_audience | 2 | 0.78% |
| record | age | 2 | 0.78% |
| record | merchant | 2 | 0.78% |
| record | heart_rate | 2 | 0.78% |
| record | device_id | 2 | 0.78% |
| record | status: Current status of the transaction | 2 | 0.78% |
| record | channel_name | 2 | 0.78% |
| record | channel_url | 2 | 0.78% |
| record | customer_address | 2 | 0.78% |
| record | physical_characteristics.observations.0.year | 2 | 0.78% |
| record | physical_characteristics.observations.0.telescope | 2 | 0.78% |
| record | orbital_characteristics.perihelion.unit | 2 | 0.78% |
| record | app_version | 2 | 0.78% |
| record | user_id | 2 | 0.78% |
| record | person | 2 | 0.78% |
| record | recipient | 2 | 0.78% |
| record | expiry_date | 2 | 0.78% |
| record | study.title | 2 | 0.78% |
| record | study.key_findings | 2 | 0.78% |
| record | study.period.end_year | 2 | 0.78% |
| record | study.author.affiliation | 2 | 0.78% |
| record | code | 2 | 0.78% |
| record | status: Current state of the ticket | 2 | 0.78% |
| record | on_site_registration.documents_required | 1 | 0.39% |
| record | on_site_registration.location.address | 1 | 0.39% |
| record | priority_registration.instructions | 1 | 0.39% |
| record | on_site_registration.location.name | 1 | 0.39% |
| record | on_site_registration.requirements | 1 | 0.39% |
| record | on_site_registration.description | 1 | 0.39% |
| record | on_site_registration.note | 1 | 0.39% |
| record | online_priority_registration.deadline | 1 | 0.39% |
| record | spo2: Oxygen saturation percentage | 1 | 0.39% |
| record | blood_pressure: Blood pressure in format systolic/diastolic | 1 | 0.39% |
| record | heart_rate: Number of heartbeats per minute | 1 | 0.39% |
| record | address: Street address including city and zip code | 1 | 0.39% |
| record | test_type | 1 | 0.39% |
| record | author_handle | 1 | 0.39% |
| record | tech_product | 1 | 0.39% |
| record | replies | 1 | 0.39% |
| record | tweet_id | 1 | 0.39% |
| record | rsvp_count | 1 | 0.39% |
| record | cost | 1 | 0.39% |
| record | expert_insights | 1 | 0.39% |
| record | status: Current status of the lease agreement | 1 | 0.39% |
| record | address: Full physical address of the leased property | 1 | 0.39% |
| record | tenant: Name of the company leasing the space | 1 | 0.39% |
| record | categories | 1 | 0.39% |
| record | manifest_id | 1 | 0.39% |
| record | receiver | 1 | 0.39% |
| record | ebit_margin | 1 | 0.39% |
| record | revenue | 1 | 0.39% |
| record | revenuebyregion | 1 | 0.39% |
| record | merchantcategorycode | 1 | 0.39% |
| record | title: Topic or title of the submitted work | 1 | 0.39% |
| record | decision_status: Final outcome of the submission | 1 | 0.39% |
| record | reference_number: Unique identifier for the invoice document | 1 | 0.39% |
| record | recipient_name: Name of the recipient organization | 1 | 0.39% |
| record | sender_name: Name of the sender in the correspondence | 1 | 0.39% |
| record | claim_amount | 1 | 0.39% |
| record | procedure | 1 | 0.39% |
| record | claim_id | 1 | 0.39% |
| record | details.calorie_burn.kettlebell_calories | 1 | 0.39% |
| record | details.calorie_burn.kettlebell_time | 1 | 0.39% |
| record | details.calorie_burn.running_calories | 1 | 0.39% |
| record | details.calorie_burn.running_time | 1 | 0.39% |
| record | details.calorie_burn.running_speed | 1 | 0.39% |
| record | training_modules | 1 | 0.39% |
| record | user | 1 | 0.39% |
| record | app_name | 1 | 0.39% |
| record | year_level | 1 | 0.39% |
| record | scientificName | 1 | 0.39% |
| record | publication.author.affiliation.country | 1 | 0.39% |
| record | order | 1 | 0.39% |
| record | discovery.location | 1 | 0.39% |
| record | publication.author.deathYear | 1 | 0.39% |
| record | publication.author.affiliation.institution | 1 | 0.39% |
| record | publication.author.name | 1 | 0.39% |
| record | publication.author.birthYear | 1 | 0.39% |
| record | video_id | 1 | 0.39% |
| record | shares | 1 | 0.39% |
| record | launched_by | 1 | 0.39% |
| record | historical_context.my_folks_series.first_book_year | 1 | 0.39% |
| record | product.origin.method_origin | 1 | 0.39% |
| record | product.storage.wrapping | 1 | 0.39% |
| record | product.preparation.steps | 1 | 0.39% |
| record | product.name | 1 | 0.39% |
| record | product.storage.containers | 1 | 0.39% |
| record | product.preparation.drying_time | 1 | 0.39% |
| record | historical_context.cappers_weekly.call_for_articles_year | 1 | 0.39% |
| record | historical_context.my_folks_series.number_of_books | 1 | 0.39% |
| record | product.origin.spread_location | 1 | 0.39% |
| record | historical_context.cappers_weekly.topic | 1 | 0.39% |
| record | investor_count: Number of investors involved in the round | 1 | 0.39% |
| record | location: City and country where the startup is headquartered | 1 | 0.39% |
| record | lead_investor: Primary investor leading the funding round | 1 | 0.39% |
| record | focus_area: Primary technological or business focus of the startup | 1 | 0.39% |
| record | line1 | 1 | 0.39% |
| record | base_currency: Original currency used in transaction | 1 | 0.39% |
| record | target_currency: Currency to which amount is converted | 1 | 0.39% |
| record | location: Geographic location of transaction | 1 | 0.39% |
| record | favorite_players | 1 | 0.39% |
| record | edit_information.timestamp | 1 | 0.39% |
| record | sentiment.opinion | 1 | 0.39% |
| record | quote.text | 1 | 0.39% |
| record | sentiment.preference | 1 | 0.39% |
| record | visit_count | 1 | 0.39% |
| record | usage_frequency | 1 | 0.39% |
| record | comments | 1 | 0.39% |
| record | payment | 1 | 0.39% |
| record | tv_program.production.location.type | 1 | 0.39% |
| record | other_programs | 1 | 0.39% |
| record | tv_program.setting.time_period | 1 | 0.39% |
| record | outbreak_event.investigation.specimen_source | 1 | 0.39% |
| record | outbreak_event.illness.pathogen.full_name | 1 | 0.39% |
| record | outbreak_event.location.country | 1 | 0.39% |
| record | outbreak_event.public_health_recommendations.target_populations | 1 | 0.39% |
| record | outbreak_event.investigation.findings.agent_details.strain_type | 1 | 0.39% |
| record | outbreak_event.illness.pathogen.abbreviation | 1 | 0.39% |
| record | outbreak_event.illness.description | 1 | 0.39% |
| record | outbreak_event.investigation.findings.agent_details.full_name | 1 | 0.39% |
| record | outbreak_event.illness.pathogen.classification | 1 | 0.39% |
| record | boulder.start.foot_holds.0.foot | 1 | 0.39% |
| record | boulder.location.direction | 1 | 0.39% |
| record | boulder.start.foot_holds.1.foot | 1 | 0.39% |
| record | boulder.location.distance | 1 | 0.39% |
| record | boulder.start.hand_holds.1.description | 1 | 0.39% |
| record | boulder.start.foot_holds.1.description | 1 | 0.39% |
| record | boulder.start.hand_holds.1.hand | 1 | 0.39% |
| record | submission.submitted_by | 1 | 0.39% |
| record | boulder.start.foot_holds.0.description | 1 | 0.39% |
| record | caveat.example.components.0.related_characters.1.meaning | 1 | 0.39% |
| record | caveat.example.components.0.related_characters.1.pronunciation | 1 | 0.39% |
| record | caveat.example.components.0.related_characters.2.meaning | 1 | 0.39% |
| record | caveat.example.components.0.meaning | 1 | 0.39% |
| record | caveat.example.components.0.position | 1 | 0.39% |
| record | caveat.example.components.1.position | 1 | 0.39% |
| record | comparison.alphabetic.learning_curve | 1 | 0.39% |
| record | caveat.example.components.0.related_characters.2.pronunciation | 1 | 0.39% |
| record | caveat.example.components.1.meaning | 1 | 0.39% |
| record | caveat.example.meaning | 1 | 0.39% |
| record | terminology.ideogram.english | 1 | 0.39% |
| record | comparison.alphabetic.pronunciation | 1 | 0.39% |
| record | caveat.example.components.0.related_characters.0.pronunciation | 1 | 0.39% |
| record | terminology.ideographic_script.english | 1 | 0.39% |
| record | comparison.ideographic.example.meaning | 1 | 0.39% |
| record | caveat.example.components.0.function | 1 | 0.39% |
| record | caveat.example.components.0.related_characters.1.character | 1 | 0.39% |
| record | caveat.example.components.0.component | 1 | 0.39% |
| record | comparison.ideographic.learning_curve | 1 | 0.39% |
| record | definition | 1 | 0.39% |
| record | review_date | 1 | 0.39% |
| record | performance_rating | 1 | 0.39% |
| record | salary_adjustment | 1 | 0.39% |
| record | action_items | 1 | 0.39% |
| record | supervisor | 1 | 0.39% |
| record | resource_name | 1 | 0.39% |
| record | regulatory_status | 1 | 0.39% |
| record | platform_name | 1 | 0.39% |
| record | earnout | 1 | 0.39% |
| record | integration_plan | 1 | 0.39% |
| record | submission_id: Unique identifier for the paper submission | 1 | 0.39% |
| record | author: Name of the submitting researcher or academic | 1 | 0.39% |
| record | institution: University or institution where author is affiliated | 1 | 0.39% |
| record | department: Academic department of the author | 1 | 0.39% |
| record | topic: Subject area or focus of the submitted paper | 1 | 0.39% |
| record | status: Current processing or acceptance status | 1 | 0.39% |
| record | event_date: Date when the event will take place | 1 | 0.39% |
| record | organizer_name: Name of the person or group organizing the event | 1 | 0.39% |
| record | city: City where the event is held | 1 | 0.39% |
| record | location: Full address including street, city, and state | 1 | 0.39% |
| record | event_code: Unique code identifying the event | 1 | 0.39% |
| record | jurisdiction | 1 | 0.39% |
| record | entity_name | 1 | 0.39% |
| record | file_id | 1 | 0.39% |
| record | author_information.education.0.institution | 1 | 0.39% |
| record | author_information.military_service.branch | 1 | 0.39% |
| record | author_information.military_service.status | 1 | 0.39% |
| record | author_information.occupation | 1 | 0.39% |
| record | author_information.location.city | 1 | 0.39% |
| record | author_information.location.state | 1 | 0.39% |
| record | author_information.family.spouse | 1 | 0.39% |
| record | author_information.family.pets.0.name | 1 | 0.39% |
| record | author_information.family.pets.0.type | 1 | 0.39% |
| record | author_information.interests | 1 | 0.39% |
| record | keywords | 1 | 0.39% |
| record | recipient_name | 1 | 0.39% |
| record | po_reference | 1 | 0.39% |
| record | paymentduedays: Number of days until payment is due after notice | 1 | 0.39% |
| record | renewal_amount: Annual premium amount for the renewal | 1 | 0.39% |
| record | policy_id: Unique identifier for the insurance policy | 1 | 0.39% |
| record | patient_address: Full address of the patient | 1 | 0.39% |
| record | patient_name: Name of the patient filing the claim | 1 | 0.39% |
| record | diagnosis_code: Medical code assigned to the patient's condition | 1 | 0.39% |
| record | claim_amount: Total cost of the medical service billed | 1 | 0.39% |
| record | discount.percentage | 1 | 0.39% |
| record | discount.description | 1 | 0.39% |
| record | location.region | 1 | 0.39% |
| record | location.hotel_count | 1 | 0.39% |
| record | booking.promotional_code | 1 | 0.39% |
| record | booking.validity | 1 | 0.39% |
| record | booking.cancellation_policy | 1 | 0.39% |
| record | terms_and_conditions.availability | 1 | 0.39% |
| record | terms_and_conditions.group_booking_restriction | 1 | 0.39% |
| record | terms_and_conditions.room_upgrades | 1 | 0.39% |
| record | terms_and_conditions.combination_restriction | 1 | 0.39% |
| record | terms_and_conditions.disclaimer | 1 | 0.39% |
| record | hotel_chain | 1 | 0.39% |
| record | rates.rate_type | 1 | 0.39% |
| record | loyalty_program.name | 1 | 0.39% |
| record | loyalty_program.membership_url | 1 | 0.39% |
| record | booking_channels | 1 | 0.39% |
| record | bracket_size | 1 | 0.39% |
| record | headlines | 1 | 0.39% |
| record | line2 | 1 | 0.39% |
| record | conference_date | 1 | 0.39% |
| record | session_id | 1 | 0.39% |
| record | revenue: Total revenue in billions as reported | 1 | 0.39% |
| record | eps: Earnings per share in dollars | 1 | 0.39% |
| record | growth_rate: Year-over-year growth percentage | 1 | 0.39% |
| record | market_cap: Market capitalization in billions | 1 | 0.39% |
| record | sunless_tanning.overview | 1 | 0.39% |
| record | sunless_tanning.concerns | 1 | 0.39% |
| record | sunless_tanning.product_collection.brand | 1 | 0.39% |
| record | sunless_tanning.product_collection.collection_name | 1 | 0.39% |
| record | sunless_tanning.product_collection.benefits | 1 | 0.39% |
| record | sunless_tanning.product_collection.products.0.name | 1 | 0.39% |
| record | sunless_tanning.product_collection.products.0.type | 1 | 0.39% |
| record | sunless_tanning.product_collection.products.0.sizes | 1 | 0.39% |
| record | sunless_tanning.product_collection.products.0.description | 1 | 0.39% |
| record | sunless_tanning.product_collection.products.0.ingredients | 1 | 0.39% |
| record | sunless_tanning.product_collection.products.0.benefits | 1 | 0.39% |
| record | sunless_tanning.product_collection.products.1.name | 1 | 0.39% |
| record | sunless_tanning.product_collection.products.1.type | 1 | 0.39% |
| record | sunless_tanning.product_collection.products.1.description | 1 | 0.39% |
| record | sunless_tanning.product_collection.products.1.ingredients | 1 | 0.39% |
| record | sunless_tanning.product_collection.products.1.benefits | 1 | 0.39% |
| record | risks_of_sun_exposure | 1 | 0.39% |
| record | series_record | 1 | 0.39% |
| record | provider | 1 | 0.39% |
| record | access_type | 1 | 0.39% |
| record | parameters | 1 | 0.39% |
| record | match_id: Unique identifier for the match as provided | 1 | 0.39% |
| record | score: Final score in format home-away | 1 | 0.39% |
| record | competition: League or cup type such as Premier League or friendly | 1 | 0.39% |
| record | case_id | 1 | 0.39% |
| record | lab_id | 1 | 0.39% |
| record | test_date | 1 | 0.39% |
| record | test_name | 1 | 0.39% |
| record | species.scientific_name | 1 | 0.39% |
| record | species.genus | 1 | 0.39% |
| record | species.description | 1 | 0.39% |
| record | campaign_id | 1 | 0.39% |
| record | post_status | 1 | 0.39% |
| record | ticket_number: Unique identifier for the RMA request | 1 | 0.39% |
| record | device_model: Model name of the returned electronic device | 1 | 0.39% |
| record | serial_number: Unique hardware identifier for the device | 1 | 0.39% |
| record | submitter: Name of the customer who submitted the ticket | 1 | 0.39% |
| record | purchase_order: Order number from original purchase | 1 | 0.39% |
| record | purchase_location: Store where device was originally bought | 1 | 0.39% |
| record | return_date: Date device was returned to repair center | 1 | 0.39% |
| record | research_context.problem_statement.incompatible_substrates | 1 | 0.39% |
| record | research_context.primary_technique | 1 | 0.39% |
| record | outcomes.impact | 1 | 0.39% |
| record | outcomes.comparative_advantage.benefit_type | 1 | 0.39% |
| record | research_context.problem_statement.limitation | 1 | 0.39% |
| record | methodology.experimental_setup.reagents.0.type | 1 | 0.39% |
| record | research_context.application_goal | 1 | 0.39% |
| record | methodology.experimental_setup.biological_model.culture_environment.material_properties | 1 | 0.39% |
| record | outcomes.comparative_advantage.alternative_to | 1 | 0.39% |
| record | methodology.experimental_setup.instrumentation | 1 | 0.39% |
| record | call_id | 1 | 0.39% |
| record | asteroid.discovery.discoverer.name | 1 | 0.39% |
| record | asteroid.designation | 1 | 0.39% |
| record | asteroid.named_after.description | 1 | 0.39% |
| record | asteroid.named_after.birth_location.country | 1 | 0.39% |
| record | asteroid.alternate_name | 1 | 0.39% |
| record | asteroid.discovery.discoverer.affiliation.observatory | 1 | 0.39% |
| record | policy_type | 1 | 0.39% |
| record | policy_renewal_notice | 1 | 0.39% |
| record | policy_holder | 1 | 0.39% |
| record | limitations_of_midwives | 1 | 0.39% |
| record | resources | 1 | 0.39% |
| record | investor: Name of the investment firm conducting the meeting | 1 | 0.39% |
| record | startup: Name of the startup receiving investment | 1 | 0.39% |
| record | amount: Investment amount in millions of dollars | 1 | 0.39% |
| record | meeting_code: Unique code assigned to the briefing session | 1 | 0.39% |
| record | visitor_team | 1 | 0.39% |
| record | points | 1 | 0.39% |
| record | match | 1 | 0.39% |
| record | report_id | 1 | 0.39% |
| record | onset_timing | 1 | 0.39% |
| record | symptom_description | 1 | 0.39% |
| record | acquired | 1 | 0.39% |
| record | valuation_multiple | 1 | 0.39% |
| record | acquisition_amount | 1 | 0.39% |
| record | person.phone_number | 1 | 0.39% |
| record | scholarship_name | 1 | 0.39% |
| record | holiday_look.collar.brand | 1 | 0.39% |
| record | holiday_look.heels.brand | 1 | 0.39% |
| record | comparison.similar_look | 1 | 0.39% |
| record | comparison.brand_with_similar_look | 1 | 0.39% |
| record | holiday_look.dress.designer | 1 | 0.39% |
| record | role | 1 | 0.39% |
| record | event | 1 | 0.39% |
| record | competition_year | 1 | 0.39% |
| record | medal | 1 | 0.39% |
| record | issue | 1 | 0.39% |
| record | audit_id | 1 | 0.39% |
| record | size | 1 | 0.39% |
| record | type | 1 | 0.39% |
| record | industry | 1 | 0.39% |
| record | country | 1 | 0.39% |
| record | staff | 1 | 0.39% |
| record | full_name: Name of the employee as recorded in the review | 1 | 0.39% |
| record | location: Physical address of the employee's workplace | 1 | 0.39% |
| record | kpideliveryaccuracy: Percentage of on-time deliveries achieved | 1 | 0.39% |
| record | department: Department where the employee is based | 1 | 0.39% |
| record | customer_satisfaction: Score from customer feedback surveys | 1 | 0.39% |
| record | employee_id: Unique identifier assigned to the employee | 1 | 0.39% |
| record | credits.contributors.0.role | 1 | 0.39% |
| record | credits.contributors.0.person | 1 | 0.39% |
| record | credits.content_implementers | 1 | 0.39% |
| record | credits.graphic_design.0.designer | 1 | 0.39% |
| record | credits.organization | 1 | 0.39% |
| record | credits.funding | 1 | 0.39% |
| record | credits.graphic_design.1.designer | 1 | 0.39% |
| record | credits.version | 1 | 0.39% |
| record | credits.graphic_design.1.copyright_year | 1 | 0.39% |
| record | credits.contributors.1.role | 1 | 0.39% |
| record | examination_date | 1 | 0.39% |
| record | finding_summary | 1 | 0.39% |
| record | modality | 1 | 0.39% |
| record | radiology_code | 1 | 0.39% |
| record | prescriber | 1 | 0.39% |
| record | prescription_date | 1 | 0.39% |
| record | fda_code | 1 | 0.39% |
| record | dosage | 1 | 0.39% |
| record | years_mentioned | 1 | 0.39% |
| record | mrn | 1 | 0.39% |
| record | consultation.practice_areas | 1 | 0.39% |
| record | practice_areas | 1 | 0.39% |
| record | states_served | 1 | 0.39% |
| record | services | 1 | 0.39% |
| record | firm_name | 1 | 0.39% |
| record | client_focus | 1 | 0.39% |
| record | additional_services | 1 | 0.39% |
| record | experience.years | 1 | 0.39% |
| record | experience.description | 1 | 0.39% |
| record | sales_volume | 1 | 0.39% |
| record | date_sold | 1 | 0.39% |
| record | century | 1 | 0.39% |
| record | images | 1 | 0.39% |
| record | reviewer: Name of the critic or editorial team | 1 | 0.39% |
| record | publication: Name of the media outlet publishing the review | 1 | 0.39% |
| record | film_title: Title of the film being reviewed | 1 | 0.39% |
| record | rating: Numerical score out of 5 given by the reviewer | 1 | 0.39% |
| record | problem_description | 1 | 0.39% |
| record | affected_systems | 1 | 0.39% |
| record | urls | 1 | 0.39% |
| record | solution | 1 | 0.39% |
| record | status: Current status of the encounter | 1 | 0.39% |
| record | visit_type: Type of medical visit conducted | 1 | 0.39% |
| record | encounter_date: Date of the patient visit | 1 | 0.39% |
| record | webmaster | 1 | 0.39% |
| record | last_modified | 1 | 0.39% |
| record | mls_number | 1 | 0.39% |
| record | agent | 1 | 0.39% |
| record | bathrooms | 1 | 0.39% |
| record | research_interests | 1 | 0.39% |
| record | author.role | 1 | 0.39% |
| record | version | 1 | 0.39% |
| record | author.name | 1 | 0.39% |
| record | ticket_id: Unique identifier for each support ticket | 1 | 0.39% |
| record | priority: Severity level of the issue | 1 | 0.39% |
| record | company: Company name of the ticket owner | 1 | 0.39% |
| record | impact_users: Number of users directly affected by the issue | 1 | 0.39% |
| record | place.location.region | 1 | 0.39% |
| record | place.location.department | 1 | 0.39% |
| record | place.location.country | 1 | 0.39% |
| record | place.location.province | 1 | 0.39% |
| record | place.location.district | 1 | 0.39% |
| record | place.type | 1 | 0.39% |
| record | id: Unique identifier for the legislative vote | 1 | 0.39% |
| record | date: Date when the vote occurred | 1 | 0.39% |
| record | chamber: Legislative chamber where the vote took place | 1 | 0.39% |
| record | bill: Bill number and type identifier | 1 | 0.39% |
| record | sponsor: Name of the bill's sponsor | 1 | 0.39% |
| record | outcome: Final result of the vote | 1 | 0.39% |
| record | amount: Financial amount associated with the bill | 1 | 0.39% |
| record | location: Geographic location of the bill's jurisdiction | 1 | 0.39% |
| record | status: Current status of the bill | 1 | 0.39% |
| record | description: Summary of the bill's purpose and scope | 1 | 0.39% |
| record | hotel_name | 1 | 0.39% |
| record | room_type | 1 | 0.39% |
| record | ticket_id | 1 | 0.39% |
| record | location: Venue city and country where the final occurred | 1 | 0.39% |
| record | tournament_name: Name of the tennis tournament such as Monte Carlo Masters | 1 | 0.39% |
| record | score: Match score in format 6-3, 4-6, 6-4 | 1 | 0.39% |
| record | revenue: Total gate revenue in millions | 1 | 0.39% |
| record | opponent: Name of the player who lost in the final | 1 | 0.39% |
| record | regionNameBerber | 1 | 0.39% |
| record | regionName | 1 | 0.39% |
| record | infrastructure.roads.0.connections | 1 | 0.39% |
| record | infrastructure.refineries.0.location | 1 | 0.39% |
| record | economy.gdpCurrency | 1 | 0.39% |
| record | infrastructure.refineries.0.status | 1 | 0.39% |
| record | geography.location | 1 | 0.39% |
| record | geography.borders.1.region | 1 | 0.39% |
| record | economy.gdpValue | 1 | 0.39% |
| record | geography.borders.2.direction | 1 | 0.39% |
| record | infrastructure.refineries.0.closureDate | 1 | 0.39% |
| record | populationYear | 1 | 0.39% |
| record | lowerLevelDivisions | 1 | 0.39% |
| record | geography.borders.1.direction | 1 | 0.39% |
| record | economy.economicBase | 1 | 0.39% |
| record | geography.borders.2.region | 1 | 0.39% |
| record | geography.borders.0.region | 1 | 0.39% |
| record | infrastructure.airports.0.trafficVolume | 1 | 0.39% |
| record | geography.borders.0.direction | 1 | 0.39% |
| record | infrastructure.roads.0.roadName | 1 | 0.39% |
| record | geography.rivers | 1 | 0.39% |
| record | resolution_team | 1 | 0.39% |
| record | priority | 1 | 0.39% |
| record | organization.year_founded | 1 | 0.39% |
| record | patient_id: Unique identifier for each patient | 1 | 0.39% |
| record | onset_date: Date when symptoms first began | 1 | 0.39% |
| record | address_street: Street address line of the patient | 1 | 0.39% |
| record | symptom_location: Body region where symptoms are felt | 1 | 0.39% |
| record | status: Current clinical status of the patient | 1 | 0.39% |
| record | topics | 1 | 0.39% |
| record | assists | 1 | 0.39% |
| record | key_player | 1 | 0.39% |
| record | sender_domain: Domain used in the fraudulent email | 1 | 0.39% |
| record | report_id: Unique identifier for the phishing incident | 1 | 0.39% |
| record | stockstatus | 1 | 0.39% |
| record | weight | 1 | 0.39% |
| record | available_units | 1 | 0.39% |
| record | organization.location_founded.city | 1 | 0.39% |
| record | physician.education.medical_school.0.institution | 1 | 0.39% |
| record | physician.education.residency.0.institution | 1 | 0.39% |
| record | physician.address.city | 1 | 0.39% |
| record | physician.address.zip | 1 | 0.39% |
| record | physician.name | 1 | 0.39% |
| record | physician.address.state | 1 | 0.39% |
| record | last_updated | 1 | 0.39% |
| record | available_quantity | 1 | 0.39% |
| record | status: Current state of the incident | 1 | 0.39% |
| record | customer_name: Name of the affected customer | 1 | 0.39% |
| record | ticket_id: Unique identifier for each incident in the system | 1 | 0.39% |
| record | root_cause: Technical reason for the service disruption | 1 | 0.39% |
| record | see_also | 1 | 0.39% |
| record | invoice_date | 1 | 0.39% |
| record | month | 1 | 0.39% |
| record | vendor | 1 | 0.39% |
| record | production_location | 1 | 0.39% |
| record | githubrepourl | 1 | 0.39% |
| record | linkedinposturl | 1 | 0.39% |
| record | powers_abilities | 1 | 0.39% |
| record | affiliation | 1 | 0.39% |
| record | transaction_type | 1 | 0.39% |
| record | dose | 1 | 0.39% |
| record | severity | 1 | 0.39% |
| record | outcome | 1 | 0.39% |
| record | target | 1 | 0.39% |
| record | revenue_uptick | 1 | 0.39% |
| record | cost_savings | 1 | 0.39% |
| record | bloodpressuresystolic | 1 | 0.39% |
| record | bloodpressurediastolic | 1 | 0.39% |
| record | monitoring_method | 1 | 0.39% |
| record | attending_physician | 1 | 0.39% |
| record | destination | 1 | 0.39% |
| record | shipment_id | 1 | 0.39% |
| record | dates | 1 | 0.39% |
| record | date: Date the exchange rate snapshot was taken | 1 | 0.39% |
| record | currency_pair: Pair of currencies involved in the exchange | 1 | 0.39% |
| record | rate: Exchange rate value from base to USD | 1 | 0.39% |
| record | sender: Name of the individual initiating the transfer | 1 | 0.39% |
| record | address: Physical address of the sender | 1 | 0.39% |
| record | encounter_id: Unique identifier for the visit | 1 | 0.39% |
| record | address: Patient's residential address | 1 | 0.39% |
| record | visit_date: Date of the medical encounter | 1 | 0.39% |
| record | primary_diagnosis: Main medical condition diagnosed | 1 | 0.39% |
| record | icd_code: ICD-10 code for the primary diagnosis | 1 | 0.39% |
| record | bp_systolic: Systolic blood pressure in mmHg | 1 | 0.39% |
| record | bp_diastolic: Diastolic blood pressure in mmHg | 1 | 0.39% |
| record | heart_rate: Heart rate in beats per minute | 1 | 0.39% |
| record | medications: List of prescribed medications with dosing | 1 | 0.39% |
| record | lab_results: List of lab test results | 1 | 0.39% |
| record | symptoms: Patient-reported symptoms or complaints | 1 | 0.39% |
| record | transfer_portal | 1 | 0.39% |
| record | from_club | 1 | 0.39% |
| record | to_club | 1 | 0.39% |
| record | transfer_fee | 1 | 0.39% |
| record | transfer_code | 1 | 0.39% |
| record | player_id | 1 | 0.39% |
| record | duration.max_years | 1 | 0.39% |
| record | project_goals | 1 | 0.39% |
| record | start_year | 1 | 0.39% |
| record | duration.min_years | 1 | 0.39% |
| record | beneficiary_focus | 1 | 0.39% |
| record | hotel_initiatives | 1 | 0.39% |
| record | title: Job title as listed in the requisition | 1 | 0.39% |
| record | requisition_id: Unique identifier for the job posting | 1 | 0.39% |
| record | office | 1 | 0.39% |
| record | wallet_address: Unique wallet address used for the transaction | 1 | 0.39% |
| record | amount: Quantity of crypto asset bought or sold | 1 | 0.39% |
| record | issue_date | 1 | 0.39% |
| record | term_length | 1 | 0.39% |
| record | monthly_premium | 1 | 0.39% |
| record | user: Name of the employee reporting the issue | 1 | 0.39% |
| record | ticket_id: Unique identifier for each ticket as per prefix format | 1 | 0.39% |
| record | affected_system: System or application where the issue occurs | 1 | 0.39% |
| record | reference: Internal reference code linked to the ticket | 1 | 0.39% |
| record | issue: Nature of the technical problem reported | 1 | 0.39% |
| record | submitted: Date when the ticket was submitted | 1 | 0.39% |
| record | genus.pollination.mechanism | 1 | 0.39% |
| record | genus.etymology.part2.name | 1 | 0.39% |
| record | genus.related_genera | 1 | 0.39% |
| record | genus.pollination.pollinators | 1 | 0.39% |
| record | genus.german_species | 1 | 0.39% |
| record | genus.etymology.part1.name | 1 | 0.39% |
| record | genus.etymology.part1.related_entity | 1 | 0.39% |
| record | genus.etymology.part1.origin | 1 | 0.39% |
| record | genus.distribution | 1 | 0.39% |
| record | genus.characteristics | 1 | 0.39% |
| record | genus.species_list | 1 | 0.39% |
| record | genus.name | 1 | 0.39% |
| record | asteroid.named_after.person.name | 1 | 0.39% |
| record | asteroid.named_after.person.birth_year | 1 | 0.39% |
| record | asteroid.name_latin | 1 | 0.39% |
| record | asteroid.asteroid_type | 1 | 0.39% |
| record | thumbnail_style | 1 | 0.39% |
| record | property_id: Unique identifier for the property | 1 | 0.39% |
| record | address: Full physical address of the property | 1 | 0.39% |
| record | description: Brief textual description of property features and appeal | 1 | 0.39% |
| record | ship_date | 1 | 0.39% |
| record | website | 1 | 0.39% |
| record | url | 1 | 0.39% |
| record | processing_code | 1 | 0.39% |
| record | product_title | 1 | 0.39% |
| record | shipping_method | 1 | 0.39% |
| record | originalordernumber | 1 | 0.39% |
| record | user_id: ID of the user who submitted the snapshot | 1 | 0.39% |
| record | status: Current status of the exchange rate record | 1 | 0.39% |
| record | snapshot_id: Unique identifier for the snapshot in format currency-yearmonthday | 1 | 0.39% |
| record | rate: Exchange rate in USD per foreign currency unit | 1 | 0.39% |
| record | series | 1 | 0.39% |
| record | purchase_date | 1 | 0.39% |
| record | serial | 1 | 0.39% |
| record | discovery.project | 1 | 0.39% |
| record | discovery.location.state | 1 | 0.39% |
| record | discovery.location.county | 1 | 0.39% |
| record | discovery.location.country | 1 | 0.39% |
| record | physical_characteristics.observations.0.publication | 1 | 0.39% |
| record | diameter.unit | 1 | 0.39% |
| record | title: Title of the research paper | 1 | 0.39% |
| record | institution: University and department affiliation of the authors | 1 | 0.39% |
| record | submission_amount: Financial amount associated with submission or grant | 1 | 0.39% |
| record | cities | 1 | 0.39% |
| record | defense_systems | 1 | 0.39% |
| record | facilities | 1 | 0.39% |
| record | wallet_address: Cryptocurrency wallet address involved in the transaction | 1 | 0.39% |
| record | user_id: Unique identifier for the user performing the transaction | 1 | 0.39% |
| record | functionality_comment | 1 | 0.39% |
| record | app_package | 1 | 0.39% |
| record | sku: Unique product identifier as assigned in inventory | 1 | 0.39% |
| record | rating: Customer rating out of 5 stars | 1 | 0.39% |
| record | material: Primary material composition of the gear | 1 | 0.39% |
| record | weight: Product weight in pounds | 1 | 0.39% |
| record | color: Available color(s) of the product | 1 | 0.39% |
| record | review_length | 1 | 0.39% |
| record | feedback_content | 1 | 0.39% |
| record | service | 1 | 0.39% |
| record | root_cause | 1 | 0.39% |
| record | resolution_time | 1 | 0.39% |
| record | incident_id | 1 | 0.39% |
| record | hire_date: Date when the position is expected to be filled | 1 | 0.39% |
| record | department: Business unit or division where the role is located | 1 | 0.39% |
| record | job_title: Title of the position being advertised | 1 | 0.39% |
| record | location: City and state where the role is based | 1 | 0.39% |
| record | description: Brief summary of the job responsibilities | 1 | 0.39% |
| record | required_qualifications: List of educational or skill-based requirements | 1 | 0.39% |
| record | asteroid.discovery_year | 1 | 0.39% |
| record | asteroid.orbital_characteristics.inclination_unit | 1 | 0.39% |
| record | restaurant_name | 1 | 0.39% |
| record | lengthofstay | 1 | 0.39% |
| record | admitting_diagnosis | 1 | 0.39% |
| record | discharge_instructions | 1 | 0.39% |
| record | bed_number | 1 | 0.39% |
| record | insurance | 1 | 0.39% |
| record | discharge_diagnosis | 1 | 0.39% |
| record | primary_treatment | 1 | 0.39% |
| record | billing_code | 1 | 0.39% |
| record | properties | 1 | 0.39% |
| record | founder: Name of the individual who founded the company | 1 | 0.39% |
| record | founding_year: Year the company was established | 1 | 0.39% |
| record | location: City and state or country where the company is based | 1 | 0.39% |
| record | funding_source: Type of funding received by the company | 1 | 0.39% |
| record | annual_revenue: Annual revenue reported in millions of dollars | 1 | 0.39% |
| record | current_status: Current operational phase of the company | 1 | 0.39% |
| record | business_focus: Core service or product area of the company | 1 | 0.39% |
| record | clientbasesize: Number of clients served across regions | 1 | 0.39% |
| record | tv_program.production.location.institution | 1 | 0.39% |
| record | tv_program.setting.location | 1 | 0.39% |
| record | tv_program.year | 1 | 0.39% |
| record | opponent | 1 | 0.39% |
| record | election_district | 1 | 0.39% |
| record | winner | 1 | 0.39% |
| record | output.unit | 1 | 0.39% |
| record | period | 1 | 0.39% |
| record | battery_life | 1 | 0.39% |
| record | release_date | 1 | 0.39% |
| record | final_match | 1 | 0.39% |
| record | organizer | 1 | 0.39% |
| record | hotel.location.nearby_attractions | 1 | 0.39% |
| record | hotel.hotel_group | 1 | 0.39% |
| record | hotel.brand | 1 | 0.39% |
| record | hotel.location.distance_from_center | 1 | 0.39% |
| record | hotel.room_features | 1 | 0.39% |
| record | institution | 1 | 0.39% |
| record | funding_grant | 1 | 0.39% |
| record | geographic_context | 1 | 0.39% |
| record | study_duration | 1 | 0.39% |
| record | sample_size | 1 | 0.39% |
| record | funding_source | 1 | 0.39% |
| record | intervention | 1 | 0.39% |
| record | treatment_arm | 1 | 0.39% |
| record | gender | 1 | 0.39% |
| record | enrollment_date | 1 | 0.39% |
| record | contact_number | 1 | 0.39% |
| record | lastfollowup | 1 | 0.39% |
| record | insurance_provider | 1 | 0.39% |
| record | question.keywords | 1 | 0.39% |
| record | question.legal_implications | 1 | 0.39% |
| record | details.date_posted | 1 | 0.39% |
| record | property | 1 | 0.39% |
| record | lease_term | 1 | 0.39% |
| record | offer_id | 1 | 0.39% |
| record | passengers_affected | 1 | 0.39% |
| record | referee | 1 | 0.39% |
| record | goals | 1 | 0.39% |
| record | primary_diagnosis | 1 | 0.39% |
| record | treatment_history | 1 | 0.39% |
| record | prior_hospitalization | 1 | 0.39% |
| record | allergies | 1 | 0.39% |
| record | current_status | 1 | 0.39% |
| record | team_mentions.0.name | 1 | 0.39% |
| record | affected_entity: Organization or business impacted by the violation | 1 | 0.39% |
| record | location: Physical address of the affected entity | 1 | 0.39% |
| record | violation_code: Code assigned to the specific violation | 1 | 0.39% |
| record | publication_date | 1 | 0.39% |
| record | result_count | 1 | 0.39% |
| record | school_address | 1 | 0.39% |
| record | speaker: Name of the person delivering the statement | 1 | 0.39% |
| record | statement_reference: Identifier code for the statement document | 1 | 0.39% |
| record | amount: Financial value associated with the policy or proposal | 1 | 0.39% |
| record | priority: Severity level assigned to the ticket | 1 | 0.39% |
| record | ticket_id: Unique identifier for the support case | 1 | 0.39% |
| record | user: Name of the user who filed the ticket | 1 | 0.39% |
| record | mrn: Unique medical record number identifying the patient | 1 | 0.39% |
| record | address: Street address of the patient | 1 | 0.39% |
| record | city: City where the patient resides | 1 | 0.39% |
| record | state: State where the patient resides | 1 | 0.39% |
| record | diagnosis_code: ICD-10 code for the patient's condition | 1 | 0.39% |
| record | stay_duration: Number of days the patient stayed in the hospital | 1 | 0.39% |
| record | treatment: Medical treatment provided during stay | 1 | 0.39% |
| record | prescription: Medication or therapy prescribed | 1 | 0.39% |
| record | followupdoctor: Name of the doctor for follow-up care | 1 | 0.39% |
| record | hospital_name: Name of the hospital where discharge occurred | 1 | 0.39% |
| record | insurance: Insurance provider covering the bill | 1 | 0.39% |
| record | transaction_id: Unique identifier for each transaction entry | 1 | 0.39% |
| record | date: Date when the transaction occurred | 1 | 0.39% |
| record | amount: Monetary value of the transaction | 1 | 0.39% |
| record | description: Brief explanation of the transaction purpose | 1 | 0.39% |
| record | participant: Name of the individual involved in the transaction | 1 | 0.39% |
| record | account: Bank account associated with the transaction | 1 | 0.39% |
| record | fruit_properties.skin.description | 1 | 0.39% |
| record | characteristics | 1 | 0.39% |
| record | concepts | 1 | 0.39% |
| record | designation | 1 | 0.39% |
| record | orbital_characteristics.classification | 1 | 0.39% |
| record | discovery.discoverer | 1 | 0.39% |
| record | discovery.observatory.location.prefecture | 1 | 0.39% |
| record | discovery.observatory.location.country | 1 | 0.39% |
| record | orbital_characteristics.semi_major_axis.unit | 1 | 0.39% |
| record | morphology.stem | 1 | 0.39% |
| record | morphology.flowering_fruiting_period | 1 | 0.39% |
| record | distribution.habitat | 1 | 0.39% |
| record | family | 1 | 0.39% |
| record | morphology.seeds | 1 | 0.39% |
| record | morphology.flower_heads | 1 | 0.39% |
| record | distribution.regions.0.region_name | 1 | 0.39% |
| record | morphology.leaves | 1 | 0.39% |
| record | distribution.regions.1.region_name | 1 | 0.39% |
| record | playertwoid | 1 | 0.39% |
| record | match_type | 1 | 0.39% |
| record | playeronename | 1 | 0.39% |
| record | match_result | 1 | 0.39% |
| record | location.country | 1 | 0.39% |
| record | location.department | 1 | 0.39% |
| record | location.district | 1 | 0.39% |
| record | location.city | 1 | 0.39% |
| record | hotel.name | 1 | 0.39% |
| record | hotel.star_rating | 1 | 0.39% |
| record | booking_options.partners | 1 | 0.39% |
| record | religious_concepts | 1 | 0.39% |
| record | first_name | 1 | 0.39% |
| record | manager | 1 | 0.39% |
| record | last_name | 1 | 0.39% |
| record | attendees | 1 | 0.39% |
| record | meeting_code | 1 | 0.39% |
| record | battery_life: Battery duration in hours under normal use | 1 | 0.39% |
| record | rating: Star rating out of 5 given by the reviewer | 1 | 0.39% |
| record | price: Purchase price in US dollars | 1 | 0.39% |
| record | affiliation.0.type | 1 | 0.39% |
| record | platform: Type of social media post shared | 1 | 0.39% |
| record | author: Name of user who posted the content | 1 | 0.39% |
| record | hotel.distance_from_city_center.value | 1 | 0.39% |
| record | hotel.number_of_rooms | 1 | 0.39% |
| record | hotel.distance_from_city_center.unit | 1 | 0.39% |
| record | hotel.distance_from_airport.value | 1 | 0.39% |
| record | hotel.nearby_attractions | 1 | 0.39% |
| record | hotel.location.region | 1 | 0.39% |
| record | major | 1 | 0.39% |
| record | required_skills | 1 | 0.39% |
| record | job_title | 1 | 0.39% |
| record | hire_date | 1 | 0.39% |
| record | cas_number | 1 | 0.39% |
| record | issuer | 1 | 0.39% |
| record | deal_type | 1 | 0.39% |
| record | enterprise_value | 1 | 0.39% |
| record | consideration_structure | 1 | 0.39% |
| record | target_cfo | 1 | 0.39% |
| record | acquirer_ceo | 1 | 0.39% |
| record | artist | 1 | 0.39% |
| record | record_label | 1 | 0.39% |
| record | agency | 1 | 0.39% |
| record | budget | 1 | 0.39% |
| record | study.focus | 1 | 0.39% |
| record | organization | 1 | 0.39% |
| record | expected_close_date | 1 | 0.39% |
| record | employees | 1 | 0.39% |
| record | ceo | 1 | 0.39% |
| record | status: Current status of the outreach interaction | 1 | 0.39% |
| record | sender: Name of the person sending the outreach | 1 | 0.39% |
| record | outreach_date: Date when the outreach email was sent | 1 | 0.39% |
| record | amount: Monetary value associated with the outreach | 1 | 0.39% |
| record | exam_date | 1 | 0.39% |
| record | study.quotes | 1 | 0.39% |
| record | study.author.name | 1 | 0.39% |
| record | location.county | 1 | 0.39% |
| record | administrative_level | 1 | 0.39% |
| record | abolished_entity | 1 | 0.39% |
| record | administrative_divisions.village_committees | 1 | 0.39% |
| record | manufacturer | 1 | 0.39% |
| record | trial_nct | 1 | 0.39% |
| record | route | 1 | 0.39% |
| record | word_count | 1 | 0.39% |
| record | due_date | 1 | 0.39% |
| record | Steinem <> adopted by | 1 | 0.39% |
| record | Priest <> brought | 1 | 0.39% |
| record | bearded friend <> smoked | 1 | 0.39% |
| record | Steinem <> color | 1 | 0.39% |
| record | Steinem <> front paws on | 1 | 0.39% |
| record | Steinem <> rode in | 1 | 0.39% |
| record | Steinem <> lived for | 1 | 0.39% |
| record | Steinem <> nudged | 1 | 0.39% |
| record | my cousin <> adopted | 1 | 0.39% |
| record | Steinem <> species | 1 | 0.39% |
| record | required_materials | 1 | 0.39% |
| record | presenter | 1 | 0.39% |
| record | attendance_limit | 1 | 0.39% |
| record | techtoolsused | 1 | 0.39% |
| record | points: Points scored by the player | 1 | 0.39% |
| record | score: Final score of the game | 1 | 0.39% |
| record | team: Name of the team playing in the game | 1 | 0.39% |
| record | assists: Assists recorded by the player | 1 | 0.39% |
| record | disputingpartya | 1 | 0.39% |
| record | disputingpartyb | 1 | 0.39% |
| record | mark_name | 1 | 0.39% |
| record | registration_number | 1 | 0.39% |
| record | player_name | 1 | 0.39% |
| record | origin_club | 1 | 0.39% |
| record | destination_club | 1 | 0.39% |
| record | transfer_type | 1 | 0.39% |
| record | goals_scored | 1 | 0.39% |
| record | discrepancy_amount | 1 | 0.39% |
| record | resolution_summary | 1 | 0.39% |
| record | resolution_status | 1 | 0.39% |
| record | underwriter | 1 | 0.39% |
| record | risk_score | 1 | 0.39% |
| record | policyholder | 1 | 0.39% |
| record | description: Detailed account of the issue reported | 1 | 0.39% |
| record | submitter: Name of the employee who submitted the ticket | 1 | 0.39% |
| record | affected_system: Software or hardware system experiencing the problem | 1 | 0.39% |
| record | priority: Urgency level of the ticket | 1 | 0.39% |
| record | south_yemen_conflict.accusations_against_governments.0.government_name | 1 | 0.39% |
| record | federalism_definition.federalism | 1 | 0.39% |
| record | south_yemen_conflict.historical_context.protests_and_killings.casualties | 1 | 0.39% |
| record | south_yemen_conflict.historical_context.actions_after_invasion | 1 | 0.39% |
| record | south_yemen_conflict.accusations_against_governments.1.government_name | 1 | 0.39% |
| record | south_yemen_conflict.historical_context.casualties_and_prisoners | 1 | 0.39% |
| record | federalism_definition.confederation_examples | 1 | 0.39% |
| record | future_outlook | 1 | 0.39% |
| record | federalism_definition.confederation | 1 | 0.39% |
| record | price: Listed sale price in dollars | 1 | 0.39% |
| record | listing_id: Unique identifier for each property listing | 1 | 0.39% |
| record | agent: Name of the real estate agent representing the seller | 1 | 0.39% |
| record | bedrooms: Number of bedrooms in the property | 1 | 0.39% |
| record | address: Full street address of the property | 1 | 0.39% |
| record | patient_gender: Gender of the patient | 1 | 0.39% |
| record | drug_name: Name of the medication involved in the reaction | 1 | 0.39% |
| record | patient_address: Full residential address of the patient | 1 | 0.39% |
| record | report_id: Unique identifier assigned to the adverse event report | 1 | 0.39% |
| record | drug_dosage: Dosage and frequency of the drug exposure | 1 | 0.39% |
| record | asteroid.designation_number | 1 | 0.39% |
| record | asteroid.discovery.discoverer | 1 | 0.39% |
| record | asteroid.discovery.date | 1 | 0.39% |
| record | ticket_id: Unique identifier for the incident ticket | 1 | 0.39% |
| record | filed_at: Timestamp when the ticket was opened | 1 | 0.39% |
| record | region: Geographic region where the issue occurred | 1 | 0.39% |
| record | reporter: Name of the engineer or team member who filed the ticket | 1 | 0.39% |
| record | error_log: Relevant error message or log snippet from the incident | 1 | 0.39% |
| record | root_cause: Hypothesized cause of the incident based on logs or analysis | 1 | 0.39% |
| record | user: Name of the customer who provided the review | 1 | 0.39% |
| record | product_id: Unique identifier for the product reviewed | 1 | 0.39% |
| record | purchase_date: Date when the product was purchased | 1 | 0.39% |
| record | rating: Numerical rating from 1 to 5 given by the user | 1 | 0.39% |
| record | price: Formatted price including dollar sign and cents | 1 | 0.39% |
| record | location: City and state or province where purchase occurred | 1 | 0.39% |
| record | review: Text content of the customer's feedback on the product | 1 | 0.39% |
| record | ppe_required | 1 | 0.39% |
| record | chemical_name | 1 | 0.39% |
| record | hazard_class | 1 | 0.39% |
| record | handling_notes | 1 | 0.39% |
| record | literary_movements | 1 | 0.39% |
| record | places | 1 | 0.39% |
| record | -LRB- | 1 | 0.39% |
| record | . | 1 | 0.39% |
| record | NN | 1 | 0.39% |
| record | IN | 1 | 0.39% |
| record | DT | 1 | 0.39% |
| record | VBD | 1 | 0.39% |
| record | CC | 1 | 0.39% |
| record | compensation_type | 1 | 0.39% |
| record | power_outage | 1 | 0.39% |
| record | total_rainfall | 1 | 0.39% |
| record | funding_allocated | 1 | 0.39% |
| record | gust_speed | 1 | 0.39% |
| record | soil_remediation_project | 1 | 0.39% |
| record | follow_up_survey | 1 | 0.39% |
| record | severity_rating | 1 | 0.39% |
| record | max_wind_speed | 1 | 0.39% |
| record | record_metadata | natural/title ×9, natural/patient_id ×4, natural/name ×3, natural/state ×3 |  |
| transaction | item | 29 | 96.67% |
| transaction | amount | 29 | 96.67% |
| transaction | date | 28 | 93.33% |
| transaction | seller | 27 | 90.00% |
| transaction | buyer | 24 | 80.00% |
| transaction | record_metadata | natural/item ×29, natural/amount ×1 |  |
| vehicle_listing | make | 2 | 100.00% |
| vehicle_listing | model | 2 | 100.00% |
| vehicle_listing | year | 2 | 100.00% |
| vehicle_listing | record_metadata | natural/make ×1 |  |

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | mixed (replay) |
| relations | mixed (replay) |
| events | mixed (replay) |
| classifications | mixed (replay) |
| structures | mixed (replay) |
