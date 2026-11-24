# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replay_eb22g_casualty | train | 33,045 | 32,392 | 204,447 | 14,539 | 24,995 | 1,670 | 18,417 | 26,756 |
| replay_eb22g_casualty | val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| replay_eb22g_casualty | test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## replay_eb22g_casualty

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 33,045 | 32,392 | 334 | 1199 | 20547 | 204,447 | 14,539 | 24,995 | 1,670 | 18,417 | 26,756 |
| val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 18,506 | 0 | 0 | 18,506 | 56.00% |
| zho | 10,969 | 0 | 0 | 10,969 | 33.19% |
| tur | 2,991 | 0 | 0 | 2,991 | 9.05% |
| und | 393 | 0 | 0 | 393 | 1.19% |
| rus | 21 | 0 | 0 | 21 | 0.06% |
| ita | 20 | 0 | 0 | 20 | 0.06% |
| deu | 16 | 0 | 0 | 16 | 0.05% |
| fra | 14 | 0 | 0 | 14 | 0.04% |
| spa | 14 | 0 | 0 | 14 | 0.04% |
| hin | 13 | 0 | 0 | 13 | 0.04% |
| ukr | 13 | 0 | 0 | 13 | 0.04% |
| pol | 12 | 0 | 0 | 12 | 0.04% |
| por | 12 | 0 | 0 | 12 | 0.04% |
| kor | 11 | 0 | 0 | 11 | 0.03% |
| nld | 9 | 0 | 0 | 9 | 0.03% |
| ara | 9 | 0 | 0 | 9 | 0.03% |
| ces | 7 | 0 | 0 | 7 | 0.02% |
| cat | 2 | 0 | 0 | 2 | 0.01% |
| nor | 1 | 0 | 0 | 1 | 0.00% |
| fin | 1 | 0 | 0 | 1 | 0.00% |
| lat | 1 | 0 | 0 | 1 | 0.00% |
| hun | 1 | 0 | 0 | 1 | 0.00% |
| srp | 1 | 0 | 0 | 1 | 0.00% |
| bre | 1 | 0 | 0 | 1 | 0.00% |
| eml | 1 | 0 | 0 | 1 | 0.00% |
| mlg | 1 | 0 | 0 | 1 | 0.00% |
| wuu | 1 | 0 | 0 | 1 | 0.00% |
| pms | 1 | 0 | 0 | 1 | 0.00% |
| ilo | 1 | 0 | 0 | 1 | 0.00% |
| dan | 1 | 0 | 0 | 1 | 0.00% |
| bel | 1 | 0 | 0 | 1 | 0.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 0 |
| test_unique_inputs | 0 |
| train_records | 33,045 |
| train_unique_inputs | 32,392 |
| val_records | 0 |
| val_unique_inputs | 0 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | n/a | 204,447 / 0 |  |
| entities (per mention) | train / test | n/a | 204,447 / 0 |  |
| relations (per instance) | train / val | n/a | 14,539 / 0 |  |
| relations (per instance) | train / test | n/a | 14,539 / 0 |  |
| event types (per trigger mention) | train / val | n/a | 17,333 / 0 |  |
| event types (per trigger mention) | train / test | n/a | 17,333 / 0 |  |
| event roles (per argument) | train / val | n/a | 27,653 / 0 |  |
| event roles (per argument) | train / test | n/a | 27,653 / 0 |  |
| classifications (true_label) | train / val | n/a | 24,995 / 0 |  |
| classifications (true_label) | train / test | n/a | 24,995 / 0 |  |


### Events

**train.** 20,127 instances in 5,586 records, 3,603 triggerless. Trigger gold 18,417. Argument gold 26,756. Arguments per instance, mean 1.37. Trigger length in whitespace tokens, mean 1.26, single-token share 81.8%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Contact.Communicate | 1,524 | 8.79% |
| Sport.Compete | 395 | 2.28% |
| Catastrophe | 375 | 2.16% |
| Competition | 352 | 2.03% |
| Hostile_encounter | 344 | 1.98% |
| Attack | 342 | 1.97% |
| Process_start | 335 | 1.93% |
| Causation | 291 | 1.68% |
| Life.Die | 288 | 1.66% |
| Motion | 262 | 1.51% |
| match_result | 253 | 1.46% |
| Personnel.StartPosition | 251 | 1.45% |
| Personnel.EndPosition | 222 | 1.28% |
| Social_event | 222 | 1.28% |
| Contact.Meet | 208 | 1.20% |
| Conflict.Attack | 204 | 1.18% |
| Killing | 203 | 1.17% |
| Life.Injure | 185 | 1.07% |
| Business.StartOrg | 176 | 1.02% |
| Conquering | 166 | 0.96% |
| Process_end | 166 | 0.96% |
| Coming_to_be | 161 | 0.93% |
| Justice.Arrest | 158 | 0.91% |
| product_publish | 157 | 0.91% |
| Self_motion | 153 | 0.88% |
| Justice.ChargeIndict | 148 | 0.85% |
| Arriving | 143 | 0.83% |
| Award.ReceiveAward | 140 | 0.81% |
| Statement | 140 | 0.81% |
| Cause_change_of_strength | 138 | 0.80% |
| Bodily_harm | 137 | 0.79% |
| Transaction.TransferOwnership | 136 | 0.78% |
| Hold | 126 | 0.73% |
| Military_operation | 124 | 0.72% |
| Cause_change_of_position_on_a_scale | 122 | 0.70% |
| Control | 117 | 0.68% |
| Creating | 117 | 0.68% |
| Justice.TrialHearing | 116 | 0.67% |
| Damaging | 115 | 0.66% |
| Destroying | 112 | 0.65% |
| Death | 109 | 0.63% |
| Becoming | 108 | 0.62% |
| Disaster.Accident | 106 | 0.61% |
| Getting | 103 | 0.59% |
| Know | 98 | 0.57% |
| Arranging | 96 | 0.55% |
| Presence | 95 | 0.55% |
| Earnings_and_losses | 93 | 0.54% |
| Escaping | 93 | 0.54% |
| Influence | 90 | 0.52% |
| Traveling | 89 | 0.51% |
| Placing | 88 | 0.51% |
| Preventing_or_letting | 87 | 0.50% |
| death | 87 | 0.50% |
| judicial_action_arrest | 85 | 0.49% |
| Change | 85 | 0.49% |
| Transaction.TransferMoney | 84 | 0.48% |
| Business.LaunchProduct | 83 | 0.48% |
| resignation | 83 | 0.48% |
| Defending | 80 | 0.46% |
| Agree_or_refuse_to_act | 77 | 0.44% |
| Justice.Sentence | 74 | 0.43% |
| Name_conferral | 74 | 0.43% |
| Motion_directional | 73 | 0.42% |
| Action | 71 | 0.41% |
| Cause_to_be_included | 71 | 0.41% |
| Participation | 70 | 0.40% |
| Communication | 67 | 0.39% |
| Removing | 67 | 0.39% |
| Supporting | 67 | 0.39% |
| Change_of_leadership | 66 | 0.38% |
| Movement.Transport | 65 | 0.38% |
| Conflict.Demonstrate | 63 | 0.36% |
| Request | 62 | 0.36% |
| Using | 62 | 0.36% |
| Movement.Arrive | 59 | 0.34% |
| Supply | 59 | 0.34% |
| Life.BeBorn | 58 | 0.33% |
| Expressing_publicly | 58 | 0.33% |
| Giving | 58 | 0.33% |
| Choosing | 58 | 0.33% |
| Use_firearm | 58 | 0.33% |
| Judgment_communication | 56 | 0.32% |
| championship_win | 51 | 0.29% |
| Bringing | 51 | 0.29% |
| Government.Vote | 51 | 0.29% |
| Recording | 50 | 0.29% |
| Sending | 50 | 0.29% |
| Manufacturing | 49 | 0.28% |
| Assistance | 49 | 0.28% |
| Tech.Release | 49 | 0.28% |
| Come_together | 49 | 0.28% |
| product_recall | 48 | 0.28% |
| Departing | 47 | 0.27% |
| Medical.Treat | 47 | 0.27% |
| Reporting | 46 | 0.27% |
| Justice.Sue | 44 | 0.25% |
| Perception_active | 44 | 0.25% |
| Response | 43 | 0.25% |
| Research.Publish | 42 | 0.24% |
| Transaction.Donate | 41 | 0.24% |
| Besieging | 41 | 0.24% |
| Recovering | 41 | 0.24% |
| Dispersal | 40 | 0.23% |
| Achieve | 40 | 0.23% |
| Disaster.NaturalDisaster | 40 | 0.23% |
| Deciding | 39 | 0.23% |
| Expansion | 39 | 0.23% |
| judicial_action_summon | 38 | 0.22% |
| Research.Discover | 38 | 0.22% |
| Cause_to_amalgamate | 37 | 0.21% |
| Life.Marry | 37 | 0.21% |
| Cause_to_make_progress | 36 | 0.21% |
| Medical.Diagnose | 35 | 0.20% |
| Receiving | 35 | 0.20% |
| life.die.deathcausedbyviolentevents | 34 | 0.20% |
| advancement | 34 | 0.20% |
| Becoming_a_member | 34 | 0.20% |
| Government.EnactLaw | 34 | 0.20% |
| Openness | 34 | 0.20% |
| Collaboration | 34 | 0.20% |
| marriage | 33 | 0.19% |
| judicial_action_fine | 33 | 0.19% |
| Aiming | 32 | 0.18% |
| Scrutiny | 32 | 0.18% |
| Building | 31 | 0.18% |
| Terrorism | 31 | 0.18% |
| Check | 31 | 0.18% |
| joined | 30 | 0.17% |
| opening | 29 | 0.17% |
| Contact.Negotiate | 29 | 0.17% |
| divorce | 29 | 0.17% |
| product_delist | 28 | 0.16% |
| Change_event_time | 28 | 0.16% |
| life.injure.injurycausedbyviolentevents | 28 | 0.16% |
| Personnel.Elect | 27 | 0.16% |
| product_release | 27 | 0.16% |
| Temporary_stay | 26 | 0.15% |
| Legal_rulings | 26 | 0.15% |
| Committing_crime | 26 | 0.15% |
| Publishing | 26 | 0.15% |
| Hindering | 25 | 0.14% |
| conflict.attack.selfdirectedbattle | 25 | 0.14% |
| departure | 25 | 0.14% |
| Disaster.Rescue | 24 | 0.14% |
| Warning | 24 | 0.14% |
| Life.Evacuate | 24 | 0.14% |
| Body_movement | 23 | 0.13% |
| Commerce_sell | 23 | 0.13% |
| apologize_to | 23 | 0.13% |
| financing | 22 | 0.13% |
| Arrest | 22 | 0.13% |
| Traffic Accident | 21 | 0.12% |
| Quarreling | 21 | 0.12% |
| Criminal_investigation | 21 | 0.12% |
| Personnel.Nominate | 20 | 0.12% |
| Research.Fund | 20 | 0.12% |
| Sign_agreement | 20 | 0.12% |
| Telling | 20 | 0.12% |
| Surrendering | 20 | 0.12% |
| Coming_to_believe | 20 | 0.12% |
| conflict.attack.n/a | 20 | 0.12% |
| judicial_action_hearing | 19 | 0.11% |
| Resolve_problem | 19 | 0.11% |
| Fire | 19 | 0.11% |
| birth | 19 | 0.11% |
| Tech.Breach | 19 | 0.11% |
| movement.transportartifact.n/a | 18 | 0.10% |
| breakup | 18 | 0.10% |
| Movement.Migrate | 18 | 0.10% |
| Being_in_operation | 18 | 0.10% |
| judicial_action_imprisonment | 18 | 0.10% |
| competition_withdrawal | 18 | 0.10% |
| Employment | 18 | 0.10% |
| Violence | 18 | 0.10% |
| product_award | 18 | 0.10% |
| Business.EndOrg | 18 | 0.10% |
| contact.discussion.meet | 18 | 0.10% |
| Business.MergeOrg | 17 | 0.10% |
| transaction.transfermoney.n/a | 17 | 0.10% |
| judicial_action_prosecution | 17 | 0.10% |
| Filling | 17 | 0.10% |
| Change_sentiment | 17 | 0.10% |
| Preserving | 17 | 0.10% |
| Justice.Appeal | 17 | 0.10% |
| Legality | 16 | 0.09% |
| Rescuing | 16 | 0.09% |
| price_limit_up | 16 | 0.09% |
| sale_acquisition | 15 | 0.09% |
| Emptying | 15 | 0.09% |
| Disaster.Outbreak | 15 | 0.09% |
| competition_ban | 15 | 0.09% |
| Justice.Fine | 15 | 0.09% |
| Business.DeclareBankruptcy | 14 | 0.08% |
| Releasing | 14 | 0.08% |
| Cure | 14 | 0.08% |
| Collapse | 14 | 0.08% |
| transaction.transfermoney.payforservice | 14 | 0.08% |
| movement.transportperson.n/a | 14 | 0.08% |
| Expend_resource | 14 | 0.08% |
| contact.requestadvise.broadcast | 14 | 0.08% |
| Connect | 14 | 0.08% |
| layoff | 14 | 0.08% |
| Forming_relationships | 14 | 0.08% |
| retirement | 13 | 0.08% |
| life.die.n/a | 13 | 0.08% |
| Convincing | 13 | 0.08% |
| Life.Divorce | 13 | 0.08% |
| personnel.startposition.hiring | 13 | 0.08% |
| dismissal | 13 | 0.08% |
| disaster.fireexplosion.fireexplosion | 13 | 0.08% |
| Kidnapping | 13 | 0.08% |
| contact.commitmentpromiseexpressintent.meet | 12 | 0.07% |
| Confronting_problem | 12 | 0.07% |
| Reforming_a_system | 12 | 0.07% |
| inspection.sensoryobserve.inspectpeopleorganization | 12 | 0.07% |
| Commitment | 12 | 0.07% |
| Government.Protest | 12 | 0.07% |
| conflict.attack.bombing | 12 | 0.07% |
| contact.requestadvise.correspondence | 12 | 0.07% |
| lost_contact | 12 | 0.07% |
| movement.transportperson.preventexit | 11 | 0.06% |
| suspension | 11 | 0.06% |
| Government.Sanction | 11 | 0.06% |
| Cost | 11 | 0.06% |
| GiveUp | 11 | 0.06% |
| engagement | 11 | 0.06% |
| Testing | 11 | 0.06% |
| justice.arrestjaildetain.arrestjaildetain | 11 | 0.06% |
| closing | 11 | 0.06% |
| Education_teaching | 11 | 0.06% |
| contact.discussion.n/a | 11 | 0.06% |
| transaction.transfermoney.purchase | 11 | 0.06% |
| wedding | 11 | 0.06% |
| judicial_action_report | 11 | 0.06% |
| GetReady | 11 | 0.06% |
| birthday_celebration | 11 | 0.06% |
| price_increase | 11 | 0.06% |
| Business.Layoff | 10 | 0.06% |
| Writing | 10 | 0.06% |
| justice.initiatejudicialprocess.chargeindict | 10 | 0.06% |
| price_limit_down | 10 | 0.06% |
| Commerce_pay | 10 | 0.06% |
| Reveal_secret | 10 | 0.06% |
| transaction.transferownership.borrowlend | 10 | 0.06% |
| price_decrease | 10 | 0.06% |
| thank | 10 | 0.06% |
| personnel.endposition.quitretire | 10 | 0.06% |
| Plane Crash | 10 | 0.06% |
| Earthquake | 10 | 0.06% |
| Protest | 10 | 0.06% |
| Justice.Acquit | 9 | 0.05% |
| Adducing | 9 | 0.05% |
| Having_or_lacking_access | 9 | 0.05% |
| contact.publicstatementinperson.broadcast | 9 | 0.05% |
| dissolution | 9 | 0.05% |
| transaction.transferownership.n/a | 9 | 0.05% |
| Movement.Deport | 9 | 0.05% |
| transaction.transfermoney.embargosanction | 9 | 0.05% |
| proposal | 9 | 0.05% |
| personnel.endposition.firinglayoff | 9 | 0.05% |
| Award | 9 | 0.05% |
| ipo | 9 | 0.05% |
| government.agreements.acceptagreementcontractceasefire | 8 | 0.05% |
| judicial_action_filing | 8 | 0.05% |
| government.agreements.n/a | 8 | 0.05% |
| Hiding_objects | 8 | 0.05% |
| life.injure.illnessdegradationhungerthirst | 8 | 0.05% |
| Containing | 8 | 0.05% |
| Research | 8 | 0.05% |
| Rewards_and_punishments | 8 | 0.05% |
| artifactexistence.damagedestroy.destroy | 8 | 0.05% |
| Limiting | 8 | 0.05% |
| Wearing | 8 | 0.05% |
| contact.requestadvise.meet | 8 | 0.05% |
| meet_with | 8 | 0.05% |
| contact.publicstatementinperson.n/a | 8 | 0.05% |
| Exchange | 8 | 0.05% |
| Lighting | 8 | 0.05% |
| rate_decrease | 7 | 0.04% |
| movement.transportartifact.preventexit | 7 | 0.04% |
| contact.commitmentpromiseexpressintent.broadcast | 7 | 0.04% |
| justice.initiatejudicialprocess.trialhearing | 7 | 0.04% |
| inspection.sensoryobserve.physicalinvestigateinspect | 7 | 0.04% |
| transaction.transferownership.purchase | 7 | 0.04% |
| justice.judicialconsequences.convict | 7 | 0.04% |
| movement.transportperson.smuggleextract | 7 | 0.04% |
| transaction.transaction.embargosanction | 7 | 0.04% |
| strike | 7 | 0.04% |
| movement.transportartifact.grantentry | 7 | 0.04% |
| artifactexistence.damagedestroy.damage | 7 | 0.04% |
| Commerce_buy | 7 | 0.04% |
| disaster.accidentcrash.accidentcrash | 7 | 0.04% |
| transaction.transfermoney.giftgrantprovideaid | 7 | 0.04% |
| visit_on_set | 7 | 0.04% |
| transaction.transferownership.giftgrantprovideaid | 7 | 0.04% |
| Patrolling | 7 | 0.04% |
| justice.judicialconsequences.execute | 7 | 0.04% |
| contact.threatencoerce.broadcast | 7 | 0.04% |
| contact.requestadvise.n/a | 7 | 0.04% |
| government.vote.castvote | 6 | 0.03% |
| government.agreements.rejectnullifyagreementcontractceasefire | 6 | 0.03% |
| personnel.elect.winelection | 6 | 0.03% |
| conflict.attack.airstrikemissilestrike | 6 | 0.03% |
| movement.transportperson.selfmotion | 6 | 0.03% |
| contact.negotiate.meet | 6 | 0.03% |
| movement.transportperson.grantentryasylum | 6 | 0.03% |
| praise | 6 | 0.03% |
| movement.transportperson.evacuationrescue | 6 | 0.03% |
| Scouring | 6 | 0.03% |
| Revenge | 6 | 0.03% |
| conflict.attack.stealrobhijack | 6 | 0.03% |
| artifactexistence.damagedestroy.n/a | 6 | 0.03% |
| Create_artwork | 6 | 0.03% |
| government.agreements.violateagreement | 6 | 0.03% |
| Explosion | 6 | 0.03% |
| personnel.endposition.n/a | 6 | 0.03% |
| transaction.transfermoney.borrowlend | 6 | 0.03% |
| transaction.transaction.giftgrantprovideaid | 6 | 0.03% |
| Flood | 6 | 0.03% |
| Robbery | 6 | 0.03% |
| government.legislate.legislate | 6 | 0.03% |
| Prison | 6 | 0.03% |
| Surrounding | 6 | 0.03% |
| movement.transportartifact.sendsupplyexport | 5 | 0.03% |
| contact.funeralvigil.meet | 5 | 0.03% |
| contact.collaborate.correspondence | 5 | 0.03% |
| justice.initiatejudicialprocess.n/a | 5 | 0.03% |
| Carry_goods | 5 | 0.03% |
| Ratification | 5 | 0.03% |
| life.injure.illnessdegradationphysical | 5 | 0.03% |
| Imposing_obligation | 5 | 0.03% |
| life.injure.n/a | 5 | 0.03% |
| Practice | 5 | 0.03% |
| transaction.transaction.n/a | 5 | 0.03% |
| justice.judicialconsequences.n/a | 5 | 0.03% |
| pregnancy | 5 | 0.03% |
| movement.transportartifact.smuggleextract | 5 | 0.03% |
| contact.threatencoerce.correspondence | 5 | 0.03% |
| movement.transportartifact.disperseseparate | 5 | 0.03% |
| movement.transportperson.prevententry | 5 | 0.03% |
| conflict.attack.firearmattack | 5 | 0.03% |
| contact.commitmentpromiseexpressintent.correspondence | 5 | 0.03% |
| transaction.transferownership.embargosanction | 5 | 0.03% |
| contact.threatencoerce.meet | 4 | 0.02% |
| Bearing_arms | 4 | 0.02% |
| Vocalizations | 4 | 0.02% |
| contact.commitmentpromiseexpressintent.n/a | 4 | 0.02% |
| conflict.demonstrate.marchprotestpoliticalgathering | 4 | 0.02% |
| contact.discussion.correspondence | 4 | 0.02% |
| contact.commandorder.n/a | 4 | 0.02% |
| contact.collaborate.n/a | 4 | 0.02% |
| transaction.transaction.transfercontrol | 4 | 0.02% |
| justice.judicialconsequences.extradite | 4 | 0.02% |
| contact.commandorder.broadcast | 4 | 0.02% |
| justice.investigate.n/a | 4 | 0.02% |
| government.formation.mergegpe | 4 | 0.02% |
| movement.transportartifact.receiveimport | 4 | 0.02% |
| manufacture.artifact.n/a | 4 | 0.02% |
| life.die.nonviolentdeath | 4 | 0.02% |
| movement.transportartifact.fall | 4 | 0.02% |
| Submitting_documents | 4 | 0.02% |
| Rite | 4 | 0.02% |
| conflict.attack.stabbing | 4 | 0.02% |
| contract_termination | 4 | 0.02% |
| infidelity | 3 | 0.02% |
| government.formation.n/a | 3 | 0.02% |
| Renting | 3 | 0.02% |
| Business.Strike | 3 | 0.02% |
| contact.prevarication.correspondence | 3 | 0.02% |
| march | 3 | 0.02% |
| Theft | 3 | 0.02% |
| conflict.demonstrate.n/a | 3 | 0.02% |
| Conflict.Ceasefire | 3 | 0.02% |
| movement.transportartifact.bringcarryunload | 3 | 0.02% |
| contact.commandorder.correspondence | 3 | 0.02% |
| conflict.attack.invade | 3 | 0.02% |
| inspection.sensoryobserve.monitorelection | 3 | 0.02% |
| contact.collaborate.meet | 3 | 0.02% |
| justice.investigate.investigatecrime | 3 | 0.02% |
| contact.prevarication.broadcast | 3 | 0.02% |
| manufacture.artifact.createmanufacture | 2 | 0.01% |
| movement.transportartifact.hide | 2 | 0.01% |
| conflict.yield.n/a | 2 | 0.01% |
| rate_increase | 2 | 0.01% |
| conflict.yield.retreat | 2 | 0.01% |
| movement.transportperson.bringcarryunload | 2 | 0.01% |
| inspection.sensoryobserve.n/a | 2 | 0.01% |
| movement.transportartifact.nonviolentthrowlaunch | 2 | 0.01% |
| contact.prevarication.meet | 2 | 0.01% |
| movement.transportartifact.prevententry | 2 | 0.01% |
| contact.mediastatement.n/a | 2 | 0.01% |
| government.spy.spy | 2 | 0.01% |
| Labeling | 2 | 0.01% |
| Justifying | 2 | 0.01% |
| contact.negotiate.correspondence | 2 | 0.01% |
| movement.transportperson.hide | 2 | 0.01% |
| movement.transportperson.fall | 2 | 0.01% |
| manufacture.artifact.build | 2 | 0.01% |
| government.vote.violationspreventvote | 2 | 0.01% |
| contact.threatencoerce.n/a | 2 | 0.01% |
| Experiment | 1 | 0.01% |
| Suspicion | 1 | 0.01% |
| contact.prevarication.n/a | 1 | 0.01% |
| movement.transportperson.disperseseparate | 1 | 0.01% |
| manufacture.artifact.createintellectualproperty | 1 | 0.01% |
| contact.negotiate.n/a | 1 | 0.01% |
| Transaction.Lend | 1 | 0.01% |
| conflict.yield.surrender | 1 | 0.01% |
| personnel.elect.n/a | 1 | 0.01% |
| contact.commandorder.meet | 1 | 0.01% |
| Ingestion | 1 | 0.01% |
| conflict.attack.setfire | 1 | 0.01% |
| Institutionalization | 1 | 0.01% |
| government.vote.n/a | 1 | 0.01% |
| government.formation.startgpe | 1 | 0.01% |
| conflict.attack.hanging | 1 | 0.01% |
| contact.funeralvigil.n/a | 1 | 0.01% |
| Risk | 1 | 0.01% |
| Emergency | 1 | 0.01% |
| conflict.attack.biologicalchemicalpoisonattack | 1 | 0.01% |
| personnel.startposition.n/a | 1 | 0.01% |
| Support | 0 | 0.00% |
| Accident | 0 | 0.00% |
| Cyber.Databreach | 0 | 0.00% |
| Exhibit | 0 | 0.00% |
| Deploy | 0 | 0.00% |
| Movement.Transportation.Unspecified | 0 | 0.00% |
| Conflict.Attack.DetonateExplode | 0 | 0.00% |
| Contact.Contact.Meet | 0 | 0.00% |
| Cyber.DiscoverVulnerability | 0 | 0.00% |
| Cyber.PatchVulnerability | 0 | 0.00% |
| Cyber.Ransom | 0 | 0.00% |
| Injure | 0 | 0.00% |
| Life.Injure.Unspecified | 0 | 0.00% |
| Manoeuvre | 0 | 0.00% |
| Conflict | 0 | 0.00% |
| Cyber.Phishing | 0 | 0.00% |
| Justice.ChargeIndict.Unspecified | 0 | 0.00% |
| Life.Die.Unspecified | 0 | 0.00% |
| Conflict.Attack.Unspecified | 0 | 0.00% |
| Justice.TrialHearing.Unspecified | 0 | 0.00% |
| Justice.ArrestJailDetain.Unspecified | 0 | 0.00% |
| Justice.InvestigateCrime.Unspecified | 0 | 0.00% |
| Transaction.ExchangeBuySell.Unspecified | 0 | 0.00% |
| Disaster.Crash.Unspecified | 0 | 0.00% |
| Contact.Contact.Broadcast | 0 | 0.00% |
| Cognitive.IdentifyCategorize.Unspecified | 0 | 0.00% |
| ArtifactExistence.DamageDestroyDisableDismantle.Damage | 0 | 0.00% |
| Control.ImpedeInterfereWith.Unspecified | 0 | 0.00% |
| ArtifactExistence.DamageDestroyDisableDismantle.Destroy | 0 | 0.00% |
| Justice.Sentence.Unspecified | 0 | 0.00% |
| Justice.Convict.Unspecified | 0 | 0.00% |
| ArtifactExistence.ManufactureAssemble.Unspecified | 0 | 0.00% |
| Medical.Intervention.Unspecified | 0 | 0.00% |
| Justice.ReleaseParole.Unspecified | 0 | 0.00% |
| Cognitive.Research.Unspecified | 0 | 0.00% |
| Contact.Contact.Unspecified | 0 | 0.00% |
| Life.Infect.Unspecified | 0 | 0.00% |
| ArtifactExistence.DamageDestroyDisableDismantle.DisableDefuse | 0 | 0.00% |
| Cognitive.Inspection.SensoryObserve | 0 | 0.00% |
| GenericCrime.GenericCrime.GenericCrime | 0 | 0.00% |
| ArtifactExistence.DamageDestroyDisableDismantle.Unspecified | 0 | 0.00% |
| Contact.RequestCommand.Unspecified | 0 | 0.00% |

**val.** No event instances.

**test.** No event instances.

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1 | 10,399 | 0.01% | 0 | 1 | 21 | 0 | 10,377 | 17,254 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

**train by role.**

| role | inside | placed | inside share | unmeasured |
| --- | --- | --- | --- | --- |
| Subject | 0 | 2,568 | 0.00% | 15 |
| Equipment | 1 | 1,262 | 0.08% | 6 |
| Date | 0 | 1,023 | 0.00% | 611 |
| Object | 0 | 777 | 0.00% | 2 |
| Materials | 0 | 733 | 0.00% | 7 |
| Location | 0 | 671 | 0.00% | 144 |
| Victim | 0 | 418 | 0.00% | 383 |
| Militaryforce | 0 | 340 | 0.00% | 3 |
| Vulnerability | 0 | 324 | 0.00% | 0 |
| Vulnerable_System | 0 | 240 | 0.00% | 0 |
| Compromised-Data | 0 | 171 | 0.00% | 0 |
| Attacker | 0 | 170 | 0.00% | 90 |
| Discoverer | 0 | 160 | 0.00% | 0 |
| Time | 0 | 147 | 0.00% | 1,942 |
| Patch | 0 | 104 | 0.00% | 0 |
| Place | 0 | 102 | 0.00% | 1,264 |
| Releaser | 0 | 95 | 0.00% | 0 |
| Tool | 0 | 93 | 0.00% | 0 |
| Attack-Pattern | 0 | 84 | 0.00% | 0 |
| Content | 0 | 67 | 0.00% | 0 |
| Result | 0 | 65 | 0.00% | 103 |
| Capabilities | 0 | 64 | 0.00% | 0 |
| Trusted-Entity | 0 | 63 | 0.00% | 0 |
| CVE | 0 | 55 | 0.00% | 0 |
| Vulnerable_System_Version | 0 | 51 | 0.00% | 0 |
| Target | 0 | 45 | 0.00% | 112 |
| Artifact | 0 | 44 | 0.00% | 141 |
| Purpose | 0 | 44 | 0.00% | 0 |
| Vulnerable_System_Owner | 0 | 42 | 0.00% | 0 |
| Quantity | 0 | 34 | 0.00% | 8 |
| Area | 0 | 30 | 0.00% | 0 |
| Defendant | 0 | 30 | 0.00% | 353 |
| ExplosiveDevice | 0 | 29 | 0.00% | 0 |
| Price | 0 | 27 | 0.00% | 103 |
| Supported_Platform | 0 | 23 | 0.00% | 0 |
| Detainee | 0 | 18 | 0.00% | 0 |
| Number-of-Victim | 0 | 16 | 0.00% | 0 |
| Destination | 0 | 14 | 0.00% | 118 |
| Issues-Addressed | 0 | 13 | 0.00% | 0 |
| PassengerArtifact | 0 | 12 | 0.00% | 0 |
| Payment-Method | 0 | 12 | 0.00% | 0 |
| Instrument | 0 | 11 | 0.00% | 62 |
| Investigator | 0 | 11 | 0.00% | 0 |
| Participant | 0 | 11 | 0.00% | 444 |
| Communicator | 0 | 8 | 0.00% | 1,225 |
| Number-of-Data | 0 | 8 | 0.00% | 0 |
| Patch-Number | 0 | 8 | 0.00% | 0 |
| Origin | 0 | 7 | 0.00% | 53 |
| IdentifiedObject | 0 | 6 | 0.00% | 0 |
| Injurer | 0 | 6 | 0.00% | 0 |
| Prosecutor | 0 | 6 | 0.00% | 35 |
| Identifier | 0 | 4 | 0.00% | 0 |
| Jailer | 0 | 4 | 0.00% | 0 |
| Researcher | 0 | 4 | 0.00% | 31 |
| Transporter | 0 | 4 | 0.00% | 0 |
| AcquiredEntity | 0 | 3 | 0.00% | 0 |
| Vehicle | 0 | 3 | 0.00% | 50 |
| Damage-Amount | 0 | 2 | 0.00% | 0 |
| Giver | 0 | 2 | 0.00% | 72 |
| Killer | 0 | 2 | 0.00% | 0 |
| Perpetrator | 0 | 2 | 0.00% | 0 |
| CrashObject | 0 | 1 | 0.00% | 0 |
| IdentifiedRole | 0 | 1 | 0.00% | 0 |
| Impeder | 0 | 1 | 0.00% | 0 |
| JudgeCourt | 0 | 1 | 0.00% | 0 |
| Observer | 0 | 1 | 0.00% | 0 |
| Recipient | 0 | 1 | 0.00% | 479 |
| Topic | 0 | 1 | 0.00% | 11 |

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.81 | 31.84% | 1.99 | 33.14% | 0.05% | 10,401 | 17,252 | 3,603 | 17,333 |
| val | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |
| test | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 20,127 | 2.53% | 16,831 | 12.79% | 12.15% | 2.51% | 1.06% | 14.29% |
| val | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |
| test | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

| split | docs | clusters | mentions | multi-mention | singletons | mention pairs | pronoun mentions | pronoun share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3,395 | 94,543 | 188,247 | 26,371 | 68,172 | 674,459 | 73,369 | 38.97% |
| val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |
| test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 509 | 1,250 | 17,653 |
| val | 0 | 0 | 0 |
| test | 0 | 0 | 0 |

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

| split | mentions the string rule adds | arguments skipped, 2+ clusters | arguments in one multi-mention cluster | other mentions of that cluster | argument spans in 2+ clusters |
| --- | --- | --- | --- | --- | --- |
| train | 1,236 | 85 | 346 | 2,292 | 0 |
| val | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 |

### Entities

#### train

| label | count | share |
| --- | --- | --- |
| person | 17,868 | 8.74% |
| location | 10,861 | 5.31% |
| date | 9,560 | 4.68% |
| Location | 8,937 | 4.37% |
| Date | 8,095 | 3.96% |
| cardinal | 5,868 | 2.87% |
| organization | 5,635 | 2.76% |
| Organization | 4,851 | 2.37% |
| StartDate | 3,872 | 1.89% |
| job title | 3,737 | 1.83% |
| Product | 3,700 | 1.81% |
| EquityHolder | 3,510 | 1.72% |
| StockCode | 3,137 | 1.53% |
| money | 3,021 | 1.48% |
| Casualties and Losses | 2,896 | 1.42% |
| geopolitical entity | 2,800 | 1.37% |
| CompanyName | 2,687 | 1.31% |
| StockAbbr | 2,664 | 1.30% |
| company | 2,549 | 1.25% |
| PledgedShares | 2,446 | 1.20% |
| sports team | 2,396 | 1.17% |
| Cause | 2,104 | 1.03% |
| media outlet | 2,016 | 0.99% |
| Pledgee | 1,993 | 0.97% |
| TotalHoldingShares | 1,950 | 0.95% |
| Quantity | 1,855 | 0.91% |
| TradedShares | 1,845 | 0.90% |
| TotalHoldingRatio | 1,777 | 0.87% |
| Person | 1,746 | 0.85% |
| percent | 1,585 | 0.78% |
| Reduction Amount | 1,529 | 0.75% |
| facility | 1,442 | 0.71% |
| TotalPledgedShares | 1,381 | 0.68% |
| product | 1,332 | 0.65% |
| age | 1,265 | 0.62% |
| Measurement | 1,256 | 0.61% |
| duration | 1,254 | 0.61% |
| quantity | 1,172 | 0.57% |
| EndDate | 1,161 | 0.57% |
| Reducing Shareholder | 1,144 | 0.56% |
| ReleasedDate | 1,128 | 0.55% |
| time | 1,112 | 0.54% |
| government agency | 1,072 | 0.52% |
| occupation | 1,020 | 0.50% |
| work of art | 976 | 0.48% |
| Reduction Start Date | 845 | 0.41% |
| Announcement Date | 824 | 0.40% |
| Company Name | 810 | 0.40% |
| PER | 805 | 0.39% |
| Policy Content | 748 | 0.37% |
| educational institution | 706 | 0.35% |
| event | 699 | 0.34% |
| Money | 649 | 0.32% |
| Event | 621 | 0.30% |
| LaterHoldingShares | 616 | 0.30% |
| nationality | 601 | 0.29% |
| Contest Participant | 600 | 0.29% |
| Nationality | 596 | 0.29% |
| Affected Areas | 497 | 0.24% |
| Areas affected | 495 | 0.24% |
| Facility | 491 | 0.24% |
| Attacker | 488 | 0.24% |
| Participants | 466 | 0.23% |
| AveragePrice | 457 | 0.22% |
| crime | 456 | 0.22% |
| pledge_amount | 453 | 0.22% |
| disease | 447 | 0.22% |
| region | 446 | 0.22% |
| Protesters | 433 | 0.21% |
| Candidates and their Political Parties | 419 | 0.20% |
| ordinal | 418 | 0.20% |
| Target | 418 | 0.20% |
| Weapon and Equippment | 416 | 0.20% |
| Suspect | 403 | 0.20% |
| Occupation | 403 | 0.20% |
| measurement | 386 | 0.19% |
| address | 374 | 0.18% |
| Protest Reason | 373 | 0.18% |
| Affected Area | 366 | 0.18% |
| Deliberating Agency | 366 | 0.18% |
| family relationship | 354 | 0.17% |
| Disease | 332 | 0.16% |
| pledgor | 331 | 0.16% |
| Charged Crime | 330 | 0.16% |
| Person under Investigation | 330 | 0.16% |
| Recipient | 328 | 0.16% |
| RepurchasedShares | 322 | 0.16% |
| sports competition | 313 | 0.15% |
| Storm Hit Location | 310 | 0.15% |
| Time | 306 | 0.15% |
| Accepting Court | 302 | 0.15% |
| Method | 301 | 0.15% |
| Game Name | 301 | 0.15% |
| HighestTradingPrice | 298 | 0.15% |
| distance | 293 | 0.14% |
| Vulnerability | 291 | 0.14% |
| Belligerents | 285 | 0.14% |
| Loss Amount | 272 | 0.13% |
| GPE | 272 | 0.13% |
| Investigative Agency | 263 | 0.13% |
| Precautionary Measure | 263 | 0.13% |
| protein | 262 | 0.13% |
| brand | 256 | 0.13% |
| The Charged Crime | 248 | 0.12% |
| Position | 242 | 0.12% |
| Individual_protein | 240 | 0.12% |
| political party | 239 | 0.12% |
| Accident Investigator | 236 | 0.12% |
| ORG | 232 | 0.11% |
| Criminal Evidence | 225 | 0.11% |
| Increase Amount | 221 | 0.11% |
| RepurchaseAmount | 221 | 0.11% |
| ClosingDate | 220 | 0.11% |
| Winner | 220 | 0.11% |
| law | 216 | 0.11% |
| film | 215 | 0.11% |
| chemical | 213 | 0.10% |
| Anti-pollution People/Organizations | 206 | 0.10% |
| Accusation | 205 | 0.10% |
| pledge_start_date | 205 | 0.10% |
| Host | 204 | 0.10% |
| Frozen Amount | 204 | 0.10% |
| Summit Theme | 204 | 0.10% |
| road | 203 | 0.10% |
| Champions | 201 | 0.10% |
| landmark | 200 | 0.10% |
| Scheduled Landing Place | 198 | 0.10% |
| award | 197 | 0.10% |
| DiseaseOrPhenotypicFeature | 197 | 0.10% |
| Award | 196 | 0.10% |
| Increasing Shareholder | 196 | 0.10% |
| Accused People | 195 | 0.10% |
| Contracting Parties | 194 | 0.09% |
| military rank | 192 | 0.09% |
| Influence People | 192 | 0.09% |
| Confirmed/Infected Cases | 191 | 0.09% |
| song | 190 | 0.09% |
| GeneOrGeneProduct | 187 | 0.09% |
| The Final Votes and Percentages | 185 | 0.09% |
| Location/Hospital | 182 | 0.09% |
| ruling_date | 181 | 0.09% |
| Strike Reason | 181 | 0.09% |
| Score | 180 | 0.09% |
| non-profit organization | 179 | 0.09% |
| Arrest Location | 177 | 0.09% |
| temperature | 176 | 0.09% |
| Flight No. | 176 | 0.09% |
| FAC | 175 | 0.09% |
| Police | 174 | 0.09% |
| Aid Agency | 170 | 0.08% |
| Country Visited | 166 | 0.08% |
| Election Name | 166 | 0.08% |
| Aircraft Agency | 164 | 0.08% |
| Survivors | 164 | 0.08% |
| Pollution Source | 163 | 0.08% |
| Capabilities | 161 | 0.08% |
| Strike Outcome | 160 | 0.08% |
| Magnitude | 159 | 0.08% |
| Visitor | 158 | 0.08% |
| Death Cases | 158 | 0.08% |
| Economic Loss | 158 | 0.08% |
| vehicle | 157 | 0.08% |
| ethnic group | 156 | 0.08% |
| Deceased | 154 | 0.08% |
| Husband | 153 | 0.07% |
| Judgement Result/Prison Term | 152 | 0.07% |
| System | 151 | 0.07% |
| publication | 150 | 0.07% |
| era | 150 | 0.07% |
| Wife | 150 | 0.07% |
| academic degree | 149 | 0.07% |
| political affiliation | 147 | 0.07% |
| frozen_shareholder | 146 | 0.07% |
| military unit | 145 | 0.07% |
| pledge_end_date | 144 | 0.07% |
| OtherScientificTerm | 144 | 0.07% |
| Competition Items | 143 | 0.07% |
| Agreement Content | 140 | 0.07% |
| medication | 139 | 0.07% |
| software | 138 | 0.07% |
| Crew | 136 | 0.07% |
| Epicenter | 136 | 0.07% |
| Policy Name & Abbreviation | 132 | 0.06% |
| Profession | 129 | 0.06% |
| Increase Start Date | 129 | 0.06% |
| Award Field | 129 | 0.06% |
| Protest Slogan | 129 | 0.06% |
| Speaker | 129 | 0.06% |
| Solution | 128 | 0.06% |
| LowestTradingPrice | 127 | 0.06% |
| team role | 126 | 0.06% |
| Effective Date | 126 | 0.06% |
| Attending Hospital | 126 | 0.06% |
| LegalInstitution | 121 | 0.06% |
| Artifacts and Their Chronology | 121 | 0.06% |
| ChemicalEntity | 119 | 0.06% |
| Freeze Start Date | 117 | 0.06% |
| Outbreak Location | 117 | 0.06% |
| Resignated Person | 115 | 0.06% |
| regulatory body | 113 | 0.06% |
| Epidemic Data Issuing Agency | 112 | 0.05% |
| Damaged Facility | 112 | 0.05% |
| name | 109 | 0.05% |
| Discover Location | 107 | 0.05% |
| Number of Vehicles Involved in the Crash | 106 | 0.05% |
| FrozeShares | 105 | 0.05% |
| Speaker Status | 103 | 0.05% |
| Passengers | 102 | 0.05% |
| Prosecutor | 102 | 0.05% |
| Death Reason | 102 | 0.05% |
| weapon | 101 | 0.05% |
| Historical Sites | 100 | 0.05% |
| Age | 99 | 0.05% |
| People/Organization who predicted the disaster | 99 | 0.05% |
| Time of the Case | 98 | 0.05% |
| Appointee | 98 | 0.05% |
| symptom | 97 | 0.05% |
| Award Reason | 97 | 0.05% |
| Resign Reason | 96 | 0.05% |
| Spectacle Location | 96 | 0.05% |
| frequency | 95 | 0.05% |
| book | 94 | 0.05% |
| PII | 94 | 0.05% |
| Patch | 93 | 0.05% |
| Court | 91 | 0.04% |
| People | 91 | 0.04% |
| Beneficiary | 91 | 0.04% |
| pseudonym | 90 | 0.04% |
| Weapon | 89 | 0.04% |
| Record Breaker | 89 | 0.04% |
| Areas Affected | 86 | 0.04% |
| Habitat | 84 | 0.04% |
| honorific | 84 | 0.04% |
| Arrest Time | 84 | 0.04% |
| Sponsor | 84 | 0.04% |
| Riot Reason | 83 | 0.04% |
| Software | 83 | 0.04% |
| Device | 83 | 0.04% |
| Agreement Name | 82 | 0.04% |
| Employment Agency | 82 | 0.04% |
| Strikers | 82 | 0.04% |
| Grades | 82 | 0.04% |
| Start Date | 81 | 0.04% |
| species | 80 | 0.04% |
| Number of Evacuated People | 80 | 0.04% |
| Organization Name | 79 | 0.04% |
| Employed Institution | 79 | 0.04% |
| Policy Proposals | 79 | 0.04% |
| Record-breaking Project | 79 | 0.04% |
| WEA | 78 | 0.04% |
| Start Time | 78 | 0.04% |
| Number of Damaged Houses | 78 | 0.04% |
| Host Country | 77 | 0.04% |
| body of water | 76 | 0.04% |
| religious group | 76 | 0.04% |
| Participating Countries | 76 | 0.04% |
| Storm Hit Time | 76 | 0.04% |
| Symptoms | 76 | 0.04% |
| Strikers Status | 76 | 0.04% |
| GeneProtein | 75 | 0.04% |
| LOC | 74 | 0.04% |
| Way for Spreading | 74 | 0.04% |
| regulation | 73 | 0.04% |
| speed | 73 | 0.04% |
| Achievement | 73 | 0.04% |
| Susceptible Population | 73 | 0.04% |
| Storm Name | 73 | 0.04% |
| Affected Industries | 73 | 0.04% |
| Generic | 73 | 0.04% |
| market index | 71 | 0.03% |
| Microorganism | 71 | 0.03% |
| Gene/protein/RNA | 71 | 0.03% |
| court case | 70 | 0.03% |
| Evidence | 70 | 0.03% |
| Related Rivers or Lakes | 70 | 0.03% |
| Conflict Duration | 70 | 0.03% |
| holiday | 69 | 0.03% |
| Released People | 69 | 0.03% |
| protein-family | 69 | 0.03% |
| Task | 69 | 0.03% |
| Aid Supplies/Amount | 68 | 0.03% |
| The Agency Who Broke the Agreement | 67 | 0.03% |
| Spacecraft | 66 | 0.03% |
| Strike Agency | 66 | 0.03% |
| Spectacle Start Time | 66 | 0.03% |
| operating system | 65 | 0.03% |
| Data | 65 | 0.03% |
| Types of the Spectacle | 64 | 0.03% |
| organism | 63 | 0.03% |
| Area Affected | 63 | 0.03% |
| compensation_amount | 63 | 0.03% |
| Funding | 63 | 0.03% |
| OrganismTaxon | 63 | 0.03% |
| Acquiree | 63 | 0.03% |
| area | 62 | 0.03% |
| currency | 62 | 0.03% |
| Previous Record Holder | 62 | 0.03% |
| Amount of Precipitation | 61 | 0.03% |
| Arrested | 61 | 0.03% |
| senior_personnel | 59 | 0.03% |
| senior_position | 59 | 0.03% |
| Bank Name | 59 | 0.03% |
| Election Goal | 59 | 0.03% |
| Fined Agency | 59 | 0.03% |
| Government Reaction | 59 | 0.03% |
| Defense Lawyer | 58 | 0.03% |
| compensation_recipient | 58 | 0.03% |
| Party Responsible for Pollution | 58 | 0.03% |
| Victim | 57 | 0.03% |
| Freeze End Date | 57 | 0.03% |
| Regulatory Authority | 57 | 0.03% |
| anatomical structure | 56 | 0.03% |
| News Release Agency | 56 | 0.03% |
| weight | 55 | 0.03% |
| video game | 55 | 0.03% |
| Perpetrators | 55 | 0.03% |
| Archaeologist | 55 | 0.03% |
| Aid Reason | 54 | 0.03% |
| Version | 53 | 0.03% |
| criminal organization | 52 | 0.03% |
| Taking-off Place | 52 | 0.03% |
| fiscal quarter | 51 | 0.02% |
| stock ticker | 51 | 0.02% |
| Research Agency | 51 | 0.02% |
| Fine Reason | 51 | 0.02% |
| model number | 50 | 0.02% |
| Summit Name | 50 | 0.02% |
| Missings | 50 | 0.02% |
| Boycotted Institutions | 50 | 0.02% |
| airport | 49 | 0.02% |
| VEH | 49 | 0.02% |
| Symptom | 47 | 0.02% |
| End Time | 47 | 0.02% |
| File | 47 | 0.02% |
| CVE | 46 | 0.02% |
| Last Job of the Appointee | 46 | 0.02% |
| Head of Institution | 46 | 0.02% |
| Organization Industry | 46 | 0.02% |
| Anatomy | 46 | 0.02% |
| Discover Time | 45 | 0.02% |
| aircraft | 44 | 0.02% |
| Purpose | 44 | 0.02% |
| Predecessor | 44 | 0.02% |
| Spokenmen | 44 | 0.02% |
| Head of Investigation Team | 43 | 0.02% |
| Army | 43 | 0.02% |
| medical procedure | 42 | 0.02% |
| language | 42 | 0.02% |
| Launch Date | 42 | 0.02% |
| conference | 41 | 0.02% |
| Number of Influenced People | 41 | 0.02% |
| Maximum Wind Speed | 40 | 0.02% |
| Material | 40 | 0.02% |
| Witness | 40 | 0.02% |
| sentence | 39 | 0.02% |
| Storm Warning Level | 39 | 0.02% |
| Military Strength | 39 | 0.02% |
| Treatment Method | 39 | 0.02% |
| CellType | 39 | 0.02% |
| Start_Date | 38 | 0.02% |
| Spokeswoman/Spokesman | 38 | 0.02% |
| Child | 37 | 0.02% |
| Judge | 37 | 0.02% |
| The Grades of the Previous Record Holder | 37 | 0.02% |
| Appointer | 37 | 0.02% |
| Pests | 37 | 0.02% |
| Phenotype | 36 | 0.02% |
| Marriage Duration | 36 | 0.02% |
| Turnout(who win who lost) | 36 | 0.02% |
| MVP | 36 | 0.02% |
| deadline | 35 | 0.02% |
| Bill Drafting Agency | 35 | 0.02% |
| Investigating agency | 35 | 0.02% |
| treaty | 35 | 0.02% |
| Death Time | 35 | 0.02% |
| Archaeologist Organization | 35 | 0.02% |
| Spacecraft Mission | 35 | 0.02% |
| Duration | 34 | 0.02% |
| Outbreak Date | 34 | 0.02% |
| id | 34 | 0.02% |
| Ship No. | 34 | 0.02% |
| Number of Destroyed Building | 34 | 0.02% |
| Penalty Amount | 34 | 0.02% |
| The Final Seats Result | 34 | 0.02% |
| hardware component | 33 | 0.02% |
| dosage | 33 | 0.02% |
| Perpetrator | 33 | 0.02% |
| Rescue Organizer | 33 | 0.02% |
| SequenceVariant | 33 | 0.02% |
| Website | 33 | 0.02% |
| Astronauts | 32 | 0.02% |
| Prosecution Lawyer | 32 | 0.02% |
| Number of Rescued People | 31 | 0.02% |
| Release Reason | 31 | 0.02% |
| End Date | 31 | 0.02% |
| Maximum Rainfall | 31 | 0.02% |
| Court Time | 31 | 0.02% |
| Malware | 30 | 0.01% |
| description | 29 | 0.01% |
| Military Exercise | 29 | 0.01% |
| DNA | 29 | 0.01% |
| Volcano Name | 29 | 0.01% |
| Declarer | 29 | 0.01% |
| Appointment Time | 29 | 0.01% |
| Countries Withdrawing from the Organization | 28 | 0.01% |
| Approver | 28 | 0.01% |
| Number of People Hospitalized | 28 | 0.01% |
| Scale | 27 | 0.01% |
| Doctor and Medical Team | 27 | 0.01% |
| Launch Country | 27 | 0.01% |
| State Before Death | 26 | 0.01% |
| Weapon Used | 26 | 0.01% |
| Term of Office | 26 | 0.01% |
| Countries Joined the Organization | 26 | 0.01% |
| Jail Time | 26 | 0.01% |
| Region | 25 | 0.01% |
| Economic loss | 25 | 0.01% |
| Tear Up Reason | 25 | 0.01% |
| Illness | 25 | 0.01% |
| Poisoning Type | 25 | 0.01% |
| Ways to Watch the Speech | 25 | 0.01% |
| Lasting Time | 25 | 0.01% |
| Cooperative Agency | 25 | 0.01% |
| Rescue Tool or Method | 25 | 0.01% |
| postal code | 24 | 0.01% |
| Warning Device | 24 | 0.01% |
| Reasons for the Formation of the Historical sites | 24 | 0.01% |
| Electoral System | 24 | 0.01% |
| Number | 24 | 0.01% |
| Wedding Venue | 24 | 0.01% |
| vessel | 23 | 0.01% |
| protein-complex | 23 | 0.01% |
| Annoncement Date | 23 | 0.01% |
| Publication | 23 | 0.01% |
| natural disaster | 22 | 0.01% |
| Storm Direction | 22 | 0.01% |
| Other Loss | 22 | 0.01% |
| Train No. | 22 | 0.01% |
| protein-domain | 22 | 0.01% |
| The Sentence Claimed by the Prosecutor's Lawyer | 22 | 0.01% |
| Last Time the Record was Broken | 22 | 0.01% |
| Species | 22 | 0.01% |
| Organization Members | 22 | 0.01% |
| Carrier Rocket | 21 | 0.01% |
| Release Time | 21 | 0.01% |
| Number of Victims | 21 | 0.01% |
| Train Agency | 21 | 0.01% |
| Candidate Scandal(Allegations of fraud, etc.) | 21 | 0.01% |
| title | 21 | 0.01% |
| Damaged Crops & Livestock | 20 | 0.01% |
| Spectacle Duration | 20 | 0.01% |
| Strike Industry | 20 | 0.01% |
| witness | 19 | 0.01% |
| Commanders and their Position | 19 | 0.01% |
| Successor | 19 | 0.01% |
| Water Level | 19 | 0.01% |
| Shipwreck Reason | 19 | 0.01% |
| Concept | 19 | 0.01% |
| Age of the Resignated Person | 19 | 0.01% |
| war | 18 | 0.01% |
| mountain | 18 | 0.01% |
| Launch Site | 18 | 0.01% |
| Launch Result | 18 | 0.01% |
| Casualty Count | 18 | 0.01% |
| Temporary Settlement | 18 | 0.01% |
| Transportation | 18 | 0.01% |
| People Casting Key Votes | 18 | 0.01% |
| Prison Term | 18 | 0.01% |
| Economists who Predicted the Crisis | 18 | 0.01% |
| CellLine | 18 | 0.01% |
| Influenced People | 18 | 0.01% |
| Influenced Crops and Livelihood | 18 | 0.01% |
| Number of Vaccinated People | 18 | 0.01% |
| clinical trial | 17 | 0.01% |
| Sentencing Location | 17 | 0.01% |
| VAL | 17 | 0.01% |
| Magnitude(Tsunami heights) | 17 | 0.01% |
| Death Rate | 17 | 0.01% |
| UnfrozeDate | 17 | 0.01% |
| Damaged Property | 17 | 0.01% |
| Company Industry | 17 | 0.01% |
| Voting method | 17 | 0.01% |
| Acquirer | 17 | 0.01% |
| role | 17 | 0.01% |
| Goal | 16 | 0.01% |
| Inauguration Time | 16 | 0.01% |
| Disaster-Stricken Farmland | 16 | 0.01% |
| TTL | 15 | 0.01% |
| Storm Center Location | 15 | 0.01% |
| Metric | 15 | 0.01% |
| Agreement Validity Period | 15 | 0.01% |
| Age of the Appointee | 15 | 0.01% |
| Hostage | 15 | 0.01% |
| Country | 15 | 0.01% |
| Detention Start Time | 15 | 0.01% |
| Acquisition amount | 15 | 0.01% |
| Casualities and Losses | 15 | 0.01% |
| Projects | 15 | 0.01% |
| contract | 14 | 0.01% |
| financial instrument | 14 | 0.01% |
| trade union | 14 | 0.01% |
| Number of Aftershocks | 14 | 0.01% |
| Head of the Institution | 14 | 0.01% |
| cell | 14 | 0.01% |
| Forecasting Agency | 14 | 0.01% |
| Head of the Merged Organization | 14 | 0.01% |
| Storm Movement Speed | 14 | 0.01% |
| Response Measures | 14 | 0.01% |
| RNA | 14 | 0.01% |
| Group | 14 | 0.01% |
| Medication | 14 | 0.01% |
| Phylum | 14 | 0.01% |
| Mission Duration | 13 | 0.01% |
| Unemployed Rate | 13 | 0.01% |
| Gene | 13 | 0.01% |
| Other Impact | 13 | 0.01% |
| Organization Leader | 13 | 0.01% |
| PaymentMethod | 12 | 0.01% |
| Tsunamis | 12 | 0.01% |
| Announce Platform | 12 | 0.01% |
| Protein_family_or_group | 12 | 0.01% |
| pathogen | 11 | 0.01% |
| city | 11 | 0.01% |
| state | 11 | 0.01% |
| license | 11 | 0.01% |
| allergen | 11 | 0.01% |
| Agreement Members | 11 | 0.01% |
| Number of Trapped People | 11 | 0.01% |
| Alternate Landing Place | 11 | 0.01% |
| Last Outbreak Time | 11 | 0.01% |
| Fire Warning Level | 11 | 0.01% |
| Live Broadcast Platform | 11 | 0.01% |
| Investigating Spokesperson | 11 | 0.01% |
| experimental-construct | 11 | 0.01% |
| experiment-tag | 11 | 0.01% |
| ai_model | 11 | 0.01% |
| mineral | 10 | 0.00% |
| target | 10 | 0.00% |
| reagent | 10 | 0.00% |
| tags | 10 | 0.00% |
| Archaeological Site | 10 | 0.00% |
| protocol | 9 | 0.00% |
| interest rate | 9 | 0.00% |
| tax | 9 | 0.00% |
| Stolen Amount | 9 | 0.00% |
| Ship | 9 | 0.00% |
| Service Years | 9 | 0.00% |
| COM | 9 | 0.00% |
| Best Way to Shoot | 9 | 0.00% |
| Complications | 9 | 0.00% |
| Condition | 9 | 0.00% |
| disclosure | 9 | 0.00% |
| page | 9 | 0.00% |
| file format | 8 | 0.00% |
| Recovered Amount | 8 | 0.00% |
| border | 8 | 0.00% |
| Production Cuts | 8 | 0.00% |
| type | 8 | 0.00% |
| Hull Discovery Time | 8 | 0.00% |
| agent | 8 | 0.00% |
| Refuge | 8 | 0.00% |
| The State of the Volcano (Dormant or Active) | 8 | 0.00% |
| email | 8 | 0.00% |
| Invited Person | 8 | 0.00% |
| Prison | 8 | 0.00% |
| Object | 8 | 0.00% |
| Planet | 8 | 0.00% |
| images | 8 | 0.00% |
| content | 8 | 0.00% |
| patent | 7 | 0.00% |
| security vulnerability | 7 | 0.00% |
| price | 7 | 0.00% |
| status | 7 | 0.00% |
| Name | 7 | 0.00% |
| unitOfMeasure | 7 | 0.00% |
| Registered Capital | 7 | 0.00% |
| The Sentence Claimed by the Defense Lawyer | 7 | 0.00% |
| Child Custody | 7 | 0.00% |
| Withdraw Reason | 7 | 0.00% |
| Commanders of the Army | 7 | 0.00% |
| Term | 7 | 0.00% |
| Text | 7 | 0.00% |
| Cured Cases | 7 | 0.00% |
| The Worst-Hit Area | 7 | 0.00% |
| model | 7 | 0.00% |
| strategies | 7 | 0.00% |
| Social Group | 7 | 0.00% |
| programming language | 6 | 0.00% |
| Hull Location | 6 | 0.00% |
| Economy Loss | 6 | 0.00% |
| Number of Damaged House | 6 | 0.00% |
| Special Medicine | 6 | 0.00% |
| phone | 6 | 0.00% |
| Invalid Date | 6 | 0.00% |
| Provisional Government | 6 | 0.00% |
| Merger Terms | 6 | 0.00% |
| Language | 6 | 0.00% |
| room_number | 6 | 0.00% |
| Adjective | 6 | 0.00% |
| notes | 6 | 0.00% |
| objectives | 6 | 0.00% |
| workCenter | 6 | 0.00% |
| Medical Condition | 6 | 0.00% |
| range | 6 | 0.00% |
| algorithm | 5 | 0.00% |
| voltage | 5 | 0.00% |
| Description | 5 | 0.00% |
| framework | 5 | 0.00% |
| process | 5 | 0.00% |
| Death Age | 5 | 0.00% |
| Tsunami Warning Level | 5 | 0.00% |
| zip | 5 | 0.00% |
| CellComponent | 5 | 0.00% |
| Suspected Cases | 5 | 0.00% |
| values | 5 | 0.00% |
| Bankrupt Business | 5 | 0.00% |
| Time Unit | 5 | 0.00% |
| Compensation | 5 | 0.00% |
| Programming Language | 5 | 0.00% |
| Organism | 5 | 0.00% |
| Substance | 5 | 0.00% |
| Document | 5 | 0.00% |
| ageRange | 5 | 0.00% |
| Art Form | 5 | 0.00% |
| architecture_component | 5 | 0.00% |
| Tic Type | 5 | 0.00% |
| verdict | 4 | 0.00% |
| gene | 4 | 0.00% |
| diagnosis | 4 | 0.00% |
| startDate | 4 | 0.00% |
| Technology | 4 | 0.00% |
| leadTime | 4 | 0.00% |
| amino-acid | 4 | 0.00% |
| Trapped Depth | 4 | 0.00% |
| Sequelae | 4 | 0.00% |
| Spectacle End Time | 4 | 0.00% |
| Postpone Reason | 4 | 0.00% |
| Head of the Government | 4 | 0.00% |
| protein-motif | 4 | 0.00% |
| Wedding Dress Designer | 4 | 0.00% |
| firstName | 4 | 0.00% |
| lastName | 4 | 0.00% |
| Religion | 4 | 0.00% |
| Test | 4 | 0.00% |
| Vessel | 4 | 0.00% |
| mutation | 4 | 0.00% |
| geometric_curve | 4 | 0.00% |
| group | 4 | 0.00% |
| zipCode | 4 | 0.00% |
| fusion-protein | 4 | 0.00% |
| length | 4 | 0.00% |
| Art Movement | 4 | 0.00% |
| Phenomenon | 4 | 0.00% |
| evaluation_metric | 4 | 0.00% |
| Structure | 4 | 0.00% |
| metrics | 4 | 0.00% |
| Process | 4 | 0.00% |
| endDate | 3 | 0.00% |
| assay | 3 | 0.00% |
| Project | 3 | 0.00% |
| productId | 3 | 0.00% |
| url | 3 | 0.00% |
| How Many Times Get Married | 3 | 0.00% |
| organelle | 3 | 0.00% |
| Culture | 3 | 0.00% |
| Ship Agency | 3 | 0.00% |
| priority | 3 | 0.00% |
| phoneNumber | 3 | 0.00% |
| End_Date | 3 | 0.00% |
| Role | 3 | 0.00% |
| Software Component | 3 | 0.00% |
| Operating System | 3 | 0.00% |
| Time Period | 3 | 0.00% |
| Moon | 3 | 0.00% |
| deliveryDate | 3 | 0.00% |
| Trapped Days | 3 | 0.00% |
| code | 3 | 0.00% |
| country | 3 | 0.00% |
| peptide | 3 | 0.00% |
| level | 3 | 0.00% |
| unit | 3 | 0.00% |
| targetDate | 3 | 0.00% |
| gender | 3 | 0.00% |
| Treatment | 3 | 0.00% |
| mathematical_construct | 3 | 0.00% |
| decoder_architecture | 3 | 0.00% |
| streetAddress | 3 | 0.00% |
| Military Unit | 3 | 0.00% |
| operationNumber | 3 | 0.00% |
| standardTime | 3 | 0.00% |
| cast | 3 | 0.00% |
| dataset | 3 | 0.00% |
| metric | 3 | 0.00% |
| amount | 3 | 0.00% |
| GRIStandard | 3 | 0.00% |
| UNGCPrinciple | 3 | 0.00% |
| SASBStandard | 3 | 0.00% |
| interests | 3 | 0.00% |
| channels | 3 | 0.00% |
| Natural Feature | 3 | 0.00% |
| Subphylum | 3 | 0.00% |
| Building | 3 | 0.00% |
| Technique | 3 | 0.00% |
| notification_methods | 3 | 0.00% |
| communication_channels | 3 | 0.00% |
| procedures_to_test | 3 | 0.00% |
| Reinforcement Material | 3 | 0.00% |
| Resin Type | 3 | 0.00% |
| api | 2 | 0.00% |
| vaccine | 2 | 0.00% |
| hotelId | 2 | 0.00% |
| reservationId | 2 | 0.00% |
| numGuests | 2 | 0.00% |
| StreetAddress | 2 | 0.00% |
| ZipCode | 2 | 0.00% |
| TelephoneNumber | 2 | 0.00% |
| Email | 2 | 0.00% |
| LanguageCode | 2 | 0.00% |
| INF | 2 | 0.00% |
| supplier | 2 | 0.00% |
| cost | 2 | 0.00% |
| Organization_Military | 2 | 0.00% |
| Entity | 2 | 0.00% |
| street | 2 | 0.00% |
| number | 2 | 0.00% |
| amenities | 2 | 0.00% |
| productName | 2 | 0.00% |
| serialNumber | 2 | 0.00% |
| attachments | 2 | 0.00% |
| Time for Dignitaries to Resign | 2 | 0.00% |
| Lawyer | 2 | 0.00% |
| userId | 2 | 0.00% |
| dateOfBirth | 2 | 0.00% |
| medicalConditions | 2 | 0.00% |
| units | 2 | 0.00% |
| instructions | 2 | 0.00% |
| color | 2 | 0.00% |
| Literature | 2 | 0.00% |
| Deity | 2 | 0.00% |
| Body Part | 2 | 0.00% |
| Activity | 2 | 0.00% |
| floor_number | 2 | 0.00% |
| protein-region | 2 | 0.00% |
| Source | 2 | 0.00% |
| Grammatical Concept | 2 | 0.00% |
| Hypothetical Planet | 2 | 0.00% |
| purchaseOrderId | 2 | 0.00% |
| supplierId | 2 | 0.00% |
| itemId | 2 | 0.00% |
| itemName | 2 | 0.00% |
| itemDescription | 2 | 0.00% |
| itemQuantity | 2 | 0.00% |
| itemPrice | 2 | 0.00% |
| issueNumber | 2 | 0.00% |
| publisher | 2 | 0.00% |
| publicationDate | 2 | 0.00% |
| Practice | 2 | 0.00% |
| year | 2 | 0.00% |
| expirationDate | 2 | 0.00% |
| Doctor | 2 | 0.00% |
| durability | 2 | 0.00% |
| Therapy | 2 | 0.00% |
| latitude | 2 | 0.00% |
| longitude | 2 | 0.00% |
| Complication | 2 | 0.00% |
| Protein_complex | 2 | 0.00% |
| mobileNumber | 2 | 0.00% |
| alternateMobileNumber | 2 | 0.00% |
| error_correcting_code | 2 | 0.00% |
| hardware_element | 2 | 0.00% |
| contactNumber | 2 | 0.00% |
| Join Conditions | 2 | 0.00% |
| Regulation | 2 | 0.00% |
| Resource | 2 | 0.00% |
| partNumber | 2 | 0.00% |
| standardCost | 2 | 0.00% |
| listPrice | 2 | 0.00% |
| reorderPoint | 2 | 0.00% |
| safetyStockLevel | 2 | 0.00% |
| locationCode | 2 | 0.00% |
| workOrderNumber | 2 | 0.00% |
| dueDate | 2 | 0.00% |
| setupTime | 2 | 0.00% |
| genres | 2 | 0.00% |
| artist | 2 | 0.00% |
| sku | 2 | 0.00% |
| width | 2 | 0.00% |
| height | 2 | 0.00% |
| technical_method | 2 | 0.00% |
| value | 2 | 0.00% |
| rate | 2 | 0.00% |
| referenceNumber | 2 | 0.00% |
| Pest | 2 | 0.00% |
| Equipment | 2 | 0.00% |
| Damage | 2 | 0.00% |
| itemNumber | 2 | 0.00% |
| texts | 2 | 0.00% |
| Political Entity | 2 | 0.00% |
| Writing System | 2 | 0.00% |
| Play Type | 2 | 0.00% |
| Cured Rate | 2 | 0.00% |
| Military_Unit | 2 | 0.00% |
| Biological Group | 2 | 0.00% |
| Geological Period | 2 | 0.00% |
| testName | 2 | 0.00% |
| Medical Test | 2 | 0.00% |
| baseline_method | 2 | 0.00% |
| method_component | 2 | 0.00% |
| database | 1 | 0.00% |
| budget line | 1 | 0.00% |
| lat | 1 | 0.00% |
| lng | 1 | 0.00% |
| roomTypeId | 1 | 0.00% |
| capacity | 1 | 0.00% |
| totalPrice | 1 | 0.00% |
| PlanType | 1 | 0.00% |
| ResiliencyLevel | 1 | 0.00% |
| City | 1 | 0.00% |
| State | 1 | 0.00% |
| ParentName | 1 | 0.00% |
| protein-isoform | 1 | 0.00% |
| SID | 1 | 0.00% |
| lastUpdatedDate | 1 | 0.00% |
| network | 1 | 0.00% |
| season | 1 | 0.00% |
| episode | 1 | 0.00% |
| source | 1 | 0.00% |
| rating | 1 | 0.00% |
| years_in_operation | 1 | 0.00% |
| number_of_employees | 1 | 0.00% |
| term | 1 | 0.00% |
| ABS | 1 | 0.00% |
| Media | 1 | 0.00% |
| Artifact | 1 | 0.00% |
| modelNumber | 1 | 0.00% |
| purchaseDate | 1 | 0.00% |
| purchasePrice | 1 | 0.00% |
| storePurchased | 1 | 0.00% |
| warrantyPeriod | 1 | 0.00% |
| warrantyCoverage | 1 | 0.00% |
| warrantyExclusions | 1 | 0.00% |
| warrantyLimitations | 1 | 0.00% |
| additionalNotes | 1 | 0.00% |
| createDate | 1 | 0.00% |
| updateDate | 1 | 0.00% |
| State of the Hull | 1 | 0.00% |
| Storm Formation Time | 1 | 0.00% |
| creator | 1 | 0.00% |
| Abbreviation | 1 | 0.00% |
| approvalStatus | 1 | 0.00% |
| prescriptionId | 1 | 0.00% |
| patientId | 1 | 0.00% |
| medicationId | 1 | 0.00% |
| refills | 1 | 0.00% |
| pharmacyId | 1 | 0.00% |
| datePrescribed | 1 | 0.00% |
| dateExpires | 1 | 0.00% |
| size | 1 | 0.00% |
| Field of Study | 1 | 0.00% |
| Philosophy | 1 | 0.00% |
| beds | 1 | 0.00% |
| Postpone Time | 1 | 0.00% |
| Property Division | 1 | 0.00% |
| Linguistic Concept | 1 | 0.00% |
| Platform | 1 | 0.00% |
| Standard | 1 | 0.00% |
| Software Version | 1 | 0.00% |
| Database | 1 | 0.00% |
| Publication Issue | 1 | 0.00% |
| Celestial System | 1 | 0.00% |
| Religious Text | 1 | 0.00% |
| Star | 1 | 0.00% |
| Scientific Theory | 1 | 0.00% |
| Celestial Object | 1 | 0.00% |
| Team | 1 | 0.00% |
| purchaseOrderDate | 1 | 0.00% |
| supplierName | 1 | 0.00% |
| supplierAddress | 1 | 0.00% |
| totalCost | 1 | 0.00% |
| paymentTerms | 1 | 0.00% |
| shippingAddress | 1 | 0.00% |
| coverPrice | 1 | 0.00% |
| synopsis | 1 | 0.00% |
| Society | 1 | 0.00% |
| Crime | 1 | 0.00% |
| make | 1 | 0.00% |
| licensePlateNumber | 1 | 0.00% |
| additionalInstructions | 1 | 0.00% |
| cardNumber | 1 | 0.00% |
| cvv | 1 | 0.00% |
| damage | 1 | 0.00% |
| defense | 1 | 0.00% |
| health | 1 | 0.00% |
| mana | 1 | 0.00% |
| strength | 1 | 0.00% |
| agility | 1 | 0.00% |
| intelligence | 1 | 0.00% |
| dependencies | 1 | 0.00% |
| unitPrice | 1 | 0.00% |
| createdBy | 1 | 0.00% |
| Disorder | 1 | 0.00% |
| Disorder Group | 1 | 0.00% |
| Demographic | 1 | 0.00% |
| Toxin | 1 | 0.00% |
| seconds | 1 | 0.00% |
| appId | 1 | 0.00% |
| appName | 1 | 0.00% |
| appVersion | 1 | 0.00% |
| networkType | 1 | 0.00% |
| appLanguage | 1 | 0.00% |
| business_name | 1 | 0.00% |
| education | 1 | 0.00% |
| institution | 1 | 0.00% |
| methods | 1 | 0.00% |
| expectedOutcomes | 1 | 0.00% |
| dateSubmitted | 1 | 0.00% |
| Cultural Artifact | 1 | 0.00% |
| Legal Concept | 1 | 0.00% |
| Economic System | 1 | 0.00% |
| Identity | 1 | 0.00% |
| externalId | 1 | 0.00% |
| zipcode | 1 | 0.00% |
| scoutingDate | 1 | 0.00% |
| humidity | 1 | 0.00% |
| windSpeed | 1 | 0.00% |
| precipitation | 1 | 0.00% |
| visibility | 1 | 0.00% |
| error_type | 1 | 0.00% |
| middleName | 1 | 0.00% |
| suffix | 1 | 0.00% |
| relationship | 1 | 0.00% |
| emailAddress | 1 | 0.00% |
| qualification | 1 | 0.00% |
| route | 1 | 0.00% |
| dispensingDate | 1 | 0.00% |
| quantityDispensed | 1 | 0.00% |
| refillsAllowed | 1 | 0.00% |
| signature | 1 | 0.00% |
| Rescue Start Time | 1 | 0.00% |
| Element | 1 | 0.00% |
| director | 1 | 0.00% |
| runtime | 1 | 0.00% |
| contact | 1 | 0.00% |
| packagingInstructions | 1 | 0.00% |
| architectural_component | 1 | 0.00% |
| posterSize | 1 | 0.00% |
| artworkType | 1 | 0.00% |
| designRequest | 1 | 0.00% |
| specialRequests | 1 | 0.00% |
| receiptNumber | 1 | 0.00% |
| fee | 1 | 0.00% |
| idNumber | 1 | 0.00% |
| Scientific Name | 1 | 0.00% |
| Control Method | 1 | 0.00% |
| Strategy | 1 | 0.00% |
| Natural Enemy | 1 | 0.00% |
| Lure | 1 | 0.00% |
| scheduleId | 1 | 0.00% |
| scheduleDate | 1 | 0.00% |
| roomId | 1 | 0.00% |
| numNights | 1 | 0.00% |
| Statistic | 1 | 0.00% |
| Definition | 1 | 0.00% |
| font | 1 | 0.00% |
| fontSize | 1 | 0.00% |
| employeeTurnoverRate | 1 | 0.00% |
| objective | 1 | 0.00% |
| videos | 1 | 0.00% |
| parameter | 1 | 0.00% |
| roomNumber | 1 | 0.00% |
| roomType | 1 | 0.00% |
| issuedAt | 1 | 0.00% |
| expiresAt | 1 | 0.00% |
| Film | 1 | 0.00% |
| Book | 1 | 0.00% |
| Behavior | 1 | 0.00% |
| training_method | 1 | 0.00% |
| mathematical_framework | 1 | 0.00% |
| hardware | 1 | 0.00% |
| data_representation | 1 | 0.00% |
| Genetic Element | 1 | 0.00% |
| releaseDocumentId | 1 | 0.00% |
| releaseDate | 1 | 0.00% |
| batchNumber | 1 | 0.00% |
| batchSubNumber | 1 | 0.00% |
| manufacturerId | 1 | 0.00% |
| manufacturerName | 1 | 0.00% |
| distributorId | 1 | 0.00% |
| distributorName | 1 | 0.00% |
| storageCondition | 1 | 0.00% |
| testResult | 1 | 0.00% |
| authorizedPerson | 1 | 0.00% |
| Movement | 1 | 0.00% |
| Style | 1 | 0.00% |
| Subject | 1 | 0.00% |
| Food | 1 | 0.00% |
| Classification | 1 | 0.00% |
| research_task | 1 | 0.00% |
| input | 1 | 0.00% |
| constraint | 1 | 0.00% |
| intermediate_representation | 1 | 0.00% |
| output | 1 | 0.00% |
| actor | 1 | 0.00% |
| Exit Conditions | 1 | 0.00% |
| Year | 1 | 0.00% |
| coverArtTitle | 1 | 0.00% |
| coverArtImage | 1 | 0.00% |
| coverArtDescription | 1 | 0.00% |
| series | 1 | 0.00% |
| handling_code | 1 | 0.00% |
| totalScore | 1 | 0.00% |
| comments | 1 | 0.00% |
| responsibilities | 1 | 0.00% |
| primary_contact | 1 | 0.00% |
| backup_contact | 1 | 0.00% |
| evaluation_criteria | 1 | 0.00% |
| responsible_person | 1 | 0.00% |
| Material Category | 1 | 0.00% |
| Material Component | 1 | 0.00% |
| Organization/Person | 1 | 0.00% |
| Infrastructure | 1 | 0.00% |
| Person/Group | 1 | 0.00% |
| credit rating | 0 | 0.00% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| no relation | 1,456 | 10.01% |
| works_for | 747 | 5.14% |
| member_of | 437 | 3.01% |
| located_in | 401 | 2.76% |
| bind | 356 | 2.45% |
| affiliated_with | 302 | 2.08% |
| based_in | 294 | 2.02% |
| leader_of | 256 | 1.76% |
| headquartered_in | 248 | 1.71% |
| employer_of | 247 | 1.70% |
| PPI | 242 | 1.66% |
| instance of | 201 | 1.38% |
| Association | 191 | 1.31% |
| competitor_of | 188 | 1.29% |
| founded | 164 | 1.13% |
| USED-FOR | 164 | 1.13% |
| part_of | 153 | 1.05% |
| produced_by | 143 | 0.98% |
| occupation | 140 | 0.96% |
| shares border with | 138 | 0.95% |
| family name | 122 | 0.84% |
| Lives_In | 119 | 0.82% |
| given name | 110 | 0.76% |
| operates_in | 109 | 0.75% |
| acquired | 93 | 0.64% |
| Positive_Correlation | 93 | 0.64% |
| country | 92 | 0.63% |
| place of birth | 87 | 0.60% |
| reports_to | 87 | 0.60% |
| colleague_of | 83 | 0.57% |
| member of | 82 | 0.56% |
| located in the administrative territorial entity | 76 | 0.52% |
| country of citizenship | 66 | 0.45% |
| participated_in | 65 | 0.45% |
| relative_of | 60 | 0.41% |
| founded by | 57 | 0.39% |
| parent_of | 54 | 0.37% |
| represents | 53 | 0.36% |
| director | 52 | 0.36% |
| supplier_of | 51 | 0.35% |
| developed_by | 51 | 0.35% |
| has part(s) | 49 | 0.34% |
| Negative_Correlation | 49 | 0.34% |
| replaces | 46 | 0.32% |
| contains the administrative territorial entity | 46 | 0.32% |
| subsidiary_of | 45 | 0.31% |
| genre | 45 | 0.31% |
| educated at | 45 | 0.31% |
| father | 45 | 0.31% |
| board_member_of | 44 | 0.30% |
| directed_by | 42 | 0.29% |
| participant in | 42 | 0.29% |
| language used | 40 | 0.28% |
| notable work | 40 | 0.28% |
| followed by | 38 | 0.26% |
| owned by | 38 | 0.26% |
| CONJUNCTION | 38 | 0.26% |
| participant | 37 | 0.25% |
| creator | 37 | 0.25% |
| sibling | 37 | 0.25% |
| position held | 37 | 0.25% |
| performer | 37 | 0.25% |
| location | 36 | 0.25% |
| spouse | 36 | 0.25% |
| different from | 36 | 0.25% |
| headquarters location | 36 | 0.25% |
| award received | 35 | 0.24% |
| place of death | 35 | 0.24% |
| languages spoken, written or signed | 34 | 0.23% |
| part of | 34 | 0.23% |
| continent | 34 | 0.23% |
| conflict | 34 | 0.23% |
| owns | 33 | 0.23% |
| citizen_of | 33 | 0.23% |
| spouse_of | 33 | 0.23% |
| child | 33 | 0.23% |
| mother | 32 | 0.22% |
| present in work | 31 | 0.21% |
| published_by | 30 | 0.21% |
| architect | 30 | 0.21% |
| author | 30 | 0.21% |
| field of work | 30 | 0.21% |
| significant person | 29 | 0.20% |
| named after | 29 | 0.20% |
| member of political party | 28 | 0.19% |
| significant event | 28 | 0.19% |
| EVALUATE-FOR | 28 | 0.19% |
| applies to jurisdiction | 28 | 0.19% |
| Exhibits | 28 | 0.19% |
| country of origin | 27 | 0.19% |
| manufactured_by | 27 | 0.19% |
| child_of | 27 | 0.19% |
| has use | 26 | 0.18% |
| said to be the same as | 26 | 0.18% |
| main subject | 26 | 0.18% |
| located in/on physical feature | 26 | 0.18% |
| partially coincident with | 25 | 0.17% |
| characters | 25 | 0.17% |
| head of government | 25 | 0.17% |
| mountain range | 25 | 0.17% |
| sport | 24 | 0.17% |
| platform | 24 | 0.17% |
| based on | 24 | 0.17% |
| uses | 23 | 0.16% |
| replaced by | 23 | 0.16% |
| nominated for | 23 | 0.16% |
| designed_by | 23 | 0.16% |
| sibling_of | 23 | 0.16% |
| chairperson | 23 | 0.16% |
| connects with | 22 | 0.15% |
| cast member | 22 | 0.15% |
| writing language | 22 | 0.15% |
| connecting line | 22 | 0.15% |
| employer | 22 | 0.15% |
| original broadcaster | 22 | 0.15% |
| family | 21 | 0.14% |
| home venue | 21 | 0.14% |
| work location | 21 | 0.14% |
| parent organization | 21 | 0.14% |
| made from material | 21 | 0.14% |
| PART-OF | 21 | 0.14% |
| occupant | 21 | 0.14% |
| subclass of | 21 | 0.14% |
| HYPONYM-OF | 21 | 0.14% |
| member of sports team | 20 | 0.14% |
| composer | 20 | 0.14% |
| league | 20 | 0.14% |
| conferred by | 20 | 0.14% |
| product or material produced | 20 | 0.14% |
| noble title | 20 | 0.14% |
| country for sport | 20 | 0.14% |
| COMPARE | 20 | 0.14% |
| discoverer or inventor | 20 | 0.14% |
| member_state_of | 20 | 0.14% |
| operator | 20 | 0.14% |
| mouth of the watercourse | 20 | 0.14% |
| regulated_by | 19 | 0.13% |
| secretary general | 19 | 0.13% |
| narrative location | 19 | 0.13% |
| production company | 19 | 0.13% |
| military branch | 19 | 0.13% |
| facet of | 19 | 0.13% |
| has effect | 19 | 0.13% |
| developer | 19 | 0.13% |
| voice type | 18 | 0.12% |
| native language | 18 | 0.12% |
| organizer | 18 | 0.12% |
| residence | 18 | 0.12% |
| manufacturer | 18 | 0.12% |
| participant of | 18 | 0.12% |
| language of work or name | 18 | 0.12% |
| item operated | 18 | 0.12% |
| capital | 18 | 0.12% |
| religion or worldview | 18 | 0.12% |
| surface played on | 18 | 0.12% |
| location of formation | 18 | 0.12% |
| located in the present-day administrative territorial entity | 18 | 0.12% |
| relative | 18 | 0.12% |
| located in protected area | 17 | 0.12% |
| located on astronomical body | 17 | 0.12% |
| partner in business or sport | 17 | 0.12% |
| stock exchange | 17 | 0.12% |
| advisor_to | 17 | 0.12% |
| stock market index | 17 | 0.12% |
| instrument | 16 | 0.11% |
| grants | 16 | 0.11% |
| drug or therapy used for treatment | 16 | 0.11% |
| basin country | 16 | 0.11% |
| military rank | 16 | 0.11% |
| form of creative work | 16 | 0.11% |
| astronaut mission | 16 | 0.11% |
| interacts_with | 16 | 0.11% |
| heritage designation | 16 | 0.11% |
| school district | 16 | 0.11% |
| owner of | 16 | 0.11% |
| place of detention | 16 | 0.11% |
| has works in the collection | 16 | 0.11% |
| officeholder | 15 | 0.10% |
| terminus location | 15 | 0.10% |
| movement | 15 | 0.10% |
| league system | 15 | 0.10% |
| head coach | 15 | 0.10% |
| brand | 15 | 0.10% |
| position played on team / speciality | 15 | 0.10% |
| carries | 15 | 0.10% |
| has part(s) of the class | 15 | 0.10% |
| online service | 15 | 0.10% |
| distributed by | 15 | 0.10% |
| located in or next to body of water | 15 | 0.10% |
| follows | 15 | 0.10% |
| member of the crew of | 15 | 0.10% |
| crosses | 14 | 0.10% |
| has subsidiary | 14 | 0.10% |
| sex or gender | 14 | 0.10% |
| ammunition | 14 | 0.10% |
| biological process | 14 | 0.10% |
| historic county | 14 | 0.10% |
| head of state | 14 | 0.10% |
| student of | 14 | 0.10% |
| has immediate cause | 14 | 0.10% |
| FEATURE-OF | 14 | 0.10% |
| place of publication | 14 | 0.10% |
| commemorates | 14 | 0.10% |
| winner | 14 | 0.10% |
| located on terrain feature | 13 | 0.09% |
| published in | 13 | 0.09% |
| located on street | 13 | 0.09% |
| diocese | 13 | 0.09% |
| lyrics by | 13 | 0.09% |
| record label | 13 | 0.09% |
| place served by transport hub | 13 | 0.09% |
| has written for | 13 | 0.09% |
| medical condition treated | 13 | 0.09% |
| party_to | 13 | 0.09% |
| interested in | 13 | 0.09% |
| customer_of | 13 | 0.09% |
| parent taxon | 13 | 0.09% |
| found in taxon | 13 | 0.09% |
| authored_by | 13 | 0.09% |
| place of burial | 13 | 0.09% |
| anatomical location | 12 | 0.08% |
| cause of death | 12 | 0.08% |
| legislated by | 12 | 0.08% |
| influenced by | 12 | 0.08% |
| capital of | 12 | 0.08% |
| dialect of | 12 | 0.08% |
| illustrator | 12 | 0.08% |
| commanded by | 12 | 0.08% |
| opposite of | 12 | 0.08% |
| genic_interaction | 12 | 0.08% |
| publisher | 12 | 0.08% |
| opposed_to | 12 | 0.08% |
| vehicle | 12 | 0.08% |
| model item | 12 | 0.08% |
| licensed to broadcast to | 12 | 0.08% |
| time period | 12 | 0.08% |
| allegiance | 12 | 0.08% |
| producer | 12 | 0.08% |
| takes place in fictional universe | 12 | 0.08% |
| dedicated to | 12 | 0.08% |
| has quality | 12 | 0.08% |
| worshipped by | 11 | 0.08% |
| unmarried partner | 11 | 0.08% |
| physically interacts with | 11 | 0.08% |
| constellation | 11 | 0.08% |
| location of first performance | 11 | 0.08% |
| born_in | 11 | 0.08% |
| composed_by | 11 | 0.08% |
| parent club | 11 | 0.08% |
| historical region | 11 | 0.08% |
| operating area | 11 | 0.08% |
| editor | 11 | 0.08% |
| candidate | 11 | 0.08% |
| presenter | 11 | 0.08% |
| student | 11 | 0.08% |
| convicted of | 11 | 0.08% |
| space launch vehicle | 11 | 0.08% |
| objective of project or action | 11 | 0.08% |
| collection | 11 | 0.08% |
| this taxon is source of | 11 | 0.08% |
| operating system | 11 | 0.08% |
| league level above | 11 | 0.08% |
| parent astronomical body | 11 | 0.08% |
| significant place | 11 | 0.08% |
| Bind | 11 | 0.08% |
| crew member(s) | 11 | 0.08% |
| part of the series | 11 | 0.08% |
| musical conductor | 10 | 0.07% |
| culture | 10 | 0.07% |
| medical condition | 10 | 0.07% |
| pole position | 10 | 0.07% |
| indigenous to | 10 | 0.07% |
| genetic association | 10 | 0.07% |
| tracklist | 10 | 0.07% |
| cathedral | 10 | 0.07% |
| recommended unit of measurement | 10 | 0.07% |
| is the study of | 10 | 0.07% |
| industry | 10 | 0.07% |
| airline alliance | 10 | 0.07% |
| location of creation | 10 | 0.07% |
| designed by | 10 | 0.07% |
| commander of (DEPRECATED) | 10 | 0.07% |
| inflows | 10 | 0.07% |
| start point | 10 | 0.07% |
| terminus | 10 | 0.07% |
| sports discipline competed in | 10 | 0.07% |
| location of discovery | 10 | 0.07% |
| legal form | 10 | 0.07% |
| radio format | 10 | 0.07% |
| tributary | 10 | 0.07% |
| architectural style | 9 | 0.06% |
| candidacy in election | 9 | 0.06% |
| territory overlaps | 9 | 0.06% |
| director / manager | 9 | 0.06% |
| copyright holder | 9 | 0.06% |
| track gauge | 9 | 0.06% |
| ancestral home | 9 | 0.06% |
| drainage basin | 9 | 0.06% |
| twinned administrative body | 9 | 0.06% |
| fictional universe described in | 9 | 0.06% |
| voice actor | 9 | 0.06% |
| reservoir created | 9 | 0.06% |
| transport network | 9 | 0.06% |
| qualifies for event | 9 | 0.06% |
| edition or translation of | 9 | 0.06% |
| manner of death | 9 | 0.06% |
| taxon rank | 9 | 0.06% |
| distribution format | 9 | 0.06% |
| located in time zone | 9 | 0.06% |
| canonization status | 9 | 0.06% |
| original language of film or TV show | 9 | 0.06% |
| merged into | 9 | 0.06% |
| victory | 9 | 0.06% |
| target | 9 | 0.06% |
| package management system | 9 | 0.06% |
| distributor_of | 8 | 0.06% |
| programming paradigm | 8 | 0.06% |
| killed by | 8 | 0.06% |
| public holiday | 8 | 0.06% |
| librettist | 8 | 0.06% |
| produced by | 8 | 0.06% |
| connecting service | 8 | 0.06% |
| chief executive officer | 8 | 0.06% |
| currency | 8 | 0.06% |
| filming location | 8 | 0.06% |
| from narrative universe | 8 | 0.06% |
| origin of the watercourse | 8 | 0.06% |
| located in the statistical territorial entity | 8 | 0.06% |
| house publication | 8 | 0.06% |
| contributed to creative work | 8 | 0.06% |
| incarnation of | 8 | 0.06% |
| subject has role | 8 | 0.06% |
| first appearance | 8 | 0.06% |
| competition class | 8 | 0.06% |
| basic form of government | 8 | 0.06% |
| highest judicial authority | 8 | 0.06% |
| has cause | 8 | 0.06% |
| game mechanics | 8 | 0.06% |
| participating team | 8 | 0.06% |
| allied_with | 8 | 0.06% |
| standards body | 8 | 0.06% |
| writing system | 7 | 0.05% |
| this zoological name is coordinate with | 7 | 0.05% |
| has edition or translation | 7 | 0.05% |
| sponsor | 7 | 0.05% |
| cause | 7 | 0.05% |
| official language | 7 | 0.05% |
| after a work by | 7 | 0.05% |
| vessel class | 7 | 0.05% |
| consecrator | 7 | 0.05% |
| sponsored_by | 7 | 0.05% |
| enemy of | 7 | 0.05% |
| engine configuration | 7 | 0.05% |
| endemic to | 7 | 0.05% |
| input method | 7 | 0.05% |
| copyright license | 7 | 0.05% |
| represented by | 7 | 0.05% |
| legislative body | 7 | 0.05% |
| adjacent station | 7 | 0.05% |
| anatomical branch of | 7 | 0.05% |
| domain of saint or deity | 7 | 0.05% |
| affiliation | 7 | 0.05% |
| honorific prefix | 7 | 0.05% |
| dam | 7 | 0.05% |
| taxonomic type | 7 | 0.05% |
| choreographer | 6 | 0.04% |
| ethnic group | 6 | 0.04% |
| general classification of race participants | 6 | 0.04% |
| anthem | 6 | 0.04% |
| structural engineer | 6 | 0.04% |
| business division | 6 | 0.04% |
| derivative work | 6 | 0.04% |
| programming language | 6 | 0.04% |
| co-driver | 6 | 0.04% |
| foundational text | 6 | 0.04% |
| religious order | 6 | 0.04% |
| Gram staining | 6 | 0.04% |
| coextensive with | 6 | 0.04% |
| color | 6 | 0.04% |
| monomer of | 6 | 0.04% |
| official religion | 6 | 0.04% |
| proved by | 6 | 0.04% |
| defendant | 6 | 0.04% |
| immediate cause of | 6 | 0.04% |
| director of photography | 6 | 0.04% |
| next higher rank | 6 | 0.04% |
| office held by head of government | 6 | 0.04% |
| newspaper format | 6 | 0.04% |
| cell component | 6 | 0.04% |
| maintained by | 6 | 0.04% |
| adjacent_to | 6 | 0.04% |
| airline hub | 6 | 0.04% |
| history of topic | 6 | 0.04% |
| sued_by | 5 | 0.03% |
| country of registry | 5 | 0.03% |
| set in period | 5 | 0.03% |
| measured physical quantity | 5 | 0.03% |
| court | 5 | 0.03% |
| wheel arrangement | 5 | 0.03% |
| used by | 5 | 0.03% |
| depicts | 5 | 0.03% |
| source of energy | 5 | 0.03% |
| central bank/issuer | 5 | 0.03% |
| invested_in | 5 | 0.03% |
| type of variable star | 5 | 0.03% |
| risk factor | 5 | 0.03% |
| chromosome | 5 | 0.03% |
| media franchise | 5 | 0.03% |
| destination point | 5 | 0.03% |
| political alignment | 5 | 0.03% |
| less than | 5 | 0.03% |
| IUCN conservation status | 5 | 0.03% |
| active ingredient in | 5 | 0.03% |
| main regulatory text | 5 | 0.03% |
| associated electoral district | 5 | 0.03% |
| interaction | 5 | 0.03% |
| mascot | 5 | 0.03% |
| natural product of taxon | 5 | 0.03% |
| separated from | 5 | 0.03% |
| predecessor_of | 5 | 0.03% |
| screenwriter | 5 | 0.03% |
| board member | 5 | 0.03% |
| instrumentation | 5 | 0.03% |
| derived_from | 5 | 0.03% |
| executive body | 5 | 0.03% |
| broadcast by | 5 | 0.03% |
| armament | 5 | 0.03% |
| practiced by | 5 | 0.03% |
| coach of sports team | 5 | 0.03% |
| territory claimed by | 4 | 0.03% |
| business model | 4 | 0.03% |
| political ideology | 4 | 0.03% |
| child astronomical body | 4 | 0.03% |
| sold_to | 4 | 0.03% |
| film editor | 4 | 0.03% |
| causes | 4 | 0.03% |
| contains | 4 | 0.03% |
| studied by | 4 | 0.03% |
| signatory | 4 | 0.03% |
| academic degree | 4 | 0.03% |
| primary destinations | 4 | 0.03% |
| measures | 4 | 0.03% |
| lake on watercourse | 4 | 0.03% |
| powered by | 4 | 0.03% |
| series spin-off | 4 | 0.03% |
| named_after | 4 | 0.03% |
| animal breed | 4 | 0.03% |
| won | 4 | 0.03% |
| capital_of | 4 | 0.03% |
| sports season of league or competition | 4 | 0.03% |
| designed to carry | 4 | 0.03% |
| speaker | 4 | 0.03% |
| substitute/deputy/replacement of office/officeholder | 4 | 0.03% |
| GUI toolkit or framework | 4 | 0.03% |
| league level below | 4 | 0.03% |
| contains settlement | 4 | 0.03% |
| title of chess person | 4 | 0.03% |
| official residence | 4 | 0.03% |
| field of this occupation | 4 | 0.03% |
| offers view on | 4 | 0.03% |
| described by source | 4 | 0.03% |
| contributor to the creative work or subject | 4 | 0.03% |
| charged_with | 4 | 0.03% |
| depicted by | 4 | 0.03% |
| health specialty | 4 | 0.03% |
| symptoms and signs | 4 | 0.03% |
| discography | 4 | 0.03% |
| implementation of | 4 | 0.03% |
| authority | 4 | 0.03% |
| commissioned by | 4 | 0.03% |
| linguistic typology | 4 | 0.03% |
| subdivision of this unit | 4 | 0.03% |
| encodes | 4 | 0.03% |
| borders | 4 | 0.03% |
| youth wing | 4 | 0.03% |
| Comparison | 4 | 0.03% |
| discovery method | 3 | 0.02% |
| fastest lap | 3 | 0.02% |
| has contributing factor | 3 | 0.02% |
| treats | 3 | 0.02% |
| game mode | 3 | 0.02% |
| has active ingredient | 3 | 0.02% |
| location of landing | 3 | 0.02% |
| possible treatment | 3 | 0.02% |
| overlies | 3 | 0.02% |
| statistical leader | 3 | 0.02% |
| minor planet group | 3 | 0.02% |
| vehicle normally used | 3 | 0.02% |
| writable file format | 3 | 0.02% |
| has boundary | 3 | 0.02% |
| temporal range start | 3 | 0.02% |
| office held by head of state | 3 | 0.02% |
| molecular function | 3 | 0.02% |
| addressee | 3 | 0.02% |
| organization directed by the office or position | 3 | 0.02% |
| update method | 3 | 0.02% |
| perpetrator | 3 | 0.02% |
| flag | 3 | 0.02% |
| cuisine | 3 | 0.02% |
| has fruit type | 3 | 0.02% |
| typically sells | 3 | 0.02% |
| highest point | 3 | 0.02% |
| size designation | 3 | 0.02% |
| day in year for periodic occurrence | 3 | 0.02% |
| party chief representative | 3 | 0.02% |
| afflicts | 3 | 0.02% |
| sexually homologous with | 3 | 0.02% |
| measurement scale | 3 | 0.02% |
| antiparticle | 3 | 0.02% |
| has goal | 3 | 0.02% |
| parliamentary group | 3 | 0.02% |
| charge | 3 | 0.02% |
| mentor_of | 3 | 0.02% |
| dual to | 3 | 0.02% |
| penalty | 3 | 0.02% |
| has anatomical branch | 3 | 0.02% |
| enclave within | 3 | 0.02% |
| intended public | 3 | 0.02% |
| next crossing upstream | 2 | 0.01% |
| economy of topic | 2 | 0.01% |
| executive producer | 2 | 0.01% |
| designated as terrorist by | 2 | 0.01% |
| diaspora | 2 | 0.01% |
| adjacent building | 2 | 0.01% |
| species kept | 2 | 0.01% |
| exclave of | 2 | 0.01% |
| patron saint | 2 | 0.01% |
| taxon synonym | 2 | 0.01% |
| points/goal scored by | 2 | 0.01% |
| set in environment | 2 | 0.01% |
| host | 2 | 0.01% |
| inspired by | 2 | 0.01% |
| prerequisite | 2 | 0.01% |
| foods traditionally associated | 2 | 0.01% |
| convicted_of | 2 | 0.01% |
| delta of | 2 | 0.01% |
| editor-in-chief | 2 | 0.01% |
| iconographic symbol | 2 | 0.01% |
| readable file format | 2 | 0.01% |
| structure replaced by | 2 | 0.01% |
| product certification | 2 | 0.01% |
| depends on software | 2 | 0.01% |
| central bank | 2 | 0.01% |
| cause of destruction | 2 | 0.01% |
| sexual orientation | 2 | 0.01% |
| has census | 2 | 0.01% |
| appears in the form of | 2 | 0.01% |
| regulated by | 2 | 0.01% |
| office held by head of the organization | 2 | 0.01% |
| produced sound | 2 | 0.01% |
| lake outflow | 2 | 0.01% |
| amended by | 2 | 0.01% |
| sidekick of | 2 | 0.01% |
| successor_of | 2 | 0.01% |
| interchange station | 2 | 0.01% |
| superhuman feature or ability | 2 | 0.01% |
| points classification | 2 | 0.01% |
| CPU | 2 | 0.01% |
| character type | 2 | 0.01% |
| fabrication method | 2 | 0.01% |
| next lower rank | 2 | 0.01% |
| recorded at studio or venue | 2 | 0.01% |
| drafted by | 2 | 0.01% |
| director of archaeological fieldwork | 2 | 0.01% |
| software engine | 2 | 0.01% |
| wears | 2 | 0.01% |
| language regulatory body | 2 | 0.01% |
| source of material | 2 | 0.01% |
| motto | 2 | 0.01% |
| academic thesis | 2 | 0.01% |
| vaccine for | 2 | 0.01% |
| type of orbit | 2 | 0.01% |
| top-level Internet domain | 2 | 0.01% |
| set during recurring event | 2 | 0.01% |
| mother house | 2 | 0.01% |
| suggests the existence of | 2 | 0.01% |
| madhhab | 2 | 0.01% |
| narrative role | 2 | 0.01% |
| home port | 2 | 0.01% |
| signatory_of | 2 | 0.01% |
| type foundry | 2 | 0.01% |
| advertises | 2 | 0.01% |
| manifestation of | 2 | 0.01% |
| successful candidate | 2 | 0.01% |
| emulates | 2 | 0.01% |
| copyright representative | 2 | 0.01% |
| final event | 2 | 0.01% |
| Drug_Interaction | 2 | 0.01% |
| statement describes | 2 | 0.01% |
| mod of | 2 | 0.01% |
| archives at | 2 | 0.01% |
| scheduled service destination | 2 | 0.01% |
| has pet | 2 | 0.01% |
| Christian liturgical rite | 2 | 0.01% |
| order of battle | 1 | 0.01% |
| season | 1 | 0.01% |
| emergency phone number | 1 | 0.01% |
| coolant | 1 | 0.01% |
| demographics of topic | 1 | 0.01% |
| captain | 1 | 0.01% |
| theorised by | 1 | 0.01% |
| innervates | 1 | 0.01% |
| shape | 1 | 0.01% |
| connector | 1 | 0.01% |
| investor | 1 | 0.01% |
| solved by | 1 | 0.01% |
| carries scientific instrument | 1 | 0.01% |
| wing configuration | 1 | 0.01% |
| has melody | 1 | 0.01% |
| official symbol | 1 | 0.01% |
| chief operating officer | 1 | 0.01% |
| tonality | 1 | 0.01% |
| epithet | 1 | 0.01% |
| by-product of | 1 | 0.01% |
| religious occupation | 1 | 0.01% |
| victim | 1 | 0.01% |
| structure replaces | 1 | 0.01% |
| website account on | 1 | 0.01% |
| microstructure | 1 | 0.01% |
| arterial supply | 1 | 0.01% |
| merged_with | 1 | 0.01% |
| sports league level | 1 | 0.01% |
| relative to | 1 | 0.01% |
| kit supplier | 1 | 0.01% |
| lowest point | 1 | 0.01% |
| family name identical to this given name | 1 | 0.01% |
| prime factor | 1 | 0.01% |
| intangible cultural heritage status | 1 | 0.01% |
| supports programming language | 1 | 0.01% |
| conjugate acid | 1 | 0.01% |
| style of karate | 1 | 0.01% |
| is metaclass for | 1 | 0.01% |
| election called by | 1 | 0.01% |
| translator | 1 | 0.01% |
| microarchitecture | 1 | 0.01% |
| criterion used | 1 | 0.01% |
| has facet polytope | 1 | 0.01% |
| given name version for other gender | 1 | 0.01% |
| approved by | 1 | 0.01% |
| venous drainage | 1 | 0.01% |
| character designer | 1 | 0.01% |
| catalog | 1 | 0.01% |
| volcano observatory | 1 | 0.01% |
| destroyed | 1 | 0.01% |
| reward program | 1 | 0.01% |
| electoral district | 1 | 0.01% |
| season of club or team | 1 | 0.01% |
| regulates (molecular biology) | 1 | 0.01% |
| contributing factor of | 1 | 0.01% |
| licensed_to | 1 | 0.01% |
| testified_against | 1 | 0.01% |
| colorist | 1 | 0.01% |
| end cause | 1 | 0.01% |
| uses capitalization for | 1 | 0.01% |
| parent peak | 1 | 0.01% |
| political coalition | 1 | 0.01% |
| narrative motif | 1 | 0.01% |
| parent of this hybrid, breed, or cultivar | 1 | 0.01% |
| programmer | 1 | 0.01% |
| dialect of computer language | 1 | 0.01% |
| by-product | 1 | 0.01% |
| crystal system | 1 | 0.01% |
| has facility | 1 | 0.01% |
| expansion of | 1 | 0.01% |
| production designer | 1 | 0.01% |
| peak bagging classification | 1 | 0.01% |
| donated by | 1 | 0.01% |
| animator | 1 | 0.01% |
| copyright status | 1 | 0.01% |
| feast day | 1 | 0.01% |
| supported sports team | 1 | 0.01% |
| blocker of | 1 | 0.01% |
| greater than | 1 | 0.01% |
| instruction set | 1 | 0.01% |
| named by | 1 | 0.01% |
| day of week | 1 | 0.01% |
| doctoral advisor | 1 | 0.01% |
| Cotreatment | 1 | 0.01% |
| appointed by | 1 | 0.01% |
| died_in | 1 | 0.01% |
| honorific suffix | 1 | 0.01% |
| has seal, badge, or sigil | 1 | 0.01% |
| has phenotype | 1 | 0.01% |
| OSI Model layer location | 1 | 0.01% |
| prohibits | 1 | 0.01% |
| complies with | 1 | 0.01% |
| stepparent | 1 | 0.01% |
| launch contractor | 1 | 0.01% |
| possible medical findings | 1 | 0.01% |
| soundtrack release | 1 | 0.01% |
| geography of topic | 1 | 0.01% |
| has list | 1 | 0.01% |
| has natural reservoir | 1 | 0.01% |
| exemplar of | 1 | 0.01% |
| asteroid spectral type | 1 | 0.01% |
| supersedes | 1 | 0.01% |
| social classification | 1 | 0.01% |
| performed at | 1 | 0.01% |
| plot expanded in | 1 | 0.01% |
| home world | 1 | 0.01% |
| biological variant of | 1 | 0.01% |
| music created for | 1 | 0.01% |
| significant drug interaction | 1 | 0.01% |
| expressed in | 1 | 0.01% |
| typing discipline | 1 | 0.01% |
| mountains classification | 1 | 0.01% |
| recording location | 1 | 0.01% |
| worn by | 1 | 0.01% |
| underlies | 1 | 0.01% |
| laws applied | 1 | 0.01% |

### Classifications

**train, actionability.** Majority class 87.6% of 726 answers. The marginal is `true_label`, not the menu.

#### train actionability

| label | count | share |
| --- | --- | --- |
| informational | 636 | 87.60% |
| advisory | 47 | 6.47% |
| requires action | 43 | 5.92% |

**train, audience.** Majority class 77.9% of 890 answers. The marginal is `true_label`, not the menu.

#### train audience

| label | count | share |
| --- | --- | --- |
| general public | 693 | 77.87% |
| expert | 107 | 12.02% |
| regulatory | 69 | 7.75% |
| internal | 21 | 2.36% |

**train, biotech_event.** Majority class 27.3% of 516 answers. The marginal is `true_label`, not the menu.

#### train biotech_event

| label | count | share |
| --- | --- | --- |
| executive statement | 141 | 27.33% |
| company description | 85 | 16.47% |
| alliance & partnership | 40 | 7.75% |
| other | 32 | 6.20% |
| product launching & presentation | 31 | 6.01% |
| new initiatives or programs | 26 | 5.04% |
| support & philanthropy | 21 | 4.07% |
| m&a | 20 | 3.88% |
| funding round | 15 | 2.91% |
| executive appointment | 14 | 2.71% |
| expanding geography | 14 | 2.71% |
| article publication | 11 | 2.13% |
| service & product providing | 10 | 1.94% |
| event organization | 10 | 1.94% |
| participation in an event | 9 | 1.74% |
| hiring | 8 | 1.55% |
| new initiatives & programs | 7 | 1.36% |
| investment in public company | 4 | 0.78% |
| product updates | 4 | 0.78% |
| expanding industry | 4 | 0.78% |
| regulatory approval | 3 | 0.58% |
| ipo exit | 2 | 0.39% |
| closing | 2 | 0.39% |
| clinical trial sponsorship | 1 | 0.19% |
| foundation | 1 | 0.19% |
| patent publication | 1 | 0.19% |

**train, certainty.** Majority class 96.4% of 804 answers. The marginal is `true_label`, not the menu.

#### train certainty

| label | count | share |
| --- | --- | --- |
| asserted | 775 | 96.39% |
| hedged | 28 | 3.48% |
| speculative | 1 | 0.12% |

**train, chfinann_event.** Majority class 48.5% of 3,221 answers. The marginal is `true_label`, not the menu.

#### train chfinann_event

| label | count | share |
| --- | --- | --- |
| EquityPledge | 1,562 | 48.49% |
| EquityUnderweight | 679 | 21.08% |
| EquityOverweight | 639 | 19.84% |
| EquityRepurchase | 246 | 7.64% |
| EquityFreeze | 95 | 2.95% |

**train, docee_event.** Majority class 6.8% of 9,061 answers. The marginal is `true_label`, not the menu.

#### train docee_event

| label | count | share |
| --- | --- | --- |
| Sports Competition | 620 | 6.84% |
| none | 521 | 5.75% |
| Government Policy Changes | 435 | 4.80% |
| Organization Fine | 384 | 4.24% |
| Fire | 382 | 4.22% |
| CommitCrime - Investigate | 379 | 4.18% |
| CommitCrime - Arrest | 358 | 3.95% |
| Armed Conflict | 323 | 3.56% |
| Protest | 247 | 2.73% |
| Air Crash | 241 | 2.66% |
| Famous Person - Death | 241 | 2.66% |
| Appoint_Inauguration | 238 | 2.63% |
| Resignation_Dismissal | 231 | 2.55% |
| Earthquakes | 217 | 2.39% |
| Road Crash | 207 | 2.28% |
| New Achievements in Aerospace | 206 | 2.27% |
| Election | 204 | 2.25% |
| Diplomatic Talks | 184 | 2.03% |
| Riot | 181 | 2.00% |
| CommitCrime - Accuse | 159 | 1.75% |
| Disease Outbreaks | 159 | 1.75% |
| Famous Person - Give a Speech | 153 | 1.69% |
| CommitCrime - Sentence | 152 | 1.68% |
| Break Historical Records | 140 | 1.55% |
| Sign Agreement | 138 | 1.52% |
| Diplomatic Visit | 137 | 1.51% |
| Awards Ceremony | 132 | 1.46% |
| Storm | 131 | 1.45% |
| Mass Poisoning | 121 | 1.34% |
| Military Excercise | 118 | 1.30% |
| Floods | 114 | 1.26% |
| Environment Pollution | 92 | 1.02% |
| Strike | 81 | 0.89% |
| Financial Crisis | 80 | 0.88% |
| New Wonders in Nature | 77 | 0.85% |
| Shipwreck | 72 | 0.79% |
| Organization Established | 70 | 0.77% |
| CommitCrime - Release | 69 | 0.76% |
| New Archeological Discoveries | 68 | 0.75% |
| Financial Aid | 64 | 0.71% |
| Withdraw from an Orgnization | 63 | 0.70% |
| Mudslides | 63 | 0.70% |
| Gas Explosion | 62 | 0.68% |
| Bank Robbery | 61 | 0.67% |
| Organization Closed | 60 | 0.66% |
| Volcano Eruption | 60 | 0.66% |
| Droughts | 55 | 0.61% |
| Famous Person - Recovered | 53 | 0.58% |
| Train Collisions | 50 | 0.55% |
| Mine Collapses | 48 | 0.53% |
| Famous Person - Divorce | 47 | 0.52% |
| Famous Person - Sick | 45 | 0.50% |
| Regime Change | 38 | 0.42% |
| Join in an Orgnization | 37 | 0.41% |
| Organization Merge | 36 | 0.40% |
| Famine | 34 | 0.38% |
| Famous Person - Marriage | 33 | 0.36% |
| Tsunamis | 32 | 0.35% |
| Tear Up Agreement | 29 | 0.32% |
| Insect Disaster | 29 | 0.32% |

**train, docfee_event.** Majority class 35.8% of 2,077 answers. The marginal is `true_label`, not the menu.

#### train docfee_event

| label | count | share |
| --- | --- | --- |
| EquityUnderweight | 744 | 35.82% |
| BankruptcyLiquidation | 340 | 16.37% |
| EquityPledge | 290 | 13.96% |
| MajorAssetLoss | 277 | 13.34% |
| EquityFreeze | 131 | 6.31% |
| EquityOverweight | 122 | 5.87% |
| ExecutiveDeath | 59 | 2.84% |
| MajorSafetyAccident | 58 | 2.79% |
| MajorExternalPayout | 56 | 2.70% |

**train, document_genre.** Majority class 73.5% of 733 answers. The marginal is `true_label`, not the menu.

#### train document_genre

| label | count | share |
| --- | --- | --- |
| news report | 539 | 73.53% |
| announcement | 93 | 12.69% |
| opinion | 38 | 5.18% |
| technical documentation | 17 | 2.32% |
| correspondence | 17 | 2.32% |
| review | 12 | 1.64% |
| legal filing | 9 | 1.23% |
| advertisement | 8 | 1.09% |

**train, formality.** Majority class 80.9% of 901 answers. The marginal is `true_label`, not the menu.

#### train formality

| label | count | share |
| --- | --- | --- |
| formal | 729 | 80.91% |
| informal | 172 | 19.09% |

**train, language_register.** Majority class 77.9% of 769 answers. The marginal is `true_label`, not the menu.

#### train language_register

| label | count | share |
| --- | --- | --- |
| plain | 599 | 77.89% |
| technical | 125 | 16.25% |
| colloquial | 27 | 3.51% |
| legal | 18 | 2.34% |

**train, risk_level.** Majority class 48.7% of 673 answers. The marginal is `true_label`, not the menu.

#### train risk_level

| label | count | share |
| --- | --- | --- |
| none | 328 | 48.74% |
| low | 126 | 18.72% |
| high | 116 | 17.24% |
| moderate | 103 | 15.30% |

**train, sentiment.** Majority class 40.1% of 890 answers. The marginal is `true_label`, not the menu.

#### train sentiment

| label | count | share |
| --- | --- | --- |
| neutral | 357 | 40.11% |
| positive | 317 | 35.62% |
| negative | 216 | 24.27% |

**train, subjectivity.** Majority class 85.7% of 704 answers. The marginal is `true_label`, not the menu.

#### train subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 603 | 85.65% |
| subjective | 101 | 14.35% |

**train, temporal_orientation.** Majority class 70.3% of 680 answers. The marginal is `true_label`, not the menu.

#### train temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 478 | 70.29% |
| present | 138 | 20.29% |
| future | 53 | 7.79% |
| timeless | 11 | 1.62% |

**train, topic.** Majority class 18.8% of 1,650 answers. The marginal is `true_label`, not the menu.

#### train topic

| label | count | share |
| --- | --- | --- |
| business | 311 | 18.85% |
| politics | 200 | 12.12% |
| legal | 180 | 10.91% |
| sports | 150 | 9.09% |
| world | 142 | 8.61% |
| finance | 124 | 7.52% |
| technology | 117 | 7.09% |
| entertainment | 112 | 6.79% |
| health | 80 | 4.85% |
| science | 64 | 3.88% |
| environment | 50 | 3.03% |
| education | 46 | 2.79% |
| defense | 27 | 1.64% |
| travel | 25 | 1.52% |
| energy | 14 | 0.85% |
| agriculture | 8 | 0.48% |

**train, urgency.** Majority class 77.3% of 700 answers. The marginal is `true_label`, not the menu.

#### train urgency

| label | count | share |
| --- | --- | --- |
| routine | 541 | 77.29% |
| elevated | 111 | 15.86% |
| critical | 39 | 5.57% |
| urgent | 9 | 1.29% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | condition | 6 | 100.00% |
| clinical_finding | patient_group | 5 | 83.33% |
| clinical_finding | treatment | 5 | 83.33% |
| clinical_finding | outcome | 5 | 83.33% |
| clinical_finding | record_metadata | natural/condition ×6 |  |
| company_profile | name | 54 | 100.00% |
| company_profile | size | 50 | 92.59% |
| company_profile | headquarters | 43 | 79.63% |
| company_profile | industry | 32 | 59.26% |
| company_profile | founded | 14 | 25.93% |
| company_profile | record_metadata | natural/name ×52 |  |
| event_listing | date | 41 | 93.18% |
| event_listing | venue | 35 | 79.55% |
| event_listing | name | 34 | 77.27% |
| event_listing | organizer | 31 | 70.45% |
| event_listing | price | 11 | 25.00% |
| event_listing | record_metadata | natural/name ×27, natural/venue ×6 |  |
| financial_report | organization | 14 | 100.00% |
| financial_report | period | 13 | 92.86% |
| financial_report | revenue | 12 | 85.71% |
| financial_report | net_income | 9 | 64.29% |
| financial_report | record_metadata | natural/organization ×14 |  |
| incident_report | severity | 53 | 98.15% |
| incident_report | date | 52 | 96.30% |
| incident_report | location | 48 | 88.89% |
| incident_report | incident_type | 39 | 72.22% |
| incident_report | casualties | 29 | 53.70% |
| incident_report | record_metadata | natural/location ×32, natural/incident_type ×19, natural/date ×2 |  |
| job_posting | title | 9 | 100.00% |
| job_posting | company | 9 | 100.00% |
| job_posting | employment_type | 9 | 100.00% |
| job_posting | location | 8 | 88.89% |
| job_posting | record_metadata | natural/location ×3, natural/title ×1 |  |
| legal_case | status | 23 | 100.00% |
| legal_case | defendant | 22 | 95.65% |
| legal_case | court | 18 | 78.26% |
| legal_case | case_name | 14 | 60.87% |
| legal_case | plaintiff | 14 | 60.87% |
| legal_case | record_metadata | natural/case_name ×14, natural/court ×9 |  |
| person_profile | name | 313 | 99.68% |
| person_profile | role | 296 | 94.27% |
| person_profile | employer | 273 | 86.94% |
| person_profile | location | 271 | 86.31% |
| person_profile | record_metadata | natural/location ×180, natural/name ×66 |  |
| product | name | 63 | 98.44% |
| product | condition | 62 | 96.88% |
| product | brand | 54 | 84.38% |
| product | price | 53 | 82.81% |
| product | record_metadata | natural/name ×55, natural/brand ×1 |  |
| real_estate_listing | address | 2 | 100.00% |
| real_estate_listing | property_type | 2 | 100.00% |
| real_estate_listing | area | 1 | 50.00% |
| real_estate_listing | record_metadata | natural/address ×2 |  |
| record | status | 123 | 12.76% |
| record | name | 71 | 7.37% |
| record | location | 68 | 7.05% |
| record | title | 43 | 4.46% |
| record | address | 42 | 4.36% |
| record | description | 36 | 3.73% |
| record | price | 26 | 2.70% |
| record | rating | 23 | 2.39% |
| record | amount | 23 | 2.39% |
| record | date | 22 | 2.28% |
| record | patient_name | 19 | 1.97% |
| record | patient_id | 19 | 1.97% |
| record | city | 17 | 1.76% |
| record | employee_id | 17 | 1.76% |
| record | reviewer | 16 | 1.66% |
| record | department | 16 | 1.66% |
| record | state | 16 | 1.66% |
| record | venue | 15 | 1.56% |
| record | company | 14 | 1.45% |
| record | transaction_id | 13 | 1.35% |
| record | source | 12 | 1.24% |
| record | product_id | 12 | 1.24% |
| record | manufacturer | 12 | 1.24% |
| record | author | 11 | 1.14% |
| record | person | 11 | 1.14% |
| record | country | 11 | 1.14% |
| record | age | 10 | 1.04% |
| record | review_id | 10 | 1.04% |
| record | claim_id | 10 | 1.04% |
| record | time | 10 | 1.04% |
| record | sku | 9 | 0.93% |
| record | category | 9 | 0.93% |
| record | year | 9 | 0.93% |
| record | away_team | 9 | 0.93% |
| record | score | 9 | 0.93% |
| record | route | 9 | 0.93% |
| record | home_team | 8 | 0.83% |
| record | ticket_id | 8 | 0.83% |
| record | mrn | 7 | 0.73% |
| record | brand | 7 | 0.73% |
| record | color | 7 | 0.73% |
| record | dimensions | 7 | 0.73% |
| record | organizations | 7 | 0.73% |
| record | asteroid.name | 7 | 0.73% |
| record | value | 7 | 0.73% |
| record | event_id | 7 | 0.73% |
| record | product_name | 7 | 0.73% |
| record | origin | 6 | 0.62% |
| record | appointment_id | 6 | 0.62% |
| record | duration | 6 | 0.62% |
| record | id | 6 | 0.62% |
| record | recipient | 6 | 0.62% |
| record | device | 6 | 0.62% |
| record | acquirer | 6 | 0.62% |
| record | dosage | 6 | 0.62% |
| record | name: Full name of the patient | 6 | 0.62% |
| record | designation | 6 | 0.62% |
| record | normal_range | 6 | 0.62% |
| record | findings | 6 | 0.62% |
| record | target_audience | 6 | 0.62% |
| record | event | 6 | 0.62% |
| record | project_id | 6 | 0.62% |
| record | symptoms | 5 | 0.52% |
| record | player_name | 5 | 0.52% |
| record | keywords | 5 | 0.52% |
| record | order | 5 | 0.52% |
| record | temperature | 5 | 0.52% |
| record | tail | 5 | 0.52% |
| record | head | 5 | 0.52% |
| record | locations | 5 | 0.52% |
| record | review_date | 5 | 0.52% |
| record | discovery.observatory.name | 5 | 0.52% |
| record | instructor | 5 | 0.52% |
| record | teams | 5 | 0.52% |
| record | release_date | 5 | 0.52% |
| record | email | 5 | 0.52% |
| record | merchant | 5 | 0.52% |
| record | availability | 5 | 0.52% |
| record | opponent | 5 | 0.52% |
| record | medication | 5 | 0.52% |
| record | tags | 4 | 0.41% |
| record | funding_source | 4 | 0.41% |
| record | vendor_name | 4 | 0.41% |
| record | authors | 4 | 0.41% |
| record | notes | 4 | 0.41% |
| record | lengthofstay | 4 | 0.41% |
| record | position | 4 | 0.41% |
| record | user | 4 | 0.41% |
| record | sender | 4 | 0.41% |
| record | sentiment | 4 | 0.41% |
| record | manager | 4 | 0.41% |
| record | material | 4 | 0.41% |
| record | product.name | 4 | 0.41% |
| record | region | 4 | 0.41% |
| record | full_name | 4 | 0.41% |
| record | expiry_date | 4 | 0.41% |
| record | reporter | 4 | 0.41% |
| record | research.researcher.name | 4 | 0.41% |
| record | research.researcher.affiliation.name | 4 | 0.41% |
| record | research.experiment_details.outcome | 4 | 0.41% |
| record | word_count | 4 | 0.41% |
| record | publication.year | 4 | 0.41% |
| record | reviewer_name | 4 | 0.41% |
| record | physical_characteristics.observations.0.telescope | 4 | 0.41% |
| record | physical_characteristics.observations.0.year | 4 | 0.41% |
| record | orbital_characteristics.perihelion.unit | 4 | 0.41% |
| record | key_player | 4 | 0.41% |
| record | patient_age | 4 | 0.41% |
| record | prescription_id | 4 | 0.41% |
| record | gender | 4 | 0.41% |
| record | weight | 4 | 0.41% |
| record | policyholder_name | 4 | 0.41% |
| record | family | 4 | 0.41% |
| record | team | 4 | 0.41% |
| record | policy_number | 4 | 0.41% |
| record | severity | 4 | 0.41% |
| record | organization | 4 | 0.41% |
| record | training_modules | 4 | 0.41% |
| record | discovery.date | 4 | 0.41% |
| record | purpose | 4 | 0.41% |
| record | features | 4 | 0.41% |
| record | funding_round | 4 | 0.41% |
| record | frequency | 4 | 0.41% |
| record | asteroid.designation | 4 | 0.41% |
| record | description: Brief summary of the job responsibilities | 4 | 0.41% |
| record | asteroid.location | 4 | 0.41% |
| record | timestamp | 4 | 0.41% |
| record | asteroid.discovery.date | 4 | 0.41% |
| record | review_score | 4 | 0.41% |
| record | incident_id | 3 | 0.31% |
| record | channel_name | 3 | 0.31% |
| record | channel_url | 3 | 0.31% |
| record | destination | 3 | 0.31% |
| record | delay_minutes | 3 | 0.31% |
| record | room | 3 | 0.31% |
| record | funding | 3 | 0.31% |
| record | submission_date | 3 | 0.31% |
| record | match_code | 3 | 0.31% |
| record | final_score | 3 | 0.31% |
| record | issue | 3 | 0.31% |
| record | fee | 3 | 0.31% |
| record | organization.name | 3 | 0.31% |
| record | discovery_year | 3 | 0.31% |
| record | policy_id: Unique identifier for the insurance policy | 3 | 0.31% |
| record | campaign_id | 3 | 0.31% |
| record | purchase_date | 3 | 0.31% |
| record | attendees | 3 | 0.31% |
| record | product_code | 3 | 0.31% |
| record | topics | 3 | 0.31% |
| record | app_version | 3 | 0.31% |
| record | user_id | 3 | 0.31% |
| record | themes | 3 | 0.31% |
| record | film_title: Title of the film being reviewed | 3 | 0.31% |
| record | donor_name | 3 | 0.31% |
| record | candidate_name | 3 | 0.31% |
| record | patient_address | 3 | 0.31% |
| record | research.title | 3 | 0.31% |
| record | research.focus | 3 | 0.31% |
| record | research.materials | 3 | 0.31% |
| record | research.temperature | 3 | 0.31% |
| record | research.experiment_details.setup | 3 | 0.31% |
| record | research.experiment_details.quantum_state | 3 | 0.31% |
| record | research.experiment_details.speckle_details | 3 | 0.31% |
| record | dates | 3 | 0.31% |
| record | address: Full address of the policyholder | 3 | 0.31% |
| record | location: Geographic location of the tournament | 3 | 0.31% |
| record | scientific_name | 3 | 0.31% |
| record | type | 3 | 0.31% |
| record | timestamp: Date and time of transaction occurrence | 3 | 0.31% |
| record | notes: Additional context or explanation for the transaction | 3 | 0.31% |
| record | exchange_rate: Rate used to convert base to target currency | 3 | 0.31% |
| record | budget | 3 | 0.31% |
| record | product: Name of the skincare or apparel product | 3 | 0.31% |
| record | comments | 3 | 0.31% |
| record | shares | 3 | 0.31% |
| record | review_status | 3 | 0.31% |
| record | referee | 3 | 0.31% |
| record | inventor | 3 | 0.31% |
| record | status: Current state of the ticket | 3 | 0.31% |
| record | founding_year | 3 | 0.31% |
| record | patent_number | 3 | 0.31% |
| record | attending_physician | 3 | 0.31% |
| record | device_id | 3 | 0.31% |
| record | cause | 3 | 0.31% |
| record | orbital_characteristics.classification | 3 | 0.31% |
| record | discovery.discoverer | 3 | 0.31% |
| record | discovery.observatory.location.country | 3 | 0.31% |
| record | size | 3 | 0.31% |
| record | zip | 3 | 0.31% |
| record | medications | 3 | 0.31% |
| record | primary_diagnosis | 3 | 0.31% |
| record | patient_gender | 3 | 0.31% |
| record | ndc | 3 | 0.31% |
| record | test_name | 3 | 0.31% |
| record | hotel_name | 3 | 0.31% |
| record | speaker: Name of the person delivering the statement | 3 | 0.31% |
| record | amount: Financial value associated with the policy or proposal | 3 | 0.31% |
| record | shipping_method | 3 | 0.31% |
| record | refund_amount | 3 | 0.31% |
| record | customer_address | 3 | 0.31% |
| record | geographic_scope | 3 | 0.31% |
| record | policy_id | 3 | 0.31% |
| record | main_competitor | 3 | 0.31% |
| record | format | 3 | 0.31% |
| record | scientificName | 3 | 0.31% |
| record | report_id | 3 | 0.31% |
| record | customer_name: Name of the affected customer | 3 | 0.31% |
| record | billing_code | 3 | 0.31% |
| record | discharge_diagnosis | 3 | 0.31% |
| record | discharge_instructions | 3 | 0.31% |
| record | bracket_id | 3 | 0.31% |
| record | prescriber | 3 | 0.31% |
| record | entity | 3 | 0.31% |
| record | definition | 3 | 0.31% |
| record | source_agency | 3 | 0.31% |
| record | tweet_id | 3 | 0.31% |
| record | target | 3 | 0.31% |
| record | topic | 3 | 0.31% |
| record | code | 3 | 0.31% |
| record | address: Full street address of the property | 3 | 0.31% |
| record | event_type | 3 | 0.31% |
| record | people | 3 | 0.31% |
| record | establishment_year | 3 | 0.31% |
| record | claim_amount | 3 | 0.31% |
| record | side_effects | 3 | 0.31% |
| record | company.name | 3 | 0.31% |
| record | warranty_period | 3 | 0.31% |
| record | quantity | 3 | 0.31% |
| record | investor | 3 | 0.31% |
| record | relationships.0.type | 3 | 0.31% |
| record | test_id | 3 | 0.31% |
| record | employee_id: Unique identifier assigned to the employee | 3 | 0.31% |
| record | hire_date: Date when the position is expected to be filled | 3 | 0.31% |
| record | job_title: Title of the position being advertised | 3 | 0.31% |
| record | location: City and state where the role is based | 3 | 0.31% |
| record | imaging_date | 3 | 0.31% |
| record | review_source | 3 | 0.31% |
| record | tv_program.production.production_area | 3 | 0.31% |
| record | tv_program.setting.year | 3 | 0.31% |
| record | tv_program.production.location.name | 3 | 0.31% |
| record | project_name | 3 | 0.31% |
| record | service_description | 3 | 0.31% |
| record | status: Current state of the transaction | 3 | 0.31% |
| record | cost | 3 | 0.31% |
| record | reporting_to: Name of the manager or department head this role reports to | 3 | 0.31% |
| record | funding_amount | 3 | 0.31% |
| record | payer | 2 | 0.21% |
| record | payee | 2 | 0.21% |
| record | root_cause | 2 | 0.21% |
| record | resolution_time | 2 | 0.21% |
| record | delay_reason | 2 | 0.21% |
| record | sender: Name of the person sending the confirmation | 2 | 0.21% |
| record | event_id: Conference or symposium identifier for the accepted abstract | 2 | 0.21% |
| record | presentation_slot: Time slot for the presentation at the conference | 2 | 0.21% |
| record | registration_fee: Monetary amount due for conference registration | 2 | 0.21% |
| record | institution | 2 | 0.21% |
| record | sample_size | 2 | 0.21% |
| record | sustainability.sourced | 2 | 0.21% |
| record | clinic_name | 2 | 0.21% |
| record | compensation: Annual salary or compensation amount | 2 | 0.21% |
| record | status: Current status of the appointment | 2 | 0.21% |
| record | role | 2 | 0.21% |
| record | concepts | 2 | 0.21% |
| record | room_type | 2 | 0.21% |
| record | last_updated | 2 | 0.21% |
| record | available_quantity | 2 | 0.21% |
| record | genus.related_genera | 2 | 0.21% |
| record | genus.pollination.pollinators | 2 | 0.21% |
| record | genus.german_species | 2 | 0.21% |
| record | genus.distribution | 2 | 0.21% |
| record | genus.characteristics | 2 | 0.21% |
| record | genus.name | 2 | 0.21% |
| record | problem_description | 2 | 0.21% |
| record | event.location | 2 | 0.21% |
| record | product.type | 2 | 0.21% |
| record | product.description | 2 | 0.21% |
| record | discharge_condition | 2 | 0.21% |
| record | league | 2 | 0.21% |
| record | match_id: Unique identifier for the match as provided | 2 | 0.21% |
| record | score: Final score in format home-away | 2 | 0.21% |
| record | nationality | 2 | 0.21% |
| record | transfertype | 2 | 0.21% |
| record | email_id: Unique identifier for the outreach email | 2 | 0.21% |
| record | message_length: Word count of the outreach message | 2 | 0.21% |
| record | organization.location_founded.country | 2 | 0.21% |
| record | dedication.description | 2 | 0.21% |
| record | app_name | 2 | 0.21% |
| record | coverage_type: Type of insurance coverage provided | 2 | 0.21% |
| record | tracking_id | 2 | 0.21% |
| record | open_rate | 2 | 0.21% |
| record | claim_id: Unique identifier for the insurance claim | 2 | 0.21% |
| record | claim_date: Date the claim was submitted | 2 | 0.21% |
| record | feedback | 2 | 0.21% |
| record | founder | 2 | 0.21% |
| record | startup | 2 | 0.21% |
| record | hotel.name | 2 | 0.21% |
| record | hotel.brand | 2 | 0.21% |
| record | hotel.location.distance_from_center | 2 | 0.21% |
| record | hotel.location.nearby_attractions | 2 | 0.21% |
| record | hotel.room_features | 2 | 0.21% |
| record | hotel.hotel_group | 2 | 0.21% |
| record | access_type | 2 | 0.21% |
| record | review_length | 2 | 0.21% |
| record | transfer_type | 2 | 0.21% |
| record | goals_scored | 2 | 0.21% |
| record | product.origin.method_origin | 2 | 0.21% |
| record | product.origin.spread_location | 2 | 0.21% |
| record | product.ingredients | 2 | 0.21% |
| record | product.preparation.steps | 2 | 0.21% |
| record | product.preparation.drying_time | 2 | 0.21% |
| record | product.storage.containers | 2 | 0.21% |
| record | product.storage.wrapping | 2 | 0.21% |
| record | historical_context.cappers_weekly.call_for_articles_year | 2 | 0.21% |
| record | historical_context.cappers_weekly.topic | 2 | 0.21% |
| record | historical_context.my_folks_series.first_book_year | 2 | 0.21% |
| record | historical_context.my_folks_series.number_of_books | 2 | 0.21% |
| record | invoice_number: Unique identifier for the invoice as assigned | 2 | 0.21% |
| record | sender: Name of the company that issued the invoice | 2 | 0.21% |
| record | recipient_address: Street address where goods or services were delivered | 2 | 0.21% |
| record | status: Current financial status of the invoice | 2 | 0.21% |
| record | risk_score | 2 | 0.21% |
| record | donor_id | 2 | 0.21% |
| record | donation_date | 2 | 0.21% |
| record | comment_id | 2 | 0.21% |
| record | upvotes | 2 | 0.21% |
| record | content | 2 | 0.21% |
| record | theme | 2 | 0.21% |
| record | cas_number | 2 | 0.21% |
| record | issuer | 2 | 0.21% |
| record | asteroid.discovery_year | 2 | 0.21% |
| record | asteroid.orbital_characteristics.inclination_unit | 2 | 0.21% |
| record | manifest_id | 2 | 0.21% |
| record | receiver | 2 | 0.21% |
| record | revenue | 2 | 0.21% |
| record | research.researcher.affiliation.abbreviation | 2 | 0.21% |
| record | research.objective | 2 | 0.21% |
| record | research.experiment_details.atoms_used | 2 | 0.21% |
| record | research.experiment_details.temperature_details | 2 | 0.21% |
| record | research.experiment_details.laser_details | 2 | 0.21% |
| record | office | 2 | 0.21% |
| record | policyholder: Name of the vehicle owner filing the claim | 2 | 0.21% |
| record | damage_amount: Total monetary damage assessed in dollars | 2 | 0.21% |
| record | vehicle: Make, model, and year of the vehicle involved | 2 | 0.21% |
| record | adjuster: Name of the assigned claims adjuster | 2 | 0.21% |
| record | location: Geographic headquarters of the company | 2 | 0.21% |
| record | net_profit: Annual net profit in millions of dollars | 2 | 0.21% |
| record | ebitda_margin: Percentage of revenue that represents EBITDA | 2 | 0.21% |
| record | current_ratio: Liquidity ratio measuring short-term obligations | 2 | 0.21% |
| record | operational_status: Current phase of business operations | 2 | 0.21% |
| record | rating: Numerical score out of 5 given by the reviewer | 2 | 0.21% |
| record | reviewer: Name of the critic or editorial team | 2 | 0.21% |
| record | duration.max_years | 2 | 0.21% |
| record | project_goals | 2 | 0.21% |
| record | duration.min_years | 2 | 0.21% |
| record | beneficiary_focus | 2 | 0.21% |
| record | hotel_initiatives | 2 | 0.21% |
| record | passenger_name | 2 | 0.21% |
| record | performance_rating | 2 | 0.21% |
| record | winner_id: Unique ID assigned to the winner | 2 | 0.21% |
| record | opponent: Name of the player or team that lost | 2 | 0.21% |
| record | date: Date the tournament was held | 2 | 0.21% |
| record | provider | 2 | 0.21% |
| record | drug_name: Name of the medication involved in the reaction | 2 | 0.21% |
| record | report_id: Unique identifier assigned to the adverse event report | 2 | 0.21% |
| record | patient_address: Full residential address of the patient | 2 | 0.21% |
| record | material_efficiency | 2 | 0.21% |
| record | base_currency: Original currency used in transaction | 2 | 0.21% |
| record | target_currency: Currency to which amount is converted | 2 | 0.21% |
| record | location: Geographic location of transaction | 2 | 0.21% |
| record | years_mentioned | 2 | 0.21% |
| record | renewal_period | 2 | 0.21% |
| record | location: Physical address of the product launch | 2 | 0.21% |
| record | product_code: Internal code assigned to the product | 2 | 0.21% |
| record | price: Selling price of the product in dollars | 2 | 0.21% |
| record | post_id | 2 | 0.21% |
| record | cost_savings | 2 | 0.21% |
| record | approval_year | 2 | 0.21% |
| record | respiratory_rate | 2 | 0.21% |
| record | ticket_id: Unique identifier for the incident ticket | 2 | 0.21% |
| record | reporter: Name of the engineer or team member who filed the ticket | 2 | 0.21% |
| record | filed_at: Timestamp when the ticket was opened | 2 | 0.21% |
| record | region: Geographic region where the issue occurred | 2 | 0.21% |
| record | root_cause: Hypothesized cause of the incident based on logs or analysis | 2 | 0.21% |
| record | error_log: Relevant error message or log snippet from the incident | 2 | 0.21% |
| record | location-gpe | 2 | 0.21% |
| record | applicant | 2 | 0.21% |
| record | claim_count | 2 | 0.21% |
| record | assignee | 2 | 0.21% |
| record | patient_id: Unique identifier for the patient | 2 | 0.21% |
| record | temperature: Body temperature in degrees Fahrenheit | 2 | 0.21% |
| record | submission_id | 2 | 0.21% |
| record | heart_rate | 2 | 0.21% |
| record | customer_name | 2 | 0.21% |
| record | match | 2 | 0.21% |
| record | album | 2 | 0.21% |
| record | required_skills | 2 | 0.21% |
| record | job_title | 2 | 0.21% |
| record | procedure | 2 | 0.21% |
| record | result | 2 | 0.21% |
| record | comparative_authors | 2 | 0.21% |
| record | product.brand | 2 | 0.21% |
| record | author.name | 2 | 0.21% |
| record | orbital_characteristics.semi_major_axis.unit | 2 | 0.21% |
| record | last_name | 2 | 0.21% |
| record | first_name | 2 | 0.21% |
| record | staff | 2 | 0.21% |
| record | powers_abilities | 2 | 0.21% |
| record | broadcast_network | 2 | 0.21% |
| record | status: Current status of the return approval | 2 | 0.21% |
| record | returner: Name of the person initiating the return | 2 | 0.21% |
| record | item: Description of the returned product | 2 | 0.21% |
| record | user: Name of the team member or user using the product | 2 | 0.21% |
| record | performance_impact: Percentage improvement in lead quality or hiring time | 2 | 0.21% |
| record | product_name: Name of the product being used | 2 | 0.21% |
| record | company: Organization where the product is used | 2 | 0.21% |
| record | to_club | 2 | 0.21% |
| record | property | 2 | 0.21% |
| record | lease_term | 2 | 0.21% |
| record | damage_type | 2 | 0.21% |
| record | environment | 2 | 0.21% |
| record | discovery.project | 2 | 0.21% |
| record | diameter.unit | 2 | 0.21% |
| record | drug_name | 2 | 0.21% |
| record | serial_number: Unique hardware identifier for the device | 2 | 0.21% |
| record | return_date: Date device was returned to repair center | 2 | 0.21% |
| record | submitter: Name of the customer who submitted the ticket | 2 | 0.21% |
| record | device_model: Model name of the returned electronic device | 2 | 0.21% |
| record | ticket_number: Unique identifier for the RMA request | 2 | 0.21% |
| record | place.location.country | 2 | 0.21% |
| record | units | 2 | 0.21% |
| record | available_units | 2 | 0.21% |
| record | line1 | 2 | 0.21% |
| record | hotel_address | 2 | 0.21% |
| record | statement_reference: Identifier code for the statement document | 2 | 0.21% |
| record | location.department | 2 | 0.21% |
| record | person.birth_date | 2 | 0.21% |
| record | person.name | 2 | 0.21% |
| record | lab_id | 2 | 0.21% |
| record | date: Date when the transaction occurred | 2 | 0.21% |
| record | policytype | 2 | 0.21% |
| record | description: Detailed account of the outage and impact | 2 | 0.21% |
| record | duration: Duration of outage in hours | 2 | 0.21% |
| record | tier: Data speed tier of the service plan | 2 | 0.21% |
| record | user_name | 2 | 0.21% |
| record | user_location | 2 | 0.21% |
| record | physical_characteristics.observations.0.telescope.name | 2 | 0.21% |
| record | physical_characteristics.observations.0.publication_year | 2 | 0.21% |
| record | discovery.observatory.location.state | 2 | 0.21% |
| record | jurisdiction | 2 | 0.21% |
| record | entity_name | 2 | 0.21% |
| record | file_id | 2 | 0.21% |
| record | status: Current state of the tournament | 2 | 0.21% |
| record | format: Match structure used in the tournament | 2 | 0.21% |
| record | name: Full name of the esports tournament | 2 | 0.21% |
| record | final_match | 2 | 0.21% |
| record | organizer | 2 | 0.21% |
| record | discovery.location | 2 | 0.21% |
| record | publication.author.name | 2 | 0.21% |
| record | publication.author.birthYear | 2 | 0.21% |
| record | publication.author.deathYear | 2 | 0.21% |
| record | publication.author.affiliation.institution | 2 | 0.21% |
| record | publication.author.affiliation.country | 2 | 0.21% |
| record | case_id | 2 | 0.21% |
| record | outreach_date: Date when the outreach email was sent | 2 | 0.21% |
| record | status: Current status of the outreach interaction | 2 | 0.21% |
| record | affected_area | 2 | 0.21% |
| record | status: Current state of the incident | 2 | 0.21% |
| record | ppe_required | 2 | 0.21% |
| record | transfer_fee | 2 | 0.21% |
| record | transfer_portal | 2 | 0.21% |
| record | description.flowers.color | 2 | 0.21% |
| record | description.inflorescence.type | 2 | 0.21% |
| record | description.seeds.color | 2 | 0.21% |
| record | description.leaves.terminal_spine.color | 2 | 0.21% |
| record | description.fruit.width_mm.min | 2 | 0.21% |
| record | description.flowering_time | 2 | 0.21% |
| record | description.fruit.length_mm.min | 2 | 0.21% |
| record | systematics_and_distribution.location.state | 2 | 0.21% |
| record | systematics_and_distribution.first_description.author | 2 | 0.21% |
| record | description.leaves.margins | 2 | 0.21% |
| record | description.leaves.width_cm.min | 2 | 0.21% |
| record | description.inflorescence.height_m.min | 2 | 0.21% |
| record | description.flowers.length_mm.max | 2 | 0.21% |
| record | common_names | 2 | 0.21% |
| record | systematics_and_distribution.location.country | 2 | 0.21% |
| record | systematics_and_distribution.similar_species | 2 | 0.21% |
| record | description.fruit.width_mm.max | 2 | 0.21% |
| record | description.flowers.arrangement | 2 | 0.21% |
| record | systematics_and_distribution.location.elevation_m.min | 2 | 0.21% |
| record | description.leaves.surface | 2 | 0.21% |
| record | primary_treatment | 2 | 0.21% |
| record | insurance | 2 | 0.21% |
| record | score: Final score of the match in set format | 2 | 0.21% |
| record | location: Venue where the match was held | 2 | 0.21% |
| record | venue_name: Name of the specific tournament venue | 2 | 0.21% |
| record | prescription_date | 2 | 0.21% |
| record | fda_code | 2 | 0.21% |
| record | shipping_cost | 2 | 0.21% |
| record | question.keywords | 2 | 0.21% |
| record | details.date_posted | 2 | 0.21% |
| record | attendee_count | 2 | 0.21% |
| record | status: Current operational status of the device | 2 | 0.21% |
| record | treatment_history | 2 | 0.21% |
| record | prior_hospitalization | 2 | 0.21% |
| record | current_status | 2 | 0.21% |
| record | icd10code | 2 | 0.21% |
| record | phone | 2 | 0.21% |
| record | dateofincident | 2 | 0.21% |
| record | sunless_tanning.product_collection.collection_name | 2 | 0.21% |
| record | sunless_tanning.product_collection.benefits | 2 | 0.21% |
| record | sunless_tanning.product_collection.products.0.type | 2 | 0.21% |
| record | sunless_tanning.product_collection.products.0.description | 2 | 0.21% |
| record | sunless_tanning.product_collection.products.0.ingredients | 2 | 0.21% |
| record | sunless_tanning.product_collection.products.1.benefits | 2 | 0.21% |
| record | sunless_tanning.product_collection.products.1.name | 2 | 0.21% |
| record | sunless_tanning.product_collection.products.1.ingredients | 2 | 0.21% |
| record | sunless_tanning.product_collection.products.1.description | 2 | 0.21% |
| record | policy_id: Policy number associated with the claim | 2 | 0.21% |
| record | incident_type: Type of incident causing the claim | 2 | 0.21% |
| record | owner: Name of the claimant or policyholder | 2 | 0.21% |
| record | venue: Location where the event took place | 2 | 0.21% |
| record | employee_id: Unique identifier assigned to each employee | 2 | 0.21% |
| record | town.population.year | 2 | 0.21% |
| record | town.country | 2 | 0.21% |
| record | town.establishment_year | 2 | 0.21% |
| record | town.area_sq_km | 2 | 0.21% |
| record | consultation.practice_areas | 2 | 0.21% |
| record | services | 2 | 0.21% |
| record | firm_name | 2 | 0.21% |
| record | client_focus | 2 | 0.21% |
| record | experience.years | 2 | 0.21% |
| record | event.date | 2 | 0.21% |
| record | power_outage | 2 | 0.21% |
| record | soil_remediation_project | 2 | 0.21% |
| record | max_wind_speed | 2 | 0.21% |
| record | laboratories | 2 | 0.21% |
| record | map_location | 2 | 0.21% |
| record | dean.year | 2 | 0.21% |
| record | staff.research_assistant_count | 2 | 0.21% |
| record | staff.teaching_staff_count | 2 | 0.21% |
| record | departments | 2 | 0.21% |
| record | campus | 2 | 0.21% |
| record | shipping | 2 | 0.21% |
| record | starting_dose | 2 | 0.21% |
| record | storage | 2 | 0.21% |
| record | drug_interactions | 2 | 0.21% |
| record | flight_number | 2 | 0.21% |
| record | original_departure | 2 | 0.21% |
| record | new_departure | 2 | 0.21% |
| record | likes | 2 | 0.21% |
| record | company.address.street | 2 | 0.21% |
| record | company.address.shop_name | 2 | 0.21% |
| record | company.locations | 2 | 0.21% |
| record | company.contact_information.fax_numbers | 2 | 0.21% |
| record | company.contact_information.phone_numbers | 2 | 0.21% |
| record | company.address.district | 2 | 0.21% |
| record | company.address.area | 2 | 0.21% |
| record | requisition_id: Unique identifier for the job posting | 2 | 0.21% |
| record | search_result_page.search_query | 2 | 0.21% |
| record | search_result_page.website_name | 2 | 0.21% |
| record | initial_text.negations | 2 | 0.21% |
| record | exam_date | 2 | 0.21% |
| record | affected_entity: Organization or business impacted by the violation | 2 | 0.21% |
| record | location: Physical address of the affected entity | 2 | 0.21% |
| record | violation_code: Code assigned to the specific violation | 2 | 0.21% |
| record | listing_id: Unique identifier for each property listing | 2 | 0.21% |
| record | venue_id | 2 | 0.21% |
| record | orbital_characteristics.inclination | 2 | 0.21% |
| record | blood_pressure | 2 | 0.21% |
| record | client_name | 2 | 0.21% |
| record | merchant_code | 2 | 0.21% |
| record | credit_hours | 2 | 0.21% |
| record | start_date | 2 | 0.21% |
| record | admitting_diagnosis | 2 | 0.21% |
| record | note | 2 | 0.21% |
| record | parties | 2 | 0.21% |
| record | plaintiff_address | 2 | 0.21% |
| record | case_number | 2 | 0.21% |
| record | industry_focus | 2 | 0.21% |
| record | feature_status | 2 | 0.21% |
| record | internal_code | 2 | 0.21% |
| record | concept | 2 | 0.21% |
| record | movements | 2 | 0.21% |
| record | related_terms | 2 | 0.21% |
| record | shipping_origin: City and state where product is shipped from | 2 | 0.21% |
| record | category: Broad product category such as home, tech, health, outdoor | 2 | 0.21% |
| record | stock_status: Current availability status such as active, backordered, on-hold, out-of-stock | 2 | 0.21% |
| record | retail_price: Price at retail in US dollars | 2 | 0.21% |
| record | district | 2 | 0.21% |
| record | report_id: Unique identifier for the phishing incident | 2 | 0.21% |
| record | match_id | 2 | 0.21% |
| record | functionality_comment | 2 | 0.21% |
| record | relationships.0.action | 2 | 0.21% |
| record | relationships.0.entity | 2 | 0.21% |
| record | rate: Exchange rate in USD per foreign currency unit | 2 | 0.21% |
| record | status: Current status of the exchange rate record | 2 | 0.21% |
| record | snapshot_id: Unique identifier for the snapshot in format currency-yearmonthday | 2 | 0.21% |
| record | user_id: ID of the user who submitted the snapshot | 2 | 0.21% |
| record | winner | 2 | 0.21% |
| record | steals: Total steals recorded during the game | 2 | 0.21% |
| record | home_score: Final score of the home team | 2 | 0.21% |
| record | bench_points: Total points scored by bench players | 2 | 0.21% |
| record | asteroid.discovery.discoverer.name | 2 | 0.21% |
| record | name: Full name of the employee as listed in the review | 2 | 0.21% |
| record | department: Department where the employee works | 2 | 0.21% |
| record | goal_achievement: Percentage of goals met, from 60% to 100% | 2 | 0.21% |
| record | legislative_details.initiators | 2 | 0.21% |
| record | session_id | 2 | 0.21% |
| record | department: Business unit or division where the role is located | 2 | 0.21% |
| record | required_qualifications: List of educational or skill-based requirements | 2 | 0.21% |
| record | offensive lineman | 2 | 0.21% |
| record | headquarters | 2 | 0.21% |
| record | entities_involved | 2 | 0.21% |
| record | author.role | 2 | 0.21% |
| record | version | 2 | 0.21% |
| record | battery_life | 2 | 0.21% |
| record | album_title | 2 | 0.21% |
| record | artist | 2 | 0.21% |
| record | tv_program.production.location.type | 2 | 0.21% |
| record | tv_program.previous_program | 2 | 0.21% |
| record | other_programs | 2 | 0.21% |
| record | product | 2 | 0.21% |
| record | author_handle | 2 | 0.21% |
| record | tech_product | 2 | 0.21% |
| record | test_type | 2 | 0.21% |
| record | scholarship_id | 2 | 0.21% |
| record | disbursement_plan | 2 | 0.21% |
| record | products | 2 | 0.21% |
| record | url | 2 | 0.21% |
| record | project_name: Name of the technology project | 2 | 0.21% |
| record | project_id: Unique identifier assigned to the project | 2 | 0.21% |
| record | accuracy_efficiency: Performance metric like accuracy or efficiency | 2 | 0.21% |
| record | technology: Core technology or method used | 2 | 0.21% |
| record | status: Current status of the encounter | 2 | 0.21% |
| record | encounter_date: Date of the patient visit | 2 | 0.21% |
| record | player | 2 | 0.21% |
| record | points | 2 | 0.21% |
| record | company.services | 2 | 0.21% |
| record | invoice_id | 2 | 0.21% |
| record | contact_address | 2 | 0.21% |
| record | product.browser_compatibility | 2 | 0.21% |
| record | product.download_link | 2 | 0.21% |
| record | context.solution | 2 | 0.21% |
| record | program_name | 2 | 0.21% |
| record | orbital_characteristics.perihelion | 2 | 0.21% |
| record | observations.last_observation_date | 2 | 0.21% |
| record | dose | 2 | 0.21% |
| record | pathway_id | 2 | 0.21% |
| record | asteroid.discovery.discoverer.nationality | 2 | 0.21% |
| record | asteroid.naming.named_after | 2 | 0.21% |
| record | asteroid.discovery.observatory.name | 2 | 0.21% |
| record | asteroid.discovery.discoverer.profession | 2 | 0.21% |
| record | asteroid.discovery.observatory.location.country | 2 | 0.21% |
| record | asteroid.orbital_period.orbiting | 2 | 0.21% |
| record | date: Date when the tournament took place | 2 | 0.21% |
| record | salary | 2 | 0.21% |
| record | lead_investor | 2 | 0.21% |
| record | publication | 2 | 0.21% |
| record | reference_id | 2 | 0.21% |
| record | supervisor | 2 | 0.21% |
| record | contact_name | 2 | 0.21% |
| record | transaction_id: Unique identifier for each transaction entry | 2 | 0.21% |
| record | asteroid.discovery.year | 2 | 0.21% |
| record | tv_program.production.location.institution | 2 | 0.21% |
| record | tv_program.setting.location | 2 | 0.21% |
| record | school_address | 2 | 0.21% |
| record | publication_date | 2 | 0.21% |
| record | result_count | 2 | 0.21% |
| record | vaccine_type: Type of vaccine administered | 2 | 0.21% |
| record | discovery.observatory.location | 2 | 0.21% |
| record | discovery.discoverers | 2 | 0.21% |
| record | useoffunds: Purpose or application of raised capital | 2 | 0.21% |
| record | funding_amount: Total amount raised in millions | 2 | 0.21% |
| record | assists | 2 | 0.21% |
| record | literary_movements | 2 | 0.21% |
| record | places | 2 | 0.21% |
| record | company_name | 2 | 0.21% |
| record | study.title | 2 | 0.21% |
| record | study.key_findings | 2 | 0.21% |
| record | study.period.end_year | 2 | 0.21% |
| record | study.author.affiliation | 2 | 0.21% |
| record | claim_date | 2 | 0.21% |
| record | claimant_name | 2 | 0.21% |
| record | person.birth_place.city | 2 | 0.21% |
| record | person.us_senator.start_year | 2 | 0.21% |
| record | person.birth_place.state | 2 | 0.21% |
| record | asteroid_name | 2 | 0.21% |
| record | strength | 2 | 0.21% |
| record | patient_name: Name of the patient filing the claim | 2 | 0.21% |
| record | positive_note | 2 | 0.21% |
| record | annual_revenue | 2 | 0.21% |
| record | opening_quarter | 2 | 0.21% |
| record | chipdesignfocus | 2 | 0.21% |
| record | caveat.example.meaning | 2 | 0.21% |
| record | comparison.ideographic.example.meaning | 2 | 0.21% |
| record | caveat.example.components.0.component | 2 | 0.21% |
| record | terminology.ideogram.english | 2 | 0.21% |
| record | comparison.alphabetic.pronunciation | 2 | 0.21% |
| record | caveat.example.components.1.position | 2 | 0.21% |
| record | announcement.divisions.2.second_star.position | 2 | 0.21% |
| record | announcement.divisions.2.honorable_mentions.4.player_name | 2 | 0.21% |
| record | announcement.divisions.4.star_of_the_week.team | 2 | 0.21% |
| record | announcement.divisions.0.honorable_mentions.6.player_name | 2 | 0.21% |
| record | announcement.divisions.4.honorable_mentions.2.position | 2 | 0.21% |
| record | announcement.divisions.5.honorable_mentions.1.position | 2 | 0.21% |
| record | announcement.divisions.2.star_of_the_week.team | 2 | 0.21% |
| record | announcement.divisions.4.honorable_mentions.0.player_name | 2 | 0.21% |
| record | announcement.divisions.0.honorable_mentions.1.player_name | 2 | 0.21% |
| record | announcement.divisions.0.honorable_mentions.6.position | 2 | 0.21% |
| record | announcement.divisions.4.honorable_mentions.1.position | 2 | 0.21% |
| record | announcement.divisions.0.honorable_mentions.2.player_name | 2 | 0.21% |
| record | announcement.divisions.3.division_name | 2 | 0.21% |
| record | announcement.divisions.5.second_star.player_name | 2 | 0.21% |
| record | announcement.divisions.2.honorable_mentions.0.team | 2 | 0.21% |
| record | announcement.divisions.1.honorable_mentions.1.player_name | 2 | 0.21% |
| record | announcement.league | 2 | 0.21% |
| record | announcement.divisions.5.honorable_mentions.0.player_name | 2 | 0.21% |
| record | announcement.divisions.4.honorable_mentions.2.player_name | 2 | 0.21% |
| record | announcement.divisions.5.honorable_mentions.2.team | 2 | 0.21% |
| record | announcement.divisions.3.honorable_mentions.1.position | 2 | 0.21% |
| record | announcement.divisions.2.honorable_mentions.2.player_name | 2 | 0.21% |
| record | announcement.divisions.3.honorable_mentions.1.team | 2 | 0.21% |
| record | announcement.divisions.0.honorable_mentions.5.team | 2 | 0.21% |
| record | announcement.divisions.4.honorable_mentions.0.position | 2 | 0.21% |
| record | announcement.divisions.1.second_star.position | 2 | 0.21% |
| record | announcement.divisions.2.honorable_mentions.4.team | 2 | 0.21% |
| record | announcement.divisions.4.second_star.team | 2 | 0.21% |
| record | announcement.divisions.0.star_of_the_week.team | 2 | 0.21% |
| record | announcement.divisions.1.second_star.player_name | 2 | 0.21% |
| record | announcement.divisions.0.honorable_mentions.7.position | 2 | 0.21% |
| record | announcement.divisions.4.honorable_mentions.1.player_name | 2 | 0.21% |
| record | announcement.divisions.0.honorable_mentions.0.position | 2 | 0.21% |
| record | announcement.divisions.5.honorable_mentions.1.team | 2 | 0.21% |
| record | announcement.divisions.5.honorable_mentions.0.team | 2 | 0.21% |
| record | announcement.divisions.3.star_of_the_week.player_name | 2 | 0.21% |
| record | announcement.divisions.0.honorable_mentions.2.team | 2 | 0.21% |
| record | announcement.divisions.1.star_of_the_week.position | 2 | 0.21% |
| record | transaction_type | 2 | 0.21% |
| record | source_provider: Provider handling the exchange rate calculation | 2 | 0.21% |
| record | user_id: Unique identifier for the user initiating transaction | 2 | 0.21% |
| record | transaction_amount: Amount being converted | 2 | 0.21% |
| record | username | 2 | 0.21% |
| record | genus.type_species | 2 | 0.21% |
| record | side dish | 2 | 0.21% |
| record | dish | 2 | 0.21% |
| record | restaurant | 2 | 0.21% |
| record | food item | 2 | 0.21% |
| record | project_url | 2 | 0.21% |
| record | contributor_count | 2 | 0.21% |
| record | politician | 2 | 0.21% |
| record | bracket_size | 2 | 0.21% |
| record | line2 | 2 | 0.21% |
| record | claim_id: Unique identifier for each claim entry | 2 | 0.21% |
| record | transaction_id: Unique identifier for the transaction | 2 | 0.21% |
| record | status: Current status of the transaction | 2 | 0.21% |
| record | monitoring_mechanism | 2 | 0.21% |
| record | target_year | 2 | 0.21% |
| record | signing_countries | 2 | 0.21% |
| record | shared_rights | 2 | 0.21% |
| record | consensus_percentage | 2 | 0.21% |
| record | meeting_id | 2 | 0.21% |
| record | topic: Subject matter discussed in the statement | 2 | 0.21% |
| record | lead_investor: Primary investor leading the funding round | 2 | 0.21% |
| record | condition | 2 | 0.21% |
| record | review_type | 2 | 0.21% |
| record | species | 2 | 0.21% |
| record | family.name | 2 | 0.21% |
| record | authority.name | 2 | 0.21% |
| record | claimant | 2 | 0.21% |
| record | publication.author | 2 | 0.21% |
| record | webmaster | 2 | 0.21% |
| record | last_modified | 2 | 0.21% |
| record | filing_date | 2 | 0.21% |
| record | salary_range: Range of expected compensation in dollar format | 2 | 0.21% |
| record | asteroid.discovery.discoverer | 2 | 0.21% |
| record | categories | 2 | 0.21% |
| record | post_id: Unique identifier for the social media post | 2 | 0.21% |
| record | target_audience: Demographic group the product targets | 2 | 0.21% |
| record | call_id | 2 | 0.21% |
| record | yellow_cards | 2 | 0.21% |
| record | red_cards | 2 | 0.21% |
| record | match_outcome | 2 | 0.21% |
| record | resources | 2 | 0.21% |
| record | github_id | 2 | 0.21% |
| record | event.person.name | 1 | 0.10% |
| record | event.new_club.contract_duration | 1 | 0.10% |
| record | event.person.age | 1 | 0.10% |
| record | event.new_club.name | 1 | 0.10% |
| record | event.opponent_club.name | 1 | 0.10% |
| record | event.person.position | 1 | 0.10% |
| record | event.person.role | 1 | 0.10% |
| record | service | 1 | 0.10% |
| record | passenger_address | 1 | 0.10% |
| record | departure_original | 1 | 0.10% |
| record | flight_id | 1 | 0.10% |
| record | departure_actual | 1 | 0.10% |
| record | title: Title of the abstract submitted for presentation | 1 | 0.10% |
| record | submission_id: Unique identifier assigned to the abstract submission | 1 | 0.10% |
| record | status: Current status of the abstract review | 1 | 0.10% |
| record | recipient: Name of the person receiving the confirmation | 1 | 0.10% |
| record | funding_grant | 1 | 0.10% |
| record | geographic_context | 1 | 0.10% |
| record | study_duration | 1 | 0.10% |
| record | intervention | 1 | 0.10% |
| record | recharge_amount | 1 | 0.10% |
| record | bonus_amount | 1 | 0.10% |
| record | publicationDate | 1 | 0.10% |
| record | notes: Additional context about the appointment or background | 1 | 0.10% |
| record | date: Start date of the appointment | 1 | 0.10% |
| record | department: Government department or agency where the role is held | 1 | 0.10% |
| record | address: Physical address of the appointee | 1 | 0.10% |
| record | mls_number | 1 | 0.10% |
| record | agent | 1 | 0.10% |
| record | bathrooms | 1 | 0.10% |
| record | article.symptoms | 1 | 0.10% |
| record | article.references.0.description | 1 | 0.10% |
| record | article.translation_note.english_version.url | 1 | 0.10% |
| record | article.id | 1 | 0.10% |
| record | article.references.0.id | 1 | 0.10% |
| record | article.last_reviewed | 1 | 0.10% |
| record | article.resolution | 1 | 0.10% |
| record | article.cause | 1 | 0.10% |
| record | platform: Type of social media post shared | 1 | 0.10% |
| record | author: Name of user who posted the content | 1 | 0.10% |
| record | bp | 1 | 0.10% |
| record | hr | 1 | 0.10% |
| record | rr | 1 | 0.10% |
| record | spo2 | 1 | 0.10% |
| record | temp | 1 | 0.10% |
| record | doctor | 1 | 0.10% |
| record | research_submission | 1 | 0.10% |
| record | nightly_rate | 1 | 0.10% |
| record | genre | 1 | 0.10% |
| record | address: Full street address of the rental property | 1 | 0.10% |
| record | status: Current availability status of the unit | 1 | 0.10% |
| record | date: Date when the availability was recorded | 1 | 0.10% |
| record | unit: Unit designation such as 4B or 6C | 1 | 0.10% |
| record | genus.pollination.mechanism | 1 | 0.10% |
| record | genus.etymology.part2.name | 1 | 0.10% |
| record | genus.etymology.part1.name | 1 | 0.10% |
| record | genus.etymology.part1.related_entity | 1 | 0.10% |
| record | genus.etymology.part1.origin | 1 | 0.10% |
| record | genus.species_list | 1 | 0.10% |
| record | problem_start_time | 1 | 0.10% |
| record | affected_network | 1 | 0.10% |
| record | founder: Name of the individual who founded the company | 1 | 0.10% |
| record | founding_year: Year the company was established | 1 | 0.10% |
| record | location: City and state or country where the company is based | 1 | 0.10% |
| record | funding_source: Type of funding received by the company | 1 | 0.10% |
| record | annual_revenue: Annual revenue reported in millions of dollars | 1 | 0.10% |
| record | current_status: Current operational phase of the company | 1 | 0.10% |
| record | business_focus: Core service or product area of the company | 1 | 0.10% |
| record | clientbasesize: Number of clients served across regions | 1 | 0.10% |
| record | product.availability | 1 | 0.10% |
| record | event.description | 1 | 0.10% |
| record | event.year | 1 | 0.10% |
| record | advertisement.celebrity | 1 | 0.10% |
| record | urgency: Priority level of the reported issue | 1 | 0.10% |
| record | location: Physical or remote location of the submitter | 1 | 0.10% |
| record | department: Department associated with the ticket | 1 | 0.10% |
| record | ticket_id: Unique identifier for the ticket in the system | 1 | 0.10% |
| record | admit_diagnosis | 1 | 0.10% |
| record | Robert <> cited | 1 | 0.10% |
| record | Robert <> interacted with | 1 | 0.10% |
| record | Robert <> practiced | 1 | 0.10% |
| record | Robert <> experienced | 1 | 0.10% |
| record | Robert <> advocated | 1 | 0.10% |
| record | Eckart <> compared to | 1 | 0.10% |
| record | Master Eckart <> responded to | 1 | 0.10% |
| record | Robert <> identified as | 1 | 0.10% |
| record | Robert <> influenced by | 1 | 0.10% |
| record | Zen master <> replied to | 1 | 0.10% |
| record | Robert <> favored | 1 | 0.10% |
| record | Robert <> acknowledged | 1 | 0.10% |
| record | Robert <> observed | 1 | 0.10% |
| record | Walker <> explained | 1 | 0.10% |
| record | Edwin Harris Walker <> developed | 1 | 0.10% |
| record | Robert <> was asked by | 1 | 0.10% |
| record | competition: League or cup type such as Premier League or friendly | 1 | 0.10% |
| record | audit_id | 1 | 0.10% |
| record | lead_reporter: Reporter providing coverage of the incident | 1 | 0.10% |
| record | title: Event title describing the nature and context of the incident | 1 | 0.10% |
| record | location: Geographic location where the incident took place | 1 | 0.10% |
| record | impact: Description of human, financial, or systemic consequences | 1 | 0.10% |
| record | date: Timestamp when the incident occurred | 1 | 0.10% |
| record | article_metadata.relative_timestamp | 1 | 0.10% |
| record | subjects_involved | 1 | 0.10% |
| record | expert_opinion.speaker.affiliation.organization | 1 | 0.10% |
| record | expert_opinion.speaker.name | 1 | 0.10% |
| record | article_metadata.headline | 1 | 0.10% |
| record | priority: Severity level assigned to the ticket | 1 | 0.10% |
| record | ticket_id: Unique identifier for the support case | 1 | 0.10% |
| record | user: Name of the user who filed the ticket | 1 | 0.10% |
| record | company: Name of the recipient's employer organization | 1 | 0.10% |
| record | organization.year_founded | 1 | 0.10% |
| record | candidate_location | 1 | 0.10% |
| record | election_type | 1 | 0.10% |
| record | battery_life: Battery duration in hours under normal use | 1 | 0.10% |
| record | rating: Star rating out of 5 given by the reviewer | 1 | 0.10% |
| record | price: Purchase price in US dollars | 1 | 0.10% |
| record | renewal_date: Date when the policy is set to renew | 1 | 0.10% |
| record | notice_type: Method used to notify policyholder of renewal | 1 | 0.10% |
| record | address: Complete street address including city and state | 1 | 0.10% |
| record | subject | 1 | 0.10% |
| record | batteryorfeature | 1 | 0.10% |
| record | jurisdiction: Court jurisdiction and division where case is filed | 1 | 0.10% |
| record | address: Physical address of the filing party | 1 | 0.10% |
| record | respondent: Name of the person or entity being sued | 1 | 0.10% |
| record | registered_attendees | 1 | 0.10% |
| record | review_period | 1 | 0.10% |
| record | ratingcategory | 1 | 0.10% |
| record | performance_score | 1 | 0.10% |
| record | briefing_id | 1 | 0.10% |
| record | hotel.location.city | 1 | 0.10% |
| record | hotel.location.district | 1 | 0.10% |
| record | hotel.description | 1 | 0.10% |
| record | hotel.amenities | 1 | 0.10% |
| record | hotel.recreation_facilities | 1 | 0.10% |
| record | parameters | 1 | 0.10% |
| record | feedback_content | 1 | 0.10% |
| record | origin_club | 1 | 0.10% |
| record | destination_club | 1 | 0.10% |
| record | product.origin.initial_location | 1 | 0.10% |
| record | product.preparation.peach_selection | 1 | 0.10% |
| record | product.serving | 1 | 0.10% |
| record | historical_context.my_folks_series.first_book_title | 1 | 0.10% |
| record | review_text: Full textual critique of the film | 1 | 0.10% |
| record | review_id: Unique identifier for the review entry | 1 | 0.10% |
| record | rating: Numerical score given to the film | 1 | 0.10% |
| record | limitations | 1 | 0.10% |
| record | taxonomy.subgenus | 1 | 0.10% |
| record | description.distinguishing_features | 1 | 0.10% |
| record | species.etymology.named_after.0.location.city | 1 | 0.10% |
| record | description.width | 1 | 0.10% |
| record | distribution.environment | 1 | 0.10% |
| record | species.etymology.named_after.0.name | 1 | 0.10% |
| record | description.color | 1 | 0.10% |
| record | taxonomy.describer.name | 1 | 0.10% |
| record | taxonomy.year_described | 1 | 0.10% |
| record | distribution.locations | 1 | 0.10% |
| record | taxonomy.describer.affiliation.0.university | 1 | 0.10% |
| record | call_status | 1 | 0.10% |
| record | weight_range_kg.min_kg | 1 | 0.10% |
| record | weight_range_kg.max_kg | 1 | 0.10% |
| record | group_name | 1 | 0.10% |
| record | description: Brief text describing the service or product delivered | 1 | 0.10% |
| record | underwriter | 1 | 0.10% |
| record | policyholder | 1 | 0.10% |
| record | regulatory_status | 1 | 0.10% |
| record | platform_name | 1 | 0.10% |
| record | earnout | 1 | 0.10% |
| record | integration_plan | 1 | 0.10% |
| record | office_location | 1 | 0.10% |
| record | appointee_id | 1 | 0.10% |
| record | prior_experience | 1 | 0.10% |
| record | verbs_past_tense | 1 | 0.10% |
| record | prepositions | 1 | 0.10% |
| record | adverbs | 1 | 0.10% |
| record | proper_nouns | 1 | 0.10% |
| record | singular_nouns | 1 | 0.10% |
| record | plural_nouns | 1 | 0.10% |
| record | possessive_pronouns | 1 | 0.10% |
| record | adjectives | 1 | 0.10% |
| record | possessive_endings | 1 | 0.10% |
| record | cardinal_numbers | 1 | 0.10% |
| record | icd10_code | 1 | 0.10% |
| record | cpt_code | 1 | 0.10% |
| record | total_amount | 1 | 0.10% |
| record | asteroid.orbital_characteristics.unit | 1 | 0.10% |
| record | ebit_margin | 1 | 0.10% |
| record | revenuebyregion | 1 | 0.10% |
| record | research.experiment_details.interference_pattern | 1 | 0.10% |
| record | expected_close_date | 1 | 0.10% |
| record | employees | 1 | 0.10% |
| record | ceo | 1 | 0.10% |
| record | tournament_type | 1 | 0.10% |
| record | entries | 1 | 0.10% |
| record | winner_nationality | 1 | 0.10% |
| record | winner_name | 1 | 0.10% |
| record | winner_ranking | 1 | 0.10% |
| record | tournament_name | 1 | 0.10% |
| record | incident_type: Description of the type of incident | 1 | 0.10% |
| record | policy_number: Unique policy number associated with the claim | 1 | 0.10% |
| record | Steinem <> adopted by | 1 | 0.10% |
| record | Priest <> brought | 1 | 0.10% |
| record | bearded friend <> smoked | 1 | 0.10% |
| record | Steinem <> color | 1 | 0.10% |
| record | Steinem <> front paws on | 1 | 0.10% |
| record | Steinem <> rode in | 1 | 0.10% |
| record | Steinem <> lived for | 1 | 0.10% |
| record | Steinem <> nudged | 1 | 0.10% |
| record | my cousin <> adopted | 1 | 0.10% |
| record | Steinem <> species | 1 | 0.10% |
| record | disease_link: Disease or condition associated with the pathway | 1 | 0.10% |
| record | cell_type: Type of cell in which the pathway operates | 1 | 0.10% |
| record | clinical_relevance: Score indicating clinical importance | 1 | 0.10% |
| record | initiating_enzyme: Key enzyme initiating the pathway | 1 | 0.10% |
| record | key_intermediate: Central metabolite in the pathway | 1 | 0.10% |
| record | costperpatient | 1 | 0.10% |
| record | population_coverage | 1 | 0.10% |
| record | comorbidity_action | 1 | 0.10% |
| record | start_year | 1 | 0.10% |
| record | due_date | 1 | 0.10% |
| record | passenger_city | 1 | 0.10% |
| record | airline | 1 | 0.10% |
| record | employee_id: Unique identifier for each new hire entry | 1 | 0.10% |
| record | name: Full name of the new hire | 1 | 0.10% |
| record | office: Regional office location where employee is based | 1 | 0.10% |
| record | department: Department assigned to the employee | 1 | 0.10% |
| record | manager: Name of the employee's manager | 1 | 0.10% |
| record | training_completed: Percentage of training completed out of 100 | 1 | 0.10% |
| record | assigned_coordinator: Name of the onboarding coordinator assigned to the employee | 1 | 0.10% |
| record | dose_sequence | 1 | 0.10% |
| record | clinic_location | 1 | 0.10% |
| record | countries | 1 | 0.10% |
| record | cholesterol: Cholesterol level in mg/dL if elevated | 1 | 0.10% |
| record | birth_date: Date of birth in MM/DD/YYYY format | 1 | 0.10% |
| record | age: Age in years as stated in the record | 1 | 0.10% |
| record | zip_code: Postal code of patient's address | 1 | 0.10% |
| record | state: State abbreviation of patient's residence | 1 | 0.10% |
| record | patient_id: Unique identifier assigned to each patient | 1 | 0.10% |
| record | treatment: Type of treatment or management plan | 1 | 0.10% |
| record | hba1c: Hemoglobin A1c percentage value | 1 | 0.10% |
| record | patient_age: Age of the patient at time of reaction | 1 | 0.10% |
| record | patient_name: Name of the affected individual as reported | 1 | 0.10% |
| record | name: Full product name as listed in catalog | 1 | 0.10% |
| record | sku: Unique product identifier for inventory tracking | 1 | 0.10% |
| record | stock: Current quantity available in warehouse or distribution center | 1 | 0.10% |
| record | status: Current operational status of the product | 1 | 0.10% |
| record | owner: Name of the product owner or manager assigned | 1 | 0.10% |
| record | rating: Average customer review score out of 5 | 1 | 0.10% |
| record | review_count: Total number of customer reviews received | 1 | 0.10% |
| record | campaign: Marketing campaign currently promoting the product | 1 | 0.10% |
| record | location: Warehouse or distribution center address and city state zip | 1 | 0.10% |
| record | home_team: Team playing at the home venue | 1 | 0.10% |
| record | away_team: Team playing at the away venue | 1 | 0.10% |
| record | grilling_instructions.cooking_time.minutes | 1 | 0.10% |
| record | boiling_instructions.cooking_time.min_minutes | 1 | 0.10% |
| record | serving_suggestions | 1 | 0.10% |
| record | ingredients | 1 | 0.10% |
| record | session_code | 1 | 0.10% |
| record | landlord | 1 | 0.10% |
| record | listing_code | 1 | 0.10% |
| record | photos_count | 1 | 0.10% |
| record | overall_score | 1 | 0.10% |
| record | time_management | 1 | 0.10% |
| record | key_achievements | 1 | 0.10% |
| record | details | 1 | 0.10% |
| record | status: Current status of the donation transaction | 1 | 0.10% |
| record | donor: Name of the individual making the donation | 1 | 0.10% |
| record | date: Date the donation was made | 1 | 0.10% |
| record | goals | 1 | 0.10% |
| record | zip_code | 1 | 0.10% |
| record | technology_area | 1 | 0.10% |
| record | vaccines | 1 | 0.10% |
| record | vaccination_status | 1 | 0.10% |
| record | drug_id | 1 | 0.10% |
| record | mechanism | 1 | 0.10% |
| record | tissue_target | 1 | 0.10% |
| record | dosage_form | 1 | 0.10% |
| record | clinical_phase | 1 | 0.10% |
| record | patient_population | 1 | 0.10% |
| record | dish_cost | 1 | 0.10% |
| record | severity: Priority level of the incident | 1 | 0.10% |
| record | affected_service: Service or component impacted by the incident | 1 | 0.10% |
| record | resolution: Action taken to resolve or mitigate the incident | 1 | 0.10% |
| record | ticket_environment: Environment (e.g., production, staging, dev) affected | 1 | 0.10% |
| record | acquired | 1 | 0.10% |
| record | valuation_multiple | 1 | 0.10% |
| record | acquisition_amount | 1 | 0.10% |
| record | patent_number: Unique identifier for the patent application or grant | 1 | 0.10% |
| record | status: Current legal status of the patent | 1 | 0.10% |
| record | claims_count: Number of claims in the patent | 1 | 0.10% |
| record | spo2: Oxygen saturation percentage | 1 | 0.10% |
| record | blood_pressure: Blood pressure in format systolic/diastolic | 1 | 0.10% |
| record | heart_rate: Number of heartbeats per minute | 1 | 0.10% |
| record | address: Street address including city and zip code | 1 | 0.10% |
| record | co_authors | 1 | 0.10% |
| record | technology | 1 | 0.10% |
| record | bloodpressuresystolic | 1 | 0.10% |
| record | bloodpressurediastolic | 1 | 0.10% |
| record | monitoring_method | 1 | 0.10% |
| record | asteroid.named_after.person.name | 1 | 0.10% |
| record | asteroid.named_after.person.birth_year | 1 | 0.10% |
| record | asteroid.name_latin | 1 | 0.10% |
| record | asteroid.asteroid_type | 1 | 0.10% |
| record | physician.education.medical_school.0.institution | 1 | 0.10% |
| record | physician.education.residency.0.institution | 1 | 0.10% |
| record | physician.address.city | 1 | 0.10% |
| record | physician.address.zip | 1 | 0.10% |
| record | physician.name | 1 | 0.10% |
| record | physician.address.state | 1 | 0.10% |
| record | schema_name | 1 | 0.10% |
| record | hire_date | 1 | 0.10% |
| record | verse_reference | 1 | 0.10% |
| record | related_verses | 1 | 0.10% |
| record | speaker_name: Full name of the individual presenting the statement | 1 | 0.10% |
| record | property_id: Unique identifier for the property | 1 | 0.10% |
| record | address: Full physical address of the property | 1 | 0.10% |
| record | description: Brief textual description of property features and appeal | 1 | 0.10% |
| record | person.phone_number | 1 | 0.10% |
| record | review.price | 1 | 0.10% |
| record | review.cons | 1 | 0.10% |
| record | review.overall_rating | 1 | 0.10% |
| record | product.size | 1 | 0.10% |
| record | series_record | 1 | 0.10% |
| record | page_count | 1 | 0.10% |
| record | camera_model | 1 | 0.10% |
| record | price.amount | 1 | 0.10% |
| record | price.currency | 1 | 0.10% |
| record | author.experience | 1 | 0.10% |
| record | alternative_title | 1 | 0.10% |
| record | related_links | 1 | 0.10% |
| record | discovery.observatory.location.prefecture | 1 | 0.10% |
| record | hire_method | 1 | 0.10% |
| record | manager_name | 1 | 0.10% |
| record | industry | 1 | 0.10% |
| record | treatment_plan | 1 | 0.10% |
| record | cpt_code: CPT code for the medical procedure performed | 1 | 0.10% |
| record | paid_amount: Amount actually received from insurance | 1 | 0.10% |
| record | diagnosis: ICD-10 code representing the medical condition | 1 | 0.10% |
| record | billed_amount: Total amount billed for the service | 1 | 0.10% |
| record | possession_rate | 1 | 0.10% |
| record | amount: Monetary value of the return in dollars | 1 | 0.10% |
| record | product_id: Unique identifier for the product instance | 1 | 0.10% |
| record | monthly_fee: Monthly cost of the product subscription | 1 | 0.10% |
| record | offer_id | 1 | 0.10% |
| record | location: City or address where return was submitted | 1 | 0.10% |
| record | filer_name | 1 | 0.10% |
| record | test_date: Date the lab test was conducted | 1 | 0.10% |
| record | clinical_note: Clinical interpretation or patient-reported symptoms and recommendations | 1 | 0.10% |
| record | test_name: Name of the laboratory test performed | 1 | 0.10% |
| record | result_value: Measured value of the lab test | 1 | 0.10% |
| record | status: Clinical interpretation of the result | 1 | 0.10% |
| record | responsible_engineer | 1 | 0.10% |
| record | basin_name | 1 | 0.10% |
| record | discovery.location.state | 1 | 0.10% |
| record | discovery.location.county | 1 | 0.10% |
| record | discovery.location.country | 1 | 0.10% |
| record | physical_characteristics.observations.0.publication | 1 | 0.10% |
| record | post_id: Unique identifier for the TikTok post | 1 | 0.10% |
| record | status: Current status of the post based on platform action | 1 | 0.10% |
| record | shares: Total number of shares the post received | 1 | 0.10% |
| record | patient_id: Unique identifier for each patient record | 1 | 0.10% |
| record | date: Date of the imaging procedure | 1 | 0.10% |
| record | findings: Clinical observations or abnormalities detected | 1 | 0.10% |
| record | position: Job title or role within the department | 1 | 0.10% |
| record | appointment_id: Unique identifier for the appointment | 1 | 0.10% |
| record | place.location.county | 1 | 0.10% |
| record | place.name | 1 | 0.10% |
| record | range | 1 | 0.10% |
| record | test_code | 1 | 0.10% |
| record | ordering_doctor | 1 | 0.10% |
| record | collection_site | 1 | 0.10% |
| record | specimen_type | 1 | 0.10% |
| record | stockstatus | 1 | 0.10% |
| record | enzymes | 1 | 0.10% |
| record | compounds | 1 | 0.10% |
| record | diseases | 1 | 0.10% |
| record | measurements | 1 | 0.10% |
| record | proteins | 1 | 0.10% |
| record | traveler_name | 1 | 0.10% |
| record | traveler_location | 1 | 0.10% |
| record | praise | 1 | 0.10% |
| record | location.country | 1 | 0.10% |
| record | location.district | 1 | 0.10% |
| record | location.city | 1 | 0.10% |
| record | hotel.star_rating | 1 | 0.10% |
| record | booking_options.partners | 1 | 0.10% |
| record | person.experience.0.start_year | 1 | 0.10% |
| record | person.name_alternative | 1 | 0.10% |
| record | person.parliamentary_role.0.start_year | 1 | 0.10% |
| record | person.experience.0.description | 1 | 0.10% |
| record | person.experience.1.description | 1 | 0.10% |
| record | person.experience.0.end_year | 1 | 0.10% |
| record | person.publications.0.language | 1 | 0.10% |
| record | person.experience.1.role | 1 | 0.10% |
| record | person.experience.1.organization | 1 | 0.10% |
| record | person.publications.0.title | 1 | 0.10% |
| record | person.military_rank | 1 | 0.10% |
| record | person.parliamentary_role.0.chamber | 1 | 0.10% |
| record | deal_type | 1 | 0.10% |
| record | enterprise_value | 1 | 0.10% |
| record | consideration_structure | 1 | 0.10% |
| record | target_cfo | 1 | 0.10% |
| record | acquirer_ceo | 1 | 0.10% |
| record | points: Points scored by the player | 1 | 0.10% |
| record | score: Final score of the game | 1 | 0.10% |
| record | team: Name of the team playing in the game | 1 | 0.10% |
| record | assists: Assists recorded by the player | 1 | 0.10% |
| record | airline_code: Two-letter code representing the operating airline | 1 | 0.10% |
| record | flight_no: Flight number as assigned by the airline | 1 | 0.10% |
| record | unique_id: Unique identifier for the delay entry | 1 | 0.10% |
| record | delay_minutes: Number of minutes flight is delayed | 1 | 0.10% |
| record | destination: Airport code where flight ends | 1 | 0.10% |
| record | processing_code | 1 | 0.10% |
| record | product_title | 1 | 0.10% |
| record | originalordernumber | 1 | 0.10% |
| record | review.persons.1.description | 1 | 0.10% |
| record | review.performance.performers.1.name | 1 | 0.10% |
| record | review.title | 1 | 0.10% |
| record | review.performance.designer | 1 | 0.10% |
| record | review.persons.2.role | 1 | 0.10% |
| record | review.performance.venue.name | 1 | 0.10% |
| record | review.persons.0.name | 1 | 0.10% |
| record | review.persons.1.name | 1 | 0.10% |
| record | review.performance.director | 1 | 0.10% |
| record | review.summary | 1 | 0.10% |
| record | review.performance.choreographer | 1 | 0.10% |
| record | review.performance.performers.0.role | 1 | 0.10% |
| record | review.performance.performers.1.role | 1 | 0.10% |
| record | review.persons.2.name | 1 | 0.10% |
| record | review.performance.composer | 1 | 0.10% |
| record | product_description | 1 | 0.10% |
| record | return_reason | 1 | 0.10% |
| record | description: Purpose or label of the transaction | 1 | 0.10% |
| record | status: Current transaction status | 1 | 0.10% |
| record | resolution_team | 1 | 0.10% |
| record | priority | 1 | 0.10% |
| record | pricepernight | 1 | 0.10% |
| record | study_metadata.measurement_technology.anatomical_site | 1 | 0.10% |
| record | outcomes.data_source | 1 | 0.10% |
| record | population.gender | 1 | 0.10% |
| record | conclusions | 1 | 0.10% |
| record | statistical_results.predictive_model.value | 1 | 0.10% |
| record | study_metadata.study_type | 1 | 0.10% |
| record | population.age_range.max | 1 | 0.10% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.0.t_score | 1 | 0.10% |
| record | outcomes.fracture_events.0.type | 1 | 0.10% |
| record | follow_up.average_duration.months | 1 | 0.10% |
| record | statistical_results.hazard_ratios.0.value | 1 | 0.10% |
| record | population.age_range.min | 1 | 0.10% |
| record | outcomes.fracture_events.0.count | 1 | 0.10% |
| record | statistical_results.hazard_ratios.0.adjustment_factors | 1 | 0.10% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.1.rate_percentage | 1 | 0.10% |
| record | statistical_results.diagnostic_metrics.sensitivity_at_threshold.percentage_of_fractures | 1 | 0.10% |
| record | statistical_results.hazard_ratios.0.comparison | 1 | 0.10% |
| record | follow_up.average_duration.years | 1 | 0.10% |
| record | physical_characteristics.observations.0.telescope.country | 1 | 0.10% |
| record | restaurant_name | 1 | 0.10% |
| record | headlines | 1 | 0.10% |
| record | course_code | 1 | 0.10% |
| record | discovery.geologicalFormation | 1 | 0.10% |
| record | discovery.age.approximateAgeMillionYears | 1 | 0.10% |
| record | discovery.age.geologicalPeriod | 1 | 0.10% |
| record | discovery.age.geologicalStage | 1 | 0.10% |
| record | morphology.wingLengthMm | 1 | 0.10% |
| record | publication.author.affiliation.location | 1 | 0.10% |
| record | publication.otherTaxaDescribed | 1 | 0.10% |
| record | publication.genusAssignment.genus | 1 | 0.10% |
| record | publication.genusAssignment.year | 1 | 0.10% |
| record | payment | 1 | 0.10% |
| record | location.endemism | 1 | 0.10% |
| record | location.altitude.minimum | 1 | 0.10% |
| record | location.altitude.unit | 1 | 0.10% |
| record | location.communities | 1 | 0.10% |
| record | location.other_locations | 1 | 0.10% |
| record | conservation_status.status | 1 | 0.10% |
| record | conservation_status.source | 1 | 0.10% |
| record | conservation_status.threats | 1 | 0.10% |
| record | habitat.terrain | 1 | 0.10% |
| record | habitat.environmental_conditions | 1 | 0.10% |
| record | habitat.associated_species | 1 | 0.10% |
| record | description.type | 1 | 0.10% |
| record | description.leaf.texture | 1 | 0.10% |
| record | description.leaf.shape | 1 | 0.10% |
| record | description.leaf.dimensions | 1 | 0.10% |
| record | description.leaf.apex | 1 | 0.10% |
| record | description.leaf.base | 1 | 0.10% |
| record | description.stipules.dimensions | 1 | 0.10% |
| record | description.floral_buds.dimensions | 1 | 0.10% |
| record | description.fruit.dimensions | 1 | 0.10% |
| record | description.carpels | 1 | 0.10% |
| record | description.styles | 1 | 0.10% |
| record | description.seed.dimensions | 1 | 0.10% |
| record | description.seed.color | 1 | 0.10% |
| record | honorary_name.person | 1 | 0.10% |
| record | honorary_name.profession | 1 | 0.10% |
| record | honorary_name.nationality | 1 | 0.10% |
| record | honorary_name.field_of_study | 1 | 0.10% |
| record | title_english | 1 | 0.10% |
| record | recipient: Name of the person receiving the outreach | 1 | 0.10% |
| record | mechanism: Biological action or mechanism of the drug | 1 | 0.10% |
| record | drug_name: Name of the prescribed medication | 1 | 0.10% |
| record | frequency: How often the medication is taken | 1 | 0.10% |
| record | fruit_properties.skin.description | 1 | 0.10% |
| record | characteristics | 1 | 0.10% |
| record | dedication.name | 1 | 0.10% |
| record | treatment_arm | 1 | 0.10% |
| record | enrollment_date | 1 | 0.10% |
| record | contact_number | 1 | 0.10% |
| record | lastfollowup | 1 | 0.10% |
| record | insurance_provider | 1 | 0.10% |
| record | research_context.problem_statement.incompatible_substrates | 1 | 0.10% |
| record | research_context.primary_technique | 1 | 0.10% |
| record | outcomes.impact | 1 | 0.10% |
| record | outcomes.comparative_advantage.benefit_type | 1 | 0.10% |
| record | research_context.problem_statement.limitation | 1 | 0.10% |
| record | methodology.experimental_setup.reagents.0.type | 1 | 0.10% |
| record | research_context.application_goal | 1 | 0.10% |
| record | methodology.experimental_setup.biological_model.culture_environment.material_properties | 1 | 0.10% |
| record | outcomes.comparative_advantage.alternative_to | 1 | 0.10% |
| record | methodology.experimental_setup.instrumentation | 1 | 0.10% |
| record | hazard_class: Primary hazard classification of the chemical | 1 | 0.10% |
| record | manufacturer: Company responsible for manufacturing the chemical | 1 | 0.10% |
| record | regulatory_compliance: List of relevant regulatory standards and codes | 1 | 0.10% |
| record | storage_conditions: Environmental and physical conditions required for safe storage | 1 | 0.10% |
| record | donor_name: Full name of the individual or organization making the donation | 1 | 0.10% |
| record | donor_address: Physical address of the donor | 1 | 0.10% |
| record | donation_description: Brief description of the donation's purpose or context | 1 | 0.10% |
| record | donation_id: Unique identifier for the donation in the batch | 1 | 0.10% |
| record | research_interests | 1 | 0.10% |
| record | affected_systems | 1 | 0.10% |
| record | urls | 1 | 0.10% |
| record | solution | 1 | 0.10% |
| record | address: Full residential or commercial address of the customer | 1 | 0.10% |
| record | impact: Description of service impact including data loss or business effects | 1 | 0.10% |
| record | handling_instructions | 1 | 0.10% |
| record | transfer_id | 1 | 0.10% |
| record | description.fruit.color | 1 | 0.10% |
| record | description.fruit.shape | 1 | 0.10% |
| record | description.inflorescence.height_m.max | 1 | 0.10% |
| record | description.flowers.perianth_tube_length_mm.max | 1 | 0.10% |
| record | systematics_and_distribution.habitat | 1 | 0.10% |
| record | description.leaves.length_cm.max | 1 | 0.10% |
| record | systematics_and_distribution.group | 1 | 0.10% |
| record | description.leaves.shape | 1 | 0.10% |
| record | team_mentions.0.name | 1 | 0.10% |
| record | record_id | 1 | 0.10% |
| record | winner: Player who won the match | 1 | 0.10% |
| record | match_round: Stage of the tournament such as final, quarterfinal, or last 16 | 1 | 0.10% |
| record | filing_id | 1 | 0.10% |
| record | filed_by | 1 | 0.10% |
| record | address: Full address of the policyholder's residence | 1 | 0.10% |
| record | description: Summary of incident cause and policy coverage rationale | 1 | 0.10% |
| record | policyholder: Name of the policyholder filing the claim | 1 | 0.10% |
| record | rental_id | 1 | 0.10% |
| record | game series | 1 | 0.10% |
| record | track set | 1 | 0.10% |
| record | game mode | 1 | 0.10% |
| record | player id | 1 | 0.10% |
| record | track name | 1 | 0.10% |
| record | githubrepourl | 1 | 0.10% |
| record | linkedinposturl | 1 | 0.10% |
| record | details.days_ago | 1 | 0.10% |
| record | term | 1 | 0.10% |
| record | clinic | 1 | 0.10% |
| record | encounter_id | 1 | 0.10% |
| record | submitter | 1 | 0.10% |
| record | outbreak_event.investigation.specimen_source | 1 | 0.10% |
| record | outbreak_event.illness.pathogen.full_name | 1 | 0.10% |
| record | outbreak_event.location.country | 1 | 0.10% |
| record | outbreak_event.public_health_recommendations.target_populations | 1 | 0.10% |
| record | outbreak_event.investigation.findings.agent_details.strain_type | 1 | 0.10% |
| record | outbreak_event.illness.pathogen.abbreviation | 1 | 0.10% |
| record | outbreak_event.illness.description | 1 | 0.10% |
| record | outbreak_event.investigation.findings.agent_details.full_name | 1 | 0.10% |
| record | outbreak_event.illness.pathogen.classification | 1 | 0.10% |
| record | date: Date the match was scheduled or played | 1 | 0.10% |
| record | score: List of two integers representing home and away scores | 1 | 0.10% |
| record | team1: First team participating in the match | 1 | 0.10% |
| record | location: Venue where the match took place | 1 | 0.10% |
| record | force_type | 1 | 0.10% |
| record | features: List of included amenities or space features | 1 | 0.10% |
| record | location: Full address of the leased property | 1 | 0.10% |
| record | examination_date | 1 | 0.10% |
| record | finding_summary | 1 | 0.10% |
| record | modality | 1 | 0.10% |
| record | radiology_code | 1 | 0.10% |
| record | party_b | 1 | 0.10% |
| record | user: Name of the customer who provided the review | 1 | 0.10% |
| record | product_id: Unique identifier for the product reviewed | 1 | 0.10% |
| record | purchase_date: Date when the product was purchased | 1 | 0.10% |
| record | rating: Numerical rating from 1 to 5 given by the user | 1 | 0.10% |
| record | price: Formatted price including dollar sign and cents | 1 | 0.10% |
| record | location: City and state or province where purchase occurred | 1 | 0.10% |
| record | review: Text content of the customer's feedback on the product | 1 | 0.10% |
| record | retweets | 1 | 0.10% |
| record | post_date | 1 | 0.10% |
| record | location_tag | 1 | 0.10% |
| record | device_id: Unique identifier for the medical device as listed in records | 1 | 0.10% |
| record | allergies | 1 | 0.10% |
| record | lead_person | 1 | 0.10% |
| record | legal_counsel | 1 | 0.10% |
| record | sender: Email of the person sending the confirmation or submission | 1 | 0.10% |
| record | recipient: Email of the person receiving the confirmation or submission | 1 | 0.10% |
| record | geologic_region | 1 | 0.10% |
| record | primary_outcome | 1 | 0.10% |
| record | key_result | 1 | 0.10% |
| record | affiliations | 1 | 0.10% |
| record | study_start | 1 | 0.10% |
| record | seller_location: Full address of the seller including city, state, and zip code | 1 | 0.10% |
| record | unit_price: Price per unit of the product | 1 | 0.10% |
| record | product_id: Unique identifier for each product in the best-sellers list | 1 | 0.10% |
| record | seller: Company name that sells the product | 1 | 0.10% |
| record | affiliation | 1 | 0.10% |
| record | sales_volume | 1 | 0.10% |
| record | date_sold | 1 | 0.10% |
| record | event_code: Unique identifier for the event as listed in records | 1 | 0.10% |
| record | discipline: Type of sport or competition such as swimming or skiing | 1 | 0.10% |
| record | year: Year the event was held | 1 | 0.10% |
| record | country: Nationality of the athlete who won the medal | 1 | 0.10% |
| record | athlete_name: Name of the athlete who achieved the result | 1 | 0.10% |
| record | age: Age of the athlete at the time of the event | 1 | 0.10% |
| record | result: Medal awarded to the athlete | 1 | 0.10% |
| record | visit_count | 1 | 0.10% |
| record | status: Current status of the claim approval | 1 | 0.10% |
| record | claim_date: Date the claim was filed | 1 | 0.10% |
| record | policyholder_id: Unique identifier for the policyholder | 1 | 0.10% |
| record | disputingpartya | 1 | 0.10% |
| record | disputingpartyb | 1 | 0.10% |
| record | mark_name | 1 | 0.10% |
| record | registration_number | 1 | 0.10% |
| record | affiliation.0.type | 1 | 0.10% |
| record | recipient_name | 1 | 0.10% |
| record | po_reference | 1 | 0.10% |
| record | funding_id: Unique identifier for each funding round | 1 | 0.10% |
| record | status: Current status of the funding round | 1 | 0.10% |
| record | location: City and state where the company is headquartered | 1 | 0.10% |
| record | amount: Total funding amount in USD with currency symbol | 1 | 0.10% |
| record | company_name: Name of the startup receiving funding | 1 | 0.10% |
| record | management_contact: Name or entity managing the property | 1 | 0.10% |
| record | city: City where the property is located | 1 | 0.10% |
| record | rentalboardstatus: Status as reported on the city’s rental board | 1 | 0.10% |
| record | property_id: Unique identifier assigned to the rental unit | 1 | 0.10% |
| record | department: Department where the employee is assigned | 1 | 0.10% |
| record | causeofloss | 1 | 0.10% |
| record | damage_amount | 1 | 0.10% |
| record | town.location | 1 | 0.10% |
| record | practice_areas | 1 | 0.10% |
| record | states_served | 1 | 0.10% |
| record | additional_services | 1 | 0.10% |
| record | experience.description | 1 | 0.10% |
| record | zebras_care_foundation.beneficiary.name | 1 | 0.10% |
| record | nhl_officials | 1 | 0.10% |
| record | zebras_care_foundation.beneficiary.age | 1 | 0.10% |
| record | zebras_care_foundation.introducer.affiliation | 1 | 0.10% |
| record | zebras_care_foundation.beneficiary.treatment | 1 | 0.10% |
| record | zebras_care_foundation.beneficiary.condition | 1 | 0.10% |
| record | invoice_date | 1 | 0.10% |
| record | month | 1 | 0.10% |
| record | vendor | 1 | 0.10% |
| record | general_admission | 1 | 0.10% |
| record | vip_price | 1 | 0.10% |
| record | genre_focus | 1 | 0.10% |
| record | storm_id | 1 | 0.10% |
| record | resolution_status | 1 | 0.10% |
| record | discount.percentage | 1 | 0.10% |
| record | discount.description | 1 | 0.10% |
| record | location.region | 1 | 0.10% |
| record | location.hotel_count | 1 | 0.10% |
| record | booking.promotional_code | 1 | 0.10% |
| record | booking.validity | 1 | 0.10% |
| record | booking.cancellation_policy | 1 | 0.10% |
| record | terms_and_conditions.availability | 1 | 0.10% |
| record | terms_and_conditions.group_booking_restriction | 1 | 0.10% |
| record | terms_and_conditions.room_upgrades | 1 | 0.10% |
| record | terms_and_conditions.combination_restriction | 1 | 0.10% |
| record | terms_and_conditions.disclaimer | 1 | 0.10% |
| record | hotel_chain | 1 | 0.10% |
| record | rates.rate_type | 1 | 0.10% |
| record | loyalty_program.name | 1 | 0.10% |
| record | loyalty_program.membership_url | 1 | 0.10% |
| record | booking_channels | 1 | 0.10% |
| record | question.legal_implications | 1 | 0.10% |
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
| record | icd_code | 1 | 0.10% |
| record | respiratory_rate: Number of breaths per minute | 1 | 0.10% |
| record | oxygen_saturation: Percentage of oxygen saturation in blood | 1 | 0.10% |
| record | device_type: Type of monitoring device such as wearable, bedside, or oximeter | 1 | 0.10% |
| record | monitoring_time: Exact time of monitoring in 24-hour format | 1 | 0.10% |
| record | patient_name: Full name of the individual being monitored | 1 | 0.10% |
| record | reason_for_status | 1 | 0.10% |
| record | creative work | 1 | 0.10% |
| record | assigned_staff: Healthcare professional assigned to manage the device | 1 | 0.10% |
| record | purchase_date: Date the device was acquired | 1 | 0.10% |
| record | serial_number: Unique device identifier assigned by manufacturer | 1 | 0.10% |
| record | expiration_date: Date after which device is no longer valid | 1 | 0.10% |
| record | location: Physical ward or room where device is located | 1 | 0.10% |
| record | form | 1 | 0.10% |
| record | warranty | 1 | 0.10% |
| record | personal_sustainability | 1 | 0.10% |
| record | industry_example.status | 1 | 0.10% |
| record | crisis_resilience | 1 | 0.10% |
| record | definition.source_organization.full_name | 1 | 0.10% |
| record | definition.text | 1 | 0.10% |
| record | industry_example.actions | 1 | 0.10% |
| record | industry_example.challenges | 1 | 0.10% |
| record | industry_example.industry | 1 | 0.10% |
| record | benefits | 1 | 0.10% |
| record | definition.document_title | 1 | 0.10% |
| record | financial_aspect.relevance_in_crisis | 1 | 0.10% |
| record | report_metadata.publication_date | 1 | 0.10% |
| record | report_metadata.source_agency | 1 | 0.10% |
| record | report_metadata.source_url | 1 | 0.10% |
| record | company.financial_results.reporting_period | 1 | 0.10% |
| record | company.financial_results.currency | 1 | 0.10% |
| record | company.financial_results.metrics.0.metric_name | 1 | 0.10% |
| record | company.financial_results.metrics.0.historical_comparison.comparison_period | 1 | 0.10% |
| record | revenue: Total revenue in billions as reported | 1 | 0.10% |
| record | eps: Earnings per share in dollars | 1 | 0.10% |
| record | growth_rate: Year-over-year growth percentage | 1 | 0.10% |
| record | market_cap: Market capitalization in billions | 1 | 0.10% |
| record | contact_phone | 1 | 0.10% |
| record | contact_person | 1 | 0.10% |
| record | paymentduedate | 1 | 0.10% |
| record | account_code | 1 | 0.10% |
| record | thumbnail_style | 1 | 0.10% |
| record | content_type | 1 | 0.10% |
| record | views | 1 | 0.10% |
| record | handle | 1 | 0.10% |
| record | content.elements.name | 1 | 0.10% |
| record | content.core_beliefs.relationship_with_society | 1 | 0.10% |
| record | title: Job title as listed in the requisition | 1 | 0.10% |
| record | visit_type | 1 | 0.10% |
| record | price: Listed sale price in dollars | 1 | 0.10% |
| record | agent: Name of the real estate agent representing the seller | 1 | 0.10% |
| record | bedrooms: Number of bedrooms in the property | 1 | 0.10% |
| record | players | 1 | 0.10% |
| record | discovery.observatory.location.city | 1 | 0.10% |
| record | physical_characteristics.albedo | 1 | 0.10% |
| record | orbital_characteristics.eccentricity | 1 | 0.10% |
| record | diameter | 1 | 0.10% |
| record | environmental_rating | 1 | 0.10% |
| record | company: Company owning or offering the product | 1 | 0.10% |
| record | target_clients: Client segment the product serves | 1 | 0.10% |
| record | launch_year: Year of product launch | 1 | 0.10% |
| record | account | 1 | 0.10% |
| record | impact_homes | 1 | 0.10% |
| record | rainfall | 1 | 0.10% |
| record | response_status | 1 | 0.10% |
| record | maintenance_notes | 1 | 0.10% |
| record | course_name | 1 | 0.10% |
| record | production_location | 1 | 0.10% |
| record | test_cases_completed | 1 | 0.10% |
| record | test_cases_remaining | 1 | 0.10% |
| record | daily_value_percentage | 1 | 0.10% |
| record | client_id | 1 | 0.10% |
| record | medication_name | 1 | 0.10% |
| record | refill_count | 1 | 0.10% |
| record | athlete: Name of the athlete who won the medal | 1 | 0.10% |
| record | nation: Country of the athlete who won the medal | 1 | 0.10% |
| record | venue: Name of the stadium or venue where the event occurred | 1 | 0.10% |
| record | time: Official race time recorded in the event | 1 | 0.10% |
| record | medal: Type of medal won gold, silver, or bronze | 1 | 0.10% |
| record | defendant_address | 1 | 0.10% |
| record | tournament_code: Unique identifier for the match event | 1 | 0.10% |
| record | pricing_model | 1 | 0.10% |
| record | price_change | 1 | 0.10% |
| record | closing_price | 1 | 0.10% |
| record | volume | 1 | 0.10% |
| record | reason | 1 | 0.10% |
| record | definitions | 1 | 0.10% |
| record | journals | 1 | 0.10% |
| record | causes_for_inedible_sprouts | 1 | 0.10% |
| record | sprouting_definition | 1 | 0.10% |
| record | drug_name: Name of the pharmaceutical product | 1 | 0.10% |
| record | manufacturer: Company that produces the drug | 1 | 0.10% |
| record | route: Administration method of the drug | 1 | 0.10% |
| record | target_receptor: Biological receptor the drug acts on | 1 | 0.10% |
| record | mechanismofaction: Biological process inhibited or modulated by the drug | 1 | 0.10% |
| record | molecular_weight: Molecular weight of the drug in daltons | 1 | 0.10% |
| record | clinical_phase: Current stage of clinical development | 1 | 0.10% |
| record | efficacy_rate: Percentage of patients showing therapeutic response | 1 | 0.10% |
| record | adverse_effects: List of side effects reported in clinical use | 1 | 0.10% |
| record | lastreviewdate: Date the drug record was last reviewed | 1 | 0.10% |
| record | title: Topic or title of the submitted work | 1 | 0.10% |
| record | decision_status: Final outcome of the submission | 1 | 0.10% |
| record | model_name: Name of the product as marketed to consumers | 1 | 0.10% |
| record | warranty_period: Duration of warranty in years or terms | 1 | 0.10% |
| record | donor | 1 | 0.10% |
| record | contribution_type | 1 | 0.10% |
| record | foods_mentioned | 1 | 0.10% |
| record | submitter: Name of employee or partner who reported the incident | 1 | 0.10% |
| record | playertwoid | 1 | 0.10% |
| record | match_type | 1 | 0.10% |
| record | playeronename | 1 | 0.10% |
| record | match_result | 1 | 0.10% |
| record | drug_name: Name of the pharmaceutical compound | 1 | 0.10% |
| record | molecular_target: Biological molecule the drug interacts with | 1 | 0.10% |
| record | pathway: Biological pathway affected by the drug | 1 | 0.10% |
| record | clinical_indication: Medical condition the drug is used to treat | 1 | 0.10% |
| record | status: Current regulatory or development status | 1 | 0.10% |
| record | dose: Dosage administered to the patient | 1 | 0.10% |
| record | date_verified: Date the drug's information was last verified | 1 | 0.10% |
| record | mechanismofaction: Description of how the drug exerts its effect | 1 | 0.10% |
| record | clinical_use: Primary medical application of the drug | 1 | 0.10% |
| record | place.location.region | 1 | 0.10% |
| record | place.location.department | 1 | 0.10% |
| record | place.location.province | 1 | 0.10% |
| record | place.location.district | 1 | 0.10% |
| record | place.type | 1 | 0.10% |
| record | genus.etymology.part2.meaning | 1 | 0.10% |
| record | genus.publication | 1 | 0.10% |
| record | genus.number_of_species.min | 1 | 0.10% |
| record | genus.established_by | 1 | 0.10% |
| record | morphology.stem | 1 | 0.10% |
| record | morphology.flowering_fruiting_period | 1 | 0.10% |
| record | distribution.habitat | 1 | 0.10% |
| record | morphology.seeds | 1 | 0.10% |
| record | morphology.flower_heads | 1 | 0.10% |
| record | distribution.regions.0.region_name | 1 | 0.10% |
| record | morphology.leaves | 1 | 0.10% |
| record | distribution.regions.1.region_name | 1 | 0.10% |
| record | material: Primary material used in construction | 1 | 0.10% |
| record | sku: Unique stock keeping unit identifier for the product | 1 | 0.10% |
| record | historical_events | 1 | 0.10% |
| record | revenue_uptick | 1 | 0.10% |
| record | currency: Currency pair such as JPY, EUR, CAD | 1 | 0.10% |
| record | office: Regional office that generated the snapshot | 1 | 0.10% |
| record | sunless_tanning.overview | 1 | 0.10% |
| record | sunless_tanning.concerns | 1 | 0.10% |
| record | sunless_tanning.product_collection.brand | 1 | 0.10% |
| record | sunless_tanning.product_collection.products.0.name | 1 | 0.10% |
| record | sunless_tanning.product_collection.products.0.sizes | 1 | 0.10% |
| record | sunless_tanning.product_collection.products.0.benefits | 1 | 0.10% |
| record | sunless_tanning.product_collection.products.1.type | 1 | 0.10% |
| record | risks_of_sun_exposure | 1 | 0.10% |
| record | publication: Name of the media outlet publishing the review | 1 | 0.10% |
| record | department: Academic department offering the course | 1 | 0.10% |
| record | location: Physical classroom location including building and room | 1 | 0.10% |
| record | instructor: Name of the faculty member teaching the course | 1 | 0.10% |
| record | status: Current operational status of the course | 1 | 0.10% |
| record | credit_hours: Number of credit hours for the course | 1 | 0.10% |
| record | certification.exams.0.name | 1 | 0.10% |
| record | certification.description | 1 | 0.10% |
| record | training.providers | 1 | 0.10% |
| record | certification.exams.1.code | 1 | 0.10% |
| record | certification.prerequisites | 1 | 0.10% |
| record | personnel.0.title | 1 | 0.10% |
| record | certification.exams.0.code | 1 | 0.10% |
| record | organization.website | 1 | 0.10% |
| record | personnel.0.name | 1 | 0.10% |
| record | review_journal: Name of the publication that reviewed the work | 1 | 0.10% |
| record | author: Name of the author or poet | 1 | 0.10% |
| record | location: Geographic location of the reviewer | 1 | 0.10% |
| record | publisher: Publishing house responsible for the work | 1 | 0.10% |
| record | review_id: Unique identifier assigned to the review | 1 | 0.10% |
| record | event_date: Date when the event will take place | 1 | 0.10% |
| record | organizer_name: Name of the person or group organizing the event | 1 | 0.10% |
| record | city: City where the event is held | 1 | 0.10% |
| record | location: Full address including street, city, and state | 1 | 0.10% |
| record | event_code: Unique code identifying the event | 1 | 0.10% |
| record | systematics_and_distribution.location.elevation_m.max | 1 | 0.10% |
| record | description.flowers.length_mm.min | 1 | 0.10% |
| record | description.rosette | 1 | 0.10% |
| record | systematics_and_distribution.distinguishing_features | 1 | 0.10% |
| record | description.flowers.perianth_tube_length_mm.min | 1 | 0.10% |
| record | initial_text.actions | 1 | 0.10% |
| record | search_result_page.content_types | 1 | 0.10% |
| record | search_result_page.related_to | 1 | 0.10% |
| record | event_code | 1 | 0.10% |
| record | discipline | 1 | 0.10% |
| record | patient_name: Full name of the patient undergoing the test | 1 | 0.10% |
| record | patient_id: Unique identifier assigned to the patient | 1 | 0.10% |
| record | test_code: Code identifying the specific laboratory test | 1 | 0.10% |
| record | test_date: Date when the test was performed | 1 | 0.10% |
| record | result_value: Numerical value of the test result | 1 | 0.10% |
| record | units: Unit of measurement for the test result | 1 | 0.10% |
| record | status: Current status of the test result | 1 | 0.10% |
| record | location: Physical location of the testing facility | 1 | 0.10% |
| record | ship_date | 1 | 0.10% |
| record | product_model | 1 | 0.10% |
| record | serial_number | 1 | 0.10% |
| record | return_date | 1 | 0.10% |
| record | rma_number | 1 | 0.10% |
| record | failure_reason | 1 | 0.10% |
| record | century | 1 | 0.10% |
| record | images | 1 | 0.10% |
| record | required_experience | 1 | 0.10% |
| record | away_score: Final score of the away team | 1 | 0.10% |
| record | hotel_experience.location.nearby_landmark | 1 | 0.10% |
| record | hotel_experience.location.city | 1 | 0.10% |
| record | hotel_experience.entertainment.tv.content | 1 | 0.10% |
| record | hotel_experience.additional_benefits | 1 | 0.10% |
| record | hotel_experience.suites.0.workspace | 1 | 0.10% |
| record | hotel_experience.suites.0.suite_type | 1 | 0.10% |
| record | hotel_experience.suites.0.bathroom_features | 1 | 0.10% |
| record | hotel_experience.suites.0.amenities | 1 | 0.10% |
| record | hotel_experience.suites.0.features | 1 | 0.10% |
| record | hotel_experience.entertainment.tv.screen_size | 1 | 0.10% |
| record | patent_number: Unique identifier assigned by USPTO to the patent | 1 | 0.10% |
| record | applicant: Organization or entity that filed the patent | 1 | 0.10% |
| record | invention: Technical description of the patented invention | 1 | 0.10% |
| record | ipc_code: International Patent Classification code for the invention | 1 | 0.10% |
| record | claim_count: Number of claims filed in the patent application | 1 | 0.10% |
| record | score: Rating given on a scale from 0 to 5 | 1 | 0.10% |
| record | chemical_name | 1 | 0.10% |
| record | hazard_class | 1 | 0.10% |
| record | handling_notes | 1 | 0.10% |
| record | meeting_code | 1 | 0.10% |
| record | properties | 1 | 0.10% |
| record | asteroid.named_after.description | 1 | 0.10% |
| record | asteroid.named_after.birth_location.country | 1 | 0.10% |
| record | asteroid.alternate_name | 1 | 0.10% |
| record | asteroid.discovery.discoverer.affiliation.observatory | 1 | 0.10% |
| record | score: Numerical performance score from 3.0 to 4.1 | 1 | 0.10% |
| record | status: Current status of the review document | 1 | 0.10% |
| record | plaintiff | 1 | 0.10% |
| record | counsel | 1 | 0.10% |
| record | judge | 1 | 0.10% |
| record | case_type | 1 | 0.10% |
| record | legislative_details.referred_committee | 1 | 0.10% |
| record | legislative_details.law_number | 1 | 0.10% |
| record | speaker | 1 | 0.10% |
| record | end_time | 1 | 0.10% |
| record | streaming_platform | 1 | 0.10% |
| record | start_time | 1 | 0.10% |
| record | paymentduedays: Number of days until payment is due after notice | 1 | 0.10% |
| record | renewal_amount: Annual premium amount for the renewal | 1 | 0.10% |
| record | company_id: Unique identifier assigned to each company | 1 | 0.10% |
| record | status: Current operational or strategic status | 1 | 0.10% |
| record | headquarters: City and country where company is headquartered | 1 | 0.10% |
| record | listing_date | 1 | 0.10% |
| record | office_code | 1 | 0.10% |
| record | functional_area | 1 | 0.10% |
| record | software | 1 | 0.10% |
| record | sound_quality | 1 | 0.10% |
| record | audiofidelityscore | 1 | 0.10% |
| record | buyer_name | 1 | 0.10% |
| record | usability_score | 1 | 0.10% |
| record | score_home | 1 | 0.10% |
| record | score_away | 1 | 0.10% |
| record | points_leader | 1 | 0.10% |
| record | rebounds_leader | 1 | 0.10% |
| record | assists_leader | 1 | 0.10% |
| record | media_analysis.events.0.national_media_response | 1 | 0.10% |
| record | media_analysis.criticisms | 1 | 0.10% |
| record | narrative.setting.time | 1 | 0.10% |
| record | media_analysis.positive_aspects | 1 | 0.10% |
| record | narrative.protagonist.background | 1 | 0.10% |
| record | narrative.protagonist.possessions | 1 | 0.10% |
| record | media_analysis.events.0.description | 1 | 0.10% |
| record | narrative.setting.description | 1 | 0.10% |
| record | media_analysis.historical_context.reference | 1 | 0.10% |
| record | renewal_amount: Monetary value of the renewal premium | 1 | 0.10% |
| record | coverage_details: Description of coverage enhancements or changes | 1 | 0.10% |
| record | contributor | 1 | 0.10% |
| record | hotel.distance_from_city_center.value | 1 | 0.10% |
| record | hotel.number_of_rooms | 1 | 0.10% |
| record | hotel.distance_from_city_center.unit | 1 | 0.10% |
| record | hotel.distance_from_airport.value | 1 | 0.10% |
| record | hotel.nearby_attractions | 1 | 0.10% |
| record | hotel.location.region | 1 | 0.10% |
| record | tv_program.setting.time_period | 1 | 0.10% |
| record | affected_period | 1 | 0.10% |
| record | diagnosis_codes | 1 | 0.10% |
| record | admitting_department | 1 | 0.10% |
| record | conference | 1 | 0.10% |
| record | pages | 1 | 0.10% |
| record | ticket_id: Unique identifier for each incident in the system | 1 | 0.10% |
| record | root_cause: Technical reason for the service disruption | 1 | 0.10% |
| record | github_url | 1 | 0.10% |
| record | new_website_url | 1 | 0.10% |
| record | replies | 1 | 0.10% |
| record | major | 1 | 0.10% |
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
| record | website | 1 | 0.10% |
| record | state: State where the property is located | 1 | 0.10% |
| record | beds: Number of bedrooms | 1 | 0.10% |
| record | discrepancy_amount | 1 | 0.10% |
| record | resolution_summary | 1 | 0.10% |
| record | location: Geographic location of the project | 1 | 0.10% |
| record | funding: Total funding amount in millions | 1 | 0.10% |
| record | visit_type: Type of medical visit conducted | 1 | 0.10% |
| record | report_date | 1 | 0.10% |
| record | checklist_id | 1 | 0.10% |
| record | medical_code | 1 | 0.10% |
| record | policy_type | 1 | 0.10% |
| record | policy_renewal_notice | 1 | 0.10% |
| record | policy_holder | 1 | 0.10% |
| record | patient_id: Unique identifier for each patient | 1 | 0.10% |
| record | onset_date: Date when symptoms first began | 1 | 0.10% |
| record | address_street: Street address line of the patient | 1 | 0.10% |
| record | symptom_location: Body region where symptoms are felt | 1 | 0.10% |
| record | status: Current clinical status of the patient | 1 | 0.10% |
| record | company.description | 1 | 0.10% |
| record | company.location | 1 | 0.10% |
| record | company.establishment_date | 1 | 0.10% |
| record | company.original_name | 1 | 0.10% |
| record | company.founding_location.city | 1 | 0.10% |
| record | company.founding_location.region | 1 | 0.10% |
| record | company.transformation_date | 1 | 0.10% |
| record | company.founders.0.name | 1 | 0.10% |
| record | company.founders.1.name | 1 | 0.10% |
| record | investors | 1 | 0.10% |
| record | funding_id | 1 | 0.10% |
| record | context.problem | 1 | 0.10% |
| record | external_links.0.title | 1 | 0.10% |
| record | classification.source | 1 | 0.10% |
| record | observations.observation_period.years | 1 | 0.10% |
| record | wiki_sky_links | 1 | 0.10% |
| record | alternative_designations | 1 | 0.10% |
| record | discovery.location.observatory | 1 | 0.10% |
| record | classification.type | 1 | 0.10% |
| record | orbital_characteristics.orbital_period.years | 1 | 0.10% |
| record | discovery.location.place | 1 | 0.10% |
| record | provider_name: Name of the healthcare provider | 1 | 0.10% |
| record | patient_name: Full name of the patient receiving treatment | 1 | 0.10% |
| record | treatment_type: Type of medical treatment provided | 1 | 0.10% |
| record | address: Full address of the provider including street city state zip | 1 | 0.10% |
| record | claim_id: Unique identifier for the claim record | 1 | 0.10% |
| record | diagnosis_code: Medical code indicating the patient's condition | 1 | 0.10% |
| record | labels | 1 | 0.10% |
| record | on_site_registration.documents_required | 1 | 0.10% |
| record | on_site_registration.location.address | 1 | 0.10% |
| record | priority_registration.instructions | 1 | 0.10% |
| record | on_site_registration.location.name | 1 | 0.10% |
| record | on_site_registration.requirements | 1 | 0.10% |
| record | on_site_registration.description | 1 | 0.10% |
| record | on_site_registration.note | 1 | 0.10% |
| record | online_priority_registration.deadline | 1 | 0.10% |
| record | audience_rating | 1 | 0.10% |
| record | age_restriction | 1 | 0.10% |
| record | festival_id | 1 | 0.10% |
| record | performance_time | 1 | 0.10% |
| record | ticket_price | 1 | 0.10% |
| record | geographic_region | 1 | 0.10% |
| record | connectivity | 1 | 0.10% |
| record | location: Venue city and country where the final occurred | 1 | 0.10% |
| record | tournament_name: Name of the tennis tournament such as Monte Carlo Masters | 1 | 0.10% |
| record | score: Match score in format 6-3, 4-6, 6-4 | 1 | 0.10% |
| record | revenue: Total gate revenue in millions | 1 | 0.10% |
| record | opponent: Name of the player who lost in the final | 1 | 0.10% |
| record | outcome | 1 | 0.10% |
| record | location_neighborhood | 1 | 0.10% |
| record | bed_number | 1 | 0.10% |
| record | key_enzyme | 1 | 0.10% |
| record | discovery_date | 1 | 0.10% |
| record | required_materials | 1 | 0.10% |
| record | presenter | 1 | 0.10% |
| record | attendance_limit | 1 | 0.10% |
| record | techtoolsused | 1 | 0.10% |
| record | record_label | 1 | 0.10% |
| record | sales_growth | 1 | 0.10% |
| record | audience_age_range | 1 | 0.10% |
| record | discount_amount | 1 | 0.10% |
| record | product_list_id | 1 | 0.10% |
| record | product_count | 1 | 0.10% |
| record | promotional_campaign | 1 | 0.10% |
| record | promotion_channels | 1 | 0.10% |
| record | discount_condition | 1 | 0.10% |
| record | video_id | 1 | 0.10% |
| record | domain.registry.name | 1 | 0.10% |
| record | domain.territory | 1 | 0.10% |
| record | domain.third_level_domains | 1 | 0.10% |
| record | domain.dispute_resolution.methods | 1 | 0.10% |
| record | vaccine | 1 | 0.10% |
| record | investor: Name of the investment firm conducting the meeting | 1 | 0.10% |
| record | startup: Name of the startup receiving investment | 1 | 0.10% |
| record | amount: Investment amount in millions of dollars | 1 | 0.10% |
| record | meeting_code: Unique code assigned to the briefing session | 1 | 0.10% |
| record | enrollment_code | 1 | 0.10% |
| record | -LRB- | 1 | 0.10% |
| record | . | 1 | 0.10% |
| record | NN | 1 | 0.10% |
| record | IN | 1 | 0.10% |
| record | DT | 1 | 0.10% |
| record | VBD | 1 | 0.10% |
| record | CC | 1 | 0.10% |
| record | amount: Monetary value of the transaction | 1 | 0.10% |
| record | description: Brief explanation of the transaction purpose | 1 | 0.10% |
| record | participant: Name of the individual involved in the transaction | 1 | 0.10% |
| record | account: Bank account associated with the transaction | 1 | 0.10% |
| record | filing_fee | 1 | 0.10% |
| record | officeactionnumber | 1 | 0.10% |
| record | asteroid.orbital_characteristics.orbital_region | 1 | 0.10% |
| record | activities | 1 | 0.10% |
| record | tv_program.year | 1 | 0.10% |
| record | competition_year | 1 | 0.10% |
| record | medal | 1 | 0.10% |
| record | product.specifications.display.resolution | 1 | 0.10% |
| record | product.specifications.display.size | 1 | 0.10% |
| record | product.availability.future_availability | 1 | 0.10% |
| record | product.launch_location | 1 | 0.10% |
| record | product.specifications.camera.features | 1 | 0.10% |
| record | product.registration_count.count | 1 | 0.10% |
| record | product.availability.currently_available | 1 | 0.10% |
| record | product.specifications.processor | 1 | 0.10% |
| record | product.model | 1 | 0.10% |
| record | product.specifications.camera.front | 1 | 0.10% |
| record | product.specifications.camera.rear | 1 | 0.10% |
| record | product.specifications.battery.capacity | 1 | 0.10% |
| record | product.specifications.memory.ram | 1 | 0.10% |
| record | institution: Hospital or organization where author is affiliated | 1 | 0.10% |
| record | recommendation: Specific treatment or intervention advised | 1 | 0.10% |
| record | issue_year: Year the guideline was published or released | 1 | 0.10% |
| record | location: City or region where vaccination occurred | 1 | 0.10% |
| record | dose_number: Sequence of the dose in the series | 1 | 0.10% |
| record | role: Title of the job position being advertised | 1 | 0.10% |
| record | location: City and country where the role is based | 1 | 0.10% |
| record | company: Name of the company issuing the job posting | 1 | 0.10% |
| record | trend_repost_id | 1 | 0.10% |
| record | creator | 1 | 0.10% |
| record | date_posted | 1 | 0.10% |
| record | asteroid_number | 1 | 0.10% |
| record | asteroid_type | 1 | 0.10% |
| record | provisional_designation | 1 | 0.10% |
| record | physical_characteristics.observations.0.wavelength | 1 | 0.10% |
| record | asteroid_designation | 1 | 0.10% |
| record | funding_id: Unique identifier assigned to the funding round | 1 | 0.10% |
| record | company: Name of the technology company receiving funds | 1 | 0.10% |
| record | device_id: Unique identifier assigned to the medical device | 1 | 0.10% |
| record | serial_number: Unique serial number for device tracking | 1 | 0.10% |
| record | usage_count: Total patient admissions or scans performed | 1 | 0.10% |
| record | prescriber_id | 1 | 0.10% |
| record | status: Current availability status of the publication | 1 | 0.10% |
| record | language: List of languages the publication is available in | 1 | 0.10% |
| record | page: Page number where the digest appears | 1 | 0.10% |
| record | author: Name of the author who wrote the digest | 1 | 0.10% |
| record | id: Unique identifier for the publication record | 1 | 0.10% |
| record | game_status | 1 | 0.10% |
| record | news_id | 1 | 0.10% |
| record | regionNameBerber | 1 | 0.10% |
| record | regionName | 1 | 0.10% |
| record | infrastructure.roads.0.connections | 1 | 0.10% |
| record | infrastructure.refineries.0.location | 1 | 0.10% |
| record | economy.gdpCurrency | 1 | 0.10% |
| record | infrastructure.refineries.0.status | 1 | 0.10% |
| record | geography.location | 1 | 0.10% |
| record | geography.borders.1.region | 1 | 0.10% |
| record | economy.gdpValue | 1 | 0.10% |
| record | geography.borders.2.direction | 1 | 0.10% |
| record | infrastructure.refineries.0.closureDate | 1 | 0.10% |
| record | populationYear | 1 | 0.10% |
| record | lowerLevelDivisions | 1 | 0.10% |
| record | geography.borders.1.direction | 1 | 0.10% |
| record | economy.economicBase | 1 | 0.10% |
| record | geography.borders.2.region | 1 | 0.10% |
| record | geography.borders.0.region | 1 | 0.10% |
| record | infrastructure.airports.0.trafficVolume | 1 | 0.10% |
| record | geography.borders.0.direction | 1 | 0.10% |
| record | infrastructure.roads.0.roadName | 1 | 0.10% |
| record | geography.rivers | 1 | 0.10% |
| record | album.chart_performance.chart_type | 1 | 0.10% |
| record | album.tracklist.dvd.0.title | 1 | 0.10% |
| record | album.versions.3.contents | 1 | 0.10% |
| record | album.tracklist.cd.3.composer | 1 | 0.10% |
| record | album.versions.2.name | 1 | 0.10% |
| record | album.tracklist.cd.0.notes | 1 | 0.10% |
| record | album.tracklist.cd.6.track_number | 1 | 0.10% |
| record | album.tracklist.cd.10.composer | 1 | 0.10% |
| record | album.chart_performance.record_holder_description | 1 | 0.10% |
| record | album.tracklist.cd.9.tie_in.0.type | 1 | 0.10% |
| record | album.tracklist.cd.7.lyrics | 1 | 0.10% |
| record | album.sources.2.reference_number | 1 | 0.10% |
| record | album.external_links.1.description | 1 | 0.10% |
| record | album.tracklist.cd.5.title | 1 | 0.10% |
| record | album.tracklist.cd.7.track_number | 1 | 0.10% |
| record | album.tracklist.cd.7.composer | 1 | 0.10% |
| record | album.tracklist.dvd.1.type | 1 | 0.10% |
| record | album.tracklist.cd.0.tie_in.0.title | 1 | 0.10% |
| record | album.tracklist.cd.1.notes | 1 | 0.10% |
| record | album.chart_performance.record_holder_rank | 1 | 0.10% |
| record | album.sources.0.source_name | 1 | 0.10% |
| record | album.sources.0.reference_number | 1 | 0.10% |
| record | album.tracklist.cd.4.notes | 1 | 0.10% |
| record | album.tracklist.cd.3.tie_in.0.title | 1 | 0.10% |
| record | album.sources.1.access_date | 1 | 0.10% |
| record | album.tracklist.cd.6.title | 1 | 0.10% |
| record | album.versions.0.type | 1 | 0.10% |
| record | album.tracklist.cd.6.composer | 1 | 0.10% |
| record | album.tracklist.cd.10.track_number | 1 | 0.10% |
| record | album.tracklist.cd.8.composer | 1 | 0.10% |
| record | album.tracklist.cd.4.composer | 1 | 0.10% |
| record | album.tracklist.dvd.0.track_number | 1 | 0.10% |
| record | album.versions.1.type | 1 | 0.10% |
| record | album.tracklist.cd.0.composer | 1 | 0.10% |
| record | album.sources.1.reference_number | 1 | 0.10% |
| record | album.tracklist.cd.3.tie_in.0.type | 1 | 0.10% |
| record | album.tracklist.cd.1.track_number | 1 | 0.10% |
| record | album.tracklist.cd.9.composer | 1 | 0.10% |
| record | album.tracklist.cd.9.notes | 1 | 0.10% |
| record | album.tracklist.dvd.0.type | 1 | 0.10% |
| record | album.sources.3.publication_date | 1 | 0.10% |
| record | album.tracklist.cd.2.notes | 1 | 0.10% |
| record | album.tracklist.cd.7.notes | 1 | 0.10% |
| record | album.tracklist.cd.5.composer | 1 | 0.10% |
| record | album.release_date | 1 | 0.10% |
| record | album.chart_performance.ranking | 1 | 0.10% |
| record | album.tracklist.cd.1.composer | 1 | 0.10% |
| record | album.tracklist.cd.6.notes | 1 | 0.10% |
| record | album.free_concert_location | 1 | 0.10% |
| record | album.tracklist.cd.2.tie_in.0.title | 1 | 0.10% |
| record | album.credits.basic_arrangement | 1 | 0.10% |
| record | album.tracklist.dvd.1.title | 1 | 0.10% |
| record | album.tracklist.cd.8.title | 1 | 0.10% |
| record | album.tracklist.cd.8.tie_in.0.type | 1 | 0.10% |
| record | album.tracklist.cd.5.lyrics | 1 | 0.10% |
| record | album.tracklist.cd.8.track_number | 1 | 0.10% |
| record | album.versions.2.contents | 1 | 0.10% |
| record | album.tracklist.cd.10.title | 1 | 0.10% |
| record | album.tracklist.cd.2.lyrics | 1 | 0.10% |
| record | album.tracklist.cd.2.tie_in.0.type | 1 | 0.10% |
| record | album.tracklist.dvd.2.type | 1 | 0.10% |
| record | album.sources.3.url | 1 | 0.10% |
| record | album.tracklist.cd.2.title | 1 | 0.10% |
| record | album.tracklist.dvd.1.track_number | 1 | 0.10% |
| record | album.tracklist.cd.0.title | 1 | 0.10% |
| record | album.sources.1.url | 1 | 0.10% |
| record | album.tracklist.cd.3.track_number | 1 | 0.10% |
| record | album.sources.1.source_name | 1 | 0.10% |
| record | album.external_links.2.description | 1 | 0.10% |
| record | album.tracklist.cd.4.title | 1 | 0.10% |
| record | album.credits.producer | 1 | 0.10% |
| record | album.sources.1.publication_date | 1 | 0.10% |
| record | album.tracklist.cd.2.track_number | 1 | 0.10% |
| record | problemDescription | 1 | 0.10% |
| record | vehicle.year | 1 | 0.10% |
| record | vehicle.make | 1 | 0.10% |
| record | vehicle.model | 1 | 0.10% |
| record | vehicle.problemComponent | 1 | 0.10% |
| record | vehicle.componentReplacements | 1 | 0.10% |
| record | possibleCauses | 1 | 0.10% |
| record | potentialSolutions | 1 | 0.10% |
| record | postInformation.date | 1 | 0.10% |
| record | indication | 1 | 0.10% |
| record | dosage_range | 1 | 0.10% |
| record | age_range | 1 | 0.10% |
| record | approval_status | 1 | 0.10% |
| record | sector | 1 | 0.10% |
| record | movement_reason | 1 | 0.10% |
| record | close_price | 1 | 0.10% |
| record | ticker | 1 | 0.10% |
| record | dream_experiments | 1 | 0.10% |
| record | stock_status | 1 | 0.10% |
| record | output.unit | 1 | 0.10% |
| record | period | 1 | 0.10% |
| record | study.focus | 1 | 0.10% |
| record | person.profession | 1 | 0.10% |
| record | person.us_senator.state | 1 | 0.10% |
| record | person.ideology | 1 | 0.10% |
| record | date: Date the transaction was initiated | 1 | 0.10% |
| record | hash: Cryptographic hash of the transaction data | 1 | 0.10% |
| record | additional_value: Extra numeric value associated with the transaction | 1 | 0.10% |
| record | owner: Name of the transaction owner | 1 | 0.10% |
| record | currency: Currency type used in the transaction | 1 | 0.10% |
| record | address: Full physical address of the owner | 1 | 0.10% |
| record | expert_insights | 1 | 0.10% |
| record | alternative_names | 1 | 0.10% |
| record | orbital_characteristics.semi_major_axis | 1 | 0.10% |
| record | physical_characteristics.observation_year | 1 | 0.10% |
| record | physical_characteristics.absolute_magnitude | 1 | 0.10% |
| record | physical_characteristics.estimated_diameter_km | 1 | 0.10% |
| record | discovery.discovery_rank | 1 | 0.10% |
| record | orbital_characteristics.orbital_period_years | 1 | 0.10% |
| record | physical_characteristics.observations_used | 1 | 0.10% |
| record | event_id: Unique identifier for the conference or symposium | 1 | 0.10% |
| record | event_type: Type of academic event registered for | 1 | 0.10% |
| record | tech_stack | 1 | 0.10% |
| record | pressnoteid | 1 | 0.10% |
| record | insurance_id | 1 | 0.10% |
| record | amount_paid | 1 | 0.10% |
| record | dateofservice | 1 | 0.10% |
| record | prescriber_specialty | 1 | 0.10% |
| record | prescriber_name | 1 | 0.10% |
| record | building | 1 | 0.10% |
| record | patient_address: Full address of the patient | 1 | 0.10% |
| record | diagnosis_code: Medical code assigned to the patient's condition | 1 | 0.10% |
| record | claim_amount: Total cost of the medical service billed | 1 | 0.10% |
| record | scholarship_name | 1 | 0.10% |
| record | negative_note | 1 | 0.10% |
| record | sender: Name of the person sending the outreach | 1 | 0.10% |
| record | amount: Monetary value associated with the outreach | 1 | 0.10% |
| record | address: Full residential address of the patient | 1 | 0.10% |
| record | policy_number: Insurance policy under which claim was submitted | 1 | 0.10% |
| record | patient_name: Name of the individual submitting the claim | 1 | 0.10% |
| record | stock_ticker | 1 | 0.10% |
| record | caveat.example.character | 1 | 0.10% |
| record | caveat.example.components.0.related_characters.0.meaning | 1 | 0.10% |
| record | comparison.ideographic.cross_lingual_understanding | 1 | 0.10% |
| record | comparison.ideographic.example.languages.0.pronunciation | 1 | 0.10% |
| record | comparison.ideographic.example.languages.1.language | 1 | 0.10% |
| record | comparison.ideographic.example.languages.0.language | 1 | 0.10% |
| record | caveat.example.components.1.component | 1 | 0.10% |
| record | caveat.example.components.0.related_characters.0.character | 1 | 0.10% |
| record | comparison.ideographic.example.character | 1 | 0.10% |
| record | finding_id: Unique identifier for the audit finding | 1 | 0.10% |
| record | violation_type: Category of compliance issue identified | 1 | 0.10% |
| record | description: Text explaining the nature and impact of the violation | 1 | 0.10% |
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
| record | announcement.divisions.2.star_of_the_week.player_name | 1 | 0.10% |
| record | announcement.divisions.3.second_star.team | 1 | 0.10% |
| record | announcement.award | 1 | 0.10% |
| record | announcement.divisions.2.honorable_mentions.0.player_name | 1 | 0.10% |
| record | announcement.divisions.2.honorable_mentions.2.position | 1 | 0.10% |
| record | announcement.divisions.2.second_star.player_name | 1 | 0.10% |
| record | announcement.divisions.1.honorable_mentions.0.position | 1 | 0.10% |
| record | announcement.divisions.5.star_of_the_week.position | 1 | 0.10% |
| record | announcement.divisions.4.star_of_the_week.position | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.1.team | 1 | 0.10% |
| record | announcement.divisions.1.division_name | 1 | 0.10% |
| record | announcement.divisions.3.honorable_mentions.0.team | 1 | 0.10% |
| record | announcement.divisions.5.honorable_mentions.0.position | 1 | 0.10% |
| record | announcement.presented_by | 1 | 0.10% |
| record | announcement.divisions.0.star_of_the_week.player_name | 1 | 0.10% |
| record | announcement.week_ending | 1 | 0.10% |
| record | announcement.divisions.1.star_of_the_week.team | 1 | 0.10% |
| record | announcement.divisions.5.star_of_the_week.player_name | 1 | 0.10% |
| record | announcement.divisions.1.star_of_the_week.player_name | 1 | 0.10% |
| record | announcement.divisions.4.division_name | 1 | 0.10% |
| record | announcement.divisions.4.second_star.position | 1 | 0.10% |
| record | announcement.divisions.2.star_of_the_week.position | 1 | 0.10% |
| record | announcement.divisions.4.second_star.player_name | 1 | 0.10% |
| record | announcement.divisions.5.second_star.team | 1 | 0.10% |
| record | announcement.divisions.2.honorable_mentions.1.position | 1 | 0.10% |
| record | appointment_code | 1 | 0.10% |
| record | background | 1 | 0.10% |
| record | effective_date | 1 | 0.10% |
| record | leagues | 1 | 0.10% |
| record | advertiser_information.contact_information.phone_number | 1 | 0.10% |
| record | public_information.brochure_name | 1 | 0.10% |
| record | distribution_information.trail_details.trail_name | 1 | 0.10% |
| record | public_information.download_share | 1 | 0.10% |
| record | advertiser_information.new_brochure_planned | 1 | 0.10% |
| record | distribution_information.distribution_locations | 1 | 0.10% |
| record | service_quality: Description of service experience at the restaurant | 1 | 0.10% |
| record | restaurant_name: Name of the dining establishment reviewed | 1 | 0.10% |
| record | rating: Star rating given to the restaurant on a scale of 1 to 5 | 1 | 0.10% |
| record | review_status: Current status of the review | 1 | 0.10% |
| record | hba1c_value | 1 | 0.10% |
| record | conference_date | 1 | 0.10% |
| record | organization.location_founded.city | 1 | 0.10% |
| record | primate.dental_formula | 1 | 0.10% |
| record | primate.name | 1 | 0.10% |
| record | primate.related_primates.0.genus | 1 | 0.10% |
| record | primate.morphological_analysis.0.type | 1 | 0.10% |
| record | primate.morphological_analysis.1.finding | 1 | 0.10% |
| record | primate.relationships | 1 | 0.10% |
| record | primate.species | 1 | 0.10% |
| record | primate.related_primates.0.species | 1 | 0.10% |
| record | primate.genus | 1 | 0.10% |
| record | primate.related_primates.0.name | 1 | 0.10% |
| record | primate.morphological_analysis.1.genus | 1 | 0.10% |
| record | primate.classification | 1 | 0.10% |
| record | primate.related_primates.0.location | 1 | 0.10% |
| record | primate.morphological_analysis.1.type | 1 | 0.10% |
| record | salary_adjustment | 1 | 0.10% |
| record | action_items | 1 | 0.10% |
| record | first_office | 1 | 0.10% |
| record | work_hours | 1 | 0.10% |
| record | funding_type | 1 | 0.10% |
| record | year_founded | 1 | 0.10% |
| record | boulder.start.foot_holds.0.foot | 1 | 0.10% |
| record | boulder.location.direction | 1 | 0.10% |
| record | boulder.start.foot_holds.1.foot | 1 | 0.10% |
| record | boulder.location.distance | 1 | 0.10% |
| record | boulder.start.hand_holds.1.description | 1 | 0.10% |
| record | boulder.start.foot_holds.1.description | 1 | 0.10% |
| record | boulder.start.hand_holds.1.hand | 1 | 0.10% |
| record | submission.submitted_by | 1 | 0.10% |
| record | boulder.start.foot_holds.0.description | 1 | 0.10% |
| record | device_model | 1 | 0.10% |
| record | purchase_source | 1 | 0.10% |
| record | usage_context | 1 | 0.10% |
| record | device_condition | 1 | 0.10% |
| record | battery_notes | 1 | 0.10% |
| record | publication.title | 1 | 0.10% |
| record | content_analysis.entry_structure.components | 1 | 0.10% |
| record | linguistic_framework.comparison_aspects | 1 | 0.10% |
| record | content_analysis.total_pieces | 1 | 0.10% |
| record | linguistic_framework.concluding_grammar_focus | 1 | 0.10% |
| record | publication.research_context.time_period | 1 | 0.10% |
| record | publication.research_context.informants.description | 1 | 0.10% |
| record | notable_years | 1 | 0.10% |
| record | positions | 1 | 0.10% |
| record | social_media | 1 | 0.10% |
| record | genus.language | 1 | 0.10% |
| record | genus.description | 1 | 0.10% |
| record | genus.period | 1 | 0.10% |
| record | genus.location | 1 | 0.10% |
| record | genus.family.name | 1 | 0.10% |
| record | genus.naming.named_by | 1 | 0.10% |
| record | genus.naming.profession | 1 | 0.10% |
| record | genus.naming.year | 1 | 0.10% |
| record | genus.species.0.reason | 1 | 0.10% |
| record | genus.species.0.name | 1 | 0.10% |
| record | genus.species.0.year | 1 | 0.10% |
| record | genus.unique_features | 1 | 0.10% |
| record | ingredient | 1 | 0.10% |
| record | project_type | 1 | 0.10% |
| record | brands | 1 | 0.10% |
| record | test_date | 1 | 0.10% |
| record | investment_amount | 1 | 0.10% |
| record | grant_code | 1 | 0.10% |
| record | tools | 1 | 0.10% |
| record | name: Full name of the employee | 1 | 0.10% |
| record | supervisor: Name of the employee's supervisor | 1 | 0.10% |
| record | location: City and state where employee is based | 1 | 0.10% |
| record | areaforimprovement: Specific skill or behavior needing development | 1 | 0.10% |
| record | articleTitle | 1 | 0.10% |
| record | circular.subject | 1 | 0.10% |
| record | circular.ministry | 1 | 0.10% |
| record | contraindications | 1 | 0.10% |
| record | shelf_life | 1 | 0.10% |
| record | founder_id | 1 | 0.10% |
| record | physical_characteristics.observations.0.telescope.year_launched | 1 | 0.10% |
| record | physical_characteristics.observations.0.telescope.type | 1 | 0.10% |
| record | physical_characteristics.observations.0.publication_details | 1 | 0.10% |
| record | discovery_details.observatory.name | 1 | 0.10% |
| record | orbital_characteristics.jpl_classification | 1 | 0.10% |
| record | asteroid_belt | 1 | 0.10% |
| record | discovery_details.observatory.location.county | 1 | 0.10% |
| record | reference_number: Unique identifier for the invoice document | 1 | 0.10% |
| record | recipient_name: Name of the recipient organization | 1 | 0.10% |
| record | sender_name: Name of the sender in the correspondence | 1 | 0.10% |
| record | reportTitle | 1 | 0.10% |
| record | address: Full street address of the claimant | 1 | 0.10% |
| record | description: Brief description of the incident and damages | 1 | 0.10% |
| record | status: Current status of the claim | 1 | 0.10% |
| record | prize_pool: Total prize money in US dollars | 1 | 0.10% |
| record | wallet_address: Unique wallet address used for the transaction | 1 | 0.10% |
| record | amount: Quantity of crypto asset bought or sold | 1 | 0.10% |
| record | promotions | 1 | 0.10% |
| record | details.calorie_burn.kettlebell_calories | 1 | 0.10% |
| record | details.calorie_burn.kettlebell_time | 1 | 0.10% |
| record | details.calorie_burn.running_calories | 1 | 0.10% |
| record | details.calorie_burn.running_time | 1 | 0.10% |
| record | details.calorie_burn.running_speed | 1 | 0.10% |
| record | serial: Unique identifier for the product unit | 1 | 0.10% |
| record | agent: Support agent assigned to handle the request | 1 | 0.10% |
| record | ticket_id: Internal ticket identifier for the RMA | 1 | 0.10% |
| record | rma_number: Assigned number for the RMA case | 1 | 0.10% |
| record | film | 1 | 0.10% |
| record | address: Full residential address of the customer | 1 | 0.10% |
| record | reason_for_difference | 1 | 0.10% |
| record | kadlub_shape | 1 | 0.10% |
| record | tax_incentive.duration.period.start_year | 1 | 0.10% |
| record | announcement_metadata.agency | 1 | 0.10% |
| record | tax_incentive.tax_rate_percent | 1 | 0.10% |
| record | announcement_metadata.source_url | 1 | 0.10% |
| record | tax_incentive.duration.period.end_year | 1 | 0.10% |
| record | announcement_metadata.coverage_provider | 1 | 0.10% |
| record | salary_range | 1 | 0.10% |
| record | company_size | 1 | 0.10% |
| record | employment_type | 1 | 0.10% |
| record | holiday_look.collar.brand | 1 | 0.10% |
| record | holiday_look.heels.brand | 1 | 0.10% |
| record | comparison.similar_look | 1 | 0.10% |
| record | comparison.brand_with_similar_look | 1 | 0.10% |
| record | holiday_look.dress.designer | 1 | 0.10% |
| record | book_title | 1 | 0.10% |
| record | description: Purpose or context of the contribution | 1 | 0.10% |
| record | donor_name: Full name of the contributing individual or organization | 1 | 0.10% |
| record | recipient_office: Office or position being supported | 1 | 0.10% |
| record | campaign_committee: Name of the political committee receiving the contribution | 1 | 0.10% |
| record | disclosure_id: Unique identifier for the disclosure entry | 1 | 0.10% |
| record | sender_domain: Domain used in the fraudulent email | 1 | 0.10% |
| record | jersey_number: Player's assigned jersey number on field | 1 | 0.10% |
| record | away_team: Team playing away | 1 | 0.10% |
| record | home_team: Team playing at home stadium | 1 | 0.10% |
| record | match_location: Stadium or venue where match occurred | 1 | 0.10% |
| record | sender_domain: Domain of the spoofed email address | 1 | 0.10% |
| record | sender_email: Email address from which the phishing message originated | 1 | 0.10% |
| record | status: Current status of the report | 1 | 0.10% |
| record | lastcommitdate | 1 | 0.10% |
| record | symbol | 1 | 0.10% |
| record | performance_reason | 1 | 0.10% |
| record | change | 1 | 0.10% |
| record | location: Headquarters city and state or country | 1 | 0.10% |
| record | investors: List of investor names or groups involved | 1 | 0.10% |
| record | location: City and state where the company was founded | 1 | 0.10% |
| record | motivation: Personal reason driving the company's founding | 1 | 0.10% |
| record | founder_name: Full name of the founder as stated in passage | 1 | 0.10% |
| record | funding_amount: Initial capital received in USD | 1 | 0.10% |
| record | service_type | 1 | 0.10% |
| record | investor_count: Number of investors involved in the round | 1 | 0.10% |
| record | location: City and country where the startup is headquartered | 1 | 0.10% |
| record | focus_area: Primary technological or business focus of the startup | 1 | 0.10% |
| record | winner: Name of the player or team that won | 1 | 0.10% |
| record | additional_investors | 1 | 0.10% |
| record | funds_use | 1 | 0.10% |
| record | procedure_code: CPT code describing the medical service performed | 1 | 0.10% |
| record | source_office: Clinic or office that submitted the claim | 1 | 0.10% |
| record | claim_id: Unique identifier for the claim as referenced in correspondence | 1 | 0.10% |
| record | drink | 1 | 0.10% |
| record | program_code | 1 | 0.10% |
| record | end_date | 1 | 0.10% |
| record | village_name | 1 | 0.10% |
| record | division | 1 | 0.10% |
| record | review_message | 1 | 0.10% |
| record | github_link | 1 | 0.10% |
| record | report_date: Date when the issue was first reported | 1 | 0.10% |
| record | resolution_action: Type of action taken to resolve the issue | 1 | 0.10% |
| record | status: Current state of the ticket after resolution | 1 | 0.10% |
| record | ticket_id: Unique identifier for the support ticket | 1 | 0.10% |
| record | customer_email: Email address associated with the ticket | 1 | 0.10% |
| record | tv_program.production.channel | 1 | 0.10% |
| record | south_yemen_conflict.accusations_against_governments.0.government_name | 1 | 0.10% |
| record | federalism_definition.federalism | 1 | 0.10% |
| record | south_yemen_conflict.historical_context.protests_and_killings.casualties | 1 | 0.10% |
| record | south_yemen_conflict.historical_context.actions_after_invasion | 1 | 0.10% |
| record | south_yemen_conflict.accusations_against_governments.1.government_name | 1 | 0.10% |
| record | south_yemen_conflict.historical_context.casualties_and_prisoners | 1 | 0.10% |
| record | federalism_definition.confederation_examples | 1 | 0.10% |
| record | future_outlook | 1 | 0.10% |
| record | federalism_definition.confederation | 1 | 0.10% |
| record | study_type | 1 | 0.10% |
| record | referring_physician | 1 | 0.10% |
| record | facility | 1 | 0.10% |
| record | facility_address | 1 | 0.10% |
| record | study_code | 1 | 0.10% |
| record | commonName | 1 | 0.10% |
| record | subgenus | 1 | 0.10% |
| record | doctor.action | 1 | 0.10% |
| record | doctor.question | 1 | 0.10% |
| record | patient.behavior | 1 | 0.10% |
| record | chair.condition | 1 | 0.10% |
| record | patient.age_estimation.full_age | 1 | 0.10% |
| record | chair.state | 1 | 0.10% |
| record | patient.request | 1 | 0.10% |
| record | patient.age_estimation.initial_estimation | 1 | 0.10% |
| record | ram | 1 | 0.10% |
| record | year_level | 1 | 0.10% |
| record | election_district | 1 | 0.10% |
| record | asteroid.alternative_name.german | 1 | 0.10% |
| record | asteroid.orbital_period.unit | 1 | 0.10% |
| record | document.economic_impact.growth_rate | 1 | 0.10% |
| record | document.intellectual_property_rights | 1 | 0.10% |
| record | document.location | 1 | 0.10% |
| record | document.date | 1 | 0.10% |
| record | document.strategy_elements.5.area | 1 | 0.10% |
| record | document.economic_impact.employment | 1 | 0.10% |
| record | document.commissioners.1.portfolio | 1 | 0.10% |
| record | document.commissioners.0.portfolio | 1 | 0.10% |
| record | document.strategy_elements.1.area | 1 | 0.10% |
| record | document.related_documents | 1 | 0.10% |
| record | document.commissioners.1.name | 1 | 0.10% |
| record | document.commissioners.0.name | 1 | 0.10% |
| record | document.title | 1 | 0.10% |
| record | id: Unique identifier assigned to the startup | 1 | 0.10% |
| record | funding_amount: Total amount raised in the investment round | 1 | 0.10% |
| record | visitor_team | 1 | 0.10% |
| record | sku: Unique product identifier as assigned in inventory | 1 | 0.10% |
| record | rating: Customer rating out of 5 stars | 1 | 0.10% |
| record | material: Primary material composition of the gear | 1 | 0.10% |
| record | weight: Product weight in pounds | 1 | 0.10% |
| record | color: Available color(s) of the product | 1 | 0.10% |
| record | client_email | 1 | 0.10% |
| record | marketing_budget | 1 | 0.10% |
| record | iporevisedvaluation | 1 | 0.10% |
| record | originalheadcountapex | 1 | 0.10% |
| record | cities | 1 | 0.10% |
| record | defense_systems | 1 | 0.10% |
| record | facilities | 1 | 0.10% |
| record | medical_terms | 1 | 0.10% |
| record | professions | 1 | 0.10% |
| record | status: Current status of the lease agreement | 1 | 0.10% |
| record | address: Full physical address of the leased property | 1 | 0.10% |
| record | tenant: Name of the company leasing the space | 1 | 0.10% |
| record | home_team: Name of the team playing at home | 1 | 0.10% |
| record | key_moment: Descriptive note about a pivotal play or event in the game | 1 | 0.10% |
| record | game_id: Unique identifier for the game as provided in the passage | 1 | 0.10% |
| record | patient_gender: Gender of the patient | 1 | 0.10% |
| record | drug_dosage: Dosage and frequency of the drug exposure | 1 | 0.10% |
| record | university | 1 | 0.10% |
| record | social_media_post.content | 1 | 0.10% |
| record | social_media_post.hashtags | 1 | 0.10% |
| record | event.subject | 1 | 0.10% |
| record | event.location.region | 1 | 0.10% |
| record | social_media_post.platform | 1 | 0.10% |
| record | event.surgery_date | 1 | 0.10% |
| record | vehicle_make | 1 | 0.10% |
| record | vehicle_model | 1 | 0.10% |
| record | adjuster | 1 | 0.10% |
| record | total_rainfall | 1 | 0.10% |
| record | funding_allocated | 1 | 0.10% |
| record | gust_speed | 1 | 0.10% |
| record | follow_up_survey | 1 | 0.10% |
| record | severity_rating | 1 | 0.10% |
| record | retail_price | 1 | 0.10% |
| record | target_demographic | 1 | 0.10% |
| record | product_category | 1 | 0.10% |
| record | deployment_status | 1 | 0.10% |
| record | distribution_channels | 1 | 0.10% |
| record | customer_satisfaction | 1 | 0.10% |
| record | location: City and state where the startup is based | 1 | 0.10% |
| record | founding_year: Year the startup was established | 1 | 0.10% |
| record | active_clients: Number of active clients or installations as of a stated year | 1 | 0.10% |
| record | founding_address: Physical address of the startup's registration | 1 | 0.10% |
| record | equity_stake: Founder's ownership percentage in the company | 1 | 0.10% |
| record | date_submitted: Date the issue was submitted to support | 1 | 0.10% |
| record | thread_id: Unique identifier for the support ticket | 1 | 0.10% |
| record | description: Detailed explanation of the issue and customer's concern | 1 | 0.10% |
| record | customer_name: Name of the customer filing the issue | 1 | 0.10% |
| record | amount: Monetary value of the transaction in USD | 1 | 0.10% |
| record | status: Current state of the transaction execution | 1 | 0.10% |
| record | reference_code: Internal code linking to external system or invoice | 1 | 0.10% |
| record | description: Brief description of the transaction purpose | 1 | 0.10% |
| record | asteroid.alternative_designations | 1 | 0.10% |
| record | asteroid.type | 1 | 0.10% |
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
| record | copyright | 1 | 0.10% |
| record | municipality.name | 1 | 0.10% |
| record | exchange | 1 | 0.10% |
| record | deck_coating_systems.0.resin_type | 1 | 0.10% |
| record | text | 1 | 0.10% |
| record | policy_type: Type of insurance policy renewed | 1 | 0.10% |
| record | policy_id: Unique identifier assigned to the policy | 1 | 0.10% |
| record | product_name: Name of the launched product | 1 | 0.10% |
| record | location: Geographic market or city where product is available | 1 | 0.10% |
| record | hashtag: Primary hashtag used in the launch post | 1 | 0.10% |
| record | account_handle: Instagram handle used for the launch post | 1 | 0.10% |
| record | from_club | 1 | 0.10% |
| record | transfer_code | 1 | 0.10% |
| record | player_id | 1 | 0.10% |
| record | stock_level | 1 | 0.10% |
| record | risk_classification | 1 | 0.10% |
| record | app | 1 | 0.10% |
| record | agency_code | 1 | 0.10% |
| record | caveat.example.components.0.related_characters.1.meaning | 1 | 0.10% |
| record | caveat.example.components.0.related_characters.1.pronunciation | 1 | 0.10% |
| record | caveat.example.components.0.related_characters.2.meaning | 1 | 0.10% |
| record | caveat.example.components.0.meaning | 1 | 0.10% |
| record | caveat.example.components.0.position | 1 | 0.10% |
| record | comparison.alphabetic.learning_curve | 1 | 0.10% |
| record | caveat.example.components.0.related_characters.2.pronunciation | 1 | 0.10% |
| record | caveat.example.components.1.meaning | 1 | 0.10% |
| record | caveat.example.components.0.related_characters.0.pronunciation | 1 | 0.10% |
| record | terminology.ideographic_script.english | 1 | 0.10% |
| record | caveat.example.components.0.function | 1 | 0.10% |
| record | caveat.example.components.0.related_characters.1.character | 1 | 0.10% |
| record | comparison.ideographic.learning_curve | 1 | 0.10% |
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
| record | booth_number | 1 | 0.10% |
| record | trade_show | 1 | 0.10% |
| record | exhibitor_id | 1 | 0.10% |
| record | resource_name | 1 | 0.10% |
| record | drug | 1 | 0.10% |
| record | reported_by | 1 | 0.10% |
| record | weather_conditions | 1 | 0.10% |
| record | name: Full name of the startup company | 1 | 0.10% |
| record | other_investors: List of additional participating investors | 1 | 0.10% |
| record | lead_investor: Primary investor leading the round | 1 | 0.10% |
| record | see_also | 1 | 0.10% |
| record | product.intended_use | 1 | 0.10% |
| record | product.technology | 1 | 0.10% |
| record | description: Detailed account of the issue reported | 1 | 0.10% |
| record | submitter: Name of the employee who submitted the ticket | 1 | 0.10% |
| record | affected_system: Software or hardware system experiencing the problem | 1 | 0.10% |
| record | priority: Urgency level of the ticket | 1 | 0.10% |
| record | statement_id: Unique identifier for the press conference statement | 1 | 0.10% |
| record | tournament_code | 1 | 0.10% |
| record | purchase_order: Order number from original purchase | 1 | 0.10% |
| record | purchase_location: Store where device was originally bought | 1 | 0.10% |
| record | wallet_address: Cryptocurrency wallet address involved in the transaction | 1 | 0.10% |
| record | user_id: Unique identifier for the user performing the transaction | 1 | 0.10% |
| record | approval_date | 1 | 0.10% |
| record | previous_jobs | 1 | 0.10% |
| record | degree | 1 | 0.10% |
| record | town.province | 1 | 0.10% |
| record | town.name | 1 | 0.10% |
| record | metabolite | 1 | 0.10% |
| record | issue_date | 1 | 0.10% |
| record | term_length | 1 | 0.10% |
| record | monthly_premium | 1 | 0.10% |
| record | street | 1 | 0.10% |
| record | affected_system | 1 | 0.10% |
| record | customer_affected | 1 | 0.10% |
| record | resolution_method | 1 | 0.10% |
| record | incident_code | 1 | 0.10% |
| record | impact | 1 | 0.10% |
| record | genus.family | 1 | 0.10% |
| record | genus.subspecies.0.name | 1 | 0.10% |
| record | genus.name_latin | 1 | 0.10% |
| record | genus.subspecies.0.authority | 1 | 0.10% |
| record | source.location | 1 | 0.10% |
| record | incident.location.city | 1 | 0.10% |
| record | incident.location.region | 1 | 0.10% |
| record | person.family.0.relation | 1 | 0.10% |
| record | source.name | 1 | 0.10% |
| record | person.family.0.name | 1 | 0.10% |
| record | incident.cause | 1 | 0.10% |
| record | patient_id: Unique identifier for each patient visit | 1 | 0.10% |
| record | diagnosis: Medical code indicating the diagnosed condition | 1 | 0.10% |
| record | clinical_outcomes.dosage_efficiency | 1 | 0.10% |
| record | clinical_outcomes.glycemic_control.effectiveness | 1 | 0.10% |
| record | administration_context.combination_therapies | 1 | 0.10% |
| record | handle: Social media handle for the brand | 1 | 0.10% |
| record | document_status | 1 | 0.10% |
| record | emergency_contact | 1 | 0.10% |
| record | grant_id | 1 | 0.10% |
| record | faculty_name | 1 | 0.10% |
| record | dean.title | 1 | 0.10% |
| record | dean.name | 1 | 0.10% |
| record | staff.personnel_count | 1 | 0.10% |
| record | company.address.city | 1 | 0.10% |
| record | company.address.building_number | 1 | 0.10% |
| record | company.products | 1 | 0.10% |
| record | assigned_agent | 1 | 0.10% |
| record | nutritional_aspects.sucrose.daily_limit_percentage | 1 | 0.10% |
| record | nutritional_aspects.carbohydrates.daily_percentage | 1 | 0.10% |
| record | nutritional_aspects.carbohydrates.low_glycemic_index.percentage_with_fiber | 1 | 0.10% |
| record | weight_loss_benefits.improvements | 1 | 0.10% |
| record | nutritional_aspects.carbohydrates.low_glycemic_index.recommendation | 1 | 0.10% |
| record | trial_nct | 1 | 0.10% |
| record | founder_name | 1 | 0.10% |
| record | mission | 1 | 0.10% |
| record | early_challenge | 1 | 0.10% |
| record | turning_point | 1 | 0.10% |
| record | current_scale | 1 | 0.10% |
| record | investor_name: Name of the investor attending | 1 | 0.10% |
| record | contact: Name of the point of contact at the company | 1 | 0.10% |
| record | status: Current status of the investment round | 1 | 0.10% |
| record | amount_usd: USD amount involved in the investment | 1 | 0.10% |
| record | location: Physical location of the briefing event | 1 | 0.10% |
| record | company_name: Name of the startup company presenting | 1 | 0.10% |
| record | analysis_context | 1 | 0.10% |
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
| record | amountindispute: Monetary value of the claim or settlement | 1 | 0.10% |
| record | defendant: Name of the individual or entity being sued | 1 | 0.10% |
| record | case_type: Category of legal matter such as contract or personal injury | 1 | 0.10% |
| record | series | 1 | 0.10% |
| record | review_gift.available_to | 1 | 0.10% |
| record | shipping_restriction.country | 1 | 0.10% |
| record | shipping_restriction.contact_seller_prompt | 1 | 0.10% |
| record | javascript_requirement.message | 1 | 0.10% |
| record | javascript_requirement.enable_instruction | 1 | 0.10% |
| record | javascript_requirement.question_prompt | 1 | 0.10% |
| record | integration_task_id | 1 | 0.10% |
| record | ui_bug_status | 1 | 0.10% |
| record | budget_remaining | 1 | 0.10% |
| record | security_compliance_delay | 1 | 0.10% |
| record | backend_integration_status | 1 | 0.10% |
| record | actual_spend | 1 | 0.10% |
| record | sales_trend | 1 | 0.10% |
| record | established | 1 | 0.10% |
| record | serial | 1 | 0.10% |
| record | favorite_players | 1 | 0.10% |
| record | edit_information.timestamp | 1 | 0.10% |
| record | sentiment.opinion | 1 | 0.10% |
| record | quote.text | 1 | 0.10% |
| record | sentiment.preference | 1 | 0.10% |
| record | adjectival_modifier | 1 | 0.10% |
| record | numerical_values | 1 | 0.10% |
| record | nouns_key_entities | 1 | 0.10% |
| record | adjectival_complement | 1 | 0.10% |
| record | title: Title of the research paper | 1 | 0.10% |
| record | institution: University and department affiliation of the authors | 1 | 0.10% |
| record | submission_amount: Financial amount associated with submission or grant | 1 | 0.10% |
| record | ties | 1 | 0.10% |
| record | final_standing | 1 | 0.10% |
| record | season | 1 | 0.10% |
| record | win_percentage | 1 | 0.10% |
| record | coach | 1 | 0.10% |
| record | theological_concepts | 1 | 0.10% |
| record | course_id | 1 | 0.10% |
| record | pricing | 1 | 0.10% |
| record | total_interactions | 1 | 0.10% |
| record | announcement.divisions.1.honorable_mentions.0.player_name | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.2.position | 1 | 0.10% |
| record | announcement.divisions.2.honorable_mentions.0.position | 1 | 0.10% |
| record | announcement.divisions.2.division_name | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.4.player_name | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.4.position | 1 | 0.10% |
| record | announcement.divisions.3.second_star.player_name | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.7.team | 1 | 0.10% |
| record | announcement.divisions.5.honorable_mentions.2.player_name | 1 | 0.10% |
| record | announcement.divisions.0.star_of_the_week.stats | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.1.position | 1 | 0.10% |
| record | announcement.divisions.2.honorable_mentions.3.player_name | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.0.team | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.5.player_name | 1 | 0.10% |
| record | announcement.divisions.5.star_of_the_week.team | 1 | 0.10% |
| record | announcement.divisions.2.honorable_mentions.2.team | 1 | 0.10% |
| record | announcement.divisions.5.division_name | 1 | 0.10% |
| record | announcement.divisions.3.honorable_mentions.0.position | 1 | 0.10% |
| record | announcement.divisions.4.honorable_mentions.1.team | 1 | 0.10% |
| record | announcement.divisions.2.honorable_mentions.1.player_name | 1 | 0.10% |
| record | announcement.divisions.3.honorable_mentions.0.player_name | 1 | 0.10% |
| record | announcement.divisions.3.honorable_mentions.1.player_name | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.5.position | 1 | 0.10% |
| record | announcement.divisions.3.star_of_the_week.position | 1 | 0.10% |
| record | announcement.divisions.0.division_name | 1 | 0.10% |
| record | announcement.divisions.2.honorable_mentions.1.team | 1 | 0.10% |
| record | announcement.divisions.2.honorable_mentions.3.position | 1 | 0.10% |
| record | announcement.divisions.0.second_star.player_name | 1 | 0.10% |
| record | announcement.divisions.2.honorable_mentions.3.team | 1 | 0.10% |
| record | announcement.divisions.1.star_of_the_week.stats | 1 | 0.10% |
| record | announcement.divisions.0.honorable_mentions.3.player_name | 1 | 0.10% |
| record | announcement.divisions.0.second_star.position | 1 | 0.10% |
| record | announcement.divisions.1.honorable_mentions.1.position | 1 | 0.10% |
| record | announcement.divisions.3.second_star.position | 1 | 0.10% |
| record | announcement.divisions.5.honorable_mentions.1.player_name | 1 | 0.10% |
| record | announcement.divisions.5.honorable_mentions.2.position | 1 | 0.10% |
| record | announcement.divisions.1.honorable_mentions.0.team | 1 | 0.10% |
| record | delayed_games | 1 | 0.10% |
| record | upcoming_games | 1 | 0.10% |
| record | steps | 1 | 0.10% |
| record | bronze_medalist: Name of athlete who won bronze medal | 1 | 0.10% |
| record | event_code: Unique identifier for the Olympic competition | 1 | 0.10% |
| record | gold_medalist: Name of athlete who won gold medal | 1 | 0.10% |
| record | app_package | 1 | 0.10% |
| record | credits.contributors.0.role | 1 | 0.10% |
| record | credits.contributors.0.person | 1 | 0.10% |
| record | credits.content_implementers | 1 | 0.10% |
| record | credits.graphic_design.0.designer | 1 | 0.10% |
| record | credits.organization | 1 | 0.10% |
| record | credits.funding | 1 | 0.10% |
| record | credits.graphic_design.1.designer | 1 | 0.10% |
| record | credits.version | 1 | 0.10% |
| record | credits.graphic_design.1.copyright_year | 1 | 0.10% |
| record | credits.contributors.1.role | 1 | 0.10% |
| record | user_name: Name of the user who created the repost | 1 | 0.10% |
| record | post_date: Date when the trend repost was published | 1 | 0.10% |
| record | trend_name: Name of the viral trend being reposted | 1 | 0.10% |
| record | description: Brief text describing the content of the post | 1 | 0.10% |
| record | shares: Number of shares the post received | 1 | 0.10% |
| record | post_status | 1 | 0.10% |
| record | magazine.name | 1 | 0.10% |
| record | magazine.issue | 1 | 0.10% |
| record | magazine.article.title | 1 | 0.10% |
| record | magazine.article.advertising_percentage | 1 | 0.10% |
| record | magazine.article.content_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.0.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.0.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.0.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.1.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.1.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.1.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.2.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.2.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.2.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.3.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.3.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.3.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.4.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.4.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.4.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.5.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.5.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.5.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.6.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.6.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.6.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.7.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.7.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.7.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.8.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.8.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.8.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.9.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.9.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.9.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.10.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.10.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.10.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.11.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.11.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.11.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.12.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.12.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.12.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.13.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.13.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.13.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.14.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.14.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.14.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.15.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.15.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.15.advertisers | 1 | 0.10% |
| record | magazine.article.page_breakdown.16.page_number | 1 | 0.10% |
| record | magazine.article.page_breakdown.16.advertising_percentage | 1 | 0.10% |
| record | magazine.article.page_breakdown.16.advertisers | 1 | 0.10% |
| record | magazine.mentioned_article.title | 1 | 0.10% |
| record | magazine.mentioned_article.author | 1 | 0.10% |
| record | magazine.mentioned_article.year | 1 | 0.10% |
| record | contact_email: Email address of primary passenger for communication | 1 | 0.10% |
| record | route: Origin-destination route string like JFK-LAX | 1 | 0.10% |
| record | delay_reason: Human-readable description of delay cause | 1 | 0.10% |
| record | passenger_names: List of passenger names affected | 1 | 0.10% |
| record | delay_minutes: Duration of delay in minutes | 1 | 0.10% |
| record | flight_number: Unique identifier for the delayed flight | 1 | 0.10% |
| record | date: Date the exchange rate snapshot was taken | 1 | 0.10% |
| record | currency_pair: Pair of currencies involved in the exchange | 1 | 0.10% |
| record | rate: Exchange rate value from base to USD | 1 | 0.10% |
| record | sender: Name of the individual initiating the transfer | 1 | 0.10% |
| record | address: Physical address of the sender | 1 | 0.10% |
| record | winner_percentage: Percentage of votes won by the winner | 1 | 0.10% |
| record | district_id: Unique identifier for the electoral district | 1 | 0.10% |
| record | capacity | 1 | 0.10% |
| record | status: Current status of the exam result | 1 | 0.10% |
| record | school: Name of the school where the exam was administered | 1 | 0.10% |
| record | score: Percentage score achieved in the exam | 1 | 0.10% |
| record | subject: Subject area of the exam | 1 | 0.10% |
| record | onset_timing | 1 | 0.10% |
| record | symptom_description | 1 | 0.10% |
| record | usage_frequency | 1 | 0.10% |
| record | followupdate | 1 | 0.10% |
| record | co_investors: List of additional investors participating in the round | 1 | 0.10% |
| record | company: Name of the technology startup | 1 | 0.10% |
| record | funding_amount: Total amount raised in millions of dollars | 1 | 0.10% |
| record | helpful_votes | 1 | 0.10% |
| record | os_version | 1 | 0.10% |
| record | causes | 1 | 0.10% |
| record | sheet_id | 1 | 0.10% |
| record | key_scorer | 1 | 0.10% |
| record | name: Full name of the contact person | 1 | 0.10% |
| record | email: Email address of the contact | 1 | 0.10% |
| record | date: Date the outreach was made | 1 | 0.10% |
| record | title: Title of the proposed solution | 1 | 0.10% |
| record | price: Total price of the proposed solution | 1 | 0.10% |
| record | status: Current status of the outreach | 1 | 0.10% |
| record | ref_id: Unique reference identifier for the proposal | 1 | 0.10% |
| record | solution: Description of the proposed solution | 1 | 0.10% |
| record | target: Target audience or department for the proposal | 1 | 0.10% |
| record | posted_on | 1 | 0.10% |
| record | passengers_affected | 1 | 0.10% |
| record | asteroid.designation_number | 1 | 0.10% |
| record | asteroid.dedication.person.nationality | 1 | 0.10% |
| record | asteroid.dedication.person.occupation | 1 | 0.10% |
| record | author_information.education.0.institution | 1 | 0.10% |
| record | author_information.military_service.branch | 1 | 0.10% |
| record | author_information.military_service.status | 1 | 0.10% |
| record | author_information.occupation | 1 | 0.10% |
| record | author_information.location.city | 1 | 0.10% |
| record | author_information.location.state | 1 | 0.10% |
| record | author_information.family.spouse | 1 | 0.10% |
| record | author_information.family.pets.0.name | 1 | 0.10% |
| record | author_information.family.pets.0.type | 1 | 0.10% |
| record | author_information.interests | 1 | 0.10% |
| record | religious_concepts | 1 | 0.10% |
| record | mrn: Unique medical record number identifying the patient | 1 | 0.10% |
| record | address: Street address of the patient | 1 | 0.10% |
| record | city: City where the patient resides | 1 | 0.10% |
| record | state: State where the patient resides | 1 | 0.10% |
| record | diagnosis_code: ICD-10 code for the patient's condition | 1 | 0.10% |
| record | stay_duration: Number of days the patient stayed in the hospital | 1 | 0.10% |
| record | treatment: Medical treatment provided during stay | 1 | 0.10% |
| record | prescription: Medication or therapy prescribed | 1 | 0.10% |
| record | followupdoctor: Name of the doctor for follow-up care | 1 | 0.10% |
| record | hospital_name: Name of the hospital where discharge occurred | 1 | 0.10% |
| record | insurance: Insurance provider covering the bill | 1 | 0.10% |
| record | dates_mentioned | 1 | 0.10% |
| record | destination: Final address of the shipment | 1 | 0.10% |
| record | consignment_id: Unique identifier for each shipment | 1 | 0.10% |
| record | appointment_type | 1 | 0.10% |
| record | institution: Name of the financial institution publishing the rate | 1 | 0.10% |
| record | currency_pair: Two-currency pair such as EUR/USD or SGD/USD | 1 | 0.10% |
| record | effective_time: Local time when the rate was effective | 1 | 0.10% |
| record | rate: Exchange rate value for the currency pair | 1 | 0.10% |
| record | validation_status: Current validation state of the rate data | 1 | 0.10% |
| record | agency.name | 1 | 0.10% |
| record | agency.abbreviation | 1 | 0.10% |
| record | agency.mission | 1 | 0.10% |
| record | agency.partners | 1 | 0.10% |
| record | citizen_corps.description | 1 | 0.10% |
| record | citizen_corps.activities | 1 | 0.10% |
| record | citizen_corps.website | 1 | 0.10% |
| record | ready_campaign_updates.target_audience | 1 | 0.10% |
| record | ready_campaign_updates.email | 1 | 0.10% |
| record | ready_campaign_updates.language_note | 1 | 0.10% |
| record | summary: Brief description of the root cause or issue | 1 | 0.10% |
| record | contact_number: Primary phone number for customer communication | 1 | 0.10% |
| record | address: Full address of the customer location | 1 | 0.10% |
| record | recipient: Name of the recipient professional contacted | 1 | 0.10% |
| record | subject: Subject line of the outreach email | 1 | 0.10% |
| record | sender: Name of the recruiter initiating the outreach | 1 | 0.10% |
| record | reviewer: Name of the film critic who wrote the review | 1 | 0.10% |
| record | user: Name of the employee reporting the issue | 1 | 0.10% |
| record | ticket_id: Unique identifier for each ticket as per prefix format | 1 | 0.10% |
| record | affected_system: System or application where the issue occurs | 1 | 0.10% |
| record | reference: Internal reference code linked to the ticket | 1 | 0.10% |
| record | issue: Nature of the technical problem reported | 1 | 0.10% |
| record | submitted: Date when the ticket was submitted | 1 | 0.10% |
| record | founded_year | 1 | 0.10% |
| record | actor.name | 1 | 0.10% |
| record | actor.quote | 1 | 0.10% |
| record | review.reviewer | 1 | 0.10% |
| record | review.publication | 1 | 0.10% |
| record | review.comment | 1 | 0.10% |
| record | source.domain | 1 | 0.10% |
| record | source.via | 1 | 0.10% |
| record | co_stars | 1 | 0.10% |
| record | movie.title | 1 | 0.10% |
| record | movie.year | 1 | 0.10% |
| record | movie.based_on.title | 1 | 0.10% |
| record | movie.based_on.author | 1 | 0.10% |
| record | movie.character | 1 | 0.10% |
| record | species.scientific_name | 1 | 0.10% |
| record | species.genus | 1 | 0.10% |
| record | species.description | 1 | 0.10% |
| record | codes | 1 | 0.10% |
| record | when_to_seek_help | 1 | 0.10% |
| record | limitations_of_midwives | 1 | 0.10% |
| record | treatment.effectiveness | 1 | 0.10% |
| record | discovery.name | 1 | 0.10% |
| record | disease.gene | 1 | 0.10% |
| record | discovery.impact | 1 | 0.10% |
| record | treatment.type | 1 | 0.10% |
| record | methodology.sample | 1 | 0.10% |
| record | compensation_type | 1 | 0.10% |
| record | merchantcategorycode | 1 | 0.10% |
| record | restaurant.cuisine | 1 | 0.10% |
| record | restaurant.description | 1 | 0.10% |
| record | restaurant.establishment_date | 1 | 0.10% |
| record | restaurant.name | 1 | 0.10% |
| record | track | 1 | 0.10% |
| record | reviewers | 1 | 0.10% |
| record | travel_support | 1 | 0.10% |
| record | encounter_id: Unique identifier for the visit | 1 | 0.10% |
| record | address: Patient's residential address | 1 | 0.10% |
| record | visit_date: Date of the medical encounter | 1 | 0.10% |
| record | primary_diagnosis: Main medical condition diagnosed | 1 | 0.10% |
| record | icd_code: ICD-10 code for the primary diagnosis | 1 | 0.10% |
| record | bp_systolic: Systolic blood pressure in mmHg | 1 | 0.10% |
| record | bp_diastolic: Diastolic blood pressure in mmHg | 1 | 0.10% |
| record | heart_rate: Heart rate in beats per minute | 1 | 0.10% |
| record | medications: List of prescribed medications with dosing | 1 | 0.10% |
| record | lab_results: List of lab test results | 1 | 0.10% |
| record | symptoms: Patient-reported symptoms or complaints | 1 | 0.10% |
| record | study.quotes | 1 | 0.10% |
| record | study.author.name | 1 | 0.10% |
| record | person.political_party | 1 | 0.10% |
| record | person.us_senator.years | 1 | 0.10% |
| record | person.religion | 1 | 0.10% |
| record | project | 1 | 0.10% |
| record | upvote_count | 1 | 0.10% |
| record | subreddit | 1 | 0.10% |
| record | tone | 1 | 0.10% |
| record | treaty_name | 1 | 0.10% |
| record | signatories | 1 | 0.10% |
| record | treaty_code | 1 | 0.10% |
| record | goal | 1 | 0.10% |
| record | catalog_id | 1 | 0.10% |
| record | rsvp_count | 1 | 0.10% |
| record | startup_capital | 1 | 0.10% |
| record | key_challenge | 1 | 0.10% |
| record | discovery.observatory.location.county | 1 | 0.10% |
| record | orbital_characteristics.inclination.reference | 1 | 0.10% |
| record | model_name | 1 | 0.10% |
| record | lease_type | 1 | 0.10% |
| record | lease_duration | 1 | 0.10% |
| record | total_changes | 1 | 0.10% |
| record | unit_id | 1 | 0.10% |
| record | amenities | 1 | 0.10% |
| record | focus_area | 1 | 0.10% |
| record | stock_exchange | 1 | 0.10% |
| record | authority.year | 1 | 0.10% |
| record | critic_name | 1 | 0.10% |
| record | film_title | 1 | 0.10% |
| record | office_address | 1 | 0.10% |
| record | critic_location | 1 | 0.10% |
| record | full_name: Name of the employee as recorded in the review | 1 | 0.10% |
| record | location: Physical address of the employee's workplace | 1 | 0.10% |
| record | kpideliveryaccuracy: Percentage of on-time deliveries achieved | 1 | 0.10% |
| record | department: Department where the employee is based | 1 | 0.10% |
| record | customer_satisfaction: Score from customer feedback surveys | 1 | 0.10% |
| record | posting platform | 1 | 0.10% |
| record | year range | 1 | 0.10% |
| record | person name | 1 | 0.10% |
| record | territory | 1 | 0.10% |
| record | vehicle model | 1 | 0.10% |
| record | network_id | 1 | 0.10% |
| record | expression_level | 1 | 0.10% |
| record | expression_units | 1 | 0.10% |
| record | gene_id | 1 | 0.10% |
| record | county | 1 | 0.10% |
| record | submission_id: Unique identifier for the paper submission | 1 | 0.10% |
| record | author: Name of the submitting researcher or academic | 1 | 0.10% |
| record | institution: University or institution where author is affiliated | 1 | 0.10% |
| record | department: Academic department of the author | 1 | 0.10% |
| record | topic: Subject area or focus of the submitted paper | 1 | 0.10% |
| record | status: Current processing or acceptance status | 1 | 0.10% |
| record | launched_by | 1 | 0.10% |
| record | ticket_id: Unique identifier for each support ticket | 1 | 0.10% |
| record | priority: Severity level of the issue | 1 | 0.10% |
| record | company: Company name of the ticket owner | 1 | 0.10% |
| record | impact_users: Number of users directly affected by the issue | 1 | 0.10% |
| record | location.county | 1 | 0.10% |
| record | administrative_level | 1 | 0.10% |
| record | abolished_entity | 1 | 0.10% |
| record | administrative_divisions.village_committees | 1 | 0.10% |
| record | patient_id: Unique identifier for the patient entry | 1 | 0.10% |
| record | status: Current status of vaccination record | 1 | 0.10% |
| record | contributor_location | 1 | 0.10% |
| record | shipment_id | 1 | 0.10% |
| record | agency | 1 | 0.10% |
| record | record_metadata | natural/name ×22, natural/status ×20, natural/title ×18, natural/patient_id ×14 |  |
| research_paper | title | 2 | 100.00% |
| research_paper | venue | 2 | 100.00% |
| research_paper | year | 2 | 100.00% |
| research_paper | authors | 1 | 50.00% |
| research_paper | record_metadata | natural/title ×2 |  |
| software_release | product | 6 | 100.00% |
| software_release | version | 4 | 66.67% |
| software_release | release_date | 4 | 66.67% |
| software_release | platform | 4 | 66.67% |
| software_release | record_metadata | natural/product ×6 |  |
| transaction | amount | 110 | 98.21% |
| transaction | item | 109 | 97.32% |
| transaction | date | 104 | 92.86% |
| transaction | buyer | 103 | 91.96% |
| transaction | seller | 102 | 91.07% |
| transaction | record_metadata | natural/item ×107, natural/amount ×3 |  |
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
