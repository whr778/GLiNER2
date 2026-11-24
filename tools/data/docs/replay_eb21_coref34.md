# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replay_eb21_coref34 | train | 13,990 | 13,928 | 95,973 | 6,821 | 11,787 | 702 | 8,644 | 12,034 |
| replay_eb21_coref34 | val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| replay_eb21_coref34 | test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## replay_eb21_coref34

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 13,990 | 13,928 | 324 | 1193 | 22512 | 95,973 | 6,821 | 11,787 | 702 | 8,644 | 12,034 |
| val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 7,129 | 0 | 0 | 7,129 | 50.96% |
| zho | 5,153 | 0 | 0 | 5,153 | 36.83% |
| tur | 1,412 | 0 | 0 | 1,412 | 10.09% |
| und | 206 | 0 | 0 | 206 | 1.47% |
| fra | 16 | 0 | 0 | 16 | 0.11% |
| por | 10 | 0 | 0 | 10 | 0.07% |
| ukr | 9 | 0 | 0 | 9 | 0.06% |
| rus | 9 | 0 | 0 | 9 | 0.06% |
| ita | 8 | 0 | 0 | 8 | 0.06% |
| spa | 6 | 0 | 0 | 6 | 0.04% |
| hin | 6 | 0 | 0 | 6 | 0.04% |
| deu | 6 | 0 | 0 | 6 | 0.04% |
| kor | 4 | 0 | 0 | 4 | 0.03% |
| nld | 3 | 0 | 0 | 3 | 0.02% |
| ces | 3 | 0 | 0 | 3 | 0.02% |
| pol | 3 | 0 | 0 | 3 | 0.02% |
| ara | 2 | 0 | 0 | 2 | 0.01% |
| wuu | 1 | 0 | 0 | 1 | 0.01% |
| vie | 1 | 0 | 0 | 1 | 0.01% |
| ilo | 1 | 0 | 0 | 1 | 0.01% |
| pms | 1 | 0 | 0 | 1 | 0.01% |
| jpn | 1 | 0 | 0 | 1 | 0.01% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 0 |
| test_unique_inputs | 0 |
| train_records | 13,990 |
| train_unique_inputs | 13,928 |
| val_records | 0 |
| val_unique_inputs | 0 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | n/a | 95,973 / 0 |  |
| entities (per mention) | train / test | n/a | 95,973 / 0 |  |
| relations (per instance) | train / val | n/a | 6,821 / 0 |  |
| relations (per instance) | train / test | n/a | 6,821 / 0 |  |
| event types (per trigger mention) | train / val | n/a | 9,799 / 0 |  |
| event types (per trigger mention) | train / test | n/a | 9,799 / 0 |  |
| event roles (per argument) | train / val | n/a | 12,386 / 0 |  |
| event roles (per argument) | train / test | n/a | 12,386 / 0 |  |
| classifications (true_label) | train / val | n/a | 11,787 / 0 |  |
| classifications (true_label) | train / test | n/a | 11,787 / 0 |  |


### Events

**train.** 9,474 instances in 2,611 records, 0 triggerless. Trigger gold 8,644. Argument gold 12,034. Arguments per instance, mean 1.31. Trigger length in whitespace tokens, mean 1.36, single-token share 71.9%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Contact.Communicate | 655 | 6.68% |
| Experiment | 423 | 4.32% |
| Support | 366 | 3.74% |
| Catastrophe | 242 | 2.47% |
| Sport.Compete | 201 | 2.05% |
| Attack | 192 | 1.96% |
| Causation | 175 | 1.79% |
| Hostile_encounter | 154 | 1.57% |
| Process_start | 150 | 1.53% |
| Life.Die | 144 | 1.47% |
| Motion | 124 | 1.27% |
| Deploy | 122 | 1.25% |
| Personnel.StartPosition | 112 | 1.14% |
| Competition | 102 | 1.04% |
| Statement | 97 | 0.99% |
| Exhibit | 96 | 0.98% |
| match_result | 95 | 0.97% |
| Conflict.Attack | 95 | 0.97% |
| Killing | 92 | 0.94% |
| Business.StartOrg | 92 | 0.94% |
| Conquering | 88 | 0.90% |
| Personnel.EndPosition | 84 | 0.86% |
| Cyber.DiscoverVulnerability | 84 | 0.86% |
| Self_motion | 81 | 0.83% |
| Cyber.Phishing | 79 | 0.81% |
| product_publish | 78 | 0.80% |
| Social_event | 78 | 0.80% |
| Arriving | 78 | 0.80% |
| Destroying | 77 | 0.79% |
| Contact.Meet | 76 | 0.78% |
| Cause_change_of_position_on_a_scale | 73 | 0.74% |
| Life.Injure | 73 | 0.74% |
| Coming_to_be | 72 | 0.73% |
| Damaging | 70 | 0.71% |
| Process_end | 68 | 0.69% |
| Justice.Arrest | 68 | 0.69% |
| Cyber.Databreach | 67 | 0.68% |
| Cause_change_of_strength | 66 | 0.67% |
| Death | 65 | 0.66% |
| Hold | 65 | 0.66% |
| Cyber.Ransom | 65 | 0.66% |
| Bodily_harm | 64 | 0.65% |
| Justice.ChargeIndict | 61 | 0.62% |
| Presence | 61 | 0.62% |
| Creating | 59 | 0.60% |
| Award.ReceiveAward | 59 | 0.60% |
| Know | 55 | 0.56% |
| Control | 55 | 0.56% |
| Earnings_and_losses | 54 | 0.55% |
| Transaction.TransferOwnership | 54 | 0.55% |
| Becoming | 52 | 0.53% |
| Motion_directional | 52 | 0.53% |
| Placing | 48 | 0.49% |
| Cyber.PatchVulnerability | 47 | 0.48% |
| Traveling | 46 | 0.47% |
| Business.LaunchProduct | 46 | 0.47% |
| Action | 46 | 0.47% |
| Preventing_or_letting | 45 | 0.46% |
| Influence | 45 | 0.46% |
| Justice.TrialHearing | 44 | 0.45% |
| Military_operation | 43 | 0.44% |
| judicial_action_arrest | 42 | 0.43% |
| Accident | 42 | 0.43% |
| Justice.Sentence | 42 | 0.43% |
| Name_conferral | 42 | 0.43% |
| Request | 41 | 0.42% |
| Cause_to_be_included | 41 | 0.42% |
| Change | 41 | 0.42% |
| Escaping | 40 | 0.41% |
| Getting | 39 | 0.40% |
| Disaster.Accident | 39 | 0.40% |
| Arranging | 38 | 0.39% |
| Expressing_publicly | 38 | 0.39% |
| Change_of_leadership | 38 | 0.39% |
| Choosing | 35 | 0.36% |
| Transaction.TransferMoney | 35 | 0.36% |
| Manufacturing | 34 | 0.35% |
| resignation | 34 | 0.35% |
| Judgment_communication | 34 | 0.35% |
| Removing | 34 | 0.35% |
| Defending | 33 | 0.34% |
| death | 33 | 0.34% |
| Cause_to_make_progress | 33 | 0.34% |
| Giving | 32 | 0.33% |
| Supply | 32 | 0.33% |
| Manoeuvre | 31 | 0.32% |
| Participation | 31 | 0.32% |
| Recording | 31 | 0.32% |
| Bringing | 31 | 0.32% |
| Using | 30 | 0.31% |
| Departing | 29 | 0.30% |
| Research.Publish | 29 | 0.30% |
| Conflict.Attack.DetonateExplode | 28 | 0.29% |
| Movement.Transport | 27 | 0.28% |
| Reporting | 27 | 0.28% |
| Life.BeBorn | 27 | 0.28% |
| Cause_to_amalgamate | 26 | 0.27% |
| life.die.deathcausedbyviolentevents | 26 | 0.27% |
| Supporting | 25 | 0.26% |
| Use_firearm | 25 | 0.26% |
| championship_win | 25 | 0.26% |
| Movement.Arrive | 25 | 0.26% |
| Conflict.Demonstrate | 25 | 0.26% |
| Scrutiny | 25 | 0.26% |
| Tech.Release | 23 | 0.23% |
| Building | 21 | 0.21% |
| Assistance | 21 | 0.21% |
| Perception_active | 21 | 0.21% |
| advancement | 21 | 0.21% |
| divorce | 21 | 0.21% |
| Agree_or_refuse_to_act | 21 | 0.21% |
| Aiming | 21 | 0.21% |
| Come_together | 20 | 0.20% |
| Communication | 19 | 0.19% |
| Check | 19 | 0.19% |
| Publishing | 19 | 0.19% |
| Life.Die.Unspecified | 19 | 0.19% |
| product_recall | 19 | 0.19% |
| Injure | 19 | 0.19% |
| Justice.Sue | 19 | 0.19% |
| Life.Marry | 18 | 0.18% |
| Research.Discover | 18 | 0.18% |
| Violence | 18 | 0.18% |
| Receiving | 18 | 0.18% |
| Dispersal | 18 | 0.18% |
| marriage | 18 | 0.18% |
| Collaboration | 17 | 0.17% |
| Medical.Diagnose | 17 | 0.17% |
| joined | 17 | 0.17% |
| Sending | 17 | 0.17% |
| Conflict | 17 | 0.17% |
| Connect | 17 | 0.17% |
| Research.Fund | 16 | 0.16% |
| Hiding_objects | 16 | 0.16% |
| Recovering | 16 | 0.16% |
| Disaster.NaturalDisaster | 16 | 0.16% |
| judicial_action_fine | 16 | 0.16% |
| price_limit_up | 15 | 0.15% |
| Government.Vote | 15 | 0.15% |
| Deciding | 15 | 0.15% |
| opening | 14 | 0.14% |
| judicial_action_summon | 14 | 0.14% |
| product_delist | 14 | 0.14% |
| Temporary_stay | 14 | 0.14% |
| Legal_rulings | 13 | 0.13% |
| Warning | 13 | 0.13% |
| Conflict.Attack.Unspecified | 13 | 0.13% |
| Response | 13 | 0.13% |
| Contact.Contact.Unspecified | 13 | 0.13% |
| Transaction.Donate | 12 | 0.12% |
| Hindering | 12 | 0.12% |
| Cost | 12 | 0.12% |
| GiveUp | 12 | 0.12% |
| product_release | 12 | 0.12% |
| Surrendering | 12 | 0.12% |
| Movement.Migrate | 11 | 0.11% |
| Fire | 11 | 0.11% |
| life.injure.injurycausedbyviolentevents | 11 | 0.11% |
| Confronting_problem | 11 | 0.11% |
| Achieve | 11 | 0.11% |
| Expansion | 11 | 0.11% |
| contact.discussion.meet | 11 | 0.11% |
| Life.Evacuate | 11 | 0.11% |
| layoff | 11 | 0.11% |
| Justice.Appeal | 11 | 0.11% |
| Government.Sanction | 10 | 0.10% |
| Disaster.Rescue | 10 | 0.10% |
| Arrest | 10 | 0.10% |
| Being_in_operation | 10 | 0.10% |
| movement.transportperson.preventexit | 10 | 0.10% |
| Contact.Contact.Broadcast | 10 | 0.10% |
| birthday_celebration | 10 | 0.10% |
| Becoming_a_member | 10 | 0.10% |
| Emptying | 10 | 0.10% |
| Openness | 10 | 0.10% |
| financing | 9 | 0.09% |
| Change_event_time | 9 | 0.09% |
| Criminal_investigation | 9 | 0.09% |
| apologize_to | 9 | 0.09% |
| judicial_action_prosecution | 9 | 0.09% |
| Terrorism | 9 | 0.09% |
| Committing_crime | 9 | 0.09% |
| departure | 9 | 0.09% |
| conflict.attack.n/a | 9 | 0.09% |
| Life.Injure.Unspecified | 9 | 0.09% |
| Testing | 9 | 0.09% |
| Limiting | 9 | 0.09% |
| Besieging | 9 | 0.09% |
| Preserving | 8 | 0.08% |
| sale_acquisition | 8 | 0.08% |
| Commitment | 8 | 0.08% |
| life.injure.n/a | 8 | 0.08% |
| Justice.ArrestJailDetain.Unspecified | 8 | 0.08% |
| birth | 8 | 0.08% |
| movement.transportartifact.n/a | 8 | 0.08% |
| Employment | 8 | 0.08% |
| transaction.transferownership.purchase | 8 | 0.08% |
| Justice.Fine | 8 | 0.08% |
| Change_sentiment | 8 | 0.08% |
| Commerce_sell | 8 | 0.08% |
| Business.MergeOrg | 7 | 0.07% |
| Quarreling | 7 | 0.07% |
| Explosion | 7 | 0.07% |
| Government.Protest | 7 | 0.07% |
| Revenge | 7 | 0.07% |
| justice.arrestjaildetain.arrestjaildetain | 7 | 0.07% |
| Coming_to_believe | 7 | 0.07% |
| conflict.attack.selfdirectedbattle | 7 | 0.07% |
| product_award | 7 | 0.07% |
| Medical.Treat | 7 | 0.07% |
| judicial_action_imprisonment | 7 | 0.07% |
| Robbery | 7 | 0.07% |
| competition_ban | 7 | 0.07% |
| justice.judicialconsequences.convict | 7 | 0.07% |
| Traffic Accident | 7 | 0.07% |
| judicial_action_hearing | 7 | 0.07% |
| competition_withdrawal | 7 | 0.07% |
| Create_artwork | 7 | 0.07% |
| Justice.ChargeIndict.Unspecified | 6 | 0.06% |
| Life.Divorce | 6 | 0.06% |
| contact.commitmentpromiseexpressintent.meet | 6 | 0.06% |
| Government.EnactLaw | 6 | 0.06% |
| Disaster.Outbreak | 6 | 0.06% |
| Movement.Transportation.Unspecified | 6 | 0.06% |
| Personnel.Elect | 6 | 0.06% |
| Prison | 6 | 0.06% |
| Earthquake | 6 | 0.06% |
| justice.initiatejudicialprocess.chargeindict | 6 | 0.06% |
| Research | 6 | 0.06% |
| transaction.transferownership.n/a | 6 | 0.06% |
| transaction.transaction.transfercontrol | 6 | 0.06% |
| artifactexistence.damagedestroy.destroy | 6 | 0.06% |
| Business.EndOrg | 6 | 0.06% |
| Collapse | 6 | 0.06% |
| Body_movement | 6 | 0.06% |
| government.agreements.acceptagreementcontractceasefire | 6 | 0.06% |
| Rescuing | 5 | 0.05% |
| Filling | 5 | 0.05% |
| march | 5 | 0.05% |
| life.die.n/a | 5 | 0.05% |
| price_increase | 5 | 0.05% |
| wedding | 5 | 0.05% |
| Having_or_lacking_access | 5 | 0.05% |
| conflict.attack.stabbing | 5 | 0.05% |
| Containing | 5 | 0.05% |
| movement.transportartifact.preventexit | 5 | 0.05% |
| judicial_action_filing | 5 | 0.05% |
| artifactexistence.damagedestroy.damage | 5 | 0.05% |
| Movement.Deport | 5 | 0.05% |
| government.agreements.n/a | 5 | 0.05% |
| ipo | 5 | 0.05% |
| transaction.transferownership.borrowlend | 5 | 0.05% |
| Tech.Breach | 5 | 0.05% |
| Surrounding | 5 | 0.05% |
| breakup | 5 | 0.05% |
| Resolve_problem | 5 | 0.05% |
| proposal | 5 | 0.05% |
| contract_termination | 5 | 0.05% |
| judicial_action_report | 5 | 0.05% |
| Sign_agreement | 5 | 0.05% |
| Education_teaching | 5 | 0.05% |
| price_limit_down | 5 | 0.05% |
| Contact.Negotiate | 5 | 0.05% |
| Forming_relationships | 5 | 0.05% |
| dismissal | 5 | 0.05% |
| Business.Strike | 5 | 0.05% |
| transaction.transferownership.giftgrantprovideaid | 5 | 0.05% |
| Rewards_and_punishments | 4 | 0.04% |
| justice.initiatejudicialprocess.trialhearing | 4 | 0.04% |
| thank | 4 | 0.04% |
| visit_on_set | 4 | 0.04% |
| Reveal_secret | 4 | 0.04% |
| Convincing | 4 | 0.04% |
| Transaction.ExchangeBuySell.Unspecified | 4 | 0.04% |
| personnel.elect.winelection | 4 | 0.04% |
| contact.requestadvise.correspondence | 4 | 0.04% |
| justice.judicialconsequences.execute | 4 | 0.04% |
| contact.requestadvise.broadcast | 4 | 0.04% |
| contact.prevarication.broadcast | 4 | 0.04% |
| Telling | 4 | 0.04% |
| Commerce_pay | 4 | 0.04% |
| Business.DeclareBankruptcy | 4 | 0.04% |
| movement.transportartifact.grantentry | 4 | 0.04% |
| transaction.transfermoney.purchase | 4 | 0.04% |
| transaction.transfermoney.embargosanction | 4 | 0.04% |
| contact.discussion.correspondence | 4 | 0.04% |
| manufacture.artifact.n/a | 4 | 0.04% |
| government.formation.n/a | 4 | 0.04% |
| life.injure.illnessdegradationhungerthirst | 4 | 0.04% |
| Submitting_documents | 4 | 0.04% |
| contact.discussion.n/a | 4 | 0.04% |
| meet_with | 4 | 0.04% |
| Patrolling | 4 | 0.04% |
| movement.transportperson.n/a | 4 | 0.04% |
| Personnel.Nominate | 4 | 0.04% |
| strike | 4 | 0.04% |
| movement.transportartifact.sendsupplyexport | 4 | 0.04% |
| GetReady | 4 | 0.04% |
| Protest | 4 | 0.04% |
| conflict.attack.bombing | 4 | 0.04% |
| contact.commitmentpromiseexpressintent.correspondence | 4 | 0.04% |
| disaster.accidentcrash.accidentcrash | 4 | 0.04% |
| artifactexistence.damagedestroy.n/a | 4 | 0.04% |
| Ingestion | 4 | 0.04% |
| Rite | 3 | 0.03% |
| closing | 3 | 0.03% |
| praise | 3 | 0.03% |
| conflict.attack.invade | 3 | 0.03% |
| engagement | 3 | 0.03% |
| GenericCrime.GenericCrime.GenericCrime | 3 | 0.03% |
| Releasing | 3 | 0.03% |
| Justice.InvestigateCrime.Unspecified | 3 | 0.03% |
| Flood | 3 | 0.03% |
| Writing | 3 | 0.03% |
| transaction.transaction.embargosanction | 3 | 0.03% |
| Practice | 3 | 0.03% |
| movement.transportartifact.bringcarryunload | 3 | 0.03% |
| transaction.transfermoney.n/a | 3 | 0.03% |
| movement.transportperson.disperseseparate | 3 | 0.03% |
| suspension | 3 | 0.03% |
| Bearing_arms | 3 | 0.03% |
| Award | 3 | 0.03% |
| justice.judicialconsequences.n/a | 3 | 0.03% |
| contact.commitmentpromiseexpressintent.n/a | 3 | 0.03% |
| Justice.Acquit | 3 | 0.03% |
| transaction.transfermoney.payforservice | 3 | 0.03% |
| government.formation.startgpe | 3 | 0.03% |
| Reforming_a_system | 3 | 0.03% |
| Exchange | 3 | 0.03% |
| contact.prevarication.correspondence | 3 | 0.03% |
| ArtifactExistence.DamageDestroyDisableDismantle.Damage | 3 | 0.03% |
| Adducing | 3 | 0.03% |
| Labeling | 3 | 0.03% |
| justice.investigate.investigatecrime | 3 | 0.03% |
| conflict.attack.stealrobhijack | 3 | 0.03% |
| Legality | 3 | 0.03% |
| contact.requestadvise.meet | 2 | 0.02% |
| movement.transportperson.evacuationrescue | 2 | 0.02% |
| dissolution | 2 | 0.02% |
| transaction.transfermoney.giftgrantprovideaid | 2 | 0.02% |
| transaction.transaction.giftgrantprovideaid | 2 | 0.02% |
| Carry_goods | 2 | 0.02% |
| price_decrease | 2 | 0.02% |
| government.formation.mergegpe | 2 | 0.02% |
| Expend_resource | 2 | 0.02% |
| contact.prevarication.n/a | 2 | 0.02% |
| Institutionalization | 2 | 0.02% |
| ArtifactExistence.DamageDestroyDisableDismantle.DisableDefuse | 2 | 0.02% |
| Life.Infect.Unspecified | 2 | 0.02% |
| Cognitive.Inspection.SensoryObserve | 2 | 0.02% |
| movement.transportartifact.prevententry | 2 | 0.02% |
| movement.transportperson.prevententry | 2 | 0.02% |
| contact.commandorder.meet | 2 | 0.02% |
| conflict.yield.n/a | 2 | 0.02% |
| personnel.endposition.n/a | 2 | 0.02% |
| lost_contact | 2 | 0.02% |
| infidelity | 2 | 0.02% |
| contact.collaborate.correspondence | 2 | 0.02% |
| contact.threatencoerce.n/a | 2 | 0.02% |
| contact.commitmentpromiseexpressintent.broadcast | 2 | 0.02% |
| manufacture.artifact.createmanufacture | 2 | 0.02% |
| government.agreements.violateagreement | 2 | 0.02% |
| pregnancy | 2 | 0.02% |
| movement.transportartifact.fall | 2 | 0.02% |
| movement.transportartifact.nonviolentthrowlaunch | 2 | 0.02% |
| Plane Crash | 2 | 0.02% |
| ArtifactExistence.ManufactureAssemble.Unspecified | 2 | 0.02% |
| ArtifactExistence.DamageDestroyDisableDismantle.Destroy | 2 | 0.02% |
| contact.publicstatementinperson.n/a | 2 | 0.02% |
| personnel.endposition.firinglayoff | 2 | 0.02% |
| contact.mediastatement.n/a | 2 | 0.02% |
| contact.collaborate.n/a | 2 | 0.02% |
| contact.negotiate.meet | 2 | 0.02% |
| contact.requestadvise.n/a | 2 | 0.02% |
| Wearing | 2 | 0.02% |
| inspection.sensoryobserve.physicalinvestigateinspect | 2 | 0.02% |
| Commerce_buy | 2 | 0.02% |
| retirement | 2 | 0.02% |
| Business.Layoff | 2 | 0.02% |
| movement.transportartifact.hide | 2 | 0.02% |
| conflict.demonstrate.marchprotestpoliticalgathering | 2 | 0.02% |
| personnel.endposition.quitretire | 2 | 0.02% |
| movement.transportartifact.disperseseparate | 2 | 0.02% |
| personnel.startposition.n/a | 2 | 0.02% |
| life.injure.illnessdegradationphysical | 2 | 0.02% |
| transaction.transferownership.embargosanction | 2 | 0.02% |
| rate_increase | 2 | 0.02% |
| transaction.transfermoney.borrowlend | 2 | 0.02% |
| Cognitive.IdentifyCategorize.Unspecified | 2 | 0.02% |
| Contact.Contact.Meet | 2 | 0.02% |
| disaster.fireexplosion.fireexplosion | 2 | 0.02% |
| contact.threatencoerce.meet | 2 | 0.02% |
| Imposing_obligation | 2 | 0.02% |
| Renting | 2 | 0.02% |
| Lighting | 1 | 0.01% |
| personnel.startposition.hiring | 1 | 0.01% |
| contact.collaborate.meet | 1 | 0.01% |
| justice.initiatejudicialprocess.n/a | 1 | 0.01% |
| Justice.Sentence.Unspecified | 1 | 0.01% |
| contact.funeralvigil.meet | 1 | 0.01% |
| rate_decrease | 1 | 0.01% |
| contact.funeralvigil.n/a | 1 | 0.01% |
| contact.mediastatement.broadcast | 1 | 0.01% |
| contact.commandorder.n/a | 1 | 0.01% |
| movement.transportperson.fall | 1 | 0.01% |
| contact.prevarication.meet | 1 | 0.01% |
| Change_tool | 1 | 0.01% |
| Justifying | 1 | 0.01% |
| transaction.transaction.n/a | 1 | 0.01% |
| movement.transportperson.selfmotion | 1 | 0.01% |
| Contact.Contact.Correspondence | 1 | 0.01% |
| movement.transportartifact.smuggleextract | 1 | 0.01% |
| inspection.sensoryobserve.inspectpeopleorganization | 1 | 0.01% |
| Incident | 1 | 0.01% |
| movement.transportperson.smuggleextract | 1 | 0.01% |
| movement.transportperson.hide | 1 | 0.01% |
| inspection.sensoryobserve.n/a | 1 | 0.01% |
| Justice.TrialHearing.Unspecified | 1 | 0.01% |
| Transaction.Lend | 1 | 0.01% |
| Cognitive.Research.Unspecified | 1 | 0.01% |
| Personnel.StartPosition.Unspecified | 1 | 0.01% |
| inspection.sensoryobserve.monitorelection | 1 | 0.01% |
| conflict.attack.setfire | 1 | 0.01% |
| contact.publicstatementinperson.broadcast | 1 | 0.01% |
| Scouring | 1 | 0.01% |
| contact.commandorder.correspondence | 1 | 0.01% |
| government.vote.violationspreventvote | 1 | 0.01% |
| Conflict.Ceasefire | 1 | 0.01% |
| contact.commandorder.broadcast | 1 | 0.01% |
| manufacture.artifact.build | 1 | 0.01% |
| movement.transportperson.grantentryasylum | 1 | 0.01% |
| conflict.yield.surrender | 1 | 0.01% |
| government.agreements.rejectnullifyagreementcontractceasefire | 1 | 0.01% |
| conflict.demonstrate.n/a | 1 | 0.01% |
| life.die.nonviolentdeath | 1 | 0.01% |

**val.** No event instances.

**test.** No event instances.

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 12,386 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.75 | 30.83% | 1.92 | 32.60% | 0.05% | 0 | 12,386 | 0 | 9,799 |
| val | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |
| test | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 9,474 | 2.15% | 7,481 | 11.44% | 10.72% | 2.02% | 0.82% | 12.98% |
| val | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |
| test | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

| split | docs | clusters | mentions | multi-mention | singletons | mention pairs | pronoun mentions | pronoun share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 12 | 234 | 1,054 | 234 | 0 | 5,077 | 269 | 25.52% |
| val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |
| test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 204 | 485 | 7,792 |
| val | 0 | 0 | 0 |
| test | 0 | 0 | 0 |

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

| split | mentions the string rule adds | arguments skipped, 2+ clusters | arguments in one multi-mention cluster | other mentions of that cluster | argument spans in 2+ clusters |
| --- | --- | --- | --- | --- | --- |
| train | 620 | 37 | 0 | 0 | 0 |
| val | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 |

### Entities

#### train

| label | count | share |
| --- | --- | --- |
| person | 8,557 | 8.92% |
| location | 4,870 | 5.07% |
| date | 4,317 | 4.50% |
| Location | 4,193 | 4.37% |
| Date | 3,632 | 3.78% |
| cardinal | 2,724 | 2.84% |
| organization | 2,567 | 2.67% |
| Organization | 2,273 | 2.37% |
| StartDate | 1,776 | 1.85% |
| Product | 1,728 | 1.80% |
| job title | 1,670 | 1.74% |
| EquityHolder | 1,627 | 1.70% |
| StockCode | 1,479 | 1.54% |
| Casualties and Losses | 1,458 | 1.52% |
| sports team | 1,428 | 1.49% |
| money | 1,342 | 1.40% |
| CompanyName | 1,263 | 1.32% |
| StockAbbr | 1,248 | 1.30% |
| geopolitical entity | 1,191 | 1.24% |
| PledgedShares | 1,086 | 1.13% |
| company | 1,048 | 1.09% |
| Cause | 929 | 0.97% |
| Person | 924 | 0.96% |
| TotalHoldingShares | 903 | 0.94% |
| media outlet | 903 | 0.94% |
| Pledgee | 888 | 0.93% |
| TradedShares | 886 | 0.92% |
| TotalHoldingRatio | 831 | 0.87% |
| Quantity | 791 | 0.82% |
| Reduction Amount | 692 | 0.72% |
| percent | 690 | 0.72% |
| facility | 683 | 0.71% |
| age | 663 | 0.69% |
| TotalPledgedShares | 638 | 0.66% |
| duration | 633 | 0.66% |
| product | 572 | 0.60% |
| EndDate | 543 | 0.57% |
| time | 538 | 0.56% |
| quantity | 532 | 0.55% |
| ReleasedDate | 522 | 0.54% |
| Measurement | 516 | 0.54% |
| Reducing Shareholder | 513 | 0.53% |
| occupation | 512 | 0.53% |
| government agency | 501 | 0.52% |
| work of art | 451 | 0.47% |
| PER | 415 | 0.43% |
| Announcement Date | 407 | 0.42% |
| Policy Content | 398 | 0.41% |
| Company Name | 395 | 0.41% |
| Reduction Start Date | 384 | 0.40% |
| educational institution | 382 | 0.40% |
| Money | 334 | 0.35% |
| Event | 304 | 0.32% |
| LaterHoldingShares | 303 | 0.32% |
| event | 293 | 0.31% |
| Candidates and their Political Parties | 293 | 0.31% |
| Contest Participant | 283 | 0.29% |
| AveragePrice | 267 | 0.28% |
| nationality | 264 | 0.28% |
| Protesters | 254 | 0.26% |
| Participants | 247 | 0.26% |
| Attacker | 240 | 0.25% |
| Target | 239 | 0.25% |
| Facility | 238 | 0.25% |
| Weapon and Equippment | 236 | 0.25% |
| Nationality | 236 | 0.25% |
| Suspect | 236 | 0.25% |
| pledge_amount | 231 | 0.24% |
| measurement | 228 | 0.24% |
| Areas affected | 224 | 0.23% |
| Protest Reason | 212 | 0.22% |
| ordinal | 205 | 0.21% |
| crime | 201 | 0.21% |
| Affected Areas | 201 | 0.21% |
| Deliberating Agency | 192 | 0.20% |
| pledgor | 187 | 0.19% |
| address | 186 | 0.19% |
| Affected Area | 186 | 0.19% |
| region | 183 | 0.19% |
| disease | 176 | 0.18% |
| brand | 172 | 0.18% |
| Method | 168 | 0.18% |
| Recipient | 165 | 0.17% |
| Occupation | 157 | 0.16% |
| RepurchasedShares | 155 | 0.16% |
| Disease | 153 | 0.16% |
| Person under Investigation | 151 | 0.16% |
| Charged Crime | 151 | 0.16% |
| Acquiree | 149 | 0.16% |
| distance | 148 | 0.15% |
| Accepting Court | 148 | 0.15% |
| The Charged Crime | 144 | 0.15% |
| Belligerents | 143 | 0.15% |
| sports competition | 142 | 0.15% |
| Accident Investigator | 140 | 0.15% |
| Loss Amount | 137 | 0.14% |
| HighestTradingPrice | 137 | 0.14% |
| Criminal Evidence | 135 | 0.14% |
| family relationship | 132 | 0.14% |
| protein | 132 | 0.14% |
| Precautionary Measure | 129 | 0.13% |
| Investigative Agency | 128 | 0.13% |
| Position | 126 | 0.13% |
| Game Name | 126 | 0.13% |
| Individual_protein | 125 | 0.13% |
| Time | 124 | 0.13% |
| Confirmed/Infected Cases | 119 | 0.12% |
| Accusation | 113 | 0.12% |
| GPE | 111 | 0.12% |
| ORG | 111 | 0.12% |
| road | 108 | 0.11% |
| award | 107 | 0.11% |
| RepurchaseAmount | 106 | 0.11% |
| landmark | 105 | 0.11% |
| Increase Amount | 105 | 0.11% |
| Arrest Location | 105 | 0.11% |
| Winner | 104 | 0.11% |
| chemical | 103 | 0.11% |
| pledge_start_date | 103 | 0.11% |
| DiseaseOrPhenotypicFeature | 101 | 0.11% |
| Summit Theme | 101 | 0.11% |
| Champions | 100 | 0.10% |
| Frozen Amount | 99 | 0.10% |
| Election Name | 98 | 0.10% |
| Increasing Shareholder | 95 | 0.10% |
| ClosingDate | 95 | 0.10% |
| Award | 94 | 0.10% |
| ruling_date | 93 | 0.10% |
| Accused People | 93 | 0.10% |
| Death Cases | 93 | 0.10% |
| FAC | 93 | 0.10% |
| era | 92 | 0.10% |
| frozen_shareholder | 91 | 0.09% |
| GeneOrGeneProduct | 91 | 0.09% |
| film | 90 | 0.09% |
| political affiliation | 89 | 0.09% |
| Scheduled Landing Place | 87 | 0.09% |
| Score | 87 | 0.09% |
| Aid Agency | 86 | 0.09% |
| Contracting Parties | 85 | 0.09% |
| team role | 85 | 0.09% |
| Storm Hit Location | 85 | 0.09% |
| Vulnerability | 84 | 0.09% |
| non-profit organization | 81 | 0.08% |
| Strike Reason | 81 | 0.08% |
| The Final Votes and Percentages | 80 | 0.08% |
| Police | 79 | 0.08% |
| law | 78 | 0.08% |
| medication | 77 | 0.08% |
| Flight No. | 77 | 0.08% |
| LowestTradingPrice | 77 | 0.08% |
| Location/Hospital | 77 | 0.08% |
| Policy Name & Abbreviation | 76 | 0.08% |
| Influence People | 76 | 0.08% |
| GeneProtein | 75 | 0.08% |
| military rank | 74 | 0.08% |
| Artifacts and Their Chronology | 74 | 0.08% |
| Strike Outcome | 74 | 0.08% |
| Economic Loss | 72 | 0.08% |
| academic degree | 71 | 0.07% |
| Aircraft Agency | 70 | 0.07% |
| Judgement Result/Prison Term | 70 | 0.07% |
| Capabilities | 70 | 0.07% |
| species | 69 | 0.07% |
| Deceased | 69 | 0.07% |
| pledge_end_date | 68 | 0.07% |
| regulatory body | 67 | 0.07% |
| publication | 67 | 0.07% |
| Crew | 67 | 0.07% |
| Epidemic Data Issuing Agency | 67 | 0.07% |
| Speaker | 66 | 0.07% |
| Protest Slogan | 66 | 0.07% |
| Host | 65 | 0.07% |
| ethnic group | 65 | 0.07% |
| temperature | 64 | 0.07% |
| Anti-pollution People/Organizations | 64 | 0.07% |
| Profession | 63 | 0.07% |
| Outbreak Location | 63 | 0.07% |
| Record-breaking Project | 61 | 0.06% |
| Husband | 61 | 0.06% |
| software | 59 | 0.06% |
| Resignated Person | 59 | 0.06% |
| Increase Start Date | 59 | 0.06% |
| OtherScientificTerm | 59 | 0.06% |
| Speaker Status | 58 | 0.06% |
| Freeze Start Date | 58 | 0.06% |
| Grades | 58 | 0.06% |
| Damaged Facility | 57 | 0.06% |
| Record Breaker | 57 | 0.06% |
| Strike Agency | 57 | 0.06% |
| Task | 56 | 0.06% |
| ChemicalEntity | 56 | 0.06% |
| Effective Date | 55 | 0.06% |
| System | 55 | 0.06% |
| Award Field | 54 | 0.06% |
| Magnitude | 54 | 0.06% |
| Discover Location | 54 | 0.06% |
| PII | 54 | 0.06% |
| Survivors | 53 | 0.06% |
| Appointee | 53 | 0.06% |
| Court | 53 | 0.06% |
| vehicle | 52 | 0.05% |
| Pollution Source | 52 | 0.05% |
| Number of Vehicles Involved in the Crash | 51 | 0.05% |
| military unit | 51 | 0.05% |
| Sponsor | 51 | 0.05% |
| Solution | 51 | 0.05% |
| Policy Proposals | 51 | 0.05% |
| Agreement Content | 50 | 0.05% |
| Competition Items | 50 | 0.05% |
| Spectacle Start Time | 49 | 0.05% |
| Wife | 49 | 0.05% |
| weapon | 48 | 0.05% |
| body of water | 48 | 0.05% |
| Time of the Case | 48 | 0.05% |
| Start Date | 48 | 0.05% |
| Resign Reason | 47 | 0.05% |
| Spectacle Location | 47 | 0.05% |
| Arrest Time | 47 | 0.05% |
| name | 47 | 0.05% |
| Participating Countries | 47 | 0.05% |
| Spacecraft | 46 | 0.05% |
| Employment Agency | 45 | 0.05% |
| song | 45 | 0.05% |
| holiday | 45 | 0.05% |
| frequency | 44 | 0.05% |
| pseudonym | 44 | 0.05% |
| Affected Industries | 44 | 0.05% |
| political party | 44 | 0.05% |
| News Release Agency | 44 | 0.05% |
| Funding | 44 | 0.05% |
| Beneficiary | 44 | 0.05% |
| Types of the Spectacle | 44 | 0.05% |
| Released People | 44 | 0.05% |
| Epicenter | 44 | 0.05% |
| Attending Hospital | 44 | 0.05% |
| Prosecutor | 44 | 0.05% |
| Microorganism | 44 | 0.05% |
| Conflict Duration | 43 | 0.04% |
| People | 43 | 0.04% |
| Visitor | 42 | 0.04% |
| anatomical structure | 42 | 0.04% |
| Previous Record Holder | 42 | 0.04% |
| WEA | 41 | 0.04% |
| symptom | 40 | 0.04% |
| Country Visited | 40 | 0.04% |
| speed | 40 | 0.04% |
| File | 40 | 0.04% |
| honorific | 39 | 0.04% |
| model number | 39 | 0.04% |
| Anatomy | 39 | 0.04% |
| CellType | 39 | 0.04% |
| Number of Evacuated People | 38 | 0.04% |
| Host Country | 38 | 0.04% |
| Age | 38 | 0.04% |
| Strikers Status | 38 | 0.04% |
| court case | 37 | 0.04% |
| Archaeologist | 37 | 0.04% |
| Start Time | 37 | 0.04% |
| Arrested | 37 | 0.04% |
| Death Reason | 37 | 0.04% |
| Military Strength | 36 | 0.04% |
| Way for Spreading | 36 | 0.04% |
| Fine Reason | 36 | 0.04% |
| religious group | 35 | 0.04% |
| The Grades of the Previous Record Holder | 35 | 0.04% |
| LegalInstitution | 35 | 0.04% |
| LOC | 35 | 0.04% |
| Weapon | 35 | 0.04% |
| regulation | 34 | 0.04% |
| Election Goal | 34 | 0.04% |
| Achievement | 34 | 0.04% |
| book | 34 | 0.04% |
| OrganismTaxon | 34 | 0.04% |
| Generic | 34 | 0.04% |
| Army | 34 | 0.04% |
| Riot Reason | 33 | 0.03% |
| protein-family | 33 | 0.03% |
| Freeze End Date | 33 | 0.03% |
| Number of Damaged Houses | 32 | 0.03% |
| Passengers | 32 | 0.03% |
| Victim | 32 | 0.03% |
| Data | 32 | 0.03% |
| Head of Institution | 32 | 0.03% |
| Habitat | 32 | 0.03% |
| area | 31 | 0.03% |
| Areas Affected | 31 | 0.03% |
| Material | 31 | 0.03% |
| Device | 31 | 0.03% |
| Research Agency | 30 | 0.03% |
| Gene/protein/RNA | 30 | 0.03% |
| FrozeShares | 30 | 0.03% |
| Party Responsible for Pollution | 30 | 0.03% |
| Discover Time | 29 | 0.03% |
| Bank Name | 29 | 0.03% |
| currency | 28 | 0.03% |
| Agreement Name | 28 | 0.03% |
| Historical Sites | 28 | 0.03% |
| Summit Name | 28 | 0.03% |
| Award Reason | 27 | 0.03% |
| Employed Institution | 27 | 0.03% |
| Government Reaction | 27 | 0.03% |
| Spokeswoman/Spokesman | 27 | 0.03% |
| Spacecraft Mission | 27 | 0.03% |
| Storm Warning Level | 27 | 0.03% |
| Organization Name | 27 | 0.03% |
| Evidence | 27 | 0.03% |
| Investigating agency | 27 | 0.03% |
| sentence | 26 | 0.03% |
| Aid Reason | 26 | 0.03% |
| senior_personnel | 26 | 0.03% |
| People/Organization who predicted the disaster | 26 | 0.03% |
| Symptoms | 26 | 0.03% |
| Taking-off Place | 26 | 0.03% |
| Defense Lawyer | 26 | 0.03% |
| compensation_recipient | 25 | 0.03% |
| senior_position | 25 | 0.03% |
| SequenceVariant | 25 | 0.03% |
| Patch | 25 | 0.03% |
| Software | 25 | 0.03% |
| Last Job of the Appointee | 25 | 0.03% |
| Perpetrators | 25 | 0.03% |
| Region | 25 | 0.03% |
| Launch Date | 24 | 0.03% |
| Purpose | 24 | 0.03% |
| Military Exercise | 24 | 0.03% |
| market index | 23 | 0.02% |
| criminal organization | 23 | 0.02% |
| organism | 23 | 0.02% |
| Head of Investigation Team | 23 | 0.02% |
| Susceptible Population | 23 | 0.02% |
| Cooperative Agency | 23 | 0.02% |
| Strikers | 23 | 0.02% |
| Boycotted Institutions | 23 | 0.02% |
| Law | 23 | 0.02% |
| fiscal quarter | 22 | 0.02% |
| Volcano Name | 22 | 0.02% |
| weight | 22 | 0.02% |
| Bill Drafting Agency | 22 | 0.02% |
| conference | 22 | 0.02% |
| Regulatory Authority | 22 | 0.02% |
| Poisoning Type | 22 | 0.02% |
| Appointer | 22 | 0.02% |
| Judge | 22 | 0.02% |
| Launch Site | 22 | 0.02% |
| Maximum Rainfall | 22 | 0.02% |
| language | 21 | 0.02% |
| operating system | 21 | 0.02% |
| End Date | 21 | 0.02% |
| Outbreak Date | 20 | 0.02% |
| Shipwreck Reason | 20 | 0.02% |
| compensation_amount | 20 | 0.02% |
| Storm Name | 20 | 0.02% |
| Symptom | 20 | 0.02% |
| Commanders and their Position | 20 | 0.02% |
| medical procedure | 19 | 0.02% |
| Archaeologist Organization | 19 | 0.02% |
| End Time | 19 | 0.02% |
| Maximum Wind Speed | 19 | 0.02% |
| Launch Country | 19 | 0.02% |
| Area Affected | 19 | 0.02% |
| DNA | 18 | 0.02% |
| Stolen Amount | 18 | 0.02% |
| CellLine | 18 | 0.02% |
| Aid Supplies/Amount | 18 | 0.02% |
| Number of People Hospitalized | 18 | 0.02% |
| Weapon Used | 18 | 0.02% |
| Damaged Property | 17 | 0.02% |
| title | 17 | 0.02% |
| Turnout(who win who lost) | 17 | 0.02% |
| Fined Agency | 17 | 0.02% |
| Countries Withdrawing from the Organization | 17 | 0.02% |
| Malware | 17 | 0.02% |
| Species | 17 | 0.02% |
| Predecessor | 17 | 0.02% |
| Spokenmen | 17 | 0.02% |
| description | 17 | 0.02% |
| Treatment Method | 17 | 0.02% |
| stock ticker | 16 | 0.02% |
| dosage | 16 | 0.02% |
| Publication | 16 | 0.02% |
| Last Time the Record was Broken | 16 | 0.02% |
| Prosecution Lawyer | 16 | 0.02% |
| The Sentence Claimed by the Prosecutor's Lawyer | 16 | 0.02% |
| Unemployed Rate | 16 | 0.02% |
| Acquirer | 16 | 0.02% |
| Number of Rescued People | 15 | 0.02% |
| Death Time | 15 | 0.02% |
| Sentencing Location | 15 | 0.02% |
| Release Reason | 15 | 0.02% |
| Rescue Organizer | 15 | 0.02% |
| Scale | 15 | 0.02% |
| MVP | 15 | 0.02% |
| Appointment Time | 15 | 0.02% |
| Related Rivers or Lakes | 15 | 0.02% |
| Court Time | 15 | 0.02% |
| Website | 15 | 0.02% |
| Number of Destroyed Building | 15 | 0.02% |
| Missings | 14 | 0.01% |
| Voting method | 14 | 0.01% |
| Forecasting Agency | 14 | 0.01% |
| airport | 14 | 0.01% |
| Lasting Time | 14 | 0.01% |
| Wedding Venue | 14 | 0.01% |
| Penalty Amount | 14 | 0.01% |
| The Agency Who Broke the Agreement | 14 | 0.01% |
| Declarer | 14 | 0.01% |
| Storm Hit Time | 14 | 0.01% |
| Approver | 14 | 0.01% |
| Marriage Duration | 14 | 0.01% |
| Tribe | 14 | 0.01% |
| Number of Victims | 13 | 0.01% |
| Spectacle End Time | 13 | 0.01% |
| Pests | 13 | 0.01% |
| Number of Trapped People | 13 | 0.01% |
| Other Loss | 13 | 0.01% |
| Company Industry | 13 | 0.01% |
| Death Rate | 13 | 0.01% |
| Astronauts | 13 | 0.01% |
| Organization Industry | 13 | 0.01% |
| Strike Industry | 13 | 0.01% |
| Doctor and Medical Team | 13 | 0.01% |
| Concept | 13 | 0.01% |
| id | 13 | 0.01% |
| Ship No. | 13 | 0.01% |
| Goal | 13 | 0.01% |
| Ethnic Group | 13 | 0.01% |
| Start_Date | 13 | 0.01% |
| Number of Aftershocks | 13 | 0.01% |
| hardware component | 12 | 0.01% |
| Spectacle Duration | 12 | 0.01% |
| protein-complex | 12 | 0.01% |
| The Final Seats Result | 12 | 0.01% |
| protein-domain | 12 | 0.01% |
| Electoral System | 12 | 0.01% |
| Launch Result | 12 | 0.01% |
| Detention Start Time | 12 | 0.01% |
| Train No. | 12 | 0.01% |
| Perpetrator | 12 | 0.01% |
| Illness | 12 | 0.01% |
| Refuge | 12 | 0.01% |
| postal code | 11 | 0.01% |
| treaty | 11 | 0.01% |
| trade union | 11 | 0.01% |
| Jail Time | 11 | 0.01% |
| VAL | 11 | 0.01% |
| Age of the Appointee | 11 | 0.01% |
| Duration | 11 | 0.01% |
| State Before Death | 11 | 0.01% |
| Carrier Rocket | 11 | 0.01% |
| Casualty Count | 11 | 0.01% |
| Transportation | 11 | 0.01% |
| Military Unit | 11 | 0.01% |
| Country | 10 | 0.01% |
| natural disaster | 10 | 0.01% |
| war | 10 | 0.01% |
| financial instrument | 10 | 0.01% |
| deadline | 10 | 0.01% |
| Successor | 10 | 0.01% |
| role | 10 | 0.01% |
| Rescue Tool or Method | 10 | 0.01% |
| Version | 10 | 0.01% |
| Water Level | 10 | 0.01% |
| VEH | 10 | 0.01% |
| Acquisition amount | 10 | 0.01% |
| Term of Office | 10 | 0.01% |
| Property Division | 10 | 0.01% |
| witness | 9 | 0.01% |
| Reasons for the Formation of the Historical sites | 9 | 0.01% |
| Inauguration Time | 9 | 0.01% |
| Group | 9 | 0.01% |
| experimental-construct | 9 | 0.01% |
| Number | 9 | 0.01% |
| Gene | 9 | 0.01% |
| Witness | 9 | 0.01% |
| Hostage | 9 | 0.01% |
| Projects | 9 | 0.01% |
| Influenced People | 9 | 0.01% |
| Age of the Resignated Person | 9 | 0.01% |
| content | 9 | 0.01% |
| disclosure | 9 | 0.01% |
| page | 9 | 0.01% |
| environment | 9 | 0.01% |
| mountain | 8 | 0.01% |
| Influenced Crops and Livelihood | 8 | 0.01% |
| Prison Term | 8 | 0.01% |
| Release Time | 8 | 0.01% |
| COM | 8 | 0.01% |
| Amount of Precipitation | 8 | 0.01% |
| Number of Influenced People | 8 | 0.01% |
| Body Part | 8 | 0.01% |
| Ways to Watch the Speech | 8 | 0.01% |
| Disaster-Stricken Farmland | 8 | 0.01% |
| RNA | 8 | 0.01% |
| author | 8 | 0.01% |
| video game | 7 | 0.01% |
| database | 7 | 0.01% |
| Number of Damaged House | 7 | 0.01% |
| aircraft | 7 | 0.01% |
| clinical trial | 7 | 0.01% |
| Trapped Depth | 7 | 0.01% |
| city | 7 | 0.01% |
| state | 7 | 0.01% |
| ai_model | 7 | 0.01% |
| Storm Direction | 7 | 0.01% |
| Last Outbreak Time | 7 | 0.01% |
| Animal | 7 | 0.01% |
| Phenotype | 7 | 0.01% |
| Bankrupt Business | 7 | 0.01% |
| Social Group | 7 | 0.01% |
| Annoncement Date | 7 | 0.01% |
| rating | 7 | 0.01% |
| cell | 6 | 0.01% |
| Temporary Settlement | 6 | 0.01% |
| Protein_family_or_group | 6 | 0.01% |
| Damaged Crops & Livestock | 6 | 0.01% |
| mathematician | 6 | 0.01% |
| mathematical_result | 6 | 0.01% |
| Organization Leader | 6 | 0.01% |
| unit | 6 | 0.01% |
| Mission Duration | 6 | 0.01% |
| Taxon | 6 | 0.01% |
| PaymentMethod | 6 | 0.01% |
| Warning Device | 6 | 0.01% |
| City | 6 | 0.01% |
| Economic loss | 6 | 0.01% |
| Child | 6 | 0.01% |
| religious_organization | 6 | 0.01% |
| Wedding Dress Designer | 6 | 0.01% |
| problem_class | 6 | 0.01% |
| constraint_property | 6 | 0.01% |
| Medical Condition | 6 | 0.01% |
| algorithm | 5 | 0.01% |
| Provisional Government | 5 | 0.01% |
| Service Years | 5 | 0.01% |
| protocol | 5 | 0.01% |
| pathogen | 5 | 0.01% |
| Alternate Landing Place | 5 | 0.01% |
| UnfrozeDate | 5 | 0.01% |
| Merger Terms | 5 | 0.01% |
| CellComponent | 5 | 0.01% |
| Agreement Validity Period | 5 | 0.01% |
| Number of Vaccinated People | 5 | 0.01% |
| Storm Movement Speed | 5 | 0.01% |
| Casualities and Losses | 5 | 0.01% |
| Recovered Amount | 5 | 0.01% |
| The State of the Volcano (Dormant or Active) | 5 | 0.01% |
| Fire Warning Level | 5 | 0.01% |
| experiment-tag | 5 | 0.01% |
| CVE | 5 | 0.01% |
| researcher | 5 | 0.01% |
| keywords | 5 | 0.01% |
| Agency | 5 | 0.01% |
| Metric | 5 | 0.01% |
| Registered Capital | 5 | 0.01% |
| Invited Person | 5 | 0.01% |
| comment | 5 | 0.01% |
| symptoms | 5 | 0.01% |
| Tic Type | 5 | 0.01% |
| People Group | 5 | 0.01% |
| tax | 4 | 0.00% |
| framework | 4 | 0.00% |
| reagent | 4 | 0.00% |
| patent | 4 | 0.00% |
| process | 4 | 0.00% |
| Head of the Merged Organization | 4 | 0.00% |
| metric | 4 | 0.00% |
| Substance | 4 | 0.00% |
| Plant | 4 | 0.00% |
| Other Impact | 4 | 0.00% |
| specificDataElements | 4 | 0.00% |
| Countries Joined the Organization | 4 | 0.00% |
| Cured Cases | 4 | 0.00% |
| mutation | 4 | 0.00% |
| Investigating Spokesperson | 4 | 0.00% |
| benefits | 4 | 0.00% |
| price | 4 | 0.00% |
| productFeatures | 4 | 0.00% |
| fusion-protein | 4 | 0.00% |
| vessel | 3 | 0.00% |
| border | 3 | 0.00% |
| Best Way to Shoot | 3 | 0.00% |
| Trapped Days | 3 | 0.00% |
| publicationDate | 3 | 0.00% |
| Special Medicine | 3 | 0.00% |
| assay | 3 | 0.00% |
| streetAddress | 3 | 0.00% |
| mathematical_concept | 3 | 0.00% |
| phone | 3 | 0.00% |
| Tear Up Reason | 3 | 0.00% |
| amount | 3 | 0.00% |
| Tsunamis | 3 | 0.00% |
| Tsunami Warning Level | 3 | 0.00% |
| Head of the Institution | 3 | 0.00% |
| Economists who Predicted the Crisis | 3 | 0.00% |
| Stamp | 3 | 0.00% |
| Value | 3 | 0.00% |
| protein-motif | 3 | 0.00% |
| Nation | 3 | 0.00% |
| The Worst-Hit Area | 3 | 0.00% |
| GRIStandard | 3 | 0.00% |
| UNGCPrinciple | 3 | 0.00% |
| SASBStandard | 3 | 0.00% |
| yield | 3 | 0.00% |
| Storm Formation Location | 3 | 0.00% |
| End_Date | 3 | 0.00% |
| software_library | 3 | 0.00% |
| hardware | 3 | 0.00% |
| Head of the Government | 2 | 0.00% |
| Commanders of the Army | 2 | 0.00% |
| voltage | 2 | 0.00% |
| file format | 2 | 0.00% |
| contract | 2 | 0.00% |
| security vulnerability | 2 | 0.00% |
| issueNumber | 2 | 0.00% |
| publisher | 2 | 0.00% |
| Withdraw Reason | 2 | 0.00% |
| Book | 2 | 0.00% |
| People Casting Key Votes | 2 | 0.00% |
| geometric_object | 2 | 0.00% |
| numerical_function | 2 | 0.00% |
| mathematical_structure | 2 | 0.00% |
| Response Measures | 2 | 0.00% |
| street1 | 2 | 0.00% |
| postalCode | 2 | 0.00% |
| email | 2 | 0.00% |
| Postpone Reason | 2 | 0.00% |
| technical_method | 2 | 0.00% |
| Sequelae | 2 | 0.00% |
| Prison | 2 | 0.00% |
| Train Agency | 2 | 0.00% |
| contactEmail | 2 | 0.00% |
| startDate | 2 | 0.00% |
| endDate | 2 | 0.00% |
| authors | 2 | 0.00% |
| link | 2 | 0.00% |
| Magnitude(Tsunami heights) | 2 | 0.00% |
| gender | 2 | 0.00% |
| Candidate Scandal(Allegations of fraud, etc.) | 2 | 0.00% |
| Document | 2 | 0.00% |
| Religion | 2 | 0.00% |
| Geographic Feature | 2 | 0.00% |
| Plant Species | 2 | 0.00% |
| Feature | 2 | 0.00% |
| Civilization | 2 | 0.00% |
| Time Period | 2 | 0.00% |
| Writing System | 2 | 0.00% |
| Vaccine Research and Development Organization | 2 | 0.00% |
| Organization Members | 2 | 0.00% |
| equipmentName | 2 | 0.00% |
| manufacturer | 2 | 0.00% |
| serialNumber | 2 | 0.00% |
| modelNumber | 2 | 0.00% |
| Announce Platform | 2 | 0.00% |
| religious_event | 2 | 0.00% |
| Child Custody | 2 | 0.00% |
| Pest | 2 | 0.00% |
| Equipment | 2 | 0.00% |
| Damage | 2 | 0.00% |
| INF | 2 | 0.00% |
| agent | 2 | 0.00% |
| target | 2 | 0.00% |
| Rescue Start Time | 2 | 0.00% |
| actor | 2 | 0.00% |
| genres | 2 | 0.00% |
| mathematical_object | 2 | 0.00% |
| formalism | 2 | 0.00% |
| zipCode | 2 | 0.00% |
| expirationDate | 2 | 0.00% |
| itemId | 2 | 0.00% |
| unitPrice | 2 | 0.00% |
| Culture | 2 | 0.00% |
| Site | 2 | 0.00% |
| protein-region | 2 | 0.00% |
| Therapy | 2 | 0.00% |
| Storm Center Location | 2 | 0.00% |
| Medical Test | 2 | 0.00% |
| Ship | 2 | 0.00% |
| url | 2 | 0.00% |
| tags | 2 | 0.00% |
| Location/Nation | 2 | 0.00% |
| mineral | 1 | 0.00% |
| verdict | 1 | 0.00% |
| programming language | 1 | 0.00% |
| api | 1 | 0.00% |
| license | 1 | 0.00% |
| diagnosis | 1 | 0.00% |
| coverPrice | 1 | 0.00% |
| synopsis | 1 | 0.00% |
| State of the Hull | 1 | 0.00% |
| MON | 1 | 0.00% |
| Field of Study | 1 | 0.00% |
| businessName | 1 | 0.00% |
| zip | 1 | 0.00% |
| sharesOwned | 1 | 0.00% |
| votingProcedures | 1 | 0.00% |
| capitalContributions | 1 | 0.00% |
| disputeResolution | 1 | 0.00% |
| publication_source | 1 | 0.00% |
| How Many Times Get Married | 1 | 0.00% |
| labelFormat | 1 | 0.00% |
| country | 1 | 0.00% |
| value | 1 | 0.00% |
| length | 1 | 0.00% |
| width | 1 | 0.00% |
| height | 1 | 0.00% |
| contentsExplanation | 1 | 0.00% |
| customsValueAmount | 1 | 0.00% |
| customsValueCurrency | 1 | 0.00% |
| countryOfOrigin | 1 | 0.00% |
| dataset | 1 | 0.00% |
| architectural_component | 1 | 0.00% |
| Source | 1 | 0.00% |
| Agreement Members | 1 | 0.00% |
| Biological Entity | 1 | 0.00% |
| Compensation | 1 | 0.00% |
| Percentage | 1 | 0.00% |
| exchangeRate | 1 | 0.00% |
| Hull Discovery Time | 1 | 0.00% |
| requestNumber | 1 | 0.00% |
| requestTitle | 1 | 0.00% |
| contactPerson | 1 | 0.00% |
| affiliation | 1 | 0.00% |
| projectStartDate | 1 | 0.00% |
| projectEndDate | 1 | 0.00% |
| numberOfSubjects | 1 | 0.00% |
| approvalNumber | 1 | 0.00% |
| approvalDate | 1 | 0.00% |
| ethicsCommittee | 1 | 0.00% |
| presentationDate | 1 | 0.00% |
| minAge | 1 | 0.00% |
| maxAge | 1 | 0.00% |
| fundingSource | 1 | 0.00% |
| conclusions | 1 | 0.00% |
| Land Designation | 1 | 0.00% |
| Designation | 1 | 0.00% |
| Lawyer | 1 | 0.00% |
| Language Family | 1 | 0.00% |
| Language Group | 1 | 0.00% |
| Plant Group | 1 | 0.00% |
| Habitat Feature | 1 | 0.00% |
| Chemical | 1 | 0.00% |
| Stamp Version | 1 | 0.00% |
| Suspected Cases | 1 | 0.00% |
| Role | 1 | 0.00% |
| Alliance | 1 | 0.00% |
| Government System | 1 | 0.00% |
| Course | 1 | 0.00% |
| Protein_complex | 1 | 0.00% |
| Year | 1 | 0.00% |
| Cultural Artifact | 1 | 0.00% |
| Legal Concept | 1 | 0.00% |
| Practice | 1 | 0.00% |
| Economic System | 1 | 0.00% |
| Identity | 1 | 0.00% |
| Organization Affiliation | 1 | 0.00% |
| procedureId | 1 | 0.00% |
| department | 1 | 0.00% |
| disposalMethod | 1 | 0.00% |
| reasonForDisposal | 1 | 0.00% |
| additionalInfo | 1 | 0.00% |
| Death Age | 1 | 0.00% |
| administrative_body | 1 | 0.00% |
| educational_institution | 1 | 0.00% |
| font | 1 | 0.00% |
| fontSize | 1 | 0.00% |
| color | 1 | 0.00% |
| employeeTurnoverRate | 1 | 0.00% |
| coverArtTitle | 1 | 0.00% |
| coverArtImage | 1 | 0.00% |
| coverArtDescription | 1 | 0.00% |
| series | 1 | 0.00% |
| Invalid Date | 1 | 0.00% |
| Scientific Name | 1 | 0.00% |
| Control Method | 1 | 0.00% |
| Strategy | 1 | 0.00% |
| Natural Enemy | 1 | 0.00% |
| Lure | 1 | 0.00% |
| Cured Rate | 1 | 0.00% |
| productionDate | 1 | 0.00% |
| totalQuantityProduced | 1 | 0.00% |
| totalDuration | 1 | 0.00% |
| overallYield | 1 | 0.00% |
| Cost | 1 | 0.00% |
| year | 1 | 0.00% |
| director | 1 | 0.00% |
| plot | 1 | 0.00% |
| poster | 1 | 0.00% |
| complexity_class | 1 | 0.00% |
| academic_field | 1 | 0.00% |
| device_id | 1 | 0.00% |
| version | 1 | 0.00% |
| binary | 1 | 0.00% |
| digest | 1 | 0.00% |
| updated_at | 1 | 0.00% |
| Join Conditions | 1 | 0.00% |
| Postpone Time | 1 | 0.00% |
| firstName | 1 | 0.00% |
| lastName | 1 | 0.00% |
| phoneNumber | 1 | 0.00% |
| emailAddress | 1 | 0.00% |
| website | 1 | 0.00% |
| equityOffered | 1 | 0.00% |
| quotationId | 1 | 0.00% |
| supplierId | 1 | 0.00% |
| supplierName | 1 | 0.00% |
| quotationDate | 1 | 0.00% |
| totalCost | 1 | 0.00% |
| termsAndConditions | 1 | 0.00% |
| Artifact | 1 | 0.00% |
| Structure | 1 | 0.00% |
| batchNumber | 1 | 0.00% |
| reviewDate | 1 | 0.00% |
| reviewComment | 1 | 0.00% |
| reviewStatus | 1 | 0.00% |
| Disorder | 1 | 0.00% |
| Disorder Group | 1 | 0.00% |
| Demographic | 1 | 0.00% |
| Toxin | 1 | 0.00% |
| Character | 1 | 0.00% |
| Theory | 1 | 0.00% |
| reporter | 1 | 0.00% |
| Classification | 1 | 0.00% |
| productId | 1 | 0.00% |
| productPrice | 1 | 0.00% |
| Company | 1 | 0.00% |
| hotelId | 1 | 0.00% |
| lat | 1 | 0.00% |
| lng | 1 | 0.00% |
| roomTypeId | 1 | 0.00% |
| capacity | 1 | 0.00% |
| reservationId | 1 | 0.00% |
| numGuests | 1 | 0.00% |
| totalPrice | 1 | 0.00% |
| status | 1 | 0.00% |
| Organization_Other | 1 | 0.00% |
| protein-isoform | 1 | 0.00% |
| network | 1 | 0.00% |
| season | 1 | 0.00% |
| episode | 1 | 0.00% |
| source | 1 | 0.00% |
| Lesson Title | 1 | 0.00% |
| Scripture Reference | 1 | 0.00% |
| Location/Structure | 1 | 0.00% |
| Organization/Website | 1 | 0.00% |
| Scientific Concept | 1 | 0.00% |
| Scientific Instrument | 1 | 0.00% |
| Symbol/Object | 1 | 0.00% |
| Object | 1 | 0.00% |
| Person/Entity | 1 | 0.00% |
| Scripture | 1 | 0.00% |
| Object/Concept | 1 | 0.00% |
| Location/Concept | 1 | 0.00% |
| Complications | 1 | 0.00% |
| Organization_Military | 1 | 0.00% |
| programming_language | 1 | 0.00% |
| organelle | 1 | 0.00% |
| recipientName | 1 | 0.00% |
| recipientEmail | 1 | 0.00% |
| senderName | 1 | 0.00% |
| senderEmail | 1 | 0.00% |
| message | 1 | 0.00% |
| product_id | 1 | 0.00% |
| num_reviews | 1 | 0.00% |
| review_sentiment | 1 | 0.00% |
| budget line | 0 | 0.00% |
| interest rate | 0 | 0.00% |
| gene | 0 | 0.00% |
| credit rating | 0 | 0.00% |
| vaccine | 0 | 0.00% |
| allergen | 0 | 0.00% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| no relation | 683 | 10.01% |
| works_for | 391 | 5.73% |
| located_in | 222 | 3.25% |
| bind | 211 | 3.09% |
| member_of | 168 | 2.46% |
| headquartered_in | 134 | 1.96% |
| PPI | 133 | 1.95% |
| affiliated_with | 130 | 1.91% |
| leader_of | 125 | 1.83% |
| Association | 122 | 1.79% |
| based_in | 121 | 1.77% |
| employer_of | 107 | 1.57% |
| instance of | 102 | 1.50% |
| USED-FOR | 89 | 1.30% |
| competitor_of | 77 | 1.13% |
| founded | 73 | 1.07% |
| produced_by | 63 | 0.92% |
| part_of | 58 | 0.85% |
| country | 57 | 0.84% |
| Lives_In | 57 | 0.84% |
| occupation | 53 | 0.78% |
| shares border with | 49 | 0.72% |
| operates_in | 48 | 0.70% |
| family name | 46 | 0.67% |
| Positive_Correlation | 45 | 0.66% |
| given name | 43 | 0.63% |
| located in the administrative territorial entity | 41 | 0.60% |
| colleague_of | 39 | 0.57% |
| represents | 36 | 0.53% |
| has part(s) | 34 | 0.50% |
| reports_to | 34 | 0.50% |
| founded by | 34 | 0.50% |
| genre | 31 | 0.45% |
| member of | 30 | 0.44% |
| director | 29 | 0.43% |
| participated_in | 29 | 0.43% |
| acquired | 28 | 0.41% |
| Negative_Correlation | 28 | 0.41% |
| CONJUNCTION | 26 | 0.38% |
| place of birth | 25 | 0.37% |
| platform | 25 | 0.37% |
| supplier_of | 23 | 0.34% |
| contains the administrative territorial entity | 23 | 0.34% |
| location | 22 | 0.32% |
| child | 22 | 0.32% |
| citizen_of | 22 | 0.32% |
| followed by | 21 | 0.31% |
| replaces | 21 | 0.31% |
| educated at | 20 | 0.29% |
| headquarters location | 20 | 0.29% |
| father | 20 | 0.29% |
| performer | 18 | 0.26% |
| different from | 18 | 0.26% |
| part of | 18 | 0.26% |
| participant | 18 | 0.26% |
| employer | 17 | 0.25% |
| award received | 17 | 0.25% |
| spouse | 17 | 0.25% |
| subsidiary_of | 17 | 0.25% |
| country of citizenship | 17 | 0.25% |
| relative_of | 17 | 0.25% |
| languages spoken, written or signed | 16 | 0.23% |
| connects with | 16 | 0.23% |
| position held | 16 | 0.23% |
| family | 16 | 0.23% |
| nominated for | 16 | 0.23% |
| notable work | 16 | 0.23% |
| sport | 16 | 0.23% |
| author | 15 | 0.22% |
| language used | 15 | 0.22% |
| cast member | 14 | 0.21% |
| residence | 14 | 0.21% |
| regulated_by | 14 | 0.21% |
| continent | 14 | 0.21% |
| composer | 14 | 0.21% |
| military branch | 14 | 0.21% |
| authored_by | 14 | 0.21% |
| developer | 14 | 0.21% |
| uses | 14 | 0.21% |
| present in work | 14 | 0.21% |
| owned by | 14 | 0.21% |
| league | 14 | 0.21% |
| developed_by | 14 | 0.21% |
| spouse_of | 14 | 0.21% |
| main subject | 13 | 0.19% |
| conflict | 13 | 0.19% |
| participant in | 13 | 0.19% |
| located in or next to body of water | 13 | 0.19% |
| noble title | 13 | 0.19% |
| named after | 13 | 0.19% |
| architect | 13 | 0.19% |
| place of death | 13 | 0.19% |
| significant event | 12 | 0.18% |
| native language | 12 | 0.18% |
| sibling | 12 | 0.18% |
| head of government | 12 | 0.18% |
| follows | 12 | 0.18% |
| charged_with | 12 | 0.18% |
| place of publication | 12 | 0.18% |
| mother | 12 | 0.18% |
| significant person | 12 | 0.18% |
| creator | 12 | 0.18% |
| place served by transport hub | 12 | 0.18% |
| characters | 12 | 0.18% |
| has use | 11 | 0.16% |
| parent organization | 11 | 0.16% |
| based on | 11 | 0.16% |
| published_by | 11 | 0.16% |
| location of formation | 11 | 0.16% |
| basin country | 11 | 0.16% |
| opposite of | 11 | 0.16% |
| invested_in | 11 | 0.16% |
| parent_of | 11 | 0.16% |
| producer | 10 | 0.15% |
| board_member_of | 10 | 0.15% |
| operator | 10 | 0.15% |
| parent taxon | 10 | 0.15% |
| biological process | 10 | 0.15% |
| located in/on physical feature | 10 | 0.15% |
| chairperson | 10 | 0.15% |
| partner in business or sport | 10 | 0.15% |
| directed_by | 10 | 0.15% |
| field of work | 10 | 0.15% |
| fictional universe described in | 10 | 0.15% |
| conferred by | 10 | 0.15% |
| distributed by | 10 | 0.15% |
| home venue | 10 | 0.15% |
| narrative location | 10 | 0.15% |
| member of political party | 10 | 0.15% |
| manufactured_by | 10 | 0.15% |
| child_of | 10 | 0.15% |
| located in protected area | 10 | 0.15% |
| facet of | 9 | 0.13% |
| production company | 9 | 0.13% |
| item operated | 9 | 0.13% |
| presenter | 9 | 0.13% |
| terminus | 9 | 0.13% |
| located on terrain feature | 9 | 0.13% |
| crosses | 9 | 0.13% |
| librettist | 9 | 0.13% |
| incarnation of | 9 | 0.13% |
| owns | 9 | 0.13% |
| connecting line | 9 | 0.13% |
| capital of | 9 | 0.13% |
| architectural style | 9 | 0.13% |
| stock market index | 9 | 0.13% |
| EVALUATE-FOR | 9 | 0.13% |
| FEATURE-OF | 9 | 0.13% |
| interacts_with | 9 | 0.13% |
| original broadcaster | 8 | 0.12% |
| sibling_of | 8 | 0.12% |
| sports discipline competed in | 8 | 0.12% |
| mouth of the watercourse | 8 | 0.12% |
| student of | 8 | 0.12% |
| occupant | 8 | 0.12% |
| secretary general | 8 | 0.12% |
| broadcast by | 8 | 0.12% |
| applies to jurisdiction | 8 | 0.12% |
| said to be the same as | 8 | 0.12% |
| influenced by | 8 | 0.12% |
| indigenous to | 8 | 0.12% |
| manufacturer | 8 | 0.12% |
| country of origin | 8 | 0.12% |
| position played on team / speciality | 8 | 0.12% |
| parent astronomical body | 8 | 0.12% |
| PART-OF | 8 | 0.12% |
| allegiance | 8 | 0.12% |
| dialect of | 8 | 0.12% |
| capital | 8 | 0.12% |
| school district | 8 | 0.12% |
| product or material produced | 8 | 0.12% |
| subclass of | 8 | 0.12% |
| relative | 8 | 0.12% |
| designed_by | 8 | 0.12% |
| crew member(s) | 8 | 0.12% |
| record label | 8 | 0.12% |
| ethnic group | 7 | 0.10% |
| military rank | 7 | 0.10% |
| has contributing factor | 7 | 0.10% |
| place of burial | 7 | 0.10% |
| work location | 7 | 0.10% |
| canonization status | 7 | 0.10% |
| historic county | 7 | 0.10% |
| officeholder | 7 | 0.10% |
| licensed to broadcast to | 7 | 0.10% |
| place of detention | 7 | 0.10% |
| qualifies for event | 7 | 0.10% |
| publisher | 7 | 0.10% |
| COMPARE | 7 | 0.10% |
| location of discovery | 7 | 0.10% |
| replaced by | 7 | 0.10% |
| movement | 7 | 0.10% |
| instrument | 7 | 0.10% |
| online service | 7 | 0.10% |
| time period | 7 | 0.10% |
| participating team | 7 | 0.10% |
| has works in the collection | 7 | 0.10% |
| surface played on | 7 | 0.10% |
| has effect | 7 | 0.10% |
| medical condition treated | 7 | 0.10% |
| this taxon is source of | 7 | 0.10% |
| brand | 7 | 0.10% |
| mountain range | 7 | 0.10% |
| location of landing | 6 | 0.09% |
| radio format | 6 | 0.09% |
| member of sports team | 6 | 0.09% |
| cause of death | 6 | 0.09% |
| located in the present-day administrative territorial entity | 6 | 0.09% |
| operating system | 6 | 0.09% |
| taxonomic type | 6 | 0.09% |
| form of creative work | 6 | 0.09% |
| writing language | 6 | 0.09% |
| pole position | 6 | 0.09% |
| vessel class | 6 | 0.09% |
| history of topic | 6 | 0.09% |
| winner | 6 | 0.09% |
| dam | 6 | 0.09% |
| commanded by | 6 | 0.09% |
| owner of | 6 | 0.09% |
| voice type | 6 | 0.09% |
| student | 6 | 0.09% |
| language of work or name | 6 | 0.09% |
| candidate | 6 | 0.09% |
| partially coincident with | 6 | 0.09% |
| has part(s) of the class | 6 | 0.09% |
| significant place | 6 | 0.09% |
| discoverer or inventor | 6 | 0.09% |
| vehicle | 6 | 0.09% |
| standards body | 6 | 0.09% |
| wheel arrangement | 6 | 0.09% |
| legislated by | 6 | 0.09% |
| derived_from | 6 | 0.09% |
| has edition or translation | 6 | 0.09% |
| testified_against | 6 | 0.09% |
| located on street | 6 | 0.09% |
| candidacy in election | 6 | 0.09% |
| after a work by | 6 | 0.09% |
| designed by | 6 | 0.09% |
| operating area | 6 | 0.09% |
| musical conductor | 6 | 0.09% |
| signatory | 6 | 0.09% |
| distributor_of | 6 | 0.09% |
| filming location | 6 | 0.09% |
| anatomical location | 6 | 0.09% |
| diocese | 6 | 0.09% |
| found in taxon | 6 | 0.09% |
| endemic to | 5 | 0.07% |
| next higher rank | 5 | 0.07% |
| religion or worldview | 5 | 0.07% |
| model item | 5 | 0.07% |
| honorific prefix | 5 | 0.07% |
| legal form | 5 | 0.07% |
| this zoological name is coordinate with | 5 | 0.07% |
| physically interacts with | 5 | 0.07% |
| child astronomical body | 5 | 0.07% |
| medical condition | 5 | 0.07% |
| edition or translation of | 5 | 0.07% |
| mentor_of | 5 | 0.07% |
| born_in | 5 | 0.07% |
| territory overlaps | 5 | 0.07% |
| transport network | 5 | 0.07% |
| grants | 5 | 0.07% |
| organizer | 5 | 0.07% |
| has quality | 5 | 0.07% |
| implementation of | 5 | 0.07% |
| part of the series | 5 | 0.07% |
| target | 5 | 0.07% |
| editor | 5 | 0.07% |
| chromosome | 5 | 0.07% |
| discography | 5 | 0.07% |
| HYPONYM-OF | 5 | 0.07% |
| country for sport | 5 | 0.07% |
| is the study of | 5 | 0.07% |
| competition class | 5 | 0.07% |
| highest judicial authority | 5 | 0.07% |
| culture | 5 | 0.07% |
| located in the statistical territorial entity | 5 | 0.07% |
| advisor_to | 5 | 0.07% |
| country of registry | 5 | 0.07% |
| has subsidiary | 5 | 0.07% |
| space launch vehicle | 5 | 0.07% |
| currency | 5 | 0.07% |
| distribution format | 5 | 0.07% |
| recommended unit of measurement | 5 | 0.07% |
| contributing factor of | 5 | 0.07% |
| perpetrator | 5 | 0.07% |
| genetic association | 5 | 0.07% |
| IUCN conservation status | 5 | 0.07% |
| coextensive with | 5 | 0.07% |
| head of state | 5 | 0.07% |
| Exhibits | 5 | 0.07% |
| legislative body | 4 | 0.06% |
| lyrics by | 4 | 0.06% |
| ammunition | 4 | 0.06% |
| located in time zone | 4 | 0.06% |
| start point | 4 | 0.06% |
| astronaut mission | 4 | 0.06% |
| parent club | 4 | 0.06% |
| terminus location | 4 | 0.06% |
| authority | 4 | 0.06% |
| published in | 4 | 0.06% |
| mascot | 4 | 0.06% |
| cathedral | 4 | 0.06% |
| sexual orientation | 4 | 0.06% |
| died_in | 4 | 0.06% |
| constellation | 4 | 0.06% |
| chief executive officer | 4 | 0.06% |
| stock exchange | 4 | 0.06% |
| league system | 4 | 0.06% |
| depicts | 4 | 0.06% |
| member of the crew of | 4 | 0.06% |
| drug or therapy used for treatment | 4 | 0.06% |
| office held by head of government | 4 | 0.06% |
| takes place in fictional universe | 4 | 0.06% |
| depicted by | 4 | 0.06% |
| separated from | 4 | 0.06% |
| title of chess person | 4 | 0.06% |
| package management system | 4 | 0.06% |
| affiliation | 4 | 0.06% |
| territory claimed by | 4 | 0.06% |
| heritage designation | 4 | 0.06% |
| fabrication method | 4 | 0.06% |
| participant of | 4 | 0.06% |
| inspired by | 4 | 0.06% |
| enemy of | 4 | 0.06% |
| convicted of | 4 | 0.06% |
| general classification of race participants | 4 | 0.06% |
| maintained by | 4 | 0.06% |
| adjacent station | 4 | 0.06% |
| programming language | 4 | 0.06% |
| location of first performance | 4 | 0.06% |
| customer_of | 4 | 0.06% |
| has cause | 4 | 0.06% |
| objective of project or action | 4 | 0.06% |
| programming paradigm | 4 | 0.06% |
| taxon rank | 4 | 0.06% |
| allied_with | 4 | 0.06% |
| carries | 4 | 0.06% |
| sponsored_by | 3 | 0.04% |
| copyright holder | 3 | 0.04% |
| highest point | 3 | 0.04% |
| translator | 3 | 0.04% |
| health specialty | 3 | 0.04% |
| structural engineer | 3 | 0.04% |
| source of energy | 3 | 0.04% |
| collection | 3 | 0.04% |
| commissioned by | 3 | 0.04% |
| designed to carry | 3 | 0.04% |
| game mode | 3 | 0.04% |
| sexually homologous with | 3 | 0.04% |
| commander of (DEPRECATED) | 3 | 0.04% |
| illustrator | 3 | 0.04% |
| dedicated to | 3 | 0.04% |
| has written for | 3 | 0.04% |
| fastest lap | 3 | 0.04% |
| measurement scale | 3 | 0.04% |
| linguistic typology | 3 | 0.04% |
| location of creation | 3 | 0.04% |
| opposed_to | 3 | 0.04% |
| vehicle normally used | 3 | 0.04% |
| substitute/deputy/replacement of office/officeholder | 3 | 0.04% |
| contributed to creative work | 3 | 0.04% |
| first appearance | 3 | 0.04% |
| co-driver | 3 | 0.04% |
| manner of death | 3 | 0.04% |
| has immediate cause | 3 | 0.04% |
| merged into | 3 | 0.04% |
| character type | 3 | 0.04% |
| ancestral home | 3 | 0.04% |
| won | 3 | 0.04% |
| league level above | 3 | 0.04% |
| party_to | 3 | 0.04% |
| game mechanics | 3 | 0.04% |
| given name version for other gender | 3 | 0.04% |
| sidekick of | 3 | 0.04% |
| patron saint | 3 | 0.04% |
| victory | 3 | 0.04% |
| field of this occupation | 3 | 0.04% |
| named_after | 3 | 0.04% |
| represented by | 3 | 0.04% |
| delta of | 3 | 0.04% |
| subject has role | 3 | 0.04% |
| airline alliance | 3 | 0.04% |
| archives at | 3 | 0.04% |
| copyright license | 3 | 0.04% |
| cause | 3 | 0.04% |
| has fruit type | 3 | 0.04% |
| airline hub | 3 | 0.04% |
| director of photography | 2 | 0.03% |
| head coach | 2 | 0.03% |
| subdivision of this unit | 2 | 0.03% |
| screenwriter | 2 | 0.03% |
| possible treatment | 2 | 0.03% |
| central bank | 2 | 0.03% |
| political ideology | 2 | 0.03% |
| made from material | 2 | 0.03% |
| basic form of government | 2 | 0.03% |
| youth wing | 2 | 0.03% |
| armament | 2 | 0.03% |
| choreographer | 2 | 0.03% |
| main regulatory text | 2 | 0.03% |
| lake outflow | 2 | 0.03% |
| Gram staining | 2 | 0.03% |
| site of astronomical discovery | 2 | 0.03% |
| killed by | 2 | 0.03% |
| industry | 2 | 0.03% |
| sponsor | 2 | 0.03% |
| emulates | 2 | 0.03% |
| religious order | 2 | 0.03% |
| stepparent | 2 | 0.03% |
| afflicts | 2 | 0.03% |
| molecular function | 2 | 0.03% |
| successor_of | 2 | 0.03% |
| inflows | 2 | 0.03% |
| treats | 2 | 0.03% |
| tributary | 2 | 0.03% |
| worshipped by | 2 | 0.03% |
| executive body | 2 | 0.03% |
| family name identical to this given name | 2 | 0.03% |
| media franchise | 2 | 0.03% |
| tracklist | 2 | 0.03% |
| character designer | 2 | 0.03% |
| day in year for periodic occurrence | 2 | 0.03% |
| color | 2 | 0.03% |
| produced by | 2 | 0.03% |
| director of archaeological fieldwork | 2 | 0.03% |
| contains settlement | 2 | 0.03% |
| destination point | 2 | 0.03% |
| lake on watercourse | 2 | 0.03% |
| league level below | 2 | 0.03% |
| located on astronomical body | 2 | 0.03% |
| has boundary | 2 | 0.03% |
| measured physical quantity | 2 | 0.03% |
| enclave within | 2 | 0.03% |
| business model | 2 | 0.03% |
| top-level Internet domain | 2 | 0.03% |
| capital_of | 2 | 0.03% |
| board member | 2 | 0.03% |
| source of material | 2 | 0.03% |
| software engine | 2 | 0.03% |
| official residence | 2 | 0.03% |
| foundational text | 2 | 0.03% |
| house publication | 2 | 0.03% |
| track gauge | 2 | 0.03% |
| flag | 2 | 0.03% |
| set in period | 2 | 0.03% |
| writing system | 2 | 0.03% |
| primary destinations | 2 | 0.03% |
| drainage basin | 2 | 0.03% |
| unmarried partner | 2 | 0.03% |
| described by source | 2 | 0.03% |
| original language of film or TV show | 2 | 0.03% |
| type foundry | 2 | 0.03% |
| twinned administrative body | 2 | 0.03% |
| home world | 2 | 0.03% |
| genic_interaction | 2 | 0.03% |
| measures | 2 | 0.03% |
| organization directed by the office or position | 2 | 0.03% |
| narrator | 2 | 0.03% |
| cause of destruction | 2 | 0.03% |
| historical region | 2 | 0.03% |
| points classification | 2 | 0.03% |
| appointed by | 2 | 0.03% |
| commemorates | 2 | 0.03% |
| has anatomical branch | 2 | 0.03% |
| contributor to the creative work or subject | 2 | 0.03% |
| reservoir created | 2 | 0.03% |
| business division | 2 | 0.03% |
| depends_on | 2 | 0.03% |
| connecting service | 2 | 0.03% |
| captain | 2 | 0.03% |
| director / manager | 2 | 0.03% |
| origin of the watercourse | 1 | 0.01% |
| charge | 1 | 0.01% |
| voice actor | 1 | 0.01% |
| leader | 1 | 0.01% |
| from narrative universe | 1 | 0.01% |
| discovery method | 1 | 0.01% |
| has grammatical case | 1 | 0.01% |
| temporal range end | 1 | 0.01% |
| defendant | 1 | 0.01% |
| order of battle | 1 | 0.01% |
| encodes | 1 | 0.01% |
| parent peak | 1 | 0.01% |
| anatomical branch of | 1 | 0.01% |
| manifestation of | 1 | 0.01% |
| office contested | 1 | 0.01% |
| record or record progression | 1 | 0.01% |
| has facet polytope | 1 | 0.01% |
| season | 1 | 0.01% |
| season of club or team | 1 | 0.01% |
| underlies | 1 | 0.01% |
| via | 1 | 0.01% |
| military casualty classification | 1 | 0.01% |
| composed_by | 1 | 0.01% |
| adjacent_to | 1 | 0.01% |
| tonality | 1 | 0.01% |
| Cotreatment | 1 | 0.01% |
| points/goal scored by | 1 | 0.01% |
| exhibition history | 1 | 0.01% |
| type of variable star | 1 | 0.01% |
| main food source | 1 | 0.01% |
| implements | 1 | 0.01% |
| powered by | 1 | 0.01% |
| feast day | 1 | 0.01% |
| fandom | 1 | 0.01% |
| antagonist muscle | 1 | 0.01% |
| has pet | 1 | 0.01% |
| officially opened by | 1 | 0.01% |
| interested in | 1 | 0.01% |
| polymer of | 1 | 0.01% |
| suggests the existence of | 1 | 0.01% |
| next crossing upstream | 1 | 0.01% |
| parliamentary term | 1 | 0.01% |
| domain of saint or deity | 1 | 0.01% |
| contains | 1 | 0.01% |
| successful candidate | 1 | 0.01% |
| prerequisite | 1 | 0.01% |
| input method | 1 | 0.01% |
| instruction set | 1 | 0.01% |
| convicted_of | 1 | 0.01% |
| depends on software | 1 | 0.01% |
| sold_to | 1 | 0.01% |
| official religion | 1 | 0.01% |
| home port | 1 | 0.01% |
| dual to | 1 | 0.01% |
| charted in | 1 | 0.01% |
| recorded at studio or venue | 1 | 0.01% |
| prime factor | 1 | 0.01% |
| mountains classification | 1 | 0.01% |
| valid in period | 1 | 0.01% |
| sued_by | 1 | 0.01% |
| Bind | 1 | 0.01% |
| final event | 1 | 0.01% |
| cites work | 1 | 0.01% |
| connector | 1 | 0.01% |
| office held by head of the organization | 1 | 0.01% |
| penalty | 1 | 0.01% |
| superhuman feature or ability | 1 | 0.01% |
| newspaper format | 1 | 0.01% |
| solved by | 1 | 0.01% |
| animal breed | 1 | 0.01% |
| members have occupation | 1 | 0.01% |
| shape | 1 | 0.01% |
| public holiday | 1 | 0.01% |
| trophy awarded | 1 | 0.01% |
| appears in the form of | 1 | 0.01% |
| scheduled service destination | 1 | 0.01% |
| coolant | 1 | 0.01% |
| series spin-off | 1 | 0.01% |
| cover art by | 1 | 0.01% |
| iconographic symbol | 1 | 0.01% |
| rector | 1 | 0.01% |
| microstructure | 1 | 0.01% |
| product certification | 1 | 0.01% |
| cuisine | 1 | 0.01% |
| coach of sports team | 1 | 0.01% |
| modified version of | 1 | 0.01% |
| host | 1 | 0.01% |
| amended by | 1 | 0.01% |
| risk factor | 1 | 0.01% |
| instrumentation | 1 | 0.01% |
| writable file format | 1 | 0.01% |
| species kept | 1 | 0.01% |
| language regulatory body | 1 | 0.01% |
| avionics | 1 | 0.01% |
| advertises | 1 | 0.01% |
| official language | 1 | 0.01% |
| venous drainage | 1 | 0.01% |
| set in environment | 1 | 0.01% |
| sex or gender | 1 | 0.01% |
| complies with | 1 | 0.01% |
| overlies | 1 | 0.01% |
| copyright status | 1 | 0.01% |
| coat of arms | 1 | 0.01% |
| innervated by | 1 | 0.01% |
| era name | 1 | 0.01% |
| editor-in-chief | 1 | 0.01% |
| engine configuration | 1 | 0.01% |
| has goal | 1 | 0.01% |
| court | 1 | 0.01% |
| taxon synonym | 1 | 0.01% |
| structure replaces | 1 | 0.01% |
| theme music | 1 | 0.01% |
| performed at | 1 | 0.01% |
| music created for | 1 | 0.01% |
| central bank/issuer | 1 | 0.01% |
| parallel street | 1 | 0.01% |
| majority opinion by | 1 | 0.01% |
| develops from | 1 | 0.01% |
| used by | 1 | 0.01% |
| state of conservation | 1 | 0.01% |
| research site | 1 | 0.01% |
| Christian liturgical rite | 1 | 0.01% |
| worn by | 1 | 0.01% |
| undercarriage | 1 | 0.01% |
| office held by head of state | 1 | 0.01% |
| political alignment | 1 | 0.01% |
| consecrator | 1 | 0.01% |
| set during recurring event | 1 | 0.01% |
| soundtrack release | 1 | 0.01% |
| repealed by | 1 | 0.01% |
| causes | 1 | 0.01% |
| derivative work | 1 | 0.01% |
| plot expanded in | 1 | 0.01% |
| collection creator | 1 | 0.01% |
| cell component | 1 | 0.01% |

### Classifications

**train, actionability.** Majority class 87.6% of 354 answers. The marginal is `true_label`, not the menu.

#### train actionability

| label | count | share |
| --- | --- | --- |
| informational | 310 | 87.57% |
| advisory | 27 | 7.63% |
| requires action | 17 | 4.80% |

**train, audience.** Majority class 78.9% of 446 answers. The marginal is `true_label`, not the menu.

#### train audience

| label | count | share |
| --- | --- | --- |
| general public | 352 | 78.92% |
| expert | 51 | 11.43% |
| regulatory | 30 | 6.73% |
| internal | 13 | 2.91% |

**train, biotech_event.** Majority class 27.8% of 245 answers. The marginal is `true_label`, not the menu.

#### train biotech_event

| label | count | share |
| --- | --- | --- |
| executive statement | 68 | 27.76% |
| company description | 40 | 16.33% |
| alliance & partnership | 20 | 8.16% |
| other | 18 | 7.35% |
| product launching & presentation | 17 | 6.94% |
| new initiatives or programs | 9 | 3.67% |
| m&a | 7 | 2.86% |
| article publication | 6 | 2.45% |
| funding round | 6 | 2.45% |
| expanding industry | 5 | 2.04% |
| executive appointment | 5 | 2.04% |
| regulatory approval | 5 | 2.04% |
| product updates | 5 | 2.04% |
| hiring | 5 | 2.04% |
| expanding geography | 5 | 2.04% |
| support & philanthropy | 5 | 2.04% |
| service & product providing | 4 | 1.63% |
| investment in public company | 3 | 1.22% |
| ipo exit | 3 | 1.22% |
| participation in an event | 3 | 1.22% |
| event organization | 2 | 0.82% |
| new initiatives & programs | 1 | 0.41% |
| foundation | 1 | 0.41% |
| department establishment | 1 | 0.41% |
| clinical trial sponsorship | 1 | 0.41% |

**train, certainty.** Majority class 97.1% of 347 answers. The marginal is `true_label`, not the menu.

#### train certainty

| label | count | share |
| --- | --- | --- |
| asserted | 337 | 97.12% |
| hedged | 9 | 2.59% |
| speculative | 1 | 0.29% |

**train, chfinann_event.** Majority class 48.3% of 1,519 answers. The marginal is `true_label`, not the menu.

#### train chfinann_event

| label | count | share |
| --- | --- | --- |
| EquityPledge | 733 | 48.26% |
| EquityOverweight | 329 | 21.66% |
| EquityUnderweight | 315 | 20.74% |
| EquityRepurchase | 115 | 7.57% |
| EquityFreeze | 27 | 1.78% |

**train, docee_event.** Majority class 6.3% of 4,271 answers. The marginal is `true_label`, not the menu.

#### train docee_event

| label | count | share |
| --- | --- | --- |
| Sports Competition | 267 | 6.25% |
| none | 234 | 5.48% |
| Government Policy Changes | 214 | 5.01% |
| Fire | 210 | 4.92% |
| CommitCrime - Arrest | 182 | 4.26% |
| CommitCrime - Investigate | 180 | 4.21% |
| Organization Fine | 171 | 4.00% |
| Armed Conflict | 165 | 3.86% |
| Appoint_Inauguration | 129 | 3.02% |
| Protest | 125 | 2.93% |
| Resignation_Dismissal | 121 | 2.83% |
| Air Crash | 116 | 2.72% |
| Famous Person - Death | 110 | 2.58% |
| Election | 108 | 2.53% |
| New Achievements in Aerospace | 103 | 2.41% |
| Road Crash | 89 | 2.08% |
| Earthquakes | 88 | 2.06% |
| Riot | 85 | 1.99% |
| Diplomatic Talks | 82 | 1.92% |
| Famous Person - Give a Speech | 80 | 1.87% |
| Disease Outbreaks | 80 | 1.87% |
| Break Historical Records | 78 | 1.83% |
| CommitCrime - Accuse | 71 | 1.66% |
| Sign Agreement | 67 | 1.57% |
| CommitCrime - Sentence | 67 | 1.57% |
| Awards Ceremony | 66 | 1.55% |
| Mass Poisoning | 60 | 1.40% |
| Military Excercise | 52 | 1.22% |
| Storm | 51 | 1.19% |
| Floods | 50 | 1.17% |
| Organization Established | 44 | 1.03% |
| New Wonders in Nature | 42 | 0.98% |
| Diplomatic Visit | 41 | 0.96% |
| Shipwreck | 39 | 0.91% |
| Strike | 37 | 0.87% |
| Environment Pollution | 36 | 0.84% |
| Financial Crisis | 33 | 0.77% |
| Bank Robbery | 32 | 0.75% |
| Withdraw from an Orgnization | 31 | 0.73% |
| New Archeological Discoveries | 31 | 0.73% |
| Volcano Eruption | 29 | 0.68% |
| Financial Aid | 28 | 0.66% |
| Famous Person - Divorce | 28 | 0.66% |
| Train Collisions | 25 | 0.59% |
| CommitCrime - Release | 25 | 0.59% |
| Mudslides | 25 | 0.59% |
| Mine Collapses | 24 | 0.56% |
| Famous Person - Recovered | 24 | 0.56% |
| Famous Person - Sick | 24 | 0.56% |
| Organization Merge | 22 | 0.52% |
| Organization Closed | 19 | 0.44% |
| Famine | 18 | 0.42% |
| Gas Explosion | 18 | 0.42% |
| Famous Person - Marriage | 18 | 0.42% |
| Droughts | 18 | 0.42% |
| Regime Change | 17 | 0.40% |
| Insect Disaster | 13 | 0.30% |
| Tsunamis | 10 | 0.23% |
| Join in an Orgnization | 10 | 0.23% |
| Tear Up Agreement | 9 | 0.21% |

**train, docfee_event.** Majority class 31.4% of 988 answers. The marginal is `true_label`, not the menu.

#### train docfee_event

| label | count | share |
| --- | --- | --- |
| EquityUnderweight | 310 | 31.38% |
| BankruptcyLiquidation | 169 | 17.11% |
| EquityPledge | 160 | 16.19% |
| MajorAssetLoss | 132 | 13.36% |
| EquityFreeze | 75 | 7.59% |
| EquityOverweight | 55 | 5.57% |
| MajorSafetyAccident | 34 | 3.44% |
| MajorExternalPayout | 27 | 2.73% |
| ExecutiveDeath | 26 | 2.63% |

**train, document_genre.** Majority class 76.5% of 332 answers. The marginal is `true_label`, not the menu.

#### train document_genre

| label | count | share |
| --- | --- | --- |
| news report | 254 | 76.51% |
| announcement | 39 | 11.75% |
| opinion | 20 | 6.02% |
| review | 7 | 2.11% |
| correspondence | 4 | 1.20% |
| advertisement | 3 | 0.90% |
| legal filing | 3 | 0.90% |
| technical documentation | 2 | 0.60% |

**train, formality.** Majority class 82.6% of 419 answers. The marginal is `true_label`, not the menu.

#### train formality

| label | count | share |
| --- | --- | --- |
| formal | 346 | 82.58% |
| informal | 73 | 17.42% |

**train, language_register.** Majority class 81.3% of 353 answers. The marginal is `true_label`, not the menu.

#### train language_register

| label | count | share |
| --- | --- | --- |
| plain | 287 | 81.30% |
| technical | 51 | 14.45% |
| colloquial | 9 | 2.55% |
| legal | 6 | 1.70% |

**train, risk_level.** Majority class 41.9% of 310 answers. The marginal is `true_label`, not the menu.

#### train risk_level

| label | count | share |
| --- | --- | --- |
| none | 130 | 41.94% |
| high | 67 | 21.61% |
| low | 66 | 21.29% |
| moderate | 47 | 15.16% |

**train, sentiment.** Majority class 38.6% of 415 answers. The marginal is `true_label`, not the menu.

#### train sentiment

| label | count | share |
| --- | --- | --- |
| positive | 160 | 38.55% |
| neutral | 136 | 32.77% |
| negative | 119 | 28.67% |

**train, subjectivity.** Majority class 87.9% of 331 answers. The marginal is `true_label`, not the menu.

#### train subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 291 | 87.92% |
| subjective | 40 | 12.08% |

**train, temporal_orientation.** Majority class 69.7% of 353 answers. The marginal is `true_label`, not the menu.

#### train temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 246 | 69.69% |
| present | 66 | 18.70% |
| future | 35 | 9.92% |
| timeless | 6 | 1.70% |

**train, topic.** Majority class 16.6% of 790 answers. The marginal is `true_label`, not the menu.

#### train topic

| label | count | share |
| --- | --- | --- |
| business | 131 | 16.58% |
| politics | 99 | 12.53% |
| sports | 95 | 12.03% |
| legal | 76 | 9.62% |
| world | 72 | 9.11% |
| entertainment | 61 | 7.72% |
| finance | 48 | 6.08% |
| technology | 47 | 5.95% |
| health | 37 | 4.68% |
| environment | 33 | 4.18% |
| education | 33 | 4.18% |
| science | 30 | 3.80% |
| travel | 8 | 1.01% |
| defense | 8 | 1.01% |
| energy | 7 | 0.89% |
| agriculture | 5 | 0.63% |

**train, urgency.** Majority class 76.4% of 314 answers. The marginal is `true_label`, not the menu.

#### train urgency

| label | count | share |
| --- | --- | --- |
| routine | 240 | 76.43% |
| elevated | 52 | 16.56% |
| critical | 18 | 5.73% |
| urgent | 4 | 1.27% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | condition | 5 | 100.00% |
| clinical_finding | patient_group | 5 | 100.00% |
| clinical_finding | treatment | 5 | 100.00% |
| clinical_finding | outcome | 4 | 80.00% |
| clinical_finding | record_metadata | natural/condition ×5 |  |
| company_profile | name | 12 | 100.00% |
| company_profile | size | 9 | 75.00% |
| company_profile | industry | 8 | 66.67% |
| company_profile | headquarters | 8 | 66.67% |
| company_profile | founded | 3 | 25.00% |
| company_profile | record_metadata | natural/name ×12 |  |
| event_listing | name | 16 | 100.00% |
| event_listing | date | 14 | 87.50% |
| event_listing | venue | 13 | 81.25% |
| event_listing | organizer | 11 | 68.75% |
| event_listing | price | 3 | 18.75% |
| event_listing | record_metadata | natural/name ×16 |  |
| financial_report | organization | 3 | 100.00% |
| financial_report | period | 3 | 100.00% |
| financial_report | revenue | 2 | 66.67% |
| financial_report | net_income | 2 | 66.67% |
| financial_report | record_metadata | natural/organization ×3 |  |
| incident_report | severity | 20 | 100.00% |
| incident_report | date | 19 | 95.00% |
| incident_report | location | 16 | 80.00% |
| incident_report | incident_type | 11 | 55.00% |
| incident_report | casualties | 9 | 45.00% |
| incident_report | record_metadata | natural/location ×13, natural/incident_type ×5, natural/date ×2 |  |
| legal_case | court | 8 | 100.00% |
| legal_case | defendant | 8 | 100.00% |
| legal_case | status | 8 | 100.00% |
| legal_case | case_name | 6 | 75.00% |
| legal_case | plaintiff | 4 | 50.00% |
| legal_case | record_metadata | natural/case_name ×6, natural/court ×2 |  |
| person_profile | name | 97 | 100.00% |
| person_profile | role | 92 | 94.85% |
| person_profile | employer | 87 | 89.69% |
| person_profile | location | 86 | 88.66% |
| person_profile | record_metadata | natural/location ×63, natural/name ×18 |  |
| product | name | 38 | 100.00% |
| product | brand | 38 | 100.00% |
| product | condition | 38 | 100.00% |
| product | price | 31 | 81.58% |
| product | record_metadata | natural/name ×38 |  |
| real_estate_listing | address | 4 | 100.00% |
| real_estate_listing | area | 4 | 100.00% |
| real_estate_listing | property_type | 4 | 100.00% |
| real_estate_listing | price | 3 | 75.00% |
| real_estate_listing | bedrooms | 3 | 75.00% |
| real_estate_listing | record_metadata | natural/address ×4 |  |
| record | status | 70 | 15.70% |
| record | location | 50 | 11.21% |
| record | name | 33 | 7.40% |
| record | address | 27 | 6.05% |
| record | title | 19 | 4.26% |
| record | description | 12 | 2.69% |
| record | price | 12 | 2.69% |
| record | state | 12 | 2.69% |
| record | age | 12 | 2.69% |
| record | patient_id | 11 | 2.47% |
| record | city | 11 | 2.47% |
| record | date | 10 | 2.24% |
| record | patient_name | 9 | 2.02% |
| record | amount | 9 | 2.02% |
| record | author | 8 | 1.79% |
| record | year | 7 | 1.57% |
| record | policy_id | 6 | 1.35% |
| record | designation | 6 | 1.35% |
| record | rating | 6 | 1.35% |
| record | funding_source | 6 | 1.35% |
| record | head | 6 | 1.35% |
| record | tail | 6 | 1.35% |
| record | venue | 6 | 1.35% |
| record | policyholder | 5 | 1.12% |
| record | region | 5 | 1.12% |
| record | product_id | 5 | 1.12% |
| record | ticket_id | 5 | 1.12% |
| record | reviewer | 5 | 1.12% |
| record | event_id | 5 | 1.12% |
| record | company | 5 | 1.12% |
| record | claim_id | 5 | 1.12% |
| record | report_id | 4 | 0.90% |
| record | time | 4 | 0.90% |
| record | origin | 4 | 0.90% |
| record | symptoms | 4 | 0.90% |
| record | discovery.date | 4 | 0.90% |
| record | physical_characteristics.observations.0.telescope.type | 4 | 0.90% |
| record | physical_characteristics.observations.0.telescope.name | 4 | 0.90% |
| record | device | 4 | 0.90% |
| record | outcome | 4 | 0.90% |
| record | founding_year | 4 | 0.90% |
| record | full_name | 4 | 0.90% |
| record | home_team | 4 | 0.90% |
| record | shares | 4 | 0.90% |
| record | policy_number | 4 | 0.90% |
| record | acquirer | 4 | 0.90% |
| record | opponent | 3 | 0.67% |
| record | destination | 3 | 0.67% |
| record | delay_reason | 3 | 0.67% |
| record | features | 3 | 0.67% |
| record | upvotes | 3 | 0.67% |
| record | reason | 3 | 0.67% |
| record | customer_name | 3 | 0.67% |
| record | value | 3 | 0.67% |
| record | sku | 3 | 0.67% |
| record | call_id | 3 | 0.67% |
| record | score | 3 | 0.67% |
| record | locations | 3 | 0.67% |
| record | physical_characteristics.observations.0.publication_year | 3 | 0.67% |
| record | discovery.observatory.location.country | 3 | 0.67% |
| record | amount: Monetary value of the transaction | 3 | 0.67% |
| record | role | 3 | 0.67% |
| record | quantity | 3 | 0.67% |
| record | topic | 3 | 0.67% |
| record | price: Selling price of the product in dollars | 3 | 0.67% |
| record | provider | 3 | 0.67% |
| record | employee_id | 3 | 0.67% |
| record | dosage | 3 | 0.67% |
| record | away_team | 3 | 0.67% |
| record | clause_id | 3 | 0.67% |
| record | creator | 3 | 0.67% |
| record | market_cap | 3 | 0.67% |
| record | discovery.observatory.name | 3 | 0.67% |
| record | reviewer_name | 3 | 0.67% |
| record | organizations | 3 | 0.67% |
| record | keywords | 3 | 0.67% |
| record | brand | 3 | 0.67% |
| record | country | 3 | 0.67% |
| record | district | 3 | 0.67% |
| record | name: Full name of the patient | 3 | 0.67% |
| record | risk_score | 3 | 0.67% |
| record | issue | 2 | 0.45% |
| record | confirmation_code | 2 | 0.45% |
| record | underwriter | 2 | 0.45% |
| record | review_period | 2 | 0.45% |
| record | performance_score | 2 | 0.45% |
| record | bathrooms | 2 | 0.45% |
| record | species.description | 2 | 0.45% |
| record | credits | 2 | 0.45% |
| record | risk_rating | 2 | 0.45% |
| record | heart_rate | 2 | 0.45% |
| record | room | 2 | 0.45% |
| record | dispute_status | 2 | 0.45% |
| record | owner | 2 | 0.45% |
| record | owner_location | 2 | 0.45% |
| record | delay_minutes | 2 | 0.45% |
| record | passenger_name | 2 | 0.45% |
| record | passenger_email | 2 | 0.45% |
| record | passenger_phone | 2 | 0.45% |
| record | medications | 2 | 0.45% |
| record | priority: Severity level assigned to the ticket | 2 | 0.45% |
| record | serial_number | 2 | 0.45% |
| record | location.city | 2 | 0.45% |
| record | discovery_year | 2 | 0.45% |
| record | net_profit | 2 | 0.45% |
| record | location: Headquarters city and state or country | 2 | 0.45% |
| record | focus: Primary technology or business area of the company | 2 | 0.45% |
| record | amount: Total capital raised in millions | 2 | 0.45% |
| record | post_valuation: Post-money valuation in millions | 2 | 0.45% |
| record | notes | 2 | 0.45% |
| record | concepts | 2 | 0.45% |
| record | cause | 2 | 0.45% |
| record | name: Name of the product as listed | 2 | 0.45% |
| record | organization | 2 | 0.45% |
| record | affiliation.0.type | 2 | 0.45% |
| record | severity | 2 | 0.45% |
| record | diameter.unit | 2 | 0.45% |
| record | discovery.discoverer | 2 | 0.45% |
| record | agency | 2 | 0.45% |
| record | duration | 2 | 0.45% |
| record | home_team: Name of the team playing at home | 2 | 0.45% |
| record | person | 2 | 0.45% |
| record | early_challenge | 2 | 0.45% |
| record | publication_date | 2 | 0.45% |
| record | location: Physical address of the product launch | 2 | 0.45% |
| record | product_code: Internal code assigned to the product | 2 | 0.45% |
| record | headquarters | 2 | 0.45% |
| record | issue_date | 2 | 0.45% |
| record | term_length | 2 | 0.45% |
| record | monthly_premium | 2 | 0.45% |
| record | terminology.ideogram.english | 2 | 0.45% |
| record | terminology.ideographic_script.english | 2 | 0.45% |
| record | comparison.alphabetic.learning_curve | 2 | 0.45% |
| record | comparison.alphabetic.pronunciation | 2 | 0.45% |
| record | comparison.ideographic.learning_curve | 2 | 0.45% |
| record | comparison.ideographic.example.meaning | 2 | 0.45% |
| record | caveat.example.meaning | 2 | 0.45% |
| record | caveat.example.components.0.function | 2 | 0.45% |
| record | caveat.example.components.0.component | 2 | 0.45% |
| record | caveat.example.components.0.meaning | 2 | 0.45% |
| record | caveat.example.components.0.related_characters.0.pronunciation | 2 | 0.45% |
| record | caveat.example.components.0.related_characters.1.pronunciation | 2 | 0.45% |
| record | caveat.example.components.0.related_characters.1.character | 2 | 0.45% |
| record | caveat.example.components.0.related_characters.1.meaning | 2 | 0.45% |
| record | caveat.example.components.0.related_characters.2.pronunciation | 2 | 0.45% |
| record | caveat.example.components.0.related_characters.2.meaning | 2 | 0.45% |
| record | caveat.example.components.0.position | 2 | 0.45% |
| record | caveat.example.components.1.meaning | 2 | 0.45% |
| record | caveat.example.components.1.position | 2 | 0.45% |
| record | education | 2 | 0.45% |
| record | listing_date | 2 | 0.45% |
| record | region.language | 2 | 0.45% |
| record | region.divisions.1.name | 2 | 0.45% |
| record | region.name | 2 | 0.45% |
| record | region.agriculture | 2 | 0.45% |
| record | region.description | 2 | 0.45% |
| record | tags | 2 | 0.45% |
| record | registration_number | 2 | 0.45% |
| record | mark_name | 2 | 0.45% |
| record | program_name | 2 | 0.45% |
| record | scientific_name | 2 | 0.45% |
| record | patient_age | 2 | 0.45% |
| record | manufacturer | 2 | 0.45% |
| record | person.name | 2 | 0.45% |
| record | campaign_id | 2 | 0.45% |
| record | subject | 2 | 0.45% |
| record | administration_date | 2 | 0.45% |
| record | problem_description | 2 | 0.45% |
| record | date: Date when the transaction occurred | 2 | 0.45% |
| record | research_background.animal_models_used | 2 | 0.45% |
| record | affiliation.institution_name | 2 | 0.45% |
| record | affiliation.department | 2 | 0.45% |
| record | ebitda_margin: Percentage of revenue that represents EBITDA | 2 | 0.45% |
| record | operational_status: Current phase of business operations | 2 | 0.45% |
| record | net_profit: Annual net profit in millions of dollars | 2 | 0.45% |
| record | location: Geographic headquarters of the company | 2 | 0.45% |
| record | imaging_date | 2 | 0.45% |
| record | findings | 2 | 0.45% |
| record | disclaimer | 2 | 0.45% |
| record | key_challenge | 2 | 0.45% |
| record | sentiment | 2 | 0.45% |
| record | coach | 2 | 0.45% |
| record | final_standing | 2 | 0.45% |
| record | win_percentage | 2 | 0.45% |
| record | ties | 2 | 0.45% |
| record | season | 2 | 0.45% |
| record | event_type: Category of the event | 2 | 0.45% |
| record | integration_plan | 2 | 0.45% |
| record | place.location.county | 2 | 0.45% |
| record | place.location.country | 2 | 0.45% |
| record | video_id | 2 | 0.45% |
| record | gender | 2 | 0.45% |
| record | company_name | 2 | 0.45% |
| record | jurisdiction | 2 | 0.45% |
| record | comments | 2 | 0.45% |
| record | likes | 2 | 0.45% |
| record | assists | 2 | 0.45% |
| record | ticker | 2 | 0.45% |
| record | capacity | 2 | 0.45% |
| record | conference_id | 2 | 0.45% |
| record | insurance | 2 | 0.45% |
| record | treatment | 2 | 0.45% |
| record | visit_type | 2 | 0.45% |
| record | asteroid_number | 2 | 0.45% |
| record | asteroid_type | 2 | 0.45% |
| record | asteroid.name | 2 | 0.45% |
| record | asteroid.type | 2 | 0.45% |
| record | acquired | 2 | 0.45% |
| record | acquisition_amount | 2 | 0.45% |
| record | valuation_method | 2 | 0.45% |
| record | device_id: Unique identifier assigned to the medical device | 2 | 0.45% |
| record | orbital_characteristics.perihelion.unit | 2 | 0.45% |
| record | review_id | 2 | 0.45% |
| record | repair_cost | 2 | 0.45% |
| record | description.leaves.terminal_spine.color | 2 | 0.45% |
| record | systematics_and_distribution.location.country | 2 | 0.45% |
| record | description.inflorescence.type | 2 | 0.45% |
| record | description.flowers.arrangement | 2 | 0.45% |
| record | description.fruit.width_mm.max | 2 | 0.45% |
| record | systematics_and_distribution.location.state | 2 | 0.45% |
| record | systematics_and_distribution.similar_species | 2 | 0.45% |
| record | description.leaves.surface | 2 | 0.45% |
| record | description.leaves.width_cm.min | 2 | 0.45% |
| record | description.fruit.length_mm.min | 2 | 0.45% |
| record | description.leaves.margins | 2 | 0.45% |
| record | description.fruit.shape | 2 | 0.45% |
| record | description.fruit.width_mm.min | 2 | 0.45% |
| record | description.flowering_time | 2 | 0.45% |
| record | id | 2 | 0.45% |
| record | sponsorship | 2 | 0.45% |
| record | vote_id: Unique identifier for the legislative vote | 2 | 0.45% |
| record | status: Current legislative status of the bill | 2 | 0.45% |
| record | scope: Broad category or focus area of the bill | 2 | 0.45% |
| record | organizer | 2 | 0.45% |
| record | cost | 2 | 0.45% |
| record | candidate_name | 2 | 0.45% |
| record | revenue | 2 | 0.45% |
| record | year: Year of financial data | 2 | 0.45% |
| record | partner_id: Unique identifier of the partner company | 2 | 0.45% |
| record | partner_status: Operational status of the partner company | 2 | 0.45% |
| record | partner_revenue: Partner's revenue in millions | 2 | 0.45% |
| record | company_id: Unique identifier for the company | 2 | 0.45% |
| record | location: City and state of the company | 2 | 0.45% |
| record | growth_rate: Percentage growth in revenue | 2 | 0.45% |
| record | partner_year: Year of partner's financial data | 2 | 0.45% |
| record | influences | 2 | 0.45% |
| record | structures | 2 | 0.45% |
| record | historical_periods | 2 | 0.45% |
| record | reportTitle | 2 | 0.45% |
| record | vehicle | 2 | 0.45% |
| record | mrn | 2 | 0.45% |
| record | zip_code | 2 | 0.45% |
| record | recommendation | 2 | 0.45% |
| record | source | 2 | 0.45% |
| record | village_name | 2 | 0.45% |
| record | punct | 2 | 0.45% |
| record | nsubj | 2 | 0.45% |
| record | ROOT | 2 | 0.45% |
| record | advmod | 2 | 0.45% |
| record | prep | 2 | 0.45% |
| record | relationships.0.type | 2 | 0.45% |
| record | lab_id | 2 | 0.45% |
| record | organizer.name | 2 | 0.45% |
| record | surveyPurpose | 2 | 0.45% |
| record | entity_name | 2 | 0.45% |
| record | dates | 2 | 0.45% |
| record | vaccine_type | 2 | 0.45% |
| record | peculiarity | 2 | 0.45% |
| record | spectral_class | 2 | 0.45% |
| record | procedure | 2 | 0.45% |
| record | case_name: Full name of the case in format plaintiff v defendant | 2 | 0.45% |
| record | claim_type: Nature of the legal claim such as breach of contract or noise violation | 2 | 0.45% |
| record | funding_id: Unique identifier for each funding round | 2 | 0.45% |
| record | roundtype: Type of funding round | 2 | 0.45% |
| record | carrier | 2 | 0.45% |
| record | filing_date | 2 | 0.45% |
| record | vendor_name | 2 | 0.45% |
| record | invoice_date | 2 | 0.45% |
| record | genus.subspecies.0.authority | 2 | 0.45% |
| record | genus.name_latin | 2 | 0.45% |
| record | genus.family | 2 | 0.45% |
| record | genus.synonyms | 2 | 0.45% |
| record | founder_name: Full name of the founder as stated in passage | 2 | 0.45% |
| record | motivation: Personal reason driving the company's founding | 2 | 0.45% |
| record | frequency | 2 | 0.45% |
| record | startup_name | 2 | 0.45% |
| record | audit_id | 1 | 0.22% |
| record | delivery_method | 1 | 0.22% |
| record | newsletter_name | 1 | 0.22% |
| record | subscription_amount | 1 | 0.22% |
| record | submission_date | 1 | 0.22% |
| record | attending_physician | 1 | 0.22% |
| record | bloodpressuresystolic | 1 | 0.22% |
| record | bloodpressurediastolic | 1 | 0.22% |
| record | monitoring_method | 1 | 0.22% |
| record | key_achievement | 1 | 0.22% |
| record | hotel.location.district | 1 | 0.22% |
| record | hotel.name | 1 | 0.22% |
| record | hotel.location.nearby_attractions | 1 | 0.22% |
| record | hotel.room_features | 1 | 0.22% |
| record | hotel.amenities | 1 | 0.22% |
| record | hotel.hotel_group | 1 | 0.22% |
| record | bedrooms | 1 | 0.22% |
| record | species.family | 1 | 0.22% |
| record | course_id | 1 | 0.22% |
| record | blood_pressure | 1 | 0.22% |
| record | research_context.primary_technique | 1 | 0.22% |
| record | research_context.application_goal | 1 | 0.22% |
| record | research_context.problem_statement.limitation | 1 | 0.22% |
| record | research_context.problem_statement.incompatible_substrates | 1 | 0.22% |
| record | research_context.problem_statement.affected_research_fields | 1 | 0.22% |
| record | methodology.proposed_solution | 1 | 0.22% |
| record | methodology.experimental_setup.biological_model.cell_line | 1 | 0.22% |
| record | methodology.experimental_setup.biological_model.culture_environment.support_type | 1 | 0.22% |
| record | methodology.experimental_setup.biological_model.culture_environment.material_properties | 1 | 0.22% |
| record | methodology.experimental_setup.reagents.0.labeling | 1 | 0.22% |
| record | methodology.experimental_setup.reagents.0.type | 1 | 0.22% |
| record | methodology.experimental_setup.instrumentation | 1 | 0.22% |
| record | outcomes.comparative_advantage.alternative_to | 1 | 0.22% |
| record | outcomes.comparative_advantage.benefit_type | 1 | 0.22% |
| record | outcomes.impact | 1 | 0.22% |
| record | renewal_amount: Monetary amount charged for the renewal | 1 | 0.22% |
| record | coverage_limit: Maximum coverage amount specified in the policy | 1 | 0.22% |
| record | policy_id: Unique identifier assigned to the policy | 1 | 0.22% |
| record | flight_id | 1 | 0.22% |
| record | departure_original | 1 | 0.22% |
| record | departure_actual | 1 | 0.22% |
| record | passenger_address | 1 | 0.22% |
| record | location: Property address including street, city, and state | 1 | 0.22% |
| record | policyholder: Name of the insured individual | 1 | 0.22% |
| record | claim_id: Unique identifier for the insurance claim | 1 | 0.22% |
| record | sender: Name of the company that issued the invoice | 1 | 0.22% |
| record | invoice_number: Unique identifier for the invoice as assigned | 1 | 0.22% |
| record | status: Current financial status of the invoice | 1 | 0.22% |
| record | recipient_address: Street address where goods or services were delivered | 1 | 0.22% |
| record | species.scientific_name | 1 | 0.22% |
| record | species.genus | 1 | 0.22% |
| record | date_added | 1 | 0.22% |
| record | hba1c | 1 | 0.22% |
| record | offer_id: Unique identifier for each leasing offer | 1 | 0.22% |
| record | tenant: Name of the company occupying the leased space | 1 | 0.22% |
| record | location: Full address of the leased property | 1 | 0.22% |
| record | features: List of included amenities or space features | 1 | 0.22% |
| record | ticket_id: Unique identifier for the support case | 1 | 0.22% |
| record | user: Name of the user who filed the ticket | 1 | 0.22% |
| record | address_city: City where patient resides | 1 | 0.22% |
| record | condition: Medical diagnosis associated with symptoms | 1 | 0.22% |
| record | patient_id: Unique identifier for each patient | 1 | 0.22% |
| record | address_street: Street address line of the patient | 1 | 0.22% |
| record | address_state: State where patient resides | 1 | 0.22% |
| record | symptom_location: Body region where symptoms are felt | 1 | 0.22% |
| record | post_id | 1 | 0.22% |
| record | subreddit | 1 | 0.22% |
| record | orbit.orbital_period.value | 1 | 0.22% |
| record | orbit.orbital_type | 1 | 0.22% |
| record | characteristics.diameter.value | 1 | 0.22% |
| record | object_type | 1 | 0.22% |
| record | orbit.semi_major_axis.unit | 1 | 0.22% |
| record | orbit.orbital_period.unit | 1 | 0.22% |
| record | characteristics.diameter.unit | 1 | 0.22% |
| record | tech_product | 1 | 0.22% |
| record | replies | 1 | 0.22% |
| record | author_handle | 1 | 0.22% |
| record | location.district | 1 | 0.22% |
| record | price_range.per | 1 | 0.22% |
| record | location.country | 1 | 0.22% |
| record | booking_options.tripadvisor_role | 1 | 0.22% |
| record | booking_options.partnerships | 1 | 0.22% |
| record | booking_options.assurance_message | 1 | 0.22% |
| record | price_range.currency | 1 | 0.22% |
| record | location.province | 1 | 0.22% |
| record | explanations | 1 | 0.22% |
| record | normal_range | 1 | 0.22% |
| record | dedication.name | 1 | 0.22% |
| record | dedication.description | 1 | 0.22% |
| record | manifest_id | 1 | 0.22% |
| record | sender | 1 | 0.22% |
| record | region_churn | 1 | 0.22% |
| record | project_id | 1 | 0.22% |
| record | open_issues | 1 | 0.22% |
| record | account_id | 1 | 0.22% |
| record | course | 1 | 0.22% |
| record | primates | 1 | 0.22% |
| record | github_id | 1 | 0.22% |
| record | project | 1 | 0.22% |
| record | review_score | 1 | 0.22% |
| record | review_type | 1 | 0.22% |
| record | affected_area | 1 | 0.22% |
| record | key_concepts | 1 | 0.22% |
| record | material: Primary materials used in construction | 1 | 0.22% |
| record | weight: Weight of the product in grams or kg | 1 | 0.22% |
| record | price: Price of the product in dollars | 1 | 0.22% |
| record | scholarship_name | 1 | 0.22% |
| record | recipient | 1 | 0.22% |
| record | major | 1 | 0.22% |
| record | scholarship_id | 1 | 0.22% |
| record | disbursement_plan | 1 | 0.22% |
| record | gpa | 1 | 0.22% |
| record | heroes | 1 | 0.22% |
| record | organizations_and_media | 1 | 0.22% |
| record | villains | 1 | 0.22% |
| record | physical_characteristics.observations.0.telescope.origin | 1 | 0.22% |
| record | asteroid_belt_classification | 1 | 0.22% |
| record | discovery.observatory.location.prefecture | 1 | 0.22% |
| record | route_name | 1 | 0.22% |
| record | positive_note | 1 | 0.22% |
| record | payee: Name of the entity receiving payment | 1 | 0.22% |
| record | description: Brief description of the transaction purpose | 1 | 0.22% |
| record | meeting_id | 1 | 0.22% |
| record | game_id: Unique identifier for the game as listed in the schedule | 1 | 0.22% |
| record | home_score: Final score for the home team | 1 | 0.22% |
| record | venue: Name of the arena where the game was played | 1 | 0.22% |
| record | sports | 1 | 0.22% |
| record | software | 1 | 0.22% |
| record | programStructure.dailySchedule.1.activities.2.description | 1 | 0.22% |
| record | programStructure.dailySchedule.1.activities.2.time | 1 | 0.22% |
| record | accommodation.options.0.website | 1 | 0.22% |
| record | accommodation.options.1.address | 1 | 0.22% |
| record | fees.certificationFee | 1 | 0.22% |
| record | fees.deposit | 1 | 0.22% |
| record | programStructure.dailySchedule.0.activities.2.time | 1 | 0.22% |
| record | programStructure.dailySchedule.1.activities.1.time | 1 | 0.22% |
| record | availability.registration | 1 | 0.22% |
| record | accommodation.options.1.website | 1 | 0.22% |
| record | programStructure.materialsProvided | 1 | 0.22% |
| record | accommodation.options.1.distance | 1 | 0.22% |
| record | programStructure.dailySchedule.0.activities.1.description | 1 | 0.22% |
| record | availability.year | 1 | 0.22% |
| record | accommodation.options.1.name | 1 | 0.22% |
| record | programStructure.dailySchedule.0.activities.3.time | 1 | 0.22% |
| record | programStructure.duration | 1 | 0.22% |
| record | offeredBy | 1 | 0.22% |
| record | programStructure.dailySchedule.1.activities.0.description | 1 | 0.22% |
| record | accommodation.responsibility | 1 | 0.22% |
| record | valueAddedBenefits | 1 | 0.22% |
| record | programStructure.dailySchedule.1.activities.0.time | 1 | 0.22% |
| record | programStructure.dailySchedule.1.activities.3.description | 1 | 0.22% |
| record | programStructure.dailySchedule.0.activities.1.time | 1 | 0.22% |
| record | programName | 1 | 0.22% |
| record | unit_location | 1 | 0.22% |
| record | rental_id | 1 | 0.22% |
| record | current_scale | 1 | 0.22% |
| record | mission | 1 | 0.22% |
| record | content.core_beliefs.relationship_with_society | 1 | 0.22% |
| record | content.elements.name | 1 | 0.22% |
| record | content.elements.music.bands | 1 | 0.22% |
| record | content.elements.cemeteries | 1 | 0.22% |
| record | content.warning | 1 | 0.22% |
| record | expert_opinion.speaker.name | 1 | 0.22% |
| record | expert_opinion.statement | 1 | 0.22% |
| record | expert_opinion.sentiment | 1 | 0.22% |
| record | article_metadata.relative_timestamp | 1 | 0.22% |
| record | handle: Social media handle for the brand | 1 | 0.22% |
| record | target_audience: Demographic group the product targets | 1 | 0.22% |
| record | investors | 1 | 0.22% |
| record | useoffunds | 1 | 0.22% |
| record | policyholder_name | 1 | 0.22% |
| record | policytype | 1 | 0.22% |
| record | product: Name of the skincare or apparel product | 1 | 0.22% |
| record | definition | 1 | 0.22% |
| record | examples | 1 | 0.22% |
| record | comparison.ideographic.cross_lingual_understanding | 1 | 0.22% |
| record | comparison.ideographic.example.character | 1 | 0.22% |
| record | comparison.ideographic.example.languages.0.language | 1 | 0.22% |
| record | comparison.ideographic.example.languages.0.pronunciation | 1 | 0.22% |
| record | comparison.ideographic.example.languages.1.language | 1 | 0.22% |
| record | comparison.ideographic.example.languages.1.pronunciation | 1 | 0.22% |
| record | comparison.ideographic.historical_understanding | 1 | 0.22% |
| record | caveat.example.character | 1 | 0.22% |
| record | caveat.example.pronunciation | 1 | 0.22% |
| record | caveat.example.components.0.related_characters.0.character | 1 | 0.22% |
| record | caveat.example.components.0.related_characters.0.meaning | 1 | 0.22% |
| record | caveat.example.components.0.related_characters.2.character | 1 | 0.22% |
| record | caveat.example.components.1.function | 1 | 0.22% |
| record | caveat.example.components.1.component | 1 | 0.22% |
| record | company_id | 1 | 0.22% |
| record | property_id | 1 | 0.22% |
| record | region.location.extends_to | 1 | 0.22% |
| record | region.location.includes | 1 | 0.22% |
| record | region.landscape | 1 | 0.22% |
| record | date_sent | 1 | 0.22% |
| record | message_id | 1 | 0.22% |
| record | outreach_status | 1 | 0.22% |
| record | job_title | 1 | 0.22% |
| record | director | 1 | 0.22% |
| record | thumbnail | 1 | 0.22% |
| record | rebounds: Rebounds recorded by the player | 1 | 0.22% |
| record | points: Points scored by the player | 1 | 0.22% |
| record | assists: Assists recorded by the player | 1 | 0.22% |
| record | years_mentioned | 1 | 0.22% |
| record | historical_context | 1 | 0.22% |
| record | contact_name | 1 | 0.22% |
| record | invoice_id | 1 | 0.22% |
| record | Edwin Harris Walker <> developed | 1 | 0.22% |
| record | Robert <> identified as | 1 | 0.22% |
| record | Robert <> influenced by | 1 | 0.22% |
| record | Robert <> considered | 1 | 0.22% |
| record | Robert <> experienced | 1 | 0.22% |
| record | Robert <> practiced | 1 | 0.22% |
| record | Robert <> favored | 1 | 0.22% |
| record | Robert <> observed | 1 | 0.22% |
| record | Robert <> valued | 1 | 0.22% |
| record | Robert <> was asked by | 1 | 0.22% |
| record | Robert <> advocated | 1 | 0.22% |
| record | Robert <> interacted with | 1 | 0.22% |
| record | Robert <> questioned | 1 | 0.22% |
| record | Robert <> cited | 1 | 0.22% |
| record | Robert <> utilized | 1 | 0.22% |
| record | description.stipules.dimensions | 1 | 0.22% |
| record | description.leaf.texture | 1 | 0.22% |
| record | family | 1 | 0.22% |
| record | location.other_locations | 1 | 0.22% |
| record | location.endemism | 1 | 0.22% |
| record | honorary_name.field_of_study | 1 | 0.22% |
| record | description.seed.color | 1 | 0.22% |
| record | honorary_name.person | 1 | 0.22% |
| record | description.leaf.base | 1 | 0.22% |
| record | description.type | 1 | 0.22% |
| record | description.leaf.dimensions | 1 | 0.22% |
| record | description.carpels | 1 | 0.22% |
| record | conservation_status.status | 1 | 0.22% |
| record | description.fruit.dimensions | 1 | 0.22% |
| record | description.floral_buds.dimensions | 1 | 0.22% |
| record | habitat.associated_species | 1 | 0.22% |
| record | description.leaf.apex | 1 | 0.22% |
| record | description.seed.dimensions | 1 | 0.22% |
| record | honorary_name.profession | 1 | 0.22% |
| record | match_round: Stage of the tournament such as final, quarterfinal, or last 16 | 1 | 0.22% |
| record | score: Final score of the match in set format | 1 | 0.22% |
| record | venue_name: Name of the specific tournament venue | 1 | 0.22% |
| record | location: Venue where the match was held | 1 | 0.22% |
| record | winner: Player who won the match | 1 | 0.22% |
| record | patient_gender | 1 | 0.22% |
| record | drug_name | 1 | 0.22% |
| record | route | 1 | 0.22% |
| record | clinic_name | 1 | 0.22% |
| record | ndc | 1 | 0.22% |
| record | prescription_id | 1 | 0.22% |
| record | person.death | 1 | 0.22% |
| record | person.title | 1 | 0.22% |
| record | affected_period | 1 | 0.22% |
| record | sender_name | 1 | 0.22% |
| record | sender_company | 1 | 0.22% |
| record | sender_location | 1 | 0.22% |
| record | receiver_name | 1 | 0.22% |
| record | receiver_company | 1 | 0.22% |
| record | receiver_location | 1 | 0.22% |
| record | product_offered | 1 | 0.22% |
| record | release_date | 1 | 0.22% |
| record | problemDescription | 1 | 0.22% |
| record | vehicle.componentReplacements | 1 | 0.22% |
| record | postInformation.date | 1 | 0.22% |
| record | potentialSolutions | 1 | 0.22% |
| record | vehicle.make | 1 | 0.22% |
| record | description: Purpose or context of the contribution | 1 | 0.22% |
| record | donor_name: Full name of the contributing individual or organization | 1 | 0.22% |
| record | recipient_office: Office or position being supported | 1 | 0.22% |
| record | campaign_committee: Name of the political committee receiving the contribution | 1 | 0.22% |
| record | disclosure_id: Unique identifier for the disclosure entry | 1 | 0.22% |
| record | setting.political_context | 1 | 0.22% |
| record | main_character.rank | 1 | 0.22% |
| record | publication_details.first_publication.year | 1 | 0.22% |
| record | publication_details.year_written | 1 | 0.22% |
| record | publication_details.first_publication_czechoslovakia | 1 | 0.22% |
| record | main_character.affiliation | 1 | 0.22% |
| record | comparison | 1 | 0.22% |
| record | original_title | 1 | 0.22% |
| record | bench_points: Total points scored by bench players | 1 | 0.22% |
| record | key_moment: Descriptive note about a pivotal play or event in the game | 1 | 0.22% |
| record | steals: Total steals recorded during the game | 1 | 0.22% |
| record | game_id: Unique identifier for the game as provided in the passage | 1 | 0.22% |
| record | home_score: Final score of the home team | 1 | 0.22% |
| record | reference_number: Unique identifier for the invoice document | 1 | 0.22% |
| record | recipient_name: Name of the recipient organization | 1 | 0.22% |
| record | political_context | 1 | 0.22% |
| record | dateofbirth | 1 | 0.22% |
| record | dose_number | 1 | 0.22% |
| record | account | 1 | 0.22% |
| record | payee | 1 | 0.22% |
| record | overall_score | 1 | 0.22% |
| record | time_management | 1 | 0.22% |
| record | review_status | 1 | 0.22% |
| record | cost_savings | 1 | 0.22% |
| record | collaboration | 1 | 0.22% |
| record | urls | 1 | 0.22% |
| record | enrollment_date | 1 | 0.22% |
| record | merchant: Name of the merchant or store involved | 1 | 0.22% |
| record | status: Current status of the transaction | 1 | 0.22% |
| record | product: Description of the product or service purchased | 1 | 0.22% |
| record | transcription_factors | 1 | 0.22% |
| record | activation_mechanism | 1 | 0.22% |
| record | copyright | 1 | 0.22% |
| record | opponent: Entity filing the dispute or opposition | 1 | 0.22% |
| record | trademark_name: Name of the registered trademark in dispute | 1 | 0.22% |
| record | dispute_type: Type of legal action such as cancellation or infringement | 1 | 0.22% |
| record | registrant: Company that owns the registered trademark | 1 | 0.22% |
| record | registration_number: Unique number assigned to the trademark registration | 1 | 0.22% |
| record | debttoebitda_ratio: Ratio of debt to EBITDA indicating leverage | 1 | 0.22% |
| record | status: Current status of the delay record | 1 | 0.22% |
| record | departure_date: Date flight departs from origin | 1 | 0.22% |
| record | email: Email address of the passenger | 1 | 0.22% |
| record | reason: Cause of the flight delay | 1 | 0.22% |
| record | passenger_name: Name of the passenger on the flight | 1 | 0.22% |
| record | origin: Airport code where flight originates | 1 | 0.22% |
| record | delay_minutes: Number of minutes flight is delayed | 1 | 0.22% |
| record | weather_conditions | 1 | 0.22% |
| record | league | 1 | 0.22% |
| record | guidance | 1 | 0.22% |
| record | cfo | 1 | 0.22% |
| record | games_behind | 1 | 0.22% |
| record | wins | 1 | 0.22% |
| record | games_played | 1 | 0.22% |
| record | losses | 1 | 0.22% |
| record | event_name: Title of the event as listed | 1 | 0.22% |
| record | organizer_name: Name of the person or group organizing the event | 1 | 0.22% |
| record | location: Full address including street, city, and state | 1 | 0.22% |
| record | event_date: Date when the event will take place | 1 | 0.22% |
| record | event_status: Current status of the event | 1 | 0.22% |
| record | attendee_count: Number of registered attendees | 1 | 0.22% |
| record | city: City where the event is held | 1 | 0.22% |
| record | event_code: Unique code identifying the event | 1 | 0.22% |
| record | board_approval_date | 1 | 0.22% |
| record | expected_close_date | 1 | 0.22% |
| record | ceo | 1 | 0.22% |
| record | employees | 1 | 0.22% |
| record | stock_symbol | 1 | 0.22% |
| record | lead_person | 1 | 0.22% |
| record | certification.prerequisites | 1 | 0.22% |
| record | organization.website | 1 | 0.22% |
| record | personnel.0.name | 1 | 0.22% |
| record | certification.exams.0.name | 1 | 0.22% |
| record | certification.exams.1.name | 1 | 0.22% |
| record | additional_information_url | 1 | 0.22% |
| record | certification.name | 1 | 0.22% |
| record | exam_centers | 1 | 0.22% |
| record | certification.description | 1 | 0.22% |
| record | personnel.0.organization | 1 | 0.22% |
| record | personnel.0.title | 1 | 0.22% |
| record | training.delivery_method | 1 | 0.22% |
| record | product.name | 1 | 0.22% |
| record | product.brand | 1 | 0.22% |
| record | product.type | 1 | 0.22% |
| record | product.description | 1 | 0.22% |
| record | product.size | 1 | 0.22% |
| record | product.ingredients | 1 | 0.22% |
| record | review.overall_rating | 1 | 0.22% |
| record | review.pros | 1 | 0.22% |
| record | review.cons | 1 | 0.22% |
| record | review.price | 1 | 0.22% |
| record | review.usage_experience | 1 | 0.22% |
| record | place.name | 1 | 0.22% |
| record | album.external_links.1.url | 1 | 0.22% |
| record | album.chart_performance.record_holder_description | 1 | 0.22% |
| record | album.tracklist.cd.0.notes | 1 | 0.22% |
| record | album.tracklist.cd.2.tie_in.0.type | 1 | 0.22% |
| record | album.external_links.3.url | 1 | 0.22% |
| record | album.tracklist.cd.1.title | 1 | 0.22% |
| record | album.tracklist.cd.7.title | 1 | 0.22% |
| record | album.tracklist.cd.9.track_number | 1 | 0.22% |
| record | album.tracklist.cd.4.tie_in.0.title | 1 | 0.22% |
| record | album.free_concert_location | 1 | 0.22% |
| record | album.artist | 1 | 0.22% |
| record | album.tracklist.cd.7.lyrics | 1 | 0.22% |
| record | album.tracklist.cd.0.title | 1 | 0.22% |
| record | album.tracklist.dvd.1.type | 1 | 0.22% |
| record | album.tracklist.cd.1.lyrics | 1 | 0.22% |
| record | album.sources.3.source_name | 1 | 0.22% |
| record | album.tracklist.cd.9.tie_in.0.title | 1 | 0.22% |
| record | album.tracklist.cd.6.title | 1 | 0.22% |
| record | album.versions.0.type | 1 | 0.22% |
| record | album.versions.1.name | 1 | 0.22% |
| record | album.tracklist.cd.3.notes | 1 | 0.22% |
| record | album.tracklist.cd.2.tie_in.0.title | 1 | 0.22% |
| record | album.external_links.1.description | 1 | 0.22% |
| record | album.anniversary | 1 | 0.22% |
| record | album.tracklist.cd.1.notes | 1 | 0.22% |
| record | album.tracklist.dvd.2.type | 1 | 0.22% |
| record | album.external_links.0.url | 1 | 0.22% |
| record | album.tracklist.cd.2.title | 1 | 0.22% |
| record | album.tracklist.cd.8.tie_in.0.title | 1 | 0.22% |
| record | album.sources.1.access_date | 1 | 0.22% |
| record | album.tracklist.cd.8.composer | 1 | 0.22% |
| record | album.sources.2.publication_date | 1 | 0.22% |
| record | album.tracklist.cd.9.lyrics | 1 | 0.22% |
| record | album.tracklist.cd.3.title | 1 | 0.22% |
| record | album.tracklist.cd.7.composer | 1 | 0.22% |
| record | album.versions.3.type | 1 | 0.22% |
| record | album.tracklist.cd.2.notes | 1 | 0.22% |
| record | album.tracklist.dvd.0.type | 1 | 0.22% |
| record | album.release_date | 1 | 0.22% |
| record | album.tracklist.cd.2.track_number | 1 | 0.22% |
| record | album.sources.3.publication_date | 1 | 0.22% |
| record | album.sources.1.publication_date | 1 | 0.22% |
| record | album.tracklist.cd.4.tie_in.0.type | 1 | 0.22% |
| record | album.tracklist.cd.4.track_number | 1 | 0.22% |
| record | album.versions.1.contents | 1 | 0.22% |
| record | album.tracklist.cd.6.track_number | 1 | 0.22% |
| record | album.chart_performance.record_holder_rank | 1 | 0.22% |
| record | album.tracklist.cd.1.track_number | 1 | 0.22% |
| record | album.tracklist.cd.5.composer | 1 | 0.22% |
| record | album.chart_performance.consecutive_number_one_albums | 1 | 0.22% |
| record | album.tracklist.cd.0.tie_in.0.type | 1 | 0.22% |
| record | album.versions.2.contents | 1 | 0.22% |
| record | album.external_links.2.description | 1 | 0.22% |
| record | album.tracklist.cd.8.track_number | 1 | 0.22% |
| record | album.sources.2.access_date | 1 | 0.22% |
| record | album.tracklist.dvd.1.track_number | 1 | 0.22% |
| record | album.tracklist.dvd.1.title | 1 | 0.22% |
| record | album.tracklist.cd.8.notes | 1 | 0.22% |
| record | album.title | 1 | 0.22% |
| record | album.sources.3.reference_number | 1 | 0.22% |
| record | album.tracklist.cd.2.composer | 1 | 0.22% |
| record | album.tracklist.cd.4.composer | 1 | 0.22% |
| record | album.tracklist.cd.9.tie_in.0.type | 1 | 0.22% |
| record | album.external_links.2.url | 1 | 0.22% |
| record | album.tracklist.dvd.2.title | 1 | 0.22% |
| record | album.sources.0.publication_date | 1 | 0.22% |
| record | album.tracklist.cd.4.notes | 1 | 0.22% |
| record | album.versions.3.name | 1 | 0.22% |
| record | album.tracklist.cd.8.tie_in.0.type | 1 | 0.22% |
| record | album.tracklist.cd.7.track_number | 1 | 0.22% |
| record | album.tracklist.dvd.0.title | 1 | 0.22% |
| record | album.tracklist.cd.3.track_number | 1 | 0.22% |
| record | channel_name | 1 | 0.22% |
| record | thumbnail_description | 1 | 0.22% |
| record | target_audience | 1 | 0.22% |
| record | report_date | 1 | 0.22% |
| record | office | 1 | 0.22% |
| record | appointment_id | 1 | 0.22% |
| record | hotel.transportation.parking.0.name | 1 | 0.22% |
| record | hotel.location.city | 1 | 0.22% |
| record | hotel.star_rating | 1 | 0.22% |
| record | hotel.transportation.driving_directions | 1 | 0.22% |
| record | hotel.transportation.airport | 1 | 0.22% |
| record | hotel.location.country | 1 | 0.22% |
| record | hotel.transportation.parking.0.walking_time | 1 | 0.22% |
| record | hotel.location.nearby_landmarks | 1 | 0.22% |
| record | hotel.transportation.parking.1.cost | 1 | 0.22% |
| record | hotel.location.postal_code | 1 | 0.22% |
| record | hotel.location.area_description | 1 | 0.22% |
| record | hotel.transportation.parking.1.name | 1 | 0.22% |
| record | hotel.transportation.parking.1.walking_time | 1 | 0.22% |
| record | hotel.contact.phone | 1 | 0.22% |
| record | hotel.transportation.parking.0.cost | 1 | 0.22% |
| record | early_challenges | 1 | 0.22% |
| record | exit_status | 1 | 0.22% |
| record | founder_name | 1 | 0.22% |
| record | founding_date | 1 | 0.22% |
| record | investment_source | 1 | 0.22% |
| record | weight | 1 | 0.22% |
| record | available_units | 1 | 0.22% |
| record | stock_status | 1 | 0.22% |
| record | shipping | 1 | 0.22% |
| record | color | 1 | 0.22% |
| record | warranty | 1 | 0.22% |
| record | break_points | 1 | 0.22% |
| record | tournament_code | 1 | 0.22% |
| record | winner | 1 | 0.22% |
| record | aces | 1 | 0.22% |
| record | url_references | 1 | 0.22% |
| record | version_number | 1 | 0.22% |
| record | enforceability_status | 1 | 0.22% |
| record | modification_flag | 1 | 0.22% |
| record | document_type | 1 | 0.22% |
| record | effective_date | 1 | 0.22% |
| record | breach_penalty_percent | 1 | 0.22% |
| record | compliance_code | 1 | 0.22% |
| record | audit_review_date | 1 | 0.22% |
| record | legal_agent | 1 | 0.22% |
| record | contract_reference | 1 | 0.22% |
| record | notice_period_days | 1 | 0.22% |
| record | term_start_date | 1 | 0.22% |
| record | clause_description | 1 | 0.22% |
| record | amount_usd | 1 | 0.22% |
| record | counterparty_address | 1 | 0.22% |
| record | party_name | 1 | 0.22% |
| record | monitoring_mechanism | 1 | 0.22% |
| record | goal | 1 | 0.22% |
| record | signatories | 1 | 0.22% |
| record | treaty_name | 1 | 0.22% |
| record | target_year | 1 | 0.22% |
| record | geographic_scope | 1 | 0.22% |
| record | price: Current selling price of the product in USD | 1 | 0.22% |
| record | last_updated: Date when product record was last modified | 1 | 0.22% |
| record | message_id: Unique identifier for each outreach message | 1 | 0.22% |
| record | sender: Name of the recruiter or agency sending the message | 1 | 0.22% |
| record | recipient: Name of the recipient at the target company | 1 | 0.22% |
| record | company: Name of the company where the recipient works | 1 | 0.22% |
| record | role: Job title or position being targeted | 1 | 0.22% |
| record | subject: Title or subject line of the outreach message | 1 | 0.22% |
| record | outreach_amount: Monetary value associated with the outreach | 1 | 0.22% |
| record | player_name | 1 | 0.22% |
| record | steals | 1 | 0.22% |
| record | home_score | 1 | 0.22% |
| record | player_id | 1 | 0.22% |
| record | start_date | 1 | 0.22% |
| record | canton_name | 1 | 0.22% |
| record | canton_name_french | 1 | 0.22% |
| record | department | 1 | 0.22% |
| record | municipalities | 1 | 0.22% |
| record | issue_type: Type of technical problem reported | 1 | 0.22% |
| record | submitted_date: Date the ticket was submitted | 1 | 0.22% |
| record | severity: Priority level of the issue | 1 | 0.22% |
| record | employee_name: Name of employee who reported the issue | 1 | 0.22% |
| record | ticket_id: Unique identifier for each service ticket | 1 | 0.22% |
| record | department: Department where the issue originated | 1 | 0.22% |
| record | csds_id | 1 | 0.22% |
| record | manufacturer_address | 1 | 0.22% |
| record | property_id: Unique identifier assigned to the rental unit | 1 | 0.22% |
| record | city: City where the property is located | 1 | 0.22% |
| record | lease_term: Type of lease term, e.g., month-to-month | 1 | 0.22% |
| record | management_contact: Name or entity managing the property | 1 | 0.22% |
| record | hotel_name | 1 | 0.22% |
| record | location.distance_from_city_center.value | 1 | 0.22% |
| record | location.distance_from_city_center.unit | 1 | 0.22% |
| record | location.distance_from_airport.value | 1 | 0.22% |
| record | location.distance_from_airport.unit | 1 | 0.22% |
| record | location.airport_travel_time.value | 1 | 0.22% |
| record | location.nearby_attractions | 1 | 0.22% |
| record | hotel_details.star_rating | 1 | 0.22% |
| record | hotel_details.year_built | 1 | 0.22% |
| record | hotel_details.year_renovated | 1 | 0.22% |
| record | hotel_details.check_in | 1 | 0.22% |
| record | hotel_details.check_out | 1 | 0.22% |
| record | hotel_details.reception_open_until | 1 | 0.22% |
| record | hotel_details.number_of_rooms | 1 | 0.22% |
| record | hotel_details.room_amenities | 1 | 0.22% |
| record | hotel_details.recreation | 1 | 0.22% |
| record | child_policy.infants.age_range | 1 | 0.22% |
| record | child_policy.children.age_range | 1 | 0.22% |
| record | child_policy.adult_age | 1 | 0.22% |
| record | child_policy.extra_bed_policy | 1 | 0.22% |
| record | booking_policy.multiple_rooms | 1 | 0.22% |
| record | property_type | 1 | 0.22% |
| record | check_in_process | 1 | 0.22% |
| record | owner_contact | 1 | 0.22% |
| record | sessions | 1 | 0.22% |
| record | crime.location.landmark | 1 | 0.22% |
| record | perpetrator.status | 1 | 0.22% |
| record | crime.location.address | 1 | 0.22% |
| record | perpetrator.arrest_details.agency | 1 | 0.22% |
| record | crime.location.sector | 1 | 0.22% |
| record | crime.motive | 1 | 0.22% |
| record | victim.name | 1 | 0.22% |
| record | investigation.agency | 1 | 0.22% |
| record | crime.weapon.type | 1 | 0.22% |
| record | crime.weapon.caliber | 1 | 0.22% |
| record | perpetrator.arrest_details.time | 1 | 0.22% |
| record | crime.date | 1 | 0.22% |
| record | investigation.medical_examiner.name | 1 | 0.22% |
| record | crime.weapon.source.stolen_from.name | 1 | 0.22% |
| record | investigation.status | 1 | 0.22% |
| record | event | 1 | 0.22% |
| record | sales | 1 | 0.22% |
| record | publisher | 1 | 0.22% |
| record | platform | 1 | 0.22% |
| record | coverage_amount | 1 | 0.22% |
| record | owner_name | 1 | 0.22% |
| record | term_years | 1 | 0.22% |
| record | storage | 1 | 0.22% |
| record | asteroid.discovery.discoverer | 1 | 0.22% |
| record | asteroid.discovery.location | 1 | 0.22% |
| record | leasing_offer | 1 | 0.22% |
| record | manager | 1 | 0.22% |
| record | onboarding_code | 1 | 0.22% |
| record | customer_address | 1 | 0.22% |
| record | diagnosis_code | 1 | 0.22% |
| record | asteroid_name | 1 | 0.22% |
| record | promotions | 1 | 0.22% |
| record | publicationDate | 1 | 0.22% |
| record | mechanismofaction: Biological process inhibited or modulated by the drug | 1 | 0.22% |
| record | lastreviewdate: Date the drug record was last reviewed | 1 | 0.22% |
| record | adverse_effects: List of side effects reported in clinical use | 1 | 0.22% |
| record | molecular_weight: Molecular weight of the drug in daltons | 1 | 0.22% |
| record | drug_name: Name of the pharmaceutical product | 1 | 0.22% |
| record | clinical_phase: Current stage of clinical development | 1 | 0.22% |
| record | error.message | 1 | 0.22% |
| record | error.possible_causes | 1 | 0.22% |
| record | asteroid.discovery_year | 1 | 0.22% |
| record | asteroid.orbital_characteristics.unit | 1 | 0.22% |
| record | asteroid.orbital_characteristics.inclination_unit | 1 | 0.22% |
| record | deal_id | 1 | 0.22% |
| record | valuation_multiple | 1 | 0.22% |
| record | owner: Name of the healthcare professional owning the device | 1 | 0.22% |
| record | model: Manufacturer model name of the medical device | 1 | 0.22% |
| record | serial: Unique serial number identifying the device | 1 | 0.22% |
| record | vendor: Company that sold or provided the device | 1 | 0.22% |
| record | location: Department or unit where device is currently located | 1 | 0.22% |
| record | status: Current operational status of the device | 1 | 0.22% |
| record | item_description.setup_charge.amount | 1 | 0.22% |
| record | item_description.shipping.ready_to_ship_in.quantity | 1 | 0.22% |
| record | item_description.features | 1 | 0.22% |
| record | item_description.shipping.rush_details | 1 | 0.22% |
| record | item_description.item_id | 1 | 0.22% |
| record | item_description.weight.value | 1 | 0.22% |
| record | item_description.shipping.ready_to_ship_in.unit | 1 | 0.22% |
| record | item_description.name | 1 | 0.22% |
| record | additional_value: Extra numeric value associated with the transaction | 1 | 0.22% |
| record | status: Current transaction state | 1 | 0.22% |
| record | address: Full physical address of the owner | 1 | 0.22% |
| record | date: Date the transaction was initiated | 1 | 0.22% |
| record | transaction_id: Unique identifier for the transaction | 1 | 0.22% |
| record | discovery.observatory.location.county | 1 | 0.22% |
| record | discovery.observatory.location.state | 1 | 0.22% |
| record | orbital_characteristics.semi_major_axis.unit | 1 | 0.22% |
| record | orbital_characteristics.inclination.unit | 1 | 0.22% |
| record | orbital_characteristics.inclination.reference | 1 | 0.22% |
| record | orbital_characteristics.asteroid_belt_classification | 1 | 0.22% |
| record | physical_characteristics.observations.0.telescope | 1 | 0.22% |
| record | physical_characteristics.observations.0.year | 1 | 0.22% |
| record | author: Name of the guideline's lead author | 1 | 0.22% |
| record | target_population: Group of patients or individuals the guideline targets | 1 | 0.22% |
| record | institution: Affiliation of the guideline's author | 1 | 0.22% |
| record | description.rosette | 1 | 0.22% |
| record | description.leaves.color | 1 | 0.22% |
| record | systematics_and_distribution.first_description.year | 1 | 0.22% |
| record | systematics_and_distribution.distinguishing_features | 1 | 0.22% |
| record | systematics_and_distribution.location.elevation_m.max | 1 | 0.22% |
| record | description.leaves.length_cm.min | 1 | 0.22% |
| record | description.flowers.length_mm.min | 1 | 0.22% |
| record | description.fruit.type | 1 | 0.22% |
| record | target_department | 1 | 0.22% |
| record | spoofed_domain | 1 | 0.22% |
| record | chemical_name | 1 | 0.22% |
| record | cas_number | 1 | 0.22% |
| record | emergency_contact | 1 | 0.22% |
| record | patents | 1 | 0.22% |
| record | value_original | 1 | 0.22% |
| record | regulatory_status | 1 | 0.22% |
| record | earnout | 1 | 0.22% |
| record | customer_contracts | 1 | 0.22% |
| record | transaction_id | 1 | 0.22% |
| record | people | 1 | 0.22% |
| record | dates_mentioned | 1 | 0.22% |
| record | topics | 1 | 0.22% |
| record | points | 1 | 0.22% |
| record | rebounds | 1 | 0.22% |
| record | player | 1 | 0.22% |
| record | content | 1 | 0.22% |
| record | comment_id | 1 | 0.22% |
| record | refund_amount | 1 | 0.22% |
| record | order_id | 1 | 0.22% |
| record | sponsor: Individual legislator who introduced the bill | 1 | 0.22% |
| record | yes_votes: Number of members voting in favor | 1 | 0.22% |
| record | no_votes: Number of members voting against | 1 | 0.22% |
| record | case_number: Unique identifier for the court case | 1 | 0.22% |
| record | plaintiff: Individual or entity initiating the lawsuit | 1 | 0.22% |
| record | defendant: Individual or entity being sued | 1 | 0.22% |
| record | damages: Monetary amount sought in damages | 1 | 0.22% |
| record | chamber | 1 | 0.22% |
| record | bill_number | 1 | 0.22% |
| record | vote_result | 1 | 0.22% |
| record | proposer | 1 | 0.22% |
| record | rsvp_count | 1 | 0.22% |
| record | brands | 1 | 0.22% |
| record | party | 1 | 0.22% |
| record | vote_percentage | 1 | 0.22% |
| record | marginofvictory | 1 | 0.22% |
| record | ballot_type | 1 | 0.22% |
| record | electoral_code | 1 | 0.22% |
| record | queue | 1 | 0.22% |
| record | patient.behavior | 1 | 0.22% |
| record | patient.request | 1 | 0.22% |
| record | patient.age_estimation.initial_estimation | 1 | 0.22% |
| record | patient.age_estimation.correction | 1 | 0.22% |
| record | patient.age_estimation.full_age | 1 | 0.22% |
| record | patient.age_estimation.previous_state | 1 | 0.22% |
| record | doctor.action | 1 | 0.22% |
| record | doctor.question | 1 | 0.22% |
| record | doctor.reaction | 1 | 0.22% |
| record | chair.state | 1 | 0.22% |
| record | chair.condition | 1 | 0.22% |
| record | fiscalyearend | 1 | 0.22% |
| record | address: Physical address of the appointee | 1 | 0.22% |
| record | notes: Additional context about the appointment or background | 1 | 0.22% |
| record | name: Full name of the appointed individual | 1 | 0.22% |
| record | date: Start date of the appointment | 1 | 0.22% |
| record | department: Government department or agency where the role is held | 1 | 0.22% |
| record | position: Job title or role within the department | 1 | 0.22% |
| record | recharge_amount | 1 | 0.22% |
| record | articleTitle | 1 | 0.22% |
| record | affiliation.location | 1 | 0.22% |
| record | research_background.focus_area | 1 | 0.22% |
| record | goals | 1 | 0.22% |
| record | nutritional_aspects.sucrose.daily_limit_percentage | 1 | 0.22% |
| record | nutritional_aspects.sucrose.daily_limit_grams | 1 | 0.22% |
| record | weight_loss_benefits.improvements | 1 | 0.22% |
| record | nutritional_aspects.carbohydrates.daily_percentage | 1 | 0.22% |
| record | nutritional_aspects.carbohydrates.importance | 1 | 0.22% |
| record | status: Current operational status | 1 | 0.22% |
| record | employees: Number of employees | 1 | 0.22% |
| record | research.experiment_details.quantum_state | 1 | 0.22% |
| record | research.focus | 1 | 0.22% |
| record | research.researcher.affiliation.name | 1 | 0.22% |
| record | research.experiment_details.setup | 1 | 0.22% |
| record | research.objective | 1 | 0.22% |
| record | research.experiment_details.outcome | 1 | 0.22% |
| record | research.materials | 1 | 0.22% |
| record | research.researcher.affiliation.abbreviation | 1 | 0.22% |
| record | research.temperature | 1 | 0.22% |
| record | research.experiment_details.temperature_details | 1 | 0.22% |
| record | research.researcher.name | 1 | 0.22% |
| record | ticket_id: Unique identifier for the return ticket | 1 | 0.22% |
| record | owner: Name of the customer filing the ticket | 1 | 0.22% |
| record | issue: Description of the reported hardware failure | 1 | 0.22% |
| record | location: City and state where the customer resides | 1 | 0.22% |
| record | building | 1 | 0.22% |
| record | lease_term | 1 | 0.22% |
| record | related_articles | 1 | 0.22% |
| record | location: Physical address where course is held | 1 | 0.22% |
| record | duration: Number of weeks the course runs | 1 | 0.22% |
| record | credits: Number of academic credits assigned to the course | 1 | 0.22% |
| record | cost: Total cost of enrollment in US dollars | 1 | 0.22% |
| record | repair_shop | 1 | 0.22% |
| record | preserved_specimens | 1 | 0.22% |
| record | imaging_type | 1 | 0.22% |
| record | referring_physician | 1 | 0.22% |
| record | facility_name | 1 | 0.22% |
| record | facility_address | 1 | 0.22% |
| record | study_code | 1 | 0.22% |
| record | finding_description | 1 | 0.22% |
| record | clinical_status | 1 | 0.22% |
| record | study_protocol | 1 | 0.22% |
| record | funding_amount: Total amount raised in millions of dollars | 1 | 0.22% |
| record | co_investors: List of additional investors participating in the round | 1 | 0.22% |
| record | focus_area: Primary technological or business focus of the startup | 1 | 0.22% |
| record | investor_count: Number of investors involved in the round | 1 | 0.22% |
| record | affected_site | 1 | 0.22% |
| record | affected_network | 1 | 0.22% |
| record | alternative_access | 1 | 0.22% |
| record | tenant_name: Name of the leasing company or business | 1 | 0.22% |
| record | lease_id: Unique identifier for each lease agreement | 1 | 0.22% |
| record | briefing_id | 1 | 0.22% |
| record | exhibitor_id | 1 | 0.22% |
| record | contact_email | 1 | 0.22% |
| record | project_id: Unique identifier assigned to the technology initiative | 1 | 0.22% |
| record | developers: List of institutions or individuals leading the project | 1 | 0.22% |
| record | funding_source: Organization or fund providing financial support | 1 | 0.22% |
| record | technology_name: Name of the algorithm, platform, or device | 1 | 0.22% |
| record | deployment_location: Geographic region where system is deployed | 1 | 0.22% |
| record | deployment_count: Number of farms, hospitals, or states using the system | 1 | 0.22% |
| record | sensor_nodes: Number of data collection nodes used in the system | 1 | 0.22% |
| record | magnitudeorindex | 1 | 0.22% |
| record | product_name | 1 | 0.22% |
| record | model_number | 1 | 0.22% |
| record | patient_province | 1 | 0.22% |
| record | followupaction | 1 | 0.22% |
| record | reporting_doctor | 1 | 0.22% |
| record | division | 1 | 0.22% |
| record | century | 1 | 0.22% |
| record | images | 1 | 0.22% |
| record | dobj | 1 | 0.22% |
| record | nmod | 1 | 0.22% |
| record | cc | 1 | 0.22% |
| record | preconj | 1 | 0.22% |
| record | case | 1 | 0.22% |
| record | npadvmod | 1 | 0.22% |
| record | intj | 1 | 0.22% |
| record | nummod | 1 | 0.22% |
| record | donor_id | 1 | 0.22% |
| record | establishment_year | 1 | 0.22% |
| record | staff.teaching_staff_count | 1 | 0.22% |
| record | faculty_name | 1 | 0.22% |
| record | map_location | 1 | 0.22% |
| record | campus | 1 | 0.22% |
| record | departments | 1 | 0.22% |
| record | dean.name | 1 | 0.22% |
| record | festival_id | 1 | 0.22% |
| record | artist | 1 | 0.22% |
| record | genre | 1 | 0.22% |
| record | ticket_price | 1 | 0.22% |
| record | age_restriction | 1 | 0.22% |
| record | audience_rating | 1 | 0.22% |
| record | performance_time | 1 | 0.22% |
| record | alternative_names | 1 | 0.22% |
| record | implementation_notes | 1 | 0.22% |
| record | guideline_code | 1 | 0.22% |
| record | cost_impact | 1 | 0.22% |
| record | symbol | 1 | 0.22% |
| record | performance_reason | 1 | 0.22% |
| record | change | 1 | 0.22% |
| record | question.text | 1 | 0.22% |
| record | question.keywords | 1 | 0.22% |
| record | question.legal_implications | 1 | 0.22% |
| record | details.date_posted | 1 | 0.22% |
| record | details.days_ago | 1 | 0.22% |
| record | activities | 1 | 0.22% |
| record | relationships.0.action | 1 | 0.22% |
| record | product_id: Unique identifier for each product | 1 | 0.22% |
| record | stock: Current quantity available in inventory | 1 | 0.22% |
| record | last_updated: Date the product record was last modified | 1 | 0.22% |
| record | sales_volume: Total units sold over time | 1 | 0.22% |
| record | status: Current product availability status | 1 | 0.22% |
| record | description: Brief product feature and use description | 1 | 0.22% |
| record | diagnosis: ICD-10 code representing the medical condition | 1 | 0.22% |
| record | cpt_code: CPT code for the medical procedure performed | 1 | 0.22% |
| record | claim_date: Date the claim was filed with the insurer | 1 | 0.22% |
| record | temperature | 1 | 0.22% |
| record | respiratory_rate | 1 | 0.22% |
| record | emergency_treatment | 1 | 0.22% |
| record | when_to_seek_help | 1 | 0.22% |
| record | codes | 1 | 0.22% |
| record | prevention | 1 | 0.22% |
| record | last_updated | 1 | 0.22% |
| record | founder | 1 | 0.22% |
| record | patient_id: Unique identifier for the patient entry | 1 | 0.22% |
| record | status: Current status of vaccination record | 1 | 0.22% |
| record | vaccine_type: Type of vaccine administered | 1 | 0.22% |
| record | sales_period | 1 | 0.22% |
| record | sales_platform | 1 | 0.22% |
| record | sale_amount | 1 | 0.22% |
| record | turning_point | 1 | 0.22% |
| record | trend_type | 1 | 0.22% |
| record | repost_count | 1 | 0.22% |
| record | user_handle | 1 | 0.22% |
| record | event_id: Unique identifier for each event | 1 | 0.22% |
| record | organizer: Person or group hosting the event | 1 | 0.22% |
| record | location: Physical address or venue of the event | 1 | 0.22% |
| record | price: Entry cost for attending the event | 1 | 0.22% |
| record | student_name: Name of the student who took the exam | 1 | 0.22% |
| record | subject: Subject area of the exam | 1 | 0.22% |
| record | publication_id: Unique identifier for the exam publication | 1 | 0.22% |
| record | status: Current status of the exam result | 1 | 0.22% |
| record | definition.source_organization.name | 1 | 0.22% |
| record | business_sustainability.elements | 1 | 0.22% |
| record | industry_example.industry | 1 | 0.22% |
| record | crisis_resilience | 1 | 0.22% |
| record | definition.source_organization.full_name | 1 | 0.22% |
| record | financial_aspect.necessity | 1 | 0.22% |
| record | definition.document_title | 1 | 0.22% |
| record | three_pillars | 1 | 0.22% |
| record | industry_example.challenges | 1 | 0.22% |
| record | services.maintenance | 1 | 0.22% |
| record | services.selection_assistance | 1 | 0.22% |
| record | distributor_information.capabilities | 1 | 0.22% |
| record | actions | 1 | 0.22% |
| record | filer: Name of the person or entity filing the case | 1 | 0.22% |
| record | intendedAudience | 1 | 0.22% |
| record | accessInstructions | 1 | 0.22% |
| record | budget_remaining | 1 | 0.22% |
| record | actual_spend | 1 | 0.22% |
| record | remaining_budget | 1 | 0.22% |
| record | onboarding_sprint_completion | 1 | 0.22% |
| record | risk_code | 1 | 0.22% |
| record | backend_integration_status | 1 | 0.22% |
| record | recipient_organization | 1 | 0.22% |
| record | donor_name | 1 | 0.22% |
| record | street | 1 | 0.22% |
| record | zip | 1 | 0.22% |
| record | file_id | 1 | 0.22% |
| record | panel_topics | 1 | 0.22% |
| record | registration_fee | 1 | 0.22% |
| record | registration_cap | 1 | 0.22% |
| record | salary_range: Expected or offered salary range for the role | 1 | 0.22% |
| record | target_role: Job title or position being targeted | 1 | 0.22% |
| record | message_id: Unique identifier for the outreach message | 1 | 0.22% |
| record | target_company: Company where the role is located | 1 | 0.22% |
| record | expert_insights | 1 | 0.22% |
| record | training_program.invitation_steps | 1 | 0.22% |
| record | training_program.payment.options | 1 | 0.22% |
| record | organization.name | 1 | 0.22% |
| record | training_program.payment.payment_instructions | 1 | 0.22% |
| record | training_program.payment.deposit_amount | 1 | 0.22% |
| record | company: Name of the technology company raising funds | 1 | 0.22% |
| record | lead_investor: List of lead investors or funds involved | 1 | 0.22% |
| record | serial_number: Unique serial number for device tracking | 1 | 0.22% |
| record | usage_count: Total patient admissions or scans performed | 1 | 0.22% |
| record | office_hours.2.start_time | 1 | 0.22% |
| record | contact.fax | 1 | 0.22% |
| record | office_hours.3.start_time | 1 | 0.22% |
| record | address.street | 1 | 0.22% |
| record | office_hours.4.day | 1 | 0.22% |
| record | office_hours.2.day | 1 | 0.22% |
| record | office_hours.2.end_time | 1 | 0.22% |
| record | office_hours.4.end_time | 1 | 0.22% |
| record | office_hours.4.start_time | 1 | 0.22% |
| record | office_hours.3.end_time | 1 | 0.22% |
| record | office_hours.3.day | 1 | 0.22% |
| record | office_hours.0.end_time | 1 | 0.22% |
| record | date: Date of the patient visit | 1 | 0.22% |
| record | spo2: Oxygen saturation percentage | 1 | 0.22% |
| record | timestamp: Time of measurement in HH:MM:SS format | 1 | 0.22% |
| record | temperature: Body temperature in degrees Fahrenheit | 1 | 0.22% |
| record | blood_pressure: Blood pressure in format systolic/diastolic | 1 | 0.22% |
| record | start_term | 1 | 0.22% |
| record | hall | 1 | 0.22% |
| record | course_code | 1 | 0.22% |
| record | enrollment | 1 | 0.22% |
| record | region.architecture | 1 | 0.22% |
| record | region.location.extends_from | 1 | 0.22% |
| record | region.route | 1 | 0.22% |
| record | app_id | 1 | 0.22% |
| record | os_version | 1 | 0.22% |
| record | word_count | 1 | 0.22% |
| record | helpful_votes | 1 | 0.22% |
| record | language | 1 | 0.22% |
| record | review_platform | 1 | 0.22% |
| record | review_link | 1 | 0.22% |
| record | players | 1 | 0.22% |
| record | venue_id | 1 | 0.22% |
| record | merchant | 1 | 0.22% |
| record | program_code | 1 | 0.22% |
| record | useoffunds: Purpose or application of raised capital | 1 | 0.22% |
| record | investors: List of investor names or groups involved | 1 | 0.22% |
| record | funding_amount: Total amount raised in millions | 1 | 0.22% |
| record | funding_id: Unique identifier assigned to the funding round | 1 | 0.22% |
| record | bp | 1 | 0.22% |
| record | doctor | 1 | 0.22% |
| record | rr | 1 | 0.22% |
| record | dish | 1 | 0.22% |
| record | kitchen appliance | 1 | 0.22% |
| record | food item | 1 | 0.22% |
| record | ingredient | 1 | 0.22% |
| record | side dish | 1 | 0.22% |
| record | underwriting_notes | 1 | 0.22% |
| record | lead_researcher | 1 | 0.22% |
| record | intervention | 1 | 0.22% |
| record | patient_count | 1 | 0.22% |
| record | product_line | 1 | 0.22% |
| record | brief_id | 1 | 0.22% |
| record | app_name | 1 | 0.22% |
| record | location: City of origin or primary market | 1 | 0.22% |
| record | features: List of technical capabilities or functions | 1 | 0.22% |
| record | retail_price: Final price at retail in US dollars | 1 | 0.22% |
| record | comment | 1 | 0.22% |
| record | primary_outcome | 1 | 0.22% |
| record | study_start | 1 | 0.22% |
| record | affiliations | 1 | 0.22% |
| record | sample_size | 1 | 0.22% |
| record | study_id | 1 | 0.22% |
| record | key_result | 1 | 0.22% |
| record | outreach_date: Date the outreach was sent | 1 | 0.22% |
| record | sender: Name of the person initiating the outreach | 1 | 0.22% |
| record | status: Current status of the outreach attempt | 1 | 0.22% |
| record | recipient: Name of the recipient being contacted | 1 | 0.22% |
| record | outreach_code: Code associated with the outreach event | 1 | 0.22% |
| record | hb: Hemoglobin level in grams per deciliter | 1 | 0.22% |
| record | test_date: Date the lab test was conducted | 1 | 0.22% |
| record | patient_name: Name of the patient undergoing the test | 1 | 0.22% |
| record | platelets: Platelet count in thousands per microliter | 1 | 0.22% |
| record | lot_number | 1 | 0.22% |
| record | keynote_time | 1 | 0.22% |
| record | keynote_topic | 1 | 0.22% |
| record | amount: Total amount raised in millions | 1 | 0.22% |
| record | company: Name of the company raising capital | 1 | 0.22% |
| record | nf_id: Unique identifier for each funding event | 1 | 0.22% |
| record | round: Series or type of funding round | 1 | 0.22% |
| record | investors: List of venture capital firms involved | 1 | 0.22% |
| record | return_id | 1 | 0.22% |
| record | vaccination_status | 1 | 0.22% |
| record | encounter_id | 1 | 0.22% |
| record | ratingcategory | 1 | 0.22% |
| record | buyer: Name of the customer who made the purchase | 1 | 0.22% |
| record | location: City and state where buyer resides | 1 | 0.22% |
| record | product: Product model identifier such as EBC-1234 or WAT-5678 | 1 | 0.22% |
| record | price: Price paid for the product in US dollars | 1 | 0.22% |
| record | rating: Star rating from 1 to 5 given by the buyer | 1 | 0.22% |
| record | shipping_method: Type of shipping used | 1 | 0.22% |
| record | tracking_number: Shipping tracking number assigned | 1 | 0.22% |
| record | order_id: Unique order identifier from the retailer | 1 | 0.22% |
| record | return_status: Whether the product was returned or kept | 1 | 0.22% |
| record | feedback: Additional comments or issues noted in review | 1 | 0.22% |
| record | filing_date: Date the patent was filed | 1 | 0.22% |
| record | inventor: Name of the patent's inventor | 1 | 0.22% |
| record | description: Technical summary of the patented invention | 1 | 0.22% |
| record | comment_count: Number of comments on the video | 1 | 0.22% |
| record | video_id: Unique identifier assigned to the video | 1 | 0.22% |
| record | like_count: Number of likes the video has accumulated | 1 | 0.22% |
| record | search_result_page.website_name | 1 | 0.22% |
| record | initial_text.actions | 1 | 0.22% |
| record | initial_text.negations | 1 | 0.22% |
| record | search_result_page.related_to | 1 | 0.22% |
| record | line2 | 1 | 0.22% |
| record | booth | 1 | 0.22% |
| record | application_id | 1 | 0.22% |
| record | filer | 1 | 0.22% |
| record | technical_focus | 1 | 0.22% |
| record | schema_name | 1 | 0.22% |
| record | isbn | 1 | 0.22% |
| record | publication_volume | 1 | 0.22% |
| record | publication_issue | 1 | 0.22% |
| record | distance.reference_point | 1 | 0.22% |
| record | naming.named_after.name | 1 | 0.22% |
| record | naming.date | 1 | 0.22% |
| record | naming.named_after.profession | 1 | 0.22% |
| record | device_type: Type of medical device such as ECG or infusion pump | 1 | 0.22% |
| record | assigned_to: Name of healthcare professional responsible for device use | 1 | 0.22% |
| record | incident_id | 1 | 0.22% |
| record | priority | 1 | 0.22% |
| record | root_cause | 1 | 0.22% |
| record | resolution_time | 1 | 0.22% |
| record | service | 1 | 0.22% |
| record | resolution_team | 1 | 0.22% |
| record | author_information.education.0.institution | 1 | 0.22% |
| record | author_information.military_service.branch | 1 | 0.22% |
| record | author_information.military_service.status | 1 | 0.22% |
| record | author_information.occupation | 1 | 0.22% |
| record | author_information.location.city | 1 | 0.22% |
| record | author_information.location.state | 1 | 0.22% |
| record | author_information.family.spouse | 1 | 0.22% |
| record | author_information.family.pets.0.name | 1 | 0.22% |
| record | author_information.family.pets.0.type | 1 | 0.22% |
| record | author_information.interests | 1 | 0.22% |
| record | reviewer_address | 1 | 0.22% |
| record | assigned_agent | 1 | 0.22% |
| record | physical_characteristics.observations.0.telescope.location | 1 | 0.22% |
| record | discovery_details.observatory.location.country | 1 | 0.22% |
| record | discovery_details.observatory.location.county | 1 | 0.22% |
| record | asteroid_belt | 1 | 0.22% |
| record | orbital_characteristics.jpl_classification | 1 | 0.22% |
| record | position | 1 | 0.22% |
| record | current_ratio: Liquidity ratio measuring short-term obligations | 1 | 0.22% |
| record | claim_amount: Monetary claim amount if applicable | 1 | 0.22% |
| record | case_id: Unique case identifier in format CV-YYYY-XXXX | 1 | 0.22% |
| record | purpose | 1 | 0.22% |
| record | citations.4.description | 1 | 0.22% |
| record | gymnosperm_diversity.species_examples.0.characteristic | 1 | 0.22% |
| record | citations.2.year | 1 | 0.22% |
| record | citations.0.year | 1 | 0.22% |
| record | gymnosperm_diversity.endemism_ranking | 1 | 0.22% |
| record | gymnosperm_diversity.species_examples.2.characteristic | 1 | 0.22% |
| record | taxonomic_reference | 1 | 0.22% |
| record | citations.3.description | 1 | 0.22% |
| record | citations.1.author | 1 | 0.22% |
| record | citations.2.author | 1 | 0.22% |
| record | gymnosperm_diversity.diversity_center | 1 | 0.22% |
| record | gymnosperm_diversity.species_examples.1.characteristic | 1 | 0.22% |
| record | location.region | 1 | 0.22% |
| record | gymnosperm_diversity.species_examples.2.name | 1 | 0.22% |
| record | gymnosperm_diversity.rare_species | 1 | 0.22% |
| record | citations.1.year | 1 | 0.22% |
| record | gymnosperm_diversity.pine_species_count | 1 | 0.22% |
| record | gymnosperm_diversity.species_examples.3.characteristic | 1 | 0.22% |
| record | test_cases_completed | 1 | 0.22% |
| record | team | 1 | 0.22% |
| record | battery_life: Battery life in hours as specified | 1 | 0.22% |
| record | availability_date: Date when product becomes available | 1 | 0.22% |
| record | price: Price in dollars without currency symbol | 1 | 0.22% |
| record | color: Color of the product | 1 | 0.22% |
| record | status: Current stock availability status | 1 | 0.22% |
| record | test_type | 1 | 0.22% |
| record | registrations | 1 | 0.22% |
| record | sponsor | 1 | 0.22% |
| record | activity | 1 | 0.22% |
| record | artistic_characteristics | 1 | 0.22% |
| record | status: Current processing or acceptance status | 1 | 0.22% |
| record | author: Name of the submitting researcher or academic | 1 | 0.22% |
| record | institution: University or institution where author is affiliated | 1 | 0.22% |
| record | Consumption | 1 | 0.22% |
| record | Food | 1 | 0.22% |
| record | priority_level | 1 | 0.22% |
| record | property_address | 1 | 0.22% |
| record | risk_category | 1 | 0.22% |
| record | coverage_limit | 1 | 0.22% |
| record | system_code | 1 | 0.22% |
| record | geographic_region | 1 | 0.22% |
| record | connectivity | 1 | 0.22% |
| record | market_segment | 1 | 0.22% |
| record | submitter: Name of employee who filed the ticket | 1 | 0.22% |
| record | ticket_id: Unique identifier for each support request | 1 | 0.22% |
| record | status: Current status of the funding round | 1 | 0.22% |
| record | company_name: Name of the startup receiving funding | 1 | 0.22% |
| record | treatment: Therapeutic approach prescribed | 1 | 0.22% |
| record | provider: Name of the physician who treated the patient | 1 | 0.22% |
| record | diagnosis: Medical condition diagnosed with ICD-10 code | 1 | 0.22% |
| record | patient_id: Unique identifier for the patient | 1 | 0.22% |
| record | event-attack/battle/war/militaryconflict | 1 | 0.22% |
| record | tracking_id | 1 | 0.22% |
| record | open_rate | 1 | 0.22% |
| record | market_cap: Market capitalization in billions | 1 | 0.22% |
| record | revenue: Total revenue in billions as reported | 1 | 0.22% |
| record | growth_rate: Year-over-year growth percentage | 1 | 0.22% |
| record | client_id | 1 | 0.22% |
| record | shipment_type | 1 | 0.22% |
| record | date_posted | 1 | 0.22% |
| record | labels | 1 | 0.22% |
| record | series_record | 1 | 0.22% |
| record | patient_name: Full name of the patient receiving the medication | 1 | 0.22% |
| record | patient_address: Street address of the patient | 1 | 0.22% |
| record | zip_code: ZIP code of the patient's residence | 1 | 0.22% |
| record | prescription_id: Unique identifier for the prescription record | 1 | 0.22% |
| record | medication: Name of the prescribed medication | 1 | 0.22% |
| record | dosage: Instructions on how and when to take the medication | 1 | 0.22% |
| record | supply_quantity: Total number of tablets dispensed | 1 | 0.22% |
| record | status: Current status of the prescription | 1 | 0.22% |
| record | candidate_id: Identifier for the candidate receiving the donation | 1 | 0.22% |
| record | donor_type: Type of donor as defined in the disclosure | 1 | 0.22% |
| record | donor_name: Full name of the individual or organization making the donation | 1 | 0.22% |
| record | donation_description: Brief description of the donation's purpose or context | 1 | 0.22% |
| record | patent_number | 1 | 0.22% |
| record | inventor | 1 | 0.22% |
| record | technology_area | 1 | 0.22% |
| record | discovery.observatory.location | 1 | 0.22% |
| record | discovery.discoverers | 1 | 0.22% |
| record | orbital_characteristics.classification | 1 | 0.22% |
| record | physical_characteristics.observations.0.telescope.orbit_year | 1 | 0.22% |
| record | paymentduedate | 1 | 0.22% |
| record | vendorinvoiceemail | 1 | 0.22% |
| record | account_code | 1 | 0.22% |
| record | product | 1 | 0.22% |
| record | model | 1 | 0.22% |
| record | review_date | 1 | 0.22% |
| record | review_length | 1 | 0.22% |
| record | release_year | 1 | 0.22% |
| record | warranty_period | 1 | 0.22% |
| record | compatibility | 1 | 0.22% |
| record | power_consumption | 1 | 0.22% |
| record | charging_time | 1 | 0.22% |
| record | storage_capacity | 1 | 0.22% |
| record | use_case | 1 | 0.22% |
| record | genus.subspecies.0.name | 1 | 0.22% |
| record | basin_name | 1 | 0.22% |
| record | city: City where the transaction took place | 1 | 0.22% |
| record | name: Full name of the customer | 1 | 0.22% |
| record | transaction_type: Mode of transaction such as online or in-person | 1 | 0.22% |
| record | amount: Total cost of the fuel refill | 1 | 0.22% |
| record | station: Name of the gas station | 1 | 0.22% |
| record | description: Description of the transaction activity | 1 | 0.22% |
| record | followupdate | 1 | 0.22% |
| record | discovery.project | 1 | 0.22% |
| record | number | 1 | 0.22% |
| record | physical_characteristics.observations.0.telescope.launch_year | 1 | 0.22% |
| record | word_count: Number of words in the abstract | 1 | 0.22% |
| record | submission_id: Unique identifier assigned to each abstract submission | 1 | 0.22% |
| record | funding_code: Code indicating eligibility for funding support | 1 | 0.22% |
| record | asteroid.named_after.location.country | 1 | 0.22% |
| record | asteroid.named_after.location.province | 1 | 0.22% |
| record | discovery.observatory | 1 | 0.22% |
| record | discovery.astronomer.nationality | 1 | 0.22% |
| record | number_of_offers | 1 | 0.22% |
| record | star.designation | 1 | 0.22% |
| record | star.constellation | 1 | 0.22% |
| record | star.distance.light_years | 1 | 0.22% |
| record | star.distance.parsecs | 1 | 0.22% |
| record | star.spectral_class | 1 | 0.22% |
| record | star.effective_temperature | 1 | 0.22% |
| record | planetary_system.planets.0.discovery_year | 1 | 0.22% |
| record | planetary_system.planets.0.designation | 1 | 0.22% |
| record | planetary_system.planets.0.discovery_team | 1 | 0.22% |
| record | death.date_julian | 1 | 0.22% |
| record | birth.date_julian | 1 | 0.22% |
| record | death.location | 1 | 0.22% |
| record | name.given_name | 1 | 0.22% |
| record | teacher | 1 | 0.22% |
| record | works.type | 1 | 0.22% |
| record | occupation | 1 | 0.22% |
| record | birth.location | 1 | 0.22% |
| record | name.family_name | 1 | 0.22% |
| record | father.occupation | 1 | 0.22% |
| record | neighborhood | 1 | 0.22% |
| record | mls_number | 1 | 0.22% |
| record | agent | 1 | 0.22% |
| record | contribution_type | 1 | 0.22% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.0.t_score | 1 | 0.22% |
| record | statistical_results.hazard_ratios.0.comparison | 1 | 0.22% |
| record | statistical_results.predictive_model.variables_included | 1 | 0.22% |
| record | statistical_results.hazard_ratios.0.value | 1 | 0.22% |
| record | study_metadata.measurement_technology.anatomical_site | 1 | 0.22% |
| record | population.age_range.max | 1 | 0.22% |
| record | statistical_results.predictive_model.value | 1 | 0.22% |
| record | conclusions | 1 | 0.22% |
| record | population.age_range.unit | 1 | 0.22% |
| record | outcomes.fracture_events.0.count | 1 | 0.22% |
| record | outcomes.data_source | 1 | 0.22% |
| record | population.gender | 1 | 0.22% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.1.t_score | 1 | 0.22% |
| record | follow_up.average_duration.years | 1 | 0.22% |
| record | statistical_results.diagnostic_metrics.sensitivity_at_threshold.t_score_threshold | 1 | 0.22% |
| record | follow_up.average_duration.months | 1 | 0.22% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.1.rate_percentage | 1 | 0.22% |
| record | genus.name_zh | 1 | 0.22% |
| record | amount: Total funding amount in USD with currency symbol | 1 | 0.22% |
| record | location: City and state where the company is headquartered | 1 | 0.22% |
| record | investor: Name of the venture capital firm that invested | 1 | 0.22% |
| record | page_count: Number of pages in the paper | 1 | 0.22% |
| record | abstract_length: Length of the abstract in words | 1 | 0.22% |
| record | review_date: Date when the paper was reviewed | 1 | 0.22% |
| record | conference_name: Name of the conference where submitted | 1 | 0.22% |
| record | funding_source: Source of research funding | 1 | 0.22% |
| record | word_count: Total number of words in the paper | 1 | 0.22% |
| record | submission_date: Date when the paper was submitted | 1 | 0.22% |
| record | affiliation: Institutional affiliation of the author | 1 | 0.22% |
| record | fieldofstudy: Academic domain of the research | 1 | 0.22% |
| record | assigned_adjuster | 1 | 0.22% |
| record | company_name: Name of the company in the record | 1 | 0.22% |
| record | partner_location: City and state of the partner company | 1 | 0.22% |
| record | city: City where patient resides | 1 | 0.22% |
| record | symptoms: List of reported symptoms like fever or headache | 1 | 0.22% |
| record | onset_date: Date when symptoms first appeared | 1 | 0.22% |
| record | address: Street address of the patient | 1 | 0.22% |
| record | block | 1 | 0.22% |
| record | flight_number | 1 | 0.22% |
| record | booking_reference | 1 | 0.22% |
| record | drexel_info | 1 | 0.22% |
| record | browse_topics | 1 | 0.22% |
| record | footer_links | 1 | 0.22% |
| record | faq_references | 1 | 0.22% |
| record | lastdosedate | 1 | 0.22% |
| record | vaccination_id | 1 | 0.22% |
| record | pro_accomplishments | 1 | 0.22% |
| record | vin | 1 | 0.22% |
| record | apparent_magnitude.band | 1 | 0.22% |
| record | star_name | 1 | 0.22% |
| record | company_name: Name of the startup founded by the individual | 1 | 0.22% |
| record | focus_area: Primary domain or mission of the company | 1 | 0.22% |
| record | location: City and state where the company was founded | 1 | 0.22% |
| record | numerical_values | 1 | 0.22% |
| record | adverb_modifier | 1 | 0.22% |
| record | adjectival_modifier | 1 | 0.22% |
| record | technical_terms | 1 | 0.22% |
| record | events | 1 | 0.22% |
| record | development_periods | 1 | 0.22% |
| record | see_also | 1 | 0.22% |
| record | stub_article.call_to_action | 1 | 0.22% |
| record | reference: Internal reference code linked to the ticket | 1 | 0.22% |
| record | user: Name of the employee reporting the issue | 1 | 0.22% |
| record | description: Detailed account of the problem and symptoms | 1 | 0.22% |
| record | status: Current state of the ticket | 1 | 0.22% |
| record | priority: Severity level of the issue | 1 | 0.22% |
| record | affected_system: System or application where the issue occurs | 1 | 0.22% |
| record | office_location | 1 | 0.22% |
| record | state_or_province | 1 | 0.22% |
| record | established_year | 1 | 0.22% |
| record | specialization | 1 | 0.22% |
| record | funding_amount: Initial capital received in USD | 1 | 0.22% |
| record | plaintiff: Name of the individual or entity initiating the lawsuit | 1 | 0.22% |
| record | defendant: Name of the individual or entity being sued | 1 | 0.22% |
| record | status: Current status of the case | 1 | 0.22% |
| record | eps | 1 | 0.22% |
| record | subscription_id | 1 | 0.22% |
| record | subscription_type | 1 | 0.22% |
| record | billing_amount | 1 | 0.22% |
| record | billing_frequency | 1 | 0.22% |
| record | email_address | 1 | 0.22% |
| record | delivery_address | 1 | 0.22% |
| record | expectedOutcome | 1 | 0.22% |
| record | surveyFocus | 1 | 0.22% |
| record | icd10code | 1 | 0.22% |
| record | policy_reference | 1 | 0.22% |
| record | themes | 1 | 0.22% |
| record | report_id: Unique identifier assigned to the phishing report | 1 | 0.22% |
| record | status: Current status of the report | 1 | 0.22% |
| record | sender_domain: Domain of the spoofed email address | 1 | 0.22% |
| record | medication | 1 | 0.22% |
| record | pharmacy_name | 1 | 0.22% |
| record | pharmacy_address | 1 | 0.22% |
| record | salary | 1 | 0.22% |
| record | phone | 1 | 0.22% |
| record | hire_date | 1 | 0.22% |
| record | currency | 1 | 0.22% |
| record | contact_person | 1 | 0.22% |
| record | last_name | 1 | 0.22% |
| record | hire_method | 1 | 0.22% |
| record | first_name | 1 | 0.22% |
| record | additional_investors | 1 | 0.22% |
| record | description.flowers.color | 1 | 0.22% |
| record | description.seeds.color | 1 | 0.22% |
| record | description.fruit.color | 1 | 0.22% |
| record | description.inflorescence.height_m.max | 1 | 0.22% |
| record | systematics_and_distribution.first_description.author | 1 | 0.22% |
| record | description.flowers.perianth_tube_length_mm.max | 1 | 0.22% |
| record | description.inflorescence.height_m.min | 1 | 0.22% |
| record | description.flowers.length_mm.max | 1 | 0.22% |
| record | systematics_and_distribution.habitat | 1 | 0.22% |
| record | common_names | 1 | 0.22% |
| record | description.leaves.length_cm.max | 1 | 0.22% |
| record | systematics_and_distribution.group | 1 | 0.22% |
| record | systematics_and_distribution.location.elevation_m.min | 1 | 0.22% |
| record | description.leaves.shape | 1 | 0.22% |
| record | claim_amount | 1 | 0.22% |
| record | posted_on | 1 | 0.22% |
| record | submission_method | 1 | 0.22% |
| record | place.type | 1 | 0.22% |
| record | beliefs | 1 | 0.22% |
| record | religion | 1 | 0.22% |
| record | scriptures | 1 | 0.22% |
| record | policyholder_id: Unique identifier for the policyholder | 1 | 0.22% |
| record | claim_date: Date the claim was filed | 1 | 0.22% |
| record | policyholder_name: Full name of the claimant | 1 | 0.22% |
| record | rate | 1 | 0.22% |
| record | snapshot_id | 1 | 0.22% |
| record | det | 1 | 0.22% |
| record | appos | 1 | 0.22% |
| record | aux | 1 | 0.22% |
| record | ccomp | 1 | 0.22% |
| record | amod | 1 | 0.22% |
| record | acl | 1 | 0.22% |
| record | pobj | 1 | 0.22% |
| record | publication_code | 1 | 0.22% |
| record | geographical_scope | 1 | 0.22% |
| record | result_type | 1 | 0.22% |
| record | assessment_name | 1 | 0.22% |
| record | entity | 1 | 0.22% |
| record | tracking_code | 1 | 0.22% |
| record | item: Description of the returned product | 1 | 0.22% |
| record | status: Current status of the return approval | 1 | 0.22% |
| record | amount: Monetary value of the return in dollars | 1 | 0.22% |
| record | priority: Urgency level of the ticket | 1 | 0.22% |
| record | resolution_time: Estimated or actual time to resolve the issue | 1 | 0.22% |
| record | description: Detailed account of the issue reported | 1 | 0.22% |
| record | department: Department the submitter belongs to | 1 | 0.22% |
| record | submitter: Name of the employee who submitted the ticket | 1 | 0.22% |
| record | scorer | 1 | 0.22% |
| record | lead_reporter: Reporter providing coverage of the incident | 1 | 0.22% |
| record | title: Event title describing the nature and context of the incident | 1 | 0.22% |
| record | location: Geographic location where the incident took place | 1 | 0.22% |
| record | impact: Description of human, financial, or systemic consequences | 1 | 0.22% |
| record | date: Timestamp when the incident occurred | 1 | 0.22% |
| record | municipality.name | 1 | 0.22% |
| record | ship_name | 1 | 0.22% |
| record | history.creation.shipyard.name | 1 | 0.22% |
| record | history.creation.commissioning_date | 1 | 0.22% |
| record | history.creation.laid_down_date | 1 | 0.22% |
| record | history.creation.shipyard.location | 1 | 0.22% |
| record | usage_frequency | 1 | 0.22% |
| record | exploited_ports | 1 | 0.22% |
| record | software_versions | 1 | 0.22% |
| record | isps | 1 | 0.22% |
| record | lead_investor | 1 | 0.22% |
| record | user_name | 1 | 0.22% |
| record | name: Name of the person associated with the transaction | 1 | 0.22% |
| record | description: Brief explanation of the transaction purpose | 1 | 0.22% |
| record | status: Current approval status of the transaction | 1 | 0.22% |
| record | disputingpartya | 1 | 0.22% |
| record | disputingpartyb | 1 | 0.22% |
| record | force_type | 1 | 0.22% |
| record | material_efficiency | 1 | 0.22% |
| record | type | 1 | 0.22% |
| record | amount: Claimed financial amount in dollars | 1 | 0.22% |
| record | sender: Name of person who submitted the report | 1 | 0.22% |
| record | report_id: Unique identifier for the phishing incident | 1 | 0.22% |
| record | subject: Subject line of the malicious email | 1 | 0.22% |
| record | malicious_content: Description of the phishing element like link or attachment | 1 | 0.22% |
| record | user | 1 | 0.22% |
| record | record_metadata | natural/status ×19, natural/name ×10, natural/location ×9, natural/title ×7 |  |
| research_paper | title | 5 | 100.00% |
| research_paper | authors | 5 | 100.00% |
| research_paper | venue | 5 | 100.00% |
| research_paper | year | 4 | 80.00% |
| research_paper | topic | 2 | 40.00% |
| research_paper | record_metadata | natural/title ×5 |  |
| transaction | buyer | 45 | 95.74% |
| transaction | date | 43 | 91.49% |
| transaction | amount | 42 | 89.36% |
| transaction | item | 42 | 89.36% |
| transaction | seller | 42 | 89.36% |
| transaction | record_metadata | natural/item ×42, natural/amount ×3 |  |
| vehicle_listing | model | 1 | 100.00% |
| vehicle_listing | year | 1 | 100.00% |
| vehicle_listing | price | 1 | 100.00% |
| vehicle_listing | record_metadata | natural/model ×1 |  |

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | mixed (replay) |
| relations | mixed (replay) |
| events | mixed (replay) |
| classifications | mixed (replay) |
| structures | mixed (replay) |
