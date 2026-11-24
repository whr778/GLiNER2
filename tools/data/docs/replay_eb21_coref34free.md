# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replay_eb21_coref34free | train | 15,133 | 15,067 | 103,952 | 7,327 | 12,747 | 764 | 9,437 | 13,079 |
| replay_eb21_coref34free | val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| replay_eb21_coref34free | test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## replay_eb21_coref34free

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 15,133 | 15,067 | 324 | 1199 | 22512 | 103,952 | 7,327 | 12,747 | 764 | 9,437 | 13,079 |
| val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 7,714 | 0 | 0 | 7,714 | 50.97% |
| zho | 5,581 | 0 | 0 | 5,581 | 36.88% |
| tur | 1,527 | 0 | 0 | 1,527 | 10.09% |
| und | 217 | 0 | 0 | 217 | 1.43% |
| fra | 18 | 0 | 0 | 18 | 0.12% |
| por | 10 | 0 | 0 | 10 | 0.07% |
| rus | 10 | 0 | 0 | 10 | 0.07% |
| ukr | 9 | 0 | 0 | 9 | 0.06% |
| ita | 8 | 0 | 0 | 8 | 0.05% |
| spa | 6 | 0 | 0 | 6 | 0.04% |
| hin | 6 | 0 | 0 | 6 | 0.04% |
| deu | 6 | 0 | 0 | 6 | 0.04% |
| pol | 4 | 0 | 0 | 4 | 0.03% |
| kor | 4 | 0 | 0 | 4 | 0.03% |
| ces | 3 | 0 | 0 | 3 | 0.02% |
| nld | 3 | 0 | 0 | 3 | 0.02% |
| ara | 2 | 0 | 0 | 2 | 0.01% |
| ilo | 1 | 0 | 0 | 1 | 0.01% |
| pms | 1 | 0 | 0 | 1 | 0.01% |
| vie | 1 | 0 | 0 | 1 | 0.01% |
| jpn | 1 | 0 | 0 | 1 | 0.01% |
| wuu | 1 | 0 | 0 | 1 | 0.01% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 0 |
| test_unique_inputs | 0 |
| train_records | 15,133 |
| train_unique_inputs | 15,067 |
| val_records | 0 |
| val_unique_inputs | 0 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | n/a | 103,952 / 0 |  |
| entities (per mention) | train / test | n/a | 103,952 / 0 |  |
| relations (per instance) | train / val | n/a | 7,327 / 0 |  |
| relations (per instance) | train / test | n/a | 7,327 / 0 |  |
| event types (per trigger mention) | train / val | n/a | 10,728 / 0 |  |
| event types (per trigger mention) | train / test | n/a | 10,728 / 0 |  |
| event roles (per argument) | train / val | n/a | 13,471 / 0 |  |
| event roles (per argument) | train / test | n/a | 13,471 / 0 |  |
| classifications (true_label) | train / val | n/a | 12,747 / 0 |  |
| classifications (true_label) | train / test | n/a | 12,747 / 0 |  |


### Events

**train.** 10,382 instances in 2,831 records, 0 triggerless. Trigger gold 9,437. Argument gold 13,079. Arguments per instance, mean 1.30. Trigger length in whitespace tokens, mean 1.36, single-token share 71.9%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Contact.Communicate | 710 | 6.62% |
| Experiment | 470 | 4.38% |
| Support | 394 | 3.67% |
| Catastrophe | 252 | 2.35% |
| Attack | 243 | 2.27% |
| Sport.Compete | 213 | 1.99% |
| Causation | 199 | 1.85% |
| Hostile_encounter | 166 | 1.55% |
| Process_start | 165 | 1.54% |
| Life.Die | 147 | 1.37% |
| Deploy | 137 | 1.28% |
| Motion | 135 | 1.26% |
| Competition | 119 | 1.11% |
| Personnel.StartPosition | 115 | 1.07% |
| Conflict.Attack | 112 | 1.04% |
| Statement | 105 | 0.98% |
| match_result | 104 | 0.97% |
| Exhibit | 101 | 0.94% |
| Business.StartOrg | 100 | 0.93% |
| Killing | 97 | 0.90% |
| Conquering | 97 | 0.90% |
| Cyber.DiscoverVulnerability | 91 | 0.85% |
| Personnel.EndPosition | 88 | 0.82% |
| Destroying | 85 | 0.79% |
| Life.Injure | 85 | 0.79% |
| Arriving | 84 | 0.78% |
| Self_motion | 83 | 0.77% |
| product_publish | 82 | 0.76% |
| Contact.Meet | 81 | 0.76% |
| Social_event | 80 | 0.75% |
| Cyber.Phishing | 79 | 0.74% |
| Coming_to_be | 78 | 0.73% |
| Process_end | 77 | 0.72% |
| Damaging | 76 | 0.71% |
| Bodily_harm | 75 | 0.70% |
| Cause_change_of_position_on_a_scale | 75 | 0.70% |
| Justice.Arrest | 75 | 0.70% |
| Cyber.Ransom | 74 | 0.69% |
| Death | 71 | 0.66% |
| Cause_change_of_strength | 70 | 0.65% |
| Cyber.Databreach | 70 | 0.65% |
| Hold | 67 | 0.62% |
| Justice.ChargeIndict | 66 | 0.62% |
| Earnings_and_losses | 65 | 0.61% |
| Presence | 65 | 0.61% |
| Creating | 64 | 0.60% |
| Military_operation | 63 | 0.59% |
| Control | 62 | 0.58% |
| Award.ReceiveAward | 61 | 0.57% |
| Know | 59 | 0.55% |
| Transaction.TransferOwnership | 59 | 0.55% |
| Cyber.PatchVulnerability | 57 | 0.53% |
| Motion_directional | 54 | 0.50% |
| Becoming | 53 | 0.49% |
| Traveling | 53 | 0.49% |
| Preventing_or_letting | 53 | 0.49% |
| Getting | 52 | 0.48% |
| Justice.Sentence | 51 | 0.48% |
| Placing | 51 | 0.48% |
| Action | 49 | 0.46% |
| Justice.TrialHearing | 48 | 0.45% |
| Influence | 48 | 0.45% |
| Business.LaunchProduct | 48 | 0.45% |
| Arranging | 47 | 0.44% |
| Accident | 46 | 0.43% |
| Change | 46 | 0.43% |
| judicial_action_arrest | 44 | 0.41% |
| Escaping | 43 | 0.40% |
| Name_conferral | 43 | 0.40% |
| Request | 43 | 0.40% |
| Cause_to_be_included | 42 | 0.39% |
| Change_of_leadership | 42 | 0.39% |
| Disaster.Accident | 41 | 0.38% |
| Expressing_publicly | 40 | 0.37% |
| Removing | 40 | 0.37% |
| Transaction.TransferMoney | 39 | 0.36% |
| Judgment_communication | 38 | 0.35% |
| death | 37 | 0.34% |
| Manoeuvre | 36 | 0.34% |
| Manufacturing | 36 | 0.34% |
| Using | 36 | 0.34% |
| Choosing | 36 | 0.34% |
| resignation | 36 | 0.34% |
| Research.Publish | 36 | 0.34% |
| Defending | 36 | 0.34% |
| Cause_to_make_progress | 35 | 0.33% |
| Giving | 34 | 0.32% |
| Participation | 32 | 0.30% |
| Bringing | 32 | 0.30% |
| Supply | 32 | 0.30% |
| Recording | 32 | 0.30% |
| Use_firearm | 31 | 0.29% |
| Departing | 30 | 0.28% |
| Conflict.Demonstrate | 29 | 0.27% |
| Reporting | 29 | 0.27% |
| Life.BeBorn | 29 | 0.27% |
| Conflict.Attack.DetonateExplode | 29 | 0.27% |
| Tech.Release | 29 | 0.27% |
| Movement.Transport | 28 | 0.26% |
| Supporting | 28 | 0.26% |
| Cause_to_amalgamate | 27 | 0.25% |
| life.die.deathcausedbyviolentevents | 27 | 0.25% |
| Aiming | 27 | 0.25% |
| Movement.Arrive | 27 | 0.25% |
| championship_win | 26 | 0.24% |
| Scrutiny | 25 | 0.23% |
| Agree_or_refuse_to_act | 25 | 0.23% |
| Perception_active | 24 | 0.22% |
| Assistance | 24 | 0.22% |
| Come_together | 22 | 0.21% |
| Check | 22 | 0.21% |
| Injure | 22 | 0.21% |
| Medical.Diagnose | 22 | 0.21% |
| Building | 22 | 0.21% |
| marriage | 22 | 0.21% |
| advancement | 21 | 0.20% |
| divorce | 21 | 0.20% |
| joined | 21 | 0.20% |
| Communication | 21 | 0.20% |
| Collaboration | 20 | 0.19% |
| Dispersal | 20 | 0.19% |
| Justice.Sue | 20 | 0.19% |
| Life.Die.Unspecified | 20 | 0.19% |
| product_recall | 19 | 0.18% |
| Conflict | 19 | 0.18% |
| Deciding | 19 | 0.18% |
| Life.Marry | 19 | 0.18% |
| Publishing | 19 | 0.18% |
| Research.Discover | 19 | 0.18% |
| judicial_action_fine | 18 | 0.17% |
| Connect | 18 | 0.17% |
| judicial_action_summon | 18 | 0.17% |
| Violence | 18 | 0.17% |
| Receiving | 18 | 0.17% |
| Sending | 18 | 0.17% |
| Recovering | 17 | 0.16% |
| Disaster.NaturalDisaster | 17 | 0.16% |
| Government.Vote | 17 | 0.16% |
| Terrorism | 17 | 0.16% |
| Hindering | 17 | 0.16% |
| Research.Fund | 17 | 0.16% |
| product_release | 16 | 0.15% |
| Hiding_objects | 16 | 0.15% |
| Arrest | 16 | 0.15% |
| opening | 15 | 0.14% |
| Response | 15 | 0.14% |
| GiveUp | 15 | 0.14% |
| Expansion | 15 | 0.14% |
| product_delist | 15 | 0.14% |
| price_limit_up | 15 | 0.14% |
| Cost | 14 | 0.13% |
| Surrendering | 14 | 0.13% |
| Temporary_stay | 14 | 0.13% |
| Legal_rulings | 13 | 0.12% |
| Warning | 13 | 0.12% |
| Becoming_a_member | 13 | 0.12% |
| Confronting_problem | 13 | 0.12% |
| Achieve | 13 | 0.12% |
| Conflict.Attack.Unspecified | 13 | 0.12% |
| Contact.Contact.Unspecified | 13 | 0.12% |
| Transaction.Donate | 12 | 0.11% |
| birthday_celebration | 12 | 0.11% |
| contact.discussion.meet | 12 | 0.11% |
| life.injure.injurycausedbyviolentevents | 11 | 0.10% |
| Justice.Appeal | 11 | 0.10% |
| movement.transportperson.preventexit | 11 | 0.10% |
| birth | 11 | 0.10% |
| Employment | 11 | 0.10% |
| layoff | 11 | 0.10% |
| Life.Evacuate | 11 | 0.10% |
| Movement.Migrate | 11 | 0.10% |
| Fire | 11 | 0.10% |
| Disaster.Rescue | 11 | 0.10% |
| Quarreling | 10 | 0.09% |
| departure | 10 | 0.09% |
| Government.Sanction | 10 | 0.09% |
| Life.Injure.Unspecified | 10 | 0.09% |
| Openness | 10 | 0.09% |
| Government.Protest | 10 | 0.09% |
| Emptying | 10 | 0.09% |
| conflict.attack.n/a | 10 | 0.09% |
| apologize_to | 10 | 0.09% |
| Limiting | 10 | 0.09% |
| Medical.Treat | 10 | 0.09% |
| Being_in_operation | 10 | 0.09% |
| Contact.Contact.Broadcast | 10 | 0.09% |
| Criminal_investigation | 9 | 0.08% |
| life.die.n/a | 9 | 0.08% |
| financing | 9 | 0.08% |
| judicial_action_prosecution | 9 | 0.08% |
| Besieging | 9 | 0.08% |
| Commitment | 9 | 0.08% |
| Committing_crime | 9 | 0.08% |
| Change_event_time | 9 | 0.08% |
| justice.arrestjaildetain.arrestjaildetain | 9 | 0.08% |
| sale_acquisition | 9 | 0.08% |
| Testing | 9 | 0.08% |
| Justice.Fine | 8 | 0.07% |
| judicial_action_imprisonment | 8 | 0.07% |
| life.injure.n/a | 8 | 0.07% |
| product_award | 8 | 0.07% |
| Business.MergeOrg | 8 | 0.07% |
| Justice.ArrestJailDetain.Unspecified | 8 | 0.07% |
| Commerce_sell | 8 | 0.07% |
| Traffic Accident | 8 | 0.07% |
| Disaster.Outbreak | 8 | 0.07% |
| Preserving | 8 | 0.07% |
| transaction.transferownership.purchase | 8 | 0.07% |
| judicial_action_hearing | 8 | 0.07% |
| Coming_to_believe | 8 | 0.07% |
| Create_artwork | 8 | 0.07% |
| Change_sentiment | 8 | 0.07% |
| movement.transportartifact.n/a | 8 | 0.07% |
| Surrounding | 8 | 0.07% |
| Collapse | 7 | 0.07% |
| Personnel.Elect | 7 | 0.07% |
| Revenge | 7 | 0.07% |
| Contact.Negotiate | 7 | 0.07% |
| Explosion | 7 | 0.07% |
| justice.judicialconsequences.convict | 7 | 0.07% |
| competition_ban | 7 | 0.07% |
| competition_withdrawal | 7 | 0.07% |
| conflict.attack.selfdirectedbattle | 7 | 0.07% |
| movement.transportartifact.preventexit | 7 | 0.07% |
| breakup | 7 | 0.07% |
| Robbery | 7 | 0.07% |
| justice.initiatejudicialprocess.chargeindict | 7 | 0.07% |
| Business.EndOrg | 6 | 0.06% |
| transaction.transferownership.n/a | 6 | 0.06% |
| price_increase | 6 | 0.06% |
| Prison | 6 | 0.06% |
| Convincing | 6 | 0.06% |
| contact.commitmentpromiseexpressintent.meet | 6 | 0.06% |
| Justice.ChargeIndict.Unspecified | 6 | 0.06% |
| Movement.Transportation.Unspecified | 6 | 0.06% |
| Life.Divorce | 6 | 0.06% |
| Government.EnactLaw | 6 | 0.06% |
| transaction.transaction.transfercontrol | 6 | 0.06% |
| Sign_agreement | 6 | 0.06% |
| judicial_action_filing | 6 | 0.06% |
| Reveal_secret | 6 | 0.06% |
| Body_movement | 6 | 0.06% |
| conflict.attack.stabbing | 6 | 0.06% |
| Earthquake | 6 | 0.06% |
| artifactexistence.damagedestroy.destroy | 6 | 0.06% |
| GetReady | 6 | 0.06% |
| government.agreements.acceptagreementcontractceasefire | 6 | 0.06% |
| Research | 6 | 0.06% |
| march | 5 | 0.05% |
| Tech.Breach | 5 | 0.05% |
| price_limit_down | 5 | 0.05% |
| Movement.Deport | 5 | 0.05% |
| ipo | 5 | 0.05% |
| contract_termination | 5 | 0.05% |
| Resolve_problem | 5 | 0.05% |
| Rewards_and_punishments | 5 | 0.05% |
| transaction.transferownership.borrowlend | 5 | 0.05% |
| movement.transportperson.n/a | 5 | 0.05% |
| Containing | 5 | 0.05% |
| transaction.transferownership.giftgrantprovideaid | 5 | 0.05% |
| contact.commitmentpromiseexpressintent.correspondence | 5 | 0.05% |
| Business.Strike | 5 | 0.05% |
| government.agreements.n/a | 5 | 0.05% |
| personnel.elect.winelection | 5 | 0.05% |
| Rescuing | 5 | 0.05% |
| Filling | 5 | 0.05% |
| Forming_relationships | 5 | 0.05% |
| Education_teaching | 5 | 0.05% |
| proposal | 5 | 0.05% |
| contact.prevarication.broadcast | 5 | 0.05% |
| Having_or_lacking_access | 5 | 0.05% |
| Exchange | 5 | 0.05% |
| dismissal | 5 | 0.05% |
| Adducing | 5 | 0.05% |
| artifactexistence.damagedestroy.damage | 5 | 0.05% |
| Business.DeclareBankruptcy | 5 | 0.05% |
| judicial_action_report | 5 | 0.05% |
| wedding | 5 | 0.05% |
| contact.requestadvise.broadcast | 4 | 0.04% |
| transaction.transfermoney.purchase | 4 | 0.04% |
| justice.judicialconsequences.execute | 4 | 0.04% |
| Submitting_documents | 4 | 0.04% |
| engagement | 4 | 0.04% |
| Transaction.ExchangeBuySell.Unspecified | 4 | 0.04% |
| Award | 4 | 0.04% |
| Writing | 4 | 0.04% |
| Personnel.Nominate | 4 | 0.04% |
| life.injure.illnessdegradationhungerthirst | 4 | 0.04% |
| contact.requestadvise.correspondence | 4 | 0.04% |
| movement.transportartifact.grantentry | 4 | 0.04% |
| government.formation.n/a | 4 | 0.04% |
| Commerce_pay | 4 | 0.04% |
| Patrolling | 4 | 0.04% |
| Telling | 4 | 0.04% |
| contact.discussion.correspondence | 4 | 0.04% |
| strike | 4 | 0.04% |
| Protest | 4 | 0.04% |
| thank | 4 | 0.04% |
| conflict.attack.bombing | 4 | 0.04% |
| meet_with | 4 | 0.04% |
| disaster.accidentcrash.accidentcrash | 4 | 0.04% |
| movement.transportartifact.sendsupplyexport | 4 | 0.04% |
| life.injure.illnessdegradationphysical | 4 | 0.04% |
| artifactexistence.damagedestroy.n/a | 4 | 0.04% |
| justice.initiatejudicialprocess.trialhearing | 4 | 0.04% |
| contact.discussion.n/a | 4 | 0.04% |
| transaction.transfermoney.embargosanction | 4 | 0.04% |
| Ingestion | 4 | 0.04% |
| manufacture.artifact.n/a | 4 | 0.04% |
| visit_on_set | 4 | 0.04% |
| Flood | 3 | 0.03% |
| transaction.transfermoney.n/a | 3 | 0.03% |
| Releasing | 3 | 0.03% |
| GenericCrime.GenericCrime.GenericCrime | 3 | 0.03% |
| Justice.InvestigateCrime.Unspecified | 3 | 0.03% |
| transaction.transfermoney.payforservice | 3 | 0.03% |
| justice.judicialconsequences.n/a | 3 | 0.03% |
| contact.commitmentpromiseexpressintent.n/a | 3 | 0.03% |
| justice.investigate.investigatecrime | 3 | 0.03% |
| suspension | 3 | 0.03% |
| Bearing_arms | 3 | 0.03% |
| Reforming_a_system | 3 | 0.03% |
| contact.publicstatementinperson.n/a | 3 | 0.03% |
| transaction.transaction.embargosanction | 3 | 0.03% |
| conflict.attack.stealrobhijack | 3 | 0.03% |
| conflict.attack.invade | 3 | 0.03% |
| Justice.Acquit | 3 | 0.03% |
| Practice | 3 | 0.03% |
| personnel.startposition.n/a | 3 | 0.03% |
| government.formation.startgpe | 3 | 0.03% |
| ArtifactExistence.DamageDestroyDisableDismantle.Damage | 3 | 0.03% |
| government.agreements.violateagreement | 3 | 0.03% |
| contact.prevarication.correspondence | 3 | 0.03% |
| Rite | 3 | 0.03% |
| movement.transportartifact.bringcarryunload | 3 | 0.03% |
| closing | 3 | 0.03% |
| transaction.transferownership.embargosanction | 3 | 0.03% |
| praise | 3 | 0.03% |
| Labeling | 3 | 0.03% |
| movement.transportperson.disperseseparate | 3 | 0.03% |
| contact.requestadvise.n/a | 3 | 0.03% |
| Legality | 3 | 0.03% |
| movement.transportperson.prevententry | 2 | 0.02% |
| movement.transportartifact.prevententry | 2 | 0.02% |
| Institutionalization | 2 | 0.02% |
| Plane Crash | 2 | 0.02% |
| contact.threatencoerce.meet | 2 | 0.02% |
| Carry_goods | 2 | 0.02% |
| conflict.demonstrate.marchprotestpoliticalgathering | 2 | 0.02% |
| Business.Layoff | 2 | 0.02% |
| ArtifactExistence.DamageDestroyDisableDismantle.DisableDefuse | 2 | 0.02% |
| inspection.sensoryobserve.physicalinvestigateinspect | 2 | 0.02% |
| Scouring | 2 | 0.02% |
| contact.requestadvise.meet | 2 | 0.02% |
| contact.commandorder.correspondence | 2 | 0.02% |
| Wearing | 2 | 0.02% |
| contact.publicstatementinperson.broadcast | 2 | 0.02% |
| movement.transportperson.evacuationrescue | 2 | 0.02% |
| contact.prevarication.n/a | 2 | 0.02% |
| transaction.transfermoney.borrowlend | 2 | 0.02% |
| Expend_resource | 2 | 0.02% |
| contact.mediastatement.n/a | 2 | 0.02% |
| conflict.yield.n/a | 2 | 0.02% |
| dissolution | 2 | 0.02% |
| inspection.sensoryobserve.inspectpeopleorganization | 2 | 0.02% |
| contact.collaborate.correspondence | 2 | 0.02% |
| movement.transportartifact.hide | 2 | 0.02% |
| contact.commandorder.meet | 2 | 0.02% |
| contact.negotiate.meet | 2 | 0.02% |
| personnel.endposition.quitretire | 2 | 0.02% |
| infidelity | 2 | 0.02% |
| personnel.endposition.firinglayoff | 2 | 0.02% |
| movement.transportartifact.nonviolentthrowlaunch | 2 | 0.02% |
| contact.collaborate.n/a | 2 | 0.02% |
| retirement | 2 | 0.02% |
| government.formation.mergegpe | 2 | 0.02% |
| manufacture.artifact.createmanufacture | 2 | 0.02% |
| movement.transportartifact.disperseseparate | 2 | 0.02% |
| lost_contact | 2 | 0.02% |
| Life.Infect.Unspecified | 2 | 0.02% |
| Contact.Contact.Meet | 2 | 0.02% |
| movement.transportartifact.fall | 2 | 0.02% |
| price_decrease | 2 | 0.02% |
| contact.commitmentpromiseexpressintent.broadcast | 2 | 0.02% |
| Cognitive.IdentifyCategorize.Unspecified | 2 | 0.02% |
| Commerce_buy | 2 | 0.02% |
| transaction.transfermoney.giftgrantprovideaid | 2 | 0.02% |
| transaction.transaction.giftgrantprovideaid | 2 | 0.02% |
| disaster.fireexplosion.fireexplosion | 2 | 0.02% |
| movement.transportperson.grantentryasylum | 2 | 0.02% |
| Cognitive.Inspection.SensoryObserve | 2 | 0.02% |
| personnel.endposition.n/a | 2 | 0.02% |
| contact.threatencoerce.n/a | 2 | 0.02% |
| pregnancy | 2 | 0.02% |
| Renting | 2 | 0.02% |
| ArtifactExistence.ManufactureAssemble.Unspecified | 2 | 0.02% |
| ArtifactExistence.DamageDestroyDisableDismantle.Destroy | 2 | 0.02% |
| Imposing_obligation | 2 | 0.02% |
| rate_increase | 2 | 0.02% |
| conflict.yield.surrender | 1 | 0.01% |
| Justice.Sentence.Unspecified | 1 | 0.01% |
| movement.transportartifact.smuggleextract | 1 | 0.01% |
| conflict.attack.setfire | 1 | 0.01% |
| Change_tool | 1 | 0.01% |
| government.agreements.rejectnullifyagreementcontractceasefire | 1 | 0.01% |
| movement.transportperson.smuggleextract | 1 | 0.01% |
| movement.transportperson.hide | 1 | 0.01% |
| government.vote.violationspreventvote | 1 | 0.01% |
| movement.transportperson.selfmotion | 1 | 0.01% |
| inspection.sensoryobserve.monitorelection | 1 | 0.01% |
| inspection.sensoryobserve.n/a | 1 | 0.01% |
| Conflict.Ceasefire | 1 | 0.01% |
| contact.funeralvigil.meet | 1 | 0.01% |
| contact.mediastatement.broadcast | 1 | 0.01% |
| justice.judicialconsequences.extradite | 1 | 0.01% |
| contact.prevarication.meet | 1 | 0.01% |
| Transaction.Lend | 1 | 0.01% |
| Incident | 1 | 0.01% |
| life.die.nonviolentdeath | 1 | 0.01% |
| contact.commandorder.n/a | 1 | 0.01% |
| Justice.TrialHearing.Unspecified | 1 | 0.01% |
| Personnel.StartPosition.Unspecified | 1 | 0.01% |
| movement.transportperson.fall | 1 | 0.01% |
| contact.collaborate.meet | 1 | 0.01% |
| contact.funeralvigil.n/a | 1 | 0.01% |
| manufacture.artifact.build | 1 | 0.01% |
| government.legislate.legislate | 1 | 0.01% |
| Contact.Contact.Correspondence | 1 | 0.01% |
| transaction.transaction.n/a | 1 | 0.01% |
| justice.initiatejudicialprocess.n/a | 1 | 0.01% |
| Lighting | 1 | 0.01% |
| conflict.attack.strangling | 1 | 0.01% |
| personnel.startposition.hiring | 1 | 0.01% |
| conflict.demonstrate.n/a | 1 | 0.01% |
| rate_decrease | 1 | 0.01% |
| Justifying | 1 | 0.01% |
| Cognitive.Research.Unspecified | 1 | 0.01% |
| contact.commandorder.broadcast | 1 | 0.01% |

**val.** No event instances.

**test.** No event instances.

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 13,471 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.77 | 31.31% | 1.94 | 32.80% | 0.06% | 0 | 13,471 | 0 | 10,728 |
| val | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |
| test | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 10,382 | 2.10% | 8,442 | 10.90% | 10.22% | 1.95% | 0.77% | 13.18% |
| val | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |
| test | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

| split | docs | clusters | mentions | multi-mention | singletons | mention pairs | pronoun mentions | pronoun share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 13 | 235 | 1,056 | 235 | 0 | 5,078 | 270 | 25.57% |
| val | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |
| test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | n/a |

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 218 | 516 | 8,766 |
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
| person | 9,328 | 8.97% |
| location | 5,303 | 5.10% |
| date | 4,659 | 4.48% |
| Location | 4,547 | 4.37% |
| Date | 3,973 | 3.82% |
| cardinal | 2,954 | 2.84% |
| organization | 2,794 | 2.69% |
| Organization | 2,472 | 2.38% |
| StartDate | 1,921 | 1.85% |
| Product | 1,870 | 1.80% |
| job title | 1,794 | 1.73% |
| EquityHolder | 1,757 | 1.69% |
| Casualties and Losses | 1,605 | 1.54% |
| sports team | 1,604 | 1.54% |
| StockCode | 1,600 | 1.54% |
| money | 1,439 | 1.38% |
| CompanyName | 1,364 | 1.31% |
| StockAbbr | 1,350 | 1.30% |
| geopolitical entity | 1,252 | 1.20% |
| PledgedShares | 1,170 | 1.13% |
| company | 1,138 | 1.09% |
| Cause | 1,032 | 0.99% |
| Person | 995 | 0.96% |
| TotalHoldingShares | 978 | 0.94% |
| media outlet | 978 | 0.94% |
| TradedShares | 971 | 0.93% |
| Pledgee | 958 | 0.92% |
| TotalHoldingRatio | 900 | 0.87% |
| Quantity | 883 | 0.85% |
| percent | 764 | 0.73% |
| Reduction Amount | 737 | 0.71% |
| facility | 730 | 0.70% |
| age | 716 | 0.69% |
| TotalPledgedShares | 691 | 0.66% |
| duration | 691 | 0.66% |
| product | 616 | 0.59% |
| time | 595 | 0.57% |
| EndDate | 594 | 0.57% |
| quantity | 578 | 0.56% |
| Measurement | 576 | 0.55% |
| occupation | 559 | 0.54% |
| ReleasedDate | 556 | 0.53% |
| Reducing Shareholder | 546 | 0.53% |
| government agency | 541 | 0.52% |
| work of art | 494 | 0.48% |
| Announcement Date | 444 | 0.43% |
| Company Name | 431 | 0.41% |
| Policy Content | 427 | 0.41% |
| PER | 419 | 0.40% |
| educational institution | 408 | 0.39% |
| Reduction Start Date | 404 | 0.39% |
| Money | 358 | 0.34% |
| LaterHoldingShares | 329 | 0.32% |
| Event | 328 | 0.32% |
| event | 320 | 0.31% |
| Candidates and their Political Parties | 316 | 0.30% |
| Contest Participant | 303 | 0.29% |
| nationality | 282 | 0.27% |
| AveragePrice | 280 | 0.27% |
| Participants | 273 | 0.26% |
| Facility | 271 | 0.26% |
| Protesters | 266 | 0.26% |
| Attacker | 259 | 0.25% |
| Target | 255 | 0.25% |
| pledge_amount | 253 | 0.24% |
| Nationality | 252 | 0.24% |
| Suspect | 248 | 0.24% |
| Weapon and Equippment | 247 | 0.24% |
| measurement | 240 | 0.23% |
| Areas affected | 240 | 0.23% |
| Protest Reason | 227 | 0.22% |
| crime | 219 | 0.21% |
| Affected Area | 219 | 0.21% |
| ordinal | 211 | 0.20% |
| Affected Areas | 209 | 0.20% |
| pledgor | 203 | 0.20% |
| Deliberating Agency | 202 | 0.19% |
| address | 198 | 0.19% |
| region | 196 | 0.19% |
| brand | 195 | 0.19% |
| disease | 187 | 0.18% |
| Recipient | 178 | 0.17% |
| Disease | 178 | 0.17% |
| Method | 172 | 0.17% |
| distance | 171 | 0.16% |
| Occupation | 171 | 0.16% |
| Person under Investigation | 168 | 0.16% |
| RepurchasedShares | 168 | 0.16% |
| Charged Crime | 168 | 0.16% |
| Precautionary Measure | 164 | 0.16% |
| Accepting Court | 163 | 0.16% |
| family relationship | 159 | 0.15% |
| Accident Investigator | 153 | 0.15% |
| Loss Amount | 152 | 0.15% |
| Belligerents | 152 | 0.15% |
| HighestTradingPrice | 150 | 0.14% |
| The Charged Crime | 149 | 0.14% |
| Acquiree | 149 | 0.14% |
| sports competition | 148 | 0.14% |
| protein | 144 | 0.14% |
| Investigative Agency | 140 | 0.13% |
| Individual_protein | 136 | 0.13% |
| Criminal Evidence | 136 | 0.13% |
| Position | 134 | 0.13% |
| Time | 133 | 0.13% |
| Game Name | 133 | 0.13% |
| Confirmed/Infected Cases | 126 | 0.12% |
| road | 125 | 0.12% |
| award | 123 | 0.12% |
| Accusation | 123 | 0.12% |
| RepurchaseAmount | 116 | 0.11% |
| pledge_start_date | 115 | 0.11% |
| landmark | 115 | 0.11% |
| GPE | 113 | 0.11% |
| ORG | 112 | 0.11% |
| chemical | 110 | 0.11% |
| Accused People | 110 | 0.11% |
| Increase Amount | 109 | 0.10% |
| Champions | 108 | 0.10% |
| Summit Theme | 108 | 0.10% |
| era | 107 | 0.10% |
| Election Name | 107 | 0.10% |
| Arrest Location | 107 | 0.10% |
| DiseaseOrPhenotypicFeature | 107 | 0.10% |
| Score | 106 | 0.10% |
| Winner | 105 | 0.10% |
| ruling_date | 103 | 0.10% |
| Frozen Amount | 103 | 0.10% |
| Award | 103 | 0.10% |
| Increasing Shareholder | 100 | 0.10% |
| GeneOrGeneProduct | 100 | 0.10% |
| ClosingDate | 99 | 0.10% |
| film | 98 | 0.09% |
| political affiliation | 96 | 0.09% |
| Vulnerability | 96 | 0.09% |
| Scheduled Landing Place | 95 | 0.09% |
| frozen_shareholder | 94 | 0.09% |
| Death Cases | 94 | 0.09% |
| FAC | 93 | 0.09% |
| team role | 90 | 0.09% |
| Contracting Parties | 90 | 0.09% |
| non-profit organization | 89 | 0.09% |
| Strike Reason | 89 | 0.09% |
| Aid Agency | 88 | 0.08% |
| The Final Votes and Percentages | 86 | 0.08% |
| Policy Name & Abbreviation | 85 | 0.08% |
| Location/Hospital | 85 | 0.08% |
| Flight No. | 85 | 0.08% |
| Storm Hit Location | 85 | 0.08% |
| law | 83 | 0.08% |
| medication | 83 | 0.08% |
| Aircraft Agency | 83 | 0.08% |
| Strike Outcome | 82 | 0.08% |
| military rank | 81 | 0.08% |
| LowestTradingPrice | 80 | 0.08% |
| Police | 80 | 0.08% |
| Judgement Result/Prison Term | 78 | 0.08% |
| Artifacts and Their Chronology | 78 | 0.08% |
| pledge_end_date | 77 | 0.07% |
| Capabilities | 77 | 0.07% |
| Deceased | 76 | 0.07% |
| Husband | 76 | 0.07% |
| Influence People | 76 | 0.07% |
| Economic Loss | 75 | 0.07% |
| Host | 75 | 0.07% |
| GeneProtein | 75 | 0.07% |
| ethnic group | 74 | 0.07% |
| Speaker | 74 | 0.07% |
| Anti-pollution People/Organizations | 74 | 0.07% |
| academic degree | 72 | 0.07% |
| Profession | 72 | 0.07% |
| Crew | 72 | 0.07% |
| Outbreak Location | 72 | 0.07% |
| regulatory body | 71 | 0.07% |
| Epidemic Data Issuing Agency | 71 | 0.07% |
| species | 70 | 0.07% |
| publication | 69 | 0.07% |
| software | 68 | 0.07% |
| Award Field | 68 | 0.07% |
| Protest Slogan | 67 | 0.06% |
| temperature | 66 | 0.06% |
| Strike Agency | 65 | 0.06% |
| System | 65 | 0.06% |
| Wife | 64 | 0.06% |
| military unit | 63 | 0.06% |
| Speaker Status | 63 | 0.06% |
| Record-breaking Project | 63 | 0.06% |
| Resignated Person | 63 | 0.06% |
| Increase Start Date | 62 | 0.06% |
| Damaged Facility | 62 | 0.06% |
| Task | 62 | 0.06% |
| Survivors | 61 | 0.06% |
| OtherScientificTerm | 61 | 0.06% |
| Freeze Start Date | 60 | 0.06% |
| Court | 60 | 0.06% |
| Magnitude | 60 | 0.06% |
| vehicle | 59 | 0.06% |
| Grades | 59 | 0.06% |
| Effective Date | 59 | 0.06% |
| Record Breaker | 58 | 0.06% |
| ChemicalEntity | 57 | 0.05% |
| Discover Location | 56 | 0.05% |
| Appointee | 56 | 0.05% |
| Time of the Case | 55 | 0.05% |
| Number of Vehicles Involved in the Crash | 54 | 0.05% |
| Prosecutor | 54 | 0.05% |
| Competition Items | 54 | 0.05% |
| Agreement Content | 54 | 0.05% |
| PII | 54 | 0.05% |
| body of water | 53 | 0.05% |
| Pollution Source | 53 | 0.05% |
| Solution | 53 | 0.05% |
| political party | 52 | 0.05% |
| Sponsor | 52 | 0.05% |
| Participating Countries | 52 | 0.05% |
| Start Date | 51 | 0.05% |
| Country Visited | 51 | 0.05% |
| Policy Proposals | 51 | 0.05% |
| pseudonym | 50 | 0.05% |
| Visitor | 50 | 0.05% |
| Arrest Time | 50 | 0.05% |
| Epicenter | 50 | 0.05% |
| frequency | 49 | 0.05% |
| Employment Agency | 49 | 0.05% |
| Resign Reason | 49 | 0.05% |
| Spectacle Start Time | 49 | 0.05% |
| weapon | 48 | 0.05% |
| Attending Hospital | 48 | 0.05% |
| News Release Agency | 48 | 0.05% |
| Spectacle Location | 48 | 0.05% |
| holiday | 47 | 0.05% |
| song | 47 | 0.05% |
| name | 47 | 0.05% |
| Spacecraft | 46 | 0.04% |
| Types of the Spectacle | 46 | 0.04% |
| Beneficiary | 45 | 0.04% |
| Funding | 45 | 0.04% |
| Affected Industries | 45 | 0.04% |
| People | 45 | 0.04% |
| Conflict Duration | 44 | 0.04% |
| Released People | 44 | 0.04% |
| Microorganism | 44 | 0.04% |
| anatomical structure | 43 | 0.04% |
| Age | 43 | 0.04% |
| Previous Record Holder | 43 | 0.04% |
| speed | 42 | 0.04% |
| WEA | 42 | 0.04% |
| Areas Affected | 42 | 0.04% |
| Strikers Status | 41 | 0.04% |
| Way for Spreading | 41 | 0.04% |
| symptom | 40 | 0.04% |
| religious group | 40 | 0.04% |
| honorific | 40 | 0.04% |
| Death Reason | 40 | 0.04% |
| Fine Reason | 40 | 0.04% |
| File | 40 | 0.04% |
| LegalInstitution | 40 | 0.04% |
| Host Country | 40 | 0.04% |
| regulation | 39 | 0.04% |
| book | 39 | 0.04% |
| model number | 39 | 0.04% |
| Archaeologist | 39 | 0.04% |
| Gene/protein/RNA | 39 | 0.04% |
| Anatomy | 39 | 0.04% |
| CellType | 39 | 0.04% |
| court case | 38 | 0.04% |
| Head of Institution | 38 | 0.04% |
| Number of Evacuated People | 38 | 0.04% |
| Arrested | 38 | 0.04% |
| Victim | 37 | 0.04% |
| Device | 37 | 0.04% |
| Generic | 37 | 0.04% |
| FrozeShares | 37 | 0.04% |
| Start Time | 37 | 0.04% |
| Military Strength | 37 | 0.04% |
| Weapon | 36 | 0.03% |
| The Grades of the Previous Record Holder | 36 | 0.03% |
| Data | 36 | 0.03% |
| Achievement | 36 | 0.03% |
| Election Goal | 35 | 0.03% |
| Freeze End Date | 35 | 0.03% |
| Passengers | 35 | 0.03% |
| LOC | 35 | 0.03% |
| OrganismTaxon | 35 | 0.03% |
| protein-family | 34 | 0.03% |
| Army | 34 | 0.03% |
| Number of Damaged Houses | 34 | 0.03% |
| Historical Sites | 34 | 0.03% |
| Riot Reason | 34 | 0.03% |
| area | 33 | 0.03% |
| Evidence | 33 | 0.03% |
| Material | 32 | 0.03% |
| Employed Institution | 32 | 0.03% |
| Habitat | 32 | 0.03% |
| Summit Name | 31 | 0.03% |
| Defense Lawyer | 30 | 0.03% |
| Spokenmen | 30 | 0.03% |
| Agreement Name | 30 | 0.03% |
| Research Agency | 30 | 0.03% |
| Bank Name | 30 | 0.03% |
| Party Responsible for Pollution | 30 | 0.03% |
| Discover Time | 29 | 0.03% |
| senior_personnel | 29 | 0.03% |
| Taking-off Place | 29 | 0.03% |
| Award Reason | 29 | 0.03% |
| Government Reaction | 29 | 0.03% |
| currency | 28 | 0.03% |
| Strikers | 28 | 0.03% |
| senior_position | 28 | 0.03% |
| Software | 28 | 0.03% |
| Susceptible Population | 28 | 0.03% |
| sentence | 27 | 0.03% |
| Boycotted Institutions | 27 | 0.03% |
| compensation_recipient | 27 | 0.03% |
| Spacecraft Mission | 27 | 0.03% |
| Spokeswoman/Spokesman | 27 | 0.03% |
| Storm Warning Level | 27 | 0.03% |
| Investigating agency | 27 | 0.03% |
| Symptoms | 27 | 0.03% |
| Region | 27 | 0.03% |
| Organization Name | 27 | 0.03% |
| SequenceVariant | 27 | 0.03% |
| conference | 26 | 0.03% |
| Head of Investigation Team | 26 | 0.03% |
| Patch | 26 | 0.03% |
| Last Job of the Appointee | 26 | 0.03% |
| Perpetrators | 26 | 0.03% |
| People/Organization who predicted the disaster | 26 | 0.03% |
| Aid Reason | 26 | 0.03% |
| weight | 25 | 0.02% |
| Military Exercise | 25 | 0.02% |
| market index | 25 | 0.02% |
| Poisoning Type | 25 | 0.02% |
| fiscal quarter | 24 | 0.02% |
| operating system | 24 | 0.02% |
| organism | 24 | 0.02% |
| Regulatory Authority | 24 | 0.02% |
| Launch Date | 24 | 0.02% |
| Purpose | 24 | 0.02% |
| criminal organization | 23 | 0.02% |
| Cooperative Agency | 23 | 0.02% |
| Law | 23 | 0.02% |
| Bill Drafting Agency | 23 | 0.02% |
| compensation_amount | 22 | 0.02% |
| Launch Site | 22 | 0.02% |
| Appointer | 22 | 0.02% |
| Volcano Name | 22 | 0.02% |
| Judge | 22 | 0.02% |
| End Time | 22 | 0.02% |
| Maximum Rainfall | 22 | 0.02% |
| language | 21 | 0.02% |
| End Date | 21 | 0.02% |
| Symptom | 21 | 0.02% |
| Commanders and their Position | 21 | 0.02% |
| Archaeologist Organization | 21 | 0.02% |
| medical procedure | 20 | 0.02% |
| Goal | 20 | 0.02% |
| Storm Name | 20 | 0.02% |
| Outbreak Date | 20 | 0.02% |
| DNA | 20 | 0.02% |
| Shipwreck Reason | 20 | 0.02% |
| Fined Agency | 19 | 0.02% |
| Launch Country | 19 | 0.02% |
| Number of People Hospitalized | 19 | 0.02% |
| Maximum Wind Speed | 19 | 0.02% |
| Treatment Method | 19 | 0.02% |
| Related Rivers or Lakes | 19 | 0.02% |
| Area Affected | 19 | 0.02% |
| title | 18 | 0.02% |
| Aid Supplies/Amount | 18 | 0.02% |
| Weapon Used | 18 | 0.02% |
| Stolen Amount | 18 | 0.02% |
| Marriage Duration | 18 | 0.02% |
| Unemployed Rate | 18 | 0.02% |
| CellLine | 18 | 0.02% |
| Predecessor | 18 | 0.02% |
| Turnout(who win who lost) | 17 | 0.02% |
| Appointment Time | 17 | 0.02% |
| Publication | 17 | 0.02% |
| Malware | 17 | 0.02% |
| The Sentence Claimed by the Prosecutor's Lawyer | 17 | 0.02% |
| Number of Rescued People | 17 | 0.02% |
| Damaged Property | 17 | 0.02% |
| Death Time | 17 | 0.02% |
| Countries Withdrawing from the Organization | 17 | 0.02% |
| Strike Industry | 17 | 0.02% |
| Species | 17 | 0.02% |
| description | 17 | 0.02% |
| dosage | 16 | 0.02% |
| stock ticker | 16 | 0.02% |
| Penalty Amount | 16 | 0.02% |
| Last Time the Record was Broken | 16 | 0.02% |
| Prosecution Lawyer | 16 | 0.02% |
| Court Time | 16 | 0.02% |
| Missings | 16 | 0.02% |
| Rescue Organizer | 16 | 0.02% |
| The Final Seats Result | 16 | 0.02% |
| Acquirer | 16 | 0.02% |
| airport | 15 | 0.01% |
| Scale | 15 | 0.01% |
| Release Reason | 15 | 0.01% |
| Sentencing Location | 15 | 0.01% |
| Number of Destroyed Building | 15 | 0.01% |
| Wedding Venue | 15 | 0.01% |
| Ship No. | 15 | 0.01% |
| Website | 15 | 0.01% |
| MVP | 15 | 0.01% |
| Voting method | 15 | 0.01% |
| Other Loss | 14 | 0.01% |
| Lasting Time | 14 | 0.01% |
| hardware component | 14 | 0.01% |
| Start_Date | 14 | 0.01% |
| Storm Hit Time | 14 | 0.01% |
| Doctor and Medical Team | 14 | 0.01% |
| The Agency Who Broke the Agreement | 14 | 0.01% |
| Approver | 14 | 0.01% |
| Electoral System | 14 | 0.01% |
| Forecasting Agency | 14 | 0.01% |
| Declarer | 14 | 0.01% |
| Death Rate | 14 | 0.01% |
| Tribe | 14 | 0.01% |
| deadline | 13 | 0.01% |
| Pests | 13 | 0.01% |
| Perpetrator | 13 | 0.01% |
| Number of Victims | 13 | 0.01% |
| Concept | 13 | 0.01% |
| Astronauts | 13 | 0.01% |
| Illness | 13 | 0.01% |
| Number of Aftershocks | 13 | 0.01% |
| VEH | 13 | 0.01% |
| Child | 13 | 0.01% |
| Organization Industry | 13 | 0.01% |
| Number of Trapped People | 13 | 0.01% |
| Version | 13 | 0.01% |
| Company Industry | 13 | 0.01% |
| Spectacle End Time | 13 | 0.01% |
| id | 13 | 0.01% |
| Ethnic Group | 13 | 0.01% |
| Launch Result | 12 | 0.01% |
| Detention Start Time | 12 | 0.01% |
| Refuge | 12 | 0.01% |
| protein-complex | 12 | 0.01% |
| Rescue Tool or Method | 12 | 0.01% |
| Water Level | 12 | 0.01% |
| State Before Death | 12 | 0.01% |
| protein-domain | 12 | 0.01% |
| Spectacle Duration | 12 | 0.01% |
| Train No. | 12 | 0.01% |
| postal code | 11 | 0.01% |
| trade union | 11 | 0.01% |
| treaty | 11 | 0.01% |
| natural disaster | 11 | 0.01% |
| Duration | 11 | 0.01% |
| Carrier Rocket | 11 | 0.01% |
| Casualty Count | 11 | 0.01% |
| VAL | 11 | 0.01% |
| Age of the Appointee | 11 | 0.01% |
| Term of Office | 11 | 0.01% |
| Jail Time | 11 | 0.01% |
| Transportation | 11 | 0.01% |
| Military Unit | 11 | 0.01% |
| financial instrument | 10 | 0.01% |
| war | 10 | 0.01% |
| Property Division | 10 | 0.01% |
| Successor | 10 | 0.01% |
| Acquisition amount | 10 | 0.01% |
| Country | 10 | 0.01% |
| Number | 10 | 0.01% |
| Inauguration Time | 10 | 0.01% |
| Age of the Resignated Person | 10 | 0.01% |
| Witness | 10 | 0.01% |
| role | 10 | 0.01% |
| aircraft | 9 | 0.01% |
| witness | 9 | 0.01% |
| Ways to Watch the Speech | 9 | 0.01% |
| Group | 9 | 0.01% |
| Reasons for the Formation of the Historical sites | 9 | 0.01% |
| Hostage | 9 | 0.01% |
| Disaster-Stricken Farmland | 9 | 0.01% |
| experimental-construct | 9 | 0.01% |
| Projects | 9 | 0.01% |
| content | 9 | 0.01% |
| Gene | 9 | 0.01% |
| disclosure | 9 | 0.01% |
| page | 9 | 0.01% |
| Influenced People | 9 | 0.01% |
| environment | 9 | 0.01% |
| mountain | 8 | 0.01% |
| Influenced Crops and Livelihood | 8 | 0.01% |
| Number of Influenced People | 8 | 0.01% |
| Body Part | 8 | 0.01% |
| COM | 8 | 0.01% |
| Trapped Depth | 8 | 0.01% |
| RNA | 8 | 0.01% |
| Amount of Precipitation | 8 | 0.01% |
| author | 8 | 0.01% |
| Bankrupt Business | 8 | 0.01% |
| Prison Term | 8 | 0.01% |
| Release Time | 8 | 0.01% |
| pathogen | 7 | 0.01% |
| database | 7 | 0.01% |
| video game | 7 | 0.01% |
| clinical trial | 7 | 0.01% |
| Animal | 7 | 0.01% |
| Last Outbreak Time | 7 | 0.01% |
| Service Years | 7 | 0.01% |
| city | 7 | 0.01% |
| state | 7 | 0.01% |
| Social Group | 7 | 0.01% |
| Storm Direction | 7 | 0.01% |
| Number of Damaged House | 7 | 0.01% |
| rating | 7 | 0.01% |
| Damaged Crops & Livestock | 7 | 0.01% |
| PaymentMethod | 7 | 0.01% |
| Phenotype | 7 | 0.01% |
| Metric | 7 | 0.01% |
| Annoncement Date | 7 | 0.01% |
| ai_model | 7 | 0.01% |
| Protein_family_or_group | 6 | 0.01% |
| Taxon | 6 | 0.01% |
| Temporary Settlement | 6 | 0.01% |
| Wedding Dress Designer | 6 | 0.01% |
| Alternate Landing Place | 6 | 0.01% |
| Economic loss | 6 | 0.01% |
| Organization Leader | 6 | 0.01% |
| unit | 6 | 0.01% |
| Agreement Validity Period | 6 | 0.01% |
| religious_organization | 6 | 0.01% |
| Mission Duration | 6 | 0.01% |
| problem_class | 6 | 0.01% |
| constraint_property | 6 | 0.01% |
| mathematician | 6 | 0.01% |
| mathematical_result | 6 | 0.01% |
| cell | 6 | 0.01% |
| CVE | 6 | 0.01% |
| UnfrozeDate | 6 | 0.01% |
| Special Medicine | 6 | 0.01% |
| Medical Condition | 6 | 0.01% |
| City | 6 | 0.01% |
| Warning Device | 6 | 0.01% |
| algorithm | 5 | 0.00% |
| protocol | 5 | 0.00% |
| Train Agency | 5 | 0.00% |
| Casualities and Losses | 5 | 0.00% |
| symptoms | 5 | 0.00% |
| Fire Warning Level | 5 | 0.00% |
| Registered Capital | 5 | 0.00% |
| Economists who Predicted the Crisis | 5 | 0.00% |
| experiment-tag | 5 | 0.00% |
| Storm Movement Speed | 5 | 0.00% |
| The State of the Volcano (Dormant or Active) | 5 | 0.00% |
| Agency | 5 | 0.00% |
| researcher | 5 | 0.00% |
| Invited Person | 5 | 0.00% |
| Merger Terms | 5 | 0.00% |
| CellComponent | 5 | 0.00% |
| Provisional Government | 5 | 0.00% |
| Recovered Amount | 5 | 0.00% |
| price | 5 | 0.00% |
| Number of Vaccinated People | 5 | 0.00% |
| keywords | 5 | 0.00% |
| People Group | 5 | 0.00% |
| comment | 5 | 0.00% |
| Tic Type | 5 | 0.00% |
| vessel | 4 | 0.00% |
| tax | 4 | 0.00% |
| patent | 4 | 0.00% |
| specificDataElements | 4 | 0.00% |
| Substance | 4 | 0.00% |
| Plant | 4 | 0.00% |
| framework | 4 | 0.00% |
| Other Impact | 4 | 0.00% |
| process | 4 | 0.00% |
| Child Custody | 4 | 0.00% |
| Head of the Merged Organization | 4 | 0.00% |
| productFeatures | 4 | 0.00% |
| reagent | 4 | 0.00% |
| benefits | 4 | 0.00% |
| Investigating Spokesperson | 4 | 0.00% |
| End_Date | 4 | 0.00% |
| Cured Cases | 4 | 0.00% |
| Countries Joined the Organization | 4 | 0.00% |
| metric | 4 | 0.00% |
| mutation | 4 | 0.00% |
| fusion-protein | 4 | 0.00% |
| border | 3 | 0.00% |
| authors | 3 | 0.00% |
| publicationDate | 3 | 0.00% |
| Announce Platform | 3 | 0.00% |
| Prison | 3 | 0.00% |
| Tear Up Reason | 3 | 0.00% |
| assay | 3 | 0.00% |
| Trapped Days | 3 | 0.00% |
| Sequelae | 3 | 0.00% |
| yield | 3 | 0.00% |
| streetAddress | 3 | 0.00% |
| The Worst-Hit Area | 3 | 0.00% |
| mathematical_concept | 3 | 0.00% |
| phone | 3 | 0.00% |
| Best Way to Shoot | 3 | 0.00% |
| GRIStandard | 3 | 0.00% |
| UNGCPrinciple | 3 | 0.00% |
| SASBStandard | 3 | 0.00% |
| protein-motif | 3 | 0.00% |
| Head of the Institution | 3 | 0.00% |
| software_library | 3 | 0.00% |
| hardware | 3 | 0.00% |
| Stamp | 3 | 0.00% |
| Value | 3 | 0.00% |
| amount | 3 | 0.00% |
| Nation | 3 | 0.00% |
| Storm Formation Location | 3 | 0.00% |
| Tsunamis | 3 | 0.00% |
| Tsunami Warning Level | 3 | 0.00% |
| voltage | 2 | 0.00% |
| security vulnerability | 2 | 0.00% |
| api | 2 | 0.00% |
| file format | 2 | 0.00% |
| contract | 2 | 0.00% |
| contactEmail | 2 | 0.00% |
| startDate | 2 | 0.00% |
| endDate | 2 | 0.00% |
| link | 2 | 0.00% |
| Culture | 2 | 0.00% |
| Site | 2 | 0.00% |
| Ship | 2 | 0.00% |
| gender | 2 | 0.00% |
| Commanders of the Army | 2 | 0.00% |
| religious_event | 2 | 0.00% |
| Document | 2 | 0.00% |
| mathematical_object | 2 | 0.00% |
| formalism | 2 | 0.00% |
| Response Measures | 2 | 0.00% |
| geometric_object | 2 | 0.00% |
| numerical_function | 2 | 0.00% |
| mathematical_structure | 2 | 0.00% |
| Candidate Scandal(Allegations of fraud, etc.) | 2 | 0.00% |
| network | 2 | 0.00% |
| zipCode | 2 | 0.00% |
| People Casting Key Votes | 2 | 0.00% |
| Postpone Reason | 2 | 0.00% |
| Rescue Start Time | 2 | 0.00% |
| Pest | 2 | 0.00% |
| Equipment | 2 | 0.00% |
| Damage | 2 | 0.00% |
| issueNumber | 2 | 0.00% |
| publisher | 2 | 0.00% |
| Vaccine Research and Development Organization | 2 | 0.00% |
| Time Period | 2 | 0.00% |
| Writing System | 2 | 0.00% |
| actor | 2 | 0.00% |
| genres | 2 | 0.00% |
| Therapy | 2 | 0.00% |
| Head of the Government | 2 | 0.00% |
| equipmentName | 2 | 0.00% |
| manufacturer | 2 | 0.00% |
| serialNumber | 2 | 0.00% |
| modelNumber | 2 | 0.00% |
| Book | 2 | 0.00% |
| Location/Nation | 2 | 0.00% |
| Feature | 2 | 0.00% |
| agent | 2 | 0.00% |
| target | 2 | 0.00% |
| Plant Species | 2 | 0.00% |
| expirationDate | 2 | 0.00% |
| Civilization | 2 | 0.00% |
| itemId | 2 | 0.00% |
| unitPrice | 2 | 0.00% |
| street1 | 2 | 0.00% |
| postalCode | 2 | 0.00% |
| email | 2 | 0.00% |
| INF | 2 | 0.00% |
| protein-region | 2 | 0.00% |
| Storm Center Location | 2 | 0.00% |
| url | 2 | 0.00% |
| tags | 2 | 0.00% |
| Medical Test | 2 | 0.00% |
| Organization Members | 2 | 0.00% |
| Religion | 2 | 0.00% |
| Geographic Feature | 2 | 0.00% |
| technical_method | 2 | 0.00% |
| Production Cuts | 2 | 0.00% |
| Withdraw Reason | 2 | 0.00% |
| Magnitude(Tsunami heights) | 2 | 0.00% |
| programming language | 1 | 0.00% |
| diagnosis | 1 | 0.00% |
| verdict | 1 | 0.00% |
| mineral | 1 | 0.00% |
| gene | 1 | 0.00% |
| license | 1 | 0.00% |
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
| Organization Affiliation | 1 | 0.00% |
| Lawyer | 1 | 0.00% |
| Biological Entity | 1 | 0.00% |
| Artifact | 1 | 0.00% |
| Structure | 1 | 0.00% |
| Cost | 1 | 0.00% |
| Suspected Cases | 1 | 0.00% |
| Organization_Military | 1 | 0.00% |
| reporter | 1 | 0.00% |
| Agreement Members | 1 | 0.00% |
| Cultural Artifact | 1 | 0.00% |
| Legal Concept | 1 | 0.00% |
| Practice | 1 | 0.00% |
| Economic System | 1 | 0.00% |
| Identity | 1 | 0.00% |
| productionDate | 1 | 0.00% |
| totalQuantityProduced | 1 | 0.00% |
| totalDuration | 1 | 0.00% |
| overallYield | 1 | 0.00% |
| businessName | 1 | 0.00% |
| zip | 1 | 0.00% |
| sharesOwned | 1 | 0.00% |
| votingProcedures | 1 | 0.00% |
| capitalContributions | 1 | 0.00% |
| disputeResolution | 1 | 0.00% |
| administrative_body | 1 | 0.00% |
| educational_institution | 1 | 0.00% |
| productId | 1 | 0.00% |
| productPrice | 1 | 0.00% |
| Land Designation | 1 | 0.00% |
| Designation | 1 | 0.00% |
| complexity_class | 1 | 0.00% |
| academic_field | 1 | 0.00% |
| product_id | 1 | 0.00% |
| num_reviews | 1 | 0.00% |
| review_sentiment | 1 | 0.00% |
| publication_source | 1 | 0.00% |
| targetNetwork | 1 | 0.00% |
| backendService | 1 | 0.00% |
| backendBuckets | 1 | 0.00% |
| forwardingRule | 1 | 0.00% |
| healthCheck | 1 | 0.00% |
| instanceGroupManagers | 1 | 0.00% |
| networkEndpointGroup | 1 | 0.00% |
| Compensation | 1 | 0.00% |
| firstName | 1 | 0.00% |
| lastName | 1 | 0.00% |
| phoneNumber | 1 | 0.00% |
| emailAddress | 1 | 0.00% |
| website | 1 | 0.00% |
| equityOffered | 1 | 0.00% |
| font | 1 | 0.00% |
| fontSize | 1 | 0.00% |
| color | 1 | 0.00% |
| employeeTurnoverRate | 1 | 0.00% |
| category | 1 | 0.00% |
| inStock | 1 | 0.00% |
| Character | 1 | 0.00% |
| Theory | 1 | 0.00% |
| Scientific Name | 1 | 0.00% |
| Control Method | 1 | 0.00% |
| Strategy | 1 | 0.00% |
| Natural Enemy | 1 | 0.00% |
| Lure | 1 | 0.00% |
| coverPrice | 1 | 0.00% |
| synopsis | 1 | 0.00% |
| Source | 1 | 0.00% |
| batchNumber | 1 | 0.00% |
| reviewDate | 1 | 0.00% |
| reviewComment | 1 | 0.00% |
| reviewStatus | 1 | 0.00% |
| State of the Hull | 1 | 0.00% |
| Percentage | 1 | 0.00% |
| Role | 1 | 0.00% |
| Alliance | 1 | 0.00% |
| Government System | 1 | 0.00% |
| Course | 1 | 0.00% |
| hotelId | 1 | 0.00% |
| lat | 1 | 0.00% |
| lng | 1 | 0.00% |
| roomTypeId | 1 | 0.00% |
| capacity | 1 | 0.00% |
| reservationId | 1 | 0.00% |
| numGuests | 1 | 0.00% |
| totalPrice | 1 | 0.00% |
| status | 1 | 0.00% |
| year | 1 | 0.00% |
| director | 1 | 0.00% |
| plot | 1 | 0.00% |
| poster | 1 | 0.00% |
| programming_language | 1 | 0.00% |
| Organization_Other | 1 | 0.00% |
| Join Conditions | 1 | 0.00% |
| Protein_complex | 1 | 0.00% |
| Death Age | 1 | 0.00% |
| Disorder | 1 | 0.00% |
| Disorder Group | 1 | 0.00% |
| Demographic | 1 | 0.00% |
| Toxin | 1 | 0.00% |
| procedureId | 1 | 0.00% |
| department | 1 | 0.00% |
| disposalMethod | 1 | 0.00% |
| reasonForDisposal | 1 | 0.00% |
| additionalInfo | 1 | 0.00% |
| minAge | 1 | 0.00% |
| maxAge | 1 | 0.00% |
| fundingSource | 1 | 0.00% |
| conclusions | 1 | 0.00% |
| Invalid Date | 1 | 0.00% |
| Field of Study | 1 | 0.00% |
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
| Hull Discovery Time | 1 | 0.00% |
| How Many Times Get Married | 1 | 0.00% |
| Stamp Version | 1 | 0.00% |
| Hull Location | 1 | 0.00% |
| Economy Loss | 1 | 0.00% |
| exchangeRate | 1 | 0.00% |
| Plant Group | 1 | 0.00% |
| Habitat Feature | 1 | 0.00% |
| Chemical | 1 | 0.00% |
| recipientName | 1 | 0.00% |
| recipientEmail | 1 | 0.00% |
| senderName | 1 | 0.00% |
| senderEmail | 1 | 0.00% |
| message | 1 | 0.00% |
| device_id | 1 | 0.00% |
| version | 1 | 0.00% |
| binary | 1 | 0.00% |
| digest | 1 | 0.00% |
| updated_at | 1 | 0.00% |
| quotationId | 1 | 0.00% |
| supplierId | 1 | 0.00% |
| supplierName | 1 | 0.00% |
| quotationDate | 1 | 0.00% |
| totalCost | 1 | 0.00% |
| termsAndConditions | 1 | 0.00% |
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
| peptide | 1 | 0.00% |
| Postpone Time | 1 | 0.00% |
| season | 1 | 0.00% |
| episode | 1 | 0.00% |
| source | 1 | 0.00% |
| coverArtTitle | 1 | 0.00% |
| coverArtImage | 1 | 0.00% |
| coverArtDescription | 1 | 0.00% |
| series | 1 | 0.00% |
| Classification | 1 | 0.00% |
| Company | 1 | 0.00% |
| Language Family | 1 | 0.00% |
| Language Group | 1 | 0.00% |
| MON | 1 | 0.00% |
| Complications | 1 | 0.00% |
| dataset | 1 | 0.00% |
| architectural_component | 1 | 0.00% |
| protein-isoform | 1 | 0.00% |
| organelle | 1 | 0.00% |
| Cured Rate | 1 | 0.00% |
| Year | 1 | 0.00% |
| interest rate | 0 | 0.00% |
| vaccine | 0 | 0.00% |
| budget line | 0 | 0.00% |
| allergen | 0 | 0.00% |
| credit rating | 0 | 0.00% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| no relation | 740 | 10.10% |
| works_for | 412 | 5.62% |
| located_in | 239 | 3.26% |
| bind | 217 | 2.96% |
| member_of | 209 | 2.85% |
| based_in | 147 | 2.01% |
| headquartered_in | 144 | 1.97% |
| PPI | 136 | 1.86% |
| affiliated_with | 135 | 1.84% |
| leader_of | 130 | 1.77% |
| Association | 124 | 1.69% |
| employer_of | 115 | 1.57% |
| instance of | 107 | 1.46% |
| USED-FOR | 97 | 1.32% |
| competitor_of | 78 | 1.06% |
| founded | 76 | 1.04% |
| produced_by | 63 | 0.86% |
| part_of | 61 | 0.83% |
| country | 60 | 0.82% |
| Lives_In | 57 | 0.78% |
| occupation | 55 | 0.75% |
| shares border with | 54 | 0.74% |
| operates_in | 52 | 0.71% |
| family name | 49 | 0.67% |
| Positive_Correlation | 46 | 0.63% |
| located in the administrative territorial entity | 44 | 0.60% |
| given name | 44 | 0.60% |
| colleague_of | 40 | 0.55% |
| founded by | 38 | 0.52% |
| represents | 37 | 0.50% |
| reports_to | 35 | 0.48% |
| director | 34 | 0.46% |
| has part(s) | 34 | 0.46% |
| genre | 33 | 0.45% |
| member of | 32 | 0.44% |
| acquired | 31 | 0.42% |
| place of birth | 29 | 0.40% |
| participated_in | 29 | 0.40% |
| Negative_Correlation | 28 | 0.38% |
| CONJUNCTION | 28 | 0.38% |
| contains the administrative territorial entity | 26 | 0.35% |
| platform | 25 | 0.34% |
| headquarters location | 25 | 0.34% |
| supplier_of | 24 | 0.33% |
| father | 23 | 0.31% |
| child | 23 | 0.31% |
| replaces | 22 | 0.30% |
| participant | 22 | 0.30% |
| location | 22 | 0.30% |
| followed by | 22 | 0.30% |
| citizen_of | 22 | 0.30% |
| sibling | 21 | 0.29% |
| part of | 21 | 0.29% |
| educated at | 20 | 0.27% |
| country of citizenship | 20 | 0.27% |
| spouse | 18 | 0.25% |
| relative_of | 18 | 0.25% |
| position held | 18 | 0.25% |
| notable work | 18 | 0.25% |
| award received | 18 | 0.25% |
| authored_by | 18 | 0.25% |
| performer | 18 | 0.25% |
| different from | 18 | 0.25% |
| subsidiary_of | 17 | 0.23% |
| nominated for | 17 | 0.23% |
| employer | 17 | 0.23% |
| present in work | 17 | 0.23% |
| connects with | 16 | 0.22% |
| league | 16 | 0.22% |
| languages spoken, written or signed | 16 | 0.22% |
| owned by | 16 | 0.22% |
| continent | 16 | 0.22% |
| family | 16 | 0.22% |
| language used | 16 | 0.22% |
| sport | 16 | 0.22% |
| conflict | 15 | 0.20% |
| developed_by | 15 | 0.20% |
| developer | 15 | 0.20% |
| regulated_by | 15 | 0.20% |
| composer | 15 | 0.20% |
| uses | 15 | 0.20% |
| author | 15 | 0.20% |
| main subject | 15 | 0.20% |
| residence | 15 | 0.20% |
| spouse_of | 14 | 0.19% |
| military branch | 14 | 0.19% |
| located in or next to body of water | 14 | 0.19% |
| location of formation | 14 | 0.19% |
| participant in | 14 | 0.19% |
| place of death | 14 | 0.19% |
| cast member | 14 | 0.19% |
| architect | 14 | 0.19% |
| noble title | 13 | 0.18% |
| characters | 13 | 0.18% |
| significant event | 13 | 0.18% |
| significant person | 13 | 0.18% |
| based on | 13 | 0.18% |
| opposite of | 13 | 0.18% |
| manufactured_by | 13 | 0.18% |
| named after | 13 | 0.18% |
| head of government | 13 | 0.18% |
| native language | 12 | 0.16% |
| follows | 12 | 0.16% |
| located in/on physical feature | 12 | 0.16% |
| creator | 12 | 0.16% |
| place of publication | 12 | 0.16% |
| parent organization | 12 | 0.16% |
| invested_in | 12 | 0.16% |
| place served by transport hub | 12 | 0.16% |
| published_by | 12 | 0.16% |
| mother | 12 | 0.16% |
| parent_of | 12 | 0.16% |
| charged_with | 12 | 0.16% |
| field of work | 11 | 0.15% |
| basin country | 11 | 0.15% |
| connecting line | 11 | 0.15% |
| producer | 11 | 0.15% |
| biological process | 11 | 0.15% |
| conferred by | 11 | 0.15% |
| located in protected area | 11 | 0.15% |
| has use | 11 | 0.15% |
| fictional universe described in | 11 | 0.15% |
| directed_by | 11 | 0.15% |
| EVALUATE-FOR | 11 | 0.15% |
| narrative location | 11 | 0.15% |
| facet of | 11 | 0.15% |
| said to be the same as | 10 | 0.14% |
| librettist | 10 | 0.14% |
| stock market index | 10 | 0.14% |
| operator | 10 | 0.14% |
| chairperson | 10 | 0.14% |
| presenter | 10 | 0.14% |
| parent taxon | 10 | 0.14% |
| board_member_of | 10 | 0.14% |
| child_of | 10 | 0.14% |
| terminus | 10 | 0.14% |
| home venue | 10 | 0.14% |
| capital of | 10 | 0.14% |
| owns | 10 | 0.14% |
| production company | 10 | 0.14% |
| member of political party | 10 | 0.14% |
| partner in business or sport | 10 | 0.14% |
| distributed by | 10 | 0.14% |
| brand | 9 | 0.12% |
| historic county | 9 | 0.12% |
| product or material produced | 9 | 0.12% |
| publisher | 9 | 0.12% |
| school district | 9 | 0.12% |
| crosses | 9 | 0.12% |
| mountain range | 9 | 0.12% |
| relative | 9 | 0.12% |
| influenced by | 9 | 0.12% |
| incarnation of | 9 | 0.12% |
| sports discipline competed in | 9 | 0.12% |
| occupant | 9 | 0.12% |
| located on terrain feature | 9 | 0.12% |
| manufacturer | 9 | 0.12% |
| place of burial | 9 | 0.12% |
| original broadcaster | 9 | 0.12% |
| FEATURE-OF | 9 | 0.12% |
| architectural style | 9 | 0.12% |
| interacts_with | 9 | 0.12% |
| item operated | 9 | 0.12% |
| sibling_of | 9 | 0.12% |
| time period | 8 | 0.11% |
| crew member(s) | 8 | 0.11% |
| applies to jurisdiction | 8 | 0.11% |
| ethnic group | 8 | 0.11% |
| capital | 8 | 0.11% |
| mouth of the watercourse | 8 | 0.11% |
| secretary general | 8 | 0.11% |
| record label | 8 | 0.11% |
| parent astronomical body | 8 | 0.11% |
| vehicle | 8 | 0.11% |
| dialect of | 8 | 0.11% |
| located on street | 8 | 0.11% |
| indigenous to | 8 | 0.11% |
| licensed to broadcast to | 8 | 0.11% |
| has effect | 8 | 0.11% |
| position played on team / speciality | 8 | 0.11% |
| designed_by | 8 | 0.11% |
| student of | 8 | 0.11% |
| subclass of | 8 | 0.11% |
| broadcast by | 8 | 0.11% |
| work location | 8 | 0.11% |
| PART-OF | 8 | 0.11% |
| country of origin | 8 | 0.11% |
| allegiance | 8 | 0.11% |
| this taxon is source of | 7 | 0.10% |
| medical condition | 7 | 0.10% |
| cause of death | 7 | 0.10% |
| medical condition treated | 7 | 0.10% |
| officeholder | 7 | 0.10% |
| military rank | 7 | 0.10% |
| online service | 7 | 0.10% |
| instrument | 7 | 0.10% |
| member of sports team | 7 | 0.10% |
| participating team | 7 | 0.10% |
| movement | 7 | 0.10% |
| has edition or translation | 7 | 0.10% |
| surface played on | 7 | 0.10% |
| dam | 7 | 0.10% |
| qualifies for event | 7 | 0.10% |
| owner of | 7 | 0.10% |
| canonization status | 7 | 0.10% |
| COMPARE | 7 | 0.10% |
| has contributing factor | 7 | 0.10% |
| form of creative work | 7 | 0.10% |
| has works in the collection | 7 | 0.10% |
| after a work by | 7 | 0.10% |
| voice type | 7 | 0.10% |
| filming location | 7 | 0.10% |
| located in the present-day administrative territorial entity | 7 | 0.10% |
| replaced by | 7 | 0.10% |
| vessel class | 7 | 0.10% |
| place of detention | 7 | 0.10% |
| location of discovery | 7 | 0.10% |
| located in the statistical territorial entity | 6 | 0.08% |
| country for sport | 6 | 0.08% |
| has part(s) of the class | 6 | 0.08% |
| signatory | 6 | 0.08% |
| commanded by | 6 | 0.08% |
| language of work or name | 6 | 0.08% |
| standards body | 6 | 0.08% |
| diocese | 6 | 0.08% |
| space launch vehicle | 6 | 0.08% |
| operating area | 6 | 0.08% |
| culture | 6 | 0.08% |
| winner | 6 | 0.08% |
| testified_against | 6 | 0.08% |
| significant place | 6 | 0.08% |
| honorific prefix | 6 | 0.08% |
| country of registry | 6 | 0.08% |
| found in taxon | 6 | 0.08% |
| born_in | 6 | 0.08% |
| partially coincident with | 6 | 0.08% |
| student | 6 | 0.08% |
| taxonomic type | 6 | 0.08% |
| candidate | 6 | 0.08% |
| wheel arrangement | 6 | 0.08% |
| highest judicial authority | 6 | 0.08% |
| pole position | 6 | 0.08% |
| coextensive with | 6 | 0.08% |
| history of topic | 6 | 0.08% |
| discoverer or inventor | 6 | 0.08% |
| candidacy in election | 6 | 0.08% |
| operating system | 6 | 0.08% |
| writing language | 6 | 0.08% |
| radio format | 6 | 0.08% |
| derived_from | 6 | 0.08% |
| territory overlaps | 6 | 0.08% |
| location of landing | 6 | 0.08% |
| anatomical location | 6 | 0.08% |
| terminus location | 6 | 0.08% |
| legislated by | 6 | 0.08% |
| distributor_of | 6 | 0.08% |
| HYPONYM-OF | 6 | 0.08% |
| musical conductor | 6 | 0.08% |
| designed by | 6 | 0.08% |
| has written for | 5 | 0.07% |
| constellation | 5 | 0.07% |
| has quality | 5 | 0.07% |
| part of the series | 5 | 0.07% |
| genetic association | 5 | 0.07% |
| head of state | 5 | 0.07% |
| physically interacts with | 5 | 0.07% |
| implementation of | 5 | 0.07% |
| child astronomical body | 5 | 0.07% |
| legal form | 5 | 0.07% |
| chief executive officer | 5 | 0.07% |
| has subsidiary | 5 | 0.07% |
| subject has role | 5 | 0.07% |
| published in | 5 | 0.07% |
| general classification of race participants | 5 | 0.07% |
| editor | 5 | 0.07% |
| grants | 5 | 0.07% |
| perpetrator | 5 | 0.07% |
| edition or translation of | 5 | 0.07% |
| competition class | 5 | 0.07% |
| transport network | 5 | 0.07% |
| currency | 5 | 0.07% |
| contributing factor of | 5 | 0.07% |
| package management system | 5 | 0.07% |
| programming language | 5 | 0.07% |
| this zoological name is coordinate with | 5 | 0.07% |
| distribution format | 5 | 0.07% |
| religion or worldview | 5 | 0.07% |
| mentor_of | 5 | 0.07% |
| heritage designation | 5 | 0.07% |
| drug or therapy used for treatment | 5 | 0.07% |
| advisor_to | 5 | 0.07% |
| recommended unit of measurement | 5 | 0.07% |
| IUCN conservation status | 5 | 0.07% |
| discography | 5 | 0.07% |
| target | 5 | 0.07% |
| is the study of | 5 | 0.07% |
| model item | 5 | 0.07% |
| located in time zone | 5 | 0.07% |
| chromosome | 5 | 0.07% |
| next higher rank | 5 | 0.07% |
| endemic to | 5 | 0.07% |
| organizer | 5 | 0.07% |
| Exhibits | 5 | 0.07% |
| member of the crew of | 4 | 0.05% |
| convicted of | 4 | 0.05% |
| stock exchange | 4 | 0.05% |
| legislative body | 4 | 0.05% |
| died_in | 4 | 0.05% |
| won | 4 | 0.05% |
| customer_of | 4 | 0.05% |
| mascot | 4 | 0.05% |
| lyrics by | 4 | 0.05% |
| contributed to creative work | 4 | 0.05% |
| depicted by | 4 | 0.05% |
| ancestral home | 4 | 0.05% |
| manner of death | 4 | 0.05% |
| named_after | 4 | 0.05% |
| fabrication method | 4 | 0.05% |
| character type | 4 | 0.05% |
| astronaut mission | 4 | 0.05% |
| enemy of | 4 | 0.05% |
| authority | 4 | 0.05% |
| office held by head of government | 4 | 0.05% |
| victory | 4 | 0.05% |
| territory claimed by | 4 | 0.05% |
| has immediate cause | 4 | 0.05% |
| cathedral | 4 | 0.05% |
| parent club | 4 | 0.05% |
| separated from | 4 | 0.05% |
| affiliation | 4 | 0.05% |
| commissioned by | 4 | 0.05% |
| location of first performance | 4 | 0.05% |
| sexual orientation | 4 | 0.05% |
| inspired by | 4 | 0.05% |
| adjacent station | 4 | 0.05% |
| airline hub | 4 | 0.05% |
| airline alliance | 4 | 0.05% |
| allied_with | 4 | 0.05% |
| start point | 4 | 0.05% |
| dedicated to | 4 | 0.05% |
| primary destinations | 4 | 0.05% |
| ammunition | 4 | 0.05% |
| carries | 4 | 0.05% |
| taxon rank | 4 | 0.05% |
| takes place in fictional universe | 4 | 0.05% |
| league system | 4 | 0.05% |
| participant of | 4 | 0.05% |
| depicts | 4 | 0.05% |
| objective of project or action | 4 | 0.05% |
| title of chess person | 4 | 0.05% |
| maintained by | 4 | 0.05% |
| has cause | 4 | 0.05% |
| programming paradigm | 4 | 0.05% |
| patron saint | 3 | 0.04% |
| sponsor | 3 | 0.04% |
| collection | 3 | 0.04% |
| linguistic typology | 3 | 0.04% |
| measurement scale | 3 | 0.04% |
| cell component | 3 | 0.04% |
| designed to carry | 3 | 0.04% |
| first appearance | 3 | 0.04% |
| given name version for other gender | 3 | 0.04% |
| game mode | 3 | 0.04% |
| connecting service | 3 | 0.04% |
| industry | 3 | 0.04% |
| sidekick of | 3 | 0.04% |
| sponsored_by | 3 | 0.04% |
| copyright holder | 3 | 0.04% |
| structural engineer | 3 | 0.04% |
| religious order | 3 | 0.04% |
| armament | 3 | 0.04% |
| sexually homologous with | 3 | 0.04% |
| commander of (DEPRECATED) | 3 | 0.04% |
| represented by | 3 | 0.04% |
| source of energy | 3 | 0.04% |
| historical region | 3 | 0.04% |
| league level above | 3 | 0.04% |
| official residence | 3 | 0.04% |
| copyright license | 3 | 0.04% |
| game mechanics | 3 | 0.04% |
| twinned administrative body | 3 | 0.04% |
| substitute/deputy/replacement of office/officeholder | 3 | 0.04% |
| opposed_to | 3 | 0.04% |
| co-driver | 3 | 0.04% |
| delta of | 3 | 0.04% |
| located on astronomical body | 3 | 0.04% |
| highest point | 3 | 0.04% |
| vehicle normally used | 3 | 0.04% |
| fastest lap | 3 | 0.04% |
| merged into | 3 | 0.04% |
| location of creation | 3 | 0.04% |
| measured physical quantity | 3 | 0.04% |
| organization directed by the office or position | 3 | 0.04% |
| translator | 3 | 0.04% |
| illustrator | 3 | 0.04% |
| party_to | 3 | 0.04% |
| has fruit type | 3 | 0.04% |
| field of this occupation | 3 | 0.04% |
| health specialty | 3 | 0.04% |
| archives at | 3 | 0.04% |
| unmarried partner | 3 | 0.04% |
| house publication | 3 | 0.04% |
| foundational text | 3 | 0.04% |
| cause | 3 | 0.04% |
| executive body | 2 | 0.03% |
| molecular function | 2 | 0.03% |
| character designer | 2 | 0.03% |
| type foundry | 2 | 0.03% |
| director of photography | 2 | 0.03% |
| measures | 2 | 0.03% |
| recorded at studio or venue | 2 | 0.03% |
| choreographer | 2 | 0.03% |
| possible treatment | 2 | 0.03% |
| lake on watercourse | 2 | 0.03% |
| has boundary | 2 | 0.03% |
| media franchise | 2 | 0.03% |
| public holiday | 2 | 0.03% |
| reservoir created | 2 | 0.03% |
| source of material | 2 | 0.03% |
| worshipped by | 2 | 0.03% |
| political ideology | 2 | 0.03% |
| youth wing | 2 | 0.03% |
| screenwriter | 2 | 0.03% |
| lake outflow | 2 | 0.03% |
| business division | 2 | 0.03% |
| successful candidate | 2 | 0.03% |
| home world | 2 | 0.03% |
| tracklist | 2 | 0.03% |
| host | 2 | 0.03% |
| site of astronomical discovery | 2 | 0.03% |
| cause of destruction | 2 | 0.03% |
| set in period | 2 | 0.03% |
| destination point | 2 | 0.03% |
| day in year for periodic occurrence | 2 | 0.03% |
| director of archaeological fieldwork | 2 | 0.03% |
| afflicts | 2 | 0.03% |
| killed by | 2 | 0.03% |
| software engine | 2 | 0.03% |
| enclave within | 2 | 0.03% |
| contributor to the creative work or subject | 2 | 0.03% |
| director / manager | 2 | 0.03% |
| points classification | 2 | 0.03% |
| treats | 2 | 0.03% |
| made from material | 2 | 0.03% |
| coach of sports team | 2 | 0.03% |
| official language | 2 | 0.03% |
| emulates | 2 | 0.03% |
| main regulatory text | 2 | 0.03% |
| family name identical to this given name | 2 | 0.03% |
| flag | 2 | 0.03% |
| basic form of government | 2 | 0.03% |
| described by source | 2 | 0.03% |
| produced by | 2 | 0.03% |
| narrator | 2 | 0.03% |
| instrumentation | 2 | 0.03% |
| successor_of | 2 | 0.03% |
| tributary | 2 | 0.03% |
| board member | 2 | 0.03% |
| member_state_of | 2 | 0.03% |
| adjacent_to | 2 | 0.03% |
| league level below | 2 | 0.03% |
| season of club or team | 2 | 0.03% |
| contains settlement | 2 | 0.03% |
| stepparent | 2 | 0.03% |
| capital_of | 2 | 0.03% |
| has anatomical branch | 2 | 0.03% |
| commemorates | 2 | 0.03% |
| color | 2 | 0.03% |
| cuisine | 2 | 0.03% |
| appears in the form of | 2 | 0.03% |
| appointed by | 2 | 0.03% |
| parallel street | 2 | 0.03% |
| writing system | 2 | 0.03% |
| track gauge | 2 | 0.03% |
| head coach | 2 | 0.03% |
| original language of film or TV show | 2 | 0.03% |
| central bank | 2 | 0.03% |
| genic_interaction | 2 | 0.03% |
| depends_on | 2 | 0.03% |
| captain | 2 | 0.03% |
| business model | 2 | 0.03% |
| subdivision of this unit | 2 | 0.03% |
| drainage basin | 2 | 0.03% |
| top-level Internet domain | 2 | 0.03% |
| inflows | 2 | 0.03% |
| Gram staining | 2 | 0.03% |
| soundtrack release | 1 | 0.01% |
| Christian liturgical rite | 1 | 0.01% |
| geomorphological unit | 1 | 0.01% |
| has seal, badge, or sigil | 1 | 0.01% |
| tonality | 1 | 0.01% |
| official religion | 1 | 0.01% |
| modified version of | 1 | 0.01% |
| manifestation of | 1 | 0.01% |
| product certification | 1 | 0.01% |
| anatomical branch of | 1 | 0.01% |
| political alignment | 1 | 0.01% |
| points/goal scored by | 1 | 0.01% |
| dual to | 1 | 0.01% |
| sued_by | 1 | 0.01% |
| used by | 1 | 0.01% |
| set during recurring event | 1 | 0.01% |
| main food source | 1 | 0.01% |
| type of variable star | 1 | 0.01% |
| discovery method | 1 | 0.01% |
| gens | 1 | 0.01% |
| office held by head of state | 1 | 0.01% |
| order of battle | 1 | 0.01% |
| domain of saint or deity | 1 | 0.01% |
| solved by | 1 | 0.01% |
| record or record progression | 1 | 0.01% |
| GUI toolkit or framework | 1 | 0.01% |
| coat of arms | 1 | 0.01% |
| sex or gender | 1 | 0.01% |
| next crossing upstream | 1 | 0.01% |
| charted in | 1 | 0.01% |
| animal breed | 1 | 0.01% |
| predecessor_of | 1 | 0.01% |
| antagonist muscle | 1 | 0.01% |
| prerequisite | 1 | 0.01% |
| theme music | 1 | 0.01% |
| prime factor | 1 | 0.01% |
| implements | 1 | 0.01% |
| final event | 1 | 0.01% |
| consecrator | 1 | 0.01% |
| office held by head of the organization | 1 | 0.01% |
| undercarriage | 1 | 0.01% |
| has goal | 1 | 0.01% |
| mountains classification | 1 | 0.01% |
| set in environment | 1 | 0.01% |
| natural product of taxon | 1 | 0.01% |
| anthem | 1 | 0.01% |
| office contested | 1 | 0.01% |
| home port | 1 | 0.01% |
| scheduled service destination | 1 | 0.01% |
| officially opened by | 1 | 0.01% |
| advertises | 1 | 0.01% |
| coolant | 1 | 0.01% |
| defendant | 1 | 0.01% |
| performed at | 1 | 0.01% |
| from narrative universe | 1 | 0.01% |
| instruction set | 1 | 0.01% |
| develops from | 1 | 0.01% |
| has grammatical case | 1 | 0.01% |
| leader | 1 | 0.01% |
| encodes | 1 | 0.01% |
| approved by | 1 | 0.01% |
| repealed by | 1 | 0.01% |
| drafted by | 1 | 0.01% |
| era name | 1 | 0.01% |
| has pet | 1 | 0.01% |
| parent peak | 1 | 0.01% |
| Cotreatment | 1 | 0.01% |
| powered by | 1 | 0.01% |
| state of conservation | 1 | 0.01% |
| temporal range end | 1 | 0.01% |
| penalty | 1 | 0.01% |
| language regulatory body | 1 | 0.01% |
| causes | 1 | 0.01% |
| venous drainage | 1 | 0.01% |
| feast day | 1 | 0.01% |
| research site | 1 | 0.01% |
| central bank/issuer | 1 | 0.01% |
| cover art by | 1 | 0.01% |
| shape | 1 | 0.01% |
| cites work | 1 | 0.01% |
| Bind | 1 | 0.01% |
| engine configuration | 1 | 0.01% |
| avionics | 1 | 0.01% |
| risk factor | 1 | 0.01% |
| collection creator | 1 | 0.01% |
| series spin-off | 1 | 0.01% |
| innervated by | 1 | 0.01% |
| via | 1 | 0.01% |
| members have occupation | 1 | 0.01% |
| interested in | 1 | 0.01% |
| court | 1 | 0.01% |
| minor planet group | 1 | 0.01% |
| amended by | 1 | 0.01% |
| music created for | 1 | 0.01% |
| structure replaces | 1 | 0.01% |
| demographics of topic | 1 | 0.01% |
| season | 1 | 0.01% |
| military casualty classification | 1 | 0.01% |
| iconographic symbol | 1 | 0.01% |
| temporal range start | 1 | 0.01% |
| sold_to | 1 | 0.01% |
| composed_by | 1 | 0.01% |
| input method | 1 | 0.01% |
| plot expanded in | 1 | 0.01% |
| suggests the existence of | 1 | 0.01% |
| overlies | 1 | 0.01% |
| depends on software | 1 | 0.01% |
| derivative work | 1 | 0.01% |
| writable file format | 1 | 0.01% |
| fandom | 1 | 0.01% |
| parliamentary term | 1 | 0.01% |
| rector | 1 | 0.01% |
| complies with | 1 | 0.01% |
| kit supplier | 1 | 0.01% |
| worn by | 1 | 0.01% |
| connector | 1 | 0.01% |
| superhuman feature or ability | 1 | 0.01% |
| polymer of | 1 | 0.01% |
| valid in period | 1 | 0.01% |
| origin of the watercourse | 1 | 0.01% |
| taxon synonym | 1 | 0.01% |
| species kept | 1 | 0.01% |
| voice actor | 1 | 0.01% |
| underlies | 1 | 0.01% |
| exhibition history | 1 | 0.01% |
| convicted_of | 1 | 0.01% |
| majority opinion by | 1 | 0.01% |
| CPU | 1 | 0.01% |
| has facet polytope | 1 | 0.01% |
| charge | 1 | 0.01% |
| contains | 1 | 0.01% |
| microstructure | 1 | 0.01% |
| copyright status | 1 | 0.01% |
| trophy awarded | 1 | 0.01% |
| newspaper format | 1 | 0.01% |
| editor-in-chief | 1 | 0.01% |

### Classifications

**train, actionability.** Majority class 87.7% of 383 answers. The marginal is `true_label`, not the menu.

#### train actionability

| label | count | share |
| --- | --- | --- |
| informational | 336 | 87.73% |
| advisory | 27 | 7.05% |
| requires action | 20 | 5.22% |

**train, audience.** Majority class 78.5% of 469 answers. The marginal is `true_label`, not the menu.

#### train audience

| label | count | share |
| --- | --- | --- |
| general public | 368 | 78.46% |
| expert | 55 | 11.73% |
| regulatory | 33 | 7.04% |
| internal | 13 | 2.77% |

**train, biotech_event.** Majority class 27.6% of 268 answers. The marginal is `true_label`, not the menu.

#### train biotech_event

| label | count | share |
| --- | --- | --- |
| executive statement | 74 | 27.61% |
| company description | 45 | 16.79% |
| alliance & partnership | 23 | 8.58% |
| other | 19 | 7.09% |
| product launching & presentation | 18 | 6.72% |
| new initiatives or programs | 9 | 3.36% |
| expanding geography | 8 | 2.99% |
| m&a | 7 | 2.61% |
| regulatory approval | 6 | 2.24% |
| support & philanthropy | 6 | 2.24% |
| funding round | 6 | 2.24% |
| article publication | 6 | 2.24% |
| product updates | 5 | 1.87% |
| expanding industry | 5 | 1.87% |
| hiring | 5 | 1.87% |
| executive appointment | 5 | 1.87% |
| ipo exit | 4 | 1.49% |
| investment in public company | 4 | 1.49% |
| service & product providing | 4 | 1.49% |
| participation in an event | 3 | 1.12% |
| event organization | 2 | 0.75% |
| foundation | 1 | 0.37% |
| department establishment | 1 | 0.37% |
| clinical trial sponsorship | 1 | 0.37% |
| new initiatives & programs | 1 | 0.37% |

**train, certainty.** Majority class 97.1% of 377 answers. The marginal is `true_label`, not the menu.

#### train certainty

| label | count | share |
| --- | --- | --- |
| asserted | 366 | 97.08% |
| hedged | 10 | 2.65% |
| speculative | 1 | 0.27% |

**train, chfinann_event.** Majority class 47.9% of 1,644 answers. The marginal is `true_label`, not the menu.

#### train chfinann_event

| label | count | share |
| --- | --- | --- |
| EquityPledge | 787 | 47.87% |
| EquityOverweight | 353 | 21.47% |
| EquityUnderweight | 347 | 21.11% |
| EquityRepurchase | 126 | 7.66% |
| EquityFreeze | 31 | 1.89% |

**train, docee_event.** Majority class 6.2% of 4,621 answers. The marginal is `true_label`, not the menu.

#### train docee_event

| label | count | share |
| --- | --- | --- |
| Sports Competition | 286 | 6.19% |
| none | 251 | 5.43% |
| Government Policy Changes | 232 | 5.02% |
| Fire | 225 | 4.87% |
| CommitCrime - Investigate | 195 | 4.22% |
| Organization Fine | 192 | 4.15% |
| CommitCrime - Arrest | 187 | 4.05% |
| Armed Conflict | 175 | 3.79% |
| Appoint_Inauguration | 138 | 2.99% |
| Protest | 132 | 2.86% |
| Resignation_Dismissal | 129 | 2.79% |
| Air Crash | 129 | 2.79% |
| Famous Person - Death | 123 | 2.66% |
| Election | 118 | 2.55% |
| New Achievements in Aerospace | 106 | 2.29% |
| Road Crash | 97 | 2.10% |
| Earthquakes | 94 | 2.03% |
| Riot | 93 | 2.01% |
| Famous Person - Give a Speech | 89 | 1.93% |
| Break Historical Records | 87 | 1.88% |
| Diplomatic Talks | 87 | 1.88% |
| Disease Outbreaks | 86 | 1.86% |
| CommitCrime - Accuse | 80 | 1.73% |
| Sign Agreement | 73 | 1.58% |
| CommitCrime - Sentence | 73 | 1.58% |
| Awards Ceremony | 72 | 1.56% |
| Mass Poisoning | 67 | 1.45% |
| Floods | 57 | 1.23% |
| Military Excercise | 54 | 1.17% |
| Storm | 53 | 1.15% |
| Organization Established | 51 | 1.10% |
| Diplomatic Visit | 49 | 1.06% |
| New Wonders in Nature | 45 | 0.97% |
| Shipwreck | 45 | 0.97% |
| Strike | 40 | 0.87% |
| Environment Pollution | 38 | 0.82% |
| Financial Crisis | 35 | 0.76% |
| Withdraw from an Orgnization | 35 | 0.76% |
| New Archeological Discoveries | 33 | 0.71% |
| Bank Robbery | 33 | 0.71% |
| Train Collisions | 31 | 0.67% |
| Famous Person - Divorce | 31 | 0.67% |
| Volcano Eruption | 30 | 0.65% |
| Financial Aid | 29 | 0.63% |
| Famous Person - Recovered | 29 | 0.63% |
| Mudslides | 27 | 0.58% |
| Famous Person - Sick | 26 | 0.56% |
| Mine Collapses | 25 | 0.54% |
| CommitCrime - Release | 25 | 0.54% |
| Organization Merge | 22 | 0.48% |
| Organization Closed | 21 | 0.45% |
| Droughts | 21 | 0.45% |
| Regime Change | 20 | 0.43% |
| Famous Person - Marriage | 19 | 0.41% |
| Gas Explosion | 19 | 0.41% |
| Famine | 18 | 0.39% |
| Insect Disaster | 13 | 0.28% |
| Tsunamis | 11 | 0.24% |
| Join in an Orgnization | 11 | 0.24% |
| Tear Up Agreement | 9 | 0.19% |

**train, docfee_event.** Majority class 31.2% of 1,070 answers. The marginal is `true_label`, not the menu.

#### train docfee_event

| label | count | share |
| --- | --- | --- |
| EquityUnderweight | 334 | 31.21% |
| BankruptcyLiquidation | 185 | 17.29% |
| EquityPledge | 174 | 16.26% |
| MajorAssetLoss | 146 | 13.64% |
| EquityFreeze | 79 | 7.38% |
| EquityOverweight | 59 | 5.51% |
| MajorSafetyAccident | 35 | 3.27% |
| MajorExternalPayout | 29 | 2.71% |
| ExecutiveDeath | 29 | 2.71% |

**train, document_genre.** Majority class 76.0% of 367 answers. The marginal is `true_label`, not the menu.

#### train document_genre

| label | count | share |
| --- | --- | --- |
| news report | 279 | 76.02% |
| announcement | 43 | 11.72% |
| opinion | 22 | 5.99% |
| review | 8 | 2.18% |
| correspondence | 6 | 1.63% |
| legal filing | 3 | 0.82% |
| technical documentation | 3 | 0.82% |
| advertisement | 3 | 0.82% |

**train, formality.** Majority class 82.2% of 456 answers. The marginal is `true_label`, not the menu.

#### train formality

| label | count | share |
| --- | --- | --- |
| formal | 375 | 82.24% |
| informal | 81 | 17.76% |

**train, language_register.** Majority class 80.8% of 381 answers. The marginal is `true_label`, not the menu.

#### train language_register

| label | count | share |
| --- | --- | --- |
| plain | 308 | 80.84% |
| technical | 57 | 14.96% |
| colloquial | 10 | 2.62% |
| legal | 6 | 1.57% |

**train, risk_level.** Majority class 42.2% of 334 answers. The marginal is `true_label`, not the menu.

#### train risk_level

| label | count | share |
| --- | --- | --- |
| none | 141 | 42.22% |
| high | 72 | 21.56% |
| low | 69 | 20.66% |
| moderate | 52 | 15.57% |

**train, sentiment.** Majority class 38.7% of 450 answers. The marginal is `true_label`, not the menu.

#### train sentiment

| label | count | share |
| --- | --- | --- |
| positive | 174 | 38.67% |
| neutral | 147 | 32.67% |
| negative | 129 | 28.67% |

**train, subjectivity.** Majority class 88.2% of 363 answers. The marginal is `true_label`, not the menu.

#### train subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 320 | 88.15% |
| subjective | 43 | 11.85% |

**train, temporal_orientation.** Majority class 69.7% of 379 answers. The marginal is `true_label`, not the menu.

#### train temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 264 | 69.66% |
| present | 73 | 19.26% |
| future | 36 | 9.50% |
| timeless | 6 | 1.58% |

**train, topic.** Majority class 16.5% of 847 answers. The marginal is `true_label`, not the menu.

#### train topic

| label | count | share |
| --- | --- | --- |
| business | 140 | 16.53% |
| politics | 106 | 12.51% |
| sports | 102 | 12.04% |
| legal | 84 | 9.92% |
| world | 78 | 9.21% |
| entertainment | 65 | 7.67% |
| finance | 52 | 6.14% |
| technology | 50 | 5.90% |
| health | 40 | 4.72% |
| environment | 35 | 4.13% |
| education | 34 | 4.01% |
| science | 30 | 3.54% |
| travel | 9 | 1.06% |
| defense | 9 | 1.06% |
| energy | 7 | 0.83% |
| agriculture | 6 | 0.71% |

**train, urgency.** Majority class 76.3% of 338 answers. The marginal is `true_label`, not the menu.

#### train urgency

| label | count | share |
| --- | --- | --- |
| routine | 258 | 76.33% |
| elevated | 58 | 17.16% |
| critical | 18 | 5.33% |
| urgent | 4 | 1.18% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | condition | 5 | 100.00% |
| clinical_finding | patient_group | 5 | 100.00% |
| clinical_finding | treatment | 5 | 100.00% |
| clinical_finding | outcome | 4 | 80.00% |
| clinical_finding | record_metadata | natural/condition ×5 |  |
| company_profile | name | 17 | 100.00% |
| company_profile | headquarters | 13 | 76.47% |
| company_profile | size | 12 | 70.59% |
| company_profile | industry | 11 | 64.71% |
| company_profile | founded | 5 | 29.41% |
| company_profile | record_metadata | natural/name ×17 |  |
| event_listing | name | 18 | 100.00% |
| event_listing | date | 16 | 88.89% |
| event_listing | venue | 14 | 77.78% |
| event_listing | organizer | 12 | 66.67% |
| event_listing | price | 3 | 16.67% |
| event_listing | record_metadata | natural/name ×18 |  |
| financial_report | organization | 4 | 100.00% |
| financial_report | period | 4 | 100.00% |
| financial_report | revenue | 2 | 50.00% |
| financial_report | net_income | 2 | 50.00% |
| financial_report | record_metadata | natural/organization ×4 |  |
| incident_report | severity | 22 | 100.00% |
| incident_report | date | 21 | 95.45% |
| incident_report | location | 18 | 81.82% |
| incident_report | incident_type | 12 | 54.55% |
| incident_report | casualties | 10 | 45.45% |
| incident_report | record_metadata | natural/location ×15, natural/incident_type ×5, natural/date ×2 |  |
| legal_case | defendant | 9 | 100.00% |
| legal_case | status | 9 | 100.00% |
| legal_case | court | 8 | 88.89% |
| legal_case | case_name | 7 | 77.78% |
| legal_case | plaintiff | 4 | 44.44% |
| legal_case | record_metadata | natural/case_name ×7, natural/court ×2 |  |
| person_profile | name | 109 | 100.00% |
| person_profile | role | 101 | 92.66% |
| person_profile | location | 96 | 88.07% |
| person_profile | employer | 95 | 87.16% |
| person_profile | record_metadata | natural/location ×69, natural/name ×20 |  |
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
| record | status | 74 | 15.29% |
| record | location | 52 | 10.74% |
| record | name | 35 | 7.23% |
| record | address | 29 | 5.99% |
| record | title | 21 | 4.34% |
| record | description | 14 | 2.89% |
| record | state | 13 | 2.69% |
| record | city | 13 | 2.69% |
| record | patient_id | 13 | 2.69% |
| record | age | 12 | 2.48% |
| record | price | 12 | 2.48% |
| record | date | 11 | 2.27% |
| record | patient_name | 10 | 2.07% |
| record | amount | 9 | 1.86% |
| record | rating | 8 | 1.65% |
| record | author | 8 | 1.65% |
| record | year | 7 | 1.45% |
| record | company | 7 | 1.45% |
| record | head | 7 | 1.45% |
| record | tail | 7 | 1.45% |
| record | policy_id | 7 | 1.45% |
| record | ticket_id | 7 | 1.45% |
| record | designation | 6 | 1.24% |
| record | funding_source | 6 | 1.24% |
| record | venue | 6 | 1.24% |
| record | time | 5 | 1.03% |
| record | product_id | 5 | 1.03% |
| record | policyholder | 5 | 1.03% |
| record | reviewer | 5 | 1.03% |
| record | event_id | 5 | 1.03% |
| record | claim_id | 5 | 1.03% |
| record | region | 5 | 1.03% |
| record | report_id | 4 | 0.83% |
| record | policy_number | 4 | 0.83% |
| record | origin | 4 | 0.83% |
| record | physical_characteristics.observations.0.telescope.type | 4 | 0.83% |
| record | physical_characteristics.observations.0.telescope.name | 4 | 0.83% |
| record | discovery.observatory.name | 4 | 0.83% |
| record | discovery.observatory.location.country | 4 | 0.83% |
| record | employee_id | 4 | 0.83% |
| record | shares | 4 | 0.83% |
| record | reason | 4 | 0.83% |
| record | acquirer | 4 | 0.83% |
| record | home_team | 4 | 0.83% |
| record | review_id | 4 | 0.83% |
| record | price: Selling price of the product in dollars | 4 | 0.83% |
| record | founding_year | 4 | 0.83% |
| record | outcome | 4 | 0.83% |
| record | value | 4 | 0.83% |
| record | device | 4 | 0.83% |
| record | full_name | 4 | 0.83% |
| record | discovery.date | 4 | 0.83% |
| record | physical_characteristics.observations.0.publication_year | 4 | 0.83% |
| record | symptoms | 4 | 0.83% |
| record | risk_score | 3 | 0.62% |
| record | locations | 3 | 0.62% |
| record | credits | 3 | 0.62% |
| record | room | 3 | 0.62% |
| record | department | 3 | 0.62% |
| record | name: Full name of the patient | 3 | 0.62% |
| record | destination | 3 | 0.62% |
| record | delay_reason | 3 | 0.62% |
| record | sku | 3 | 0.62% |
| record | likes | 3 | 0.62% |
| record | creator | 3 | 0.62% |
| record | keywords | 3 | 0.62% |
| record | amount: Monetary value of the transaction | 3 | 0.62% |
| record | event_type: Category of the event | 3 | 0.62% |
| record | tags | 3 | 0.62% |
| record | role | 3 | 0.62% |
| record | reviewer_name | 3 | 0.62% |
| record | cause | 3 | 0.62% |
| record | country | 3 | 0.62% |
| record | provider | 3 | 0.62% |
| record | upvotes | 3 | 0.62% |
| record | market_cap | 3 | 0.62% |
| record | call_id | 3 | 0.62% |
| record | dosage | 3 | 0.62% |
| record | manufacturer | 3 | 0.62% |
| record | clause_id | 3 | 0.62% |
| record | score | 3 | 0.62% |
| record | opponent | 3 | 0.62% |
| record | customer_name | 3 | 0.62% |
| record | features | 3 | 0.62% |
| record | away_team | 3 | 0.62% |
| record | quantity | 3 | 0.62% |
| record | disclaimer | 3 | 0.62% |
| record | district | 3 | 0.62% |
| record | organizations | 3 | 0.62% |
| record | brand | 3 | 0.62% |
| record | topic | 3 | 0.62% |
| record | concepts | 2 | 0.41% |
| record | founder_name: Full name of the founder as stated in passage | 2 | 0.41% |
| record | motivation: Personal reason driving the company's founding | 2 | 0.41% |
| record | course_id | 2 | 0.41% |
| record | visit_type | 2 | 0.41% |
| record | treatment | 2 | 0.41% |
| record | medications | 2 | 0.41% |
| record | insurance | 2 | 0.41% |
| record | discovery_year | 2 | 0.41% |
| record | delay_minutes | 2 | 0.41% |
| record | passenger_name | 2 | 0.41% |
| record | passenger_email | 2 | 0.41% |
| record | passenger_phone | 2 | 0.41% |
| record | id | 2 | 0.41% |
| record | sponsorship | 2 | 0.41% |
| record | orbital_characteristics.perihelion.unit | 2 | 0.41% |
| record | date: Date when the transaction occurred | 2 | 0.41% |
| record | scientific_name | 2 | 0.41% |
| record | cost | 2 | 0.41% |
| record | activities | 2 | 0.41% |
| record | comments | 2 | 0.41% |
| record | confirmation_code | 2 | 0.41% |
| record | manager | 2 | 0.41% |
| record | place.location.country | 2 | 0.41% |
| record | place.location.county | 2 | 0.41% |
| record | influences | 2 | 0.41% |
| record | structures | 2 | 0.41% |
| record | historical_periods | 2 | 0.41% |
| record | event_id: Unique identifier for each event | 2 | 0.41% |
| record | location: Physical address or venue of the event | 2 | 0.41% |
| record | subject | 2 | 0.41% |
| record | vaccine_type | 2 | 0.41% |
| record | campaign_id | 2 | 0.41% |
| record | acquired | 2 | 0.41% |
| record | acquisition_amount | 2 | 0.41% |
| record | valuation_method | 2 | 0.41% |
| record | conference_id | 2 | 0.41% |
| record | risk_rating | 2 | 0.41% |
| record | person.name | 2 | 0.41% |
| record | startup_name | 2 | 0.41% |
| record | organizer | 2 | 0.41% |
| record | assists | 2 | 0.41% |
| record | company_name | 2 | 0.41% |
| record | mrn | 2 | 0.41% |
| record | gender | 2 | 0.41% |
| record | zip_code | 2 | 0.41% |
| record | recommendation | 2 | 0.41% |
| record | location: Headquarters city and state or country | 2 | 0.41% |
| record | focus: Primary technology or business area of the company | 2 | 0.41% |
| record | amount: Total capital raised in millions | 2 | 0.41% |
| record | post_valuation: Post-money valuation in millions | 2 | 0.41% |
| record | location: Physical address of the product launch | 2 | 0.41% |
| record | product_code: Internal code assigned to the product | 2 | 0.41% |
| record | caveat.example.components.0.related_characters.1.meaning | 2 | 0.41% |
| record | caveat.example.components.0.related_characters.1.pronunciation | 2 | 0.41% |
| record | caveat.example.components.0.related_characters.2.meaning | 2 | 0.41% |
| record | caveat.example.components.0.meaning | 2 | 0.41% |
| record | caveat.example.components.0.position | 2 | 0.41% |
| record | caveat.example.components.1.position | 2 | 0.41% |
| record | comparison.alphabetic.learning_curve | 2 | 0.41% |
| record | caveat.example.components.0.related_characters.2.pronunciation | 2 | 0.41% |
| record | caveat.example.components.1.meaning | 2 | 0.41% |
| record | caveat.example.meaning | 2 | 0.41% |
| record | terminology.ideogram.english | 2 | 0.41% |
| record | comparison.alphabetic.pronunciation | 2 | 0.41% |
| record | caveat.example.components.0.related_characters.0.pronunciation | 2 | 0.41% |
| record | terminology.ideographic_script.english | 2 | 0.41% |
| record | comparison.ideographic.example.meaning | 2 | 0.41% |
| record | caveat.example.components.0.function | 2 | 0.41% |
| record | caveat.example.components.0.related_characters.1.character | 2 | 0.41% |
| record | caveat.example.components.0.component | 2 | 0.41% |
| record | comparison.ideographic.learning_curve | 2 | 0.41% |
| record | relationships.0.type | 2 | 0.41% |
| record | funding_id: Unique identifier for each funding round | 2 | 0.41% |
| record | roundtype: Type of funding round | 2 | 0.41% |
| record | case_name: Full name of the case in format plaintiff v defendant | 2 | 0.41% |
| record | claim_type: Nature of the legal claim such as breach of contract or noise violation | 2 | 0.41% |
| record | scope: Broad category or focus area of the bill | 2 | 0.41% |
| record | vote_id: Unique identifier for the legislative vote | 2 | 0.41% |
| record | status: Current legislative status of the bill | 2 | 0.41% |
| record | carrier | 2 | 0.41% |
| record | notes | 2 | 0.41% |
| record | vehicle | 2 | 0.41% |
| record | mark_name | 2 | 0.41% |
| record | registration_number | 2 | 0.41% |
| record | bathrooms | 2 | 0.41% |
| record | peculiarity | 2 | 0.41% |
| record | spectral_class | 2 | 0.41% |
| record | research_background.animal_models_used | 2 | 0.41% |
| record | affiliation.institution_name | 2 | 0.41% |
| record | affiliation.department | 2 | 0.41% |
| record | key_challenge | 2 | 0.41% |
| record | description.inflorescence.type | 2 | 0.41% |
| record | description.leaves.terminal_spine.color | 2 | 0.41% |
| record | description.fruit.width_mm.min | 2 | 0.41% |
| record | description.flowering_time | 2 | 0.41% |
| record | description.fruit.shape | 2 | 0.41% |
| record | description.fruit.length_mm.min | 2 | 0.41% |
| record | systematics_and_distribution.location.state | 2 | 0.41% |
| record | description.leaves.margins | 2 | 0.41% |
| record | description.leaves.width_cm.min | 2 | 0.41% |
| record | systematics_and_distribution.location.country | 2 | 0.41% |
| record | systematics_and_distribution.similar_species | 2 | 0.41% |
| record | description.fruit.width_mm.max | 2 | 0.41% |
| record | description.flowers.arrangement | 2 | 0.41% |
| record | description.leaves.surface | 2 | 0.41% |
| record | status: Current product availability status | 2 | 0.41% |
| record | subreddit | 2 | 0.41% |
| record | publication_date | 2 | 0.41% |
| record | vendor_name | 2 | 0.41% |
| record | invoice_date | 2 | 0.41% |
| record | listing_date | 2 | 0.41% |
| record | organizer.name | 2 | 0.41% |
| record | surveyPurpose | 2 | 0.41% |
| record | year: Year of financial data | 2 | 0.41% |
| record | growth_rate: Percentage growth in revenue | 2 | 0.41% |
| record | partner_year: Year of partner's financial data | 2 | 0.41% |
| record | partner_id: Unique identifier of the partner company | 2 | 0.41% |
| record | location: City and state of the company | 2 | 0.41% |
| record | partner_revenue: Partner's revenue in millions | 2 | 0.41% |
| record | company_id: Unique identifier for the company | 2 | 0.41% |
| record | partner_status: Operational status of the partner company | 2 | 0.41% |
| record | video_id | 2 | 0.41% |
| record | procedure | 2 | 0.41% |
| record | integration_plan | 2 | 0.41% |
| record | transaction_id | 2 | 0.41% |
| record | patient_age | 2 | 0.41% |
| record | drug_name | 2 | 0.41% |
| record | route | 2 | 0.41% |
| record | lab_id | 2 | 0.41% |
| record | species.description | 2 | 0.41% |
| record | dispute_status | 2 | 0.41% |
| record | owner | 2 | 0.41% |
| record | owner_location | 2 | 0.41% |
| record | issue | 2 | 0.41% |
| record | ties | 2 | 0.41% |
| record | final_standing | 2 | 0.41% |
| record | season | 2 | 0.41% |
| record | win_percentage | 2 | 0.41% |
| record | coach | 2 | 0.41% |
| record | dates | 2 | 0.41% |
| record | ticker | 2 | 0.41% |
| record | name: Name of the product as listed | 2 | 0.41% |
| record | repair_cost | 2 | 0.41% |
| record | serial_number | 2 | 0.41% |
| record | education | 2 | 0.41% |
| record | early_challenge | 2 | 0.41% |
| record | administration_date | 2 | 0.41% |
| record | location: Geographic headquarters of the company | 2 | 0.41% |
| record | net_profit: Annual net profit in millions of dollars | 2 | 0.41% |
| record | ebitda_margin: Percentage of revenue that represents EBITDA | 2 | 0.41% |
| record | operational_status: Current phase of business operations | 2 | 0.41% |
| record | diameter.unit | 2 | 0.41% |
| record | discovery.discoverer | 2 | 0.41% |
| record | orbital_characteristics.semi_major_axis.unit | 2 | 0.41% |
| record | orbital_characteristics.inclination.unit | 2 | 0.41% |
| record | physical_characteristics.observations.0.telescope | 2 | 0.41% |
| record | source | 2 | 0.41% |
| record | net_profit | 2 | 0.41% |
| record | funding_amount: Total amount raised in millions | 2 | 0.41% |
| record | device_id: Unique identifier assigned to the medical device | 2 | 0.41% |
| record | agency | 2 | 0.41% |
| record | problem_description | 2 | 0.41% |
| record | home_team: Name of the team playing at home | 2 | 0.41% |
| record | frequency | 2 | 0.41% |
| record | issue_date | 2 | 0.41% |
| record | asteroid.name | 2 | 0.41% |
| record | asteroid.type | 2 | 0.41% |
| record | sentiment | 2 | 0.41% |
| record | severity | 2 | 0.41% |
| record | candidate_name | 2 | 0.41% |
| record | region.agriculture | 2 | 0.41% |
| record | region.description | 2 | 0.41% |
| record | region.language | 2 | 0.41% |
| record | region.divisions.1.name | 2 | 0.41% |
| record | region.name | 2 | 0.41% |
| record | person | 2 | 0.41% |
| record | genus.synonyms | 2 | 0.41% |
| record | genus.family | 2 | 0.41% |
| record | genus.name_latin | 2 | 0.41% |
| record | genus.subspecies.0.authority | 2 | 0.41% |
| record | review_period | 2 | 0.41% |
| record | performance_score | 2 | 0.41% |
| record | entity_name | 2 | 0.41% |
| record | filing_date | 2 | 0.41% |
| record | underwriter | 2 | 0.41% |
| record | content | 2 | 0.41% |
| record | affiliation.0.type | 2 | 0.41% |
| record | organization | 2 | 0.41% |
| record | jurisdiction | 2 | 0.41% |
| record | revenue | 2 | 0.41% |
| record | priority: Severity level assigned to the ticket | 2 | 0.41% |
| record | monthly_premium | 2 | 0.41% |
| record | location.city | 2 | 0.41% |
| record | zip | 2 | 0.41% |
| record | location: City and state where the company is headquartered | 2 | 0.41% |
| record | reportTitle | 2 | 0.41% |
| record | capacity | 2 | 0.41% |
| record | heart_rate | 2 | 0.41% |
| record | imaging_date | 2 | 0.41% |
| record | findings | 2 | 0.41% |
| record | asteroid_number | 2 | 0.41% |
| record | asteroid_type | 2 | 0.41% |
| record | punct | 2 | 0.41% |
| record | nsubj | 2 | 0.41% |
| record | ROOT | 2 | 0.41% |
| record | advmod | 2 | 0.41% |
| record | prep | 2 | 0.41% |
| record | office_location | 2 | 0.41% |
| record | claim_id: Unique identifier for the insurance claim | 2 | 0.41% |
| record | term_length | 2 | 0.41% |
| record | headquarters | 2 | 0.41% |
| record | village_name | 2 | 0.41% |
| record | program_name | 2 | 0.41% |
| record | duration | 2 | 0.41% |
| record | hotel_name | 2 | 0.41% |
| record | political_context | 1 | 0.21% |
| record | priority: Urgency level of the ticket | 1 | 0.21% |
| record | resolution_time: Estimated or actual time to resolve the issue | 1 | 0.21% |
| record | description: Detailed account of the issue reported | 1 | 0.21% |
| record | department: Department the submitter belongs to | 1 | 0.21% |
| record | submitter: Name of the employee who submitted the ticket | 1 | 0.21% |
| record | municipality.name | 1 | 0.21% |
| record | underwriting_notes | 1 | 0.21% |
| record | course | 1 | 0.21% |
| record | primates | 1 | 0.21% |
| record | company_name: Name of the startup founded by the individual | 1 | 0.21% |
| record | focus_area: Primary domain or mission of the company | 1 | 0.21% |
| record | location: City and state where the company was founded | 1 | 0.21% |
| record | question.text | 1 | 0.21% |
| record | question.keywords | 1 | 0.21% |
| record | question.legal_implications | 1 | 0.21% |
| record | details.date_posted | 1 | 0.21% |
| record | details.days_ago | 1 | 0.21% |
| record | publication_ref | 1 | 0.21% |
| record | functional_role | 1 | 0.21% |
| record | institution | 1 | 0.21% |
| record | confidence_score | 1 | 0.21% |
| record | gene_count | 1 | 0.21% |
| record | start_term | 1 | 0.21% |
| record | hall | 1 | 0.21% |
| record | course_code | 1 | 0.21% |
| record | enrollment | 1 | 0.21% |
| record | target_department | 1 | 0.21% |
| record | spoofed_domain | 1 | 0.21% |
| record | instructor | 1 | 0.21% |
| record | days | 1 | 0.21% |
| record | enrollmentstatus | 1 | 0.21% |
| record | encounter_id | 1 | 0.21% |
| record | dedication.name | 1 | 0.21% |
| record | dedication.description | 1 | 0.21% |
| record | startup: Name of the startup receiving investment | 1 | 0.21% |
| record | research_context.primary_technique | 1 | 0.21% |
| record | research_context.application_goal | 1 | 0.21% |
| record | research_context.problem_statement.limitation | 1 | 0.21% |
| record | research_context.problem_statement.incompatible_substrates | 1 | 0.21% |
| record | research_context.problem_statement.affected_research_fields | 1 | 0.21% |
| record | methodology.proposed_solution | 1 | 0.21% |
| record | methodology.experimental_setup.biological_model.cell_line | 1 | 0.21% |
| record | methodology.experimental_setup.biological_model.culture_environment.support_type | 1 | 0.21% |
| record | methodology.experimental_setup.biological_model.culture_environment.material_properties | 1 | 0.21% |
| record | methodology.experimental_setup.reagents.0.labeling | 1 | 0.21% |
| record | methodology.experimental_setup.reagents.0.type | 1 | 0.21% |
| record | methodology.experimental_setup.instrumentation | 1 | 0.21% |
| record | outcomes.comparative_advantage.alternative_to | 1 | 0.21% |
| record | outcomes.comparative_advantage.benefit_type | 1 | 0.21% |
| record | outcomes.impact | 1 | 0.21% |
| record | url_references | 1 | 0.21% |
| record | expert_insights | 1 | 0.21% |
| record | lead_researcher | 1 | 0.21% |
| record | intervention | 1 | 0.21% |
| record | patient_count | 1 | 0.21% |
| record | orbit.orbital_period.value | 1 | 0.21% |
| record | orbit.orbital_type | 1 | 0.21% |
| record | characteristics.diameter.value | 1 | 0.21% |
| record | object_type | 1 | 0.21% |
| record | orbit.semi_major_axis.unit | 1 | 0.21% |
| record | orbit.orbital_period.unit | 1 | 0.21% |
| record | characteristics.diameter.unit | 1 | 0.21% |
| record | date: Date of the patient visit | 1 | 0.21% |
| record | spo2: Oxygen saturation percentage | 1 | 0.21% |
| record | timestamp: Time of measurement in HH:MM:SS format | 1 | 0.21% |
| record | temperature: Body temperature in degrees Fahrenheit | 1 | 0.21% |
| record | blood_pressure: Blood pressure in format systolic/diastolic | 1 | 0.21% |
| record | flight_id | 1 | 0.21% |
| record | departure_original | 1 | 0.21% |
| record | departure_actual | 1 | 0.21% |
| record | passenger_address | 1 | 0.21% |
| record | booth | 1 | 0.21% |
| record | manifest_id | 1 | 0.21% |
| record | sender | 1 | 0.21% |
| record | discovery.project | 1 | 0.21% |
| record | number | 1 | 0.21% |
| record | physical_characteristics.observations.0.telescope.launch_year | 1 | 0.21% |
| record | merchant: Name of the merchant or store involved | 1 | 0.21% |
| record | status: Current status of the transaction | 1 | 0.21% |
| record | product: Description of the product or service purchased | 1 | 0.21% |
| record | storage | 1 | 0.21% |
| record | comment_count: Number of comments on the video | 1 | 0.21% |
| record | video_id: Unique identifier assigned to the video | 1 | 0.21% |
| record | like_count: Number of likes the video has accumulated | 1 | 0.21% |
| record | description.stipules.dimensions | 1 | 0.21% |
| record | description.leaf.texture | 1 | 0.21% |
| record | family | 1 | 0.21% |
| record | location.other_locations | 1 | 0.21% |
| record | location.endemism | 1 | 0.21% |
| record | honorary_name.field_of_study | 1 | 0.21% |
| record | description.seed.color | 1 | 0.21% |
| record | honorary_name.person | 1 | 0.21% |
| record | description.leaf.base | 1 | 0.21% |
| record | description.type | 1 | 0.21% |
| record | description.leaf.dimensions | 1 | 0.21% |
| record | description.carpels | 1 | 0.21% |
| record | conservation_status.status | 1 | 0.21% |
| record | description.fruit.dimensions | 1 | 0.21% |
| record | description.floral_buds.dimensions | 1 | 0.21% |
| record | habitat.associated_species | 1 | 0.21% |
| record | description.leaf.apex | 1 | 0.21% |
| record | description.seed.dimensions | 1 | 0.21% |
| record | honorary_name.profession | 1 | 0.21% |
| record | program_code | 1 | 0.21% |
| record | primary_outcome | 1 | 0.21% |
| record | study_start | 1 | 0.21% |
| record | affiliations | 1 | 0.21% |
| record | sample_size | 1 | 0.21% |
| record | study_id | 1 | 0.21% |
| record | key_result | 1 | 0.21% |
| record | status: Current processing or acceptance status | 1 | 0.21% |
| record | author: Name of the submitting researcher or academic | 1 | 0.21% |
| record | institution: University or institution where author is affiliated | 1 | 0.21% |
| record | subscription_id | 1 | 0.21% |
| record | subscription_type | 1 | 0.21% |
| record | billing_amount | 1 | 0.21% |
| record | billing_frequency | 1 | 0.21% |
| record | email_address | 1 | 0.21% |
| record | delivery_address | 1 | 0.21% |
| record | historical_context | 1 | 0.21% |
| record | enrollment_date | 1 | 0.21% |
| record | overall_score | 1 | 0.21% |
| record | time_management | 1 | 0.21% |
| record | review_status | 1 | 0.21% |
| record | cost_savings | 1 | 0.21% |
| record | collaboration | 1 | 0.21% |
| record | session_id | 1 | 0.21% |
| record | speaker | 1 | 0.21% |
| record | key_achievements | 1 | 0.21% |
| record | place.type | 1 | 0.21% |
| record | related_articles | 1 | 0.21% |
| record | closing_price | 1 | 0.21% |
| record | salary_range: Expected or offered salary range for the role | 1 | 0.21% |
| record | target_role: Job title or position being targeted | 1 | 0.21% |
| record | message_id: Unique identifier for the outreach message | 1 | 0.21% |
| record | target_company: Company where the role is located | 1 | 0.21% |
| record | report_id: Unique identifier assigned to the phishing report | 1 | 0.21% |
| record | status: Current status of the report | 1 | 0.21% |
| record | sender_domain: Domain of the spoofed email address | 1 | 0.21% |
| record | application_id | 1 | 0.21% |
| record | filer | 1 | 0.21% |
| record | technical_focus | 1 | 0.21% |
| record | tenant_name: Name of the leasing company or business | 1 | 0.21% |
| record | lease_id: Unique identifier for each lease agreement | 1 | 0.21% |
| record | name: Name of the person associated with the transaction | 1 | 0.21% |
| record | description: Brief explanation of the transaction purpose | 1 | 0.21% |
| record | status: Current approval status of the transaction | 1 | 0.21% |
| record | organizer: Person or group hosting the event | 1 | 0.21% |
| record | price: Entry cost for attending the event | 1 | 0.21% |
| record | alternative_names | 1 | 0.21% |
| record | publicationDate | 1 | 0.21% |
| record | lot_number | 1 | 0.21% |
| record | tracking_id | 1 | 0.21% |
| record | open_rate | 1 | 0.21% |
| record | line2 | 1 | 0.21% |
| record | github_id | 1 | 0.21% |
| record | project | 1 | 0.21% |
| record | review_score | 1 | 0.21% |
| record | review_type | 1 | 0.21% |
| record | keynote_time | 1 | 0.21% |
| record | keynote_topic | 1 | 0.21% |
| record | years_mentioned | 1 | 0.21% |
| record | additional_investors | 1 | 0.21% |
| record | name: Full name of the player involved in transfer | 1 | 0.21% |
| record | nationality: Country of origin of the player | 1 | 0.21% |
| record | position: Playing position such as midfielder, forward, defender | 1 | 0.21% |
| record | age: Player's age at time of transfer | 1 | 0.21% |
| record | transfer_fee: Amount paid in million euros for the transfer | 1 | 0.21% |
| record | contract_length: Duration of contract in years | 1 | 0.21% |
| record | appearances: Total games played in last season | 1 | 0.21% |
| record | goals: Goals scored in last season | 1 | 0.21% |
| record | status: Current status of transfer | 1 | 0.21% |
| record | club_from: Previous club before transfer | 1 | 0.21% |
| record | club_to: New club after transfer | 1 | 0.21% |
| record | player_id: Unique identifier assigned to the player transfer record | 1 | 0.21% |
| record | rsvp_count | 1 | 0.21% |
| record | player_name | 1 | 0.21% |
| record | steals | 1 | 0.21% |
| record | home_score | 1 | 0.21% |
| record | player_id | 1 | 0.21% |
| record | death.date_julian | 1 | 0.21% |
| record | birth.date_julian | 1 | 0.21% |
| record | death.location | 1 | 0.21% |
| record | name.given_name | 1 | 0.21% |
| record | teacher | 1 | 0.21% |
| record | works.type | 1 | 0.21% |
| record | occupation | 1 | 0.21% |
| record | birth.location | 1 | 0.21% |
| record | name.family_name | 1 | 0.21% |
| record | father.occupation | 1 | 0.21% |
| record | thumbnail | 1 | 0.21% |
| record | development_periods | 1 | 0.21% |
| record | see_also | 1 | 0.21% |
| record | stub_article.call_to_action | 1 | 0.21% |
| record | early_challenges | 1 | 0.21% |
| record | exit_status | 1 | 0.21% |
| record | founder_name | 1 | 0.21% |
| record | founding_date | 1 | 0.21% |
| record | investment_source | 1 | 0.21% |
| record | client_id | 1 | 0.21% |
| record | location: City of origin or primary market | 1 | 0.21% |
| record | features: List of technical capabilities or functions | 1 | 0.21% |
| record | retail_price: Final price at retail in US dollars | 1 | 0.21% |
| record | purpose | 1 | 0.21% |
| record | citations.4.description | 1 | 0.21% |
| record | gymnosperm_diversity.species_examples.0.characteristic | 1 | 0.21% |
| record | citations.2.year | 1 | 0.21% |
| record | citations.0.year | 1 | 0.21% |
| record | gymnosperm_diversity.endemism_ranking | 1 | 0.21% |
| record | gymnosperm_diversity.species_examples.2.characteristic | 1 | 0.21% |
| record | taxonomic_reference | 1 | 0.21% |
| record | citations.3.description | 1 | 0.21% |
| record | citations.1.author | 1 | 0.21% |
| record | citations.2.author | 1 | 0.21% |
| record | gymnosperm_diversity.diversity_center | 1 | 0.21% |
| record | gymnosperm_diversity.species_examples.1.characteristic | 1 | 0.21% |
| record | location.region | 1 | 0.21% |
| record | gymnosperm_diversity.species_examples.2.name | 1 | 0.21% |
| record | gymnosperm_diversity.rare_species | 1 | 0.21% |
| record | citations.1.year | 1 | 0.21% |
| record | gymnosperm_diversity.pine_species_count | 1 | 0.21% |
| record | gymnosperm_diversity.species_examples.3.characteristic | 1 | 0.21% |
| record | reviewer_address | 1 | 0.21% |
| record | policy_reference | 1 | 0.21% |
| record | programStructure.dailySchedule.1.activities.2.description | 1 | 0.21% |
| record | programStructure.dailySchedule.1.activities.2.time | 1 | 0.21% |
| record | accommodation.options.0.website | 1 | 0.21% |
| record | accommodation.options.1.address | 1 | 0.21% |
| record | fees.certificationFee | 1 | 0.21% |
| record | fees.deposit | 1 | 0.21% |
| record | programStructure.dailySchedule.0.activities.2.time | 1 | 0.21% |
| record | programStructure.dailySchedule.1.activities.1.time | 1 | 0.21% |
| record | availability.registration | 1 | 0.21% |
| record | accommodation.options.1.website | 1 | 0.21% |
| record | programStructure.materialsProvided | 1 | 0.21% |
| record | accommodation.options.1.distance | 1 | 0.21% |
| record | programStructure.dailySchedule.0.activities.1.description | 1 | 0.21% |
| record | availability.year | 1 | 0.21% |
| record | accommodation.options.1.name | 1 | 0.21% |
| record | programStructure.dailySchedule.0.activities.3.time | 1 | 0.21% |
| record | programStructure.duration | 1 | 0.21% |
| record | offeredBy | 1 | 0.21% |
| record | programStructure.dailySchedule.1.activities.0.description | 1 | 0.21% |
| record | accommodation.responsibility | 1 | 0.21% |
| record | valueAddedBenefits | 1 | 0.21% |
| record | programStructure.dailySchedule.1.activities.0.time | 1 | 0.21% |
| record | programStructure.dailySchedule.1.activities.3.description | 1 | 0.21% |
| record | programStructure.dailySchedule.0.activities.1.time | 1 | 0.21% |
| record | programName | 1 | 0.21% |
| record | imaging_type | 1 | 0.21% |
| record | referring_physician | 1 | 0.21% |
| record | facility_name | 1 | 0.21% |
| record | facility_address | 1 | 0.21% |
| record | study_code | 1 | 0.21% |
| record | finding_description | 1 | 0.21% |
| record | clinical_status | 1 | 0.21% |
| record | study_protocol | 1 | 0.21% |
| record | trend_type | 1 | 0.21% |
| record | repost_count | 1 | 0.21% |
| record | user_handle | 1 | 0.21% |
| record | contact_name | 1 | 0.21% |
| record | invoice_id | 1 | 0.21% |
| record | number_of_offers | 1 | 0.21% |
| record | affected_area | 1 | 0.21% |
| record | beliefs | 1 | 0.21% |
| record | religion | 1 | 0.21% |
| record | scriptures | 1 | 0.21% |
| record | last_name | 1 | 0.21% |
| record | hire_method | 1 | 0.21% |
| record | first_name | 1 | 0.21% |
| record | deal_id | 1 | 0.21% |
| record | valuation_multiple | 1 | 0.21% |
| record | sales_period | 1 | 0.21% |
| record | sales_platform | 1 | 0.21% |
| record | sale_amount | 1 | 0.21% |
| record | handle: Social media handle for the brand | 1 | 0.21% |
| record | target_audience: Demographic group the product targets | 1 | 0.21% |
| record | festival_id | 1 | 0.21% |
| record | artist | 1 | 0.21% |
| record | genre | 1 | 0.21% |
| record | ticket_price | 1 | 0.21% |
| record | age_restriction | 1 | 0.21% |
| record | audience_rating | 1 | 0.21% |
| record | performance_time | 1 | 0.21% |
| record | scholarship_name | 1 | 0.21% |
| record | recipient | 1 | 0.21% |
| record | major | 1 | 0.21% |
| record | scholarship_id | 1 | 0.21% |
| record | disbursement_plan | 1 | 0.21% |
| record | gpa | 1 | 0.21% |
| record | posted_on | 1 | 0.21% |
| record | ship_name | 1 | 0.21% |
| record | history.creation.shipyard.name | 1 | 0.21% |
| record | history.creation.commissioning_date | 1 | 0.21% |
| record | history.creation.laid_down_date | 1 | 0.21% |
| record | history.creation.shipyard.location | 1 | 0.21% |
| record | relationships.0.action | 1 | 0.21% |
| record | brands | 1 | 0.21% |
| record | status: Current status of the funding round | 1 | 0.21% |
| record | company_name: Name of the startup receiving funding | 1 | 0.21% |
| record | building | 1 | 0.21% |
| record | lease_term | 1 | 0.21% |
| record | plaintiff: Name of the individual or entity initiating the lawsuit | 1 | 0.21% |
| record | defendant: Name of the individual or entity being sued | 1 | 0.21% |
| record | status: Current status of the case | 1 | 0.21% |
| record | filing_date: Date the patent was filed | 1 | 0.21% |
| record | inventor: Name of the patent's inventor | 1 | 0.21% |
| record | description: Technical summary of the patented invention | 1 | 0.21% |
| record | release_date | 1 | 0.21% |
| record | title_fr | 1 | 0.21% |
| record | flight_number | 1 | 0.21% |
| record | booking_reference | 1 | 0.21% |
| record | donor_id | 1 | 0.21% |
| record | drug_name: Name of the pharmaceutical compound | 1 | 0.21% |
| record | clinical_indication: Medical condition the drug is used to treat | 1 | 0.21% |
| record | date_verified: Date the drug's information was last verified | 1 | 0.21% |
| record | status: Current regulatory or development status | 1 | 0.21% |
| record | clinical_use: Primary medical application of the drug | 1 | 0.21% |
| record | vin | 1 | 0.21% |
| record | disputingpartya | 1 | 0.21% |
| record | disputingpartyb | 1 | 0.21% |
| record | mls_number | 1 | 0.21% |
| record | agent | 1 | 0.21% |
| record | distance.reference_point | 1 | 0.21% |
| record | battery_life: Battery life in hours as specified | 1 | 0.21% |
| record | availability_date: Date when product becomes available | 1 | 0.21% |
| record | price: Price in dollars without currency symbol | 1 | 0.21% |
| record | color: Color of the product | 1 | 0.21% |
| record | status: Current stock availability status | 1 | 0.21% |
| record | founder | 1 | 0.21% |
| record | description.flowers.color | 1 | 0.21% |
| record | description.seeds.color | 1 | 0.21% |
| record | description.fruit.color | 1 | 0.21% |
| record | description.inflorescence.height_m.max | 1 | 0.21% |
| record | systematics_and_distribution.first_description.author | 1 | 0.21% |
| record | description.flowers.perianth_tube_length_mm.max | 1 | 0.21% |
| record | description.inflorescence.height_m.min | 1 | 0.21% |
| record | description.flowers.length_mm.max | 1 | 0.21% |
| record | systematics_and_distribution.habitat | 1 | 0.21% |
| record | common_names | 1 | 0.21% |
| record | description.leaves.length_cm.max | 1 | 0.21% |
| record | systematics_and_distribution.group | 1 | 0.21% |
| record | systematics_and_distribution.location.elevation_m.min | 1 | 0.21% |
| record | description.leaves.shape | 1 | 0.21% |
| record | account_id | 1 | 0.21% |
| record | region: Geographic region where product is sold | 1 | 0.21% |
| record | sales_volume: Total units sold | 1 | 0.21% |
| record | brand: Manufacturer or brand name | 1 | 0.21% |
| record | description: Brief product feature and function details | 1 | 0.21% |
| record | shipping_method: Delivery method for the product | 1 | 0.21% |
| record | category: General product category such as apparel or electronics | 1 | 0.21% |
| record | post_id | 1 | 0.21% |
| record | implementation_notes | 1 | 0.21% |
| record | guideline_code | 1 | 0.21% |
| record | cost_impact | 1 | 0.21% |
| record | currency | 1 | 0.21% |
| record | contact_person | 1 | 0.21% |
| record | ticket_id: Unique identifier for the return ticket | 1 | 0.21% |
| record | owner: Name of the customer filing the ticket | 1 | 0.21% |
| record | issue: Description of the reported hardware failure | 1 | 0.21% |
| record | location: City and state where the customer resides | 1 | 0.21% |
| record | eps | 1 | 0.21% |
| record | asteroid.discovery.discoverer | 1 | 0.21% |
| record | asteroid.discovery.location | 1 | 0.21% |
| record | property_id | 1 | 0.21% |
| record | intendedAudience | 1 | 0.21% |
| record | accessInstructions | 1 | 0.21% |
| record | author_information.education.0.institution | 1 | 0.21% |
| record | author_information.military_service.branch | 1 | 0.21% |
| record | author_information.military_service.status | 1 | 0.21% |
| record | author_information.occupation | 1 | 0.21% |
| record | author_information.location.city | 1 | 0.21% |
| record | author_information.location.state | 1 | 0.21% |
| record | author_information.family.spouse | 1 | 0.21% |
| record | author_information.family.pets.0.name | 1 | 0.21% |
| record | author_information.family.pets.0.type | 1 | 0.21% |
| record | author_information.interests | 1 | 0.21% |
| record | company_name: Name of the company in the record | 1 | 0.21% |
| record | partner_location: City and state of the partner company | 1 | 0.21% |
| record | candidate_id: Identifier for the candidate receiving the donation | 1 | 0.21% |
| record | donor_type: Type of donor as defined in the disclosure | 1 | 0.21% |
| record | donor_name: Full name of the individual or organization making the donation | 1 | 0.21% |
| record | donation_description: Brief description of the donation's purpose or context | 1 | 0.21% |
| record | force_type | 1 | 0.21% |
| record | material_efficiency | 1 | 0.21% |
| record | type | 1 | 0.21% |
| record | artistic_characteristics | 1 | 0.21% |
| record | themes | 1 | 0.21% |
| record | product.name | 1 | 0.21% |
| record | product.brand | 1 | 0.21% |
| record | product.type | 1 | 0.21% |
| record | product.description | 1 | 0.21% |
| record | product.size | 1 | 0.21% |
| record | product.ingredients | 1 | 0.21% |
| record | review.overall_rating | 1 | 0.21% |
| record | review.pros | 1 | 0.21% |
| record | review.cons | 1 | 0.21% |
| record | review.price | 1 | 0.21% |
| record | review.usage_experience | 1 | 0.21% |
| record | patents | 1 | 0.21% |
| record | value_original | 1 | 0.21% |
| record | regulatory_status | 1 | 0.21% |
| record | earnout | 1 | 0.21% |
| record | customer_contracts | 1 | 0.21% |
| record | patient_gender | 1 | 0.21% |
| record | clinic_name | 1 | 0.21% |
| record | ndc | 1 | 0.21% |
| record | prescription_id | 1 | 0.21% |
| record | test_type | 1 | 0.21% |
| record | definition | 1 | 0.21% |
| record | examples | 1 | 0.21% |
| record | comparison.ideographic.cross_lingual_understanding | 1 | 0.21% |
| record | comparison.ideographic.example.character | 1 | 0.21% |
| record | comparison.ideographic.example.languages.0.language | 1 | 0.21% |
| record | comparison.ideographic.example.languages.0.pronunciation | 1 | 0.21% |
| record | comparison.ideographic.example.languages.1.language | 1 | 0.21% |
| record | comparison.ideographic.example.languages.1.pronunciation | 1 | 0.21% |
| record | comparison.ideographic.historical_understanding | 1 | 0.21% |
| record | caveat.example.character | 1 | 0.21% |
| record | caveat.example.pronunciation | 1 | 0.21% |
| record | caveat.example.components.0.related_characters.0.character | 1 | 0.21% |
| record | caveat.example.components.0.related_characters.0.meaning | 1 | 0.21% |
| record | caveat.example.components.0.related_characters.2.character | 1 | 0.21% |
| record | caveat.example.components.1.function | 1 | 0.21% |
| record | caveat.example.components.1.component | 1 | 0.21% |
| record | species.scientific_name | 1 | 0.21% |
| record | species.genus | 1 | 0.21% |
| record | positive_note | 1 | 0.21% |
| record | break_points | 1 | 0.21% |
| record | tournament_code | 1 | 0.21% |
| record | winner | 1 | 0.21% |
| record | aces | 1 | 0.21% |
| record | amount: Total amount raised in millions | 1 | 0.21% |
| record | company: Name of the company raising capital | 1 | 0.21% |
| record | nf_id: Unique identifier for each funding event | 1 | 0.21% |
| record | round: Series or type of funding round | 1 | 0.21% |
| record | investors: List of venture capital firms involved | 1 | 0.21% |
| record | onboarding_code | 1 | 0.21% |
| record | person.death | 1 | 0.21% |
| record | person.title | 1 | 0.21% |
| record | place.name | 1 | 0.21% |
| record | leasing_offer | 1 | 0.21% |
| record | error.message | 1 | 0.21% |
| record | error.possible_causes | 1 | 0.21% |
| record | rate | 1 | 0.21% |
| record | snapshot_id | 1 | 0.21% |
| record | audit_id | 1 | 0.21% |
| record | key_concepts | 1 | 0.21% |
| record | student_name: Name of the student who took the exam | 1 | 0.21% |
| record | subject: Subject area of the exam | 1 | 0.21% |
| record | publication_id: Unique identifier for the exam publication | 1 | 0.21% |
| record | status: Current status of the exam result | 1 | 0.21% |
| record | publication_code | 1 | 0.21% |
| record | geographical_scope | 1 | 0.21% |
| record | result_type | 1 | 0.21% |
| record | assessment_name | 1 | 0.21% |
| record | channel_name | 1 | 0.21% |
| record | thumbnail_description | 1 | 0.21% |
| record | target_audience | 1 | 0.21% |
| record | comment | 1 | 0.21% |
| record | diagnosis: ICD-10 code representing the medical condition | 1 | 0.21% |
| record | cpt_code: CPT code for the medical procedure performed | 1 | 0.21% |
| record | claim_date: Date the claim was filed with the insurer | 1 | 0.21% |
| record | material: Primary materials used in construction | 1 | 0.21% |
| record | weight: Weight of the product in grams or kg | 1 | 0.21% |
| record | price: Price of the product in dollars | 1 | 0.21% |
| record | naming.named_after.name | 1 | 0.21% |
| record | naming.date | 1 | 0.21% |
| record | naming.named_after.profession | 1 | 0.21% |
| record | customer_address | 1 | 0.21% |
| record | offer_id: Unique identifier for each leasing offer | 1 | 0.21% |
| record | tenant: Name of the company occupying the leased space | 1 | 0.21% |
| record | location: Full address of the leased property | 1 | 0.21% |
| record | features: List of included amenities or space features | 1 | 0.21% |
| record | salary | 1 | 0.21% |
| record | phone | 1 | 0.21% |
| record | hire_date | 1 | 0.21% |
| record | date_added | 1 | 0.21% |
| record | geographic_region | 1 | 0.21% |
| record | connectivity | 1 | 0.21% |
| record | market_segment | 1 | 0.21% |
| record | current_scale | 1 | 0.21% |
| record | mission | 1 | 0.21% |
| record | current_ratio: Liquidity ratio measuring short-term obligations | 1 | 0.21% |
| record | pro_accomplishments | 1 | 0.21% |
| record | discovery.observatory.location.county | 1 | 0.21% |
| record | discovery.observatory.location.state | 1 | 0.21% |
| record | orbital_characteristics.inclination.reference | 1 | 0.21% |
| record | orbital_characteristics.asteroid_belt_classification | 1 | 0.21% |
| record | physical_characteristics.observations.0.year | 1 | 0.21% |
| record | definition.source_organization.name | 1 | 0.21% |
| record | business_sustainability.elements | 1 | 0.21% |
| record | industry_example.industry | 1 | 0.21% |
| record | crisis_resilience | 1 | 0.21% |
| record | definition.source_organization.full_name | 1 | 0.21% |
| record | financial_aspect.necessity | 1 | 0.21% |
| record | definition.document_title | 1 | 0.21% |
| record | three_pillars | 1 | 0.21% |
| record | industry_example.challenges | 1 | 0.21% |
| record | search_result_page.website_name | 1 | 0.21% |
| record | initial_text.actions | 1 | 0.21% |
| record | initial_text.negations | 1 | 0.21% |
| record | search_result_page.related_to | 1 | 0.21% |
| record | region_churn | 1 | 0.21% |
| record | event_name: Title of the event as listed | 1 | 0.21% |
| record | organizer_name: Name of the person or group organizing the event | 1 | 0.21% |
| record | location: Full address including street, city, and state | 1 | 0.21% |
| record | event_date: Date when the event will take place | 1 | 0.21% |
| record | event_status: Current status of the event | 1 | 0.21% |
| record | attendee_count: Number of registered attendees | 1 | 0.21% |
| record | city: City where the event is held | 1 | 0.21% |
| record | event_code: Unique code identifying the event | 1 | 0.21% |
| record | useoffunds: Purpose or application of raised capital | 1 | 0.21% |
| record | investors: List of investor names or groups involved | 1 | 0.21% |
| record | funding_id: Unique identifier assigned to the funding round | 1 | 0.21% |
| record | owner: Name of the healthcare professional owning the device | 1 | 0.21% |
| record | model: Manufacturer model name of the medical device | 1 | 0.21% |
| record | serial: Unique serial number identifying the device | 1 | 0.21% |
| record | vendor: Company that sold or provided the device | 1 | 0.21% |
| record | location: Department or unit where device is currently located | 1 | 0.21% |
| record | status: Current operational status of the device | 1 | 0.21% |
| record | weather_conditions | 1 | 0.21% |
| record | league | 1 | 0.21% |
| record | bedrooms | 1 | 0.21% |
| record | account | 1 | 0.21% |
| record | payee | 1 | 0.21% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.0.t_score | 1 | 0.21% |
| record | statistical_results.hazard_ratios.0.comparison | 1 | 0.21% |
| record | statistical_results.predictive_model.variables_included | 1 | 0.21% |
| record | statistical_results.hazard_ratios.0.value | 1 | 0.21% |
| record | study_metadata.measurement_technology.anatomical_site | 1 | 0.21% |
| record | population.age_range.max | 1 | 0.21% |
| record | statistical_results.predictive_model.value | 1 | 0.21% |
| record | conclusions | 1 | 0.21% |
| record | population.age_range.unit | 1 | 0.21% |
| record | outcomes.fracture_events.0.count | 1 | 0.21% |
| record | outcomes.data_source | 1 | 0.21% |
| record | population.gender | 1 | 0.21% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.1.t_score | 1 | 0.21% |
| record | follow_up.average_duration.years | 1 | 0.21% |
| record | statistical_results.diagnostic_metrics.sensitivity_at_threshold.t_score_threshold | 1 | 0.21% |
| record | follow_up.average_duration.months | 1 | 0.21% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.1.rate_percentage | 1 | 0.21% |
| record | urls | 1 | 0.21% |
| record | status: Current operational status | 1 | 0.21% |
| record | employees: Number of employees | 1 | 0.21% |
| record | game_id: Unique identifier for the game as listed in the schedule | 1 | 0.21% |
| record | home_score: Final score for the home team | 1 | 0.21% |
| record | venue: Name of the arena where the game was played | 1 | 0.21% |
| record | expectedOutcome | 1 | 0.21% |
| record | surveyFocus | 1 | 0.21% |
| record | explanations | 1 | 0.21% |
| record | payee: Name of the entity receiving payment | 1 | 0.21% |
| record | description: Brief description of the transaction purpose | 1 | 0.21% |
| record | physical_characteristics.observations.0.telescope.origin | 1 | 0.21% |
| record | asteroid_belt_classification | 1 | 0.21% |
| record | discovery.observatory.location.prefecture | 1 | 0.21% |
| record | office_hours.2.start_time | 1 | 0.21% |
| record | contact.fax | 1 | 0.21% |
| record | office_hours.3.start_time | 1 | 0.21% |
| record | address.street | 1 | 0.21% |
| record | office_hours.4.day | 1 | 0.21% |
| record | office_hours.2.day | 1 | 0.21% |
| record | office_hours.2.end_time | 1 | 0.21% |
| record | office_hours.4.end_time | 1 | 0.21% |
| record | office_hours.4.start_time | 1 | 0.21% |
| record | office_hours.3.end_time | 1 | 0.21% |
| record | office_hours.3.day | 1 | 0.21% |
| record | office_hours.0.end_time | 1 | 0.21% |
| record | setting.political_context | 1 | 0.21% |
| record | main_character.rank | 1 | 0.21% |
| record | publication_details.first_publication.year | 1 | 0.21% |
| record | publication_details.year_written | 1 | 0.21% |
| record | publication_details.first_publication_czechoslovakia | 1 | 0.21% |
| record | main_character.affiliation | 1 | 0.21% |
| record | comparison | 1 | 0.21% |
| record | original_title | 1 | 0.21% |
| record | medication | 1 | 0.21% |
| record | pharmacy_name | 1 | 0.21% |
| record | pharmacy_address | 1 | 0.21% |
| record | causes | 1 | 0.21% |
| record | support | 1 | 0.21% |
| record | lead_investor | 1 | 0.21% |
| record | hb: Hemoglobin level in grams per deciliter | 1 | 0.21% |
| record | test_date: Date the lab test was conducted | 1 | 0.21% |
| record | patient_name: Name of the patient undergoing the test | 1 | 0.21% |
| record | platelets: Platelet count in thousands per microliter | 1 | 0.21% |
| record | verbs_past_tense | 1 | 0.21% |
| record | prepositions | 1 | 0.21% |
| record | adverbs | 1 | 0.21% |
| record | proper_nouns | 1 | 0.21% |
| record | singular_nouns | 1 | 0.21% |
| record | plural_nouns | 1 | 0.21% |
| record | possessive_pronouns | 1 | 0.21% |
| record | adjectives | 1 | 0.21% |
| record | possessive_endings | 1 | 0.21% |
| record | cardinal_numbers | 1 | 0.21% |
| record | asteroid.discovery_year | 1 | 0.21% |
| record | asteroid.orbital_characteristics.unit | 1 | 0.21% |
| record | asteroid.orbital_characteristics.inclination_unit | 1 | 0.21% |
| record | app_id | 1 | 0.21% |
| record | os_version | 1 | 0.21% |
| record | word_count | 1 | 0.21% |
| record | helpful_votes | 1 | 0.21% |
| record | language | 1 | 0.21% |
| record | review_platform | 1 | 0.21% |
| record | review_link | 1 | 0.21% |
| record | project_id: Unique identifier assigned to the technology initiative | 1 | 0.21% |
| record | developers: List of institutions or individuals leading the project | 1 | 0.21% |
| record | funding_source: Organization or fund providing financial support | 1 | 0.21% |
| record | technology_name: Name of the algorithm, platform, or device | 1 | 0.21% |
| record | deployment_location: Geographic region where system is deployed | 1 | 0.21% |
| record | deployment_count: Number of farms, hospitals, or states using the system | 1 | 0.21% |
| record | sensor_nodes: Number of data collection nodes used in the system | 1 | 0.21% |
| record | turning_point | 1 | 0.21% |
| record | normal_range | 1 | 0.21% |
| record | filer: Name of the person or entity filing the case | 1 | 0.21% |
| record | star.designation | 1 | 0.21% |
| record | star.constellation | 1 | 0.21% |
| record | star.distance.light_years | 1 | 0.21% |
| record | star.distance.parsecs | 1 | 0.21% |
| record | star.spectral_class | 1 | 0.21% |
| record | star.effective_temperature | 1 | 0.21% |
| record | planetary_system.planets.0.discovery_year | 1 | 0.21% |
| record | planetary_system.planets.0.designation | 1 | 0.21% |
| record | planetary_system.planets.0.discovery_team | 1 | 0.21% |
| record | sender_name | 1 | 0.21% |
| record | sender_company | 1 | 0.21% |
| record | sender_location | 1 | 0.21% |
| record | receiver_name | 1 | 0.21% |
| record | receiver_company | 1 | 0.21% |
| record | receiver_location | 1 | 0.21% |
| record | product_offered | 1 | 0.21% |
| record | sender: Name of the company that issued the invoice | 1 | 0.21% |
| record | invoice_number: Unique identifier for the invoice as assigned | 1 | 0.21% |
| record | status: Current financial status of the invoice | 1 | 0.21% |
| record | recipient_address: Street address where goods or services were delivered | 1 | 0.21% |
| record | usage_frequency | 1 | 0.21% |
| record | event-attack/battle/war/militaryconflict | 1 | 0.21% |
| record | event | 1 | 0.21% |
| record | sales | 1 | 0.21% |
| record | publisher | 1 | 0.21% |
| record | platform | 1 | 0.21% |
| record | test_cases_completed | 1 | 0.21% |
| record | team | 1 | 0.21% |
| record | mechanismofaction: Biological process inhibited or modulated by the drug | 1 | 0.21% |
| record | lastreviewdate: Date the drug record was last reviewed | 1 | 0.21% |
| record | adverse_effects: List of side effects reported in clinical use | 1 | 0.21% |
| record | molecular_weight: Molecular weight of the drug in daltons | 1 | 0.21% |
| record | drug_name: Name of the pharmaceutical product | 1 | 0.21% |
| record | clinical_phase: Current stage of clinical development | 1 | 0.21% |
| record | city: City where the transaction took place | 1 | 0.21% |
| record | name: Full name of the customer | 1 | 0.21% |
| record | transaction_type: Mode of transaction such as online or in-person | 1 | 0.21% |
| record | amount: Total cost of the fuel refill | 1 | 0.21% |
| record | station: Name of the gas station | 1 | 0.21% |
| record | description: Description of the transaction activity | 1 | 0.21% |
| record | bench_points: Total points scored by bench players | 1 | 0.21% |
| record | key_moment: Descriptive note about a pivotal play or event in the game | 1 | 0.21% |
| record | steals: Total steals recorded during the game | 1 | 0.21% |
| record | game_id: Unique identifier for the game as provided in the passage | 1 | 0.21% |
| record | home_score: Final score of the home team | 1 | 0.21% |
| record | guidance | 1 | 0.21% |
| record | cfo | 1 | 0.21% |
| record | refund_amount | 1 | 0.21% |
| record | order_id | 1 | 0.21% |
| record | region.architecture | 1 | 0.21% |
| record | region.location.extends_from | 1 | 0.21% |
| record | region.route | 1 | 0.21% |
| record | funding_amount: Initial capital received in USD | 1 | 0.21% |
| record | dish | 1 | 0.21% |
| record | kitchen appliance | 1 | 0.21% |
| record | food item | 1 | 0.21% |
| record | ingredient | 1 | 0.21% |
| record | side dish | 1 | 0.21% |
| record | merchant | 1 | 0.21% |
| record | description: Purpose or context of the contribution | 1 | 0.21% |
| record | donor_name: Full name of the contributing individual or organization | 1 | 0.21% |
| record | recipient_office: Office or position being supported | 1 | 0.21% |
| record | campaign_committee: Name of the political committee receiving the contribution | 1 | 0.21% |
| record | disclosure_id: Unique identifier for the disclosure entry | 1 | 0.21% |
| record | issue_type: Type of technical problem reported | 1 | 0.21% |
| record | submitted_date: Date the ticket was submitted | 1 | 0.21% |
| record | severity: Priority level of the issue | 1 | 0.21% |
| record | employee_name: Name of employee who reported the issue | 1 | 0.21% |
| record | ticket_id: Unique identifier for each service ticket | 1 | 0.21% |
| record | department: Department where the issue originated | 1 | 0.21% |
| record | reference_number: Unique identifier for the invoice document | 1 | 0.21% |
| record | recipient_name: Name of the recipient organization | 1 | 0.21% |
| record | genus.name_zh | 1 | 0.21% |
| record | contribution_type | 1 | 0.21% |
| record | reference: Internal reference code linked to the ticket | 1 | 0.21% |
| record | user: Name of the employee reporting the issue | 1 | 0.21% |
| record | description: Detailed account of the problem and symptoms | 1 | 0.21% |
| record | status: Current state of the ticket | 1 | 0.21% |
| record | priority: Severity level of the issue | 1 | 0.21% |
| record | affected_system: System or application where the issue occurs | 1 | 0.21% |
| record | expert_opinion.speaker.name | 1 | 0.21% |
| record | expert_opinion.statement | 1 | 0.21% |
| record | expert_opinion.sentiment | 1 | 0.21% |
| record | article_metadata.relative_timestamp | 1 | 0.21% |
| record | discovery.observatory.location | 1 | 0.21% |
| record | discovery.discoverers | 1 | 0.21% |
| record | orbital_characteristics.classification | 1 | 0.21% |
| record | physical_characteristics.observations.0.telescope.orbit_year | 1 | 0.21% |
| record | position | 1 | 0.21% |
| record | key_achievement | 1 | 0.21% |
| record | rebounds: Rebounds recorded by the player | 1 | 0.21% |
| record | points: Points scored by the player | 1 | 0.21% |
| record | assists: Assists recorded by the player | 1 | 0.21% |
| record | hba1c | 1 | 0.21% |
| record | venue: Name of the stadium or venue where the event occurred | 1 | 0.21% |
| record | nation: Country of the athlete who won the medal | 1 | 0.21% |
| record | time: Official race time recorded in the event | 1 | 0.21% |
| record | medal: Type of medal won gold, silver, or bronze | 1 | 0.21% |
| record | assigned_adjuster | 1 | 0.21% |
| record | word_count: Number of words in the abstract | 1 | 0.21% |
| record | submission_id: Unique identifier assigned to each abstract submission | 1 | 0.21% |
| record | funding_code: Code indicating eligibility for funding support | 1 | 0.21% |
| record | shipment_type | 1 | 0.21% |
| record | claim_amount | 1 | 0.21% |
| record | crime.location.landmark | 1 | 0.21% |
| record | perpetrator.status | 1 | 0.21% |
| record | crime.location.address | 1 | 0.21% |
| record | perpetrator.arrest_details.agency | 1 | 0.21% |
| record | crime.location.sector | 1 | 0.21% |
| record | crime.motive | 1 | 0.21% |
| record | victim.name | 1 | 0.21% |
| record | investigation.agency | 1 | 0.21% |
| record | crime.weapon.type | 1 | 0.21% |
| record | crime.weapon.caliber | 1 | 0.21% |
| record | perpetrator.arrest_details.time | 1 | 0.21% |
| record | crime.date | 1 | 0.21% |
| record | investigation.medical_examiner.name | 1 | 0.21% |
| record | crime.weapon.source.stolen_from.name | 1 | 0.21% |
| record | investigation.status | 1 | 0.21% |
| record | description.rosette | 1 | 0.21% |
| record | description.leaves.color | 1 | 0.21% |
| record | systematics_and_distribution.first_description.year | 1 | 0.21% |
| record | systematics_and_distribution.distinguishing_features | 1 | 0.21% |
| record | systematics_and_distribution.location.elevation_m.max | 1 | 0.21% |
| record | description.leaves.length_cm.min | 1 | 0.21% |
| record | description.flowers.length_mm.min | 1 | 0.21% |
| record | description.fruit.type | 1 | 0.21% |
| record | status: Current status of the delay record | 1 | 0.21% |
| record | departure_date: Date flight departs from origin | 1 | 0.21% |
| record | email: Email address of the passenger | 1 | 0.21% |
| record | reason: Cause of the flight delay | 1 | 0.21% |
| record | passenger_name: Name of the passenger on the flight | 1 | 0.21% |
| record | origin: Airport code where flight originates | 1 | 0.21% |
| record | delay_minutes: Number of minutes flight is delayed | 1 | 0.21% |
| record | product: Name of the skincare or apparel product | 1 | 0.21% |
| record | market_cap: Market capitalization in billions | 1 | 0.21% |
| record | revenue: Total revenue in billions as reported | 1 | 0.21% |
| record | growth_rate: Year-over-year growth percentage | 1 | 0.21% |
| record | century | 1 | 0.21% |
| record | images | 1 | 0.21% |
| record | assigned_agent | 1 | 0.21% |
| record | Consumption | 1 | 0.21% |
| record | Food | 1 | 0.21% |
| record | diagnosis_code | 1 | 0.21% |
| record | magnitudeorindex | 1 | 0.21% |
| record | exhibitor_id | 1 | 0.21% |
| record | contact_email | 1 | 0.21% |
| record | author: Name of the guideline's lead author | 1 | 0.21% |
| record | target_population: Group of patients or individuals the guideline targets | 1 | 0.21% |
| record | institution: Affiliation of the guideline's author | 1 | 0.21% |
| record | submission_method | 1 | 0.21% |
| record | opponent: Entity filing the dispute or opposition | 1 | 0.21% |
| record | trademark_name: Name of the registered trademark in dispute | 1 | 0.21% |
| record | dispute_type: Type of legal action such as cancellation or infringement | 1 | 0.21% |
| record | registrant: Company that owns the registered trademark | 1 | 0.21% |
| record | registration_number: Unique number assigned to the trademark registration | 1 | 0.21% |
| record | Edwin Harris Walker <> developed | 1 | 0.21% |
| record | Robert <> identified as | 1 | 0.21% |
| record | Robert <> influenced by | 1 | 0.21% |
| record | Robert <> considered | 1 | 0.21% |
| record | Robert <> experienced | 1 | 0.21% |
| record | Robert <> practiced | 1 | 0.21% |
| record | Robert <> favored | 1 | 0.21% |
| record | Robert <> observed | 1 | 0.21% |
| record | Robert <> valued | 1 | 0.21% |
| record | Robert <> was asked by | 1 | 0.21% |
| record | Robert <> advocated | 1 | 0.21% |
| record | Robert <> interacted with | 1 | 0.21% |
| record | Robert <> questioned | 1 | 0.21% |
| record | Robert <> cited | 1 | 0.21% |
| record | Robert <> utilized | 1 | 0.21% |
| record | property_address | 1 | 0.21% |
| record | risk_category | 1 | 0.21% |
| record | coverage_limit | 1 | 0.21% |
| record | system_code | 1 | 0.21% |
| record | user | 1 | 0.21% |
| record | user_id | 1 | 0.21% |
| record | album.external_links.1.url | 1 | 0.21% |
| record | album.chart_performance.record_holder_description | 1 | 0.21% |
| record | album.tracklist.cd.0.notes | 1 | 0.21% |
| record | album.tracklist.cd.2.tie_in.0.type | 1 | 0.21% |
| record | album.external_links.3.url | 1 | 0.21% |
| record | album.tracklist.cd.1.title | 1 | 0.21% |
| record | album.tracklist.cd.7.title | 1 | 0.21% |
| record | album.tracklist.cd.9.track_number | 1 | 0.21% |
| record | album.tracklist.cd.4.tie_in.0.title | 1 | 0.21% |
| record | album.free_concert_location | 1 | 0.21% |
| record | album.artist | 1 | 0.21% |
| record | album.tracklist.cd.7.lyrics | 1 | 0.21% |
| record | album.tracklist.cd.0.title | 1 | 0.21% |
| record | album.tracklist.dvd.1.type | 1 | 0.21% |
| record | album.tracklist.cd.1.lyrics | 1 | 0.21% |
| record | album.sources.3.source_name | 1 | 0.21% |
| record | album.tracklist.cd.9.tie_in.0.title | 1 | 0.21% |
| record | album.tracklist.cd.6.title | 1 | 0.21% |
| record | album.versions.0.type | 1 | 0.21% |
| record | album.versions.1.name | 1 | 0.21% |
| record | album.tracklist.cd.3.notes | 1 | 0.21% |
| record | album.tracklist.cd.2.tie_in.0.title | 1 | 0.21% |
| record | album.external_links.1.description | 1 | 0.21% |
| record | album.anniversary | 1 | 0.21% |
| record | album.tracklist.cd.1.notes | 1 | 0.21% |
| record | album.tracklist.dvd.2.type | 1 | 0.21% |
| record | album.external_links.0.url | 1 | 0.21% |
| record | album.tracklist.cd.2.title | 1 | 0.21% |
| record | album.tracklist.cd.8.tie_in.0.title | 1 | 0.21% |
| record | album.sources.1.access_date | 1 | 0.21% |
| record | album.tracklist.cd.8.composer | 1 | 0.21% |
| record | album.sources.2.publication_date | 1 | 0.21% |
| record | album.tracklist.cd.9.lyrics | 1 | 0.21% |
| record | album.tracklist.cd.3.title | 1 | 0.21% |
| record | album.tracklist.cd.7.composer | 1 | 0.21% |
| record | album.versions.3.type | 1 | 0.21% |
| record | album.tracklist.cd.2.notes | 1 | 0.21% |
| record | album.tracklist.dvd.0.type | 1 | 0.21% |
| record | album.release_date | 1 | 0.21% |
| record | album.tracklist.cd.2.track_number | 1 | 0.21% |
| record | album.sources.3.publication_date | 1 | 0.21% |
| record | album.sources.1.publication_date | 1 | 0.21% |
| record | album.tracklist.cd.4.tie_in.0.type | 1 | 0.21% |
| record | album.tracklist.cd.4.track_number | 1 | 0.21% |
| record | album.versions.1.contents | 1 | 0.21% |
| record | album.tracklist.cd.6.track_number | 1 | 0.21% |
| record | album.chart_performance.record_holder_rank | 1 | 0.21% |
| record | album.tracklist.cd.1.track_number | 1 | 0.21% |
| record | album.tracklist.cd.5.composer | 1 | 0.21% |
| record | album.chart_performance.consecutive_number_one_albums | 1 | 0.21% |
| record | album.tracklist.cd.0.tie_in.0.type | 1 | 0.21% |
| record | album.versions.2.contents | 1 | 0.21% |
| record | album.external_links.2.description | 1 | 0.21% |
| record | album.tracklist.cd.8.track_number | 1 | 0.21% |
| record | album.sources.2.access_date | 1 | 0.21% |
| record | album.tracklist.dvd.1.track_number | 1 | 0.21% |
| record | album.tracklist.dvd.1.title | 1 | 0.21% |
| record | album.tracklist.cd.8.notes | 1 | 0.21% |
| record | album.title | 1 | 0.21% |
| record | album.sources.3.reference_number | 1 | 0.21% |
| record | album.tracklist.cd.2.composer | 1 | 0.21% |
| record | album.tracklist.cd.4.composer | 1 | 0.21% |
| record | album.tracklist.cd.9.tie_in.0.type | 1 | 0.21% |
| record | album.external_links.2.url | 1 | 0.21% |
| record | album.tracklist.dvd.2.title | 1 | 0.21% |
| record | album.sources.0.publication_date | 1 | 0.21% |
| record | album.tracklist.cd.4.notes | 1 | 0.21% |
| record | album.versions.3.name | 1 | 0.21% |
| record | album.tracklist.cd.8.tie_in.0.type | 1 | 0.21% |
| record | album.tracklist.cd.7.track_number | 1 | 0.21% |
| record | album.tracklist.dvd.0.title | 1 | 0.21% |
| record | album.tracklist.cd.3.track_number | 1 | 0.21% |
| record | sports | 1 | 0.21% |
| record | software | 1 | 0.21% |
| record | report_date | 1 | 0.21% |
| record | transcription_factors | 1 | 0.21% |
| record | activation_mechanism | 1 | 0.21% |
| record | incident_id | 1 | 0.21% |
| record | priority | 1 | 0.21% |
| record | root_cause | 1 | 0.21% |
| record | resolution_time | 1 | 0.21% |
| record | service | 1 | 0.21% |
| record | resolution_team | 1 | 0.21% |
| record | entity | 1 | 0.21% |
| record | tracking_code | 1 | 0.21% |
| record | heroes | 1 | 0.21% |
| record | organizations_and_media | 1 | 0.21% |
| record | villains | 1 | 0.21% |
| record | drug_id | 1 | 0.21% |
| record | form | 1 | 0.21% |
| record | indication | 1 | 0.21% |
| record | response | 1 | 0.21% |
| record | file_id | 1 | 0.21% |
| record | renewal_amount: Monetary amount charged for the renewal | 1 | 0.21% |
| record | coverage_limit: Maximum coverage amount specified in the policy | 1 | 0.21% |
| record | policy_id: Unique identifier assigned to the policy | 1 | 0.21% |
| record | affected_site | 1 | 0.21% |
| record | affected_network | 1 | 0.21% |
| record | alternative_access | 1 | 0.21% |
| record | queue | 1 | 0.21% |
| record | patient.behavior | 1 | 0.21% |
| record | patient.request | 1 | 0.21% |
| record | patient.age_estimation.initial_estimation | 1 | 0.21% |
| record | patient.age_estimation.correction | 1 | 0.21% |
| record | patient.age_estimation.full_age | 1 | 0.21% |
| record | patient.age_estimation.previous_state | 1 | 0.21% |
| record | doctor.action | 1 | 0.21% |
| record | doctor.question | 1 | 0.21% |
| record | doctor.reaction | 1 | 0.21% |
| record | chair.state | 1 | 0.21% |
| record | chair.condition | 1 | 0.21% |
| record | product_line | 1 | 0.21% |
| record | brief_id | 1 | 0.21% |
| record | submitter: Name of employee who filed the ticket | 1 | 0.21% |
| record | ticket_id: Unique identifier for each support request | 1 | 0.21% |
| record | product_id: Unique identifier for each product | 1 | 0.21% |
| record | stock: Current quantity available in inventory | 1 | 0.21% |
| record | last_updated: Date the product record was last modified | 1 | 0.21% |
| record | sales_volume: Total units sold over time | 1 | 0.21% |
| record | description: Brief product feature and use description | 1 | 0.21% |
| record | coverage_amount | 1 | 0.21% |
| record | owner_name | 1 | 0.21% |
| record | term_years | 1 | 0.21% |
| record | labels | 1 | 0.21% |
| record | schema_name | 1 | 0.21% |
| record | location.district | 1 | 0.21% |
| record | price_range.per | 1 | 0.21% |
| record | location.country | 1 | 0.21% |
| record | booking_options.tripadvisor_role | 1 | 0.21% |
| record | booking_options.partnerships | 1 | 0.21% |
| record | booking_options.assurance_message | 1 | 0.21% |
| record | price_range.currency | 1 | 0.21% |
| record | location.province | 1 | 0.21% |
| record | additional_value: Extra numeric value associated with the transaction | 1 | 0.21% |
| record | status: Current transaction state | 1 | 0.21% |
| record | address: Full physical address of the owner | 1 | 0.21% |
| record | date: Date the transaction was initiated | 1 | 0.21% |
| record | transaction_id: Unique identifier for the transaction | 1 | 0.21% |
| record | panel_topics | 1 | 0.21% |
| record | registration_fee | 1 | 0.21% |
| record | registration_cap | 1 | 0.21% |
| record | weight | 1 | 0.21% |
| record | available_units | 1 | 0.21% |
| record | stock_status | 1 | 0.21% |
| record | shipping | 1 | 0.21% |
| record | color | 1 | 0.21% |
| record | warranty | 1 | 0.21% |
| record | vaccine | 1 | 0.21% |
| record | dose | 1 | 0.21% |
| record | code | 1 | 0.21% |
| record | people | 1 | 0.21% |
| record | dates_mentioned | 1 | 0.21% |
| record | topics | 1 | 0.21% |
| record | amount: Total funding amount in USD with currency symbol | 1 | 0.21% |
| record | investor: Name of the venture capital firm that invested | 1 | 0.21% |
| record | event.action | 1 | 0.21% |
| record | event.target | 1 | 0.21% |
| record | event.location.city | 1 | 0.21% |
| record | event.location.region | 1 | 0.21% |
| record | social_media_post.platform | 1 | 0.21% |
| record | event.location.venue | 1 | 0.21% |
| record | previous_incident.year | 1 | 0.21% |
| record | event.date | 1 | 0.21% |
| record | event.subject | 1 | 0.21% |
| record | source.type | 1 | 0.21% |
| record | emergency_treatment | 1 | 0.21% |
| record | when_to_seek_help | 1 | 0.21% |
| record | codes | 1 | 0.21% |
| record | prevention | 1 | 0.21% |
| record | last_updated | 1 | 0.21% |
| record | attending_physician | 1 | 0.21% |
| record | bloodpressuresystolic | 1 | 0.21% |
| record | bloodpressurediastolic | 1 | 0.21% |
| record | monitoring_method | 1 | 0.21% |
| record | registrations | 1 | 0.21% |
| record | sponsor | 1 | 0.21% |
| record | activity | 1 | 0.21% |
| record | sponsor: Individual legislator who introduced the bill | 1 | 0.21% |
| record | yes_votes: Number of members voting in favor | 1 | 0.21% |
| record | no_votes: Number of members voting against | 1 | 0.21% |
| record | comment_id | 1 | 0.21% |
| record | patient_name: Full name of the patient receiving the medication | 1 | 0.21% |
| record | patient_address: Street address of the patient | 1 | 0.21% |
| record | zip_code: ZIP code of the patient's residence | 1 | 0.21% |
| record | prescription_id: Unique identifier for the prescription record | 1 | 0.21% |
| record | medication: Name of the prescribed medication | 1 | 0.21% |
| record | dosage: Instructions on how and when to take the medication | 1 | 0.21% |
| record | supply_quantity: Total number of tablets dispensed | 1 | 0.21% |
| record | status: Current status of the prescription | 1 | 0.21% |
| record | temperature | 1 | 0.21% |
| record | respiratory_rate | 1 | 0.21% |
| record | patent_number | 1 | 0.21% |
| record | inventor | 1 | 0.21% |
| record | technology_area | 1 | 0.21% |
| record | amount: Claimed financial amount in dollars | 1 | 0.21% |
| record | sender: Name of person who submitted the report | 1 | 0.21% |
| record | report_id: Unique identifier for the phishing incident | 1 | 0.21% |
| record | subject: Subject line of the malicious email | 1 | 0.21% |
| record | malicious_content: Description of the phishing element like link or attachment | 1 | 0.21% |
| record | chamber | 1 | 0.21% |
| record | bill_number | 1 | 0.21% |
| record | vote_result | 1 | 0.21% |
| record | proposer | 1 | 0.21% |
| record | date_posted | 1 | 0.21% |
| record | item: Description of the returned product | 1 | 0.21% |
| record | status: Current status of the return approval | 1 | 0.21% |
| record | amount: Monetary value of the return in dollars | 1 | 0.21% |
| record | apparent_magnitude.band | 1 | 0.21% |
| record | star_name | 1 | 0.21% |
| record | status: Current phase of the event lifecycle | 1 | 0.21% |
| record | attendees: Number of people registered for the event | 1 | 0.21% |
| record | name: Title or name of the event | 1 | 0.21% |
| record | services.maintenance | 1 | 0.21% |
| record | services.selection_assistance | 1 | 0.21% |
| record | distributor_information.capabilities | 1 | 0.21% |
| record | actions | 1 | 0.21% |
| record | serial_number: Unique serial number for device tracking | 1 | 0.21% |
| record | usage_count: Total patient admissions or scans performed | 1 | 0.21% |
| record | outreach_date: Date the outreach was sent | 1 | 0.21% |
| record | sender: Name of the person initiating the outreach | 1 | 0.21% |
| record | status: Current status of the outreach attempt | 1 | 0.21% |
| record | recipient: Name of the recipient being contacted | 1 | 0.21% |
| record | outreach_code: Code associated with the outreach event | 1 | 0.21% |
| record | goals | 1 | 0.21% |
| record | nutritional_aspects.sucrose.daily_limit_percentage | 1 | 0.21% |
| record | nutritional_aspects.sucrose.daily_limit_grams | 1 | 0.21% |
| record | weight_loss_benefits.improvements | 1 | 0.21% |
| record | nutritional_aspects.carbohydrates.daily_percentage | 1 | 0.21% |
| record | nutritional_aspects.carbohydrates.importance | 1 | 0.21% |
| record | policyholder_id: Unique identifier for the policyholder | 1 | 0.21% |
| record | claim_date: Date the claim was filed | 1 | 0.21% |
| record | policyholder_name: Full name of the claimant | 1 | 0.21% |
| record | match_id: Unique identifier for the match as provided | 1 | 0.21% |
| record | home_team: Team playing at the home venue | 1 | 0.21% |
| record | away_team: Team playing at the away venue | 1 | 0.21% |
| record | score: Final score in format home-away | 1 | 0.21% |
| record | price: Current selling price of the product in USD | 1 | 0.21% |
| record | last_updated: Date when product record was last modified | 1 | 0.21% |
| record | address_city: City where patient resides | 1 | 0.21% |
| record | condition: Medical diagnosis associated with symptoms | 1 | 0.21% |
| record | patient_id: Unique identifier for each patient | 1 | 0.21% |
| record | address_street: Street address line of the patient | 1 | 0.21% |
| record | address_state: State where patient resides | 1 | 0.21% |
| record | symptom_location: Body region where symptoms are felt | 1 | 0.21% |
| record | players | 1 | 0.21% |
| record | venue_id | 1 | 0.21% |
| record | claim_amount: Monetary claim amount if applicable | 1 | 0.21% |
| record | case_id: Unique case identifier in format CV-YYYY-XXXX | 1 | 0.21% |
| record | csds_id | 1 | 0.21% |
| record | manufacturer_address | 1 | 0.21% |
| record | recipient_email | 1 | 0.21% |
| record | recipient_name | 1 | 0.21% |
| record | vendor_address | 1 | 0.21% |
| record | monitoring_mechanism | 1 | 0.21% |
| record | goal | 1 | 0.21% |
| record | signatories | 1 | 0.21% |
| record | treaty_name | 1 | 0.21% |
| record | target_year | 1 | 0.21% |
| record | geographic_scope | 1 | 0.21% |
| record | patient_id: Unique identifier for the patient entry | 1 | 0.21% |
| record | status: Current status of vaccination record | 1 | 0.21% |
| record | vaccine_type: Type of vaccine administered | 1 | 0.21% |
| record | repair_shop | 1 | 0.21% |
| record | articleTitle | 1 | 0.21% |
| record | preserved_specimens | 1 | 0.21% |
| record | device_type: Type of medical device such as ECG or infusion pump | 1 | 0.21% |
| record | assigned_to: Name of healthcare professional responsible for device use | 1 | 0.21% |
| record | dobj | 1 | 0.21% |
| record | nmod | 1 | 0.21% |
| record | cc | 1 | 0.21% |
| record | preconj | 1 | 0.21% |
| record | case | 1 | 0.21% |
| record | npadvmod | 1 | 0.21% |
| record | intj | 1 | 0.21% |
| record | nummod | 1 | 0.21% |
| record | treatment: Therapeutic approach prescribed | 1 | 0.21% |
| record | provider: Name of the physician who treated the patient | 1 | 0.21% |
| record | diagnosis: Medical condition diagnosed with ICD-10 code | 1 | 0.21% |
| record | patient_id: Unique identifier for the patient | 1 | 0.21% |
| record | city: City where patient resides | 1 | 0.21% |
| record | symptoms: List of reported symptoms like fever or headache | 1 | 0.21% |
| record | onset_date: Date when symptoms first appeared | 1 | 0.21% |
| record | address: Street address of the patient | 1 | 0.21% |
| record | points | 1 | 0.21% |
| record | rebounds | 1 | 0.21% |
| record | player | 1 | 0.21% |
| record | scorer | 1 | 0.21% |
| record | state_or_province | 1 | 0.21% |
| record | established_year | 1 | 0.21% |
| record | specialization | 1 | 0.21% |
| record | office | 1 | 0.21% |
| record | appointment_id | 1 | 0.21% |
| record | service_type | 1 | 0.21% |
| record | location: Property address including street, city, and state | 1 | 0.21% |
| record | policyholder: Name of the insured individual | 1 | 0.21% |
| record | priority_level | 1 | 0.21% |
| record | affected_period | 1 | 0.21% |
| record | patient_name: Name of the patient filing the claim | 1 | 0.21% |
| record | claim_amount: Total cost of the medical service billed | 1 | 0.21% |
| record | patient_address: Full address of the patient | 1 | 0.21% |
| record | physical_characteristics.observations.0.telescope.location | 1 | 0.21% |
| record | discovery_details.observatory.location.country | 1 | 0.21% |
| record | discovery_details.observatory.location.county | 1 | 0.21% |
| record | asteroid_belt | 1 | 0.21% |
| record | orbital_characteristics.jpl_classification | 1 | 0.21% |
| record | asteroid_name | 1 | 0.21% |
| record | budget_remaining | 1 | 0.21% |
| record | actual_spend | 1 | 0.21% |
| record | remaining_budget | 1 | 0.21% |
| record | onboarding_sprint_completion | 1 | 0.21% |
| record | risk_code | 1 | 0.21% |
| record | backend_integration_status | 1 | 0.21% |
| record | street | 1 | 0.21% |
| record | sessions | 1 | 0.21% |
| record | dateofbirth | 1 | 0.21% |
| record | dose_number | 1 | 0.21% |
| record | start_date | 1 | 0.21% |
| record | neighborhood | 1 | 0.21% |
| record | fiscalyearend | 1 | 0.21% |
| record | genus.subspecies.0.name | 1 | 0.21% |
| record | block | 1 | 0.21% |
| record | date_sent | 1 | 0.21% |
| record | message_id | 1 | 0.21% |
| record | outreach_status | 1 | 0.21% |
| record | job_title | 1 | 0.21% |
| record | currentheadcountapex | 1 | 0.21% |
| record | marketing_budget | 1 | 0.21% |
| record | iporevisedvaluation | 1 | 0.21% |
| record | hotel.location.district | 1 | 0.21% |
| record | hotel.name | 1 | 0.21% |
| record | hotel.location.nearby_attractions | 1 | 0.21% |
| record | hotel.room_features | 1 | 0.21% |
| record | hotel.amenities | 1 | 0.21% |
| record | hotel.hotel_group | 1 | 0.21% |
| record | page_count: Number of pages in the paper | 1 | 0.21% |
| record | abstract_length: Length of the abstract in words | 1 | 0.21% |
| record | review_date: Date when the paper was reviewed | 1 | 0.21% |
| record | conference_name: Name of the conference where submitted | 1 | 0.21% |
| record | funding_source: Source of research funding | 1 | 0.21% |
| record | word_count: Total number of words in the paper | 1 | 0.21% |
| record | submission_date: Date when the paper was submitted | 1 | 0.21% |
| record | affiliation: Institutional affiliation of the author | 1 | 0.21% |
| record | fieldofstudy: Academic domain of the research | 1 | 0.21% |
| record | series_record | 1 | 0.21% |
| record | training_program.invitation_steps | 1 | 0.21% |
| record | training_program.payment.options | 1 | 0.21% |
| record | organization.name | 1 | 0.21% |
| record | training_program.payment.payment_instructions | 1 | 0.21% |
| record | training_program.payment.deposit_amount | 1 | 0.21% |
| record | lastdosedate | 1 | 0.21% |
| record | vaccination_id | 1 | 0.21% |
| record | problemDescription | 1 | 0.21% |
| record | vehicle.componentReplacements | 1 | 0.21% |
| record | postInformation.date | 1 | 0.21% |
| record | potentialSolutions | 1 | 0.21% |
| record | vehicle.make | 1 | 0.21% |
| record | lead_person | 1 | 0.21% |
| record | radiologist | 1 | 0.21% |
| record | ordering_physician | 1 | 0.21% |
| record | protocol | 1 | 0.21% |
| record | exam_type | 1 | 0.21% |
| record | det | 1 | 0.21% |
| record | appos | 1 | 0.21% |
| record | aux | 1 | 0.21% |
| record | ccomp | 1 | 0.21% |
| record | amod | 1 | 0.21% |
| record | acl | 1 | 0.21% |
| record | pobj | 1 | 0.21% |
| record | recipient_organization | 1 | 0.21% |
| record | donor_name | 1 | 0.21% |
| record | establishment_year | 1 | 0.21% |
| record | staff.teaching_staff_count | 1 | 0.21% |
| record | faculty_name | 1 | 0.21% |
| record | map_location | 1 | 0.21% |
| record | campus | 1 | 0.21% |
| record | departments | 1 | 0.21% |
| record | dean.name | 1 | 0.21% |
| record | region.location.extends_to | 1 | 0.21% |
| record | region.location.includes | 1 | 0.21% |
| record | region.landscape | 1 | 0.21% |
| record | isbn | 1 | 0.21% |
| record | publication_volume | 1 | 0.21% |
| record | publication_issue | 1 | 0.21% |
| record | match | 1 | 0.21% |
| record | product_name | 1 | 0.21% |
| record | model_number | 1 | 0.21% |
| record | version_number | 1 | 0.21% |
| record | enforceability_status | 1 | 0.21% |
| record | modification_flag | 1 | 0.21% |
| record | document_type | 1 | 0.21% |
| record | effective_date | 1 | 0.21% |
| record | breach_penalty_percent | 1 | 0.21% |
| record | compliance_code | 1 | 0.21% |
| record | audit_review_date | 1 | 0.21% |
| record | legal_agent | 1 | 0.21% |
| record | contract_reference | 1 | 0.21% |
| record | notice_period_days | 1 | 0.21% |
| record | term_start_date | 1 | 0.21% |
| record | clause_description | 1 | 0.21% |
| record | amount_usd | 1 | 0.21% |
| record | counterparty_address | 1 | 0.21% |
| record | party_name | 1 | 0.21% |
| record | copyright | 1 | 0.21% |
| record | drexel_info | 1 | 0.21% |
| record | browse_topics | 1 | 0.21% |
| record | footer_links | 1 | 0.21% |
| record | faq_references | 1 | 0.21% |
| record | ticket_id: Unique identifier for the support case | 1 | 0.21% |
| record | user: Name of the user who filed the ticket | 1 | 0.21% |
| record | user_name | 1 | 0.21% |
| record | debttoebitda_ratio: Ratio of debt to EBITDA indicating leverage | 1 | 0.21% |
| record | funding_amount: Total amount raised in millions of dollars | 1 | 0.21% |
| record | co_investors: List of additional investors participating in the round | 1 | 0.21% |
| record | focus_area: Primary technological or business focus of the startup | 1 | 0.21% |
| record | investor_count: Number of investors involved in the round | 1 | 0.21% |
| record | ratingcategory | 1 | 0.21% |
| record | message_id: Unique identifier for each outreach message | 1 | 0.21% |
| record | sender: Name of the recruiter or agency sending the message | 1 | 0.21% |
| record | recipient: Name of the recipient at the target company | 1 | 0.21% |
| record | company: Name of the company where the recipient works | 1 | 0.21% |
| record | role: Job title or position being targeted | 1 | 0.21% |
| record | subject: Title or subject line of the outreach message | 1 | 0.21% |
| record | outreach_amount: Monetary value associated with the outreach | 1 | 0.21% |
| record | item_description.setup_charge.amount | 1 | 0.21% |
| record | item_description.shipping.ready_to_ship_in.quantity | 1 | 0.21% |
| record | item_description.features | 1 | 0.21% |
| record | item_description.shipping.rush_details | 1 | 0.21% |
| record | item_description.item_id | 1 | 0.21% |
| record | item_description.weight.value | 1 | 0.21% |
| record | item_description.shipping.ready_to_ship_in.unit | 1 | 0.21% |
| record | item_description.name | 1 | 0.21% |
| record | return_id | 1 | 0.21% |
| record | tech_product | 1 | 0.21% |
| record | replies | 1 | 0.21% |
| record | author_handle | 1 | 0.21% |
| record | address: Physical address of the appointee | 1 | 0.21% |
| record | notes: Additional context about the appointment or background | 1 | 0.21% |
| record | name: Full name of the appointed individual | 1 | 0.21% |
| record | date: Start date of the appointment | 1 | 0.21% |
| record | department: Government department or agency where the role is held | 1 | 0.21% |
| record | position: Job title or role within the department | 1 | 0.21% |
| record | project_id | 1 | 0.21% |
| record | open_issues | 1 | 0.21% |
| record | submission_date | 1 | 0.21% |
| record | games_behind | 1 | 0.21% |
| record | wins | 1 | 0.21% |
| record | games_played | 1 | 0.21% |
| record | losses | 1 | 0.21% |
| record | party | 1 | 0.21% |
| record | vote_percentage | 1 | 0.21% |
| record | marginofvictory | 1 | 0.21% |
| record | ballot_type | 1 | 0.21% |
| record | electoral_code | 1 | 0.21% |
| record | product | 1 | 0.21% |
| record | model | 1 | 0.21% |
| record | review_date | 1 | 0.21% |
| record | review_length | 1 | 0.21% |
| record | release_year | 1 | 0.21% |
| record | warranty_period | 1 | 0.21% |
| record | compatibility | 1 | 0.21% |
| record | power_consumption | 1 | 0.21% |
| record | charging_time | 1 | 0.21% |
| record | storage_capacity | 1 | 0.21% |
| record | use_case | 1 | 0.21% |
| record | blood_pressure | 1 | 0.21% |
| record | affiliation.location | 1 | 0.21% |
| record | research_background.focus_area | 1 | 0.21% |
| record | match_round: Stage of the tournament such as final, quarterfinal, or last 16 | 1 | 0.21% |
| record | score: Final score of the match in set format | 1 | 0.21% |
| record | venue_name: Name of the specific tournament venue | 1 | 0.21% |
| record | location: Venue where the match was held | 1 | 0.21% |
| record | winner: Player who won the match | 1 | 0.21% |
| record | unit_location | 1 | 0.21% |
| record | rental_id | 1 | 0.21% |
| record | board_approval_date | 1 | 0.21% |
| record | expected_close_date | 1 | 0.21% |
| record | ceo | 1 | 0.21% |
| record | employees | 1 | 0.21% |
| record | stock_symbol | 1 | 0.21% |
| record | certification.prerequisites | 1 | 0.21% |
| record | organization.website | 1 | 0.21% |
| record | personnel.0.name | 1 | 0.21% |
| record | certification.exams.0.name | 1 | 0.21% |
| record | certification.exams.1.name | 1 | 0.21% |
| record | additional_information_url | 1 | 0.21% |
| record | certification.name | 1 | 0.21% |
| record | exam_centers | 1 | 0.21% |
| record | certification.description | 1 | 0.21% |
| record | personnel.0.organization | 1 | 0.21% |
| record | personnel.0.title | 1 | 0.21% |
| record | training.delivery_method | 1 | 0.21% |
| record | basin_name | 1 | 0.21% |
| record | recharge_amount | 1 | 0.21% |
| record | canton_name | 1 | 0.21% |
| record | canton_name_french | 1 | 0.21% |
| record | municipalities | 1 | 0.21% |
| record | promotions | 1 | 0.21% |
| record | division | 1 | 0.21% |
| record | app_name | 1 | 0.21% |
| record | paymentduedate | 1 | 0.21% |
| record | vendorinvoiceemail | 1 | 0.21% |
| record | account_code | 1 | 0.21% |
| record | name: Full name of the startup company | 1 | 0.21% |
| record | lead_investor: Primary investor leading the round | 1 | 0.21% |
| record | other_investors: List of additional participating investors | 1 | 0.21% |
| record | icd10code | 1 | 0.21% |
| record | patient_province | 1 | 0.21% |
| record | followupaction | 1 | 0.21% |
| record | reporting_doctor | 1 | 0.21% |
| record | investors | 1 | 0.21% |
| record | useoffunds | 1 | 0.21% |
| record | bp | 1 | 0.21% |
| record | doctor | 1 | 0.21% |
| record | rr | 1 | 0.21% |
| record | case_number: Unique identifier for the court case | 1 | 0.21% |
| record | plaintiff: Individual or entity initiating the lawsuit | 1 | 0.21% |
| record | defendant: Individual or entity being sued | 1 | 0.21% |
| record | damages: Monetary amount sought in damages | 1 | 0.21% |
| record | party_a | 1 | 0.21% |
| record | party_b | 1 | 0.21% |
| record | delivery_method | 1 | 0.21% |
| record | newsletter_name | 1 | 0.21% |
| record | subscription_amount | 1 | 0.21% |
| record | pricepernight | 1 | 0.21% |
| record | briefing_id | 1 | 0.21% |
| record | lead_reporter: Reporter providing coverage of the incident | 1 | 0.21% |
| record | title: Event title describing the nature and context of the incident | 1 | 0.21% |
| record | location: Geographic location where the incident took place | 1 | 0.21% |
| record | impact: Description of human, financial, or systemic consequences | 1 | 0.21% |
| record | date: Timestamp when the incident occurred | 1 | 0.21% |
| record | policyholder_name | 1 | 0.21% |
| record | policytype | 1 | 0.21% |
| record | exploited_ports | 1 | 0.21% |
| record | software_versions | 1 | 0.21% |
| record | isps | 1 | 0.21% |
| record | person1 | 1 | 0.21% |
| record | hotel.transportation.parking.0.name | 1 | 0.21% |
| record | hotel.location.city | 1 | 0.21% |
| record | hotel.star_rating | 1 | 0.21% |
| record | hotel.transportation.driving_directions | 1 | 0.21% |
| record | hotel.transportation.airport | 1 | 0.21% |
| record | hotel.location.country | 1 | 0.21% |
| record | hotel.transportation.parking.0.walking_time | 1 | 0.21% |
| record | hotel.location.nearby_landmarks | 1 | 0.21% |
| record | hotel.transportation.parking.1.cost | 1 | 0.21% |
| record | hotel.location.postal_code | 1 | 0.21% |
| record | hotel.location.area_description | 1 | 0.21% |
| record | hotel.transportation.parking.1.name | 1 | 0.21% |
| record | hotel.transportation.parking.1.walking_time | 1 | 0.21% |
| record | hotel.contact.phone | 1 | 0.21% |
| record | hotel.transportation.parking.0.cost | 1 | 0.21% |
| record | species.family | 1 | 0.21% |
| record | content.core_beliefs.relationship_with_society | 1 | 0.21% |
| record | content.elements.name | 1 | 0.21% |
| record | content.elements.music.bands | 1 | 0.21% |
| record | content.elements.cemeteries | 1 | 0.21% |
| record | content.warning | 1 | 0.21% |
| record | asteroid.named_after.location.country | 1 | 0.21% |
| record | asteroid.named_after.location.province | 1 | 0.21% |
| record | discovery.observatory | 1 | 0.21% |
| record | discovery.astronomer.nationality | 1 | 0.21% |
| record | followupdate | 1 | 0.21% |
| record | dividend_yield | 1 | 0.21% |
| record | ebitda_margin | 1 | 0.21% |
| record | auditstatus | 1 | 0.21% |
| record | sector | 1 | 0.21% |
| record | vaccination_status | 1 | 0.21% |
| record | symbol | 1 | 0.21% |
| record | performance_reason | 1 | 0.21% |
| record | change | 1 | 0.21% |
| record | meeting_id | 1 | 0.21% |
| record | location.distance_from_city_center.value | 1 | 0.21% |
| record | location.distance_from_city_center.unit | 1 | 0.21% |
| record | location.distance_from_airport.value | 1 | 0.21% |
| record | location.distance_from_airport.unit | 1 | 0.21% |
| record | location.airport_travel_time.value | 1 | 0.21% |
| record | location.nearby_attractions | 1 | 0.21% |
| record | hotel_details.star_rating | 1 | 0.21% |
| record | hotel_details.year_built | 1 | 0.21% |
| record | hotel_details.year_renovated | 1 | 0.21% |
| record | hotel_details.check_in | 1 | 0.21% |
| record | hotel_details.check_out | 1 | 0.21% |
| record | hotel_details.reception_open_until | 1 | 0.21% |
| record | hotel_details.number_of_rooms | 1 | 0.21% |
| record | hotel_details.room_amenities | 1 | 0.21% |
| record | hotel_details.recreation | 1 | 0.21% |
| record | child_policy.infants.age_range | 1 | 0.21% |
| record | child_policy.children.age_range | 1 | 0.21% |
| record | child_policy.adult_age | 1 | 0.21% |
| record | child_policy.extra_bed_policy | 1 | 0.21% |
| record | booking_policy.multiple_rooms | 1 | 0.21% |
| record | property_type | 1 | 0.21% |
| record | check_in_process | 1 | 0.21% |
| record | owner_contact | 1 | 0.21% |
| record | director | 1 | 0.21% |
| record | route_name | 1 | 0.21% |
| record | property_id: Unique identifier assigned to the rental unit | 1 | 0.21% |
| record | city: City where the property is located | 1 | 0.21% |
| record | lease_term: Type of lease term, e.g., month-to-month | 1 | 0.21% |
| record | management_contact: Name or entity managing the property | 1 | 0.21% |
| record | location: Physical address where course is held | 1 | 0.21% |
| record | duration: Number of weeks the course runs | 1 | 0.21% |
| record | credits: Number of academic credits assigned to the course | 1 | 0.21% |
| record | cost: Total cost of enrollment in US dollars | 1 | 0.21% |
| record | research.experiment_details.quantum_state | 1 | 0.21% |
| record | research.focus | 1 | 0.21% |
| record | research.researcher.affiliation.name | 1 | 0.21% |
| record | research.experiment_details.setup | 1 | 0.21% |
| record | research.objective | 1 | 0.21% |
| record | research.experiment_details.outcome | 1 | 0.21% |
| record | research.materials | 1 | 0.21% |
| record | research.researcher.affiliation.abbreviation | 1 | 0.21% |
| record | research.temperature | 1 | 0.21% |
| record | research.experiment_details.temperature_details | 1 | 0.21% |
| record | research.researcher.name | 1 | 0.21% |
| record | chemical_name | 1 | 0.21% |
| record | cas_number | 1 | 0.21% |
| record | emergency_contact | 1 | 0.21% |
| record | buyer: Name of the customer who made the purchase | 1 | 0.21% |
| record | location: City and state where buyer resides | 1 | 0.21% |
| record | product: Product model identifier such as EBC-1234 or WAT-5678 | 1 | 0.21% |
| record | price: Price paid for the product in US dollars | 1 | 0.21% |
| record | rating: Star rating from 1 to 5 given by the buyer | 1 | 0.21% |
| record | shipping_method: Type of shipping used | 1 | 0.21% |
| record | tracking_number: Shipping tracking number assigned | 1 | 0.21% |
| record | order_id: Unique order identifier from the retailer | 1 | 0.21% |
| record | return_status: Whether the product was returned or kept | 1 | 0.21% |
| record | feedback: Additional comments or issues noted in review | 1 | 0.21% |
| record | numerical_values | 1 | 0.21% |
| record | adverb_modifier | 1 | 0.21% |
| record | adjectival_modifier | 1 | 0.21% |
| record | technical_terms | 1 | 0.21% |
| record | events | 1 | 0.21% |
| record | company: Name of the technology company raising funds | 1 | 0.21% |
| record | lead_investor: List of lead investors or funds involved | 1 | 0.21% |
| record | company_id | 1 | 0.21% |
| record | record_metadata | natural/status ×20, natural/name ×10, natural/location ×10, natural/patient_id ×7 |  |
| research_paper | title | 5 | 100.00% |
| research_paper | authors | 5 | 100.00% |
| research_paper | venue | 5 | 100.00% |
| research_paper | year | 4 | 80.00% |
| research_paper | topic | 2 | 40.00% |
| research_paper | record_metadata | natural/title ×5 |  |
| transaction | buyer | 46 | 95.83% |
| transaction | date | 44 | 91.67% |
| transaction | item | 43 | 89.58% |
| transaction | amount | 43 | 89.58% |
| transaction | seller | 43 | 89.58% |
| transaction | record_metadata | natural/item ×43, natural/amount ×3 |  |
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
