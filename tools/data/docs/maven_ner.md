# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| maven_ner | train | 2,623 | 2,623 | 59,143 | 0 | 0 | 0 | 0 | 0 |
| maven_ner | val | 145 | 145 | 3,045 | 0 | 0 | 0 | 0 | 0 |
| maven_ner | test | 145 | 145 | 3,507 | 0 | 0 | 0 | 0 | 0 |

## maven_ner

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2,623 | 2,623 | 254 | 496 | 1754 | 59,143 | 0 | 0 | 0 | 0 | 0 |
| val | 145 | 145 | 229 | 446 | 877 | 3,045 | 0 | 0 | 0 | 0 | 0 |
| test | 145 | 145 | 260 | 576 | 1017 | 3,507 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 2,623 | 145 | 145 | 2,913 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 145 |
| test_unique_inputs | 145 |
| train_records | 2,623 |
| train_unique_inputs | 2,623 |
| val_records | 145 |
| val_unique_inputs | 145 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0876 | 59,143 / 3,045 | Motion 3%→3%, Destroying 2%→2%, Attack 3%→3%, Conquering 2%→2%, Hostile_encounter 3%→4% |
| entities (per mention) | train / test | 0.0894 | 59,143 / 3,507 | Competition 2%→2%, Process_end 2%→2%, Social_event 2%→1%, Conquering 2%→1%, Cause_to_be_included 1%→1% |
| entities (per mention) | val / test | 0.1324 | 3,045 / 3,507 | Competition 3%→2%, Conquering 2%→1%, Social_event 2%→1%, Destroying 2%→1%, Hostile_encounter 4%→3% |

Thin support in val, entities (per mention): 75 label(s) under 10 gold. Theft ×1, Change_tool ×1, Submitting_documents ×1, Employment ×1, Wearing ×1, Scouring ×1, Hiding_objects ×1, Create_artwork ×1, Renting ×1, Risk ×1, Education_teaching ×1, Justifying ×1, Bearing_arms ×2, Having_or_lacking_access ×2, Reveal_secret ×2, Labeling ×2, Practice ×2, Commerce_buy ×2, Containing ×2, Terrorism ×2, Cost ×2, Rewards_and_punishments ×3, Prison ×3, Adducing ×3, Robbery ×3, Filling ×3, Body_movement ×3, Award ×3, Kidnapping ×3, Releasing ×3, Exchange ×3, Legality ×3, Expend_resource ×4, Reforming_a_system ×4, Carry_goods ×4, Cure ×4, Patrolling ×4, Lighting ×4, Limiting ×4, Change_sentiment ×4, Legal_rulings ×5, Commitment ×5, Check ×5, Commerce_pay ×5, Connect ×5, Writing ×5, GetReady ×6, Committing_crime ×6, Imposing_obligation ×6, Forming_relationships ×6, Telling ×6, Convincing ×6, Confronting_problem ×6, Revenge ×6, Quarreling ×7, Arrest ×7, Sign_agreement ×7, Commerce_sell ×7, Emptying ×7, Surrounding ×7, Besieging ×7, Response ×7, Protest ×7, Testing ×8, Publishing ×8, Hindering ×8, Openness ×8, Criminal_investigation ×9, Violence ×9, Coming_to_believe ×9, Surrendering ×9, Rescuing ×9, Resolve_problem ×9, Warning ×9, Collaboration ×9.
Thin support in test, entities (per mention): 64 label(s) under 10 gold. Commerce_buy ×1, Vocalizations ×1, Labeling ×1, Suspicion ×1, Reforming_a_system ×1, Justifying ×1, Having_or_lacking_access ×1, Carry_goods ×1, Award ×1, Revenge ×1, Renting ×1, Patrolling ×2, Legality ×2, Prison ×2, Cost ×2, Robbery ×2, Practice ×2, Limiting ×2, Education_teaching ×3, Cure ×3, GetReady ×3, Releasing ×3, Kidnapping ×3, Exchange ×3, Commitment ×3, Protest ×4, Filling ×4, Bearing_arms ×4, Committing_crime ×4, Rite ×4, Testing ×4, Research ×5, Adducing ×5, Imposing_obligation ×5, Rescuing ×5, Ratification ×5, Expend_resource ×5, Commerce_pay ×5, Being_in_operation ×5, Submitting_documents ×5, Surrendering ×6, Quarreling ×6, Forming_relationships ×6, Containing ×6, Create_artwork ×6, Writing ×6, Resolve_problem ×6, Connect ×6, Wearing ×6, Sign_agreement ×6, Emptying ×7, Commerce_sell ×7, Surrounding ×7, Reveal_secret ×7, Temporary_stay ×7, GiveUp ×8, Body_movement ×8, Come_together ×8, Rewards_and_punishments ×8, Legal_rulings ×9, Warning ×9, Violence ×9, Hiding_objects ×9, Change_event_time ×9.

### Events

No event gold.

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

No event arguments.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

No trigger or argument surfaces.

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

No event instances.

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

No event lists more than one trigger, and no document has two events of one type, so there is no trigger pair.

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

No event arguments.

### Entities

#### train

| label | count | share |
| --- | --- | --- |
| Causation | 2,107 | 3.56% |
| Process_start | 2,071 | 3.50% |
| Attack | 1,950 | 3.30% |
| Hostile_encounter | 1,851 | 3.13% |
| Motion | 1,784 | 3.02% |
| Catastrophe | 1,389 | 2.35% |
| Competition | 1,368 | 2.31% |
| Statement | 1,142 | 1.93% |
| Process_end | 1,138 | 1.92% |
| Killing | 1,102 | 1.86% |
| Conquering | 1,100 | 1.86% |
| Self_motion | 1,056 | 1.79% |
| Arriving | 1,007 | 1.70% |
| Destroying | 963 | 1.63% |
| Social_event | 932 | 1.58% |
| Coming_to_be | 928 | 1.57% |
| Bodily_harm | 857 | 1.45% |
| Creating | 789 | 1.33% |
| Cause_change_of_position_on_a_scale | 785 | 1.33% |
| Cause_change_of_strength | 768 | 1.30% |
| Death | 765 | 1.29% |
| Earnings_and_losses | 713 | 1.21% |
| Control | 710 | 1.20% |
| Getting | 684 | 1.16% |
| Becoming | 676 | 1.14% |
| Preventing_or_letting | 669 | 1.13% |
| Military_operation | 666 | 1.13% |
| Know | 653 | 1.10% |
| Arranging | 653 | 1.10% |
| Hold | 647 | 1.09% |
| Presence | 643 | 1.09% |
| Damaging | 616 | 1.04% |
| Escaping | 614 | 1.04% |
| Defending | 595 | 1.01% |
| Change | 586 | 0.99% |
| Action | 585 | 0.99% |
| Participation | 567 | 0.96% |
| Placing | 566 | 0.96% |
| Cause_to_be_included | 559 | 0.95% |
| Motion_directional | 556 | 0.94% |
| Influence | 518 | 0.88% |
| Judgment_communication | 514 | 0.87% |
| Request | 512 | 0.87% |
| Name_conferral | 501 | 0.85% |
| Change_of_leadership | 487 | 0.82% |
| Expressing_publicly | 486 | 0.82% |
| Giving | 482 | 0.81% |
| Removing | 445 | 0.75% |
| Agree_or_refuse_to_act | 424 | 0.72% |
| Recording | 422 | 0.71% |
| Using | 410 | 0.69% |
| Supporting | 410 | 0.69% |
| Supply | 407 | 0.69% |
| Communication | 382 | 0.65% |
| Traveling | 365 | 0.62% |
| Choosing | 354 | 0.60% |
| Sending | 353 | 0.60% |
| Bringing | 345 | 0.58% |
| Assistance | 339 | 0.57% |
| Reporting | 327 | 0.55% |
| Departing | 326 | 0.55% |
| Perception_active | 294 | 0.50% |
| Building | 293 | 0.50% |
| Deciding | 293 | 0.50% |
| Cause_to_make_progress | 292 | 0.49% |
| Dispersal | 288 | 0.49% |
| Use_firearm | 276 | 0.47% |
| Come_together | 275 | 0.46% |
| Manufacturing | 265 | 0.45% |
| Hindering | 253 | 0.43% |
| Aiming | 249 | 0.42% |
| Cause_to_amalgamate | 248 | 0.42% |
| Receiving | 242 | 0.41% |
| Arrest | 242 | 0.41% |
| Response | 240 | 0.41% |
| Achieve | 236 | 0.40% |
| Recovering | 232 | 0.39% |
| Expansion | 223 | 0.38% |
| Change_event_time | 221 | 0.37% |
| Violence | 220 | 0.37% |
| Check | 206 | 0.35% |
| Scrutiny | 185 | 0.31% |
| Becoming_a_member | 181 | 0.31% |
| Openness | 180 | 0.30% |
| Connect | 179 | 0.30% |
| Besieging | 178 | 0.30% |
| Committing_crime | 175 | 0.30% |
| Coming_to_believe | 172 | 0.29% |
| Collaboration | 172 | 0.29% |
| Publishing | 167 | 0.28% |
| Surrendering | 166 | 0.28% |
| Criminal_investigation | 164 | 0.28% |
| Being_in_operation | 164 | 0.28% |
| Terrorism | 158 | 0.27% |
| Legal_rulings | 157 | 0.27% |
| Confronting_problem | 155 | 0.26% |
| Preserving | 146 | 0.25% |
| Warning | 146 | 0.25% |
| Sign_agreement | 144 | 0.24% |
| Quarreling | 143 | 0.24% |
| Temporary_stay | 143 | 0.24% |
| Convincing | 140 | 0.24% |
| GiveUp | 136 | 0.23% |
| Change_sentiment | 134 | 0.23% |
| Telling | 118 | 0.20% |
| Legality | 115 | 0.19% |
| Commitment | 113 | 0.19% |
| Protest | 110 | 0.19% |
| Resolve_problem | 110 | 0.19% |
| Employment | 108 | 0.18% |
| Rescuing | 106 | 0.18% |
| Body_movement | 101 | 0.17% |
| Emptying | 98 | 0.17% |
| Releasing | 97 | 0.16% |
| Commerce_sell | 95 | 0.16% |
| Education_teaching | 95 | 0.16% |
| GetReady | 91 | 0.15% |
| Hiding_objects | 88 | 0.15% |
| Filling | 88 | 0.15% |
| Reveal_secret | 87 | 0.15% |
| Surrounding | 83 | 0.14% |
| Writing | 81 | 0.14% |
| Forming_relationships | 79 | 0.13% |
| Testing | 79 | 0.13% |
| Expend_resource | 79 | 0.13% |
| Commerce_pay | 78 | 0.13% |
| Bearing_arms | 77 | 0.13% |
| Cost | 77 | 0.13% |
| Reforming_a_system | 75 | 0.13% |
| Patrolling | 71 | 0.12% |
| Rewards_and_punishments | 71 | 0.12% |
| Limiting | 68 | 0.11% |
| Adducing | 68 | 0.11% |
| Wearing | 63 | 0.11% |
| Cure | 60 | 0.10% |
| Research | 59 | 0.10% |
| Kidnapping | 57 | 0.10% |
| Having_or_lacking_access | 56 | 0.09% |
| Create_artwork | 55 | 0.09% |
| Exchange | 54 | 0.09% |
| Containing | 47 | 0.08% |
| Robbery | 47 | 0.08% |
| Imposing_obligation | 47 | 0.08% |
| Commerce_buy | 46 | 0.08% |
| Prison | 44 | 0.07% |
| Revenge | 43 | 0.07% |
| Award | 42 | 0.07% |
| Submitting_documents | 40 | 0.07% |
| Carry_goods | 39 | 0.07% |
| Rite | 37 | 0.06% |
| Labeling | 32 | 0.05% |
| Scouring | 30 | 0.05% |
| Practice | 30 | 0.05% |
| Lighting | 30 | 0.05% |
| Justifying | 25 | 0.04% |
| Ratification | 19 | 0.03% |
| Suspicion | 18 | 0.03% |
| Vocalizations | 18 | 0.03% |
| Theft | 17 | 0.03% |
| Ingestion | 16 | 0.03% |
| Renting | 15 | 0.03% |
| Incident | 13 | 0.02% |
| Extradition | 11 | 0.02% |
| Institutionalization | 11 | 0.02% |
| Risk | 9 | 0.02% |
| Breathing | 6 | 0.01% |
| Emergency | 6 | 0.01% |
| Change_tool | 3 | 0.01% |

#### val

| label | count | share |
| --- | --- | --- |
| Causation | 115 | 3.78% |
| Hostile_encounter | 108 | 3.55% |
| Process_start | 102 | 3.35% |
| Attack | 87 | 2.86% |
| Competition | 82 | 2.69% |
| Catastrophe | 79 | 2.59% |
| Motion | 77 | 2.53% |
| Conquering | 70 | 2.30% |
| Destroying | 63 | 2.07% |
| Process_end | 61 | 2.00% |
| Social_event | 60 | 1.97% |
| Statement | 57 | 1.87% |
| Arriving | 54 | 1.77% |
| Killing | 54 | 1.77% |
| Self_motion | 46 | 1.51% |
| Earnings_and_losses | 45 | 1.48% |
| Know | 42 | 1.38% |
| Cause_change_of_position_on_a_scale | 39 | 1.28% |
| Getting | 39 | 1.28% |
| Damaging | 38 | 1.25% |
| Bodily_harm | 38 | 1.25% |
| Creating | 38 | 1.25% |
| Coming_to_be | 38 | 1.25% |
| Death | 37 | 1.22% |
| Defending | 37 | 1.22% |
| Control | 37 | 1.22% |
| Preventing_or_letting | 36 | 1.18% |
| Hold | 36 | 1.18% |
| Cause_to_be_included | 34 | 1.12% |
| Name_conferral | 33 | 1.08% |
| Action | 33 | 1.08% |
| Arranging | 33 | 1.08% |
| Becoming | 32 | 1.05% |
| Influence | 32 | 1.05% |
| Change | 32 | 1.05% |
| Cause_change_of_strength | 31 | 1.02% |
| Participation | 30 | 0.99% |
| Presence | 29 | 0.95% |
| Military_operation | 28 | 0.92% |
| Recording | 28 | 0.92% |
| Giving | 28 | 0.92% |
| Change_of_leadership | 26 | 0.85% |
| Judgment_communication | 25 | 0.82% |
| Request | 25 | 0.82% |
| Supply | 25 | 0.82% |
| Traveling | 24 | 0.79% |
| Motion_directional | 24 | 0.79% |
| Placing | 23 | 0.76% |
| Removing | 23 | 0.76% |
| Expressing_publicly | 23 | 0.76% |
| Perception_active | 20 | 0.66% |
| Deciding | 20 | 0.66% |
| Escaping | 20 | 0.66% |
| Bringing | 19 | 0.62% |
| Recovering | 18 | 0.59% |
| Using | 17 | 0.56% |
| Choosing | 17 | 0.56% |
| Communication | 17 | 0.56% |
| Departing | 16 | 0.53% |
| Cause_to_make_progress | 16 | 0.53% |
| Achieve | 15 | 0.49% |
| Expansion | 15 | 0.49% |
| Scrutiny | 15 | 0.49% |
| Agree_or_refuse_to_act | 14 | 0.46% |
| Change_event_time | 14 | 0.46% |
| Building | 14 | 0.46% |
| Reporting | 14 | 0.46% |
| Becoming_a_member | 14 | 0.46% |
| Sending | 14 | 0.46% |
| Assistance | 13 | 0.43% |
| Manufacturing | 13 | 0.43% |
| Cause_to_amalgamate | 12 | 0.39% |
| Supporting | 12 | 0.39% |
| Use_firearm | 12 | 0.39% |
| GiveUp | 12 | 0.39% |
| Dispersal | 12 | 0.39% |
| Temporary_stay | 11 | 0.36% |
| Research | 11 | 0.36% |
| Aiming | 11 | 0.36% |
| Being_in_operation | 11 | 0.36% |
| Preserving | 10 | 0.33% |
| Receiving | 10 | 0.33% |
| Come_together | 10 | 0.33% |
| Criminal_investigation | 9 | 0.30% |
| Violence | 9 | 0.30% |
| Coming_to_believe | 9 | 0.30% |
| Surrendering | 9 | 0.30% |
| Rescuing | 9 | 0.30% |
| Resolve_problem | 9 | 0.30% |
| Warning | 9 | 0.30% |
| Collaboration | 9 | 0.30% |
| Testing | 8 | 0.26% |
| Publishing | 8 | 0.26% |
| Hindering | 8 | 0.26% |
| Openness | 8 | 0.26% |
| Quarreling | 7 | 0.23% |
| Arrest | 7 | 0.23% |
| Sign_agreement | 7 | 0.23% |
| Commerce_sell | 7 | 0.23% |
| Emptying | 7 | 0.23% |
| Surrounding | 7 | 0.23% |
| Besieging | 7 | 0.23% |
| Response | 7 | 0.23% |
| Protest | 7 | 0.23% |
| GetReady | 6 | 0.20% |
| Committing_crime | 6 | 0.20% |
| Imposing_obligation | 6 | 0.20% |
| Forming_relationships | 6 | 0.20% |
| Telling | 6 | 0.20% |
| Convincing | 6 | 0.20% |
| Confronting_problem | 6 | 0.20% |
| Revenge | 6 | 0.20% |
| Legal_rulings | 5 | 0.16% |
| Commitment | 5 | 0.16% |
| Check | 5 | 0.16% |
| Commerce_pay | 5 | 0.16% |
| Connect | 5 | 0.16% |
| Writing | 5 | 0.16% |
| Expend_resource | 4 | 0.13% |
| Reforming_a_system | 4 | 0.13% |
| Carry_goods | 4 | 0.13% |
| Cure | 4 | 0.13% |
| Patrolling | 4 | 0.13% |
| Lighting | 4 | 0.13% |
| Limiting | 4 | 0.13% |
| Change_sentiment | 4 | 0.13% |
| Rewards_and_punishments | 3 | 0.10% |
| Prison | 3 | 0.10% |
| Adducing | 3 | 0.10% |
| Robbery | 3 | 0.10% |
| Filling | 3 | 0.10% |
| Body_movement | 3 | 0.10% |
| Award | 3 | 0.10% |
| Kidnapping | 3 | 0.10% |
| Releasing | 3 | 0.10% |
| Exchange | 3 | 0.10% |
| Legality | 3 | 0.10% |
| Bearing_arms | 2 | 0.07% |
| Having_or_lacking_access | 2 | 0.07% |
| Reveal_secret | 2 | 0.07% |
| Labeling | 2 | 0.07% |
| Practice | 2 | 0.07% |
| Commerce_buy | 2 | 0.07% |
| Containing | 2 | 0.07% |
| Terrorism | 2 | 0.07% |
| Cost | 2 | 0.07% |
| Theft | 1 | 0.03% |
| Change_tool | 1 | 0.03% |
| Submitting_documents | 1 | 0.03% |
| Employment | 1 | 0.03% |
| Wearing | 1 | 0.03% |
| Scouring | 1 | 0.03% |
| Hiding_objects | 1 | 0.03% |
| Create_artwork | 1 | 0.03% |
| Renting | 1 | 0.03% |
| Risk | 1 | 0.03% |
| Education_teaching | 1 | 0.03% |
| Justifying | 1 | 0.03% |

#### test

| label | count | share |
| --- | --- | --- |
| Causation | 138 | 3.93% |
| Process_start | 136 | 3.88% |
| Motion | 110 | 3.14% |
| Attack | 104 | 2.97% |
| Hostile_encounter | 102 | 2.91% |
| Process_end | 87 | 2.48% |
| Catastrophe | 78 | 2.22% |
| Arriving | 71 | 2.02% |
| Statement | 68 | 1.94% |
| Killing | 65 | 1.85% |
| Self_motion | 64 | 1.82% |
| Competition | 59 | 1.68% |
| Coming_to_be | 54 | 1.54% |
| Death | 52 | 1.48% |
| Creating | 52 | 1.48% |
| Earnings_and_losses | 49 | 1.40% |
| Cause_change_of_strength | 49 | 1.40% |
| Conquering | 48 | 1.37% |
| Arranging | 47 | 1.34% |
| Becoming | 45 | 1.28% |
| Escaping | 45 | 1.28% |
| Destroying | 44 | 1.25% |
| Control | 42 | 1.20% |
| Bodily_harm | 41 | 1.17% |
| Hold | 40 | 1.14% |
| Judgment_communication | 40 | 1.14% |
| Preventing_or_letting | 40 | 1.14% |
| Know | 39 | 1.11% |
| Cause_change_of_position_on_a_scale | 39 | 1.11% |
| Social_event | 38 | 1.08% |
| Defending | 38 | 1.08% |
| Action | 37 | 1.06% |
| Change | 37 | 1.06% |
| Change_of_leadership | 37 | 1.06% |
| Getting | 36 | 1.03% |
| Placing | 36 | 1.03% |
| Request | 34 | 0.97% |
| Damaging | 33 | 0.94% |
| Presence | 32 | 0.91% |
| Expressing_publicly | 31 | 0.88% |
| Traveling | 31 | 0.88% |
| Motion_directional | 31 | 0.88% |
| Agree_or_refuse_to_act | 31 | 0.88% |
| Supporting | 30 | 0.86% |
| Participation | 29 | 0.83% |
| Communication | 29 | 0.83% |
| Name_conferral | 28 | 0.80% |
| Reporting | 28 | 0.80% |
| Giving | 27 | 0.77% |
| Removing | 27 | 0.77% |
| Military_operation | 26 | 0.74% |
| Using | 25 | 0.71% |
| Scrutiny | 25 | 0.71% |
| Departing | 24 | 0.68% |
| Influence | 24 | 0.68% |
| Use_firearm | 24 | 0.68% |
| Recording | 23 | 0.66% |
| Hindering | 21 | 0.60% |
| Manufacturing | 20 | 0.57% |
| Bringing | 19 | 0.54% |
| Cause_to_be_included | 19 | 0.54% |
| Cause_to_make_progress | 19 | 0.54% |
| Expansion | 18 | 0.51% |
| Deciding | 18 | 0.51% |
| Collaboration | 18 | 0.51% |
| Choosing | 18 | 0.51% |
| Response | 17 | 0.48% |
| Recovering | 16 | 0.46% |
| Supply | 16 | 0.46% |
| Criminal_investigation | 16 | 0.46% |
| Building | 15 | 0.43% |
| Dispersal | 15 | 0.43% |
| Assistance | 15 | 0.43% |
| Arrest | 15 | 0.43% |
| Achieve | 14 | 0.40% |
| Cause_to_amalgamate | 14 | 0.40% |
| Check | 14 | 0.40% |
| Openness | 14 | 0.40% |
| Aiming | 13 | 0.37% |
| Perception_active | 13 | 0.37% |
| Coming_to_believe | 13 | 0.37% |
| Publishing | 12 | 0.34% |
| Convincing | 12 | 0.34% |
| Telling | 12 | 0.34% |
| Sending | 11 | 0.31% |
| Receiving | 11 | 0.31% |
| Employment | 11 | 0.31% |
| Besieging | 11 | 0.31% |
| Confronting_problem | 11 | 0.31% |
| Becoming_a_member | 10 | 0.29% |
| Preserving | 10 | 0.29% |
| Change_sentiment | 10 | 0.29% |
| Terrorism | 10 | 0.29% |
| Legal_rulings | 9 | 0.26% |
| Warning | 9 | 0.26% |
| Violence | 9 | 0.26% |
| Hiding_objects | 9 | 0.26% |
| Change_event_time | 9 | 0.26% |
| GiveUp | 8 | 0.23% |
| Body_movement | 8 | 0.23% |
| Come_together | 8 | 0.23% |
| Rewards_and_punishments | 8 | 0.23% |
| Emptying | 7 | 0.20% |
| Commerce_sell | 7 | 0.20% |
| Surrounding | 7 | 0.20% |
| Reveal_secret | 7 | 0.20% |
| Temporary_stay | 7 | 0.20% |
| Surrendering | 6 | 0.17% |
| Quarreling | 6 | 0.17% |
| Forming_relationships | 6 | 0.17% |
| Containing | 6 | 0.17% |
| Create_artwork | 6 | 0.17% |
| Writing | 6 | 0.17% |
| Resolve_problem | 6 | 0.17% |
| Connect | 6 | 0.17% |
| Wearing | 6 | 0.17% |
| Sign_agreement | 6 | 0.17% |
| Research | 5 | 0.14% |
| Adducing | 5 | 0.14% |
| Imposing_obligation | 5 | 0.14% |
| Rescuing | 5 | 0.14% |
| Ratification | 5 | 0.14% |
| Expend_resource | 5 | 0.14% |
| Commerce_pay | 5 | 0.14% |
| Being_in_operation | 5 | 0.14% |
| Submitting_documents | 5 | 0.14% |
| Protest | 4 | 0.11% |
| Filling | 4 | 0.11% |
| Bearing_arms | 4 | 0.11% |
| Committing_crime | 4 | 0.11% |
| Rite | 4 | 0.11% |
| Testing | 4 | 0.11% |
| Education_teaching | 3 | 0.09% |
| Cure | 3 | 0.09% |
| GetReady | 3 | 0.09% |
| Releasing | 3 | 0.09% |
| Kidnapping | 3 | 0.09% |
| Exchange | 3 | 0.09% |
| Commitment | 3 | 0.09% |
| Patrolling | 2 | 0.06% |
| Legality | 2 | 0.06% |
| Prison | 2 | 0.06% |
| Cost | 2 | 0.06% |
| Robbery | 2 | 0.06% |
| Practice | 2 | 0.06% |
| Limiting | 2 | 0.06% |
| Commerce_buy | 1 | 0.03% |
| Vocalizations | 1 | 0.03% |
| Labeling | 1 | 0.03% |
| Suspicion | 1 | 0.03% |
| Reforming_a_system | 1 | 0.03% |
| Justifying | 1 | 0.03% |
| Having_or_lacking_access | 1 | 0.03% |
| Carry_goods | 1 | 0.03% |
| Award | 1 | 0.03% |
| Revenge | 1 | 0.03% |
| Renting | 1 | 0.03% |

### Relations

None.

### Classifications

None.

### Structures

None.

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | unknown |
| relations | unknown |
| events | unknown |
| classifications | unknown |
| structures | unknown |
