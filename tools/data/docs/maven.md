# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| maven | train | 2,913 | 2,913 | 0 | 0 | 0 | 0 | 65,695 | 0 |
| maven | val | 355 | 355 | 0 | 0 | 0 | 0 | 7,867 | 0 |
| maven | test | 355 | 355 | 0 | 0 | 0 | 0 | 7,932 | 0 |

## maven

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2,913 | 2,913 | 252 | 497 | 1754 | 0 | 0 | 0 | 0 | 65,695 | 0 |
| val | 355 | 355 | 240 | 529 | 1968 | 0 | 0 | 0 | 0 | 7,867 | 0 |
| test | 355 | 355 | 258 | 487 | 1574 | 0 | 0 | 0 | 0 | 7,932 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 2,913 | 355 | 355 | 3,623 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 355 |
| test_unique_inputs | 355 |
| train_records | 2,913 |
| train_unique_inputs | 2,913 |
| val_records | 355 |
| val_unique_inputs | 355 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| event types (per trigger mention) | train / val | 0.0576 | 77,993 / 9,260 | Attack 4%→3%, Traveling 1%→1%, Motion 3%→2%, Self_motion 2%→2%, Damaging 1%→1% |
| event types (per trigger mention) | train / test | 0.0663 | 77,993 / 9,644 | Competition 3%→5%, Hostile_encounter 4%→5%, Catastrophe 4%→5%, Motion 3%→2%, Damaging 1%→2% |
| event types (per trigger mention) | val / test | 0.0977 | 9,260 / 9,644 | Competition 3%→5%, Hostile_encounter 4%→5%, Damaging 1%→2%, Catastrophe 4%→5%, Self_motion 2%→2% |

Thin support in val, event types (per trigger mention): 35 label(s) under 10 gold. Breathing ×1, Institutionalization ×1, Emergency ×1, Renting ×1, Extradition ×2, Suspicion ×2, Theft ×3, Ingestion ×3, Risk ×3, Vocalizations ×3, Ratification ×3, Robbery ×3, Justifying ×4, Practice ×4, Wearing ×4, Testing ×4, Labeling ×4, Commerce_buy ×5, Lighting ×5, Scouring ×5, Prison ×5, Award ×5, Submitting_documents ×5, Expend_resource ×6, Cost ×7, Patrolling ×8, Rite ×8, Revenge ×8, Imposing_obligation ×8, Exchange ×8, Containing ×8, Body_movement ×9, Carry_goods ×9, Having_or_lacking_access ×9, Commerce_pay ×9.
Thin support in test, event types (per trigger mention): 35 label(s) under 10 gold. Change_tool ×1, Extradition ×1, Emergency ×1, Incident ×2, Institutionalization ×2, Theft ×2, Vocalizations ×2, Scouring ×2, Suspicion ×3, Renting ×3, Justifying ×3, Ratification ×3, Lighting ×3, Rite ×3, Revenge ×4, Ingestion ×4, Labeling ×4, Practice ×5, Carry_goods ×5, Cure ×6, Submitting_documents ×6, Exchange ×7, Limiting ×7, Reforming_a_system ×7, Containing ×7, Imposing_obligation ×8, Bearing_arms ×8, Award ×8, Protest ×8, Commerce_buy ×8, Adducing ×8, Reveal_secret ×9, Rewards_and_punishments ×9, Prison ×9, Filling ×9.

### Events

**train.** 77,993 instances in 2,913 records, 0 triggerless. Trigger gold 65,695. Argument gold 0. Arguments per instance, mean 0.00. Trigger length in whitespace tokens, mean 1.03, single-token share 96.8%.
Trigger-only: this split cannot teach or score event arguments.

#### train event types

| label | count | share |
| --- | --- | --- |
| Catastrophe | 3,145 | 4.03% |
| Attack | 2,920 | 3.74% |
| Hostile_encounter | 2,856 | 3.66% |
| Causation | 2,728 | 3.50% |
| Process_start | 2,628 | 3.37% |
| Competition | 2,460 | 3.15% |
| Motion | 2,165 | 2.78% |
| Social_event | 1,653 | 2.12% |
| Killing | 1,625 | 2.08% |
| Conquering | 1,432 | 1.84% |
| Statement | 1,392 | 1.78% |
| Process_end | 1,392 | 1.78% |
| Self_motion | 1,282 | 1.64% |
| Arriving | 1,236 | 1.58% |
| Destroying | 1,176 | 1.51% |
| Coming_to_be | 1,120 | 1.44% |
| Bodily_harm | 1,024 | 1.31% |
| Military_operation | 1,022 | 1.31% |
| Death | 1,001 | 1.28% |
| Damaging | 981 | 1.26% |
| Cause_change_of_strength | 948 | 1.22% |
| Creating | 936 | 1.20% |
| Hold | 922 | 1.18% |
| Control | 911 | 1.17% |
| Cause_change_of_position_on_a_scale | 903 | 1.16% |
| Earnings_and_losses | 891 | 1.14% |
| Becoming | 848 | 1.09% |
| Getting | 842 | 1.08% |
| Arranging | 817 | 1.05% |
| Know | 810 | 1.04% |
| Traveling | 793 | 1.02% |
| Presence | 777 | 1.00% |
| Preventing_or_letting | 772 | 0.99% |
| Defending | 764 | 0.98% |
| Escaping | 764 | 0.98% |
| Action | 709 | 0.91% |
| Change_of_leadership | 704 | 0.90% |
| Change | 684 | 0.88% |
| Motion_directional | 684 | 0.88% |
| Cause_to_be_included | 678 | 0.87% |
| Participation | 677 | 0.87% |
| Placing | 662 | 0.85% |
| Influence | 654 | 0.84% |
| Judgment_communication | 640 | 0.82% |
| Expressing_publicly | 626 | 0.80% |
| Request | 620 | 0.79% |
| Name_conferral | 618 | 0.79% |
| Giving | 561 | 0.72% |
| Supporting | 554 | 0.71% |
| Recording | 533 | 0.68% |
| Removing | 520 | 0.67% |
| Using | 505 | 0.65% |
| Agree_or_refuse_to_act | 504 | 0.65% |
| Supply | 485 | 0.62% |
| Communication | 471 | 0.60% |
| Reporting | 459 | 0.59% |
| Choosing | 453 | 0.58% |
| Use_firearm | 434 | 0.56% |
| Sending | 417 | 0.53% |
| Bringing | 410 | 0.53% |
| Departing | 403 | 0.52% |
| Assistance | 379 | 0.49% |
| Cause_to_make_progress | 363 | 0.47% |
| Building | 350 | 0.45% |
| Deciding | 350 | 0.45% |
| Perception_active | 346 | 0.44% |
| Dispersal | 341 | 0.44% |
| Violence | 340 | 0.44% |
| Manufacturing | 326 | 0.42% |
| Come_together | 314 | 0.40% |
| Arrest | 307 | 0.39% |
| Response | 299 | 0.38% |
| Cause_to_amalgamate | 299 | 0.38% |
| Hindering | 295 | 0.38% |
| Receiving | 283 | 0.36% |
| Aiming | 282 | 0.36% |
| Achieve | 279 | 0.36% |
| Recovering | 275 | 0.35% |
| Expansion | 265 | 0.34% |
| Scrutiny | 263 | 0.34% |
| Change_event_time | 258 | 0.33% |
| Terrorism | 254 | 0.33% |
| Besieging | 251 | 0.32% |
| Check | 242 | 0.31% |
| Publishing | 231 | 0.30% |
| Criminal_investigation | 228 | 0.29% |
| Becoming_a_member | 225 | 0.29% |
| Committing_crime | 224 | 0.29% |
| Openness | 222 | 0.28% |
| Collaboration | 221 | 0.28% |
| Coming_to_believe | 203 | 0.26% |
| Connect | 203 | 0.26% |
| Surrendering | 199 | 0.26% |
| Being_in_operation | 199 | 0.26% |
| Legal_rulings | 191 | 0.24% |
| Confronting_problem | 179 | 0.23% |
| Warning | 178 | 0.23% |
| Sign_agreement | 178 | 0.23% |
| Preserving | 174 | 0.22% |
| Convincing | 168 | 0.22% |
| GiveUp | 167 | 0.21% |
| Temporary_stay | 165 | 0.21% |
| Quarreling | 163 | 0.21% |
| Change_sentiment | 156 | 0.20% |
| Telling | 155 | 0.20% |
| Protest | 153 | 0.20% |
| Legality | 138 | 0.18% |
| Testing | 137 | 0.18% |
| Rescuing | 132 | 0.17% |
| Commitment | 131 | 0.17% |
| Resolve_problem | 129 | 0.17% |
| Employment | 128 | 0.16% |
| Emptying | 124 | 0.16% |
| Commerce_sell | 124 | 0.16% |
| Body_movement | 115 | 0.15% |
| Releasing | 114 | 0.15% |
| Education_teaching | 104 | 0.13% |
| GetReady | 103 | 0.13% |
| Surrounding | 101 | 0.13% |
| Reveal_secret | 101 | 0.13% |
| Hiding_objects | 99 | 0.13% |
| Forming_relationships | 97 | 0.12% |
| Filling | 97 | 0.12% |
| Writing | 95 | 0.12% |
| Bearing_arms | 93 | 0.12% |
| Commerce_pay | 93 | 0.12% |
| Reforming_a_system | 93 | 0.12% |
| Expend_resource | 91 | 0.12% |
| Patrolling | 87 | 0.11% |
| Kidnapping | 87 | 0.11% |
| Research | 87 | 0.11% |
| Rewards_and_punishments | 86 | 0.11% |
| Cost | 85 | 0.11% |
| Adducing | 82 | 0.11% |
| Limiting | 78 | 0.10% |
| Wearing | 72 | 0.09% |
| Cure | 71 | 0.09% |
| Create_artwork | 66 | 0.08% |
| Exchange | 64 | 0.08% |
| Having_or_lacking_access | 61 | 0.08% |
| Imposing_obligation | 59 | 0.08% |
| Award | 58 | 0.07% |
| Robbery | 57 | 0.07% |
| Containing | 55 | 0.07% |
| Prison | 55 | 0.07% |
| Revenge | 52 | 0.07% |
| Commerce_buy | 51 | 0.07% |
| Carry_goods | 48 | 0.06% |
| Rite | 48 | 0.06% |
| Submitting_documents | 47 | 0.06% |
| Practice | 37 | 0.05% |
| Lighting | 36 | 0.05% |
| Labeling | 35 | 0.04% |
| Scouring | 32 | 0.04% |
| Ratification | 27 | 0.03% |
| Justifying | 27 | 0.03% |
| Suspicion | 22 | 0.03% |
| Vocalizations | 20 | 0.03% |
| Theft | 19 | 0.02% |
| Renting | 17 | 0.02% |
| Ingestion | 16 | 0.02% |
| Incident | 15 | 0.02% |
| Risk | 12 | 0.02% |
| Extradition | 11 | 0.01% |
| Institutionalization | 11 | 0.01% |
| Emergency | 7 | 0.01% |
| Breathing | 6 | 0.01% |
| Change_tool | 4 | 0.01% |

**val.** 9,260 instances in 355 records, 0 triggerless. Trigger gold 7,867. Argument gold 0. Arguments per instance, mean 0.00. Trigger length in whitespace tokens, mean 1.04, single-token share 96.5%.
Trigger-only: this split cannot teach or score event arguments.

#### val event types

| label | count | share |
| --- | --- | --- |
| Catastrophe | 379 | 4.09% |
| Hostile_encounter | 328 | 3.54% |
| Causation | 318 | 3.43% |
| Process_start | 309 | 3.34% |
| Attack | 299 | 3.23% |
| Competition | 271 | 2.93% |
| Social_event | 223 | 2.41% |
| Motion | 215 | 2.32% |
| Killing | 212 | 2.29% |
| Self_motion | 190 | 2.05% |
| Process_end | 160 | 1.73% |
| Statement | 157 | 1.70% |
| Conquering | 152 | 1.64% |
| Coming_to_be | 144 | 1.56% |
| Becoming | 136 | 1.47% |
| Arriving | 133 | 1.44% |
| Control | 124 | 1.34% |
| Cause_change_of_position_on_a_scale | 123 | 1.33% |
| Hold | 120 | 1.30% |
| Change_of_leadership | 117 | 1.26% |
| Creating | 113 | 1.22% |
| Death | 113 | 1.22% |
| Military_operation | 113 | 1.22% |
| Destroying | 110 | 1.19% |
| Bodily_harm | 108 | 1.17% |
| Getting | 105 | 1.13% |
| Arranging | 105 | 1.13% |
| Preventing_or_letting | 101 | 1.09% |
| Earnings_and_losses | 100 | 1.08% |
| Cause_change_of_strength | 99 | 1.07% |
| Escaping | 97 | 1.05% |
| Know | 95 | 1.03% |
| Defending | 91 | 0.98% |
| Participation | 86 | 0.93% |
| Change | 85 | 0.92% |
| Agree_or_refuse_to_act | 82 | 0.89% |
| Motion_directional | 82 | 0.89% |
| Cause_to_be_included | 81 | 0.87% |
| Damaging | 81 | 0.87% |
| Action | 79 | 0.85% |
| Placing | 76 | 0.82% |
| Reporting | 75 | 0.81% |
| Removing | 74 | 0.80% |
| Presence | 74 | 0.80% |
| Expressing_publicly | 74 | 0.80% |
| Choosing | 74 | 0.80% |
| Influence | 72 | 0.78% |
| Request | 72 | 0.78% |
| Name_conferral | 71 | 0.77% |
| Giving | 68 | 0.73% |
| Communication | 65 | 0.70% |
| Judgment_communication | 62 | 0.67% |
| Supporting | 59 | 0.64% |
| Recording | 57 | 0.62% |
| Supply | 55 | 0.59% |
| Bringing | 51 | 0.55% |
| Departing | 51 | 0.55% |
| Traveling | 50 | 0.54% |
| Come_together | 50 | 0.54% |
| Sending | 49 | 0.53% |
| Dispersal | 49 | 0.53% |
| Assistance | 47 | 0.51% |
| Using | 47 | 0.51% |
| Building | 46 | 0.50% |
| Violence | 45 | 0.49% |
| Cause_to_make_progress | 45 | 0.49% |
| Deciding | 43 | 0.46% |
| Use_firearm | 43 | 0.46% |
| Becoming_a_member | 42 | 0.45% |
| Change_event_time | 40 | 0.43% |
| Perception_active | 40 | 0.43% |
| Scrutiny | 40 | 0.43% |
| Arrest | 39 | 0.42% |
| Manufacturing | 38 | 0.41% |
| Achieve | 37 | 0.40% |
| Cause_to_amalgamate | 37 | 0.40% |
| Recovering | 36 | 0.39% |
| Response | 35 | 0.38% |
| Terrorism | 34 | 0.37% |
| Receiving | 33 | 0.36% |
| Connect | 32 | 0.35% |
| Coming_to_believe | 32 | 0.35% |
| Expansion | 32 | 0.35% |
| Publishing | 30 | 0.32% |
| Aiming | 30 | 0.32% |
| Openness | 29 | 0.31% |
| Hindering | 27 | 0.29% |
| Check | 26 | 0.28% |
| Being_in_operation | 26 | 0.28% |
| Collaboration | 26 | 0.28% |
| Surrendering | 26 | 0.28% |
| Besieging | 26 | 0.28% |
| Criminal_investigation | 24 | 0.26% |
| Preserving | 24 | 0.26% |
| Legal_rulings | 24 | 0.26% |
| Temporary_stay | 24 | 0.26% |
| Quarreling | 24 | 0.26% |
| Protest | 24 | 0.26% |
| Convincing | 23 | 0.25% |
| Confronting_problem | 22 | 0.24% |
| GiveUp | 22 | 0.24% |
| Sign_agreement | 20 | 0.22% |
| Commitment | 19 | 0.21% |
| Warning | 19 | 0.21% |
| Rescuing | 18 | 0.19% |
| Commerce_sell | 18 | 0.19% |
| Change_sentiment | 17 | 0.18% |
| Releasing | 16 | 0.17% |
| Reforming_a_system | 15 | 0.16% |
| Committing_crime | 14 | 0.15% |
| Reveal_secret | 14 | 0.15% |
| Bearing_arms | 14 | 0.15% |
| Filling | 13 | 0.14% |
| Telling | 13 | 0.14% |
| Forming_relationships | 13 | 0.14% |
| Legality | 13 | 0.14% |
| Education_teaching | 13 | 0.14% |
| Employment | 12 | 0.13% |
| Hiding_objects | 12 | 0.13% |
| GetReady | 12 | 0.13% |
| Writing | 12 | 0.13% |
| Resolve_problem | 11 | 0.12% |
| Cure | 11 | 0.12% |
| Research | 11 | 0.12% |
| Rewards_and_punishments | 11 | 0.12% |
| Limiting | 11 | 0.12% |
| Kidnapping | 11 | 0.12% |
| Create_artwork | 10 | 0.11% |
| Surrounding | 10 | 0.11% |
| Emptying | 10 | 0.11% |
| Adducing | 10 | 0.11% |
| Body_movement | 9 | 0.10% |
| Carry_goods | 9 | 0.10% |
| Having_or_lacking_access | 9 | 0.10% |
| Commerce_pay | 9 | 0.10% |
| Patrolling | 8 | 0.09% |
| Rite | 8 | 0.09% |
| Revenge | 8 | 0.09% |
| Imposing_obligation | 8 | 0.09% |
| Exchange | 8 | 0.09% |
| Containing | 8 | 0.09% |
| Cost | 7 | 0.08% |
| Expend_resource | 6 | 0.06% |
| Commerce_buy | 5 | 0.05% |
| Lighting | 5 | 0.05% |
| Scouring | 5 | 0.05% |
| Prison | 5 | 0.05% |
| Award | 5 | 0.05% |
| Submitting_documents | 5 | 0.05% |
| Justifying | 4 | 0.04% |
| Practice | 4 | 0.04% |
| Wearing | 4 | 0.04% |
| Testing | 4 | 0.04% |
| Labeling | 4 | 0.04% |
| Theft | 3 | 0.03% |
| Ingestion | 3 | 0.03% |
| Risk | 3 | 0.03% |
| Vocalizations | 3 | 0.03% |
| Ratification | 3 | 0.03% |
| Robbery | 3 | 0.03% |
| Extradition | 2 | 0.02% |
| Suspicion | 2 | 0.02% |
| Breathing | 1 | 0.01% |
| Institutionalization | 1 | 0.01% |
| Emergency | 1 | 0.01% |
| Renting | 1 | 0.01% |

**test.** 9,644 instances in 355 records, 0 triggerless. Trigger gold 7,932. Argument gold 0. Arguments per instance, mean 0.00. Trigger length in whitespace tokens, mean 1.03, single-token share 96.7%.
Trigger-only: this split cannot teach or score event arguments.

#### test event types

| label | count | share |
| --- | --- | --- |
| Catastrophe | 450 | 4.67% |
| Hostile_encounter | 436 | 4.52% |
| Competition | 436 | 4.52% |
| Causation | 341 | 3.54% |
| Process_start | 338 | 3.50% |
| Attack | 331 | 3.43% |
| Social_event | 220 | 2.28% |
| Motion | 218 | 2.26% |
| Killing | 193 | 2.00% |
| Process_end | 178 | 1.85% |
| Arriving | 173 | 1.79% |
| Statement | 172 | 1.78% |
| Conquering | 170 | 1.76% |
| Damaging | 165 | 1.71% |
| Destroying | 159 | 1.65% |
| Coming_to_be | 154 | 1.60% |
| Self_motion | 148 | 1.53% |
| Cause_change_of_strength | 133 | 1.38% |
| Bodily_harm | 123 | 1.28% |
| Military_operation | 122 | 1.27% |
| Getting | 120 | 1.24% |
| Earnings_and_losses | 113 | 1.17% |
| Cause_change_of_position_on_a_scale | 112 | 1.16% |
| Change_of_leadership | 112 | 1.16% |
| Death | 112 | 1.16% |
| Creating | 110 | 1.14% |
| Motion_directional | 103 | 1.07% |
| Know | 102 | 1.06% |
| Becoming | 101 | 1.05% |
| Control | 97 | 1.01% |
| Arranging | 95 | 0.99% |
| Escaping | 95 | 0.99% |
| Hold | 95 | 0.99% |
| Traveling | 94 | 0.97% |
| Cause_to_be_included | 91 | 0.94% |
| Influence | 86 | 0.89% |
| Preventing_or_letting | 85 | 0.88% |
| Participation | 84 | 0.87% |
| Supply | 78 | 0.81% |
| Placing | 77 | 0.80% |
| Change | 77 | 0.80% |
| Action | 76 | 0.79% |
| Presence | 75 | 0.78% |
| Name_conferral | 71 | 0.74% |
| Agree_or_refuse_to_act | 69 | 0.72% |
| Defending | 68 | 0.71% |
| Supporting | 67 | 0.69% |
| Giving | 66 | 0.68% |
| Judgment_communication | 65 | 0.67% |
| Recording | 65 | 0.67% |
| Removing | 61 | 0.63% |
| Use_firearm | 61 | 0.63% |
| Using | 60 | 0.62% |
| Expressing_publicly | 56 | 0.58% |
| Request | 55 | 0.57% |
| Manufacturing | 47 | 0.49% |
| Receiving | 46 | 0.48% |
| Sending | 46 | 0.48% |
| Communication | 46 | 0.48% |
| Bringing | 45 | 0.47% |
| Reporting | 44 | 0.46% |
| Choosing | 42 | 0.44% |
| Perception_active | 42 | 0.44% |
| Deciding | 41 | 0.43% |
| Assistance | 41 | 0.43% |
| Cause_to_make_progress | 40 | 0.41% |
| Cause_to_amalgamate | 38 | 0.39% |
| Come_together | 36 | 0.37% |
| Dispersal | 36 | 0.37% |
| Hindering | 35 | 0.36% |
| Besieging | 35 | 0.36% |
| Aiming | 35 | 0.36% |
| Departing | 34 | 0.35% |
| Violence | 34 | 0.35% |
| Arrest | 33 | 0.34% |
| Publishing | 33 | 0.34% |
| Achieve | 32 | 0.33% |
| Check | 30 | 0.31% |
| Building | 29 | 0.30% |
| Quarreling | 28 | 0.29% |
| Change_sentiment | 28 | 0.29% |
| Expansion | 28 | 0.29% |
| Telling | 27 | 0.28% |
| Committing_crime | 27 | 0.28% |
| Recovering | 26 | 0.27% |
| Legal_rulings | 26 | 0.27% |
| Criminal_investigation | 25 | 0.26% |
| Collaboration | 25 | 0.26% |
| Confronting_problem | 24 | 0.25% |
| Preserving | 24 | 0.25% |
| Being_in_operation | 23 | 0.24% |
| Warning | 23 | 0.24% |
| Change_event_time | 23 | 0.24% |
| Becoming_a_member | 22 | 0.23% |
| Scrutiny | 22 | 0.23% |
| Surrendering | 21 | 0.22% |
| Terrorism | 21 | 0.22% |
| Testing | 21 | 0.22% |
| Openness | 20 | 0.21% |
| Coming_to_believe | 20 | 0.21% |
| Emptying | 19 | 0.20% |
| Sign_agreement | 19 | 0.20% |
| Temporary_stay | 18 | 0.19% |
| GiveUp | 18 | 0.19% |
| Legality | 18 | 0.19% |
| Resolve_problem | 17 | 0.18% |
| Response | 17 | 0.18% |
| Body_movement | 16 | 0.17% |
| Employment | 16 | 0.17% |
| Connect | 16 | 0.17% |
| Wearing | 16 | 0.17% |
| Convincing | 15 | 0.16% |
| Education_teaching | 15 | 0.16% |
| Surrounding | 14 | 0.15% |
| Patrolling | 13 | 0.13% |
| Forming_relationships | 13 | 0.13% |
| Releasing | 13 | 0.13% |
| Writing | 13 | 0.13% |
| Commerce_pay | 12 | 0.12% |
| Hiding_objects | 12 | 0.12% |
| Cost | 12 | 0.12% |
| Rescuing | 12 | 0.12% |
| Expend_resource | 11 | 0.11% |
| GetReady | 11 | 0.11% |
| Kidnapping | 11 | 0.11% |
| Create_artwork | 11 | 0.11% |
| Commerce_sell | 11 | 0.11% |
| Having_or_lacking_access | 10 | 0.10% |
| Robbery | 10 | 0.10% |
| Research | 10 | 0.10% |
| Commitment | 10 | 0.10% |
| Reveal_secret | 9 | 0.09% |
| Rewards_and_punishments | 9 | 0.09% |
| Prison | 9 | 0.09% |
| Filling | 9 | 0.09% |
| Imposing_obligation | 8 | 0.08% |
| Bearing_arms | 8 | 0.08% |
| Award | 8 | 0.08% |
| Protest | 8 | 0.08% |
| Commerce_buy | 8 | 0.08% |
| Adducing | 8 | 0.08% |
| Exchange | 7 | 0.07% |
| Limiting | 7 | 0.07% |
| Reforming_a_system | 7 | 0.07% |
| Containing | 7 | 0.07% |
| Cure | 6 | 0.06% |
| Submitting_documents | 6 | 0.06% |
| Practice | 5 | 0.05% |
| Carry_goods | 5 | 0.05% |
| Revenge | 4 | 0.04% |
| Ingestion | 4 | 0.04% |
| Labeling | 4 | 0.04% |
| Suspicion | 3 | 0.03% |
| Renting | 3 | 0.03% |
| Justifying | 3 | 0.03% |
| Ratification | 3 | 0.03% |
| Lighting | 3 | 0.03% |
| Rite | 3 | 0.03% |
| Incident | 2 | 0.02% |
| Institutionalization | 2 | 0.02% |
| Theft | 2 | 0.02% |
| Vocalizations | 2 | 0.02% |
| Scouring | 2 | 0.02% |
| Change_tool | 1 | 0.01% |
| Extradition | 1 | 0.01% |
| Emergency | 1 | 0.01% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

No event arguments.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.83 | 31.10% | 0.00 | n/a | 0.00% | 0 | 0 | 0 | 77,993 |
| val | 1.74 | 30.27% | 0.00 | n/a | 0.00% | 0 | 0 | 0 | 9,260 |
| test | 2.01 | 33.86% | 0.00 | n/a | 0.00% | 0 | 0 | 0 | 9,644 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 77,993 | 0.00% | 76,326 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| val | 9,260 | 0.00% | 8,273 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| test | 9,644 | 0.00% | 12,065 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 0 | 0 | 76,326 |
| val | 0 | 0 | 8,273 |
| test | 0 | 0 | 12,065 |

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

No event arguments.

### Entities

None.

### Relations

None.

### Classifications

None.

### Structures

None.

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | human |
| relations | human |
| events | human |
| classifications | human |
| structures | human |
