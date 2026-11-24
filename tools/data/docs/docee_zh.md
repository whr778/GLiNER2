# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| docee_zh | train | 27,122 | 27,122 | 31,369 | 0 | 27,122 | 0 | 0 | 0 |
| docee_zh | val | 3,390 | 3,390 | 3,994 | 0 | 3,390 | 0 | 0 | 0 |
| docee_zh | test | 3,398 | 3,398 | 3,921 | 0 | 3,398 | 0 | 0 | 0 |

## docee_zh

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 27,122 | 27,122 | 584 | 1970 | 30008 | 31,369 | 0 | 27,122 | 0 | 0 | 0 |
| val | 3,390 | 3,390 | 580 | 1943 | 19355 | 3,994 | 0 | 3,390 | 0 | 0 | 0 |
| test | 3,398 | 3,398 | 585 | 1953 | 30555 | 3,921 | 0 | 3,398 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| zho | 26,909 | 3,369 | 3,376 | 33,654 | 99.25% |
| eng | 158 | 14 | 12 | 184 | 0.54% |
| und | 50 | 7 | 8 | 65 | 0.19% |
| wuu | 2 | 0 | 0 | 2 | 0.01% |
| kor | 1 | 0 | 1 | 2 | 0.01% |
| hun | 1 | 0 | 0 | 1 | 0.00% |
| srp | 1 | 0 | 0 | 1 | 0.00% |
| deu | 0 | 0 | 1 | 1 | 0.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 3,398 |
| test_unique_inputs | 3,398 |
| train_records | 27,122 |
| train_unique_inputs | 27,122 |
| val_records | 3,390 |
| val_unique_inputs | 3,390 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0017 | 31,369 / 3,994 | Location 27%→27%, Cause 14%→14%, Date 40%→40%, Casualties and Losses 19%→19% |
| entities (per mention) | train / test | 0.0086 | 31,369 / 3,921 | Cause 14%→13%, Location 27%→28%, Date 40%→40%, Casualties and Losses 19%→19% |
| entities (per mention) | val / test | 0.0097 | 3,994 / 3,921 | Cause 14%→13%, Location 27%→28%, Casualties and Losses 19%→19%, Date 40%→40% |
| classifications (true_label) | train / val | 0.0053 | 27,122 / 3,390 | docee_event: Appoint_Inauguration 5%→5%, docee_event: Environment Pollution 0%→0%, docee_event: Tsunamis 0%→0%, docee_event: Famous Person - Divorce 1%→1%, docee_event: Organization Fine 10%→10% |
| classifications (true_label) | train / test | 0.0036 | 27,122 / 3,398 | docee_event: CommitCrime - Accuse 0%→0%, docee_event: Road Crash 2%→2%, docee_event: Strike 1%→1%, docee_event: Gas Explosion 0%→0%, docee_event: Mudslides 1%→1% |
| classifications (true_label) | val / test | 0.0066 | 3,390 / 3,398 | docee_event: Appoint_Inauguration 5%→5%, docee_event: Tsunamis 0%→0%, docee_event: Famous Person - Divorce 1%→1%, docee_event: Riot 3%→3%, docee_event: Road Crash 2%→2% |

Thin support in val, classifications (true_label): 12 label(s) under 10 gold. docee_event: Financial Crisis ×1, docee_event: Disease Outbreaks ×4, docee_event: Join in an Orgnization ×4, docee_event: Organization Merge ×5, docee_event: Insect Disaster ×5, docee_event: CommitCrime - Accuse ×5, docee_event: Famous Person - Sick ×6, docee_event: Financial Aid ×7, docee_event: Tear Up Agreement ×7, docee_event: Organization Established ×7, docee_event: CommitCrime - Release ×7, docee_event: Tsunamis ×8.
Thin support in test, classifications (true_label): 11 label(s) under 10 gold. docee_event: Financial Crisis ×2, docee_event: Disease Outbreaks ×3, docee_event: Join in an Orgnization ×3, docee_event: CommitCrime - Accuse ×4, docee_event: Organization Merge ×5, docee_event: Famous Person - Sick ×5, docee_event: Insect Disaster ×6, docee_event: Financial Aid ×6, docee_event: Tear Up Agreement ×7, docee_event: Organization Established ×8, docee_event: CommitCrime - Release ×8.

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
| Date | 12,639 | 40.29% |
| Location | 8,488 | 27.06% |
| Casualties and Losses | 5,955 | 18.98% |
| Cause | 4,287 | 13.67% |

#### val

| label | count | share |
| --- | --- | --- |
| Date | 1,612 | 40.36% |
| Location | 1,076 | 26.94% |
| Casualties and Losses | 756 | 18.93% |
| Cause | 550 | 13.77% |

#### test

| label | count | share |
| --- | --- | --- |
| Date | 1,587 | 40.47% |
| Location | 1,085 | 27.67% |
| Casualties and Losses | 747 | 19.05% |
| Cause | 502 | 12.80% |

### Relations

None.

### Classifications

**train, docee_event.** Majority class 10.6% of 27,122 answers. The marginal is `true_label`, not the menu.

#### train docee_event

| label | count | share |
| --- | --- | --- |
| Sports Competition | 2,885 | 10.64% |
| Organization Fine | 2,747 | 10.13% |
| Fire | 2,037 | 7.51% |
| CommitCrime - Arrest | 1,404 | 5.18% |
| Appoint_Inauguration | 1,257 | 4.63% |
| CommitCrime - Investigate | 1,195 | 4.41% |
| Resignation_Dismissal | 1,190 | 4.39% |
| New Achievements in Aerospace | 973 | 3.59% |
| Famous Person - Death | 924 | 3.41% |
| Riot | 742 | 2.74% |
| Mass Poisoning | 709 | 2.61% |
| Government Policy Changes | 673 | 2.48% |
| Road Crash | 659 | 2.43% |
| Earthquakes | 658 | 2.43% |
| Break Historical Records | 643 | 2.37% |
| Military Excercise | 626 | 2.31% |
| Air Crash | 516 | 1.90% |
| Sign Agreement | 410 | 1.51% |
| Famous Person - Give a Speech | 399 | 1.47% |
| Storm | 396 | 1.46% |
| Floods | 347 | 1.28% |
| Awards Ceremony | 331 | 1.22% |
| New Wonders in Nature | 316 | 1.17% |
| Volcano Eruption | 305 | 1.12% |
| Withdraw from an Orgnization | 303 | 1.12% |
| Mudslides | 294 | 1.08% |
| Strike | 285 | 1.05% |
| Protest | 282 | 1.04% |
| Shipwreck | 279 | 1.03% |
| Droughts | 264 | 0.97% |
| Famous Person - Recovered | 263 | 0.97% |
| Famous Person - Divorce | 252 | 0.93% |
| CommitCrime - Sentence | 237 | 0.87% |
| Election | 192 | 0.71% |
| Diplomatic Talks | 180 | 0.66% |
| Regime Change | 174 | 0.64% |
| Bank Robbery | 155 | 0.57% |
| Organization Closed | 155 | 0.57% |
| Diplomatic Visit | 138 | 0.51% |
| Train Collisions | 138 | 0.51% |
| Gas Explosion | 118 | 0.44% |
| New Archeological Discoveries | 113 | 0.42% |
| Mine Collapses | 111 | 0.41% |
| Famine | 107 | 0.39% |
| Environment Pollution | 95 | 0.35% |
| Famous Person - Marriage | 93 | 0.34% |
| Tsunamis | 76 | 0.28% |
| CommitCrime - Release | 66 | 0.24% |
| Organization Established | 61 | 0.22% |
| Tear Up Agreement | 58 | 0.21% |
| Financial Aid | 52 | 0.19% |
| CommitCrime - Accuse | 47 | 0.17% |
| Insect Disaster | 44 | 0.16% |
| Organization Merge | 41 | 0.15% |
| Famous Person - Sick | 41 | 0.15% |
| Disease Outbreaks | 27 | 0.10% |
| Join in an Orgnization | 27 | 0.10% |
| Financial Crisis | 12 | 0.04% |

**val, docee_event.** Majority class 10.6% of 3,390 answers. The marginal is `true_label`, not the menu.

#### val docee_event

| label | count | share |
| --- | --- | --- |
| Sports Competition | 361 | 10.65% |
| Organization Fine | 342 | 10.09% |
| Fire | 255 | 7.52% |
| CommitCrime - Arrest | 176 | 5.19% |
| Appoint_Inauguration | 154 | 4.54% |
| CommitCrime - Investigate | 150 | 4.42% |
| Resignation_Dismissal | 149 | 4.40% |
| New Achievements in Aerospace | 121 | 3.57% |
| Famous Person - Death | 115 | 3.39% |
| Riot | 94 | 2.77% |
| Mass Poisoning | 89 | 2.63% |
| Government Policy Changes | 84 | 2.48% |
| Earthquakes | 82 | 2.42% |
| Road Crash | 82 | 2.42% |
| Break Historical Records | 80 | 2.36% |
| Military Excercise | 78 | 2.30% |
| Air Crash | 65 | 1.92% |
| Sign Agreement | 52 | 1.53% |
| Famous Person - Give a Speech | 50 | 1.47% |
| Storm | 49 | 1.45% |
| Floods | 43 | 1.27% |
| Awards Ceremony | 42 | 1.24% |
| Withdraw from an Orgnization | 39 | 1.15% |
| Volcano Eruption | 39 | 1.15% |
| New Wonders in Nature | 39 | 1.15% |
| Mudslides | 37 | 1.09% |
| Protest | 36 | 1.06% |
| Strike | 36 | 1.06% |
| Shipwreck | 35 | 1.03% |
| Famous Person - Recovered | 33 | 0.97% |
| Droughts | 33 | 0.97% |
| Famous Person - Divorce | 30 | 0.88% |
| CommitCrime - Sentence | 29 | 0.86% |
| Election | 24 | 0.71% |
| Diplomatic Talks | 23 | 0.68% |
| Regime Change | 22 | 0.65% |
| Bank Robbery | 20 | 0.59% |
| Organization Closed | 20 | 0.59% |
| Diplomatic Visit | 18 | 0.53% |
| Train Collisions | 18 | 0.53% |
| Gas Explosion | 15 | 0.44% |
| New Archeological Discoveries | 14 | 0.41% |
| Environment Pollution | 14 | 0.41% |
| Famine | 13 | 0.38% |
| Mine Collapses | 13 | 0.38% |
| Famous Person - Marriage | 11 | 0.32% |
| Tsunamis | 8 | 0.24% |
| Financial Aid | 7 | 0.21% |
| Tear Up Agreement | 7 | 0.21% |
| Organization Established | 7 | 0.21% |
| CommitCrime - Release | 7 | 0.21% |
| Famous Person - Sick | 6 | 0.18% |
| Organization Merge | 5 | 0.15% |
| Insect Disaster | 5 | 0.15% |
| CommitCrime - Accuse | 5 | 0.15% |
| Disease Outbreaks | 4 | 0.12% |
| Join in an Orgnization | 4 | 0.12% |
| Financial Crisis | 1 | 0.03% |

**test, docee_event.** Majority class 10.7% of 3,398 answers. The marginal is `true_label`, not the menu.

#### test docee_event

| label | count | share |
| --- | --- | --- |
| Sports Competition | 362 | 10.65% |
| Organization Fine | 343 | 10.09% |
| Fire | 255 | 7.50% |
| CommitCrime - Arrest | 176 | 5.18% |
| Appoint_Inauguration | 157 | 4.62% |
| CommitCrime - Investigate | 150 | 4.41% |
| Resignation_Dismissal | 149 | 4.38% |
| New Achievements in Aerospace | 122 | 3.59% |
| Famous Person - Death | 116 | 3.41% |
| Riot | 93 | 2.74% |
| Mass Poisoning | 89 | 2.62% |
| Government Policy Changes | 84 | 2.47% |
| Earthquakes | 82 | 2.41% |
| Break Historical Records | 81 | 2.38% |
| Road Crash | 81 | 2.38% |
| Military Excercise | 78 | 2.30% |
| Air Crash | 65 | 1.91% |
| Sign Agreement | 51 | 1.50% |
| Storm | 50 | 1.47% |
| Famous Person - Give a Speech | 50 | 1.47% |
| Floods | 44 | 1.29% |
| Awards Ceremony | 41 | 1.21% |
| New Wonders in Nature | 40 | 1.18% |
| Mudslides | 38 | 1.12% |
| Withdraw from an Orgnization | 38 | 1.12% |
| Volcano Eruption | 38 | 1.12% |
| Strike | 37 | 1.09% |
| Shipwreck | 35 | 1.03% |
| Protest | 35 | 1.03% |
| Famous Person - Recovered | 33 | 0.97% |
| Droughts | 33 | 0.97% |
| Famous Person - Divorce | 32 | 0.94% |
| CommitCrime - Sentence | 30 | 0.88% |
| Election | 24 | 0.71% |
| Regime Change | 22 | 0.65% |
| Diplomatic Talks | 22 | 0.65% |
| Bank Robbery | 19 | 0.56% |
| Organization Closed | 19 | 0.56% |
| Diplomatic Visit | 17 | 0.50% |
| Train Collisions | 17 | 0.50% |
| Gas Explosion | 16 | 0.47% |
| Famine | 14 | 0.41% |
| Mine Collapses | 14 | 0.41% |
| New Archeological Discoveries | 14 | 0.41% |
| Environment Pollution | 13 | 0.38% |
| Famous Person - Marriage | 12 | 0.35% |
| Tsunamis | 10 | 0.29% |
| Organization Established | 8 | 0.24% |
| CommitCrime - Release | 8 | 0.24% |
| Tear Up Agreement | 7 | 0.21% |
| Insect Disaster | 6 | 0.18% |
| Financial Aid | 6 | 0.18% |
| Organization Merge | 5 | 0.15% |
| Famous Person - Sick | 5 | 0.15% |
| CommitCrime - Accuse | 4 | 0.12% |
| Disease Outbreaks | 3 | 0.09% |
| Join in an Orgnization | 3 | 0.09% |
| Financial Crisis | 2 | 0.06% |

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
