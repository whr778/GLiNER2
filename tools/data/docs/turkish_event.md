# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| turkish_event | train | 24,028 | 24,028 | 141,576 | 0 | 24,028 | 0 | 0 | 0 |
| turkish_event | val | 3,016 | 3,016 | 17,933 | 0 | 3,016 | 0 | 0 | 0 |
| turkish_event | test | 3,090 | 3,090 | 18,465 | 0 | 3,090 | 0 | 0 | 0 |

## turkish_event

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 24,028 | 24,028 | 282 | 817 | 1249 | 141,576 | 0 | 24,028 | 0 | 0 | 0 |
| val | 3,016 | 3,016 | 278 | 825 | 1164 | 17,933 | 0 | 3,016 | 0 | 0 | 0 |
| test | 3,090 | 3,090 | 283 | 854 | 1206 | 18,465 | 0 | 3,090 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| tur | 24,027 | 3,016 | 3,089 | 30,132 | 99.99% |
| und | 1 | 0 | 1 | 2 | 0.01% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 3,090 |
| test_unique_inputs | 3,090 |
| train_records | 24,028 |
| train_unique_inputs | 24,028 |
| val_records | 3,016 |
| val_unique_inputs | 3,016 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0655 | 141,576 / 17,933 | Participants 2%→2%, Policy Content 3%→3%, Person under Investigation 2%→2%, Winner 1%→0%, Casualties and Losses 3%→3% |
| entities (per mention) | train / test | 0.0691 | 141,576 / 18,465 | Location 15%→16%, Beneficiary 0%→1%, Policy Content 3%→3%, Deliberating Agency 2%→1%, The Charged Crime 1%→1% |
| entities (per mention) | val / test | 0.0897 | 17,933 / 18,465 | Location 15%→16%, Person under Investigation 2%→2%, Participants 2%→2%, The Charged Crime 1%→1%, Winner 0%→1% |
| classifications (true_label) | train / val | 0.0525 | 24,028 / 3,016 | docee_event: Famous Person - Give a Speech 3%→3%, docee_event: Diplomatic Talks 4%→3%, docee_event: CommitCrime - Investigate 6%→7%, docee_event: Diplomatic Visit 2%→2%, docee_event: Protest 3%→3% |
| classifications (true_label) | train / test | 0.0450 | 24,028 / 3,090 | docee_event: Famous Person - Death 2%→3%, docee_event: Government Policy Changes 9%→8%, docee_event: Break Historical Records 1%→1%, docee_event: CommitCrime - Arrest 5%→4%, docee_event: Protest 3%→3% |
| classifications (true_label) | val / test | 0.0566 | 3,016 / 3,090 | docee_event: Famous Person - Death 2%→3%, docee_event: Diplomatic Visit 2%→2%, docee_event: CommitCrime - Arrest 5%→4%, docee_event: Famous Person - Give a Speech 3%→3%, docee_event: Diplomatic Talks 3%→3% |

Unseen in train, present in val, entities (per mention): Economic loss ×1.
Thin support in val, entities (per mention): 121 label(s) under 10 gold. Number of Aftershocks ×1, Countries Withdrawing from the Organization ×1, Economic loss ×1, Recovered Amount ×1, Service Years ×1, Marriage Duration ×1, Storm Movement Speed ×1, Best Way to Shoot ×1, Spectacle End Time ×1, Spectacle Duration ×1, Economy Loss ×1, Hostage ×1, Trapped Days ×1, End Date ×1, Postpone Time ×1, Successor ×1, Water Level ×1, Last Outbreak Time ×1, Strikers Status ×1, Trapped Depth ×1, Number of Trapped People ×1, Bank Name ×1, Strike Outcome ×2, The Agency Who Broke the Agreement ×2, Flight No. ×2, Withdraw Reason ×2, Acquisition amount ×2, Carrier Rocket ×2, Maximum Rainfall ×2, Volcano Name ×2, Outbreak Date ×2, The State of the Volcano (Dormant or Active) ×2, Train No. ×2, Launch Result ×2, Storm Warning Level ×2, Spectacle Start Time ×2, Weapon Used ×2, Transportation ×2, Number of Influenced People ×2, Disaster-Stricken Farmland ×2, End_Date ×2, Head of the Merged Organization ×2, Tear Up Reason ×3, Train Agency ×3, Storm Center Location ×3, Prison Term ×3, Temporary Settlement ×3, Age of the Resignated Person ×3, Passengers ×3, Mission Duration ×3, How Many Times Get Married ×3, Shipwreck Reason ×3, Provisional Government ×3, Storm Direction ×3, Postpone Reason ×3, Prison ×4, Jail Time ×4, Crew ×4, Amount of Precipitation ×4, Maximum Wind Speed ×4, The Grades of the Previous Record Holder ×4, Launch Site ×4, The Worst-Hit Area ×4, Registered Capital ×4, Stolen Amount ×4, Investigating agency ×4, Rescue Start Time ×4, Number of Destroyed Building ×4, Head of the Government ×4, Spectacle Location ×4, Boycotted Institutions ×4, Perpetrator ×5, Start Date ×5, Strike Industry ×5, Strike Agency ×5, Duration ×5, Bankrupt Business ×5, Wedding Venue ×5, Rescue Organizer ×5, Ship No. ×5, Aid Supplies/Amount ×5, Rescue Tool or Method ×5, Invited Person ×5, Types of the Spectacle ×5, Countries Joined the Organization ×5, Head of the Institution ×5, Judge ×6, Agreement Members ×6, Goal ×6, Sequelae ×6, Voting method ×6, Approver ×6, Acquirer ×6, Poisoning Type ×6, Production Cuts ×6, Strikers ×7, People/Organization who predicted the disaster ×7, Prosecution Lawyer ×7, Organization Industry ×7, Perpetrators ×7, Sentencing Location ×8, Scheduled Landing Place ×8, Damaged Property ×8, Launch Date ×8, Spacecraft Mission ×8, People Casting Key Votes ×8, Accident Investigator ×8, Cooperative Agency ×8, Fined Agency ×9, Fine Reason ×9, Penalty Amount ×9, Regulatory Authority ×9, Number of Evacuated People ×9, Missings ×9, Number of Damaged Houses ×9, Annoncement Date ×9, Organization Leader ×9, Acquiree ×9, Country ×9, Inauguration Time ×9, Release Time ×9.
Thin support in val, classifications (true_label): 17 label(s) under 10 gold. docee_event: Withdraw from an Orgnization ×1, docee_event: Famine ×1, docee_event: Volcano Eruption ×2, docee_event: Tear Up Agreement ×3, docee_event: Famous Person - Divorce ×3, docee_event: Bank Robbery ×4, docee_event: Organization Closed ×4, docee_event: Strike ×5, docee_event: Mudslides ×5, docee_event: Organization Merge ×5, docee_event: New Wonders in Nature ×5, docee_event: Train Collisions ×6, docee_event: Mass Poisoning ×6, docee_event: Shipwreck ×7, docee_event: Organization Fine ×8, docee_event: Mine Collapses ×9, docee_event: Droughts ×9.
Unseen in train, present in test, entities (per mention): Refuge ×1.
Thin support in test, entities (per mention): 125 label(s) under 10 gold. Service Years ×1, Countries Withdrawing from the Organization ×1, Withdraw Reason ×1, Acquisition amount ×1, Merger Terms ×1, Head of the Merged Organization ×1, Hull Discovery Time ×1, Temporary Settlement ×1, Alternate Landing Place ×1, Hull Location ×1, Refuge ×1, Recovered Amount ×1, Storm Movement Speed ×1, Prison Term ×1, Provisional Government ×1, Train No. ×1, Time for Dignitaries to Resign ×1, Astronauts ×1, Last Outbreak Time ×1, Marriage Duration ×2, Goal ×2, Penalty Amount ×2, Rescue Start Time ×2, Weapon Used ×2, Investigating agency ×2, Property Division ×2, Strike Outcome ×2, Passengers ×2, The Worst-Hit Area ×2, Head of the Institution ×2, Outbreak Date ×2, Number of Damaged House ×2, Stolen Amount ×2, Head of the Government ×2, Spectacle Location ×2, The Agency Who Broke the Agreement ×2, Tear Up Reason ×2, Influenced People ×2, Launch Site ×2, Storm Warning Level ×3, Strikers Status ×3, Countries Joined the Organization ×3, Hostage ×3, Bankrupt Business ×3, Crew ×3, Postpone Reason ×3, Commanders of the Army ×3, Storm Center Location ×3, Area Affected ×3, Shipwreck Reason ×3, Disaster-Stricken Farmland ×3, Age of the Resignated Person ×3, Flight No. ×3, Launch Result ×4, Prison ×4, Prosecution Lawyer ×4, Number of Destroyed Building ×4, Strike Agency ×4, Jail Time ×4, Announce Platform ×4, Train Agency ×4, Taking-off Place ×4, Annoncement Date ×4, Ship No. ×4, Accident Investigator ×4, Acquirer ×4, Volcano Name ×4, Approver ×4, Organization Leader ×4, Electoral System ×5, Regulatory Authority ×5, Number of Aftershocks ×5, Duration ×5, Invited Person ×5, Number of Influenced People ×5, Number of Trapped People ×5, Judge ×5, Perpetrators ×5, Voting method ×5, People Casting Key Votes ×5, Invalid Date ×5, How Many Times Get Married ×5, Economists who Predicted the Crisis ×5, Acquiree ×5, Mission Duration ×5, Bank Name ×5, Launch Date ×5, Agreement Members ×5, The Final Seats Result ×6, Release Time ×6, Detention Start Time ×6, Wedding Venue ×6, Casualities and Losses ×6, Last Time the Record was Broken ×6, Declarer ×6, Scale ×6, Organization Industry ×6, Successor ×6, Illness ×7, Fine Reason ×7, Strike Industry ×7, The Grades of the Previous Record Holder ×7, Scheduled Landing Place ×7, Unemployed Rate ×7, Term of Office ×7, Storm Direction ×7, Sequelae ×7, Join Conditions ×7, Symptom ×8, Country ×8, Launch Country ×8, Start Date ×8, Rescue Tool or Method ×8, Court Time ×8, Prosecutor ×8, Boycotted Institutions ×8, Army ×8, Treatment Method ×9, Candidate Scandal(Allegations of fraud, etc.) ×9, Sentencing Location ×9, Missings ×9, Evidence ×9, Registered Capital ×9, Number of People Hospitalized ×9, Damaged Property ×9.
Thin support in test, classifications (true_label): 20 label(s) under 10 gold. docee_event: Tsunamis ×1, docee_event: Withdraw from an Orgnization ×1, docee_event: New Wonders in Nature ×1, docee_event: Famine ×2, docee_event: Tear Up Agreement ×2, docee_event: Volcano Eruption ×3, docee_event: Famous Person - Divorce ×4, docee_event: Bank Robbery ×4, docee_event: Droughts ×4, docee_event: Organization Closed ×5, docee_event: Train Collisions ×5, docee_event: Organization Merge ×5, docee_event: Mudslides ×6, docee_event: Famous Person - Sick ×7, docee_event: Organization Fine ×7, docee_event: Strike ×8, docee_event: Shipwreck ×8, docee_event: Mine Collapses ×8, docee_event: Regime Change ×9, docee_event: Air Crash ×9.

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
| Location | 21,182 | 14.96% |
| Date | 10,918 | 7.71% |
| Cause | 5,290 | 3.74% |
| Policy Content | 4,940 | 3.49% |
| Casualties and Losses | 3,769 | 2.66% |
| Contest Participant | 3,460 | 2.44% |
| Participants | 2,974 | 2.10% |
| Suspect | 2,501 | 1.77% |
| Person under Investigation | 2,331 | 1.65% |
| Areas affected | 2,249 | 1.59% |
| Deliberating Agency | 2,137 | 1.51% |
| Investigative Agency | 1,688 | 1.19% |
| Target | 1,576 | 1.11% |
| Attacker | 1,549 | 1.09% |
| Weapon and Equippment | 1,512 | 1.07% |
| Criminal Evidence | 1,506 | 1.06% |
| The Charged Crime | 1,445 | 1.02% |
| Candidates and their Political Parties | 1,425 | 1.01% |
| Game Name | 1,408 | 0.99% |
| Disease | 1,393 | 0.98% |
| Protesters | 1,362 | 0.96% |
| Precautionary Measure | 1,355 | 0.96% |
| Arrest Location | 1,291 | 0.91% |
| Winner | 1,211 | 0.86% |
| Policy Name & Abbreviation | 1,093 | 0.77% |
| Police | 1,083 | 0.76% |
| Affected Area | 1,081 | 0.76% |
| Host | 1,062 | 0.75% |
| Speaker | 1,030 | 0.73% |
| Method | 994 | 0.70% |
| Announcement Date | 969 | 0.68% |
| Confirmed/Infected Cases | 959 | 0.68% |
| Protest Reason | 959 | 0.68% |
| Summit Theme | 950 | 0.67% |
| Award | 911 | 0.64% |
| Champions | 887 | 0.63% |
| Contracting Parties | 845 | 0.60% |
| Anti-pollution People/Organizations | 823 | 0.58% |
| Affected Areas | 807 | 0.57% |
| Death Cases | 785 | 0.55% |
| Location/Hospital | 754 | 0.53% |
| Storm Hit Location | 740 | 0.52% |
| Effective Date | 721 | 0.51% |
| Score | 710 | 0.50% |
| Position | 672 | 0.47% |
| Outbreak Location | 665 | 0.47% |
| Epidemic Data Issuing Agency | 661 | 0.47% |
| Profession | 649 | 0.46% |
| Deceased | 647 | 0.46% |
| Attending Hospital | 634 | 0.45% |
| Protest Slogan | 624 | 0.44% |
| Host Country | 591 | 0.42% |
| Pollution Source | 583 | 0.41% |
| Number of Vehicles Involved in the Crash | 566 | 0.40% |
| Competition Items | 563 | 0.40% |
| Organization | 559 | 0.39% |
| Symptoms | 557 | 0.39% |
| Survivors | 554 | 0.39% |
| News Release Agency | 552 | 0.39% |
| Visitor | 549 | 0.39% |
| Agreement Content | 548 | 0.39% |
| Accusation | 545 | 0.38% |
| Award Field | 544 | 0.38% |
| Speaker Status | 532 | 0.38% |
| Election Name | 526 | 0.37% |
| Beneficiary | 503 | 0.36% |
| Charged Crime | 488 | 0.34% |
| Appointee | 488 | 0.34% |
| Start Time | 479 | 0.34% |
| Country Visited | 472 | 0.33% |
| Damaged Facility | 466 | 0.33% |
| Artifacts and Their Chronology | 436 | 0.31% |
| Susceptible Population | 418 | 0.30% |
| Judgement Result/Prison Term | 418 | 0.30% |
| The Final Votes and Percentages | 416 | 0.29% |
| Spokenmen | 411 | 0.29% |
| Sponsor | 404 | 0.29% |
| Achievement | 391 | 0.28% |
| Arrest Time | 388 | 0.27% |
| Accused People | 360 | 0.25% |
| Aid Agency | 358 | 0.25% |
| Summit Name | 357 | 0.25% |
| Head of Institution | 354 | 0.25% |
| Age | 348 | 0.25% |
| People | 344 | 0.24% |
| Solution | 343 | 0.24% |
| Magnitude | 342 | 0.24% |
| Funding | 333 | 0.24% |
| Way for Spreading | 333 | 0.24% |
| Employment Agency | 333 | 0.24% |
| Affected Industries | 330 | 0.23% |
| Employed Institution | 327 | 0.23% |
| Historical Sites | 326 | 0.23% |
| Agreement Name | 324 | 0.23% |
| Discover Location | 319 | 0.23% |
| Government Reaction | 317 | 0.22% |
| Death Reason | 308 | 0.22% |
| Record Breaker | 301 | 0.21% |
| Aid Reason | 299 | 0.21% |
| Grades | 299 | 0.21% |
| Victim | 299 | 0.21% |
| Court | 284 | 0.20% |
| Head of Investigation Team | 276 | 0.19% |
| Economic Loss | 269 | 0.19% |
| Resignated Person | 263 | 0.19% |
| Conflict Duration | 262 | 0.19% |
| Archaeologist | 254 | 0.18% |
| Record-breaking Project | 244 | 0.17% |
| Party Responsible for Pollution | 243 | 0.17% |
| Related Rivers or Lakes | 239 | 0.17% |
| Released People | 227 | 0.16% |
| Storm Hit Time | 226 | 0.16% |
| Archaeologist Organization | 218 | 0.15% |
| End Time | 216 | 0.15% |
| Areas Affected | 213 | 0.15% |
| MVP | 212 | 0.15% |
| Research Agency | 203 | 0.14% |
| Epicenter | 199 | 0.14% |
| Last Job of the Appointee | 198 | 0.14% |
| Influence People | 197 | 0.14% |
| Spacecraft | 192 | 0.14% |
| Discover Time | 191 | 0.13% |
| Military Strength | 187 | 0.13% |
| Reasons for the Formation of the Historical sites | 183 | 0.13% |
| Appointer | 183 | 0.13% |
| Bill Drafting Agency | 177 | 0.13% |
| Organization Members | 174 | 0.12% |
| Organization Name | 171 | 0.12% |
| Aid Supplies/Amount | 171 | 0.12% |
| Previous Record Holder | 170 | 0.12% |
| Belligerents | 169 | 0.12% |
| Appointment Time | 167 | 0.12% |
| Symptom | 162 | 0.11% |
| State Before Death | 157 | 0.11% |
| Participating Countries | 155 | 0.11% |
| Evidence | 153 | 0.11% |
| Spokeswoman/Spokesman | 153 | 0.11% |
| Illness | 152 | 0.11% |
| Storm Name | 150 | 0.11% |
| Arrested | 147 | 0.10% |
| The Sentence Claimed by the Prosecutor's Lawyer | 147 | 0.10% |
| Resign Reason | 145 | 0.10% |
| The Grades of the Previous Record Holder | 144 | 0.10% |
| Number of People Hospitalized | 143 | 0.10% |
| Witness | 139 | 0.10% |
| Award Reason | 138 | 0.10% |
| Cooperative Agency | 137 | 0.10% |
| Election Goal | 137 | 0.10% |
| Time of the Case | 126 | 0.09% |
| Number of Victims | 126 | 0.09% |
| People/Organization who predicted the disaster | 123 | 0.09% |
| Defense Lawyer | 123 | 0.09% |
| Predecessor | 122 | 0.09% |
| Turnout(who win who lost) | 121 | 0.09% |
| The Final Seats Result | 121 | 0.09% |
| Launch Country | 118 | 0.08% |
| Weapon | 118 | 0.08% |
| Death Rate | 117 | 0.08% |
| Commanders and their Position | 116 | 0.08% |
| Treatment Method | 115 | 0.08% |
| Wife | 114 | 0.08% |
| Candidate Scandal(Allegations of fraud, etc.) | 114 | 0.08% |
| Husband | 111 | 0.08% |
| Spectacle Location | 108 | 0.08% |
| Army | 106 | 0.07% |
| Types of the Spectacle | 105 | 0.07% |
| Start_Date | 96 | 0.07% |
| Missings | 96 | 0.07% |
| Agreement Validity Period | 95 | 0.07% |
| Military Exercise | 94 | 0.07% |
| Doctor and Medical Team | 92 | 0.06% |
| Number of Damaged Houses | 92 | 0.06% |
| Policy Proposals | 91 | 0.06% |
| Ways to Watch the Speech | 90 | 0.06% |
| Perpetrator | 89 | 0.06% |
| Court Time | 89 | 0.06% |
| Declarer | 87 | 0.06% |
| Spacecraft Mission | 85 | 0.06% |
| Prosecutor | 84 | 0.06% |
| Riot Reason | 82 | 0.06% |
| Lasting Time | 80 | 0.06% |
| Release Reason | 79 | 0.06% |
| Country | 78 | 0.06% |
| Fined Agency | 77 | 0.05% |
| Last Time the Record was Broken | 77 | 0.05% |
| Fine Reason | 75 | 0.05% |
| Economists who Predicted the Crisis | 73 | 0.05% |
| Detention Start Time | 72 | 0.05% |
| Launch Date | 71 | 0.05% |
| Join Conditions | 71 | 0.05% |
| Accident Investigator | 71 | 0.05% |
| Sentencing Location | 70 | 0.05% |
| Amount of Precipitation | 70 | 0.05% |
| Age of the Appointee | 70 | 0.05% |
| Number of Evacuated People | 69 | 0.05% |
| Aircraft Agency | 69 | 0.05% |
| Judge | 68 | 0.05% |
| Term of Office | 68 | 0.05% |
| Prosecution Lawyer | 67 | 0.05% |
| Number of Rescued People | 67 | 0.05% |
| Number of Destroyed Building | 64 | 0.05% |
| Strikers | 63 | 0.04% |
| Organization Leader | 61 | 0.04% |
| Inauguration Time | 60 | 0.04% |
| Regulatory Authority | 59 | 0.04% |
| Voting method | 59 | 0.04% |
| Maximum Rainfall | 59 | 0.04% |
| Scale | 57 | 0.04% |
| Maximum Wind Speed | 57 | 0.04% |
| Goal | 57 | 0.04% |
| Acquiree | 54 | 0.04% |
| Crew | 54 | 0.04% |
| Annoncement Date | 53 | 0.04% |
| Strike Reason | 53 | 0.04% |
| Influenced People | 53 | 0.04% |
| Poisoning Type | 52 | 0.04% |
| Astronauts | 52 | 0.04% |
| Unemployed Rate | 51 | 0.04% |
| Penalty Amount | 50 | 0.04% |
| Commanders of the Army | 50 | 0.04% |
| Electoral System | 49 | 0.03% |
| Boycotted Institutions | 48 | 0.03% |
| Rescue Tool or Method | 48 | 0.03% |
| Taking-off Place | 46 | 0.03% |
| Strike Industry | 46 | 0.03% |
| Organization Industry | 45 | 0.03% |
| Damaged Property | 44 | 0.03% |
| Storm Direction | 44 | 0.03% |
| Successor | 44 | 0.03% |
| Launch Site | 43 | 0.03% |
| Damaged Crops & Livestock | 42 | 0.03% |
| Release Time | 42 | 0.03% |
| Acquirer | 42 | 0.03% |
| Invalid Date | 41 | 0.03% |
| Jail Time | 40 | 0.03% |
| Countries Joined the Organization | 40 | 0.03% |
| The Worst-Hit Area | 40 | 0.03% |
| Scheduled Landing Place | 40 | 0.03% |
| Rescue Organizer | 39 | 0.03% |
| Sequelae | 39 | 0.03% |
| Temporary Settlement | 39 | 0.03% |
| Number of Aftershocks | 39 | 0.03% |
| Wedding Venue | 38 | 0.03% |
| Bankrupt Business | 38 | 0.03% |
| Approver | 38 | 0.03% |
| Shipwreck Reason | 37 | 0.03% |
| Disaster-Stricken Farmland | 37 | 0.03% |
| Start Date | 37 | 0.03% |
| Passengers | 37 | 0.03% |
| People Casting Key Votes | 37 | 0.03% |
| Mission Duration | 36 | 0.03% |
| Strike Agency | 35 | 0.02% |
| Agreement Members | 34 | 0.02% |
| Invited Person | 34 | 0.02% |
| Storm Warning Level | 32 | 0.02% |
| Head of the Government | 32 | 0.02% |
| Marriage Duration | 31 | 0.02% |
| Child | 31 | 0.02% |
| Head of the Institution | 31 | 0.02% |
| Production Cuts | 29 | 0.02% |
| Registered Capital | 29 | 0.02% |
| Age of the Resignated Person | 28 | 0.02% |
| Launch Result | 27 | 0.02% |
| Carrier Rocket | 27 | 0.02% |
| Casualities and Losses | 24 | 0.02% |
| Number of Influenced People | 23 | 0.02% |
| Prison Term | 23 | 0.02% |
| Duration | 22 | 0.02% |
| Prison | 22 | 0.02% |
| Postpone Reason | 22 | 0.02% |
| Provisional Government | 21 | 0.01% |
| Ship No. | 20 | 0.01% |
| Acquisition amount | 18 | 0.01% |
| Spectacle Duration | 17 | 0.01% |
| Storm Center Location | 17 | 0.01% |
| Number of Trapped People | 16 | 0.01% |
| How Many Times Get Married | 15 | 0.01% |
| Strikers Status | 15 | 0.01% |
| Spectacle Start Time | 14 | 0.01% |
| Pests | 14 | 0.01% |
| Tear Up Reason | 14 | 0.01% |
| Area Affected | 14 | 0.01% |
| The Agency Who Broke the Agreement | 13 | 0.01% |
| Flight No. | 13 | 0.01% |
| Volcano Name | 12 | 0.01% |
| Announce Platform | 12 | 0.01% |
| Train Agency | 12 | 0.01% |
| Rescue Start Time | 11 | 0.01% |
| Economy Loss | 11 | 0.01% |
| Stolen Amount | 11 | 0.01% |
| Withdraw Reason | 11 | 0.01% |
| Perpetrators | 11 | 0.01% |
| Head of the Merged Organization | 10 | 0.01% |
| Forecasting Agency | 9 | 0.01% |
| Water Level | 9 | 0.01% |
| Best Way to Shoot | 9 | 0.01% |
| End_Date | 8 | 0.01% |
| Strike Outcome | 8 | 0.01% |
| Trapped Depth | 8 | 0.01% |
| Influenced Crops and Livelihood | 7 | 0.00% |
| Hull Location | 7 | 0.00% |
| Postpone Time | 7 | 0.00% |
| Countries Withdrawing from the Organization | 7 | 0.00% |
| Merger Terms | 7 | 0.00% |
| Response Measures | 6 | 0.00% |
| Ship Agency | 5 | 0.00% |
| Spectacle End Time | 5 | 0.00% |
| Weapon Used | 5 | 0.00% |
| Outbreak Date | 5 | 0.00% |
| Magnitude(Tsunami heights) | 5 | 0.00% |
| Child Custody | 5 | 0.00% |
| Wedding Dress Designer | 4 | 0.00% |
| Alternate Landing Place | 4 | 0.00% |
| Transportation | 4 | 0.00% |
| Tsunamis | 4 | 0.00% |
| Investigating agency | 4 | 0.00% |
| The State of the Volcano (Dormant or Active) | 3 | 0.00% |
| Storm Movement Speed | 3 | 0.00% |
| Bank Name | 3 | 0.00% |
| End Date | 3 | 0.00% |
| Last Outbreak Time | 3 | 0.00% |
| Warning Device | 3 | 0.00% |
| Cost | 3 | 0.00% |
| Exit Conditions | 3 | 0.00% |
| Number of Damaged House | 3 | 0.00% |
| Train No. | 3 | 0.00% |
| Property Division | 3 | 0.00% |
| Hull Discovery Time | 2 | 0.00% |
| Fire Warning Level | 2 | 0.00% |
| Service Years | 2 | 0.00% |
| Projects | 2 | 0.00% |
| Trapped Days | 2 | 0.00% |
| Live Broadcast Platform | 2 | 0.00% |
| Investigating Spokesperson | 1 | 0.00% |
| Hostage | 1 | 0.00% |
| Recovered Amount | 1 | 0.00% |
| Lawyer | 1 | 0.00% |
| Time for Dignitaries to Resign | 1 | 0.00% |

#### val

| label | count | share |
| --- | --- | --- |
| Location | 2,674 | 14.91% |
| Date | 1,392 | 7.76% |
| Cause | 683 | 3.81% |
| Casualties and Losses | 538 | 3.00% |
| Policy Content | 536 | 2.99% |
| Contest Participant | 460 | 2.57% |
| Person under Investigation | 366 | 2.04% |
| Suspect | 350 | 1.95% |
| Participants | 279 | 1.56% |
| Areas affected | 277 | 1.54% |
| Deliberating Agency | 245 | 1.37% |
| Investigative Agency | 226 | 1.26% |
| Protesters | 217 | 1.21% |
| Target | 215 | 1.20% |
| Weapon and Equippment | 206 | 1.15% |
| Disease | 204 | 1.14% |
| Criminal Evidence | 201 | 1.12% |
| The Charged Crime | 200 | 1.12% |
| Attacker | 192 | 1.07% |
| Game Name | 189 | 1.05% |
| Candidates and their Political Parties | 178 | 0.99% |
| Precautionary Measure | 175 | 0.98% |
| Arrest Location | 166 | 0.93% |
| Affected Area | 166 | 0.93% |
| Police | 141 | 0.79% |
| Method | 138 | 0.77% |
| Protest Reason | 136 | 0.76% |
| Confirmed/Infected Cases | 136 | 0.76% |
| Announcement Date | 129 | 0.72% |
| Policy Name & Abbreviation | 129 | 0.72% |
| Contracting Parties | 125 | 0.70% |
| Protest Slogan | 118 | 0.66% |
| Award | 115 | 0.64% |
| Effective Date | 107 | 0.60% |
| Epidemic Data Issuing Agency | 104 | 0.58% |
| Anti-pollution People/Organizations | 101 | 0.56% |
| Speaker | 98 | 0.55% |
| Location/Hospital | 98 | 0.55% |
| Host | 97 | 0.54% |
| Attending Hospital | 96 | 0.54% |
| Death Cases | 95 | 0.53% |
| Summit Theme | 94 | 0.52% |
| Position | 92 | 0.51% |
| Champions | 91 | 0.51% |
| Score | 89 | 0.50% |
| Survivors | 88 | 0.49% |
| Number of Vehicles Involved in the Crash | 85 | 0.47% |
| Winner | 83 | 0.46% |
| Agreement Content | 83 | 0.46% |
| Award Field | 81 | 0.45% |
| Outbreak Location | 81 | 0.45% |
| Affected Areas | 81 | 0.45% |
| Accusation | 80 | 0.45% |
| Pollution Source | 78 | 0.43% |
| Affected Industries | 77 | 0.43% |
| Storm Hit Location | 73 | 0.41% |
| Appointee | 72 | 0.40% |
| Susceptible Population | 69 | 0.38% |
| Organization | 66 | 0.37% |
| Damaged Facility | 66 | 0.37% |
| Profession | 65 | 0.36% |
| Charged Crime | 65 | 0.36% |
| Deceased | 63 | 0.35% |
| Beneficiary | 61 | 0.34% |
| Host Country | 60 | 0.33% |
| The Final Votes and Percentages | 59 | 0.33% |
| Start Time | 59 | 0.33% |
| Symptoms | 59 | 0.33% |
| Judgement Result/Prison Term | 58 | 0.32% |
| Competition Items | 54 | 0.30% |
| Way for Spreading | 54 | 0.30% |
| Spokenmen | 53 | 0.30% |
| News Release Agency | 53 | 0.30% |
| Election Name | 52 | 0.29% |
| Head of Institution | 52 | 0.29% |
| Funding | 52 | 0.29% |
| People | 51 | 0.28% |
| Agreement Name | 50 | 0.28% |
| Government Reaction | 50 | 0.28% |
| Speaker Status | 50 | 0.28% |
| Employed Institution | 49 | 0.27% |
| Visitor | 49 | 0.27% |
| Economic Loss | 49 | 0.27% |
| Sponsor | 49 | 0.27% |
| Accused People | 45 | 0.25% |
| Country Visited | 44 | 0.25% |
| Released People | 44 | 0.25% |
| Age | 43 | 0.24% |
| Arrest Time | 42 | 0.23% |
| Summit Name | 41 | 0.23% |
| Arrested | 41 | 0.23% |
| Court | 37 | 0.21% |
| Head of Investigation Team | 37 | 0.21% |
| Death Reason | 37 | 0.21% |
| Belligerents | 37 | 0.21% |
| End Time | 36 | 0.20% |
| Historical Sites | 36 | 0.20% |
| Aid Reason | 36 | 0.20% |
| Solution | 36 | 0.20% |
| Artifacts and Their Chronology | 34 | 0.19% |
| Employment Agency | 34 | 0.19% |
| Record Breaker | 34 | 0.19% |
| Magnitude | 34 | 0.19% |
| Achievement | 34 | 0.19% |
| Conflict Duration | 31 | 0.17% |
| Grades | 31 | 0.17% |
| Aid Agency | 31 | 0.17% |
| Economists who Predicted the Crisis | 31 | 0.17% |
| Number of People Hospitalized | 30 | 0.17% |
| Areas Affected | 30 | 0.17% |
| Resignated Person | 29 | 0.16% |
| Policy Proposals | 29 | 0.16% |
| Victim | 28 | 0.16% |
| Influence People | 28 | 0.16% |
| Party Responsible for Pollution | 27 | 0.15% |
| Symptom | 27 | 0.15% |
| The Sentence Claimed by the Prosecutor's Lawyer | 26 | 0.14% |
| Discover Location | 25 | 0.14% |
| Record-breaking Project | 25 | 0.14% |
| Organization Name | 25 | 0.14% |
| Appointment Time | 25 | 0.14% |
| Evidence | 24 | 0.13% |
| Archaeologist | 24 | 0.13% |
| Military Strength | 24 | 0.13% |
| Number of Victims | 24 | 0.13% |
| Bill Drafting Agency | 23 | 0.13% |
| MVP | 23 | 0.13% |
| Agreement Validity Period | 22 | 0.12% |
| Treatment Method | 22 | 0.12% |
| Predecessor | 22 | 0.12% |
| Last Job of the Appointee | 22 | 0.12% |
| Appointer | 21 | 0.12% |
| Election Goal | 20 | 0.11% |
| Witness | 20 | 0.11% |
| Release Reason | 19 | 0.11% |
| Epicenter | 19 | 0.11% |
| Army | 19 | 0.11% |
| Turnout(who win who lost) | 18 | 0.10% |
| Resign Reason | 18 | 0.10% |
| Related Rivers or Lakes | 18 | 0.10% |
| State Before Death | 18 | 0.10% |
| Time of the Case | 17 | 0.09% |
| Illness | 17 | 0.09% |
| Spacecraft | 17 | 0.09% |
| Defense Lawyer | 16 | 0.09% |
| Detention Start Time | 16 | 0.09% |
| Weapon | 16 | 0.09% |
| Storm Hit Time | 16 | 0.09% |
| Declarer | 16 | 0.09% |
| The Final Seats Result | 16 | 0.09% |
| Death Rate | 15 | 0.08% |
| Court Time | 15 | 0.08% |
| Discover Time | 15 | 0.08% |
| Aircraft Agency | 15 | 0.08% |
| Riot Reason | 15 | 0.08% |
| Archaeologist Organization | 15 | 0.08% |
| Lasting Time | 15 | 0.08% |
| Start_Date | 14 | 0.08% |
| Military Exercise | 14 | 0.08% |
| Organization Members | 14 | 0.08% |
| Research Agency | 14 | 0.08% |
| Reasons for the Formation of the Historical sites | 14 | 0.08% |
| Term of Office | 13 | 0.07% |
| Commanders and their Position | 13 | 0.07% |
| Commanders of the Army | 13 | 0.07% |
| Age of the Appointee | 12 | 0.07% |
| Taking-off Place | 12 | 0.07% |
| Participating Countries | 12 | 0.07% |
| Storm Name | 12 | 0.07% |
| Number of Rescued People | 12 | 0.07% |
| Ways to Watch the Speech | 12 | 0.07% |
| Damaged Crops & Livestock | 12 | 0.07% |
| Previous Record Holder | 11 | 0.06% |
| Prosecutor | 11 | 0.06% |
| Candidate Scandal(Allegations of fraud, etc.) | 11 | 0.06% |
| Scale | 11 | 0.06% |
| Spokeswoman/Spokesman | 11 | 0.06% |
| Strike Reason | 10 | 0.06% |
| Unemployed Rate | 10 | 0.06% |
| Join Conditions | 10 | 0.06% |
| Last Time the Record was Broken | 10 | 0.06% |
| Doctor and Medical Team | 10 | 0.06% |
| Award Reason | 10 | 0.06% |
| Invalid Date | 10 | 0.06% |
| Wife | 10 | 0.06% |
| Husband | 10 | 0.06% |
| Launch Country | 10 | 0.06% |
| Electoral System | 10 | 0.06% |
| Influenced People | 10 | 0.06% |
| Fined Agency | 9 | 0.05% |
| Fine Reason | 9 | 0.05% |
| Penalty Amount | 9 | 0.05% |
| Regulatory Authority | 9 | 0.05% |
| Number of Evacuated People | 9 | 0.05% |
| Missings | 9 | 0.05% |
| Number of Damaged Houses | 9 | 0.05% |
| Annoncement Date | 9 | 0.05% |
| Organization Leader | 9 | 0.05% |
| Acquiree | 9 | 0.05% |
| Country | 9 | 0.05% |
| Inauguration Time | 9 | 0.05% |
| Release Time | 9 | 0.05% |
| Sentencing Location | 8 | 0.04% |
| Scheduled Landing Place | 8 | 0.04% |
| Damaged Property | 8 | 0.04% |
| Launch Date | 8 | 0.04% |
| Spacecraft Mission | 8 | 0.04% |
| People Casting Key Votes | 8 | 0.04% |
| Accident Investigator | 8 | 0.04% |
| Cooperative Agency | 8 | 0.04% |
| Strikers | 7 | 0.04% |
| People/Organization who predicted the disaster | 7 | 0.04% |
| Prosecution Lawyer | 7 | 0.04% |
| Organization Industry | 7 | 0.04% |
| Perpetrators | 7 | 0.04% |
| Judge | 6 | 0.03% |
| Agreement Members | 6 | 0.03% |
| Goal | 6 | 0.03% |
| Sequelae | 6 | 0.03% |
| Voting method | 6 | 0.03% |
| Approver | 6 | 0.03% |
| Acquirer | 6 | 0.03% |
| Poisoning Type | 6 | 0.03% |
| Production Cuts | 6 | 0.03% |
| Perpetrator | 5 | 0.03% |
| Start Date | 5 | 0.03% |
| Strike Industry | 5 | 0.03% |
| Strike Agency | 5 | 0.03% |
| Duration | 5 | 0.03% |
| Bankrupt Business | 5 | 0.03% |
| Wedding Venue | 5 | 0.03% |
| Rescue Organizer | 5 | 0.03% |
| Ship No. | 5 | 0.03% |
| Aid Supplies/Amount | 5 | 0.03% |
| Rescue Tool or Method | 5 | 0.03% |
| Invited Person | 5 | 0.03% |
| Types of the Spectacle | 5 | 0.03% |
| Countries Joined the Organization | 5 | 0.03% |
| Head of the Institution | 5 | 0.03% |
| Prison | 4 | 0.02% |
| Jail Time | 4 | 0.02% |
| Crew | 4 | 0.02% |
| Amount of Precipitation | 4 | 0.02% |
| Maximum Wind Speed | 4 | 0.02% |
| The Grades of the Previous Record Holder | 4 | 0.02% |
| Launch Site | 4 | 0.02% |
| The Worst-Hit Area | 4 | 0.02% |
| Registered Capital | 4 | 0.02% |
| Stolen Amount | 4 | 0.02% |
| Investigating agency | 4 | 0.02% |
| Rescue Start Time | 4 | 0.02% |
| Number of Destroyed Building | 4 | 0.02% |
| Head of the Government | 4 | 0.02% |
| Spectacle Location | 4 | 0.02% |
| Boycotted Institutions | 4 | 0.02% |
| Tear Up Reason | 3 | 0.02% |
| Train Agency | 3 | 0.02% |
| Storm Center Location | 3 | 0.02% |
| Prison Term | 3 | 0.02% |
| Temporary Settlement | 3 | 0.02% |
| Age of the Resignated Person | 3 | 0.02% |
| Passengers | 3 | 0.02% |
| Mission Duration | 3 | 0.02% |
| How Many Times Get Married | 3 | 0.02% |
| Shipwreck Reason | 3 | 0.02% |
| Provisional Government | 3 | 0.02% |
| Storm Direction | 3 | 0.02% |
| Postpone Reason | 3 | 0.02% |
| Strike Outcome | 2 | 0.01% |
| The Agency Who Broke the Agreement | 2 | 0.01% |
| Flight No. | 2 | 0.01% |
| Withdraw Reason | 2 | 0.01% |
| Acquisition amount | 2 | 0.01% |
| Carrier Rocket | 2 | 0.01% |
| Maximum Rainfall | 2 | 0.01% |
| Volcano Name | 2 | 0.01% |
| Outbreak Date | 2 | 0.01% |
| The State of the Volcano (Dormant or Active) | 2 | 0.01% |
| Train No. | 2 | 0.01% |
| Launch Result | 2 | 0.01% |
| Storm Warning Level | 2 | 0.01% |
| Spectacle Start Time | 2 | 0.01% |
| Weapon Used | 2 | 0.01% |
| Transportation | 2 | 0.01% |
| Number of Influenced People | 2 | 0.01% |
| Disaster-Stricken Farmland | 2 | 0.01% |
| End_Date | 2 | 0.01% |
| Head of the Merged Organization | 2 | 0.01% |
| Number of Aftershocks | 1 | 0.01% |
| Countries Withdrawing from the Organization | 1 | 0.01% |
| Economic loss | 1 | 0.01% |
| Recovered Amount | 1 | 0.01% |
| Service Years | 1 | 0.01% |
| Marriage Duration | 1 | 0.01% |
| Storm Movement Speed | 1 | 0.01% |
| Best Way to Shoot | 1 | 0.01% |
| Spectacle End Time | 1 | 0.01% |
| Spectacle Duration | 1 | 0.01% |
| Economy Loss | 1 | 0.01% |
| Hostage | 1 | 0.01% |
| Trapped Days | 1 | 0.01% |
| End Date | 1 | 0.01% |
| Postpone Time | 1 | 0.01% |
| Successor | 1 | 0.01% |
| Water Level | 1 | 0.01% |
| Last Outbreak Time | 1 | 0.01% |
| Strikers Status | 1 | 0.01% |
| Trapped Depth | 1 | 0.01% |
| Number of Trapped People | 1 | 0.01% |
| Bank Name | 1 | 0.01% |

#### test

| label | count | share |
| --- | --- | --- |
| Location | 2,990 | 16.19% |
| Date | 1,466 | 7.94% |
| Cause | 684 | 3.70% |
| Policy Content | 582 | 3.15% |
| Casualties and Losses | 491 | 2.66% |
| Contest Participant | 434 | 2.35% |
| Suspect | 368 | 1.99% |
| Participants | 363 | 1.97% |
| Person under Investigation | 285 | 1.54% |
| Areas affected | 268 | 1.45% |
| Deliberating Agency | 223 | 1.21% |
| Protesters | 223 | 1.21% |
| Weapon and Equippment | 211 | 1.14% |
| Attacker | 205 | 1.11% |
| Game Name | 200 | 1.08% |
| Investigative Agency | 199 | 1.08% |
| Target | 198 | 1.07% |
| Precautionary Measure | 177 | 0.96% |
| Disease | 166 | 0.90% |
| Candidates and their Political Parties | 165 | 0.89% |
| Method | 156 | 0.84% |
| Winner | 155 | 0.84% |
| Host | 152 | 0.82% |
| Criminal Evidence | 152 | 0.82% |
| Protest Reason | 145 | 0.79% |
| Arrest Location | 143 | 0.77% |
| Anti-pollution People/Organizations | 137 | 0.74% |
| The Charged Crime | 135 | 0.73% |
| Policy Name & Abbreviation | 133 | 0.72% |
| Affected Area | 132 | 0.71% |
| Speaker | 130 | 0.70% |
| Beneficiary | 129 | 0.70% |
| Police | 127 | 0.69% |
| Announcement Date | 123 | 0.67% |
| Protest Slogan | 117 | 0.63% |
| Award | 116 | 0.63% |
| Champions | 112 | 0.61% |
| Deceased | 110 | 0.60% |
| Profession | 108 | 0.58% |
| Storm Hit Location | 108 | 0.58% |
| Summit Theme | 107 | 0.58% |
| Effective Date | 105 | 0.57% |
| Pollution Source | 102 | 0.55% |
| Contracting Parties | 101 | 0.55% |
| Position | 100 | 0.54% |
| Confirmed/Infected Cases | 99 | 0.54% |
| Attending Hospital | 98 | 0.53% |
| Score | 98 | 0.53% |
| Location/Hospital | 97 | 0.53% |
| Competition Items | 96 | 0.52% |
| Visitor | 94 | 0.51% |
| Host Country | 85 | 0.46% |
| Death Cases | 82 | 0.44% |
| Affected Areas | 82 | 0.44% |
| Speaker Status | 80 | 0.43% |
| Election Name | 80 | 0.43% |
| Damaged Facility | 78 | 0.42% |
| Organization | 77 | 0.42% |
| Number of Vehicles Involved in the Crash | 77 | 0.42% |
| Appointee | 77 | 0.42% |
| Outbreak Location | 74 | 0.40% |
| Epidemic Data Issuing Agency | 73 | 0.40% |
| Country Visited | 73 | 0.40% |
| Sponsor | 70 | 0.38% |
| Accusation | 69 | 0.37% |
| Survivors | 68 | 0.37% |
| News Release Agency | 67 | 0.36% |
| Agreement Content | 65 | 0.35% |
| Award Field | 65 | 0.35% |
| Spokenmen | 64 | 0.35% |
| Solution | 62 | 0.34% |
| Susceptible Population | 60 | 0.32% |
| Head of Institution | 56 | 0.30% |
| Affected Industries | 56 | 0.30% |
| Charged Crime | 55 | 0.30% |
| Symptoms | 53 | 0.29% |
| Artifacts and Their Chronology | 53 | 0.29% |
| Aid Reason | 53 | 0.29% |
| Government Reaction | 52 | 0.28% |
| Age | 51 | 0.28% |
| Start Time | 51 | 0.28% |
| Judgement Result/Prison Term | 51 | 0.28% |
| Aid Agency | 50 | 0.27% |
| Funding | 50 | 0.27% |
| Resignated Person | 48 | 0.26% |
| Conflict Duration | 47 | 0.25% |
| Death Reason | 46 | 0.25% |
| Way for Spreading | 45 | 0.24% |
| Accused People | 44 | 0.24% |
| Achievement | 44 | 0.24% |
| Economic Loss | 42 | 0.23% |
| Court | 42 | 0.23% |
| Employment Agency | 41 | 0.22% |
| Arrest Time | 40 | 0.22% |
| Discover Location | 39 | 0.21% |
| Storm Hit Time | 38 | 0.21% |
| Agreement Name | 37 | 0.20% |
| Belligerents | 34 | 0.18% |
| The Final Votes and Percentages | 34 | 0.18% |
| Victim | 34 | 0.18% |
| Last Job of the Appointee | 34 | 0.18% |
| MVP | 32 | 0.17% |
| Magnitude | 32 | 0.17% |
| Archaeologist | 31 | 0.17% |
| Historical Sites | 31 | 0.17% |
| Archaeologist Organization | 31 | 0.17% |
| Appointer | 31 | 0.17% |
| Areas Affected | 31 | 0.17% |
| Party Responsible for Pollution | 30 | 0.16% |
| Appointment Time | 29 | 0.16% |
| Head of Investigation Team | 29 | 0.16% |
| Aid Supplies/Amount | 29 | 0.16% |
| Summit Name | 28 | 0.15% |
| Bill Drafting Agency | 28 | 0.15% |
| Time of the Case | 27 | 0.15% |
| Weapon | 27 | 0.15% |
| People | 26 | 0.14% |
| Record Breaker | 26 | 0.14% |
| End Time | 26 | 0.14% |
| Turnout(who win who lost) | 26 | 0.14% |
| Influence People | 26 | 0.14% |
| Related Rivers or Lakes | 26 | 0.14% |
| People/Organization who predicted the disaster | 24 | 0.13% |
| State Before Death | 23 | 0.12% |
| Discover Time | 23 | 0.12% |
| Released People | 23 | 0.12% |
| Arrested | 23 | 0.12% |
| Record-breaking Project | 22 | 0.12% |
| Storm Name | 22 | 0.12% |
| Research Agency | 22 | 0.12% |
| Organization Members | 21 | 0.11% |
| Military Strength | 21 | 0.11% |
| Employed Institution | 21 | 0.11% |
| Spacecraft | 20 | 0.11% |
| Predecessor | 19 | 0.10% |
| Policy Proposals | 19 | 0.10% |
| Epicenter | 18 | 0.10% |
| Reasons for the Formation of the Historical sites | 18 | 0.10% |
| Ways to Watch the Speech | 17 | 0.09% |
| Organization Name | 16 | 0.09% |
| Grades | 16 | 0.09% |
| Cooperative Agency | 16 | 0.09% |
| Riot Reason | 16 | 0.09% |
| Inauguration Time | 15 | 0.08% |
| Military Exercise | 15 | 0.08% |
| Commanders and their Position | 15 | 0.08% |
| Amount of Precipitation | 15 | 0.08% |
| Fined Agency | 15 | 0.08% |
| Maximum Rainfall | 15 | 0.08% |
| Perpetrator | 15 | 0.08% |
| Award Reason | 15 | 0.08% |
| Participating Countries | 15 | 0.08% |
| Death Rate | 14 | 0.08% |
| The Sentence Claimed by the Prosecutor's Lawyer | 14 | 0.08% |
| Number of Evacuated People | 14 | 0.08% |
| Strikers | 14 | 0.08% |
| Number of Rescued People | 13 | 0.07% |
| Aircraft Agency | 13 | 0.07% |
| Spokeswoman/Spokesman | 13 | 0.07% |
| Wife | 12 | 0.06% |
| Doctor and Medical Team | 12 | 0.06% |
| Election Goal | 12 | 0.06% |
| Start_Date | 12 | 0.06% |
| Husband | 11 | 0.06% |
| Release Reason | 11 | 0.06% |
| Resign Reason | 11 | 0.06% |
| Poisoning Type | 11 | 0.06% |
| Defense Lawyer | 11 | 0.06% |
| Number of Damaged Houses | 11 | 0.06% |
| Spacecraft Mission | 11 | 0.06% |
| Rescue Organizer | 11 | 0.06% |
| Agreement Validity Period | 10 | 0.05% |
| Age of the Appointee | 10 | 0.05% |
| Maximum Wind Speed | 10 | 0.05% |
| Strike Reason | 10 | 0.05% |
| Previous Record Holder | 10 | 0.05% |
| Lasting Time | 10 | 0.05% |
| Number of Victims | 10 | 0.05% |
| Witness | 10 | 0.05% |
| Treatment Method | 9 | 0.05% |
| Candidate Scandal(Allegations of fraud, etc.) | 9 | 0.05% |
| Sentencing Location | 9 | 0.05% |
| Missings | 9 | 0.05% |
| Evidence | 9 | 0.05% |
| Registered Capital | 9 | 0.05% |
| Number of People Hospitalized | 9 | 0.05% |
| Damaged Property | 9 | 0.05% |
| Symptom | 8 | 0.04% |
| Country | 8 | 0.04% |
| Launch Country | 8 | 0.04% |
| Start Date | 8 | 0.04% |
| Rescue Tool or Method | 8 | 0.04% |
| Court Time | 8 | 0.04% |
| Prosecutor | 8 | 0.04% |
| Boycotted Institutions | 8 | 0.04% |
| Army | 8 | 0.04% |
| Illness | 7 | 0.04% |
| Fine Reason | 7 | 0.04% |
| Strike Industry | 7 | 0.04% |
| The Grades of the Previous Record Holder | 7 | 0.04% |
| Scheduled Landing Place | 7 | 0.04% |
| Unemployed Rate | 7 | 0.04% |
| Term of Office | 7 | 0.04% |
| Storm Direction | 7 | 0.04% |
| Sequelae | 7 | 0.04% |
| Join Conditions | 7 | 0.04% |
| The Final Seats Result | 6 | 0.03% |
| Release Time | 6 | 0.03% |
| Detention Start Time | 6 | 0.03% |
| Wedding Venue | 6 | 0.03% |
| Casualities and Losses | 6 | 0.03% |
| Last Time the Record was Broken | 6 | 0.03% |
| Declarer | 6 | 0.03% |
| Scale | 6 | 0.03% |
| Organization Industry | 6 | 0.03% |
| Successor | 6 | 0.03% |
| Electoral System | 5 | 0.03% |
| Regulatory Authority | 5 | 0.03% |
| Number of Aftershocks | 5 | 0.03% |
| Duration | 5 | 0.03% |
| Invited Person | 5 | 0.03% |
| Number of Influenced People | 5 | 0.03% |
| Number of Trapped People | 5 | 0.03% |
| Judge | 5 | 0.03% |
| Perpetrators | 5 | 0.03% |
| Voting method | 5 | 0.03% |
| People Casting Key Votes | 5 | 0.03% |
| Invalid Date | 5 | 0.03% |
| How Many Times Get Married | 5 | 0.03% |
| Economists who Predicted the Crisis | 5 | 0.03% |
| Acquiree | 5 | 0.03% |
| Mission Duration | 5 | 0.03% |
| Bank Name | 5 | 0.03% |
| Launch Date | 5 | 0.03% |
| Agreement Members | 5 | 0.03% |
| Launch Result | 4 | 0.02% |
| Prison | 4 | 0.02% |
| Prosecution Lawyer | 4 | 0.02% |
| Number of Destroyed Building | 4 | 0.02% |
| Strike Agency | 4 | 0.02% |
| Jail Time | 4 | 0.02% |
| Announce Platform | 4 | 0.02% |
| Train Agency | 4 | 0.02% |
| Taking-off Place | 4 | 0.02% |
| Annoncement Date | 4 | 0.02% |
| Ship No. | 4 | 0.02% |
| Accident Investigator | 4 | 0.02% |
| Acquirer | 4 | 0.02% |
| Volcano Name | 4 | 0.02% |
| Approver | 4 | 0.02% |
| Organization Leader | 4 | 0.02% |
| Storm Warning Level | 3 | 0.02% |
| Strikers Status | 3 | 0.02% |
| Countries Joined the Organization | 3 | 0.02% |
| Hostage | 3 | 0.02% |
| Bankrupt Business | 3 | 0.02% |
| Crew | 3 | 0.02% |
| Postpone Reason | 3 | 0.02% |
| Commanders of the Army | 3 | 0.02% |
| Storm Center Location | 3 | 0.02% |
| Area Affected | 3 | 0.02% |
| Shipwreck Reason | 3 | 0.02% |
| Disaster-Stricken Farmland | 3 | 0.02% |
| Age of the Resignated Person | 3 | 0.02% |
| Flight No. | 3 | 0.02% |
| Marriage Duration | 2 | 0.01% |
| Goal | 2 | 0.01% |
| Penalty Amount | 2 | 0.01% |
| Rescue Start Time | 2 | 0.01% |
| Weapon Used | 2 | 0.01% |
| Investigating agency | 2 | 0.01% |
| Property Division | 2 | 0.01% |
| Strike Outcome | 2 | 0.01% |
| Passengers | 2 | 0.01% |
| The Worst-Hit Area | 2 | 0.01% |
| Head of the Institution | 2 | 0.01% |
| Outbreak Date | 2 | 0.01% |
| Number of Damaged House | 2 | 0.01% |
| Stolen Amount | 2 | 0.01% |
| Head of the Government | 2 | 0.01% |
| Spectacle Location | 2 | 0.01% |
| The Agency Who Broke the Agreement | 2 | 0.01% |
| Tear Up Reason | 2 | 0.01% |
| Influenced People | 2 | 0.01% |
| Launch Site | 2 | 0.01% |
| Service Years | 1 | 0.01% |
| Countries Withdrawing from the Organization | 1 | 0.01% |
| Withdraw Reason | 1 | 0.01% |
| Acquisition amount | 1 | 0.01% |
| Merger Terms | 1 | 0.01% |
| Head of the Merged Organization | 1 | 0.01% |
| Hull Discovery Time | 1 | 0.01% |
| Temporary Settlement | 1 | 0.01% |
| Alternate Landing Place | 1 | 0.01% |
| Hull Location | 1 | 0.01% |
| Refuge | 1 | 0.01% |
| Recovered Amount | 1 | 0.01% |
| Storm Movement Speed | 1 | 0.01% |
| Prison Term | 1 | 0.01% |
| Provisional Government | 1 | 0.01% |
| Train No. | 1 | 0.01% |
| Time for Dignitaries to Resign | 1 | 0.01% |
| Astronauts | 1 | 0.01% |
| Last Outbreak Time | 1 | 0.01% |

### Relations

None.

### Classifications

**train, docee_event.** Majority class 16.8% of 24,028 answers. The marginal is `true_label`, not the menu.

#### train docee_event

| label | count | share |
| --- | --- | --- |
| none | 4,040 | 16.81% |
| Government Policy Changes | 2,175 | 9.05% |
| CommitCrime - Investigate | 1,531 | 6.37% |
| Sports Competition | 1,266 | 5.27% |
| CommitCrime - Arrest | 1,083 | 4.51% |
| Armed Conflict | 930 | 3.87% |
| Diplomatic Talks | 874 | 3.64% |
| Famous Person - Give a Speech | 829 | 3.45% |
| Disease Outbreaks | 811 | 3.38% |
| Road Crash | 707 | 2.94% |
| Protest | 684 | 2.85% |
| Famous Person - Death | 595 | 2.48% |
| Appoint_Inauguration | 556 | 2.31% |
| Election | 548 | 2.28% |
| Awards Ceremony | 535 | 2.23% |
| Diplomatic Visit | 527 | 2.19% |
| Organization Established | 473 | 1.97% |
| New Achievements in Aerospace | 436 | 1.81% |
| Sign Agreement | 406 | 1.69% |
| CommitCrime - Accuse | 363 | 1.51% |
| Fire | 344 | 1.43% |
| CommitCrime - Sentence | 330 | 1.37% |
| Break Historical Records | 314 | 1.31% |
| Environment Pollution | 308 | 1.28% |
| Financial Crisis | 301 | 1.25% |
| Earthquakes | 268 | 1.12% |
| Financial Aid | 256 | 1.07% |
| New Archeological Discoveries | 239 | 0.99% |
| Resignation_Dismissal | 208 | 0.87% |
| Floods | 203 | 0.84% |
| Storm | 197 | 0.82% |
| Join in an Orgnization | 195 | 0.81% |
| CommitCrime - Release | 132 | 0.55% |
| Famous Person - Sick | 123 | 0.51% |
| Famous Person - Recovered | 93 | 0.39% |
| Military Excercise | 89 | 0.37% |
| Riot | 87 | 0.36% |
| Regime Change | 85 | 0.35% |
| Air Crash | 85 | 0.35% |
| Famous Person - Marriage | 73 | 0.30% |
| New Wonders in Nature | 70 | 0.29% |
| Droughts | 70 | 0.29% |
| Gas Explosion | 66 | 0.27% |
| Organization Fine | 60 | 0.25% |
| Shipwreck | 55 | 0.23% |
| Mudslides | 52 | 0.22% |
| Famous Person - Divorce | 48 | 0.20% |
| Organization Closed | 47 | 0.20% |
| Organization Merge | 42 | 0.17% |
| Mass Poisoning | 42 | 0.17% |
| Strike | 40 | 0.17% |
| Mine Collapses | 34 | 0.14% |
| Train Collisions | 19 | 0.08% |
| Tear Up Agreement | 18 | 0.07% |
| Bank Robbery | 14 | 0.06% |
| Famine | 13 | 0.05% |
| Withdraw from an Orgnization | 12 | 0.05% |
| Volcano Eruption | 11 | 0.05% |
| Insect Disaster | 10 | 0.04% |
| Tsunamis | 6 | 0.02% |

**val, docee_event.** Majority class 17.0% of 3,016 answers. The marginal is `true_label`, not the menu.

#### val docee_event

| label | count | share |
| --- | --- | --- |
| none | 513 | 17.01% |
| Government Policy Changes | 260 | 8.62% |
| CommitCrime - Investigate | 209 | 6.93% |
| Sports Competition | 165 | 5.47% |
| CommitCrime - Arrest | 144 | 4.77% |
| Armed Conflict | 115 | 3.81% |
| Disease Outbreaks | 110 | 3.65% |
| Road Crash | 103 | 3.42% |
| Protest | 102 | 3.38% |
| Diplomatic Talks | 87 | 2.88% |
| Famous Person - Give a Speech | 81 | 2.69% |
| Appoint_Inauguration | 75 | 2.49% |
| Organization Established | 65 | 2.16% |
| Awards Ceremony | 65 | 2.16% |
| Election | 64 | 2.12% |
| Famous Person - Death | 62 | 2.06% |
| Sign Agreement | 56 | 1.86% |
| CommitCrime - Sentence | 51 | 1.69% |
| Diplomatic Visit | 50 | 1.66% |
| New Achievements in Aerospace | 47 | 1.56% |
| Financial Crisis | 47 | 1.56% |
| CommitCrime - Accuse | 46 | 1.53% |
| Fire | 42 | 1.39% |
| Environment Pollution | 38 | 1.26% |
| Financial Aid | 35 | 1.16% |
| Break Historical Records | 34 | 1.13% |
| Resignation_Dismissal | 25 | 0.83% |
| Earthquakes | 25 | 0.83% |
| Join in an Orgnization | 24 | 0.80% |
| CommitCrime - Release | 23 | 0.76% |
| New Archeological Discoveries | 20 | 0.66% |
| Storm | 20 | 0.66% |
| Riot | 18 | 0.60% |
| Famous Person - Sick | 18 | 0.60% |
| Floods | 17 | 0.56% |
| Air Crash | 17 | 0.56% |
| Military Excercise | 13 | 0.43% |
| Famous Person - Recovered | 13 | 0.43% |
| Gas Explosion | 12 | 0.40% |
| Famous Person - Marriage | 11 | 0.36% |
| Regime Change | 11 | 0.36% |
| Mine Collapses | 9 | 0.30% |
| Droughts | 9 | 0.30% |
| Organization Fine | 8 | 0.27% |
| Shipwreck | 7 | 0.23% |
| Train Collisions | 6 | 0.20% |
| Mass Poisoning | 6 | 0.20% |
| Strike | 5 | 0.17% |
| Mudslides | 5 | 0.17% |
| Organization Merge | 5 | 0.17% |
| New Wonders in Nature | 5 | 0.17% |
| Bank Robbery | 4 | 0.13% |
| Organization Closed | 4 | 0.13% |
| Tear Up Agreement | 3 | 0.10% |
| Famous Person - Divorce | 3 | 0.10% |
| Volcano Eruption | 2 | 0.07% |
| Withdraw from an Orgnization | 1 | 0.03% |
| Famine | 1 | 0.03% |

**test, docee_event.** Majority class 17.0% of 3,090 answers. The marginal is `true_label`, not the menu.

#### test docee_event

| label | count | share |
| --- | --- | --- |
| none | 526 | 17.02% |
| Government Policy Changes | 260 | 8.41% |
| CommitCrime - Investigate | 199 | 6.44% |
| Sports Competition | 169 | 5.47% |
| Armed Conflict | 127 | 4.11% |
| CommitCrime - Arrest | 126 | 4.08% |
| Diplomatic Talks | 106 | 3.43% |
| Famous Person - Give a Speech | 103 | 3.33% |
| Disease Outbreaks | 101 | 3.27% |
| Protest | 101 | 3.27% |
| Road Crash | 98 | 3.17% |
| Famous Person - Death | 97 | 3.14% |
| Diplomatic Visit | 74 | 2.39% |
| Election | 74 | 2.39% |
| Appoint_Inauguration | 74 | 2.39% |
| Organization Established | 68 | 2.20% |
| Awards Ceremony | 61 | 1.97% |
| New Achievements in Aerospace | 51 | 1.65% |
| Sign Agreement | 50 | 1.62% |
| Environment Pollution | 48 | 1.55% |
| CommitCrime - Accuse | 44 | 1.42% |
| Fire | 44 | 1.42% |
| Financial Crisis | 41 | 1.33% |
| CommitCrime - Sentence | 40 | 1.29% |
| Financial Aid | 39 | 1.26% |
| Storm | 32 | 1.04% |
| Earthquakes | 31 | 1.00% |
| New Archeological Discoveries | 26 | 0.84% |
| Break Historical Records | 26 | 0.84% |
| Resignation_Dismissal | 25 | 0.81% |
| Floods | 25 | 0.81% |
| Riot | 19 | 0.61% |
| Join in an Orgnization | 16 | 0.52% |
| Military Excercise | 16 | 0.52% |
| CommitCrime - Release | 13 | 0.42% |
| Famous Person - Recovered | 11 | 0.36% |
| Gas Explosion | 10 | 0.32% |
| Mass Poisoning | 10 | 0.32% |
| Famous Person - Marriage | 10 | 0.32% |
| Regime Change | 9 | 0.29% |
| Air Crash | 9 | 0.29% |
| Strike | 8 | 0.26% |
| Shipwreck | 8 | 0.26% |
| Mine Collapses | 8 | 0.26% |
| Famous Person - Sick | 7 | 0.23% |
| Organization Fine | 7 | 0.23% |
| Mudslides | 6 | 0.19% |
| Organization Closed | 5 | 0.16% |
| Train Collisions | 5 | 0.16% |
| Organization Merge | 5 | 0.16% |
| Famous Person - Divorce | 4 | 0.13% |
| Bank Robbery | 4 | 0.13% |
| Droughts | 4 | 0.13% |
| Volcano Eruption | 3 | 0.10% |
| Famine | 2 | 0.06% |
| Tear Up Agreement | 2 | 0.06% |
| Tsunamis | 1 | 0.03% |
| Withdraw from an Orgnization | 1 | 0.03% |
| New Wonders in Nature | 1 | 0.03% |

### Structures

None.

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | llm_real |
| relations | llm_real |
| events | llm_real |
| classifications | llm_real |
| structures | llm_real |
