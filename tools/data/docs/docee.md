# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| docee | train | 21,842 | 21,842 | 189,888 | 0 | 21,842 | 0 | 0 | 0 |
| docee | val | 2,721 | 2,721 | 18,846 | 0 | 2,721 | 0 | 0 | 0 |
| docee | test | 2,744 | 2,744 | 19,078 | 0 | 2,744 | 0 | 0 | 0 |

## docee

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 21,842 | 21,842 | 537 | 1297 | 5513 | 189,888 | 0 | 21,842 | 0 | 0 | 0 |
| val | 2,721 | 2,721 | 547 | 1407 | 5113 | 18,846 | 0 | 2,721 | 0 | 0 | 0 |
| test | 2,744 | 2,744 | 558 | 1384 | 5136 | 19,078 | 0 | 2,744 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 21,826 | 2,719 | 2,742 | 27,287 | 99.93% |
| und | 8 | 2 | 2 | 12 | 0.04% |
| spa | 3 | 0 | 0 | 3 | 0.01% |
| tur | 2 | 0 | 0 | 2 | 0.01% |
| nld | 1 | 0 | 0 | 1 | 0.00% |
| fra | 1 | 0 | 0 | 1 | 0.00% |
| lat | 1 | 0 | 0 | 1 | 0.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 2,744 |
| test_unique_inputs | 2,744 |
| train_records | 21,842 |
| train_unique_inputs | 21,842 |
| val_records | 2,721 |
| val_unique_inputs | 2,721 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.1386 | 189,888 / 18,846 | Date 10%→11%, Location 11%→10%, Confirmed/Infected Cases 0%→1%, Areas affected 1%→1%, Precautionary Measure 0%→1% |
| entities (per mention) | train / test | 0.1560 WARN | 189,888 / 19,078 | Location 11%→9%, Confirmed/Infected Cases 0%→2%, Date 10%→11%, Areas affected 1%→2%, Casualties and Losses 7%→6% |
| entities (per mention) | val / test | 0.1001 | 18,846 / 19,078 | Confirmed/Infected Cases 1%→2%, Location 10%→9%, Suspect 1%→1%, Precautionary Measure 1%→1%, Candidates and their Political Parties 1%→2% |
| classifications (true_label) | train / val | 0.0074 | 21,842 / 2,721 | docee_event: CommitCrime - Arrest 1%→1%, docee_event: CommitCrime - Accuse 4%→4%, docee_event: Election 4%→4%, docee_event: Fire 3%→3%, docee_event: Storm 2%→2% |
| classifications (true_label) | train / test | 0.0096 | 21,842 / 2,744 | docee_event: CommitCrime - Accuse 4%→3%, docee_event: CommitCrime - Arrest 1%→2%, docee_event: Election 4%→4%, docee_event: Withdraw from an Orgnization 1%→1%, docee_event: Shipwreck 1%→1% |
| classifications (true_label) | val / test | 0.0150 | 2,721 / 2,744 | docee_event: CommitCrime - Accuse 4%→3%, docee_event: CommitCrime - Arrest 1%→2%, docee_event: Protest 4%→4%, docee_event: Mudslides 0%→0%, docee_event: Sports Competition 4%→4% |

Thin support in val, entities (per mention): 86 label(s) under 10 gold. State Before Death ×1, Reasons for the Formation of the Historical sites ×1, Merger Terms ×1, The Worst-Hit Area ×1, State of the Hull ×1, Storm Center Location ×1, Storm Formation Location ×1, Sequelae ×1, Recovered Amount ×1, Invalid Date ×1, Spokeswoman/Spokesman ×1, Time for Dignitaries to Resign ×1, People Casting Key Votes ×1, Doctor ×2, Acquisition amount ×2, Economists who Predicted the Crisis ×2, Invited Person ×2, Wedding Dress Designer ×2, Cost ×2, Detention Start Time ×2, Production Cuts ×2, Shipwreck Reason ×2, Storm Formation Time ×2, Head of the Institution ×2, Child Custody ×2, Astronauts ×2, Lawyer ×2, Number of Damaged House ×2, Withdraw Reason ×2, Commanders of the Army ×2, Sentencing Location ×2, How Many Times Get Married ×3, MVP ×3, Organization Leader ×3, Storm Movement Speed ×3, Announce Platform ×3, Economy Loss ×3, Release Reason ×3, End_Date ×4, The Sentence Claimed by the Prosecutor's Lawyer ×4, Tear Up Reason ×4, Damaged Crops & Livestock ×4, Influenced People ×4, Rescue Start Time ×4, Last Time the Record was Broken ×4, Lasting Time ×5, The Sentence Claimed by the Defense Lawyer ×5, Aid Reason ×5, Criminal Evidence ×5, Hull Discovery Time ×5, Agreement Validity Period ×5, Suspected Cases ×5, Cured Rate ×5, Temporary Settlement ×5, Provisional Government ×5, Perpetrator ×6, Archaeologist Organization ×6, Start_Date ×6, Prison ×6, Complications ×6, Warning Device ×6, Electoral System ×6, Head of Institution ×6, Live Broadcast Platform ×6, Unemployed Rate ×7, Doctor and Medical Team ×7, Witness ×7, Storm Direction ×7, Transportation ×7, Number of Trapped People ×7, Cooperative Agency ×7, Last Outbreak Time ×7, Alternate Landing Place ×7, Country ×7, Strike Industry ×7, Damaged Property ×7, News Release Agency ×8, Postpone Reason ×8, Postpone Time ×8, Mission Duration ×8, Launch Result ×8, Refuge ×8, Projects ×8, Appointment Time ×9, Casualities and Losses ×9, Commanders and their Position ×9.
Thin support in val, classifications (true_label): 1 label(s) under 10 gold. docee_event: Regime Change ×7.
Thin support in test, entities (per mention): 89 label(s) under 10 gold. Reasons for the Formation of the Historical sites ×1, Economists who Predicted the Crisis ×1, Start_Date ×1, Join Conditions ×1, Damaged Crops & Livestock ×1, The Worst-Hit Area ×1, Compensation ×1, Storm Movement Speed ×1, Sequelae ×1, Trapped Days ×1, Tsunamis ×1, Head of the Government ×1, Candidate Scandal(Allegations of fraud, etc.) ×1, State Before Death ×2, Archaeologist Organization ×2, Wedding Dress Designer ×2, MVP ×2, Influenced People ×2, Production Cuts ×2, Storm Center Location ×2, Head of the Institution ×2, Hostage ×2, Vaccine Research and Development Organization ×2, Spokeswoman/Spokesman ×2, Number of Damaged House ×2, Warning Device ×2, Commanders of the Army ×2, End Date ×2, Perpetrator ×3, Merger Terms ×3, Postpone Time ×3, Aid Reason ×3, Shipwreck Reason ×3, Treatment Method ×3, Bill Drafting Agency ×3, Refuge ×3, Fire Warning Level ×3, Withdraw Reason ×3, Provisional Government ×3, Economy Loss ×3, Postpone Reason ×4, Ship Agency ×4, Stolen Amount ×4, Announce Platform ×4, Lawyer ×4, Temporary Settlement ×4, Term of Office ×4, Sentencing Location ×4, Live Broadcast Platform ×4, Spectacle End Time ×4, Unemployed Rate ×5, How Many Times Get Married ×5, The Sentence Claimed by the Defense Lawyer ×5, Trapped Depth ×5, Invalid Date ×5, Symptoms ×5, Susceptible Population ×5, Special Medicine ×5, Head of Investigation Team ×5, Alternate Landing Place ×5, Damaged Property ×5, Turnout(who win who lost) ×5, Jail Time ×5, Spectacle Duration ×5, Invited Person ×6, Detention Start Time ×6, Rescue Start Time ×6, State of the Hull ×6, Age of the Appointee ×6, Mission Duration ×6, Approver ×6, Strike Industry ×6, Best Way to Shoot ×6, Bankrupt Business ×7, Doctor and Medical Team ×7, Witness ×7, Lasting Time ×7, Prosecution Lawyer ×7, Storm Direction ×7, Child Custody ×7, Property Division ×7, Policy Name & Abbreviation ×7, Launch Result ×7, Last Outbreak Time ×7, Magnitude(Tsunami heights) ×7, Hull Discovery Time ×8, The State of the Volcano (Dormant or Active) ×8, Rescue Tool or Method ×9, Astronauts ×9.
Thin support in test, classifications (true_label): 1 label(s) under 10 gold. docee_event: Regime Change ×7.

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
| Location | 20,188 | 10.63% |
| Date | 18,206 | 9.59% |
| Casualties and Losses | 13,755 | 7.24% |
| Cause | 6,919 | 3.64% |
| Affected Areas | 2,857 | 1.50% |
| Weapon and Equippment | 2,412 | 1.27% |
| Attacker | 2,322 | 1.22% |
| Contest Participant | 2,280 | 1.20% |
| Protest Reason | 2,260 | 1.19% |
| Protesters | 2,243 | 1.18% |
| Candidates and their Political Parties | 2,207 | 1.16% |
| Charged Crime | 2,123 | 1.12% |
| Target | 2,029 | 1.07% |
| Accident Investigator | 1,970 | 1.04% |
| Belligerents | 1,810 | 0.95% |
| Affected Area | 1,800 | 0.95% |
| Scheduled Landing Place | 1,548 | 0.82% |
| Position | 1,518 | 0.80% |
| Husband | 1,461 | 0.77% |
| Aid Agency | 1,448 | 0.76% |
| Wife | 1,421 | 0.75% |
| Flight No. | 1,409 | 0.74% |
| Accusation | 1,344 | 0.71% |
| Accused People | 1,313 | 0.69% |
| Strike Reason | 1,293 | 0.68% |
| Policy Content | 1,274 | 0.67% |
| Suspect | 1,273 | 0.67% |
| Aircraft Agency | 1,237 | 0.65% |
| Strike Outcome | 1,218 | 0.64% |
| Areas affected | 1,209 | 0.64% |
| Storm Hit Location | 1,176 | 0.62% |
| Influence People | 1,125 | 0.59% |
| Crew | 1,119 | 0.59% |
| Game Name | 1,108 | 0.58% |
| Disease | 1,103 | 0.58% |
| Judgement Result/Prison Term | 1,068 | 0.56% |
| Participants | 1,009 | 0.53% |
| Champions | 925 | 0.49% |
| Organization | 879 | 0.46% |
| Magnitude | 847 | 0.45% |
| Solution | 840 | 0.44% |
| Anti-pollution People/Organizations | 834 | 0.44% |
| Acquiree | 819 | 0.43% |
| Score | 793 | 0.42% |
| Policy Proposals | 789 | 0.42% |
| The Final Votes and Percentages | 785 | 0.41% |
| Confirmed/Infected Cases | 768 | 0.40% |
| Epicenter | 737 | 0.39% |
| Economic Loss | 714 | 0.38% |
| Strike Agency | 701 | 0.37% |
| Start Date | 700 | 0.37% |
| Passengers | 692 | 0.36% |
| Resignated Person | 691 | 0.36% |
| Deliberating Agency | 690 | 0.36% |
| Time of the Case | 687 | 0.36% |
| Summit Theme | 686 | 0.36% |
| Prosecutor | 684 | 0.36% |
| Survivors | 677 | 0.36% |
| Contracting Parties | 666 | 0.35% |
| Election Name | 663 | 0.35% |
| Pollution Source | 655 | 0.34% |
| Precautionary Measure | 649 | 0.34% |
| Riot Reason | 648 | 0.34% |
| Award | 618 | 0.33% |
| Spectacle Location | 616 | 0.32% |
| Number of Vehicles Involved in the Crash | 596 | 0.31% |
| Resign Reason | 593 | 0.31% |
| Location/Hospital | 591 | 0.31% |
| Country Visited | 587 | 0.31% |
| Damaged Facility | 586 | 0.31% |
| Death Cases | 583 | 0.31% |
| Spectacle Start Time | 579 | 0.30% |
| Strikers Status | 576 | 0.30% |
| Weapon | 565 | 0.30% |
| Method | 565 | 0.30% |
| Winner | 553 | 0.29% |
| Organization Name | 534 | 0.28% |
| Court | 532 | 0.28% |
| Aid Supplies/Amount | 530 | 0.28% |
| Types of the Spectacle | 512 | 0.27% |
| Deceased | 493 | 0.26% |
| Competition Items | 478 | 0.25% |
| Appointee | 475 | 0.25% |
| People/Organization who predicted the disaster | 474 | 0.25% |
| Profession | 466 | 0.25% |
| Evidence | 461 | 0.24% |
| The Agency Who Broke the Agreement | 455 | 0.24% |
| Participating Countries | 452 | 0.24% |
| Number of Damaged Houses | 450 | 0.24% |
| Strikers | 449 | 0.24% |
| Record Breaker | 445 | 0.23% |
| Taking-off Place | 443 | 0.23% |
| Record-breaking Project | 439 | 0.23% |
| Number of Evacuated People | 436 | 0.23% |
| Attending Hospital | 419 | 0.22% |
| Employment Agency | 418 | 0.22% |
| Discover Location | 412 | 0.22% |
| The Charged Crime | 412 | 0.22% |
| Grades | 410 | 0.22% |
| Effective Date | 397 | 0.21% |
| Spacecraft | 396 | 0.21% |
| Speaker Status | 387 | 0.20% |
| Award Reason | 381 | 0.20% |
| Artifacts and Their Chronology | 376 | 0.20% |
| Announcement Date | 376 | 0.20% |
| Army | 376 | 0.20% |
| Boycotted Institutions | 371 | 0.20% |
| Death Reason | 369 | 0.19% |
| Award Field | 365 | 0.19% |
| Outbreak Location | 361 | 0.19% |
| Storm Warning Level | 360 | 0.19% |
| People | 357 | 0.19% |
| Agreement Name | 357 | 0.19% |
| Fine Reason | 354 | 0.19% |
| Agreement Content | 353 | 0.19% |
| Defense Lawyer | 345 | 0.18% |
| Arrested | 344 | 0.18% |
| Area Affected | 341 | 0.18% |
| Storm Hit Time | 340 | 0.18% |
| Countries Withdrawing from the Organization | 340 | 0.18% |
| Bank Name | 337 | 0.18% |
| Visitor | 334 | 0.18% |
| Host | 327 | 0.17% |
| Storm Name | 326 | 0.17% |
| Released People | 325 | 0.17% |
| Fined Agency | 319 | 0.17% |
| Marriage Duration | 318 | 0.17% |
| Achievement | 317 | 0.17% |
| Child | 317 | 0.17% |
| End Date | 317 | 0.17% |
| Age | 315 | 0.17% |
| Perpetrators | 313 | 0.16% |
| Epidemic Data Issuing Agency | 312 | 0.16% |
| Pests | 311 | 0.16% |
| Historical Sites | 307 | 0.16% |
| End Time | 307 | 0.16% |
| Affected Industries | 306 | 0.16% |
| Organization Industry | 305 | 0.16% |
| Sponsor | 301 | 0.16% |
| Previous Record Holder | 301 | 0.16% |
| Speaker | 299 | 0.16% |
| Conflict Duration | 298 | 0.16% |
| Launch Date | 296 | 0.16% |
| Investigative Agency | 292 | 0.15% |
| Poisoning Type | 287 | 0.15% |
| Way for Spreading | 284 | 0.15% |
| Regulatory Authority | 281 | 0.15% |
| Protest Slogan | 280 | 0.15% |
| Missings | 276 | 0.15% |
| Outbreak Date | 274 | 0.14% |
| Maximum Wind Speed | 272 | 0.14% |
| Volcano Name | 271 | 0.14% |
| Wedding Venue | 270 | 0.14% |
| Police | 270 | 0.14% |
| Number of Influenced People | 268 | 0.14% |
| Investigating agency | 261 | 0.14% |
| Penalty Amount | 260 | 0.14% |
| The Final Seats Result | 260 | 0.14% |
| Person under Investigation | 256 | 0.13% |
| Court Time | 252 | 0.13% |
| Military Strength | 252 | 0.13% |
| Election Goal | 249 | 0.13% |
| Military Exercise | 247 | 0.13% |
| Last Job of the Appointee | 246 | 0.13% |
| Approver | 242 | 0.13% |
| Arrest Time | 238 | 0.13% |
| Goal | 236 | 0.12% |
| Party Responsible for Pollution | 231 | 0.12% |
| Declarer | 228 | 0.12% |
| Number of Destroyed Building | 224 | 0.12% |
| Government Reaction | 224 | 0.12% |
| Victim | 223 | 0.12% |
| Judge | 218 | 0.11% |
| Beneficiary | 217 | 0.11% |
| Spacecraft Mission | 215 | 0.11% |
| Areas Affected | 213 | 0.11% |
| Predecessor | 211 | 0.11% |
| Research Agency | 210 | 0.11% |
| Start Time | 207 | 0.11% |
| Release Reason | 207 | 0.11% |
| Arrest Location | 204 | 0.11% |
| Symptoms | 202 | 0.11% |
| Rescue Organizer | 200 | 0.11% |
| The Grades of the Previous Record Holder | 199 | 0.10% |
| Amount of Precipitation | 191 | 0.10% |
| Duration | 190 | 0.10% |
| Related Rivers or Lakes | 189 | 0.10% |
| Water Level | 184 | 0.10% |
| Symptom | 183 | 0.10% |
| Host Country | 173 | 0.09% |
| Funding | 172 | 0.09% |
| Witness | 171 | 0.09% |
| Criminal Evidence | 171 | 0.09% |
| Term of Office | 170 | 0.09% |
| Property Division | 168 | 0.09% |
| Weapon Used | 167 | 0.09% |
| Archaeologist | 165 | 0.09% |
| Number of Rescued People | 165 | 0.09% |
| Economic loss | 162 | 0.09% |
| Spokeswoman/Spokesman | 162 | 0.09% |
| Prosecution Lawyer | 160 | 0.08% |
| Ship No. | 160 | 0.08% |
| Train No. | 157 | 0.08% |
| Number of Victims | 155 | 0.08% |
| Annoncement Date | 154 | 0.08% |
| Temporary Settlement | 153 | 0.08% |
| Carrier Rocket | 150 | 0.08% |
| Invited Person | 149 | 0.08% |
| Projects | 149 | 0.08% |
| Countries Joined the Organization | 148 | 0.08% |
| Policy Name & Abbreviation | 146 | 0.08% |
| Influenced Crops and Livelihood | 146 | 0.08% |
| Scale | 146 | 0.08% |
| Jail Time | 145 | 0.08% |
| Maximum Rainfall | 143 | 0.08% |
| Sentencing Location | 143 | 0.08% |
| Spectacle Duration | 143 | 0.08% |
| Tear Up Reason | 141 | 0.07% |
| Launch Site | 140 | 0.07% |
| Successor | 140 | 0.07% |
| Start_Date | 137 | 0.07% |
| Prison Term | 136 | 0.07% |
| Last Time the Record was Broken | 136 | 0.07% |
| Discover Time | 134 | 0.07% |
| Treatment Method | 134 | 0.07% |
| Head of the Merged Organization | 133 | 0.07% |
| Summit Name | 133 | 0.07% |
| Appointer | 132 | 0.07% |
| Unemployed Rate | 131 | 0.07% |
| Aid Reason | 131 | 0.07% |
| Release Time | 130 | 0.07% |
| Acquisition amount | 128 | 0.07% |
| Transportation | 125 | 0.07% |
| Rescue Tool or Method | 124 | 0.07% |
| Turnout(who win who lost) | 123 | 0.06% |
| Lasting Time | 122 | 0.06% |
| Forecasting Agency | 121 | 0.06% |
| Appointment Time | 119 | 0.06% |
| Age of the Appointee | 119 | 0.06% |
| Ways to Watch the Speech | 118 | 0.06% |
| Susceptible Population | 117 | 0.06% |
| Electoral System | 117 | 0.06% |
| Withdraw Reason | 116 | 0.06% |
| News Release Agency | 115 | 0.06% |
| Storm Direction | 112 | 0.06% |
| Inauguration Time | 112 | 0.06% |
| Commanders and their Position | 112 | 0.06% |
| Acquirer | 111 | 0.06% |
| Alternate Landing Place | 111 | 0.06% |
| Shipwreck Reason | 110 | 0.06% |
| Strike Industry | 109 | 0.06% |
| Voting method | 109 | 0.06% |
| MVP | 108 | 0.06% |
| Response Measures | 108 | 0.06% |
| Age of the Resignated Person | 108 | 0.06% |
| Damaged Property | 108 | 0.06% |
| Bill Drafting Agency | 107 | 0.06% |
| Number of Trapped People | 105 | 0.06% |
| Casualities and Losses | 105 | 0.06% |
| Candidate Scandal(Allegations of fraud, etc.) | 105 | 0.06% |
| Train Agency | 105 | 0.06% |
| Head of Institution | 104 | 0.05% |
| Archaeologist Organization | 102 | 0.05% |
| Storm Movement Speed | 101 | 0.05% |
| Launch Result | 101 | 0.05% |
| Doctor and Medical Team | 100 | 0.05% |
| Perpetrator | 99 | 0.05% |
| Spectacle End Time | 99 | 0.05% |
| Last Outbreak Time | 97 | 0.05% |
| Disaster-Stricken Farmland | 96 | 0.05% |
| Employed Institution | 95 | 0.05% |
| Fire Warning Level | 92 | 0.05% |
| Magnitude(Tsunami heights) | 91 | 0.05% |
| Number of Aftershocks | 91 | 0.05% |
| Cooperative Agency | 90 | 0.05% |
| Launch Country | 89 | 0.05% |
| Child Custody | 88 | 0.05% |
| Number of Vaccinated People | 88 | 0.05% |
| Spokenmen | 87 | 0.05% |
| Illness | 85 | 0.04% |
| Number of People Hospitalized | 85 | 0.04% |
| Death Rate | 85 | 0.04% |
| Bankrupt Business | 82 | 0.04% |
| Stolen Amount | 79 | 0.04% |
| Organization Members | 77 | 0.04% |
| Hostage | 75 | 0.04% |
| Service Years | 75 | 0.04% |
| Head of the Institution | 74 | 0.04% |
| The State of the Volcano (Dormant or Active) | 74 | 0.04% |
| Complications | 73 | 0.04% |
| Head of Investigation Team | 73 | 0.04% |
| Best Way to Shoot | 73 | 0.04% |
| Prison | 72 | 0.04% |
| Country | 71 | 0.04% |
| Postpone Reason | 70 | 0.04% |
| Announce Platform | 69 | 0.04% |
| Organization Leader | 67 | 0.04% |
| Agreement Validity Period | 65 | 0.03% |
| Astronauts | 65 | 0.03% |
| Investigating Spokesperson | 64 | 0.03% |
| Merger Terms | 63 | 0.03% |
| Warning Device | 63 | 0.03% |
| Compensation | 61 | 0.03% |
| Damaged Crops & Livestock | 60 | 0.03% |
| Cured Cases | 59 | 0.03% |
| Mission Duration | 59 | 0.03% |
| Recovered Amount | 57 | 0.03% |
| State Before Death | 55 | 0.03% |
| Influenced People | 55 | 0.03% |
| Special Medicine | 54 | 0.03% |
| Detention Start Time | 51 | 0.03% |
| Wedding Dress Designer | 49 | 0.03% |
| Hull Location | 49 | 0.03% |
| Refuge | 47 | 0.02% |
| Trapped Depth | 46 | 0.02% |
| Storm Center Location | 44 | 0.02% |
| Agreement Members | 41 | 0.02% |
| Economy Loss | 41 | 0.02% |
| Suspected Cases | 37 | 0.02% |
| End_Date | 36 | 0.02% |
| The Sentence Claimed by the Prosecutor's Lawyer | 35 | 0.02% |
| Postpone Time | 33 | 0.02% |
| Hull Discovery Time | 31 | 0.02% |
| Live Broadcast Platform | 31 | 0.02% |
| Reasons for the Formation of the Historical sites | 30 | 0.02% |
| Lawyer | 30 | 0.02% |
| Economists who Predicted the Crisis | 29 | 0.02% |
| Number of Damaged House | 29 | 0.02% |
| Provisional Government | 29 | 0.02% |
| Tsunamis | 26 | 0.01% |
| Head of the Government | 26 | 0.01% |
| People Casting Key Votes | 26 | 0.01% |
| Invalid Date | 25 | 0.01% |
| Commanders of the Army | 25 | 0.01% |
| The Worst-Hit Area | 24 | 0.01% |
| Rescue Start Time | 23 | 0.01% |
| Trapped Days | 23 | 0.01% |
| How Many Times Get Married | 20 | 0.01% |
| Ship Agency | 20 | 0.01% |
| Registered Capital | 19 | 0.01% |
| Join Conditions | 18 | 0.01% |
| Sequelae | 18 | 0.01% |
| Cured Rate | 15 | 0.01% |
| Exit Conditions | 15 | 0.01% |
| Vaccine Research and Development Organization | 11 | 0.01% |
| Cost | 10 | 0.01% |
| The Sentence Claimed by the Defense Lawyer | 10 | 0.01% |
| Storm Formation Location | 10 | 0.01% |
| Doctor | 8 | 0.00% |
| Tsunami Warning Level | 8 | 0.00% |
| Production Cuts | 6 | 0.00% |
| State of the Hull | 6 | 0.00% |
| Time for Dignitaries to Resign | 6 | 0.00% |
| Storm Formation Time | 4 | 0.00% |
| Number of Rebuilding House | 3 | 0.00% |
| Lost Contact Time | 1 | 0.00% |

#### val

| label | count | share |
| --- | --- | --- |
| Date | 2,083 | 11.05% |
| Location | 1,805 | 9.58% |
| Casualties and Losses | 1,225 | 6.50% |
| Cause | 745 | 3.95% |
| Affected Areas | 297 | 1.58% |
| Areas affected | 276 | 1.46% |
| Belligerents | 267 | 1.42% |
| Confirmed/Infected Cases | 236 | 1.25% |
| Weapon and Equippment | 232 | 1.23% |
| Candidates and their Political Parties | 216 | 1.15% |
| Precautionary Measure | 214 | 1.14% |
| Disease | 206 | 1.09% |
| Affected Area | 189 | 1.00% |
| Attacker | 173 | 0.92% |
| Charged Crime | 172 | 0.91% |
| Target | 166 | 0.88% |
| Policy Proposals | 163 | 0.86% |
| Protest Reason | 162 | 0.86% |
| Protesters | 161 | 0.85% |
| Accident Investigator | 148 | 0.79% |
| Influence People | 145 | 0.77% |
| Flight No. | 145 | 0.77% |
| Aircraft Agency | 144 | 0.76% |
| Death Cases | 140 | 0.74% |
| Accused People | 136 | 0.72% |
| Contest Participant | 126 | 0.67% |
| Spectacle Start Time | 126 | 0.67% |
| Policy Content | 125 | 0.66% |
| Scheduled Landing Place | 123 | 0.65% |
| Game Name | 117 | 0.62% |
| Suspect | 113 | 0.60% |
| Magnitude | 110 | 0.58% |
| Wife | 109 | 0.58% |
| Husband | 106 | 0.56% |
| Economic Loss | 105 | 0.56% |
| Spectacle Location | 105 | 0.56% |
| Position | 103 | 0.55% |
| Storm Hit Location | 100 | 0.53% |
| Organization | 100 | 0.53% |
| Participants | 99 | 0.53% |
| Aid Agency | 99 | 0.53% |
| Accusation | 93 | 0.49% |
| Judgement Result/Prison Term | 89 | 0.47% |
| Start Time | 88 | 0.47% |
| Types of the Spectacle | 88 | 0.47% |
| Time of the Case | 85 | 0.45% |
| Artifacts and Their Chronology | 80 | 0.42% |
| Survivors | 80 | 0.42% |
| Strike Reason | 79 | 0.42% |
| Number of Damaged Houses | 78 | 0.41% |
| Way for Spreading | 75 | 0.40% |
| Anti-pollution People/Organizations | 75 | 0.40% |
| Epidemic Data Issuing Agency | 72 | 0.38% |
| Crew | 72 | 0.38% |
| Location/Hospital | 71 | 0.38% |
| Contracting Parties | 71 | 0.38% |
| Riot Reason | 70 | 0.37% |
| Pollution Source | 69 | 0.37% |
| People | 68 | 0.36% |
| Weapon | 66 | 0.35% |
| Resignated Person | 66 | 0.35% |
| The Final Votes and Percentages | 66 | 0.35% |
| Election Name | 66 | 0.35% |
| Acquiree | 62 | 0.33% |
| End Time | 62 | 0.33% |
| Champions | 60 | 0.32% |
| Strikers | 60 | 0.32% |
| Number of Evacuated People | 59 | 0.31% |
| Discover Location | 55 | 0.29% |
| Winner | 55 | 0.29% |
| Deceased | 52 | 0.28% |
| Competition Items | 52 | 0.28% |
| Award | 52 | 0.28% |
| Attending Hospital | 52 | 0.28% |
| Appointee | 52 | 0.28% |
| Epicenter | 50 | 0.27% |
| Host Country | 49 | 0.26% |
| Person under Investigation | 49 | 0.26% |
| Record-breaking Project | 48 | 0.25% |
| Record Breaker | 48 | 0.25% |
| Profession | 47 | 0.25% |
| Country Visited | 47 | 0.25% |
| Aid Supplies/Amount | 47 | 0.25% |
| Organization Name | 46 | 0.24% |
| Court | 46 | 0.24% |
| Employment Agency | 46 | 0.24% |
| The Agency Who Broke the Agreement | 44 | 0.23% |
| Strikers Status | 44 | 0.23% |
| Number of Influenced People | 43 | 0.23% |
| Age | 42 | 0.22% |
| Effective Date | 42 | 0.22% |
| Cured Cases | 42 | 0.22% |
| Launch Date | 42 | 0.22% |
| Spacecraft | 42 | 0.22% |
| Start Date | 42 | 0.22% |
| Participating Countries | 42 | 0.22% |
| Host | 41 | 0.22% |
| Announcement Date | 41 | 0.22% |
| Pests | 41 | 0.22% |
| Solution | 41 | 0.22% |
| Grades | 41 | 0.22% |
| Number of Vehicles Involved in the Crash | 40 | 0.21% |
| Boycotted Institutions | 40 | 0.21% |
| Death Reason | 39 | 0.21% |
| Score | 39 | 0.21% |
| Number of Destroyed Building | 39 | 0.21% |
| Speaker Status | 38 | 0.20% |
| Taking-off Place | 38 | 0.20% |
| Area Affected | 38 | 0.20% |
| Resign Reason | 38 | 0.20% |
| Army | 38 | 0.20% |
| Affected Industries | 37 | 0.20% |
| Arrested | 37 | 0.20% |
| Method | 37 | 0.20% |
| People/Organization who predicted the disaster | 36 | 0.19% |
| Damaged Facility | 36 | 0.19% |
| Visitor | 35 | 0.19% |
| Agreement Content | 35 | 0.19% |
| Summit Theme | 35 | 0.19% |
| Deliberating Agency | 35 | 0.19% |
| Number of Vaccinated People | 35 | 0.19% |
| Released People | 35 | 0.19% |
| Speaker | 33 | 0.18% |
| Penalty Amount | 33 | 0.18% |
| Passengers | 33 | 0.18% |
| Military Exercise | 33 | 0.18% |
| Marriage Duration | 32 | 0.17% |
| Volcano Name | 32 | 0.17% |
| Previous Record Holder | 32 | 0.17% |
| Symptom | 31 | 0.16% |
| Missings | 31 | 0.16% |
| Perpetrators | 31 | 0.16% |
| Investigative Agency | 31 | 0.16% |
| Maximum Rainfall | 31 | 0.16% |
| Poisoning Type | 31 | 0.16% |
| Spokenmen | 31 | 0.16% |
| Organization Members | 30 | 0.16% |
| Symptoms | 30 | 0.16% |
| Outbreak Date | 30 | 0.16% |
| Agreement Name | 29 | 0.15% |
| Ship No. | 29 | 0.15% |
| Bank Name | 29 | 0.15% |
| Fined Agency | 29 | 0.15% |
| Government Reaction | 29 | 0.15% |
| Award Reason | 28 | 0.15% |
| Storm Name | 28 | 0.15% |
| Storm Hit Time | 28 | 0.15% |
| Fine Reason | 28 | 0.15% |
| Water Level | 28 | 0.15% |
| Award Field | 27 | 0.14% |
| Influenced Crops and Livelihood | 27 | 0.14% |
| Countries Withdrawing from the Organization | 27 | 0.14% |
| Forecasting Agency | 27 | 0.14% |
| Ways to Watch the Speech | 26 | 0.14% |
| Declarer | 26 | 0.14% |
| Prosecutor | 26 | 0.14% |
| Organization Industry | 25 | 0.13% |
| Child | 25 | 0.13% |
| Carrier Rocket | 25 | 0.13% |
| Research Agency | 25 | 0.13% |
| Strike Agency | 25 | 0.13% |
| Strike Outcome | 24 | 0.13% |
| Evidence | 24 | 0.13% |
| Judge | 23 | 0.12% |
| Property Division | 23 | 0.12% |
| Special Medicine | 23 | 0.12% |
| Release Time | 23 | 0.12% |
| Historical Sites | 22 | 0.12% |
| Last Job of the Appointee | 22 | 0.12% |
| The Grades of the Previous Record Holder | 22 | 0.12% |
| Archaeologist | 21 | 0.11% |
| Defense Lawyer | 21 | 0.11% |
| Appointer | 21 | 0.11% |
| Regulatory Authority | 21 | 0.11% |
| Conflict Duration | 21 | 0.11% |
| Spectacle Duration | 21 | 0.11% |
| Discover Time | 20 | 0.11% |
| Beneficiary | 20 | 0.11% |
| Funding | 20 | 0.11% |
| Arrest Location | 20 | 0.11% |
| Economic loss | 20 | 0.11% |
| Outbreak Location | 20 | 0.11% |
| The Final Seats Result | 20 | 0.11% |
| Maximum Wind Speed | 19 | 0.10% |
| Turnout(who win who lost) | 19 | 0.10% |
| Prison Term | 19 | 0.10% |
| Train No. | 19 | 0.10% |
| Acquirer | 18 | 0.10% |
| Wedding Venue | 18 | 0.10% |
| Launch Site | 18 | 0.10% |
| Head of the Merged Organization | 17 | 0.09% |
| Areas Affected | 17 | 0.09% |
| Inauguration Time | 17 | 0.09% |
| Response Measures | 17 | 0.09% |
| Duration | 17 | 0.09% |
| Court Time | 16 | 0.08% |
| Related Rivers or Lakes | 16 | 0.08% |
| Arrest Time | 16 | 0.08% |
| Police | 16 | 0.08% |
| Predecessor | 16 | 0.08% |
| Hull Location | 15 | 0.08% |
| Weapon Used | 15 | 0.08% |
| Investigating agency | 15 | 0.08% |
| Susceptible Population | 15 | 0.08% |
| Head of Investigation Team | 15 | 0.08% |
| Disaster-Stricken Farmland | 15 | 0.08% |
| Successor | 15 | 0.08% |
| Military Strength | 15 | 0.08% |
| Goal | 15 | 0.08% |
| Number of Aftershocks | 15 | 0.08% |
| Number of Victims | 15 | 0.08% |
| Spectacle End Time | 15 | 0.08% |
| Victim | 14 | 0.07% |
| Sponsor | 14 | 0.07% |
| Agreement Members | 14 | 0.07% |
| Stolen Amount | 14 | 0.07% |
| Age of the Appointee | 14 | 0.07% |
| Election Goal | 14 | 0.07% |
| Voting method | 14 | 0.07% |
| Prosecution Lawyer | 13 | 0.07% |
| Ship Agency | 13 | 0.07% |
| Death Rate | 13 | 0.07% |
| Spacecraft Mission | 13 | 0.07% |
| Fire Warning Level | 13 | 0.07% |
| Annoncement Date | 13 | 0.07% |
| Age of the Resignated Person | 13 | 0.07% |
| Protest Slogan | 13 | 0.07% |
| Treatment Method | 12 | 0.06% |
| Summit Name | 12 | 0.06% |
| Bill Drafting Agency | 12 | 0.06% |
| Policy Name & Abbreviation | 12 | 0.06% |
| Launch Country | 12 | 0.06% |
| Service Years | 12 | 0.06% |
| Jail Time | 12 | 0.06% |
| Countries Joined the Organization | 11 | 0.06% |
| Compensation | 11 | 0.06% |
| Storm Warning Level | 11 | 0.06% |
| Amount of Precipitation | 11 | 0.06% |
| Number of Rescued People | 11 | 0.06% |
| The State of the Volcano (Dormant or Active) | 11 | 0.06% |
| Scale | 11 | 0.06% |
| Train Agency | 11 | 0.06% |
| Bankrupt Business | 10 | 0.05% |
| Employed Institution | 10 | 0.05% |
| The Charged Crime | 10 | 0.05% |
| Rescue Tool or Method | 10 | 0.05% |
| Rescue Organizer | 10 | 0.05% |
| Investigating Spokesperson | 10 | 0.05% |
| Term of Office | 10 | 0.05% |
| Achievement | 10 | 0.05% |
| Number of People Hospitalized | 10 | 0.05% |
| Magnitude(Tsunami heights) | 10 | 0.05% |
| Approver | 10 | 0.05% |
| End Date | 10 | 0.05% |
| Party Responsible for Pollution | 10 | 0.05% |
| Best Way to Shoot | 10 | 0.05% |
| Appointment Time | 9 | 0.05% |
| Casualities and Losses | 9 | 0.05% |
| Commanders and their Position | 9 | 0.05% |
| News Release Agency | 8 | 0.04% |
| Postpone Reason | 8 | 0.04% |
| Postpone Time | 8 | 0.04% |
| Mission Duration | 8 | 0.04% |
| Launch Result | 8 | 0.04% |
| Refuge | 8 | 0.04% |
| Projects | 8 | 0.04% |
| Unemployed Rate | 7 | 0.04% |
| Doctor and Medical Team | 7 | 0.04% |
| Witness | 7 | 0.04% |
| Storm Direction | 7 | 0.04% |
| Transportation | 7 | 0.04% |
| Number of Trapped People | 7 | 0.04% |
| Cooperative Agency | 7 | 0.04% |
| Last Outbreak Time | 7 | 0.04% |
| Alternate Landing Place | 7 | 0.04% |
| Country | 7 | 0.04% |
| Strike Industry | 7 | 0.04% |
| Damaged Property | 7 | 0.04% |
| Perpetrator | 6 | 0.03% |
| Archaeologist Organization | 6 | 0.03% |
| Start_Date | 6 | 0.03% |
| Prison | 6 | 0.03% |
| Complications | 6 | 0.03% |
| Warning Device | 6 | 0.03% |
| Electoral System | 6 | 0.03% |
| Head of Institution | 6 | 0.03% |
| Live Broadcast Platform | 6 | 0.03% |
| Lasting Time | 5 | 0.03% |
| The Sentence Claimed by the Defense Lawyer | 5 | 0.03% |
| Aid Reason | 5 | 0.03% |
| Criminal Evidence | 5 | 0.03% |
| Hull Discovery Time | 5 | 0.03% |
| Agreement Validity Period | 5 | 0.03% |
| Suspected Cases | 5 | 0.03% |
| Cured Rate | 5 | 0.03% |
| Temporary Settlement | 5 | 0.03% |
| Provisional Government | 5 | 0.03% |
| End_Date | 4 | 0.02% |
| The Sentence Claimed by the Prosecutor's Lawyer | 4 | 0.02% |
| Tear Up Reason | 4 | 0.02% |
| Damaged Crops & Livestock | 4 | 0.02% |
| Influenced People | 4 | 0.02% |
| Rescue Start Time | 4 | 0.02% |
| Last Time the Record was Broken | 4 | 0.02% |
| How Many Times Get Married | 3 | 0.02% |
| MVP | 3 | 0.02% |
| Organization Leader | 3 | 0.02% |
| Storm Movement Speed | 3 | 0.02% |
| Announce Platform | 3 | 0.02% |
| Economy Loss | 3 | 0.02% |
| Release Reason | 3 | 0.02% |
| Doctor | 2 | 0.01% |
| Acquisition amount | 2 | 0.01% |
| Economists who Predicted the Crisis | 2 | 0.01% |
| Invited Person | 2 | 0.01% |
| Wedding Dress Designer | 2 | 0.01% |
| Cost | 2 | 0.01% |
| Detention Start Time | 2 | 0.01% |
| Production Cuts | 2 | 0.01% |
| Shipwreck Reason | 2 | 0.01% |
| Storm Formation Time | 2 | 0.01% |
| Head of the Institution | 2 | 0.01% |
| Child Custody | 2 | 0.01% |
| Astronauts | 2 | 0.01% |
| Lawyer | 2 | 0.01% |
| Number of Damaged House | 2 | 0.01% |
| Withdraw Reason | 2 | 0.01% |
| Commanders of the Army | 2 | 0.01% |
| Sentencing Location | 2 | 0.01% |
| State Before Death | 1 | 0.01% |
| Reasons for the Formation of the Historical sites | 1 | 0.01% |
| Merger Terms | 1 | 0.01% |
| The Worst-Hit Area | 1 | 0.01% |
| State of the Hull | 1 | 0.01% |
| Storm Center Location | 1 | 0.01% |
| Storm Formation Location | 1 | 0.01% |
| Sequelae | 1 | 0.01% |
| Recovered Amount | 1 | 0.01% |
| Invalid Date | 1 | 0.01% |
| Spokeswoman/Spokesman | 1 | 0.01% |
| Time for Dignitaries to Resign | 1 | 0.01% |
| People Casting Key Votes | 1 | 0.01% |

#### test

| label | count | share |
| --- | --- | --- |
| Date | 2,080 | 10.90% |
| Location | 1,687 | 8.84% |
| Casualties and Losses | 1,209 | 6.34% |
| Cause | 692 | 3.63% |
| Confirmed/Infected Cases | 381 | 2.00% |
| Areas affected | 330 | 1.73% |
| Candidates and their Political Parties | 292 | 1.53% |
| Affected Areas | 281 | 1.47% |
| Suspect | 249 | 1.31% |
| Belligerents | 244 | 1.28% |
| Weapon and Equippment | 219 | 1.15% |
| Charged Crime | 197 | 1.03% |
| Attacker | 191 | 1.00% |
| Accident Investigator | 171 | 0.90% |
| Death Cases | 168 | 0.88% |
| Flight No. | 165 | 0.86% |
| Target | 163 | 0.85% |
| Judgement Result/Prison Term | 160 | 0.84% |
| Affected Area | 151 | 0.79% |
| Accusation | 151 | 0.79% |
| Aircraft Agency | 150 | 0.79% |
| Influence People | 138 | 0.72% |
| Disease | 137 | 0.72% |
| Protesters | 137 | 0.72% |
| Policy Proposals | 136 | 0.71% |
| Protest Reason | 135 | 0.71% |
| Scheduled Landing Place | 128 | 0.67% |
| Policy Content | 114 | 0.60% |
| Storm Hit Location | 112 | 0.59% |
| Game Name | 109 | 0.57% |
| Participants | 109 | 0.57% |
| Organization | 108 | 0.57% |
| Position | 108 | 0.57% |
| Precautionary Measure | 108 | 0.57% |
| Economic Loss | 106 | 0.56% |
| Accused People | 105 | 0.55% |
| The Final Votes and Percentages | 101 | 0.53% |
| Magnitude | 98 | 0.51% |
| Artifacts and Their Chronology | 90 | 0.47% |
| Start Time | 89 | 0.47% |
| Strike Reason | 88 | 0.46% |
| Husband | 86 | 0.45% |
| Types of the Spectacle | 86 | 0.45% |
| Competition Items | 85 | 0.45% |
| Wife | 84 | 0.44% |
| Judge | 83 | 0.44% |
| Survivors | 83 | 0.44% |
| Contracting Parties | 83 | 0.44% |
| Contest Participant | 79 | 0.41% |
| Number of Influenced People | 79 | 0.41% |
| Spectacle Start Time | 79 | 0.41% |
| Number of Damaged Houses | 77 | 0.40% |
| Crew | 76 | 0.40% |
| Weapon | 75 | 0.39% |
| Acquiree | 74 | 0.39% |
| End Time | 72 | 0.38% |
| Maximum Rainfall | 71 | 0.37% |
| Riot Reason | 70 | 0.37% |
| Election Name | 70 | 0.37% |
| Strikers | 69 | 0.36% |
| Record Breaker | 69 | 0.36% |
| Location/Hospital | 67 | 0.35% |
| Resignated Person | 67 | 0.35% |
| Grades | 66 | 0.35% |
| Record-breaking Project | 66 | 0.35% |
| Champions | 64 | 0.34% |
| Announcement Date | 62 | 0.32% |
| Aid Agency | 59 | 0.31% |
| Strikers Status | 59 | 0.31% |
| Solution | 59 | 0.31% |
| Court Time | 58 | 0.30% |
| Spectacle Location | 58 | 0.30% |
| The Charged Crime | 57 | 0.30% |
| Deceased | 56 | 0.29% |
| Method | 56 | 0.29% |
| Aid Supplies/Amount | 53 | 0.28% |
| Number of Vehicles Involved in the Crash | 52 | 0.27% |
| Appointee | 52 | 0.27% |
| Epidemic Data Issuing Agency | 51 | 0.27% |
| Time of the Case | 51 | 0.27% |
| Effective Date | 50 | 0.26% |
| Court | 49 | 0.26% |
| Winner | 49 | 0.26% |
| Country Visited | 49 | 0.26% |
| The Agency Who Broke the Agreement | 48 | 0.25% |
| Taking-off Place | 48 | 0.25% |
| Profession | 47 | 0.25% |
| Discover Location | 47 | 0.25% |
| Spacecraft | 47 | 0.25% |
| Pollution Source | 47 | 0.25% |
| Affected Industries | 46 | 0.24% |
| Victim | 45 | 0.24% |
| Missings | 45 | 0.24% |
| Score | 44 | 0.23% |
| Summit Theme | 44 | 0.23% |
| Cured Cases | 44 | 0.23% |
| Age | 43 | 0.23% |
| Resign Reason | 43 | 0.23% |
| Army | 43 | 0.23% |
| Epicenter | 43 | 0.23% |
| Host Country | 42 | 0.22% |
| Number of Evacuated People | 42 | 0.22% |
| Boycotted Institutions | 42 | 0.22% |
| Influenced Crops and Livelihood | 41 | 0.21% |
| Organization Name | 40 | 0.21% |
| Military Exercise | 40 | 0.21% |
| People | 39 | 0.20% |
| Host | 39 | 0.20% |
| Maximum Wind Speed | 39 | 0.20% |
| Passengers | 39 | 0.20% |
| Pests | 39 | 0.20% |
| Defense Lawyer | 38 | 0.20% |
| Police | 38 | 0.20% |
| Participating Countries | 38 | 0.20% |
| Released People | 38 | 0.20% |
| Deliberating Agency | 37 | 0.19% |
| Way for Spreading | 37 | 0.19% |
| Number of Vaccinated People | 37 | 0.19% |
| Launch Date | 37 | 0.19% |
| Discover Time | 36 | 0.19% |
| Attending Hospital | 36 | 0.19% |
| Death Reason | 35 | 0.18% |
| Speaker Status | 35 | 0.18% |
| Fined Agency | 35 | 0.18% |
| Anti-pollution People/Organizations | 35 | 0.18% |
| Forecasting Agency | 35 | 0.18% |
| Speaker | 34 | 0.18% |
| Award | 34 | 0.18% |
| Storm Hit Time | 34 | 0.18% |
| Agreement Content | 34 | 0.18% |
| Arrested | 34 | 0.18% |
| Research Agency | 34 | 0.18% |
| Start Date | 34 | 0.18% |
| Bank Name | 33 | 0.17% |
| Death Rate | 33 | 0.17% |
| Penalty Amount | 33 | 0.17% |
| Damaged Facility | 33 | 0.17% |
| Prison Term | 33 | 0.17% |
| Beneficiary | 32 | 0.17% |
| Employed Institution | 32 | 0.17% |
| Visitor | 32 | 0.17% |
| Previous Record Holder | 32 | 0.17% |
| Award Reason | 31 | 0.16% |
| Agreement Name | 31 | 0.16% |
| Countries Withdrawing from the Organization | 31 | 0.16% |
| Award Field | 30 | 0.16% |
| Investigative Agency | 30 | 0.16% |
| Fine Reason | 30 | 0.16% |
| Poisoning Type | 30 | 0.16% |
| Organization Industry | 29 | 0.15% |
| Symptom | 29 | 0.15% |
| Person under Investigation | 29 | 0.15% |
| Protest Slogan | 29 | 0.15% |
| Scale | 29 | 0.15% |
| Prosecutor | 29 | 0.15% |
| Archaeologist | 28 | 0.15% |
| Arrest Location | 28 | 0.15% |
| Ship No. | 28 | 0.15% |
| Storm Name | 28 | 0.15% |
| Employment Agency | 28 | 0.15% |
| Strike Agency | 28 | 0.15% |
| Sponsor | 26 | 0.14% |
| People/Organization who predicted the disaster | 25 | 0.13% |
| Weapon Used | 25 | 0.13% |
| Last Job of the Appointee | 25 | 0.13% |
| Marriage Duration | 25 | 0.13% |
| Strike Outcome | 25 | 0.13% |
| Conflict Duration | 25 | 0.13% |
| Volcano Name | 24 | 0.13% |
| Area Affected | 24 | 0.13% |
| Projects | 24 | 0.13% |
| Historical Sites | 23 | 0.12% |
| Carrier Rocket | 23 | 0.12% |
| Outbreak Date | 23 | 0.12% |
| Duration | 23 | 0.12% |
| Goal | 23 | 0.12% |
| Perpetrators | 22 | 0.12% |
| Investigating agency | 22 | 0.12% |
| The Final Seats Result | 22 | 0.12% |
| Train No. | 22 | 0.12% |
| Spokenmen | 22 | 0.12% |
| Funding | 20 | 0.10% |
| Arrest Time | 20 | 0.10% |
| Predecessor | 20 | 0.10% |
| Child | 20 | 0.10% |
| Number of People Hospitalized | 20 | 0.10% |
| Launch Site | 20 | 0.10% |
| Amount of Precipitation | 19 | 0.10% |
| Appointment Time | 19 | 0.10% |
| Inauguration Time | 19 | 0.10% |
| Annoncement Date | 19 | 0.10% |
| The Grades of the Previous Record Holder | 19 | 0.10% |
| Head of the Merged Organization | 18 | 0.09% |
| Areas Affected | 18 | 0.09% |
| Hull Location | 18 | 0.09% |
| Economic loss | 18 | 0.09% |
| Appointer | 18 | 0.09% |
| Regulatory Authority | 18 | 0.09% |
| Response Measures | 18 | 0.09% |
| Number of Destroyed Building | 18 | 0.09% |
| Evidence | 18 | 0.09% |
| Ways to Watch the Speech | 17 | 0.09% |
| Declarer | 17 | 0.09% |
| Tear Up Reason | 17 | 0.09% |
| Criminal Evidence | 17 | 0.09% |
| Train Agency | 17 | 0.09% |
| Related Rivers or Lakes | 16 | 0.08% |
| Storm Warning Level | 16 | 0.08% |
| Achievement | 16 | 0.08% |
| Number of Trapped People | 16 | 0.08% |
| Military Strength | 16 | 0.08% |
| Number of Aftershocks | 16 | 0.08% |
| Service Years | 15 | 0.08% |
| The Sentence Claimed by the Prosecutor's Lawyer | 14 | 0.07% |
| Release Time | 14 | 0.07% |
| Summit Name | 14 | 0.07% |
| Number of Rescued People | 14 | 0.07% |
| Water Level | 14 | 0.07% |
| Number of Victims | 14 | 0.07% |
| Head of Institution | 14 | 0.07% |
| News Release Agency | 13 | 0.07% |
| Wedding Venue | 13 | 0.07% |
| Age of the Resignated Person | 13 | 0.07% |
| Voting method | 13 | 0.07% |
| Election Goal | 13 | 0.07% |
| Acquisition amount | 12 | 0.06% |
| Organization Leader | 12 | 0.06% |
| Transportation | 12 | 0.06% |
| Suspected Cases | 12 | 0.06% |
| Cured Rate | 12 | 0.06% |
| Release Reason | 12 | 0.06% |
| Party Responsible for Pollution | 12 | 0.06% |
| Acquirer | 11 | 0.06% |
| Countries Joined the Organization | 11 | 0.06% |
| Organization Members | 11 | 0.06% |
| Rescue Organizer | 11 | 0.06% |
| Agreement Validity Period | 11 | 0.06% |
| Spacecraft Mission | 11 | 0.06% |
| Government Reaction | 11 | 0.06% |
| Commanders and their Position | 11 | 0.06% |
| Prison | 10 | 0.05% |
| Investigating Spokesperson | 10 | 0.05% |
| Outbreak Location | 10 | 0.05% |
| Launch Country | 10 | 0.05% |
| Cooperative Agency | 10 | 0.05% |
| Disaster-Stricken Farmland | 10 | 0.05% |
| Successor | 10 | 0.05% |
| Country | 10 | 0.05% |
| Electoral System | 10 | 0.05% |
| Rescue Tool or Method | 9 | 0.05% |
| Astronauts | 9 | 0.05% |
| Hull Discovery Time | 8 | 0.04% |
| The State of the Volcano (Dormant or Active) | 8 | 0.04% |
| Bankrupt Business | 7 | 0.04% |
| Doctor and Medical Team | 7 | 0.04% |
| Witness | 7 | 0.04% |
| Lasting Time | 7 | 0.04% |
| Prosecution Lawyer | 7 | 0.04% |
| Storm Direction | 7 | 0.04% |
| Child Custody | 7 | 0.04% |
| Property Division | 7 | 0.04% |
| Policy Name & Abbreviation | 7 | 0.04% |
| Launch Result | 7 | 0.04% |
| Last Outbreak Time | 7 | 0.04% |
| Magnitude(Tsunami heights) | 7 | 0.04% |
| Invited Person | 6 | 0.03% |
| Detention Start Time | 6 | 0.03% |
| Rescue Start Time | 6 | 0.03% |
| State of the Hull | 6 | 0.03% |
| Age of the Appointee | 6 | 0.03% |
| Mission Duration | 6 | 0.03% |
| Approver | 6 | 0.03% |
| Strike Industry | 6 | 0.03% |
| Best Way to Shoot | 6 | 0.03% |
| Unemployed Rate | 5 | 0.03% |
| How Many Times Get Married | 5 | 0.03% |
| The Sentence Claimed by the Defense Lawyer | 5 | 0.03% |
| Trapped Depth | 5 | 0.03% |
| Invalid Date | 5 | 0.03% |
| Symptoms | 5 | 0.03% |
| Susceptible Population | 5 | 0.03% |
| Special Medicine | 5 | 0.03% |
| Head of Investigation Team | 5 | 0.03% |
| Alternate Landing Place | 5 | 0.03% |
| Damaged Property | 5 | 0.03% |
| Turnout(who win who lost) | 5 | 0.03% |
| Jail Time | 5 | 0.03% |
| Spectacle Duration | 5 | 0.03% |
| Postpone Reason | 4 | 0.02% |
| Ship Agency | 4 | 0.02% |
| Stolen Amount | 4 | 0.02% |
| Announce Platform | 4 | 0.02% |
| Lawyer | 4 | 0.02% |
| Temporary Settlement | 4 | 0.02% |
| Term of Office | 4 | 0.02% |
| Sentencing Location | 4 | 0.02% |
| Live Broadcast Platform | 4 | 0.02% |
| Spectacle End Time | 4 | 0.02% |
| Perpetrator | 3 | 0.02% |
| Merger Terms | 3 | 0.02% |
| Postpone Time | 3 | 0.02% |
| Aid Reason | 3 | 0.02% |
| Shipwreck Reason | 3 | 0.02% |
| Treatment Method | 3 | 0.02% |
| Bill Drafting Agency | 3 | 0.02% |
| Refuge | 3 | 0.02% |
| Fire Warning Level | 3 | 0.02% |
| Withdraw Reason | 3 | 0.02% |
| Provisional Government | 3 | 0.02% |
| Economy Loss | 3 | 0.02% |
| State Before Death | 2 | 0.01% |
| Archaeologist Organization | 2 | 0.01% |
| Wedding Dress Designer | 2 | 0.01% |
| MVP | 2 | 0.01% |
| Influenced People | 2 | 0.01% |
| Production Cuts | 2 | 0.01% |
| Storm Center Location | 2 | 0.01% |
| Head of the Institution | 2 | 0.01% |
| Hostage | 2 | 0.01% |
| Vaccine Research and Development Organization | 2 | 0.01% |
| Spokeswoman/Spokesman | 2 | 0.01% |
| Number of Damaged House | 2 | 0.01% |
| Warning Device | 2 | 0.01% |
| Commanders of the Army | 2 | 0.01% |
| End Date | 2 | 0.01% |
| Reasons for the Formation of the Historical sites | 1 | 0.01% |
| Economists who Predicted the Crisis | 1 | 0.01% |
| Start_Date | 1 | 0.01% |
| Join Conditions | 1 | 0.01% |
| Damaged Crops & Livestock | 1 | 0.01% |
| The Worst-Hit Area | 1 | 0.01% |
| Compensation | 1 | 0.01% |
| Storm Movement Speed | 1 | 0.01% |
| Sequelae | 1 | 0.01% |
| Trapped Days | 1 | 0.01% |
| Tsunamis | 1 | 0.01% |
| Head of the Government | 1 | 0.01% |
| Candidate Scandal(Allegations of fraud, etc.) | 1 | 0.01% |

### Relations

None.

### Classifications

**train, docee_event.** Majority class 8.0% of 21,842 answers. The marginal is `true_label`, not the menu.

#### train docee_event

| label | count | share |
| --- | --- | --- |
| Armed Conflict | 1,758 | 8.05% |
| Air Crash | 1,388 | 6.35% |
| Protest | 966 | 4.42% |
| Election | 883 | 4.04% |
| CommitCrime - Accuse | 866 | 3.96% |
| Sports Competition | 840 | 3.85% |
| Fire | 739 | 3.38% |
| CommitCrime - Sentence | 714 | 3.27% |
| Government Policy Changes | 670 | 3.07% |
| Earthquakes | 604 | 2.77% |
| Riot | 575 | 2.63% |
| Resignation_Dismissal | 533 | 2.44% |
| Famous Person - Death | 453 | 2.07% |
| Disease Outbreaks | 440 | 2.01% |
| Floods | 412 | 1.89% |
| Appoint_Inauguration | 406 | 1.86% |
| Storm | 403 | 1.85% |
| Diplomatic Talks | 400 | 1.83% |
| Strike | 376 | 1.72% |
| Road Crash | 334 | 1.53% |
| Environment Pollution | 308 | 1.41% |
| Organization Closed | 305 | 1.40% |
| Gas Explosion | 296 | 1.36% |
| Bank Robbery | 290 | 1.33% |
| Break Historical Records | 287 | 1.31% |
| CommitCrime - Arrest | 282 | 1.29% |
| Sign Agreement | 272 | 1.25% |
| Awards Ceremony | 263 | 1.20% |
| Famous Person - Give a Speech | 261 | 1.19% |
| Shipwreck | 255 | 1.17% |
| Mine Collapses | 254 | 1.16% |
| CommitCrime - Release | 250 | 1.14% |
| Military Excercise | 245 | 1.12% |
| Mass Poisoning | 242 | 1.11% |
| Financial Crisis | 241 | 1.10% |
| New Achievements in Aerospace | 239 | 1.09% |
| Train Collisions | 237 | 1.09% |
| Withdraw from an Orgnization | 223 | 1.02% |
| Diplomatic Visit | 215 | 0.98% |
| Organization Merge | 212 | 0.97% |
| Insect Disaster | 212 | 0.97% |
| CommitCrime - Investigate | 208 | 0.95% |
| Organization Fine | 208 | 0.95% |
| New Wonders in Nature | 204 | 0.93% |
| Volcano Eruption | 197 | 0.90% |
| Famine | 197 | 0.90% |
| Famous Person - Sick | 195 | 0.89% |
| New Archeological Discoveries | 192 | 0.88% |
| Tear Up Agreement | 177 | 0.81% |
| Famous Person - Marriage | 155 | 0.71% |
| Financial Aid | 151 | 0.69% |
| Organization Established | 133 | 0.61% |
| Famous Person - Divorce | 121 | 0.55% |
| Tsunamis | 112 | 0.51% |
| Droughts | 103 | 0.47% |
| Mudslides | 98 | 0.45% |
| Famous Person - Recovered | 94 | 0.43% |
| Join in an Orgnization | 93 | 0.43% |
| Regime Change | 55 | 0.25% |

**val, docee_event.** Majority class 8.0% of 2,721 answers. The marginal is `true_label`, not the menu.

#### val docee_event

| label | count | share |
| --- | --- | --- |
| Armed Conflict | 219 | 8.05% |
| Air Crash | 173 | 6.36% |
| Protest | 121 | 4.45% |
| CommitCrime - Accuse | 113 | 4.15% |
| Election | 111 | 4.08% |
| Sports Competition | 105 | 3.86% |
| Fire | 93 | 3.42% |
| CommitCrime - Sentence | 89 | 3.27% |
| Government Policy Changes | 84 | 3.09% |
| Earthquakes | 76 | 2.79% |
| Riot | 72 | 2.65% |
| Resignation_Dismissal | 67 | 2.46% |
| Famous Person - Death | 57 | 2.09% |
| Disease Outbreaks | 55 | 2.02% |
| Floods | 52 | 1.91% |
| Storm | 51 | 1.87% |
| Appoint_Inauguration | 51 | 1.87% |
| Diplomatic Talks | 50 | 1.84% |
| Strike | 47 | 1.73% |
| Road Crash | 42 | 1.54% |
| Organization Closed | 38 | 1.40% |
| Environment Pollution | 38 | 1.40% |
| Gas Explosion | 37 | 1.36% |
| Bank Robbery | 36 | 1.32% |
| Break Historical Records | 36 | 1.32% |
| Sign Agreement | 34 | 1.25% |
| Famous Person - Give a Speech | 33 | 1.21% |
| Awards Ceremony | 33 | 1.21% |
| Shipwreck | 32 | 1.18% |
| Mine Collapses | 32 | 1.18% |
| Military Excercise | 31 | 1.14% |
| CommitCrime - Release | 31 | 1.14% |
| Financial Crisis | 30 | 1.10% |
| New Achievements in Aerospace | 30 | 1.10% |
| Mass Poisoning | 30 | 1.10% |
| Train Collisions | 30 | 1.10% |
| Withdraw from an Orgnization | 28 | 1.03% |
| Diplomatic Visit | 27 | 0.99% |
| Organization Merge | 26 | 0.96% |
| CommitCrime - Investigate | 26 | 0.96% |
| Organization Fine | 26 | 0.96% |
| Insect Disaster | 26 | 0.96% |
| Volcano Eruption | 25 | 0.92% |
| Famine | 25 | 0.92% |
| New Wonders in Nature | 25 | 0.92% |
| New Archeological Discoveries | 24 | 0.88% |
| Famous Person - Sick | 24 | 0.88% |
| Tear Up Agreement | 22 | 0.81% |
| Famous Person - Marriage | 19 | 0.70% |
| Financial Aid | 19 | 0.70% |
| CommitCrime - Arrest | 18 | 0.66% |
| Organization Established | 17 | 0.62% |
| Famous Person - Divorce | 15 | 0.55% |
| Tsunamis | 14 | 0.51% |
| Droughts | 13 | 0.48% |
| Join in an Orgnization | 12 | 0.44% |
| Famous Person - Recovered | 12 | 0.44% |
| Mudslides | 12 | 0.44% |
| Regime Change | 7 | 0.26% |

**test, docee_event.** Majority class 8.0% of 2,744 answers. The marginal is `true_label`, not the menu.

#### test docee_event

| label | count | share |
| --- | --- | --- |
| Armed Conflict | 220 | 8.02% |
| Air Crash | 174 | 6.34% |
| Protest | 121 | 4.41% |
| Election | 112 | 4.08% |
| Sports Competition | 105 | 3.83% |
| Fire | 93 | 3.39% |
| CommitCrime - Sentence | 90 | 3.28% |
| CommitCrime - Accuse | 86 | 3.13% |
| Government Policy Changes | 84 | 3.06% |
| Earthquakes | 76 | 2.77% |
| Riot | 72 | 2.62% |
| Resignation_Dismissal | 67 | 2.44% |
| Famous Person - Death | 57 | 2.08% |
| Disease Outbreaks | 55 | 2.00% |
| Floods | 52 | 1.90% |
| Storm | 51 | 1.86% |
| Appoint_Inauguration | 51 | 1.86% |
| Diplomatic Talks | 51 | 1.86% |
| Strike | 48 | 1.75% |
| CommitCrime - Arrest | 44 | 1.60% |
| Road Crash | 42 | 1.53% |
| Organization Closed | 39 | 1.42% |
| Environment Pollution | 39 | 1.42% |
| Gas Explosion | 37 | 1.35% |
| Bank Robbery | 37 | 1.35% |
| Break Historical Records | 36 | 1.31% |
| Sign Agreement | 35 | 1.28% |
| Awards Ceremony | 34 | 1.24% |
| Famous Person - Give a Speech | 33 | 1.20% |
| Shipwreck | 33 | 1.20% |
| Mine Collapses | 32 | 1.17% |
| CommitCrime - Release | 32 | 1.17% |
| Financial Crisis | 31 | 1.13% |
| Mass Poisoning | 31 | 1.13% |
| Military Excercise | 31 | 1.13% |
| New Achievements in Aerospace | 30 | 1.09% |
| Train Collisions | 30 | 1.09% |
| Withdraw from an Orgnization | 29 | 1.06% |
| Organization Merge | 27 | 0.98% |
| Diplomatic Visit | 27 | 0.98% |
| Insect Disaster | 27 | 0.98% |
| CommitCrime - Investigate | 26 | 0.95% |
| Organization Fine | 26 | 0.95% |
| New Wonders in Nature | 26 | 0.95% |
| Famous Person - Sick | 25 | 0.91% |
| Volcano Eruption | 25 | 0.91% |
| Famine | 25 | 0.91% |
| New Archeological Discoveries | 24 | 0.87% |
| Tear Up Agreement | 23 | 0.84% |
| Famous Person - Marriage | 20 | 0.73% |
| Financial Aid | 19 | 0.69% |
| Organization Established | 17 | 0.62% |
| Famous Person - Divorce | 16 | 0.58% |
| Tsunamis | 14 | 0.51% |
| Droughts | 13 | 0.47% |
| Mudslides | 13 | 0.47% |
| Join in an Orgnization | 12 | 0.44% |
| Famous Person - Recovered | 12 | 0.44% |
| Regime Change | 7 | 0.26% |

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
