# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loc_control | train | 17,749 | 17,728 | 21,320 | 1,676 | 2,651 | 31,041 | 4,122 | 4,114 |
| loc_control | val | 527 | 527 | 955 | 53 | 105 | 813 | 161 | 190 |
| loc_control | test | 527 | 527 | 975 | 64 | 112 | 821 | 206 | 159 |

## loc_control

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 17,749 | 17,728 | 199 | 533 | 19490 | 21,320 | 1,676 | 2,651 | 31,041 | 4,122 | 4,114 |
| val | 527 | 527 | 200 | 694 | 7176 | 955 | 53 | 105 | 813 | 161 | 190 |
| test | 527 | 527 | 199 | 836 | 5668 | 975 | 64 | 112 | 821 | 206 | 159 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 15,186 | 430 | 433 | 16,049 | 85.35% |
| zho | 2,405 | 94 | 89 | 2,588 | 13.76% |
| und | 105 | 2 | 5 | 112 | 0.60% |
| por | 8 | 0 | 0 | 8 | 0.04% |
| hin | 6 | 1 | 0 | 7 | 0.04% |
| rus | 6 | 0 | 0 | 6 | 0.03% |
| fra | 5 | 0 | 0 | 5 | 0.03% |
| kor | 5 | 0 | 0 | 5 | 0.03% |
| ita | 4 | 0 | 0 | 4 | 0.02% |
| ara | 4 | 0 | 0 | 4 | 0.02% |
| nld | 3 | 0 | 0 | 3 | 0.02% |
| tur | 3 | 0 | 0 | 3 | 0.02% |
| spa | 2 | 0 | 0 | 2 | 0.01% |
| ukr | 2 | 0 | 0 | 2 | 0.01% |
| deu | 2 | 0 | 0 | 2 | 0.01% |
| fin | 1 | 0 | 0 | 1 | 0.01% |
| pol | 1 | 0 | 0 | 1 | 0.01% |
| nor | 1 | 0 | 0 | 1 | 0.01% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 527 |
| test_unique_inputs | 527 |
| train_records | 17,749 |
| train_unique_inputs | 17,728 |
| val_records | 527 |
| val_unique_inputs | 527 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.2779 WARN | 21,320 / 955 | StartDate 6%→7%, EndDate 2%→3%, GeneOrGeneProduct 0%→1%, Reduction Start Date 1%→2%, Casualties and Losses 2%→1% |
| entities (per mention) | train / test | 0.2564 WARN | 21,320 / 975 | Vulnerability 0%→2%, CVE 0%→1%, Reducing Shareholder 2%→3%, Product 0%→1%, Date 3%→2% |
| entities (per mention) | val / test | 0.3142 STRONG | 955 / 975 | StartDate 7%→5%, Reduction Start Date 2%→1%, EndDate 3%→1%, Protesters 0%→1%, CVE 0%→1% |
| relations (per instance) | train / val | 0.8719 STRONG | 1,676 / 53 | Association 0%→9%, bind 7%→0%, Lives_In 4%→0%, PPI 5%→2%, Bind 0%→2% |
| relations (per instance) | train / test | 0.8329 STRONG | 1,676 / 64 | PPI 5%→0%, Lives_In 4%→0%, bind 7%→11%, maintained by 0%→3%, terminus location 0%→3% |
| relations (per instance) | val / test | 0.9375 STRONG | 53 / 64 | bind 0%→11%, Association 9%→0%, track gauge 0%→3%, terminus location 0%→3%, maintained by 0%→3% |
| event types (per trigger mention) | train / val | 0.4713 STRONG | 4,744 / 182 | Cyber.DiscoverVulnerability 1%→12%, Event 3%→7%, Process_start 2%→6%, Competition 3%→0%, Support 5%→7% |
| event types (per trigger mention) | train / test | 0.4657 STRONG | 4,744 / 239 | Cyber.DiscoverVulnerability 1%→6%, Experiment 6%→2%, Support 5%→0%, Event 3%→7%, Know 0%→3% |
| event types (per trigger mention) | val / test | 0.5206 STRONG | 182 / 239 | Support 7%→0%, Cyber.DiscoverVulnerability 12%→6%, Experiment 7%→2%, Process_start 6%→2%, Know 0%→3% |
| event roles (per argument) | train / val | 0.4070 STRONG | 4,354 / 204 | Vulnerability 2%→10%, Discoverer 1%→7%, Location 6%→1%, Vulnerable_System 1%→6%, Date 13%→8% |
| event roles (per argument) | train / test | 0.5238 STRONG | 4,354 / 168 | Subject 19%→9%, Vulnerability 2%→11%, CVE 0%→8%, Vulnerable_System 1%→8%, Equipment 10%→4% |
| event roles (per argument) | val / test | 0.4212 STRONG | 204 / 168 | CVE 0%→8%, Subject 16%→9%, Materials 6%→1%, Object 7%→1%, Time 4%→1% |
| classifications (true_label) | train / val | 0.2972 WARN | 2,651 / 105 | chfinann_event: EquityPledge 18%→22%, chfinann_event: EquityUnderweight 7%→10%, docee_event: CommitCrime - Sentence 1%→4%, docfee_event: EquityUnderweight 8%→10%, docee_event: Break Historical Records 0%→3% |
| classifications (true_label) | train / test | 0.2877 WARN | 2,651 / 112 | chfinann_event: EquityPledge 18%→23%, chfinann_event: EquityUnderweight 7%→4%, chfinann_event: EquityOverweight 8%→5%, docee_event: Insect Disaster 0%→3%, docee_event: Air Crash 2%→4% |
| classifications (true_label) | val / test | 0.3262 STRONG | 105 / 112 | chfinann_event: EquityUnderweight 10%→4%, docee_event: CommitCrime - Sentence 4%→0%, chfinann_event: EquityRepurchase 0%→4%, docee_event: Insect Disaster 0%→3%, docfee_event: EquityUnderweight 10%→8% |

Unseen in train, present in val, entities (per mention): Population ×3, model ×3, embodiment ×3, SequenceVariant ×3, Enzyme ×2, Body Part ×2, Disease Form ×2, Medical Test ×2, modality ×2, metric ×2, Substance ×1, Biological Entity ×1, Service ×1, environment ×1, task_category ×1, capability ×1, group ×1.
Thin support in val, entities (per mention): 121 label(s) under 10 gold. Carrier Rocket ×1, Launch Site ×1, Spacecraft ×1, Launch Result ×1, Launch Date ×1, Substance ×1, Biological Entity ×1, Service ×1, Freeze Start Date ×1, Spokenmen ×1, Affected Areas ×1, Number of Influenced People ×1, Belligerents ×1, Weapon ×1, dataset ×1, environment ×1, task_category ×1, capability ×1, group ×1, Precautionary Measure ×1, Types of the Spectacle ×1, Spectacle Location ×1, Spectacle End Time ×1, Summit Name ×1, Summit Theme ×1, pledge_start_date ×1, Head of Investigation Team ×1, Person under Investigation ×1, Approver ×1, Resignated Person ×1, Resign Reason ×1, Position ×1, Age of the Resignated Person ×1, Survivors ×1, Prison ×1, Storm Name ×1, Storm Warning Level ×1, Storm Hit Time ×1, Influence People ×1, People/Organization who predicted the disaster ×1, Areas Affected ×1, Train Agency ×1, LegalInstitution ×1, UnfrozeDate ×1, Discover Time ×1, Discover Location ×1, End Date ×1, Economy Loss ×1, Increase Start Date ×2, Increasing Shareholder ×2, Increase Amount ×2, Court Time ×2, Judge ×2, Enzyme ×2, Body Part ×2, Disease Form ×2, Medical Test ×2, frozen_shareholder ×2, Effective Date ×2, modality ×2, metric ×2, Loss Amount ×2, Deliberating Agency ×2, Areas affected ×2, Spectacle Start Time ×2, Participants ×2, Aircraft Agency ×2, Attending Hospital ×2, Train No. ×2, ChemicalEntity ×2, Weapon and Equippment ×2, ruling_date ×2, Gene ×2, Number of Vehicles Involved in the Crash ×2, FrozeShares ×2, Artifacts and Their Chronology ×2, Strikers Status ×2, Strike Agency ×2, Strikers ×2, Suspect ×3, Disease ×3, Population ×3, Frozen Amount ×3, model ×3, embodiment ×3, Investigative Agency ×3, Accident Investigator ×3, OrganismTaxon ×3, DiseaseOrPhenotypicFeature ×3, SequenceVariant ×3, Target ×3, The Grades of the Previous Record Holder ×3, Previous Record Holder ×4, Court ×4, Judgement Result/Prison Term ×4, Accusation ×4, Symptom ×4, Accepting Court ×4, pledgor ×4, pledge_amount ×4, Capabilities ×4, Scheduled Landing Place ×4, Record-breaking Project ×5, Policy Content ×5, Last Time the Record was Broken ×5, Recipient ×5, Patch ×5, Storm Hit Location ×5, Strike Outcome ×5, Start Date ×5, Record Breaker ×6, Attacker ×6, Strike Reason ×6, Company Name ×7, Device ×7, Time ×7, Flight No. ×7, Announcement Date ×8, LaterHoldingShares ×8, Grades ×9, Person ×9.
Unseen in train, present in val, relations (per instance): Bind ×1, location of first performance ×1, decay mode ×1.
Thin support in val, relations (per instance): 49 label(s) under 10 gold. taxon rank ×1, replaces ×1, publisher ×1, commemorates ×1, title of chess person ×1, derivative work ×1, military branch ×1, drug or therapy used for treatment ×1, discography ×1, office held by head of state ×1, instrument ×1, based on ×1, taxonomic type ×1, platform ×1, occupation ×1, location of landing ×1, astronaut mission ×1, discoverer or inventor ×1, product or material produced ×1, affiliation ×1, partner in business or sport ×1, endemic to ×1, head of government ×1, movement ×1, crew member(s) ×1, location of formation ×1, notable work ×1, director / manager ×1, member of political party ×1, voice type ×1, child ×1, fabrication method ×1, place of detention ×1, father ×1, distribution format ×1, diocese ×1, Positive_Correlation ×1, Bind ×1, PPI ×1, head of state ×1, basin country ×1, instance of ×1, official language ×1, secretary general ×1, honorific suffix ×1, location of first performance ×1, decay mode ×1, presenter ×1, Association ×5.
Unseen in train, present in val, event types (per trigger mention): rate_decrease ×1, Rewards_and_punishments ×1.
Thin support in val, event types (per trigger mention): 68 label(s) under 10 gold. price_increase ×1, rate_decrease ×1, Arrest ×1, Testing ×1, Choosing ×1, birthday_celebration ×1, dissolution ×1, price_decrease ×1, Accident ×1, product_publish ×1, Escaping ×1, Catastrophe ×1, Rewards_and_punishments ×1, Reveal_secret ×1, GiveUp ×1, Arriving ×1, Revenge ×1, Connect ×1, Cause_change_of_strength ×1, Request ×1, Change_of_leadership ×1, Getting ×1, Cause_to_amalgamate ×1, Creating ×1, Assistance ×1, Quarreling ×1, proposal ×1, Deploy ×1, resignation ×1, dismissal ×1, judicial_action_prosecution ×1, product_delist ×1, judicial_action_arrest ×1, Becoming ×1, Receiving ×1, Influence ×1, Motion_directional ×1, Dispersal ×1, Causation ×1, Check ×1, Aiming ×1, Placing ×1, Earnings_and_losses ×1, Traveling ×1, joined ×2, Coming_to_be ×2, Name_conferral ×2, Preventing_or_letting ×2, Sending ×2, Death ×2, Supply ×2, Self_motion ×2, Building ×2, Military_operation ×2, Statement ×2, Presence ×2, Bodily_harm ×3, Hostile_encounter ×3, Destroying ×3, Conquering ×3, Supporting ×3, Control ×3, Cause_to_be_included ×3, Process_end ×4, Attack ×5, Killing ×5, Cyber.PatchVulnerability ×5, match_result ×6.
Unseen in train, present in val, event roles (per argument): rate_cutting_institution ×1, delisted_party ×1.
Thin support in val, event roles (per argument): 33 label(s) under 10 gold. Price Increased Item ×1, rate_cutting_institution ×1, Birthday Person ×1, dissolution_party ×1, Result ×1, Released Product ×1, Proposer ×1, Proposal Target ×1, Militaryforce ×1, Departing Employee ×1, dismissed_employee ×1, dismissal_party ×1, Plaintiff ×1, defendant ×1, delist_by ×1, delisted_party ×1, Delisted Product ×1, arrested_person ×1, Arrester ×1, attacker ×1, event_name ×2, Joined Organization ×2, Celebrant ×2, price_reduced_item ×2, Releaser ×2, Vulnerable_System_Owner ×2, Location ×3, Franchisee ×3, Capabilities ×4, Winner ×5, Patch ×5, loser ×6, Time ×9.
Thin support in val, classifications (true_label): 35 label(s) under 10 gold. chfinann_event: EquityFreeze ×1, docee_event: New Achievements in Aerospace ×1, docee_event: Organization Established ×1, docee_event: Mine Collapses ×1, docee_event: Famine ×1, docee_event: Riot ×1, docee_event: Disease Outbreaks ×1, docee_event: New Wonders in Nature ×1, docee_event: Diplomatic Talks ×1, docee_event: Resignation_Dismissal ×1, docee_event: Storm ×1, docee_event: Droughts ×1, docee_event: Train Collisions ×1, docee_event: Road Crash ×1, docee_event: New Archeological Discoveries ×1, docee_event: Strike ×1, docfee_event: MajorSafetyAccident ×1, biotech_event: new initiatives or programs ×1, biotech_event: company description ×1, biotech_event: support & philanthropy ×1, biotech_event: other ×1, biotech_event: product launching & presentation ×1, docee_event: Government Policy Changes ×2, docee_event: CommitCrime - Investigate ×2, docee_event: Air Crash ×2, docee_event: Armed Conflict ×2, docfee_event: EquityOverweight ×2, docfee_event: EquityFreeze ×2, docfee_event: MajorAssetLoss ×2, docee_event: Break Historical Records ×3, biotech_event: executive statement ×3, docee_event: CommitCrime - Sentence ×4, docfee_event: BankruptcyLiquidation ×4, docfee_event: EquityPledge ×4, chfinann_event: EquityOverweight ×6.
Unseen in train, present in test, entities (per mention): Product ×10, RNA ×4, Pest ×2, Equipment ×2, Metric ×2, Damage ×2, Commanders and their Position ×1, Scientific Name ×1, Control Method ×1, Strategy ×1, Natural Enemy ×1, Lure ×1, organelle ×1.
Thin support in test, entities (per mention): 118 label(s) under 10 gold. Wife ×1, Accident Investigator ×1, Scale ×1, Goal ×1, Commanders and their Position ×1, Start_Date ×1, Affected Area ×1, End_Date ×1, Strikers Status ×1, Start Date ×1, Strike Industry ×1, Casualty Count ×1, FrozeShares ×1, LegalInstitution ×1, Record-breaking Project ×1, Last Time the Record was Broken ×1, Bank Name ×1, Weapon Used ×1, People ×1, Survivors ×1, Passengers ×1, Time ×1, File ×1, Scientific Name ×1, Control Method ×1, Strategy ×1, Natural Enemy ×1, Lure ×1, AveragePrice ×1, Employed Institution ×1, Winner ×1, Award Field ×1, Award ×1, Crew ×1, Launch Country ×1, Research Agency ×1, Spacecraft Mission ×1, Launch Result ×1, Types of the Spectacle ×1, Spectacle Location ×1, Spectacle Start Time ×1, protein-domain ×1, organelle ×1, senior_personnel ×1, Magnitude ×1, Number of Destroyed Building ×1, LowestTradingPrice ×1, Increase Start Date ×1, Husband ×2, Economic Loss ×2, Attending Hospital ×2, Attacker ×2, Accused People ×2, Influenced Crops and Livelihood ×2, Aid Agency ×2, Response Measures ×2, protein-family ×2, Number of Vehicles Involved in the Crash ×2, Previous Record Holder ×2, Record Breaker ×2, Grades ×2, The Grades of the Previous Record Holder ×2, RepurchaseAmount ×2, Pest ×2, Equipment ×2, Metric ×2, Damage ×2, Data ×2, Launch Date ×2, Cooperative Agency ×2, Carrier Rocket ×2, Increasing Shareholder ×2, Increase Amount ×2, Taking-off Place ×3, Scheduled Landing Place ×3, Weapon ×3, Doctor and Medical Team ×3, Symptom ×3, Unemployed Rate ×3, Prosecutor ×3, Charged Crime ×3, Candidates and their Political Parties ×3, Election Name ×3, ruling_date ×3, Version ×3, Patch ×3, Capabilities ×3, Spacecraft ×3, protein ×3, Aircraft Agency ×4, pledge_start_date ×4, pledge_end_date ×4, HighestTradingPrice ×4, Weapon and Equippment ×4, Target ×4, Location/Hospital ×4, Loss Amount ×4, RNA ×4, pledgor ×5, ClosingDate ×5, Accepting Court ×5, Pests ×5, The Final Votes and Percentages ×5, Flight No. ×5, Device ×5, Recipient ×6, Belligerents ×6, Protest Reason ×6, Organization ×6, RepurchasedShares ×7, Reduction Start Date ×7, Arrested ×7, Investigating agency ×7, Cause ×8, pledge_amount ×8, Announcement Date ×8, LaterHoldingShares ×8, Person ×9.
Unseen in train, present in test, relations (per instance): amended by ×1, wears ×1, via ×1, CPU ×1, has active ingredient ×1, different from ×1, narrator ×1, editor ×1.
Thin support in test, relations (per instance): 55 label(s) under 10 gold. ammunition ×1, amended by ×1, wears ×1, studied by ×1, board member ×1, newspaper format ×1, cause of destruction ×1, member of sports team ×1, surface played on ×1, manner of death ×1, crosses ×1, commissioned by ×1, via ×1, CPU ×1, lake outflow ×1, victory ×1, owned by ×1, enemy of ×1, has active ingredient ×1, authority ×1, has immediate cause ×1, biological process ×1, different from ×1, astronaut mission ×1, item operated ×1, legislative body ×1, product certification ×1, father ×1, contains the administrative territorial entity ×1, residence ×1, relative ×1, located in the statistical territorial entity ×1, located in or next to body of water ×1, narrator ×1, genre ×1, legislated by ×1, editor ×1, significant event ×1, represented by ×1, league ×1, child ×1, chairperson ×1, director ×1, brand ×1, office held by head of state ×1, found in taxon ×1, located in/on physical feature ×1, present in work ×1, killed by ×1, performer ×1, narrative location ×1, track gauge ×2, maintained by ×2, terminus location ×2, bind ×7.
Unseen in train, present in test, event types (per trigger mention): Lighting ×1, Theft ×1, Cure ×1.
Thin support in test, event types (per trigger mention): 95 label(s) under 10 gold. opening ×1, thank ×1, product_publish ×1, judicial_action_arrest ×1, Achieve ×1, Employment ×1, Come_together ×1, Control ×1, Being_in_operation ×1, Limiting ×1, Warning ×1, Lighting ×1, Openness ×1, Writing ×1, Exhibit ×1, judicial_action_filing ×1, Explosion ×1, Support ×1, judicial_action_hearing ×1, Action ×1, Cause_to_amalgamate ×1, Check ×1, Publishing ×1, Legality ×1, Quarreling ×1, Theft ×1, Choosing ×1, Expressing_publicly ×1, Cure ×1, Committing_crime ×1, Change ×1, Having_or_lacking_access ×1, Manufacturing ×1, Cyber.Phishing ×1, price_limit_up ×1, closing ×1, Influence ×1, Scouring ×1, judicial_action_imprisonment ×1, Request ×1, Arriving ×1, Cause_to_be_included ×1, Defending ×1, Receiving ×1, Conflict ×1, Arranging ×2, Hostile_encounter ×2, Removing ×2, Judgment_communication ×2, death ×2, Accident ×2, Earnings_and_losses ×2, Reveal_secret ×2, Becoming ×2, Participation ×2, match_result ×2, Deploy ×2, Escaping ×2, Preventing_or_letting ×2, Violence ×2, Perception_active ×2, Supply ×2, Giving ×2, Agree_or_refuse_to_act ×2, Statement ×2, Bodily_harm ×2, product_recall ×2, Cyber.Databreach ×2, Coming_to_be ×2, Placing ×2, Getting ×2, Scrutiny ×3, Education_teaching ×3, financing ×3, Motion ×3, championship_win ×3, Manoeuvre ×3, Cause_change_of_position_on_a_scale ×3, Award ×3, Research ×4, Supporting ×4, Creating ×4, Experiment ×4, Process_end ×4, Process_start ×5, Killing ×5, Death ×5, Name_conferral ×5, Social_event ×5, Catastrophe ×6, Attack ×6, Self_motion ×6, Cyber.PatchVulnerability ×7, Know ×8, Causation ×9.
Thin support in test, event roles (per argument): 46 label(s) under 10 gold. Thanking Party ×1, thanked_person ×1, Publisher ×1, Released Product ×1, Arrester ×1, arrested_person ×1, lead_investor ×1, Result ×1, loser ×1, event_name ×1, Case Subject ×1, Case Authority ×1, Area ×1, Materials ×1, Court Case ×1, Time ×1, Vulnerable_System_Owner ×1, Supported_Platform ×1, Trusted-Entity ×1, Trading Halt Stock ×1, attacker ×1, attack_target ×1, Sentence Length ×1, Imprisoned Person ×1, Activity Name ×2, Deceased ×2, Funding Amount ×2, Championship Event ×2, Champion ×2, Winner ×2, Militaryforce ×2, Object ×2, Releaser ×2, Recall Party ×2, Recall Content ×2, Compromised-Data ×2, Attacker ×2, Location ×3, Financing Party ×3, Funding Round ×3, Content ×3, Vulnerable_System_Version ×3, Capabilities ×3, Patch ×3, Victim ×5, Equipment ×6.
Unseen in train, present in test, classifications (true_label): biotech_event: executive appointment ×1, biotech_event: expanding industry ×1.
Thin support in test, classifications (true_label): 40 label(s) under 10 gold. docee_event: Famous Person - Divorce ×1, docee_event: Riot ×1, docee_event: Train Collisions ×1, docee_event: Military Excercise ×1, docee_event: Financial Crisis ×1, docee_event: Strike ×1, docee_event: Road Crash ×1, docee_event: Break Historical Records ×1, docee_event: Bank Robbery ×1, docee_event: Awards Ceremony ×1, docee_event: New Achievements in Aerospace ×1, docee_event: New Wonders in Nature ×1, docee_event: Earthquakes ×1, docfee_event: MajorSafetyAccident ×1, docfee_event: ExecutiveDeath ×1, chfinann_event: EquityFreeze ×1, biotech_event: funding round ×1, biotech_event: executive appointment ×1, biotech_event: company description ×1, biotech_event: new initiatives or programs ×1, biotech_event: expanding industry ×1, biotech_event: regulatory approval ×1, biotech_event: product launching & presentation ×1, docee_event: Famous Person - Sick ×2, docee_event: CommitCrime - Accuse ×2, docee_event: Protest ×2, docee_event: Election ×2, docfee_event: EquityOverweight ×2, biotech_event: m&a ×2, biotech_event: executive statement ×2, docee_event: Armed Conflict ×3, docee_event: Insect Disaster ×3, docee_event: Air Crash ×4, docfee_event: MajorAssetLoss ×4, chfinann_event: EquityRepurchase ×4, docfee_event: BankruptcyLiquidation ×5, chfinann_event: EquityUnderweight ×5, docfee_event: EquityPledge ×6, chfinann_event: EquityOverweight ×6, docfee_event: EquityUnderweight ×9.

### Events

**train.** 4,664 instances in 1,032 records, 0 triggerless. Trigger gold 4,122. Argument gold 4,114. Arguments per instance, mean 0.93. Trigger length in whitespace tokens, mean 1.37, single-token share 67.7%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Experiment | 303 | 6.39% |
| Support | 216 | 4.55% |
| Hostile_encounter | 138 | 2.91% |
| Event | 128 | 2.70% |
| Competition | 124 | 2.61% |
| Catastrophe | 121 | 2.55% |
| Deploy | 119 | 2.51% |
| Process_start | 117 | 2.47% |
| Attack | 108 | 2.28% |
| Causation | 95 | 2.00% |
| Exhibit | 73 | 1.54% |
| Motion | 71 | 1.50% |
| Cyber.DiscoverVulnerability | 71 | 1.50% |
| match_result | 70 | 1.48% |
| product_publish | 59 | 1.24% |
| Cyber.PatchVulnerability | 59 | 1.24% |
| Killing | 57 | 1.20% |
| Cyber.Databreach | 57 | 1.20% |
| Arriving | 54 | 1.14% |
| Death | 52 | 1.10% |
| Process_end | 50 | 1.05% |
| Control | 48 | 1.01% |
| Conquering | 48 | 1.01% |
| Social_event | 45 | 0.95% |
| Coming_to_be | 42 | 0.89% |
| Getting | 42 | 0.89% |
| Cyber.Ransom | 40 | 0.84% |
| Self_motion | 40 | 0.84% |
| Military_operation | 40 | 0.84% |
| Placing | 39 | 0.82% |
| Arranging | 39 | 0.82% |
| Statement | 38 | 0.80% |
| Earnings_and_losses | 38 | 0.80% |
| Cause_change_of_strength | 37 | 0.78% |
| Bodily_harm | 36 | 0.76% |
| Removing | 35 | 0.74% |
| Creating | 35 | 0.74% |
| Cause_change_of_position_on_a_scale | 34 | 0.72% |
| Becoming | 34 | 0.72% |
| Hold | 34 | 0.72% |
| Accident | 33 | 0.70% |
| Destroying | 33 | 0.70% |
| Presence | 32 | 0.67% |
| Influence | 31 | 0.65% |
| Motion_directional | 31 | 0.65% |
| Name_conferral | 31 | 0.65% |
| Participation | 31 | 0.65% |
| Violence | 30 | 0.63% |
| Preventing_or_letting | 30 | 0.63% |
| Cyber.Phishing | 30 | 0.63% |
| Escaping | 28 | 0.59% |
| Defending | 28 | 0.59% |
| Supporting | 27 | 0.57% |
| Damaging | 26 | 0.55% |
| Expressing_publicly | 26 | 0.55% |
| Change | 26 | 0.55% |
| death | 26 | 0.55% |
| Terrorism | 26 | 0.55% |
| Arrest | 26 | 0.55% |
| Cause_to_be_included | 25 | 0.53% |
| Agree_or_refuse_to_act | 23 | 0.48% |
| Giving | 23 | 0.48% |
| Supply | 23 | 0.48% |
| judicial_action_arrest | 23 | 0.48% |
| Know | 22 | 0.46% |
| Judgment_communication | 21 | 0.44% |
| advancement | 21 | 0.44% |
| resignation | 21 | 0.44% |
| Request | 21 | 0.44% |
| Cause_to_make_progress | 19 | 0.40% |
| Action | 19 | 0.40% |
| Change_event_time | 17 | 0.36% |
| Manoeuvre | 17 | 0.36% |
| Aiming | 17 | 0.36% |
| Recovering | 16 | 0.34% |
| Besieging | 16 | 0.34% |
| Assistance | 16 | 0.34% |
| Using | 16 | 0.34% |
| Reporting | 15 | 0.32% |
| divorce | 15 | 0.32% |
| Injure | 15 | 0.32% |
| Change_of_leadership | 15 | 0.32% |
| Recording | 15 | 0.32% |
| Committing_crime | 15 | 0.32% |
| Departing | 14 | 0.30% |
| Perception_active | 14 | 0.30% |
| Choosing | 14 | 0.30% |
| Dispersal | 13 | 0.27% |
| marriage | 13 | 0.27% |
| Receiving | 13 | 0.27% |
| product_recall | 13 | 0.27% |
| Legal_rulings | 13 | 0.27% |
| Check | 13 | 0.27% |
| Come_together | 13 | 0.27% |
| Deciding | 13 | 0.27% |
| championship_win | 13 | 0.27% |
| Traveling | 13 | 0.27% |
| joined | 12 | 0.25% |
| Manufacturing | 11 | 0.23% |
| Communication | 11 | 0.23% |
| Sending | 11 | 0.23% |
| Building | 11 | 0.23% |
| Scrutiny | 11 | 0.23% |
| GiveUp | 11 | 0.23% |
| Bringing | 11 | 0.23% |
| Hindering | 11 | 0.23% |
| Expansion | 11 | 0.23% |
| opening | 11 | 0.23% |
| price_limit_up | 10 | 0.21% |
| Surrendering | 10 | 0.21% |
| Response | 10 | 0.21% |
| Use_firearm | 10 | 0.21% |
| Becoming_a_member | 10 | 0.21% |
| Quarreling | 10 | 0.21% |
| departure | 10 | 0.21% |
| judicial_action_fine | 10 | 0.21% |
| judicial_action_summon | 10 | 0.21% |
| Openness | 10 | 0.21% |
| Achieve | 10 | 0.21% |
| Collaboration | 9 | 0.19% |
| sale_acquisition | 9 | 0.19% |
| Telling | 9 | 0.19% |
| Commitment | 9 | 0.19% |
| Commerce_pay | 8 | 0.17% |
| Coming_to_believe | 8 | 0.17% |
| Testing | 8 | 0.17% |
| judicial_action_imprisonment | 8 | 0.17% |
| Cause_to_amalgamate | 8 | 0.17% |
| breakup | 8 | 0.17% |
| product_release | 8 | 0.17% |
| Earthquake | 8 | 0.17% |
| Sign_agreement | 8 | 0.17% |
| Traffic Accident | 7 | 0.15% |
| competition_ban | 7 | 0.15% |
| Reforming_a_system | 7 | 0.15% |
| Criminal_investigation | 7 | 0.15% |
| suspension | 7 | 0.15% |
| Fire | 7 | 0.15% |
| Filling | 7 | 0.15% |
| Bearing_arms | 7 | 0.15% |
| Conflict | 6 | 0.13% |
| judicial_action_filing | 6 | 0.13% |
| competition_withdrawal | 6 | 0.13% |
| Limiting | 6 | 0.13% |
| judicial_action_prosecution | 6 | 0.13% |
| Collapse | 5 | 0.11% |
| Preserving | 5 | 0.11% |
| Education_teaching | 5 | 0.11% |
| product_award | 5 | 0.11% |
| apologize_to | 5 | 0.11% |
| Convincing | 5 | 0.11% |
| meet_with | 5 | 0.11% |
| birth | 5 | 0.11% |
| Publishing | 5 | 0.11% |
| Confronting_problem | 5 | 0.11% |
| Warning | 5 | 0.11% |
| proposal | 5 | 0.11% |
| dissolution | 5 | 0.11% |
| Connect | 5 | 0.11% |
| Emptying | 5 | 0.11% |
| birthday_celebration | 5 | 0.11% |
| Protest | 5 | 0.11% |
| Being_in_operation | 5 | 0.11% |
| Temporary_stay | 5 | 0.11% |
| Labeling | 5 | 0.11% |
| strike | 4 | 0.08% |
| Body_movement | 4 | 0.08% |
| lost_contact | 4 | 0.08% |
| Commerce_sell | 4 | 0.08% |
| Justifying | 4 | 0.08% |
| judicial_action_report | 4 | 0.08% |
| thank | 4 | 0.08% |
| visit_on_set | 4 | 0.08% |
| financing | 4 | 0.08% |
| layoff | 4 | 0.08% |
| price_decrease | 4 | 0.08% |
| Releasing | 4 | 0.08% |
| judicial_action_hearing | 3 | 0.06% |
| Cost | 3 | 0.06% |
| Surrounding | 3 | 0.06% |
| Resolve_problem | 3 | 0.06% |
| Imposing_obligation | 3 | 0.06% |
| Forming_relationships | 3 | 0.06% |
| Having_or_lacking_access | 3 | 0.06% |
| Create_artwork | 3 | 0.06% |
| Rite | 3 | 0.06% |
| price_limit_down | 2 | 0.04% |
| dismissal | 2 | 0.04% |
| pregnancy | 2 | 0.04% |
| closing | 2 | 0.04% |
| march | 2 | 0.04% |
| praise | 2 | 0.04% |
| Adducing | 2 | 0.04% |
| Patrolling | 2 | 0.04% |
| contract_termination | 2 | 0.04% |
| Robbery | 2 | 0.04% |
| Employment | 2 | 0.04% |
| product_delist | 2 | 0.04% |
| Reveal_secret | 2 | 0.04% |
| Submitting_documents | 2 | 0.04% |
| GetReady | 2 | 0.04% |
| Carry_goods | 2 | 0.04% |
| Award | 2 | 0.04% |
| engagement | 2 | 0.04% |
| Research | 2 | 0.04% |
| Change_sentiment | 2 | 0.04% |
| retirement | 2 | 0.04% |
| Expend_resource | 2 | 0.04% |
| price_increase | 2 | 0.04% |
| Writing | 2 | 0.04% |
| Plane Crash | 2 | 0.04% |
| Rescuing | 2 | 0.04% |
| wedding | 2 | 0.04% |
| Vocalizations | 2 | 0.04% |
| rate_increase | 1 | 0.02% |
| Explosion | 1 | 0.02% |
| Revenge | 1 | 0.02% |
| Exchange | 1 | 0.02% |
| Scouring | 1 | 0.02% |
| Prison | 1 | 0.02% |
| Flood | 1 | 0.02% |
| Commerce_buy | 1 | 0.02% |
| Wearing | 1 | 0.02% |
| Legality | 1 | 0.02% |
| Incident | 1 | 0.02% |
| Institutionalization | 1 | 0.02% |

**val.** 174 instances in 39 records, 0 triggerless. Trigger gold 161. Argument gold 190. Arguments per instance, mean 1.17. Trigger length in whitespace tokens, mean 1.36, single-token share 66.5%.

#### val event types

| label | count | share |
| --- | --- | --- |
| Cyber.DiscoverVulnerability | 21 | 11.54% |
| Experiment | 12 | 6.59% |
| Support | 12 | 6.59% |
| Event | 12 | 6.59% |
| Process_start | 11 | 6.04% |
| match_result | 6 | 3.30% |
| Attack | 5 | 2.75% |
| Killing | 5 | 2.75% |
| Cyber.PatchVulnerability | 5 | 2.75% |
| Process_end | 4 | 2.20% |
| Bodily_harm | 3 | 1.65% |
| Hostile_encounter | 3 | 1.65% |
| Destroying | 3 | 1.65% |
| Conquering | 3 | 1.65% |
| Supporting | 3 | 1.65% |
| Control | 3 | 1.65% |
| Cause_to_be_included | 3 | 1.65% |
| joined | 2 | 1.10% |
| Coming_to_be | 2 | 1.10% |
| Name_conferral | 2 | 1.10% |
| Preventing_or_letting | 2 | 1.10% |
| Sending | 2 | 1.10% |
| Death | 2 | 1.10% |
| Supply | 2 | 1.10% |
| Self_motion | 2 | 1.10% |
| Building | 2 | 1.10% |
| Military_operation | 2 | 1.10% |
| Statement | 2 | 1.10% |
| Presence | 2 | 1.10% |
| price_increase | 1 | 0.55% |
| rate_decrease | 1 | 0.55% |
| Arrest | 1 | 0.55% |
| Testing | 1 | 0.55% |
| Choosing | 1 | 0.55% |
| birthday_celebration | 1 | 0.55% |
| dissolution | 1 | 0.55% |
| price_decrease | 1 | 0.55% |
| Accident | 1 | 0.55% |
| product_publish | 1 | 0.55% |
| Escaping | 1 | 0.55% |
| Catastrophe | 1 | 0.55% |
| Rewards_and_punishments | 1 | 0.55% |
| Reveal_secret | 1 | 0.55% |
| GiveUp | 1 | 0.55% |
| Arriving | 1 | 0.55% |
| Revenge | 1 | 0.55% |
| Connect | 1 | 0.55% |
| Cause_change_of_strength | 1 | 0.55% |
| Request | 1 | 0.55% |
| Change_of_leadership | 1 | 0.55% |
| Getting | 1 | 0.55% |
| Cause_to_amalgamate | 1 | 0.55% |
| Creating | 1 | 0.55% |
| Assistance | 1 | 0.55% |
| Quarreling | 1 | 0.55% |
| proposal | 1 | 0.55% |
| Deploy | 1 | 0.55% |
| resignation | 1 | 0.55% |
| dismissal | 1 | 0.55% |
| judicial_action_prosecution | 1 | 0.55% |
| product_delist | 1 | 0.55% |
| judicial_action_arrest | 1 | 0.55% |
| Becoming | 1 | 0.55% |
| Receiving | 1 | 0.55% |
| Influence | 1 | 0.55% |
| Motion_directional | 1 | 0.55% |
| Dispersal | 1 | 0.55% |
| Causation | 1 | 0.55% |
| Check | 1 | 0.55% |
| Aiming | 1 | 0.55% |
| Placing | 1 | 0.55% |
| Earnings_and_losses | 1 | 0.55% |
| Traveling | 1 | 0.55% |

**test.** 227 instances in 34 records, 0 triggerless. Trigger gold 206. Argument gold 159. Arguments per instance, mean 0.74. Trigger length in whitespace tokens, mean 1.25, single-token share 79.5%.

#### test event types

| label | count | share |
| --- | --- | --- |
| Event | 16 | 6.69% |
| Cyber.DiscoverVulnerability | 15 | 6.28% |
| Causation | 9 | 3.77% |
| Know | 8 | 3.35% |
| Cyber.PatchVulnerability | 7 | 2.93% |
| Catastrophe | 6 | 2.51% |
| Attack | 6 | 2.51% |
| Self_motion | 6 | 2.51% |
| Process_start | 5 | 2.09% |
| Killing | 5 | 2.09% |
| Death | 5 | 2.09% |
| Name_conferral | 5 | 2.09% |
| Social_event | 5 | 2.09% |
| Research | 4 | 1.67% |
| Supporting | 4 | 1.67% |
| Creating | 4 | 1.67% |
| Experiment | 4 | 1.67% |
| Process_end | 4 | 1.67% |
| Scrutiny | 3 | 1.26% |
| Education_teaching | 3 | 1.26% |
| financing | 3 | 1.26% |
| Motion | 3 | 1.26% |
| championship_win | 3 | 1.26% |
| Manoeuvre | 3 | 1.26% |
| Cause_change_of_position_on_a_scale | 3 | 1.26% |
| Award | 3 | 1.26% |
| Arranging | 2 | 0.84% |
| Hostile_encounter | 2 | 0.84% |
| Removing | 2 | 0.84% |
| Judgment_communication | 2 | 0.84% |
| death | 2 | 0.84% |
| Accident | 2 | 0.84% |
| Earnings_and_losses | 2 | 0.84% |
| Reveal_secret | 2 | 0.84% |
| Becoming | 2 | 0.84% |
| Participation | 2 | 0.84% |
| match_result | 2 | 0.84% |
| Deploy | 2 | 0.84% |
| Escaping | 2 | 0.84% |
| Preventing_or_letting | 2 | 0.84% |
| Violence | 2 | 0.84% |
| Perception_active | 2 | 0.84% |
| Supply | 2 | 0.84% |
| Giving | 2 | 0.84% |
| Agree_or_refuse_to_act | 2 | 0.84% |
| Statement | 2 | 0.84% |
| Bodily_harm | 2 | 0.84% |
| product_recall | 2 | 0.84% |
| Cyber.Databreach | 2 | 0.84% |
| Coming_to_be | 2 | 0.84% |
| Placing | 2 | 0.84% |
| Getting | 2 | 0.84% |
| opening | 1 | 0.42% |
| thank | 1 | 0.42% |
| product_publish | 1 | 0.42% |
| judicial_action_arrest | 1 | 0.42% |
| Achieve | 1 | 0.42% |
| Employment | 1 | 0.42% |
| Come_together | 1 | 0.42% |
| Control | 1 | 0.42% |
| Being_in_operation | 1 | 0.42% |
| Limiting | 1 | 0.42% |
| Warning | 1 | 0.42% |
| Lighting | 1 | 0.42% |
| Openness | 1 | 0.42% |
| Writing | 1 | 0.42% |
| Exhibit | 1 | 0.42% |
| judicial_action_filing | 1 | 0.42% |
| Explosion | 1 | 0.42% |
| Support | 1 | 0.42% |
| judicial_action_hearing | 1 | 0.42% |
| Action | 1 | 0.42% |
| Cause_to_amalgamate | 1 | 0.42% |
| Check | 1 | 0.42% |
| Publishing | 1 | 0.42% |
| Legality | 1 | 0.42% |
| Quarreling | 1 | 0.42% |
| Theft | 1 | 0.42% |
| Choosing | 1 | 0.42% |
| Expressing_publicly | 1 | 0.42% |
| Cure | 1 | 0.42% |
| Committing_crime | 1 | 0.42% |
| Change | 1 | 0.42% |
| Having_or_lacking_access | 1 | 0.42% |
| Manufacturing | 1 | 0.42% |
| Cyber.Phishing | 1 | 0.42% |
| price_limit_up | 1 | 0.42% |
| closing | 1 | 0.42% |
| Influence | 1 | 0.42% |
| Scouring | 1 | 0.42% |
| judicial_action_imprisonment | 1 | 0.42% |
| Request | 1 | 0.42% |
| Arriving | 1 | 0.42% |
| Cause_to_be_included | 1 | 0.42% |
| Defending | 1 | 0.42% |
| Receiving | 1 | 0.42% |
| Conflict | 1 | 0.42% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 4,354 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 204 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 168 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.69 | 30.12% | 1.73 | 29.27% | 0.04% | 0 | 4,354 | 0 | 4,744 |
| val | 1.58 | 31.32% | 1.98 | 35.47% | 0.26% | 0 | 204 | 0 | 182 |
| test | 1.60 | 30.96% | 1.77 | 27.98% | 0.00% | 0 | 168 | 0 | 239 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4,664 | 0.77% | 3,777 | 9.93% | 9.40% | 1.40% | 0.56% | 14.83% |
| val | 174 | 1.72% | 283 | 12.72% | 12.01% | 2.83% | 0.35% | 17.95% |
| test | 227 | 1.32% | 156 | 6.41% | 6.41% | 1.28% | 0.64% | 17.65% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 36 | 157 | 3,777 |
| val | 3 | 15 | 283 |
| test | 3 | 34 | 156 |

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

| split | mentions the string rule adds | arguments skipped, 2+ clusters | arguments in one multi-mention cluster | other mentions of that cluster | argument spans in 2+ clusters |
| --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | 0 | 0 | 0 |
| val | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 |

### Entities

#### train

| label | count | share |
| --- | --- | --- |
| StartDate | 1,214 | 5.69% |
| EquityHolder | 1,119 | 5.25% |
| StockCode | 970 | 4.55% |
| CompanyName | 829 | 3.89% |
| StockAbbr | 821 | 3.85% |
| PledgedShares | 787 | 3.69% |
| Location | 777 | 3.64% |
| Date | 720 | 3.38% |
| Pledgee | 644 | 3.02% |
| TotalHoldingShares | 620 | 2.91% |
| TotalHoldingRatio | 558 | 2.62% |
| TradedShares | 556 | 2.61% |
| Casualties and Losses | 521 | 2.44% |
| Reduction Amount | 442 | 2.07% |
| TotalPledgedShares | 433 | 2.03% |
| EndDate | 363 | 1.70% |
| ReleasedDate | 338 | 1.59% |
| Reducing Shareholder | 321 | 1.51% |
| Cause | 291 | 1.36% |
| Company Name | 264 | 1.24% |
| Reduction Start Date | 259 | 1.21% |
| Announcement Date | 234 | 1.10% |
| LaterHoldingShares | 204 | 0.96% |
| AveragePrice | 143 | 0.67% |
| pledge_amount | 141 | 0.66% |
| Organization | 141 | 0.66% |
| Candidates and their Political Parties | 118 | 0.55% |
| pledgor | 115 | 0.54% |
| Person | 107 | 0.50% |
| Recipient | 107 | 0.50% |
| Loss Amount | 107 | 0.50% |
| Affected Areas | 102 | 0.48% |
| Weapon and Equippment | 95 | 0.45% |
| Attacker | 94 | 0.44% |
| Accepting Court | 94 | 0.44% |
| Protesters | 91 | 0.43% |
| Protest Reason | 90 | 0.42% |
| Target | 89 | 0.42% |
| protein | 87 | 0.41% |
| Vulnerability | 87 | 0.41% |
| RepurchasedShares | 83 | 0.39% |
| Accident Investigator | 81 | 0.38% |
| Contest Participant | 76 | 0.36% |
| Position | 75 | 0.35% |
| Accusation | 74 | 0.35% |
| Individual_protein | 71 | 0.33% |
| Charged Crime | 70 | 0.33% |
| HighestTradingPrice | 68 | 0.32% |
| Strike Reason | 65 | 0.30% |
| pledge_start_date | 63 | 0.30% |
| Frozen Amount | 61 | 0.29% |
| Increase Amount | 61 | 0.29% |
| Husband | 61 | 0.29% |
| ruling_date | 59 | 0.28% |
| ClosingDate | 58 | 0.27% |
| Aid Agency | 55 | 0.26% |
| Scheduled Landing Place | 55 | 0.26% |
| frozen_shareholder | 54 | 0.25% |
| Wife | 54 | 0.25% |
| Policy Content | 53 | 0.25% |
| Flight No. | 52 | 0.24% |
| Judgement Result/Prison Term | 52 | 0.24% |
| Summit Theme | 52 | 0.24% |
| Storm Hit Location | 51 | 0.24% |
| Country Visited | 51 | 0.24% |
| Suspect | 50 | 0.23% |
| Belligerents | 49 | 0.23% |
| Increasing Shareholder | 49 | 0.23% |
| Historical Sites | 49 | 0.23% |
| Anti-pollution People/Organizations | 48 | 0.23% |
| Capabilities | 46 | 0.22% |
| LegalInstitution | 46 | 0.22% |
| Crew | 45 | 0.21% |
| Influence People | 45 | 0.21% |
| RepurchaseAmount | 44 | 0.21% |
| Time | 43 | 0.20% |
| Aircraft Agency | 43 | 0.20% |
| Affected Area | 43 | 0.20% |
| Participants | 42 | 0.20% |
| System | 41 | 0.19% |
| Solution | 41 | 0.19% |
| Resignated Person | 40 | 0.19% |
| Accused People | 40 | 0.19% |
| Strike Agency | 40 | 0.19% |
| Resign Reason | 39 | 0.18% |
| Speaker Status | 39 | 0.18% |
| Game Name | 38 | 0.18% |
| Patch | 38 | 0.18% |
| Habitat | 38 | 0.18% |
| Increase Start Date | 36 | 0.17% |
| Microorganism | 35 | 0.16% |
| FrozeShares | 34 | 0.16% |
| Organization Name | 34 | 0.16% |
| Strike Outcome | 34 | 0.16% |
| Freeze Start Date | 33 | 0.15% |
| Award | 33 | 0.15% |
| Area Affected | 33 | 0.15% |
| pledge_end_date | 32 | 0.15% |
| PII | 32 | 0.15% |
| Discover Location | 32 | 0.15% |
| Start Date | 32 | 0.15% |
| LowestTradingPrice | 30 | 0.14% |
| Software | 30 | 0.14% |
| Pollution Source | 30 | 0.14% |
| Areas affected | 30 | 0.14% |
| Employment Agency | 29 | 0.14% |
| The Final Votes and Percentages | 29 | 0.14% |
| Champions | 29 | 0.14% |
| Speaker | 29 | 0.14% |
| compensation_amount | 28 | 0.13% |
| Deliberating Agency | 28 | 0.13% |
| Disease | 28 | 0.13% |
| Host | 28 | 0.13% |
| Magnitude | 28 | 0.13% |
| Damaged Facility | 27 | 0.13% |
| Poisoning Type | 27 | 0.13% |
| Artifacts and Their Chronology | 27 | 0.13% |
| CellType | 27 | 0.13% |
| Aid Supplies/Amount | 26 | 0.12% |
| Marriage Duration | 25 | 0.12% |
| Countries Withdrawing from the Organization | 25 | 0.12% |
| Election Name | 24 | 0.11% |
| Location/Hospital | 24 | 0.11% |
| Score | 24 | 0.11% |
| Acquiree | 24 | 0.11% |
| Approver | 23 | 0.11% |
| Court | 23 | 0.11% |
| Winner | 23 | 0.11% |
| Visitor | 23 | 0.11% |
| Profession | 22 | 0.10% |
| Protest Slogan | 22 | 0.10% |
| People | 22 | 0.10% |
| protein-family | 22 | 0.10% |
| Economic Loss | 21 | 0.10% |
| compensation_recipient | 20 | 0.09% |
| Deceased | 20 | 0.09% |
| Passengers | 20 | 0.09% |
| The Charged Crime | 20 | 0.09% |
| Participating Countries | 20 | 0.09% |
| Effective Date | 20 | 0.09% |
| Phenotype | 20 | 0.09% |
| Strikers Status | 20 | 0.09% |
| Policy Proposals | 20 | 0.09% |
| Survivors | 19 | 0.09% |
| Appointee | 19 | 0.09% |
| Released People | 19 | 0.09% |
| Spectacle Location | 19 | 0.09% |
| Anatomy | 19 | 0.09% |
| Method | 18 | 0.08% |
| senior_personnel | 18 | 0.08% |
| Record-breaking Project | 17 | 0.08% |
| senior_position | 17 | 0.08% |
| Child | 17 | 0.08% |
| Version | 17 | 0.08% |
| Outbreak Date | 17 | 0.08% |
| Previous Record Holder | 16 | 0.08% |
| Arrested | 16 | 0.08% |
| Storm Name | 16 | 0.08% |
| Freeze End Date | 16 | 0.08% |
| Data | 16 | 0.08% |
| Attending Hospital | 16 | 0.08% |
| Competition Items | 16 | 0.08% |
| Storm Hit Time | 16 | 0.08% |
| Time of the Case | 16 | 0.08% |
| Prosecutor | 16 | 0.08% |
| ChemicalEntity | 16 | 0.08% |
| Release Reason | 16 | 0.08% |
| Device | 15 | 0.07% |
| Death Reason | 15 | 0.07% |
| Defense Lawyer | 15 | 0.07% |
| Number of Vehicles Involved in the Crash | 15 | 0.07% |
| Award Reason | 15 | 0.07% |
| Spacecraft | 15 | 0.07% |
| Investigative Agency | 15 | 0.07% |
| Grades | 15 | 0.07% |
| Number of Influenced People | 15 | 0.07% |
| Train No. | 15 | 0.07% |
| Other Loss | 14 | 0.07% |
| Number of Damaged Houses | 14 | 0.07% |
| Death Time | 14 | 0.07% |
| End Time | 14 | 0.07% |
| Organization Industry | 14 | 0.07% |
| Death Cases | 14 | 0.07% |
| Declarer | 14 | 0.07% |
| Precautionary Measure | 14 | 0.07% |
| Evidence | 14 | 0.07% |
| Volcano Name | 14 | 0.07% |
| Tribe | 14 | 0.07% |
| Phylum | 14 | 0.07% |
| Age | 13 | 0.06% |
| Record Breaker | 13 | 0.06% |
| Judge | 13 | 0.06% |
| Number of Evacuated People | 13 | 0.06% |
| Website | 13 | 0.06% |
| Symptom | 13 | 0.06% |
| Maximum Wind Speed | 13 | 0.06% |
| Contracting Parties | 13 | 0.06% |
| Strikers | 13 | 0.06% |
| Weapon | 12 | 0.06% |
| Predecessor | 12 | 0.06% |
| Spectacle Start Time | 12 | 0.06% |
| Types of the Spectacle | 12 | 0.06% |
| Ways to Watch the Speech | 12 | 0.06% |
| Sponsor | 12 | 0.06% |
| Annoncement Date | 12 | 0.06% |
| Malware | 12 | 0.06% |
| Boycotted Institutions | 12 | 0.06% |
| Archaeologist | 12 | 0.06% |
| Symptoms | 12 | 0.06% |
| Treatment Method | 12 | 0.06% |
| Magnitude(Tsunami heights) | 12 | 0.06% |
| GPE | 11 | 0.05% |
| Riot Reason | 11 | 0.05% |
| Outbreak Location | 11 | 0.05% |
| Spacecraft Mission | 11 | 0.05% |
| Term of Office | 11 | 0.05% |
| Agreement Name | 11 | 0.05% |
| Court Time | 11 | 0.05% |
| Fined Agency | 11 | 0.05% |
| Arrest Time | 11 | 0.05% |
| Police | 11 | 0.05% |
| Epicenter | 11 | 0.05% |
| Child Custody | 11 | 0.05% |
| Work | 11 | 0.05% |
| The Grades of the Previous Record Holder | 10 | 0.05% |
| protein-complex | 10 | 0.05% |
| Conflict Duration | 10 | 0.05% |
| Party Responsible for Pollution | 10 | 0.05% |
| Money | 10 | 0.05% |
| Military Strength | 10 | 0.05% |
| The Agency Who Broke the Agreement | 10 | 0.05% |
| Company Industry | 10 | 0.05% |
| Discover Time | 10 | 0.05% |
| Fine Reason | 10 | 0.05% |
| Launch Date | 10 | 0.05% |
| Casualities and Losses | 10 | 0.05% |
| Host Country | 10 | 0.05% |
| End Date | 10 | 0.05% |
| Duration | 10 | 0.05% |
| DiseaseOrPhenotypicFeature | 10 | 0.05% |
| GeneProtein | 10 | 0.05% |
| Pests | 9 | 0.04% |
| Successor | 9 | 0.04% |
| Release Time | 9 | 0.04% |
| Criminal Evidence | 9 | 0.04% |
| chemical | 9 | 0.04% |
| Research Agency | 9 | 0.04% |
| Country | 9 | 0.04% |
| Person under Investigation | 9 | 0.04% |
| Withdraw Reason | 9 | 0.04% |
| CVE | 9 | 0.04% |
| Award Field | 9 | 0.04% |
| Missings | 8 | 0.04% |
| Turnout(who win who lost) | 8 | 0.04% |
| Prison Term | 8 | 0.04% |
| People/Organization who predicted the disaster | 8 | 0.04% |
| Head of the Merged Organization | 8 | 0.04% |
| Taking-off Place | 8 | 0.04% |
| Investigating agency | 8 | 0.04% |
| Penalty Amount | 8 | 0.04% |
| Victim | 8 | 0.04% |
| Army | 8 | 0.04% |
| GeneOrGeneProduct | 8 | 0.04% |
| File | 8 | 0.04% |
| Last Job of the Appointee | 7 | 0.03% |
| Sentencing Location | 7 | 0.03% |
| Jail Time | 7 | 0.03% |
| Acquirer | 7 | 0.03% |
| Confirmed/Infected Cases | 7 | 0.03% |
| cell | 7 | 0.03% |
| Purpose | 7 | 0.03% |
| Gene/protein/RNA | 7 | 0.03% |
| DNA | 7 | 0.03% |
| Way for Spreading | 7 | 0.03% |
| Rescue Organizer | 7 | 0.03% |
| Concept | 7 | 0.03% |
| Inauguration Time | 6 | 0.03% |
| Arrest Location | 6 | 0.03% |
| Age of the Resignated Person | 6 | 0.03% |
| Appointer | 6 | 0.03% |
| Beneficiary | 6 | 0.03% |
| Spokeswoman/Spokesman | 6 | 0.03% |
| News Release Agency | 6 | 0.03% |
| species | 6 | 0.03% |
| Bank Name | 6 | 0.03% |
| Number | 6 | 0.03% |
| Prosecution Lawyer | 6 | 0.03% |
| Storm Warning Level | 6 | 0.03% |
| Military Exercise | 6 | 0.03% |
| Number of Destroyed Building | 6 | 0.03% |
| Start Time | 6 | 0.03% |
| Lasting Time | 6 | 0.03% |
| Affected Industries | 6 | 0.03% |
| Government Reaction | 6 | 0.03% |
| Launch Result | 6 | 0.03% |
| Merger Terms | 6 | 0.03% |
| Casualty Count | 6 | 0.03% |
| Ship No. | 6 | 0.03% |
| Shipwreck Reason | 5 | 0.02% |
| Witness | 5 | 0.02% |
| Appointment Time | 5 | 0.02% |
| Epidemic Data Issuing Agency | 5 | 0.02% |
| Funding | 5 | 0.02% |
| Regulatory Authority | 5 | 0.02% |
| Compensation | 5 | 0.02% |
| Acquisition amount | 5 | 0.02% |
| Policy Name & Abbreviation | 5 | 0.02% |
| Start_Date | 5 | 0.02% |
| Last Time the Record was Broken | 5 | 0.02% |
| Train Agency | 5 | 0.02% |
| Doctor and Medical Team | 5 | 0.02% |
| Summit Name | 5 | 0.02% |
| The State of the Volcano (Dormant or Active) | 5 | 0.02% |
| Last Outbreak Time | 5 | 0.02% |
| Stolen Amount | 5 | 0.02% |
| Sequelae | 5 | 0.02% |
| Literary Work | 5 | 0.02% |
| mathematician | 5 | 0.02% |
| Voting method | 4 | 0.02% |
| Disaster-Stricken Farmland | 4 | 0.02% |
| MVP | 4 | 0.02% |
| Protein_family_or_group | 4 | 0.02% |
| Employed Institution | 4 | 0.02% |
| Aid Reason | 4 | 0.02% |
| Achievement | 4 | 0.02% |
| UnfrozeDate | 4 | 0.02% |
| fusion-protein | 4 | 0.02% |
| Carrier Rocket | 4 | 0.02% |
| Storm Movement Speed | 4 | 0.02% |
| Tear Up Reason | 4 | 0.02% |
| Storm Direction | 4 | 0.02% |
| Influenced Crops and Livelihood | 4 | 0.02% |
| Rescue Tool or Method | 4 | 0.02% |
| Economists who Predicted the Crisis | 4 | 0.02% |
| Property Division | 4 | 0.02% |
| Number of Rescued People | 4 | 0.02% |
| Unemployed Rate | 4 | 0.02% |
| Cured Cases | 4 | 0.02% |
| Number of Trapped People | 4 | 0.02% |
| Organization Members | 4 | 0.02% |
| Strike Industry | 4 | 0.02% |
| Other Impact | 4 | 0.02% |
| ai_model | 4 | 0.02% |
| Weapon Used | 4 | 0.02% |
| Perpetrators | 4 | 0.02% |
| Bankrupt Business | 4 | 0.02% |
| Response Measures | 4 | 0.02% |
| mathematical_operator | 4 | 0.02% |
| Tsunamis | 4 | 0.02% |
| Species | 4 | 0.02% |
| State Before Death | 3 | 0.01% |
| Illness | 3 | 0.01% |
| Economic loss | 3 | 0.01% |
| Forecasting Agency | 3 | 0.01% |
| Spectacle End Time | 3 | 0.01% |
| Spectacle Duration | 3 | 0.01% |
| reagent | 3 | 0.01% |
| Announce Platform | 3 | 0.01% |
| Agreement Members | 3 | 0.01% |
| Election Goal | 3 | 0.01% |
| Investigating Spokesperson | 3 | 0.01% |
| Hostage | 3 | 0.01% |
| Candidate Scandal(Allegations of fraud, etc.) | 3 | 0.01% |
| People Casting Key Votes | 3 | 0.01% |
| Agreement Content | 3 | 0.01% |
| Projects | 3 | 0.01% |
| The Final Seats Result | 3 | 0.01% |
| Economy Loss | 3 | 0.01% |
| OrganismTaxon | 3 | 0.01% |
| Number of Vaccinated People | 3 | 0.01% |
| Susceptible Population | 3 | 0.01% |
| Material | 3 | 0.01% |
| Launch Site | 3 | 0.01% |
| Damaged Property | 3 | 0.01% |
| Age of the Appointee | 3 | 0.01% |
| Number of Aftershocks | 3 | 0.01% |
| Archaeologist Organization | 3 | 0.01% |
| Bill Drafting Agency | 3 | 0.01% |
| Nation | 3 | 0.01% |
| Subphylum | 3 | 0.01% |
| Organism | 3 | 0.01% |
| algebraic_structure | 3 | 0.01% |
| mathematical_construct | 3 | 0.01% |
| Service Years | 2 | 0.01% |
| Water Level | 2 | 0.01% |
| Best Way to Shoot | 2 | 0.01% |
| Production Cuts | 2 | 0.01% |
| Influenced People | 2 | 0.01% |
| Related Rivers or Lakes | 2 | 0.01% |
| Mission Duration | 2 | 0.01% |
| Countries Joined the Organization | 2 | 0.01% |
| Launch Country | 2 | 0.01% |
| experimental-construct | 2 | 0.01% |
| Scale | 2 | 0.01% |
| Postpone Reason | 2 | 0.01% |
| Wedding Venue | 2 | 0.01% |
| Invited Person | 2 | 0.01% |
| Special Medicine | 2 | 0.01% |
| Perpetrator | 2 | 0.01% |
| The Sentence Claimed by the Prosecutor's Lawyer | 2 | 0.01% |
| Organization Leader | 2 | 0.01% |
| Head of Investigation Team | 2 | 0.01% |
| PaymentMethod | 2 | 0.01% |
| Cooperative Agency | 2 | 0.01% |
| Plant Species | 2 | 0.01% |
| Commanders of the Army | 2 | 0.01% |
| amino-acid | 2 | 0.01% |
| evaluation_metric | 2 | 0.01% |
| baseline_method | 2 | 0.01% |
| method_component | 2 | 0.01% |
| Transportation | 2 | 0.01% |
| Number of Victims | 2 | 0.01% |
| Agreement Validity Period | 2 | 0.01% |
| Fire Warning Level | 2 | 0.01% |
| protein-domain | 2 | 0.01% |
| Ship Agency | 2 | 0.01% |
| Temporary Settlement | 2 | 0.01% |
| Group | 2 | 0.01% |
| Civilization | 2 | 0.01% |
| Law | 2 | 0.01% |
| Biological Group | 2 | 0.01% |
| Geological Period | 2 | 0.01% |
| Amount of Precipitation | 2 | 0.01% |
| CellComponent | 2 | 0.01% |
| Damaged Crops & Livestock | 2 | 0.01% |
| Areas Affected | 2 | 0.01% |
| mutation | 2 | 0.01% |
| Field | 2 | 0.01% |
| Prison | 1 | 0.00% |
| The Worst-Hit Area | 1 | 0.00% |
| protein-isoform | 1 | 0.00% |
| protein-motif | 1 | 0.00% |
| Provisional Government | 1 | 0.00% |
| Electoral System | 1 | 0.00% |
| Invalid Date | 1 | 0.00% |
| Head of the Government | 1 | 0.00% |
| Storm Center Location | 1 | 0.00% |
| End_Date | 1 | 0.00% |
| Number of People Hospitalized | 1 | 0.00% |
| The Sentence Claimed by the Defense Lawyer | 1 | 0.00% |
| Complications | 1 | 0.00% |
| Measurement | 1 | 0.00% |
| Plant Group | 1 | 0.00% |
| Habitat Feature | 1 | 0.00% |
| Chemical | 1 | 0.00% |
| Animal | 1 | 0.00% |
| Trapped Depth | 1 | 0.00% |
| Spokenmen | 1 | 0.00% |
| research_task | 1 | 0.00% |
| framework | 1 | 0.00% |
| organization | 1 | 0.00% |
| dataset | 1 | 0.00% |
| input | 1 | 0.00% |
| constraint | 1 | 0.00% |
| intermediate_representation | 1 | 0.00% |
| output | 1 | 0.00% |
| actor | 1 | 0.00% |
| Death Rate | 1 | 0.00% |
| Lawyer | 1 | 0.00% |
| Detention Start Time | 1 | 0.00% |
| Goal | 1 | 0.00% |
| State of the Hull | 1 | 0.00% |
| Refuge | 1 | 0.00% |
| Event | 1 | 0.00% |
| Document | 1 | 0.00% |
| Genetic Element | 1 | 0.00% |
| Protein_complex | 1 | 0.00% |
| Field of Study | 1 | 0.00% |
| Narrative | 1 | 0.00% |
| Text | 1 | 0.00% |
| Invention | 1 | 0.00% |
| Maximum Rainfall | 1 | 0.00% |
| topological_space | 1 | 0.00% |
| mathematical_identity | 1 | 0.00% |
| Gene | 1 | 0.00% |
| Head of Institution | 1 | 0.00% |
| Reasons for the Formation of the Historical sites | 1 | 0.00% |
| Death Age | 1 | 0.00% |
| Recovered Amount | 1 | 0.00% |
| How Many Times Get Married | 1 | 0.00% |
| Warning Device | 1 | 0.00% |
| agent | 1 | 0.00% |
| target | 1 | 0.00% |
| Language | 1 | 0.00% |
| Word | 1 | 0.00% |
| Object | 1 | 0.00% |
| Medium | 1 | 0.00% |

#### val

| label | count | share |
| --- | --- | --- |
| StartDate | 67 | 7.02% |
| EquityHolder | 54 | 5.65% |
| StockCode | 41 | 4.29% |
| StockAbbr | 37 | 3.87% |
| CompanyName | 34 | 3.56% |
| PledgedShares | 34 | 3.56% |
| TotalHoldingShares | 31 | 3.25% |
| TotalHoldingRatio | 31 | 3.25% |
| Location | 30 | 3.14% |
| Reduction Amount | 30 | 3.14% |
| EndDate | 28 | 2.93% |
| TradedShares | 28 | 2.93% |
| Pledgee | 27 | 2.83% |
| Reduction Start Date | 23 | 2.41% |
| Date | 21 | 2.20% |
| TotalPledgedShares | 18 | 1.88% |
| Cause | 18 | 1.88% |
| Reducing Shareholder | 16 | 1.68% |
| Vulnerability | 15 | 1.57% |
| ReleasedDate | 13 | 1.36% |
| Organization | 13 | 1.36% |
| AveragePrice | 13 | 1.36% |
| Casualties and Losses | 12 | 1.26% |
| GeneOrGeneProduct | 12 | 1.26% |
| Grades | 9 | 0.94% |
| Person | 9 | 0.94% |
| Announcement Date | 8 | 0.84% |
| LaterHoldingShares | 8 | 0.84% |
| Company Name | 7 | 0.73% |
| Device | 7 | 0.73% |
| Time | 7 | 0.73% |
| Flight No. | 7 | 0.73% |
| Record Breaker | 6 | 0.63% |
| Attacker | 6 | 0.63% |
| Strike Reason | 6 | 0.63% |
| Record-breaking Project | 5 | 0.52% |
| Policy Content | 5 | 0.52% |
| Last Time the Record was Broken | 5 | 0.52% |
| Recipient | 5 | 0.52% |
| Patch | 5 | 0.52% |
| Storm Hit Location | 5 | 0.52% |
| Strike Outcome | 5 | 0.52% |
| Start Date | 5 | 0.52% |
| Previous Record Holder | 4 | 0.42% |
| Court | 4 | 0.42% |
| Judgement Result/Prison Term | 4 | 0.42% |
| Accusation | 4 | 0.42% |
| Symptom | 4 | 0.42% |
| Accepting Court | 4 | 0.42% |
| pledgor | 4 | 0.42% |
| pledge_amount | 4 | 0.42% |
| Capabilities | 4 | 0.42% |
| Scheduled Landing Place | 4 | 0.42% |
| Suspect | 3 | 0.31% |
| Disease | 3 | 0.31% |
| Population | 3 | 0.31% |
| Frozen Amount | 3 | 0.31% |
| model | 3 | 0.31% |
| embodiment | 3 | 0.31% |
| Investigative Agency | 3 | 0.31% |
| Accident Investigator | 3 | 0.31% |
| OrganismTaxon | 3 | 0.31% |
| DiseaseOrPhenotypicFeature | 3 | 0.31% |
| SequenceVariant | 3 | 0.31% |
| Target | 3 | 0.31% |
| The Grades of the Previous Record Holder | 3 | 0.31% |
| Increase Start Date | 2 | 0.21% |
| Increasing Shareholder | 2 | 0.21% |
| Increase Amount | 2 | 0.21% |
| Court Time | 2 | 0.21% |
| Judge | 2 | 0.21% |
| Enzyme | 2 | 0.21% |
| Body Part | 2 | 0.21% |
| Disease Form | 2 | 0.21% |
| Medical Test | 2 | 0.21% |
| frozen_shareholder | 2 | 0.21% |
| Effective Date | 2 | 0.21% |
| modality | 2 | 0.21% |
| metric | 2 | 0.21% |
| Loss Amount | 2 | 0.21% |
| Deliberating Agency | 2 | 0.21% |
| Areas affected | 2 | 0.21% |
| Spectacle Start Time | 2 | 0.21% |
| Participants | 2 | 0.21% |
| Aircraft Agency | 2 | 0.21% |
| Attending Hospital | 2 | 0.21% |
| Train No. | 2 | 0.21% |
| ChemicalEntity | 2 | 0.21% |
| Weapon and Equippment | 2 | 0.21% |
| ruling_date | 2 | 0.21% |
| Gene | 2 | 0.21% |
| Number of Vehicles Involved in the Crash | 2 | 0.21% |
| FrozeShares | 2 | 0.21% |
| Artifacts and Their Chronology | 2 | 0.21% |
| Strikers Status | 2 | 0.21% |
| Strike Agency | 2 | 0.21% |
| Strikers | 2 | 0.21% |
| Carrier Rocket | 1 | 0.10% |
| Launch Site | 1 | 0.10% |
| Spacecraft | 1 | 0.10% |
| Launch Result | 1 | 0.10% |
| Launch Date | 1 | 0.10% |
| Substance | 1 | 0.10% |
| Biological Entity | 1 | 0.10% |
| Service | 1 | 0.10% |
| Freeze Start Date | 1 | 0.10% |
| Spokenmen | 1 | 0.10% |
| Affected Areas | 1 | 0.10% |
| Number of Influenced People | 1 | 0.10% |
| Belligerents | 1 | 0.10% |
| Weapon | 1 | 0.10% |
| dataset | 1 | 0.10% |
| environment | 1 | 0.10% |
| task_category | 1 | 0.10% |
| capability | 1 | 0.10% |
| group | 1 | 0.10% |
| Precautionary Measure | 1 | 0.10% |
| Types of the Spectacle | 1 | 0.10% |
| Spectacle Location | 1 | 0.10% |
| Spectacle End Time | 1 | 0.10% |
| Summit Name | 1 | 0.10% |
| Summit Theme | 1 | 0.10% |
| pledge_start_date | 1 | 0.10% |
| Head of Investigation Team | 1 | 0.10% |
| Person under Investigation | 1 | 0.10% |
| Approver | 1 | 0.10% |
| Resignated Person | 1 | 0.10% |
| Resign Reason | 1 | 0.10% |
| Position | 1 | 0.10% |
| Age of the Resignated Person | 1 | 0.10% |
| Survivors | 1 | 0.10% |
| Prison | 1 | 0.10% |
| Storm Name | 1 | 0.10% |
| Storm Warning Level | 1 | 0.10% |
| Storm Hit Time | 1 | 0.10% |
| Influence People | 1 | 0.10% |
| People/Organization who predicted the disaster | 1 | 0.10% |
| Areas Affected | 1 | 0.10% |
| Train Agency | 1 | 0.10% |
| LegalInstitution | 1 | 0.10% |
| UnfrozeDate | 1 | 0.10% |
| Discover Time | 1 | 0.10% |
| Discover Location | 1 | 0.10% |
| End Date | 1 | 0.10% |
| Economy Loss | 1 | 0.10% |

#### test

| label | count | share |
| --- | --- | --- |
| StartDate | 50 | 5.13% |
| EquityHolder | 49 | 5.03% |
| PledgedShares | 45 | 4.62% |
| StockCode | 41 | 4.21% |
| CompanyName | 37 | 3.79% |
| Location | 36 | 3.69% |
| StockAbbr | 35 | 3.59% |
| Pledgee | 33 | 3.38% |
| TotalHoldingShares | 31 | 3.18% |
| TotalHoldingRatio | 26 | 2.67% |
| Casualties and Losses | 25 | 2.56% |
| Reducing Shareholder | 25 | 2.56% |
| TradedShares | 24 | 2.46% |
| Reduction Amount | 24 | 2.46% |
| Date | 23 | 2.36% |
| TotalPledgedShares | 23 | 2.36% |
| Vulnerability | 18 | 1.85% |
| ReleasedDate | 17 | 1.74% |
| EndDate | 14 | 1.44% |
| Protesters | 13 | 1.33% |
| CVE | 13 | 1.33% |
| Company Name | 11 | 1.13% |
| Affected Areas | 11 | 1.13% |
| System | 11 | 1.13% |
| Strike Reason | 10 | 1.03% |
| Perpetrators | 10 | 1.03% |
| Product | 10 | 1.03% |
| Person | 9 | 0.92% |
| Cause | 8 | 0.82% |
| pledge_amount | 8 | 0.82% |
| Announcement Date | 8 | 0.82% |
| LaterHoldingShares | 8 | 0.82% |
| RepurchasedShares | 7 | 0.72% |
| Reduction Start Date | 7 | 0.72% |
| Arrested | 7 | 0.72% |
| Investigating agency | 7 | 0.72% |
| Recipient | 6 | 0.62% |
| Belligerents | 6 | 0.62% |
| Protest Reason | 6 | 0.62% |
| Organization | 6 | 0.62% |
| pledgor | 5 | 0.51% |
| ClosingDate | 5 | 0.51% |
| Accepting Court | 5 | 0.51% |
| Pests | 5 | 0.51% |
| The Final Votes and Percentages | 5 | 0.51% |
| Flight No. | 5 | 0.51% |
| Device | 5 | 0.51% |
| Aircraft Agency | 4 | 0.41% |
| pledge_start_date | 4 | 0.41% |
| pledge_end_date | 4 | 0.41% |
| HighestTradingPrice | 4 | 0.41% |
| Weapon and Equippment | 4 | 0.41% |
| Target | 4 | 0.41% |
| Location/Hospital | 4 | 0.41% |
| Loss Amount | 4 | 0.41% |
| RNA | 4 | 0.41% |
| Taking-off Place | 3 | 0.31% |
| Scheduled Landing Place | 3 | 0.31% |
| Weapon | 3 | 0.31% |
| Doctor and Medical Team | 3 | 0.31% |
| Symptom | 3 | 0.31% |
| Unemployed Rate | 3 | 0.31% |
| Prosecutor | 3 | 0.31% |
| Charged Crime | 3 | 0.31% |
| Candidates and their Political Parties | 3 | 0.31% |
| Election Name | 3 | 0.31% |
| ruling_date | 3 | 0.31% |
| Version | 3 | 0.31% |
| Patch | 3 | 0.31% |
| Capabilities | 3 | 0.31% |
| Spacecraft | 3 | 0.31% |
| protein | 3 | 0.31% |
| Husband | 2 | 0.21% |
| Economic Loss | 2 | 0.21% |
| Attending Hospital | 2 | 0.21% |
| Attacker | 2 | 0.21% |
| Accused People | 2 | 0.21% |
| Influenced Crops and Livelihood | 2 | 0.21% |
| Aid Agency | 2 | 0.21% |
| Response Measures | 2 | 0.21% |
| protein-family | 2 | 0.21% |
| Number of Vehicles Involved in the Crash | 2 | 0.21% |
| Previous Record Holder | 2 | 0.21% |
| Record Breaker | 2 | 0.21% |
| Grades | 2 | 0.21% |
| The Grades of the Previous Record Holder | 2 | 0.21% |
| RepurchaseAmount | 2 | 0.21% |
| Pest | 2 | 0.21% |
| Equipment | 2 | 0.21% |
| Metric | 2 | 0.21% |
| Damage | 2 | 0.21% |
| Data | 2 | 0.21% |
| Launch Date | 2 | 0.21% |
| Cooperative Agency | 2 | 0.21% |
| Carrier Rocket | 2 | 0.21% |
| Increasing Shareholder | 2 | 0.21% |
| Increase Amount | 2 | 0.21% |
| Wife | 1 | 0.10% |
| Accident Investigator | 1 | 0.10% |
| Scale | 1 | 0.10% |
| Goal | 1 | 0.10% |
| Commanders and their Position | 1 | 0.10% |
| Start_Date | 1 | 0.10% |
| Affected Area | 1 | 0.10% |
| End_Date | 1 | 0.10% |
| Strikers Status | 1 | 0.10% |
| Start Date | 1 | 0.10% |
| Strike Industry | 1 | 0.10% |
| Casualty Count | 1 | 0.10% |
| FrozeShares | 1 | 0.10% |
| LegalInstitution | 1 | 0.10% |
| Record-breaking Project | 1 | 0.10% |
| Last Time the Record was Broken | 1 | 0.10% |
| Bank Name | 1 | 0.10% |
| Weapon Used | 1 | 0.10% |
| People | 1 | 0.10% |
| Survivors | 1 | 0.10% |
| Passengers | 1 | 0.10% |
| Time | 1 | 0.10% |
| File | 1 | 0.10% |
| Scientific Name | 1 | 0.10% |
| Control Method | 1 | 0.10% |
| Strategy | 1 | 0.10% |
| Natural Enemy | 1 | 0.10% |
| Lure | 1 | 0.10% |
| AveragePrice | 1 | 0.10% |
| Employed Institution | 1 | 0.10% |
| Winner | 1 | 0.10% |
| Award Field | 1 | 0.10% |
| Award | 1 | 0.10% |
| Crew | 1 | 0.10% |
| Launch Country | 1 | 0.10% |
| Research Agency | 1 | 0.10% |
| Spacecraft Mission | 1 | 0.10% |
| Launch Result | 1 | 0.10% |
| Types of the Spectacle | 1 | 0.10% |
| Spectacle Location | 1 | 0.10% |
| Spectacle Start Time | 1 | 0.10% |
| protein-domain | 1 | 0.10% |
| organelle | 1 | 0.10% |
| senior_personnel | 1 | 0.10% |
| Magnitude | 1 | 0.10% |
| Number of Destroyed Building | 1 | 0.10% |
| LowestTradingPrice | 1 | 0.10% |
| Increase Start Date | 1 | 0.10% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| bind | 124 | 7.40% |
| PPI | 84 | 5.01% |
| Lives_In | 73 | 4.36% |
| Exhibits | 15 | 0.89% |
| original broadcaster | 12 | 0.72% |
| platform | 12 | 0.72% |
| employer | 11 | 0.66% |
| located in the administrative territorial entity | 10 | 0.60% |
| architect | 10 | 0.60% |
| present in work | 9 | 0.54% |
| father | 9 | 0.54% |
| mountain range | 9 | 0.54% |
| country | 9 | 0.54% |
| characters | 9 | 0.54% |
| crosses | 9 | 0.54% |
| owned by | 9 | 0.54% |
| discoverer or inventor | 8 | 0.48% |
| publisher | 8 | 0.48% |
| nominated for | 8 | 0.48% |
| author | 8 | 0.48% |
| partner in business or sport | 8 | 0.48% |
| Positive_Correlation | 8 | 0.48% |
| Negative_Correlation | 8 | 0.48% |
| medical condition treated | 7 | 0.42% |
| location | 7 | 0.42% |
| medical condition | 7 | 0.42% |
| operator | 7 | 0.42% |
| primary destinations | 7 | 0.42% |
| member of sports team | 7 | 0.42% |
| position held | 7 | 0.42% |
| located in the statistical territorial entity | 7 | 0.42% |
| parent taxon | 7 | 0.42% |
| cathedral | 7 | 0.42% |
| part of the series | 6 | 0.36% |
| founded by | 6 | 0.36% |
| member of political party | 6 | 0.36% |
| position played on team / speciality | 6 | 0.36% |
| participating team | 6 | 0.36% |
| place served by transport hub | 6 | 0.36% |
| record label | 6 | 0.36% |
| stock market index | 6 | 0.36% |
| located in protected area | 6 | 0.36% |
| worshipped by | 6 | 0.36% |
| winner | 6 | 0.36% |
| field of work | 6 | 0.36% |
| first appearance | 6 | 0.36% |
| dam | 6 | 0.36% |
| instrument | 6 | 0.36% |
| historic county | 6 | 0.36% |
| member of | 6 | 0.36% |
| stock exchange | 6 | 0.36% |
| production company | 6 | 0.36% |
| participant in | 6 | 0.36% |
| candidate | 5 | 0.30% |
| connects with | 5 | 0.30% |
| has effect | 5 | 0.30% |
| director | 5 | 0.30% |
| convicted of | 5 | 0.30% |
| form of creative work | 5 | 0.30% |
| creator | 5 | 0.30% |
| ethnic group | 5 | 0.30% |
| owner of | 5 | 0.30% |
| located in the present-day administrative territorial entity | 5 | 0.30% |
| mouth of the watercourse | 5 | 0.30% |
| director / manager | 5 | 0.30% |
| home venue | 5 | 0.30% |
| performer | 5 | 0.30% |
| item operated | 5 | 0.30% |
| award received | 5 | 0.30% |
| spouse | 5 | 0.30% |
| field of this occupation | 5 | 0.30% |
| media franchise | 5 | 0.30% |
| main subject | 5 | 0.30% |
| terminus | 5 | 0.30% |
| commissioned by | 5 | 0.30% |
| crew member(s) | 5 | 0.30% |
| military branch | 5 | 0.30% |
| conferred by | 5 | 0.30% |
| pole position | 5 | 0.30% |
| head of state | 5 | 0.30% |
| replaces | 5 | 0.30% |
| location of formation | 5 | 0.30% |
| history of topic | 5 | 0.30% |
| chromosome | 5 | 0.30% |
| cast member | 5 | 0.30% |
| represented by | 5 | 0.30% |
| basic form of government | 5 | 0.30% |
| educated at | 5 | 0.30% |
| secretary general | 5 | 0.30% |
| connecting service | 5 | 0.30% |
| standards body | 5 | 0.30% |
| significant event | 5 | 0.30% |
| genre | 5 | 0.30% |
| mascot | 5 | 0.30% |
| applies to jurisdiction | 4 | 0.24% |
| taxonomic type | 4 | 0.24% |
| time period | 4 | 0.24% |
| taxon rank | 4 | 0.24% |
| instance of | 4 | 0.24% |
| has works in the collection | 4 | 0.24% |
| is the study of | 4 | 0.24% |
| endemic to | 4 | 0.24% |
| country of origin | 4 | 0.24% |
| after a work by | 4 | 0.24% |
| uses | 4 | 0.24% |
| space launch vehicle | 4 | 0.24% |
| this zoological name is coordinate with | 4 | 0.24% |
| parent organization | 4 | 0.24% |
| mother | 4 | 0.24% |
| movement | 4 | 0.24% |
| has use | 4 | 0.24% |
| shares border with | 4 | 0.24% |
| presenter | 4 | 0.24% |
| grants | 4 | 0.24% |
| league | 4 | 0.24% |
| operating area | 4 | 0.24% |
| highest judicial authority | 4 | 0.24% |
| indigenous to | 4 | 0.24% |
| located in time zone | 4 | 0.24% |
| surface played on | 4 | 0.24% |
| located on astronomical body | 4 | 0.24% |
| officeholder | 4 | 0.24% |
| powered by | 4 | 0.24% |
| parent club | 4 | 0.24% |
| located in or next to body of water | 4 | 0.24% |
| conflict | 4 | 0.24% |
| vehicle | 4 | 0.24% |
| origin of the watercourse | 4 | 0.24% |
| substitute/deputy/replacement of office/officeholder | 4 | 0.24% |
| diocese | 4 | 0.24% |
| fabrication method | 4 | 0.24% |
| located in/on physical feature | 4 | 0.24% |
| producer | 4 | 0.24% |
| manifestation of | 4 | 0.24% |
| carries | 4 | 0.24% |
| broadcast by | 4 | 0.24% |
| opposite of | 4 | 0.24% |
| place of detention | 4 | 0.24% |
| sports discipline competed in | 4 | 0.24% |
| perpetrator | 4 | 0.24% |
| noble title | 4 | 0.24% |
| Association | 4 | 0.24% |
| sex or gender | 4 | 0.24% |
| has subsidiary | 4 | 0.24% |
| commanded by | 4 | 0.24% |
| basin country | 4 | 0.24% |
| continent | 4 | 0.24% |
| language of work or name | 4 | 0.24% |
| radio format | 4 | 0.24% |
| country for sport | 4 | 0.24% |
| astronaut mission | 4 | 0.24% |
| candidacy in election | 4 | 0.24% |
| merged into | 4 | 0.24% |
| physically interacts with | 4 | 0.24% |
| located on street | 4 | 0.24% |
| Comparison | 4 | 0.24% |
| writing language | 3 | 0.18% |
| distributed by | 3 | 0.18% |
| start point | 3 | 0.18% |
| member of the crew of | 3 | 0.18% |
| said to be the same as | 3 | 0.18% |
| chairperson | 3 | 0.18% |
| shape | 3 | 0.18% |
| contributed to creative work | 3 | 0.18% |
| work location | 3 | 0.18% |
| sport | 3 | 0.18% |
| dialect of | 3 | 0.18% |
| location of discovery | 3 | 0.18% |
| represents | 3 | 0.18% |
| has part(s) | 3 | 0.18% |
| transport network | 3 | 0.18% |
| signatory | 3 | 0.18% |
| adjacent station | 3 | 0.18% |
| affiliation | 3 | 0.18% |
| board member | 3 | 0.18% |
| commander of (DEPRECATED) | 3 | 0.18% |
| cause of destruction | 3 | 0.18% |
| inflows | 3 | 0.18% |
| family | 3 | 0.18% |
| headquarters location | 3 | 0.18% |
| child astronomical body | 3 | 0.18% |
| allegiance | 3 | 0.18% |
| business division | 3 | 0.18% |
| named after | 3 | 0.18% |
| operating system | 3 | 0.18% |
| sponsor | 3 | 0.18% |
| GUI toolkit or framework | 3 | 0.18% |
| airline hub | 3 | 0.18% |
| occupant | 3 | 0.18% |
| used by | 3 | 0.18% |
| home world | 3 | 0.18% |
| official religion | 3 | 0.18% |
| track gauge | 3 | 0.18% |
| software engine | 3 | 0.18% |
| package management system | 3 | 0.18% |
| military rank | 3 | 0.18% |
| objective of project or action | 3 | 0.18% |
| child | 3 | 0.18% |
| model item | 3 | 0.18% |
| immediate cause of | 3 | 0.18% |
| airline alliance | 3 | 0.18% |
| occupation | 3 | 0.18% |
| tracklist | 3 | 0.18% |
| natural product of taxon | 3 | 0.18% |
| structural engineer | 3 | 0.18% |
| followed by | 3 | 0.18% |
| location of landing | 3 | 0.18% |
| culture | 3 | 0.18% |
| composer | 3 | 0.18% |
| drug or therapy used for treatment | 3 | 0.18% |
| has immediate cause | 3 | 0.18% |
| lake outflow | 3 | 0.18% |
| fictional universe described in | 3 | 0.18% |
| has anatomical branch | 3 | 0.18% |
| approved by | 3 | 0.18% |
| historical region | 3 | 0.18% |
| licensed to broadcast to | 3 | 0.18% |
| notable work | 3 | 0.18% |
| legislated by | 3 | 0.18% |
| copyright holder | 3 | 0.18% |
| qualifies for event | 3 | 0.18% |
| librettist | 3 | 0.18% |
| incarnation of | 3 | 0.18% |
| has part(s) of the class | 3 | 0.18% |
| manufacturer | 3 | 0.18% |
| parent of this hybrid, breed, or cultivar | 3 | 0.18% |
| online service | 3 | 0.18% |
| commemorates | 3 | 0.18% |
| screenwriter | 3 | 0.18% |
| facet of | 3 | 0.18% |
| highest point | 3 | 0.18% |
| house publication | 3 | 0.18% |
| heritage designation | 3 | 0.18% |
| religious order | 2 | 0.12% |
| executive body | 2 | 0.12% |
| family name identical to this given name | 2 | 0.12% |
| industry | 2 | 0.12% |
| vehicle normally used | 2 | 0.12% |
| religion or worldview | 2 | 0.12% |
| made from material | 2 | 0.12% |
| minor planet group | 2 | 0.12% |
| armament | 2 | 0.12% |
| legal form | 2 | 0.12% |
| constellation | 2 | 0.12% |
| ancestral home | 2 | 0.12% |
| place of publication | 2 | 0.12% |
| territory overlaps | 2 | 0.12% |
| office held by head of the organization | 2 | 0.12% |
| contributing factor of | 2 | 0.12% |
| musical conductor | 2 | 0.12% |
| practiced by | 2 | 0.12% |
| has cause | 2 | 0.12% |
| central bank/issuer | 2 | 0.12% |
| domain of saint or deity | 2 | 0.12% |
| participant | 2 | 0.12% |
| edition or translation of | 2 | 0.12% |
| part of | 2 | 0.12% |
| overlies | 2 | 0.12% |
| from narrative universe | 2 | 0.12% |
| recommended unit of measurement | 2 | 0.12% |
| has quality | 2 | 0.12% |
| political ideology | 2 | 0.12% |
| choreographer | 2 | 0.12% |
| programming language | 2 | 0.12% |
| developer | 2 | 0.12% |
| input method | 2 | 0.12% |
| place of birth | 2 | 0.12% |
| has written for | 2 | 0.12% |
| league level above | 2 | 0.12% |
| significant place | 2 | 0.12% |
| product or material produced | 2 | 0.12% |
| advertises | 2 | 0.12% |
| cause of death | 2 | 0.12% |
| significant person | 2 | 0.12% |
| coextensive with | 2 | 0.12% |
| reservoir created | 2 | 0.12% |
| canonization status | 2 | 0.12% |
| league system | 2 | 0.12% |
| terminus location | 2 | 0.12% |
| distribution format | 2 | 0.12% |
| brand | 2 | 0.12% |
| depicts | 2 | 0.12% |
| described by source | 2 | 0.12% |
| party chief representative | 2 | 0.12% |
| victory | 2 | 0.12% |
| engine configuration | 2 | 0.12% |
| successful candidate | 2 | 0.12% |
| language used | 2 | 0.12% |
| replaced by | 2 | 0.12% |
| school district | 2 | 0.12% |
| foundational text | 2 | 0.12% |
| illustrator | 2 | 0.12% |
| contains | 2 | 0.12% |
| lake on watercourse | 2 | 0.12% |
| student | 2 | 0.12% |
| narrative location | 2 | 0.12% |
| sibling | 2 | 0.12% |
| collection | 2 | 0.12% |
| editor-in-chief | 2 | 0.12% |
| implementation of | 2 | 0.12% |
| sports season of league or competition | 2 | 0.12% |
| killed by | 2 | 0.12% |
| honorific prefix | 2 | 0.12% |
| lyrics by | 2 | 0.12% |
| ammunition | 2 | 0.12% |
| partially coincident with | 2 | 0.12% |
| currency | 2 | 0.12% |
| has edition or translation | 2 | 0.12% |
| subclass of | 2 | 0.12% |
| mountains classification | 2 | 0.12% |
| measured physical quantity | 2 | 0.12% |
| anatomical location | 2 | 0.12% |
| solved by | 2 | 0.12% |
| residence | 2 | 0.12% |
| anatomical branch of | 2 | 0.12% |
| director of archaeological fieldwork | 2 | 0.12% |
| day in year for periodic occurrence | 2 | 0.12% |
| mount | 2 | 0.12% |
| depicted by | 2 | 0.12% |
| next higher rank | 2 | 0.12% |
| character designer | 2 | 0.12% |
| defendant | 2 | 0.12% |
| architectural style | 2 | 0.12% |
| political alignment | 2 | 0.12% |
| writing system | 2 | 0.12% |
| afflicts | 2 | 0.12% |
| head of government | 2 | 0.12% |
| place of death | 2 | 0.12% |
| takes place in fictional universe | 2 | 0.12% |
| appears in the form of | 1 | 0.06% |
| tributary | 1 | 0.06% |
| main regulatory text | 1 | 0.06% |
| product certification | 1 | 0.06% |
| interaction | 1 | 0.06% |
| official podcast | 1 | 0.06% |
| catalog | 1 | 0.06% |
| relative | 1 | 0.06% |
| honorific suffix | 1 | 0.06% |
| soundtrack release | 1 | 0.06% |
| named by | 1 | 0.06% |
| type of variable star | 1 | 0.06% |
| antiparticle | 1 | 0.06% |
| drafted by | 1 | 0.06% |
| interested in | 1 | 0.06% |
| office held by head of state | 1 | 0.06% |
| this taxon is source of | 1 | 0.06% |
| statistical leader | 1 | 0.06% |
| title of chess person | 1 | 0.06% |
| contains the administrative territorial entity | 1 | 0.06% |
| route of administration | 1 | 0.06% |
| derivative work | 1 | 0.06% |
| translator | 1 | 0.06% |
| type of orbit | 1 | 0.06% |
| family name | 1 | 0.06% |
| connector | 1 | 0.06% |
| stepparent | 1 | 0.06% |
| twinned administrative body | 1 | 0.06% |
| place of burial | 1 | 0.06% |
| found in taxon | 1 | 0.06% |
| first aid measures | 1 | 0.06% |
| studied by | 1 | 0.06% |
| consecrator | 1 | 0.06% |
| anthem | 1 | 0.06% |
| based on | 1 | 0.06% |
| lifestyle | 1 | 0.06% |
| has facility | 1 | 0.06% |
| statement describes | 1 | 0.06% |
| filming location | 1 | 0.06% |
| doctoral advisor | 1 | 0.06% |
| in opposition to | 1 | 0.06% |
| permanent resident of | 1 | 0.06% |
| public holiday | 1 | 0.06% |
| aspect ratio (W:H) | 1 | 0.06% |
| discography | 1 | 0.06% |
| type foundry | 1 | 0.06% |
| flag | 1 | 0.06% |
| languages spoken, written or signed | 1 | 0.06% |
| superhuman feature or ability | 1 | 0.06% |
| speaker | 1 | 0.06% |
| offers view on | 1 | 0.06% |
| parallel street | 1 | 0.06% |
| scheduled service destination | 1 | 0.06% |
| does not have part | 1 | 0.06% |
| academic degree | 1 | 0.06% |
| structure replaced by | 1 | 0.06% |
| general manager | 1 | 0.06% |
| performed at | 1 | 0.06% |
| designed to carry | 1 | 0.06% |
| office contested | 1 | 0.06% |
| wheel arrangement | 1 | 0.06% |
| designed by | 1 | 0.06% |
| regulated by | 1 | 0.06% |
| destination point | 1 | 0.06% |
| style of karate | 1 | 0.06% |
| copyright representative | 1 | 0.06% |
| Christian liturgical rite | 1 | 0.06% |
| IUCN conservation status | 1 | 0.06% |
| general classification of race participants | 1 | 0.06% |
| chief executive officer | 1 | 0.06% |
| foods traditionally associated | 1 | 0.06% |
| published in | 1 | 0.06% |
| color | 1 | 0.06% |
| source of material | 1 | 0.06% |
| instrumentation | 1 | 0.06% |
| launch contractor | 1 | 0.06% |
| set during recurring event | 1 | 0.06% |
| typically sells | 1 | 0.06% |
| music created for | 1 | 0.06% |
| instruction set | 1 | 0.06% |
| newspaper format | 1 | 0.06% |
| location of creation | 1 | 0.06% |
| cell component | 1 | 0.06% |
| source of energy | 1 | 0.06% |
| sexual orientation | 1 | 0.06% |
| competition class | 1 | 0.06% |
| inspired by | 1 | 0.06% |
| follows | 1 | 0.06% |
| uses capitalization for | 1 | 0.06% |
| Gram staining | 1 | 0.06% |
| significant drug interaction | 1 | 0.06% |
| has boundary | 1 | 0.06% |
| central bank | 1 | 0.06% |
| geography of topic | 1 | 0.06% |
| court | 1 | 0.06% |
| site of astronomical discovery | 1 | 0.06% |
| trophy awarded | 1 | 0.06% |
| parent astronomical body | 1 | 0.06% |
| native language | 1 | 0.06% |
| league level below | 1 | 0.06% |
| office held by head of government | 1 | 0.06% |
| expressed in | 1 | 0.06% |
| genetic association | 1 | 0.06% |
| maintained by | 1 | 0.06% |
| enclave within | 1 | 0.06% |
| doctoral student | 1 | 0.06% |
| gens | 1 | 0.06% |
| madhhab | 1 | 0.06% |
| charge | 1 | 0.06% |
| organizer | 1 | 0.06% |
| authority | 1 | 0.06% |
| voice type | 1 | 0.06% |
| coach of sports team | 1 | 0.06% |
| has phenotype | 1 | 0.06% |
| connecting line | 1 | 0.06% |
| possible treatment | 1 | 0.06% |
| set in period | 1 | 0.06% |
| associated electoral district | 1 | 0.06% |
| less than | 1 | 0.06% |
| official symbol | 1 | 0.06% |
| decays to | 1 | 0.06% |
| readable file format | 1 | 0.06% |
| final event | 1 | 0.06% |
| season | 1 | 0.06% |
| enemy of | 1 | 0.06% |
| military casualty classification | 1 | 0.06% |
| molecular function | 1 | 0.06% |
| unmarried partner | 1 | 0.06% |
| language regulatory body | 1 | 0.06% |
| director of photography | 1 | 0.06% |
| biological process | 1 | 0.06% |
| executive producer | 1 | 0.06% |
| copyright license | 1 | 0.06% |
| official language | 1 | 0.06% |
| narrative role | 1 | 0.06% |
| chassis | 1 | 0.06% |
| legislative body | 1 | 0.06% |
| genic_interaction | 1 | 0.06% |
| produced by | 1 | 0.06% |
| manner of death | 1 | 0.06% |

#### val

| label | count | share |
| --- | --- | --- |
| Association | 5 | 9.43% |
| taxon rank | 1 | 1.89% |
| replaces | 1 | 1.89% |
| publisher | 1 | 1.89% |
| commemorates | 1 | 1.89% |
| title of chess person | 1 | 1.89% |
| derivative work | 1 | 1.89% |
| military branch | 1 | 1.89% |
| drug or therapy used for treatment | 1 | 1.89% |
| discography | 1 | 1.89% |
| office held by head of state | 1 | 1.89% |
| instrument | 1 | 1.89% |
| based on | 1 | 1.89% |
| taxonomic type | 1 | 1.89% |
| platform | 1 | 1.89% |
| occupation | 1 | 1.89% |
| location of landing | 1 | 1.89% |
| astronaut mission | 1 | 1.89% |
| discoverer or inventor | 1 | 1.89% |
| product or material produced | 1 | 1.89% |
| affiliation | 1 | 1.89% |
| partner in business or sport | 1 | 1.89% |
| endemic to | 1 | 1.89% |
| head of government | 1 | 1.89% |
| movement | 1 | 1.89% |
| crew member(s) | 1 | 1.89% |
| location of formation | 1 | 1.89% |
| notable work | 1 | 1.89% |
| director / manager | 1 | 1.89% |
| member of political party | 1 | 1.89% |
| voice type | 1 | 1.89% |
| child | 1 | 1.89% |
| fabrication method | 1 | 1.89% |
| place of detention | 1 | 1.89% |
| father | 1 | 1.89% |
| distribution format | 1 | 1.89% |
| diocese | 1 | 1.89% |
| Positive_Correlation | 1 | 1.89% |
| Bind | 1 | 1.89% |
| PPI | 1 | 1.89% |
| head of state | 1 | 1.89% |
| basin country | 1 | 1.89% |
| instance of | 1 | 1.89% |
| official language | 1 | 1.89% |
| secretary general | 1 | 1.89% |
| honorific suffix | 1 | 1.89% |
| location of first performance | 1 | 1.89% |
| decay mode | 1 | 1.89% |
| presenter | 1 | 1.89% |

#### test

| label | count | share |
| --- | --- | --- |
| bind | 7 | 10.94% |
| track gauge | 2 | 3.12% |
| maintained by | 2 | 3.12% |
| terminus location | 2 | 3.12% |
| ammunition | 1 | 1.56% |
| amended by | 1 | 1.56% |
| wears | 1 | 1.56% |
| studied by | 1 | 1.56% |
| board member | 1 | 1.56% |
| newspaper format | 1 | 1.56% |
| cause of destruction | 1 | 1.56% |
| member of sports team | 1 | 1.56% |
| surface played on | 1 | 1.56% |
| manner of death | 1 | 1.56% |
| crosses | 1 | 1.56% |
| commissioned by | 1 | 1.56% |
| via | 1 | 1.56% |
| CPU | 1 | 1.56% |
| lake outflow | 1 | 1.56% |
| victory | 1 | 1.56% |
| owned by | 1 | 1.56% |
| enemy of | 1 | 1.56% |
| has active ingredient | 1 | 1.56% |
| authority | 1 | 1.56% |
| has immediate cause | 1 | 1.56% |
| biological process | 1 | 1.56% |
| different from | 1 | 1.56% |
| astronaut mission | 1 | 1.56% |
| item operated | 1 | 1.56% |
| legislative body | 1 | 1.56% |
| product certification | 1 | 1.56% |
| father | 1 | 1.56% |
| contains the administrative territorial entity | 1 | 1.56% |
| residence | 1 | 1.56% |
| relative | 1 | 1.56% |
| located in the statistical territorial entity | 1 | 1.56% |
| located in or next to body of water | 1 | 1.56% |
| narrator | 1 | 1.56% |
| genre | 1 | 1.56% |
| legislated by | 1 | 1.56% |
| editor | 1 | 1.56% |
| significant event | 1 | 1.56% |
| represented by | 1 | 1.56% |
| league | 1 | 1.56% |
| child | 1 | 1.56% |
| chairperson | 1 | 1.56% |
| director | 1 | 1.56% |
| brand | 1 | 1.56% |
| office held by head of state | 1 | 1.56% |
| found in taxon | 1 | 1.56% |
| located in/on physical feature | 1 | 1.56% |
| present in work | 1 | 1.56% |
| killed by | 1 | 1.56% |
| performer | 1 | 1.56% |
| narrative location | 1 | 1.56% |

### Classifications

**train, biotech_event.** Majority class 30.3% of 152 answers. The marginal is `true_label`, not the menu.

#### train biotech_event

| label | count | share |
| --- | --- | --- |
| executive statement | 46 | 30.26% |
| company description | 22 | 14.47% |
| alliance & partnership | 10 | 6.58% |
| other | 9 | 5.92% |
| m&a | 9 | 5.92% |
| service & product providing | 7 | 4.61% |
| funding round | 7 | 4.61% |
| expanding geography | 6 | 3.95% |
| new initiatives or programs | 6 | 3.95% |
| product launching & presentation | 5 | 3.29% |
| support & philanthropy | 4 | 2.63% |
| regulatory approval | 4 | 2.63% |
| product updates | 3 | 1.97% |
| event organization | 2 | 1.32% |
| closing | 2 | 1.32% |
| participation in an event | 2 | 1.32% |
| hiring | 2 | 1.32% |
| article publication | 2 | 1.32% |
| clinical trial sponsorship | 1 | 0.66% |
| ipo exit | 1 | 0.66% |
| new initiatives & programs | 1 | 0.66% |
| foundation | 1 | 0.66% |

**train, chfinann_event.** Majority class 49.3% of 994 answers. The marginal is `true_label`, not the menu.

#### train chfinann_event

| label | count | share |
| --- | --- | --- |
| EquityPledge | 490 | 49.30% |
| EquityOverweight | 214 | 21.53% |
| EquityUnderweight | 194 | 19.52% |
| EquityRepurchase | 58 | 5.84% |
| EquityFreeze | 38 | 3.82% |

**train, docee_event.** Majority class 8.7% of 848 answers. The marginal is `true_label`, not the menu.

#### train docee_event

| label | count | share |
| --- | --- | --- |
| Armed Conflict | 74 | 8.73% |
| Air Crash | 43 | 5.07% |
| Protest | 39 | 4.60% |
| Election | 36 | 4.25% |
| CommitCrime - Sentence | 34 | 4.01% |
| Fire | 33 | 3.89% |
| Resignation_Dismissal | 29 | 3.42% |
| CommitCrime - Accuse | 28 | 3.30% |
| Government Policy Changes | 28 | 3.30% |
| Sports Competition | 25 | 2.95% |
| Diplomatic Talks | 19 | 2.24% |
| Famous Person - Death | 17 | 2.00% |
| Mass Poisoning | 17 | 2.00% |
| Withdraw from an Orgnization | 17 | 2.00% |
| Earthquakes | 17 | 2.00% |
| Riot | 15 | 1.77% |
| Appoint_Inauguration | 15 | 1.77% |
| Storm | 15 | 1.77% |
| Environment Pollution | 14 | 1.65% |
| Diplomatic Visit | 14 | 1.65% |
| Famine | 14 | 1.65% |
| CommitCrime - Release | 13 | 1.53% |
| Gas Explosion | 13 | 1.53% |
| Famous Person - Give a Speech | 13 | 1.53% |
| Strike | 13 | 1.53% |
| Break Historical Records | 12 | 1.42% |
| Famous Person - Sick | 12 | 1.42% |
| Awards Ceremony | 12 | 1.42% |
| Shipwreck | 11 | 1.30% |
| CommitCrime - Arrest | 11 | 1.30% |
| Train Collisions | 11 | 1.30% |
| Road Crash | 10 | 1.18% |
| Disease Outbreaks | 10 | 1.18% |
| New Achievements in Aerospace | 10 | 1.18% |
| New Archeological Discoveries | 10 | 1.18% |
| Financial Crisis | 10 | 1.18% |
| Floods | 9 | 1.06% |
| Organization Merge | 9 | 1.06% |
| Organization Closed | 9 | 1.06% |
| Mine Collapses | 9 | 1.06% |
| Famous Person - Marriage | 7 | 0.83% |
| Bank Robbery | 7 | 0.83% |
| Organization Fine | 7 | 0.83% |
| CommitCrime - Investigate | 7 | 0.83% |
| Volcano Eruption | 7 | 0.83% |
| Insect Disaster | 6 | 0.71% |
| Mudslides | 6 | 0.71% |
| New Wonders in Nature | 6 | 0.71% |
| Famous Person - Divorce | 6 | 0.71% |
| Tsunamis | 6 | 0.71% |
| Financial Aid | 5 | 0.59% |
| Tear Up Agreement | 5 | 0.59% |
| Famous Person - Recovered | 5 | 0.59% |
| Military Excercise | 4 | 0.47% |
| Regime Change | 4 | 0.47% |
| Sign Agreement | 4 | 0.47% |
| Droughts | 2 | 0.24% |
| Join in an Orgnization | 2 | 0.24% |
| Organization Established | 2 | 0.24% |

**train, docfee_event.** Majority class 32.3% of 657 answers. The marginal is `true_label`, not the menu.

#### train docfee_event

| label | count | share |
| --- | --- | --- |
| EquityUnderweight | 212 | 32.27% |
| EquityPledge | 104 | 15.83% |
| BankruptcyLiquidation | 104 | 15.83% |
| MajorAssetLoss | 97 | 14.76% |
| EquityFreeze | 42 | 6.39% |
| EquityOverweight | 33 | 5.02% |
| MajorSafetyAccident | 29 | 4.41% |
| MajorExternalPayout | 18 | 2.74% |
| ExecutiveDeath | 18 | 2.74% |

**val, biotech_event.** Majority class 37.5% of 8 answers. The marginal is `true_label`, not the menu.

#### val biotech_event

| label | count | share |
| --- | --- | --- |
| executive statement | 3 | 37.50% |
| new initiatives or programs | 1 | 12.50% |
| company description | 1 | 12.50% |
| support & philanthropy | 1 | 12.50% |
| other | 1 | 12.50% |
| product launching & presentation | 1 | 12.50% |

**val, chfinann_event.** Majority class 56.1% of 41 answers. The marginal is `true_label`, not the menu.

#### val chfinann_event

| label | count | share |
| --- | --- | --- |
| EquityPledge | 23 | 56.10% |
| EquityUnderweight | 11 | 26.83% |
| EquityOverweight | 6 | 14.63% |
| EquityFreeze | 1 | 2.44% |

**val, docee_event.** Majority class 13.3% of 30 answers. The marginal is `true_label`, not the menu.

#### val docee_event

| label | count | share |
| --- | --- | --- |
| CommitCrime - Sentence | 4 | 13.33% |
| Break Historical Records | 3 | 10.00% |
| Government Policy Changes | 2 | 6.67% |
| CommitCrime - Investigate | 2 | 6.67% |
| Air Crash | 2 | 6.67% |
| Armed Conflict | 2 | 6.67% |
| New Achievements in Aerospace | 1 | 3.33% |
| Organization Established | 1 | 3.33% |
| Mine Collapses | 1 | 3.33% |
| Famine | 1 | 3.33% |
| Riot | 1 | 3.33% |
| Disease Outbreaks | 1 | 3.33% |
| New Wonders in Nature | 1 | 3.33% |
| Diplomatic Talks | 1 | 3.33% |
| Resignation_Dismissal | 1 | 3.33% |
| Storm | 1 | 3.33% |
| Droughts | 1 | 3.33% |
| Train Collisions | 1 | 3.33% |
| Road Crash | 1 | 3.33% |
| New Archeological Discoveries | 1 | 3.33% |
| Strike | 1 | 3.33% |

**val, docfee_event.** Majority class 42.3% of 26 answers. The marginal is `true_label`, not the menu.

#### val docfee_event

| label | count | share |
| --- | --- | --- |
| EquityUnderweight | 11 | 42.31% |
| BankruptcyLiquidation | 4 | 15.38% |
| EquityPledge | 4 | 15.38% |
| EquityOverweight | 2 | 7.69% |
| EquityFreeze | 2 | 7.69% |
| MajorAssetLoss | 2 | 7.69% |
| MajorSafetyAccident | 1 | 3.85% |

**test, biotech_event.** Majority class 18.2% of 11 answers. The marginal is `true_label`, not the menu.

#### test biotech_event

| label | count | share |
| --- | --- | --- |
| m&a | 2 | 18.18% |
| executive statement | 2 | 18.18% |
| funding round | 1 | 9.09% |
| executive appointment | 1 | 9.09% |
| company description | 1 | 9.09% |
| new initiatives or programs | 1 | 9.09% |
| expanding industry | 1 | 9.09% |
| regulatory approval | 1 | 9.09% |
| product launching & presentation | 1 | 9.09% |

**test, chfinann_event.** Majority class 61.9% of 42 answers. The marginal is `true_label`, not the menu.

#### test chfinann_event

| label | count | share |
| --- | --- | --- |
| EquityPledge | 26 | 61.90% |
| EquityOverweight | 6 | 14.29% |
| EquityUnderweight | 5 | 11.90% |
| EquityRepurchase | 4 | 9.52% |
| EquityFreeze | 1 | 2.38% |

**test, docee_event.** Majority class 12.9% of 31 answers. The marginal is `true_label`, not the menu.

#### test docee_event

| label | count | share |
| --- | --- | --- |
| Air Crash | 4 | 12.90% |
| Armed Conflict | 3 | 9.68% |
| Insect Disaster | 3 | 9.68% |
| Famous Person - Sick | 2 | 6.45% |
| CommitCrime - Accuse | 2 | 6.45% |
| Protest | 2 | 6.45% |
| Election | 2 | 6.45% |
| Famous Person - Divorce | 1 | 3.23% |
| Riot | 1 | 3.23% |
| Train Collisions | 1 | 3.23% |
| Military Excercise | 1 | 3.23% |
| Financial Crisis | 1 | 3.23% |
| Strike | 1 | 3.23% |
| Road Crash | 1 | 3.23% |
| Break Historical Records | 1 | 3.23% |
| Bank Robbery | 1 | 3.23% |
| Awards Ceremony | 1 | 3.23% |
| New Achievements in Aerospace | 1 | 3.23% |
| New Wonders in Nature | 1 | 3.23% |
| Earthquakes | 1 | 3.23% |

**test, docfee_event.** Majority class 32.1% of 28 answers. The marginal is `true_label`, not the menu.

#### test docfee_event

| label | count | share |
| --- | --- | --- |
| EquityUnderweight | 9 | 32.14% |
| EquityPledge | 6 | 21.43% |
| BankruptcyLiquidation | 5 | 17.86% |
| MajorAssetLoss | 4 | 14.29% |
| EquityOverweight | 2 | 7.14% |
| MajorSafetyAccident | 1 | 3.57% |
| ExecutiveDeath | 1 | 3.57% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| casualty_report | location | 28,571 | 92.95% |
| casualty_report | injured | 14,352 | 46.69% |
| casualty_report | missing | 12,402 | 40.35% |
| casualty_report | dead | 12,226 | 39.77% |
| casualty_report | record_metadata | natural/dead ×8,439, natural/injured ×2,744, natural/missing ×863, natural/location ×380 |  |
| record | status | 46 | 15.18% |
| record | location | 32 | 10.56% |
| record | name | 22 | 7.26% |
| record | title | 13 | 4.29% |
| record | address | 13 | 4.29% |
| record | description | 13 | 4.29% |
| record | employee_id | 8 | 2.64% |
| record | date | 7 | 2.31% |
| record | rating | 6 | 1.98% |
| record | product_id | 6 | 1.98% |
| record | amount | 6 | 1.98% |
| record | claim_id | 6 | 1.98% |
| record | product_name | 5 | 1.65% |
| record | score | 5 | 1.65% |
| record | year | 5 | 1.65% |
| record | price | 4 | 1.32% |
| record | patient_id | 4 | 1.32% |
| record | funding_amount | 4 | 1.32% |
| record | age | 4 | 1.32% |
| record | city | 4 | 1.32% |
| record | author | 4 | 1.32% |
| record | transaction_id | 4 | 1.32% |
| record | tags | 4 | 1.32% |
| record | value | 4 | 1.32% |
| record | likes | 4 | 1.32% |
| record | country | 4 | 1.32% |
| record | region | 4 | 1.32% |
| record | report_id | 4 | 1.32% |
| record | features | 3 | 0.99% |
| record | dosage | 3 | 0.99% |
| record | institution | 3 | 0.99% |
| record | review_id | 3 | 0.99% |
| record | venue | 3 | 0.99% |
| record | symptoms | 3 | 0.99% |
| record | diagnosis_code | 3 | 0.99% |
| record | funding_source | 3 | 0.99% |
| record | amenities | 3 | 0.99% |
| record | match | 3 | 0.99% |
| record | video_id | 3 | 0.99% |
| record | company | 3 | 0.99% |
| record | normal_range | 3 | 0.99% |
| record | manufacturer | 3 | 0.99% |
| record | patient_name | 3 | 0.99% |
| record | target_audience | 3 | 0.99% |
| record | project_name | 3 | 0.99% |
| record | role | 3 | 0.99% |
| record | reviewer | 2 | 0.66% |
| record | hotel_name | 2 | 0.66% |
| record | user | 2 | 0.66% |
| record | incident_id | 2 | 0.66% |
| record | asteroid.discovery.date | 2 | 0.66% |
| record | vaccine_type | 2 | 0.66% |
| record | administration_date | 2 | 0.66% |
| record | state | 2 | 0.66% |
| record | resolution_time | 2 | 0.66% |
| record | issues | 2 | 0.66% |
| record | key_features | 2 | 0.66% |
| record | insurance | 2 | 0.66% |
| record | treatment | 2 | 0.66% |
| record | policy_number | 2 | 0.66% |
| record | visit_type | 2 | 0.66% |
| record | education | 2 | 0.66% |
| record | company_id | 2 | 0.66% |
| record | rate | 2 | 0.66% |
| record | winner | 2 | 0.66% |
| record | creator | 2 | 0.66% |
| record | thumbnail | 2 | 0.66% |
| record | brand | 2 | 0.66% |
| record | lab_id | 2 | 0.66% |
| record | severity | 2 | 0.66% |
| record | comments | 2 | 0.66% |
| record | shares | 2 | 0.66% |
| record | dose_number | 2 | 0.66% |
| record | email | 2 | 0.66% |
| record | product_category | 2 | 0.66% |
| record | product_name: Name of the product being used | 2 | 0.66% |
| record | company: Organization where the product is used | 2 | 0.66% |
| record | user: Name of the team member or user using the product | 2 | 0.66% |
| record | revenue | 2 | 0.66% |
| record | recipient | 2 | 0.66% |
| record | job_title | 2 | 0.66% |
| record | salary_range | 2 | 0.66% |
| record | review_date | 2 | 0.66% |
| record | locations | 2 | 0.66% |
| record | enrollment | 2 | 0.66% |
| record | asteroid.designation_number | 2 | 0.66% |
| record | attending_physician | 2 | 0.66% |
| record | location.airport_travel_time.value | 2 | 0.66% |
| record | hotel_details.recreation | 2 | 0.66% |
| record | location.city | 2 | 0.66% |
| record | hotel_details.room_amenities | 2 | 0.66% |
| record | booking_policy.multiple_rooms | 2 | 0.66% |
| record | hotel_details.year_built | 2 | 0.66% |
| record | child_policy.infants.age_range | 2 | 0.66% |
| record | child_policy.adult_age | 2 | 0.66% |
| record | location.nearby_attractions | 2 | 0.66% |
| record | location.distance_from_airport.value | 2 | 0.66% |
| record | initial_symptoms | 2 | 0.66% |
| record | university | 2 | 0.66% |
| record | species.scientific_name | 2 | 0.66% |
| record | species.genus | 2 | 0.66% |
| record | species.description | 2 | 0.66% |
| record | attendee_count | 2 | 0.66% |
| record | invoice_id | 2 | 0.66% |
| record | policyholder | 2 | 0.66% |
| record | sender | 2 | 0.66% |
| record | year_in_era | 2 | 0.66% |
| record | organization | 2 | 0.66% |
| record | department | 2 | 0.66% |
| record | organizations | 2 | 0.66% |
| record | dateofservice | 2 | 0.66% |
| record | full_name | 2 | 0.66% |
| record | category | 2 | 0.66% |
| record | description: Brief text describing the content of the post | 2 | 0.66% |
| record | status: Current status of the trend repost | 2 | 0.66% |
| record | shares: Number of shares the post received | 2 | 0.66% |
| record | user_name: Name of the user who created the repost | 2 | 0.66% |
| record | trend_name: Name of the viral trend being reposted | 2 | 0.66% |
| record | municipality.name | 2 | 0.66% |
| record | ticket_id | 2 | 0.66% |
| record | headquarters: City and country where company is headquartered | 2 | 0.66% |
| record | status: Current operational or strategic status | 2 | 0.66% |
| record | claim_amount | 2 | 0.66% |
| record | material | 2 | 0.66% |
| record | submitter | 2 | 0.66% |
| record | company_name | 2 | 0.66% |
| record | company_name: Name of the startup founded by the founder | 1 | 0.33% |
| record | equity_stake: Founder's ownership percentage in the company | 1 | 0.33% |
| record | location: City and state where the startup is based | 1 | 0.33% |
| record | founding_year: Year the startup was established | 1 | 0.33% |
| record | copyright | 1 | 0.33% |
| record | date_reported | 1 | 0.33% |
| record | environment | 1 | 0.33% |
| record | battery_life | 1 | 0.33% |
| record | perpetrator.arrest_details.time | 1 | 0.33% |
| record | victim.relationship_to_perpetrator | 1 | 0.33% |
| record | crime.location.landmark | 1 | 0.33% |
| record | crime.motive | 1 | 0.33% |
| record | crime.weapon.source.stolen_from.name | 1 | 0.33% |
| record | investigation.status | 1 | 0.33% |
| record | victim.injury | 1 | 0.33% |
| record | perpetrator.legal_actions | 1 | 0.33% |
| record | perpetrator.name | 1 | 0.33% |
| record | victim.name | 1 | 0.33% |
| record | crime.weapon.model | 1 | 0.33% |
| record | address: Full street address including suite and city | 1 | 0.33% |
| record | state: State where the office is located | 1 | 0.33% |
| record | office_code: Unique internal identifier for the office location | 1 | 0.33% |
| record | city: City where the office is physically located | 1 | 0.33% |
| record | name: Official name of the business unit operating the office | 1 | 0.33% |
| record | seller | 1 | 0.33% |
| record | date_added | 1 | 0.33% |
| record | main_competitor | 1 | 0.33% |
| record | format | 1 | 0.33% |
| record | bracket_size | 1 | 0.33% |
| record | prescriber | 1 | 0.33% |
| record | fda_code | 1 | 0.33% |
| record | medication | 1 | 0.33% |
| record | research_project | 1 | 0.33% |
| record | principal_investigator | 1 | 0.33% |
| record | project_status | 1 | 0.33% |
| record | asteroid.designation | 1 | 0.33% |
| record | asteroid.discovery.year | 1 | 0.33% |
| record | item_description.shipping.ready_to_ship_in.quantity | 1 | 0.33% |
| record | item_description.shipping.rush_details | 1 | 0.33% |
| record | item_description.weight.unit | 1 | 0.33% |
| record | item_description.dimensions.length | 1 | 0.33% |
| record | item_description.setup_charge.per | 1 | 0.33% |
| record | item_description.setup_charge.amount | 1 | 0.33% |
| record | item_description.weight.value | 1 | 0.33% |
| record | faq_references | 1 | 0.33% |
| record | navigation | 1 | 0.33% |
| record | footer_links | 1 | 0.33% |
| record | browse_topics | 1 | 0.33% |
| record | customer_id | 1 | 0.33% |
| record | customer_name | 1 | 0.33% |
| record | ticket_count | 1 | 0.33% |
| record | open_date | 1 | 0.33% |
| record | assigned_agent | 1 | 0.33% |
| record | trial_title | 1 | 0.33% |
| record | model | 1 | 0.33% |
| record | supplier | 1 | 0.33% |
| record | serial_number | 1 | 0.33% |
| record | invoice_number | 1 | 0.33% |
| record | home_team | 1 | 0.33% |
| record | final_score | 1 | 0.33% |
| record | match_code | 1 | 0.33% |
| record | discovery_year | 1 | 0.33% |
| record | dedication.name | 1 | 0.33% |
| record | funding: Total funding amount in millions | 1 | 0.33% |
| record | project_name: Name of the technology project | 1 | 0.33% |
| record | location: Geographic location of the project | 1 | 0.33% |
| record | project_id: Unique identifier assigned to the project | 1 | 0.33% |
| record | founding_year | 1 | 0.33% |
| record | key_challenge | 1 | 0.33% |
| record | outcome | 1 | 0.33% |
| record | client_name | 1 | 0.33% |
| record | client_address | 1 | 0.33% |
| record | tournament_code | 1 | 0.33% |
| record | loser | 1 | 0.33% |
| record | aces | 1 | 0.33% |
| record | break_points | 1 | 0.33% |
| record | error.possible_causes | 1 | 0.33% |
| record | error.message | 1 | 0.33% |
| record | recipient_name | 1 | 0.33% |
| record | award_name | 1 | 0.33% |
| record | fieldofstudy | 1 | 0.33% |
| record | recommendations | 1 | 0.33% |
| record | statement_reference: Identifier code for the statement document | 1 | 0.33% |
| record | amount: Financial value associated with the policy or proposal | 1 | 0.33% |
| record | location: City and state where the event occurred | 1 | 0.33% |
| record | topic: Subject matter discussed in the statement | 1 | 0.33% |
| record | speaker: Name of the person delivering the statement | 1 | 0.33% |
| record | patent_number | 1 | 0.33% |
| record | zip | 1 | 0.33% |
| record | sender_domain: Domain of the spoofed email address | 1 | 0.33% |
| record | status: Current status of the report | 1 | 0.33% |
| record | personal_experience.raw_corn_experience | 1 | 0.33% |
| record | serving_suggestions | 1 | 0.33% |
| record | boiling_instructions.cooking_time.max_minutes | 1 | 0.33% |
| record | cooking_methods | 1 | 0.33% |
| record | ingredients | 1 | 0.33% |
| record | boiling_instructions.salt_advice | 1 | 0.33% |
| record | grilling_instructions.cooking_time.minutes | 1 | 0.33% |
| record | patient_id: Unique identifier for the patient | 1 | 0.33% |
| record | encounter_id: Unique identifier for the visit | 1 | 0.33% |
| record | name: Full name of the patient | 1 | 0.33% |
| record | address: Patient's residential address | 1 | 0.33% |
| record | visit_date: Date of the medical encounter | 1 | 0.33% |
| record | primary_diagnosis: Main medical condition diagnosed | 1 | 0.33% |
| record | icd_code: ICD-10 code for the primary diagnosis | 1 | 0.33% |
| record | bp_systolic: Systolic blood pressure in mmHg | 1 | 0.33% |
| record | bp_diastolic: Diastolic blood pressure in mmHg | 1 | 0.33% |
| record | heart_rate: Heart rate in beats per minute | 1 | 0.33% |
| record | temperature: Body temperature in degrees Fahrenheit | 1 | 0.33% |
| record | medications: List of prescribed medications with dosing | 1 | 0.33% |
| record | lab_results: List of lab test results | 1 | 0.33% |
| record | symptoms: Patient-reported symptoms or complaints | 1 | 0.33% |
| record | unit_id | 1 | 0.33% |
| record | amount: Monetary value of the returned item | 1 | 0.33% |
| record | return_date: Date when the return was submitted | 1 | 0.33% |
| record | product_name: Name of the product returned | 1 | 0.33% |
| record | order_id: Identifier of the original order associated with the return | 1 | 0.33% |
| record | question.text | 1 | 0.33% |
| record | question.keywords | 1 | 0.33% |
| record | question.legal_implications | 1 | 0.33% |
| record | details.date_posted | 1 | 0.33% |
| record | details.days_ago | 1 | 0.33% |
| record | schema_name | 1 | 0.33% |
| record | post_id | 1 | 0.33% |
| record | upvotes | 1 | 0.33% |
| record | subreddit | 1 | 0.33% |
| record | bedrooms | 1 | 0.33% |
| record | technique_name | 1 | 0.33% |
| record | abbreviation | 1 | 0.33% |
| record | cohort_size | 1 | 0.33% |
| record | specific_outcomes.postoperative_cholangiography_rate | 1 | 0.33% |
| record | study_id | 1 | 0.33% |
| record | authors | 1 | 0.33% |
| record | affiliations | 1 | 0.33% |
| record | study_start | 1 | 0.33% |
| record | study_end | 1 | 0.33% |
| record | sample_size | 1 | 0.33% |
| record | primary_outcome | 1 | 0.33% |
| record | key_result | 1 | 0.33% |
| record | budget_remaining | 1 | 0.33% |
| record | actual_spend | 1 | 0.33% |
| record | remaining_budget | 1 | 0.33% |
| record | onboarding_sprint_completion | 1 | 0.33% |
| record | risk_code | 1 | 0.33% |
| record | backend_integration_status | 1 | 0.33% |
| record | date: Date the match was scheduled or played | 1 | 0.33% |
| record | score: List of two integers representing home and away scores | 1 | 0.33% |
| record | team1: First team participating in the match | 1 | 0.33% |
| record | location: Venue where the match took place | 1 | 0.33% |
| record | hotel_id | 1 | 0.33% |
| record | asset | 1 | 0.33% |
| record | dosage_form: Physical form of the drug such as tablet or injection | 1 | 0.33% |
| record | routeofadministration: Method by which the drug is administered | 1 | 0.33% |
| record | drug: Name of the medication as listed in the database | 1 | 0.33% |
| record | approved_date: Date the drug received regulatory approval | 1 | 0.33% |
| record | patient_population: Group of patients for whom the drug is intended | 1 | 0.33% |
| record | funding_id: Unique identifier for each funding round | 1 | 0.33% |
| record | roundtype: Type of funding round | 1 | 0.33% |
| record | status: Current status of the funding round | 1 | 0.33% |
| record | company_name: Name of the startup receiving funding | 1 | 0.33% |
| record | hba1c: Hemoglobin A1c percentage value | 1 | 0.33% |
| record | address: Street address including number and street name | 1 | 0.33% |
| record | cholesterol: Cholesterol level in mg/dL if elevated | 1 | 0.33% |
| record | visit_date: Date of last medical check-up | 1 | 0.33% |
| record | state: State abbreviation of patient's residence | 1 | 0.33% |
| record | diagnosis: Medical condition such as hypertension or diabetes | 1 | 0.33% |
| record | age: Age in years as stated in the record | 1 | 0.33% |
| record | zip_code: Postal code of patient's address | 1 | 0.33% |
| record | event_date | 1 | 0.33% |
| record | test_date | 1 | 0.33% |
| record | injury_count | 1 | 0.33% |
| record | indication | 1 | 0.33% |
| record | drug_name | 1 | 0.33% |
| record | patient_address | 1 | 0.33% |
| record | patient_age | 1 | 0.33% |
| record | ndc | 1 | 0.33% |
| record | start_date | 1 | 0.33% |
| record | medals.silver | 1 | 0.33% |
| record | games | 1 | 0.33% |
| record | medals.total | 1 | 0.33% |
| record | political_entity | 1 | 0.33% |
| record | service_type: Type of service impacted such as broadband or mobile | 1 | 0.33% |
| record | outage_end: Time when the outage ended in HH:MM AM/PM format | 1 | 0.33% |
| record | service_code: Internal code identifying the service type | 1 | 0.33% |
| record | ticket_number: Unique identifier for the outage ticket | 1 | 0.33% |
| record | outage_start: Time when the outage began in HH:MM AM/PM format | 1 | 0.33% |
| record | visit_count | 1 | 0.33% |
| record | platform | 1 | 0.33% |
| record | handle | 1 | 0.33% |
| record | retweets | 1 | 0.33% |
| record | content | 1 | 0.33% |
| record | lot_number | 1 | 0.33% |
| record | hire_date | 1 | 0.33% |
| record | salary | 1 | 0.33% |
| record | promotional_campaign | 1 | 0.33% |
| record | product_list_id | 1 | 0.33% |
| record | discount_amount | 1 | 0.33% |
| record | discount_condition | 1 | 0.33% |
| record | promotion_channels | 1 | 0.33% |
| record | engagement_rate | 1 | 0.33% |
| record | performance_code | 1 | 0.33% |
| record | sales_growth | 1 | 0.33% |
| record | audience_age_range | 1 | 0.33% |
| record | product_count | 1 | 0.33% |
| record | key_metric | 1 | 0.33% |
| record | product_focus | 1 | 0.33% |
| record | industry_segment | 1 | 0.33% |
| record | financial_figure | 1 | 0.33% |
| record | lead_analyst | 1 | 0.33% |
| record | last_updated | 1 | 0.33% |
| record | priority_registration.submission_methods | 1 | 0.33% |
| record | priority_registration.documents_required | 1 | 0.33% |
| record | on_site_registration.documents_required | 1 | 0.33% |
| record | important_notice | 1 | 0.33% |
| record | on_site_registration.description | 1 | 0.33% |
| record | priority_registration.instructions | 1 | 0.33% |
| record | on_site_registration.note | 1 | 0.33% |
| record | on_site_registration.location.address | 1 | 0.33% |
| record | online_priority_registration.deadline | 1 | 0.33% |
| record | disease.pathogenesis.transmission | 1 | 0.33% |
| record | disease.subspecies.2.name | 1 | 0.33% |
| record | disease.subspecies.0.affected_species | 1 | 0.33% |
| record | disease.subspecies.1.name | 1 | 0.33% |
| record | disease.subspecies.0.name | 1 | 0.33% |
| record | disease.description | 1 | 0.33% |
| record | disease.subspecies.1.associated_diseases | 1 | 0.33% |
| record | place.location.department | 1 | 0.33% |
| record | place.location.country | 1 | 0.33% |
| record | place.demographics.0.year | 1 | 0.33% |
| record | place.location.province | 1 | 0.33% |
| record | place.name | 1 | 0.33% |
| record | location: City and state where company is headquartered | 1 | 0.33% |
| record | investors: List of venture capital firms involved | 1 | 0.33% |
| record | company: Name of the company raising capital | 1 | 0.33% |
| record | round: Series or type of funding round | 1 | 0.33% |
| record | amount: Total amount raised in millions | 1 | 0.33% |
| record | product_id: Unique identifier for the product instance | 1 | 0.33% |
| record | passengers_affected | 1 | 0.33% |
| record | event_id | 1 | 0.33% |
| record | visit_date | 1 | 0.33% |
| record | condition | 1 | 0.33% |
| record | claims_count | 1 | 0.33% |
| record | filing_location | 1 | 0.33% |
| record | eps | 1 | 0.33% |
| record | net_margin | 1 | 0.33% |
| record | call_duration | 1 | 0.33% |
| record | risk_rating | 1 | 0.33% |
| record | property_type | 1 | 0.33% |
| record | appearances: Total games played in last season | 1 | 0.33% |
| record | contract_length: Duration of contract in years | 1 | 0.33% |
| record | position: Playing position such as midfielder, forward, defender | 1 | 0.33% |
| record | club_from: Previous club before transfer | 1 | 0.33% |
| record | club_to: New club after transfer | 1 | 0.33% |
| record | status: Current status of transfer | 1 | 0.33% |
| record | player_id: Unique identifier assigned to the player transfer record | 1 | 0.33% |
| record | age: Player's age at time of transfer | 1 | 0.33% |
| record | major | 1 | 0.33% |
| record | disbursement_plan | 1 | 0.33% |
| record | gpa | 1 | 0.33% |
| record | operating_margin | 1 | 0.33% |
| record | company_code | 1 | 0.33% |
| record | net_profit | 1 | 0.33% |
| record | album_series | 1 | 0.33% |
| record | album_title | 1 | 0.33% |
| record | message_id | 1 | 0.33% |
| record | outreach_status | 1 | 0.33% |
| record | outreach_code | 1 | 0.33% |
| record | film_title | 1 | 0.33% |
| record | word_count | 1 | 0.33% |
| record | study_metadata.study_type | 1 | 0.33% |
| record | study_metadata.measurement_technology.abbreviation | 1 | 0.33% |
| record | study_metadata.measurement_technology.full_name | 1 | 0.33% |
| record | study_metadata.measurement_technology.device_name | 1 | 0.33% |
| record | study_metadata.measurement_technology.anatomical_site | 1 | 0.33% |
| record | population.gender | 1 | 0.33% |
| record | population.age_range.min | 1 | 0.33% |
| record | population.age_range.max | 1 | 0.33% |
| record | population.age_range.unit | 1 | 0.33% |
| record | follow_up.average_duration.years | 1 | 0.33% |
| record | follow_up.average_duration.months | 1 | 0.33% |
| record | outcomes.data_source | 1 | 0.33% |
| record | outcomes.fracture_events.0.count | 1 | 0.33% |
| record | outcomes.fracture_events.0.type | 1 | 0.33% |
| record | statistical_results.hazard_ratios.0.adjustment_factors | 1 | 0.33% |
| record | statistical_results.hazard_ratios.0.value | 1 | 0.33% |
| record | statistical_results.hazard_ratios.0.comparison | 1 | 0.33% |
| record | statistical_results.diagnostic_metrics.sensitivity_at_threshold.t_score_threshold | 1 | 0.33% |
| record | statistical_results.diagnostic_metrics.sensitivity_at_threshold.percentage_of_fractures | 1 | 0.33% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.0.t_score | 1 | 0.33% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.0.rate_percentage | 1 | 0.33% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.1.t_score | 1 | 0.33% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.1.rate_percentage | 1 | 0.33% |
| record | statistical_results.predictive_model.value | 1 | 0.33% |
| record | statistical_results.predictive_model.variables_included | 1 | 0.33% |
| record | conclusions | 1 | 0.33% |
| record | statement_id | 1 | 0.33% |
| record | favorite_players | 1 | 0.33% |
| record | edit_information.timestamp | 1 | 0.33% |
| record | sentiment.opinion | 1 | 0.33% |
| record | quote.text | 1 | 0.33% |
| record | sentiment.preference | 1 | 0.33% |
| record | snapshot_id | 1 | 0.33% |
| record | source_currency | 1 | 0.33% |
| record | target_currency | 1 | 0.33% |
| record | first_client | 1 | 0.33% |
| record | year_founded | 1 | 0.33% |
| record | github_url | 1 | 0.33% |
| record | primates | 1 | 0.33% |
| record | course | 1 | 0.33% |
| record | status: Current status of the outreach interaction | 1 | 0.33% |
| record | sender: Name of the person sending the outreach | 1 | 0.33% |
| record | outreach_date: Date when the outreach email was sent | 1 | 0.33% |
| record | amount: Monetary value associated with the outreach | 1 | 0.33% |
| record | start_term | 1 | 0.33% |
| record | credits | 1 | 0.33% |
| record | hall | 1 | 0.33% |
| record | course_code | 1 | 0.33% |
| record | room | 1 | 0.33% |
| record | provider | 1 | 0.33% |
| record | vaccination_id | 1 | 0.33% |
| record | asteroid.orbital_characteristics.reference_plane | 1 | 0.33% |
| record | video_title | 1 | 0.33% |
| record | return_id | 1 | 0.33% |
| record | customer | 1 | 0.33% |
| record | product | 1 | 0.33% |
| record | product_model | 1 | 0.33% |
| record | order_id | 1 | 0.33% |
| record | refund_amount | 1 | 0.33% |
| record | reason | 1 | 0.33% |
| record | return_method | 1 | 0.33% |
| record | tracking_number | 1 | 0.33% |
| record | title_english | 1 | 0.33% |
| record | features: List of key technological or functional capabilities | 1 | 0.33% |
| record | price: Base price of the product in US dollars | 1 | 0.33% |
| record | regions: Geographic areas where the product is distributed or in use | 1 | 0.33% |
| record | name: Name of the product as introduced | 1 | 0.33% |
| record | building | 1 | 0.33% |
| record | property | 1 | 0.33% |
| record | lease_term | 1 | 0.33% |
| record | address: Patient's residential address including street, unit, and city | 1 | 0.33% |
| record | status: Current state of prescription fulfillment | 1 | 0.33% |
| record | pharmacy_name: Name of the pharmacy that processed the prescription | 1 | 0.33% |
| record | quantity: Number of tablets or units dispensed | 1 | 0.33% |
| record | dosage: Specific amount or form of medication prescribed | 1 | 0.33% |
| record | subscriber_name | 1 | 0.33% |
| record | id: Unique identifier for each rental listing | 1 | 0.33% |
| record | pet_policy: Pet policy including restrictions or fees | 1 | 0.33% |
| record | property_type: Type of housing unit | 1 | 0.33% |
| record | description: Brief description of unit features or conditions | 1 | 0.33% |
| record | address: Full property address including street and city | 1 | 0.33% |
| record | provider: Name of the medical professional or facility providing service | 1 | 0.33% |
| record | claim_amount: Total cost of the medical service billed | 1 | 0.33% |
| record | patient_address: Full address of the patient | 1 | 0.33% |
| record | primary_treatment | 1 | 0.33% |
| record | record_id | 1 | 0.33% |
| record | billing_code | 1 | 0.33% |
| record | discharge_diagnosis | 1 | 0.33% |
| record | discharge_instructions | 1 | 0.33% |
| record | physician.name | 1 | 0.33% |
| record | physician.degree | 1 | 0.33% |
| record | physician.address.street | 1 | 0.33% |
| record | physician.address.city | 1 | 0.33% |
| record | physician.address.state | 1 | 0.33% |
| record | physician.address.zip | 1 | 0.33% |
| record | physician.education.fellowship.0.institution | 1 | 0.33% |
| record | physician.education.residency.0.institution | 1 | 0.33% |
| record | physician.education.medical_school.0.institution | 1 | 0.33% |
| record | hotel_details.reception_open_until | 1 | 0.33% |
| record | hotel_details.check_out | 1 | 0.33% |
| record | hotel_details.number_of_rooms | 1 | 0.33% |
| record | child_policy.children.age_range | 1 | 0.33% |
| record | hotel_details.check_in | 1 | 0.33% |
| record | posted_on | 1 | 0.33% |
| record | title: Event title describing the nature and context of the incident | 1 | 0.33% |
| record | date: Timestamp when the incident occurred | 1 | 0.33% |
| record | involved_parties: List of organizations or individuals involved in the incident | 1 | 0.33% |
| record | source: Media or organization reporting the incident | 1 | 0.33% |
| record | lead_reporter: Reporter providing coverage of the incident | 1 | 0.33% |
| record | geographic_code: Unique code identifying the incident's geographic region | 1 | 0.33% |
| record | followupinterval | 1 | 0.33% |
| record | ticker | 1 | 0.33% |
| record | market_cap | 1 | 0.33% |
| record | attendance: Percentage of days attended by participant | 1 | 0.33% |
| record | participant_id: Unique employee identifier from enrollment data | 1 | 0.33% |
| record | status: Current status of the enrollment | 1 | 0.33% |
| record | program_name: Title of the training program enrolled in | 1 | 0.33% |
| record | delivery_mode: Mode of training delivery | 1 | 0.33% |
| record | opponent | 1 | 0.33% |
| record | game_status | 1 | 0.33% |
| record | event_type | 1 | 0.33% |
| record | performance_notes | 1 | 0.33% |
| record | goals | 1 | 0.33% |
| record | player | 1 | 0.33% |
| record | team | 1 | 0.33% |
| record | salary_range: Expected or offered salary range for the role | 1 | 0.33% |
| record | target_role: Job title or position being targeted | 1 | 0.33% |
| record | message_id: Unique identifier for the outreach message | 1 | 0.33% |
| record | target_company: Company where the role is located | 1 | 0.33% |
| record | policyholder_id: Unique identifier for the policyholder | 1 | 0.33% |
| record | policyholder_name: Full name of the claimant | 1 | 0.33% |
| record | claim_id: Unique identifier for the submitted claim | 1 | 0.33% |
| record | ebitda_margin | 1 | 0.33% |
| record | sector | 1 | 0.33% |
| record | liquidity_ratio | 1 | 0.33% |
| record | ceo | 1 | 0.33% |
| record | auditstatus | 1 | 0.33% |
| record | dividend_yield | 1 | 0.33% |
| record | dateofincident | 1 | 0.33% |
| record | policyholder_name | 1 | 0.33% |
| record | ram | 1 | 0.33% |
| record | species.family | 1 | 0.33% |
| record | session_id | 1 | 0.33% |
| record | streaming_platform | 1 | 0.33% |
| record | end_time | 1 | 0.33% |
| record | speaker | 1 | 0.33% |
| record | genre | 1 | 0.33% |
| record | affiliation.institution_name | 1 | 0.33% |
| record | research_background.animal_models_used | 1 | 0.33% |
| record | affiliation.department | 1 | 0.33% |
| record | cpt_code: CPT code for the medical procedure performed | 1 | 0.33% |
| record | paid_amount: Amount actually received from insurance | 1 | 0.33% |
| record | diagnosis: ICD-10 code representing the medical condition | 1 | 0.33% |
| record | billed_amount: Total amount billed for the service | 1 | 0.33% |
| record | metabolite | 1 | 0.33% |
| record | contact_name | 1 | 0.33% |
| record | vehicle | 1 | 0.33% |
| record | vin | 1 | 0.33% |
| record | birth_information.birth_year | 1 | 0.33% |
| record | profession | 1 | 0.33% |
| record | biography.education.0.teacher | 1 | 0.33% |
| record | biography.education.4.institution | 1 | 0.33% |
| record | biography.awards.0.award_name | 1 | 0.33% |
| record | biography.concert_tours | 1 | 0.33% |
| record | biography.education.4.teacher | 1 | 0.33% |
| record | biography.collaborations | 1 | 0.33% |
| record | biography.education.3.institution | 1 | 0.33% |
| record | biography.education.2.teacher | 1 | 0.33% |
| record | biography.teaching_position.institution | 1 | 0.33% |
| record | biography.teaching_position.mentor | 1 | 0.33% |
| record | birth_information.birth_place.state | 1 | 0.33% |
| record | birth_information.birth_place.county | 1 | 0.33% |
| record | capacity | 1 | 0.33% |
| record | sponsor | 1 | 0.33% |
| record | transaction_id: Unique identifier for the transaction entry | 1 | 0.33% |
| record | date: Date when the transaction occurred | 1 | 0.33% |
| record | payee: Name of the recipient or entity involved | 1 | 0.33% |
| record | email_id | 1 | 0.33% |
| record | event-attack/battle/war/militaryconflict | 1 | 0.33% |
| record | percentage | 1 | 0.33% |
| record | headlines | 1 | 0.33% |
| record | patient_name: Name of the affected individual as reported | 1 | 0.33% |
| record | patient_age: Age of the patient at time of reaction | 1 | 0.33% |
| record | patient_gender: Gender of the patient | 1 | 0.33% |
| record | patient_address: Full residential address of the patient | 1 | 0.33% |
| record | drug_name: Name of the medication involved in the reaction | 1 | 0.33% |
| record | drug_dosage: Dosage and frequency of the drug exposure | 1 | 0.33% |
| record | report_id: Unique identifier assigned to the adverse event report | 1 | 0.33% |
| record | acquired_company: Name of the company being acquired | 1 | 0.33% |
| record | sector: Industry or market sector involved in the acquisition | 1 | 0.33% |
| record | acquirer_ceo: CEO of the acquiring company | 1 | 0.33% |
| record | deal_id: Unique identifier for the acquisition transaction | 1 | 0.33% |
| record | RP | 1 | 0.33% |
| record | VBD | 1 | 0.33% |
| record | singles | 1 | 0.33% |
| record | item_description | 1 | 0.33% |
| record | po_reference | 1 | 0.33% |
| record | location-gpe | 1 | 0.33% |
| record | passenger_name: Name of the passenger on the flight | 1 | 0.33% |
| record | flight_no: Flight number as assigned by the airline | 1 | 0.33% |
| record | delay_minutes: Number of minutes flight is delayed | 1 | 0.33% |
| record | destination: Airport code where flight ends | 1 | 0.33% |
| record | email: Email address of the passenger | 1 | 0.33% |
| record | unique_id: Unique identifier for the delay entry | 1 | 0.33% |
| record | departure_date: Date flight departs from origin | 1 | 0.33% |
| record | diagnosis | 1 | 0.33% |
| record | lab_results | 1 | 0.33% |
| record | mrn | 1 | 0.33% |
| record | priority | 1 | 0.33% |
| record | root_cause | 1 | 0.33% |
| record | service | 1 | 0.33% |
| record | resolution_team | 1 | 0.33% |
| record | grant_id | 1 | 0.33% |
| record | planning | 1 | 0.33% |
| record | startup | 1 | 0.33% |
| record | investor | 1 | 0.33% |
| record | funding_round | 1 | 0.33% |
| record | release_target | 1 | 0.33% |
| record | utilities_included | 1 | 0.33% |
| record | contact_method | 1 | 0.33% |
| record | preferredtenanttype | 1 | 0.33% |
| record | name: Full title of the course including subject and focus | 1 | 0.33% |
| record | instructor: Name of the faculty member teaching the course | 1 | 0.33% |
| record | department: Academic department offering the course | 1 | 0.33% |
| record | credit_hours: Number of credit hours for the course | 1 | 0.33% |
| record | tournament | 1 | 0.33% |
| record | founder | 1 | 0.33% |
| record | founded_year | 1 | 0.33% |
| record | era_name | 1 | 0.33% |
| record | awayteampoints | 1 | 0.33% |
| record | key_player | 1 | 0.33% |
| record | assists | 1 | 0.33% |
| record | points | 1 | 0.33% |
| record | steals | 1 | 0.33% |
| record | blocks | 1 | 0.33% |
| record | person | 1 | 0.33% |
| record | boulder.start.foot_holds.1.foot | 1 | 0.33% |
| record | boulder.start.foot_holds.0.foot | 1 | 0.33% |
| record | boulder.start.hand_holds.1.description | 1 | 0.33% |
| record | boulder.start.hand_holds.0.description | 1 | 0.33% |
| record | boulder.start.foot_holds.0.description | 1 | 0.33% |
| record | boulder.start.hand_holds.1.hand | 1 | 0.33% |
| record | boulder.location.relative_position | 1 | 0.33% |
| record | boulder.location.distance_unit | 1 | 0.33% |
| record | boulder.location.reference_point | 1 | 0.33% |
| record | investigators | 1 | 0.33% |
| record | funding_sources | 1 | 0.33% |
| record | secondary_endpoints | 1 | 0.33% |
| record | median_age | 1 | 0.33% |
| record | onboarding_stage | 1 | 0.33% |
| record | score: Numerical performance score from 3.0 to 4.1 | 1 | 0.33% |
| record | employee_id: Unique identifier assigned to the employee | 1 | 0.33% |
| record | department: Department where the employee works | 1 | 0.33% |
| record | goal_achievement: Percentage of goals met, from 60% to 100% | 1 | 0.33% |
| record | isbn | 1 | 0.33% |
| record | publication_volume | 1 | 0.33% |
| record | publication_issue | 1 | 0.33% |
| record | places | 1 | 0.33% |
| record | spoofed_domain | 1 | 0.33% |
| record | target_department | 1 | 0.33% |
| record | risk_classification | 1 | 0.33% |
| record | incident_type | 1 | 0.33% |
| record | disruption_duration | 1 | 0.33% |
| record | foods_mentioned | 1 | 0.33% |
| record | rating: Star rating given by the reviewer out of 5 | 1 | 0.33% |
| record | status: Current status of the review | 1 | 0.33% |
| record | model: Unique identifier or model number of the device | 1 | 0.33% |
| record | word_count: Number of words in the review text | 1 | 0.33% |
| record | property_address: Full address of the leased property | 1 | 0.33% |
| record | leasetermlength: Duration of the lease in years | 1 | 0.33% |
| record | purchase_location | 1 | 0.33% |
| record | town.population.density_per_sq_km | 1 | 0.33% |
| record | town.name | 1 | 0.33% |
| record | town.area_sq_km | 1 | 0.33% |
| record | town.population.year | 1 | 0.33% |
| record | town.location | 1 | 0.33% |
| record | status: Current status of the exam result | 1 | 0.33% |
| record | school: Name of the school where the exam was administered | 1 | 0.33% |
| record | score: Percentage score achieved in the exam | 1 | 0.33% |
| record | subject: Subject area of the exam | 1 | 0.33% |
| record | political_context | 1 | 0.33% |
| record | thumbnail_style | 1 | 0.33% |
| record | channel_name | 1 | 0.33% |
| record | description_length | 1 | 0.33% |
| record | insurance_plan | 1 | 0.33% |
| record | claim_status | 1 | 0.33% |
| record | patient | 1 | 0.33% |
| record | provider_address | 1 | 0.33% |
| record | phone | 1 | 0.33% |
| record | guidelines.distinction_results_interpretation | 1 | 0.33% |
| record | optional_information | 1 | 0.33% |
| record | research_framework_items | 1 | 0.33% |
| record | publication_date.city | 1 | 0.33% |
| record | publication_date.month | 1 | 0.33% |
| record | guidelines.adherence | 1 | 0.33% |
| record | organization.name | 1 | 0.33% |
| record | user_id | 1 | 0.33% |
| record | subject_line | 1 | 0.33% |
| record | district | 1 | 0.33% |
| record | tehsil.tehsil_name | 1 | 0.33% |
| record | tehsil.district | 1 | 0.33% |
| record | professions | 1 | 0.33% |
| record | award | 1 | 0.33% |
| record | user_id: Unique identifier for the user on TikTok | 1 | 0.33% |
| record | destination | 1 | 0.33% |
| record | patient_id: Unique patient identifier from the medical record | 1 | 0.33% |
| record | modality: Type of imaging modality used | 1 | 0.33% |
| record | findings: Key clinical observations or abnormalities detected | 1 | 0.33% |
| record | report_id: Unique identifier for the imaging report | 1 | 0.33% |
| record | status: Current status of the report workflow | 1 | 0.33% |
| record | submitter: Name of employee who filed the ticket | 1 | 0.33% |
| record | priority: Severity level assigned to the ticket | 1 | 0.33% |
| record | ticket_id: Unique identifier for each support request | 1 | 0.33% |
| record | test_name | 1 | 0.33% |
| record | lead_person | 1 | 0.33% |
| record | leasing_offer | 1 | 0.33% |
| record | tenant_name | 1 | 0.33% |
| record | municipality.name_local | 1 | 0.33% |
| record | municipality.location.region | 1 | 0.33% |
| record | municipality.location.country | 1 | 0.33% |
| record | demographics.population.0.count | 1 | 0.33% |
| record | demographics.population.0.year | 1 | 0.33% |
| record | hotel_experience.location.nearby_landmark | 1 | 0.33% |
| record | hotel_experience.location.city | 1 | 0.33% |
| record | hotel_experience.entertainment.tv.content | 1 | 0.33% |
| record | hotel_experience.additional_benefits | 1 | 0.33% |
| record | hotel_experience.suites.0.workspace | 1 | 0.33% |
| record | hotel_experience.suites.0.suite_type | 1 | 0.33% |
| record | hotel_experience.suites.0.bathroom_features | 1 | 0.33% |
| record | hotel_experience.suites.0.amenities | 1 | 0.33% |
| record | hotel_experience.suites.0.features | 1 | 0.33% |
| record | hotel_experience.entertainment.tv.screen_size | 1 | 0.33% |
| record | keynote_time | 1 | 0.33% |
| record | isps | 1 | 0.33% |
| record | software_versions | 1 | 0.33% |
| record | line1 | 1 | 0.33% |
| record | patient_mrn: Unique medical record number identifying the patient | 1 | 0.33% |
| record | primary_diagnosis: Clinical diagnosis code and description | 1 | 0.33% |
| record | treatments: Prescribed medications or therapies during stay | 1 | 0.33% |
| record | followupprovider: Name of provider for follow-up visit | 1 | 0.33% |
| record | followupclinic_address: Physical address of follow-up clinic | 1 | 0.33% |
| record | discharge_instructions: Written guidance given to patient post-discharge | 1 | 0.33% |
| record | clause_id | 1 | 0.33% |
| record | party_name | 1 | 0.33% |
| record | duration | 1 | 0.33% |
| record | upload_date | 1 | 0.33% |
| record | resolution_summary | 1 | 0.33% |
| record | customer_address | 1 | 0.33% |
| record | source | 1 | 0.33% |
| record | steps | 1 | 0.33% |
| record | home_team: Name of the team playing at home | 1 | 0.33% |
| record | bench_points: Total points scored by bench players | 1 | 0.33% |
| record | key_moment: Descriptive note about a pivotal play or event in the game | 1 | 0.33% |
| record | steals: Total steals recorded during the game | 1 | 0.33% |
| record | game_id: Unique identifier for the game as provided in the passage | 1 | 0.33% |
| record | home_score: Final score of the home team | 1 | 0.33% |
| record | acquirer | 1 | 0.33% |
| record | target | 1 | 0.33% |
| record | launch_date | 1 | 0.33% |
| record | target_market | 1 | 0.33% |
| record | deployment_status | 1 | 0.33% |
| record | target_demographic | 1 | 0.33% |
| record | distribution_channels | 1 | 0.33% |
| record | customer_satisfaction | 1 | 0.33% |
| record | agency_code | 1 | 0.33% |
| record | office_address | 1 | 0.33% |
| record | bloodpressurediastolic | 1 | 0.33% |
| record | monitoring_method | 1 | 0.33% |
| record | bloodpressuresystolic | 1 | 0.33% |
| record | respiratory_rate | 1 | 0.33% |
| record | consultation.practice_areas | 1 | 0.33% |
| record | practice_areas | 1 | 0.33% |
| record | firm_name | 1 | 0.33% |
| record | states_served | 1 | 0.33% |
| record | services | 1 | 0.33% |
| record | client_focus | 1 | 0.33% |
| record | external_links.0.title | 1 | 0.33% |
| record | classification.source | 1 | 0.33% |
| record | observations.observation_period.years | 1 | 0.33% |
| record | wiki_sky_links | 1 | 0.33% |
| record | alternative_designations | 1 | 0.33% |
| record | orbital_characteristics.perihelion | 1 | 0.33% |
| record | discovery.location.observatory | 1 | 0.33% |
| record | observations.last_observation_date | 1 | 0.33% |
| record | classification.type | 1 | 0.33% |
| record | orbital_characteristics.orbital_period.years | 1 | 0.33% |
| record | discovery.location.place | 1 | 0.33% |
| record | company_id: Unique identifier assigned to each company | 1 | 0.33% |
| record | program_name | 1 | 0.33% |
| record | dimensions: Dimensions in cm format | 1 | 0.33% |
| record | name: Name of the product as listed | 1 | 0.33% |
| record | material: Primary materials used in construction | 1 | 0.33% |
| record | weight: Weight of the product in grams or kg | 1 | 0.33% |
| record | hotel_details.year_renovated | 1 | 0.33% |
| record | location.distance_from_airport.unit | 1 | 0.33% |
| record | owner_contact | 1 | 0.33% |
| record | location.distance_from_city_center.unit | 1 | 0.33% |
| record | location.distance_from_city_center.value | 1 | 0.33% |
| record | search_result_page.search_query | 1 | 0.33% |
| record | search_result_page.website_name | 1 | 0.33% |
| record | initial_text.negations | 1 | 0.33% |
| record | operational_status: Current phase of business operations | 1 | 0.33% |
| record | net_profit: Annual net profit in millions of dollars | 1 | 0.33% |
| record | location: Geographic headquarters of the company | 1 | 0.33% |
| record | ebitda_margin: Percentage of revenue that represents EBITDA | 1 | 0.33% |
| record | current_ratio: Liquidity ratio measuring short-term obligations | 1 | 0.33% |
| record | claimant_name | 1 | 0.33% |
| record | claim_date | 1 | 0.33% |
| record | bracket_path | 1 | 0.33% |
| record | team_name | 1 | 0.33% |
| record | team_id | 1 | 0.33% |
| record | registered_on | 1 | 0.33% |
| record | hotel_details.star_rating | 1 | 0.33% |
| record | location.region | 1 | 0.33% |
| record | ranking.location | 1 | 0.33% |
| record | hotel_details.location | 1 | 0.33% |
| record | booking_options.partners | 1 | 0.33% |
| record | location.country | 1 | 0.33% |
| record | booking_options.description | 1 | 0.33% |
| record | price_range.currency | 1 | 0.33% |
| record | hotel_details.name | 1 | 0.33% |
| record | flight_number | 1 | 0.33% |
| record | origin | 1 | 0.33% |
| record | booking_reference | 1 | 0.33% |
| record | passenger_name | 1 | 0.33% |
| record | rating: Numerical score out of 5 given by the reviewer | 1 | 0.33% |
| record | publication: Name of the media outlet publishing the review | 1 | 0.33% |
| record | merchantcategorycode | 1 | 0.33% |
| record | keywords | 1 | 0.33% |
| record | publication_time | 1 | 0.33% |
| record | company.financial_results.reporting_period | 1 | 0.33% |
| record | report_metadata.source_url | 1 | 0.33% |
| record | company.name | 1 | 0.33% |
| record | reportTitle | 1 | 0.33% |
| record | pathway_id | 1 | 0.33% |
| record | patient_cohort | 1 | 0.33% |
| record | distributor_information.capabilities | 1 | 0.33% |
| record | services.maintenance | 1 | 0.33% |
| record | distributor_information.benefits | 1 | 0.33% |
| record | procedure_code | 1 | 0.33% |
| record | study.title | 1 | 0.33% |
| record | study.quotes | 1 | 0.33% |
| record | study.key_findings | 1 | 0.33% |
| record | study.period.end_year | 1 | 0.33% |
| record | study.author.name | 1 | 0.33% |
| record | study.author.affiliation | 1 | 0.33% |
| record | amount: Numerical value of the exchange rate | 1 | 0.33% |
| record | description: Brief explanation of market event influencing the rate | 1 | 0.33% |
| record | currency_pair: Pair of currencies represented in the rate | 1 | 0.33% |
| record | appointment_id | 1 | 0.33% |
| record | organizer: Person or group hosting the event | 1 | 0.33% |
| record | name: Title or name of the event | 1 | 0.33% |
| record | location: Physical address or venue of the event | 1 | 0.33% |
| record | price: Entry cost for attending the event | 1 | 0.33% |
| record | status: Current phase of the event lifecycle | 1 | 0.33% |
| record | event_id: Unique identifier for each event | 1 | 0.33% |
| record | event_type: Category of the event | 1 | 0.33% |
| record | trademark_name: Name of the registered trademark in dispute | 1 | 0.33% |
| record | registrant: Company that owns the registered trademark | 1 | 0.33% |
| record | registration_number: Unique number assigned to the trademark registration | 1 | 0.33% |
| record | opponent: Entity filing the dispute or opposition | 1 | 0.33% |
| record | dispute_date: Date the dispute or opposition was filed | 1 | 0.33% |
| record | status: Current legal status of the dispute | 1 | 0.33% |
| record | dispute_type: Type of legal action such as cancellation or infringement | 1 | 0.33% |
| record | sector: Technology domain such as health tech, AI, clean energy | 1 | 0.33% |
| record | funding_amount: Total amount raised in the investment round | 1 | 0.33% |
| record | id: Unique identifier assigned to the startup | 1 | 0.33% |
| record | magazine.article.page_breakdown.6.page_number | 1 | 0.33% |
| record | magazine.article.page_breakdown.13.advertising_percentage | 1 | 0.33% |
| record | magazine.article.page_breakdown.8.page_number | 1 | 0.33% |
| record | magazine.article.page_breakdown.14.page_number | 1 | 0.33% |
| record | magazine.article.page_breakdown.13.advertisers | 1 | 0.33% |
| record | magazine.article.page_breakdown.16.page_number | 1 | 0.33% |
| record | magazine.article.page_breakdown.3.advertising_percentage | 1 | 0.33% |
| record | magazine.issue | 1 | 0.33% |
| record | magazine.article.page_breakdown.15.advertisers | 1 | 0.33% |
| record | magazine.article.page_breakdown.8.advertisers | 1 | 0.33% |
| record | magazine.article.page_breakdown.11.advertising_percentage | 1 | 0.33% |
| record | magazine.article.page_breakdown.7.advertisers | 1 | 0.33% |
| record | magazine.article.page_breakdown.6.advertisers | 1 | 0.33% |
| record | magazine.article.page_breakdown.1.advertisers | 1 | 0.33% |
| record | magazine.article.page_breakdown.12.advertisers | 1 | 0.33% |
| record | magazine.article.page_breakdown.16.advertising_percentage | 1 | 0.33% |
| record | magazine.article.page_breakdown.9.page_number | 1 | 0.33% |
| record | magazine.article.page_breakdown.0.page_number | 1 | 0.33% |
| record | magazine.article.page_breakdown.3.page_number | 1 | 0.33% |
| record | magazine.article.page_breakdown.0.advertisers | 1 | 0.33% |
| record | magazine.article.page_breakdown.12.page_number | 1 | 0.33% |
| record | magazine.article.page_breakdown.1.advertising_percentage | 1 | 0.33% |
| record | magazine.name | 1 | 0.33% |
| record | magazine.article.content_percentage | 1 | 0.33% |
| record | magazine.article.page_breakdown.12.advertising_percentage | 1 | 0.33% |
| record | magazine.article.page_breakdown.5.advertising_percentage | 1 | 0.33% |
| record | magazine.article.page_breakdown.4.advertising_percentage | 1 | 0.33% |
| record | magazine.article.page_breakdown.8.advertising_percentage | 1 | 0.33% |
| record | magazine.article.page_breakdown.0.advertising_percentage | 1 | 0.33% |
| record | magazine.article.page_breakdown.9.advertisers | 1 | 0.33% |
| record | magazine.article.page_breakdown.10.advertisers | 1 | 0.33% |
| record | magazine.article.page_breakdown.10.advertising_percentage | 1 | 0.33% |
| record | magazine.article.page_breakdown.2.page_number | 1 | 0.33% |
| record | magazine.article.page_breakdown.6.advertising_percentage | 1 | 0.33% |
| record | magazine.article.page_breakdown.11.page_number | 1 | 0.33% |
| record | company_name: Name of the startup founded by the individual | 1 | 0.33% |
| record | founder_name: Full name of the founder as stated in passage | 1 | 0.33% |
| record | motivation: Personal reason driving the company's founding | 1 | 0.33% |
| record | focus_area: Primary domain or mission of the company | 1 | 0.33% |
| record | location: City and state where the company was founded | 1 | 0.33% |
| record | marketing_budget | 1 | 0.33% |
| record | iporevisedvaluation | 1 | 0.33% |
| record | originalheadcountapex | 1 | 0.33% |
| record | resolution_timeline | 1 | 0.33% |
| record | correctiveactionplan | 1 | 0.33% |
| record | document_id | 1 | 0.33% |
| record | responsible_party | 1 | 0.33% |
| record | regulation_referenced | 1 | 0.33% |
| record | siblings.0.name | 1 | 0.33% |
| record | siblings_in_law.0.relationship | 1 | 0.33% |
| record | employment.2.company | 1 | 0.33% |
| record | nieces | 1 | 0.33% |
| record | siblings_in_law.3.relationship | 1 | 0.33% |
| record | predeceased.2.relationship | 1 | 0.33% |
| record | predeceased.5.relationship | 1 | 0.33% |
| record | siblings_in_law.1.relationship | 1 | 0.33% |
| record | employment.1.company | 1 | 0.33% |
| record | siblings.0.location | 1 | 0.33% |
| record | nephews | 1 | 0.33% |
| record | grandchildren | 1 | 0.33% |
| record | siblings_in_law.3.name | 1 | 0.33% |
| record | siblings_in_law.1.name | 1 | 0.33% |
| record | predeceased.7.name | 1 | 0.33% |
| record | residence | 1 | 0.33% |
| record | employment.3.company | 1 | 0.33% |
| record | predeceased.3.relationship | 1 | 0.33% |
| record | siblings_in_law.1.location | 1 | 0.33% |
| record | siblings_in_law.0.name | 1 | 0.33% |
| record | parents.0.name | 1 | 0.33% |
| record | predeceased.3.name | 1 | 0.33% |
| record | employment.4.company | 1 | 0.33% |
| record | predeceased.4.name | 1 | 0.33% |
| record | children.1.spouse | 1 | 0.33% |
| record | marriage_duration | 1 | 0.33% |
| record | siblings_in_law.0.location | 1 | 0.33% |
| record | predeceased.0.name | 1 | 0.33% |
| record | predeceased.7.relationship | 1 | 0.33% |
| record | document_details.series_title | 1 | 0.33% |
| record | document_details.description | 1 | 0.33% |
| record | document_details.oclc_number | 1 | 0.33% |
| record | merchant | 1 | 0.33% |
| record | assessment_name | 1 | 0.33% |
| record | publication_code | 1 | 0.33% |
| record | office: Regional office location where employee is based | 1 | 0.33% |
| record | manager: Name of the employee's manager | 1 | 0.33% |
| record | name: Full name of the new hire | 1 | 0.33% |
| record | assigned_coordinator: Name of the onboarding coordinator assigned to the employee | 1 | 0.33% |
| record | punct | 1 | 0.33% |
| record | dobj | 1 | 0.33% |
| record | nmod | 1 | 0.33% |
| record | cc | 1 | 0.33% |
| record | preconj | 1 | 0.33% |
| record | nsubj | 1 | 0.33% |
| record | case | 1 | 0.33% |
| record | ROOT | 1 | 0.33% |
| record | advmod | 1 | 0.33% |
| record | npadvmod | 1 | 0.33% |
| record | prep | 1 | 0.33% |
| record | intj | 1 | 0.33% |
| record | nummod | 1 | 0.33% |
| record | transfer_id | 1 | 0.33% |
| record | player_id | 1 | 0.33% |
| record | player_name | 1 | 0.33% |
| record | from_club | 1 | 0.33% |
| record | to_club | 1 | 0.33% |
| record | transfer_fee | 1 | 0.33% |
| record | transfer_code | 1 | 0.33% |
| record | transfer_portal | 1 | 0.33% |
| record | asteroid.discovery.location | 1 | 0.33% |
| record | asteroid.name | 1 | 0.33% |
| record | album | 1 | 0.33% |
| record | artist | 1 | 0.33% |
| record | record_label | 1 | 0.33% |
| record | founder_id | 1 | 0.33% |
| record | country: Nationality of the athlete who won the medal | 1 | 0.33% |
| record | venue: Location where the event took place | 1 | 0.33% |
| record | result: Medal awarded to the athlete | 1 | 0.33% |
| record | discipline: Type of sport or competition such as swimming or skiing | 1 | 0.33% |
| record | athlete_name: Name of the athlete who achieved the result | 1 | 0.33% |
| record | insurance_id | 1 | 0.33% |
| record | amount_paid | 1 | 0.33% |
| record | performance_impact: Percentage improvement in lead quality or hiring time | 1 | 0.33% |
| record | budget | 1 | 0.33% |
| record | agency | 1 | 0.33% |
| record | technology | 1 | 0.33% |
| record | protecting_group.targets.1.abbreviation | 1 | 0.33% |
| record | protecting_group.targets.1.amino_acid_name | 1 | 0.33% |
| record | protecting_group.targets.1.derivative_form | 1 | 0.33% |
| record | protecting_group.targets.0.amino_acid_name | 1 | 0.33% |
| record | protecting_group.abbreviation | 1 | 0.33% |
| record | protecting_group.targets.1.protected_functional_group | 1 | 0.33% |
| record | deprotection_process.reagents.0.concentration_level | 1 | 0.33% |
| record | protecting_group.name | 1 | 0.33% |
| record | synthesis_context.intermediates_synthesized.0.components | 1 | 0.33% |
| record | protecting_group.targets.0.abbreviation | 1 | 0.33% |
| record | synthesis_context.methodology | 1 | 0.33% |
| record | synthesis_context.carboxyl_protection_requirement.reasoning | 1 | 0.33% |
| record | synthesis_context.intermediates_synthesized.0.type | 1 | 0.33% |
| record | protecting_group.targets.0.protected_functional_group | 1 | 0.33% |
| record | expiry_date | 1 | 0.33% |
| record | cas_number | 1 | 0.33% |
| record | issuer | 1 | 0.33% |
| record | filer | 1 | 0.33% |
| record | asteroid.dedication.person.name | 1 | 0.33% |
| record | asteroid.dedication.person.occupation | 1 | 0.33% |
| record | release_date | 1 | 0.33% |
| record | availability | 1 | 0.33% |
| record | color | 1 | 0.33% |
| record | stock_level | 1 | 0.33% |
| record | dimensions | 1 | 0.33% |
| record | entity_name | 1 | 0.33% |
| record | file_id | 1 | 0.33% |
| record | jurisdiction | 1 | 0.33% |
| record | post_date: Date when the trend repost was published | 1 | 0.33% |
| record | cpt_code | 1 | 0.33% |
| record | incident.suspects.0.age | 1 | 0.33% |
| record | incident.suspects.1.age | 1 | 0.33% |
| record | incident.charges.0.count | 1 | 0.33% |
| record | incident.charges.0.description | 1 | 0.33% |
| record | incident.charges.1.count | 1 | 0.33% |
| record | incident.charges.1.description | 1 | 0.33% |
| record | incident.charges.2.count | 1 | 0.33% |
| record | incident.charges.2.description | 1 | 0.33% |
| record | incident.additional_acts | 1 | 0.33% |
| record | incident.affected_locations | 1 | 0.33% |
| record | incident.number_of_complaints | 1 | 0.33% |
| record | source.newspaper | 1 | 0.33% |
| record | source.author | 1 | 0.33% |
| record | post_date | 1 | 0.33% |
| record | user_handle | 1 | 0.33% |
| record | description: Brief text describing the service or product delivered | 1 | 0.33% |
| record | invoice_number: Unique identifier for the invoice as assigned | 1 | 0.33% |
| record | status: Current financial status of the invoice | 1 | 0.33% |
| record | naming.named_after.profession | 1 | 0.33% |
| record | naming.named_after.name | 1 | 0.33% |
| record | discovery.observatory | 1 | 0.33% |
| record | discovery.date | 1 | 0.33% |
| record | size_and_fit.length.unit | 1 | 0.33% |
| record | size_and_fit.measurements.2.value | 1 | 0.33% |
| record | price.installment_option.currency | 1 | 0.33% |
| record | size_and_fit.measurements.2.unit | 1 | 0.33% |
| record | price.installment_option.installment_amount | 1 | 0.33% |
| record | care_instructions | 1 | 0.33% |
| record | size_and_fit.measurements.1.type | 1 | 0.33% |
| record | size_and_fit.length.size | 1 | 0.33% |
| record | size_and_fit.measurements.1.unit | 1 | 0.33% |
| record | price.installment_option.installments | 1 | 0.33% |
| record | size_and_fit.details | 1 | 0.33% |
| record | size_and_fit.measurements.2.type | 1 | 0.33% |
| record | establishment_year | 1 | 0.33% |
| record | departments | 1 | 0.33% |
| record | staff.research_assistant_count | 1 | 0.33% |
| record | staff.teaching_staff_count | 1 | 0.33% |
| record | dean.name | 1 | 0.33% |
| record | laboratories | 1 | 0.33% |
| record | dean.title | 1 | 0.33% |
| record | donor_name | 1 | 0.33% |
| record | donor_id | 1 | 0.33% |
| record | donation_date | 1 | 0.33% |
| record | candidate_name | 1 | 0.33% |
| record | submission_id: Unique identifier for the paper submission | 1 | 0.33% |
| record | author_name: Name of the paper's author | 1 | 0.33% |
| record | affiliation: Institutional affiliation of the author | 1 | 0.33% |
| record | paper_title: Title of the submitted research paper | 1 | 0.33% |
| record | submission_date: Date when the paper was submitted | 1 | 0.33% |
| record | word_count: Total number of words in the paper | 1 | 0.33% |
| record | fieldofstudy: Academic domain of the research | 1 | 0.33% |
| record | conference_name: Name of the conference where submitted | 1 | 0.33% |
| record | review_date: Date when the paper was reviewed | 1 | 0.33% |
| record | funding_source: Source of research funding | 1 | 0.33% |
| record | page_count: Number of pages in the paper | 1 | 0.33% |
| record | abstract_length: Length of the abstract in words | 1 | 0.33% |
| record | risk_score: Numerical value indicating the level of risk exposure | 1 | 0.33% |
| record | risk_category: Categorical classification of risk profile | 1 | 0.33% |
| record | policy_id: Identifier assigned to the insurance policy | 1 | 0.33% |
| record | notes | 1 | 0.33% |
| record | report_date | 1 | 0.33% |
| record | checklist_id | 1 | 0.33% |
| record | medical_code | 1 | 0.33% |
| record | launch_year: Year the company launched its platform | 1 | 0.33% |
| record | physical_characteristics.observations.0.telescope | 1 | 0.33% |
| record | designation | 1 | 0.33% |
| record | orbital_characteristics.classification | 1 | 0.33% |
| record | discovery.discoverer | 1 | 0.33% |
| record | discovery.observatory.location.prefecture | 1 | 0.33% |
| record | physical_characteristics.observations.0.year | 1 | 0.33% |
| record | discovery.observatory.location.country | 1 | 0.33% |
| record | orbital_characteristics.semi_major_axis.unit | 1 | 0.33% |
| record | orbital_characteristics.perihelion.unit | 1 | 0.33% |
| record | booth_size | 1 | 0.33% |
| record | exhibitor_id | 1 | 0.33% |
| record | registration_date | 1 | 0.33% |
| record | founder_name | 1 | 0.33% |
| record | founding_date | 1 | 0.33% |
| record | initial_funding | 1 | 0.33% |
| record | investment_source | 1 | 0.33% |
| record | industry | 1 | 0.33% |
| record | early_challenges | 1 | 0.33% |
| record | exit_status | 1 | 0.33% |
| record | current_status | 1 | 0.33% |
| record | risk_score | 1 | 0.33% |
| record | risk_category | 1 | 0.33% |
| record | geographic_region | 1 | 0.33% |
| record | dynasty | 1 | 0.33% |
| record | description: Brief description of the transaction activity | 1 | 0.33% |
| record | creditcardlast_four: Last four digits of the linked credit card | 1 | 0.33% |
| record | merchant_name: Name of the merchant or store | 1 | 0.33% |
| record | transaction_amount: Amount of the individual transaction | 1 | 0.33% |
| record | late_fee: Fee assessed for missing payment deadline | 1 | 0.33% |
| record | status: Transaction approval or rejection status | 1 | 0.33% |
| record | finance_charge: Fee due to unpaid balance from previous cycle | 1 | 0.33% |
| record | location: Headquarters city and country of the company | 1 | 0.33% |
| record | lead_investor: Primary investor or consortium leading the round | 1 | 0.33% |
| record | problem_description | 1 | 0.33% |
| record | affected_site | 1 | 0.33% |
| record | affected_network | 1 | 0.33% |
| record | alternative_access | 1 | 0.33% |
| record | record_metadata | natural/name ×7, natural/status ×6, natural/title ×6, natural/location ×5 |  |

**val.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| casualty_report | location | 756 | 93.68% |
| casualty_report | injured | 373 | 46.22% |
| casualty_report | dead | 339 | 42.01% |
| casualty_report | missing | 310 | 38.41% |
| casualty_report | record_metadata | natural/dead ×242, natural/injured ×64, natural/missing ×20, natural/location ×7 |  |
| record | role | 1 | 16.67% |
| record | responsibility | 1 | 16.67% |
| record | event | 1 | 16.67% |
| record | cpt_code | 1 | 16.67% |
| record | exam_type | 1 | 16.67% |
| record | patient_name | 1 | 16.67% |
| record | patient_address | 1 | 16.67% |
| record | protocol | 1 | 16.67% |
| record | causeofloss | 1 | 16.67% |
| record | address | 1 | 16.67% |
| record | damage_amount | 1 | 16.67% |
| record | claim_id | 1 | 16.67% |
| record | donor_name | 1 | 16.67% |
| record | donor_address | 1 | 16.67% |
| record | disclosure_id | 1 | 16.67% |
| record | description | 1 | 16.67% |
| record | race | 1 | 16.67% |
| record | customer_retention | 1 | 16.67% |
| record | net_income | 1 | 16.67% |
| record | ebit_margin | 1 | 16.67% |
| record | record_metadata | natural/role ×1, natural/event ×1, natural/cpt_code ×1, natural/causeofloss ×1 |  |

**test.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| casualty_report | location | 753 | 92.51% |
| casualty_report | injured | 376 | 46.19% |
| casualty_report | missing | 334 | 41.03% |
| casualty_report | dead | 304 | 37.35% |
| casualty_report | record_metadata | natural/dead ×209, natural/injured ×81, natural/missing ×21, natural/location ×10 |  |
| record | status | 2 | 28.57% |
| record | allergies: List of known allergies with reactions | 1 | 14.29% |
| record | currentmedicationlist: List of active medications prescribed | 1 | 14.29% |
| record | comorbidities: List of additional health conditions present | 1 | 14.29% |
| record | notes: Additional clinical observations or patient-reported symptoms | 1 | 14.29% |
| record | current_status: Current clinical status of the patient's condition | 1 | 14.29% |
| record | diagnosis_date: Date when primary diagnosis was established | 1 | 14.29% |
| record | full_name: Complete name of the patient as recorded | 1 | 14.29% |
| record | insurance_provider: Name of the patient's health insurance provider | 1 | 14.29% |
| record | dateofbirth: Patient's birth date in YYYY-MM-DD format | 1 | 14.29% |
| record | primary_diagnosis: Main medical condition diagnosed | 1 | 14.29% |
| record | game_id: Unique identifier for the game as listed in the schedule | 1 | 14.29% |
| record | home_team: Name of the team playing at home | 1 | 14.29% |
| record | away_team: Name of the team playing away | 1 | 14.29% |
| record | home_score: Final score for the home team | 1 | 14.29% |
| record | away_score: Final score for the away team | 1 | 14.29% |
| record | venue: Name of the arena where the game was played | 1 | 14.29% |
| record | sender_company | 1 | 14.29% |
| record | subject | 1 | 14.29% |
| record | recipient | 1 | 14.29% |
| record | recipient_location | 1 | 14.29% |
| record | sender | 1 | 14.29% |
| record | mrn | 1 | 14.29% |
| record | patient_name | 1 | 14.29% |
| record | exam_date | 1 | 14.29% |
| record | findings | 1 | 14.29% |
| record | user_id | 1 | 14.29% |
| record | location | 1 | 14.29% |
| record | product | 1 | 14.29% |
| record | company_name | 1 | 14.29% |
| record | compliance | 1 | 14.29% |
| record | compensation_type | 1 | 14.29% |
| record | department | 1 | 14.29% |
| record | name | 1 | 14.29% |
| record | record_metadata | natural/allergies: List of known allergies with reactions ×1, natural/game_id: Unique identifier for the game as listed in the schedule ×1, natural/status ×1, natural/mrn ×1 |  |

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | unknown |
| relations | unknown |
| events | unknown |
| classifications | unknown |
| structures | unknown |
