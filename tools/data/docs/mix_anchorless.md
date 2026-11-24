# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mix_anchorless | train | 83,880 | 83,880 | 355,651 | 7,821 | 11,166 | 71,205 | 19,355 | 23,745 |
| mix_anchorless | val | 1,718 | 1,718 | 7,225 | 139 | 227 | 1,516 | 422 | 451 |
| mix_anchorless | test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## mix_anchorless

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 83,880 | 83,880 | 152 | 377 | 1913 | 355,651 | 7,821 | 11,166 | 71,205 | 19,355 | 23,745 |
| val | 1,718 | 1,718 | 157 | 370 | 1772 | 7,225 | 139 | 227 | 1,516 | 422 | 451 |
| test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 67,314 | 1,388 | 0 | 68,702 | 80.26% |
| zho | 14,054 | 279 | 0 | 14,333 | 16.74% |
| und | 1,978 | 44 | 0 | 2,022 | 2.36% |
| rus | 52 | 1 | 0 | 53 | 0.06% |
| por | 50 | 2 | 0 | 52 | 0.06% |
| spa | 52 | 0 | 0 | 52 | 0.06% |
| fra | 51 | 0 | 0 | 51 | 0.06% |
| deu | 44 | 0 | 0 | 44 | 0.05% |
| ita | 32 | 1 | 0 | 33 | 0.04% |
| nld | 27 | 0 | 0 | 27 | 0.03% |
| kor | 24 | 0 | 0 | 24 | 0.03% |
| jpn | 22 | 0 | 0 | 22 | 0.03% |
| pol | 19 | 1 | 0 | 20 | 0.02% |
| ces | 14 | 1 | 0 | 15 | 0.02% |
| ukr | 14 | 0 | 0 | 14 | 0.02% |
| tur | 12 | 0 | 0 | 12 | 0.01% |
| ara | 11 | 0 | 0 | 11 | 0.01% |
| fin | 11 | 0 | 0 | 11 | 0.01% |
| hin | 10 | 0 | 0 | 10 | 0.01% |
| swe | 9 | 0 | 0 | 9 | 0.01% |
| ron | 8 | 0 | 0 | 8 | 0.01% |
| nor | 6 | 1 | 0 | 7 | 0.01% |
| dan | 7 | 0 | 0 | 7 | 0.01% |
| hrv | 5 | 0 | 0 | 5 | 0.01% |
| ind | 5 | 0 | 0 | 5 | 0.01% |
| hun | 5 | 0 | 0 | 5 | 0.01% |
| est | 3 | 0 | 0 | 3 | 0.00% |
| ceb | 3 | 0 | 0 | 3 | 0.00% |
| lit | 3 | 0 | 0 | 3 | 0.00% |
| srp | 3 | 0 | 0 | 3 | 0.00% |
| slk | 3 | 0 | 0 | 3 | 0.00% |
| bul | 3 | 0 | 0 | 3 | 0.00% |
| vie | 3 | 0 | 0 | 3 | 0.00% |
| cym | 2 | 0 | 0 | 2 | 0.00% |
| tha | 2 | 0 | 0 | 2 | 0.00% |
| ell | 2 | 0 | 0 | 2 | 0.00% |
| bre | 2 | 0 | 0 | 2 | 0.00% |
| epo | 2 | 0 | 0 | 2 | 0.00% |
| wuu | 1 | 0 | 0 | 1 | 0.00% |
| isl | 1 | 0 | 0 | 1 | 0.00% |
| gle | 1 | 0 | 0 | 1 | 0.00% |
| fil | 1 | 0 | 0 | 1 | 0.00% |
| slv | 1 | 0 | 0 | 1 | 0.00% |
| eus | 1 | 0 | 0 | 1 | 0.00% |
| ltz | 1 | 0 | 0 | 1 | 0.00% |
| msa | 1 | 0 | 0 | 1 | 0.00% |
| mar | 1 | 0 | 0 | 1 | 0.00% |
| war | 1 | 0 | 0 | 1 | 0.00% |
| mlg | 1 | 0 | 0 | 1 | 0.00% |
| heb | 1 | 0 | 0 | 1 | 0.00% |
| lav | 1 | 0 | 0 | 1 | 0.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 0 |
| test_unique_inputs | 0 |
| train_records | 83,880 |
| train_unique_inputs | 83,880 |
| val_records | 1,718 |
| val_unique_inputs | 1,718 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.1602 WARN | 355,651 / 7,225 | e_0 7%→6%, e_3 5%→6%, e_6 4%→4%, TradedShares 1%→1%, e_12 1%→2% |
| entities (per mention) | train / test | n/a | 355,651 / 0 |  |
| entities (per mention) | val / test | n/a | 7,225 / 0 |  |
| relations (per instance) | train / val | 0.7194 STRONG | 7,821 / 139 | bind 7%→2%, PPI 4%→7%, Association 3%→0%, heritage designation 0%→2%, performer 0%→2% |
| relations (per instance) | train / test | n/a | 7,821 / 0 |  |
| relations (per instance) | val / test | n/a | 139 / 0 |  |
| event types (per trigger mention) | train / val | 0.3004 STRONG | 22,293 / 499 | Event 5%→9%, Social_event 2%→4%, Competition 2%→1%, Support 7%→5%, Becoming 1%→2% |
| event types (per trigger mention) | train / test | n/a | 22,293 / 0 |  |
| event types (per trigger mention) | val / test | n/a | 499 / 0 |  |
| event roles (per argument) | train / val | 0.1681 WARN | 25,146 / 485 | Arrester 0%→1%, Released Product 1%→0%, Quantity 0%→1%, arrested_person 1%→2%, Joined Organization 0%→1% |
| event roles (per argument) | train / test | n/a | 25,146 / 0 |  |
| event roles (per argument) | val / test | n/a | 485 / 0 |  |
| classifications (true_label) | train / val | 0.1562 WARN | 11,166 / 227 | chfinann_event: EquityRepurchase 2%→0%, chfinann_event: EquityOverweight 11%→13%, chfinann_event: EquityPledge 29%→27%, docfee_event: ExecutiveDeath 1%→2%, docfee_event: MajorAssetLoss 2%→1% |
| classifications (true_label) | train / test | n/a | 11,166 / 0 |  |
| classifications (true_label) | val / test | n/a | 227 / 0 |  |

Unseen in train, present in val, entities (per mention): Animal subgroup ×7, Auto part ×6, Operation ×4, Divine being ×4, Major ×3, credit card information ×3, game equipment ×3, codes ×3, Scent/Flavor ×3, Measurement Value ×3, Strategy/Methodology ×3, legal action ×2, Negative Experience ×2, planting method ×2, Recipe Instruction ×2, Slot Feature ×2, technical analysis term ×2, Medical Service Type ×2, Damage Type ×2, Lighting Equipment ×2, texture/physical feature ×2, game rule ×2, Financial Effect ×2, disposition ×2, parasite ×2, Menu Item ×2, musical expression ×2, hair condition ×2, award category ×2, Design Project ×2, race event ×2, animal companions ×1, combat ×1, aftermath ×1, punctuation ×1, sku_id ×1, Sales Number ×1, Market Condition ×1, Sales Range ×1, Idea/Policy ×1, legal relation ×1, health practice ×1, Scholarship Programme ×1, mishap ×1, Cashless platform ×1, Ride-hailing service ×1, Business/Financial Institution ×1, negative impact on budget ×1, critical level ×1, type of plant species ×1, packing material ×1, type of material ×1, type of flooring ×1, Record Keeping ×1, Analytical Technique ×1, Chemical Analysis ×1, plant-based substance ×1, Preparation Step ×1, Time Relation ×1, primate.morphological_analysis.0.finding ×1, primate.related_primates.0.location ×1, primate.related_primates.0.name ×1, primate.morphological_analysis.0.type ×1, primate.genus ×1, primate.dental_formula ×1, primate.weight_grams ×1, primate.relationships ×1, primate.name ×1, primate.morphological_analysis.1.finding ×1, primate.morphological_analysis.1.type ×1, primate.classification ×1, primate.fossil_origin.location ×1, primate.morphological_analysis.1.genus ×1, vocal expression ×1, Meal Plan ×1, Supplements ×1, Support Team ×1, Debriefing ×1, market research ×1, Document Format ×1, Physiological process ×1, organization.personnel_changes.0.individual.name ×1, article_metadata.source ×1, organization.personnel_changes.0.effective_date_description ×1, organization.legal_form ×1, article_metadata.origin_newsroom ×1, organization.full_name ×1, Race/Species ×1, organizational entity ×1, meeting/event ×1, guestId ×1, checkoutDate ×1, checkoutTime ×1, Work Cycle ×1, service level agreements ×1, system performance ×1, task load ×1, storage management tasks ×1, systems management integration ×1, computing system ×1, security mirroring ×1, digital interface ×1, item for sale ×1, work done for others ×1, File extension ×1, financial intermediary ×1, Society/Culture ×1, claim_number: Unique identifier for the health claim ×1, patient_name: Full name of the claimant ×1, age: Patient's age in years ×1, address: Patient's residential address ×1, diagnosis: Medical diagnosis code and description ×1, amount: Total cost of the medical service ×1, landlord ×1, pet_policy ×1, audio book service ×1, Amount of people ×1, Transportation route ×1, Alert/Notification ×1, Guardian/Protector ×1, Symbol/Indicator ×1, shooting ×1, safety rules ×1, waterway ×1, community center ×1, intersection ×1, athletics ×1, patriotism ×1, expression of approval ×1, funding ×1, Economic hardship ×1, Nation ×1, Agricultural products ×1, exoplanet ×1, planet type ×1, star type ×1, application form ×1, post office ×1, county ×1, Sustainable Energy ×1, Sustainable Resources ×1, Waste Management ×1, Consulting Services ×1, tax policy ×1, game location ×1, account: Bank account associated with the transaction ×1, description: Brief explanation of the transaction purpose ×1, claim ×1, technology tool ×1, political system ×1, form field ×1, input field ×1, output field ×1, resource limit ×1, Application Type ×1, medication adherence ×1, medication regimen ×1, natural ingredient ×1, massage technique ×1, holistic practice ×1, musical activity ×1, negative change ×1, group of post categories ×1, access level ×1, Writing genre ×1, call_details_contact ×1, call_status ×1, age milestone ×1, Football Position/Unit ×1, Quantity/Group of People ×1, Body Parts ×1, outdoor area ×1, outdoor amenity ×1, problem_start_time ×1, affected_network ×1, Religious Activity ×1, Website Component ×1, Exam/Test ×1, Score/Grade ×1, Archangel ×1, patient_id: Unique identifier for each patient visit ×1, encounter_date: Date of the patient visit ×1, diagnosis: Medical code indicating the diagnosed condition ×1, status: Current status of the encounter ×1, submittedBy ×1, dateSubmitted ×1, Industry/Service ×1, animal name ×1, Children's musical ×1, Physical Injury ×1, restaurant public relations firm ×1, re-branding ×1, executive vice president and chief creative officer ×1, Title of a book or movie ×1, Investigation Status ×1, search engine result ×1, review entity ×1, Time Unit ×1, result_id ×1, Companies ×1, Smoking device ×1, Smoke type ×1, Smoking experience ×1, insect ×1, sense organ ×1, Benefit/Feature ×1, Sport Type ×1, Development/Progress ×1, Measurement Value  ×1, genus.family ×1, genus.subspecies.0.name ×1, genus.name_latin ×1, genus.subspecies.0.authority ×1, formNumber ×1, passportNumber ×1, totalValue ×1, Quantity Entity ×1, Goal Setting ×1, Coaching Services ×1, Guidance and Support ×1, Group/People ×1, Biological Entities ×1, Business Action ×1, Coach/Recruiter ×1, Sports Ability ×1, clause_type ×1, media series ×1, biblical event ×1, evaluation metric ×1, Author/Publisher ×1, Ability/Skill ×1, Instruction/Teaching ×1, Book/Printed Material ×1, Plant/Flora ×1, Object/Lighting ×1, title: Topic or title of the submitted work ×1, decision_status: Final outcome of the submission ×1, tense ×1, expiry_date ×1, handling_instructions ×1, ppe_required ×1, travel location ×1, travel cost ×1, travel preference ×1, Business Services ×1, Service/Platform ×1, practices ×1, reviewComment ×1, reviewStatus ×1, Place/Store ×1, multiplier values ×1, free spin rewards ×1, Biome/Environment ×1, Sizing ×1, training phase ×1, recovery phase ×1, Password ×1, Password Recovery Program ×1.
Thin support in val, entities (per mention): 851 label(s) under 10 gold. real estate ×1, animal companions ×1, Animal group ×1, Countries Withdrawing from the Organization ×1, Effective Date ×1, Organization Leader ×1, Company/Brand ×1, Crew ×1, Passengers ×1, conflict ×1, combat ×1, aftermath ×1, enemy ×1, communication ×1, athlete ×1, digital content ×1, punctuation ×1, compensation_recipient ×1, sku_id ×1, last_updated ×1, product_name ×1, Sales Number ×1, Market Condition ×1, Sales Range ×1, Business Term ×1, Opinion ×1, geographic location ×1, Idea/Policy ×1, family relation ×1, legal relation ×1, health practice ×1, design style ×1, features ×1, music ×1, symptom ×1, Police ×1, Criminal Evidence ×1, Government Agency ×1, Scholarship Programme ×1, Time of day ×1, Visual media ×1, Language ×1, Room ×1, Structure ×1, substance ×1, crime ×1, mishap ×1, harm ×1, punishment ×1, service offering ×1, work environment ×1, Education/Training ×1, Digital Platform ×1, Website ×1, Cashless platform ×1, Ride-hailing service ×1, Business/Financial Institution ×1, Continent ×1, negative impact on budget ×1, critical level ×1, Military Strength ×1, type of plant species ×1, packing material ×1, type of material ×1, type of flooring ×1, entertainment ×1, Insurance ×1, Record Keeping ×1, Possession ×1, Analytical Technique ×1, Chemical Analysis ×1, Publisher ×1, plant-based substance ×1, product type ×1, Department ×1, Writing ×1, Outbreak Date ×1, Preparation Step ×1, Time Relation ×1, Music Festival ×1, TV Show ×1, skill/ability ×1, religious practice ×1, communication device ×1, request ×1, donation ×1, Demographic group ×1, facility ×1, frequency ×1, Attribute ×1, primate.morphological_analysis.0.finding ×1, primate.related_primates.0.location ×1, primate.related_primates.0.name ×1, primate.morphological_analysis.0.type ×1, primate.genus ×1, primate.dental_formula ×1, primate.weight_grams ×1, primate.relationships ×1, primate.name ×1, primate.morphological_analysis.1.finding ×1, primate.morphological_analysis.1.type ×1, primate.classification ×1, primate.fossil_origin.location ×1, primate.morphological_analysis.1.genus ×1, vocal expression ×1, Volume Number ×1, Meal Plan ×1, Supplements ×1, Support Team ×1, Debriefing ×1, Assessment ×1, marketing strategy ×1, market research ×1, reference point ×1, Transportation ×1, Evidence ×1, protein-family ×1, protein-domain ×1, Color ×1, Previous Record Holder ×1, Document Title ×1, Document Format ×1, Website Feature ×1, Spatial Location ×1, software ×1, Event Name ×1, Movie Title ×1, Degree ×1, Hardware ×1, Artistic expression ×1, Physical activity ×1, Social activity ×1, Physiological process ×1, organization.personnel_changes.0.individual.name ×1, article_metadata.source ×1, organization.personnel_changes.0.effective_date_description ×1, organization.legal_form ×1, article_metadata.origin_newsroom ×1, organization.full_name ×1, Event Organizer ×1, Countries Joined the Organization ×1, Declarer ×1, product feature ×1, Missings ×1, Number of Damaged Houses ×1, Race/Species ×1, organizational entity ×1, meeting/event ×1, component ×1, activity/event ×1, procedure ×1, company/organization ×1, guestId ×1, firstName ×1, lastName ×1, email ×1, roomId ×1, roomType ×1, capacity ×1, beds ×1, checkoutDate ×1, checkoutTime ×1, note ×1, personality trait ×1, Hazard ×1, Work Cycle ×1, award ×1, Government Reaction ×1, service level agreements ×1, system performance ×1, task load ×1, storage management tasks ×1, systems management integration ×1, software vulnerability ×1, software update ×1, computing system ×1, security mirroring ×1, vehicle ×1, written work ×1, issue ×1, media format ×1, digital interface ×1, customer type ×1, item for sale ×1, work done for others ×1, File format ×1, File extension ×1, Commodity ×1, achievement ×1, financial intermediary ×1, financial institution ×1, Society/Culture ×1, claim_number: Unique identifier for the health claim ×1, patient_name: Full name of the claimant ×1, age: Patient's age in years ×1, address: Patient's residential address ×1, diagnosis: Medical diagnosis code and description ×1, amount: Total cost of the medical service ×1, occupation/role ×1, landlord ×1, zip_code ×1, unit_id ×1, pet_policy ×1, audio book service ×1, Medical Treatment ×1, Amount of people ×1, Transportation route ×1, Alert/Notification ×1, Real estate ×1, Guardian/Protector ×1, Symbol/Indicator ×1, payment method ×1, shooting ×1, safety rules ×1, participants ×1, waterway ×1, UnfrozeDate ×1, Product Name ×1, Software Product ×1, medical facility ×1, date/time ×1, community center ×1, intersection ×1, payment ×1, system ×1, personal growth ×1, athletics ×1, patriotism ×1, family relationship ×1, expression of approval ×1, submission_date ×1, funding ×1, institution ×1, authors ×1, title ×1, Livestock ×1, Economic hardship ×1, Nation ×1, Agricultural products ×1, Government official ×1, exoplanet ×1, planet type ×1, star type ×1, unknown entity ×1, numerical value ×1, travel document ×1, application form ×1, post office ×1, county ×1, Sustainable Energy ×1, Sustainable Resources ×1, Infrastructure ×1, Waste Management ×1, Consulting Services ×1, tax policy ×1, disease ×1, equipment ×1, game item ×1, game location ×1, amount: Monetary value of the transaction ×1, account: Bank account associated with the transaction ×1, description: Brief explanation of the transaction purpose ×1, date: Date when the transaction occurred ×1, song title ×1, claim ×1, Document ×1, financial instrument ×1, TV show ×1, Medical Procedure ×1, Agreement Name ×1, Tear Up Reason ×1, Bank Name ×1, Investigating agency ×1, Weapon Used ×1, Investigating Spokesperson ×1, environment ×1, relationship issue ×1, technology tool ×1, political system ×1, Release Time ×1, form field ×1, input field ×1, contact information ×1, output field ×1, resource limit ×1, Communication ×1, Application Type ×1, Hairstyle ×1, Information ×1, Medical Practice ×1, Geographic Location ×1, book ×1, Sports team ×1, Outcome ×1, brand/product ×1, characteristic ×1, furniture ×1, mental state ×1, product model ×1, media type ×1, store ×1, medication adherence ×1, medication regimen ×1, Currency ×1, Date Range ×1, Business Name ×1, Platform ×1, natural ingredient ×1, animal type ×1, pest ×1, Product/Service ×1, massage technique ×1, holistic practice ×1, Government Body ×1, musical instrument ×1, musical activity ×1, negative change ×1, group of post categories ×1, access level ×1, idea ×1, Game Title ×1, User Interface Element ×1, family member ×1, Game ×1, Literary device ×1, Writing genre ×1, call_details_contact ×1, call_status ×1, age milestone ×1, Areas Affected ×1, Influenced People ×1, accommodation ×1, physical object ×1, decoration ×1, Job Title ×1, Winner ×1, Body Part ×1, NFL Team ×1, Football Position/Unit ×1, Quantity/Group of People ×1, Football Position ×1, Day of the Week ×1, Sports Event ×1, Qualification ×1, Fee ×1, Process ×1, Format ×1, Content ×1, Dish ×1, weather condition ×1, Body Parts ×1, Road/Highway ×1, clothing item ×1, accessory ×1, Sentencing Location ×1, Prison Term ×1, outdoor area ×1, outdoor amenity ×1, visual media ×1, Competition Items ×1, problem_start_time ×1, affected_network ×1, problem_description ×1, Education Level ×1, Religious Activity ×1, Social Group ×1, Topic ×1, Interface Element ×1, Website Component ×1, Exam/Test ×1, Score/Grade ×1, Voting method ×1, Time Span ×1, Programming Language ×1, Judgement Result/Prison Term ×1, Archangel ×1, Percentage ×1, Electoral System ×1, plot element ×1, media ×1, feedback ×1, patient_id: Unique identifier for each patient visit ×1, encounter_date: Date of the patient visit ×1, name: Full name of the patient ×1, diagnosis: Medical code indicating the diagnosed condition ×1, status: Current status of the encounter ×1, e_26 ×1, Occupation ×1, serialNumber ×1, model ×1, contact ×1, notes ×1, submittedBy ×1, dateSubmitted ×1, subscription service ×1, container ×1, Achievement ×1, Industry/Service ×1, animal name ×1, taste ×1, Venue ×1, Address ×1, Children's musical ×1, Price ×1, Physical Injury ×1, lifestyle ×1, situation ×1, capability ×1, restaurant public relations firm ×1, re-branding ×1, executive vice president and chief creative officer ×1, Title of a book or movie ×1, Ship No. ×1, Issue ×1, Investigation Status ×1, search engine result ×1, advertising strategy ×1, review entity ×1, Numeric Value ×1, Time Unit ×1, effectiveDate ×1, expirationDate ×1, result_id ×1, score ×1, Entity ×1, Industry ×1, Companies ×1, Legal case ×1, physical force ×1, Smoking device ×1, Smoke type ×1, Smoking experience ×1, Temperature control ×1, size ×1, insect ×1, anatomy ×1, sense organ ×1, Brand/Product ×1, Ingredient ×1, Product Type ×1, Benefit/Feature ×1, Sport Type ×1, Development/Progress ×1, Organization/Company ×1, Fashion Brand ×1, Country/Geopolitical entity ×1, Time entity ×1, Religious entity ×1, Clothing Brand ×1, Fashion Designer ×1, Clothing Size ×1, Clothing Measurement ×1, Profession/Role ×1, Measurement Value  ×1, genus.family ×1, genus.subspecies.0.name ×1, genus.name_latin ×1, genus.type_species ×1, genus.subspecies.0.authority ×1, Music genre ×1, Service provider ×1, formNumber ×1, destination ×1, travelPurpose ×1, passportNumber ×1, currency ×1, totalValue ×1, signature ×1, Quantity Entity ×1, Occupation Entity ×1, Location Entity ×1, Goal Setting ×1, Personal Development ×1, Coaching Services ×1, Guidance and Support ×1, Event/Tradition ×1, Business/Organization ×1, Group/People ×1, Brand/Product Name ×1, Biological Entities ×1, Substance ×1, Physical Entity ×1, Business Action ×1, ordinal number ×1, Goal ×1, Scale ×1, work ethic ×1, decision-making ×1, outcome ×1, Coach/Recruiter ×1, Sports Ability ×1, Community ×1, art form ×1, age ×1, social media platform ×1, Business Location ×1, Architecture Style ×1, Beverage ×1, reference_number ×1, jurisdiction ×1, clause_type ×1, religious text ×1, media series ×1, educational material ×1, biblical event ×1, evaluation metric ×1, Territory ×1, Skincare product ×1, expense ×1, Measurement ×1, Author/Publisher ×1, Category ×1, Physical Object ×1, Food/Drink ×1, Shopping ×1, Emotion/Feeling ×1, Ability/Skill ×1, Instruction/Teaching ×1, Book/Printed Material ×1, Person/Author ×1, Plant/Flora ×1, Object/Lighting ×1, PII ×1, day of the week ×1, title: Topic or title of the submitted work ×1, decision_status: Final outcome of the submission ×1, group/person ×1, negation ×1, tense ×1, modal verb ×1, Clothing item ×1, Clothing accessory ×1, expiry_date ×1, handling_instructions ×1, manufacturer ×1, ppe_required ×1, travel location ×1, travel cost ×1, travel preference ×1, Athlete ×1, Narrative element ×1, measurement ×1, Business Partners ×1, Business Services ×1, written communication ×1, year ×1, Service/Platform ×1, therapy ×1, practices ×1, disorder ×1, Physical sensation ×1, Mental state ×1, Physical entity ×1, Tourist Destination ×1, Culture ×1, kitchen utensil ×1, temperature ×1, cooking method ×1, Deity ×1, Comparison ×1, batchNumber ×1, reviewDate ×1, reviewComment ×1, reviewStatus ×1, Data ×1, Purpose ×1, injury ×1, dog breed ×1, physical ability ×1, Place/Store ×1, Activity/Event ×1, process ×1, information ×1, customer ×1, quality ×1, multiplier values ×1, free spin rewards ×1, game mechanics ×1, Biome/Environment ×1, Date/Time ×1, Event/Activity ×1, Holiday ×1, Social Relationship ×1, target audience ×1, Symptom ×1, Nutrient ×1, social circle ×1, physical appearance ×1, clothing ×1, Sizing ×1, Team ×1, Project ×1, Budget ×1, training phase ×1, recovery phase ×1, Password ×1, Password Recovery Program ×1, Email Client ×1, Aircraft Agency ×2, Flight No. ×2, Organization Name ×2, demographic group ×2, ethnic group ×2, Group ×2, legal action ×2, social group ×2, property ×2, Suspect ×2, Negative Experience ×2, Day of the week ×2, Ethnic Group ×2, mechanical component ×2, Company/Organization ×2, Media ×2, Region ×2, planting method ×2, Documentation ×2, File Format ×2, Volcano Name ×2, Recipe Instruction ×2, Publication ×2, Natural Feature ×2, Accused People ×2, experimental-construct ×2, Record-breaking Project ×2, Grades ×2, Slot Feature ×2, Input Device ×2, Software ×2, Time/Date ×2, mechanism ×2, Survivors ×2, experience ×2, Insurance Policy ×2, Belligerents ×2, Weapon and Equippment ×2, mode of transportation ×2, service provider ×2, Crafting Material ×2, state ×2, author ×2, Medication ×2, Weather condition ×2, Animal ×2, body of water ×2, medical treatment ×2, personal trait ×2, government agency ×2, group ×2, Chemical compound ×2, technical analysis term ×2, religion ×2, belief system ×2, Agreement Content ×2, topic ×2, Released People ×2, Release Reason ×2, Work Schedule ×2, Medical Service Type ×2, Sport ×2, Status ×2, University ×2, architecture ×2, Damage Type ×2, pronoun ×2, Character ×2, e_24 ×2, e_25 ×2, task/action ×2, pledge_start_date ×2, pledge_end_date ×2, Award ×2, Award Field ×2, Lighting Equipment ×2, Material ×2, Gene ×2, texture/physical feature ×2, hardware ×2, medication ×2, room ×2, Game Name ×2, Start Time ×2, Contest Participant ×2, game rule ×2, Time of Day ×2, Accusation ×2, Media Outlet ×2, Information Service ×2, Financial Effect ×2, employee ×2, type ×2, image ×2, disposition ×2, Company Industry ×2, meal type ×2, Sound ×2, City ×2, parasite ×2, Demographic ×2, Application ×2, nationality ×2, quantity ×2, Art/Culture ×2, Freeze End Date ×2, Army ×2, Military Exercise ×2, Business ×2, Menu Item ×2, parties ×2, System ×2, Money ×2, musical expression ×2, Clothing feature ×2, Person/Group ×2, Football Club ×2, Feature ×2, Interactive element ×2, feeling/emotion ×2, behavior/action ×2, action/behavior ×2, id ×2, game interface ×2, Entertainment ×2, time frame ×2, Aid Agency ×2, Doctor and Medical Team ×2, hair condition ×2, award category ×2, color ×2, Design Project ×2, race event ×2, Scheduled Landing Place ×3, age group ×3, The Charged Crime ×3, Time period ×3, Accommodation ×3, tool ×3, Educational Institution ×3, Major ×3, measurement unit ×3, profession ×3, Person Name ×3, experiment-tag ×3, Record Breaker ×3, Loss Amount ×3, Entity Type ×3, Affected Areas ×3, attribute ×3, emotion ×3, Time Period ×3, Product/Brand ×3, Increase Start Date ×3, credit card information ×3, Author ×3, address ×3, Medical Practitioner ×3, Judge ×3, Death Age ×3, time ×3, Transportation Service ×3, Legal Document ×3, healthcare service ×3, amenity ×3, Award Reason ×3, weather phenomenon ×3, natural feature ×3, physical space ×3, game equipment ×3, Election Goal ×3, Device ×3, codes ×3, units ×3, code ×3, unit ×3, amount ×3, Tourist Attraction ×3, astronomical object ×3, Scent/Flavor ×3, Freeze Start Date ×3, frozen_shareholder ×3, Measurement Value ×3, Physical Attribute ×3, value ×3, Strategy/Methodology ×3, ingredient ×3, People ×3, Attacker ×4, occupation ×4, city ×4, protein ×4, food item ×4, Operation ×4, Policy Content ×4, Deliberating Agency ×4, group of people ×4, Protest Reason ×4, Protesters ×4, Increase Amount ×4, Increasing Shareholder ×4, Emotion ×4, Food ×4, Quantity ×4, Geographical Location ×4, The Agency Who Broke the Agreement ×4, e_23 ×4, object ×4, reagent ×4, holiday decoration ×4, Candidates and their Political Parties ×4, Divine being ×4, Frozen Amount ×4, Malware ×4, activity type ×4, Target ×5, chemical ×5, Damaged Facility ×5, service ×5, ruling_date ×5, Business Entity ×5, Weapon ×5, FrozeShares ×5, LegalInstitution ×5, technology ×5, Service ×5, Contracting Parties ×5, Death Time ×5, senior_personnel ×5, senior_position ×5, food ×5, description ×5, Capabilities ×5, Year ×6, event ×6, Country ×6, Auto part ×6, animal ×6, Activity ×6, name ×6, Animal subgroup ×7, e_22 ×7, Charged Crime ×7, Music Artist ×7, Recipient ×8, pledgor ×8, Accident Investigator ×8, Technology ×8, Sensory Perception ×8, Entity type ×8, Product ×9, activity ×9, Action ×9.
Unseen in train, present in val, relations (per instance): peak bagging classification ×1.
Thin support in val, relations (per instance): 107 label(s) under 10 gold. separated from ×1, worshipped by ×1, significant person ×1, position held ×1, copyright license ×1, crew member(s) ×1, enemy of ×1, base ×1, co-driver ×1, game mode ×1, produced by ×1, manner of death ×1, significant event ×1, field of this occupation ×1, writing language ×1, peak bagging classification ×1, executive body ×1, parent taxon ×1, sport ×1, sponsor ×1, platform ×1, unmarried partner ×1, located in the administrative territorial entity ×1, reservoir created ×1, located on astronomical body ×1, movement ×1, award received ×1, diplomatic relation ×1, color ×1, sports discipline competed in ×1, stock market index ×1, noble title ×1, military rank ×1, fictional universe described in ×1, cell component ×1, member of political party ×1, mouth of the watercourse ×1, presenter ×1, characters ×1, convicted of ×1, office held by head of government ×1, has use ×1, tributary ×1, architect ×1, drug or therapy used for treatment ×1, IUCN conservation status ×1, participant in ×1, measurement scale ×1, position played on team / speciality ×1, has subsidiary ×1, authority ×1, creator ×1, approved by ×1, voice actor ×1, child ×1, organization directed by the office or position ×1, officially opened by ×1, indigenous to ×1, conferred by ×1, applies to jurisdiction ×1, crosses ×1, member of the crew of ×1, next higher rank ×1, place of birth ×1, defendant ×1, employer ×1, follows ×1, director ×1, interchange station ×1, youth wing ×1, taxonomic type ×1, connects with ×1, notable work ×1, instrument ×1, different from ×1, voice type ×1, dedicated to ×1, author ×1, publisher ×1, space launch vehicle ×1, candidacy in election ×1, named after ×1, distributed by ×1, producer ×1, location ×1, original broadcaster ×1, dam ×1, cause of death ×1, taxon rank ×2, inspired by ×2, less than ×2, product or material produced ×2, from narrative universe ×2, licensed to broadcast to ×2, present in work ×2, model item ×2, first appearance ×2, place served by transport hub ×2, ammunition ×2, occupation ×2, parent organization ×2, endemic to ×2, medical condition treated ×2, country for sport ×2, heritage designation ×3, performer ×3, bind ×3.
Thin support in val, event types (per trigger mention): 121 label(s) under 10 gold. judicial_action_summon ×1, Exchange ×1, Getting ×1, Becoming_a_member ×1, Releasing ×1, Surrounding ×1, Cause_to_amalgamate ×1, Besieging ×1, Escaping ×1, march ×1, Plane Crash ×1, Arrest ×1, Theft ×1, Criminal_investigation ×1, Committing_crime ×1, Reveal_secret ×1, Expressing_publicly ×1, price_increase ×1, Telling ×1, Arranging ×1, Commerce_sell ×1, Deciding ×1, praise ×1, Supply ×1, Institutionalization ×1, Testing ×1, Violence ×1, product_delist ×1, lost_contact ×1, Connect ×1, Removing ×1, Convincing ×1, strike ×1, ipo ×1, Earthquake ×1, layoff ×1, Preserving ×1, Dispersal ×1, proposal ×1, Expansion ×1, Come_together ×1, Fire ×1, Limiting ×1, Traveling ×1, opening ×1, wedding ×1, product_release ×2, marriage ×2, Agree_or_refuse_to_act ×2, Change_sentiment ×2, Perception_active ×2, Control ×2, Presence ×2, Action ×2, Filling ×2, Bodily_harm ×2, sale_acquisition ×2, Judgment_communication ×2, Warning ×2, Coming_to_believe ×2, Departing ×2, GiveUp ×2, Creating ×2, Reforming_a_system ×2, price_limit_down ×2, product_publish ×2, Achieve ×2, death ×2, Cyber.Databreach ×2, Giving ×2, financing ×3, birth ×3, Hold ×3, Destroying ×3, Communication ×3, Preventing_or_letting ×3, Name_conferral ×3, Recording ×3, Competition ×3, championship_win ×3, Death ×3, Conquering ×3, Accident ×3, Aiming ×3, Using ×3, dismissal ×3, resignation ×3, Reporting ×3, Change ×3, Participation ×3, Cause_change_of_strength ×3, Process_end ×3, Damaging ×3, Conflict ×3, Earnings_and_losses ×4, Manufacturing ×4, Motion_directional ×4, Know ×4, Self_motion ×4, Being_in_operation ×4, Influence ×4, Manoeuvre ×4, Publishing ×4, joined ×5, Supporting ×5, Hostile_encounter ×5, Motion ×5, Cause_change_of_position_on_a_scale ×6, Injure ×6, Terrorism ×6, Coming_to_be ×6, Cause_to_be_included ×7, Process_start ×7, Attack ×7, Exhibit ×7, Cyber.Ransom ×7, Killing ×8, judicial_action_arrest ×8, match_result ×8, Statement ×9, Becoming ×9.
Thin support in val, event roles (per argument): 56 label(s) under 10 gold. lead_investor ×1, Interview Initiator ×1, Interview Subject ×1, dismissal_party ×1, Sale Price ×1, Price Increased Item ×1, Price Increase Rate ×1, Liker ×1, Like Target ×1, Delisted Product ×1, Affiliated Organization ×1, Striker ×1, Listed Company ×1, magnitude ×1, layoff_party ×1, Proposer ×1, Proposal Target ×1, Purpose ×1, Activity Name ×1, Released Media ×2, Funding Round ×2, Result ×2, Acquirer ×2, Transaction Item ×2, Original Organization ×2, limit_down_stock ×2, Publisher ×2, Released Product ×2, Deceased ×2, Time ×2, Price ×2, Compromised-Data ×2, Financing Party ×3, Funding Amount ×3, Giver Birth ×3, Born Person ×3, Champion ×3, Championship Event ×3, dismissed_employee ×3, event_name ×3, Departing Employee ×3, Area ×3, Missing Person ×3, Marriage Parties ×4, Franchisee ×4, Content ×4, Attacker ×4, Tool ×4, loser ×5, Attack-Pattern ×5, Joined Organization ×6, Quantity ×6, Arrester ×7, Victim ×7, arrested_person ×8, Winner ×9.
Thin support in val, classifications (true_label): 40 label(s) under 10 gold. docee_event: Withdraw from an Orgnization ×1, docee_event: CommitCrime - Arrest ×1, docee_event: Volcano Eruption ×1, docee_event: CommitCrime - Accuse ×1, docee_event: Mine Collapses ×1, docee_event: Join in an Orgnization ×1, docee_event: Floods ×1, docee_event: Protest ×1, docee_event: Riot ×1, docee_event: Tear Up Agreement ×1, docee_event: Bank Robbery ×1, docee_event: Droughts ×1, docee_event: Awards Ceremony ×1, docee_event: Sports Competition ×1, docee_event: CommitCrime - Sentence ×1, docee_event: Shipwreck ×1, docee_event: Military Excercise ×1, docee_event: Fire ×1, docee_event: Famous Person - Sick ×1, docfee_event: MajorExternalPayout ×1, biotech_event: company description ×1, biotech_event: closing ×1, biotech_event: executive statement ×1, docee_event: Break Historical Records ×2, docee_event: Government Policy Changes ×2, docee_event: Sign Agreement ×2, docee_event: CommitCrime - Release ×2, docee_event: Election ×2, docee_event: Gas Explosion ×2, docfee_event: MajorAssetLoss ×2, docfee_event: EquityOverweight ×2, biotech_event: service & product providing ×2, docee_event: Air Crash ×3, docee_event: Road Crash ×3, docfee_event: MajorSafetyAccident ×3, docfee_event: EquityFreeze ×3, docee_event: Armed Conflict ×4, chfinann_event: EquityFreeze ×5, docfee_event: ExecutiveDeath ×5, docfee_event: EquityPledge ×6.

### Events

**train.** 21,532 instances in 6,454 records, 0 triggerless. Trigger gold 19,355. Argument gold 23,745. Arguments per instance, mean 1.17. Trigger length in whitespace tokens, mean 1.45, single-token share 59.2%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Experiment | 1,972 | 8.85% |
| Support | 1,484 | 6.66% |
| Event | 1,127 | 5.06% |
| Deploy | 652 | 2.92% |
| Catastrophe | 629 | 2.82% |
| Competition | 534 | 2.40% |
| match_result | 504 | 2.26% |
| Causation | 443 | 1.99% |
| Hostile_encounter | 438 | 1.96% |
| Process_start | 426 | 1.91% |
| Attack | 420 | 1.88% |
| Exhibit | 392 | 1.76% |
| Social_event | 375 | 1.68% |
| Motion | 355 | 1.59% |
| product_publish | 310 | 1.39% |
| Killing | 272 | 1.22% |
| Process_end | 231 | 1.04% |
| Self_motion | 205 | 0.92% |
| Conquering | 202 | 0.91% |
| death | 200 | 0.90% |
| Accident | 192 | 0.86% |
| judicial_action_arrest | 191 | 0.86% |
| Coming_to_be | 191 | 0.86% |
| Statement | 190 | 0.85% |
| Creating | 180 | 0.81% |
| Arriving | 178 | 0.80% |
| Hold | 177 | 0.79% |
| Destroying | 175 | 0.78% |
| resignation | 166 | 0.74% |
| Cyber.DiscoverVulnerability | 166 | 0.74% |
| Manoeuvre | 163 | 0.73% |
| Damaging | 160 | 0.72% |
| Death | 154 | 0.69% |
| Cause_change_of_strength | 154 | 0.69% |
| Bodily_harm | 152 | 0.68% |
| Becoming | 139 | 0.62% |
| Traveling | 139 | 0.62% |
| Cyber.PatchVulnerability | 135 | 0.61% |
| Arranging | 132 | 0.59% |
| Military_operation | 128 | 0.57% |
| Influence | 125 | 0.56% |
| Getting | 124 | 0.56% |
| Presence | 123 | 0.55% |
| Earnings_and_losses | 122 | 0.55% |
| Cause_to_be_included | 121 | 0.54% |
| Action | 121 | 0.54% |
| Cause_change_of_position_on_a_scale | 120 | 0.54% |
| Participation | 119 | 0.53% |
| championship_win | 118 | 0.53% |
| Control | 117 | 0.52% |
| Defending | 112 | 0.50% |
| Recording | 105 | 0.47% |
| Preventing_or_letting | 101 | 0.45% |
| Know | 101 | 0.45% |
| Injure | 99 | 0.44% |
| Change | 99 | 0.44% |
| Motion_directional | 98 | 0.44% |
| joined | 95 | 0.43% |
| Cyber.Databreach | 92 | 0.41% |
| Name_conferral | 91 | 0.41% |
| Placing | 89 | 0.40% |
| Judgment_communication | 88 | 0.39% |
| Escaping | 88 | 0.39% |
| Expressing_publicly | 87 | 0.39% |
| marriage | 83 | 0.37% |
| Reporting | 83 | 0.37% |
| Giving | 82 | 0.37% |
| Conflict | 81 | 0.36% |
| Communication | 80 | 0.36% |
| Cyber.Phishing | 80 | 0.36% |
| product_release | 79 | 0.35% |
| Request | 78 | 0.35% |
| product_recall | 77 | 0.35% |
| advancement | 77 | 0.35% |
| Agree_or_refuse_to_act | 76 | 0.34% |
| Change_of_leadership | 76 | 0.34% |
| divorce | 73 | 0.33% |
| Using | 71 | 0.32% |
| Removing | 71 | 0.32% |
| judicial_action_summon | 70 | 0.31% |
| Cyber.Ransom | 69 | 0.31% |
| Supply | 67 | 0.30% |
| Supporting | 67 | 0.30% |
| judicial_action_fine | 67 | 0.30% |
| Sending | 67 | 0.30% |
| opening | 66 | 0.30% |
| Manufacturing | 63 | 0.28% |
| Bringing | 61 | 0.27% |
| Use_firearm | 57 | 0.26% |
| Traffic Accident | 55 | 0.25% |
| price_limit_up | 53 | 0.24% |
| Violence | 53 | 0.24% |
| Choosing | 52 | 0.23% |
| product_delist | 51 | 0.23% |
| judicial_action_prosecution | 49 | 0.22% |
| Departing | 49 | 0.22% |
| Fire | 48 | 0.22% |
| Dispersal | 48 | 0.22% |
| Cause_to_make_progress | 46 | 0.21% |
| Building | 46 | 0.21% |
| Achieve | 45 | 0.20% |
| Deciding | 44 | 0.20% |
| Aiming | 44 | 0.20% |
| sale_acquisition | 44 | 0.20% |
| Come_together | 43 | 0.19% |
| birthday_celebration | 43 | 0.19% |
| Being_in_operation | 43 | 0.19% |
| Assistance | 42 | 0.19% |
| Collaboration | 42 | 0.19% |
| layoff | 42 | 0.19% |
| judicial_action_imprisonment | 41 | 0.18% |
| product_award | 40 | 0.18% |
| competition_withdrawal | 40 | 0.18% |
| Cause_to_amalgamate | 40 | 0.18% |
| departure | 40 | 0.18% |
| apologize_to | 39 | 0.17% |
| Perception_active | 39 | 0.17% |
| Response | 39 | 0.17% |
| competition_ban | 38 | 0.17% |
| Preserving | 37 | 0.17% |
| Receiving | 37 | 0.17% |
| Scrutiny | 36 | 0.16% |
| Coming_to_believe | 36 | 0.16% |
| Committing_crime | 36 | 0.16% |
| Arrest | 36 | 0.16% |
| Publishing | 36 | 0.16% |
| Terrorism | 35 | 0.16% |
| price_limit_down | 35 | 0.16% |
| Check | 34 | 0.15% |
| Expansion | 34 | 0.15% |
| Criminal_investigation | 32 | 0.14% |
| Change_event_time | 32 | 0.14% |
| Surrendering | 32 | 0.14% |
| Becoming_a_member | 31 | 0.14% |
| Besieging | 31 | 0.14% |
| Earthquake | 31 | 0.14% |
| breakup | 31 | 0.14% |
| Openness | 29 | 0.13% |
| Recovering | 29 | 0.13% |
| lost_contact | 29 | 0.13% |
| Hindering | 29 | 0.13% |
| Change_sentiment | 27 | 0.12% |
| Temporary_stay | 26 | 0.12% |
| retirement | 26 | 0.12% |
| Convincing | 26 | 0.12% |
| Warning | 26 | 0.12% |
| judicial_action_report | 25 | 0.11% |
| Rescuing | 25 | 0.11% |
| Education_teaching | 24 | 0.11% |
| price_decrease | 24 | 0.11% |
| Legal_rulings | 24 | 0.11% |
| Collapse | 23 | 0.10% |
| judicial_action_hearing | 23 | 0.10% |
| suspension | 23 | 0.10% |
| Quarreling | 22 | 0.10% |
| Plane Crash | 22 | 0.10% |
| GiveUp | 22 | 0.10% |
| financing | 22 | 0.10% |
| birth | 21 | 0.09% |
| Commerce_sell | 21 | 0.09% |
| Employment | 21 | 0.09% |
| Explosion | 21 | 0.09% |
| dismissal | 21 | 0.09% |
| strike | 21 | 0.09% |
| Reveal_secret | 20 | 0.09% |
| dissolution | 20 | 0.09% |
| Emptying | 20 | 0.09% |
| Sign_agreement | 20 | 0.09% |
| Protest | 20 | 0.09% |
| Connect | 19 | 0.09% |
| GetReady | 19 | 0.09% |
| Expend_resource | 19 | 0.09% |
| meet_with | 19 | 0.09% |
| Body_movement | 19 | 0.09% |
| Reforming_a_system | 19 | 0.09% |
| judicial_action_filing | 18 | 0.08% |
| pregnancy | 18 | 0.08% |
| closing | 18 | 0.08% |
| engagement | 18 | 0.08% |
| Telling | 17 | 0.08% |
| Research | 17 | 0.08% |
| praise | 16 | 0.07% |
| visit_on_set | 16 | 0.07% |
| Limiting | 16 | 0.07% |
| Resolve_problem | 16 | 0.07% |
| wedding | 16 | 0.07% |
| Testing | 16 | 0.07% |
| proposal | 15 | 0.07% |
| Releasing | 15 | 0.07% |
| march | 15 | 0.07% |
| Surrounding | 15 | 0.07% |
| Hiding_objects | 15 | 0.07% |
| price_increase | 14 | 0.06% |
| Commitment | 14 | 0.06% |
| thank | 14 | 0.06% |
| Legality | 14 | 0.06% |
| Having_or_lacking_access | 13 | 0.06% |
| ipo | 13 | 0.06% |
| Rewards_and_punishments | 13 | 0.06% |
| Flood | 12 | 0.05% |
| contract_termination | 11 | 0.05% |
| Bearing_arms | 11 | 0.05% |
| Submitting_documents | 10 | 0.04% |
| Confronting_problem | 10 | 0.04% |
| Writing | 10 | 0.04% |
| Filling | 10 | 0.04% |
| Patrolling | 10 | 0.04% |
| Forming_relationships | 10 | 0.04% |
| Exchange | 9 | 0.04% |
| Practice | 9 | 0.04% |
| Award | 9 | 0.04% |
| Adducing | 9 | 0.04% |
| Kidnapping | 9 | 0.04% |
| Commerce_buy | 9 | 0.04% |
| Wearing | 9 | 0.04% |
| infidelity | 9 | 0.04% |
| Imposing_obligation | 8 | 0.04% |
| Lighting | 8 | 0.04% |
| Rite | 7 | 0.03% |
| Containing | 7 | 0.03% |
| Robbery | 7 | 0.03% |
| Cure | 7 | 0.03% |
| Prison | 7 | 0.03% |
| rate_increase | 7 | 0.03% |
| Commerce_pay | 6 | 0.03% |
| Labeling | 6 | 0.03% |
| Revenge | 6 | 0.03% |
| Cost | 6 | 0.03% |
| Create_artwork | 6 | 0.03% |
| Scouring | 6 | 0.03% |
| rate_decrease | 5 | 0.02% |
| Institutionalization | 4 | 0.02% |
| Carry_goods | 4 | 0.02% |
| Theft | 3 | 0.01% |
| Extradition | 3 | 0.01% |
| Justifying | 3 | 0.01% |
| Suspicion | 3 | 0.01% |
| Ratification | 3 | 0.01% |
| Ingestion | 2 | 0.01% |
| Renting | 2 | 0.01% |
| Vocalizations | 2 | 0.01% |
| Risk | 1 | 0.00% |
| Emergency | 1 | 0.00% |

**val.** 469 instances in 128 records, 0 triggerless. Trigger gold 422. Argument gold 451. Arguments per instance, mean 1.03. Trigger length in whitespace tokens, mean 1.37, single-token share 65.1%.

#### val event types

| label | count | share |
| --- | --- | --- |
| Event | 43 | 8.62% |
| Experiment | 42 | 8.42% |
| Support | 26 | 5.21% |
| Social_event | 22 | 4.41% |
| Catastrophe | 19 | 3.81% |
| Causation | 13 | 2.61% |
| Deploy | 11 | 2.20% |
| Statement | 9 | 1.80% |
| Becoming | 9 | 1.80% |
| Killing | 8 | 1.60% |
| judicial_action_arrest | 8 | 1.60% |
| match_result | 8 | 1.60% |
| Cause_to_be_included | 7 | 1.40% |
| Process_start | 7 | 1.40% |
| Attack | 7 | 1.40% |
| Exhibit | 7 | 1.40% |
| Cyber.Ransom | 7 | 1.40% |
| Cause_change_of_position_on_a_scale | 6 | 1.20% |
| Injure | 6 | 1.20% |
| Terrorism | 6 | 1.20% |
| Coming_to_be | 6 | 1.20% |
| joined | 5 | 1.00% |
| Supporting | 5 | 1.00% |
| Hostile_encounter | 5 | 1.00% |
| Motion | 5 | 1.00% |
| Earnings_and_losses | 4 | 0.80% |
| Manufacturing | 4 | 0.80% |
| Motion_directional | 4 | 0.80% |
| Know | 4 | 0.80% |
| Self_motion | 4 | 0.80% |
| Being_in_operation | 4 | 0.80% |
| Influence | 4 | 0.80% |
| Manoeuvre | 4 | 0.80% |
| Publishing | 4 | 0.80% |
| financing | 3 | 0.60% |
| birth | 3 | 0.60% |
| Hold | 3 | 0.60% |
| Destroying | 3 | 0.60% |
| Communication | 3 | 0.60% |
| Preventing_or_letting | 3 | 0.60% |
| Name_conferral | 3 | 0.60% |
| Recording | 3 | 0.60% |
| Competition | 3 | 0.60% |
| championship_win | 3 | 0.60% |
| Death | 3 | 0.60% |
| Conquering | 3 | 0.60% |
| Accident | 3 | 0.60% |
| Aiming | 3 | 0.60% |
| Using | 3 | 0.60% |
| dismissal | 3 | 0.60% |
| resignation | 3 | 0.60% |
| Reporting | 3 | 0.60% |
| Change | 3 | 0.60% |
| Participation | 3 | 0.60% |
| Cause_change_of_strength | 3 | 0.60% |
| Process_end | 3 | 0.60% |
| Damaging | 3 | 0.60% |
| Conflict | 3 | 0.60% |
| product_release | 2 | 0.40% |
| marriage | 2 | 0.40% |
| Agree_or_refuse_to_act | 2 | 0.40% |
| Change_sentiment | 2 | 0.40% |
| Perception_active | 2 | 0.40% |
| Control | 2 | 0.40% |
| Presence | 2 | 0.40% |
| Action | 2 | 0.40% |
| Filling | 2 | 0.40% |
| Bodily_harm | 2 | 0.40% |
| sale_acquisition | 2 | 0.40% |
| Judgment_communication | 2 | 0.40% |
| Warning | 2 | 0.40% |
| Coming_to_believe | 2 | 0.40% |
| Departing | 2 | 0.40% |
| GiveUp | 2 | 0.40% |
| Creating | 2 | 0.40% |
| Reforming_a_system | 2 | 0.40% |
| price_limit_down | 2 | 0.40% |
| product_publish | 2 | 0.40% |
| Achieve | 2 | 0.40% |
| death | 2 | 0.40% |
| Cyber.Databreach | 2 | 0.40% |
| Giving | 2 | 0.40% |
| judicial_action_summon | 1 | 0.20% |
| Exchange | 1 | 0.20% |
| Getting | 1 | 0.20% |
| Becoming_a_member | 1 | 0.20% |
| Releasing | 1 | 0.20% |
| Surrounding | 1 | 0.20% |
| Cause_to_amalgamate | 1 | 0.20% |
| Besieging | 1 | 0.20% |
| Escaping | 1 | 0.20% |
| march | 1 | 0.20% |
| Plane Crash | 1 | 0.20% |
| Arrest | 1 | 0.20% |
| Theft | 1 | 0.20% |
| Criminal_investigation | 1 | 0.20% |
| Committing_crime | 1 | 0.20% |
| Reveal_secret | 1 | 0.20% |
| Expressing_publicly | 1 | 0.20% |
| price_increase | 1 | 0.20% |
| Telling | 1 | 0.20% |
| Arranging | 1 | 0.20% |
| Commerce_sell | 1 | 0.20% |
| Deciding | 1 | 0.20% |
| praise | 1 | 0.20% |
| Supply | 1 | 0.20% |
| Institutionalization | 1 | 0.20% |
| Testing | 1 | 0.20% |
| Violence | 1 | 0.20% |
| product_delist | 1 | 0.20% |
| lost_contact | 1 | 0.20% |
| Connect | 1 | 0.20% |
| Removing | 1 | 0.20% |
| Convincing | 1 | 0.20% |
| strike | 1 | 0.20% |
| ipo | 1 | 0.20% |
| Earthquake | 1 | 0.20% |
| layoff | 1 | 0.20% |
| Preserving | 1 | 0.20% |
| Dispersal | 1 | 0.20% |
| proposal | 1 | 0.20% |
| Expansion | 1 | 0.20% |
| Come_together | 1 | 0.20% |
| Fire | 1 | 0.20% |
| Limiting | 1 | 0.20% |
| Traveling | 1 | 0.20% |
| opening | 1 | 0.20% |
| wedding | 1 | 0.20% |

**test.** No event instances.

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 25,146 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 485 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 0 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.74 | 31.04% | 1.62 | 27.31% | 0.04% | 0 | 25,146 | 0 | 22,293 |
| val | 1.89 | 33.87% | 1.81 | 32.85% | 0.10% | 0 | 485 | 0 | 499 |
| test | 0.00 | n/a | 0.00 | n/a | n/a | 0 | 0 | 0 | 0 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 21,532 | 1.31% | 14,409 | 14.30% | 13.93% | 1.78% | 0.74% | 14.66% |
| val | 469 | 2.35% | 335 | 11.34% | 11.34% | 0.90% | 0.30% | 14.06% |
| test | 0 | n/a | 0 | n/a | n/a | n/a | n/a | n/a |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 282 | 1,712 | 14,409 |
| val | 11 | 62 | 335 |
| test | 0 | 0 | 0 |

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
| e_0 | 23,393 | 6.58% |
| e_1 | 22,011 | 6.19% |
| e_2 | 20,960 | 5.89% |
| e_3 | 19,377 | 5.45% |
| e_4 | 17,712 | 4.98% |
| e_5 | 15,846 | 4.46% |
| e_6 | 14,133 | 3.97% |
| e_7 | 12,271 | 3.45% |
| e_8 | 10,548 | 2.97% |
| e_9 | 8,751 | 2.46% |
| StartDate | 7,689 | 2.16% |
| e_10 | 7,299 | 2.05% |
| EquityHolder | 7,135 | 2.01% |
| StockCode | 6,261 | 1.76% |
| e_11 | 6,126 | 1.72% |
| CompanyName | 5,263 | 1.48% |
| StockAbbr | 5,235 | 1.47% |
| e_12 | 4,956 | 1.39% |
| PledgedShares | 4,955 | 1.39% |
| Pledgee | 4,161 | 1.17% |
| TotalHoldingShares | 4,054 | 1.14% |
| e_13 | 3,996 | 1.12% |
| TotalHoldingRatio | 3,711 | 1.04% |
| Location | 3,696 | 1.04% |
| TradedShares | 3,475 | 0.98% |
| e_14 | 3,211 | 0.90% |
| TotalPledgedShares | 2,841 | 0.80% |
| e_15 | 2,502 | 0.70% |
| Person | 2,417 | 0.68% |
| EndDate | 2,372 | 0.67% |
| ReleasedDate | 2,261 | 0.64% |
| e_16 | 1,951 | 0.55% |
| Date | 1,905 | 0.54% |
| Reduction Amount | 1,578 | 0.44% |
| e_17 | 1,506 | 0.42% |
| Casualties and Losses | 1,180 | 0.33% |
| LaterHoldingShares | 1,179 | 0.33% |
| Reducing Shareholder | 1,175 | 0.33% |
| e_18 | 1,132 | 0.32% |
| Company Name | 1,122 | 0.32% |
| Organization | 1,032 | 0.29% |
| Reduction Start Date | 987 | 0.28% |
| Announcement Date | 896 | 0.25% |
| e_19 | 871 | 0.24% |
| AveragePrice | 859 | 0.24% |
| location | 777 | 0.22% |
| Company | 728 | 0.20% |
| Event | 705 | 0.20% |
| Product | 654 | 0.18% |
| e_20 | 644 | 0.18% |
| person | 639 | 0.18% |
| pledge_amount | 579 | 0.16% |
| Time | 538 | 0.15% |
| e_21 | 485 | 0.14% |
| action | 453 | 0.13% |
| Accepting Court | 434 | 0.12% |
| product | 426 | 0.12% |
| Cause | 417 | 0.12% |
| pledgor | 407 | 0.11% |
| name | 404 | 0.11% |
| organization | 403 | 0.11% |
| Action | 393 | 0.11% |
| Recipient | 382 | 0.11% |
| protein | 374 | 0.11% |
| e_22 | 371 | 0.10% |
| Activity | 357 | 0.10% |
| Country | 340 | 0.10% |
| Individual_protein | 336 | 0.09% |
| pledge_start_date | 323 | 0.09% |
| HighestTradingPrice | 321 | 0.09% |
| LegalInstitution | 305 | 0.09% |
| activity | 296 | 0.08% |
| Technology | 291 | 0.08% |
| RepurchasedShares | 288 | 0.08% |
| RepurchaseAmount | 286 | 0.08% |
| Object | 286 | 0.08% |
| Increase Amount | 283 | 0.08% |
| event | 281 | 0.08% |
| Frozen Amount | 275 | 0.08% |
| ruling_date | 270 | 0.08% |
| object | 268 | 0.08% |
| time period | 260 | 0.07% |
| e_23 | 257 | 0.07% |
| technology | 257 | 0.07% |
| FrozeShares | 256 | 0.07% |
| Service | 253 | 0.07% |
| Place | 247 | 0.07% |
| ClosingDate | 246 | 0.07% |
| Loss Amount | 244 | 0.07% |
| GeneOrGeneProduct | 237 | 0.07% |
| Increasing Shareholder | 230 | 0.06% |
| pledge_end_date | 219 | 0.06% |
| Time period | 212 | 0.06% |
| Increase Start Date | 207 | 0.06% |
| Attacker | 206 | 0.06% |
| Occupation | 202 | 0.06% |
| DiseaseOrPhenotypicFeature | 202 | 0.06% |
| Vulnerability | 202 | 0.06% |
| e_24 | 200 | 0.06% |
| LowestTradingPrice | 200 | 0.06% |
| Target | 197 | 0.06% |
| Weapon and Equippment | 197 | 0.06% |
| frozen_shareholder | 195 | 0.05% |
| profession | 192 | 0.05% |
| medical condition | 186 | 0.05% |
| Gene/protein/RNA | 184 | 0.05% |
| City | 173 | 0.05% |
| Group | 167 | 0.05% |
| Freeze Start Date | 165 | 0.05% |
| Position | 164 | 0.05% |
| Material | 163 | 0.05% |
| Charged Crime | 158 | 0.04% |
| service | 153 | 0.04% |
| Accident Investigator | 150 | 0.04% |
| Affected Areas | 148 | 0.04% |
| Educational Institution | 148 | 0.04% |
| e_25 | 144 | 0.04% |
| Business | 143 | 0.04% |
| ingredient | 142 | 0.04% |
| Suspect | 136 | 0.04% |
| emotion | 136 | 0.04% |
| food | 135 | 0.04% |
| date | 133 | 0.04% |
| body part | 132 | 0.04% |
| material | 132 | 0.04% |
| Protest Reason | 130 | 0.04% |
| Quantity | 129 | 0.04% |
| Sports Team | 127 | 0.04% |
| status | 126 | 0.04% |
| description | 126 | 0.04% |
| time | 123 | 0.03% |
| attribute | 122 | 0.03% |
| Software | 121 | 0.03% |
| Year | 120 | 0.03% |
| animal | 119 | 0.03% |
| Protesters | 118 | 0.03% |
| Accused People | 118 | 0.03% |
| Affected Area | 117 | 0.03% |
| Candidates and their Political Parties | 117 | 0.03% |
| process | 117 | 0.03% |
| occupation | 117 | 0.03% |
| physical object | 115 | 0.03% |
| ChemicalEntity | 113 | 0.03% |
| Scheduled Landing Place | 112 | 0.03% |
| country | 110 | 0.03% |
| Accusation | 109 | 0.03% |
| Habitat | 108 | 0.03% |
| Transportation | 107 | 0.03% |
| food item | 107 | 0.03% |
| Publication | 106 | 0.03% |
| Profession | 105 | 0.03% |
| Capabilities | 105 | 0.03% |
| Company/Organization | 104 | 0.03% |
| protein-family | 104 | 0.03% |
| Measurement | 104 | 0.03% |
| Furniture | 103 | 0.03% |
| measurement | 103 | 0.03% |
| group | 103 | 0.03% |
| Emotion | 102 | 0.03% |
| furniture | 101 | 0.03% |
| Influence People | 100 | 0.03% |
| company | 99 | 0.03% |
| Food | 99 | 0.03% |
| Currency | 98 | 0.03% |
| Animal | 96 | 0.03% |
| System | 96 | 0.03% |
| e_26 | 95 | 0.03% |
| Website | 94 | 0.03% |
| Time Period | 94 | 0.03% |
| senior_personnel | 92 | 0.03% |
| senior_position | 92 | 0.03% |
| product/service | 92 | 0.03% |
| entity | 92 | 0.03% |
| Athlete | 92 | 0.03% |
| website | 91 | 0.03% |
| Judgement Result/Prison Term | 90 | 0.03% |
| id | 90 | 0.03% |
| Belligerents | 90 | 0.03% |
| Accommodation | 90 | 0.03% |
| group of people | 89 | 0.03% |
| Contest Participant | 89 | 0.03% |
| Microorganism | 89 | 0.03% |
| Brand | 88 | 0.02% |
| Job Title | 88 | 0.02% |
| city | 88 | 0.02% |
| Entity | 87 | 0.02% |
| Freeze End Date | 87 | 0.02% |
| Storm Hit Location | 87 | 0.02% |
| Flight No. | 86 | 0.02% |
| Aircraft Agency | 86 | 0.02% |
| Award | 86 | 0.02% |
| Product/Service | 86 | 0.02% |
| Game Name | 85 | 0.02% |
| Entity Type | 83 | 0.02% |
| Magnitude | 82 | 0.02% |
| Aid Agency | 81 | 0.02% |
| Epicenter | 80 | 0.02% |
| quantity | 80 | 0.02% |
| Attribute | 79 | 0.02% |
| Patch | 79 | 0.02% |
| People | 78 | 0.02% |
| state | 76 | 0.02% |
| Color | 76 | 0.02% |
| OrganismTaxon | 76 | 0.02% |
| Geographic Location | 76 | 0.02% |
| address | 76 | 0.02% |
| geographical location | 76 | 0.02% |
| email | 75 | 0.02% |
| Pronoun | 74 | 0.02% |
| product feature | 74 | 0.02% |
| Medical Condition | 73 | 0.02% |
| Feature | 73 | 0.02% |
| title | 73 | 0.02% |
| Policy Content | 73 | 0.02% |
| Number of Vehicles Involved in the Crash | 72 | 0.02% |
| Person Name | 72 | 0.02% |
| quality | 72 | 0.02% |
| Geographical Location | 72 | 0.02% |
| place | 71 | 0.02% |
| Character | 71 | 0.02% |
| Participants | 70 | 0.02% |
| Bank Name | 70 | 0.02% |
| e_27 | 70 | 0.02% |
| compensation_recipient | 70 | 0.02% |
| SequenceVariant | 70 | 0.02% |
| Date/Time | 69 | 0.02% |
| software | 69 | 0.02% |
| job title | 69 | 0.02% |
| transportation | 69 | 0.02% |
| compensation_amount | 68 | 0.02% |
| Disease | 66 | 0.02% |
| people | 66 | 0.02% |
| Group of people | 66 | 0.02% |
| feature | 66 | 0.02% |
| Business Entity | 66 | 0.02% |
| Document | 65 | 0.02% |
| Passengers | 65 | 0.02% |
| clothing item | 64 | 0.02% |
| Death Time | 63 | 0.02% |
| Region | 63 | 0.02% |
| State | 63 | 0.02% |
| price | 63 | 0.02% |
| The Charged Crime | 62 | 0.02% |
| social group | 62 | 0.02% |
| Relationship | 61 | 0.02% |
| Investigating agency | 60 | 0.02% |
| Platform | 60 | 0.02% |
| Device | 60 | 0.02% |
| Operating System | 59 | 0.02% |
| Method | 59 | 0.02% |
| chemical | 59 | 0.02% |
| Data | 59 | 0.02% |
| currency | 58 | 0.02% |
| Nationality | 58 | 0.02% |
| geographic location | 58 | 0.02% |
| color | 57 | 0.02% |
| Book Title | 57 | 0.02% |
| Clothing Item | 57 | 0.02% |
| beverage | 56 | 0.02% |
| Address | 56 | 0.02% |
| Product Name | 56 | 0.02% |
| substance | 56 | 0.02% |
| Quality | 56 | 0.02% |
| Contracting Parties | 55 | 0.02% |
| Company/Brand | 55 | 0.02% |
| Language | 55 | 0.02% |
| UnfrozeDate | 55 | 0.02% |
| Number | 55 | 0.02% |
| Industry | 55 | 0.02% |
| Crew | 54 | 0.02% |
| e_28 | 54 | 0.02% |
| Age | 54 | 0.02% |
| Ingredient | 54 | 0.02% |
| tool | 54 | 0.02% |
| Time of the Case | 54 | 0.02% |
| DNA | 53 | 0.01% |
| Prosecutor | 53 | 0.01% |
| Casualty Count | 52 | 0.01% |
| equipment | 52 | 0.01% |
| communication | 52 | 0.01% |
| medical procedure | 51 | 0.01% |
| Perpetrators | 51 | 0.01% |
| Resignated Person | 51 | 0.01% |
| Institution | 51 | 0.01% |
| pronoun | 50 | 0.01% |
| Emotion/Feeling | 50 | 0.01% |
| PII | 50 | 0.01% |
| Survivors | 49 | 0.01% |
| brand | 49 | 0.01% |
| Brand Name | 49 | 0.01% |
| Attending Hospital | 49 | 0.01% |
| Champions | 49 | 0.01% |
| document | 48 | 0.01% |
| demographic group | 48 | 0.01% |
| Artwork | 48 | 0.01% |
| personality trait | 48 | 0.01% |
| Version | 48 | 0.01% |
| The Final Votes and Percentages | 48 | 0.01% |
| Court | 47 | 0.01% |
| Medical condition | 47 | 0.01% |
| Education | 47 | 0.01% |
| Price | 47 | 0.01% |
| Achievement | 47 | 0.01% |
| business | 47 | 0.01% |
| Money | 47 | 0.01% |
| Food Item | 46 | 0.01% |
| protein-complex | 46 | 0.01% |
| Confirmed/Infected Cases | 46 | 0.01% |
| Appointee | 46 | 0.01% |
| Areas affected | 45 | 0.01% |
| sports team | 45 | 0.01% |
| Location/Hospital | 45 | 0.01% |
| Venue | 45 | 0.01% |
| Beverage | 44 | 0.01% |
| educational institution | 44 | 0.01% |
| Resource | 44 | 0.01% |
| University | 43 | 0.01% |
| individual | 43 | 0.01% |
| family member | 43 | 0.01% |
| Deliberating Agency | 43 | 0.01% |
| Economic Loss | 43 | 0.01% |
| fashion item | 43 | 0.01% |
| Purpose | 43 | 0.01% |
| Restaurant | 42 | 0.01% |
| other_municipalities | 42 | 0.01% |
| vehicle | 42 | 0.01% |
| role | 42 | 0.01% |
| Vehicle | 42 | 0.01% |
| entity type | 42 | 0.01% |
| characteristic | 41 | 0.01% |
| Resign Reason | 41 | 0.01% |
| Types of the Spectacle | 41 | 0.01% |
| duration | 41 | 0.01% |
| Body Part | 41 | 0.01% |
| Sports team | 41 | 0.01% |
| File | 41 | 0.01% |
| Building | 40 | 0.01% |
| physical attribute | 40 | 0.01% |
| Sports Event | 40 | 0.01% |
| Author | 40 | 0.01% |
| physical location | 40 | 0.01% |
| Musician | 40 | 0.01% |
| Score | 40 | 0.01% |
| Election Name | 39 | 0.01% |
| Damaged Facility | 39 | 0.01% |
| amount | 39 | 0.01% |
| Other Loss | 39 | 0.01% |
| task | 39 | 0.01% |
| media | 39 | 0.01% |
| Entertainment | 39 | 0.01% |
| medication | 38 | 0.01% |
| Medication | 38 | 0.01% |
| Plant | 38 | 0.01% |
| Police | 38 | 0.01% |
| Process | 38 | 0.01% |
| device | 38 | 0.01% |
| Person/Group | 38 | 0.01% |
| Financial Entity | 38 | 0.01% |
| Summit Theme | 37 | 0.01% |
| date/time | 37 | 0.01% |
| Record Breaker | 37 | 0.01% |
| TV Show | 37 | 0.01% |
| dish | 37 | 0.01% |
| entertainment | 37 | 0.01% |
| e_29 | 37 | 0.01% |
| Information | 37 | 0.01% |
| Media | 37 | 0.01% |
| Name | 37 | 0.01% |
| People/Organization who predicted the disaster | 37 | 0.01% |
| protein-domain | 36 | 0.01% |
| architectural feature | 36 | 0.01% |
| behavior | 36 | 0.01% |
| relationship | 36 | 0.01% |
| Competition Items | 36 | 0.01% |
| Winner | 36 | 0.01% |
| Record-breaking Project | 36 | 0.01% |
| Grades | 36 | 0.01% |
| Property | 36 | 0.01% |
| information | 36 | 0.01% |
| phone | 36 | 0.01% |
| Acquiree | 36 | 0.01% |
| Weapon | 35 | 0.01% |
| component | 35 | 0.01% |
| Poisoning Type | 35 | 0.01% |
| Room | 35 | 0.01% |
| Goal | 35 | 0.01% |
| Tool | 35 | 0.01% |
| resource | 35 | 0.01% |
| Landmark | 34 | 0.01% |
| category | 34 | 0.01% |
| Social Media Platform | 34 | 0.01% |
| Experience | 34 | 0.01% |
| Riot Reason | 34 | 0.01% |
| Holiday | 34 | 0.01% |
| Communication | 34 | 0.01% |
| trait | 34 | 0.01% |
| gender | 33 | 0.01% |
| Family member | 33 | 0.01% |
| Adjective | 33 | 0.01% |
| Employment Agency | 33 | 0.01% |
| notes | 33 | 0.01% |
| Deceased | 33 | 0.01% |
| age group | 33 | 0.01% |
| document type | 33 | 0.01% |
| Product Feature | 33 | 0.01% |
| Equipment | 33 | 0.01% |
| Month | 32 | 0.01% |
| Deity | 32 | 0.01% |
| legal document | 32 | 0.01% |
| Malware | 32 | 0.01% |
| Geographic location | 32 | 0.01% |
| Agreement Content | 31 | 0.01% |
| author | 31 | 0.01% |
| platform | 31 | 0.01% |
| Car Model | 31 | 0.01% |
| food ingredient | 31 | 0.01% |
| business entity | 31 | 0.01% |
| medical treatment | 31 | 0.01% |
| Movie | 31 | 0.01% |
| product attribute | 31 | 0.01% |
| phoneNumber | 30 | 0.01% |
| year | 30 | 0.01% |
| Death Cases | 30 | 0.01% |
| property | 30 | 0.01% |
| Previous Record Holder | 30 | 0.01% |
| type | 30 | 0.01% |
| Spectacle Start Time | 30 | 0.01% |
| skill | 30 | 0.01% |
| Spectacle Location | 30 | 0.01% |
| Weapon Used | 30 | 0.01% |
| Phenotype | 30 | 0.01% |
| Strike Reason | 29 | 0.01% |
| physical property | 29 | 0.01% |
| pricing | 29 | 0.01% |
| service type | 29 | 0.01% |
| character | 29 | 0.01% |
| Game | 29 | 0.01% |
| Actor | 29 | 0.01% |
| Clothing | 29 | 0.01% |
| Group of People | 29 | 0.01% |
| cost | 28 | 0.01% |
| Infrastructure | 28 | 0.01% |
| Physical Attribute | 28 | 0.01% |
| Criminal Evidence | 28 | 0.01% |
| Field of study | 28 | 0.01% |
| Political Party | 28 | 0.01% |
| Facility | 28 | 0.01% |
| Army | 28 | 0.01% |
| Book | 28 | 0.01% |
| Storm Name | 28 | 0.01% |
| Storm Hit Time | 28 | 0.01% |
| Wife | 28 | 0.01% |
| Event/Activity | 28 | 0.01% |
| Timeframe | 28 | 0.01% |
| software tool | 27 | 0.01% |
| value | 27 | 0.01% |
| Substance | 27 | 0.01% |
| age | 27 | 0.01% |
| Item | 27 | 0.01% |
| Taking-off Place | 27 | 0.01% |
| Arrest Time | 27 | 0.01% |
| Family | 27 | 0.01% |
| Social Group | 27 | 0.01% |
| Appliance | 27 | 0.01% |
| person/group | 27 | 0.01% |
| Participating Countries | 27 | 0.01% |
| physical entity | 27 | 0.01% |
| Husband | 27 | 0.01% |
| software feature | 27 | 0.01% |
| musical instrument | 27 | 0.01% |
| Product Type | 27 | 0.01% |
| firstName | 26 | 0.01% |
| Pests | 26 | 0.01% |
| Movie Title | 26 | 0.01% |
| Brand/Company | 26 | 0.01% |
| Death Reason | 26 | 0.01% |
| Arrest Location | 26 | 0.01% |
| temperature | 26 | 0.01% |
| Day of the week | 26 | 0.01% |
| Service Type | 26 | 0.01% |
| Government Entity | 26 | 0.01% |
| item | 26 | 0.01% |
| industry | 25 | 0.01% |
| Food item | 25 | 0.01% |
| Company Industry | 25 | 0.01% |
| weather condition | 25 | 0.01% |
| Dish | 25 | 0.01% |
| City/Town | 25 | 0.01% |
| Outbreak Location | 25 | 0.01% |
| Species | 25 | 0.01% |
| building material | 25 | 0.01% |
| hardware | 25 | 0.01% |
| Sport | 25 | 0.01% |
| Last Job of the Appointee | 25 | 0.01% |
| education | 24 | 0.01% |
| adjective | 24 | 0.01% |
| Approver | 24 | 0.01% |
| lastName | 24 | 0.01% |
| Contact Information | 24 | 0.01% |
| Team | 24 | 0.01% |
| product type | 24 | 0.01% |
| Number of Evacuated People | 24 | 0.01% |
| achievement | 24 | 0.01% |
| athlete | 24 | 0.01% |
| Clothing item | 24 | 0.01% |
| art form | 24 | 0.01% |
| Time/Date | 24 | 0.01% |
| action/event | 24 | 0.01% |
| Time Interval | 24 | 0.01% |
| Video Game | 24 | 0.01% |
| season | 24 | 0.01% |
| appliance | 24 | 0.01% |
| CVE | 24 | 0.01% |
| supported_cryptocurrencies | 24 | 0.01% |
| Strike Outcome | 23 | 0.01% |
| department | 23 | 0.01% |
| Real Estate | 23 | 0.01% |
| Start Time | 23 | 0.01% |
| Percentage | 23 | 0.01% |
| monetary value | 23 | 0.01% |
| TV show | 23 | 0.01% |
| End Time | 23 | 0.01% |
| Trait | 23 | 0.01% |
| Architecture | 23 | 0.01% |
| clothing | 23 | 0.01% |
| transaction | 23 | 0.01% |
| rating | 23 | 0.01% |
| Volcano Name | 23 | 0.01% |
| infrastructure | 22 | 0.01% |
| user interface element | 22 | 0.01% |
| phone number | 22 | 0.01% |
| The Agency Who Broke the Agreement | 22 | 0.01% |
| time duration | 22 | 0.01% |
| features | 22 | 0.01% |
| Evidence | 22 | 0.01% |
| electronic device | 22 | 0.01% |
| Organization Name | 22 | 0.01% |
| Transaction | 22 | 0.01% |
| Effective Date | 22 | 0.01% |
| Award Reason | 22 | 0.01% |
| Maximum Wind Speed | 22 | 0.01% |
| nutrient | 22 | 0.01% |
| Username | 22 | 0.01% |
| Neighborhood | 22 | 0.01% |
| Certification | 22 | 0.01% |
| Day of the Week | 22 | 0.01% |
| Geographical Entity | 22 | 0.01% |
| Property Feature | 22 | 0.01% |
| publication | 22 | 0.01% |
| Musician/Band | 22 | 0.01% |
| Discover Location | 22 | 0.01% |
| Historical Sites | 22 | 0.01% |
| Court Time | 21 | 0.01% |
| communication method | 21 | 0.01% |
| field of study | 21 | 0.01% |
| container | 21 | 0.01% |
| structure | 21 | 0.01% |
| Functionality | 21 | 0.01% |
| cell | 21 | 0.01% |
| Season | 21 | 0.01% |
| number | 21 | 0.01% |
| Symptom | 21 | 0.01% |
| e_30 | 21 | 0.01% |
| Age Group | 21 | 0.01% |
| Pricing | 21 | 0.01% |
| Category | 21 | 0.01% |
| visual media | 21 | 0.01% |
| Precautionary Measure | 20 | 0.01% |
| Phone Number | 20 | 0.01% |
| User Interface Element | 20 | 0.01% |
| The Grades of the Previous Record Holder | 20 | 0.01% |
| Product feature | 20 | 0.01% |
| time frame | 20 | 0.01% |
| Geographical location | 20 | 0.01% |
| emotional state | 20 | 0.01% |
| agent | 20 | 0.01% |
| design element | 20 | 0.01% |
| Released People | 20 | 0.01% |
| natural feature | 20 | 0.01% |
| Artist | 20 | 0.01% |
| music genre | 20 | 0.01% |
| Car model | 20 | 0.01% |
| home decor | 20 | 0.01% |
| event type | 20 | 0.01% |
| software application | 20 | 0.01% |
| chemical compound | 20 | 0.01% |
| Fashion Item | 19 | 0.01% |
| Temperature | 19 | 0.01% |
| Conflict Duration | 19 | 0.01% |
| Album | 19 | 0.01% |
| productId | 19 | 0.01% |
| architecture | 19 | 0.01% |
| Body part | 19 | 0.01% |
| Person/Artist | 19 | 0.01% |
| Gender | 19 | 0.01% |
| Medical Procedure | 19 | 0.01% |
| target | 19 | 0.01% |
| Politician | 19 | 0.01% |
| Protein_family_or_group | 19 | 0.01% |
| Missings | 19 | 0.01% |
| finance | 19 | 0.01% |
| feeling | 19 | 0.01% |
| Natural Feature | 19 | 0.01% |
| Natural Disaster | 19 | 0.01% |
| Storm Warning Level | 19 | 0.01% |
| texture | 19 | 0.01% |
| Government Agency | 19 | 0.01% |
| Music Band | 19 | 0.01% |
| Plant species | 19 | 0.01% |
| physical feature | 19 | 0.01% |
| product line | 19 | 0.01% |
| Business Process | 19 | 0.01% |
| Legal Entity | 19 | 0.01% |
| financial transaction | 19 | 0.01% |
| Product Model | 19 | 0.01% |
| Phone number | 18 | 0.01% |
| disease | 18 | 0.01% |
| Other Impact | 18 | 0.01% |
| Crime | 18 | 0.01% |
| dosage | 18 | 0.01% |
| accommodation | 18 | 0.01% |
| Demographic group | 18 | 0.01% |
| zipCode | 18 | 0.01% |
| nationality | 18 | 0.01% |
| Verb | 18 | 0.01% |
| Product Line | 18 | 0.01% |
| Condition | 18 | 0.01% |
| profession/occupation | 18 | 0.01% |
| unit | 18 | 0.01% |
| Defense Lawyer | 18 | 0.01% |
| human being | 18 | 0.01% |
| Geopolitical Entity | 18 | 0.01% |
| experience | 18 | 0.01% |
| Media Outlet | 18 | 0.01% |
| Gene | 18 | 0.01% |
| environment | 18 | 0.01% |
| Sports event | 18 | 0.01% |
| Fictional character | 18 | 0.01% |
| operating system | 18 | 0.01% |
| room type | 18 | 0.01% |
| topic | 18 | 0.01% |
| Investigative Agency | 18 | 0.01% |
| activity/event | 18 | 0.01% |
| Protest Slogan | 18 | 0.01% |
| Organization Industry | 18 | 0.01% |
| code | 18 | 0.01% |
| Product Category | 18 | 0.01% |
| symptoms | 18 | 0.01% |
| Victim | 17 | 0.00% |
| Degree | 17 | 0.00% |
| mode of transportation | 17 | 0.00% |
| Duration | 17 | 0.00% |
| real estate | 17 | 0.00% |
| Weather condition | 17 | 0.00% |
| startDate | 17 | 0.00% |
| Fine Reason | 17 | 0.00% |
| plant | 17 | 0.00% |
| beauty product | 17 | 0.00% |
| Environment | 17 | 0.00% |
| Strikers Status | 17 | 0.00% |
| brand name | 17 | 0.00% |
| fictional character | 17 | 0.00% |
| Promotion | 17 | 0.00% |
| Role | 17 | 0.00% |
| accessory | 17 | 0.00% |
| size | 17 | 0.00% |
| Direction | 17 | 0.00% |
| Host Country | 17 | 0.00% |
| educational program | 17 | 0.00% |
| project | 17 | 0.00% |
| Hotel | 17 | 0.00% |
| match | 17 | 0.00% |
| Program | 17 | 0.00% |
| symptom | 17 | 0.00% |
| File sharing service | 17 | 0.00% |
| Electronic device | 17 | 0.00% |
| Invited Person | 17 | 0.00% |
| physical sensation | 17 | 0.00% |
| Title | 17 | 0.00% |
| Fictional Character | 17 | 0.00% |
| ability | 17 | 0.00% |
| Band | 17 | 0.00% |
| Physical Entity | 17 | 0.00% |
| Historical Event | 17 | 0.00% |
| Ranking | 17 | 0.00% |
| medical device | 17 | 0.00% |
| Community | 17 | 0.00% |
| Number of Damaged Houses | 17 | 0.00% |
| content | 17 | 0.00% |
| Payment Method | 17 | 0.00% |
| tags | 17 | 0.00% |
| Sponsor | 17 | 0.00% |
| Start Date | 16 | 0.00% |
| Epidemic Data Issuing Agency | 16 | 0.00% |
| person name | 16 | 0.00% |
| service provider | 16 | 0.00% |
| venue | 16 | 0.00% |
| Fined Agency | 16 | 0.00% |
| kitchenware | 16 | 0.00% |
| numerical value | 16 | 0.00% |
| Product/Item | 16 | 0.00% |
| Business Name | 16 | 0.00% |
| reagent | 16 | 0.00% |
| Music Group | 16 | 0.00% |
| Genre | 16 | 0.00% |
| building | 16 | 0.00% |
| product component | 16 | 0.00% |
| Social group | 16 | 0.00% |
| Agreement Name | 16 | 0.00% |
| Project | 16 | 0.00% |
| zip | 16 | 0.00% |
| Hardware | 16 | 0.00% |
| natural phenomenon | 16 | 0.00% |
| Religious Figure | 16 | 0.00% |
| Speaker Status | 16 | 0.00% |
| Speaker | 16 | 0.00% |
| Election Goal | 16 | 0.00% |
| weather phenomenon | 16 | 0.00% |
| Solution | 16 | 0.00% |
| Natural Phenomenon | 16 | 0.00% |
| outcome | 16 | 0.00% |
| unit of measurement | 16 | 0.00% |
| team | 16 | 0.00% |
| sport | 16 | 0.00% |
| Job title | 16 | 0.00% |
| Outbreak Date | 16 | 0.00% |
| Personality Trait | 16 | 0.00% |
| initial_text.actions | 15 | 0.00% |
| Train No. | 15 | 0.00% |
| room | 15 | 0.00% |
| Music Genre | 15 | 0.00% |
| outdoor space | 15 | 0.00% |
| Last Time the Record was Broken | 15 | 0.00% |
| treatment | 15 | 0.00% |
| amenities | 15 | 0.00% |
| Weather Phenomenon | 15 | 0.00% |
| Cryptocurrency | 15 | 0.00% |
| organism | 15 | 0.00% |
| weight | 15 | 0.00% |
| event/activity | 15 | 0.00% |
| Occupation/Role | 15 | 0.00% |
| biological process | 15 | 0.00% |
| biological entity | 15 | 0.00% |
| Policy | 15 | 0.00% |
| Establishment | 15 | 0.00% |
| Stolen Amount | 15 | 0.00% |
| Number of Influenced People | 15 | 0.00% |
| File format | 15 | 0.00% |
| Document Type | 15 | 0.00% |
| contact information | 15 | 0.00% |
| Number of Destroyed Building | 15 | 0.00% |
| Field of Study | 15 | 0.00% |
| Cuisine | 15 | 0.00% |
| language | 15 | 0.00% |
| reason | 15 | 0.00% |
| Sports League | 15 | 0.00% |
| Museum | 15 | 0.00% |
| Outcome | 15 | 0.00% |
| Service Provider | 15 | 0.00% |
| Response Measures | 15 | 0.00% |
| Software Product | 15 | 0.00% |
| Pollution Source | 15 | 0.00% |
| Predecessor | 15 | 0.00% |
| target audience | 15 | 0.00% |
| Video game | 15 | 0.00% |
| physical activity | 15 | 0.00% |
| payment method | 15 | 0.00% |
| Protein | 15 | 0.00% |
| Medical Treatment | 15 | 0.00% |
| Legal Document | 15 | 0.00% |
| Building Material | 15 | 0.00% |
| head | 15 | 0.00% |
| tail | 15 | 0.00% |
| skill level | 15 | 0.00% |
| experimental-construct | 15 | 0.00% |
| Artifacts and Their Chronology | 15 | 0.00% |
| institution | 14 | 0.00% |
| employee_id | 14 | 0.00% |
| Country/Region | 14 | 0.00% |
| Task | 14 | 0.00% |
| endDate | 14 | 0.00% |
| audience | 14 | 0.00% |
| streetAddress | 14 | 0.00% |
| customer service | 14 | 0.00% |
| Legal document | 14 | 0.00% |
| Identifier | 14 | 0.00% |
| Commodity | 14 | 0.00% |
| Financial Instrument | 14 | 0.00% |
| car model | 14 | 0.00% |
| Host | 14 | 0.00% |
| verb | 14 | 0.00% |
| Religious Text | 14 | 0.00% |
| Lasting Time | 14 | 0.00% |
| digital platform | 14 | 0.00% |
| Song Title | 14 | 0.00% |
| Funding | 14 | 0.00% |
| sports event | 14 | 0.00% |
| Skill | 14 | 0.00% |
| Marketing Strategy | 14 | 0.00% |
| Natural feature | 14 | 0.00% |
| establishment | 14 | 0.00% |
| Brand/Product | 14 | 0.00% |
| Emotional state | 14 | 0.00% |
| social media platform | 14 | 0.00% |
| Personal Information | 14 | 0.00% |
| time interval | 14 | 0.00% |
| Number of Rescued People | 14 | 0.00% |
| fish species | 14 | 0.00% |
| Anatomy | 14 | 0.00% |
| technique | 14 | 0.00% |
| Street Name | 14 | 0.00% |
| Gaming Console | 14 | 0.00% |
| food/dish | 14 | 0.00% |
| vegetable | 14 | 0.00% |
| plant species | 14 | 0.00% |
| Adverb | 14 | 0.00% |
| Football Team | 14 | 0.00% |
| Component | 14 | 0.00% |
| Anti-pollution People/Organizations | 14 | 0.00% |
| Aid Supplies/Amount | 14 | 0.00% |
| meal | 14 | 0.00% |
| Composer | 14 | 0.00% |
| manufacturer | 14 | 0.00% |
| Acquisition amount | 14 | 0.00% |
| Football player | 14 | 0.00% |
| position | 14 | 0.00% |
| Numeric Value | 14 | 0.00% |
| Vehicle Type | 14 | 0.00% |
| car part | 14 | 0.00% |
| Hostage | 14 | 0.00% |
| Legal entity | 14 | 0.00% |
| financial product | 14 | 0.00% |
| Fixture | 14 | 0.00% |
| products | 14 | 0.00% |
| Forecasting Agency | 14 | 0.00% |
| personal trait | 14 | 0.00% |
| Town | 14 | 0.00% |
| Control Number | 14 | 0.00% |
| household item | 13 | 0.00% |
| Finance | 13 | 0.00% |
| Topic | 13 | 0.00% |
| financial metric | 13 | 0.00% |
| Term of Office | 13 | 0.00% |
| Ethnic Group | 13 | 0.00% |
| File Format | 13 | 0.00% |
| Country Visited | 13 | 0.00% |
| Medical treatment | 13 | 0.00% |
| Recovered Amount | 13 | 0.00% |
| job position | 13 | 0.00% |
| kitchen appliance | 13 | 0.00% |
| Electronic Device | 13 | 0.00% |
| necessity | 13 | 0.00% |
| Tourist Attraction | 13 | 0.00% |
| Educational institution | 13 | 0.00% |
| Biological process | 13 | 0.00% |
| facility | 13 | 0.00% |
| material type | 13 | 0.00% |
| Release Reason | 13 | 0.00% |
| data | 13 | 0.00% |
| technology device | 13 | 0.00% |
| street | 13 | 0.00% |
| Jail Time | 13 | 0.00% |
| Wedding Venue | 13 | 0.00% |
| score | 13 | 0.00% |
| Monetary Value | 13 | 0.00% |
| expirationDate | 13 | 0.00% |
| dateOfBirth | 13 | 0.00% |
| Issue | 13 | 0.00% |
| Inauguration Time | 13 | 0.00% |
| Author Name | 13 | 0.00% |
| Content | 13 | 0.00% |
| Real Estate Property | 13 | 0.00% |
| emotion/feeling | 13 | 0.00% |
| file format | 13 | 0.00% |
| Communication Method | 13 | 0.00% |
| drug | 13 | 0.00% |
| Video Game Title | 13 | 0.00% |
| measurement unit | 13 | 0.00% |
| Size | 13 | 0.00% |
| Appointer | 13 | 0.00% |
| Time Duration | 13 | 0.00% |
| physical space | 13 | 0.00% |
| Nutrient | 13 | 0.00% |
| Area Affected | 13 | 0.00% |
| The Final Seats Result | 13 | 0.00% |
| Job position | 13 | 0.00% |
| Metric | 13 | 0.00% |
| Beneficiary | 13 | 0.00% |
| CellType | 13 | 0.00% |
| patient_id | 12 | 0.00% |
| Way for Spreading | 12 | 0.00% |
| Fashion Style | 12 | 0.00% |
| group of individuals | 12 | 0.00% |
| creative work | 12 | 0.00% |
| individuals | 12 | 0.00% |
| Ethnic group | 12 | 0.00% |
| Law | 12 | 0.00% |
| identifier | 12 | 0.00% |
| program | 12 | 0.00% |
| Herb | 12 | 0.00% |
| day of the week | 12 | 0.00% |
| Countries Withdrawing from the Organization | 12 | 0.00% |
| Communication method | 12 | 0.00% |
| Chemical compound | 12 | 0.00% |
| Music | 12 | 0.00% |
| Arrested | 12 | 0.00% |
| resources | 12 | 0.00% |
| medical professional | 12 | 0.00% |
| Candidate Scandal(Allegations of fraud, etc.) | 12 | 0.00% |
| street name | 12 | 0.00% |
| Judge | 12 | 0.00% |
| Record Label | 12 | 0.00% |
| award | 12 | 0.00% |
| Music genre | 12 | 0.00% |
| Programming Language | 12 | 0.00% |
| physical appearance | 12 | 0.00% |
| animal species | 12 | 0.00% |
| cryptocurrency | 12 | 0.00% |
| amod | 12 | 0.00% |
| compound | 12 | 0.00% |
| skincare product | 12 | 0.00% |
| visual element | 12 | 0.00% |
| hardware component | 12 | 0.00% |
| software component | 12 | 0.00% |
| home improvement | 12 | 0.00% |
| Lifestyle | 12 | 0.00% |
| frequency | 12 | 0.00% |
| Publisher | 12 | 0.00% |
| Age group | 12 | 0.00% |
| Occasion | 12 | 0.00% |
| School | 12 | 0.00% |
| online platform | 12 | 0.00% |
| job role | 12 | 0.00% |
| Biological Entity | 12 | 0.00% |
| Strike Agency | 12 | 0.00% |
| Mineral | 12 | 0.00% |
| personal information | 12 | 0.00% |
| species | 12 | 0.00% |
| Military Exercise | 12 | 0.00% |
| medium | 12 | 0.00% |
| Collection | 12 | 0.00% |
| skill/ability | 12 | 0.00% |
| food product | 12 | 0.00% |
| Time Entity | 12 | 0.00% |
| jewelry | 12 | 0.00% |
| Product type | 12 | 0.00% |
| Application | 12 | 0.00% |
| Natural Resource | 12 | 0.00% |
| experiment-tag | 12 | 0.00% |
| booking_information.booking_partners | 12 | 0.00% |
| Toy | 12 | 0.00% |
| Call Number | 12 | 0.00% |
| Flavor | 12 | 0.00% |
| Aid Reason | 12 | 0.00% |
| serialNumber | 11 | 0.00% |
| End Date | 11 | 0.00% |
| bird species | 11 | 0.00% |
| condition | 11 | 0.00% |
| Kitchen Appliance | 11 | 0.00% |
| educational material | 11 | 0.00% |
| Regulatory Authority | 11 | 0.00% |
| Successor | 11 | 0.00% |
| product_id | 11 | 0.00% |
| Game Title | 11 | 0.00% |
| Fashion item | 11 | 0.00% |
| advice | 11 | 0.00% |
| vehicle model | 11 | 0.00% |
| Strikers | 11 | 0.00% |
| Entity type | 11 | 0.00% |
| Education Level | 11 | 0.00% |
| process/activity | 11 | 0.00% |
| Declarer | 11 | 0.00% |
| Physical Property | 11 | 0.00% |
| Role/Position | 11 | 0.00% |
| Physical Space | 11 | 0.00% |
| sku | 11 | 0.00% |
| height | 11 | 0.00% |
| financial asset | 11 | 0.00% |
| Group/Community | 11 | 0.00% |
| condiment | 11 | 0.00% |
| natural disaster | 11 | 0.00% |
| Physical Object | 11 | 0.00% |
| Award Field | 11 | 0.00% |
| financial entity | 11 | 0.00% |
| property type | 11 | 0.00% |
| expression | 11 | 0.00% |
| Family Member | 11 | 0.00% |
| Related Rivers or Lakes | 11 | 0.00% |
| media outlet | 11 | 0.00% |
| Government Body | 11 | 0.00% |
| Shipwreck Reason | 11 | 0.00% |
| Action/Verb | 11 | 0.00% |
| style | 11 | 0.00% |
| website feature | 11 | 0.00% |
| game | 11 | 0.00% |
| url | 11 | 0.00% |
| issue | 11 | 0.00% |
| Military Strength | 11 | 0.00% |
| Book title | 11 | 0.00% |
| Street Address | 11 | 0.00% |
| Style | 11 | 0.00% |
| Discount | 11 | 0.00% |
| human | 11 | 0.00% |
| demographic | 11 | 0.00% |
| food type | 11 | 0.00% |
| Areas Affected | 11 | 0.00% |
| exercise equipment | 11 | 0.00% |
| Postal Code | 11 | 0.00% |
| e_31 | 11 | 0.00% |
| design feature | 11 | 0.00% |
| Amenity | 11 | 0.00% |
| cooking technique | 11 | 0.00% |
| Age of the Appointee | 11 | 0.00% |
| Demographic | 11 | 0.00% |
| Living space | 11 | 0.00% |
| financial instrument | 11 | 0.00% |
| amino-acid | 11 | 0.00% |
| Continent | 11 | 0.00% |
| government agency | 11 | 0.00% |
| Trait/Characteristic | 11 | 0.00% |
| cooking ingredient | 11 | 0.00% |
| Music Album | 11 | 0.00% |
| Treatment | 11 | 0.00% |
| technical term | 11 | 0.00% |
| Economic loss | 10 | 0.00% |
| productName | 10 | 0.00% |
| Retail Store | 10 | 0.00% |
| Work Schedule | 10 | 0.00% |
| Service/Product | 10 | 0.00% |
| media type | 10 | 0.00% |
| Idea | 10 | 0.00% |
| Noun | 10 | 0.00% |
| individual person | 10 | 0.00% |
| Meal | 10 | 0.00% |
| data storage | 10 | 0.00% |
| Educational Program | 10 | 0.00% |
| Musical Instrument | 10 | 0.00% |
| restaurant | 10 | 0.00% |
| performance | 10 | 0.00% |
| Legislation | 10 | 0.00% |
| Commanders and their Position | 10 | 0.00% |
| Scale | 10 | 0.00% |
| Horse | 10 | 0.00% |
| Spacecraft | 10 | 0.00% |
| machine | 10 | 0.00% |
| Food product | 10 | 0.00% |
| Magazine | 10 | 0.00% |
| literary device | 10 | 0.00% |
| Action/Behavior | 10 | 0.00% |
| Alternate Landing Place | 10 | 0.00% |
| Physical object | 10 | 0.00% |
| Skill/Ability | 10 | 0.00% |
| skillset | 10 | 0.00% |
| chemical element | 10 | 0.00% |
| Historical event | 10 | 0.00% |
| food/dessert | 10 | 0.00% |
| dish type | 10 | 0.00% |
| Product/Technology | 10 | 0.00% |
| certification | 10 | 0.00% |
| Time duration | 10 | 0.00% |
| Religious Institution | 10 | 0.00% |
| comments | 10 | 0.00% |
| payment | 10 | 0.00% |
| Human | 10 | 0.00% |
| noun | 10 | 0.00% |
| ethnic group | 10 | 0.00% |
| Software Tool | 10 | 0.00% |
| landmark | 10 | 0.00% |
| deity | 10 | 0.00% |
| retail store | 10 | 0.00% |
| decoration | 10 | 0.00% |
| Course | 10 | 0.00% |
| Automotive Part | 10 | 0.00% |
| Emotional State | 10 | 0.00% |
| UI element | 10 | 0.00% |
| City/Location | 10 | 0.00% |
| Football Club | 10 | 0.00% |
| Material Type | 10 | 0.00% |
| Legal Term | 10 | 0.00% |
| academic writing | 10 | 0.00% |
| benefits | 10 | 0.00% |
| product name | 10 | 0.00% |
| Film | 10 | 0.00% |
| marketing strategy | 10 | 0.00% |
| software version | 10 | 0.00% |
| Feeling/Emotion | 10 | 0.00% |
| model | 10 | 0.00% |
| Group of individuals | 10 | 0.00% |
| Customer | 10 | 0.00% |
| distance | 10 | 0.00% |
| e_32 | 10 | 0.00% |
| time of day | 10 | 0.00% |
| issue/problem | 10 | 0.00% |
| natural resource | 10 | 0.00% |
| County | 10 | 0.00% |
| Place Name | 10 | 0.00% |
| timestamp | 10 | 0.00% |
| Room/Space | 10 | 0.00% |
| geographical feature | 10 | 0.00% |
| store | 10 | 0.00% |
| Action/Process | 10 | 0.00% |
| aesthetic feature | 10 | 0.00% |
| Discover Time | 10 | 0.00% |
| Property feature | 10 | 0.00% |
| Occupation/Job Title | 10 | 0.00% |
| physical characteristic | 10 | 0.00% |
| sports term | 10 | 0.00% |
| financial information | 10 | 0.00% |
| book title | 10 | 0.00% |
| municipalities | 10 | 0.00% |
| severity | 9 | 0.00% |
| ticket_id | 9 | 0.00% |
| support | 9 | 0.00% |
| Agreement | 9 | 0.00% |
| asteroid.wiki_sky_links | 9 | 0.00% |
| Biological entity | 9 | 0.00% |
| Workplace | 9 | 0.00% |
| month | 9 | 0.00% |
| taste | 9 | 0.00% |
| User | 9 | 0.00% |
| technology product | 9 | 0.00% |
| legal process | 9 | 0.00% |
| Exam | 9 | 0.00% |
| problem | 9 | 0.00% |
| vehicle type | 9 | 0.00% |
| Policy Name & Abbreviation | 9 | 0.00% |
| Person under Investigation | 9 | 0.00% |
| Apparel | 9 | 0.00% |
| mental state | 9 | 0.00% |
| mechanical part | 9 | 0.00% |
| Historical Figure | 9 | 0.00% |
| Obstacle | 9 | 0.00% |
| Projects | 9 | 0.00% |
| Aircraft model | 9 | 0.00% |
| Physical Location | 9 | 0.00% |
| Weather | 9 | 0.00% |
| plant part | 9 | 0.00% |
| width | 9 | 0.00% |
| company/organization | 9 | 0.00% |
| Plant part | 9 | 0.00% |
| Musical instrument | 9 | 0.00% |
| Musical Group | 9 | 0.00% |
| customer | 9 | 0.00% |
| kitchen utensil | 9 | 0.00% |
| Design Element | 9 | 0.00% |
| peptide | 9 | 0.00% |
| Game title | 9 | 0.00% |
| Person/Name | 9 | 0.00% |
| Entertainment Venue | 9 | 0.00% |
| brand/company | 9 | 0.00% |
| Store | 9 | 0.00% |
| lifestyle | 9 | 0.00% |
| skin condition | 9 | 0.00% |
| Literature | 9 | 0.00% |
| Webpage | 9 | 0.00% |
| Distance | 9 | 0.00% |
| Music Artist | 9 | 0.00% |
| Car Feature | 9 | 0.00% |
| Real estate property | 9 | 0.00% |
| service quality | 9 | 0.00% |
| Car part | 9 | 0.00% |
| legal term | 9 | 0.00% |
| Treatment Method | 9 | 0.00% |
| Military Unit | 9 | 0.00% |
| technology/software | 9 | 0.00% |
| Software/Application | 9 | 0.00% |
| amenity | 9 | 0.00% |
| PaymentMethod | 9 | 0.00% |
| data type | 9 | 0.00% |
| Financial | 9 | 0.00% |
| Medical Specialty | 9 | 0.00% |
| method | 9 | 0.00% |
| policy | 9 | 0.00% |
| Strike Industry | 9 | 0.00% |
| Cost | 9 | 0.00% |
| Hobby | 9 | 0.00% |
| cuisine | 9 | 0.00% |
| requirement | 9 | 0.00% |
| product category | 9 | 0.00% |
| Temporal Entity | 9 | 0.00% |
| Visual element | 9 | 0.00% |
| Product part | 9 | 0.00% |
| anatomical structure | 9 | 0.00% |
| home feature | 9 | 0.00% |
| medical facility | 9 | 0.00% |
| sensation | 9 | 0.00% |
| Consumer | 9 | 0.00% |
| Job Role | 9 | 0.00% |
| energy source | 9 | 0.00% |
| Online Marketplace | 9 | 0.00% |
| Fashion Brand | 9 | 0.00% |
| Activity Type | 9 | 0.00% |
| Job Position | 9 | 0.00% |
| Last Outbreak Time | 9 | 0.00% |
| proper noun, singular | 9 | 0.00% |
| function | 9 | 0.00% |
| belief system | 9 | 0.00% |
| goal | 9 | 0.00% |
| service/product | 9 | 0.00% |
| animal breed | 9 | 0.00% |
| Person/Individual | 9 | 0.00% |
| preference | 9 | 0.00% |
| artist | 9 | 0.00% |
| Shape | 9 | 0.00% |
| historical event | 9 | 0.00% |
| Medical procedure | 9 | 0.00% |
| Chemical Compound | 9 | 0.00% |
| Symbol | 9 | 0.00% |
| Emotion/State | 9 | 0.00% |
| Number of Trapped People | 9 | 0.00% |
| Boycotted Institutions | 9 | 0.00% |
| Highway | 9 | 0.00% |
| Jewelry | 9 | 0.00% |
| Academic Institution | 9 | 0.00% |
| Turnout(who win who lost) | 9 | 0.00% |
| None | 9 | 0.00% |
| design | 9 | 0.00% |
| literary genre | 9 | 0.00% |
| Temporary Settlement | 9 | 0.00% |
| clothing feature | 9 | 0.00% |
| protein-motif | 9 | 0.00% |
| football club/team | 9 | 0.00% |
| Action/Event | 8 | 0.00% |
| goals | 8 | 0.00% |
| Voting method | 8 | 0.00% |
| Tool/Software | 8 | 0.00% |
| promotion | 8 | 0.00% |
| programming language | 8 | 0.00% |
| Ability | 8 | 0.00% |
| family | 8 | 0.00% |
| Penalty Amount | 8 | 0.00% |
| fruit | 8 | 0.00% |
| Demographic Group | 8 | 0.00% |
| cooking method | 8 | 0.00% |
| Software Component | 8 | 0.00% |
| recreational facility | 8 | 0.00% |
| financial resource | 8 | 0.00% |
| Abbreviation | 8 | 0.00% |
| Entity Group | 8 | 0.00% |
| pest | 8 | 0.00% |
| Blog | 8 | 0.00% |
| computer model | 8 | 0.00% |
| Vehicle Model | 8 | 0.00% |
| Dish/Recipe | 8 | 0.00% |
| gene | 8 | 0.00% |
| Visitor | 8 | 0.00% |
| speed | 8 | 0.00% |
| religion | 8 | 0.00% |
| hairstyle | 8 | 0.00% |
| Room Feature | 8 | 0.00% |
| architectural element | 8 | 0.00% |
| Law Firm | 8 | 0.00% |
| environmental factor | 8 | 0.00% |
| literary work | 8 | 0.00% |
| insurance policy | 8 | 0.00% |
| Sports league | 8 | 0.00% |
| Social Issue | 8 | 0.00% |
| damage | 8 | 0.00% |
| anatomical entity | 8 | 0.00% |
| Bill Drafting Agency | 8 | 0.00% |
| Contact information | 8 | 0.00% |
| Medical facility | 8 | 0.00% |
| protein-isoform | 8 | 0.00% |
| body parts | 8 | 0.00% |
| Activity/Experience | 8 | 0.00% |
| Storm Movement Speed | 8 | 0.00% |
| Quality Standard | 8 | 0.00% |
| Flower | 8 | 0.00% |
| Room type | 8 | 0.00% |
| Business Strategy | 8 | 0.00% |
| Child | 8 | 0.00% |
| health condition | 8 | 0.00% |
| bathroom fixture | 8 | 0.00% |
| software issue | 8 | 0.00% |
| video game | 8 | 0.00% |
| political entity | 8 | 0.00% |
| schedule | 8 | 0.00% |
| Weight | 8 | 0.00% |
| beer type | 8 | 0.00% |
| religious figure | 8 | 0.00% |
| Class | 8 | 0.00% |
| data structure | 8 | 0.00% |
| inquiry | 8 | 0.00% |
| Investigating Spokesperson | 8 | 0.00% |
| Physical Condition | 8 | 0.00% |
| Health Condition | 8 | 0.00% |
| Summit Name | 8 | 0.00% |
| Space | 8 | 0.00% |
| National Football Team | 8 | 0.00% |
| Design Feature | 8 | 0.00% |
| City Name | 8 | 0.00% |
| education level | 8 | 0.00% |
| Art form | 8 | 0.00% |
| Activity/Process | 8 | 0.00% |
| Food/Ingredient | 8 | 0.00% |
| Drink | 8 | 0.00% |
| time unit | 8 | 0.00% |
| Audience | 8 | 0.00% |
| Song title | 8 | 0.00% |
| Ways to Watch the Speech | 8 | 0.00% |
| Geographical entity | 8 | 0.00% |
| Town/City | 8 | 0.00% |
| workplace | 8 | 0.00% |
| social issue | 8 | 0.00% |
| personal experience | 8 | 0.00% |
| Processor | 8 | 0.00% |
| Possession | 8 | 0.00% |
| Expression | 8 | 0.00% |
| Aircraft Model | 8 | 0.00% |
| subject | 8 | 0.00% |
| Property Type | 8 | 0.00% |
| Video game title | 8 | 0.00% |
| Physical entity | 8 | 0.00% |
| region | 8 | 0.00% |
| Insurance Company | 8 | 0.00% |
| Ethnicity | 8 | 0.00% |
| Design element | 8 | 0.00% |
| stock ticker | 8 | 0.00% |
| financial institution | 8 | 0.00% |
| Financial Term | 8 | 0.00% |
| Website Element | 8 | 0.00% |
| Performance | 8 | 0.00% |
| Water Level | 8 | 0.00% |
| personal identification | 8 | 0.00% |
| Song | 8 | 0.00% |
| Travel | 8 | 0.00% |
| Sports | 8 | 0.00% |
| Fire Warning Level | 8 | 0.00% |
| Skill/Expertise | 8 | 0.00% |
| destination | 8 | 0.00% |
| symbol | 8 | 0.00% |
| customerId | 8 | 0.00% |
| Locks | 8 | 0.00% |
| Target audience | 8 | 0.00% |
| Festival | 8 | 0.00% |
| Furniture Item | 8 | 0.00% |
| Beauty Product | 8 | 0.00% |
| Business/Service | 8 | 0.00% |
| Measurement Unit | 8 | 0.00% |
| Structure | 8 | 0.00% |
| Historical Period | 8 | 0.00% |
| personal relationship | 8 | 0.00% |
| Environmental entity | 8 | 0.00% |
| organization/group | 8 | 0.00% |
| natural element | 8 | 0.00% |
| written work | 8 | 0.00% |
| Movie/TV Show | 8 | 0.00% |
| start_date | 8 | 0.00% |
| software product | 8 | 0.00% |
| celestial body | 8 | 0.00% |
| Religion | 8 | 0.00% |
| Marketing strategy | 8 | 0.00% |
| categories | 8 | 0.00% |
| location/real estate | 8 | 0.00% |
| musical group | 8 | 0.00% |
| musician | 8 | 0.00% |
| Amount of Precipitation | 8 | 0.00% |
| Social media platform | 8 | 0.00% |
| The State of the Volcano (Dormant or Active) | 8 | 0.00% |
| geological formation | 8 | 0.00% |
| Job Title/Role | 8 | 0.00% |
| Unknown | 8 | 0.00% |
| fashion style | 8 | 0.00% |
| food/drink | 8 | 0.00% |
| Activity/Behavior | 8 | 0.00% |
| content type | 8 | 0.00% |
| Online username | 8 | 0.00% |
| hobby/interest | 8 | 0.00% |
| geographic region | 8 | 0.00% |
| Political Entity | 8 | 0.00% |
| favorite_players | 8 | 0.00% |
| Government Reaction | 8 | 0.00% |
| images | 8 | 0.00% |
| NNP | 8 | 0.00% |
| Golf Player | 8 | 0.00% |
| Numeric value | 7 | 0.00% |
| Mobile device | 7 | 0.00% |
| Problem | 7 | 0.00% |
| information source | 7 | 0.00% |
| Tear Up Reason | 7 | 0.00% |
| asteroid.designation | 7 | 0.00% |
| geographical entity | 7 | 0.00% |
| ideology | 7 | 0.00% |
| Wind speed | 7 | 0.00% |
| Head of Institution | 7 | 0.00% |
| software development | 7 | 0.00% |
| art style | 7 | 0.00% |
| visual content | 7 | 0.00% |
| Historical figure | 7 | 0.00% |
| patient_name | 7 | 0.00% |
| idiom | 7 | 0.00% |
| personnel | 7 | 0.00% |
| Financial Transaction | 7 | 0.00% |
| art supply | 7 | 0.00% |
| interests | 7 | 0.00% |
| creation | 7 | 0.00% |
| Head of the Institution | 7 | 0.00% |
| academic discipline | 7 | 0.00% |
| Rider | 7 | 0.00% |
| Behavior | 7 | 0.00% |
| Acronym | 7 | 0.00% |
| electronic component | 7 | 0.00% |
| Medium | 7 | 0.00% |
| Metaphor | 7 | 0.00% |
| Carrier Rocket | 7 | 0.00% |
| Security | 7 | 0.00% |
| fall item | 7 | 0.00% |
| software update | 7 | 0.00% |
| Advice | 7 | 0.00% |
| existence | 7 | 0.00% |
| mythical creature | 7 | 0.00% |
| Media Type | 7 | 0.00% |
| length | 7 | 0.00% |
| personal attribute | 7 | 0.00% |
| education/training | 7 | 0.00% |
| political event | 7 | 0.00% |
| Medical Professional | 7 | 0.00% |
| Idiom | 7 | 0.00% |
| Release Time | 7 | 0.00% |
| source | 7 | 0.00% |
| Wine | 7 | 0.00% |
| Knowledge | 7 | 0.00% |
| documentation | 7 | 0.00% |
| standard | 7 | 0.00% |
| Container | 7 | 0.00% |
| recipe | 7 | 0.00% |
| Skincare product | 7 | 0.00% |
| Mountain | 7 | 0.00% |
| family members | 7 | 0.00% |
| Medical Facility | 7 | 0.00% |
| environmental issue | 7 | 0.00% |
| units | 7 | 0.00% |
| artwork | 7 | 0.00% |
| financial market | 7 | 0.00% |
| Physical attribute | 7 | 0.00% |
| Field of work | 7 | 0.00% |
| Living organism | 7 | 0.00% |
| Market | 7 | 0.00% |
| Perpetrator | 7 | 0.00% |
| Storm Direction | 7 | 0.00% |
| collection | 7 | 0.00% |
| Company/Service | 7 | 0.00% |
| brand/product | 7 | 0.00% |
| religious organization | 7 | 0.00% |
| sensory experience | 7 | 0.00% |
| Influenced Crops and Livelihood | 7 | 0.00% |
| objects | 7 | 0.00% |
| distribution_information.distribution_locations | 7 | 0.00% |
| Title/Position | 7 | 0.00% |
| Economic Indicator | 7 | 0.00% |
| social event | 7 | 0.00% |
| recreational activity | 7 | 0.00% |
| Symptoms | 7 | 0.00% |
| Service quality | 7 | 0.00% |
| Design | 7 | 0.00% |
| Fabric | 7 | 0.00% |
| leisure activity | 7 | 0.00% |
| Hotel Name | 7 | 0.00% |
| project name | 7 | 0.00% |
| business type | 7 | 0.00% |
| Real estate | 7 | 0.00% |
| Landform | 7 | 0.00% |
| Product Code | 7 | 0.00% |
| Page Numbers | 7 | 0.00% |
| Newspaper | 7 | 0.00% |
| Skin Condition | 7 | 0.00% |
| Accomplishment | 7 | 0.00% |
| physical process | 7 | 0.00% |
| building type | 7 | 0.00% |
| mealtime | 7 | 0.00% |
| narrative | 7 | 0.00% |
| regulation | 7 | 0.00% |
| Sentencing Location | 7 | 0.00% |
| Time of day | 7 | 0.00% |
| Football Player | 7 | 0.00% |
| discovery.observatory.name | 7 | 0.00% |
| Fitness Activity | 7 | 0.00% |
| software/tool | 7 | 0.00% |
| Retail store | 7 | 0.00% |
| Individual | 7 | 0.00% |
| promotional material | 7 | 0.00% |
| Service Offering | 7 | 0.00% |
| distance measurement | 7 | 0.00% |
| Political Figure | 7 | 0.00% |
| Methodology | 7 | 0.00% |
| kitchen tool | 7 | 0.00% |
| Sports Venue | 7 | 0.00% |
| webpage | 7 | 0.00% |
| Technical term | 7 | 0.00% |
| footwear | 7 | 0.00% |
| obstacle | 7 | 0.00% |
| Countries Joined the Organization | 7 | 0.00% |
| Location Entity | 7 | 0.00% |
| shape | 7 | 0.00% |
| Legal Case | 7 | 0.00% |
| - | 7 | 0.00% |
| Company/Product | 7 | 0.00% |
| Time/Period | 7 | 0.00% |
| Influenced People | 7 | 0.00% |
| asteroid.name | 7 | 0.00% |
| character name | 7 | 0.00% |
| product quality | 7 | 0.00% |
| Movie title | 7 | 0.00% |
| Bank | 7 | 0.00% |
| Kitchenware | 7 | 0.00% |
| Military unit | 7 | 0.00% |
| Military force | 7 | 0.00% |
| furniture item | 7 | 0.00% |
| Designer | 7 | 0.00% |
| Occupation/Activity | 7 | 0.00% |
| Product Attribute | 7 | 0.00% |
| result | 7 | 0.00% |
| Price Range | 7 | 0.00% |
| Business entity | 7 | 0.00% |
| Injury | 7 | 0.00% |
| Number of Aftershocks | 7 | 0.00% |
| Government Program | 7 | 0.00% |
| Hardware Component | 7 | 0.00% |
| visual representation | 7 | 0.00% |
| Product/Brand | 7 | 0.00% |
| Religious Organization | 7 | 0.00% |
| Personal Relationship | 7 | 0.00% |
| construction material | 7 | 0.00% |
| problem/concern | 7 | 0.00% |
| system | 7 | 0.00% |
| Utensil | 7 | 0.00% |
| Image | 7 | 0.00% |
| Weather Condition | 7 | 0.00% |
| Ingredient Type | 7 | 0.00% |
| arguments_for_using_gdz | 7 | 0.00% |
| social relationship | 7 | 0.00% |
| Hobby/Interest | 7 | 0.00% |
| Building material | 7 | 0.00% |
| Corporation | 7 | 0.00% |
| occupation/role | 7 | 0.00% |
| Event Type | 7 | 0.00% |
| communication device | 7 | 0.00% |
| Work | 7 | 0.00% |
| Church | 7 | 0.00% |
| celebrity with bristles | 7 | 0.00% |
| Architectural feature | 7 | 0.00% |
| Mechanical Component | 7 | 0.00% |
| organization-other | 7 | 0.00% |
| Clothing Brand | 7 | 0.00% |
| social gathering | 7 | 0.00% |
| Date Range | 7 | 0.00% |
| plumbing fixture | 7 | 0.00% |
| training program | 7 | 0.00% |
| required_documents | 7 | 0.00% |
| place name | 7 | 0.00% |
| company/vendor | 7 | 0.00% |
| Browser | 7 | 0.00% |
| disease.pathogenesis.symptoms | 7 | 0.00% |
| Rescue Organizer | 7 | 0.00% |
| hardware specification | 7 | 0.00% |
| bordering_municipalities | 7 | 0.00% |
| football player | 7 | 0.00% |
| geopolitical entity | 6 | 0.00% |
| nickname | 6 | 0.00% |
| decor | 6 | 0.00% |
| qualification | 6 | 0.00% |
| Business/Company | 6 | 0.00% |
| clothing accessory | 6 | 0.00% |
| Pet | 6 | 0.00% |
| Process/Action | 6 | 0.00% |
| methodology | 6 | 0.00% |
| asteroid.type | 6 | 0.00% |
| Home Decor | 6 | 0.00% |
| Pasta Type | 6 | 0.00% |
| designation | 6 | 0.00% |
| Postpone Reason | 6 | 0.00% |
| Person/Username | 6 | 0.00% |
| Object/Item | 6 | 0.00% |
| diagnosis | 6 | 0.00% |
| Aircraft | 6 | 0.00% |
| action/behavior | 6 | 0.00% |
| cooking utensil | 6 | 0.00% |
| Journal Title | 6 | 0.00% |
| hardware device | 6 | 0.00% |
| target_audience | 6 | 0.00% |
| Boxer | 6 | 0.00% |
| exercise | 6 | 0.00% |
| measurement tool | 6 | 0.00% |
| Spectacle End Time | 6 | 0.00% |
| Fashion brand | 6 | 0.00% |
| household appliance | 6 | 0.00% |
| holiday | 6 | 0.00% |
| Accommodation Type | 6 | 0.00% |
| Park | 6 | 0.00% |
| booking_options.partners | 6 | 0.00% |
| Education Program | 6 | 0.00% |
| Natural phenomenon | 6 | 0.00% |
| historical period | 6 | 0.00% |
| Place of Worship | 6 | 0.00% |
| vehicle part | 6 | 0.00% |
| skills | 6 | 0.00% |
| News Outlet | 6 | 0.00% |
| product design | 6 | 0.00% |
| version | 6 | 0.00% |
| National Park | 6 | 0.00% |
| education program | 6 | 0.00% |
| decision-making | 6 | 0.00% |
| expertise | 6 | 0.00% |
| mechanical component | 6 | 0.00% |
| Transportation Service | 6 | 0.00% |
| natural entity | 6 | 0.00% |
| computer hardware | 6 | 0.00% |
| ROOT | 6 | 0.00% |
| entity/group | 6 | 0.00% |
| inventory | 6 | 0.00% |
| Person name | 6 | 0.00% |
| healthcare | 6 | 0.00% |
| Education/Training | 6 | 0.00% |
| Monetary value | 6 | 0.00% |
| competition | 6 | 0.00% |
| legislation | 6 | 0.00% |
| review_id | 6 | 0.00% |
| product_name | 6 | 0.00% |
| performance metric | 6 | 0.00% |
| health issue | 6 | 0.00% |
| reviewer | 6 | 0.00% |
| Scenery | 6 | 0.00% |
| Conference | 6 | 0.00% |
| GPE | 6 | 0.00% |
| relationship status | 6 | 0.00% |
| branding | 6 | 0.00% |
| route | 6 | 0.00% |
| instructions | 6 | 0.00% |
| Cured Cases | 6 | 0.00% |
| outdoor activity | 6 | 0.00% |
| Sensation | 6 | 0.00% |
| State of being | 6 | 0.00% |
| Molecule | 6 | 0.00% |
| language proficiency | 6 | 0.00% |
| Sports term | 6 | 0.00% |
| Publication Title | 6 | 0.00% |
| Furniture Material | 6 | 0.00% |
| Train Agency | 6 | 0.00% |
| publication.otherTaxaDescribed | 6 | 0.00% |
| Expression of gratitude | 6 | 0.00% |
| business process | 6 | 0.00% |
| Compliment | 6 | 0.00% |
| Art Gallery | 6 | 0.00% |
| patientId | 6 | 0.00% |
| specialty | 6 | 0.00% |
| time management | 6 | 0.00% |
| Superlative | 6 | 0.00% |
| Electoral System | 6 | 0.00% |
| Time/Event | 6 | 0.00% |
| transaction type | 6 | 0.00% |
| Literary Genre | 6 | 0.00% |
| physical phenomenon | 6 | 0.00% |
| Marketing | 6 | 0.00% |
| Flooring material | 6 | 0.00% |
| Event Venue | 6 | 0.00% |
| unitOfMeasure | 6 | 0.00% |
| Trapped Depth | 6 | 0.00% |
| Academic Subject | 6 | 0.00% |
| Prison Term | 6 | 0.00% |
| Financial Market | 6 | 0.00% |
| Appointment Time | 6 | 0.00% |
| Instrument | 6 | 0.00% |
| government body | 6 | 0.00% |
| consumer | 6 | 0.00% |
| capacity | 6 | 0.00% |
| negative outcome | 6 | 0.00% |
| media format | 6 | 0.00% |
| Plant Part | 6 | 0.00% |
| values | 6 | 0.00% |
| Software/Tool | 6 | 0.00% |
| Non-profit organization | 6 | 0.00% |
| Online Store | 6 | 0.00% |
| modification | 6 | 0.00% |
| discovery.discoverer | 6 | 0.00% |
| user | 6 | 0.00% |
| physical environment | 6 | 0.00% |
| financial account | 6 | 0.00% |
| itemId | 6 | 0.00% |
| Service/Task | 6 | 0.00% |
| reporter | 6 | 0.00% |
| Fruit | 6 | 0.00% |
| River | 6 | 0.00% |
| Medical Practice | 6 | 0.00% |
| Status | 6 | 0.00% |
| transportation infrastructure | 6 | 0.00% |
| reservation | 6 | 0.00% |
| event_id | 6 | 0.00% |
| commercial entity | 6 | 0.00% |
| Noun phrase | 6 | 0.00% |
| ranking | 6 | 0.00% |
| Insurance Policy | 6 | 0.00% |
| actor name | 6 | 0.00% |
| Personality trait | 6 | 0.00% |
| Descriptor | 6 | 0.00% |
| Product Quality | 6 | 0.00% |
| Social Media | 6 | 0.00% |
| items | 6 | 0.00% |
| Online Platform | 6 | 0.00% |
| Hardware device | 6 | 0.00% |
| image | 6 | 0.00% |
| departureTime | 6 | 0.00% |
| Ship No. | 6 | 0.00% |
| Rescue Tool or Method | 6 | 0.00% |
| recipe ingredient | 6 | 0.00% |
| industry sector | 6 | 0.00% |
| time measurement | 6 | 0.00% |
| academic task | 6 | 0.00% |
| subject matter | 6 | 0.00% |
| accommodation type | 6 | 0.00% |
| Health Issue | 6 | 0.00% |
| project management | 6 | 0.00% |
| feeling/emotion | 6 | 0.00% |
| Brand/Product Name | 6 | 0.00% |
| hotel.booking_options.partners | 6 | 0.00% |
| hotel.alternative_names | 6 | 0.00% |
| Sports Achievement | 6 | 0.00% |
| geographic entity | 6 | 0.00% |
| Damaged Property | 6 | 0.00% |
| Holiday/Event | 6 | 0.00% |
| Occupation/Title | 6 | 0.00% |
| Time Frame | 6 | 0.00% |
| Insurance company | 6 | 0.00% |
| Financial Institution | 6 | 0.00% |
| child | 6 | 0.00% |
| Financial Service | 6 | 0.00% |
| Place/Location | 6 | 0.00% |
| Roadway | 6 | 0.00% |
| Visual media | 6 | 0.00% |
| Illness | 6 | 0.00% |
| specificationName | 6 | 0.00% |
| specificationValue | 6 | 0.00% |
| Description | 6 | 0.00% |
| game element | 6 | 0.00% |
| utensil | 6 | 0.00% |
| mineral | 6 | 0.00% |
| Business/Brand | 6 | 0.00% |
| Financial term | 6 | 0.00% |
| financial term | 6 | 0.00% |
| roomNumber | 6 | 0.00% |
| sentiment | 6 | 0.00% |
| commercial establishment | 6 | 0.00% |
| Job opportunity | 6 | 0.00% |
| genres | 6 | 0.00% |
| Business/Industry | 6 | 0.00% |
| Archaeologist Organization | 6 | 0.00% |
| Archaeologist | 6 | 0.00% |
| unitPrice | 6 | 0.00% |
| Street name | 6 | 0.00% |
| Evaluation | 6 | 0.00% |
| format | 6 | 0.00% |
| RNA | 6 | 0.00% |
| medical professionals | 6 | 0.00% |
| hobby | 6 | 0.00% |
| Withdraw Reason | 6 | 0.00% |
| Fee | 6 | 0.00% |
| GeneProtein | 6 | 0.00% |
| music.genres | 6 | 0.00% |
| Religious Entity | 6 | 0.00% |
| Consumer Group | 6 | 0.00% |
| parameter | 6 | 0.00% |
| Start_Date | 6 | 0.00% |
| Greeting | 6 | 0.00% |
| Car Part | 6 | 0.00% |
| required_skills | 6 | 0.00% |
| travel document | 6 | 0.00% |
| methodology.assessment_tool.categories_measured | 6 | 0.00% |
| stock price indicator | 6 | 0.00% |
| Music Production Element | 6 | 0.00% |
| range | 6 | 0.00% |
| color option | 6 | 0.00% |
| Agreement Validity Period | 5 | 0.00% |
| Positive feedback | 5 | 0.00% |
| service offering | 5 | 0.00% |
| flooring | 5 | 0.00% |
| firearm part | 5 | 0.00% |
| Fashion Designer | 5 | 0.00% |
| Legal term | 5 | 0.00% |
| initial_text.negations | 5 | 0.00% |
| orbital_characteristics.classification | 5 | 0.00% |
| discovery.date | 5 | 0.00% |
| student group | 5 | 0.00% |
| Car brand | 5 | 0.00% |
| Tourist attraction | 5 | 0.00% |
| asset | 5 | 0.00% |
| Geography | 5 | 0.00% |
| Music Piece | 5 | 0.00% |
| Car Brand | 5 | 0.00% |
| Photographer | 5 | 0.00% |
| ratings | 5 | 0.00% |
| job | 5 | 0.00% |
| appearance | 5 | 0.00% |
| real estate property | 5 | 0.00% |
| Company/Business | 5 | 0.00% |
| Person/Role | 5 | 0.00% |
| Spectacle Duration | 5 | 0.00% |
| serving_suggestions | 5 | 0.00% |
| transportation vehicle | 5 | 0.00% |
| expense | 5 | 0.00% |
| Physical Activity | 5 | 0.00% |
| research area | 5 | 0.00% |
| Economic Entity | 5 | 0.00% |
| Benefit | 5 | 0.00% |
| customer group | 5 | 0.00% |
| Launch Date | 5 | 0.00% |
| home decor item | 5 | 0.00% |
| percentage | 5 | 0.00% |
| Business Metric | 5 | 0.00% |
| Requirement | 5 | 0.00% |
| Time Range | 5 | 0.00% |
| lighting fixture | 5 | 0.00% |
| living being | 5 | 0.00% |
| Web browser | 5 | 0.00% |
| Element | 5 | 0.00% |
| economic system | 5 | 0.00% |
| claim_id | 5 | 0.00% |
| Medical device | 5 | 0.00% |
| military equipment | 5 | 0.00% |
| Framework | 5 | 0.00% |
| actor | 5 | 0.00% |
| Tool Feature | 5 | 0.00% |
| cast | 5 | 0.00% |
| maintenance | 5 | 0.00% |
| Event/Program | 5 | 0.00% |
| Accessory | 5 | 0.00% |
| Technology product | 5 | 0.00% |
| community | 5 | 0.00% |
| state of being | 5 | 0.00% |
| financial | 5 | 0.00% |
| direction | 5 | 0.00% |
| Design Style | 5 | 0.00% |
| abstract entity | 5 | 0.00% |
| Linguistics | 5 | 0.00% |
| astronomical object | 5 | 0.00% |
| orbital_characteristics.perihelion.unit | 5 | 0.00% |
| diameter.unit | 5 | 0.00% |
| Physical condition | 5 | 0.00% |
| company name | 5 | 0.00% |
| Financial instrument | 5 | 0.00% |
| medications | 5 | 0.00% |
| human resources | 5 | 0.00% |
| Government Institution | 5 | 0.00% |
| printer model | 5 | 0.00% |
| conj | 5 | 0.00% |
| Tennis player | 5 | 0.00% |
| spice | 5 | 0.00% |
| Event/Action | 5 | 0.00% |
| Art | 5 | 0.00% |
| transportation hub | 5 | 0.00% |
| Day | 5 | 0.00% |
| greeting | 5 | 0.00% |
| manufacturing process | 5 | 0.00% |
| Building Feature | 5 | 0.00% |
| Safety Feature | 5 | 0.00% |
| Government official | 5 | 0.00% |
| location.city | 5 | 0.00% |
| personal preference | 5 | 0.00% |
| Residential Property | 5 | 0.00% |
| Religious Building | 5 | 0.00% |
| Mileage | 5 | 0.00% |
| opponent | 5 | 0.00% |
| strategy | 5 | 0.00% |
| financial status | 5 | 0.00% |
| username | 5 | 0.00% |
| pricing strategy | 5 | 0.00% |
| Legal Services | 5 | 0.00% |
| Trait/Quality | 5 | 0.00% |
| Food/Drink | 5 | 0.00% |
| motivation | 5 | 0.00% |
| recommendation | 5 | 0.00% |
| social interaction | 5 | 0.00% |
| recipient | 5 | 0.00% |
| Business Service | 5 | 0.00% |
| online presence | 5 | 0.00% |
| Format | 5 | 0.00% |
| physical symptom | 5 | 0.00% |
| Event/Tradition | 5 | 0.00% |
| game title | 5 | 0.00% |
| Footwear | 5 | 0.00% |
| Historical Entity | 5 | 0.00% |
| Electronic Component | 5 | 0.00% |
| opportunity | 5 | 0.00% |
| Design feature | 5 | 0.00% |
| technology infrastructure | 5 | 0.00% |
| Art style | 5 | 0.00% |
| Credential | 5 | 0.00% |
| TV Channel | 5 | 0.00% |
| action/verb | 5 | 0.00% |
| Software tool | 5 | 0.00% |
| Volume Number | 5 | 0.00% |
| funding_source | 5 | 0.00% |
| Assessment | 5 | 0.00% |
| CellLine | 5 | 0.00% |
| Organism | 5 | 0.00% |
| injury | 5 | 0.00% |
| Food/beverage | 5 | 0.00% |
| Online platform | 5 | 0.00% |
| Director | 5 | 0.00% |
| weapon | 5 | 0.00% |
| Procedure | 5 | 0.00% |
| gender identity | 5 | 0.00% |
| Zip Code | 5 | 0.00% |
| Theme Park | 5 | 0.00% |
| Marketing Material | 5 | 0.00% |
| discovery.observatory.location.country | 5 | 0.00% |
| Chemical Substance | 5 | 0.00% |
| Dosage | 5 | 0.00% |
| job opportunity | 5 | 0.00% |
| Outdoor space | 5 | 0.00% |
| Operating system | 5 | 0.00% |
| consumer group | 5 | 0.00% |
| hotel.services | 5 | 0.00% |
| safety feature | 5 | 0.00% |
| Printer Model | 5 | 0.00% |
| Transportation service | 5 | 0.00% |
| Law Enforcement | 5 | 0.00% |
| Document type | 5 | 0.00% |
| Number of People Hospitalized | 5 | 0.00% |
| Airport | 5 | 0.00% |
| Event Name | 5 | 0.00% |
| bodily fluid | 5 | 0.00% |
| fashion trend | 5 | 0.00% |
| fashion/style | 5 | 0.00% |
| owner | 5 | 0.00% |
| Art supplies | 5 | 0.00% |
| Product name | 5 | 0.00% |
| discount code | 5 | 0.00% |
| Hospital | 5 | 0.00% |
| Social issue | 5 | 0.00% |
| Database | 5 | 0.00% |
| consumer behavior | 5 | 0.00% |
| Prison | 5 | 0.00% |
| Venue Name | 5 | 0.00% |
| Breakfast item | 5 | 0.00% |
| Movie/Entertainment | 5 | 0.00% |
| Street | 5 | 0.00% |
| technical issue | 5 | 0.00% |
| Performance Metric | 5 | 0.00% |
| Skillset | 5 | 0.00% |
| createdAt | 5 | 0.00% |
| anatomy | 5 | 0.00% |
| exercise facility | 5 | 0.00% |
| supplierId | 5 | 0.00% |
| supplierName | 5 | 0.00% |
| profession/group | 5 | 0.00% |
| latitude | 5 | 0.00% |
| longitude | 5 | 0.00% |
| Software Feature | 5 | 0.00% |
| action type | 5 | 0.00% |
| Maximum Rainfall | 5 | 0.00% |
| workspace | 5 | 0.00% |
| name: Full name of the patient | 5 | 0.00% |
| Legal | 5 | 0.00% |
| Political Affiliation | 5 | 0.00% |
| state system | 5 | 0.00% |
| Opinion | 5 | 0.00% |
| Brewery | 5 | 0.00% |
| e_33 | 5 | 0.00% |
| Society | 5 | 0.00% |
| Music project | 5 | 0.00% |
| crafting material | 5 | 0.00% |
| communication entity | 5 | 0.00% |
| Time entity | 5 | 0.00% |
| financial service | 5 | 0.00% |
| Control mode | 5 | 0.00% |
| discount | 5 | 0.00% |
| Exercise | 5 | 0.00% |
| purpose | 5 | 0.00% |
| context | 5 | 0.00% |
| manner | 5 | 0.00% |
| task management | 5 | 0.00% |
| Technical Issue | 5 | 0.00% |
| training_modules | 5 | 0.00% |
| Reward | 5 | 0.00% |
| Celebration | 5 | 0.00% |
| educational setting | 5 | 0.00% |
| Financial product | 5 | 0.00% |
| Drug | 5 | 0.00% |
| total | 5 | 0.00% |
| idea | 5 | 0.00% |
| Medical Test | 5 | 0.00% |
| Activity/Event | 5 | 0.00% |
| Sports award | 5 | 0.00% |
| reaction | 5 | 0.00% |
| healthcare facility | 5 | 0.00% |
| Software Application | 5 | 0.00% |
| arrivalTime | 5 | 0.00% |
| legal status | 5 | 0.00% |
| project_id | 5 | 0.00% |
| conflict | 5 | 0.00% |
| Sports Equipment | 5 | 0.00% |
| Culture | 5 | 0.00% |
| Comparison | 5 | 0.00% |
| assay | 5 | 0.00% |
| Technical Term | 5 | 0.00% |
| life event | 5 | 0.00% |
| Literary Device | 5 | 0.00% |
| Location/City | 5 | 0.00% |
| Time/Duration | 5 | 0.00% |
| Availability | 5 | 0.00% |
| Real Estate Entity | 5 | 0.00% |
| Library | 5 | 0.00% |
| Interest | 5 | 0.00% |
| Retailer | 5 | 0.00% |
| Offer | 5 | 0.00% |
| website metric | 5 | 0.00% |
| location/place | 5 | 0.00% |
| Financial Asset | 5 | 0.00% |
| occupation/group | 5 | 0.00% |
| releaseDate | 5 | 0.00% |
| culture | 5 | 0.00% |
| Physical Quantity | 5 | 0.00% |
| pet | 5 | 0.00% |
| social activity | 5 | 0.00% |
| Franchise | 5 | 0.00% |
| seafood | 5 | 0.00% |
| Medical Field | 5 | 0.00% |
| impact | 5 | 0.00% |
| Disaster-Stricken Farmland | 5 | 0.00% |
| medical equipment | 5 | 0.00% |
| Sports Facility | 5 | 0.00% |
| possession | 5 | 0.00% |
| end_date | 5 | 0.00% |
| life stage | 5 | 0.00% |
| mobile application | 5 | 0.00% |
| physical state | 5 | 0.00% |
| Opportunity | 5 | 0.00% |
| Person/Animal | 5 | 0.00% |
| Athlete/Player | 5 | 0.00% |
| fashion accessory | 5 | 0.00% |
| Social Class | 5 | 0.00% |
| zip_code | 5 | 0.00% |
| findings | 5 | 0.00% |
| Financial entity | 5 | 0.00% |
| Medical Equipment | 5 | 0.00% |
| Creative Work | 5 | 0.00% |
| Creation process | 5 | 0.00% |
| Safety | 5 | 0.00% |
| person/occupation | 5 | 0.00% |
| visual object | 5 | 0.00% |
| dimension | 5 | 0.00% |
| event/location | 5 | 0.00% |
| policy_id | 5 | 0.00% |
| away_team | 5 | 0.00% |
| home_team | 5 | 0.00% |
| подправка | 5 | 0.00% |
| Real estate agency | 5 | 0.00% |
| celestial object | 5 | 0.00% |
| government policy | 5 | 0.00% |
| technical specification | 5 | 0.00% |
| Type of entity | 5 | 0.00% |
| URL | 5 | 0.00% |
| academic institution | 5 | 0.00% |
| Device Type | 5 | 0.00% |
| Local Venue | 5 | 0.00% |
| digital media | 5 | 0.00% |
| Natural Landmark | 5 | 0.00% |
| Car Type | 5 | 0.00% |
| Interior Design | 5 | 0.00% |
| Person/Organization | 5 | 0.00% |
| communication tool | 5 | 0.00% |
| music group | 5 | 0.00% |
| Government agency | 5 | 0.00% |
| full_name | 5 | 0.00% |
| Media outlet | 5 | 0.00% |
| Physical Structure | 5 | 0.00% |
| Author name | 5 | 0.00% |
| songName | 5 | 0.00% |
| songLength | 5 | 0.00% |
| songPosition | 5 | 0.00% |
| Hairstyle | 5 | 0.00% |
| Action/Activity | 5 | 0.00% |
| temporal entity | 5 | 0.00% |
| Technology/Software | 5 | 0.00% |
| Scholarship | 5 | 0.00% |
| artistic technique | 5 | 0.00% |
| Literary Work | 5 | 0.00% |
| chemical process | 5 | 0.00% |
| movie title | 5 | 0.00% |
| Housing Type | 5 | 0.00% |
| Sensory Perception/Quality | 5 | 0.00% |
| Chemical Element | 5 | 0.00% |
| Environmental Issue | 5 | 0.00% |
| Province | 5 | 0.00% |
| Opera | 5 | 0.00% |
| Hardware component | 5 | 0.00% |
| physical measurement | 5 | 0.00% |
| Property/Feature | 5 | 0.00% |
| Action/Task | 5 | 0.00% |
| martial arts discipline | 5 | 0.00% |
| ratingValue | 5 | 0.00% |
| registration.steps | 5 | 0.00% |
| registration.data_required | 5 | 0.00% |
| safety equipment | 5 | 0.00% |
| min | 5 | 0.00% |
| max | 5 | 0.00% |
| supplement ingredient | 5 | 0.00% |
| Software version | 5 | 0.00% |
| part of home | 5 | 0.00% |
| Royalty | 5 | 0.00% |
| Material Property | 5 | 0.00% |
| restaurant.events | 5 | 0.00% |
| Government Building | 5 | 0.00% |
| Network port | 5 | 0.00% |
| Physical Media | 5 | 0.00% |
| Protected Characteristic | 5 | 0.00% |
| descriptors | 5 | 0.00% |
| Committee | 5 | 0.00% |
| disease_profile.typical_manifestations | 5 | 0.00% |
| protein-region | 5 | 0.00% |
| State/Condition | 5 | 0.00% |
| causes | 5 | 0.00% |
| Business Location | 5 | 0.00% |
| space/room | 5 | 0.00% |
| art supplies | 5 | 0.00% |
| punct | 5 | 0.00% |
| linguistic_framework.comparison_aspects | 5 | 0.00% |
| hotel.amenities | 5 | 0.00% |
| level | 5 | 0.00% |
| Fashion Collection | 5 | 0.00% |
| LegalAction | 5 | 0.00% |
| booking_options | 5 | 0.00% |
| License action | 5 | 0.00% |
| architectural entity | 5 | 0.00% |
| Physical state | 5 | 0.00% |
| product part | 5 | 0.00% |
| Carotenoid | 5 | 0.00% |
| gear/accessory | 5 | 0.00% |
| company/brand name | 5 | 0.00% |
| ageRange | 5 | 0.00% |
| Cooking Instruction | 5 | 0.00% |
| Utility room | 5 | 0.00% |
| Evaluation Score | 5 | 0.00% |
| Audio setting | 5 | 0.00% |
| specialized track | 5 | 0.00% |
| research_framework_items | 4 | 0.00% |
| organization.name | 4 | 0.00% |
| modelNumber | 4 | 0.00% |
| purchaseDate | 4 | 0.00% |
| Occupation/Group | 4 | 0.00% |
| Mobile application | 4 | 0.00% |
| financial burden | 4 | 0.00% |
| firearm model | 4 | 0.00% |
| insurance type | 4 | 0.00% |
| personal pronoun | 4 | 0.00% |
| Physical property | 4 | 0.00% |
| Project Name | 4 | 0.00% |
| physical_characteristics.observations.0.publication_year | 4 | 0.00% |
| allergies | 4 | 0.00% |
| sustainability | 4 | 0.00% |
| space | 4 | 0.00% |
| Sales | 4 | 0.00% |
| Government policy | 4 | 0.00% |
| variable | 4 | 0.00% |
| cosmetic product | 4 | 0.00% |
| Function | 4 | 0.00% |
| routine | 4 | 0.00% |
| administrative division | 4 | 0.00% |
| Financial Metric | 4 | 0.00% |
| Tour Package | 4 | 0.00% |
| Jewelry item | 4 | 0.00% |
| Racing team | 4 | 0.00% |
| Spokenmen | 4 | 0.00% |
| transaction_id | 4 | 0.00% |
| exchange | 4 | 0.00% |
| Document Title | 4 | 0.00% |
| Crop | 4 | 0.00% |
| Vehicle Parts | 4 | 0.00% |
| Coloring Page | 4 | 0.00% |
| Human beings | 4 | 0.00% |
| genus | 4 | 0.00% |
| Home Decor Item | 4 | 0.00% |
| Natural Environment | 4 | 0.00% |
| visual art | 4 | 0.00% |
| Service offering | 4 | 0.00% |
| Research Agency | 4 | 0.00% |
| Attraction | 4 | 0.00% |
| sound | 4 | 0.00% |
| equipment subtype | 4 | 0.00% |
| Annoncement Date | 4 | 0.00% |
| Environmental Condition | 4 | 0.00% |
| Protective gear | 4 | 0.00% |
| planet | 4 | 0.00% |
| Animal species | 4 | 0.00% |
| Academic Field | 4 | 0.00% |
| choice | 4 | 0.00% |
| business metric | 4 | 0.00% |
| phenomenon | 4 | 0.00% |
| policy_number | 4 | 0.00% |
| missions | 4 | 0.00% |
| Service type | 4 | 0.00% |
| facial expression | 4 | 0.00% |
| movie | 4 | 0.00% |
| Challenge | 4 | 0.00% |
| Online Community | 4 | 0.00% |
| political party | 4 | 0.00% |
| procedure | 4 | 0.00% |
| Date-Time | 4 | 0.00% |
| professionals | 4 | 0.00% |
| Chemical substance | 4 | 0.00% |
| Political group | 4 | 0.00% |
| Website/Platform | 4 | 0.00% |
| terrain | 4 | 0.00% |
| Business Partners | 4 | 0.00% |
| weather | 4 | 0.00% |
| physical_characteristics.observations.0.telescope | 4 | 0.00% |
| environmental impact | 4 | 0.00% |
| food/beverage | 4 | 0.00% |
| Relationship status | 4 | 0.00% |
| fabric type | 4 | 0.00% |
| clothing type | 4 | 0.00% |
| Food Product | 4 | 0.00% |
| Programming language | 4 | 0.00% |
| jewelry component | 4 | 0.00% |
| Visual art | 4 | 0.00% |
| commodity | 4 | 0.00% |
| Date/time | 4 | 0.00% |
| Competition | 4 | 0.00% |
| Service Years | 4 | 0.00% |
| scenery | 4 | 0.00% |
| Page Number | 4 | 0.00% |
| origin | 4 | 0.00% |
| instructional material | 4 | 0.00% |
| surface | 4 | 0.00% |
| Motorcycle Model | 4 | 0.00% |
| Safety Equipment | 4 | 0.00% |
| literature | 4 | 0.00% |
| Dentist | 4 | 0.00% |
| Celebrity | 4 | 0.00% |
| Kitchen Utensil | 4 | 0.00% |
| Date/Event | 4 | 0.00% |
| Target Audience | 4 | 0.00% |
| Phenomenon | 4 | 0.00% |
| organelle | 4 | 0.00% |
| Water body | 4 | 0.00% |
| Tree type | 4 | 0.00% |
| reward | 4 | 0.00% |
| interest | 4 | 0.00% |
| musical element | 4 | 0.00% |
| lighting | 4 | 0.00% |
| visual aesthetics | 4 | 0.00% |
| Susceptible Population | 4 | 0.00% |
| Golf Course | 4 | 0.00% |
| Negation | 4 | 0.00% |
| societal construct | 4 | 0.00% |
| reviewId | 4 | 0.00% |
| travel package | 4 | 0.00% |
| text | 4 | 0.00% |
| Clothing Style | 4 | 0.00% |
| Castle | 4 | 0.00% |
| ownership | 4 | 0.00% |
| Political ideology | 4 | 0.00% |
| physical place | 4 | 0.00% |
| funding_amount | 4 | 0.00% |
| contract template | 4 | 0.00% |
| website section | 4 | 0.00% |
| Compound | 4 | 0.00% |
| Band/Group | 4 | 0.00% |
| criteria | 4 | 0.00% |
| Record label | 4 | 0.00% |
| Album title | 4 | 0.00% |
| Home Improvement | 4 | 0.00% |
| Game genre | 4 | 0.00% |
| dish ingredient/recipe component | 4 | 0.00% |
| communication medium | 4 | 0.00% |
| Construction Material | 4 | 0.00% |
| Unknown Entity | 4 | 0.00% |
| Specification | 4 | 0.00% |
| water source | 4 | 0.00% |
| Car Color | 4 | 0.00% |
| negative feedback | 4 | 0.00% |
| Preposition | 4 | 0.00% |
| Cricketer | 4 | 0.00% |
| situation | 4 | 0.00% |
| job function | 4 | 0.00% |
| Negative Outcome | 4 | 0.00% |
| software improvement | 4 | 0.00% |
| occasion | 4 | 0.00% |
| Environmental Factor | 4 | 0.00% |
| priority | 4 | 0.00% |
| People/Group | 4 | 0.00% |
| Building Type | 4 | 0.00% |
| Organization Leader | 4 | 0.00% |
| Decoration | 4 | 0.00% |
| Housing | 4 | 0.00% |
| Infrastructure Entity | 4 | 0.00% |
| Winery | 4 | 0.00% |
| Discount code | 4 | 0.00% |
| Trading platform | 4 | 0.00% |
| toy | 4 | 0.00% |
| Product Entity | 4 | 0.00% |
| Supplement | 4 | 0.00% |
| affiliation | 4 | 0.00% |
| Airline | 4 | 0.00% |
| Course/Program | 4 | 0.00% |
| sender | 4 | 0.00% |
| paymentTerms | 4 | 0.00% |
| crime | 4 | 0.00% |
| occupation/interest | 4 | 0.00% |
| money | 4 | 0.00% |
| Event Ticket | 4 | 0.00% |
| Ticket Type | 4 | 0.00% |
| Game Mode | 4 | 0.00% |
| business sector | 4 | 0.00% |
| watch part | 4 | 0.00% |
| Technical Error | 4 | 0.00% |
| residence | 4 | 0.00% |
| transportation route | 4 | 0.00% |
| Product line | 4 | 0.00% |
| Social entity | 4 | 0.00% |
| features_and_benefits | 4 | 0.00% |
| Personal Development | 4 | 0.00% |
| benefit | 4 | 0.00% |
| Clothing brand | 4 | 0.00% |
| Video Game Character | 4 | 0.00% |
| historical figure | 4 | 0.00% |
| Dental Clinic | 4 | 0.00% |
| Government Policy | 4 | 0.00% |
| Measurement unit | 4 | 0.00% |
| Individuals | 4 | 0.00% |
| creature | 4 | 0.00% |
| hotel | 4 | 0.00% |
| Road | 4 | 0.00% |
| art | 4 | 0.00% |
| N/A | 4 | 0.00% |
| negative consequence | 4 | 0.00% |
| assessment tool | 4 | 0.00% |
| Political Group | 4 | 0.00% |
| assists | 4 | 0.00% |
| deliveryDate | 4 | 0.00% |
| protective gear | 4 | 0.00% |
| identification | 4 | 0.00% |
| security | 4 | 0.00% |
| Financial Information | 4 | 0.00% |
| Personal pronoun | 4 | 0.00% |
| gemstone | 4 | 0.00% |
| medical specialty | 4 | 0.00% |
| Asset | 4 | 0.00% |
| physical condition | 4 | 0.00% |
| Event/Occasion | 4 | 0.00% |
| media content | 4 | 0.00% |
| creative expression | 4 | 0.00% |
| natural water feature | 4 | 0.00% |
| Event/Competition | 4 | 0.00% |
| audio equipment | 4 | 0.00% |
| woodworking equipment | 4 | 0.00% |
| sports facility | 4 | 0.00% |
| market | 4 | 0.00% |
| flooring material | 4 | 0.00% |
| action/program | 4 | 0.00% |
| Inquiry | 4 | 0.00% |
| neighborhood | 4 | 0.00% |
| Magnitude(Tsunami heights) | 4 | 0.00% |
| biological classification | 4 | 0.00% |
| Academic Journal | 4 | 0.00% |
| suggestion | 4 | 0.00% |
| Journal | 4 | 0.00% |
| Death Age | 4 | 0.00% |
| Celestial Body | 4 | 0.00% |
| Museum/Attraction | 4 | 0.00% |
| Publication Name | 4 | 0.00% |
| Student name | 4 | 0.00% |
| Digital Platform | 4 | 0.00% |
| Business/Organization | 4 | 0.00% |
| Profession/Title | 4 | 0.00% |
| Entertainment Service | 4 | 0.00% |
| Sensory Perception | 4 | 0.00% |
| Architectural Style | 4 | 0.00% |
| durability | 4 | 0.00% |
| accomplishment | 4 | 0.00% |
| app_name | 4 | 0.00% |
| Task/Action | 4 | 0.00% |
| plural noun | 4 | 0.00% |
| determiner | 4 | 0.00% |
| superlative | 4 | 0.00% |
| sports | 4 | 0.00% |
| departureDate | 4 | 0.00% |
| Personal quality | 4 | 0.00% |
| developmental stage | 4 | 0.00% |
| Color Combination | 4 | 0.00% |
| Natural Entity | 4 | 0.00% |
| Service Quality | 4 | 0.00% |
| Labor Union | 4 | 0.00% |
| insurance | 4 | 0.00% |
| Advertisement | 4 | 0.00% |
| categoryName | 4 | 0.00% |
| Government body | 4 | 0.00% |
| Modification | 4 | 0.00% |
| setting | 4 | 0.00% |
| solution | 4 | 0.00% |
| lodging | 4 | 0.00% |
| taste/flavor | 4 | 0.00% |
| dental condition | 4 | 0.00% |
| Amplifier | 4 | 0.00% |
| Preparation | 4 | 0.00% |
| Natural Element | 4 | 0.00% |
| behavior trait | 4 | 0.00% |
| community members | 4 | 0.00% |
| academic subject | 4 | 0.00% |
| Biological Process | 4 | 0.00% |
| Building type | 4 | 0.00% |
| product model | 4 | 0.00% |
| Product code | 4 | 0.00% |
| Bathroom Fixture | 4 | 0.00% |
| Music group | 4 | 0.00% |
| children | 4 | 0.00% |
| fee | 4 | 0.00% |
| personal care product | 4 | 0.00% |
| Person/Title | 4 | 0.00% |
| Hotel/Resort | 4 | 0.00% |
| Experience/Emotion | 4 | 0.00% |
| study | 4 | 0.00% |
| religious event | 4 | 0.00% |
| supernatural beliefs | 4 | 0.00% |
| function name | 4 | 0.00% |
| military unit | 4 | 0.00% |
| Sports statistic | 4 | 0.00% |
| Human Group | 4 | 0.00% |
| circumstance | 4 | 0.00% |
| camera feature | 4 | 0.00% |
| Statistic | 4 | 0.00% |
| interaction | 4 | 0.00% |
| Data Type | 4 | 0.00% |
| Musician/group | 4 | 0.00% |
| Medical Device | 4 | 0.00% |
| Connection | 4 | 0.00% |
| Service/Program | 4 | 0.00% |
| Amusement Park | 4 | 0.00% |
| meal category | 4 | 0.00% |
| Compensation | 4 | 0.00% |
| Furniture Piece | 4 | 0.00% |
| Intellectual Property | 4 | 0.00% |
| Group/Organization | 4 | 0.00% |
| Government Department | 4 | 0.00% |
| Military personnel | 4 | 0.00% |
| Business/Store | 4 | 0.00% |
| episodeNumber | 4 | 0.00% |
| synopsis | 4 | 0.00% |
| revenue | 4 | 0.00% |
| crop | 4 | 0.00% |
| Education Degree | 4 | 0.00% |
| Protocol | 4 | 0.00% |
| transportation company | 4 | 0.00% |
| Body of water | 4 | 0.00% |
| Artistic Expression | 4 | 0.00% |
| unknown | 4 | 0.00% |
| Punctuation | 4 | 0.00% |
| financial assistance | 4 | 0.00% |
| safety measure | 4 | 0.00% |
| Maintenance task | 4 | 0.00% |
| currency value | 4 | 0.00% |
| element | 4 | 0.00% |
| cell line | 4 | 0.00% |
| type of entity | 4 | 0.00% |
| hire_date | 4 | 0.00% |
| Territory | 4 | 0.00% |
| cell type | 4 | 0.00% |
| Weather phenomenon | 4 | 0.00% |
| Home decor | 4 | 0.00% |
| Personal Experience | 4 | 0.00% |
| visual quality | 4 | 0.00% |
| military rank | 4 | 0.00% |
| positive experience | 4 | 0.00% |
| services | 4 | 0.00% |
| Title of an anime/manga | 4 | 0.00% |
| Social innovation initiative | 4 | 0.00% |
| Housing Unit | 4 | 0.00% |
| parent | 4 | 0.00% |
| electrical component | 4 | 0.00% |
| Management | 4 | 0.00% |
| Artistic Work | 4 | 0.00% |
| Source | 4 | 0.00% |
| location type | 4 | 0.00% |
| Geographical Feature | 4 | 0.00% |
| Nickname | 4 | 0.00% |
| Production Company | 4 | 0.00% |
| Mechanical Part | 4 | 0.00% |
| direction/location | 4 | 0.00% |
| Living Space | 4 | 0.00% |
| Emotion/Experience | 4 | 0.00% |
| player_id | 4 | 0.00% |
| Recreational area | 4 | 0.00% |
| Contact Info | 4 | 0.00% |
| Service Offered | 4 | 0.00% |
| Football Statistic | 4 | 0.00% |
| testing method | 4 | 0.00% |
| organ | 4 | 0.00% |
| Facilities | 4 | 0.00% |
| Beach | 4 | 0.00% |
| Contest | 4 | 0.00% |
| business relationship | 4 | 0.00% |
| beverage type | 4 | 0.00% |
| Transportation Company | 4 | 0.00% |
| Business metric | 4 | 0.00% |
| salary | 4 | 0.00% |
| Researcher | 4 | 0.00% |
| Brand/Store | 4 | 0.00% |
| due_date | 4 | 0.00% |
| Manufacturing process | 4 | 0.00% |
| Transportation Infrastructure | 4 | 0.00% |
| Property Division | 4 | 0.00% |
| Artistic Style | 4 | 0.00% |
| law enforcement | 4 | 0.00% |
| printing method | 4 | 0.00% |
| Medical Institution | 4 | 0.00% |
| difficulty | 4 | 0.00% |
| fixture | 4 | 0.00% |
| Jewelry Item | 4 | 0.00% |
| issueDate | 4 | 0.00% |
| copay | 4 | 0.00% |
| song title | 4 | 0.00% |
| play structure | 4 | 0.00% |
| Religious text | 4 | 0.00% |
| Website feature | 4 | 0.00% |
| Field of study/work | 4 | 0.00% |
| Geopolitical entity | 4 | 0.00% |
| socioeconomic status | 4 | 0.00% |
| Visual Content | 4 | 0.00% |
| Business Type | 4 | 0.00% |
| Actor Name | 4 | 0.00% |
| skill set | 4 | 0.00% |
| Business type | 4 | 0.00% |
| technology/tool | 4 | 0.00% |
| Product Accessory | 4 | 0.00% |
| alternative_names | 4 | 0.00% |
| fashion | 4 | 0.00% |
| Reasons for the Formation of the Historical sites | 4 | 0.00% |
| bodily function | 4 | 0.00% |
| Industry/Field | 4 | 0.00% |
| Clothing accessory | 4 | 0.00% |
| Functionality/Feature | 4 | 0.00% |
| Geographical Area | 4 | 0.00% |
| Frequency | 4 | 0.00% |
| testName | 4 | 0.00% |
| game mechanics | 4 | 0.00% |
| negation | 4 | 0.00% |
| Branding | 4 | 0.00% |
| color scheme | 4 | 0.00% |
| geographical coordinates | 4 | 0.00% |
| Sensory experience | 4 | 0.00% |
| totalPrice | 4 | 0.00% |
| Dog Breed | 4 | 0.00% |
| effect | 4 | 0.00% |
| Software/Product | 4 | 0.00% |
| Task/Functionality | 4 | 0.00% |
| Best Way to Shoot | 4 | 0.00% |
| expectation | 4 | 0.00% |
| Communication platform | 4 | 0.00% |
| Literary Element | 4 | 0.00% |
| candy bar | 4 | 0.00% |
| protagonist | 4 | 0.00% |
| Property type | 4 | 0.00% |
| Trading Card Set | 4 | 0.00% |
| Physical sensation | 4 | 0.00% |
| Situation | 4 | 0.00% |
| digital tool | 4 | 0.00% |
| recommendations | 4 | 0.00% |
| Technical Specification | 4 | 0.00% |
| terrain feature | 4 | 0.00% |
| person.us_senator.years | 4 | 0.00% |
| activity/purpose | 4 | 0.00% |
| Route | 4 | 0.00% |
| Software Version | 4 | 0.00% |
| architecture element | 4 | 0.00% |
| Window Treatment | 4 | 0.00% |
| genus.related_genera | 4 | 0.00% |
| publisher | 4 | 0.00% |
| mistake | 4 | 0.00% |
| trait/quality | 4 | 0.00% |
| landscape feature | 4 | 0.00% |
| IT infrastructure | 4 | 0.00% |
| Performer | 4 | 0.00% |
| personal identifier | 4 | 0.00% |
| Art medium | 4 | 0.00% |
| Food/Beverage | 4 | 0.00% |
| ratingId | 4 | 0.00% |
| ratingName | 4 | 0.00% |
| activity/service | 4 | 0.00% |
| personal value | 4 | 0.00% |
| Physical Trait | 4 | 0.00% |
| administrative leader | 4 | 0.00% |
| phrase | 4 | 0.00% |
| Group of Individuals | 4 | 0.00% |
| product code | 4 | 0.00% |
| Customer Group | 4 | 0.00% |
| harmful substance | 4 | 0.00% |
| Place/Venue | 4 | 0.00% |
| mobile device | 4 | 0.00% |
| Pet Name | 4 | 0.00% |
| Schedule | 4 | 0.00% |
| Online Alias | 4 | 0.00% |
| Type of Entertainment | 4 | 0.00% |
| Memory | 4 | 0.00% |
| Home feature | 4 | 0.00% |
| Hazard | 4 | 0.00% |
| meal type | 4 | 0.00% |
| activityAvailability | 4 | 0.00% |
| health | 4 | 0.00% |
| employment status | 4 | 0.00% |
| Sports Conference | 4 | 0.00% |
| Product/Component | 4 | 0.00% |
| human beings | 4 | 0.00% |
| Power Tool | 4 | 0.00% |
| Connection/Network | 4 | 0.00% |
| Quality Rating | 4 | 0.00% |
| Household item | 4 | 0.00% |
| Social media handle | 4 | 0.00% |
| Transportation mode | 4 | 0.00% |
| Legal process | 4 | 0.00% |
| Exam Type | 4 | 0.00% |
| stepNumber | 4 | 0.00% |
| tolerance | 4 | 0.00% |
| leather type | 4 | 0.00% |
| material property | 4 | 0.00% |
| Rank | 4 | 0.00% |
| connectedPorts | 4 | 0.00% |
| Vehicle Model Name | 4 | 0.00% |
| grouping | 4 | 0.00% |
| Amount | 4 | 0.00% |
| patient_case.clinical_presentation.skin_lesions.characteristics | 4 | 0.00% |
| economics | 4 | 0.00% |
| Cooking | 4 | 0.00% |
| storage capacity | 4 | 0.00% |
| galaxy.alternative_designations | 4 | 0.00% |
| Football team | 4 | 0.00% |
| discount offer | 4 | 0.00% |
| storage furniture | 4 | 0.00% |
| spatial location | 4 | 0.00% |
| disease.pathogenesis.risk_factors | 4 | 0.00% |
| Ethical standard | 4 | 0.00% |
| Coloring Pages | 4 | 0.00% |
| Building feature | 4 | 0.00% |
| music album | 4 | 0.00% |
| fort material | 4 | 0.00% |
| yield | 4 | 0.00% |
| Education Entity | 4 | 0.00% |
| child care service | 4 | 0.00% |
| Study material | 4 | 0.00% |
| Sports Move | 4 | 0.00% |
| file_name | 4 | 0.00% |
| line_number | 4 | 0.00% |
| article.products | 4 | 0.00% |
| Spice | 4 | 0.00% |
| home appliance | 4 | 0.00% |
| Medical equipment | 4 | 0.00% |
| Cryptic Clue Element | 4 | 0.00% |
| content_analysis.themes | 4 | 0.00% |
| product material | 4 | 0.00% |
| offering | 4 | 0.00% |
| Media Format | 4 | 0.00% |
| Video bingo game | 4 | 0.00% |
| Theatre Work | 4 | 0.00% |
| Relationship Status | 4 | 0.00% |
| Smartphone Model | 4 | 0.00% |
| computing device | 4 | 0.00% |
| engine type | 4 | 0.00% |
| visual design | 4 | 0.00% |
| Android app | 4 | 0.00% |
| sports equipment | 4 | 0.00% |
| People Casting Key Votes | 4 | 0.00% |
| System/Technology | 4 | 0.00% |
| musical equipment | 4 | 0.00% |
| Testing facility | 4 | 0.00% |
| Tree characteristic | 4 | 0.00% |
| Grinder Pumps | 4 | 0.00% |
| contraction | 4 | 0.00% |
| Lighting technology | 4 | 0.00% |
| inputMaterialIds | 4 | 0.00% |
| Building/Architecture | 4 | 0.00% |
| training material | 4 | 0.00% |
| interface | 3 | 0.00% |
| Listing Number | 3 | 0.00% |
| Area Measurement | 3 | 0.00% |
| software program | 3 | 0.00% |
| patient | 3 | 0.00% |
| user interface | 3 | 0.00% |
| familial relation | 3 | 0.00% |
| review_date | 3 | 0.00% |
| service request | 3 | 0.00% |
| weight_loss_benefits.improvements | 3 | 0.00% |
| assistance | 3 | 0.00% |
| collaboration | 3 | 0.00% |
| entity group | 3 | 0.00% |
| product or material | 3 | 0.00% |
| asteroid.discovery.discoverer | 3 | 0.00% |
| Museum/Art Gallery | 3 | 0.00% |
| Fashion Accessory | 3 | 0.00% |
| Transportation equipment | 3 | 0.00% |
| Probability | 3 | 0.00% |
| class | 3 | 0.00% |
| Menu item | 3 | 0.00% |
| proficiency level | 3 | 0.00% |
| brand identity | 3 | 0.00% |
| drinkware | 3 | 0.00% |
| Mental state | 3 | 0.00% |
| financial management | 3 | 0.00% |
| Online Username | 3 | 0.00% |
| game feature | 3 | 0.00% |
| Restaurant Name | 3 | 0.00% |
| produce | 3 | 0.00% |
| Currency Limit | 3 | 0.00% |
| Question | 3 | 0.00% |
| Employed Institution | 3 | 0.00% |
| Publication Year | 3 | 0.00% |
| Data Structure | 3 | 0.00% |
| romantic relationship | 3 | 0.00% |
| Island | 3 | 0.00% |
| averageRating | 3 | 0.00% |
| Gift item | 3 | 0.00% |
| visual aid | 3 | 0.00% |
| Objectives | 3 | 0.00% |
| industry/type of business | 3 | 0.00% |
| TV drama serial | 3 | 0.00% |
| methodology.population.inclusion_criteria | 3 | 0.00% |
| Literary work | 3 | 0.00% |
| Personal Preference | 3 | 0.00% |
| teaching approach | 3 | 0.00% |
| Certificate | 3 | 0.00% |
| consequence | 3 | 0.00% |
| Healthcare company | 3 | 0.00% |
| Insects | 3 | 0.00% |
| Age of the Resignated Person | 3 | 0.00% |
| Event/Conference | 3 | 0.00% |
| cooking_methods | 3 | 0.00% |
| Recognition | 3 | 0.00% |
| Religious Event | 3 | 0.00% |
| Key | 3 | 0.00% |
| Well-being | 3 | 0.00% |
| Borough | 3 | 0.00% |
| Sports Facilities | 3 | 0.00% |
| location.country | 3 | 0.00% |
| Game character | 3 | 0.00% |
| user type | 3 | 0.00% |
| computer program | 3 | 0.00% |
| event/conference | 3 | 0.00% |
| biography.concert_tours | 3 | 0.00% |
| Fuel Product | 3 | 0.00% |
| Housing Development | 3 | 0.00% |
| office equipment | 3 | 0.00% |
| Clothing Accessory | 3 | 0.00% |
| municipality.name | 3 | 0.00% |
| Spacecraft Mission | 3 | 0.00% |
| Launch Site | 3 | 0.00% |
| form | 3 | 0.00% |
| connection | 3 | 0.00% |
| pressure | 3 | 0.00% |
| request | 3 | 0.00% |
| negative behavior | 3 | 0.00% |
| Death Rate | 3 | 0.00% |
| skincare ingredient | 3 | 0.00% |
| Environmental issue | 3 | 0.00% |
| Occupation/Profession | 3 | 0.00% |
| Musical group | 3 | 0.00% |
| health risk | 3 | 0.00% |
| Tree species | 3 | 0.00% |
| Online community | 3 | 0.00% |
| Quality standard | 3 | 0.00% |
| Person/Character | 3 | 0.00% |
| vehicle feature | 3 | 0.00% |
| exclamation | 3 | 0.00% |
| artistic creation | 3 | 0.00% |
| organizations | 3 | 0.00% |
| Military operation | 3 | 0.00% |
| Content type | 3 | 0.00% |
| Geological feature | 3 | 0.00% |
| director | 3 | 0.00% |
| Instruction | 3 | 0.00% |
| cognition | 3 | 0.00% |
| focus | 3 | 0.00% |
| Energy source | 3 | 0.00% |
| adverb | 3 | 0.00% |
| Measurement Tool | 3 | 0.00% |
| Time Point | 3 | 0.00% |
| conveyor type | 3 | 0.00% |
| educational activity | 3 | 0.00% |
| written content | 3 | 0.00% |
| Sports facility | 3 | 0.00% |
| Provisional Government | 3 | 0.00% |
| companies | 3 | 0.00% |
| Network | 3 | 0.00% |
| Housing type | 3 | 0.00% |
| Transportation Hub | 3 | 0.00% |
| Mathematical Operation | 3 | 0.00% |
| LED Product | 3 | 0.00% |
| social_media_post.hashtags | 3 | 0.00% |
| Mythical creature | 3 | 0.00% |
| Clothing part | 3 | 0.00% |
| Deception | 3 | 0.00% |
| accessory type | 3 | 0.00% |
| Family Relationship | 3 | 0.00% |
| breed of dog | 3 | 0.00% |
| Sports Term | 3 | 0.00% |
| absence | 3 | 0.00% |
| Family Relation | 3 | 0.00% |
| Person/Rapper | 3 | 0.00% |
| Capability | 3 | 0.00% |
| business strategy | 3 | 0.00% |
| negative action | 3 | 0.00% |
| Cleaning product | 3 | 0.00% |
| natural environment | 3 | 0.00% |
| jurisdiction | 3 | 0.00% |
| skin concern | 3 | 0.00% |
| Natural Object | 3 | 0.00% |
| Email Client | 3 | 0.00% |
| Photography | 3 | 0.00% |
| invoice_date | 3 | 0.00% |
| npadvmod | 3 | 0.00% |
| aux | 3 | 0.00% |
| aspect/type | 3 | 0.00% |
| Furniture Part | 3 | 0.00% |
| design style | 3 | 0.00% |
| Tennis term | 3 | 0.00% |
| holiday/event | 3 | 0.00% |
| approvalStatus | 3 | 0.00% |
| damage type | 3 | 0.00% |
| splicing_regulation.pathological_outcomes | 3 | 0.00% |
| safety | 3 | 0.00% |
| HTML tag | 3 | 0.00% |
| financial support | 3 | 0.00% |
| High School | 3 | 0.00% |
| Clothing Material | 3 | 0.00% |
| Car Engine | 3 | 0.00% |
| Financial Product | 3 | 0.00% |
| presentation | 3 | 0.00% |
| Consumer group | 3 | 0.00% |
| intellectual_compatibility.shared_traits | 3 | 0.00% |
| timepiece | 3 | 0.00% |
| instructional document | 3 | 0.00% |
| Occupation/Position | 3 | 0.00% |
| Service/Activity | 3 | 0.00% |
| Animal Type | 3 | 0.00% |
| nature | 3 | 0.00% |
| beer attribute | 3 | 0.00% |
| Plant Species | 3 | 0.00% |
| Online store | 3 | 0.00% |
| Diet Plan | 3 | 0.00% |
| Division | 3 | 0.00% |
| Location/Place | 3 | 0.00% |
| drink | 3 | 0.00% |
| responsibility | 3 | 0.00% |
| interior design | 3 | 0.00% |
| animals | 3 | 0.00% |
| social unit | 3 | 0.00% |
| travel | 3 | 0.00% |
| healthcare service | 3 | 0.00% |
| menu item | 3 | 0.00% |
| cocktail | 3 | 0.00% |
| Efficiency | 3 | 0.00% |
| Error | 3 | 0.00% |
| Anniversary | 3 | 0.00% |
| Model | 3 | 0.00% |
| metaphorical entity | 3 | 0.00% |
| social entity | 3 | 0.00% |
| Positive sentiment | 3 | 0.00% |
| job requirement | 3 | 0.00% |
| physical store | 3 | 0.00% |
| Video Quality | 3 | 0.00% |
| clothing/accessory | 3 | 0.00% |
| Criminal Activity | 3 | 0.00% |
| legal procedure | 3 | 0.00% |
| Room Type | 3 | 0.00% |
| work ethic | 3 | 0.00% |
| Improvement | 3 | 0.00% |
| interior design element | 3 | 0.00% |
| plant anatomy | 3 | 0.00% |
| Life event | 3 | 0.00% |
| Torque | 3 | 0.00% |
| Decision-making | 3 | 0.00% |
| Food Ingredient | 3 | 0.00% |
| time/date | 3 | 0.00% |
| Information sources | 3 | 0.00% |
| Sporting Event | 3 | 0.00% |
| Mobile app | 3 | 0.00% |
| personal name | 3 | 0.00% |
| digital object | 3 | 0.00% |
| university | 3 | 0.00% |
| Service Feature | 3 | 0.00% |
| Result | 3 | 0.00% |
| Political affiliation | 3 | 0.00% |
| order | 3 | 0.00% |
| organization/company | 3 | 0.00% |
| Security Feature | 3 | 0.00% |
| Education & Training | 3 | 0.00% |
| Financial Indicator | 3 | 0.00% |
| Music band | 3 | 0.00% |
| asteroid_name | 3 | 0.00% |
| Bacteria | 3 | 0.00% |
| Education Institution | 3 | 0.00% |
| dining places | 3 | 0.00% |
| person.name | 3 | 0.00% |
| Health Insurance | 3 | 0.00% |
| animal toy | 3 | 0.00% |
| Motorcycle model | 3 | 0.00% |
| Product/Artwork | 3 | 0.00% |
| project type | 3 | 0.00% |
| Taxation | 3 | 0.00% |
| industrial work center | 3 | 0.00% |
| Verb phrase | 3 | 0.00% |
| Financial Support | 3 | 0.00% |
| bacteria | 3 | 0.00% |
| offer | 3 | 0.00% |
| Manner | 3 | 0.00% |
| Sports Score | 3 | 0.00% |
| Digital Currency | 3 | 0.00% |
| issue_type | 3 | 0.00% |
| clothing material | 3 | 0.00% |
| entertainment service | 3 | 0.00% |
| organizational structure | 3 | 0.00% |
| Keyboard model | 3 | 0.00% |
| clothing style | 3 | 0.00% |
| distribution.regions.1.countries | 3 | 0.00% |
| distribution.regions.0.countries | 3 | 0.00% |
| Business Function | 3 | 0.00% |
| Mood/Emotion | 3 | 0.00% |
| Dimension | 3 | 0.00% |
| orbital_characteristics.inclination.unit | 3 | 0.00% |
| discovery.observatory.location.state | 3 | 0.00% |
| discovery.observatory.location.county | 3 | 0.00% |
| customers | 3 | 0.00% |
| Desktop Customization | 3 | 0.00% |
| Health condition | 3 | 0.00% |
| Creature | 3 | 0.00% |
| phone_number | 3 | 0.00% |
| informational material | 3 | 0.00% |
| waste management | 3 | 0.00% |
| Political entity | 3 | 0.00% |
| reproductive entity | 3 | 0.00% |
| Smartphone model | 3 | 0.00% |
| biology | 3 | 0.00% |
| Communication Platform | 3 | 0.00% |
| Action/State | 3 | 0.00% |
| Forum | 3 | 0.00% |
| property_address | 3 | 0.00% |
| Work of Art | 3 | 0.00% |
| The Sentence Claimed by the Prosecutor's Lawyer | 3 | 0.00% |
| Mental State | 3 | 0.00% |
| hotel.nearby_attractions | 3 | 0.00% |
| safety-requirements | 3 | 0.00% |
| estimated-time | 3 | 0.00% |
| customer request | 3 | 0.00% |
| Immigration Avenue | 3 | 0.00% |
| Communication protocol | 3 | 0.00% |
| Radio Station | 3 | 0.00% |
| Fuel type | 3 | 0.00% |
| land management | 3 | 0.00% |
| agriculture | 3 | 0.00% |
| soil quality | 3 | 0.00% |
| Department | 3 | 0.00% |
| government program | 3 | 0.00% |
| goods | 3 | 0.00% |
| autobiographical writing | 3 | 0.00% |
| academic evaluation | 3 | 0.00% |
| academic course | 3 | 0.00% |
| academic achievement | 3 | 0.00% |
| mixture | 3 | 0.00% |
| color code | 3 | 0.00% |
| color value | 3 | 0.00% |
| DJ/Artist | 3 | 0.00% |
| TV show/movie | 3 | 0.00% |
| sensitive information | 3 | 0.00% |
| medical test | 3 | 0.00% |
| error | 3 | 0.00% |
| Product/service | 3 | 0.00% |
| Educational Level | 3 | 0.00% |
| Religious Group | 3 | 0.00% |
| Term of Endearment | 3 | 0.00% |
| Law/Regulation | 3 | 0.00% |
| security measure | 3 | 0.00% |
| Physical Feature | 3 | 0.00% |
| recreational area | 3 | 0.00% |
| Stock | 3 | 0.00% |
| roomType | 3 | 0.00% |
| specialRequests | 3 | 0.00% |
| shoe component | 3 | 0.00% |
| e-commerce platform | 3 | 0.00% |
| Coupon Code | 3 | 0.00% |
| government entity | 3 | 0.00% |
| Messaging Channel | 3 | 0.00% |
| hospitality | 3 | 0.00% |
| Academic Program | 3 | 0.00% |
| Movement | 3 | 0.00% |
| family relation | 3 | 0.00% |
| Non-playable characters | 3 | 0.00% |
| function/operation | 3 | 0.00% |
| Company name | 3 | 0.00% |
| Standard | 3 | 0.00% |
| credits.content_implementers | 3 | 0.00% |
| Vehicle Part | 3 | 0.00% |
| workflow state | 3 | 0.00% |
| Process/Activity | 3 | 0.00% |
| on_site_registration.requirements | 3 | 0.00% |
| priority_registration.documents_required | 3 | 0.00% |
| Video game genre | 3 | 0.00% |
| Game feature | 3 | 0.00% |
| Unknown entity | 3 | 0.00% |
| jewelry item | 3 | 0.00% |
| car part/problem | 3 | 0.00% |
| Change | 3 | 0.00% |
| privacy | 3 | 0.00% |
| Theatrical performance | 3 | 0.00% |
| decision-making process | 3 | 0.00% |
| comparison | 3 | 0.00% |
| Ticket price | 3 | 0.00% |
| totalCost | 3 | 0.00% |
| mental health condition | 3 | 0.00% |
| Education program | 3 | 0.00% |
| Fabric type | 3 | 0.00% |
| user group | 3 | 0.00% |
| digital entity | 3 | 0.00% |
| software repository | 3 | 0.00% |
| software process | 3 | 0.00% |
| limit | 3 | 0.00% |
| window | 3 | 0.00% |
| pattern | 3 | 0.00% |
| action entity | 3 | 0.00% |
| alteration | 3 | 0.00% |
| Person/Profession | 3 | 0.00% |
| personal development | 3 | 0.00% |
| addiction/problematic behavior | 3 | 0.00% |
| Government entity | 3 | 0.00% |
| Location entity | 3 | 0.00% |
| Medical Service | 3 | 0.00% |
| Product/Device | 3 | 0.00% |
| equipmentId | 3 | 0.00% |
| tool/service | 3 | 0.00% |
| success | 3 | 0.00% |
| failure | 3 | 0.00% |
| Bible Verse | 3 | 0.00% |
| historical civilization | 3 | 0.00% |
| Program/Project | 3 | 0.00% |
| interface elements | 3 | 0.00% |
| Event/Experience | 3 | 0.00% |
| Supplier | 3 | 0.00% |
| Fence type/property | 3 | 0.00% |
| Gift Idea | 3 | 0.00% |
| Email Address | 3 | 0.00% |
| stage of buying process | 3 | 0.00% |
| statistical_results.predictive_model.variables_included | 3 | 0.00% |
| Stock Exchange | 3 | 0.00% |
| Lawyer | 3 | 0.00% |
| Climate | 3 | 0.00% |
| Transportation Route | 3 | 0.00% |
| firefighting equipment | 3 | 0.00% |
| User Action | 3 | 0.00% |
| familial relationship | 3 | 0.00% |
| Visual Media | 3 | 0.00% |
| Activity/Sport | 3 | 0.00% |
| instruction | 3 | 0.00% |
| Advertising platform | 3 | 0.00% |
| Lighting Fixture | 3 | 0.00% |
| hobby/activity | 3 | 0.00% |
| departments | 3 | 0.00% |
| font style | 3 | 0.00% |
| time entity | 3 | 0.00% |
| Law enforcement agency | 3 | 0.00% |
| Demographic entity | 3 | 0.00% |
| Legal Issue | 3 | 0.00% |
| Music Artist/Band | 3 | 0.00% |
| network device | 3 | 0.00% |
| Practice | 3 | 0.00% |
| landscaping material | 3 | 0.00% |
| Warranty | 3 | 0.00% |
| packaging material | 3 | 0.00% |
| film | 3 | 0.00% |
| Episode Title | 3 | 0.00% |
| Marketing Campaign | 3 | 0.00% |
| Security Measure | 3 | 0.00% |
| legal matter | 3 | 0.00% |
| verb, past tense | 3 | 0.00% |
| Facility Type | 3 | 0.00% |
| Computer hardware | 3 | 0.00% |
| Healthcare facility | 3 | 0.00% |
| reviewDate | 3 | 0.00% |
| Award Ceremony | 3 | 0.00% |
| health product | 3 | 0.00% |
| weather element | 3 | 0.00% |
| food category | 3 | 0.00% |
| Wind | 3 | 0.00% |
| Progression | 3 | 0.00% |
| Position/Title | 3 | 0.00% |
| musical interval | 3 | 0.00% |
| Legal action | 3 | 0.00% |
| Wine classification | 3 | 0.00% |
| Company/Brand Name | 3 | 0.00% |
| Feedback | 3 | 0.00% |
| Political Leader | 3 | 0.00% |
| decision | 3 | 0.00% |
| Artistic Creation | 3 | 0.00% |
| carrier | 3 | 0.00% |
| Ownership | 3 | 0.00% |
| Distribution | 3 | 0.00% |
| topics | 3 | 0.00% |
| Market Segment | 3 | 0.00% |
| Display Technology | 3 | 0.00% |
| Ski Resort | 3 | 0.00% |
| recreational areas | 3 | 0.00% |
| Wellness | 3 | 0.00% |
| Food and Beverage | 3 | 0.00% |
| cuisine type | 3 | 0.00% |
| Accommodation type | 3 | 0.00% |
| religious document | 3 | 0.00% |
| towing equipment | 3 | 0.00% |
| Technique | 3 | 0.00% |
| arrivalDate | 3 | 0.00% |
| Game Item | 3 | 0.00% |
| Uncertainty | 3 | 0.00% |
| makeup product | 3 | 0.00% |
| Hotel name | 3 | 0.00% |
| Mobile Devices | 3 | 0.00% |
| discomfort | 3 | 0.00% |
| Bath Feature | 3 | 0.00% |
| surface treatment | 3 | 0.00% |
| subscription_id | 3 | 0.00% |
| confirmation_code | 3 | 0.00% |
| Software Platform | 3 | 0.00% |
| License | 3 | 0.00% |
| aesthetic quality | 3 | 0.00% |
| properties | 3 | 0.00% |
| Company/organization | 3 | 0.00% |
| travel destination | 3 | 0.00% |
| Physical space | 3 | 0.00% |
| Expense | 3 | 0.00% |
| Head of Investigation Team | 3 | 0.00% |
| advertising material | 3 | 0.00% |
| Web Browser | 3 | 0.00% |
| Law enforcement unit | 3 | 0.00% |
| Recreation | 3 | 0.00% |
| federalism_definition.federalism_examples | 3 | 0.00% |
| possibility | 3 | 0.00% |
| file | 3 | 0.00% |
| concern | 3 | 0.00% |
| area | 3 | 0.00% |
| Launch Result | 3 | 0.00% |
| Taste | 3 | 0.00% |
| politician | 3 | 0.00% |
| Messaging Service | 3 | 0.00% |
| Company/Website | 3 | 0.00% |
| Attitude | 3 | 0.00% |
| Residence | 3 | 0.00% |
| Real estate property feature | 3 | 0.00% |
| student housing | 3 | 0.00% |
| technology entity | 3 | 0.00% |
| process step | 3 | 0.00% |
| written material | 3 | 0.00% |
| Data storage | 3 | 0.00% |
| Object/Thing | 3 | 0.00% |
| gratitude | 3 | 0.00% |
| authority | 3 | 0.00% |
| bathroom tile usage | 3 | 0.00% |
| Artistic Technique | 3 | 0.00% |
| car feature | 3 | 0.00% |
| airDate | 3 | 0.00% |
| currency/money | 3 | 0.00% |
| physical well-being | 3 | 0.00% |
| Academic Year | 3 | 0.00% |
| Group of Islands | 3 | 0.00% |
| packaging | 3 | 0.00% |
| Animal Material | 3 | 0.00% |
| Product Quantity | 3 | 0.00% |
| Tourist Destination | 3 | 0.00% |
| national symbol | 3 | 0.00% |
| Book Series | 3 | 0.00% |
| Manufacturer | 3 | 0.00% |
| transportation service | 3 | 0.00% |
| Social Gathering | 3 | 0.00% |
| Website/Company | 3 | 0.00% |
| Training Course | 3 | 0.00% |
| TV character | 3 | 0.00% |
| Sports Category | 3 | 0.00% |
| biological sample | 3 | 0.00% |
| job benefits | 3 | 0.00% |
| Website Section | 3 | 0.00% |
| occupation/activity | 3 | 0.00% |
| political figure | 3 | 0.00% |
| physical structure | 3 | 0.00% |
| colour | 3 | 0.00% |
| Machine Part | 3 | 0.00% |
| Transportation Entity | 3 | 0.00% |
| territory | 3 | 0.00% |
| product brand | 3 | 0.00% |
| School Name | 3 | 0.00% |
| Socioeconomic Status | 3 | 0.00% |
| cooking process | 3 | 0.00% |
| location/infrastructure | 3 | 0.00% |
| negative emotion | 3 | 0.00% |
| customer satisfaction | 3 | 0.00% |
| customer_name | 3 | 0.00% |
| Monetary Value Entity | 3 | 0.00% |
| Government program | 3 | 0.00% |
| Landscaping | 3 | 0.00% |
| customer base | 3 | 0.00% |
| Gaming Software Developer | 3 | 0.00% |
| Gender Group | 3 | 0.00% |
| publicationDate | 3 | 0.00% |
| Natural features | 3 | 0.00% |
| Age Range | 3 | 0.00% |
| product.browser_compatibility | 3 | 0.00% |
| Event venue | 3 | 0.00% |
| Character Name | 3 | 0.00% |
| human entity | 3 | 0.00% |
| artifact | 3 | 0.00% |
| capability | 3 | 0.00% |
| cost/price | 3 | 0.00% |
| Population Group | 3 | 0.00% |
| field of work | 3 | 0.00% |
| personal identity | 3 | 0.00% |
| Event Schedule | 3 | 0.00% |
| Incentive | 3 | 0.00% |
| stressor | 3 | 0.00% |
| animal type | 3 | 0.00% |
| job type | 3 | 0.00% |
| Sport statistics | 3 | 0.00% |
| issue/concern | 3 | 0.00% |
| Software Type | 3 | 0.00% |
| Health/Wellness | 3 | 0.00% |
| Dating app | 3 | 0.00% |
| Material/Property | 3 | 0.00% |
| Business process | 3 | 0.00% |
| Symbolic item | 3 | 0.00% |
| organization/entity | 3 | 0.00% |
| organic material | 3 | 0.00% |
| Book Subject | 3 | 0.00% |
| person/individual | 3 | 0.00% |
| Engine Type | 3 | 0.00% |
| Subject | 3 | 0.00% |
| asteroid.named_after.person.occupation | 3 | 0.00% |
| plan | 3 | 0.00% |
| Job Opportunity | 3 | 0.00% |
| Television Programme | 3 | 0.00% |
| employees | 3 | 0.00% |
| Interior Design Element | 3 | 0.00% |
| Medical test | 3 | 0.00% |
| objective | 3 | 0.00% |
| player_name | 3 | 0.00% |
| points | 3 | 0.00% |
| Gaming | 3 | 0.00% |
| Legal issue | 3 | 0.00% |
| Role/Title | 3 | 0.00% |
| Show | 3 | 0.00% |
| Quantity/Measurement | 3 | 0.00% |
| Visual Element | 3 | 0.00% |
| Life | 3 | 0.00% |
| Prosecution Lawyer | 3 | 0.00% |
| Literary genre | 3 | 0.00% |
| salary_range | 3 | 0.00% |
| Life stage | 3 | 0.00% |
| precious metal | 3 | 0.00% |
| Business Sector | 3 | 0.00% |
| TestName | 3 | 0.00% |
| TestResult | 3 | 0.00% |
| Social activity | 3 | 0.00% |
| Texture/Quality | 3 | 0.00% |
| vehicle component | 3 | 0.00% |
| consumable | 3 | 0.00% |
| research findings | 3 | 0.00% |
| market segment | 3 | 0.00% |
| Corporation/Company | 3 | 0.00% |
| Jazz Competition | 3 | 0.00% |
| Behavior/Trait | 3 | 0.00% |
| Property Amenities | 3 | 0.00% |
| section | 3 | 0.00% |
| word_count | 3 | 0.00% |
| Sports Statistic | 3 | 0.00% |
| Health Indicator | 3 | 0.00% |
| battery_life | 3 | 0.00% |
| Software/Program | 3 | 0.00% |
| Game Feature | 3 | 0.00% |
| Sports Team Strength | 3 | 0.00% |
| Treatment program | 3 | 0.00% |
| Animal group | 3 | 0.00% |
| self | 3 | 0.00% |
| Application Name | 3 | 0.00% |
| musical instrument group | 3 | 0.00% |
| education-related entity | 3 | 0.00% |
| Eyewear | 3 | 0.00% |
| skill/expertise | 3 | 0.00% |
| Medical Technology | 3 | 0.00% |
| Payment | 3 | 0.00% |
| academic program | 3 | 0.00% |
| deductible | 3 | 0.00% |
| In-game currency | 3 | 0.00% |
| Food and Drink | 3 | 0.00% |
| Location/Room | 3 | 0.00% |
| political affiliation | 3 | 0.00% |
| Speed | 3 | 0.00% |
| authors | 3 | 0.00% |
| Motorcycle Brand | 3 | 0.00% |
| hyperlink | 3 | 0.00% |
| hotelId | 3 | 0.00% |
| Conflict | 3 | 0.00% |
| probability | 3 | 0.00% |
| perception | 3 | 0.00% |
| Mountain Range | 3 | 0.00% |
| medical service | 3 | 0.00% |
| Appreciation | 3 | 0.00% |
| Occupation Entity | 3 | 0.00% |
| atmosphere | 3 | 0.00% |
| Medical Information | 3 | 0.00% |
| Technology company | 3 | 0.00% |
| body of water | 3 | 0.00% |
| outdoor furniture | 3 | 0.00% |
| News Release Agency | 3 | 0.00% |
| meat | 3 | 0.00% |
| communication content | 3 | 0.00% |
| Music element | 3 | 0.00% |
| machineId | 3 | 0.00% |
| Cryptocurrency token | 3 | 0.00% |
| sports surface | 3 | 0.00% |
| education facility | 3 | 0.00% |
| political position | 3 | 0.00% |
| type of plant | 3 | 0.00% |
| clothing item feature | 3 | 0.00% |
| Choice | 3 | 0.00% |
| Person/Entertainer | 3 | 0.00% |
| goods and services | 3 | 0.00% |
| Coach | 3 | 0.00% |
| symptoms: List of reported symptoms like fever or headache | 3 | 0.00% |
| testId | 3 | 0.00% |
| rate | 3 | 0.00% |
| Academic Department | 3 | 0.00% |
| Utility | 3 | 0.00% |
| cooking equipment | 3 | 0.00% |
| selection process | 3 | 0.00% |
| Clothing Store | 3 | 0.00% |
| living organism | 3 | 0.00% |
| machine part | 3 | 0.00% |
| Dinosaur Species | 3 | 0.00% |
| consumer goods | 3 | 0.00% |
| Art Exhibition | 3 | 0.00% |
| School District | 3 | 0.00% |
| beauty service | 3 | 0.00% |
| Window | 3 | 0.00% |
| Crystal | 3 | 0.00% |
| Baseball player | 3 | 0.00% |
| financial activity | 3 | 0.00% |
| options strategy | 3 | 0.00% |
| environmental stressor | 3 | 0.00% |
| Head of the Merged Organization | 3 | 0.00% |
| payment type | 3 | 0.00% |
| structure type | 3 | 0.00% |
| research lab | 3 | 0.00% |
| vaping equipment | 3 | 0.00% |
| philosophy | 3 | 0.00% |
| promotional offer | 3 | 0.00% |
| Safety Rating | 3 | 0.00% |
| Sector | 3 | 0.00% |
| competition category | 3 | 0.00% |
| Environmental Hazard | 3 | 0.00% |
| Career | 3 | 0.00% |
| Pest | 3 | 0.00% |
| Area | 3 | 0.00% |
| artists | 3 | 0.00% |
| mechanical device | 3 | 0.00% |
| Skin condition | 3 | 0.00% |
| celebration | 3 | 0.00% |
| biblical reference | 3 | 0.00% |
| trait/characteristic | 3 | 0.00% |
| supplier | 3 | 0.00% |
| Visual Effect | 3 | 0.00% |
| Website element | 3 | 0.00% |
| Decorative Item | 3 | 0.00% |
| Geographical Region | 3 | 0.00% |
| Weather event | 3 | 0.00% |
| workforce | 3 | 0.00% |
| Data type | 3 | 0.00% |
| mechanism | 3 | 0.00% |
| Consumer Goods | 3 | 0.00% |
| reviewer_name | 3 | 0.00% |
| customization technique | 3 | 0.00% |
| video game title | 3 | 0.00% |
| television series | 3 | 0.00% |
| book | 3 | 0.00% |
| Linguistics/ Grammar | 3 | 0.00% |
| part/component | 3 | 0.00% |
| academic status | 3 | 0.00% |
| Appearance | 3 | 0.00% |
| Educational program | 3 | 0.00% |
| Stock Market Technical Analysis | 3 | 0.00% |
| object/item | 3 | 0.00% |
| Personal goal | 3 | 0.00% |
| Meat Ingredient | 3 | 0.00% |
| Engine Part | 3 | 0.00% |
| Airline Company | 3 | 0.00% |
| Campsite | 3 | 0.00% |
| Music album | 3 | 0.00% |
| building component | 3 | 0.00% |
| Social Event | 3 | 0.00% |
| file type | 3 | 0.00% |
| topic/category | 3 | 0.00% |
| song | 3 | 0.00% |
| ZipCode | 3 | 0.00% |
| Email | 3 | 0.00% |
| Academic Result | 3 | 0.00% |
| Technology/Industry | 3 | 0.00% |
| asteroid.discovery.discoverer.nationality | 3 | 0.00% |
| shopping location | 3 | 0.00% |
| expression of gratitude | 3 | 0.00% |
| healing technique | 3 | 0.00% |
| Field of expertise | 3 | 0.00% |
| football team | 3 | 0.00% |
| flower | 3 | 0.00% |
| Band/Artist | 3 | 0.00% |
| Vehicles | 3 | 0.00% |
| train | 3 | 0.00% |
| artwork element | 3 | 0.00% |
| storeId | 3 | 0.00% |
| Loan | 3 | 0.00% |
| Organization Type | 3 | 0.00% |
| type of property | 3 | 0.00% |
| career path | 3 | 0.00% |
| TV series | 3 | 0.00% |
| Actress | 3 | 0.00% |
| communal space | 3 | 0.00% |
| Geographic Area | 3 | 0.00% |
| residential property | 3 | 0.00% |
| Debt | 3 | 0.00% |
| Warning | 3 | 0.00% |
| Feature/Attribute | 3 | 0.00% |
| Computer Architecture | 3 | 0.00% |
| social movement | 3 | 0.00% |
| Product Sub-Category | 3 | 0.00% |
| Feature/Specification | 3 | 0.00% |
| Promotion/Deal | 3 | 0.00% |
| mission type | 3 | 0.00% |
| negative experience | 3 | 0.00% |
| Fashion | 3 | 0.00% |
| enzyme | 3 | 0.00% |
| artistic work | 3 | 0.00% |
| legal entity | 3 | 0.00% |
| type of rack | 3 | 0.00% |
| academic leader | 3 | 0.00% |
| Environment Attribute | 3 | 0.00% |
| WWE wrestler | 3 | 0.00% |
| team name | 3 | 0.00% |
| wildlife | 3 | 0.00% |
| Gaming Platform | 3 | 0.00% |
| medicalConditions | 3 | 0.00% |
| trait/personality | 3 | 0.00% |
| Online retailer | 3 | 0.00% |
| task/activity | 3 | 0.00% |
| retail location | 3 | 0.00% |
| teams | 3 | 0.00% |
| Medical Entity | 3 | 0.00% |
| product specification | 3 | 0.00% |
| Fly fishing lure | 3 | 0.00% |
| Historical figures | 3 | 0.00% |
| Food items | 3 | 0.00% |
| Membership | 3 | 0.00% |
| time/event | 3 | 0.00% |
| physical ability | 3 | 0.00% |
| Craft | 3 | 0.00% |
| Time frame | 3 | 0.00% |
| Publication Date | 3 | 0.00% |
| carpet accessory | 3 | 0.00% |
| rug accessory | 3 | 0.00% |
| platform feature | 3 | 0.00% |
| Identity | 3 | 0.00% |
| advertisement | 3 | 0.00% |
| Personal identity | 3 | 0.00% |
| musical attribute | 3 | 0.00% |
| bike feature | 3 | 0.00% |
| addressLine1 | 3 | 0.00% |
| postalCode | 3 | 0.00% |
| task/job management | 3 | 0.00% |
| family relationship | 3 | 0.00% |
| restaurant.features | 3 | 0.00% |
| movie aspect | 3 | 0.00% |
| Norse mythology deity | 3 | 0.00% |
| knowledge | 3 | 0.00% |
| Environment/Landscape | 3 | 0.00% |
| Port | 3 | 0.00% |
| Job | 3 | 0.00% |
| agricultural product | 3 | 0.00% |
| dating site | 3 | 0.00% |
| Yarn Feature | 3 | 0.00% |
| Furniture Type | 3 | 0.00% |
| Test | 3 | 0.00% |
| Style/Fashion | 3 | 0.00% |
| Language dialects | 3 | 0.00% |
| Organization/Group | 3 | 0.00% |
| article.references.0.offenses | 3 | 0.00% |
| cloud storage | 3 | 0.00% |
| Competition Level | 3 | 0.00% |
| Subcategory | 3 | 0.00% |
| characteristics | 3 | 0.00% |
| Computing resources | 3 | 0.00% |
| Software product | 3 | 0.00% |
| Tablet | 3 | 0.00% |
| illness | 3 | 0.00% |
| Product/Model | 3 | 0.00% |
| Real Estate Agency | 3 | 0.00% |
| Health Insurance Plan | 3 | 0.00% |
| medical procedures | 3 | 0.00% |
| measuredValue | 3 | 0.00% |
| type of written work | 3 | 0.00% |
| Tire size | 3 | 0.00% |
| Rugby strategy | 3 | 0.00% |
| Product Version | 3 | 0.00% |
| residential building | 3 | 0.00% |
| advantage | 3 | 0.00% |
| Religious leader | 3 | 0.00% |
| chemical substance | 3 | 0.00% |
| establishment type | 3 | 0.00% |
| Policy Proposals | 3 | 0.00% |
| medical product | 3 | 0.00% |
| combination | 3 | 0.00% |
| Telecommunications company | 3 | 0.00% |
| Stock Symbol | 3 | 0.00% |
| numeric value | 3 | 0.00% |
| Infrastructure/Asset | 3 | 0.00% |
| Skin type | 3 | 0.00% |
| waste type/container | 3 | 0.00% |
| household infrastructure | 3 | 0.00% |
| sanitary appliance | 3 | 0.00% |
| Ingredient/food | 3 | 0.00% |
| task process | 3 | 0.00% |
| Processor model | 3 | 0.00% |
| Technology System | 3 | 0.00% |
| Process Entity | 3 | 0.00% |
| warning | 3 | 0.00% |
| Professional Association | 3 | 0.00% |
| body | 3 | 0.00% |
| Communication Entity | 3 | 0.00% |
| Time/Planning | 3 | 0.00% |
| Woodworking entity | 3 | 0.00% |
| measurement technique | 3 | 0.00% |
| Financial Account | 3 | 0.00% |
| Vehicle Option | 3 | 0.00% |
| term | 3 | 0.00% |
| Technology/Instrument | 3 | 0.00% |
| handmade items | 3 | 0.00% |
| Healthcare professionals | 3 | 0.00% |
| Dietary restriction | 3 | 0.00% |
| gift | 3 | 0.00% |
| microbiological entity | 3 | 0.00% |
| Soccer player | 3 | 0.00% |
| Game Mechanic | 3 | 0.00% |
| transportation system | 3 | 0.00% |
| Person Type | 3 | 0.00% |
| camera model | 3 | 0.00% |
| Cooking method | 3 | 0.00% |
| Sweepstakes | 3 | 0.00% |
| hotel_experience.additional_benefits | 3 | 0.00% |
| fabric/material | 3 | 0.00% |
| celebrity | 3 | 0.00% |
| Person/Celebrity | 3 | 0.00% |
| Planet | 3 | 0.00% |
| Legal decision | 3 | 0.00% |
| Challenge/Obstacle | 3 | 0.00% |
| Mountain Peak | 3 | 0.00% |
| Electrical Component | 3 | 0.00% |
| qualityChecks | 3 | 0.00% |
| Ski Trail | 3 | 0.00% |
| Makeup product | 3 | 0.00% |
| deposit | 3 | 0.00% |
| vaccines: List of vaccine types administered to the patient | 3 | 0.00% |
| Outdoor Furniture | 3 | 0.00% |
| TV Show Episode | 3 | 0.00% |
| Person/Actor | 3 | 0.00% |
| child care provider | 3 | 0.00% |
| product size | 3 | 0.00% |
| nsubj | 3 | 0.00% |
| class_name | 3 | 0.00% |
| method_name | 3 | 0.00% |
| engineering | 3 | 0.00% |
| Building/Location | 3 | 0.00% |
| dairy product | 3 | 0.00% |
| App/game | 3 | 0.00% |
| academic record | 3 | 0.00% |
| Product format | 3 | 0.00% |
| linguistic_framework.concluding_grammar_focus | 3 | 0.00% |
| Mobile Network Technology | 3 | 0.00% |
| Rendering Tool | 3 | 0.00% |
| hotel.location.nearby_attractions | 3 | 0.00% |
| Events | 3 | 0.00% |
| bed treatment | 3 | 0.00% |
| materialId | 3 | 0.00% |
| Position/Role | 3 | 0.00% |
| Sister | 3 | 0.00% |
| Real estate status | 3 | 0.00% |
| Dental Term | 3 | 0.00% |
| computing device vendor | 3 | 0.00% |
| Fish | 3 | 0.00% |
| IN | 3 | 0.00% |
| marketing activity | 3 | 0.00% |
| person/role | 3 | 0.00% |
| Snack Item | 3 | 0.00% |
| Aircraft part | 3 | 0.00% |
| film festival | 3 | 0.00% |
| Software system | 3 | 0.00% |
| Food/Menu Item | 3 | 0.00% |
| entities | 3 | 0.00% |
| Hardware Device | 3 | 0.00% |
| assessment criteria | 3 | 0.00% |
| business scenario | 3 | 0.00% |
| architecture feature | 3 | 0.00% |
| Association | 3 | 0.00% |
| Health | 3 | 0.00% |
| Multiplayer Mode | 3 | 0.00% |
| training_program.invitation_steps | 3 | 0.00% |
| Network Type | 3 | 0.00% |
| Educational tool | 3 | 0.00% |
| kitchen item | 3 | 0.00% |
| version number | 3 | 0.00% |
| additionalInfo | 3 | 0.00% |
| product property | 3 | 0.00% |
| Fitness program | 3 | 0.00% |
| device model | 3 | 0.00% |
| location/type | 3 | 0.00% |
| Exercise term | 3 | 0.00% |
| decorative material | 3 | 0.00% |
| agency.partners | 3 | 0.00% |
| sales metric | 3 | 0.00% |
| Dance style | 3 | 0.00% |
| Fitness goal | 3 | 0.00% |
| unit_price | 3 | 0.00% |
| Social Media Site | 3 | 0.00% |
| shoe attribute | 3 | 0.00% |
| Gaming console | 3 | 0.00% |
| Housing/Real Estate | 3 | 0.00% |
| Place/Event | 3 | 0.00% |
| American brand | 3 | 0.00% |
| Festival/Celebration | 3 | 0.00% |
| channels | 3 | 0.00% |
| Body posture | 3 | 0.00% |
| stationTypeId | 3 | 0.00% |
| outputMaterialId | 3 | 0.00% |
| Interior design style | 3 | 0.00% |
| aircraft type | 3 | 0.00% |
| Spa Facility | 3 | 0.00% |
| hazard | 3 | 0.00% |
| Study Tool | 3 | 0.00% |
| Stock Number | 3 | 0.00% |
| NFL Team | 3 | 0.00% |
| plant feature | 3 | 0.00% |
| camera accessory | 3 | 0.00% |
| SEO optimization | 3 | 0.00% |
| optional_information | 2 | 0.00% |
| government organization | 2 | 0.00% |
| additionalNotes | 2 | 0.00% |
| attachments | 2 | 0.00% |
| personal belongings | 2 | 0.00% |
| Stadium | 2 | 0.00% |
| report_date | 2 | 0.00% |
| icd10code | 2 | 0.00% |
| virus | 2 | 0.00% |
| occurrence | 2 | 0.00% |
| Real Estate Company | 2 | 0.00% |
| performance testing | 2 | 0.00% |
| Negative outcome | 2 | 0.00% |
| technology/feature | 2 | 0.00% |
| Human Resource | 2 | 0.00% |
| client_name | 2 | 0.00% |
| Organization/Institution | 2 | 0.00% |
| audience/consumer | 2 | 0.00% |
| goods/services | 2 | 0.00% |
| understanding | 2 | 0.00% |
| action_items | 2 | 0.00% |
| hazardous material | 2 | 0.00% |
| referee | 2 | 0.00% |
| match_id | 2 | 0.00% |
| assets | 2 | 0.00% |
| permission | 2 | 0.00% |
| Financial account | 2 | 0.00% |
| Mythical Creature | 2 | 0.00% |
| App version | 2 | 0.00% |
| International organization | 2 | 0.00% |
| Pet Grooming | 2 | 0.00% |
| typography | 2 | 0.00% |
| asteroid.location | 2 | 0.00% |
| product availability | 2 | 0.00% |
| Skill/Quality | 2 | 0.00% |
| social media hashtag | 2 | 0.00% |
| Sustainable material | 2 | 0.00% |
| Storage container | 2 | 0.00% |
| Business opportunity | 2 | 0.00% |
| Distance measurement | 2 | 0.00% |
| physical_characteristics.observations.0.telescope.name | 2 | 0.00% |
| physical_characteristics.observations.0.telescope.type | 2 | 0.00% |
| entertainment genre | 2 | 0.00% |
| fictional entity | 2 | 0.00% |
| annotation | 2 | 0.00% |
| licenseNumber | 2 | 0.00% |
| normal_range | 2 | 0.00% |
| events | 2 | 0.00% |
| availability | 2 | 0.00% |
| Hormone | 2 | 0.00% |
| cooking appliance | 2 | 0.00% |
| Goods | 2 | 0.00% |
| retailer | 2 | 0.00% |
| chemical_name | 2 | 0.00% |
| Audio equipment | 2 | 0.00% |
| process/method | 2 | 0.00% |
| Islands | 2 | 0.00% |
| Geographical region | 2 | 0.00% |
| self-improvement | 2 | 0.00% |
| web hosting service | 2 | 0.00% |
| Musical Composition | 2 | 0.00% |
| plot | 2 | 0.00% |
| narrative element | 2 | 0.00% |
| prediction | 2 | 0.00% |
| social science | 2 | 0.00% |
| Service Offer | 2 | 0.00% |
| market_share | 2 | 0.00% |
| research_context.problem_statement.incompatible_substrates | 2 | 0.00% |
| drivetrain | 2 | 0.00% |
| Clothing feature | 2 | 0.00% |
| product offering | 2 | 0.00% |
| Negative experience | 2 | 0.00% |
| Automobiles | 2 | 0.00% |
| user_name | 2 | 0.00% |
| Research Topic | 2 | 0.00% |
| Location/Venue | 2 | 0.00% |
| Community Name | 2 | 0.00% |
| expenses | 2 | 0.00% |
| Pattern | 2 | 0.00% |
| Familienmitglied | 2 | 0.00% |
| software libraries | 2 | 0.00% |
| Verb Phrase | 2 | 0.00% |
| Noun Phrase | 2 | 0.00% |
| location.region | 2 | 0.00% |
| price_range.currency | 2 | 0.00% |
| product group | 2 | 0.00% |
| biomolecule/lipid | 2 | 0.00% |
| News Source | 2 | 0.00% |
| guitar part | 2 | 0.00% |
| Movie Genre | 2 | 0.00% |
| biography.mentors | 2 | 0.00% |
| reliability | 2 | 0.00% |
| Cultural heritage | 2 | 0.00% |
| Residential area | 2 | 0.00% |
| Casino | 2 | 0.00% |
| Fabric Material | 2 | 0.00% |
| State of mind | 2 | 0.00% |
| Art/Culture | 2 | 0.00% |
| notification system | 2 | 0.00% |
| sense | 2 | 0.00% |
| Social Media Handle | 2 | 0.00% |
| material object | 2 | 0.00% |
| Mechanical Property | 2 | 0.00% |
| Art Supply | 2 | 0.00% |
| airline company | 2 | 0.00% |
| airport | 2 | 0.00% |
| Medical examination | 2 | 0.00% |
| person/name | 2 | 0.00% |
| border/location | 2 | 0.00% |
| game outcome | 2 | 0.00% |
| Environmental technology | 2 | 0.00% |
| Historical Event/Period | 2 | 0.00% |
| Empire | 2 | 0.00% |
| Economic System | 2 | 0.00% |
| color/texture | 2 | 0.00% |
| Game Component | 2 | 0.00% |
| outdoor item | 2 | 0.00% |
| maintenance process | 2 | 0.00% |
| sports field | 2 | 0.00% |
| type of club | 2 | 0.00% |
| Broadcasting | 2 | 0.00% |
| beauty practice | 2 | 0.00% |
| Feeling | 2 | 0.00% |
| contact | 2 | 0.00% |
| Printer model | 2 | 0.00% |
| Technology type | 2 | 0.00% |
| factor | 2 | 0.00% |
| policyholder_name | 2 | 0.00% |
| claim_date | 2 | 0.00% |
| Ordinal Number | 2 | 0.00% |
| supernatural being | 2 | 0.00% |
| Safety measure | 2 | 0.00% |
| Physical product | 2 | 0.00% |
| galaxy | 2 | 0.00% |
| pregnancy-related entity | 2 | 0.00% |
| household items | 2 | 0.00% |
| product order | 2 | 0.00% |
| Musician/Band member | 2 | 0.00% |
| cooking tool | 2 | 0.00% |
| liquid | 2 | 0.00% |
| Crop Variety | 2 | 0.00% |
| Data Storage | 2 | 0.00% |
| video_id: Unique identifier assigned to the video | 2 | 0.00% |
| person/politician | 2 | 0.00% |
| reflexive pronoun | 2 | 0.00% |
| infrastructure.highways.2.connections | 2 | 0.00% |
| Company Quality | 2 | 0.00% |
| Discount Code | 2 | 0.00% |
| Lake | 2 | 0.00% |
| claim_id: Unique identifier for each claim entry | 2 | 0.00% |
| Herb/plant | 2 | 0.00% |
| Intensity | 2 | 0.00% |
| home furnishing | 2 | 0.00% |
| Software application | 2 | 0.00% |
| Student | 2 | 0.00% |
| Grade level | 2 | 0.00% |
| troubleshooting | 2 | 0.00% |
| damaged property | 2 | 0.00% |
| Educational Entity | 2 | 0.00% |
| client | 2 | 0.00% |
| Sound characteristic | 2 | 0.00% |
| Architectural Feature | 2 | 0.00% |
| Event/Location | 2 | 0.00% |
| Person or animal | 2 | 0.00% |
| article | 2 | 0.00% |
| visual impairment | 2 | 0.00% |
| Fashion style | 2 | 0.00% |
| Clothing style | 2 | 0.00% |
| unit_location | 2 | 0.00% |
| Legislative Body | 2 | 0.00% |
| Judicial Body | 2 | 0.00% |
| Administrative Region | 2 | 0.00% |
| Generation | 2 | 0.00% |
| Ski run | 2 | 0.00% |
| Stock exchange | 2 | 0.00% |
| Stock symbol | 2 | 0.00% |
| tissue | 2 | 0.00% |
| government officials | 2 | 0.00% |
| Resource Type | 2 | 0.00% |
| Local Authority | 2 | 0.00% |
| personal entity | 2 | 0.00% |
| measurement/observation | 2 | 0.00% |
| service category | 2 | 0.00% |
| Activity/Service | 2 | 0.00% |
| physical_characteristics.observations.0.year | 2 | 0.00% |
| discovery.location.state | 2 | 0.00% |
| discovery.location.country | 2 | 0.00% |
| submission_id | 2 | 0.00% |
| clothing/fashion | 2 | 0.00% |
| product variety | 2 | 0.00% |
| Celestial body | 2 | 0.00% |
| Resource/Tool | 2 | 0.00% |
| Sports field | 2 | 0.00% |
| Sports equipment | 2 | 0.00% |
| Postal service | 2 | 0.00% |
| blood_pressure | 2 | 0.00% |
| heart_rate | 2 | 0.00% |
| sponsor | 2 | 0.00% |
| Brand/Model | 2 | 0.00% |
| ranking system | 2 | 0.00% |
| Organization/Record Label | 2 | 0.00% |
| taste/quality | 2 | 0.00% |
| follow_up | 2 | 0.00% |
| discharge_date | 2 | 0.00% |
| business operations | 2 | 0.00% |
| Material property | 2 | 0.00% |
| Casualities and Losses | 2 | 0.00% |
| food establishment | 2 | 0.00% |
| negative trait | 2 | 0.00% |
| Time Schedule | 2 | 0.00% |
| Program/Initiative | 2 | 0.00% |
| Education level | 2 | 0.00% |
| entity_name | 2 | 0.00% |
| financial indicator | 2 | 0.00% |
| financial trend | 2 | 0.00% |
| Government Officials | 2 | 0.00% |
| Legal Action | 2 | 0.00% |
| Natural resource | 2 | 0.00% |
| invoice_number | 2 | 0.00% |
| Social Media Engagement | 2 | 0.00% |
| nummod | 2 | 0.00% |
| cc | 2 | 0.00% |
| det | 2 | 0.00% |
| quantity/group | 2 | 0.00% |
| Medical term | 2 | 0.00% |
| Businessman | 2 | 0.00% |
| Industry sector | 2 | 0.00% |
| Business term | 2 | 0.00% |
| userId | 2 | 0.00% |
| insurance term | 2 | 0.00% |
| skincare routine | 2 | 0.00% |
| skincare process | 2 | 0.00% |
| Monument | 2 | 0.00% |
| natural landmarks | 2 | 0.00% |
| Music Fans | 2 | 0.00% |
| Technology/Platform | 2 | 0.00% |
| cannabis strain | 2 | 0.00% |
| cannabinoid | 2 | 0.00% |
| Musical Quality | 2 | 0.00% |
| text attribute | 2 | 0.00% |
| Home decor item | 2 | 0.00% |
| status: Current status of the transaction | 2 | 0.00% |
| command | 2 | 0.00% |
| district | 2 | 0.00% |
| Directional Verb | 2 | 0.00% |
| membership program | 2 | 0.00% |
| Car Modification | 2 | 0.00% |
| Manufacturing | 2 | 0.00% |
| VPN feature | 2 | 0.00% |
| legislative body | 2 | 0.00% |
| excipient | 2 | 0.00% |
| historical site | 2 | 0.00% |
| Packaging Material | 2 | 0.00% |
| player_age | 2 | 0.00% |
| winner | 2 | 0.00% |
| discovery_year | 2 | 0.00% |
| patient_care.group_therapy.benefits | 2 | 0.00% |
| geological feature | 2 | 0.00% |
| Personal trait | 2 | 0.00% |
| action/condition | 2 | 0.00% |
| Sightseeing | 2 | 0.00% |
| Mobile Application | 2 | 0.00% |
| Fictional characters | 2 | 0.00% |
| dental procedure | 2 | 0.00% |
| project_name | 2 | 0.00% |
| Lighting | 2 | 0.00% |
| Car feature | 2 | 0.00% |
| Restaurant/Bar | 2 | 0.00% |
| Dental treatment | 2 | 0.00% |
| Evaluation Criteria | 2 | 0.00% |
| Plant Family | 2 | 0.00% |
| Analysis Technique | 2 | 0.00% |
| pause | 2 | 0.00% |
| Tax | 2 | 0.00% |
| human resource | 2 | 0.00% |
| Condiment | 2 | 0.00% |
| day of week | 2 | 0.00% |
| distribution_information.trail_details.trail_endpoints | 2 | 0.00% |
| environmental condition | 2 | 0.00% |
| dessert | 2 | 0.00% |
| Emotion/State of mind | 2 | 0.00% |
| employment | 2 | 0.00% |
| health status | 2 | 0.00% |
| Property Features | 2 | 0.00% |
| stretching tool | 2 | 0.00% |
| negative attitude | 2 | 0.00% |
| contactNumber | 2 | 0.00% |
| signature | 2 | 0.00% |
| Service Plans | 2 | 0.00% |
| Budget | 2 | 0.00% |
| Insurance Provider | 2 | 0.00% |
| Undefined Entity | 2 | 0.00% |
| sacrament | 2 | 0.00% |
| Date and Time | 2 | 0.00% |
| educational entity | 2 | 0.00% |
| Government Sector | 2 | 0.00% |
| Person/Company | 2 | 0.00% |
| Person/Composer | 2 | 0.00% |
| malware | 2 | 0.00% |
| Author(s) | 2 | 0.00% |
| filing_date | 2 | 0.00% |
| damage_type | 2 | 0.00% |
| incident | 2 | 0.00% |
| Performance venue/Theater | 2 | 0.00% |
| scheduling | 2 | 0.00% |
| unexpected events | 2 | 0.00% |
| supportingDocuments | 2 | 0.00% |
| learning | 2 | 0.00% |
| course_id | 2 | 0.00% |
| credits | 2 | 0.00% |
| Military Conflict | 2 | 0.00% |
| Military Rank | 2 | 0.00% |
| Finish | 2 | 0.00% |
| financial gain | 2 | 0.00% |
| publication_date | 2 | 0.00% |
| Engine Model | 2 | 0.00% |
| Horsepower | 2 | 0.00% |
| Medical Event | 2 | 0.00% |
| Shipping Method | 2 | 0.00% |
| Financial resource | 2 | 0.00% |
| Address/Location | 2 | 0.00% |
| Country/Geographical Location | 2 | 0.00% |
| Building/Structure | 2 | 0.00% |
| Honorific | 2 | 0.00% |
| Rodenticide | 2 | 0.00% |
| Conference Name | 2 | 0.00% |
| legal agreement | 2 | 0.00% |
| Sports Season | 2 | 0.00% |
| Competition Placement | 2 | 0.00% |
| Language Translation | 2 | 0.00% |
| Imaging technology | 2 | 0.00% |
| Political Organization | 2 | 0.00% |
| Political Event | 2 | 0.00% |
| Streaming Service | 2 | 0.00% |
| Treatment Product | 2 | 0.00% |
| Professional Group | 2 | 0.00% |
| preservation method | 2 | 0.00% |
| Government Service | 2 | 0.00% |
| activity/task | 2 | 0.00% |
| Authentication Encryption | 2 | 0.00% |
| Game platform | 2 | 0.00% |
| task type | 2 | 0.00% |
| Artistic work | 2 | 0.00% |
| asteroid_number | 2 | 0.00% |
| Betting Term | 2 | 0.00% |
| Plumbing Issue | 2 | 0.00% |
| quality of work | 2 | 0.00% |
| Educational programme | 2 | 0.00% |
| Financial Burden | 2 | 0.00% |
| Automotive Service | 2 | 0.00% |
| Setting | 2 | 0.00% |
| pattern piece | 2 | 0.00% |
| pattern instruction | 2 | 0.00% |
| type of fabric | 2 | 0.00% |
| Brakes | 2 | 0.00% |
| Request | 2 | 0.00% |
| Access type | 2 | 0.00% |
| functionality | 2 | 0.00% |
| Style element | 2 | 0.00% |
| Music Notation | 2 | 0.00% |
| Preference | 2 | 0.00% |
| Education and Training | 2 | 0.00% |
| language: List of languages the publication is available in | 2 | 0.00% |
| military rank/title | 2 | 0.00% |
| Field Type | 2 | 0.00% |
| academic term | 2 | 0.00% |
| asteroid.discovery.date | 2 | 0.00% |
| Media/Content | 2 | 0.00% |
| Person/Identity | 2 | 0.00% |
| social space | 2 | 0.00% |
| record keeping | 2 | 0.00% |
| metric | 2 | 0.00% |
| threat | 2 | 0.00% |
| object/technology | 2 | 0.00% |
| Social gathering | 2 | 0.00% |
| Personal information | 2 | 0.00% |
| Internet Service | 2 | 0.00% |
| office | 2 | 0.00% |
| Criminals | 2 | 0.00% |
| person/work | 2 | 0.00% |
| addiction | 2 | 0.00% |
| Activity/Entertainment | 2 | 0.00% |
| service feature | 2 | 0.00% |
| Financial Offering | 2 | 0.00% |
| Trail markers | 2 | 0.00% |
| Public land | 2 | 0.00% |
| Material Science | 2 | 0.00% |
| Person entity | 2 | 0.00% |
| providerId | 2 | 0.00% |
| SEO strategy | 2 | 0.00% |
| TV show season | 2 | 0.00% |
| sports competition | 2 | 0.00% |
| scientific_name | 2 | 0.00% |
| Personal Pronoun | 2 | 0.00% |
| Nature Reserve | 2 | 0.00% |
| Collection Name | 2 | 0.00% |
| orbital_characteristics.asteroid_belt_classification | 2 | 0.00% |
| news organization | 2 | 0.00% |
| adverse_effects: List of side effects reported in clinical use | 2 | 0.00% |
| music | 2 | 0.00% |
| Bird | 2 | 0.00% |
| legal system | 2 | 0.00% |
| physical_characteristics.observations.0.wavelength | 2 | 0.00% |
| monetary transaction | 2 | 0.00% |
| production process | 2 | 0.00% |
| quality control | 2 | 0.00% |
| patient type | 2 | 0.00% |
| people/individuals | 2 | 0.00% |
| technology standard | 2 | 0.00% |
| employee benefits | 2 | 0.00% |
| comprehension | 2 | 0.00% |
| business term | 2 | 0.00% |
| attire | 2 | 0.00% |
| home improvement item | 2 | 0.00% |
| Person/Instructor | 2 | 0.00% |
| Mindfulness/Approach | 2 | 0.00% |
| Investment | 2 | 0.00% |
| Educational resource | 2 | 0.00% |
| Fitness equipment | 2 | 0.00% |
| Game Element | 2 | 0.00% |
| challenge/problem | 2 | 0.00% |
| Natural entity | 2 | 0.00% |
| Restriction | 2 | 0.00% |
| GPS device | 2 | 0.00% |
| Occupation/Interest | 2 | 0.00% |
| Religious figure | 2 | 0.00% |
| Legal Institution | 2 | 0.00% |
| Agriculture | 2 | 0.00% |
| Political Faction | 2 | 0.00% |
| Extension Number | 2 | 0.00% |
| membership | 2 | 0.00% |
| starting point | 2 | 0.00% |
| career | 2 | 0.00% |
| City/State | 2 | 0.00% |
| Story | 2 | 0.00% |
| Relationship type | 2 | 0.00% |
| instructor | 2 | 0.00% |
| Transportation hub | 2 | 0.00% |
| literary writing | 2 | 0.00% |
| Business Entities | 2 | 0.00% |
| Metaphor/Symbol | 2 | 0.00% |
| Task/Activity | 2 | 0.00% |
| Spiritual Practice | 2 | 0.00% |
| submission_date | 2 | 0.00% |
| deliveryTerms | 2 | 0.00% |
| Event Date | 2 | 0.00% |
| Nail art service | 2 | 0.00% |
| Cultural Group | 2 | 0.00% |
| Natural Resources | 2 | 0.00% |
| E-commerce platform | 2 | 0.00% |
| job seekers | 2 | 0.00% |
| claim_amount | 2 | 0.00% |
| modification type | 2 | 0.00% |
| access control | 2 | 0.00% |
| Movie Character | 2 | 0.00% |
| innovation | 2 | 0.00% |
| website/forum | 2 | 0.00% |
| software functionality | 2 | 0.00% |
| malfunction | 2 | 0.00% |
| Negative quality | 2 | 0.00% |
| Consumer goods stocks | 2 | 0.00% |
| Commodity Entity | 2 | 0.00% |
| Trapped Days | 2 | 0.00% |
| Educational Board | 2 | 0.00% |
| Capacity | 2 | 0.00% |
| Possessive Pronoun | 2 | 0.00% |
| protective equipment | 2 | 0.00% |
| CBD product | 2 | 0.00% |
| Option | 2 | 0.00% |
| Taste/Texture | 2 | 0.00% |
| group name | 2 | 0.00% |
| flower species | 2 | 0.00% |
| Construction project | 2 | 0.00% |
| Government Ministry | 2 | 0.00% |
| Entity type (sporting events) | 2 | 0.00% |
| governmental programs | 2 | 0.00% |
| religious building | 2 | 0.00% |
| Music/Art | 2 | 0.00% |
| Chess player | 2 | 0.00% |
| programming language construct | 2 | 0.00% |
| confirmationNumber | 2 | 0.00% |
| hotelName | 2 | 0.00% |
| occupancy | 2 | 0.00% |
| specialInstructions | 2 | 0.00% |
| foot condition | 2 | 0.00% |
| Service entity | 2 | 0.00% |
| fashion items | 2 | 0.00% |
| Personal Care Product | 2 | 0.00% |
| industry/domain | 2 | 0.00% |
| Chapter Number | 2 | 0.00% |
| custom | 2 | 0.00% |
| Product Review | 2 | 0.00% |
| Wildlife | 2 | 0.00% |
| Time Indicator | 2 | 0.00% |
| Racer/driver | 2 | 0.00% |
| television program | 2 | 0.00% |
| Quality Entity | 2 | 0.00% |
| evaluation | 2 | 0.00% |
| occupational group | 2 | 0.00% |
| payment process | 2 | 0.00% |
| cost savings | 2 | 0.00% |
| Postal code | 2 | 0.00% |
| Media file | 2 | 0.00% |
| behavior/moral value | 2 | 0.00% |
| action/moral value | 2 | 0.00% |
| naming convention | 2 | 0.00% |
| home decor store | 2 | 0.00% |
| earning opportunity | 2 | 0.00% |
| advertising | 2 | 0.00% |
| commerce | 2 | 0.00% |
| launch_year | 2 | 0.00% |
| Storage Capacity | 2 | 0.00% |
| score: List of two integers representing home and away scores | 2 | 0.00% |
| Saint | 2 | 0.00% |
| Religious building | 2 | 0.00% |
| Religious group | 2 | 0.00% |
| Environmental Factors | 2 | 0.00% |
| research study | 2 | 0.00% |
| company.location | 2 | 0.00% |
| Basic Need | 2 | 0.00% |
| Necessity | 2 | 0.00% |
| Technology/Innovation | 2 | 0.00% |
| Culinary Arts | 2 | 0.00% |
| social norm | 2 | 0.00% |
| personal challenge | 2 | 0.00% |
| Computer System | 2 | 0.00% |
| societal group | 2 | 0.00% |
| Publication/Brand | 2 | 0.00% |
| Sports season | 2 | 0.00% |
| Mode of transportation | 2 | 0.00% |
| system function | 2 | 0.00% |
| filter | 2 | 0.00% |
| Membership type | 2 | 0.00% |
| Dog breed | 2 | 0.00% |
| priority_registration.submission_methods | 2 | 0.00% |
| Checkout Process | 2 | 0.00% |
| Vehicle model | 2 | 0.00% |
| Human Entity | 2 | 0.00% |
| book edition | 2 | 0.00% |
| Skill set | 2 | 0.00% |
| research method | 2 | 0.00% |
| Decision | 2 | 0.00% |
| vehicles | 2 | 0.00% |
| Climate phenomenon | 2 | 0.00% |
| Academic year | 2 | 0.00% |
| postal code | 2 | 0.00% |
| Time/Day | 2 | 0.00% |
| Person/Designer | 2 | 0.00% |
| Player | 2 | 0.00% |
| type of insurance | 2 | 0.00% |
| business expense | 2 | 0.00% |
| technology/infrastructure | 2 | 0.00% |
| Zoo | 2 | 0.00% |
| updatedAt | 2 | 0.00% |
| from_club | 2 | 0.00% |
| player | 2 | 0.00% |
| Medical Practitioner | 2 | 0.00% |
| body modification | 2 | 0.00% |
| physical fitness | 2 | 0.00% |
| Artist/Creator | 2 | 0.00% |
| Consumer/Group | 2 | 0.00% |
| purchaseOrderId | 2 | 0.00% |
| supplierAddress | 2 | 0.00% |
| itemName | 2 | 0.00% |
| itemDescription | 2 | 0.00% |
| itemQuantity | 2 | 0.00% |
| itemPrice | 2 | 0.00% |
| Community/Group | 2 | 0.00% |
| Religious affiliation | 2 | 0.00% |
| Manufacturing Process | 2 | 0.00% |
| Research Institute | 2 | 0.00% |
| training equipment | 2 | 0.00% |
| dimensions | 2 | 0.00% |
| software package | 2 | 0.00% |
| Interest Group | 2 | 0.00% |
| policy_type | 2 | 0.00% |
| flora | 2 | 0.00% |
| aesthetic | 2 | 0.00% |
| Documentation | 2 | 0.00% |
| Characteristics | 2 | 0.00% |
| Company type | 2 | 0.00% |
| personal well-being | 2 | 0.00% |
| road marking | 2 | 0.00% |
| treatment.drugs | 2 | 0.00% |
| plot point | 2 | 0.00% |
| Service Availability | 2 | 0.00% |
| Law Enforcement Activities | 2 | 0.00% |
| magazine | 2 | 0.00% |
| travel amenity | 2 | 0.00% |
| Test outcome | 2 | 0.00% |
| Technical issue | 2 | 0.00% |
| equipmentType | 2 | 0.00% |
| dwelling | 2 | 0.00% |
| heating system | 2 | 0.00% |
| dog breed | 2 | 0.00% |
| dog weight | 2 | 0.00% |
| medical profession | 2 | 0.00% |
| educational resources | 2 | 0.00% |
| Director Name | 2 | 0.00% |
| car audio component | 2 | 0.00% |
| business owners | 2 | 0.00% |
| martial arts style | 2 | 0.00% |
| training center | 2 | 0.00% |
| challenge | 2 | 0.00% |
| cultural practice | 2 | 0.00% |
| man-made structure | 2 | 0.00% |
| taste quality | 2 | 0.00% |
| historical figures | 2 | 0.00% |
| Literature Works | 2 | 0.00% |
| Institution/Organization | 2 | 0.00% |
| Government Organization | 2 | 0.00% |
| Object/Tool | 2 | 0.00% |
| device type | 2 | 0.00% |
| Permission | 2 | 0.00% |
| timekeeping device | 2 | 0.00% |
| Database Management System | 2 | 0.00% |
| Safety measures | 2 | 0.00% |
| system feature | 2 | 0.00% |
| donor_name | 2 | 0.00% |
| List | 2 | 0.00% |
| business/service | 2 | 0.00% |
| Skill/Talent | 2 | 0.00% |
| Gemstone | 2 | 0.00% |
| General entity | 2 | 0.00% |
| Temporal entity | 2 | 0.00% |
| Event entity | 2 | 0.00% |
| Bacterium | 2 | 0.00% |
| Tourist destination | 2 | 0.00% |
| Meeting event | 2 | 0.00% |
| Customer Requirement | 2 | 0.00% |
| positive sentiment | 2 | 0.00% |
| User Account | 2 | 0.00% |
| Membership Level | 2 | 0.00% |
| Bathroom Feature | 2 | 0.00% |
| Ship | 2 | 0.00% |
| host | 2 | 0.00% |
| attendees | 2 | 0.00% |
| Webpage element | 2 | 0.00% |
| Data field | 2 | 0.00% |
| financial document | 2 | 0.00% |
| firefighting terminology | 2 | 0.00% |
| Online forum | 2 | 0.00% |
| Product Size | 2 | 0.00% |
| Fashion Trend | 2 | 0.00% |
| Movie franchise | 2 | 0.00% |
| Volume/Issue Number | 2 | 0.00% |
| Security System | 2 | 0.00% |
| image quality | 2 | 0.00% |
| Writing | 2 | 0.00% |
| social media account | 2 | 0.00% |
| email address | 2 | 0.00% |
| Football League | 2 | 0.00% |
| age category | 2 | 0.00% |
| Regulation | 2 | 0.00% |
| State of Being | 2 | 0.00% |
| functional regulatory element | 2 | 0.00% |
| Action/Move | 2 | 0.00% |
| e_34 | 2 | 0.00% |
| e_35 | 2 | 0.00% |
| Land Area | 2 | 0.00% |
| Decorative Object | 2 | 0.00% |
| Teaching Material | 2 | 0.00% |
| Heritage Site | 2 | 0.00% |
| Trophy | 2 | 0.00% |
| Graduating class | 2 | 0.00% |
| Assistance | 2 | 0.00% |
| requisition_id: Unique identifier for the job posting | 2 | 0.00% |
| data copy | 2 | 0.00% |
| Hardware/Software | 2 | 0.00% |
| emotional entity | 2 | 0.00% |
| Material/Design Element | 2 | 0.00% |
| process/action | 2 | 0.00% |
| Business Clients | 2 | 0.00% |
| Console | 2 | 0.00% |
| Gift | 2 | 0.00% |
| aspiration | 2 | 0.00% |
| Strategy | 2 | 0.00% |
| exhibition/event | 2 | 0.00% |
| Unit of measurement | 2 | 0.00% |
| educational course | 2 | 0.00% |
| network communication | 2 | 0.00% |
| lawyer | 2 | 0.00% |
| confusion | 2 | 0.00% |
| time period/year | 2 | 0.00% |
| crafting tool | 2 | 0.00% |
| News source | 2 | 0.00% |
| legal profession | 2 | 0.00% |
| illegal activity | 2 | 0.00% |
| retentionPeriodUnit | 2 | 0.00% |
| Outdoor property feature | 2 | 0.00% |
| Room in a house | 2 | 0.00% |
| dish/food | 2 | 0.00% |
| physical discomfort | 2 | 0.00% |
| currency amount | 2 | 0.00% |
| period | 2 | 0.00% |
| process type | 2 | 0.00% |
| Information Type | 2 | 0.00% |
| database | 2 | 0.00% |
| manager | 2 | 0.00% |
| People Group | 2 | 0.00% |
| experience/activity | 2 | 0.00% |
| Keyboard shortcut | 2 | 0.00% |
| Environmental factor | 2 | 0.00% |
| ordinal number | 2 | 0.00% |
| weapon type | 2 | 0.00% |
| file directory | 2 | 0.00% |
| technical entity | 2 | 0.00% |
| certification holders | 2 | 0.00% |
| Purchase | 2 | 0.00% |
| fuel source | 2 | 0.00% |
| Athletic conference | 2 | 0.00% |
| Victory | 2 | 0.00% |
| supply | 2 | 0.00% |
| Objects | 2 | 0.00% |
| living space | 2 | 0.00% |
| Technology Product | 2 | 0.00% |
| provider | 2 | 0.00% |
| Home Interior | 2 | 0.00% |
| Taxonomy | 2 | 0.00% |
| Witness | 2 | 0.00% |
| Gesture | 2 | 0.00% |
| Plant structure | 2 | 0.00% |
| currency conversion rate | 2 | 0.00% |
| coin denomination | 2 | 0.00% |
| Unknown Entity Type | 2 | 0.00% |
| cleaning tool | 2 | 0.00% |
| Height | 2 | 0.00% |
| Contact Number | 2 | 0.00% |
| personal trait/quality | 2 | 0.00% |
| Referral source | 2 | 0.00% |
| musical note | 2 | 0.00% |
| Legal status | 2 | 0.00% |
| amount: Monetary value of the transaction | 2 | 0.00% |
| Movie/Book | 2 | 0.00% |
| personal growth | 2 | 0.00% |
| Sport/Activity | 2 | 0.00% |
| deliveryTime | 2 | 0.00% |
| Design Elements | 2 | 0.00% |
| Building Structure | 2 | 0.00% |
| speed measurement | 2 | 0.00% |
| interest/hobby | 2 | 0.00% |
| service.interventions | 2 | 0.00% |
| shipping_information.bank_transfer_details.sepa_transfer_fee.other_banks | 2 | 0.00% |
| bathroom feature | 2 | 0.00% |
| Safety feature | 2 | 0.00% |
| Winter Sports | 2 | 0.00% |
| part | 2 | 0.00% |
| user action | 2 | 0.00% |
| vacation | 2 | 0.00% |
| metal | 2 | 0.00% |
| Navigation | 2 | 0.00% |
| Clothing Collection | 2 | 0.00% |
| Theme | 2 | 0.00% |
| child care arrangement | 2 | 0.00% |
| Human Resources | 2 | 0.00% |
| Artistic activity | 2 | 0.00% |
| Service/Process | 2 | 0.00% |
| cleanliness | 2 | 0.00% |
| flavor | 2 | 0.00% |
| Sports position | 2 | 0.00% |
| dental product | 2 | 0.00% |
| Financial Document | 2 | 0.00% |
| Product quality | 2 | 0.00% |
| smell | 2 | 0.00% |
| Volume Remote Controler | 2 | 0.00% |
| Experience Level | 2 | 0.00% |
| Financing Option | 2 | 0.00% |
| Workplace Issue | 2 | 0.00% |
| cause | 2 | 0.00% |
| urls | 2 | 0.00% |
| Material Quality | 2 | 0.00% |
| drug type | 2 | 0.00% |
| Business/Finance | 2 | 0.00% |
| Professional Footballer | 2 | 0.00% |
| amountPaid | 2 | 0.00% |
| person or group of people | 2 | 0.00% |
| Financial metric | 2 | 0.00% |
| dataset | 2 | 0.00% |
| individuals receiving medical care | 2 | 0.00% |
| comfort | 2 | 0.00% |
| Life decision | 2 | 0.00% |
| medical school | 2 | 0.00% |
| Goal/Intention | 2 | 0.00% |
| writing style | 2 | 0.00% |
| type of accommodation | 2 | 0.00% |
| dispute_id | 2 | 0.00% |
| Entertainment facility | 2 | 0.00% |
| Community Group | 2 | 0.00% |
| Kite Performance | 2 | 0.00% |
| product features | 2 | 0.00% |
| File System | 2 | 0.00% |
| Dietary Restriction | 2 | 0.00% |
| policy/law | 2 | 0.00% |
| Staff | 2 | 0.00% |
| tenant_name | 2 | 0.00% |
| additional_features | 2 | 0.00% |
| Personal | 2 | 0.00% |
| product identifier | 2 | 0.00% |
| Publication Type | 2 | 0.00% |
| Flooring Material | 2 | 0.00% |
| plumbing issue | 2 | 0.00% |
| species.scientific_name | 2 | 0.00% |
| species.description | 2 | 0.00% |
| Product Color | 2 | 0.00% |
| Technology Platform | 2 | 0.00% |
| Online Casino | 2 | 0.00% |
| exhibitor_id | 2 | 0.00% |
| company_name | 2 | 0.00% |
| registration_date | 2 | 0.00% |
| cognitive process | 2 | 0.00% |
| Institution/Building | 2 | 0.00% |
| Event/Theme | 2 | 0.00% |
| geographic feature | 2 | 0.00% |
| program duration | 2 | 0.00% |
| Beauty product | 2 | 0.00% |
| Temperature Range | 2 | 0.00% |
| founder | 2 | 0.00% |
| Leisure activity | 2 | 0.00% |
| Magnification | 2 | 0.00% |
| airline | 2 | 0.00% |
| person/company | 2 | 0.00% |
| Boat Equipment | 2 | 0.00% |
| budget | 2 | 0.00% |
| Web Page | 2 | 0.00% |
| Story Element | 2 | 0.00% |
| Emotions/Feelings | 2 | 0.00% |
| Law enforcement rank | 2 | 0.00% |
| quality/attribute | 2 | 0.00% |
| activationDate | 2 | 0.00% |
| Activity/Topic | 2 | 0.00% |
| south_yemen_conflict.historical_context.actions_after_invasion | 2 | 0.00% |
| Musician/Artist | 2 | 0.00% |
| Concert Tour | 2 | 0.00% |
| Personal Entity | 2 | 0.00% |
| Spiritual Belief | 2 | 0.00% |
| financial planning | 2 | 0.00% |
| taboo behavior | 2 | 0.00% |
| Online Presence | 2 | 0.00% |
| travel experience | 2 | 0.00% |
| Chess tournament | 2 | 0.00% |
| Tour Duration | 2 | 0.00% |
| Launch Country | 2 | 0.00% |
| person/group of people | 2 | 0.00% |
| phone accessory | 2 | 0.00% |
| Individual entity | 2 | 0.00% |
| Person/Professional Wrestler | 2 | 0.00% |
| Tag Team/Professional Wrestlers | 2 | 0.00% |
| Championship/Professional Wrestling | 2 | 0.00% |
| Funeral Home | 2 | 0.00% |
| Vehicle history report | 2 | 0.00% |
| time measurement device | 2 | 0.00% |
| population | 2 | 0.00% |
| Supernatural Entity | 2 | 0.00% |
| Game Genre | 2 | 0.00% |
| Rescue Start Time | 2 | 0.00% |
| Legal requirement | 2 | 0.00% |
| animal housing | 2 | 0.00% |
| Natural environment | 2 | 0.00% |
| Business Agreement | 2 | 0.00% |
| restriction | 2 | 0.00% |
| location.district | 2 | 0.00% |
| online forum | 2 | 0.00% |
| Visual design | 2 | 0.00% |
| Resort | 2 | 0.00% |
| Memory allocation | 2 | 0.00% |
| Bike Light | 2 | 0.00% |
| Animal/Pet | 2 | 0.00% |
| educational degree | 2 | 0.00% |
| color type | 2 | 0.00% |
| authority.name | 2 | 0.00% |
| aircraft types | 2 | 0.00% |
| Government Position | 2 | 0.00% |
| Personnel | 2 | 0.00% |
| furniture component | 2 | 0.00% |
| feedback | 2 | 0.00% |
| gardening | 2 | 0.00% |
| insurance product | 2 | 0.00% |
| seasonNumber | 2 | 0.00% |
| customer relation | 2 | 0.00% |
| rules and regulations | 2 | 0.00% |
| Hair product | 2 | 0.00% |
| National Team | 2 | 0.00% |
| focus_area | 2 | 0.00% |
| Decorative Element | 2 | 0.00% |
| physical quantity | 2 | 0.00% |
| recreational feature | 2 | 0.00% |
| Plant Material | 2 | 0.00% |
| Art Technique | 2 | 0.00% |
| Journey | 2 | 0.00% |
| Breathing Event | 2 | 0.00% |
| Cryptography | 2 | 0.00% |
| Legal entities | 2 | 0.00% |
| Towels | 2 | 0.00% |
| Edible Item | 2 | 0.00% |
| serial_number | 2 | 0.00% |
| mobile operating system | 2 | 0.00% |
| Music and Podcast Platform | 2 | 0.00% |
| mrn | 2 | 0.00% |
| explanation | 2 | 0.00% |
| Country/Location | 2 | 0.00% |
| Online content | 2 | 0.00% |
| Military Units | 2 | 0.00% |
| Difficulty level | 2 | 0.00% |
| Visual Entity | 2 | 0.00% |
| social status | 2 | 0.00% |
| lead_researcher | 2 | 0.00% |
| launch_date | 2 | 0.00% |
| Cemetery | 2 | 0.00% |
| booking_options.partnerships | 2 | 0.00% |
| location.province | 2 | 0.00% |
| TV episode | 2 | 0.00% |
| Malware Type | 2 | 0.00% |
| data format | 2 | 0.00% |
| strength | 2 | 0.00% |
| economic entity | 2 | 0.00% |
| Environmental trigger | 2 | 0.00% |
| irrigation equipment | 2 | 0.00% |
| Example Document | 2 | 0.00% |
| patient_age | 2 | 0.00% |
| hospital | 2 | 0.00% |
| Government | 2 | 0.00% |
| Communication Channel | 2 | 0.00% |
| problem type | 2 | 0.00% |
| Disaster | 2 | 0.00% |
| Media Content | 2 | 0.00% |
| Cognitive Process | 2 | 0.00% |
| carotenoid | 2 | 0.00% |
| optical property | 2 | 0.00% |
| biological tissue | 2 | 0.00% |
| molecular entity | 2 | 0.00% |
| Communication Protocol | 2 | 0.00% |
| Computing Device | 2 | 0.00% |
| tool feature | 2 | 0.00% |
| microorganism | 2 | 0.00% |
| Convenience | 2 | 0.00% |
| transportation facility | 2 | 0.00% |
| mental process | 2 | 0.00% |
| Fast food chain | 2 | 0.00% |
| Doctor and Medical Team | 2 | 0.00% |
| stationName | 2 | 0.00% |
| scheduleDate | 2 | 0.00% |
| Abbreviation/Measurement | 2 | 0.00% |
| chance | 2 | 0.00% |
| genus.name | 2 | 0.00% |
| Campaign | 2 | 0.00% |
| Technical parameter | 2 | 0.00% |
| Music artist | 2 | 0.00% |
| hotel_information.alternative_names | 2 | 0.00% |
| environmental policy | 2 | 0.00% |
| Printer | 2 | 0.00% |
| Color Option | 2 | 0.00% |
| educational status | 2 | 0.00% |
| Animal breed | 2 | 0.00% |
| spending | 2 | 0.00% |
| healthcare services | 2 | 0.00% |
| rhetorical device | 2 | 0.00% |
| print media | 2 | 0.00% |
| Event Registration | 2 | 0.00% |
| Drying instruction | 2 | 0.00% |
| mimeType | 2 | 0.00% |
| Man-made features | 2 | 0.00% |
| regulatory body | 2 | 0.00% |
| Race Car Driver | 2 | 0.00% |
| Racing Team | 2 | 0.00% |
| Wedding venue | 2 | 0.00% |
| Social event | 2 | 0.00% |
| Service provider | 2 | 0.00% |
| Toy/Recreation | 2 | 0.00% |
| Physical Appearance | 2 | 0.00% |
| Skill/Activity | 2 | 0.00% |
| Academic credit system | 2 | 0.00% |
| Technology/Service | 2 | 0.00% |
| meeting | 2 | 0.00% |
| Degree/Certification | 2 | 0.00% |
| graphic element | 2 | 0.00% |
| Data type entity | 2 | 0.00% |
| photography service | 2 | 0.00% |
| Person/Pro Wrestler | 2 | 0.00% |
| Emotion/Action | 2 | 0.00% |
| Mode | 2 | 0.00% |
| food dish | 2 | 0.00% |
| subtotal | 2 | 0.00% |
| tax | 2 | 0.00% |
| Musical Ensemble | 2 | 0.00% |
| note | 2 | 0.00% |
| report_id: Unique identifier for the phishing incident | 2 | 0.00% |
| business model | 2 | 0.00% |
| building structure | 2 | 0.00% |
| Digital media | 2 | 0.00% |
| Report | 2 | 0.00% |
| Financial Value | 2 | 0.00% |
| place.location.country | 2 | 0.00% |
| place.type | 2 | 0.00% |
| value proposition | 2 | 0.00% |
| Geological formation | 2 | 0.00% |
| Architecture/Design | 2 | 0.00% |
| project_status | 2 | 0.00% |
| Treatment Facility | 2 | 0.00% |
| Company/Manufacturer | 2 | 0.00% |
| Organ/Body Part | 2 | 0.00% |
| marine organisms | 2 | 0.00% |
| educators | 2 | 0.00% |
| scoring method | 2 | 0.00% |
| Clothing Feature | 2 | 0.00% |
| Clothing Size | 2 | 0.00% |
| Connection Type | 2 | 0.00% |
| Television show | 2 | 0.00% |
| booth | 2 | 0.00% |
| organizational unit | 2 | 0.00% |
| 학교 | 2 | 0.00% |
| 언어 | 2 | 0.00% |
| 일정표 | 2 | 0.00% |
| 웹사이트 | 2 | 0.00% |
| religious holiday | 2 | 0.00% |
| Product Design | 2 | 0.00% |
| Zip code | 2 | 0.00% |
| Size/Measurement | 2 | 0.00% |
| improvement | 2 | 0.00% |
| gardening practice | 2 | 0.00% |
| Holiday season | 2 | 0.00% |
| religious activity | 2 | 0.00% |
| allergen/food intolerance | 2 | 0.00% |
| Textiles | 2 | 0.00% |
| inventor | 2 | 0.00% |
| visual feature | 2 | 0.00% |
| ethnicity | 2 | 0.00% |
| type of image | 2 | 0.00% |
| Certification Exam | 2 | 0.00% |
| Food/ Dish | 2 | 0.00% |
| NBA team | 2 | 0.00% |
| post_id: Unique identifier for the social media post | 2 | 0.00% |
| Real Estate Development | 2 | 0.00% |
| Ticket Option | 2 | 0.00% |
| Temple | 2 | 0.00% |
| Village | 2 | 0.00% |
| academic document | 2 | 0.00% |
| Hardware specification | 2 | 0.00% |
| Eye type | 2 | 0.00% |
| Money/Currency | 2 | 0.00% |
| Position/Location | 2 | 0.00% |
| Time interval | 2 | 0.00% |
| city/town | 2 | 0.00% |
| land measurement | 2 | 0.00% |
| appointment | 2 | 0.00% |
| photography equipment | 2 | 0.00% |
| Game Period | 2 | 0.00% |
| Game Outcome | 2 | 0.00% |
| Game Strategy | 2 | 0.00% |
| Part/Component | 2 | 0.00% |
| personal relationships | 2 | 0.00% |
| action or behavior | 2 | 0.00% |
| Film genre | 2 | 0.00% |
| Event/Period | 2 | 0.00% |
| software interface element | 2 | 0.00% |
| Artistic creation | 2 | 0.00% |
| animal behavior | 2 | 0.00% |
| Belief | 2 | 0.00% |
| scent | 2 | 0.00% |
| Style/Approach | 2 | 0.00% |
| Exhibition | 2 | 0.00% |
| Archaeological site | 2 | 0.00% |
| Geographical feature | 2 | 0.00% |
| Nationality/Demonym | 2 | 0.00% |
| material/style | 2 | 0.00% |
| precious stone | 2 | 0.00% |
| queue | 2 | 0.00% |
| exam | 2 | 0.00% |
| guarantee | 2 | 0.00% |
| Material/Design | 2 | 0.00% |
| Band/Music Group | 2 | 0.00% |
| Job Responsibility | 2 | 0.00% |
| date: Date when the transaction occurred | 2 | 0.00% |
| Length | 2 | 0.00% |
| Booking | 2 | 0.00% |
| Religious activity | 2 | 0.00% |
| concert stage equipment | 2 | 0.00% |
| Legal Status | 2 | 0.00% |
| Entity type not clear | 2 | 0.00% |
| CNC Router | 2 | 0.00% |
| stopId | 2 | 0.00% |
| stopName | 2 | 0.00% |
| alcoholic beverage | 2 | 0.00% |
| forum category | 2 | 0.00% |
| mineral property | 2 | 0.00% |
| Principal | 2 | 0.00% |
| therapy | 2 | 0.00% |
| Analytics Tool | 2 | 0.00% |
| indefinite pronoun | 2 | 0.00% |
| familiarity | 2 | 0.00% |
| behavior/action | 2 | 0.00% |
| appointment_id | 2 | 0.00% |
| Certification Program | 2 | 0.00% |
| Human being | 2 | 0.00% |
| online content | 2 | 0.00% |
| Prospect Type | 2 | 0.00% |
| key_player | 2 | 0.00% |
| entertainment product | 2 | 0.00% |
| museums/artifacts | 2 | 0.00% |
| language processing tool | 2 | 0.00% |
| dwelling place | 2 | 0.00% |
| HTML element | 2 | 0.00% |
| Unit of Measure | 2 | 0.00% |
| enjoyable experience | 2 | 0.00% |
| hashtags | 2 | 0.00% |
| Social Role | 2 | 0.00% |
| aircraft | 2 | 0.00% |
| mechanical issue | 2 | 0.00% |
| month/year | 2 | 0.00% |
| type of car rental | 2 | 0.00% |
| Land | 2 | 0.00% |
| microscopy technique | 2 | 0.00% |
| Email service | 2 | 0.00% |
| Dog | 2 | 0.00% |
| Social media account | 2 | 0.00% |
| educational institutions | 2 | 0.00% |
| policyNumber | 2 | 0.00% |
| online gaming | 2 | 0.00% |
| wine type | 2 | 0.00% |
| basic needs | 2 | 0.00% |
| street address | 2 | 0.00% |
| Energy capacity | 2 | 0.00% |
| dish ingredient | 2 | 0.00% |
| Industry Sector | 2 | 0.00% |
| Artwork/Title | 2 | 0.00% |
| Company/Platform | 2 | 0.00% |
| non-human control | 2 | 0.00% |
| Airline Passengers | 2 | 0.00% |
| Energy | 2 | 0.00% |
| effectiveDate | 2 | 0.00% |
| TV Network | 2 | 0.00% |
| computer vulnerability | 2 | 0.00% |
| Family Members | 2 | 0.00% |
| audio content | 2 | 0.00% |
| guidance | 2 | 0.00% |
| employee | 2 | 0.00% |
| perspective | 2 | 0.00% |
| Product/Brand Name | 2 | 0.00% |
| play equipment | 2 | 0.00% |
| fishing rig/construction | 2 | 0.00% |
| sports venue | 2 | 0.00% |
| data source | 2 | 0.00% |
| Holiday/Tradition | 2 | 0.00% |
| governmental entity | 2 | 0.00% |
| Bible verse | 2 | 0.00% |
| Sales Record | 2 | 0.00% |
| Job/Position | 2 | 0.00% |
| Written communication | 2 | 0.00% |
| financial organization | 2 | 0.00% |
| installedDate | 2 | 0.00% |
| costume | 2 | 0.00% |
| error message | 2 | 0.00% |
| error type | 2 | 0.00% |
| Career Path | 2 | 0.00% |
| Entrepreneurship | 2 | 0.00% |
| Political event | 2 | 0.00% |
| Data Protection Law | 2 | 0.00% |
| Comedian | 2 | 0.00% |
| Garden | 2 | 0.00% |
| emotional symptom | 2 | 0.00% |
| product color | 2 | 0.00% |
| Dating website | 2 | 0.00% |
| Religion/Philosophy | 2 | 0.00% |
| Casino Venue | 2 | 0.00% |
| Emotional Entity | 2 | 0.00% |
| group/team | 2 | 0.00% |
| paymentType | 2 | 0.00% |
| Business Growth | 2 | 0.00% |
| Self | 2 | 0.00% |
| flightNumber | 2 | 0.00% |
| Singer Name | 2 | 0.00% |
| disaster/event | 2 | 0.00% |
| Maintenance Item | 2 | 0.00% |
| Road Name | 2 | 0.00% |
| Music Style | 2 | 0.00% |
| Art Form | 2 | 0.00% |
| attraction | 2 | 0.00% |
| game objects | 2 | 0.00% |
| Talent/Skill | 2 | 0.00% |
| home furnishings | 2 | 0.00% |
| Organization/Community | 2 | 0.00% |
| Ticketing Company | 2 | 0.00% |
| staff | 2 | 0.00% |
| protein family | 2 | 0.00% |
| Home Improvement Product | 2 | 0.00% |
| Online Retailer | 2 | 0.00% |
| financial document entity | 2 | 0.00% |
| attachmentName | 2 | 0.00% |
| attachmentType | 2 | 0.00% |
| attachmentUrl | 2 | 0.00% |
| humans | 2 | 0.00% |
| Supply | 2 | 0.00% |
| Military branch | 2 | 0.00% |
| Entity Type: Publication/Series | 2 | 0.00% |
| negative thoughts/ emotions | 2 | 0.00% |
| Study Material | 2 | 0.00% |
| Apparel Item | 2 | 0.00% |
| assistance/support | 2 | 0.00% |
| beauty treatment | 2 | 0.00% |
| Religious Role | 2 | 0.00% |
| plant structure | 2 | 0.00% |
| Exam Section | 2 | 0.00% |
| Real Estate Listing | 2 | 0.00% |
| Technology/Device | 2 | 0.00% |
| Golf Club | 2 | 0.00% |
| second person pronoun | 2 | 0.00% |
| Commerce | 2 | 0.00% |
| Religious gathering | 2 | 0.00% |
| Authors | 2 | 0.00% |
| person/organization | 2 | 0.00% |
| Time of Day | 2 | 0.00% |
| program_code | 2 | 0.00% |
| response | 2 | 0.00% |
| turbine component | 2 | 0.00% |
| Fashion event | 2 | 0.00% |
| asteroid.discovery.observatory.location.country | 2 | 0.00% |
| asteroid.named_after.name | 2 | 0.00% |
| low | 2 | 0.00% |
| high | 2 | 0.00% |
| receiptNumber | 2 | 0.00% |
| referenceNumber | 2 | 0.00% |
| Website Name | 2 | 0.00% |
| orbital_characteristics.semi_major_axis.unit | 2 | 0.00% |
| Texture | 2 | 0.00% |
| Architectural style | 2 | 0.00% |
| Dwelling Place | 2 | 0.00% |
| authors: List of physicians who developed the guideline | 2 | 0.00% |
| Personal computer | 2 | 0.00% |
| callId | 2 | 0.00% |
| callDate | 2 | 0.00% |
| Event/Entertainment | 2 | 0.00% |
| Personal name | 2 | 0.00% |
| event/program | 2 | 0.00% |
| Lubricant | 2 | 0.00% |
| Storage space | 2 | 0.00% |
| Cultural practice | 2 | 0.00% |
| wine event | 2 | 0.00% |
| Notification | 2 | 0.00% |
| Online presence | 2 | 0.00% |
| Service Option | 2 | 0.00% |
| sector | 2 | 0.00% |
| physical force | 2 | 0.00% |
| Retail | 2 | 0.00% |
| Recipe | 2 | 0.00% |
| Automobile | 2 | 0.00% |
| Skådespelare | 2 | 0.00% |
| poem.stanzas.1.lines | 2 | 0.00% |
| poem.stanzas.0.lines | 2 | 0.00% |
| poem.stanzas.2.lines | 2 | 0.00% |
| general | 2 | 0.00% |
| home design | 2 | 0.00% |
| customer support | 2 | 0.00% |
| recognition | 2 | 0.00% |
| Sports enthusiasts | 2 | 0.00% |
| artistic talent | 2 | 0.00% |
| marketing term | 2 | 0.00% |
| Development | 2 | 0.00% |
| stage element | 2 | 0.00% |
| object property | 2 | 0.00% |
| Shoe Type | 2 | 0.00% |
| Law Enforcement Agency | 2 | 0.00% |
| Human Being | 2 | 0.00% |
| editing technique | 2 | 0.00% |
| Clothing Pattern | 2 | 0.00% |
| Location/Service | 2 | 0.00% |
| options | 2 | 0.00% |
| system component | 2 | 0.00% |
| urban area | 2 | 0.00% |
| connector | 2 | 0.00% |
| passengerNumber | 2 | 0.00% |
| trainClass | 2 | 0.00% |
| seatNumber | 2 | 0.00% |
| seatType | 2 | 0.00% |
| term of endearment | 2 | 0.00% |
| Page Range | 2 | 0.00% |
| exercise unit | 2 | 0.00% |
| activity/sport | 2 | 0.00% |
| Career Achievement | 2 | 0.00% |
| Numerical Value | 2 | 0.00% |
| product accessory | 2 | 0.00% |
| product variant | 2 | 0.00% |
| recovery | 2 | 0.00% |
| Investment blogger | 2 | 0.00% |
| Encryption technology | 2 | 0.00% |
| Business Unit | 2 | 0.00% |
| Raw Material | 2 | 0.00% |
| bee colonies | 2 | 0.00% |
| bee breeds | 2 | 0.00% |
| Passion/Interest | 2 | 0.00% |
| Weight Capacity | 2 | 0.00% |
| Tournament | 2 | 0.00% |
| economic sector | 2 | 0.00% |
| Exclamation | 2 | 0.00% |
| Game mode | 2 | 0.00% |
| Color Scheme | 2 | 0.00% |
| Location/Type | 2 | 0.00% |
| Dairy Product | 2 | 0.00% |
| road/highway | 2 | 0.00% |
| Project/Initiative | 2 | 0.00% |
| species_name | 2 | 0.00% |
| family.name | 2 | 0.00% |
| orchestra | 2 | 0.00% |
| Email Service | 2 | 0.00% |
| Demographic Entity | 2 | 0.00% |
| Profession/Job Title | 2 | 0.00% |
| image resolution | 2 | 0.00% |
| patient_id: Unique identifier for the patient | 2 | 0.00% |
| sales promotion | 2 | 0.00% |
| field | 2 | 0.00% |
| activity/transaction | 2 | 0.00% |
| Dataset | 2 | 0.00% |
| service entity | 2 | 0.00% |
| Learning materials | 2 | 0.00% |
| measurement quantity | 2 | 0.00% |
| materials | 2 | 0.00% |
| Surface condition | 2 | 0.00% |
| assessment | 2 | 0.00% |
| Product category | 2 | 0.00% |
| program_name: Title of the training program enrolled in | 2 | 0.00% |
| issueId | 2 | 0.00% |
| 3D printer model | 2 | 0.00% |
| Ingredient/Component | 2 | 0.00% |
| Academic Achievement | 2 | 0.00% |
| Life Transition | 2 | 0.00% |
| theme | 2 | 0.00% |
| Music Performance | 2 | 0.00% |
| User Type | 2 | 0.00% |
| actionDueDate | 2 | 0.00% |
| actionStatus | 2 | 0.00% |
| actionResponsible | 2 | 0.00% |
| actionNotes | 2 | 0.00% |
| documentName | 2 | 0.00% |
| documentType | 2 | 0.00% |
| documentDate | 2 | 0.00% |
| documentRevision | 2 | 0.00% |
| documentLocation | 2 | 0.00% |
| vendor | 2 | 0.00% |
| business partners | 2 | 0.00% |
| tutorial | 2 | 0.00% |
| Mutant | 2 | 0.00% |
| Dish name | 2 | 0.00% |
| Organization/Initiative | 2 | 0.00% |
| evidence | 2 | 0.00% |
| Rocket | 2 | 0.00% |
| Satellite | 2 | 0.00% |
| Biological Property | 2 | 0.00% |
| Publication Size | 2 | 0.00% |
| Date/Day of the week | 2 | 0.00% |
| educational event | 2 | 0.00% |
| educator | 2 | 0.00% |
| Social Relationship | 2 | 0.00% |
| Sports Club | 2 | 0.00% |
| sample | 2 | 0.00% |
| feedbackRating | 2 | 0.00% |
| feedbackStatus | 2 | 0.00% |
| cultural institutions | 2 | 0.00% |
| cultural heritage | 2 | 0.00% |
| Hull Discovery Time | 2 | 0.00% |
| Hull Location | 2 | 0.00% |
| assignment | 2 | 0.00% |
| living arrangement | 2 | 0.00% |
| Facial Feature | 2 | 0.00% |
| customer feedback | 2 | 0.00% |
| Stock Market Index | 2 | 0.00% |
| technology/product | 2 | 0.00% |
| Term | 2 | 0.00% |
| bicycle part | 2 | 0.00% |
| Kangaroo | 2 | 0.00% |
| totalConsumption | 2 | 0.00% |
| Pokemon creature | 2 | 0.00% |
| Quality Level | 2 | 0.00% |
| Customization | 2 | 0.00% |
| website/blog | 2 | 0.00% |
| Program fees | 2 | 0.00% |
| academic work | 2 | 0.00% |
| Numeric Quantity | 2 | 0.00% |
| beauty | 2 | 0.00% |
| work schedule | 2 | 0.00% |
| molecule | 2 | 0.00% |
| uncertainty | 2 | 0.00% |
| Consumer Behavior | 2 | 0.00% |
| Physical activity | 2 | 0.00% |
| Superhero | 2 | 0.00% |
| Geographic Region | 2 | 0.00% |
| Location Type | 2 | 0.00% |
| Software Update | 2 | 0.00% |
| quote | 2 | 0.00% |
| Characters | 2 | 0.00% |
| Situation/Condition | 2 | 0.00% |
| Design Service | 2 | 0.00% |
| batchNumber | 2 | 0.00% |
| substance/product | 2 | 0.00% |
| orderId | 2 | 0.00% |
| None <> | 2 | 0.00% |
| Academic Document | 2 | 0.00% |
| mistreatment | 2 | 0.00% |
| Business Term | 2 | 0.00% |
| Measurement/Quantity | 2 | 0.00% |
| breathing exercise | 2 | 0.00% |
| geographical region | 2 | 0.00% |
| sports ranking | 2 | 0.00% |
| Behavior/Action | 2 | 0.00% |
| software system | 2 | 0.00% |
| Legal Process | 2 | 0.00% |
| functional space | 2 | 0.00% |
| StreetAddress | 2 | 0.00% |
| TelephoneNumber | 2 | 0.00% |
| LanguageCode | 2 | 0.00% |
| q1revenueguidance | 2 | 0.00% |
| Wine brand | 2 | 0.00% |
| Property Management | 2 | 0.00% |
| Delivery Service | 2 | 0.00% |
| Bill | 2 | 0.00% |
| travel reservation | 2 | 0.00% |
| Sport Event | 2 | 0.00% |
| Error Type | 2 | 0.00% |
| Boat | 2 | 0.00% |
| Time Management | 2 | 0.00% |
| Shipping | 2 | 0.00% |
| Journalist | 2 | 0.00% |
| musical aspect | 2 | 0.00% |
| musical technique | 2 | 0.00% |
| Settlement | 2 | 0.00% |
| Financial Status | 2 | 0.00% |
| computer processor | 2 | 0.00% |
| Product Line Name | 2 | 0.00% |
| negative attribute | 2 | 0.00% |
| Director/Producer | 2 | 0.00% |
| Business/Establishment | 2 | 0.00% |
| Art Venue | 2 | 0.00% |
| Object/Personal Item | 2 | 0.00% |
| Product/Category | 2 | 0.00% |
| Dog Name | 2 | 0.00% |
| Maintenance | 2 | 0.00% |
| Determiner | 2 | 0.00% |
| Modeling technique | 2 | 0.00% |
| document template | 2 | 0.00% |
| opposing force | 2 | 0.00% |
| Business/Location | 2 | 0.00% |
| Household Item | 2 | 0.00% |
| Party Responsible for Pollution | 2 | 0.00% |
| Subsidiary | 2 | 0.00% |
| album name | 2 | 0.00% |
| circumstances | 2 | 0.00% |
| utensil/material | 2 | 0.00% |
| proposal | 2 | 0.00% |
| limitation | 2 | 0.00% |
| Payment method | 2 | 0.00% |
| Culinary ingredient | 2 | 0.00% |
| Objective | 2 | 0.00% |
| label/nickname | 2 | 0.00% |
| Watch | 2 | 0.00% |
| Geographical area | 2 | 0.00% |
| shares | 2 | 0.00% |
| likes | 2 | 0.00% |
| creator | 2 | 0.00% |
| Residential Development | 2 | 0.00% |
| texture issue | 2 | 0.00% |
| preparation method | 2 | 0.00% |
| camera equipment | 2 | 0.00% |
| digital content | 2 | 0.00% |
| mistake/error | 2 | 0.00% |
| roofing material | 2 | 0.00% |
| Location/Country | 2 | 0.00% |
| Tour Activity | 2 | 0.00% |
| social validation | 2 | 0.00% |
| credentials | 2 | 0.00% |
| Instructions | 2 | 0.00% |
| function/action | 2 | 0.00% |
| efficiency | 2 | 0.00% |
| Computing System | 2 | 0.00% |
| Performance Issue | 2 | 0.00% |
| System Configuration | 2 | 0.00% |
| Senator | 2 | 0.00% |
| Industry/Category | 2 | 0.00% |
| Park Name | 2 | 0.00% |
| Audio component | 2 | 0.00% |
| review_id: Unique identifier for the review entry | 2 | 0.00% |
| tire maintenance | 2 | 0.00% |
| candidate_name | 2 | 0.00% |
| career_summary.roles | 2 | 0.00% |
| Server | 2 | 0.00% |
| Door handle type | 2 | 0.00% |
| Location/Point of Interest | 2 | 0.00% |
| Stress factor | 2 | 0.00% |
| planning | 2 | 0.00% |
| visual entity | 2 | 0.00% |
| construction method | 2 | 0.00% |
| Branch Location | 2 | 0.00% |
| Unit of measure | 2 | 0.00% |
| Medical professional | 2 | 0.00% |
| Boxer name | 2 | 0.00% |
| Military equipment | 2 | 0.00% |
| Political party | 2 | 0.00% |
| Electoral system | 2 | 0.00% |
| Political position | 2 | 0.00% |
| learning resources | 2 | 0.00% |
| Opinion/Action | 2 | 0.00% |
| profession/group of people | 2 | 0.00% |
| asteroid.name_latin | 2 | 0.00% |
| Activity entity | 2 | 0.00% |
| Car Audio Equipment | 2 | 0.00% |
| content_type | 2 | 0.00% |
| enemy | 2 | 0.00% |
| Political Institution | 2 | 0.00% |
| employeeId | 2 | 0.00% |
| jobTitle | 2 | 0.00% |
| Event Attendees | 2 | 0.00% |
| abbreviation | 2 | 0.00% |
| Process/Methodology | 2 | 0.00% |
| chemical reaction | 2 | 0.00% |
| Difficulty | 2 | 0.00% |
| login.steps | 2 | 0.00% |
| login.data_required | 2 | 0.00% |
| goal/purpose | 2 | 0.00% |
| Tool/Instrument | 2 | 0.00% |
| mental attribute | 2 | 0.00% |
| Atmosphere | 2 | 0.00% |
| to_club | 2 | 0.00% |
| type of energy | 2 | 0.00% |
| communication platform | 2 | 0.00% |
| medical entity | 2 | 0.00% |
| Graphics Card | 2 | 0.00% |
| mutation | 2 | 0.00% |
| Crafting Material | 2 | 0.00% |
| data summary | 2 | 0.00% |
| Networking | 2 | 0.00% |
| Music band/group | 2 | 0.00% |
| discovery.observatory.location.city | 2 | 0.00% |
| precipitation | 2 | 0.00% |
| Food ingredients | 2 | 0.00% |
| Game Character | 2 | 0.00% |
| Pricing strategy | 2 | 0.00% |
| activity type | 2 | 0.00% |
| medicationId | 2 | 0.00% |
| Work of art | 2 | 0.00% |
| search engine | 2 | 0.00% |
| productionDate | 2 | 0.00% |
| Project/Task | 2 | 0.00% |
| Royal Family Member | 2 | 0.00% |
| DVD title | 2 | 0.00% |
| Embellishment | 2 | 0.00% |
| dish component | 2 | 0.00% |
| Character trait | 2 | 0.00% |
| scientific method | 2 | 0.00% |
| research | 2 | 0.00% |
| Business Event | 2 | 0.00% |
| Template | 2 | 0.00% |
| transfer_id | 2 | 0.00% |
| driven | 2 | 0.00% |
| product warranty | 2 | 0.00% |
| TV channel | 2 | 0.00% |
| Software feature | 2 | 0.00% |
| deviceId | 2 | 0.00% |
| sensorType | 2 | 0.00% |
| Cryptocurrency exchange | 2 | 0.00% |
| bedding | 2 | 0.00% |
| insult | 2 | 0.00% |
| legal authority | 2 | 0.00% |
| project/task | 2 | 0.00% |
| Person/Entity | 2 | 0.00% |
| art product | 2 | 0.00% |
| physiological effect | 2 | 0.00% |
| Business/Brand Name | 2 | 0.00% |
| Music Producer | 2 | 0.00% |
| Financial Data | 2 | 0.00% |
| medical evaluation | 2 | 0.00% |
| medical appointment | 2 | 0.00% |
| Company Entity | 2 | 0.00% |
| Geographic Entity | 2 | 0.00% |
| bedding item | 2 | 0.00% |
| personal space | 2 | 0.00% |
| Degree Program | 2 | 0.00% |
| social gathering place | 2 | 0.00% |
| tool/instrument | 2 | 0.00% |
| TV Host | 2 | 0.00% |
| mental health issue | 2 | 0.00% |
| Law/Policy | 2 | 0.00% |
| Financial market | 2 | 0.00% |
| work location | 2 | 0.00% |
| job_title | 2 | 0.00% |
| game show | 2 | 0.00% |
| reality TV show | 2 | 0.00% |
| DocumentDate | 2 | 0.00% |
| DocumentName | 2 | 0.00% |
| DocumentType | 2 | 0.00% |
| DocumentUrl | 2 | 0.00% |
| gift card | 2 | 0.00% |
| Plant entity | 2 | 0.00% |
| river_details.countries | 2 | 0.00% |
| river_details.tributaries | 2 | 0.00% |
| Page Count | 2 | 0.00% |
| video quality | 2 | 0.00% |
| transportation mode | 2 | 0.00% |
| paranormal activity | 2 | 0.00% |
| Program/Event | 2 | 0.00% |
| Battery | 2 | 0.00% |
| Sports event/team title | 2 | 0.00% |
| Professional | 2 | 0.00% |
| Identification | 2 | 0.00% |
| Interior design feature | 2 | 0.00% |
| Education Course | 2 | 0.00% |
| Residential property | 2 | 0.00% |
| emailAddress | 2 | 0.00% |
| Visual Art | 2 | 0.00% |
| firewall | 2 | 0.00% |
| Commercial Entity | 2 | 0.00% |
| Coupon/Discount Code | 2 | 0.00% |
| Problematic situation | 2 | 0.00% |
| apparel item | 2 | 0.00% |
| incident.affected_locations | 2 | 0.00% |
| incident.additional_acts | 2 | 0.00% |
| Media type | 2 | 0.00% |
| character trait | 2 | 0.00% |
| Mobile Learning Resource | 2 | 0.00% |
| Hockey Player | 2 | 0.00% |
| food preparation method | 2 | 0.00% |
| home renovation | 2 | 0.00% |
| State/Action | 2 | 0.00% |
| Internet Protocol version | 2 | 0.00% |
| GPS technology | 2 | 0.00% |
| profession/service | 2 | 0.00% |
| Quality/Opinion | 2 | 0.00% |
| Emotional Experience | 2 | 0.00% |
| healthcare management | 2 | 0.00% |
| Medical field | 2 | 0.00% |
| Academic achievements | 2 | 0.00% |
| registration page | 2 | 0.00% |
| login information | 2 | 0.00% |
| Performance Level | 2 | 0.00% |
| Racing regulations | 2 | 0.00% |
| human infant | 2 | 0.00% |
| customerName | 2 | 0.00% |
| Document/Template | 2 | 0.00% |
| event stationery | 2 | 0.00% |
| Vacation rental platform | 2 | 0.00% |
| event attendees | 2 | 0.00% |
| tangible object | 2 | 0.00% |
| Rating | 2 | 0.00% |
| itemNumber | 2 | 0.00% |
| Yoga Practice | 2 | 0.00% |
| notification | 2 | 0.00% |
| Color attribute | 2 | 0.00% |
| mobileNumber | 2 | 0.00% |
| alternateMobileNumber | 2 | 0.00% |
| Attorney | 2 | 0.00% |
| Music subgenre | 2 | 0.00% |
| Problematic Entity | 2 | 0.00% |
| additionalDocuments | 2 | 0.00% |
| Soccer Team | 2 | 0.00% |
| Flora | 2 | 0.00% |
| restaurant.cuisine | 2 | 0.00% |
| ship component | 2 | 0.00% |
| monument | 2 | 0.00% |
| shaving tool feature | 2 | 0.00% |
| reasonForVisit | 2 | 0.00% |
| conditions | 2 | 0.00% |
| Hotel location | 2 | 0.00% |
| Performance Evaluation | 2 | 0.00% |
| Political Issue | 2 | 0.00% |
| Business/Shop | 2 | 0.00% |
| marginalized writers | 2 | 0.00% |
| creative output | 2 | 0.00% |
| Time Zone | 2 | 0.00% |
| Technology/Engineering | 2 | 0.00% |
| Demographic/Group | 2 | 0.00% |
| Award ceremony | 2 | 0.00% |
| study_id | 2 | 0.00% |
| posterUrl | 2 | 0.00% |
| imageUrl | 2 | 0.00% |
| remedy | 2 | 0.00% |
| Behavioral pattern | 2 | 0.00% |
| Book volume | 2 | 0.00% |
| low-quality content | 2 | 0.00% |
| Business Activity | 2 | 0.00% |
| thought process | 2 | 0.00% |
| Football Position | 2 | 0.00% |
| fees | 2 | 0.00% |
| Yarn Brand | 2 | 0.00% |
| Technology/Computing | 2 | 0.00% |
| Commission | 2 | 0.00% |
| religious text | 2 | 0.00% |
| Identity Verification Service | 2 | 0.00% |
| Insects/Animals | 2 | 0.00% |
| shoe feature | 2 | 0.00% |
| Retail establishment | 2 | 0.00% |
| Resource scarcity | 2 | 0.00% |
| Resource measurement | 2 | 0.00% |
| Fuel Efficiency | 2 | 0.00% |
| record | 2 | 0.00% |
| Orchestra | 2 | 0.00% |
| Contract | 2 | 0.00% |
| Historical era | 2 | 0.00% |
| comment | 2 | 0.00% |
| Shopping Center | 2 | 0.00% |
| Animal Feature | 2 | 0.00% |
| event management | 2 | 0.00% |
| domain | 2 | 0.00% |
| subjectAlternativeNames | 2 | 0.00% |
| legal issue | 2 | 0.00% |
| personal interest | 2 | 0.00% |
| Gift Card | 2 | 0.00% |
| Sports Play | 2 | 0.00% |
| Historical Group | 2 | 0.00% |
| Military Personnel | 2 | 0.00% |
| job benefit | 2 | 0.00% |
| fashion brand | 2 | 0.00% |
| Legislative Act | 2 | 0.00% |
| research project | 2 | 0.00% |
| Character Class Attribute | 2 | 0.00% |
| computer | 2 | 0.00% |
| customer_address | 2 | 0.00% |
| Celebration Event | 2 | 0.00% |
| URL resource | 2 | 0.00% |
| Trade goods | 2 | 0.00% |
| Philosophy | 2 | 0.00% |
| watercraft | 2 | 0.00% |
| Ethnicity/Culture | 2 | 0.00% |
| baby feeding product | 2 | 0.00% |
| Illustrator | 2 | 0.00% |
| Understanding | 2 | 0.00% |
| social class | 2 | 0.00% |
| gardening material | 2 | 0.00% |
| Currency Value | 2 | 0.00% |
| Directional term | 2 | 0.00% |
| non-profit organization | 2 | 0.00% |
| Physical location | 2 | 0.00% |
| Business network | 2 | 0.00% |
| Business group | 2 | 0.00% |
| Access | 2 | 0.00% |
| entertainment venue | 2 | 0.00% |
| Dish/Ingredient | 2 | 0.00% |
| human trait | 2 | 0.00% |
| human relationship | 2 | 0.00% |
| human group | 2 | 0.00% |
| facility type | 2 | 0.00% |
| anatomical structures | 2 | 0.00% |
| living place | 2 | 0.00% |
| accident | 2 | 0.00% |
| procedureId | 2 | 0.00% |
| calibrationDate | 2 | 0.00% |
| Software License | 2 | 0.00% |
| isbn | 2 | 0.00% |
| Learning opportunity | 2 | 0.00% |
| Version Number | 2 | 0.00% |
| Technology and Software | 2 | 0.00% |
| Sports Outcome | 2 | 0.00% |
| Christmas Tree Decoration | 2 | 0.00% |
| Rugby team | 2 | 0.00% |
| scientific measurement | 2 | 0.00% |
| Authority | 2 | 0.00% |
| vegetation | 2 | 0.00% |
| Beauty Routine | 2 | 0.00% |
| orbital_characteristics.semi_major_axis | 2 | 0.00% |
| orbital_characteristics.inclination | 2 | 0.00% |
| external_links | 2 | 0.00% |
| testing standard | 2 | 0.00% |
| Funding source | 2 | 0.00% |
| Pet Door Model | 2 | 0.00% |
| Employment | 2 | 0.00% |
| Object/Container | 2 | 0.00% |
| CPU | 2 | 0.00% |
| Student-Athlete | 2 | 0.00% |
| Inventory | 2 | 0.00% |
| aromatherapy product | 2 | 0.00% |
| Tea type | 2 | 0.00% |
| Tea brewing vessel | 2 | 0.00% |
| navigation | 2 | 0.00% |
| Geopolitical entities | 2 | 0.00% |
| End_Date | 2 | 0.00% |
| Service Providers | 2 | 0.00% |
| open_issues | 2 | 0.00% |
| flora entity | 2 | 0.00% |
| reportId | 2 | 0.00% |
| inspectionCriteria | 2 | 0.00% |
| Mission | 2 | 0.00% |
| winemaking process | 2 | 0.00% |
| group/organization | 2 | 0.00% |
| cultural feature | 2 | 0.00% |
| online platforms | 2 | 0.00% |
| Logo | 2 | 0.00% |
| boat material | 2 | 0.00% |
| snack food | 2 | 0.00% |
| Training Institution | 2 | 0.00% |
| satisfaction | 2 | 0.00% |
| Hotel rating | 2 | 0.00% |
| year_in_era | 2 | 0.00% |
| Moving Average | 2 | 0.00% |
| Design style | 2 | 0.00% |
| physics | 2 | 0.00% |
| Curator | 2 | 0.00% |
| storeName | 2 | 0.00% |
| team/group | 2 | 0.00% |
| Form | 2 | 0.00% |
| asteroid.discovery.discoverer.name | 2 | 0.00% |
| Quality of wine | 2 | 0.00% |
| Gaming gear | 2 | 0.00% |
| Role/Identity | 2 | 0.00% |
| Car accessories | 2 | 0.00% |
| fundamental knowledge | 2 | 0.00% |
| network data | 2 | 0.00% |
| Eating Establishment | 2 | 0.00% |
| Customer service | 2 | 0.00% |
| Body of Water | 2 | 0.00% |
| information retrieval | 2 | 0.00% |
| Landscape Feature | 2 | 0.00% |
| grape variety | 2 | 0.00% |
| Date range | 2 | 0.00% |
| Individual Person | 2 | 0.00% |
| startTime | 2 | 0.00% |
| endTime | 2 | 0.00% |
| machineName | 2 | 0.00% |
| machineStatus | 2 | 0.00% |
| productionCapacity | 2 | 0.00% |
| actualProduction | 2 | 0.00% |
| productionRate | 2 | 0.00% |
| Food Category | 2 | 0.00% |
| Camera Model | 2 | 0.00% |
| part of garment | 2 | 0.00% |
| Collection Type | 2 | 0.00% |
| Ecosystem | 2 | 0.00% |
| Menu Category | 2 | 0.00% |
| Wiring diagram | 2 | 0.00% |
| organization.partners | 2 | 0.00% |
| scholarship_program.conferences.purpose | 2 | 0.00% |
| measurementStandards | 2 | 0.00% |
| referenceMaterials | 2 | 0.00% |
| street1 | 2 | 0.00% |
| TV show episode | 2 | 0.00% |
| unknown entity | 2 | 0.00% |
| Furniture Collection | 2 | 0.00% |
| numGuests | 2 | 0.00% |
| Holiday Decoration | 2 | 0.00% |
| internet speed | 2 | 0.00% |
| Podcast Name | 2 | 0.00% |
| Training Program | 2 | 0.00% |
| Electric Kettle | 2 | 0.00% |
| Community/Forum | 2 | 0.00% |
| herb | 2 | 0.00% |
| book design | 2 | 0.00% |
| medical technology | 2 | 0.00% |
| medical tool | 2 | 0.00% |
| type of water | 2 | 0.00% |
| baby food | 2 | 0.00% |
| stock index | 2 | 0.00% |
| finance term | 2 | 0.00% |
| Clothing and accessories | 2 | 0.00% |
| service/cost | 2 | 0.00% |
| language group | 2 | 0.00% |
| organization/enterprise | 2 | 0.00% |
| agreement | 2 | 0.00% |
| Skin Disease | 2 | 0.00% |
| Circus company | 2 | 0.00% |
| fitness class | 2 | 0.00% |
| outbreak_event.public_health_recommendations.target_populations | 2 | 0.00% |
| outbreak_event.investigation.methods | 2 | 0.00% |
| fashion event | 2 | 0.00% |
| Lighting fixture | 2 | 0.00% |
| Window treatment | 2 | 0.00% |
| Demonym | 2 | 0.00% |
| benchmark | 2 | 0.00% |
| Healthcare service | 2 | 0.00% |
| protecting_group.chemical_properties | 2 | 0.00% |
| Activities | 2 | 0.00% |
| Family members | 2 | 0.00% |
| Vehicle Brand | 2 | 0.00% |
| company size | 2 | 0.00% |
| Sneaker Model | 2 | 0.00% |
| Spell Speed | 2 | 0.00% |
| Mixture | 2 | 0.00% |
| Penalty | 2 | 0.00% |
| Color/Finish | 2 | 0.00% |
| Chemical Ingredient | 2 | 0.00% |
| Rating System | 2 | 0.00% |
| financial event | 2 | 0.00% |
| scientific experiment | 2 | 0.00% |
| writing activity | 2 | 0.00% |
| Numerical value | 2 | 0.00% |
| timeFrame | 2 | 0.00% |
| Gender Identity | 2 | 0.00% |
| price changes | 2 | 0.00% |
| Software Class | 2 | 0.00% |
| Health Behavior | 2 | 0.00% |
| Travel document | 2 | 0.00% |
| disease.subspecies.1.associated_diseases | 2 | 0.00% |
| disease.pathogenesis.transmission | 2 | 0.00% |
| Emotional connection | 2 | 0.00% |
| Medical professionals | 2 | 0.00% |
| Database function | 2 | 0.00% |
| physical harm | 2 | 0.00% |
| hotel_experience.suites.0.amenities | 2 | 0.00% |
| Band member | 2 | 0.00% |
| Bathroom | 2 | 0.00% |
| Educational Resource | 2 | 0.00% |
| Emotional Response | 2 | 0.00% |
| Psychological phenomenon | 2 | 0.00% |
| Physical Component | 2 | 0.00% |
| inspiration | 2 | 0.00% |
| venue feature | 2 | 0.00% |
| media title | 2 | 0.00% |
| commercial promotion | 2 | 0.00% |
| footwear brand | 2 | 0.00% |
| communication service | 2 | 0.00% |
| film title | 2 | 0.00% |
| Plant Part Damage | 2 | 0.00% |
| Ticket Outlet | 2 | 0.00% |
| language writing system | 2 | 0.00% |
| Musical Artist | 2 | 0.00% |
| asteroid.discovery.observatory.name | 2 | 0.00% |
| Software Release | 2 | 0.00% |
| actions | 2 | 0.00% |
| distributor_information.benefits | 2 | 0.00% |
| Hair Color Type | 2 | 0.00% |
| Employment Opportunity | 2 | 0.00% |
| regulations | 2 | 0.00% |
| Vehicle/Type | 2 | 0.00% |
| physical storage device | 2 | 0.00% |
| military technology | 2 | 0.00% |
| classification | 2 | 0.00% |
| Hosting Service | 2 | 0.00% |
| parties | 2 | 0.00% |
| CSS Property | 2 | 0.00% |
| Court System | 2 | 0.00% |
| Terrain | 2 | 0.00% |
| Football Strategy | 2 | 0.00% |
| Historical Object | 2 | 0.00% |
| Game Object | 2 | 0.00% |
| Building Modification | 2 | 0.00% |
| treatmentPlan | 2 | 0.00% |
| Economic indicator | 2 | 0.00% |
| Accident | 2 | 0.00% |
| fishing lure | 2 | 0.00% |
| Physical ability | 2 | 0.00% |
| type of restaurant | 2 | 0.00% |
| Suburb | 2 | 0.00% |
| dobj | 2 | 0.00% |
| prep | 2 | 0.00% |
| Product/Equipment | 2 | 0.00% |
| Home Interiors | 2 | 0.00% |
| yacht builder | 2 | 0.00% |
| app_version | 2 | 0.00% |
| bundle_id | 2 | 0.00% |
| device_model | 2 | 0.00% |
| device_os_version | 2 | 0.00% |
| device_locale | 2 | 0.00% |
| device_manufacturer | 2 | 0.00% |
| exception_type | 2 | 0.00% |
| exception_message | 2 | 0.00% |
| additional_info | 2 | 0.00% |
| Mythical Place | 2 | 0.00% |
| Contact Lens Brand | 2 | 0.00% |
| article.symptoms | 2 | 0.00% |
| Dungeon Level | 2 | 0.00% |
| social media feature | 2 | 0.00% |
| Backup Service | 2 | 0.00% |
| market behavior | 2 | 0.00% |
| facilities | 2 | 0.00% |
| academic field | 2 | 0.00% |
| reviewContent | 2 | 0.00% |
| Damaged Crops & Livestock | 2 | 0.00% |
| Engine | 2 | 0.00% |
| Account Type | 2 | 0.00% |
| Religious institution | 2 | 0.00% |
| business goal | 2 | 0.00% |
| rock type | 2 | 0.00% |
| tax form | 2 | 0.00% |
| Fuel efficiency | 2 | 0.00% |
| academic position | 2 | 0.00% |
| Realm/Level | 2 | 0.00% |
| mythical entity | 2 | 0.00% |
| tuning modification | 2 | 0.00% |
| racing driver | 2 | 0.00% |
| Car Dealership | 2 | 0.00% |
| Meat Type | 2 | 0.00% |
| organizational feature | 2 | 0.00% |
| Biometric authentication technology | 2 | 0.00% |
| recreation | 2 | 0.00% |
| Opinion/Emotion | 2 | 0.00% |
| content_analysis.entry_structure.components | 2 | 0.00% |
| content_analysis.entry_structure.notes_types | 2 | 0.00% |
| Deceased Person | 2 | 0.00% |
| dough texture | 2 | 0.00% |
| Warning Device | 2 | 0.00% |
| terminationTerms | 2 | 0.00% |
| event name | 2 | 0.00% |
| Autonomous Region | 2 | 0.00% |
| Artifact | 2 | 0.00% |
| prop | 2 | 0.00% |
| reservationId | 2 | 0.00% |
| error.possible_causes | 2 | 0.00% |
| business practice | 2 | 0.00% |
| Shoe style | 2 | 0.00% |
| Home Appliance | 2 | 0.00% |
| Technology Service | 2 | 0.00% |
| Information Service | 2 | 0.00% |
| Tour | 2 | 0.00% |
| adhesive product | 2 | 0.00% |
| Hindu Deity | 2 | 0.00% |
| unit of weight | 2 | 0.00% |
| car specification | 2 | 0.00% |
| Discounts | 2 | 0.00% |
| Student Group | 2 | 0.00% |
| Business Relationship | 2 | 0.00% |
| tax law | 2 | 0.00% |
| Automobile part | 2 | 0.00% |
| Person/Author | 2 | 0.00% |
| payment system | 2 | 0.00% |
| bodily sensation | 2 | 0.00% |
| reportDate | 2 | 0.00% |
| materialDescription | 2 | 0.00% |
| quantityRejected | 2 | 0.00% |
| action/outcome | 2 | 0.00% |
| Motion | 2 | 0.00% |
| mapping schema type | 2 | 0.00% |
| Comic Strip | 2 | 0.00% |
| Catchphrase/Slogan | 2 | 0.00% |
| retail | 2 | 0.00% |
| Fertilizer quantity | 2 | 0.00% |
| Album/Music Release | 2 | 0.00% |
| personal mindset | 2 | 0.00% |
| Legal authority | 2 | 0.00% |
| Entity type: Listing | 2 | 0.00% |
| Trait/Ability | 2 | 0.00% |
| Video game graphics | 2 | 0.00% |
| learning resource | 2 | 0.00% |
| Sports Figure | 2 | 0.00% |
| Technological advancement | 2 | 0.00% |
| Communication device | 2 | 0.00% |
| Technology/Operating System | 2 | 0.00% |
| Government Benefit | 2 | 0.00% |
| Support | 2 | 0.00% |
| play | 2 | 0.00% |
| HVAC system | 2 | 0.00% |
| theater | 2 | 0.00% |
| CD | 2 | 0.00% |
| NN | 2 | 0.00% |
| School Event | 2 | 0.00% |
| furniture entity | 2 | 0.00% |
| Adventure Location | 2 | 0.00% |
| elevator type | 2 | 0.00% |
| businessName | 2 | 0.00% |
| Bag | 2 | 0.00% |
| Property/Characteristic | 2 | 0.00% |
| sports event preview | 2 | 0.00% |
| Dish/Menu Item | 2 | 0.00% |
| bicycle component | 2 | 0.00% |
| Household | 2 | 0.00% |
| surface type | 2 | 0.00% |
| exercise/fitness technique | 2 | 0.00% |
| Energy Company | 2 | 0.00% |
| Pokemon | 2 | 0.00% |
| Sports competition | 2 | 0.00% |
| Religious Practice | 2 | 0.00% |
| Kitchen Cabinets | 2 | 0.00% |
| assessment attribute | 2 | 0.00% |
| assessment category | 2 | 0.00% |
| computer function | 2 | 0.00% |
| condition/group | 2 | 0.00% |
| heraldry | 2 | 0.00% |
| Network issue | 2 | 0.00% |
| type of housing | 2 | 0.00% |
| International Organization | 2 | 0.00% |
| System Attribute | 2 | 0.00% |
| size range | 2 | 0.00% |
| Staff housing | 2 | 0.00% |
| Social circle | 2 | 0.00% |
| Regional Group | 2 | 0.00% |
| exercise regimen | 2 | 0.00% |
| Skincare product/action | 2 | 0.00% |
| Clothing size | 2 | 0.00% |
| craft object | 2 | 0.00% |
| Sports Competition | 2 | 0.00% |
| Child Custody | 2 | 0.00% |
| Weather Action | 2 | 0.00% |
| toilet type | 2 | 0.00% |
| Product Volume | 2 | 0.00% |
| entities_involved | 2 | 0.00% |
| DOI | 2 | 0.00% |
| laundry load size | 2 | 0.00% |
| domain name | 2 | 0.00% |
| College/University | 2 | 0.00% |
| temperature range | 2 | 0.00% |
| Plant anatomy | 2 | 0.00% |
| Task List | 2 | 0.00% |
| Adverb/Preposition | 2 | 0.00% |
| Mood | 2 | 0.00% |
| Tsunamis | 2 | 0.00% |
| keywords | 2 | 0.00% |
| Clothing detail | 2 | 0.00% |
| jewelry material | 2 | 0.00% |
| Unhealthy habit | 2 | 0.00% |
| Car class | 2 | 0.00% |
| experience type | 2 | 0.00% |
| travel activity | 2 | 0.00% |
| network security service | 2 | 0.00% |
| Event/Conflict | 2 | 0.00% |
| Investment entities | 2 | 0.00% |
| hygiene activity | 2 | 0.00% |
| personal task | 2 | 0.00% |
| skin characteristic | 2 | 0.00% |
| Charity Organization | 2 | 0.00% |
| clock component | 2 | 0.00% |
| basketball team | 2 | 0.00% |
| access | 2 | 0.00% |
| citizen_corps.activities | 2 | 0.00% |
| start drills | 2 | 0.00% |
| emotional state/physical reaction | 2 | 0.00% |
| Offer/Promotion | 2 | 0.00% |
| Structure Type | 2 | 0.00% |
| Lounge room | 2 | 0.00% |
| modal verb | 2 | 0.00% |
| total_price | 2 | 0.00% |
| Life Event/Experience | 2 | 0.00% |
| Tool/Equipment | 2 | 0.00% |
| Experience/Activity | 2 | 0.00% |
| Football Team Fans | 2 | 0.00% |
| Geographical Location/Landmark | 2 | 0.00% |
| Activity/Job | 2 | 0.00% |
| ease | 2 | 0.00% |
| Anatomical structure | 2 | 0.00% |
| population group | 2 | 0.00% |
| Speech Disfluency | 2 | 0.00% |
| industry/sector | 2 | 0.00% |
| Prize Money | 2 | 0.00% |
| Food/Recipe | 2 | 0.00% |
| Character name | 2 | 0.00% |
| User interface state | 2 | 0.00% |
| Service Organization | 2 | 0.00% |
| Aircraft Limitation | 2 | 0.00% |
| instrument | 2 | 0.00% |
| Quality Standards | 2 | 0.00% |
| Health and Wellness | 2 | 0.00% |
| social expression | 2 | 0.00% |
| Business/Service Provider | 2 | 0.00% |
| Society Type | 2 | 0.00% |
| motion type | 2 | 0.00% |
| media collection | 2 | 0.00% |
| texts | 2 | 0.00% |
| equipmentName | 2 | 0.00% |
| watch model | 2 | 0.00% |
| cat color/markings | 2 | 0.00% |
| human age group | 2 | 0.00% |
| Video Games | 2 | 0.00% |
| streaming service | 2 | 0.00% |
| Country/Group | 2 | 0.00% |
| Price/Cost | 2 | 0.00% |
| Meadow types | 2 | 0.00% |
| signatories: List of officials who co-signed the treaty | 2 | 0.00% |
| religious title | 2 | 0.00% |
| pets | 2 | 0.00% |
| Song/Music | 2 | 0.00% |
| Tree Type | 2 | 0.00% |
| Reward/Benefit | 2 | 0.00% |
| experimental factor | 2 | 0.00% |
| art material | 2 | 0.00% |
| lab_order: List of lab tests ordered for the patient | 2 | 0.00% |
| partyID | 2 | 0.00% |
| clothing feature/design | 2 | 0.00% |
| Event name | 2 | 0.00% |
| Information/Content | 2 | 0.00% |
| archive | 2 | 0.00% |
| Security Exploit | 2 | 0.00% |
| Graphics library | 2 | 0.00% |
| file transfer action | 2 | 0.00% |
| Action Phrase | 2 | 0.00% |
| place of worship | 2 | 0.00% |
| game item | 2 | 0.00% |
| Agricultural policy program | 2 | 0.00% |
| Attribute/Property | 2 | 0.00% |
| Award/Recognition | 2 | 0.00% |
| Economic Impact | 2 | 0.00% |
| Biological group | 2 | 0.00% |
| dance music | 2 | 0.00% |
| Musical Element | 2 | 0.00% |
| real estate development | 2 | 0.00% |
| life insurance policy | 2 | 0.00% |
| Historical Building | 2 | 0.00% |
| Web Hosting Service | 2 | 0.00% |
| Chemical property | 2 | 0.00% |
| Pageant Title | 2 | 0.00% |
| Organization/Company | 2 | 0.00% |
| Hair Care Product | 2 | 0.00% |
| Testing Process | 2 | 0.00% |
| Pseudonym | 2 | 0.00% |
| Regional Cooperation Project | 2 | 0.00% |
| Centre/Institution | 2 | 0.00% |
| Personality Type | 2 | 0.00% |
| key_achievements | 2 | 0.00% |
| Aesthetic feature | 2 | 0.00% |
| Meal event | 2 | 0.00% |
| Last name | 2 | 0.00% |
| floral element | 2 | 0.00% |
| Technology/Feature | 2 | 0.00% |
| Testing Equipment | 2 | 0.00% |
| Road/Highway | 2 | 0.00% |
| gender group | 2 | 0.00% |
| relationship type | 2 | 0.00% |
| administrative task | 2 | 0.00% |
| insurance action | 2 | 0.00% |
| home décor | 2 | 0.00% |
| Horse Racing Event | 2 | 0.00% |
| item_id | 2 | 0.00% |
| plant type | 2 | 0.00% |
| Laundry service | 2 | 0.00% |
| Plant characteristic | 2 | 0.00% |
| religious monument | 2 | 0.00% |
| file naming convention | 2 | 0.00% |
| package | 2 | 0.00% |
| type of rice | 2 | 0.00% |
| Mythological figure | 2 | 0.00% |
| sponsoring organization | 2 | 0.00% |
| phases | 2 | 0.00% |
| Football team/club | 2 | 0.00% |
| classification approach | 2 | 0.00% |
| electronic cigarette | 1 | 0.00% |
| municipal buildings | 1 | 0.00% |
| Information type | 1 | 0.00% |
| Emotional support | 1 | 0.00% |
| review_code | 1 | 0.00% |
| book_title | 1 | 0.00% |
| Tape type | 1 | 0.00% |
| Audio level | 1 | 0.00% |
| interface option | 1 | 0.00% |
| publication_date.month | 1 | 0.00% |
| guidelines.adherence | 1 | 0.00% |
| publication_date.city | 1 | 0.00% |
| guidelines.moa_members.obligation | 1 | 0.00% |
| political organization | 1 | 0.00% |
| purchasePrice | 1 | 0.00% |
| storePurchased | 1 | 0.00% |
| warrantyPeriod | 1 | 0.00% |
| warrantyCoverage | 1 | 0.00% |
| warrantyExclusions | 1 | 0.00% |
| warrantyLimitations | 1 | 0.00% |
| createDate | 1 | 0.00% |
| updateDate | 1 | 0.00% |
| design elements | 1 | 0.00% |
| Intention/Action | 1 | 0.00% |
| examination | 1 | 0.00% |
| issue_description | 1 | 0.00% |
| resolution_timeline | 1 | 0.00% |
| Architectural Element | 1 | 0.00% |
| computer science | 1 | 0.00% |
| product/service example | 1 | 0.00% |
| experiences | 1 | 0.00% |
| Camera model | 1 | 0.00% |
| Camera line | 1 | 0.00% |
| Service/Offering | 1 | 0.00% |
| Streaming service | 1 | 0.00% |
| furniture element | 1 | 0.00% |
| performance_rating | 1 | 0.00% |
| salary_adjustment | 1 | 0.00% |
| supervisor | 1 | 0.00% |
| information request | 1 | 0.00% |
| nutritional_aspects.sucrose.daily_limit_percentage | 1 | 0.00% |
| nutritional_aspects.sucrose.daily_limit_grams | 1 | 0.00% |
| nutritional_aspects.carbohydrates.daily_percentage | 1 | 0.00% |
| nutritional_aspects.carbohydrates.importance | 1 | 0.00% |
| revenue: Total revenue in millions | 1 | 0.00% |
| year: Year of financial data | 1 | 0.00% |
| partner_year: Year of partner's financial data | 1 | 0.00% |
| partner_revenue: Partner's revenue in millions | 1 | 0.00% |
| company_id: Unique identifier for the company | 1 | 0.00% |
| partner_company: Name of the partner company | 1 | 0.00% |
| partner_location: City and state of the partner company | 1 | 0.00% |
| partner_status: Operational status of the partner company | 1 | 0.00% |
| status: Current operational status | 1 | 0.00% |
| location: City and state of the company | 1 | 0.00% |
| partnergrowthrate: Partner's growth rate percentage | 1 | 0.00% |
| Clinic/Center | 1 | 0.00% |
| personal endeavor | 1 | 0.00% |
| second person singular pronoun | 1 | 0.00% |
| method or manner | 1 | 0.00% |
| negative particle | 1 | 0.00% |
| match_outcome | 1 | 0.00% |
| key_event | 1 | 0.00% |
| final_score | 1 | 0.00% |
| firearm type | 1 | 0.00% |
| action item | 1 | 0.00% |
| location/feature | 1 | 0.00% |
| animal/pet | 1 | 0.00% |
| location: Physical address of the company | 1 | 0.00% |
| launch_date: Date the product was first released | 1 | 0.00% |
| name: Product name as marketed to consumers | 1 | 0.00% |
| TV Show/Entertainment | 1 | 0.00% |
| App name | 1 | 0.00% |
| religious affiliation | 1 | 0.00% |
| religious community | 1 | 0.00% |
| search_result_page.website_name | 1 | 0.00% |
| search_result_page.related_to | 1 | 0.00% |
| mode | 1 | 0.00% |
| data storage device | 1 | 0.00% |
| Diplomatic document | 1 | 0.00% |
| United Nations official | 1 | 0.00% |
| Pet Care | 1 | 0.00% |
| Pet Exercise | 1 | 0.00% |
| Veterinary Care | 1 | 0.00% |
| Pet Therapy | 1 | 0.00% |
| emotion/idea | 1 | 0.00% |
| action/idea | 1 | 0.00% |
| asteroid.description.discovery_details.location.country | 1 | 0.00% |
| asteroid.orbital_characteristics.inclination | 1 | 0.00% |
| asteroid.description.discovery_details.location.county | 1 | 0.00% |
| asteroid.description.discovery_details.location.state | 1 | 0.00% |
| asteroid.description.discovery_details.date | 1 | 0.00% |
| asteroid.description.orbital_period_years | 1 | 0.00% |
| asteroid.external_links.0.title | 1 | 0.00% |
| asteroid.orbital_characteristics.semi_major_axis | 1 | 0.00% |
| asteroid.discovery.discovery_rank | 1 | 0.00% |
| asteroid.orbital_characteristics.perihelion | 1 | 0.00% |
| asteroid.orbital_characteristics.eccentricity | 1 | 0.00% |
| Person/Philanthropist | 1 | 0.00% |
| feature/functionality | 1 | 0.00% |
| task or project | 1 | 0.00% |
| Ranking Entity | 1 | 0.00% |
| Geological entity | 1 | 0.00% |
| Wind direction | 1 | 0.00% |
| donation_id: Unique identifier for each donation entry | 1 | 0.00% |
| donor: Name of the individual making the donation | 1 | 0.00% |
| status: Current status of the donation transaction | 1 | 0.00% |
| club | 1 | 0.00% |
| Professional network | 1 | 0.00% |
| Market knowledge | 1 | 0.00% |
| Remote work | 1 | 0.00% |
| Exploration | 1 | 0.00% |
| filesystem | 1 | 0.00% |
| enrollment_date | 1 | 0.00% |
| physical_characteristics.observations.0.telescope.orbit_year | 1 | 0.00% |
| media platform | 1 | 0.00% |
| video content | 1 | 0.00% |
| real estate content | 1 | 0.00% |
| travel content | 1 | 0.00% |
| image manipulation | 1 | 0.00% |
| Production process | 1 | 0.00% |
| Car dealership | 1 | 0.00% |
| physical dimension | 1 | 0.00% |
| Category/Topic | 1 | 0.00% |
| Training Center | 1 | 0.00% |
| Review process | 1 | 0.00% |
| Publication type | 1 | 0.00% |
| division | 1 | 0.00% |
| tehsil | 1 | 0.00% |
| environmental action | 1 | 0.00% |
| art conservation | 1 | 0.00% |
| energy conservation | 1 | 0.00% |
| economic status | 1 | 0.00% |
| technology/service | 1 | 0.00% |
| Boat name | 1 | 0.00% |
| City name | 1 | 0.00% |
| patient_id: Unique identifier for each patient | 1 | 0.00% |
| discharge_date: Date patient was discharged from hospital | 1 | 0.00% |
| diagnosis: Medical condition diagnosed upon admission | 1 | 0.00% |
| treatment: Medical intervention or therapy administered | 1 | 0.00% |
| colour scheme | 1 | 0.00% |
| contract | 1 | 0.00% |
| Songs | 1 | 0.00% |
| Search Result | 1 | 0.00% |
| Food Type | 1 | 0.00% |
| Holiday decoration | 1 | 0.00% |
| Music Movements | 1 | 0.00% |
| Music Movement | 1 | 0.00% |
| Emotion/Quality | 1 | 0.00% |
| Music Element | 1 | 0.00% |
| Subscription Service | 1 | 0.00% |
| SMS Service | 1 | 0.00% |
| claim_date: Date the claim was filed with the insurer | 1 | 0.00% |
| claim_number: Identifier for the specific health claim submitted | 1 | 0.00% |
| billed_amount: Total amount billed for the service | 1 | 0.00% |
| diagnosis: ICD-10 code representing the medical condition | 1 | 0.00% |
| patient_id: Unique identifier for the patient filing the claim | 1 | 0.00% |
| state of food item | 1 | 0.00% |
| Chemicals | 1 | 0.00% |
| Alternatives | 1 | 0.00% |
| Issue Number | 1 | 0.00% |
| data partition | 1 | 0.00% |
| formatting command | 1 | 0.00% |
| data storage system | 1 | 0.00% |
| emergency_contact | 1 | 0.00% |
| issuing_company | 1 | 0.00% |
| Physical quality | 1 | 0.00% |
| Installation equipment | 1 | 0.00% |
| seasonal product | 1 | 0.00% |
| product purchase | 1 | 0.00% |
| Attribute/Measurement | 1 | 0.00% |
| Attribute/Identifier | 1 | 0.00% |
| Attribute/Status | 1 | 0.00% |
| technology/domain | 1 | 0.00% |
| marital status | 1 | 0.00% |
| Entertainment district | 1 | 0.00% |
| ratingDate | 1 | 0.00% |
| numberOfRatings | 1 | 0.00% |
| Internet connectivity status | 1 | 0.00% |
| Audio content | 1 | 0.00% |
| Sensory perception | 1 | 0.00% |
| Absence | 1 | 0.00% |
| building process | 1 | 0.00% |
| data filtering | 1 | 0.00% |
| data access | 1 | 0.00% |
| Change in Value | 1 | 0.00% |
| Trait/Issue | 1 | 0.00% |
| Organizational Unit | 1 | 0.00% |
| online discussion | 1 | 0.00% |
| enhancement | 1 | 0.00% |
| display | 1 | 0.00% |
| digital presence | 1 | 0.00% |
| website management | 1 | 0.00% |
| Legal proceeding | 1 | 0.00% |
| Legal official | 1 | 0.00% |
| project element | 1 | 0.00% |
| endeavor | 1 | 0.00% |
| outcomes.overall_survival_months.max | 1 | 0.00% |
| treatment_details.pulmonary_resection_types.0.count | 1 | 0.00% |
| patient_demographics.gender_distribution.women.count | 1 | 0.00% |
| pathological_findings.statistical_factors.0.factor | 1 | 0.00% |
| study_overview.aim | 1 | 0.00% |
| outcomes.mortality_by_nodal_involvement.0.location | 1 | 0.00% |
| pathological_findings.nodal_involvement_distribution.2.stage | 1 | 0.00% |
| methodology.study_period.end_year | 1 | 0.00% |
| treatment_details.pulmonary_resection_types.1.procedure | 1 | 0.00% |
| pathological_findings.statistical_factors.1.p_value | 1 | 0.00% |
| outcomes.survival_rates.one_year_percentage | 1 | 0.00% |
| treatment_details.radiotherapy.type | 1 | 0.00% |
| methodology.population.study_group_size | 1 | 0.00% |
| outcomes.overall_survival_months.average | 1 | 0.00% |
| outcomes.survival_rates.two_year_percentage | 1 | 0.00% |
| pathological_findings.nodal_involvement_distribution.2.patient_count | 1 | 0.00% |
| patient_demographics.gender_distribution.men.percentage | 1 | 0.00% |
| treatment_details.pulmonary_resection_types.0.procedure | 1 | 0.00% |
| study_overview.disease | 1 | 0.00% |
| outcomes.disease_free_survival_months_average | 1 | 0.00% |
| study_overview.condition_detail | 1 | 0.00% |
| outcomes.survival_rates.five_year_percentage | 1 | 0.00% |
| patient_demographics.gender_distribution.men.count | 1 | 0.00% |
| outcomes.overall_survival_months.min | 1 | 0.00% |
| treatment_details.pulmonary_resection_types.1.count | 1 | 0.00% |
| patient_demographics.age_stats.average | 1 | 0.00% |
| pathological_findings.nodal_involvement_distribution.1.stage | 1 | 0.00% |
| outcomes.mortality_by_nodal_involvement.0.survival_limit_months | 1 | 0.00% |
| Brand/Website | 1 | 0.00% |
| food festival | 1 | 0.00% |
| food preference | 1 | 0.00% |
| type of chocolate | 1 | 0.00% |
| Chapter quotes | 1 | 0.00% |
| Pinterest board | 1 | 0.00% |
| Flooring Materials | 1 | 0.00% |
| Flooring Design | 1 | 0.00% |
| economic theory | 1 | 0.00% |
| public policy | 1 | 0.00% |
| Team Captain | 1 | 0.00% |
| Pop-up | 1 | 0.00% |
| Bike model | 1 | 0.00% |
| orbital_period.duration | 1 | 0.00% |
| wearable accessory | 1 | 0.00% |
| design principle | 1 | 0.00% |
| audio device | 1 | 0.00% |
| audio device feature | 1 | 0.00% |
| Precious metal | 1 | 0.00% |
| Blog posts | 1 | 0.00% |
| protest action | 1 | 0.00% |
| political movement | 1 | 0.00% |
| electric car | 1 | 0.00% |
| type of vehicle | 1 | 0.00% |
| Person/Position | 1 | 0.00% |
| Animal Husbandry | 1 | 0.00% |
| staffing_change | 1 | 0.00% |
| Paper title | 1 | 0.00% |
| research_context.primary_technique | 1 | 0.00% |
| outcomes.impact | 1 | 0.00% |
| outcomes.comparative_advantage.benefit_type | 1 | 0.00% |
| research_context.problem_statement.limitation | 1 | 0.00% |
| methodology.experimental_setup.reagents.0.type | 1 | 0.00% |
| research_context.application_goal | 1 | 0.00% |
| methodology.experimental_setup.biological_model.culture_environment.material_properties | 1 | 0.00% |
| outcomes.comparative_advantage.alternative_to | 1 | 0.00% |
| methodology.experimental_setup.instrumentation | 1 | 0.00% |
| website theme | 1 | 0.00% |
| website builder feature | 1 | 0.00% |
| website add-ons | 1 | 0.00% |
| website visibility | 1 | 0.00% |
| website developer | 1 | 0.00% |
| boiling_instructions.cooking_time.min_minutes | 1 | 0.00% |
| boiling_instructions.cooking_time.max_minutes | 1 | 0.00% |
| boiling_instructions.salt_advice | 1 | 0.00% |
| ideal state | 1 | 0.00% |
| symbol: Stock ticker symbol as listed in the data | 1 | 0.00% |
| change_percent: Percentage change in stock price for the day | 1 | 0.00% |
| close_price: Final closing price of the stock | 1 | 0.00% |
| volume: Total shares traded on the day | 1 | 0.00% |
| market_cap: Market capitalization in billions | 1 | 0.00% |
| movement_reason: Brief reason for price movement or performance | 1 | 0.00% |
| Location/Play Area | 1 | 0.00% |
| Program/service | 1 | 0.00% |
| Dishonesty | 1 | 0.00% |
| wallet_address | 1 | 0.00% |
| Document Structure | 1 | 0.00% |
| Religious Location | 1 | 0.00% |
| Family/Political dynasty | 1 | 0.00% |
| Government/Regime | 1 | 0.00% |
| Legacy/Political dynasty | 1 | 0.00% |
| Nepotism/Political dynasty | 1 | 0.00% |
| Restaurant chain | 1 | 0.00% |
| Canine Buddy | 1 | 0.00% |
| Coffee Pod | 1 | 0.00% |
| Time/Availability | 1 | 0.00% |
| Object Part | 1 | 0.00% |
| Biological pathway | 1 | 0.00% |
| Factors | 1 | 0.00% |
| task/work | 1 | 0.00% |
| negative financial outcome | 1 | 0.00% |
| Task section | 1 | 0.00% |
| School Term | 1 | 0.00% |
| online recharge services | 1 | 0.00% |
| Art Blog | 1 | 0.00% |
| Artistic Talent | 1 | 0.00% |
| Livestock | 1 | 0.00% |
| Mechanism | 1 | 0.00% |
| disorder | 1 | 0.00% |
| Emotionen | 1 | 0.00% |
| Eigenschaften | 1 | 0.00% |
| Musikstück | 1 | 0.00% |
| Organization Entity | 1 | 0.00% |
| internet meme | 1 | 0.00% |
| application scenarios | 1 | 0.00% |
| File/Document | 1 | 0.00% |
| Business strategy | 1 | 0.00% |
| Developmental milestone | 1 | 0.00% |
| Foundation | 1 | 0.00% |
| food scene | 1 | 0.00% |
| travelogue | 1 | 0.00% |
| food trucks | 1 | 0.00% |
| Image Dimension | 1 | 0.00% |
| Recreation Center | 1 | 0.00% |
| Weather/Climate | 1 | 0.00% |
| hotel_details.star_rating | 1 | 0.00% |
| ranking.location | 1 | 0.00% |
| hotel_details.location | 1 | 0.00% |
| booking_options.description | 1 | 0.00% |
| hotel_details.name | 1 | 0.00% |
| Brand/Manufacturer | 1 | 0.00% |
| Film Festival Venue | 1 | 0.00% |
| Film Festival | 1 | 0.00% |
| Film Selection | 1 | 0.00% |
| medical treatment/drug | 1 | 0.00% |
| medication/drug | 1 | 0.00% |
| biomolecule | 1 | 0.00% |
| process steps | 1 | 0.00% |
| Role/Activity | 1 | 0.00% |
| Object/Project | 1 | 0.00% |
| wood type | 1 | 0.00% |
| Movie Theatre | 1 | 0.00% |
| Filmmaker | 1 | 0.00% |
| Chemical ingredient | 1 | 0.00% |
| biography.awards.0.year | 1 | 0.00% |
| biography.education.0.teacher | 1 | 0.00% |
| birth_information.birth_place.state | 1 | 0.00% |
| biography.teaching_position.institution | 1 | 0.00% |
| birth_information.birth_year | 1 | 0.00% |
| biography.education.4.institution | 1 | 0.00% |
| biography.education.3.teacher | 1 | 0.00% |
| birth_information.birth_place.city | 1 | 0.00% |
| biography.education.1.institution | 1 | 0.00% |
| biography.awards.0.organization | 1 | 0.00% |
| biography.education.2.institution | 1 | 0.00% |
| appVersion | 1 | 0.00% |
| key1 | 1 | 0.00% |
| key2 | 1 | 0.00% |
| Mine | 1 | 0.00% |
| Housing Development Phase | 1 | 0.00% |
| car type | 1 | 0.00% |
| municipality.creation_date | 1 | 0.00% |
| municipality.province | 1 | 0.00% |
| Data storage device | 1 | 0.00% |
| Skill/Qualification | 1 | 0.00% |
| Job Title/Profession | 1 | 0.00% |
| E-commerce Website | 1 | 0.00% |
| drilling fluid | 1 | 0.00% |
| mixer | 1 | 0.00% |
| Mineral type | 1 | 0.00% |
| fictional object | 1 | 0.00% |
| Food/Cuisine | 1 | 0.00% |
| software tools | 1 | 0.00% |
| software frameworks | 1 | 0.00% |
| real estate market | 1 | 0.00% |
| parameters | 1 | 0.00% |
| Salon Name | 1 | 0.00% |
| Customer Type | 1 | 0.00% |
| Hotel brand | 1 | 0.00% |
| Discount/Deal | 1 | 0.00% |
| Mobile Device | 1 | 0.00% |
| Youth | 1 | 0.00% |
| Worship | 1 | 0.00% |
| fungal disease | 1 | 0.00% |
| plant population | 1 | 0.00% |
| instrument/tool | 1 | 0.00% |
| Website/Page | 1 | 0.00% |
| Artwork Design | 1 | 0.00% |
| Mining company | 1 | 0.00% |
| Illumination device | 1 | 0.00% |
| Scarcity | 1 | 0.00% |
| deceitful behavior | 1 | 0.00% |
| dishonest behavior | 1 | 0.00% |
| self-centered behavior | 1 | 0.00% |
| basic requirement | 1 | 0.00% |
| flight service | 1 | 0.00% |
| Plant Life | 1 | 0.00% |
| Vegetable Crop | 1 | 0.00% |
| Plant Structure | 1 | 0.00% |
| skincare term | 1 | 0.00% |
| Ownership type | 1 | 0.00% |
| Aesthetic Feature | 1 | 0.00% |
| email server software | 1 | 0.00% |
| cryptographic protocol | 1 | 0.00% |
| project/achievement | 1 | 0.00% |
| client/satisfaction | 1 | 0.00% |
| activity/hobby | 1 | 0.00% |
| trail/location | 1 | 0.00% |
| Environmental Event | 1 | 0.00% |
| beings | 1 | 0.00% |
| instrument collection | 1 | 0.00% |
| weaponry collection | 1 | 0.00% |
| Ritual/Culture | 1 | 0.00% |
| performing arts | 1 | 0.00% |
| habit/action | 1 | 0.00% |
| continuation of habit/action | 1 | 0.00% |
| accomplishment/goal | 1 | 0.00% |
| Dynasty | 1 | 0.00% |
| material/tool | 1 | 0.00% |
| inheritance | 1 | 0.00% |
| contribution | 1 | 0.00% |
| investment strategy | 1 | 0.00% |
| market analysis | 1 | 0.00% |
| market trends | 1 | 0.00% |
| Room Quantity | 1 | 0.00% |
| interest group | 1 | 0.00% |
| webpage reference | 1 | 0.00% |
| Payment service | 1 | 0.00% |
| Security technology | 1 | 0.00% |
| Financial institution | 1 | 0.00% |
| physical infrastructure | 1 | 0.00% |
| sports court | 1 | 0.00% |
| linguistic feature | 1 | 0.00% |
| Folklore | 1 | 0.00% |
| option | 1 | 0.00% |
| form element | 1 | 0.00% |
| Process/Guideline | 1 | 0.00% |
| Quality/Standard | 1 | 0.00% |
| packagingInstructions | 1 | 0.00% |
| Growth stage | 1 | 0.00% |
| Performance metric | 1 | 0.00% |
| Book Pick List | 1 | 0.00% |
| Bookclub Pick List | 1 | 0.00% |
| service/repair | 1 | 0.00% |
| service/replacement | 1 | 0.00% |
| damage_amount | 1 | 0.00% |
| causeofloss | 1 | 0.00% |
| discussion/communication | 1 | 0.00% |
| social/political environment | 1 | 0.00% |
| Publication/Communication | 1 | 0.00% |
| metadata.source_links.0.language | 1 | 0.00% |
| contract_details.financial_value.currency | 1 | 0.00% |
| metadata.report_date | 1 | 0.00% |
| entity.coverage_details.headline_news_provider | 1 | 0.00% |
| metadata.news_agency | 1 | 0.00% |
| entity.company_name | 1 | 0.00% |
| contract_details.contract_type | 1 | 0.00% |
| salary.hourly_rate.currency | 1 | 0.00% |
| salary.monthly_estimate.hours_per_week | 1 | 0.00% |
| salary.hourly_rate.amount | 1 | 0.00% |
| supplement | 1 | 0.00% |
| adverse reaction | 1 | 0.00% |
| herbalist | 1 | 0.00% |
| school culture/values | 1 | 0.00% |
| Activist Group | 1 | 0.00% |
| camouflage technique | 1 | 0.00% |
| merchant ships | 1 | 0.00% |
| Health hazard | 1 | 0.00% |
| Consumable | 1 | 0.00% |
| insurance process | 1 | 0.00% |
| Character Type | 1 | 0.00% |
| telescope | 1 | 0.00% |
| catalog | 1 | 0.00% |
| Event/Title | 1 | 0.00% |
| emotion-related entity | 1 | 0.00% |
| Error type | 1 | 0.00% |
| price quote | 1 | 0.00% |
| patent_number: Unique identifier for the patent application or grant | 1 | 0.00% |
| status: Current legal status of the patent | 1 | 0.00% |
| Investment process | 1 | 0.00% |
| Financing type | 1 | 0.00% |
| author.role | 1 | 0.00% |
| author.name | 1 | 0.00% |
| Photography Service | 1 | 0.00% |
| Energy source and wattage | 1 | 0.00% |
| Energy source and type of light | 1 | 0.00% |
| Data Object | 1 | 0.00% |
| application document | 1 | 0.00% |
| document set | 1 | 0.00% |
| channel: Name of the creator's YouTube channel | 1 | 0.00% |
| thumbnail: Description of the video thumbnail design and colors | 1 | 0.00% |
| share_count: Number of times the video has been shared | 1 | 0.00% |
| title: Title of the video as displayed in the feed | 1 | 0.00% |
| future event | 1 | 0.00% |
| Technical Resource | 1 | 0.00% |
| dose_number | 1 | 0.00% |
| dateofbirth | 1 | 0.00% |
| administration_date | 1 | 0.00% |
| vaccination_id | 1 | 0.00% |
| Video Platform | 1 | 0.00% |
| Digital Document | 1 | 0.00% |
| social class/powerful individuals | 1 | 0.00% |
| Performer/Musician | 1 | 0.00% |
| geography.rivers | 1 | 0.00% |
| geography.location | 1 | 0.00% |
| geography.borders.2.direction | 1 | 0.00% |
| infrastructure.refineries.0.status | 1 | 0.00% |
| infrastructure.highways.2.highwayName | 1 | 0.00% |
| history.mergerDescription | 1 | 0.00% |
| infrastructure.highways.0.highwayName | 1 | 0.00% |
| economy.agriculturalArea | 1 | 0.00% |
| infrastructure.railways | 1 | 0.00% |
| regionNameBerber | 1 | 0.00% |
| populationYear | 1 | 0.00% |
| infrastructure.roads.0.roadName | 1 | 0.00% |
| infrastructure.airports.0.trafficVolume | 1 | 0.00% |
| infrastructure.airports.0.passengersYear | 1 | 0.00% |
| infrastructure.airports.0.location | 1 | 0.00% |
| infrastructure.refineries.0.location | 1 | 0.00% |
| economy.gdpRankAmongRegions | 1 | 0.00% |
| economy.gdpValue | 1 | 0.00% |
| geography.borders.0.region | 1 | 0.00% |
| geography.borders.1.region | 1 | 0.00% |
| infrastructure.airports.0.airportName | 1 | 0.00% |
| infrastructure.refineries.0.refineryName | 1 | 0.00% |
| regionNameArabic | 1 | 0.00% |
| Guidelines | 1 | 0.00% |
| Current events | 1 | 0.00% |
| Global network | 1 | 0.00% |
| Reading interface | 1 | 0.00% |
| Information portal | 1 | 0.00% |
| social behavior | 1 | 0.00% |
| Spatial Location | 1 | 0.00% |
| Webpage Component | 1 | 0.00% |
| production company | 1 | 0.00% |
| Company Success | 1 | 0.00% |
| Customer Needs | 1 | 0.00% |
| demand | 1 | 0.00% |
| health model | 1 | 0.00% |
| mental practice | 1 | 0.00% |
| Vehicle brand | 1 | 0.00% |
| Type of rock | 1 | 0.00% |
| Quality of sand stone | 1 | 0.00% |
| Plan/Program | 1 | 0.00% |
| Agreement/Commitment | 1 | 0.00% |
| Purchase Requirement | 1 | 0.00% |
| Excluded Products | 1 | 0.00% |
| Hygiene Practice | 1 | 0.00% |
| address: Full street address of the policyholder | 1 | 0.00% |
| collision_type: Type of collision or damage event | 1 | 0.00% |
| Body system | 1 | 0.00% |
| Medical measurement | 1 | 0.00% |
| Instruction manual | 1 | 0.00% |
| healthcare recipient | 1 | 0.00% |
| Support service | 1 | 0.00% |
| RVs | 1 | 0.00% |
| Safety equipment | 1 | 0.00% |
| Road hazard | 1 | 0.00% |
| Road type | 1 | 0.00% |
| Listing | 1 | 0.00% |
| Review | 1 | 0.00% |
| Family Name | 1 | 0.00% |
| initiative | 1 | 0.00% |
| community development | 1 | 0.00% |
| forestry management | 1 | 0.00% |
| forest resources exploitation | 1 | 0.00% |
| supply chain management | 1 | 0.00% |
| forest products marketing | 1 | 0.00% |
| data management | 1 | 0.00% |
| forecasting | 1 | 0.00% |
| irregularities | 1 | 0.00% |
| logical thinking | 1 | 0.00% |
| Organization/Time Period | 1 | 0.00% |
| Judicial Entity | 1 | 0.00% |
| Network protocol | 1 | 0.00% |
| Network topology | 1 | 0.00% |
| Power over Ethernet | 1 | 0.00% |
| Guitar part | 1 | 0.00% |
| User group | 1 | 0.00% |
| emotional aid | 1 | 0.00% |
| Autism Spectrum Disorder (ASD) | 1 | 0.00% |
| Sensory Processing Disorder (SPD) | 1 | 0.00% |
| Short film | 1 | 0.00% |
| Man-made Feature | 1 | 0.00% |
| measurement entity | 1 | 0.00% |
| version: Product version number as specified in the passage | 1 | 0.00% |
| region: Geographic region where the product is targeted | 1 | 0.00% |
| status: Current operational status of the product | 1 | 0.00% |
| product_id: Unique identifier for the product version | 1 | 0.00% |
| Social media event | 1 | 0.00% |
| Parting words | 1 | 0.00% |
| account_number | 1 | 0.00% |
| merchant_name | 1 | 0.00% |
| people/group of people | 1 | 0.00% |
| internship | 1 | 0.00% |
| security code | 1 | 0.00% |
| swimming event | 1 | 0.00% |
| Clothing material | 1 | 0.00% |
| Social relationship | 1 | 0.00% |
| Music quality | 1 | 0.00% |
| unit_id | 1 | 0.00% |
| acquisition_id | 1 | 0.00% |
| Personal identification number | 1 | 0.00% |
| Identification process | 1 | 0.00% |
| style type | 1 | 0.00% |
| Health goal | 1 | 0.00% |
| Transportation Station | 1 | 0.00% |
| sound journey | 1 | 0.00% |
| intuitive healing touch | 1 | 0.00% |
| Structural Component | 1 | 0.00% |
| caretakers | 1 | 0.00% |
| school event | 1 | 0.00% |
| response or opinion | 1 | 0.00% |
| presence | 1 | 0.00% |
| Art Collector | 1 | 0.00% |
| Work Title | 1 | 0.00% |
| Event Production | 1 | 0.00% |
| Beer type | 1 | 0.00% |
| Musical element | 1 | 0.00% |
| Architecture structure | 1 | 0.00% |
| Community culture | 1 | 0.00% |
| Business revenue | 1 | 0.00% |
| food ingredients | 1 | 0.00% |
| financial institutions | 1 | 0.00% |
| economic crisis | 1 | 0.00% |
| financial liquidity | 1 | 0.00% |
| economic decline rate | 1 | 0.00% |
| social_media_post.content | 1 | 0.00% |
| event.subject | 1 | 0.00% |
| event.location.region | 1 | 0.00% |
| social_media_post.platform | 1 | 0.00% |
| event.surgery_date | 1 | 0.00% |
| event.date | 1 | 0.00% |
| spatial entity | 1 | 0.00% |
| User Behavior | 1 | 0.00% |
| Action/Service | 1 | 0.00% |
| discovery.project | 1 | 0.00% |
| discovery.location.county | 1 | 0.00% |
| physical_characteristics.observations.0.publication | 1 | 0.00% |
| real estate listing number | 1 | 0.00% |
| property location/type | 1 | 0.00% |
| quantity of individuals | 1 | 0.00% |
| reality | 1 | 0.00% |
| state of being occupied | 1 | 0.00% |
| Business/Entertainment | 1 | 0.00% |
| candy/brand | 1 | 0.00% |
| product form | 1 | 0.00% |
| animal part | 1 | 0.00% |
| Sleep Accessory | 1 | 0.00% |
| Sleep Standard | 1 | 0.00% |
| parking type | 1 | 0.00% |
| Memory/Emotion | 1 | 0.00% |
| Collectible Coin | 1 | 0.00% |
| Scientific Field | 1 | 0.00% |
| Effort | 1 | 0.00% |
| decision making process | 1 | 0.00% |
| suitability | 1 | 0.00% |
| Target Group | 1 | 0.00% |
| Price Deal | 1 | 0.00% |
| Price Type | 1 | 0.00% |
| Digital image | 1 | 0.00% |
| Deal/Discount | 1 | 0.00% |
| Clothing fit | 1 | 0.00% |
| Fabric pattern | 1 | 0.00% |
| Level | 1 | 0.00% |
| Avatar | 1 | 0.00% |
| Font Configuration | 1 | 0.00% |
| State Representative | 1 | 0.00% |
| Medical Examiner/School Board Member | 1 | 0.00% |
| biological molecule | 1 | 0.00% |
| Binary options broker | 1 | 0.00% |
| Negative emotion | 1 | 0.00% |
| Tile | 1 | 0.00% |
| young dogs | 1 | 0.00% |
| affected_area | 1 | 0.00% |
| wind_speed | 1 | 0.00% |
| military object | 1 | 0.00% |
| trend/activity | 1 | 0.00% |
| prohibition | 1 | 0.00% |
| Person/Producer | 1 | 0.00% |
| Location/Street Name | 1 | 0.00% |
| primary_diagnosis | 1 | 0.00% |
| Music Festival | 1 | 0.00% |
| computer security | 1 | 0.00% |
| computer software | 1 | 0.00% |
| Harvested area | 1 | 0.00% |
| Color name | 1 | 0.00% |
| Design/Style | 1 | 0.00% |
| Material/Attribute | 1 | 0.00% |
| accommodation facility | 1 | 0.00% |
| drinking establishment | 1 | 0.00% |
| numerical data | 1 | 0.00% |
| Amusement park | 1 | 0.00% |
| professional trait | 1 | 0.00% |
| Application Feature | 1 | 0.00% |
| Data Entity | 1 | 0.00% |
| Medical equipment/supplies | 1 | 0.00% |
| Medical treatment/healthcare services | 1 | 0.00% |
| Diagnostic imaging test | 1 | 0.00% |
| Dermatological treatment | 1 | 0.00% |
| Minor medical procedures | 1 | 0.00% |
| personal hygiene | 1 | 0.00% |
| temperature control | 1 | 0.00% |
| Political Action | 1 | 0.00% |
| file_id | 1 | 0.00% |
| financial model | 1 | 0.00% |
| nature-inspired design | 1 | 0.00% |
| finishing technique | 1 | 0.00% |
| Round | 1 | 0.00% |
| Puzzle Game | 1 | 0.00% |
| Crossword Clue | 1 | 0.00% |
| Racing class | 1 | 0.00% |
| Racing term | 1 | 0.00% |
| travel route | 1 | 0.00% |
| Legal Arrangement | 1 | 0.00% |
| input device | 1 | 0.00% |
| input action | 1 | 0.00% |
| Brand/Trademark | 1 | 0.00% |
| Health and Fitness | 1 | 0.00% |
| Nutrition | 1 | 0.00% |
| winner: Name of the player or team that won | 1 | 0.00% |
| winner_id: Unique ID assigned to the winner | 1 | 0.00% |
| date: Date the tournament was held | 1 | 0.00% |
| opponent: Name of the player or team that lost | 1 | 0.00% |
| Building materials | 1 | 0.00% |
| total_amount | 1 | 0.00% |
| tax_amount | 1 | 0.00% |
| shipping_amount | 1 | 0.00% |
| warehouse_location: Geographic warehouse where product is stored | 1 | 0.00% |
| last_updated: Date when product record was last modified | 1 | 0.00% |
| Social Media Audience | 1 | 0.00% |
| Online Discussion Platform | 1 | 0.00% |
| Creative Motivation | 1 | 0.00% |
| organ system | 1 | 0.00% |
| poss | 1 | 0.00% |
| intj | 1 | 0.00% |
| Critique | 1 | 0.00% |
| Conversation | 1 | 0.00% |
| Organizational level | 1 | 0.00% |
| Sales strategy | 1 | 0.00% |
| Event Process | 1 | 0.00% |
| House name | 1 | 0.00% |
| Attitude/Emotion | 1 | 0.00% |
| User Interface | 1 | 0.00% |
| lens component | 1 | 0.00% |
| camera component | 1 | 0.00% |
| repair action | 1 | 0.00% |
| Fishing charter | 1 | 0.00% |
| Russian cosmonaut | 1 | 0.00% |
| NASA astronaut | 1 | 0.00% |
| British astronaut | 1 | 0.00% |
| Greyhound | 1 | 0.00% |
| Kitchen area | 1 | 0.00% |
| Door type | 1 | 0.00% |
| Bedroom space | 1 | 0.00% |
| skincare activity | 1 | 0.00% |
| facial mask | 1 | 0.00% |
| music artists | 1 | 0.00% |
| music release | 1 | 0.00% |
| Government Official | 1 | 0.00% |
| Limited Edition Merchandise | 1 | 0.00% |
| Trade Show Event | 1 | 0.00% |
| product/component type | 1 | 0.00% |
| fictional location | 1 | 0.00% |
| Test/Quiz | 1 | 0.00% |
| clinical_observations.activation_status | 1 | 0.00% |
| clinical_observations.prevalence | 1 | 0.00% |
| splicing_regulation.current_research_status | 1 | 0.00% |
| splicing_regulation.impact_on_pathway_components.0.regulatory_mode | 1 | 0.00% |
| splicing_regulation.impact_on_pathway_components.0.component_description | 1 | 0.00% |
| pathway_name | 1 | 0.00% |
| Community center | 1 | 0.00% |
| tour event | 1 | 0.00% |
| Commune | 1 | 0.00% |
| finance guru | 1 | 0.00% |
| magic number | 1 | 0.00% |
| news source | 1 | 0.00% |
| CBD strain | 1 | 0.00% |
| Size/Region | 1 | 0.00% |
| Lifestyle/Philosophy | 1 | 0.00% |
| Unicode characters | 1 | 0.00% |
| Input device | 1 | 0.00% |
| date: Date the exchange rate snapshot was taken | 1 | 0.00% |
| sender: Name of the individual initiating the transfer | 1 | 0.00% |
| address: Physical address of the sender | 1 | 0.00% |
| Manuscript Title | 1 | 0.00% |
| Wrongdoing | 1 | 0.00% |
| Evil Work | 1 | 0.00% |
| scholarship | 1 | 0.00% |
| cutting method | 1 | 0.00% |
| glassware | 1 | 0.00% |
| utensils | 1 | 0.00% |
| food items | 1 | 0.00% |
| space/area | 1 | 0.00% |
| beverages | 1 | 0.00% |
| Group of Men | 1 | 0.00% |
| Written Works | 1 | 0.00% |
| airport facility | 1 | 0.00% |
| passenger class/type | 1 | 0.00% |
| document/payment type | 1 | 0.00% |
| Project Plan | 1 | 0.00% |
| Land Development | 1 | 0.00% |
| generation | 1 | 0.00% |
| Contract Type | 1 | 0.00% |
| tehsil.tehsil_name | 1 | 0.00% |
| tehsil.district | 1 | 0.00% |
| Connection issue | 1 | 0.00% |
| Car | 1 | 0.00% |
| Location Adverb | 1 | 0.00% |
| loyalty program | 1 | 0.00% |
| Suspension Component | 1 | 0.00% |
| medical practice | 1 | 0.00% |
| bathroom | 1 | 0.00% |
| City/Community | 1 | 0.00% |
| Socioeconomic status group | 1 | 0.00% |
| Geographical boundary | 1 | 0.00% |
| Aid/Support | 1 | 0.00% |
| in-game currency | 1 | 0.00% |
| in-game milestones | 1 | 0.00% |
| in-game resource | 1 | 0.00% |
| in-game character type | 1 | 0.00% |
| in-game action | 1 | 0.00% |
| Behavioral Trait | 1 | 0.00% |
| Dance School | 1 | 0.00% |
| Spanish phrase | 1 | 0.00% |
| Company Email Address | 1 | 0.00% |
| Email Hosting Service | 1 | 0.00% |
| key leader | 1 | 0.00% |
| creative outlet | 1 | 0.00% |
| specific target audience | 1 | 0.00% |
| Cleaning Service | 1 | 0.00% |
| computer file property | 1 | 0.00% |
| computer file | 1 | 0.00% |
| concentration | 1 | 0.00% |
| dosage form | 1 | 0.00% |
| active ingredient | 1 | 0.00% |
| sensory property | 1 | 0.00% |
| culinary item | 1 | 0.00% |
| cultural site | 1 | 0.00% |
| National Symbols | 1 | 0.00% |
| Colors | 1 | 0.00% |
| Packaging Method | 1 | 0.00% |
| pigment | 1 | 0.00% |
| tournament | 1 | 0.00% |
| court | 1 | 0.00% |
| diseases_treated.scope | 1 | 0.00% |
| treatment_methods.selection_criteria | 1 | 0.00% |
| patient_care.station_assignment | 1 | 0.00% |
| treatment_methods.description | 1 | 0.00% |
| Engine Component | 1 | 0.00% |
| Piston Brand | 1 | 0.00% |
| Engine Usage Time | 1 | 0.00% |
| agenda item | 1 | 0.00% |
| Building Address | 1 | 0.00% |
| Administrative District | 1 | 0.00% |
| Hobbies/Interests | 1 | 0.00% |
| Emotional bond | 1 | 0.00% |
| Product performance | 1 | 0.00% |
| intellectual_compatibility.gemini_perspective.approach | 1 | 0.00% |
| intellectual_compatibility.virgo_perspective.focus | 1 | 0.00% |
| intellectual_compatibility.gemini_perspective.focus | 1 | 0.00% |
| event.casualties | 1 | 0.00% |
| source.name | 1 | 0.00% |
| event.impact.target | 1 | 0.00% |
| Data file | 1 | 0.00% |
| goal or objective | 1 | 0.00% |
| Event Edition | 1 | 0.00% |
| Tournament Group | 1 | 0.00% |
| tournament round | 1 | 0.00% |
| sports teams | 1 | 0.00% |
| tractor model | 1 | 0.00% |
| edition type | 1 | 0.00% |
| digital document | 1 | 0.00% |
| assigned_agent | 1 | 0.00% |
| Palestinian camp | 1 | 0.00% |
| political groups | 1 | 0.00% |
| hydraulic lifting equipment | 1 | 0.00% |
| Internet Service Provider | 1 | 0.00% |
| Business Performance | 1 | 0.00% |
| Marketing Channel | 1 | 0.00% |
| Convenience store chain | 1 | 0.00% |
| Payment system | 1 | 0.00% |
| Store location | 1 | 0.00% |
| Cultural Building | 1 | 0.00% |
| bridgework component | 1 | 0.00% |
| dental service | 1 | 0.00% |
| Group of Companies/Brands | 1 | 0.00% |
| company entity | 1 | 0.00% |
| caution | 1 | 0.00% |
| Skin damage | 1 | 0.00% |
| Dirt | 1 | 0.00% |
| Product Collection | 1 | 0.00% |
| session_type | 1 | 0.00% |
| Car body type | 1 | 0.00% |
| occupation/nickname | 1 | 0.00% |
| victory | 1 | 0.00% |
| subscription | 1 | 0.00% |
| Diamond clarity | 1 | 0.00% |
| Diamond color | 1 | 0.00% |
| life journey | 1 | 0.00% |
| action steps | 1 | 0.00% |
| brewing process | 1 | 0.00% |
| Journal Name | 1 | 0.00% |
| Username/Handle | 1 | 0.00% |
| Network Hardware | 1 | 0.00% |
| Meal Time | 1 | 0.00% |
| Tap Capacity | 1 | 0.00% |
| Tool Name | 1 | 0.00% |
| Link | 1 | 0.00% |
| Cooking Technique | 1 | 0.00% |
| Culinary Ingredient | 1 | 0.00% |
| Culinary Descriptor | 1 | 0.00% |
| Person/Religious Figure | 1 | 0.00% |
| Motor specification | 1 | 0.00% |
| Propeller size | 1 | 0.00% |
| Digital File Format | 1 | 0.00% |
| social/political issue | 1 | 0.00% |
| shelfLifeInDays | 1 | 0.00% |
| storageConditions | 1 | 0.00% |
| establishment date | 1 | 0.00% |
| website portfolio | 1 | 0.00% |
| work project | 1 | 0.00% |
| online resources | 1 | 0.00% |
| natural events | 1 | 0.00% |
| print quality | 1 | 0.00% |
| delivery speed | 1 | 0.00% |
| Physical Harm | 1 | 0.00% |
| Emotional Harm | 1 | 0.00% |
| Product Liability | 1 | 0.00% |
| concrete entity | 1 | 0.00% |
| Group/Identity | 1 | 0.00% |
| Job title/Role | 1 | 0.00% |
| Data Set | 1 | 0.00% |
| Data Handling | 1 | 0.00% |
| public_information.download_share | 1 | 0.00% |
| advertiser_information.contact_information.phone_number | 1 | 0.00% |
| distribution_information.trail_details.trail_name | 1 | 0.00% |
| emotional response | 1 | 0.00% |
| Painting style | 1 | 0.00% |
| Action Plan | 1 | 0.00% |
| natural occurrence | 1 | 0.00% |
| Course Material | 1 | 0.00% |
| Course Level | 1 | 0.00% |
| Course Access | 1 | 0.00% |
| Greeting/Welcome | 1 | 0.00% |
| Person/Contact | 1 | 0.00% |
| Contact/Email | 1 | 0.00% |
| Company/Stock | 1 | 0.00% |
| Change in Quantity | 1 | 0.00% |
| travel class | 1 | 0.00% |
| wind intensity | 1 | 0.00% |
| Park/Trail | 1 | 0.00% |
| Family/Relationships | 1 | 0.00% |
| social support group | 1 | 0.00% |
| units_sold: Total units sold by a specific date | 1 | 0.00% |
| region: Primary customer region where product is popular | 1 | 0.00% |
| name: Name of the product as listed in best-sellers | 1 | 0.00% |
| embellishment | 1 | 0.00% |
| fabric | 1 | 0.00% |
| web development | 1 | 0.00% |
| airport.icao_code | 1 | 0.00% |
| airport.altitude.unit | 1 | 0.00% |
| airport.iata_code | 1 | 0.00% |
| airport.passenger_flow.unit | 1 | 0.00% |
| airport.funding_source | 1 | 0.00% |
| statistics.source | 1 | 0.00% |
| airport.status | 1 | 0.00% |
| airport.location.city | 1 | 0.00% |
| airport.description | 1 | 0.00% |
| musician/artist | 1 | 0.00% |
| music production | 1 | 0.00% |
| food and beverage | 1 | 0.00% |
| Residential Community | 1 | 0.00% |
| sender_address | 1 | 0.00% |
| muscle group | 1 | 0.00% |
| personal goal | 1 | 0.00% |
| selflessness | 1 | 0.00% |
| task assigned | 1 | 0.00% |
| Holiday traditions | 1 | 0.00% |
| Food and beverage | 1 | 0.00% |
| architectural project | 1 | 0.00% |
| dispensingDate | 1 | 0.00% |
| quantityDispensed | 1 | 0.00% |
| refillsAllowed | 1 | 0.00% |
| Building Value | 1 | 0.00% |
| Building Systems | 1 | 0.00% |
| design/style | 1 | 0.00% |
| Adversary | 1 | 0.00% |
| Puppy | 1 | 0.00% |
| Baggage Management Service | 1 | 0.00% |
| Massage Program | 1 | 0.00% |
| Massage Technique | 1 | 0.00% |
| Massage Feature | 1 | 0.00% |
| Massage Experience | 1 | 0.00% |
| Document Status | 1 | 0.00% |
| document request | 1 | 0.00% |
| offensive lineman | 1 | 0.00% |
| Furniture/Structure | 1 | 0.00% |
| Indoor Air Quality Factor | 1 | 0.00% |
| Healthcare Provider | 1 | 0.00% |
| Healthcare Facility Location | 1 | 0.00% |
| Healthcare Facility Name | 1 | 0.00% |
| Person/Musician | 1 | 0.00% |
| Organization/Music Group | 1 | 0.00% |
| Person/Conductor | 1 | 0.00% |
| video format | 1 | 0.00% |
| software vulnerability | 1 | 0.00% |
| Cultural Event | 1 | 0.00% |
| fishing bait | 1 | 0.00% |
| Medical Research | 1 | 0.00% |
| Ease of use/convenience | 1 | 0.00% |
| Facility/Building | 1 | 0.00% |
| job responsibility | 1 | 0.00% |
| Economy | 1 | 0.00% |
| camping gear | 1 | 0.00% |
| treatment_overview.experience | 1 | 0.00% |
| treatment_examples.items.3.conditions | 1 | 0.00% |
| additional_treatments | 1 | 0.00% |
| treatment_examples.items.1.category | 1 | 0.00% |
| treatment_examples.items.2.category | 1 | 0.00% |
| treatment_examples.items.3.category | 1 | 0.00% |
| treatment_overview.principle | 1 | 0.00% |
| treatment_examples.items.1.conditions | 1 | 0.00% |
| treatment_examples.items.2.conditions | 1 | 0.00% |
| product release | 1 | 0.00% |
| ship_name | 1 | 0.00% |
| ship_id | 1 | 0.00% |
| ship_type | 1 | 0.00% |
| ship_class | 1 | 0.00% |
| navy | 1 | 0.00% |
| history.creation.laid_down_date | 1 | 0.00% |
| history.creation.shipyard.name | 1 | 0.00% |
| history.creation.shipyard.location | 1 | 0.00% |
| history.creation.commissioning_date | 1 | 0.00% |
| Doctor | 1 | 0.00% |
| File Format<> A type of digital video file format | 1 | 0.00% |
| Political Campaign | 1 | 0.00% |
| Electoral District | 1 | 0.00% |
| legal role | 1 | 0.00% |
| Music artist/band | 1 | 0.00% |
| guideline | 1 | 0.00% |
| Job Policy | 1 | 0.00% |
| Education/Qualification | 1 | 0.00% |
| Work Experience/Qualification | 1 | 0.00% |
| Sequelae | 1 | 0.00% |
| Distance Measurement | 1 | 0.00% |
| Scenery/Environment | 1 | 0.00% |
| vehicle_make | 1 | 0.00% |
| Music band/Group | 1 | 0.00% |
| document feature | 1 | 0.00% |
| death.date_julian | 1 | 0.00% |
| birth.date_gregorian | 1 | 0.00% |
| works.type | 1 | 0.00% |
| birth.date_julian | 1 | 0.00% |
| teacher | 1 | 0.00% |
| death.location | 1 | 0.00% |
| father.occupation | 1 | 0.00% |
| birth.location | 1 | 0.00% |
| Location Feature | 1 | 0.00% |
| baseball talent level | 1 | 0.00% |
| potential for growth | 1 | 0.00% |
| recruitment strategy | 1 | 0.00% |
| community space | 1 | 0.00% |
| pilot | 1 | 0.00% |
| Activity/Place | 1 | 0.00% |
| hot sauce | 1 | 0.00% |
| speciality | 1 | 0.00% |
| personal health concerns | 1 | 0.00% |
| work leave | 1 | 0.00% |
| craftmanship | 1 | 0.00% |
| furniture design element | 1 | 0.00% |
| architectural detail | 1 | 0.00% |
| construction element | 1 | 0.00% |
| Home renovation/construction | 1 | 0.00% |
| Error/Blunder | 1 | 0.00% |
| Planning/Preparation | 1 | 0.00% |
| computer board | 1 | 0.00% |
| electronic display | 1 | 0.00% |
| Country/Territory | 1 | 0.00% |
| Service/Feature | 1 | 0.00% |
| Domain Name | 1 | 0.00% |
| Country Code | 1 | 0.00% |
| Registration Process | 1 | 0.00% |
| Job Category | 1 | 0.00% |
| reflection | 1 | 0.00% |
| positivity | 1 | 0.00% |
| trade agreement | 1 | 0.00% |
| geopolitical entities | 1 | 0.00% |
| employment opportunities | 1 | 0.00% |
| travel service | 1 | 0.00% |
| Furniture part | 1 | 0.00% |
| offering products | 1 | 0.00% |
| offering services | 1 | 0.00% |
| Football Tournament | 1 | 0.00% |
| entities.social_media_platform | 1 | 0.00% |
| entities.number_of_backgrounds | 1 | 0.00% |
| investor_name | 1 | 0.00% |
| Written Content | 1 | 0.00% |
| Written Marketing Content | 1 | 0.00% |
| Delegation | 1 | 0.00% |
| visualization tool | 1 | 0.00% |
| furniture style | 1 | 0.00% |
| furniture brand | 1 | 0.00% |
| anonymity | 1 | 0.00% |
| Financial Info | 1 | 0.00% |
| speaker.name | 1 | 0.00% |
| software name | 1 | 0.00% |
| Bird image | 1 | 0.00% |
| Project/Programme | 1 | 0.00% |
| Building/Place | 1 | 0.00% |
| Settlement/Place | 1 | 0.00% |
| Measurement/Place | 1 | 0.00% |
| Business/Place | 1 | 0.00% |
| Service/Place | 1 | 0.00% |
| Region/Place | 1 | 0.00% |
| Country/Place | 1 | 0.00% |
| score: Final score of the match in set format | 1 | 0.00% |
| location: Venue where the match was held | 1 | 0.00% |
| winner: Player who won the match | 1 | 0.00% |
| venue_name: Name of the specific tournament venue | 1 | 0.00% |
| match_round: Stage of the tournament such as final, quarterfinal, or last 16 | 1 | 0.00% |
| Certification standard | 1 | 0.00% |
| estate features | 1 | 0.00% |
| location: City and state where company is headquartered | 1 | 0.00% |
| investors: List of venture capital firms involved | 1 | 0.00% |
| company: Name of the company raising capital | 1 | 0.00% |
| round: Series or type of funding round | 1 | 0.00% |
| amount: Total amount raised in millions | 1 | 0.00% |
| TV Drama | 1 | 0.00% |
| Organelle | 1 | 0.00% |
| Product Code/Identifier | 1 | 0.00% |
| Investment option | 1 | 0.00% |
| Property characteristic | 1 | 0.00% |
| Room/Location | 1 | 0.00% |
| Merchandise | 1 | 0.00% |
| Company/Assistance Provider | 1 | 0.00% |
| Continent/Geographical Location | 1 | 0.00% |
| sku: unique product identifier from supply chain database | 1 | 0.00% |
| product_name: name of the physical item such as earbuds or lantern | 1 | 0.00% |
| warehouse_id: unique identifier for the storage facility | 1 | 0.00% |
| warehouse_location: location name of the warehouse facility | 1 | 0.00% |
| quantity: number of units currently in stock | 1 | 0.00% |
| closing_price | 1 | 0.00% |
| price_change | 1 | 0.00% |
| Art Movement | 1 | 0.00% |
| root_cause: Identified cause of the outage or under investigation | 1 | 0.00% |
| provider_code: Service provider code assigned to the ticket | 1 | 0.00% |
| city: City where the customer resides | 1 | 0.00% |
| Game/Anime Title | 1 | 0.00% |
| Fanbase/Community | 1 | 0.00% |
| Anime Title | 1 | 0.00% |
| Athletic Team | 1 | 0.00% |
| Equipment/Tool | 1 | 0.00% |
| Device modification | 1 | 0.00% |
| System software | 1 | 0.00% |
| Custom recovery software | 1 | 0.00% |
| Mouse Poison | 1 | 0.00% |
| Rodent Bait Station | 1 | 0.00% |
| Product Package | 1 | 0.00% |
| Passenger Bus | 1 | 0.00% |
| Urban Transit Bus | 1 | 0.00% |
| Tourist Coach | 1 | 0.00% |
| Commercial Vehicle | 1 | 0.00% |
| Intercity Bus | 1 | 0.00% |
| Sleeper Coach | 1 | 0.00% |
| Podcast | 1 | 0.00% |
| Audio format | 1 | 0.00% |
| Download Option | 1 | 0.00% |
| Publication Volume | 1 | 0.00% |
| Research Group Name | 1 | 0.00% |
| Conference Date | 1 | 0.00% |
| Conference Location | 1 | 0.00% |
| Album/Project | 1 | 0.00% |
| Quote Collection | 1 | 0.00% |
| Goal/Achievement | 1 | 0.00% |
| economic policy | 1 | 0.00% |
| economic indicator | 1 | 0.00% |
| Classification System | 1 | 0.00% |
| Ethnicity/Nationality | 1 | 0.00% |
| Event/Phenomenon | 1 | 0.00% |
| geographic location/weather condition | 1 | 0.00% |
| Coupon | 1 | 0.00% |
| investment property | 1 | 0.00% |
| tenancy | 1 | 0.00% |
| Worship Flow | 1 | 0.00% |
| Worship Element | 1 | 0.00% |
| Holy Communion | 1 | 0.00% |
| Closing Ritual | 1 | 0.00% |
| Sacrament | 1 | 0.00% |
| year_founded | 1 | 0.00% |
| work_hours | 1 | 0.00% |
| funding_type | 1 | 0.00% |
| first_client | 1 | 0.00% |
| first_office | 1 | 0.00% |
| best practices | 1 | 0.00% |
| deletedBy | 1 | 0.00% |
| Occupation/Industry | 1 | 0.00% |
| shelves.count | 1 | 0.00% |
| warranty.duration | 1 | 0.00% |
| shelves.material | 1 | 0.00% |
| dimensions.rough_opening | 1 | 0.00% |
| dimensions.rough_opening_depth | 1 | 0.00% |
| degree | 1 | 0.00% |
| continent | 1 | 0.00% |
| vote_count | 1 | 0.00% |
| Character Class | 1 | 0.00% |
| Dice Roll | 1 | 0.00% |
| Character Level | 1 | 0.00% |
| Pagination | 1 | 0.00% |
| offer/benefit | 1 | 0.00% |
| currency/benefit | 1 | 0.00% |
| virtual world | 1 | 0.00% |
| perilous situation | 1 | 0.00% |
| video games | 1 | 0.00% |
| heroism | 1 | 0.00% |
| villainy | 1 | 0.00% |
| Yacht | 1 | 0.00% |
| gymnastics equipment | 1 | 0.00% |
| gymnastics facility | 1 | 0.00% |
| Home improvement product | 1 | 0.00% |
| Gift Ideas | 1 | 0.00% |
| publication.author.affiliation.location | 1 | 0.00% |
| morphology.wingLengthMm | 1 | 0.00% |
| publication.year | 1 | 0.00% |
| discovery.age.geologicalPeriod | 1 | 0.00% |
| discovery.geologicalFormation | 1 | 0.00% |
| publication.author.name | 1 | 0.00% |
| publication.author.deathYear | 1 | 0.00% |
| publication.author.affiliation.country | 1 | 0.00% |
| Catastrophe | 1 | 0.00% |
| workplace culture | 1 | 0.00% |
| workplace design | 1 | 0.00% |
| common area | 1 | 0.00% |
| Data Security | 1 | 0.00% |
| Security Threat | 1 | 0.00% |
| economy.primary_occupation | 1 | 0.00% |
| location.state | 1 | 0.00% |
| location.tehsil | 1 | 0.00% |
| economy.employment_source | 1 | 0.00% |
| Mortgage product | 1 | 0.00% |
| Mortgage backed security | 1 | 0.00% |
| Artwork feature | 1 | 0.00% |
| Distributor | 1 | 0.00% |
| Succession | 1 | 0.00% |
| fuel type | 1 | 0.00% |
| insurance coverage | 1 | 0.00% |
| business insurance | 1 | 0.00% |
| Relocation | 1 | 0.00% |
| Reestablishment | 1 | 0.00% |
| Financial issue | 1 | 0.00% |
| Production Process | 1 | 0.00% |
| mls_number | 1 | 0.00% |
| bathrooms | 1 | 0.00% |
| object/location | 1 | 0.00% |
| Interface Element | 1 | 0.00% |
| PC game | 1 | 0.00% |
| Game developer | 1 | 0.00% |
| authentication key | 1 | 0.00% |
| Demographics | 1 | 0.00% |
| Pet Kitten | 1 | 0.00% |
| Independent Kitten | 1 | 0.00% |
| forum section | 1 | 0.00% |
| collaborative project | 1 | 0.00% |
| legal document section | 1 | 0.00% |
| legal document exhibit | 1 | 0.00% |
| waste disposal equipment | 1 | 0.00% |
| Disease/Health Issue | 1 | 0.00% |
| Virtual Internet Service Provider | 1 | 0.00% |
| venue type | 1 | 0.00% |
| competition ranking | 1 | 0.00% |
| Purchase option | 1 | 0.00% |
| discovery_date | 1 | 0.00% |
| discovery_observatory | 1 | 0.00% |
| Industry Type | 1 | 0.00% |
| degree of variation | 1 | 0.00% |
| Social media content | 1 | 0.00% |
| Plumbing System | 1 | 0.00% |
| Plumbing Material | 1 | 0.00% |
| business location | 1 | 0.00% |
| person.death | 1 | 0.00% |
| Video quality | 1 | 0.00% |
| Video streaming | 1 | 0.00% |
| Social media channels | 1 | 0.00% |
| Lyricist | 1 | 0.00% |
| Producer | 1 | 0.00% |
| price range | 1 | 0.00% |
| website quality | 1 | 0.00% |
| Health Insurance Provider | 1 | 0.00% |
| Health Insurance Plans | 1 | 0.00% |
| Personal Health | 1 | 0.00% |
| lunch gathering | 1 | 0.00% |
| eating habit | 1 | 0.00% |
| dining out | 1 | 0.00% |
| unhealthy eating habit | 1 | 0.00% |
| Event/Trade Show | 1 | 0.00% |
| Health Points | 1 | 0.00% |
| Defense Equipment | 1 | 0.00% |
| Defense Attribute | 1 | 0.00% |
| food state | 1 | 0.00% |
| instruction step | 1 | 0.00% |
| Configuration | 1 | 0.00% |
| alert | 1 | 0.00% |
| meteorology | 1 | 0.00% |
| atmosphere/mood | 1 | 0.00% |
| production management technique | 1 | 0.00% |
| production issue | 1 | 0.00% |
| company attribute | 1 | 0.00% |
| market factor | 1 | 0.00% |
| production attribute | 1 | 0.00% |
| date: Date when the tournament took place | 1 | 0.00% |
| name: Full name of the esports tournament | 1 | 0.00% |
| location: Geographic location of the tournament | 1 | 0.00% |
| prize_pool: Total prize money in US dollars | 1 | 0.00% |
| poison-laced food | 1 | 0.00% |
| legendary animal | 1 | 0.00% |
| Perfume Product | 1 | 0.00% |
| Motorcycle seat | 1 | 0.00% |
| Unknown/Invalid entity type | 1 | 0.00% |
| Cleaning procedure | 1 | 0.00% |
| Blog website | 1 | 0.00% |
| Tile color | 1 | 0.00% |
| Filename | 1 | 0.00% |
| scenario | 1 | 0.00% |
| Music Publisher | 1 | 0.00% |
| Music Collection | 1 | 0.00% |
| Music Composer | 1 | 0.00% |
| Faulty Item | 1 | 0.00% |
| Exercise Program | 1 | 0.00% |
| show | 1 | 0.00% |
| sports training | 1 | 0.00% |
| sports activity | 1 | 0.00% |
| fitness training | 1 | 0.00% |
| sports level | 1 | 0.00% |
| rugby team | 1 | 0.00% |
| rugby teams | 1 | 0.00% |
| Art material | 1 | 0.00% |
| Event type | 1 | 0.00% |
| fire | 1 | 0.00% |
| explosion | 1 | 0.00% |
| chemicals | 1 | 0.00% |
| academic challenges | 1 | 0.00% |
| young people | 1 | 0.00% |
| emotions | 1 | 0.00% |
| childhood experiences | 1 | 0.00% |
| fearfulness | 1 | 0.00% |
| anger | 1 | 0.00% |
| author: Name of the author who wrote the digest | 1 | 0.00% |
| page: Page number where the digest appears | 1 | 0.00% |
| funding_body: Organization funding the publication | 1 | 0.00% |
| human interest | 1 | 0.00% |
| storytelling | 1 | 0.00% |
| current events | 1 | 0.00% |
| global perspective | 1 | 0.00% |
| Sports field element | 1 | 0.00% |
| Cricket time period | 1 | 0.00% |
| civilian | 1 | 0.00% |
| type of living organism | 1 | 0.00% |
| microorganisms | 1 | 0.00% |
| type of microorganism | 1 | 0.00% |
| policyholder: Name of the insurance claimant | 1 | 0.00% |
| policyholder_id: Unique identifier for the insurance policyholder | 1 | 0.00% |
| diagnosis_code: Medical code associated with the claim condition | 1 | 0.00% |
| claim_type: Type of medical procedure covered | 1 | 0.00% |
| address: Full address of the policyholder | 1 | 0.00% |
| administrative office | 1 | 0.00% |
| asteroid.discovery.year | 1 | 0.00% |
| Social/Community | 1 | 0.00% |
| Trading Tool | 1 | 0.00% |
| event/action | 1 | 0.00% |
| fandom character | 1 | 0.00% |
| communication style | 1 | 0.00% |
| data transmission | 1 | 0.00% |
| Car Manufacturer | 1 | 0.00% |
| business/brand | 1 | 0.00% |
| Navigation System | 1 | 0.00% |
| Power supply parameters | 1 | 0.00% |
| Voltage parameter | 1 | 0.00% |
| Current parameter | 1 | 0.00% |
| Load parameter | 1 | 0.00% |
| negative entity | 1 | 0.00% |
| Person/Proper Noun | 1 | 0.00% |
| Location/Proper Noun | 1 | 0.00% |
| Education_Level | 1 | 0.00% |
| Investment Group | 1 | 0.00% |
| Trading Group | 1 | 0.00% |
| Patient | 1 | 0.00% |
| Proportion of individuals | 1 | 0.00% |
| Chemical formula | 1 | 0.00% |
| Championship | 1 | 0.00% |
| realtor | 1 | 0.00% |
| conference_name | 1 | 0.00% |
| Driving Behavior | 1 | 0.00% |
| Wisdom | 1 | 0.00% |
| emotional support | 1 | 0.00% |
| Crafting Supplies | 1 | 0.00% |
| Holiday Tradition/Celebration | 1 | 0.00% |
| Technology/Design | 1 | 0.00% |
| Trail system | 1 | 0.00% |
| Access permit | 1 | 0.00% |
| Financial object | 1 | 0.00% |
| Entity Type: Baseball Players | 1 | 0.00% |
| Entity Type: Athletes | 1 | 0.00% |
| Music related entity | 1 | 0.00% |
| Occupation entity | 1 | 0.00% |
| Memory Aid | 1 | 0.00% |
| Music Professional | 1 | 0.00% |
| Language Usage | 1 | 0.00% |
| Trait/Emotion | 1 | 0.00% |
| Motivation/Influence | 1 | 0.00% |
| Motivation/Action | 1 | 0.00% |
| media source | 1 | 0.00% |
| digital video recorder | 1 | 0.00% |
| data storage capacity | 1 | 0.00% |
| Photography technique | 1 | 0.00% |
| communication interface | 1 | 0.00% |
| configuration interface | 1 | 0.00% |
| software developer | 1 | 0.00% |
| software project | 1 | 0.00% |
| modifications | 1 | 0.00% |
| Singer/Performer | 1 | 0.00% |
| Hotel/Building | 1 | 0.00% |
| Government Act | 1 | 0.00% |
| Physical Constant | 1 | 0.00% |
| Geometrical Property | 1 | 0.00% |
| Optical Component | 1 | 0.00% |
| Religious building/place of worship | 1 | 0.00% |
| Real estate agency/company | 1 | 0.00% |
| Document Citation | 1 | 0.00% |
| donation | 1 | 0.00% |
| Experience/Opportunity | 1 | 0.00% |
| Performance/Event | 1 | 0.00% |
| Discussion/Activity | 1 | 0.00% |
| company/service | 1 | 0.00% |
| website language identifier | 1 | 0.00% |
| search engine tools | 1 | 0.00% |
| online journal | 1 | 0.00% |
| Device ID | 1 | 0.00% |
| Identification information | 1 | 0.00% |
| Authentication credentials | 1 | 0.00% |
| Service/application | 1 | 0.00% |
| brand symbol | 1 | 0.00% |
| maintenance equipment | 1 | 0.00% |
| maintenance personnel | 1 | 0.00% |
| reference material | 1 | 0.00% |
| TV show genre | 1 | 0.00% |
| distribution.regions.2.region_name | 1 | 0.00% |
| distribution.regions.0.region_name | 1 | 0.00% |
| distribution.regions.1.region_name | 1 | 0.00% |
| morphology.flower_heads | 1 | 0.00% |
| morphology.flowering_fruiting_period | 1 | 0.00% |
| morphology.seeds | 1 | 0.00% |
| portion size | 1 | 0.00% |
| storage device | 1 | 0.00% |
| test_date | 1 | 0.00% |
| lab_id | 1 | 0.00% |
| Gaming Tournament | 1 | 0.00% |
| People/Community | 1 | 0.00% |
| Exam/Qualification | 1 | 0.00% |
| Attitude/Behavior | 1 | 0.00% |
| Coin type | 1 | 0.00% |
| Conversion Type | 1 | 0.00% |
| molecular_weight: Molecular weight of the drug in daltons | 1 | 0.00% |
| drug_name: Name of the pharmaceutical product | 1 | 0.00% |
| mechanismofaction: Biological process inhibited or modulated by the drug | 1 | 0.00% |
| lastreviewdate: Date the drug record was last reviewed | 1 | 0.00% |
| route: Administration method of the drug | 1 | 0.00% |
| target_receptor: Biological receptor the drug acts on | 1 | 0.00% |
| Service feature | 1 | 0.00% |
| Movie releases | 1 | 0.00% |
| Design/Pattern Name | 1 | 0.00% |
| listen | 1 | 0.00% |
| Desktop Theme | 1 | 0.00% |
| branding/logo | 1 | 0.00% |
| design/pattern | 1 | 0.00% |
| Artistic movement | 1 | 0.00% |
| charitable organization | 1 | 0.00% |
| Input Channel | 1 | 0.00% |
| Receiver Type | 1 | 0.00% |
| Port Type | 1 | 0.00% |
| Module | 1 | 0.00% |
| Display | 1 | 0.00% |
| date range | 1 | 0.00% |
| Health practice | 1 | 0.00% |
| dance group | 1 | 0.00% |
| musical notation | 1 | 0.00% |
| Ingredient/Flavor | 1 | 0.00% |
| quality standard | 1 | 0.00% |
| business growth | 1 | 0.00% |
| financial plan | 1 | 0.00% |
| cost reduction | 1 | 0.00% |
| traveler_name | 1 | 0.00% |
| discovery.survey | 1 | 0.00% |
| orbital_characteristics.database | 1 | 0.00% |
| physical_characteristics.publication_year | 1 | 0.00% |
| decision/action | 1 | 0.00% |
| craft | 1 | 0.00% |
| designer | 1 | 0.00% |
| sales team | 1 | 0.00% |
| vm_id | 1 | 0.00% |
| created_at | 1 | 0.00% |
| size_bytes | 1 | 0.00% |
| geographic group | 1 | 0.00% |
| personal hygiene product | 1 | 0.00% |
| Technology Tool | 1 | 0.00% |
| Hindu deity | 1 | 0.00% |
| type of medication | 1 | 0.00% |
| criminal act | 1 | 0.00% |
| place of employment | 1 | 0.00% |
| job subscription | 1 | 0.00% |
| natural phenomena | 1 | 0.00% |
| Park/Recreation Area | 1 | 0.00% |
| Fitness and Health | 1 | 0.00% |
| Car model and year | 1 | 0.00% |
| Diagram | 1 | 0.00% |
| Sale Event | 1 | 0.00% |
| skill/technique | 1 | 0.00% |
| procedural steps | 1 | 0.00% |
| recycling | 1 | 0.00% |
| resource management | 1 | 0.00% |
| Children | 1 | 0.00% |
| company/service provider | 1 | 0.00% |
| procedure/plan | 1 | 0.00% |
| acquaintance | 1 | 0.00% |
| global activity | 1 | 0.00% |
| genetic entity | 1 | 0.00% |
| Product or service | 1 | 0.00% |
| Safety device | 1 | 0.00% |
| Photography website | 1 | 0.00% |
| interest/love | 1 | 0.00% |
| musical genre | 1 | 0.00% |
| Certification/Training | 1 | 0.00% |
| bank | 1 | 0.00% |
| Marketing Tool | 1 | 0.00% |
| Maintenance Entity | 1 | 0.00% |
| Movement/Technique | 1 | 0.00% |
| Focus/Teaching Method | 1 | 0.00% |
| Body Part/Function | 1 | 0.00% |
| Activity/Type | 1 | 0.00% |
| Historical Reference | 1 | 0.00% |
| Winery Feature | 1 | 0.00% |
| educational level | 1 | 0.00% |
| platform: Type of social media post shared | 1 | 0.00% |
| author: Name of user who posted the content | 1 | 0.00% |
| Clothing/Jewelry | 1 | 0.00% |
| Event Supplies | 1 | 0.00% |
| Athlete/ MMA Fighter | 1 | 0.00% |
| Music/ Theme Song | 1 | 0.00% |
| Sports/ MMA Cage | 1 | 0.00% |
| Emotion/ Aggression | 1 | 0.00% |
| Airline company | 1 | 0.00% |
| Financial strategy | 1 | 0.00% |
| biology/technology | 1 | 0.00% |
| sports penalty | 1 | 0.00% |
| Location/Market | 1 | 0.00% |
| Traditional Dish | 1 | 0.00% |
| company culture | 1 | 0.00% |
| lease_end | 1 | 0.00% |
| offer_id | 1 | 0.00% |
| Event Service | 1 | 0.00% |
| punctuality | 1 | 0.00% |
| repair service | 1 | 0.00% |
| installation service | 1 | 0.00% |
| funding_id: Unique identifier for each funding round | 1 | 0.00% |
| amount: Total funding amount in USD with currency symbol | 1 | 0.00% |
| roundtype: Type of funding round | 1 | 0.00% |
| location: City and state where the company is headquartered | 1 | 0.00% |
| investor: Name of the venture capital firm that invested | 1 | 0.00% |
| Origin entity | 1 | 0.00% |
| Trade Fair | 1 | 0.00% |
| Transportation Technology | 1 | 0.00% |
| hotel.distance_from_airport.value | 1 | 0.00% |
| hotel.distance_from_city_center.unit | 1 | 0.00% |
| hotel.location.city | 1 | 0.00% |
| hotel.distance_from_airport.unit | 1 | 0.00% |
| hotel.rating | 1 | 0.00% |
| Product Code/Model Number | 1 | 0.00% |
| Legal Precariousness | 1 | 0.00% |
| Processor Speed | 1 | 0.00% |
| RAM Timings | 1 | 0.00% |
| Homeless individuals | 1 | 0.00% |
| Community leaders | 1 | 0.00% |
| Urban area | 1 | 0.00% |
| Address the issue | 1 | 0.00% |
| Avionics standard | 1 | 0.00% |
| Nav-Comm device | 1 | 0.00% |
| Control interface | 1 | 0.00% |
| Navigation mode | 1 | 0.00% |
| Natural Border | 1 | 0.00% |
| Military Music | 1 | 0.00% |
| academic department | 1 | 0.00% |
| academic school | 1 | 0.00% |
| country/people | 1 | 0.00% |
| plan/outline | 1 | 0.00% |
| person.birth_location.country | 1 | 0.00% |
| person.birth_location.country_extended | 1 | 0.00% |
| person.death_date | 1 | 0.00% |
| person.death_location.region | 1 | 0.00% |
| person.occupation | 1 | 0.00% |
| Experiment | 1 | 0.00% |
| field of study/science | 1 | 0.00% |
| negative quality/trait | 1 | 0.00% |
| positive quality/trait | 1 | 0.00% |
| Religious leaders | 1 | 0.00% |
| Religious buildings | 1 | 0.00% |
| Religious belief | 1 | 0.00% |
| geographic_region | 1 | 0.00% |
| connectivity | 1 | 0.00% |
| market_segment | 1 | 0.00% |
| Legal Position | 1 | 0.00% |
| School/Institution | 1 | 0.00% |
| Fishing Lure | 1 | 0.00% |
| Marine life | 1 | 0.00% |
| Punishment Method | 1 | 0.00% |
| weather/event | 1 | 0.00% |
| food/event | 1 | 0.00% |
| progress | 1 | 0.00% |
| scientific name | 1 | 0.00% |
| Technology/Access Control | 1 | 0.00% |
| Communication/Alerts | 1 | 0.00% |
| Communication/Messaging | 1 | 0.00% |
| Communication/Filtering | 1 | 0.00% |
| Output Quality | 1 | 0.00% |
| Shopping center | 1 | 0.00% |
| group of writers | 1 | 0.00% |
| genre of writing | 1 | 0.00% |
| medium of distribution | 1 | 0.00% |
| measure of distribution | 1 | 0.00% |
| personal correspondence | 1 | 0.00% |
| journalistic writing | 1 | 0.00% |
| visual documentation | 1 | 0.00% |
| cartographic representation | 1 | 0.00% |
| graphical representation | 1 | 0.00% |
| religious discourse | 1 | 0.00% |
| personal testimony | 1 | 0.00% |
| Routine/Custom | 1 | 0.00% |
| opportunity for improvement | 1 | 0.00% |
| Business Competitors | 1 | 0.00% |
| Business Objective | 1 | 0.00% |
| State of Confusion | 1 | 0.00% |
| Kitchen appliances | 1 | 0.00% |
| planting process | 1 | 0.00% |
| plant care | 1 | 0.00% |
| innovation type | 1 | 0.00% |
| personal feelings | 1 | 0.00% |
| asteroid_type | 1 | 0.00% |
| weight capacity | 1 | 0.00% |
| intellectual property | 1 | 0.00% |
| Interest Rate | 1 | 0.00% |
| Maturity Date | 1 | 0.00% |
| taste perception | 1 | 0.00% |
| attack_method | 1 | 0.00% |
| Animal Product | 1 | 0.00% |
| color model | 1 | 0.00% |
| color space | 1 | 0.00% |
| wavelength | 1 | 0.00% |
| Music Format | 1 | 0.00% |
| inspectionTerms | 1 | 0.00% |
| warrantyTerms | 1 | 0.00% |
| governingLaw | 1 | 0.00% |
| performance location | 1 | 0.00% |
| habit | 1 | 0.00% |
| Medical Services | 1 | 0.00% |
| Policy/Regulation | 1 | 0.00% |
| Industry/Infrastructure | 1 | 0.00% |
| paralympic_games.year | 1 | 0.00% |
| delegation.country | 1 | 0.00% |
| paralympic_games.season | 1 | 0.00% |
| Ritual Space | 1 | 0.00% |
| Spiritual Entities | 1 | 0.00% |
| Genealogical Group | 1 | 0.00% |
| State Before Death | 1 | 0.00% |
| odds | 1 | 0.00% |
| horse race | 1 | 0.00% |
| horse racing classification | 1 | 0.00% |
| Sales funnel builder | 1 | 0.00% |
| screen size | 1 | 0.00% |
| audio format | 1 | 0.00% |
| profession/occupation support | 1 | 0.00% |
| job search tools and aids | 1 | 0.00% |
| regulations and policies | 1 | 0.00% |
| Vehicle modification | 1 | 0.00% |
| healing process | 1 | 0.00% |
| claimant_name | 1 | 0.00% |
| Event/Protest | 1 | 0.00% |
| graphical element | 1 | 0.00% |
| digital image | 1 | 0.00% |
| Ticket Website | 1 | 0.00% |
| occupation/hobby | 1 | 0.00% |
| building problem | 1 | 0.00% |
| Podcast/Show | 1 | 0.00% |
| User/Player | 1 | 0.00% |
| discovery_details.observatory.location.state | 1 | 0.00% |
| orbital_characteristics.jpl_classification | 1 | 0.00% |
| physical_characteristics.observations.0.telescope.year_launched | 1 | 0.00% |
| discovery_details.observatory.name | 1 | 0.00% |
| industry participant | 1 | 0.00% |
| market participants | 1 | 0.00% |
| work unit | 1 | 0.00% |
| fastener | 1 | 0.00% |
| Yoga Gesture | 1 | 0.00% |
| Hand Gesture | 1 | 0.00% |
| Healthcare stocks | 1 | 0.00% |
| finding_id | 1 | 0.00% |
| access_duration | 1 | 0.00% |
| access_type | 1 | 0.00% |
| data_code | 1 | 0.00% |
| Exam Material | 1 | 0.00% |
| Collectible Item | 1 | 0.00% |
| contact address | 1 | 0.00% |
| financial record keeping | 1 | 0.00% |
| Health Statistic | 1 | 0.00% |
| Blog Post | 1 | 0.00% |
| Blog Website | 1 | 0.00% |
| Reading Activity | 1 | 0.00% |
| raw material | 1 | 0.00% |
| Employer | 1 | 0.00% |
| difficulty/problem | 1 | 0.00% |
| social needs | 1 | 0.00% |
| emotional needs | 1 | 0.00% |
| logistical needs | 1 | 0.00% |
| Guideline/Standard | 1 | 0.00% |
| motivational term | 1 | 0.00% |
| Laptop Model | 1 | 0.00% |
| Government Committee | 1 | 0.00% |
| Government Report | 1 | 0.00% |
| Government Legislation | 1 | 0.00% |
| Number of Vaccinated People | 1 | 0.00% |
| printing supply | 1 | 0.00% |
| printing limit/consumable quantity | 1 | 0.00% |
| printing quality | 1 | 0.00% |
| Indoor Plants | 1 | 0.00% |
| Outdoor furniture | 1 | 0.00% |
| Youtube channel | 1 | 0.00% |
| Show title | 1 | 0.00% |
| Entity type (music) | 1 | 0.00% |
| commercial area | 1 | 0.00% |
| fitness center | 1 | 0.00% |
| Philosophy/Worldview | 1 | 0.00% |
| Warranty Terms | 1 | 0.00% |
| Warrior race | 1 | 0.00% |
| Suits of armor | 1 | 0.00% |
| Chess opening | 1 | 0.00% |
| Uncertainty/inaccuracy | 1 | 0.00% |
| Biomechanical signal | 1 | 0.00% |
| Disc metabolism | 1 | 0.00% |
| Pathology | 1 | 0.00% |
| XAML element | 1 | 0.00% |
| hotelAddress | 1 | 0.00% |
| problem/challenge | 1 | 0.00% |
| substances | 1 | 0.00% |
| visual ability | 1 | 0.00% |
| celebrity influencer | 1 | 0.00% |
| social media influencer | 1 | 0.00% |
| demographic/group | 1 | 0.00% |
| unspecified | 1 | 0.00% |
| Personal care product | 1 | 0.00% |
| Observatory | 1 | 0.00% |
| organizational element | 1 | 0.00% |
| dispute_status | 1 | 0.00% |
| owner_location | 1 | 0.00% |
| General term | 1 | 0.00% |
| storage devices | 1 | 0.00% |
| computer partitions | 1 | 0.00% |
| research type | 1 | 0.00% |
| access type | 1 | 0.00% |
| Visual entity | 1 | 0.00% |
| Descriptive entity | 1 | 0.00% |
| drawbacks | 1 | 0.00% |
| Website Feature | 1 | 0.00% |
| product.description | 1 | 0.00% |
| standards | 1 | 0.00% |
| editor/writer of introduction | 1 | 0.00% |
| medicine/drug | 1 | 0.00% |
| opinion | 1 | 0.00% |
| self-awareness | 1 | 0.00% |
| potentiality | 1 | 0.00% |
| maximum potential | 1 | 0.00% |
| behavior/activity | 1 | 0.00% |
| digital container | 1 | 0.00% |
| duplicate | 1 | 0.00% |
| Performance feature | 1 | 0.00% |
| Fish species | 1 | 0.00% |
| Architectural design | 1 | 0.00% |
| Sports stadium | 1 | 0.00% |
| Calendar day | 1 | 0.00% |
| company/brand | 1 | 0.00% |
| project_name: Name of the technology project | 1 | 0.00% |
| funding: Total funding amount in millions | 1 | 0.00% |
| accuracy_efficiency: Performance metric like accuracy or efficiency | 1 | 0.00% |
| Fun Competition | 1 | 0.00% |
| function call | 1 | 0.00% |
| object instance | 1 | 0.00% |
| Graphics style | 1 | 0.00% |
| Game location/theme | 1 | 0.00% |
| Score tracking system | 1 | 0.00% |
| Online platforms | 1 | 0.00% |
| Item/Service | 1 | 0.00% |
| initiative/project | 1 | 0.00% |
| Marriage Duration | 1 | 0.00% |
| studio | 1 | 0.00% |
| superhero | 1 | 0.00% |
| comic book series | 1 | 0.00% |
| assurance | 1 | 0.00% |
| Movie title/Entertainment | 1 | 0.00% |
| Person/Entertainment | 1 | 0.00% |
| exclusion | 1 | 0.00% |
| Financial Benchmark | 1 | 0.00% |
| Regulating Body | 1 | 0.00% |
| luxury | 1 | 0.00% |
| Educational Specialization | 1 | 0.00% |
| Location/Action | 1 | 0.00% |
| funding_threshold | 1 | 0.00% |
| committee | 1 | 0.00% |
| voter_name | 1 | 0.00% |
| vote_outcome | 1 | 0.00% |
| Racing vehicle | 1 | 0.00% |
| Ranking position | 1 | 0.00% |
| Competition/race participant | 1 | 0.00% |
| Knocked down | 1 | 0.00% |
| Aggressive behavior | 1 | 0.00% |
| Accelerate quickly | 1 | 0.00% |
| Emotion/Expression | 1 | 0.00% |
| project/creative work | 1 | 0.00% |
| Communication Infrastructure | 1 | 0.00% |
| Data Processing Entity | 1 | 0.00% |
| Security Entity | 1 | 0.00% |
| Report Title | 1 | 0.00% |
| Program Name | 1 | 0.00% |
| experience/opportunity | 1 | 0.00% |
| nature/scenery | 1 | 0.00% |
| cultural/historical events | 1 | 0.00% |
| landmarks/points of interest | 1 | 0.00% |
| party: Political party of the winner | 1 | 0.00% |
| opponent: Candidate who lost the district | 1 | 0.00% |
| Rule | 1 | 0.00% |
| cultural reference | 1 | 0.00% |
| Event/Attraction | 1 | 0.00% |
| Hotel/Casino | 1 | 0.00% |
| Racecourse | 1 | 0.00% |
| Tables | 1 | 0.00% |
| language learners | 1 | 0.00% |
| skin type | 1 | 0.00% |
| vitamin | 1 | 0.00% |
| health indicator | 1 | 0.00% |
| completion | 1 | 0.00% |
| Car rental service | 1 | 0.00% |
| match_id: Unique identifier for the match event | 1 | 0.00% |
| team1: First team participating in the match | 1 | 0.00% |
| team2: Second team participating in the match | 1 | 0.00% |
| date: Date the match was scheduled or played | 1 | 0.00% |
| location: Venue where the match took place | 1 | 0.00% |
| status: Current match status | 1 | 0.00% |
| Directory item | 1 | 0.00% |
| belief/idea | 1 | 0.00% |
| TV show/music | 1 | 0.00% |
| Composer/musician | 1 | 0.00% |
| academic period | 1 | 0.00% |
| Academic records | 1 | 0.00% |
| Financial status | 1 | 0.00% |
| project/endeavor | 1 | 0.00% |
| goal/aim | 1 | 0.00% |
| Clothing brand abbreviation | 1 | 0.00% |
| Client | 1 | 0.00% |
| Sports Tournament | 1 | 0.00% |
| Sports Group | 1 | 0.00% |
| Redesign/Makeover | 1 | 0.00% |
| credits.creation_date | 1 | 0.00% |
| credits.organization | 1 | 0.00% |
| credits.funding | 1 | 0.00% |
| credits.advertising_revenue_use | 1 | 0.00% |
| credits.version | 1 | 0.00% |
| credits.contributors.0.role | 1 | 0.00% |
| credits.contributors.0.person | 1 | 0.00% |
| credits.contributors.1.role | 1 | 0.00% |
| credits.contributors.1.person | 1 | 0.00% |
| credits.graphic_design.0.designer | 1 | 0.00% |
| credits.graphic_design.0.copyright_year | 1 | 0.00% |
| credits.graphic_design.0.area | 1 | 0.00% |
| credits.graphic_design.1.designer | 1 | 0.00% |
| credits.graphic_design.1.copyright_year | 1 | 0.00% |
| credits.graphic_design.1.area | 1 | 0.00% |
| company.legal_type | 1 | 0.00% |
| company.history.0.year | 1 | 0.00% |
| company.history.0.event | 1 | 0.00% |
| company.employee_count | 1 | 0.00% |
| company.production_capacity.unit | 1 | 0.00% |
| company.founders.0.number | 1 | 0.00% |
| company.production_capacity.frequency | 1 | 0.00% |
| company.quality_standards | 1 | 0.00% |
| Ideology | 1 | 0.00% |
| Art/Design | 1 | 0.00% |
| Creativity/Innovation | 1 | 0.00% |
| medical_professional | 1 | 0.00% |
| medical_condition | 1 | 0.00% |
| emergency_service | 1 | 0.00% |
| Genetic Marker | 1 | 0.00% |
| financial liability | 1 | 0.00% |
| label/identifier | 1 | 0.00% |
| Media Exposure | 1 | 0.00% |
| General Group | 1 | 0.00% |
| UI Element | 1 | 0.00% |
| writing instrument part | 1 | 0.00% |
| Infrastructure Issue | 1 | 0.00% |
| Facial expression | 1 | 0.00% |
| Communication style | 1 | 0.00% |
| description.growth_habit | 1 | 0.00% |
| description.rosette | 1 | 0.00% |
| description.leaves.color | 1 | 0.00% |
| description.leaves.shape | 1 | 0.00% |
| description.leaves.surface | 1 | 0.00% |
| description.leaves.length_cm.min | 1 | 0.00% |
| description.leaves.length_cm.max | 1 | 0.00% |
| description.leaves.width_cm.min | 1 | 0.00% |
| description.leaves.width_cm.max | 1 | 0.00% |
| description.leaves.margins | 1 | 0.00% |
| description.leaves.terminal_spine.color | 1 | 0.00% |
| description.inflorescence.type | 1 | 0.00% |
| description.inflorescence.shape | 1 | 0.00% |
| description.inflorescence.height_m.min | 1 | 0.00% |
| description.inflorescence.height_m.max | 1 | 0.00% |
| description.flowers.color | 1 | 0.00% |
| description.flowers.length_mm.min | 1 | 0.00% |
| description.flowers.length_mm.max | 1 | 0.00% |
| description.flowers.arrangement | 1 | 0.00% |
| description.flowers.perianth_tube_length_mm.min | 1 | 0.00% |
| description.flowers.perianth_tube_length_mm.max | 1 | 0.00% |
| description.fruit.type | 1 | 0.00% |
| description.fruit.shape | 1 | 0.00% |
| description.fruit.color | 1 | 0.00% |
| description.fruit.length_mm.min | 1 | 0.00% |
| description.fruit.length_mm.max | 1 | 0.00% |
| description.fruit.width_mm.min | 1 | 0.00% |
| description.fruit.width_mm.max | 1 | 0.00% |
| description.seeds.color | 1 | 0.00% |
| description.flowering_time | 1 | 0.00% |
| common_names | 1 | 0.00% |
| systematics_and_distribution.habitat | 1 | 0.00% |
| systematics_and_distribution.location.country | 1 | 0.00% |
| systematics_and_distribution.location.state | 1 | 0.00% |
| systematics_and_distribution.location.elevation_m.min | 1 | 0.00% |
| systematics_and_distribution.location.elevation_m.max | 1 | 0.00% |
| systematics_and_distribution.first_description.author | 1 | 0.00% |
| systematics_and_distribution.first_description.year | 1 | 0.00% |
| systematics_and_distribution.group | 1 | 0.00% |
| systematics_and_distribution.similar_species | 1 | 0.00% |
| systematics_and_distribution.distinguishing_features | 1 | 0.00% |
| Diet plan | 1 | 0.00% |
| Body Function | 1 | 0.00% |
| Facial Expression | 1 | 0.00% |
| foundDate | 1 | 0.00% |
| foundLocation | 1 | 0.00% |
| Organization/Position | 1 | 0.00% |
| Safety and Security | 1 | 0.00% |
| Tree Species | 1 | 0.00% |
| DateTime | 1 | 0.00% |
| important_notice | 1 | 0.00% |
| priority_registration.deadline | 1 | 0.00% |
| priority_registration.instructions | 1 | 0.00% |
| on_site_registration.note | 1 | 0.00% |
| on_site_registration.description | 1 | 0.00% |
| on_site_registration.location.name | 1 | 0.00% |
| Media franchise | 1 | 0.00% |
| Narrative element | 1 | 0.00% |
| Interactive element | 1 | 0.00% |
| Comic Character | 1 | 0.00% |
| Time/Calendar | 1 | 0.00% |
| Promotional Code | 1 | 0.00% |
| website content | 1 | 0.00% |
| personal website | 1 | 0.00% |
| precious stones | 1 | 0.00% |
| Part measurement | 1 | 0.00% |
| Health Entity | 1 | 0.00% |
| alternate_designations | 1 | 0.00% |
| orbit.semi_major_axis.unit | 1 | 0.00% |
| physical_properties.diameter.unit | 1 | 0.00% |
| orbit.aphelion.unit | 1 | 0.00% |
| Genetic condition | 1 | 0.00% |
| Score Value | 1 | 0.00% |
| Academic Course | 1 | 0.00% |
| Educational Institutions | 1 | 0.00% |
| Event Series | 1 | 0.00% |
| Pronoun <> | 1 | 0.00% |
| Machine | 1 | 0.00% |
| natural compound | 1 | 0.00% |
| product/service feature | 1 | 0.00% |
| sales_volume | 1 | 0.00% |
| date_sold | 1 | 0.00% |
| audio technology | 1 | 0.00% |
| film technology | 1 | 0.00% |
| terminationDate | 1 | 0.00% |
| terminationReason | 1 | 0.00% |
| job title/occupation | 1 | 0.00% |
| Unclear/Unknown | 1 | 0.00% |
| rate chart | 1 | 0.00% |
| Fear/Phobia | 1 | 0.00% |
| customer support contact | 1 | 0.00% |
| Key Blank | 1 | 0.00% |
| Padlocks | 1 | 0.00% |
| TV show episodes | 1 | 0.00% |
| page | 1 | 0.00% |
| harm | 1 | 0.00% |
| broken system | 1 | 0.00% |
| animal/pest | 1 | 0.00% |
| healthcare professional | 1 | 0.00% |
| Pump feature | 1 | 0.00% |
| Pump type | 1 | 0.00% |
| Person/Writer | 1 | 0.00% |
| Person/Athlete | 1 | 0.00% |
| Short Film/Entertainment | 1 | 0.00% |
| Account | 1 | 0.00% |
| disclaimer | 1 | 0.00% |
| item to avoid | 1 | 0.00% |
| item to give | 1 | 0.00% |
| fitness equipment | 1 | 0.00% |
| equipment part | 1 | 0.00% |
| exercise position | 1 | 0.00% |
| database management system | 1 | 0.00% |
| group/person | 1 | 0.00% |
| website name | 1 | 0.00% |
| security breach | 1 | 0.00% |
| payment amount | 1 | 0.00% |
| Fictional Character/Movie/Comic Book | 1 | 0.00% |
| registration | 1 | 0.00% |
| Cannabis strain | 1 | 0.00% |
| Cannabinoid concentration | 1 | 0.00% |
| crafting supplies | 1 | 0.00% |
| financial benefit | 1 | 0.00% |
| Style/Genre | 1 | 0.00% |
| Movement/Action | 1 | 0.00% |
| temporary residence | 1 | 0.00% |
| permanent residence | 1 | 0.00% |
| process/procedure | 1 | 0.00% |
| product standard | 1 | 0.00% |
| project schedule | 1 | 0.00% |
| quality assurance | 1 | 0.00% |
| Stock image website | 1 | 0.00% |
| Event management website | 1 | 0.00% |
| player_action: Description of player's contribution such as goal or interception | 1 | 0.00% |
| home_team: Team playing at home stadium | 1 | 0.00% |
| match_location: Stadium or venue where match occurred | 1 | 0.00% |
| away_team: Team playing away | 1 | 0.00% |
| src | 1 | 0.00% |
| alt | 1 | 0.00% |
| caption | 1 | 0.00% |
| therapy type | 1 | 0.00% |
| work philosophy | 1 | 0.00% |
| Product packaging | 1 | 0.00% |
| Performing arts event | 1 | 0.00% |
| Political Manipulation | 1 | 0.00% |
| Transformation | 1 | 0.00% |
| Fraudulent Activity | 1 | 0.00% |
| transfer_type | 1 | 0.00% |
| problem-solving technique | 1 | 0.00% |
| Business activity | 1 | 0.00% |
| Newsletter name | 1 | 0.00% |
| Store Name | 1 | 0.00% |
| game_status | 1 | 0.00% |
| event_type | 1 | 0.00% |
| news_id | 1 | 0.00% |
| setting.political_context | 1 | 0.00% |
| main_character.rank | 1 | 0.00% |
| publication_details.first_publication.year | 1 | 0.00% |
| publication_details.year_written | 1 | 0.00% |
| publication_details.first_publication_czechoslovakia | 1 | 0.00% |
| main_character.affiliation | 1 | 0.00% |
| original_title | 1 | 0.00% |
| Animation Technique | 1 | 0.00% |
| Charity event | 1 | 0.00% |
| Location/Area | 1 | 0.00% |
| Event/Exhibit | 1 | 0.00% |
| fictional character/object | 1 | 0.00% |
| educational tool | 1 | 0.00% |
| educational assessment | 1 | 0.00% |
| vocabulary words | 1 | 0.00% |
| Business Competitor | 1 | 0.00% |
| Business Opportunity | 1 | 0.00% |
| Business Impact | 1 | 0.00% |
| Stadium staff | 1 | 0.00% |
| purchaseOrderDate | 1 | 0.00% |
| shippingAddress | 1 | 0.00% |
| Cultural Movement | 1 | 0.00% |
| Exercise equipment | 1 | 0.00% |
| Legal documents | 1 | 0.00% |
| Cause of harm | 1 | 0.00% |
| Actor/Person | 1 | 0.00% |
| Fictional Character/Person | 1 | 0.00% |
| Christian meditation and prayer | 1 | 0.00% |
| Interfaith dialogue | 1 | 0.00% |
| Visa status | 1 | 0.00% |
| Education status | 1 | 0.00% |
| Academic progress | 1 | 0.00% |
| Work experience | 1 | 0.00% |
| command option | 1 | 0.00% |
| Quality/Validity | 1 | 0.00% |
| combat sports | 1 | 0.00% |
| martial artists | 1 | 0.00% |
| product_code | 1 | 0.00% |
| Stage of plant growth | 1 | 0.00% |
| Size/height | 1 | 0.00% |
| warehouse: Warehouse location where product is stored | 1 | 0.00% |
| price: Listed retail price in US dollars | 1 | 0.00% |
| rating: Customer rating out of 5 stars | 1 | 0.00% |
| sku: Unique product identifier as assigned in inventory | 1 | 0.00% |
| update_notes | 1 | 0.00% |
| CRM software | 1 | 0.00% |
| Task planner | 1 | 0.00% |
| Sports team position | 1 | 0.00% |
| Sports statistics | 1 | 0.00% |
| comfort and convenience | 1 | 0.00% |
| Dealership | 1 | 0.00% |
| experience/event | 1 | 0.00% |
| technical indicator | 1 | 0.00% |
| animate | 1 | 0.00% |
| departureScheduled | 1 | 0.00% |
| arrivalScheduled | 1 | 0.00% |
| notificationTime | 1 | 0.00% |
| feeling/experience | 1 | 0.00% |
| website link | 1 | 0.00% |
| plant entity | 1 | 0.00% |
| product entity | 1 | 0.00% |
| food entity | 1 | 0.00% |
| Protein/enzyme | 1 | 0.00% |
| Mobile Phone Model | 1 | 0.00% |
| Investment type | 1 | 0.00% |
| Facility type | 1 | 0.00% |
| Date/Month | 1 | 0.00% |
| Movie Rating | 1 | 0.00% |
| Dwelling place | 1 | 0.00% |
| Print Setting | 1 | 0.00% |
| marine animal | 1 | 0.00% |
| professional growth | 1 | 0.00% |
| job-related task | 1 | 0.00% |
| registration process | 1 | 0.00% |
| venture | 1 | 0.00% |
| herb/plant | 1 | 0.00% |
| kitchenware/container | 1 | 0.00% |
| treatment.type | 1 | 0.00% |
| discovery.name | 1 | 0.00% |
| mutation.effect | 1 | 0.00% |
| methodology.techniques | 1 | 0.00% |
| treatment.target | 1 | 0.00% |
| mutation.mechanism | 1 | 0.00% |
| discovery.function | 1 | 0.00% |
| disease.type | 1 | 0.00% |
| discovery.impact | 1 | 0.00% |
| treatment.effectiveness | 1 | 0.00% |
| line | 1 | 0.00% |
| Warranty Type | 1 | 0.00% |
| audience/type of user | 1 | 0.00% |
| entity name | 1 | 0.00% |
| diagnosis_code | 1 | 0.00% |
| lab_results | 1 | 0.00% |
| followupinstruction | 1 | 0.00% |
| Governmental Organization | 1 | 0.00% |
| Course/Exam | 1 | 0.00% |
| Platform/Tool | 1 | 0.00% |
| Platform/Tool Feature | 1 | 0.00% |
| Course/Exam Material | 1 | 0.00% |
| Emergency Locksmith Firm | 1 | 0.00% |
| Door Jam | 1 | 0.00% |
| Law Enforcement Personnel | 1 | 0.00% |
| Law Enforcement Division | 1 | 0.00% |
| Missing Persons | 1 | 0.00% |
| Criminal Investigations | 1 | 0.00% |
| Traffic Management | 1 | 0.00% |
| loan type | 1 | 0.00% |
| loan management | 1 | 0.00% |
| Trait/Behavior | 1 | 0.00% |
| Music Experience | 1 | 0.00% |
| Music Streaming Service | 1 | 0.00% |
| visual design element | 1 | 0.00% |
| textual content | 1 | 0.00% |
| design resource | 1 | 0.00% |
| validation status | 1 | 0.00% |
| usage statistics | 1 | 0.00% |
| Study strategy | 1 | 0.00% |
| title: Title of the video as presented in the description | 1 | 0.00% |
| upload_time: Time of upload in 12-hour format with AM/PM | 1 | 0.00% |
| duration: Duration of the video in minutes and seconds | 1 | 0.00% |
| likes: Number of likes the video has accumulated | 1 | 0.00% |
| comments: Number of comments on the video | 1 | 0.00% |
| location: Physical location where the video was filmed | 1 | 0.00% |
| creator: Name of the video creator or subject | 1 | 0.00% |
| materials_used: List of materials or components used in the video's creation | 1 | 0.00% |
| cost: Total cost of equipment or materials used in the video | 1 | 0.00% |
| Organization entity | 1 | 0.00% |
| Tax entity | 1 | 0.00% |
| event list | 1 | 0.00% |
| Information entity | 1 | 0.00% |
| workplace hazard | 1 | 0.00% |
| Ceremony element | 1 | 0.00% |
| Application/Project | 1 | 0.00% |
| sequence | 1 | 0.00% |
| material component | 1 | 0.00% |
| construction tool | 1 | 0.00% |
| equipmentCondition | 1 | 0.00% |
| dog food | 1 | 0.00% |
| feeding age | 1 | 0.00% |
| fee: Financial fee paid for filing the patent application | 1 | 0.00% |
| status: Current status of the patent application | 1 | 0.00% |
| protein structure | 1 | 0.00% |
| protein sequence analysis | 1 | 0.00% |
| history | 1 | 0.00% |
| Bible Version | 1 | 0.00% |
| Greek word | 1 | 0.00% |
| logistical problem | 1 | 0.00% |
| emergency facilities | 1 | 0.00% |
| emergency strategy | 1 | 0.00% |
| NBA Team | 1 | 0.00% |
| Company/Contractor | 1 | 0.00% |
| report_id | 1 | 0.00% |
| corrective_action | 1 | 0.00% |
| Action/Rule | 1 | 0.00% |
| industry/company | 1 | 0.00% |
| Negative verb phrase | 1 | 0.00% |
| content element | 1 | 0.00% |
| media content element | 1 | 0.00% |
| Organizational Structure | 1 | 0.00% |
| Performance/Measurement | 1 | 0.00% |
| customer acquisition | 1 | 0.00% |
| personal network | 1 | 0.00% |
| Tech services company | 1 | 0.00% |
| Bollywood actor | 1 | 0.00% |
| Indian film industry | 1 | 0.00% |
| Demographic/Personal Attribute | 1 | 0.00% |
| Network Protocol | 1 | 0.00% |
| IP Address | 1 | 0.00% |
| climate phenomenon | 1 | 0.00% |
| data reference | 1 | 0.00% |
| gren_id | 1 | 0.00% |
| mapping_period | 1 | 0.00% |
| aesthetic outcome | 1 | 0.00% |
| customer-owned media | 1 | 0.00% |
| legal consent | 1 | 0.00% |
| Factory | 1 | 0.00% |
| asteroid.number | 1 | 0.00% |
| fitness program | 1 | 0.00% |
| martial arts program | 1 | 0.00% |
| training facility | 1 | 0.00% |
| work opportunity | 1 | 0.00% |
| mode of communication | 1 | 0.00% |
| development framework | 1 | 0.00% |
| audio equipment component | 1 | 0.00% |
| system/environment | 1 | 0.00% |
| system/resource | 1 | 0.00% |
| gambling machine | 1 | 0.00% |
| product comparison | 1 | 0.00% |
| woodworking process | 1 | 0.00% |
| Work/Influence | 1 | 0.00% |
| furniture/woodwork | 1 | 0.00% |
| Mortgage Reduction Program | 1 | 0.00% |
| Home Equity | 1 | 0.00% |
| cholesterol: Cholesterol level in mg/dL if elevated | 1 | 0.00% |
| birth_date: Date of birth in MM/DD/YYYY format | 1 | 0.00% |
| age: Age in years as stated in the record | 1 | 0.00% |
| zip_code: Postal code of patient's address | 1 | 0.00% |
| state: State abbreviation of patient's residence | 1 | 0.00% |
| patient_id: Unique identifier assigned to each patient | 1 | 0.00% |
| treatment: Type of treatment or management plan | 1 | 0.00% |
| hba1c: Hemoglobin A1c percentage value | 1 | 0.00% |
| sports item | 1 | 0.00% |
| baseball part | 1 | 0.00% |
| geometric shape | 1 | 0.00% |
| material item | 1 | 0.00% |
| Competition participant | 1 | 0.00% |
| facility room | 1 | 0.00% |
| Web Development Framework | 1 | 0.00% |
| educational work | 1 | 0.00% |
| urban planning | 1 | 0.00% |
| technology solutions | 1 | 0.00% |
| environmental challenge | 1 | 0.00% |
| Event Setup | 1 | 0.00% |
| Opportunity/Prospect | 1 | 0.00% |
| Network/Community | 1 | 0.00% |
| Privacy/Security | 1 | 0.00% |
| Threats/Dangers | 1 | 0.00% |
| Opportunities | 1 | 0.00% |
| Exhibition Venue | 1 | 0.00% |
| Title/Qualification | 1 | 0.00% |
| software architecture | 1 | 0.00% |
| Idea/Advice | 1 | 0.00% |
| Plan/measurements | 1 | 0.00% |
| Fraternity | 1 | 0.00% |
| Ornament Option | 1 | 0.00% |
| Error Screenshot | 1 | 0.00% |
| Resource Link | 1 | 0.00% |
| buying process | 1 | 0.00% |
| Technical event | 1 | 0.00% |
| donor_address | 1 | 0.00% |
| disclosure_id | 1 | 0.00% |
| Publication/Document | 1 | 0.00% |
| population.age_range.max | 1 | 0.00% |
| study_metadata.measurement_technology.full_name | 1 | 0.00% |
| statistical_results.predictive_model.value | 1 | 0.00% |
| outcomes.data_source | 1 | 0.00% |
| statistical_results.diagnostic_metrics.annual_fracture_rates.1.rate_percentage | 1 | 0.00% |
| outcomes.fracture_events.0.type | 1 | 0.00% |
| follow_up.average_duration.years | 1 | 0.00% |
| population.gender | 1 | 0.00% |
| outcomes.fracture_events.0.count | 1 | 0.00% |
| population.age_range.unit | 1 | 0.00% |
| follow_up.average_duration.months | 1 | 0.00% |
| study_metadata.study_type | 1 | 0.00% |
| statistical_results.hazard_ratios.0.adjustment_factors | 1 | 0.00% |
| study_metadata.measurement_technology.abbreviation | 1 | 0.00% |
| statistical_results.diagnostic_metrics.sensitivity_at_threshold.percentage_of_fractures | 1 | 0.00% |
| statistical_results.diagnostic_metrics.annual_fracture_rates.0.rate_percentage | 1 | 0.00% |
| statistical_results.hazard_ratios.0.value | 1 | 0.00% |
| study_metadata.measurement_technology.anatomical_site | 1 | 0.00% |
| File sharing protocol | 1 | 0.00% |
| Download method | 1 | 0.00% |
| Pricing/Policy | 1 | 0.00% |
| medical condition status | 1 | 0.00% |
| Adventure Sports | 1 | 0.00% |
| personal career | 1 | 0.00% |
| cosmetic tool | 1 | 0.00% |
| cosmetic packaging | 1 | 0.00% |
| law enforcement strategy | 1 | 0.00% |
| geographical areas | 1 | 0.00% |
| issue or challenge | 1 | 0.00% |
| return_id: Unique identifier for the return transaction | 1 | 0.00% |
| returner: Name of the person initiating the return | 1 | 0.00% |
| location: City or address where return was submitted | 1 | 0.00% |
| item: Description of the returned product | 1 | 0.00% |
| amount: Monetary value of the return in dollars | 1 | 0.00% |
| status: Current status of the return approval | 1 | 0.00% |
| community/group | 1 | 0.00% |
| birthplace | 1 | 0.00% |
| lodging place | 1 | 0.00% |
| Cryptocurrency Hardware Wallet | 1 | 0.00% |
| Animal Jewelry | 1 | 0.00% |
| Physical attribute entity | 1 | 0.00% |
| Physical location entity | 1 | 0.00% |
| coverage_type: Type of insurance coverage provided | 1 | 0.00% |
| policy_id: Unique identifier for the insurance policy | 1 | 0.00% |
| renewal_date: Date when the policy is set to renew | 1 | 0.00% |
| notice_type: Method used to notify policyholder of renewal | 1 | 0.00% |
| address: Complete street address including city and state | 1 | 0.00% |
| location safety | 1 | 0.00% |
| security level | 1 | 0.00% |
| automotive part | 1 | 0.00% |
| Band/Group Name | 1 | 0.00% |
| Door | 1 | 0.00% |
| positive aspects | 1 | 0.00% |
| negative aspects | 1 | 0.00% |
| contemplation | 1 | 0.00% |
| Date entity | 1 | 0.00% |
| Archive entity | 1 | 0.00% |
| CMS solution | 1 | 0.00% |
| Hosting provider | 1 | 0.00% |
| Professional Services | 1 | 0.00% |
| Process/Functionality | 1 | 0.00% |
| Image Data | 1 | 0.00% |
| Computer Process | 1 | 0.00% |
| Process Status | 1 | 0.00% |
| Geolocation Data | 1 | 0.00% |
| activity/ritual | 1 | 0.00% |
| mental state/goal | 1 | 0.00% |
| identity | 1 | 0.00% |
| spiritual practice | 1 | 0.00% |
| Handbag Collection | 1 | 0.00% |
| commercial places | 1 | 0.00% |
| stock symbol/code | 1 | 0.00% |
| Hotel Chain | 1 | 0.00% |
| Color variation | 1 | 0.00% |
| kindness | 1 | 0.00% |
| Kitchen Feature | 1 | 0.00% |
| Outdoor Space | 1 | 0.00% |
| geographic classification | 1 | 0.00% |
| Plant taxa | 1 | 0.00% |
| Art/Music Style | 1 | 0.00% |
| aesthetic appeal | 1 | 0.00% |
| sender: Email of the person sending the confirmation or submission | 1 | 0.00% |
| recipient: Email of the person receiving the confirmation or submission | 1 | 0.00% |
| Community/Culture | 1 | 0.00% |
| Organization name | 1 | 0.00% |
| Lost City | 1 | 0.00% |
| Cafe/Store Name | 1 | 0.00% |
| Area/Neighborhood | 1 | 0.00% |
| fuel pipe | 1 | 0.00% |
| Hosted Service Operator | 1 | 0.00% |
| User Preference | 1 | 0.00% |
| Web Analytics | 1 | 0.00% |
| Safety Device | 1 | 0.00% |
| Policy/Program | 1 | 0.00% |
| airport code | 1 | 0.00% |
| Tree Name | 1 | 0.00% |
| Business Category | 1 | 0.00% |
| Human Society | 1 | 0.00% |
| indefinite | 1 | 0.00% |
| Index/Measurement | 1 | 0.00% |
| Bacteria Strain | 1 | 0.00% |
| operation_hours | 1 | 0.00% |
| hangul_name | 1 | 0.00% |
| hanja_name | 1 | 0.00% |
| kingdom | 1 | 0.00% |
| rank_level | 1 | 0.00% |
| rank_order | 1 | 0.00% |
| creation.year | 1 | 0.00% |
| creation.king | 1 | 0.00% |
| creation.king_reign_year | 1 | 0.00% |
| creation.event | 1 | 0.00% |
| details.uniform.robe_color | 1 | 0.00% |
| details.uniform.belt_color | 1 | 0.00% |
| details.uniform.belt_type | 1 | 0.00% |
| numerical entity | 1 | 0.00% |
| Section of a book | 1 | 0.00% |
| Distinctive attribute or aspect | 1 | 0.00% |
| Monitoring mechanism | 1 | 0.00% |
| Advancement or development | 1 | 0.00% |
| Phase or step | 1 | 0.00% |
| Transformation journey | 1 | 0.00% |
| Pet accessory | 1 | 0.00% |
| service/offer | 1 | 0.00% |
| hairstyle element | 1 | 0.00% |
| royalty | 1 | 0.00% |
| album or music project | 1 | 0.00% |
| song or track | 1 | 0.00% |
| Project Title | 1 | 0.00% |
| website changes | 1 | 0.00% |
| e_36 | 1 | 0.00% |
| e_37 | 1 | 0.00% |
| e_38 | 1 | 0.00% |
| e_39 | 1 | 0.00% |
| Wooded area | 1 | 0.00% |
| religious item | 1 | 0.00% |
| decorative item | 1 | 0.00% |
| natural item | 1 | 0.00% |
| Winery/Vineyard | 1 | 0.00% |
| Winemaker | 1 | 0.00% |
| Grape Variety | 1 | 0.00% |
| Wine Variations | 1 | 0.00% |
| Wine Year | 1 | 0.00% |
| Web analytics | 1 | 0.00% |
| Operational issue | 1 | 0.00% |
| Compliance | 1 | 0.00% |
| sport equipment | 1 | 0.00% |
| Website/Blog | 1 | 0.00% |
| Product Name/ID | 1 | 0.00% |
| establishment_year | 1 | 0.00% |
| staff.teaching_staff_count | 1 | 0.00% |
| faculty_name | 1 | 0.00% |
| map_location | 1 | 0.00% |
| campus | 1 | 0.00% |
| dean.name | 1 | 0.00% |
| Employment Status | 1 | 0.00% |
| Artifacts | 1 | 0.00% |
| Event Prediction | 1 | 0.00% |
| font family | 1 | 0.00% |
| Graduating student | 1 | 0.00% |
| Educator | 1 | 0.00% |
| Emotional response | 1 | 0.00% |
| Music instrument | 1 | 0.00% |
| company: Name of the organization hiring | 1 | 0.00% |
| location: City and state where the job is based | 1 | 0.00% |
| department: Business unit or division responsible for the role | 1 | 0.00% |
| required_qualifications: List of required skills or certifications | 1 | 0.00% |
| employee_id: Assigned employee ID for the role | 1 | 0.00% |
| data copies | 1 | 0.00% |
| Subdivision | 1 | 0.00% |
| Real estate database | 1 | 0.00% |
| PowerShell Command | 1 | 0.00% |
| Numeric entity | 1 | 0.00% |
| Crime entity | 1 | 0.00% |
| Law enforcement officer entity | 1 | 0.00% |
| Food Part | 1 | 0.00% |
| Service Offering<> ProvideSupport offering a free trial period of 10 days | 1 | 0.00% |
| Digital Account | 1 | 0.00% |
| Entertainment Industry | 1 | 0.00% |
| Ski resort | 1 | 0.00% |
| tool/object | 1 | 0.00% |
| task/event | 1 | 0.00% |
| Laundry items | 1 | 0.00% |
| Holiday Merchandise | 1 | 0.00% |
| Financial Benefit | 1 | 0.00% |
| festival_id | 1 | 0.00% |
| document security | 1 | 0.00% |
| document format | 1 | 0.00% |
| computer interface | 1 | 0.00% |
| TV Show/Media Outlet | 1 | 0.00% |
| Music Album/Song | 1 | 0.00% |
| programming feature | 1 | 0.00% |
| video game project | 1 | 0.00% |
| determination | 1 | 0.00% |
| Writing Guidelines | 1 | 0.00% |
| Personal Goal | 1 | 0.00% |
| Furniture item | 1 | 0.00% |
| Textile material | 1 | 0.00% |
| Watercraft | 1 | 0.00% |
| Career Development | 1 | 0.00% |
| Stage Light | 1 | 0.00% |
| Lighting equipment | 1 | 0.00% |
| Terrorist group member | 1 | 0.00% |
| Alias name | 1 | 0.00% |
| product bundle | 1 | 0.00% |
| purchasing option | 1 | 0.00% |
| resilience | 1 | 0.00% |
| Online discussion forum | 1 | 0.00% |
| network security policy | 1 | 0.00% |
| ambulance driver | 1 | 0.00% |
| Recovery | 1 | 0.00% |
| membership plan | 1 | 0.00% |
| renewal notice | 1 | 0.00% |
| financial resources | 1 | 0.00% |
| algorithm | 1 | 0.00% |
| protectionLevel | 1 | 0.00% |
| nextRotationTime | 1 | 0.00% |
| rotationPeriod | 1 | 0.00% |
| createTime | 1 | 0.00% |
| Ship type/Measurement unit | 1 | 0.00% |
| Legal code | 1 | 0.00% |
| Handmade item | 1 | 0.00% |
| Crafting material | 1 | 0.00% |
| Expertise | 1 | 0.00% |
| Service Agreement | 1 | 0.00% |
| trading strategy | 1 | 0.00% |
| trading term | 1 | 0.00% |
| Death | 1 | 0.00% |
| meal time | 1 | 0.00% |
| harmless adventure | 1 | 0.00% |
| Military rank | 1 | 0.00% |
| Terrorist group | 1 | 0.00% |
| Storage unit | 1 | 0.00% |
| Location in a house | 1 | 0.00% |
| Beach Name | 1 | 0.00% |
| Studio | 1 | 0.00% |
| cultural phenomenon | 1 | 0.00% |
| Ability or proficiency | 1 | 0.00% |
| Personal milestone | 1 | 0.00% |
| method/process | 1 | 0.00% |
| mindset/attitude | 1 | 0.00% |
| perspective/philosophy | 1 | 0.00% |
| ripple effects/consequences | 1 | 0.00% |
| company/business entity | 1 | 0.00% |
| artist_name | 1 | 0.00% |
| ticket_price | 1 | 0.00% |
| General | 1 | 0.00% |
| fruit/ingredient | 1 | 0.00% |
| ingredient/dish | 1 | 0.00% |
| Place/Service | 1 | 0.00% |
| Place/Organization | 1 | 0.00% |
| Attack Type | 1 | 0.00% |
| government initiative | 1 | 0.00% |
| country/city-state | 1 | 0.00% |
| Security Risk/Action | 1 | 0.00% |
| Recording Studio | 1 | 0.00% |
| learning activity | 1 | 0.00% |
| industry type | 1 | 0.00% |
| hotel_name | 1 | 0.00% |
| total_nights | 1 | 0.00% |
| total_cost | 1 | 0.00% |
| room_type | 1 | 0.00% |
| search type | 1 | 0.00% |
| airport location | 1 | 0.00% |
| type of fish | 1 | 0.00% |
| hair type | 1 | 0.00% |
| hair care | 1 | 0.00% |
| political contest | 1 | 0.00% |
| voting document | 1 | 0.00% |
| Cheating device | 1 | 0.00% |
| bird | 1 | 0.00% |
| School community | 1 | 0.00% |
| Building addition | 1 | 0.00% |
| yoga pose | 1 | 0.00% |
| physical therapy technique | 1 | 0.00% |
| physical fitness exercise | 1 | 0.00% |
| bone structure | 1 | 0.00% |
| job ID | 1 | 0.00% |
| department name | 1 | 0.00% |
| employment type | 1 | 0.00% |
| education requirement | 1 | 0.00% |
| experience requirement | 1 | 0.00% |
| opening quotation mark | 1 | 0.00% |
| verb, non-3rd person singular present | 1 | 0.00% |
| cardinal number | 1 | 0.00% |
| comparative adverb | 1 | 0.00% |
| closing quotation mark | 1 | 0.00% |
| website element | 1 | 0.00% |
| academic tasks | 1 | 0.00% |
| financial incentive | 1 | 0.00% |
| expense category | 1 | 0.00% |
| Record Type | 1 | 0.00% |
| Audience/Segment | 1 | 0.00% |
| review_period | 1 | 0.00% |
| ratingcategory | 1 | 0.00% |
| performance_score | 1 | 0.00% |
| moral principle | 1 | 0.00% |
| Beer Name | 1 | 0.00% |
| Digital distribution platform | 1 | 0.00% |
| enemy type | 1 | 0.00% |
| Ecological System | 1 | 0.00% |
| departure | 1 | 0.00% |
| arrival | 1 | 0.00% |
| Author/Poet | 1 | 0.00% |
| Social action | 1 | 0.00% |
| keynote_count | 1 | 0.00% |
| workshop_count | 1 | 0.00% |
| special_session | 1 | 0.00% |
| organizer | 1 | 0.00% |
| performance standard | 1 | 0.00% |
| evaluation process | 1 | 0.00% |
| smart phone | 1 | 0.00% |
| recognition tag | 1 | 0.00% |
| County Seat | 1 | 0.00% |
| action/process | 1 | 0.00% |
| Awareness Event | 1 | 0.00% |
| time/day | 1 | 0.00% |
| Video game modification | 1 | 0.00% |
| purchase action | 1 | 0.00% |
| Negative trend | 1 | 0.00% |
| List of standings | 1 | 0.00% |
| lease_agreements.number | 1 | 0.00% |
| vacancy_rate.rate | 1 | 0.00% |
| vacancy_rate.change | 1 | 0.00% |
| Art/Entertainment | 1 | 0.00% |
| Imprint | 1 | 0.00% |
| reviewText | 1 | 0.00% |
| school role | 1 | 0.00% |
| educational outcome | 1 | 0.00% |
| refugee camp | 1 | 0.00% |
| town | 1 | 0.00% |
| terrorist group | 1 | 0.00% |
| Title of Podcast | 1 | 0.00% |
| Medical Study | 1 | 0.00% |
| first person pronoun | 1 | 0.00% |
| Fertility Clinic | 1 | 0.00% |
| Medical Organization | 1 | 0.00% |
| service_type | 1 | 0.00% |
| Award Object | 1 | 0.00% |
| rating: Star rating given by the reviewer out of 5 | 1 | 0.00% |
| status: Current status of the review | 1 | 0.00% |
| model: Unique identifier or model number of the device | 1 | 0.00% |
| word_count: Number of words in the review text | 1 | 0.00% |
| camp type | 1 | 0.00% |
| Data integrity | 1 | 0.00% |
| Artistic Medium | 1 | 0.00% |
| Artistic Material | 1 | 0.00% |
| Microbiology | 1 | 0.00% |
| Bioinformatics | 1 | 0.00% |
| Trading position | 1 | 0.00% |
| Stock trading | 1 | 0.00% |
| Trading methodology | 1 | 0.00% |
| Apocalypse | 1 | 0.00% |
| Catastrophic event | 1 | 0.00% |
| End of the world events | 1 | 0.00% |
| Violent content | 1 | 0.00% |
| Offensive language | 1 | 0.00% |
| Design company | 1 | 0.00% |
| Typeface design | 1 | 0.00% |
| Typeface designer | 1 | 0.00% |
| Physical characteristic | 1 | 0.00% |
| report_date: Date when the issue was first reported | 1 | 0.00% |
| resolution_action: Type of action taken to resolve the issue | 1 | 0.00% |
| status: Current state of the ticket after resolution | 1 | 0.00% |
| ticket_id: Unique identifier for the support ticket | 1 | 0.00% |
| customer_email: Email address associated with the ticket | 1 | 0.00% |
| Pronoun Entity | 1 | 0.00% |
| former currency | 1 | 0.00% |
| current currency | 1 | 0.00% |
| Cultural Heritage | 1 | 0.00% |
| Holistic Wellness | 1 | 0.00% |
| Spa Treatments | 1 | 0.00% |
| environmental factors | 1 | 0.00% |
| permissions | 1 | 0.00% |
| discussion topic | 1 | 0.00% |
| forum feature | 1 | 0.00% |
| musical instrument part | 1 | 0.00% |
| cleaning solution | 1 | 0.00% |
| cleaning substance | 1 | 0.00% |
| cleaning action | 1 | 0.00% |
| line1 | 1 | 0.00% |
| dining options | 1 | 0.00% |
| commercial center | 1 | 0.00% |
| retail space | 1 | 0.00% |
| public space | 1 | 0.00% |
| e-commerce activity | 1 | 0.00% |
| customer service activity | 1 | 0.00% |
| Number of Rooms | 1 | 0.00% |
| Precipitation | 1 | 0.00% |
| Precipitation level | 1 | 0.00% |
| Waterfall Name | 1 | 0.00% |
| movement | 1 | 0.00% |
| increase | 1 | 0.00% |
| admin | 1 | 0.00% |
| Linguistic element | 1 | 0.00% |
| Time Availability | 1 | 0.00% |
| Service Location | 1 | 0.00% |
| Event/Show | 1 | 0.00% |
| Game Design Technique | 1 | 0.00% |
| archery equipment | 1 | 0.00% |
| Role/Job | 1 | 0.00% |
| Game characters | 1 | 0.00% |
| website or domain name | 1 | 0.00% |
| action or event | 1 | 0.00% |
| physical object or item | 1 | 0.00% |
| teaching staff | 1 | 0.00% |
| teaching style | 1 | 0.00% |
| musical chord | 1 | 0.00% |
| date: Date of transaction occurrence | 1 | 0.00% |
| account_number: Unique account identifier shared across entries | 1 | 0.00% |
| pickup_type: Type of service or pickup method | 1 | 0.00% |
| product series | 1 | 0.00% |
| part numbers | 1 | 0.00% |
| Online Discussion forum | 1 | 0.00% |
| Weekly event | 1 | 0.00% |
| Time slot | 1 | 0.00% |
| Featured game | 1 | 0.00% |
| Tailoring Service | 1 | 0.00% |
| Packaging Quality | 1 | 0.00% |
| home security | 1 | 0.00% |
| advertising/promotion | 1 | 0.00% |
| marketing/legal | 1 | 0.00% |
| General Population | 1 | 0.00% |
| Job Vacancy | 1 | 0.00% |
| Company abbreviation | 1 | 0.00% |
| nominationId | 1 | 0.00% |
| awardName | 1 | 0.00% |
| awardingBody | 1 | 0.00% |
| yearReleased | 1 | 0.00% |
| Award Name | 1 | 0.00% |
| cycling team members | 1 | 0.00% |
| cycling event | 1 | 0.00% |
| cycling distance measurement | 1 | 0.00% |
| group of cyclists | 1 | 0.00% |
| cyclist | 1 | 0.00% |
| Payment Service | 1 | 0.00% |
| Educational intervention | 1 | 0.00% |
| passing down traits | 1 | 0.00% |
| Ancient Civilization | 1 | 0.00% |
| Physical Phenomenon | 1 | 0.00% |
| Physical States | 1 | 0.00% |
| participant | 1 | 0.00% |
| duration: Number of weeks the program lasts | 1 | 0.00% |
| program_name: Name of the training program enrolled in | 1 | 0.00% |
| Email folder | 1 | 0.00% |
| year_level | 1 | 0.00% |
| Software improvement | 1 | 0.00% |
| Historical Year | 1 | 0.00% |
| Title/Religious Figure | 1 | 0.00% |
| Mobile phone identification number | 1 | 0.00% |
| Mobile phone brand and operating system | 1 | 0.00% |
| Online account for Google services | 1 | 0.00% |
| Email service by Google | 1 | 0.00% |
| First Information Report complaint | 1 | 0.00% |
| Mobile phone model | 1 | 0.00% |
| Central processing unit model | 1 | 0.00% |
| Random access memory capacity | 1 | 0.00% |
| Operating system version | 1 | 0.00% |
| Yearly Production | 1 | 0.00% |
| Increase in production | 1 | 0.00% |
| Product Price | 1 | 0.00% |
| Volume and Issue Number | 1 | 0.00% |
| State/Province | 1 | 0.00% |
| food storage | 1 | 0.00% |
| food preparation | 1 | 0.00% |
| Alcohol Brand | 1 | 0.00% |
| service.opening_hours.0.day | 1 | 0.00% |
| service.contact_information.email | 1 | 0.00% |
| service.location.building | 1 | 0.00% |
| service.responsible_person.role | 1 | 0.00% |
| service.problems_addressed | 1 | 0.00% |
| service.location.city | 1 | 0.00% |
| service.location.address | 1 | 0.00% |
| service.responsible_person.experience | 1 | 0.00% |
| floor maintenance | 1 | 0.00% |
| floor improvement | 1 | 0.00% |
| floor renovation | 1 | 0.00% |
| shop_name | 1 | 0.00% |
| shipping_information.payment_methods | 1 | 0.00% |
| shipping_information.bank_transfer_details.preferred_bank.bank_name | 1 | 0.00% |
| shipping_information.bank_transfer_details.preferred_bank.country | 1 | 0.00% |
| shipping_information.bank_transfer_details.preferred_bank.numeric_code | 1 | 0.00% |
| shipping_information.bank_transfer_details.preferred_bank.fee_remark | 1 | 0.00% |
| shipping_information.bank_transfer_details.sepa_transfer_fee.fee_amount | 1 | 0.00% |
| shipping_information.bank_transfer_details.sepa_transfer_fee.currency | 1 | 0.00% |
| shipping_information.bank_transfer_details.sepa_transfer_fee.condition | 1 | 0.00% |
| exchange_rate.from_currency | 1 | 0.00% |
| exchange_rate.to_currency | 1 | 0.00% |
| Graphics Library | 1 | 0.00% |
| Terrain Features | 1 | 0.00% |
| Tourists | 1 | 0.00% |
| Traditional Application | 1 | 0.00% |
| Product Range | 1 | 0.00% |
| Exterior Applications | 1 | 0.00% |
| camera settings | 1 | 0.00% |
| photographs | 1 | 0.00% |
| Emotion/State of Mind | 1 | 0.00% |
| test pattern | 1 | 0.00% |
| Gutter Fittings | 1 | 0.00% |
| work task | 1 | 0.00% |
| Automotive Repair Shop | 1 | 0.00% |
| Event Equipment | 1 | 0.00% |
| Sanitation Facilities | 1 | 0.00% |
| Public Space | 1 | 0.00% |
| Administrative Divisions | 1 | 0.00% |
| coupon code | 1 | 0.00% |
| reduced price | 1 | 0.00% |
| incentive | 1 | 0.00% |
| message_id | 1 | 0.00% |
| Technical Process | 1 | 0.00% |
| Animal Behavior | 1 | 0.00% |
| device interface | 1 | 0.00% |
| theme park | 1 | 0.00% |
| script name | 1 | 0.00% |
| location/obstacle | 1 | 0.00% |
| animal obstacle | 1 | 0.00% |
| Political philosophy | 1 | 0.00% |
| Healthcare Services | 1 | 0.00% |
| Wellness Practice | 1 | 0.00% |
| Song/Album title | 1 | 0.00% |
| Dress | 1 | 0.00% |
| Designer/Brand | 1 | 0.00% |
| website page | 1 | 0.00% |
| website technology | 1 | 0.00% |
| Currency amount | 1 | 0.00% |
| Lodging | 1 | 0.00% |
| Nationality/Passport type | 1 | 0.00% |
| Requirement/Criterion | 1 | 0.00% |
| storage containers | 1 | 0.00% |
| Social Norms | 1 | 0.00% |
| Causality | 1 | 0.00% |
| dietary restriction | 1 | 0.00% |
| Business/Brand/Person | 1 | 0.00% |
| name: Official title of the health care guideline | 1 | 0.00% |
| author: Name of the guideline's lead author | 1 | 0.00% |
| institution: Affiliation of the guideline's author | 1 | 0.00% |
| recommendation: Key clinical recommendation or action advised | 1 | 0.00% |
| region: Geographic area or setting where guideline applies | 1 | 0.00% |
| target_population: Group of patients or individuals the guideline targets | 1 | 0.00% |
| Travel Experience | 1 | 0.00% |
| toothpaste brand/type | 1 | 0.00% |
| body tissue | 1 | 0.00% |
| Vehicle Identification Number | 1 | 0.00% |
| Travelers/Pilgrims | 1 | 0.00% |
| Pilgrimage/Journey | 1 | 0.00% |
| Tour operator company | 1 | 0.00% |
| office_code | 1 | 0.00% |
| Achievement/Status | 1 | 0.00% |
| Limitation/Barrier | 1 | 0.00% |
| Cost-effective strategies | 1 | 0.00% |
| Service excellence | 1 | 0.00% |
| Value proposition | 1 | 0.00% |
| Living beings | 1 | 0.00% |
| Incident/accident | 1 | 0.00% |
| Requester | 1 | 0.00% |
| VCA card | 1 | 0.00% |
| Event/Occurrence | 1 | 0.00% |
| survey | 1 | 0.00% |
| annual report | 1 | 0.00% |
| Mozilla feature | 1 | 0.00% |
| Gender-based group | 1 | 0.00% |
| Sports club | 1 | 0.00% |
| Business Issue | 1 | 0.00% |
| Workplace Behavior | 1 | 0.00% |
| article_code | 1 | 0.00% |
| problem_description | 1 | 0.00% |
| affected_systems | 1 | 0.00% |
| business structure | 1 | 0.00% |
| symptom_date | 1 | 0.00% |
| graduation year | 1 | 0.00% |
| Athletes | 1 | 0.00% |
| Athletic events | 1 | 0.00% |
| bathroom accessory | 1 | 0.00% |
| ethical guide | 1 | 0.00% |
| Hurricane | 1 | 0.00% |
| Waste Material | 1 | 0.00% |
| trend | 1 | 0.00% |
| Title/Design Name | 1 | 0.00% |
| Contact number | 1 | 0.00% |
| Options | 1 | 0.00% |
| Customer concerns | 1 | 0.00% |
| Appointment scheduling | 1 | 0.00% |
| Task/project | 1 | 0.00% |
| software platform | 1 | 0.00% |
| software licensing | 1 | 0.00% |
| Game Version | 1 | 0.00% |
| Fashion Product | 1 | 0.00% |
| Advocacy | 1 | 0.00% |
| side effect | 1 | 0.00% |
| time-related entity | 1 | 0.00% |
| software interface | 1 | 0.00% |
| Football Stadium | 1 | 0.00% |
| reservationNumber | 1 | 0.00% |
| ticketClass | 1 | 0.00% |
| bookingDate | 1 | 0.00% |
| bookingTime | 1 | 0.00% |
| paymentMode | 1 | 0.00% |
| political division | 1 | 0.00% |
| property_id | 1 | 0.00% |
| listing_date | 1 | 0.00% |
| progression | 1 | 0.00% |
| cultural identity | 1 | 0.00% |
| language ability | 1 | 0.00% |
| family structure | 1 | 0.00% |
| Race track | 1 | 0.00% |
| Measuring Component | 1 | 0.00% |
| Container Material | 1 | 0.00% |
| posterSize | 1 | 0.00% |
| artworkType | 1 | 0.00% |
| designRequest | 1 | 0.00% |
| professional field | 1 | 0.00% |
| ability or skill | 1 | 0.00% |
| strong ability or skill | 1 | 0.00% |
| Business philosophy | 1 | 0.00% |
| property features | 1 | 0.00% |
| housing types | 1 | 0.00% |
| Nationality/Country of Origin | 1 | 0.00% |
| Nationality/Country of Residence | 1 | 0.00% |
| Cocktail Name | 1 | 0.00% |
| warning sign | 1 | 0.00% |
| Creative Effect/Style | 1 | 0.00% |
| Creative Project/Artwork | 1 | 0.00% |
| Pump | 1 | 0.00% |
| Cord length | 1 | 0.00% |
| Low wattage pump | 1 | 0.00% |
| Small pond | 1 | 0.00% |
| Architecture Style | 1 | 0.00% |
| medical institution | 1 | 0.00% |
| mythological creature | 1 | 0.00% |
| Furniture Store | 1 | 0.00% |
| Commercial Establishment | 1 | 0.00% |
| Design Plan | 1 | 0.00% |
| knowledge and experience | 1 | 0.00% |
| career goal | 1 | 0.00% |
| Cost/Price | 1 | 0.00% |
| jurisdiction: Court jurisdiction and division where case is filed | 1 | 0.00% |
| address: Physical address of the filing party | 1 | 0.00% |
| respondent: Name of the person or entity being sued | 1 | 0.00% |
| Dress Name | 1 | 0.00% |
| Iconic Actress | 1 | 0.00% |
| printing industry | 1 | 0.00% |
| outcome/consequence | 1 | 0.00% |
| downloadable content | 1 | 0.00% |
| team_size | 1 | 0.00% |
| Restaurant name | 1 | 0.00% |
| Food/drink | 1 | 0.00% |
| exercise routine | 1 | 0.00% |
| product catalogue | 1 | 0.00% |
| input | 1 | 0.00% |
| Personal Attribute | 1 | 0.00% |
| Benchmarking Tool | 1 | 0.00% |
| amount_usd | 1 | 0.00% |
| term_end_date | 1 | 0.00% |
| clause_description | 1 | 0.00% |
| version_number | 1 | 0.00% |
| attachment_type | 1 | 0.00% |
| document_type | 1 | 0.00% |
| reference_number | 1 | 0.00% |
| approval_level | 1 | 0.00% |
| modification_flag | 1 | 0.00% |
| audit_review_date | 1 | 0.00% |
| legal_agent | 1 | 0.00% |
| execution_date | 1 | 0.00% |
| notice_period_days | 1 | 0.00% |
| party_name | 1 | 0.00% |
| academic profession | 1 | 0.00% |
| collectible item | 1 | 0.00% |
| storage place | 1 | 0.00% |
| object/pronoun | 1 | 0.00% |
| pronoun <> | 1 | 0.00% |
| Religious Site | 1 | 0.00% |
| Time Span | 1 | 0.00% |
| mark | 1 | 0.00% |
| damages | 1 | 0.00% |
| Writing Materials | 1 | 0.00% |
| Research | 1 | 0.00% |
| Commercial establishment | 1 | 0.00% |
| offer_id: Unique identifier for each leasing offer | 1 | 0.00% |
| tenant: Name of the company occupying the leased space | 1 | 0.00% |
| location: Full address of the leased property | 1 | 0.00% |
| features: List of included amenities or space features | 1 | 0.00% |
| machine component | 1 | 0.00% |
| emotional baggage | 1 | 0.00% |
| consumer good | 1 | 0.00% |
| Degree abbreviation | 1 | 0.00% |
| Bathroom Design | 1 | 0.00% |
| Conclusion | 1 | 0.00% |
| event/meal | 1 | 0.00% |
| Design Object | 1 | 0.00% |
| Publication/Resource | 1 | 0.00% |
| profession/hobby | 1 | 0.00% |
| knowledge/ability | 1 | 0.00% |
| Kite Model | 1 | 0.00% |
| Kiteboarding trick | 1 | 0.00% |
| Kiteboarding maneuver | 1 | 0.00% |
| Software Team | 1 | 0.00% |
| Benefits Plan | 1 | 0.00% |
| Storage Device | 1 | 0.00% |
| Boosting Website | 1 | 0.00% |
| property_code | 1 | 0.00% |
| rentescalaionclause | 1 | 0.00% |
| tenantimprovementsallowance | 1 | 0.00% |
| Resource/Asset | 1 | 0.00% |
| Animated Series | 1 | 0.00% |
| radiation | 1 | 0.00% |
| Business ranking | 1 | 0.00% |
| process feature | 1 | 0.00% |
| software/online platform | 1 | 0.00% |
| Social Media Content | 1 | 0.00% |
| Interior Design Feature | 1 | 0.00% |
| Home Improvement Items | 1 | 0.00% |
| Outdoor Living Feature | 1 | 0.00% |
| oven feature | 1 | 0.00% |
| vehicle body type | 1 | 0.00% |
| restaurant name | 1 | 0.00% |
| restaurant address | 1 | 0.00% |
| Artistic performance | 1 | 0.00% |
| Message of peace | 1 | 0.00% |
| Physical expression | 1 | 0.00% |
| Intellectual activity | 1 | 0.00% |
| significant other | 1 | 0.00% |
| Entity type: Flowers | 1 | 0.00% |
| Entity type: Funeral | 1 | 0.00% |
| Entity type: Emotions/Expressions | 1 | 0.00% |
| File Name | 1 | 0.00% |
| File Extension | 1 | 0.00% |
| Betting Limit | 1 | 0.00% |
| Software Provider | 1 | 0.00% |
| fraud scheme | 1 | 0.00% |
| booth_size | 1 | 0.00% |
| contact_email | 1 | 0.00% |
| blog | 1 | 0.00% |
| award show | 1 | 0.00% |
| graphics | 1 | 0.00% |
| Design approach | 1 | 0.00% |
| Work context | 1 | 0.00% |
| Group of Soldiers | 1 | 0.00% |
| Program/Theme | 1 | 0.00% |
| price point | 1 | 0.00% |
| promotion/sale | 1 | 0.00% |
| program type | 1 | 0.00% |
| desire | 1 | 0.00% |
| estate or property | 1 | 0.00% |
| skill or knowledge | 1 | 0.00% |
| Interview Type | 1 | 0.00% |
| Prescription drug | 1 | 0.00% |
| Network Port Number | 1 | 0.00% |
| Accommodation capacity | 1 | 0.00% |
| founded_year | 1 | 0.00% |
| Settlements | 1 | 0.00% |
| Food establishment | 1 | 0.00% |
| Academic subject | 1 | 0.00% |
| weather event | 1 | 0.00% |
| food and drink | 1 | 0.00% |
| soil_type | 1 | 0.00% |
| upvotes | 1 | 0.00% |
| advertising space | 1 | 0.00% |
| availability status | 1 | 0.00% |
| API | 1 | 0.00% |
| Software implementation | 1 | 0.00% |
| Customer Persona | 1 | 0.00% |
| Law enforcement position | 1 | 0.00% |
| Program coordinator position | 1 | 0.00% |
| Sports Action | 1 | 0.00% |
| Sports Object | 1 | 0.00% |
| Accommodation/Hotel | 1 | 0.00% |
| profession/job title | 1 | 0.00% |
| Secret | 1 | 0.00% |
| Download Manager | 1 | 0.00% |
| Product Key | 1 | 0.00% |
| key | 1 | 0.00% |
| seats | 1 | 0.00% |
| Internet domain names | 1 | 0.00% |
| Computer program | 1 | 0.00% |
| Legal permission | 1 | 0.00% |
| Internet domain registrar | 1 | 0.00% |
| Software updates | 1 | 0.00% |
| Time measurement | 1 | 0.00% |
| education funding | 1 | 0.00% |
| Accommodation/Service | 1 | 0.00% |
| Grant | 1 | 0.00% |
| Entrepreneur | 1 | 0.00% |
| Impact | 1 | 0.00% |
| Landmark/Recreation | 1 | 0.00% |
| Landmark/Memorial | 1 | 0.00% |
| Political Campaign Committee | 1 | 0.00% |
| Political Candidate | 1 | 0.00% |
| south_yemen_conflict.accusations_against_governments.1.accusations | 1 | 0.00% |
| federalism_definition.confederation_examples | 1 | 0.00% |
| federalism_definition.confederation | 1 | 0.00% |
| south_yemen_conflict.call_for_independence | 1 | 0.00% |
| south_yemen_conflict.historical_context.discrimination_and_exclusion | 1 | 0.00% |
| south_yemen_conflict.accusations_against_governments.1.government_name | 1 | 0.00% |
| south_yemen_conflict.historical_context.casualties_and_prisoners | 1 | 0.00% |
| Rule/Principle | 1 | 0.00% |
| Sculpture | 1 | 0.00% |
| Birth year | 1 | 0.00% |
| symbol: Company ticker symbol as listed in the report | 1 | 0.00% |
| brand or company | 1 | 0.00% |
| digital artwork or design | 1 | 0.00% |
| amplifier mode | 1 | 0.00% |
| Cultural/Personal Entity | 1 | 0.00% |
| Language/Cultural Entity | 1 | 0.00% |
| Educators | 1 | 0.00% |
| Learners | 1 | 0.00% |
| Medical Recipients | 1 | 0.00% |
| Political Ideology | 1 | 0.00% |
| individuals with autism | 1 | 0.00% |
| negative social interaction | 1 | 0.00% |
| discussion forum | 1 | 0.00% |
| forum thread | 1 | 0.00% |
| Chipset | 1 | 0.00% |
| traveler_location | 1 | 0.00% |
| approach/strategy | 1 | 0.00% |
| version control system | 1 | 0.00% |
| freedom | 1 | 0.00% |
| miscellaneous | 1 | 0.00% |
| contact person | 1 | 0.00% |
| potential renters | 1 | 0.00% |
| lot_number | 1 | 0.00% |
| vaccine_type | 1 | 0.00% |
| Teaching Offer | 1 | 0.00% |
| usage | 1 | 0.00% |
| ticket_number: Unique identifier for the RMA request | 1 | 0.00% |
| device_model: Model name of the returned electronic device | 1 | 0.00% |
| serial_number: Unique hardware identifier for the device | 1 | 0.00% |
| submitter: Name of the customer who submitted the ticket | 1 | 0.00% |
| purchase_order: Order number from original purchase | 1 | 0.00% |
| purchase_location: Store where device was originally bought | 1 | 0.00% |
| return_date: Date device was returned to repair center | 1 | 0.00% |
| Political action | 1 | 0.00% |
| trip planning | 1 | 0.00% |
| knowledge resource | 1 | 0.00% |
| Lunar Event | 1 | 0.00% |
| Celestial Event | 1 | 0.00% |
| Chess website | 1 | 0.00% |
| Special Day | 1 | 0.00% |
| Medium/Platform | 1 | 0.00% |
| Compatibility | 1 | 0.00% |
| Tour Month | 1 | 0.00% |
| Spokeswoman/Spokesman | 1 | 0.00% |
| course quality/type | 1 | 0.00% |
| Acoustic Foam Product | 1 | 0.00% |
| Home textile product | 1 | 0.00% |
| Historical Research | 1 | 0.00% |
| Primary Sources | 1 | 0.00% |
| Fashion Illustration | 1 | 0.00% |
| military personnel | 1 | 0.00% |
| injured individuals | 1 | 0.00% |
| Legislative Committees | 1 | 0.00% |
| Packaging material | 1 | 0.00% |
| kite decoration | 1 | 0.00% |
| outdoor toy | 1 | 0.00% |
| kite line | 1 | 0.00% |
| kite accessory | 1 | 0.00% |
| kite instruction | 1 | 0.00% |
| Productivity Tool | 1 | 0.00% |
| Movie Adaptation | 1 | 0.00% |
| Flooring type | 1 | 0.00% |
| Cultural influence | 1 | 0.00% |
| Wrestling Promotion/Company | 1 | 0.00% |
| business activity | 1 | 0.00% |
| population_density | 1 | 0.00% |
| total_region_communes | 1 | 0.00% |
| physician.education.medical_school.0.institution | 1 | 0.00% |
| physician.address.city | 1 | 0.00% |
| physician.education.residency.0.institution | 1 | 0.00% |
| physician.education.fellowship.0.institution | 1 | 0.00% |
| physician.degree | 1 | 0.00% |
| perpetrator of a violent crime | 1 | 0.00% |
| lack of bravery | 1 | 0.00% |
| live entertainment | 1 | 0.00% |
| Country/Team | 1 | 0.00% |
| Pet Grooming Mistake | 1 | 0.00% |
| Natural Landmarks | 1 | 0.00% |
| Business Districts | 1 | 0.00% |
| orthodontic appliance | 1 | 0.00% |
| depth | 1 | 0.00% |
| Workstation | 1 | 0.00% |
| Board | 1 | 0.00% |
| Job Seeker | 1 | 0.00% |
| Music Segment | 1 | 0.00% |
| hair tool | 1 | 0.00% |
| hair product | 1 | 0.00% |
| partnership success factor | 1 | 0.00% |
| Security Protocol | 1 | 0.00% |
| village name | 1 | 0.00% |
| village number | 1 | 0.00% |
| distance measure | 1 | 0.00% |
| instance_id | 1 | 0.00% |
| zone | 1 | 0.00% |
| textPayload | 1 | 0.00% |
| job_name | 1 | 0.00% |
| Merger Terms | 1 | 0.00% |
| Machine/Equipment | 1 | 0.00% |
| Model number | 1 | 0.00% |
| Real estate property type | 1 | 0.00% |
| Business Interest | 1 | 0.00% |
| software entity | 1 | 0.00% |
| Cultural Venue | 1 | 0.00% |
| data privacy | 1 | 0.00% |
| user data management | 1 | 0.00% |
| user authentication | 1 | 0.00% |
| Architecture Feature | 1 | 0.00% |
| object or task | 1 | 0.00% |
| Database System | 1 | 0.00% |
| reusable item | 1 | 0.00% |
| artistic_intent.meaning_zero_point | 1 | 0.00% |
| viewer_experience.interpretation | 1 | 0.00% |
| location.department | 1 | 0.00% |
| Tourism attraction | 1 | 0.00% |
| User interface | 1 | 0.00% |
| Motion graphics | 1 | 0.00% |
| Leisure time | 1 | 0.00% |
| Desire | 1 | 0.00% |
| tree species | 1 | 0.00% |
| physical characteristics | 1 | 0.00% |
| data processing | 1 | 0.00% |
| image processing | 1 | 0.00% |
| programming | 1 | 0.00% |
| dietary preference | 1 | 0.00% |
| CI server | 1 | 0.00% |
| Job role | 1 | 0.00% |
| Obstacles | 1 | 0.00% |
| Memory management | 1 | 0.00% |
| Program execution | 1 | 0.00% |
| Stadium/Venue | 1 | 0.00% |
| Size Range | 1 | 0.00% |
| Size Exclusion | 1 | 0.00% |
| Material type | 1 | 0.00% |
| Home Decor Style | 1 | 0.00% |
| Intellectual Property Creator | 1 | 0.00% |
| Intellectual Property License | 1 | 0.00% |
| review.hotel_name | 1 | 0.00% |
| review.stay_date | 1 | 0.00% |
| review.value_for_money | 1 | 0.00% |
| review.city | 1 | 0.00% |
| review.duration.reason | 1 | 0.00% |
| Event Winners | 1 | 0.00% |
| Netball team | 1 | 0.00% |
| 2016 Success | 1 | 0.00% |
| Deity/Religious Figure | 1 | 0.00% |
| Company/Agency | 1 | 0.00% |
| quantity/size | 1 | 0.00% |
| Brand/Collection Name | 1 | 0.00% |
| type of beverage | 1 | 0.00% |
| paradox.initial_payment_per_person | 1 | 0.00% |
| paradox.total_initial_payment | 1 | 0.00% |
| paradox.change_returned_to_people | 1 | 0.00% |
| paradox.change_given | 1 | 0.00% |
| hotel.location | 1 | 0.00% |
| hotel.name | 1 | 0.00% |
| review.stay_duration | 1 | 0.00% |
| Video Format | 1 | 0.00% |
| fruit type | 1 | 0.00% |
| texture type | 1 | 0.00% |
| taste type | 1 | 0.00% |
| authority.year | 1 | 0.00% |
| Business team | 1 | 0.00% |
| building/location | 1 | 0.00% |
| National Correct Coding Initiative (NCCI) section | 1 | 0.00% |
| State Name | 1 | 0.00% |
| baby gear | 1 | 0.00% |
| Result/Goal | 1 | 0.00% |
| Shopping Mall | 1 | 0.00% |
| Supermarket Chain | 1 | 0.00% |
| Aircraft type | 1 | 0.00% |
| Aircraft component | 1 | 0.00% |
| Protein/Receptor | 1 | 0.00% |
| Antibody | 1 | 0.00% |
| car interior | 1 | 0.00% |
| Human entity | 1 | 0.00% |
| Task/Project | 1 | 0.00% |
| Event/Art exhibition | 1 | 0.00% |
| Location/Studio | 1 | 0.00% |
| Group/Support system | 1 | 0.00% |
| Interior Design Company | 1 | 0.00% |
| Road Type | 1 | 0.00% |
| season/weather condition | 1 | 0.00% |
| file organization | 1 | 0.00% |
| duplication | 1 | 0.00% |
| revision control | 1 | 0.00% |
| desired location | 1 | 0.00% |
| farming | 1 | 0.00% |
| poultry farming | 1 | 0.00% |
| payment processing device | 1 | 0.00% |
| financial agreement | 1 | 0.00% |
| payment approval | 1 | 0.00% |
| Quantity measurement | 1 | 0.00% |
| car brand | 1 | 0.00% |
| beverage/food | 1 | 0.00% |
| numberOfEpisodes | 1 | 0.00% |
| collection of images | 1 | 0.00% |
| image property | 1 | 0.00% |
| Medical process | 1 | 0.00% |
| Haunted Attraction | 1 | 0.00% |
| Hardware/Accessory | 1 | 0.00% |
| Experience/Feeling | 1 | 0.00% |
| program/methodology | 1 | 0.00% |
| health and wellness promotion | 1 | 0.00% |
| lifespan extension | 1 | 0.00% |
| emotional and mental wellness | 1 | 0.00% |
| moral compass | 1 | 0.00% |
| moral education | 1 | 0.00% |
| morality | 1 | 0.00% |
| immorality | 1 | 0.00% |
| lack of morality | 1 | 0.00% |
| devotion | 1 | 0.00% |
| love | 1 | 0.00% |
| compassion | 1 | 0.00% |
| Visuals | 1 | 0.00% |
| art medium | 1 | 0.00% |
| artwork series | 1 | 0.00% |
| Chemical/Compound | 1 | 0.00% |
| Broadcasting Channel | 1 | 0.00% |
| File type | 1 | 0.00% |
| financial offer | 1 | 0.00% |
| finalize | 1 | 0.00% |
| willingness to transact | 1 | 0.00% |
| stock_exchange | 1 | 0.00% |
| Type of Room | 1 | 0.00% |
| artistry | 1 | 0.00% |
| admiration | 1 | 0.00% |
| soil type | 1 | 0.00% |
| climate type | 1 | 0.00% |
| Domain Name Type | 1 | 0.00% |
| Domain Name Extension | 1 | 0.00% |
| Discount Range | 1 | 0.00% |
| Technical Feature | 1 | 0.00% |
| sensory input | 1 | 0.00% |
| Medical complication | 1 | 0.00% |
| advice/opinion | 1 | 0.00% |
| prescription_date | 1 | 0.00% |
| Restaurant Group | 1 | 0.00% |
| Nationality/Cuisine | 1 | 0.00% |
| Food/dish | 1 | 0.00% |
| Metal | 1 | 0.00% |
| Altitude | 1 | 0.00% |
| spatial relationship | 1 | 0.00% |
| animal group | 1 | 0.00% |
| dateofincident | 1 | 0.00% |
| professional activity | 1 | 0.00% |
| Garden Decoration | 1 | 0.00% |
| Business Team | 1 | 0.00% |
| Business Operations | 1 | 0.00% |
| Business Marketing | 1 | 0.00% |
| Business Purchasing | 1 | 0.00% |
| Business Technology | 1 | 0.00% |
| Advertising Material | 1 | 0.00% |
| Membership group | 1 | 0.00% |
| Student group | 1 | 0.00% |
| Ticket sales | 1 | 0.00% |
| Event/Organization | 1 | 0.00% |
| Marble | 1 | 0.00% |
| Design object | 1 | 0.00% |
| software/version | 1 | 0.00% |
| coincidence | 1 | 0.00% |
| personal style | 1 | 0.00% |
| Archive | 1 | 0.00% |
| tool type | 1 | 0.00% |
| Job/Title | 1 | 0.00% |
| Military Organization | 1 | 0.00% |
| Breathing Rhythm | 1 | 0.00% |
| group of people/organizations involved or affected by the bankruptcy proceedings | 1 | 0.00% |
| assets and liabilities of the debtor that are subject to the bankruptcy proceedings | 1 | 0.00% |
| court responsible for overseeing bankruptcy proceedings | 1 | 0.00% |
| Communication software | 1 | 0.00% |
| Innovation | 1 | 0.00% |
| Writing Process | 1 | 0.00% |
| Narrative Control | 1 | 0.00% |
| Plot Progression | 1 | 0.00% |
| Plot Twist | 1 | 0.00% |
| Plot Development | 1 | 0.00% |
| Creative Control | 1 | 0.00% |
| Industry/Market | 1 | 0.00% |
| Coordinates | 1 | 0.00% |
| Destination | 1 | 0.00% |
| computer threat | 1 | 0.00% |
| financial system | 1 | 0.00% |
| Musical Equipment | 1 | 0.00% |
| Drum Pedal | 1 | 0.00% |
| Drum Stand | 1 | 0.00% |
| Stores | 1 | 0.00% |
| Biomolecule | 1 | 0.00% |
| Application/Software | 1 | 0.00% |
| Anime | 1 | 0.00% |
| Physical behavior | 1 | 0.00% |
| Political Term | 1 | 0.00% |
| return_date | 1 | 0.00% |
| rma_number | 1 | 0.00% |
| failure_reason | 1 | 0.00% |
| small businesses | 1 | 0.00% |
| Podcast Platform | 1 | 0.00% |
| imaging_type | 1 | 0.00% |
| award_name | 1 | 0.00% |
| student_id | 1 | 0.00% |
| discipline | 1 | 0.00% |
| Alcoholic Beverage | 1 | 0.00% |
| Cooking Tool | 1 | 0.00% |
| Artist/Band | 1 | 0.00% |
| Song/Album | 1 | 0.00% |
| Event category | 1 | 0.00% |
| number_of_offers | 1 | 0.00% |
| skill/achievement | 1 | 0.00% |
| strategy/tactics | 1 | 0.00% |
| mattress brand | 1 | 0.00% |
| mattress size | 1 | 0.00% |
| mattress type | 1 | 0.00% |
| excellence | 1 | 0.00% |
| Standardized technical document | 1 | 0.00% |
| Orientation Detection | 1 | 0.00% |
| Image Scaling | 1 | 0.00% |
| Image Enhancement | 1 | 0.00% |
| Military Aircraft | 1 | 0.00% |
| Water quality | 1 | 0.00% |
| Emotion/Opinion | 1 | 0.00% |
| color pattern | 1 | 0.00% |
| Natural Resource/Animals | 1 | 0.00% |
| Physical Features | 1 | 0.00% |
| Social/Cultural | 1 | 0.00% |
| returnLabelId | 1 | 0.00% |
| trackingNumber | 1 | 0.00% |
| serviceLevel | 1 | 0.00% |
| returnReason | 1 | 0.00% |
| returnPolicy | 1 | 0.00% |
| Architecture Structure<> An artificial underground structure built by humans | 1 | 0.00% |
| Architecture Structure | 1 | 0.00% |
| Aesthetic Entity | 1 | 0.00% |
| passenger_street | 1 | 0.00% |
| flight_number | 1 | 0.00% |
| passenger_name | 1 | 0.00% |
| handle | 1 | 0.00% |
| Actor/Actress | 1 | 0.00% |
| Digital Rights Management | 1 | 0.00% |
| holiday_look.collar.brand | 1 | 0.00% |
| holiday_look.collar.material | 1 | 0.00% |
| comparison.brand_with_similar_look | 1 | 0.00% |
| holiday_look.coat.brand | 1 | 0.00% |
| Summoned Companion | 1 | 0.00% |
| Financial Aid | 1 | 0.00% |
| Person/athlete | 1 | 0.00% |
| Course/Subject | 1 | 0.00% |
| defect | 1 | 0.00% |
| automobile | 1 | 0.00% |
| Event/Collection | 1 | 0.00% |
| Furniture style | 1 | 0.00% |
| experiment | 1 | 0.00% |
| mystical power | 1 | 0.00% |
| individual/customer | 1 | 0.00% |
| aspiration/goal | 1 | 0.00% |
| entertainment activity | 1 | 0.00% |
| legal assistance service | 1 | 0.00% |
| physical being | 1 | 0.00% |
| antihistamine | 1 | 0.00% |
| cough suppressant | 1 | 0.00% |
| decongestant | 1 | 0.00% |
| anxiety | 1 | 0.00% |
| Not an entity, just a description of where the pencil marking is located. | 1 | 0.00% |
| Charity organization | 1 | 0.00% |
| ranking.location_type | 1 | 0.00% |
| price_range.per | 1 | 0.00% |
| ranking.rank | 1 | 0.00% |
| tenant_id | 1 | 0.00% |
| sensory perception | 1 | 0.00% |
| entity type: insurance provider | 1 | 0.00% |
| entity type: insurance quote | 1 | 0.00% |
| entity type: car model | 1 | 0.00% |
| Service providers | 1 | 0.00% |
| Peace of mind | 1 | 0.00% |
| Character/TV Show | 1 | 0.00% |
| Coloring page | 1 | 0.00% |
| Coloring book pages | 1 | 0.00% |
| Coloring pages | 1 | 0.00% |
| Coloring pages for kids | 1 | 0.00% |
| Printable coloring pages | 1 | 0.00% |
| Online coloring pages | 1 | 0.00% |
| exhibitor_id: Unique identifier assigned to the exhibitor | 1 | 0.00% |
| status: Current status of the exhibitor's participation | 1 | 0.00% |
| booth_number: Letter and number designation of the exhibitor's booth | 1 | 0.00% |
| address: Physical address of the exhibitor | 1 | 0.00% |
| name: Full name of the exhibiting company | 1 | 0.00% |
| Participant | 1 | 0.00% |
| Ranking/Recommendation | 1 | 0.00% |
| reward/prize | 1 | 0.00% |
| account/balance | 1 | 0.00% |
| Production facility/Location | 1 | 0.00% |
| Customer/Role | 1 | 0.00% |
| Manufacturing/Process | 1 | 0.00% |
| Logistics/Process | 1 | 0.00% |
| Cooking/Process | 1 | 0.00% |
| website ranking metric | 1 | 0.00% |
| Legal claim | 1 | 0.00% |
| burial type | 1 | 0.00% |
| ground condition | 1 | 0.00% |
| budget_used | 1 | 0.00% |
| team_member | 1 | 0.00% |
| work environment | 1 | 0.00% |
| approval_date | 1 | 0.00% |
| ndc_code | 1 | 0.00% |
| indication | 1 | 0.00% |
| storage_temp | 1 | 0.00% |
| Medicine attribute | 1 | 0.00% |
| Online Tool | 1 | 0.00% |
| scientific entity | 1 | 0.00% |
| demographic entity | 1 | 0.00% |
| cultural tradition | 1 | 0.00% |
| Material/Feature | 1 | 0.00% |
| Company Name/Brand | 1 | 0.00% |
| Abbreviation/Event name | 1 | 0.00% |
| recreational object | 1 | 0.00% |
| pet access device | 1 | 0.00% |
| Job Security | 1 | 0.00% |
| Business Relationships | 1 | 0.00% |
| Job Application Material | 1 | 0.00% |
| Example Documents | 1 | 0.00% |
| Talent Type | 1 | 0.00% |
| Amount of money | 1 | 0.00% |
| Structural Design | 1 | 0.00% |
| Theatrical role | 1 | 0.00% |
| Theatrical production | 1 | 0.00% |
| patient_gender | 1 | 0.00% |
| drug_name | 1 | 0.00% |
| clinic_name | 1 | 0.00% |
| ndc | 1 | 0.00% |
| prescription_id | 1 | 0.00% |
| Section of a Book | 1 | 0.00% |
| diet/nutrition | 1 | 0.00% |
| event/challenge | 1 | 0.00% |
| Twitter Handle | 1 | 0.00% |
| vital_signs | 1 | 0.00% |
| admission_date | 1 | 0.00% |
| open_date | 1 | 0.00% |
| resolution_amount | 1 | 0.00% |
| school class | 1 | 0.00% |
| sports score | 1 | 0.00% |
| Travel Interests | 1 | 0.00% |
| Communication Format | 1 | 0.00% |
| Hotel Preferences | 1 | 0.00% |
| Marketing Consent | 1 | 0.00% |
| episode_number | 1 | 0.00% |
| actions.0.location | 1 | 0.00% |
| actions.0.target | 1 | 0.00% |
| Music Release | 1 | 0.00% |
| Movie Theater | 1 | 0.00% |
| Blog/Website Title | 1 | 0.00% |
| Blog Post Title | 1 | 0.00% |
| Writing Prompts | 1 | 0.00% |
| Social Media Trend | 1 | 0.00% |
| stomatopod crustacean | 1 | 0.00% |
| scale | 1 | 0.00% |
| organic molecule | 1 | 0.00% |
| group of activities | 1 | 0.00% |
| Contextual frame | 1 | 0.00% |
| Professional Designation | 1 | 0.00% |
| service metric | 1 | 0.00% |
| Art Exhibit | 1 | 0.00% |
| Remote Control | 1 | 0.00% |
| Power Connector | 1 | 0.00% |
| Flower type | 1 | 0.00% |
| Presentation | 1 | 0.00% |
| Delivery service | 1 | 0.00% |
| Machine Name | 1 | 0.00% |
| Machine Type | 1 | 0.00% |
| City Entity | 1 | 0.00% |
| DOS number | 1 | 0.00% |
| human activity | 1 | 0.00% |
| pharmaceutical product | 1 | 0.00% |
| chemical solution | 1 | 0.00% |
| Genealogy | 1 | 0.00% |
| Sports player | 1 | 0.00% |
| Natural location | 1 | 0.00% |
| vehicle trim level | 1 | 0.00% |
| year range | 1 | 0.00% |
| posting platform | 1 | 0.00% |
| Healthcare Institution | 1 | 0.00% |
| Social Media Account | 1 | 0.00% |
| Subscriber Count | 1 | 0.00% |
| equipment/tool | 1 | 0.00% |
| cultural difference | 1 | 0.00% |
| rivals | 1 | 0.00% |
| Radio talk show | 1 | 0.00% |
| Playlist | 1 | 0.00% |
| College Type | 1 | 0.00% |
| Fear/Concern | 1 | 0.00% |
| public figure/celebrity | 1 | 0.00% |
| Proof | 1 | 0.00% |
| Housing development | 1 | 0.00% |
| financial strategy | 1 | 0.00% |
| Membership Group | 1 | 0.00% |
| trainScheduleId | 1 | 0.00% |
| serviceNumber | 1 | 0.00% |
| destinationStation | 1 | 0.00% |
| startingStation | 1 | 0.00% |
| cooking device | 1 | 0.00% |
| food texture | 1 | 0.00% |
| customer relationship | 1 | 0.00% |
| backing | 1 | 0.00% |
| significant | 1 | 0.00% |
| Holy Book | 1 | 0.00% |
| Cultural Region | 1 | 0.00% |
| characteristic/property | 1 | 0.00% |
| document/report | 1 | 0.00% |
| directional location | 1 | 0.00% |
| task requirement | 1 | 0.00% |
| Protected Area | 1 | 0.00% |
| technology/device | 1 | 0.00% |
| Home Automation Product | 1 | 0.00% |
| Market Sector | 1 | 0.00% |
| Product Component | 1 | 0.00% |
| Table of Contents | 1 | 0.00% |
| Enumeration | 1 | 0.00% |
| Tradition/Habit | 1 | 0.00% |
| Business Directory | 1 | 0.00% |
| Health Service | 1 | 0.00% |
| genus.language | 1 | 0.00% |
| genus.description | 1 | 0.00% |
| genus.period | 1 | 0.00% |
| genus.location | 1 | 0.00% |
| genus.family.name | 1 | 0.00% |
| genus.naming.named_by | 1 | 0.00% |
| genus.naming.profession | 1 | 0.00% |
| genus.naming.year | 1 | 0.00% |
| genus.type_species | 1 | 0.00% |
| genus.species.0.reason | 1 | 0.00% |
| genus.species.0.name | 1 | 0.00% |
| genus.species.0.year | 1 | 0.00% |
| genus.unique_features | 1 | 0.00% |
| number/persons | 1 | 0.00% |
| body type categories | 1 | 0.00% |
| sales channel | 1 | 0.00% |
| Research Facility | 1 | 0.00% |
| negative mindset | 1 | 0.00% |
| mindfulness | 1 | 0.00% |
| biological factor | 1 | 0.00% |
| external factor | 1 | 0.00% |
| security concern | 1 | 0.00% |
| printer component | 1 | 0.00% |
| Videographer | 1 | 0.00% |
| review_summary.room_description | 1 | 0.00% |
| review_summary.location_quality | 1 | 0.00% |
| review_summary.breakfast_quality | 1 | 0.00% |
| adversity | 1 | 0.00% |
| sexual_act.insertion.vaginal_wall_dilation | 1 | 0.00% |
| sexual_act.cervix_retraction.reflux_triggers | 1 | 0.00% |
| sexual_act.cervix_retraction.muscle_contractions | 1 | 0.00% |
| sexual_act.ejaculation.sperm_release | 1 | 0.00% |
| sexual_act.thrusting.lubrication | 1 | 0.00% |
| sexual_act.cervix_retraction.semen_liquefaction | 1 | 0.00% |
| sexual_act.insertion.initial_alignment | 1 | 0.00% |
| sexual_act.ejaculation.semen_pool_formation | 1 | 0.00% |
| sexual_act.thrusting.cervix_transformation.trunk_contact | 1 | 0.00% |
| sexual_act.post_ejaculation.penis_withdrawal | 1 | 0.00% |
| sexual_act.cervix_retraction.reflux_inevitability | 1 | 0.00% |
| sexual_act.ejaculation.trunk_interaction | 1 | 0.00% |
| sexual_act.insertion.cervix_appearance.initial_appearance | 1 | 0.00% |
| sexual_act.thrusting.vaginal_wall_movement | 1 | 0.00% |
| sexual_act.sperm_migration.trunk_analogy_again | 1 | 0.00% |
| sexual_act.post_ejaculation.loss_of_camera_support | 1 | 0.00% |
| sexual_act.sperm_migration.sperm_departure | 1 | 0.00% |
| sexual_act.post_ejaculation.internal_changes | 1 | 0.00% |
| sexual_act.ejaculation.cervix_immersion | 1 | 0.00% |
| sexual_act.sperm_migration.mucus_sperm_interaction | 1 | 0.00% |
| sexual_act.insertion.cervix_appearance.hole_description | 1 | 0.00% |
| sexual_act.thrusting.cervix_transformation.descent | 1 | 0.00% |
| sexual_act.thrusting.vaginal_chamber_formation | 1 | 0.00% |
| sexual_act.insertion.cervix_appearance.changes_during_sex | 1 | 0.00% |
| sexual_act.sperm_migration.cervical_mucus_pathway | 1 | 0.00% |
| sexual_act.thrusting.cervix_transformation.elongation | 1 | 0.00% |
| sexual_act.thrusting.cervix_transformation.shape_change | 1 | 0.00% |
| opening | 1 | 0.00% |
| man-made object | 1 | 0.00% |
| Dance Style | 1 | 0.00% |
| Ecommerce Platform | 1 | 0.00% |
| Online site | 1 | 0.00% |
| Business School | 1 | 0.00% |
| contract duration | 1 | 0.00% |
| Group/Team | 1 | 0.00% |
| Metaphor/Emotion | 1 | 0.00% |
| Perception/Realization | 1 | 0.00% |
| Health and Medical Center | 1 | 0.00% |
| task or activity | 1 | 0.00% |
| linguistic unit | 1 | 0.00% |
| printing service | 1 | 0.00% |
| stitching service | 1 | 0.00% |
| advertising service | 1 | 0.00% |
| sla_number | 1 | 0.00% |
| support_level | 1 | 0.00% |
| approval_status | 1 | 0.00% |
| uptime_guarantee | 1 | 0.00% |
| initial_response_time | 1 | 0.00% |
| resolution_time | 1 | 0.00% |
| credit_amount | 1 | 0.00% |
| maximum_credit | 1 | 0.00% |
| LTCI insurance | 1 | 0.00% |
| Board Game | 1 | 0.00% |
| In-App Purchases | 1 | 0.00% |
| Computer Programs | 1 | 0.00% |
| File Organization | 1 | 0.00% |
| Title of a movie | 1 | 0.00% |
| skill/talent | 1 | 0.00% |
| activity/art | 1 | 0.00% |
| Admission rate | 1 | 0.00% |
| Number of applicants | 1 | 0.00% |
| Historical topic | 1 | 0.00% |
| Day and Month | 1 | 0.00% |
| Deck | 1 | 0.00% |
| Garage | 1 | 0.00% |
| School district | 1 | 0.00% |
| Lakes | 1 | 0.00% |
| Shopping | 1 | 0.00% |
| Restaurants | 1 | 0.00% |
| City center | 1 | 0.00% |
| Person/Event | 1 | 0.00% |
| gym chain | 1 | 0.00% |
| fitness routine | 1 | 0.00% |
| Game System | 1 | 0.00% |
| internet platform | 1 | 0.00% |
| healthcare facility room | 1 | 0.00% |
| media companies | 1 | 0.00% |
| closure | 1 | 0.00% |
| Registration Basis | 1 | 0.00% |
| Early Registration | 1 | 0.00% |
| payment processor | 1 | 0.00% |
| digital product | 1 | 0.00% |
| payment form | 1 | 0.00% |
| business contact | 1 | 0.00% |
| competitive advantage | 1 | 0.00% |
| Legal Offense | 1 | 0.00% |
| Consequence | 1 | 0.00% |
| Financial Action | 1 | 0.00% |
| Cleaning instruction | 1 | 0.00% |
| Biology/Physiology | 1 | 0.00% |
| Biochemistry | 1 | 0.00% |
| rationality | 1 | 0.00% |
| Casino Games | 1 | 0.00% |
| Casino Game | 1 | 0.00% |
| episodeTitle | 1 | 0.00% |
| fileUrl | 1 | 0.00% |
| miniature models | 1 | 0.00% |
| productivity/creativity | 1 | 0.00% |
| game pieces | 1 | 0.00% |
| Technology Framework | 1 | 0.00% |
| indoor space | 1 | 0.00% |
| mental health | 1 | 0.00% |
| Musical Instruments | 1 | 0.00% |
| Music chart | 1 | 0.00% |
| quality level | 1 | 0.00% |
| client type | 1 | 0.00% |
| Reference | 1 | 0.00% |
| significance | 1 | 0.00% |
| legal resolution | 1 | 0.00% |
| legal venue | 1 | 0.00% |
| context.solution | 1 | 0.00% |
| product.intended_use | 1 | 0.00% |
| product.name | 1 | 0.00% |
| product.download_link | 1 | 0.00% |
| product.technology | 1 | 0.00% |
| sms | 1 | 0.00% |
| preferredLanguage | 1 | 0.00% |
| Exhibition stand | 1 | 0.00% |
| Time/History | 1 | 0.00% |
| transcription_factor | 1 | 0.00% |
| gene_id | 1 | 0.00% |
| Engineering | 1 | 0.00% |
| Geoscience | 1 | 0.00% |
| Workflow | 1 | 0.00% |
| electrical wire | 1 | 0.00% |
| E-commerce system component | 1 | 0.00% |
| Price reduction | 1 | 0.00% |
| Limitations | 1 | 0.00% |
| filing_id | 1 | 0.00% |
| cusip | 1 | 0.00% |
| filing_amount | 1 | 0.00% |
| submission_method | 1 | 0.00% |
| Spa/Resort | 1 | 0.00% |
| Travel/Feature | 1 | 0.00% |
| Health/Advice | 1 | 0.00% |
| Health/Recipe | 1 | 0.00% |
| Food/Dish | 1 | 0.00% |
| Telecommunications Provider | 1 | 0.00% |
| Athlete/Performer | 1 | 0.00% |
| Achievement/Performance | 1 | 0.00% |
| Victory/Success | 1 | 0.00% |
| Financial amount | 1 | 0.00% |
| Academic department | 1 | 0.00% |
| energy demand | 1 | 0.00% |
| problem solution | 1 | 0.00% |
| location/industry | 1 | 0.00% |
| ancient settlement | 1 | 0.00% |
| administrative district | 1 | 0.00% |
| Limited Edition Product | 1 | 0.00% |
| prokeimenon.mode | 1 | 0.00% |
| epistle.book | 1 | 0.00% |
| subscription_prompt | 1 | 0.00% |
| prokeimenon.text | 1 | 0.00% |
| Public Health Effort | 1 | 0.00% |
| Outcome/Result | 1 | 0.00% |
| Technology/Protocol | 1 | 0.00% |
| Trust Mark | 1 | 0.00% |
| expert | 1 | 0.00% |
| Website/Online Platform | 1 | 0.00% |
| Newsletter/Publication | 1 | 0.00% |
| Fitness Community/Brand | 1 | 0.00% |
| Health and Wellness Website/Online Platform | 1 | 0.00% |
| Fitness Program/Brand | 1 | 0.00% |
| Health and Wellness Book/Program | 1 | 0.00% |
| Supplement Brand/Company | 1 | 0.00% |
| Situation/Status | 1 | 0.00% |
| Hobby/Passion | 1 | 0.00% |
| configuration setting | 1 | 0.00% |
| System entity | 1 | 0.00% |
| Network entity | 1 | 0.00% |
| Syntax entity | 1 | 0.00% |
| municipality.name_local | 1 | 0.00% |
| municipality.location.region | 1 | 0.00% |
| municipality.location.country | 1 | 0.00% |
| demographics.population.0.count | 1 | 0.00% |
| demographics.population.0.year | 1 | 0.00% |
| Document Version Number | 1 | 0.00% |
| floral arrangement | 1 | 0.00% |
| decor item | 1 | 0.00% |
| food cart | 1 | 0.00% |
| savings club | 1 | 0.00% |
| parental figure | 1 | 0.00% |
| parenting community | 1 | 0.00% |
| Sports Championship | 1 | 0.00% |
| inner turmoil | 1 | 0.00% |
| log_id | 1 | 0.00% |
| bp_diastolic | 1 | 0.00% |
| spo2 | 1 | 0.00% |
| bp_systolic | 1 | 0.00% |
| Age category | 1 | 0.00% |
| cooking temperature | 1 | 0.00% |
| Calendar Year | 1 | 0.00% |
| Event Admission | 1 | 0.00% |
| Collectibles | 1 | 0.00% |
| hormone | 1 | 0.00% |
| protein receptors | 1 | 0.00% |
| brain region | 1 | 0.00% |
| chemical signaling | 1 | 0.00% |
| Automotive Service Provider | 1 | 0.00% |
| Composition Title | 1 | 0.00% |
| Life Events | 1 | 0.00% |
| renewal_amount | 1 | 0.00% |
| Trade Policy | 1 | 0.00% |
| Nation/Geopolitical entity | 1 | 0.00% |
| Organizational goals/values | 1 | 0.00% |
| client_id | 1 | 0.00% |
| recipient: Email address of the report recipient | 1 | 0.00% |
| location: Geographic location of the incident | 1 | 0.00% |
| reference: Unique reference code for the incident | 1 | 0.00% |
| ip_address: Source IP address involved in unauthorized access | 1 | 0.00% |
| source: Internal system or process generating the report | 1 | 0.00% |
| sender: Email address of the report sender | 1 | 0.00% |
| Historical Era | 1 | 0.00% |
| Illegal Action | 1 | 0.00% |
| Mortality | 1 | 0.00% |
| Animated film | 1 | 0.00% |
| Environment/Location | 1 | 0.00% |
| Gender/Person | 1 | 0.00% |
| carpentry technique | 1 | 0.00% |
| window treatment | 1 | 0.00% |
| farming technique | 1 | 0.00% |
| Watering system | 1 | 0.00% |
| Watering system accessory | 1 | 0.00% |
| Growing medium | 1 | 0.00% |
| work area | 1 | 0.00% |
| meeting space | 1 | 0.00% |
| literary works | 1 | 0.00% |
| Detention Start Time | 1 | 0.00% |
| Church Name | 1 | 0.00% |
| Job applicant | 1 | 0.00% |
| Time constraint | 1 | 0.00% |
| Web Development Platform | 1 | 0.00% |
| Style Sheet Language | 1 | 0.00% |
| Financial Performance | 1 | 0.00% |
| Positive Direction | 1 | 0.00% |
| Market Competition | 1 | 0.00% |
| Cryptid Phenomenon | 1 | 0.00% |
| place.location.region | 1 | 0.00% |
| place.location.department | 1 | 0.00% |
| place.location.province | 1 | 0.00% |
| place.location.district | 1 | 0.00% |
| special needs support services | 1 | 0.00% |
| educational system | 1 | 0.00% |
| website design | 1 | 0.00% |
| brand strategy | 1 | 0.00% |
| customer segmentation | 1 | 0.00% |
| financial stability | 1 | 0.00% |
| financial responsibility | 1 | 0.00% |
| employment stability | 1 | 0.00% |
| Academic Examination | 1 | 0.00% |
| Title/Descriptive Word | 1 | 0.00% |
| Insurance Service | 1 | 0.00% |
| Process Type | 1 | 0.00% |
| research_project | 1 | 0.00% |
| principal_investigator | 1 | 0.00% |
| Insurance companies | 1 | 0.00% |
| Automotive industry | 1 | 0.00% |
| Event/Annual Celebration | 1 | 0.00% |
| Place/City | 1 | 0.00% |
| Street address | 1 | 0.00% |
| service representative | 1 | 0.00% |
| schedule event | 1 | 0.00% |
| TV Show/Blog | 1 | 0.00% |
| Activity/Attraction | 1 | 0.00% |
| Attraction/Location | 1 | 0.00% |
| Episode | 1 | 0.00% |
| Tour Type | 1 | 0.00% |
| Natural Area | 1 | 0.00% |
| learners | 1 | 0.00% |
| accountability | 1 | 0.00% |
| website type | 1 | 0.00% |
| financial process | 1 | 0.00% |
| survey type | 1 | 0.00% |
| Color Code | 1 | 0.00% |
| Home Automation Service | 1 | 0.00% |
| Clothing Fit | 1 | 0.00% |
| Protective Equipment | 1 | 0.00% |
| Data Sampling | 1 | 0.00% |
| Artwork Type | 1 | 0.00% |
| Musician/Performer | 1 | 0.00% |
| Person/Dealer | 1 | 0.00% |
| Music/Album | 1 | 0.00% |
| Brewing Process | 1 | 0.00% |
| Brewing Parameter | 1 | 0.00% |
| Book/Movie Title | 1 | 0.00% |
| Book Genre | 1 | 0.00% |
| food/spread | 1 | 0.00% |
| Financial System | 1 | 0.00% |
| Community Type | 1 | 0.00% |
| Consultation Process | 1 | 0.00% |
| activity  | 1 | 0.00% |
| Development board | 1 | 0.00% |
| Backup software | 1 | 0.00% |
| Internet service provider (ISP) | 1 | 0.00% |
| Internet speed | 1 | 0.00% |
| Universal Serial Bus | 1 | 0.00% |
| Standard operating procedure | 1 | 0.00% |
| Performance outcome | 1 | 0.00% |
| Activity Equipment | 1 | 0.00% |
| disputingpartya | 1 | 0.00% |
| registration_number | 1 | 0.00% |
| Angel | 1 | 0.00% |
| Angelic feature | 1 | 0.00% |
| Personal Brand/Blogger | 1 | 0.00% |
| Social Norm/Value | 1 | 0.00% |
| decorative design | 1 | 0.00% |
| legal field | 1 | 0.00% |
| legal aid organization | 1 | 0.00% |
| Wildlife Conservation | 1 | 0.00% |
| Agricultural product | 1 | 0.00% |
| sponsorship | 1 | 0.00% |
| status: Current status of vaccination record | 1 | 0.00% |
| vaccine_type: Type of vaccine administered | 1 | 0.00% |
| Footwear type | 1 | 0.00% |
| company type | 1 | 0.00% |
| 일정 | 1 | 0.00% |
| Marketing statement | 1 | 0.00% |
| Taxonomy/Classification | 1 | 0.00% |
| infrastructure/project | 1 | 0.00% |
| physical debris | 1 | 0.00% |
| Diversity | 1 | 0.00% |
| Routine | 1 | 0.00% |
| gardening location | 1 | 0.00% |
| pest type | 1 | 0.00% |
| disease type | 1 | 0.00% |
| Access point | 1 | 0.00% |
| Company/Group | 1 | 0.00% |
| last_updated | 1 | 0.00% |
| departure_time | 1 | 0.00% |
| seat_number | 1 | 0.00% |
| utility | 1 | 0.00% |
| Entertainment venue | 1 | 0.00% |
| organization type | 1 | 0.00% |
| political philosophy | 1 | 0.00% |
| political ideology | 1 | 0.00% |
| Property Development | 1 | 0.00% |
| dietary plan | 1 | 0.00% |
| Cinema | 1 | 0.00% |
| Economic Action | 1 | 0.00% |
| cause of injury | 1 | 0.00% |
| process or method | 1 | 0.00% |
| Clothing quality | 1 | 0.00% |
| Isotope | 1 | 0.00% |
| Gas | 1 | 0.00% |
| Woodwork | 1 | 0.00% |
| Ceramics | 1 | 0.00% |
| Metal Art | 1 | 0.00% |
| Painting | 1 | 0.00% |
| Glass | 1 | 0.00% |
| Crafts | 1 | 0.00% |
| Payment Option | 1 | 0.00% |
| technology_area | 1 | 0.00% |
| business owner | 1 | 0.00% |
| important data | 1 | 0.00% |
| driving condition | 1 | 0.00% |
| Medical Process | 1 | 0.00% |
| Work hours | 1 | 0.00% |
| Work duration | 1 | 0.00% |
| Academic term | 1 | 0.00% |
| chemical component | 1 | 0.00% |
| appointmentType | 1 | 0.00% |
| Job applicants | 1 | 0.00% |
| Completion of application | 1 | 0.00% |
| Hardcopy | 1 | 0.00% |
| Future use | 1 | 0.00% |
| Neurotransmitter | 1 | 0.00% |
| NBA player | 1 | 0.00% |
| Draft pick | 1 | 0.00% |
| Draft round | 1 | 0.00% |
| Professional sports draft | 1 | 0.00% |
| Cooling System | 1 | 0.00% |
| Engine Displacement | 1 | 0.00% |
| listing_id | 1 | 0.00% |
| beds | 1 | 0.00% |
| baths | 1 | 0.00% |
| family size | 1 | 0.00% |
| television network | 1 | 0.00% |
| location/venue | 1 | 0.00% |
| origin/creator | 1 | 0.00% |
| name/title | 1 | 0.00% |
| Refrigerant | 1 | 0.00% |
| Environmental Concerns | 1 | 0.00% |
| Air Conditioning Systems | 1 | 0.00% |
| Cooling Efficiency | 1 | 0.00% |
| Energy Efficiency | 1 | 0.00% |
| target_audience: Demographic group the product is marketed to | 1 | 0.00% |
| name: Name of the beauty product launched | 1 | 0.00% |
| campaign_status: Current status of the campaign active, in review, paused | 1 | 0.00% |
| price: Price of the product in US dollars | 1 | 0.00% |
| category: Type of beauty product such as serum, foundation, oil, balm | 1 | 0.00% |
| Market/Shopping Area | 1 | 0.00% |
| Historical Site | 1 | 0.00% |
| societal issue | 1 | 0.00% |
| asteroid.discovery.location | 1 | 0.00% |
| Campground/Park | 1 | 0.00% |
| residential buildings | 1 | 0.00% |
| Cooling system | 1 | 0.00% |
| Temperature control | 1 | 0.00% |
| Chemical identifier | 1 | 0.00% |
| Geographic regions | 1 | 0.00% |
| educational platform | 1 | 0.00% |
| People/Customers | 1 | 0.00% |
| name: Full name of the enrolled individual | 1 | 0.00% |
| enrollment_date: Date when enrollment began | 1 | 0.00% |
| program: Training or program name enrolled in | 1 | 0.00% |
| id: Unique identifier for the enrollment record | 1 | 0.00% |
| location: Geographic location of the enrollment site | 1 | 0.00% |
| Publication Edition | 1 | 0.00% |
| certainty | 1 | 0.00% |
| Natural occurrence | 1 | 0.00% |
| Event Organizer | 1 | 0.00% |
| Person/Target | 1 | 0.00% |
| job candidate | 1 | 0.00% |
| job application process | 1 | 0.00% |
| Quiz title | 1 | 0.00% |
| imaging_modality | 1 | 0.00% |
| report_signer | 1 | 0.00% |
| bodily response | 1 | 0.00% |
| body system | 1 | 0.00% |
| company_name: Name of the startup founded by the individual | 1 | 0.00% |
| founder_name: Full name of the founder as stated in passage | 1 | 0.00% |
| motivation: Personal reason driving the company's founding | 1 | 0.00% |
| focus_area: Primary domain or mission of the company | 1 | 0.00% |
| location: City and state where the company was founded | 1 | 0.00% |
| Battery capacity | 1 | 0.00% |
| Energy storage device | 1 | 0.00% |
| Electrical phenomenon | 1 | 0.00% |
| billing_amount | 1 | 0.00% |
| subscription_type | 1 | 0.00% |
| email_address | 1 | 0.00% |
| medicareNumber | 1 | 0.00% |
| qualifications | 1 | 0.00% |
| registrationNumber | 1 | 0.00% |
| neighborhood community | 1 | 0.00% |
| family name | 1 | 0.00% |
| craft item | 1 | 0.00% |
| Pet Product | 1 | 0.00% |
| Pet Service | 1 | 0.00% |
| Dining | 1 | 0.00% |
| Students | 1 | 0.00% |
| physiological measurement | 1 | 0.00% |
| genetic trait | 1 | 0.00% |
| activity level | 1 | 0.00% |
| Story Arc | 1 | 0.00% |
| Story Event | 1 | 0.00% |
| Student Body | 1 | 0.00% |
| Future Plans | 1 | 0.00% |
| page size | 1 | 0.00% |
| parenting method | 1 | 0.00% |
| solution type | 1 | 0.00% |
| policyholder_name: Full name of the claimant | 1 | 0.00% |
| claim_id: Unique identifier for the submitted claim | 1 | 0.00% |
| claim_date: Date the claim was filed | 1 | 0.00% |
| condition: Medical condition or procedure described | 1 | 0.00% |
| Disease type | 1 | 0.00% |
| Ticker | 1 | 0.00% |
| grade | 1 | 0.00% |
| reorderLevel | 1 | 0.00% |
| reorderQuantity | 1 | 0.00% |
| bin | 1 | 0.00% |
| leadTime | 1 | 0.00% |
| safetyStock | 1 | 0.00% |
| Chance | 1 | 0.00% |
| Color Finish | 1 | 0.00% |
| Scoring Play | 1 | 0.00% |
| Assisted Living Equipment | 1 | 0.00% |
| industry domain | 1 | 0.00% |
| supply chain | 1 | 0.00% |
| Technological Entity | 1 | 0.00% |
| Life Stage | 1 | 0.00% |
| Living Condition | 1 | 0.00% |
| person/filmmaker | 1 | 0.00% |
| Ability/Action | 1 | 0.00% |
| Physical Sensation | 1 | 0.00% |
| Speech Disorder | 1 | 0.00% |
| loading_docks | 1 | 0.00% |
| lease_id | 1 | 0.00% |
| gift suggestions | 1 | 0.00% |
| rebounds | 1 | 0.00% |
| Electronic game | 1 | 0.00% |
| defendant: Individual or entity being sued | 1 | 0.00% |
| case_number: Unique identifier for the court case | 1 | 0.00% |
| chemical_id: Unique identifier for each chemical substance | 1 | 0.00% |
| hazard_class: Hazard classification of the chemical substance | 1 | 0.00% |
| handling_method: Method used to handle the chemical | 1 | 0.00% |
| exposure_limit: Maximum allowable concentration in air | 1 | 0.00% |
| Cashback Service | 1 | 0.00% |
| Bulletin | 1 | 0.00% |
| Update | 1 | 0.00% |
| Product range | 1 | 0.00% |
| hair | 1 | 0.00% |
| Wind Direction | 1 | 0.00% |
| Wind Speed | 1 | 0.00% |
| Time Mark | 1 | 0.00% |
| Product/Capital Good | 1 | 0.00% |
| gardening equipment | 1 | 0.00% |
| Survey | 1 | 0.00% |
| Hygiene | 1 | 0.00% |
| Object entity | 1 | 0.00% |
| Conservation Area | 1 | 0.00% |
| Class Character | 1 | 0.00% |
| Insurance Product | 1 | 0.00% |
| posted_on | 1 | 0.00% |
| productDescription | 1 | 0.00% |
| productCategory | 1 | 0.00% |
| way of life | 1 | 0.00% |
| distinction | 1 | 0.00% |
| Writer | 1 | 0.00% |
| data cells | 1 | 0.00% |
| data cell | 1 | 0.00% |
| formula | 1 | 0.00% |
| data range | 1 | 0.00% |
| multimedia resources | 1 | 0.00% |
| presentation materials | 1 | 0.00% |
| Travel Service | 1 | 0.00% |
| Customer Service | 1 | 0.00% |
| Football Field | 1 | 0.00% |
| Football Technique | 1 | 0.00% |
| Group/Population | 1 | 0.00% |
| Website/Brand | 1 | 0.00% |
| Cuisine/Culinary Style | 1 | 0.00% |
| software development tool | 1 | 0.00% |
| identification number | 1 | 0.00% |
| mindfulness practice | 1 | 0.00% |
| Entertainment/Recreation | 1 | 0.00% |
| Vocal expression | 1 | 0.00% |
| establishment/venue | 1 | 0.00% |
| Status/Quality | 1 | 0.00% |
| Product Label | 1 | 0.00% |
| Title/Event | 1 | 0.00% |
| educational facility | 1 | 0.00% |
| child care facility | 1 | 0.00% |
| parking facility | 1 | 0.00% |
| Product bundle | 1 | 0.00% |
| Payment options | 1 | 0.00% |
| medical supply | 1 | 0.00% |
| Project/Service | 1 | 0.00% |
| Magic item | 1 | 0.00% |
| Website/Online Community | 1 | 0.00% |
| Hospital/Healthcare Institution | 1 | 0.00% |
| Artificial Intelligence Company | 1 | 0.00% |
| Quality of internet service | 1 | 0.00% |
| Artistic expression | 1 | 0.00% |
| Ancient City | 1 | 0.00% |
| Fundraising Page | 1 | 0.00% |
| lifestyle/activity | 1 | 0.00% |
| expression/emotion | 1 | 0.00% |
| inventory status | 1 | 0.00% |
| Military Equipment/Weaponry | 1 | 0.00% |
| Medical Study Type | 1 | 0.00% |
| Housing Scheme | 1 | 0.00% |
| Year/Decade | 1 | 0.00% |
| position_id | 1 | 0.00% |
| Job-related topic | 1 | 0.00% |
| clothing/style | 1 | 0.00% |
| quality/style | 1 | 0.00% |
| production method | 1 | 0.00% |
| Societal Trend | 1 | 0.00% |
| self-control | 1 | 0.00% |
| quantity/amount | 1 | 0.00% |
| Fishing Equipment | 1 | 0.00% |
| patient.age_estimation.previous_state | 1 | 0.00% |
| doctor.reaction | 1 | 0.00% |
| patient.age_estimation.correction | 1 | 0.00% |
| patient.behavior | 1 | 0.00% |
| chair.condition | 1 | 0.00% |
| patient.request | 1 | 0.00% |
| Game type | 1 | 0.00% |
| Game event | 1 | 0.00% |
| study resource | 1 | 0.00% |
| part of speech | 1 | 0.00% |
| Physical process | 1 | 0.00% |
| Action/Policy | 1 | 0.00% |
| Material/Color | 1 | 0.00% |
| physical challenge | 1 | 0.00% |
| topography | 1 | 0.00% |
| action or thing given to benefit a cause or person | 1 | 0.00% |
| transaction_id: Unique identifier for the transaction entry | 1 | 0.00% |
| amount: Monetary value of the transaction in USD | 1 | 0.00% |
| Baseball team | 1 | 0.00% |
| Baseball players | 1 | 0.00% |
| Strikeout-to-walk ratio | 1 | 0.00% |
| socialization | 1 | 0.00% |
| human interaction | 1 | 0.00% |
| behavior modification | 1 | 0.00% |
| Music Track | 1 | 0.00% |
| ItemID | 1 | 0.00% |
| InspectionDate | 1 | 0.00% |
| OperatorID | 1 | 0.00% |
| Width | 1 | 0.00% |
| MeasurementUnit | 1 | 0.00% |
| Notes | 1 | 0.00% |
| Learning Center | 1 | 0.00% |
| art and media | 1 | 0.00% |
| Business/Company Name | 1 | 0.00% |
| Query terms | 1 | 0.00% |
| Fulfillment of needs | 1 | 0.00% |
| Image Resolution | 1 | 0.00% |
| governmental body | 1 | 0.00% |
| taxation | 1 | 0.00% |
| drinking game | 1 | 0.00% |
| Citizen Group | 1 | 0.00% |
| leave type | 1 | 0.00% |
| shopping | 1 | 0.00% |
| consumer product | 1 | 0.00% |
| Person/Artist/Puppeteer | 1 | 0.00% |
| Production/Play | 1 | 0.00% |
| learning method | 1 | 0.00% |
| fitness company | 1 | 0.00% |
| Tax Form | 1 | 0.00% |
| Expense Deductions | 1 | 0.00% |
| Blog/Website | 1 | 0.00% |
| Sewing notion | 1 | 0.00% |
| Speed Limit | 1 | 0.00% |
| general event | 1 | 0.00% |
| Use Case | 1 | 0.00% |
| Investment Strategy | 1 | 0.00% |
| Illegal Material | 1 | 0.00% |
| Negative Emotion | 1 | 0.00% |
| vehicle metric | 1 | 0.00% |
| ticketId | 1 | 0.00% |
| ticketType | 1 | 0.00% |
| passengerName | 1 | 0.00% |
| Vehicle Status | 1 | 0.00% |
| Dating | 1 | 0.00% |
| sweetener | 1 | 0.00% |
| citrus juice | 1 | 0.00% |
| frozen water | 1 | 0.00% |
| Book/TV Series | 1 | 0.00% |
| air pollution | 1 | 0.00% |
| discount rate | 1 | 0.00% |
| document_id | 1 | 0.00% |
| applicable_products | 1 | 0.00% |
| sql_server_versions | 1 | 0.00% |
| network_protocols | 1 | 0.00% |
| ports | 1 | 0.00% |
| server_hiding.description | 1 | 0.00% |
| server_hiding.port_conflict | 1 | 0.00% |
| server_hiding.error_message | 1 | 0.00% |
| server_hiding.resolution | 1 | 0.00% |
| dates.last_reviewed | 1 | 0.00% |
| contact_information.help | 1 | 0.00% |
| contact_information.expert_help | 1 | 0.00% |
| storage location | 1 | 0.00% |
| Ballet Company | 1 | 0.00% |
| counseling | 1 | 0.00% |
| outreach_id: Unique identifier assigned to the outreach | 1 | 0.00% |
| sender: Person initiating the outreach contact | 1 | 0.00% |
| recipient: Name of the recipient contacted | 1 | 0.00% |
| status: Current status of the outreach message | 1 | 0.00% |
| message_subject: Subject line of the outreach message | 1 | 0.00% |
| Marketing Campaigns | 1 | 0.00% |
| impacted | 1 | 0.00% |
| research funding | 1 | 0.00% |
| research methodology | 1 | 0.00% |
| research assumptions | 1 | 0.00% |
| research outcome | 1 | 0.00% |
| funding source | 1 | 0.00% |
| Computer operating system | 1 | 0.00% |
| government regulation | 1 | 0.00% |
| philanthropy | 1 | 0.00% |
| social responsibility | 1 | 0.00% |
| public perception | 1 | 0.00% |
| cannabis industry | 1 | 0.00% |
| Mapping Company | 1 | 0.00% |
| Time Period/Experience Level | 1 | 0.00% |
| body part/organ | 1 | 0.00% |
| Account Balance | 1 | 0.00% |
| Points | 1 | 0.00% |
| Book club | 1 | 0.00% |
| Novel | 1 | 0.00% |
| Workshop Title | 1 | 0.00% |
| Accreditation Body | 1 | 0.00% |
| Construction zone | 1 | 0.00% |
| event_id: Unique identifier for each event | 1 | 0.00% |
| event_type: Category of the event | 1 | 0.00% |
| organizer: Person or group hosting the event | 1 | 0.00% |
| location: Physical address or venue of the event | 1 | 0.00% |
| price: Entry cost for attending the event | 1 | 0.00% |
| charity organisation | 1 | 0.00% |
| volunteers | 1 | 0.00% |
| asteroid.discovery_year | 1 | 0.00% |
| asteroid.orbital_characteristics.inclination_unit | 1 | 0.00% |
| Measurement Quantity | 1 | 0.00% |
| Tax type | 1 | 0.00% |
| Report title | 1 | 0.00% |
| Award recipient | 1 | 0.00% |
| Chess move | 1 | 0.00% |
| Chess piece | 1 | 0.00% |
| Weather Element | 1 | 0.00% |
| Speed Range | 1 | 0.00% |
| Market trend | 1 | 0.00% |
| Stack | 1 | 0.00% |
| appointment_type | 1 | 0.00% |
| rehab center | 1 | 0.00% |
| Volcano | 1 | 0.00% |
| Natural disaster | 1 | 0.00% |
| Belief system | 1 | 0.00% |
| Blog Name | 1 | 0.00% |
| Home Decor Store | 1 | 0.00% |
| approach to learning | 1 | 0.00% |
| identification marker | 1 | 0.00% |
| Color Shade | 1 | 0.00% |
| Artistic object | 1 | 0.00% |
| Artistic medium | 1 | 0.00% |
| Artistic element | 1 | 0.00% |
| cut | 1 | 0.00% |
| risk_rating | 1 | 0.00% |
| Article | 1 | 0.00% |
| Content Management System | 1 | 0.00% |
| Plugin | 1 | 0.00% |
| project phase | 1 | 0.00% |
| Discouragement Rate | 1 | 0.00% |
| Marketing Method | 1 | 0.00% |
| irregular hours | 1 | 0.00% |
| Competition Enhancement | 1 | 0.00% |
| Material/Textile | 1 | 0.00% |
| vocational education | 1 | 0.00% |
| Facility/Service | 1 | 0.00% |
| Emphasis | 1 | 0.00% |
| Product/Ingredient | 1 | 0.00% |
| moisture-management technology | 1 | 0.00% |
| scam | 1 | 0.00% |
| dollar game | 1 | 0.00% |
| low quality product | 1 | 0.00% |
| Place/Group | 1 | 0.00% |
| rules/policies | 1 | 0.00% |
| people/roles | 1 | 0.00% |
| service/activity | 1 | 0.00% |
| feature/component | 1 | 0.00% |
| Event Planning | 1 | 0.00% |
| Consultation | 1 | 0.00% |
| Gratitude | 1 | 0.00% |
| Emotional Satisfaction | 1 | 0.00% |
| Album/Song | 1 | 0.00% |
| Bathroom Decor and Design | 1 | 0.00% |
| mosquito repellent | 1 | 0.00% |
| unpleasantness | 1 | 0.00% |
| air pollutant | 1 | 0.00% |
| Module Loader | 1 | 0.00% |
| Configuration file | 1 | 0.00% |
| Software packages | 1 | 0.00% |
| contact_name | 1 | 0.00% |
| Emotion/State of being | 1 | 0.00% |
| Activity type | 1 | 0.00% |
| Statistical Measure | 1 | 0.00% |
| Fundraising campaign | 1 | 0.00% |
| Legal residency | 1 | 0.00% |
| Legal age | 1 | 0.00% |
| Participation | 1 | 0.00% |
| Event period | 1 | 0.00% |
| Event end | 1 | 0.00% |
| Pipe brand and model | 1 | 0.00% |
| cute animals | 1 | 0.00% |
| post_date | 1 | 0.00% |
| location_tag | 1 | 0.00% |
| photo_quality | 1 | 0.00% |
| retweets | 1 | 0.00% |
| surface finish | 1 | 0.00% |
| folklore/myth | 1 | 0.00% |
| Historical site | 1 | 0.00% |
| educational measurement | 1 | 0.00% |
| Accommodation Facilities | 1 | 0.00% |
| Food Services | 1 | 0.00% |
| Recreational Facilities | 1 | 0.00% |
| Duty | 1 | 0.00% |
| Military position | 1 | 0.00% |
| Surface Finish | 1 | 0.00% |
| fuel | 1 | 0.00% |
| Mealtime | 1 | 0.00% |
| Television program | 1 | 0.00% |
| topic/question | 1 | 0.00% |
| Neighborhood Grocery Store | 1 | 0.00% |
| Parenting Struggles | 1 | 0.00% |
| Parenting Mishaps | 1 | 0.00% |
| Children's Entertainment | 1 | 0.00% |
| perpetrator | 1 | 0.00% |
| law enforcement officer | 1 | 0.00% |
| unknown identity | 1 | 0.00% |
| unknown motive | 1 | 0.00% |
| act of violence | 1 | 0.00% |
| Sports Program/Initiative | 1 | 0.00% |
| Bird species | 1 | 0.00% |
| professional pursuits | 1 | 0.00% |
| academic disciplines | 1 | 0.00% |
| Wine Director | 1 | 0.00% |
| brand/business | 1 | 0.00% |
| Cigar manufacturing process | 1 | 0.00% |
| Cigar manufacturer | 1 | 0.00% |
| Location/distance | 1 | 0.00% |
| UI component | 1 | 0.00% |
| Technical feature | 1 | 0.00% |
| surveyPurpose | 1 | 0.00% |
| intendedAudience | 1 | 0.00% |
| organizer.name | 1 | 0.00% |
| surveyFocus | 1 | 0.00% |
| expectedOutcome | 1 | 0.00% |
| accessInstructions | 1 | 0.00% |
| Property asset | 1 | 0.00% |
| Company/Online Store | 1 | 0.00% |
| Personality Trait/Interest | 1 | 0.00% |
| listing_id: Unique identifier for each property listing | 1 | 0.00% |
| beds: Number of bedrooms | 1 | 0.00% |
| state: State where the property is located | 1 | 0.00% |
| city: City where the property is located | 1 | 0.00% |
| price: Listed price in dollars | 1 | 0.00% |
| Marketing outlet | 1 | 0.00% |
| Business goals | 1 | 0.00% |
| Revenue stream | 1 | 0.00% |
| Planning | 1 | 0.00% |
| Trigger warning | 1 | 0.00% |
| workout of the day | 1 | 0.00% |
| alert system | 1 | 0.00% |
| digital device | 1 | 0.00% |
| planId | 1 | 0.00% |
| npiNumber | 1 | 0.00% |
| online slot game | 1 | 0.00% |
| Health Plan | 1 | 0.00% |
| technology-related entity | 1 | 0.00% |
| motivational entity | 1 | 0.00% |
| wallet | 1 | 0.00% |
| Input Device | 1 | 0.00% |
| Display Component | 1 | 0.00% |
| Technical Requirements | 1 | 0.00% |
| wine brand | 1 | 0.00% |
| Demographic Change | 1 | 0.00% |
| Prosperous Years | 1 | 0.00% |
| Fitness Tracker Data | 1 | 0.00% |
| priority/attention | 1 | 0.00% |
| plan/goal | 1 | 0.00% |
| change | 1 | 0.00% |
| progress/improvement | 1 | 0.00% |
| Interface | 1 | 0.00% |
| Music Streaming Platform | 1 | 0.00% |
| Website section | 1 | 0.00% |
| Group of organizations | 1 | 0.00% |
| Beauty Pageant Contestant | 1 | 0.00% |
| Beauty Pageant Winner | 1 | 0.00% |
| government facility | 1 | 0.00% |
| Incident | 1 | 0.00% |
| endearment | 1 | 0.00% |
| Kitchen cabinet brand | 1 | 0.00% |
| planetary_system.planets.0.discovery_year | 1 | 0.00% |
| star.effective_temperature | 1 | 0.00% |
| star.distance.light_years | 1 | 0.00% |
| planetary_system.planets.0.designation | 1 | 0.00% |
| planetary_system.planets.0.discovery_team | 1 | 0.00% |
| star.designation | 1 | 0.00% |
| Type | 1 | 0.00% |
| pricing/affordability | 1 | 0.00% |
| productivity/efficiency | 1 | 0.00% |
| humidity/moisture | 1 | 0.00% |
| temperature control/cooling | 1 | 0.00% |
| sustainability/environmentalism | 1 | 0.00% |
| Service reliability | 1 | 0.00% |
| Application Deadline | 1 | 0.00% |
| meal conclusion | 1 | 0.00% |
| seasonings | 1 | 0.00% |
| cessation of eating | 1 | 0.00% |
| remaining food | 1 | 0.00% |
| photography | 1 | 0.00% |
| Location/Building | 1 | 0.00% |
| Group size | 1 | 0.00% |
| Shared living space | 1 | 0.00% |
| Individual rooms | 1 | 0.00% |
| Warning Message | 1 | 0.00% |
| Steering mechanism | 1 | 0.00% |
| Grooming Product | 1 | 0.00% |
| Physical Characteristics | 1 | 0.00% |
| job duty | 1 | 0.00% |
| Health Risk | 1 | 0.00% |
| Safety Measures | 1 | 0.00% |
| Health Promotion | 1 | 0.00% |
| Royal title | 1 | 0.00% |
| Numeric | 1 | 0.00% |
| quality check | 1 | 0.00% |
| Math Operation | 1 | 0.00% |
| Quality of Construction | 1 | 0.00% |
| Construction Method | 1 | 0.00% |
| Ministry/Department | 1 | 0.00% |
| issuedAt | 1 | 0.00% |
| expiresAt | 1 | 0.00% |
| Airport Lounge | 1 | 0.00% |
| service contract | 1 | 0.00% |
| Consumer feedback | 1 | 0.00% |
| invoice_id: Unique identifier for the invoice document | 1 | 0.00% |
| order type | 1 | 0.00% |
| Environmental Policy | 1 | 0.00% |
| Acronym/Brand | 1 | 0.00% |
| software technology | 1 | 0.00% |
| education/food/drink | 1 | 0.00% |
| commodities | 1 | 0.00% |
| tourist destinations | 1 | 0.00% |
| Area of land | 1 | 0.00% |
| wine producer | 1 | 0.00% |
| business ownership | 1 | 0.00% |
| cultural trend | 1 | 0.00% |
| culture/leisure | 1 | 0.00% |
| realization | 1 | 0.00% |
| play area | 1 | 0.00% |
| working conditions | 1 | 0.00% |
| social consciousness | 1 | 0.00% |
| changing standards | 1 | 0.00% |
| fishing gear/material | 1 | 0.00% |
| radio show | 1 | 0.00% |
| Material/Furniture | 1 | 0.00% |
| Pathway | 1 | 0.00% |
| nappy brand | 1 | 0.00% |
| baby product | 1 | 0.00% |
| feature option | 1 | 0.00% |
| building/utility | 1 | 0.00% |
| plant/flower | 1 | 0.00% |
| plant/crop | 1 | 0.00% |
| hyperlink/anchor text | 1 | 0.00% |
| professional identifier | 1 | 0.00% |
| professional license | 1 | 0.00% |
| license status | 1 | 0.00% |
| functional entity | 1 | 0.00% |
| legislative entity | 1 | 0.00% |
| musical instrument accessory | 1 | 0.00% |
| audio effect | 1 | 0.00% |
| Hymn number | 1 | 0.00% |
| Architecture style | 1 | 0.00% |
| Off-road vehicle | 1 | 0.00% |
| filer_name | 1 | 0.00% |
| flower/plant | 1 | 0.00% |
| place/environment | 1 | 0.00% |
| quantity/number | 1 | 0.00% |
| price_unit | 1 | 0.00% |
| building area | 1 | 0.00% |
| atmosphere or feeling | 1 | 0.00% |
| event/entertainment | 1 | 0.00% |
| participant_count | 1 | 0.00% |
| Methodology/Process | 1 | 0.00% |
| Emotional entity | 1 | 0.00% |
| labels | 1 | 0.00% |
| Event/ Festival | 1 | 0.00% |
| agricultural output | 1 | 0.00% |
| Sampling rate | 1 | 0.00% |
| Workplace/Location | 1 | 0.00% |
| Website/Category | 1 | 0.00% |
| Task/Process | 1 | 0.00% |
| Self-empowerment | 1 | 0.00% |
| Healing Process | 1 | 0.00% |
| Medications | 1 | 0.00% |
| political group | 1 | 0.00% |
| website identifier | 1 | 0.00% |
| communication channel | 1 | 0.00% |
| osName | 1 | 0.00% |
| osVersion | 1 | 0.00% |
| ipAddress | 1 | 0.00% |
| installationDate | 1 | 0.00% |
| installer | 1 | 0.00% |
| method/strategy | 1 | 0.00% |
| object/asset | 1 | 0.00% |
| occupation/business | 1 | 0.00% |
| Financial Identifier | 1 | 0.00% |
| Financial Section | 1 | 0.00% |
| Character from The Beatles' movie "Yellow Submarine" | 1 | 0.00% |
| exception | 1 | 0.00% |
| Professional Growth | 1 | 0.00% |
| Knowledge/Experience | 1 | 0.00% |
| installation process | 1 | 0.00% |
| portable vaporizer | 1 | 0.00% |
| vape expert | 1 | 0.00% |
| product classification | 1 | 0.00% |
| product characteristic | 1 | 0.00% |
| Economic Situation | 1 | 0.00% |
| technology and business innovation | 1 | 0.00% |
| added value | 1 | 0.00% |
| Political process | 1 | 0.00% |
| Discussion | 1 | 0.00% |
| Tree | 1 | 0.00% |
| bodily process | 1 | 0.00% |
| bodily system | 1 | 0.00% |
| psychological symptom | 1 | 0.00% |
| Selection Criteria | 1 | 0.00% |
| video game edition | 1 | 0.00% |
| video game series | 1 | 0.00% |
| internet feature | 1 | 0.00% |
| location/activity | 1 | 0.00% |
| brake type | 1 | 0.00% |
| branding/marketing | 1 | 0.00% |
| purchase_date | 1 | 0.00% |
| ties | 1 | 0.00% |
| final_standing | 1 | 0.00% |
| win_percentage | 1 | 0.00% |
| coach | 1 | 0.00% |
| Emotion/Activity | 1 | 0.00% |
| Issue/Problem | 1 | 0.00% |
| Multimedia | 1 | 0.00% |
| gold alloy | 1 | 0.00% |
| diamond shape | 1 | 0.00% |
| diamond cut | 1 | 0.00% |
| ring band | 1 | 0.00% |
| diamond setting | 1 | 0.00% |
| Negative Object | 1 | 0.00% |
| Date/Year | 1 | 0.00% |
| Geographic region | 1 | 0.00% |
| Technical specification | 1 | 0.00% |
| digital communication | 1 | 0.00% |
| file hosting service | 1 | 0.00% |
| Social Norm | 1 | 0.00% |
| Plant Entity | 1 | 0.00% |
| Office Complex | 1 | 0.00% |
| Landmark/Location | 1 | 0.00% |
| Housing Project | 1 | 0.00% |
| bandages | 1 | 0.00% |
| gauze | 1 | 0.00% |
| tape | 1 | 0.00% |
| antiseptic | 1 | 0.00% |
| pain_relievers | 1 | 0.00% |
| antihistamines | 1 | 0.00% |
| government department | 1 | 0.00% |
| Government department | 1 | 0.00% |
| Waste | 1 | 0.00% |
| Quote | 1 | 0.00% |
| travel planning | 1 | 0.00% |
| lodging options | 1 | 0.00% |
| privacy regulations | 1 | 0.00% |
| Electronic device component | 1 | 0.00% |
| pet food product | 1 | 0.00% |
| product issue | 1 | 0.00% |
| totalBill | 1 | 0.00% |
| Man-made Landmark | 1 | 0.00% |
| poster | 1 | 0.00% |
| terrain type | 1 | 0.00% |
| Positive Outcome | 1 | 0.00% |
| Color and Material | 1 | 0.00% |
| government fees | 1 | 0.00% |
| Purpose/Use | 1 | 0.00% |
| month and year | 1 | 0.00% |
| discount percentage | 1 | 0.00% |
| discount codes | 1 | 0.00% |
| sense of perception | 1 | 0.00% |
| changeType | 1 | 0.00% |
| amusement park ride | 1 | 0.00% |
| height measurement | 1 | 0.00% |
| roller coaster design | 1 | 0.00% |
| Film Name | 1 | 0.00% |
| Online Resource | 1 | 0.00% |
| Webpage Category | 1 | 0.00% |
| Media Player | 1 | 0.00% |
| gift item | 1 | 0.00% |
| taste sensation | 1 | 0.00% |
| Guarantee/Promise | 1 | 0.00% |
| medium/channel | 1 | 0.00% |
| organization/institution | 1 | 0.00% |
| research capability | 1 | 0.00% |
| Legal provision | 1 | 0.00% |
| score: Rating given on a scale from 0 to 5 | 1 | 0.00% |
| Landmark/Building | 1 | 0.00% |
| Ranking/Title | 1 | 0.00% |
| Name/Person | 1 | 0.00% |
| Gift Certificate | 1 | 0.00% |
| Gift Certificate Template | 1 | 0.00% |
| culture/language | 1 | 0.00% |
| culture/ethnicity | 1 | 0.00% |
| Service Requirement | 1 | 0.00% |
| statistical model | 1 | 0.00% |
| Plant supplement | 1 | 0.00% |
| Tool/Machinery | 1 | 0.00% |
| religious practice | 1 | 0.00% |
| social event attendees | 1 | 0.00% |
| social connections | 1 | 0.00% |
| belief systems | 1 | 0.00% |
| headliner | 1 | 0.00% |
| Magic Trick Device | 1 | 0.00% |
| Magic Supplies Company | 1 | 0.00% |
| Comparison tool | 1 | 0.00% |
| Music Composition | 1 | 0.00% |
| Earphone Model | 1 | 0.00% |
| Connectivity Type | 1 | 0.00% |
| game_description | 1 | 0.00% |
| gameplay_elements.towers.0.tower_type | 1 | 0.00% |
| gameplay_elements.towers.0.placement_strategy | 1 | 0.00% |
| gameplay_elements.towers.0.abilities | 1 | 0.00% |
| gameplay_elements.enemies.0.zombie_type | 1 | 0.00% |
| gameplay_elements.enemies.0.characteristics | 1 | 0.00% |
| gameplay_elements.enemies.1.zombie_type | 1 | 0.00% |
| gameplay_elements.enemies.1.characteristics | 1 | 0.00% |
| gameplay_elements.enemies.2.zombie_type | 1 | 0.00% |
| gameplay_elements.enemies.2.characteristics | 1 | 0.00% |
| gameplay_elements.enemies.3.zombie_type | 1 | 0.00% |
| gameplay_elements.enemies.3.characteristics | 1 | 0.00% |
| gameplay_elements.strategy | 1 | 0.00% |
| project/program | 1 | 0.00% |
| trait/ability | 1 | 0.00% |
| Car User | 1 | 0.00% |
| political right | 1 | 0.00% |
| Racing event | 1 | 0.00% |
| Racing venue | 1 | 0.00% |
| Racer/team owner | 1 | 0.00% |
| Historical Place | 1 | 0.00% |
| Entertainer | 1 | 0.00% |
| Entertainment piece | 1 | 0.00% |
| broadcast_network | 1 | 0.00% |
| possession_rate | 1 | 0.00% |
| Personal identifier | 1 | 0.00% |
| byproduct | 1 | 0.00% |
| Social phenomena | 1 | 0.00% |
| General population | 1 | 0.00% |
| Societal progress | 1 | 0.00% |
| Social behavior | 1 | 0.00% |
| Positive state | 1 | 0.00% |
| Football Competition | 1 | 0.00% |
| exercise technique | 1 | 0.00% |
| problem-solving task | 1 | 0.00% |
| game entities | 1 | 0.00% |
| game UI element | 1 | 0.00% |
| Age group and subculture | 1 | 0.00% |
| horse | 1 | 0.00% |
| Interest/Passion | 1 | 0.00% |
| recommendation/opinion | 1 | 0.00% |
| Decorative Art | 1 | 0.00% |
| Psychology | 1 | 0.00% |
| Sociology | 1 | 0.00% |
| Trade Exhibition | 1 | 0.00% |
| Natural Disaster Entity | 1 | 0.00% |
| Mineral Resource Entity | 1 | 0.00% |
| Geographic Location Entity | 1 | 0.00% |
| Adjective Entity | 1 | 0.00% |
| facial hair | 1 | 0.00% |
| floor level | 1 | 0.00% |
| Mythical place | 1 | 0.00% |
| Atmospheric condition | 1 | 0.00% |
| hotel name | 1 | 0.00% |
| Business resource | 1 | 0.00% |
| Status/Attribute | 1 | 0.00% |
| Industrial Product | 1 | 0.00% |
| digital object identifier (DOI) | 1 | 0.00% |
| Entertainment Equipment | 1 | 0.00% |
| Research and Development Activity | 1 | 0.00% |
| Chapter | 1 | 0.00% |
| Verse | 1 | 0.00% |
| Movie genre | 1 | 0.00% |
| Movie sub-genre | 1 | 0.00% |
| Movie type | 1 | 0.00% |
| housing unit | 1 | 0.00% |
| brand logo | 1 | 0.00% |
| Fraudulent entities | 1 | 0.00% |
| Online personalities | 1 | 0.00% |
| Facial recognition technology | 1 | 0.00% |
| restroom facility | 1 | 0.00% |
| renovation | 1 | 0.00% |
| genetic material | 1 | 0.00% |
| non-coding regions | 1 | 0.00% |
| mtDNA | 1 | 0.00% |
| defense system | 1 | 0.00% |
| oxidative stress | 1 | 0.00% |
| organelles | 1 | 0.00% |
| reactive oxidative stress | 1 | 0.00% |
| architectural style | 1 | 0.00% |
| Proficiency Level | 1 | 0.00% |
| data set | 1 | 0.00% |
| unit of time | 1 | 0.00% |
| unit of money | 1 | 0.00% |
| Athlete and entrepreneur | 1 | 0.00% |
| kitchen equipment | 1 | 0.00% |
| Fundraising Target | 1 | 0.00% |
| Transport Vehicle | 1 | 0.00% |
| Target Beneficiary Group | 1 | 0.00% |
| Fundraising Campaign | 1 | 0.00% |
| software features | 1 | 0.00% |
| media player | 1 | 0.00% |
| data collections | 1 | 0.00% |
| Creative agency | 1 | 0.00% |
| communication technology | 1 | 0.00% |
| navigation system | 1 | 0.00% |
| Safety Concerns | 1 | 0.00% |
| Pharmacy Operations | 1 | 0.00% |
| Nursing Practice | 1 | 0.00% |
| Music piece | 1 | 0.00% |
| controversy.victim_actresses.7.actress_name | 1 | 0.00% |
| controversy.krk_allegation.comment | 1 | 0.00% |
| controversy.person_involved.0.role | 1 | 0.00% |
| controversy.victim_actresses.1.actress_name | 1 | 0.00% |
| controversy.victim_actresses.3.krk_comment | 1 | 0.00% |
| controversy.victim_actresses.1.krk_comment | 1 | 0.00% |
| controversy.person_involved.0.name | 1 | 0.00% |
| controversy.victim_actresses.5.krk_comment | 1 | 0.00% |
| controversy.victim_actresses.6.krk_comment | 1 | 0.00% |
| controversy.person_involved.2.name | 1 | 0.00% |
| controversy.krk_allegation.target | 1 | 0.00% |
| controversy.victim_actresses.2.krk_comment | 1 | 0.00% |
| controversy.victim_actresses.3.actress_name | 1 | 0.00% |
| controversy.victim_actresses.6.actress_name | 1 | 0.00% |
| time-based entity | 1 | 0.00% |
| maintenanceType | 1 | 0.00% |
| maintenanceDate | 1 | 0.00% |
| maintenanceDuration | 1 | 0.00% |
| maintenanceCost | 1 | 0.00% |
| performedBy | 1 | 0.00% |
| affiliation.department | 1 | 0.00% |
| tribute_to.person_name | 1 | 0.00% |
| affiliation.organization_name | 1 | 0.00% |
| establishment_resolution | 1 | 0.00% |
| sectorial_library_system | 1 | 0.00% |
| tribute_to.profession | 1 | 0.00% |
| tribute_to.location | 1 | 0.00% |
| entertainment medium | 1 | 0.00% |
| hidden agenda | 1 | 0.00% |
| comedic tone | 1 | 0.00% |
| creators | 1 | 0.00% |
| possessions | 1 | 0.00% |
| solitude | 1 | 0.00% |
| Door Type | 1 | 0.00% |
| completeness | 1 | 0.00% |
| Entity Type: Company/Organization | 1 | 0.00% |
| ref_code | 1 | 0.00% |
| pop culture | 1 | 0.00% |
| family relationships | 1 | 0.00% |
| generational differences | 1 | 0.00% |
| Border security | 1 | 0.00% |
| recurring annoyance | 1 | 0.00% |
| software testing | 1 | 0.00% |
| cloud computing | 1 | 0.00% |
| performance evaluation | 1 | 0.00% |
| genre | 1 | 0.00% |
| Aesthetic Effect | 1 | 0.00% |
| Technological Advancement | 1 | 0.00% |
| Durability | 1 | 0.00% |
| Ammunition Capacity | 1 | 0.00% |
| Magazine Material | 1 | 0.00% |
| Percentage/Number | 1 | 0.00% |
| Goal/Strategy | 1 | 0.00% |
| academic pursuit | 1 | 0.00% |
| company/business | 1 | 0.00% |
| general data | 1 | 0.00% |
| approach to design | 1 | 0.00% |
| Musical | 1 | 0.00% |
| military forces | 1 | 0.00% |
| alternative medicine | 1 | 0.00% |
| physiology | 1 | 0.00% |
| pharmacology | 1 | 0.00% |
| botany | 1 | 0.00% |
| Contact | 1 | 0.00% |
| Caravan Park | 1 | 0.00% |
| plant family | 1 | 0.00% |
| alternative name for corn | 1 | 0.00% |
| specific plant species | 1 | 0.00% |
| Software development | 1 | 0.00% |
| Abilities & Aptitudes | 1 | 0.00% |
| athlete group | 1 | 0.00% |
| spectator group | 1 | 0.00% |
| supporter group | 1 | 0.00% |
| transition point | 1 | 0.00% |
| ability/attribute | 1 | 0.00% |
| season/month | 1 | 0.00% |
| Power/Ability | 1 | 0.00% |
| Living being | 1 | 0.00% |
| Criminals/Threats | 1 | 0.00% |
| Military Unit/Rescue Team | 1 | 0.00% |
| People/Travelers | 1 | 0.00% |
| Status/State | 1 | 0.00% |
| trade policy | 1 | 0.00% |
| Quality/Ranking | 1 | 0.00% |
| Expert/Person | 1 | 0.00% |
| Man-Made Feature | 1 | 0.00% |
| natural landmark | 1 | 0.00% |
| Activity/Medication | 1 | 0.00% |
| town.population.density_per_sq_km | 1 | 0.00% |
| town.name | 1 | 0.00% |
| town.area_sq_km | 1 | 0.00% |
| town.population.year | 1 | 0.00% |
| town.location | 1 | 0.00% |
| job category | 1 | 0.00% |
| Technology/Product | 1 | 0.00% |
| orderNumber | 1 | 0.00% |
| provider_name: Name of the healthcare provider | 1 | 0.00% |
| diagnosis_code: Medical code indicating the patient's condition | 1 | 0.00% |
| treatment_type: Type of medical treatment provided | 1 | 0.00% |
| policy_id: Insurance policy number associated with the claim | 1 | 0.00% |
| address: Full address of the provider including street city state zip | 1 | 0.00% |
| election process | 1 | 0.00% |
| Administrative Body | 1 | 0.00% |
| Technology/Infrastructure | 1 | 0.00% |
| Event/Publication | 1 | 0.00% |
| food recipe | 1 | 0.00% |
| software enhancement | 1 | 0.00% |
| software add-ons | 1 | 0.00% |
| Success | 1 | 0.00% |
| Expectation | 1 | 0.00% |
| Kitchen Equipment | 1 | 0.00% |
| article.title | 1 | 0.00% |
| journal.title | 1 | 0.00% |
| journal.issn | 1 | 0.00% |
| article.publicationDetails.pages | 1 | 0.00% |
| article.author.givenName | 1 | 0.00% |
| article.publicationDetails.issue | 1 | 0.00% |
| Civil Rights Activist | 1 | 0.00% |
| Art/Music Genre | 1 | 0.00% |
| refreshment | 1 | 0.00% |
| education and learning | 1 | 0.00% |
| political entities | 1 | 0.00% |
| global platform | 1 | 0.00% |
| intergovernmental organization | 1 | 0.00% |
| personalisation | 1 | 0.00% |
| puzzle type | 1 | 0.00% |
| user support | 1 | 0.00% |
| location/building | 1 | 0.00% |
| base or groundwork | 1 | 0.00% |
| personal care routine | 1 | 0.00% |
| Date/time entity | 1 | 0.00% |
| Industrial entity | 1 | 0.00% |
| Risk Level | 1 | 0.00% |
| Geological Formation | 1 | 0.00% |
| Golf Course Feature | 1 | 0.00% |
| E-commerce Tool | 1 | 0.00% |
| File organization | 1 | 0.00% |
| File system path | 1 | 0.00% |
| company: Name of the startup receiving funding | 1 | 0.00% |
| lead_investor: List of investors leading the round | 1 | 0.00% |
| Religious entity | 1 | 0.00% |
| enrollment_notes | 1 | 0.00% |
| Holiday preparation | 1 | 0.00% |
| Journalism | 1 | 0.00% |
| task/job performance | 1 | 0.00% |
| human body part | 1 | 0.00% |
| puzzle | 1 | 0.00% |
| contract terms | 1 | 0.00% |
| medical facilities | 1 | 0.00% |
| amphibious feature | 1 | 0.00% |
| Scientific term | 1 | 0.00% |
| list | 1 | 0.00% |
| job title/group | 1 | 0.00% |
| academic title | 1 | 0.00% |
| Airline Staff | 1 | 0.00% |
| Flight Issue | 1 | 0.00% |
| Fashion magazine | 1 | 0.00% |
| Fashion designer | 1 | 0.00% |
| cook | 1 | 0.00% |
| baker | 1 | 0.00% |
| Health Concern | 1 | 0.00% |
| Leadership | 1 | 0.00% |
| Influence | 1 | 0.00% |
| clothes | 1 | 0.00% |
| mental activity | 1 | 0.00% |
| game component | 1 | 0.00% |
| musical instruments | 1 | 0.00% |
| musical instrument model | 1 | 0.00% |
| asteroid.named_after.nationality | 1 | 0.00% |
| asteroid.named_after.profession | 1 | 0.00% |
| city: City where patient resides | 1 | 0.00% |
| onset_date: Date when symptoms first appeared | 1 | 0.00% |
| address: Street address of the patient | 1 | 0.00% |
| wearable technology | 1 | 0.00% |
| Journal title | 1 | 0.00% |
| geographic area | 1 | 0.00% |
| artistName | 1 | 0.00% |
| concertVenue | 1 | 0.00% |
| eventName | 1 | 0.00% |
| tour | 1 | 0.00% |
| idNumber | 1 | 0.00% |
| government expenditure | 1 | 0.00% |
| mode of action | 1 | 0.00% |
| Financial services | 1 | 0.00% |
| Approval status | 1 | 0.00% |
| Availability status | 1 | 0.00% |
| Business entities | 1 | 0.00% |
| Registration status | 1 | 0.00% |
| seasoning | 1 | 0.00% |
| cooking mistake | 1 | 0.00% |
| tool component | 1 | 0.00% |
| Holiday event | 1 | 0.00% |
| Profession/Role | 1 | 0.00% |
| In-app purchase services | 1 | 0.00% |
| Orbital Path | 1 | 0.00% |
| Moons | 1 | 0.00% |
| Brand/Product Line | 1 | 0.00% |
| harmful substances | 1 | 0.00% |
| skincare brand | 1 | 0.00% |
| Tool/Platform | 1 | 0.00% |
| magnitudeorindex | 1 | 0.00% |
| cleaning activity | 1 | 0.00% |
| decluttering activity | 1 | 0.00% |
| Leisure and Entertainment Venue | 1 | 0.00% |
| monitoring_focus: Key adverse effects or outcomes to monitor during implementation | 1 | 0.00% |
| target_population: Age and health condition group the guideline targets | 1 | 0.00% |
| origin: Country or region where the guideline was issued | 1 | 0.00% |
| recommendations: Number of specific clinical recommendations included | 1 | 0.00% |
| version: Current active version of the guideline | 1 | 0.00% |
| title: Full title of the clinical guideline document | 1 | 0.00% |
| language: Primary language of the guideline | 1 | 0.00% |
| Online Content | 1 | 0.00% |
| hiring process | 1 | 0.00% |
| process scope | 1 | 0.00% |
| audience/readership | 1 | 0.00% |
| tone/mood | 1 | 0.00% |
| Bathroom Decor Item | 1 | 0.00% |
| stockstatus | 1 | 0.00% |
| available_units | 1 | 0.00% |
| warranty | 1 | 0.00% |
| Support System | 1 | 0.00% |
| Manner/Quality | 1 | 0.00% |
| Laptop brand/model | 1 | 0.00% |
| Computer storage capacity | 1 | 0.00% |
| Music Award | 1 | 0.00% |
| activationId | 1 | 0.00% |
| msisdn | 1 | 0.00% |
| activationStatus | 1 | 0.00% |
| dataUsage | 1 | 0.00% |
| abstract space | 1 | 0.00% |
| boat part | 1 | 0.00% |
| boat accessory | 1 | 0.00% |
| brewing process step | 1 | 0.00% |
| brewing equipment | 1 | 0.00% |
| brewing problem | 1 | 0.00% |
| Magical beings | 1 | 0.00% |
| Barber shop | 1 | 0.00% |
| stage name | 1 | 0.00% |
| product_line | 1 | 0.00% |
| manner/adjective | 1 | 0.00% |
| Cancellation | 1 | 0.00% |
| Refund | 1 | 0.00% |
| Entertainment equipment | 1 | 0.00% |
| Bathroom fixture | 1 | 0.00% |
| Environmental setting | 1 | 0.00% |
| guarantee type | 1 | 0.00% |
| review type | 1 | 0.00% |
| Type of Connector | 1 | 0.00% |
| wine producers | 1 | 0.00% |
| Psychotherapy | 1 | 0.00% |
| water body | 1 | 0.00% |
| beliefs | 1 | 0.00% |
| type of vacation | 1 | 0.00% |
| management style | 1 | 0.00% |
| aspect of planning | 1 | 0.00% |
| desired outcome | 1 | 0.00% |
| law/rule | 1 | 0.00% |
| Personal Identification/Information | 1 | 0.00% |
| patient group | 1 | 0.00% |
| Cooperative Agency | 1 | 0.00% |
| unexpected expenses | 1 | 0.00% |
| bureaucratic requirements | 1 | 0.00% |
| long-term obligation | 1 | 0.00% |
| significant amount of money | 1 | 0.00% |
| Time Period/Event | 1 | 0.00% |
| Recreational Vehicle | 1 | 0.00% |
| Enthusiasts | 1 | 0.00% |
| pricing policy | 1 | 0.00% |
| inr_value | 1 | 0.00% |
| drug_involved | 1 | 0.00% |
| Curtain Accessories | 1 | 0.00% |
| Curtain Rod Set | 1 | 0.00% |
| Version Control System | 1 | 0.00% |
| Expression of surprise/fear | 1 | 0.00% |
| Direction/location | 1 | 0.00% |
| Emotional struggles | 1 | 0.00% |
| website ranking service | 1 | 0.00% |
| website category | 1 | 0.00% |
| Regissör | 1 | 0.00% |
| Agent | 1 | 0.00% |
| Dinosaur Diet Type | 1 | 0.00% |
| Time Period/Location | 1 | 0.00% |
| poem.source | 1 | 0.00% |
| Organisation | 1 | 0.00% |
| educational approach | 1 | 0.00% |
| product/service quality | 1 | 0.00% |
| Musical piece | 1 | 0.00% |
| event/entity | 1 | 0.00% |
| time/entity | 1 | 0.00% |
| documents | 1 | 0.00% |
| cutting tool | 1 | 0.00% |
| greenery | 1 | 0.00% |
| spiritual symbol | 1 | 0.00% |
| SEO term | 1 | 0.00% |
| exhibit | 1 | 0.00% |
| Sports news outlet | 1 | 0.00% |
| floral decoration | 1 | 0.00% |
| Mutual Fund | 1 | 0.00% |
| Criticism | 1 | 0.00% |
| Reaction | 1 | 0.00% |
| Financial success | 1 | 0.00% |
| Conference Title | 1 | 0.00% |
| Research Field | 1 | 0.00% |
| Diagnosis/Test | 1 | 0.00% |
| environmental hazard | 1 | 0.00% |
| Treatment approach | 1 | 0.00% |
| home office | 1 | 0.00% |
| technological advancements | 1 | 0.00% |
| Biblical Text | 1 | 0.00% |
| Political Agreement | 1 | 0.00% |
| relationship_to_subject | 1 | 0.00% |
| Heating system | 1 | 0.00% |
| Entrance/Exit | 1 | 0.00% |
| Transactional Entity | 1 | 0.00% |
| Terrorist organization | 1 | 0.00% |
| Room Rate | 1 | 0.00% |
| affected_entity: Organization or business impacted by the violation | 1 | 0.00% |
| description: Text explaining the nature and impact of the violation | 1 | 0.00% |
| violation_type: Category of compliance issue identified | 1 | 0.00% |
| portion of text | 1 | 0.00% |
| state of completion | 1 | 0.00% |
| music composition | 1 | 0.00% |
| work of art | 1 | 0.00% |
| quality and value | 1 | 0.00% |
| Designers | 1 | 0.00% |
| Photography subjects | 1 | 0.00% |
| Fashion items | 1 | 0.00% |
| law | 1 | 0.00% |
| Award Recipients | 1 | 0.00% |
| Weather Event | 1 | 0.00% |
| visual decoration | 1 | 0.00% |
| Baseball position | 1 | 0.00% |
| Baseball play | 1 | 0.00% |
| Workout Equipment | 1 | 0.00% |
| Exercise Routine | 1 | 0.00% |
| Exercise Technique | 1 | 0.00% |
| Exercise Surface | 1 | 0.00% |
| Fitness Goal | 1 | 0.00% |
| location/nature | 1 | 0.00% |
| album_title | 1 | 0.00% |
| sales and marketing | 1 | 0.00% |
| advertising and marketing | 1 | 0.00% |
| management and leadership | 1 | 0.00% |
| playful activity | 1 | 0.00% |
| task details | 1 | 0.00% |
| Legislation/Policy | 1 | 0.00% |
| traffic congestion | 1 | 0.00% |
| Recreation/Club Membership | 1 | 0.00% |
| source of uninterrupted energy | 1 | 0.00% |
| ground wire | 1 | 0.00% |
| electrical device | 1 | 0.00% |
| carriageNumber | 1 | 0.00% |
| trainNumber | 1 | 0.00% |
| trainName | 1 | 0.00% |
| departureStation | 1 | 0.00% |
| arrivalStation | 1 | 0.00% |
| business organization | 1 | 0.00% |
| legal violation | 1 | 0.00% |
| artist organization | 1 | 0.00% |
| event invitation | 1 | 0.00% |
| written language | 1 | 0.00% |
| Legal Form | 1 | 0.00% |
| Identification Number | 1 | 0.00% |
| Establishment Date | 1 | 0.00% |
| destruction | 1 | 0.00% |
| Pet breed | 1 | 0.00% |
| Healthcare policy | 1 | 0.00% |
| Corporate influence | 1 | 0.00% |
| Social problem | 1 | 0.00% |
| discharge_destination | 1 | 0.00% |
| insurance_provider | 1 | 0.00% |
| Acquirer | 1 | 0.00% |
| process changes | 1 | 0.00% |
| permit type | 1 | 0.00% |
| recipe title | 1 | 0.00% |
| craving or desire | 1 | 0.00% |
| frosting flavor | 1 | 0.00% |
| sweetness level | 1 | 0.00% |
| sustainability.sourced | 1 | 0.00% |
| ticket purchase | 1 | 0.00% |
| ticket holder | 1 | 0.00% |
| ticket count | 1 | 0.00% |
| entry point | 1 | 0.00% |
| ticket identifier | 1 | 0.00% |
| ATMs | 1 | 0.00% |
| Communities | 1 | 0.00% |
| Engineering failure | 1 | 0.00% |
| Investigation document | 1 | 0.00% |
| Architectural plans | 1 | 0.00% |
| Quality control | 1 | 0.00% |
| output | 1 | 0.00% |
| Cartoon Characters | 1 | 0.00% |
| plant material | 1 | 0.00% |
| nut | 1 | 0.00% |
| output.unit | 1 | 0.00% |
| relationship aspect | 1 | 0.00% |
| Labor exploitation | 1 | 0.00% |
| Labor rights | 1 | 0.00% |
| Corporate misconduct | 1 | 0.00% |
| Labor strikes | 1 | 0.00% |
| Product safety issues | 1 | 0.00% |
| Person/Place | 1 | 0.00% |
| travel destinations | 1 | 0.00% |
| Magazine edition | 1 | 0.00% |
| Magazine size | 1 | 0.00% |
| technology/science | 1 | 0.00% |
| scientific practice | 1 | 0.00% |
| sports division | 1 | 0.00% |
| sports organization | 1 | 0.00% |
| Institute Name | 1 | 0.00% |
| Credit History | 1 | 0.00% |
| outlier | 1 | 0.00% |
| Human resource | 1 | 0.00% |
| cooking term | 1 | 0.00% |
| Network-attached storage device | 1 | 0.00% |
| Work in progress | 1 | 0.00% |
| beekeeping equipment | 1 | 0.00% |
| bee nucleus colonies | 1 | 0.00% |
| timeline event | 1 | 0.00% |
| registrant | 1 | 0.00% |
| infringer | 1 | 0.00% |
| registered_mark | 1 | 0.00% |
| registrantusptoid | 1 | 0.00% |
| rental_unit | 1 | 0.00% |
| bedrooms | 1 | 0.00% |
| outdoor_space | 1 | 0.00% |
| Serendipity | 1 | 0.00% |
| Social Media User | 1 | 0.00% |
| Architecture/Structure | 1 | 0.00% |
| Animal Breed | 1 | 0.00% |
| Application/Website | 1 | 0.00% |
| Hashtag | 1 | 0.00% |
| Building/Infrastructure | 1 | 0.00% |
| Postal Code/Zip Code | 1 | 0.00% |
| residential area | 1 | 0.00% |
| product prototypes | 1 | 0.00% |
| performance metrics | 1 | 0.00% |
| Epic | 1 | 0.00% |
| Tradition | 1 | 0.00% |
| Issue/Challenge | 1 | 0.00% |
| GP training organization | 1 | 0.00% |
| Promotional material | 1 | 0.00% |
| Video game console | 1 | 0.00% |
| sports enthusiasts | 1 | 0.00% |
| payday loan provider | 1 | 0.00% |
| gaming feature | 1 | 0.00% |
| racing series | 1 | 0.00% |
| advancement | 1 | 0.00% |
| Cleaner | 1 | 0.00% |
| event/outcome | 1 | 0.00% |
| Marketing/Business | 1 | 0.00% |
| compensation package | 1 | 0.00% |
| job applicants | 1 | 0.00% |
| woodworking technique | 1 | 0.00% |
| public service | 1 | 0.00% |
| Tools | 1 | 0.00% |
| traffic control device | 1 | 0.00% |
| educational programs | 1 | 0.00% |
| Surveillance | 1 | 0.00% |
| physical action | 1 | 0.00% |
| workout type | 1 | 0.00% |
| quality or characteristic | 1 | 0.00% |
| emotion/activity | 1 | 0.00% |
| Garden/Plant | 1 | 0.00% |
| Email Protocol | 1 | 0.00% |
| Campground | 1 | 0.00% |
| Camping facility | 1 | 0.00% |
| National forest | 1 | 0.00% |
| Tent camping area | 1 | 0.00% |
| RV camping area | 1 | 0.00% |
| business hours | 1 | 0.00% |
| subgenre of photography | 1 | 0.00% |
| Online Game Store | 1 | 0.00% |
| Graphic Designer | 1 | 0.00% |
| Decade | 1 | 0.00% |
| object/action | 1 | 0.00% |
| Event/Feeling | 1 | 0.00% |
| Possession/Feeling | 1 | 0.00% |
| blood_pressure: Blood pressure in format systolic/diastolic | 1 | 0.00% |
| heart_rate: Number of heartbeats per minute | 1 | 0.00% |
| date: Date of the patient visit | 1 | 0.00% |
| address: Street address including city and zip code | 1 | 0.00% |
| spo2: Oxygen saturation percentage | 1 | 0.00% |
| personal aspiration | 1 | 0.00% |
| social reaction | 1 | 0.00% |
| mechanical entity | 1 | 0.00% |
| technological entity | 1 | 0.00% |
| National Story | 1 | 0.00% |
| International Story | 1 | 0.00% |
| Political Expertise | 1 | 0.00% |
| establishment_approval | 1 | 0.00% |
| abolished_entity | 1 | 0.00% |
| administrative_level | 1 | 0.00% |
| Real Estate Team | 1 | 0.00% |
| Clients | 1 | 0.00% |
| shipping cost | 1 | 0.00% |
| stock availability | 1 | 0.00% |
| device_id: Unique identifier for the medical device | 1 | 0.00% |
| device_name: Name of the medical device model | 1 | 0.00% |
| facility: Facility name and address where device is located | 1 | 0.00% |
| purchase_date: Date when the device was purchased | 1 | 0.00% |
| expiry_date: Date when the device's warranty or use period ends | 1 | 0.00% |
| Office/Location | 1 | 0.00% |
| Metaphorical Power | 1 | 0.00% |
| Surrendering/Retreating | 1 | 0.00% |
| Profession/Action | 1 | 0.00% |
| Traps/Obstacles | 1 | 0.00% |
| Nursing Home Resident | 1 | 0.00% |
| Long-term Care Facility | 1 | 0.00% |
| Quality of Life | 1 | 0.00% |
| Personal Traits | 1 | 0.00% |
| Trial Period | 1 | 0.00% |
| Customer Interaction | 1 | 0.00% |
| View Count | 1 | 0.00% |
| Perspective | 1 | 0.00% |
| Dish/Food | 1 | 0.00% |
| Self-identity | 1 | 0.00% |
| Circumstance | 1 | 0.00% |
| human rights issue | 1 | 0.00% |
| Clothing product | 1 | 0.00% |
| Game progression | 1 | 0.00% |
| Design principle | 1 | 0.00% |
| Location technology | 1 | 0.00% |
| object entity | 1 | 0.00% |
| Music Venue | 1 | 0.00% |
| Photographs | 1 | 0.00% |
| Language program | 1 | 0.00% |
| child with disability | 1 | 0.00% |
| caregiving | 1 | 0.00% |
| Building/Property | 1 | 0.00% |
| asteroid.named_after.location.country | 1 | 0.00% |
| asteroid.named_after.location.province | 1 | 0.00% |
| discovery.observatory | 1 | 0.00% |
| discovery.astronomer.nationality | 1 | 0.00% |
| Industrial Facility | 1 | 0.00% |
| Industrial Process | 1 | 0.00% |
| Industrial Equipment | 1 | 0.00% |
| AI Tool | 1 | 0.00% |
| company.services | 1 | 0.00% |
| company.founders.1.name | 1 | 0.00% |
| company.founding_location.region | 1 | 0.00% |
| company.establishment_date | 1 | 0.00% |
| company.description | 1 | 0.00% |
| company.transformation_date | 1 | 0.00% |
| Power Source | 1 | 0.00% |
| Beauty tool | 1 | 0.00% |
| cost: Total cost paid for the training program | 1 | 0.00% |
| employee_name: Full name of the enrolled employee | 1 | 0.00% |
| Geography/Location | 1 | 0.00% |
| sports/object | 1 | 0.00% |
| Website/Organization | 1 | 0.00% |
| major | 1 | 0.00% |
| minor | 1 | 0.00% |
| patch | 1 | 0.00% |
| knownIssues | 1 | 0.00% |
| hose accessory | 1 | 0.00% |
| hose length | 1 | 0.00% |
| material protection | 1 | 0.00% |
| Economic Factors | 1 | 0.00% |
| Macroeconomic Equilibrium | 1 | 0.00% |
| Economic Theory | 1 | 0.00% |
| footnote number | 1 | 0.00% |
| 3D printer part | 1 | 0.00% |
| Furniture Style | 1 | 0.00% |
| Building Height | 1 | 0.00% |
| Time and Date Entity | 1 | 0.00% |
| Day of the Week Entity | 1 | 0.00% |
| Date Entity | 1 | 0.00% |
| Year Entity | 1 | 0.00% |
| Field of Study Entity | 1 | 0.00% |
| therapeutic treatments | 1 | 0.00% |
| Bank ATM | 1 | 0.00% |
| Field/Industry | 1 | 0.00% |
| Glassware Product | 1 | 0.00% |
| food substance | 1 | 0.00% |
| Virtual Instrument | 1 | 0.00% |
| poker move | 1 | 0.00% |
| poker term | 1 | 0.00% |
| sportsperson | 1 | 0.00% |
| supply_id | 1 | 0.00% |
| Perception/Feeling | 1 | 0.00% |
| Flavor/Component | 1 | 0.00% |
| Screen Resolution | 1 | 0.00% |
| Athletic Achievement | 1 | 0.00% |
| Travel destination | 1 | 0.00% |
| Veterinary Medicine | 1 | 0.00% |
| Website/Internet | 1 | 0.00% |
| fashion item accessory | 1 | 0.00% |
| Bedding | 1 | 0.00% |
| Living Beings | 1 | 0.00% |
| Music venue | 1 | 0.00% |
| Entity name | 1 | 0.00% |
| music piece | 1 | 0.00% |
| Stickers | 1 | 0.00% |
| Culture/Style | 1 | 0.00% |
| Shop | 1 | 0.00% |
| Religious Service | 1 | 0.00% |
| planNumber | 1 | 0.00% |
| planDate | 1 | 0.00% |
| planDueDate | 1 | 0.00% |
| planStatus | 1 | 0.00% |
| Ethnic/Community | 1 | 0.00% |
| Contract agreement | 1 | 0.00% |
| Property condition | 1 | 0.00% |
| Property size | 1 | 0.00% |
| Action/Outcome | 1 | 0.00% |
| employee commitment | 1 | 0.00% |
| corporate culture | 1 | 0.00% |
| contracted workers | 1 | 0.00% |
| Astronomy Entity | 1 | 0.00% |
| Attribute/Trait | 1 | 0.00% |
| Educational Course | 1 | 0.00% |
| Creative Workshop | 1 | 0.00% |
| Spirituality | 1 | 0.00% |
| company location | 1 | 0.00% |
| economic event | 1 | 0.00% |
| profession/academic title | 1 | 0.00% |
| event/series | 1 | 0.00% |
| personal time | 1 | 0.00% |
| Website Optimization | 1 | 0.00% |
| SERP Ranking Improvement | 1 | 0.00% |
| Physical remains | 1 | 0.00% |
| personal habit/entity | 1 | 0.00% |
| profession/entity | 1 | 0.00% |
| reservation or appointment | 1 | 0.00% |
| diet | 1 | 0.00% |
| Soil Condition | 1 | 0.00% |
| Environmental Goal | 1 | 0.00% |
| Emotional sensation | 1 | 0.00% |
| Outcome/event | 1 | 0.00% |
| Human/Mutant | 1 | 0.00% |
| network port | 1 | 0.00% |
| future attendees | 1 | 0.00% |
| ideal attendees | 1 | 0.00% |
| Military Base | 1 | 0.00% |
| News Agency | 1 | 0.00% |
| Fishing Rod | 1 | 0.00% |
| Desire/Emotion | 1 | 0.00% |
| investigator | 1 | 0.00% |
| pathway_id | 1 | 0.00% |
| date_initiated | 1 | 0.00% |
| News organization | 1 | 0.00% |
| fictional beings | 1 | 0.00% |
| Launch Center | 1 | 0.00% |
| founder_id | 1 | 0.00% |
| Game Expansion | 1 | 0.00% |
| Musical Performance | 1 | 0.00% |
| Musical Notation | 1 | 0.00% |
| Publication Code | 1 | 0.00% |
| Person/Online User | 1 | 0.00% |
| Media/Entertainment | 1 | 0.00% |
| Emotion/Reaction | 1 | 0.00% |
| performance area | 1 | 0.00% |
| sales policy | 1 | 0.00% |
| Sports team roster | 1 | 0.00% |
| guests | 1 | 0.00% |
| item_description.setup_charge.amount | 1 | 0.00% |
| item_description.shipping.ready_to_ship_in.quantity | 1 | 0.00% |
| item_description.features | 1 | 0.00% |
| item_description.shipping.rush_details | 1 | 0.00% |
| item_description.item_id | 1 | 0.00% |
| item_description.weight.value | 1 | 0.00% |
| item_description.shipping.ready_to_ship_in.unit | 1 | 0.00% |
| item_description.name | 1 | 0.00% |
| Manga Title | 1 | 0.00% |
| species.specific_epithet | 1 | 0.00% |
| species.taxonomy.subfamily | 1 | 0.00% |
| species.full_name | 1 | 0.00% |
| event_id: Conference or symposium identifier for the accepted abstract | 1 | 0.00% |
| recipient: Name of the person receiving the confirmation | 1 | 0.00% |
| submission_id: Unique identifier assigned to the abstract submission | 1 | 0.00% |
| sender: Name of the person sending the confirmation | 1 | 0.00% |
| Demographic feature | 1 | 0.00% |
| review_length | 1 | 0.00% |
| restaurant_name | 1 | 0.00% |
| cuisine_type | 1 | 0.00% |
| restaurant chain | 1 | 0.00% |
| Social media engagement | 1 | 0.00% |
| Social media sharing | 1 | 0.00% |
| plant name | 1 | 0.00% |
| location name | 1 | 0.00% |
| animal product | 1 | 0.00% |
| personal data | 1 | 0.00% |
| crime type | 1 | 0.00% |
| law enforcement activity | 1 | 0.00% |
| restoration | 1 | 0.00% |
| government aid | 1 | 0.00% |
| inside influences | 1 | 0.00% |
| instructional materials | 1 | 0.00% |
| installation guide | 1 | 0.00% |
| public speaking | 1 | 0.00% |
| People Count | 1 | 0.00% |
| Volunteer Recognition | 1 | 0.00% |
| labeling system | 1 | 0.00% |
| Artwork Element | 1 | 0.00% |
| Sports Level | 1 | 0.00% |
| Sports Teams | 1 | 0.00% |
| Problem/Constraint | 1 | 0.00% |
| merchant | 1 | 0.00% |
| Crochet Block | 1 | 0.00% |
| laboratory | 1 | 0.00% |
| skill/attribute | 1 | 0.00% |
| compensation/financial | 1 | 0.00% |
| Refugee Camp | 1 | 0.00% |
| Job/Work | 1 | 0.00% |
| e-commerce site | 1 | 0.00% |
| Fitness activity | 1 | 0.00% |
| Hazardous activity | 1 | 0.00% |
| product listing | 1 | 0.00% |
| podcast | 1 | 0.00% |
| documentary | 1 | 0.00% |
| game mechanic | 1 | 0.00% |
| player level | 1 | 0.00% |
| game character | 1 | 0.00% |
| game interface | 1 | 0.00% |
| observation | 1 | 0.00% |
| statement.event_details.dates | 1 | 0.00% |
| statement.time | 1 | 0.00% |
| statement.event_details.location | 1 | 0.00% |
| statement.recommendations | 1 | 0.00% |
| statement.eu_role | 1 | 0.00% |
| hardware_support.gpu.amd.api | 1 | 0.00% |
| hardware_support.gpu.amd.version | 1 | 0.00% |
| hardware_support.gpu.nvidia.api | 1 | 0.00% |
| hardware_support.gpu.nvidia.version | 1 | 0.00% |
| Recreational amenity | 1 | 0.00% |
| college life | 1 | 0.00% |
| competition participants | 1 | 0.00% |
| variant | 1 | 0.00% |
| Meal Type | 1 | 0.00% |
| Group/Class | 1 | 0.00% |
| role: Title of the job position being advertised | 1 | 0.00% |
| location: City and country where the role is based | 1 | 0.00% |
| job_id: Unique identifier assigned to the job posting | 1 | 0.00% |
| Search algorithm | 1 | 0.00% |
| precautionary measures | 1 | 0.00% |
| advancement/networking | 1 | 0.00% |
| continuation | 1 | 0.00% |
| Temperature Control | 1 | 0.00% |
| Application Method | 1 | 0.00% |
| Design/Decoration | 1 | 0.00% |
| Mode of Transportation | 1 | 0.00% |
| effort | 1 | 0.00% |
| software developers | 1 | 0.00% |
| unrealistic expectations | 1 | 0.00% |
| work process | 1 | 0.00% |
| Mythical creatures | 1 | 0.00% |
| Software Issue | 1 | 0.00% |
| publication type | 1 | 0.00% |
| unhealthy food | 1 | 0.00% |
| unhealthy snack | 1 | 0.00% |
| Indian condiment | 1 | 0.00% |
| varied produce | 1 | 0.00% |
| Fictional creature | 1 | 0.00% |
| Personal relationship | 1 | 0.00% |
| Cognitive process | 1 | 0.00% |
| Family background | 1 | 0.00% |
| shipping | 1 | 0.00% |
| stock_status | 1 | 0.00% |
| payment model | 1 | 0.00% |
| suggestion/idea | 1 | 0.00% |
| insufficient data | 1 | 0.00% |
| antagonist | 1 | 0.00% |
| Ticket type | 1 | 0.00% |
| behavior/regular practice | 1 | 0.00% |
| sensation/emotion | 1 | 0.00% |
| factoryName | 1 | 0.00% |
| index | 1 | 0.00% |
| group of countries | 1 | 0.00% |
| Cooking Method | 1 | 0.00% |
| work of literature | 1 | 0.00% |
| Insect | 1 | 0.00% |
| Language translation | 1 | 0.00% |
| professional organization | 1 | 0.00% |
| Pokemon creatures | 1 | 0.00% |
| task/action | 1 | 0.00% |
| outcome/result | 1 | 0.00% |
| warning/proverb | 1 | 0.00% |
| Building unit | 1 | 0.00% |
| Physical Process | 1 | 0.00% |
| Educational qualification | 1 | 0.00% |
| visual aspect | 1 | 0.00% |
| person.birth_place.city | 1 | 0.00% |
| person.profession | 1 | 0.00% |
| person.religion | 1 | 0.00% |
| person.us_senator.start_year | 1 | 0.00% |
| person.birth_date | 1 | 0.00% |
| Education Counsellor | 1 | 0.00% |
| Sports coach | 1 | 0.00% |
| Combat sports technique | 1 | 0.00% |
| chemical content | 1 | 0.00% |
| multinational pharmaceutical company | 1 | 0.00% |
| Funding organization | 1 | 0.00% |
| negative tactic | 1 | 0.00% |
| EUR | 1 | 0.00% |
| GBP | 1 | 0.00% |
| JPY | 1 | 0.00% |
| biochemical entity | 1 | 0.00% |
| protein domain | 1 | 0.00% |
| protein conformation | 1 | 0.00% |
| Commanders of the Army | 1 | 0.00% |
| Technical Skills | 1 | 0.00% |
| Adventure | 1 | 0.00% |
| Education/Assessment | 1 | 0.00% |
| Customer Experience | 1 | 0.00% |
| Knowledge/Insight | 1 | 0.00% |
| Marketing Term | 1 | 0.00% |
| actress | 1 | 0.00% |
| living beings | 1 | 0.00% |
| inherent | 1 | 0.00% |
| Metric Selection | 1 | 0.00% |
| Performance Index | 1 | 0.00% |
| Scoring Methodology | 1 | 0.00% |
| SUP Model | 1 | 0.00% |
| Watersports Equipment | 1 | 0.00% |
| Watersports activity | 1 | 0.00% |
| Surfing condition | 1 | 0.00% |
| plant pigment | 1 | 0.00% |
| Educational district | 1 | 0.00% |
| Residential Property Feature | 1 | 0.00% |
| Quality/Attribute | 1 | 0.00% |
| Style/Design | 1 | 0.00% |
| person/position | 1 | 0.00% |
| Data Format | 1 | 0.00% |
| photography style | 1 | 0.00% |
| Fence | 1 | 0.00% |
| Feed | 1 | 0.00% |
| Restaurant/Café | 1 | 0.00% |
| Practice/Routine | 1 | 0.00% |
| Spiritual Text | 1 | 0.00% |
| MLS Listing | 1 | 0.00% |
| Bedrooms | 1 | 0.00% |
| kitchenware item | 1 | 0.00% |
| product/set | 1 | 0.00% |
| County/Region | 1 | 0.00% |
| software attribute | 1 | 0.00% |
| Digital Library Account | 1 | 0.00% |
| Event/Anniversary | 1 | 0.00% |
| Music arrangement | 1 | 0.00% |
| Person/Group of people | 1 | 0.00% |
| media company | 1 | 0.00% |
| Ethnicity/Religion | 1 | 0.00% |
| Family heritage | 1 | 0.00% |
| UN General Assembly | 1 | 0.00% |
| International Day of Disabled Persons | 1 | 0.00% |
| person with disability | 1 | 0.00% |
| Fashion trend | 1 | 0.00% |
| Fashion event location | 1 | 0.00% |
| Taste/Characteristic | 1 | 0.00% |
| Film Director | 1 | 0.00% |
| Person/Influencer | 1 | 0.00% |
| Analysis | 1 | 0.00% |
| ecommerce entity | 1 | 0.00% |
| contact detail entity | 1 | 0.00% |
| Airport Code | 1 | 0.00% |
| Keyword/Phrase | 1 | 0.00% |
| cheese type | 1 | 0.00% |
| dietary habit | 1 | 0.00% |
| pattern/design | 1 | 0.00% |
| collar style | 1 | 0.00% |
| pocket | 1 | 0.00% |
| back design | 1 | 0.00% |
| design detail | 1 | 0.00% |
| Job type | 1 | 0.00% |
| Certificate Authority | 1 | 0.00% |
| Certification Authorities | 1 | 0.00% |
| Self-proclaimed Title | 1 | 0.00% |
| Personal Trait | 1 | 0.00% |
| releaseDocumentId | 1 | 0.00% |
| batchSubNumber | 1 | 0.00% |
| manufacturerId | 1 | 0.00% |
| manufacturerName | 1 | 0.00% |
| distributorId | 1 | 0.00% |
| distributorName | 1 | 0.00% |
| storageCondition | 1 | 0.00% |
| testResult | 1 | 0.00% |
| authorizedPerson | 1 | 0.00% |
| Room Size | 1 | 0.00% |
| Alcohol/Ingredient | 1 | 0.00% |
| process/event | 1 | 0.00% |
| website/organization | 1 | 0.00% |
| refundStatus | 1 | 0.00% |
| refundComments | 1 | 0.00% |
| refundDate | 1 | 0.00% |
| Type of Insurance | 1 | 0.00% |
| Redemption Instructions | 1 | 0.00% |
| Redemption Method | 1 | 0.00% |
| soil property | 1 | 0.00% |
| civil engineering structures | 1 | 0.00% |
| weight loss supplement | 1 | 0.00% |
| Window Type | 1 | 0.00% |
| golf club | 1 | 0.00% |
| metaphorical object | 1 | 0.00% |
| social media content | 1 | 0.00% |
| marketing content | 1 | 0.00% |
| network location | 1 | 0.00% |
| social gatherings | 1 | 0.00% |
| utensil/tool | 1 | 0.00% |
| establishment/business | 1 | 0.00% |
| person or entity | 1 | 0.00% |
| Country/Geopolitical Entity | 1 | 0.00% |
| Historical Event/War | 1 | 0.00% |
| Software Integration | 1 | 0.00% |
| Email Marketing Platform | 1 | 0.00% |
| visual aesthetic | 1 | 0.00% |
| document_details.document_type | 1 | 0.00% |
| document_details.authors.0.name | 1 | 0.00% |
| document_details.authors.1.name | 1 | 0.00% |
| document_details.authors.2.name | 1 | 0.00% |
| document_details.oclc_number | 1 | 0.00% |
| document_details.description | 1 | 0.00% |
| document_details.series_title | 1 | 0.00% |
| document_details.responsibility | 1 | 0.00% |
| age requirement | 1 | 0.00% |
| trade | 1 | 0.00% |
| security information | 1 | 0.00% |
| Living Organisms | 1 | 0.00% |
| Biological Component | 1 | 0.00% |
| funding stage | 1 | 0.00% |
| investment type | 1 | 0.00% |
| increase type | 1 | 0.00% |
| Militant Group | 1 | 0.00% |
| Natural phenomena | 1 | 0.00% |
| music label | 1 | 0.00% |
| inventory management | 1 | 0.00% |
| department: Department where the employee is assigned | 1 | 0.00% |
| projects: Type of projects the employee is involved in | 1 | 0.00% |
| genus.established_year | 1 | 0.00% |
| genus.established_by | 1 | 0.00% |
| genus.publication | 1 | 0.00% |
| genus.number_of_species.min | 1 | 0.00% |
| genus.number_of_species.max | 1 | 0.00% |
| genus.distribution | 1 | 0.00% |
| genus.pollination.mechanism | 1 | 0.00% |
| genus.pollination.pollinators | 1 | 0.00% |
| genus.etymology.part1.name | 1 | 0.00% |
| genus.etymology.part1.meaning | 1 | 0.00% |
| genus.etymology.part1.origin | 1 | 0.00% |
| genus.etymology.part1.related_entity | 1 | 0.00% |
| genus.etymology.part2.name | 1 | 0.00% |
| genus.etymology.part2.meaning | 1 | 0.00% |
| genus.characteristics | 1 | 0.00% |
| genus.european_species | 1 | 0.00% |
| genus.german_species | 1 | 0.00% |
| genus.species_list | 1 | 0.00% |
| Hardware Product | 1 | 0.00% |
| processor | 1 | 0.00% |
| PlanType | 1 | 0.00% |
| ResiliencyLevel | 1 | 0.00% |
| ParentName | 1 | 0.00% |
| Character limit | 1 | 0.00% |
| Social media post | 1 | 0.00% |
| ceo_name | 1 | 0.00% |
| growth_rate | 1 | 0.00% |
| Object/Component | 1 | 0.00% |
| meteorological phenomenon | 1 | 0.00% |
| wedding attire | 1 | 0.00% |
| wedding rehearsal | 1 | 0.00% |
| bridal shower | 1 | 0.00% |
| Academic Schedule | 1 | 0.00% |
| Registered individuals | 1 | 0.00% |
| Cultural identity | 1 | 0.00% |
| Ethnic groups | 1 | 0.00% |
| character type | 1 | 0.00% |
| golf equipment | 1 | 0.00% |
| rejuvenation items | 1 | 0.00% |
| expression/reaction | 1 | 0.00% |
| brood cell | 1 | 0.00% |
| brood production | 1 | 0.00% |
| Social Life | 1 | 0.00% |
| Shared Living Quarters | 1 | 0.00% |
| Common Areas | 1 | 0.00% |
| Leisure Activities | 1 | 0.00% |
| Learning Activities | 1 | 0.00% |
| Collective Duties | 1 | 0.00% |
| time period/season | 1 | 0.00% |
| Fashion collection | 1 | 0.00% |
| Textile design element | 1 | 0.00% |
| Building renovation | 1 | 0.00% |
| Aesthetic features | 1 | 0.00% |
| Building personality | 1 | 0.00% |
| Aesthetic diversity | 1 | 0.00% |
| Design Work | 1 | 0.00% |
| Method of communication | 1 | 0.00% |
| accounting standard | 1 | 0.00% |
| pension plans | 1 | 0.00% |
| valuation methods | 1 | 0.00% |
| Negative feedback | 1 | 0.00% |
| Neutral feedback | 1 | 0.00% |
| property of furnishing | 1 | 0.00% |
| asteroid.name_number | 1 | 0.00% |
| asteroid.name_origin.relation | 1 | 0.00% |
| asteroid.name_original | 1 | 0.00% |
| asteroid.name_origin.description | 1 | 0.00% |
| asteroid.name_origin.name | 1 | 0.00% |
| asteroid.discovery.location.country | 1 | 0.00% |
| Video Card | 1 | 0.00% |
| Video Card Series | 1 | 0.00% |
| dining reservation | 1 | 0.00% |
| Sport Equipment | 1 | 0.00% |
| Fire Center | 1 | 0.00% |
| Schools | 1 | 0.00% |
| Universities | 1 | 0.00% |
| Sports leagues | 1 | 0.00% |
| Brand/Event | 1 | 0.00% |
| object location | 1 | 0.00% |
| Trade Association | 1 | 0.00% |
| field of study/interest | 1 | 0.00% |
| origin/place of origin | 1 | 0.00% |
| Lighting ceremony | 1 | 0.00% |
| Historical building | 1 | 0.00% |
| Festive activities | 1 | 0.00% |
| Singers | 1 | 0.00% |
| Food and drink | 1 | 0.00% |
| Art and craft | 1 | 0.00% |
| Mascots | 1 | 0.00% |
| Historical characters | 1 | 0.00% |
| transportation service provider | 1 | 0.00% |
| hb: Hemoglobin level in grams per deciliter | 1 | 0.00% |
| patient_id: Unique patient identifier from lab system | 1 | 0.00% |
| patient_name: Name of the patient undergoing the test | 1 | 0.00% |
| test_date: Date the lab test was conducted | 1 | 0.00% |
| rbc: Red blood cell count in millions per microliter | 1 | 0.00% |
| Refuge | 1 | 0.00% |
| Logistics | 1 | 0.00% |
| protein receptor | 1 | 0.00% |
| biological system | 1 | 0.00% |
| physiological function | 1 | 0.00% |
| brain structure | 1 | 0.00% |
| Interest/Hobby | 1 | 0.00% |
| accessibility | 1 | 0.00% |
| inaccessibility/scientific research | 1 | 0.00% |
| guitar measurement | 1 | 0.00% |
| guitar design | 1 | 0.00% |
| report | 1 | 0.00% |
| physical measurement unit | 1 | 0.00% |
| seat | 1 | 0.00% |
| guestName | 1 | 0.00% |
| sales | 1 | 0.00% |
| Past action | 1 | 0.00% |
| job loss | 1 | 0.00% |
| Military Entity | 1 | 0.00% |
| Terrorist Organization | 1 | 0.00% |
| Narrator/Author | 1 | 0.00% |
| Direction/Choice | 1 | 0.00% |
| Existence | 1 | 0.00% |
| Difficulties | 1 | 0.00% |
| brand/model | 1 | 0.00% |
| motorcycle style | 1 | 0.00% |
| manufacturing technology | 1 | 0.00% |
| Body Measurement | 1 | 0.00% |
| Clothing Measurement | 1 | 0.00% |
| security policy | 1 | 0.00% |
| user device | 1 | 0.00% |
| file sharing protocol | 1 | 0.00% |
| data logs | 1 | 0.00% |
| analytics service | 1 | 0.00% |
| Reading Material | 1 | 0.00% |
| Book Length | 1 | 0.00% |
| Reading Sessions | 1 | 0.00% |
| data entry | 1 | 0.00% |
| possibility of harm or loss | 1 | 0.00% |
| Car part/Accessory | 1 | 0.00% |
| digital storage | 1 | 0.00% |
| digital files | 1 | 0.00% |
| football coach | 1 | 0.00% |
| football league | 1 | 0.00% |
| boredom | 1 | 0.00% |
| complaints | 1 | 0.00% |
| occurrences | 1 | 0.00% |
| unaltered | 1 | 0.00% |
| sauce | 1 | 0.00% |
| place of residence | 1 | 0.00% |
| Product or Service | 1 | 0.00% |
| Business Segment | 1 | 0.00% |
| school | 1 | 0.00% |
| Swimming Lesson Program | 1 | 0.00% |
| Aquatic Center | 1 | 0.00% |
| Health Club | 1 | 0.00% |
| Environmental feature | 1 | 0.00% |
| Visual experience | 1 | 0.00% |
| viewpoint | 1 | 0.00% |
| named_after.name | 1 | 0.00% |
| discovery.location.city | 1 | 0.00% |
| Punctuation mark | 1 | 0.00% |
| artificial flower | 1 | 0.00% |
| innovative technology | 1 | 0.00% |
| natural process | 1 | 0.00% |
| lifestyle habits | 1 | 0.00% |
| skin damage | 1 | 0.00% |
| Service/Function | 1 | 0.00% |
| Website/Domain | 1 | 0.00% |
| Request type | 1 | 0.00% |
| device component | 1 | 0.00% |
| Operational Hours | 1 | 0.00% |
| educational field | 1 | 0.00% |
| professional qualification | 1 | 0.00% |
| Reservation | 1 | 0.00% |
| Nature | 1 | 0.00% |
| People/Relationship | 1 | 0.00% |
| Departure event | 1 | 0.00% |
| Military Branch | 1 | 0.00% |
| female reproductive health | 1 | 0.00% |
| Jewish law | 1 | 0.00% |
| online purchase | 1 | 0.00% |
| Government office | 1 | 0.00% |
| merchandise/product | 1 | 0.00% |
| Safety Measure | 1 | 0.00% |
| person2 | 1 | 0.00% |
| Dessert type | 1 | 0.00% |
| Cake design style | 1 | 0.00% |
| Recipe Ideas | 1 | 0.00% |
| Movie trailer | 1 | 0.00% |
| personal responsibilities | 1 | 0.00% |
| surgical procedure | 1 | 0.00% |
| Academic entity | 1 | 0.00% |
| Musician entity | 1 | 0.00% |
| Security update | 1 | 0.00% |
| Religious organization | 1 | 0.00% |
| Religious expression | 1 | 0.00% |
| Publication details | 1 | 0.00% |
| garbage truck fleet | 1 | 0.00% |
| Job Losses | 1 | 0.00% |
| travel pass | 1 | 0.00% |
| furniture maintenance | 1 | 0.00% |
| cleaning | 1 | 0.00% |
| id: Unique identifier for the line item entry | 1 | 0.00% |
| status: Current approval status of the transaction | 1 | 0.00% |
| legal document template | 1 | 0.00% |
| course module | 1 | 0.00% |
| course schedule | 1 | 0.00% |
| reputation | 1 | 0.00% |
| emergency | 1 | 0.00% |
| conversation | 1 | 0.00% |
| emotion/mental state | 1 | 0.00% |
| Exit | 1 | 0.00% |
| Signage | 1 | 0.00% |
| Storage Unit | 1 | 0.00% |
| Number of Victims | 1 | 0.00% |
| Rating/Review | 1 | 0.00% |
| event venue | 1 | 0.00% |
| architectural space | 1 | 0.00% |
| cultural institution | 1 | 0.00% |
| Medicine | 1 | 0.00% |
| plant requirement | 1 | 0.00% |
| Industry Growth | 1 | 0.00% |
| Future Goals/Plans | 1 | 0.00% |
| Extraterrestrial beings | 1 | 0.00% |
| food waste | 1 | 0.00% |
| waste disposal | 1 | 0.00% |
| organic waste | 1 | 0.00% |
| Distance Entity | 1 | 0.00% |
| Emotion Entity | 1 | 0.00% |
| Goal Entity | 1 | 0.00% |
| Person Entity | 1 | 0.00% |
| Life Entity | 1 | 0.00% |
| floor | 1 | 0.00% |
| warrantyExpirationDate | 1 | 0.00% |
| Development methodology | 1 | 0.00% |
| Property Attribute | 1 | 0.00% |
| Sale Condition | 1 | 0.00% |
| offerId | 1 | 0.00% |
| couponCode | 1 | 0.00% |
| System/Industry | 1 | 0.00% |
| Landmarks/Geographical Features | 1 | 0.00% |
| Property/Real Estate | 1 | 0.00% |
| patients | 1 | 0.00% |
| individuals in need of medical attention who may be hesitant or unwilling to receive it | 1 | 0.00% |
| individuals who are afraid of medical procedures or treatments | 1 | 0.00% |
| language style | 1 | 0.00% |
| medical treatments | 1 | 0.00% |
| choices | 1 | 0.00% |
| pain management technique | 1 | 0.00% |
| Financial borrowing | 1 | 0.00% |
| Financial standing | 1 | 0.00% |
| Financial burden | 1 | 0.00% |
| Mismanagement | 1 | 0.00% |
| Lack of financial literacy | 1 | 0.00% |
| Precision | 1 | 0.00% |
| Historical civilization | 1 | 0.00% |
| Culinary item | 1 | 0.00% |
| communication channels | 1 | 0.00% |
| written materials | 1 | 0.00% |
| Criminal Organization | 1 | 0.00% |
| Criminal Figure | 1 | 0.00% |
| Sensitive Data | 1 | 0.00% |
| Ethnicity/Physical Appearance | 1 | 0.00% |
| Design/Methodology | 1 | 0.00% |
| Book Design Element | 1 | 0.00% |
| Furniture Set | 1 | 0.00% |
| Furniture Accessory | 1 | 0.00% |
| Marketing tactic | 1 | 0.00% |
| Search term | 1 | 0.00% |
| Online tools | 1 | 0.00% |
| Sale Date | 1 | 0.00% |
| Medical emergency | 1 | 0.00% |
| Religion/ Belief System | 1 | 0.00% |
| mental stress | 1 | 0.00% |
| nationality/ethnicity | 1 | 0.00% |
| healthcare system | 1 | 0.00% |
| date_posted | 1 | 0.00% |
| Retirement plan | 1 | 0.00% |
| skill/hobby | 1 | 0.00% |
| Fuse Rating | 1 | 0.00% |
| Circuit Breaker | 1 | 0.00% |
| Power Integrations IC | 1 | 0.00% |
| Diode | 1 | 0.00% |
| SMD Fuse | 1 | 0.00% |
| Transistor | 1 | 0.00% |
| Boost Converter | 1 | 0.00% |
| Temperature Coefficient | 1 | 0.00% |
| Capacitor | 1 | 0.00% |
| Aquatic feature | 1 | 0.00% |
| Aquarium filtration system | 1 | 0.00% |
| Aesthetic quality | 1 | 0.00% |
| dishware | 1 | 0.00% |
| person/character | 1 | 0.00% |
| Technology/Data storage | 1 | 0.00% |
| Technology/Logging | 1 | 0.00% |
| Technology/Settings | 1 | 0.00% |
| Technology/Hardware monitoring | 1 | 0.00% |
| Technology/Networking | 1 | 0.00% |
| performance event | 1 | 0.00% |
| choreography | 1 | 0.00% |
| event charge | 1 | 0.00% |
| Cricket Term | 1 | 0.00% |
| type of paper | 1 | 0.00% |
| Gambling Licence | 1 | 0.00% |
| Gaming Environment | 1 | 0.00% |
| Cash Prize | 1 | 0.00% |
| Bingo Review | 1 | 0.00% |
| Bonus | 1 | 0.00% |
| Bingo Site | 1 | 0.00% |
| visual property | 1 | 0.00% |
| policy_holder | 1 | 0.00% |
| Environmental Noise | 1 | 0.00% |
| Unpleasant odor | 1 | 0.00% |
| Government entities | 1 | 0.00% |
| Government position | 1 | 0.00% |
| Whiskey | 1 | 0.00% |
| Social Cause | 1 | 0.00% |
| Financial Knowledge | 1 | 0.00% |
| Physical State | 1 | 0.00% |
| government function | 1 | 0.00% |
| personal trait/action | 1 | 0.00% |
| Movie Franchise | 1 | 0.00% |
| psychological evaluation | 1 | 0.00% |
| irrational belief | 1 | 0.00% |
| content delivery mechanism | 1 | 0.00% |
| Shipping Service | 1 | 0.00% |
| population size | 1 | 0.00% |
| population growth | 1 | 0.00% |
| scientific advancement | 1 | 0.00% |
| medical subjects | 1 | 0.00% |
| health management | 1 | 0.00% |
| genetic variant | 1 | 0.00% |
| Exception Message | 1 | 0.00% |
| Exception Handler Element | 1 | 0.00% |
| Manga Series | 1 | 0.00% |
| Biological system | 1 | 0.00% |
| healthcare facilities | 1 | 0.00% |
| Video game developer | 1 | 0.00% |
| Video game publisher | 1 | 0.00% |
| occasions | 1 | 0.00% |
| exhibition | 1 | 0.00% |
| Cyber Threat | 1 | 0.00% |
| business environment | 1 | 0.00% |
| Cognitive function | 1 | 0.00% |
| printable item | 1 | 0.00% |
| Decorator | 1 | 0.00% |
| Orca Grow Film | 1 | 0.00% |
| Orca Reflective Film | 1 | 0.00% |
| Hydroponics | 1 | 0.00% |
| Plants | 1 | 0.00% |
| Water | 1 | 0.00% |
| Regulatory Agency | 1 | 0.00% |
| Environmental Disaster | 1 | 0.00% |
| annotation_id | 1 | 0.00% |
| contract_id | 1 | 0.00% |
| party1 | 1 | 0.00% |
| party2 | 1 | 0.00% |
| address1 | 1 | 0.00% |
| dispute_duration | 1 | 0.00% |
| hair styling tool | 1 | 0.00% |
| temperature level | 1 | 0.00% |
| buying | 1 | 0.00% |
| cautiously | 1 | 0.00% |
| sentiment.opinion | 1 | 0.00% |
| quote.text | 1 | 0.00% |
| Networking Equipment | 1 | 0.00% |
| Personal Project Name | 1 | 0.00% |
| Resting Periods | 1 | 0.00% |
| man-made feature | 1 | 0.00% |
| loading time | 1 | 0.00% |
| Wedding Officiant | 1 | 0.00% |
| decor element | 1 | 0.00% |
| status: Current status of the enrollment | 1 | 0.00% |
| duration: Number of days the program lasts | 1 | 0.00% |
| employee_id: Unique identifier for the employee enrollment | 1 | 0.00% |
| negative force | 1 | 0.00% |
| financial well-being | 1 | 0.00% |
| professional well-being | 1 | 0.00% |
| size/fitting | 1 | 0.00% |
| politics | 1 | 0.00% |
| review_text: Full textual critique of the film | 1 | 0.00% |
| rating: Numerical score given to the film | 1 | 0.00% |
| film_title: Title of the film being reviewed | 1 | 0.00% |
| Social Groups | 1 | 0.00% |
| Social Organizations | 1 | 0.00% |
| tire part | 1 | 0.00% |
| mystical substance | 1 | 0.00% |
| religious group | 1 | 0.00% |
| religious object | 1 | 0.00% |
| Insurance | 1 | 0.00% |
| software release | 1 | 0.00% |
| language version | 1 | 0.00% |
| punishment/consequence | 1 | 0.00% |
| device_name: Name of the medical device as listed | 1 | 0.00% |
| model: Device model name and version | 1 | 0.00% |
| serial_number: Unique device identifier assigned by manufacturer | 1 | 0.00% |
| manufacturer: Company that produced the device | 1 | 0.00% |
| location: Physical ward or room where device is located | 1 | 0.00% |
| purchase_date: Date the device was acquired | 1 | 0.00% |
| cost: Total cost of the device in dollars | 1 | 0.00% |
| status: Current operational status of the device | 1 | 0.00% |
| expiration_date: Date after which device is no longer valid | 1 | 0.00% |
| assigned_staff: Healthcare professional assigned to manage the device | 1 | 0.00% |
| career_summary.location | 1 | 0.00% |
| future_plans | 1 | 0.00% |
| copyright | 1 | 0.00% |
| legacy | 1 | 0.00% |
| Mechanical component | 1 | 0.00% |
| Car rental company | 1 | 0.00% |
| Service guarantee | 1 | 0.00% |
| Establishment year | 1 | 0.00% |
| Service department | 1 | 0.00% |
| Auction House | 1 | 0.00% |
| Auction item | 1 | 0.00% |
| Coin grading | 1 | 0.00% |
| art exhibition | 1 | 0.00% |
| size category | 1 | 0.00% |
| enumeration type | 1 | 0.00% |
| enumerator | 1 | 0.00% |
| Installation Service | 1 | 0.00% |
| Standard organization | 1 | 0.00% |
| Quality rating | 1 | 0.00% |
| Strength rating | 1 | 0.00% |
| Lock mechanism | 1 | 0.00% |
| Certification mark | 1 | 0.00% |
| Lock type | 1 | 0.00% |
| Social media influencer | 1 | 0.00% |
| extra_options_prompt | 1 | 0.00% |
| passenger_information.infants.prompt | 1 | 0.00% |
| passenger_information.infants.age_range | 1 | 0.00% |
| passenger_information.adults.age_range | 1 | 0.00% |
| flight_search_instructions | 1 | 0.00% |
| tel_aviv_attractions | 1 | 0.00% |
| israel_airports.largest_airport.airport_name | 1 | 0.00% |
| return_date_prompt | 1 | 0.00% |
| departure_airport_prompt | 1 | 0.00% |
| israel_airports.largest_airport.city | 1 | 0.00% |
| israel_airports.total_count | 1 | 0.00% |
| israel_airports.largest_airport.departure_city | 1 | 0.00% |
| passenger_information.children.age_range | 1 | 0.00% |
| Organization Member | 1 | 0.00% |
| personal practice | 1 | 0.00% |
| students | 1 | 0.00% |
| subjects | 1 | 0.00% |
| law firm | 1 | 0.00% |
| Navigation point | 1 | 0.00% |
| Digital storage | 1 | 0.00% |
| Navigation page | 1 | 0.00% |
| Library service/facility | 1 | 0.00% |
| Coupon/Deal | 1 | 0.00% |
| E-commerce | 1 | 0.00% |
| file path | 1 | 0.00% |
| logo | 1 | 0.00% |
| game mode | 1 | 0.00% |
| game difficulty | 1 | 0.00% |
| game participant | 1 | 0.00% |
| product development | 1 | 0.00% |
| prioritization | 1 | 0.00% |
| deployment | 1 | 0.00% |
| Entity Type/Postal Code | 1 | 0.00% |
| assessment type | 1 | 0.00% |
| Computer Performance | 1 | 0.00% |
| golf accessories | 1 | 0.00% |
| golf accessory | 1 | 0.00% |
| Word Puzzle | 1 | 0.00% |
| Scratch Card | 1 | 0.00% |
| data report | 1 | 0.00% |
| cleaning products | 1 | 0.00% |
| disinfectants | 1 | 0.00% |
| plot device | 1 | 0.00% |
| issueNumber | 1 | 0.00% |
| coverPrice | 1 | 0.00% |
| Incentives | 1 | 0.00% |
| Nature/Animals | 1 | 0.00% |
| Aesthetics/Beauty | 1 | 0.00% |
| Milestones/Experiences | 1 | 0.00% |
| weight array | 1 | 0.00% |
| Activity level | 1 | 0.00% |
| Health issue | 1 | 0.00% |
| TV show title | 1 | 0.00% |
| competitor | 1 | 0.00% |
| Political alliance | 1 | 0.00% |
| Concert Venue | 1 | 0.00% |
| Political subdivision | 1 | 0.00% |
| Software file | 1 | 0.00% |
| Business/venue | 1 | 0.00% |
| recognition event | 1 | 0.00% |
| Sales Entity | 1 | 0.00% |
| asteroid.named_after.person.name | 1 | 0.00% |
| asteroid.named_after.person.birth_year | 1 | 0.00% |
| asteroid.asteroid_type | 1 | 0.00% |
| course_id: Unique identifier for the course | 1 | 0.00% |
| title: Name of the course such as Introduction to English Composition | 1 | 0.00% |
| location: Physical address where course is held | 1 | 0.00% |
| duration: Number of weeks the course runs | 1 | 0.00% |
| cost: Total cost of enrollment in US dollars | 1 | 0.00% |
| status: Current enrollment or operational status | 1 | 0.00% |
| instructor: Name of the course instructor | 1 | 0.00% |
| credits: Number of academic credits assigned to the course | 1 | 0.00% |
| Product characteristic | 1 | 0.00% |
| legislative_assembly.structure.political_associations.1.seats | 1 | 0.00% |
| legislative_assembly.leadership.speaker | 1 | 0.00% |
| legislative_assembly.establishment_year | 1 | 0.00% |
| legislative_assembly.structure.political_associations.3.party | 1 | 0.00% |
| legislative_assembly.name | 1 | 0.00% |
| legislative_assembly.type | 1 | 0.00% |
| legislative_assembly.structure.number_of_members | 1 | 0.00% |
| legislative_assembly.state | 1 | 0.00% |
| legislative_assembly.electoral_system | 1 | 0.00% |
| legislative_assembly.last_election_date | 1 | 0.00% |
| legislative_assembly.references.0.access_date | 1 | 0.00% |
| legislative_assembly.structure.political_associations.3.seats | 1 | 0.00% |
| legislative_assembly.structure.political_associations.2.seats | 1 | 0.00% |
| legislative_assembly.current_government | 1 | 0.00% |
| legislative_assembly.structure.political_associations.1.party | 1 | 0.00% |
| legislative_assembly.structure.political_associations.2.party | 1 | 0.00% |
| Property detail | 1 | 0.00% |
| Computing Infrastructure | 1 | 0.00% |
| Container Technology | 1 | 0.00% |
| Security Configuration | 1 | 0.00% |
| Software Feedback | 1 | 0.00% |
| Network Action | 1 | 0.00% |
| file system object | 1 | 0.00% |
| average_score | 1 | 0.00% |
| school_id | 1 | 0.00% |
| Visitor Count | 1 | 0.00% |
| Artist/Photographer | 1 | 0.00% |
| protein type | 1 | 0.00% |
| Holiday entity | 1 | 0.00% |
| Account Information | 1 | 0.00% |
| Call to Action | 1 | 0.00% |
| Database Management | 1 | 0.00% |
| blockage | 1 | 0.00% |
| Activity/Location | 1 | 0.00% |
| Lottery | 1 | 0.00% |
| Advertising agency | 1 | 0.00% |
| Production company | 1 | 0.00% |
| Post-production company | 1 | 0.00% |
| insects | 1 | 0.00% |
| Marketing Event | 1 | 0.00% |
| Brand Name/Product Name | 1 | 0.00% |
| Tire | 1 | 0.00% |
| post_id | 1 | 0.00% |
| data measurement | 1 | 0.00% |
| measurement interpretation | 1 | 0.00% |
| Person/Official | 1 | 0.00% |
| Expression/Action | 1 | 0.00% |
| reporter_company: Company name of the customer reporter | 1 | 0.00% |
| ticket_id: Unique identifier for each support ticket | 1 | 0.00% |
| assigned_agent: Name of the agent handling the ticket | 1 | 0.00% |
| reporter_name: Name of the customer who reported the issue | 1 | 0.00% |
| transportation schedule | 1 | 0.00% |
| Dating Site | 1 | 0.00% |
| Event/Deal | 1 | 0.00% |
| nostalgia | 1 | 0.00% |
| appraisalId | 1 | 0.00% |
| appraisalYear | 1 | 0.00% |
| departmentCode | 1 | 0.00% |
| departmentName | 1 | 0.00% |
| appraisalDate | 1 | 0.00% |
| Processor speed | 1 | 0.00% |
| Processor configuration | 1 | 0.00% |
| Wedding Party Members | 1 | 0.00% |
| Audience/People | 1 | 0.00% |
| Aspirations | 1 | 0.00% |
| Group/Category | 1 | 0.00% |
| date-time | 1 | 0.00% |
| Group/Role | 1 | 0.00% |
| service provided | 1 | 0.00% |
| commonName | 1 | 0.00% |
| subgenus | 1 | 0.00% |
| transportation equipment | 1 | 0.00% |
| Financial Losses | 1 | 0.00% |
| Market Movements | 1 | 0.00% |
| Financial Gain | 1 | 0.00% |
| Numeric Entity | 1 | 0.00% |
| Abstract Entity | 1 | 0.00% |
| Historical landmark | 1 | 0.00% |
| Educational activity | 1 | 0.00% |
| occupation/service | 1 | 0.00% |
| advice/suggestion | 1 | 0.00% |
| Property management software | 1 | 0.00% |
| Search engine | 1 | 0.00% |
| healthcare standards | 1 | 0.00% |
| Music track | 1 | 0.00% |
| Video game event | 1 | 0.00% |
| application_name | 1 | 0.00% |
| registration.benefits | 1 | 0.00% |
| music.organization | 1 | 0.00% |
| Lebensstil | 1 | 0.00% |
| customer inquiry | 1 | 0.00% |
| business offerings | 1 | 0.00% |
| deity/religious figure | 1 | 0.00% |
| treatment program | 1 | 0.00% |
| Office Supplies | 1 | 0.00% |
| official documents | 1 | 0.00% |
| building/venue | 1 | 0.00% |
| material/feature | 1 | 0.00% |
| wire gauge | 1 | 0.00% |
| wire connectors | 1 | 0.00% |
| wire connector | 1 | 0.00% |
| Flowers | 1 | 0.00% |
| Garden Types | 1 | 0.00% |
| Orchard | 1 | 0.00% |
| Cabin | 1 | 0.00% |
| transfer_fee | 1 | 0.00% |
| people entity | 1 | 0.00% |
| event-attack/battle/war/militaryconflict | 1 | 0.00% |
| Graphics Card Model | 1 | 0.00% |
| release_date | 1 | 0.00% |
| Physical feature | 1 | 0.00% |
| firearm | 1 | 0.00% |
| concealed carry | 1 | 0.00% |
| self-defense | 1 | 0.00% |
| Service availability | 1 | 0.00% |
| Manufacturing Company | 1 | 0.00% |
| Sports Team/Competition | 1 | 0.00% |
| Fictional Character/Protagonist | 1 | 0.00% |
| Fictional Object | 1 | 0.00% |
| leadership programme | 1 | 0.00% |
| Job requirement | 1 | 0.00% |
| Skill/Experience requirement | 1 | 0.00% |
| Organization/Event | 1 | 0.00% |
| categorization entity | 1 | 0.00% |
| military operation | 1 | 0.00% |
| Crafting Technique | 1 | 0.00% |
| education/literacy | 1 | 0.00% |
| task-oriented document | 1 | 0.00% |
| collection of items | 1 | 0.00% |
| controlled environment | 1 | 0.00% |
| test takers | 1 | 0.00% |
| online testing | 1 | 0.00% |
| corporation/group | 1 | 0.00% |
| data/record | 1 | 0.00% |
| medical examination | 1 | 0.00% |
| Team Name | 1 | 0.00% |
| Transaction Type | 1 | 0.00% |
| Business Stage | 1 | 0.00% |
| course | 1 | 0.00% |
| Wrestling event | 1 | 0.00% |
| Wrestling team | 1 | 0.00% |
| WWE championship title | 1 | 0.00% |
| Community hub | 1 | 0.00% |
| Game development company | 1 | 0.00% |
| Technical problems | 1 | 0.00% |
| recipe series | 1 | 0.00% |
| writing tool | 1 | 0.00% |
| Music term | 1 | 0.00% |
| people or group of individuals | 1 | 0.00% |
| scarcity | 1 | 0.00% |
| provisional_designation | 1 | 0.00% |
| asteroid_belt | 1 | 0.00% |
| discovery.observatory.location.region | 1 | 0.00% |
| orbital_characteristics.asteroid_family | 1 | 0.00% |
| river | 1 | 0.00% |
| hiking trail | 1 | 0.00% |
| small river/stream | 1 | 0.00% |
| Restaurant/Cuisine | 1 | 0.00% |
| Cuisine/Dish | 1 | 0.00% |
| Trading Type | 1 | 0.00% |
| Investment Type | 1 | 0.00% |
| Geological Location | 1 | 0.00% |
| evaluation feedback | 1 | 0.00% |
| performance venue | 1 | 0.00% |
| type of health insurance | 1 | 0.00% |
| Field of Work | 1 | 0.00% |
| Government administration | 1 | 0.00% |
| Kinship | 1 | 0.00% |
| Traditional Casinos | 1 | 0.00% |
| Modeled after | 1 | 0.00% |
| clientbasesize: Number of clients served across regions | 1 | 0.00% |
| location: City and state or country where the company is based | 1 | 0.00% |
| funding_source: Type of funding received by the company | 1 | 0.00% |
| business_focus: Core service or product area of the company | 1 | 0.00% |
| pseudonym | 1 | 0.00% |
| people/group | 1 | 0.00% |
| Communication technology | 1 | 0.00% |
| Mobile operating system | 1 | 0.00% |
| Luxury item | 1 | 0.00% |
| professional | 1 | 0.00% |
| process/project | 1 | 0.00% |
| contact detail | 1 | 0.00% |
| alphabetical index letter | 1 | 0.00% |
| registrant: Company that owns the registered trademark | 1 | 0.00% |
| opponent: Entity filing the dispute or opposition | 1 | 0.00% |
| trademark_name: Name of the registered trademark in dispute | 1 | 0.00% |
| Course Credits | 1 | 0.00% |
| Product offering | 1 | 0.00% |
| Service flexibility | 1 | 0.00% |
| Book/Devotional | 1 | 0.00% |
| Convention | 1 | 0.00% |
| watch customization | 1 | 0.00% |
| prescriptionId | 1 | 0.00% |
| refills | 1 | 0.00% |
| pharmacyId | 1 | 0.00% |
| datePrescribed | 1 | 0.00% |
| dateExpires | 1 | 0.00% |
| Patch organization | 1 | 0.00% |
| Patch parameter storage | 1 | 0.00% |
| Silent patch selection | 1 | 0.00% |
| Award/Badge | 1 | 0.00% |
| book/chapter/verse | 1 | 0.00% |
| professional network | 1 | 0.00% |
| microblogging platform | 1 | 0.00% |
| Furniture Pieces | 1 | 0.00% |
| productionQuantity | 1 | 0.00% |
| otherIndirectCosts | 1 | 0.00% |
| productionQuality | 1 | 0.00% |
| Military facility | 1 | 0.00% |
| Air base | 1 | 0.00% |
| Flight approach | 1 | 0.00% |
| Pipe Type | 1 | 0.00% |
| Date of Birth | 1 | 0.00% |
| Appliance Brand | 1 | 0.00% |
| beauty effect | 1 | 0.00% |
| beauty feature | 1 | 0.00% |
| Legal team | 1 | 0.00% |
| Legal outcome | 1 | 0.00% |
| ingredient list | 1 | 0.00% |
| professional development | 1 | 0.00% |
| ability or expertise | 1 | 0.00% |
| tactic | 1 | 0.00% |
| Physical Energy | 1 | 0.00% |
| Moral/Value System | 1 | 0.00% |
| Business Domain | 1 | 0.00% |
| Job title/position | 1 | 0.00% |
| Job seekers | 1 | 0.00% |
| Qualification criteria | 1 | 0.00% |
| research facility | 1 | 0.00% |
| medical study | 1 | 0.00% |
| Health & Wellness | 1 | 0.00% |
| Leisure & Entertainment | 1 | 0.00% |
| Daily Life | 1 | 0.00% |
| Legal Rights | 1 | 0.00% |
| Traffic Violation Points | 1 | 0.00% |
| film editing | 1 | 0.00% |
| Job sector | 1 | 0.00% |
| organization purpose | 1 | 0.00% |
| circular.target_audience | 1 | 0.00% |
| circular.ministry | 1 | 0.00% |
| destination_club | 1 | 0.00% |
| origin_club | 1 | 0.00% |
| appearances | 1 | 0.00% |
| playerbirthcity | 1 | 0.00% |
| Announcement | 1 | 0.00% |
| validation | 1 | 0.00% |
| acknowledgement | 1 | 0.00% |
| consequences | 1 | 0.00% |
| legal actor | 1 | 0.00% |
| involved | 1 | 0.00% |
| Product model | 1 | 0.00% |
| TV network | 1 | 0.00% |
| Social Network | 1 | 0.00% |
| material feature | 1 | 0.00% |
| personal preferences | 1 | 0.00% |
| thorough examination | 1 | 0.00% |
| Viewability measurement tool | 1 | 0.00% |
| Video sharing website | 1 | 0.00% |
| Ad format | 1 | 0.00% |
| Nursery/Plant Seller | 1 | 0.00% |
| Plant Packaging | 1 | 0.00% |
| Software component | 1 | 0.00% |
| Computer system | 1 | 0.00% |
| File directory | 1 | 0.00% |
| deviceType | 1 | 0.00% |
| deviceName | 1 | 0.00% |
| dataCollectionInterval | 1 | 0.00% |
| sensorId | 1 | 0.00% |
| reportingInterval | 1 | 0.00% |
| Entertainment Franchise | 1 | 0.00% |
| nursery decor | 1 | 0.00% |
| nursery room | 1 | 0.00% |
| bedding sets | 1 | 0.00% |
| bedding set | 1 | 0.00% |
| State of Matter | 1 | 0.00% |
| dental issue | 1 | 0.00% |
| visual effect | 1 | 0.00% |
| development environment | 1 | 0.00% |
| Tax Law | 1 | 0.00% |
| Power Output | 1 | 0.00% |
| Teeth Whitening product | 1 | 0.00% |
| button/link | 1 | 0.00% |
| product evaluation | 1 | 0.00% |
| manufacturing cost | 1 | 0.00% |
| knowledge/skill | 1 | 0.00% |
| precaution | 1 | 0.00% |
| court of last resort | 1 | 0.00% |
| violation of constitutional rights | 1 | 0.00% |
| legal proof | 1 | 0.00% |
| funding event | 1 | 0.00% |
| Dental Service | 1 | 0.00% |
| Teeth Whitening Product | 1 | 0.00% |
| text feature | 1 | 0.00% |
| text version | 1 | 0.00% |
| Type/Category | 1 | 0.00% |
| Media/Entertainment Show | 1 | 0.00% |
| Venue/Entertainment Venue | 1 | 0.00% |
| golf term | 1 | 0.00% |
| Fanbase | 1 | 0.00% |
| Law/Document | 1 | 0.00% |
| danger | 1 | 0.00% |
| Credit Union | 1 | 0.00% |
| Language Pair | 1 | 0.00% |
| Challenge Name | 1 | 0.00% |
| Website/Service | 1 | 0.00% |
| Space Points | 1 | 0.00% |
| Limitation | 1 | 0.00% |
| Data Points | 1 | 0.00% |
| Geometric Object | 1 | 0.00% |
| E-commerce Platform | 1 | 0.00% |
| Art Genre | 1 | 0.00% |
| Art Product/Service | 1 | 0.00% |
| Customer Metric | 1 | 0.00% |
| website components | 1 | 0.00% |
| Bonus codes | 1 | 0.00% |
| Subscription plan | 1 | 0.00% |
| tournament_name | 1 | 0.00% |
| team_count | 1 | 0.00% |
| bracket_id | 1 | 0.00% |
| Invasive species | 1 | 0.00% |
| Natural system | 1 | 0.00% |
| TV personality | 1 | 0.00% |
| Shows | 1 | 0.00% |
| Animal type | 1 | 0.00% |
| cognitive ability | 1 | 0.00% |
| productivity strategies | 1 | 0.00% |
| Chemical Entity | 1 | 0.00% |
| Stuffed animal | 1 | 0.00% |
| location.country.name | 1 | 0.00% |
| location.distance_from_capital.capital_city | 1 | 0.00% |
| location.distance_from_capital.unit | 1 | 0.00% |
| location.name | 1 | 0.00% |
| location.zone | 1 | 0.00% |
| location.type | 1 | 0.00% |
| Parking restriction | 1 | 0.00% |
| Location feature | 1 | 0.00% |
| patio | 1 | 0.00% |
| greenspace | 1 | 0.00% |
| townhome | 1 | 0.00% |
| residential complex | 1 | 0.00% |
| weight unit | 1 | 0.00% |
| legal event | 1 | 0.00% |
| school supplies | 1 | 0.00% |
| Online Discussion | 1 | 0.00% |
| software type | 1 | 0.00% |
| Administrative Division | 1 | 0.00% |
| Historical Region | 1 | 0.00% |
| fabric quality | 1 | 0.00% |
| Domain name contact | 1 | 0.00% |
| Work Experience | 1 | 0.00% |
| Salary | 1 | 0.00% |
| Undesirable appearance | 1 | 0.00% |
| Mental disorder | 1 | 0.00% |
| imaging_type: Type of medical imaging performed | 1 | 0.00% |
| patient_id: Unique identifier for each patient record | 1 | 0.00% |
| room: Room number where imaging was performed | 1 | 0.00% |
| Door canopy | 1 | 0.00% |
| prize | 1 | 0.00% |
| Printer Driver | 1 | 0.00% |
| specification | 1 | 0.00% |
| Product Models | 1 | 0.00% |
| Sales metric | 1 | 0.00% |
| Sales promotion | 1 | 0.00% |
| product/service ID | 1 | 0.00% |
| experience/place | 1 | 0.00% |
| Transaction/Event | 1 | 0.00% |
| Ship name | 1 | 0.00% |
| Venue name | 1 | 0.00% |
| Venue type | 1 | 0.00% |
| Space type | 1 | 0.00% |
| Atmosphere type | 1 | 0.00% |
| Holiday tradition | 1 | 0.00% |
| Gift category | 1 | 0.00% |
| data collection | 1 | 0.00% |
| consent | 1 | 0.00% |
| ailment | 1 | 0.00% |
| medical measurement | 1 | 0.00% |
| medical issue | 1 | 0.00% |
| statistical measurement | 1 | 0.00% |
| media file | 1 | 0.00% |
| Room Number | 1 | 0.00% |
| Total Quantity | 1 | 0.00% |
| Natural element | 1 | 0.00% |
| Learning process | 1 | 0.00% |
| Architecture firm | 1 | 0.00% |
| javascript_requirement.enable_instruction | 1 | 0.00% |
| shipping_restriction.contact_seller_prompt | 1 | 0.00% |
| shipping_restriction.country | 1 | 0.00% |
| javascript_requirement.question_prompt | 1 | 0.00% |
| Context | 1 | 0.00% |
| Social Security Benefit | 1 | 0.00% |
| product details | 1 | 0.00% |
| financial details | 1 | 0.00% |
| Play/Media | 1 | 0.00% |
| scripture | 1 | 0.00% |
| consumer/audience | 1 | 0.00% |
| buyer/seller | 1 | 0.00% |
| Sneaker | 1 | 0.00% |
| requisition_id | 1 | 0.00% |
| Computer File System | 1 | 0.00% |
| master of ceremonies | 1 | 0.00% |
| entertainment program | 1 | 0.00% |
| seller | 1 | 0.00% |
| printing process | 1 | 0.00% |
| orientation | 1 | 0.00% |
| Place of worship | 1 | 0.00% |
| Island Resort | 1 | 0.00% |
| Weather Conditions | 1 | 0.00% |
| Locations | 1 | 0.00% |
| Summary | 1 | 0.00% |
| time_period.age | 1 | 0.00% |
| time_period.end_ma | 1 | 0.00% |
| original_name.author | 1 | 0.00% |
| current_name.species | 1 | 0.00% |
| original_name.genus | 1 | 0.00% |
| current_name.genus | 1 | 0.00% |
| taxonomic_status | 1 | 0.00% |
| Coffee brand | 1 | 0.00% |
| Coffee type | 1 | 0.00% |
| Method of Packaging | 1 | 0.00% |
| office supply | 1 | 0.00% |
| cleaning material | 1 | 0.00% |
| Dental Treatment | 1 | 0.00% |
| rental equipment | 1 | 0.00% |
| Transaction Event | 1 | 0.00% |
| Problem/Issue | 1 | 0.00% |
| District | 1 | 0.00% |
| political parties | 1 | 0.00% |
| voter segment | 1 | 0.00% |
| date and time | 1 | 0.00% |
| Wallpaper | 1 | 0.00% |
| Literature Series | 1 | 0.00% |
| PhoneNumber | 1 | 0.00% |
| AdditionalNotes | 1 | 0.00% |
| Continent/Region | 1 | 0.00% |
| Software/Template | 1 | 0.00% |
| Action/Command | 1 | 0.00% |
| Programmes | 1 | 0.00% |
| video recording | 1 | 0.00% |
| censorship | 1 | 0.00% |
| quality rating | 1 | 0.00% |
| Event ticket | 1 | 0.00% |
| Event access | 1 | 0.00% |
| Retail Location | 1 | 0.00% |
| gift box | 1 | 0.00% |
| Jolie Beauty box | 1 | 0.00% |
| Atmospheric phenomenon | 1 | 0.00% |
| reason for registration | 1 | 0.00% |
| Atmospheric Phenomenon | 1 | 0.00% |
| art technique | 1 | 0.00% |
| art subject | 1 | 0.00% |
| art series | 1 | 0.00% |
| river_details.bridges | 1 | 0.00% |
| river_details.usage | 1 | 0.00% |
| river_details.activities | 1 | 0.00% |
| river_details.length.unit | 1 | 0.00% |
| river_details.whitewater_rating | 1 | 0.00% |
| river_details.origin | 1 | 0.00% |
| river_details.source_elevation.unit | 1 | 0.00% |
| river_details.discharge.unit | 1 | 0.00% |
| Graphic design elements | 1 | 0.00% |
| Blog Series | 1 | 0.00% |
| Series | 1 | 0.00% |
| credit card company | 1 | 0.00% |
| technology feature | 1 | 0.00% |
| Financial management | 1 | 0.00% |
| Decision-making process | 1 | 0.00% |
| Media appearance | 1 | 0.00% |
| Operating System Version | 1 | 0.00% |
| time estimate | 1 | 0.00% |
| guide | 1 | 0.00% |
| Objective/Goal | 1 | 0.00% |
| Attribute/Characteristic | 1 | 0.00% |
| Main Entity | 1 | 0.00% |
| Border | 1 | 0.00% |
| Guarantee | 1 | 0.00% |
| lease_term | 1 | 0.00% |
| Pet Care Service | 1 | 0.00% |
| Dog Walkers | 1 | 0.00% |
| Pets | 1 | 0.00% |
| User type | 1 | 0.00% |
| UI Component | 1 | 0.00% |
| sports league division | 1 | 0.00% |
| sports tournament | 1 | 0.00% |
| Fantasy name | 1 | 0.00% |
| Sports Role | 1 | 0.00% |
| Event planner/brand | 1 | 0.00% |
| Decor theme | 1 | 0.00% |
| Product/Flavor | 1 | 0.00% |
| Ingredient/Food | 1 | 0.00% |
| Training Institute | 1 | 0.00% |
| Golf course | 1 | 0.00% |
| Political District | 1 | 0.00% |
| Website Feature/Functionality | 1 | 0.00% |
| magic | 1 | 0.00% |
| Fee/Charge | 1 | 0.00% |
| collection/team | 1 | 0.00% |
| equityOffered | 1 | 0.00% |
| Sensory Experience | 1 | 0.00% |
| Progress Entity | 1 | 0.00% |
| Security Question/Answer | 1 | 0.00% |
| Encryption | 1 | 0.00% |
| Unencrypted Data | 1 | 0.00% |
| Data Breach | 1 | 0.00% |
| group dynamic | 1 | 0.00% |
| baseball position | 1 | 0.00% |
| name: Full name of the contact person | 1 | 0.00% |
| email: Email address of the contact | 1 | 0.00% |
| date: Date the outreach was made | 1 | 0.00% |
| title: Title of the proposed solution | 1 | 0.00% |
| price: Total price of the proposed solution | 1 | 0.00% |
| status: Current status of the outreach | 1 | 0.00% |
| ref_id: Unique reference identifier for the proposal | 1 | 0.00% |
| solution: Description of the proposed solution | 1 | 0.00% |
| target: Target audience or department for the proposal | 1 | 0.00% |
| Coupon/Discount Website | 1 | 0.00% |
| Growth factors | 1 | 0.00% |
| navigation control | 1 | 0.00% |
| Legal Paperwork | 1 | 0.00% |
| Consumer Rights | 1 | 0.00% |
| Financial Recovery | 1 | 0.00% |
| Legal Dispute | 1 | 0.00% |
| Battle/Event | 1 | 0.00% |
| footwear feature | 1 | 0.00% |
| apparel feature | 1 | 0.00% |
| apparel pattern | 1 | 0.00% |
| serial: Unique identifier for the product unit | 1 | 0.00% |
| agent: Support agent assigned to handle the request | 1 | 0.00% |
| ticket_id: Internal ticket identifier for the RMA | 1 | 0.00% |
| rma_number: Assigned number for the RMA case | 1 | 0.00% |
| Court/Tribunal | 1 | 0.00% |
| reported_by | 1 | 0.00% |
| Household Size | 1 | 0.00% |
| Show/Program | 1 | 0.00% |
| Performance Format | 1 | 0.00% |
| Host/Presenter | 1 | 0.00% |
| gaming console | 1 | 0.00% |
| hardware compatibility | 1 | 0.00% |
| address entity | 1 | 0.00% |
| incident.charges.1.description | 1 | 0.00% |
| incident.suspects.1.age | 1 | 0.00% |
| incident.number_of_complaints | 1 | 0.00% |
| incident.charges.2.description | 1 | 0.00% |
| incident.charges.2.count | 1 | 0.00% |
| source.author | 1 | 0.00% |
| Item number | 1 | 0.00% |
| Currency value | 1 | 0.00% |
| real life person | 1 | 0.00% |
| ink type | 1 | 0.00% |
| golf course | 1 | 0.00% |
| Utilization | 1 | 0.00% |
| Financial Resource | 1 | 0.00% |
| Negative Event | 1 | 0.00% |
| project details | 1 | 0.00% |
| storage appliance | 1 | 0.00% |
| Person/Entrepreneur | 1 | 0.00% |
| Cognitive state | 1 | 0.00% |
| Time/Action | 1 | 0.00% |
| Time/Position | 1 | 0.00% |
| academic units | 1 | 0.00% |
| performer/musician | 1 | 0.00% |
| type of outdoor structure | 1 | 0.00% |
| incidence.rarity | 1 | 0.00% |
| software module | 1 | 0.00% |
| Mixed Martial Arts King | 1 | 0.00% |
| details | 1 | 0.00% |
| statement_reference: Identifier code for the statement document | 1 | 0.00% |
| amount: Financial value associated with the policy or proposal | 1 | 0.00% |
| location: City and state where the event occurred | 1 | 0.00% |
| topic: Subject matter discussed in the statement | 1 | 0.00% |
| speaker: Name of the person delivering the statement | 1 | 0.00% |
| initial meeting | 1 | 0.00% |
| important event | 1 | 0.00% |
| Font | 1 | 0.00% |
| Network device | 1 | 0.00% |
| Shoe component | 1 | 0.00% |
| type of insect | 1 | 0.00% |
| entertainment system | 1 | 0.00% |
| beverage station | 1 | 0.00% |
| security device | 1 | 0.00% |
| environmental control system | 1 | 0.00% |
| personal care device | 1 | 0.00% |
| food planning tool | 1 | 0.00% |
| Animal/Species | 1 | 0.00% |
| Task/Work | 1 | 0.00% |
| Service/Business | 1 | 0.00% |
| Course Subject | 1 | 0.00% |
| Job Description | 1 | 0.00% |
| File content | 1 | 0.00% |
| Hyperlinks | 1 | 0.00% |
| Physics | 1 | 0.00% |
| Emotion Control | 1 | 0.00% |
| Therapy | 1 | 0.00% |
| Therapy Technique | 1 | 0.00% |
| Emotional Pain | 1 | 0.00% |
| Emotional Recovery | 1 | 0.00% |
| material surface | 1 | 0.00% |
| Physical locations | 1 | 0.00% |
| Business areas | 1 | 0.00% |
| Business unit | 1 | 0.00% |
| healthcare professionals | 1 | 0.00% |
| healthcare data management | 1 | 0.00% |
| financial compensation | 1 | 0.00% |
| healthcare policies and regulations | 1 | 0.00% |
| Strain of Cannabis | 1 | 0.00% |
| actions/goals | 1 | 0.00% |
| action/goal | 1 | 0.00% |
| Decade/Time Period | 1 | 0.00% |
| Honorific/Title | 1 | 0.00% |
| Blockchain-based application platform | 1 | 0.00% |
| Software applications | 1 | 0.00% |
| Blockchain customization | 1 | 0.00% |
| People with sleep problems | 1 | 0.00% |
| Short sleep cycle | 1 | 0.00% |
| Insomnia | 1 | 0.00% |
| Light sleepers | 1 | 0.00% |
| Irregular sleep cycle | 1 | 0.00% |
| Region/Country | 1 | 0.00% |
| societal change | 1 | 0.00% |
| Review Evaluation | 1 | 0.00% |
| Product Cost | 1 | 0.00% |
| Purchase Decision | 1 | 0.00% |
| bike race | 1 | 0.00% |
| bike activity | 1 | 0.00% |
| bike type | 1 | 0.00% |
| historical contributions | 1 | 0.00% |
| nutrition program | 1 | 0.00% |
| healthy eating | 1 | 0.00% |
| technical error | 1 | 0.00% |
| emotional reaction | 1 | 0.00% |
| verbal abuse | 1 | 0.00% |
| Furniture modification | 1 | 0.00% |
| Rating Agency | 1 | 0.00% |
| education and training | 1 | 0.00% |
| reservationConfirmation | 1 | 0.00% |
| numberOfTravelers | 1 | 0.00% |
| activityId | 1 | 0.00% |
| activityName | 1 | 0.00% |
| providerName | 1 | 0.00% |
| activityPrice | 1 | 0.00% |
| locationId | 1 | 0.00% |
| locationName | 1 | 0.00% |
| set design | 1 | 0.00% |
| plot points | 1 | 0.00% |
| Province/State | 1 | 0.00% |
| Infrastructure project | 1 | 0.00% |
| Bathroom Type | 1 | 0.00% |
| Fruit quality attribute | 1 | 0.00% |
| Values | 1 | 0.00% |
| mindset | 1 | 0.00% |
| creativity | 1 | 0.00% |
| Technical Information | 1 | 0.00% |
| Website/Resource | 1 | 0.00% |
| quotationId | 1 | 0.00% |
| quotationDate | 1 | 0.00% |
| termsAndConditions | 1 | 0.00% |
| Hair Extension Accessory | 1 | 0.00% |
| mediaFormat | 1 | 0.00% |
| platforms | 1 | 0.00% |
| esrbRating | 1 | 0.00% |
| stock | 1 | 0.00% |
| mascot | 1 | 0.00% |
| television station | 1 | 0.00% |
| Loan limit | 1 | 0.00% |
| Film/Documentary | 1 | 0.00% |
| negative relationship | 1 | 0.00% |
| romantic interest | 1 | 0.00% |
| Personal Data | 1 | 0.00% |
| Login Credentials | 1 | 0.00% |
| Confidential Data | 1 | 0.00% |
| Account Recovery | 1 | 0.00% |
| service/department | 1 | 0.00% |
| customer requirement | 1 | 0.00% |
| holiday decoration | 1 | 0.00% |
| biological pathway | 1 | 0.00% |
| group of friends | 1 | 0.00% |
| online community | 1 | 0.00% |
| document section | 1 | 0.00% |
| protocol | 1 | 0.00% |
| startPort | 1 | 0.00% |
| endPort | 1 | 0.00% |
| projectId | 1 | 0.00% |
| network | 1 | 0.00% |
| sourceRanges | 1 | 0.00% |
| targetTags | 1 | 0.00% |
| type of photograph | 1 | 0.00% |
| task/job | 1 | 0.00% |
| visual art entity | 1 | 0.00% |
| Work/Product | 1 | 0.00% |
| Website/Social Media | 1 | 0.00% |
| conversation topic | 1 | 0.00% |
| timeline | 1 | 0.00% |
| time ahead | 1 | 0.00% |
| Floor Number | 1 | 0.00% |
| Building Name | 1 | 0.00% |
| Building Process | 1 | 0.00% |
| electronic communication | 1 | 0.00% |
| dining establishment | 1 | 0.00% |
| private bathroom | 1 | 0.00% |
| display device | 1 | 0.00% |
| climate control system | 1 | 0.00% |
| Performance Group | 1 | 0.00% |
| Library/Software | 1 | 0.00% |
| Data Encoding | 1 | 0.00% |
| Data Processing | 1 | 0.00% |
| Data Presentation | 1 | 0.00% |
| road type | 1 | 0.00% |
| Mobile App | 1 | 0.00% |
| scheduleId | 1 | 0.00% |
| Color shade | 1 | 0.00% |
| Loyalty Program | 1 | 0.00% |
| externalId | 1 | 0.00% |
| zipcode | 1 | 0.00% |
| scoutingDate | 1 | 0.00% |
| humidity | 1 | 0.00% |
| windSpeed | 1 | 0.00% |
| visibility | 1 | 0.00% |
| business network | 1 | 0.00% |
| farewell expression | 1 | 0.00% |
| Return on Spending | 1 | 0.00% |
| Miles Earned | 1 | 0.00% |
| Purchase Type | 1 | 0.00% |
| Rewards Points Earned | 1 | 0.00% |
| statistic | 1 | 0.00% |
| financial donation | 1 | 0.00% |
| sports sponsorship | 1 | 0.00% |
| lack of attention | 1 | 0.00% |
| Identification Document | 1 | 0.00% |
| expression of dissatisfaction | 1 | 0.00% |
| defense mechanism | 1 | 0.00% |
| TV theme song | 1 | 0.00% |
| birthDate | 1 | 0.00% |
| issuingCountry | 1 | 0.00% |
| durationInDays | 1 | 0.00% |
| numberOfEntries | 1 | 0.00% |
| issuingDate | 1 | 0.00% |
| expiryDate | 1 | 0.00% |
| travelPurpose | 1 | 0.00% |
| expectedArrivalDate | 1 | 0.00% |
| applicationDate | 1 | 0.00% |
| processingFee | 1 | 0.00% |
| Product Part | 1 | 0.00% |
| celebration event | 1 | 0.00% |
| Soccer Match | 1 | 0.00% |
| Computer | 1 | 0.00% |
| owner: Name of the claimant or policyholder | 1 | 0.00% |
| address: Full street address of the claimant | 1 | 0.00% |
| policy_id: Policy number associated with the claim | 1 | 0.00% |
| incident_type: Type of incident causing the claim | 1 | 0.00% |
| Health insurance | 1 | 0.00% |
| Photography studio | 1 | 0.00% |
| Blogging | 1 | 0.00% |
| restaurant.name | 1 | 0.00% |
| restaurant.establishment_date | 1 | 0.00% |
| restaurant.description | 1 | 0.00% |
| restaurant.renovations.0.year | 1 | 0.00% |
| restaurant.renovations.0.type | 1 | 0.00% |
| condition/problem | 1 | 0.00% |
| Political Process | 1 | 0.00% |
| shaving tool | 1 | 0.00% |
| Anatomy/Body Parts | 1 | 0.00% |
| Animals | 1 | 0.00% |
| chess move | 1 | 0.00% |
| game strategy | 1 | 0.00% |
| Business segment | 1 | 0.00% |
| door | 1 | 0.00% |
| Technology/Resolution | 1 | 0.00% |
| artistic asset | 1 | 0.00% |
| professional domain | 1 | 0.00% |
| comparative adjective | 1 | 0.00% |
| color combination | 1 | 0.00% |
| dateOfService | 1 | 0.00% |
| timeOfService | 1 | 0.00% |
| durationInMinutes | 1 | 0.00% |
| bloodPressure | 1 | 0.00% |
| pulse | 1 | 0.00% |
| respiratoryRate | 1 | 0.00% |
| physicalExam | 1 | 0.00% |
| Small group ministry | 1 | 0.00% |
| Church organization | 1 | 0.00% |
| Hotel establishment date | 1 | 0.00% |
| Certification/Approval | 1 | 0.00% |
| beverage item | 1 | 0.00% |
| price item | 1 | 0.00% |
| community administrator | 1 | 0.00% |
| Furniture set | 1 | 0.00% |
| Furniture piece | 1 | 0.00% |
| materials collection | 1 | 0.00% |
| rental_id | 1 | 0.00% |
| Action/Quality | 1 | 0.00% |
| Entity Type/Quality | 1 | 0.00% |
| Fuel Level | 1 | 0.00% |
| Car Rental Company | 1 | 0.00% |
| Fueling Process | 1 | 0.00% |
| natural skin pigment | 1 | 0.00% |
| Church group | 1 | 0.00% |
| distance.calculation_method | 1 | 0.00% |
| latin_name | 1 | 0.00% |
| distance.parsecs | 1 | 0.00% |
| apparent_magnitude.max | 1 | 0.00% |
| characteristics.effective_temperature.value | 1 | 0.00% |
| characteristics.type | 1 | 0.00% |
| distance.light_years | 1 | 0.00% |
| constellation | 1 | 0.00% |
| apparent_magnitude.unit | 1 | 0.00% |
| organizational tool | 1 | 0.00% |
| interactive element | 1 | 0.00% |
| Event/Cause | 1 | 0.00% |
| Geologic formation | 1 | 0.00% |
| Sedimentary rocks | 1 | 0.00% |
| Process of wearing away | 1 | 0.00% |
| Geologic feature | 1 | 0.00% |
| cat fence system | 1 | 0.00% |
| industry inequality | 1 | 0.00% |
| Music Video | 1 | 0.00% |
| Emoji | 1 | 0.00% |
| Jewelry component | 1 | 0.00% |
| Creative Activity | 1 | 0.00% |
| medication.dosing_schedule.0.timing_offset_seconds | 1 | 0.00% |
| group_label | 1 | 0.00% |
| medication.dosing_schedule.0.dose_fraction | 1 | 0.00% |
| medication.administration_type | 1 | 0.00% |
| medication.agent | 1 | 0.00% |
| storage facility | 1 | 0.00% |
| physical altercation | 1 | 0.00% |
| Business Transaction | 1 | 0.00% |
| Feature/Characteristic | 1 | 0.00% |
| status: Current status of the outreach attempt | 1 | 0.00% |
| company: Company advertising the job position | 1 | 0.00% |
| contact_id: Unique identifier for the outreach contact | 1 | 0.00% |
| subject: Subject line of the outreach email | 1 | 0.00% |
| sender: Name of the person initiating the outreach | 1 | 0.00% |
| body: Content of the outreach message | 1 | 0.00% |
| opinion/feeling | 1 | 0.00% |
| Sponsorship level | 1 | 0.00% |
| process event | 1 | 0.00% |
| service or product | 1 | 0.00% |
| Problem/Challenge | 1 | 0.00% |
| Study Criteria | 1 | 0.00% |
| historic metropolis | 1 | 0.00% |
| seaside resort | 1 | 0.00% |
| organization.location_founded.country | 1 | 0.00% |
| social service | 1 | 0.00% |
| Format Type | 1 | 0.00% |
| young students | 1 | 0.00% |
| taking action | 1 | 0.00% |
| aid or assistance | 1 | 0.00% |
| manage or deal with | 1 | 0.00% |
| variation | 1 | 0.00% |
| hobby/group | 1 | 0.00% |
| Basketball Player | 1 | 0.00% |
| Basketball Team | 1 | 0.00% |
| leisure time | 1 | 0.00% |
| Performance Art | 1 | 0.00% |
| spacecraft | 1 | 0.00% |
| planetary formation | 1 | 0.00% |
| Service Request | 1 | 0.00% |
| Gardening setup | 1 | 0.00% |
| Land measurement | 1 | 0.00% |
| technology platform | 1 | 0.00% |
| job sector/group | 1 | 0.00% |
| court ruling | 1 | 0.00% |
| Data Plan | 1 | 0.00% |
| Book series | 1 | 0.00% |
| Model Year Range | 1 | 0.00% |
| Debugging Tool | 1 | 0.00% |
| negative impact on productivity | 1 | 0.00% |
| negative impact on well-being | 1 | 0.00% |
| human race | 1 | 0.00% |
| marketplace | 1 | 0.00% |
| Financial document | 1 | 0.00% |
| company staff | 1 | 0.00% |
| market trend | 1 | 0.00% |
| company performance | 1 | 0.00% |
| Process Improvement | 1 | 0.00% |
| Goods/Products | 1 | 0.00% |
| External Audit/Inspection | 1 | 0.00% |
| Internal Audit/Inspection | 1 | 0.00% |
| finance and economics | 1 | 0.00% |
| errands | 1 | 0.00% |
| Meal type | 1 | 0.00% |
| sport event | 1 | 0.00% |
| Nonprofit organization | 1 | 0.00% |
| Career opportunity | 1 | 0.00% |
| Educational support | 1 | 0.00% |
| Partnership organization | 1 | 0.00% |
| Place/Museum | 1 | 0.00% |
| Age/Time period | 1 | 0.00% |
| housing type | 1 | 0.00% |
| governmental institution | 1 | 0.00% |
| postal service | 1 | 0.00% |
| nation | 1 | 0.00% |
| transaction process | 1 | 0.00% |
| Yarn Composition | 1 | 0.00% |
| Yarn Weight | 1 | 0.00% |
| Software/Security | 1 | 0.00% |
| Technology/Entertainment | 1 | 0.00% |
| Luxury Designer Replicas | 1 | 0.00% |
| startup_name | 1 | 0.00% |
| Media content type | 1 | 0.00% |
| Person/Expert | 1 | 0.00% |
| Situation/Scenario | 1 | 0.00% |
| sibling | 1 | 0.00% |
| Product Code/Number | 1 | 0.00% |
| CSS property | 1 | 0.00% |
| religious institution | 1 | 0.00% |
| divine being | 1 | 0.00% |
| religious followers | 1 | 0.00% |
| spiritual growth | 1 | 0.00% |
| domain.territory | 1 | 0.00% |
| domain.country_code | 1 | 0.00% |
| domain.dispute_resolution.methods | 1 | 0.00% |
| domain.registry.name | 1 | 0.00% |
| art/craftwork | 1 | 0.00% |
| supernatural being/organization | 1 | 0.00% |
| physical asset | 1 | 0.00% |
| infant trainer | 1 | 0.00% |
| shoe material | 1 | 0.00% |
| Phrase | 1 | 0.00% |
| Game development stage | 1 | 0.00% |
| Game version | 1 | 0.00% |
| Course Materials | 1 | 0.00% |
| Assessment Format | 1 | 0.00% |
| Competition Category | 1 | 0.00% |
| conservation of financial resources | 1 | 0.00% |
| high-quality materials | 1 | 0.00% |
| passage of events | 1 | 0.00% |
| competency | 1 | 0.00% |
| society | 1 | 0.00% |
| skilled | 1 | 0.00% |
| Ship Name | 1 | 0.00% |
| laboratory equipment | 1 | 0.00% |
| technology resource | 1 | 0.00% |
| smartphone model | 1 | 0.00% |
| hardware issue | 1 | 0.00% |
| camera issue | 1 | 0.00% |
| Chamber Ensemble | 1 | 0.00% |
| Opera Company | 1 | 0.00% |
| MVP | 1 | 0.00% |
| Course Code | 1 | 0.00% |
| Credit Rating Agency | 1 | 0.00% |
| Township | 1 | 0.00% |
| productivity tool | 1 | 0.00% |
| Car type | 1 | 0.00% |
| assignedTo | 1 | 0.00% |
| fileName | 1 | 0.00% |
| person/anthropomorphic entity | 1 | 0.00% |
| Product Styles | 1 | 0.00% |
| functional item | 1 | 0.00% |
| location.relative_location | 1 | 0.00% |
| german_name | 1 | 0.00% |
| Certification/Standard | 1 | 0.00% |
| Podcast episode | 1 | 0.00% |
| dynamic environment | 1 | 0.00% |
| Competition category | 1 | 0.00% |
| House Feature | 1 | 0.00% |
| submitter | 1 | 0.00% |
| assigned_to | 1 | 0.00% |
| event location | 1 | 0.00% |
| natural beauty | 1 | 0.00% |
| ambiance | 1 | 0.00% |
| event planning | 1 | 0.00% |
| Process/Service | 1 | 0.00% |
| Object/Infrastructure | 1 | 0.00% |
| Transportation System | 1 | 0.00% |
| travel booking website | 1 | 0.00% |
| medical device component | 1 | 0.00% |
| anatomical location | 1 | 0.00% |
| start | 1 | 0.00% |
| end | 1 | 0.00% |
| keyAlgorithm | 1 | 0.00% |
| signatureAlgorithm | 1 | 0.00% |
| years | 1 | 0.00% |
| months | 1 | 0.00% |
| days | 1 | 0.00% |
| process/result | 1 | 0.00% |
| Supernatural Being | 1 | 0.00% |
| Supernatural Events | 1 | 0.00% |
| Religious Followers | 1 | 0.00% |
| Belief System | 1 | 0.00% |
| Person/Brand | 1 | 0.00% |
| legal institution | 1 | 0.00% |
| legal entities | 1 | 0.00% |
| legal proceeding | 1 | 0.00% |
| legal cases | 1 | 0.00% |
| legal decision-makers | 1 | 0.00% |
| Fashion Design | 1 | 0.00% |
| conflict entity | 1 | 0.00% |
| socioeconomic entity | 1 | 0.00% |
| Cloud Storage | 1 | 0.00% |
| business partner | 1 | 0.00% |
| Analyzer instrument | 1 | 0.00% |
| Test equipment | 1 | 0.00% |
| Oscilloscope instrument | 1 | 0.00% |
| Oscilloscope model | 1 | 0.00% |
| Mixer instrument | 1 | 0.00% |
| Temperature range | 1 | 0.00% |
| Bandwidth specification | 1 | 0.00% |
| Sampling rate specification | 1 | 0.00% |
| User Interface Design | 1 | 0.00% |
| Product description | 1 | 0.00% |
| Product origin | 1 | 0.00% |
| transaction platform | 1 | 0.00% |
| intellectual process | 1 | 0.00% |
| Checkpoint command | 1 | 0.00% |
| System action | 1 | 0.00% |
| Email Marketing Solution | 1 | 0.00% |
| Email Format | 1 | 0.00% |
| Military Operation | 1 | 0.00% |
| Academic Degree | 1 | 0.00% |
| Order Retrieval | 1 | 0.00% |
| Product Posting | 1 | 0.00% |
| Data Payload | 1 | 0.00% |
| Authentication Key | 1 | 0.00% |
| job compensation | 1 | 0.00% |
| financial transaction company | 1 | 0.00% |
| website security certification | 1 | 0.00% |
| emotion/action | 1 | 0.00% |
| object/reference | 1 | 0.00% |
| Contaminant | 1 | 0.00% |
| Commercial space | 1 | 0.00% |
| Harmful substance | 1 | 0.00% |
| Polysaccharide | 1 | 0.00% |
| Microorganism strain | 1 | 0.00% |
| Company registration number | 1 | 0.00% |
| perk | 1 | 0.00% |
| Playing style | 1 | 0.00% |
| Player group | 1 | 0.00% |
| sports league | 1 | 0.00% |
| Cryptocurrency exchange rate | 1 | 0.00% |
| insect species | 1 | 0.00% |
| tree part | 1 | 0.00% |
| fungal species | 1 | 0.00% |
| People/Residents | 1 | 0.00% |
| test_date: Date when the test was performed | 1 | 0.00% |
| patient_name: Full name of the patient undergoing the test | 1 | 0.00% |
| test_code: Code identifying the specific laboratory test | 1 | 0.00% |
| status: Current status of the test result | 1 | 0.00% |
| result_value: Numerical value of the test result | 1 | 0.00% |
| Offer/Service | 1 | 0.00% |
| level of analysis | 1 | 0.00% |
| Ranking/List | 1 | 0.00% |
| software/program | 1 | 0.00% |
| textual formatting | 1 | 0.00% |
| application interface | 1 | 0.00% |
| output format | 1 | 0.00% |
| Postpone Time | 1 | 0.00% |
| Testing procedure | 1 | 0.00% |
| Electrical current | 1 | 0.00% |
| article.references.0.statute_name | 1 | 0.00% |
| article.references.0.date | 1 | 0.00% |
| subsection.description | 1 | 0.00% |
| code_name | 1 | 0.00% |
| article.number | 1 | 0.00% |
| server | 1 | 0.00% |
| digital album | 1 | 0.00% |
| smartphone | 1 | 0.00% |
| tablet computer | 1 | 0.00% |
| return_amount | 1 | 0.00% |
| product_sku | 1 | 0.00% |
| flaw | 1 | 0.00% |
| wrongdoing | 1 | 0.00% |
| disclosure | 1 | 0.00% |
| preventive measure | 1 | 0.00% |
| file encryption | 1 | 0.00% |
| Competition League | 1 | 0.00% |
| Achievement Level | 1 | 0.00% |
| personal struggle | 1 | 0.00% |
| Instagram Account | 1 | 0.00% |
| pet accessory | 1 | 0.00% |
| hygiene product | 1 | 0.00% |
| digital version | 1 | 0.00% |
| Furniture accessory | 1 | 0.00% |
| Textile materials | 1 | 0.00% |
| Furniture type | 1 | 0.00% |
| Furniture measurement | 1 | 0.00% |
| Fitness Exercise | 1 | 0.00% |
| Outcome/Effect | 1 | 0.00% |
| File Type | 1 | 0.00% |
| Process/Function | 1 | 0.00% |
| Syntax/Keyword | 1 | 0.00% |
| Spa/Wellness retreats | 1 | 0.00% |
| Spa Services | 1 | 0.00% |
| Relaxation technique | 1 | 0.00% |
| Fashion product | 1 | 0.00% |
| thing | 1 | 0.00% |
| E-commerce site | 1 | 0.00% |
| Sale promotion | 1 | 0.00% |
| Baby products | 1 | 0.00% |
| Coupon code | 1 | 0.00% |
| Time limit | 1 | 0.00% |
| expiration date | 1 | 0.00% |
| decrease in value | 1 | 0.00% |
| time constraint | 1 | 0.00% |
| instructional resources | 1 | 0.00% |
| 3D model | 1 | 0.00% |
| replacing | 1 | 0.00% |
| 3D models | 1 | 0.00% |
| type of building | 1 | 0.00% |
| type of room | 1 | 0.00% |
| Entity Ownership | 1 | 0.00% |
| Entity Location | 1 | 0.00% |
| Anatomy/Emotion | 1 | 0.00% |
| Artistic technique | 1 | 0.00% |
| Occupation/Task | 1 | 0.00% |
| Legal Terms | 1 | 0.00% |
| Written work | 1 | 0.00% |
| Occult practice | 1 | 0.00% |
| Paranormal entity | 1 | 0.00% |
| Man who tries to pick up women | 1 | 0.00% |
| pub_date | 1 | 0.00% |
| review_rating | 1 | 0.00% |
| review_status | 1 | 0.00% |
| Promotional Unit | 1 | 0.00% |
| construction/repair work | 1 | 0.00% |
| Science | 1 | 0.00% |
| consumer feedback | 1 | 0.00% |
| business opportunity | 1 | 0.00% |
| Chance of precipitation | 1 | 0.00% |
| Clothing Category | 1 | 0.00% |
| media quality | 1 | 0.00% |
| GPS Coordinate | 1 | 0.00% |
| written communication | 1 | 0.00% |
| general population | 1 | 0.00% |
| Excel function | 1 | 0.00% |
| Album/EP | 1 | 0.00% |
| Pop Culture Phenomenon | 1 | 0.00% |
| Information product | 1 | 0.00% |
| American flag | 1 | 0.00% |
| educational | 1 | 0.00% |
| cybercriminals | 1 | 0.00% |
| malicious actors | 1 | 0.00% |
| advertising strategy | 1 | 0.00% |
| malicious advertisements | 1 | 0.00% |
| Idiom/Expression | 1 | 0.00% |
| academic study | 1 | 0.00% |
| Tax Rate | 1 | 0.00% |
| Tax Liability | 1 | 0.00% |
| Tax Credit | 1 | 0.00% |
| Event opportunity | 1 | 0.00% |
| audit_id | 1 | 0.00% |
| API version | 1 | 0.00% |
| Advocacy Campaign | 1 | 0.00% |
| Fundraiser Event | 1 | 0.00% |
| Charitable Giving | 1 | 0.00% |
| totalAmount | 1 | 0.00% |
| Affiliation | 1 | 0.00% |
| Time Requirement | 1 | 0.00% |
| court: Name of the court handling the case | 1 | 0.00% |
| plaintiff_name: Name of the individual or entity filing the case | 1 | 0.00% |
| defendant_name: Name of the individual or entity being sued | 1 | 0.00% |
| claim_amount: Monetary amount claimed in damages | 1 | 0.00% |
| docket_number: Unique identifier assigned to the case | 1 | 0.00% |
| Mobile device model | 1 | 0.00% |
| Software type | 1 | 0.00% |
| Legal case | 1 | 0.00% |
| photography type | 1 | 0.00% |
| website traffic | 1 | 0.00% |
| website revenue | 1 | 0.00% |
| computer system | 1 | 0.00% |
| website hosting | 1 | 0.00% |
| natural resources | 1 | 0.00% |
| Harmful Behavior | 1 | 0.00% |
| Living Being | 1 | 0.00% |
| magnetic object | 1 | 0.00% |
| magnetic field lines | 1 | 0.00% |
| Drinking establishment | 1 | 0.00% |
| product dimension | 1 | 0.00% |
| Adverbial phrase | 1 | 0.00% |
| agricultural land | 1 | 0.00% |
| celestial entity | 1 | 0.00% |
| Historical Figures | 1 | 0.00% |
| Genetic Factor | 1 | 0.00% |
| event space | 1 | 0.00% |
| Holiday treat | 1 | 0.00% |
| positive feedback | 1 | 0.00% |
| Support and Guidance | 1 | 0.00% |
| Craft Material | 1 | 0.00% |
| Class Name | 1 | 0.00% |
| Method Name | 1 | 0.00% |
| Flower part | 1 | 0.00% |
| Legal Agreement | 1 | 0.00% |
| Employment Policy | 1 | 0.00% |
| Employment Law | 1 | 0.00% |
| Legal Policy | 1 | 0.00% |
| Shipping option | 1 | 0.00% |
| Book/Title | 1 | 0.00% |
| Advertising/Marketing | 1 | 0.00% |
| Creative/Artistic Work | 1 | 0.00% |
| Sport event | 1 | 0.00% |
| Punishment | 1 | 0.00% |
| Event management | 1 | 0.00% |
| disappointment | 1 | 0.00% |
| variable reference | 1 | 0.00% |
| technical document | 1 | 0.00% |
| people's well-being | 1 | 0.00% |
| eps | 1 | 0.00% |
| net_margin | 1 | 0.00% |
| call_duration | 1 | 0.00% |
| market_cap | 1 | 0.00% |
| Memorial Service | 1 | 0.00% |
| Support Service | 1 | 0.00% |
| engine | 1 | 0.00% |
| transmission | 1 | 0.00% |
| Organizational Management | 1 | 0.00% |
| Security policy | 1 | 0.00% |
| care facility | 1 | 0.00% |
| creative professionals/artists | 1 | 0.00% |
| live music events | 1 | 0.00% |
| certificateId | 1 | 0.00% |
| calibrationMethod | 1 | 0.00% |
| calibrationStatus | 1 | 0.00% |
| procedureName | 1 | 0.00% |
| standardId | 1 | 0.00% |
| standardType | 1 | 0.00% |
| type of activity | 1 | 0.00% |
| High-flow S400 model | 1 | 0.00% |
| FMW 67 | 1 | 0.00% |
| material properties | 1 | 0.00% |
| fiber type | 1 | 0.00% |
| metal type | 1 | 0.00% |
| hat type | 1 | 0.00% |
| pages | 1 | 0.00% |
| developer | 1 | 0.00% |
| program_name | 1 | 0.00% |
| research field | 1 | 0.00% |
| genetic sequence | 1 | 0.00% |
| breakthrough | 1 | 0.00% |
| scientific investigation | 1 | 0.00% |
| medical advancement | 1 | 0.00% |
| Mountain range | 1 | 0.00% |
| Snow | 1 | 0.00% |
| prison official | 1 | 0.00% |
| authorization | 1 | 0.00% |
| Language proficiency | 1 | 0.00% |
| artwork/production | 1 | 0.00% |
| international development goals | 1 | 0.00% |
| Dynasty/Ruling House | 1 | 0.00% |
| Monarch/Ruler | 1 | 0.00% |
| Technology standard | 1 | 0.00% |
| residential unit | 1 | 0.00% |
| Desire/Action | 1 | 0.00% |
| Societal norm | 1 | 0.00% |
| Gaming terminology | 1 | 0.00% |
| Travel company | 1 | 0.00% |
| Music equipment | 1 | 0.00% |
| Blogging platform | 1 | 0.00% |
| Scientific method | 1 | 0.00% |
| audio files | 1 | 0.00% |
| design project | 1 | 0.00% |
| decision-making group | 1 | 0.00% |
| decision process | 1 | 0.00% |
| Braking system | 1 | 0.00% |
| Braking component | 1 | 0.00% |
| claim_count | 1 | 0.00% |
| filing_fee | 1 | 0.00% |
| Occupation/Group of people | 1 | 0.00% |
| Interference | 1 | 0.00% |
| Financial event | 1 | 0.00% |
| Christmas Decoration | 1 | 0.00% |
| object part | 1 | 0.00% |
| Rugby location | 1 | 0.00% |
| scientific facility | 1 | 0.00% |
| Power | 1 | 0.00% |
| Service/Practice | 1 | 0.00% |
| geological formations | 1 | 0.00% |
| fruit tree | 1 | 0.00% |
| hill station | 1 | 0.00% |
| experiment phase | 1 | 0.00% |
| buildings | 1 | 0.00% |
| seasonal planting | 1 | 0.00% |
| Electronic device accessory | 1 | 0.00% |
| storage method | 1 | 0.00% |
| Health service | 1 | 0.00% |
| page number | 1 | 0.00% |
| sensitive data | 1 | 0.00% |
| non-sensitive data | 1 | 0.00% |
| Skincare Product | 1 | 0.00% |
| Skincare Ingredient | 1 | 0.00% |
| Note | 1 | 0.00% |
| Opinion/Quality | 1 | 0.00% |
| Environmental condition | 1 | 0.00% |
| gift card balance | 1 | 0.00% |
| victim device | 1 | 0.00% |
| means of attack | 1 | 0.00% |
| victim files | 1 | 0.00% |
| performancerating | 1 | 0.00% |
| Game Achievement | 1 | 0.00% |
| discovery.discovery_rank | 1 | 0.00% |
| observations.last_observation_date | 1 | 0.00% |
| observations.observation_period_years | 1 | 0.00% |
| orbital_characteristics.orbital_period_years | 1 | 0.00% |
| orbital_characteristics.perihelion | 1 | 0.00% |
| orbital_characteristics.eccentricity | 1 | 0.00% |
| physical_characteristics.absolute_magnitude | 1 | 0.00% |
| physical_characteristics.albedo | 1 | 0.00% |
| physical_characteristics.estimated_diameter_km | 1 | 0.00% |
| physical_characteristics.observations_used | 1 | 0.00% |
| physical_characteristics.observation_year | 1 | 0.00% |
| Musician/Band Member | 1 | 0.00% |
| Competition Round | 1 | 0.00% |
| employee well-being | 1 | 0.00% |
| neurochemicals | 1 | 0.00% |
| work setting | 1 | 0.00% |
| career advancement | 1 | 0.00% |
| discount_amount | 1 | 0.00% |
| min_subtotal | 1 | 0.00% |
| max_discount | 1 | 0.00% |
| waste material | 1 | 0.00% |
| emotion/state | 1 | 0.00% |
| accident/incident | 1 | 0.00% |
| organization/team | 1 | 0.00% |
| Storage device | 1 | 0.00% |
| interpretation | 1 | 0.00% |
| Investment outcomes | 1 | 0.00% |
| competitors | 1 | 0.00% |
| Pet Door | 1 | 0.00% |
| onboarding_code | 1 | 0.00% |
| domesticated animal | 1 | 0.00% |
| model_number | 1 | 0.00% |
| responsible_staff | 1 | 0.00% |
| Grape variety | 1 | 0.00% |
| JavaScript library | 1 | 0.00% |
| web development technique | 1 | 0.00% |
| research initiative | 1 | 0.00% |
| education initiative | 1 | 0.00% |
| Website/Platform Name | 1 | 0.00% |
| Web Address/URL | 1 | 0.00% |
| Government Involvement | 1 | 0.00% |
| Criminal organizations | 1 | 0.00% |
| Geographical locations | 1 | 0.00% |
| Political instability | 1 | 0.00% |
| Aggravating conflict | 1 | 0.00% |
| Contributing to Boko Haram's growth | 1 | 0.00% |
| Volunteer group | 1 | 0.00% |
| college applicants | 1 | 0.00% |
| insect pest | 1 | 0.00% |
| Moving company | 1 | 0.00% |
| medical process | 1 | 0.00% |
| personal property | 1 | 0.00% |
| row | 1 | 0.00% |
| rack | 1 | 0.00% |
| cabling | 1 | 0.00% |
| Academic document | 1 | 0.00% |
| Athletic Department | 1 | 0.00% |
| webpagina | 1 | 0.00% |
| fysiek object | 1 | 0.00% |
| webwinkel | 1 | 0.00% |
| mythical creatures | 1 | 0.00% |
| marketing_budget | 1 | 0.00% |
| iporevisedvaluation | 1 | 0.00% |
| originalheadcountapex | 1 | 0.00% |
| guitar effect | 1 | 0.00% |
| guitar tuning | 1 | 0.00% |
| guitar feature | 1 | 0.00% |
| feeding schedule | 1 | 0.00% |
| Hobby/Activity | 1 | 0.00% |
| Movie/Book Title | 1 | 0.00% |
| parenting technique | 1 | 0.00% |
| clothing size | 1 | 0.00% |
| Graph/Image | 1 | 0.00% |
| Biological substance | 1 | 0.00% |
| Industry/Manufacturing | 1 | 0.00% |
| Event series | 1 | 0.00% |
| Marathon event | 1 | 0.00% |
| Emotion/Preference | 1 | 0.00% |
| future outlook | 1 | 0.00% |
| Medical Outcome | 1 | 0.00% |
| Type of Treatment | 1 | 0.00% |
| diagnosis_date | 1 | 0.00% |
| treatment_status | 1 | 0.00% |
| allergy | 1 | 0.00% |
| bmi | 1 | 0.00% |
| vehicle.operators.0.acquisition.funding_details | 1 | 0.00% |
| vehicle.operators.0.acquisition.cost | 1 | 0.00% |
| vehicle.specifications.dimensions.length | 1 | 0.00% |
| vehicle.specifications.crew_capacity | 1 | 0.00% |
| vehicle.manufacturer.location | 1 | 0.00% |
| vehicle.combat_usage.0.conflict | 1 | 0.00% |
| vehicle.name | 1 | 0.00% |
| vehicle.specifications.fording_depth | 1 | 0.00% |
| vehicle.specifications.range | 1 | 0.00% |
| vehicle.specifications.transmission | 1 | 0.00% |
| vehicle.specifications.ground_clearance | 1 | 0.00% |
| vehicle.specifications.speed | 1 | 0.00% |
| vehicle.operators.0.details | 1 | 0.00% |
| vehicle.specifications.dimensions.height | 1 | 0.00% |
| outdoor place | 1 | 0.00% |
| place to eat | 1 | 0.00% |
| Live Broadcast Platform | 1 | 0.00% |
| cutting style | 1 | 0.00% |
| Online communities | 1 | 0.00% |
| Evaluation of user-generated content | 1 | 0.00% |
| sale_amount | 1 | 0.00% |
| sales_period | 1 | 0.00% |
| Position/Action | 1 | 0.00% |
| Tea region | 1 | 0.00% |
| Flavor profile | 1 | 0.00% |
| Tea variety | 1 | 0.00% |
| Tea component | 1 | 0.00% |
| Temperature measurement | 1 | 0.00% |
| Tax relief | 1 | 0.00% |
| Maximum corporation tax liability | 1 | 0.00% |
| Education and training | 1 | 0.00% |
| Ability or proficiency in a particular activity or subject | 1 | 0.00% |
| Well-being and nutrition | 1 | 0.00% |
| Financial planning and management | 1 | 0.00% |
| file location | 1 | 0.00% |
| tank size | 1 | 0.00% |
| tank dimensions | 1 | 0.00% |
| Author/Creator | 1 | 0.00% |
| CD/Album Name | 1 | 0.00% |
| Website/Webpage | 1 | 0.00% |
| queue management | 1 | 0.00% |
| age identity | 1 | 0.00% |
| cultural event | 1 | 0.00% |
| Financial Investment | 1 | 0.00% |
| Sovereign Wealth Funds | 1 | 0.00% |
| Financial instruments | 1 | 0.00% |
| Political systems | 1 | 0.00% |
| Economic groupings | 1 | 0.00% |
| Government actions | 1 | 0.00% |
| Interior Design Ideas | 1 | 0.00% |
| medical drug | 1 | 0.00% |
| irrigation | 1 | 0.00% |
| soil level | 1 | 0.00% |
| Boat model | 1 | 0.00% |
| crossword solving technique | 1 | 0.00% |
| Distribution method | 1 | 0.00% |
| Custom | 1 | 0.00% |
| individual item | 1 | 0.00% |
| Data samples | 1 | 0.00% |
| Airport Terminal | 1 | 0.00% |
| mathematics | 1 | 0.00% |
| Tea Brand | 1 | 0.00% |
| fitness tracker | 1 | 0.00% |
| heart rate monitor | 1 | 0.00% |
| communication feature | 1 | 0.00% |
| wearable device attachment | 1 | 0.00% |
| Internet Companies | 1 | 0.00% |
| task/process | 1 | 0.00% |
| author.experience | 1 | 0.00% |
| price.amount | 1 | 0.00% |
| price.currency | 1 | 0.00% |
| camera_model | 1 | 0.00% |
| page_count | 1 | 0.00% |
| Architecture/Residential Buildings | 1 | 0.00% |
| coping mechanism | 1 | 0.00% |
| social situation | 1 | 0.00% |
| trading technique | 1 | 0.00% |
| trading data | 1 | 0.00% |
| trading strategies | 1 | 0.00% |
| literary character | 1 | 0.00% |
| Pedal steel boss | 1 | 0.00% |
| Drummer | 1 | 0.00% |
| Recording equipment | 1 | 0.00% |
| Event/Project name | 1 | 0.00% |
| Firmware Name | 1 | 0.00% |
| inspectionDate | 1 | 0.00% |
| rawMaterialName | 1 | 0.00% |
| rawMaterialGrade | 1 | 0.00% |
| quantityInspected | 1 | 0.00% |
| overallInspectionStatus | 1 | 0.00% |
| inspectorName | 1 | 0.00% |
| inspectorSignature | 1 | 0.00% |
| approverName | 1 | 0.00% |
| approverSignature | 1 | 0.00% |
| labelPlacement | 1 | 0.00% |
| labelOrientation | 1 | 0.00% |
| templateName | 1 | 0.00% |
| templateVersion | 1 | 0.00% |
| relational entity | 1 | 0.00% |
| atmospheric phenomenon | 1 | 0.00% |
| Religious Person | 1 | 0.00% |
| Template Name | 1 | 0.00% |
| Design Resource | 1 | 0.00% |
| Zodiac Sign | 1 | 0.00% |
| Classmates | 1 | 0.00% |
| craft skill | 1 | 0.00% |
| educational resource | 1 | 0.00% |
| Entertainment entity | 1 | 0.00% |
| professional sports league | 1 | 0.00% |
| Device Interface | 1 | 0.00% |
| task/assignment | 1 | 0.00% |
| Religious practice | 1 | 0.00% |
| Yoga Pose | 1 | 0.00% |
| company/enterprise | 1 | 0.00% |
| business expansion | 1 | 0.00% |
| Operating System issues | 1 | 0.00% |
| Malware program | 1 | 0.00% |
| personal quality | 1 | 0.00% |
| Computing device | 1 | 0.00% |
| contact details | 1 | 0.00% |
| additional information | 1 | 0.00% |
| group of people working together towards a common goal | 1 | 0.00% |
| people working for a company | 1 | 0.00% |
| organized occasion or gathering | 1 | 0.00% |
| person's name | 1 | 0.00% |
| Smartphone | 1 | 0.00% |
| Telecommunications standard | 1 | 0.00% |
| Product subcategory | 1 | 0.00% |
| Algorithm | 1 | 0.00% |
| Layer | 1 | 0.00% |
| Algorithm Set | 1 | 0.00% |
| Emergency Service | 1 | 0.00% |
| biochemistry | 1 | 0.00% |
| health and fitness | 1 | 0.00% |
| Financial benefit | 1 | 0.00% |
| diameter | 1 | 0.00% |
| discovery.observatory.location.prefecture | 1 | 0.00% |
| era_name | 1 | 0.00% |
| emotion/attitude | 1 | 0.00% |
| judgment/evaluation | 1 | 0.00% |
| scientific technique | 1 | 0.00% |
| shape/conformation | 1 | 0.00% |
| structural feature | 1 | 0.00% |
| body support | 1 | 0.00% |
| construction | 1 | 0.00% |
| educational materials | 1 | 0.00% |
| Sale Discount | 1 | 0.00% |
| Information/Resource | 1 | 0.00% |
| Linguistic resource | 1 | 0.00% |
| power-up | 1 | 0.00% |
| person/business | 1 | 0.00% |
| Social network platform | 1 | 0.00% |
| Profession/Religion | 1 | 0.00% |
| Militant group | 1 | 0.00% |
| Military Equipment | 1 | 0.00% |
| Social Status | 1 | 0.00% |
| Artistic Work/Musical Production | 1 | 0.00% |
| Place/Region | 1 | 0.00% |
| renewable energy technology | 1 | 0.00% |
| publication_volume | 1 | 0.00% |
| publication_issue | 1 | 0.00% |
| Affected Industries | 1 | 0.00% |
| report_context.epidemiology_notes | 1 | 0.00% |
| patient_case.clinical_presentation.skin_lesions.description | 1 | 0.00% |
| patient_case.clinical_presentation.medical_history.0.status | 1 | 0.00% |
| patient_case.genetic_analysis.mutation_details.dna_nomenclature | 1 | 0.00% |
| disease_profile.genetic_mapping.associated_gene | 1 | 0.00% |
| disease_profile.inheritance_pattern | 1 | 0.00% |
| disease_profile.classification | 1 | 0.00% |
| patient_case.genetic_analysis.mutation_details.exon_location | 1 | 0.00% |
| patient_case.clinical_presentation.skin_lesions.size_range.max_mm | 1 | 0.00% |
| disease_profile.abbreviation | 1 | 0.00% |
| patient_case.clinical_presentation.skin_lesions.symptoms | 1 | 0.00% |
| report_context.geographic_significance | 1 | 0.00% |
| patient_case.demographics.age | 1 | 0.00% |
| patient_case.genetic_analysis.mutation_details.functional_effect | 1 | 0.00% |
| patient_case.genetic_analysis.mutation_details.protein_nomenclature | 1 | 0.00% |
| patient_case.genetic_analysis.gene_tested | 1 | 0.00% |
| disease_profile.genetic_mapping.gene_aliases | 1 | 0.00% |
| Price Level | 1 | 0.00% |
| Music Subgenre | 1 | 0.00% |
| Visual aesthetics | 1 | 0.00% |
| User Interface Component | 1 | 0.00% |
| government | 1 | 0.00% |
| Sports Organization | 1 | 0.00% |
| Sports Regulation | 1 | 0.00% |
| unexpected event | 1 | 0.00% |
| currency range | 1 | 0.00% |
| brand/logo | 1 | 0.00% |
| size specification | 1 | 0.00% |
| Product/Regulation | 1 | 0.00% |
| Regulation parts/Sections | 1 | 0.00% |
| Quantity/Unit of measurement | 1 | 0.00% |
| Food/Condiment | 1 | 0.00% |
| Culture/Ethnicity | 1 | 0.00% |
| electromagnetic scattering method | 1 | 0.00% |
| electromagnetic scattering coefficients | 1 | 0.00% |
| computer hardware architecture | 1 | 0.00% |
| personal drive | 1 | 0.00% |
| interruption | 1 | 0.00% |
| situation type | 1 | 0.00% |
| knockout count | 1 | 0.00% |
| weight class | 1 | 0.00% |
| Object/Service | 1 | 0.00% |
| involvement | 1 | 0.00% |
| Sponsorship | 1 | 0.00% |
| League | 1 | 0.00% |
| Policy Entity | 1 | 0.00% |
| Product/Service Entity | 1 | 0.00% |
| Business goal | 1 | 0.00% |
| Membership ID | 1 | 0.00% |
| asteroid.named_after.occupation | 1 | 0.00% |
| asteroid.named_after.description | 1 | 0.00% |
| asteroid.discovery.discoverer.affiliation.observatory | 1 | 0.00% |
| asteroid.named_after.birth_location.country | 1 | 0.00% |
| information resource | 1 | 0.00% |
| spoofing_details | 1 | 0.00% |
| risk_level | 1 | 0.00% |
| stationary | 1 | 0.00% |
| Equipment Type | 1 | 0.00% |
| clothing defect | 1 | 0.00% |
| finish type | 1 | 0.00% |
| hardware entity | 1 | 0.00% |
| Content strategy | 1 | 0.00% |
| Content style | 1 | 0.00% |
| donor_id | 1 | 0.00% |
| Public art piece | 1 | 0.00% |
| Public space | 1 | 0.00% |
| Life Event | 1 | 0.00% |
| Dental Procedure | 1 | 0.00% |
| Dental Appliance | 1 | 0.00% |
| Aromatherapy ingredients | 1 | 0.00% |
| Color palette | 1 | 0.00% |
| constraint | 1 | 0.00% |
| Gaming feature | 1 | 0.00% |
| Person/Occupant | 1 | 0.00% |
| Expense/Cost | 1 | 0.00% |
| Quality/Construction | 1 | 0.00% |
| report_time | 1 | 0.00% |
| Password Manager | 1 | 0.00% |
| Password Reset | 1 | 0.00% |
| Web Portal | 1 | 0.00% |
| Loss | 1 | 0.00% |
| Anatomical Entity | 1 | 0.00% |
| application technique | 1 | 0.00% |
| Waste material | 1 | 0.00% |
| Electronic waste | 1 | 0.00% |
| Garbage | 1 | 0.00% |
| First Responders | 1 | 0.00% |
| Fire Department | 1 | 0.00% |
| Hazmat Team | 1 | 0.00% |
| Worker | 1 | 0.00% |
| Suspicious Package | 1 | 0.00% |
| book cover | 1 | 0.00% |
| digital book copy | 1 | 0.00% |
| Production capacity | 1 | 0.00% |
| advice and tactics | 1 | 0.00% |
| information examination | 1 | 0.00% |
| web design proficiency | 1 | 0.00% |
| experts | 1 | 0.00% |
| Collection of artworks | 1 | 0.00% |
| Enforcement Program | 1 | 0.00% |
| Professional Organization | 1 | 0.00% |
| wearing style | 1 | 0.00% |
| Medical Measurement | 1 | 0.00% |
| Sensory Attribute | 1 | 0.00% |
| visual report | 1 | 0.00% |
| network management tool | 1 | 0.00% |
| Regime | 1 | 0.00% |
| Transportation Mode | 1 | 0.00% |
| Group/Band | 1 | 0.00% |
| website navigation | 1 | 0.00% |
| directionality | 1 | 0.00% |
| Service/Training Program | 1 | 0.00% |
| investment opportunity | 1 | 0.00% |
| investment guideline | 1 | 0.00% |
| Components | 1 | 0.00% |
| Fast Food Chain | 1 | 0.00% |
| Fast Food Item | 1 | 0.00% |
| clothing fastener | 1 | 0.00% |
| Authorized Heartland RV Dealer | 1 | 0.00% |
| Heartland RV Product | 1 | 0.00% |
| set of rules | 1 | 0.00% |
| flag retirement | 1 | 0.00% |
| honor and reverence | 1 | 0.00% |
| Model name | 1 | 0.00% |
| Ending | 1 | 0.00% |
| Object/Resource | 1 | 0.00% |
| Government/Political Entity | 1 | 0.00% |
| Technology/Computer Science | 1 | 0.00% |
| Technology/Photography | 1 | 0.00% |
| Educational Activity | 1 | 0.00% |
| Retail Activity | 1 | 0.00% |
| Drinking Establishment | 1 | 0.00% |
| military tactic | 1 | 0.00% |
| scheming | 1 | 0.00% |
| fictional character/group | 1 | 0.00% |
| Christian dating agency | 1 | 0.00% |
| Listing Identifier | 1 | 0.00% |
| Construction Object | 1 | 0.00% |
| recycling method | 1 | 0.00% |
| waste type | 1 | 0.00% |
| Sea Temple | 1 | 0.00% |
| Transport | 1 | 0.00% |
| tool/material | 1 | 0.00% |
| food/drink ingredient | 1 | 0.00% |
| Sunglasses | 1 | 0.00% |
| Candle | 1 | 0.00% |
| Ankle Boots | 1 | 0.00% |
| Human Offspring | 1 | 0.00% |
| computer network protocol | 1 | 0.00% |
| Real estate company | 1 | 0.00% |
| Music album series | 1 | 0.00% |
| job application document | 1 | 0.00% |
| biological organ | 1 | 0.00% |
| unpleasant fact | 1 | 0.00% |
| antidepressant | 1 | 0.00% |
| Wilderness Area | 1 | 0.00% |
| Year Establishment | 1 | 0.00% |
| cultural symbol | 1 | 0.00% |
| Dish/food | 1 | 0.00% |
| Document/Instructions | 1 | 0.00% |
| Sensation/Reaction | 1 | 0.00% |
| Statistic Entity | 1 | 0.00% |
| fishing technique | 1 | 0.00% |
| fishing equipment | 1 | 0.00% |
| Course name | 1 | 0.00% |
| Symposium name | 1 | 0.00% |
| Natural landmark | 1 | 0.00% |
| date of birth | 1 | 0.00% |
| body measurements | 1 | 0.00% |
| Cleaning activity | 1 | 0.00% |
| Power-saving feature | 1 | 0.00% |
| Power cord accessory | 1 | 0.00% |
| Sustainability practice | 1 | 0.00% |
| Reusable item | 1 | 0.00% |
| Metaphor for long-term use | 1 | 0.00% |
| people group | 1 | 0.00% |
| Programming | 1 | 0.00% |
| mental disorder | 1 | 0.00% |
| interpersonal relationships | 1 | 0.00% |
| degree of intensity | 1 | 0.00% |
| product_availability_prediction_id | 1 | 0.00% |
| store_id | 1 | 0.00% |
| confidence_level | 1 | 0.00% |
| prediction_date | 1 | 0.00% |
| Stakeholder engagement | 1 | 0.00% |
| Environmental regulation | 1 | 0.00% |
| Service Management | 1 | 0.00% |
| DNS Record Type | 1 | 0.00% |
| Account Management | 1 | 0.00% |
| productionLineId | 1 | 0.00% |
| productionLineName | 1 | 0.00% |
| shiftId | 1 | 0.00% |
| shiftName | 1 | 0.00% |
| overallEfficiency | 1 | 0.00% |
| Contract/Term | 1 | 0.00% |
| Military Force | 1 | 0.00% |
| traffic safety program | 1 | 0.00% |
| traffic safety concern | 1 | 0.00% |
| motorists | 1 | 0.00% |
| location/environment | 1 | 0.00% |
| precision | 1 | 0.00% |
| need | 1 | 0.00% |
| humor | 1 | 0.00% |
| fiction | 1 | 0.00% |
| caveat | 1 | 0.00% |
| discovery | 1 | 0.00% |
| Game Developer | 1 | 0.00% |
| compartment separator | 1 | 0.00% |
| storage compartment | 1 | 0.00% |
| ventilation material | 1 | 0.00% |
| Image resolution | 1 | 0.00% |
| Image compression | 1 | 0.00% |
| Grade Level | 1 | 0.00% |
| military outcome | 1 | 0.00% |
| visual attribute | 1 | 0.00% |
| vehicle rental | 1 | 0.00% |
| Military Weapon | 1 | 0.00% |
| Military Vehicle | 1 | 0.00% |
| Database Schema | 1 | 0.00% |
| Basic Step | 1 | 0.00% |
| Finish Type | 1 | 0.00% |
| Patriotism symbol | 1 | 0.00% |
| marketing tactic | 1 | 0.00% |
| Waste Treatment | 1 | 0.00% |
| Water Quality Management | 1 | 0.00% |
| Sewage Treatment | 1 | 0.00% |
| policy_type: Type of insurance policy renewed | 1 | 0.00% |
| policy_id: Unique identifier assigned to the policy | 1 | 0.00% |
| status update | 1 | 0.00% |
| negative sentiment | 1 | 0.00% |
| Medical facilities | 1 | 0.00% |
| Personality traits | 1 | 0.00% |
| Winch | 1 | 0.00% |
| Movie Name | 1 | 0.00% |
| Scene Type | 1 | 0.00% |
| Filming Process | 1 | 0.00% |
| scholarship_program.annual_cohort_size | 1 | 0.00% |
| scholarship_program.annual_graduation_size | 1 | 0.00% |
| scholarship_program.goal | 1 | 0.00% |
| scholarship_program.conferences.frequency | 1 | 0.00% |
| academic_performance.total_students | 1 | 0.00% |
| academic_performance.passing_grade_to_university | 1 | 0.00% |
| academic_performance.government_sponsored | 1 | 0.00% |
| support.donation_page | 1 | 0.00% |
| support.call_to_action | 1 | 0.00% |
| Naming convention | 1 | 0.00% |
| Phrase/Expression | 1 | 0.00% |
| Medical item | 1 | 0.00% |
| Play/Work of literature | 1 | 0.00% |
| calibrationReference | 1 | 0.00% |
| calibrationType | 1 | 0.00% |
| instrumentId | 1 | 0.00% |
| organizationId | 1 | 0.00% |
| calibrationAgency | 1 | 0.00% |
| calibrationProcedure | 1 | 0.00% |
| calibrator | 1 | 0.00% |
| contractual commitment | 1 | 0.00% |
| Fashion/Design | 1 | 0.00% |
| Environment/Setting | 1 | 0.00% |
| landscaping/horticulture | 1 | 0.00% |
| labelFormat | 1 | 0.00% |
| contentsExplanation | 1 | 0.00% |
| customsValueAmount | 1 | 0.00% |
| customsValueCurrency | 1 | 0.00% |
| countryOfOrigin | 1 | 0.00% |
| place of interest | 1 | 0.00% |
| service/technology | 1 | 0.00% |
| Land property | 1 | 0.00% |
| legal relationship | 1 | 0.00% |
| Event registration | 1 | 0.00% |
| Online form | 1 | 0.00% |
| medical personnel | 1 | 0.00% |
| Clothing/Jewelry Brand | 1 | 0.00% |
| Astrological sign | 1 | 0.00% |
| Radio station | 1 | 0.00% |
| Radio show | 1 | 0.00% |
| Financial Assistance Program | 1 | 0.00% |
| roomId | 1 | 0.00% |
| numNights | 1 | 0.00% |
| football strategy | 1 | 0.00% |
| Community/Recreation | 1 | 0.00% |
| Health Promoters | 1 | 0.00% |
| Guest | 1 | 0.00% |
| Podcast Topic | 1 | 0.00% |
| biological structure | 1 | 0.00% |
| biological property | 1 | 0.00% |
| Co-working group | 1 | 0.00% |
| Workspace cluster | 1 | 0.00% |
| Information sharing | 1 | 0.00% |
| Impromptu meetings | 1 | 0.00% |
| galaxy.type | 1 | 0.00% |
| galaxy.name | 1 | 0.00% |
| Land Use | 1 | 0.00% |
| Political Power | 1 | 0.00% |
| Direction/Location | 1 | 0.00% |
| chemical phenomena | 1 | 0.00% |
| environmental quality | 1 | 0.00% |
| complicated issues | 1 | 0.00% |
| Material Entity | 1 | 0.00% |
| Medical Device Entity | 1 | 0.00% |
| event or process | 1 | 0.00% |
| country or nation | 1 | 0.00% |
| movement or pattern | 1 | 0.00% |
| Music Industry Analytics Company | 1 | 0.00% |
| Trek itinerary | 1 | 0.00% |
| video game genre | 1 | 0.00% |
| Power Rating | 1 | 0.00% |
| Creative Individuals | 1 | 0.00% |
| Communication Material | 1 | 0.00% |
| Negative social group | 1 | 0.00% |
| Communication channel | 1 | 0.00% |
| Romantic interest | 1 | 0.00% |
| programming paradigm | 1 | 0.00% |
| programming construct | 1 | 0.00% |
| database construct | 1 | 0.00% |
| Field of study/industry | 1 | 0.00% |
| Company/brand | 1 | 0.00% |
| Entity type: Financial institution | 1 | 0.00% |
| Customer type | 1 | 0.00% |
| estimatedDeliveryDays | 1 | 0.00% |
| orderDate | 1 | 0.00% |
| orderStatus | 1 | 0.00% |
| price reduction | 1 | 0.00% |
| service improvement | 1 | 0.00% |
| Publication Platform | 1 | 0.00% |
| Long | 1 | 0.00% |
| Manufacturing equipment | 1 | 0.00% |
| Invalid Date | 1 | 0.00% |
| Streaming Platform | 1 | 0.00% |
| computer session | 1 | 0.00% |
| criminal organization | 1 | 0.00% |
| illegal trade | 1 | 0.00% |
| Massage Treatment | 1 | 0.00% |
| Accommodation/Building | 1 | 0.00% |
| Type of building | 1 | 0.00% |
| Tradition/Heritage/Culture | 1 | 0.00% |
| Fashion/Trend | 1 | 0.00% |
| cleaning product | 1 | 0.00% |
| mineral residue | 1 | 0.00% |
| Internet content | 1 | 0.00% |
| Capital city | 1 | 0.00% |
| Guide/Manual | 1 | 0.00% |
| Property features | 1 | 0.00% |
| Value enhancement | 1 | 0.00% |
| Process/Method | 1 | 0.00% |
| Educational Institution Type | 1 | 0.00% |
| Linguistic Theory | 1 | 0.00% |
| Linguistic Rules | 1 | 0.00% |
| edited | 1 | 0.00% |
| Board game | 1 | 0.00% |
| Crowdfunding platform | 1 | 0.00% |
| surfing spot | 1 | 0.00% |
| Football league | 1 | 0.00% |
| Sports terminology | 1 | 0.00% |
| Topic of Study | 1 | 0.00% |
| report_metadata.source_agency | 1 | 0.00% |
| company.financial_results.metrics.0.metric_name | 1 | 0.00% |
| company.financial_results.reporting_period | 1 | 0.00% |
| company.financial_results.metrics.0.historical_comparison.comparison_period | 1 | 0.00% |
| report_metadata.publication_date | 1 | 0.00% |
| Material/Thread | 1 | 0.00% |
| Tutorial/Instruction | 1 | 0.00% |
| Action/Technique | 1 | 0.00% |
| advice/wisdom | 1 | 0.00% |
| tone | 1 | 0.00% |
| Structural Feature | 1 | 0.00% |
| objectives | 1 | 0.00% |
| interpersonal communication | 1 | 0.00% |
| Wood Type | 1 | 0.00% |
| Product Brand | 1 | 0.00% |
| Behavior/Activity | 1 | 0.00% |
| stock exchange | 1 | 0.00% |
| Education and training program | 1 | 0.00% |
| Learning environment | 1 | 0.00% |
| Healthcare facilities | 1 | 0.00% |
| time/cost | 1 | 0.00% |
| Resource Library | 1 | 0.00% |
| Skill/Experience | 1 | 0.00% |
| Job title/Position | 1 | 0.00% |
| Technological platform | 1 | 0.00% |
| legal/ethical concern | 1 | 0.00% |
| Food Experience | 1 | 0.00% |
| workload | 1 | 0.00% |
| seafood product | 1 | 0.00% |
| payment instrument | 1 | 0.00% |
| Ethnicity/Heritage | 1 | 0.00% |
| Research Study | 1 | 0.00% |
| Pheasants Forever Chapter | 1 | 0.00% |
| Conservation Organization | 1 | 0.00% |
| vehicle condition | 1 | 0.00% |
| fraudulent activity | 1 | 0.00% |
| car parts | 1 | 0.00% |
| document fraud | 1 | 0.00% |
| Web hosting company | 1 | 0.00% |
| Website directory | 1 | 0.00% |
| Business Deal | 1 | 0.00% |
| Beer | 1 | 0.00% |
| Environmental testing device | 1 | 0.00% |
| body part/attribute | 1 | 0.00% |
| Medical specialty | 1 | 0.00% |
| Medical institution | 1 | 0.00% |
| animal healthcare provider | 1 | 0.00% |
| subscription service | 1 | 0.00% |
| furniture store | 1 | 0.00% |
| style preference | 1 | 0.00% |
| Calendar Date | 1 | 0.00% |
| Utility Service | 1 | 0.00% |
| Festival/Tradition | 1 | 0.00% |
| Pop culture characters | 1 | 0.00% |
| Vehicle Accessory | 1 | 0.00% |
| Vehicle Color/Model | 1 | 0.00% |
| Kitchenware item | 1 | 0.00% |
| game_id | 1 | 0.00% |
| steals | 1 | 0.00% |
| time and date | 1 | 0.00% |
| dose | 1 | 0.00% |
| vaccine | 1 | 0.00% |
| Product Benefit | 1 | 0.00% |
| Cloud computing | 1 | 0.00% |
| recipe type | 1 | 0.00% |
| recipe list | 1 | 0.00% |
| Dental Office | 1 | 0.00% |
| early_challenges | 1 | 0.00% |
| exit_status | 1 | 0.00% |
| founder_name | 1 | 0.00% |
| initial_funding | 1 | 0.00% |
| current_status | 1 | 0.00% |
| Resort Community | 1 | 0.00% |
| user_id | 1 | 0.00% |
| National Anthem | 1 | 0.00% |
| Opinion/Feedback | 1 | 0.00% |
| Engine type | 1 | 0.00% |
| Transmission type | 1 | 0.00% |
| fictional work | 1 | 0.00% |
| definition | 1 | 0.00% |
| Customers | 1 | 0.00% |
| Index | 1 | 0.00% |
| undefined | 1 | 0.00% |
| Person/Doctor | 1 | 0.00% |
| outbreak_event.investigation.findings.agent_details.abbreviation | 1 | 0.00% |
| outbreak_event.location.setting | 1 | 0.00% |
| outbreak_event.illness.pathogen.abbreviation | 1 | 0.00% |
| outbreak_event.investigation.specimen_source | 1 | 0.00% |
| outbreak_event.temporal_data.start_period | 1 | 0.00% |
| outbreak_event.investigation.findings.agent_details.full_name | 1 | 0.00% |
| outbreak_event.location.country | 1 | 0.00% |
| outbreak_event.illness.pathogen.classification | 1 | 0.00% |
| outbreak_event.illness.pathogen.full_name | 1 | 0.00% |
| Bridal Coach | 1 | 0.00% |
| educational technology | 1 | 0.00% |
| learning experience | 1 | 0.00% |
| Command line interface | 1 | 0.00% |
| File system | 1 | 0.00% |
| Software package | 1 | 0.00% |
| loyalty points | 1 | 0.00% |
| physical product | 1 | 0.00% |
| virtual currency | 1 | 0.00% |
| product inventory | 1 | 0.00% |
| channel_name | 1 | 0.00% |
| channel_url | 1 | 0.00% |
| thumbnail_style | 1 | 0.00% |
| Access Door | 1 | 0.00% |
| body measurement | 1 | 0.00% |
| consistency | 1 | 0.00% |
| brand message | 1 | 0.00% |
| visual communication | 1 | 0.00% |
| professionalism | 1 | 0.00% |
| brand image | 1 | 0.00% |
| calorie measurement | 1 | 0.00% |
| food company | 1 | 0.00% |
| Security Measures | 1 | 0.00% |
| Security Threats | 1 | 0.00% |
| music category | 1 | 0.00% |
| Industry/Market Sector | 1 | 0.00% |
| Financial Statement | 1 | 0.00% |
| Economic Condition | 1 | 0.00% |
| web page element | 1 | 0.00% |
| monthly_premium | 1 | 0.00% |
| issue_date | 1 | 0.00% |
| conclusions.overall_effectiveness | 1 | 0.00% |
| synthesis_context.carboxyl_protection_requirement.reasoning | 1 | 0.00% |
| deprotection_process.reagents.0.concentration_level | 1 | 0.00% |
| synthesis_context.intermediates_synthesized.0.components | 1 | 0.00% |
| deprotection_process.reagents.0.name | 1 | 0.00% |
| protecting_group.targets.0.protected_functional_group | 1 | 0.00% |
| protecting_group.targets.1.amino_acid_name | 1 | 0.00% |
| protecting_group.abbreviation | 1 | 0.00% |
| synthesis_context.methodology | 1 | 0.00% |
| protecting_group.name | 1 | 0.00% |
| protecting_group.targets.0.derivative_form | 1 | 0.00% |
| Beauty Treatment | 1 | 0.00% |
| Drawing Tutorial | 1 | 0.00% |
| Proper Noun | 1 | 0.00% |
| Common Noun | 1 | 0.00% |
| Warriors | 1 | 0.00% |
| Mythical Creatures | 1 | 0.00% |
| Property Value | 1 | 0.00% |
| Income Property | 1 | 0.00% |
| Legal framework | 1 | 0.00% |
| Technology maintenance | 1 | 0.00% |
| Listing Code | 1 | 0.00% |
| commercial item | 1 | 0.00% |
| Life Journey | 1 | 0.00% |
| salary_min | 1 | 0.00% |
| salary_max | 1 | 0.00% |
| Items | 1 | 0.00% |
| Automatic Evaluation Metric | 1 | 0.00% |
| suitable choice | 1 | 0.00% |
| requirements | 1 | 0.00% |
| Academic Term | 1 | 0.00% |
| Room Features | 1 | 0.00% |
| Building Features | 1 | 0.00% |
| Marketing Entity | 1 | 0.00% |
| Offspring | 1 | 0.00% |
| Animal dwelling | 1 | 0.00% |
| Fitness Centers | 1 | 0.00% |
| Control systems for central heating pumps | 1 | 0.00% |
| Central heating pumps | 1 | 0.00% |
| Publication/Journal | 1 | 0.00% |
| Research Methodology | 1 | 0.00% |
| Fiscal Policy | 1 | 0.00% |
| Commodity and Forex | 1 | 0.00% |
| Fixed Income | 1 | 0.00% |
| Long-term bonds | 1 | 0.00% |
| website rules | 1 | 0.00% |
| legal documents | 1 | 0.00% |
| Extracurricular Activity | 1 | 0.00% |
| agricultural input | 1 | 0.00% |
| environmental entity | 1 | 0.00% |
| Loyalty program | 1 | 0.00% |
| Forum/thread | 1 | 0.00% |
| Discussion participation | 1 | 0.00% |
| Data source | 1 | 0.00% |
| pickupTime | 1 | 0.00% |
| dropoffTime | 1 | 0.00% |
| Soccer team | 1 | 0.00% |
| Activity/Interest | 1 | 0.00% |
| Fitness App | 1 | 0.00% |
| Car audio system | 1 | 0.00% |
| isotope | 1 | 0.00% |
| natural satellite | 1 | 0.00% |
| clean protein | 1 | 0.00% |
| desire for sweets | 1 | 0.00% |
| cooking instructions | 1 | 0.00% |
| nutritional benefits | 1 | 0.00% |
| urgency | 1 | 0.00% |
| ticker | 1 | 0.00% |
| net_profit | 1 | 0.00% |
| fiscalyearend | 1 | 0.00% |
| headquarters | 1 | 0.00% |
| Cost of Treatment | 1 | 0.00% |
| Plant Extract | 1 | 0.00% |
| clothing measurement | 1 | 0.00% |
| Time/Decade | 1 | 0.00% |
| Sports group | 1 | 0.00% |
| Governance Issue | 1 | 0.00% |
| Personal Motto/Phrase | 1 | 0.00% |
| relationship issue | 1 | 0.00% |
| conflict resolution | 1 | 0.00% |
| people/investors | 1 | 0.00% |
| Retirement Age | 1 | 0.00% |
| Age criteria | 1 | 0.00% |
| Aspiration | 1 | 0.00% |
| Relationship Type | 1 | 0.00% |
| meal/event | 1 | 0.00% |
| Twitter account | 1 | 0.00% |
| Space mission | 1 | 0.00% |
| Profession/Group | 1 | 0.00% |
| Astronomical system | 1 | 0.00% |
| Informational Entity | 1 | 0.00% |
| Protein receptor | 1 | 0.00% |
| Marital status | 1 | 0.00% |
| Relationship issue | 1 | 0.00% |
| Emotional distress | 1 | 0.00% |
| receiptID | 1 | 0.00% |
| employeeID | 1 | 0.00% |
| Clothing attribute | 1 | 0.00% |
| review_source | 1 | 0.00% |
| Sensory phenomenon | 1 | 0.00% |
| Quantity indicator | 1 | 0.00% |
| phase | 1 | 0.00% |
| principalInvestigator | 1 | 0.00% |
| Group/Occupation | 1 | 0.00% |
| event/service | 1 | 0.00% |
| Honor | 1 | 0.00% |
| Subwoofer | 1 | 0.00% |
| num_reviews | 1 | 0.00% |
| review_sentiment | 1 | 0.00% |
| Chemical reaction | 1 | 0.00% |
| Unit | 1 | 0.00% |
| Miniature | 1 | 0.00% |
| Time/Experience | 1 | 0.00% |
| Mental Health Condition | 1 | 0.00% |
| market types | 1 | 0.00% |
| Event or Location | 1 | 0.00% |
| organization/member | 1 | 0.00% |
| document/event | 1 | 0.00% |
| Tourist Resort | 1 | 0.00% |
| camera brand | 1 | 0.00% |
| photography attribute | 1 | 0.00% |
| camera sensor type | 1 | 0.00% |
| camera body material | 1 | 0.00% |
| image file type | 1 | 0.00% |
| logical consequence | 1 | 0.00% |
| reward system | 1 | 0.00% |
| communication technique | 1 | 0.00% |
| nutrition | 1 | 0.00% |
| Play Title | 1 | 0.00% |
| Hospital ward | 1 | 0.00% |
| Patient type | 1 | 0.00% |
| Cooking equipment | 1 | 0.00% |
| Symbolism | 1 | 0.00% |
| Recipe Submission | 1 | 0.00% |
| Online Interaction | 1 | 0.00% |
| Medical Imaging Technology | 1 | 0.00% |
| Education Requirements | 1 | 0.00% |
| Medical Guidelines | 1 | 0.00% |
| aesthetic feature/material | 1 | 0.00% |
| artificial entity | 1 | 0.00% |
| disease.subspecies.2.affected_species | 1 | 0.00% |
| disease.subspecies.1.name | 1 | 0.00% |
| disease.name | 1 | 0.00% |
| disease.subspecies.0.name | 1 | 0.00% |
| disease.subspecies.2.name | 1 | 0.00% |
| grant_number | 1 | 0.00% |
| sensitivity | 1 | 0.00% |
| list of volunteers | 1 | 0.00% |
| food service | 1 | 0.00% |
| Proper noun | 1 | 0.00% |
| Sound | 1 | 0.00% |
| dining venue | 1 | 0.00% |
| economic activity | 1 | 0.00% |
| Leadership trait | 1 | 0.00% |
| Visual representation | 1 | 0.00% |
| Professional role | 1 | 0.00% |
| appliance part | 1 | 0.00% |
| appliance repair service | 1 | 0.00% |
| file/document | 1 | 0.00% |
| camera | 1 | 0.00% |
| composting partners | 1 | 0.00% |
| composting materials | 1 | 0.00% |
| Character Trait | 1 | 0.00% |
| Medical degree | 1 | 0.00% |
| transport | 1 | 0.00% |
| Data encoding | 1 | 0.00% |
| Non-renewable resources | 1 | 0.00% |
| Environmental impact | 1 | 0.00% |
| Category Entity | 1 | 0.00% |
| Technology Entity | 1 | 0.00% |
| Blog Entity | 1 | 0.00% |
| Technology/software | 1 | 0.00% |
| young person | 1 | 0.00% |
| grown-ups | 1 | 0.00% |
| Project cost | 1 | 0.00% |
| Building wing | 1 | 0.00% |
| Building side | 1 | 0.00% |
| user guide | 1 | 0.00% |
| Accommodation<> A building or group of buildings where people can live, usually temporarily | 1 | 0.00% |
| Measurement<> A numerical value that rates the walkability of a location based on the distance to nearby amenities | 1 | 0.00% |
| Measurement<> A score out of 100 indicating the walkability of a location | 1 | 0.00% |
| Rating<> A description indicating that the location has excellent walkability and most errands can be accomplished on foot | 1 | 0.00% |
| Location<> A city in Saskatchewan, Canada | 1 | 0.00% |
| module function | 1 | 0.00% |
| investment | 1 | 0.00% |
| meeting_code | 1 | 0.00% |
| Poker Move | 1 | 0.00% |
| Type of Business | 1 | 0.00% |
| Place of Business | 1 | 0.00% |
| negative event | 1 | 0.00% |
| preventable situation | 1 | 0.00% |
| lack of responsibility | 1 | 0.00% |
| dangerous place | 1 | 0.00% |
| hotel_experience.description | 1 | 0.00% |
| hotel_experience.location.nearby_landmark | 1 | 0.00% |
| hotel_experience.location.city | 1 | 0.00% |
| hotel_experience.location.postal_code_accessibility | 1 | 0.00% |
| hotel_experience.suites.0.bathroom_features | 1 | 0.00% |
| hotel_experience.suites.0.workspace | 1 | 0.00% |
| hotel_experience.suites.0.suite_type | 1 | 0.00% |
| hotel_experience.suites.0.suite_count | 1 | 0.00% |
| hotel_experience.suites.0.features | 1 | 0.00% |
| hotel_experience.services | 1 | 0.00% |
| hotel_experience.entertainment.tv.screen_size | 1 | 0.00% |
| hotel_experience.entertainment.tv.content | 1 | 0.00% |
| Music Band/Group | 1 | 0.00% |
| Band name | 1 | 0.00% |
| Music festival | 1 | 0.00% |
| literary technique | 1 | 0.00% |
| software documentation | 1 | 0.00% |
| internet address | 1 | 0.00% |
| date/year | 1 | 0.00% |
| packaging type | 1 | 0.00% |
| comparison/reference | 1 | 0.00% |
| Leisure | 1 | 0.00% |
| Socializing | 1 | 0.00% |
| Completion | 1 | 0.00% |
| Video Game Developer | 1 | 0.00% |
| Action Entity | 1 | 0.00% |
| Housing Design | 1 | 0.00% |
| Cooking Space | 1 | 0.00% |
| Sleeping Space | 1 | 0.00% |
| Printing technique | 1 | 0.00% |
| Print size | 1 | 0.00% |
| Print edition | 1 | 0.00% |
| publishing program | 1 | 0.00% |
| audience/community | 1 | 0.00% |
| Short Story | 1 | 0.00% |
| Writing Contest | 1 | 0.00% |
| Theme park | 1 | 0.00% |
| feedback_id | 1 | 0.00% |
| overall_rating | 1 | 0.00% |
| provider_id | 1 | 0.00% |
| Currency in words | 1 | 0.00% |
| function signature | 1 | 0.00% |
| code optimization | 1 | 0.00% |
| memory address | 1 | 0.00% |
| code line number | 1 | 0.00% |
| error trace | 1 | 0.00% |
| imaging_findings | 1 | 0.00% |
| plot element | 1 | 0.00% |
| delivery method | 1 | 0.00% |
| film format | 1 | 0.00% |
| activity/type | 1 | 0.00% |
| Product/Project | 1 | 0.00% |
| Emotion/Attitude | 1 | 0.00% |
| Research Project | 1 | 0.00% |
| Information Resource | 1 | 0.00% |
| Instructional Resource | 1 | 0.00% |
| Beer brand | 1 | 0.00% |
| Service/Entertainment | 1 | 0.00% |
| academic level | 1 | 0.00% |
| water park | 1 | 0.00% |
| resort | 1 | 0.00% |
| Persuasion | 1 | 0.00% |
| Value | 1 | 0.00% |
| Hiking trail | 1 | 0.00% |
| Sound/Music | 1 | 0.00% |
| Appearance/Physical Feature | 1 | 0.00% |
| Landscape/Environment | 1 | 0.00% |
| music artist/band | 1 | 0.00% |
| music element | 1 | 0.00% |
| performance style | 1 | 0.00% |
| certification program | 1 | 0.00% |
| learning style | 1 | 0.00% |
| Priority Access | 1 | 0.00% |
| Emotion/Behavior | 1 | 0.00% |
| Action/Function | 1 | 0.00% |
| Personal brand | 1 | 0.00% |
| strategy/plan | 1 | 0.00% |
| action/ability | 1 | 0.00% |
| tangible item | 1 | 0.00% |
| Web Page Element | 1 | 0.00% |
| educational environment | 1 | 0.00% |
| social media action | 1 | 0.00% |
| Electrical Equipment | 1 | 0.00% |
| HVAC Equipment | 1 | 0.00% |
| Fuel | 1 | 0.00% |
| purchase | 1 | 0.00% |
| Music Album Collection | 1 | 0.00% |
| storage space | 1 | 0.00% |
| Bus Route | 1 | 0.00% |
| job search website | 1 | 0.00% |
| job listings | 1 | 0.00% |
| career document storage | 1 | 0.00% |
| professional development resources | 1 | 0.00% |
| childhood memory | 1 | 0.00% |
| work experience | 1 | 0.00% |
| personal objective | 1 | 0.00% |
| Internet Security | 1 | 0.00% |
| Importance | 1 | 0.00% |
| Postpartum Care Product | 1 | 0.00% |
| packaging style | 1 | 0.00% |
| packaging quality | 1 | 0.00% |
| packaging impact | 1 | 0.00% |
| brand aesthetics | 1 | 0.00% |
| film type | 1 | 0.00% |
| School/Team | 1 | 0.00% |
| Business operation | 1 | 0.00% |
| Other things | 1 | 0.00% |
| Effectiveness | 1 | 0.00% |
| Project Management | 1 | 0.00% |
| Collaboration Platform | 1 | 0.00% |
| Domain Configuration | 1 | 0.00% |
| Blogging Platform | 1 | 0.00% |
| commercial enterprise | 1 | 0.00% |
| room feature | 1 | 0.00% |
| Holiday/Celebration | 1 | 0.00% |
| Dish Type | 1 | 0.00% |
| Skill/Task | 1 | 0.00% |
| Relationship Partner | 1 | 0.00% |
| Global Community | 1 | 0.00% |
| Digital Platform Administrator | 1 | 0.00% |
| Data Management | 1 | 0.00% |
| Business Development | 1 | 0.00% |
| Business Sectors | 1 | 0.00% |
| Sales Prospects | 1 | 0.00% |
| Sales and Marketing | 1 | 0.00% |
| Meal Occasion | 1 | 0.00% |
| services_offered | 1 | 0.00% |
| Furniture Design | 1 | 0.00% |
| family group | 1 | 0.00% |
| legal group | 1 | 0.00% |
| Domain Name System (DNS) | 1 | 0.00% |
| Instructor/Author | 1 | 0.00% |
| name: Full name of the LinkedIn product | 1 | 0.00% |
| product_id: Unique identifier assigned to the product | 1 | 0.00% |
| monthly_price: Subscription cost per month in USD | 1 | 0.00% |
| target_audience: Group of users or professionals it serves | 1 | 0.00% |
| features: Key functionalities or tools included | 1 | 0.00% |
| pricing_tier: Subscription level such as monthly or free | 1 | 0.00% |
| News Category | 1 | 0.00% |
| Review Category | 1 | 0.00% |
| Image Category | 1 | 0.00% |
| Video Category | 1 | 0.00% |
| Game manipulation | 1 | 0.00% |
| asteroid.discovery.observatory.location.city | 1 | 0.00% |
| asteroid.discovery.observatory.location.distance_km | 1 | 0.00% |
| asteroid.named_after | 1 | 0.00% |
| bull statant | 1 | 0.00% |
| lantern sable | 1 | 0.00% |
| chief | 1 | 0.00% |
| Slang term | 1 | 0.00% |
| Business venture | 1 | 0.00% |
| Earthing Mat Design | 1 | 0.00% |
| Electrical Engineering Portal | 1 | 0.00% |
| Substation | 1 | 0.00% |
| Grid | 1 | 0.00% |
| Earth Mat Design Software | 1 | 0.00% |
| campaign_id | 1 | 0.00% |
| group/collective | 1 | 0.00% |
| Film Format | 1 | 0.00% |
| Photo Size | 1 | 0.00% |
| Camera Type | 1 | 0.00% |
| Camera Material | 1 | 0.00% |
| Game/App | 1 | 0.00% |
| data protection | 1 | 0.00% |
| network users | 1 | 0.00% |
| network type | 1 | 0.00% |
| Crime Group | 1 | 0.00% |
| distributor_information.authorization | 1 | 0.00% |
| distributor_information.capabilities | 1 | 0.00% |
| services.maintenance | 1 | 0.00% |
| Business Venue | 1 | 0.00% |
| device_id: Unique identifier assigned to the product | 1 | 0.00% |
| retail_price: Final price at retail in US dollars | 1 | 0.00% |
| battery_life: Duration of battery operation in days | 1 | 0.00% |
| Hair Color Brand | 1 | 0.00% |
| Healthcare | 1 | 0.00% |
| Date/Day | 1 | 0.00% |
| Cuisine/Category | 1 | 0.00% |
| Taste/Attribute | 1 | 0.00% |
| Eating Technique/Category | 1 | 0.00% |
| Experience/Action | 1 | 0.00% |
| Job Search Platform | 1 | 0.00% |
| Job Placement Services | 1 | 0.00% |
| Bar/club name | 1 | 0.00% |
| Drink special | 1 | 0.00% |
| Talent | 1 | 0.00% |
| guidelines | 1 | 0.00% |
| user input | 1 | 0.00% |
| targetQuantity | 1 | 0.00% |
| shiftDate | 1 | 0.00% |
| totalProducedQuantity | 1 | 0.00% |
| acceptedQuantity | 1 | 0.00% |
| rejectedQuantity | 1 | 0.00% |
| Law firm | 1 | 0.00% |
| severity: Level of reaction intensity | 1 | 0.00% |
| address: Complete address of the patient | 1 | 0.00% |
| date: Date when the reaction was reported | 1 | 0.00% |
| dosage: Prescribed dosage and frequency of the drug | 1 | 0.00% |
| patient_id: Unique identifier for the patient in the record | 1 | 0.00% |
| duration: Number of days the treatment lasted for the reaction | 1 | 0.00% |
| status: Current status of the reaction | 1 | 0.00% |
| personal history | 1 | 0.00% |
| Atmosphere/Environment | 1 | 0.00% |
| species.genus | 1 | 0.00% |
| Marvel villain | 1 | 0.00% |
| Consumer/Target Audience | 1 | 0.00% |
| Attribute/Price | 1 | 0.00% |
| Attribute/Budget | 1 | 0.00% |
| Spa Treatment | 1 | 0.00% |
| Ski Trails | 1 | 0.00% |
| logical storage division | 1 | 0.00% |
| product_example.size | 1 | 0.00% |
| product_example.currency | 1 | 0.00% |
| product_example.color | 1 | 0.00% |
| availability.locations.retail_outlets | 1 | 0.00% |
| distributor.location | 1 | 0.00% |
| availability.locations.online_stores.0.location | 1 | 0.00% |
| product_usage | 1 | 0.00% |
| product_example.name | 1 | 0.00% |
| availability.locations.online_stores.0.url | 1 | 0.00% |
| Oyster type | 1 | 0.00% |
| Diagnostic procedure | 1 | 0.00% |
| Action verb | 1 | 0.00% |
| First-year Information Technology students | 1 | 0.00% |
| Networking event | 1 | 0.00% |
| Capital City | 1 | 0.00% |
| Medical solution | 1 | 0.00% |
| patient_id: Unique identifier assigned to the vaccination record | 1 | 0.00% |
| address: Residential address of the patient | 1 | 0.00% |
| status: Current status of the vaccination series | 1 | 0.00% |
| event/occasion | 1 | 0.00% |
| product/item | 1 | 0.00% |
| Technology Incubator | 1 | 0.00% |
| Organization/Website | 1 | 0.00% |
| authenticity | 1 | 0.00% |
| affordability | 1 | 0.00% |
| Personal Trait/Characteristic | 1 | 0.00% |
| fruit family | 1 | 0.00% |
| ripeness level | 1 | 0.00% |
| Academic program | 1 | 0.00% |
| fashion/accessory | 1 | 0.00% |
| industry/type of work | 1 | 0.00% |
| body part/anatomy | 1 | 0.00% |
| property_name | 1 | 0.00% |
| leasestartdate | 1 | 0.00% |
| Football Coach | 1 | 0.00% |
| healthcare provider | 1 | 0.00% |
| healthcare network | 1 | 0.00% |
| home decoration | 1 | 0.00% |
| Website Platform | 1 | 0.00% |
| Website Theme | 1 | 0.00% |
| Property Quality | 1 | 0.00% |
| Deadline | 1 | 0.00% |
| Location/Position | 1 | 0.00% |
| Credential/Recognition | 1 | 0.00% |
| Selection/Evaluation | 1 | 0.00% |
| Fan Community | 1 | 0.00% |
| Subject Matter | 1 | 0.00% |
| Fictional Characters | 1 | 0.00% |
| Fictional Universe | 1 | 0.00% |
| Medical technology | 1 | 0.00% |
| Business park | 1 | 0.00% |
| institution: Hospital or organization where author is affiliated | 1 | 0.00% |
| recommendation: Specific treatment or intervention advised | 1 | 0.00% |
| issue_year: Year the guideline was published or released | 1 | 0.00% |
| Service/Insurance | 1 | 0.00% |
| input type | 1 | 0.00% |
| Mitigation Strategy | 1 | 0.00% |
| scheduling tool | 1 | 0.00% |
| color palette | 1 | 0.00% |
| material finish | 1 | 0.00% |
| fastener type | 1 | 0.00% |
| game state | 1 | 0.00% |
| Game Ability | 1 | 0.00% |
| Game Resource | 1 | 0.00% |
| Person/Child | 1 | 0.00% |
| current state | 1 | 0.00% |
| Criminal group | 1 | 0.00% |
| Numerical quantity | 1 | 0.00% |
| potion | 1 | 0.00% |
| variation of object | 1 | 0.00% |
| race | 1 | 0.00% |
| practitionerId | 1 | 0.00% |
| visitId | 1 | 0.00% |
| sideEffects | 1 | 0.00% |
| contraindications | 1 | 0.00% |
| conditionId | 1 | 0.00% |
| ICD10Code | 1 | 0.00% |
| dateDiagnosed | 1 | 0.00% |
| Switch mode | 1 | 0.00% |
| Networking Hardware | 1 | 0.00% |
| Networking Topology | 1 | 0.00% |
| Networking Protocol | 1 | 0.00% |
| Exam Result | 1 | 0.00% |
| Exam Criteria | 1 | 0.00% |
| Exam Ranking | 1 | 0.00% |
| Game Type | 1 | 0.00% |
| Image Variety | 1 | 0.00% |
| Difficulty Level | 1 | 0.00% |
| Product Effect | 1 | 0.00% |
| Product Ownership | 1 | 0.00% |
| colors | 1 | 0.00% |
| Technology/Field | 1 | 0.00% |
| Conductor/Performer | 1 | 0.00% |
| Mezzo-soprano/Singer | 1 | 0.00% |
| Retail staff | 1 | 0.00% |
| Fashion tips | 1 | 0.00% |
| Customer experience | 1 | 0.00% |
| Generation/Population | 1 | 0.00% |
| Personality Trait/Characteristic | 1 | 0.00% |
| TV Show Episode/Entertainment | 1 | 0.00% |
| running routes | 1 | 0.00% |
| fitness programs | 1 | 0.00% |
| running community | 1 | 0.00% |
| Cell Type | 1 | 0.00% |
| Animal Model | 1 | 0.00% |
| Fashion line | 1 | 0.00% |
| Themed launch | 1 | 0.00% |
| ride | 1 | 0.00% |
| Business Model | 1 | 0.00% |
| marketing campaign | 1 | 0.00% |
| participants | 1 | 0.00% |
| mental ability | 1 | 0.00% |
| Academic Subjects | 1 | 0.00% |
| Examination | 1 | 0.00% |
| Employment opportunity | 1 | 0.00% |
| Volume/Issue | 1 | 0.00% |
| type of food | 1 | 0.00% |
| exercise type | 1 | 0.00% |
| Housing model | 1 | 0.00% |
| environmental feature | 1 | 0.00% |
| botanical feature | 1 | 0.00% |
| water quality | 1 | 0.00% |
| Retail Category | 1 | 0.00% |
| Group of animals | 1 | 0.00% |
| personal commitment | 1 | 0.00% |
| communication software | 1 | 0.00% |
| video display setting | 1 | 0.00% |
| computer window setting | 1 | 0.00% |
| appos | 1 | 0.00% |
| advmod | 1 | 0.00% |
| auxpass | 1 | 0.00% |
| acl | 1 | 0.00% |
| injury type | 1 | 0.00% |
| Education phase | 1 | 0.00% |
| Skill development | 1 | 0.00% |
| Study technique | 1 | 0.00% |
| Learning difficulties | 1 | 0.00% |
| Puzzle | 1 | 0.00% |
| Visual puzzles | 1 | 0.00% |
| Material/Chemical | 1 | 0.00% |
| Artist/Sculptor | 1 | 0.00% |
| Character/Person | 1 | 0.00% |
| Ability/Power | 1 | 0.00% |
| Installation type | 1 | 0.00% |
| General Group of People | 1 | 0.00% |
| Online Followers | 1 | 0.00% |
| Financial Stability | 1 | 0.00% |
| Online Communities | 1 | 0.00% |
| application_link | 1 | 0.00% |
| job_id | 1 | 0.00% |
| marketing/advertising | 1 | 0.00% |
| middleName | 1 | 0.00% |
| suffix | 1 | 0.00% |
| tableware item | 1 | 0.00% |
| tableware variety | 1 | 0.00% |
| webpage element | 1 | 0.00% |
| series | 1 | 0.00% |
| fashion blog | 1 | 0.00% |
| clothing line | 1 | 0.00% |
| Academic Division | 1 | 0.00% |
| Academic Event | 1 | 0.00% |
| Academic Building | 1 | 0.00% |
| Wall Decor | 1 | 0.00% |
| Quest type | 1 | 0.00% |
| company/industry | 1 | 0.00% |
| store_location | 1 | 0.00% |
| technological system | 1 | 0.00% |
| Year of release | 1 | 0.00% |
| sailing yacht | 1 | 0.00% |
| Origin | 1 | 0.00% |
| Game or Puzzle | 1 | 0.00% |
| Idiom or Expression | 1 | 0.00% |
| caregivers | 1 | 0.00% |
| talents | 1 | 0.00% |
| Cooking utensil | 1 | 0.00% |
| Course/subject | 1 | 0.00% |
| hardware devices | 1 | 0.00% |
| answers | 1 | 0.00% |
| Cultural Community | 1 | 0.00% |
| language translation service | 1 | 0.00% |
| decision-making factors | 1 | 0.00% |
| content plan | 1 | 0.00% |
| Legal Profession | 1 | 0.00% |
| small farm | 1 | 0.00% |
| chef | 1 | 0.00% |
| artisan | 1 | 0.00% |
| school district | 1 | 0.00% |
| disinfectant solution | 1 | 0.00% |
| dispenser | 1 | 0.00% |
| chemical entity | 1 | 0.00% |
| Human Beings | 1 | 0.00% |
| Reproduction | 1 | 0.00% |
| Supernatural Entities | 1 | 0.00% |
| Social Attribute | 1 | 0.00% |
| hard drive model | 1 | 0.00% |
| hard drive manufacturer | 1 | 0.00% |
| Command Line Parameter | 1 | 0.00% |
| Software Porting | 1 | 0.00% |
| location: City and state where the hotel is located | 1 | 0.00% |
| hotel_name: Name of the hotel as mentioned in the review | 1 | 0.00% |
| rating: Star rating from 1 to 5 given by the reviewer | 1 | 0.00% |
| Contact Lens Company | 1 | 0.00% |
| Contact Lens Type | 1 | 0.00% |
| Environmental Feature | 1 | 0.00% |
| person.expertise | 1 | 0.00% |
| person.media_appearances.0.content_description | 1 | 0.00% |
| search_context.topic | 1 | 0.00% |
| person.title | 1 | 0.00% |
| person.media_appearances.0.type | 1 | 0.00% |
| person.description | 1 | 0.00% |
| science | 1 | 0.00% |
| chemistry | 1 | 0.00% |
| Professional Wrestler | 1 | 0.00% |
| Wrestling Historian | 1 | 0.00% |
| Wrestling Advocacy Group | 1 | 0.00% |
| Tool Set | 1 | 0.00% |
| Food crop | 1 | 0.00% |
| cultural diversity | 1 | 0.00% |
| discourse | 1 | 0.00% |
| persuasive language | 1 | 0.00% |
| Slot machine | 1 | 0.00% |
| color/nature | 1 | 0.00% |
| article.id | 1 | 0.00% |
| article.cause | 1 | 0.00% |
| article.resolution | 1 | 0.00% |
| article.references.0.id | 1 | 0.00% |
| article.references.0.url | 1 | 0.00% |
| article.references.0.description | 1 | 0.00% |
| article.last_reviewed | 1 | 0.00% |
| article.version | 1 | 0.00% |
| article.translation_note.disclaimer | 1 | 0.00% |
| article.translation_note.english_version.id | 1 | 0.00% |
| article.translation_note.english_version.url | 1 | 0.00% |
| article.disclaimer | 1 | 0.00% |
| trade strategy | 1 | 0.00% |
| forex trading platform | 1 | 0.00% |
| Acting | 1 | 0.00% |
| recipientName | 1 | 0.00% |
| recipientEmail | 1 | 0.00% |
| senderName | 1 | 0.00% |
| senderEmail | 1 | 0.00% |
| message | 1 | 0.00% |
| Law Enforcement Officer | 1 | 0.00% |
| Violence event | 1 | 0.00% |
| AI Assistant | 1 | 0.00% |
| Procedure/Task | 1 | 0.00% |
| User Activity | 1 | 0.00% |
| Biblical passage | 1 | 0.00% |
| ammunition | 1 | 0.00% |
| competitions | 1 | 0.00% |
| numerical quantity | 1 | 0.00% |
| shooting technique | 1 | 0.00% |
| shooting location | 1 | 0.00% |
| target type | 1 | 0.00% |
| descriptive term | 1 | 0.00% |
| achievement goal | 1 | 0.00% |
| social success | 1 | 0.00% |
| health complication | 1 | 0.00% |
| anatomical feature | 1 | 0.00% |
| Coffee Brewing Method | 1 | 0.00% |
| Coffee Brand | 1 | 0.00% |
| account information | 1 | 0.00% |
| online tool | 1 | 0.00% |
| road surface | 1 | 0.00% |
| value increase | 1 | 0.00% |
| reference point | 1 | 0.00% |
| non-profit organization support | 1 | 0.00% |
| regulatory_status.recognition_type | 1 | 0.00% |
| announcement_metadata.date | 1 | 0.00% |
| tax_incentive.duration.period.end_year | 1 | 0.00% |
| regulatory_status.review_status | 1 | 0.00% |
| announcement_metadata.source_url | 1 | 0.00% |
| tax_incentive.duration.period.start_year | 1 | 0.00% |
| ratingCount | 1 | 0.00% |
| The Worst-Hit Area | 1 | 0.00% |
| Coordinate | 1 | 0.00% |
| Toy/Model | 1 | 0.00% |
| Role/Job Title | 1 | 0.00% |
| Chemical Composition | 1 | 0.00% |
| Exporter | 1 | 0.00% |
| Sales Team | 1 | 0.00% |
| Cash Machine | 1 | 0.00% |
| wastewater treatment | 1 | 0.00% |
| water management | 1 | 0.00% |
| water treatment equipment | 1 | 0.00% |
| wastewater component | 1 | 0.00% |
| school term | 1 | 0.00% |
| governing body | 1 | 0.00% |
| Treatment Solution | 1 | 0.00% |
| instruction manual | 1 | 0.00% |
| installation equipment | 1 | 0.00% |
| Toy/Entertainment | 1 | 0.00% |
| Liquid | 1 | 0.00% |
| Military camp | 1 | 0.00% |
| betting company | 1 | 0.00% |
| customer experience | 1 | 0.00% |
| Monetary Prize | 1 | 0.00% |
| advertisement intent | 1 | 0.00% |
| Composition | 1 | 0.00% |
| Tool/Machine | 1 | 0.00% |
| person with strong emotional attachment | 1 | 0.00% |
| emotional and/or physical assistance | 1 | 0.00% |
| Defense alliance | 1 | 0.00% |
| People/Occupation | 1 | 0.00% |
| tool/equipment | 1 | 0.00% |
| tool/equipment part | 1 | 0.00% |
| Legal procedure | 1 | 0.00% |
| job level | 1 | 0.00% |
| experience level | 1 | 0.00% |
| compensation | 1 | 0.00% |
| printing | 1 | 0.00% |
| machine/tool | 1 | 0.00% |
| risk_score | 1 | 0.00% |
| risk_category | 1 | 0.00% |
| coverage_limit | 1 | 0.00% |
| underwriter | 1 | 0.00% |
| system_code | 1 | 0.00% |
| music event | 1 | 0.00% |
| gathering of people | 1 | 0.00% |
| physical object/ability | 1 | 0.00% |
| Skill or expertise | 1 | 0.00% |
| Control Method | 1 | 0.00% |
| tax return | 1 | 0.00% |
| tax knowledge | 1 | 0.00% |
| tax schedule | 1 | 0.00% |
| Power output | 1 | 0.00% |
| Engine model | 1 | 0.00% |
| Regulatory body | 1 | 0.00% |
| Location type | 1 | 0.00% |
| Technology Device | 1 | 0.00% |
| Pay it forward program | 1 | 0.00% |
| Philanthropist | 1 | 0.00% |
| error/bug | 1 | 0.00% |
| software/application | 1 | 0.00% |
| academic process | 1 | 0.00% |
| academic assessment | 1 | 0.00% |
| botanical entity | 1 | 0.00% |
| review_status.message | 1 | 0.00% |
| Musical Artist/Band | 1 | 0.00% |
| Time off | 1 | 0.00% |
| group of living beings | 1 | 0.00% |
| texture/quality | 1 | 0.00% |
| Opinion/Preference | 1 | 0.00% |
| flower type | 1 | 0.00% |
| service user group | 1 | 0.00% |
| general appointment | 1 | 0.00% |
| vacation activity | 1 | 0.00% |
| Betting website | 1 | 0.00% |
| Resin Cement | 1 | 0.00% |
| Task Management | 1 | 0.00% |
| Sports rules | 1 | 0.00% |
| creative space | 1 | 0.00% |
| art exhibition space | 1 | 0.00% |
| industrial building | 1 | 0.00% |
| research and development | 1 | 0.00% |
| healthcare recipients | 1 | 0.00% |
| tax type | 1 | 0.00% |
| Center | 1 | 0.00% |
| Center of Excellence | 1 | 0.00% |
| marketing opportunity | 1 | 0.00% |
| football tactic | 1 | 0.00% |
| Carrier | 1 | 0.00% |
| Caregiver | 1 | 0.00% |
| Social Service Entity | 1 | 0.00% |
| sexual orientation | 1 | 0.00% |
| government office | 1 | 0.00% |
| App | 1 | 0.00% |
| Product/Transportation | 1 | 0.00% |
| Historical record | 1 | 0.00% |
| Family relationship | 1 | 0.00% |
| Life status | 1 | 0.00% |
| legume | 1 | 0.00% |
| Biology | 1 | 0.00% |
| Travel type | 1 | 0.00% |
| Linguistic Element | 1 | 0.00% |
| Alert | 1 | 0.00% |
| Album Title | 1 | 0.00% |
| motorsport event | 1 | 0.00% |
| fitness facilities | 1 | 0.00% |
| Maintenance Service | 1 | 0.00% |
| health recovery | 1 | 0.00% |
| Award recipients | 1 | 0.00% |
| construction project | 1 | 0.00% |
| kitchen feature | 1 | 0.00% |
| outdoor feature | 1 | 0.00% |
| Plan | 1 | 0.00% |
| Fish Type | 1 | 0.00% |
| Garnish | 1 | 0.00% |
| Side Dish | 1 | 0.00% |
| regulation/law | 1 | 0.00% |
| Business service | 1 | 0.00% |
| Maintenance Task | 1 | 0.00% |
| Daily Habit | 1 | 0.00% |
| Publication/Website | 1 | 0.00% |
| Professional Title | 1 | 0.00% |
| item/request | 1 | 0.00% |
| Computer storage device | 1 | 0.00% |
| Product manual | 1 | 0.00% |
| writing event | 1 | 0.00% |
| Slang Term | 1 | 0.00% |
| Jersey Number | 1 | 0.00% |
| Date/Day of the Week | 1 | 0.00% |
| Moving Company | 1 | 0.00% |
| publication.title | 1 | 0.00% |
| publication.research_context.time_period | 1 | 0.00% |
| publication.research_context.informants.description | 1 | 0.00% |
| publication.language.name | 1 | 0.00% |
| publication.language.classification | 1 | 0.00% |
| publication.language.region | 1 | 0.00% |
| content_analysis.total_pieces | 1 | 0.00% |
| linguistic_framework.reference_scholar | 1 | 0.00% |
| Platform/Social Media | 1 | 0.00% |
| Sports outcome | 1 | 0.00% |
| Food truck | 1 | 0.00% |
| Food group | 1 | 0.00% |
| dynasty | 1 | 0.00% |
| effect/action | 1 | 0.00% |
| tools/equipment | 1 | 0.00% |
| Battery technology | 1 | 0.00% |
| stock market | 1 | 0.00% |
| purpose/functionality | 1 | 0.00% |
| Confidential Information | 1 | 0.00% |
| Emotional Support | 1 | 0.00% |
| Advice & Guidance | 1 | 0.00% |
| Spiritual Beliefs | 1 | 0.00% |
| Power Plant | 1 | 0.00% |
| service/infrastructure | 1 | 0.00% |
| musical instrument brand | 1 | 0.00% |
| agreementId | 1 | 0.00% |
| additionalTerms | 1 | 0.00% |
| canton_name_french | 1 | 0.00% |
| Research field | 1 | 0.00% |
| Bike Brand | 1 | 0.00% |
| Struggle | 1 | 0.00% |
| Psychological state | 1 | 0.00% |
| Event/Meeting | 1 | 0.00% |
| from | 1 | 0.00% |
| to | 1 | 0.00% |
| design service | 1 | 0.00% |
| Course Name | 1 | 0.00% |
| data visualization | 1 | 0.00% |
| user data | 1 | 0.00% |
| user proficiency | 1 | 0.00% |
| Title/Work | 1 | 0.00% |
| Time/Year | 1 | 0.00% |
| Prefix crossword clue | 1 | 0.00% |
| Crossword game publication | 1 | 0.00% |
| Crossword game term | 1 | 0.00% |
| health and wellness | 1 | 0.00% |
| education and development | 1 | 0.00% |
| psychology | 1 | 0.00% |
| scientific_description.described_by | 1 | 0.00% |
| vulnerability | 1 | 0.00% |
| Opposition group | 1 | 0.00% |
| Administrative Unit | 1 | 0.00% |
| Biometric sensor | 1 | 0.00% |
| Body dimension | 1 | 0.00% |
| Financial transaction | 1 | 0.00% |
| interactive content | 1 | 0.00% |
| Fraction/Quantity | 1 | 0.00% |
| Data Volume | 1 | 0.00% |
| Month/Time | 1 | 0.00% |
| make | 1 | 0.00% |
| licensePlateNumber | 1 | 0.00% |
| additionalInstructions | 1 | 0.00% |
| cardNumber | 1 | 0.00% |
| cvv | 1 | 0.00% |
| Stock Index | 1 | 0.00% |
| fruit_properties.size | 1 | 0.00% |
| fruit_properties.sugar_content | 1 | 0.00% |
| fruit_properties.skin.description | 1 | 0.00% |
| fruit_properties.skin.hairiness | 1 | 0.00% |
| error.message | 1 | 0.00% |
| advertisement.text | 1 | 0.00% |
| Website Navigation | 1 | 0.00% |
| healthcare startup | 1 | 0.00% |
| Internet Domain Name | 1 | 0.00% |
| Website Category | 1 | 0.00% |
| Personal interest | 1 | 0.00% |
| Cybercrime | 1 | 0.00% |
| Authenticity | 1 | 0.00% |
| User account | 1 | 0.00% |
| Accident type | 1 | 0.00% |
| Digital Content | 1 | 0.00% |
| Occasion/Event | 1 | 0.00% |
| Countertop Material | 1 | 0.00% |
| Tile Material | 1 | 0.00% |
| meal event | 1 | 0.00% |
| Library Service | 1 | 0.00% |
| Education Service | 1 | 0.00% |
| Job Service | 1 | 0.00% |
| Entity Type Unknown | 1 | 0.00% |
| Challenge Group | 1 | 0.00% |
| Expertise/Experience | 1 | 0.00% |
| Family history | 1 | 0.00% |
| hotel.description | 1 | 0.00% |
| hotel.recreation_facilities | 1 | 0.00% |
| hotel.location.district | 1 | 0.00% |
| hotel.location.distance_from_center | 1 | 0.00% |
| hotel.hotel_group | 1 | 0.00% |
| Display Resolution | 1 | 0.00% |
| Display Feature | 1 | 0.00% |
| measurement system | 1 | 0.00% |
| type of cake | 1 | 0.00% |
| representation of the Earth's surface | 1 | 0.00% |
| territories governed by Britain | 1 | 0.00% |
| legislative period | 1 | 0.00% |
| service style | 1 | 0.00% |
| service efficiency | 1 | 0.00% |
| Service request | 1 | 0.00% |
| Organizations | 1 | 0.00% |
| Services | 1 | 0.00% |
| cover | 1 | 0.00% |
| birth_year | 1 | 0.00% |
| website performance | 1 | 0.00% |
| Website/brand | 1 | 0.00% |
| ex-race horse | 1 | 0.00% |
| animal rescue organization | 1 | 0.00% |
| customization option | 1 | 0.00% |
| car modification | 1 | 0.00% |
| auto part | 1 | 0.00% |
| Recreation equipment | 1 | 0.00% |
| Retailers | 1 | 0.00% |
| Equestrian Center | 1 | 0.00% |
| People/Visitors | 1 | 0.00% |
| Skilled/Expert | 1 | 0.00% |
| Experienced | 1 | 0.00% |
| Professionals | 1 | 0.00% |
| Golf club | 1 | 0.00% |
| group/people | 1 | 0.00% |
| Natural Event | 1 | 0.00% |
| Title of anime/manga series | 1 | 0.00% |
| Genre of media | 1 | 0.00% |
| Type of comic book | 1 | 0.00% |
| Type of anime/manga adaptation | 1 | 0.00% |
| Group of people with shared interest | 1 | 0.00% |
| Type of entity in fiction | 1 | 0.00% |
| manifest_id | 1 | 0.00% |
| physical setup | 1 | 0.00% |
| action/result | 1 | 0.00% |
| multimedia | 1 | 0.00% |
| Cooking/Ingredients | 1 | 0.00% |
| Mechanical device | 1 | 0.00% |
| Road condition | 1 | 0.00% |
| Free will | 1 | 0.00% |
| Title of Artwork | 1 | 0.00% |
| Time setting | 1 | 0.00% |
| transaction document | 1 | 0.00% |
| painting | 1 | 0.00% |
| behavior change | 1 | 0.00% |
| distraction | 1 | 0.00% |
| Medical code system | 1 | 0.00% |
| Dog titles | 1 | 0.00% |
| Dog name | 1 | 0.00% |
| Art supply | 1 | 0.00% |
| Art creation | 1 | 0.00% |
| Mythology/ Folklore | 1 | 0.00% |
| Ideal Object | 1 | 0.00% |
| receivingDocumentNumber | 1 | 0.00% |
| rejectionReasonCode | 1 | 0.00% |
| rejectionReasonDescription | 1 | 0.00% |
| correctiveAction | 1 | 0.00% |
| speaker/character | 1 | 0.00% |
| Nickname/Title | 1 | 0.00% |
| living condition/housing standard | 1 | 0.00% |
| Writer/Journalist | 1 | 0.00% |
| Diamond Company | 1 | 0.00% |
| Business Innovation | 1 | 0.00% |
| Marketing Technique | 1 | 0.00% |
| Sleep deprivation | 1 | 0.00% |
| Emotional stress | 1 | 0.00% |
| Motherhood | 1 | 0.00% |
| Mental health issue | 1 | 0.00% |
| Housework | 1 | 0.00% |
| Domestic environment | 1 | 0.00% |
| Explanation | 1 | 0.00% |
| foliage | 1 | 0.00% |
| garden feature | 1 | 0.00% |
| Committee/Group | 1 | 0.00% |
| learning environment | 1 | 0.00% |
| Academic Qualification | 1 | 0.00% |
| social connection | 1 | 0.00% |
| Spatial location | 1 | 0.00% |
| Light | 1 | 0.00% |
| Vacation Plan | 1 | 0.00% |
| Education platform | 1 | 0.00% |
| Home Feature | 1 | 0.00% |
| noun phrase | 1 | 0.00% |
| photo quality | 1 | 0.00% |
| Diet | 1 | 0.00% |
| Reward Program | 1 | 0.00% |
| Celestial Object | 1 | 0.00% |
| Automobile Entity | 1 | 0.00% |
| Repair Entity | 1 | 0.00% |
| Relationship Outcome | 1 | 0.00% |
| totalQuantityProduced | 1 | 0.00% |
| totalDuration | 1 | 0.00% |
| overallYield | 1 | 0.00% |
| country name | 1 | 0.00% |
| Land/Property | 1 | 0.00% |
| Geology/Science | 1 | 0.00% |
| Element/Chemistry | 1 | 0.00% |
| Natural resources | 1 | 0.00% |
| graphics element | 1 | 0.00% |
| Activity/Action | 1 | 0.00% |
| person/animal | 1 | 0.00% |
| Photography Equipment | 1 | 0.00% |
| Photography Technique | 1 | 0.00% |
| Music playlist | 1 | 0.00% |
| subscriber_name | 1 | 0.00% |
| subscription_amount | 1 | 0.00% |
| Carnival Parade Group | 1 | 0.00% |
| Carnival Float | 1 | 0.00% |
| Application name | 1 | 0.00% |
| Address book entry | 1 | 0.00% |
| Number limit | 1 | 0.00% |
| Research tool | 1 | 0.00% |
| Paper size | 1 | 0.00% |
| beauty brand | 1 | 0.00% |
| award event | 1 | 0.00% |
| damsel in distress | 1 | 0.00% |
| home automation system | 1 | 0.00% |
| Reason for donation | 1 | 0.00% |
| Religious Order | 1 | 0.00% |
| healthcare plan | 1 | 0.00% |
| dining | 1 | 0.00% |
| technology/electronics | 1 | 0.00% |
| danger/safety | 1 | 0.00% |
| architecture/history | 1 | 0.00% |
| scholarship program | 1 | 0.00% |
| job/occupation | 1 | 0.00% |
| Fertilizer | 1 | 0.00% |
| Agricultural technique | 1 | 0.00% |
| Product Style | 1 | 0.00% |
| Logging technique | 1 | 0.00% |
| Influencing factors | 1 | 0.00% |
| Logging mistake | 1 | 0.00% |
| Forest location | 1 | 0.00% |
| Acquisition Target | 1 | 0.00% |
| Acquisition Event | 1 | 0.00% |
| Singer/Artist | 1 | 0.00% |
| business and commerce | 1 | 0.00% |
| holiday season | 1 | 0.00% |
| historical artifact | 1 | 0.00% |
| Document/Event | 1 | 0.00% |
| Department Name | 1 | 0.00% |
| mechanism: Biological action or mechanism of the drug | 1 | 0.00% |
| drug_name: Name of the prescribed medication | 1 | 0.00% |
| frequency: How often the medication is taken | 1 | 0.00% |
| garden property | 1 | 0.00% |
| natural wonder | 1 | 0.00% |
| sex workers | 1 | 0.00% |
| sex toys | 1 | 0.00% |
| escorts | 1 | 0.00% |
| route decision | 1 | 0.00% |
| Achievements | 1 | 0.00% |
| Struggles | 1 | 0.00% |
| Emotion/Desire | 1 | 0.00% |
| Auxiliary verb | 1 | 0.00% |
| Family Structure | 1 | 0.00% |
| Cultural Beliefs | 1 | 0.00% |
| TechnicalProcess | 1 | 0.00% |
| airline_code: Two-letter code representing the operating airline | 1 | 0.00% |
| flight_no: Flight number as assigned by the airline | 1 | 0.00% |
| unique_id: Unique identifier for the delay entry | 1 | 0.00% |
| delay_minutes: Number of minutes flight is delayed | 1 | 0.00% |
| destination: Airport code where flight ends | 1 | 0.00% |
| Sport Statistic | 1 | 0.00% |
| Agricultural Crop | 1 | 0.00% |
| automotive issue | 1 | 0.00% |
| brand slogan | 1 | 0.00% |
| quest | 1 | 0.00% |
| Grade | 1 | 0.00% |
| Admission process | 1 | 0.00% |
| Registration deadline | 1 | 0.00% |
| Entity type: Statistics | 1 | 0.00% |
| Entity type: Map | 1 | 0.00% |
| Entity type: Professional | 1 | 0.00% |
| Entity type: Date | 1 | 0.00% |
| Dental Issue | 1 | 0.00% |
| Sensory Reaction | 1 | 0.00% |
| programming framework | 1 | 0.00% |
| Personal Care/Wellness | 1 | 0.00% |
| Facilities/Comfort | 1 | 0.00% |
| Technology/Communication | 1 | 0.00% |
| Technology/Functionality | 1 | 0.00% |
| claims_count | 1 | 0.00% |
| filing_location | 1 | 0.00% |
| Art Presentation | 1 | 0.00% |
| president | 1 | 0.00% |
| Artist and spouse | 1 | 0.00% |
| hairstyle tool | 1 | 0.00% |
| Payment platform | 1 | 0.00% |
| Trail | 1 | 0.00% |
| Music Event | 1 | 0.00% |
| Astrology | 1 | 0.00% |
| Horoscope Predictions | 1 | 0.00% |
| Religion/Spirituality | 1 | 0.00% |
| Predictions | 1 | 0.00% |
| stars | 1 | 0.00% |
| project_url | 1 | 0.00% |
| maintainers_location | 1 | 0.00% |
| sync_frequency | 1 | 0.00% |
| group of competitors | 1 | 0.00% |
| robotic models | 1 | 0.00% |
| clothing attribute | 1 | 0.00% |
| Authentication method | 1 | 0.00% |
| Encryption protocol | 1 | 0.00% |
| Security setting | 1 | 0.00% |
| Light source | 1 | 0.00% |
| Legal Official | 1 | 0.00% |
| tattoo style | 1 | 0.00% |
| collection of visual art | 1 | 0.00% |
| Education Policy | 1 | 0.00% |
| Architecture and construction | 1 | 0.00% |
| Viewpoints | 1 | 0.00% |
| Band/Musician | 1 | 0.00% |
| Music Magazine/Publication | 1 | 0.00% |
| Music Reviewer/Critic | 1 | 0.00% |
| Offered services | 1 | 0.00% |
| historical document | 1 | 0.00% |
| spiritual entity | 1 | 0.00% |
| Plug and play car engine control unit | 1 | 0.00% |
| Command | 1 | 0.00% |
| Time Difference | 1 | 0.00% |
| adaptability | 1 | 0.00% |
| Urban Area | 1 | 0.00% |
| Volume | 1 | 0.00% |
| Source of Pollution | 1 | 0.00% |
| Architectural Plan | 1 | 0.00% |
| Educational Material | 1 | 0.00% |
| Initiative | 1 | 0.00% |
| Signing | 1 | 0.00% |
| Technology/Internet Service Provider | 1 | 0.00% |
| Property/Asset | 1 | 0.00% |
| management_comment_prompt | 1 | 0.00% |
| winter | 1 | 0.00% |
| disruption | 1 | 0.00% |
| unsteady | 1 | 0.00% |
| Club/Organization | 1 | 0.00% |
| Restaurant/Hotel | 1 | 0.00% |
| security system | 1 | 0.00% |
| bio | 1 | 0.00% |
| Intellectual property | 1 | 0.00% |
| License type | 1 | 0.00% |
| Geographic scope | 1 | 0.00% |
| License permanence | 1 | 0.00% |
| License purpose | 1 | 0.00% |
| Data visualization | 1 | 0.00% |
| party | 1 | 0.00% |
| vote_percentage | 1 | 0.00% |
| JJ | 1 | 0.00% |
| DT | 1 | 0.00% |
| . | 1 | 0.00% |
| CC | 1 | 0.00% |
| knowledge domains | 1 | 0.00% |
| picture size | 1 | 0.00% |
| picture frame | 1 | 0.00% |
| frame size | 1 | 0.00% |
| School Year | 1 | 0.00% |
| First Day of School | 1 | 0.00% |
| School Information | 1 | 0.00% |
| decorative entity | 1 | 0.00% |
| Location/Activity | 1 | 0.00% |
| Time/Comparison | 1 | 0.00% |
| Product Condition | 1 | 0.00% |
| Academic Faculty | 1 | 0.00% |
| website ranking | 1 | 0.00% |
| online platform for personal writing | 1 | 0.00% |
| written summary or report | 1 | 0.00% |
| influence or persuade | 1 | 0.00% |
| examine or review | 1 | 0.00% |
| Project/Task/Assignment | 1 | 0.00% |
| hotel/resort | 1 | 0.00% |
| prescribed drug | 1 | 0.00% |
| Animal Group | 1 | 0.00% |
| sensory system | 1 | 0.00% |
| auditory perception | 1 | 0.00% |
| visual perception | 1 | 0.00% |
| olfactory perception | 1 | 0.00% |
| gustatory perception | 1 | 0.00% |
| somatosensory system | 1 | 0.00% |
| genetic makeup | 1 | 0.00% |
| CellComponent | 1 | 0.00% |
| customization options | 1 | 0.00% |
| personalization option | 1 | 0.00% |
| Personal Life | 1 | 0.00% |
| financial constraint | 1 | 0.00% |
| data/result | 1 | 0.00% |
| absence or lack of something | 1 | 0.00% |
| habitat | 1 | 0.00% |
| Rebate Program | 1 | 0.00% |
| liquid substance | 1 | 0.00% |
| project/creation | 1 | 0.00% |
| opinion/judgement | 1 | 0.00% |
| Substance/Element | 1 | 0.00% |
| mountain range | 1 | 0.00% |
| Structural Entity | 1 | 0.00% |
| Statistic/Record | 1 | 0.00% |
| game event | 1 | 0.00% |
| Group/Profession | 1 | 0.00% |
| software development methodology | 1 | 0.00% |
| Academic Position | 1 | 0.00% |
| Physical structure | 1 | 0.00% |
| Environmental quality | 1 | 0.00% |
| Ecological system | 1 | 0.00% |
| bookingId | 1 | 0.00% |
| email_id | 1 | 0.00% |
| date_sent | 1 | 0.00% |
| Assistant Coach | 1 | 0.00% |
| sports event analysis | 1 | 0.00% |
| psychological test | 1 | 0.00% |
| resolution_detail | 1 | 0.00% |
| Time Period Entity | 1 | 0.00% |
| Event Entity | 1 | 0.00% |
| brand/shop | 1 | 0.00% |
| interest/group | 1 | 0.00% |
| Flight | 1 | 0.00% |
| Space for Dining | 1 | 0.00% |
| Personal well-being | 1 | 0.00% |
| External events | 1 | 0.00% |
| simplicity | 1 | 0.00% |
| dependability | 1 | 0.00% |
| flexibility | 1 | 0.00% |
| user/group | 1 | 0.00% |
| Firefox extension | 1 | 0.00% |
| Extension API | 1 | 0.00% |
| Exam preparation materials | 1 | 0.00% |
| funding_id | 1 | 0.00% |
| Backup service | 1 | 0.00% |
| product/material | 1 | 0.00% |
| attribute/characteristic | 1 | 0.00% |
| Company/Restaurant | 1 | 0.00% |
| Building complex | 1 | 0.00% |
| Building name | 1 | 0.00% |
| Fitness facility | 1 | 0.00% |
| Swimming facility | 1 | 0.00% |
| Email Spam | 1 | 0.00% |
| person with addiction issue | 1 | 0.00% |
| treatment facility | 1 | 0.00% |
| Interest level | 1 | 0.00% |
| bicycle accessory | 1 | 0.00% |
| products/services | 1 | 0.00% |
| purposes/use cases | 1 | 0.00% |
| Decoration item | 1 | 0.00% |
| Small Architecture Machine | 1 | 0.00% |
| dose_number: Sequence of the dose in the series | 1 | 0.00% |
| record_id: Unique identifier for the vaccination record | 1 | 0.00% |
| location: City or region where vaccination occurred | 1 | 0.00% |
| Biological specimen | 1 | 0.00% |
| Education/Performance | 1 | 0.00% |
| survey participants | 1 | 0.00% |
| UFO | 1 | 0.00% |
| Supernatural substance | 1 | 0.00% |
| UFO subtype | 1 | 0.00% |
| Smell | 1 | 0.00% |
| Constraint | 1 | 0.00% |
| Match | 1 | 0.00% |
| doctorId | 1 | 0.00% |
| lowerBound | 1 | 0.00% |
| upperBound | 1 | 0.00% |
| resultValue | 1 | 0.00% |
| resultDate | 1 | 0.00% |
| Cloth Diapers | 1 | 0.00% |
| Seller | 1 | 0.00% |
| unknown entity type | 1 | 0.00% |
| Magic practice | 1 | 0.00% |
| Religion/Culture | 1 | 0.00% |
| general item | 1 | 0.00% |
| forum name | 1 | 0.00% |
| sweepstakes type | 1 | 0.00% |
| promotion type | 1 | 0.00% |
| vendor pricing | 1 | 0.00% |
| Assisted Living Facility | 1 | 0.00% |
| publication medium | 1 | 0.00% |
| legal building | 1 | 0.00% |
| Village Museum | 1 | 0.00% |
| Architectural Collection | 1 | 0.00% |
| writing ability | 1 | 0.00% |
| challenging endeavor | 1 | 0.00% |
| instructional content | 1 | 0.00% |
| website rating | 1 | 0.00% |
| smart servo | 1 | 0.00% |
| replacement part | 1 | 0.00% |
| Seasons | 1 | 0.00% |
| Social groups | 1 | 0.00% |
| positive trait | 1 | 0.00% |
| Idiom/Phrase | 1 | 0.00% |
| child type | 1 | 0.00% |
| Medical | 1 | 0.00% |
| Social class | 1 | 0.00% |
| photography/film industry | 1 | 0.00% |
| mental state/behavior | 1 | 0.00% |
| physical appearance/behavior | 1 | 0.00% |
| Affiliate company | 1 | 0.00% |
| Subsidiary company | 1 | 0.00% |
| task or action | 1 | 0.00% |
| Pokemon type | 1 | 0.00% |
| Pokemon stage | 1 | 0.00% |
| Pokemon expansion | 1 | 0.00% |
| seasonal period | 1 | 0.00% |
| Novelist | 1 | 0.00% |
| Group of entities | 1 | 0.00% |
| Interior design | 1 | 0.00% |
| Occupants | 1 | 0.00% |
| Planning and execution of a project | 1 | 0.00% |
| Protection | 1 | 0.00% |
| Amusement ride | 1 | 0.00% |
| Ride car | 1 | 0.00% |
| Ride station | 1 | 0.00% |
| housing | 1 | 0.00% |
| time/distance | 1 | 0.00% |
| military | 1 | 0.00% |
| Kitchen Backsplash Ideas | 1 | 0.00% |
| Certification Body | 1 | 0.00% |
| Ski equipment | 1 | 0.00% |
| security technology | 1 | 0.00% |
| SSL brands | 1 | 0.00% |
| Water source | 1 | 0.00% |
| Business/Directory | 1 | 0.00% |
| assessment company | 1 | 0.00% |
| Road Junction | 1 | 0.00% |
| Bridge | 1 | 0.00% |
| Personal Grooming | 1 | 0.00% |
| Opinion/Response | 1 | 0.00% |
| Sentiment/Emotion | 1 | 0.00% |
| ship | 1 | 0.00% |
| Brand name | 1 | 0.00% |
| Educational Class | 1 | 0.00% |
| Computer programming language | 1 | 0.00% |
| Educational classroom | 1 | 0.00% |
| Industry average | 1 | 0.00% |
| technology/website | 1 | 0.00% |
| event feature/artist type | 1 | 0.00% |
| step | 1 | 0.00% |
| checkout | 1 | 0.00% |
| Customization Service | 1 | 0.00% |
| Building features | 1 | 0.00% |
| Building number | 1 | 0.00% |
| Service time | 1 | 0.00% |
| storage unit | 1 | 0.00% |
| state of being at ease or relaxed | 1 | 0.00% |
| Event/Website | 1 | 0.00% |
| Community/Website | 1 | 0.00% |
| bird symbol | 1 | 0.00% |
| honorific order | 1 | 0.00% |
| customization | 1 | 0.00% |
| Request Type | 1 | 0.00% |
| industry/field | 1 | 0.00% |
| commercial use | 1 | 0.00% |
| mark on surface | 1 | 0.00% |
| startup | 1 | 0.00% |
| investor | 1 | 0.00% |
| funding_round | 1 | 0.00% |
| release_target | 1 | 0.00% |
| Judaica Item | 1 | 0.00% |
| Legal System | 1 | 0.00% |
| Military Action | 1 | 0.00% |
| amount: Claimed financial amount in dollars | 1 | 0.00% |
| malicious_content: Description of the phishing element like link or attachment | 1 | 0.00% |
| sender: Name of person who submitted the report | 1 | 0.00% |
| Travel Accommodation | 1 | 0.00% |
| Patient wards | 1 | 0.00% |
| Food preparation area | 1 | 0.00% |
| Food production area | 1 | 0.00% |
| Washing facility | 1 | 0.00% |
| Power generation unit | 1 | 0.00% |
| Waste management facility | 1 | 0.00% |
| Patient housing | 1 | 0.00% |
| animal sound | 1 | 0.00% |
| Product family | 1 | 0.00% |
| Debt Division | 1 | 0.00% |
| Supernatural entity | 1 | 0.00% |
| accessibility feature | 1 | 0.00% |
| energy | 1 | 0.00% |
| communication document | 1 | 0.00% |
| violation | 1 | 0.00% |
| legal responsibility | 1 | 0.00% |
| financial cost | 1 | 0.00% |
| Interaction | 1 | 0.00% |
| Person/group of people | 1 | 0.00% |
| Loan option | 1 | 0.00% |
| Special situation | 1 | 0.00% |
| templates | 1 | 0.00% |
| Software/Service | 1 | 0.00% |
| outdoor living area | 1 | 0.00% |
| HVAC component | 1 | 0.00% |
| alcohol consumption | 1 | 0.00% |
| Part number | 1 | 0.00% |
| Product availability | 1 | 0.00% |
| Shipping time | 1 | 0.00% |
| individual/group of people | 1 | 0.00% |
| Law enforcement action | 1 | 0.00% |
| Political opinion | 1 | 0.00% |
| Credit Card | 1 | 0.00% |
| model_id | 1 | 0.00% |
| Question word | 1 | 0.00% |
| Personal transformation | 1 | 0.00% |
| addressee | 1 | 0.00% |
| qualification/certification | 1 | 0.00% |
| Organization/Building | 1 | 0.00% |
| Human Right | 1 | 0.00% |
| Figure Reference | 1 | 0.00% |
| Body measurement | 1 | 0.00% |
| results.outcomes.1.post_intervention.mean | 1 | 0.00% |
| study_metadata.objective | 1 | 0.00% |
| results.outcomes.4.pre_intervention.standard_deviation | 1 | 0.00% |
| participants.groups.1.allocation_method | 1 | 0.00% |
| results.outcomes.5.pre_intervention.mean | 1 | 0.00% |
| results.outcomes.5.post_intervention.mean | 1 | 0.00% |
| participants.population_type | 1 | 0.00% |
| results.outcomes.1.construct | 1 | 0.00% |
| results.outcomes.0.construct | 1 | 0.00% |
| results.outcomes.5.construct | 1 | 0.00% |
| study_metadata.setting.location | 1 | 0.00% |
| methodology.assessment_tool.validation_metrics.0.value | 1 | 0.00% |
| results.outcomes.4.pre_intervention.mean | 1 | 0.00% |
| results.statistical_significance_threshold | 1 | 0.00% |
| results.outcomes.1.pre_intervention.standard_deviation | 1 | 0.00% |
| conclusions.intervention_effect | 1 | 0.00% |
| results.outcomes.5.post_intervention.standard_deviation | 1 | 0.00% |
| participants.groups.0.sample_size | 1 | 0.00% |
| study_metadata.study_design | 1 | 0.00% |
| results.outcomes.2.post_intervention.standard_deviation | 1 | 0.00% |
| results.outcomes.1.post_intervention.standard_deviation | 1 | 0.00% |
| results.outcomes.4.post_intervention.standard_deviation | 1 | 0.00% |
| study_metadata.topic | 1 | 0.00% |
| results.outcomes.3.pre_intervention.standard_deviation | 1 | 0.00% |
| results.outcomes.1.pre_intervention.mean | 1 | 0.00% |
| methodology.assessment_tool.type | 1 | 0.00% |
| participants.total_sample_size | 1 | 0.00% |
| results.outcomes.2.pre_intervention.mean | 1 | 0.00% |
| results.outcomes.3.construct | 1 | 0.00% |
| participants.groups.0.allocation_method | 1 | 0.00% |
| study_metadata.setting.institution | 1 | 0.00% |
| results.outcomes.4.construct | 1 | 0.00% |
| methodology.assessment_tool.validation_metrics.0.metric_name | 1 | 0.00% |
| results.outcomes.3.post_intervention.standard_deviation | 1 | 0.00% |
| results.outcomes.0.pre_intervention.mean | 1 | 0.00% |
| User/Username | 1 | 0.00% |
| Malfunction | 1 | 0.00% |
| Nationality/Gender | 1 | 0.00% |
| Scam type | 1 | 0.00% |
| defense | 1 | 0.00% |
| mana | 1 | 0.00% |
| agility | 1 | 0.00% |
| intelligence | 1 | 0.00% |
| Android setting | 1 | 0.00% |
| caregiver | 1 | 0.00% |
| treat | 1 | 0.00% |
| snack | 1 | 0.00% |
| people/role | 1 | 0.00% |
| process/type of product | 1 | 0.00% |
| Exalted character avatar | 1 | 0.00% |
| Dynastic dilettante | 1 | 0.00% |
| assessmentDate | 1 | 0.00% |
| assessor | 1 | 0.00% |
| overallRisk | 1 | 0.00% |
| Nationality/Relationship | 1 | 0.00% |
| data storage technology | 1 | 0.00% |
| bandwidth capacity | 1 | 0.00% |
| website limitation | 1 | 0.00% |
| Savior Figure | 1 | 0.00% |
| cardinal direction | 1 | 0.00% |
| transmission medium | 1 | 0.00% |
| Amenities | 1 | 0.00% |
| driving location | 1 | 0.00% |
| racing type | 1 | 0.00% |
| Educational level | 1 | 0.00% |
| transaction_type | 1 | 0.00% |
| Musical duo | 1 | 0.00% |
| Musical production | 1 | 0.00% |
| principle | 1 | 0.00% |
| loyalty | 1 | 0.00% |
| Legal Risk | 1 | 0.00% |
| Theme/Decoration | 1 | 0.00% |
| cooking action | 1 | 0.00% |
| Drug Delivery System | 1 | 0.00% |
| gear | 1 | 0.00% |
| personal care items | 1 | 0.00% |
| self-defense weapon | 1 | 0.00% |
| Race | 1 | 0.00% |
| Arabic greeting | 1 | 0.00% |
| Female Muslim | 1 | 0.00% |
| Book publisher | 1 | 0.00% |
| alternative spelling | 1 | 0.00% |
| Medical Center | 1 | 0.00% |
| Biological Organ | 1 | 0.00% |
| cuisine/dish | 1 | 0.00% |
| social circle | 1 | 0.00% |
| unknown audience | 1 | 0.00% |
| personal hygiene habit | 1 | 0.00% |
| offer/promotion | 1 | 0.00% |
| Article ID | 1 | 0.00% |
| Fax Number | 1 | 0.00% |
| shopping event | 1 | 0.00% |
| personal item | 1 | 0.00% |
| Supernatural beings | 1 | 0.00% |
| Fictional creatures | 1 | 0.00% |
| Judiciary | 1 | 0.00% |
| Sales performance | 1 | 0.00% |
| behavioral issue | 1 | 0.00% |
| support group | 1 | 0.00% |
| weight loss and fitness group | 1 | 0.00% |
| Character/Brand | 1 | 0.00% |
| Fraction | 1 | 0.00% |
| Discrimination | 1 | 0.00% |
| bonus feature | 1 | 0.00% |
| Era | 1 | 0.00% |
| Organization/Charity | 1 | 0.00% |
| Work Ethic | 1 | 0.00% |
| Attention to Detail | 1 | 0.00% |
| Food Preparation | 1 | 0.00% |
| tool or software | 1 | 0.00% |
| product or service | 1 | 0.00% |
| Edition | 1 | 0.00% |
| Athletic division | 1 | 0.00% |
| quality product/service | 1 | 0.00% |
| cost-saving strategy | 1 | 0.00% |
| assistance/help | 1 | 0.00% |
| affordable price/quality ratio | 1 | 0.00% |
| health benefits | 1 | 0.00% |
| alternative option | 1 | 0.00% |
| person/recipient | 1 | 0.00% |
| person/sender | 1 | 0.00% |
| Real Estate Data | 1 | 0.00% |
| web address | 1 | 0.00% |
| vehicle.make | 1 | 0.00% |
| vehicle.componentReplacements | 1 | 0.00% |
| vehicle.year | 1 | 0.00% |
| problemDescription | 1 | 0.00% |
| vehicle.model | 1 | 0.00% |
| Game Environment | 1 | 0.00% |
| Game Objective | 1 | 0.00% |
| Percentage Range | 1 | 0.00% |
| gym equipment | 1 | 0.00% |
| data records | 1 | 0.00% |
| View | 1 | 0.00% |
| Entryway | 1 | 0.00% |
| Staircase | 1 | 0.00% |
| experience_level | 1 | 0.00% |
| Year of establishment | 1 | 0.00% |
| action/feeling | 1 | 0.00% |
| Interior Designer | 1 | 0.00% |
| Bail Company | 1 | 0.00% |
| Website links | 1 | 0.00% |
| Websites | 1 | 0.00% |
| Tool/Service | 1 | 0.00% |
| work performance | 1 | 0.00% |
| work attendance | 1 | 0.00% |
| policyholder | 1 | 0.00% |
| coverage_amount | 1 | 0.00% |
| term_length | 1 | 0.00% |
| Conjunction | 1 | 0.00% |
| Adverb/Adjective | 1 | 0.00% |
| role/job | 1 | 0.00% |
| training_program.name | 1 | 0.00% |
| training_program.payment.options | 1 | 0.00% |
| training_program.payment.full_payment_amount | 1 | 0.00% |
| training_program.payment.deposit_amount | 1 | 0.00% |
| training_program.payment.payment_instructions | 1 | 0.00% |
| Literary device | 1 | 0.00% |
| employment history | 1 | 0.00% |
| job reference | 1 | 0.00% |
| job search | 1 | 0.00% |
| professional conduct | 1 | 0.00% |
| TV Show/Media | 1 | 0.00% |
| Medical Document | 1 | 0.00% |
| Error indicator | 1 | 0.00% |
| Printing issue | 1 | 0.00% |
| Information source | 1 | 0.00% |
| Counter mechanism | 1 | 0.00% |
| Technical support | 1 | 0.00% |
| datePublished | 1 | 0.00% |
| dateModified | 1 | 0.00% |
| Singer | 1 | 0.00% |
| GUI Control | 1 | 0.00% |
| Window Handle | 1 | 0.00% |
| Protein_complex | 1 | 0.00% |
| Lighting System | 1 | 0.00% |
| Object/Collection | 1 | 0.00% |
| Body_part/Organ | 1 | 0.00% |
| Garden Decor | 1 | 0.00% |
| Building/Art and Craft | 1 | 0.00% |
| thing/object | 1 | 0.00% |
| time/period | 1 | 0.00% |
| Discourse/Communication | 1 | 0.00% |
| issue/problem/concern | 1 | 0.00% |
| Kitchen Design | 1 | 0.00% |
| Kitchen Countertops | 1 | 0.00% |
| Job System | 1 | 0.00% |
| Meditation Host | 1 | 0.00% |
| Radio Show | 1 | 0.00% |
| House Builder | 1 | 0.00% |
| substance abuse | 1 | 0.00% |
| Stylus | 1 | 0.00% |
| Partnership | 1 | 0.00% |
| Pokemon character | 1 | 0.00% |
| atmospheric pollutant | 1 | 0.00% |
| software updates | 1 | 0.00% |
| Service/Brand | 1 | 0.00% |
| Service/Industry | 1 | 0.00% |
| Verification | 1 | 0.00% |
| Proof of income | 1 | 0.00% |
| Health and wellness | 1 | 0.00% |
| Event/Exhibition | 1 | 0.00% |
| technical problem | 1 | 0.00% |
| Event Time | 1 | 0.00% |
| interpretation.0.meaning | 1 | 0.00% |
| interpretation.0.details.health | 1 | 0.00% |
| interpretation.1.meaning | 1 | 0.00% |
| interpretation.1.details.health | 1 | 0.00% |
| dream_element | 1 | 0.00% |
| Vineyards | 1 | 0.00% |
| Wines | 1 | 0.00% |
| Tower | 1 | 0.00% |
| treatment requirement | 1 | 0.00% |
| personal activity | 1 | 0.00% |
| Movie Enthusiast | 1 | 0.00% |
| Personal attribute | 1 | 0.00% |
| Local Distributor | 1 | 0.00% |
| Training Provider | 1 | 0.00% |
| sound stimuli | 1 | 0.00% |
| auditory cue | 1 | 0.00% |
| sound pattern | 1 | 0.00% |
| same | 1 | 0.00% |
| Landmarks/Architecture | 1 | 0.00% |
| Life experiences | 1 | 0.00% |
| Clothing items | 1 | 0.00% |
| affiliation.0.type | 1 | 0.00% |
| person/author | 1 | 0.00% |
| title/section | 1 | 0.00% |
| Dates | 1 | 0.00% |
| Criteria | 1 | 0.00% |
| Skill level | 1 | 0.00% |
| Blogging tips | 1 | 0.00% |
| Marketing course | 1 | 0.00% |
| Marketing network | 1 | 0.00% |
| Email marketing advice | 1 | 0.00% |
| Business model | 1 | 0.00% |
| Work arrangement | 1 | 0.00% |
| Wealthy individuals | 1 | 0.00% |
| Purchasing goods | 1 | 0.00% |
| Desired item | 1 | 0.00% |
| Approach the desired item | 1 | 0.00% |
| Observe the desired item | 1 | 0.00% |
| tournament_code | 1 | 0.00% |
| aces | 1 | 0.00% |
| Terrestrial Biome | 1 | 0.00% |
| clause_id | 1 | 0.00% |
| Conference Attendee | 1 | 0.00% |
| Cultural Clothing | 1 | 0.00% |
| Planning and Organization | 1 | 0.00% |
| Beer Type | 1 | 0.00% |
| education status | 1 | 0.00% |
| scientific phenomenon | 1 | 0.00% |
| natural object | 1 | 0.00% |
| Yarn Store | 1 | 0.00% |
| Software Installation | 1 | 0.00% |
| Directory | 1 | 0.00% |
| Review Methodology | 1 | 0.00% |
| Sports Field | 1 | 0.00% |
| travel intention | 1 | 0.00% |
| approvalDate | 1 | 0.00% |
| approvalNumber | 1 | 0.00% |
| approvalType | 1 | 0.00% |
| drug name | 1 | 0.00% |
| reference book | 1 | 0.00% |
| author(s) | 1 | 0.00% |
| publication year | 1 | 0.00% |
| trophy | 1 | 0.00% |
| sports achievement | 1 | 0.00% |
| YouTube channel | 1 | 0.00% |
| AlmazanKitchen | 1 | 0.00% |
| album_series | 1 | 0.00% |
| dating website | 1 | 0.00% |
| group of people/things | 1 | 0.00% |
| cultural artifact | 1 | 0.00% |
| industrial facility | 1 | 0.00% |
| Home Automation System | 1 | 0.00% |
| Tagline | 1 | 0.00% |
| area measurement | 1 | 0.00% |
| architectural feature/construction material | 1 | 0.00% |
| Qualifier | 1 | 0.00% |
| Television Channel | 1 | 0.00% |
| Art/Craft | 1 | 0.00% |
| Slogan/Phrase | 1 | 0.00% |
| neurodevelopmental disorder | 1 | 0.00% |
| individual instance | 1 | 0.00% |
| website URL | 1 | 0.00% |
| Idiomatic Phrase | 1 | 0.00% |
| Zoning Classification | 1 | 0.00% |
| Land Use Plan | 1 | 0.00% |
| Local Government Body | 1 | 0.00% |
| blockchain-based service | 1 | 0.00% |
| decentralized domain system | 1 | 0.00% |
| Geopolitical Location | 1 | 0.00% |
| Real estate entities | 1 | 0.00% |
| Wealth entities | 1 | 0.00% |
| technology component | 1 | 0.00% |
| tire type | 1 | 0.00% |
| product/name | 1 | 0.00% |
| stage of career | 1 | 0.00% |
| surgical incisions | 1 | 0.00% |
| ingredient group | 1 | 0.00% |
| biological relationship | 1 | 0.00% |
| Glass type | 1 | 0.00% |
| Investment Accounts | 1 | 0.00% |
| Exercise position | 1 | 0.00% |
| availability_date | 1 | 0.00% |
| organism component | 1 | 0.00% |
| sound recording technique | 1 | 0.00% |
| difficult task | 1 | 0.00% |
| maintenance service | 1 | 0.00% |
| financial value | 1 | 0.00% |
| Movie/Sci-Fi Series | 1 | 0.00% |
| TV Show/Sci-Fi Series | 1 | 0.00% |
| Movie/Comic Book Series | 1 | 0.00% |
| Legal party | 1 | 0.00% |
| Salary range | 1 | 0.00% |
| Qualification | 1 | 0.00% |
| Reimbursement | 1 | 0.00% |
| Career growth | 1 | 0.00% |
| musical band/group | 1 | 0.00% |
| cultural element | 1 | 0.00% |
| match_id: Unique identifier for the match as provided | 1 | 0.00% |
| home_team: Team playing at the home venue | 1 | 0.00% |
| away_team: Team playing at the away venue | 1 | 0.00% |
| score: Final score in format home-away | 1 | 0.00% |
| venue: Stadium where the match took place | 1 | 0.00% |
| competition: League or cup type such as Premier League or friendly | 1 | 0.00% |
| post parameter | 1 | 0.00% |
| Job Activity | 1 | 0.00% |
| Snack food | 1 | 0.00% |
| Packaging state | 1 | 0.00% |
| Watch Model | 1 | 0.00% |
| Watch Accessory | 1 | 0.00% |
| fictional world | 1 | 0.00% |
| printing technology | 1 | 0.00% |
| moving | 1 | 0.00% |
| professional task | 1 | 0.00% |
| aesthetics | 1 | 0.00% |
| Direction/Orientation | 1 | 0.00% |
| transaction/event | 1 | 0.00% |
| document/type of writing | 1 | 0.00% |
| possessive pronoun | 1 | 0.00% |
| Legal Title | 1 | 0.00% |
| Legal Abbreviation | 1 | 0.00% |
| lat | 1 | 0.00% |
| lng | 1 | 0.00% |
| roomTypeId | 1 | 0.00% |
| Virtualization Software | 1 | 0.00% |
| group of visitors | 1 | 0.00% |
| author/owner of the studio | 1 | 0.00% |
| places of artistic creation | 1 | 0.00% |
| specific studio name | 1 | 0.00% |
| other artist's studio | 1 | 0.00% |
| studio that was fully accessible to public | 1 | 0.00% |
| Social Circle | 1 | 0.00% |
| Visual Outcome | 1 | 0.00% |
| stress reduction | 1 | 0.00% |
| dietary pattern | 1 | 0.00% |
| physiological process | 1 | 0.00% |
| Tool/Utensil | 1 | 0.00% |
| Tree condition | 1 | 0.00% |
| physical component | 1 | 0.00% |
| physical wellness | 1 | 0.00% |
| documentary film | 1 | 0.00% |
| display setting | 1 | 0.00% |
| phone model | 1 | 0.00% |
| Action/Functionality | 1 | 0.00% |
| Regional Identity | 1 | 0.00% |
| Productivity | 1 | 0.00% |
| cost-effective method | 1 | 0.00% |
| clock assembly | 1 | 0.00% |
| contents | 1 | 0.00% |
| search parameter | 1 | 0.00% |
| Event/Concert | 1 | 0.00% |
| leafy green vegetable | 1 | 0.00% |
| Drone | 1 | 0.00% |
| document/page | 1 | 0.00% |
| basketball effort | 1 | 0.00% |
| basketball court area | 1 | 0.00% |
| basketball foul line | 1 | 0.00% |
| basketball record | 1 | 0.00% |
| Book Size | 1 | 0.00% |
| Photo | 1 | 0.00% |
| Book Binding | 1 | 0.00% |
| Price Increase | 1 | 0.00% |
| skincare tool | 1 | 0.00% |
| Geographic data sources | 1 | 0.00% |
| Map visualization | 1 | 0.00% |
| past events/achievements | 1 | 0.00% |
| achievements/successes | 1 | 0.00% |
| perks/advantages | 1 | 0.00% |
| Treatment Effect | 1 | 0.00% |
| date/time information | 1 | 0.00% |
| attitude/behavior | 1 | 0.00% |
| Sense | 1 | 0.00% |
| Object/Toy | 1 | 0.00% |
| Academic Unit | 1 | 0.00% |
| Mortgage Type | 1 | 0.00% |
| agency.mission | 1 | 0.00% |
| ready_campaign_updates.target_audience | 1 | 0.00% |
| ready_campaign_updates.email | 1 | 0.00% |
| ready_campaign_updates.language_note | 1 | 0.00% |
| WTS Atlas <> managed by | 1 | 0.00% |
| Sparta Wroclaw <> managed by | 1 | 0.00% |
| Dariusz Śledź <> participated in | 1 | 0.00% |
| Dariusz Śledź <> born on | 1 | 0.00% |
| Stal Rzeszów <> managed by | 1 | 0.00% |
| Dariusz Śledź <> nationality | 1 | 0.00% |
| Book Chapter | 1 | 0.00% |
| Submersible Grinder Pumps | 1 | 0.00% |
| Residential Grinder Pump | 1 | 0.00% |
| Teaching method | 1 | 0.00% |
| procedureType | 1 | 0.00% |
| Competition winners | 1 | 0.00% |
| Competition event | 1 | 0.00% |
| Competition participants | 1 | 0.00% |
| Art Supplies | 1 | 0.00% |
| Location/Platform | 1 | 0.00% |
| Service/Job | 1 | 0.00% |
| School/University | 1 | 0.00% |
| Degree/Credential | 1 | 0.00% |
| Academic Goal | 1 | 0.00% |
| Mythological Figure | 1 | 0.00% |
| Residents | 1 | 0.00% |
| Accommodation platform | 1 | 0.00% |
| profession/role | 1 | 0.00% |
| group of adoptive parents | 1 | 0.00% |
| Country/Geopolitical entity | 1 | 0.00% |
| Data Security Tool | 1 | 0.00% |
| Computer Security Tool | 1 | 0.00% |
| assessmentReason | 1 | 0.00% |
| diagnosticImpression | 1 | 0.00% |
| Time off from work | 1 | 0.00% |
| Dangerous situation | 1 | 0.00% |
| Help | 1 | 0.00% |
| equipment system | 1 | 0.00% |
| Status/Group | 1 | 0.00% |
| Floor | 1 | 0.00% |
| Cleanliness | 1 | 0.00% |
| Comfort | 1 | 0.00% |
| socio-economic status | 1 | 0.00% |
| Response | 1 | 0.00% |
| Emission standard | 1 | 0.00% |
| adoption process | 1 | 0.00% |
| artistic skill | 1 | 0.00% |
| creative field | 1 | 0.00% |
| sale | 1 | 0.00% |
| Friend | 1 | 0.00% |
| Mission Duration | 1 | 0.00% |
| Mythological creatures | 1 | 0.00% |
| Metropolitan area | 1 | 0.00% |
| supernatural beings | 1 | 0.00% |
| financial issue | 1 | 0.00% |
| natural weed killer | 1 | 0.00% |
| Divine Plan | 1 | 0.00% |
| Atmospheric element | 1 | 0.00% |
| Workspaces | 1 | 0.00% |
| color of the body part of the bag | 1 | 0.00% |
| color of the handle and the back of the hangtag | 1 | 0.00% |
| dirt on the handle and the back of the hangtag (see last picture) | 1 | 0.00% |
| activity/work | 1 | 0.00% |
| activity/entertainment | 1 | 0.00% |
| color accents | 1 | 0.00% |
| military hierarchy | 1 | 0.00% |
| storytelling element | 1 | 0.00% |
| task/operation | 1 | 0.00% |
| House Design Ideas | 1 | 0.00% |
| Chance/opportunity | 1 | 0.00% |
| Show/Media | 1 | 0.00% |
| Style Element | 1 | 0.00% |
| Headlamp | 1 | 0.00% |
| stock chart indicator | 1 | 0.00% |
| bathroom fittings | 1 | 0.00% |
| семе | 1 | 0.00% |
| subscriptionId | 1 | 0.00% |
| displayName | 1 | 0.00% |
| membershipState | 1 | 0.00% |
| issuerUrl | 1 | 0.00% |
| workloadIdentityProviders | 1 | 0.00% |
| Animal Name | 1 | 0.00% |
| Animal Species | 1 | 0.00% |
| makeup finish | 1 | 0.00% |
| makeup coverage | 1 | 0.00% |
| Coloring Book | 1 | 0.00% |
| group of physical objects | 1 | 0.00% |
| naming.named_after.name | 1 | 0.00% |
| naming.date | 1 | 0.00% |
| naming.named_after.profession | 1 | 0.00% |
| integration_task_id | 1 | 0.00% |
| ui_bug_status | 1 | 0.00% |
| budget_remaining | 1 | 0.00% |
| security_compliance_delay | 1 | 0.00% |
| backend_integration_status | 1 | 0.00% |
| actual_spend | 1 | 0.00% |
| Feature/Task | 1 | 0.00% |
| Military vehicle | 1 | 0.00% |
| Mining Equipment | 1 | 0.00% |
| Land Parcel/Property | 1 | 0.00% |
| Geographical Location/City | 1 | 0.00% |
| Effect | 1 | 0.00% |
| motorcycle brand | 1 | 0.00% |
| discounted motorcycles | 1 | 0.00% |
| motorcycle activity | 1 | 0.00% |
| landmark/plant garden | 1 | 0.00% |
| country/city | 1 | 0.00% |
| plant garden | 1 | 0.00% |
| competition/event | 1 | 0.00% |
| entertainment industry | 1 | 0.00% |
| emotions/feelings | 1 | 0.00% |
| personal data/technology | 1 | 0.00% |
| theatrical production | 1 | 0.00% |
| Communication medium | 1 | 0.00% |
| pricing attribute | 1 | 0.00% |
| Online marketplace | 1 | 0.00% |
| Antenna | 1 | 0.00% |
| Frequency Range | 1 | 0.00% |
| Volume number | 1 | 0.00% |
| Page range | 1 | 0.00% |
| Hospital/Clinic | 1 | 0.00% |
| exam_date | 1 | 0.00% |
| Animal Shelter | 1 | 0.00% |
| historical record | 1 | 0.00% |
| legal record | 1 | 0.00% |
| road | 1 | 0.00% |
| Sports team fan base | 1 | 0.00% |
| Advertising Program | 1 | 0.00% |
| Trail Name | 1 | 0.00% |
| Map Name | 1 | 0.00% |
| Trail Section | 1 | 0.00% |
| Cell type | 1 | 0.00% |
| Organization unit | 1 | 0.00% |
| Screen Printing Ink | 1 | 0.00% |
| Accommodation Provider | 1 | 0.00% |
| Furniture storage | 1 | 0.00% |
| Wood finish | 1 | 0.00% |
| roles | 1 | 0.00% |
| group of three people | 1 | 0.00% |
| beverage ingredient | 1 | 0.00% |
| Exercise Equipment | 1 | 0.00% |
| Muscle Group | 1 | 0.00% |
| office supplies | 1 | 0.00% |
| Anthology Title | 1 | 0.00% |
| website/service | 1 | 0.00% |
| research program | 1 | 0.00% |
| Individuals who have experienced past sexual abuse | 1 | 0.00% |
| Project Reference Number | 1 | 0.00% |
| Object/Location | 1 | 0.00% |
| Musician/Producer | 1 | 0.00% |
| businessType | 1 | 0.00% |
| licenseType | 1 | 0.00% |
| licenseStatus | 1 | 0.00% |
| renewalDate | 1 | 0.00% |
| TV Series | 1 | 0.00% |
| Government organization | 1 | 0.00% |
| Judiciary body | 1 | 0.00% |
| Manuscript | 1 | 0.00% |
| Sports analytics company | 1 | 0.00% |
| Family Entity | 1 | 0.00% |
| Future Plans Entity | 1 | 0.00% |
| Software Applications | 1 | 0.00% |
| Usage Data | 1 | 0.00% |
| Nutrient application | 1 | 0.00% |
| Nitrogen inhibitors | 1 | 0.00% |
| Nutrient concentration | 1 | 0.00% |
| Plant reproductive structure | 1 | 0.00% |
| Nutrient source | 1 | 0.00% |
| Agricultural ecosystem | 1 | 0.00% |
| Ecological effect | 1 | 0.00% |
| Architecture/Building Design | 1 | 0.00% |
| Chef/Person | 1 | 0.00% |
| pet food packaging | 1 | 0.00% |
| identification labels | 1 | 0.00% |
| collectible cards | 1 | 0.00% |
| Data grouping | 1 | 0.00% |
| Data sets | 1 | 0.00% |
| fauna | 1 | 0.00% |
| stationary item | 1 | 0.00% |
| corporation | 1 | 0.00% |
| business plan | 1 | 0.00% |
| Web Design | 1 | 0.00% |
| Student Organization | 1 | 0.00% |
| Honor Society | 1 | 0.00% |
| Humanitarian Organization | 1 | 0.00% |
| Non-profit Organization | 1 | 0.00% |
| Network hardware | 1 | 0.00% |
| weight measurement | 1 | 0.00% |
| User Experience | 1 | 0.00% |
| User Group | 1 | 0.00% |
| Clothing measurement | 1 | 0.00% |
| Price quality | 1 | 0.00% |
| living situation/option | 1 | 0.00% |
| memorable experience | 1 | 0.00% |
| Special Event | 1 | 0.00% |
| Lipid | 1 | 0.00% |
| Body fluid | 1 | 0.00% |
| funding opportunity | 1 | 0.00% |
| time limit | 1 | 0.00% |
| identification numbers | 1 | 0.00% |
| funding opportunities | 1 | 0.00% |
| landscaping | 1 | 0.00% |
| Festival/Celebration/Social Media | 1 | 0.00% |
| lastUpdatedDate | 1 | 0.00% |
| kitchen sink | 1 | 0.00% |
| educational method | 1 | 0.00% |
| length measurement | 1 | 0.00% |
| Positive outcome | 1 | 0.00% |
| Sports score | 1 | 0.00% |
| Sports ranking | 1 | 0.00% |
| Possibility | 1 | 0.00% |
| Sales tactics | 1 | 0.00% |
| Parent Company | 1 | 0.00% |
| Insurance type | 1 | 0.00% |
| Manufacturing Location | 1 | 0.00% |
| Warranty Requirements | 1 | 0.00% |
| demographics.elaboration | 1 | 0.00% |
| demographics.source | 1 | 0.00% |
| psychological trait | 1 | 0.00% |
| task category | 1 | 0.00% |
| Military Vessel | 1 | 0.00% |
| Job Recruitment | 1 | 0.00% |
| Event Theme | 1 | 0.00% |
| Library Branch | 1 | 0.00% |
| Library System | 1 | 0.00% |
| legal consequence | 1 | 0.00% |
| action/item | 1 | 0.00% |
| payment/remuneration | 1 | 0.00% |
| Product/ Material | 1 | 0.00% |
| residential establishment | 1 | 0.00% |
| medical establishment | 1 | 0.00% |
| conjunction | 1 | 0.00% |
| Apartment Community | 1 | 0.00% |
| Street Intersection | 1 | 0.00% |
| general public | 1 | 0.00% |
| compromise | 1 | 0.00% |
| important information | 1 | 0.00% |
| content theme | 1 | 0.00% |
| music groups | 1 | 0.00% |
| false information | 1 | 0.00% |
| authentication | 1 | 0.00% |
| commitment | 1 | 0.00% |
| Social Relation | 1 | 0.00% |
| Geographic entity | 1 | 0.00% |
| Construction | 1 | 0.00% |
| Large areas | 1 | 0.00% |
| Investment Funds | 1 | 0.00% |
| approach | 1 | 0.00% |
| Job Provider | 1 | 0.00% |
| Business Service Provider | 1 | 0.00% |
| level of experience | 1 | 0.00% |
| Charity/Non-profit organization | 1 | 0.00% |
| Transaction Monitoring | 1 | 0.00% |
| Flower Shop | 1 | 0.00% |
| sound type | 1 | 0.00% |
| role/type | 1 | 0.00% |
| pension limit | 1 | 0.00% |
| financial penalty | 1 | 0.00% |
| Service/Company | 1 | 0.00% |
| Job Title/Service | 1 | 0.00% |
| moment | 1 | 0.00% |
| question | 1 | 0.00% |
| natural landscape | 1 | 0.00% |
| ceremony | 1 | 0.00% |
| location indicator | 1 | 0.00% |
| Job category | 1 | 0.00% |
| Task or Assignment | 1 | 0.00% |
| Housing community | 1 | 0.00% |
| Housing unit | 1 | 0.00% |
| Planning Tool | 1 | 0.00% |
| Loan Term | 1 | 0.00% |
| Monthly Payment | 1 | 0.00% |
| Occasional Payment | 1 | 0.00% |
| No Payment | 1 | 0.00% |
| Protein structure | 1 | 0.00% |
| Protein function | 1 | 0.00% |
| Candidate | 1 | 0.00% |
| Election | 1 | 0.00% |
| reviewer_location | 1 | 0.00% |
| Magazine Title | 1 | 0.00% |
| Makeup Product | 1 | 0.00% |
| Beauty Result | 1 | 0.00% |
| videos | 1 | 0.00% |
| Traffic signal | 1 | 0.00% |
| Program/Service | 1 | 0.00% |
| Hormones | 1 | 0.00% |
| Muscles | 1 | 0.00% |
| holder_name | 1 | 0.00% |
| effective_date | 1 | 0.00% |
| Outdoor Activity | 1 | 0.00% |
| ticket/entry | 1 | 0.00% |
| activity/workshop | 1 | 0.00% |
| building structures | 1 | 0.00% |
| unfinished work | 1 | 0.00% |
| disposalMethod | 1 | 0.00% |
| reasonForDisposal | 1 | 0.00% |
| Pronoun referring to the Lord | 1 | 0.00% |
| Service/Role | 1 | 0.00% |
| Attribute/Quality | 1 | 0.00% |
| Right | 1 | 0.00% |
| cat breed | 1 | 0.00% |
| Printable calendar | 1 | 0.00% |
| Charitable Organizations | 1 | 0.00% |
| Foot position | 1 | 0.00% |
| recipe name | 1 | 0.00% |
| strategy type | 1 | 0.00% |
| Loan Applicants | 1 | 0.00% |
| Financial Obligation | 1 | 0.00% |
| Financial Assistance | 1 | 0.00% |
| Beneficial Action | 1 | 0.00% |
| Supplementary Debt | 1 | 0.00% |
| Loan Acceptance | 1 | 0.00% |
| Natural Formation | 1 | 0.00% |
| Natural Process | 1 | 0.00% |
| age range | 1 | 0.00% |
| productionLineTypeId | 1 | 0.00% |
| Floor Protection | 1 | 0.00% |
| techniques | 1 | 0.00% |
| demographics | 1 | 0.00% |
| description: Brief text describing the content of the post | 1 | 0.00% |
| user_id: Unique identifier for the user on TikTok | 1 | 0.00% |
| status: Current status of the trend repost | 1 | 0.00% |
| shares: Number of shares the post received | 1 | 0.00% |
| user_name: Name of the user who created the repost | 1 | 0.00% |
| trend_name: Name of the viral trend being reposted | 1 | 0.00% |
| Academic institution | 1 | 0.00% |
| ISBN number | 1 | 0.00% |
| Book length | 1 | 0.00% |
| Product component | 1 | 0.00% |
| art/architecture | 1 | 0.00% |
| construction/materials | 1 | 0.00% |
| infrastructure/industry | 1 | 0.00% |
| personal skill | 1 | 0.00% |
| Product/Software | 1 | 0.00% |
| Topic/Field | 1 | 0.00% |
| Crowdfunding Platform | 1 | 0.00% |
| Funding Target | 1 | 0.00% |
| Funding Status | 1 | 0.00% |
| waste disposal container | 1 | 0.00% |
| community area | 1 | 0.00% |
| limitation or prohibition | 1 | 0.00% |
| Ethiopian food | 1 | 0.00% |
| Blog post category | 1 | 0.00% |
| consumer lifestyle | 1 | 0.00% |
| market opportunity | 1 | 0.00% |
| potential market | 1 | 0.00% |
| natural event | 1 | 0.00% |
| automobile part | 1 | 0.00% |
| educational demographic | 1 | 0.00% |
| racial identity | 1 | 0.00% |
| Ticket Vendor | 1 | 0.00% |
| libraries | 1 | 0.00% |
| campuses | 1 | 0.00% |
| indie film and TV label | 1 | 0.00% |
| potential customer | 1 | 0.00% |
| attitude | 1 | 0.00% |
| Health outcome | 1 | 0.00% |
| Research study | 1 | 0.00% |
| Future possibility | 1 | 0.00% |
| risk factor | 1 | 0.00% |
| Deal | 1 | 0.00% |
| Video editing platform | 1 | 0.00% |
| hair style element | 1 | 0.00% |
| military base | 1 | 0.00% |
| military leader | 1 | 0.00% |
| military organization | 1 | 0.00% |
| military station | 1 | 0.00% |
| program/course | 1 | 0.00% |
| gender-neutral pronoun | 1 | 0.00% |
| male pronoun | 1 | 0.00% |
| female pronoun | 1 | 0.00% |
| transaction_id: Unique identifier for each transaction | 1 | 0.00% |
| customer_name: Name of the customer making the purchase | 1 | 0.00% |
| customer_address: Full address of the customer | 1 | 0.00% |
| amount: Total charge amount in dollars | 1 | 0.00% |
| transaction_date: Date when the transaction occurred | 1 | 0.00% |
| store_name: Name of the store where purchase was made | 1 | 0.00% |
| store_address: Address of the store location | 1 | 0.00% |
| status: Payment transaction status | 1 | 0.00% |
| category: Category code for the purchase type | 1 | 0.00% |
| description: Detailed description of the purchased item or service | 1 | 0.00% |
| deal | 1 | 0.00% |
| medical condition/personal struggle | 1 | 0.00% |
| action/strategy | 1 | 0.00% |
| Music app | 1 | 0.00% |
| Group/Quantity | 1 | 0.00% |
| Promotion/Discount | 1 | 0.00% |
| Writing style | 1 | 0.00% |
| qualifier | 1 | 0.00% |
| software license | 1 | 0.00% |
| Time/Meal | 1 | 0.00% |
| name: Unique identifier of the treaty agreement | 1 | 0.00% |
| funding_type: Description of funding commitment such as operational or adaptation | 1 | 0.00% |
| location: City where the treaty was signed | 1 | 0.00% |
| trainingId | 1 | 0.00% |
| employeeName | 1 | 0.00% |
| trainingName | 1 | 0.00% |
| materialName | 1 | 0.00% |
| Error Code | 1 | 0.00% |
| storage | 1 | 0.00% |
| medical outcome | 1 | 0.00% |
| plastic bottle brand | 1 | 0.00% |
| storage units | 1 | 0.00% |
| components | 1 | 0.00% |
| Date and time | 1 | 0.00% |
| githubrepourl | 1 | 0.00% |
| linkedinposturl | 1 | 0.00% |
| living situation | 1 | 0.00% |
| Accessibility feature | 1 | 0.00% |
| reference management | 1 | 0.00% |
| citation style | 1 | 0.00% |
| citation format | 1 | 0.00% |
| customer type | 1 | 0.00% |
| commercial property | 1 | 0.00% |
| Magazine/Publication | 1 | 0.00% |
| Motivation | 1 | 0.00% |
| Vegetation | 1 | 0.00% |
| merchandise | 1 | 0.00% |
| Threshold | 1 | 0.00% |
| media player software | 1 | 0.00% |
| self-identity | 1 | 0.00% |
| job status | 1 | 0.00% |
| Cuisine Type | 1 | 0.00% |
| Scenic View | 1 | 0.00% |
| asteroid.orbital_period.orbiting | 1 | 0.00% |
| asteroid.alternative_name.german | 1 | 0.00% |
| game piece | 1 | 0.00% |
| game action | 1 | 0.00% |
| cell line/type | 1 | 0.00% |
| Education/Training Program | 1 | 0.00% |
| Exam Preparation Material | 1 | 0.00% |
| monitoring | 1 | 0.00% |
| Humor | 1 | 0.00% |
| sin | 1 | 0.00% |
| mortality | 1 | 0.00% |
| divine favor | 1 | 0.00% |
| salvation | 1 | 0.00% |
| savior | 1 | 0.00% |
| Height measurement | 1 | 0.00% |
| Width measurement | 1 | 0.00% |
| Depth measurement | 1 | 0.00% |
| batteryorfeature | 1 | 0.00% |
| upscale ambiance | 1 | 0.00% |
| outdoor fabrics | 1 | 0.00% |
| flight maneuver | 1 | 0.00% |
| aviation training | 1 | 0.00% |
| computer command | 1 | 0.00% |
| computing space | 1 | 0.00% |
| weight loss goal | 1 | 0.00% |
| Education and Finance | 1 | 0.00% |
| location: Office location where the visit occurred | 1 | 0.00% |
| treatment: Therapeutic approach prescribed | 1 | 0.00% |
| provider: Name of the physician who treated the patient | 1 | 0.00% |
| medication: Specific medication prescribed with dosage | 1 | 0.00% |
| Dachshund | 1 | 0.00% |
| Quality/Experience | 1 | 0.00% |
| Electronic component | 1 | 0.00% |
| orderID | 1 | 0.00% |
| productID | 1 | 0.00% |
| Trade Agreement | 1 | 0.00% |
| Fitness Facility | 1 | 0.00% |
| Facility Amenity | 1 | 0.00% |
| person/designer | 1 | 0.00% |
| cheese | 1 | 0.00% |
| Age range | 1 | 0.00% |
| subscription type | 1 | 0.00% |
| viewing allowance | 1 | 0.00% |
| uncertain quantity | 1 | 0.00% |
| Order Entity | 1 | 0.00% |
| Botanists | 1 | 0.00% |
| Botanist | 1 | 0.00% |
| Play/Musical | 1 | 0.00% |
| Playwright/Author | 1 | 0.00% |
| Music Chart | 1 | 0.00% |
| Web technology | 1 | 0.00% |
| User role | 1 | 0.00% |
| Fashion Collection Name | 1 | 0.00% |
| venue space | 1 | 0.00% |
| Move/Ability | 1 | 0.00% |
| Pokémon Type | 1 | 0.00% |
| website directory | 1 | 0.00% |
| healthcare model | 1 | 0.00% |
| Story title | 1 | 0.00% |
| group of characters | 1 | 0.00% |
| positive attitude | 1 | 0.00% |
| Part of Disneyland | 1 | 0.00% |
| Residential Complex | 1 | 0.00% |
| Business/Corporate Event | 1 | 0.00% |
| list item | 1 | 0.00% |
| data cleaning | 1 | 0.00% |
| security threat | 1 | 0.00% |
| data element | 1 | 0.00% |
| availabilityDate | 1 | 0.00% |
| availabilityStatus | 1 | 0.00% |
| storeLocation | 1 | 0.00% |
| storePhone | 1 | 0.00% |
| quantityAvailable | 1 | 0.00% |
| estimatedShipDate | 1 | 0.00% |
| sense perception | 1 | 0.00% |
| Security Risk | 1 | 0.00% |
| Material Object | 1 | 0.00% |
| positive outcome | 1 | 0.00% |
| sports stadium | 1 | 0.00% |
| Reviewer | 1 | 0.00% |
| Brothel Name | 1 | 0.00% |
| metricType | 1 | 0.00% |
| TV show cast | 1 | 0.00% |
| Advertisement material | 1 | 0.00% |
| Business Management Approach | 1 | 0.00% |
| Cultural Diversity | 1 | 0.00% |
| lightbulb | 1 | 0.00% |
| power rating | 1 | 0.00% |
| voltage rating | 1 | 0.00% |
| lifespan | 1 | 0.00% |
| Course Difficulty | 1 | 0.00% |
| Course Enjoyment | 1 | 0.00% |
| Research Size | 1 | 0.00% |
| positive outlook | 1 | 0.00% |
| disintegration | 1 | 0.00% |
| Music product | 1 | 0.00% |
| Political Position | 1 | 0.00% |
| Proposed Legislation | 1 | 0.00% |
| Mobile Game | 1 | 0.00% |
| utility service | 1 | 0.00% |
| Travel Destination | 1 | 0.00% |
| Medical substances | 1 | 0.00% |
| Attire | 1 | 0.00% |
| event/competition | 1 | 0.00% |
| terrain/location | 1 | 0.00% |
| terrain/challenge | 1 | 0.00% |
| service/amenity | 1 | 0.00% |
| event/cause | 1 | 0.00% |
| Person/Expectant Mother | 1 | 0.00% |
| Object/Keepsake | 1 | 0.00% |
| Person/Unborn Child | 1 | 0.00% |
| Person/Expectant Parents | 1 | 0.00% |
| Person/Male Child | 1 | 0.00% |
| Person/Female Child | 1 | 0.00% |
| religious buildings | 1 | 0.00% |
| Kitchen Style | 1 | 0.00% |
| Goal/Aspiration | 1 | 0.00% |
| Task/To-Do | 1 | 0.00% |
| title: Job title as listed in the requisition | 1 | 0.00% |
| statement | 1 | 0.00% |
| Product ID | 1 | 0.00% |
| Product version | 1 | 0.00% |
| Image quality | 1 | 0.00% |
| Software program | 1 | 0.00% |
| Software library | 1 | 0.00% |
| Measurement criterion | 1 | 0.00% |
| website_name | 1 | 0.00% |
| data file type | 1 | 0.00% |
| data storage service | 1 | 0.00% |
| professional designation | 1 | 0.00% |
| professional recognition | 1 | 0.00% |
| Entity Type: Trauma Victim | 1 | 0.00% |
| Metaphorical entity | 1 | 0.00% |
| knowledge acquisition | 1 | 0.00% |
| Property Damage | 1 | 0.00% |
| Cruise ship | 1 | 0.00% |
| Furniture Brand | 1 | 0.00% |
| Modal Verb | 1 | 0.00% |
| religious artifact | 1 | 0.00% |
| military threat | 1 | 0.00% |
| users | 1 | 0.00% |
| content creators | 1 | 0.00% |
| Propaganda | 1 | 0.00% |
| Spy | 1 | 0.00% |
| Former Spy | 1 | 0.00% |
| Coin | 1 | 0.00% |
| Metal composition | 1 | 0.00% |
| Mint | 1 | 0.00% |
| Coin series | 1 | 0.00% |
| Seasonal Plant | 1 | 0.00% |
| Landscaping Element | 1 | 0.00% |
| Seasonal Element | 1 | 0.00% |
| subscription box | 1 | 0.00% |
| individual person/patient | 1 | 0.00% |
| individual characteristic | 1 | 0.00% |
| inclination | 1 | 0.00% |
| neglect | 1 | 0.00% |
| sabotage | 1 | 0.00% |
| Soft skill | 1 | 0.00% |
| Land ownership and management | 1 | 0.00% |
| Flooring | 1 | 0.00% |
| verb phrase | 1 | 0.00% |
| historical event/entity | 1 | 0.00% |
| personal choice | 1 | 0.00% |
| immigration status | 1 | 0.00% |
| immigration document | 1 | 0.00% |
| Quality/Adjective | 1 | 0.00% |
| animal anatomy | 1 | 0.00% |
| Artist Name | 1 | 0.00% |
| Text Content | 1 | 0.00% |
| physical_characteristics.observations.0.year_orbited | 1 | 0.00% |
| Data Size | 1 | 0.00% |
| Digital Objects | 1 | 0.00% |
| Family name | 1 | 0.00% |
| Language/Writing System | 1 | 0.00% |
| punctuation mark | 1 | 0.00% |
| entertainment content | 1 | 0.00% |
| system configuration file | 1 | 0.00% |
| Geographical group | 1 | 0.00% |
| boulder.location.relative_position | 1 | 0.00% |
| boulder.location.distance | 1 | 0.00% |
| boulder.location.distance_unit | 1 | 0.00% |
| boulder.location.direction | 1 | 0.00% |
| boulder.location.reference_point | 1 | 0.00% |
| boulder.start.hand_holds.0.hand | 1 | 0.00% |
| boulder.start.hand_holds.0.description | 1 | 0.00% |
| boulder.start.hand_holds.1.hand | 1 | 0.00% |
| boulder.start.hand_holds.1.description | 1 | 0.00% |
| boulder.start.foot_holds.0.foot | 1 | 0.00% |
| boulder.start.foot_holds.0.description | 1 | 0.00% |
| boulder.start.foot_holds.1.foot | 1 | 0.00% |
| boulder.start.foot_holds.1.description | 1 | 0.00% |
| boulder.moves | 1 | 0.00% |
| submission.submitted_by | 1 | 0.00% |
| submission.submission_date | 1 | 0.00% |
| route_id | 1 | 0.00% |
| commemoration | 1 | 0.00% |
| memorial design | 1 | 0.00% |
| material well-being | 1 | 0.00% |
| Nickname/Place | 1 | 0.00% |
| publishing company | 1 | 0.00% |
| library system | 1 | 0.00% |
| Self-Reference | 1 | 0.00% |
| Apartment complex | 1 | 0.00% |
| Tool/Technology | 1 | 0.00% |
| Greeting cards | 1 | 0.00% |
| Text-based media | 1 | 0.00% |
| Building/Architecture/Religion | 1 | 0.00% |
| punishment | 1 | 0.00% |
| type of glass | 1 | 0.00% |
| animal friend | 1 | 0.00% |
| financial outcome | 1 | 0.00% |
| technical analysis tool | 1 | 0.00% |
| weight loss program | 1 | 0.00% |
| Event/Service | 1 | 0.00% |
| diagnosis: Medical condition coded using ICD-10 standard | 1 | 0.00% |
| claim_id: Unique identifier for the insurance claim | 1 | 0.00% |
| address: Full residential address of the patient | 1 | 0.00% |
| urban development | 1 | 0.00% |
| machine type | 1 | 0.00% |
| price entity | 1 | 0.00% |
| selfie entity | 1 | 0.00% |
| happiness entity | 1 | 0.00% |
| city.birthplace_of.0.person.name | 1 | 0.00% |
| skin issue | 1 | 0.00% |
| skin outcome | 1 | 0.00% |
| Distance and mode of transportation | 1 | 0.00% |
| Money raised | 1 | 0.00% |
| Meal/Activity | 1 | 0.00% |
| Organization/Location | 1 | 0.00% |
| key_challenge | 1 | 0.00% |
| founding_year | 1 | 0.00% |
| Camera | 1 | 0.00% |
| Financial Planning | 1 | 0.00% |
| education institution | 1 | 0.00% |
| mode of education | 1 | 0.00% |
| Transactional Item | 1 | 0.00% |
| Physical Well-being | 1 | 0.00% |
| transfer_code | 1 | 0.00% |
| transfer_portal | 1 | 0.00% |
| real estate agent | 1 | 0.00% |
| fuel quality | 1 | 0.00% |
| Dog temperament | 1 | 0.00% |
| Website role | 1 | 0.00% |
| Literary Works | 1 | 0.00% |
| attention | 1 | 0.00% |
| Business Analysis | 1 | 0.00% |
| Written material | 1 | 0.00% |
| Music streaming service | 1 | 0.00% |
| Fan-made ship name | 1 | 0.00% |
| Radio broadcasting service | 1 | 0.00% |
| Function/Action | 1 | 0.00% |
| Social phenomenon | 1 | 0.00% |
| Degree of seriousness | 1 | 0.00% |
| target_audience: Demographic group the product targets | 1 | 0.00% |
| product: Name of the skincare or apparel product | 1 | 0.00% |
| Political Party/Person | 1 | 0.00% |
| Date Time | 1 | 0.00% |
| Hosting Plan | 1 | 0.00% |
| Work Behavior | 1 | 0.00% |
| Home Type | 1 | 0.00% |
| Health and Wellness Center | 1 | 0.00% |
| Plant disease | 1 | 0.00% |
| laboratory technique | 1 | 0.00% |
| Wig Style | 1 | 0.00% |
| Education Topic | 1 | 0.00% |
| runtime | 1 | 0.00% |
| Couple | 1 | 0.00% |
| dependencies | 1 | 0.00% |
| Engine Capacity | 1 | 0.00% |
| Revolutions per minute | 1 | 0.00% |
| Torque range | 1 | 0.00% |
| Music Work | 1 | 0.00% |
| Online Community Members | 1 | 0.00% |
| Multimedia Work | 1 | 0.00% |
| payRate | 1 | 0.00% |
| jobDescription | 1 | 0.00% |
| lifestyle immersion | 1 | 0.00% |
| business competition | 1 | 0.00% |
| dealership | 1 | 0.00% |
| working together | 1 | 0.00% |
| web pages | 1 | 0.00% |
| online articles | 1 | 0.00% |
| learning opportunity | 1 | 0.00% |
| driving behavior | 1 | 0.00% |
| fitness activity | 1 | 0.00% |
| Geographic feature | 1 | 0.00% |
| song/tune | 1 | 0.00% |
| video resolution | 1 | 0.00% |
| font type | 1 | 0.00% |
| software output | 1 | 0.00% |
| Occasion/Celebration | 1 | 0.00% |
| Hospital/Institution | 1 | 0.00% |
| Clinic/Institution | 1 | 0.00% |
| gardening technique | 1 | 0.00% |
| virus protein | 1 | 0.00% |
| cell surface receptor | 1 | 0.00% |
| scientific journal | 1 | 0.00% |
| Trait/State of being | 1 | 0.00% |
| insurance plan | 1 | 0.00% |
| digital marketing | 1 | 0.00% |
| time_management | 1 | 0.00% |
| overall_score | 1 | 0.00% |
| Game Task | 1 | 0.00% |
| town/city | 1 | 0.00% |
| History | 1 | 0.00% |
| social environment | 1 | 0.00% |
| Motor | 1 | 0.00% |
| field type | 1 | 0.00% |
| event handler type | 1 | 0.00% |
| dataset type | 1 | 0.00% |
| Building/Property Name | 1 | 0.00% |
| Quality/Characteristic | 1 | 0.00% |
| occupational health service | 1 | 0.00% |
| Status/Position | 1 | 0.00% |
| First name | 1 | 0.00% |
| First and Middle name | 1 | 0.00% |
| Convenience/Experience | 1 | 0.00% |
| Fabric characteristic | 1 | 0.00% |
| celestial event | 1 | 0.00% |
| astrological sign | 1 | 0.00% |
| lunar event | 1 | 0.00% |
| Entity Type: Tool/Functionality | 1 | 0.00% |
| Communication/Action | 1 | 0.00% |
| Reflection | 1 | 0.00% |
| Evaluation Score with Fractional Value | 1 | 0.00% |
| Number of Votes | 1 | 0.00% |
| agency | 1 | 0.00% |
| Chemical Property | 1 | 0.00% |
| Chemical Synthesis Method | 1 | 0.00% |
| booking status | 1 | 0.00% |
| Event Outcome | 1 | 0.00% |
| reference | 1 | 0.00% |
| Computer model | 1 | 0.00% |
| Error code | 1 | 0.00% |
| real estate settlement services | 1 | 0.00% |
| Entity Type: Aspiring Authors | 1 | 0.00% |
| medical data | 1 | 0.00% |
| mechanical parts | 1 | 0.00% |
| warning system components | 1 | 0.00% |
| insurance service | 1 | 0.00% |
| preposition | 1 | 0.00% |
| hair extensions | 1 | 0.00% |
| hair extension | 1 | 0.00% |
| sleep aid | 1 | 0.00% |
| Customer Data | 1 | 0.00% |
| Jockey | 1 | 0.00% |
| Racehorse | 1 | 0.00% |
| Horse Racing Event Level | 1 | 0.00% |
| Achievement/Event | 1 | 0.00% |
| Plan/Goal | 1 | 0.00% |
| Action quality | 1 | 0.00% |
| Blogger | 1 | 0.00% |
| location: Venue name and city with state for the festival | 1 | 0.00% |
| name: Name of the festival as listed in the schedule | 1 | 0.00% |
| headliner: Main performing artist for the festival | 1 | 0.00% |
| textile design | 1 | 0.00% |
| Food/Restaurant | 1 | 0.00% |
| Bar/Alcohol | 1 | 0.00% |
| NFL Player | 1 | 0.00% |
| designer/artist | 1 | 0.00% |
| product collection | 1 | 0.00% |
| design industry | 1 | 0.00% |
| Audio settings | 1 | 0.00% |
| Audio feature | 1 | 0.00% |
| Scale model | 1 | 0.00% |
| Medical Personnel | 1 | 0.00% |
| symbol/mark | 1 | 0.00% |
| response/communication | 1 | 0.00% |
| Hair Color Chart | 1 | 0.00% |
| Hair Color Options | 1 | 0.00% |
| Hair Dye Brand | 1 | 0.00% |
| Hair Growth Ideas | 1 | 0.00% |
| Hair Dye Chart | 1 | 0.00% |
| Service/Transportation | 1 | 0.00% |
| specialization | 1 | 0.00% |
| academic degree | 1 | 0.00% |
| state of matter | 1 | 0.00% |
| notice_date | 1 | 0.00% |
| premium | 1 | 0.00% |
| Label | 1 | 0.00% |
| protein/enzyme | 1 | 0.00% |
| visit_reason | 1 | 0.00% |
| pain_level | 1 | 0.00% |
| cultural exploration | 1 | 0.00% |
| luxury amenities | 1 | 0.00% |
| Movie Record | 1 | 0.00% |
| Political district | 1 | 0.00% |
| transgender pioneer | 1 | 0.00% |
| objects/furniture | 1 | 0.00% |
| artifacts | 1 | 0.00% |
| historical era | 1 | 0.00% |
| Race Event | 1 | 0.00% |
| Plant cultivar | 1 | 0.00% |
| technology type | 1 | 0.00% |
| Duration of Employment | 1 | 0.00% |
| product/model | 1 | 0.00% |
| product/type | 1 | 0.00% |
| measurement/unit | 1 | 0.00% |
| organ/tissue | 1 | 0.00% |
| type of legume | 1 | 0.00% |
| type of fruit | 1 | 0.00% |
| Congregation | 1 | 0.00% |
| totalScore | 1 | 0.00% |
| Timespan | 1 | 0.00% |
| Rental Property | 1 | 0.00% |
| Scheduling | 1 | 0.00% |
| Reporting | 1 | 0.00% |
| contractId | 1 | 0.00% |
| salesperson | 1 | 0.00% |
| Load balancer | 1 | 0.00% |
| Visual Representation | 1 | 0.00% |
| unit of energy | 1 | 0.00% |
| physical change | 1 | 0.00% |
| Legal Settlement | 1 | 0.00% |
| Traffic Violation | 1 | 0.00% |
| state of mind | 1 | 0.00% |
| omen | 1 | 0.00% |
| community connection | 1 | 0.00% |
| transportation companies | 1 | 0.00% |
| decision making | 1 | 0.00% |
| Assignment | 1 | 0.00% |
| color options | 1 | 0.00% |
| software versions | 1 | 0.00% |
| Film franchise | 1 | 0.00% |
| Accident Type | 1 | 0.00% |
| Plot Points | 1 | 0.00% |
| Character Development | 1 | 0.00% |
| Character Attribute | 1 | 0.00% |
| completionDate | 1 | 0.00% |
| Game element | 1 | 0.00% |
| Video | 1 | 0.00% |
| Digital asset | 1 | 0.00% |
| Recommendation | 1 | 0.00% |
| anticipation | 1 | 0.00% |
| Currency/price | 1 | 0.00% |
| Event/result | 1 | 0.00% |
| Camera Equipment | 1 | 0.00% |
| Decision making/Advice | 1 | 0.00% |
| Product Warranty | 1 | 0.00% |
| Vehicle Model and Year | 1 | 0.00% |
| computer network | 1 | 0.00% |
| gift/flower arrangement | 1 | 0.00% |
| Academic Support | 1 | 0.00% |
| artistic medium | 1 | 0.00% |
| Transportation system | 1 | 0.00% |
| Animal shelter | 1 | 0.00% |
| task/domain | 1 | 0.00% |
| atmosphere/style | 1 | 0.00% |
| clothing detail | 1 | 0.00% |
| Features | 1 | 0.00% |

#### val

| label | count | share |
| --- | --- | --- |
| e_1 | 442 | 6.12% |
| e_2 | 433 | 5.99% |
| e_0 | 432 | 5.98% |
| e_3 | 417 | 5.77% |
| e_4 | 371 | 5.13% |
| e_5 | 322 | 4.46% |
| e_6 | 264 | 3.65% |
| e_7 | 257 | 3.56% |
| e_8 | 207 | 2.87% |
| e_9 | 169 | 2.34% |
| StartDate | 166 | 2.30% |
| e_10 | 149 | 2.06% |
| EquityHolder | 146 | 2.02% |
| StockCode | 126 | 1.74% |
| e_11 | 118 | 1.63% |
| e_12 | 116 | 1.61% |
| StockAbbr | 100 | 1.38% |
| CompanyName | 100 | 1.38% |
| PledgedShares | 94 | 1.30% |
| TradedShares | 87 | 1.20% |
| e_13 | 83 | 1.15% |
| TotalHoldingShares | 78 | 1.08% |
| Pledgee | 77 | 1.07% |
| TotalHoldingRatio | 69 | 0.96% |
| Location | 69 | 0.96% |
| e_14 | 68 | 0.94% |
| e_15 | 57 | 0.79% |
| TotalPledgedShares | 55 | 0.76% |
| ReleasedDate | 51 | 0.71% |
| e_16 | 44 | 0.61% |
| EndDate | 43 | 0.60% |
| e_17 | 40 | 0.55% |
| Reduction Amount | 40 | 0.55% |
| Person | 38 | 0.53% |
| LaterHoldingShares | 34 | 0.47% |
| e_18 | 34 | 0.47% |
| Date | 33 | 0.46% |
| Casualties and Losses | 26 | 0.36% |
| e_19 | 26 | 0.36% |
| Reducing Shareholder | 25 | 0.35% |
| AveragePrice | 22 | 0.30% |
| Company Name | 22 | 0.30% |
| action | 21 | 0.29% |
| e_20 | 20 | 0.28% |
| Reduction Start Date | 20 | 0.28% |
| Organization | 17 | 0.24% |
| e_21 | 16 | 0.22% |
| person | 14 | 0.19% |
| Announcement Date | 14 | 0.19% |
| Event | 14 | 0.19% |
| medical condition | 13 | 0.18% |
| Time | 13 | 0.18% |
| body part | 12 | 0.17% |
| Object | 12 | 0.17% |
| location | 12 | 0.17% |
| Company | 12 | 0.17% |
| time period | 11 | 0.15% |
| Accepting Court | 11 | 0.15% |
| Individual_protein | 11 | 0.15% |
| Cause | 10 | 0.14% |
| organization | 10 | 0.14% |
| Place | 10 | 0.14% |
| pledge_amount | 10 | 0.14% |
| product | 10 | 0.14% |
| Number of Vehicles Involved in the Crash | 10 | 0.14% |
| Product | 9 | 0.12% |
| activity | 9 | 0.12% |
| Action | 9 | 0.12% |
| Recipient | 8 | 0.11% |
| pledgor | 8 | 0.11% |
| Accident Investigator | 8 | 0.11% |
| Technology | 8 | 0.11% |
| Sensory Perception | 8 | 0.11% |
| Entity type | 8 | 0.11% |
| Animal subgroup | 7 | 0.10% |
| e_22 | 7 | 0.10% |
| Charged Crime | 7 | 0.10% |
| Music Artist | 7 | 0.10% |
| Year | 6 | 0.08% |
| event | 6 | 0.08% |
| Country | 6 | 0.08% |
| Auto part | 6 | 0.08% |
| animal | 6 | 0.08% |
| Activity | 6 | 0.08% |
| name | 6 | 0.08% |
| Target | 5 | 0.07% |
| chemical | 5 | 0.07% |
| Damaged Facility | 5 | 0.07% |
| service | 5 | 0.07% |
| ruling_date | 5 | 0.07% |
| Business Entity | 5 | 0.07% |
| Weapon | 5 | 0.07% |
| FrozeShares | 5 | 0.07% |
| LegalInstitution | 5 | 0.07% |
| technology | 5 | 0.07% |
| Service | 5 | 0.07% |
| Contracting Parties | 5 | 0.07% |
| Death Time | 5 | 0.07% |
| senior_personnel | 5 | 0.07% |
| senior_position | 5 | 0.07% |
| food | 5 | 0.07% |
| description | 5 | 0.07% |
| Capabilities | 5 | 0.07% |
| Attacker | 4 | 0.06% |
| occupation | 4 | 0.06% |
| city | 4 | 0.06% |
| protein | 4 | 0.06% |
| food item | 4 | 0.06% |
| Operation | 4 | 0.06% |
| Policy Content | 4 | 0.06% |
| Deliberating Agency | 4 | 0.06% |
| group of people | 4 | 0.06% |
| Protest Reason | 4 | 0.06% |
| Protesters | 4 | 0.06% |
| Increase Amount | 4 | 0.06% |
| Increasing Shareholder | 4 | 0.06% |
| Emotion | 4 | 0.06% |
| Food | 4 | 0.06% |
| Quantity | 4 | 0.06% |
| Geographical Location | 4 | 0.06% |
| The Agency Who Broke the Agreement | 4 | 0.06% |
| e_23 | 4 | 0.06% |
| object | 4 | 0.06% |
| reagent | 4 | 0.06% |
| holiday decoration | 4 | 0.06% |
| Candidates and their Political Parties | 4 | 0.06% |
| Divine being | 4 | 0.06% |
| Frozen Amount | 4 | 0.06% |
| Malware | 4 | 0.06% |
| activity type | 4 | 0.06% |
| Scheduled Landing Place | 3 | 0.04% |
| age group | 3 | 0.04% |
| The Charged Crime | 3 | 0.04% |
| Time period | 3 | 0.04% |
| Accommodation | 3 | 0.04% |
| tool | 3 | 0.04% |
| Educational Institution | 3 | 0.04% |
| Major | 3 | 0.04% |
| measurement unit | 3 | 0.04% |
| profession | 3 | 0.04% |
| Person Name | 3 | 0.04% |
| experiment-tag | 3 | 0.04% |
| Record Breaker | 3 | 0.04% |
| Loss Amount | 3 | 0.04% |
| Entity Type | 3 | 0.04% |
| Affected Areas | 3 | 0.04% |
| attribute | 3 | 0.04% |
| emotion | 3 | 0.04% |
| Time Period | 3 | 0.04% |
| Product/Brand | 3 | 0.04% |
| Increase Start Date | 3 | 0.04% |
| credit card information | 3 | 0.04% |
| Author | 3 | 0.04% |
| address | 3 | 0.04% |
| Medical Practitioner | 3 | 0.04% |
| Judge | 3 | 0.04% |
| Death Age | 3 | 0.04% |
| time | 3 | 0.04% |
| Transportation Service | 3 | 0.04% |
| Legal Document | 3 | 0.04% |
| healthcare service | 3 | 0.04% |
| amenity | 3 | 0.04% |
| Award Reason | 3 | 0.04% |
| weather phenomenon | 3 | 0.04% |
| natural feature | 3 | 0.04% |
| physical space | 3 | 0.04% |
| game equipment | 3 | 0.04% |
| Election Goal | 3 | 0.04% |
| Device | 3 | 0.04% |
| codes | 3 | 0.04% |
| units | 3 | 0.04% |
| code | 3 | 0.04% |
| unit | 3 | 0.04% |
| amount | 3 | 0.04% |
| Tourist Attraction | 3 | 0.04% |
| astronomical object | 3 | 0.04% |
| Scent/Flavor | 3 | 0.04% |
| Freeze Start Date | 3 | 0.04% |
| frozen_shareholder | 3 | 0.04% |
| Measurement Value | 3 | 0.04% |
| Physical Attribute | 3 | 0.04% |
| value | 3 | 0.04% |
| Strategy/Methodology | 3 | 0.04% |
| ingredient | 3 | 0.04% |
| People | 3 | 0.04% |
| Aircraft Agency | 2 | 0.03% |
| Flight No. | 2 | 0.03% |
| Organization Name | 2 | 0.03% |
| demographic group | 2 | 0.03% |
| ethnic group | 2 | 0.03% |
| Group | 2 | 0.03% |
| legal action | 2 | 0.03% |
| social group | 2 | 0.03% |
| property | 2 | 0.03% |
| Suspect | 2 | 0.03% |
| Negative Experience | 2 | 0.03% |
| Day of the week | 2 | 0.03% |
| Ethnic Group | 2 | 0.03% |
| mechanical component | 2 | 0.03% |
| Company/Organization | 2 | 0.03% |
| Media | 2 | 0.03% |
| Region | 2 | 0.03% |
| planting method | 2 | 0.03% |
| Documentation | 2 | 0.03% |
| File Format | 2 | 0.03% |
| Volcano Name | 2 | 0.03% |
| Recipe Instruction | 2 | 0.03% |
| Publication | 2 | 0.03% |
| Natural Feature | 2 | 0.03% |
| Accused People | 2 | 0.03% |
| experimental-construct | 2 | 0.03% |
| Record-breaking Project | 2 | 0.03% |
| Grades | 2 | 0.03% |
| Slot Feature | 2 | 0.03% |
| Input Device | 2 | 0.03% |
| Software | 2 | 0.03% |
| Time/Date | 2 | 0.03% |
| mechanism | 2 | 0.03% |
| Survivors | 2 | 0.03% |
| experience | 2 | 0.03% |
| Insurance Policy | 2 | 0.03% |
| Belligerents | 2 | 0.03% |
| Weapon and Equippment | 2 | 0.03% |
| mode of transportation | 2 | 0.03% |
| service provider | 2 | 0.03% |
| Crafting Material | 2 | 0.03% |
| state | 2 | 0.03% |
| author | 2 | 0.03% |
| Medication | 2 | 0.03% |
| Weather condition | 2 | 0.03% |
| Animal | 2 | 0.03% |
| body of water | 2 | 0.03% |
| medical treatment | 2 | 0.03% |
| personal trait | 2 | 0.03% |
| government agency | 2 | 0.03% |
| group | 2 | 0.03% |
| Chemical compound | 2 | 0.03% |
| technical analysis term | 2 | 0.03% |
| religion | 2 | 0.03% |
| belief system | 2 | 0.03% |
| Agreement Content | 2 | 0.03% |
| topic | 2 | 0.03% |
| Released People | 2 | 0.03% |
| Release Reason | 2 | 0.03% |
| Work Schedule | 2 | 0.03% |
| Medical Service Type | 2 | 0.03% |
| Sport | 2 | 0.03% |
| Status | 2 | 0.03% |
| University | 2 | 0.03% |
| architecture | 2 | 0.03% |
| Damage Type | 2 | 0.03% |
| pronoun | 2 | 0.03% |
| Character | 2 | 0.03% |
| e_24 | 2 | 0.03% |
| e_25 | 2 | 0.03% |
| task/action | 2 | 0.03% |
| pledge_start_date | 2 | 0.03% |
| pledge_end_date | 2 | 0.03% |
| Award | 2 | 0.03% |
| Award Field | 2 | 0.03% |
| Lighting Equipment | 2 | 0.03% |
| Material | 2 | 0.03% |
| Gene | 2 | 0.03% |
| texture/physical feature | 2 | 0.03% |
| hardware | 2 | 0.03% |
| medication | 2 | 0.03% |
| room | 2 | 0.03% |
| Game Name | 2 | 0.03% |
| Start Time | 2 | 0.03% |
| Contest Participant | 2 | 0.03% |
| game rule | 2 | 0.03% |
| Time of Day | 2 | 0.03% |
| Accusation | 2 | 0.03% |
| Media Outlet | 2 | 0.03% |
| Information Service | 2 | 0.03% |
| Financial Effect | 2 | 0.03% |
| employee | 2 | 0.03% |
| type | 2 | 0.03% |
| image | 2 | 0.03% |
| disposition | 2 | 0.03% |
| Company Industry | 2 | 0.03% |
| meal type | 2 | 0.03% |
| Sound | 2 | 0.03% |
| City | 2 | 0.03% |
| parasite | 2 | 0.03% |
| Demographic | 2 | 0.03% |
| Application | 2 | 0.03% |
| nationality | 2 | 0.03% |
| quantity | 2 | 0.03% |
| Art/Culture | 2 | 0.03% |
| Freeze End Date | 2 | 0.03% |
| Army | 2 | 0.03% |
| Military Exercise | 2 | 0.03% |
| Business | 2 | 0.03% |
| Menu Item | 2 | 0.03% |
| parties | 2 | 0.03% |
| System | 2 | 0.03% |
| Money | 2 | 0.03% |
| musical expression | 2 | 0.03% |
| Clothing feature | 2 | 0.03% |
| Person/Group | 2 | 0.03% |
| Football Club | 2 | 0.03% |
| Feature | 2 | 0.03% |
| Interactive element | 2 | 0.03% |
| feeling/emotion | 2 | 0.03% |
| behavior/action | 2 | 0.03% |
| action/behavior | 2 | 0.03% |
| id | 2 | 0.03% |
| game interface | 2 | 0.03% |
| Entertainment | 2 | 0.03% |
| time frame | 2 | 0.03% |
| Aid Agency | 2 | 0.03% |
| Doctor and Medical Team | 2 | 0.03% |
| hair condition | 2 | 0.03% |
| award category | 2 | 0.03% |
| color | 2 | 0.03% |
| Design Project | 2 | 0.03% |
| race event | 2 | 0.03% |
| real estate | 1 | 0.01% |
| animal companions | 1 | 0.01% |
| Animal group | 1 | 0.01% |
| Countries Withdrawing from the Organization | 1 | 0.01% |
| Effective Date | 1 | 0.01% |
| Organization Leader | 1 | 0.01% |
| Company/Brand | 1 | 0.01% |
| Crew | 1 | 0.01% |
| Passengers | 1 | 0.01% |
| conflict | 1 | 0.01% |
| combat | 1 | 0.01% |
| aftermath | 1 | 0.01% |
| enemy | 1 | 0.01% |
| communication | 1 | 0.01% |
| athlete | 1 | 0.01% |
| digital content | 1 | 0.01% |
| punctuation | 1 | 0.01% |
| compensation_recipient | 1 | 0.01% |
| sku_id | 1 | 0.01% |
| last_updated | 1 | 0.01% |
| product_name | 1 | 0.01% |
| Sales Number | 1 | 0.01% |
| Market Condition | 1 | 0.01% |
| Sales Range | 1 | 0.01% |
| Business Term | 1 | 0.01% |
| Opinion | 1 | 0.01% |
| geographic location | 1 | 0.01% |
| Idea/Policy | 1 | 0.01% |
| family relation | 1 | 0.01% |
| legal relation | 1 | 0.01% |
| health practice | 1 | 0.01% |
| design style | 1 | 0.01% |
| features | 1 | 0.01% |
| music | 1 | 0.01% |
| symptom | 1 | 0.01% |
| Police | 1 | 0.01% |
| Criminal Evidence | 1 | 0.01% |
| Government Agency | 1 | 0.01% |
| Scholarship Programme | 1 | 0.01% |
| Time of day | 1 | 0.01% |
| Visual media | 1 | 0.01% |
| Language | 1 | 0.01% |
| Room | 1 | 0.01% |
| Structure | 1 | 0.01% |
| substance | 1 | 0.01% |
| crime | 1 | 0.01% |
| mishap | 1 | 0.01% |
| harm | 1 | 0.01% |
| punishment | 1 | 0.01% |
| service offering | 1 | 0.01% |
| work environment | 1 | 0.01% |
| Education/Training | 1 | 0.01% |
| Digital Platform | 1 | 0.01% |
| Website | 1 | 0.01% |
| Cashless platform | 1 | 0.01% |
| Ride-hailing service | 1 | 0.01% |
| Business/Financial Institution | 1 | 0.01% |
| Continent | 1 | 0.01% |
| negative impact on budget | 1 | 0.01% |
| critical level | 1 | 0.01% |
| Military Strength | 1 | 0.01% |
| type of plant species | 1 | 0.01% |
| packing material | 1 | 0.01% |
| type of material | 1 | 0.01% |
| type of flooring | 1 | 0.01% |
| entertainment | 1 | 0.01% |
| Insurance | 1 | 0.01% |
| Record Keeping | 1 | 0.01% |
| Possession | 1 | 0.01% |
| Analytical Technique | 1 | 0.01% |
| Chemical Analysis | 1 | 0.01% |
| Publisher | 1 | 0.01% |
| plant-based substance | 1 | 0.01% |
| product type | 1 | 0.01% |
| Department | 1 | 0.01% |
| Writing | 1 | 0.01% |
| Outbreak Date | 1 | 0.01% |
| Preparation Step | 1 | 0.01% |
| Time Relation | 1 | 0.01% |
| Music Festival | 1 | 0.01% |
| TV Show | 1 | 0.01% |
| skill/ability | 1 | 0.01% |
| religious practice | 1 | 0.01% |
| communication device | 1 | 0.01% |
| request | 1 | 0.01% |
| donation | 1 | 0.01% |
| Demographic group | 1 | 0.01% |
| facility | 1 | 0.01% |
| frequency | 1 | 0.01% |
| Attribute | 1 | 0.01% |
| primate.morphological_analysis.0.finding | 1 | 0.01% |
| primate.related_primates.0.location | 1 | 0.01% |
| primate.related_primates.0.name | 1 | 0.01% |
| primate.morphological_analysis.0.type | 1 | 0.01% |
| primate.genus | 1 | 0.01% |
| primate.dental_formula | 1 | 0.01% |
| primate.weight_grams | 1 | 0.01% |
| primate.relationships | 1 | 0.01% |
| primate.name | 1 | 0.01% |
| primate.morphological_analysis.1.finding | 1 | 0.01% |
| primate.morphological_analysis.1.type | 1 | 0.01% |
| primate.classification | 1 | 0.01% |
| primate.fossil_origin.location | 1 | 0.01% |
| primate.morphological_analysis.1.genus | 1 | 0.01% |
| vocal expression | 1 | 0.01% |
| Volume Number | 1 | 0.01% |
| Meal Plan | 1 | 0.01% |
| Supplements | 1 | 0.01% |
| Support Team | 1 | 0.01% |
| Debriefing | 1 | 0.01% |
| Assessment | 1 | 0.01% |
| marketing strategy | 1 | 0.01% |
| market research | 1 | 0.01% |
| reference point | 1 | 0.01% |
| Transportation | 1 | 0.01% |
| Evidence | 1 | 0.01% |
| protein-family | 1 | 0.01% |
| protein-domain | 1 | 0.01% |
| Color | 1 | 0.01% |
| Previous Record Holder | 1 | 0.01% |
| Document Title | 1 | 0.01% |
| Document Format | 1 | 0.01% |
| Website Feature | 1 | 0.01% |
| Spatial Location | 1 | 0.01% |
| software | 1 | 0.01% |
| Event Name | 1 | 0.01% |
| Movie Title | 1 | 0.01% |
| Degree | 1 | 0.01% |
| Hardware | 1 | 0.01% |
| Artistic expression | 1 | 0.01% |
| Physical activity | 1 | 0.01% |
| Social activity | 1 | 0.01% |
| Physiological process | 1 | 0.01% |
| organization.personnel_changes.0.individual.name | 1 | 0.01% |
| article_metadata.source | 1 | 0.01% |
| organization.personnel_changes.0.effective_date_description | 1 | 0.01% |
| organization.legal_form | 1 | 0.01% |
| article_metadata.origin_newsroom | 1 | 0.01% |
| organization.full_name | 1 | 0.01% |
| Event Organizer | 1 | 0.01% |
| Countries Joined the Organization | 1 | 0.01% |
| Declarer | 1 | 0.01% |
| product feature | 1 | 0.01% |
| Missings | 1 | 0.01% |
| Number of Damaged Houses | 1 | 0.01% |
| Race/Species | 1 | 0.01% |
| organizational entity | 1 | 0.01% |
| meeting/event | 1 | 0.01% |
| component | 1 | 0.01% |
| activity/event | 1 | 0.01% |
| procedure | 1 | 0.01% |
| company/organization | 1 | 0.01% |
| guestId | 1 | 0.01% |
| firstName | 1 | 0.01% |
| lastName | 1 | 0.01% |
| email | 1 | 0.01% |
| roomId | 1 | 0.01% |
| roomType | 1 | 0.01% |
| capacity | 1 | 0.01% |
| beds | 1 | 0.01% |
| checkoutDate | 1 | 0.01% |
| checkoutTime | 1 | 0.01% |
| note | 1 | 0.01% |
| personality trait | 1 | 0.01% |
| Hazard | 1 | 0.01% |
| Work Cycle | 1 | 0.01% |
| award | 1 | 0.01% |
| Government Reaction | 1 | 0.01% |
| service level agreements | 1 | 0.01% |
| system performance | 1 | 0.01% |
| task load | 1 | 0.01% |
| storage management tasks | 1 | 0.01% |
| systems management integration | 1 | 0.01% |
| software vulnerability | 1 | 0.01% |
| software update | 1 | 0.01% |
| computing system | 1 | 0.01% |
| security mirroring | 1 | 0.01% |
| vehicle | 1 | 0.01% |
| written work | 1 | 0.01% |
| issue | 1 | 0.01% |
| media format | 1 | 0.01% |
| digital interface | 1 | 0.01% |
| customer type | 1 | 0.01% |
| item for sale | 1 | 0.01% |
| work done for others | 1 | 0.01% |
| File format | 1 | 0.01% |
| File extension | 1 | 0.01% |
| Commodity | 1 | 0.01% |
| achievement | 1 | 0.01% |
| financial intermediary | 1 | 0.01% |
| financial institution | 1 | 0.01% |
| Society/Culture | 1 | 0.01% |
| claim_number: Unique identifier for the health claim | 1 | 0.01% |
| patient_name: Full name of the claimant | 1 | 0.01% |
| age: Patient's age in years | 1 | 0.01% |
| address: Patient's residential address | 1 | 0.01% |
| diagnosis: Medical diagnosis code and description | 1 | 0.01% |
| amount: Total cost of the medical service | 1 | 0.01% |
| occupation/role | 1 | 0.01% |
| landlord | 1 | 0.01% |
| zip_code | 1 | 0.01% |
| unit_id | 1 | 0.01% |
| pet_policy | 1 | 0.01% |
| audio book service | 1 | 0.01% |
| Medical Treatment | 1 | 0.01% |
| Amount of people | 1 | 0.01% |
| Transportation route | 1 | 0.01% |
| Alert/Notification | 1 | 0.01% |
| Real estate | 1 | 0.01% |
| Guardian/Protector | 1 | 0.01% |
| Symbol/Indicator | 1 | 0.01% |
| payment method | 1 | 0.01% |
| shooting | 1 | 0.01% |
| safety rules | 1 | 0.01% |
| participants | 1 | 0.01% |
| waterway | 1 | 0.01% |
| UnfrozeDate | 1 | 0.01% |
| Product Name | 1 | 0.01% |
| Software Product | 1 | 0.01% |
| medical facility | 1 | 0.01% |
| date/time | 1 | 0.01% |
| community center | 1 | 0.01% |
| intersection | 1 | 0.01% |
| payment | 1 | 0.01% |
| system | 1 | 0.01% |
| personal growth | 1 | 0.01% |
| athletics | 1 | 0.01% |
| patriotism | 1 | 0.01% |
| family relationship | 1 | 0.01% |
| expression of approval | 1 | 0.01% |
| submission_date | 1 | 0.01% |
| funding | 1 | 0.01% |
| institution | 1 | 0.01% |
| authors | 1 | 0.01% |
| title | 1 | 0.01% |
| Livestock | 1 | 0.01% |
| Economic hardship | 1 | 0.01% |
| Nation | 1 | 0.01% |
| Agricultural products | 1 | 0.01% |
| Government official | 1 | 0.01% |
| exoplanet | 1 | 0.01% |
| planet type | 1 | 0.01% |
| star type | 1 | 0.01% |
| unknown entity | 1 | 0.01% |
| numerical value | 1 | 0.01% |
| travel document | 1 | 0.01% |
| application form | 1 | 0.01% |
| post office | 1 | 0.01% |
| county | 1 | 0.01% |
| Sustainable Energy | 1 | 0.01% |
| Sustainable Resources | 1 | 0.01% |
| Infrastructure | 1 | 0.01% |
| Waste Management | 1 | 0.01% |
| Consulting Services | 1 | 0.01% |
| tax policy | 1 | 0.01% |
| disease | 1 | 0.01% |
| equipment | 1 | 0.01% |
| game item | 1 | 0.01% |
| game location | 1 | 0.01% |
| amount: Monetary value of the transaction | 1 | 0.01% |
| account: Bank account associated with the transaction | 1 | 0.01% |
| description: Brief explanation of the transaction purpose | 1 | 0.01% |
| date: Date when the transaction occurred | 1 | 0.01% |
| song title | 1 | 0.01% |
| claim | 1 | 0.01% |
| Document | 1 | 0.01% |
| financial instrument | 1 | 0.01% |
| TV show | 1 | 0.01% |
| Medical Procedure | 1 | 0.01% |
| Agreement Name | 1 | 0.01% |
| Tear Up Reason | 1 | 0.01% |
| Bank Name | 1 | 0.01% |
| Investigating agency | 1 | 0.01% |
| Weapon Used | 1 | 0.01% |
| Investigating Spokesperson | 1 | 0.01% |
| environment | 1 | 0.01% |
| relationship issue | 1 | 0.01% |
| technology tool | 1 | 0.01% |
| political system | 1 | 0.01% |
| Release Time | 1 | 0.01% |
| form field | 1 | 0.01% |
| input field | 1 | 0.01% |
| contact information | 1 | 0.01% |
| output field | 1 | 0.01% |
| resource limit | 1 | 0.01% |
| Communication | 1 | 0.01% |
| Application Type | 1 | 0.01% |
| Hairstyle | 1 | 0.01% |
| Information | 1 | 0.01% |
| Medical Practice | 1 | 0.01% |
| Geographic Location | 1 | 0.01% |
| book | 1 | 0.01% |
| Sports team | 1 | 0.01% |
| Outcome | 1 | 0.01% |
| brand/product | 1 | 0.01% |
| characteristic | 1 | 0.01% |
| furniture | 1 | 0.01% |
| mental state | 1 | 0.01% |
| product model | 1 | 0.01% |
| media type | 1 | 0.01% |
| store | 1 | 0.01% |
| medication adherence | 1 | 0.01% |
| medication regimen | 1 | 0.01% |
| Currency | 1 | 0.01% |
| Date Range | 1 | 0.01% |
| Business Name | 1 | 0.01% |
| Platform | 1 | 0.01% |
| natural ingredient | 1 | 0.01% |
| animal type | 1 | 0.01% |
| pest | 1 | 0.01% |
| Product/Service | 1 | 0.01% |
| massage technique | 1 | 0.01% |
| holistic practice | 1 | 0.01% |
| Government Body | 1 | 0.01% |
| musical instrument | 1 | 0.01% |
| musical activity | 1 | 0.01% |
| negative change | 1 | 0.01% |
| group of post categories | 1 | 0.01% |
| access level | 1 | 0.01% |
| idea | 1 | 0.01% |
| Game Title | 1 | 0.01% |
| User Interface Element | 1 | 0.01% |
| family member | 1 | 0.01% |
| Game | 1 | 0.01% |
| Literary device | 1 | 0.01% |
| Writing genre | 1 | 0.01% |
| call_details_contact | 1 | 0.01% |
| call_status | 1 | 0.01% |
| age milestone | 1 | 0.01% |
| Areas Affected | 1 | 0.01% |
| Influenced People | 1 | 0.01% |
| accommodation | 1 | 0.01% |
| physical object | 1 | 0.01% |
| decoration | 1 | 0.01% |
| Job Title | 1 | 0.01% |
| Winner | 1 | 0.01% |
| Body Part | 1 | 0.01% |
| NFL Team | 1 | 0.01% |
| Football Position/Unit | 1 | 0.01% |
| Quantity/Group of People | 1 | 0.01% |
| Football Position | 1 | 0.01% |
| Day of the Week | 1 | 0.01% |
| Sports Event | 1 | 0.01% |
| Qualification | 1 | 0.01% |
| Fee | 1 | 0.01% |
| Process | 1 | 0.01% |
| Format | 1 | 0.01% |
| Content | 1 | 0.01% |
| Dish | 1 | 0.01% |
| weather condition | 1 | 0.01% |
| Body Parts | 1 | 0.01% |
| Road/Highway | 1 | 0.01% |
| clothing item | 1 | 0.01% |
| accessory | 1 | 0.01% |
| Sentencing Location | 1 | 0.01% |
| Prison Term | 1 | 0.01% |
| outdoor area | 1 | 0.01% |
| outdoor amenity | 1 | 0.01% |
| visual media | 1 | 0.01% |
| Competition Items | 1 | 0.01% |
| problem_start_time | 1 | 0.01% |
| affected_network | 1 | 0.01% |
| problem_description | 1 | 0.01% |
| Education Level | 1 | 0.01% |
| Religious Activity | 1 | 0.01% |
| Social Group | 1 | 0.01% |
| Topic | 1 | 0.01% |
| Interface Element | 1 | 0.01% |
| Website Component | 1 | 0.01% |
| Exam/Test | 1 | 0.01% |
| Score/Grade | 1 | 0.01% |
| Voting method | 1 | 0.01% |
| Time Span | 1 | 0.01% |
| Programming Language | 1 | 0.01% |
| Judgement Result/Prison Term | 1 | 0.01% |
| Archangel | 1 | 0.01% |
| Percentage | 1 | 0.01% |
| Electoral System | 1 | 0.01% |
| plot element | 1 | 0.01% |
| media | 1 | 0.01% |
| feedback | 1 | 0.01% |
| patient_id: Unique identifier for each patient visit | 1 | 0.01% |
| encounter_date: Date of the patient visit | 1 | 0.01% |
| name: Full name of the patient | 1 | 0.01% |
| diagnosis: Medical code indicating the diagnosed condition | 1 | 0.01% |
| status: Current status of the encounter | 1 | 0.01% |
| e_26 | 1 | 0.01% |
| Occupation | 1 | 0.01% |
| serialNumber | 1 | 0.01% |
| model | 1 | 0.01% |
| contact | 1 | 0.01% |
| notes | 1 | 0.01% |
| submittedBy | 1 | 0.01% |
| dateSubmitted | 1 | 0.01% |
| subscription service | 1 | 0.01% |
| container | 1 | 0.01% |
| Achievement | 1 | 0.01% |
| Industry/Service | 1 | 0.01% |
| animal name | 1 | 0.01% |
| taste | 1 | 0.01% |
| Venue | 1 | 0.01% |
| Address | 1 | 0.01% |
| Children's musical | 1 | 0.01% |
| Price | 1 | 0.01% |
| Physical Injury | 1 | 0.01% |
| lifestyle | 1 | 0.01% |
| situation | 1 | 0.01% |
| capability | 1 | 0.01% |
| restaurant public relations firm | 1 | 0.01% |
| re-branding | 1 | 0.01% |
| executive vice president and chief creative officer | 1 | 0.01% |
| Title of a book or movie | 1 | 0.01% |
| Ship No. | 1 | 0.01% |
| Issue | 1 | 0.01% |
| Investigation Status | 1 | 0.01% |
| search engine result | 1 | 0.01% |
| advertising strategy | 1 | 0.01% |
| review entity | 1 | 0.01% |
| Numeric Value | 1 | 0.01% |
| Time Unit | 1 | 0.01% |
| effectiveDate | 1 | 0.01% |
| expirationDate | 1 | 0.01% |
| result_id | 1 | 0.01% |
| score | 1 | 0.01% |
| Entity | 1 | 0.01% |
| Industry | 1 | 0.01% |
| Companies | 1 | 0.01% |
| Legal case | 1 | 0.01% |
| physical force | 1 | 0.01% |
| Smoking device | 1 | 0.01% |
| Smoke type | 1 | 0.01% |
| Smoking experience | 1 | 0.01% |
| Temperature control | 1 | 0.01% |
| size | 1 | 0.01% |
| insect | 1 | 0.01% |
| anatomy | 1 | 0.01% |
| sense organ | 1 | 0.01% |
| Brand/Product | 1 | 0.01% |
| Ingredient | 1 | 0.01% |
| Product Type | 1 | 0.01% |
| Benefit/Feature | 1 | 0.01% |
| Sport Type | 1 | 0.01% |
| Development/Progress | 1 | 0.01% |
| Organization/Company | 1 | 0.01% |
| Fashion Brand | 1 | 0.01% |
| Country/Geopolitical entity | 1 | 0.01% |
| Time entity | 1 | 0.01% |
| Religious entity | 1 | 0.01% |
| Clothing Brand | 1 | 0.01% |
| Fashion Designer | 1 | 0.01% |
| Clothing Size | 1 | 0.01% |
| Clothing Measurement | 1 | 0.01% |
| Profession/Role | 1 | 0.01% |
| Measurement Value  | 1 | 0.01% |
| genus.family | 1 | 0.01% |
| genus.subspecies.0.name | 1 | 0.01% |
| genus.name_latin | 1 | 0.01% |
| genus.type_species | 1 | 0.01% |
| genus.subspecies.0.authority | 1 | 0.01% |
| Music genre | 1 | 0.01% |
| Service provider | 1 | 0.01% |
| formNumber | 1 | 0.01% |
| destination | 1 | 0.01% |
| travelPurpose | 1 | 0.01% |
| passportNumber | 1 | 0.01% |
| currency | 1 | 0.01% |
| totalValue | 1 | 0.01% |
| signature | 1 | 0.01% |
| Quantity Entity | 1 | 0.01% |
| Occupation Entity | 1 | 0.01% |
| Location Entity | 1 | 0.01% |
| Goal Setting | 1 | 0.01% |
| Personal Development | 1 | 0.01% |
| Coaching Services | 1 | 0.01% |
| Guidance and Support | 1 | 0.01% |
| Event/Tradition | 1 | 0.01% |
| Business/Organization | 1 | 0.01% |
| Group/People | 1 | 0.01% |
| Brand/Product Name | 1 | 0.01% |
| Biological Entities | 1 | 0.01% |
| Substance | 1 | 0.01% |
| Physical Entity | 1 | 0.01% |
| Business Action | 1 | 0.01% |
| ordinal number | 1 | 0.01% |
| Goal | 1 | 0.01% |
| Scale | 1 | 0.01% |
| work ethic | 1 | 0.01% |
| decision-making | 1 | 0.01% |
| outcome | 1 | 0.01% |
| Coach/Recruiter | 1 | 0.01% |
| Sports Ability | 1 | 0.01% |
| Community | 1 | 0.01% |
| art form | 1 | 0.01% |
| age | 1 | 0.01% |
| social media platform | 1 | 0.01% |
| Business Location | 1 | 0.01% |
| Architecture Style | 1 | 0.01% |
| Beverage | 1 | 0.01% |
| reference_number | 1 | 0.01% |
| jurisdiction | 1 | 0.01% |
| clause_type | 1 | 0.01% |
| religious text | 1 | 0.01% |
| media series | 1 | 0.01% |
| educational material | 1 | 0.01% |
| biblical event | 1 | 0.01% |
| evaluation metric | 1 | 0.01% |
| Territory | 1 | 0.01% |
| Skincare product | 1 | 0.01% |
| expense | 1 | 0.01% |
| Measurement | 1 | 0.01% |
| Author/Publisher | 1 | 0.01% |
| Category | 1 | 0.01% |
| Physical Object | 1 | 0.01% |
| Food/Drink | 1 | 0.01% |
| Shopping | 1 | 0.01% |
| Emotion/Feeling | 1 | 0.01% |
| Ability/Skill | 1 | 0.01% |
| Instruction/Teaching | 1 | 0.01% |
| Book/Printed Material | 1 | 0.01% |
| Person/Author | 1 | 0.01% |
| Plant/Flora | 1 | 0.01% |
| Object/Lighting | 1 | 0.01% |
| PII | 1 | 0.01% |
| day of the week | 1 | 0.01% |
| title: Topic or title of the submitted work | 1 | 0.01% |
| decision_status: Final outcome of the submission | 1 | 0.01% |
| group/person | 1 | 0.01% |
| negation | 1 | 0.01% |
| tense | 1 | 0.01% |
| modal verb | 1 | 0.01% |
| Clothing item | 1 | 0.01% |
| Clothing accessory | 1 | 0.01% |
| expiry_date | 1 | 0.01% |
| handling_instructions | 1 | 0.01% |
| manufacturer | 1 | 0.01% |
| ppe_required | 1 | 0.01% |
| travel location | 1 | 0.01% |
| travel cost | 1 | 0.01% |
| travel preference | 1 | 0.01% |
| Athlete | 1 | 0.01% |
| Narrative element | 1 | 0.01% |
| measurement | 1 | 0.01% |
| Business Partners | 1 | 0.01% |
| Business Services | 1 | 0.01% |
| written communication | 1 | 0.01% |
| year | 1 | 0.01% |
| Service/Platform | 1 | 0.01% |
| therapy | 1 | 0.01% |
| practices | 1 | 0.01% |
| disorder | 1 | 0.01% |
| Physical sensation | 1 | 0.01% |
| Mental state | 1 | 0.01% |
| Physical entity | 1 | 0.01% |
| Tourist Destination | 1 | 0.01% |
| Culture | 1 | 0.01% |
| kitchen utensil | 1 | 0.01% |
| temperature | 1 | 0.01% |
| cooking method | 1 | 0.01% |
| Deity | 1 | 0.01% |
| Comparison | 1 | 0.01% |
| batchNumber | 1 | 0.01% |
| reviewDate | 1 | 0.01% |
| reviewComment | 1 | 0.01% |
| reviewStatus | 1 | 0.01% |
| Data | 1 | 0.01% |
| Purpose | 1 | 0.01% |
| injury | 1 | 0.01% |
| dog breed | 1 | 0.01% |
| physical ability | 1 | 0.01% |
| Place/Store | 1 | 0.01% |
| Activity/Event | 1 | 0.01% |
| process | 1 | 0.01% |
| information | 1 | 0.01% |
| customer | 1 | 0.01% |
| quality | 1 | 0.01% |
| multiplier values | 1 | 0.01% |
| free spin rewards | 1 | 0.01% |
| game mechanics | 1 | 0.01% |
| Biome/Environment | 1 | 0.01% |
| Date/Time | 1 | 0.01% |
| Event/Activity | 1 | 0.01% |
| Holiday | 1 | 0.01% |
| Social Relationship | 1 | 0.01% |
| target audience | 1 | 0.01% |
| Symptom | 1 | 0.01% |
| Nutrient | 1 | 0.01% |
| social circle | 1 | 0.01% |
| physical appearance | 1 | 0.01% |
| clothing | 1 | 0.01% |
| Sizing | 1 | 0.01% |
| Team | 1 | 0.01% |
| Project | 1 | 0.01% |
| Budget | 1 | 0.01% |
| training phase | 1 | 0.01% |
| recovery phase | 1 | 0.01% |
| Password | 1 | 0.01% |
| Password Recovery Program | 1 | 0.01% |
| Email Client | 1 | 0.01% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| bind | 559 | 7.15% |
| PPI | 284 | 3.63% |
| Association | 243 | 3.11% |
| Positive_Correlation | 128 | 1.64% |
| Lives_In | 105 | 1.34% |
| Negative_Correlation | 64 | 0.82% |
| Exhibits | 38 | 0.49% |
| league | 37 | 0.47% |
| present in work | 37 | 0.47% |
| connecting line | 36 | 0.46% |
| country | 36 | 0.46% |
| performer | 35 | 0.45% |
| platform | 33 | 0.42% |
| spouse | 32 | 0.41% |
| characters | 31 | 0.40% |
| sport | 31 | 0.40% |
| participant | 31 | 0.40% |
| movement | 30 | 0.38% |
| original broadcaster | 30 | 0.38% |
| location | 30 | 0.38% |
| terminus location | 30 | 0.38% |
| record label | 30 | 0.38% |
| located in the administrative territorial entity | 30 | 0.38% |
| heritage designation | 30 | 0.38% |
| family | 30 | 0.38% |
| located on astronomical body | 29 | 0.37% |
| online service | 29 | 0.37% |
| architect | 29 | 0.37% |
| member of sports team | 29 | 0.37% |
| father | 29 | 0.37% |
| genre | 29 | 0.37% |
| cast member | 29 | 0.37% |
| astronaut mission | 28 | 0.36% |
| award received | 28 | 0.36% |
| employer | 28 | 0.36% |
| licensed to broadcast to | 28 | 0.36% |
| position held | 28 | 0.36% |
| contains the administrative territorial entity | 28 | 0.36% |
| position played on team / speciality | 28 | 0.36% |
| military branch | 27 | 0.35% |
| item operated | 27 | 0.35% |
| sibling | 26 | 0.33% |
| historic county | 26 | 0.33% |
| conflict | 26 | 0.33% |
| director | 26 | 0.33% |
| author | 26 | 0.33% |
| creator | 25 | 0.32% |
| operator | 25 | 0.32% |
| religion or worldview | 25 | 0.32% |
| place served by transport hub | 25 | 0.32% |
| crosses | 25 | 0.32% |
| discoverer or inventor | 24 | 0.31% |
| notable work | 24 | 0.31% |
| secretary general | 24 | 0.31% |
| educated at | 24 | 0.31% |
| continent | 24 | 0.31% |
| located in or next to body of water | 24 | 0.31% |
| sports discipline competed in | 24 | 0.31% |
| has use | 23 | 0.29% |
| home venue | 23 | 0.29% |
| officeholder | 23 | 0.29% |
| member of political party | 23 | 0.29% |
| competition class | 23 | 0.29% |
| surface played on | 23 | 0.29% |
| mountain range | 23 | 0.29% |
| occupation | 23 | 0.29% |
| conferred by | 22 | 0.28% |
| broadcast by | 22 | 0.28% |
| genic_interaction | 22 | 0.28% |
| participating team | 22 | 0.28% |
| owned by | 22 | 0.28% |
| country of citizenship | 21 | 0.27% |
| form of creative work | 21 | 0.27% |
| developer | 21 | 0.27% |
| participant in | 21 | 0.27% |
| ammunition | 21 | 0.27% |
| located in/on physical feature | 21 | 0.27% |
| instance of | 21 | 0.27% |
| founded by | 21 | 0.27% |
| has works in the collection | 21 | 0.27% |
| chairperson | 21 | 0.27% |
| production company | 20 | 0.26% |
| librettist | 20 | 0.26% |
| presenter | 20 | 0.26% |
| instrument | 20 | 0.26% |
| school district | 20 | 0.26% |
| field of work | 20 | 0.26% |
| worshipped by | 20 | 0.26% |
| historical region | 20 | 0.26% |
| court | 20 | 0.26% |
| location of discovery | 20 | 0.26% |
| commemorates | 20 | 0.26% |
| named after | 20 | 0.26% |
| partner in business or sport | 20 | 0.26% |
| significant event | 20 | 0.26% |
| copyright holder | 20 | 0.26% |
| occupant | 20 | 0.26% |
| pole position | 19 | 0.24% |
| distributed by | 19 | 0.24% |
| carries | 19 | 0.24% |
| publisher | 19 | 0.24% |
| has subsidiary | 19 | 0.24% |
| reservoir created | 19 | 0.24% |
| located in the present-day administrative territorial entity | 19 | 0.24% |
| medical condition | 19 | 0.24% |
| location of first performance | 19 | 0.24% |
| nominated for | 19 | 0.24% |
| member of | 19 | 0.24% |
| commanded by | 19 | 0.24% |
| parent organization | 19 | 0.24% |
| transport network | 19 | 0.24% |
| mouth of the watercourse | 19 | 0.24% |
| legislated by | 18 | 0.23% |
| location of formation | 18 | 0.23% |
| cathedral | 18 | 0.23% |
| winner | 18 | 0.23% |
| chief executive officer | 18 | 0.23% |
| manufacturer | 18 | 0.23% |
| track gauge | 18 | 0.23% |
| grants | 18 | 0.23% |
| place of publication | 18 | 0.23% |
| takes place in fictional universe | 18 | 0.23% |
| convicted of | 18 | 0.23% |
| has cause | 18 | 0.23% |
| has effect | 18 | 0.23% |
| model item | 18 | 0.23% |
| parent taxon | 18 | 0.23% |
| mother | 18 | 0.23% |
| crew member(s) | 18 | 0.23% |
| represents | 17 | 0.22% |
| chromosome | 17 | 0.22% |
| composer | 17 | 0.22% |
| part of the series | 17 | 0.22% |
| uses | 17 | 0.22% |
| radio format | 17 | 0.22% |
| candidacy in election | 17 | 0.22% |
| original language of film or TV show | 17 | 0.22% |
| musical conductor | 17 | 0.22% |
| part of | 17 | 0.22% |
| editor | 17 | 0.22% |
| headquarters location | 17 | 0.22% |
| basin country | 17 | 0.22% |
| represented by | 17 | 0.22% |
| highest judicial authority | 16 | 0.20% |
| said to be the same as | 16 | 0.20% |
| this zoological name is coordinate with | 16 | 0.20% |
| objective of project or action | 16 | 0.20% |
| language used | 16 | 0.20% |
| stock exchange | 16 | 0.20% |
| located in the statistical territorial entity | 16 | 0.20% |
| found in taxon | 16 | 0.20% |
| substitute/deputy/replacement of office/officeholder | 16 | 0.20% |
| noble title | 16 | 0.20% |
| product or material produced | 16 | 0.20% |
| is the study of | 16 | 0.20% |
| vehicle | 16 | 0.20% |
| terminus | 16 | 0.20% |
| voice type | 16 | 0.20% |
| has written for | 16 | 0.20% |
| place of birth | 16 | 0.20% |
| coach of sports team | 15 | 0.19% |
| connects with | 15 | 0.19% |
| allegiance | 15 | 0.19% |
| airline alliance | 15 | 0.19% |
| filming location | 15 | 0.19% |
| architectural style | 15 | 0.19% |
| biological process | 15 | 0.19% |
| taxonomic type | 15 | 0.19% |
| place of detention | 15 | 0.19% |
| genetic association | 15 | 0.19% |
| incarnation of | 15 | 0.19% |
| destination point | 15 | 0.19% |
| drug or therapy used for treatment | 15 | 0.19% |
| country of origin | 15 | 0.19% |
| has quality | 15 | 0.19% |
| recommended unit of measurement | 15 | 0.19% |
| facet of | 14 | 0.18% |
| child | 14 | 0.18% |
| significant person | 14 | 0.18% |
| head of state | 14 | 0.18% |
| parent club | 14 | 0.18% |
| league system | 14 | 0.18% |
| located in time zone | 14 | 0.18% |
| standards body | 14 | 0.18% |
| published in | 14 | 0.18% |
| distribution format | 14 | 0.18% |
| unmarried partner | 14 | 0.18% |
| produced by | 14 | 0.18% |
| package management system | 14 | 0.18% |
| business division | 14 | 0.18% |
| wheel arrangement | 14 | 0.18% |
| country for sport | 14 | 0.18% |
| indigenous to | 14 | 0.18% |
| adjacent station | 14 | 0.18% |
| ethnic group | 14 | 0.18% |
| manner of death | 14 | 0.18% |
| dam | 14 | 0.18% |
| basic form of government | 14 | 0.18% |
| dedicated to | 14 | 0.18% |
| vessel class | 14 | 0.18% |
| made from material | 14 | 0.18% |
| diocese | 13 | 0.17% |
| illustrator | 13 | 0.17% |
| taxon rank | 13 | 0.17% |
| after a work by | 13 | 0.17% |
| fictional universe described in | 13 | 0.17% |
| relative | 13 | 0.17% |
| cause of death | 13 | 0.17% |
| enemy of | 13 | 0.17% |
| lyrics by | 13 | 0.17% |
| origin of the watercourse | 13 | 0.17% |
| this taxon is source of | 13 | 0.17% |
| writing language | 13 | 0.17% |
| replaced by | 13 | 0.17% |
| residence | 13 | 0.17% |
| military rank | 13 | 0.17% |
| operating area | 13 | 0.17% |
| cell component | 13 | 0.17% |
| organization directed by the office or position | 13 | 0.17% |
| location of creation | 13 | 0.17% |
| target | 13 | 0.17% |
| lake on watercourse | 13 | 0.17% |
| merged into | 13 | 0.17% |
| head of government | 13 | 0.17% |
| voice actor | 12 | 0.15% |
| language of work or name | 12 | 0.15% |
| measured physical quantity | 12 | 0.15% |
| stock market index | 12 | 0.15% |
| maintained by | 12 | 0.15% |
| commander of (DEPRECATED) | 12 | 0.15% |
| drainage basin | 12 | 0.15% |
| member of the crew of | 12 | 0.15% |
| industry | 12 | 0.15% |
| honorific prefix | 12 | 0.15% |
| candidate | 12 | 0.15% |
| writing system | 12 | 0.15% |
| natural product of taxon | 12 | 0.15% |
| main subject | 12 | 0.15% |
| anatomical location | 12 | 0.15% |
| subclass of | 12 | 0.15% |
| copyright license | 12 | 0.15% |
| culture | 12 | 0.15% |
| applies to jurisdiction | 12 | 0.15% |
| child astronomical body | 12 | 0.15% |
| languages spoken, written or signed | 12 | 0.15% |
| co-driver | 12 | 0.15% |
| primary destinations | 12 | 0.15% |
| brand | 12 | 0.15% |
| designed by | 12 | 0.15% |
| medical condition treated | 12 | 0.15% |
| used by | 12 | 0.15% |
| producer | 11 | 0.14% |
| parent astronomical body | 11 | 0.14% |
| currency | 11 | 0.14% |
| domain of saint or deity | 11 | 0.14% |
| operating system | 11 | 0.14% |
| constellation | 11 | 0.14% |
| authority | 11 | 0.14% |
| dialect of | 11 | 0.14% |
| opposite of | 11 | 0.14% |
| owner of | 11 | 0.14% |
| powered by | 11 | 0.14% |
| has part(s) of the class | 11 | 0.14% |
| discography | 11 | 0.14% |
| described by source | 11 | 0.14% |
| edition or translation of | 11 | 0.14% |
| work location | 11 | 0.14% |
| anthem | 11 | 0.14% |
| symptoms and signs | 10 | 0.13% |
| has immediate cause | 10 | 0.13% |
| official language | 10 | 0.13% |
| political alignment | 10 | 0.13% |
| commissioned by | 10 | 0.13% |
| programming language | 10 | 0.13% |
| religious order | 10 | 0.13% |
| contributed to creative work | 10 | 0.13% |
| physically interacts with | 10 | 0.13% |
| killed by | 10 | 0.13% |
| linguistic typology | 10 | 0.13% |
| interested in | 10 | 0.13% |
| office held by head of the organization | 10 | 0.13% |
| IUCN conservation status | 10 | 0.13% |
| newspaper format | 10 | 0.13% |
| start point | 10 | 0.13% |
| endemic to | 10 | 0.13% |
| native language | 10 | 0.13% |
| advertises | 10 | 0.13% |
| legal form | 10 | 0.13% |
| health specialty | 10 | 0.13% |
| has part(s) | 10 | 0.13% |
| official religion | 10 | 0.13% |
| different from | 10 | 0.13% |
| replaces | 10 | 0.13% |
| follows | 10 | 0.13% |
| vehicle normally used | 9 | 0.12% |
| qualifies for event | 9 | 0.12% |
| contains | 9 | 0.12% |
| measures | 9 | 0.12% |
| coextensive with | 9 | 0.12% |
| collection | 9 | 0.12% |
| director / manager | 9 | 0.12% |
| lake outflow | 9 | 0.12% |
| mascot | 9 | 0.12% |
| director of photography | 9 | 0.12% |
| structural engineer | 9 | 0.12% |
| highest point | 9 | 0.12% |
| signatory | 9 | 0.12% |
| field of this occupation | 9 | 0.12% |
| shares border with | 9 | 0.12% |
| tracklist | 9 | 0.12% |
| head coach | 9 | 0.12% |
| Gram staining | 9 | 0.12% |
| from narrative universe | 9 | 0.12% |
| color | 8 | 0.10% |
| separated from | 8 | 0.10% |
| fastest lap | 8 | 0.10% |
| connecting service | 8 | 0.10% |
| official residence | 8 | 0.10% |
| music created for | 8 | 0.10% |
| tributary | 8 | 0.10% |
| narrative location | 8 | 0.10% |
| followed by | 8 | 0.10% |
| located in protected area | 8 | 0.10% |
| first appearance | 8 | 0.10% |
| defendant | 8 | 0.10% |
| family name | 8 | 0.10% |
| canonization status | 8 | 0.10% |
| capital of | 8 | 0.10% |
| next higher rank | 8 | 0.10% |
| screenwriter | 8 | 0.10% |
| possible treatment | 8 | 0.10% |
| has edition or translation | 8 | 0.10% |
| place of burial | 8 | 0.10% |
| molecular function | 8 | 0.10% |
| foundational text | 8 | 0.10% |
| intended public | 7 | 0.09% |
| software engine | 7 | 0.09% |
| affiliation | 7 | 0.09% |
| sponsor | 7 | 0.09% |
| history of topic | 7 | 0.09% |
| immediate cause of | 7 | 0.09% |
| studied by | 7 | 0.09% |
| partially coincident with | 7 | 0.09% |
| victory | 7 | 0.09% |
| inspired by | 7 | 0.09% |
| anatomical branch of | 7 | 0.09% |
| afflicts | 7 | 0.09% |
| contains settlement | 7 | 0.09% |
| measurement scale | 7 | 0.09% |
| has list | 7 | 0.09% |
| place of death | 7 | 0.09% |
| student of | 7 | 0.09% |
| character type | 7 | 0.09% |
| title of chess person | 7 | 0.09% |
| location of landing | 7 | 0.09% |
| engine configuration | 7 | 0.09% |
| main regulatory text | 7 | 0.09% |
| has anatomical branch | 7 | 0.09% |
| general classification of race participants | 7 | 0.09% |
| practiced by | 7 | 0.09% |
| recorded at studio or venue | 7 | 0.09% |
| encodes | 7 | 0.09% |
| space launch vehicle | 7 | 0.09% |
| student | 6 | 0.08% |
| drafted by | 6 | 0.08% |
| instrumentation | 6 | 0.08% |
| flag | 6 | 0.08% |
| social classification | 6 | 0.08% |
| contributor to the creative work or subject | 6 | 0.08% |
| GUI toolkit or framework | 6 | 0.08% |
| time period | 6 | 0.08% |
| cause of destruction | 6 | 0.08% |
| members have occupation | 6 | 0.08% |
| public holiday | 6 | 0.08% |
| iconographic symbol | 6 | 0.08% |
| series spin-off | 6 | 0.08% |
| approved by | 6 | 0.08% |
| victim | 6 | 0.08% |
| organizer | 6 | 0.08% |
| sex or gender | 6 | 0.08% |
| derivative work | 6 | 0.08% |
| based on | 6 | 0.08% |
| legislative body | 6 | 0.08% |
| depicts | 6 | 0.08% |
| twinned administrative body | 6 | 0.08% |
| economy of topic | 6 | 0.08% |
| league level below | 6 | 0.08% |
| site of astronomical discovery | 6 | 0.08% |
| fabrication method | 6 | 0.08% |
| significant place | 6 | 0.08% |
| consecrator | 5 | 0.06% |
| CPU | 5 | 0.06% |
| executive body | 5 | 0.06% |
| business model | 5 | 0.06% |
| choreographer | 5 | 0.06% |
| vaccine for | 5 | 0.06% |
| located on street | 5 | 0.06% |
| board member | 5 | 0.06% |
| carries scientific instrument | 5 | 0.06% |
| diplomatic relation | 5 | 0.06% |
| editor-in-chief | 5 | 0.06% |
| ancestral home | 5 | 0.06% |
| has active ingredient | 5 | 0.06% |
| game mode | 5 | 0.06% |
| superhuman feature or ability | 5 | 0.06% |
| language regulatory body | 5 | 0.06% |
| stepparent | 5 | 0.06% |
| office held by head of government | 5 | 0.06% |
| implementation of | 5 | 0.06% |
| executive producer | 5 | 0.06% |
| central bank/issuer | 5 | 0.06% |
| type of variable star | 5 | 0.06% |
| has seal, badge, or sigil | 5 | 0.06% |
| source of energy | 5 | 0.06% |
| political ideology | 5 | 0.06% |
| successful candidate | 5 | 0.06% |
| designed to carry | 5 | 0.06% |
| minor planet group | 5 | 0.06% |
| home world | 5 | 0.06% |
| airline hub | 5 | 0.06% |
| points classification | 5 | 0.06% |
| youth wing | 5 | 0.06% |
| programming paradigm | 4 | 0.05% |
| house publication | 4 | 0.05% |
| has contributing factor | 4 | 0.05% |
| underlies | 4 | 0.05% |
| subject has role | 4 | 0.05% |
| capital | 4 | 0.05% |
| species kept | 4 | 0.05% |
| theme music | 4 | 0.05% |
| patron saint | 4 | 0.05% |
| translator | 4 | 0.05% |
| party chief representative | 4 | 0.05% |
| office held by head of state | 4 | 0.05% |
| readable file format | 4 | 0.05% |
| type of orbit | 4 | 0.05% |
| emulates | 4 | 0.05% |
| connector | 4 | 0.05% |
| charge | 4 | 0.05% |
| family name identical to this given name | 4 | 0.05% |
| league level above | 4 | 0.05% |
| medical examination | 4 | 0.05% |
| penalty | 4 | 0.05% |
| subdivision of this unit | 4 | 0.05% |
| sexually homologous with | 4 | 0.05% |
| feast day | 4 | 0.05% |
| in opposition to | 4 | 0.05% |
| captain | 4 | 0.05% |
| antiparticle | 4 | 0.05% |
| Bind | 4 | 0.05% |
| armament | 4 | 0.05% |
| supported sports team | 4 | 0.05% |
| writable file format | 4 | 0.05% |
| geography of topic | 4 | 0.05% |
| active ingredient in | 4 | 0.05% |
| territory overlaps | 4 | 0.05% |
| game mechanics | 4 | 0.05% |
| parliamentary group | 4 | 0.05% |
| season | 4 | 0.05% |
| manifestation of | 4 | 0.05% |
| parent of this hybrid, breed, or cultivar | 3 | 0.04% |
| statement describes | 3 | 0.04% |
| wears | 3 | 0.04% |
| wing configuration | 3 | 0.04% |
| scheduled service destination | 3 | 0.04% |
| destroyed | 3 | 0.04% |
| typically sells | 3 | 0.04% |
| sports season of league or competition | 3 | 0.04% |
| amended by | 3 | 0.04% |
| calculated from | 3 | 0.04% |
| influenced by | 3 | 0.04% |
| exclave of | 3 | 0.04% |
| size designation | 3 | 0.04% |
| academic thesis | 3 | 0.04% |
| Christian liturgical rite | 3 | 0.04% |
| cuisine | 3 | 0.04% |
| geomorphological unit | 3 | 0.04% |
| honorific suffix | 3 | 0.04% |
| speaker | 3 | 0.04% |
| narrator | 3 | 0.04% |
| madhhab | 3 | 0.04% |
| has natural reservoir | 3 | 0.04% |
| designated as terrorist by | 3 | 0.04% |
| crystal system | 3 | 0.04% |
| top-level Internet domain | 3 | 0.04% |
| parallel street | 3 | 0.04% |
| contributing factor of | 3 | 0.04% |
| type foundry | 3 | 0.04% |
| offers view on | 3 | 0.04% |
| demographics of topic | 3 | 0.04% |
| country of registry | 3 | 0.04% |
| archives at | 3 | 0.04% |
| perpetrator | 3 | 0.04% |
| overlies | 3 | 0.04% |
| statistical leader | 3 | 0.04% |
| inflows | 3 | 0.04% |
| home port | 3 | 0.04% |
| depends on software | 3 | 0.04% |
| performed at | 3 | 0.04% |
| director of archaeological fieldwork | 3 | 0.04% |
| dual to | 3 | 0.04% |
| is a list of | 2 | 0.03% |
| animal breed | 2 | 0.03% |
| appears in the form of | 2 | 0.03% |
| enclave within | 2 | 0.03% |
| foods traditionally associated | 2 | 0.03% |
| academic degree | 2 | 0.03% |
| binding of software library | 2 | 0.03% |
| antagonist muscle | 2 | 0.03% |
| prohibits | 2 | 0.03% |
| Comparison | 2 | 0.03% |
| next lower rank | 2 | 0.03% |
| produced sound | 2 | 0.03% |
| final event | 2 | 0.03% |
| has grammatical case | 2 | 0.03% |
| category of associated people | 2 | 0.03% |
| mountains classification | 2 | 0.03% |
| regulated by | 2 | 0.03% |
| majority opinion by | 2 | 0.03% |
| via | 2 | 0.03% |
| depicted by | 2 | 0.03% |
| collection creator | 2 | 0.03% |
| set in environment | 2 | 0.03% |
| mount | 2 | 0.03% |
| stereoisomer of | 2 | 0.03% |
| Conversion | 2 | 0.03% |
| territory claimed by | 2 | 0.03% |
| month of the year | 2 | 0.03% |
| office contested | 2 | 0.03% |
| tonality | 2 | 0.03% |
| rector | 2 | 0.03% |
| polymer of | 2 | 0.03% |
| given name version for other gender | 2 | 0.03% |
| dialect of computer language | 2 | 0.03% |
| soundtrack release | 2 | 0.03% |
| less than | 2 | 0.03% |
| adjacent building | 2 | 0.03% |
| interchange station | 2 | 0.03% |
| day in year for periodic occurrence | 2 | 0.03% |
| postsynaptic connection | 2 | 0.03% |
| OSI Model layer location | 2 | 0.03% |
| coat of arms | 2 | 0.03% |
| media franchise | 2 | 0.03% |
| source of material | 2 | 0.03% |
| trophy awarded | 2 | 0.03% |
| innervated by | 2 | 0.03% |
| asteroid spectral type | 2 | 0.03% |
| update method | 2 | 0.03% |
| issued by | 2 | 0.03% |
| has boundary | 2 | 0.03% |
| discovery method | 2 | 0.03% |
| chassis | 2 | 0.03% |
| sidekick of | 2 | 0.03% |
| film editor | 2 | 0.03% |
| set in period | 2 | 0.03% |
| diaspora | 2 | 0.03% |
| expansion of | 2 | 0.03% |
| mother house | 2 | 0.03% |
| official symbol | 2 | 0.03% |
| input method | 2 | 0.03% |
| appointed by | 2 | 0.03% |
| has fruit type | 2 | 0.03% |
| port of registry | 1 | 0.01% |
| temporal range end | 1 | 0.01% |
| style of karate | 1 | 0.01% |
| has facet polytope | 1 | 0.01% |
| next crossing downstream | 1 | 0.01% |
| base | 1 | 0.01% |
| repealed by | 1 | 0.01% |
| fossil found in this unit | 1 | 0.01% |
| godparent | 1 | 0.01% |
| reply to | 1 | 0.01% |
| demonym of | 1 | 0.01% |
| samples from work | 1 | 0.01% |
| route of administration | 1 | 0.01% |
| plaintiff | 1 | 0.01% |
| computes solution to | 1 | 0.01% |
| given name | 1 | 0.01% |
| Roman agnomen | 1 | 0.01% |
| greater than | 1 | 0.01% |
| charted in | 1 | 0.01% |
| investor | 1 | 0.01% |
| expressed in | 1 | 0.01% |
| taxon synonym | 1 | 0.01% |
| means of locomotion | 1 | 0.01% |
| decay mode | 1 | 0.01% |
| interaction | 1 | 0.01% |
| socket supported | 1 | 0.01% |
| colorist | 1 | 0.01% |
| official podcast | 1 | 0.01% |
| structure replaced by | 1 | 0.01% |
| adapted by | 1 | 0.01% |
| officially opened by | 1 | 0.01% |
| narrative motif | 1 | 0.01% |
| shape | 1 | 0.01% |
| funder | 1 | 0.01% |
| indexed in bibliographic review | 1 | 0.01% |
| item disputed by | 1 | 0.01% |
| risk factor | 1 | 0.01% |
| source of income | 1 | 0.01% |
| UNESCO language status | 1 | 0.01% |
| sexual orientation | 1 | 0.01% |
| intangible cultural heritage status | 1 | 0.01% |
| has melody | 1 | 0.01% |
| lowest point | 1 | 0.01% |
| avionics | 1 | 0.01% |
| doctoral student | 1 | 0.01% |
| microarchitecture | 1 | 0.01% |
| associated electoral district | 1 | 0.01% |
| proved by | 1 | 0.01% |
| solved by | 1 | 0.01% |
| lifestyle | 1 | 0.01% |
| facial hair | 1 | 0.01% |
| valid in period | 1 | 0.01% |
| disjoint union of | 1 | 0.01% |
| possible medical findings | 1 | 0.01% |
| combination classification | 1 | 0.01% |
| addressee | 1 | 0.01% |
| has pattern | 1 | 0.01% |
| permanent resident of | 1 | 0.01% |
| conjugate acid | 1 | 0.01% |
| innervates | 1 | 0.01% |
| does not have part | 1 | 0.01% |
| significant drug interaction | 1 | 0.01% |
| together with | 1 | 0.01% |
| prime factor | 1 | 0.01% |
| doctoral advisor | 1 | 0.01% |
| temporal range start | 1 | 0.01% |
| era name | 1 | 0.01% |
| undercarriage | 1 | 0.01% |
| volcano observatory | 1 | 0.01% |
| production designer | 1 | 0.01% |
| type locality (geology) | 1 | 0.01% |
| disease transmission process | 1 | 0.01% |
| elector | 1 | 0.01% |
| laws applied | 1 | 0.01% |
| therapeutic area | 1 | 0.01% |
| gens | 1 | 0.01% |
| set during recurring event | 1 | 0.01% |
| coolant | 1 | 0.01% |
| start grade | 1 | 0.01% |
| muscle insertion | 1 | 0.01% |
| research site | 1 | 0.01% |
| delta of | 1 | 0.01% |
| kit supplier | 1 | 0.01% |
| programmer | 1 | 0.01% |
| reissue of | 1 | 0.01% |
| aspect ratio (W:H) | 1 | 0.01% |
| cover art by | 1 | 0.01% |
| copyright representative | 1 | 0.01% |
| referee | 1 | 0.01% |
| general manager | 1 | 0.01% |
| has facility | 1 | 0.01% |
| symbolizes | 1 | 0.01% |
| venous drainage | 1 | 0.01% |
| military casualty classification | 1 | 0.01% |
| has phenotype | 1 | 0.01% |
| emergency phone number | 1 | 0.01% |
| repeals | 1 | 0.01% |
| place of origin (Switzerland) | 1 | 0.01% |

#### val

| label | count | share |
| --- | --- | --- |
| PPI | 10 | 7.19% |
| heritage designation | 3 | 2.16% |
| performer | 3 | 2.16% |
| bind | 3 | 2.16% |
| taxon rank | 2 | 1.44% |
| inspired by | 2 | 1.44% |
| less than | 2 | 1.44% |
| product or material produced | 2 | 1.44% |
| from narrative universe | 2 | 1.44% |
| licensed to broadcast to | 2 | 1.44% |
| present in work | 2 | 1.44% |
| model item | 2 | 1.44% |
| first appearance | 2 | 1.44% |
| place served by transport hub | 2 | 1.44% |
| ammunition | 2 | 1.44% |
| occupation | 2 | 1.44% |
| parent organization | 2 | 1.44% |
| endemic to | 2 | 1.44% |
| medical condition treated | 2 | 1.44% |
| country for sport | 2 | 1.44% |
| separated from | 1 | 0.72% |
| worshipped by | 1 | 0.72% |
| significant person | 1 | 0.72% |
| position held | 1 | 0.72% |
| copyright license | 1 | 0.72% |
| crew member(s) | 1 | 0.72% |
| enemy of | 1 | 0.72% |
| base | 1 | 0.72% |
| co-driver | 1 | 0.72% |
| game mode | 1 | 0.72% |
| produced by | 1 | 0.72% |
| manner of death | 1 | 0.72% |
| significant event | 1 | 0.72% |
| field of this occupation | 1 | 0.72% |
| writing language | 1 | 0.72% |
| peak bagging classification | 1 | 0.72% |
| executive body | 1 | 0.72% |
| parent taxon | 1 | 0.72% |
| sport | 1 | 0.72% |
| sponsor | 1 | 0.72% |
| platform | 1 | 0.72% |
| unmarried partner | 1 | 0.72% |
| located in the administrative territorial entity | 1 | 0.72% |
| reservoir created | 1 | 0.72% |
| located on astronomical body | 1 | 0.72% |
| movement | 1 | 0.72% |
| award received | 1 | 0.72% |
| diplomatic relation | 1 | 0.72% |
| color | 1 | 0.72% |
| sports discipline competed in | 1 | 0.72% |
| stock market index | 1 | 0.72% |
| noble title | 1 | 0.72% |
| military rank | 1 | 0.72% |
| fictional universe described in | 1 | 0.72% |
| cell component | 1 | 0.72% |
| member of political party | 1 | 0.72% |
| mouth of the watercourse | 1 | 0.72% |
| presenter | 1 | 0.72% |
| characters | 1 | 0.72% |
| convicted of | 1 | 0.72% |
| office held by head of government | 1 | 0.72% |
| has use | 1 | 0.72% |
| tributary | 1 | 0.72% |
| architect | 1 | 0.72% |
| drug or therapy used for treatment | 1 | 0.72% |
| IUCN conservation status | 1 | 0.72% |
| participant in | 1 | 0.72% |
| measurement scale | 1 | 0.72% |
| position played on team / speciality | 1 | 0.72% |
| has subsidiary | 1 | 0.72% |
| authority | 1 | 0.72% |
| creator | 1 | 0.72% |
| approved by | 1 | 0.72% |
| voice actor | 1 | 0.72% |
| child | 1 | 0.72% |
| organization directed by the office or position | 1 | 0.72% |
| officially opened by | 1 | 0.72% |
| indigenous to | 1 | 0.72% |
| conferred by | 1 | 0.72% |
| applies to jurisdiction | 1 | 0.72% |
| crosses | 1 | 0.72% |
| member of the crew of | 1 | 0.72% |
| next higher rank | 1 | 0.72% |
| place of birth | 1 | 0.72% |
| defendant | 1 | 0.72% |
| employer | 1 | 0.72% |
| follows | 1 | 0.72% |
| director | 1 | 0.72% |
| interchange station | 1 | 0.72% |
| youth wing | 1 | 0.72% |
| taxonomic type | 1 | 0.72% |
| connects with | 1 | 0.72% |
| notable work | 1 | 0.72% |
| instrument | 1 | 0.72% |
| different from | 1 | 0.72% |
| voice type | 1 | 0.72% |
| dedicated to | 1 | 0.72% |
| author | 1 | 0.72% |
| publisher | 1 | 0.72% |
| space launch vehicle | 1 | 0.72% |
| candidacy in election | 1 | 0.72% |
| named after | 1 | 0.72% |
| distributed by | 1 | 0.72% |
| producer | 1 | 0.72% |
| location | 1 | 0.72% |
| original broadcaster | 1 | 0.72% |
| dam | 1 | 0.72% |
| cause of death | 1 | 0.72% |

### Classifications

**train, biotech_event.** Majority class 24.6% of 240 answers. The marginal is `true_label`, not the menu.

#### train biotech_event

| label | count | share |
| --- | --- | --- |
| executive statement | 59 | 24.58% |
| other | 31 | 12.92% |
| company description | 21 | 8.75% |
| product launching & presentation | 20 | 8.33% |
| expanding geography | 12 | 5.00% |
| alliance & partnership | 12 | 5.00% |
| event organization | 9 | 3.75% |
| m&a | 9 | 3.75% |
| executive appointment | 8 | 3.33% |
| hiring | 8 | 3.33% |
| funding round | 8 | 3.33% |
| support & philanthropy | 7 | 2.92% |
| participation in an event | 7 | 2.92% |
| new initiatives & programs | 5 | 2.08% |
| article publication | 5 | 2.08% |
| product updates | 4 | 1.67% |
| regulatory approval | 4 | 1.67% |
| expanding industry | 3 | 1.25% |
| service & product providing | 2 | 0.83% |
| subsidiary establishment | 1 | 0.42% |
| closing | 1 | 0.42% |
| investment in public company | 1 | 0.42% |
| foundation | 1 | 0.42% |
| clinical trial sponsorship | 1 | 0.42% |
| new initiatives or programs | 1 | 0.42% |

**train, chfinann_event.** Majority class 51.9% of 6,324 answers. The marginal is `true_label`, not the menu.

#### train chfinann_event

| label | count | share |
| --- | --- | --- |
| EquityPledge | 3,281 | 51.88% |
| EquityUnderweight | 1,313 | 20.76% |
| EquityOverweight | 1,237 | 19.56% |
| EquityRepurchase | 264 | 4.17% |
| EquityFreeze | 229 | 3.62% |

**train, docee_event.** Majority class 11.1% of 1,858 answers. The marginal is `true_label`, not the menu.

#### train docee_event

| label | count | share |
| --- | --- | --- |
| Armed Conflict | 207 | 11.14% |
| Air Crash | 111 | 5.97% |
| CommitCrime - Accuse | 78 | 4.20% |
| Earthquakes | 66 | 3.55% |
| CommitCrime - Sentence | 66 | 3.55% |
| Fire | 66 | 3.55% |
| Protest | 64 | 3.44% |
| Election | 63 | 3.39% |
| Bank Robbery | 63 | 3.39% |
| Sports Competition | 62 | 3.34% |
| Government Policy Changes | 53 | 2.85% |
| Riot | 47 | 2.53% |
| Resignation_Dismissal | 46 | 2.48% |
| Road Crash | 43 | 2.31% |
| CommitCrime - Arrest | 40 | 2.15% |
| Storm | 39 | 2.10% |
| Appoint_Inauguration | 38 | 2.05% |
| Train Collisions | 37 | 1.99% |
| Gas Explosion | 35 | 1.88% |
| Mass Poisoning | 35 | 1.88% |
| Organization Closed | 32 | 1.72% |
| Disease Outbreaks | 31 | 1.67% |
| Famous Person - Death | 30 | 1.61% |
| Floods | 30 | 1.61% |
| Diplomatic Talks | 29 | 1.56% |
| Break Historical Records | 28 | 1.51% |
| Mine Collapses | 27 | 1.45% |
| Sign Agreement | 26 | 1.40% |
| Shipwreck | 21 | 1.13% |
| New Wonders in Nature | 20 | 1.08% |
| Military Excercise | 19 | 1.02% |
| CommitCrime - Release | 19 | 1.02% |
| Volcano Eruption | 19 | 1.02% |
| Awards Ceremony | 18 | 0.97% |
| Insect Disaster | 18 | 0.97% |
| Famous Person - Sick | 17 | 0.91% |
| Organization Merge | 17 | 0.91% |
| Strike | 15 | 0.81% |
| New Archeological Discoveries | 15 | 0.81% |
| CommitCrime - Investigate | 14 | 0.75% |
| Famous Person - Give a Speech | 13 | 0.70% |
| Organization Established | 10 | 0.54% |
| Organization Fine | 10 | 0.54% |
| Mudslides | 10 | 0.54% |
| Famous Person - Recovered | 10 | 0.54% |
| Financial Aid | 10 | 0.54% |
| Tear Up Agreement | 9 | 0.48% |
| Famous Person - Divorce | 9 | 0.48% |
| Withdraw from an Orgnization | 8 | 0.43% |
| Tsunamis | 8 | 0.43% |
| Diplomatic Visit | 7 | 0.38% |
| Famous Person - Marriage | 7 | 0.38% |
| Environment Pollution | 7 | 0.38% |
| Join in an Orgnization | 7 | 0.38% |
| New Achievements in Aerospace | 6 | 0.32% |
| Famine | 6 | 0.32% |
| Droughts | 6 | 0.32% |
| Financial Crisis | 6 | 0.32% |
| Regime Change | 5 | 0.27% |

**train, docfee_event.** Majority class 35.1% of 2,744 answers. The marginal is `true_label`, not the menu.

#### train docfee_event

| label | count | share |
| --- | --- | --- |
| EquityUnderweight | 962 | 35.06% |
| BankruptcyLiquidation | 496 | 18.08% |
| EquityPledge | 387 | 14.10% |
| MajorAssetLoss | 230 | 8.38% |
| EquityOverweight | 183 | 6.67% |
| EquityFreeze | 179 | 6.52% |
| MajorSafetyAccident | 143 | 5.21% |
| ExecutiveDeath | 92 | 3.35% |
| MajorExternalPayout | 72 | 2.62% |

**val, biotech_event.** Majority class 40.0% of 5 answers. The marginal is `true_label`, not the menu.

#### val biotech_event

| label | count | share |
| --- | --- | --- |
| service & product providing | 2 | 40.00% |
| company description | 1 | 20.00% |
| closing | 1 | 20.00% |
| executive statement | 1 | 20.00% |

**val, chfinann_event.** Majority class 49.2% of 126 answers. The marginal is `true_label`, not the menu.

#### val chfinann_event

| label | count | share |
| --- | --- | --- |
| EquityPledge | 62 | 49.21% |
| EquityOverweight | 30 | 23.81% |
| EquityUnderweight | 29 | 23.02% |
| EquityFreeze | 5 | 3.97% |

**val, docee_event.** Majority class 9.8% of 41 answers. The marginal is `true_label`, not the menu.

#### val docee_event

| label | count | share |
| --- | --- | --- |
| Armed Conflict | 4 | 9.76% |
| Air Crash | 3 | 7.32% |
| Road Crash | 3 | 7.32% |
| Break Historical Records | 2 | 4.88% |
| Government Policy Changes | 2 | 4.88% |
| Sign Agreement | 2 | 4.88% |
| CommitCrime - Release | 2 | 4.88% |
| Election | 2 | 4.88% |
| Gas Explosion | 2 | 4.88% |
| Withdraw from an Orgnization | 1 | 2.44% |
| CommitCrime - Arrest | 1 | 2.44% |
| Volcano Eruption | 1 | 2.44% |
| CommitCrime - Accuse | 1 | 2.44% |
| Mine Collapses | 1 | 2.44% |
| Join in an Orgnization | 1 | 2.44% |
| Floods | 1 | 2.44% |
| Protest | 1 | 2.44% |
| Riot | 1 | 2.44% |
| Tear Up Agreement | 1 | 2.44% |
| Bank Robbery | 1 | 2.44% |
| Droughts | 1 | 2.44% |
| Awards Ceremony | 1 | 2.44% |
| Sports Competition | 1 | 2.44% |
| CommitCrime - Sentence | 1 | 2.44% |
| Shipwreck | 1 | 2.44% |
| Military Excercise | 1 | 2.44% |
| Fire | 1 | 2.44% |
| Famous Person - Sick | 1 | 2.44% |

**val, docfee_event.** Majority class 40.0% of 55 answers. The marginal is `true_label`, not the menu.

#### val docfee_event

| label | count | share |
| --- | --- | --- |
| EquityUnderweight | 22 | 40.00% |
| BankruptcyLiquidation | 11 | 20.00% |
| EquityPledge | 6 | 10.91% |
| ExecutiveDeath | 5 | 9.09% |
| MajorSafetyAccident | 3 | 5.45% |
| EquityFreeze | 3 | 5.45% |
| MajorAssetLoss | 2 | 3.64% |
| EquityOverweight | 2 | 3.64% |
| MajorExternalPayout | 1 | 1.82% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| casualty_report | location | 66,560 | 93.48% |
| casualty_report | injured | 32,306 | 45.37% |
| casualty_report | missing | 28,213 | 39.62% |
| casualty_report | dead | 27,687 | 38.88% |
| casualty_report | record_metadata | anchorless/None ×28,718 |  |

**val.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| casualty_report | location | 1,407 | 92.81% |
| casualty_report | injured | 708 | 46.70% |
| casualty_report | missing | 597 | 39.38% |
| casualty_report | dead | 592 | 39.05% |
| casualty_report | record_metadata | anchorless/None ×603 |  |

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | unknown |
| relations | unknown |
| events | unknown |
| classifications | unknown |
| structures | unknown |
