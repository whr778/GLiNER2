# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sentence_rex | train | 34,314 | 33,247 | 0 | 34,314 | 0 | 0 | 0 | 0 |
| sentence_rex | val | 4,268 | 4,126 | 0 | 4,268 | 0 | 0 | 0 | 0 |
| sentence_rex | test | 4,283 | 4,153 | 0 | 4,283 | 0 | 0 | 0 | 0 |
| sentence_rex | all | 42,865 | 41,526 | 0 | 42,865 | 0 | 0 | 0 | 0 |

## sentence_rex

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 34,314 | 33,247 | 24 | 30 | 63 | 0 | 34,314 | 0 | 0 | 0 | 0 |
| val | 4,268 | 4,126 | 24 | 30 | 59 | 0 | 4,268 | 0 | 0 | 0 | 0 |
| test | 4,283 | 4,153 | 24 | 30 | 68 | 0 | 4,283 | 0 | 0 | 0 | 0 |
| all | 42,865 | 41,526 | 24 | 30 | 68 | 0 | 42,865 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | all | records | share |
| --- | --- | --- | --- | --- | --- | --- |
| eng | 33,340 | 4,151 | 4,151 | 41,642 | 83,284 | 97.15% |
| und | 885 | 113 | 125 | 1,123 | 2,246 | 2.62% |
| ita | 16 | 1 | 1 | 18 | 36 | 0.04% |
| fra | 16 | 1 | 0 | 17 | 34 | 0.04% |
| deu | 14 | 0 | 2 | 16 | 32 | 0.04% |
| spa | 8 | 0 | 2 | 10 | 20 | 0.02% |
| pol | 5 | 0 | 1 | 6 | 12 | 0.01% |
| fin | 3 | 0 | 0 | 3 | 6 | 0.01% |
| nor | 2 | 1 | 0 | 3 | 6 | 0.01% |
| nld | 3 | 0 | 0 | 3 | 6 | 0.01% |
| jpn | 3 | 0 | 0 | 3 | 6 | 0.01% |
| cat | 2 | 0 | 0 | 2 | 4 | 0.00% |
| hun | 2 | 0 | 0 | 2 | 4 | 0.00% |
| dan | 2 | 0 | 0 | 2 | 4 | 0.00% |
| tur | 1 | 0 | 0 | 1 | 2 | 0.00% |
| cym | 1 | 0 | 0 | 1 | 2 | 0.00% |
| mkd | 1 | 0 | 0 | 1 | 2 | 0.00% |
| ces | 1 | 0 | 0 | 1 | 2 | 0.00% |
| ori | 1 | 0 | 0 | 1 | 2 | 0.00% |
| bre | 1 | 0 | 0 | 1 | 2 | 0.00% |
| vie | 1 | 0 | 0 | 1 | 2 | 0.00% |
| isl | 1 | 0 | 0 | 1 | 2 | 0.00% |
| srp | 1 | 0 | 0 | 1 | 2 | 0.00% |
| yor | 1 | 0 | 0 | 1 | 2 | 0.00% |
| kir | 1 | 0 | 0 | 1 | 2 | 0.00% |
| kat | 1 | 0 | 0 | 1 | 2 | 0.00% |
| bcl | 1 | 0 | 0 | 1 | 2 | 0.00% |
| kan | 0 | 1 | 0 | 1 | 2 | 0.00% |
| tha | 0 | 0 | 1 | 1 | 2 | 0.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| all_records | 42,865 |
| all_unique_inputs | 41,526 |
| test_records | 4,283 |
| test_unique_inputs | 4,153 |
| train_records | 34,314 |
| train_unique_inputs | 33,247 |
| val_records | 4,268 |
| val_unique_inputs | 4,126 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| relations (per instance) | train / val | 0.1683 WARN | 34,314 / 4,268 | member of sports team 0%→0%, place of publication 0%→0%, grants 0%→1%, competition class 0%→1%, terminus location 0%→1% |
| relations (per instance) | train / test | 0.1659 WARN | 34,314 / 4,283 | sibling 0%→1%, military branch 0%→1%, has edition or translation 0%→0%, item operated 0%→0%, broadcast by 0%→1% |
| relations (per instance) | val / test | 0.2181 WARN | 4,268 / 4,283 | competition class 1%→0%, voice type 0%→1%, broadcast by 0%→1%, partner in business or sport 0%→0%, member of sports team 0%→0% |

Unseen in train, present in val, relations (per instance): fault ×2, scenographer ×1, place of origin (Switzerland) ×1, artistic director ×1, ICTV virus genome composition ×1, inscription mentions ×1, therapeutic area ×1, is pollinated by ×1, admissible rule in ×1, ortholog ×1, damaged ×1, UNESCO language status ×1.
Thin support in val, relations (per instance): 446 label(s) under 10 gold. nominee ×1, character designer ×1, depends on software ×1, foods traditionally associated ×1, reward program ×1, structure replaces ×1, central bank/issuer ×1, companion of ×1, crystal system ×1, Fach ×1, has contributing factor ×1, notation ×1, scenographer ×1, prime factor ×1, manifestation of ×1, issued by ×1, place of origin (Switzerland) ×1, parent peak ×1, cover art by ×1, artistic director ×1, twinned administrative body ×1, successful candidate ×1, fastest lap ×1, supplement to ×1, temporal range end ×1, season of club or team ×1, carries scientific instrument ×1, possible medical findings ×1, has list ×1, scheduled service destination ×1, Roman agnomen ×1, stepparent ×1, ICTV virus genome composition ×1, OSI Model layer location ×1, inscription mentions ×1, charted in ×1, engine configuration ×1, production designer ×1, therapeutic area ×1, contributor to the creative work or subject ×1, statement describes ×1, is pollinated by ×1, diaspora ×1, supported sports team ×1, has natural reservoir ×1, admissible rule in ×1, gens ×1, innervated by ×1, sports league level ×1, overlies ×1, does not have part ×1, archives at ×1, landscape architect ×1, tournament format ×1, day of week ×1, samples from work ×1, ortholog ×1, motto ×1, has pet ×1, league level below ×1, set in environment ×1, programmer ×1, demographics of topic ×1, statistical leader ×1, narrative role ×1, temporal range start ×1, original film format ×1, exclave of ×1, translator ×1, polymer of ×1, antagonist muscle ×1, codomain ×1, addressee ×1, penalty ×1, interaction ×1, majority opinion by ×1, microarchitecture ×1, given name version for other gender ×1, update method ×1, prohibits ×1, month of the year ×1, exhibition history ×1, approved by ×1, connector ×1, end cause ×1, main building contractor ×1, rector ×1, soundtrack release ×1, costume designer ×1, laws applied ×1, electoral district ×1, has seal, badge, or sigil ×1, official symbol ×1, captain ×1, color ×1, copyright status ×1, disease transmission process ×1, edibility ×1, has facet polytope ×1, is a list of ×1, next crossing downstream ×1, species kept ×1, animal breed ×1, delta of ×1, train depot ×1, base ×1, damaged ×1, microstructure ×1, develops from ×1, list of works ×1, UNESCO language status ×1, home port ×1, interchange station ×1, permanent resident of ×1, narrator ×1, era name ×1, province ×1, business model ×2, climbing route ×2, CPU ×2, category of associated people ×2, military casualty classification ×2, named by ×2, parent of this hybrid, breed, or cultivar ×2, set in period ×2, type of orbit ×2, valid in period ×2, vehicle normally used ×2, monomer of ×2, symptoms and signs ×2, commissioned by ×2, film editor ×2, academic thesis ×2, fandom ×2, godparent ×2, country of registry ×2, cuisine ×2, dialect of computer language ×2, output method ×2, solved by ×2, in opposition to ×2, depicted by ×2, capital of ×2, research site ×2, ancestral home ×2, economy of topic ×2, geomorphological unit ×2, party chief representative ×2, theme music ×2, peak bagging classification ×2, less than ×2, fault ×2, medical examination ×2, narrative motif ×2, screenwriter ×2, enclave within ×2, curator ×2, game mode ×2, diplomatic relation ×2, parliamentary group ×2, field of this occupation ×2, has anatomical branch ×2, origin of the watercourse ×2, officially opened by ×2, partnership with ×2, social classification ×2, political alignment ×2, capital ×2, influenced by ×2, drafted by ×2, office held by head of state ×2, contributing factor of ×2, trophy awarded ×2, has grammatical case ×2, instrumentation ×2, collection creator ×2, readable file format ×2, sports season of league or competition ×2, enemy of ×2, academic degree ×2, place of death ×2, destroyed ×2, Christian liturgical rite ×2, product certification ×2, librettist ×3, minor planet group ×3, computes solution to ×3, office contested ×3, dual to ×3, writable file format ×3, family name identical to this given name ×3, wing configuration ×3, legal form ×3, executive producer ×3, located in protected area ×3, recorded at studio or venue ×3, distribution format ×3, typically sells ×3, measures ×3, adjacent building ×3, character type ×3, place of burial ×3, emulates ×3, honorific suffix ×3, sidekick of ×3, sexual orientation ×3, afflicts ×3, language regulatory body ×3, victory ×3, feast day ×3, postsynaptic connection ×3, produced by ×3, writing system ×3, wears ×3, charge ×3, editor-in-chief ×3, amended by ×3, airline hub ×3, underlies ×3, derivative work ×3, lifestyle ×3, designed to carry ×3, league level above ×3, signatory ×3, professorship ×3, risk factor ×3, organizer ×3, music created for ×3, student of ×3, subject has role ×3, legislative body ×3, territory claimed by ×3, commander of (DEPRECATED) ×3, has phenotype ×3, has quality ×3, religious order ×3, first appearance ×3, points/goal scored by ×3, game mechanics ×3, follows ×3, source of energy ×3, intended public ×3, suggests the existence of ×3, implementation of ×4, appointed by ×4, has active ingredient ×4, next crossing upstream ×4, title of chess person ×4, flag ×4, health specialty ×4, greater than ×4, linguistic typology ×4, political ideology ×4, office held by head of government ×4, depicts ×4, mount ×4, subdivision of this unit ×4, newspaper format ×4, choreographer ×4, Gram staining ×4, contributed to creative work ×4, parent astronomical body ×4, maintained by ×4, series spin-off ×4, designated as terrorist by ×4, head coach ×4, performed at ×4, shape ×4, director / manager ×4, proved by ×4, advertises ×4, has boundary ×4, public holiday ×4, discography ×4, iconographic symbol ×4, measurement scale ×4, studied by ×4, industry ×4, structure replaced by ×4, cause of death ×4, used by ×4, director of archaeological fieldwork ×4, house publication ×4, expressed in ×5, season ×5, office held by head of the organization ×5, mountains classification ×5, official language ×5, day in year for periodic occurrence ×5, objective of project or action ×5, fabrication method ×5, honorific prefix ×5, sponsor ×5, top-level Internet domain ×5, domain of saint or deity ×5, taxonomic type ×5, possible treatment ×5, youth wing ×5, designed by ×5, immediate cause of ×5, consecrator ×5, drainage basin ×5, primary destinations ×5, subclass of ×5, replaces ×5, manner of death ×5, vaccine for ×5, antiparticle ×5, facet of ×5, separated from ×5, anatomical branch of ×5, sexually homologous with ×5, canonization status ×5, general classification of race participants ×5, narrative location ×5, adjacent station ×5, inspired by ×5, home world ×6, native language ×6, musical conductor ×6, killed by ×6, co-driver ×6, has immediate cause ×6, cell component ×6, encodes ×6, foundational text ×6, innervates ×6, location of creation ×6, business division ×6, currency ×6, media franchise ×6, location of landing ×6, operating system ×6, defendant ×6, has cause ×6, uses ×6, type of variable star ×6, located in the statistical territorial entity ×6, GUI toolkit or framework ×6, perpetrator ×6, student ×6, languages spoken, written or signed ×6, lyrics by ×6, armament ×6, patron saint ×6, partially coincident with ×6, production company ×6, candidacy in election ×6, offers view on ×6, different from ×6, executive body ×7, organization directed by the office or position ×7, carries ×7, central bank ×7, commanded by ×7, mascot ×7, members have occupation ×7, contains settlement ×7, dedicated to ×7, main regulatory text ×7, programming language ×7, target ×7, transport network ×7, significant place ×7, measured physical quantity ×7, affiliation ×7, taxon rank ×7, found in taxon ×7, dialect of ×7, board member ×7, followed by ×7, coextensive with ×7, copyright holder ×7, interested in ×7, takes place in fictional universe ×7, described by source ×7, chromosome ×7, owner of ×7, child astronomical body ×7, voice type ×8, victim ×8, historical region ×8, writing language ×8, qualifies for event ×8, tracklist ×8, culture ×8, highest judicial authority ×8, official residence ×8, substitute/deputy/replacement of office/officeholder ×8, filming location ×8, editor ×8, inflows ×8, merged into ×8, cathedral ×8, diocese ×8, reservoir created ×8, next higher rank ×8, track gauge ×8, based on ×8, lake outflow ×8, structural engineer ×8, has effect ×8, has part(s) of the class ×8, cause of destruction ×8, site of astronomical discovery ×8, educated at ×8, worshipped by ×9, operating area ×9, recommended unit of measurement ×9, has written for ×9, model item ×9, airline alliance ×9, broadcast by ×9, appears in the form of ×9, collection ×9, commemorates ×9, vehicle ×9, school district ×9, start point ×9, made from material ×9, country for sport ×9, space launch vehicle ×9, illustrator ×9, legislated by ×9, vessel class ×9, powered by ×9, edition or translation of ×9, family name ×9, dam ×9, headquarters location ×9, applies to jurisdiction ×9, crosses ×9, home venue ×9, territory overlaps ×9, shares border with ×9, member of sports team ×9, distributed by ×9, head of government ×9, highest point ×9, director of photography ×9, work location ×9, anthem ×9, part of the series ×9, country of origin ×9, main subject ×9, coach of sports team ×9.
Unseen in train, present in test, relations (per instance): defender ×1, bowling style ×1, hybrid ×1, course ×1, has vertex figure ×1, therapeutic area ×1, UNESCO language status ×1, prosecutor ×1, Thibaudeau classification ×1, mountain ×1.
Thin support in test, relations (per instance): 434 label(s) under 10 gold. defender ×1, bowling style ×1, academic degree ×1, charge ×1, day in year for periodic occurrence ×1, narrative motif ×1, plot expanded in ×1, risk factor ×1, office held by head of state ×1, doctoral student ×1, develops from ×1, fandom ×1, character designer ×1, majority opinion by ×1, introduced feature ×1, may prevent ×1, colorist ×1, country of registry ×1, via ×1, cites work ×1, parliamentary group ×1, film editor ×1, narrator ×1, has tense ×1, parent peak ×1, subdivision of this unit ×1, appointed by ×1, guidance system ×1, facial hair ×1, solid solution series with ×1, stepparent ×1, hybrid ×1, supports programming language ×1, uses capitalization for ×1, delta of ×1, adjacent building ×1, does not have part ×1, programmer ×1, military unit ×1, postsynaptic connection ×1, OSI Model layer location ×1, combination classification ×1, sexually homologous with ×1, exhibition history ×1, complies with ×1, course ×1, has vertex figure ×1, overlies ×1, therapeutic area ×1, appears in the form of ×1, writable file format ×1, Fach ×1, possible medical findings ×1, UNESCO language status ×1, unmarried partner ×1, cuisine ×1, doctoral advisor ×1, flag ×1, interchange station ×1, animator ×1, approved by ×1, military casualty classification ×1, stereoisomer of ×1, next lower rank ×1, has fruit type ×1, officially opened by ×1, prosecutor ×1, solved by ×1, structure replaces ×1, temporal range end ×1, modified version of ×1, conjugate acid ×1, grammatical option indicates ×1, emergency phone number ×1, landscape architect ×1, active ingredient in ×1, mountains classification ×1, cover art by ×1, geomorphological unit ×1, translator ×1, peak bagging classification ×1, patron saint ×1, microstructure ×1, end cause ×1, curator ×1, parent of this hybrid, breed, or cultivar ×1, madhhab ×1, Christian liturgical rite ×1, named by ×1, next crossing upstream ×1, Thibaudeau classification ×1, art director ×1, diplomatic relation ×1, record held ×1, has part(s) of the class ×1, emulates ×1, has melody ×1, mountain ×1, expansion of ×2, destroyed ×2, has grammatical case ×2, superhuman feature or ability ×2, supported sports team ×2, wears ×2, carries scientific instrument ×2, prerequisite ×2, sponsor ×2, source of energy ×2, child astronomical body ×2, conjugate base ×2, depends on software ×2, soundtrack release ×2, intended public ×2, people or cargo transported ×2, molecular function ×2, motto ×2, designated as terrorist by ×2, expressed in ×2, perpetrator ×2, depicted by ×2, speaker ×2, language regulatory body ×2, statement describes ×2, muscle insertion ×2, depicts ×2, field of this occupation ×2, subject has role ×2, measures ×2, typically sells ×2, youth wing ×2, set in environment ×2, innervated by ×2, shares border with ×2, sidekick of ×2, top-level Internet domain ×2, rector ×2, legal form ×2, list of works ×2, diaspora ×2, site of astronomical discovery ×2, office contested ×2, official residence ×2, twinned administrative body ×2, fastest lap ×2, type foundry ×2, addressee ×2, calculated from ×2, game mechanics ×2, iconographic symbol ×2, has facet polytope ×2, territory claimed by ×2, lake outflow ×2, polymer of ×2, has list ×2, final event ×2, innervates ×2, successful candidate ×2, drafted by ×2, academic thesis ×2, GUI toolkit or framework ×2, has superpartner ×2, temporal range start ×2, funder ×2, general manager ×2, influenced by ×2, connector ×2, family name identical to this given name ×2, engine configuration ×2, medical examination ×2, party chief representative ×2, foundational text ×3, crystal system ×3, music created for ×3, animal breed ×3, central bank ×3, consecrator ×3, designed to carry ×3, home port ×3, input method ×3, reward program ×3, trophy awarded ×3, amended by ×3, family name ×3, structure replaced by ×3, aspect ratio (W:H) ×3, monomer of ×3, official language ×3, contains ×3, offers view on ×3, dialect of computer language ×3, defendant ×3, color ×3, wing configuration ×3, members have occupation ×3, season ×3, main food source ×3, director of archaeological fieldwork ×3, contains settlement ×3, coextensive with ×3, demonym of ×3, less than ×3, set in period ×3, taxon synonym ×3, points classification ×3, cell component ×3, contributing factor of ×3, immediate cause of ×3, time period ×3, studied by ×3, given name version for other gender ×3, legislative body ×3, enclave within ×3, computes solution to ×3, capital of ×3, sports season of league or competition ×3, economy of topic ×3, has pet ×3, sex or gender ×3, practiced by ×3, political ideology ×3, derivative work ×3, symptoms and signs ×3, theme music ×3, coat of arms ×3, voice actor ×3, used by ×3, editor-in-chief ×3, recorded at studio or venue ×3, demographics of topic ×3, organization directed by the office or position ×4, central bank/issuer ×4, encodes ×4, mod of ×4, lifestyle ×4, objective of project or action ×4, official religion ×4, regulated by ×4, place of detention ×4, dual to ×4, director of photography ×4, enemy of ×4, possible treatment ×4, archives at ×4, minor planet group ×4, discography ×4, has boundary ×4, in opposition to ×4, candidate ×4, capital ×4, contributed to creative work ×4, described by source ×4, head coach ×4, media franchise ×4, ancestral home ×4, greater than ×4, victory ×4, executive producer ×4, structural engineer ×4, honorific suffix ×4, location of formation ×4, victim ×4, foods traditionally associated ×4, student ×4, separated from ×4, has anatomical branch ×4, character type ×4, measurement scale ×4, mascot ×4, instrumentation ×4, highest point ×5, native language ×5, military rank ×5, league level above ×5, home world ×5, game mode ×5, performed at ×5, powered by ×5, this zoological name is coordinate with ×5, manifestation of ×5, office held by head of government ×5, public holiday ×5, social classification ×5, title of chess person ×5, type of orbit ×5, type of variable star ×5, wheel arrangement ×5, director / manager ×5, commissioned by ×5, manner of death ×5, track gauge ×5, points/goal scored by ×5, physically interacts with ×5, measured physical quantity ×5, captain ×5, location of landing ×5, inspired by ×5, dedicated to ×5, allegiance ×5, business model ×5, copyright holder ×5, drainage basin ×5, parent club ×5, health specialty ×5, office held by head of the organization ×5, writing system ×5, airline alliance ×5, underlies ×5, candidacy in election ×5, has contributing factor ×5, political alignment ×5, relative ×5, pole position ×6, vehicle normally used ×6, canonization status ×6, from narrative universe ×6, house publication ×6, lake on watercourse ×6, software engine ×6, target ×6, programming language ×6, series spin-off ×6, narrative location ×6, anthem ×6, afflicts ×6, distribution format ×6, vaccine for ×6, mount ×6, business division ×6, educated at ×6, taxon rank ×6, associated electoral district ×6, fabrication method ×6, implementation of ×6, sexual orientation ×6, territory overlaps ×6, history of topic ×6, organizer ×6, contributor to the creative work or subject ×6, IUCN conservation status ×7, legislated by ×7, general classification of race participants ×7, location of creation ×7, penalty ×7, published in ×7, this taxon is source of ×7, competition class ×7, cause of death ×7, programming paradigm ×7, astronaut mission ×7, operating system ×7, CPU ×7, armament ×7, statistical leader ×7, interested in ×7, origin of the watercourse ×7, partially coincident with ×7, follows ×7, designed by ×7, league level below ×7, facet of ×7, Gram staining ×7, qualifies for event ×7, anatomical branch of ×7, replaces ×7, connecting service ×7, brand ×8, librettist ×8, space launch vehicle ×8, diocese ×8, parent astronomical body ×8, authority ×8, dialect of ×8, merged into ×8, natural product of taxon ×8, substitute/deputy/replacement of office/officeholder ×8, transport network ×8, architectural style ×8, inflows ×8, genetic association ×8, production company ×8, start point ×8, signatory ×8, co-driver ×8, form of creative work ×8, recommended unit of measurement ×8, opposite of ×8, producer ×8, place of death ×8, lyrics by ×8, item operated ×8, domain of saint or deity ×8, screenwriter ×8, student of ×8, league system ×8, surface played on ×8, affiliation ×8, school district ×8, partner in business or sport ×8, edition or translation of ×8, location of first performance ×8, primary destinations ×8, has quality ×8, part of the series ×8, language of work or name ×8, place of burial ×8, copyright license ×9, found in taxon ×9, stock market index ×9, destination point ×9, musical conductor ×9, culture ×9, editor ×9, worshipped by ×9, killed by ×9, industry ×9, tracklist ×9, coach of sports team ×9, subclass of ×9, is the study of ×9, work location ×9, sport ×9, has use ×9, highest judicial authority ×9, taxonomic type ×9, said to be the same as ×9, head of government ×9, maintained by ×9, linguistic typology ×9, ethnic group ×9, residence ×9, commanded by ×9, connects with ×9, publisher ×9, based on ×9.
Unseen in train, present in all, relations (per instance): fault ×2, therapeutic area ×2, UNESCO language status ×2, defender ×1, bowling style ×1, scenographer ×1, place of origin (Switzerland) ×1, artistic director ×1, ICTV virus genome composition ×1, inscription mentions ×1, hybrid ×1, is pollinated by ×1, admissible rule in ×1, course ×1, has vertex figure ×1, ortholog ×1, prosecutor ×1, damaged ×1, Thibaudeau classification ×1, mountain ×1.
Thin support in all, relations (per instance): 268 label(s) under 10 gold. defender ×1, bowling style ×1, epithet ×1, GPU ×1, catalog ×1, external auditor ×1, muscle origin ×1, overrules ×1, radix ×1, scenographer ×1, socket supported ×1, used metre ×1, fictional analog of ×1, taxon found at location ×1, recording location ×1, place of origin (Switzerland) ×1, start grade ×1, artistic director ×1, activating neurotransmitter ×1, has member ×1, ecoregion (WWF) ×1, including ×1, pendant of ×1, real estate developer ×1, storyboard artist ×1, oath made by ×1, dissertation submitted to ×1, established from medical condition ×1, ICTV virus genome composition ×1, inscription mentions ×1, card network ×1, currency symbol description ×1, hybrid ×1, mode of inheritance ×1, bibliography ×1, is pollinated by ×1, admissible rule in ×1, index case of ×1, gait ×1, reissue of ×1, has pattern ×1, course ×1, has vertex figure ×1, ortholog ×1, founder ×1, fictional character ×1, film crew member ×1, entry receptor ×1, source of income ×1, decay mode ×1, follower of ×1, criterion used ×1, exonerated of ×1, prosecutor ×1, Vox-ATypI classification ×1, has phoneme ×1, leader ×1, record or record progression ×1, place of devotion ×1, academic appointment ×1, fossil found in this unit ×1, indexed in bibliographic review ×1, issuing agent of work ×1, young rider classification ×1, presynaptic connection ×1, has role in modeling ×1, means of locomotion ×1, generalization of ×1, located on linear feature ×1, reward ×1, inhibitor of ×1, facilitates flow of ×1, damaged ×1, Thibaudeau classification ×1, associated hazard ×1, mushroom ecological type ×1, parliamentary term ×1, election called by ×1, kit supplier ×1, blocker of ×1, mountain ×1, nominee ×2, base ×2, original film format ×2, by-product of ×2, display technology ×2, has tense ×2, open data portal ×2, companion of ×2, reply to ×2, tournament format ×2, website account on ×2, may prevent ×2, decreased expression in ×2, derived from organism type ×2, distributary ×2, item disputed by ×2, supplement to ×2, biological variant of ×2, Roman agnomen ×2, encoded by ×2, undercarriage ×2, metrically compatible typeface ×2, route of administration ×2, fault ×2, therapeutic area ×2, relative to ×2, avionics ×2, production statistics ×2, first aid measures ×2, invasive to ×2, samples from work ×2, has facility ×2, side effect ×2, UNESCO language status ×2, volcano observatory ×2, habitat ×2, codomain ×2, chief operating officer ×2, regulates (molecular biology) ×2, grammatical option indicates ×2, promoted ×2, province ×2, religious movement ×2, port of registry ×3, climbing route ×3, combination classification ×3, comorbidity ×3, intangible cultural heritage status ×3, statement is subject of ×3, visitor centre ×3, permanent resident of ×3, typing discipline ×3, does not have quality ×3, doctoral student ×3, edibility ×3, plaintiff ×3, guidance system ×3, introduced feature ×3, facial hair ×3, symbolizes ×3, art director ×3, investor ×3, possible medical findings ×3, muscle insertion ×3, lowest point ×3, binding of software library ×3, relegated ×3, has listed ingredient ×3, electoral district ×3, together with ×3, disjoint union of ×3, repealed by ×3, official color ×3, has superpartner ×3, protocol ×3, is metaclass for ×3, emergency phone number ×4, day of week ×4, order of battle ×4, category of associated people ×4, Fach ×4, plot expanded in ×4, prime factor ×4, record held ×4, solid solution series with ×4, substrate of ×4, nominated by ×4, by-product ×4, doctoral advisor ×4, style of karate ×4, godparent ×4, coolant ×4, ratified by ×4, temporal range end ×4, laws applied ×4, landscape architect ×4, decays to ×4, demonym of ×4, next crossing downstream ×4, output method ×4, religious occupation ×4, costume designer ×4, modified version of ×4, elector ×4, list of works ×4, asteroid spectral type ×4, given name ×4, has census ×4, partnership with ×4, professorship ×4, era name ×4, month of the year ×5, adapted by ×5, complies with ×5, official podcast ×5, conjugate acid ×5, discovery method ×5, notation ×5, uses capitalization for ×5, conjugate base ×5, donated by ×5, venous drainage ×5, majority opinion by ×5, repeals ×5, season of club or team ×5, military unit ×5, research site ×5, produced sound ×5, microstructure ×5, has melody ×5, choral conductor ×5, exemplar of ×5, launch contractor ×5, collection creator ×5, animator ×5, calculated from ×6, fandom ×6, issued by ×6, narrative motif ×6, antagonist muscle ×6, arterial supply ×6, develops from ×6, political coalition ×6, supports programming language ×6, sports league level ×6, funder ×6, microarchitecture ×6, curator ×6, suggests the existence of ×6, disease transmission process ×6, copyright status ×7, character designer ×7, narrative role ×7, train depot ×7, type locality (geology) ×7, worn by ×7, production designer ×7, theorised by ×7, valid in period ×7, official symbol ×7, referee ×7, cites work ×7, postsynaptic connection ×7, end cause ×7, chassis ×7, programmer ×8, stereoisomer of ×8, aspect ratio (W:H) ×8, significant drug interaction ×8, does not have part ×8, OSI Model layer location ×8, general manager ×8, has seal, badge, or sigil ×8, has facet polytope ×8, rector ×9, has phenotype ×9, size designation ×9, tonality ×9, via ×9, geography of topic ×9, taxon synonym ×9, colorist ×9, main building contractor ×9, peak bagging classification ×9, exhibition history ×9, is a list of ×9, geomorphological unit ×9.

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

None.

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| conferred by | 182 | 0.53% |
| located in the administrative territorial entity | 182 | 0.53% |
| director | 179 | 0.52% |
| cast member | 177 | 0.52% |
| architect | 169 | 0.49% |
| platform | 166 | 0.48% |
| present in work | 166 | 0.48% |
| father | 164 | 0.48% |
| occupation | 162 | 0.47% |
| conflict | 162 | 0.47% |
| league | 161 | 0.47% |
| record label | 160 | 0.47% |
| award received | 159 | 0.46% |
| historic county | 157 | 0.46% |
| employer | 157 | 0.46% |
| nominated for | 155 | 0.45% |
| original broadcaster | 154 | 0.45% |
| member of sports team | 153 | 0.45% |
| heritage designation | 152 | 0.44% |
| performer | 152 | 0.44% |
| place served by transport hub | 151 | 0.44% |
| country | 150 | 0.44% |
| genre | 149 | 0.43% |
| author | 148 | 0.43% |
| location | 147 | 0.43% |
| licensed to broadcast to | 147 | 0.43% |
| member of political party | 146 | 0.43% |
| notable work | 145 | 0.42% |
| located in or next to body of water | 145 | 0.42% |
| member of | 143 | 0.42% |
| item operated | 143 | 0.42% |
| participant in | 142 | 0.41% |
| characters | 142 | 0.41% |
| creator | 138 | 0.40% |
| connecting line | 138 | 0.40% |
| mother | 137 | 0.40% |
| discoverer or inventor | 136 | 0.40% |
| located in/on physical feature | 136 | 0.40% |
| operator | 136 | 0.40% |
| military branch | 134 | 0.39% |
| movement | 133 | 0.39% |
| position played on team / speciality | 132 | 0.38% |
| position held | 132 | 0.38% |
| surface played on | 131 | 0.38% |
| mountain range | 131 | 0.38% |
| parent organization | 131 | 0.38% |
| instrument | 131 | 0.38% |
| family | 130 | 0.38% |
| mouth of the watercourse | 129 | 0.38% |
| participant | 126 | 0.37% |
| basin country | 125 | 0.36% |
| sport | 124 | 0.36% |
| ammunition | 122 | 0.36% |
| stock market index | 120 | 0.35% |
| secretary general | 120 | 0.35% |
| participating team | 119 | 0.35% |
| partner in business or sport | 118 | 0.34% |
| located in the present-day administrative territorial entity | 118 | 0.34% |
| spouse | 117 | 0.34% |
| significant event | 116 | 0.34% |
| owned by | 116 | 0.34% |
| located on astronomical body | 115 | 0.34% |
| parent taxon | 115 | 0.34% |
| founded by | 114 | 0.33% |
| educated at | 114 | 0.33% |
| officeholder | 114 | 0.33% |
| instance of | 113 | 0.33% |
| voice type | 112 | 0.33% |
| crosses | 112 | 0.33% |
| online service | 111 | 0.32% |
| represents | 110 | 0.32% |
| part of | 110 | 0.32% |
| composer | 110 | 0.32% |
| place of detention | 109 | 0.32% |
| sibling | 109 | 0.32% |
| located in the statistical territorial entity | 108 | 0.31% |
| continent | 108 | 0.31% |
| chairperson | 108 | 0.31% |
| terminus location | 107 | 0.31% |
| developer | 107 | 0.31% |
| headquarters location | 107 | 0.31% |
| competition class | 106 | 0.31% |
| presenter | 105 | 0.31% |
| religion or worldview | 105 | 0.31% |
| dam | 104 | 0.30% |
| noble title | 104 | 0.30% |
| language used | 104 | 0.30% |
| form of creative work | 102 | 0.30% |
| medical condition | 102 | 0.30% |
| child | 102 | 0.30% |
| publisher | 102 | 0.30% |
| has subsidiary | 102 | 0.30% |
| grants | 101 | 0.29% |
| package management system | 101 | 0.29% |
| has use | 100 | 0.29% |
| location of formation | 100 | 0.29% |
| product or material produced | 100 | 0.29% |
| historical region | 99 | 0.29% |
| home venue | 99 | 0.29% |
| broadcast by | 98 | 0.29% |
| crew member(s) | 97 | 0.28% |
| represented by | 97 | 0.28% |
| takes place in fictional universe | 96 | 0.28% |
| astronaut mission | 96 | 0.28% |
| this zoological name is coordinate with | 95 | 0.28% |
| located in time zone | 95 | 0.28% |
| production company | 94 | 0.27% |
| terminus | 94 | 0.27% |
| drug or therapy used for treatment | 93 | 0.27% |
| recommended unit of measurement | 93 | 0.27% |
| member of the crew of | 93 | 0.27% |
| winner | 93 | 0.27% |
| model item | 92 | 0.27% |
| found in taxon | 92 | 0.27% |
| commemorates | 92 | 0.27% |
| work location | 92 | 0.27% |
| stock exchange | 91 | 0.27% |
| distributed by | 91 | 0.27% |
| manufacturer | 91 | 0.27% |
| replaced by | 91 | 0.27% |
| this taxon is source of | 90 | 0.26% |
| convicted of | 90 | 0.26% |
| incarnation of | 90 | 0.26% |
| carries | 90 | 0.26% |
| said to be the same as | 90 | 0.26% |
| main subject | 90 | 0.26% |
| taxonomic type | 89 | 0.26% |
| musical conductor | 88 | 0.26% |
| medical condition treated | 88 | 0.26% |
| vehicle | 88 | 0.26% |
| location of discovery | 88 | 0.26% |
| biological process | 88 | 0.26% |
| contains the administrative territorial entity | 88 | 0.26% |
| school district | 86 | 0.25% |
| occupant | 86 | 0.25% |
| court | 85 | 0.25% |
| fictional universe described in | 85 | 0.25% |
| is the study of | 85 | 0.25% |
| ethnic group | 85 | 0.25% |
| field of work | 85 | 0.25% |
| published in | 84 | 0.24% |
| commanded by | 84 | 0.24% |
| has part(s) | 84 | 0.24% |
| named after | 84 | 0.24% |
| connects with | 84 | 0.24% |
| country of citizenship | 84 | 0.24% |
| has effect | 84 | 0.24% |
| parent club | 83 | 0.24% |
| military rank | 83 | 0.24% |
| place of publication | 83 | 0.24% |
| tracklist | 83 | 0.24% |
| constellation | 83 | 0.24% |
| significant person | 83 | 0.24% |
| filming location | 82 | 0.24% |
| has written for | 82 | 0.24% |
| owner of | 82 | 0.24% |
| indigenous to | 82 | 0.24% |
| track gauge | 81 | 0.24% |
| sports discipline competed in | 81 | 0.24% |
| uses | 81 | 0.24% |
| dialect of | 80 | 0.23% |
| legislated by | 80 | 0.23% |
| merged into | 80 | 0.23% |
| pole position | 80 | 0.23% |
| country of origin | 80 | 0.23% |
| writing language | 79 | 0.23% |
| head of state | 79 | 0.23% |
| original language of film or TV show | 78 | 0.23% |
| worshipped by | 78 | 0.23% |
| located on street | 78 | 0.23% |
| country for sport | 78 | 0.23% |
| collection | 77 | 0.22% |
| reservoir created | 77 | 0.22% |
| dedicated to | 77 | 0.22% |
| cathedral | 77 | 0.22% |
| airline alliance | 76 | 0.22% |
| vessel class | 76 | 0.22% |
| objective of project or action | 76 | 0.22% |
| head of government | 76 | 0.22% |
| IUCN conservation status | 75 | 0.22% |
| league system | 74 | 0.22% |
| produced by | 74 | 0.22% |
| after a work by | 74 | 0.22% |
| opposite of | 74 | 0.22% |
| operating area | 73 | 0.21% |
| location of first performance | 73 | 0.21% |
| start point | 73 | 0.21% |
| brand | 73 | 0.21% |
| anatomical location | 73 | 0.21% |
| wheel arrangement | 73 | 0.21% |
| place of birth | 73 | 0.21% |
| producer | 73 | 0.21% |
| from narrative universe | 72 | 0.21% |
| copyright license | 72 | 0.21% |
| standards body | 72 | 0.21% |
| followed by | 72 | 0.21% |
| has works in the collection | 71 | 0.21% |
| manner of death | 71 | 0.21% |
| chief executive officer | 71 | 0.21% |
| culture | 70 | 0.20% |
| transport network | 70 | 0.20% |
| allegiance | 70 | 0.20% |
| made from material | 70 | 0.20% |
| currency | 69 | 0.20% |
| architectural style | 69 | 0.20% |
| editor | 69 | 0.20% |
| co-driver | 68 | 0.20% |
| commander of (DEPRECATED) | 68 | 0.20% |
| taxon rank | 68 | 0.20% |
| next higher rank | 68 | 0.20% |
| residence | 68 | 0.20% |
| located in protected area | 67 | 0.20% |
| interested in | 67 | 0.20% |
| basic form of government | 67 | 0.20% |
| operating system | 67 | 0.20% |
| facet of | 67 | 0.20% |
| librettist | 66 | 0.19% |
| natural product of taxon | 65 | 0.19% |
| has cause | 65 | 0.19% |
| lyrics by | 65 | 0.19% |
| has edition or translation | 64 | 0.19% |
| diocese | 64 | 0.19% |
| significant place | 64 | 0.19% |
| history of topic | 64 | 0.19% |
| discography | 64 | 0.19% |
| voice actor | 64 | 0.19% |
| has part(s) of the class | 64 | 0.19% |
| illustrator | 63 | 0.18% |
| space launch vehicle | 63 | 0.18% |
| part of the series | 63 | 0.18% |
| physically interacts with | 62 | 0.18% |
| chromosome | 62 | 0.18% |
| programming language | 62 | 0.18% |
| board member | 61 | 0.18% |
| highest judicial authority | 61 | 0.18% |
| based on | 61 | 0.18% |
| target | 60 | 0.17% |
| edition or translation of | 60 | 0.17% |
| copyright holder | 60 | 0.17% |
| has immediate cause | 60 | 0.17% |
| executive body | 60 | 0.17% |
| qualifies for event | 60 | 0.17% |
| applies to jurisdiction | 60 | 0.17% |
| parent astronomical body | 59 | 0.17% |
| distribution format | 59 | 0.17% |
| encodes | 59 | 0.17% |
| territory overlaps | 59 | 0.17% |
| endemic to | 59 | 0.17% |
| relative | 59 | 0.17% |
| replaces | 59 | 0.17% |
| origin of the watercourse | 58 | 0.17% |
| drainage basin | 58 | 0.17% |
| genetic association | 58 | 0.17% |
| measured physical quantity | 58 | 0.17% |
| main regulatory text | 58 | 0.17% |
| affiliation | 58 | 0.17% |
| has quality | 58 | 0.17% |
| subclass of | 58 | 0.17% |
| powered by | 57 | 0.17% |
| commissioned by | 57 | 0.17% |
| lake on watercourse | 57 | 0.17% |
| director / manager | 57 | 0.17% |
| language of work or name | 57 | 0.17% |
| candidate | 57 | 0.17% |
| unmarried partner | 57 | 0.17% |
| candidacy in election | 57 | 0.17% |
| substitute/deputy/replacement of office/officeholder | 56 | 0.16% |
| destination point | 56 | 0.16% |
| killed by | 56 | 0.16% |
| radio format | 56 | 0.16% |
| designed by | 56 | 0.16% |
| head coach | 56 | 0.16% |
| subject has role | 55 | 0.16% |
| primary destinations | 55 | 0.16% |
| tributary | 55 | 0.16% |
| canonization status | 54 | 0.16% |
| inflows | 54 | 0.16% |
| location of creation | 54 | 0.16% |
| house publication | 54 | 0.16% |
| narrative location | 54 | 0.16% |
| coach of sports team | 54 | 0.16% |
| languages spoken, written or signed | 54 | 0.16% |
| organizer | 54 | 0.16% |
| maintained by | 54 | 0.16% |
| time period | 54 | 0.16% |
| adjacent station | 54 | 0.16% |
| follows | 54 | 0.16% |
| enemy of | 53 | 0.15% |
| authority | 52 | 0.15% |
| general classification of race participants | 52 | 0.15% |
| official language | 51 | 0.15% |
| fabrication method | 51 | 0.15% |
| official religion | 51 | 0.15% |
| anthem | 51 | 0.15% |
| contributed to creative work | 51 | 0.15% |
| legal form | 51 | 0.15% |
| political ideology | 51 | 0.15% |
| native language | 50 | 0.15% |
| family name | 49 | 0.14% |
| first appearance | 49 | 0.14% |
| mascot | 49 | 0.14% |
| place of burial | 49 | 0.14% |
| signatory | 49 | 0.14% |
| student of | 49 | 0.14% |
| lake outflow | 48 | 0.14% |
| office held by head of the organization | 48 | 0.14% |
| cell component | 48 | 0.14% |
| honorific prefix | 48 | 0.14% |
| consecrator | 48 | 0.14% |
| public holiday | 48 | 0.14% |
| victory | 48 | 0.14% |
| different from | 48 | 0.14% |
| foundational text | 47 | 0.14% |
| field of this occupation | 47 | 0.14% |
| writing system | 47 | 0.14% |
| cause of death | 47 | 0.14% |
| industry | 47 | 0.14% |
| religious order | 47 | 0.14% |
| shares border with | 47 | 0.14% |
| health specialty | 46 | 0.13% |
| political alignment | 46 | 0.13% |
| perpetrator | 45 | 0.13% |
| sex or gender | 45 | 0.13% |
| official residence | 45 | 0.13% |
| child astronomical body | 45 | 0.13% |
| armament | 44 | 0.13% |
| coextensive with | 44 | 0.13% |
| implementation of | 44 | 0.13% |
| place of death | 44 | 0.13% |
| domain of saint or deity | 43 | 0.13% |
| ancestral home | 43 | 0.13% |
| choreographer | 43 | 0.13% |
| business division | 43 | 0.13% |
| described by source | 43 | 0.13% |
| structural engineer | 42 | 0.12% |
| sponsor | 42 | 0.12% |
| media franchise | 42 | 0.12% |
| anatomical branch of | 42 | 0.12% |
| location of landing | 42 | 0.12% |
| game mechanics | 41 | 0.12% |
| Gram staining | 41 | 0.12% |
| vehicle normally used | 41 | 0.12% |
| defendant | 41 | 0.12% |
| screenwriter | 41 | 0.12% |
| symptoms and signs | 40 | 0.12% |
| engine configuration | 40 | 0.12% |
| legislative body | 40 | 0.12% |
| connecting service | 40 | 0.12% |
| programming paradigm | 39 | 0.11% |
| highest point | 39 | 0.11% |
| immediate cause of | 39 | 0.11% |
| used by | 39 | 0.11% |
| cause of destruction | 38 | 0.11% |
| linguistic typology | 38 | 0.11% |
| airline hub | 38 | 0.11% |
| partially coincident with | 38 | 0.11% |
| organization directed by the office or position | 37 | 0.11% |
| league level above | 37 | 0.11% |
| title of chess person | 37 | 0.11% |
| inspired by | 37 | 0.11% |
| manifestation of | 37 | 0.11% |
| derivative work | 37 | 0.11% |
| subdivision of this unit | 36 | 0.10% |
| newspaper format | 36 | 0.10% |
| office held by head of government | 36 | 0.10% |
| contains | 35 | 0.10% |
| software engine | 35 | 0.10% |
| source of energy | 35 | 0.10% |
| fastest lap | 34 | 0.10% |
| depicts | 34 | 0.10% |
| capital of | 34 | 0.10% |
| series spin-off | 33 | 0.10% |
| afflicts | 33 | 0.10% |
| practiced by | 33 | 0.10% |
| flag | 32 | 0.09% |
| depicted by | 32 | 0.09% |
| has anatomical branch | 32 | 0.09% |
| director of photography | 32 | 0.09% |
| dual to | 31 | 0.09% |
| type of variable star | 31 | 0.09% |
| home world | 31 | 0.09% |
| GUI toolkit or framework | 31 | 0.09% |
| possible treatment | 31 | 0.09% |
| site of astronomical discovery | 31 | 0.09% |
| successful candidate | 30 | 0.09% |
| advertises | 30 | 0.09% |
| character type | 29 | 0.08% |
| measures | 29 | 0.08% |
| minor planet group | 29 | 0.08% |
| office held by head of state | 28 | 0.08% |
| instrumentation | 28 | 0.08% |
| game mode | 28 | 0.08% |
| writable file format | 28 | 0.08% |
| color | 28 | 0.08% |
| sexually homologous with | 28 | 0.08% |
| league level below | 28 | 0.08% |
| enclave within | 27 | 0.08% |
| contributor to the creative work or subject | 27 | 0.08% |
| interchange station | 27 | 0.08% |
| music created for | 27 | 0.08% |
| capital | 27 | 0.08% |
| mount | 26 | 0.08% |
| superhuman feature or ability | 26 | 0.08% |
| central bank | 26 | 0.08% |
| separated from | 26 | 0.08% |
| territory claimed by | 26 | 0.08% |
| contains settlement | 25 | 0.07% |
| less than | 25 | 0.07% |
| interaction | 25 | 0.07% |
| director of archaeological fieldwork | 25 | 0.07% |
| editor-in-chief | 25 | 0.07% |
| party chief representative | 25 | 0.07% |
| sports season of league or competition | 25 | 0.07% |
| patron saint | 24 | 0.07% |
| monomer of | 24 | 0.07% |
| molecular function | 24 | 0.07% |
| central bank/issuer | 24 | 0.07% |
| drafted by | 24 | 0.07% |
| studied by | 24 | 0.07% |
| speaker | 23 | 0.07% |
| economy of topic | 23 | 0.07% |
| members have occupation | 23 | 0.07% |
| set in period | 23 | 0.07% |
| social classification | 23 | 0.07% |
| victim | 23 | 0.07% |
| type of orbit | 22 | 0.06% |
| measurement scale | 22 | 0.06% |
| cuisine | 22 | 0.06% |
| contributing factor of | 22 | 0.06% |
| sexual orientation | 22 | 0.06% |
| associated electoral district | 21 | 0.06% |
| has pet | 21 | 0.06% |
| home port | 21 | 0.06% |
| student | 21 | 0.06% |
| input method | 21 | 0.06% |
| language regulatory body | 21 | 0.06% |
| country of registry | 21 | 0.06% |
| wing configuration | 20 | 0.06% |
| designated as terrorist by | 20 | 0.06% |
| archives at | 20 | 0.06% |
| overlies | 20 | 0.06% |
| sidekick of | 20 | 0.06% |
| youth wing | 20 | 0.06% |
| business model | 20 | 0.06% |
| final event | 20 | 0.06% |
| greater than | 20 | 0.06% |
| mother house | 20 | 0.06% |
| readable file format | 20 | 0.06% |
| CPU | 20 | 0.06% |
| scheduled service destination | 20 | 0.06% |
| risk factor | 19 | 0.06% |
| statistical leader | 19 | 0.06% |
| animal breed | 19 | 0.06% |
| active ingredient in | 19 | 0.06% |
| day in year for periodic occurrence | 19 | 0.06% |
| penalty | 19 | 0.06% |
| depends on software | 19 | 0.06% |
| has boundary | 18 | 0.05% |
| intended public | 18 | 0.05% |
| stepparent | 18 | 0.05% |
| underlies | 18 | 0.05% |
| wears | 18 | 0.05% |
| appears in the form of | 18 | 0.05% |
| influenced by | 18 | 0.05% |
| captain | 17 | 0.05% |
| carries scientific instrument | 17 | 0.05% |
| has contributing factor | 17 | 0.05% |
| proved by | 17 | 0.05% |
| structure replaced by | 17 | 0.05% |
| foods traditionally associated | 17 | 0.05% |
| emulates | 17 | 0.05% |
| parent of this hybrid, breed, or cultivar | 17 | 0.05% |
| vaccine for | 17 | 0.05% |
| approved by | 16 | 0.05% |
| recorded at studio or venue | 16 | 0.05% |
| charge | 16 | 0.05% |
| soundtrack release | 16 | 0.05% |
| next lower rank | 16 | 0.05% |
| season | 16 | 0.05% |
| diplomatic relation | 16 | 0.05% |
| designed to carry | 16 | 0.05% |
| mountains classification | 16 | 0.05% |
| statement describes | 15 | 0.04% |
| amended by | 15 | 0.04% |
| family name identical to this given name | 15 | 0.04% |
| in opposition to | 15 | 0.04% |
| instruction set | 15 | 0.04% |
| given name version for other gender | 15 | 0.04% |
| has active ingredient | 15 | 0.04% |
| connector | 15 | 0.04% |
| twinned administrative body | 15 | 0.04% |
| appointed by | 14 | 0.04% |
| executive producer | 14 | 0.04% |
| has natural reservoir | 14 | 0.04% |
| structure replaces | 14 | 0.04% |
| has grammatical case | 14 | 0.04% |
| regulated by | 14 | 0.04% |
| gens | 14 | 0.04% |
| offers view on | 14 | 0.04% |
| type foundry | 13 | 0.04% |
| prohibits | 13 | 0.04% |
| office contested | 13 | 0.04% |
| Christian liturgical rite | 13 | 0.04% |
| destroyed | 13 | 0.04% |
| has list | 13 | 0.04% |
| motto | 13 | 0.04% |
| species kept | 13 | 0.04% |
| theme music | 13 | 0.04% |
| medical examination | 13 | 0.04% |
| performed at | 13 | 0.04% |
| points classification | 13 | 0.04% |
| film editor | 12 | 0.03% |
| coat of arms | 12 | 0.03% |
| polymer of | 12 | 0.03% |
| expansion of | 12 | 0.03% |
| antiparticle | 12 | 0.03% |
| has fruit type | 12 | 0.03% |
| innervated by | 12 | 0.03% |
| academic degree | 12 | 0.03% |
| solved by | 12 | 0.03% |
| next crossing upstream | 11 | 0.03% |
| update method | 11 | 0.03% |
| copyright representative | 11 | 0.03% |
| parliamentary group | 11 | 0.03% |
| source of material | 11 | 0.03% |
| iconographic symbol | 11 | 0.03% |
| exclave of | 11 | 0.03% |
| set during recurring event | 11 | 0.03% |
| honorific suffix | 11 | 0.03% |
| translator | 11 | 0.03% |
| points/goal scored by | 11 | 0.03% |
| delta of | 11 | 0.03% |
| main food source | 11 | 0.03% |
| expressed in | 11 | 0.03% |
| mod of | 10 | 0.03% |
| host | 10 | 0.03% |
| product certification | 10 | 0.03% |
| narrator | 10 | 0.03% |
| named by | 10 | 0.03% |
| charted in | 10 | 0.03% |
| feast day | 10 | 0.03% |
| parallel street | 10 | 0.03% |
| prerequisite | 10 | 0.03% |
| parent peak | 10 | 0.03% |
| diaspora | 10 | 0.03% |
| trophy awarded | 9 | 0.03% |
| shape | 9 | 0.03% |
| size designation | 9 | 0.03% |
| tonality | 9 | 0.03% |
| madhhab | 9 | 0.03% |
| typically sells | 9 | 0.03% |
| geography of topic | 9 | 0.03% |
| set in environment | 9 | 0.03% |
| people or cargo transported | 9 | 0.03% |
| dialect of computer language | 8 | 0.02% |
| via | 8 | 0.02% |
| adjacent building | 8 | 0.02% |
| significant drug interaction | 8 | 0.02% |
| computes solution to | 8 | 0.02% |
| temporal range start | 8 | 0.02% |
| colorist | 8 | 0.02% |
| main building contractor | 8 | 0.02% |
| innervates | 8 | 0.02% |
| is a list of | 8 | 0.02% |
| cover art by | 8 | 0.02% |
| military casualty classification | 7 | 0.02% |
| top-level Internet domain | 7 | 0.02% |
| type locality (geology) | 7 | 0.02% |
| worn by | 7 | 0.02% |
| addressee | 7 | 0.02% |
| stereoisomer of | 7 | 0.02% |
| theorised by | 7 | 0.02% |
| officially opened by | 7 | 0.02% |
| referee | 7 | 0.02% |
| crystal system | 7 | 0.02% |
| exhibition history | 7 | 0.02% |
| supported sports team | 7 | 0.02% |
| chassis | 7 | 0.02% |
| has seal, badge, or sigil | 7 | 0.02% |
| rector | 6 | 0.02% |
| copyright status | 6 | 0.02% |
| reward program | 6 | 0.02% |
| narrative role | 6 | 0.02% |
| train depot | 6 | 0.02% |
| arterial supply | 6 | 0.02% |
| has phenotype | 6 | 0.02% |
| lifestyle | 6 | 0.02% |
| production designer | 6 | 0.02% |
| programmer | 6 | 0.02% |
| official symbol | 6 | 0.02% |
| academic thesis | 6 | 0.02% |
| taxon synonym | 6 | 0.02% |
| cites work | 6 | 0.02% |
| does not have part | 6 | 0.02% |
| demographics of topic | 6 | 0.02% |
| peak bagging classification | 6 | 0.02% |
| political coalition | 6 | 0.02% |
| geomorphological unit | 6 | 0.02% |
| general manager | 6 | 0.02% |
| OSI Model layer location | 6 | 0.02% |
| adapted by | 5 | 0.01% |
| issued by | 5 | 0.01% |
| official podcast | 5 | 0.01% |
| antagonist muscle | 5 | 0.01% |
| character designer | 5 | 0.01% |
| discovery method | 5 | 0.01% |
| donated by | 5 | 0.01% |
| venous drainage | 5 | 0.01% |
| aspect ratio (W:H) | 5 | 0.01% |
| repeals | 5 | 0.01% |
| supports programming language | 5 | 0.01% |
| produced sound | 5 | 0.01% |
| sports league level | 5 | 0.01% |
| microarchitecture | 5 | 0.01% |
| choral conductor | 5 | 0.01% |
| end cause | 5 | 0.01% |
| exemplar of | 5 | 0.01% |
| launch contractor | 5 | 0.01% |
| has facet polytope | 5 | 0.01% |
| valid in period | 5 | 0.01% |
| disease transmission process | 5 | 0.01% |
| month of the year | 4 | 0.01% |
| calculated from | 4 | 0.01% |
| complies with | 4 | 0.01% |
| order of battle | 4 | 0.01% |
| conjugate acid | 4 | 0.01% |
| substrate of | 4 | 0.01% |
| uses capitalization for | 4 | 0.01% |
| nominated by | 4 | 0.01% |
| by-product | 4 | 0.01% |
| develops from | 4 | 0.01% |
| style of karate | 4 | 0.01% |
| coolant | 4 | 0.01% |
| ratified by | 4 | 0.01% |
| decays to | 4 | 0.01% |
| season of club or team | 4 | 0.01% |
| military unit | 4 | 0.01% |
| religious occupation | 4 | 0.01% |
| funder | 4 | 0.01% |
| has melody | 4 | 0.01% |
| elector | 4 | 0.01% |
| asteroid spectral type | 4 | 0.01% |
| given name | 4 | 0.01% |
| has census | 4 | 0.01% |
| notation | 4 | 0.01% |
| animator | 4 | 0.01% |
| emergency phone number | 3 | 0.01% |
| fandom | 3 | 0.01% |
| port of registry | 3 | 0.01% |
| comorbidity | 3 | 0.01% |
| day of week | 3 | 0.01% |
| intangible cultural heritage status | 3 | 0.01% |
| statement is subject of | 3 | 0.01% |
| visitor centre | 3 | 0.01% |
| prime factor | 3 | 0.01% |
| record held | 3 | 0.01% |
| solid solution series with | 3 | 0.01% |
| typing discipline | 3 | 0.01% |
| does not have quality | 3 | 0.01% |
| doctoral advisor | 3 | 0.01% |
| plaintiff | 3 | 0.01% |
| laws applied | 3 | 0.01% |
| symbolizes | 3 | 0.01% |
| investor | 3 | 0.01% |
| majority opinion by | 3 | 0.01% |
| next crossing downstream | 3 | 0.01% |
| narrative motif | 3 | 0.01% |
| lowest point | 3 | 0.01% |
| binding of software library | 3 | 0.01% |
| relegated | 3 | 0.01% |
| microstructure | 3 | 0.01% |
| costume designer | 3 | 0.01% |
| conjugate base | 3 | 0.01% |
| modified version of | 3 | 0.01% |
| postsynaptic connection | 3 | 0.01% |
| curator | 3 | 0.01% |
| has listed ingredient | 3 | 0.01% |
| research site | 3 | 0.01% |
| together with | 3 | 0.01% |
| suggests the existence of | 3 | 0.01% |
| collection creator | 3 | 0.01% |
| disjoint union of | 3 | 0.01% |
| repealed by | 3 | 0.01% |
| era name | 3 | 0.01% |
| official color | 3 | 0.01% |
| plot expanded in | 3 | 0.01% |
| protocol | 3 | 0.01% |
| is metaclass for | 3 | 0.01% |
| by-product of | 2 | 0.01% |
| combination classification | 2 | 0.01% |
| display technology | 2 | 0.01% |
| open data portal | 2 | 0.01% |
| permanent resident of | 2 | 0.01% |
| reply to | 2 | 0.01% |
| edibility | 2 | 0.01% |
| Fach | 2 | 0.01% |
| guidance system | 2 | 0.01% |
| temporal range end | 2 | 0.01% |
| website account on | 2 | 0.01% |
| landscape architect | 2 | 0.01% |
| facial hair | 2 | 0.01% |
| decreased expression in | 2 | 0.01% |
| art director | 2 | 0.01% |
| derived from organism type | 2 | 0.01% |
| distributary | 2 | 0.01% |
| item disputed by | 2 | 0.01% |
| introduced feature | 2 | 0.01% |
| biological variant of | 2 | 0.01% |
| category of associated people | 2 | 0.01% |
| encoded by | 2 | 0.01% |
| undercarriage | 2 | 0.01% |
| metrically compatible typeface | 2 | 0.01% |
| route of administration | 2 | 0.01% |
| relative to | 2 | 0.01% |
| avionics | 2 | 0.01% |
| production statistics | 2 | 0.01% |
| electoral district | 2 | 0.01% |
| first aid measures | 2 | 0.01% |
| godparent | 2 | 0.01% |
| output method | 2 | 0.01% |
| invasive to | 2 | 0.01% |
| has facility | 2 | 0.01% |
| side effect | 2 | 0.01% |
| doctoral student | 2 | 0.01% |
| volcano observatory | 2 | 0.01% |
| habitat | 2 | 0.01% |
| chief operating officer | 2 | 0.01% |
| regulates (molecular biology) | 2 | 0.01% |
| partnership with | 2 | 0.01% |
| promoted | 2 | 0.01% |
| religious movement | 2 | 0.01% |
| base | 1 | 0.00% |
| original film format | 1 | 0.00% |
| epithet | 1 | 0.00% |
| GPU | 1 | 0.00% |
| has tense | 1 | 0.00% |
| catalog | 1 | 0.00% |
| external auditor | 1 | 0.00% |
| muscle origin | 1 | 0.00% |
| overrules | 1 | 0.00% |
| radix | 1 | 0.00% |
| socket supported | 1 | 0.00% |
| used metre | 1 | 0.00% |
| fictional analog of | 1 | 0.00% |
| taxon found at location | 1 | 0.00% |
| recording location | 1 | 0.00% |
| tournament format | 1 | 0.00% |
| start grade | 1 | 0.00% |
| activating neurotransmitter | 1 | 0.00% |
| climbing route | 1 | 0.00% |
| has member | 1 | 0.00% |
| ecoregion (WWF) | 1 | 0.00% |
| including | 1 | 0.00% |
| pendant of | 1 | 0.00% |
| real estate developer | 1 | 0.00% |
| storyboard artist | 1 | 0.00% |
| oath made by | 1 | 0.00% |
| dissertation submitted to | 1 | 0.00% |
| established from medical condition | 1 | 0.00% |
| demonym of | 1 | 0.00% |
| nominee | 1 | 0.00% |
| card network | 1 | 0.00% |
| currency symbol description | 1 | 0.00% |
| mode of inheritance | 1 | 0.00% |
| list of works | 1 | 0.00% |
| bibliography | 1 | 0.00% |
| index case of | 1 | 0.00% |
| gait | 1 | 0.00% |
| muscle insertion | 1 | 0.00% |
| reissue of | 1 | 0.00% |
| has pattern | 1 | 0.00% |
| founder | 1 | 0.00% |
| fictional character | 1 | 0.00% |
| film crew member | 1 | 0.00% |
| entry receptor | 1 | 0.00% |
| supplement to | 1 | 0.00% |
| source of income | 1 | 0.00% |
| decay mode | 1 | 0.00% |
| may prevent | 1 | 0.00% |
| samples from work | 1 | 0.00% |
| professorship | 1 | 0.00% |
| follower of | 1 | 0.00% |
| criterion used | 1 | 0.00% |
| exonerated of | 1 | 0.00% |
| Vox-ATypI classification | 1 | 0.00% |
| has phoneme | 1 | 0.00% |
| leader | 1 | 0.00% |
| record or record progression | 1 | 0.00% |
| place of devotion | 1 | 0.00% |
| companion of | 1 | 0.00% |
| Roman agnomen | 1 | 0.00% |
| academic appointment | 1 | 0.00% |
| fossil found in this unit | 1 | 0.00% |
| indexed in bibliographic review | 1 | 0.00% |
| issuing agent of work | 1 | 0.00% |
| young rider classification | 1 | 0.00% |
| possible medical findings | 1 | 0.00% |
| presynaptic connection | 1 | 0.00% |
| has role in modeling | 1 | 0.00% |
| means of locomotion | 1 | 0.00% |
| generalization of | 1 | 0.00% |
| located on linear feature | 1 | 0.00% |
| reward | 1 | 0.00% |
| inhibitor of | 1 | 0.00% |
| has superpartner | 1 | 0.00% |
| facilitates flow of | 1 | 0.00% |
| associated hazard | 1 | 0.00% |
| grammatical option indicates | 1 | 0.00% |
| codomain | 1 | 0.00% |
| mushroom ecological type | 1 | 0.00% |
| parliamentary term | 1 | 0.00% |
| election called by | 1 | 0.00% |
| kit supplier | 1 | 0.00% |
| province | 1 | 0.00% |
| blocker of | 1 | 0.00% |

#### val

| label | count | share |
| --- | --- | --- |
| characters | 26 | 0.61% |
| architect | 25 | 0.59% |
| notable work | 24 | 0.56% |
| participant | 23 | 0.54% |
| conflict | 23 | 0.54% |
| grants | 22 | 0.52% |
| competition class | 22 | 0.52% |
| terminus location | 22 | 0.52% |
| officeholder | 22 | 0.52% |
| platform | 22 | 0.52% |
| military branch | 22 | 0.52% |
| author | 21 | 0.49% |
| language used | 21 | 0.49% |
| child | 21 | 0.49% |
| discoverer or inventor | 21 | 0.49% |
| participating team | 21 | 0.49% |
| place of publication | 20 | 0.47% |
| director | 20 | 0.47% |
| cast member | 20 | 0.47% |
| member of political party | 20 | 0.47% |
| performer | 20 | 0.47% |
| father | 20 | 0.47% |
| partner in business or sport | 20 | 0.47% |
| located on astronomical body | 19 | 0.45% |
| heritage designation | 19 | 0.45% |
| founded by | 19 | 0.45% |
| located in or next to body of water | 19 | 0.45% |
| located in the administrative territorial entity | 19 | 0.45% |
| sport | 19 | 0.45% |
| secretary general | 18 | 0.42% |
| movement | 18 | 0.42% |
| historic county | 18 | 0.42% |
| league | 18 | 0.42% |
| noble title | 18 | 0.42% |
| present in work | 18 | 0.42% |
| position held | 18 | 0.42% |
| creator | 18 | 0.42% |
| connecting line | 18 | 0.42% |
| basin country | 18 | 0.42% |
| constellation | 17 | 0.40% |
| manufacturer | 17 | 0.40% |
| nominated for | 17 | 0.40% |
| spouse | 17 | 0.40% |
| contains the administrative territorial entity | 17 | 0.40% |
| occupation | 17 | 0.40% |
| mouth of the watercourse | 17 | 0.40% |
| has use | 17 | 0.40% |
| publisher | 17 | 0.40% |
| location of discovery | 16 | 0.37% |
| relative | 16 | 0.37% |
| is the study of | 16 | 0.37% |
| original broadcaster | 16 | 0.37% |
| member of | 16 | 0.37% |
| opposite of | 16 | 0.37% |
| owned by | 16 | 0.37% |
| conferred by | 16 | 0.37% |
| located in/on physical feature | 16 | 0.37% |
| product or material produced | 16 | 0.37% |
| located in the present-day administrative territorial entity | 16 | 0.37% |
| licensed to broadcast to | 16 | 0.37% |
| medical condition treated | 15 | 0.35% |
| astronaut mission | 15 | 0.35% |
| sports discipline competed in | 15 | 0.35% |
| military rank | 15 | 0.35% |
| league system | 15 | 0.35% |
| online service | 15 | 0.35% |
| has works in the collection | 15 | 0.35% |
| place served by transport hub | 15 | 0.35% |
| occupant | 15 | 0.35% |
| country of citizenship | 15 | 0.35% |
| award received | 15 | 0.35% |
| published in | 15 | 0.35% |
| sibling | 15 | 0.35% |
| mountain range | 15 | 0.35% |
| allegiance | 15 | 0.35% |
| named after | 15 | 0.35% |
| parent club | 14 | 0.33% |
| fictional universe described in | 14 | 0.33% |
| place of detention | 14 | 0.33% |
| IUCN conservation status | 14 | 0.33% |
| located in time zone | 14 | 0.33% |
| has part(s) | 14 | 0.33% |
| location | 14 | 0.33% |
| instance of | 14 | 0.33% |
| field of work | 14 | 0.33% |
| participant in | 14 | 0.33% |
| record label | 14 | 0.33% |
| package management system | 14 | 0.33% |
| court | 14 | 0.33% |
| country | 14 | 0.33% |
| developer | 14 | 0.33% |
| parent taxon | 14 | 0.33% |
| continent | 14 | 0.33% |
| wheel arrangement | 14 | 0.33% |
| member of the crew of | 13 | 0.30% |
| incarnation of | 13 | 0.30% |
| pole position | 13 | 0.30% |
| surface played on | 13 | 0.30% |
| this zoological name is coordinate with | 13 | 0.30% |
| medical condition | 13 | 0.30% |
| time period | 13 | 0.30% |
| position played on team / speciality | 13 | 0.30% |
| form of creative work | 13 | 0.30% |
| part of | 13 | 0.30% |
| location of formation | 13 | 0.30% |
| item operated | 13 | 0.30% |
| employer | 13 | 0.30% |
| replaced by | 13 | 0.30% |
| operator | 13 | 0.30% |
| genre | 13 | 0.30% |
| voice actor | 13 | 0.30% |
| chief executive officer | 12 | 0.28% |
| location of first performance | 12 | 0.28% |
| endemic to | 12 | 0.28% |
| has edition or translation | 12 | 0.28% |
| natural product of taxon | 12 | 0.28% |
| practiced by | 12 | 0.28% |
| original language of film or TV show | 12 | 0.28% |
| presenter | 12 | 0.28% |
| connects with | 12 | 0.28% |
| terminus | 12 | 0.28% |
| mother | 12 | 0.28% |
| standards body | 12 | 0.28% |
| after a work by | 12 | 0.28% |
| genetic association | 12 | 0.28% |
| residence | 12 | 0.28% |
| family | 12 | 0.28% |
| copyright license | 11 | 0.26% |
| destination point | 11 | 0.26% |
| convicted of | 11 | 0.26% |
| represented by | 11 | 0.26% |
| ammunition | 11 | 0.26% |
| physically interacts with | 11 | 0.26% |
| drug or therapy used for treatment | 11 | 0.26% |
| from narrative universe | 11 | 0.26% |
| this taxon is source of | 11 | 0.26% |
| architectural style | 11 | 0.26% |
| connecting service | 11 | 0.26% |
| stock market index | 11 | 0.26% |
| anatomical location | 11 | 0.26% |
| unmarried partner | 11 | 0.26% |
| winner | 11 | 0.26% |
| indigenous to | 11 | 0.26% |
| programming paradigm | 11 | 0.26% |
| significant event | 11 | 0.26% |
| has subsidiary | 11 | 0.26% |
| place of birth | 11 | 0.26% |
| composer | 11 | 0.26% |
| software engine | 10 | 0.23% |
| crew member(s) | 10 | 0.23% |
| history of topic | 10 | 0.23% |
| basic form of government | 10 | 0.23% |
| biological process | 10 | 0.23% |
| lake on watercourse | 10 | 0.23% |
| represents | 10 | 0.23% |
| authority | 10 | 0.23% |
| brand | 10 | 0.23% |
| language of work or name | 10 | 0.23% |
| radio format | 10 | 0.23% |
| contains | 10 | 0.23% |
| producer | 10 | 0.23% |
| candidate | 10 | 0.23% |
| located on street | 10 | 0.23% |
| input method | 10 | 0.23% |
| stock exchange | 10 | 0.23% |
| ethnic group | 10 | 0.23% |
| head of state | 10 | 0.23% |
| significant person | 10 | 0.23% |
| chairperson | 10 | 0.23% |
| instrument | 10 | 0.23% |
| said to be the same as | 10 | 0.23% |
| parent organization | 10 | 0.23% |
| religion or worldview | 10 | 0.23% |
| tributary | 10 | 0.23% |
| worshipped by | 9 | 0.21% |
| operating area | 9 | 0.21% |
| recommended unit of measurement | 9 | 0.21% |
| has written for | 9 | 0.21% |
| model item | 9 | 0.21% |
| airline alliance | 9 | 0.21% |
| broadcast by | 9 | 0.21% |
| appears in the form of | 9 | 0.21% |
| collection | 9 | 0.21% |
| commemorates | 9 | 0.21% |
| vehicle | 9 | 0.21% |
| school district | 9 | 0.21% |
| start point | 9 | 0.21% |
| made from material | 9 | 0.21% |
| country for sport | 9 | 0.21% |
| space launch vehicle | 9 | 0.21% |
| illustrator | 9 | 0.21% |
| legislated by | 9 | 0.21% |
| vessel class | 9 | 0.21% |
| powered by | 9 | 0.21% |
| edition or translation of | 9 | 0.21% |
| family name | 9 | 0.21% |
| dam | 9 | 0.21% |
| headquarters location | 9 | 0.21% |
| applies to jurisdiction | 9 | 0.21% |
| crosses | 9 | 0.21% |
| home venue | 9 | 0.21% |
| territory overlaps | 9 | 0.21% |
| shares border with | 9 | 0.21% |
| member of sports team | 9 | 0.21% |
| distributed by | 9 | 0.21% |
| head of government | 9 | 0.21% |
| highest point | 9 | 0.21% |
| director of photography | 9 | 0.21% |
| work location | 9 | 0.21% |
| anthem | 9 | 0.21% |
| part of the series | 9 | 0.21% |
| country of origin | 9 | 0.21% |
| main subject | 9 | 0.21% |
| coach of sports team | 9 | 0.21% |
| voice type | 8 | 0.19% |
| victim | 8 | 0.19% |
| historical region | 8 | 0.19% |
| writing language | 8 | 0.19% |
| qualifies for event | 8 | 0.19% |
| tracklist | 8 | 0.19% |
| culture | 8 | 0.19% |
| highest judicial authority | 8 | 0.19% |
| official residence | 8 | 0.19% |
| substitute/deputy/replacement of office/officeholder | 8 | 0.19% |
| filming location | 8 | 0.19% |
| editor | 8 | 0.19% |
| inflows | 8 | 0.19% |
| merged into | 8 | 0.19% |
| cathedral | 8 | 0.19% |
| diocese | 8 | 0.19% |
| reservoir created | 8 | 0.19% |
| next higher rank | 8 | 0.19% |
| track gauge | 8 | 0.19% |
| based on | 8 | 0.19% |
| lake outflow | 8 | 0.19% |
| structural engineer | 8 | 0.19% |
| has effect | 8 | 0.19% |
| has part(s) of the class | 8 | 0.19% |
| cause of destruction | 8 | 0.19% |
| site of astronomical discovery | 8 | 0.19% |
| educated at | 8 | 0.19% |
| executive body | 7 | 0.16% |
| organization directed by the office or position | 7 | 0.16% |
| carries | 7 | 0.16% |
| central bank | 7 | 0.16% |
| commanded by | 7 | 0.16% |
| mascot | 7 | 0.16% |
| members have occupation | 7 | 0.16% |
| contains settlement | 7 | 0.16% |
| dedicated to | 7 | 0.16% |
| main regulatory text | 7 | 0.16% |
| programming language | 7 | 0.16% |
| target | 7 | 0.16% |
| transport network | 7 | 0.16% |
| significant place | 7 | 0.16% |
| measured physical quantity | 7 | 0.16% |
| affiliation | 7 | 0.16% |
| taxon rank | 7 | 0.16% |
| found in taxon | 7 | 0.16% |
| dialect of | 7 | 0.16% |
| board member | 7 | 0.16% |
| followed by | 7 | 0.16% |
| coextensive with | 7 | 0.16% |
| copyright holder | 7 | 0.16% |
| interested in | 7 | 0.16% |
| takes place in fictional universe | 7 | 0.16% |
| described by source | 7 | 0.16% |
| chromosome | 7 | 0.16% |
| owner of | 7 | 0.16% |
| child astronomical body | 7 | 0.16% |
| home world | 6 | 0.14% |
| native language | 6 | 0.14% |
| musical conductor | 6 | 0.14% |
| killed by | 6 | 0.14% |
| co-driver | 6 | 0.14% |
| has immediate cause | 6 | 0.14% |
| cell component | 6 | 0.14% |
| encodes | 6 | 0.14% |
| foundational text | 6 | 0.14% |
| innervates | 6 | 0.14% |
| location of creation | 6 | 0.14% |
| business division | 6 | 0.14% |
| currency | 6 | 0.14% |
| media franchise | 6 | 0.14% |
| location of landing | 6 | 0.14% |
| operating system | 6 | 0.14% |
| defendant | 6 | 0.14% |
| has cause | 6 | 0.14% |
| uses | 6 | 0.14% |
| type of variable star | 6 | 0.14% |
| located in the statistical territorial entity | 6 | 0.14% |
| GUI toolkit or framework | 6 | 0.14% |
| perpetrator | 6 | 0.14% |
| student | 6 | 0.14% |
| languages spoken, written or signed | 6 | 0.14% |
| lyrics by | 6 | 0.14% |
| armament | 6 | 0.14% |
| patron saint | 6 | 0.14% |
| partially coincident with | 6 | 0.14% |
| production company | 6 | 0.14% |
| candidacy in election | 6 | 0.14% |
| offers view on | 6 | 0.14% |
| different from | 6 | 0.14% |
| expressed in | 5 | 0.12% |
| season | 5 | 0.12% |
| office held by head of the organization | 5 | 0.12% |
| mountains classification | 5 | 0.12% |
| official language | 5 | 0.12% |
| day in year for periodic occurrence | 5 | 0.12% |
| objective of project or action | 5 | 0.12% |
| fabrication method | 5 | 0.12% |
| honorific prefix | 5 | 0.12% |
| sponsor | 5 | 0.12% |
| top-level Internet domain | 5 | 0.12% |
| domain of saint or deity | 5 | 0.12% |
| taxonomic type | 5 | 0.12% |
| possible treatment | 5 | 0.12% |
| youth wing | 5 | 0.12% |
| designed by | 5 | 0.12% |
| immediate cause of | 5 | 0.12% |
| consecrator | 5 | 0.12% |
| drainage basin | 5 | 0.12% |
| primary destinations | 5 | 0.12% |
| subclass of | 5 | 0.12% |
| replaces | 5 | 0.12% |
| manner of death | 5 | 0.12% |
| vaccine for | 5 | 0.12% |
| antiparticle | 5 | 0.12% |
| facet of | 5 | 0.12% |
| separated from | 5 | 0.12% |
| anatomical branch of | 5 | 0.12% |
| sexually homologous with | 5 | 0.12% |
| canonization status | 5 | 0.12% |
| general classification of race participants | 5 | 0.12% |
| narrative location | 5 | 0.12% |
| adjacent station | 5 | 0.12% |
| inspired by | 5 | 0.12% |
| implementation of | 4 | 0.09% |
| appointed by | 4 | 0.09% |
| has active ingredient | 4 | 0.09% |
| next crossing upstream | 4 | 0.09% |
| title of chess person | 4 | 0.09% |
| flag | 4 | 0.09% |
| health specialty | 4 | 0.09% |
| greater than | 4 | 0.09% |
| linguistic typology | 4 | 0.09% |
| political ideology | 4 | 0.09% |
| office held by head of government | 4 | 0.09% |
| depicts | 4 | 0.09% |
| mount | 4 | 0.09% |
| subdivision of this unit | 4 | 0.09% |
| newspaper format | 4 | 0.09% |
| choreographer | 4 | 0.09% |
| Gram staining | 4 | 0.09% |
| contributed to creative work | 4 | 0.09% |
| parent astronomical body | 4 | 0.09% |
| maintained by | 4 | 0.09% |
| series spin-off | 4 | 0.09% |
| designated as terrorist by | 4 | 0.09% |
| head coach | 4 | 0.09% |
| performed at | 4 | 0.09% |
| shape | 4 | 0.09% |
| director / manager | 4 | 0.09% |
| proved by | 4 | 0.09% |
| advertises | 4 | 0.09% |
| has boundary | 4 | 0.09% |
| public holiday | 4 | 0.09% |
| discography | 4 | 0.09% |
| iconographic symbol | 4 | 0.09% |
| measurement scale | 4 | 0.09% |
| studied by | 4 | 0.09% |
| industry | 4 | 0.09% |
| structure replaced by | 4 | 0.09% |
| cause of death | 4 | 0.09% |
| used by | 4 | 0.09% |
| director of archaeological fieldwork | 4 | 0.09% |
| house publication | 4 | 0.09% |
| librettist | 3 | 0.07% |
| minor planet group | 3 | 0.07% |
| computes solution to | 3 | 0.07% |
| office contested | 3 | 0.07% |
| dual to | 3 | 0.07% |
| writable file format | 3 | 0.07% |
| family name identical to this given name | 3 | 0.07% |
| wing configuration | 3 | 0.07% |
| legal form | 3 | 0.07% |
| executive producer | 3 | 0.07% |
| located in protected area | 3 | 0.07% |
| recorded at studio or venue | 3 | 0.07% |
| distribution format | 3 | 0.07% |
| typically sells | 3 | 0.07% |
| measures | 3 | 0.07% |
| adjacent building | 3 | 0.07% |
| character type | 3 | 0.07% |
| place of burial | 3 | 0.07% |
| emulates | 3 | 0.07% |
| honorific suffix | 3 | 0.07% |
| sidekick of | 3 | 0.07% |
| sexual orientation | 3 | 0.07% |
| afflicts | 3 | 0.07% |
| language regulatory body | 3 | 0.07% |
| victory | 3 | 0.07% |
| feast day | 3 | 0.07% |
| postsynaptic connection | 3 | 0.07% |
| produced by | 3 | 0.07% |
| writing system | 3 | 0.07% |
| wears | 3 | 0.07% |
| charge | 3 | 0.07% |
| editor-in-chief | 3 | 0.07% |
| amended by | 3 | 0.07% |
| airline hub | 3 | 0.07% |
| underlies | 3 | 0.07% |
| derivative work | 3 | 0.07% |
| lifestyle | 3 | 0.07% |
| designed to carry | 3 | 0.07% |
| league level above | 3 | 0.07% |
| signatory | 3 | 0.07% |
| professorship | 3 | 0.07% |
| risk factor | 3 | 0.07% |
| organizer | 3 | 0.07% |
| music created for | 3 | 0.07% |
| student of | 3 | 0.07% |
| subject has role | 3 | 0.07% |
| legislative body | 3 | 0.07% |
| territory claimed by | 3 | 0.07% |
| commander of (DEPRECATED) | 3 | 0.07% |
| has phenotype | 3 | 0.07% |
| has quality | 3 | 0.07% |
| religious order | 3 | 0.07% |
| first appearance | 3 | 0.07% |
| points/goal scored by | 3 | 0.07% |
| game mechanics | 3 | 0.07% |
| follows | 3 | 0.07% |
| source of energy | 3 | 0.07% |
| intended public | 3 | 0.07% |
| suggests the existence of | 3 | 0.07% |
| business model | 2 | 0.05% |
| climbing route | 2 | 0.05% |
| CPU | 2 | 0.05% |
| category of associated people | 2 | 0.05% |
| military casualty classification | 2 | 0.05% |
| named by | 2 | 0.05% |
| parent of this hybrid, breed, or cultivar | 2 | 0.05% |
| set in period | 2 | 0.05% |
| type of orbit | 2 | 0.05% |
| valid in period | 2 | 0.05% |
| vehicle normally used | 2 | 0.05% |
| monomer of | 2 | 0.05% |
| symptoms and signs | 2 | 0.05% |
| commissioned by | 2 | 0.05% |
| film editor | 2 | 0.05% |
| academic thesis | 2 | 0.05% |
| fandom | 2 | 0.05% |
| godparent | 2 | 0.05% |
| country of registry | 2 | 0.05% |
| cuisine | 2 | 0.05% |
| dialect of computer language | 2 | 0.05% |
| output method | 2 | 0.05% |
| solved by | 2 | 0.05% |
| in opposition to | 2 | 0.05% |
| depicted by | 2 | 0.05% |
| capital of | 2 | 0.05% |
| research site | 2 | 0.05% |
| ancestral home | 2 | 0.05% |
| economy of topic | 2 | 0.05% |
| geomorphological unit | 2 | 0.05% |
| party chief representative | 2 | 0.05% |
| theme music | 2 | 0.05% |
| peak bagging classification | 2 | 0.05% |
| less than | 2 | 0.05% |
| fault | 2 | 0.05% |
| medical examination | 2 | 0.05% |
| narrative motif | 2 | 0.05% |
| screenwriter | 2 | 0.05% |
| enclave within | 2 | 0.05% |
| curator | 2 | 0.05% |
| game mode | 2 | 0.05% |
| diplomatic relation | 2 | 0.05% |
| parliamentary group | 2 | 0.05% |
| field of this occupation | 2 | 0.05% |
| has anatomical branch | 2 | 0.05% |
| origin of the watercourse | 2 | 0.05% |
| officially opened by | 2 | 0.05% |
| partnership with | 2 | 0.05% |
| social classification | 2 | 0.05% |
| political alignment | 2 | 0.05% |
| capital | 2 | 0.05% |
| influenced by | 2 | 0.05% |
| drafted by | 2 | 0.05% |
| office held by head of state | 2 | 0.05% |
| contributing factor of | 2 | 0.05% |
| trophy awarded | 2 | 0.05% |
| has grammatical case | 2 | 0.05% |
| instrumentation | 2 | 0.05% |
| collection creator | 2 | 0.05% |
| readable file format | 2 | 0.05% |
| sports season of league or competition | 2 | 0.05% |
| enemy of | 2 | 0.05% |
| academic degree | 2 | 0.05% |
| place of death | 2 | 0.05% |
| destroyed | 2 | 0.05% |
| Christian liturgical rite | 2 | 0.05% |
| product certification | 2 | 0.05% |
| nominee | 1 | 0.02% |
| character designer | 1 | 0.02% |
| depends on software | 1 | 0.02% |
| foods traditionally associated | 1 | 0.02% |
| reward program | 1 | 0.02% |
| structure replaces | 1 | 0.02% |
| central bank/issuer | 1 | 0.02% |
| companion of | 1 | 0.02% |
| crystal system | 1 | 0.02% |
| Fach | 1 | 0.02% |
| has contributing factor | 1 | 0.02% |
| notation | 1 | 0.02% |
| scenographer | 1 | 0.02% |
| prime factor | 1 | 0.02% |
| manifestation of | 1 | 0.02% |
| issued by | 1 | 0.02% |
| place of origin (Switzerland) | 1 | 0.02% |
| parent peak | 1 | 0.02% |
| cover art by | 1 | 0.02% |
| artistic director | 1 | 0.02% |
| twinned administrative body | 1 | 0.02% |
| successful candidate | 1 | 0.02% |
| fastest lap | 1 | 0.02% |
| supplement to | 1 | 0.02% |
| temporal range end | 1 | 0.02% |
| season of club or team | 1 | 0.02% |
| carries scientific instrument | 1 | 0.02% |
| possible medical findings | 1 | 0.02% |
| has list | 1 | 0.02% |
| scheduled service destination | 1 | 0.02% |
| Roman agnomen | 1 | 0.02% |
| stepparent | 1 | 0.02% |
| ICTV virus genome composition | 1 | 0.02% |
| OSI Model layer location | 1 | 0.02% |
| inscription mentions | 1 | 0.02% |
| charted in | 1 | 0.02% |
| engine configuration | 1 | 0.02% |
| production designer | 1 | 0.02% |
| therapeutic area | 1 | 0.02% |
| contributor to the creative work or subject | 1 | 0.02% |
| statement describes | 1 | 0.02% |
| is pollinated by | 1 | 0.02% |
| diaspora | 1 | 0.02% |
| supported sports team | 1 | 0.02% |
| has natural reservoir | 1 | 0.02% |
| admissible rule in | 1 | 0.02% |
| gens | 1 | 0.02% |
| innervated by | 1 | 0.02% |
| sports league level | 1 | 0.02% |
| overlies | 1 | 0.02% |
| does not have part | 1 | 0.02% |
| archives at | 1 | 0.02% |
| landscape architect | 1 | 0.02% |
| tournament format | 1 | 0.02% |
| day of week | 1 | 0.02% |
| samples from work | 1 | 0.02% |
| ortholog | 1 | 0.02% |
| motto | 1 | 0.02% |
| has pet | 1 | 0.02% |
| league level below | 1 | 0.02% |
| set in environment | 1 | 0.02% |
| programmer | 1 | 0.02% |
| demographics of topic | 1 | 0.02% |
| statistical leader | 1 | 0.02% |
| narrative role | 1 | 0.02% |
| temporal range start | 1 | 0.02% |
| original film format | 1 | 0.02% |
| exclave of | 1 | 0.02% |
| translator | 1 | 0.02% |
| polymer of | 1 | 0.02% |
| antagonist muscle | 1 | 0.02% |
| codomain | 1 | 0.02% |
| addressee | 1 | 0.02% |
| penalty | 1 | 0.02% |
| interaction | 1 | 0.02% |
| majority opinion by | 1 | 0.02% |
| microarchitecture | 1 | 0.02% |
| given name version for other gender | 1 | 0.02% |
| update method | 1 | 0.02% |
| prohibits | 1 | 0.02% |
| month of the year | 1 | 0.02% |
| exhibition history | 1 | 0.02% |
| approved by | 1 | 0.02% |
| connector | 1 | 0.02% |
| end cause | 1 | 0.02% |
| main building contractor | 1 | 0.02% |
| rector | 1 | 0.02% |
| soundtrack release | 1 | 0.02% |
| costume designer | 1 | 0.02% |
| laws applied | 1 | 0.02% |
| electoral district | 1 | 0.02% |
| has seal, badge, or sigil | 1 | 0.02% |
| official symbol | 1 | 0.02% |
| captain | 1 | 0.02% |
| color | 1 | 0.02% |
| copyright status | 1 | 0.02% |
| disease transmission process | 1 | 0.02% |
| edibility | 1 | 0.02% |
| has facet polytope | 1 | 0.02% |
| is a list of | 1 | 0.02% |
| next crossing downstream | 1 | 0.02% |
| species kept | 1 | 0.02% |
| animal breed | 1 | 0.02% |
| delta of | 1 | 0.02% |
| train depot | 1 | 0.02% |
| base | 1 | 0.02% |
| damaged | 1 | 0.02% |
| microstructure | 1 | 0.02% |
| develops from | 1 | 0.02% |
| list of works | 1 | 0.02% |
| UNESCO language status | 1 | 0.02% |
| home port | 1 | 0.02% |
| interchange station | 1 | 0.02% |
| permanent resident of | 1 | 0.02% |
| narrator | 1 | 0.02% |
| era name | 1 | 0.02% |
| province | 1 | 0.02% |

#### test

| label | count | share |
| --- | --- | --- |
| league | 28 | 0.65% |
| military branch | 27 | 0.63% |
| country | 26 | 0.61% |
| sibling | 25 | 0.58% |
| present in work | 25 | 0.58% |
| spouse | 24 | 0.56% |
| employer | 24 | 0.56% |
| voice type | 23 | 0.54% |
| historic county | 23 | 0.54% |
| position held | 23 | 0.54% |
| heritage designation | 23 | 0.54% |
| occupation | 23 | 0.54% |
| broadcast by | 22 | 0.51% |
| genre | 22 | 0.51% |
| member of | 22 | 0.51% |
| instrument | 22 | 0.51% |
| located in time zone | 21 | 0.49% |
| position played on team / speciality | 21 | 0.49% |
| member of sports team | 21 | 0.49% |
| creator | 21 | 0.49% |
| basin country | 21 | 0.49% |
| chairperson | 21 | 0.49% |
| constellation | 20 | 0.47% |
| online service | 20 | 0.47% |
| connecting line | 20 | 0.47% |
| discoverer or inventor | 20 | 0.47% |
| director | 20 | 0.47% |
| original broadcaster | 20 | 0.47% |
| mountain range | 19 | 0.44% |
| cast member | 19 | 0.44% |
| location | 19 | 0.44% |
| author | 19 | 0.44% |
| child | 19 | 0.44% |
| father | 19 | 0.44% |
| licensed to broadcast to | 19 | 0.44% |
| has works in the collection | 18 | 0.42% |
| fictional universe described in | 18 | 0.42% |
| drug or therapy used for treatment | 18 | 0.42% |
| has edition or translation | 18 | 0.42% |
| located on astronomical body | 18 | 0.42% |
| award received | 18 | 0.42% |
| located in the administrative territorial entity | 18 | 0.42% |
| located in the present-day administrative territorial entity | 18 | 0.42% |
| located in or next to body of water | 18 | 0.42% |
| member of political party | 18 | 0.42% |
| court | 17 | 0.40% |
| occupant | 17 | 0.40% |
| manufacturer | 17 | 0.40% |
| nominated for | 17 | 0.40% |
| characters | 17 | 0.40% |
| performer | 17 | 0.40% |
| mouth of the watercourse | 17 | 0.40% |
| incarnation of | 16 | 0.37% |
| medical condition | 16 | 0.37% |
| crosses | 16 | 0.37% |
| conferred by | 16 | 0.37% |
| officeholder | 16 | 0.37% |
| has subsidiary | 16 | 0.37% |
| home venue | 16 | 0.37% |
| developer | 16 | 0.37% |
| filming location | 15 | 0.35% |
| carries | 15 | 0.35% |
| terminus location | 15 | 0.35% |
| cathedral | 15 | 0.35% |
| architect | 15 | 0.35% |
| religion or worldview | 15 | 0.35% |
| mother | 15 | 0.35% |
| presenter | 15 | 0.35% |
| indigenous to | 15 | 0.35% |
| continent | 15 | 0.35% |
| vehicle | 14 | 0.33% |
| member of the crew of | 14 | 0.33% |
| model item | 14 | 0.33% |
| medical condition treated | 14 | 0.33% |
| languages spoken, written or signed | 14 | 0.33% |
| distributed by | 14 | 0.33% |
| notable work | 14 | 0.33% |
| place of birth | 14 | 0.33% |
| contains the administrative territorial entity | 14 | 0.33% |
| participating team | 14 | 0.33% |
| named after | 14 | 0.33% |
| adjacent station | 14 | 0.33% |
| conflict | 14 | 0.33% |
| participant in | 14 | 0.33% |
| currency | 13 | 0.30% |
| sports discipline competed in | 13 | 0.30% |
| represented by | 13 | 0.30% |
| historical region | 13 | 0.30% |
| crew member(s) | 13 | 0.30% |
| religious order | 13 | 0.30% |
| dam | 13 | 0.30% |
| noble title | 13 | 0.30% |
| commemorates | 13 | 0.30% |
| anatomical location | 13 | 0.30% |
| headquarters location | 13 | 0.30% |
| has part(s) | 13 | 0.30% |
| family | 13 | 0.30% |
| terminus | 13 | 0.30% |
| located in/on physical feature | 13 | 0.30% |
| platform | 13 | 0.30% |
| uses | 13 | 0.30% |
| record label | 13 | 0.30% |
| advertises | 12 | 0.28% |
| grants | 12 | 0.28% |
| location of discovery | 12 | 0.28% |
| reservoir created | 12 | 0.28% |
| head of state | 12 | 0.28% |
| located in the statistical territorial entity | 12 | 0.28% |
| instance of | 12 | 0.28% |
| different from | 12 | 0.28% |
| convicted of | 12 | 0.28% |
| parent organization | 12 | 0.28% |
| made from material | 12 | 0.28% |
| ammunition | 12 | 0.28% |
| writing language | 12 | 0.28% |
| located on street | 12 | 0.28% |
| country of origin | 12 | 0.28% |
| operator | 12 | 0.28% |
| owner of | 12 | 0.28% |
| tributary | 12 | 0.28% |
| applies to jurisdiction | 12 | 0.28% |
| commander of (DEPRECATED) | 12 | 0.28% |
| participant | 12 | 0.28% |
| collection | 11 | 0.26% |
| package management system | 11 | 0.26% |
| standards body | 11 | 0.26% |
| takes place in fictional universe | 11 | 0.26% |
| honorific prefix | 11 | 0.26% |
| first appearance | 11 | 0.26% |
| illustrator | 11 | 0.26% |
| located in protected area | 11 | 0.26% |
| movement | 11 | 0.26% |
| represents | 11 | 0.26% |
| winner | 11 | 0.26% |
| country of citizenship | 11 | 0.26% |
| radio format | 11 | 0.26% |
| founded by | 11 | 0.26% |
| parent taxon | 11 | 0.26% |
| main subject | 11 | 0.26% |
| significant person | 11 | 0.26% |
| owned by | 11 | 0.26% |
| product or material produced | 11 | 0.26% |
| executive body | 11 | 0.26% |
| replaced by | 11 | 0.26% |
| has effect | 11 | 0.26% |
| vessel class | 10 | 0.23% |
| significant place | 10 | 0.23% |
| biological process | 10 | 0.23% |
| secretary general | 10 | 0.23% |
| stock exchange | 10 | 0.23% |
| chief executive officer | 10 | 0.23% |
| chromosome | 10 | 0.23% |
| main regulatory text | 10 | 0.23% |
| place of publication | 10 | 0.23% |
| produced by | 10 | 0.23% |
| has written for | 10 | 0.23% |
| board member | 10 | 0.23% |
| next higher rank | 10 | 0.23% |
| country for sport | 10 | 0.23% |
| after a work by | 10 | 0.23% |
| has cause | 10 | 0.23% |
| choreographer | 10 | 0.23% |
| endemic to | 10 | 0.23% |
| followed by | 10 | 0.23% |
| part of | 10 | 0.23% |
| cause of destruction | 10 | 0.23% |
| basic form of government | 10 | 0.23% |
| place served by transport hub | 10 | 0.23% |
| original language of film or TV show | 10 | 0.23% |
| significant event | 10 | 0.23% |
| has immediate cause | 10 | 0.23% |
| operating area | 10 | 0.23% |
| field of work | 10 | 0.23% |
| composer | 10 | 0.23% |
| language used | 10 | 0.23% |
| copyright license | 9 | 0.21% |
| found in taxon | 9 | 0.21% |
| stock market index | 9 | 0.21% |
| destination point | 9 | 0.21% |
| musical conductor | 9 | 0.21% |
| culture | 9 | 0.21% |
| editor | 9 | 0.21% |
| worshipped by | 9 | 0.21% |
| killed by | 9 | 0.21% |
| industry | 9 | 0.21% |
| tracklist | 9 | 0.21% |
| coach of sports team | 9 | 0.21% |
| subclass of | 9 | 0.21% |
| is the study of | 9 | 0.21% |
| work location | 9 | 0.21% |
| sport | 9 | 0.21% |
| has use | 9 | 0.21% |
| highest judicial authority | 9 | 0.21% |
| taxonomic type | 9 | 0.21% |
| said to be the same as | 9 | 0.21% |
| head of government | 9 | 0.21% |
| maintained by | 9 | 0.21% |
| linguistic typology | 9 | 0.21% |
| ethnic group | 9 | 0.21% |
| residence | 9 | 0.21% |
| commanded by | 9 | 0.21% |
| connects with | 9 | 0.21% |
| publisher | 9 | 0.21% |
| based on | 9 | 0.21% |
| brand | 8 | 0.19% |
| librettist | 8 | 0.19% |
| space launch vehicle | 8 | 0.19% |
| diocese | 8 | 0.19% |
| parent astronomical body | 8 | 0.19% |
| authority | 8 | 0.19% |
| dialect of | 8 | 0.19% |
| merged into | 8 | 0.19% |
| natural product of taxon | 8 | 0.19% |
| substitute/deputy/replacement of office/officeholder | 8 | 0.19% |
| transport network | 8 | 0.19% |
| architectural style | 8 | 0.19% |
| inflows | 8 | 0.19% |
| genetic association | 8 | 0.19% |
| production company | 8 | 0.19% |
| start point | 8 | 0.19% |
| signatory | 8 | 0.19% |
| co-driver | 8 | 0.19% |
| form of creative work | 8 | 0.19% |
| recommended unit of measurement | 8 | 0.19% |
| opposite of | 8 | 0.19% |
| producer | 8 | 0.19% |
| place of death | 8 | 0.19% |
| lyrics by | 8 | 0.19% |
| item operated | 8 | 0.19% |
| domain of saint or deity | 8 | 0.19% |
| screenwriter | 8 | 0.19% |
| student of | 8 | 0.19% |
| league system | 8 | 0.19% |
| surface played on | 8 | 0.19% |
| affiliation | 8 | 0.19% |
| school district | 8 | 0.19% |
| partner in business or sport | 8 | 0.19% |
| edition or translation of | 8 | 0.19% |
| location of first performance | 8 | 0.19% |
| primary destinations | 8 | 0.19% |
| has quality | 8 | 0.19% |
| part of the series | 8 | 0.19% |
| language of work or name | 8 | 0.19% |
| place of burial | 8 | 0.19% |
| IUCN conservation status | 7 | 0.16% |
| legislated by | 7 | 0.16% |
| general classification of race participants | 7 | 0.16% |
| location of creation | 7 | 0.16% |
| penalty | 7 | 0.16% |
| published in | 7 | 0.16% |
| this taxon is source of | 7 | 0.16% |
| competition class | 7 | 0.16% |
| cause of death | 7 | 0.16% |
| programming paradigm | 7 | 0.16% |
| astronaut mission | 7 | 0.16% |
| operating system | 7 | 0.16% |
| CPU | 7 | 0.16% |
| armament | 7 | 0.16% |
| statistical leader | 7 | 0.16% |
| interested in | 7 | 0.16% |
| origin of the watercourse | 7 | 0.16% |
| partially coincident with | 7 | 0.16% |
| follows | 7 | 0.16% |
| designed by | 7 | 0.16% |
| league level below | 7 | 0.16% |
| facet of | 7 | 0.16% |
| Gram staining | 7 | 0.16% |
| qualifies for event | 7 | 0.16% |
| anatomical branch of | 7 | 0.16% |
| replaces | 7 | 0.16% |
| connecting service | 7 | 0.16% |
| pole position | 6 | 0.14% |
| vehicle normally used | 6 | 0.14% |
| canonization status | 6 | 0.14% |
| from narrative universe | 6 | 0.14% |
| house publication | 6 | 0.14% |
| lake on watercourse | 6 | 0.14% |
| software engine | 6 | 0.14% |
| target | 6 | 0.14% |
| programming language | 6 | 0.14% |
| series spin-off | 6 | 0.14% |
| narrative location | 6 | 0.14% |
| anthem | 6 | 0.14% |
| afflicts | 6 | 0.14% |
| distribution format | 6 | 0.14% |
| vaccine for | 6 | 0.14% |
| mount | 6 | 0.14% |
| business division | 6 | 0.14% |
| educated at | 6 | 0.14% |
| taxon rank | 6 | 0.14% |
| associated electoral district | 6 | 0.14% |
| fabrication method | 6 | 0.14% |
| implementation of | 6 | 0.14% |
| sexual orientation | 6 | 0.14% |
| territory overlaps | 6 | 0.14% |
| history of topic | 6 | 0.14% |
| organizer | 6 | 0.14% |
| contributor to the creative work or subject | 6 | 0.14% |
| highest point | 5 | 0.12% |
| native language | 5 | 0.12% |
| military rank | 5 | 0.12% |
| league level above | 5 | 0.12% |
| home world | 5 | 0.12% |
| game mode | 5 | 0.12% |
| performed at | 5 | 0.12% |
| powered by | 5 | 0.12% |
| this zoological name is coordinate with | 5 | 0.12% |
| manifestation of | 5 | 0.12% |
| office held by head of government | 5 | 0.12% |
| public holiday | 5 | 0.12% |
| social classification | 5 | 0.12% |
| title of chess person | 5 | 0.12% |
| type of orbit | 5 | 0.12% |
| type of variable star | 5 | 0.12% |
| wheel arrangement | 5 | 0.12% |
| director / manager | 5 | 0.12% |
| commissioned by | 5 | 0.12% |
| manner of death | 5 | 0.12% |
| track gauge | 5 | 0.12% |
| points/goal scored by | 5 | 0.12% |
| physically interacts with | 5 | 0.12% |
| measured physical quantity | 5 | 0.12% |
| captain | 5 | 0.12% |
| location of landing | 5 | 0.12% |
| inspired by | 5 | 0.12% |
| dedicated to | 5 | 0.12% |
| allegiance | 5 | 0.12% |
| business model | 5 | 0.12% |
| copyright holder | 5 | 0.12% |
| drainage basin | 5 | 0.12% |
| parent club | 5 | 0.12% |
| health specialty | 5 | 0.12% |
| office held by head of the organization | 5 | 0.12% |
| writing system | 5 | 0.12% |
| airline alliance | 5 | 0.12% |
| underlies | 5 | 0.12% |
| candidacy in election | 5 | 0.12% |
| has contributing factor | 5 | 0.12% |
| political alignment | 5 | 0.12% |
| relative | 5 | 0.12% |
| organization directed by the office or position | 4 | 0.09% |
| central bank/issuer | 4 | 0.09% |
| encodes | 4 | 0.09% |
| mod of | 4 | 0.09% |
| lifestyle | 4 | 0.09% |
| objective of project or action | 4 | 0.09% |
| official religion | 4 | 0.09% |
| regulated by | 4 | 0.09% |
| place of detention | 4 | 0.09% |
| dual to | 4 | 0.09% |
| director of photography | 4 | 0.09% |
| enemy of | 4 | 0.09% |
| possible treatment | 4 | 0.09% |
| archives at | 4 | 0.09% |
| minor planet group | 4 | 0.09% |
| discography | 4 | 0.09% |
| has boundary | 4 | 0.09% |
| in opposition to | 4 | 0.09% |
| candidate | 4 | 0.09% |
| capital | 4 | 0.09% |
| contributed to creative work | 4 | 0.09% |
| described by source | 4 | 0.09% |
| head coach | 4 | 0.09% |
| media franchise | 4 | 0.09% |
| ancestral home | 4 | 0.09% |
| greater than | 4 | 0.09% |
| victory | 4 | 0.09% |
| executive producer | 4 | 0.09% |
| structural engineer | 4 | 0.09% |
| honorific suffix | 4 | 0.09% |
| location of formation | 4 | 0.09% |
| victim | 4 | 0.09% |
| foods traditionally associated | 4 | 0.09% |
| student | 4 | 0.09% |
| separated from | 4 | 0.09% |
| has anatomical branch | 4 | 0.09% |
| character type | 4 | 0.09% |
| measurement scale | 4 | 0.09% |
| mascot | 4 | 0.09% |
| instrumentation | 4 | 0.09% |
| foundational text | 3 | 0.07% |
| crystal system | 3 | 0.07% |
| music created for | 3 | 0.07% |
| animal breed | 3 | 0.07% |
| central bank | 3 | 0.07% |
| consecrator | 3 | 0.07% |
| designed to carry | 3 | 0.07% |
| home port | 3 | 0.07% |
| input method | 3 | 0.07% |
| reward program | 3 | 0.07% |
| trophy awarded | 3 | 0.07% |
| amended by | 3 | 0.07% |
| family name | 3 | 0.07% |
| structure replaced by | 3 | 0.07% |
| aspect ratio (W:H) | 3 | 0.07% |
| monomer of | 3 | 0.07% |
| official language | 3 | 0.07% |
| contains | 3 | 0.07% |
| offers view on | 3 | 0.07% |
| dialect of computer language | 3 | 0.07% |
| defendant | 3 | 0.07% |
| color | 3 | 0.07% |
| wing configuration | 3 | 0.07% |
| members have occupation | 3 | 0.07% |
| season | 3 | 0.07% |
| main food source | 3 | 0.07% |
| director of archaeological fieldwork | 3 | 0.07% |
| contains settlement | 3 | 0.07% |
| coextensive with | 3 | 0.07% |
| demonym of | 3 | 0.07% |
| less than | 3 | 0.07% |
| set in period | 3 | 0.07% |
| taxon synonym | 3 | 0.07% |
| points classification | 3 | 0.07% |
| cell component | 3 | 0.07% |
| contributing factor of | 3 | 0.07% |
| immediate cause of | 3 | 0.07% |
| time period | 3 | 0.07% |
| studied by | 3 | 0.07% |
| given name version for other gender | 3 | 0.07% |
| legislative body | 3 | 0.07% |
| enclave within | 3 | 0.07% |
| computes solution to | 3 | 0.07% |
| capital of | 3 | 0.07% |
| sports season of league or competition | 3 | 0.07% |
| economy of topic | 3 | 0.07% |
| has pet | 3 | 0.07% |
| sex or gender | 3 | 0.07% |
| practiced by | 3 | 0.07% |
| political ideology | 3 | 0.07% |
| derivative work | 3 | 0.07% |
| symptoms and signs | 3 | 0.07% |
| theme music | 3 | 0.07% |
| coat of arms | 3 | 0.07% |
| voice actor | 3 | 0.07% |
| used by | 3 | 0.07% |
| editor-in-chief | 3 | 0.07% |
| recorded at studio or venue | 3 | 0.07% |
| demographics of topic | 3 | 0.07% |
| expansion of | 2 | 0.05% |
| destroyed | 2 | 0.05% |
| has grammatical case | 2 | 0.05% |
| superhuman feature or ability | 2 | 0.05% |
| supported sports team | 2 | 0.05% |
| wears | 2 | 0.05% |
| carries scientific instrument | 2 | 0.05% |
| prerequisite | 2 | 0.05% |
| sponsor | 2 | 0.05% |
| source of energy | 2 | 0.05% |
| child astronomical body | 2 | 0.05% |
| conjugate base | 2 | 0.05% |
| depends on software | 2 | 0.05% |
| soundtrack release | 2 | 0.05% |
| intended public | 2 | 0.05% |
| people or cargo transported | 2 | 0.05% |
| molecular function | 2 | 0.05% |
| motto | 2 | 0.05% |
| designated as terrorist by | 2 | 0.05% |
| expressed in | 2 | 0.05% |
| perpetrator | 2 | 0.05% |
| depicted by | 2 | 0.05% |
| speaker | 2 | 0.05% |
| language regulatory body | 2 | 0.05% |
| statement describes | 2 | 0.05% |
| muscle insertion | 2 | 0.05% |
| depicts | 2 | 0.05% |
| field of this occupation | 2 | 0.05% |
| subject has role | 2 | 0.05% |
| measures | 2 | 0.05% |
| typically sells | 2 | 0.05% |
| youth wing | 2 | 0.05% |
| set in environment | 2 | 0.05% |
| innervated by | 2 | 0.05% |
| shares border with | 2 | 0.05% |
| sidekick of | 2 | 0.05% |
| top-level Internet domain | 2 | 0.05% |
| rector | 2 | 0.05% |
| legal form | 2 | 0.05% |
| list of works | 2 | 0.05% |
| diaspora | 2 | 0.05% |
| site of astronomical discovery | 2 | 0.05% |
| office contested | 2 | 0.05% |
| official residence | 2 | 0.05% |
| twinned administrative body | 2 | 0.05% |
| fastest lap | 2 | 0.05% |
| type foundry | 2 | 0.05% |
| addressee | 2 | 0.05% |
| calculated from | 2 | 0.05% |
| game mechanics | 2 | 0.05% |
| iconographic symbol | 2 | 0.05% |
| has facet polytope | 2 | 0.05% |
| territory claimed by | 2 | 0.05% |
| lake outflow | 2 | 0.05% |
| polymer of | 2 | 0.05% |
| has list | 2 | 0.05% |
| final event | 2 | 0.05% |
| innervates | 2 | 0.05% |
| successful candidate | 2 | 0.05% |
| drafted by | 2 | 0.05% |
| academic thesis | 2 | 0.05% |
| GUI toolkit or framework | 2 | 0.05% |
| has superpartner | 2 | 0.05% |
| temporal range start | 2 | 0.05% |
| funder | 2 | 0.05% |
| general manager | 2 | 0.05% |
| influenced by | 2 | 0.05% |
| connector | 2 | 0.05% |
| family name identical to this given name | 2 | 0.05% |
| engine configuration | 2 | 0.05% |
| medical examination | 2 | 0.05% |
| party chief representative | 2 | 0.05% |
| defender | 1 | 0.02% |
| bowling style | 1 | 0.02% |
| academic degree | 1 | 0.02% |
| charge | 1 | 0.02% |
| day in year for periodic occurrence | 1 | 0.02% |
| narrative motif | 1 | 0.02% |
| plot expanded in | 1 | 0.02% |
| risk factor | 1 | 0.02% |
| office held by head of state | 1 | 0.02% |
| doctoral student | 1 | 0.02% |
| develops from | 1 | 0.02% |
| fandom | 1 | 0.02% |
| character designer | 1 | 0.02% |
| majority opinion by | 1 | 0.02% |
| introduced feature | 1 | 0.02% |
| may prevent | 1 | 0.02% |
| colorist | 1 | 0.02% |
| country of registry | 1 | 0.02% |
| via | 1 | 0.02% |
| cites work | 1 | 0.02% |
| parliamentary group | 1 | 0.02% |
| film editor | 1 | 0.02% |
| narrator | 1 | 0.02% |
| has tense | 1 | 0.02% |
| parent peak | 1 | 0.02% |
| subdivision of this unit | 1 | 0.02% |
| appointed by | 1 | 0.02% |
| guidance system | 1 | 0.02% |
| facial hair | 1 | 0.02% |
| solid solution series with | 1 | 0.02% |
| stepparent | 1 | 0.02% |
| hybrid | 1 | 0.02% |
| supports programming language | 1 | 0.02% |
| uses capitalization for | 1 | 0.02% |
| delta of | 1 | 0.02% |
| adjacent building | 1 | 0.02% |
| does not have part | 1 | 0.02% |
| programmer | 1 | 0.02% |
| military unit | 1 | 0.02% |
| postsynaptic connection | 1 | 0.02% |
| OSI Model layer location | 1 | 0.02% |
| combination classification | 1 | 0.02% |
| sexually homologous with | 1 | 0.02% |
| exhibition history | 1 | 0.02% |
| complies with | 1 | 0.02% |
| course | 1 | 0.02% |
| has vertex figure | 1 | 0.02% |
| overlies | 1 | 0.02% |
| therapeutic area | 1 | 0.02% |
| appears in the form of | 1 | 0.02% |
| writable file format | 1 | 0.02% |
| Fach | 1 | 0.02% |
| possible medical findings | 1 | 0.02% |
| UNESCO language status | 1 | 0.02% |
| unmarried partner | 1 | 0.02% |
| cuisine | 1 | 0.02% |
| doctoral advisor | 1 | 0.02% |
| flag | 1 | 0.02% |
| interchange station | 1 | 0.02% |
| animator | 1 | 0.02% |
| approved by | 1 | 0.02% |
| military casualty classification | 1 | 0.02% |
| stereoisomer of | 1 | 0.02% |
| next lower rank | 1 | 0.02% |
| has fruit type | 1 | 0.02% |
| officially opened by | 1 | 0.02% |
| prosecutor | 1 | 0.02% |
| solved by | 1 | 0.02% |
| structure replaces | 1 | 0.02% |
| temporal range end | 1 | 0.02% |
| modified version of | 1 | 0.02% |
| conjugate acid | 1 | 0.02% |
| grammatical option indicates | 1 | 0.02% |
| emergency phone number | 1 | 0.02% |
| landscape architect | 1 | 0.02% |
| active ingredient in | 1 | 0.02% |
| mountains classification | 1 | 0.02% |
| cover art by | 1 | 0.02% |
| geomorphological unit | 1 | 0.02% |
| translator | 1 | 0.02% |
| peak bagging classification | 1 | 0.02% |
| patron saint | 1 | 0.02% |
| microstructure | 1 | 0.02% |
| end cause | 1 | 0.02% |
| curator | 1 | 0.02% |
| parent of this hybrid, breed, or cultivar | 1 | 0.02% |
| madhhab | 1 | 0.02% |
| Christian liturgical rite | 1 | 0.02% |
| named by | 1 | 0.02% |
| next crossing upstream | 1 | 0.02% |
| Thibaudeau classification | 1 | 0.02% |
| art director | 1 | 0.02% |
| diplomatic relation | 1 | 0.02% |
| record held | 1 | 0.02% |
| has part(s) of the class | 1 | 0.02% |
| emulates | 1 | 0.02% |
| has melody | 1 | 0.02% |
| mountain | 1 | 0.02% |

#### all

| label | count | share |
| --- | --- | --- |
| director | 219 | 0.51% |
| located in the administrative territorial entity | 219 | 0.51% |
| cast member | 216 | 0.50% |
| conferred by | 214 | 0.50% |
| present in work | 209 | 0.49% |
| architect | 209 | 0.49% |
| league | 207 | 0.48% |
| father | 203 | 0.47% |
| occupation | 202 | 0.47% |
| platform | 201 | 0.47% |
| conflict | 199 | 0.46% |
| historic county | 198 | 0.46% |
| heritage designation | 194 | 0.45% |
| employer | 194 | 0.45% |
| award received | 192 | 0.45% |
| original broadcaster | 190 | 0.44% |
| country | 190 | 0.44% |
| nominated for | 189 | 0.44% |
| performer | 189 | 0.44% |
| author | 188 | 0.44% |
| record label | 187 | 0.44% |
| characters | 185 | 0.43% |
| member of political party | 184 | 0.43% |
| genre | 184 | 0.43% |
| notable work | 183 | 0.43% |
| member of sports team | 183 | 0.43% |
| military branch | 183 | 0.43% |
| located in or next to body of water | 182 | 0.42% |
| licensed to broadcast to | 182 | 0.42% |
| member of | 181 | 0.42% |
| location | 180 | 0.42% |
| creator | 177 | 0.41% |
| discoverer or inventor | 177 | 0.41% |
| place served by transport hub | 176 | 0.41% |
| connecting line | 176 | 0.41% |
| position held | 173 | 0.40% |
| participant in | 170 | 0.40% |
| position played on team / speciality | 166 | 0.39% |
| mountain range | 165 | 0.38% |
| located in/on physical feature | 165 | 0.38% |
| basin country | 164 | 0.38% |
| item operated | 164 | 0.38% |
| mother | 164 | 0.38% |
| mouth of the watercourse | 163 | 0.38% |
| instrument | 163 | 0.38% |
| movement | 162 | 0.38% |
| participant | 161 | 0.38% |
| operator | 161 | 0.38% |
| spouse | 158 | 0.37% |
| family | 155 | 0.36% |
| participating team | 154 | 0.36% |
| parent organization | 153 | 0.36% |
| surface played on | 152 | 0.35% |
| located on astronomical body | 152 | 0.35% |
| located in the present-day administrative territorial entity | 152 | 0.35% |
| officeholder | 152 | 0.35% |
| sport | 152 | 0.35% |
| sibling | 149 | 0.35% |
| secretary general | 148 | 0.35% |
| online service | 146 | 0.34% |
| partner in business or sport | 146 | 0.34% |
| ammunition | 145 | 0.34% |
| terminus location | 144 | 0.34% |
| founded by | 144 | 0.34% |
| voice type | 143 | 0.33% |
| owned by | 143 | 0.33% |
| child | 142 | 0.33% |
| stock market index | 140 | 0.33% |
| parent taxon | 140 | 0.33% |
| instance of | 139 | 0.32% |
| chairperson | 139 | 0.32% |
| crosses | 137 | 0.32% |
| developer | 137 | 0.32% |
| significant event | 137 | 0.32% |
| continent | 137 | 0.32% |
| grants | 135 | 0.31% |
| competition class | 135 | 0.31% |
| noble title | 135 | 0.31% |
| language used | 135 | 0.31% |
| part of | 133 | 0.31% |
| presenter | 132 | 0.31% |
| represents | 131 | 0.31% |
| medical condition | 131 | 0.31% |
| composer | 131 | 0.31% |
| located in time zone | 130 | 0.30% |
| religion or worldview | 130 | 0.30% |
| broadcast by | 129 | 0.30% |
| headquarters location | 129 | 0.30% |
| has subsidiary | 129 | 0.30% |
| educated at | 128 | 0.30% |
| publisher | 128 | 0.30% |
| place of detention | 127 | 0.30% |
| product or material produced | 127 | 0.30% |
| dam | 126 | 0.29% |
| located in the statistical territorial entity | 126 | 0.29% |
| package management system | 126 | 0.29% |
| has use | 126 | 0.29% |
| manufacturer | 125 | 0.29% |
| home venue | 124 | 0.29% |
| form of creative work | 123 | 0.29% |
| drug or therapy used for treatment | 122 | 0.28% |
| represented by | 121 | 0.28% |
| constellation | 120 | 0.28% |
| historical region | 120 | 0.28% |
| crew member(s) | 120 | 0.28% |
| member of the crew of | 120 | 0.28% |
| incarnation of | 119 | 0.28% |
| contains the administrative territorial entity | 119 | 0.28% |
| terminus | 119 | 0.28% |
| astronaut mission | 118 | 0.28% |
| occupant | 118 | 0.28% |
| medical condition treated | 117 | 0.27% |
| fictional universe described in | 117 | 0.27% |
| location of formation | 117 | 0.27% |
| court | 116 | 0.27% |
| location of discovery | 116 | 0.27% |
| model item | 115 | 0.27% |
| winner | 115 | 0.27% |
| replaced by | 115 | 0.27% |
| takes place in fictional universe | 114 | 0.27% |
| commemorates | 114 | 0.27% |
| distributed by | 114 | 0.27% |
| convicted of | 113 | 0.26% |
| place of publication | 113 | 0.26% |
| this zoological name is coordinate with | 113 | 0.26% |
| named after | 113 | 0.26% |
| carries | 112 | 0.26% |
| vehicle | 111 | 0.26% |
| stock exchange | 111 | 0.26% |
| has part(s) | 111 | 0.26% |
| recommended unit of measurement | 110 | 0.26% |
| is the study of | 110 | 0.26% |
| work location | 110 | 0.26% |
| country of citizenship | 110 | 0.26% |
| main subject | 110 | 0.26% |
| sports discipline competed in | 109 | 0.25% |
| said to be the same as | 109 | 0.25% |
| field of work | 109 | 0.25% |
| this taxon is source of | 108 | 0.25% |
| found in taxon | 108 | 0.25% |
| biological process | 108 | 0.25% |
| production company | 108 | 0.25% |
| indigenous to | 108 | 0.25% |
| published in | 106 | 0.25% |
| filming location | 105 | 0.24% |
| connects with | 105 | 0.24% |
| has works in the collection | 104 | 0.24% |
| significant person | 104 | 0.24% |
| ethnic group | 104 | 0.24% |
| musical conductor | 103 | 0.24% |
| military rank | 103 | 0.24% |
| taxonomic type | 103 | 0.24% |
| school district | 103 | 0.24% |
| has effect | 103 | 0.24% |
| parent club | 102 | 0.24% |
| has written for | 101 | 0.24% |
| head of state | 101 | 0.24% |
| owner of | 101 | 0.24% |
| country of origin | 101 | 0.24% |
| original language of film or TV show | 100 | 0.23% |
| tracklist | 100 | 0.23% |
| cathedral | 100 | 0.23% |
| commanded by | 100 | 0.23% |
| uses | 100 | 0.23% |
| located on street | 100 | 0.23% |
| writing language | 99 | 0.23% |
| pole position | 99 | 0.23% |
| opposite of | 98 | 0.23% |
| place of birth | 98 | 0.23% |
| collection | 97 | 0.23% |
| reservoir created | 97 | 0.23% |
| anatomical location | 97 | 0.23% |
| league system | 97 | 0.23% |
| country for sport | 97 | 0.23% |
| worshipped by | 96 | 0.22% |
| legislated by | 96 | 0.22% |
| IUCN conservation status | 96 | 0.22% |
| merged into | 96 | 0.22% |
| after a work by | 96 | 0.22% |
| vessel class | 95 | 0.22% |
| dialect of | 95 | 0.22% |
| standards body | 95 | 0.22% |
| has edition or translation | 94 | 0.22% |
| track gauge | 94 | 0.22% |
| head of government | 94 | 0.22% |
| chief executive officer | 93 | 0.22% |
| location of first performance | 93 | 0.22% |
| operating area | 92 | 0.21% |
| copyright license | 92 | 0.21% |
| wheel arrangement | 92 | 0.21% |
| brand | 91 | 0.21% |
| made from material | 91 | 0.21% |
| producer | 91 | 0.21% |
| start point | 90 | 0.21% |
| airline alliance | 90 | 0.21% |
| allegiance | 90 | 0.21% |
| from narrative universe | 89 | 0.21% |
| dedicated to | 89 | 0.21% |
| followed by | 89 | 0.21% |
| residence | 89 | 0.21% |
| currency | 88 | 0.21% |
| architectural style | 88 | 0.21% |
| culture | 87 | 0.20% |
| basic form of government | 87 | 0.20% |
| produced by | 87 | 0.20% |
| editor | 86 | 0.20% |
| next higher rank | 86 | 0.20% |
| transport network | 85 | 0.20% |
| natural product of taxon | 85 | 0.20% |
| objective of project or action | 85 | 0.20% |
| illustrator | 83 | 0.19% |
| commander of (DEPRECATED) | 83 | 0.19% |
| co-driver | 82 | 0.19% |
| located in protected area | 81 | 0.19% |
| significant place | 81 | 0.19% |
| interested in | 81 | 0.19% |
| manner of death | 81 | 0.19% |
| endemic to | 81 | 0.19% |
| has cause | 81 | 0.19% |
| taxon rank | 81 | 0.19% |
| applies to jurisdiction | 81 | 0.19% |
| diocese | 80 | 0.19% |
| history of topic | 80 | 0.19% |
| space launch vehicle | 80 | 0.19% |
| operating system | 80 | 0.19% |
| relative | 80 | 0.19% |
| voice actor | 80 | 0.19% |
| part of the series | 80 | 0.19% |
| chromosome | 79 | 0.18% |
| lyrics by | 79 | 0.18% |
| facet of | 79 | 0.18% |
| board member | 78 | 0.18% |
| physically interacts with | 78 | 0.18% |
| executive body | 78 | 0.18% |
| genetic association | 78 | 0.18% |
| highest judicial authority | 78 | 0.18% |
| based on | 78 | 0.18% |
| librettist | 77 | 0.18% |
| edition or translation of | 77 | 0.18% |
| radio format | 77 | 0.18% |
| tributary | 77 | 0.18% |
| destination point | 76 | 0.18% |
| has immediate cause | 76 | 0.18% |
| programming language | 75 | 0.17% |
| qualifies for event | 75 | 0.17% |
| main regulatory text | 75 | 0.17% |
| language of work or name | 75 | 0.17% |
| territory overlaps | 74 | 0.17% |
| languages spoken, written or signed | 74 | 0.17% |
| target | 73 | 0.17% |
| lake on watercourse | 73 | 0.17% |
| affiliation | 73 | 0.17% |
| has part(s) of the class | 73 | 0.17% |
| adjacent station | 73 | 0.17% |
| substitute/deputy/replacement of office/officeholder | 72 | 0.17% |
| copyright holder | 72 | 0.17% |
| discography | 72 | 0.17% |
| coach of sports team | 72 | 0.17% |
| subclass of | 72 | 0.17% |
| powered by | 71 | 0.17% |
| parent astronomical body | 71 | 0.17% |
| killed by | 71 | 0.17% |
| candidate | 71 | 0.17% |
| replaces | 71 | 0.17% |
| inflows | 70 | 0.16% |
| authority | 70 | 0.16% |
| measured physical quantity | 70 | 0.16% |
| time period | 70 | 0.16% |
| encodes | 69 | 0.16% |
| has quality | 69 | 0.16% |
| unmarried partner | 69 | 0.16% |
| drainage basin | 68 | 0.16% |
| distribution format | 68 | 0.16% |
| designed by | 68 | 0.16% |
| primary destinations | 68 | 0.16% |
| candidacy in election | 68 | 0.16% |
| origin of the watercourse | 67 | 0.16% |
| location of creation | 67 | 0.16% |
| maintained by | 67 | 0.16% |
| anthem | 66 | 0.15% |
| director / manager | 66 | 0.15% |
| different from | 66 | 0.15% |
| canonization status | 65 | 0.15% |
| narrative location | 65 | 0.15% |
| commissioned by | 64 | 0.15% |
| house publication | 64 | 0.15% |
| honorific prefix | 64 | 0.15% |
| general classification of race participants | 64 | 0.15% |
| head coach | 64 | 0.15% |
| follows | 64 | 0.15% |
| first appearance | 63 | 0.15% |
| organizer | 63 | 0.15% |
| religious order | 63 | 0.15% |
| fabrication method | 62 | 0.14% |
| family name | 61 | 0.14% |
| native language | 61 | 0.14% |
| mascot | 60 | 0.14% |
| place of burial | 60 | 0.14% |
| subject has role | 60 | 0.14% |
| signatory | 60 | 0.14% |
| industry | 60 | 0.14% |
| student of | 60 | 0.14% |
| official language | 59 | 0.14% |
| contributed to creative work | 59 | 0.14% |
| enemy of | 59 | 0.14% |
| lake outflow | 58 | 0.14% |
| office held by head of the organization | 58 | 0.14% |
| cause of death | 58 | 0.14% |
| connecting service | 58 | 0.14% |
| political ideology | 58 | 0.14% |
| shares border with | 58 | 0.14% |
| programming paradigm | 57 | 0.13% |
| cell component | 57 | 0.13% |
| armament | 57 | 0.13% |
| choreographer | 57 | 0.13% |
| public holiday | 57 | 0.13% |
| domain of saint or deity | 56 | 0.13% |
| cause of destruction | 56 | 0.13% |
| foundational text | 56 | 0.13% |
| consecrator | 56 | 0.13% |
| legal form | 56 | 0.13% |
| official religion | 55 | 0.13% |
| health specialty | 55 | 0.13% |
| business division | 55 | 0.13% |
| official residence | 55 | 0.13% |
| writing system | 55 | 0.13% |
| victory | 55 | 0.13% |
| structural engineer | 54 | 0.13% |
| implementation of | 54 | 0.13% |
| coextensive with | 54 | 0.13% |
| anatomical branch of | 54 | 0.13% |
| described by source | 54 | 0.13% |
| child astronomical body | 54 | 0.13% |
| place of death | 54 | 0.13% |
| highest point | 53 | 0.12% |
| location of landing | 53 | 0.12% |
| perpetrator | 53 | 0.12% |
| political alignment | 53 | 0.12% |
| media franchise | 52 | 0.12% |
| Gram staining | 52 | 0.12% |
| software engine | 51 | 0.12% |
| field of this occupation | 51 | 0.12% |
| linguistic typology | 51 | 0.12% |
| partially coincident with | 51 | 0.12% |
| screenwriter | 51 | 0.12% |
| defendant | 50 | 0.12% |
| sponsor | 49 | 0.11% |
| ancestral home | 49 | 0.11% |
| vehicle normally used | 49 | 0.11% |
| contains | 48 | 0.11% |
| organization directed by the office or position | 48 | 0.11% |
| sex or gender | 48 | 0.11% |
| practiced by | 48 | 0.11% |
| immediate cause of | 47 | 0.11% |
| inspired by | 47 | 0.11% |
| game mechanics | 46 | 0.11% |
| title of chess person | 46 | 0.11% |
| advertises | 46 | 0.11% |
| legislative body | 46 | 0.11% |
| used by | 46 | 0.11% |
| symptoms and signs | 45 | 0.10% |
| league level above | 45 | 0.10% |
| office held by head of government | 45 | 0.10% |
| director of photography | 45 | 0.10% |
| series spin-off | 43 | 0.10% |
| engine configuration | 43 | 0.10% |
| manifestation of | 43 | 0.10% |
| derivative work | 43 | 0.10% |
| home world | 42 | 0.10% |
| type of variable star | 42 | 0.10% |
| afflicts | 42 | 0.10% |
| subdivision of this unit | 41 | 0.10% |
| site of astronomical discovery | 41 | 0.10% |
| airline hub | 41 | 0.10% |
| possible treatment | 40 | 0.09% |
| newspaper format | 40 | 0.09% |
| source of energy | 40 | 0.09% |
| depicts | 40 | 0.09% |
| GUI toolkit or framework | 39 | 0.09% |
| capital of | 39 | 0.09% |
| dual to | 38 | 0.09% |
| has anatomical branch | 38 | 0.09% |
| flag | 37 | 0.09% |
| fastest lap | 37 | 0.09% |
| minor planet group | 36 | 0.08% |
| character type | 36 | 0.08% |
| mount | 36 | 0.08% |
| central bank | 36 | 0.08% |
| depicted by | 36 | 0.08% |
| league level below | 36 | 0.08% |
| victim | 35 | 0.08% |
| contains settlement | 35 | 0.08% |
| game mode | 35 | 0.08% |
| separated from | 35 | 0.08% |
| measures | 34 | 0.08% |
| instrumentation | 34 | 0.08% |
| contributor to the creative work or subject | 34 | 0.08% |
| input method | 34 | 0.08% |
| sexually homologous with | 34 | 0.08% |
| music created for | 33 | 0.08% |
| members have occupation | 33 | 0.08% |
| successful candidate | 33 | 0.08% |
| capital | 33 | 0.08% |
| enclave within | 32 | 0.07% |
| writable file format | 32 | 0.07% |
| director of archaeological fieldwork | 32 | 0.07% |
| color | 32 | 0.07% |
| patron saint | 31 | 0.07% |
| office held by head of state | 31 | 0.07% |
| editor-in-chief | 31 | 0.07% |
| studied by | 31 | 0.07% |
| sexual orientation | 31 | 0.07% |
| student | 31 | 0.07% |
| territory claimed by | 31 | 0.07% |
| measurement scale | 30 | 0.07% |
| less than | 30 | 0.07% |
| social classification | 30 | 0.07% |
| sports season of league or competition | 30 | 0.07% |
| type of orbit | 29 | 0.07% |
| central bank/issuer | 29 | 0.07% |
| CPU | 29 | 0.07% |
| monomer of | 29 | 0.07% |
| interchange station | 29 | 0.07% |
| party chief representative | 29 | 0.07% |
| superhuman feature or ability | 28 | 0.07% |
| economy of topic | 28 | 0.07% |
| set in period | 28 | 0.07% |
| appears in the form of | 28 | 0.07% |
| greater than | 28 | 0.07% |
| drafted by | 28 | 0.07% |
| vaccine for | 28 | 0.07% |
| statistical leader | 27 | 0.06% |
| associated electoral district | 27 | 0.06% |
| business model | 27 | 0.06% |
| youth wing | 27 | 0.06% |
| penalty | 27 | 0.06% |
| contributing factor of | 27 | 0.06% |
| wing configuration | 26 | 0.06% |
| has boundary | 26 | 0.06% |
| designated as terrorist by | 26 | 0.06% |
| interaction | 26 | 0.06% |
| underlies | 26 | 0.06% |
| molecular function | 26 | 0.06% |
| language regulatory body | 26 | 0.06% |
| speaker | 25 | 0.06% |
| archives at | 25 | 0.06% |
| day in year for periodic occurrence | 25 | 0.06% |
| has pet | 25 | 0.06% |
| home port | 25 | 0.06% |
| sidekick of | 25 | 0.06% |
| cuisine | 25 | 0.06% |
| season | 24 | 0.06% |
| structure replaced by | 24 | 0.06% |
| country of registry | 24 | 0.06% |
| risk factor | 23 | 0.05% |
| animal breed | 23 | 0.05% |
| captain | 23 | 0.05% |
| has contributing factor | 23 | 0.05% |
| intended public | 23 | 0.05% |
| wears | 23 | 0.05% |
| offers view on | 23 | 0.05% |
| mountains classification | 22 | 0.05% |
| depends on software | 22 | 0.05% |
| designed to carry | 22 | 0.05% |
| foods traditionally associated | 22 | 0.05% |
| overlies | 22 | 0.05% |
| performed at | 22 | 0.05% |
| recorded at studio or venue | 22 | 0.05% |
| final event | 22 | 0.05% |
| readable file format | 22 | 0.05% |
| influenced by | 22 | 0.05% |
| executive producer | 21 | 0.05% |
| amended by | 21 | 0.05% |
| in opposition to | 21 | 0.05% |
| proved by | 21 | 0.05% |
| scheduled service destination | 21 | 0.05% |
| emulates | 21 | 0.05% |
| active ingredient in | 20 | 0.05% |
| carries scientific instrument | 20 | 0.05% |
| charge | 20 | 0.05% |
| family name identical to this given name | 20 | 0.05% |
| stepparent | 20 | 0.05% |
| mother house | 20 | 0.05% |
| parent of this hybrid, breed, or cultivar | 20 | 0.05% |
| appointed by | 19 | 0.04% |
| has active ingredient | 19 | 0.04% |
| soundtrack release | 19 | 0.04% |
| points/goal scored by | 19 | 0.04% |
| given name version for other gender | 19 | 0.04% |
| diplomatic relation | 19 | 0.04% |
| expressed in | 18 | 0.04% |
| approved by | 18 | 0.04% |
| office contested | 18 | 0.04% |
| statement describes | 18 | 0.04% |
| has grammatical case | 18 | 0.04% |
| theme music | 18 | 0.04% |
| regulated by | 18 | 0.04% |
| honorific suffix | 18 | 0.04% |
| twinned administrative body | 18 | 0.04% |
| connector | 18 | 0.04% |
| destroyed | 17 | 0.04% |
| antiparticle | 17 | 0.04% |
| iconographic symbol | 17 | 0.04% |
| next lower rank | 17 | 0.04% |
| medical examination | 17 | 0.04% |
| next crossing upstream | 16 | 0.04% |
| structure replaces | 16 | 0.04% |
| Christian liturgical rite | 16 | 0.04% |
| has list | 16 | 0.04% |
| motto | 16 | 0.04% |
| innervates | 16 | 0.04% |
| points classification | 16 | 0.04% |
| type foundry | 15 | 0.03% |
| film editor | 15 | 0.03% |
| has natural reservoir | 15 | 0.03% |
| coat of arms | 15 | 0.03% |
| academic degree | 15 | 0.03% |
| polymer of | 15 | 0.03% |
| instruction set | 15 | 0.03% |
| gens | 15 | 0.03% |
| innervated by | 15 | 0.03% |
| solved by | 15 | 0.03% |
| expansion of | 14 | 0.03% |
| prohibits | 14 | 0.03% |
| top-level Internet domain | 14 | 0.03% |
| computes solution to | 14 | 0.03% |
| mod of | 14 | 0.03% |
| parliamentary group | 14 | 0.03% |
| species kept | 14 | 0.03% |
| trophy awarded | 14 | 0.03% |
| typically sells | 14 | 0.03% |
| main food source | 14 | 0.03% |
| dialect of computer language | 13 | 0.03% |
| lifestyle | 13 | 0.03% |
| named by | 13 | 0.03% |
| shape | 13 | 0.03% |
| translator | 13 | 0.03% |
| has fruit type | 13 | 0.03% |
| delta of | 13 | 0.03% |
| feast day | 13 | 0.03% |
| diaspora | 13 | 0.03% |
| update method | 12 | 0.03% |
| prerequisite | 12 | 0.03% |
| product certification | 12 | 0.03% |
| adjacent building | 12 | 0.03% |
| narrator | 12 | 0.03% |
| exclave of | 12 | 0.03% |
| set in environment | 12 | 0.03% |
| parent peak | 12 | 0.03% |
| crystal system | 11 | 0.03% |
| copyright representative | 11 | 0.03% |
| source of material | 11 | 0.03% |
| set during recurring event | 11 | 0.03% |
| temporal range start | 11 | 0.03% |
| people or cargo transported | 11 | 0.03% |
| charted in | 11 | 0.03% |
| military casualty classification | 10 | 0.02% |
| reward program | 10 | 0.02% |
| supported sports team | 10 | 0.02% |
| addressee | 10 | 0.02% |
| host | 10 | 0.02% |
| madhhab | 10 | 0.02% |
| officially opened by | 10 | 0.02% |
| academic thesis | 10 | 0.02% |
| cover art by | 10 | 0.02% |
| demographics of topic | 10 | 0.02% |
| parallel street | 10 | 0.02% |
| rector | 9 | 0.02% |
| has phenotype | 9 | 0.02% |
| size designation | 9 | 0.02% |
| tonality | 9 | 0.02% |
| via | 9 | 0.02% |
| geography of topic | 9 | 0.02% |
| taxon synonym | 9 | 0.02% |
| colorist | 9 | 0.02% |
| main building contractor | 9 | 0.02% |
| peak bagging classification | 9 | 0.02% |
| exhibition history | 9 | 0.02% |
| is a list of | 9 | 0.02% |
| geomorphological unit | 9 | 0.02% |
| programmer | 8 | 0.02% |
| stereoisomer of | 8 | 0.02% |
| aspect ratio (W:H) | 8 | 0.02% |
| significant drug interaction | 8 | 0.02% |
| does not have part | 8 | 0.02% |
| OSI Model layer location | 8 | 0.02% |
| general manager | 8 | 0.02% |
| has seal, badge, or sigil | 8 | 0.02% |
| has facet polytope | 8 | 0.02% |
| copyright status | 7 | 0.02% |
| character designer | 7 | 0.02% |
| narrative role | 7 | 0.02% |
| train depot | 7 | 0.02% |
| type locality (geology) | 7 | 0.02% |
| worn by | 7 | 0.02% |
| production designer | 7 | 0.02% |
| theorised by | 7 | 0.02% |
| valid in period | 7 | 0.02% |
| official symbol | 7 | 0.02% |
| referee | 7 | 0.02% |
| cites work | 7 | 0.02% |
| postsynaptic connection | 7 | 0.02% |
| end cause | 7 | 0.02% |
| chassis | 7 | 0.02% |
| calculated from | 6 | 0.01% |
| fandom | 6 | 0.01% |
| issued by | 6 | 0.01% |
| narrative motif | 6 | 0.01% |
| antagonist muscle | 6 | 0.01% |
| arterial supply | 6 | 0.01% |
| develops from | 6 | 0.01% |
| political coalition | 6 | 0.01% |
| supports programming language | 6 | 0.01% |
| sports league level | 6 | 0.01% |
| funder | 6 | 0.01% |
| microarchitecture | 6 | 0.01% |
| curator | 6 | 0.01% |
| suggests the existence of | 6 | 0.01% |
| disease transmission process | 6 | 0.01% |
| month of the year | 5 | 0.01% |
| adapted by | 5 | 0.01% |
| complies with | 5 | 0.01% |
| official podcast | 5 | 0.01% |
| conjugate acid | 5 | 0.01% |
| discovery method | 5 | 0.01% |
| notation | 5 | 0.01% |
| uses capitalization for | 5 | 0.01% |
| conjugate base | 5 | 0.01% |
| donated by | 5 | 0.01% |
| venous drainage | 5 | 0.01% |
| majority opinion by | 5 | 0.01% |
| repeals | 5 | 0.01% |
| season of club or team | 5 | 0.01% |
| military unit | 5 | 0.01% |
| research site | 5 | 0.01% |
| produced sound | 5 | 0.01% |
| microstructure | 5 | 0.01% |
| has melody | 5 | 0.01% |
| choral conductor | 5 | 0.01% |
| exemplar of | 5 | 0.01% |
| launch contractor | 5 | 0.01% |
| collection creator | 5 | 0.01% |
| animator | 5 | 0.01% |
| emergency phone number | 4 | 0.01% |
| day of week | 4 | 0.01% |
| order of battle | 4 | 0.01% |
| category of associated people | 4 | 0.01% |
| Fach | 4 | 0.01% |
| plot expanded in | 4 | 0.01% |
| prime factor | 4 | 0.01% |
| record held | 4 | 0.01% |
| solid solution series with | 4 | 0.01% |
| substrate of | 4 | 0.01% |
| nominated by | 4 | 0.01% |
| by-product | 4 | 0.01% |
| doctoral advisor | 4 | 0.01% |
| style of karate | 4 | 0.01% |
| godparent | 4 | 0.01% |
| coolant | 4 | 0.01% |
| ratified by | 4 | 0.01% |
| temporal range end | 4 | 0.01% |
| laws applied | 4 | 0.01% |
| landscape architect | 4 | 0.01% |
| decays to | 4 | 0.01% |
| demonym of | 4 | 0.01% |
| next crossing downstream | 4 | 0.01% |
| output method | 4 | 0.01% |
| religious occupation | 4 | 0.01% |
| costume designer | 4 | 0.01% |
| modified version of | 4 | 0.01% |
| elector | 4 | 0.01% |
| list of works | 4 | 0.01% |
| asteroid spectral type | 4 | 0.01% |
| given name | 4 | 0.01% |
| has census | 4 | 0.01% |
| partnership with | 4 | 0.01% |
| professorship | 4 | 0.01% |
| era name | 4 | 0.01% |
| port of registry | 3 | 0.01% |
| climbing route | 3 | 0.01% |
| combination classification | 3 | 0.01% |
| comorbidity | 3 | 0.01% |
| intangible cultural heritage status | 3 | 0.01% |
| statement is subject of | 3 | 0.01% |
| visitor centre | 3 | 0.01% |
| permanent resident of | 3 | 0.01% |
| typing discipline | 3 | 0.01% |
| does not have quality | 3 | 0.01% |
| doctoral student | 3 | 0.01% |
| edibility | 3 | 0.01% |
| plaintiff | 3 | 0.01% |
| guidance system | 3 | 0.01% |
| introduced feature | 3 | 0.01% |
| facial hair | 3 | 0.01% |
| symbolizes | 3 | 0.01% |
| art director | 3 | 0.01% |
| investor | 3 | 0.01% |
| possible medical findings | 3 | 0.01% |
| muscle insertion | 3 | 0.01% |
| lowest point | 3 | 0.01% |
| binding of software library | 3 | 0.01% |
| relegated | 3 | 0.01% |
| has listed ingredient | 3 | 0.01% |
| electoral district | 3 | 0.01% |
| together with | 3 | 0.01% |
| disjoint union of | 3 | 0.01% |
| repealed by | 3 | 0.01% |
| official color | 3 | 0.01% |
| has superpartner | 3 | 0.01% |
| protocol | 3 | 0.01% |
| is metaclass for | 3 | 0.01% |
| nominee | 2 | 0.00% |
| base | 2 | 0.00% |
| original film format | 2 | 0.00% |
| by-product of | 2 | 0.00% |
| display technology | 2 | 0.00% |
| has tense | 2 | 0.00% |
| open data portal | 2 | 0.00% |
| companion of | 2 | 0.00% |
| reply to | 2 | 0.00% |
| tournament format | 2 | 0.00% |
| website account on | 2 | 0.00% |
| may prevent | 2 | 0.00% |
| decreased expression in | 2 | 0.00% |
| derived from organism type | 2 | 0.00% |
| distributary | 2 | 0.00% |
| item disputed by | 2 | 0.00% |
| supplement to | 2 | 0.00% |
| biological variant of | 2 | 0.00% |
| Roman agnomen | 2 | 0.00% |
| encoded by | 2 | 0.00% |
| undercarriage | 2 | 0.00% |
| metrically compatible typeface | 2 | 0.00% |
| route of administration | 2 | 0.00% |
| fault | 2 | 0.00% |
| therapeutic area | 2 | 0.00% |
| relative to | 2 | 0.00% |
| avionics | 2 | 0.00% |
| production statistics | 2 | 0.00% |
| first aid measures | 2 | 0.00% |
| invasive to | 2 | 0.00% |
| samples from work | 2 | 0.00% |
| has facility | 2 | 0.00% |
| side effect | 2 | 0.00% |
| UNESCO language status | 2 | 0.00% |
| volcano observatory | 2 | 0.00% |
| habitat | 2 | 0.00% |
| codomain | 2 | 0.00% |
| chief operating officer | 2 | 0.00% |
| regulates (molecular biology) | 2 | 0.00% |
| grammatical option indicates | 2 | 0.00% |
| promoted | 2 | 0.00% |
| province | 2 | 0.00% |
| religious movement | 2 | 0.00% |
| defender | 1 | 0.00% |
| bowling style | 1 | 0.00% |
| epithet | 1 | 0.00% |
| GPU | 1 | 0.00% |
| catalog | 1 | 0.00% |
| external auditor | 1 | 0.00% |
| muscle origin | 1 | 0.00% |
| overrules | 1 | 0.00% |
| radix | 1 | 0.00% |
| scenographer | 1 | 0.00% |
| socket supported | 1 | 0.00% |
| used metre | 1 | 0.00% |
| fictional analog of | 1 | 0.00% |
| taxon found at location | 1 | 0.00% |
| recording location | 1 | 0.00% |
| place of origin (Switzerland) | 1 | 0.00% |
| start grade | 1 | 0.00% |
| artistic director | 1 | 0.00% |
| activating neurotransmitter | 1 | 0.00% |
| has member | 1 | 0.00% |
| ecoregion (WWF) | 1 | 0.00% |
| including | 1 | 0.00% |
| pendant of | 1 | 0.00% |
| real estate developer | 1 | 0.00% |
| storyboard artist | 1 | 0.00% |
| oath made by | 1 | 0.00% |
| dissertation submitted to | 1 | 0.00% |
| established from medical condition | 1 | 0.00% |
| ICTV virus genome composition | 1 | 0.00% |
| inscription mentions | 1 | 0.00% |
| card network | 1 | 0.00% |
| currency symbol description | 1 | 0.00% |
| hybrid | 1 | 0.00% |
| mode of inheritance | 1 | 0.00% |
| bibliography | 1 | 0.00% |
| is pollinated by | 1 | 0.00% |
| admissible rule in | 1 | 0.00% |
| index case of | 1 | 0.00% |
| gait | 1 | 0.00% |
| reissue of | 1 | 0.00% |
| has pattern | 1 | 0.00% |
| course | 1 | 0.00% |
| has vertex figure | 1 | 0.00% |
| ortholog | 1 | 0.00% |
| founder | 1 | 0.00% |
| fictional character | 1 | 0.00% |
| film crew member | 1 | 0.00% |
| entry receptor | 1 | 0.00% |
| source of income | 1 | 0.00% |
| decay mode | 1 | 0.00% |
| follower of | 1 | 0.00% |
| criterion used | 1 | 0.00% |
| exonerated of | 1 | 0.00% |
| prosecutor | 1 | 0.00% |
| Vox-ATypI classification | 1 | 0.00% |
| has phoneme | 1 | 0.00% |
| leader | 1 | 0.00% |
| record or record progression | 1 | 0.00% |
| place of devotion | 1 | 0.00% |
| academic appointment | 1 | 0.00% |
| fossil found in this unit | 1 | 0.00% |
| indexed in bibliographic review | 1 | 0.00% |
| issuing agent of work | 1 | 0.00% |
| young rider classification | 1 | 0.00% |
| presynaptic connection | 1 | 0.00% |
| has role in modeling | 1 | 0.00% |
| means of locomotion | 1 | 0.00% |
| generalization of | 1 | 0.00% |
| located on linear feature | 1 | 0.00% |
| reward | 1 | 0.00% |
| inhibitor of | 1 | 0.00% |
| facilitates flow of | 1 | 0.00% |
| damaged | 1 | 0.00% |
| Thibaudeau classification | 1 | 0.00% |
| associated hazard | 1 | 0.00% |
| mushroom ecological type | 1 | 0.00% |
| parliamentary term | 1 | 0.00% |
| election called by | 1 | 0.00% |
| kit supplier | 1 | 0.00% |
| blocker of | 1 | 0.00% |
| mountain | 1 | 0.00% |

### Classifications

None.

### Structures

None.

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | distant |
| relations | distant |
| events | distant |
| classifications | distant |
| structures | distant |
