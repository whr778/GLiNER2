# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| text2json | train | 7,976 | 1,715 | 5,598 | 0 | 0 | 7,754 | 0 | 0 |
| text2json | val | 891 | 193 | 618 | 0 | 0 | 863 | 0 | 0 |
| text2json | test | 872 | 186 | 398 | 0 | 0 | 856 | 0 | 0 |

## text2json

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 7,976 | 1,715 | 429 | 1490 | 24923 | 5,598 | 0 | 0 | 7,754 | 0 | 0 |
| val | 891 | 193 | 409 | 1702 | 36347 | 618 | 0 | 0 | 863 | 0 | 0 |
| test | 872 | 186 | 515 | 1509 | 9260 | 398 | 0 | 0 | 856 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 6,051 | 702 | 646 | 7,399 | 75.97% |
| und | 602 | 52 | 45 | 699 | 7.18% |
| fra | 157 | 5 | 20 | 182 | 1.87% |
| rus | 134 | 10 | 20 | 164 | 1.68% |
| por | 109 | 20 | 10 | 139 | 1.43% |
| spa | 81 | 15 | 25 | 121 | 1.24% |
| pol | 92 | 15 | 10 | 117 | 1.20% |
| hin | 89 | 20 | 5 | 114 | 1.17% |
| ukr | 85 | 5 | 10 | 100 | 1.03% |
| nld | 69 | 14 | 17 | 100 | 1.03% |
| kor | 81 | 5 | 9 | 95 | 0.98% |
| tur | 75 | 10 | 5 | 90 | 0.92% |
| deu | 69 | 5 | 15 | 89 | 0.91% |
| ita | 73 | 5 | 10 | 88 | 0.90% |
| ara | 77 | 5 | 5 | 87 | 0.89% |
| ces | 63 | 3 | 20 | 86 | 0.88% |
| zho | 65 | 0 | 0 | 65 | 0.67% |
| epo | 4 | 0 | 0 | 4 | 0.04% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 872 |
| test_unique_inputs | 186 |
| train_records | 7,976 |
| train_unique_inputs | 1,715 |
| val_records | 891 |
| val_unique_inputs | 193 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.6310 STRONG | 5,598 / 618 | Person 15%→8%, person 0%→4%, physical_process 0%→2%, Cognitive Process 0%→2%, Disease Subtype 0%→2% |
| entities (per mention) | train / test | 0.7572 STRONG | 5,598 / 398 | Location 10%→0%, Organization 7%→2%, Person 15%→11%, Design Principle 0%→4%, 3D Shape 0%→3% |
| entities (per mention) | val / test | 0.8386 STRONG | 618 / 398 | Location 9%→0%, Organization 7%→2%, person 4%→0%, Design Principle 0%→4%, 3D Shape 0%→3% |

Unseen in train, present in val, entities (per mention): Cognitive Process ×14, physical_process ×14, string_theory_entity ×12, Device ×10, formal_result ×10, Medical Procedure ×10, Skill Area ×10, astrophysical_phenomenon ×8, scientific_instrument ×8, mathematical_matrix ×8, mathematical_parameter ×6, Cellular Structure ×6, Chemical Element ×6, Data Visualization ×6, Attribute ×6, theoretical_physics_concept ×6, Application ×4, particle ×4, celestial_body ×4, mathematical_method ×4, mathematical concept ×4, Nutrient ×4, Geographical Feature ×4, Physics ×2, experiment ×2, spacecraft ×2, Pollutant Category ×2, Syndrome ×2, mathematical property ×2, Physical Force ×2, Biological Unit ×2, Award ×2, Cellular Component ×2, Biological Process ×2, Agent ×2, Climate Metric ×2, Atmospheric Phenomenon ×2, Season ×2, Instruction ×2, Skill ×2, Framework ×2, research_group ×2, field_theory_model ×2.
Thin support in val, entities (per mention): 80 label(s) under 10 gold. Material ×2, Physics ×2, experiment ×2, spacecraft ×2, publication ×2, Pollutant Category ×2, Program ×2, Disorder ×2, Syndrome ×2, Symptom ×2, Publication ×2, Test ×2, Cell Type ×2, Anatomy ×2, mathematical property ×2, Physical Force ×2, Organ ×2, Medical Condition ×2, Biological Unit ×2, Award ×2, Cellular Component ×2, Biological Process ×2, Biological System ×2, Biological Entity ×2, Cause ×2, Category ×2, Factor ×2, Structure ×2, Agent ×2, Substance ×2, Standard ×2, Climate Metric ×2, Atmospheric Phenomenon ×2, Event ×2, Season ×2, Literary Work ×2, Demographic ×2, Strategy ×2, Group ×2, Instruction ×2, Tool ×2, Skill ×2, Framework ×2, Document ×2, academic_institution ×2, research_group ×2, field_theory_model ×2, physical_quantity ×2, Field ×4, Application ×4, Phenomenon ×4, particle ×4, celestial_body ×4, mathematical_method ×4, Book ×4, mathematical concept ×4, Nutrient ×4, Demographic Group ×4, Process ×4, Time Period ×4, Geographical Feature ×4, Treatment ×4, geographic_location ×6, physical_property ×6, mathematical_parameter ×6, mathematical object ×6, Cellular Structure ×6, Molecule ×6, Chemical Element ×6, Food ×6, Data Visualization ×6, Attribute ×6, theoretical_physics_concept ×6, Concept ×8, Product ×8, astrophysical_phenomenon ×8, scientific_instrument ×8, mathematical_matrix ×8, Protein ×8, Value ×8.
Unseen in train, present in test, entities (per mention): Design Principle ×14, UI Element ×12, Visual Element ×12, 3D Shape ×12, Variable ×10, Visual Indicator ×8, Geometric Transformation ×8, Keyboard Shortcut ×6, statistical concept ×6, statistical measure ×6, statistical method ×6, statistical value ×6, statistical parameter ×6, Geometric Element ×6, 2D Shape ×6, Error Type ×4, Genre ×4, Usage ×4, physical object ×4, online community ×4, training_objective ×4, computational_resource ×4, Configuration File ×2, Software Extension ×2, Hardware Component ×2, Debugging Concept ×2, Tradition ×2, Artwork ×2, Assessment ×2, Output ×2, Biological Structure ×2, Intervention ×2, chemical property ×2, statistical theorem ×2, website ×2, book ×2, Geometric Concept ×2, Mathematical System ×2, Mathematical Formula ×2, Software/Game ×2, Educational Standard ×2, Textbook ×2.
Thin support in test, entities (per mention): 72 label(s) under 10 gold. Programming Language ×2, Configuration File ×2, Software Extension ×2, Hardware Component ×2, Debugging Concept ×2, Practice ×2, Work ×2, Tradition ×2, Region ×2, Technique ×2, Artwork ×2, Assessment ×2, Media ×2, Course ×2, Output ×2, Organ ×2, Biological Entity ×2, Model ×2, Chemical ×2, Biological Structure ×2, Cell Type ×2, Publication ×2, Disease ×2, Population ×2, Behavior ×2, Intervention ×2, Outcome ×2, chemical property ×2, statistical theorem ×2, website ×2, book ×2, Geometric Concept ×2, Mathematical System ×2, Mathematical Formula ×2, Software/Game ×2, Website ×2, Educational Standard ×2, Textbook ×2, Building ×2, Error Type ×4, Genre ×4, Art Style ×4, Book ×4, Usage ×4, Time Period ×4, Profession ×4, Technology ×4, physical object ×4, online community ×4, Material ×4, training_objective ×4, computational_resource ×4, model_component ×4, Software ×6, Keyboard Shortcut ×6, Text ×6, Method ×6, Organization ×6, statistical concept ×6, statistical measure ×6, statistical method ×6, statistical value ×6, statistical parameter ×6, organization ×6, Geometric Element ×6, 2D Shape ×6, Measurement ×6, Visual Indicator ×8, Tool ×8, Geometric Transformation ×8, dataset ×8, researcher_group ×8.

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
| Person | 836 | 14.93% |
| Location | 536 | 9.57% |
| Organization | 388 | 6.93% |
| Concept | 154 | 2.75% |
| Group | 104 | 1.86% |
| Event | 90 | 1.61% |
| Product | 66 | 1.18% |
| Law | 62 | 1.11% |
| Ethnic Group | 58 | 1.04% |
| researcher | 56 | 1.00% |
| Substance | 54 | 0.96% |
| model | 54 | 0.96% |
| organization | 52 | 0.93% |
| mathematician | 48 | 0.86% |
| dataset | 48 | 0.86% |
| Document | 48 | 0.86% |
| Publication | 44 | 0.79% |
| Object | 42 | 0.75% |
| Animal | 38 | 0.68% |
| Date | 38 | 0.68% |
| algorithm | 38 | 0.68% |
| metric | 30 | 0.54% |
| Species | 30 | 0.54% |
| Material | 30 | 0.54% |
| Work | 30 | 0.54% |
| Medical Condition | 28 | 0.50% |
| Medication | 28 | 0.50% |
| Software | 28 | 0.50% |
| Phylum | 28 | 0.50% |
| Tribe | 28 | 0.50% |
| City | 26 | 0.46% |
| Country | 26 | 0.46% |
| environment | 26 | 0.46% |
| Body Part | 24 | 0.43% |
| Process | 24 | 0.43% |
| Grammatical Term | 24 | 0.43% |
| Region | 24 | 0.43% |
| person | 24 | 0.43% |
| Phenomenon | 22 | 0.39% |
| Technology | 22 | 0.39% |
| ai_model | 22 | 0.39% |
| Activity | 20 | 0.36% |
| Religion | 20 | 0.36% |
| institution | 20 | 0.36% |
| Tool | 20 | 0.36% |
| location | 20 | 0.36% |
| People | 20 | 0.36% |
| Disease | 20 | 0.36% |
| Archaeological Site | 20 | 0.36% |
| Condition | 20 | 0.36% |
| Spacecraft | 20 | 0.36% |
| quantum_state | 20 | 0.36% |
| Term | 18 | 0.32% |
| Text | 18 | 0.32% |
| Symptom | 18 | 0.32% |
| Book | 18 | 0.32% |
| Component | 18 | 0.32% |
| Method | 18 | 0.32% |
| Mathematical Concept | 18 | 0.32% |
| Language | 16 | 0.29% |
| Role | 16 | 0.29% |
| mathematical object | 16 | 0.29% |
| Military Unit | 16 | 0.29% |
| Anatomy | 16 | 0.29% |
| communication_task | 16 | 0.29% |
| Mathematical Structure | 16 | 0.29% |
| Planet | 16 | 0.29% |
| Field of Study | 14 | 0.25% |
| Structure | 14 | 0.25% |
| mathematical_construct | 14 | 0.25% |
| formal_statement | 14 | 0.25% |
| Social Group | 14 | 0.25% |
| task | 14 | 0.25% |
| architecture_component | 14 | 0.25% |
| Organism | 12 | 0.21% |
| hardware | 12 | 0.21% |
| theorem | 12 | 0.21% |
| method | 12 | 0.21% |
| Metric | 12 | 0.21% |
| Theory | 12 | 0.21% |
| physical_quantity | 12 | 0.21% |
| Project | 12 | 0.21% |
| Facility | 12 | 0.21% |
| group | 12 | 0.21% |
| Time Period | 12 | 0.21% |
| Food | 12 | 0.21% |
| mathematical_result | 12 | 0.21% |
| component | 12 | 0.21% |
| religious_organization | 12 | 0.21% |
| Plant | 12 | 0.21% |
| Taxon | 12 | 0.21% |
| problem_class | 12 | 0.21% |
| constraint_property | 12 | 0.21% |
| Adjective | 12 | 0.21% |
| mathematical_representation | 12 | 0.21% |
| Time Unit | 10 | 0.18% |
| model_architecture | 10 | 0.18% |
| evaluation_metric | 10 | 0.18% |
| Tic Type | 10 | 0.18% |
| Medical Test | 10 | 0.18% |
| Program | 10 | 0.18% |
| Agency | 10 | 0.18% |
| Writing System | 10 | 0.18% |
| software | 10 | 0.18% |
| Literary Work | 10 | 0.18% |
| Culture | 10 | 0.18% |
| Art Form | 10 | 0.18% |
| Natural Feature | 10 | 0.18% |
| Crop | 10 | 0.18% |
| loss_function | 10 | 0.18% |
| Scale | 10 | 0.18% |
| Example | 10 | 0.18% |
| People Group | 10 | 0.18% |
| Programming Language | 10 | 0.18% |
| coding_scheme | 10 | 0.18% |
| theoretical_limit | 10 | 0.18% |
| computational_algorithm | 10 | 0.18% |
| Deity | 8 | 0.14% |
| funding_agency | 8 | 0.14% |
| Field | 8 | 0.14% |
| Word | 8 | 0.14% |
| Technique | 8 | 0.14% |
| mathematical_operator | 8 | 0.14% |
| Government Body | 8 | 0.14% |
| Province | 8 | 0.14% |
| modality | 8 | 0.14% |
| Art Movement | 8 | 0.14% |
| geometric_curve | 8 | 0.14% |
| chemical_compound | 8 | 0.14% |
| software_library | 8 | 0.14% |
| Ecosystem | 8 | 0.14% |
| Time | 8 | 0.14% |
| molecular_property | 8 | 0.14% |
| Vehicle Type | 8 | 0.14% |
| Test | 8 | 0.14% |
| Treatment | 8 | 0.14% |
| concept | 8 | 0.14% |
| Educational Stage | 8 | 0.14% |
| Biological System | 8 | 0.14% |
| Artifact | 6 | 0.11% |
| mathematical_object | 6 | 0.11% |
| technique | 6 | 0.11% |
| Strategy | 6 | 0.11% |
| Equipment | 6 | 0.11% |
| Entity | 6 | 0.11% |
| Measurement | 6 | 0.11% |
| System | 6 | 0.11% |
| theoretical_model | 6 | 0.11% |
| Course | 6 | 0.11% |
| Subject | 6 | 0.11% |
| Material State | 6 | 0.11% |
| algebraic_structure | 6 | 0.11% |
| Office | 6 | 0.11% |
| Legal Document | 6 | 0.11% |
| Regulation | 6 | 0.11% |
| Element | 6 | 0.11% |
| embodiment | 6 | 0.11% |
| Political Entity | 6 | 0.11% |
| decoder_architecture | 6 | 0.11% |
| chemical_group | 6 | 0.11% |
| physical_property | 6 | 0.11% |
| Virus | 6 | 0.11% |
| Specimen | 6 | 0.11% |
| Quantity | 6 | 0.11% |
| Policy | 6 | 0.11% |
| mathematical_concept | 6 | 0.11% |
| Rock Type | 6 | 0.11% |
| Rock Category | 6 | 0.11% |
| Population | 6 | 0.11% |
| Media | 6 | 0.11% |
| molecular_representation | 6 | 0.11% |
| publication | 6 | 0.11% |
| Building | 6 | 0.11% |
| Tissue | 6 | 0.11% |
| Cell Type | 6 | 0.11% |
| Reinforcement Material | 6 | 0.11% |
| Resin Type | 6 | 0.11% |
| Stamp | 6 | 0.11% |
| Value | 6 | 0.11% |
| Stage | 6 | 0.11% |
| Procedure | 6 | 0.11% |
| model_component | 6 | 0.11% |
| Website | 6 | 0.11% |
| Financial Instrument | 6 | 0.11% |
| Rocket | 6 | 0.11% |
| Research Field | 6 | 0.11% |
| nonclassical_phenomenon | 6 | 0.11% |
| Software Component | 6 | 0.11% |
| Operating System | 6 | 0.11% |
| Subphylum | 6 | 0.11% |
| backbone | 6 | 0.11% |
| stellar model | 6 | 0.11% |
| performance_metric | 6 | 0.11% |
| Nation | 6 | 0.11% |
| Moon | 6 | 0.11% |
| pharmaceutical_process | 6 | 0.11% |
| Disease Subtype | 6 | 0.11% |
| Medical Professional | 6 | 0.11% |
| compiler_abstraction | 4 | 0.07% |
| framework | 4 | 0.07% |
| optimization_technique | 4 | 0.07% |
| machine_learning_model | 4 | 0.07% |
| Literature | 4 | 0.07% |
| Occupation | 4 | 0.07% |
| process_phase | 4 | 0.07% |
| baseline_method | 4 | 0.07% |
| method_component | 4 | 0.07% |
| Pest | 4 | 0.07% |
| Scientific Name | 4 | 0.07% |
| Damage | 4 | 0.07% |
| technical_method | 4 | 0.07% |
| Currency | 4 | 0.07% |
| Play Type | 4 | 0.07% |
| Resource | 4 | 0.07% |
| Outcome | 4 | 0.07% |
| Infrastructure | 4 | 0.07% |
| Plant Species | 4 | 0.07% |
| Chemical | 4 | 0.07% |
| physical_system | 4 | 0.07% |
| research_network | 4 | 0.07% |
| scientific_field | 4 | 0.07% |
| Empire | 4 | 0.07% |
| Lesson Title | 4 | 0.07% |
| programming language | 4 | 0.07% |
| Journal | 4 | 0.07% |
| Practice | 4 | 0.07% |
| Section | 4 | 0.07% |
| error_correcting_code | 4 | 0.07% |
| hardware_element | 4 | 0.07% |
| molecular_orbital | 4 | 0.07% |
| researcher_group | 4 | 0.07% |
| institutional_unit | 4 | 0.07% |
| code_parameter | 4 | 0.07% |
| distance_metric | 4 | 0.07% |
| Principle | 4 | 0.07% |
| Model | 4 | 0.07% |
| Movement | 4 | 0.07% |
| Profession | 4 | 0.07% |
| Taxonomic Group | 4 | 0.07% |
| geometric_object | 4 | 0.07% |
| numerical_function | 4 | 0.07% |
| mathematical_structure | 4 | 0.07% |
| Enzyme | 4 | 0.07% |
| Disease Form | 4 | 0.07% |
| Biological Entity | 4 | 0.07% |
| Service | 4 | 0.07% |
| Vehicle Model | 4 | 0.07% |
| religious_event | 4 | 0.07% |
| Site | 4 | 0.07% |
| Feature | 4 | 0.07% |
| Complication | 4 | 0.07% |
| formalism | 4 | 0.07% |
| alignment_technique | 4 | 0.07% |
| adversarial_attack | 4 | 0.07% |
| Geographic Feature | 4 | 0.07% |
| Unit | 4 | 0.07% |
| Statistical Measure | 4 | 0.07% |
| Data Type | 4 | 0.07% |
| Risk Factor | 4 | 0.07% |
| Medical Device | 4 | 0.07% |
| Political Faction | 4 | 0.07% |
| Therapy | 4 | 0.07% |
| Grammatical Concept | 4 | 0.07% |
| Function | 4 | 0.07% |
| geographic_location | 4 | 0.07% |
| Pathogen | 4 | 0.07% |
| numerical_method | 4 | 0.07% |
| scientific_framework | 4 | 0.07% |
| Location/Nation | 4 | 0.07% |
| Biological Group | 4 | 0.07% |
| Geological Period | 4 | 0.07% |
| open cluster | 4 | 0.07% |
| Dataset | 4 | 0.07% |
| Vehicle | 4 | 0.07% |
| information_model | 4 | 0.07% |
| Civilization | 4 | 0.07% |
| Hypothetical Planet | 4 | 0.07% |
| software_system | 4 | 0.07% |
| biological_property | 4 | 0.07% |
| Statistic | 2 | 0.04% |
| Definition | 2 | 0.04% |
| program_construct | 2 | 0.04% |
| compiler_design | 2 | 0.04% |
| software_platform | 2 | 0.04% |
| Philosophy | 2 | 0.04% |
| Textual Section | 2 | 0.04% |
| mathematical_property | 2 | 0.04% |
| scientific_event | 2 | 0.04% |
| software_component | 2 | 0.04% |
| research_task | 2 | 0.04% |
| input | 2 | 0.04% |
| constraint | 2 | 0.04% |
| intermediate_representation | 2 | 0.04% |
| output | 2 | 0.04% |
| actor | 2 | 0.04% |
| Control Method | 2 | 0.04% |
| Natural Enemy | 2 | 0.04% |
| Lure | 2 | 0.04% |
| architectural_component | 2 | 0.04% |
| Classification | 2 | 0.04% |
| Hazard | 2 | 0.04% |
| Behavior | 2 | 0.04% |
| Acronym | 2 | 0.04% |
| Organization/Person | 2 | 0.04% |
| Person/Group | 2 | 0.04% |
| Habitat | 2 | 0.04% |
| Plant Group | 2 | 0.04% |
| Habitat Feature | 2 | 0.04% |
| Land Designation | 2 | 0.04% |
| Designation | 2 | 0.04% |
| Volcano | 2 | 0.04% |
| Name | 2 | 0.04% |
| Event Type | 2 | 0.04% |
| Mythological Figure | 2 | 0.04% |
| Grade Level | 2 | 0.04% |
| Art Style | 2 | 0.04% |
| Accessory | 2 | 0.04% |
| mathematical problem | 2 | 0.04% |
| Virus Variant | 2 | 0.04% |
| Biological Substance | 2 | 0.04% |
| Molecule | 2 | 0.04% |
| Narrative | 2 | 0.04% |
| Invention | 2 | 0.04% |
| topological_space | 2 | 0.04% |
| mathematical_identity | 2 | 0.04% |
| Legal Process | 2 | 0.04% |
| Dynasty | 2 | 0.04% |
| River Name | 2 | 0.04% |
| River | 2 | 0.04% |
| Society | 2 | 0.04% |
| Physical System | 2 | 0.04% |
| Physical Quantity | 2 | 0.04% |
| Force | 2 | 0.04% |
| task_category | 2 | 0.04% |
| capability | 2 | 0.04% |
| Film | 2 | 0.04% |
| error_type | 2 | 0.04% |
| element | 2 | 0.04% |
| computational_method | 2 | 0.04% |
| software_package | 2 | 0.04% |
| Regulatory Body | 2 | 0.04% |
| Demographic Group | 2 | 0.04% |
| binary_code | 2 | 0.04% |
| proof_method | 2 | 0.04% |
| mathematical_function | 2 | 0.04% |
| programming_language | 2 | 0.04% |
| Cultural Artifact | 2 | 0.04% |
| Legal Concept | 2 | 0.04% |
| Economic System | 2 | 0.04% |
| Identity | 2 | 0.04% |
| Species Group | 2 | 0.04% |
| Energy Source | 2 | 0.04% |
| publication_source | 2 | 0.04% |
| Natural Disaster | 2 | 0.04% |
| Factor | 2 | 0.04% |
| Historical Period | 2 | 0.04% |
| Medium | 2 | 0.04% |
| Substance Category | 2 | 0.04% |
| administrative_body | 2 | 0.04% |
| educational_institution | 2 | 0.04% |
| award | 2 | 0.04% |
| Style | 2 | 0.04% |
| Organ | 2 | 0.04% |
| Tissue Classification | 2 | 0.04% |
| Anatomical Space | 2 | 0.04% |
| Body Fluid | 2 | 0.04% |
| Material Category | 2 | 0.04% |
| Material Component | 2 | 0.04% |
| Diet | 2 | 0.04% |
| Ingredient | 2 | 0.04% |
| Stamp Version | 2 | 0.04% |
| training_method | 2 | 0.04% |
| mathematical_framework | 2 | 0.04% |
| data_representation | 2 | 0.04% |
| data_format | 2 | 0.04% |
| data_structure | 2 | 0.04% |
| publication_venue | 2 | 0.04% |
| complexity_class | 2 | 0.04% |
| academic_field | 2 | 0.04% |
| Level | 2 | 0.04% |
| Alliance | 2 | 0.04% |
| Government System | 2 | 0.04% |
| evaluation_method | 2 | 0.04% |
| Language Family | 2 | 0.04% |
| Language Group | 2 | 0.04% |
| Commodity | 2 | 0.04% |
| Industry | 2 | 0.04% |
| Disorder | 2 | 0.04% |
| Disorder Group | 2 | 0.04% |
| Demographic | 2 | 0.04% |
| Cause | 2 | 0.04% |
| Toxin | 2 | 0.04% |
| Linguistic Concept | 2 | 0.04% |
| Platform | 2 | 0.04% |
| Legislation | 2 | 0.04% |
| Rocket Stage | 2 | 0.04% |
| Motor Skill | 2 | 0.04% |
| academic_institution | 2 | 0.04% |
| experimental_technique | 2 | 0.04% |
| optical_process | 2 | 0.04% |
| Character | 2 | 0.04% |
| Category | 2 | 0.04% |
| software_code | 2 | 0.04% |
| astrophysical_object | 2 | 0.04% |
| Scripture Reference | 2 | 0.04% |
| Location/Structure | 2 | 0.04% |
| Organization/Website | 2 | 0.04% |
| Scientific Concept | 2 | 0.04% |
| Scientific Instrument | 2 | 0.04% |
| Symbol/Object | 2 | 0.04% |
| Person/Entity | 2 | 0.04% |
| Scripture | 2 | 0.04% |
| Object/Concept | 2 | 0.04% |
| Location/Concept | 2 | 0.04% |
| Standard | 2 | 0.04% |
| Software Version | 2 | 0.04% |
| Database | 2 | 0.04% |
| Genetic Element | 2 | 0.04% |
| Disease Category | 2 | 0.04% |
| Clinical Syndrome | 2 | 0.04% |
| Clinical Sign | 2 | 0.04% |
| Drug | 2 | 0.04% |
| Medical Intervention | 2 | 0.04% |
| telescope | 2 | 0.04% |
| instrument | 2 | 0.04% |
| project | 2 | 0.04% |
| Nationality | 2 | 0.04% |
| Mathematical Result | 2 | 0.04% |
| Mathematical Theorem | 2 | 0.04% |
| Symbol | 2 | 0.04% |
| Publication Issue | 2 | 0.04% |
| Celestial System | 2 | 0.04% |
| Religious Text | 2 | 0.04% |
| Star | 2 | 0.04% |
| Scientific Theory | 2 | 0.04% |
| Celestial Object | 2 | 0.04% |
| representation | 2 | 0.04% |
| Skill Category | 2 | 0.04% |
| Protein | 2 | 0.04% |
| Community Group | 2 | 0.04% |

#### val

| label | count | share |
| --- | --- | --- |
| Location | 56 | 9.06% |
| Person | 48 | 7.77% |
| Organization | 44 | 7.12% |
| person | 26 | 4.21% |
| Technology | 14 | 2.27% |
| Disease Subtype | 14 | 2.27% |
| Cognitive Process | 14 | 2.27% |
| physical_process | 14 | 2.27% |
| researcher | 12 | 1.94% |
| string_theory_entity | 12 | 1.94% |
| Device | 10 | 1.62% |
| group | 10 | 1.62% |
| formal_result | 10 | 1.62% |
| Chemical | 10 | 1.62% |
| Organism | 10 | 1.62% |
| Disease | 10 | 1.62% |
| Medical Procedure | 10 | 1.62% |
| Condition | 10 | 1.62% |
| Skill Area | 10 | 1.62% |
| Concept | 8 | 1.29% |
| Product | 8 | 1.29% |
| astrophysical_phenomenon | 8 | 1.29% |
| scientific_instrument | 8 | 1.29% |
| mathematical_matrix | 8 | 1.29% |
| Protein | 8 | 1.29% |
| Value | 8 | 1.29% |
| geographic_location | 6 | 0.97% |
| physical_property | 6 | 0.97% |
| mathematical_parameter | 6 | 0.97% |
| mathematical object | 6 | 0.97% |
| Cellular Structure | 6 | 0.97% |
| Molecule | 6 | 0.97% |
| Chemical Element | 6 | 0.97% |
| Food | 6 | 0.97% |
| Data Visualization | 6 | 0.97% |
| Attribute | 6 | 0.97% |
| theoretical_physics_concept | 6 | 0.97% |
| Field | 4 | 0.65% |
| Application | 4 | 0.65% |
| Phenomenon | 4 | 0.65% |
| particle | 4 | 0.65% |
| celestial_body | 4 | 0.65% |
| mathematical_method | 4 | 0.65% |
| Book | 4 | 0.65% |
| mathematical concept | 4 | 0.65% |
| Nutrient | 4 | 0.65% |
| Demographic Group | 4 | 0.65% |
| Process | 4 | 0.65% |
| Time Period | 4 | 0.65% |
| Geographical Feature | 4 | 0.65% |
| Treatment | 4 | 0.65% |
| Material | 2 | 0.32% |
| Physics | 2 | 0.32% |
| experiment | 2 | 0.32% |
| spacecraft | 2 | 0.32% |
| publication | 2 | 0.32% |
| Pollutant Category | 2 | 0.32% |
| Program | 2 | 0.32% |
| Disorder | 2 | 0.32% |
| Syndrome | 2 | 0.32% |
| Symptom | 2 | 0.32% |
| Publication | 2 | 0.32% |
| Test | 2 | 0.32% |
| Cell Type | 2 | 0.32% |
| Anatomy | 2 | 0.32% |
| mathematical property | 2 | 0.32% |
| Physical Force | 2 | 0.32% |
| Organ | 2 | 0.32% |
| Medical Condition | 2 | 0.32% |
| Biological Unit | 2 | 0.32% |
| Award | 2 | 0.32% |
| Cellular Component | 2 | 0.32% |
| Biological Process | 2 | 0.32% |
| Biological System | 2 | 0.32% |
| Biological Entity | 2 | 0.32% |
| Cause | 2 | 0.32% |
| Category | 2 | 0.32% |
| Factor | 2 | 0.32% |
| Structure | 2 | 0.32% |
| Agent | 2 | 0.32% |
| Substance | 2 | 0.32% |
| Standard | 2 | 0.32% |
| Climate Metric | 2 | 0.32% |
| Atmospheric Phenomenon | 2 | 0.32% |
| Event | 2 | 0.32% |
| Season | 2 | 0.32% |
| Literary Work | 2 | 0.32% |
| Demographic | 2 | 0.32% |
| Strategy | 2 | 0.32% |
| Group | 2 | 0.32% |
| Instruction | 2 | 0.32% |
| Tool | 2 | 0.32% |
| Skill | 2 | 0.32% |
| Framework | 2 | 0.32% |
| Document | 2 | 0.32% |
| academic_institution | 2 | 0.32% |
| research_group | 2 | 0.32% |
| field_theory_model | 2 | 0.32% |
| physical_quantity | 2 | 0.32% |

#### test

| label | count | share |
| --- | --- | --- |
| Person | 42 | 10.55% |
| Concept | 16 | 4.02% |
| Design Principle | 14 | 3.52% |
| UI Element | 12 | 3.02% |
| Visual Element | 12 | 3.02% |
| Program | 12 | 3.02% |
| 3D Shape | 12 | 3.02% |
| Variable | 10 | 2.51% |
| model | 10 | 2.51% |
| Visual Indicator | 8 | 2.01% |
| Tool | 8 | 2.01% |
| Geometric Transformation | 8 | 2.01% |
| dataset | 8 | 2.01% |
| researcher_group | 8 | 2.01% |
| Software | 6 | 1.51% |
| Keyboard Shortcut | 6 | 1.51% |
| Text | 6 | 1.51% |
| Method | 6 | 1.51% |
| Organization | 6 | 1.51% |
| statistical concept | 6 | 1.51% |
| statistical measure | 6 | 1.51% |
| statistical method | 6 | 1.51% |
| statistical value | 6 | 1.51% |
| statistical parameter | 6 | 1.51% |
| organization | 6 | 1.51% |
| Geometric Element | 6 | 1.51% |
| 2D Shape | 6 | 1.51% |
| Measurement | 6 | 1.51% |
| Error Type | 4 | 1.01% |
| Genre | 4 | 1.01% |
| Art Style | 4 | 1.01% |
| Book | 4 | 1.01% |
| Usage | 4 | 1.01% |
| Time Period | 4 | 1.01% |
| Profession | 4 | 1.01% |
| Technology | 4 | 1.01% |
| physical object | 4 | 1.01% |
| online community | 4 | 1.01% |
| Material | 4 | 1.01% |
| training_objective | 4 | 1.01% |
| computational_resource | 4 | 1.01% |
| model_component | 4 | 1.01% |
| Programming Language | 2 | 0.50% |
| Configuration File | 2 | 0.50% |
| Software Extension | 2 | 0.50% |
| Hardware Component | 2 | 0.50% |
| Debugging Concept | 2 | 0.50% |
| Practice | 2 | 0.50% |
| Work | 2 | 0.50% |
| Tradition | 2 | 0.50% |
| Region | 2 | 0.50% |
| Technique | 2 | 0.50% |
| Artwork | 2 | 0.50% |
| Assessment | 2 | 0.50% |
| Media | 2 | 0.50% |
| Course | 2 | 0.50% |
| Output | 2 | 0.50% |
| Organ | 2 | 0.50% |
| Biological Entity | 2 | 0.50% |
| Model | 2 | 0.50% |
| Chemical | 2 | 0.50% |
| Biological Structure | 2 | 0.50% |
| Cell Type | 2 | 0.50% |
| Publication | 2 | 0.50% |
| Disease | 2 | 0.50% |
| Population | 2 | 0.50% |
| Behavior | 2 | 0.50% |
| Intervention | 2 | 0.50% |
| Outcome | 2 | 0.50% |
| chemical property | 2 | 0.50% |
| statistical theorem | 2 | 0.50% |
| website | 2 | 0.50% |
| book | 2 | 0.50% |
| Geometric Concept | 2 | 0.50% |
| Mathematical System | 2 | 0.50% |
| Mathematical Formula | 2 | 0.50% |
| Software/Game | 2 | 0.50% |
| Website | 2 | 0.50% |
| Educational Standard | 2 | 0.50% |
| Textbook | 2 | 0.50% |
| Building | 2 | 0.50% |

### Relations

None.

### Classifications

None.

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| record | status | 1,014 | 13.08% |
| record | location | 782 | 10.09% |
| record | name | 549 | 7.08% |
| record | address | 409 | 5.27% |
| record | title | 298 | 3.84% |
| record | description | 282 | 3.64% |
| record | date | 225 | 2.90% |
| record | price | 217 | 2.80% |
| record | city | 183 | 2.36% |
| record | patient_name | 161 | 2.08% |
| record | rating | 158 | 2.04% |
| record | amount | 148 | 1.91% |
| record | state | 147 | 1.90% |
| record | department | 146 | 1.88% |
| record | author | 142 | 1.83% |
| record | patient_id | 127 | 1.64% |
| record | product_id | 121 | 1.56% |
| record | age | 116 | 1.50% |
| record | employee_id | 112 | 1.44% |
| record | company | 108 | 1.39% |
| record | venue | 105 | 1.35% |
| record | claim_id | 96 | 1.24% |
| record | ticket_id | 89 | 1.15% |
| record | category | 84 | 1.08% |
| record | review_id | 84 | 1.08% |
| record | year | 80 | 1.03% |
| record | product_name | 80 | 1.03% |
| record | manufacturer | 79 | 1.02% |
| record | country | 79 | 1.02% |
| record | reviewer | 75 | 0.97% |
| record | score | 73 | 0.94% |
| record | transaction_id | 70 | 0.90% |
| record | event_id | 69 | 0.89% |
| record | person | 68 | 0.88% |
| record | head | 66 | 0.85% |
| record | tail | 66 | 0.85% |
| record | source | 65 | 0.84% |
| record | home_team | 62 | 0.80% |
| record | policy_id | 59 | 0.76% |
| record | dosage | 59 | 0.76% |
| record | time | 56 | 0.72% |
| record | customer_name | 55 | 0.71% |
| record | funding_source | 54 | 0.70% |
| record | region | 53 | 0.68% |
| record | away_team | 52 | 0.67% |
| record | role | 51 | 0.66% |
| record | severity | 51 | 0.66% |
| record | company_name | 49 | 0.63% |
| record | sku | 49 | 0.63% |
| record | patient_address | 46 | 0.59% |
| record | review_date | 46 | 0.59% |
| record | symptoms | 45 | 0.58% |
| record | route | 44 | 0.57% |
| record | notes | 44 | 0.57% |
| record | organization | 44 | 0.57% |
| record | recipient | 43 | 0.55% |
| record | serial_number | 43 | 0.55% |
| record | medication | 43 | 0.55% |
| record | quantity | 42 | 0.54% |
| record | features | 42 | 0.54% |
| record | sender | 42 | 0.54% |
| record | organizations | 41 | 0.53% |
| record | funding_amount | 41 | 0.53% |
| record | brand | 41 | 0.53% |
| record | policyholder | 41 | 0.53% |
| record | institution | 40 | 0.52% |
| record | asteroid.name | 40 | 0.52% |
| record | provider | 39 | 0.50% |
| record | diagnosis_code | 39 | 0.50% |
| record | user | 38 | 0.49% |
| record | policy_number | 38 | 0.49% |
| record | origin | 38 | 0.49% |
| record | email | 37 | 0.48% |
| record | discovery.observatory.name | 37 | 0.48% |
| record | mrn | 37 | 0.48% |
| record | comments | 37 | 0.48% |
| record | device | 36 | 0.46% |
| record | full_name | 36 | 0.46% |
| record | designation | 36 | 0.46% |
| record | event | 36 | 0.46% |
| record | patient_age | 36 | 0.46% |
| record | reviewer_name | 35 | 0.45% |
| record | frequency | 35 | 0.45% |
| record | project_id | 35 | 0.45% |
| record | duration | 35 | 0.45% |
| record | report_id | 35 | 0.45% |
| record | name: Full name of the patient | 35 | 0.45% |
| record | topic | 35 | 0.45% |
| record | player_name | 35 | 0.45% |
| record | target_audience | 35 | 0.45% |
| record | discovery.date | 34 | 0.44% |
| record | destination | 34 | 0.44% |
| record | tags | 33 | 0.43% |
| record | zip_code | 33 | 0.43% |
| record | medications | 33 | 0.43% |
| record | revenue | 32 | 0.41% |
| record | discovery.discoverer | 32 | 0.41% |
| record | locations | 32 | 0.41% |
| record | person.name | 32 | 0.41% |
| record | color | 31 | 0.40% |
| record | product | 31 | 0.40% |
| record | instructor | 31 | 0.40% |
| record | job_title | 31 | 0.40% |
| record | start_date | 31 | 0.40% |
| record | battery_life | 30 | 0.39% |
| record | subject | 30 | 0.39% |
| record | organizer | 30 | 0.39% |
| record | id | 30 | 0.39% |
| record | outcome | 30 | 0.39% |
| record | acquirer | 29 | 0.37% |
| record | salary | 29 | 0.37% |
| record | founding_year | 29 | 0.37% |
| record | publication_date | 29 | 0.37% |
| record | diagnosis | 29 | 0.37% |
| record | asteroid.designation | 28 | 0.36% |
| record | manager | 28 | 0.36% |
| record | orbital_characteristics.classification | 28 | 0.36% |
| record | discovery.observatory.location.country | 28 | 0.36% |
| record | gender | 27 | 0.35% |
| record | owner | 27 | 0.35% |
| record | teams | 27 | 0.35% |
| record | jurisdiction | 27 | 0.35% |
| record | physical_characteristics.observations.0.telescope | 27 | 0.35% |
| record | room | 27 | 0.35% |
| record | affiliation | 26 | 0.34% |
| record | salary_range | 26 | 0.34% |
| record | dates | 26 | 0.34% |
| record | district | 26 | 0.34% |
| record | word_count | 25 | 0.32% |
| record | model | 25 | 0.32% |
| record | alternative_names | 25 | 0.32% |
| record | diameter.unit | 25 | 0.32% |
| record | return_id | 25 | 0.32% |
| record | likes | 25 | 0.32% |
| record | shares | 25 | 0.32% |
| record | passenger_name | 25 | 0.32% |
| record | asteroid.discovery.discoverer | 25 | 0.32% |
| record | condition | 24 | 0.31% |
| record | assists | 24 | 0.31% |
| record | physical_characteristics.observations.0.publication_year | 24 | 0.31% |
| record | risk_score | 24 | 0.31% |
| record | policyholder_name | 24 | 0.31% |
| record | submitter | 24 | 0.31% |
| record | type | 24 | 0.31% |
| record | points | 24 | 0.31% |
| record | lead_investor | 24 | 0.31% |
| record | dimensions | 24 | 0.31% |
| record | cost | 23 | 0.30% |
| record | sector | 23 | 0.30% |
| record | lead_researcher | 23 | 0.30% |
| record | issue | 23 | 0.30% |
| record | video_id | 23 | 0.30% |
| record | customer_address | 23 | 0.30% |
| record | orbital_characteristics.perihelion.unit | 23 | 0.30% |
| record | asteroid.type | 23 | 0.30% |
| record | value | 23 | 0.30% |
| record | final_score | 23 | 0.30% |
| record | study_id | 23 | 0.30% |
| record | material | 22 | 0.28% |
| record | filing_id | 22 | 0.28% |
| record | hotel_name | 22 | 0.28% |
| record | prescription_id | 22 | 0.28% |
| record | effective_date | 22 | 0.28% |
| record | blood_pressure | 22 | 0.28% |
| record | themes | 22 | 0.28% |
| record | status: Current state of the ticket | 22 | 0.28% |
| record | refund_amount | 22 | 0.28% |
| record | course_id | 22 | 0.28% |
| record | orbital_characteristics.semi_major_axis.unit | 22 | 0.28% |
| record | location.city | 21 | 0.27% |
| record | keywords | 21 | 0.27% |
| record | zip | 21 | 0.27% |
| record | product.name | 21 | 0.27% |
| record | filing_date | 21 | 0.27% |
| record | user_name | 21 | 0.27% |
| record | authors | 21 | 0.27% |
| record | entity | 20 | 0.26% |
| record | url | 20 | 0.26% |
| record | patient_id: Unique identifier for the patient | 20 | 0.26% |
| record | heart_rate | 20 | 0.26% |
| record | date: Date when the transaction occurred | 20 | 0.26% |
| record | content | 20 | 0.26% |
| record | program_name | 20 | 0.26% |
| record | sentiment | 20 | 0.26% |
| record | user_id | 20 | 0.26% |
| record | opponent | 20 | 0.26% |
| record | people | 20 | 0.26% |
| record | appointment_id | 20 | 0.26% |
| record | discovery.observatory.location.county | 20 | 0.26% |
| record | sponsor | 20 | 0.26% |
| record | ticker | 20 | 0.26% |
| record | target | 20 | 0.26% |
| record | last_updated | 20 | 0.26% |
| record | physical_characteristics.observations.0.year | 19 | 0.25% |
| record | bracket_id | 19 | 0.25% |
| record | findings | 19 | 0.25% |
| record | priority | 19 | 0.25% |
| record | entity_name | 19 | 0.25% |
| record | scientific_name | 19 | 0.25% |
| record | visit_type | 19 | 0.25% |
| record | purpose | 19 | 0.25% |
| record | review_status | 18 | 0.23% |
| record | event.date | 18 | 0.23% |
| record | patent_number | 18 | 0.23% |
| record | match | 18 | 0.23% |
| record | indication | 18 | 0.23% |
| record | orbital_characteristics.inclination.unit | 18 | 0.23% |
| record | net_profit | 18 | 0.23% |
| record | funding | 18 | 0.23% |
| record | copyright | 18 | 0.23% |
| record | product.description | 18 | 0.23% |
| record | fee | 18 | 0.23% |
| record | vendor_name | 18 | 0.23% |
| record | invoice_id | 18 | 0.23% |
| record | release_date | 18 | 0.23% |
| record | credits | 18 | 0.23% |
| record | rebounds | 18 | 0.23% |
| record | family | 18 | 0.23% |
| record | organization.name | 18 | 0.23% |
| record | submission_id | 17 | 0.22% |
| record | exhibitor_id | 17 | 0.22% |
| record | requisition_id | 17 | 0.22% |
| record | performance_rating | 17 | 0.22% |
| record | claim_id: Unique identifier for the insurance claim | 17 | 0.22% |
| record | topics | 17 | 0.22% |
| record | replies | 17 | 0.22% |
| record | availability | 17 | 0.22% |
| record | team | 17 | 0.22% |
| record | resolution_time | 17 | 0.22% |
| record | claim_amount | 17 | 0.22% |
| record | treatment | 17 | 0.22% |
| record | amount: Monetary value of the transaction | 17 | 0.22% |
| record | position | 17 | 0.22% |
| record | flight_id | 17 | 0.22% |
| record | end_date | 17 | 0.22% |
| record | asteroid.discovery.date | 16 | 0.21% |
| record | submission_date | 16 | 0.21% |
| record | reporter | 16 | 0.21% |
| record | weight | 16 | 0.21% |
| record | genre | 16 | 0.21% |
| record | location.region | 16 | 0.21% |
| record | retweets | 16 | 0.21% |
| record | discovery_year | 16 | 0.21% |
| record | dose | 16 | 0.21% |
| record | referee | 16 | 0.21% |
| record | physical_characteristics.observations.0.telescope.name | 16 | 0.21% |
| record | key_player | 16 | 0.21% |
| record | goals | 16 | 0.21% |
| record | order_id | 16 | 0.21% |
| record | registration_fee | 16 | 0.21% |
| record | player_id | 16 | 0.21% |
| record | launch_date | 16 | 0.21% |
| record | review_length | 16 | 0.21% |
| record | handle | 16 | 0.21% |
| record | winner | 15 | 0.19% |
| record | subreddit | 15 | 0.19% |
| record | status: Current status of the transaction | 15 | 0.19% |
| record | project_name | 15 | 0.19% |
| record | timestamp | 15 | 0.19% |
| record | temperature | 15 | 0.19% |
| record | device_id | 15 | 0.19% |
| record | drug_name | 15 | 0.19% |
| record | heart_rate: Heart rate in beats per minute | 15 | 0.19% |
| record | temperature: Body temperature in degrees Fahrenheit | 15 | 0.19% |
| record | publication | 15 | 0.19% |
| record | hire_date | 15 | 0.19% |
| record | normal_range | 15 | 0.19% |
| record | delay_minutes | 15 | 0.19% |
| record | contact_name | 15 | 0.19% |
| record | patient_gender | 15 | 0.19% |
| record | insurer | 15 | 0.19% |
| record | candidate_name | 15 | 0.19% |
| record | capacity | 15 | 0.19% |
| record | reason | 15 | 0.19% |
| record | post_id | 15 | 0.19% |
| record | attendees | 15 | 0.19% |
| record | upvotes | 14 | 0.18% |
| record | founder | 14 | 0.18% |
| record | version | 14 | 0.18% |
| record | warranty_period | 14 | 0.18% |
| record | clause_id | 14 | 0.18% |
| record | agent | 14 | 0.18% |
| record | drug | 14 | 0.18% |
| record | technology | 14 | 0.18% |
| record | code | 14 | 0.18% |
| record | asteroid_name | 14 | 0.18% |
| record | match_id | 14 | 0.18% |
| record | policy_type | 14 | 0.18% |
| record | channel_name | 14 | 0.18% |
| record | startup_name | 14 | 0.18% |
| record | recommendation | 14 | 0.18% |
| record | delay_reason | 14 | 0.18% |
| record | discovery.observatory.location.state | 14 | 0.18% |
| record | rate | 14 | 0.18% |
| record | merchant | 14 | 0.18% |
| record | cause | 14 | 0.18% |
| record | discovery.observatory | 14 | 0.18% |
| record | transaction_type | 14 | 0.18% |
| record | vendor | 14 | 0.18% |
| record | surface | 14 | 0.18% |
| record | donor_name | 14 | 0.18% |
| record | chemical_name | 13 | 0.17% |
| record | book_title | 13 | 0.17% |
| record | premium | 13 | 0.17% |
| record | sample_size | 13 | 0.17% |
| record | hotel.name | 13 | 0.17% |
| record | parties | 13 | 0.17% |
| record | categories | 13 | 0.17% |
| record | recipient_name | 13 | 0.17% |
| record | ticket_id: Unique identifier for the support ticket | 13 | 0.17% |
| record | municipality.name | 13 | 0.17% |
| record | orbital_characteristics.inclination | 13 | 0.17% |
| record | education | 13 | 0.17% |
| record | discovery_date | 13 | 0.17% |
| record | patient_id: Unique identifier for each patient | 13 | 0.17% |
| record | format | 13 | 0.17% |
| record | resources | 13 | 0.17% |
| record | transaction_id: Unique identifier for each transaction entry | 13 | 0.17% |
| record | event_type | 13 | 0.17% |
| record | procedure | 13 | 0.17% |
| record | ticket_price | 13 | 0.17% |
| record | creator | 13 | 0.17% |
| record | review_id: Unique identifier for the review entry | 13 | 0.17% |
| record | product.type | 13 | 0.17% |
| record | network_id | 13 | 0.17% |
| record | device_name | 13 | 0.17% |
| record | match_code | 13 | 0.17% |
| record | products | 13 | 0.17% |
| record | from_club | 13 | 0.17% |
| record | to_club | 13 | 0.17% |
| record | definition | 13 | 0.17% |
| record | term_length | 13 | 0.17% |
| record | hotel.location.city | 13 | 0.17% |
| record | provider_name | 13 | 0.17% |
| record | conference_id | 13 | 0.17% |
| record | physical_characteristics.observations.0.telescope.type | 13 | 0.17% |
| record | audit_id | 13 | 0.17% |
| record | investor | 13 | 0.17% |
| record | exchange | 12 | 0.15% |
| record | event.location | 12 | 0.15% |
| record | phone | 12 | 0.15% |
| record | session_id | 12 | 0.15% |
| record | concepts | 12 | 0.15% |
| record | investors | 12 | 0.15% |
| record | festival_id | 12 | 0.15% |
| record | document_type | 12 | 0.15% |
| record | tenant | 12 | 0.15% |
| record | risk_category | 12 | 0.15% |
| record | lease_term | 12 | 0.15% |
| record | service_type | 12 | 0.15% |
| record | briefing_id | 12 | 0.15% |
| record | pathway_id | 12 | 0.15% |
| record | nationality | 12 | 0.15% |
| record | cpt_code | 12 | 0.15% |
| record | project_status | 12 | 0.15% |
| record | device_model | 12 | 0.15% |
| record | flight_number | 12 | 0.15% |
| record | lab_id | 12 | 0.15% |
| record | key_challenge | 12 | 0.15% |
| record | asteroid.discovery.observatory.name | 12 | 0.15% |
| record | product.brand | 12 | 0.15% |
| record | report_id: Unique identifier for the phishing incident | 12 | 0.15% |
| record | current_status | 12 | 0.15% |
| record | property_address | 12 | 0.15% |
| record | enrollment_date | 12 | 0.15% |
| record | call_id | 12 | 0.15% |
| record | coverage_amount | 12 | 0.15% |
| record | shipping_method | 12 | 0.15% |
| record | text | 12 | 0.15% |
| record | dateofservice | 12 | 0.15% |
| record | unit_id | 12 | 0.15% |
| record | medal | 12 | 0.15% |
| record | icd10code | 12 | 0.15% |
| record | campaign_id | 12 | 0.15% |
| record | steals | 12 | 0.15% |
| record | number | 12 | 0.15% |
| record | assigned_agent | 11 | 0.14% |
| record | author.name | 11 | 0.14% |
| record | tenant_name | 11 | 0.14% |
| record | approval_year | 11 | 0.14% |
| record | plaintiff_address | 11 | 0.14% |
| record | case_type | 11 | 0.14% |
| record | issue_description | 11 | 0.14% |
| record | post_id: Unique identifier for the social media post | 11 | 0.14% |
| record | match_type | 11 | 0.14% |
| record | status: Current operational status of the product | 11 | 0.14% |
| record | return_reason | 11 | 0.14% |
| record | currency | 11 | 0.14% |
| record | school_name | 11 | 0.14% |
| record | bedrooms | 11 | 0.14% |
| record | film_title | 11 | 0.14% |
| record | property | 11 | 0.14% |
| record | asset | 11 | 0.14% |
| record | asteroid_number | 11 | 0.14% |
| record | species.scientific_name | 11 | 0.14% |
| record | province | 11 | 0.14% |
| record | competition | 11 | 0.14% |
| record | root_cause | 11 | 0.14% |
| record | units | 11 | 0.14% |
| record | insurance | 11 | 0.14% |
| record | website | 11 | 0.14% |
| record | ndc_code | 11 | 0.14% |
| record | event_code | 11 | 0.14% |
| record | person.birth_date | 11 | 0.14% |
| record | vaccine_type | 11 | 0.14% |
| record | issue_date | 11 | 0.14% |
| record | listing_id | 11 | 0.14% |
| record | reviewer_location | 11 | 0.14% |
| record | review_score | 11 | 0.14% |
| record | person.occupation | 11 | 0.14% |
| record | user_location | 11 | 0.14% |
| record | guideline_id | 11 | 0.14% |
| record | amenities | 11 | 0.14% |
| record | ndc | 11 | 0.14% |
| record | asteroid.location | 11 | 0.14% |
| record | listing_date | 11 | 0.14% |
| record | agency | 11 | 0.14% |
| record | focus_area | 11 | 0.14% |
| record | headquarters | 11 | 0.14% |
| record | applications | 11 | 0.14% |
| record | offer_id: Unique identifier for each leasing offer | 11 | 0.14% |
| record | transaction_id: Unique identifier for the transaction | 11 | 0.14% |
| record | discovery.observatory.location.prefecture | 11 | 0.14% |
| record | reference_id | 11 | 0.14% |
| record | division | 11 | 0.14% |
| record | orbital_characteristics.semi_major_axis | 11 | 0.14% |
| record | orbital_characteristics.perihelion | 11 | 0.14% |
| record | blocks | 11 | 0.14% |
| record | funding_round | 10 | 0.13% |
| record | medication_name | 10 | 0.13% |
| record | product_code | 10 | 0.13% |
| record | contact_person | 10 | 0.13% |
| record | app_name | 10 | 0.13% |
| record | funding_id | 10 | 0.13% |
| record | subscriber_name | 10 | 0.13% |
| record | confirmation_code | 10 | 0.13% |
| record | client_name | 10 | 0.13% |
| record | opponent_name | 10 | 0.13% |
| record | serial | 10 | 0.13% |
| record | references | 10 | 0.13% |
| record | policy_id: Unique identifier for the insurance policy | 10 | 0.13% |
| record | coverage_type: Type of insurance coverage provided | 10 | 0.13% |
| record | account_id | 10 | 0.13% |
| record | offer_id | 10 | 0.13% |
| record | company_id | 10 | 0.13% |
| record | review_type | 10 | 0.13% |
| record | billed_amount | 10 | 0.13% |
| record | product.features | 10 | 0.13% |
| record | underwriter | 10 | 0.13% |
| record | species.genus | 10 | 0.13% |
| record | address: Street address of the patient | 10 | 0.13% |
| record | return_date | 10 | 0.13% |
| record | risk_rating | 10 | 0.13% |
| record | supervisor | 10 | 0.13% |
| record | claim_id: Unique identifier for each claim entry | 10 | 0.13% |
| record | product_focus | 10 | 0.13% |
| record | airline | 10 | 0.13% |
| record | target_population | 10 | 0.13% |
| record | publisher | 10 | 0.13% |
| record | model_number | 10 | 0.13% |
| record | employee_name | 10 | 0.13% |
| record | latin_name | 10 | 0.13% |
| record | ROOT | 10 | 0.13% |
| record | film_title: Title of the film being reviewed | 10 | 0.13% |
| record | beneficiary | 10 | 0.13% |
| record | vehicle | 10 | 0.13% |
| record | status: Current state of the transaction execution | 10 | 0.13% |
| record | transfer_fee | 10 | 0.13% |
| record | restaurant_name | 10 | 0.13% |
| record | specialization | 10 | 0.13% |
| record | department: Academic department offering the course | 10 | 0.13% |
| record | disclaimer | 10 | 0.13% |
| record | asteroid.discovery.location | 10 | 0.13% |
| record | asteroid.named_after.person.name | 10 | 0.13% |
| record | merchant_name | 10 | 0.13% |
| record | credit_hours | 10 | 0.13% |
| record | compatibility | 10 | 0.13% |
| record | result | 10 | 0.13% |
| record | damage_type | 10 | 0.13% |
| record | university | 10 | 0.13% |
| record | filing_type | 10 | 0.13% |
| record | report_date | 10 | 0.13% |
| record | booth | 10 | 0.13% |
| record | price: Selling price of the product in dollars | 10 | 0.13% |
| record | order | 10 | 0.13% |
| record | game_id | 10 | 0.13% |
| record | program_name: Title of the training program enrolled in | 10 | 0.13% |
| record | manifest_id | 10 | 0.13% |
| record | carrier | 10 | 0.13% |
| record | event.location.city | 9 | 0.12% |
| record | side_effects | 9 | 0.12% |
| record | ceo | 9 | 0.12% |
| record | subscription_id | 9 | 0.12% |
| record | case_id | 9 | 0.12% |
| record | attending_physician | 9 | 0.12% |
| record | vote_id | 9 | 0.12% |
| record | available_units | 9 | 0.12% |
| record | claim_count | 9 | 0.12% |
| record | room_type | 9 | 0.12% |
| record | key_achievements | 9 | 0.12% |
| record | performance_score | 9 | 0.12% |
| record | integration_plan | 9 | 0.12% |
| record | party_name | 9 | 0.12% |
| record | employee_id: Unique identifier assigned to each employee | 9 | 0.12% |
| record | address: Patient's residential address | 9 | 0.12% |
| record | event_type: Category of the event | 9 | 0.12% |
| record | status: Current status of the appointment | 9 | 0.12% |
| record | platform | 9 | 0.12% |
| record | industry | 9 | 0.12% |
| record | orbit.aphelion.unit | 9 | 0.12% |
| record | orbital_characteristics.eccentricity | 9 | 0.12% |
| record | trial_id | 9 | 0.12% |
| record | event.name | 9 | 0.12% |
| record | training_modules | 9 | 0.12% |
| record | office_location | 9 | 0.12% |
| record | user_handle | 9 | 0.12% |
| record | policyholder: Name of the vehicle owner filing the claim | 9 | 0.12% |
| record | vaccine_type: Type of vaccine administered | 9 | 0.12% |
| record | market_cap: Market capitalization in billions | 9 | 0.12% |
| record | claim_date | 9 | 0.12% |
| record | provisional_designation | 9 | 0.12% |
| record | physical_characteristics.observations.0.wavelength | 9 | 0.12% |
| record | note | 9 | 0.12% |
| record | thumbnail | 9 | 0.12% |
| record | status: Current operational status of the device | 9 | 0.12% |
| record | home_team: Team playing at the home venue | 9 | 0.12% |
| record | insurance_provider | 9 | 0.12% |
| record | post_date | 9 | 0.12% |
| record | movie.title | 9 | 0.12% |
| record | places | 9 | 0.12% |
| record | advmod | 9 | 0.12% |
| record | problem_description | 9 | 0.12% |
| record | insurance_plan | 9 | 0.12% |
| record | enrollment | 9 | 0.12% |
| record | review_source | 9 | 0.12% |
| record | office_address | 9 | 0.12% |
| record | software | 9 | 0.12% |
| record | instructor: Name of the faculty member teaching the course | 9 | 0.12% |
| record | affiliation.department | 9 | 0.12% |
| record | examples | 9 | 0.12% |
| record | asteroid.discovery_year | 9 | 0.12% |
| record | genus | 9 | 0.12% |
| record | geographic_scope | 9 | 0.12% |
| record | reference_code | 9 | 0.12% |
| record | date_added | 9 | 0.12% |
| record | tournament_name | 9 | 0.12% |
| record | approval_status | 9 | 0.12% |
| record | party | 9 | 0.12% |
| record | publication.year | 9 | 0.12% |
| record | service_description | 9 | 0.12% |
| record | location: City and state where the user is based | 9 | 0.12% |
| record | purchase_date | 9 | 0.12% |
| record | respiratory_rate | 9 | 0.12% |
| record | company.name | 9 | 0.12% |
| record | species.description | 9 | 0.12% |
| record | responsible_staff | 9 | 0.12% |
| record | vaccination_id | 9 | 0.12% |
| record | customer_name: Name of the affected customer | 9 | 0.12% |
| record | speaker | 9 | 0.12% |
| record | status: Current product availability status | 9 | 0.12% |
| record | discharge_date | 9 | 0.12% |
| record | user_feedback | 9 | 0.12% |
| record | establishment_year | 9 | 0.12% |
| record | market_cap | 9 | 0.12% |
| record | causes | 9 | 0.12% |
| record | target_market | 9 | 0.12% |
| record | prescriber | 9 | 0.12% |
| record | tournament_code | 8 | 0.10% |
| record | open_date | 8 | 0.10% |
| record | comment_id | 8 | 0.10% |
| record | helpful_votes | 8 | 0.10% |
| record | language | 8 | 0.10% |
| record | sheet_id | 8 | 0.10% |
| record | expiry_date | 8 | 0.10% |
| record | author.role | 8 | 0.10% |
| record | event.location.venue | 8 | 0.10% |
| record | patient_id: Unique identifier for the patient in the record | 8 | 0.10% |
| record | conference | 8 | 0.10% |
| record | eps | 8 | 0.10% |
| record | warranty | 8 | 0.10% |
| record | project_type | 8 | 0.10% |
| record | newsletter_name | 8 | 0.10% |
| record | defendant | 8 | 0.10% |
| record | performance_metric | 8 | 0.10% |
| record | employees | 8 | 0.10% |
| record | family.name | 8 | 0.10% |
| record | ticket_id: Unique identifier for each support ticket | 8 | 0.10% |
| record | publication_id | 8 | 0.10% |
| record | apparent_magnitude.unit | 8 | 0.10% |
| record | funding_amount: Total amount raised in millions | 8 | 0.10% |
| record | location: City and state where the company is headquartered | 8 | 0.10% |
| record | character | 8 | 0.10% |
| record | primary_diagnosis: Main medical condition diagnosed | 8 | 0.10% |
| record | total_amount | 8 | 0.10% |
| record | compliance | 8 | 0.10% |
| record | object_type | 8 | 0.10% |
| record | case_number: Unique identifier for the court case | 8 | 0.10% |
| record | discovery.observatory.location.city | 8 | 0.10% |
| record | characteristics | 8 | 0.10% |
| record | skills | 8 | 0.10% |
| record | asteroid_type | 8 | 0.10% |
| record | incident_id | 8 | 0.10% |
| record | discipline | 8 | 0.10% |
| record | location: City where the treaty was signed | 8 | 0.10% |
| record | campaign_id: Unique identifier for each promotional campaign | 8 | 0.10% |
| record | artist_name | 8 | 0.10% |
| record | departure_time | 8 | 0.10% |
| record | passenger_email | 8 | 0.10% |
| record | visit_reason | 8 | 0.10% |
| record | description: Brief explanation of the transaction purpose | 8 | 0.10% |
| record | exam_date | 8 | 0.10% |
| record | asteroid.discovery.observatory.location.country | 8 | 0.10% |
| record | review_code | 8 | 0.10% |
| record | sender_location | 8 | 0.10% |
| record | media | 8 | 0.10% |
| record | strength | 8 | 0.10% |
| record | required_skills | 8 | 0.10% |
| record | status: Current status of the enrollment | 8 | 0.10% |
| record | distance.unit | 8 | 0.10% |
| record | aux | 8 | 0.10% |
| record | nsubj | 8 | 0.10% |
| record | person.birth_place | 8 | 0.10% |
| record | article.title | 8 | 0.10% |
| record | provider_address | 8 | 0.10% |
| record | defendant: Name of the individual or entity being sued | 8 | 0.10% |
| record | author_handle | 8 | 0.10% |
| record | snapshot_id | 8 | 0.10% |
| record | modality | 8 | 0.10% |
| record | comment | 8 | 0.10% |
| record | sender_name | 8 | 0.10% |
| record | promotion_channels | 8 | 0.10% |
| record | medicalrecordid | 8 | 0.10% |
| record | listing_id: Unique identifier for each property listing | 8 | 0.10% |
| record | recommendations | 8 | 0.10% |
| record | address: Full address of the policyholder | 8 | 0.10% |
| record | place.location.country | 8 | 0.10% |
| record | place.type | 8 | 0.10% |
| record | VBD | 8 | 0.10% |
| record | stars | 8 | 0.10% |
| record | sales_trend | 8 | 0.10% |
| record | player | 8 | 0.10% |
| record | property_type | 8 | 0.10% |
| record | activity | 8 | 0.10% |
| record | asteroid.discovery.observatory.location.state | 8 | 0.10% |
| record | activities | 8 | 0.10% |
| record | change | 8 | 0.10% |
| record | office_code | 8 | 0.10% |
| record | payee | 8 | 0.10% |
| record | primary_diagnosis | 8 | 0.10% |
| record | asteroid.designation_number | 8 | 0.10% |
| record | location.country | 8 | 0.10% |
| record | person.death_date | 8 | 0.10% |
| record | articleTitle | 8 | 0.10% |
| record | status: Current status of the patent | 8 | 0.10% |
| record | unit: Unit of measurement for the result value | 8 | 0.10% |
| record | contributor | 8 | 0.10% |
| record | vote_percentage | 8 | 0.10% |
| record | contact_phone | 8 | 0.10% |
| record | beliefs | 8 | 0.10% |
| record | award | 8 | 0.10% |
| record | location: Geographic location of the tournament | 8 | 0.10% |
| record | clinic_location | 8 | 0.10% |
| record | lead_person | 8 | 0.10% |
| record | event.description | 8 | 0.10% |
| record | registration_date | 8 | 0.10% |
| record | height | 8 | 0.10% |
| record | useoffunds | 8 | 0.10% |
| record | percentage | 8 | 0.10% |
| record | product_id: Unique identifier assigned to the product | 8 | 0.10% |
| record | tenant_name: Name of the leasing company or business | 8 | 0.10% |
| record | open_issues | 8 | 0.10% |
| record | visit_date | 8 | 0.10% |
| record | follow_up | 8 | 0.10% |
| record | publicationDate | 8 | 0.10% |
| record | price: Price of the product in US dollars | 8 | 0.10% |
| record | artist | 8 | 0.10% |
| record | landlord | 8 | 0.10% |
| record | year_in_era | 8 | 0.10% |
| record | origin_club | 8 | 0.10% |
| record | access_type | 8 | 0.10% |
| record | product_name: Name of the launched product | 8 | 0.10% |
| record | proposer | 8 | 0.10% |
| record | document_id | 8 | 0.10% |
| record | test_date: Date the lab test was conducted | 8 | 0.10% |
| record | asteroid.name_latin | 8 | 0.10% |
| record | cas_number | 7 | 0.09% |
| record | plaintiff | 7 | 0.09% |
| record | company_size | 7 | 0.09% |
| record | tournament | 7 | 0.09% |
| record | amount_usd | 7 | 0.09% |
| record | result_status | 7 | 0.09% |
| record | apparent_magnitude.max | 7 | 0.09% |
| record | icd10_code | 7 | 0.09% |
| record | resolution_status | 7 | 0.09% |
| record | risk_classification | 7 | 0.09% |
| record | geographic_region | 7 | 0.09% |
| record | payment_method | 7 | 0.09% |
| record | hazard_class | 7 | 0.09% |
| record | see_also | 7 | 0.09% |
| record | theme | 7 | 0.09% |
| record | donor_name: Full name of the individual or organization making the donation | 7 | 0.09% |
| record | donor_address: Physical address of the donor | 7 | 0.09% |
| record | city: City where patient resides | 7 | 0.09% |
| record | services | 7 | 0.09% |
| record | product_category | 7 | 0.09% |
| record | expression_level | 7 | 0.09% |
| record | location.district | 7 | 0.09% |
| record | price_range.currency | 7 | 0.09% |
| record | location.province | 7 | 0.09% |
| record | location: City and state where the role is based | 7 | 0.09% |
| record | bathrooms | 7 | 0.09% |
| record | discovery.discoverers | 7 | 0.09% |
| record | sensitivity | 7 | 0.09% |
| record | distribution.habitat | 7 | 0.09% |
| record | product_model | 7 | 0.09% |
| record | shipping_cost | 7 | 0.09% |
| record | mission | 7 | 0.09% |
| record | department: Department where the employee works | 7 | 0.09% |
| record | founder_name | 7 | 0.09% |
| record | event_name | 7 | 0.09% |
| record | office | 7 | 0.09% |
| record | distance.parsecs | 7 | 0.09% |
| record | characteristics.type | 7 | 0.09% |
| record | vendor_address | 7 | 0.09% |
| record | punct | 7 | 0.09% |
| record | prep | 7 | 0.09% |
| record | amod | 7 | 0.09% |
| record | key_concepts | 7 | 0.09% |
| record | dispute_id | 7 | 0.09% |
| record | summary | 7 | 0.09% |
| record | support | 7 | 0.09% |
| record | asteroid.orbital_characteristics.classification | 7 | 0.09% |
| record | amount: Monetary value of the transaction in USD | 7 | 0.09% |
| record | hall | 7 | 0.09% |
| record | discovery.location.country | 7 | 0.09% |
| record | email_id | 7 | 0.09% |
| record | referring_physician | 7 | 0.09% |
| record | study_code | 7 | 0.09% |
| record | physical_characteristics.observations.0.publication | 7 | 0.09% |
| record | population | 7 | 0.09% |
| record | impact | 7 | 0.09% |
| record | launch_year | 7 | 0.09% |
| record | asteroid.discovery.observatory | 7 | 0.09% |
| record | priority: Severity level of the issue | 7 | 0.09% |
| record | claim_amount: Total monetary value of the claim | 7 | 0.09% |
| record | cities | 7 | 0.09% |
| record | oxygen_saturation: Percentage of oxygen saturation in blood | 7 | 0.09% |
| record | age_range | 7 | 0.09% |
| record | applicant | 7 | 0.09% |
| record | upload_date | 7 | 0.09% |
| record | address: Full street address of the property | 7 | 0.09% |
| record | city: City where the property is located | 7 | 0.09% |
| record | product_offered | 7 | 0.09% |
| record | device_id: Unique identifier assigned to the medical device | 7 | 0.09% |
| record | supply_duration | 7 | 0.09% |
| record | property_address: Full address of the leased property | 7 | 0.09% |
| record | location.department | 7 | 0.09% |
| record | lead_investor: Primary investor leading the round | 7 | 0.09% |
| record | attendee_count | 7 | 0.09% |
| record | genus.type_species | 7 | 0.09% |
| record | submission_id: Unique identifier for the paper submission | 7 | 0.09% |
| record | imaging_type | 7 | 0.09% |
| record | identifier | 7 | 0.09% |
| record | hotel.location.region | 7 | 0.09% |
| record | thumbnail_style | 7 | 0.09% |
| record | scorer | 7 | 0.09% |
| record | investor_name | 7 | 0.09% |
| record | det | 7 | 0.09% |
| record | meeting_id | 7 | 0.09% |
| record | actions | 7 | 0.09% |
| record | disclosure_id | 7 | 0.09% |
| record | mark_name | 7 | 0.09% |
| record | lead_investor: Primary investor leading the funding round | 7 | 0.09% |
| record | filer | 7 | 0.09% |
| record | status: Clinical interpretation of the result | 7 | 0.09% |
| record | donor_id | 7 | 0.09% |
| record | passenger_address | 7 | 0.09% |
| record | village_name | 7 | 0.09% |
| record | tracking_number | 7 | 0.09% |
| record | status: Current status of the funding round | 7 | 0.09% |
| record | patient_name: Name of the patient filing the claim | 7 | 0.09% |
| record | patient_id: Unique identifier assigned to the patient | 7 | 0.09% |
| record | transfer_type | 7 | 0.09% |
| record | market_share | 7 | 0.09% |
| record | employee_id: Unique identifier assigned to the employee | 7 | 0.09% |
| record | observations.last_observation_date | 7 | 0.09% |
| record | size | 7 | 0.09% |
| record | study_duration | 7 | 0.09% |
| record | environment | 7 | 0.09% |
| record | additional_features | 7 | 0.09% |
| record | dosage_form | 7 | 0.09% |
| record | sku_id | 7 | 0.09% |
| record | unit_location | 7 | 0.09% |
| record | supplier | 7 | 0.09% |
| record | length | 7 | 0.09% |
| record | administration_date | 7 | 0.09% |
| record | asteroid.number | 7 | 0.09% |
| record | dose_number | 7 | 0.09% |
| record | lengthofstay | 7 | 0.09% |
| record | violation_code | 7 | 0.09% |
| record | department: Academic department of the author | 7 | 0.09% |
| record | email_id: Unique identifier for the outreach email | 7 | 0.09% |
| record | discovery.project | 7 | 0.09% |
| record | booking_reference | 7 | 0.09% |
| record | home_team: Name of the team playing at home | 6 | 0.08% |
| record | pharmacy | 6 | 0.08% |
| record | event_date | 6 | 0.08% |
| record | os_version | 6 | 0.08% |
| record | abstract_id | 6 | 0.08% |
| record | inventor | 6 | 0.08% |
| record | assignee | 6 | 0.08% |
| record | location: City and state where the employee is based | 6 | 0.08% |
| record | acquired | 6 | 0.08% |
| record | scientificName | 6 | 0.08% |
| record | constellation | 6 | 0.08% |
| record | game | 6 | 0.08% |
| record | album_title | 6 | 0.08% |
| record | ticket_number | 6 | 0.08% |
| record | github_link | 6 | 0.08% |
| record | areasforimprovement | 6 | 0.08% |
| record | description: Brief description of the transaction purpose | 6 | 0.08% |
| record | patent_id: Unique identifier for the patent application | 6 | 0.08% |
| record | target_demographic | 6 | 0.08% |
| record | claim_status | 6 | 0.08% |
| record | ranking.rank | 6 | 0.08% |
| record | service | 6 | 0.08% |
| record | priority: Severity level assigned to the ticket | 6 | 0.08% |
| record | conclusions | 6 | 0.08% |
| record | clinic_name | 6 | 0.08% |
| record | discovery.observatory.location | 6 | 0.08% |
| record | invoice_date | 6 | 0.08% |
| record | product.ingredients | 6 | 0.08% |
| record | shares: Total number of shares the post received | 6 | 0.08% |
| record | claimant_name | 6 | 0.08% |
| record | sessions | 6 | 0.08% |
| record | passenger_city | 6 | 0.08% |
| record | tweet_id | 6 | 0.08% |
| record | invoice_number | 6 | 0.08% |
| record | compound | 6 | 0.08% |
| record | person.publications.0.title | 6 | 0.08% |
| record | monthly_premium | 6 | 0.08% |
| record | product_id: Unique identifier assigned to the chemical product | 6 | 0.08% |
| record | orbital_characteristics.asteroid_belt_classification | 6 | 0.08% |
| record | amount_billed | 6 | 0.08% |
| record | host | 6 | 0.08% |
| record | incident_code | 6 | 0.08% |
| record | cost_savings | 6 | 0.08% |
| record | project | 6 | 0.08% |
| record | issue_description: Detailed description of the customer's problem | 6 | 0.08% |
| record | drug_code | 6 | 0.08% |
| record | additional_investors | 6 | 0.08% |
| record | pricepernight | 6 | 0.08% |
| record | location: Headquarters city and state or country | 6 | 0.08% |
| record | asteroid.named_after.person.occupation | 6 | 0.08% |
| record | tissue_specificity | 6 | 0.08% |
| record | age: Age of the patient in years | 6 | 0.08% |
| record | support_ticket | 6 | 0.08% |
| record | countries | 6 | 0.08% |
| record | seller | 6 | 0.08% |
| record | social_media | 6 | 0.08% |
| record | league | 6 | 0.08% |
| record | acquisition_id | 6 | 0.08% |
| record | visit_status | 6 | 0.08% |
| record | currency_pair: Pair of currencies involved in the exchange | 6 | 0.08% |
| record | storage | 6 | 0.08% |
| record | dateofincident | 6 | 0.08% |
| record | patient_name: Full name of the patient undergoing the test | 6 | 0.08% |
| record | company.services | 6 | 0.08% |
| record | religion | 6 | 0.08% |
| record | genus.name | 6 | 0.08% |
| record | issue_type | 6 | 0.08% |
| record | status: Current enrollment or completion status | 6 | 0.08% |
| record | prescriber_name | 6 | 0.08% |
| record | claim_history | 6 | 0.08% |
| record | athlete_name | 6 | 0.08% |
| record | payment_terms | 6 | 0.08% |
| record | test_type | 6 | 0.08% |
| record | course_code | 6 | 0.08% |
| record | building | 6 | 0.08% |
| record | name: Name of the product as listed | 6 | 0.08% |
| record | form | 6 | 0.08% |
| record | venue: Location where the event took place | 6 | 0.08% |
| record | price_range | 6 | 0.08% |
| record | budget | 6 | 0.08% |
| record | stock_level | 6 | 0.08% |
| record | status: Current clinical status of the patient | 6 | 0.08% |
| record | physical_characteristics.absolute_magnitude | 6 | 0.08% |
| record | hotel.star_rating | 6 | 0.08% |
| record | hotel.location.country | 6 | 0.08% |
| record | intervention | 6 | 0.08% |
| record | treatment_plan | 6 | 0.08% |
| record | retention_rate | 6 | 0.08% |
| record | key_metric | 6 | 0.08% |
| record | email_address | 6 | 0.08% |
| record | view_count | 6 | 0.08% |
| record | spo2 | 6 | 0.08% |
| record | address: Full physical address of the customer | 6 | 0.08% |
| record | sender_company | 6 | 0.08% |
| record | stock_symbol | 6 | 0.08% |
| record | destination_club | 6 | 0.08% |
| record | influences | 6 | 0.08% |
| record | duration: Number of days the program lasts | 6 | 0.08% |
| record | startup | 6 | 0.08% |
| record | home_score | 6 | 0.08% |
| record | symptom | 6 | 0.08% |
| record | donor_address | 6 | 0.08% |
| record | subject: Subject line of the outreach email | 6 | 0.08% |
| record | orbit.perihelion.unit | 6 | 0.08% |
| record | publication_issue | 6 | 0.08% |
| record | content_type | 6 | 0.08% |
| record | aces | 5 | 0.06% |
| record | originalpostlink | 5 | 0.06% |
| record | metabolite | 5 | 0.06% |
| record | founded_year | 5 | 0.06% |
| record | assigned_to | 5 | 0.06% |
| record | review_platform | 5 | 0.06% |
| record | size_and_fit.length.unit | 5 | 0.06% |
| record | size_and_fit.measurements.2.value | 5 | 0.06% |
| record | size_and_fit.measurements.1.unit | 5 | 0.06% |
| record | issuer | 5 | 0.06% |
| record | biography.education.4.institution | 5 | 0.06% |
| record | performers | 5 | 0.06% |
| record | route: Origin-destination route string like JFK-LAX | 5 | 0.06% |
| record | city: City where the recipient resides | 5 | 0.06% |
| record | contact: Name of the point of contact at the company | 5 | 0.06% |
| record | firmware_version | 5 | 0.06% |
| record | event.activities | 5 | 0.06% |
| record | trial_title | 5 | 0.06% |
| record | hours_of_operation.1.close_time | 5 | 0.06% |
| record | hours_of_operation.4.close_time | 5 | 0.06% |
| record | hours_of_operation.1.open_time | 5 | 0.06% |
| record | hours_of_operation.4.open_time | 5 | 0.06% |
| record | hours_of_operation.0.day | 5 | 0.06% |
| record | hours_of_operation.6.open_time | 5 | 0.06% |
| record | priority_registration.instructions | 5 | 0.06% |
| record | zebras_care_foundation.beneficiary.name | 5 | 0.06% |
| record | zebras_care_foundation.beneficiary.age | 5 | 0.06% |
| record | abstract_length: Length of the abstract in words | 5 | 0.06% |
| record | review_date: Date when the paper was reviewed | 5 | 0.06% |
| record | conference_name: Name of the conference where submitted | 5 | 0.06% |
| record | funding_source: Source of research funding | 5 | 0.06% |
| record | fieldofstudy: Academic domain of the research | 5 | 0.06% |
| record | game_id: Unique identifier for the game as listed in the schedule | 5 | 0.06% |
| record | purchase_location | 5 | 0.06% |
| record | supportresponsetime | 5 | 0.06% |
| record | devicebatterylife | 5 | 0.06% |
| record | region.divisions.1.name | 5 | 0.06% |
| record | region.name | 5 | 0.06% |
| record | date: Date when the reaction was reported | 5 | 0.06% |
| record | organization-other | 5 | 0.06% |
| record | contributor_count | 5 | 0.06% |
| record | ebitda_margin | 5 | 0.06% |
| record | hotel.location.nearby_attractions | 5 | 0.06% |
| record | hotel.hotel_group | 5 | 0.06% |
| record | location: Venue where the match was held | 5 | 0.06% |
| record | venue_name: Name of the specific tournament venue | 5 | 0.06% |
| record | defendant_address | 5 | 0.06% |
| record | judge | 5 | 0.06% |
| record | status: Current status of the subscription | 5 | 0.06% |
| record | date: Date the match was scheduled or played | 5 | 0.06% |
| record | name: Full name of the contact person | 5 | 0.06% |
| record | date: Date the outreach was made | 5 | 0.06% |
| record | status: Current status of the outreach | 5 | 0.06% |
| record | policy_holder | 5 | 0.06% |
| record | target_audience: Demographic group the product is marketed to | 5 | 0.06% |
| record | properties | 5 | 0.06% |
| record | Gary Coleman <> former resident of | 5 | 0.06% |
| record | freshman girl <> attacked | 5 | 0.06% |
| record | Assurant Health <> provider of | 5 | 0.06% |
| record | United Airlines <> dominated | 5 | 0.06% |
| record | inventors | 5 | 0.06% |
| record | registrant: Company that owns the registered trademark | 5 | 0.06% |
| record | opponent: Entity filing the dispute or opposition | 5 | 0.06% |
| record | player1_address | 5 | 0.06% |
| record | launch_date: Date the product was launched | 5 | 0.06% |
| record | employee_id: Unique identifier assigned to each new hire | 5 | 0.06% |
| record | review_period | 5 | 0.06% |
| record | regulatory_status | 5 | 0.06% |
| record | species | 5 | 0.06% |
| record | version_number | 5 | 0.06% |
| record | address: Complete street address including city and state | 5 | 0.06% |
| record | student_name | 5 | 0.06% |
| record | exam_name | 5 | 0.06% |
| record | author: Name of the author who wrote the digest | 5 | 0.06% |
| record | status: Current status of the outreach interaction | 5 | 0.06% |
| record | annual_revenue | 5 | 0.06% |
| record | spoofed_domain | 5 | 0.06% |
| record | target_department | 5 | 0.06% |
| record | controversy.victim_actresses.1.actress_name | 5 | 0.06% |
| record | controversy.victim_actresses.2.krk_comment | 5 | 0.06% |
| record | controversy.victim_actresses.1.krk_comment | 5 | 0.06% |
| record | political_context | 5 | 0.06% |
| record | event.subject | 5 | 0.06% |
| record | asteroid.discovery.discoverer.affiliation.observatory | 5 | 0.06% |
| record | asteroid.named_after.birth_location.country | 5 | 0.06% |
| record | asteroid.discovery.discoverer.name | 5 | 0.06% |
| record | transaction_id: Unique identifier for the transaction entry | 5 | 0.06% |
| record | urgency: Priority level of the reported issue | 5 | 0.06% |
| record | last_updated: Date when product record was last modified | 5 | 0.06% |
| record | supplement | 5 | 0.06% |
| record | submitter: Name of the user who opened the ticket | 5 | 0.06% |
| record | discharge_date: Date the patient was discharged | 5 | 0.06% |
| record | admission_diagnosis: Primary diagnosis at admission | 5 | 0.06% |
| record | medications: List of prescribed medications | 5 | 0.06% |
| record | founder_id | 5 | 0.06% |
| record | path_design.disadvantages | 5 | 0.06% |
| record | investor_list | 5 | 0.06% |
| record | sales_period | 5 | 0.06% |
| record | status: Current legislative status of the bill | 5 | 0.06% |
| record | founder_names | 5 | 0.06% |
| record | orbit.orbital_type | 5 | 0.06% |
| record | characteristics.diameter.value | 5 | 0.06% |
| record | donation_description: Brief description of the donation's purpose or context | 5 | 0.06% |
| record | physical_characteristics.albedo | 5 | 0.06% |
| record | product.browser_compatibility | 5 | 0.06% |
| record | product.download_link | 5 | 0.06% |
| record | vehicle.specifications.transmission | 5 | 0.06% |
| record | vehicle.specifications.crew_capacity | 5 | 0.06% |
| record | creative work | 5 | 0.06% |
| record | symptoms: List of reported symptoms like fever or headache | 5 | 0.06% |
| record | action | 5 | 0.06% |
| record | research.researcher.affiliation.name | 5 | 0.06% |
| record | research.experiment_details.outcome | 5 | 0.06% |
| record | company.address.area | 5 | 0.06% |
| record | company.address.street | 5 | 0.06% |
| record | severity: Priority level of the issue | 5 | 0.06% |
| record | person.birth_date.year | 5 | 0.06% |
| record | island | 5 | 0.06% |
| record | waterway | 5 | 0.06% |
| record | inventors: List of inventor names involved in the filing | 5 | 0.06% |
| record | author_information.occupation | 5 | 0.06% |
| record | author_information.family.pets.0.type | 5 | 0.06% |
| record | author_information.military_service.status | 5 | 0.06% |
| record | reviewer: Name of the person who wrote the review | 5 | 0.06% |
| record | pickup_type: Type of service or pickup method | 5 | 0.06% |
| record | diagnosis_code: ICD-10 code for the patient's condition | 5 | 0.06% |
| record | state: State where the patient resides | 5 | 0.06% |
| record | max_wind_speed | 5 | 0.06% |
| record | price_range.per | 5 | 0.06% |
| record | description: Brief summary of the job responsibilities | 5 | 0.06% |
| record | award_name | 5 | 0.06% |
| record | treaty | 5 | 0.06% |
| record | creation.event | 5 | 0.06% |
| record | ticket_id: Unique identifier for the support case | 5 | 0.06% |
| record | initial_challenge: Primary early operational or product issue faced | 5 | 0.06% |
| record | reviewer_status | 5 | 0.06% |
| record | linkedinposturl | 5 | 0.06% |
| record | availability_date | 5 | 0.06% |
| record | predeceased.3.relationship | 5 | 0.06% |
| record | predeceased.4.name | 5 | 0.06% |
| record | siblings_in_law.0.name | 5 | 0.06% |
| record | parents.0.name | 5 | 0.06% |
| record | study_metadata.measurement_technology.anatomical_site | 5 | 0.06% |
| record | population.gender | 5 | 0.06% |
| record | population.age_range.max | 5 | 0.06% |
| record | follow_up.average_duration.years | 5 | 0.06% |
| record | outcomes.fracture_events.0.count | 5 | 0.06% |
| record | statistical_results.predictive_model.value | 5 | 0.06% |
| record | test_code | 5 | 0.06% |
| record | test_name | 5 | 0.06% |
| record | direction_of_travel | 5 | 0.06% |
| record | assigned_adjuster | 5 | 0.06% |
| record | channel_name: Name of the YouTube channel hosting the video | 5 | 0.06% |
| record | Sparta Wroclaw <> managed by | 5 | 0.06% |
| record | Stal Rzeszów <> managed by | 5 | 0.06% |
| record | seller_location: Full address of the seller including city, state, and zip code | 5 | 0.06% |
| record | unit_price: Price per unit of the product | 5 | 0.06% |
| record | paymentduedays: Number of days until payment is due after notice | 5 | 0.06% |
| record | facility: Facility name and address where device is located | 5 | 0.06% |
| record | status: Current status of vaccination record | 5 | 0.06% |
| record | sunless_tanning.product_collection.products.1.description | 5 | 0.06% |
| record | performance_improvement | 5 | 0.06% |
| record | sender_domain: Domain of the spoofed email address | 5 | 0.06% |
| record | status: Current status of the report | 5 | 0.06% |
| record | last_modified | 5 | 0.06% |
| record | research_context.primary_technique | 5 | 0.06% |
| record | research_context.application_goal | 5 | 0.06% |
| record | content.elements.name | 5 | 0.06% |
| record | movements | 5 | 0.06% |
| record | submission_method | 5 | 0.06% |
| record | advertiser_information.contact_information.phone_number | 5 | 0.06% |
| record | distribution_information.trail_details.trail_name | 5 | 0.06% |
| record | asteroid_designation | 5 | 0.06% |
| record | nutritional_aspects.carbohydrates.daily_percentage | 5 | 0.06% |
| record | nutritional_aspects.sucrose.daily_limit_percentage | 5 | 0.06% |
| record | representatives.1.statements | 5 | 0.06% |
| record | representatives.1.role | 5 | 0.06% |
| record | distribution.regions.0.region_name | 5 | 0.06% |
| record | distribution.regions.1.region_name | 5 | 0.06% |
| record | risk_score: Numerical value indicating the level of risk exposure | 5 | 0.06% |
| record | product.version | 5 | 0.06% |
| record | product.categories | 5 | 0.06% |
| record | popular | 5 | 0.06% |
| record | risks | 5 | 0.06% |
| record | removal_instructions.browser_specific_instructions.2.browser | 5 | 0.06% |
| record | followupprovider: Name of provider for follow-up visit | 5 | 0.06% |
| record | wallet | 5 | 0.06% |
| record | genus.pollination.pollinators | 5 | 0.06% |
| record | sender: Name of the person sending the confirmation | 5 | 0.06% |
| record | asteroid.discovery.discoverer.nationality | 5 | 0.06% |
| record | narrative.setting.description | 5 | 0.06% |
| record | narrative.protagonist.background | 5 | 0.06% |
| record | publisher: Publishing house responsible for the work | 5 | 0.06% |
| record | review.overall_rating | 5 | 0.06% |
| record | disbursement_plan | 5 | 0.06% |
| record | description: Brief description of the incident and damages | 5 | 0.06% |
| record | industry_example.industry | 5 | 0.06% |
| record | days | 5 | 0.06% |
| record | areaforimprovement: Specific skill or behavior needing development | 5 | 0.06% |
| record | author: Name of user who posted the content | 5 | 0.06% |
| record | venue_capacity | 5 | 0.06% |
| record | senior_rate | 5 | 0.06% |
| record | expected_revenue | 5 | 0.06% |
| record | history.creation.shipyard.location | 5 | 0.06% |
| record | history.creation.shipyard.name | 5 | 0.06% |
| record | vehicle: Make, model, and year of the vehicle involved | 5 | 0.06% |
| record | location_tag | 5 | 0.06% |
| record | album.tracklist.cd.9.tie_in.0.type | 5 | 0.06% |
| record | album.tracklist.cd.4.notes | 5 | 0.06% |
| record | album.tracklist.dvd.1.track_number | 5 | 0.06% |
| record | album.tracklist.dvd.0.type | 5 | 0.06% |
| record | album.tracklist.cd.7.track_number | 5 | 0.06% |
| record | album.versions.2.contents | 5 | 0.06% |
| record | album.sources.1.publication_date | 5 | 0.06% |
| record | album.tracklist.dvd.2.type | 5 | 0.06% |
| record | album.chart_performance.record_holder_description | 5 | 0.06% |
| record | album.tracklist.cd.6.title | 5 | 0.06% |
| record | album.tracklist.cd.1.track_number | 5 | 0.06% |
| record | album.tracklist.cd.2.notes | 5 | 0.06% |
| record | album.external_links.1.description | 5 | 0.06% |
| record | album.sources.1.access_date | 5 | 0.06% |
| record | album.sources.3.publication_date | 5 | 0.06% |
| record | experience_level | 5 | 0.06% |
| record | delivery_mode: Mode of training delivery | 5 | 0.06% |
| record | orbital_properties.separation.unit | 5 | 0.06% |
| record | serial_number: Unique device identifier assigned by manufacturer | 5 | 0.06% |
| record | terminal.name | 5 | 0.06% |
| record | terminal.history.1.description | 5 | 0.06% |
| record | terminal.airlines | 5 | 0.06% |
| record | date: Date the product was launched | 5 | 0.06% |
| record | distance.light_years | 5 | 0.06% |
| record | characteristics.spectral_class | 5 | 0.06% |
| record | recipient_email | 5 | 0.06% |
| record | rating: Numerical score given to the film | 5 | 0.06% |
| record | symbol: Company ticker symbol as listed in the report | 5 | 0.06% |
| record | match_outcome | 5 | 0.06% |
| record | yellow_cards | 5 | 0.06% |
| record | reported_by | 5 | 0.06% |
| record | article.entities.people.0.title | 5 | 0.06% |
| record | article.entities.people.0.name | 5 | 0.06% |
| record | article.entities.people.1.title | 5 | 0.06% |
| record | article.summary | 5 | 0.06% |
| record | article.copyright_details.copyright_transparency_report_url | 5 | 0.06% |
| record | food | 5 | 0.06% |
| record | year_founded | 5 | 0.06% |
| record | day_of_year | 5 | 0.06% |
| record | term_years | 5 | 0.06% |
| record | emergency_contact: Phone number for emergency assistance | 5 | 0.06% |
| record | urls | 5 | 0.06% |
| record | status: Current status of transfer | 5 | 0.06% |
| record | chamber: Legislative chamber where the vote took place | 5 | 0.06% |
| record | patient | 5 | 0.06% |
| record | case_name: Full name of the case in format plaintiff v defendant | 5 | 0.06% |
| record | primary_tissue | 5 | 0.06% |
| record | patient_province | 5 | 0.06% |
| record | vote_count | 5 | 0.06% |
| record | movement_reason | 5 | 0.06% |
| record | asteroid.description.orbital_period_years | 5 | 0.06% |
| record | asteroid.description.discovery_details.location.county | 5 | 0.06% |
| record | transfer_portal | 5 | 0.06% |
| record | tech_product | 5 | 0.06% |
| record | stub_article.call_to_action | 5 | 0.06% |
| record | github_id | 5 | 0.06% |
| record | product.specifications.camera.features | 5 | 0.06% |
| record | product.model | 5 | 0.06% |
| record | context.strategic_approach | 5 | 0.06% |
| record | lead_agency.name | 5 | 0.06% |
| record | discovery.survey | 5 | 0.06% |
| record | target_population: Group of patients or individuals the guideline targets | 5 | 0.06% |
| record | quantity: number of units currently in stock | 5 | 0.06% |
| record | period | 5 | 0.06% |
| record | test_cases_completed | 5 | 0.06% |
| record | study_protocol | 5 | 0.06% |
| record | name: Product name as listed in inventory records | 5 | 0.06% |
| record | description.leaves.terminal_spine.color | 5 | 0.06% |
| record | description.fruit.length_mm.min | 5 | 0.06% |
| record | description.leaves.margins | 5 | 0.06% |
| record | description.leaves.width_cm.min | 5 | 0.06% |
| record | systematics_and_distribution.location.country | 5 | 0.06% |
| record | systematics_and_distribution.similar_species | 5 | 0.06% |
| record | description.flowers.arrangement | 5 | 0.06% |
| record | description.leaves.surface | 5 | 0.06% |
| record | shipping | 5 | 0.06% |
| record | sectorial_library_system | 5 | 0.06% |
| record | tribute_to.profession | 5 | 0.06% |
| record | presenter | 5 | 0.06% |
| record | agent: Name of the real estate agent representing the seller | 5 | 0.06% |
| record | source.location | 5 | 0.06% |
| record | tv_program.production.production_area | 5 | 0.06% |
| record | injury_count | 5 | 0.06% |
| record | product_title | 5 | 0.06% |
| record | target_company: Company where the role is located | 5 | 0.06% |
| record | salary_range: Expected or offered salary range for the role | 5 | 0.06% |
| record | product_example.currency | 5 | 0.06% |
| record | product_example.name | 5 | 0.06% |
| record | dateofexam: Date when the imaging examination was performed | 5 | 0.06% |
| record | address_street: Street address line of the patient | 5 | 0.06% |
| record | symptom_location: Body region where symptoms are felt | 5 | 0.06% |
| record | measurement.size_comparison | 5 | 0.06% |
| record | origin.year | 5 | 0.06% |
| record | address: Vendor's physical business address | 5 | 0.06% |
| record | orbit.semi_major_axis.unit | 5 | 0.06% |
| record | review.persons.1.description | 5 | 0.06% |
| record | review.performance.venue.name | 5 | 0.06% |
| record | review.persons.1.name | 5 | 0.06% |
| record | review.performance.performers.1.role | 5 | 0.06% |
| record | diagnosis_code: Medical code associated with the claim condition | 5 | 0.06% |
| record | reference: Internal reference code linked to the ticket | 5 | 0.06% |
| record | duration.max_years | 5 | 0.06% |
| record | beneficiary_focus | 5 | 0.06% |
| record | recipient: Name of the recipient at the target company | 5 | 0.06% |
| record | deductiblestatus: Whether patient's deductible has been met | 5 | 0.06% |
| record | treatment_type: Type of medical treatment or procedure | 5 | 0.06% |
| record | location: City and country where the role is based | 5 | 0.06% |
| record | weather_incident | 5 | 0.06% |
| record | ticket_id: Unique identifier for the incident ticket | 5 | 0.06% |
| record | region: Geographic region where the issue occurred | 5 | 0.06% |
| record | error_log: Relevant error message or log snippet from the incident | 5 | 0.06% |
| record | delay_cause | 5 | 0.06% |
| record | surveyPurpose | 5 | 0.06% |
| record | historical_periods | 5 | 0.06% |
| record | structures | 5 | 0.06% |
| record | budget_remaining | 5 | 0.06% |
| record | backend_integration_status | 5 | 0.06% |
| record | shipping_method: Type of shipping used | 5 | 0.06% |
| record | heroes | 5 | 0.06% |
| record | place.location.department | 5 | 0.06% |
| record | place.location.province | 5 | 0.06% |
| record | immunogen | 5 | 0.06% |
| record | application_notes | 5 | 0.06% |
| record | developmental_stage | 5 | 0.06% |
| record | CC | 5 | 0.06% |
| record | NN | 5 | 0.06% |
| record | IN | 5 | 0.06% |
| record | shipping_origin: City and state where product is shipped from | 5 | 0.06% |
| record | team_count | 5 | 0.06% |
| record | monitoring_time: Exact time of monitoring in 24-hour format | 5 | 0.06% |
| record | respiratory_rate: Number of breaths per minute | 5 | 0.06% |
| record | paradox.change_returned_to_people | 5 | 0.06% |
| record | product_description | 5 | 0.06% |
| record | tournament_name: Name of the tennis tournament such as Monte Carlo Masters | 5 | 0.06% |
| record | credits.graphic_design.0.designer | 5 | 0.06% |
| record | credits.contributors.1.role | 5 | 0.06% |
| record | submitted_to: Journal or publication to which the paper is submitted | 5 | 0.06% |
| record | topic_area: Subject matter or research focus of the paper | 5 | 0.06% |
| record | description: Detailed account of the issue reported | 5 | 0.06% |
| record | file_id | 5 | 0.06% |
| record | focus: Primary technology or business area of the company | 5 | 0.06% |
| record | delay_minutes: Number of minutes flight is delayed | 5 | 0.06% |
| record | launched_by | 5 | 0.06% |
| record | headliner: Main performing artist for the festival | 5 | 0.06% |
| record | food item | 5 | 0.06% |
| record | dish | 5 | 0.06% |
| record | canton_name_french | 5 | 0.06% |
| record | time: Time range of the session as shown in the Time column | 5 | 0.06% |
| record | speaker: Name of the speaker assigned to the session | 5 | 0.06% |
| record | date: Date the event is scheduled as per the Date column | 5 | 0.06% |
| record | venue: Location where the event is held as listed in Venue column | 5 | 0.06% |
| record | company_code | 5 | 0.06% |
| record | launch_year: Year the product was first introduced | 5 | 0.06% |
| record | status: Current status of appointment | 5 | 0.06% |
| record | name: Full name of the appointed official | 5 | 0.06% |
| record | political_entity | 5 | 0.06% |
| record | games | 5 | 0.06% |
| record | medals.total | 5 | 0.06% |
| record | claimant | 5 | 0.06% |
| record | vehicle_make | 5 | 0.06% |
| record | adjuster | 5 | 0.06% |
| record | quote.text | 5 | 0.06% |
| record | favorite_players | 5 | 0.06% |
| record | asteroid.discovery.observatory.location.county | 5 | 0.06% |
| record | target_clients: Client segment the product serves | 5 | 0.06% |
| record | assigned_coordinator: Name of the onboarding coordinator assigned to the employee | 5 | 0.06% |
| record | description: Brief note on reason or context of the transaction | 5 | 0.06% |
| record | property_id: Unique identifier assigned to the rental unit | 5 | 0.06% |
| record | management_contact: Name or entity managing the property | 5 | 0.06% |
| record | person.birth_place.city | 5 | 0.06% |
| record | person.profession | 5 | 0.06% |
| record | person.us_senator.start_year | 5 | 0.06% |
| record | owner: Name of the healthcare professional owning the device | 5 | 0.06% |
| record | strategic_impact | 5 | 0.06% |
| record | patient_care.group_therapy.benefits | 5 | 0.06% |
| record | team_name | 5 | 0.06% |
| record | symbol | 5 | 0.06% |
| record | asteroid.dedication.person.occupation | 5 | 0.06% |
| record | diagnosis_icd10 | 5 | 0.06% |
| record | companies | 5 | 0.06% |
| record | payer | 5 | 0.06% |
| record | account | 5 | 0.06% |
| record | coach | 5 | 0.06% |
| record | passenger_information.adults.age_range | 5 | 0.06% |
| record | passenger_information.children.age_range | 5 | 0.06% |
| record | departure_airport_prompt | 5 | 0.06% |
| record | user_id: ID of the user who submitted the snapshot | 5 | 0.06% |
| record | status: Current status of the exchange rate record | 5 | 0.06% |
| record | description: Brief text describing the content of the post | 5 | 0.06% |
| record | user_name: Name of the user who created the repost | 5 | 0.06% |
| record | trend_name: Name of the viral trend being reposted | 5 | 0.06% |
| record | demand | 5 | 0.06% |
| record | comment_content: Text of the comment including personal anecdotes and tool usage | 5 | 0.06% |
| record | naming.named_after.name | 5 | 0.06% |
| record | additional_information | 5 | 0.06% |
| record | sender: Name of person who submitted the report | 5 | 0.06% |
| record | amount: Claimed financial amount in dollars | 5 | 0.06% |
| record | warehouse_location | 5 | 0.06% |
| record | lastrevieweddate | 5 | 0.06% |
| record | service.responsible_person.role | 5 | 0.06% |
| record | service.interventions | 5 | 0.06% |
| record | home_team: Team playing at home stadium | 5 | 0.06% |
| record | match_location: Stadium or venue where match occurred | 5 | 0.06% |
| record | requisition_id: Unique identifier for the job posting | 5 | 0.06% |
| record | total_nights | 5 | 0.06% |
| record | person.birth_location.country | 5 | 0.06% |
| record | cause: Reason for the reported issue or system failure | 5 | 0.06% |
| record | date: Date the transaction was initiated | 5 | 0.06% |
| record | additional_value: Extra numeric value associated with the transaction | 5 | 0.06% |
| record | release_year | 5 | 0.06% |
| record | date_submitted: Date the issue was submitted to support | 5 | 0.06% |
| record | template_substitution.preview_code_example | 5 | 0.06% |
| record | advantages | 5 | 0.06% |
| record | release_details.artist.occupation | 5 | 0.06% |
| record | release_details.album.themes | 5 | 0.06% |
| record | app | 5 | 0.06% |
| record | event-attack/battle/war/militaryconflict | 5 | 0.06% |
| record | industry_focus | 5 | 0.06% |
| record | treatment_examples.items.2.conditions | 5 | 0.06% |
| record | treatment_examples.items.3.conditions | 5 | 0.06% |
| record | beds | 5 | 0.06% |
| record | location: Physical address of the employee's workplace | 5 | 0.06% |
| record | status: Current status of the donation transaction | 5 | 0.06% |
| record | description.leaf.apex | 5 | 0.06% |
| record | description.fruit.dimensions | 5 | 0.06% |
| record | description.stipules.dimensions | 5 | 0.06% |
| record | notes: Additional clinical observations or patient-reported symptoms | 5 | 0.06% |
| record | status: Current status of the test result | 5 | 0.06% |
| record | amount: Funding amount in millions | 5 | 0.06% |
| record | father.occupation | 5 | 0.06% |
| record | comparison.brand_with_similar_look | 5 | 0.06% |
| record | user: Name of the team member or user using the product | 5 | 0.06% |
| record | governance.overseeing_office | 5 | 0.06% |
| record | patient_id: Unique identifier for the monitored patient | 5 | 0.06% |
| record | citations.4.description | 5 | 0.06% |
| record | citations.0.year | 5 | 0.06% |
| record | taxonomic_reference | 5 | 0.06% |
| record | gymnosperm_diversity.pine_species_count | 5 | 0.06% |
| record | gymnosperm_diversity.species_examples.1.characteristic | 5 | 0.06% |
| record | citations.1.year | 5 | 0.06% |
| record | caveat.example.components.0.component | 5 | 0.06% |
| record | population.count | 5 | 0.06% |
| record | volume | 5 | 0.06% |
| record | price: Retail price of the product in US dollars | 5 | 0.06% |
| record | statue.restoration.0.funding.amount_raised.currency | 5 | 0.06% |
| record | statue.restoration.0.funding.source | 5 | 0.06% |
| record | statue.restoration.0.start_date | 5 | 0.06% |
| record | statue.location.museum | 5 | 0.06% |
| record | statue.name.greek | 5 | 0.06% |
| record | drug: Name of the medication as listed in the database | 5 | 0.06% |
| record | manufacturer: Company that produces the drug | 5 | 0.06% |
| record | routeofadministration: Method by which the drug is administered | 5 | 0.06% |
| record | patient_population: Group of patients for whom the drug is intended | 5 | 0.06% |
| record | reviewer_name: Name of the reviewer providing feedback | 5 | 0.06% |
| record | forks: Number of forks the repository has received | 5 | 0.06% |
| record | research_framework_items | 5 | 0.06% |
| record | related_content | 5 | 0.06% |
| record | diseases | 5 | 0.06% |
| record | measurements | 5 | 0.06% |
| record | physical_characteristics.observation_year | 5 | 0.06% |
| record | growth_rate | 5 | 0.06% |
| record | review_status: Current status of the review | 5 | 0.06% |
| record | end_time | 5 | 0.06% |
| record | status: Current status of the encounter | 5 | 0.06% |
| record | genus.unique_features | 5 | 0.06% |
| record | hotel_details.name | 5 | 0.06% |
| record | hotel_details.star_rating | 5 | 0.06% |
| record | bench_points: Total points scored by bench players | 5 | 0.06% |
| record | reviewer_location: City and state where the reviewer is located | 5 | 0.06% |
| record | serial: Unique identifier for the product unit | 5 | 0.06% |
| record | customer_address: Full address including street, apt, city, state, zip | 5 | 0.06% |
| record | exam | 5 | 0.06% |
| record | company: Name of the company hiring | 5 | 0.06% |
| record | responsibilities: Key duties and tasks associated with the role | 5 | 0.06% |
| record | magazine.article.page_breakdown.0.page_number | 5 | 0.06% |
| record | magazine.article.page_breakdown.1.advertising_percentage | 5 | 0.06% |
| record | magazine.article.page_breakdown.2.page_number | 5 | 0.06% |
| record | magazine.article.page_breakdown.3.page_number | 5 | 0.06% |
| record | magazine.article.page_breakdown.8.advertising_percentage | 5 | 0.06% |
| record | magazine.article.page_breakdown.9.advertisers | 5 | 0.06% |
| record | magazine.article.page_breakdown.10.advertisers | 5 | 0.06% |
| record | magazine.article.page_breakdown.12.page_number | 5 | 0.06% |
| record | magazine.article.page_breakdown.12.advertising_percentage | 5 | 0.06% |
| record | magazine.article.page_breakdown.13.advertising_percentage | 5 | 0.06% |
| record | magazine.article.page_breakdown.15.advertisers | 5 | 0.06% |
| record | invoice_number: Unique identifier for the invoice as assigned | 5 | 0.06% |
| record | anatomy | 5 | 0.06% |
| record | item_description.weight.value | 5 | 0.06% |
| record | item_description.shipping.rush_details | 5 | 0.06% |
| record | status: Current state of the ticket after resolution | 5 | 0.06% |
| record | title: Job title as listed in the requisition | 5 | 0.06% |
| record | brands | 5 | 0.06% |
| record | dosage: Specific amount or form of medication prescribed | 5 | 0.06% |
| record | quantity: Number of tablets or units dispensed | 5 | 0.06% |
| record | date: Date the tournament was held | 5 | 0.06% |
| record | hashtags | 5 | 0.06% |
| record | name: Unique identifier of the treaty agreement | 5 | 0.06% |
| record | signatories: List of officials who co-signed the treaty | 5 | 0.06% |
| record | activation_mechanism | 5 | 0.06% |
| record | entity.company_name | 5 | 0.06% |
| record | location.airport_travel_time.value | 5 | 0.06% |
| record | child_policy.infants.age_range | 5 | 0.06% |
| record | child_policy.adult_age | 5 | 0.06% |
| record | booking_policy.multiple_rooms | 5 | 0.06% |
| record | channel_url | 5 | 0.06% |
| record | acquired_company: Name of the company being acquired | 5 | 0.06% |
| record | sector: Industry or market sector involved in the acquisition | 5 | 0.06% |
| record | deal_id: Unique identifier for the acquisition transaction | 5 | 0.06% |
| record | amount: Total amount raised in millions | 5 | 0.06% |
| record | investors: List of venture capital firms involved | 5 | 0.06% |
| record | sender: Name of the person initiating the outreach | 5 | 0.06% |
| record | operational_status: Current phase of business operations | 5 | 0.06% |
| record | net_profit: Annual net profit in millions of dollars | 5 | 0.06% |
| record | ebitda_margin: Percentage of revenue that represents EBITDA | 5 | 0.06% |
| record | affiliation.0.type | 5 | 0.06% |
| record | adverse_effects | 5 | 0.06% |
| record | footer_links | 5 | 0.06% |
| record | npadvmod | 5 | 0.06% |
| record | pobj | 5 | 0.06% |
| record | paralympic_games.season | 5 | 0.06% |
| record | primate.name | 5 | 0.06% |
| record | primate.morphological_analysis.1.type | 5 | 0.06% |
| record | entries | 5 | 0.06% |
| record | market_share: Percentage of market share held by the company | 5 | 0.06% |
| record | last_name | 5 | 0.06% |
| record | product_price | 5 | 0.06% |
| record | sponsorship | 5 | 0.06% |
| record | units_sold: Total units sold by a specific date | 5 | 0.06% |
| record | facility_address | 5 | 0.06% |
| record | submission_amount: Financial amount associated with submission or grant | 5 | 0.06% |
| record | event.person.role | 5 | 0.06% |
| record | event.new_club.contract_duration | 5 | 0.06% |
| record | home_team_score | 5 | 0.06% |
| record | place.name | 5 | 0.06% |
| record | sku: Unique product identifier as assigned in inventory | 5 | 0.06% |
| record | provider_institution | 5 | 0.06% |
| record | submitter: Name of the user who reported the phishing incident | 5 | 0.06% |
| record | location: Geographic office or city where the report originated | 5 | 0.06% |
| record | disputingpartya | 5 | 0.06% |
| record | registration_number | 5 | 0.06% |
| record | location.relative_location | 5 | 0.06% |
| record | assists_leader | 5 | 0.06% |
| record | device_model: Model name of the returned electronic device | 5 | 0.06% |
| record | address: Full physical address of the property | 5 | 0.06% |
| record | description: Brief textual description of property features and appeal | 5 | 0.06% |
| record | founder_name: Full name of the founder as stated in passage | 5 | 0.06% |
| record | motivation: Personal reason driving the company's founding | 5 | 0.06% |
| record | result_count | 5 | 0.06% |
| record | physical_characteristics.observations.0.telescope.year_launched | 5 | 0.06% |
| record | arguments_for_using_gdz | 5 | 0.06% |
| record | client_id | 5 | 0.06% |
| record | partner_revenue: Partner's revenue in millions | 5 | 0.06% |
| record | location: City and state of the company | 5 | 0.06% |
| record | partner_status: Operational status of the partner company | 5 | 0.06% |
| record | claim_date: Date the claim was submitted | 5 | 0.06% |
| record | label | 5 | 0.06% |
| record | patient_name: Full name of the vaccinated individual | 5 | 0.06% |
| record | property_type: Type of housing unit | 5 | 0.06% |
| record | sales_volume: Total units sold over time | 5 | 0.06% |
| record | details.calorie_burn.running_speed | 5 | 0.06% |
| record | announcement_metadata.source_url | 5 | 0.06% |
| record | career_summary.location | 5 | 0.06% |
| record | administration_context.combination_therapies | 5 | 0.06% |
| record | test_date: Date the test was conducted | 5 | 0.06% |
| record | game_status | 5 | 0.06% |
| record | filer_name | 5 | 0.06% |
| record | time_management | 5 | 0.06% |
| record | overall_score | 5 | 0.06% |
| record | recipient_organization | 5 | 0.06% |
| record | headquarters.city | 5 | 0.06% |
| record | costperpatient | 5 | 0.06% |
| record | away_team: Team playing away from home | 5 | 0.06% |
| record | discharge_diagnosis | 5 | 0.06% |
| record | discharge_instructions | 5 | 0.06% |
| record | historical_context | 5 | 0.06% |
| record | sexual_act.thrusting.vaginal_chamber_formation | 5 | 0.06% |
| record | sexual_act.thrusting.lubrication | 5 | 0.06% |
| record | sexual_act.thrusting.cervix_transformation.elongation | 5 | 0.06% |
| record | sexual_act.thrusting.cervix_transformation.trunk_contact | 5 | 0.06% |
| record | sexual_act.ejaculation.semen_pool_formation | 5 | 0.06% |
| record | sexual_act.post_ejaculation.penis_withdrawal | 5 | 0.06% |
| record | sexual_act.post_ejaculation.internal_changes | 5 | 0.06% |
| record | sexual_act.sperm_migration.sperm_departure | 5 | 0.06% |
| record | hotel_experience.suites.0.suite_type | 5 | 0.06% |
| record | description: Brief product feature and function details | 5 | 0.06% |
| record | content_analysis.entry_structure.components | 5 | 0.06% |
| record | publication.research_context.informants.description | 5 | 0.06% |
| record | vaccine | 5 | 0.06% |
| record | cost: Total cost of enrollment in US dollars | 5 | 0.06% |
| record | test_code: Code identifying the specific laboratory test | 5 | 0.06% |
| record | outbreak_event.investigation.findings.agent_details.full_name | 5 | 0.06% |
| record | outbreak_event.public_health_recommendations.target_populations | 5 | 0.06% |
| record | office_hours.2.start_time | 5 | 0.06% |
| record | office_hours.3.start_time | 5 | 0.06% |
| record | contact_email | 5 | 0.06% |
| record | county | 5 | 0.06% |
| record | boulder.start.foot_holds.1.foot | 5 | 0.06% |
| record | boulder.start.foot_holds.0.foot | 5 | 0.06% |
| record | boulder.start.hand_holds.1.hand | 5 | 0.06% |
| record | net_income: Total reported net income in millions | 5 | 0.06% |
| record | entity_name: Company name associated with filing | 5 | 0.06% |
| record | session_code | 5 | 0.06% |
| record | speaker: Name of the person delivering the statement | 5 | 0.06% |
| record | amount: Financial value associated with the policy or proposal | 5 | 0.06% |
| record | performance_reason | 5 | 0.06% |
| record | monthly_price: Subscription cost per month in USD | 5 | 0.06% |
| record | pricing_tier: Subscription level such as monthly or free | 5 | 0.06% |
| record | promotional_campaign | 5 | 0.06% |
| record | audience_age_range | 5 | 0.06% |
| record | product_list_id | 5 | 0.06% |
| record | supply_quantity: Total number of tablets dispensed | 5 | 0.06% |
| record | person.name_alternative | 5 | 0.06% |
| record | person.experience.0.end_year | 5 | 0.06% |
| record | historical_context.time_periods.2.events.0.measures | 5 | 0.06% |
| record | historical_context.time_periods.3.events.0.medical_advancements | 5 | 0.06% |
| record | historical_context.time_periods.1.events.0.event_type | 5 | 0.06% |
| record | staff.teaching_staff_count | 5 | 0.06% |
| record | departments | 5 | 0.06% |
| record | property_id | 5 | 0.06% |
| record | contact_method | 5 | 0.06% |
| record | hotel.location.area_description | 5 | 0.06% |
| record | hotel.location.nearby_landmarks | 5 | 0.06% |
| record | hotel.transportation.parking.1.name | 5 | 0.06% |
| record | count | 5 | 0.06% |
| record | transaction_type: Mode of transaction such as online or in-person | 5 | 0.06% |
| record | classification.source | 5 | 0.06% |
| record | Priest <> brought | 5 | 0.06% |
| record | Steinem <> front paws on | 5 | 0.06% |
| record | Steinem <> rode in | 5 | 0.06% |
| record | defendant_name | 5 | 0.06% |
| record | event_date: Date when the event will take place | 5 | 0.06% |
| record | event_code: Unique code identifying the event | 5 | 0.06% |
| record | breastfeeding_situation.causes | 5 | 0.06% |
| record | countries.1.name | 5 | 0.06% |
| record | renewal_period | 5 | 0.06% |
| record | study.author.affiliation | 5 | 0.06% |
| record | study.title | 5 | 0.06% |
| record | study.key_findings | 5 | 0.06% |
| record | personnel.0.title | 5 | 0.06% |
| record | certification.prerequisites | 5 | 0.06% |
| record | certification.description | 5 | 0.06% |
| record | treatment.type | 5 | 0.06% |
| record | diagnosis: ICD-10 code representing the medical condition | 5 | 0.06% |
| record | era_name | 5 | 0.06% |
| record | announcement.divisions.0.star_of_the_week.team | 5 | 0.06% |
| record | announcement.divisions.0.honorable_mentions.1.player_name | 5 | 0.06% |
| record | announcement.divisions.0.honorable_mentions.2.player_name | 5 | 0.06% |
| record | announcement.divisions.0.honorable_mentions.5.team | 5 | 0.06% |
| record | announcement.divisions.0.honorable_mentions.6.player_name | 5 | 0.06% |
| record | announcement.divisions.0.honorable_mentions.7.position | 5 | 0.06% |
| record | announcement.divisions.1.star_of_the_week.position | 5 | 0.06% |
| record | announcement.divisions.1.second_star.position | 5 | 0.06% |
| record | announcement.divisions.2.honorable_mentions.4.player_name | 5 | 0.06% |
| record | announcement.divisions.3.division_name | 5 | 0.06% |
| record | announcement.divisions.4.second_star.team | 5 | 0.06% |
| record | announcement.divisions.4.honorable_mentions.1.player_name | 5 | 0.06% |
| record | announcement.divisions.5.honorable_mentions.1.position | 5 | 0.06% |
| record | announcement.divisions.5.honorable_mentions.1.team | 5 | 0.06% |
| record | mechanism | 5 | 0.06% |
| record | business_focus: Core service or product area of the company | 5 | 0.06% |
| record | athlete | 5 | 0.06% |
| record | disease_profile.differential_diagnoses.1.overlapping_features | 5 | 0.06% |
| record | case_report.clinical_presentation.lesion_description.0.features | 5 | 0.06% |
| record | disease_profile.differential_diagnoses.0.overlapping_features | 5 | 0.06% |
| record | goals_scored | 5 | 0.06% |
| record | available_quantity | 5 | 0.06% |
| record | movie.comments.2.time | 5 | 0.06% |
| record | movie.comments.11.date | 5 | 0.06% |
| record | movie.comments.10.date | 5 | 0.06% |
| record | movie.comments.9.text | 5 | 0.06% |
| record | movie.actors | 5 | 0.06% |
| record | movie.comments.18.time | 5 | 0.06% |
| record | movie.comments.17.text | 5 | 0.06% |
| record | movie.comments.16.date | 5 | 0.06% |
| record | movie.comments.19.date | 5 | 0.06% |
| record | movie.comments.5.date | 5 | 0.06% |
| record | due_date | 5 | 0.06% |
| record | awayteampoints | 5 | 0.06% |
| record | blood_pressure: Blood pressure in format systolic/diastolic | 5 | 0.06% |
| record | verbs_past_tense | 5 | 0.06% |
| record | adjectives | 5 | 0.06% |
| record | promotional_code | 5 | 0.06% |
| record | company.transformation_date | 5 | 0.06% |
| record | account: Bank account associated with the transaction | 5 | 0.06% |
| record | timestamp: Date and time of transaction occurrence | 5 | 0.06% |
| record | status: Current state of the transaction | 5 | 0.06% |
| record | exchange_rate: Rate used to convert base to target currency | 5 | 0.06% |
| record | subscription_prompt | 5 | 0.06% |
| record | dynasty | 5 | 0.06% |
| record | affiliations | 5 | 0.06% |
| record | bronze_medalist: Name of athlete who won bronze medal | 5 | 0.06% |
| record | reason_for_status | 5 | 0.06% |
| record | exam_type | 5 | 0.06% |
| record | domain.registry.name | 5 | 0.06% |
| record | domain.territory | 5 | 0.06% |
| record | description_length | 5 | 0.06% |
| record | south_yemen_conflict.historical_context.actions_after_invasion | 5 | 0.06% |
| record | south_yemen_conflict.historical_context.casualties_and_prisoners | 5 | 0.06% |
| record | south_yemen_conflict.accusations_against_governments.1.government_name | 5 | 0.06% |
| record | patient_case.clinical_presentation.skin_lesions.characteristics | 5 | 0.06% |
| record | patient_case.clinical_presentation.skin_lesions.description | 5 | 0.06% |
| record | patient_case.clinical_presentation.skin_lesions.symptoms | 5 | 0.06% |
| record | patient_case.genetic_analysis.mutation_details.exon_location | 5 | 0.06% |
| record | disease.subspecies.1.associated_diseases | 5 | 0.06% |
| record | disease.subspecies.2.name | 5 | 0.06% |
| record | disease.pathogenesis.transmission | 5 | 0.06% |
| record | unit: Unit designation such as 4B or 6C | 5 | 0.06% |
| record | candidate_location | 5 | 0.06% |
| record | attendance | 5 | 0.06% |
| record | actions.0.location | 5 | 0.06% |
| record | status: Current status of the exhibitor's participation | 5 | 0.06% |
| record | booth_number: Letter and number designation of the exhibitor's booth | 5 | 0.06% |
| record | cholesterol: Cholesterol level in mg/dL if elevated | 5 | 0.06% |
| record | order_id: Identifier of the original order associated with the return | 5 | 0.06% |
| record | new_departure | 5 | 0.06% |
| record | labels | 5 | 0.06% |
| record | historical_context.cappers_weekly.topic | 5 | 0.06% |
| record | historical_context.cappers_weekly.call_for_articles_year | 5 | 0.06% |
| record | product.preparation.steps | 5 | 0.06% |
| record | contact_information.help | 5 | 0.06% |
| record | patient_count | 5 | 0.06% |
| record | main_character.affiliation | 5 | 0.06% |
| record | social_media_platforms | 5 | 0.06% |
| record | abolished_entity | 5 | 0.06% |
| record | medium | 5 | 0.06% |
| record | status: Current legal status of the patent | 5 | 0.06% |
| record | context | 5 | 0.06% |
| record | assists: Assists recorded by the player | 5 | 0.06% |
| record | claim_amount: Total cost of the medical service billed | 5 | 0.06% |
| record | patient_address: Full address of the patient | 5 | 0.06% |
| record | rating: Star rating out of 5 given by the reviewer | 5 | 0.06% |
| record | initial_symptoms | 5 | 0.06% |
| record | accommodation.options.1.website | 5 | 0.06% |
| record | valueAddedBenefits | 5 | 0.06% |
| record | programStructure.duration | 5 | 0.06% |
| record | availability.year | 5 | 0.06% |
| record | accommodation.options.1.distance | 5 | 0.06% |
| record | programStructure.dailySchedule.1.activities.2.time | 5 | 0.06% |
| record | initial_text.negations | 5 | 0.06% |
| record | town.population.year | 5 | 0.06% |
| record | town.area_sq_km | 5 | 0.06% |
| record | citizen_corps.activities | 5 | 0.06% |
| record | ready_campaign_updates.target_audience | 5 | 0.06% |
| record | ready_campaign_updates.language_note | 5 | 0.06% |
| record | agreement_id | 5 | 0.06% |
| record | open_rate | 5 | 0.06% |
| record | Robert <> advocated | 5 | 0.06% |
| record | Robert <> favored | 5 | 0.06% |
| record | Robert <> interacted with | 5 | 0.06% |
| record | Robert <> observed | 5 | 0.06% |
| record | Robert <> practiced | 5 | 0.06% |
| record | Robert <> was asked by | 5 | 0.06% |
| record | asteroid.etymology.origin | 5 | 0.06% |
| record | hotel.alternative_names | 5 | 0.06% |
| record | root_cause: Identified cause of the outage or under investigation | 5 | 0.06% |
| record | molecular_weight: Molecular weight of the drug in daltons | 5 | 0.06% |
| record | post_status | 5 | 0.06% |
| record | finance_charge: Fee due to unpaid balance from previous cycle | 5 | 0.06% |
| record | transaction_amount: Amount of the individual transaction | 5 | 0.06% |
| record | close_price: Final closing price of the stock | 5 | 0.06% |
| record | booking_options.partners | 5 | 0.06% |
| record | owner_location | 5 | 0.06% |
| record | dispute_status | 5 | 0.06% |
| record | street | 5 | 0.06% |
| record | time: Official race time recorded in the event | 5 | 0.06% |
| record | goal_achievement: Percentage of goals met, from 60% to 100% | 5 | 0.06% |
| record | error.possible_causes | 5 | 0.06% |
| record | start_year: Year when research began | 5 | 0.06% |
| record | codes | 5 | 0.06% |
| record | when_to_seek_help | 5 | 0.06% |
| record | transfer_id | 5 | 0.06% |
| record | playerbirthcity | 5 | 0.06% |
| record | filing_date: Date the patent was filed | 5 | 0.06% |
| record | geography.borders.0.region | 5 | 0.06% |
| record | infrastructure.airports.0.trafficVolume | 5 | 0.06% |
| record | geography.rivers | 5 | 0.06% |
| record | infrastructure.refineries.0.status | 5 | 0.06% |
| record | regionNameBerber | 5 | 0.06% |
| record | book_title: Title of the book being reviewed | 5 | 0.06% |
| record | consignee | 5 | 0.06% |
| record | word_count: Number of words in the abstract | 5 | 0.06% |
| record | invoice_id: Unique identifier for the invoice document | 5 | 0.06% |
| record | description.distinguishing_features | 5 | 0.06% |
| record | description.color | 5 | 0.06% |
| record | taxonomy.subgenus | 5 | 0.06% |
| record | taxonomy.describer.name | 5 | 0.06% |
| record | connectivity | 5 | 0.06% |
| record | distributor_information.capabilities | 5 | 0.06% |
| record | services.maintenance | 5 | 0.06% |
| record | consultation.practice_areas | 5 | 0.06% |
| record | firm_name | 5 | 0.06% |
| record | client_focus | 5 | 0.06% |
| record | genus.subspecies.0.authority | 5 | 0.06% |
| record | RP | 5 | 0.06% |
| record | hotel_address | 5 | 0.06% |
| record | study_overview.disease | 5 | 0.06% |
| record | study_overview.aim | 5 | 0.06% |
| record | treatment_details.pulmonary_resection_types.0.procedure | 5 | 0.06% |
| record | pathological_findings.nodal_involvement_distribution.2.stage | 5 | 0.06% |
| record | pathological_findings.statistical_factors.1.p_value | 5 | 0.06% |
| record | outcomes.survival_rates.two_year_percentage | 5 | 0.06% |
| record | outage_start: Time when the outage began in HH:MM AM/PM format | 5 | 0.06% |
| record | question.keywords | 5 | 0.06% |
| record | company_name: Name of the technology company receiving funding | 5 | 0.06% |
| record | cell_type: Type of cell in which the pathway operates | 5 | 0.06% |
| record | synthesis_context.carboxyl_protection_requirement.reasoning | 5 | 0.06% |
| record | protecting_group.targets.0.protected_functional_group | 5 | 0.06% |
| record | deprotection_process.reagents.0.concentration_level | 5 | 0.06% |
| record | protecting_group.name | 5 | 0.06% |
| record | synthesis_context.intermediates_synthesized.0.components | 5 | 0.06% |
| record | merchant_code | 5 | 0.06% |
| record | document.commissioners.1.portfolio | 5 | 0.06% |
| record | contribution_id: Unique identifier for the contribution record | 5 | 0.06% |
| record | journal.title | 5 | 0.06% |
| record | article.publicationDetails.issue | 5 | 0.06% |
| record | founding_date | 5 | 0.06% |
| record | regulation_referenced | 5 | 0.06% |
| record | customer_id | 4 | 0.05% |
| record | engagement_status | 4 | 0.05% |
| record | app_id | 4 | 0.05% |
| record | size_and_fit.measurements.2.type | 4 | 0.05% |
| record | price.installment_option.installments | 4 | 0.05% |
| record | price.installment_option.installment_amount | 4 | 0.05% |
| record | size_and_fit.length.size | 4 | 0.05% |
| record | size_and_fit.model_size | 4 | 0.05% |
| record | size_and_fit.model_height.value | 4 | 0.05% |
| record | roles | 4 | 0.05% |
| record | biography.education.4.teacher | 4 | 0.05% |
| record | birth_information.birth_place.city | 4 | 0.05% |
| record | biography.concert_tours | 4 | 0.05% |
| record | biography.education.0.teacher | 4 | 0.05% |
| record | biography.awards.0.year | 4 | 0.05% |
| record | biography.collaborations | 4 | 0.05% |
| record | biography.awards.0.organization | 4 | 0.05% |
| record | delay_reason: Human-readable description of delay cause | 4 | 0.05% |
| record | destination: Airport code where flight arrives | 4 | 0.05% |
| record | passenger_names: List of passenger names affected | 4 | 0.05% |
| record | flight_number: Unique identifier for the delayed flight | 4 | 0.05% |
| record | state: State or province where the recipient resides | 4 | 0.05% |
| record | location: Physical location of the briefing event | 4 | 0.05% |
| record | amount_usd: USD amount involved in the investment | 4 | 0.05% |
| record | event.unit.name | 4 | 0.05% |
| record | event.graduation_details.number_of_graduates | 4 | 0.05% |
| record | event.officials_present.0.name | 4 | 0.05% |
| record | event.legal_basis.year | 4 | 0.05% |
| record | hours_of_operation.2.open_time | 4 | 0.05% |
| record | hours_of_operation.6.day | 4 | 0.05% |
| record | hours_of_operation.0.open_time | 4 | 0.05% |
| record | hours_of_operation.1.day | 4 | 0.05% |
| record | hours_of_operation.3.day | 4 | 0.05% |
| record | offerings | 4 | 0.05% |
| record | location.area | 4 | 0.05% |
| record | sourcing.coffee_source.company_name | 4 | 0.05% |
| record | atmosphere | 4 | 0.05% |
| record | hours_of_operation.4.day | 4 | 0.05% |
| record | priority_registration.submission_methods | 4 | 0.05% |
| record | priority_registration.documents_required | 4 | 0.05% |
| record | important_notice | 4 | 0.05% |
| record | on_site_registration.description | 4 | 0.05% |
| record | on_site_registration.note | 4 | 0.05% |
| record | on_site_registration.location.address | 4 | 0.05% |
| record | online_priority_registration.deadline | 4 | 0.05% |
| record | event.teams | 4 | 0.05% |
| record | nhl_officials | 4 | 0.05% |
| record | zebras_care_foundation.beneficiary.condition | 4 | 0.05% |
| record | zebras_care_foundation.beneficiary.treatment | 4 | 0.05% |
| record | zebras_care_foundation.introducer.role | 4 | 0.05% |
| record | notice_date | 4 | 0.05% |
| record | asteroid.orbital_characteristics.orbital_region | 4 | 0.05% |
| record | decision_status: Final outcome of the submission | 4 | 0.05% |
| record | author: Name of the submitting researcher | 4 | 0.05% |
| record | target_region | 4 | 0.05% |
| record | incident.affected_locations | 4 | 0.05% |
| record | incident.charges.2.count | 4 | 0.05% |
| record | incident.additional_acts | 4 | 0.05% |
| record | incident.charges.0.count | 4 | 0.05% |
| record | amount: Quantity of crypto asset bought or sold | 4 | 0.05% |
| record | customerservicerating | 4 | 0.05% |
| record | uptime | 4 | 0.05% |
| record | integration_capabilities | 4 | 0.05% |
| record | booth_number | 4 | 0.05% |
| record | region.landscape | 4 | 0.05% |
| record | region.description | 4 | 0.05% |
| record | region.location.extends_from | 4 | 0.05% |
| record | region.location.extends_to | 4 | 0.05% |
| record | article.cause | 4 | 0.05% |
| record | article.symptoms | 4 | 0.05% |
| record | article.translation_note.english_version.id | 4 | 0.05% |
| record | article.products | 4 | 0.05% |
| record | article.resolution | 4 | 0.05% |
| record | article.translation_note.english_version.url | 4 | 0.05% |
| record | drug: Name and dosage form of the medication taken | 4 | 0.05% |
| record | duration: Number of days the treatment lasted for the reaction | 4 | 0.05% |
| record | call_duration | 4 | 0.05% |
| record | store | 4 | 0.05% |
| record | grant_id | 4 | 0.05% |
| record | project_url | 4 | 0.05% |
| record | hotel.location.district | 4 | 0.05% |
| record | founders | 4 | 0.05% |
| record | subscription_amount | 4 | 0.05% |
| record | score: Final score of the match in set format | 4 | 0.05% |
| record | match_round: Stage of the tournament such as final, quarterfinal, or last 16 | 4 | 0.05% |
| record | tier: Subscription tier level such as basic or premium | 4 | 0.05% |
| record | city: City where the subscriber resides | 4 | 0.05% |
| record | subscription_id: Unique identifier for the subscription confirmation | 4 | 0.05% |
| record | delivery_frequency: How often the newsletter is delivered | 4 | 0.05% |
| record | monitoring_method | 4 | 0.05% |
| record | location: Venue where the match took place | 4 | 0.05% |
| record | score: List of two integers representing home and away scores | 4 | 0.05% |
| record | yes_votes | 4 | 0.05% |
| record | vote_margin | 4 | 0.05% |
| record | measure | 4 | 0.05% |
| record | required_experience | 4 | 0.05% |
| record | reviewer_address | 4 | 0.05% |
| record | title: Title of the proposed solution | 4 | 0.05% |
| record | shipping_cost: Cost to ship the audio device in US dollars | 4 | 0.05% |
| record | weight: Weight of the device in grams | 4 | 0.05% |
| record | rsvp_count | 4 | 0.05% |
| record | job_id | 4 | 0.05% |
| record | degree | 4 | 0.05% |
| record | leagues | 4 | 0.05% |
| record | campaign_status: Current status of the campaign active, in review, paused | 4 | 0.05% |
| record | category: Type of beauty product such as serum, foundation, oil, balm | 4 | 0.05% |
| record | record labels <> surprised by | 4 | 0.05% |
| record | Ivan Suvanjieff <> founder of | 4 | 0.05% |
| record | consumers <> surprised by | 4 | 0.05% |
| record | Andrew Bulkeley <> attended | 4 | 0.05% |
| record | Frontier <> local to | 4 | 0.05% |
| record | Amazon <> introduced | 4 | 0.05% |
| record | Amendment 64 <> relates to | 4 | 0.05% |
| record | application_type | 4 | 0.05% |
| record | registration_number: Unique number assigned to the trademark registration | 4 | 0.05% |
| record | status: Direction of price movement | 4 | 0.05% |
| record | symbol: Unique ticker identifier for the company | 4 | 0.05% |
| record | location: Geographic location where the incident took place | 4 | 0.05% |
| record | date: Timestamp when the incident occurred | 4 | 0.05% |
| record | source: Media or organization reporting the incident | 4 | 0.05% |
| record | circular.target_audience | 4 | 0.05% |
| record | circular.subject | 4 | 0.05% |
| record | performance_time | 4 | 0.05% |
| record | match_duration | 4 | 0.05% |
| record | player1_seed | 4 | 0.05% |
| record | target_audience: Primary user group for the product | 4 | 0.05% |
| record | features: List of core functionality features | 4 | 0.05% |
| record | training_modules: List of specific training topics assigned to the employee | 4 | 0.05% |
| record | value_original | 4 | 0.05% |
| record | early_bird_deadline | 4 | 0.05% |
| record | po_reference | 4 | 0.05% |
| record | region.language | 4 | 0.05% |
| record | region.agriculture | 4 | 0.05% |
| record | lead_reporter: Reporter providing coverage of the incident | 4 | 0.05% |
| record | legal_agent | 4 | 0.05% |
| record | modification_flag | 4 | 0.05% |
| record | approval_level | 4 | 0.05% |
| record | compliance_code | 4 | 0.05% |
| record | notice_period_days | 4 | 0.05% |
| record | audit_review_date | 4 | 0.05% |
| record | counterparty_address | 4 | 0.05% |
| record | contract_reference | 4 | 0.05% |
| record | notice_type: Method used to notify policyholder of renewal | 4 | 0.05% |
| record | reporter_company: Company name of the customer reporter | 4 | 0.05% |
| record | assigned_agent: Name of the agent handling the ticket | 4 | 0.05% |
| record | client_address | 4 | 0.05% |
| record | distance.calculation_method | 4 | 0.05% |
| record | language: List of languages the publication is available in | 4 | 0.05% |
| record | funding_body: Organization funding the publication | 4 | 0.05% |
| record | team1: First team participating in the match | 4 | 0.05% |
| record | outlet | 4 | 0.05% |
| record | amount: Monetary value associated with the outreach | 4 | 0.05% |
| record | other_investors: List of additional participating investors | 4 | 0.05% |
| record | location.type | 4 | 0.05% |
| record | location.distance_from_capital.unit | 4 | 0.05% |
| record | location.distance_from_capital.capital_city | 4 | 0.05% |
| record | track name | 4 | 0.05% |
| record | console | 4 | 0.05% |
| record | game series | 4 | 0.05% |
| record | color: Color of the product | 4 | 0.05% |
| record | price: Price in dollars without currency symbol | 4 | 0.05% |
| record | battery_life: Battery life in hours as specified | 4 | 0.05% |
| record | symptom_fever | 4 | 0.05% |
| record | weight_loss_kg | 4 | 0.05% |
| record | sleep_hours | 4 | 0.05% |
| record | lab_tests_ordered | 4 | 0.05% |
| record | circular.ministry | 4 | 0.05% |
| record | location: Property address including street, city, and state | 4 | 0.05% |
| record | limitations | 4 | 0.05% |
| record | regulatory_compliance: List of relevant regulatory standards and codes | 4 | 0.05% |
| record | hazard_class: Primary hazard classification of the chemical | 4 | 0.05% |
| record | department: Department where the employee is assigned | 4 | 0.05% |
| record | chipdesignfocus | 4 | 0.05% |
| record | opening_quarter | 4 | 0.05% |
| record | outcomes_and_findings.prognosis.critical_success_factors | 4 | 0.05% |
| record | pathophysiology.etiology | 4 | 0.05% |
| record | pathophysiology.mechanism.secondary_event.anatomical_locations | 4 | 0.05% |
| record | outcomes_and_findings.prognosis.visual_recovery_potential | 4 | 0.05% |
| record | shift | 4 | 0.05% |
| record | institution: Name of the financial institution publishing the rate | 4 | 0.05% |
| record | effective_time: Local time when the rate was effective | 4 | 0.05% |
| record | rate: Exchange rate value for the currency pair | 4 | 0.05% |
| record | validation_status: Current validation state of the rate data | 4 | 0.05% |
| record | controversy.vikram_response | 4 | 0.05% |
| record | controversy.person_involved.2.name | 4 | 0.05% |
| record | controversy.victim_actresses.7.krk_comment | 4 | 0.05% |
| record | controversy.person_involved.0.role | 4 | 0.05% |
| record | sec_id | 4 | 0.05% |
| record | event_id: Unique identifier for each event | 4 | 0.05% |
| record | location: Physical address or venue of the event | 4 | 0.05% |
| record | status: Current phase of the event lifecycle | 4 | 0.05% |
| record | buyer_name | 4 | 0.05% |
| record | event.location.region | 4 | 0.05% |
| record | event.surgery_date | 4 | 0.05% |
| record | event.target | 4 | 0.05% |
| record | asteroid.named_after.occupation | 4 | 0.05% |
| record | payee: Name of the recipient or entity involved | 4 | 0.05% |
| record | department: Department associated with the ticket | 4 | 0.05% |
| record | ticket_id: Unique identifier for the ticket in the system | 4 | 0.05% |
| record | warehouse_location: Geographic warehouse where product is stored | 4 | 0.05% |
| record | reason_for_difference | 4 | 0.05% |
| record | legislative_details.initiators | 4 | 0.05% |
| record | brain part | 4 | 0.05% |
| record | email: Contact email address of the ticket submitter | 4 | 0.05% |
| record | department: Government department or agency where the appointment occurs | 4 | 0.05% |
| record | lab_results: List of lab values and results | 4 | 0.05% |
| record | contribution_type | 4 | 0.05% |
| record | csds_id | 4 | 0.05% |
| record | manufacturer_address | 4 | 0.05% |
| record | path_design.concrete_tile_making.mold_size_example | 4 | 0.05% |
| record | path_design.concrete_tile_making.steps | 4 | 0.05% |
| record | path_design.materials | 4 | 0.05% |
| record | path_design.concrete_tile_making.design_options | 4 | 0.05% |
| record | pilot_hubs | 4 | 0.05% |
| record | briefing_event | 4 | 0.05% |
| record | sale_amount | 4 | 0.05% |
| record | sales_platform | 4 | 0.05% |
| record | total_funding | 4 | 0.05% |
| record | person.publication_information.commentary_with | 4 | 0.05% |
| record | person.time_period.era | 4 | 0.05% |
| record | person.publication_information.available_part | 4 | 0.05% |
| record | person.publication_information.publisher | 4 | 0.05% |
| record | person.publication_information.location | 4 | 0.05% |
| record | scope: Broad category or focus area of the bill | 4 | 0.05% |
| record | yes_votes: Number of members voting in favor | 4 | 0.05% |
| record | characteristics.diameter.unit | 4 | 0.05% |
| record | defendant: Individual or entity being sued | 4 | 0.05% |
| record | diameter | 4 | 0.05% |
| record | restaurant.description | 4 | 0.05% |
| record | restaurant.renovations.0.year | 4 | 0.05% |
| record | restaurant.establishment_date | 4 | 0.05% |
| record | restaurant.events | 4 | 0.05% |
| record | product.intended_use | 4 | 0.05% |
| record | product.technology | 4 | 0.05% |
| record | vehicle.type | 4 | 0.05% |
| record | vehicle.operators.0.acquisition.funding_details | 4 | 0.05% |
| record | vehicle.name | 4 | 0.05% |
| record | vehicle.specifications.speed | 4 | 0.05% |
| record | vehicle.specifications.full_weight | 4 | 0.05% |
| record | vehicle.specifications.fording_depth | 4 | 0.05% |
| record | vehicle.specifications.dimensions.length | 4 | 0.05% |
| record | vehicle.operators.0.acquisition.cost | 4 | 0.05% |
| record | prior_experience | 4 | 0.05% |
| record | appointee_id | 4 | 0.05% |
| record | trial_phase | 4 | 0.05% |
| record | winner_team | 4 | 0.05% |
| record | research.experiment_details.setup | 4 | 0.05% |
| record | research.experiment_details.speckle_details | 4 | 0.05% |
| record | research.researcher.name | 4 | 0.05% |
| record | missions | 4 | 0.05% |
| record | salary.monthly_estimate.hours_per_week | 4 | 0.05% |
| record | total_interactions | 4 | 0.05% |
| record | company.contact_information.fax_numbers | 4 | 0.05% |
| record | company.address.building_number | 4 | 0.05% |
| record | company.address.district | 4 | 0.05% |
| record | department: Department where the issue originated | 4 | 0.05% |
| record | event.stage_count | 4 | 0.05% |
| record | event.finish_time | 4 | 0.05% |
| record | drivers.1.nationality | 4 | 0.05% |
| record | drivers.1.name | 4 | 0.05% |
| record | team.vehicle.make | 4 | 0.05% |
| record | team.vehicle.fuel_type | 4 | 0.05% |
| record | person.biography.early_life.family.siblings.sisters | 4 | 0.05% |
| record | person.biography.early_life.family.siblings.brothers | 4 | 0.05% |
| record | person.biography.marriage.spouse_father | 4 | 0.05% |
| record | person.biography.political_involvement.description | 4 | 0.05% |
| record | person.biography.early_life.family.status | 4 | 0.05% |
| record | person.works.0.description | 4 | 0.05% |
| record | communication system | 4 | 0.05% |
| record | military unit | 4 | 0.05% |
| record | price.installment_option.currency | 4 | 0.05% |
| record | size_and_fit.details | 4 | 0.05% |
| record | assignee: Entity owning the patent application | 4 | 0.05% |
| record | classification_codes: List of classification codes assigned to the patent | 4 | 0.05% |
| record | retail_price | 4 | 0.05% |
| record | customer_satisfaction | 4 | 0.05% |
| record | author_information.family.pets.0.name | 4 | 0.05% |
| record | author_information.location.state | 4 | 0.05% |
| record | model: Unique identifier or model number of the device | 4 | 0.05% |
| record | rating: Star rating given by the reviewer out of 5 | 4 | 0.05% |
| record | price: Purchase price of the device in dollars | 4 | 0.05% |
| record | status: Current status of the review | 4 | 0.05% |
| record | date: Date of transaction occurrence | 4 | 0.05% |
| record | name: Full name of the transaction participant | 4 | 0.05% |
| record | imaging_date | 4 | 0.05% |
| record | disease_context | 4 | 0.05% |
| record | provider_npi | 4 | 0.05% |
| record | demographics.source | 4 | 0.05% |
| record | bordering_municipalities | 4 | 0.05% |
| record | cfo | 4 | 0.05% |
| record | guidance | 4 | 0.05% |
| record | discovery_observatory | 4 | 0.05% |
| record | discovery_location | 4 | 0.05% |
| record | dosage: Amount of medication prescribed | 4 | 0.05% |
| record | team_mentions.0.name | 4 | 0.05% |
| record | insurance: Insurance provider covering the bill | 4 | 0.05% |
| record | treatment: Medical treatment provided during stay | 4 | 0.05% |
| record | storm_id | 4 | 0.05% |
| record | power_outage | 4 | 0.05% |
| record | booking_options.partnerships | 4 | 0.05% |
| record | resolution_team | 4 | 0.05% |
| record | hire_date: Date when the position is expected to be filled | 4 | 0.05% |
| record | job_title: Title of the position being advertised | 4 | 0.05% |
| record | student_id | 4 | 0.05% |
| record | hanja_name | 4 | 0.05% |
| record | details.uniform.belt_color | 4 | 0.05% |
| record | details.uniform.belt_type | 4 | 0.05% |
| record | creation.king | 4 | 0.05% |
| record | user: Name of the user who filed the ticket | 4 | 0.05% |
| record | location.name | 4 | 0.05% |
| record | location.zone | 4 | 0.05% |
| record | stockstatus | 4 | 0.05% |
| record | rating: Average customer review score out of 5 | 4 | 0.05% |
| record | review_count: Total number of customer reviews received | 4 | 0.05% |
| record | explanation | 4 | 0.05% |
| record | rma_number | 4 | 0.05% |
| record | status: Current operational status of the campaign | 4 | 0.05% |
| record | urgency | 4 | 0.05% |
| record | founding_year: Year the startup was founded | 4 | 0.05% |
| record | funding_source: Type of funding such as grant, loan, venture fund, crowdfunding | 4 | 0.05% |
| record | location: City and state where the startup was launched | 4 | 0.05% |
| record | review_category | 4 | 0.05% |
| record | name: Full name of the enrolled individual | 4 | 0.05% |
| record | program: Training or program name enrolled in | 4 | 0.05% |
| record | id: Unique identifier for the enrollment record | 4 | 0.05% |
| record | headliner | 4 | 0.05% |
| record | policy_reference | 4 | 0.05% |
| record | status: Current status of the exam result | 4 | 0.05% |
| record | subject: Subject area of the exam | 4 | 0.05% |
| record | description_snippet | 4 | 0.05% |
| record | siblings.0.name | 4 | 0.05% |
| record | siblings.0.location | 4 | 0.05% |
| record | siblings.0.relationship | 4 | 0.05% |
| record | great_grandnieces_and_nephews | 4 | 0.05% |
| record | employment.3.company | 4 | 0.05% |
| record | employment.4.company | 4 | 0.05% |
| record | nieces | 4 | 0.05% |
| record | grandchildren | 4 | 0.05% |
| record | predeceased.3.name | 4 | 0.05% |
| record | predeceased.5.relationship | 4 | 0.05% |
| record | predeceased.6.name | 4 | 0.05% |
| record | predeceased.6.relationship | 4 | 0.05% |
| record | predeceased.7.name | 4 | 0.05% |
| record | predeceased.7.relationship | 4 | 0.05% |
| record | siblings_in_law.0.location | 4 | 0.05% |
| record | siblings_in_law.0.relationship | 4 | 0.05% |
| record | siblings_in_law.1.name | 4 | 0.05% |
| record | siblings_in_law.1.location | 4 | 0.05% |
| record | siblings_in_law.1.relationship | 4 | 0.05% |
| record | siblings_in_law.2.relationship | 4 | 0.05% |
| record | siblings_in_law.3.location | 4 | 0.05% |
| record | children.0.name | 4 | 0.05% |
| record | study_metadata.study_type | 4 | 0.05% |
| record | population.age_range.unit | 4 | 0.05% |
| record | follow_up.average_duration.months | 4 | 0.05% |
| record | outcomes.data_source | 4 | 0.05% |
| record | statistical_results.hazard_ratios.0.value | 4 | 0.05% |
| record | statistical_results.diagnostic_metrics.sensitivity_at_threshold.percentage_of_fractures | 4 | 0.05% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.0.t_score | 4 | 0.05% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.1.rate_percentage | 4 | 0.05% |
| record | range | 4 | 0.05% |
| record | ordering_doctor | 4 | 0.05% |
| record | location: Physical address of the affected entity | 4 | 0.05% |
| record | co_authors | 4 | 0.05% |
| record | policyholder: Name of the insured individual | 4 | 0.05% |
| record | title: Full title of the clinical guideline document | 4 | 0.05% |
| record | recommendations: Number of specific clinical recommendations included | 4 | 0.05% |
| record | version: Current active version of the guideline | 4 | 0.05% |
| record | cost_estimate: Annual cost to implement the guideline in a region | 4 | 0.05% |
| record | audience: Professional groups advised by the guideline | 4 | 0.05% |
| record | origin: Country or region where the guideline was issued | 4 | 0.05% |
| record | monitoring_focus: Key adverse effects or outcomes to monitor during implementation | 4 | 0.05% |
| record | name: Full name of the appointed individual | 4 | 0.05% |
| record | position: Job title or role within the department | 4 | 0.05% |
| record | date: Start date of the appointment | 4 | 0.05% |
| record | compensation: Annual salary or compensation amount | 4 | 0.05% |
| record | notes: Additional context about the appointment or background | 4 | 0.05% |
| record | tissue_context | 4 | 0.05% |
| record | tags: List of keywords, hashtags, or product mentions from the description | 4 | 0.05% |
| record | splicing_regulation.impact_on_pathway_components.0.component_description | 4 | 0.05% |
| record | splicing_regulation.impact_on_pathway_components.0.regulatory_mode | 4 | 0.05% |
| record | splicing_regulation.pathological_outcomes | 4 | 0.05% |
| record | WTS Atlas <> managed by | 4 | 0.05% |
| record | lead_investor: Primary investor or consortium leading the round | 4 | 0.05% |
| record | patient_name: Full name of the patient completing the checklist | 4 | 0.05% |
| record | symptoms: List of symptoms reported by the patient | 4 | 0.05% |
| record | cpt_code: CPT code associated with the clinical visit type | 4 | 0.05% |
| record | contribution_id: Unique identifier for each contribution entry | 4 | 0.05% |
| record | status: Current status of the contribution record | 4 | 0.05% |
| record | campaign_target: Name of the candidate or office being supported | 4 | 0.05% |
| record | publication_id: Unique identifier for the exam publication | 4 | 0.05% |
| record | usage_context | 4 | 0.05% |
| record | sunless_tanning.product_collection.products.0.type | 4 | 0.05% |
| record | sunless_tanning.product_collection.products.0.ingredients | 4 | 0.05% |
| record | sunless_tanning.product_collection.products.0.description | 4 | 0.05% |
| record | sunless_tanning.product_collection.benefits | 4 | 0.05% |
| record | sunless_tanning.product_collection.products.1.name | 4 | 0.05% |
| record | passing_rate | 4 | 0.05% |
| record | geographic_location | 4 | 0.05% |
| record | targetgradelevel | 4 | 0.05% |
| record | funding_code | 4 | 0.05% |
| record | launch_location: City and state where the product was launched | 4 | 0.05% |
| record | product_name: Name of the launched skincare or apparel product | 4 | 0.05% |
| record | price: Selling price of the product in US dollars | 4 | 0.05% |
| record | creation.year | 4 | 0.05% |
| record | rank_level | 4 | 0.05% |
| record | date_posted | 4 | 0.05% |
| record | revenue: Total revenue in billions as reported | 4 | 0.05% |
| record | baggage_count | 4 | 0.05% |
| record | arrival_time | 4 | 0.05% |
| record | passenger_type | 4 | 0.05% |
| record | outcomes.comparative_advantage.alternative_to | 4 | 0.05% |
| record | research_context.problem_statement.limitation | 4 | 0.05% |
| record | outcomes.comparative_advantage.benefit_type | 4 | 0.05% |
| record | content.warning | 4 | 0.05% |
| record | content.core_beliefs.relationship_with_society | 4 | 0.05% |
| record | login.data_required | 4 | 0.05% |
| record | registration.steps | 4 | 0.05% |
| record | application_name | 4 | 0.05% |
| record | music.genres | 4 | 0.05% |
| record | registration.benefits | 4 | 0.05% |
| record | hotel_name: Name of the hotel as mentioned in the review | 4 | 0.05% |
| record | related_terms | 4 | 0.05% |
| record | discrepancy_amount | 4 | 0.05% |
| record | damage_amount | 4 | 0.05% |
| record | causeofloss | 4 | 0.05% |
| record | filing_amount | 4 | 0.05% |
| record | research_context.problem_statement.incompatible_substrates | 4 | 0.05% |
| record | methodology.experimental_setup.reagents.0.type | 4 | 0.05% |
| record | regions: Geographic areas where the product is distributed or in use | 4 | 0.05% |
| record | base_price: Original retail price of the product | 4 | 0.05% |
| record | features: List of key technological or functional capabilities | 4 | 0.05% |
| record | orbital_characteristics.database | 4 | 0.05% |
| record | physical_characteristics.observations.0.telescope_type | 4 | 0.05% |
| record | physical_characteristics.observations.0.year_launched | 4 | 0.05% |
| record | weight_loss_benefits.improvements | 4 | 0.05% |
| record | nutritional_aspects.carbohydrates.low_glycemic_index.recommendation | 4 | 0.05% |
| record | promotions | 4 | 0.05% |
| record | participants | 4 | 0.05% |
| record | diagnosis: Medical diagnosis code and description | 4 | 0.05% |
| record | representatives.0.name | 4 | 0.05% |
| record | representatives.0.role | 4 | 0.05% |
| record | vehicle model | 4 | 0.05% |
| record | year range | 4 | 0.05% |
| record | territory | 4 | 0.05% |
| record | person name | 4 | 0.05% |
| record | morphology.flower_heads | 4 | 0.05% |
| record | morphology.stem | 4 | 0.05% |
| record | distribution.regions.1.countries | 4 | 0.05% |
| record | policy_id: Identifier assigned to the insurance policy | 4 | 0.05% |
| record | research_interests | 4 | 0.05% |
| record | highlights | 4 | 0.05% |
| record | infection_method | 4 | 0.05% |
| record | affected_browsers | 4 | 0.05% |
| record | removal_instructions.browser_specific_instructions.0.steps | 4 | 0.05% |
| record | malware_name | 4 | 0.05% |
| record | removal_instructions.browser_specific_instructions.1.steps | 4 | 0.05% |
| record | followupclinic_address: Physical address of follow-up clinic | 4 | 0.05% |
| record | treatments: Prescribed medications or therapies during stay | 4 | 0.05% |
| record | reporting_period | 4 | 0.05% |
| record | genus.related_genera | 4 | 0.05% |
| record | genus.etymology.part2.meaning | 4 | 0.05% |
| record | genus.publication | 4 | 0.05% |
| record | genus.characteristics | 4 | 0.05% |
| record | genus.distribution | 4 | 0.05% |
| record | event_id: Conference or symposium identifier for the accepted abstract | 4 | 0.05% |
| record | acquisition_year | 4 | 0.05% |
| record | asteroid.orbital_period.orbiting | 4 | 0.05% |
| record | media_analysis.criticisms | 4 | 0.05% |
| record | media_analysis.events.0.description | 4 | 0.05% |
| record | narrative.protagonist.description | 4 | 0.05% |
| record | narrative.protagonist.possessions | 4 | 0.05% |
| record | media_analysis.comparison_national_media | 4 | 0.05% |
| record | account_code | 4 | 0.05% |
| record | location: Geographic location of the reviewer | 4 | 0.05% |
| record | review.cons | 4 | 0.05% |
| record | post_id: Unique identifier for the TikTok post | 4 | 0.05% |
| record | location: City and state or country where posted | 4 | 0.05% |
| record | status: Current status of the post based on platform action | 4 | 0.05% |
| record | device_id: Unique identifier for the medical device as listed in records | 4 | 0.05% |
| record | device_type: Type of medical device such as ECG or infusion pump | 4 | 0.05% |
| record | availability_status | 4 | 0.05% |
| record | study_details.design | 4 | 0.05% |
| record | device model | 4 | 0.05% |
| record | developer | 4 | 0.05% |
| record | delay_minutes: Duration of delay in minutes | 4 | 0.05% |
| record | submitter: Name of employee or partner who reported the incident | 4 | 0.05% |
| record | sender_domain: Domain used in the fraudulent email | 4 | 0.05% |
| record | description: Purpose or label of the transaction | 4 | 0.05% |
| record | weapon | 4 | 0.05% |
| record | first_contact_resolution_rate | 4 | 0.05% |
| record | impact_percentage | 4 | 0.05% |
| record | person2 | 4 | 0.05% |
| record | amount_requested: Total amount requested for compensation | 4 | 0.05% |
| record | policyholder_name: Name of the policyholder filing the claim | 4 | 0.05% |
| record | score: Final score in format home-away | 4 | 0.05% |
| record | early_challenge | 4 | 0.05% |
| record | report_id: Unique identifier assigned to the adverse event report | 4 | 0.05% |
| record | patient_gender: Gender of the patient | 4 | 0.05% |
| record | patient_name: Name of the affected individual as reported | 4 | 0.05% |
| record | recipient: Name of the student receiving the award | 4 | 0.05% |
| record | status: Current status of the award | 4 | 0.05% |
| record | definition.source_organization.full_name | 4 | 0.05% |
| record | definition.document_title | 4 | 0.05% |
| record | personal_sustainability | 4 | 0.05% |
| record | industry_example.challenges | 4 | 0.05% |
| record | industry_example.status | 4 | 0.05% |
| record | three_pillars | 4 | 0.05% |
| record | benefits | 4 | 0.05% |
| record | crisis_resilience | 4 | 0.05% |
| record | enrollmentstatus | 4 | 0.05% |
| record | steps | 4 | 0.05% |
| record | performance_rating: Numerical score from 3.0 to 5.0 | 4 | 0.05% |
| record | name: Full name of the employee | 4 | 0.05% |
| record | exit_status | 4 | 0.05% |
| record | investment_source | 4 | 0.05% |
| record | content: Text or description of the post content | 4 | 0.05% |
| record | post_id: Unique identifier assigned to the post | 4 | 0.05% |
| record | specific_outcomes.postoperative_cholangiography_rate | 4 | 0.05% |
| record | keynote_sessions | 4 | 0.05% |
| record | lightning_talks | 4 | 0.05% |
| record | educational_focus | 4 | 0.05% |
| record | post_conference_event | 4 | 0.05% |
| record | core_values | 4 | 0.05% |
| record | registration_type | 4 | 0.05% |
| record | panel_discussions | 4 | 0.05% |
| record | history.creation.laid_down_date | 4 | 0.05% |
| record | ship_name | 4 | 0.05% |
| record | expression_units | 4 | 0.05% |
| record | system_code | 4 | 0.05% |
| record | coverage_limit | 4 | 0.05% |
| record | finding_id | 4 | 0.05% |
| record | album.tracklist.cd.10.track_number | 4 | 0.05% |
| record | album.free_concert_location | 4 | 0.05% |
| record | album.sources.0.source_name | 4 | 0.05% |
| record | album.tracklist.cd.6.track_number | 4 | 0.05% |
| record | album.release_date | 4 | 0.05% |
| record | album.versions.1.contents | 4 | 0.05% |
| record | album.tracklist.cd.3.track_number | 4 | 0.05% |
| record | album.tracklist.cd.9.track_number | 4 | 0.05% |
| record | album.versions.0.type | 4 | 0.05% |
| record | album.tracklist.cd.2.title | 4 | 0.05% |
| record | album.tracklist.cd.8.title | 4 | 0.05% |
| record | album.external_links.0.url | 4 | 0.05% |
| record | album.tracklist.cd.2.tie_in.0.title | 4 | 0.05% |
| record | album.artist | 4 | 0.05% |
| record | album.tracklist.cd.1.title | 4 | 0.05% |
| record | album.tracklist.cd.8.tie_in.0.type | 4 | 0.05% |
| record | album.tracklist.cd.2.track_number | 4 | 0.05% |
| record | album.sources.2.reference_number | 4 | 0.05% |
| record | album.sources.3.url | 4 | 0.05% |
| record | album.chart_performance.chart_type | 4 | 0.05% |
| record | album.tracklist.cd.7.lyrics | 4 | 0.05% |
| record | album.sources.1.reference_number | 4 | 0.05% |
| record | album.tracklist.cd.8.notes | 4 | 0.05% |
| record | album.versions.2.name | 4 | 0.05% |
| record | album.anniversary | 4 | 0.05% |
| record | album.tracklist.cd.3.notes | 4 | 0.05% |
| record | album.tracklist.cd.3.composer | 4 | 0.05% |
| record | album.versions.1.type | 4 | 0.05% |
| record | album.chart_performance.ranking | 4 | 0.05% |
| record | other_collaborations | 4 | 0.05% |
| record | actor.name | 4 | 0.05% |
| record | movie.year | 4 | 0.05% |
| record | source.via | 4 | 0.05% |
| record | review.publication | 4 | 0.05% |
| record | product_sku | 4 | 0.05% |
| record | onboarding_stage | 4 | 0.05% |
| record | case_number | 4 | 0.05% |
| record | general_admission | 4 | 0.05% |
| record | cultural_references.0.medium | 4 | 0.05% |
| record | cultural_references.0.year | 4 | 0.05% |
| record | host_star.name | 4 | 0.05% |
| record | host_star.type | 4 | 0.05% |
| record | discovery.telescopes.1.name | 4 | 0.05% |
| record | lease_end | 4 | 0.05% |
| record | clinical_use: Primary medical application of the drug | 4 | 0.05% |
| record | status: Current regulatory or development status | 4 | 0.05% |
| record | drug_name: Name of the pharmaceutical compound | 4 | 0.05% |
| record | molecular_target: Biological molecule the drug interacts with | 4 | 0.05% |
| record | airport.name | 4 | 0.05% |
| record | airport.operator | 4 | 0.05% |
| record | terminal.closure_reason | 4 | 0.05% |
| record | terminal.impact.cargo_terminal_delay.delay_years | 4 | 0.05% |
| record | terminal.history.1.period | 4 | 0.05% |
| record | terminal.history.2.description | 4 | 0.05% |
| record | terminal.history.3.description | 4 | 0.05% |
| record | explanations | 4 | 0.05% |
| record | status: Current campaign status | 4 | 0.05% |
| record | characteristics.effective_temperature.value | 4 | 0.05% |
| record | call_status | 4 | 0.05% |
| record | delivery_address | 4 | 0.05% |
| record | subscription_type | 4 | 0.05% |
| record | billing_frequency | 4 | 0.05% |
| record | dobj | 4 | 0.05% |
| record | appos | 4 | 0.05% |
| record | acl | 4 | 0.05% |
| record | exploited_ports | 4 | 0.05% |
| record | software_versions | 4 | 0.05% |
| record | action_items | 4 | 0.05% |
| record | affected_streets | 4 | 0.05% |
| record | affected_homes | 4 | 0.05% |
| record | review_text: Full textual critique of the film | 4 | 0.05% |
| record | address: Full street address including suite and city | 4 | 0.05% |
| record | name: Official name of the business unit operating the office | 4 | 0.05% |
| record | staff_count: Number of employees currently working at the office | 4 | 0.05% |
| record | person.publications.0.year | 4 | 0.05% |
| record | key_scorer | 4 | 0.05% |
| record | tools | 4 | 0.05% |
| record | currency_pair: Pair of currencies represented in the rate | 4 | 0.05% |
| record | amount: Numerical value of the exchange rate | 4 | 0.05% |
| record | description: Brief explanation of market event influencing the rate | 4 | 0.05% |
| record | decor | 4 | 0.05% |
| record | sensory | 4 | 0.05% |
| record | metaphors | 4 | 0.05% |
| record | objects | 4 | 0.05% |
| record | related_conditions | 4 | 0.05% |
| record | first_client | 4 | 0.05% |
| record | disclaimer.subject_of_responsibility_denial | 4 | 0.05% |
| record | content_providers | 4 | 0.05% |
| record | disclaimer.responsibility_denial | 4 | 0.05% |
| record | issue_type: Type of technical problem reported | 4 | 0.05% |
| record | ticket_id: Unique identifier for each service ticket | 4 | 0.05% |
| record | usage_purpose | 4 | 0.05% |
| record | manufacturer: Company that produced the chemical | 4 | 0.05% |
| record | address: Manufacturer's physical address | 4 | 0.05% |
| record | municipality.location.region | 4 | 0.05% |
| record | demographics.population.0.count | 4 | 0.05% |
| record | demographics.population.0.year | 4 | 0.05% |
| record | description: Detailed account of the outage and impact | 4 | 0.05% |
| record | duration: Duration of outage in hours | 4 | 0.05% |
| record | tier: Data speed tier of the service plan | 4 | 0.05% |
| record | position: Playing position such as midfielder, forward, defender | 4 | 0.05% |
| record | club_to: New club after transfer | 4 | 0.05% |
| record | player_id: Unique identifier assigned to the player transfer record | 4 | 0.05% |
| record | id: Unique identifier for the legislative vote | 4 | 0.05% |
| record | outcome: Final result of the vote | 4 | 0.05% |
| record | location: Geographic location of the bill's jurisdiction | 4 | 0.05% |
| record | plaintiff: Name of the individual or entity initiating the lawsuit | 4 | 0.05% |
| record | status: Current status of the case | 4 | 0.05% |
| record | claim_type: Nature of the legal claim such as breach of contract or noise violation | 4 | 0.05% |
| record | studyenddate | 4 | 0.05% |
| record | investigators | 4 | 0.05% |
| record | funding_sources | 4 | 0.05% |
| record | secondary_endpoints | 4 | 0.05% |
| record | reporting_doctor | 4 | 0.05% |
| record | region_churn | 4 | 0.05% |
| record | geologic_region | 4 | 0.05% |
| record | headliners | 4 | 0.05% |
| record | opposition_count | 4 | 0.05% |
| record | close_price | 4 | 0.05% |
| record | critic_location | 4 | 0.05% |
| record | asteroid.orbital_characteristics.eccentricity | 4 | 0.05% |
| record | asteroid.description.observations.last_observation_date | 4 | 0.05% |
| record | asteroid.orbital_characteristics.perihelion | 4 | 0.05% |
| record | asteroid.description.observations.observation_period_years | 4 | 0.05% |
| record | description: Brief contextual note about the transaction purpose | 4 | 0.05% |
| record | wallet_address: Cryptocurrency wallet address involved in the transaction | 4 | 0.05% |
| record | sprouting_definition | 4 | 0.05% |
| record | property_name | 4 | 0.05% |
| record | morphology.flowering_fruiting_period | 4 | 0.05% |
| record | morphology.seeds | 4 | 0.05% |
| record | transfer_code | 4 | 0.05% |
| record | source_currency | 4 | 0.05% |
| record | merchant: Name of the merchant or store involved | 4 | 0.05% |
| record | address_street | 4 | 0.05% |
| record | address_city | 4 | 0.05% |
| record | review_words | 4 | 0.05% |
| record | physician.degree | 4 | 0.05% |
| record | physician.address.city | 4 | 0.05% |
| record | physician.education.residency.0.institution | 4 | 0.05% |
| record | physician.education.medical_school.0.institution | 4 | 0.05% |
| record | catalog_id | 4 | 0.05% |
| record | types | 4 | 0.05% |
| record | development_periods | 4 | 0.05% |
| record | product.specifications.display.resolution | 4 | 0.05% |
| record | product.specifications.display.size | 4 | 0.05% |
| record | product.launch_location | 4 | 0.05% |
| record | product.registration_count.count | 4 | 0.05% |
| record | product.specifications.camera.rear | 4 | 0.05% |
| record | product.specifications.battery.capacity | 4 | 0.05% |
| record | product.specifications.memory.ram | 4 | 0.05% |
| record | study_date | 4 | 0.05% |
| record | location: Geographic location of the user or company | 4 | 0.05% |
| record | project.scope.setting_type | 4 | 0.05% |
| record | outcomes.qualitative_impact | 4 | 0.05% |
| record | project.timeline.duration_years | 4 | 0.05% |
| record | lead_agency.responsibilities | 4 | 0.05% |
| record | project.operational_model.stakeholders.0.description | 4 | 0.05% |
| record | project.operational_model.stakeholders.0.entity_type | 4 | 0.05% |
| record | physical_characteristics.publication_year | 4 | 0.05% |
| record | funding: Monetary commitment in millions of dollars | 4 | 0.05% |
| record | physical_characteristics.observations.0.year_orbited | 4 | 0.05% |
| record | institution: Affiliation of the guideline's author | 4 | 0.05% |
| record | sku: unique product identifier from supply chain database | 4 | 0.05% |
| record | commodity | 4 | 0.05% |
| record | output.unit | 4 | 0.05% |
| record | current_registration_count | 4 | 0.05% |
| record | competitor_id | 4 | 0.05% |
| record | resolution_summary | 4 | 0.05% |
| record | finding_description | 4 | 0.05% |
| record | facility_name | 4 | 0.05% |
| record | credit_hours: Number of credit hours for the course | 4 | 0.05% |
| record | status: Current operational status of the course | 4 | 0.05% |
| record | sku_id: Unique identifier for the product in inventory system | 4 | 0.05% |
| record | quantity: Current stock level of the SKU | 4 | 0.05% |
| record | location: Warehouse or hub where the SKU is stored | 4 | 0.05% |
| record | description.flowers.color | 4 | 0.05% |
| record | description.inflorescence.type | 4 | 0.05% |
| record | description.seeds.color | 4 | 0.05% |
| record | description.fruit.width_mm.min | 4 | 0.05% |
| record | description.flowering_time | 4 | 0.05% |
| record | systematics_and_distribution.location.state | 4 | 0.05% |
| record | systematics_and_distribution.first_description.author | 4 | 0.05% |
| record | description.inflorescence.height_m.min | 4 | 0.05% |
| record | common_names | 4 | 0.05% |
| record | description.fruit.width_mm.max | 4 | 0.05% |
| record | systematics_and_distribution.location.elevation_m.min | 4 | 0.05% |
| record | stock_status | 4 | 0.05% |
| record | expiration_date | 4 | 0.05% |
| record | acquisition_date | 4 | 0.05% |
| record | professions | 4 | 0.05% |
| record | location: City where the training session is held | 4 | 0.05% |
| record | techtoolsused | 4 | 0.05% |
| record | bathrooms: Number of bathrooms in the property | 4 | 0.05% |
| record | price: Listed sale price in dollars | 4 | 0.05% |
| record | build_year: Year the property was constructed | 4 | 0.05% |
| record | Attribute | 4 | 0.05% |
| record | positive_note | 4 | 0.05% |
| record | lease_type | 4 | 0.05% |
| record | cusip | 4 | 0.05% |
| record | context.demographic_challenge | 4 | 0.05% |
| record | project.objectives | 4 | 0.05% |
| record | incident.location.city | 4 | 0.05% |
| record | incident.location.region | 4 | 0.05% |
| record | incident.cause | 4 | 0.05% |
| record | person.family.0.name | 4 | 0.05% |
| record | person.family.0.occupation | 4 | 0.05% |
| record | person.family.0.relation | 4 | 0.05% |
| record | tv_program.production.location.institution | 4 | 0.05% |
| record | tv_program.production.location.name | 4 | 0.05% |
| record | tv_program.setting.location | 4 | 0.05% |
| record | tv_program.setting.year | 4 | 0.05% |
| record | discovery.astronomer.nationality | 4 | 0.05% |
| record | investor_leads | 4 | 0.05% |
| record | owner: Name of the customer filing the ticket | 4 | 0.05% |
| record | issue: Description of the reported hardware failure | 4 | 0.05% |
| record | location: City and state where the customer resides | 4 | 0.05% |
| record | ppe_required | 4 | 0.05% |
| record | handling_notes | 4 | 0.05% |
| record | admit_diagnosis | 4 | 0.05% |
| record | person.commentary_based_on | 4 | 0.05% |
| record | isotope | 4 | 0.05% |
| record | processing_code | 4 | 0.05% |
| record | systematics_and_distribution.distinguishing_features | 4 | 0.05% |
| record | hardware_support.gpu.amd.api | 4 | 0.05% |
| record | hardware_support.gpu.amd.version | 4 | 0.05% |
| record | supported_cryptocurrencies | 4 | 0.05% |
| record | status: Current status of the outreach message | 4 | 0.05% |
| record | message_id: Unique identifier for the outreach message | 4 | 0.05% |
| record | location: Geographic location of the recipient or company | 4 | 0.05% |
| record | reviewer: Name of the critic or editorial team | 4 | 0.05% |
| record | company: Company name of the ticket owner | 4 | 0.05% |
| record | abbreviation | 4 | 0.05% |
| record | full_term | 4 | 0.05% |
| record | article_metadata.relative_timestamp | 4 | 0.05% |
| record | article_metadata.headline | 4 | 0.05% |
| record | availability.locations.retail_outlets | 4 | 0.05% |
| record | product_usage | 4 | 0.05% |
| record | social_media_post.platform | 4 | 0.05% |
| record | findings: Diagnostic observations and conclusions from the imaging study | 4 | 0.05% |
| record | address_city: City where patient resides | 4 | 0.05% |
| record | condition: Medical diagnosis associated with symptoms | 4 | 0.05% |
| record | academic_year: Academic year covered by the scholarship | 4 | 0.05% |
| record | service_years: Number of years of required research service | 4 | 0.05% |
| record | potential | 4 | 0.05% |
| record | applications.medical | 4 | 0.05% |
| record | spectral_class | 4 | 0.05% |
| record | distance.reference_point | 4 | 0.05% |
| record | physical_properties.diameter.value | 4 | 0.05% |
| record | physical_properties.diameter.unit | 4 | 0.05% |
| record | review.performance.work | 4 | 0.05% |
| record | review.persons.2.role | 4 | 0.05% |
| record | review.performance.performers.1.name | 4 | 0.05% |
| record | review.performance.designer | 4 | 0.05% |
| record | review.title | 4 | 0.05% |
| record | user: Name of the employee reporting the issue | 4 | 0.05% |
| record | duration.min_years | 4 | 0.05% |
| record | project_goals | 4 | 0.05% |
| record | role: Job title or position being targeted | 4 | 0.05% |
| record | subject: Title or subject line of the outreach message | 4 | 0.05% |
| record | party: Political party of the winner | 4 | 0.05% |
| record | role: Title of the job position being advertised | 4 | 0.05% |
| record | asteroid.orbital_characteristics.inclination_unit | 4 | 0.05% |
| record | subgenus | 4 | 0.05% |
| record | commonName | 4 | 0.05% |
| record | affected_residents | 4 | 0.05% |
| record | emergency_status | 4 | 0.05% |
| record | filed_at: Timestamp when the ticket was opened | 4 | 0.05% |
| record | batteryorfeature | 4 | 0.05% |
| record | dosage: Prescribed dosage and frequency of the drug | 4 | 0.05% |
| record | severity: Level of reaction intensity | 4 | 0.05% |
| record | surveyFocus | 4 | 0.05% |
| record | accessInstructions | 4 | 0.05% |
| record | artistic_characteristics | 4 | 0.05% |
| record | consensus_percentage | 4 | 0.05% |
| record | entities_involved | 4 | 0.05% |
| record | service_impact | 4 | 0.05% |
| record | remaining_budget | 4 | 0.05% |
| record | onboarding_sprint_completion | 4 | 0.05% |
| record | location: City and state where buyer resides | 4 | 0.05% |
| record | return_status: Whether the product was returned or kept | 4 | 0.05% |
| record | organizations_and_media | 4 | 0.05% |
| record | villains | 4 | 0.05% |
| record | merchant_address | 4 | 0.05% |
| record | receipt_id | 4 | 0.05% |
| record | purity | 4 | 0.05% |
| record | cellular_localization | 4 | 0.05% |
| record | NNP | 4 | 0.05% |
| record | . | 4 | 0.05% |
| record | CD | 4 | 0.05% |
| record | JJ | 4 | 0.05% |
| record | asteroid.named_after.description | 4 | 0.05% |
| record | publication_ref | 4 | 0.05% |
| record | material: Primary material used in construction | 4 | 0.05% |
| record | stock_status: Current availability status such as active, backordered, on-hold, out-of-stock | 4 | 0.05% |
| record | patient_name: Full name of the individual being monitored | 4 | 0.05% |
| record | temperature: Body temperature in degrees Celsius | 4 | 0.05% |
| record | dosage_range | 4 | 0.05% |
| record | repair_shop | 4 | 0.05% |
| record | perpetrator.name | 4 | 0.05% |
| record | perpetrator.arrest_details.agency | 4 | 0.05% |
| record | perpetrator.status | 4 | 0.05% |
| record | victim.name | 4 | 0.05% |
| record | perpetrator.legal_actions | 4 | 0.05% |
| record | investigation.status | 4 | 0.05% |
| record | investigation.agency | 4 | 0.05% |
| record | crypto_type | 4 | 0.05% |
| record | paradox.change_given | 4 | 0.05% |
| record | paradox.total_initial_payment | 4 | 0.05% |
| record | paradox.explanation | 4 | 0.05% |
| record | deployment_status | 4 | 0.05% |
| record | opponent: Name of the player who lost in the final | 4 | 0.05% |
| record | message_id | 4 | 0.05% |
| record | outreach_status | 4 | 0.05% |
| record | policy_type: Type of insurance policy renewed | 4 | 0.05% |
| record | education_result_publication | 4 | 0.05% |
| record | average_score | 4 | 0.05% |
| record | credits.contributors.0.role | 4 | 0.05% |
| record | credits.creation_date | 4 | 0.05% |
| record | credits.graphic_design.1.designer | 4 | 0.05% |
| record | credits.graphic_design.0.area | 4 | 0.05% |
| record | credits.graphic_design.0.copyright_year | 4 | 0.05% |
| record | credits.contributors.0.person | 4 | 0.05% |
| record | promo_code: Discount or offer code associated with the campaign | 4 | 0.05% |
| record | facilities | 4 | 0.05% |
| record | weapons | 4 | 0.05% |
| record | defense_systems | 4 | 0.05% |
| record | time_references | 4 | 0.05% |
| record | authors: List of author names associated with the submission | 4 | 0.05% |
| record | acquisition_amount | 4 | 0.05% |
| record | policy_number: Insurance policy under which claim was submitted | 4 | 0.05% |
| record | diagnosis: Medical condition coded using ICD-10 standard | 4 | 0.05% |
| record | submitter: Name of the employee who submitted the ticket | 4 | 0.05% |
| record | department: Department the submitter belongs to | 4 | 0.05% |
| record | priority: Urgency level of the ticket | 4 | 0.05% |
| record | post_valuation: Post-money valuation in millions | 4 | 0.05% |
| record | amount: Total capital raised in millions | 4 | 0.05% |
| record | person.designation | 4 | 0.05% |
| record | unique_id: Unique identifier for the delay entry | 4 | 0.05% |
| record | investors: List of firms or individuals providing capital | 4 | 0.05% |
| record | name: Name of the festival as listed in the schedule | 4 | 0.05% |
| record | creator_name | 4 | 0.05% |
| record | additional_notes | 4 | 0.05% |
| record | price_unit | 4 | 0.05% |
| record | series | 4 | 0.05% |
| record | side dish | 4 | 0.05% |
| record | restaurant | 4 | 0.05% |
| record | counsel | 4 | 0.05% |
| record | source_agency | 4 | 0.05% |
| record | policy_id: Insurance policy number associated with the claim | 4 | 0.05% |
| record | diagnosis_code: Medical code indicating the patient's condition | 4 | 0.05% |
| record | address: Full address of the provider including street city state zip | 4 | 0.05% |
| record | reportTitle | 4 | 0.05% |
| record | event_id: Unique identifier for the conference or symposium | 4 | 0.05% |
| record | recipient: Email of the person receiving the confirmation or submission | 4 | 0.05% |
| record | availability_date: Date when product becomes available | 4 | 0.05% |
| record | stock | 4 | 0.05% |
| record | organization.full_name | 4 | 0.05% |
| record | organization.short_name | 4 | 0.05% |
| record | organization.personnel_changes.0.individual.name | 4 | 0.05% |
| record | article_metadata.source | 4 | 0.05% |
| record | source: Origin of the exchange rate data feed | 4 | 0.05% |
| record | entry_id: Unique identifier for each exchange rate entry | 4 | 0.05% |
| record | capacity: Maximum number of attendees allowed as per Capacity | 4 | 0.05% |
| record | status: Current status of the event like Confirmed or Scheduled | 4 | 0.05% |
| record | technique_name | 4 | 0.05% |
| record | era | 4 | 0.05% |
| record | deal_value | 4 | 0.05% |
| record | follow_up_survey | 4 | 0.05% |
| record | severity_rating | 4 | 0.05% |
| record | funding_allocated | 4 | 0.05% |
| record | status: Current market availability status | 4 | 0.05% |
| record | ingredients | 4 | 0.05% |
| record | cooking_methods | 4 | 0.05% |
| record | boiling_instructions.cooking_time.max_minutes | 4 | 0.05% |
| record | developer_fee | 4 | 0.05% |
| record | municipality.population.year | 4 | 0.05% |
| record | customer_location | 4 | 0.05% |
| record | closing_price | 4 | 0.05% |
| record | candidate | 4 | 0.05% |
| record | game mode | 4 | 0.05% |
| record | track set | 4 | 0.05% |
| record | launch_year: Year of product launch | 4 | 0.05% |
| record | manager: Name of the employee's manager | 4 | 0.05% |
| record | booking_information.pricing_details | 4 | 0.05% |
| record | booking_information.room_occupancy | 4 | 0.05% |
| record | asset: Cryptocurrency asset involved in the transaction BTC ETH SOL | 4 | 0.05% |
| record | amount: Quantity of asset traded or held in the transaction | 4 | 0.05% |
| record | affected_network | 4 | 0.05% |
| record | alternative_access | 4 | 0.05% |
| record | discovery.geologicalFormation | 4 | 0.05% |
| record | discovery.age.geologicalPeriod | 4 | 0.05% |
| record | publication.author.affiliation.country | 4 | 0.05% |
| record | publication.author.birthYear | 4 | 0.05% |
| record | discovery.location | 4 | 0.05% |
| record | publication.author.deathYear | 4 | 0.05% |
| record | lease_term: Type of lease term, e.g., month-to-month | 4 | 0.05% |
| record | person.birth_place.state | 4 | 0.05% |
| record | location: Department or unit where device is currently located | 4 | 0.05% |
| record | vendor: Company that sold or provided the device | 4 | 0.05% |
| record | acquired_company | 4 | 0.05% |
| record | revenue_projection | 4 | 0.05% |
| record | patient_care.group_therapy.importance | 4 | 0.05% |
| record | treatment_methods.description | 4 | 0.05% |
| record | item: Description of the returned product | 4 | 0.05% |
| record | status: Current status of the return approval | 4 | 0.05% |
| record | amount: Monetary value of the return in dollars | 4 | 0.05% |
| record | address: Physical address of the payee or recipient | 4 | 0.05% |
| record | reference_code: Internal code linking to external system or invoice | 4 | 0.05% |
| record | team_id | 4 | 0.05% |
| record | registered_on | 4 | 0.05% |
| record | team_member | 4 | 0.05% |
| record | history.creation.commissioning_date | 4 | 0.05% |
| record | final_standing | 4 | 0.05% |
| record | win_percentage | 4 | 0.05% |
| record | tel_aviv_attractions | 4 | 0.05% |
| record | israel_airports.total_count | 4 | 0.05% |
| record | tel_aviv_description | 4 | 0.05% |
| record | israel_airports.largest_airport.airline_homebase | 4 | 0.05% |
| record | passenger_information.infants.age_range | 4 | 0.05% |
| record | departure_date_prompt | 4 | 0.05% |
| record | supply | 4 | 0.05% |
| record | vacancy_rate.rate | 4 | 0.05% |
| record | prior_hospitalization | 4 | 0.05% |
| record | currency: Currency pair such as JPY, EUR, CAD | 4 | 0.05% |
| record | rate: Exchange rate in USD per foreign currency unit | 4 | 0.05% |
| record | status: Current status of the trend repost | 4 | 0.05% |
| record | peculiarity | 4 | 0.05% |
| record | case_type: Category of legal matter such as contract or personal injury | 4 | 0.05% |
| record | amountindispute: Monetary value of the claim or settlement | 4 | 0.05% |
| record | industry_segment | 4 | 0.05% |
| record | author: Reddit username of the comment author | 4 | 0.05% |
| record | tool: Productivity tool recommended in the comment | 4 | 0.05% |
| record | alternative_title | 4 | 0.05% |
| record | price.currency | 4 | 0.05% |
| record | feedback: Additional comments or issues noted in review | 4 | 0.05% |
| record | address: Full residential address of the customer | 4 | 0.05% |
| record | naming.named_after.profession | 4 | 0.05% |
| record | location.distance | 4 | 0.05% |
| record | nepa_guidance | 4 | 0.05% |
| record | funds_use | 4 | 0.05% |
| record | tissue | 4 | 0.05% |
| record | subject: Subject line of the malicious email | 4 | 0.05% |
| record | supplier_code | 4 | 0.05% |
| record | processing_location: Office or location where the transaction was processed | 4 | 0.05% |
| record | tournament_id | 4 | 0.05% |
| record | municipality.creation_date | 4 | 0.05% |
| record | service.location.postal_code | 4 | 0.05% |
| record | service.opening_hours.0.day | 4 | 0.05% |
| record | service.location.building | 4 | 0.05% |
| record | service.contact_information.phone | 4 | 0.05% |
| record | service.target_audience | 4 | 0.05% |
| record | service.responsible_person.profession | 4 | 0.05% |
| record | consideration_structure | 4 | 0.05% |
| record | enterprise_value | 4 | 0.05% |
| record | reporting_to: Name of the manager or department head this role reports to | 4 | 0.05% |
| record | policyholder_id: Unique identifier for the policyholder | 4 | 0.05% |
| record | policyholder_name: Full name of the claimant | 4 | 0.05% |
| record | claim_date: Date the claim was filed | 4 | 0.05% |
| record | treatment_type | 4 | 0.05% |
| record | originator | 4 | 0.05% |
| record | diagnosis_codes | 4 | 0.05% |
| record | admitting_department | 4 | 0.05% |
| record | person.associated_bands | 4 | 0.05% |
| record | person.birth_location.country_extended | 4 | 0.05% |
| record | person.birth_location.city | 4 | 0.05% |
| record | support_agent: Name of the agent assigned to handle the case | 4 | 0.05% |
| record | tenant_name: Name of the company occupying the space | 4 | 0.05% |
| record | leasetermlength: Duration of the lease in years | 4 | 0.05% |
| record | bill_name | 4 | 0.05% |
| record | owner: Name of the transaction owner | 4 | 0.05% |
| record | status: Current transaction state | 4 | 0.05% |
| record | address: Full physical address of the owner | 4 | 0.05% |
| record | email_address: Email address associated with the customer | 4 | 0.05% |
| record | amount_usd: Monetary amount related to the issue in USD | 4 | 0.05% |
| record | owner: Name of the claimant or policyholder | 4 | 0.05% |
| record | address: Full street address of the claimant | 4 | 0.05% |
| record | policy_id: Policy number associated with the claim | 4 | 0.05% |
| record | incident_type: Type of incident causing the claim | 4 | 0.05% |
| record | template_types.ordinary_template | 4 | 0.05% |
| record | templates_to_substitute | 4 | 0.05% |
| record | release_details.album.availability.0.access_type | 4 | 0.05% |
| record | release_details.artist.genres | 4 | 0.05% |
| record | promotion.video_series.content | 4 | 0.05% |
| record | press_release.source_url | 4 | 0.05% |
| record | promotion.video_series.frequency | 4 | 0.05% |
| record | press_release.date | 4 | 0.05% |
| record | promotion.short_film.description | 4 | 0.05% |
| record | press_release.source | 4 | 0.05% |
| record | promotion.short_film.availability | 4 | 0.05% |
| record | release_details.artist.location | 4 | 0.05% |
| record | research_project | 4 | 0.05% |
| record | ambiance_comment | 4 | 0.05% |
| record | sector_focus | 4 | 0.05% |
| record | person.death_date.month | 4 | 0.05% |
| record | person.biography.early_life.early_career | 4 | 0.05% |
| record | internal_code | 4 | 0.05% |
| record | treatment_overview.duration | 4 | 0.05% |
| record | treatment_overview.focus | 4 | 0.05% |
| record | treatment_examples.items.1.conditions | 4 | 0.05% |
| record | treatment_examples.items.2.category | 4 | 0.05% |
| record | treatment_examples.items.3.category | 4 | 0.05% |
| record | target_year | 4 | 0.05% |
| record | block.name | 4 | 0.05% |
| record | product.language | 4 | 0.05% |
| record | address: Physical address of the inventor | 4 | 0.05% |
| record | cost_impact | 4 | 0.05% |
| record | customer_satisfaction: Score from customer feedback surveys | 4 | 0.05% |
| record | kpideliveryaccuracy: Percentage of on-time deliveries achieved | 4 | 0.05% |
| record | donation_id: Unique identifier for each donation entry | 4 | 0.05% |
| record | donor: Name of the individual making the donation | 4 | 0.05% |
| record | resolution_timeline | 4 | 0.05% |
| record | description.leaf.texture | 4 | 0.05% |
| record | honorary_name.field_of_study | 4 | 0.05% |
| record | location.endemism | 4 | 0.05% |
| record | description.carpels | 4 | 0.05% |
| record | description.seed.color | 4 | 0.05% |
| record | honorary_name.person | 4 | 0.05% |
| record | location.communities | 4 | 0.05% |
| record | full_name: Complete name of the patient as recorded | 4 | 0.05% |
| record | dateofbirth: Patient's birth date in YYYY-MM-DD format | 4 | 0.05% |
| record | diagnosis_date: Date when primary diagnosis was established | 4 | 0.05% |
| record | currentmedicationlist: List of active medications prescribed | 4 | 0.05% |
| record | allergies: List of known allergies with reactions | 4 | 0.05% |
| record | insurance_provider: Name of the patient's health insurance provider | 4 | 0.05% |
| record | current_status: Current clinical status of the patient's condition | 4 | 0.05% |
| record | patient_id: Unique identifier for the patient in the system | 4 | 0.05% |
| record | test_code: Code identifying the type of laboratory test | 4 | 0.05% |
| record | result_value: Numerical value of the lab test result | 4 | 0.05% |
| record | co_investors: List of additional investors participating | 4 | 0.05% |
| record | birth.date_julian | 4 | 0.05% |
| record | occupation | 4 | 0.05% |
| record | death.date_julian | 4 | 0.05% |
| record | name.family_name | 4 | 0.05% |
| record | policytype | 4 | 0.05% |
| record | company.contact_information.phone_numbers | 4 | 0.05% |
| record | physical_characteristics.observations.0.telescope.origin | 4 | 0.05% |
| record | asteroid_belt_classification | 4 | 0.05% |
| record | subject: Field of study or research area awarded | 4 | 0.05% |
| record | recipient: Name of the student recipient | 4 | 0.05% |
| record | reviewer: Name of the film critic who wrote the review | 4 | 0.05% |
| record | trend_duration: Number of days the trend remained active or trending | 4 | 0.05% |
| record | holiday_look.heels.brand | 4 | 0.05% |
| record | holiday_look.dress.designer | 4 | 0.05% |
| record | trial_nct | 4 | 0.05% |
| record | product.specifications.connectivity | 4 | 0.05% |
| record | product.specifications.memory.internal_storage | 4 | 0.05% |
| record | person.title | 4 | 0.05% |
| record | service_code | 4 | 0.05% |
| record | provider_location | 4 | 0.05% |
| record | product_name: Name of the product being used | 4 | 0.05% |
| record | performance_impact: Percentage improvement in lead quality or hiring time | 4 | 0.05% |
| record | company: Organization where the product is used | 4 | 0.05% |
| record | format: Match structure used in the tournament | 4 | 0.05% |
| record | status: Current state of the tournament | 4 | 0.05% |
| record | process_details.negotiation_structure.outcome_format | 4 | 0.05% |
| record | process_details.negotiation_structure.decision_makers | 4 | 0.05% |
| record | governance.established_year | 4 | 0.05% |
| record | lastdosedate | 4 | 0.05% |
| record | district: Electoral district designation such as 4A or 15C | 4 | 0.05% |
| record | vote_share: Percentage of total votes the candidate received | 4 | 0.05% |
| record | address: Physical address of the candidate | 4 | 0.05% |
| record | status: Updated clinical status like stable, unstable, critical, or resolved | 4 | 0.05% |
| record | procedure_code | 4 | 0.05% |
| record | copyright_note | 4 | 0.05% |
| record | gymnosperm_diversity.diversity_center | 4 | 0.05% |
| record | location.mentioned_states | 4 | 0.05% |
| record | citations.3.author | 4 | 0.05% |
| record | gymnosperm_diversity.species_examples.3.characteristic | 4 | 0.05% |
| record | citations.1.author | 4 | 0.05% |
| record | gymnosperm_diversity.total_species | 4 | 0.05% |
| record | gymnosperm_diversity.species_examples.0.characteristic | 4 | 0.05% |
| record | patient.age_estimation.initial_estimation | 4 | 0.05% |
| record | queue | 4 | 0.05% |
| record | doctor.action | 4 | 0.05% |
| record | patient.request | 4 | 0.05% |
| record | patient.age_estimation.previous_state | 4 | 0.05% |
| record | ticket_id: Unique identifier for each support request | 4 | 0.05% |
| record | terminology.ideogram.english | 4 | 0.05% |
| record | comparison.alphabetic.learning_curve | 4 | 0.05% |
| record | comparison.alphabetic.pronunciation | 4 | 0.05% |
| record | comparison.ideographic.example.character | 4 | 0.05% |
| record | comparison.ideographic.example.languages.0.pronunciation | 4 | 0.05% |
| record | caveat.example.character | 4 | 0.05% |
| record | caveat.example.meaning | 4 | 0.05% |
| record | caveat.example.components.0.related_characters.0.meaning | 4 | 0.05% |
| record | caveat.example.components.0.related_characters.1.pronunciation | 4 | 0.05% |
| record | caveat.example.components.0.related_characters.1.character | 4 | 0.05% |
| record | caveat.example.components.0.related_characters.2.pronunciation | 4 | 0.05% |
| record | caveat.example.components.0.related_characters.2.meaning | 4 | 0.05% |
| record | caveat.example.components.1.component | 4 | 0.05% |
| record | positions | 4 | 0.05% |
| record | username | 4 | 0.05% |
| record | currency_pair | 4 | 0.05% |
| record | lot_size | 4 | 0.05% |
| record | citations.2.year | 4 | 0.05% |
| record | citations.2.author | 4 | 0.05% |
| record | gymnosperm_diversity.species_examples.2.name | 4 | 0.05% |
| record | census_inclusion | 4 | 0.05% |
| record | geography.coordinates.latitude | 4 | 0.05% |
| record | applicant: Organization or entity that filed the patent | 4 | 0.05% |
| record | location: Headquarters city and country of the company | 4 | 0.05% |
| record | company: Name of the startup receiving funding | 4 | 0.05% |
| record | brand: Name of the fashion or wellness brand launching the product | 4 | 0.05% |
| record | target_audience: Demographic group the campaign is aimed at | 4 | 0.05% |
| record | post_id: Unique identifier for the Instagram post announcing the product | 4 | 0.05% |
| record | statue.restoration.0.description | 4 | 0.05% |
| record | statue.restoration.0.funding.contribution_range.currency | 4 | 0.05% |
| record | statue.discovery.date | 4 | 0.05% |
| record | statue.transport_to_france | 4 | 0.05% |
| record | statue.public_display.stairs_completion_date | 4 | 0.05% |
| record | statue.material | 4 | 0.05% |
| record | statue.location.city | 4 | 0.05% |
| record | statue.restoration.0.funding.number_of_contributors | 4 | 0.05% |
| record | statue.dedication | 4 | 0.05% |
| record | participant_name | 4 | 0.05% |
| record | id: Unique identifier assigned to the drug | 4 | 0.05% |
| record | project_id: Unique identifier for the project in format GH-YYYY-NNN | 4 | 0.05% |
| record | repository_url: GitHub URL hosting the project | 4 | 0.05% |
| record | wallet_address | 4 | 0.05% |
| record | guidelines.moa_members.obligation | 4 | 0.05% |
| record | publication_date.month | 4 | 0.05% |
| record | compounds | 4 | 0.05% |
| record | techniques | 4 | 0.05% |
| record | proteins | 4 | 0.05% |
| record | external_links | 4 | 0.05% |
| record | physical_characteristics.observations_used | 4 | 0.05% |
| record | observations.observation_period_years | 4 | 0.05% |
| record | physical_characteristics.estimated_diameter_km | 4 | 0.05% |
| record | address: Full street address of the restaurant | 4 | 0.05% |
| record | priceperperson: Cost per person for the meal | 4 | 0.05% |
| record | service_quality: Description of service experience at the restaurant | 4 | 0.05% |
| record | retail_price: Final price at retail in US dollars | 4 | 0.05% |
| record | features: List of technical capabilities or functions | 4 | 0.05% |
| record | location: City of origin or primary market | 4 | 0.05% |
| record | patient_id: Unique identifier for each patient visit | 4 | 0.05% |
| record | diagnosis: Medical code indicating the diagnosed condition | 4 | 0.05% |
| record | pricing | 4 | 0.05% |
| record | genus.naming.year | 4 | 0.05% |
| record | genus.naming.named_by | 4 | 0.05% |
| record | genus.species.0.reason | 4 | 0.05% |
| record | asteroid.discovery.year | 4 | 0.05% |
| record | ranking.total_hotels | 4 | 0.05% |
| record | author: Name of the submitting researcher or academic | 4 | 0.05% |
| record | key_moment: Descriptive note about a pivotal play or event in the game | 4 | 0.05% |
| record | home_score: Final score of the home team | 4 | 0.05% |
| record | steals: Total steals recorded during the game | 4 | 0.05% |
| record | game_id: Unique identifier for the game as provided in the passage | 4 | 0.05% |
| record | physical property | 4 | 0.05% |
| record | organic molecule | 4 | 0.05% |
| record | carotenoid | 4 | 0.05% |
| record | location: Venue city and country where the final occurred | 4 | 0.05% |
| record | reviewer_name: Name of the customer who submitted the review | 4 | 0.05% |
| record | ticket_id: Internal ticket identifier for the RMA | 4 | 0.05% |
| record | patient_id: Unique identifier for each patient record | 4 | 0.05% |
| record | date: Date of the imaging procedure | 4 | 0.05% |
| record | findings: Clinical observations or abnormalities detected | 4 | 0.05% |
| record | issue_description: Description of the customer's reported problem | 4 | 0.05% |
| record | school | 4 | 0.05% |
| record | title: Job title such as Senior Data Analyst or Marketing Manager | 4 | 0.05% |
| record | promotion.video_series.end_date | 4 | 0.05% |
| record | promotion.video_series.platform | 4 | 0.05% |
| record | release_details.songs.0.title | 4 | 0.05% |
| record | release_details.album.title | 4 | 0.05% |
| record | adjectival_modifier | 4 | 0.05% |
| record | numerical_values | 4 | 0.05% |
| record | magazine.name | 4 | 0.05% |
| record | magazine.article.title | 4 | 0.05% |
| record | magazine.article.advertising_percentage | 4 | 0.05% |
| record | magazine.article.page_breakdown.5.advertising_percentage | 4 | 0.05% |
| record | magazine.article.page_breakdown.5.advertisers | 4 | 0.05% |
| record | magazine.article.page_breakdown.6.advertising_percentage | 4 | 0.05% |
| record | magazine.article.page_breakdown.6.advertisers | 4 | 0.05% |
| record | magazine.article.page_breakdown.7.advertisers | 4 | 0.05% |
| record | magazine.article.page_breakdown.8.page_number | 4 | 0.05% |
| record | magazine.article.page_breakdown.9.page_number | 4 | 0.05% |
| record | magazine.article.page_breakdown.11.page_number | 4 | 0.05% |
| record | magazine.article.page_breakdown.13.advertisers | 4 | 0.05% |
| record | magazine.article.page_breakdown.14.page_number | 4 | 0.05% |
| record | magazine.article.page_breakdown.14.advertisers | 4 | 0.05% |
| record | magazine.article.page_breakdown.16.page_number | 4 | 0.05% |
| record | magazine.article.page_breakdown.16.advertising_percentage | 4 | 0.05% |
| record | magazine.mentioned_article.author | 4 | 0.05% |
| record | album.tracklist.cd.0.notes | 4 | 0.05% |
| record | album.tracklist.cd.2.tie_in.0.type | 4 | 0.05% |
| record | album.tracklist.cd.0.title | 4 | 0.05% |
| record | album.tracklist.dvd.1.type | 4 | 0.05% |
| record | album.tracklist.cd.1.notes | 4 | 0.05% |
| record | album.tracklist.cd.7.composer | 4 | 0.05% |
| record | album.chart_performance.record_holder_rank | 4 | 0.05% |
| record | album.tracklist.cd.5.composer | 4 | 0.05% |
| record | album.external_links.2.description | 4 | 0.05% |
| record | album.tracklist.cd.8.track_number | 4 | 0.05% |
| record | album.tracklist.dvd.1.title | 4 | 0.05% |
| record | album.tracklist.dvd.0.title | 4 | 0.05% |
| record | recipient_address: Street address where goods or services were delivered | 4 | 0.05% |
| record | sender: Name of the company that issued the invoice | 4 | 0.05% |
| record | start_date: Date when the course begins | 4 | 0.05% |
| record | mapping_period | 4 | 0.05% |
| record | address: Physical address of the sender | 4 | 0.05% |
| record | sender: Name of the individual initiating the transfer | 4 | 0.05% |
| record | state_or_province | 4 | 0.05% |
| record | established_year | 4 | 0.05% |
| record | item_description.features | 4 | 0.05% |
| record | item_description.setup_charge.per | 4 | 0.05% |
| record | item_description.shipping.ready_to_ship_in.unit | 4 | 0.05% |
| record | item_description.dimensions.length | 4 | 0.05% |
| record | item_description.shipping.ready_to_ship_in.quantity | 4 | 0.05% |
| record | resolving_agent: Name of agent who resolved or handled the ticket | 4 | 0.05% |
| record | customer_email: Email address associated with the ticket | 4 | 0.05% |
| record | deployment_count: Number of farms, hospitals, or states using the system | 4 | 0.05% |
| record | deployment_location: Geographic region where system is deployed | 4 | 0.05% |
| record | games_played | 4 | 0.05% |
| record | tournament_id: Unique identifier for the tournament event | 4 | 0.05% |
| record | winner: Name of the player or team that won | 4 | 0.05% |
| record | winner_id: Unique ID assigned to the winner | 4 | 0.05% |
| record | opponent: Name of the player or team that lost | 4 | 0.05% |
| record | keynote_time | 4 | 0.05% |
| record | keynote_topic | 4 | 0.05% |
| record | root_cause: Technical reason for the service disruption | 4 | 0.05% |
| record | impact: Description of service impact including data loss or business effects | 4 | 0.05% |
| record | analysis_context | 4 | 0.05% |
| record | vaccination_date | 4 | 0.05% |
| record | funding_type: Description of funding commitment such as operational or adaptation | 4 | 0.05% |
| record | disease_links | 4 | 0.05% |
| record | discovery_details.observatory.location.county | 4 | 0.05% |
| record | asteroid_belt | 4 | 0.05% |
| record | orbital_characteristics.jpl_classification | 4 | 0.05% |
| record | underwriting_model | 4 | 0.05% |
| record | dose_sequence | 4 | 0.05% |
| record | expert_opinion.statement | 4 | 0.05% |
| record | expert_opinion.sentiment | 4 | 0.05% |
| record | hotel.distance_from_city_center.unit | 4 | 0.05% |
| record | hotel.distance_from_airport.value | 4 | 0.05% |
| record | hotel.nearby_attractions | 4 | 0.05% |
| record | metadata.news_agency | 4 | 0.05% |
| record | contract_details.financial_value.currency | 4 | 0.05% |
| record | contract_details.financial_value.formatted_value | 4 | 0.05% |
| record | security_compliance_delay | 4 | 0.05% |
| record | radius_km | 4 | 0.05% |
| record | report_metadata.source_url | 4 | 0.05% |
| record | company.financial_results.reporting_period | 4 | 0.05% |
| record | company.financial_results.metrics.0.historical_comparison.comparison_period | 4 | 0.05% |
| record | medication: Specific medication prescribed with dosage | 4 | 0.05% |
| record | lab_order: List of lab tests ordered for the patient | 4 | 0.05% |
| record | deductible: Increased deductible amount if applicable | 4 | 0.05% |
| record | location.distance_from_city_center.unit | 4 | 0.05% |
| record | location.distance_from_airport.value | 4 | 0.05% |
| record | location.distance_from_airport.unit | 4 | 0.05% |
| record | location.nearby_attractions | 4 | 0.05% |
| record | hotel_details.year_built | 4 | 0.05% |
| record | hotel_details.check_out | 4 | 0.05% |
| record | hotel_details.room_amenities | 4 | 0.05% |
| record | hotel_details.recreation | 4 | 0.05% |
| record | tissue_target | 4 | 0.05% |
| record | acquirer_location: Headquarters location of the acquiring company | 4 | 0.05% |
| record | nf_id: Unique identifier for each funding event | 4 | 0.05% |
| record | round: Series or type of funding round | 4 | 0.05% |
| record | outreach_date: Date the outreach was sent | 4 | 0.05% |
| record | status: Current status of the outreach attempt | 4 | 0.05% |
| record | recipient: Name of the recipient being contacted | 4 | 0.05% |
| record | contact_number: Primary phone number for customer communication | 4 | 0.05% |
| record | impact_depth | 4 | 0.05% |
| record | location: Geographic headquarters of the company | 4 | 0.05% |
| record | current_ratio: Liquidity ratio measuring short-term obligations | 4 | 0.05% |
| record | company_name: Name of the startup receiving funding | 4 | 0.05% |
| record | roundtype: Type of funding round | 4 | 0.05% |
| record | amount: Total funding amount in USD with currency symbol | 4 | 0.05% |
| record | retail_price: Price at retail in US dollars | 4 | 0.05% |
| record | headquarters: City and country where company is headquartered | 4 | 0.05% |
| record | status: Current operational or strategic status | 4 | 0.05% |
| record | next_milestone | 4 | 0.05% |
| record | downstream_genes | 4 | 0.05% |
| record | reference_number: Unique identifier for the invoice document | 4 | 0.05% |
| record | orbital_characteristics.inclination.reference | 4 | 0.05% |
| record | checklist_id | 4 | 0.05% |
| record | pathway_id: Unique identifier assigned to the pathway | 4 | 0.05% |
| record | discovery_date: Year the pathway was first identified | 4 | 0.05% |
| record | features: List of included amenities or space features | 4 | 0.05% |
| record | sales_volume | 4 | 0.05% |
| record | browse_topics | 4 | 0.05% |
| record | match_id: Unique identifier for the match as provided | 4 | 0.05% |
| record | player1_name | 4 | 0.05% |
| record | nmod | 4 | 0.05% |
| record | conj | 4 | 0.05% |
| record | preconj | 4 | 0.05% |
| record | destination: Final address of the shipment | 4 | 0.05% |
| record | consignment_id: Unique identifier for each shipment | 4 | 0.05% |
| record | paralympic_games.year | 4 | 0.05% |
| record | delegation.country | 4 | 0.05% |
| record | event.organizer.abbreviation | 4 | 0.05% |
| record | event.organizer.name | 4 | 0.05% |
| record | organization_info.focus | 4 | 0.05% |
| record | organization_info.facilities.generating_facilities | 4 | 0.05% |
| record | primate.dental_formula | 4 | 0.05% |
| record | primate.morphological_analysis.0.type | 4 | 0.05% |
| record | primate.related_primates.0.species | 4 | 0.05% |
| record | primate.genus | 4 | 0.05% |
| record | primate.related_primates.0.name | 4 | 0.05% |
| record | primate.morphological_analysis.1.genus | 4 | 0.05% |
| record | primate.classification | 4 | 0.05% |
| record | primate.related_primates.0.location | 4 | 0.05% |
| record | winner_ranking | 4 | 0.05% |
| record | vehicle.specifications.dimensions.height | 4 | 0.05% |
| record | vehicle.specifications.ground_clearance | 4 | 0.05% |
| record | park | 4 | 0.05% |
| record | semester | 4 | 0.05% |
| record | birth_place | 4 | 0.05% |
| record | shipping_options | 4 | 0.05% |
| record | first_name | 4 | 0.05% |
| record | launch_location | 4 | 0.05% |
| record | submitter: Name of the user who filed the ticket | 4 | 0.05% |
| record | kb_article | 4 | 0.05% |
| record | review_rating | 4 | 0.05% |
| record | pub_date | 4 | 0.05% |
| record | newsletter_type: Name of the newsletter subscribed to | 4 | 0.05% |
| record | submission_id: Unique identifier for each paper submission | 4 | 0.05% |
| record | institution: University and department affiliation of the authors | 4 | 0.05% |
| record | guideline_id: Unique identifier for the guideline document | 4 | 0.05% |
| record | price: Formatted price including dollar sign and cents | 4 | 0.05% |
| record | login.steps | 4 | 0.05% |
| record | institution_name: Name of the school or institute releasing results | 4 | 0.05% |
| record | location: Geographic region or city where the institution is located | 4 | 0.05% |
| record | grade_distribution: Breakdown of grades achieved by students | 4 | 0.05% |
| record | publication_code: Unique identifier assigned to the result publication | 4 | 0.05% |
| record | subject_focus: Key subjects or pathways emphasized in the results | 4 | 0.05% |
| record | institution: Hospital or organization where author is affiliated | 4 | 0.05% |
| record | event.person.age | 4 | 0.05% |
| record | event.opponent_club.name | 4 | 0.05% |
| record | event.person.position | 4 | 0.05% |
| record | away_team_score | 4 | 0.05% |
| record | place.demographics.0.year | 4 | 0.05% |
| record | investment | 4 | 0.05% |
| record | price: Listed retail price in US dollars | 4 | 0.05% |
| record | location: Office or room where issue occurred | 4 | 0.05% |
| record | person.death | 4 | 0.05% |
| record | approval_turnaround_days | 4 | 0.05% |
| record | examination_date | 4 | 0.05% |
| record | finding_summary | 4 | 0.05% |
| record | geographic_focus | 4 | 0.05% |
| record | state: State where the property is located | 4 | 0.05% |
| record | beds: Number of bedrooms | 4 | 0.05% |
| record | reference.name | 4 | 0.05% |
| record | reason_for_drinking | 4 | 0.05% |
| record | funding_amount: Total amount raised in millions of dollars | 4 | 0.05% |
| record | focus_area: Primary technological or business focus of the startup | 4 | 0.05% |
| record | score_home | 4 | 0.05% |
| record | product_type | 4 | 0.05% |
| record | procedure_code: CPT code describing the medical service performed | 4 | 0.05% |
| record | source_office: Clinic or office that submitted the claim | 4 | 0.05% |
| record | claim_id: Unique identifier for the claim as referenced in correspondence | 4 | 0.05% |
| record | document_details.authors.2.name | 4 | 0.05% |
| record | document_details.description | 4 | 0.05% |
| record | substances | 4 | 0.05% |
| record | consideration_cash | 4 | 0.05% |
| record | consideration_stock | 4 | 0.05% |
| record | status: Current status of the claim | 4 | 0.05% |
| record | serial_number: Unique hardware identifier for the device | 4 | 0.05% |
| record | ticket_number: Unique identifier for the RMA request | 4 | 0.05% |
| record | diseases_treated.scope | 4 | 0.05% |
| record | usability_score | 4 | 0.05% |
| record | tribute_to.location | 4 | 0.05% |
| record | establishment_resolution | 4 | 0.05% |
| record | funding_amount: Initial capital received in USD | 4 | 0.05% |
| record | school_address | 4 | 0.05% |
| record | starting_dose | 4 | 0.05% |
| record | shelf_life | 4 | 0.05% |
| record | visitor_team | 4 | 0.05% |
| record | operating_margin | 4 | 0.05% |
| record | attendance_notes | 4 | 0.05% |
| record | competition_year | 4 | 0.05% |
| record | meeting_days | 4 | 0.05% |
| record | partner_location: City and state of the partner company | 4 | 0.05% |
| record | status: Current operational status | 4 | 0.05% |
| record | growth_rate: Percentage growth in revenue | 4 | 0.05% |
| record | pharmacy_address | 4 | 0.05% |
| record | name: Product model name as branded | 4 | 0.05% |
| record | description: Summary of incident cause and policy coverage rationale | 4 | 0.05% |
| record | location: City or region where vaccination occurred | 4 | 0.05% |
| record | dose_number: Sequence of the dose in the series | 4 | 0.05% |
| record | tenant_name: Name of the current tenant or prospective renter | 4 | 0.05% |
| record | lab_tests | 4 | 0.05% |
| record | product_id: Unique identifier for each product | 4 | 0.05% |
| record | stock: Current quantity available in inventory | 4 | 0.05% |
| record | lease_id: Unique identifier for each lease agreement | 4 | 0.05% |
| record | property_location: Full address including street, suite, and city | 4 | 0.05% |
| record | publication.author | 4 | 0.05% |
| record | webmaster | 4 | 0.05% |
| record | maintainers_location | 4 | 0.05% |
| record | sync_frequency | 4 | 0.05% |
| record | details.calorie_burn.running_time | 4 | 0.05% |
| record | announcement_metadata.coverage_provider | 4 | 0.05% |
| record | regulatory_status.review_status | 4 | 0.05% |
| record | announcement_metadata.date | 4 | 0.05% |
| record | announcement_metadata.agency | 4 | 0.05% |
| record | doctor_name | 4 | 0.05% |
| record | career_summary.commitment | 4 | 0.05% |
| record | career_summary.roles | 4 | 0.05% |
| record | stock_exchange | 4 | 0.05% |
| record | facility | 4 | 0.05% |
| record | comment_rank | 4 | 0.05% |
| record | clinical_outcomes.dosage_efficiency | 4 | 0.05% |
| record | coverage_details: Description of coverage enhancements or changes | 4 | 0.05% |
| record | renewal_reason: Justification for renewal or decline | 4 | 0.05% |
| record | holiday_look.collar.brand | 4 | 0.05% |
| record | result_value: Measured value from the test | 4 | 0.05% |
| record | template_substitution.prefix | 4 | 0.05% |
| record | room: Room number where imaging was performed | 4 | 0.05% |
| record | hospital | 4 | 0.05% |
| record | source_type: Type of data source such as interbank feed or real-time trading | 4 | 0.05% |
| record | discipline: Type of sport or competition such as swimming or skiing | 4 | 0.05% |
| record | athlete_name: Name of the athlete who achieved the result | 4 | 0.05% |
| record | payment | 4 | 0.05% |
| record | donorfecid | 4 | 0.05% |
| record | doctor | 4 | 0.05% |
| record | rr | 4 | 0.05% |
| record | repair_cost | 4 | 0.05% |
| record | charge_amount | 4 | 0.05% |
| record | original_plan | 4 | 0.05% |
| record | tehsil.district | 4 | 0.05% |
| record | donor_name: Full name of the contributing individual or organization | 4 | 0.05% |
| record | fashion_trends | 4 | 0.05% |
| record | headquarters.neighborhood | 4 | 0.05% |
| record | discount.description | 4 | 0.05% |
| record | location.hotel_count | 4 | 0.05% |
| record | booking.promotional_code | 4 | 0.05% |
| record | booking.validity | 4 | 0.05% |
| record | terms_and_conditions.availability | 4 | 0.05% |
| record | terms_and_conditions.group_booking_restriction | 4 | 0.05% |
| record | terms_and_conditions.room_upgrades | 4 | 0.05% |
| record | money | 4 | 0.05% |
| record | coverage_rate | 4 | 0.05% |
| record | comorbidity_action | 4 | 0.05% |
| record | traveler_location | 4 | 0.05% |
| record | blood_pressure: Blood pressure in mmHg format | 4 | 0.05% |
| record | primary_treatment | 4 | 0.05% |
| record | billing_code | 4 | 0.05% |
| record | primate.morphological_analysis.0.finding | 4 | 0.05% |
| record | genus.pollination.mechanism | 4 | 0.05% |
| record | genus.etymology.part1.name | 4 | 0.05% |
| record | genus.species_list | 4 | 0.05% |
| record | sexual_act.insertion.initial_alignment | 4 | 0.05% |
| record | sexual_act.insertion.vaginal_wall_dilation | 4 | 0.05% |
| record | sexual_act.insertion.cervix_appearance.initial_appearance | 4 | 0.05% |
| record | sexual_act.insertion.cervix_appearance.hole_description | 4 | 0.05% |
| record | sexual_act.insertion.cervix_appearance.changes_during_sex | 4 | 0.05% |
| record | sexual_act.thrusting.cervix_transformation.shape_change | 4 | 0.05% |
| record | sexual_act.ejaculation.sperm_release | 4 | 0.05% |
| record | sexual_act.ejaculation.trunk_interaction | 4 | 0.05% |
| record | sexual_act.ejaculation.cervix_immersion | 4 | 0.05% |
| record | sexual_act.ejaculation.trunk_analogy | 4 | 0.05% |
| record | sexual_act.sperm_migration.cervical_mucus_pathway | 4 | 0.05% |
| record | sexual_act.sperm_migration.mucus_sperm_interaction | 4 | 0.05% |
| record | sexual_act.cervix_retraction.sperm_pathway_cutoff | 4 | 0.05% |
| record | sexual_act.cervix_retraction.reflux_inevitability | 4 | 0.05% |
| record | hotel_experience.location.city | 4 | 0.05% |
| record | hotel_experience.additional_benefits | 4 | 0.05% |
| record | hotel_experience.suites.0.bathroom_features | 4 | 0.05% |
| record | hotel_experience.suites.0.features | 4 | 0.05% |
| record | hotel_experience.entertainment.tv.screen_size | 4 | 0.05% |
| record | streaming_platform | 4 | 0.05% |
| record | start_time | 4 | 0.05% |
| record | cost: Total cost paid for the training program | 4 | 0.05% |
| record | departure_date | 4 | 0.05% |
| record | region: Geographic region where product is sold | 4 | 0.05% |
| record | brand: Manufacturer or brand name | 4 | 0.05% |
| record | shipping_method: Delivery method for the product | 4 | 0.05% |
| record | category: General product category such as apparel or electronics | 4 | 0.05% |
| record | publication.research_context.time_period | 4 | 0.05% |
| record | linguistic_framework.reference_scholar | 4 | 0.05% |
| record | linguistic_framework.comparison_aspects | 4 | 0.05% |
| record | publication.title | 4 | 0.05% |
| record | affected_service: List of services disrupted during the outage | 4 | 0.05% |
| record | ticket_id: Unique identifier for the outage incident | 4 | 0.05% |
| record | duration: Number of weeks the course runs | 4 | 0.05% |
| record | credits: Number of academic credits assigned to the course | 4 | 0.05% |
| record | Chinese farmers <> used | 4 | 0.05% |
| record | James Holmes <> fired shots | 4 | 0.05% |
| record | Project SAM <> opposes | 4 | 0.05% |
| record | funding_id: Unique identifier for each funding round | 4 | 0.05% |
| record | cuisine_type | 4 | 0.05% |
| record | salary_max | 4 | 0.05% |
| record | test_date: Date when the test was performed | 4 | 0.05% |
| record | result_value: Numerical value of the test result | 4 | 0.05% |
| record | holder_name | 4 | 0.05% |
| record | article.references.0.description | 4 | 0.05% |
| record | outbreak_event.illness.pathogen.classification | 4 | 0.05% |
| record | outbreak_event.location.setting | 4 | 0.05% |
| record | outbreak_event.illness.description | 4 | 0.05% |
| record | outbreak_event.investigation.findings.agent_details.strain_type | 4 | 0.05% |
| record | outbreak_event.investigation.findings.agent_details.abbreviation | 4 | 0.05% |
| record | address.street | 4 | 0.05% |
| record | office_hours.4.day | 4 | 0.05% |
| record | office_hours.2.end_time | 4 | 0.05% |
| record | office_hours.4.end_time | 4 | 0.05% |
| record | office_hours.4.start_time | 4 | 0.05% |
| record | office_hours.3.end_time | 4 | 0.05% |
| record | office_hours.0.end_time | 4 | 0.05% |
| record | passenger_name: Name of the passenger on the flight | 4 | 0.05% |
| record | product_line | 4 | 0.05% |
| record | game_outcome | 4 | 0.05% |
| record | key_player_points | 4 | 0.05% |
| record | turnovers | 4 | 0.05% |
| record | encounter_date: Date of the patient visit | 4 | 0.05% |
| record | event.person.name | 4 | 0.05% |
| record | event.new_club.name | 4 | 0.05% |
| record | boulder.start.hand_holds.0.description | 4 | 0.05% |
| record | boulder.start.foot_holds.0.description | 4 | 0.05% |
| record | boulder.location.distance_unit | 4 | 0.05% |
| record | research.focus | 4 | 0.05% |
| record | research.materials | 4 | 0.05% |
| record | research.temperature | 4 | 0.05% |
| record | hotel.location.distance_from_center | 4 | 0.05% |
| record | appointment_type | 4 | 0.05% |
| record | originalheadcountapex | 4 | 0.05% |
| record | currentheadcountapex | 4 | 0.05% |
| record | iporevisedvaluation | 4 | 0.05% |
| record | marketing_budget | 4 | 0.05% |
| record | star.constellation | 4 | 0.05% |
| record | star.distance.light_years | 4 | 0.05% |
| record | star.effective_temperature | 4 | 0.05% |
| record | planetary_system.planets.0.designation | 4 | 0.05% |
| record | medal_type | 4 | 0.05% |
| record | hotel.services | 4 | 0.05% |
| record | hotel.distance_from_airport.unit | 4 | 0.05% |
| record | incident.suspects.1.age | 4 | 0.05% |
| record | source.author | 4 | 0.05% |
| record | sensor_nodes: Number of data collection nodes used in the system | 4 | 0.05% |
| record | pharmacy_name: Name of the pharmacy that processed the prescription | 4 | 0.05% |
| record | date: Date when the vote occurred | 4 | 0.05% |
| record | treatment_arm | 4 | 0.05% |
| record | lastfollowup | 4 | 0.05% |
| record | topic: Subject matter discussed in the statement | 4 | 0.05% |
| record | statement_reference: Identifier code for the statement document | 4 | 0.05% |
| record | founding_address: Physical address of the startup's registration | 4 | 0.05% |
| record | founding_year: Year the startup was established | 4 | 0.05% |
| record | equity_stake: Founder's ownership percentage in the company | 4 | 0.05% |
| record | scale | 4 | 0.05% |
| record | channel: Name of the creator's YouTube channel | 4 | 0.05% |
| record | share_count: Number of times the video has been shared | 4 | 0.05% |
| record | target_audience: Group of users or professionals it serves | 4 | 0.05% |
| record | social_handle | 4 | 0.05% |
| record | tx_id | 4 | 0.05% |
| record | weight_range_kg.min_kg | 4 | 0.05% |
| record | group_name | 4 | 0.05% |
| record | arrival_city | 4 | 0.05% |
| record | seat_class | 4 | 0.05% |
| record | departure_city | 4 | 0.05% |
| record | month | 4 | 0.05% |
| record | symptom_description | 4 | 0.05% |
| record | name: Name of the beauty product launched | 4 | 0.05% |
| record | venue_id | 4 | 0.05% |
| record | nationality: Country of origin of the player | 4 | 0.05% |
| record | transfer_fee: Amount paid in million euros for the transfer | 4 | 0.05% |
| record | product.availability | 4 | 0.05% |
| record | event.year | 4 | 0.05% |
| record | performance_code | 4 | 0.05% |
| record | engagement_rate | 4 | 0.05% |
| record | sources | 4 | 0.05% |
| record | onset_date: Date when symptoms first appeared | 4 | 0.05% |
| record | full_name: Name of the employee as recorded in the review | 4 | 0.05% |
| record | department: Department where the employee is based | 4 | 0.05% |
| record | removal_instructions.tools | 4 | 0.05% |
| record | patient_address: Street address of the patient | 4 | 0.05% |
| record | dosage: Instructions on how and when to take the medication | 4 | 0.05% |
| record | person.experience.0.role | 4 | 0.05% |
| record | person.experience.1.role | 4 | 0.05% |
| record | person.experience.1.description | 4 | 0.05% |
| record | person.parliamentary_role.0.start_year | 4 | 0.05% |
| record | person.military_rank | 4 | 0.05% |
| record | historical_context.time_periods.0.events.0.diseases | 4 | 0.05% |
| record | historical_context.cities | 4 | 0.05% |
| record | historical_context.time_periods.2.events.0.event_type | 4 | 0.05% |
| record | historical_context.time_periods.3.events.0.event_type | 4 | 0.05% |
| record | historical_context.time_periods.3.period | 4 | 0.05% |
| record | historical_context.time_periods.2.events.0.description | 4 | 0.05% |
| record | author.experience | 4 | 0.05% |
| record | price.amount | 4 | 0.05% |
| record | laboratories | 4 | 0.05% |
| record | staff.research_assistant_count | 4 | 0.05% |
| record | model_name | 4 | 0.05% |
| record | utilities_included | 4 | 0.05% |
| record | hotel.contact.phone | 4 | 0.05% |
| record | hotel.contact.email | 4 | 0.05% |
| record | hotel.transportation.airport | 4 | 0.05% |
| record | hotel.transportation.parking.0.name | 4 | 0.05% |
| record | hotel.transportation.parking.1.walking_time | 4 | 0.05% |
| record | hotel.transportation.parking.1.directions | 4 | 0.05% |
| record | record_label | 4 | 0.05% |
| record | description: Description of the transaction activity | 4 | 0.05% |
| record | name: Full name of the customer | 4 | 0.05% |
| record | address: Customer's street address | 4 | 0.05% |
| record | station: Name of the gas station | 4 | 0.05% |
| record | category: Type of transaction like transportation | 4 | 0.05% |
| record | species_name | 4 | 0.05% |
| record | document_details.oclc_number | 4 | 0.05% |
| record | assessment_name | 4 | 0.05% |
| record | affected_entity: Organization or business impacted by the violation | 4 | 0.05% |
| record | discovery.location.place | 4 | 0.05% |
| record | external_links.0.title | 4 | 0.05% |
| record | funding: Total funding amount in millions | 4 | 0.05% |
| record | project_name: Name of the technology project | 4 | 0.05% |
| record | project_id: Unique identifier assigned to the project | 4 | 0.05% |
| record | status: Current financial status of the invoice | 4 | 0.05% |
| record | Steinem <> named after | 4 | 0.05% |
| record | Steinem <> nudged | 4 | 0.05% |
| record | bearded friend <> smoked | 4 | 0.05% |
| record | my cousin <> adopted | 4 | 0.05% |
| record | encounter_id | 4 | 0.05% |
| record | manufacturer: Company responsible for manufacturing the chemical | 4 | 0.05% |
| record | assigned_judge | 4 | 0.05% |
| record | event_name: Title of the event as listed | 4 | 0.05% |
| record | city: City where the event is held | 4 | 0.05% |
| record | claims_count | 4 | 0.05% |
| record | funding_grant | 4 | 0.05% |
| record | geographic_context | 4 | 0.05% |
| record | place.location.county | 4 | 0.05% |
| record | events.0.purpose | 4 | 0.05% |
| record | breastfeeding_situation.recommendations | 4 | 0.05% |
| record | countries.0.breastfeeding_rate | 4 | 0.05% |
| record | events.0.type | 4 | 0.05% |
| record | year: Year of financial data | 4 | 0.05% |
| record | partner_year: Year of partner's financial data | 4 | 0.05% |
| record | company_id: Unique identifier for the company | 4 | 0.05% |
| record | amount_spent | 4 | 0.05% |
| record | listing_code | 4 | 0.05% |
| record | study.period.end_year | 4 | 0.05% |
| record | study.focus | 4 | 0.05% |
| record | additional_information_url | 4 | 0.05% |
| record | certification.exams.1.name | 4 | 0.05% |
| record | organization.website | 4 | 0.05% |
| record | findings: Key clinical observations or abnormalities detected | 4 | 0.05% |
| record | patient_id: Unique patient identifier from the medical record | 4 | 0.05% |
| record | status: Current status of the report workflow | 4 | 0.05% |
| record | treatment.effectiveness | 4 | 0.05% |
| record | disease.gene | 4 | 0.05% |
| record | discovery.impact | 4 | 0.05% |
| record | treatment_status | 4 | 0.05% |
| record | claim_date: Date the claim was filed with the insurer | 4 | 0.05% |
| record | expected_attendance | 4 | 0.05% |
| record | politician | 4 | 0.05% |
| record | treatment: Therapeutic approach prescribed | 4 | 0.05% |
| record | provider: Name of the physician who treated the patient | 4 | 0.05% |
| record | refund_amount: Amount refunded or credited to customer | 4 | 0.05% |
| record | support_response: Agent's response to the customer's issue | 4 | 0.05% |
| record | postInformation.date | 4 | 0.05% |
| record | vehicle.componentReplacements | 4 | 0.05% |
| record | vehicle.year | 4 | 0.05% |
| record | announcement.league | 4 | 0.05% |
| record | announcement.divisions.0.honorable_mentions.0.position | 4 | 0.05% |
| record | announcement.divisions.0.honorable_mentions.1.team | 4 | 0.05% |
| record | announcement.divisions.0.honorable_mentions.4.player_name | 4 | 0.05% |
| record | announcement.divisions.1.star_of_the_week.player_name | 4 | 0.05% |
| record | announcement.divisions.1.star_of_the_week.team | 4 | 0.05% |
| record | announcement.divisions.1.second_star.player_name | 4 | 0.05% |
| record | announcement.divisions.1.honorable_mentions.0.player_name | 4 | 0.05% |
| record | announcement.divisions.1.honorable_mentions.0.position | 4 | 0.05% |
| record | announcement.divisions.2.second_star.position | 4 | 0.05% |
| record | announcement.divisions.2.division_name | 4 | 0.05% |
| record | announcement.divisions.2.honorable_mentions.0.position | 4 | 0.05% |
| record | announcement.divisions.2.honorable_mentions.0.team | 4 | 0.05% |
| record | announcement.divisions.2.honorable_mentions.1.position | 4 | 0.05% |
| record | announcement.divisions.2.honorable_mentions.1.team | 4 | 0.05% |
| record | announcement.divisions.2.honorable_mentions.4.team | 4 | 0.05% |
| record | announcement.divisions.3.second_star.team | 4 | 0.05% |
| record | announcement.divisions.3.honorable_mentions.0.player_name | 4 | 0.05% |
| record | announcement.divisions.3.honorable_mentions.1.position | 4 | 0.05% |
| record | announcement.divisions.3.honorable_mentions.1.team | 4 | 0.05% |
| record | announcement.divisions.4.division_name | 4 | 0.05% |
| record | announcement.divisions.4.honorable_mentions.0.player_name | 4 | 0.05% |
| record | announcement.divisions.4.honorable_mentions.0.position | 4 | 0.05% |
| record | announcement.divisions.4.honorable_mentions.1.position | 4 | 0.05% |
| record | announcement.divisions.4.honorable_mentions.2.player_name | 4 | 0.05% |
| record | announcement.divisions.4.honorable_mentions.2.position | 4 | 0.05% |
| record | announcement.divisions.5.star_of_the_week.player_name | 4 | 0.05% |
| record | announcement.divisions.5.star_of_the_week.position | 4 | 0.05% |
| record | announcement.divisions.5.star_of_the_week.team | 4 | 0.05% |
| record | announcement.divisions.5.second_star.player_name | 4 | 0.05% |
| record | announcement.divisions.5.second_star.team | 4 | 0.05% |
| record | announcement.divisions.5.division_name | 4 | 0.05% |
| record | announcement.divisions.5.honorable_mentions.0.team | 4 | 0.05% |
| record | announcement.divisions.5.honorable_mentions.2.position | 4 | 0.05% |
| record | announcement.divisions.5.honorable_mentions.2.team | 4 | 0.05% |
| record | director | 4 | 0.05% |
| record | playertwoid | 4 | 0.05% |
| record | playeronename | 4 | 0.05% |
| record | match_result | 4 | 0.05% |
| record | observations.observation_period.years | 4 | 0.05% |
| record | wiki_sky_links | 4 | 0.05% |
| record | storage_location | 4 | 0.05% |
| record | report_id: Unique identifier assigned to the imaging report | 4 | 0.05% |
| record | company_id: Unique identifier assigned to each company | 4 | 0.05% |
| record | video_id: Unique identifier assigned to the video | 4 | 0.05% |
| record | location: City and state or country where the company is based | 4 | 0.05% |
| record | current_status: Current operational phase of the company | 4 | 0.05% |
| record | founder: Name of the individual who founded the company | 4 | 0.05% |
| record | clientbasesize: Number of clients served across regions | 4 | 0.05% |
| record | religious_concepts | 4 | 0.05% |
| record | valuation_method | 4 | 0.05% |
| record | disease_profile.differential_diagnoses.1.disease_name | 4 | 0.05% |
| record | case_report.clinical_presentation.lesion_description.1.count | 4 | 0.05% |
| record | case_report.diagnostics.trichoscopy.findings.2.feature | 4 | 0.05% |
| record | disease_profile.prognosis.recovery_timeline | 4 | 0.05% |
| record | disease_profile.risk_factors | 4 | 0.05% |
| record | disease_profile.pathophysiology | 4 | 0.05% |
| record | case_report.patient.medical_history | 4 | 0.05% |
| record | workshop_count | 4 | 0.05% |
| record | keynote_count | 4 | 0.05% |
| record | movie.comments.16.time | 4 | 0.05% |
| record | movie.comments.14.date | 4 | 0.05% |
| record | movie.comments.19.text | 4 | 0.05% |
| record | movie.comments.5.text | 4 | 0.05% |
| record | movie.comments.13.time | 4 | 0.05% |
| record | movie.comments.21.time | 4 | 0.05% |
| record | movie.comments.20.text | 4 | 0.05% |
| record | movie.comments.10.text | 4 | 0.05% |
| record | movie.comments.1.date | 4 | 0.05% |
| record | movie.genre | 4 | 0.05% |
| record | movie.comments.18.text | 4 | 0.05% |
| record | movie.comments.12.time | 4 | 0.05% |
| record | movie.comments.8.time | 4 | 0.05% |
| record | movie.comments.3.text | 4 | 0.05% |
| record | movie.comments.9.time | 4 | 0.05% |
| record | movie.comments.11.time | 4 | 0.05% |
| record | movie.comments.3.date | 4 | 0.05% |
| record | movie.comments.4.time | 4 | 0.05% |
| record | movie.comments.8.text | 4 | 0.05% |
| record | movie.production_country | 4 | 0.05% |
| record | merchantcategorycode | 4 | 0.05% |
| record | emergency_contact | 4 | 0.05% |
| record | url_references | 4 | 0.05% |
| record | spo2: Oxygen saturation percentage | 4 | 0.05% |
| record | determiners | 4 | 0.05% |
| record | prepositions | 4 | 0.05% |
| record | possessive_endings | 4 | 0.05% |
| record | singular_nouns | 4 | 0.05% |
| record | address: Physical address of the filing party | 4 | 0.05% |
| record | jurisdiction: Court jurisdiction and division where case is filed | 4 | 0.05% |
| record | company.location | 4 | 0.05% |
| record | organization.location_founded.country | 4 | 0.05% |
| record | relationships.0.type | 4 | 0.05% |
| record | certification.exams.0.name | 4 | 0.05% |
| record | position: Title or role of the appointed person | 4 | 0.05% |
| record | target_currency: Currency to which amount is converted | 4 | 0.05% |
| record | symptom_date | 4 | 0.05% |
| record | outage_code | 4 | 0.05% |
| record | numberofteams | 4 | 0.05% |
| record | operation_hours | 4 | 0.05% |
| record | album | 4 | 0.05% |
| record | return_id: Unique identifier for the return transaction | 4 | 0.05% |
| record | event_code: Unique identifier for the event as listed in records | 4 | 0.05% |
| record | study_start | 4 | 0.05% |
| record | event_code: Unique identifier for the Olympic competition | 4 | 0.05% |
| record | disclosure_id: Unique identifier for the disclosure entry | 4 | 0.05% |
| record | electoral_code | 4 | 0.05% |
| record | asteroid.discovery.observatory.location.mountains | 4 | 0.05% |
| record | asteroid.discovery.observatory.location.city | 4 | 0.05% |
| record | asteroid.physical_characteristics.observations.0.telescope.wavelength | 4 | 0.05% |
| record | policyholder_id: Unique identifier for the insurance policyholder | 4 | 0.05% |
| record | staffing_change | 4 | 0.05% |
| record | holiday | 4 | 0.05% |
| record | ordering_physician | 4 | 0.05% |
| record | price: Price of the product in USD | 4 | 0.05% |
| record | policy_id: Unique identifier assigned to the policy | 4 | 0.05% |
| record | purchase_order: Order number from original purchase | 4 | 0.05% |
| record | channel_location | 4 | 0.05% |
| record | plaintiff: Individual or entity initiating the lawsuit | 4 | 0.05% |
| record | startup: Name of the startup receiving investment | 4 | 0.05% |
| record | tenant: Name of the company leasing the space | 4 | 0.05% |
| record | affected_area | 4 | 0.05% |
| record | dedication.description | 4 | 0.05% |
| record | location: City and state where the startup is based | 4 | 0.05% |
| record | problemDescription | 4 | 0.05% |
| record | vehicle.make | 4 | 0.05% |
| record | disease_name | 4 | 0.05% |
| record | ref_code | 4 | 0.05% |
| record | federalism_definition.federalism | 4 | 0.05% |
| record | federalism_definition.confederation | 4 | 0.05% |
| record | federalism_definition.confederation_examples | 4 | 0.05% |
| record | south_yemen_conflict.accusations_against_governments.0.government_name | 4 | 0.05% |
| record | patient_case.clinical_presentation.medical_history.0.status | 4 | 0.05% |
| record | disease_profile.genetic_mapping.chromosome_location | 4 | 0.05% |
| record | disease_profile.abbreviation | 4 | 0.05% |
| record | report_context.geographic_significance | 4 | 0.05% |
| record | patient_case.genetic_analysis.mutation_details.dna_nomenclature | 4 | 0.05% |
| record | report_context.epidemiology_notes | 4 | 0.05% |
| record | patient_case.genetic_analysis.mutation_details.affected_protein | 4 | 0.05% |
| record | patient_case.clinical_presentation.skin_lesions.locations | 4 | 0.05% |
| record | patient_case.genetic_analysis.mutation_details.functional_effect | 4 | 0.05% |
| record | gender_distribution | 4 | 0.05% |
| record | disease.subspecies.1.name | 4 | 0.05% |
| record | disease.subspecies.0.name | 4 | 0.05% |
| record | travel_support | 4 | 0.05% |
| record | final_payment | 4 | 0.05% |
| record | technical_terms | 4 | 0.05% |
| record | events | 4 | 0.05% |
| record | promo_code: Discount code offered with the product | 4 | 0.05% |
| record | post_id: Unique identifier for the Instagram post | 4 | 0.05% |
| record | date: Date when the availability was recorded | 4 | 0.05% |
| record | status: Current availability status of the unit | 4 | 0.05% |
| record | approval_date | 4 | 0.05% |
| record | treatment_history | 4 | 0.05% |
| record | election_type | 4 | 0.05% |
| record | disease.subspecies.0.affected_species | 4 | 0.05% |
| record | address: Full address of the reviewer | 4 | 0.05% |
| record | reporting_to: Name of superior or reporting authority | 4 | 0.05% |
| record | title: Official job title or position | 4 | 0.05% |
| record | actions.0.target | 4 | 0.05% |
| record | attendance: Percentage of days attended by participant | 4 | 0.05% |
| record | address: Physical address of the exhibitor | 4 | 0.05% |
| record | name: Full name of the exhibiting company | 4 | 0.05% |
| record | insurance_id | 4 | 0.05% |
| record | year_level | 4 | 0.05% |
| record | contract_id | 4 | 0.05% |
| record | participation | 4 | 0.05% |
| record | german_name | 4 | 0.05% |
| record | treatment: Type of treatment or management plan | 4 | 0.05% |
| record | zip_code: Postal code of patient's address | 4 | 0.05% |
| record | birth_date: Date of birth in MM/DD/YYYY format | 4 | 0.05% |
| record | state: State abbreviation of patient's residence | 4 | 0.05% |
| record | efficacy | 4 | 0.05% |
| record | prescriber_id | 4 | 0.05% |
| record | original_departure | 4 | 0.05% |
| record | id: Unique identifier assigned to the startup | 4 | 0.05% |
| record | sector: Technology domain such as health tech, AI, clean energy | 4 | 0.05% |
| record | function | 4 | 0.05% |
| record | birth_information.birth_place.state | 4 | 0.05% |
| record | bill_number | 4 | 0.05% |
| record | sunless_tanning.product_collection.collection_name | 4 | 0.05% |
| record | sunless_tanning.product_collection.products.1.benefits | 4 | 0.05% |
| record | consent_status | 4 | 0.05% |
| record | study_site | 4 | 0.05% |
| record | product.origin.spread_location | 4 | 0.05% |
| record | product.storage.wrapping | 4 | 0.05% |
| record | product.serving | 4 | 0.05% |
| record | yield_improvement | 4 | 0.05% |
| record | server_hiding.port_conflict | 4 | 0.05% |
| record | server_hiding.error_message | 4 | 0.05% |
| record | ports | 4 | 0.05% |
| record | treatment_dosage | 4 | 0.05% |
| record | arm | 4 | 0.05% |
| record | setting.political_context | 4 | 0.05% |
| record | main_character.rank | 4 | 0.05% |
| record | publication_details.first_publication.year | 4 | 0.05% |
| record | publication_details.year_written | 4 | 0.05% |
| record | publication_details.first_publication_czechoslovakia | 4 | 0.05% |
| record | original_title | 4 | 0.05% |
| record | return_status | 4 | 0.05% |
| record | eps: Earnings per share in dollars | 4 | 0.05% |
| record | shares: Number of shares the post received | 4 | 0.05% |
| record | recipient_location | 4 | 0.05% |
| record | administrative_level | 4 | 0.05% |
| record | location.county | 4 | 0.05% |
| record | artistic_intent.meaning_zero_point | 4 | 0.05% |
| record | israel_airports.largest_airport.departure_city | 4 | 0.05% |
| record | extra_options_prompt | 4 | 0.05% |
| record | passenger_information.infants.prompt | 4 | 0.05% |
| record | patent_number: Unique identifier for the patent application or grant | 4 | 0.05% |
| record | media_analysis.positive_aspects | 4 | 0.05% |
| record | agent: Support agent assigned to handle the request | 4 | 0.05% |
| record | current_inventory | 4 | 0.05% |
| record | advice | 4 | 0.05% |
| record | shipping_company | 4 | 0.05% |
| record | movie.based_on.author | 4 | 0.05% |
| record | actor.quote | 4 | 0.05% |
| record | points: Points scored by the player | 4 | 0.05% |
| record | expert_insights | 4 | 0.05% |
| record | years_mentioned | 4 | 0.05% |
| record | character_name | 4 | 0.05% |
| record | prescription_date | 4 | 0.05% |
| record | functionality_comment | 4 | 0.05% |
| record | app_version | 4 | 0.05% |
| record | damages | 4 | 0.05% |
| record | location: City and country of the reviewer | 4 | 0.05% |
| record | followupinterval | 4 | 0.05% |
| record | offeredBy | 4 | 0.05% |
| record | programStructure.dailySchedule.0.activities.1.description | 4 | 0.05% |
| record | programStructure.dailySchedule.1.activities.3.description | 4 | 0.05% |
| record | programStructure.dailySchedule.1.activities.0.description | 4 | 0.05% |
| record | programStructure.dailySchedule.0.activities.1.time | 4 | 0.05% |
| record | programName | 4 | 0.05% |
| record | certificationLevel | 4 | 0.05% |
| record | targetAudience | 4 | 0.05% |
| record | accommodation.options.1.phone | 4 | 0.05% |
| record | accommodation.options.0.website | 4 | 0.05% |
| record | accommodation.options.1.address | 4 | 0.05% |
| record | initiation_date | 4 | 0.05% |
| record | search_result_page.website_name | 4 | 0.05% |
| record | chamber | 4 | 0.05% |
| record | patient_case.demographics.age | 4 | 0.05% |
| record | disease_profile.typical_manifestations | 4 | 0.05% |
| record | game title | 4 | 0.05% |
| record | sales | 4 | 0.05% |
| record | town.location | 4 | 0.05% |
| record | organization_info.website | 4 | 0.05% |
| record | donations.recipient.type | 4 | 0.05% |
| record | organization_info.funding | 4 | 0.05% |
| record | filed_date: Date when the ticket was submitted | 4 | 0.05% |
| record | acquirer_ceo | 4 | 0.05% |
| record | heart_rate: Number of heartbeats per minute | 4 | 0.05% |
| record | condition_studied | 4 | 0.05% |
| record | agency.partners | 4 | 0.05% |
| record | citizen_corps.description | 4 | 0.05% |
| record | context.solution | 4 | 0.05% |
| record | test_cases_remaining | 4 | 0.05% |
| record | Edwin Harris Walker <> developed | 4 | 0.05% |
| record | Master Eckart <> responded to | 4 | 0.05% |
| record | Robert <> cited | 4 | 0.05% |
| record | Robert <> considered | 4 | 0.05% |
| record | Robert <> experienced | 4 | 0.05% |
| record | Robert <> identified as | 4 | 0.05% |
| record | Robert <> influenced by | 4 | 0.05% |
| record | Walker <> explained | 4 | 0.05% |
| record | Zen master <> replied to | 4 | 0.05% |
| record | asteroid.etymology.deity.name | 4 | 0.05% |
| record | stars: Number of stars the repository has received | 4 | 0.05% |
| record | hotel.location | 4 | 0.05% |
| record | review.positive_aspects | 4 | 0.05% |
| record | accuracy_efficiency: Performance metric like accuracy or efficiency | 4 | 0.05% |
| record | service_impact: Type of service affected, such as internet or mobile | 4 | 0.05% |
| record | ticket_id: Unique identifier for the outage ticket | 4 | 0.05% |
| record | key_features | 4 | 0.05% |
| record | issues | 4 | 0.05% |
| record | result_value: Measured value of the lab test | 4 | 0.05% |
| record | clinical_note: Clinical interpretation or patient-reported symptoms and recommendations | 4 | 0.05% |
| record | test_name: Name of the laboratory test performed | 4 | 0.05% |
| record | drug_involved | 4 | 0.05% |
| record | inr_value | 4 | 0.05% |
| record | boulder.location.distance | 4 | 0.05% |
| record | boulder.location.direction | 4 | 0.05% |
| record | drug_name: Name of the pharmaceutical product | 4 | 0.05% |
| record | mechanismofaction: Biological process inhibited or modulated by the drug | 4 | 0.05% |
| record | lastreviewdate: Date the drug record was last reviewed | 4 | 0.05% |
| record | adverse_effects: List of side effects reported in clinical use | 4 | 0.05% |
| record | target_receptor: Biological receptor the drug acts on | 4 | 0.05% |
| record | historical_context.my_folks_series.number_of_books | 4 | 0.05% |
| record | product.storage.containers | 4 | 0.05% |
| record | movie.production_year | 4 | 0.05% |
| record | movie.comments.9.date | 4 | 0.05% |
| record | movie.comments.10.time | 4 | 0.05% |
| record | movie.comments.12.text | 4 | 0.05% |
| record | movie.comments.14.time | 4 | 0.05% |
| record | movie.comments.19.time | 4 | 0.05% |
| record | target_regions | 4 | 0.05% |
| record | description: Brief description of the transaction activity | 4 | 0.05% |
| record | transaction_id: Unique identifier for each transaction | 4 | 0.05% |
| record | creditcardlast_four: Last four digits of the linked credit card | 4 | 0.05% |
| record | account_address: Full address associated with the account | 4 | 0.05% |
| record | rental_id | 4 | 0.05% |
| record | asteroid.asteroid_type | 4 | 0.05% |
| record | asteroid.named_after.person.birth_year | 4 | 0.05% |
| record | masters_of_ceremony | 4 | 0.05% |
| record | organizers | 4 | 0.05% |
| record | volume: Total shares traded on the day | 4 | 0.05% |
| record | fruit_properties.skin.description | 4 | 0.05% |
| record | rentescalaionclause | 4 | 0.05% |
| record | responsibility | 4 | 0.05% |
| record | dates_mentioned | 4 | 0.05% |
| record | contact_address | 4 | 0.05% |
| record | technologies_mentioned | 4 | 0.05% |
| record | description.type | 4 | 0.05% |
| record | description.leaf.dimensions | 4 | 0.05% |
| record | description.seed.dimensions | 4 | 0.05% |
| record | honorary_name.profession | 4 | 0.05% |
| record | dream_experiments | 4 | 0.05% |
| record | name: Full name of the esports tournament | 4 | 0.05% |
| record | venue: Name of the stadium or venue where the event occurred | 4 | 0.05% |
| record | publication.author.name | 4 | 0.05% |
| record | distribution | 4 | 0.05% |
| record | gene_id | 4 | 0.05% |
| record | rebounds: Rebounds recorded by the player | 4 | 0.05% |
| record | age_range: Age range of participants in the study | 4 | 0.05% |
| record | funding_source: Organization or grant funding the study | 4 | 0.05% |
| record | candidate_id: Identifier for the candidate receiving the donation | 4 | 0.05% |
| record | availability.locations.online_stores.0.url | 4 | 0.05% |
| record | course | 4 | 0.05% |
| record | expected_close_date | 4 | 0.05% |
| record | perpetrator.arrest_details.time | 4 | 0.05% |
| record | crime.weapon.source.stolen_from.name | 4 | 0.05% |
| record | receiver_name | 4 | 0.05% |
| record | product_range | 4 | 0.05% |
| record | contact_information.phone_number | 4 | 0.05% |
| record | headquarters.state | 4 | 0.05% |
| record | disease_profile.synonyms | 4 | 0.05% |
| record | target_audience: Demographic group the product targets | 4 | 0.05% |
| record | name: Full name of the employee as listed in the review | 4 | 0.05% |
| record | rainfall | 4 | 0.05% |
| record | resolution_method: Action taken to resolve the issue | 4 | 0.05% |
| record | error.message | 4 | 0.05% |
| record | claim_count: Number of claims in the patent | 4 | 0.05% |
| record | start_term | 4 | 0.05% |
| record | disease.type | 4 | 0.05% |
| record | methodology.techniques | 4 | 0.05% |
| record | employee_id: Unique identifier for the employee enrollment | 4 | 0.05% |
| record | pathway_name | 4 | 0.05% |
| record | protocol | 4 | 0.05% |
| record | claims_count: Number of claims in the patent | 4 | 0.05% |
| record | controversy.victim_actresses.6.krk_comment | 4 | 0.05% |
| record | controversy.victim_actresses.3.actress_name | 4 | 0.05% |
| record | medical condition | 4 | 0.05% |
| record | location: Geographic location of the research institution | 4 | 0.05% |
| record | institution: Research institution or lab conducting the study | 4 | 0.05% |
| record | participant_count: Total number of study participants | 4 | 0.05% |
| record | lead_investigator: Name of the principal researcher leading the network | 4 | 0.05% |
| record | funding_source: Grant or funding body supporting the network | 4 | 0.05% |
| record | player_age | 4 | 0.05% |
| record | appearances | 4 | 0.05% |
| record | publication_code | 4 | 0.05% |
| record | applicable_products | 4 | 0.05% |
| record | brand: Company name responsible for the product launch | 4 | 0.05% |
| record | location: Geographic market or city where product is available | 4 | 0.05% |
| record | study_type | 4 | 0.05% |
| record | description: Technical summary of the patented invention | 4 | 0.05% |
| record | geography.borders.2.direction | 4 | 0.05% |
| record | geography.borders.1.direction | 4 | 0.05% |
| record | infrastructure.refineries.0.refineryName | 4 | 0.05% |
| record | economy.gdpCurrency | 4 | 0.05% |
| record | infrastructure.highways.2.highwayName | 4 | 0.05% |
| record | economy.economicBase | 4 | 0.05% |
| record | history.mergerDescription | 4 | 0.05% |
| record | infrastructure.highways.0.highwayName | 4 | 0.05% |
| record | regionName | 4 | 0.05% |
| record | lowerLevelDivisions | 4 | 0.05% |
| record | infrastructure.refineries.0.closureDate | 4 | 0.05% |
| record | geography.borders.2.region | 4 | 0.05% |
| record | entity.coverage_details.headline_news_provider | 4 | 0.05% |
| record | name: Name of the product as listed in best-sellers | 4 | 0.05% |
| record | region: Primary customer region where product is popular | 4 | 0.05% |
| record | key_result | 4 | 0.05% |
| record | product_code: Internal code assigned to the product | 4 | 0.05% |
| record | location: Physical address of the product launch | 4 | 0.05% |
| record | technology_focus: Key technology or application area of the patent | 4 | 0.05% |
| record | assignee: Company owning the patent | 4 | 0.05% |
| record | assignee_location: Location of the patent assignee | 4 | 0.05% |
| record | credits.organization | 4 | 0.05% |
| record | credits.graphic_design.1.copyright_year | 4 | 0.05% |
| record | shipper | 4 | 0.05% |
| record | onset_timing | 4 | 0.05% |
| record | programStructure.dailySchedule.1.activities.0.time | 4 | 0.05% |
| record | fees.deposit | 4 | 0.05% |
| record | accommodation.options.1.name | 4 | 0.05% |
| record | Dariusz Śledź <> coached | 4 | 0.05% |
| record | program_code | 4 | 0.05% |
| record | main_character.appears_in_other_work | 4 | 0.05% |
| record | title: Title of the abstract work | 4 | 0.05% |
| record | submission_id: Unique identifier assigned to each abstract submission | 4 | 0.05% |
| record | outbreak_event.investigation.specimen_source | 4 | 0.05% |
| record | company: Name of the recipient's employer organization | 4 | 0.05% |
| record | species.etymology.named_after.0.location.city | 4 | 0.05% |
| record | description.size | 4 | 0.05% |
| record | description.width | 4 | 0.05% |
| record | species.etymology.named_after.0.location.country | 4 | 0.05% |
| record | patient_id: Unique patient identifier from lab system | 4 | 0.05% |
| record | patient_name: Name of the patient undergoing the test | 4 | 0.05% |
| record | hb: Hemoglobin level in grams per deciliter | 4 | 0.05% |
| record | platelets: Platelet count in thousands per microliter | 4 | 0.05% |
| record | status: Current status of invoice payment | 4 | 0.05% |
| record | details.calorie_burn.kettlebell_calories | 4 | 0.05% |
| record | genus.name_latin | 4 | 0.05% |
| record | genus.family | 4 | 0.05% |
| record | genus.synonyms | 4 | 0.05% |
| record | genus.subspecies.0.name | 4 | 0.05% |
| record | home_score: Final score for the home team | 4 | 0.05% |
| record | office_hours.0.start_time | 4 | 0.05% |
| record | office_hours.1.start_time | 4 | 0.05% |
| record | office_hours.0.day | 4 | 0.05% |
| record | distribution.locations | 4 | 0.05% |
| record | publisher: Publishing house responsible for the book | 4 | 0.05% |
| record | tax_incentive.duration.period.end_year | 4 | 0.05% |
| record | person.email_address | 4 | 0.05% |
| record | departure_actual | 4 | 0.05% |
| record | departure_original | 4 | 0.05% |
| record | publication_year: Year the network's findings were published | 4 | 0.05% |
| record | discovery.discovery_rank | 4 | 0.05% |
| record | affiliation.institution_name | 4 | 0.05% |
| record | research_background.animal_models_used | 4 | 0.05% |
| record | orbit.orbital_period.unit | 4 | 0.05% |
| record | address: Full physical address of the leased property | 4 | 0.05% |
| record | status: Current status of the lease agreement | 4 | 0.05% |
| record | adverbs | 4 | 0.05% |
| record | cardinal_numbers | 4 | 0.05% |
| record | geography.location | 4 | 0.05% |
| record | population_density | 4 | 0.05% |
| record | notes: Additional remarks or conditions for participation | 4 | 0.05% |
| record | underwriting_notes | 4 | 0.05% |
| record | guidelines.adherence | 4 | 0.05% |
| record | singles | 4 | 0.05% |
| record | works.type | 4 | 0.05% |
| record | birth.location | 4 | 0.05% |
| record | location-gpe | 4 | 0.05% |
| record | audience_rating | 4 | 0.05% |
| record | historical_context.time_periods.0.events.0.locations | 4 | 0.05% |
| record | historical_context.time_periods.2.period | 4 | 0.05% |
| record | training_program.payment.payment_instructions | 4 | 0.05% |
| record | training_program.payment.deposit_amount | 4 | 0.05% |
| record | dean.name | 4 | 0.05% |
| record | hashtags: List of relevant hashtags used in the tweet | 4 | 0.05% |
| record | likes: Number of likes the tweet received | 4 | 0.05% |
| record | useoffunds: Purpose or application of raised capital | 4 | 0.05% |
| record | person.inventions | 4 | 0.05% |
| record | schema_name | 4 | 0.05% |
| record | funding_request | 4 | 0.05% |
| record | contact_id: Unique identifier for the outreach contact | 4 | 0.05% |
| record | net_income | 4 | 0.05% |
| record | offensive lineman | 4 | 0.05% |
| record | primary_function: Core biological role of the pathway | 4 | 0.05% |
| record | related_articles | 4 | 0.05% |
| record | user_name: Name of the creator who posted the trend | 4 | 0.05% |
| record | video_id: Unique identifier for the video content | 4 | 0.05% |
| record | city: City where the office is physically located | 4 | 0.05% |
| record | office_code: Unique internal identifier for the office location | 4 | 0.05% |
| record | prokeimenon.mode | 4 | 0.05% |
| record | magnitudeorindex | 4 | 0.05% |
| record | grant_number | 4 | 0.05% |
| record | daily_value_percentage | 4 | 0.05% |
| record | methodology.population.inclusion_criteria | 4 | 0.05% |
| record | patient_demographics.gender_distribution.men.count | 4 | 0.05% |
| record | patient_demographics.gender_distribution.women.count | 4 | 0.05% |
| record | patient_demographics.age_stats.average | 4 | 0.05% |
| record | treatment_details.radiotherapy.type | 4 | 0.05% |
| record | treatment_details.surgical_timeline.interval_brain_to_lung_resection_days.range_max | 4 | 0.05% |
| record | treatment_details.pulmonary_resection_types.1.count | 4 | 0.05% |
| record | treatment_details.pulmonary_resection_types.2.procedure | 4 | 0.05% |
| record | pathological_findings.nodal_involvement_distribution.0.stage | 4 | 0.05% |
| record | pathological_findings.nodal_involvement_distribution.2.patient_count | 4 | 0.05% |
| record | pathological_findings.statistical_factors.0.factor | 4 | 0.05% |
| record | outcomes.overall_survival_months.average | 4 | 0.05% |
| record | outcomes.overall_survival_months.max | 4 | 0.05% |
| record | outcomes.survival_rates.one_year_percentage | 4 | 0.05% |
| record | outcomes.mortality_by_nodal_involvement.0.survival_limit_months | 4 | 0.05% |
| record | service_code: Internal code identifying the service type | 4 | 0.05% |
| record | outage_end: Time when the outage ended in HH:MM AM/PM format | 4 | 0.05% |
| record | ticket_number: Unique identifier for the outage ticket | 4 | 0.05% |
| record | training_program.name | 4 | 0.05% |
| record | details.days_ago | 4 | 0.05% |
| record | question.legal_implications | 4 | 0.05% |
| record | location: Physical address of the company headquarters | 4 | 0.05% |
| record | receiver | 4 | 0.05% |
| record | views | 4 | 0.05% |
| record | disease_link: Disease or condition associated with the pathway | 4 | 0.05% |
| record | clinical_relevance: Score indicating clinical importance | 4 | 0.05% |
| record | initiating_enzyme: Key enzyme initiating the pathway | 4 | 0.05% |
| record | protecting_group.targets.1.protected_functional_group | 4 | 0.05% |
| record | protecting_group.targets.0.abbreviation | 4 | 0.05% |
| record | protecting_group.targets.1.derivative_form | 4 | 0.05% |
| record | protecting_group.targets.1.amino_acid_name | 4 | 0.05% |
| record | conclusions.overall_effectiveness | 4 | 0.05% |
| record | protecting_group.targets.0.amino_acid_name | 4 | 0.05% |
| record | account_number | 4 | 0.05% |
| record | verse_reference | 4 | 0.05% |
| record | plaintiff_name | 4 | 0.05% |
| record | vote_id: Unique identifier for the legislative vote | 4 | 0.05% |
| record | document.date | 4 | 0.05% |
| record | document.key_issues | 4 | 0.05% |
| record | document.economic_impact.brand_value | 4 | 0.05% |
| record | document.economic_impact.growth_rate | 4 | 0.05% |
| record | document.strategy_elements.3.area | 4 | 0.05% |
| record | document.commissioners.0.name | 4 | 0.05% |
| record | consoles | 4 | 0.05% |
| record | review_gift.available_to | 4 | 0.05% |
| record | landmark | 4 | 0.05% |
| record | on_site_registration.location.name | 4 | 0.05% |
| record | material: Primary material composition of the gear | 4 | 0.05% |
| record | color: Available color(s) of the product | 4 | 0.05% |
| record | service_type: Type of service impacted such as broadband or mobile | 4 | 0.05% |
| record | regulation_mechanism: Mechanism governing pathway activity | 4 | 0.05% |
| record | discovery_year: Year the pathway was first identified | 4 | 0.05% |
| record | tenantimprovementsallowance | 4 | 0.05% |
| record | city.name | 4 | 0.05% |
| record | recipient_position: Office or position the contribution supports | 4 | 0.05% |
| record | area | 4 | 0.05% |
| record | reporting_agency | 4 | 0.05% |
| record | enrollment_capacity | 4 | 0.05% |
| record | user_email | 4 | 0.05% |
| record | title_fr | 4 | 0.05% |
| record | id: Unique identifier for each rental listing | 4 | 0.05% |
| record | material: Primary materials used in construction | 4 | 0.05% |
| record | weight: Weight of the product in grams or kg | 4 | 0.05% |
| record | article.publicationDetails.volume | 4 | 0.05% |
| record | film | 4 | 0.05% |
| record | price: Price of the product in dollars | 4 | 0.05% |
| record | search_result_page.related_to | 4 | 0.05% |
| record | fda_code | 4 | 0.05% |
| record | assigned_staff: Healthcare professional assigned to manage the device | 4 | 0.05% |
| record | pet_policy | 4 | 0.05% |
| record | section | 4 | 0.05% |
| record | headlines | 4 | 0.05% |
| record | faq_references | 4 | 0.05% |
| record | staff | 4 | 0.05% |
| record | article.author.givenName | 4 | 0.05% |
| record | party_a | 4 | 0.05% |
| record | party_b | 4 | 0.05% |
| record | serving_suggestions | 4 | 0.05% |
| record | requester | 4 | 0.05% |
| record | log_id | 4 | 0.05% |
| record | hba1c: Hemoglobin A1c percentage value | 4 | 0.05% |
| record | age: Age in years as stated in the record | 4 | 0.05% |
| record | responsible_party | 4 | 0.05% |
| record | correctiveactionplan | 4 | 0.05% |
| record | bp_diastolic | 4 | 0.05% |
| record | bp_systolic | 4 | 0.05% |
| record | functional_area | 4 | 0.05% |
| record | deck_coating_systems.0.resin_type | 4 | 0.05% |
| record | document.location | 4 | 0.05% |
| record | document.strategy_elements.1.area | 4 | 0.05% |
| record | document.commissioners.1.name | 4 | 0.05% |
| record | size_and_fit.measurements.0.value | 3 | 0.04% |
| record | size_and_fit.size_guide | 3 | 0.04% |
| record | size_and_fit.model_height.unit | 3 | 0.04% |
| record | care_instructions | 3 | 0.04% |
| record | size_and_fit.measurements.0.type | 3 | 0.04% |
| record | biography.teaching_position.mentor | 3 | 0.04% |
| record | birth_information.birth_place.county | 3 | 0.04% |
| record | filed_by | 3 | 0.04% |
| record | compensation_offered: Amount offered in dollars per passenger | 3 | 0.04% |
| record | award_code: Unique identifier code for the award | 3 | 0.04% |
| record | leasing_offer | 3 | 0.04% |
| record | date: Date the briefing was conducted | 3 | 0.04% |
| record | event_type: Type of event held during the briefing | 3 | 0.04% |
| record | weather | 3 | 0.04% |
| record | event.graduation_details.class_number | 3 | 0.04% |
| record | event.graduation_details.origin | 3 | 0.04% |
| record | event.officials_present.0.title | 3 | 0.04% |
| record | event.officials_present.0.affiliation | 3 | 0.04% |
| record | event.officials_present.0.rank | 3 | 0.04% |
| record | event.legal_basis.law_number | 3 | 0.04% |
| record | event.legal_basis.issuer | 3 | 0.04% |
| record | hours_of_operation.0.close_time | 3 | 0.04% |
| record | sourcing.baked_goods_source | 3 | 0.04% |
| record | sourcing.general_sourcing | 3 | 0.04% |
| record | hours_of_operation.6.close_time | 3 | 0.04% |
| record | on_site_registration.documents_required | 3 | 0.04% |
| record | table_of_contents | 3 | 0.04% |
| record | zebras_care_foundation.introducer.name | 3 | 0.04% |
| record | word_count: Total number of words in the paper | 3 | 0.04% |
| record | submission_date: Date when the paper was submitted | 3 | 0.04% |
| record | affiliation: Institutional affiliation of the author | 3 | 0.04% |
| record | renewal_date | 3 | 0.04% |
| record | away_team: Name of the team playing away | 3 | 0.04% |
| record | away_score: Final score for the away team | 3 | 0.04% |
| record | incident.charges.2.description | 3 | 0.04% |
| record | incident.charges.1.count | 3 | 0.04% |
| record | incident.suspects.0.age | 3 | 0.04% |
| record | incident.number_of_complaints | 3 | 0.04% |
| record | region.route | 3 | 0.04% |
| record | region.architecture | 3 | 0.04% |
| record | article.id | 3 | 0.04% |
| record | article.references.0.url | 3 | 0.04% |
| record | article.translation_note.disclaimer | 3 | 0.04% |
| record | status: Current status of the reaction | 3 | 0.04% |
| record | pages | 3 | 0.04% |
| record | research_submission | 3 | 0.04% |
| record | net_margin | 3 | 0.04% |
| record | claim_count: Number of claims filed in the patent application | 3 | 0.04% |
| record | invention: Technical description of the patented invention | 3 | 0.04% |
| record | auditstatus | 3 | 0.04% |
| record | liquidity_ratio | 3 | 0.04% |
| record | hotel.room_features | 3 | 0.04% |
| record | hotel.amenities | 3 | 0.04% |
| record | winner: Player who won the match | 3 | 0.04% |
| record | court | 3 | 0.04% |
| record | status: Current match status | 3 | 0.04% |
| record | no_votes | 3 | 0.04% |
| record | solution: Description of the proposed solution | 3 | 0.04% |
| record | serial_number: Unique serial number for device tracking | 3 | 0.04% |
| record | material: Material used in device construction | 3 | 0.04% |
| record | powers_abilities | 3 | 0.04% |
| record | policy_renewal_notice | 3 | 0.04% |
| record | experience | 3 | 0.04% |
| record | filing_fee | 3 | 0.04% |
| record | location: Geographic region where the product is targeted | 3 | 0.04% |
| record | product_description: Brief description of the product's benefits | 3 | 0.04% |
| record | nightly_rate | 3 | 0.04% |
| record | Obama administration drug czar <> discussed | 3 | 0.04% |
| record | United Food Commercial Workers Local 7 <> in talks with | 3 | 0.04% |
| record | Corona <> consumed in | 3 | 0.04% |
| record | Frontier <> competitor to | 3 | 0.04% |
| record | Southwest Airlines <> negotiated with | 3 | 0.04% |
| record | John McAfee <> involved in | 3 | 0.04% |
| record | John Mellencamp <> criticized | 3 | 0.04% |
| record | trademark_name: Name of the registered trademark in dispute | 3 | 0.04% |
| record | dispute_date: Date the dispute or opposition was filed | 3 | 0.04% |
| record | involved_parties: List of organizations or individuals involved in the incident | 3 | 0.04% |
| record | geographic_code: Unique code identifying the incident's geographic region | 3 | 0.04% |
| record | age_restriction | 3 | 0.04% |
| record | break_points | 3 | 0.04% |
| record | round | 3 | 0.04% |
| record | region: Geographic region where product is available | 3 | 0.04% |
| record | ratingcategory | 3 | 0.04% |
| record | patents | 3 | 0.04% |
| record | platform_name | 3 | 0.04% |
| record | earnout | 3 | 0.04% |
| record | sustainability_commitment | 3 | 0.04% |
| record | authority.year | 3 | 0.04% |
| record | authority.name | 3 | 0.04% |
| record | pricing: Monthly cost in dollars | 3 | 0.04% |
| record | product_id: Unique identifier for the LinkedIn product | 3 | 0.04% |
| record | impact: Description of human, financial, or systemic consequences | 3 | 0.04% |
| record | clause_description | 3 | 0.04% |
| record | reference_number | 3 | 0.04% |
| record | execution_date | 3 | 0.04% |
| record | enforceability_status | 3 | 0.04% |
| record | breach_penalty_percent | 3 | 0.04% |
| record | term_start_date | 3 | 0.04% |
| record | term_end_date | 3 | 0.04% |
| record | name: Full name of the policyholder | 3 | 0.04% |
| record | renewal_date: Date when the policy is set to renew | 3 | 0.04% |
| record | reporter_address: Full address of the customer reporter | 3 | 0.04% |
| record | charging_type | 3 | 0.04% |
| record | mls_number | 3 | 0.04% |
| record | characteristics.subtype | 3 | 0.04% |
| record | characteristics.effective_temperature.unit | 3 | 0.04% |
| record | id: Unique identifier for the publication record | 3 | 0.04% |
| record | title: Topic or subject of the diplomatic coverage | 3 | 0.04% |
| record | page: Page number where the digest appears | 3 | 0.04% |
| record | status: Current availability status of the publication | 3 | 0.04% |
| record | response | 3 | 0.04% |
| record | sender: Name of the person sending the outreach | 3 | 0.04% |
| record | outreach_date: Date when the outreach email was sent | 3 | 0.04% |
| record | location.distance_from_capital.direction | 3 | 0.04% |
| record | location.country.part_of | 3 | 0.04% |
| record | online service | 3 | 0.04% |
| record | location: City and state where product is located | 3 | 0.04% |
| record | name: Product display name as listed in the entry | 3 | 0.04% |
| record | symptom_fatigue | 3 | 0.04% |
| record | symptom_joint_pain | 3 | 0.04% |
| record | weight_loss_start_date | 3 | 0.04% |
| record | referral_specialty | 3 | 0.04% |
| record | chemical_name: Name of the chemical substance | 3 | 0.04% |
| record | grant_code | 3 | 0.04% |
| record | medications: List of prescribed medications with dosing | 3 | 0.04% |
| record | encounter_id: Unique identifier for the visit | 3 | 0.04% |
| record | icd_code: ICD-10 code for the primary diagnosis | 3 | 0.04% |
| record | lab_results: List of lab test results | 3 | 0.04% |
| record | bp_diastolic: Diastolic blood pressure in mmHg | 3 | 0.04% |
| record | comparative_authors | 3 | 0.04% |
| record | study_details.cohort.condition_studied | 3 | 0.04% |
| record | nurse | 3 | 0.04% |
| record | currency_pair: Two-currency pair such as EUR/USD or SGD/USD | 3 | 0.04% |
| record | controversy.victim_actresses.0.krk_comment | 3 | 0.04% |
| record | controversy.person_involved.0.name | 3 | 0.04% |
| record | controversy.victim_actresses.5.actress_name | 3 | 0.04% |
| record | controversy.victim_actresses.0.actress_name | 3 | 0.04% |
| record | controversy.person_involved.1.name | 3 | 0.04% |
| record | name: Title or name of the event | 3 | 0.04% |
| record | organizer: Person or group hosting the event | 3 | 0.04% |
| record | price: Entry cost for attending the event | 3 | 0.04% |
| record | attendees: Number of people registered for the event | 3 | 0.04% |
| record | sound_quality | 3 | 0.04% |
| record | social_media_post.hashtags | 3 | 0.04% |
| record | previous_incident.year | 3 | 0.04% |
| record | location: Physical or remote location of the submitter | 3 | 0.04% |
| record | price: Current selling price of the product in USD | 3 | 0.04% |
| record | legislative_details.referred_committee | 3 | 0.04% |
| record | album_series | 3 | 0.04% |
| record | position_code: Unique identifier for the appointment | 3 | 0.04% |
| record | delivery_frequency | 3 | 0.04% |
| record | feature | 3 | 0.04% |
| record | exposure_route | 3 | 0.04% |
| record | path_design.concrete_tile_making.equipment | 3 | 0.04% |
| record | path_design.methods | 3 | 0.04% |
| record | path_design.concrete_tile_making.tips | 3 | 0.04% |
| record | path_design.advantages | 3 | 0.04% |
| record | product_maturity | 3 | 0.04% |
| record | follow_up_interest | 3 | 0.04% |
| record | pathway | 3 | 0.04% |
| record | no_votes: Number of members voting against | 3 | 0.04% |
| record | investor_group | 3 | 0.04% |
| record | alternate_designation | 3 | 0.04% |
| record | vehicle.specifications.fuel_tank | 3 | 0.04% |
| record | vehicle.features | 3 | 0.04% |
| record | vehicle.operators.0.acquisition.quantity | 3 | 0.04% |
| record | vehicle.combat_usage.0.description | 3 | 0.04% |
| record | vehicle.operators.0.country | 3 | 0.04% |
| record | vehicle.specifications.dimensions.width | 3 | 0.04% |
| record | tissue_type | 3 | 0.04% |
| record | elements | 3 | 0.04% |
| record | drug_type | 3 | 0.04% |
| record | status: Current medical status of the patient | 3 | 0.04% |
| record | research.title | 3 | 0.04% |
| record | research.experiment_details.atoms_used | 3 | 0.04% |
| record | salary.hourly_rate.amount | 3 | 0.04% |
| record | salary.hourly_rate.currency | 3 | 0.04% |
| record | salary.monthly_estimate.currency | 3 | 0.04% |
| record | restaurant.renovations.0.type | 3 | 0.04% |
| record | restaurant.name | 3 | 0.04% |
| record | company.address.shop_name | 3 | 0.04% |
| record | company.address.city | 3 | 0.04% |
| record | company.products | 3 | 0.04% |
| record | submitted_date: Date the ticket was submitted | 3 | 0.04% |
| record | employee_name: Name of employee who reported the issue | 3 | 0.04% |
| record | address: Full address of the payee including street, city, state, and zip | 3 | 0.04% |
| record | event.start_time | 3 | 0.04% |
| record | drivers.0.nationality | 3 | 0.04% |
| record | person.biography.exile.year | 3 | 0.04% |
| record | person.works.0.year | 3 | 0.04% |
| record | person.works.0.title | 3 | 0.04% |
| record | person.works.0.type | 3 | 0.04% |
| record | person.biography.political_involvement.start_year | 3 | 0.04% |
| record | person.biography.marriage.spouse | 3 | 0.04% |
| record | size_and_fit.measurements.2.unit | 3 | 0.04% |
| record | size_and_fit.length.value | 3 | 0.04% |
| record | author_information.location.city | 3 | 0.04% |
| record | email: Email address associated with the reviewer | 3 | 0.04% |
| record | device: Name of the wearable device being reviewed | 3 | 0.04% |
| record | word_count: Number of words in the review text | 3 | 0.04% |
| record | status: Current transaction processing state | 3 | 0.04% |
| record | drug_name: Name of the prescribed medication | 3 | 0.04% |
| record | frequency: How often the medication is taken | 3 | 0.04% |
| record | mechanism: Biological action or mechanism of the drug | 3 | 0.04% |
| record | prescription: Medication or therapy prescribed | 3 | 0.04% |
| record | mrn: Unique medical record number identifying the patient | 3 | 0.04% |
| record | soil_remediation_project | 3 | 0.04% |
| record | ranking.location_type | 3 | 0.04% |
| record | department: Business unit or division where the role is located | 3 | 0.04% |
| record | required_qualifications: List of educational or skill-based requirements | 3 | 0.04% |
| record | parties: List of countries involved in the treaty | 3 | 0.04% |
| record | treaty_id: Unique identifier assigned to the treaty | 3 | 0.04% |
| record | creation.king_reign_year | 3 | 0.04% |
| record | kingdom | 3 | 0.04% |
| record | location.country.name | 3 | 0.04% |
| record | location: Warehouse or distribution center address and city state zip | 3 | 0.04% |
| record | sku: Unique product identifier for inventory tracking | 3 | 0.04% |
| record | failure_reason | 3 | 0.04% |
| record | enrollment_date: Date when enrollment began | 3 | 0.04% |
| record | location: Geographic location of the enrollment site | 3 | 0.04% |
| record | githubrepourl | 3 | 0.04% |
| record | policy_amount | 3 | 0.04% |
| record | school: Name of the school where the exam was administered | 3 | 0.04% |
| record | video_title | 3 | 0.04% |
| record | employment.0.company | 3 | 0.04% |
| record | employment.1.company | 3 | 0.04% |
| record | employment.2.company | 3 | 0.04% |
| record | predeceased.0.name | 3 | 0.04% |
| record | predeceased.0.relationship | 3 | 0.04% |
| record | predeceased.1.relationship | 3 | 0.04% |
| record | predeceased.2.relationship | 3 | 0.04% |
| record | predeceased.4.relationship | 3 | 0.04% |
| record | predeceased.5.name | 3 | 0.04% |
| record | siblings_in_law.2.name | 3 | 0.04% |
| record | siblings_in_law.2.location | 3 | 0.04% |
| record | death_location | 3 | 0.04% |
| record | grandnieces_and_nephews | 3 | 0.04% |
| record | children.0.spouse | 3 | 0.04% |
| record | children.1.name | 3 | 0.04% |
| record | children.1.spouse | 3 | 0.04% |
| record | residence | 3 | 0.04% |
| record | spouse | 3 | 0.04% |
| record | study_metadata.measurement_technology.abbreviation | 3 | 0.04% |
| record | study_metadata.measurement_technology.full_name | 3 | 0.04% |
| record | population.age_range.min | 3 | 0.04% |
| record | outcomes.fracture_events.0.type | 3 | 0.04% |
| record | statistical_results.hazard_ratios.0.adjustment_factors | 3 | 0.04% |
| record | statistical_results.hazard_ratios.0.comparison | 3 | 0.04% |
| record | statistical_results.diagnostic_metrics.sensitivity_at_threshold.t_score_threshold | 3 | 0.04% |
| record | statistical_results.predictive_model.variables_included | 3 | 0.04% |
| record | specimen_type | 3 | 0.04% |
| record | violation_type: Category of compliance issue identified | 3 | 0.04% |
| record | finding_id: Unique identifier for the audit finding | 3 | 0.04% |
| record | violation_code: Code assigned to the specific violation | 3 | 0.04% |
| record | vehiclemakeand_model | 3 | 0.04% |
| record | authors: List of physicians who developed the guideline | 3 | 0.04% |
| record | target_population: Age and health condition group the guideline targets | 3 | 0.04% |
| record | language: Primary language of the guideline | 3 | 0.04% |
| record | department: Government department or agency where the role is held | 3 | 0.04% |
| record | address: Physical address of the appointee | 3 | 0.04% |
| record | location: Geographic location of the channel's base | 3 | 0.04% |
| record | clinical_observations.activation_status | 3 | 0.04% |
| record | Dariusz Śledź <> participated in | 3 | 0.04% |
| record | Dariusz Śledź <> nationality | 3 | 0.04% |
| record | checklist_id: Unique identifier for the symptom checklist form | 3 | 0.04% |
| record | status: Current status of the checklist record | 3 | 0.04% |
| record | services_offered | 3 | 0.04% |
| record | address: Physical address of the donor as recorded | 3 | 0.04% |
| record | product_id: Unique identifier for each product in the best-sellers list | 3 | 0.04% |
| record | vehicle: Make and model of the vehicle involved | 3 | 0.04% |
| record | collision_type: Type of collision or damage event | 3 | 0.04% |
| record | adjuster: Name of the insurance adjuster assigned | 3 | 0.04% |
| record | location: Geographic location where the incident occurred | 3 | 0.04% |
| record | renewal_amount: Annual premium amount for the renewal | 3 | 0.04% |
| record | exam_code: Code identifying the specific exam conducted | 3 | 0.04% |
| record | funding_target | 3 | 0.04% |
| record | battery_notes | 3 | 0.04% |
| record | expiry_date: Date when the device's warranty or use period ends | 3 | 0.04% |
| record | sunless_tanning.product_collection.products.1.type | 3 | 0.04% |
| record | sunless_tanning.product_collection.brand | 3 | 0.04% |
| record | sunless_tanning.product_collection.products.0.benefits | 3 | 0.04% |
| record | sunless_tanning.product_collection.products.0.sizes | 3 | 0.04% |
| record | sunless_tanning.concerns | 3 | 0.04% |
| record | hangul_name | 3 | 0.04% |
| record | growth_rate: Year-over-year growth percentage | 3 | 0.04% |
| record | arrival_code | 3 | 0.04% |
| record | reporter_name: Name of the customer who reported the issue | 3 | 0.04% |
| record | methodology.proposed_solution | 3 | 0.04% |
| record | methodology.experimental_setup.biological_model.culture_environment.material_properties | 3 | 0.04% |
| record | methodology.experimental_setup.reagents.0.labeling | 3 | 0.04% |
| record | rank_order | 3 | 0.04% |
| record | location: City and state where the hotel is located | 3 | 0.04% |
| record | rating: Star rating from 1 to 5 given by the reviewer | 3 | 0.04% |
| record | test_date | 3 | 0.04% |
| record | research_context.problem_statement.affected_research_fields | 3 | 0.04% |
| record | methodology.experimental_setup.instrumentation | 3 | 0.04% |
| record | methodology.experimental_setup.biological_model.culture_environment.support_type | 3 | 0.04% |
| record | advertiser_information.new_brochure_planned | 3 | 0.04% |
| record | distribution_information.trail_details.trail_length | 3 | 0.04% |
| record | relationship_to_subject | 3 | 0.04% |
| record | nutritional_aspects.carbohydrates.importance | 3 | 0.04% |
| record | nutritional_aspects.carbohydrates.low_glycemic_index.percentage_with_fiber | 3 | 0.04% |
| record | nutritional_aspects.sucrose.daily_limit_grams | 3 | 0.04% |
| record | representatives.0.statements | 3 | 0.04% |
| record | posting platform | 3 | 0.04% |
| record | morphology.leaves | 3 | 0.04% |
| record | policyholder: Name of the policyholder or business owner | 3 | 0.04% |
| record | risk_category: Categorical classification of risk profile | 3 | 0.04% |
| record | professional_memberships | 3 | 0.04% |
| record | product.included_software | 3 | 0.04% |
| record | product.execution_file | 3 | 0.04% |
| record | removal_instructions.browser_specific_instructions.2.steps | 3 | 0.04% |
| record | behavior | 3 | 0.04% |
| record | discharge_instructions: Written guidance given to patient post-discharge | 3 | 0.04% |
| record | incidence.rarity | 3 | 0.04% |
| record | name: Name of the person associated with the transaction | 3 | 0.04% |
| record | status: Current approval status of the transaction | 3 | 0.04% |
| record | genus.german_species | 3 | 0.04% |
| record | genus.established_by | 3 | 0.04% |
| record | recharge_amount | 3 | 0.04% |
| record | recipient: Name of the person receiving the confirmation | 3 | 0.04% |
| record | transfertype | 3 | 0.04% |
| record | startup_capital | 3 | 0.04% |
| record | asteroid.orbital_period.unit | 3 | 0.04% |
| record | asteroid.discovery.discoverer.profession | 3 | 0.04% |
| record | asteroid.naming.named_after | 3 | 0.04% |
| record | media_analysis.events.0.national_media_response | 3 | 0.04% |
| record | vendorinvoiceemail | 3 | 0.04% |
| record | paymentduedate | 3 | 0.04% |
| record | title: Title of the literary work reviewed | 3 | 0.04% |
| record | isbn: Unique identifier for the book or collection | 3 | 0.04% |
| record | author: Name of the author or poet | 3 | 0.04% |
| record | product.size | 3 | 0.04% |
| record | review.usage_experience | 3 | 0.04% |
| record | role: Job title or position held by the employee | 3 | 0.04% |
| record | department: Department where the individual is assigned | 3 | 0.04% |
| record | device_condition | 3 | 0.04% |
| record | serial_number: Unique serial number assigned to the device | 3 | 0.04% |
| record | scholarship_id | 3 | 0.04% |
| record | pathophysiology.mechanism.secondary_event.dislocated_structure | 3 | 0.04% |
| record | pathophysiology.mechanism.primary_event | 3 | 0.04% |
| record | contact_email: Email address of primary passenger for communication | 3 | 0.04% |
| record | claim_date: Date when the incident occurred | 3 | 0.04% |
| record | recipient_company | 3 | 0.04% |
| record | venue: Stadium where the match took place | 3 | 0.04% |
| record | current_scale | 3 | 0.04% |
| record | patient_age: Age of the patient at time of reaction | 3 | 0.04% |
| record | definition.source_organization.name | 3 | 0.04% |
| record | definition.text | 3 | 0.04% |
| record | aspects | 3 | 0.04% |
| record | business_sustainability.elements | 3 | 0.04% |
| record | financial_aspect.necessity | 3 | 0.04% |
| record | financial_aspect.relevance_in_crisis | 3 | 0.04% |
| record | industry_example.actions | 3 | 0.04% |
| record | initial_funding | 3 | 0.04% |
| record | storage_temp | 3 | 0.04% |
| record | cohort_size | 3 | 0.04% |
| record | base_fee | 3 | 0.04% |
| record | pre_conference_workshop | 3 | 0.04% |
| record | student_rate | 3 | 0.04% |
| record | complimentary_items | 3 | 0.04% |
| record | no_show_reduction | 3 | 0.04% |
| record | passenger_street | 3 | 0.04% |
| record | discharge_destination | 3 | 0.04% |
| record | network_context | 3 | 0.04% |
| record | model_id | 3 | 0.04% |
| record | output_power | 3 | 0.04% |
| record | policy_number: Unique policy number associated with the claim | 3 | 0.04% |
| record | incident_type: Description of the type of incident | 3 | 0.04% |
| record | data_code | 3 | 0.04% |
| record | album.sources.2.access_date | 3 | 0.04% |
| record | album.tracklist.cd.10.title | 3 | 0.04% |
| record | album.tracklist.cd.5.lyrics | 3 | 0.04% |
| record | album.tracklist.cd.5.track_number | 3 | 0.04% |
| record | album.tracklist.cd.4.tie_in.0.title | 3 | 0.04% |
| record | album.tracklist.cd.9.title | 3 | 0.04% |
| record | album.tracklist.cd.9.tie_in.0.title | 3 | 0.04% |
| record | album.sources.1.url | 3 | 0.04% |
| record | album.sources.0.publication_date | 3 | 0.04% |
| record | album.tracklist.dvd.2.track_number | 3 | 0.04% |
| record | album.tracklist.cd.5.notes | 3 | 0.04% |
| record | album.sources.3.access_date | 3 | 0.04% |
| record | album.versions.3.type | 3 | 0.04% |
| record | album.tracklist.cd.0.track_number | 3 | 0.04% |
| record | album.tracklist.cd.7.notes | 3 | 0.04% |
| record | album.versions.3.name | 3 | 0.04% |
| record | album.credits.basic_arrangement | 3 | 0.04% |
| record | album.tracklist.cd.1.composer | 3 | 0.04% |
| record | album.tracklist.cd.3.tie_in.0.type | 3 | 0.04% |
| record | album.sources.2.publication_date | 3 | 0.04% |
| record | album.tracklist.dvd.2.title | 3 | 0.04% |
| record | album.versions.3.contents | 3 | 0.04% |
| record | album.versions.1.name | 3 | 0.04% |
| record | album.chart_performance.consecutive_number_one_albums | 3 | 0.04% |
| record | album.sources.3.reference_number | 3 | 0.04% |
| record | album.tracklist.cd.0.tie_in.0.title | 3 | 0.04% |
| record | album.versions.0.name | 3 | 0.04% |
| record | album.tracklist.cd.4.tie_in.0.type | 3 | 0.04% |
| record | album.sources.3.source_name | 3 | 0.04% |
| record | album.sources.0.url | 3 | 0.04% |
| record | album.tracklist.cd.0.lyrics | 3 | 0.04% |
| record | album.credits.producer | 3 | 0.04% |
| record | album.tracklist.cd.7.title | 3 | 0.04% |
| record | appointment_code | 3 | 0.04% |
| record | background | 3 | 0.04% |
| record | colors | 3 | 0.04% |
| record | birth_date | 3 | 0.04% |
| record | movie.based_on.title | 3 | 0.04% |
| record | movie.character | 3 | 0.04% |
| record | status: Current transaction status | 3 | 0.04% |
| record | name: Full name of the enrolled participant | 3 | 0.04% |
| record | return_amount | 3 | 0.04% |
| record | sunset_valley_fest | 3 | 0.04% |
| record | physical_properties.temperature.value | 3 | 0.04% |
| record | physical_properties.temperature.unit | 3 | 0.04% |
| record | formation.formation_mechanism | 3 | 0.04% |
| record | observations.0.telescope_instrument | 3 | 0.04% |
| record | observations.0.wavelength | 3 | 0.04% |
| record | distance.value | 3 | 0.04% |
| record | cultural_references.0.title | 3 | 0.04% |
| record | object_name | 3 | 0.04% |
| record | orbital_properties.separation.value | 3 | 0.04% |
| record | discovery.telescopes.0.instruments | 3 | 0.04% |
| record | discovery.telescopes.0.name | 3 | 0.04% |
| record | discovery.year_detected | 3 | 0.04% |
| record | date_verified: Date the drug's information was last verified | 3 | 0.04% |
| record | pathway: Biological pathway affected by the drug | 3 | 0.04% |
| record | model_id: Unique identifier for each wireless audio device | 3 | 0.04% |
| record | listed_price: Retail price of the audio device in US dollars | 3 | 0.04% |
| record | expiration_date: Date after which device is no longer valid | 3 | 0.04% |
| record | cost: Total cost of the device in dollars | 3 | 0.04% |
| record | device_name: Name of the medical device as listed | 3 | 0.04% |
| record | purchase_date: Date the device was acquired | 3 | 0.04% |
| record | model: Device model name and version | 3 | 0.04% |
| record | airport.location | 3 | 0.04% |
| record | terminal.impact.passenger_decrease | 3 | 0.04% |
| record | terminal.history.0.description | 3 | 0.04% |
| record | terminal.history.0.period | 3 | 0.04% |
| record | terminal.history.2.period | 3 | 0.04% |
| record | related_topics | 3 | 0.04% |
| record | conference_date | 3 | 0.04% |
| record | brand: Name of the brand launching the product | 3 | 0.04% |
| record | target_audience: Demographic or interest group targeted | 3 | 0.04% |
| record | campaign_code: Internal code for the campaign | 3 | 0.04% |
| record | impact_rainfall | 3 | 0.04% |
| record | zip_code: Postal code associated with the office address | 3 | 0.04% |
| record | person.professions | 3 | 0.04% |
| record | person.memberships | 3 | 0.04% |
| record | price: Current stock price in dollars | 3 | 0.04% |
| record | reaction | 3 | 0.04% |
| record | key_enzyme | 3 | 0.04% |
| record | lot_number | 3 | 0.04% |
| record | article.entities.people.1.organization | 3 | 0.04% |
| record | article.entities.people.1.name | 3 | 0.04% |
| record | article.publication_details.publication_date | 3 | 0.04% |
| record | vaccination_status | 3 | 0.04% |
| record | id: Unique identifier for each exchange rate record | 3 | 0.04% |
| record | door | 3 | 0.04% |
| record | natural_elements | 3 | 0.04% |
| record | supernatural | 3 | 0.04% |
| record | work_hours | 3 | 0.04% |
| record | funding_type | 3 | 0.04% |
| record | first_office | 3 | 0.04% |
| record | resolution_time: Duration in hours to resolve the issue | 3 | 0.04% |
| record | days_left_in_year | 3 | 0.04% |
| record | chemical_name: Name of the chemical compound | 3 | 0.04% |
| record | hazard_class: Classification of chemical hazard | 3 | 0.04% |
| record | hazard_statement: Statement describing the chemical's hazard | 3 | 0.04% |
| record | storage: Conditions required for safe storage | 3 | 0.04% |
| record | disposal: Procedure for proper disposal of the chemical | 3 | 0.04% |
| record | municipality.name_local | 3 | 0.04% |
| record | posted_on | 3 | 0.04% |
| record | affected_systems | 3 | 0.04% |
| record | solution | 3 | 0.04% |
| record | line1 | 3 | 0.04% |
| record | appearances: Total games played in last season | 3 | 0.04% |
| record | club_from: Previous club before transfer | 3 | 0.04% |
| record | age: Player's age at time of transfer | 3 | 0.04% |
| record | description: Summary of the bill's purpose and scope | 3 | 0.04% |
| record | sponsor: Name of the bill's sponsor | 3 | 0.04% |
| record | validation_method | 3 | 0.04% |
| record | vin | 3 | 0.04% |
| record | median_age | 3 | 0.04% |
| record | whoadrcode | 3 | 0.04% |
| record | followupaction | 3 | 0.04% |
| record | promo_code | 3 | 0.04% |
| record | affected_system | 3 | 0.04% |
| record | resolution_method | 3 | 0.04% |
| record | critic_name | 3 | 0.04% |
| record | asteroid.description.discovery_details.project | 3 | 0.04% |
| record | asteroid.orbital_characteristics.semi_major_axis | 3 | 0.04% |
| record | asteroid.description.discovery_details.location.country | 3 | 0.04% |
| record | asteroid.alternative_designations | 3 | 0.04% |
| record | transaction_id: Unique identifier for each transaction event | 3 | 0.04% |
| record | causes_for_inedible_sprouts | 3 | 0.04% |
| record | key_feature | 3 | 0.04% |
| record | distribution.regions.2.region_name | 3 | 0.04% |
| record | user: Name of the user submitting the ticket | 3 | 0.04% |
| record | email: User's contact email address | 3 | 0.04% |
| record | donor | 3 | 0.04% |
| record | product: Description of the product or service purchased | 3 | 0.04% |
| record | address_state | 3 | 0.04% |
| record | address_zip | 3 | 0.04% |
| record | physician.name | 3 | 0.04% |
| record | physician.address.street | 3 | 0.04% |
| record | physician.address.zip | 3 | 0.04% |
| record | physician.education.fellowship.0.institution | 3 | 0.04% |
| record | stub_article.message | 3 | 0.04% |
| record | project_category | 3 | 0.04% |
| record | product.availability.future_availability | 3 | 0.04% |
| record | product.availability.currently_available | 3 | 0.04% |
| record | product.specifications.processor | 3 | 0.04% |
| record | product.specifications.camera.front | 3 | 0.04% |
| record | lead_agency.primary_goal | 3 | 0.04% |
| record | lead_agency.abbreviation | 3 | 0.04% |
| record | author: Name of the guideline's lead author | 3 | 0.04% |
| record | early_bird_discount_ends | 3 | 0.04% |
| record | tracks | 3 | 0.04% |
| record | keynote_speakers | 3 | 0.04% |
| record | workshop_date | 3 | 0.04% |
| record | event_status | 3 | 0.04% |
| record | lead_organizer | 3 | 0.04% |
| record | name: Full title of the course including subject and focus | 3 | 0.04% |
| record | last_updated: Date when inventory was last modified | 3 | 0.04% |
| record | description.fruit.shape | 3 | 0.04% |
| record | description.flowers.perianth_tube_length_mm.max | 3 | 0.04% |
| record | description.flowers.length_mm.max | 3 | 0.04% |
| record | systematics_and_distribution.habitat | 3 | 0.04% |
| record | description.leaves.shape | 3 | 0.04% |
| record | reorder_point: Minimum stock level before triggering reorder | 3 | 0.04% |
| record | drug_prescribed | 3 | 0.04% |
| record | affiliation.organization_name | 3 | 0.04% |
| record | sectorial_libraries_coordinated | 3 | 0.04% |
| record | tribute_to.person_name | 3 | 0.04% |
| record | library_name | 3 | 0.04% |
| record | duration: Duration of the training program in weeks | 3 | 0.04% |
| record | employee_name: Full name of the enrolled employee | 3 | 0.04% |
| record | required_materials | 3 | 0.04% |
| record | attendance_limit | 3 | 0.04% |
| record | Food | 3 | 0.04% |
| record | officer | 3 | 0.04% |
| record | context.region | 3 | 0.04% |
| record | project.operational_model.stakeholders.2.description | 3 | 0.04% |
| record | project.financials.budget | 3 | 0.04% |
| record | source.name | 3 | 0.04% |
| record | tv_program.production.location.type | 3 | 0.04% |
| record | other_programs | 3 | 0.04% |
| record | tv_program.production.channel | 3 | 0.04% |
| record | turning_point | 3 | 0.04% |
| record | discovery.astronomer.name | 3 | 0.04% |
| record | company: Name of the technology company receiving funds | 3 | 0.04% |
| record | ticket_id: Unique identifier for the return ticket | 3 | 0.04% |
| record | handling_instructions | 3 | 0.04% |
| record | person.associated_with | 3 | 0.04% |
| record | originalordernumber | 3 | 0.04% |
| record | description.rosette | 3 | 0.04% |
| record | description.leaves.color | 3 | 0.04% |
| record | systematics_and_distribution.first_description.year | 3 | 0.04% |
| record | systematics_and_distribution.location.elevation_m.max | 3 | 0.04% |
| record | description.flowers.length_mm.min | 3 | 0.04% |
| record | description.fruit.type | 3 | 0.04% |
| record | hardware_support.gpu.nvidia.api | 3 | 0.04% |
| record | clinical_indication: Medical condition the drug is used to treat | 3 | 0.04% |
| record | mechanismofaction: Description of how the drug exerts its effect | 3 | 0.04% |
| record | resolution_estimate: Estimated hours to resolve the issue | 3 | 0.04% |
| record | issue_description: Detailed description of the reported problem | 3 | 0.04% |
| record | subjects_involved | 3 | 0.04% |
| record | expert_opinion.speaker.affiliation.organization | 3 | 0.04% |
| record | expert_opinion.speaker.name | 3 | 0.04% |
| record | availability.locations.online_stores.0.name | 3 | 0.04% |
| record | product_example.color | 3 | 0.04% |
| record | product_example.size | 3 | 0.04% |
| record | product_example.price | 3 | 0.04% |
| record | event.action | 3 | 0.04% |
| record | social_media_post.content | 3 | 0.04% |
| record | source.type | 3 | 0.04% |
| record | school: Name of the high school or institution | 3 | 0.04% |
| record | measurement.unit | 3 | 0.04% |
| record | measurement.abbreviation | 3 | 0.04% |
| record | origin.originator | 3 | 0.04% |
| record | origin.university | 3 | 0.04% |
| record | applications.industrial | 3 | 0.04% |
| record | star_name | 3 | 0.04% |
| record | review.persons.1.role | 3 | 0.04% |
| record | review.performance.choreographer | 3 | 0.04% |
| record | review.persons.2.description | 3 | 0.04% |
| record | review.performance.director | 3 | 0.04% |
| record | review.author.name | 3 | 0.04% |
| record | review.date | 3 | 0.04% |
| record | location: Physical office location affected | 3 | 0.04% |
| record | title_english | 3 | 0.04% |
| record | start_year | 3 | 0.04% |
| record | partner_types | 3 | 0.04% |
| record | hotel_initiatives | 3 | 0.04% |
| record | tone | 3 | 0.04% |
| record | sender: Name of the recruiter or agency sending the message | 3 | 0.04% |
| record | winner: Candidate who won the district | 3 | 0.04% |
| record | district_id: Unique identifier for the electoral district | 3 | 0.04% |
| record | claim_id: Unique identifier for each claim | 3 | 0.04% |
| record | notes: Additional remarks or clinical context about the claim | 3 | 0.04% |
| record | financial_parameters | 3 | 0.04% |
| record | red_cards | 3 | 0.04% |
| record | root_cause: Hypothesized cause of the incident based on logs or analysis | 3 | 0.04% |
| record | address: Complete address of the patient | 3 | 0.04% |
| record | meddra_code: Standard code for the reaction in MedDRA system | 3 | 0.04% |
| record | organizer.name | 3 | 0.04% |
| record | expectedOutcome | 3 | 0.04% |
| record | goal | 3 | 0.04% |
| record | treaty_name | 3 | 0.04% |
| record | signatories | 3 | 0.04% |
| record | treaty_code | 3 | 0.04% |
| record | signing_countries | 3 | 0.04% |
| record | maintenance_notes | 3 | 0.04% |
| record | actual_spend | 3 | 0.04% |
| record | risk_code | 3 | 0.04% |
| record | tracking_number: Shipping tracking number assigned | 3 | 0.04% |
| record | equipment | 3 | 0.04% |
| record | place.location.region | 3 | 0.04% |
| record | severity: Priority level of the incident | 3 | 0.04% |
| record | affected_service: Service or component impacted by the incident | 3 | 0.04% |
| record | ticket_environment: Environment (e.g., production, staging, dev) affected | 3 | 0.04% |
| record | impact_users: Number of users directly affected by the issue | 3 | 0.04% |
| record | positive_control | 3 | 0.04% |
| record | sequence_similarities | 3 | 0.04% |
| record | abpromise_guarantee | 3 | 0.04% |
| record | asteroid.alternate_name | 3 | 0.04% |
| record | asteroid.named_after.name | 3 | 0.04% |
| record | apparent_magnitude.min | 3 | 0.04% |
| record | characteristics.variable_type_code | 3 | 0.04% |
| record | confidence_score | 3 | 0.04% |
| record | gene_count | 3 | 0.04% |
| record | release_year: Year of product release extracted from release date | 3 | 0.04% |
| record | model_name: Name of the product as marketed to consumers | 3 | 0.04% |
| record | device_model: Model of the monitoring device used | 3 | 0.04% |
| record | device_type: Type of monitoring device such as wearable, bedside, or oximeter | 3 | 0.04% |
| record | return_type | 3 | 0.04% |
| record | asteroid.discovery.discovery_rank | 3 | 0.04% |
| record | asteroid.description.discovery_details.date | 3 | 0.04% |
| record | asteroid.description.discovery_details.location.state | 3 | 0.04% |
| record | asteroid.orbital_characteristics.inclination | 3 | 0.04% |
| record | asteroid.external_links.0.title | 3 | 0.04% |
| record | asteroid.wiki_sky_links | 3 | 0.04% |
| record | crime.weapon.model | 3 | 0.04% |
| record | victim.children | 3 | 0.04% |
| record | victim.age | 3 | 0.04% |
| record | crime.location.sector | 3 | 0.04% |
| record | crime.weapon.type | 3 | 0.04% |
| record | crime.location.address | 3 | 0.04% |
| record | lastcommitdate | 3 | 0.04% |
| record | usage_frequency | 3 | 0.04% |
| record | issue: Nature of the technical problem reported | 3 | 0.04% |
| record | submitted: Date when the ticket was submitted | 3 | 0.04% |
| record | affected_system: System or application where the issue occurs | 3 | 0.04% |
| record | description: Detailed account of the problem and symptoms | 3 | 0.04% |
| record | major | 3 | 0.04% |
| record | gpa | 3 | 0.04% |
| record | winner: Name of the player who won the final | 3 | 0.04% |
| record | score: Match score in format 6-3, 4-6, 6-4 | 3 | 0.04% |
| record | id: Unique identifier for the briefing session | 3 | 0.04% |
| record | investor_name: Name of the investor attending | 3 | 0.04% |
| record | outreach_code | 3 | 0.04% |
| record | holder: Name of the policyholder involved in renewal | 3 | 0.04% |
| record | coverage_limit: Maximum coverage amount specified in the policy | 3 | 0.04% |
| record | officeactionnumber | 3 | 0.04% |
| record | trademark_name | 3 | 0.04% |
| record | course_name | 3 | 0.04% |
| record | credits.graphic_design.1.area | 3 | 0.04% |
| record | credits.contributors.1.person | 3 | 0.04% |
| record | targetagegroup: Age range of intended audience | 3 | 0.04% |
| record | funding_source: Name of the grant or program supporting the research | 3 | 0.04% |
| record | peerreviewstatus: Description of peer review process or outcome | 3 | 0.04% |
| record | valuation_multiple | 3 | 0.04% |
| record | dish_description | 3 | 0.04% |
| record | lifecycle_status | 3 | 0.04% |
| record | affected_system: Software or hardware system experiencing the problem | 3 | 0.04% |
| record | resolution_time: Estimated or actual time to resolve the issue | 3 | 0.04% |
| record | lead_investor: List of lead investors or funds involved | 3 | 0.04% |
| record | products: List of items included in the campaign bundle | 3 | 0.04% |
| record | drug_name: Name of the medication involved in the reaction | 3 | 0.04% |
| record | patient_address: Full residential address of the patient | 3 | 0.04% |
| record | drug_dosage: Dosage and frequency of the drug exposure | 3 | 0.04% |
| record | statement | 3 | 0.04% |
| record | flight_no: Flight number as assigned by the airline | 3 | 0.04% |
| record | destination: Airport code where flight ends | 3 | 0.04% |
| record | suggestions | 3 | 0.04% |
| record | drink | 3 | 0.04% |
| record | canton_name | 3 | 0.04% |
| record | key_achievement | 3 | 0.04% |
| record | attorney | 3 | 0.04% |
| record | claim_id: Unique identifier for the claim record | 3 | 0.04% |
| record | patient_name: Full name of the patient receiving treatment | 3 | 0.04% |
| record | dateofservice: Date when medical service was rendered | 3 | 0.04% |
| record | treatment_type: Type of medical treatment provided | 3 | 0.04% |
| record | provider_name: Name of the healthcare provider | 3 | 0.04% |
| record | status: Current stock availability status | 3 | 0.04% |
| record | accuracy | 3 | 0.04% |
| record | units_sold | 3 | 0.04% |
| record | article_metadata.publish_date | 3 | 0.04% |
| record | bedrooms: Number of bedrooms in the property | 3 | 0.04% |
| record | benefit_type | 3 | 0.04% |
| record | status: Current state of the transaction processing | 3 | 0.04% |
| record | transaction_type: Type of transaction such as shipment, purchase, or invoice | 3 | 0.04% |
| record | name: Full name of the event as listed in the Name column | 3 | 0.04% |
| record | gust_speed | 3 | 0.04% |
| record | target_market: Intended user segment or environment | 3 | 0.04% |
| record | company: Manufacturer or developer of the product | 3 | 0.04% |
| record | price: Product cost in US dollars | 3 | 0.04% |
| record | operating_systems | 3 | 0.04% |
| record | supported_algorithms | 3 | 0.04% |
| record | municipality.province | 3 | 0.04% |
| record | municipality.merger_municipalities | 3 | 0.04% |
| record | medals.silver | 3 | 0.04% |
| record | medals.bronze | 3 | 0.04% |
| record | vehicle_model | 3 | 0.04% |
| record | incident | 3 | 0.04% |
| record | edit_information.timestamp | 3 | 0.04% |
| record | sentiment.preference | 3 | 0.04% |
| record | price_change | 3 | 0.04% |
| record | hours_of_operation.5.day | 3 | 0.04% |
| record | hours_of_operation.5.close_time | 3 | 0.04% |
| record | name: Full name of the new hire | 3 | 0.04% |
| record | department: Department assigned to the employee | 3 | 0.04% |
| record | access_method | 3 | 0.04% |
| record | affected_site | 3 | 0.04% |
| record | problem_start_time | 3 | 0.04% |
| record | discovery.age.approximateAgeMillionYears | 3 | 0.04% |
| record | publication.author.affiliation.location | 3 | 0.04% |
| record | publication.otherTaxaDescribed | 3 | 0.04% |
| record | person.political_party | 3 | 0.04% |
| record | person.us_senator.years | 3 | 0.04% |
| record | person.ideology | 3 | 0.04% |
| record | person.religion | 3 | 0.04% |
| record | serial: Unique serial number identifying the device | 3 | 0.04% |
| record | regulatory_notes | 3 | 0.04% |
| record | antitrust_review | 3 | 0.04% |
| record | treatment_methods.selection_criteria | 3 | 0.04% |
| record | therapy_basis | 3 | 0.04% |
| record | transaction_type: Type of transaction as recorded in the system | 3 | 0.04% |
| record | ticket_count | 3 | 0.04% |
| record | stage | 3 | 0.04% |
| record | bracket_path | 3 | 0.04% |
| record | asteroid.dedication.person.name | 3 | 0.04% |
| record | patient_mrn | 3 | 0.04% |
| record | damage_amount: Total monetary damage assessed in dollars | 3 | 0.04% |
| record | vaccines | 3 | 0.04% |
| record | basin_name | 3 | 0.04% |
| record | ship_class | 3 | 0.04% |
| record | ship_type | 3 | 0.04% |
| record | call_to_action | 3 | 0.04% |
| record | ties | 3 | 0.04% |
| record | season | 3 | 0.04% |
| record | israel_airports.largest_airport.flight_duration | 3 | 0.04% |
| record | israel_airports.largest_airport.alternate_name | 3 | 0.04% |
| record | passenger_information.adults.prompt | 3 | 0.04% |
| record | travel_class_options | 3 | 0.04% |
| record | passenger_information.children.prompt | 3 | 0.04% |
| record | lease_agreements.number | 3 | 0.04% |
| record | allergies | 3 | 0.04% |
| record | model: Manufacturer model name of the medical device | 3 | 0.04% |
| record | title_en | 3 | 0.04% |
| record | financial_figure | 3 | 0.04% |
| record | upvotes: Number of upvotes the comment received | 3 | 0.04% |
| record | downvotes: Number of downvotes the comment received | 3 | 0.04% |
| record | related_links | 3 | 0.04% |
| record | buyer: Name of the customer who made the purchase | 3 | 0.04% |
| record | rating: Star rating from 1 to 5 given by the buyer | 3 | 0.04% |
| record | asteroid.description | 3 | 0.04% |
| record | bp_systolic: Systolic blood pressure in mmHg | 3 | 0.04% |
| record | visit_date: Date of the medical encounter | 3 | 0.04% |
| record | symptoms: Patient-reported symptoms or complaints | 3 | 0.04% |
| record | target: Email address of the intended victim | 3 | 0.04% |
| record | malicious_content: Description of the phishing element like link or attachment | 3 | 0.04% |
| record | booking_options.tripadvisor_role | 3 | 0.04% |
| record | patient_location | 3 | 0.04% |
| record | municipality.former_province | 3 | 0.04% |
| record | service.location.office | 3 | 0.04% |
| record | service.location.city | 3 | 0.04% |
| record | deal_type | 3 | 0.04% |
| record | issue_type: Category of the reported issue | 3 | 0.04% |
| record | company_id: Corporate ID if the account is business-linked | 3 | 0.04% |
| record | jersey_number: Player's assigned jersey number on field | 3 | 0.04% |
| record | away_team: Team playing away | 3 | 0.04% |
| record | salary_range: Range of expected compensation in dollar format | 3 | 0.04% |
| record | claim_id: Unique identifier for the submitted claim | 3 | 0.04% |
| record | outcomes.impact | 3 | 0.04% |
| record | asteroid.dedication.person.nationality | 3 | 0.04% |
| record | rma_id | 3 | 0.04% |
| record | person.death_location.city | 3 | 0.04% |
| record | sustainability.certification | 3 | 0.04% |
| record | email_id: Unique identifier for the customer's support email | 3 | 0.04% |
| record | salary_adjustment | 3 | 0.04% |
| record | voting_body | 3 | 0.04% |
| record | currency: Currency type used in the transaction | 3 | 0.04% |
| record | hash: Cryptographic hash of the transaction data | 3 | 0.04% |
| record | power_consumption | 3 | 0.04% |
| record | charging_time | 3 | 0.04% |
| record | storage_capacity | 3 | 0.04% |
| record | use_case | 3 | 0.04% |
| record | customer_name: Name of the customer filing the issue | 3 | 0.04% |
| record | description: Detailed explanation of the issue and customer's concern | 3 | 0.04% |
| record | office: Regional office location where employee is based | 3 | 0.04% |
| record | disadvantages | 3 | 0.04% |
| record | price_per_person | 3 | 0.04% |
| record | food_description | 3 | 0.04% |
| record | visit_count | 3 | 0.04% |
| record | average_rating | 3 | 0.04% |
| record | ownership_structure | 3 | 0.04% |
| record | acquisition_leader | 3 | 0.04% |
| record | person.death_date.year | 3 | 0.04% |
| record | person.birth_date.location | 3 | 0.04% |
| record | person.awards | 3 | 0.04% |
| record | feature_status | 3 | 0.04% |
| record | treatment_overview.principle | 3 | 0.04% |
| record | treatment_overview.experience | 3 | 0.04% |
| record | treatment_examples.items.0.conditions | 3 | 0.04% |
| record | treatment_examples.items.0.category | 3 | 0.04% |
| record | additional_treatments | 3 | 0.04% |
| record | baths | 3 | 0.04% |
| record | block.district | 3 | 0.04% |
| record | person.death_location.region | 3 | 0.04% |
| record | product.vendor | 3 | 0.04% |
| record | application_type: Type of patent application filed | 3 | 0.04% |
| record | patent_id: Unique identifier for the patent filing | 3 | 0.04% |
| record | guideline_code | 3 | 0.04% |
| record | implementation_notes | 3 | 0.04% |
| record | weather_conditions | 3 | 0.04% |
| record | kpivendorcompliance: Percentage of vendor contract adherence met | 3 | 0.04% |
| record | manager_name: Name of the employee's direct manager | 3 | 0.04% |
| record | performance_rating: Numerical score on a 5-point scale reflecting performance | 3 | 0.04% |
| record | required_degree | 3 | 0.04% |
| record | date: Date the donation was made | 3 | 0.04% |
| record | distribution_method | 3 | 0.04% |
| record | description.styles | 3 | 0.04% |
| record | conservation_status.threats | 3 | 0.04% |
| record | habitat.environmental_conditions | 3 | 0.04% |
| record | location.altitude.minimum | 3 | 0.04% |
| record | habitat.terrain | 3 | 0.04% |
| record | conservation_status.source | 3 | 0.04% |
| record | patient_id: Unique identifier for the patient in the medical system | 3 | 0.04% |
| record | dateoftest: Date when the laboratory test was conducted | 3 | 0.04% |
| record | subcommittee | 3 | 0.04% |
| record | birth.date_gregorian | 3 | 0.04% |
| record | death.date_gregorian | 3 | 0.04% |
| record | father.name | 3 | 0.04% |
| record | agent_id | 3 | 0.04% |
| record | riskfactorcode | 3 | 0.04% |
| record | festival_id: Unique identifier for the festival event | 3 | 0.04% |
| record | bloodpressuresystolic | 3 | 0.04% |
| record | bloodpressurediastolic | 3 | 0.04% |
| record | company.locations | 3 | 0.04% |
| record | tags: List of hashtags used in the video post | 3 | 0.04% |
| record | comparison.similar_look | 3 | 0.04% |
| record | product.registration_count.time_period | 3 | 0.04% |
| record | product.price.1.currency | 3 | 0.04% |
| record | product.launch_date | 3 | 0.04% |
| record | date: Date when the tournament took place | 3 | 0.04% |
| record | environmental_rating | 3 | 0.04% |
| record | process_details.mediator_role.authority_limitations | 3 | 0.04% |
| record | resources.additional_information_source | 3 | 0.04% |
| record | candidate_id: Unique identifier assigned to the candidate | 3 | 0.04% |
| record | region: Geographic area or setting where guideline applies | 3 | 0.04% |
| record | recommendation: Key clinical recommendation or action advised | 3 | 0.04% |
| record | name: Official title of the health care guideline | 3 | 0.04% |
| record | respiratory_rate: Breaths per minute respiratory rate | 3 | 0.04% |
| record | gymnosperm_diversity.rare_species | 3 | 0.04% |
| record | gymnosperm_diversity.endemism_ranking | 3 | 0.04% |
| record | citations.4.author | 3 | 0.04% |
| record | citations.4.year | 3 | 0.04% |
| record | gymnosperm_diversity.species_examples.0.name | 3 | 0.04% |
| record | gymnosperm_diversity.species_examples.1.name | 3 | 0.04% |
| record | chair.state | 3 | 0.04% |
| record | terminology.ideographic_script.english | 3 | 0.04% |
| record | comparison.ideographic.learning_curve | 3 | 0.04% |
| record | comparison.ideographic.cross_lingual_understanding | 3 | 0.04% |
| record | comparison.ideographic.example.meaning | 3 | 0.04% |
| record | comparison.ideographic.example.languages.0.language | 3 | 0.04% |
| record | comparison.ideographic.example.languages.1.language | 3 | 0.04% |
| record | caveat.example.components.0.meaning | 3 | 0.04% |
| record | caveat.example.components.0.related_characters.0.pronunciation | 3 | 0.04% |
| record | caveat.example.components.0.related_characters.0.character | 3 | 0.04% |
| record | caveat.example.components.0.related_characters.1.meaning | 3 | 0.04% |
| record | caveat.example.components.0.position | 3 | 0.04% |
| record | caveat.example.components.1.function | 3 | 0.04% |
| record | caveat.example.components.1.position | 3 | 0.04% |
| record | notable_years | 3 | 0.04% |
| record | historical_events | 3 | 0.04% |
| record | street_address | 3 | 0.04% |
| record | mentions | 3 | 0.04% |
| record | gymnosperm_diversity.species_examples.2.characteristic | 3 | 0.04% |
| record | geography.coordinates.longitude | 3 | 0.04% |
| record | product_name: Name of the specific product launched on Instagram | 3 | 0.04% |
| record | statue.name.french | 3 | 0.04% |
| record | statue.discovery.discoverer.name | 3 | 0.04% |
| record | statue.discovery.location | 3 | 0.04% |
| record | statue.restoration.0.funding.contribution_range.min | 3 | 0.04% |
| record | statue.restoration.0.budget.currency | 3 | 0.04% |
| record | statue.discovery.discoverer.title | 3 | 0.04% |
| record | technologies | 3 | 0.04% |
| record | typical_use: Primary medical indication for the drug | 3 | 0.04% |
| record | status: Current regulatory status of the drug | 3 | 0.04% |
| record | review_score: Numerical score out of 5 based on user feedback | 3 | 0.04% |
| record | guidelines.moa_members.research_framework_request | 3 | 0.04% |
| record | guidelines.rounding_percentages | 3 | 0.04% |
| record | population.year | 3 | 0.04% |
| record | enzymes | 3 | 0.04% |
| record | chemistry | 3 | 0.04% |
| record | reviewer_name: Full name of the reviewer | 3 | 0.04% |
| record | report_id: Unique identifier assigned to the phishing report | 3 | 0.04% |
| record | orbital_characteristics.orbital_period_years | 3 | 0.04% |
| record | rating: Star rating given to the restaurant on a scale of 1 to 5 | 3 | 0.04% |
| record | battery_life: Duration of battery operation in days | 3 | 0.04% |
| record | entity_location | 3 | 0.04% |
| record | asteroid.name_only | 3 | 0.04% |
| record | visit_type: Type of medical visit conducted | 3 | 0.04% |
| record | tv_program.year | 3 | 0.04% |
| record | tv_program.setting.time_period | 3 | 0.04% |
| record | tv_program.previous_program | 3 | 0.04% |
| record | address: Street address including city and state | 3 | 0.04% |
| record | account_number: Unique account identifier shared across entries | 3 | 0.04% |
| record | theological_concepts | 3 | 0.04% |
| record | book | 3 | 0.04% |
| record | genus.species.0.name | 3 | 0.04% |
| record | genus.species.0.year | 3 | 0.04% |
| record | hotel_details.location | 3 | 0.04% |
| record | status: Current processing or acceptance status | 3 | 0.04% |
| record | biological tissue | 3 | 0.04% |
| record | revenue: Total gate revenue in millions | 3 | 0.04% |
| record | rma_number: Assigned number for the RMA case | 3 | 0.04% |
| record | cost: Repair cost, if any, as stated in the request | 3 | 0.04% |
| record | student | 3 | 0.04% |
| record | benefits: List of included employee benefits | 3 | 0.04% |
| record | experience_required: Years of experience required for the role | 3 | 0.04% |
| record | release_details.album.type | 3 | 0.04% |
| record | promotion.social_media | 3 | 0.04% |
| record | release_details.artist.name | 3 | 0.04% |
| record | release_details.next_release.release_date | 3 | 0.04% |
| record | nouns_key_entities | 3 | 0.04% |
| record | adjectival_complement | 3 | 0.04% |
| record | magazine.issue | 3 | 0.04% |
| record | magazine.article.content_percentage | 3 | 0.04% |
| record | magazine.article.page_breakdown.0.advertisers | 3 | 0.04% |
| record | magazine.article.page_breakdown.1.advertisers | 3 | 0.04% |
| record | magazine.article.page_breakdown.2.advertisers | 3 | 0.04% |
| record | magazine.article.page_breakdown.3.advertising_percentage | 3 | 0.04% |
| record | magazine.article.page_breakdown.3.advertisers | 3 | 0.04% |
| record | magazine.article.page_breakdown.4.advertising_percentage | 3 | 0.04% |
| record | magazine.article.page_breakdown.4.advertisers | 3 | 0.04% |
| record | magazine.article.page_breakdown.5.page_number | 3 | 0.04% |
| record | magazine.article.page_breakdown.8.advertisers | 3 | 0.04% |
| record | magazine.article.page_breakdown.9.advertising_percentage | 3 | 0.04% |
| record | magazine.article.page_breakdown.10.page_number | 3 | 0.04% |
| record | magazine.article.page_breakdown.10.advertising_percentage | 3 | 0.04% |
| record | magazine.article.page_breakdown.11.advertising_percentage | 3 | 0.04% |
| record | magazine.article.page_breakdown.12.advertisers | 3 | 0.04% |
| record | magazine.article.page_breakdown.13.page_number | 3 | 0.04% |
| record | magazine.article.page_breakdown.15.page_number | 3 | 0.04% |
| record | magazine.article.page_breakdown.16.advertisers | 3 | 0.04% |
| record | magazine.mentioned_article.title | 3 | 0.04% |
| record | magazine.mentioned_article.year | 3 | 0.04% |
| record | series_record | 3 | 0.04% |
| record | album.external_links.1.url | 3 | 0.04% |
| record | album.tracklist.cd.1.lyrics | 3 | 0.04% |
| record | album.tracklist.cd.8.tie_in.0.title | 3 | 0.04% |
| record | album.tracklist.cd.8.composer | 3 | 0.04% |
| record | album.tracklist.cd.3.title | 3 | 0.04% |
| record | album.tracklist.cd.2.composer | 3 | 0.04% |
| record | album.tracklist.cd.4.composer | 3 | 0.04% |
| record | album.external_links.2.url | 3 | 0.04% |
| record | public_information.download_share | 3 | 0.04% |
| record | public_information.brochure_name | 3 | 0.04% |
| record | distribution_information.distribution_locations | 3 | 0.04% |
| record | distribution_information.trail_details.trail_endpoints | 3 | 0.04% |
| record | claim_amount: Monetary claim amount if applicable | 3 | 0.04% |
| record | block.state | 3 | 0.04% |
| record | course_id: Unique identifier for the course as given in the XML | 3 | 0.04% |
| record | company: Company owning or offering the product | 3 | 0.04% |
| record | employee_count | 3 | 0.04% |
| record | resolution_action: Type of action taken to resolve the issue | 3 | 0.04% |
| record | technology_name: Name of the algorithm, platform, or device | 3 | 0.04% |
| record | developers: List of institutions or individuals leading the project | 3 | 0.04% |
| record | prescription_id: Unique identifier for the prescription entry | 3 | 0.04% |
| record | address: Patient's residential address including street, unit, and city | 3 | 0.04% |
| record | wins | 3 | 0.04% |
| record | games_behind | 3 | 0.04% |
| record | ticket_id: Unique identifier for each incident in the system | 3 | 0.04% |
| record | status: Current state of the incident | 3 | 0.04% |
| record | photo_quality | 3 | 0.04% |
| record | term | 3 | 0.04% |
| record | date_administered | 3 | 0.04% |
| record | id: Unique identifier for the line item entry | 3 | 0.04% |
| record | review.persons.0.description | 3 | 0.04% |
| record | review.performance.performers.0.role | 3 | 0.04% |
| record | physical_characteristics.observations.0.telescope.location | 3 | 0.04% |
| record | discovery_details.observatory.location.country | 3 | 0.04% |
| record | hotel.distance_from_city_center.value | 3 | 0.04% |
| record | hotel.number_of_rooms | 3 | 0.04% |
| record | contract_details.financial_value.amount | 3 | 0.04% |
| record | contract_details.financial_value.scale | 3 | 0.04% |
| record | support_tickets_open | 3 | 0.04% |
| record | ui_bug_status | 3 | 0.04% |
| record | product_name: name of the physical item such as earbuds or lantern | 3 | 0.04% |
| record | report_metadata.publication_date | 3 | 0.04% |
| record | report_metadata.source_agency | 3 | 0.04% |
| record | company.financial_results.currency | 3 | 0.04% |
| record | users | 3 | 0.04% |
| record | location: Office location where the visit occurred | 3 | 0.04% |
| record | location.distance_from_city_center.value | 3 | 0.04% |
| record | hotel_details.reception_open_until | 3 | 0.04% |
| record | hotel_details.number_of_rooms | 3 | 0.04% |
| record | child_policy.extra_bed_policy | 3 | 0.04% |
| record | check_in_process | 3 | 0.04% |
| record | owner_contact | 3 | 0.04% |
| record | clinical_phase | 3 | 0.04% |
| record | acquirer_ceo: CEO of the acquiring company | 3 | 0.04% |
| record | sentiment.opinion | 3 | 0.04% |
| record | company: Name of the company raising capital | 3 | 0.04% |
| record | summary: Brief description of the root cause or issue | 3 | 0.04% |
| record | address: Full address of the customer location | 3 | 0.04% |
| record | investor: Name of the venture capital firm that invested | 3 | 0.04% |
| record | category: Broad product category such as home, tech, health, outdoor | 3 | 0.04% |
| record | sku: Unique stock keeping unit identifier for the product | 3 | 0.04% |
| record | network_interaction | 3 | 0.04% |
| record | medical_code | 3 | 0.04% |
| record | key_enzymes: List of enzymes critical to pathway activity | 3 | 0.04% |
| record | article.publication_details.source | 3 | 0.04% |
| record | hotel.nearby_locations | 3 | 0.04% |
| record | date_sold | 3 | 0.04% |
| record | navigation | 3 | 0.04% |
| record | name: Full name of the new employee | 3 | 0.04% |
| record | prize_money | 3 | 0.04% |
| record | case | 3 | 0.04% |
| record | supervisor: Name of the employee's supervisor | 3 | 0.04% |
| record | event.location.address | 3 | 0.04% |
| record | organization_info.electricity_source | 3 | 0.04% |
| record | donations.type | 3 | 0.04% |
| record | event.location.name | 3 | 0.04% |
| record | event.time.photo_op_end | 3 | 0.04% |
| record | primate.morphological_analysis.1.finding | 3 | 0.04% |
| record | primate.relationships | 3 | 0.04% |
| record | long_term_debt | 3 | 0.04% |
| record | clinical_status | 3 | 0.04% |
| record | winner_nationality | 3 | 0.04% |
| record | winner_name | 3 | 0.04% |
| record | project_code | 3 | 0.04% |
| record | vehicle.protection_level | 3 | 0.04% |
| record | vehicle.manufacturer.name | 3 | 0.04% |
| record | vehicle.manufacturer.location | 3 | 0.04% |
| record | vehicle.specifications.weight | 3 | 0.04% |
| record | vehicle.specifications.armor | 3 | 0.04% |
| record | vehicle.combat_usage.0.conflict | 3 | 0.04% |
| record | vehicle.operators.0.acquisition.date | 3 | 0.04% |
| record | difficulty_level | 3 | 0.04% |
| record | performancerating | 3 | 0.04% |
| record | sustainability.sourced | 3 | 0.04% |
| record | revenue: Annual revenue in millions of dollars | 3 | 0.04% |
| record | date: Date of the competitor record entry | 3 | 0.04% |
| record | id: Unique identifier assigned to each competitor | 3 | 0.04% |
| record | ticket_id: Unique identifier for each outage report | 3 | 0.04% |
| record | hire_method | 3 | 0.04% |
| record | manager_name | 3 | 0.04% |
| record | product_id: Unique identifier for the product instance | 3 | 0.04% |
| record | price: Product price in USD format | 3 | 0.04% |
| record | trade_show | 3 | 0.04% |
| record | opening_year | 3 | 0.04% |
| record | tehsil | 3 | 0.04% |
| record | issue_year: Year the guideline was published or released | 3 | 0.04% |
| record | user: Name of the customer who provided the review | 3 | 0.04% |
| record | review: Text content of the customer's feedback on the product | 3 | 0.04% |
| record | purchase_date: Date when the product was purchased | 3 | 0.04% |
| record | project.operational_model.stakeholders.1.description | 3 | 0.04% |
| record | project.abbreviation | 3 | 0.04% |
| record | project.name | 3 | 0.04% |
| record | music.organization | 3 | 0.04% |
| record | recommendation: Specific treatment or intervention advised | 3 | 0.04% |
| record | lease_id | 3 | 0.04% |
| record | key_player_assists | 3 | 0.04% |
| record | field_goal_percentage | 3 | 0.04% |
| record | bench_score | 3 | 0.04% |
| record | rating: Customer rating out of 5 stars | 3 | 0.04% |
| record | issue: Description of the technical problem reported | 3 | 0.04% |
| record | report_id: Unique identifier for the phishing report | 3 | 0.04% |
| record | sender_email: Email address of the phishing sender | 3 | 0.04% |
| record | pain_level | 3 | 0.04% |
| record | project_description | 3 | 0.04% |
| record | disputingpartyb | 3 | 0.04% |
| record | radiology_code | 3 | 0.04% |
| record | lot_size: Size of the lot in acres | 3 | 0.04% |
| record | location.proximity | 3 | 0.04% |
| record | location: City and country where the startup is headquartered | 3 | 0.04% |
| record | co_investors: List of additional investors participating in the round | 3 | 0.04% |
| record | investor_count: Number of investors involved in the round | 3 | 0.04% |
| record | patient_name: Full name of the patient | 3 | 0.04% |
| record | postdischargemonitoring: Instructions for post-discharge care | 3 | 0.04% |
| record | fieldofstudy: Academic discipline or program the recipient is in | 3 | 0.04% |
| record | document_details.document_type | 3 | 0.04% |
| record | document_details.responsibility | 3 | 0.04% |
| record | document_details.authors.1.name | 3 | 0.04% |
| record | path_design.concrete_tile_making.mold_description | 3 | 0.04% |
| record | path_design.skill_level | 3 | 0.04% |
| record | path | 3 | 0.04% |
| record | rooms | 3 | 0.04% |
| record | growth_note | 3 | 0.04% |
| record | report_code | 3 | 0.04% |
| record | return_date: Date device was returned to repair center | 3 | 0.04% |
| record | submitter: Name of the customer who submitted the ticket | 3 | 0.04% |
| record | patient_care.station_assignment | 3 | 0.04% |
| record | store_reference | 3 | 0.04% |
| record | audiofidelityscore | 3 | 0.04% |
| record | property_id: Unique identifier for the property | 3 | 0.04% |
| record | block | 3 | 0.04% |
| record | discovery_details.observatory.name | 3 | 0.04% |
| record | discovery_details.observatory.location.state | 3 | 0.04% |
| record | contraindications | 3 | 0.04% |
| record | drug_interactions | 3 | 0.04% |
| record | visitor_score | 3 | 0.04% |
| record | warranty_period: Duration of warranty in years or terms | 3 | 0.04% |
| record | donation_purpose | 3 | 0.04% |
| record | pharmacy_name | 3 | 0.04% |
| record | product_id: Unique identifier assigned by the manufacturer | 3 | 0.04% |
| record | value: Total acquisition value in millions of dollars | 3 | 0.04% |
| record | tech_stack | 3 | 0.04% |
| record | address: Full address of the policyholder's residence | 3 | 0.04% |
| record | user_id: Unique identifier for the user performing the transaction | 3 | 0.04% |
| record | record_id: Unique identifier for the vaccination record | 3 | 0.04% |
| record | status: Current status of the lease | 3 | 0.04% |
| record | pet_policy: Pet policy including restrictions or fees | 3 | 0.04% |
| record | last_updated: Date the product record was last modified | 3 | 0.04% |
| record | description: Brief product feature and use description | 3 | 0.04% |
| record | tenant_type: Industry or sector of the leasing business | 3 | 0.04% |
| record | genus.language | 3 | 0.04% |
| record | genus.description | 3 | 0.04% |
| record | genus.period | 3 | 0.04% |
| record | genus.location | 3 | 0.04% |
| record | genus.naming.profession | 3 | 0.04% |
| record | premium_adjustment | 3 | 0.04% |
| record | details.calorie_burn.kettlebell_time | 3 | 0.04% |
| record | details.calorie_burn.running_calories | 3 | 0.04% |
| record | regulatory_status.recognition_type | 3 | 0.04% |
| record | prescription | 3 | 0.04% |
| record | date: Month and year of funding occurrence | 3 | 0.04% |
| record | career_summary.career_length | 3 | 0.04% |
| record | future_plans | 3 | 0.04% |
| record | personal_life.reason_for_leaving | 3 | 0.04% |
| record | legacy | 3 | 0.04% |
| record | activation_type | 3 | 0.04% |
| record | diagnosis_code: Standard code indicating the patient's condition | 3 | 0.04% |
| record | host_location | 3 | 0.04% |
| record | date_sent | 3 | 0.04% |
| record | content_length | 3 | 0.04% |
| record | downvotes | 3 | 0.04% |
| record | storm_characteristics_at_stage | 3 | 0.04% |
| record | neighborhood | 3 | 0.04% |
| record | clinical_outcomes.glycemic_control.marker | 3 | 0.04% |
| record | clinical_outcomes.hypoglycemia_risk | 3 | 0.04% |
| record | clinical_outcomes.weight_impact | 3 | 0.04% |
| record | renewal_amount: Monetary value of the renewal premium | 3 | 0.04% |
| record | address: Full residential address of the policyholder | 3 | 0.04% |
| record | test_code: Code assigned to the specific test | 3 | 0.04% |
| record | normal_range: Reference range for the measured value | 3 | 0.04% |
| record | collection_site: Name of the clinic or lab where test was collected | 3 | 0.04% |
| record | templates_not_to_substitute | 3 | 0.04% |
| record | service.opening_hours.0.time | 3 | 0.04% |
| record | service.problems_addressed | 3 | 0.04% |
| record | service.responsible_person.experience | 3 | 0.04% |
| record | organization.location_founded.city | 3 | 0.04% |
| record | organization.year_founded | 3 | 0.04% |
| record | match_score | 3 | 0.04% |
| record | location: Name of the medical facility conducting the imaging | 3 | 0.04% |
| record | payee: Name of the entity receiving payment | 3 | 0.04% |
| record | country: Nationality of the athlete who won the medal | 3 | 0.04% |
| record | publication: Name of the media outlet publishing the review | 3 | 0.04% |
| record | reasons | 3 | 0.04% |
| record | settlement_time | 3 | 0.04% |
| record | citations.3.description | 3 | 0.04% |
| record | researcher | 3 | 0.04% |
| record | bp | 3 | 0.04% |
| record | delay_date | 3 | 0.04% |
| record | transaction_date | 3 | 0.04% |
| record | physical_characteristics.observations.0.type | 3 | 0.04% |
| record | campaign_committee: Name of the political committee receiving the contribution | 3 | 0.04% |
| record | return_method | 3 | 0.04% |
| record | contact_information.website | 3 | 0.04% |
| record | style | 3 | 0.04% |
| record | business_type | 3 | 0.04% |
| record | discount.percentage | 3 | 0.04% |
| record | booking.cancellation_policy | 3 | 0.04% |
| record | terms_and_conditions.combination_restriction | 3 | 0.04% |
| record | terms_and_conditions.disclaimer | 3 | 0.04% |
| record | rates.rate_type | 3 | 0.04% |
| record | loyalty_program.name | 3 | 0.04% |
| record | loyalty_program.membership_url | 3 | 0.04% |
| record | booking_channels | 3 | 0.04% |
| record | job_id: Unique identifier assigned to the job posting | 3 | 0.04% |
| record | company: Name of the company issuing the job posting | 3 | 0.04% |
| record | article_code | 3 | 0.04% |
| record | nummod | 3 | 0.04% |
| record | cc | 3 | 0.04% |
| record | poss | 3 | 0.04% |
| record | intj | 3 | 0.04% |
| record | affected_period | 3 | 0.04% |
| record | traveler_name | 3 | 0.04% |
| record | venue: Stadium name where the match was held | 3 | 0.04% |
| record | scorers: List of players who scored goals in the match | 3 | 0.04% |
| record | location.land_ownership | 3 | 0.04% |
| record | record_id | 3 | 0.04% |
| record | primate.fossil_origin.epoch | 3 | 0.04% |
| record | tenant_id | 3 | 0.04% |
| record | genus.etymology.part2.name | 3 | 0.04% |
| record | genus.etymology.part1.related_entity | 3 | 0.04% |
| record | genus.etymology.part1.origin | 3 | 0.04% |
| record | location: City and state where the company was founded | 3 | 0.04% |
| record | status: Current payment or compliance status | 3 | 0.04% |
| record | service_description: Brief description of goods or services provided | 3 | 0.04% |
| record | sexual_act.insertion.depth_perception | 3 | 0.04% |
| record | sexual_act.thrusting.vaginal_wall_movement | 3 | 0.04% |
| record | sexual_act.thrusting.cervix_transformation.purpose_of_trunk_position | 3 | 0.04% |
| record | sexual_act.post_ejaculation.semen_pool_retention | 3 | 0.04% |
| record | sexual_act.post_ejaculation.loss_of_camera_support | 3 | 0.04% |
| record | sexual_act.sperm_migration.semen_coagulation | 3 | 0.04% |
| record | sexual_act.sperm_migration.sperm_movement_within_fingers | 3 | 0.04% |
| record | sexual_act.cervix_retraction.fate_of_remaining_sperm | 3 | 0.04% |
| record | sexual_act.cervix_retraction.semen_liquefaction | 3 | 0.04% |
| record | sexual_act.cervix_retraction.muscle_contractions | 3 | 0.04% |
| record | sexual_act.cervix_retraction.reflux_formation | 3 | 0.04% |
| record | sexual_act.cervix_retraction.reflux_timing | 3 | 0.04% |
| record | sexual_act.cervix_retraction.reflux_triggers | 3 | 0.04% |
| record | hotel_experience.location.nearby_landmark | 3 | 0.04% |
| record | hotel_experience.entertainment.tv.content | 3 | 0.04% |
| record | hotel_experience.suites.0.workspace | 3 | 0.04% |
| record | hotel_experience.suites.0.amenities | 3 | 0.04% |
| record | tools: List of software or platforms used in the role | 3 | 0.04% |
| record | salary_range: Annual salary range in dollars | 3 | 0.04% |
| record | event_rescheduled_to | 3 | 0.04% |
| record | rescheduled_event_time | 3 | 0.04% |
| record | sales_volume: Total units sold | 3 | 0.04% |
| record | content_analysis.themes | 3 | 0.04% |
| record | location: Physical address where course is held | 3 | 0.04% |
| record | status: Current enrollment or operational status | 3 | 0.04% |
| record | vital_signs | 3 | 0.04% |
| record | renewal_amount | 3 | 0.04% |
| record | DT | 3 | 0.04% |
| record | Andrew Bulkeley <> works for | 3 | 0.04% |
| record | Dawn Engle <> founder of | 3 | 0.04% |
| record | Denver Mayor Michael Hancock <> commented on | 3 | 0.04% |
| record | John McAfee <> questioned about | 3 | 0.04% |
| record | adverse_event | 3 | 0.04% |
| record | support_code | 3 | 0.04% |
| record | payment_method: Method used to process payment for the claim | 3 | 0.04% |
| record | units: Unit of measurement for the test result | 3 | 0.04% |
| record | article.references.0.id | 3 | 0.04% |
| record | article.last_reviewed | 3 | 0.04% |
| record | article.version | 3 | 0.04% |
| record | outbreak_event.investigation.methods | 3 | 0.04% |
| record | outbreak_event.temporal_data.start_period | 3 | 0.04% |
| record | outbreak_event.investigation.findings.agent_details.classification | 3 | 0.04% |
| record | contact.fax | 3 | 0.04% |
| record | office_hours.2.day | 3 | 0.04% |
| record | office_hours.3.day | 3 | 0.04% |
| record | name: Full product name as listed in catalog | 3 | 0.04% |
| record | stock: Current quantity available in warehouse or distribution center | 3 | 0.04% |
| record | owner: Name of the product owner or manager assigned | 3 | 0.04% |
| record | campaign: Marketing campaign currently promoting the product | 3 | 0.04% |
| record | hiring_manager | 3 | 0.04% |
| record | required_qualifications | 3 | 0.04% |
| record | email: Email address of the passenger | 3 | 0.04% |
| record | departure_date: Date flight departs from origin | 3 | 0.04% |
| record | holiday_look.coat.brand | 3 | 0.04% |
| record | booth_size | 3 | 0.04% |
| record | title: Title of the abstract submitted for presentation | 3 | 0.04% |
| record | status: Current status of the abstract review | 3 | 0.04% |
| record | boulder.start.hand_holds.1.description | 3 | 0.04% |
| record | boulder.location.reference_point | 3 | 0.04% |
| record | research.experiment_details.quantum_state | 3 | 0.04% |
| record | research.objective | 3 | 0.04% |
| record | research.experiment_details.temperature_details | 3 | 0.04% |
| record | hotel.recreation_facilities | 3 | 0.04% |
| record | hotel.description | 3 | 0.04% |
| record | contributor_location | 3 | 0.04% |
| record | star.designation | 3 | 0.04% |
| record | star.spectral_class | 3 | 0.04% |
| record | planetary_system.planets.0.discovery_year | 3 | 0.04% |
| record | planetary_system.planets.0.discovery_team | 3 | 0.04% |
| record | incident.charges.1.description | 3 | 0.04% |
| record | project_id: Unique identifier assigned to the technology initiative | 3 | 0.04% |
| record | funding_source: Organization or fund providing financial support | 3 | 0.04% |
| record | release_target | 3 | 0.04% |
| record | filing_id: Unique identifier assigned to the filing | 3 | 0.04% |
| record | dividend_yield | 3 | 0.04% |
| record | mental_health_conditions | 3 | 0.04% |
| record | medication_name: Name of the prescribed medication | 3 | 0.04% |
| record | bill: Bill number and type identifier | 3 | 0.04% |
| record | amount: Financial amount associated with the bill | 3 | 0.04% |
| record | status: Current status of the bill | 3 | 0.04% |
| record | target_role: Job title or position being targeted | 3 | 0.04% |
| record | investor_participants | 3 | 0.04% |
| record | contact_number | 3 | 0.04% |
| record | active_clients: Number of active clients or installations as of a stated year | 3 | 0.04% |
| record | founder_name: Name of the founder who established the startup | 3 | 0.04% |
| record | stomatopod crustacean | 3 | 0.04% |
| record | purchase_date: Date when the device was purchased | 3 | 0.04% |
| record | comment_count: Number of comments on the video | 3 | 0.04% |
| record | title: Title of the video as displayed in the feed | 3 | 0.04% |
| record | like_count: Number of likes the video has accumulated | 3 | 0.04% |
| record | remarks | 3 | 0.04% |
| record | entities.social_media_platform | 3 | 0.04% |
| record | onset_date: Date when symptoms first began | 3 | 0.04% |
| record | key_player_rebounds | 3 | 0.04% |
| record | key_player_name | 3 | 0.04% |
| record | players | 3 | 0.04% |
| record | name: Full name of the player involved in transfer | 3 | 0.04% |
| record | goals: Goals scored in last season | 3 | 0.04% |
| record | user_rating | 3 | 0.04% |
| record | hashtag: Hashtag used in the campaign | 3 | 0.04% |
| record | product: Name of the product being launched | 3 | 0.04% |
| record | advertisement.celebrity | 3 | 0.04% |
| record | discount_condition | 3 | 0.04% |
| record | prevention_tips | 3 | 0.04% |
| record | item_description | 3 | 0.04% |
| record | negative_note | 3 | 0.04% |
| record | prescription_id: Unique identifier for the prescription record | 3 | 0.04% |
| record | person.experience.0.start_year | 3 | 0.04% |
| record | person.experience.0.description | 3 | 0.04% |
| record | person.experience.0.organization | 3 | 0.04% |
| record | person.experience.1.organization | 3 | 0.04% |
| record | person.publications.0.title_alternative | 3 | 0.04% |
| record | person.parliamentary_role.0.end_year | 3 | 0.04% |
| record | person.parliamentary_role.0.chamber | 3 | 0.04% |
| record | historical_context.time_periods.0.period | 3 | 0.04% |
| record | historical_context.time_periods.0.events.0.event_type | 3 | 0.04% |
| record | access_information.description | 3 | 0.04% |
| record | historical_context.time_periods.1.period | 3 | 0.04% |
| record | historical_context.time_periods.0.events.0.description | 3 | 0.04% |
| record | status: Current shipping status of the consignment | 3 | 0.04% |
| record | map_location | 3 | 0.04% |
| record | dean.year | 3 | 0.04% |
| record | campus | 3 | 0.04% |
| record | preferredtenanttype | 3 | 0.04% |
| record | hotel.location.postal_code | 3 | 0.04% |
| record | hotel.contact.fax | 3 | 0.04% |
| record | hotel.transportation.metro.line | 3 | 0.04% |
| record | hotel.transportation.train_station | 3 | 0.04% |
| record | hotel.transportation.driving_directions | 3 | 0.04% |
| record | hotel.transportation.parking.0.walking_time | 3 | 0.04% |
| record | hotel.transportation.parking.0.cost | 3 | 0.04% |
| record | payment_method: Type of payment used for the transaction | 3 | 0.04% |
| record | described_by.year | 3 | 0.04% |
| record | gren_id | 3 | 0.04% |
| record | description: Text explaining the nature and impact of the violation | 3 | 0.04% |
| record | classification.type | 3 | 0.04% |
| record | description: Brief text describing the service or product delivered | 3 | 0.04% |
| record | budget_used | 3 | 0.04% |
| record | Steinem <> adopted by | 3 | 0.04% |
| record | Steinem <> color | 3 | 0.04% |
| record | Steinem <> favorite food | 3 | 0.04% |
| record | Steinem <> lived for | 3 | 0.04% |
| record | Steinem <> loved the smell of | 3 | 0.04% |
| record | Steinem <> sat between | 3 | 0.04% |
| record | Steinem <> slept with | 3 | 0.04% |
| record | Steinem <> species | 3 | 0.04% |
| record | firstaidmeasures: Immediate actions to take in case of exposure | 3 | 0.04% |
| record | storage_conditions: Environmental and physical conditions required for safe storage | 3 | 0.04% |
| record | force_type | 3 | 0.04% |
| record | asteroid.orbital_characteristics.unit | 3 | 0.04% |
| record | location: Full address including street, city, and state | 3 | 0.04% |
| record | event_status: Current status of the event | 3 | 0.04% |
| record | product_name: Name of the product being promoted | 3 | 0.04% |
| record | target_audience: Demographic or user segment targeted by the campaign | 3 | 0.04% |
| record | filing_location | 3 | 0.04% |
| record | organizations.0.name | 3 | 0.04% |
| record | events.0.location | 3 | 0.04% |
| record | partner_id: Unique identifier of the partner company | 3 | 0.04% |
| record | donation_date | 3 | 0.04% |
| record | person.death_date.day | 3 | 0.04% |
| record | person.death_date.location | 3 | 0.04% |
| record | certification.name | 3 | 0.04% |
| record | certification.exams.1.code | 3 | 0.04% |
| record | personnel.0.organization | 3 | 0.04% |
| record | certification.exams.0.code | 3 | 0.04% |
| record | recipient_name: Name of the recipient organization | 3 | 0.04% |
| record | link | 3 | 0.04% |
| record | release_details.album.availability.1.platform | 3 | 0.04% |
| record | status: Current status of the investment round | 3 | 0.04% |
| record | legal_counsel | 3 | 0.04% |
| record | discovery.name | 3 | 0.04% |
| record | origin: Airport code where flight originates | 3 | 0.04% |
| record | reason: Cause of the flight delay | 3 | 0.04% |
| record | status: Current status of the delay record | 3 | 0.04% |
| record | responsible_engineer | 3 | 0.04% |
| record | product: Product model identifier such as EBC-1234 or WAT-5678 | 3 | 0.04% |
| record | price: Price paid for the product in US dollars | 3 | 0.04% |
| record | item_description.item_id | 3 | 0.04% |
| record | item_description.setup_charge.amount | 3 | 0.04% |
| record | billed_amount: Total amount billed for the service | 3 | 0.04% |
| record | patient_id: Unique identifier for the patient filing the claim | 3 | 0.04% |
| record | registration_requirement | 3 | 0.04% |
| record | diagnosis: Medical condition diagnosed with ICD-10 code | 3 | 0.04% |
| record | location: City where the reviewer is based | 3 | 0.04% |
| record | vehicle.problemComponent | 3 | 0.04% |
| record | possibleCauses | 3 | 0.04% |
| record | name: Full name of the startup company | 3 | 0.04% |
| record | announcement.presented_by | 3 | 0.04% |
| record | announcement.week_ending | 3 | 0.04% |
| record | announcement.divisions.0.star_of_the_week.player_name | 3 | 0.04% |
| record | announcement.divisions.0.second_star.player_name | 3 | 0.04% |
| record | announcement.divisions.0.second_star.position | 3 | 0.04% |
| record | announcement.divisions.0.division_name | 3 | 0.04% |
| record | announcement.divisions.0.honorable_mentions.0.player_name | 3 | 0.04% |
| record | announcement.divisions.0.honorable_mentions.0.team | 3 | 0.04% |
| record | announcement.divisions.0.honorable_mentions.2.team | 3 | 0.04% |
| record | announcement.divisions.0.honorable_mentions.3.player_name | 3 | 0.04% |
| record | announcement.divisions.0.honorable_mentions.3.position | 3 | 0.04% |
| record | announcement.divisions.0.honorable_mentions.4.position | 3 | 0.04% |
| record | announcement.divisions.0.honorable_mentions.5.position | 3 | 0.04% |
| record | announcement.divisions.0.honorable_mentions.6.position | 3 | 0.04% |
| record | announcement.divisions.0.honorable_mentions.7.team | 3 | 0.04% |
| record | announcement.divisions.1.star_of_the_week.stats | 3 | 0.04% |
| record | announcement.divisions.1.division_name | 3 | 0.04% |
| record | announcement.divisions.1.honorable_mentions.0.team | 3 | 0.04% |
| record | announcement.divisions.1.honorable_mentions.1.player_name | 3 | 0.04% |
| record | announcement.divisions.1.honorable_mentions.1.team | 3 | 0.04% |
| record | announcement.divisions.2.star_of_the_week.player_name | 3 | 0.04% |
| record | announcement.divisions.2.star_of_the_week.team | 3 | 0.04% |
| record | announcement.divisions.2.second_star.player_name | 3 | 0.04% |
| record | announcement.divisions.2.honorable_mentions.0.player_name | 3 | 0.04% |
| record | announcement.divisions.2.honorable_mentions.2.player_name | 3 | 0.04% |
| record | announcement.divisions.2.honorable_mentions.2.position | 3 | 0.04% |
| record | announcement.divisions.2.honorable_mentions.2.team | 3 | 0.04% |
| record | announcement.divisions.2.honorable_mentions.3.player_name | 3 | 0.04% |
| record | announcement.divisions.3.star_of_the_week.player_name | 3 | 0.04% |
| record | announcement.divisions.3.star_of_the_week.position | 3 | 0.04% |
| record | announcement.divisions.3.second_star.position | 3 | 0.04% |
| record | announcement.divisions.3.honorable_mentions.1.player_name | 3 | 0.04% |
| record | announcement.divisions.4.star_of_the_week.team | 3 | 0.04% |
| record | announcement.divisions.5.honorable_mentions.0.player_name | 3 | 0.04% |
| record | announcement.divisions.5.honorable_mentions.2.player_name | 3 | 0.04% |
| record | annotation_id | 3 | 0.04% |
| record | article_issues | 3 | 0.04% |
| record | observations.number_of_observations | 3 | 0.04% |
| record | discovery.location.observatory | 3 | 0.04% |
| record | photo_count | 3 | 0.04% |
| record | price: Base price of the product in US dollars | 3 | 0.04% |
| record | drug_id | 3 | 0.04% |
| record | affected_population | 3 | 0.04% |
| record | annual_revenue: Annual revenue reported in millions of dollars | 3 | 0.04% |
| record | date: Date the exchange rate snapshot was taken | 3 | 0.04% |
| record | associated_diseases | 3 | 0.04% |
| record | disease_profile.prognosis.potential_complications | 3 | 0.04% |
| record | case_report.patient.recent_medical_events.0.complications.0.condition | 3 | 0.04% |
| record | case_report.clinical_presentation.lesion_locations | 3 | 0.04% |
| record | case_report.diagnostics.histopathology.absent_features | 3 | 0.04% |
| record | case_report.diagnostics.trichoscopy.findings.0.feature | 3 | 0.04% |
| record | case_report.patient.recent_medical_events.0.reason | 3 | 0.04% |
| record | case_report.diagnostics.histopathology.procedure | 3 | 0.04% |
| record | case_report.diagnostics.trichoscopy.findings.0.location_on_lesion | 3 | 0.04% |
| record | title: Event title describing the nature and context of the incident | 3 | 0.04% |
| record | players_not_returning | 3 | 0.04% |
| record | technology_area | 3 | 0.04% |
| record | concept | 3 | 0.04% |
| record | definitions | 3 | 0.04% |
| record | source_name | 3 | 0.04% |
| record | movie.comments.11.text | 3 | 0.04% |
| record | movie.comments.0.text | 3 | 0.04% |
| record | movie.rating.max_score | 3 | 0.04% |
| record | movie.comments.16.text | 3 | 0.04% |
| record | movie.comments.2.date | 3 | 0.04% |
| record | movie.comments.8.date | 3 | 0.04% |
| record | movie.comments.13.date | 3 | 0.04% |
| record | movie.comments.15.time | 3 | 0.04% |
| record | movie.comments.14.text | 3 | 0.04% |
| record | movie.rating.score | 3 | 0.04% |
| record | movie.director | 3 | 0.04% |
| record | movie.comments.15.date | 3 | 0.04% |
| record | movie.comments.18.date | 3 | 0.04% |
| record | movie.comments.0.time | 3 | 0.04% |
| record | early_challenges | 3 | 0.04% |
| record | specificity | 3 | 0.04% |
| record | issuing_company | 3 | 0.04% |
| record | document_status | 3 | 0.04% |
| record | funding_source: Type of funding received by the company | 3 | 0.04% |
| record | departure_code | 3 | 0.04% |
| record | date: Date of the patient visit | 3 | 0.04% |
| record | timestamp: Time of measurement in HH:MM:SS format | 3 | 0.04% |
| record | grade | 3 | 0.04% |
| record | symbols | 3 | 0.04% |
| record | verbs_past_participle | 3 | 0.04% |
| record | punctuation | 3 | 0.04% |
| record | case_number: Unique identifier for the legal case | 3 | 0.04% |
| record | respondent: Name of the person or entity being sued | 3 | 0.04% |
| record | credit_hours: Number of credit hours assigned to the course | 3 | 0.04% |
| record | location: Physical building and room where the course is held | 3 | 0.04% |
| record | end_date: Date when the course ends | 3 | 0.04% |
| record | municipalities | 3 | 0.04% |
| record | company.description | 3 | 0.04% |
| record | company.establishment_date | 3 | 0.04% |
| record | company.original_name | 3 | 0.04% |
| record | company.founding_location.city | 3 | 0.04% |
| record | company.founding_location.region | 3 | 0.04% |
| record | company.founders.1.name | 3 | 0.04% |
| record | limitations_of_midwives | 3 | 0.04% |
| record | relationships.0.entity | 3 | 0.04% |
| record | relationships.0.action | 3 | 0.04% |
| record | personnel.0.name | 3 | 0.04% |
| record | exam_centers | 3 | 0.04% |
| record | training.delivery_method | 3 | 0.04% |
| record | location: City and state where the individual is based | 3 | 0.04% |
| record | user_id: Unique identifier for the user initiating transaction | 3 | 0.04% |
| record | transaction_amount: Amount being converted | 3 | 0.04% |
| record | xcomp | 3 | 0.04% |
| record | ccomp | 3 | 0.04% |
| record | nsubjpass | 3 | 0.04% |
| record | snapshot_id: Unique identifier for the snapshot in format currency-yearmonthday | 3 | 0.04% |
| record | reading_type | 3 | 0.04% |
| record | date_range | 3 | 0.04% |
| record | official_name | 3 | 0.04% |
| record | technical_focus | 3 | 0.04% |
| record | application_id | 3 | 0.04% |
| record | bonus_amount | 3 | 0.04% |
| record | returner: Name of the person initiating the return | 3 | 0.04% |
| record | location: City or address where return was submitted | 3 | 0.04% |
| record | passengers_affected | 3 | 0.04% |
| record | year: Year the event was held | 3 | 0.04% |
| record | silver_medalist: Name of athlete who won silver medal | 3 | 0.04% |
| record | path_design.tile_based_paths.application_areas | 3 | 0.04% |
| record | review_message | 3 | 0.04% |
| record | ballot_type | 3 | 0.04% |
| record | asteroid.provisional_designation | 3 | 0.04% |
| record | asteroid.orbital_characteristics.jpl_classification | 3 | 0.04% |
| record | asteroid.physical_characteristics.observations.0.publication.year | 3 | 0.04% |
| record | asteroid.physical_characteristics.observations.0.publication.description | 3 | 0.04% |
| record | asteroid.physical_characteristics.observations.0.telescope.name | 3 | 0.04% |
| record | asteroid.physical_characteristics.observations.0.telescope.acronym | 3 | 0.04% |
| record | asteroid.physical_characteristics.observations.0.telescope.type | 3 | 0.04% |
| record | imaging_center | 3 | 0.04% |
| record | domain.requirements.minimum_length | 3 | 0.04% |
| record | domain.third_level_domains | 3 | 0.04% |
| record | claim_type: Type of medical procedure covered | 3 | 0.04% |
| record | ebit_margin | 3 | 0.04% |
| record | revenuebyregion | 3 | 0.04% |
| record | pass_rate | 3 | 0.04% |
| record | shares: Total number of shares received on the post | 3 | 0.04% |
| record | end | 3 | 0.04% |
| record | restaurant.cuisine | 3 | 0.04% |
| record | monitoring_mechanism | 3 | 0.04% |
| record | status: Current enrollment status | 3 | 0.04% |
| record | physical_characteristics.observations.0.telescope.orbit_year | 3 | 0.04% |
| record | term_years: Duration of the lease in years | 3 | 0.04% |
| record | warehouse_location: location name of the warehouse facility | 3 | 0.04% |
| record | administrative_division | 3 | 0.04% |
| record | dedication.name | 3 | 0.04% |
| record | reporter: Name of the employee who submitted the report | 3 | 0.04% |
| record | reporter_email: Email address of the reporter | 3 | 0.04% |
| record | ip_location: Geographic location of the suspicious IP address | 3 | 0.04% |
| record | transaction_amount: Amount of money transferred due to the phishing | 3 | 0.04% |
| record | address: Physical address of the company | 3 | 0.04% |
| record | federalism_definition.federalism_examples | 3 | 0.04% |
| record | south_yemen_conflict.allegations_against_sanaa_and_takfiris | 3 | 0.04% |
| record | south_yemen_conflict.historical_context.protests_and_killings.casualties | 3 | 0.04% |
| record | south_yemen_conflict.historical_context.discrimination_and_exclusion | 3 | 0.04% |
| record | south_yemen_conflict.accusations_against_governments.1.accusations | 3 | 0.04% |
| record | south_yemen_conflict.call_for_independence | 3 | 0.04% |
| record | patient_case.demographics.gender | 3 | 0.04% |
| record | patient_case.clinical_presentation.medical_history.0.condition | 3 | 0.04% |
| record | patient_case.genetic_analysis.mutation_details.type | 3 | 0.04% |
| record | design | 3 | 0.04% |
| record | disease.pathogenesis.risk_factors | 3 | 0.04% |
| record | disease.name | 3 | 0.04% |
| record | pro_accomplishments | 3 | 0.04% |
| record | features: Key functionalities or tools included | 3 | 0.04% |
| record | asteroid.named_after.location.province | 3 | 0.04% |
| record | disease.description | 3 | 0.04% |
| record | disease.subspecies.2.associated_diseases | 3 | 0.04% |
| record | paralympic_games.country | 3 | 0.04% |
| record | organizer_email | 3 | 0.04% |
| record | Phenomenon | 3 | 0.04% |
| record | incident_type | 3 | 0.04% |
| record | feedback: Numerical feedback score from participant | 3 | 0.04% |
| record | amount: Investment amount in millions of dollars | 3 | 0.04% |
| record | event_date: Date of the event the exhibitor participates in | 3 | 0.04% |
| record | initial_status | 3 | 0.04% |
| record | thumbnail_description | 3 | 0.04% |
| record | opponent: Candidate who lost the district | 3 | 0.04% |
| record | product_count | 3 | 0.04% |
| record | discount_amount | 3 | 0.04% |
| record | amount_paid | 3 | 0.04% |
| record | modality: Type of imaging modality used | 3 | 0.04% |
| record | stay_duration: Number of days the patient stayed in the hospital | 3 | 0.04% |
| record | hospital_name: Name of the hospital where discharge occurred | 3 | 0.04% |
| record | diagnosis: Medical condition such as hypertension or diabetes | 3 | 0.04% |
| record | visit_date: Date of last medical check-up | 3 | 0.04% |
| record | return_reason: Description of why the product was returned | 3 | 0.04% |
| record | name: Full name of the technology startup | 3 | 0.04% |
| record | clonality | 3 | 0.04% |
| record | sponsorship_level: Tier of sponsorship paid by the exhibitor | 3 | 0.04% |
| record | exhibitor_id: Unique identifier assigned to the exhibitor | 3 | 0.04% |
| record | birth_information.birth_year | 3 | 0.04% |
| record | profession | 3 | 0.04% |
| record | biography.education.1.institution | 3 | 0.04% |
| record | biography.education.2.teacher | 3 | 0.04% |
| record | biography.education.3.institution | 3 | 0.04% |
| record | biography.mentors | 3 | 0.04% |
| record | biography.teaching_position.institution | 3 | 0.04% |
| record | biography.awards.0.award_name | 3 | 0.04% |
| record | study.author.name | 3 | 0.04% |
| record | sunless_tanning.product_collection.products.1.ingredients | 3 | 0.04% |
| record | content_analysis.total_pieces | 3 | 0.04% |
| record | publication.language.classification | 3 | 0.04% |
| record | product.origin.initial_location | 3 | 0.04% |
| record | product.preparation.drying_time | 3 | 0.04% |
| record | historical_context.my_folks_series.first_book_title | 3 | 0.04% |
| record | contact_information.expert_help | 3 | 0.04% |
| record | transcription_factors | 3 | 0.04% |
| record | trial_type | 3 | 0.04% |
| record | upvote_count | 3 | 0.04% |
| record | comparison | 3 | 0.04% |
| record | post_date: Date when the trend repost was published | 3 | 0.04% |
| record | facebook_features | 3 | 0.04% |
| record | administrative_divisions.community_committees | 3 | 0.04% |
| record | administrative_divisions.village_committees | 3 | 0.04% |
| record | viewer_experience.interpretation | 3 | 0.04% |
| record | patient_name: Full name of the claimant | 3 | 0.04% |
| record | amount: Total cost of the medical service | 3 | 0.04% |
| record | israel_airports.largest_airport.city | 3 | 0.04% |
| record | israel_airports.largest_airport.description | 3 | 0.04% |
| record | israel_airports.largest_airport.airport_name | 3 | 0.04% |
| record | return_date_prompt | 3 | 0.04% |
| record | media_analysis.events.0.media_response | 3 | 0.04% |
| record | product: Model name of the defective product | 3 | 0.04% |
| record | estimatedrepairtime: Number of days for repair completion | 3 | 0.04% |
| record | playeroneid | 3 | 0.04% |
| record | playertwoname | 3 | 0.04% |
| record | innovation_type | 3 | 0.04% |
| record | device_type | 3 | 0.04% |
| record | hotel_experience.suites.0.suite_count | 3 | 0.04% |
| record | hotel_experience.description | 3 | 0.04% |
| record | review.reviewer | 3 | 0.04% |
| record | review.comment | 3 | 0.04% |
| record | revenue: Total reported revenue in millions | 3 | 0.04% |
| record | seller_id | 3 | 0.04% |
| record | team: Name of the team playing in the game | 3 | 0.04% |
| record | domain.dispute_resolution.methods | 3 | 0.04% |
| record | outreach_amount: Monetary value associated with the outreach | 3 | 0.04% |
| record | patient.behavior | 3 | 0.04% |
| record | patient.age_estimation.full_age | 3 | 0.04% |
| record | chair.condition | 3 | 0.04% |
| record | location: Geographic location of product launch | 3 | 0.04% |
| record | customer_name: Name of the customer initiating the return | 3 | 0.04% |
| record | return_date: Date when the return was submitted | 3 | 0.04% |
| record | ram | 3 | 0.04% |
| record | projects: Type of projects the employee is involved in | 3 | 0.04% |
| record | policyholder: Name of the policyholder filing the claim | 3 | 0.04% |
| record | mark | 3 | 0.04% |
| record | events.0.participants | 3 | 0.04% |
| record | countries.2.name | 3 | 0.04% |
| record | report_details.source | 3 | 0.04% |
| record | countries.1.breastfeeding_rate | 3 | 0.04% |
| record | principal_investigator | 3 | 0.04% |
| record | ingredient | 3 | 0.04% |
| record | feedback_status | 3 | 0.04% |
| record | fiscalyearend | 3 | 0.04% |
| record | programStructure.dailySchedule.0.day | 3 | 0.04% |
| record | fees.inclusions | 3 | 0.04% |
| record | programStructure.dailySchedule.1.activities.2.description | 3 | 0.04% |
| record | programStructure.dailySchedule.1.activities.3.time | 3 | 0.04% |
| record | programStructure.dailySchedule.0.activities.3.description | 3 | 0.04% |
| record | programStructure.dailySchedule.0.activities.0.time | 3 | 0.04% |
| record | programStructure.dailySchedule.0.activities.0.description | 3 | 0.04% |
| record | accommodation.responsibility | 3 | 0.04% |
| record | programStructure.dailySchedule.0.activities.2.description | 3 | 0.04% |
| record | accommodation.options.0.phone | 3 | 0.04% |
| record | usage_count: Total patient admissions or scans performed | 3 | 0.04% |
| record | data_collection | 3 | 0.04% |
| record | search_result_page.search_query | 3 | 0.04% |
| record | foods_mentioned | 3 | 0.04% |
| record | patient_case.demographics.ethnicity | 3 | 0.04% |
| record | patient_case.clinical_presentation.skin_lesions.size_range.max_mm | 3 | 0.04% |
| record | disease_profile.classification | 3 | 0.04% |
| record | target audience | 3 | 0.04% |
| record | opposing_members | 3 | 0.04% |
| record | integration_task_id | 3 | 0.04% |
| record | town.country | 3 | 0.04% |
| record | town.establishment_year | 3 | 0.04% |
| record | rebounds_leader | 3 | 0.04% |
| record | donations.recipient.name | 3 | 0.04% |
| record | event.time.photo_op_start | 3 | 0.04% |
| record | address: Street address where the incident happened | 3 | 0.04% |
| record | cause: Reason for the damage such as flood or fire | 3 | 0.04% |
| record | main_competitor | 3 | 0.04% |
| record | final_match | 3 | 0.04% |
| record | sender_email: Email address from which the phishing message originated | 3 | 0.04% |
| record | material_efficiency | 3 | 0.04% |
| record | enrollment_id | 3 | 0.04% |
| record | agency.name | 3 | 0.04% |
| record | agency.mission | 3 | 0.04% |
| record | citizen_corps.website | 3 | 0.04% |
| record | prescriber_specialty | 3 | 0.04% |
| record | post_title | 3 | 0.04% |
| record | tracking_id | 3 | 0.04% |
| record | finalscore: Scoreline in format homescore-away_score | 3 | 0.04% |
| record | diagnosis_code: Medical code assigned to the patient's condition | 3 | 0.04% |
| record | Eckart <> compared to | 3 | 0.04% |
| record | Robert <> questioned | 3 | 0.04% |
| record | Robert <> referenced | 3 | 0.04% |
| record | Robert <> utilized | 3 | 0.04% |
| record | asteroid.characteristics.type | 3 | 0.04% |
| record | album.sources.1.source_name | 3 | 0.04% |
| record | album.tracklist.cd.0.composer | 3 | 0.04% |
| record | album.tracklist.cd.9.notes | 3 | 0.04% |
| record | album.tracklist.cd.3.tie_in.0.title | 3 | 0.04% |
| record | album.tracklist.cd.9.composer | 3 | 0.04% |
| record | album.tracklist.cd.10.composer | 3 | 0.04% |
| record | album.tracklist.cd.5.title | 3 | 0.04% |
| record | project_title: Title of the open-source project reviewed | 3 | 0.04% |
| record | sunless_tanning.product_collection.products.0.name | 3 | 0.04% |
| record | hotel.booking_options.partners | 3 | 0.04% |
| record | review.stay_duration | 3 | 0.04% |
| record | review.travelers.0.relationship | 3 | 0.04% |
| record | review.travelers.0.age | 3 | 0.04% |
| record | review.overall_experience | 3 | 0.04% |
| record | technology: Core technology or method used | 3 | 0.04% |
| record | address: Full address of the customer | 3 | 0.04% |
| record | outage_end: End time of the service disruption | 3 | 0.04% |
| record | clinical_outcomes.glycemic_control.effectiveness | 3 | 0.04% |
| record | registered_attendees | 3 | 0.04% |
| record | issuing_body | 3 | 0.04% |
| record | originalorderid | 3 | 0.04% |
| record | app_package | 3 | 0.04% |
| record | motion_status: Status of any filed motions such as denied or pending | 3 | 0.04% |
| record | submission.submitted_by | 3 | 0.04% |
| record | submission.submission_date | 3 | 0.04% |
| record | route: Administration method of the drug | 3 | 0.04% |
| record | movie.rating.votes | 3 | 0.04% |
| record | movie.comments.1.text | 3 | 0.04% |
| record | movie.comments.4.date | 3 | 0.04% |
| record | movie.comments.5.time | 3 | 0.04% |
| record | movie.comments.12.date | 3 | 0.04% |
| record | movie.comments.21.text | 3 | 0.04% |
| record | scriptures | 3 | 0.04% |
| record | asteroid.named_after.person.name_native | 3 | 0.04% |
| record | asteroid.named_after.person.awards | 3 | 0.04% |
| record | edition | 3 | 0.04% |
| record | speaker_id: Unique identifier for each press conference statement | 3 | 0.04% |
| record | speaker_name: Full name of the individual presenting the statement | 3 | 0.04% |
| record | movement_reason: Brief reason for price movement or performance | 3 | 0.04% |
| record | shipment_id | 3 | 0.04% |
| record | fruit_properties.size | 3 | 0.04% |
| record | discharge_condition | 3 | 0.04% |
| record | parameters | 3 | 0.04% |
| record | product_name: Name of the product as listed | 3 | 0.04% |
| record | listing_date: Date the product was listed | 3 | 0.04% |
| record | asteroid.alternative_name.german | 3 | 0.04% |
| record | apparent_magnitude.band | 3 | 0.04% |
| record | asteroid.name_original | 3 | 0.04% |
| record | town.name | 3 | 0.04% |
| record | town.population.density_per_sq_km | 3 | 0.04% |
| record | location: City and state or province where purchase occurred | 3 | 0.04% |
| record | product_id: Unique identifier for the product reviewed | 3 | 0.04% |
| record | date_reported | 3 | 0.04% |
| record | election_district | 3 | 0.04% |
| record | conservation_status.status | 3 | 0.04% |
| record | description.leaf.base | 3 | 0.04% |
| record | location.other_locations | 3 | 0.04% |
| record | discovery.location.state | 3 | 0.04% |
| record | discovery.location.county | 3 | 0.04% |
| record | athlete: Name of the athlete who won the medal | 3 | 0.04% |
| record | nation: Country of the athlete who won the medal | 3 | 0.04% |
| record | medal: Type of medal won gold, silver, or bronze | 3 | 0.04% |
| record | status: Current status of the prescription | 3 | 0.04% |
| record | patient_name: Full name of the patient receiving the medication | 3 | 0.04% |
| record | abstract_id: Unique identifier for the research abstract | 3 | 0.04% |
| record | institution: Name of the medical institution conducting the study | 3 | 0.04% |
| record | comorbidities: List of underlying health conditions present in patients | 3 | 0.04% |
| record | primary_outcome: Mean reduction in symptom duration in days | 3 | 0.04% |
| record | adverse_events: Percentage of patients reporting side effects | 3 | 0.04% |
| record | distributor.name | 3 | 0.04% |
| record | distributor.location | 3 | 0.04% |
| record | primates | 3 | 0.04% |
| record | source_website | 3 | 0.04% |
| record | focus_area: Primary domain or mission of the company | 3 | 0.04% |
| record | crime.location.landmark | 3 | 0.04% |
| record | crime.motive | 3 | 0.04% |
| record | crime.weapon.caliber | 3 | 0.04% |
| record | crime.date | 3 | 0.04% |
| record | market_position | 3 | 0.04% |
| record | monthly_fee | 3 | 0.04% |
| record | enrollment_code | 3 | 0.04% |
| record | case_report.diagnostics.histopathology.findings | 3 | 0.04% |
| record | case_report.clinical_presentation.lesion_description.1.features | 3 | 0.04% |
| record | case_report.patient.recent_medical_events.0.duration_days | 3 | 0.04% |
| record | case_report.patient.recent_medical_events.0.complications.0.duration_days | 3 | 0.04% |
| record | product: Name of the skincare or apparel product | 3 | 0.04% |
| record | status: Current status of the review document | 3 | 0.04% |
| record | habitat.associated_species | 3 | 0.04% |
| record | description.floral_buds.dimensions | 3 | 0.04% |
| record | funding_amount: Total amount raised in the investment round | 3 | 0.04% |
| record | inventor: Name of the patent applicant | 3 | 0.04% |
| record | filing_date: Date the patent application was filed | 3 | 0.04% |
| record | funding_source: Source of funding for the patent development | 3 | 0.04% |
| record | board_approval_date | 3 | 0.04% |
| record | department_head | 3 | 0.04% |
| record | mutation.genes | 3 | 0.04% |
| record | mutation.effect | 3 | 0.04% |
| record | status: Current state of prescription fulfillment | 3 | 0.04% |
| record | vacancy_rate.change | 3 | 0.04% |
| record | location: City where the employee is based | 3 | 0.04% |
| record | state: State where the customer resides | 3 | 0.04% |
| record | admitting_diagnosis | 3 | 0.04% |
| record | controversy.victim_actresses.7.actress_name | 3 | 0.04% |
| record | controversy.krk_allegation.comment | 3 | 0.04% |
| record | controversy.victim_actresses.3.krk_comment | 3 | 0.04% |
| record | controversy.victim_actresses.5.krk_comment | 3 | 0.04% |
| record | controversy.krk_allegation.target | 3 | 0.04% |
| record | controversy.victim_actresses.6.actress_name | 3 | 0.04% |
| record | feedback_content | 3 | 0.04% |
| record | current_attendees | 3 | 0.04% |
| record | context.problem | 3 | 0.04% |
| record | sql_server_versions | 3 | 0.04% |
| record | dates.last_reviewed | 3 | 0.04% |
| record | network_protocols | 3 | 0.04% |
| record | country: Country code from patent number prefix | 3 | 0.04% |
| record | address: Full address of the inventor | 3 | 0.04% |
| record | infrastructure.roads.0.connections | 3 | 0.04% |
| record | infrastructure.airports.0.location | 3 | 0.04% |
| record | infrastructure.airports.0.passengersYear | 3 | 0.04% |
| record | economy.gdpRankAmongRegions | 3 | 0.04% |
| record | history.establishmentDate | 3 | 0.04% |
| record | outcome_measure | 3 | 0.04% |
| record | victim.injury | 3 | 0.04% |
| record | metadata.source_links.0.url | 3 | 0.04% |
| record | primary_outcome | 3 | 0.04% |
| record | disease_profile.inheritance_pattern | 3 | 0.04% |
| record | disease_profile.genetic_mapping.associated_gene | 3 | 0.04% |
| record | disease_profile.genetic_mapping.gene_aliases | 3 | 0.04% |
| record | patient_case.genetic_analysis.mutation_details.protein_nomenclature | 3 | 0.04% |
| record | asteroid.discovery_date | 3 | 0.04% |
| record | participant: Name of the individual involved in the transaction | 3 | 0.04% |
| record | candidate_name: Full name of the candidate as listed in results | 3 | 0.04% |
| record | note: Brief descriptive commentary on the book's content and quality | 3 | 0.04% |
| record | word_count: Total number of words in the book | 3 | 0.04% |
| record | genre: Literary category of the book such as fiction or poetry | 3 | 0.04% |
| record | handle: Social media handle for the brand | 3 | 0.04% |
| record | inventors: List of individuals who filed the patent | 3 | 0.04% |
| record | inventor_location: Location of the inventor(s) | 3 | 0.04% |
| record | disease_profile.prognosis.expected_outcome | 3 | 0.04% |
| record | credits.funding | 3 | 0.04% |
| record | destination_address | 3 | 0.04% |
| record | medication: Name of the prescribed medication | 3 | 0.04% |
| record | devices | 3 | 0.04% |
| record | programStructure.dailySchedule.0.activities.2.time | 3 | 0.04% |
| record | programStructure.materialsProvided | 3 | 0.04% |
| record | availability.registration | 3 | 0.04% |
| record | fees.certificationFee | 3 | 0.04% |
| record | linguistic_framework.concluding_grammar_focus | 3 | 0.04% |
| record | trend_type | 3 | 0.04% |
| record | repost_count | 3 | 0.04% |
| record | delegation.medals.gold | 3 | 0.04% |
| record | delegation.medals.bronze | 3 | 0.04% |
| record | model_or_theory | 3 | 0.04% |
| record | Dariusz Śledź <> occupation | 3 | 0.04% |
| record | content: Text content of the tweet describing a product or topic | 3 | 0.04% |
| record | domain.country_code | 3 | 0.04% |
| record | main_character.name | 3 | 0.04% |
| record | publication_details.first_publication.location | 3 | 0.04% |
| record | polish_titles | 3 | 0.04% |
| record | away_score | 3 | 0.04% |
| record | javascript_requirement.enable_instruction | 3 | 0.04% |
| record | shipping_restriction.contact_seller_prompt | 3 | 0.04% |
| record | shipping_restriction.country | 3 | 0.04% |
| record | javascript_requirement.question_prompt | 3 | 0.04% |
| record | update_notes | 3 | 0.04% |
| record | publication.author.affiliation.institution | 3 | 0.04% |
| record | place.demographics.0.population | 3 | 0.04% |
| record | taxonomy.describer.affiliation.0.laboratory | 3 | 0.04% |
| record | taxonomy.describer.affiliation.0.university | 3 | 0.04% |
| record | taxonomy.describer.affiliation.0.location.city | 3 | 0.04% |
| record | taxonomy.year_described | 3 | 0.04% |
| record | taxonomy.type_materials | 3 | 0.04% |
| record | onboarding_code | 3 | 0.04% |
| record | seller_name: Name of the seller or vendor | 3 | 0.04% |
| record | market_segment | 3 | 0.04% |
| record | resolution_hours | 3 | 0.04% |
| record | vendor_name: Name of the supplying vendor company | 3 | 0.04% |
| record | distributor_information.benefits | 3 | 0.04% |
| record | practice_areas | 3 | 0.04% |
| record | states_served | 3 | 0.04% |
| record | experience.years | 3 | 0.04% |
| record | dateofbirth | 3 | 0.04% |
| record | outbreak_event.illness.pathogen.full_name | 3 | 0.04% |
| record | outbreak_event.illness.pathogen.abbreviation | 3 | 0.04% |
| record | outbreak_event.location.country | 3 | 0.04% |
| record | cpt_code: CPT code for the medical procedure performed | 3 | 0.04% |
| record | founder_name: Name of the founder who started the venture | 3 | 0.04% |
| record | genus.name_zh | 3 | 0.04% |
| record | employee_id: Unique identifier for each new hire entry | 3 | 0.04% |
| record | training_completed: Percentage of training completed out of 100 | 3 | 0.04% |
| record | ship_date | 3 | 0.04% |
| record | sku: Stock keeping unit assigned to each product | 3 | 0.04% |
| record | seller: Company name that sells the product | 3 | 0.04% |
| record | access_duration | 3 | 0.04% |
| record | video_id: Unique identifier for the video as assigned by YouTube | 3 | 0.04% |
| record | icd_code | 3 | 0.04% |
| record | contact.phone | 3 | 0.04% |
| record | office_hours.1.end_time | 3 | 0.04% |
| record | address.building | 3 | 0.04% |
| record | address.city | 3 | 0.04% |
| record | address.postal_code | 3 | 0.04% |
| record | passenger_phone | 3 | 0.04% |
| record | vip_price | 3 | 0.04% |
| record | recipient: Name of the recipient professional contacted | 3 | 0.04% |
| record | sender: Name of the recruiter initiating the outreach | 3 | 0.04% |
| record | hashtag: Primary hashtag used in the launch post | 3 | 0.04% |
| record | event.time.end | 3 | 0.04% |
| record | species.etymology.type_locality | 3 | 0.04% |
| record | distribution.environment | 3 | 0.04% |
| record | description.diet | 3 | 0.04% |
| record | services.selection_assistance | 3 | 0.04% |
| record | process_details.mediator_role.function | 3 | 0.04% |
| record | availability_start | 3 | 0.04% |
| record | author_name: Name of the paper's author | 3 | 0.04% |
| record | paper_title: Title of the submitted research paper | 3 | 0.04% |
| record | support_agent | 3 | 0.04% |
| record | address: Full residential address of the patient | 3 | 0.04% |
| record | rating: Numerical score given to the book out of five | 3 | 0.04% |
| record | languages | 3 | 0.04% |
| record | address.room | 3 | 0.04% |
| record | affected_people | 3 | 0.04% |
| record | person.full_name | 3 | 0.04% |
| record | person.phone_number | 3 | 0.04% |
| record | metadata.report_date | 3 | 0.04% |
| record | metadata.source_links.0.language | 3 | 0.04% |
| record | lab_results | 3 | 0.04% |
| record | address: Full street address of the rental property | 3 | 0.04% |
| record | match_code: Unique identifier for the match as recorded in the database | 3 | 0.04% |
| record | VB | 3 | 0.04% |
| record | program | 3 | 0.04% |
| record | rating: Numerical score out of 5 given by the reviewer | 3 | 0.04% |
| record | player_action: Description of player's contribution such as goal or interception | 3 | 0.04% |
| record | lineup | 3 | 0.04% |
| record | affiliation: Department or institute where authors are affiliated | 3 | 0.04% |
| record | price_open | 3 | 0.04% |
| record | code_of_conduct | 3 | 0.04% |
| record | race | 3 | 0.04% |
| record | response_status | 3 | 0.04% |
| record | tax_incentive.duration.period.start_year | 3 | 0.04% |
| record | primate.weight_grams | 3 | 0.04% |
| record | github_url | 3 | 0.04% |
| record | player_id: Unique identifier of a player involved in action | 3 | 0.04% |
| record | price: Purchase price in US dollars | 3 | 0.04% |
| record | battery_life: Battery duration in hours under normal use | 3 | 0.04% |
| record | document_details.series_title | 3 | 0.04% |
| record | proper_nouns | 3 | 0.04% |
| record | possessive_pronouns | 3 | 0.04% |
| record | plural_nouns | 3 | 0.04% |
| record | infrastructure.railways | 3 | 0.04% |
| record | infrastructure.highways.2.connections | 3 | 0.04% |
| record | populationYear | 3 | 0.04% |
| record | infrastructure.roads.0.roadName | 3 | 0.04% |
| record | infrastructure.refineries.0.location | 3 | 0.04% |
| record | economy.gdpValue | 3 | 0.04% |
| record | geography.borders.1.region | 3 | 0.04% |
| record | infrastructure.airports.0.airportName | 3 | 0.04% |
| record | regionNameArabic | 3 | 0.04% |
| record | claims: Number of claims covered by the patent | 3 | 0.04% |
| record | hotel_experience.location.postal_code_accessibility | 3 | 0.04% |
| record | event_id: Unique identifier for the event as shown in ID column | 3 | 0.04% |
| record | guidelines.distinction_results_interpretation | 3 | 0.04% |
| record | optional_information | 3 | 0.04% |
| record | publication_date.city | 3 | 0.04% |
| record | report_date: Date when the issue was first reported | 3 | 0.04% |
| record | business_activities | 3 | 0.04% |
| record | teacher | 3 | 0.04% |
| record | death.location | 3 | 0.04% |
| record | owner: Name of the customer who submitted the ticket | 3 | 0.04% |
| record | supplies_provided | 3 | 0.04% |
| record | score_away | 3 | 0.04% |
| record | recipient_office: Office or position being supported | 3 | 0.04% |
| record | timestamp: Date when the transaction occurred | 3 | 0.04% |
| record | wallet_address: Wallet address used for the transaction | 3 | 0.04% |
| record | user_name: Name of the user associated with the transaction | 3 | 0.04% |
| record | message_length: Word count of the outreach message | 3 | 0.04% |
| record | rating: Numerical rating from 1 to 5 given by the user | 3 | 0.04% |
| record | page_count | 3 | 0.04% |
| record | historical_context.time_periods.0.events.0.measures | 3 | 0.04% |
| record | historical_context.time_periods.3.events.0.measures | 3 | 0.04% |
| record | historical_context.diseases | 3 | 0.04% |
| record | access_information.access_type | 3 | 0.04% |
| record | training_program.invitation_steps | 3 | 0.04% |
| record | training_program.payment.options | 3 | 0.04% |
| record | link: URL included in the tweet pointing to additional content | 3 | 0.04% |
| record | user_id: Unique identifier for the tweet author | 3 | 0.04% |
| record | retweets: Number of times the tweet was retweeted | 3 | 0.04% |
| record | stock_ticker | 3 | 0.04% |
| record | funding_id: Unique identifier assigned to the funding round | 3 | 0.04% |
| record | publication_date: Date the book was released | 3 | 0.04% |
| record | bracket_code | 3 | 0.04% |
| record | inventor: Name of the patent's inventor | 3 | 0.04% |
| record | funding_code: Code indicating eligibility for funding support | 3 | 0.04% |
| record | person.publications.0.volume | 3 | 0.04% |
| record | service.location.address | 3 | 0.04% |
| record | service.contact_information.email | 3 | 0.04% |
| record | person.death_place | 3 | 0.04% |
| record | company: Company advertising the job position | 3 | 0.04% |
| record | body: Content of the outreach message | 3 | 0.04% |
| record | date_listed | 3 | 0.04% |
| record | episode_number | 3 | 0.04% |
| record | price: Current selling price of the SKU in USD | 3 | 0.04% |
| record | thread_id | 3 | 0.04% |
| record | customer_retention | 3 | 0.04% |
| record | pathway_name: Name of the biological pathway such as glycolysis or insulin signaling | 3 | 0.04% |
| record | organism: Species in which the pathway is observed | 3 | 0.04% |
| record | company_name: Name of the technology company raising funds | 3 | 0.04% |
| record | alternative_designations | 3 | 0.04% |
| record | comments: Total number of comments on the video | 3 | 0.04% |
| record | compliance_status: Compliance classification with local regulations | 3 | 0.04% |
| record | booking_options.description | 3 | 0.04% |
| record | ranking.location | 3 | 0.04% |
| record | isbn | 3 | 0.04% |
| record | investors: List of investor names or groups involved | 3 | 0.04% |
| record | co_investors | 3 | 0.04% |
| record | title: Topic or title of the submitted work | 3 | 0.04% |
| record | conference_name | 3 | 0.04% |
| record | company_name: Name of the startup founded by the founder | 3 | 0.04% |
| record | epistle.book | 3 | 0.04% |
| record | prokeimenon.text | 3 | 0.04% |
| record | -LRB- | 3 | 0.04% |
| record | naming.date | 3 | 0.04% |
| record | study_overview.condition_detail | 3 | 0.04% |
| record | methodology.study_period.end_year | 3 | 0.04% |
| record | methodology.population.total_department_nsclc_patients | 3 | 0.04% |
| record | methodology.population.study_group_size | 3 | 0.04% |
| record | methodology.population.study_group_percentage | 3 | 0.04% |
| record | patient_demographics.gender_distribution.men.percentage | 3 | 0.04% |
| record | patient_demographics.gender_distribution.women.percentage | 3 | 0.04% |
| record | patient_demographics.age_stats.min | 3 | 0.04% |
| record | patient_demographics.age_stats.max | 3 | 0.04% |
| record | treatment_details.radiotherapy.regimen | 3 | 0.04% |
| record | treatment_details.pulmonary_resection_types.0.count | 3 | 0.04% |
| record | treatment_details.pulmonary_resection_types.2.count | 3 | 0.04% |
| record | pathological_findings.nodal_involvement_distribution.0.patient_count | 3 | 0.04% |
| record | pathological_findings.nodal_involvement_distribution.1.patient_count | 3 | 0.04% |
| record | pathological_findings.statistical_factors.0.p_value | 3 | 0.04% |
| record | pathological_findings.statistical_factors.1.factor | 3 | 0.04% |
| record | outcomes.survival_rates.five_year_percentage | 3 | 0.04% |
| record | outcomes.disease_free_survival_months_average | 3 | 0.04% |
| record | outcomes.mortality_by_nodal_involvement.0.location | 3 | 0.04% |
| record | outcomes.mortality_by_nodal_involvement.1.survival_limit_months | 3 | 0.04% |
| record | away_score: Final score of the away team | 3 | 0.04% |
| record | refill_count | 3 | 0.04% |
| record | merchant_name: Name of the merchant or store | 3 | 0.04% |
| record | late_fee: Fee assessed for missing payment deadline | 3 | 0.04% |
| record | status: Transaction approval or rejection status | 3 | 0.04% |
| record | regions_or_states | 3 | 0.04% |
| record | key_comment: Specific positive or negative observation about the stay | 3 | 0.04% |
| record | entities.number_of_backgrounds | 3 | 0.04% |
| record | research_background.focus_area | 3 | 0.04% |
| record | key_intermediate: Central metabolite in the pathway | 3 | 0.04% |
| record | physical_characteristics.observations.0.telescope.country | 3 | 0.04% |
| record | conclusions.advantages | 3 | 0.04% |
| record | photos_count | 3 | 0.04% |
| record | payment_type | 3 | 0.04% |
| record | topic: Subject area or focus of the submitted paper | 3 | 0.04% |
| record | deprotection_process.reagents.0.name | 3 | 0.04% |
| record | test_id | 3 | 0.04% |
| record | document.economic_impact.employment | 3 | 0.04% |
| record | document.strategy_elements.6.area | 3 | 0.04% |
| record | document.counterfeiting_statistics.estimated_losses | 3 | 0.04% |
| record | document.reference_id | 3 | 0.04% |
| record | document.commissioners.0.portfolio | 3 | 0.04% |
| record | document.strategy_elements.5.area | 3 | 0.04% |
| record | game_system | 3 | 0.04% |
| record | account_number: Unique identifier for the customer's account | 3 | 0.04% |
| record | minimum_payment: Minimum amount required to be paid | 3 | 0.04% |
| record | article_metadata.origin_newsroom | 3 | 0.04% |
| record | organization.legal_form | 3 | 0.04% |
| record | delayed_games | 3 | 0.04% |
| record | upcoming_games | 3 | 0.04% |
| record | address: Street address including city and zip code | 3 | 0.04% |
| record | credits.content_implementers | 3 | 0.04% |
| record | recipient_jurisdiction: Jurisdiction such as city, state, or federal level | 3 | 0.04% |
| record | donor_location: City and state of the donor | 3 | 0.04% |
| record | javascript_requirement.message | 3 | 0.04% |
| record | historical_context.my_folks_series.first_book_year | 3 | 0.04% |
| record | sourcing.coffee_source.location | 3 | 0.04% |
| record | hours_of_operation.3.close_time | 3 | 0.04% |
| record | treatment.target | 3 | 0.04% |
| record | treatment.drugs | 3 | 0.04% |
| record | material: Composition of the product's physical materials | 3 | 0.04% |
| record | mountain | 3 | 0.04% |
| record | trail | 3 | 0.04% |
| record | name: Full name of the enrolled employee | 3 | 0.04% |
| record | on_site_registration.requirements | 3 | 0.04% |
| record | weight: Product weight in pounds | 3 | 0.04% |
| record | review.persons.2.name | 3 | 0.04% |
| record | review.summary | 3 | 0.04% |
| record | review.persons.0.name | 3 | 0.04% |
| record | review.performance.composer | 3 | 0.04% |
| record | tracking_code | 3 | 0.04% |
| record | registration_fee: Monetary amount due for conference registration | 3 | 0.04% |
| record | presentation_slot: Time slot for the presentation at the conference | 3 | 0.04% |
| record | city.birthplace_of.0.person.occupation | 3 | 0.04% |
| record | city.location | 3 | 0.04% |
| record | training.providers | 3 | 0.04% |
| record | demographics.variation_years | 3 | 0.04% |
| record | demographics.elaboration | 3 | 0.04% |
| record | epistle.chapter | 3 | 0.04% |
| record | debttoebitda_ratio: Ratio of debt to EBITDA indicating leverage | 3 | 0.04% |
| record | hometeampoints | 3 | 0.04% |
| record | gold_medalist: Name of athlete who won gold medal | 3 | 0.04% |
| record | battery_life: Battery duration noted by the reviewer | 3 | 0.04% |
| record | agency_code | 3 | 0.04% |
| record | description: Brief description of unit features or conditions | 3 | 0.04% |
| record | dimensions: Dimensions in cm format | 3 | 0.04% |
| record | arrival_airport | 3 | 0.04% |
| record | departure_airport | 3 | 0.04% |
| record | venue: Name of the arena where the game was played | 3 | 0.04% |
| record | protecting_group.abbreviation | 3 | 0.04% |
| record | synthesis_context.methodology | 3 | 0.04% |
| record | details.date_posted | 3 | 0.04% |
| record | review_id: Unique identifier assigned to the review | 3 | 0.04% |
| record | birth_date: Patient's date of birth | 3 | 0.04% |
| record | antibody_id | 3 | 0.04% |
| record | product_id: Unique identifier for the product | 3 | 0.04% |
| record | symbol: Stock ticker symbol as listed in the data | 3 | 0.04% |
| record | change_percent: Percentage change in stock price for the day | 3 | 0.04% |
| record | performance_notes | 3 | 0.04% |
| record | search_result_page.content_types | 3 | 0.04% |
| record | notes: Additional context or explanation for the transaction | 3 | 0.04% |
| record | source_provider: Provider handling the exchange rate calculation | 3 | 0.04% |
| record | author_information.education.0.institution | 3 | 0.04% |
| record | sector: Industry sector classification | 3 | 0.04% |
| record | statue.location.code | 3 | 0.04% |
| record | statue.restoration.0.reason | 3 | 0.04% |
| record | materials | 3 | 0.04% |
| record | description.flowers.perianth_tube_length_mm.min | 3 | 0.04% |
| record | amount: Monetary value of the returned item | 3 | 0.04% |
| record | product_name: Name of the product returned | 3 | 0.04% |
| record | manufacturer: Company that produced the device | 3 | 0.04% |
| record | subject_line | 3 | 0.04% |
| record | priority_registration.deadline | 3 | 0.04% |
| record | location: Geographic location of transaction | 3 | 0.04% |
| record | base_currency: Original currency used in transaction | 3 | 0.04% |
| record | prevention | 3 | 0.04% |
| record | drexel_info | 3 | 0.04% |
| record | journal.issn | 3 | 0.04% |
| record | donor_type: Type of donor as defined in the disclosure | 3 | 0.04% |
| record | breastfeeding_situation.trend | 3 | 0.04% |
| record | special_session | 3 | 0.04% |
| record | salary_min | 3 | 0.04% |
| record | distribution_channels | 3 | 0.04% |
| record | established | 3 | 0.04% |
| record | boiling_instructions.salt_advice | 3 | 0.04% |
| record | boiling_instructions.cooking_time.min_minutes | 3 | 0.04% |
| record | grilling_instructions.cooking_time.minutes | 3 | 0.04% |
| record | address: Full property address including street and city | 3 | 0.04% |
| record | kitchen appliance | 3 | 0.04% |
| record | diagnosis_date | 3 | 0.04% |
| record | allergy | 3 | 0.04% |
| record | bmi | 3 | 0.04% |
| record | tehsil.tehsil_name | 3 | 0.04% |
| record | patient_cohort | 3 | 0.04% |
| record | temp_f | 3 | 0.04% |
| record | address: Street address including number and street name | 3 | 0.04% |
| record | audit_team | 3 | 0.04% |
| record | previous_jobs | 3 | 0.04% |
| record | artistic_intent.style | 3 | 0.04% |
| record | narrative.setting.time | 3 | 0.04% |
| record | media_analysis.historical_context.reference | 3 | 0.04% |
| record | city.state | 3 | 0.04% |
| record | geography.borders.0.direction | 3 | 0.04% |
| record | deck_coating_systems.0.materials | 3 | 0.04% |
| record | news_id | 3 | 0.04% |
| record | clinical_phase: Current stage of clinical development | 3 | 0.04% |
| record | items | 3 | 0.04% |
| record | brand_name | 3 | 0.04% |
| record | century | 3 | 0.04% |
| record | images | 3 | 0.04% |
| record | document.related_documents | 3 | 0.04% |
| record | document.title | 3 | 0.04% |
| record | document.strategy_elements.0.area | 3 | 0.04% |
| record | document.counterfeiting_statistics.job_losses | 3 | 0.04% |
| record | recipient: Name of the person receiving the outreach | 3 | 0.04% |
| record | dish_cost | 2 | 0.03% |
| record | biography.education.1.teacher | 2 | 0.03% |
| record | address: Full street address of the recipient | 2 | 0.03% |
| record | hours_of_operation.2.day | 2 | 0.03% |
| record | card_id | 2 | 0.03% |
| record | zebras_care_foundation.beneficiary.family | 2 | 0.03% |
| record | zebras_care_foundation.introducer.affiliation | 2 | 0.03% |
| record | page_count: Number of pages in the paper | 2 | 0.03% |
| record | source.newspaper | 2 | 0.03% |
| record | wallet_address: Unique wallet address used for the transaction | 2 | 0.03% |
| record | reaction: Observed adverse effect or symptom | 2 | 0.03% |
| record | delivery_method | 2 | 0.03% |
| record | areas_for_improvement | 2 | 0.03% |
| record | fieldofstudy | 2 | 0.03% |
| record | John McAfee <> fled from | 2 | 0.03% |
| record | dispute_type: Type of legal action such as cancellation or infringement | 2 | 0.03% |
| record | player1_rank | 2 | 0.03% |
| record | customer_contracts | 2 | 0.03% |
| record | region.location.includes | 2 | 0.03% |
| record | term_length_months | 2 | 0.03% |
| record | attachment_type | 2 | 0.03% |
| record | dispute_resolution_method | 2 | 0.03% |
| record | regulatory_threshold | 2 | 0.03% |
| record | renewal_amount: Monetary amount required for policy renewal | 2 | 0.03% |
| record | status: Current status of policy renewal | 2 | 0.03% |
| record | exam_grade | 2 | 0.03% |
| record | access_status: Whether the content is open access or paywalled | 2 | 0.03% |
| record | team2: Second team participating in the match | 2 | 0.03% |
| record | required_documents | 2 | 0.03% |
| record | investment_amount | 2 | 0.03% |
| record | disruption_duration | 2 | 0.03% |
| record | monitoring_time | 2 | 0.03% |
| record | controversy.victim_actresses.2.actress_name | 2 | 0.03% |
| record | controversy.victim_actresses.4.krk_comment | 2 | 0.03% |
| record | date: Date when the event takes place | 2 | 0.03% |
| record | description: Brief summary of the event's purpose or activities | 2 | 0.03% |
| record | previous_incident.person | 2 | 0.03% |
| record | source.platform | 2 | 0.03% |
| record | asteroid.named_after.nationality | 2 | 0.03% |
| record | mrn: Unique medical record number | 2 | 0.03% |
| record | usage | 2 | 0.03% |
| record | publication_time | 2 | 0.03% |
| record | commercial_rollout | 2 | 0.03% |
| record | pitch_focus | 2 | 0.03% |
| record | donation_id: Unique identifier for the donation in the batch | 2 | 0.03% |
| record | product.developers.1.name | 2 | 0.03% |
| record | vehicle.description | 2 | 0.03% |
| record | vehicle.specifications.obstacle_crossing | 2 | 0.03% |
| record | research.experiment_details.laser_details | 2 | 0.03% |
| record | training.recommended | 2 | 0.03% |
| record | training.age_requirement | 2 | 0.03% |
| record | drivers.0.name | 2 | 0.03% |
| record | team.name | 2 | 0.03% |
| record | team.vehicle.model | 2 | 0.03% |
| record | abstract: Brief summary of the invention's technical approach and purpose | 2 | 0.03% |
| record | filing_fee: Amount of filing fee paid in local currency | 2 | 0.03% |
| record | discoverers | 2 | 0.03% |
| record | orbital_characteristics.reference_plane | 2 | 0.03% |
| record | team_mentions.0.analysis | 2 | 0.03% |
| record | score: Percentage score achieved in the exam | 2 | 0.03% |
| record | siblings_in_law.3.name | 2 | 0.03% |
| record | siblings_in_law.3.relationship | 2 | 0.03% |
| record | marriage_duration | 2 | 0.03% |
| record | nephews | 2 | 0.03% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.0.rate_percentage | 2 | 0.03% |
| record | statistical_results.diagnostic_metrics.annual_fracture_rates.1.t_score | 2 | 0.03% |
| record | collection_site | 2 | 0.03% |
| record | appointment_id: Unique identifier for the appointment | 2 | 0.03% |
| record | client_email | 2 | 0.03% |
| record | Dariusz Śledź <> born on | 2 | 0.03% |
| record | status: Current stage of development or deployment plan | 2 | 0.03% |
| record | age: Patient's age at time of visit | 2 | 0.03% |
| record | visit_type: Type of medical visit based on urgency | 2 | 0.03% |
| record | name: Full product name as listed in the store catalog | 2 | 0.03% |
| record | address: Full street address of the policyholder | 2 | 0.03% |
| record | damage_type: Specific parts damaged in the incident | 2 | 0.03% |
| record | patient_id: Unique identifier for the patient entry | 2 | 0.03% |
| record | score: Rating given on a scale from 0 to 5 | 2 | 0.03% |
| record | details.uniform.robe_color | 2 | 0.03% |
| record | journals | 2 | 0.03% |
| record | methodology.experimental_setup.biological_model.cell_line | 2 | 0.03% |
| record | advertiser_information.contact_information.additional_information | 2 | 0.03% |
| record | claim_number: Unique identifier for the health claim | 2 | 0.03% |
| record | arguments | 2 | 0.03% |
| record | representatives.1.name | 2 | 0.03% |
| record | vehicle trim level | 2 | 0.03% |
| record | payment method | 2 | 0.03% |
| record | distribution.regions.2.countries | 2 | 0.03% |
| record | report_id: Unique identifier for the risk assessment report | 2 | 0.03% |
| record | patient_mrn: Unique medical record number identifying the patient | 2 | 0.03% |
| record | trader | 2 | 0.03% |
| record | genus.number_of_species.min | 2 | 0.03% |
| record | submission_id: Unique identifier assigned to the abstract submission | 2 | 0.03% |
| record | review.pros | 2 | 0.03% |
| record | review.price | 2 | 0.03% |
| record | cost: Total cost of the enrollment program | 2 | 0.03% |
| record | purchase_source | 2 | 0.03% |
| record | assigned_to: Name of healthcare professional responsible for device use | 2 | 0.03% |
| record | scholarship_name | 2 | 0.03% |
| record | place | 2 | 0.03% |
| record | preserved_specimens | 2 | 0.03% |
| record | award_name: Title of the scholarship or grant awarded | 2 | 0.03% |
| record | business_sustainability.product_lifecycle_analysis | 2 | 0.03% |
| record | included_access | 2 | 0.03% |
| record | ship_id | 2 | 0.03% |
| record | navy | 2 | 0.03% |
| record | album.versions.0.contents | 2 | 0.03% |
| record | album.sources.2.url | 2 | 0.03% |
| record | album.tracklist.cd.6.lyrics | 2 | 0.03% |
| record | album.tracklist.cd.10.lyrics | 2 | 0.03% |
| record | album.external_links.3.description | 2 | 0.03% |
| record | co_stars | 2 | 0.03% |
| record | related_reading | 2 | 0.03% |
| record | contact | 2 | 0.03% |
| record | discovery.discovery_team | 2 | 0.03% |
| record | discovery.telescopes.0.location | 2 | 0.03% |
| record | terminal.impact.employee_reduction | 2 | 0.03% |
| record | terminal.history.3.period | 2 | 0.03% |
| record | auxpass | 2 | 0.03% |
| record | status: Current state of the rate entry | 2 | 0.03% |
| record | location: Physical address where the issue occurred | 2 | 0.03% |
| record | first_aid: Instructions for immediate response to exposure | 2 | 0.03% |
| record | municipality.location.country | 2 | 0.03% |
| record | line2 | 2 | 0.03% |
| record | contract_length: Duration of contract in years | 2 | 0.03% |
| record | review_link | 2 | 0.03% |
| record | filing_location: County or jurisdiction where the case was filed | 2 | 0.03% |
| record | tickets | 2 | 0.03% |
| record | patient_city | 2 | 0.03% |
| record | hba1c_value | 2 | 0.03% |
| record | venue_code | 2 | 0.03% |
| record | open_price | 2 | 0.03% |
| record | features_removed | 2 | 0.03% |
| record | asteroid.description.discovery_details.observatory | 2 | 0.03% |
| record | distribution.regions.0.countries | 2 | 0.03% |
| record | target_currency | 2 | 0.03% |
| record | address_unit | 2 | 0.03% |
| record | cuisine_style | 2 | 0.03% |
| record | physician.address.state | 2 | 0.03% |
| record | revenue_uptick | 2 | 0.03% |
| record | legislative_details.law_number | 2 | 0.03% |
| record | outcomes.monitored_metrics | 2 | 0.03% |
| record | project.operational_model.key_roles | 2 | 0.03% |
| record | project.operational_model.stakeholders.2.entity_type | 2 | 0.03% |
| record | filing_entity | 2 | 0.03% |
| record | target: Target audience or department for the proposal | 2 | 0.03% |
| record | warehouse_id: unique identifier for the storage facility | 2 | 0.03% |
| record | backup_plan | 2 | 0.03% |
| record | final_registration_cutoff | 2 | 0.03% |
| record | course_id: Unique identifier for the course as assigned by the university | 2 | 0.03% |
| record | location: Physical classroom location including building and room | 2 | 0.03% |
| record | description.fruit.color | 2 | 0.03% |
| record | description.inflorescence.height_m.max | 2 | 0.03% |
| record | description.leaves.length_cm.max | 2 | 0.03% |
| record | systematics_and_distribution.group | 2 | 0.03% |
| record | lastinspectiondate | 2 | 0.03% |
| record | studyTitle | 2 | 0.03% |
| record | medical_terms | 2 | 0.03% |
| record | lease_duration | 2 | 0.03% |
| record | person.location_at_time | 2 | 0.03% |
| record | pet.species | 2 | 0.03% |
| record | description.leaves.length_cm.min | 2 | 0.03% |
| record | hardware_support.gpu.nvidia.version | 2 | 0.03% |
| record | previous_incident.source | 2 | 0.03% |
| record | address_state: State where patient resides | 2 | 0.03% |
| record | location: City and state of the recipient's school | 2 | 0.03% |
| record | vendor: Name of the supplying company | 2 | 0.03% |
| record | review.persons.0.role | 2 | 0.03% |
| record | review.performance.performers.0.name | 2 | 0.03% |
| record | department: Department where the user works | 2 | 0.03% |
| record | owner_name | 2 | 0.03% |
| record | support_provided | 2 | 0.03% |
| record | message_id: Unique identifier for each outreach message | 2 | 0.03% |
| record | insurance_provider: Name of the insurance company covering the claim | 2 | 0.03% |
| record | work_type: Type of work arrangement for the role | 2 | 0.03% |
| record | reporter: Name of the engineer or team member who filed the ticket | 2 | 0.03% |
| record | intendedAudience | 2 | 0.03% |
| record | place.location.district | 2 | 0.03% |
| record | resolution: Action taken to resolve or mitigate the incident | 2 | 0.03% |
| record | -RRB- | 2 | 0.03% |
| record | subcode | 2 | 0.03% |
| record | functional_role | 2 | 0.03% |
| record | patient_id: Unique identifier assigned to the vital signs recording | 2 | 0.03% |
| record | residence: Street address where the patient resides | 2 | 0.03% |
| record | blood_pressure: Blood pressure in format systolic/diastolic mmHg | 2 | 0.03% |
| record | prize_pool | 2 | 0.03% |
| record | ticket_id: Unique identifier for each ticket as per prefix format | 2 | 0.03% |
| record | ticket_id: Unique identifier for each ticket in the queue | 2 | 0.03% |
| record | airline_code: Two-letter code representing the operating airline | 2 | 0.03% |
| record | udi | 2 | 0.03% |
| record | lot_size: Size of the property in acres | 2 | 0.03% |
| record | grilling_instructions.method | 2 | 0.03% |
| record | player id | 2 | 0.03% |
| record | name: Full product name as referenced in the passage | 2 | 0.03% |
| record | hours_of_operation.5.open_time | 2 | 0.03% |
| record | transaction_type: Type of trade such as buy, sell, swap, withdraw, deposit | 2 | 0.03% |
| record | user_id: Pseudonymized user identifier from the platform | 2 | 0.03% |
| record | publication.genusAssignment.year | 2 | 0.03% |
| record | person.us_senator.state | 2 | 0.03% |
| record | deal_date | 2 | 0.03% |
| record | integration_lead | 2 | 0.03% |
| record | diseases_treated.exceptions | 2 | 0.03% |
| record | payee: Name of the entity receiving or sending funds | 2 | 0.03% |
| record | glucose_level | 2 | 0.03% |
| record | adjuster: Name of the assigned claims adjuster | 2 | 0.03% |
| record | completion_date | 2 | 0.03% |
| record | job_type | 2 | 0.03% |
| record | paradox.corrected_calculation | 2 | 0.03% |
| record | office: Regional office that generated the snapshot | 2 | 0.03% |
| record | study_period | 2 | 0.03% |
| record | likes: Number of likes the post received | 2 | 0.03% |
| record | assigned_judge: Name of the judge overseeing the case | 2 | 0.03% |
| record | lead_analyst | 2 | 0.03% |
| record | engagement_type: Engagement level based on upvotes or visibility | 2 | 0.03% |
| record | permit_holder | 2 | 0.03% |
| record | key_protein | 2 | 0.03% |
| record | isps | 2 | 0.03% |
| record | renewal_status | 2 | 0.03% |
| record | speaker.name | 2 | 0.03% |
| record | ranking.total | 2 | 0.03% |
| record | booking_options.assurance_message | 2 | 0.03% |
| record | rate_value: Mid-market exchange rate value as a numeric figure | 2 | 0.03% |
| record | other_municipalities | 2 | 0.03% |
| record | condition: Medical condition or procedure described | 2 | 0.03% |
| record | status: Current status of the claim approval | 2 | 0.03% |
| record | issue_type: Category of the reported problem | 2 | 0.03% |
| record | thread_id: Unique identifier for the support ticket | 2 | 0.03% |
| record | release_details.next_release.title | 2 | 0.03% |
| record | release_details.album.track_count | 2 | 0.03% |
| record | integration_timeline | 2 | 0.03% |
| record | revenue_estimate | 2 | 0.03% |
| record | pricing_model | 2 | 0.03% |
| record | treatment_overview.treatment_type | 2 | 0.03% |
| record | treatment_examples.items.1.category | 2 | 0.03% |
| record | address: Full street address including city, state, and zip | 2 | 0.03% |
| record | description.leaf.shape | 2 | 0.03% |
| record | address: Patient's current residential address | 2 | 0.03% |
| record | comorbidities: List of additional health conditions present | 2 | 0.03% |
| record | past_surgeries: List of previous surgical procedures with dates | 2 | 0.03% |
| record | coverage_id: Unique identifier for the insurance coverage | 2 | 0.03% |
| record | lastfollowupvisitdate: Date of the most recent clinical visit | 2 | 0.03% |
| record | works.popularity | 2 | 0.03% |
| record | name.full_name | 2 | 0.03% |
| record | location: Venue name and city with state for the festival | 2 | 0.03% |
| record | registrations | 2 | 0.03% |
| record | holiday_look.dress.type | 2 | 0.03% |
| record | diagnosis_description | 2 | 0.03% |
| record | gymnosperm_diversity.endemic_species | 2 | 0.03% |
| record | citations.2.description | 2 | 0.03% |
| record | submitter: Name of employee who filed the ticket | 2 | 0.03% |
| record | comparison.ideographic.example.languages.1.pronunciation | 2 | 0.03% |
| record | comparison.ideographic.historical_understanding | 2 | 0.03% |
| record | caveat.example.pronunciation | 2 | 0.03% |
| record | caveat.example.components.0.function | 2 | 0.03% |
| record | caveat.example.components.0.related_characters.2.character | 2 | 0.03% |
| record | caveat.example.components.1.meaning | 2 | 0.03% |
| record | citations.3.year | 2 | 0.03% |
| record | citations.0.author | 2 | 0.03% |
| record | gymnosperm_diversity.species_examples.3.name | 2 | 0.03% |
| record | patent_number: Unique identifier assigned by USPTO to the patent | 2 | 0.03% |
| record | statue.public_display.reopening_date | 2 | 0.03% |
| record | dosage_form: Physical form of the drug such as tablet or injection | 2 | 0.03% |
| record | approved_date: Date the drug received regulatory approval | 2 | 0.03% |
| record | publication_date.year | 2 | 0.03% |
| record | amino acids | 2 | 0.03% |
| record | reviewer_email: Email address of the reviewer | 2 | 0.03% |
| record | restaurant_name: Name of the dining establishment reviewed | 2 | 0.03% |
| record | comments_count: Number of comments on the published review | 2 | 0.03% |
| record | device_id: Unique identifier assigned to the product | 2 | 0.03% |
| record | ipc_code: International Patent Classification code for the invention | 2 | 0.03% |
| record | feedback | 2 | 0.03% |
| record | report_time | 2 | 0.03% |
| record | product_id: Unique identifier for the reviewed tech product | 2 | 0.03% |
| record | customer: Name of the product owner submitting the request | 2 | 0.03% |
| record | thread_id: Unique identifier for the support thread | 2 | 0.03% |
| record | release_details.album.availability.0.platform | 2 | 0.03% |
| record | magazine.article.page_breakdown.0.advertising_percentage | 2 | 0.03% |
| record | magazine.article.page_breakdown.2.advertising_percentage | 2 | 0.03% |
| record | magazine.article.page_breakdown.4.page_number | 2 | 0.03% |
| record | magazine.article.page_breakdown.6.page_number | 2 | 0.03% |
| record | magazine.article.page_breakdown.7.page_number | 2 | 0.03% |
| record | magazine.article.page_breakdown.7.advertising_percentage | 2 | 0.03% |
| record | magazine.article.page_breakdown.14.advertising_percentage | 2 | 0.03% |
| record | magazine.article.page_breakdown.15.advertising_percentage | 2 | 0.03% |
| record | album.external_links.3.url | 2 | 0.03% |
| record | album.tracklist.cd.9.lyrics | 2 | 0.03% |
| record | album.tracklist.cd.4.track_number | 2 | 0.03% |
| record | album.tracklist.cd.0.tie_in.0.type | 2 | 0.03% |
| record | album.title | 2 | 0.03% |
| record | current_enrollment | 2 | 0.03% |
| record | case_id: Unique case identifier in format CV-YYYY-XXXX | 2 | 0.03% |
| record | patient_name: Full name of the prescribed patient | 2 | 0.03% |
| record | score: Match score in format like 6-4, 7-5 or 3-6, 6-3, 7-5 | 2 | 0.03% |
| record | address: Full residential or commercial address of the customer | 2 | 0.03% |
| record | related_activities | 2 | 0.03% |
| record | review.performance.elements | 2 | 0.03% |
| record | review.performance.venue.location | 2 | 0.03% |
| record | company: Name of the technology company raising funds | 2 | 0.03% |
| record | expert_opinion.speaker.affiliation.parent_institution | 2 | 0.03% |
| record | integration_delay_days | 2 | 0.03% |
| record | company.financial_results.metrics.0.metric_name | 2 | 0.03% |
| record | visit_date: Date of the patient visit | 2 | 0.03% |
| record | hotel_details.year_renovated | 2 | 0.03% |
| record | hotel_details.check_in | 2 | 0.03% |
| record | child_policy.children.age_range | 2 | 0.03% |
| record | student_name: Name of the student who took the exam | 2 | 0.03% |
| record | document_version | 2 | 0.03% |
| record | related_amount: Financial amount involved if applicable | 2 | 0.03% |
| record | outreach_code: Code associated with the outreach event | 2 | 0.03% |
| record | funding_date: Date when the funding round was completed or announced | 2 | 0.03% |
| record | launch_year: Year the company launched its platform | 2 | 0.03% |
| record | affiliation.0.name | 2 | 0.03% |
| record | tenant: Name of the company occupying the leased space | 2 | 0.03% |
| record | location: Full address of the leased property | 2 | 0.03% |
| record | booking_information.extra_charges | 2 | 0.03% |
| record | away_team: Team playing at the away venue | 2 | 0.03% |
| record | competition: League or cup type such as Premier League or friendly | 2 | 0.03% |
| record | location: City and state where employee is based | 2 | 0.03% |
| record | primate.related_primates.0.genus | 2 | 0.03% |
| record | primate.species | 2 | 0.03% |
| record | size_and_fit.measurements.1.type | 2 | 0.03% |
| record | tournament_type | 2 | 0.03% |
| record | key_figures | 2 | 0.03% |
| record | vehicle.specifications.engine | 2 | 0.03% |
| record | vehicle.specifications.torque | 2 | 0.03% |
| record | vehicle.specifications.range | 2 | 0.03% |
| record | vehicle.operators.0.details | 2 | 0.03% |
| record | vehicle.operators.0.acquisition.organization | 2 | 0.03% |
| record | edit_information.editor | 2 | 0.03% |
| record | document_date | 2 | 0.03% |
| record | subscriber_name: Full name of the subscriber | 2 | 0.03% |
| record | address: Street address of the subscriber | 2 | 0.03% |
| record | authors: List of researchers who submitted the paper | 2 | 0.03% |
| record | title: Title of the research paper | 2 | 0.03% |
| record | author: Name of the medical professional who authored the guideline | 2 | 0.03% |
| record | outcomes.preliminary_findings | 2 | 0.03% |
| record | project.operational_model.stakeholders.1.entity_type | 2 | 0.03% |
| record | project.financials.funding_source | 2 | 0.03% |
| record | format: Format in which results were published | 2 | 0.03% |
| record | comments: Total number of comments on the post | 2 | 0.03% |
| record | statement_id | 2 | 0.03% |
| record | warehouse: Warehouse location where product is stored | 2 | 0.03% |
| record | radiologist_name | 2 | 0.03% |
| record | price: Listed price in dollars | 2 | 0.03% |
| record | features: List of unit amenities such as balcony, kitchen, patio | 2 | 0.03% |
| record | company: Name of the technology startup | 2 | 0.03% |
| record | primary_provider: Name of the attending physician | 2 | 0.03% |
| record | other_youth_programs_with_congressional_charters | 2 | 0.03% |
| record | country: Country where the recipient resides | 2 | 0.03% |
| record | lighting | 2 | 0.03% |
| record | acquisition_value | 2 | 0.03% |
| record | discovery_details.discoverer | 2 | 0.03% |
| record | physical_characteristics.observations.0.publication_details | 2 | 0.03% |
| record | registration_link | 2 | 0.03% |
| record | partner_employees: Number of employees in the partner company | 2 | 0.03% |
| record | lease_start | 2 | 0.03% |
| record | github_repo | 2 | 0.03% |
| record | city: Metropolitan area where the property is located | 2 | 0.03% |
| record | store_name | 2 | 0.03% |
| record | genus.family.name | 2 | 0.03% |
| record | hotel_id | 2 | 0.03% |
| record | pressnoteid | 2 | 0.03% |
| record | location: City and state where company is headquartered | 2 | 0.03% |
| record | uploader | 2 | 0.03% |
| record | new_website_url | 2 | 0.03% |
| record | achievements | 2 | 0.03% |
| record | tool_name | 2 | 0.03% |
| record | registration.data_required | 2 | 0.03% |
| record | policyholder_name: Full name of the policyholder | 2 | 0.03% |
| record | test_type: Type of laboratory test performed | 2 | 0.03% |
| record | asteroid.orbital_characteristics.reference_plane | 2 | 0.03% |
| record | template_substitution.examples | 2 | 0.03% |
| record | organization.founder | 2 | 0.03% |
| record | dish_name | 2 | 0.03% |
| record | location: Office address of the institution | 2 | 0.03% |
| record | result: Medal awarded to the athlete | 2 | 0.03% |
| record | citations.1.description | 2 | 0.03% |
| record | special_features: List of unique space features or amenities | 2 | 0.03% |
| record | numbered_designation | 2 | 0.03% |
| record | returner | 2 | 0.03% |
| record | fork_of | 2 | 0.03% |
| record | medical procedure | 2 | 0.03% |
| record | hotel_chain | 2 | 0.03% |
| record | praise | 2 | 0.03% |
| record | status: Current operational or strategic status of the company | 2 | 0.03% |
| record | primate.similarities | 2 | 0.03% |
| record | invoice_id: Unique identifier for the invoice as assigned by vendor | 2 | 0.03% |
| record | sexual_act.thrusting.cervix_transformation.descent | 2 | 0.03% |
| record | sexual_act.sperm_migration.trunk_analogy_again | 2 | 0.03% |
| record | sexual_act.sperm_migration.sperm_finger_formation | 2 | 0.03% |
| record | sexual_act.cervix_retraction.shape_reversion | 2 | 0.03% |
| record | event_purpose | 2 | 0.03% |
| record | rescheduled_event_date | 2 | 0.03% |
| record | publication.language.region | 2 | 0.03% |
| record | course_id: Unique identifier for the course | 2 | 0.03% |
| record | title: Name of the course such as Introduction to English Composition | 2 | 0.03% |
| record | instructor: Name of the course instructor | 2 | 0.03% |
| record | admission_date | 2 | 0.03% |
| record | notice_id | 2 | 0.03% |
| record | patient_address: Full address including street, apt, city, state, zip | 2 | 0.03% |
| record | status: Current processing status of the claim | 2 | 0.03% |
| record | co_payment: Patient's required co-payment amount | 2 | 0.03% |
| record | location: Physical location of the testing facility | 2 | 0.03% |
| record | return_address | 2 | 0.03% |
| record | pathophysiology.mechanism.secondary_event.description | 2 | 0.03% |
| record | clinical_analysis_scope | 2 | 0.03% |
| record | client | 2 | 0.03% |
| record | underwriting_status | 2 | 0.03% |
| record | node_count | 2 | 0.03% |
| record | brief_id | 2 | 0.03% |
| record | holiday_look.collar.material | 2 | 0.03% |
| record | boulder.location.relative_position | 2 | 0.03% |
| record | research.researcher.affiliation.abbreviation | 2 | 0.03% |
| record | frequency: How often medication is taken, e.g., daily, as needed, twice daily | 2 | 0.03% |
| record | statement_id: Unique identifier for the press conference statement | 2 | 0.03% |
| record | location: City and state where the event occurred | 2 | 0.03% |
| record | research.experiment_details.interference_pattern | 2 | 0.03% |
| record | optical property | 2 | 0.03% |
| record | device_id: Unique identifier for the medical device | 2 | 0.03% |
| record | device_name: Name of the medical device model | 2 | 0.03% |
| record | view_count: Total number of views the video has received | 2 | 0.03% |
| record | name: Full name of the LinkedIn product | 2 | 0.03% |
| record | official_status | 2 | 0.03% |
| record | losses | 2 | 0.03% |
| record | publication_magazine | 2 | 0.03% |
| record | clinical_status: Current clinical condition such as normal or elevated | 2 | 0.03% |
| record | removal_instructions.browser_specific_instructions.0.browser | 2 | 0.03% |
| record | tax_rate | 2 | 0.03% |
| record | person.experience.1.start_year | 2 | 0.03% |
| record | person.publications.0.language | 2 | 0.03% |
| record | historical_context.time_periods.3.events.0.description | 2 | 0.03% |
| record | origin: Starting address of the shipment | 2 | 0.03% |
| record | hotel.location.address | 2 | 0.03% |
| record | hotel.transportation.metro.station | 2 | 0.03% |
| record | hotel.transportation.parking.1.cost | 2 | 0.03% |
| record | amount: Total cost of the fuel refill | 2 | 0.03% |
| record | city: City where the transaction took place | 2 | 0.03% |
| record | state: Two-letter state abbreviation | 2 | 0.03% |
| record | reference_id: Unique reference code for the transaction | 2 | 0.03% |
| record | described_by.author | 2 | 0.03% |
| record | document_details.authors.0.name | 2 | 0.03% |
| record | result_type | 2 | 0.03% |
| record | nurse_manager | 2 | 0.03% |
| record | location: Geographic location of the project | 2 | 0.03% |
| record | price: Total price of the proposed solution | 2 | 0.03% |
| record | Steinem <> birth date | 2 | 0.03% |
| record | clinic | 2 | 0.03% |
| record | docket_number | 2 | 0.03% |
| record | organizer_name: Name of the person or group organizing the event | 2 | 0.03% |
| record | attendee_count: Number of registered attendees | 2 | 0.03% |
| record | company_name: Name of the company in the record | 2 | 0.03% |
| record | person.biography.marriage.spouse_father_occupation | 2 | 0.03% |
| record | certification.validity | 2 | 0.03% |
| record | certification.technologies | 2 | 0.03% |
| record | hba1c | 2 | 0.03% |
| record | release_details.album.availability.1.access_type | 2 | 0.03% |
| record | press_release.location | 2 | 0.03% |
| record | company_name: Name of the startup company presenting | 2 | 0.03% |
| record | methodology.sample | 2 | 0.03% |
| record | item_description.name | 2 | 0.03% |
| record | item_description.weight.unit | 2 | 0.03% |
| record | claim_number: Identifier for the specific health claim submitted | 2 | 0.03% |
| record | disease | 2 | 0.03% |
| record | link: URL to the product launch page | 2 | 0.03% |
| record | miner_name | 2 | 0.03% |
| record | group | 2 | 0.03% |
| record | announcement.award | 2 | 0.03% |
| record | announcement.divisions.0.star_of_the_week.position | 2 | 0.03% |
| record | announcement.divisions.0.star_of_the_week.stats | 2 | 0.03% |
| record | announcement.divisions.0.second_star.team | 2 | 0.03% |
| record | announcement.divisions.0.honorable_mentions.1.position | 2 | 0.03% |
| record | announcement.divisions.0.honorable_mentions.2.position | 2 | 0.03% |
| record | announcement.divisions.0.honorable_mentions.3.team | 2 | 0.03% |
| record | announcement.divisions.0.honorable_mentions.4.team | 2 | 0.03% |
| record | announcement.divisions.0.honorable_mentions.5.player_name | 2 | 0.03% |
| record | announcement.divisions.0.honorable_mentions.6.team | 2 | 0.03% |
| record | announcement.divisions.0.honorable_mentions.7.player_name | 2 | 0.03% |
| record | announcement.divisions.1.second_star.team | 2 | 0.03% |
| record | announcement.divisions.1.honorable_mentions.1.position | 2 | 0.03% |
| record | announcement.divisions.2.star_of_the_week.position | 2 | 0.03% |
| record | announcement.divisions.2.honorable_mentions.1.player_name | 2 | 0.03% |
| record | announcement.divisions.2.honorable_mentions.3.position | 2 | 0.03% |
| record | announcement.divisions.2.honorable_mentions.3.team | 2 | 0.03% |
| record | announcement.divisions.2.honorable_mentions.4.position | 2 | 0.03% |
| record | announcement.divisions.3.star_of_the_week.team | 2 | 0.03% |
| record | announcement.divisions.3.second_star.player_name | 2 | 0.03% |
| record | announcement.divisions.3.honorable_mentions.0.position | 2 | 0.03% |
| record | announcement.divisions.3.honorable_mentions.0.team | 2 | 0.03% |
| record | announcement.divisions.4.star_of_the_week.position | 2 | 0.03% |
| record | announcement.divisions.4.second_star.player_name | 2 | 0.03% |
| record | announcement.divisions.4.second_star.position | 2 | 0.03% |
| record | announcement.divisions.4.honorable_mentions.1.team | 2 | 0.03% |
| record | announcement.divisions.4.honorable_mentions.2.team | 2 | 0.03% |
| record | announcement.divisions.5.second_star.position | 2 | 0.03% |
| record | announcement.divisions.5.honorable_mentions.0.position | 2 | 0.03% |
| record | announcement.divisions.5.honorable_mentions.1.player_name | 2 | 0.03% |
| record | penalty | 2 | 0.03% |
| record | location: Anatomical region examined | 2 | 0.03% |
| record | name: Name of the product as introduced | 2 | 0.03% |
| record | patient_population | 2 | 0.03% |
| record | thumbnail: Description of the video thumbnail design and colors | 2 | 0.03% |
| record | rate: Exchange rate value from base to USD | 2 | 0.03% |
| record | customer | 2 | 0.03% |
| record | deal_id | 2 | 0.03% |
| record | case_report.final_diagnosis | 2 | 0.03% |
| record | movie.comments.17.date | 2 | 0.03% |
| record | movie.comments.1.time | 2 | 0.03% |
| record | movie.comments.21.date | 2 | 0.03% |
| record | movie.comments.15.text | 2 | 0.03% |
| record | movie.comments.20.time | 2 | 0.03% |
| record | movie.comments.6.time | 2 | 0.03% |
| record | movie.comments.2.text | 2 | 0.03% |
| record | collaboration | 2 | 0.03% |
| record | hazard_classification | 2 | 0.03% |
| record | filer: Name of the person or entity filing the case | 2 | 0.03% |
| record | company.founders.0.name | 2 | 0.03% |
| record | statue.name.ukrainian | 2 | 0.03% |
| record | document_actions | 2 | 0.03% |
| record | outage_start | 2 | 0.03% |
| record | population_coverage | 2 | 0.03% |
| record | age: Age of the athlete at the time of the event | 2 | 0.03% |
| record | marginofvictory | 2 | 0.03% |
| record | asteroid.physical_characteristics.observations.0.telescope.year_launched | 2 | 0.03% |
| record | renewal_amount: Monetary amount charged for the renewal | 2 | 0.03% |
| record | purchase_location: Store where device was originally bought | 2 | 0.03% |
| record | restaurant.features | 2 | 0.03% |
| record | shared_rights | 2 | 0.03% |
| record | meeting_code: Unique code assigned to the briefing session | 2 | 0.03% |
| record | content.elements.music.bands | 2 | 0.03% |
| record | content.elements.cemeteries | 2 | 0.03% |
| record | temp | 2 | 0.03% |
| record | vehicle.model | 2 | 0.03% |
| record | potentialSolutions | 2 | 0.03% |
| record | incidence.rate | 2 | 0.03% |
| record | outreach_type | 2 | 0.03% |
| record | south_yemen_conflict.accusations_against_governments.0.accusations | 2 | 0.03% |
| record | south_yemen_conflict.justice_and_restitution | 2 | 0.03% |
| record | future_outlook | 2 | 0.03% |
| record | disease_profile.name | 2 | 0.03% |
| record | patient_case.clinical_presentation.skin_lesions.size_range.min_mm | 2 | 0.03% |
| record | disease.subspecies.2.affected_species | 2 | 0.03% |
| record | disease.pathogenesis.symptoms | 2 | 0.03% |
| record | route_name | 2 | 0.03% |
| record | billing_amount | 2 | 0.03% |
| record | department: Government department or agency where appointed | 2 | 0.03% |
| record | instruction | 2 | 0.03% |
| record | investor: Name of the investment firm conducting the meeting | 2 | 0.03% |
| record | event_type: Type of academic event registered for | 2 | 0.03% |
| record | clinicorhospital: Name of the clinic or hospital | 2 | 0.03% |
| record | policy_type: Type of insurance policy being renewed | 2 | 0.03% |
| record | biography.education.2.institution | 2 | 0.03% |
| record | biography.education.3.teacher | 2 | 0.03% |
| record | study.quotes | 2 | 0.03% |
| record | asteroid.named_after.location.city | 2 | 0.03% |
| record | product.preparation.peach_selection | 2 | 0.03% |
| record | server_hiding.resolution | 2 | 0.03% |
| record | key_components | 2 | 0.03% |
| record | number_of_offers | 2 | 0.03% |
| record | establishment_approval | 2 | 0.03% |
| record | age: Patient's age in years | 2 | 0.03% |
| record | lastreceivingdate | 2 | 0.03% |
| record | score: Final score of the game | 2 | 0.03% |
| record | title: Official name of the course | 2 | 0.03% |
| record | patient.age_estimation.correction | 2 | 0.03% |
| record | doctor.question | 2 | 0.03% |
| record | doctor.reaction | 2 | 0.03% |
| record | provider: Name of the medical professional or facility providing service | 2 | 0.03% |
| record | product_name: Name of the electronic device being reviewed | 2 | 0.03% |
| record | status: Current status of the policy announcement | 2 | 0.03% |
| record | hr | 2 | 0.03% |
| record | availability.locations | 2 | 0.03% |
| record | domain | 2 | 0.03% |
| record | vote_result | 2 | 0.03% |
| record | report_context.clinical_recommendations | 2 | 0.03% |
| record | points_leader | 2 | 0.03% |
| record | organization_info.facilities.transmission_lines | 2 | 0.03% |
| record | event.time.start | 2 | 0.03% |
| record | organization_info.size | 2 | 0.03% |
| record | resolution_cost | 2 | 0.03% |
| record | serial: Hardware serial number associated with the device | 2 | 0.03% |
| record | target_cfo | 2 | 0.03% |
| record | status: Current health status of the patient | 2 | 0.03% |
| record | server_name | 2 | 0.03% |
| record | agency.abbreviation | 2 | 0.03% |
| record | ready_campaign_updates.email | 2 | 0.03% |
| record | total_rainfall | 2 | 0.03% |
| record | DJB <> questioned | 2 | 0.03% |
| record | Robert <> acknowledged | 2 | 0.03% |
| record | Robert <> criticized | 2 | 0.03% |
| record | Robert <> valued | 2 | 0.03% |
| record | album.sources.0.access_date | 2 | 0.03% |
| record | album.sources.2.source_name | 2 | 0.03% |
| record | album.versions.2.type | 2 | 0.03% |
| record | album.external_links.0.description | 2 | 0.03% |
| record | album.tracklist.cd.3.lyrics | 2 | 0.03% |
| record | album.tracklist.cd.4.lyrics | 2 | 0.03% |
| record | winner_percentage: Percentage of votes won by the winner | 2 | 0.03% |
| record | boulder.start.hand_holds.0.hand | 2 | 0.03% |
| record | boulder.start.foot_holds.1.description | 2 | 0.03% |
| record | boulder.moves | 2 | 0.03% |
| record | route_id | 2 | 0.03% |
| record | movie.comments.3.time | 2 | 0.03% |
| record | movie.comments.4.text | 2 | 0.03% |
| record | movie.comments.6.text | 2 | 0.03% |
| record | movie.comments.6.date | 2 | 0.03% |
| record | movie.comments.7.time | 2 | 0.03% |
| record | movie.comments.7.date | 2 | 0.03% |
| record | movie.comments.13.text | 2 | 0.03% |
| record | movie.comments.17.time | 2 | 0.03% |
| record | movie.comments.20.date | 2 | 0.03% |
| record | sender_name: Name of the sender in the correspondence | 2 | 0.03% |
| record | asteroid.named_after.person.death_year | 2 | 0.03% |
| record | study.comparison_countries | 2 | 0.03% |
| record | fruit_properties.sugar_content | 2 | 0.03% |
| record | fruit_properties.skin.hairiness | 2 | 0.03% |
| record | root_cause: Identified reason for the service disruption | 2 | 0.03% |
| record | town.name_spanish | 2 | 0.03% |
| record | town.province | 2 | 0.03% |
| record | companies_mentioned | 2 | 0.03% |
| record | location.altitude.unit | 2 | 0.03% |
| record | patient_name: Name of the patient associated with the claim | 2 | 0.03% |
| record | prize_pool: Total prize money in US dollars | 2 | 0.03% |
| record | morphology.wingLengthMm | 2 | 0.03% |
| record | key_achievement: Major accomplishment noted in the review | 2 | 0.03% |
| record | creator: Name of the video creator | 2 | 0.03% |
| record | email: Email address of the contact | 2 | 0.03% |
| record | ref_id: Unique reference identifier for the proposal | 2 | 0.03% |
| record | patient_count: Total number of patients enrolled in the trial | 2 | 0.03% |
| record | duration: Duration of treatment in days and frequency | 2 | 0.03% |
| record | city: City where the patient resides | 2 | 0.03% |
| record | followupdoctor: Name of the doctor for follow-up care | 2 | 0.03% |
| record | distributor.products_distributed | 2 | 0.03% |
| record | availability.locations.online_stores.0.location | 2 | 0.03% |
| record | tournament_code: Unique identifier for the match event | 2 | 0.03% |
| record | company_name: Name of the startup founded by the individual | 2 | 0.03% |
| record | investigation.medical_examiner.name | 2 | 0.03% |
| record | relocation_date | 2 | 0.03% |
| record | case_report.patient.recent_medical_events.0.event_type | 2 | 0.03% |
| record | asteroid.named_after.location.country | 2 | 0.03% |
| record | use | 2 | 0.03% |
| record | honorary_name.nationality | 2 | 0.03% |
| record | splicing_regulation.current_research_status | 2 | 0.03% |
| record | result_rank | 2 | 0.03% |
| record | inventor: Name of the patent applicant or inventor | 2 | 0.03% |
| record | end_year: Year when data collection or study concluded | 2 | 0.03% |
| record | acomp | 2 | 0.03% |
| record | capital | 2 | 0.03% |
| record | perpetrator.age | 2 | 0.03% |
| record | crime.location.district | 2 | 0.03% |
| record | patient_case.genetic_analysis.gene_tested | 2 | 0.03% |
| record | item | 2 | 0.03% |
| record | case_report.diagnostics.trichoscopy.findings.1.location_on_lesion | 2 | 0.03% |
| record | disease_profile.condition_name | 2 | 0.03% |
| record | zip_code: ZIP code of the patient's residence | 2 | 0.03% |
| record | encounter_status | 2 | 0.03% |
| record | physiological_measures | 2 | 0.03% |
| record | programStructure.dailySchedule.0.activities.3.time | 2 | 0.03% |
| record | programStructure.dailySchedule.1.day | 2 | 0.03% |
| record | programStructure.dailySchedule.1.activities.1.time | 2 | 0.03% |
| record | evaluationCriteria | 2 | 0.03% |
| record | replies: Number of replies to the tweet | 2 | 0.03% |
| record | domain.tld | 2 | 0.03% |
| record | setting.time_period | 2 | 0.03% |
| record | participant_id: Unique employee identifier from enrollment data | 2 | 0.03% |
| record | loser: Player who lost the match | 2 | 0.03% |
| record | hotel.rating | 2 | 0.03% |
| record | outbreak_event.temporal_data.duration_context | 2 | 0.03% |
| record | outbreak_event.public_health_recommendations.action_required | 2 | 0.03% |
| record | taxonomy.describer.affiliation.0.location.country | 2 | 0.03% |
| record | bed_number | 2 | 0.03% |
| record | rbc: Red blood cell count in millions per microliter | 2 | 0.03% |
| record | athlete_id | 2 | 0.03% |
| record | risks_of_sun_exposure | 2 | 0.03% |
| record | sunless_tanning.overview | 2 | 0.03% |
| record | acquirer_company: Name of the company acquiring another firm | 2 | 0.03% |
| record | video_record | 2 | 0.03% |
| record | flight_search_instructions | 2 | 0.03% |
| record | paid_amount: Amount actually received from insurance | 2 | 0.03% |
| record | asteroid.etymology.deity.description | 2 | 0.03% |
| record | asteroid.characteristics.size | 2 | 0.03% |
| record | alternate_designations | 2 | 0.03% |
| record | meeting_code | 2 | 0.03% |
| record | rentalboardstatus: Status as reported on the city’s rental board | 2 | 0.03% |
| record | subordinated_to | 2 | 0.03% |
| record | office_hours.1.day | 2 | 0.03% |
| record | genre_focus | 2 | 0.03% |
| record | groups | 2 | 0.03% |
| record | shipment_type | 2 | 0.03% |
| record | species.etymology.named_after.0.name | 2 | 0.03% |
| record | description.length | 2 | 0.03% |
| record | taxonomy.describer.affiliation.0.department | 2 | 0.03% |
| record | distributor_information.authorization | 2 | 0.03% |
| record | process_details.methodology | 2 | 0.03% |
| record | application_areas | 2 | 0.03% |
| record | company: Name of the company where the recipient works | 2 | 0.03% |
| record | patient_name: Name of the individual submitting the claim | 2 | 0.03% |
| record | organization.company_name | 2 | 0.03% |
| record | locations: Cities where the product has been sold or used | 2 | 0.03% |
| record | contract_details.contract_type | 2 | 0.03% |
| record | agent_name | 2 | 0.03% |
| record | resolution_time: Hours between opening and closing of the ticket | 2 | 0.03% |
| record | VBP | 2 | 0.03% |
| record | radiologist | 2 | 0.03% |
| record | Dariusz Śledź <> won | 2 | 0.03% |
| record | species.family | 2 | 0.03% |
| record | production_location | 2 | 0.03% |
| record | price_close | 2 | 0.03% |
| record | damage_report | 2 | 0.03% |
| record | score: Numerical performance score from 3.0 to 4.1 | 2 | 0.03% |
| record | network_id: Unique identifier for each gene regulator network | 2 | 0.03% |
| record | tax_incentive.tax_rate_percent | 2 | 0.03% |
| record | primate.fossil_origin.location | 2 | 0.03% |
| record | primate.related_primates.0.temporal_context | 2 | 0.03% |
| record | primate.morphological_analysis.0.genus | 2 | 0.03% |
| record | sales_growth | 2 | 0.03% |
| record | source.domain | 2 | 0.03% |
| record | report_id: Unique identifier for the imaging report | 2 | 0.03% |
| record | verbs_present_participle | 2 | 0.03% |
| record | economy.agriculturalArea | 2 | 0.03% |
| record | hotel_experience.services | 2 | 0.03% |
| record | session_type: Type of session such as Keynote or Workshop | 2 | 0.03% |
| record | duration: Duration of the session in minutes as given in Duration column | 2 | 0.03% |
| record | department: Business unit or division the employee is assigned to | 2 | 0.03% |
| record | customer_name: Name of the ticket owner reporting the issue | 2 | 0.03% |
| record | resolution_date: Date when the issue was resolved or action taken | 2 | 0.03% |
| record | general_description | 2 | 0.03% |
| record | person1 | 2 | 0.03% |
| record | kadlub_shape | 2 | 0.03% |
| record | player: Name of the player who scored | 2 | 0.03% |
| record | description: Purpose or context of the contribution | 2 | 0.03% |
| record | Consumption | 2 | 0.03% |
| record | literary_movements | 2 | 0.03% |
| record | rental_property | 2 | 0.03% |
| record | revenue: Total revenue in millions | 2 | 0.03% |
| record | partner_company: Name of the partner company | 2 | 0.03% |
| record | partnergrowthrate: Partner's growth rate percentage | 2 | 0.03% |
| record | imaging_type: Type of medical imaging performed | 2 | 0.03% |
| record | employment_type | 2 | 0.03% |
| record | historical_context.time_periods.1.events.0.description | 2 | 0.03% |
| record | historical_context.time_periods.1.events.0.measures | 2 | 0.03% |
| record | historical_context.public_health_measures | 2 | 0.03% |
| record | dean.title | 2 | 0.03% |
| record | project_lead | 2 | 0.03% |
| record | studystartdate | 2 | 0.03% |
| record | transcription_factor | 2 | 0.03% |
| record | alliance | 2 | 0.03% |
| record | author: Name of the abstract's author | 2 | 0.03% |
| record | advertisement.text | 2 | 0.03% |
| record | individual_involved | 2 | 0.03% |
| record | service.responsible_person.affiliation | 2 | 0.03% |
| record | radiologist: Name of the physician who interpreted the report | 2 | 0.03% |
| record | account_handle: Instagram handle used for the launch post | 2 | 0.03% |
| record | damages: Monetary amount sought in damages | 2 | 0.03% |
| record | dose: Dosage administered to the patient | 2 | 0.03% |
| record | victim.relationship_to_perpetrator | 2 | 0.03% |
| record | hq: Headquarters city and state of the company | 2 | 0.03% |
| record | integration | 2 | 0.03% |
| record | quarterback | 2 | 0.03% |
| record | hotel.brand | 2 | 0.03% |
| record | article.publication_details.author | 2 | 0.03% |
| record | article.entities.organizations | 2 | 0.03% |
| record | article.entities.people.0.organization | 2 | 0.03% |
| record | associated_diseases: Diseases linked to pathway dysregulation | 2 | 0.03% |
| record | sector: Industry domain such as logistics, healthcare, fintech | 2 | 0.03% |
| record | main_themes | 2 | 0.03% |
| record | rulers | 2 | 0.03% |
| record | shares: Total number of shares the video received | 2 | 0.03% |
| record | publication_volume | 2 | 0.03% |
| record | registration_cap | 2 | 0.03% |
| record | panel_topics | 2 | 0.03% |
| record | camera_model | 2 | 0.03% |
| record | VBN | 2 | 0.03% |
| record | methodology.study_period.start_year | 2 | 0.03% |
| record | treatment_details.surgical_timeline.interval_brain_to_lung_resection_days.average | 2 | 0.03% |
| record | treatment_details.surgical_timeline.interval_brain_to_lung_resection_days.range_min | 2 | 0.03% |
| record | treatment_details.pulmonary_resection_types.1.procedure | 2 | 0.03% |
| record | pathological_findings.nodal_involvement_distribution.1.stage | 2 | 0.03% |
| record | outcomes.overall_survival_months.min | 2 | 0.03% |
| record | outcomes.mortality_by_nodal_involvement.1.location | 2 | 0.03% |
| record | track | 2 | 0.03% |
| record | session_type | 2 | 0.03% |
| record | education_com_content | 2 | 0.03% |
| record | affiliation.location | 2 | 0.03% |
| record | deprotection_process.additives | 2 | 0.03% |
| record | patient_care.weekly_plan | 2 | 0.03% |
| record | blood_pressure: Blood pressure in mmHg format as systolic/diastolic | 2 | 0.03% |
| record | customer: Name of the affected customer | 2 | 0.03% |
| record | organization.personnel_changes.0.effective_date_description | 2 | 0.03% |
| record | actor_name | 2 | 0.03% |
| record | album.tracklist.cd.2.lyrics | 2 | 0.03% |
| record | album.tracklist.cd.4.title | 2 | 0.03% |
| record | album.tracklist.cd.6.composer | 2 | 0.03% |
| record | album.tracklist.cd.6.notes | 2 | 0.03% |
| record | album.tracklist.dvd.0.track_number | 2 | 0.03% |
| record | album.sources.0.reference_number | 2 | 0.03% |
| record | credits.advertising_revenue_use | 2 | 0.03% |
| record | submission_id: Unique identifier assigned to the submission | 2 | 0.03% |
| record | product.origin.method_origin | 2 | 0.03% |
| record | call_details_contact | 2 | 0.03% |
| record | paradox.initial_payment_per_person | 2 | 0.03% |
| record | hours_of_operation.2.close_time | 2 | 0.03% |
| record | business_name | 2 | 0.03% |
| record | mutation.mechanism | 2 | 0.03% |
| record | discovery.function | 2 | 0.03% |
| record | reviewers | 2 | 0.03% |
| record | orbital_characteristics.orbital_period.years | 2 | 0.03% |
| record | impact: List of services affected including internet or voice | 2 | 0.03% |
| record | review.performance.conductor | 2 | 0.03% |
| record | pathway_id: Unique identifier for the biological pathway | 2 | 0.03% |
| record | passenger_location | 2 | 0.03% |
| record | accuracyorefficiency | 2 | 0.03% |
| record | donor_name: Full name of the contributor | 2 | 0.03% |
| record | prokeimenon.verse | 2 | 0.03% |
| record | prize: Prize money won by the winner in currency | 2 | 0.03% |
| record | classification | 2 | 0.03% |
| record | institution: University or institution where author is affiliated | 2 | 0.03% |
| record | organization-media/newspaper | 2 | 0.03% |
| record | broadcast_network | 2 | 0.03% |
| record | possession_rate | 2 | 0.03% |
| record | receiver_company | 2 | 0.03% |
| record | receiver_location | 2 | 0.03% |
| record | sport: Type of athletic competition such as sprint or rowing | 2 | 0.03% |
| record | product_price: Price of the product in USD | 2 | 0.03% |
| record | property_code | 2 | 0.03% |
| record | protecting_group.chemical_properties | 2 | 0.03% |
| record | protecting_group.targets.0.derivative_form | 2 | 0.03% |
| record | article.author.name | 2 | 0.03% |
| record | article.publicationDetails.year | 2 | 0.03% |
| record | orbit.orbital_period.value | 2 | 0.03% |
| record | product.translation | 2 | 0.03% |
| record | freetierusers | 2 | 0.03% |
| record | review_journal: Name of the publication that reviewed the work | 2 | 0.03% |
| record | assigned_technician: Name of the technician handling the ticket | 2 | 0.03% |
| record | total_cost | 2 | 0.03% |
| record | impact_homes | 2 | 0.03% |
| record | leasestartdate | 2 | 0.03% |
| record | incident.charges.0.description | 2 | 0.03% |
| record | author_information.interests | 2 | 0.03% |
| record | price: Closing stock price in dollars | 2 | 0.03% |
| record | employees: Number of employees | 2 | 0.03% |
| record | credits.version | 2 | 0.03% |
| record | provider_code: Service provider code assigned to the ticket | 2 | 0.03% |
| record | city: City where the customer resides | 2 | 0.03% |
| record | statue.school | 2 | 0.03% |
| record | statue.description | 2 | 0.03% |
| record | statue.historical_context | 2 | 0.03% |
| record | furniture | 2 | 0.03% |
| record | priority_level | 2 | 0.03% |
| record | utilities | 2 | 0.03% |
| record | platform: Type of social media post shared | 2 | 0.03% |
| record | related_verses | 2 | 0.03% |
| record | priority_registration.description | 2 | 0.03% |
| record | location: Physical ward or room where device is located | 2 | 0.03% |
| record | emergency_treatment | 2 | 0.03% |
| record | removal_instructions.browser_specific_instructions.1.browser | 2 | 0.03% |
| record | related_adware | 2 | 0.03% |
| record | risk_flags | 2 | 0.03% |
| record | product_id: Unique identifier for each product in the store | 2 | 0.03% |
| record | author_information.military_service.branch | 2 | 0.03% |
| record | author_information.family.spouse | 2 | 0.03% |
| record | article.author.familyName | 2 | 0.03% |
| record | article.publicationDetails.pages | 2 | 0.03% |
| record | article.keywords | 2 | 0.03% |
| record | sponsor: Individual legislator who introduced the bill | 2 | 0.03% |
| record | compensation_type | 2 | 0.03% |
| record | countries.0.name | 2 | 0.03% |
| record | report_details.publication_dates | 2 | 0.03% |
| record | rating: Numerical rating given to the work out of 5 | 2 | 0.03% |
| record | bracket_size | 2 | 0.03% |
| record | contact_information | 2 | 0.03% |
| record | actor | 2 | 0.03% |
| record | user_name: Name of the customer involved in the support request | 2 | 0.03% |
| record | personal_experience.raw_corn_experience | 2 | 0.03% |
| record | unit_type | 2 | 0.03% |
| record | monthly_fee: Monthly cost of the product subscription | 2 | 0.03% |
| record | chemical | 2 | 0.03% |
| record | service_amount | 2 | 0.03% |
| record | statue.location.country | 2 | 0.03% |
| record | statue.location.gallery | 2 | 0.03% |
| record | statue.restoration.0.funding.contribution_range.max | 2 | 0.03% |
| record | sender: Email of the person sending the confirmation or submission | 2 | 0.03% |
| record | results_range | 2 | 0.03% |
| record | state: State where the office is located | 2 | 0.03% |
| record | physical_characteristics.observations.0.telescope.launch_year | 2 | 0.03% |
| record | city.birthplace_of.0.person.name | 2 | 0.03% |
| record | user_id: Unique identifier for the user on TikTok | 2 | 0.03% |
| record | author.nationality | 2 | 0.03% |
| record | feedback_method | 2 | 0.03% |
| record | focus_topics | 2 | 0.03% |
| record | monthly_price | 2 | 0.03% |
| record | concurrentuserslimit | 2 | 0.03% |
| record | location: Physical address of the affected residence | 2 | 0.03% |
| record | permit_action | 2 | 0.03% |
| record | location.waterbody | 2 | 0.03% |
| record | case_report.diagnostics.trichoscopy.findings.1.feature | 2 | 0.03% |
| record | disease_profile.differential_diagnoses.0.disease_name | 2 | 0.03% |
| record | initial_text.actions | 2 | 0.03% |
| record | medication: Name and dosage of prescribed medication | 2 | 0.03% |
| record | name.given_name | 2 | 0.03% |
| record | planning | 2 | 0.03% |
| record | infrastructure.highways.1.connections | 2 | 0.03% |
| record | day | 2 | 0.03% |
| record | subjects | 2 | 0.03% |
| record | discoverer: Name of the researcher who identified the pathway | 2 | 0.03% |
| record | protecting_group.targets.1.abbreviation | 2 | 0.03% |
| record | synthesis_context.intermediates_synthesized.0.type | 2 | 0.03% |
| record | submission_id: Unique identifier for each submission record | 2 | 0.03% |
| record | document.intellectual_property_rights | 2 | 0.03% |
| record | deck_coating_systems.0.membrane_type | 2 | 0.03% |
| record | description.inflorescence.shape | 2 | 0.03% |
| record | description.growth_habit | 2 | 0.03% |
| record | description.leaves.width_cm.max | 2 | 0.03% |
| record | sports | 2 | 0.03% |
| record | weight_range_kg.max_kg | 2 | 0.03% |
| record | genus.number_of_species.max | 2 | 0.03% |
| record | genus.etymology.part1.meaning | 2 | 0.03% |
| record | location_neighborhood | 2 | 0.03% |
| record | storage_instructions | 2 | 0.03% |
| record | document.economic_impact.gdp_contribution | 2 | 0.03% |
| record | document.strategy_elements.2.area | 2 | 0.03% |
| record | controversy.victim_actresses.4.actress_name | 2 | 0.03% |
| record | geographical_scope | 2 | 0.03% |
| record | document.strategy_elements.4.area | 2 | 0.03% |
| record | reference_range: Normal range for the test value | 2 | 0.03% |
| record | patient_id: Unique identifier assigned to each patient | 2 | 0.03% |
| record | ability | 2 | 0.03% |
| record | clinical_observations.prevalence | 2 | 0.03% |
| record | cheat_tweet | 2 | 0.03% |
| record | funding_amount: Total capital raised in millions | 2 | 0.03% |
| record | adverb_modifier | 2 | 0.03% |
| record | enrollment_status | 2 | 0.03% |
| record | faculty_name | 2 | 0.03% |
| record | website_link | 2 | 0.03% |
| record | outage_start: Start time of the service disruption | 2 | 0.03% |
| record | zebras_care_foundation.beneficiary.interests | 1 | 0.01% |
| record | details | 1 | 0.01% |
| record | venues | 1 | 0.01% |
| record | status: Current legal status of the dispute | 1 | 0.01% |
| record | signing_role | 1 | 0.01% |
| record | renewal_option | 1 | 0.01% |
| record | study_metadata.measurement_technology.device_name | 1 | 0.01% |
| record | compliance_rate: Percentage of providers who followed the guideline in pilot programs | 1 | 0.01% |
| record | training_objective | 1 | 0.01% |
| record | employee_location | 1 | 0.01% |
| record | trend_repost_id | 1 | 0.01% |
| record | nutritional_aspects.sucrose.sources | 1 | 0.01% |
| record | news | 1 | 0.01% |
| record | host_venue | 1 | 0.01% |
| record | recipient: Email address of the victim or intended recipient | 1 | 0.01% |
| record | risk_level: Assessed risk level of the phishing attempt | 1 | 0.01% |
| record | definition.year_introduced | 1 | 0.01% |
| record | reviewer_email | 1 | 0.01% |
| record | reviewstatus | 1 | 0.01% |
| record | pet.name | 1 | 0.01% |
| record | cost: Monetary cost associated with resolving the ticket | 1 | 0.01% |
| record | medals.gold | 1 | 0.01% |
| record | visibility_status: Whether the comment was highlighted or pinned in the thread | 1 | 0.01% |
| record | geographic_reach | 1 | 0.01% |
| record | claimhistorycount | 1 | 0.01% |
| record | commonsideeffects: List of side effects commonly reported | 1 | 0.01% |
| record | magazine.article.page_breakdown.1.page_number | 1 | 0.01% |
| record | magazine.article.page_breakdown.11.advertisers | 1 | 0.01% |
| record | opponent_id: Unique ID assigned to the opponent | 1 | 0.01% |
| record | dob | 1 | 0.01% |
| record | launch_time | 1 | 0.01% |
| record | nextvideotitle | 1 | 0.01% |
| record | evacuation_orders | 1 | 0.01% |
| record | size_and_fit.measurements.0.unit | 1 | 0.01% |
| record | size_and_fit.measurements.1.value | 1 | 0.01% |
| record | vehicle.specifications.power | 1 | 0.01% |
| record | vehicle.operators.0.acquisition.recipient | 1 | 0.01% |
| record | status: Current status of the result publication | 1 | 0.01% |
| record | followupdate: Scheduled follow-up date | 1 | 0.01% |
| record | template_types.substitution_template | 1 | 0.01% |
| record | origin: Airport code where flight departs | 1 | 0.01% |
| record | sexual_act.thrusting.cervix_visibility | 1 | 0.01% |
| record | sexual_act.cervix_retraction.trunk_retraction | 1 | 0.01% |
| record | journalists <> surprised by | 1 | 0.01% |
| record | article.disclaimer | 1 | 0.01% |
| record | star.distance.parsecs | 1 | 0.01% |
| record | recipient: Name of the candidate recipient | 1 | 0.01% |
| record | tone: Descriptive tone of the message such as formal, friendly, urgent | 1 | 0.01% |
| record | peerfeedbackrating: Percentage of positive peer feedback received | 1 | 0.01% |
| record | physical_characteristics.diameter | 1 | 0.01% |
| record | person.experience.1.end_year | 1 | 0.01% |
| record | hotel.location.street | 1 | 0.01% |
| record | order_id: Unique order identifier from the retailer | 1 | 0.01% |
| record | announcement.divisions.2.second_star.team | 1 | 0.01% |
| record | announcement.divisions.4.star_of_the_week.player_name | 1 | 0.01% |
| record | announcement.divisions.4.honorable_mentions.0.team | 1 | 0.01% |
| record | article_title | 1 | 0.01% |
| record | asteroid.discovery.observatory.location.description | 1 | 0.01% |
| record | policyholder: Name of the insurance claimant | 1 | 0.01% |
| record | name: Full name of the patient involved | 1 | 0.01% |
| record | disease.pathogenesis.affected_organs | 1 | 0.01% |
| record | event.time | 1 | 0.01% |
| record | event.person.nationality | 1 | 0.01% |
| record | biography.awards.0.location | 1 | 0.01% |
| record | region.divisions.0.name | 1 | 0.01% |
| record | start | 1 | 0.01% |
| record | match_id: Unique identifier for the match event | 1 | 0.01% |
| record | signing_entities: List of countries or organizations involved | 1 | 0.01% |
| record | plaintiff: Name of the individual or entity initiating the case | 1 | 0.01% |
| record | outage_end | 1 | 0.01% |
| record | movie.comments.0.date | 1 | 0.01% |
| record | timeframe | 1 | 0.01% |
| record | asteroid.named_after.person.profession | 1 | 0.01% |
| record | valuation: Post-money valuation after the round | 1 | 0.01% |
| record | customer_affected | 1 | 0.01% |
| record | study_end | 1 | 0.01% |
| record | clinical_evidence | 1 | 0.01% |
| record | programStructure.dailySchedule.1.activities.1.description | 1 | 0.01% |
| record | publication.language.name | 1 | 0.01% |
| record | content_analysis.entry_structure.notes_types | 1 | 0.01% |
| record | donor_location | 1 | 0.01% |
| record | delegation.medals.silver | 1 | 0.01% |
| record | reviewer_location: Geographic location of the reviewer | 1 | 0.01% |
| record | comments: Number of verified user comments on the review | 1 | 0.01% |
| record | shooting | 1 | 0.01% |
| record | wbc: White blood cell count in thousands per microliter | 1 | 0.01% |
| record | additional_services | 1 | 0.01% |
| record | experience.description | 1 | 0.01% |
| record | product.price.0.currency | 1 | 0.01% |
| record | product.specifications.display.type | 1 | 0.01% |
| record | product.specifications.battery.charging_technology | 1 | 0.01% |
| record | location.state | 1 | 0.01% |
| record | species.etymology.named_after.0.reason | 1 | 0.01% |
| record | distribution.countries | 1 | 0.01% |
| record | taxonomy.describer.country | 1 | 0.01% |
| record | contact_email: Email address for contacting the exhibitor | 1 | 0.01% |
| record | sku: Unique product identifier assigned by the e-commerce platform | 1 | 0.01% |
| record | physical_exam | 1 | 0.01% |
| record | pipeline_companies | 1 | 0.01% |
| record | pathway: Primary biological pathway under investigation | 1 | 0.01% |
| record | registration_fee: Fee required for registration as stated in Registration Fee | 1 | 0.01% |
| record | study.period.start_year | 1 | 0.01% |
| record | actors | 1 | 0.01% |
| record | supplier: Company that supplies the product | 1 | 0.01% |
| record | article.copyright_details.copyright_request_period | 1 | 0.01% |
| record | loser | 1 | 0.01% |
| record | , | 1 | 0.01% |
| record | album.tracklist.cd.8.lyrics | 1 | 0.01% |
| record | contact.email | 1 | 0.01% |
| record | location: City where the job is based | 1 | 0.01% |
| record | paradox.waiter_pocketed | 1 | 0.01% |
| record | hours_of_operation.3.open_time | 1 | 0.01% |
| record | quote.author | 1 | 0.01% |
| record | monthly_rent | 1 | 0.01% |
| record | resource_name | 1 | 0.01% |
| record | server_hiding.description | 1 | 0.01% |
| record | person.biography.exile.reason | 1 | 0.01% |
| record | question.text | 1 | 0.01% |
| record | appointment_code: Internal code for the appointment decision | 1 | 0.01% |
| record | program_year | 1 | 0.01% |
| record | total_changes | 1 | 0.01% |
| record | technical_description | 1 | 0.01% |
| record | stockstatus: Current inventory condition of the product | 1 | 0.01% |
| record | discovery.age.geologicalStage | 1 | 0.01% |
| record | publication.genusAssignment.genus | 1 | 0.01% |
| record | duration_days | 1 | 0.01% |
| record | comments: Number of comments on the post | 1 | 0.01% |
| record | manufacturer: Company that produces the product | 1 | 0.01% |
| record | scholarship_id: Unique identifier assigned to each scholarship | 1 | 0.01% |
| record | location.nearby_area | 1 | 0.01% |
| record | severity: Numerical rating of symptom intensity | 1 | 0.01% |
| record | efficacy_rate: Percentage of patients showing therapeutic response | 1 | 0.01% |
| record | followupdate | 1 | 0.01% |
| record | account_name: Name of the account holder as listed on the statement | 1 | 0.01% |
| record | protecting_group.reactivity_notes | 1 | 0.01% |
| record | packaging_type | 1 | 0.01% |
| record | description.fruit.length_mm.max | 1 | 0.01% |
| record | customer_name: Name of the customer involved | 1 | 0.01% |
| record | priority: Priority level assigned to the case | 1 | 0.01% |
| record | case_report.diagnostics.trichoscopy.findings.2.location_on_lesion | 1 | 0.01% |
| record | product.developers.0.name | 1 | 0.01% |
| record | drive_types | 1 | 0.01% |
| record | narrative.setting.location | 1 | 0.01% |
| record | narrative.protagonist.motivation | 1 | 0.01% |
| record | media_analysis.historical_context.legacy | 1 | 0.01% |
| record | delivery_days | 1 | 0.01% |
| record | founding_year: Year the company was established | 1 | 0.01% |
| record | geography.reservoirs | 1 | 0.01% |
| record | infrastructure.highways.0.connections | 1 | 0.01% |
| record | infrastructure.highways.1.highwayName | 1 | 0.01% |
| record | economy.gdpYear | 1 | 0.01% |
| record | countries.2.breastfeeding_rate | 1 | 0.01% |
| record | primary_diagnosis: Clinical diagnosis code and description | 1 | 0.01% |
| record | author: Name of the person who wrote the review | 1 | 0.01% |
| record | product.perfumers | 1 | 0.01% |
| record | production_company | 1 | 0.01% |
| record | address: Street address including city and zip | 1 | 0.01% |
| record | physician: Name of the attending physician | 1 | 0.01% |
| record | acquired_location: Headquarters location of the acquired company | 1 | 0.01% |
| record | case_title | 1 | 0.01% |
| record | external_resources | 1 | 0.01% |
| record | genus.established_year | 1 | 0.01% |
| record | genus.european_species | 1 | 0.01% |
| record | training_program.payment.full_payment_amount | 1 | 0.01% |
| record | staff.personnel_count | 1 | 0.01% |
| record | record_metadata | natural/name ×175, natural/status ×174, natural/location ×131, natural/title ×126 |  |

**val.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| record | status | 150 | 17.38% |
| record | location | 101 | 11.70% |
| record | name | 71 | 8.23% |
| record | address | 51 | 5.91% |
| record | description | 47 | 5.45% |
| record | title | 44 | 5.10% |
| record | company | 32 | 3.71% |
| record | role | 20 | 2.32% |
| record | score | 19 | 2.20% |
| record | patient_name | 19 | 2.20% |
| record | institution | 18 | 2.09% |
| record | id | 18 | 2.09% |
| record | department | 18 | 2.09% |
| record | date | 17 | 1.97% |
| record | rating | 17 | 1.97% |
| record | product_name | 17 | 1.97% |
| record | price | 16 | 1.85% |
| record | asteroid.type | 16 | 1.85% |
| record | ticket_id | 14 | 1.62% |
| record | employee_id | 14 | 1.62% |
| record | outcome | 13 | 1.51% |
| record | patient_id | 13 | 1.51% |
| record | recipient | 13 | 1.51% |
| record | salary_range | 13 | 1.51% |
| record | quantity | 12 | 1.39% |
| record | location.state | 12 | 1.39% |
| record | source | 12 | 1.39% |
| record | funding_source | 12 | 1.39% |
| record | video_id | 11 | 1.27% |
| record | project_id | 11 | 1.27% |
| record | user | 11 | 1.27% |
| record | author | 10 | 1.16% |
| record | comments | 10 | 1.16% |
| record | amount | 10 | 1.16% |
| record | product_id | 10 | 1.16% |
| record | city | 10 | 1.16% |
| record | year | 9 | 1.04% |
| record | report_id | 9 | 1.04% |
| record | statement_id | 9 | 1.04% |
| record | drug_name | 9 | 1.04% |
| record | age | 9 | 1.04% |
| record | technology | 9 | 1.04% |
| record | sku | 8 | 0.93% |
| record | material | 8 | 0.93% |
| record | venue | 8 | 0.93% |
| record | home_team | 8 | 0.93% |
| record | concepts | 8 | 0.93% |
| record | creator | 8 | 0.93% |
| record | dispute_id | 8 | 0.93% |
| record | album_type | 8 | 0.93% |
| record | parties | 8 | 0.93% |
| record | start_date | 8 | 0.93% |
| record | end_date | 8 | 0.93% |
| record | asteroid.name | 8 | 0.93% |
| record | manufacturer | 8 | 0.93% |
| record | tags | 8 | 0.93% |
| record | employee_name | 8 | 0.93% |
| record | email | 8 | 0.93% |
| record | claim_id | 8 | 0.93% |
| record | asteroid.discovery.date | 8 | 0.93% |
| record | brand | 7 | 0.81% |
| record | referee | 7 | 0.81% |
| record | funding_amount | 7 | 0.81% |
| record | type | 7 | 0.81% |
| record | jurisdiction | 7 | 0.81% |
| record | position | 7 | 0.81% |
| record | action | 7 | 0.81% |
| record | student_id | 7 | 0.81% |
| record | pathway_id | 7 | 0.81% |
| record | notice_id | 7 | 0.81% |
| record | funding | 7 | 0.81% |
| record | away_team | 6 | 0.70% |
| record | reviewer | 6 | 0.70% |
| record | restaurant_name | 6 | 0.70% |
| record | opponent | 6 | 0.70% |
| record | medication | 6 | 0.70% |
| record | patient_age | 6 | 0.70% |
| record | patient_address | 6 | 0.70% |
| record | tail | 6 | 0.70% |
| record | head | 6 | 0.70% |
| record | duration | 6 | 0.70% |
| record | entity | 6 | 0.70% |
| record | gender | 6 | 0.70% |
| record | album_title | 6 | 0.70% |
| record | customer_name | 6 | 0.70% |
| record | asteroid.designation | 6 | 0.70% |
| record | salary | 6 | 0.70% |
| record | dosage_form | 6 | 0.70% |
| record | cost | 6 | 0.70% |
| record | brand_name | 6 | 0.70% |
| record | conclusions.summary | 6 | 0.70% |
| record | symbol | 6 | 0.70% |
| record | subscription_id | 6 | 0.70% |
| record | weight | 5 | 0.58% |
| record | match_type | 5 | 0.58% |
| record | final_minutes | 5 | 0.58% |
| record | interaction_score: Average strength of regulatory interactions | 5 | 0.58% |
| record | designation | 5 | 0.58% |
| record | q1revenueguidance | 5 | 0.58% |
| record | winner | 5 | 0.58% |
| record | investigator | 5 | 0.58% |
| record | ambiance | 5 | 0.58% |
| record | video_id: Unique identifier assigned to the video | 5 | 0.58% |
| record | disadvantages.health_issues | 5 | 0.58% |
| record | disadvantages.general | 5 | 0.58% |
| record | final_remarks | 5 | 0.58% |
| record | exchange_rate | 5 | 0.58% |
| record | severity | 5 | 0.58% |
| record | topic | 5 | 0.58% |
| record | materials | 5 | 0.58% |
| record | average_score | 5 | 0.58% |
| record | channel_name | 5 | 0.58% |
| record | person | 5 | 0.58% |
| record | brand_a | 5 | 0.58% |
| record | methodology.participants.demographics.age.mean | 5 | 0.58% |
| record | methodology.participants.demographics.gender_distribution.0.percentage | 5 | 0.58% |
| record | results.comparative_analysis.0.grouping_factor | 5 | 0.58% |
| record | methodology.participants.demographics.age.std_dev | 5 | 0.58% |
| record | team_size | 5 | 0.58% |
| record | final_score: Score in format home-away | 5 | 0.58% |
| record | home_team: Team playing at home | 5 | 0.58% |
| record | match_id: Unique identifier for the match | 5 | 0.58% |
| record | monthlyactiveusers | 5 | 0.58% |
| record | imaging_modality | 5 | 0.58% |
| record | allergies | 5 | 0.58% |
| record | retailer | 5 | 0.58% |
| record | mentor | 5 | 0.58% |
| record | renewal_code | 5 | 0.58% |
| record | report_id: Unique identifier for the phishing incident | 5 | 0.58% |
| record | campaign_id | 5 | 0.58% |
| record | startup_name | 5 | 0.58% |
| record | legislative_assembly.structure.political_associations.3.seats | 5 | 0.58% |
| record | trend_description | 5 | 0.58% |
| record | diagnosis: Medical diagnosis code and description | 5 | 0.58% |
| record | wind_speed | 5 | 0.58% |
| record | store_location | 5 | 0.58% |
| record | champion | 5 | 0.58% |
| record | prescription_id | 5 | 0.58% |
| record | strength | 5 | 0.58% |
| record | status: Current status of the outreach message | 5 | 0.58% |
| record | client_name | 5 | 0.58% |
| record | risk_score | 5 | 0.58% |
| record | enrollment_notes | 5 | 0.58% |
| record | participants.groups.0.sample_size | 5 | 0.58% |
| record | methodology.assessment_tool.validation_metrics.0.value | 5 | 0.58% |
| record | methodology.assessment_tool.type | 5 | 0.58% |
| record | results.outcomes.1.post_intervention.standard_deviation | 5 | 0.58% |
| record | results.outcomes.5.post_intervention.mean | 5 | 0.58% |
| record | methodology.assessment_tool.validation_metrics.0.metric_name | 5 | 0.58% |
| record | upvotes | 5 | 0.58% |
| record | submission_id | 5 | 0.58% |
| record | time: Time of the performance in HH:MM PM/AM format | 5 | 0.58% |
| record | genre: Music genre or blend of the performance | 5 | 0.58% |
| record | orbital_period.duration | 5 | 0.58% |
| record | state | 5 | 0.58% |
| record | asteroid.lagrange_point | 5 | 0.58% |
| record | discovery.discoverer.name | 5 | 0.58% |
| record | discovery.discoverer.nationality | 5 | 0.58% |
| record | coordinating_body: International organization responsible for oversight and coordination | 5 | 0.58% |
| record | tracks_per_cd | 5 | 0.58% |
| record | publication_journal | 5 | 0.58% |
| record | primary_endpoint | 5 | 0.58% |
| record | indication | 5 | 0.58% |
| record | study_result | 5 | 0.58% |
| record | warranty.duration | 5 | 0.58% |
| record | shelves.count | 5 | 0.58% |
| record | determiner | 5 | 0.58% |
| record | plural noun | 5 | 0.58% |
| record | educational institution | 5 | 0.58% |
| record | future_work | 5 | 0.58% |
| record | likes | 5 | 0.58% |
| record | poem.stanzas.1.lines | 5 | 0.58% |
| record | proteins | 5 | 0.58% |
| record | sector | 5 | 0.58% |
| record | subject.universities | 5 | 0.58% |
| record | revision_status | 5 | 0.58% |
| record | population.number_of_inhabitants | 5 | 0.58% |
| record | administrative_division.country | 5 | 0.58% |
| record | airport.altitude.unit | 5 | 0.58% |
| record | statistics.source | 5 | 0.58% |
| record | away_team: Name of the away basketball team | 5 | 0.58% |
| record | asteroid.name_original | 5 | 0.58% |
| record | study_id | 5 | 0.58% |
| record | seller_id: Identifier of the seller who provided the product | 5 | 0.58% |
| record | status: Current operational status of the company | 5 | 0.58% |
| record | location.city | 5 | 0.58% |
| record | office_id | 5 | 0.58% |
| record | manager | 5 | 0.58% |
| record | launch_date: Date the product was first released | 5 | 0.58% |
| record | owner | 5 | 0.58% |
| record | asteroid.discovery.discoverers.0.name | 5 | 0.58% |
| record | time | 5 | 0.58% |
| record | object_metadata.type | 5 | 0.58% |
| record | credits.photographer | 5 | 0.58% |
| record | vote_outcome | 5 | 0.58% |
| record | medications | 5 | 0.58% |
| record | hospital_name | 5 | 0.58% |
| record | economy.primary_occupation | 5 | 0.58% |
| record | department: Business unit or division responsible for the role | 5 | 0.58% |
| record | software | 4 | 0.46% |
| record | enrollment_date: Date the patient joined the trial | 4 | 0.46% |
| record | name: Full name of the enrolled patient | 4 | 0.46% |
| record | attendance | 4 | 0.46% |
| record | weather | 4 | 0.46% |
| record | validation_status: Status of network validation | 4 | 0.46% |
| record | network_id: Unique identifier for the regulatory network | 4 | 0.46% |
| record | activity_phase: Time or developmental phase of network activity | 4 | 0.46% |
| record | target_genes: Number of genes directly regulated by the regulator | 4 | 0.46% |
| record | regulatory_strength: Average interaction score indicating binding strength | 4 | 0.46% |
| record | company: Name of the startup receiving funding | 4 | 0.46% |
| record | subscription_type | 4 | 0.46% |
| record | delivery_frequency | 4 | 0.46% |
| record | growth_rate | 4 | 0.46% |
| record | ticket_id: Unique identifier for the support ticket | 4 | 0.46% |
| record | management_comment_prompt | 4 | 0.46% |
| record | price_range.currency | 4 | 0.46% |
| record | star_rating | 4 | 0.46% |
| record | prize_pool | 4 | 0.46% |
| record | player_age | 4 | 0.46% |
| record | court | 4 | 0.46% |
| record | date_initiated | 4 | 0.46% |
| record | no_count | 4 | 0.46% |
| record | lead_investor: List of investors leading the round | 4 | 0.46% |
| record | locations | 4 | 0.46% |
| record | upload_time: Time of upload in 12-hour format with AM/PM | 4 | 0.46% |
| record | likes: Number of likes the video has accumulated | 4 | 0.46% |
| record | materials_used: List of materials or components used in the video's creation | 4 | 0.46% |
| record | conclusion.advice_for_consultants.career_progression | 4 | 0.46% |
| record | advantages.general | 4 | 0.46% |
| record | conclusion.advice_for_consultants.required_skills | 4 | 0.46% |
| record | host: Name of the person or group organizing the event | 4 | 0.46% |
| record | location: Physical address or venue name of the event | 4 | 0.46% |
| record | status: Current status of the event | 4 | 0.46% |
| record | capacity: Maximum number of attendees allowed | 4 | 0.46% |
| record | page_count | 4 | 0.46% |
| record | base_currency | 4 | 0.46% |
| record | user_agent | 4 | 0.46% |
| record | app_id | 4 | 0.46% |
| record | key_features | 4 | 0.46% |
| record | review_status | 4 | 0.46% |
| record | engagement_level | 4 | 0.46% |
| record | diagnosis_code | 4 | 0.46% |
| record | earningscalldate: Date when the earnings call was held | 4 | 0.46% |
| record | netincomemillions: Net income in millions of dollars reported | 4 | 0.46% |
| record | ceo_name: Name of the CEO who presented the call | 4 | 0.46% |
| record | status: Current status of the earnings call delivery | 4 | 0.46% |
| record | content | 4 | 0.46% |
| record | organization_type | 4 | 0.46% |
| record | audience | 4 | 0.46% |
| record | cause | 4 | 0.46% |
| record | product.type | 4 | 0.46% |
| record | product.description | 4 | 0.46% |
| record | date_opened | 4 | 0.46% |
| record | ownera | 4 | 0.46% |
| record | ownerlocationa | 4 | 0.46% |
| record | study_overview.target_instrument.version_language | 4 | 0.46% |
| record | study_overview.disease_focus | 4 | 0.46% |
| record | results.validity_metrics.correlations.1.p_value | 4 | 0.46% |
| record | study_overview.objective | 4 | 0.46% |
| record | results.reliability_metrics.cronbach_alpha | 4 | 0.46% |
| record | results.validity_metrics.correlations.0.r_value | 4 | 0.46% |
| record | methodology.participants.demographics.gender_distribution.0.gender | 4 | 0.46% |
| record | results.validity_metrics.correlations.1.instrument_b | 4 | 0.46% |
| record | results.validity_metrics.correlations.1.instrument_a | 4 | 0.46% |
| record | methodology.assessment_tools | 4 | 0.46% |
| record | dept_code | 4 | 0.46% |
| record | post_date | 4 | 0.46% |
| record | program_name: Name of the training program enrolled in | 4 | 0.46% |
| record | away_team: Team playing away | 4 | 0.46% |
| record | targetusersegment | 4 | 0.46% |
| record | band | 4 | 0.46% |
| record | report_location | 4 | 0.46% |
| record | findings | 4 | 0.46% |
| record | investor_id | 4 | 0.46% |
| record | defendant | 4 | 0.46% |
| record | insurance_policy | 4 | 0.46% |
| record | dream_element | 4 | 0.46% |
| record | article.event.name | 4 | 0.46% |
| record | article.ranking.1.player | 4 | 0.46% |
| record | article.players.0.name | 4 | 0.46% |
| record | article.social_media_gaffe.player | 4 | 0.46% |
| record | article.social_media_gaffe.device | 4 | 0.46% |
| record | article.ranking.1.ranking | 4 | 0.46% |
| record | article.ranking.0.player | 4 | 0.46% |
| record | article.players.0.performance.score | 4 | 0.46% |
| record | article.players.0.nationality | 4 | 0.46% |
| record | article.players.1.name | 4 | 0.46% |
| record | article.ranking.0.ranking | 4 | 0.46% |
| record | article.social_media_gaffe.brand_conflict | 4 | 0.46% |
| record | team | 4 | 0.46% |
| record | years | 4 | 0.46% |
| record | address: Patient's residential address including street, city, state, zip | 4 | 0.46% |
| record | age: Patient's age in years | 4 | 0.46% |
| record | dose: Dosage amount administered as text | 4 | 0.46% |
| record | duration: Duration of medication use before reaction | 4 | 0.46% |
| record | payroll_id | 4 | 0.46% |
| record | health_email | 4 | 0.46% |
| record | location: Geographic location of the incident | 4 | 0.46% |
| record | ip_address: Source IP address involved in unauthorized access | 4 | 0.46% |
| record | date: Date the phishing report was submitted | 4 | 0.46% |
| record | model | 4 | 0.46% |
| record | dimensions | 4 | 0.46% |
| record | shipping_time | 4 | 0.46% |
| record | founder_name | 4 | 0.46% |
| record | role: Official title or position held | 4 | 0.46% |
| record | name: Full name of the appointed individual | 4 | 0.46% |
| record | agency: Department or office where appointment is made | 4 | 0.46% |
| record | legislative_assembly.structure.political_associations.0.seats | 4 | 0.46% |
| record | legislative_assembly.structure.political_associations.1.party | 4 | 0.46% |
| record | legislative_assembly.country | 4 | 0.46% |
| record | legislative_assembly.state | 4 | 0.46% |
| record | legislative_assembly.name | 4 | 0.46% |
| record | patient_id: Unique identifier for the patient | 4 | 0.46% |
| record | provider: Name of the healthcare provider conducting the visit | 4 | 0.46% |
| record | festival_id | 4 | 0.46% |
| record | capacity | 4 | 0.46% |
| record | match_id | 4 | 0.46% |
| record | booking_options | 4 | 0.46% |
| record | additional_info | 4 | 0.46% |
| record | decade | 4 | 0.46% |
| record | delay_minutes | 4 | 0.46% |
| record | rainfall | 4 | 0.46% |
| record | clause_type | 4 | 0.46% |
| record | related_devices.1.brand | 4 | 0.46% |
| record | related_devices.1.specifications.resolution | 4 | 0.46% |
| record | related_devices.0.name | 4 | 0.46% |
| record | related_devices.1.name | 4 | 0.46% |
| record | related_devices.2.brand | 4 | 0.46% |
| record | related_devices.0.specifications.display_size | 4 | 0.46% |
| record | series_code | 4 | 0.46% |
| record | player_id | 4 | 0.46% |
| record | document | 4 | 0.46% |
| record | concepts_taught | 4 | 0.46% |
| record | total_region_communes | 4 | 0.46% |
| record | region_ranking_area | 4 | 0.46% |
| record | keywords | 4 | 0.46% |
| record | message_subject: Subject line of the outreach message | 4 | 0.46% |
| record | sender: Person initiating the outreach contact | 4 | 0.46% |
| record | review_status.message | 4 | 0.46% |
| record | email: Contact email address of the ticket owner | 4 | 0.46% |
| record | underwriter_id | 4 | 0.46% |
| record | risk_category | 4 | 0.46% |
| record | policy_type | 4 | 0.46% |
| record | transaction_id | 4 | 0.46% |
| record | wallet_address | 4 | 0.46% |
| record | infringer | 4 | 0.46% |
| record | results.outcomes.0.post_intervention.standard_deviation | 4 | 0.46% |
| record | results.outcomes.3.post_intervention.standard_deviation | 4 | 0.46% |
| record | participants.groups.1.group_name | 4 | 0.46% |
| record | participants.sampling_method | 4 | 0.46% |
| record | results.outcomes.1.post_intervention.mean | 4 | 0.46% |
| record | study_metadata.study_design | 4 | 0.46% |
| record | results.outcomes.1.pre_intervention.mean | 4 | 0.46% |
| record | results.outcomes.0.pre_intervention.mean | 4 | 0.46% |
| record | results.outcomes.5.pre_intervention.mean | 4 | 0.46% |
| record | results.outcomes.2.construct | 4 | 0.46% |
| record | results.outcomes.3.pre_intervention.standard_deviation | 4 | 0.46% |
| record | conclusions.intervention_effect | 4 | 0.46% |
| record | results.outcomes.3.post_intervention.mean | 4 | 0.46% |
| record | study_metadata.objective | 4 | 0.46% |
| record | participants.population_type | 4 | 0.46% |
| record | participants.groups.0.allocation_method | 4 | 0.46% |
| record | soil_type | 4 | 0.46% |
| record | average_distance.unit | 4 | 0.46% |
| record | seconddoseid | 4 | 0.46% |
| record | zip | 4 | 0.46% |
| record | asteroid.greek_name | 4 | 0.46% |
| record | asteroid.planet | 4 | 0.46% |
| record | asteroid.position_relative_to_planet | 4 | 0.46% |
| record | discovery.observatory | 4 | 0.46% |
| record | naming.named_after.name | 4 | 0.46% |
| record | naming.named_after.description | 4 | 0.46% |
| record | executive_signatory: Name of the official who signed the treaty on behalf of a nation | 4 | 0.46% |
| record | location: City and country where the treaty was signed | 4 | 0.46% |
| record | financial_commitment: Amount of money pledged under the treaty in USD or EUR | 4 | 0.46% |
| record | signing_date: Date when the treaty was officially signed | 4 | 0.46% |
| record | store | 4 | 0.46% |
| record | review_type | 4 | 0.46% |
| record | flight_id | 4 | 0.46% |
| record | related_devices.0.model | 4 | 0.46% |
| record | principal_investigator | 4 | 0.46% |
| record | required_skills | 4 | 0.46% |
| record | application_link | 4 | 0.46% |
| record | details | 4 | 0.46% |
| record | filing_id | 4 | 0.46% |
| record | date_started | 4 | 0.46% |
| record | route | 4 | 0.46% |
| record | participants.groups.1.allocation_method | 4 | 0.46% |
| record | participants.total_sample_size | 4 | 0.46% |
| record | methodology.assessment_tool.categories_measured | 4 | 0.46% |
| record | study_metadata.setting.institution | 4 | 0.46% |
| record | results.outcomes.4.pre_intervention.standard_deviation | 4 | 0.46% |
| record | results.statistical_significance_threshold | 4 | 0.46% |
| record | results.outcomes.1.pre_intervention.standard_deviation | 4 | 0.46% |
| record | study_metadata.topic | 4 | 0.46% |
| record | dimensions.shelf_depth | 4 | 0.46% |
| record | dimensions.rough_opening_depth | 4 | 0.46% |
| record | owner: Name of the engineer responsible for the ticket | 4 | 0.46% |
| record | technology: Specific system or service involved in the issue | 4 | 0.46% |
| record | resolution: Action taken to resolve or mitigate the issue | 4 | 0.46% |
| record | duration: Time taken to resolve the incident in minutes or hours | 4 | 0.46% |
| record | cost: Financial cost associated with resolving the ticket | 4 | 0.46% |
| record | authors | 4 | 0.46% |
| record | legal_rep | 4 | 0.46% |
| record | verb, past tense | 4 | 0.46% |
| record | comma | 4 | 0.46% |
| record | comparative adverb | 4 | 0.46% |
| record | proper noun, singular | 4 | 0.46% |
| record | colon | 4 | 0.46% |
| record | review_date | 4 | 0.46% |
| record | saves | 4 | 0.46% |
| record | date_posted | 4 | 0.46% |
| record | game_title | 4 | 0.46% |
| record | poem.stanzas.0.lines | 4 | 0.46% |
| record | poem.stanzas.4.lines | 4 | 0.46% |
| record | methodology.participants.disease_subtype | 4 | 0.46% |
| record | methodology.participants.demographics.disease_duration_years.mean | 4 | 0.46% |
| record | results.validity_metrics.correlations.1.r_value | 4 | 0.46% |
| record | results.comparative_analysis.1.significant_items | 4 | 0.46% |
| record | company_name | 4 | 0.46% |
| record | subject.person.name | 4 | 0.46% |
| record | subject.person.offers | 4 | 0.46% |
| record | subject.person.school.name | 4 | 0.46% |
| record | dimensions.rough_opening | 4 | 0.46% |
| record | outreach_id: Unique identifier assigned to the outreach | 4 | 0.46% |
| record | reviewer_name | 4 | 0.46% |
| record | administrative_division.province_prefecture | 4 | 0.46% |
| record | sheikhdom.name | 4 | 0.46% |
| record | administrative_division.commune | 4 | 0.46% |
| record | vehicle_year | 4 | 0.46% |
| record | policyholder | 4 | 0.46% |
| record | airport.description | 4 | 0.46% |
| record | airport.name | 4 | 0.46% |
| record | airport.iata_code | 4 | 0.46% |
| record | airport.location.city | 4 | 0.46% |
| record | overview | 4 | 0.46% |
| record | conclusion.advice_for_companies | 4 | 0.46% |
| record | tiktok_username | 4 | 0.46% |
| record | retail_price | 4 | 0.46% |
| record | vehicle_make | 4 | 0.46% |
| record | address: Complete street address including city, state, and zip | 4 | 0.46% |
| record | patient_id: Unique identifier for the patient in the trial | 4 | 0.46% |
| record | job_id | 4 | 0.46% |
| record | model_id | 4 | 0.46% |
| record | priority | 4 | 0.46% |
| record | fg_percentage: Field goal percentage of the player | 4 | 0.46% |
| record | three_percentage: Three-point field goal percentage of the player | 4 | 0.46% |
| record | player_name: Name of the player with performance stats | 4 | 0.46% |
| record | assists: Total assists recorded by the player | 4 | 0.46% |
| record | rebounds: Total rebounds recorded by the player | 4 | 0.46% |
| record | home_team: Name of the home basketball team | 4 | 0.46% |
| record | asteroid.discovery.discoverer.nationality | 4 | 0.46% |
| record | orbit | 4 | 0.46% |
| record | asteroid.name_number | 4 | 0.46% |
| record | recruiter_id | 4 | 0.46% |
| record | interview_date | 4 | 0.46% |
| record | sender: Email address of the report sender | 4 | 0.46% |
| record | source: Internal system or process generating the report | 4 | 0.46% |
| record | airport.icao_code | 4 | 0.46% |
| record | poem.source | 4 | 0.46% |
| record | serial_number: Unique hardware identifier for the device | 4 | 0.46% |
| record | issue_date | 4 | 0.46% |
| record | shipping_time: Number of days for shipping from order to delivery | 4 | 0.46% |
| record | book_title | 4 | 0.46% |
| record | release_year_end | 4 | 0.46% |
| record | release_countries | 4 | 0.46% |
| record | alternate_name | 4 | 0.46% |
| record | market_share: Percentage of market share held by the company | 4 | 0.46% |
| record | game_id: Unique identifier for each game | 4 | 0.46% |
| record | legislative_assembly.current_government | 4 | 0.46% |
| record | legislative_assembly.structure.number_of_members | 4 | 0.46% |
| record | legislative_assembly.structure.political_associations.2.seats | 4 | 0.46% |
| record | population | 4 | 0.46% |
| record | name: Product name as marketed to consumers | 4 | 0.46% |
| record | close_date | 4 | 0.46% |
| record | party1 | 4 | 0.46% |
| record | address1 | 4 | 0.46% |
| record | insurance_provider | 4 | 0.46% |
| record | hba1c | 4 | 0.46% |
| record | asteroid.discovery.discoverers.1.name | 4 | 0.46% |
| record | asteroid.description | 4 | 0.46% |
| record | provider_name | 4 | 0.46% |
| record | provider_address | 4 | 0.46% |
| record | cpt_code | 4 | 0.46% |
| record | chemical_name | 4 | 0.46% |
| record | supplier | 4 | 0.46% |
| record | rsvp_count | 4 | 0.46% |
| record | player_name | 4 | 0.46% |
| record | date_sent | 4 | 0.46% |
| record | confirmation_code | 4 | 0.46% |
| record | event_type | 4 | 0.46% |
| record | establishment_year | 4 | 0.46% |
| record | customer_name: Name of the customer who made the purchase | 4 | 0.46% |
| record | purchase_date: Date when the product was bought | 4 | 0.46% |
| record | product_name: Name of the product purchased | 4 | 0.46% |
| record | purchase_price: Price paid for the product in US dollars | 4 | 0.46% |
| record | revenue | 4 | 0.46% |
| record | focus_area | 4 | 0.46% |
| record | position_id | 4 | 0.46% |
| record | gameplay_elements.towers.0.placement_strategy | 4 | 0.46% |
| record | gameplay_elements.enemies.2.characteristics | 4 | 0.46% |
| record | gameplay_elements.enemies.1.characteristics | 4 | 0.46% |
| record | gameplay_elements.enemies.3.zombie_type | 4 | 0.46% |
| record | gameplay_elements.enemies.0.zombie_type | 4 | 0.46% |
| record | game_description | 4 | 0.46% |
| record | alma_collaboration.description | 4 | 0.46% |
| record | image_description.alma_status.antenna_count.seven_meter | 4 | 0.46% |
| record | image_description.location.country | 4 | 0.46% |
| record | credits.translation | 4 | 0.46% |
| record | image_description.alma_status.performance | 4 | 0.46% |
| record | image_metadata.publication_date | 4 | 0.46% |
| record | image_description.alma_status.first_observations | 4 | 0.46% |
| record | alma_collaboration.management.0.region | 4 | 0.46% |
| record | committee | 4 | 0.46% |
| record | vote_count | 4 | 0.46% |
| record | zodiac_signs | 4 | 0.46% |
| record | intellectual_compatibility.shared_traits | 4 | 0.46% |
| record | intellectual_compatibility.gemini_perspective.approach | 4 | 0.46% |
| record | intellectual_compatibility.virgo_perspective.focus | 4 | 0.46% |
| record | advice | 4 | 0.46% |
| record | opening quotation mark | 4 | 0.46% |
| record | verb, non-3rd person singular present | 4 | 0.46% |
| record | period | 4 | 0.46% |
| record | displaced_persons | 4 | 0.46% |
| record | image_description.location.altitude | 4 | 0.46% |
| record | image_description.alma_status.completion_date | 4 | 0.46% |
| record | alma_collaboration.management.0.organization | 4 | 0.46% |
| record | policyholder_id | 4 | 0.46% |
| record | subject.person.rating | 4 | 0.46% |
| record | vitalsignstemp | 4 | 0.46% |
| record | staystartdate | 4 | 0.46% |
| record | admitting_diagnosis | 4 | 0.46% |
| record | medicalrecordid | 4 | 0.46% |
| record | primarycareprovider | 4 | 0.46% |
| record | lab_wbc | 4 | 0.46% |
| record | status: Current product availability status | 4 | 0.46% |
| record | report_signer | 4 | 0.46% |
| record | key_event | 4 | 0.46% |
| record | final_score | 4 | 0.46% |
| record | status: Current state of the ticket | 4 | 0.46% |
| record | submitter_address: Full address of the submitter | 4 | 0.46% |
| record | target_audience | 4 | 0.46% |
| record | speaker | 4 | 0.46% |
| record | participant_id: Unique identifier for the participant | 4 | 0.46% |
| record | galaxy_type_code | 4 | 0.46% |
| record | object_name | 4 | 0.46% |
| record | views_past_three_months | 4 | 0.46% |
| record | article.players.1.nationality | 4 | 0.46% |
| record | location: City and state where the job is based | 4 | 0.46% |
| record | company: Name of the organization hiring | 4 | 0.46% |
| record | analysis_date: Date when the company analysis was conducted | 4 | 0.46% |
| record | market_cap | 4 | 0.46% |
| record | discharge_date | 4 | 0.46% |
| record | interpretation.1.meaning | 3 | 0.35% |
| record | interpretation.0.meaning | 3 | 0.35% |
| record | vehicle | 3 | 0.35% |
| record | money | 3 | 0.35% |
| record | status: Current enrollment status of the patient | 3 | 0.35% |
| record | key_player | 3 | 0.35% |
| record | location: City and state where the company is headquartered | 3 | 0.35% |
| record | amount: Total funding amount in millions | 3 | 0.35% |
| record | payment_amount | 3 | 0.35% |
| record | actions | 3 | 0.35% |
| record | ceo_name | 3 | 0.35% |
| record | date: Date the ticket was created or reported | 3 | 0.35% |
| record | owner: Name of the ticket owner or user | 3 | 0.35% |
| record | hotel_name | 3 | 0.35% |
| record | loser | 3 | 0.35% |
| record | tournament | 3 | 0.35% |
| record | match_duration | 3 | 0.35% |
| record | general_topics | 3 | 0.35% |
| record | cuisine_type | 3 | 0.35% |
| record | flavor_score | 3 | 0.35% |
| record | bill_id | 3 | 0.35% |
| record | yes_count | 3 | 0.35% |
| record | funding_details | 3 | 0.35% |
| record | sponsor | 3 | 0.35% |
| record | title: Title of the video as presented in the description | 3 | 0.35% |
| record | comments: Number of comments on the video | 3 | 0.35% |
| record | definition.description | 3 | 0.35% |
| record | start_time: Start time of the event in 12-hour format | 3 | 0.35% |
| record | classification | 3 | 0.35% |
| record | other_works_mentioned | 3 | 0.35% |
| record | school_id | 3 | 0.35% |
| record | id: Unique identifier for the earnings call | 3 | 0.35% |
| record | company_name: Name of the company presenting earnings | 3 | 0.35% |
| record | quarter: Financial quarter reported, e.g., Q1 2023 | 3 | 0.35% |
| record | revenue_millions: Revenue in millions of dollars reported | 3 | 0.35% |
| record | session_type | 3 | 0.35% |
| record | experience | 3 | 0.35% |
| record | service | 3 | 0.35% |
| record | activity | 3 | 0.35% |
| record | standards | 3 | 0.35% |
| record | product.name | 3 | 0.35% |
| record | inventors | 3 | 0.35% |
| record | assignee | 3 | 0.35% |
| record | assignee_address | 3 | 0.35% |
| record | methodology.statistical_methods | 3 | 0.35% |
| record | methodology.participants.demographics.gender_distribution.1.count | 3 | 0.35% |
| record | results.comparative_analysis.1.grouping_factor | 3 | 0.35% |
| record | results.validity_metrics.correlations.0.instrument_b | 3 | 0.35% |
| record | duration: Number of weeks the program lasts | 3 | 0.35% |
| record | dose | 3 | 0.35% |
| record | frequency | 3 | 0.35% |
| record | associated_with | 3 | 0.35% |
| record | lead_name | 3 | 0.35% |
| record | lead_id | 3 | 0.35% |
| record | goal_minutes: List of minutes when goals were scored | 3 | 0.35% |
| record | referee: Name of the match referee | 3 | 0.35% |
| record | plaintiff | 3 | 0.35% |
| record | bmi | 3 | 0.35% |
| record | contact_id | 3 | 0.35% |
| record | article.players.1.performance.score | 3 | 0.35% |
| record | patient_name: Full name of the affected patient | 3 | 0.35% |
| record | severity: Numerical rating of symptom severity from 1 to 5 | 3 | 0.35% |
| record | healthpolicyid | 3 | 0.35% |
| record | recipient: Email address of the report recipient | 3 | 0.35% |
| record | status: Current status of the report | 3 | 0.35% |
| record | fda_class | 3 | 0.35% |
| record | category | 3 | 0.35% |
| record | regulator: Transcription factor or protein regulating gene expression | 3 | 0.35% |
| record | sample_size: Number of patient or cell samples used in study | 3 | 0.35% |
| record | customer_address | 3 | 0.35% |
| record | defendant_company | 3 | 0.35% |
| record | legislative_assembly.seat.building | 3 | 0.35% |
| record | state_politics.state_name | 3 | 0.35% |
| record | legislative_assembly.voting_rights_history.0.year | 3 | 0.35% |
| record | legislative_assembly.structure.political_associations.1.seats | 3 | 0.35% |
| record | legislative_assembly.structure.political_associations.2.party | 3 | 0.35% |
| record | interpretation.0.details.health | 3 | 0.35% |
| record | content_type | 3 | 0.35% |
| record | asteroid.discovery.year | 3 | 0.35% |
| record | asteroid.name_origin | 3 | 0.35% |
| record | medications: Prescribed medication with dosage and frequency | 3 | 0.35% |
| record | status: Current status of the patient's visit or condition | 3 | 0.35% |
| record | health_address | 3 | 0.35% |
| record | health_status | 3 | 0.35% |
| record | location.country | 3 | 0.35% |
| record | passenger_name | 3 | 0.35% |
| record | carrier | 3 | 0.35% |
| record | related_devices.1.model | 3 | 0.35% |
| record | related_devices.0.brand | 3 | 0.35% |
| record | related_devices.2.model | 3 | 0.35% |
| record | related_devices.1.device_type | 3 | 0.35% |
| record | product_code | 3 | 0.35% |
| record | to_club | 3 | 0.35% |
| record | features_and_benefits | 3 | 0.35% |
| record | analysis_protocol.data_display | 3 | 0.35% |
| record | concept | 3 | 0.35% |
| record | feature | 3 | 0.35% |
| record | platform | 3 | 0.35% |
| record | website | 3 | 0.35% |
| record | region_ranking_population | 3 | 0.35% |
| record | tools_and_models | 3 | 0.35% |
| record | meeting_date | 3 | 0.35% |
| record | company: Company where the recipient works | 3 | 0.35% |
| record | program_title | 3 | 0.35% |
| record | asset | 3 | 0.35% |
| record | participants.groups.1.sample_size | 3 | 0.35% |
| record | results.outcomes.0.construct | 3 | 0.35% |
| record | results.outcomes.0.post_intervention.mean | 3 | 0.35% |
| record | methodology.intervention_type | 3 | 0.35% |
| record | results.outcomes.1.construct | 3 | 0.35% |
| record | results.outcomes.2.pre_intervention.standard_deviation | 3 | 0.35% |
| record | results.outcomes.4.construct | 3 | 0.35% |
| record | results.outcomes.4.pre_intervention.mean | 3 | 0.35% |
| record | budget | 3 | 0.35% |
| record | performer: Name of the musical act performing | 3 | 0.35% |
| record | orbital_period.unit | 3 | 0.35% |
| record | average_distance.distance | 3 | 0.35% |
| record | discovery.date | 3 | 0.35% |
| record | legal_code: Reference to the legal framework or regulation under which the treaty operates | 3 | 0.35% |
| record | brand_b | 3 | 0.35% |
| record | filing_date | 3 | 0.35% |
| record | purchase_date | 3 | 0.35% |
| record | compensation_type: Type of pay structure | 3 | 0.35% |
| record | number_of_cds | 3 | 0.35% |
| record | related_devices.0.device_type | 3 | 0.35% |
| record | submission_method | 3 | 0.35% |
| record | active_ingredient | 3 | 0.35% |
| record | subject | 3 | 0.35% |
| record | results.outcomes.2.post_intervention.mean | 3 | 0.35% |
| record | results.outcomes.3.construct | 3 | 0.35% |
| record | results.outcomes.5.post_intervention.standard_deviation | 3 | 0.35% |
| record | results.outcomes.5.construct | 3 | 0.35% |
| record | results.outcomes.4.post_intervention.standard_deviation | 3 | 0.35% |
| record | results.outcomes.5.pre_intervention.standard_deviation | 3 | 0.35% |
| record | results.outcomes.3.pre_intervention.mean | 3 | 0.35% |
| record | role_id: Unique identifier for the job position | 3 | 0.35% |
| record | location: City and state where the role is based | 3 | 0.35% |
| record | dimensions.depth | 3 | 0.35% |
| record | dimensions.overall_size | 3 | 0.35% |
| record | ticket_id: Unique identifier for each incident as seen in the log | 3 | 0.35% |
| record | date: Date the incident was reported or occurred | 3 | 0.35% |
| record | duration: Duration of the video in minutes and seconds | 3 | 0.35% |
| record | location: Physical location where the video was filmed | 3 | 0.35% |
| record | cost: Total cost of equipment or materials used in the video | 3 | 0.35% |
| record | participant_count | 3 | 0.35% |
| record | trademark_name | 3 | 0.35% |
| record | comparative adjective | 3 | 0.35% |
| record | movie_title | 3 | 0.35% |
| record | person_name | 3 | 0.35% |
| record | team_count | 3 | 0.35% |
| record | study_overview.target_instrument.full_name_original | 3 | 0.35% |
| record | study_overview.target_instrument.target_population_region | 3 | 0.35% |
| record | methodology.participants.demographics.gender_distribution.1.percentage | 3 | 0.35% |
| record | methodology.participants.clinical_status.in_relapse_phase.percentage | 3 | 0.35% |
| record | methodology.reliability_testing_procedure.retest_interval_weeks | 3 | 0.35% |
| record | results.reliability_metrics.icc_score | 3 | 0.35% |
| record | results.validity_metrics.correlations.0.instrument_a | 3 | 0.35% |
| record | tissue | 3 | 0.35% |
| record | dish_description | 3 | 0.35% |
| record | service_quality | 3 | 0.35% |
| record | presentation_score | 3 | 0.35% |
| record | features | 3 | 0.35% |
| record | recipient: Name of the recipient contacted | 3 | 0.35% |
| record | price_range | 3 | 0.35% |
| record | external_links | 3 | 0.35% |
| record | population.census_year | 3 | 0.35% |
| record | name: Full title of the treaty agreement | 3 | 0.35% |
| record | incident_code | 3 | 0.35% |
| record | airport.location.region | 3 | 0.35% |
| record | airport.name_chinese | 3 | 0.35% |
| record | warning_code | 3 | 0.35% |
| record | affected_area | 3 | 0.35% |
| record | post_id | 3 | 0.35% |
| record | discovery_date | 3 | 0.35% |
| record | dosage | 3 | 0.35% |
| record | product_model | 3 | 0.35% |
| record | lead_researcher | 3 | 0.35% |
| record | vehicle_model | 3 | 0.35% |
| record | availability_date | 3 | 0.35% |
| record | blocks: Total blocks recorded by the player | 3 | 0.35% |
| record | points: Total points scored by the player | 3 | 0.35% |
| record | asteroid.discovery.location.country | 3 | 0.35% |
| record | reference: Unique reference code for the incident | 3 | 0.35% |
| record | airport.status | 3 | 0.35% |
| record | poem.stanzas.2.lines | 3 | 0.35% |
| record | model: Manufacturer and model name of the medical device | 3 | 0.35% |
| record | location: Physical department or room where device is stored or used | 3 | 0.35% |
| record | supply_id | 3 | 0.35% |
| record | emergency_contact | 3 | 0.35% |
| record | product_condition: Whether the product was new or used at purchase | 3 | 0.35% |
| record | product_id: Unique identifier for the purchased product | 3 | 0.35% |
| record | section | 3 | 0.35% |
| record | vocalists | 3 | 0.35% |
| record | artist | 3 | 0.35% |
| record | program_code | 3 | 0.35% |
| record | population.census_type | 3 | 0.35% |
| record | revenue: Annual revenue in USD million | 3 | 0.35% |
| record | away_score: Final score of the away team | 3 | 0.35% |
| record | steals: Total steals recorded by the player | 3 | 0.35% |
| record | state_politics.political_system | 3 | 0.35% |
| record | legislative_assembly.structure.political_associations.0.party | 3 | 0.35% |
| record | legislative_assembly.electoral_system | 3 | 0.35% |
| record | legislative_assembly.last_election_date | 3 | 0.35% |
| record | legislative_assembly.structure.political_associations.3.party | 3 | 0.35% |
| record | legislative_assembly.establishment_year | 3 | 0.35% |
| record | annotation_id | 3 | 0.35% |
| record | party2 | 3 | 0.35% |
| record | dispute_duration | 3 | 0.35% |
| record | nursery_name | 3 | 0.35% |
| record | issue_type | 3 | 0.35% |
| record | resolution_amount | 3 | 0.35% |
| record | diseases_claimed_to_be_treated | 3 | 0.35% |
| record | reference_number | 3 | 0.35% |
| record | organizer | 3 | 0.35% |
| record | fundraising_amount | 3 | 0.35% |
| record | transfer_fee | 3 | 0.35% |
| record | training_week | 3 | 0.35% |
| record | job_title | 3 | 0.35% |
| record | asteroid.name_origin.relation | 3 | 0.35% |
| record | asteroid.name_origin.name | 3 | 0.35% |
| record | asteroid.discovery.discoverer.name | 3 | 0.35% |
| record | event_title: Title of the community event as posted | 3 | 0.35% |
| record | event_id: Unique identifier assigned to the event | 3 | 0.35% |
| record | festival_id: Unique identifier for the festival event | 3 | 0.35% |
| record | ticket_price: Price per ticket in dollars | 3 | 0.35% |
| record | sender | 3 | 0.35% |
| record | device | 3 | 0.35% |
| record | gameplay_elements.enemies.0.characteristics | 3 | 0.35% |
| record | image_description.celestial_phenomena | 3 | 0.35% |
| record | image_metadata.type | 3 | 0.35% |
| record | image_metadata.id | 3 | 0.35% |
| record | alma_collaboration.joint_observatory | 3 | 0.35% |
| record | alma_collaboration.management.1.organization | 3 | 0.35% |
| record | image_metadata.language | 3 | 0.35% |
| record | alma_collaboration.management.2.region | 3 | 0.35% |
| record | funding_threshold | 3 | 0.35% |
| record | voter_name | 3 | 0.35% |
| record | voter_state | 3 | 0.35% |
| record | voter_party | 3 | 0.35% |
| record | intellectual_compatibility.differences | 3 | 0.35% |
| record | adjective | 3 | 0.35% |
| record | cardinal number | 3 | 0.35% |
| record | closing quotation mark | 3 | 0.35% |
| record | image_description.alma_description | 3 | 0.35% |
| record | image_description.alma_status.antenna_count.twelve_meter | 3 | 0.35% |
| record | alma_collaboration.partners | 3 | 0.35% |
| record | alma_collaboration.management.2.organization | 3 | 0.35% |
| record | alma_collaboration.joint_observatory_responsibilities | 3 | 0.35% |
| record | image_metadata.size | 3 | 0.35% |
| record | object_metadata.name | 3 | 0.35% |
| record | achievement_description | 3 | 0.35% |
| record | issue | 3 | 0.35% |
| record | enrollment | 3 | 0.35% |
| record | scholarship_title | 3 | 0.35% |
| record | shares | 3 | 0.35% |
| record | hashtags | 3 | 0.35% |
| record | airport.passenger_flow.unit | 3 | 0.35% |
| record | airport.funding_source | 3 | 0.35% |
| record | id: Unique product identifier assigned by manufacturer | 3 | 0.35% |
| record | submitter_name: Name of the user who opened the ticket | 3 | 0.35% |
| record | ticket_id: Unique identifier for the helpdesk ticket | 3 | 0.35% |
| record | registration_cost | 3 | 0.35% |
| record | publisher | 3 | 0.35% |
| record | supervisor_name: Name of the supervisor approving participation | 3 | 0.35% |
| record | village_name | 3 | 0.35% |
| record | location.tehsil | 3 | 0.35% |
| record | economy.employment_source | 3 | 0.35% |
| record | asteroid.name_origin.description | 3 | 0.35% |
| record | source_institution | 3 | 0.35% |
| record | candidate_id | 3 | 0.35% |
| record | outreach_date | 3 | 0.35% |
| record | constellation | 3 | 0.35% |
| record | phone_number | 3 | 0.35% |
| record | filing_amount | 3 | 0.35% |
| record | article.players.0.performance.winners | 3 | 0.35% |
| record | article.ranking.0.change | 3 | 0.35% |
| record | article.players.1.performance.winners | 3 | 0.35% |
| record | article.brands | 3 | 0.35% |
| record | legislative_assembly.type | 3 | 0.35% |
| record | legislative_assembly.leadership.speaker | 3 | 0.35% |
| record | questions | 3 | 0.35% |
| record | results.outcomes.2.post_intervention.standard_deviation | 3 | 0.35% |
| record | results.outcomes.2.pre_intervention.mean | 3 | 0.35% |
| record | participant_name: Name of the enrolled employee | 3 | 0.35% |
| record | required_qualifications: List of required skills or certifications | 3 | 0.35% |
| record | movie_house | 3 | 0.35% |
| record | related_devices.2.specifications.resolution | 3 | 0.35% |
| record | company_name: Name of the company as listed in the dataset | 3 | 0.35% |
| record | gameplay_elements.strategy | 3 | 0.35% |
| record | gameplay_elements.towers.0.tower_type | 3 | 0.35% |
| record | gameplay_elements.enemies.2.zombie_type | 3 | 0.35% |
| record | firstdoseid | 3 | 0.35% |
| record | match_outcome | 3 | 0.35% |
| record | reference | 3 | 0.35% |
| record | priority: Severity level of the reported issue | 3 | 0.35% |
| record | device_type: Type of device experiencing the issue | 3 | 0.35% |
| record | device_os: Operating system of the affected device | 3 | 0.35% |
| record | object_type | 3 | 0.35% |
| record | performance_cause | 3 | 0.35% |
| record | serial | 3 | 0.35% |
| record | insurance_plan | 3 | 0.35% |
| record | discharge_diagnosis | 3 | 0.35% |
| record | discharge_status | 3 | 0.35% |
| record | discharge_instructions | 3 | 0.35% |
| record | lengthofstay | 3 | 0.35% |
| record | comorbidities | 3 | 0.35% |
| record | trial_code: Code identifying the clinical trial | 2 | 0.23% |
| record | surface | 2 | 0.23% |
| record | blank_count | 2 | 0.23% |
| record | companies | 2 | 0.23% |
| record | disadvantages.consultant_drawbacks | 2 | 0.23% |
| record | treatment | 2 | 0.23% |
| record | app_name | 2 | 0.23% |
| record | application_no | 2 | 0.23% |
| record | claims_count | 2 | 0.23% |
| record | domain | 2 | 0.23% |
| record | ownerb | 2 | 0.23% |
| record | methodology.participants.demographics.gender_distribution.1.gender | 2 | 0.23% |
| record | methodology.participants.demographics.disease_duration_years.std_dev | 2 | 0.23% |
| record | color | 2 | 0.23% |
| record | blood_pressure | 2 | 0.23% |
| record | product | 2 | 0.23% |
| record | units_sold | 2 | 0.23% |
| record | tissue_context: Tissue or cell type where network operates | 2 | 0.23% |
| record | hgnc_id: Human Gene Nomenclature Committee identifier for regulator | 2 | 0.23% |
| record | founding_year | 2 | 0.23% |
| record | registration_number | 2 | 0.23% |
| record | medication: Name of the drug involved in the reaction | 2 | 0.23% |
| record | interpretation.1.details.health | 2 | 0.23% |
| record | visit_date: Date of the patient's medical visit | 2 | 0.23% |
| record | address: Full street address including city, state, and zip | 2 | 0.23% |
| record | dates | 2 | 0.23% |
| record | goalscoringplayer | 2 | 0.23% |
| record | limitations | 2 | 0.23% |
| record | related_devices.2.device_type | 2 | 0.23% |
| record | from_club | 2 | 0.23% |
| record | resolution: Description of resolution or action taken, if any | 2 | 0.23% |
| record | registrant | 2 | 0.23% |
| record | registered_mark | 2 | 0.23% |
| record | registrantusptoid | 2 | 0.23% |
| record | participants.groups.0.group_name | 2 | 0.23% |
| record | venue_id: Identifier for the festival venue | 2 | 0.23% |
| record | release_year_start | 2 | 0.23% |
| record | passenger_email | 2 | 0.23% |
| record | related_devices.2.specifications.display_size | 2 | 0.23% |
| record | secondary_endpoints | 2 | 0.23% |
| record | approval_year | 2 | 0.23% |
| record | administration_route | 2 | 0.23% |
| record | results.outcomes.0.pre_intervention.standard_deviation | 2 | 0.23% |
| record | title: Descriptive name of the technical issue reported | 2 | 0.23% |
| record | root_cause: Primary reason leading to the incident | 2 | 0.23% |
| record | creator: Name of the video creator or subject | 2 | 0.23% |
| record | claimant_company | 2 | 0.23% |
| record | singular noun | 2 | 0.23% |
| record | adverb | 2 | 0.23% |
| record | possessive pronoun | 2 | 0.23% |
| record | tournament_id | 2 | 0.23% |
| record | stop_words | 2 | 0.23% |
| record | study_overview.target_instrument.acronym | 2 | 0.23% |
| record | results.validity_metrics.correlations.0.p_value | 2 | 0.23% |
| record | results.comparative_analysis.0.significant_items | 2 | 0.23% |
| record | exposure_level | 2 | 0.23% |
| record | exchange | 2 | 0.23% |
| record | subject.person.position | 2 | 0.23% |
| record | shelves.material | 2 | 0.23% |
| record | target_currency | 2 | 0.23% |
| record | role: Job title being offered or requested | 2 | 0.23% |
| record | salary: Annual salary offered in dollars | 2 | 0.23% |
| record | adjuster | 2 | 0.23% |
| record | publicreviewperiod | 2 | 0.23% |
| record | identifier | 2 | 0.23% |
| record | original_release_year | 2 | 0.23% |
| record | plaintiff_address | 2 | 0.23% |
| record | defendant_address | 2 | 0.23% |
| record | asteroid.discovery.location.city | 2 | 0.23% |
| record | intervention | 2 | 0.23% |
| record | airport.location.country | 2 | 0.23% |
| record | device_id: Unique identifier for each device in inventory | 2 | 0.23% |
| record | address: Full address of the customer including city, state, and zip | 2 | 0.23% |
| record | author_location | 2 | 0.23% |
| record | id: Unique identifier assigned to the treaty | 2 | 0.23% |
| record | full_name | 2 | 0.23% |
| record | date: Date the game was played | 2 | 0.23% |
| record | home_score: Final score of the home team | 2 | 0.23% |
| record | room_count | 2 | 0.23% |
| record | content_topics | 2 | 0.23% |
| record | location: Physical address of the company | 2 | 0.23% |
| record | certification | 2 | 0.23% |
| record | contract_id | 2 | 0.23% |
| record | open_date | 2 | 0.23% |
| record | claim_amount | 2 | 0.23% |
| record | hazard_code | 2 | 0.23% |
| record | email_id | 2 | 0.23% |
| record | asteroid.discovery.location | 2 | 0.23% |
| record | description: Brief summary of the trial's purpose and phase | 2 | 0.23% |
| record | event_id | 2 | 0.23% |
| record | qualifications | 2 | 0.23% |
| record | net_profit | 2 | 0.23% |
| record | entry_fee: Cost per person to attend, zero for free | 2 | 0.23% |
| record | occupation | 2 | 0.23% |
| record | launch_date | 2 | 0.23% |
| record | population_density | 2 | 0.23% |
| record | gameplay_elements.enemies.1.zombie_type | 2 | 0.23% |
| record | alma_collaboration.management.1.region | 2 | 0.23% |
| record | venue: Name and location of the match venue | 2 | 0.23% |
| record | intellectual_compatibility.gemini_perspective.focus | 2 | 0.23% |
| record | intellectual_compatibility.virgo_perspective.approach | 2 | 0.23% |
| record | image_description.location.name | 2 | 0.23% |
| record | alma_collaboration.host_country | 2 | 0.23% |
| record | premium | 2 | 0.23% |
| record | subject.person.school.location | 2 | 0.23% |
| record | admission_reason | 2 | 0.23% |
| record | review_id | 2 | 0.23% |
| record | company: Company that produces and sells the product | 2 | 0.23% |
| record | price: Monetary value of the product in USD | 2 | 0.23% |
| record | signing_parties: List of governmental or organizational entities that signed the treaty | 2 | 0.23% |
| record | organism | 2 | 0.23% |
| record | coverage_limits | 2 | 0.23% |
| record | location.district | 2 | 0.23% |
| record | transaction_amount | 2 | 0.23% |
| record | legislative_assembly.voting_rights_history.0.description | 2 | 0.23% |
| record | legislative_assembly.voting_rights_history.1.description | 2 | 0.23% |
| record | legislative_assembly.voting_rights_history.1.year | 2 | 0.23% |
| record | legislative_assembly.references.0.source | 2 | 0.23% |
| record | legislative_assembly.references.0.access_date | 2 | 0.23% |
| record | administrative_division.region | 2 | 0.23% |
| record | results.outcomes.4.post_intervention.mean | 2 | 0.23% |
| record | employee_id: Assigned employee ID for the role | 2 | 0.23% |
| record | study_metadata.setting.location | 2 | 0.23% |
| record | gameplay_elements.towers.0.abilities | 2 | 0.23% |
| record | gameplay_elements.enemies.3.characteristics | 2 | 0.23% |
| record | requisition_id: Unique identifier for the job posting | 2 | 0.23% |
| record | related_devices.0.specifications.resolution | 2 | 0.23% |
| record | location: Geographic location of the company headquarters | 2 | 0.23% |
| record | verb, gerund or present participle | 2 | 0.23% |
| record | preposition or subordinating conjunction | 2 | 0.23% |
| record | article.title | 2 | 0.23% |
| record | article.event.location | 2 | 0.23% |
| record | article.social_media_gaffe.platform | 2 | 0.23% |
| record | article.ranking.1.change | 2 | 0.23% |
| record | thumbnail | 2 | 0.23% |
| record | close_price | 2 | 0.23% |
| record | prize_money | 2 | 0.23% |
| record | related_devices.1.specifications.display_size | 2 | 0.23% |
| record | related_devices.2.name | 2 | 0.23% |
| record | id: Unique identifier for each company entry | 2 | 0.23% |
| record | cost: Total cost of the training program | 2 | 0.23% |
| record | lab_hb | 2 | 0.23% |
| record | asteroid.latin_name | 1 | 0.12% |
| record | molecular_target | 1 | 0.12% |
| record | version | 1 | 0.12% |
| record | eventtype | 1 | 0.12% |
| record | amount: Financial loss associated with the phishing attempt | 1 | 0.12% |
| record | asteroid.orbit.reference_plane | 1 | 0.12% |
| record | rating: Customer rating out of 5.0 | 1 | 0.12% |
| record | airport.cost.currency | 1 | 0.12% |
| record | poem.stanzas.3.lines | 1 | 0.12% |
| record | legislative_assembly.seat.location | 1 | 0.12% |
| record | article.ranking.2.player | 1 | 0.12% |
| record | registration_benefits | 1 | 0.12% |
| record | payment_method | 1 | 0.12% |
| record | ownerlocationb | 1 | 0.12% |
| record | sheikhdom.number_of_douars | 1 | 0.12% |
| record | vitalsignsbp | 1 | 0.12% |
| record | record_metadata | natural/status ×25, natural/name ×23, natural/title ×20, natural/location ×17 |  |

**test.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| record | status | 121 | 14.14% |
| record | location | 84 | 9.81% |
| record | name | 53 | 6.19% |
| record | address | 52 | 6.07% |
| record | description | 47 | 5.49% |
| record | title | 29 | 3.39% |
| record | date | 27 | 3.15% |
| record | ticket_id | 19 | 2.22% |
| record | amount | 17 | 1.99% |
| record | patient_id | 16 | 1.87% |
| record | reporter | 15 | 1.75% |
| record | rating | 14 | 1.64% |
| record | start_date | 13 | 1.52% |
| record | region | 13 | 1.52% |
| record | department | 13 | 1.52% |
| record | country | 12 | 1.40% |
| record | passenger_name | 12 | 1.40% |
| record | product_id | 12 | 1.40% |
| record | program_name | 12 | 1.40% |
| record | report_id | 11 | 1.29% |
| record | end_date | 11 | 1.29% |
| record | copyright | 10 | 1.17% |
| record | company_name | 10 | 1.17% |
| record | recipient | 10 | 1.17% |
| record | target_audience | 10 | 1.17% |
| record | duration | 10 | 1.17% |
| record | course_id | 9 | 1.05% |
| record | monitoring_body | 9 | 1.05% |
| record | status: Current status of the review | 9 | 1.05% |
| record | review_date | 9 | 1.05% |
| record | funding_source | 9 | 1.05% |
| record | upload_date | 9 | 1.05% |
| record | funding_amount | 9 | 1.05% |
| record | source | 8 | 0.93% |
| record | asteroid.designation | 8 | 0.93% |
| record | sender | 8 | 0.93% |
| record | product_name | 8 | 0.93% |
| record | disease | 8 | 0.93% |
| record | email | 8 | 0.93% |
| record | tail | 8 | 0.93% |
| record | head | 8 | 0.93% |
| record | flight_number | 8 | 0.93% |
| record | destination | 8 | 0.93% |
| record | topics | 8 | 0.93% |
| record | price | 8 | 0.93% |
| record | employee_id | 8 | 0.93% |
| record | genus | 8 | 0.93% |
| record | submitter | 7 | 0.82% |
| record | asteroid.discovery.observatory.name | 7 | 0.82% |
| record | issuer | 7 | 0.82% |
| record | event_id | 7 | 0.82% |
| record | product | 7 | 0.82% |
| record | temperature | 7 | 0.82% |
| record | player_name | 7 | 0.82% |
| record | manufacturer | 7 | 0.82% |
| record | asteroid.discovery.observatory.location.city | 7 | 0.82% |
| record | scheduled_departure | 7 | 0.82% |
| record | transaction_id | 7 | 0.82% |
| record | city | 7 | 0.82% |
| record | policy_id | 6 | 0.70% |
| record | instructor | 6 | 0.70% |
| record | issue | 6 | 0.70% |
| record | signatories | 6 | 0.70% |
| record | unit | 6 | 0.70% |
| record | author | 6 | 0.70% |
| record | passenger_address | 6 | 0.70% |
| record | policyholder_name | 6 | 0.70% |
| record | passenger_phone | 6 | 0.70% |
| record | discovery.discoverer | 6 | 0.70% |
| record | diagnosis_code | 6 | 0.70% |
| record | location.country | 6 | 0.70% |
| record | monthly_rent | 6 | 0.70% |
| record | booth | 5 | 0.58% |
| record | priority | 5 | 0.58% |
| record | attendees | 5 | 0.58% |
| record | risk_level | 5 | 0.58% |
| record | asteroid.discovery.discoverer | 5 | 0.58% |
| record | mathematical_concepts | 5 | 0.58% |
| record | incentive | 5 | 0.58% |
| record | resolution_detail | 5 | 0.58% |
| record | url | 5 | 0.58% |
| record | substance | 5 | 0.58% |
| record | species.specific_epithet | 5 | 0.58% |
| record | address: Residential address of the patient | 5 | 0.58% |
| record | venue_id | 5 | 0.58% |
| record | ticket_price | 5 | 0.58% |
| record | city.coordinates.latitude | 5 | 0.58% |
| record | city.religion.religious_sites | 5 | 0.58% |
| record | city.coordinates.longitude | 5 | 0.58% |
| record | article.number | 5 | 0.58% |
| record | article.references.0.offenses | 5 | 0.58% |
| record | assigned_to | 5 | 0.58% |
| record | status: Current status of the guideline in practice | 5 | 0.58% |
| record | products | 5 | 0.58% |
| record | exchange_rate.to_currency | 5 | 0.58% |
| record | company.legal_type | 5 | 0.58% |
| record | company.quality_standards | 5 | 0.58% |
| record | company.production_capacity.frequency | 5 | 0.58% |
| record | review_summary.location_quality | 5 | 0.58% |
| record | hashtags | 5 | 0.58% |
| record | replies | 5 | 0.58% |
| record | location: Geographic origin of the content | 5 | 0.58% |
| record | ticket_id: Unique identifier for the incident ticket | 5 | 0.58% |
| record | lease_id | 5 | 0.58% |
| record | game_id: Unique identifier for the game | 5 | 0.58% |
| record | provider_name: Name of the attending physician | 5 | 0.58% |
| record | product_launch | 5 | 0.58% |
| record | geographic_region | 5 | 0.58% |
| record | ticker | 5 | 0.58% |
| record | date: Start and end date range of the tournament | 5 | 0.58% |
| record | owner | 5 | 0.58% |
| record | origin | 5 | 0.58% |
| record | delay_reason | 5 | 0.58% |
| record | product.included_components.4.name | 5 | 0.58% |
| record | product.included_components.9.name | 5 | 0.58% |
| record | product.accessories | 5 | 0.58% |
| record | product.battery_life.unit | 5 | 0.58% |
| record | product.included_components.7.name | 5 | 0.58% |
| record | product.included_components.5.name | 5 | 0.58% |
| record | dates | 5 | 0.58% |
| record | clinical_id | 5 | 0.58% |
| record | player_id | 5 | 0.58% |
| record | position | 5 | 0.58% |
| record | review_id: Unique identifier for each review entry | 5 | 0.58% |
| record | species.taxonomy.original_publication.year | 5 | 0.58% |
| record | species.taxonomy.current_classification.author | 5 | 0.58% |
| record | species.characteristics.shell.inner_margin | 5 | 0.58% |
| record | species.common_name | 5 | 0.58% |
| record | species.characteristics.shell.periostracum | 5 | 0.58% |
| record | financial_discussion.person.division | 5 | 0.58% |
| record | legend.outcome | 5 | 0.58% |
| record | legend.person.relation | 5 | 0.58% |
| record | legend.person.name | 5 | 0.58% |
| record | financial_figures.bank | 5 | 0.58% |
| record | Anatoly Pavlovich Sukhanov <> age | 5 | 0.58% |
| record | Kozintsev <> introduced | 5 | 0.58% |
| record | Belkin <> friend of | 5 | 0.58% |
| record | Sukhanov <> member of | 5 | 0.58% |
| record | heart_rate | 5 | 0.58% |
| record | album.tracks.sideB.1.title | 5 | 0.58% |
| record | album.tracks.sideA.5.title | 5 | 0.58% |
| record | album.tracks.sideB.3.title | 5 | 0.58% |
| record | album.tracks.sideB.2.trackNumber | 5 | 0.58% |
| record | statement_type | 5 | 0.58% |
| record | firm.name | 5 | 0.58% |
| record | attorney_benefits | 5 | 0.58% |
| record | estate_planning.benefits_of_planning | 5 | 0.58% |
| record | battery_life | 5 | 0.58% |
| record | review_id | 5 | 0.58% |
| record | after_angelus.greetings.0.text | 5 | 0.58% |
| record | medication.dosing_schedule.0.dose_fraction | 5 | 0.58% |
| record | asteroid.discovery.discoverer.nationality | 5 | 0.58% |
| record | case_study.radiographic_findings.visual_features.definition | 5 | 0.58% |
| record | case_study.clinical_examination.periodontal_pocket.base_type | 5 | 0.58% |
| record | case_study.treatment_and_management.surgical_procedure.tissues_removed | 5 | 0.58% |
| record | case_study.radiographic_findings.anatomical_extension.superior_limit | 5 | 0.58% |
| record | lesion_classification.general_characteristics.typical_age_range | 5 | 0.58% |
| record | outcome: Measurable result or improvement observed | 5 | 0.58% |
| record | price_information | 5 | 0.58% |
| record | hotel_name | 5 | 0.58% |
| record | discovery_year | 5 | 0.58% |
| record | reason | 5 | 0.58% |
| record | word | 5 | 0.58% |
| record | discovery.location.name | 5 | 0.58% |
| record | discovery.date | 5 | 0.58% |
| record | physical_characteristics.diameter_unit | 5 | 0.58% |
| record | csds_id | 5 | 0.58% |
| record | statement.event_details.dates | 5 | 0.58% |
| record | statement.time | 5 | 0.58% |
| record | review.duration.reason | 5 | 0.58% |
| record | transfer_fee | 5 | 0.58% |
| record | name: Unique identifier for the smart device model | 5 | 0.58% |
| record | price: Listed price of the device in dollars | 5 | 0.58% |
| record | placement | 5 | 0.58% |
| record | employees | 5 | 0.58% |
| record | office_location | 5 | 0.58% |
| record | creator_address | 5 | 0.58% |
| record | shares | 5 | 0.58% |
| record | industry | 5 | 0.58% |
| record | organs | 5 | 0.58% |
| record | event.casualties | 5 | 0.58% |
| record | source.name | 5 | 0.58% |
| record | event.impact.target | 5 | 0.58% |
| record | root_cause | 5 | 0.58% |
| record | availability | 5 | 0.58% |
| record | plaintiff_name: Name of the individual or entity filing the case | 5 | 0.58% |
| record | topic: subject area of the competitor analysis | 5 | 0.58% |
| record | river_details.discharge.unit | 5 | 0.58% |
| record | river_details.countries | 5 | 0.58% |
| record | icd10code: Clinical code assigned to the reported symptoms | 5 | 0.58% |
| record | patient_address: Residential address of the patient | 5 | 0.58% |
| record | treatment: Medical intervention or therapy administered | 5 | 0.58% |
| record | date: Date of the financial report or update | 5 | 0.58% |
| record | guidance: Projected growth or performance forecast | 5 | 0.58% |
| record | asteroid_belt | 5 | 0.58% |
| record | discovery.observatory.location.country | 5 | 0.58% |
| record | reported_amount: Monetary value reported in the incident, if applicable | 5 | 0.58% |
| record | ip_address: IP address associated with the phishing email source | 5 | 0.58% |
| record | discovery.location.state | 5 | 0.58% |
| record | named_after.name | 5 | 0.58% |
| record | discharge_date | 5 | 0.58% |
| record | customer_name: Name of the customer making the purchase | 5 | 0.58% |
| record | description: Detailed description of the purchased item or service | 5 | 0.58% |
| record | employee_name | 5 | 0.58% |
| record | participant | 5 | 0.58% |
| record | opposing_members | 5 | 0.58% |
| record | location: City and state or province of patient residence | 5 | 0.58% |
| record | physician: Name of attending physician | 5 | 0.58% |
| record | qype_features | 5 | 0.58% |
| record | tracking_id | 5 | 0.58% |
| record | sector | 5 | 0.58% |
| record | enrollment: Number of currently enrolled students | 5 | 0.58% |
| record | meeting_times: Time range when class sessions occur | 5 | 0.58% |
| record | heatstroke.first_aid | 5 | 0.58% |
| record | family | 5 | 0.58% |
| record | registrant_name | 5 | 0.58% |
| record | evaluation.pros | 5 | 0.58% |
| record | sentiment: Emotional tone of the review | 5 | 0.58% |
| record | pr_number: Unique identifier for the pull request | 5 | 0.58% |
| record | key_enzyme: Central enzyme in the pathway activation | 5 | 0.58% |
| record | pathway | 5 | 0.58% |
| record | research_institution | 5 | 0.58% |
| record | exhibitor_id | 4 | 0.47% |
| record | status: Current operational status of the product | 4 | 0.47% |
| record | research_contribution | 4 | 0.47% |
| record | student_name | 4 | 0.47% |
| record | renewal_amount | 4 | 0.47% |
| record | host_id | 4 | 0.47% |
| record | entry_price | 4 | 0.47% |
| record | report_date | 4 | 0.47% |
| record | asteroid.named_after.nationality | 4 | 0.47% |
| record | formal_systems | 4 | 0.47% |
| record | theorems_and_paradoxes | 4 | 0.47% |
| record | advertisements | 4 | 0.47% |
| record | bracket_id | 4 | 0.47% |
| record | confirmed_teams | 4 | 0.47% |
| record | pending_teams | 4 | 0.47% |
| record | venue | 4 | 0.47% |
| record | broadcast_platforms | 4 | 0.47% |
| record | operations_team_size | 4 | 0.47% |
| record | data_dashboard_update_interval | 4 | 0.47% |
| record | issue_date | 4 | 0.47% |
| record | page_reference | 4 | 0.47% |
| record | clause | 4 | 0.47% |
| record | capacity | 4 | 0.47% |
| record | message_id | 4 | 0.47% |
| record | subject | 4 | 0.47% |
| record | filing_location: City and state where patent was filed | 4 | 0.47% |
| record | status: Current status of the patent application | 4 | 0.47% |
| record | description: Brief textual description of the transaction | 4 | 0.47% |
| record | transaction_id: Unique identifier for each spending event | 4 | 0.47% |
| record | merchant_category: Broad category of the merchant activity | 4 | 0.47% |
| record | protein | 4 | 0.47% |
| record | close_price: Final closing price of the stock | 4 | 0.47% |
| record | company_name: Full name of the company as referenced in the passage | 4 | 0.47% |
| record | warehouse_location | 4 | 0.47% |
| record | monitoring_frequency | 4 | 0.47% |
| record | parties | 4 | 0.47% |
| record | species.full_name | 4 | 0.47% |
| record | species.publication.year | 4 | 0.47% |
| record | species.taxonomy.subfamily | 4 | 0.47% |
| record | status: Current status of the vaccination series | 4 | 0.47% |
| record | patient_id: Unique identifier assigned to the vaccination record | 4 | 0.47% |
| record | genre | 4 | 0.47% |
| record | numberofslots | 4 | 0.47% |
| record | handling_method: Method used to handle the chemical | 4 | 0.47% |
| record | hazard_class: Hazard classification of the chemical substance | 4 | 0.47% |
| record | chemical_id: Unique identifier for each chemical substance | 4 | 0.47% |
| record | city.climate.description | 4 | 0.47% |
| record | city.ethnic_groups | 4 | 0.47% |
| record | city.tourism.nearby_attractions | 4 | 0.47% |
| record | city.religion.diversity | 4 | 0.47% |
| record | city.population.description | 4 | 0.47% |
| record | city.country | 4 | 0.47% |
| record | city.transportation.airport.purpose | 4 | 0.47% |
| record | city.name | 4 | 0.47% |
| record | city.distance_from_capital.capital_city | 4 | 0.47% |
| record | subsection.description | 4 | 0.47% |
| record | article.references.0.date | 4 | 0.47% |
| record | article.references.0.statute_name | 4 | 0.47% |
| record | guideline: Title of the medical guideline being updated | 4 | 0.47% |
| record | jurisdiction: Legal jurisdiction governing the clause | 4 | 0.47% |
| record | financial_amount: Monetary value associated with the clause | 4 | 0.47% |
| record | shop_name | 4 | 0.47% |
| record | shipping_information.bank_transfer_details.preferred_bank.country | 4 | 0.47% |
| record | shipping_information.bank_transfer_details.preferred_bank.fee_remark | 4 | 0.47% |
| record | shipping_information.bank_transfer_details.sepa_transfer_fee.other_banks | 4 | 0.47% |
| record | mission.goal | 4 | 0.47% |
| record | company.location | 4 | 0.47% |
| record | company.founders.0.number | 4 | 0.47% |
| record | review_summary.room_description | 4 | 0.47% |
| record | booking_information.booking_partners | 4 | 0.47% |
| record | attack_method | 4 | 0.47% |
| record | submission_date | 4 | 0.47% |
| record | handle: Social media handle associated with the creator | 4 | 0.47% |
| record | followupinstruction | 4 | 0.47% |
| record | incident_time: Time of incident occurrence in 24-hour format | 4 | 0.47% |
| record | game_status: Current status of the game | 4 | 0.47% |
| record | date: Date the game was played | 4 | 0.47% |
| record | awayteampoints: Points scored by the away team | 4 | 0.47% |
| record | home_team: Team playing at home | 4 | 0.47% |
| record | topic | 4 | 0.47% |
| record | patient_id: Unique identifier for the patient | 4 | 0.47% |
| record | visit_date: Date of the clinical encounter | 4 | 0.47% |
| record | provider_address: Physical address of the provider | 4 | 0.47% |
| record | visit_type: Type of visit such as follow-up or urgent | 4 | 0.47% |
| record | pain_level: Patient-reported pain intensity on a scale | 4 | 0.47% |
| record | comments: Total number of comments on the video | 4 | 0.47% |
| record | qoq_growth | 4 | 0.47% |
| record | net_profit_m | 4 | 0.47% |
| record | market_cap_b | 4 | 0.47% |
| record | call_transcript | 4 | 0.47% |
| record | organization.name | 4 | 0.47% |
| record | scholarship_program.goal | 4 | 0.47% |
| record | scholarship_program.conferences.purpose | 4 | 0.47% |
| record | organization.partners | 4 | 0.47% |
| record | scholarship_program.annual_graduation_size | 4 | 0.47% |
| record | operational_status | 4 | 0.47% |
| record | seed: Initial ranking of the tournament seed | 4 | 0.47% |
| record | policy_type | 4 | 0.47% |
| record | exposure | 4 | 0.47% |
| record | route | 4 | 0.47% |
| record | carrier | 4 | 0.47% |
| record | passenger_contact | 4 | 0.47% |
| record | value | 4 | 0.47% |
| record | sender_address | 4 | 0.47% |
| record | manifest_id | 4 | 0.47% |
| record | product.included_components.3.name | 4 | 0.47% |
| record | product.measured_parameters | 4 | 0.47% |
| record | product.connectivity | 4 | 0.47% |
| record | product.features | 4 | 0.47% |
| record | product.included_components.0.name | 4 | 0.47% |
| record | product.software | 4 | 0.47% |
| record | registration_cost | 4 | 0.47% |
| record | user_base | 4 | 0.47% |
| record | pain_level | 4 | 0.47% |
| record | related_movements | 4 | 0.47% |
| record | editor: Name of the magazine editor responsible | 4 | 0.47% |
| record | words: Word count of the review content | 4 | 0.47% |
| record | type: Genre or tone of the review | 4 | 0.47% |
| record | topic: Central theme or subject of the review | 4 | 0.47% |
| record | imaging_findings | 4 | 0.47% |
| record | visit_date | 4 | 0.47% |
| record | company_information.writer_requirements.experience.description | 4 | 0.47% |
| record | service_types | 4 | 0.47% |
| record | presentation_benefits | 4 | 0.47% |
| record | company_information.writer_qualities | 4 | 0.47% |
| record | company_information.writer_requirements.knowledge | 4 | 0.47% |
| record | species.taxonomy.original_publication.original_name | 4 | 0.47% |
| record | species.scientific_name | 4 | 0.47% |
| record | species.distribution_habitat_lifestyle.geographic_distribution | 4 | 0.47% |
| record | species.references.0.details | 4 | 0.47% |
| record | species.characteristics.shell.ligament | 4 | 0.47% |
| record | species.order | 4 | 0.47% |
| record | species.distribution_habitat_lifestyle.habitat | 4 | 0.47% |
| record | species.characteristics.shell.hinge | 4 | 0.47% |
| record | species.references.0.type | 4 | 0.47% |
| record | species.characteristics.shell.surface | 4 | 0.47% |
| record | device_id | 4 | 0.47% |
| record | maintenance_notes | 4 | 0.47% |
| record | financial_figures.latin_america_profit_percentage | 4 | 0.47% |
| record | legend.beggar.action | 4 | 0.47% |
| record | financial_figures.business_volume | 4 | 0.47% |
| record | legend.secretary.role | 4 | 0.47% |
| record | financial_discussion.latin_america_examples | 4 | 0.47% |
| record | legend.person.location | 4 | 0.47% |
| record | financial_discussion.person.bank | 4 | 0.47% |
| record | financial_discussion.concepts | 4 | 0.47% |
| record | Cynthia Grenier <> sent | 4 | 0.47% |
| record | Cynthia Grenier <> attended | 4 | 0.47% |
| record | first film <> made during | 4 | 0.47% |
| record | Babel <> rehabilitated in | 4 | 0.47% |
| record | Grigory Kozintsev <> made | 4 | 0.47% |
| record | Belkin <> kept | 4 | 0.47% |
| record | album.tracks.sideB.5.trackNumber | 4 | 0.47% |
| record | album.tracks.sideA.1.trackNumber | 4 | 0.47% |
| record | album.tracks.sideA.5.trackNumber | 4 | 0.47% |
| record | album.tracks.sideB.4.trackNumber | 4 | 0.47% |
| record | album.artist | 4 | 0.47% |
| record | product_id: Unique identifier for the product version | 4 | 0.47% |
| record | sku_id | 4 | 0.47% |
| record | property_code | 4 | 0.47% |
| record | tenant_name | 4 | 0.47% |
| record | headliner | 4 | 0.47% |
| record | estate_planning.conservatorship.duties | 4 | 0.47% |
| record | estate_planning.conservatorship.reasons_for_conservatorship | 4 | 0.47% |
| record | estate_planning.incapacity.assessment_criteria | 4 | 0.47% |
| record | purchase_price | 4 | 0.47% |
| record | greeting | 4 | 0.47% |
| record | after_angelus.greetings.1.text | 4 | 0.47% |
| record | speaker.name | 4 | 0.47% |
| record | service_description | 4 | 0.47% |
| record | medication.agent | 4 | 0.47% |
| record | technique_name | 4 | 0.47% |
| record | policyholder_name: Name of the policyholder receiving the renewal notice | 4 | 0.47% |
| record | feedback | 4 | 0.47% |
| record | asteroid.name_latin | 4 | 0.47% |
| record | product.applications | 4 | 0.47% |
| record | product.data_storage | 4 | 0.47% |
| record | product.included_components.10.name | 4 | 0.47% |
| record | case_study.radiographic_findings.visual_features.opacity | 4 | 0.47% |
| record | case_study.patient_demographics.age | 4 | 0.47% |
| record | lesion_classification.general_characteristics.growth_rate | 4 | 0.47% |
| record | case_study.radiographic_findings.modality | 4 | 0.47% |
| record | case_study.patient_demographics.gender | 4 | 0.47% |
| record | case_study.treatment_and_management.follow_up.radiographic_evidence | 4 | 0.47% |
| record | lesion_classification.general_characteristics.typical_location | 4 | 0.47% |
| record | differential_diagnosis | 4 | 0.47% |
| record | case_study.clinical_history.duration_of_awareness | 4 | 0.47% |
| record | case_study.clinical_examination.vitality_tests.1.tooth | 4 | 0.47% |
| record | case_study.radiographic_findings.visual_features.borders | 4 | 0.47% |
| record | lesion_classification.primary_diagnosis | 4 | 0.47% |
| record | etiology_and_histogenesis.histological_notes.inflammatory_status | 4 | 0.47% |
| record | case_study.treatment_and_management.post_operative_complications.0.final_resolution_method | 4 | 0.47% |
| record | case_study.clinical_examination.periodontal_pocket.contents | 4 | 0.47% |
| record | case_study.treatment_and_management.surgical_procedure.materials_used | 4 | 0.47% |
| record | case_study.treatment_and_management.follow_up.outcome | 4 | 0.47% |
| record | nearby_attractions | 4 | 0.47% |
| record | customer_information | 4 | 0.47% |
| record | copyright_information | 4 | 0.47% |
| record | priority: Severity level of the reported issue | 4 | 0.47% |
| record | activity | 4 | 0.47% |
| record | discovered_by | 4 | 0.47% |
| record | orbital_characteristics.classification | 4 | 0.47% |
| record | storage_conditions | 4 | 0.47% |
| record | handling_procedures | 4 | 0.47% |
| record | statement.event_details.embargo.destination | 4 | 0.47% |
| record | student_major | 4 | 0.47% |
| record | review_summary.breakfast_quality | 4 | 0.47% |
| record | review.hotel_name | 4 | 0.47% |
| record | review.value_for_money | 4 | 0.47% |
| record | review.city | 4 | 0.47% |
| record | support.donation_page | 4 | 0.47% |
| record | scholarship_program.annual_cohort_size | 4 | 0.47% |
| record | from_entity | 4 | 0.47% |
| record | annotation_id: Unique identifier for each clause annotation | 4 | 0.47% |
| record | passenger_email | 4 | 0.47% |
| record | customer | 4 | 0.47% |
| record | locations | 4 | 0.47% |
| record | estimatedresolutiontime: Number of business days to resolve the issue | 4 | 0.47% |
| record | lab_results | 4 | 0.47% |
| record | inventor.name | 4 | 0.47% |
| record | inventor.year | 4 | 0.47% |
| record | rules.bomb_carrier.placement | 4 | 0.47% |
| record | rules.bomb_carrier.activation | 4 | 0.47% |
| record | player_position | 4 | 0.47% |
| record | player_age | 4 | 0.47% |
| record | author.name | 4 | 0.47% |
| record | symptoms: List of symptoms reported in the checklist | 4 | 0.47% |
| record | statement.recommendations | 4 | 0.47% |
| record | galaxy.name | 4 | 0.47% |
| record | galaxy.alternative_designations | 4 | 0.47% |
| record | galaxy.type | 4 | 0.47% |
| record | physical_characteristics.observations.0.telescope | 4 | 0.47% |
| record | physical_characteristics.observations.0.year_operational | 4 | 0.47% |
| record | physical_characteristics.observations.0.publication_year | 4 | 0.47% |
| record | nation | 4 | 0.47% |
| record | revenue_contribution | 4 | 0.47% |
| record | transaction_value | 4 | 0.47% |
| record | target_company | 4 | 0.47% |
| record | time | 4 | 0.47% |
| record | claim_id | 4 | 0.47% |
| record | notes | 4 | 0.47% |
| record | key_achievement | 4 | 0.47% |
| record | quote | 4 | 0.47% |
| record | conditions_mentioned | 4 | 0.47% |
| record | species.characteristics.shell.shape | 4 | 0.47% |
| record | species.characteristics.shell.muscle_scars | 4 | 0.47% |
| record | acquiring_company | 4 | 0.47% |
| record | exposure_limit: Maximum allowable concentration in air | 4 | 0.47% |
| record | album.tracks.sideA.2.trackNumber | 4 | 0.47% |
| record | album.tracks.sideA.3.title | 4 | 0.47% |
| record | corrective_action | 4 | 0.47% |
| record | id | 4 | 0.47% |
| record | unit_type | 4 | 0.47% |
| record | court: Name of the court handling the case | 4 | 0.47% |
| record | defendant_name: Name of the individual or entity being sued | 4 | 0.47% |
| record | rank | 4 | 0.47% |
| record | result_id | 4 | 0.47% |
| record | time_period.age | 4 | 0.47% |
| record | time_period.end_ma | 4 | 0.47% |
| record | original_name.genus | 4 | 0.47% |
| record | original_name.author | 4 | 0.47% |
| record | current_name.genus | 4 | 0.47% |
| record | taxonomic_status | 4 | 0.47% |
| record | duration: Time period over which results were measured | 4 | 0.47% |
| record | vaccines: List of vaccine types administered to the patient | 4 | 0.47% |
| record | turnaround_days: average days from issuance to completion | 4 | 0.47% |
| record | company: name of the technology company conducting the brief | 4 | 0.47% |
| record | rental_unit | 4 | 0.47% |
| record | flooring | 4 | 0.47% |
| record | bedrooms | 4 | 0.47% |
| record | outdoor_space | 4 | 0.47% |
| record | severity | 4 | 0.47% |
| record | team_lead: individual responsible for overseeing the brief | 4 | 0.47% |
| record | funding_source: budget line funding the brief | 4 | 0.47% |
| record | creator_name | 4 | 0.47% |
| record | Kozintsev <> met | 4 | 0.47% |
| record | Shostakovitch <> composer of | 4 | 0.47% |
| record | river_name | 4 | 0.47% |
| record | river_details.whitewater_rating | 4 | 0.47% |
| record | river_details.length.unit | 4 | 0.47% |
| record | status: Current status of monitoring or follow-up | 4 | 0.47% |
| record | revenue_m | 4 | 0.47% |
| record | investor_sentiment_score | 4 | 0.47% |
| record | operating_expenses_m | 4 | 0.47% |
| record | dischargestatus: Patient's health status at discharge | 4 | 0.47% |
| record | patient_id: Unique identifier for each patient | 4 | 0.47% |
| record | diagnosis: Medical condition diagnosed upon admission | 4 | 0.47% |
| record | ceo: Chief executive officer of the company | 4 | 0.47% |
| record | revenue: Total annual revenue reported in dollar format | 4 | 0.47% |
| record | market_reaction: Market percentage change in response to the report | 4 | 0.47% |
| record | call_duration: Duration of the call in minutes | 4 | 0.47% |
| record | regional_focus: Geographic regions where the company operates | 4 | 0.47% |
| record | contact_name | 4 | 0.47% |
| record | discovery.observatory.location.city | 4 | 0.47% |
| record | discovery.observatory.location.region | 4 | 0.47% |
| record | price: Total price per night or for stay period | 4 | 0.47% |
| record | case_study.treatment_and_management.post_operative_complications.0.response_to_conservative_treatment | 4 | 0.47% |
| record | case_study.clinical_examination.periodontal_pocket.depth_description | 4 | 0.47% |
| record | prognosis_and_risks.neoplastic_transformation_potential | 4 | 0.47% |
| record | city.distance_from_capital.distance | 4 | 0.47% |
| record | city.distance_from_capital.unit | 4 | 0.47% |
| record | city.tourism.status | 4 | 0.47% |
| record | city.economy.key_industries | 4 | 0.47% |
| record | date_submitted: Date when the phishing report was submitted | 4 | 0.47% |
| record | report_id: Unique identifier for the phishing incident entry | 4 | 0.47% |
| record | search_context.topic | 4 | 0.47% |
| record | person.expertise | 4 | 0.47% |
| record | person.media_appearances.0.type | 4 | 0.47% |
| record | person.media_appearances.0.benefits | 4 | 0.47% |
| record | person.title | 4 | 0.47% |
| record | type | 4 | 0.47% |
| record | scientific_name | 4 | 0.47% |
| record | legend.beggar.description | 4 | 0.47% |
| record | customer_address: Full address of the customer | 4 | 0.47% |
| record | amount: Total charge amount in dollars | 4 | 0.47% |
| record | river_details.tributaries | 4 | 0.47% |
| record | river_details.activities | 4 | 0.47% |
| record | cloud_provider | 4 | 0.47% |
| record | severity: Symptom severity rated from 1 to 10 | 4 | 0.47% |
| record | diagnosis: Clinical diagnosis made by physician | 4 | 0.47% |
| record | semester: Academic semester when the course is offered | 4 | 0.47% |
| record | credits: Number of credit hours assigned to the course | 4 | 0.47% |
| record | person.description | 4 | 0.47% |
| record | person.media_appearances.0.content_description | 4 | 0.47% |
| record | prevention | 4 | 0.47% |
| record | heatstroke.risk_groups | 4 | 0.47% |
| record | heatstroke.definition | 4 | 0.47% |
| record | leasestartdate | 4 | 0.47% |
| record | reviewer_name: Name of the person who wrote the review | 4 | 0.47% |
| record | purchase_amount: Amount spent to purchase the app | 4 | 0.47% |
| record | reviewlengthwords: Number of words in the review text | 4 | 0.47% |
| record | review_id: Unique identifier for the review entry | 4 | 0.47% |
| record | linting_status: Result of linting rules check | 4 | 0.47% |
| record | year | 4 | 0.47% |
| record | team | 4 | 0.47% |
| record | order | 4 | 0.47% |
| record | experimental_phase: Current clinical or experimental stage of the pathway study | 4 | 0.47% |
| record | metabolite_mw: Molecular weight of the key metabolite in kDa or Da | 4 | 0.47% |
| record | name: Full title of the course including subject and topic | 4 | 0.47% |
| record | related_topics | 4 | 0.47% |
| record | heatstroke.when_to_seek_medical_attention | 4 | 0.47% |
| record | disclaimer | 4 | 0.47% |
| record | clause_type: Type of contractual obligation such as indemnification or confidentiality | 4 | 0.47% |
| record | description: Textual explanation of clause terms and conditions | 4 | 0.47% |
| record | case_title: Title of the legal case as filed | 4 | 0.47% |
| record | lead_researcher: Name of the principal researcher involved | 4 | 0.47% |
| record | company.history.0.event | 4 | 0.47% |
| record | party_name: Name of the party entering the agreement | 4 | 0.47% |
| record | symptoms: List of reported symptoms including fatigue, fever, etc | 4 | 0.47% |
| record | region: Geographic region where the review was written | 4 | 0.47% |
| record | rating: Star rating given to the application | 4 | 0.47% |
| record | rating: Star rating given by the reviewer | 4 | 0.47% |
| record | evaluation.rating | 4 | 0.47% |
| record | key_themes | 4 | 0.47% |
| record | course_name | 4 | 0.47% |
| record | credit_hours | 4 | 0.47% |
| record | level | 4 | 0.47% |
| record | region: Geographic region where the product is targeted | 3 | 0.35% |
| record | article_source | 3 | 0.35% |
| record | scholarship_id | 3 | 0.35% |
| record | issue_type | 3 | 0.35% |
| record | contact_method | 3 | 0.35% |
| record | target_email | 3 | 0.35% |
| record | spoofing_details | 3 | 0.35% |
| record | philosophical_positions | 3 | 0.35% |
| record | tag_archive | 3 | 0.35% |
| record | total_teams | 3 | 0.35% |
| record | match_duration | 3 | 0.35% |
| record | break_duration | 3 | 0.35% |
| record | backup_venue | 3 | 0.35% |
| record | credits | 3 | 0.35% |
| record | fee: Financial fee paid for filing the patent application | 3 | 0.35% |
| record | card_end: Last four digits of the associated payment card | 3 | 0.35% |
| record | annotation_id | 3 | 0.35% |
| record | theory | 3 | 0.35% |
| record | symbol: Stock ticker symbol as listed in the passage | 3 | 0.35% |
| record | species.taxonomy.family | 3 | 0.35% |
| record | dose_count: Number of doses administered per vaccine type | 3 | 0.35% |
| record | patient_name: Full name of the registered patient | 3 | 0.35% |
| record | issue_date: Date the safety document was issued | 3 | 0.35% |
| record | chemical_name: Name of the chemical compound | 3 | 0.35% |
| record | city.location_description | 3 | 0.35% |
| record | city.recent_developments | 3 | 0.35% |
| record | article.references.0.articles | 3 | 0.35% |
| record | institution: Hospital or clinic where the update was implemented | 3 | 0.35% |
| record | species.genus | 3 | 0.35% |
| record | species.description | 3 | 0.35% |
| record | shipping_information.bank_transfer_details.preferred_bank.bank_name | 3 | 0.35% |
| record | shipping_information.bank_transfer_details.preferred_bank.numeric_code | 3 | 0.35% |
| record | shipping_information.bank_transfer_details.sepa_transfer_fee.currency | 3 | 0.35% |
| record | exchange_rate.from_currency | 3 | 0.35% |
| record | company.name | 3 | 0.35% |
| record | company.customer_segment | 3 | 0.35% |
| record | company.production_capacity.unit | 3 | 0.35% |
| record | booking_information.hotel_name | 3 | 0.35% |
| record | retweets | 3 | 0.35% |
| record | creator_name: Name of the creator who posted the content | 3 | 0.35% |
| record | discharge_destination | 3 | 0.35% |
| record | assignee: Name of the team member assigned to the ticket | 3 | 0.35% |
| record | ceiling_height | 3 | 0.35% |
| record | away_team: Team playing away | 3 | 0.35% |
| record | region_focus | 3 | 0.35% |
| record | medication: Prescribed medication with dosage and frequency | 3 | 0.35% |
| record | tiktok_id: Unique identifier for the repost event | 3 | 0.35% |
| record | insurance_plan: Name of the insurance provider | 3 | 0.35% |
| record | trailing_eps | 3 | 0.35% |
| record | product_line | 3 | 0.35% |
| record | currency_code | 3 | 0.35% |
| record | ceo_name | 3 | 0.35% |
| record | key_issue_raised | 3 | 0.35% |
| record | call_duration | 3 | 0.35% |
| record | scholarship_program.conferences.frequency | 3 | 0.35% |
| record | academic_performance.passing_grade_to_university | 3 | 0.35% |
| record | tournament_id: Unique identifier for the tournament | 3 | 0.35% |
| record | host: Organization or company hosting the tournament | 3 | 0.35% |
| record | subsection.title | 3 | 0.35% |
| record | code_name | 3 | 0.35% |
| record | property_age | 3 | 0.35% |
| record | risk_category | 3 | 0.35% |
| record | soil_type | 3 | 0.35% |
| record | application_id | 3 | 0.35% |
| record | efficacy | 3 | 0.35% |
| record | dosage | 3 | 0.35% |
| record | event_name | 3 | 0.35% |
| record | character | 3 | 0.35% |
| record | product.included_components.1.name | 3 | 0.35% |
| record | product.target_users | 3 | 0.35% |
| record | product.included_components.8.name | 3 | 0.35% |
| record | product.included_components.1.specifications.wavelength_range | 3 | 0.35% |
| record | name: Official name of the esports tournament | 3 | 0.35% |
| record | location: City and country where tournament is held | 3 | 0.35% |
| record | monthly_price | 3 | 0.35% |
| record | original_club | 3 | 0.35% |
| record | book: Title of the book being reviewed | 3 | 0.35% |
| record | date: Date the review was published | 3 | 0.35% |
| record | species.characteristics.shell.mantle_line | 3 | 0.35% |
| record | species.taxonomy.original_publication.author | 3 | 0.35% |
| record | species.characteristics.shell.umbo | 3 | 0.35% |
| record | species.characteristics.shell.size | 3 | 0.35% |
| record | species.characteristics.shell.description | 3 | 0.35% |
| record | serial_number | 3 | 0.35% |
| record | legend.secretary.action | 3 | 0.35% |
| record | financial_figures.year | 3 | 0.35% |
| record | legend.person.bank | 3 | 0.35% |
| record | The Dream Life of Sukhanov <> written by | 3 | 0.35% |
| record | Cynthia Grenier <> saw | 3 | 0.35% |
| record | Isaac Babel <> author of | 3 | 0.35% |
| record | Hamlet <> film shown at | 3 | 0.35% |
| record | album.tracks.sideA.4.title | 3 | 0.35% |
| record | album.title | 3 | 0.35% |
| record | album.tracks.sideA.4.trackNumber | 3 | 0.35% |
| record | album.tracks.sideA.0.trackNumber | 3 | 0.35% |
| record | quantity_on_hand | 3 | 0.35% |
| record | property_address | 3 | 0.35% |
| record | firm.location.city | 3 | 0.35% |
| record | purchase_channel | 3 | 0.35% |
| record | reviewer_name | 3 | 0.35% |
| record | after_angelus.greetings.4.language | 3 | 0.35% |
| record | after_angelus.greetings.0.language | 3 | 0.35% |
| record | after_angelus.greetings.4.text | 3 | 0.35% |
| record | contact_information | 3 | 0.35% |
| record | medication.dosing_schedule.0.timing_offset_seconds | 3 | 0.35% |
| record | medication.concentration | 3 | 0.35% |
| record | policy_id: Unique identifier assigned to the policy renewal | 3 | 0.35% |
| record | renewal_cost: Monetary amount required for the policy renewal | 3 | 0.35% |
| record | address: Full residential address of the policyholder | 3 | 0.35% |
| record | reviewer | 3 | 0.35% |
| record | asteroid.discovery.observatory.location.distance_km | 3 | 0.35% |
| record | asteroid.discovery.date | 3 | 0.35% |
| record | product.name | 3 | 0.35% |
| record | product.battery_life.duration | 3 | 0.35% |
| record | product.included_components.2.name | 3 | 0.35% |
| record | case_study.radiographic_findings.anatomical_extension.lateral_relation | 3 | 0.35% |
| record | reviewer_id | 3 | 0.35% |
| record | institution: University or medical center conducting the trial | 3 | 0.35% |
| record | location: Geographic region where the trial was conducted | 3 | 0.35% |
| record | assigned_agent: Support agent responsible for resolving the ticket | 3 | 0.35% |
| record | company: Name of the customer's organization | 3 | 0.35% |
| record | actual_departure | 3 | 0.35% |
| record | report_type | 3 | 0.35% |
| record | physical_characteristics.observations.0.telescope_location | 3 | 0.35% |
| record | orbital_characteristics.classification_source | 3 | 0.35% |
| record | chemical_name | 3 | 0.35% |
| record | chemical_code | 3 | 0.35% |
| record | statement.subject | 3 | 0.35% |
| record | location.place | 3 | 0.35% |
| record | after_angelus.greetings.5.language | 3 | 0.35% |
| record | hotel_information.alternative_names | 3 | 0.35% |
| record | review_summary.reception_quality | 3 | 0.35% |
| record | review.stay_date | 3 | 0.35% |
| record | academic_performance.total_students | 3 | 0.35% |
| record | to_entity | 3 | 0.35% |
| record | exchange_rate | 3 | 0.35% |
| record | target_experience | 3 | 0.35% |
| record | thread_id | 3 | 0.35% |
| record | drugs | 3 | 0.35% |
| record | asteroid.discovery.discoverer.name | 3 | 0.35% |
| record | customer_name: Name of the customer filing the ticket | 3 | 0.35% |
| record | status: Current state of the ticket | 3 | 0.35% |
| record | asteroid.named_after.name | 3 | 0.35% |
| record | asteroid.named_after.profession | 3 | 0.35% |
| record | occupancy_type | 3 | 0.35% |
| record | rules.bomb_carrier.restriction | 3 | 0.35% |
| record | rules.capture_inspection | 3 | 0.35% |
| record | team_b | 3 | 0.35% |
| record | mentions_product_or_service | 3 | 0.35% |
| record | currency_pair | 3 | 0.35% |
| record | registration_date | 3 | 0.35% |
| record | note: Additional clinical or patient-reported notes about the checklist | 3 | 0.35% |
| record | exposure_response | 3 | 0.35% |
| record | statement.event_details.embargo.good | 3 | 0.35% |
| record | organizations | 3 | 0.35% |
| record | galaxy.constellation | 3 | 0.35% |
| record | event_type | 3 | 0.35% |
| record | deal_id | 3 | 0.35% |
| record | validity_period | 3 | 0.35% |
| record | final_score: Final score in format home-away | 3 | 0.35% |
| record | hometeampoints: Points scored by the home team | 3 | 0.35% |
| record | player_stats: List of player performance records | 3 | 0.35% |
| record | action | 3 | 0.35% |
| record | incident_type: Nature of the reported issue such as crash or outage | 3 | 0.35% |
| record | cause | 3 | 0.35% |
| record | repair_contractor | 3 | 0.35% |
| record | founding_year | 3 | 0.35% |
| record | species.characteristics.shell.symmetry | 3 | 0.35% |
| record | species.characteristics.shell.dorsal_margin | 3 | 0.35% |
| record | species.characteristics.shell.area | 3 | 0.35% |
| record | species.characteristics.shell.color | 3 | 0.35% |
| record | species.characteristics.similar_species.0.distinguishing_features | 3 | 0.35% |
| record | species.taxonomy.current_classification.genus | 3 | 0.35% |
| record | structure_ratio | 3 | 0.35% |
| record | after_angelus.greetings.2.language | 3 | 0.35% |
| record | after_angelus.greetings.1.language | 3 | 0.35% |
| record | after_angelus.greetings.5.text | 3 | 0.35% |
| record | facility: Location where chemical is stored or used | 3 | 0.35% |
| record | match | 3 | 0.35% |
| record | album.year | 3 | 0.35% |
| record | album.tracks.sideA.0.title | 3 | 0.35% |
| record | album.tracks.sideA.2.title | 3 | 0.35% |
| record | album.tracks.sideB.0.trackNumber | 3 | 0.35% |
| record | album.tracks.sideB.2.title | 3 | 0.35% |
| record | album.tracks.sideB.4.title | 3 | 0.35% |
| record | album.tracks.sideB.5.title | 3 | 0.35% |
| record | channel | 3 | 0.35% |
| record | resolution_method | 3 | 0.35% |
| record | tenant_profile | 3 | 0.35% |
| record | claim_amount: Monetary amount claimed in damages | 3 | 0.35% |
| record | docket_number: Unique identifier assigned to the case | 3 | 0.35% |
| record | time_period.epoch | 3 | 0.35% |
| record | time_period.start_ma | 3 | 0.35% |
| record | location.formation | 3 | 0.35% |
| record | location.region | 3 | 0.35% |
| record | original_name.species | 3 | 0.35% |
| record | airline | 3 | 0.35% |
| record | finding_id | 3 | 0.35% |
| record | responsible_party | 3 | 0.35% |
| record | brief_details: number of data points or metrics included in the analysis | 3 | 0.35% |
| record | date: fiscal date when the brief was initiated | 3 | 0.35% |
| record | data_sources: types of data used for the analysis | 3 | 0.35% |
| record | news_id | 3 | 0.35% |
| record | schema_name: unique identifier for each competitor brief | 3 | 0.35% |
| record | creator_city | 3 | 0.35% |
| record | author: Name of the reviewer writing the assessment | 3 | 0.35% |
| record | magazine: Name of the publication featuring the review | 3 | 0.35% |
| record | language: Language in which the review is written | 3 | 0.35% |
| record | amount: Monetary value of the transaction | 3 | 0.35% |
| record | software | 3 | 0.35% |
| record | 1959 film festival <> held in | 3 | 0.35% |
| record | internal_code | 3 | 0.35% |
| record | lead_contact | 3 | 0.35% |
| record | Boris Pasternak <> translator of | 3 | 0.35% |
| record | Cynthia Grenier <> lives in | 3 | 0.35% |
| record | Stalin <> leader of | 3 | 0.35% |
| record | river_details.usage | 3 | 0.35% |
| record | river_details.origin | 3 | 0.35% |
| record | river_details.source_elevation.unit | 3 | 0.35% |
| record | river_details.provinces | 3 | 0.35% |
| record | river_details.bridges | 3 | 0.35% |
| record | river_details.destination | 3 | 0.35% |
| record | checklist_id: Unique identifier assigned to each checklist instance | 3 | 0.35% |
| record | user_id | 3 | 0.35% |
| record | currency | 3 | 0.35% |
| record | call_status | 3 | 0.35% |
| record | version: Version number or date of the guideline source | 3 | 0.35% |
| record | eps: Earnings per share value as reported | 3 | 0.35% |
| record | description: Narrative summary of performance drivers or key points | 3 | 0.35% |
| record | provisional_designation | 3 | 0.35% |
| record | discovery.observatory.name | 3 | 0.35% |
| record | mechanism | 3 | 0.35% |
| record | hotel_name: Name of the hotel being reviewed | 3 | 0.35% |
| record | room_type: Type of room booked, e.g., standard, delux | 3 | 0.35% |
| record | address: Full address of the hotel location | 3 | 0.35% |
| record | case_study.treatment_and_management.surgical_procedure.type | 3 | 0.35% |
| record | case_study.clinical_examination.affected_teeth | 3 | 0.35% |
| record | case_study.radiographic_findings.anatomical_extension.inferior_limit | 3 | 0.35% |
| record | case_study.clinical_examination.vitality_tests.0.tooth | 3 | 0.35% |
| record | case_study.treatment_and_management.follow_up.duration | 3 | 0.35% |
| record | case_study.clinical_history.symptoms | 3 | 0.35% |
| record | city.transportation.airport.name | 3 | 0.35% |
| record | city.transportation.airport.status | 3 | 0.35% |
| record | city.climate.features | 3 | 0.35% |
| record | intellectual_influences | 3 | 0.35% |
| record | review.positive_aspects | 3 | 0.35% |
| record | organization: Company name associated with the victim or sender | 3 | 0.35% |
| record | person.name | 3 | 0.35% |
| record | statement.event_details.location | 3 | 0.35% |
| record | statement.eu_role | 3 | 0.35% |
| record | event_code | 3 | 0.35% |
| record | designation | 3 | 0.35% |
| record | discovery.observatory | 3 | 0.35% |
| record | discovery.location.city | 3 | 0.35% |
| record | named_after.nationality | 3 | 0.35% |
| record | condition | 3 | 0.35% |
| record | financial_discussion.person.name | 3 | 0.35% |
| record | financial_figures.clients | 3 | 0.35% |
| record | legend.moral | 3 | 0.35% |
| record | release_date | 3 | 0.35% |
| record | transaction_date: Date when the transaction occurred | 3 | 0.35% |
| record | store_name: Name of the store where purchase was made | 3 | 0.35% |
| record | store_address: Address of the store location | 3 | 0.35% |
| record | status: Payment transaction status | 3 | 0.35% |
| record | category: Category code for the purchase type | 3 | 0.35% |
| record | diagnosis | 3 | 0.35% |
| record | river_details.nearby_attractions | 3 | 0.35% |
| record | institution | 3 | 0.35% |
| record | medication.administration_type | 3 | 0.35% |
| record | group_label | 3 | 0.35% |
| record | outcome | 3 | 0.35% |
| record | impact | 3 | 0.35% |
| record | icd10code: ICD-10 diagnosis code assigned | 3 | 0.35% |
| record | inventor | 3 | 0.35% |
| record | name: Full name of the patient as reported | 3 | 0.35% |
| record | symptom_duration: Duration of symptoms as described | 3 | 0.35% |
| record | country_parties | 3 | 0.35% |
| record | instructor: Name of the course instructor or professor | 3 | 0.35% |
| record | total_cost | 3 | 0.35% |
| record | materials | 3 | 0.35% |
| record | issue_description | 3 | 0.35% |
| record | dehydration.emergency_actions.information_to_provide | 3 | 0.35% |
| record | dehydration.causes | 3 | 0.35% |
| record | currentstatus | 3 | 0.35% |
| record | location.city | 3 | 0.35% |
| record | client_name | 3 | 0.35% |
| record | score | 3 | 0.35% |
| record | storage_temperature | 3 | 0.35% |
| record | device_model: Model of the device used to write the review | 3 | 0.35% |
| record | features: Key functionalities such as motion detection or color control | 3 | 0.35% |
| record | place_name | 3 | 0.35% |
| record | unit_id | 3 | 0.35% |
| record | diagnosis: Medical condition diagnosed during the visit | 3 | 0.35% |
| record | case_id | 3 | 0.35% |
| record | scientific_description.described_by | 3 | 0.35% |
| record | affected_areas | 3 | 0.35% |
| record | impactradiuskm | 3 | 0.35% |
| record | pathway_id: Unique identifier for the signaling pathway | 3 | 0.35% |
| record | step_count: Number of steps in the pathway | 3 | 0.35% |
| record | institution: University or institution where research was conducted | 3 | 0.35% |
| record | room: Room number and building where class is held | 3 | 0.35% |
| record | inventor_address | 3 | 0.35% |
| record | dehydration.emergency_actions.call_emergency_services | 3 | 0.35% |
| record | species_name | 3 | 0.35% |
| record | id: Unique identifier for the contract clause | 3 | 0.35% |
| record | status: Current state of the agreement | 3 | 0.35% |
| record | address: Full address of the party involved | 3 | 0.35% |
| record | source_media | 3 | 0.35% |
| record | transfer_status | 3 | 0.35% |
| record | case_type: Nature of the legal dispute, such as breach or property issue | 3 | 0.35% |
| record | disease_association | 3 | 0.35% |
| record | primarycelltypes | 3 | 0.35% |
| record | user_handle | 3 | 0.35% |
| record | likes | 3 | 0.35% |
| record | merchant_name: Name of the business or service provider | 3 | 0.35% |
| record | offer_id | 3 | 0.35% |
| record | sender: Email address of the person who reported the phishing attempt | 3 | 0.35% |
| record | status: Current status of the incident review | 3 | 0.35% |
| record | loading_docks | 3 | 0.35% |
| record | vote_id | 3 | 0.35% |
| record | vote_type | 3 | 0.35% |
| record | asteroid.number | 3 | 0.35% |
| record | asteroid.name | 3 | 0.35% |
| record | discovery_year: Year the pathway was first identified | 3 | 0.35% |
| record | company.founders.0.role | 3 | 0.35% |
| record | values | 3 | 0.35% |
| record | review.return_reason | 3 | 0.35% |
| record | effective_date: Date when the clause becomes effective | 3 | 0.35% |
| record | amount: Monetary value of the clause in dollars | 3 | 0.35% |
| record | checklist_id: Unique identifier for each symptom checklist entry | 3 | 0.35% |
| record | purpose | 3 | 0.35% |
| record | pathway_name | 3 | 0.35% |
| record | service | 3 | 0.35% |
| record | case_study.clinical_examination.oral_hygiene_status | 3 | 0.35% |
| record | case_study.treatment_and_management.post_operative_complications.0.symptom | 3 | 0.35% |
| record | guarantees | 3 | 0.35% |
| record | estate_planning.conservatorship.definition | 3 | 0.35% |
| record | review_date: Date when the review was submitted | 3 | 0.35% |
| record | reviewer_name: Name of the user who posted the review | 3 | 0.35% |
| record | comments: Number of comments on the review | 3 | 0.35% |
| record | estate_planning.conservatorship.necessity | 3 | 0.35% |
| record | handle | 3 | 0.35% |
| record | financial_figures.loans | 3 | 0.35% |
| record | financial_figures.latin_america_assets_percentage | 3 | 0.35% |
| record | registration_code | 3 | 0.35% |
| record | host_city | 2 | 0.23% |
| record | monthly_premium | 2 | 0.23% |
| record | compliance_standard | 2 | 0.23% |
| record | newsletter_title | 2 | 0.23% |
| record | fee | 2 | 0.23% |
| record | festival_name | 2 | 0.23% |
| record | city.transportation.roads | 2 | 0.23% |
| record | shipping_information.payment_methods | 2 | 0.23% |
| record | shipping_information.bank_transfer_details.sepa_transfer_fee.fee_amount | 2 | 0.23% |
| record | company.establishment_year | 2 | 0.23% |
| record | fieldgoalpercentage: Percentage of field goals made, formatted with % sign | 2 | 0.23% |
| record | insurance_id: Identifier for the insurance plan | 2 | 0.23% |
| record | blood_pressure: Measured blood pressure values | 2 | 0.23% |
| record | margin_percent | 2 | 0.23% |
| record | call_date | 2 | 0.23% |
| record | path | 2 | 0.23% |
| record | coverage_limit | 2 | 0.23% |
| record | loss_ratio | 2 | 0.23% |
| record | underwriter | 2 | 0.23% |
| record | reroute_path | 2 | 0.23% |
| record | patient_name | 2 | 0.23% |
| record | age | 2 | 0.23% |
| record | company_information.writer_requirements.academic_performance | 2 | 0.23% |
| record | species.characteristics.shell.siphons | 2 | 0.23% |
| record | species.characteristics.shell.lunula | 2 | 0.23% |
| record | legend.beggar.request | 2 | 0.23% |
| record | album.tracks.sideB.0.title | 2 | 0.23% |
| record | album.tracks.sideA.3.trackNumber | 2 | 0.23% |
| record | product.description | 2 | 0.23% |
| record | product.included_components.2.specifications.wavelength_range | 2 | 0.23% |
| record | product.included_components.4.specifications.slit_size | 2 | 0.23% |
| record | product.included_components.6.name | 2 | 0.23% |
| record | case_study.radiographic_findings.visual_features.shape | 2 | 0.23% |
| record | etiology_and_histogenesis.histological_notes.epithelial_lining_description | 2 | 0.23% |
| record | prognosis_and_risks.recurrence_notes | 2 | 0.23% |
| record | pathway_id | 2 | 0.23% |
| record | cell_type | 2 | 0.23% |
| record | discovery.location.country | 2 | 0.23% |
| record | statement.opinion | 2 | 0.23% |
| record | academic_performance.government_sponsored | 2 | 0.23% |
| record | medical_conditions | 2 | 0.23% |
| record | enrollment_date | 2 | 0.23% |
| record | ticket_id: Unique identifier for each support case | 2 | 0.23% |
| record | asteroid.discovery.observatory.location.country | 2 | 0.23% |
| record | rules.bomb_carrier.effect | 2 | 0.23% |
| record | launch_date | 2 | 0.23% |
| record | asteroid.type | 2 | 0.23% |
| record | galaxy.catalog_membership | 2 | 0.23% |
| record | connectivity: Supported voice or wireless protocols like Alexa, Bluetooth | 2 | 0.23% |
| record | physical_characteristics.observations.0.wavelength | 2 | 0.23% |
| record | physical_characteristics.observations.0.telescope_type | 2 | 0.23% |
| record | established | 2 | 0.23% |
| record | team_a | 2 | 0.23% |
| record | support.call_to_action | 2 | 0.23% |
| record | species.characteristics.similar_species.0.species_name | 2 | 0.23% |
| record | species.taxonomy.current_classification.year | 2 | 0.23% |
| record | speaker | 2 | 0.23% |
| record | event.date | 2 | 0.23% |
| record | owner_comment | 2 | 0.23% |
| record | status: Current status of the safety document | 2 | 0.23% |
| record | album.tracks.sideA.1.title | 2 | 0.23% |
| record | album.tracks.sideB.1.trackNumber | 2 | 0.23% |
| record | album.tracks.sideB.3.trackNumber | 2 | 0.23% |
| record | creator | 2 | 0.23% |
| record | thumbnail_description | 2 | 0.23% |
| record | tags | 2 | 0.23% |
| record | current_name.species | 2 | 0.23% |
| record | passenger_age | 2 | 0.23% |
| record | ticket_number | 2 | 0.23% |
| record | score: Numerical rating given to the book | 2 | 0.23% |
| record | region: Geographic region associated with the review | 2 | 0.23% |
| record | event.impact.damage | 2 | 0.23% |
| record | Cynthia Grenier <> reviewed | 2 | 0.23% |
| record | instance_id | 2 | 0.23% |
| record | stock_price_change | 2 | 0.23% |
| record | risk_factor | 2 | 0.23% |
| record | company: Name of the company as listed in the report | 2 | 0.23% |
| record | status: Current status of the report or filing | 2 | 0.23% |
| record | orbital_characteristics.asteroid_family | 2 | 0.23% |
| record | case_study.radiographic_findings.diagnostic_aids | 2 | 0.23% |
| record | comments | 2 | 0.23% |
| record | city.region | 2 | 0.23% |
| record | city.economic_plan.plan_name | 2 | 0.23% |
| record | city.economic_plan.role_in_plan | 2 | 0.23% |
| record | city.economic_plan.future_developments | 2 | 0.23% |
| record | athlete_name | 2 | 0.23% |
| record | named_after.profession | 2 | 0.23% |
| record | authority.year | 2 | 0.23% |
| record | transaction_id: Unique identifier for each transaction | 2 | 0.23% |
| record | asteroid.named_after | 2 | 0.23% |
| record | discharge_date: Date patient was discharged from hospital | 2 | 0.23% |
| record | affected_service | 2 | 0.23% |
| record | code | 2 | 0.23% |
| record | course_code: Unique identifier for the course as assigned by department | 2 | 0.23% |
| record | person.attributes | 2 | 0.23% |
| record | author: Name of the medical professional who updated the guideline | 2 | 0.23% |
| record | approach_description | 2 | 0.23% |
| record | app_name: Name of the application being reviewed | 2 | 0.23% |
| record | contact_email | 2 | 0.23% |
| record | postmortemaction | 2 | 0.23% |
| record | general_themes | 2 | 0.23% |
| record | hazard_class | 2 | 0.23% |
| record | reviewer: Name of the person who reviewed the pull request | 2 | 0.23% |
| record | repo: Repository path where the PR is hosted | 2 | 0.23% |
| record | services | 2 | 0.23% |
| record | registration_number | 2 | 0.23% |
| record | Babel <> executed in | 2 | 0.23% |
| record | Kozintsev <> sent | 2 | 0.23% |
| record | host | 2 | 0.23% |
| record | agent | 2 | 0.23% |
| record | campus: Physical campus location where the course is held | 2 | 0.23% |
| record | meeting_days: Days of the week the class meets | 2 | 0.23% |
| record | last_updated | 2 | 0.23% |
| record | heatstroke.symptoms | 2 | 0.23% |
| record | scientific_description.year | 2 | 0.23% |
| record | new_club | 2 | 0.23% |
| record | themes_mentioned | 2 | 0.23% |
| record | deviceortool | 2 | 0.23% |
| record | dehydration.symptoms | 2 | 0.23% |
| record | supporting_members | 2 | 0.23% |
| record | treaty_name | 2 | 0.23% |
| record | review.duration.nights | 2 | 0.23% |
| record | patent_number: Unique identifier assigned to the patent application | 2 | 0.23% |
| record | company | 2 | 0.23% |
| record | status: Current state of the incident ticket | 2 | 0.23% |
| record | term | 2 | 0.23% |
| record | status: Current state of the symptom checklist | 2 | 0.23% |
| record | after_angelus.greetings.3.language | 2 | 0.23% |
| record | trader_name | 2 | 0.23% |
| record | key_components | 2 | 0.23% |
| record | river_details.water_level | 2 | 0.23% |
| record | case_study.radiographic_findings.visual_features.locularity | 2 | 0.23% |
| record | case_study.treatment_and_management.surgical_procedure.techniques | 2 | 0.23% |
| record | etiology_and_histogenesis.theories_of_origin | 2 | 0.23% |
| record | recipient: Email address of the victim affected by the phishing | 2 | 0.23% |
| record | company_information.company_name | 2 | 0.23% |
| record | battery_life: Duration of battery operation for devices with battery | 2 | 0.23% |
| record | key_metabolite: Primary metabolite produced by the pathway | 2 | 0.23% |
| record | study_id: Unique identifier assigned to the trial | 2 | 0.23% |
| record | kegg_id | 2 | 0.23% |
| record | followupdate: Scheduled date for next visit | 2 | 0.23% |
| record | company.history.0.year | 2 | 0.23% |
| record | company.employee_count | 2 | 0.23% |
| record | version: Product version number as specified in the passage | 2 | 0.23% |
| record | sideeffectrate | 2 | 0.23% |
| record | patient_name: Full name of the patient completing the form | 2 | 0.23% |
| record | content | 2 | 0.23% |
| record | admission_date: Date patient was admitted to hospital | 2 | 0.23% |
| record | plaintiff_address: Residential address of the plaintiff | 2 | 0.23% |
| record | legend.person.action | 2 | 0.23% |
| record | firm.location.state | 2 | 0.23% |
| record | semester | 2 | 0.23% |
| record | financial_discussion.person.title | 2 | 0.23% |
| record | authority.name | 2 | 0.23% |
| record | heatstroke.causes | 2 | 0.23% |
| record | lead_researcher: Name of principal investigator leading the study | 2 | 0.23% |
| record | intervention: Treatment or method used in the trial | 2 | 0.23% |
| record | patient_count | 2 | 0.23% |
| record | species.publication.author | 1 | 0.12% |
| record | shipping_information.bank_transfer_details.sepa_transfer_fee.condition | 1 | 0.12% |
| record | device_code | 1 | 0.12% |
| record | product.included_components.9.specifications.calibration_range | 1 | 0.12% |
| record | submission: Date the review was submitted | 1 | 0.12% |
| record | previous_year_comparison | 1 | 0.12% |
| record | age: Patient's age in years | 1 | 0.12% |
| record | resolution_timeline | 1 | 0.12% |
| record | regional_spread: Number of states or regions where the trend was repeated | 1 | 0.12% |
| record | resolution_time | 1 | 0.12% |
| record | statement.ministry | 1 | 0.12% |
| record | case_study.clinical_history.referral_reason | 1 | 0.12% |
| record | case_study.clinical_examination.gingival_condition | 1 | 0.12% |
| record | monthly_quantity | 1 | 0.12% |
| record | competitive_advantage | 1 | 0.12% |
| record | discovery.location.province | 1 | 0.12% |
| record | estate_planning.incapacity.definition | 1 | 0.12% |
| record | region: Geographic region where the incident occurred | 1 | 0.12% |
| record | status: Current status of the review entry | 1 | 0.12% |
| record | patient_age: Age of the patient at time of assessment | 1 | 0.12% |
| record | river_details.length.value | 1 | 0.12% |
| record | clause_id: Identifier of the clause being annotated | 1 | 0.12% |
| record | state | 1 | 0.12% |
| record | record_metadata | natural/status ×21, natural/title ×17, natural/location ×17, natural/name ×16 |  |

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | unknown |
| relations | unknown |
| events | unknown |
| classifications | unknown |
| structures | unknown |
