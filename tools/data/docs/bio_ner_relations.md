# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bio_ner_relations | train | 2,084 | 2,084 | 11,327 | 5,925 | 0 | 0 | 0 | 0 |
| bio_ner_relations | val | 256 | 256 | 1,304 | 731 | 0 | 0 | 0 | 0 |
| bio_ner_relations | test | 258 | 258 | 1,083 | 709 | 0 | 0 | 0 | 0 |
| bio_ner_relations | all | 2,599 | 2,598 | 13,717 | 7,367 | 0 | 0 | 0 | 0 |

## bio_ner_relations

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2,084 | 2,084 | 31 | 75 | 12546 | 11,327 | 5,925 | 0 | 0 | 0 | 0 |
| val | 256 | 256 | 29 | 74 | 8751 | 1,304 | 731 | 0 | 0 | 0 | 0 |
| test | 258 | 258 | 29 | 65 | 7550 | 1,083 | 709 | 0 | 0 | 0 | 0 |
| all | 2,599 | 2,598 | 31 | 74 | 12546 | 13,717 | 7,367 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | all | records | share |
| --- | --- | --- | --- | --- | --- | --- |
| eng | 1,988 | 244 | 241 | 2,474 | 4,947 | 95.19% |
| und | 93 | 12 | 17 | 122 | 244 | 4.70% |
| epo | 1 | 0 | 0 | 1 | 2 | 0.04% |
| ilo | 1 | 0 | 0 | 1 | 2 | 0.04% |
| fra | 1 | 0 | 0 | 1 | 2 | 0.04% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| all_records | 2,599 |
| all_unique_inputs | 2,598 |
| test_records | 258 |
| test_unique_inputs | 258 |
| train_records | 2,084 |
| train_unique_inputs | 2,084 |
| val_records | 256 |
| val_unique_inputs | 256 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.1337 | 11,327 / 1,304 | GeneProtein 9%→4%, protein 18%→21%, Habitat 6%→4%, CellType 4%→6%, protein-family 5%→7% |
| entities (per mention) | train / test | 0.2087 WARN | 11,327 / 1,083 | GeneProtein 9%→0%, protein 18%→23%, Gene/protein/RNA 7%→10%, CellType 4%→1%, Gene 1%→3% |
| entities (per mention) | val / test | 0.1956 WARN | 1,304 / 1,083 | CellType 6%→1%, GeneProtein 4%→0%, Individual_protein 14%→17%, Anatomy 4%→1%, Gene/protein/RNA 7%→10% |
| relations (per instance) | train / val | 0.1293 | 5,925 / 731 | bind 54%→66%, PPI 29%→20%, Lives_In 12%→9%, Exhibits 4%→3%, genic_interaction 2%→2% |
| relations (per instance) | train / test | 0.0559 | 5,925 / 709 | Lives_In 12%→7%, bind 54%→58%, genic_interaction 2%→3%, PPI 29%→28%, Exhibits 4%→4% |
| relations (per instance) | val / test | 0.1022 | 731 / 709 | bind 66%→58%, PPI 20%→28%, genic_interaction 2%→3%, Lives_In 9%→7%, Exhibits 3%→4% |

Thin support in val, entities (per mention): 19 label(s) under 10 gold. fusion-protein ×1, drug ×1, protein-isoform ×1, organism ×1, process ×2, protein-motif ×2, mutation ×2, peptide ×2, organelle ×2, protein-region ×3, gene ×4, assay ×5, reagent ×5, amino-acid ×5, experiment-tag ×6, RNA ×6, Protein_family_or_group ×7, agent ×8, target ×8.
Unseen in train, present in test, entities (per mention): DNA_family_or_group ×3.
Thin support in test, entities (per mention): 21 label(s) under 10 gold. tissue ×1, protein-RNA-complex ×1, drug ×1, CellComponent ×2, GeneProtein ×2, organism ×2, amino-acid ×2, parameter ×2, Species ×3, organelle ×3, DNA_family_or_group ×3, Protein_complex ×3, experimental-construct ×4, process ×4, experiment-tag ×5, assay ×5, protein-isoform ×5, protein-region ×6, protein-motif ×7, Anatomy ×8, cell ×8.
Unseen in train, present in all, entities (per mention): DNA_family_or_group ×3.
Thin support in all, entities (per mention): 8 label(s) under 10 gold. tissue ×2, RNA-family ×2, protein-DNA-complex ×3, DNA_family_or_group ×3, parameter ×4, organism ×4, brand ×4, disease ×5.

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
| protein | 2,043 | 18.04% |
| Individual_protein | 1,743 | 15.39% |
| GeneProtein | 983 | 8.68% |
| Gene/protein/RNA | 769 | 6.79% |
| Habitat | 702 | 6.20% |
| protein-family | 588 | 5.19% |
| species | 577 | 5.09% |
| Microorganism | 445 | 3.93% |
| CellType | 432 | 3.81% |
| Anatomy | 326 | 2.88% |
| chemical | 299 | 2.64% |
| DNA | 289 | 2.55% |
| protein-complex | 241 | 2.13% |
| Phenotype | 205 | 1.81% |
| protein-domain | 175 | 1.54% |
| CellComponent | 119 | 1.05% |
| Species | 108 | 0.95% |
| target | 99 | 0.87% |
| Protein_family_or_group | 96 | 0.85% |
| agent | 92 | 0.81% |
| RNA | 91 | 0.80% |
| protein-motif | 87 | 0.77% |
| Gene | 85 | 0.75% |
| experimental-construct | 84 | 0.74% |
| cell | 83 | 0.73% |
| experiment-tag | 82 | 0.72% |
| gene | 82 | 0.72% |
| reagent | 74 | 0.65% |
| CellLine | 49 | 0.43% |
| amino-acid | 49 | 0.43% |
| protein-region | 43 | 0.38% |
| assay | 27 | 0.24% |
| organelle | 25 | 0.22% |
| peptide | 23 | 0.20% |
| fusion-protein | 23 | 0.20% |
| protein-isoform | 14 | 0.12% |
| process | 14 | 0.12% |
| Protein_complex | 12 | 0.11% |
| mutation | 11 | 0.10% |
| protein-RNA-complex | 11 | 0.10% |
| drug | 9 | 0.08% |
| disease | 5 | 0.04% |
| brand | 4 | 0.04% |
| protein-DNA-complex | 3 | 0.03% |
| parameter | 2 | 0.02% |
| RNA-family | 2 | 0.02% |
| tissue | 1 | 0.01% |
| organism | 1 | 0.01% |

#### val

| label | count | share |
| --- | --- | --- |
| protein | 278 | 21.32% |
| Individual_protein | 181 | 13.88% |
| protein-family | 91 | 6.98% |
| Gene/protein/RNA | 90 | 6.90% |
| CellType | 79 | 6.06% |
| species | 55 | 4.22% |
| Microorganism | 51 | 3.91% |
| Habitat | 51 | 3.91% |
| GeneProtein | 50 | 3.83% |
| Anatomy | 48 | 3.68% |
| chemical | 43 | 3.30% |
| DNA | 39 | 2.99% |
| protein-complex | 36 | 2.76% |
| Phenotype | 30 | 2.30% |
| protein-domain | 28 | 2.15% |
| Gene | 22 | 1.69% |
| cell | 14 | 1.07% |
| CellLine | 13 | 1.00% |
| experimental-construct | 12 | 0.92% |
| Species | 11 | 0.84% |
| CellComponent | 11 | 0.84% |
| agent | 8 | 0.61% |
| target | 8 | 0.61% |
| Protein_family_or_group | 7 | 0.54% |
| experiment-tag | 6 | 0.46% |
| RNA | 6 | 0.46% |
| assay | 5 | 0.38% |
| reagent | 5 | 0.38% |
| amino-acid | 5 | 0.38% |
| gene | 4 | 0.31% |
| protein-region | 3 | 0.23% |
| process | 2 | 0.15% |
| protein-motif | 2 | 0.15% |
| mutation | 2 | 0.15% |
| peptide | 2 | 0.15% |
| organelle | 2 | 0.15% |
| fusion-protein | 1 | 0.08% |
| drug | 1 | 0.08% |
| protein-isoform | 1 | 0.08% |
| organism | 1 | 0.08% |

#### test

| label | count | share |
| --- | --- | --- |
| protein | 251 | 23.18% |
| Individual_protein | 188 | 17.36% |
| Gene/protein/RNA | 105 | 9.70% |
| protein-family | 68 | 6.28% |
| Habitat | 59 | 5.45% |
| species | 45 | 4.16% |
| chemical | 41 | 3.79% |
| Gene | 37 | 3.42% |
| DNA | 30 | 2.77% |
| protein-domain | 29 | 2.68% |
| Microorganism | 29 | 2.68% |
| protein-complex | 22 | 2.03% |
| target | 19 | 1.75% |
| reagent | 13 | 1.20% |
| agent | 13 | 1.20% |
| gene | 12 | 1.11% |
| Phenotype | 12 | 1.11% |
| Protein_family_or_group | 12 | 1.11% |
| RNA | 11 | 1.02% |
| CellType | 10 | 0.92% |
| Anatomy | 8 | 0.74% |
| cell | 8 | 0.74% |
| protein-motif | 7 | 0.65% |
| protein-region | 6 | 0.55% |
| experiment-tag | 5 | 0.46% |
| assay | 5 | 0.46% |
| protein-isoform | 5 | 0.46% |
| experimental-construct | 4 | 0.37% |
| process | 4 | 0.37% |
| Species | 3 | 0.28% |
| organelle | 3 | 0.28% |
| DNA_family_or_group | 3 | 0.28% |
| Protein_complex | 3 | 0.28% |
| CellComponent | 2 | 0.18% |
| GeneProtein | 2 | 0.18% |
| organism | 2 | 0.18% |
| amino-acid | 2 | 0.18% |
| parameter | 2 | 0.18% |
| tissue | 1 | 0.09% |
| protein-RNA-complex | 1 | 0.09% |
| drug | 1 | 0.09% |

#### all

| label | count | share |
| --- | --- | --- |
| protein | 2,572 | 18.75% |
| Individual_protein | 2,112 | 15.40% |
| GeneProtein | 1,035 | 7.55% |
| Gene/protein/RNA | 964 | 7.03% |
| Habitat | 812 | 5.92% |
| protein-family | 749 | 5.46% |
| species | 677 | 4.94% |
| Microorganism | 525 | 3.83% |
| CellType | 521 | 3.80% |
| chemical | 383 | 2.79% |
| Anatomy | 382 | 2.78% |
| DNA | 358 | 2.61% |
| protein-complex | 300 | 2.19% |
| Phenotype | 247 | 1.80% |
| protein-domain | 232 | 1.69% |
| Gene | 144 | 1.05% |
| CellComponent | 132 | 0.96% |
| target | 126 | 0.92% |
| Species | 122 | 0.89% |
| Protein_family_or_group | 115 | 0.84% |
| agent | 113 | 0.82% |
| RNA | 108 | 0.79% |
| cell | 105 | 0.77% |
| experimental-construct | 100 | 0.73% |
| gene | 98 | 0.71% |
| protein-motif | 96 | 0.70% |
| experiment-tag | 93 | 0.68% |
| reagent | 92 | 0.67% |
| CellLine | 62 | 0.45% |
| amino-acid | 56 | 0.41% |
| protein-region | 52 | 0.38% |
| assay | 37 | 0.27% |
| organelle | 30 | 0.22% |
| peptide | 25 | 0.18% |
| fusion-protein | 24 | 0.17% |
| protein-isoform | 20 | 0.15% |
| process | 20 | 0.15% |
| Protein_complex | 15 | 0.11% |
| mutation | 13 | 0.09% |
| protein-RNA-complex | 12 | 0.09% |
| drug | 11 | 0.08% |
| disease | 5 | 0.04% |
| parameter | 4 | 0.03% |
| organism | 4 | 0.03% |
| brand | 4 | 0.03% |
| protein-DNA-complex | 3 | 0.02% |
| DNA_family_or_group | 3 | 0.02% |
| tissue | 2 | 0.01% |
| RNA-family | 2 | 0.01% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| bind | 3,173 | 53.55% |
| PPI | 1,722 | 29.06% |
| Lives_In | 682 | 11.51% |
| Exhibits | 221 | 3.73% |
| genic_interaction | 127 | 2.14% |

#### val

| label | count | share |
| --- | --- | --- |
| bind | 486 | 66.48% |
| PPI | 145 | 19.84% |
| Lives_In | 63 | 8.62% |
| Exhibits | 24 | 3.28% |
| genic_interaction | 13 | 1.78% |

#### test

| label | count | share |
| --- | --- | --- |
| bind | 410 | 57.83% |
| PPI | 198 | 27.93% |
| Lives_In | 50 | 7.05% |
| Exhibits | 27 | 3.81% |
| genic_interaction | 24 | 3.39% |

#### all

| label | count | share |
| --- | --- | --- |
| bind | 4,071 | 55.26% |
| PPI | 2,065 | 28.03% |
| Lives_In | 795 | 10.79% |
| Exhibits | 272 | 3.69% |
| genic_interaction | 164 | 2.23% |

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
