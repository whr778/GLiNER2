# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bionlp13cg | train | 2,936 | 2,936 | 10,132 | 0 | 0 | 0 | 0 | 0 |
| bionlp13cg | val | 964 | 964 | 3,396 | 0 | 0 | 0 | 0 | 0 |
| bionlp13cg | test | 1,829 | 1,829 | 6,395 | 0 | 0 | 0 | 0 | 0 |

## bionlp13cg

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2,936 | 2,936 | 26 | 44 | 183 | 10,132 | 0 | 0 | 0 | 0 | 0 |
| val | 964 | 964 | 25 | 44 | 97 | 3,396 | 0 | 0 | 0 | 0 | 0 |
| test | 1,829 | 1,829 | 25 | 44 | 145 | 6,395 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 2,865 | 945 | 1,778 | 5,588 | 97.54% |
| und | 70 | 19 | 51 | 140 | 2.44% |
| deu | 1 | 0 | 0 | 1 | 0.02% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 1,829 |
| test_unique_inputs | 1,829 |
| train_records | 2,936 |
| train_unique_inputs | 2,936 |
| val_records | 964 |
| val_unique_inputs | 964 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0470 | 10,132 / 3,396 | Cell 18%→15%, Simple_chemical 10%→12%, Gene_or_gene_product 36%→37%, Tissue 3%→3%, Cancer 12%→12% |
| entities (per mention) | train / test | 0.0495 | 10,132 / 6,395 | Cell 18%→15%, Cancer 12%→14%, Organism 8%→7%, Gene_or_gene_product 36%→36%, Multi-tissue_structure 4%→5% |
| entities (per mention) | val / test | 0.0408 | 3,396 / 6,395 | Simple_chemical 12%→11%, Cancer 12%→14%, Gene_or_gene_product 37%→36%, Organism 8%→7%, Organism_substance 1%→2% |

Thin support in val, entities (per mention): 2 label(s) under 10 gold. Anatomical_system ×3, Developing_anatomical_structure ×5.

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
| Gene_or_gene_product | 3,656 | 36.08% |
| Cell | 1,844 | 18.20% |
| Cancer | 1,169 | 11.54% |
| Simple_chemical | 1,025 | 10.12% |
| Organism | 848 | 8.37% |
| Multi-tissue_structure | 408 | 4.03% |
| Tissue | 311 | 3.07% |
| Cellular_component | 283 | 2.79% |
| Organ | 185 | 1.83% |
| Organism_substance | 142 | 1.40% |
| Pathological_formation | 89 | 0.88% |
| Immaterial_anatomical_entity | 51 | 0.50% |
| Organism_subdivision | 47 | 0.46% |
| Amino_acid | 40 | 0.39% |
| Anatomical_system | 21 | 0.21% |
| Developing_anatomical_structure | 13 | 0.13% |

#### val

| label | count | share |
| --- | --- | --- |
| Gene_or_gene_product | 1,252 | 36.87% |
| Cell | 510 | 15.02% |
| Simple_chemical | 424 | 12.49% |
| Cancer | 407 | 11.98% |
| Organism | 275 | 8.10% |
| Multi-tissue_structure | 137 | 4.03% |
| Cellular_component | 93 | 2.74% |
| Tissue | 85 | 2.50% |
| Organ | 70 | 2.06% |
| Pathological_formation | 42 | 1.24% |
| Organism_substance | 34 | 1.00% |
| Amino_acid | 28 | 0.82% |
| Immaterial_anatomical_entity | 19 | 0.56% |
| Organism_subdivision | 12 | 0.35% |
| Developing_anatomical_structure | 5 | 0.15% |
| Anatomical_system | 3 | 0.09% |

#### test

| label | count | share |
| --- | --- | --- |
| Gene_or_gene_product | 2,277 | 35.61% |
| Cell | 970 | 15.17% |
| Cancer | 876 | 13.70% |
| Simple_chemical | 677 | 10.59% |
| Organism | 472 | 7.38% |
| Multi-tissue_structure | 288 | 4.50% |
| Tissue | 180 | 2.81% |
| Cellular_component | 167 | 2.61% |
| Organ | 146 | 2.28% |
| Organism_substance | 98 | 1.53% |
| Pathological_formation | 86 | 1.34% |
| Amino_acid | 55 | 0.86% |
| Organism_subdivision | 38 | 0.59% |
| Immaterial_anatomical_entity | 31 | 0.48% |
| Anatomical_system | 17 | 0.27% |
| Developing_anatomical_structure | 17 | 0.27% |

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
