# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| biored | train | 308 | 308 | 4,927 | 2,877 | 0 | 0 | 0 | 0 |
| biored | val | 47 | 47 | 711 | 459 | 0 | 0 | 0 | 0 |
| biored | test | 45 | 45 | 767 | 491 | 0 | 0 | 0 | 0 |

## biored

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 308 | 308 | 292 | 377 | 635 | 4,927 | 2,877 | 0 | 0 | 0 | 0 |
| val | 47 | 47 | 291 | 404 | 463 | 711 | 459 | 0 | 0 | 0 | 0 |
| test | 45 | 45 | 288 | 455 | 640 | 767 | 491 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 308 | 47 | 45 | 400 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 45 |
| test_unique_inputs | 45 |
| train_records | 308 |
| train_unique_inputs | 308 |
| val_records | 47 |
| val_unique_inputs | 47 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0478 | 4,927 / 711 | DiseaseOrPhenotypicFeature 29%→33%, ChemicalEntity 19%→16%, OrganismTaxon 10%→9%, GeneOrGeneProduct 31%→33%, SequenceVariant 10%→9% |
| entities (per mention) | train / test | 0.0671 | 4,927 / 767 | GeneOrGeneProduct 31%→37%, ChemicalEntity 19%→15%, SequenceVariant 10%→7%, DiseaseOrPhenotypicFeature 29%→31%, OrganismTaxon 10%→10% |
| entities (per mention) | val / test | 0.0559 | 711 / 767 | GeneOrGeneProduct 33%→37%, DiseaseOrPhenotypicFeature 33%→31%, SequenceVariant 9%→7%, OrganismTaxon 9%→10%, ChemicalEntity 16%→15% |
| relations (per instance) | train / val | 0.0412 | 2,877 / 459 | Association 51%→55%, Bind 2%→0%, Negative_Correlation 18%→17%, Comparison 1%→0%, Cotreatment 1%→0% |
| relations (per instance) | train / test | 0.0440 | 2,877 / 491 | Positive_Correlation 27%→24%, Negative_Correlation 18%→21%, Association 51%→53%, Cotreatment 1%→0%, Comparison 1%→1% |
| relations (per instance) | val / test | 0.0644 | 459 / 491 | Negative_Correlation 17%→21%, Positive_Correlation 27%→24%, Association 55%→53%, Bind 0%→1%, Comparison 0%→1% |

Thin support in val, entities (per mention): 1 label(s) under 10 gold. CellLine ×6.
Thin support in val, relations (per instance): 3 label(s) under 10 gold. Bind ×1, Drug_Interaction ×2, Cotreatment ×2.
Thin support in test, entities (per mention): 1 label(s) under 10 gold. CellLine ×6.
Thin support in test, relations (per instance): 4 label(s) under 10 gold. Drug_Interaction ×1, Cotreatment ×2, Comparison ×5, Bind ×7.

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
| GeneOrGeneProduct | 1,549 | 31.44% |
| DiseaseOrPhenotypicFeature | 1,438 | 29.19% |
| ChemicalEntity | 924 | 18.75% |
| OrganismTaxon | 498 | 10.11% |
| SequenceVariant | 474 | 9.62% |
| CellLine | 44 | 0.89% |

#### val

| label | count | share |
| --- | --- | --- |
| DiseaseOrPhenotypicFeature | 233 | 32.77% |
| GeneOrGeneProduct | 232 | 32.63% |
| ChemicalEntity | 116 | 16.32% |
| SequenceVariant | 62 | 8.72% |
| OrganismTaxon | 62 | 8.72% |
| CellLine | 6 | 0.84% |

#### test

| label | count | share |
| --- | --- | --- |
| GeneOrGeneProduct | 281 | 36.64% |
| DiseaseOrPhenotypicFeature | 234 | 30.51% |
| ChemicalEntity | 114 | 14.86% |
| OrganismTaxon | 79 | 10.30% |
| SequenceVariant | 53 | 6.91% |
| CellLine | 6 | 0.78% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| Association | 1,478 | 51.37% |
| Positive_Correlation | 783 | 27.22% |
| Negative_Correlation | 511 | 17.76% |
| Bind | 44 | 1.53% |
| Cotreatment | 27 | 0.94% |
| Comparison | 23 | 0.80% |
| Drug_Interaction | 8 | 0.28% |
| Conversion | 3 | 0.10% |

#### val

| label | count | share |
| --- | --- | --- |
| Association | 254 | 55.34% |
| Positive_Correlation | 124 | 27.02% |
| Negative_Correlation | 76 | 16.56% |
| Drug_Interaction | 2 | 0.44% |
| Cotreatment | 2 | 0.44% |
| Bind | 1 | 0.22% |

#### test

| label | count | share |
| --- | --- | --- |
| Association | 258 | 52.55% |
| Positive_Correlation | 116 | 23.63% |
| Negative_Correlation | 102 | 20.77% |
| Bind | 7 | 1.43% |
| Comparison | 5 | 1.02% |
| Cotreatment | 2 | 0.41% |
| Drug_Interaction | 1 | 0.20% |

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
