# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bionlp13pc | train | 2,365 | 2,365 | 7,239 | 0 | 0 | 0 | 0 | 0 |
| bionlp13pc | val | 812 | 812 | 2,497 | 0 | 0 | 0 | 0 | 0 |
| bionlp13pc | test | 1,575 | 1,575 | 4,835 | 0 | 0 | 0 | 0 | 0 |

## bionlp13pc

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2,365 | 2,365 | 26 | 43 | 162 | 7,239 | 0 | 0 | 0 | 0 | 0 |
| val | 812 | 812 | 26 | 43 | 101 | 2,497 | 0 | 0 | 0 | 0 | 0 |
| test | 1,575 | 1,575 | 26 | 42 | 245 | 4,835 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 2,302 | 784 | 1,508 | 4,594 | 96.68% |
| und | 63 | 27 | 67 | 157 | 3.30% |
| tur | 0 | 1 | 0 | 1 | 0.02% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 1,575 |
| test_unique_inputs | 1,575 |
| train_records | 2,365 |
| train_unique_inputs | 2,365 |
| val_records | 812 |
| val_unique_inputs | 812 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0305 | 7,239 / 2,497 | Gene_or_gene_product 69%→66%, Simple_chemical 15%→17%, Cellular_component 6%→8%, Complex 9%→9% |
| entities (per mention) | train / test | 0.0231 | 7,239 / 4,835 | Gene_or_gene_product 69%→67%, Simple_chemical 15%→17%, Complex 9%→10%, Cellular_component 6%→7% |
| entities (per mention) | val / test | 0.0138 | 2,497 / 4,835 | Cellular_component 8%→7%, Complex 9%→10%, Gene_or_gene_product 66%→67%, Simple_chemical 17%→17% |


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
| Gene_or_gene_product | 4,988 | 68.90% |
| Simple_chemical | 1,113 | 15.38% |
| Complex | 674 | 9.31% |
| Cellular_component | 464 | 6.41% |

#### val

| label | count | share |
| --- | --- | --- |
| Gene_or_gene_product | 1,654 | 66.24% |
| Simple_chemical | 428 | 17.14% |
| Complex | 223 | 8.93% |
| Cellular_component | 192 | 7.69% |

#### test

| label | count | share |
| --- | --- | --- |
| Gene_or_gene_product | 3,220 | 66.60% |
| Simple_chemical | 816 | 16.88% |
| Complex | 481 | 9.95% |
| Cellular_component | 318 | 6.58% |

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
