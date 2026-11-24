# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| craft | train | 8,291 | 8,291 | 25,567 | 0 | 0 | 0 | 0 | 0 |
| craft | val | 2,742 | 2,742 | 8,770 | 0 | 0 | 0 | 0 | 0 |
| craft | test | 5,652 | 5,652 | 17,097 | 0 | 0 | 0 | 0 | 0 |

## craft

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 8,291 | 8,291 | 27 | 49 | 426 | 25,567 | 0 | 0 | 0 | 0 | 0 |
| val | 2,742 | 2,742 | 30 | 53 | 273 | 8,770 | 0 | 0 | 0 | 0 | 0 |
| test | 5,652 | 5,652 | 28 | 51 | 327 | 17,097 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 7,886 | 2,632 | 5,336 | 15,854 | 95.02% |
| und | 399 | 105 | 306 | 810 | 4.85% |
| spa | 2 | 1 | 1 | 4 | 0.02% |
| nld | 1 | 0 | 2 | 3 | 0.02% |
| ita | 0 | 2 | 1 | 3 | 0.02% |
| fil | 1 | 1 | 0 | 2 | 0.01% |
| ceb | 0 | 1 | 1 | 2 | 0.01% |
| fra | 0 | 0 | 2 | 2 | 0.01% |
| fin | 0 | 0 | 2 | 2 | 0.01% |
| nor | 1 | 0 | 0 | 1 | 0.01% |
| bre | 1 | 0 | 0 | 1 | 0.01% |
| fry | 0 | 0 | 1 | 1 | 0.01% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 5,652 |
| test_unique_inputs | 5,652 |
| train_records | 8,291 |
| train_unique_inputs | 8,291 |
| val_records | 2,742 |
| val_unique_inputs | 2,742 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.1179 | 25,567 / 8,770 | SO 31%→39%, GGP 29%→23%, Taxon 11%→15%, GO 8%→5%, CL 10%→8% |
| entities (per mention) | train / test | 0.0239 | 25,567 / 17,097 | SO 31%→32%, GGP 29%→28%, CL 10%→9%, CHEBI 11%→11%, Taxon 11%→12% |
| entities (per mention) | val / test | 0.1009 | 8,770 / 17,097 | SO 39%→32%, GGP 23%→28%, Taxon 15%→12%, GO 5%→8%, CHEBI 10%→11% |


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
| SO | 7,882 | 30.83% |
| GGP | 7,326 | 28.65% |
| Taxon | 2,862 | 11.19% |
| CHEBI | 2,738 | 10.71% |
| CL | 2,654 | 10.38% |
| GO | 2,105 | 8.23% |

#### val

| label | count | share |
| --- | --- | --- |
| SO | 3,435 | 39.17% |
| GGP | 2,032 | 23.17% |
| Taxon | 1,284 | 14.64% |
| CHEBI | 841 | 9.59% |
| CL | 744 | 8.48% |
| GO | 434 | 4.95% |

#### test

| label | count | share |
| --- | --- | --- |
| SO | 5,499 | 32.16% |
| GGP | 4,710 | 27.55% |
| Taxon | 1,975 | 11.55% |
| CHEBI | 1,950 | 11.41% |
| CL | 1,610 | 9.42% |
| GO | 1,353 | 7.91% |

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
