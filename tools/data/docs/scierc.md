# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scierc | train | 265 | 265 | 3,740 | 2,422 | 0 | 0 | 0 | 0 |
| scierc | val | 46 | 46 | 741 | 467 | 0 | 0 | 0 | 0 |
| scierc | test | 38 | 38 | 506 | 330 | 0 | 0 | 0 | 0 |

## scierc

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 265 | 265 | 128 | 201 | 320 | 3,740 | 2,422 | 0 | 0 | 0 | 0 |
| val | 46 | 46 | 136 | 228 | 362 | 741 | 467 | 0 | 0 | 0 | 0 |
| test | 38 | 38 | 118 | 195 | 215 | 506 | 330 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 265 | 46 | 38 | 349 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 38 |
| test_unique_inputs | 38 |
| train_records | 265 |
| train_unique_inputs | 265 |
| val_records | 46 |
| val_unique_inputs | 46 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0360 | 3,740 / 741 | Task 17%→14%, Method 26%→27%, Metric 4%→5%, Material 10%→9%, OtherScientificTerm 28%→28% |
| entities (per mention) | train / test | 0.0473 | 3,740 / 506 | OtherScientificTerm 28%→25%, Metric 4%→6%, Task 17%→16%, Method 26%→27%, Generic 15%→16% |
| entities (per mention) | val / test | 0.0370 | 741 / 506 | OtherScientificTerm 28%→25%, Material 9%→11%, Task 14%→16%, Generic 15%→16%, Metric 5%→6% |
| relations (per instance) | train / val | 0.0781 | 2,422 / 467 | USED-FOR 53%→57%, EVALUATE-FOR 10%→6%, CONJUNCTION 13%→10%, PART-OF 6%→4%, FEATURE-OF 5%→7% |
| relations (per instance) | train / test | 0.0812 | 2,422 / 330 | USED-FOR 53%→46%, EVALUATE-FOR 10%→15%, FEATURE-OF 5%→7%, HYPONYM-OF 9%→10%, PART-OF 6%→5% |
| relations (per instance) | val / test | 0.1287 | 467 / 330 | USED-FOR 57%→46%, EVALUATE-FOR 6%→15%, CONJUNCTION 10%→13%, COMPARE 6%→5%, PART-OF 4%→5% |


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
| OtherScientificTerm | 1,043 | 27.89% |
| Method | 966 | 25.83% |
| Task | 649 | 17.35% |
| Generic | 562 | 15.03% |
| Material | 379 | 10.13% |
| Metric | 141 | 3.77% |

#### val

| label | count | share |
| --- | --- | --- |
| OtherScientificTerm | 211 | 28.48% |
| Method | 203 | 27.40% |
| Generic | 113 | 15.25% |
| Task | 107 | 14.44% |
| Material | 70 | 9.45% |
| Metric | 37 | 4.99% |

#### test

| label | count | share |
| --- | --- | --- |
| Method | 138 | 27.27% |
| OtherScientificTerm | 126 | 24.90% |
| Generic | 81 | 16.01% |
| Task | 79 | 15.61% |
| Material | 54 | 10.67% |
| Metric | 28 | 5.53% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| USED-FOR | 1,272 | 52.52% |
| CONJUNCTION | 310 | 12.80% |
| EVALUATE-FOR | 234 | 9.66% |
| HYPONYM-OF | 220 | 9.08% |
| PART-OF | 143 | 5.90% |
| COMPARE | 123 | 5.08% |
| FEATURE-OF | 120 | 4.95% |

#### val

| label | count | share |
| --- | --- | --- |
| USED-FOR | 267 | 57.17% |
| CONJUNCTION | 48 | 10.28% |
| HYPONYM-OF | 45 | 9.64% |
| FEATURE-OF | 31 | 6.64% |
| EVALUATE-FOR | 29 | 6.21% |
| COMPARE | 28 | 6.00% |
| PART-OF | 19 | 4.07% |

#### test

| label | count | share |
| --- | --- | --- |
| USED-FOR | 151 | 45.76% |
| EVALUATE-FOR | 50 | 15.15% |
| CONJUNCTION | 42 | 12.73% |
| HYPONYM-OF | 33 | 10.00% |
| FEATURE-OF | 22 | 6.67% |
| PART-OF | 17 | 5.15% |
| COMPARE | 15 | 4.55% |

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
