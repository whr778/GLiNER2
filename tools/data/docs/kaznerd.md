# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kaznerd | train | 47,536 | 47,536 | 84,666 | 0 | 0 | 0 | 0 | 0 |
| kaznerd | val | 5,884 | 5,884 | 10,524 | 0 | 0 | 0 | 0 | 0 |
| kaznerd | test | 5,961 | 5,961 | 10,553 | 0 | 0 | 0 | 0 | 0 |

## kaznerd

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 47,536 | 47,536 | 12 | 18 | 69 | 84,666 | 0 | 0 | 0 | 0 | 0 |
| val | 5,884 | 5,884 | 12 | 18 | 45 | 10,524 | 0 | 0 | 0 | 0 | 0 |
| test | 5,961 | 5,961 | 12 | 18 | 39 | 10,553 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| kaz | 45,675 | 5,630 | 5,699 | 57,004 | 96.00% |
| und | 1,042 | 135 | 144 | 1,321 | 2.22% |
| ukr | 531 | 73 | 77 | 681 | 1.15% |
| rus | 111 | 15 | 17 | 143 | 0.24% |
| kir | 82 | 18 | 15 | 115 | 0.19% |
| tat | 62 | 8 | 6 | 76 | 0.13% |
| bel | 13 | 2 | 1 | 16 | 0.03% |
| mkd | 7 | 0 | 1 | 8 | 0.01% |
| bak | 5 | 1 | 0 | 6 | 0.01% |
| mon | 2 | 1 | 1 | 4 | 0.01% |
| mhr | 4 | 0 | 0 | 4 | 0.01% |
| bul | 2 | 1 | 0 | 3 | 0.01% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 5,961 |
| test_unique_inputs | 5,961 |
| train_records | 47,536 |
| train_unique_inputs | 47,536 |
| val_records | 5,884 |
| val_unique_inputs | 5,884 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0173 | 84,666 / 10,524 | PERSON 9%→9%, GPE 13%→13%, NORP 3%→2%, POSITION 5%→5%, MONEY 3%→4% |
| entities (per mention) | train / test | 0.0207 | 84,666 / 10,553 | DATE 19%→20%, CARDINAL 22%→21%, ORGANISATION 6%→6%, PERSON 9%→9%, PERCENTAGE 3%→3% |
| entities (per mention) | val / test | 0.0244 | 10,524 / 10,553 | DATE 19%→20%, POSITION 5%→4%, CARDINAL 22%→21%, PERSON 9%→9%, NORP 2%→3% |


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
| CARDINAL | 18,392 | 21.72% |
| DATE | 15,896 | 18.77% |
| GPE | 10,829 | 12.79% |
| PERSON | 7,933 | 9.37% |
| ORGANISATION | 4,708 | 5.56% |
| POSITION | 3,869 | 4.57% |
| MONEY | 2,888 | 3.41% |
| PERCENTAGE | 2,709 | 3.20% |
| QUANTITY | 2,478 | 2.93% |
| ORDINAL | 2,429 | 2.87% |
| NORP | 2,359 | 2.79% |
| TIME | 1,438 | 1.70% |
| LOCATION | 1,381 | 1.63% |
| FACILITY | 1,357 | 1.60% |
| PROJECT | 1,349 | 1.59% |
| ART | 1,249 | 1.48% |
| EVENT | 1,066 | 1.26% |
| DISEASE | 814 | 0.96% |
| PRODUCT | 470 | 0.56% |
| LAW | 338 | 0.40% |
| LANGUAGE | 288 | 0.34% |
| MISCELLANEOUS | 160 | 0.19% |
| CONTACT | 138 | 0.16% |
| ADAGE | 122 | 0.14% |
| NON_HUMAN | 6 | 0.01% |

#### val

| label | count | share |
| --- | --- | --- |
| CARDINAL | 2,284 | 21.70% |
| DATE | 1,997 | 18.98% |
| GPE | 1,383 | 13.14% |
| PERSON | 905 | 8.60% |
| ORGANISATION | 603 | 5.73% |
| POSITION | 507 | 4.82% |
| MONEY | 384 | 3.65% |
| PERCENTAGE | 334 | 3.17% |
| ORDINAL | 306 | 2.91% |
| QUANTITY | 289 | 2.75% |
| NORP | 257 | 2.44% |
| FACILITY | 191 | 1.81% |
| PROJECT | 180 | 1.71% |
| TIME | 178 | 1.69% |
| LOCATION | 163 | 1.55% |
| ART | 148 | 1.41% |
| EVENT | 138 | 1.31% |
| DISEASE | 105 | 1.00% |
| PRODUCT | 60 | 0.57% |
| LAW | 47 | 0.45% |
| LANGUAGE | 25 | 0.24% |
| ADAGE | 16 | 0.15% |
| CONTACT | 12 | 0.11% |
| MISCELLANEOUS | 12 | 0.11% |

#### test

| label | count | share |
| --- | --- | --- |
| CARDINAL | 2,242 | 21.25% |
| DATE | 2,083 | 19.74% |
| GPE | 1,359 | 12.88% |
| PERSON | 954 | 9.04% |
| ORGANISATION | 628 | 5.95% |
| POSITION | 456 | 4.32% |
| MONEY | 379 | 3.59% |
| PERCENTAGE | 309 | 2.93% |
| ORDINAL | 299 | 2.83% |
| QUANTITY | 298 | 2.82% |
| NORP | 293 | 2.78% |
| TIME | 202 | 1.91% |
| LOCATION | 187 | 1.77% |
| FACILITY | 176 | 1.67% |
| PROJECT | 149 | 1.41% |
| ART | 136 | 1.29% |
| EVENT | 126 | 1.19% |
| DISEASE | 104 | 0.99% |
| PRODUCT | 56 | 0.53% |
| LAW | 35 | 0.33% |
| LANGUAGE | 35 | 0.33% |
| MISCELLANEOUS | 18 | 0.17% |
| ADAGE | 16 | 0.15% |
| CONTACT | 13 | 0.12% |

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
