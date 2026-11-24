# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_ewe | train | 2,357 | 2,348 | 5,194 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_ewe | val | 469 | 466 | 1,145 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_ewe | test | 913 | 908 | 2,309 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_ewe

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2,357 | 2,348 | 18 | 31 | 76 | 5,194 | 0 | 0 | 0 | 0 | 0 |
| val | 469 | 466 | 18 | 30 | 59 | 1,145 | 0 | 0 | 0 | 0 | 0 |
| test | 913 | 908 | 19 | 34 | 93 | 2,309 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 1,913 | 415 | 774 | 3,102 | 82.96% |
| epo | 184 | 16 | 30 | 230 | 6.15% |
| fra | 96 | 15 | 49 | 160 | 4.28% |
| sun | 18 | 4 | 5 | 27 | 0.72% |
| pol | 12 | 4 | 5 | 21 | 0.56% |
| roh | 9 | 0 | 12 | 21 | 0.56% |
| mlt | 14 | 1 | 2 | 17 | 0.45% |
| ina | 12 | 1 | 4 | 17 | 0.45% |
| ido | 5 | 1 | 5 | 11 | 0.29% |
| fin | 6 | 3 | 2 | 11 | 0.29% |
| cor | 9 | 0 | 2 | 11 | 0.29% |
| jbo | 6 | 1 | 3 | 10 | 0.27% |
| scn | 7 | 2 | 0 | 9 | 0.24% |
| spa | 5 | 0 | 3 | 8 | 0.21% |
| ita | 5 | 1 | 2 | 8 | 0.21% |
| wln | 5 | 0 | 1 | 6 | 0.16% |
| hsb | 4 | 0 | 1 | 5 | 0.13% |
| eng | 3 | 1 | 1 | 5 | 0.13% |
| deu | 3 | 1 | 0 | 4 | 0.11% |
| cat | 2 | 0 | 2 | 4 | 0.11% |
| grn | 4 | 0 | 0 | 4 | 0.11% |
| mlg | 1 | 1 | 2 | 4 | 0.11% |
| swe | 3 | 0 | 0 | 3 | 0.08% |
| ast | 2 | 0 | 1 | 3 | 0.08% |
| war | 2 | 0 | 1 | 3 | 0.08% |
| ceb | 3 | 0 | 0 | 3 | 0.08% |
| tur | 2 | 0 | 0 | 2 | 0.05% |
| bre | 2 | 0 | 0 | 2 | 0.05% |
| swa | 1 | 0 | 1 | 2 | 0.05% |
| bih | 2 | 0 | 0 | 2 | 0.05% |
| jav | 2 | 0 | 0 | 2 | 0.05% |
| que | 1 | 0 | 1 | 2 | 0.05% |
| nds | 1 | 0 | 0 | 1 | 0.03% |
| glg | 1 | 0 | 0 | 1 | 0.03% |
| cbk | 1 | 0 | 0 | 1 | 0.03% |
| srp | 1 | 0 | 0 | 1 | 0.03% |
| pms | 1 | 0 | 0 | 1 | 0.03% |
| tuk | 1 | 0 | 0 | 1 | 0.03% |
| hun | 1 | 0 | 0 | 1 | 0.03% |
| asm | 1 | 0 | 0 | 1 | 0.03% |
| ltz | 1 | 0 | 0 | 1 | 0.03% |
| nap | 1 | 0 | 0 | 1 | 0.03% |
| gle | 1 | 0 | 0 | 1 | 0.03% |
| uzb | 1 | 0 | 0 | 1 | 0.03% |
| sco | 1 | 0 | 0 | 1 | 0.03% |
| glv | 1 | 0 | 0 | 1 | 0.03% |
| azb | 0 | 1 | 0 | 1 | 0.03% |
| ilo | 0 | 1 | 0 | 1 | 0.03% |
| srd | 0 | 0 | 1 | 1 | 0.03% |
| por | 0 | 0 | 1 | 1 | 0.03% |
| als | 0 | 0 | 1 | 1 | 0.03% |
| cym | 0 | 0 | 1 | 1 | 0.03% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 913 |
| test_unique_inputs | 908 |
| train_records | 2,357 |
| train_unique_inputs | 2,348 |
| val_records | 469 |
| val_unique_inputs | 466 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.1030 | 5,194 / 1,145 | location 43%→53%, organization 14%→10%, person 25%→22%, date 18%→16% |
| entities (per mention) | train / test | 0.1320 | 5,194 / 2,309 | location 43%→56%, person 25%→19%, organization 14%→10%, date 18%→15% |
| entities (per mention) | val / test | 0.0351 | 1,145 / 2,309 | location 53%→56%, person 22%→19%, date 16%→15%, organization 10%→10% |


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
| location | 2,223 | 42.80% |
| person | 1,304 | 25.11% |
| date | 923 | 17.77% |
| organization | 744 | 14.32% |

#### val

| label | count | share |
| --- | --- | --- |
| location | 608 | 53.10% |
| person | 247 | 21.57% |
| date | 179 | 15.63% |
| organization | 111 | 9.69% |

#### test

| label | count | share |
| --- | --- | --- |
| location | 1,293 | 56.00% |
| person | 442 | 19.14% |
| date | 336 | 14.55% |
| organization | 238 | 10.31% |

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
