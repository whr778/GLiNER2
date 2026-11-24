# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_kin | train | 4,683 | 4,683 | 10,353 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_kin | val | 834 | 834 | 2,056 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_kin | test | 1,476 | 1,476 | 3,614 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_kin

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4,683 | 4,683 | 27 | 47 | 105 | 10,353 | 0 | 0 | 0 | 0 | 0 |
| val | 834 | 834 | 22 | 43 | 100 | 2,056 | 0 | 0 | 0 | 0 | 0 |
| test | 1,476 | 1,476 | 28 | 47 | 83 | 3,614 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 3,652 | 671 | 1,173 | 5,496 | 78.59% |
| swa | 441 | 62 | 120 | 623 | 8.91% |
| epo | 342 | 26 | 87 | 455 | 6.51% |
| tur | 26 | 23 | 4 | 53 | 0.76% |
| ceb | 27 | 1 | 8 | 36 | 0.51% |
| ido | 22 | 2 | 7 | 31 | 0.44% |
| fil | 18 | 3 | 8 | 29 | 0.41% |
| fin | 13 | 6 | 8 | 27 | 0.39% |
| sun | 14 | 6 | 6 | 26 | 0.37% |
| jav | 19 | 3 | 4 | 26 | 0.37% |
| cor | 15 | 1 | 5 | 21 | 0.30% |
| bre | 13 | 1 | 5 | 19 | 0.27% |
| eng | 5 | 8 | 4 | 17 | 0.24% |
| eus | 9 | 2 | 3 | 14 | 0.20% |
| pol | 5 | 5 | 3 | 13 | 0.19% |
| cat | 2 | 0 | 10 | 12 | 0.17% |
| jbo | 9 | 1 | 1 | 11 | 0.16% |
| ile | 7 | 2 | 1 | 10 | 0.14% |
| que | 8 | 1 | 0 | 9 | 0.13% |
| ilo | 4 | 0 | 5 | 9 | 0.13% |
| war | 6 | 0 | 2 | 8 | 0.11% |
| lat | 3 | 2 | 3 | 8 | 0.11% |
| vol | 5 | 1 | 1 | 7 | 0.10% |
| cym | 5 | 0 | 1 | 6 | 0.09% |
| fra | 2 | 2 | 2 | 6 | 0.09% |
| srp | 3 | 1 | 0 | 4 | 0.06% |
| scn | 1 | 1 | 1 | 3 | 0.04% |
| roh | 2 | 0 | 0 | 2 | 0.03% |
| min | 2 | 0 | 0 | 2 | 0.03% |
| hsb | 0 | 1 | 1 | 2 | 0.03% |
| slv | 0 | 0 | 2 | 2 | 0.03% |
| gla | 1 | 0 | 0 | 1 | 0.01% |
| diq | 1 | 0 | 0 | 1 | 0.01% |
| pam | 1 | 0 | 0 | 1 | 0.01% |
| hat | 0 | 1 | 0 | 1 | 0.01% |
| mlg | 0 | 1 | 0 | 1 | 0.01% |
| ltz | 0 | 0 | 1 | 1 | 0.01% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 1,476 |
| test_unique_inputs | 1,476 |
| train_records | 4,683 |
| train_unique_inputs | 4,683 |
| val_records | 834 |
| val_unique_inputs | 834 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.1072 | 10,353 / 2,056 | location 34%→25%, person 26%→32%, organization 21%→25%, date 19%→18% |
| entities (per mention) | train / test | 0.0914 | 10,353 / 3,614 | location 34%→42%, person 26%→19%, organization 21%→18%, date 19%→21% |
| entities (per mention) | val / test | 0.1986 WARN | 2,056 / 3,614 | location 25%→42%, person 32%→19%, organization 25%→18%, date 18%→21% |


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
| location | 3,557 | 34.36% |
| person | 2,674 | 25.83% |
| organization | 2,128 | 20.55% |
| date | 1,994 | 19.26% |

#### val

| label | count | share |
| --- | --- | --- |
| person | 666 | 32.39% |
| location | 513 | 24.95% |
| organization | 508 | 24.71% |
| date | 369 | 17.95% |

#### test

| label | count | share |
| --- | --- | --- |
| location | 1,506 | 41.67% |
| date | 762 | 21.08% |
| person | 691 | 19.12% |
| organization | 655 | 18.12% |

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
