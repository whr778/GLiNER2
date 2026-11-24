# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_ibo | train | 3,980 | 3,980 | 9,041 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_ibo | val | 490 | 490 | 1,101 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_ibo | test | 1,323 | 1,323 | 3,122 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_ibo

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3,980 | 3,980 | 42 | 61 | 110 | 9,041 | 0 | 0 | 0 | 0 | 0 |
| val | 490 | 490 | 45 | 59 | 72 | 1,101 | 0 | 0 | 0 | 0 | 0 |
| test | 1,323 | 1,323 | 42 | 58 | 71 | 3,122 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 2,581 | 304 | 819 | 3,704 | 63.94% |
| vie | 1,069 | 149 | 405 | 1,623 | 28.02% |
| fil | 159 | 20 | 35 | 214 | 3.69% |
| eng | 35 | 6 | 18 | 59 | 1.02% |
| kom | 31 | 2 | 6 | 39 | 0.67% |
| swa | 26 | 1 | 4 | 31 | 0.54% |
| hrv | 8 | 0 | 11 | 19 | 0.33% |
| hat | 9 | 1 | 3 | 13 | 0.22% |
| jav | 8 | 3 | 2 | 13 | 0.22% |
| yor | 9 | 0 | 1 | 10 | 0.17% |
| roh | 7 | 0 | 2 | 9 | 0.16% |
| pms | 6 | 0 | 1 | 7 | 0.12% |
| ilo | 5 | 0 | 1 | 6 | 0.10% |
| lat | 3 | 0 | 1 | 4 | 0.07% |
| eus | 2 | 0 | 1 | 3 | 0.05% |
| epo | 3 | 0 | 0 | 3 | 0.05% |
| sco | 2 | 1 | 0 | 3 | 0.05% |
| ceb | 2 | 1 | 0 | 3 | 0.05% |
| grn | 1 | 1 | 1 | 3 | 0.05% |
| bre | 1 | 1 | 1 | 3 | 0.05% |
| srp | 2 | 0 | 0 | 2 | 0.03% |
| cor | 2 | 0 | 0 | 2 | 0.03% |
| eml | 1 | 0 | 1 | 2 | 0.03% |
| lmo | 1 | 0 | 1 | 2 | 0.03% |
| war | 1 | 0 | 1 | 2 | 0.03% |
| swe | 1 | 0 | 0 | 1 | 0.02% |
| kur | 1 | 0 | 0 | 1 | 0.02% |
| nds | 1 | 0 | 0 | 1 | 0.02% |
| lez | 1 | 0 | 0 | 1 | 0.02% |
| glv | 1 | 0 | 0 | 1 | 0.02% |
| ido | 1 | 0 | 0 | 1 | 0.02% |
| ita | 0 | 0 | 1 | 1 | 0.02% |
| bcl | 0 | 0 | 1 | 1 | 0.02% |
| ind | 0 | 0 | 1 | 1 | 0.02% |
| mlg | 0 | 0 | 1 | 1 | 0.02% |
| fin | 0 | 0 | 1 | 1 | 0.02% |
| lao | 0 | 0 | 1 | 1 | 0.02% |
| slv | 0 | 0 | 1 | 1 | 0.02% |
| cat | 0 | 0 | 1 | 1 | 0.02% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 1,323 |
| test_unique_inputs | 1,323 |
| train_records | 3,980 |
| train_unique_inputs | 3,980 |
| val_records | 490 |
| val_unique_inputs | 490 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0395 | 9,041 / 1,101 | location 38%→42%, person 36%→34%, date 10%→9%, organization 16%→15% |
| entities (per mention) | train / test | 0.0491 | 9,041 / 3,122 | location 38%→42%, person 36%→33%, organization 16%→13%, date 10%→11% |
| entities (per mention) | val / test | 0.0236 | 1,101 / 3,122 | date 9%→11%, organization 15%→13%, person 34%→33%, location 42%→42% |


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
| location | 3,453 | 38.19% |
| person | 3,223 | 35.65% |
| organization | 1,417 | 15.67% |
| date | 948 | 10.49% |

#### val

| label | count | share |
| --- | --- | --- |
| location | 464 | 42.14% |
| person | 376 | 34.15% |
| organization | 161 | 14.62% |
| date | 100 | 9.08% |

#### test

| label | count | share |
| --- | --- | --- |
| location | 1,316 | 42.15% |
| person | 1,034 | 33.12% |
| organization | 415 | 13.29% |
| date | 357 | 11.43% |

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
