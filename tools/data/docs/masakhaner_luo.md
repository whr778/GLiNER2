# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_luo | train | 3,978 | 3,971 | 10,797 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_luo | val | 443 | 438 | 1,048 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_luo | test | 1,078 | 1,075 | 2,484 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_luo

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3,978 | 3,971 | 34 | 57 | 91 | 10,797 | 0 | 0 | 0 | 0 | 0 |
| val | 443 | 438 | 35 | 59 | 81 | 1,048 | 0 | 0 | 0 | 0 | 0 |
| test | 1,078 | 1,075 | 31 | 56 | 95 | 2,484 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 3,034 | 367 | 811 | 4,212 | 76.60% |
| epo | 827 | 65 | 227 | 1,119 | 20.35% |
| war | 60 | 7 | 26 | 93 | 1.69% |
| ido | 10 | 1 | 1 | 12 | 0.22% |
| ceb | 6 | 0 | 5 | 11 | 0.20% |
| srp | 6 | 0 | 2 | 8 | 0.15% |
| eng | 8 | 0 | 0 | 8 | 0.15% |
| ilo | 3 | 2 | 1 | 6 | 0.11% |
| spa | 2 | 1 | 1 | 4 | 0.07% |
| sun | 3 | 0 | 1 | 4 | 0.07% |
| fin | 3 | 0 | 0 | 3 | 0.05% |
| ita | 1 | 0 | 1 | 2 | 0.04% |
| pol | 2 | 0 | 0 | 2 | 0.04% |
| cor | 2 | 0 | 0 | 2 | 0.04% |
| fil | 1 | 0 | 1 | 2 | 0.04% |
| bre | 1 | 0 | 1 | 2 | 0.04% |
| swa | 1 | 0 | 0 | 1 | 0.02% |
| jbo | 1 | 0 | 0 | 1 | 0.02% |
| fry | 1 | 0 | 0 | 1 | 0.02% |
| que | 1 | 0 | 0 | 1 | 0.02% |
| pms | 1 | 0 | 0 | 1 | 0.02% |
| deu | 1 | 0 | 0 | 1 | 0.02% |
| pam | 1 | 0 | 0 | 1 | 0.02% |
| uzb | 1 | 0 | 0 | 1 | 0.02% |
| eus | 1 | 0 | 0 | 1 | 0.02% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 1,078 |
| test_unique_inputs | 1,075 |
| train_records | 3,978 |
| train_unique_inputs | 3,971 |
| val_records | 443 |
| val_unique_inputs | 438 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0894 | 10,797 / 1,048 | person 30%→21%, location 26%→30%, organization 25%→28%, date 19%→21% |
| entities (per mention) | train / test | 0.0590 | 10,797 / 2,484 | organization 25%→21%, location 26%→29%, person 30%→33%, date 19%→17% |
| entities (per mention) | val / test | 0.1144 | 1,048 / 2,484 | person 21%→33%, organization 28%→21%, date 21%→17%, location 30%→29% |


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
| person | 3,273 | 30.31% |
| location | 2,814 | 26.06% |
| organization | 2,683 | 24.85% |
| date | 2,027 | 18.77% |

#### val

| label | count | share |
| --- | --- | --- |
| location | 312 | 29.77% |
| organization | 290 | 27.67% |
| person | 224 | 21.37% |
| date | 222 | 21.18% |

#### test

| label | count | share |
| --- | --- | --- |
| person | 815 | 32.81% |
| location | 732 | 29.47% |
| organization | 525 | 21.14% |
| date | 412 | 16.59% |

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
