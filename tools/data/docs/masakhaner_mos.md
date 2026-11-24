# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_mos | train | 2,115 | 2,104 | 4,074 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_mos | val | 369 | 369 | 781 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_mos | test | 620 | 620 | 1,177 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_mos

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2,115 | 2,104 | 31 | 56 | 95 | 4,074 | 0 | 0 | 0 | 0 | 0 |
| val | 369 | 369 | 29 | 50 | 72 | 781 | 0 | 0 | 0 | 0 | 0 |
| test | 620 | 620 | 24 | 48 | 77 | 1,177 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 1,775 | 297 | 482 | 2,554 | 82.28% |
| est | 136 | 26 | 66 | 228 | 7.35% |
| sun | 24 | 5 | 10 | 39 | 1.26% |
| bre | 26 | 4 | 7 | 37 | 1.19% |
| por | 20 | 6 | 4 | 30 | 0.97% |
| grn | 19 | 1 | 5 | 25 | 0.81% |
| epo | 14 | 6 | 5 | 25 | 0.81% |
| hat | 16 | 3 | 1 | 20 | 0.64% |
| que | 14 | 1 | 2 | 17 | 0.55% |
| lat | 7 | 0 | 5 | 12 | 0.39% |
| ltz | 3 | 1 | 7 | 11 | 0.35% |
| fry | 5 | 1 | 4 | 10 | 0.32% |
| swa | 3 | 5 | 2 | 10 | 0.32% |
| cor | 6 | 1 | 2 | 9 | 0.29% |
| ido | 6 | 2 | 0 | 8 | 0.26% |
| fin | 7 | 1 | 0 | 8 | 0.26% |
| fra | 4 | 1 | 3 | 8 | 0.26% |
| bih | 3 | 0 | 2 | 5 | 0.16% |
| cym | 3 | 1 | 1 | 5 | 0.16% |
| ceb | 3 | 0 | 1 | 4 | 0.13% |
| deu | 3 | 0 | 0 | 3 | 0.10% |
| hun | 3 | 0 | 0 | 3 | 0.10% |
| ita | 3 | 0 | 0 | 3 | 0.10% |
| gla | 3 | 0 | 0 | 3 | 0.10% |
| pms | 1 | 0 | 2 | 3 | 0.10% |
| eng | 1 | 1 | 1 | 3 | 0.10% |
| jbo | 1 | 0 | 1 | 2 | 0.06% |
| nds | 1 | 1 | 0 | 2 | 0.06% |
| nld | 1 | 0 | 1 | 2 | 0.06% |
| glv | 0 | 2 | 0 | 2 | 0.06% |
| war | 0 | 0 | 2 | 2 | 0.06% |
| pam | 1 | 0 | 0 | 1 | 0.03% |
| vol | 1 | 0 | 0 | 1 | 0.03% |
| eus | 1 | 0 | 0 | 1 | 0.03% |
| roh | 1 | 0 | 0 | 1 | 0.03% |
| jav | 0 | 1 | 0 | 1 | 0.03% |
| hsb | 0 | 1 | 0 | 1 | 0.03% |
| ilo | 0 | 1 | 0 | 1 | 0.03% |
| isl | 0 | 0 | 1 | 1 | 0.03% |
| hrv | 0 | 0 | 1 | 1 | 0.03% |
| scn | 0 | 0 | 1 | 1 | 0.03% |
| als | 0 | 0 | 1 | 1 | 0.03% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 620 |
| test_unique_inputs | 620 |
| train_records | 2,115 |
| train_unique_inputs | 2,104 |
| val_records | 369 |
| val_unique_inputs | 369 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0240 | 4,074 / 781 | location 37%→39%, person 23%→22%, organization 21%→20%, date 19%→19% |
| entities (per mention) | train / test | 0.0757 | 4,074 / 1,177 | organization 21%→29%, person 23%→17%, location 37%→34%, date 19%→19% |
| entities (per mention) | val / test | 0.0891 | 781 / 1,177 | organization 20%→29%, location 39%→34%, person 22%→17%, date 19%→19% |


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
| location | 1,488 | 36.52% |
| person | 921 | 22.61% |
| organization | 872 | 21.40% |
| date | 793 | 19.46% |

#### val

| label | count | share |
| --- | --- | --- |
| location | 304 | 38.92% |
| person | 169 | 21.64% |
| organization | 160 | 20.49% |
| date | 148 | 18.95% |

#### test

| label | count | share |
| --- | --- | --- |
| location | 403 | 34.24% |
| organization | 341 | 28.97% |
| date | 228 | 19.37% |
| person | 205 | 17.42% |

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
