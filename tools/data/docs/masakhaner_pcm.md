# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_pcm | train | 3,327 | 3,327 | 7,627 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_pcm | val | 493 | 493 | 1,308 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_pcm | test | 990 | 990 | 2,315 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_pcm

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3,327 | 3,327 | 27 | 47 | 104 | 7,627 | 0 | 0 | 0 | 0 | 0 |
| val | 493 | 493 | 29 | 50 | 97 | 1,308 | 0 | 0 | 0 | 0 | 0 |
| test | 990 | 990 | 28 | 46 | 104 | 2,315 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 1,686 | 265 | 492 | 2,443 | 50.79% |
| und | 1,547 | 217 | 465 | 2,229 | 46.34% |
| ita | 70 | 11 | 22 | 103 | 2.14% |
| deu | 13 | 0 | 7 | 20 | 0.42% |
| war | 3 | 0 | 2 | 5 | 0.10% |
| nds | 2 | 0 | 0 | 2 | 0.04% |
| hat | 2 | 0 | 0 | 2 | 0.04% |
| por | 1 | 0 | 1 | 2 | 0.04% |
| nno | 1 | 0 | 0 | 1 | 0.02% |
| dan | 1 | 0 | 0 | 1 | 0.02% |
| fra | 1 | 0 | 0 | 1 | 0.02% |
| nld | 0 | 0 | 1 | 1 | 0.02% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 990 |
| test_unique_inputs | 990 |
| train_records | 3,327 |
| train_unique_inputs | 3,327 |
| val_records | 493 |
| val_unique_inputs | 493 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0399 | 7,627 / 1,308 | organization 27%→31%, person 24%→21%, date 19%→19%, location 30%→30% |
| entities (per mention) | train / test | 0.0435 | 7,627 / 2,315 | person 24%→27%, date 19%→17%, location 30%→28%, organization 27%→28% |
| entities (per mention) | val / test | 0.0652 | 1,308 / 2,315 | person 21%→27%, organization 31%→28%, location 30%→28%, date 19%→17% |


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
| location | 2,282 | 29.92% |
| organization | 2,026 | 26.56% |
| person | 1,866 | 24.47% |
| date | 1,453 | 19.05% |

#### val

| label | count | share |
| --- | --- | --- |
| organization | 399 | 30.50% |
| location | 392 | 29.97% |
| person | 274 | 20.95% |
| date | 243 | 18.58% |

#### test

| label | count | share |
| --- | --- | --- |
| organization | 646 | 27.90% |
| location | 645 | 27.86% |
| person | 636 | 27.47% |
| date | 388 | 16.76% |

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
