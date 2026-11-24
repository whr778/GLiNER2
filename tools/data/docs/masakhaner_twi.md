# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_twi | train | 2,179 | 2,179 | 4,498 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_twi | val | 210 | 210 | 378 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_twi | test | 390 | 389 | 719 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_twi

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2,179 | 2,179 | 30 | 51 | 76 | 4,498 | 0 | 0 | 0 | 0 | 0 |
| val | 210 | 210 | 29 | 51 | 66 | 378 | 0 | 0 | 0 | 0 | 0 |
| test | 390 | 389 | 28 | 50 | 71 | 719 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 1,969 | 179 | 339 | 2,487 | 89.49% |
| epo | 116 | 13 | 23 | 152 | 5.47% |
| eng | 12 | 2 | 4 | 18 | 0.65% |
| ido | 9 | 2 | 1 | 12 | 0.43% |
| fin | 5 | 2 | 2 | 9 | 0.32% |
| ceb | 5 | 2 | 1 | 8 | 0.29% |
| cor | 6 | 0 | 2 | 8 | 0.29% |
| bre | 4 | 0 | 3 | 7 | 0.25% |
| sun | 3 | 1 | 2 | 6 | 0.22% |
| oci | 4 | 1 | 0 | 5 | 0.18% |
| hun | 3 | 1 | 1 | 5 | 0.18% |
| hat | 3 | 1 | 1 | 5 | 0.18% |
| mlg | 4 | 0 | 1 | 5 | 0.18% |
| eus | 3 | 1 | 0 | 4 | 0.14% |
| ita | 3 | 0 | 1 | 4 | 0.14% |
| que | 3 | 0 | 0 | 3 | 0.11% |
| lat | 2 | 0 | 1 | 3 | 0.11% |
| fil | 2 | 1 | 0 | 3 | 0.11% |
| jav | 2 | 1 | 0 | 3 | 0.11% |
| por | 2 | 0 | 1 | 3 | 0.11% |
| gom | 1 | 0 | 1 | 2 | 0.07% |
| cbk | 2 | 0 | 0 | 2 | 0.07% |
| srp | 1 | 0 | 1 | 2 | 0.07% |
| lmo | 2 | 0 | 0 | 2 | 0.07% |
| grn | 2 | 0 | 0 | 2 | 0.07% |
| ron | 1 | 0 | 1 | 2 | 0.07% |
| swa | 0 | 0 | 2 | 2 | 0.07% |
| pms | 1 | 0 | 0 | 1 | 0.04% |
| deu | 1 | 0 | 0 | 1 | 0.04% |
| war | 1 | 0 | 0 | 1 | 0.04% |
| als | 1 | 0 | 0 | 1 | 0.04% |
| cym | 1 | 0 | 0 | 1 | 0.04% |
| nap | 1 | 0 | 0 | 1 | 0.04% |
| scn | 1 | 0 | 0 | 1 | 0.04% |
| spa | 1 | 0 | 0 | 1 | 0.04% |
| nor | 1 | 0 | 0 | 1 | 0.04% |
| sco | 1 | 0 | 0 | 1 | 0.04% |
| nds | 0 | 1 | 0 | 1 | 0.04% |
| vls | 0 | 1 | 0 | 1 | 0.04% |
| tuk | 0 | 1 | 0 | 1 | 0.04% |
| jbo | 0 | 0 | 1 | 1 | 0.04% |
| wln | 0 | 0 | 1 | 1 | 0.04% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 390 |
| test_unique_inputs | 389 |
| train_records | 2,179 |
| train_unique_inputs | 2,179 |
| val_records | 210 |
| val_unique_inputs | 210 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0767 | 4,498 / 378 | location 34%→26%, organization 21%→25%, date 11%→12%, person 35%→36% |
| entities (per mention) | train / test | 0.0903 | 4,498 / 719 | organization 21%→26%, location 34%→29%, person 35%→31%, date 11%→14% |
| entities (per mention) | val / test | 0.0565 | 378 / 719 | person 36%→31%, location 26%→29%, date 12%→14%, organization 25%→26% |


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
| person | 1,579 | 35.10% |
| location | 1,511 | 33.59% |
| organization | 928 | 20.63% |
| date | 480 | 10.67% |

#### val

| label | count | share |
| --- | --- | --- |
| person | 137 | 36.24% |
| location | 98 | 25.93% |
| organization | 96 | 25.40% |
| date | 47 | 12.43% |

#### test

| label | count | share |
| --- | --- | --- |
| person | 220 | 30.60% |
| location | 209 | 29.07% |
| organization | 189 | 26.29% |
| date | 101 | 14.05% |

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
