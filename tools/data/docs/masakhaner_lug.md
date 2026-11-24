# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_lug | train | 3,879 | 3,798 | 7,727 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_lug | val | 640 | 639 | 1,217 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_lug | test | 1,238 | 1,236 | 2,252 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_lug

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3,879 | 3,798 | 20 | 33 | 87 | 7,727 | 0 | 0 | 0 | 0 | 0 |
| val | 640 | 639 | 19 | 27 | 37 | 1,217 | 0 | 0 | 0 | 0 | 0 |
| test | 1,238 | 1,236 | 19 | 26 | 51 | 2,252 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 2,818 | 469 | 911 | 4,198 | 72.92% |
| war | 473 | 75 | 163 | 711 | 12.35% |
| epo | 129 | 12 | 15 | 156 | 2.71% |
| swa | 103 | 21 | 27 | 151 | 2.62% |
| ceb | 89 | 14 | 29 | 132 | 2.29% |
| fin | 74 | 20 | 30 | 124 | 2.15% |
| eng | 49 | 5 | 8 | 62 | 1.08% |
| sun | 35 | 9 | 10 | 54 | 0.94% |
| fil | 31 | 4 | 19 | 54 | 0.94% |
| ilo | 22 | 3 | 6 | 31 | 0.54% |
| eus | 6 | 0 | 3 | 9 | 0.16% |
| hun | 4 | 1 | 3 | 8 | 0.14% |
| ind | 4 | 0 | 4 | 8 | 0.14% |
| lat | 5 | 1 | 1 | 7 | 0.12% |
| tur | 4 | 0 | 2 | 6 | 0.10% |
| cym | 5 | 1 | 0 | 6 | 0.10% |
| cor | 5 | 0 | 1 | 6 | 0.10% |
| swe | 3 | 0 | 3 | 6 | 0.10% |
| vol | 3 | 2 | 0 | 5 | 0.09% |
| ido | 4 | 0 | 0 | 4 | 0.07% |
| bih | 2 | 0 | 1 | 3 | 0.05% |
| urd | 1 | 1 | 0 | 2 | 0.03% |
| pol | 2 | 0 | 0 | 2 | 0.03% |
| que | 1 | 1 | 0 | 2 | 0.03% |
| scn | 1 | 0 | 1 | 2 | 0.03% |
| min | 1 | 0 | 0 | 1 | 0.02% |
| spa | 1 | 0 | 0 | 1 | 0.02% |
| nno | 1 | 0 | 0 | 1 | 0.02% |
| nld | 1 | 0 | 0 | 1 | 0.02% |
| ile | 1 | 0 | 0 | 1 | 0.02% |
| cbk | 1 | 0 | 0 | 1 | 0.02% |
| bre | 0 | 1 | 0 | 1 | 0.02% |
| als | 0 | 0 | 1 | 1 | 0.02% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 1,238 |
| test_unique_inputs | 1,236 |
| train_records | 3,879 |
| train_unique_inputs | 3,798 |
| val_records | 640 |
| val_unique_inputs | 639 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0325 | 7,727 / 1,217 | date 11%→14%, organization 22%→21%, person 38%→37%, location 28%→28% |
| entities (per mention) | train / test | 0.0536 | 7,727 / 2,252 | person 38%→35%, organization 22%→25%, date 11%→14%, location 28%→26% |
| entities (per mention) | val / test | 0.0457 | 1,217 / 2,252 | organization 21%→25%, location 28%→26%, person 37%→35%, date 14%→14% |


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
| person | 2,949 | 38.16% |
| location | 2,159 | 27.94% |
| organization | 1,734 | 22.44% |
| date | 885 | 11.45% |

#### val

| label | count | share |
| --- | --- | --- |
| person | 445 | 36.57% |
| location | 345 | 28.35% |
| organization | 253 | 20.79% |
| date | 174 | 14.30% |

#### test

| label | count | share |
| --- | --- | --- |
| person | 787 | 34.95% |
| location | 581 | 25.80% |
| organization | 571 | 25.36% |
| date | 313 | 13.90% |

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
