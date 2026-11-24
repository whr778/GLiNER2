# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_wol | train | 2,492 | 2,492 | 5,022 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_wol | val | 375 | 375 | 758 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_wol | test | 723 | 723 | 1,499 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_wol

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2,492 | 2,492 | 26 | 53 | 82 | 5,022 | 0 | 0 | 0 | 0 | 0 |
| val | 375 | 375 | 29 | 58 | 79 | 758 | 0 | 0 | 0 | 0 | 0 |
| test | 723 | 723 | 28 | 53 | 82 | 1,499 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 1,650 | 239 | 484 | 2,373 | 66.10% |
| sun | 192 | 29 | 43 | 264 | 7.35% |
| cor | 110 | 13 | 34 | 157 | 4.37% |
| lmo | 59 | 12 | 16 | 87 | 2.42% |
| cym | 52 | 9 | 15 | 76 | 2.12% |
| bre | 46 | 2 | 12 | 60 | 1.67% |
| scn | 39 | 7 | 13 | 59 | 1.64% |
| ast | 27 | 7 | 16 | 50 | 1.39% |
| war | 29 | 6 | 10 | 45 | 1.25% |
| hat | 33 | 5 | 6 | 44 | 1.23% |
| wln | 20 | 5 | 7 | 32 | 0.89% |
| kur | 11 | 1 | 6 | 18 | 0.50% |
| fry | 13 | 1 | 3 | 17 | 0.47% |
| grn | 8 | 0 | 7 | 15 | 0.42% |
| ilo | 5 | 4 | 5 | 14 | 0.39% |
| roh | 10 | 2 | 2 | 14 | 0.39% |
| srd | 9 | 3 | 1 | 13 | 0.36% |
| pms | 6 | 4 | 3 | 13 | 0.36% |
| mlt | 4 | 3 | 6 | 13 | 0.36% |
| glv | 5 | 3 | 4 | 12 | 0.33% |
| ile | 7 | 2 | 2 | 11 | 0.31% |
| nap | 8 | 2 | 1 | 11 | 0.31% |
| yor | 6 | 1 | 2 | 9 | 0.25% |
| que | 8 | 0 | 1 | 9 | 0.25% |
| cat | 7 | 0 | 1 | 8 | 0.22% |
| nah | 6 | 0 | 2 | 8 | 0.22% |
| gom | 6 | 1 | 1 | 8 | 0.22% |
| lat | 8 | 0 | 0 | 8 | 0.22% |
| jav | 4 | 2 | 1 | 7 | 0.19% |
| diq | 7 | 0 | 0 | 7 | 0.19% |
| epo | 6 | 0 | 1 | 7 | 0.19% |
| ido | 5 | 2 | 0 | 7 | 0.19% |
| als | 5 | 1 | 1 | 7 | 0.19% |
| jbo | 5 | 0 | 1 | 6 | 0.17% |
| ltz | 5 | 0 | 1 | 6 | 0.17% |
| fil | 5 | 0 | 1 | 6 | 0.17% |
| ceb | 6 | 0 | 0 | 6 | 0.17% |
| lim | 5 | 0 | 1 | 6 | 0.17% |
| cos | 6 | 0 | 0 | 6 | 0.17% |
| sco | 4 | 0 | 1 | 5 | 0.14% |
| ina | 3 | 0 | 2 | 5 | 0.14% |
| oci | 3 | 1 | 1 | 5 | 0.14% |
| sqi | 3 | 1 | 1 | 5 | 0.14% |
| vec | 4 | 0 | 0 | 4 | 0.11% |
| eng | 3 | 0 | 1 | 4 | 0.11% |
| swa | 3 | 1 | 0 | 4 | 0.11% |
| cbk | 2 | 1 | 0 | 3 | 0.08% |
| ita | 1 | 0 | 2 | 3 | 0.08% |
| mlg | 1 | 1 | 1 | 3 | 0.08% |
| nds | 1 | 1 | 0 | 2 | 0.06% |
| tur | 1 | 1 | 0 | 2 | 0.06% |
| bcl | 2 | 0 | 0 | 2 | 0.06% |
| hsb | 1 | 1 | 0 | 2 | 0.06% |
| fin | 1 | 0 | 1 | 2 | 0.06% |
| pam | 2 | 0 | 0 | 2 | 0.06% |
| xal | 0 | 0 | 2 | 2 | 0.06% |
| gle | 1 | 0 | 0 | 1 | 0.03% |
| afr | 1 | 0 | 0 | 1 | 0.03% |
| est | 1 | 0 | 0 | 1 | 0.03% |
| vls | 1 | 0 | 0 | 1 | 0.03% |
| pfl | 1 | 0 | 0 | 1 | 0.03% |
| che | 1 | 0 | 0 | 1 | 0.03% |
| fra | 1 | 0 | 0 | 1 | 0.03% |
| eus | 1 | 0 | 0 | 1 | 0.03% |
| hun | 1 | 0 | 0 | 1 | 0.03% |
| tuk | 1 | 0 | 0 | 1 | 0.03% |
| vep | 1 | 0 | 0 | 1 | 0.03% |
| uzb | 1 | 0 | 0 | 1 | 0.03% |
| div | 1 | 0 | 0 | 1 | 0.03% |
| por | 1 | 0 | 0 | 1 | 0.03% |
| pnb | 0 | 1 | 0 | 1 | 0.03% |
| som | 0 | 0 | 1 | 1 | 0.03% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 723 |
| test_unique_inputs | 723 |
| train_records | 2,492 |
| train_unique_inputs | 2,492 |
| val_records | 375 |
| val_unique_inputs | 375 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0382 | 5,022 / 758 | person 31%→34%, location 34%→31%, date 12%→10%, organization 24%→24% |
| entities (per mention) | train / test | 0.0112 | 5,022 / 1,499 | location 34%→35%, person 31%→30%, date 12%→11%, organization 24%→24% |
| entities (per mention) | val / test | 0.0454 | 758 / 1,499 | person 34%→30%, location 31%→35%, date 10%→11%, organization 24%→24% |


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
| location | 1,696 | 33.77% |
| person | 1,536 | 30.59% |
| organization | 1,207 | 24.03% |
| date | 583 | 11.61% |

#### val

| label | count | share |
| --- | --- | --- |
| person | 260 | 34.30% |
| location | 236 | 31.13% |
| organization | 183 | 24.14% |
| date | 79 | 10.42% |

#### test

| label | count | share |
| --- | --- | --- |
| location | 523 | 34.89% |
| person | 450 | 30.02% |
| organization | 358 | 23.88% |
| date | 168 | 11.21% |

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
