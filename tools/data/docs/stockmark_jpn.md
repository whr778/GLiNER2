# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stockmark_jpn | train | 3,900 | 3,900 | 10,302 | 0 | 0 | 0 | 0 | 0 |
| stockmark_jpn | val | 486 | 486 | 1,242 | 0 | 0 | 0 | 0 | 0 |
| stockmark_jpn | test | 473 | 473 | 1,243 | 0 | 0 | 0 | 0 | 0 |

## stockmark_jpn

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3,900 | 3,900 | 50 | 95 | 289 | 10,302 | 0 | 0 | 0 | 0 | 0 |
| val | 486 | 486 | 47 | 94 | 208 | 1,242 | 0 | 0 | 0 | 0 | 0 |
| test | 473 | 473 | 51 | 96 | 215 | 1,243 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| jpn | 3,890 | 485 | 471 | 4,846 | 99.73% |
| und | 6 | 1 | 1 | 8 | 0.16% |
| zho | 4 | 0 | 1 | 5 | 0.10% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 473 |
| test_unique_inputs | 473 |
| train_records | 3,900 |
| train_unique_inputs | 3,900 |
| val_records | 486 |
| val_unique_inputs | 486 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0326 | 10,302 / 1,242 | 人名 23%→21%, 政治的組織名 9%→10%, 法人名 19%→18%, 製品名 9%→10%, イベント名 8%→8% |
| entities (per mention) | train / test | 0.0404 | 10,302 / 1,243 | 法人名 19%→17%, 地名 16%→18%, 施設名 8%→10%, 人名 23%→22%, 製品名 9%→8% |
| entities (per mention) | val / test | 0.0405 | 1,242 / 1,243 | 製品名 10%→8%, 施設名 8%→10%, 政治的組織名 10%→9%, 地名 16%→18%, 法人名 18%→17% |


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
| 人名 | 2,366 | 22.97% |
| 法人名 | 1,943 | 18.86% |
| 地名 | 1,653 | 16.05% |
| 製品名 | 965 | 9.37% |
| 政治的組織名 | 902 | 8.76% |
| 施設名 | 863 | 8.38% |
| その他の組織名 | 818 | 7.94% |
| イベント名 | 792 | 7.69% |

#### val

| label | count | share |
| --- | --- | --- |
| 人名 | 260 | 20.93% |
| 法人名 | 223 | 17.95% |
| 地名 | 204 | 16.43% |
| 政治的組織名 | 127 | 10.23% |
| 製品名 | 126 | 10.14% |
| イベント名 | 101 | 8.13% |
| その他の組織名 | 101 | 8.13% |
| 施設名 | 100 | 8.05% |

#### test

| label | count | share |
| --- | --- | --- |
| 人名 | 269 | 21.64% |
| 地名 | 219 | 17.62% |
| 法人名 | 214 | 17.22% |
| 施設名 | 121 | 9.73% |
| 政治的組織名 | 108 | 8.69% |
| イベント名 | 104 | 8.37% |
| 製品名 | 104 | 8.37% |
| その他の組織名 | 104 | 8.37% |

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
