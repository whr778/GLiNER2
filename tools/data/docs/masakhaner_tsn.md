# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_tsn | train | 2,292 | 2,288 | 4,901 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_tsn | val | 331 | 331 | 746 | 0 | 0 | 0 | 0 | 0 |
| masakhaner_tsn | test | 729 | 728 | 1,487 | 0 | 0 | 0 | 0 | 0 |

## masakhaner_tsn

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2,292 | 2,288 | 30 | 52 | 84 | 4,901 | 0 | 0 | 0 | 0 | 0 |
| val | 331 | 331 | 30 | 51 | 76 | 746 | 0 | 0 | 0 | 0 | 0 |
| test | 729 | 728 | 27 | 46 | 80 | 1,487 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 1,221 | 176 | 389 | 1,786 | 53.28% |
| swa | 902 | 125 | 272 | 1,299 | 38.75% |
| epo | 88 | 9 | 34 | 131 | 3.91% |
| fil | 23 | 6 | 8 | 37 | 1.10% |
| war | 8 | 1 | 7 | 16 | 0.48% |
| sun | 6 | 0 | 6 | 12 | 0.36% |
| ita | 6 | 2 | 1 | 9 | 0.27% |
| eus | 6 | 1 | 1 | 8 | 0.24% |
| ceb | 4 | 0 | 2 | 6 | 0.18% |
| ina | 4 | 2 | 0 | 6 | 0.18% |
| ido | 3 | 1 | 0 | 4 | 0.12% |
| srd | 3 | 1 | 0 | 4 | 0.12% |
| bre | 2 | 0 | 2 | 4 | 0.12% |
| ile | 1 | 2 | 0 | 3 | 0.09% |
| que | 2 | 1 | 0 | 3 | 0.09% |
| jbo | 1 | 1 | 1 | 3 | 0.09% |
| sco | 1 | 1 | 1 | 3 | 0.09% |
| fra | 2 | 0 | 0 | 2 | 0.06% |
| spa | 2 | 0 | 0 | 2 | 0.06% |
| cor | 1 | 1 | 0 | 2 | 0.06% |
| roh | 1 | 0 | 1 | 2 | 0.06% |
| eng | 1 | 0 | 1 | 2 | 0.06% |
| bcl | 1 | 0 | 0 | 1 | 0.03% |
| mlg | 1 | 0 | 0 | 1 | 0.03% |
| scn | 1 | 0 | 0 | 1 | 0.03% |
| deu | 1 | 0 | 0 | 1 | 0.03% |
| pam | 0 | 1 | 0 | 1 | 0.03% |
| cym | 0 | 0 | 1 | 1 | 0.03% |
| wln | 0 | 0 | 1 | 1 | 0.03% |
| est | 0 | 0 | 1 | 1 | 0.03% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 729 |
| test_unique_inputs | 728 |
| train_records | 2,292 |
| train_unique_inputs | 2,288 |
| val_records | 331 |
| val_unique_inputs | 331 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0553 | 4,901 / 746 | date 18%→24%, location 25%→21%, organization 22%→21%, person 35%→34% |
| entities (per mention) | train / test | 0.1440 | 4,901 / 1,487 | organization 22%→30%, location 25%→17%, date 18%→12%, person 35%→41% |
| entities (per mention) | val / test | 0.1599 WARN | 746 / 1,487 | date 24%→12%, organization 21%→30%, person 34%→41%, location 21%→17% |


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
| person | 1,721 | 35.12% |
| location | 1,211 | 24.71% |
| organization | 1,077 | 21.98% |
| date | 892 | 18.20% |

#### val

| label | count | share |
| --- | --- | --- |
| person | 257 | 34.45% |
| date | 177 | 23.73% |
| organization | 157 | 21.05% |
| location | 155 | 20.78% |

#### test

| label | count | share |
| --- | --- | --- |
| person | 612 | 41.16% |
| organization | 451 | 30.33% |
| location | 246 | 16.54% |
| date | 178 | 11.97% |

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
