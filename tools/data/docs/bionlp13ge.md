# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bionlp13ge | train | 1,504 | 1,499 | 3,044 | 0 | 0 | 0 | 0 | 0 |
| bionlp13ge | val | 1,663 | 1,655 | 3,619 | 0 | 0 | 0 | 0 | 0 |
| bionlp13ge | test | 1,941 | 1,936 | 3,818 | 0 | 0 | 0 | 0 | 0 |

## bionlp13ge

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1,504 | 1,499 | 32 | 55 | 162 | 3,044 | 0 | 0 | 0 | 0 | 0 |
| val | 1,663 | 1,655 | 31 | 55 | 192 | 3,619 | 0 | 0 | 0 | 0 | 0 |
| test | 1,941 | 1,936 | 32 | 58 | 218 | 3,818 | 0 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 1,461 | 1,594 | 1,812 | 4,867 | 95.28% |
| und | 43 | 65 | 118 | 226 | 4.42% |
| ita | 0 | 0 | 7 | 7 | 0.14% |
| fil | 0 | 1 | 1 | 2 | 0.04% |
| fra | 0 | 1 | 0 | 1 | 0.02% |
| cat | 0 | 1 | 0 | 1 | 0.02% |
| dan | 0 | 1 | 0 | 1 | 0.02% |
| epo | 0 | 0 | 1 | 1 | 0.02% |
| deu | 0 | 0 | 1 | 1 | 0.02% |
| msa | 0 | 0 | 1 | 1 | 0.02% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 1,941 |
| test_unique_inputs | 1,936 |
| train_records | 1,504 |
| train_unique_inputs | 1,499 |
| val_records | 1,663 |
| val_unique_inputs | 1,655 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0000 | 3,044 / 3,619 | Protein 100%→100% |
| entities (per mention) | train / test | 0.0000 | 3,044 / 3,818 | Protein 100%→100% |
| entities (per mention) | val / test | 0.0000 | 3,619 / 3,818 | Protein 100%→100% |


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
| Protein | 3,044 | 100.00% |

#### val

| label | count | share |
| --- | --- | --- |
| Protein | 3,619 | 100.00% |

#### test

| label | count | share |
| --- | --- | --- |
| Protein | 3,818 | 100.00% |

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
