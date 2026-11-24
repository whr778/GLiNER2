# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhanews_lug | train | 771 | 771 | 0 | 0 | 771 | 0 | 0 | 0 |
| masakhanews_lug | val | 110 | 110 | 0 | 0 | 110 | 0 | 0 | 0 |
| masakhanews_lug | test | 223 | 223 | 0 | 0 | 223 | 0 | 0 | 0 |

## masakhanews_lug

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 771 | 771 | 285 | 417 | 980 | 0 | 0 | 771 | 0 | 0 | 0 |
| val | 110 | 110 | 292 | 443 | 671 | 0 | 0 | 110 | 0 | 0 | 0 |
| test | 223 | 223 | 282 | 415 | 575 | 0 | 0 | 223 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| und | 561 | 71 | 162 | 794 | 71.92% |
| war | 159 | 33 | 48 | 240 | 21.74% |
| fil | 32 | 1 | 8 | 41 | 3.71% |
| epo | 14 | 3 | 3 | 20 | 1.81% |
| lat | 1 | 1 | 1 | 3 | 0.27% |
| urd | 2 | 0 | 0 | 2 | 0.18% |
| ceb | 2 | 0 | 0 | 2 | 0.18% |
| swa | 0 | 1 | 1 | 2 | 0.18% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 223 |
| test_unique_inputs | 223 |
| train_records | 771 |
| train_unique_inputs | 771 |
| val_records | 110 |
| val_unique_inputs | 110 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| classifications (true_label) | train / val | 0.0051 | 771 / 110 | news topic: sports 11%→10%, news topic: health 21%→21%, news topic: business 15%→15%, news topic: politics 45%→45%, news topic: religion 8%→8% |
| classifications (true_label) | train / test | 0.0061 | 771 / 223 | news topic: politics 45%→45%, news topic: religion 8%→9%, news topic: sports 11%→11%, news topic: business 15%→15%, news topic: health 21%→21% |
| classifications (true_label) | val / test | 0.0110 | 110 / 223 | news topic: sports 10%→11%, news topic: politics 45%→45%, news topic: religion 8%→9%, news topic: health 21%→21%, news topic: business 15%→15% |

Thin support in val, classifications (true_label): 1 label(s) under 10 gold. news topic: religion ×9.

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

None.

### Relations

None.

### Classifications

**train, news topic.** Majority class 45.4% of 771 answers. The marginal is `true_label`, not the menu.

#### train news topic

| label | count | share |
| --- | --- | --- |
| politics | 350 | 45.40% |
| health | 159 | 20.62% |
| business | 118 | 15.30% |
| sports | 81 | 10.51% |
| religion | 63 | 8.17% |

**val, news topic.** Majority class 45.5% of 110 answers. The marginal is `true_label`, not the menu.

#### val news topic

| label | count | share |
| --- | --- | --- |
| politics | 50 | 45.45% |
| health | 23 | 20.91% |
| business | 17 | 15.45% |
| sports | 11 | 10.00% |
| religion | 9 | 8.18% |

**test, news topic.** Majority class 44.8% of 223 answers. The marginal is `true_label`, not the menu.

#### test news topic

| label | count | share |
| --- | --- | --- |
| politics | 100 | 44.84% |
| health | 46 | 20.63% |
| business | 34 | 15.25% |
| sports | 24 | 10.76% |
| religion | 19 | 8.52% |

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
