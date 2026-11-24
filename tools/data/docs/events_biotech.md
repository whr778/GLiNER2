# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| events_biotech | train | 2,217 | 2,011 | 0 | 0 | 4,082 | 0 | 0 | 0 |
| events_biotech | val | 279 | 259 | 0 | 0 | 527 | 0 | 0 | 0 |
| events_biotech | test | 263 | 241 | 0 | 0 | 479 | 0 | 0 | 0 |
| events_biotech | all | 2,759 | 2,511 | 0 | 0 | 5,088 | 0 | 0 | 0 |

## events_biotech

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2,217 | 2,011 | 500 | 1162 | 1673 | 0 | 0 | 4,082 | 0 | 0 | 0 |
| val | 279 | 259 | 518 | 1069 | 1388 | 0 | 0 | 527 | 0 | 0 | 0 |
| test | 263 | 241 | 477 | 1018 | 1493 | 0 | 0 | 479 | 0 | 0 | 0 |
| all | 2,759 | 2,511 | 500 | 1144 | 1673 | 0 | 0 | 5,088 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | all | records | share |
| --- | --- | --- | --- | --- | --- | --- |
| eng | 2,213 | 277 | 263 | 2,753 | 5,506 | 99.78% |
| und | 2 | 2 | 0 | 4 | 8 | 0.14% |
| nld | 1 | 0 | 0 | 1 | 2 | 0.04% |
| spa | 1 | 0 | 0 | 1 | 2 | 0.04% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| all_records | 2,759 |
| all_unique_inputs | 2,511 |
| test_records | 263 |
| test_unique_inputs | 241 |
| train_records | 2,217 |
| train_unique_inputs | 2,011 |
| val_records | 279 |
| val_unique_inputs | 259 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| classifications (true_label) | train / val | 0.0593 | 4,082 / 527 | biotech_event: other 8%→6%, biotech_event: company description 15%→16%, biotech_event: alliance & partnership 7%→8%, biotech_event: product updates 2%→2%, biotech_event: expanding geography 3%→2% |
| classifications (true_label) | train / test | 0.0510 | 4,082 / 479 | biotech_event: support & philanthropy 3%→2%, biotech_event: hiring 1%→2%, biotech_event: alliance & partnership 7%→6%, biotech_event: company description 15%→15%, biotech_event: regulatory approval 1%→1% |
| classifications (true_label) | val / test | 0.0724 | 527 / 479 | biotech_event: alliance & partnership 8%→6%, biotech_event: support & philanthropy 3%→2%, biotech_event: other 6%→8%, biotech_event: hiring 1%→2%, biotech_event: expanding geography 2%→3% |

Thin support in val, classifications (true_label): 11 label(s) under 10 gold. biotech_event: ipo exit ×1, biotech_event: expanding industry ×1, biotech_event: closing ×2, biotech_event: regulatory approval ×3, biotech_event: subsidiary establishment ×3, biotech_event: clinical trial sponsorship ×4, biotech_event: investment in public company ×6, biotech_event: hiring ×6, biotech_event: service & product providing ×9, biotech_event: new initiatives & programs ×9, biotech_event: event organization ×9.
Thin support in test, classifications (true_label): 14 label(s) under 10 gold. biotech_event: partnerships & alliances ×1, biotech_event: subsidiary establishment ×1, biotech_event: ipo exit ×2, biotech_event: closing ×2, biotech_event: patent publication ×2, biotech_event: department establishment ×2, biotech_event: clinical trial sponsorship ×3, biotech_event: investment in public company ×4, biotech_event: regulatory approval ×4, biotech_event: expanding industry ×4, biotech_event: new initiatives & programs ×6, biotech_event: event organization ×7, biotech_event: support & philanthropy ×8, biotech_event: participation in an event ×9.
Thin support in all, classifications (true_label): 4 label(s) under 10 gold. biotech_event: partnerships & alliances ×2, biotech_event: patent publication ×6, biotech_event: subsidiary establishment ×8, biotech_event: department establishment ×9.

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

**train, biotech_event.** Majority class 28.2% of 4,082 answers. The marginal is `true_label`, not the menu.

#### train biotech_event

| label | count | share |
| --- | --- | --- |
| executive statement | 1,150 | 28.17% |
| company description | 608 | 14.89% |
| other | 328 | 8.04% |
| alliance & partnership | 291 | 7.13% |
| product launching & presentation | 243 | 5.95% |
| new initiatives or programs | 164 | 4.02% |
| m&a | 149 | 3.65% |
| support & philanthropy | 133 | 3.26% |
| expanding geography | 121 | 2.96% |
| article publication | 115 | 2.82% |
| executive appointment | 111 | 2.72% |
| funding round | 98 | 2.40% |
| service & product providing | 84 | 2.06% |
| product updates | 64 | 1.57% |
| event organization | 59 | 1.45% |
| participation in an event | 58 | 1.42% |
| hiring | 56 | 1.37% |
| regulatory approval | 56 | 1.37% |
| investment in public company | 50 | 1.22% |
| new initiatives & programs | 38 | 0.93% |
| expanding industry | 31 | 0.76% |
| ipo exit | 17 | 0.42% |
| clinical trial sponsorship | 16 | 0.39% |
| closing | 15 | 0.37% |
| foundation | 11 | 0.27% |
| department establishment | 7 | 0.17% |
| patent publication | 4 | 0.10% |
| subsidiary establishment | 4 | 0.10% |
| partnerships & alliances | 1 | 0.02% |

**val, biotech_event.** Majority class 28.1% of 527 answers. The marginal is `true_label`, not the menu.

#### val biotech_event

| label | count | share |
| --- | --- | --- |
| executive statement | 148 | 28.08% |
| company description | 85 | 16.13% |
| alliance & partnership | 43 | 8.16% |
| other | 34 | 6.45% |
| product launching & presentation | 31 | 5.88% |
| new initiatives or programs | 22 | 4.17% |
| m&a | 20 | 3.80% |
| support & philanthropy | 17 | 3.23% |
| article publication | 15 | 2.85% |
| product updates | 13 | 2.47% |
| funding round | 13 | 2.47% |
| executive appointment | 12 | 2.28% |
| expanding geography | 11 | 2.09% |
| participation in an event | 10 | 1.90% |
| service & product providing | 9 | 1.71% |
| new initiatives & programs | 9 | 1.71% |
| event organization | 9 | 1.71% |
| investment in public company | 6 | 1.14% |
| hiring | 6 | 1.14% |
| clinical trial sponsorship | 4 | 0.76% |
| regulatory approval | 3 | 0.57% |
| subsidiary establishment | 3 | 0.57% |
| closing | 2 | 0.38% |
| ipo exit | 1 | 0.19% |
| expanding industry | 1 | 0.19% |

**test, biotech_event.** Majority class 28.6% of 479 answers. The marginal is `true_label`, not the menu.

#### test biotech_event

| label | count | share |
| --- | --- | --- |
| executive statement | 137 | 28.60% |
| company description | 74 | 15.45% |
| other | 38 | 7.93% |
| alliance & partnership | 31 | 6.47% |
| product launching & presentation | 29 | 6.05% |
| new initiatives or programs | 17 | 3.55% |
| m&a | 16 | 3.34% |
| expanding geography | 14 | 2.92% |
| funding round | 14 | 2.92% |
| article publication | 12 | 2.51% |
| executive appointment | 11 | 2.30% |
| hiring | 11 | 2.30% |
| product updates | 10 | 2.09% |
| service & product providing | 10 | 2.09% |
| participation in an event | 9 | 1.88% |
| support & philanthropy | 8 | 1.67% |
| event organization | 7 | 1.46% |
| new initiatives & programs | 6 | 1.25% |
| investment in public company | 4 | 0.84% |
| regulatory approval | 4 | 0.84% |
| expanding industry | 4 | 0.84% |
| clinical trial sponsorship | 3 | 0.63% |
| ipo exit | 2 | 0.42% |
| closing | 2 | 0.42% |
| patent publication | 2 | 0.42% |
| department establishment | 2 | 0.42% |
| partnerships & alliances | 1 | 0.21% |
| subsidiary establishment | 1 | 0.21% |

**all, biotech_event.** Majority class 28.2% of 5,088 answers. The marginal is `true_label`, not the menu.

#### all biotech_event

| label | count | share |
| --- | --- | --- |
| executive statement | 1,435 | 28.20% |
| company description | 767 | 15.07% |
| other | 400 | 7.86% |
| alliance & partnership | 365 | 7.17% |
| product launching & presentation | 303 | 5.96% |
| new initiatives or programs | 203 | 3.99% |
| m&a | 185 | 3.64% |
| support & philanthropy | 158 | 3.11% |
| expanding geography | 146 | 2.87% |
| article publication | 142 | 2.79% |
| executive appointment | 134 | 2.63% |
| funding round | 125 | 2.46% |
| service & product providing | 103 | 2.02% |
| product updates | 87 | 1.71% |
| participation in an event | 77 | 1.51% |
| event organization | 75 | 1.47% |
| hiring | 73 | 1.43% |
| regulatory approval | 63 | 1.24% |
| investment in public company | 60 | 1.18% |
| new initiatives & programs | 53 | 1.04% |
| expanding industry | 36 | 0.71% |
| clinical trial sponsorship | 23 | 0.45% |
| ipo exit | 20 | 0.39% |
| closing | 19 | 0.37% |
| foundation | 11 | 0.22% |
| department establishment | 9 | 0.18% |
| subsidiary establishment | 8 | 0.16% |
| patent publication | 6 | 0.12% |
| partnerships & alliances | 2 | 0.04% |

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
