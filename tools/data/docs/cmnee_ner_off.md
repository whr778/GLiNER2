# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cmnee_ner_off | train | 9,281 | 9,281 | 120,985 | 0 | 0 | 0 | 15,962 | 57,641 |

## cmnee_ner_off

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 9,281 | 9,281 | 302 | 395 | 487 | 120,985 | 0 | 0 | 0 | 15,962 | 57,641 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | records | share |
| --- | --- | --- | --- |
| zho | 9,281 | 9,281 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| train_records | 9,281 |
| train_unique_inputs | 9,281 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

No label marginal has gold on both sides of a split pair.


### Events

**train.** 19,422 instances in 9,281 records, 19,420 triggerless. Trigger gold 15,962. Argument gold 57,641. Arguments per instance, mean 3.22. Trigger length in whitespace tokens, mean 1.50, single-token share 50.0%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Experiment | 1 | 50.00% |
| Deploy | 1 | 50.00% |
| Support | 0 | 0.00% |
| Manoeuvre | 0 | 0.00% |
| Exhibit | 0 | 0.00% |
| Accident | 0 | 0.00% |
| Conflict | 0 | 0.00% |
| Injure | 0 | 0.00% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2 | 62,284 | 0.00% | 0 | 2 | 202 | 3 | 62,077 | 289 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

**train by role.**

| role | inside | placed | inside share | unmeasured |
| --- | --- | --- | --- | --- |
| Subject | 0 | 20,632 | 0.00% | 111 |
| Equipment | 1 | 10,438 | 0.01% | 58 |
| Date | 0 | 8,926 | 0.00% | 58 |
| Object | 0 | 6,280 | 0.00% | 9 |
| Materials | 1 | 5,854 | 0.02% | 21 |
| Location | 0 | 5,470 | 0.00% | 6 |
| Militaryforce | 0 | 2,714 | 0.00% | 22 |
| Content | 0 | 642 | 0.00% | 2 |
| Result | 0 | 602 | 0.00% | 0 |
| Quantity | 0 | 389 | 0.00% | 1 |
| Area | 0 | 337 | 0.00% | 1 |

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2.19 | 51.28% | 1.84 | 36.04% | 0.02% | 62,290 | 283 | 19,420 | 2 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 19,422 | 0.00% | 8,651 | 76.50% | 75.56% | 7.24% | 2.33% | 32.23% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 0 | 0 | 8,651 |

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

| split | mentions the string rule adds | arguments skipped, 2+ clusters | arguments in one multi-mention cluster | other mentions of that cluster | argument spans in 2+ clusters |
| --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | 0 | 0 | 0 |

### Entities

#### train

| label | count | share |
| --- | --- | --- |
| Product | 26,528 | 21.93% |
| Organization | 24,449 | 20.21% |
| Date | 16,908 | 13.98% |
| Location | 16,311 | 13.48% |
| Quantity | 9,885 | 8.17% |
| Measurement | 8,616 | 7.12% |
| Money | 3,926 | 3.25% |
| Nationality | 3,488 | 2.88% |
| Facility | 3,436 | 2.84% |
| Person | 2,986 | 2.47% |
| Event | 2,016 | 1.67% |
| Occupation | 1,406 | 1.16% |
| Time | 541 | 0.45% |
| Publication | 126 | 0.10% |
| Region | 107 | 0.09% |
| Ship | 57 | 0.05% |
| Military_Unit | 25 | 0.02% |
| Military Unit | 21 | 0.02% |
| Weapon | 20 | 0.02% |
| Year | 17 | 0.01% |
| Source | 14 | 0.01% |
| Vessel | 12 | 0.01% |
| Military | 10 | 0.01% |
| Company | 9 | 0.01% |
| Media | 6 | 0.00% |
| Project | 6 | 0.00% |
| Organization_Military | 5 | 0.00% |
| Technology | 4 | 0.00% |
| Document | 4 | 0.00% |
| System | 4 | 0.00% |
| Aircraft | 4 | 0.00% |
| Equipment | 3 | 0.00% |
| Product_Class | 3 | 0.00% |
| Organization_other | 3 | 0.00% |
| Material | 3 | 0.00% |
| Language | 2 | 0.00% |
| Rank | 2 | 0.00% |
| Country | 2 | 0.00% |
| Legislation | 2 | 0.00% |
| Newspaper | 2 | 0.00% |
| Time_Period | 2 | 0.00% |
| Frequency | 2 | 0.00% |
| Vehicle | 1 | 0.00% |
| Program | 1 | 0.00% |
| Abbreviation | 1 | 0.00% |
| Date_planned | 1 | 0.00% |
| Organization_Other | 1 | 0.00% |
| Military_Rank | 1 | 0.00% |
| Regulation | 1 | 0.00% |
| Percentage | 1 | 0.00% |
| Ship_Class | 1 | 0.00% |
| Client | 1 | 0.00% |
| Organization Affiliation | 1 | 0.00% |
| Acronym | 1 | 0.00% |

### Relations

None.

### Classifications

None.

### Structures

None.

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | human |
| relations | human |
| events | human |
| classifications | human |
| structures | human |
