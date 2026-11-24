# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scientific_text | train | 40,045 | 40,045 | 0 | 0 | 40,045 | 0 | 0 | 0 |
| scientific_text | val | 4,996 | 4,996 | 0 | 0 | 4,996 | 0 | 0 | 0 |
| scientific_text | test | 4,956 | 4,956 | 0 | 0 | 4,956 | 0 | 0 | 0 |
| scientific_text | all | 50,000 | 49,997 | 0 | 0 | 50,000 | 0 | 0 | 0 |

## scientific_text

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 40,045 | 40,045 | 178 | 291 | 903 | 0 | 0 | 40,045 | 0 | 0 | 0 |
| val | 4,996 | 4,996 | 176 | 292 | 714 | 0 | 0 | 4,996 | 0 | 0 | 0 |
| test | 4,956 | 4,956 | 178 | 287 | 761 | 0 | 0 | 4,956 | 0 | 0 | 0 |
| all | 50,000 | 49,997 | 178 | 291 | 903 | 0 | 0 | 50,000 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | all | records | share |
| --- | --- | --- | --- | --- | --- | --- |
| eng | 39,960 | 4,983 | 4,946 | 49,892 | 99,781 | 99.78% |
| und | 83 | 13 | 10 | 106 | 212 | 0.21% |
| deu | 2 | 0 | 0 | 2 | 4 | 0.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| all_records | 50,000 |
| all_unique_inputs | 49,997 |
| test_records | 4,956 |
| test_unique_inputs | 4,956 |
| train_records | 40,045 |
| train_unique_inputs | 40,045 |
| val_records | 4,996 |
| val_unique_inputs | 4,996 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| classifications (true_label) | train / val | 0.0185 | 40,045 / 4,996 | scientific_domain: computer science 10%→9%, scientific_domain: statistics 10%→10%, scientific_domain: electrical engineering and systems science 10%→10%, scientific_domain: physics 10%→10%, scientific_domain: high energy physics theory 10%→10% |
| classifications (true_label) | train / test | 0.0145 | 40,045 / 4,956 | scientific_domain: statistics 10%→11%, scientific_domain: quantum physics 10%→10%, scientific_domain: high energy physics theory 10%→10%, scientific_domain: condensed matter 10%→10%, scientific_domain: astrophysics 10%→10% |
| classifications (true_label) | val / test | 0.0213 | 4,996 / 4,956 | scientific_domain: computer science 9%→10%, scientific_domain: high energy physics theory 10%→10%, scientific_domain: quantum physics 10%→10%, scientific_domain: electrical engineering and systems science 10%→10%, scientific_domain: physics 10%→10% |


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

**train, scientific_domain.** Majority class 10.2% of 40,045 answers. The marginal is `true_label`, not the menu.

#### train scientific_domain

| label | count | share |
| --- | --- | --- |
| quantum physics | 4,069 | 10.16% |
| computer science | 4,031 | 10.07% |
| high energy physics theory | 4,026 | 10.05% |
| mathematics | 4,025 | 10.05% |
| electrical engineering and systems science | 4,024 | 10.05% |
| astrophysics | 4,013 | 10.02% |
| physics | 4,003 | 10.00% |
| condensed matter | 3,993 | 9.97% |
| high energy physics phenomenology | 3,977 | 9.93% |
| statistics | 3,884 | 9.70% |

**val, scientific_domain.** Majority class 10.5% of 4,996 answers. The marginal is `true_label`, not the menu.

#### val scientific_domain

| label | count | share |
| --- | --- | --- |
| statistics | 523 | 10.47% |
| physics | 518 | 10.37% |
| high energy physics theory | 517 | 10.35% |
| mathematics | 515 | 10.31% |
| quantum physics | 514 | 10.29% |
| astrophysics | 502 | 10.05% |
| high energy physics phenomenology | 493 | 9.87% |
| condensed matter | 489 | 9.79% |
| electrical engineering and systems science | 475 | 9.51% |
| computer science | 450 | 9.01% |

**test, scientific_domain.** Majority class 10.7% of 4,956 answers. The marginal is `true_label`, not the menu.

#### test scientific_domain

| label | count | share |
| --- | --- | --- |
| statistics | 531 | 10.71% |
| computer science | 507 | 10.23% |
| mathematics | 502 | 10.13% |
| physics | 499 | 10.07% |
| high energy physics phenomenology | 498 | 10.05% |
| electrical engineering and systems science | 495 | 9.99% |
| astrophysics | 486 | 9.81% |
| high energy physics theory | 481 | 9.71% |
| quantum physics | 480 | 9.69% |
| condensed matter | 477 | 9.62% |

**all, scientific_domain.** Majority class 10.1% of 50,000 answers. The marginal is `true_label`, not the menu.

#### all scientific_domain

| label | count | share |
| --- | --- | --- |
| quantum physics | 5,064 | 10.13% |
| mathematics | 5,042 | 10.08% |
| high energy physics theory | 5,024 | 10.05% |
| physics | 5,020 | 10.04% |
| astrophysics | 5,001 | 10.00% |
| electrical engineering and systems science | 4,994 | 9.99% |
| computer science | 4,988 | 9.98% |
| high energy physics phenomenology | 4,970 | 9.94% |
| condensed matter | 4,959 | 9.92% |
| statistics | 4,938 | 9.88% |

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
