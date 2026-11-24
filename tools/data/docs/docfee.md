# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| docfee | train | 16,384 | 16,384 | 68,698 | 0 | 16,737 | 0 | 0 | 0 |
| docfee | val | 1,823 | 1,823 | 7,539 | 0 | 1,858 | 0 | 0 | 0 |
| docfee | test | 800 | 800 | 3,236 | 0 | 808 | 0 | 0 | 0 |

## docfee

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 16,384 | 16,384 | 1337 | 3442 | 31970 | 68,698 | 0 | 16,737 | 0 | 0 | 0 |
| val | 1,823 | 1,823 | 1320 | 3572 | 29953 | 7,539 | 0 | 1,858 | 0 | 0 | 0 |
| test | 800 | 800 | 896 | 2173 | 29957 | 3,236 | 0 | 808 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| zho | 16,384 | 1,823 | 800 | 19,007 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 800 |
| test_unique_inputs | 800 |
| train_records | 16,384 |
| train_unique_inputs | 16,384 |
| val_records | 1,823 |
| val_unique_inputs | 1,823 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0334 | 68,698 / 7,539 | Company Name 10%→10%, Reduction Amount 17%→16%, Accepting Court 4%→4%, ruling_date 2%→3%, Announcement Date 8%→8% |
| entities (per mention) | train / test | 0.2122 WARN | 68,698 / 3,236 | Reduction Amount 17%→10%, Reducing Shareholder 13%→6%, Reduction Start Date 10%→7%, Increase Amount 3%→5%, Company Name 10%→12% |
| entities (per mention) | val / test | 0.2009 WARN | 7,539 / 3,236 | Reduction Amount 16%→10%, Reducing Shareholder 12%→6%, Increase Amount 2%→5%, Reduction Start Date 9%→7%, Increase Start Date 1%→4% |
| classifications (true_label) | train / val | 0.0303 | 16,737 / 1,858 | docfee_event: BankruptcyLiquidation 17%→19%, docfee_event: EquityUnderweight 34%→33%, docfee_event: MajorAssetLoss 13%→14%, docfee_event: EquityOverweight 6%→5%, docfee_event: EquityPledge 15%→15% |
| classifications (true_label) | train / test | 0.2240 WARN | 16,737 / 808 | docfee_event: EquityUnderweight 34%→18%, docfee_event: BankruptcyLiquidation 17%→22%, docfee_event: EquityOverweight 6%→11%, docfee_event: ExecutiveDeath 3%→7%, docfee_event: EquityFreeze 6%→11% |
| classifications (true_label) | val / test | 0.2104 WARN | 1,858 / 808 | docfee_event: EquityUnderweight 33%→18%, docfee_event: EquityOverweight 5%→11%, docfee_event: EquityFreeze 6%→11%, docfee_event: ExecutiveDeath 3%→7%, docfee_event: EquityPledge 15%→11% |

Thin support in val, entities (per mention): 2 label(s) under 10 gold. Other Impact ×6, Death Age ×8.
Thin support in test, entities (per mention): 3 label(s) under 10 gold. Other Loss ×4, Death Age ×4, Other Impact ×5.

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
| Reduction Amount | 11,590 | 16.87% |
| Reducing Shareholder | 8,842 | 12.87% |
| Reduction Start Date | 6,648 | 9.68% |
| Company Name | 6,627 | 9.65% |
| Announcement Date | 5,477 | 7.97% |
| pledge_amount | 3,747 | 5.45% |
| pledgor | 2,826 | 4.11% |
| Recipient | 2,664 | 3.88% |
| Accepting Court | 2,506 | 3.65% |
| Loss Amount | 2,179 | 3.17% |
| Increase Amount | 1,755 | 2.55% |
| pledge_start_date | 1,622 | 2.36% |
| ruling_date | 1,547 | 2.25% |
| Increasing Shareholder | 1,532 | 2.23% |
| Frozen Amount | 1,530 | 2.23% |
| frozen_shareholder | 1,231 | 1.79% |
| Increase Start Date | 1,073 | 1.56% |
| pledge_end_date | 1,040 | 1.51% |
| Freeze Start Date | 907 | 1.32% |
| compensation_amount | 568 | 0.83% |
| compensation_recipient | 507 | 0.74% |
| senior_personnel | 449 | 0.65% |
| senior_position | 443 | 0.64% |
| Freeze End Date | 413 | 0.60% |
| Death Time | 298 | 0.43% |
| Other Loss | 224 | 0.33% |
| Casualty Count | 210 | 0.31% |
| Company Industry | 127 | 0.18% |
| Other Impact | 78 | 0.11% |
| Death Age | 38 | 0.06% |

#### val

| label | count | share |
| --- | --- | --- |
| Reduction Amount | 1,223 | 16.22% |
| Reducing Shareholder | 940 | 12.47% |
| Company Name | 780 | 10.35% |
| Reduction Start Date | 709 | 9.40% |
| Announcement Date | 633 | 8.40% |
| pledge_amount | 381 | 5.05% |
| Accepting Court | 312 | 4.14% |
| pledgor | 311 | 4.13% |
| Recipient | 295 | 3.91% |
| Loss Amount | 269 | 3.57% |
| ruling_date | 205 | 2.72% |
| Increase Amount | 175 | 2.32% |
| Increasing Shareholder | 162 | 2.15% |
| Frozen Amount | 156 | 2.07% |
| pledge_start_date | 147 | 1.95% |
| frozen_shareholder | 144 | 1.91% |
| pledge_end_date | 103 | 1.37% |
| Increase Start Date | 101 | 1.34% |
| Freeze Start Date | 90 | 1.19% |
| compensation_amount | 75 | 0.99% |
| compensation_recipient | 63 | 0.84% |
| senior_personnel | 55 | 0.73% |
| senior_position | 55 | 0.73% |
| Death Time | 43 | 0.57% |
| Freeze End Date | 39 | 0.52% |
| Casualty Count | 28 | 0.37% |
| Other Loss | 16 | 0.21% |
| Company Industry | 15 | 0.20% |
| Death Age | 8 | 0.11% |
| Other Impact | 6 | 0.08% |

#### test

| label | count | share |
| --- | --- | --- |
| Company Name | 393 | 12.14% |
| Reduction Amount | 317 | 9.80% |
| Announcement Date | 316 | 9.77% |
| Reduction Start Date | 227 | 7.01% |
| Reducing Shareholder | 206 | 6.37% |
| Increase Amount | 166 | 5.13% |
| Accepting Court | 151 | 4.67% |
| pledge_amount | 138 | 4.26% |
| Increasing Shareholder | 133 | 4.11% |
| Frozen Amount | 127 | 3.92% |
| Increase Start Date | 120 | 3.71% |
| frozen_shareholder | 96 | 2.97% |
| pledgor | 94 | 2.90% |
| Freeze Start Date | 94 | 2.90% |
| Recipient | 93 | 2.87% |
| ruling_date | 91 | 2.81% |
| pledge_start_date | 85 | 2.63% |
| Loss Amount | 74 | 2.29% |
| senior_personnel | 56 | 1.73% |
| senior_position | 54 | 1.67% |
| pledge_end_date | 54 | 1.67% |
| Freeze End Date | 37 | 1.14% |
| Death Time | 33 | 1.02% |
| Casualty Count | 23 | 0.71% |
| compensation_amount | 18 | 0.56% |
| compensation_recipient | 17 | 0.53% |
| Company Industry | 10 | 0.31% |
| Other Impact | 5 | 0.15% |
| Other Loss | 4 | 0.12% |
| Death Age | 4 | 0.12% |

### Relations

None.

### Classifications

**train, docfee_event.** Majority class 34.5% of 16,737 answers. The marginal is `true_label`, not the menu.

#### train docfee_event

| label | count | share |
| --- | --- | --- |
| EquityUnderweight | 5,768 | 34.46% |
| BankruptcyLiquidation | 2,803 | 16.75% |
| EquityPledge | 2,507 | 14.98% |
| MajorAssetLoss | 2,134 | 12.75% |
| EquityFreeze | 1,066 | 6.37% |
| EquityOverweight | 970 | 5.80% |
| MajorSafetyAccident | 573 | 3.42% |
| MajorExternalPayout | 466 | 2.78% |
| ExecutiveDeath | 450 | 2.69% |

**val, docfee_event.** Majority class 32.9% of 1,858 answers. The marginal is `true_label`, not the menu.

#### val docfee_event

| label | count | share |
| --- | --- | --- |
| EquityUnderweight | 612 | 32.94% |
| BankruptcyLiquidation | 347 | 18.68% |
| EquityPledge | 271 | 14.59% |
| MajorAssetLoss | 251 | 13.51% |
| EquityFreeze | 114 | 6.14% |
| EquityOverweight | 95 | 5.11% |
| MajorSafetyAccident | 65 | 3.50% |
| ExecutiveDeath | 55 | 2.96% |
| MajorExternalPayout | 48 | 2.58% |

**test, docfee_event.** Majority class 22.2% of 808 answers. The marginal is `true_label`, not the menu.

#### test docfee_event

| label | count | share |
| --- | --- | --- |
| BankruptcyLiquidation | 179 | 22.15% |
| EquityUnderweight | 144 | 17.82% |
| MajorAssetLoss | 91 | 11.26% |
| EquityPledge | 89 | 11.01% |
| EquityOverweight | 88 | 10.89% |
| EquityFreeze | 85 | 10.52% |
| ExecutiveDeath | 56 | 6.93% |
| MajorSafetyAccident | 56 | 6.93% |
| MajorExternalPayout | 20 | 2.48% |

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
