# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chfinann | train | 25,632 | 25,627 | 259,451 | 0 | 25,894 | 0 | 0 | 0 |
| chfinann | val | 3,204 | 3,204 | 34,614 | 0 | 3,221 | 0 | 0 | 0 |
| chfinann | test | 3,204 | 3,204 | 33,703 | 0 | 3,224 | 0 | 0 | 0 |

## chfinann

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 25,632 | 25,627 | 563 | 1189 | 5477 | 259,451 | 0 | 25,894 | 0 | 0 | 0 |
| val | 3,204 | 3,204 | 665 | 1433 | 4806 | 34,614 | 0 | 3,221 | 0 | 0 | 0 |
| test | 3,204 | 3,204 | 597 | 1325 | 5024 | 33,703 | 0 | 3,224 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| zho | 25,632 | 3,204 | 3,204 | 32,040 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 3,204 |
| test_unique_inputs | 3,204 |
| train_records | 25,632 |
| train_unique_inputs | 25,627 |
| val_records | 3,204 |
| val_unique_inputs | 3,204 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0968 | 259,451 / 34,614 | TradedShares 6%→4%, EquityHolder 11%→9%, HighestTradingPrice 1%→3%, ClosingDate 1%→2%, RepurchaseAmount 1%→2% |
| entities (per mention) | train / test | 0.1839 WARN | 259,451 / 33,703 | HighestTradingPrice 1%→4%, RepurchaseAmount 1%→4%, EquityHolder 11%→8%, ClosingDate 1%→4%, TradedShares 6%→3% |
| entities (per mention) | val / test | 0.0906 | 34,614 / 33,703 | RepurchaseAmount 2%→4%, HighestTradingPrice 3%→4%, StartDate 11%→10%, ClosingDate 2%→4%, LowestTradingPrice 1%→3% |
| classifications (true_label) | train / val | 0.1649 WARN | 25,894 / 3,221 | chfinann_event: EquityRepurchase 7%→21%, chfinann_event: EquityUnderweight 20%→9%, chfinann_event: EquityPledge 50%→46%, chfinann_event: EquityFreeze 3%→6%, chfinann_event: EquityOverweight 20%→18% |
| classifications (true_label) | train / test | 0.3132 STRONG | 25,894 / 3,224 | chfinann_event: EquityRepurchase 7%→35%, chfinann_event: EquityUnderweight 20%→9%, chfinann_event: EquityPledge 50%→39%, chfinann_event: EquityOverweight 20%→11%, chfinann_event: EquityFreeze 3%→6% |
| classifications (true_label) | val / test | 0.1483 | 3,221 / 3,224 | chfinann_event: EquityRepurchase 21%→35%, chfinann_event: EquityPledge 46%→39%, chfinann_event: EquityOverweight 18%→11%, chfinann_event: EquityFreeze 6%→6%, chfinann_event: EquityUnderweight 9%→9% |


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
| StartDate | 31,365 | 12.09% |
| EquityHolder | 28,360 | 10.93% |
| StockCode | 25,299 | 9.75% |
| CompanyName | 21,457 | 8.27% |
| StockAbbr | 21,363 | 8.23% |
| PledgedShares | 20,053 | 7.73% |
| Pledgee | 16,435 | 6.33% |
| TotalHoldingShares | 16,110 | 6.21% |
| TotalHoldingRatio | 14,620 | 5.63% |
| TradedShares | 14,374 | 5.54% |
| TotalPledgedShares | 11,346 | 4.37% |
| EndDate | 9,628 | 3.71% |
| ReleasedDate | 9,232 | 3.56% |
| LaterHoldingShares | 4,859 | 1.87% |
| AveragePrice | 3,713 | 1.43% |
| RepurchasedShares | 2,500 | 0.96% |
| HighestTradingPrice | 2,267 | 0.87% |
| RepurchaseAmount | 1,698 | 0.65% |
| ClosingDate | 1,597 | 0.62% |
| LegalInstitution | 1,105 | 0.43% |
| FrozeShares | 945 | 0.36% |
| LowestTradingPrice | 916 | 0.35% |
| UnfrozeDate | 209 | 0.08% |

#### val

| label | count | share |
| --- | --- | --- |
| StartDate | 3,897 | 11.26% |
| EquityHolder | 3,186 | 9.20% |
| CompanyName | 3,183 | 9.20% |
| StockCode | 3,171 | 9.16% |
| StockAbbr | 3,171 | 9.16% |
| PledgedShares | 2,692 | 7.78% |
| TotalHoldingShares | 2,067 | 5.97% |
| TotalHoldingRatio | 1,864 | 5.39% |
| Pledgee | 1,766 | 5.10% |
| TotalPledgedShares | 1,484 | 4.29% |
| TradedShares | 1,286 | 3.72% |
| EndDate | 1,216 | 3.51% |
| HighestTradingPrice | 884 | 2.55% |
| RepurchasedShares | 836 | 2.42% |
| RepurchaseAmount | 747 | 2.16% |
| ClosingDate | 740 | 2.14% |
| ReleasedDate | 739 | 2.13% |
| LowestTradingPrice | 496 | 1.43% |
| LaterHoldingShares | 388 | 1.12% |
| AveragePrice | 330 | 0.95% |
| LegalInstitution | 288 | 0.83% |
| FrozeShares | 172 | 0.50% |
| UnfrozeDate | 11 | 0.03% |

#### test

| label | count | share |
| --- | --- | --- |
| StartDate | 3,237 | 9.60% |
| CompanyName | 3,189 | 9.46% |
| StockCode | 3,184 | 9.45% |
| StockAbbr | 3,174 | 9.42% |
| EquityHolder | 2,621 | 7.78% |
| PledgedShares | 2,374 | 7.04% |
| TotalHoldingShares | 1,724 | 5.12% |
| TotalHoldingRatio | 1,535 | 4.55% |
| HighestTradingPrice | 1,487 | 4.41% |
| Pledgee | 1,471 | 4.36% |
| RepurchaseAmount | 1,384 | 4.11% |
| TotalPledgedShares | 1,287 | 3.82% |
| ClosingDate | 1,267 | 3.76% |
| RepurchasedShares | 1,228 | 3.64% |
| EndDate | 1,100 | 3.26% |
| LowestTradingPrice | 1,002 | 2.97% |
| TradedShares | 910 | 2.70% |
| ReleasedDate | 547 | 1.62% |
| LaterHoldingShares | 344 | 1.02% |
| LegalInstitution | 290 | 0.86% |
| FrozeShares | 173 | 0.51% |
| AveragePrice | 162 | 0.48% |
| UnfrozeDate | 13 | 0.04% |

### Relations

None.

### Classifications

**train, chfinann_event.** Majority class 49.7% of 25,894 answers. The marginal is `true_label`, not the menu.

#### train chfinann_event

| label | count | share |
| --- | --- | --- |
| EquityPledge | 12,857 | 49.65% |
| EquityUnderweight | 5,268 | 20.34% |
| EquityOverweight | 5,101 | 19.70% |
| EquityRepurchase | 1,862 | 7.19% |
| EquityFreeze | 806 | 3.11% |

**val, chfinann_event.** Majority class 46.3% of 3,221 answers. The marginal is `true_label`, not the menu.

#### val chfinann_event

| label | count | share |
| --- | --- | --- |
| EquityPledge | 1,491 | 46.29% |
| EquityRepurchase | 677 | 21.02% |
| EquityOverweight | 570 | 17.70% |
| EquityUnderweight | 297 | 9.22% |
| EquityFreeze | 186 | 5.77% |

**test, chfinann_event.** Majority class 38.9% of 3,224 answers. The marginal is `true_label`, not the menu.

#### test chfinann_event

| label | count | share |
| --- | --- | --- |
| EquityPledge | 1,254 | 38.90% |
| EquityRepurchase | 1,138 | 35.30% |
| EquityOverweight | 346 | 10.73% |
| EquityUnderweight | 282 | 8.75% |
| EquityFreeze | 204 | 6.33% |

### Structures

None.

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | distant |
| relations | distant |
| events | distant |
| classifications | distant |
| structures | distant |
