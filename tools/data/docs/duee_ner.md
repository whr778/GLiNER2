# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| duee_ner | train | 11,323 | 11,320 | 46,330 | 0 | 0 | 0 | 12,764 | 27,899 |
| duee_ner | val | 1,414 | 1,414 | 5,841 | 0 | 0 | 0 | 1,620 | 3,519 |

## duee_ner

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 11,323 | 11,320 | 37 | 110 | 355 | 46,330 | 0 | 0 | 0 | 12,764 | 27,899 |
| val | 1,414 | 1,414 | 36 | 112 | 316 | 5,841 | 0 | 0 | 0 | 1,620 | 3,519 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | records | share |
| --- | --- | --- | --- | --- |
| zho | 9,938 | 1,240 | 11,178 | 87.76% |
| und | 1,257 | 154 | 1,411 | 11.08% |
| eng | 26 | 8 | 34 | 0.27% |
| fra | 15 | 2 | 17 | 0.13% |
| por | 9 | 3 | 12 | 0.09% |
| wuu | 7 | 2 | 9 | 0.07% |
| rus | 8 | 1 | 9 | 0.07% |
| kor | 8 | 0 | 8 | 0.06% |
| ita | 7 | 0 | 7 | 0.05% |
| deu | 3 | 2 | 5 | 0.04% |
| spa | 5 | 0 | 5 | 0.04% |
| bre | 3 | 1 | 4 | 0.03% |
| jpn | 3 | 0 | 3 | 0.02% |
| fin | 3 | 0 | 3 | 0.02% |
| hun | 3 | 0 | 3 | 0.02% |
| ukr | 3 | 0 | 3 | 0.02% |
| epo | 3 | 0 | 3 | 0.02% |
| ceb | 2 | 0 | 2 | 0.02% |
| ces | 2 | 0 | 2 | 0.02% |
| vie | 2 | 0 | 2 | 0.02% |
| nor | 2 | 0 | 2 | 0.02% |
| slk | 1 | 0 | 1 | 0.01% |
| fry | 1 | 0 | 1 | 0.01% |
| cat | 1 | 0 | 1 | 0.01% |
| lav | 1 | 0 | 1 | 0.01% |
| eml | 1 | 0 | 1 | 0.01% |
| min | 1 | 0 | 1 | 0.01% |
| war | 1 | 0 | 1 | 0.01% |
| lit | 1 | 0 | 1 | 0.01% |
| swe | 1 | 0 | 1 | 0.01% |
| ind | 1 | 0 | 1 | 0.01% |
| pms | 1 | 0 | 1 | 0.01% |
| mlg | 1 | 0 | 1 | 0.01% |
| eus | 1 | 0 | 1 | 0.01% |
| pol | 1 | 0 | 1 | 0.01% |
| tur | 0 | 1 | 1 | 0.01% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| train_records | 11,323 |
| train_unique_inputs | 11,320 |
| val_records | 1,414 |
| val_unique_inputs | 1,414 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0229 | 46,330 / 5,841 | Person 19%→20%, Organization 23%→22%, Quantity 9%→9%, Occupation 3%→4%, Location 11%→10% |
| event types (per trigger mention) | train / val | 0.0366 | 13,120 / 1,681 | match_result 15%→16%, advancement 2%→2%, championship_win 3%→4%, Earthquake 1%→1%, resignation 4%→4% |
| event roles (per argument) | train / val | 0.0422 | 28,399 / 3,619 | Publisher 3%→2%, event_name 4%→5%, Advancing Party 1%→1%, loser 6%→6%, Winner 7%→7% |

Unseen in train, present in val, entities (per mention): Law ×2, Standard ×2, Activity ×1, Feature ×1.
Thin support in val, entities (per mention): 5 label(s) under 10 gold. Activity ×1, Feature ×1, Law ×2, Standard ×2, Processor ×3.
Thin support in val, event types (per trigger mention): 19 label(s) under 10 gold. rate_increase ×3, infidelity ×4, rate_decrease ×4, price_increase ×5, wedding ×5, contract_termination ×5, Flood ×5, pregnancy ×7, strike ×7, engagement ×7, ipo ×8, thank ×8, suspension ×8, judicial_action_filing ×9, meet_with ×9, Explosion ×9, price_decrease ×9, Collapse ×9, closing ×9.
Thin support in val, event roles (per argument): 48 label(s) under 10 gold. release_by ×1, Affair Partner ×1, Attendee ×1, Strike Participant Count ×1, Rate Increase Magnitude ×1, Protest Organizer ×2, apology_target ×3, rate_cut_magnitude ×3, contract_termination_target ×3, Rate Raising Institution ×3, Cheating Party ×4, price_reducer ×4, rate_cutting_institution ×4, epicenter ×5, delisted_party ×5, Price Increased Item ×5, Born Person ×5, thanked_person ×5, attack_target ×5, Striker ×5, contract_terminator ×5, Protest Participants ×5, Birthday Person Age ×6, focal_depth ×6, price_reduction_magnitude ×6, Liker ×7, Pregnant Person ×7, report_initiator ×7, Listed Company ×8, Case Authority ×8, Court ×8, Proposer ×8, Ban Authority ×8, withdrawn_event ×8, Thanking Party ×8, Suspended Personnel ×8, Visit Target ×9, dismissal_party ×9, Case Subject ×9, Ban Duration ×9, meet_object ×9, price_reduced_item ×9, follow_investor ×9, Missing Person ×9, attacker ×9, layoff_count ×9, Collapsed Structure ×9, engagement_subject ×9.

### Events

**train.** 13,120 instances in 11,323 records, 0 triggerless. Trigger gold 12,764. Argument gold 27,899. Arguments per instance, mean 2.16. Trigger length in whitespace tokens, mean 2.05, single-token share 3.1%.

#### train event types

| label | count | share |
| --- | --- | --- |
| match_result | 1,982 | 15.11% |
| product_publish | 1,226 | 9.34% |
| death | 702 | 5.35% |
| judicial_action_arrest | 648 | 4.94% |
| resignation | 587 | 4.47% |
| championship_win | 456 | 3.48% |
| joined | 354 | 2.70% |
| advancement | 322 | 2.45% |
| product_recall | 297 | 2.26% |
| product_release | 294 | 2.24% |
| marriage | 293 | 2.23% |
| divorce | 269 | 2.05% |
| judicial_action_summon | 252 | 1.92% |
| opening | 236 | 1.80% |
| judicial_action_fine | 235 | 1.79% |
| price_limit_up | 215 | 1.64% |
| product_delist | 195 | 1.49% |
| Traffic Accident | 193 | 1.47% |
| departure | 191 | 1.46% |
| sale_acquisition | 191 | 1.46% |
| Fire | 181 | 1.38% |
| judicial_action_imprisonment | 160 | 1.22% |
| judicial_action_prosecution | 152 | 1.16% |
| competition_withdrawal | 142 | 1.08% |
| layoff | 138 | 1.05% |
| apologize_to | 137 | 1.04% |
| competition_ban | 136 | 1.04% |
| birthday_celebration | 133 | 1.01% |
| product_award | 133 | 1.01% |
| breakup | 119 | 0.91% |
| Attack | 118 | 0.90% |
| financing | 117 | 0.89% |
| Earthquake | 110 | 0.84% |
| birth | 101 | 0.77% |
| retirement | 99 | 0.75% |
| judicial_action_hearing | 99 | 0.75% |
| price_limit_down | 95 | 0.72% |
| dismissal | 91 | 0.69% |
| lost_contact | 90 | 0.69% |
| praise | 90 | 0.69% |
| judicial_action_report | 89 | 0.68% |
| suspension | 83 | 0.63% |
| price_decrease | 82 | 0.62% |
| dissolution | 81 | 0.62% |
| judicial_action_filing | 78 | 0.59% |
| Collapse | 77 | 0.59% |
| Plane Crash | 77 | 0.59% |
| proposal | 73 | 0.56% |
| meet_with | 70 | 0.53% |
| visit_on_set | 68 | 0.52% |
| strike | 66 | 0.50% |
| Explosion | 65 | 0.50% |
| price_increase | 61 | 0.46% |
| closing | 61 | 0.46% |
| wedding | 60 | 0.46% |
| thank | 58 | 0.44% |
| pregnancy | 58 | 0.44% |
| engagement | 58 | 0.44% |
| march | 53 | 0.40% |
| ipo | 50 | 0.38% |
| contract_termination | 45 | 0.34% |
| Flood | 40 | 0.30% |
| infidelity | 32 | 0.24% |
| rate_decrease | 31 | 0.24% |
| rate_increase | 25 | 0.19% |

**val.** 1,681 instances in 1,414 records, 0 triggerless. Trigger gold 1,620. Argument gold 3,519. Arguments per instance, mean 2.15. Trigger length in whitespace tokens, mean 2.07, single-token share 3.5%.

#### val event types

| label | count | share |
| --- | --- | --- |
| match_result | 269 | 16.00% |
| product_publish | 153 | 9.10% |
| death | 91 | 5.41% |
| judicial_action_arrest | 79 | 4.70% |
| resignation | 71 | 4.22% |
| championship_win | 64 | 3.81% |
| joined | 42 | 2.50% |
| marriage | 41 | 2.44% |
| product_recall | 36 | 2.14% |
| divorce | 36 | 2.14% |
| product_release | 34 | 2.02% |
| advancement | 34 | 2.02% |
| judicial_action_fine | 33 | 1.96% |
| judicial_action_summon | 31 | 1.84% |
| opening | 30 | 1.78% |
| price_limit_up | 26 | 1.55% |
| Traffic Accident | 26 | 1.55% |
| Fire | 25 | 1.49% |
| sale_acquisition | 24 | 1.43% |
| departure | 22 | 1.31% |
| product_delist | 22 | 1.31% |
| judicial_action_imprisonment | 21 | 1.25% |
| judicial_action_prosecution | 21 | 1.25% |
| Earthquake | 19 | 1.13% |
| layoff | 19 | 1.13% |
| competition_withdrawal | 18 | 1.07% |
| apologize_to | 18 | 1.07% |
| breakup | 18 | 1.07% |
| competition_ban | 17 | 1.01% |
| birthday_celebration | 16 | 0.95% |
| financing | 16 | 0.95% |
| birth | 15 | 0.89% |
| product_award | 15 | 0.89% |
| price_limit_down | 14 | 0.83% |
| Attack | 14 | 0.83% |
| judicial_action_hearing | 13 | 0.77% |
| dismissal | 12 | 0.71% |
| lost_contact | 12 | 0.71% |
| Plane Crash | 12 | 0.71% |
| praise | 11 | 0.65% |
| judicial_action_report | 11 | 0.65% |
| visit_on_set | 10 | 0.59% |
| proposal | 10 | 0.59% |
| retirement | 10 | 0.59% |
| march | 10 | 0.59% |
| dissolution | 10 | 0.59% |
| judicial_action_filing | 9 | 0.54% |
| meet_with | 9 | 0.54% |
| Explosion | 9 | 0.54% |
| price_decrease | 9 | 0.54% |
| Collapse | 9 | 0.54% |
| closing | 9 | 0.54% |
| ipo | 8 | 0.48% |
| thank | 8 | 0.48% |
| suspension | 8 | 0.48% |
| pregnancy | 7 | 0.42% |
| strike | 7 | 0.42% |
| engagement | 7 | 0.42% |
| price_increase | 5 | 0.30% |
| wedding | 5 | 0.30% |
| contract_termination | 5 | 0.30% |
| Flood | 5 | 0.30% |
| infidelity | 4 | 0.24% |
| rate_decrease | 4 | 0.24% |
| rate_increase | 3 | 0.18% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 28,399 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 3,619 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.15 | 11.08% | 1.14 | 9.49% | 0.09% | 0 | 28,399 | 0 | 13,120 |
| val | 1.21 | 13.38% | 1.15 | 9.18% | 0.02% | 0 | 3,619 | 0 | 1,681 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 13,120 | 0.00% | 1,123 | 84.51% | 78.63% | 27.34% | 20.12% | 3.70% |
| val | 1,681 | 0.00% | 236 | 79.24% | 72.46% | 30.93% | 23.73% | 4.17% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 0 | 0 | 1,123 |
| val | 0 | 0 | 236 |

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

| split | mentions the string rule adds | arguments skipped, 2+ clusters | arguments in one multi-mention cluster | other mentions of that cluster | argument spans in 2+ clusters |
| --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | 0 | 0 | 0 |
| val | 0 | 0 | 0 | 0 | 0 |

### Entities

#### train

| label | count | share |
| --- | --- | --- |
| Organization | 10,590 | 22.86% |
| Person | 8,688 | 18.75% |
| Location | 5,026 | 10.85% |
| Date | 4,532 | 9.78% |
| Quantity | 4,338 | 9.36% |
| Product | 3,498 | 7.55% |
| Event | 2,799 | 6.04% |
| Occupation | 1,597 | 3.45% |
| Measurement | 1,257 | 2.71% |
| Nationality | 1,195 | 2.58% |
| Money | 1,186 | 2.56% |
| Facility | 729 | 1.57% |
| Time | 661 | 1.43% |
| Region | 132 | 0.28% |
| Age | 34 | 0.07% |
| Team | 20 | 0.04% |
| Award | 7 | 0.02% |
| Industry | 5 | 0.01% |
| Device | 4 | 0.01% |
| Disease | 3 | 0.01% |
| Animal | 3 | 0.01% |
| Nickname | 3 | 0.01% |
| Crime | 2 | 0.00% |
| Publication | 2 | 0.00% |
| Document | 2 | 0.00% |
| Processor | 2 | 0.00% |
| Company | 2 | 0.00% |
| Franchise | 2 | 0.00% |
| Vehicle | 1 | 0.00% |
| Currency | 1 | 0.00% |
| Injury | 1 | 0.00% |
| Sport | 1 | 0.00% |
| Category | 1 | 0.00% |
| Manufacturer | 1 | 0.00% |
| Acronym | 1 | 0.00% |
| Year | 1 | 0.00% |
| Brand | 1 | 0.00% |
| Media | 1 | 0.00% |
| Regulation | 1 | 0.00% |

#### val

| label | count | share |
| --- | --- | --- |
| Organization | 1,294 | 22.15% |
| Person | 1,142 | 19.55% |
| Location | 607 | 10.39% |
| Date | 579 | 9.91% |
| Quantity | 507 | 8.68% |
| Product | 440 | 7.53% |
| Event | 358 | 6.13% |
| Occupation | 234 | 4.01% |
| Measurement | 182 | 3.12% |
| Nationality | 146 | 2.50% |
| Money | 144 | 2.47% |
| Facility | 97 | 1.66% |
| Time | 81 | 1.39% |
| Region | 21 | 0.36% |
| Processor | 3 | 0.05% |
| Law | 2 | 0.03% |
| Standard | 2 | 0.03% |
| Activity | 1 | 0.02% |
| Feature | 1 | 0.02% |

### Relations

None.

### Classifications

None.

### Structures

None.

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | llm_real |
| relations | unknown |
| events | human |
| classifications | unknown |
| structures | unknown |
