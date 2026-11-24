# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| duee | train | 11,603 | 11,600 | 0 | 0 | 0 | 0 | 13,065 | 28,371 |
| duee | val | 1,453 | 1,453 | 0 | 0 | 0 | 0 | 1,661 | 3,581 |

## duee

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 11,603 | 11,600 | 36 | 110 | 355 | 0 | 0 | 0 | 0 | 13,065 | 28,371 |
| val | 1,453 | 1,453 | 35 | 112 | 316 | 0 | 0 | 0 | 0 | 1,661 | 3,581 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | records | share |
| --- | --- | --- | --- | --- |
| zho | 10,174 | 1,269 | 11,443 | 87.65% |
| und | 1,299 | 164 | 1,463 | 11.21% |
| eng | 26 | 8 | 34 | 0.26% |
| fra | 15 | 2 | 17 | 0.13% |
| por | 9 | 3 | 12 | 0.09% |
| kor | 9 | 0 | 9 | 0.07% |
| wuu | 7 | 2 | 9 | 0.07% |
| rus | 8 | 1 | 9 | 0.07% |
| ita | 7 | 0 | 7 | 0.05% |
| spa | 5 | 0 | 5 | 0.04% |
| deu | 3 | 2 | 5 | 0.04% |
| bre | 3 | 1 | 4 | 0.03% |
| jpn | 3 | 0 | 3 | 0.02% |
| ukr | 3 | 0 | 3 | 0.02% |
| hun | 3 | 0 | 3 | 0.02% |
| epo | 3 | 0 | 3 | 0.02% |
| fin | 3 | 0 | 3 | 0.02% |
| vie | 2 | 0 | 2 | 0.02% |
| nor | 2 | 0 | 2 | 0.02% |
| ces | 2 | 0 | 2 | 0.02% |
| ceb | 2 | 0 | 2 | 0.02% |
| slk | 1 | 0 | 1 | 0.01% |
| eus | 1 | 0 | 1 | 0.01% |
| swe | 1 | 0 | 1 | 0.01% |
| msa | 1 | 0 | 1 | 0.01% |
| war | 1 | 0 | 1 | 0.01% |
| lit | 1 | 0 | 1 | 0.01% |
| cat | 1 | 0 | 1 | 0.01% |
| lav | 1 | 0 | 1 | 0.01% |
| mlg | 1 | 0 | 1 | 0.01% |
| pms | 1 | 0 | 1 | 0.01% |
| ind | 1 | 0 | 1 | 0.01% |
| min | 1 | 0 | 1 | 0.01% |
| eml | 1 | 0 | 1 | 0.01% |
| fry | 1 | 0 | 1 | 0.01% |
| pol | 1 | 0 | 1 | 0.01% |
| tur | 0 | 1 | 1 | 0.01% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| train_records | 11,603 |
| train_unique_inputs | 11,600 |
| val_records | 1,453 |
| val_unique_inputs | 1,453 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| event types (per trigger mention) | train / val | 0.0388 | 13,423 / 1,723 | match_result 15%→16%, advancement 2%→2%, joined 3%→3%, product_publish 9%→9%, judicial_action_arrest 5%→5% |
| event roles (per argument) | train / val | 0.0424 | 28,875 / 3,681 | Publisher 3%→2%, event_name 4%→4%, Advancing Party 1%→1%, Winner 6%→7%, loser 6%→6% |

Thin support in val, event types (per trigger mention): 16 label(s) under 10 gold. rate_increase ×3, rate_decrease ×4, infidelity ×4, contract_termination ×5, Flood ×5, wedding ×5, price_increase ×5, strike ×7, thank ×8, pregnancy ×8, ipo ×8, Explosion ×9, meet_with ×9, closing ×9, judicial_action_filing ×9, engagement ×9.
Thin support in val, event roles (per argument): 43 label(s) under 10 gold. release_by ×1, Rate Increase Magnitude ×1, Affair Partner ×1, Attendee ×1, Strike Participant Count ×1, Protest Organizer ×2, contract_termination_target ×3, rate_cut_magnitude ×3, Rate Raising Institution ×3, apology_target ×4, rate_cutting_institution ×4, price_reducer ×4, Cheating Party ×4, contract_terminator ×5, thanked_person ×5, attack_target ×5, Born Person ×5, Protest Participants ×5, delisted_party ×5, Striker ×5, Price Increased Item ×5, epicenter ×5, price_reduction_magnitude ×6, Birthday Person Age ×6, focal_depth ×6, Liker ×7, Thanking Party ×8, withdrawn_event ×8, Pregnant Person ×8, Case Authority ×8, Court ×8, Proposer ×8, Ban Authority ×8, report_initiator ×8, Listed Company ×8, layoff_count ×9, Missing Person ×9, attacker ×9, meet_object ×9, Visit Target ×9, dismissal_party ×9, Case Subject ×9, follow_investor ×9.

### Events

**train.** 13,423 instances in 11,603 records, 0 triggerless. Trigger gold 13,065. Argument gold 28,371. Arguments per instance, mean 2.15. Trigger length in whitespace tokens, mean 2.05, single-token share 3.0%.

#### train event types

| label | count | share |
| --- | --- | --- |
| match_result | 1,983 | 14.77% |
| product_publish | 1,240 | 9.24% |
| death | 751 | 5.59% |
| judicial_action_arrest | 710 | 5.29% |
| resignation | 598 | 4.46% |
| championship_win | 457 | 3.40% |
| joined | 354 | 2.64% |
| advancement | 322 | 2.40% |
| product_recall | 298 | 2.22% |
| marriage | 296 | 2.21% |
| product_release | 295 | 2.20% |
| divorce | 270 | 2.01% |
| judicial_action_summon | 265 | 1.97% |
| opening | 248 | 1.85% |
| judicial_action_fine | 236 | 1.76% |
| price_limit_up | 220 | 1.64% |
| Traffic Accident | 208 | 1.55% |
| product_delist | 196 | 1.46% |
| departure | 193 | 1.44% |
| sale_acquisition | 191 | 1.42% |
| Fire | 185 | 1.38% |
| judicial_action_prosecution | 170 | 1.27% |
| judicial_action_imprisonment | 168 | 1.25% |
| apologize_to | 150 | 1.12% |
| product_award | 144 | 1.07% |
| competition_withdrawal | 142 | 1.06% |
| layoff | 138 | 1.03% |
| competition_ban | 136 | 1.01% |
| birthday_celebration | 134 | 1.00% |
| financing | 127 | 0.95% |
| Attack | 120 | 0.89% |
| breakup | 119 | 0.89% |
| Earthquake | 110 | 0.82% |
| birth | 103 | 0.77% |
| judicial_action_hearing | 101 | 0.75% |
| retirement | 99 | 0.74% |
| price_limit_down | 96 | 0.72% |
| judicial_action_report | 94 | 0.70% |
| dismissal | 93 | 0.69% |
| lost_contact | 92 | 0.69% |
| praise | 92 | 0.69% |
| suspension | 86 | 0.64% |
| price_decrease | 82 | 0.61% |
| dissolution | 81 | 0.60% |
| judicial_action_filing | 80 | 0.60% |
| Collapse | 79 | 0.59% |
| Plane Crash | 77 | 0.57% |
| proposal | 76 | 0.57% |
| meet_with | 70 | 0.52% |
| visit_on_set | 68 | 0.51% |
| Explosion | 67 | 0.50% |
| strike | 66 | 0.49% |
| price_increase | 64 | 0.48% |
| thank | 62 | 0.46% |
| pregnancy | 62 | 0.46% |
| engagement | 61 | 0.45% |
| closing | 61 | 0.45% |
| wedding | 60 | 0.45% |
| march | 54 | 0.40% |
| ipo | 50 | 0.37% |
| contract_termination | 45 | 0.34% |
| Flood | 40 | 0.30% |
| infidelity | 32 | 0.24% |
| rate_decrease | 31 | 0.23% |
| rate_increase | 25 | 0.19% |

**val.** 1,723 instances in 1,453 records, 0 triggerless. Trigger gold 1,661. Argument gold 3,581. Arguments per instance, mean 2.14. Trigger length in whitespace tokens, mean 2.07, single-token share 3.4%.

#### val event types

| label | count | share |
| --- | --- | --- |
| match_result | 269 | 15.61% |
| product_publish | 153 | 8.88% |
| death | 96 | 5.57% |
| judicial_action_arrest | 85 | 4.93% |
| resignation | 71 | 4.12% |
| championship_win | 64 | 3.71% |
| joined | 52 | 3.02% |
| marriage | 41 | 2.38% |
| divorce | 38 | 2.21% |
| product_recall | 36 | 2.09% |
| product_release | 34 | 1.97% |
| advancement | 34 | 1.97% |
| judicial_action_fine | 33 | 1.92% |
| judicial_action_summon | 32 | 1.86% |
| opening | 30 | 1.74% |
| Fire | 27 | 1.57% |
| price_limit_up | 26 | 1.51% |
| Traffic Accident | 26 | 1.51% |
| sale_acquisition | 24 | 1.39% |
| departure | 22 | 1.28% |
| product_delist | 22 | 1.28% |
| judicial_action_imprisonment | 21 | 1.22% |
| judicial_action_prosecution | 21 | 1.22% |
| layoff | 19 | 1.10% |
| apologize_to | 19 | 1.10% |
| Earthquake | 19 | 1.10% |
| competition_withdrawal | 18 | 1.04% |
| breakup | 18 | 1.04% |
| competition_ban | 18 | 1.04% |
| financing | 16 | 0.93% |
| birthday_celebration | 16 | 0.93% |
| birth | 15 | 0.87% |
| product_award | 15 | 0.87% |
| Attack | 14 | 0.81% |
| judicial_action_hearing | 14 | 0.81% |
| price_limit_down | 14 | 0.81% |
| dismissal | 13 | 0.75% |
| lost_contact | 12 | 0.70% |
| Plane Crash | 12 | 0.70% |
| judicial_action_report | 12 | 0.70% |
| Collapse | 11 | 0.64% |
| retirement | 11 | 0.64% |
| praise | 11 | 0.64% |
| visit_on_set | 11 | 0.64% |
| suspension | 11 | 0.64% |
| price_decrease | 10 | 0.58% |
| march | 10 | 0.58% |
| proposal | 10 | 0.58% |
| dissolution | 10 | 0.58% |
| Explosion | 9 | 0.52% |
| meet_with | 9 | 0.52% |
| closing | 9 | 0.52% |
| judicial_action_filing | 9 | 0.52% |
| engagement | 9 | 0.52% |
| thank | 8 | 0.46% |
| pregnancy | 8 | 0.46% |
| ipo | 8 | 0.46% |
| strike | 7 | 0.41% |
| contract_termination | 5 | 0.29% |
| Flood | 5 | 0.29% |
| wedding | 5 | 0.29% |
| price_increase | 5 | 0.29% |
| rate_decrease | 4 | 0.23% |
| infidelity | 4 | 0.23% |
| rate_increase | 3 | 0.17% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 28,875 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 3,681 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.15 | 10.92% | 1.14 | 9.41% | 0.09% | 0 | 28,875 | 0 | 13,423 |
| val | 1.21 | 13.17% | 1.15 | 9.18% | 0.02% | 0 | 3,681 | 0 | 1,723 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 13,423 | 0.00% | 1,126 | 84.46% | 78.60% | 27.35% | 20.16% | 3.63% |
| val | 1,723 | 0.00% | 237 | 78.90% | 72.15% | 30.80% | 23.63% | 4.06% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 0 | 0 | 1,126 |
| val | 0 | 0 | 237 |

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

| split | mentions the string rule adds | arguments skipped, 2+ clusters | arguments in one multi-mention cluster | other mentions of that cluster | argument spans in 2+ clusters |
| --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | 0 | 0 | 0 |
| val | 0 | 0 | 0 | 0 | 0 |

### Entities

None.

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
