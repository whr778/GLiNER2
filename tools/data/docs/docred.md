# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| docred | train | 83,940 | 83,940 | 1,760,738 | 1,234,317 | 0 | 0 | 0 | 0 |
| docred | val | 10,421 | 10,421 | 217,903 | 153,305 | 0 | 0 | 0 | 0 |
| docred | test | 10,554 | 10,554 | 221,646 | 156,140 | 0 | 0 | 0 | 0 |

## docred

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 83,940 | 83,940 | 184 | 292 | 542 | 1,760,738 | 1,234,317 | 0 | 0 | 0 | 0 |
| val | 10,421 | 10,421 | 185 | 292 | 520 | 217,903 | 153,305 | 0 | 0 | 0 | 0 |
| test | 10,554 | 10,554 | 185 | 294 | 523 | 221,646 | 156,140 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 83,856 | 10,410 | 10,538 | 104,804 | 99.89% |
| und | 81 | 10 | 16 | 107 | 0.10% |
| fra | 2 | 0 | 0 | 2 | 0.00% |
| ces | 1 | 0 | 0 | 1 | 0.00% |
| vie | 0 | 1 | 0 | 1 | 0.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 10,554 |
| test_unique_inputs | 10,554 |
| train_records | 83,940 |
| train_unique_inputs | 83,940 |
| val_records | 10,421 |
| val_unique_inputs | 10,421 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0035 | 1,760,738 / 217,903 | ORG 14%→15%, MISC 15%→15%, TIME 17%→17%, PER 20%→20%, LOC 30%→30% |
| entities (per mention) | train / test | 0.0059 | 1,760,738 / 221,646 | LOC 30%→30%, MISC 15%→16%, ORG 14%→14%, TIME 17%→17%, PER 20%→20% |
| entities (per mention) | val / test | 0.0075 | 217,903 / 221,646 | ORG 15%→14%, MISC 15%→16%, LOC 30%→30%, NUM 4%→3%, TIME 17%→17% |
| relations (per instance) | train / val | 0.0112 | 1,234,317 / 153,305 | country 21%→21%, headquarters location 1%→1%, country of citizenship 8%→8%, creator 0%→0%, contains administrative territorial entity 4%→4% |
| relations (per instance) | train / test | 0.0166 | 1,234,317 / 156,140 | country 21%→21%, located in the administrative territorial entity 10%→10%, sibling 1%→1%, date of birth 2%→2%, publication date 3%→3% |
| relations (per instance) | val / test | 0.0225 | 153,305 / 156,140 | country 21%→21%, date of birth 2%→2%, sibling 1%→1%, located in the administrative territorial entity 10%→10%, country of citizenship 8%→8% |


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
| LOC | 531,110 | 30.16% |
| PER | 346,101 | 19.66% |
| TIME | 301,590 | 17.13% |
| MISC | 269,932 | 15.33% |
| ORG | 250,463 | 14.22% |
| NUM | 61,542 | 3.50% |

#### val

| label | count | share |
| --- | --- | --- |
| LOC | 65,614 | 30.11% |
| PER | 42,641 | 19.57% |
| TIME | 37,094 | 17.02% |
| MISC | 33,168 | 15.22% |
| ORG | 31,675 | 14.54% |
| NUM | 7,711 | 3.54% |

#### test

| label | count | share |
| --- | --- | --- |
| LOC | 67,568 | 30.48% |
| PER | 43,288 | 19.53% |
| TIME | 37,586 | 16.96% |
| MISC | 34,584 | 15.60% |
| ORG | 30,978 | 13.98% |
| NUM | 7,642 | 3.45% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| country | 257,920 | 20.90% |
| located in the administrative territorial entity | 117,360 | 9.51% |
| country of citizenship | 103,191 | 8.36% |
| contains administrative territorial entity | 51,489 | 4.17% |
| publication date | 30,858 | 2.50% |
| country of origin | 29,217 | 2.37% |
| date of birth | 28,150 | 2.28% |
| capital | 27,322 | 2.21% |
| place of birth | 25,325 | 2.05% |
| capital of | 23,926 | 1.94% |
| date of death | 23,367 | 1.89% |
| performer | 23,144 | 1.88% |
| part of | 23,095 | 1.87% |
| inception | 21,754 | 1.76% |
| place of death | 20,083 | 1.63% |
| has part | 18,547 | 1.50% |
| cast member | 17,384 | 1.41% |
| continent | 15,203 | 1.23% |
| headquarters location | 13,790 | 1.12% |
| member of | 12,557 | 1.02% |
| record label | 12,272 | 0.99% |
| member of sports team | 10,283 | 0.83% |
| follows | 9,892 | 0.80% |
| child | 9,805 | 0.79% |
| member of political party | 9,661 | 0.78% |
| followed by | 9,371 | 0.76% |
| sibling | 9,315 | 0.75% |
| sister city | 9,120 | 0.74% |
| applies to jurisdiction | 8,203 | 0.66% |
| spouse | 8,109 | 0.66% |
| director | 8,030 | 0.65% |
| father | 7,505 | 0.61% |
| author | 7,429 | 0.60% |
| educated at | 7,036 | 0.57% |
| conflict | 6,739 | 0.55% |
| screenwriter | 6,452 | 0.52% |
| founded by | 6,274 | 0.51% |
| ethnic group | 6,143 | 0.50% |
| owned by | 6,088 | 0.49% |
| work location | 6,060 | 0.49% |
| present in work | 5,767 | 0.47% |
| head of government | 5,713 | 0.46% |
| dissolved, abolished or demolished | 5,663 | 0.46% |
| employer | 5,563 | 0.45% |
| producer | 5,480 | 0.44% |
| location | 5,452 | 0.44% |
| located in or next to body of water | 5,423 | 0.44% |
| start time | 5,330 | 0.43% |
| official language | 5,317 | 0.43% |
| developer | 5,226 | 0.42% |
| award received | 5,208 | 0.42% |
| languages spoken, written or signed | 5,157 | 0.42% |
| creator | 5,100 | 0.41% |
| end time | 4,962 | 0.40% |
| league | 4,943 | 0.40% |
| platform | 4,914 | 0.40% |
| instance of | 4,543 | 0.37% |
| notable work | 4,323 | 0.35% |
| religion | 4,264 | 0.35% |
| located on terrain feature | 4,154 | 0.34% |
| participant | 4,106 | 0.33% |
| publisher | 3,717 | 0.30% |
| location of formation | 3,647 | 0.30% |
| head of state | 3,560 | 0.29% |
| original network | 3,543 | 0.29% |
| composer | 3,452 | 0.28% |
| series | 3,195 | 0.26% |
| participant of | 3,041 | 0.25% |
| characters | 2,958 | 0.24% |
| parent organization | 2,758 | 0.22% |
| basin country | 2,652 | 0.21% |
| residence | 2,621 | 0.21% |
| legislative body | 2,515 | 0.20% |
| operator | 2,490 | 0.20% |
| point in time | 2,378 | 0.19% |
| mother | 2,343 | 0.19% |
| subclass of | 2,275 | 0.18% |
| replaced by | 2,275 | 0.18% |
| manufacturer | 2,260 | 0.18% |
| military branch | 2,217 | 0.18% |
| narrative location | 2,093 | 0.17% |
| subsidiary | 2,050 | 0.17% |
| mouth of the watercourse | 1,990 | 0.16% |
| lyrics by | 1,982 | 0.16% |
| original language of work | 1,891 | 0.15% |
| chairperson | 1,789 | 0.14% |
| parent taxon | 1,761 | 0.14% |
| production company | 1,753 | 0.14% |
| separated from | 1,717 | 0.14% |
| influenced by | 1,702 | 0.14% |
| genre | 1,624 | 0.13% |
| replaces | 1,467 | 0.12% |
| position held | 1,370 | 0.11% |
| unemployment rate | 1,319 | 0.11% |
| territory claimed by | 1,317 | 0.11% |
| product or material produced | 518 | 0.04% |

#### val

| label | count | share |
| --- | --- | --- |
| country | 31,822 | 20.76% |
| located in the administrative territorial entity | 14,659 | 9.56% |
| country of citizenship | 12,698 | 8.28% |
| contains administrative territorial entity | 6,509 | 4.25% |
| publication date | 3,770 | 2.46% |
| country of origin | 3,612 | 2.36% |
| date of birth | 3,578 | 2.33% |
| capital | 3,361 | 2.19% |
| place of birth | 3,204 | 2.09% |
| capital of | 2,974 | 1.94% |
| date of death | 2,915 | 1.90% |
| performer | 2,847 | 1.86% |
| part of | 2,845 | 1.86% |
| inception | 2,681 | 1.75% |
| place of death | 2,586 | 1.69% |
| has part | 2,281 | 1.49% |
| cast member | 2,144 | 1.40% |
| continent | 1,951 | 1.27% |
| headquarters location | 1,854 | 1.21% |
| member of | 1,637 | 1.07% |
| record label | 1,602 | 1.04% |
| child | 1,277 | 0.83% |
| follows | 1,252 | 0.82% |
| sister city | 1,244 | 0.81% |
| sibling | 1,214 | 0.79% |
| member of political party | 1,176 | 0.77% |
| member of sports team | 1,168 | 0.76% |
| followed by | 1,166 | 0.76% |
| applies to jurisdiction | 1,044 | 0.68% |
| spouse | 992 | 0.65% |
| director | 983 | 0.64% |
| father | 951 | 0.62% |
| educated at | 855 | 0.56% |
| owned by | 849 | 0.55% |
| founded by | 847 | 0.55% |
| conflict | 834 | 0.54% |
| author | 810 | 0.53% |
| screenwriter | 776 | 0.51% |
| work location | 736 | 0.48% |
| employer | 729 | 0.48% |
| dissolved, abolished or demolished | 720 | 0.47% |
| ethnic group | 712 | 0.46% |
| award received | 701 | 0.46% |
| official language | 699 | 0.46% |
| developer | 679 | 0.44% |
| location | 676 | 0.44% |
| present in work | 676 | 0.44% |
| located in or next to body of water | 668 | 0.44% |
| producer | 666 | 0.43% |
| head of government | 665 | 0.43% |
| languages spoken, written or signed | 649 | 0.42% |
| league | 638 | 0.42% |
| start time | 618 | 0.40% |
| end time | 594 | 0.39% |
| platform | 583 | 0.38% |
| creator | 516 | 0.34% |
| notable work | 505 | 0.33% |
| instance of | 504 | 0.33% |
| located on terrain feature | 503 | 0.33% |
| location of formation | 487 | 0.32% |
| religion | 478 | 0.31% |
| publisher | 472 | 0.31% |
| participant | 470 | 0.31% |
| head of state | 434 | 0.28% |
| composer | 421 | 0.27% |
| series | 402 | 0.26% |
| original network | 398 | 0.26% |
| participant of | 345 | 0.23% |
| basin country | 341 | 0.22% |
| legislative body | 338 | 0.22% |
| characters | 338 | 0.22% |
| parent organization | 333 | 0.22% |
| residence | 332 | 0.22% |
| mother | 322 | 0.21% |
| operator | 321 | 0.21% |
| subclass of | 285 | 0.19% |
| point in time | 282 | 0.18% |
| manufacturer | 278 | 0.18% |
| military branch | 276 | 0.18% |
| narrative location | 264 | 0.17% |
| replaced by | 257 | 0.17% |
| subsidiary | 255 | 0.17% |
| parent taxon | 243 | 0.16% |
| mouth of the watercourse | 240 | 0.16% |
| chairperson | 238 | 0.16% |
| production company | 232 | 0.15% |
| separated from | 232 | 0.15% |
| lyrics by | 224 | 0.15% |
| influenced by | 211 | 0.14% |
| genre | 208 | 0.14% |
| original language of work | 196 | 0.13% |
| territory claimed by | 194 | 0.13% |
| replaces | 176 | 0.11% |
| position held | 166 | 0.11% |
| unemployment rate | 149 | 0.10% |
| product or material produced | 62 | 0.04% |

#### test

| label | count | share |
| --- | --- | --- |
| country | 33,129 | 21.22% |
| located in the administrative territorial entity | 15,178 | 9.72% |
| country of citizenship | 13,159 | 8.43% |
| contains administrative territorial entity | 6,652 | 4.26% |
| publication date | 4,052 | 2.60% |
| country of origin | 3,739 | 2.39% |
| capital | 3,447 | 2.21% |
| date of birth | 3,314 | 2.12% |
| place of birth | 3,214 | 2.06% |
| performer | 3,002 | 1.92% |
| capital of | 2,992 | 1.92% |
| part of | 2,900 | 1.86% |
| date of death | 2,837 | 1.82% |
| inception | 2,734 | 1.75% |
| place of death | 2,471 | 1.58% |
| has part | 2,398 | 1.54% |
| cast member | 2,232 | 1.43% |
| continent | 1,994 | 1.28% |
| headquarters location | 1,706 | 1.09% |
| record label | 1,513 | 0.97% |
| member of | 1,491 | 0.95% |
| follows | 1,279 | 0.82% |
| member of sports team | 1,240 | 0.79% |
| followed by | 1,229 | 0.79% |
| member of political party | 1,151 | 0.74% |
| sister city | 1,111 | 0.71% |
| child | 1,109 | 0.71% |
| director | 1,098 | 0.70% |
| applies to jurisdiction | 995 | 0.64% |
| author | 937 | 0.60% |
| sibling | 929 | 0.59% |
| spouse | 925 | 0.59% |
| father | 882 | 0.56% |
| screenwriter | 880 | 0.56% |
| educated at | 838 | 0.54% |
| owned by | 825 | 0.53% |
| ethnic group | 785 | 0.50% |
| work location | 777 | 0.50% |
| conflict | 758 | 0.49% |
| dissolved, abolished or demolished | 750 | 0.48% |
| producer | 712 | 0.46% |
| start time | 711 | 0.46% |
| developer | 701 | 0.45% |
| location | 698 | 0.45% |
| head of government | 691 | 0.44% |
| located in or next to body of water | 688 | 0.44% |
| employer | 679 | 0.43% |
| founded by | 677 | 0.43% |
| languages spoken, written or signed | 662 | 0.42% |
| official language | 660 | 0.42% |
| creator | 651 | 0.42% |
| end time | 639 | 0.41% |
| platform | 632 | 0.40% |
| league | 628 | 0.40% |
| present in work | 618 | 0.40% |
| instance of | 615 | 0.39% |
| participant | 600 | 0.38% |
| notable work | 597 | 0.38% |
| award received | 586 | 0.38% |
| religion | 545 | 0.35% |
| located on terrain feature | 543 | 0.35% |
| location of formation | 459 | 0.29% |
| composer | 455 | 0.29% |
| original network | 448 | 0.29% |
| publisher | 427 | 0.27% |
| participant of | 411 | 0.26% |
| head of state | 403 | 0.26% |
| basin country | 391 | 0.25% |
| point in time | 356 | 0.23% |
| series | 347 | 0.22% |
| mouth of the watercourse | 340 | 0.22% |
| parent organization | 336 | 0.22% |
| characters | 314 | 0.20% |
| legislative body | 302 | 0.19% |
| operator | 295 | 0.19% |
| manufacturer | 282 | 0.18% |
| residence | 279 | 0.18% |
| replaced by | 275 | 0.18% |
| narrative location | 264 | 0.17% |
| separated from | 262 | 0.17% |
| original language of work | 253 | 0.16% |
| chairperson | 252 | 0.16% |
| subclass of | 250 | 0.16% |
| military branch | 248 | 0.16% |
| production company | 248 | 0.16% |
| lyrics by | 245 | 0.16% |
| parent taxon | 238 | 0.15% |
| mother | 235 | 0.15% |
| genre | 227 | 0.15% |
| subsidiary | 223 | 0.14% |
| replaces | 186 | 0.12% |
| position held | 179 | 0.11% |
| influenced by | 167 | 0.11% |
| unemployment rate | 156 | 0.10% |
| territory claimed by | 122 | 0.08% |
| product or material produced | 80 | 0.05% |

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
