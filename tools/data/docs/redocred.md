# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| redocred | train | 3,052 | 3,052 | 65,112 | 85,818 | 0 | 0 | 0 | 0 |
| redocred | val | 500 | 500 | 10,660 | 17,277 | 0 | 0 | 0 | 0 |
| redocred | test | 500 | 500 | 10,705 | 17,432 | 0 | 0 | 0 | 0 |

## redocred

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3,052 | 3,052 | 182 | 289 | 530 | 65,112 | 85,818 | 0 | 0 | 0 | 0 |
| val | 500 | 500 | 185 | 300 | 487 | 10,660 | 17,277 | 0 | 0 | 0 | 0 |
| test | 500 | 500 | 181 | 289 | 520 | 10,705 | 17,432 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 3,050 | 499 | 500 | 4,049 | 99.93% |
| und | 1 | 1 | 0 | 2 | 0.05% |
| fra | 1 | 0 | 0 | 1 | 0.02% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 500 |
| test_unique_inputs | 500 |
| train_records | 3,052 |
| train_unique_inputs | 3,052 |
| val_records | 500 |
| val_unique_inputs | 500 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0133 | 65,112 / 10,660 | ORG 14%→15%, TIME 18%→18%, LOC 29%→29%, PER 17%→17%, NUM 6%→6% |
| entities (per mention) | train / test | 0.0115 | 65,112 / 10,705 | LOC 29%→30%, TIME 18%→18%, MISC 16%→15%, ORG 14%→14%, PER 17%→17% |
| entities (per mention) | val / test | 0.0190 | 10,660 / 10,705 | ORG 15%→14%, LOC 29%→30%, MISC 16%→15%, PER 17%→17%, NUM 6%→6% |
| relations (per instance) | train / val | 0.0613 | 85,818 / 17,277 | located in the administrative territorial entity 24%→22%, country 17%→16%, country of citizenship 5%→5%, located in or next to body of water 1%→1%, owned by 0%→1% |
| relations (per instance) | train / test | 0.0586 | 85,818 / 17,432 | country of citizenship 5%→4%, country 17%→16%, located in the administrative territorial entity 24%→23%, present in work 1%→1%, performer 2%→2% |
| relations (per instance) | val / test | 0.0576 | 17,277 / 17,432 | located in the administrative territorial entity 22%→23%, contains administrative territorial entity 4%→4%, owned by 1%→0%, series 0%→0%, present in work 1%→1% |

Thin support in val, relations (per instance): 6 label(s) under 10 gold. influenced by ×1, territory claimed by ×2, sister city ×2, separated from ×2, narrative location ×7, position held ×8.
Thin support in test, relations (per instance): 3 label(s) under 10 gold. unemployment rate ×1, separated from ×3, parent taxon ×4.

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
| LOC | 18,757 | 28.81% |
| TIME | 11,817 | 18.15% |
| PER | 11,354 | 17.44% |
| MISC | 10,187 | 15.65% |
| ORG | 9,136 | 14.03% |
| NUM | 3,861 | 5.93% |

#### val

| label | count | share |
| --- | --- | --- |
| LOC | 3,039 | 28.51% |
| TIME | 1,883 | 17.66% |
| PER | 1,827 | 17.14% |
| MISC | 1,669 | 15.66% |
| ORG | 1,636 | 15.35% |
| NUM | 606 | 5.68% |

#### test

| label | count | share |
| --- | --- | --- |
| LOC | 3,207 | 29.96% |
| TIME | 1,879 | 17.55% |
| PER | 1,859 | 17.37% |
| MISC | 1,646 | 15.38% |
| ORG | 1,482 | 13.84% |
| NUM | 632 | 5.90% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| located in the administrative territorial entity | 20,394 | 23.76% |
| country | 14,391 | 16.77% |
| country of citizenship | 4,664 | 5.43% |
| contains administrative territorial entity | 3,369 | 3.93% |
| notable work | 3,044 | 3.55% |
| has part | 2,309 | 2.69% |
| part of | 2,108 | 2.46% |
| performer | 1,762 | 2.05% |
| publication date | 1,621 | 1.89% |
| member of | 1,299 | 1.51% |
| applies to jurisdiction | 1,207 | 1.41% |
| date of birth | 1,172 | 1.37% |
| participant | 1,158 | 1.35% |
| participant of | 1,158 | 1.35% |
| date of death | 1,000 | 1.17% |
| country of origin | 948 | 1.10% |
| record label | 923 | 1.08% |
| cast member | 919 | 1.07% |
| inception | 867 | 1.01% |
| continent | 761 | 0.89% |
| sibling | 712 | 0.83% |
| child | 703 | 0.82% |
| place of birth | 692 | 0.81% |
| present in work | 665 | 0.77% |
| spouse | 640 | 0.75% |
| conflict | 567 | 0.66% |
| followed by | 506 | 0.59% |
| follows | 506 | 0.59% |
| educated at | 503 | 0.59% |
| author | 489 | 0.57% |
| father | 466 | 0.54% |
| platform | 460 | 0.54% |
| located in or next to body of water | 431 | 0.50% |
| employer | 421 | 0.49% |
| creator | 410 | 0.48% |
| member of political party | 403 | 0.47% |
| developer | 402 | 0.47% |
| owned by | 389 | 0.45% |
| member of sports team | 379 | 0.44% |
| head of government | 368 | 0.43% |
| characters | 368 | 0.43% |
| languages spoken, written or signed | 366 | 0.43% |
| religion | 357 | 0.42% |
| league | 356 | 0.41% |
| director | 341 | 0.40% |
| award received | 340 | 0.40% |
| location | 336 | 0.39% |
| legislative body | 305 | 0.36% |
| publisher | 298 | 0.35% |
| located on terrain feature | 293 | 0.34% |
| head of state | 292 | 0.34% |
| place of death | 281 | 0.33% |
| official language | 266 | 0.31% |
| original network | 264 | 0.31% |
| headquarters location | 261 | 0.30% |
| producer | 249 | 0.29% |
| series | 245 | 0.29% |
| genre | 239 | 0.28% |
| screenwriter | 237 | 0.28% |
| subsidiary | 230 | 0.27% |
| parent organization | 230 | 0.27% |
| instance of | 225 | 0.26% |
| start time | 222 | 0.26% |
| work location | 204 | 0.24% |
| founded by | 203 | 0.24% |
| mouth of the watercourse | 200 | 0.23% |
| point in time | 191 | 0.22% |
| operator | 191 | 0.22% |
| military branch | 189 | 0.22% |
| dissolved, abolished or demolished | 181 | 0.21% |
| capital of | 176 | 0.21% |
| capital | 176 | 0.21% |
| basin country | 174 | 0.20% |
| composer | 171 | 0.20% |
| mother | 168 | 0.20% |
| ethnic group | 153 | 0.18% |
| production company | 152 | 0.18% |
| subclass of | 152 | 0.18% |
| chairperson | 145 | 0.17% |
| manufacturer | 144 | 0.17% |
| parent taxon | 117 | 0.14% |
| original language of work | 107 | 0.12% |
| end time | 105 | 0.12% |
| location of formation | 101 | 0.12% |
| replaced by | 96 | 0.11% |
| replaces | 96 | 0.11% |
| narrative location | 83 | 0.10% |
| lyrics by | 79 | 0.09% |
| residence | 66 | 0.08% |
| product or material produced | 65 | 0.08% |
| territory claimed by | 57 | 0.07% |
| position held | 49 | 0.06% |
| influenced by | 22 | 0.03% |
| separated from | 8 | 0.01% |
| sister city | 8 | 0.01% |
| unemployment rate | 2 | 0.00% |

#### val

| label | count | share |
| --- | --- | --- |
| located in the administrative territorial entity | 3,846 | 22.26% |
| country | 2,739 | 15.85% |
| country of citizenship | 823 | 4.76% |
| contains administrative territorial entity | 662 | 3.83% |
| notable work | 603 | 3.49% |
| has part | 469 | 2.71% |
| part of | 444 | 2.57% |
| performer | 329 | 1.90% |
| publication date | 295 | 1.71% |
| participant of | 281 | 1.63% |
| participant | 281 | 1.63% |
| member of | 268 | 1.55% |
| applies to jurisdiction | 247 | 1.43% |
| country of origin | 205 | 1.19% |
| date of birth | 193 | 1.12% |
| cast member | 190 | 1.10% |
| inception | 186 | 1.08% |
| record label | 183 | 1.06% |
| located in or next to body of water | 180 | 1.04% |
| continent | 175 | 1.01% |
| present in work | 160 | 0.93% |
| date of death | 158 | 0.91% |
| sibling | 156 | 0.90% |
| conflict | 149 | 0.86% |
| owned by | 145 | 0.84% |
| child | 129 | 0.75% |
| place of birth | 126 | 0.73% |
| located on terrain feature | 124 | 0.72% |
| religion | 117 | 0.68% |
| spouse | 112 | 0.65% |
| developer | 107 | 0.62% |
| author | 100 | 0.58% |
| followed by | 97 | 0.56% |
| follows | 97 | 0.56% |
| creator | 97 | 0.56% |
| characters | 93 | 0.54% |
| location | 86 | 0.50% |
| member of sports team | 86 | 0.50% |
| educated at | 85 | 0.49% |
| parent organization | 85 | 0.49% |
| subsidiary | 85 | 0.49% |
| father | 85 | 0.49% |
| head of government | 78 | 0.45% |
| composer | 78 | 0.45% |
| employer | 77 | 0.45% |
| manufacturer | 76 | 0.44% |
| point in time | 73 | 0.42% |
| basin country | 72 | 0.42% |
| languages spoken, written or signed | 69 | 0.40% |
| mouth of the watercourse | 66 | 0.38% |
| director | 64 | 0.37% |
| instance of | 63 | 0.36% |
| head of state | 63 | 0.36% |
| award received | 62 | 0.36% |
| start time | 59 | 0.34% |
| legislative body | 59 | 0.34% |
| official language | 57 | 0.33% |
| league | 55 | 0.32% |
| producer | 55 | 0.32% |
| member of political party | 54 | 0.31% |
| work location | 54 | 0.31% |
| operator | 54 | 0.31% |
| military branch | 51 | 0.30% |
| subclass of | 50 | 0.29% |
| ethnic group | 50 | 0.29% |
| platform | 46 | 0.27% |
| headquarters location | 45 | 0.26% |
| original network | 45 | 0.26% |
| place of death | 44 | 0.25% |
| publisher | 43 | 0.25% |
| chairperson | 40 | 0.23% |
| production company | 39 | 0.23% |
| founded by | 37 | 0.21% |
| capital of | 37 | 0.21% |
| capital | 37 | 0.21% |
| screenwriter | 37 | 0.21% |
| dissolved, abolished or demolished | 36 | 0.21% |
| mother | 32 | 0.19% |
| parent taxon | 28 | 0.16% |
| series | 27 | 0.16% |
| end time | 26 | 0.15% |
| location of formation | 24 | 0.14% |
| product or material produced | 22 | 0.13% |
| lyrics by | 19 | 0.11% |
| original language of work | 19 | 0.11% |
| residence | 15 | 0.09% |
| genre | 14 | 0.08% |
| replaced by | 13 | 0.08% |
| replaces | 13 | 0.08% |
| position held | 8 | 0.05% |
| narrative location | 7 | 0.04% |
| territory claimed by | 2 | 0.01% |
| sister city | 2 | 0.01% |
| separated from | 2 | 0.01% |
| influenced by | 1 | 0.01% |

#### test

| label | count | share |
| --- | --- | --- |
| located in the administrative territorial entity | 4,047 | 23.22% |
| country | 2,766 | 15.87% |
| country of citizenship | 781 | 4.48% |
| contains administrative territorial entity | 733 | 4.20% |
| notable work | 597 | 3.42% |
| has part | 522 | 2.99% |
| part of | 453 | 2.60% |
| publication date | 342 | 1.96% |
| member of | 308 | 1.77% |
| performer | 282 | 1.62% |
| participant of | 263 | 1.51% |
| participant | 263 | 1.51% |
| country of origin | 250 | 1.43% |
| applies to jurisdiction | 238 | 1.37% |
| present in work | 217 | 1.24% |
| record label | 209 | 1.20% |
| date of birth | 200 | 1.15% |
| cast member | 198 | 1.14% |
| continent | 190 | 1.09% |
| inception | 180 | 1.03% |
| date of death | 170 | 0.98% |
| conflict | 154 | 0.88% |
| located in or next to body of water | 132 | 0.76% |
| sibling | 118 | 0.68% |
| developer | 116 | 0.67% |
| characters | 111 | 0.64% |
| child | 106 | 0.61% |
| place of birth | 102 | 0.59% |
| author | 102 | 0.59% |
| educated at | 98 | 0.56% |
| award received | 93 | 0.53% |
| producer | 92 | 0.53% |
| platform | 91 | 0.52% |
| located on terrain feature | 87 | 0.50% |
| publisher | 86 | 0.49% |
| series | 85 | 0.49% |
| followed by | 84 | 0.48% |
| follows | 84 | 0.48% |
| owned by | 84 | 0.48% |
| spouse | 80 | 0.46% |
| member of sports team | 80 | 0.46% |
| employer | 79 | 0.45% |
| location | 77 | 0.44% |
| father | 75 | 0.43% |
| point in time | 73 | 0.42% |
| director | 72 | 0.41% |
| languages spoken, written or signed | 71 | 0.41% |
| league | 71 | 0.41% |
| creator | 70 | 0.40% |
| military branch | 68 | 0.39% |
| head of state | 68 | 0.39% |
| religion | 67 | 0.38% |
| head of government | 66 | 0.38% |
| mouth of the watercourse | 63 | 0.36% |
| founded by | 61 | 0.35% |
| legislative body | 60 | 0.34% |
| place of death | 60 | 0.34% |
| composer | 57 | 0.33% |
| instance of | 56 | 0.32% |
| genre | 56 | 0.32% |
| screenwriter | 52 | 0.30% |
| dissolved, abolished or demolished | 51 | 0.29% |
| start time | 47 | 0.27% |
| parent organization | 47 | 0.27% |
| subsidiary | 47 | 0.27% |
| work location | 46 | 0.26% |
| member of political party | 44 | 0.25% |
| production company | 40 | 0.23% |
| headquarters location | 40 | 0.23% |
| operator | 37 | 0.21% |
| original language of work | 36 | 0.21% |
| original network | 36 | 0.21% |
| subclass of | 36 | 0.21% |
| official language | 31 | 0.18% |
| capital of | 31 | 0.18% |
| capital | 31 | 0.18% |
| basin country | 31 | 0.18% |
| manufacturer | 29 | 0.17% |
| end time | 27 | 0.15% |
| mother | 27 | 0.15% |
| ethnic group | 26 | 0.15% |
| chairperson | 24 | 0.14% |
| lyrics by | 21 | 0.12% |
| narrative location | 20 | 0.11% |
| territory claimed by | 19 | 0.11% |
| location of formation | 17 | 0.10% |
| residence | 13 | 0.07% |
| product or material produced | 13 | 0.07% |
| position held | 11 | 0.06% |
| replaced by | 10 | 0.06% |
| replaces | 10 | 0.06% |
| influenced by | 10 | 0.06% |
| parent taxon | 4 | 0.02% |
| separated from | 3 | 0.02% |
| unemployment rate | 1 | 0.01% |

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
