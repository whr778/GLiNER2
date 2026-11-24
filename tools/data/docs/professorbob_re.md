# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| professorbob_re | train | 13,926 | 13,926 | 0 | 31,358 | 0 | 0 | 0 | 0 |
| professorbob_re | val | 1,737 | 1,737 | 0 | 3,818 | 0 | 0 | 0 | 0 |
| professorbob_re | test | 1,728 | 1,728 | 0 | 3,846 | 0 | 0 | 0 | 0 |

## professorbob_re

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 13,926 | 13,926 | 248 | 753 | 4243 | 0 | 31,358 | 0 | 0 | 0 | 0 |
| val | 1,737 | 1,737 | 255 | 771 | 3840 | 0 | 3,818 | 0 | 0 | 0 | 0 |
| test | 1,728 | 1,728 | 244 | 742 | 3346 | 0 | 3,846 | 0 | 0 | 0 | 0 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 13,773 | 1,722 | 1,708 | 17,203 | 98.92% |
| und | 142 | 14 | 18 | 174 | 1.00% |
| ces | 3 | 0 | 1 | 4 | 0.02% |
| fra | 2 | 0 | 0 | 2 | 0.01% |
| bel | 2 | 0 | 0 | 2 | 0.01% |
| epo | 1 | 0 | 0 | 1 | 0.01% |
| spa | 1 | 0 | 0 | 1 | 0.01% |
| lat | 1 | 0 | 0 | 1 | 0.01% |
| pms | 1 | 0 | 0 | 1 | 0.01% |
| ben | 0 | 1 | 0 | 1 | 0.01% |
| tam | 0 | 0 | 1 | 1 | 0.01% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 1,728 |
| test_unique_inputs | 1,728 |
| train_records | 13,926 |
| train_unique_inputs | 13,926 |
| val_records | 1,737 |
| val_unique_inputs | 1,737 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| relations (per instance) | train / val | 0.0633 | 31,358 / 3,818 | given name 2%→3%, occupation 3%→2%, no relation 37%→37%, award received 0%→1%, notable work 0%→1% |
| relations (per instance) | train / test | 0.0626 | 31,358 / 3,846 | no relation 37%→38%, shares border with 3%→2%, occupation 3%→2%, educated at 1%→0%, headquarters location 1%→0% |
| relations (per instance) | val / test | 0.0808 | 3,818 / 3,846 | no relation 37%→38%, award received 1%→0%, given name 3%→2%, shares border with 3%→2%, educated at 1%→0% |

Unseen in train, present in val, relations (per instance): immediate cause of ×1.
Thin support in val, relations (per instance): 96 label(s) under 10 gold. foundational text ×1, winner ×1, writing system ×1, place of publication ×1, distribution format ×1, sponsor ×1, place of detention ×1, movement ×1, contributor to the creative work or subject ×1, affiliation ×1, crosses ×1, has immediate cause ×1, described by source ×1, film editor ×1, home venue ×1, editor ×1, legal form ×1, has facility ×1, has contributing factor ×1, cause of destruction ×1, immediate cause of ×1, territory claimed by ×1, sex or gender ×2, instrument ×2, collection ×2, transport network ×2, time period ×2, architectural style ×2, location of creation ×2, indigenous to ×2, occupant ×2, military branch ×3, publisher ×3, has quality ×3, manner of death ×3, manufacturer ×3, office held by head of government ×3, cause of death ×3, head coach ×3, color ×3, located on street ×3, ethnic group ×3, discoverer or inventor ×3, allegiance ×3, chairperson ×3, lyrics by ×3, organizer ×4, depicts ×4, military rank ×4, nominated for ×4, sports discipline competed in ×4, connecting service ×4, country for sport ×4, candidate ×4, start point ×4, has goal ×4, family ×5, made from material ×5, presenter ×5, located in protected area ×5, writing language ×6, subject has role ×6, production company ×6, religion or worldview ×6, developer ×6, relative ×6, member of political party ×6, location of formation ×6, industry ×6, has subsidiary ×6, architect ×6, capital of ×6, legislated by ×6, separated from ×6, replaced by ×7, residence ×7, producer ×7, composer ×7, has works in the collection ×7, parent organization ×7, place of burial ×8, author ×8, form of creative work ×8, product or material produced ×8, student of ×8, influenced by ×8, member of sports team ×9, employer ×9, noble title ×9, follows ×9, operator ×9, head of government ×9, connects with ×9, student ×9, uses ×9, has part(s) of the class ×9.
Thin support in test, relations (per instance): 99 label(s) under 10 gold. exhibition history ×1, collection ×1, sponsor ×1, ethnic group ×1, writing system ×1, maintained by ×1, distribution format ×1, day of week ×1, film editor ×1, located on street ×1, published in ×1, head coach ×1, contributor to the creative work or subject ×1, contains settlement ×1, religious order ×1, start point ×1, separated from ×1, office held by head of government ×2, dedicated to ×2, crosses ×2, sex or gender ×2, nominated for ×2, place of publication ×2, ancestral home ×2, military rank ×2, manufacturer ×2, architectural style ×2, organizer ×2, country of registry ×2, has goal ×2, owner of ×2, lyrics by ×2, topic's main category ×2, color ×3, made from material ×3, foundational text ×3, allegiance ×3, manner of death ×3, winner ×3, subject has role ×3, heritage designation ×3, participating team ×3, place of detention ×3, candidate ×3, legislated by ×3, chairperson ×3, used by ×3, time period ×4, sports discipline competed in ×4, has quality ×4, academic degree ×4, member of political party ×4, occupant ×4, replaced by ×4, home venue ×4, transport network ×4, instrument ×4, industry ×4, student of ×4, production company ×4, has part(s) of the class ×4, territory claimed by ×4, family ×5, discoverer or inventor ×5, has works in the collection ×5, cause of death ×5, operator ×5, presenter ×5, has contributing factor ×5, located on terrain feature ×5, mountain range ×6, member of sports team ×6, characters ×6, publisher ×6, relative ×6, place of burial ×6, product or material produced ×6, influenced by ×6, residence ×7, architect ×7, military branch ×7, capital of ×7, religion or worldview ×7, located in protected area ×7, cause ×7, depicts ×8, location of formation ×8, field of work ×8, student ×8, has subsidiary ×8, noble title ×9, present in work ×9, author ×9, writing language ×9, country for sport ×9, developer ×9, said to be the same as ×9, follows ×9, composer ×9.

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

#### train

| label | count | share |
| --- | --- | --- |
| no relation | 11,630 | 37.09% |
| instance of | 1,465 | 4.67% |
| occupation | 849 | 2.71% |
| family name | 800 | 2.55% |
| shares border with | 787 | 2.51% |
| given name | 753 | 2.40% |
| country | 724 | 2.31% |
| place of birth | 534 | 1.70% |
| located in the administrative territorial entity | 452 | 1.44% |
| has part(s) | 411 | 1.31% |
| country of citizenship | 403 | 1.29% |
| founded by | 390 | 1.24% |
| contains the administrative territorial entity | 362 | 1.15% |
| member of | 336 | 1.07% |
| participant | 256 | 0.82% |
| genre | 245 | 0.78% |
| followed by | 232 | 0.74% |
| languages spoken, written or signed | 207 | 0.66% |
| educated at | 207 | 0.66% |
| replaces | 201 | 0.64% |
| different from | 200 | 0.64% |
| performer | 195 | 0.62% |
| father | 188 | 0.60% |
| part of | 187 | 0.60% |
| named after | 186 | 0.59% |
| headquarters location | 186 | 0.59% |
| place of death | 180 | 0.57% |
| language used | 175 | 0.56% |
| capital | 169 | 0.54% |
| based on | 169 | 0.54% |
| sibling | 168 | 0.54% |
| owned by | 162 | 0.52% |
| location | 158 | 0.50% |
| significant event | 152 | 0.48% |
| award received | 152 | 0.48% |
| child | 150 | 0.48% |
| director | 149 | 0.48% |
| notable work | 142 | 0.45% |
| platform | 142 | 0.45% |
| spouse | 140 | 0.45% |
| participant in | 132 | 0.42% |
| facet of | 131 | 0.42% |
| has use | 129 | 0.41% |
| participant of | 127 | 0.41% |
| country of origin | 126 | 0.40% |
| sport | 123 | 0.39% |
| uses | 123 | 0.39% |
| present in work | 119 | 0.38% |
| significant person | 118 | 0.38% |
| applies to jurisdiction | 117 | 0.37% |
| head of government | 112 | 0.36% |
| position held | 112 | 0.36% |
| field of work | 110 | 0.35% |
| narrative location | 109 | 0.35% |
| located on terrain feature | 104 | 0.33% |
| developer | 103 | 0.33% |
| continent | 100 | 0.32% |
| mother | 99 | 0.32% |
| native language | 97 | 0.31% |
| opposite of | 97 | 0.31% |
| conflict | 95 | 0.30% |
| connects with | 95 | 0.30% |
| said to be the same as | 93 | 0.30% |
| main subject | 88 | 0.28% |
| partially coincident with | 88 | 0.28% |
| subclass of | 87 | 0.28% |
| has effect | 85 | 0.27% |
| work location | 80 | 0.26% |
| language of work or name | 77 | 0.25% |
| capital of | 76 | 0.24% |
| twinned administrative body | 75 | 0.24% |
| noble title | 74 | 0.24% |
| located in/on physical feature | 73 | 0.23% |
| characters | 73 | 0.23% |
| writing language | 72 | 0.23% |
| creator | 71 | 0.23% |
| residence | 70 | 0.22% |
| influenced by | 70 | 0.22% |
| employer | 66 | 0.21% |
| follows | 66 | 0.21% |
| producer | 66 | 0.21% |
| composer | 65 | 0.21% |
| parent organization | 64 | 0.20% |
| form of creative work | 62 | 0.20% |
| author | 60 | 0.19% |
| location of formation | 59 | 0.19% |
| operator | 59 | 0.19% |
| architect | 57 | 0.18% |
| place of burial | 55 | 0.18% |
| made from material | 55 | 0.18% |
| member of political party | 53 | 0.17% |
| has part(s) of the class | 53 | 0.17% |
| student | 52 | 0.17% |
| student of | 52 | 0.17% |
| religion or worldview | 51 | 0.16% |
| has works in the collection | 50 | 0.16% |
| has goal | 50 | 0.16% |
| depicts | 49 | 0.16% |
| instrument | 48 | 0.15% |
| industry | 48 | 0.15% |
| replaced by | 46 | 0.15% |
| cause | 45 | 0.14% |
| publisher | 44 | 0.14% |
| production company | 44 | 0.14% |
| member of sports team | 43 | 0.14% |
| country for sport | 43 | 0.14% |
| relative | 43 | 0.14% |
| product or material produced | 43 | 0.14% |
| located in protected area | 43 | 0.14% |
| manufacturer | 42 | 0.13% |
| discoverer or inventor | 41 | 0.13% |
| military branch | 41 | 0.13% |
| home venue | 41 | 0.13% |
| owner of | 41 | 0.13% |
| presenter | 40 | 0.13% |
| sports discipline competed in | 39 | 0.12% |
| has subsidiary | 38 | 0.12% |
| candidate | 35 | 0.11% |
| has quality | 34 | 0.11% |
| separated from | 34 | 0.11% |
| time period | 31 | 0.10% |
| chairperson | 31 | 0.10% |
| military rank | 31 | 0.10% |
| has contributing factor | 30 | 0.10% |
| allegiance | 29 | 0.09% |
| family | 28 | 0.09% |
| organizer | 28 | 0.09% |
| cause of death | 27 | 0.09% |
| subject has role | 27 | 0.09% |
| office held by head of government | 26 | 0.08% |
| nominated for | 24 | 0.08% |
| legislated by | 23 | 0.07% |
| contains settlement | 23 | 0.07% |
| sex or gender | 22 | 0.07% |
| territory claimed by | 22 | 0.07% |
| color | 21 | 0.07% |
| winner | 21 | 0.07% |
| legal form | 20 | 0.06% |
| used by | 17 | 0.05% |
| collection | 16 | 0.05% |
| contributor to the creative work or subject | 16 | 0.05% |
| head coach | 16 | 0.05% |
| lyrics by | 16 | 0.05% |
| manner of death | 15 | 0.05% |
| movement | 15 | 0.05% |
| crosses | 15 | 0.05% |
| distribution format | 15 | 0.05% |
| located on street | 15 | 0.05% |
| mountain range | 14 | 0.04% |
| place of publication | 14 | 0.04% |
| start point | 14 | 0.04% |
| foundational text | 14 | 0.04% |
| country of registry | 14 | 0.04% |
| indigenous to | 13 | 0.04% |
| participating team | 13 | 0.04% |
| ancestral home | 12 | 0.04% |
| occupant | 12 | 0.04% |
| film editor | 12 | 0.04% |
| heritage designation | 11 | 0.04% |
| has immediate cause | 11 | 0.04% |
| connecting service | 11 | 0.04% |
| religious order | 11 | 0.04% |
| architectural style | 11 | 0.04% |
| sponsor | 10 | 0.03% |
| victory | 10 | 0.03% |
| archives at | 8 | 0.03% |
| editor | 8 | 0.03% |
| affiliation | 8 | 0.03% |
| ethnic group | 8 | 0.03% |
| commemorates | 8 | 0.03% |
| location of creation | 7 | 0.02% |
| academic degree | 7 | 0.02% |
| maintained by | 7 | 0.02% |
| transport network | 7 | 0.02% |
| published in | 6 | 0.02% |
| distributed by | 6 | 0.02% |
| place of detention | 6 | 0.02% |
| dedicated to | 6 | 0.02% |
| described by source | 6 | 0.02% |
| honorific prefix | 5 | 0.02% |
| exhibition history | 4 | 0.01% |
| writing system | 4 | 0.01% |
| has facility | 4 | 0.01% |
| addressee | 4 | 0.01% |
| topic's main category | 4 | 0.01% |
| repealed by | 3 | 0.01% |
| significant place | 3 | 0.01% |
| contributing factor of | 3 | 0.01% |
| candidacy in election | 2 | 0.01% |
| state of conservation | 2 | 0.01% |
| day of week | 2 | 0.01% |
| fabrication method | 2 | 0.01% |
| does not have cause | 2 | 0.01% |
| cause of destruction | 2 | 0.01% |
| referee | 1 | 0.00% |
| artist files at | 1 | 0.00% |
| officeholder | 1 | 0.00% |

#### val

| label | count | share |
| --- | --- | --- |
| no relation | 1,429 | 37.43% |
| instance of | 170 | 4.45% |
| given name | 106 | 2.78% |
| shares border with | 96 | 2.51% |
| country | 94 | 2.46% |
| occupation | 90 | 2.36% |
| family name | 88 | 2.30% |
| place of birth | 57 | 1.49% |
| has part(s) | 56 | 1.47% |
| country of citizenship | 54 | 1.41% |
| founded by | 54 | 1.41% |
| located in the administrative territorial entity | 50 | 1.31% |
| genre | 40 | 1.05% |
| participant | 36 | 0.94% |
| contains the administrative territorial entity | 34 | 0.89% |
| member of | 34 | 0.89% |
| award received | 31 | 0.81% |
| followed by | 31 | 0.81% |
| notable work | 29 | 0.76% |
| based on | 28 | 0.73% |
| part of | 27 | 0.71% |
| educated at | 27 | 0.71% |
| place of death | 26 | 0.68% |
| replaces | 26 | 0.68% |
| father | 24 | 0.63% |
| named after | 24 | 0.63% |
| significant event | 22 | 0.58% |
| performer | 22 | 0.58% |
| participant of | 22 | 0.58% |
| different from | 20 | 0.52% |
| continent | 19 | 0.50% |
| languages spoken, written or signed | 19 | 0.50% |
| language of work or name | 19 | 0.50% |
| language used | 19 | 0.50% |
| owned by | 19 | 0.50% |
| headquarters location | 19 | 0.50% |
| location | 18 | 0.47% |
| director | 18 | 0.47% |
| conflict | 18 | 0.47% |
| significant person | 18 | 0.47% |
| present in work | 17 | 0.45% |
| country of origin | 17 | 0.45% |
| child | 16 | 0.42% |
| position held | 16 | 0.42% |
| narrative location | 16 | 0.42% |
| capital | 15 | 0.39% |
| native language | 14 | 0.37% |
| main subject | 14 | 0.37% |
| participant in | 14 | 0.37% |
| has use | 14 | 0.37% |
| mother | 13 | 0.34% |
| spouse | 13 | 0.34% |
| subclass of | 13 | 0.34% |
| creator | 13 | 0.34% |
| sport | 13 | 0.34% |
| facet of | 13 | 0.34% |
| applies to jurisdiction | 13 | 0.34% |
| sibling | 12 | 0.31% |
| work location | 12 | 0.31% |
| partially coincident with | 12 | 0.31% |
| has effect | 12 | 0.31% |
| located on terrain feature | 12 | 0.31% |
| located in/on physical feature | 11 | 0.29% |
| opposite of | 11 | 0.29% |
| platform | 10 | 0.26% |
| characters | 10 | 0.26% |
| field of work | 10 | 0.26% |
| said to be the same as | 10 | 0.26% |
| cause | 10 | 0.26% |
| member of sports team | 9 | 0.24% |
| employer | 9 | 0.24% |
| noble title | 9 | 0.24% |
| follows | 9 | 0.24% |
| operator | 9 | 0.24% |
| head of government | 9 | 0.24% |
| connects with | 9 | 0.24% |
| student | 9 | 0.24% |
| uses | 9 | 0.24% |
| has part(s) of the class | 9 | 0.24% |
| place of burial | 8 | 0.21% |
| author | 8 | 0.21% |
| form of creative work | 8 | 0.21% |
| product or material produced | 8 | 0.21% |
| student of | 8 | 0.21% |
| influenced by | 8 | 0.21% |
| replaced by | 7 | 0.18% |
| residence | 7 | 0.18% |
| producer | 7 | 0.18% |
| composer | 7 | 0.18% |
| has works in the collection | 7 | 0.18% |
| parent organization | 7 | 0.18% |
| writing language | 6 | 0.16% |
| subject has role | 6 | 0.16% |
| production company | 6 | 0.16% |
| religion or worldview | 6 | 0.16% |
| developer | 6 | 0.16% |
| relative | 6 | 0.16% |
| member of political party | 6 | 0.16% |
| location of formation | 6 | 0.16% |
| industry | 6 | 0.16% |
| has subsidiary | 6 | 0.16% |
| architect | 6 | 0.16% |
| capital of | 6 | 0.16% |
| legislated by | 6 | 0.16% |
| separated from | 6 | 0.16% |
| family | 5 | 0.13% |
| made from material | 5 | 0.13% |
| presenter | 5 | 0.13% |
| located in protected area | 5 | 0.13% |
| organizer | 4 | 0.10% |
| depicts | 4 | 0.10% |
| military rank | 4 | 0.10% |
| nominated for | 4 | 0.10% |
| sports discipline competed in | 4 | 0.10% |
| connecting service | 4 | 0.10% |
| country for sport | 4 | 0.10% |
| candidate | 4 | 0.10% |
| start point | 4 | 0.10% |
| has goal | 4 | 0.10% |
| military branch | 3 | 0.08% |
| publisher | 3 | 0.08% |
| has quality | 3 | 0.08% |
| manner of death | 3 | 0.08% |
| manufacturer | 3 | 0.08% |
| office held by head of government | 3 | 0.08% |
| cause of death | 3 | 0.08% |
| head coach | 3 | 0.08% |
| color | 3 | 0.08% |
| located on street | 3 | 0.08% |
| ethnic group | 3 | 0.08% |
| discoverer or inventor | 3 | 0.08% |
| allegiance | 3 | 0.08% |
| chairperson | 3 | 0.08% |
| lyrics by | 3 | 0.08% |
| sex or gender | 2 | 0.05% |
| instrument | 2 | 0.05% |
| collection | 2 | 0.05% |
| transport network | 2 | 0.05% |
| time period | 2 | 0.05% |
| architectural style | 2 | 0.05% |
| location of creation | 2 | 0.05% |
| indigenous to | 2 | 0.05% |
| occupant | 2 | 0.05% |
| foundational text | 1 | 0.03% |
| winner | 1 | 0.03% |
| writing system | 1 | 0.03% |
| place of publication | 1 | 0.03% |
| distribution format | 1 | 0.03% |
| sponsor | 1 | 0.03% |
| place of detention | 1 | 0.03% |
| movement | 1 | 0.03% |
| contributor to the creative work or subject | 1 | 0.03% |
| affiliation | 1 | 0.03% |
| crosses | 1 | 0.03% |
| has immediate cause | 1 | 0.03% |
| described by source | 1 | 0.03% |
| film editor | 1 | 0.03% |
| home venue | 1 | 0.03% |
| editor | 1 | 0.03% |
| legal form | 1 | 0.03% |
| has facility | 1 | 0.03% |
| has contributing factor | 1 | 0.03% |
| cause of destruction | 1 | 0.03% |
| immediate cause of | 1 | 0.03% |
| territory claimed by | 1 | 0.03% |

#### test

| label | count | share |
| --- | --- | --- |
| no relation | 1,471 | 38.25% |
| instance of | 178 | 4.63% |
| country | 99 | 2.57% |
| family name | 92 | 2.39% |
| given name | 88 | 2.29% |
| occupation | 87 | 2.26% |
| shares border with | 79 | 2.05% |
| place of birth | 64 | 1.66% |
| located in the administrative territorial entity | 61 | 1.59% |
| country of citizenship | 56 | 1.46% |
| has part(s) | 53 | 1.38% |
| founded by | 49 | 1.27% |
| member of | 43 | 1.12% |
| contains the administrative territorial entity | 40 | 1.04% |
| participant | 38 | 0.99% |
| languages spoken, written or signed | 31 | 0.81% |
| different from | 30 | 0.78% |
| genre | 29 | 0.75% |
| performer | 28 | 0.73% |
| language used | 28 | 0.73% |
| named after | 27 | 0.70% |
| followed by | 27 | 0.70% |
| child | 26 | 0.68% |
| place of death | 26 | 0.68% |
| part of | 26 | 0.68% |
| based on | 26 | 0.68% |
| spouse | 24 | 0.62% |
| capital | 21 | 0.55% |
| sibling | 21 | 0.55% |
| owned by | 21 | 0.55% |
| significant event | 20 | 0.52% |
| location | 19 | 0.49% |
| language of work or name | 19 | 0.49% |
| notable work | 18 | 0.47% |
| father | 17 | 0.44% |
| uses | 16 | 0.42% |
| has effect | 16 | 0.42% |
| significant person | 16 | 0.42% |
| native language | 15 | 0.39% |
| participant in | 15 | 0.39% |
| replaces | 15 | 0.39% |
| platform | 15 | 0.39% |
| narrative location | 15 | 0.39% |
| participant of | 15 | 0.39% |
| head of government | 14 | 0.36% |
| sport | 14 | 0.36% |
| applies to jurisdiction | 14 | 0.36% |
| located in/on physical feature | 13 | 0.34% |
| position held | 13 | 0.34% |
| educated at | 13 | 0.34% |
| continent | 13 | 0.34% |
| facet of | 13 | 0.34% |
| connects with | 12 | 0.31% |
| form of creative work | 12 | 0.31% |
| main subject | 12 | 0.31% |
| award received | 12 | 0.31% |
| has use | 12 | 0.31% |
| headquarters location | 12 | 0.31% |
| director | 12 | 0.31% |
| twinned administrative body | 11 | 0.29% |
| conflict | 11 | 0.29% |
| creator | 11 | 0.29% |
| country of origin | 11 | 0.29% |
| producer | 11 | 0.29% |
| parent organization | 11 | 0.29% |
| mother | 10 | 0.26% |
| work location | 10 | 0.26% |
| employer | 10 | 0.26% |
| subclass of | 10 | 0.26% |
| opposite of | 10 | 0.26% |
| partially coincident with | 10 | 0.26% |
| noble title | 9 | 0.23% |
| present in work | 9 | 0.23% |
| author | 9 | 0.23% |
| writing language | 9 | 0.23% |
| country for sport | 9 | 0.23% |
| developer | 9 | 0.23% |
| said to be the same as | 9 | 0.23% |
| follows | 9 | 0.23% |
| composer | 9 | 0.23% |
| depicts | 8 | 0.21% |
| location of formation | 8 | 0.21% |
| field of work | 8 | 0.21% |
| student | 8 | 0.21% |
| has subsidiary | 8 | 0.21% |
| residence | 7 | 0.18% |
| architect | 7 | 0.18% |
| military branch | 7 | 0.18% |
| capital of | 7 | 0.18% |
| religion or worldview | 7 | 0.18% |
| located in protected area | 7 | 0.18% |
| cause | 7 | 0.18% |
| mountain range | 6 | 0.16% |
| member of sports team | 6 | 0.16% |
| characters | 6 | 0.16% |
| publisher | 6 | 0.16% |
| relative | 6 | 0.16% |
| place of burial | 6 | 0.16% |
| product or material produced | 6 | 0.16% |
| influenced by | 6 | 0.16% |
| family | 5 | 0.13% |
| discoverer or inventor | 5 | 0.13% |
| has works in the collection | 5 | 0.13% |
| cause of death | 5 | 0.13% |
| operator | 5 | 0.13% |
| presenter | 5 | 0.13% |
| has contributing factor | 5 | 0.13% |
| located on terrain feature | 5 | 0.13% |
| time period | 4 | 0.10% |
| sports discipline competed in | 4 | 0.10% |
| has quality | 4 | 0.10% |
| academic degree | 4 | 0.10% |
| member of political party | 4 | 0.10% |
| occupant | 4 | 0.10% |
| replaced by | 4 | 0.10% |
| home venue | 4 | 0.10% |
| transport network | 4 | 0.10% |
| instrument | 4 | 0.10% |
| industry | 4 | 0.10% |
| student of | 4 | 0.10% |
| production company | 4 | 0.10% |
| has part(s) of the class | 4 | 0.10% |
| territory claimed by | 4 | 0.10% |
| color | 3 | 0.08% |
| made from material | 3 | 0.08% |
| foundational text | 3 | 0.08% |
| allegiance | 3 | 0.08% |
| manner of death | 3 | 0.08% |
| winner | 3 | 0.08% |
| subject has role | 3 | 0.08% |
| heritage designation | 3 | 0.08% |
| participating team | 3 | 0.08% |
| place of detention | 3 | 0.08% |
| candidate | 3 | 0.08% |
| legislated by | 3 | 0.08% |
| chairperson | 3 | 0.08% |
| used by | 3 | 0.08% |
| office held by head of government | 2 | 0.05% |
| dedicated to | 2 | 0.05% |
| crosses | 2 | 0.05% |
| sex or gender | 2 | 0.05% |
| nominated for | 2 | 0.05% |
| place of publication | 2 | 0.05% |
| ancestral home | 2 | 0.05% |
| military rank | 2 | 0.05% |
| manufacturer | 2 | 0.05% |
| architectural style | 2 | 0.05% |
| organizer | 2 | 0.05% |
| country of registry | 2 | 0.05% |
| has goal | 2 | 0.05% |
| owner of | 2 | 0.05% |
| lyrics by | 2 | 0.05% |
| topic's main category | 2 | 0.05% |
| exhibition history | 1 | 0.03% |
| collection | 1 | 0.03% |
| sponsor | 1 | 0.03% |
| ethnic group | 1 | 0.03% |
| writing system | 1 | 0.03% |
| maintained by | 1 | 0.03% |
| distribution format | 1 | 0.03% |
| day of week | 1 | 0.03% |
| film editor | 1 | 0.03% |
| located on street | 1 | 0.03% |
| published in | 1 | 0.03% |
| head coach | 1 | 0.03% |
| contributor to the creative work or subject | 1 | 0.03% |
| contains settlement | 1 | 0.03% |
| religious order | 1 | 0.03% |
| start point | 1 | 0.03% |
| separated from | 1 | 0.03% |

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
