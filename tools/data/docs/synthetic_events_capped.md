# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_events_capped | train | 4,727 | 4,727 | 122,082 | 16,414 | 17,277 | 3,261 | 7,416 | 24,819 |

## synthetic_events_capped

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4,727 | 4,727 | 272 | 305 | 429 | 122,082 | 16,414 | 17,277 | 3,261 | 7,416 | 24,819 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | records | share |
| --- | --- | --- | --- |
| eng | 4,727 | 4,727 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| train_records | 4,727 |
| train_unique_inputs | 4,727 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

No label marginal has gold on both sides of a split pair.


### Events

**train.** 7,514 instances in 2,566 records, 0 triggerless. Trigger gold 7,416. Argument gold 24,819. Arguments per instance, mean 3.31. Trigger length in whitespace tokens, mean 1.17, single-token share 86.0%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Personnel.StartPosition | 1,070 | 14.24% |
| Business.StartOrg | 951 | 12.66% |
| Transaction.TransferOwnership | 744 | 9.90% |
| Personnel.EndPosition | 654 | 8.70% |
| Contact.Communicate | 621 | 8.26% |
| Transaction.TransferMoney | 548 | 7.29% |
| Contact.Meet | 433 | 5.76% |
| Award.ReceiveAward | 191 | 2.54% |
| Life.BeBorn | 165 | 2.20% |
| Movement.Transport | 165 | 2.20% |
| Justice.TrialHearing | 160 | 2.13% |
| Justice.ChargeIndict | 128 | 1.70% |
| Justice.Sentence | 127 | 1.69% |
| Conflict.Demonstrate | 108 | 1.44% |
| Research.Fund | 97 | 1.29% |
| Justice.Arrest | 96 | 1.28% |
| Life.Injure | 87 | 1.16% |
| Business.MergeOrg | 86 | 1.14% |
| Business.LaunchProduct | 84 | 1.12% |
| Business.DeclareBankruptcy | 81 | 1.08% |
| Disaster.NaturalDisaster | 81 | 1.08% |
| Research.Discover | 71 | 0.94% |
| Life.Die | 61 | 0.81% |
| Conflict.Attack | 61 | 0.81% |
| Tech.Release | 60 | 0.80% |
| Research.Publish | 59 | 0.79% |
| Business.Layoff | 47 | 0.63% |
| Personnel.Elect | 43 | 0.57% |
| Medical.Treat | 38 | 0.51% |
| Disaster.Outbreak | 36 | 0.48% |
| Disaster.Accident | 36 | 0.48% |
| Tech.Breach | 35 | 0.47% |
| Business.EndOrg | 27 | 0.36% |
| Medical.Diagnose | 26 | 0.35% |
| Life.Evacuate | 25 | 0.33% |
| Contact.Negotiate | 20 | 0.27% |
| Disaster.Rescue | 20 | 0.27% |
| Personnel.Nominate | 18 | 0.24% |
| Sport.Compete | 18 | 0.24% |
| Justice.Fine | 18 | 0.24% |
| Justice.Sue | 17 | 0.23% |
| Government.Sanction | 16 | 0.21% |
| Movement.Arrive | 15 | 0.20% |
| Transaction.Donate | 13 | 0.17% |
| Transaction.Lend | 11 | 0.15% |
| Conflict.Ceasefire | 11 | 0.15% |
| Government.EnactLaw | 10 | 0.13% |
| Business.Strike | 8 | 0.11% |
| Justice.Appeal | 4 | 0.05% |
| Government.Protest | 4 | 0.05% |
| Movement.Migrate | 4 | 0.05% |
| Life.Marry | 2 | 0.03% |
| Government.Vote | 2 | 0.03% |
| Life.Divorce | 1 | 0.01% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 24,834 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.14 | 10.29% | 1.49 | 25.39% | 0.10% | 0 | 24,834 | 0 | 7,514 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 7,514 | 0.00% | 765 | 60.00% | 58.82% | 0.52% | 0.00% | 9.00% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 0 | 0 | 765 |

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

| split | mentions the string rule adds | arguments skipped, 2+ clusters | arguments in one multi-mention cluster | other mentions of that cluster | argument spans in 2+ clusters |
| --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | 0 | 0 | 0 |

### Entities

#### train

| label | count | share |
| --- | --- | --- |
| date | 18,676 | 15.30% |
| location | 12,339 | 10.11% |
| organization | 11,337 | 9.29% |
| person | 11,201 | 9.17% |
| cardinal | 9,255 | 7.58% |
| money | 7,254 | 5.94% |
| job title | 6,572 | 5.38% |
| quantity | 4,695 | 3.85% |
| percent | 4,644 | 3.80% |
| company | 3,370 | 2.76% |
| geopolitical entity | 2,892 | 2.37% |
| duration | 2,712 | 2.22% |
| facility | 2,221 | 1.82% |
| product | 2,148 | 1.76% |
| time | 1,897 | 1.55% |
| government agency | 1,407 | 1.15% |
| measurement | 1,250 | 1.02% |
| educational institution | 1,152 | 0.94% |
| disease | 968 | 0.79% |
| region | 917 | 0.75% |
| ordinal | 801 | 0.66% |
| distance | 767 | 0.63% |
| academic degree | 640 | 0.52% |
| nationality | 569 | 0.47% |
| chemical | 556 | 0.46% |
| address | 535 | 0.44% |
| occupation | 501 | 0.41% |
| regulatory body | 470 | 0.38% |
| age | 468 | 0.38% |
| sports team | 427 | 0.35% |
| landmark | 424 | 0.35% |
| fiscal quarter | 418 | 0.34% |
| work of art | 403 | 0.33% |
| medication | 389 | 0.32% |
| area | 383 | 0.31% |
| event | 359 | 0.29% |
| temperature | 349 | 0.29% |
| law | 313 | 0.26% |
| software | 297 | 0.24% |
| frequency | 296 | 0.24% |
| road | 288 | 0.24% |
| speed | 273 | 0.22% |
| military unit | 248 | 0.20% |
| dosage | 242 | 0.20% |
| regulation | 213 | 0.17% |
| non-profit organization | 211 | 0.17% |
| body of water | 197 | 0.16% |
| media outlet | 196 | 0.16% |
| award | 193 | 0.16% |
| deadline | 185 | 0.15% |
| model number | 181 | 0.15% |
| publication | 166 | 0.14% |
| organism | 159 | 0.13% |
| team role | 154 | 0.13% |
| clinical trial | 152 | 0.12% |
| airport | 151 | 0.12% |
| symptom | 149 | 0.12% |
| weight | 130 | 0.11% |
| brand | 130 | 0.11% |
| court case | 127 | 0.10% |
| language | 119 | 0.10% |
| postal code | 118 | 0.10% |
| operating system | 116 | 0.10% |
| military rank | 115 | 0.09% |
| era | 111 | 0.09% |
| crime | 109 | 0.09% |
| conference | 99 | 0.08% |
| honorific | 85 | 0.07% |
| vehicle | 65 | 0.05% |
| mountain | 65 | 0.05% |
| sports competition | 59 | 0.05% |
| protein | 56 | 0.05% |
| family relationship | 53 | 0.04% |
| patent | 51 | 0.04% |
| stock ticker | 48 | 0.04% |
| natural disaster | 44 | 0.04% |
| ethnic group | 43 | 0.04% |
| voltage | 43 | 0.04% |
| programming language | 42 | 0.03% |
| currency | 42 | 0.03% |
| contract | 40 | 0.03% |
| criminal organization | 37 | 0.03% |
| security vulnerability | 37 | 0.03% |
| pathogen | 37 | 0.03% |
| hardware component | 36 | 0.03% |
| protocol | 32 | 0.03% |
| medical procedure | 30 | 0.02% |
| trade union | 23 | 0.02% |
| border | 23 | 0.02% |
| license | 21 | 0.02% |
| species | 19 | 0.02% |
| aircraft | 18 | 0.01% |
| gene | 17 | 0.01% |
| database | 16 | 0.01% |
| vaccine | 16 | 0.01% |
| algorithm | 16 | 0.01% |
| framework | 16 | 0.01% |
| pseudonym | 15 | 0.01% |
| weapon | 13 | 0.01% |
| treaty | 13 | 0.01% |
| anatomical structure | 12 | 0.01% |
| witness | 11 | 0.01% |
| sentence | 10 | 0.01% |
| political affiliation | 9 | 0.01% |
| api | 9 | 0.01% |
| interest rate | 8 | 0.01% |
| file format | 6 | 0.00% |
| holiday | 6 | 0.00% |
| credit rating | 6 | 0.00% |
| political party | 4 | 0.00% |
| book | 4 | 0.00% |
| religious group | 4 | 0.00% |
| war | 4 | 0.00% |
| market index | 4 | 0.00% |
| diagnosis | 4 | 0.00% |
| verdict | 3 | 0.00% |
| vessel | 1 | 0.00% |
| video game | 1 | 0.00% |
| film | 1 | 0.00% |
| allergen | 0 | 0.00% |
| mineral | 0 | 0.00% |
| budget line | 0 | 0.00% |
| financial instrument | 0 | 0.00% |
| song | 0 | 0.00% |
| tax | 0 | 0.00% |

### Relations

#### train

| label | count | share |
| --- | --- | --- |
| works_for | 3,334 | 20.31% |
| headquartered_in | 1,686 | 10.27% |
| located_in | 1,247 | 7.60% |
| leader_of | 1,097 | 6.68% |
| founded | 1,073 | 6.54% |
| competitor_of | 1,072 | 6.53% |
| based_in | 804 | 4.90% |
| produced_by | 699 | 4.26% |
| acquired | 616 | 3.75% |
| affiliated_with | 557 | 3.39% |
| employer_of | 544 | 3.31% |
| member_of | 526 | 3.20% |
| operates_in | 401 | 2.44% |
| supplier_of | 313 | 1.91% |
| subsidiary_of | 233 | 1.42% |
| part_of | 227 | 1.38% |
| reports_to | 175 | 1.07% |
| colleague_of | 159 | 0.97% |
| citizen_of | 144 | 0.88% |
| spouse_of | 136 | 0.83% |
| parent_of | 135 | 0.82% |
| developed_by | 111 | 0.68% |
| board_member_of | 100 | 0.61% |
| regulated_by | 80 | 0.49% |
| owns | 75 | 0.46% |
| sibling_of | 67 | 0.41% |
| invested_in | 56 | 0.34% |
| manufactured_by | 53 | 0.32% |
| designed_by | 50 | 0.30% |
| customer_of | 45 | 0.27% |
| sponsored_by | 44 | 0.27% |
| distributor_of | 42 | 0.26% |
| represents | 41 | 0.25% |
| interacts_with | 39 | 0.24% |
| advisor_to | 36 | 0.22% |
| party_to | 35 | 0.21% |
| compatible_with | 32 | 0.19% |
| participated_in | 29 | 0.18% |
| treats | 24 | 0.15% |
| published_by | 24 | 0.15% |
| authored_by | 22 | 0.13% |
| born_in | 22 | 0.13% |
| mentor_of | 16 | 0.10% |
| sold_to | 15 | 0.09% |
| adjacent_to | 15 | 0.09% |
| merged_with | 14 | 0.09% |
| allied_with | 12 | 0.07% |
| testified_against | 12 | 0.07% |
| relative_of | 11 | 0.07% |
| sued_by | 10 | 0.06% |
| charged_with | 9 | 0.05% |
| derived_from | 9 | 0.05% |
| opposed_to | 9 | 0.05% |
| predecessor_of | 7 | 0.04% |
| named_after | 7 | 0.04% |
| member_state_of | 7 | 0.04% |
| capital_of | 6 | 0.04% |
| died_in | 6 | 0.04% |
| supersedes | 5 | 0.03% |
| licensed_to | 5 | 0.03% |
| won | 4 | 0.02% |
| child_of | 4 | 0.02% |
| borders | 4 | 0.02% |
| implements | 3 | 0.02% |
| successor_of | 3 | 0.02% |
| directed_by | 3 | 0.02% |
| signatory_of | 3 | 0.02% |
| depends_on | 3 | 0.02% |
| composed_by | 3 | 0.02% |
| convicted_of | 2 | 0.01% |
| causes | 1 | 0.01% |
| measured_in | 1 | 0.01% |

### Classifications

**train, actionability.** Majority class 53.8% of 875 answers. The marginal is `true_label`, not the menu.

#### train actionability

| label | count | share |
| --- | --- | --- |
| informational | 471 | 53.83% |
| requires action | 293 | 33.49% |
| advisory | 111 | 12.69% |

**train, audience.** Majority class 34.1% of 1,570 answers. The marginal is `true_label`, not the menu.

#### train audience

| label | count | share |
| --- | --- | --- |
| expert | 536 | 34.14% |
| general public | 491 | 31.27% |
| regulatory | 384 | 24.46% |
| internal | 159 | 10.13% |

**train, certainty.** Majority class 99.5% of 884 answers. The marginal is `true_label`, not the menu.

#### train certainty

| label | count | share |
| --- | --- | --- |
| asserted | 880 | 99.55% |
| hedged | 4 | 0.45% |

**train, document_genre.** Majority class 26.4% of 901 answers. The marginal is `true_label`, not the menu.

#### train document_genre

| label | count | share |
| --- | --- | --- |
| news report | 238 | 26.42% |
| announcement | 228 | 25.31% |
| technical documentation | 116 | 12.87% |
| correspondence | 99 | 10.99% |
| legal filing | 78 | 8.66% |
| advertisement | 56 | 6.22% |
| review | 56 | 6.22% |
| opinion | 30 | 3.33% |

**train, formality.** Majority class 90.6% of 2,174 answers. The marginal is `true_label`, not the menu.

#### train formality

| label | count | share |
| --- | --- | --- |
| formal | 1,969 | 90.57% |
| informal | 205 | 9.43% |

**train, language_register.** Majority class 63.4% of 847 answers. The marginal is `true_label`, not the menu.

#### train language_register

| label | count | share |
| --- | --- | --- |
| technical | 537 | 63.40% |
| plain | 215 | 25.38% |
| legal | 88 | 10.39% |
| colloquial | 7 | 0.83% |

**train, risk_level.** Majority class 33.1% of 825 answers. The marginal is `true_label`, not the menu.

#### train risk_level

| label | count | share |
| --- | --- | --- |
| low | 273 | 33.09% |
| high | 237 | 28.73% |
| none | 205 | 24.85% |
| moderate | 110 | 13.33% |

**train, sentiment.** Majority class 59.6% of 2,113 answers. The marginal is `true_label`, not the menu.

#### train sentiment

| label | count | share |
| --- | --- | --- |
| positive | 1,260 | 59.63% |
| negative | 437 | 20.68% |
| neutral | 416 | 19.69% |

**train, subjectivity.** Majority class 93.2% of 836 answers. The marginal is `true_label`, not the menu.

#### train subjectivity

| label | count | share |
| --- | --- | --- |
| objective | 779 | 93.18% |
| subjective | 57 | 6.82% |

**train, temporal_orientation.** Majority class 71.5% of 835 answers. The marginal is `true_label`, not the menu.

#### train temporal_orientation

| label | count | share |
| --- | --- | --- |
| past | 597 | 71.50% |
| present | 147 | 17.60% |
| future | 71 | 8.50% |
| timeless | 20 | 2.40% |

**train, topic.** Majority class 29.4% of 4,579 answers. The marginal is `true_label`, not the menu.

#### train topic

| label | count | share |
| --- | --- | --- |
| business | 1,347 | 29.42% |
| finance | 691 | 15.09% |
| technology | 567 | 12.38% |
| science | 427 | 9.33% |
| health | 361 | 7.88% |
| legal | 280 | 6.11% |
| environment | 207 | 4.52% |
| politics | 172 | 3.76% |
| sports | 158 | 3.45% |
| education | 98 | 2.14% |
| world | 94 | 2.05% |
| defense | 63 | 1.38% |
| travel | 52 | 1.14% |
| agriculture | 32 | 0.70% |
| energy | 23 | 0.50% |
| entertainment | 7 | 0.15% |

**train, urgency.** Majority class 58.4% of 838 answers. The marginal is `true_label`, not the menu.

#### train urgency

| label | count | share |
| --- | --- | --- |
| routine | 489 | 58.35% |
| elevated | 195 | 23.27% |
| critical | 142 | 16.95% |
| urgent | 12 | 1.43% |

### Structures

**train.**

| record type | field | count | coverage of records |
| --- | --- | --- | --- |
| clinical_finding | condition | 50 | 94.34% |
| clinical_finding | patient_group | 47 | 88.68% |
| clinical_finding | treatment | 40 | 75.47% |
| clinical_finding | outcome | 35 | 66.04% |
| clinical_finding | record_metadata | natural/condition ×50, natural/patient_group ×3 |  |
| company_profile | name | 189 | 100.00% |
| company_profile | size | 187 | 98.94% |
| company_profile | headquarters | 170 | 89.95% |
| company_profile | industry | 119 | 62.96% |
| company_profile | founded | 74 | 39.15% |
| company_profile | record_metadata | natural/name ×182 |  |
| event_listing | date | 45 | 84.91% |
| event_listing | venue | 44 | 83.02% |
| event_listing | organizer | 43 | 81.13% |
| event_listing | name | 41 | 77.36% |
| event_listing | price | 9 | 16.98% |
| event_listing | record_metadata | natural/name ×41, natural/venue ×12 |  |
| financial_report | organization | 53 | 100.00% |
| financial_report | period | 49 | 92.45% |
| financial_report | revenue | 45 | 84.91% |
| financial_report | net_income | 36 | 67.92% |
| financial_report | record_metadata | natural/organization ×53 |  |
| incident_report | severity | 110 | 100.00% |
| incident_report | date | 109 | 99.09% |
| incident_report | location | 105 | 95.45% |
| incident_report | incident_type | 72 | 65.45% |
| incident_report | casualties | 63 | 57.27% |
| incident_report | record_metadata | natural/location ×105, natural/incident_type ×2, natural/date ×2, natural/severity ×1 |  |
| job_posting | title | 20 | 100.00% |
| job_posting | company | 20 | 100.00% |
| job_posting | employment_type | 20 | 100.00% |
| job_posting | location | 19 | 95.00% |
| job_posting | record_metadata | natural/location ×14, natural/title ×1 |  |
| legal_case | status | 29 | 100.00% |
| legal_case | court | 28 | 96.55% |
| legal_case | plaintiff | 26 | 89.66% |
| legal_case | defendant | 25 | 86.21% |
| legal_case | case_name | 15 | 51.72% |
| legal_case | record_metadata | natural/case_name ×15, natural/court ×13, natural/status ×1 |  |
| person_profile | name | 1,522 | 99.93% |
| person_profile | employer | 1,499 | 98.42% |
| person_profile | location | 1,474 | 96.78% |
| person_profile | role | 1,421 | 93.30% |
| person_profile | record_metadata | natural/location ×1,159, natural/name ×47 |  |
| product | name | 347 | 99.43% |
| product | condition | 345 | 98.85% |
| product | brand | 343 | 98.28% |
| product | price | 294 | 84.24% |
| product | record_metadata | natural/name ×346, natural/brand ×2 |  |
| real_estate_listing | address | 22 | 95.65% |
| real_estate_listing | price | 22 | 95.65% |
| real_estate_listing | bedrooms | 22 | 95.65% |
| real_estate_listing | property_type | 22 | 95.65% |
| real_estate_listing | area | 21 | 91.30% |
| real_estate_listing | record_metadata | natural/address ×22, natural/price ×1 |  |
| research_paper | year | 31 | 96.88% |
| research_paper | title | 29 | 90.62% |
| research_paper | venue | 28 | 87.50% |
| research_paper | authors | 22 | 68.75% |
| research_paper | topic | 9 | 28.12% |
| research_paper | record_metadata | natural/title ×29, natural/authors ×2, natural/venue ×1 |  |
| software_release | product | 37 | 100.00% |
| software_release | release_date | 35 | 94.59% |
| software_release | version | 32 | 86.49% |
| software_release | platform | 23 | 62.16% |
| software_release | record_metadata | natural/product ×36 |  |
| transaction | amount | 767 | 98.71% |
| transaction | item | 752 | 96.78% |
| transaction | date | 752 | 96.78% |
| transaction | buyer | 745 | 95.88% |
| transaction | seller | 723 | 93.05% |
| transaction | record_metadata | natural/item ×752, natural/amount ×23 |  |
| vehicle_listing | model | 13 | 100.00% |
| vehicle_listing | price | 13 | 100.00% |
| vehicle_listing | year | 12 | 92.31% |
| vehicle_listing | make | 11 | 84.62% |
| vehicle_listing | mileage | 4 | 30.77% |
| vehicle_listing | record_metadata | natural/make ×11, natural/model ×2 |  |

### Provenance

| head | who wrote the gold |
| --- | --- |
| entities | llm_synthetic |
| relations | llm_synthetic |
| events | llm_synthetic |
| classifications | llm_synthetic |
| structures | llm_synthetic |
