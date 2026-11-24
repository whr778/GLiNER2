# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| casie_off | train | 798 | 797 | 15,946 | 0 | 0 | 0 | 6,110 | 17,817 |
| casie_off | val | 95 | 95 | 1,927 | 0 | 0 | 0 | 765 | 2,171 |
| casie_off | test | 107 | 107 | 2,136 | 0 | 0 | 0 | 856 | 2,413 |

## casie_off

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 798 | 797 | 410 | 678 | 1272 | 15,946 | 0 | 0 | 0 | 6,110 | 17,817 |
| val | 95 | 95 | 480 | 666 | 1248 | 1,927 | 0 | 0 | 0 | 765 | 2,171 |
| test | 107 | 107 | 431 | 609 | 965 | 2,136 | 0 | 0 | 0 | 856 | 2,413 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 798 | 95 | 107 | 1,000 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| test_records | 107 |
| test_unique_inputs | 107 |
| train_records | 798 |
| train_unique_inputs | 797 |
| val_records | 95 |
| val_unique_inputs | 95 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / val | 0.0503 | 15,946 / 1,927 | Organization 13%→16%, Device 4%→2%, System 7%→5%, Person 16%→17%, File 3%→3% |
| entities (per mention) | train / test | 0.0580 | 15,946 / 2,136 | Person 16%→18%, Data 5%→3%, Vulnerability 11%→13%, System 7%→5%, Patch 4%→3% |
| entities (per mention) | val / test | 0.0640 | 1,927 / 2,136 | Organization 16%→13%, Data 5%→3%, Vulnerability 11%→13%, Person 17%→18%, Device 2%→3% |
| event types (per trigger mention) | train / val | 0.1352 | 25 / 838 | Cyber.Ransom 28%→20%, Cyber.DiscoverVulnerability 28%→22%, Cyber.PatchVulnerability 12%→17%, Cyber.Phishing 12%→17%, Cyber.Databreach 20%→24% |
| event types (per trigger mention) | train / test | 0.1111 | 25 / 938 | Cyber.Phishing 12%→20%, Cyber.Ransom 28%→20%, Cyber.Databreach 20%→17%, Cyber.PatchVulnerability 12%→14%, Cyber.DiscoverVulnerability 28%→29% |
| event types (per trigger mention) | val / test | 0.0979 | 838 / 938 | Cyber.Databreach 24%→17%, Cyber.DiscoverVulnerability 22%→29%, Cyber.Phishing 17%→20%, Cyber.PatchVulnerability 17%→14%, Cyber.Ransom 20%→20% |
| event roles (per argument) | train / val | 0.0425 | 17,992 / 2,197 | Victim 15%→17%, Trusted-Entity 4%→3%, Vulnerable_System_Owner 2%→1%, Attacker 7%→8%, Vulnerable_System 9%→9% |
| event roles (per argument) | train / test | 0.0586 | 17,992 / 2,439 | Compromised-Data 9%→7%, Vulnerability 11%→12%, Vulnerable_System_Owner 2%→3%, Tool 5%→6%, Patch 4%→3% |
| event roles (per argument) | val / test | 0.0675 | 2,197 / 2,439 | Victim 17%→15%, Compromised-Data 9%→7%, Vulnerable_System_Owner 1%→3%, Tool 5%→6%, Vulnerability 11%→12% |

Thin support in val, event roles (per argument): 2 label(s) under 10 gold. Damage-Amount ×5, Supported_Platform ×6.
Thin support in test, event roles (per argument): 2 label(s) under 10 gold. Damage-Amount ×1, Supported_Platform ×2.

### Events

**train.** 6,708 instances in 798 records, 6,683 triggerless. Trigger gold 6,110. Argument gold 17,817. Arguments per instance, mean 2.68. Trigger length in whitespace tokens, mean 1.84, single-token share 24.0%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Cyber.Ransom | 7 | 28.00% |
| Cyber.DiscoverVulnerability | 7 | 28.00% |
| Cyber.Databreach | 5 | 20.00% |
| Cyber.Phishing | 3 | 12.00% |
| Cyber.PatchVulnerability | 3 | 12.00% |

**val.** 838 instances in 95 records, 0 triggerless. Trigger gold 765. Argument gold 2,171. Arguments per instance, mean 2.62. Trigger length in whitespace tokens, mean 1.85, single-token share 45.0%.

#### val event types

| label | count | share |
| --- | --- | --- |
| Cyber.Databreach | 200 | 23.87% |
| Cyber.DiscoverVulnerability | 188 | 22.43% |
| Cyber.Ransom | 168 | 20.05% |
| Cyber.PatchVulnerability | 142 | 16.95% |
| Cyber.Phishing | 140 | 16.71% |

**test.** 938 instances in 107 records, 0 triggerless. Trigger gold 856. Argument gold 2,413. Arguments per instance, mean 2.60. Trigger length in whitespace tokens, mean 1.88, single-token share 43.7%.

#### test event types

| label | count | share |
| --- | --- | --- |
| Cyber.DiscoverVulnerability | 268 | 28.57% |
| Cyber.Ransom | 191 | 20.36% |
| Cyber.Phishing | 188 | 20.04% |
| Cyber.Databreach | 155 | 16.52% |
| Cyber.PatchVulnerability | 136 | 14.50% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 17,911 | 0.00% | 0 | 0 | 0 | 0 | 17,911 | 81 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 2,197 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 2,439 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

**train by role.**

| role | inside | placed | inside share | unmeasured |
| --- | --- | --- | --- | --- |
| Victim | 0 | 2,692 | 0.00% | 19 |
| Vulnerability | 0 | 1,992 | 0.00% | 10 |
| Vulnerable_System | 0 | 1,638 | 0.00% | 4 |
| Compromised-Data | 0 | 1,627 | 0.00% | 3 |
| Attacker | 0 | 1,301 | 0.00% | 5 |
| Time | 0 | 1,106 | 0.00% | 9 |
| Discoverer | 0 | 1,076 | 0.00% | 7 |
| Tool | 0 | 868 | 0.00% | 2 |
| Attack-Pattern | 0 | 827 | 0.00% | 0 |
| Trusted-Entity | 0 | 699 | 0.00% | 1 |
| Releaser | 0 | 634 | 0.00% | 3 |
| Patch | 0 | 630 | 0.00% | 1 |
| Capabilities | 0 | 439 | 0.00% | 5 |
| Purpose | 0 | 437 | 0.00% | 0 |
| Price | 0 | 305 | 0.00% | 1 |
| Vulnerable_System_Owner | 0 | 294 | 0.00% | 0 |
| CVE | 0 | 251 | 0.00% | 0 |
| Vulnerable_System_Version | 0 | 241 | 0.00% | 0 |
| Payment-Method | 0 | 154 | 0.00% | 1 |
| Place | 0 | 136 | 0.00% | 6 |
| Number-of-Data | 0 | 123 | 0.00% | 0 |
| Patch-Number | 0 | 111 | 0.00% | 0 |
| Supported_Platform | 0 | 109 | 0.00% | 0 |
| Issues-Addressed | 0 | 100 | 0.00% | 1 |
| Number-of-Victim | 0 | 85 | 0.00% | 2 |
| Damage-Amount | 0 | 36 | 0.00% | 1 |

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.51 | 25.88% | 2.27 | 36.93% | 0.10% | 17,911 | 81 | 6,683 | 25 |
| val | 1.49 | 25.66% | 2.28 | 39.24% | 0.00% | 0 | 2,197 | 0 | 838 |
| test | 1.49 | 26.01% | 2.28 | 37.65% | 0.03% | 0 | 2,439 | 0 | 938 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 6,708 | 0.00% | 22,714 | 10.27% | 9.69% | 2.25% | 0.29% | 63.16% |
| val | 838 | 0.00% | 2,947 | 10.21% | 9.98% | 1.93% | 0.34% | 65.26% |
| test | 938 | 0.00% | 3,397 | 10.39% | 9.48% | 2.94% | 0.50% | 70.09% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 0 | 0 | 22,714 |
| val | 0 | 0 | 2,947 |
| test | 0 | 0 | 3,397 |

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

| split | mentions the string rule adds | arguments skipped, 2+ clusters | arguments in one multi-mention cluster | other mentions of that cluster | argument spans in 2+ clusters |
| --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | 0 | 0 | 0 |
| val | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 |

### Entities

#### train

| label | count | share |
| --- | --- | --- |
| Person | 2,544 | 15.95% |
| Organization | 2,106 | 13.21% |
| Vulnerability | 1,782 | 11.18% |
| Capabilities | 1,350 | 8.47% |
| Time | 1,041 | 6.53% |
| System | 1,038 | 6.51% |
| PII | 798 | 5.00% |
| Data | 737 | 4.62% |
| Software | 574 | 3.60% |
| Patch | 573 | 3.59% |
| Device | 571 | 3.58% |
| File | 439 | 2.75% |
| Purpose | 434 | 2.72% |
| Malware | 348 | 2.18% |
| Version | 334 | 2.09% |
| Money | 322 | 2.02% |
| Website | 256 | 1.61% |
| CVE | 217 | 1.36% |
| Number | 197 | 1.24% |
| PaymentMethod | 149 | 0.93% |
| GPE | 136 | 0.85% |

#### val

| label | count | share |
| --- | --- | --- |
| Person | 325 | 16.87% |
| Organization | 304 | 15.78% |
| Vulnerability | 211 | 10.95% |
| Capabilities | 166 | 8.61% |
| Time | 126 | 6.54% |
| System | 104 | 5.40% |
| Data | 95 | 4.93% |
| PII | 90 | 4.67% |
| Software | 69 | 3.58% |
| Patch | 64 | 3.32% |
| File | 62 | 3.22% |
| Purpose | 51 | 2.65% |
| Version | 49 | 2.54% |
| Device | 41 | 2.13% |
| Malware | 34 | 1.76% |
| Money | 33 | 1.71% |
| Number | 27 | 1.40% |
| Website | 25 | 1.30% |
| CVE | 22 | 1.14% |
| GPE | 15 | 0.78% |
| PaymentMethod | 14 | 0.73% |

#### test

| label | count | share |
| --- | --- | --- |
| Person | 383 | 17.93% |
| Organization | 285 | 13.34% |
| Vulnerability | 270 | 12.64% |
| Capabilities | 196 | 9.18% |
| Time | 142 | 6.65% |
| System | 116 | 5.43% |
| PII | 97 | 4.54% |
| Software | 72 | 3.37% |
| Device | 68 | 3.18% |
| Purpose | 67 | 3.14% |
| Data | 62 | 2.90% |
| File | 62 | 2.90% |
| Patch | 60 | 2.81% |
| Malware | 52 | 2.43% |
| Version | 52 | 2.43% |
| Money | 36 | 1.69% |
| CVE | 30 | 1.40% |
| Website | 29 | 1.36% |
| PaymentMethod | 24 | 1.12% |
| Number | 23 | 1.08% |
| GPE | 10 | 0.47% |

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
