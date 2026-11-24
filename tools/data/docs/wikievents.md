# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wikievents | train | 199 | 199 | 16,094 | 0 | 0 | 0 | 2,263 | 4,198 |
| wikievents | dev | 20 | 20 | 1,427 | 0 | 0 | 0 | 269 | 416 |
| wikievents | test | 20 | 20 | 1,602 | 0 | 0 | 0 | 239 | 515 |

## wikievents

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 199 | 199 | 562 | 1450 | 11271 | 16,094 | 0 | 0 | 0 | 2,263 | 4,198 |
| dev | 20 | 20 | 549 | 919 | 2083 | 1,427 | 0 | 0 | 0 | 269 | 416 |
| test | 20 | 20 | 601 | 1093 | 2068 | 1,602 | 0 | 0 | 0 | 239 | 515 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | dev | test | records | share |
| --- | --- | --- | --- | --- | --- |
| eng | 199 | 20 | 20 | 239 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| dev_records | 20 |
| dev_unique_inputs | 20 |
| test_records | 20 |
| test_unique_inputs | 20 |
| train_records | 199 |
| train_unique_inputs | 199 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| entities (per mention) | train / dev | 0.0377 | 16,094 / 1,427 | LOC 4%→6%, PER 48%→46%, WEA 4%→3%, ORG 15%→16%, MHI 0%→1% |
| entities (per mention) | train / test | 0.0458 | 16,094 / 1,602 | PER 48%→45%, WEA 4%→6%, VAL 2%→3%, COM 1%→0%, GPE 14%→15% |
| entities (per mention) | dev / test | 0.0628 | 1,427 / 1,602 | WEA 3%→6%, LOC 6%→4%, VAL 1%→3%, PER 46%→45%, ORG 16%→15% |
| event types (per trigger mention) | train / dev | 0.1691 WARN | 3,191 / 345 | Disaster.DiseaseOutbreak.Unspecified 0%→3%, Contact.RequestCommand.Unspecified 1%→3%, Contact.Contact.Meet 1%→3%, Conflict.Attack.DetonateExplode 21%→19%, Justice.ArrestJailDetain.Unspecified 5%→3% |
| event types (per trigger mention) | train / test | 0.1537 WARN | 3,191 / 365 | Contact.Contact.Broadcast 5%→8%, Transaction.ExchangeBuySell.Unspecified 1%→4%, Justice.InvestigateCrime.Unspecified 3%→0%, Movement.Transportation.Unspecified 4%→2%, Conflict.Attack.DetonateExplode 21%→23% |
| event types (per trigger mention) | dev / test | 0.2091 WARN | 345 / 365 | Contact.Contact.Broadcast 4%→8%, Conflict.Attack.DetonateExplode 19%→23%, Movement.Transportation.Unspecified 5%→2%, Disaster.DiseaseOutbreak.Unspecified 3%→0%, Transaction.ExchangeBuySell.Unspecified 2%→4% |
| event roles (per argument) | train / dev | 0.2076 WARN | 4,435 / 428 | Recipient 2%→6%, Victim 16%→19%, Transporter 1%→4%, Communicator 4%→7%, Participant 7%→9% |
| event roles (per argument) | train / test | 0.1942 WARN | 4,435 / 566 | Target 8%→13%, Place 12%→15%, Communicator 4%→6%, Defendant 5%→3%, AcquiredEntity 1%→3% |
| event roles (per argument) | dev / test | 0.2490 WARN | 428 / 566 | Target 6%→13%, Victim 19%→16%, Transporter 4%→0%, Participant 9%→6%, Recipient 6%→3% |

Thin support in dev, entities (per mention): 3 label(s) under 10 gold. MON ×1, ABS ×2, SID ×3.
Thin support in dev, event types (per trigger mention): 24 label(s) under 10 gold. Movement.Transportation.IllegalTransportation ×1, Conflict.Demonstrate.DemonstrateWithViolence ×1, Justice.Sentence.Unspecified ×1, Movement.Transportation.Evacuation ×1, Cognitive.IdentifyCategorize.Unspecified ×1, Contact.RequestCommand.Meet ×1, Personnel.EndPosition.Unspecified ×2, Contact.RequestCommand.Broadcast ×2, Contact.ThreatenCoerce.Broadcast ×2, Medical.Intervention.Unspecified ×2, Cognitive.Inspection.SensoryObserve ×2, ArtifactExistence.ManufactureAssemble.Unspecified ×2, Justice.Convict.Unspecified ×2, Justice.ReleaseParole.Unspecified ×3, ArtifactExistence.DamageDestroyDisableDismantle.Destroy ×3, Control.ImpedeInterfereWith.Unspecified ×4, Justice.TrialHearing.Unspecified ×4, Justice.InvestigateCrime.Unspecified ×4, ArtifactExistence.DamageDestroyDisableDismantle.Damage ×4, Conflict.Demonstrate.Unspecified ×5, Justice.ChargeIndict.Unspecified ×6, Transaction.ExchangeBuySell.Unspecified ×6, Contact.ThreatenCoerce.Unspecified ×7, Disaster.DiseaseOutbreak.Unspecified ×9.
Unseen in train, present in dev, event roles (per argument): Regulator ×1.
Thin support in dev, event roles (per argument): 20 label(s) under 10 gold. Regulator ×1, Patient ×1, PaymentBarter ×2, Injurer ×3, Jailer ×3, Demonstrator ×3, JudgeCourt ×3, Investigator ×3, Origin ×3, Giver ×3, Instrument ×4, Employee ×4, Impeder ×5, Vehicle ×5, AcquiredEntity ×6, Killer ×7, Perpetrator ×7, Topic ×8, PassengerArtifact ×8, Artifact ×8.
Thin support in test, entities (per mention): 5 label(s) under 10 gold. BOD ×1, CRM ×1, MON ×2, ABS ×4, TTL ×5.
Unseen in train, present in test, event types (per trigger mention): Contact.RequestCommand.Correspondence ×1.
Thin support in test, event types (per trigger mention): 26 label(s) under 10 gold. Cognitive.Research.Unspecified ×1, Personnel.StartPosition.Unspecified ×1, Contact.RequestCommand.Correspondence ×1, Movement.Transportation.IllegalTransportation ×1, Justice.InvestigateCrime.Unspecified ×1, ArtifactExistence.DamageDestroyDisableDismantle.DisableDefuse ×1, Control.ImpedeInterfereWith.Unspecified ×1, Conflict.Demonstrate.Unspecified ×1, Disaster.DiseaseOutbreak.Unspecified ×1, Contact.Contact.Correspondence ×2, ArtifactExistence.ManufactureAssemble.Unspecified ×2, Contact.ThreatenCoerce.Unspecified ×2, ArtifactExistence.DamageDestroyDisableDismantle.Dismantle ×2, Medical.Intervention.Unspecified ×2, Justice.TrialHearing.Unspecified ×3, Contact.RequestCommand.Unspecified ×3, Contact.Contact.Meet ×4, ArtifactExistence.DamageDestroyDisableDismantle.Damage ×4, ArtifactExistence.DamageDestroyDisableDismantle.Destroy ×4, Cognitive.Inspection.SensoryObserve ×4, Disaster.Crash.Unspecified ×4, Cognitive.IdentifyCategorize.Unspecified ×5, GenericCrime.GenericCrime.GenericCrime ×5, Movement.Transportation.Unspecified ×6, Justice.Sentence.Unspecified ×6, Justice.ChargeIndict.Unspecified ×9.
Unseen in train, present in test, event roles (per argument): Position ×1.
Thin support in test, event roles (per argument): 30 label(s) under 10 gold. Origin ×1, Destroyer ×1, Researcher ×1, Dismantler ×1, Treater ×1, Position ×1, Investigator ×1, Disabler ×1, Impeder ×1, Demonstrator ×1, Transporter ×2, Perpetrator ×2, ManufacturerAssembler ×2, ObservedEntity ×2, Patient ×2, Employee ×2, Vehicle ×2, CrashObject ×2, Identifier ×3, Topic ×3, Damager ×3, Observer ×3, JudgeCourt ×4, Prosecutor ×4, IdentifiedObject ×5, Destination ×5, Jailer ×5, Giver ×5, Injurer ×6, PassengerArtifact ×7.

### Events

**train.** 3,191 instances in 191 records, 0 triggerless. Trigger gold 2,263. Argument gold 4,198. Arguments per instance, mean 1.39. Trigger length in whitespace tokens, mean 1.04, single-token share 96.3%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Conflict.Attack.DetonateExplode | 663 | 20.78% |
| Life.Die.Unspecified | 420 | 13.16% |
| Conflict.Attack.Unspecified | 333 | 10.44% |
| Life.Injure.Unspecified | 242 | 7.58% |
| Justice.ArrestJailDetain.Unspecified | 163 | 5.11% |
| Contact.Contact.Unspecified | 157 | 4.92% |
| Contact.Contact.Broadcast | 150 | 4.70% |
| Movement.Transportation.Unspecified | 124 | 3.89% |
| Justice.ChargeIndict.Unspecified | 89 | 2.79% |
| Justice.InvestigateCrime.Unspecified | 86 | 2.70% |
| ArtifactExistence.ManufactureAssemble.Unspecified | 65 | 2.04% |
| GenericCrime.GenericCrime.GenericCrime | 58 | 1.82% |
| Cognitive.IdentifyCategorize.Unspecified | 54 | 1.69% |
| Justice.Sentence.Unspecified | 54 | 1.69% |
| Justice.Convict.Unspecified | 50 | 1.57% |
| ArtifactExistence.DamageDestroyDisableDismantle.Destroy | 44 | 1.38% |
| ArtifactExistence.DamageDestroyDisableDismantle.Damage | 40 | 1.25% |
| Contact.Contact.Meet | 39 | 1.22% |
| Justice.TrialHearing.Unspecified | 34 | 1.07% |
| Transaction.ExchangeBuySell.Unspecified | 32 | 1.00% |
| Cognitive.Inspection.SensoryObserve | 31 | 0.97% |
| Conflict.Demonstrate.Unspecified | 26 | 0.81% |
| Movement.Transportation.Evacuation | 25 | 0.78% |
| Movement.Transportation.PreventPassage | 22 | 0.69% |
| Medical.Intervention.Unspecified | 19 | 0.60% |
| Contact.RequestCommand.Unspecified | 19 | 0.60% |
| Contact.ThreatenCoerce.Unspecified | 17 | 0.53% |
| Contact.Contact.Correspondence | 15 | 0.47% |
| Movement.Transportation.IllegalTransportation | 12 | 0.38% |
| Justice.ReleaseParole.Unspecified | 12 | 0.38% |
| ArtifactExistence.DamageDestroyDisableDismantle.Unspecified | 11 | 0.34% |
| Cognitive.Research.Unspecified | 10 | 0.31% |
| Control.ImpedeInterfereWith.Unspecified | 9 | 0.28% |
| ArtifactExistence.DamageDestroyDisableDismantle.DisableDefuse | 7 | 0.22% |
| Cognitive.TeachingTrainingLearning.Unspecified | 7 | 0.22% |
| ArtifactExistence.DamageDestroyDisableDismantle.Dismantle | 6 | 0.19% |
| Disaster.Crash.Unspecified | 6 | 0.19% |
| Conflict.Defeat.Unspecified | 5 | 0.16% |
| Disaster.DiseaseOutbreak.Unspecified | 5 | 0.16% |
| Conflict.Demonstrate.DemonstrateWithViolence | 4 | 0.13% |
| Justice.Acquit.Unspecified | 4 | 0.13% |
| Life.Infect.Unspecified | 4 | 0.13% |
| Personnel.EndPosition.Unspecified | 4 | 0.13% |
| Contact.RequestCommand.Broadcast | 3 | 0.09% |
| Contact.ThreatenCoerce.Broadcast | 3 | 0.09% |
| Contact.ThreatenCoerce.Correspondence | 3 | 0.09% |
| Personnel.StartPosition.Unspecified | 3 | 0.09% |
| Contact.RequestCommand.Meet | 2 | 0.06% |

**dev.** 345 instances in 20 records, 0 triggerless. Trigger gold 269. Argument gold 416. Arguments per instance, mean 1.24. Trigger length in whitespace tokens, mean 1.07, single-token share 96.5%.

#### dev event types

| label | count | share |
| --- | --- | --- |
| Conflict.Attack.DetonateExplode | 66 | 19.13% |
| Life.Die.Unspecified | 47 | 13.62% |
| Conflict.Attack.Unspecified | 36 | 10.43% |
| Life.Injure.Unspecified | 25 | 7.25% |
| Contact.Contact.Unspecified | 22 | 6.38% |
| Movement.Transportation.Unspecified | 17 | 4.93% |
| Contact.Contact.Broadcast | 14 | 4.06% |
| Justice.ArrestJailDetain.Unspecified | 12 | 3.48% |
| GenericCrime.GenericCrime.GenericCrime | 11 | 3.19% |
| Contact.RequestCommand.Unspecified | 10 | 2.90% |
| Contact.Contact.Meet | 10 | 2.90% |
| Disaster.DiseaseOutbreak.Unspecified | 9 | 2.61% |
| Contact.ThreatenCoerce.Unspecified | 7 | 2.03% |
| Justice.ChargeIndict.Unspecified | 6 | 1.74% |
| Transaction.ExchangeBuySell.Unspecified | 6 | 1.74% |
| Conflict.Demonstrate.Unspecified | 5 | 1.45% |
| Control.ImpedeInterfereWith.Unspecified | 4 | 1.16% |
| Justice.TrialHearing.Unspecified | 4 | 1.16% |
| Justice.InvestigateCrime.Unspecified | 4 | 1.16% |
| ArtifactExistence.DamageDestroyDisableDismantle.Damage | 4 | 1.16% |
| Justice.ReleaseParole.Unspecified | 3 | 0.87% |
| ArtifactExistence.DamageDestroyDisableDismantle.Destroy | 3 | 0.87% |
| Personnel.EndPosition.Unspecified | 2 | 0.58% |
| Contact.RequestCommand.Broadcast | 2 | 0.58% |
| Contact.ThreatenCoerce.Broadcast | 2 | 0.58% |
| Medical.Intervention.Unspecified | 2 | 0.58% |
| Cognitive.Inspection.SensoryObserve | 2 | 0.58% |
| ArtifactExistence.ManufactureAssemble.Unspecified | 2 | 0.58% |
| Justice.Convict.Unspecified | 2 | 0.58% |
| Movement.Transportation.IllegalTransportation | 1 | 0.29% |
| Conflict.Demonstrate.DemonstrateWithViolence | 1 | 0.29% |
| Justice.Sentence.Unspecified | 1 | 0.29% |
| Movement.Transportation.Evacuation | 1 | 0.29% |
| Cognitive.IdentifyCategorize.Unspecified | 1 | 0.29% |
| Contact.RequestCommand.Meet | 1 | 0.29% |

**test.** 365 instances in 19 records, 0 triggerless. Trigger gold 239. Argument gold 515. Arguments per instance, mean 1.55. Trigger length in whitespace tokens, mean 1.05, single-token share 95.3%.

#### test event types

| label | count | share |
| --- | --- | --- |
| Conflict.Attack.DetonateExplode | 84 | 23.01% |
| Life.Die.Unspecified | 55 | 15.07% |
| Conflict.Attack.Unspecified | 40 | 10.96% |
| Life.Injure.Unspecified | 31 | 8.49% |
| Contact.Contact.Broadcast | 31 | 8.49% |
| Contact.Contact.Unspecified | 21 | 5.75% |
| Transaction.ExchangeBuySell.Unspecified | 14 | 3.84% |
| Justice.ArrestJailDetain.Unspecified | 13 | 3.56% |
| Justice.ChargeIndict.Unspecified | 9 | 2.47% |
| Movement.Transportation.Unspecified | 6 | 1.64% |
| Justice.Sentence.Unspecified | 6 | 1.64% |
| Cognitive.IdentifyCategorize.Unspecified | 5 | 1.37% |
| GenericCrime.GenericCrime.GenericCrime | 5 | 1.37% |
| Contact.Contact.Meet | 4 | 1.10% |
| ArtifactExistence.DamageDestroyDisableDismantle.Damage | 4 | 1.10% |
| ArtifactExistence.DamageDestroyDisableDismantle.Destroy | 4 | 1.10% |
| Cognitive.Inspection.SensoryObserve | 4 | 1.10% |
| Disaster.Crash.Unspecified | 4 | 1.10% |
| Justice.TrialHearing.Unspecified | 3 | 0.82% |
| Contact.RequestCommand.Unspecified | 3 | 0.82% |
| Contact.Contact.Correspondence | 2 | 0.55% |
| ArtifactExistence.ManufactureAssemble.Unspecified | 2 | 0.55% |
| Contact.ThreatenCoerce.Unspecified | 2 | 0.55% |
| ArtifactExistence.DamageDestroyDisableDismantle.Dismantle | 2 | 0.55% |
| Medical.Intervention.Unspecified | 2 | 0.55% |
| Cognitive.Research.Unspecified | 1 | 0.27% |
| Personnel.StartPosition.Unspecified | 1 | 0.27% |
| Contact.RequestCommand.Correspondence | 1 | 0.27% |
| Movement.Transportation.IllegalTransportation | 1 | 0.27% |
| Justice.InvestigateCrime.Unspecified | 1 | 0.27% |
| ArtifactExistence.DamageDestroyDisableDismantle.DisableDefuse | 1 | 0.27% |
| Control.ImpedeInterfereWith.Unspecified | 1 | 0.27% |
| Conflict.Demonstrate.Unspecified | 1 | 0.27% |
| Disaster.DiseaseOutbreak.Unspecified | 1 | 0.27% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 4,435 |
| dev | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 428 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 566 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3.30 | 56.39% | 4.54 | 64.41% | 0.31% | 0 | 4,435 | 0 | 3,191 |
| dev | 2.22 | 46.22% | 3.64 | 58.22% | 0.39% | 0 | 428 | 0 | 345 |
| test | 5.09 | 61.64% | 4.22 | 68.03% | 0.32% | 0 | 566 | 0 | 365 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3,191 | 0.00% | 8,292 | 10.94% | 10.14% | 7.25% | 4.40% | 70.16% |
| dev | 345 | 0.00% | 712 | 11.38% | 9.97% | 6.60% | 3.65% | 70.00% |
| test | 365 | 0.00% | 1,034 | 15.86% | 14.41% | 10.06% | 5.71% | 89.47% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

| split | docs | clusters | mentions | multi-mention | singletons | mention pairs | pronoun mentions | pronoun share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 194 | 4,609 | 20,941 | 4,609 | 0 | 124,879 | 4,500 | 21.49% |
| dev | 20 | 402 | 1,648 | 402 | 0 | 5,851 | 364 | 22.09% |
| test | 20 | 451 | 1,941 | 451 | 0 | 9,604 | 441 | 22.72% |

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 0 | 0 | 8,292 |
| dev | 0 | 0 | 712 |
| test | 0 | 0 | 1,034 |

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

| split | mentions the string rule adds | arguments skipped, 2+ clusters | arguments in one multi-mention cluster | other mentions of that cluster | argument spans in 2+ clusters |
| --- | --- | --- | --- | --- | --- |
| train | 17,392 | 656 | 0 | 0 | 0 |
| dev | 1,056 | 65 | 0 | 0 | 0 |
| test | 2,111 | 77 | 0 | 0 | 0 |

### Entities

#### train

| label | count | share |
| --- | --- | --- |
| PER | 7,705 | 47.87% |
| ORG | 2,398 | 14.90% |
| GPE | 2,199 | 13.66% |
| FAC | 1,411 | 8.77% |
| WEA | 715 | 4.44% |
| LOC | 636 | 3.95% |
| VEH | 422 | 2.62% |
| VAL | 263 | 1.63% |
| COM | 181 | 1.12% |
| TTL | 66 | 0.41% |
| INF | 25 | 0.16% |
| MON | 19 | 0.12% |
| ABS | 18 | 0.11% |
| MHI | 16 | 0.10% |
| BOD | 13 | 0.08% |
| SID | 4 | 0.02% |
| CRM | 3 | 0.02% |

#### dev

| label | count | share |
| --- | --- | --- |
| PER | 663 | 46.46% |
| ORG | 225 | 15.77% |
| GPE | 196 | 13.74% |
| FAC | 120 | 8.41% |
| LOC | 81 | 5.68% |
| WEA | 48 | 3.36% |
| VEH | 37 | 2.59% |
| VAL | 20 | 1.40% |
| COM | 11 | 0.77% |
| TTL | 10 | 0.70% |
| MHI | 10 | 0.70% |
| SID | 3 | 0.21% |
| ABS | 2 | 0.14% |
| MON | 1 | 0.07% |

#### test

| label | count | share |
| --- | --- | --- |
| PER | 725 | 45.26% |
| GPE | 235 | 14.67% |
| ORG | 235 | 14.67% |
| FAC | 152 | 9.49% |
| WEA | 93 | 5.81% |
| LOC | 62 | 3.87% |
| VAL | 47 | 2.93% |
| VEH | 40 | 2.50% |
| TTL | 5 | 0.31% |
| ABS | 4 | 0.25% |
| MON | 2 | 0.12% |
| BOD | 1 | 0.06% |
| CRM | 1 | 0.06% |

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
