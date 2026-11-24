# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wikievents_off | train | 199 | 199 | 16,094 | 0 | 0 | 0 | 2,263 | 4,198 |

## wikievents_off

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 199 | 199 | 562 | 1450 | 11271 | 16,094 | 0 | 0 | 0 | 2,263 | 4,198 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | records | share |
| --- | --- | --- | --- |
| eng | 199 | 199 | 100.00% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| train_records | 199 |
| train_unique_inputs | 199 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

No label marginal has gold on both sides of a split pair.


### Events

**train.** 3,191 instances in 191 records, 3,191 triggerless. Trigger gold 2,263. Argument gold 4,198. Arguments per instance, mean 1.39. Trigger length in whitespace tokens, mean 0.00, single-token share 0.0%.

#### train event types

| label | count | share |
| --- | --- | --- |
| Cognitive.IdentifyCategorize.Unspecified | 0 | n/a |
| Cognitive.Inspection.SensoryObserve | 0 | n/a |
| Conflict.Attack.Unspecified | 0 | n/a |
| Life.Injure.Unspecified | 0 | n/a |
| Conflict.Attack.DetonateExplode | 0 | n/a |
| Justice.ChargeIndict.Unspecified | 0 | n/a |
| Justice.ArrestJailDetain.Unspecified | 0 | n/a |
| Justice.Convict.Unspecified | 0 | n/a |
| Justice.InvestigateCrime.Unspecified | 0 | n/a |
| Contact.Contact.Unspecified | 0 | n/a |
| Life.Die.Unspecified | 0 | n/a |
| ArtifactExistence.DamageDestroyDisableDismantle.Damage | 0 | n/a |
| Control.ImpedeInterfereWith.Unspecified | 0 | n/a |
| GenericCrime.GenericCrime.GenericCrime | 0 | n/a |
| Movement.Transportation.Unspecified | 0 | n/a |
| Contact.Contact.Broadcast | 0 | n/a |
| ArtifactExistence.DamageDestroyDisableDismantle.Destroy | 0 | n/a |
| Medical.Intervention.Unspecified | 0 | n/a |
| Conflict.Demonstrate.DemonstrateWithViolence | 0 | n/a |
| Conflict.Demonstrate.Unspecified | 0 | n/a |
| Contact.ThreatenCoerce.Unspecified | 0 | n/a |
| Contact.RequestCommand.Broadcast | 0 | n/a |
| Contact.Contact.Meet | 0 | n/a |
| Movement.Transportation.Evacuation | 0 | n/a |
| Justice.Acquit.Unspecified | 0 | n/a |
| ArtifactExistence.ManufactureAssemble.Unspecified | 0 | n/a |
| ArtifactExistence.DamageDestroyDisableDismantle.Dismantle | 0 | n/a |
| Justice.Sentence.Unspecified | 0 | n/a |
| Justice.TrialHearing.Unspecified | 0 | n/a |
| Transaction.ExchangeBuySell.Unspecified | 0 | n/a |
| Movement.Transportation.PreventPassage | 0 | n/a |
| Contact.Contact.Correspondence | 0 | n/a |
| Contact.ThreatenCoerce.Broadcast | 0 | n/a |
| Contact.RequestCommand.Unspecified | 0 | n/a |
| Conflict.Defeat.Unspecified | 0 | n/a |
| Life.Infect.Unspecified | 0 | n/a |
| Cognitive.Research.Unspecified | 0 | n/a |
| Disaster.Crash.Unspecified | 0 | n/a |
| ArtifactExistence.DamageDestroyDisableDismantle.Unspecified | 0 | n/a |
| Movement.Transportation.IllegalTransportation | 0 | n/a |
| Contact.ThreatenCoerce.Correspondence | 0 | n/a |
| Personnel.EndPosition.Unspecified | 0 | n/a |
| ArtifactExistence.DamageDestroyDisableDismantle.DisableDefuse | 0 | n/a |
| Personnel.StartPosition.Unspecified | 0 | n/a |
| Cognitive.TeachingTrainingLearning.Unspecified | 0 | n/a |
| Justice.ReleaseParole.Unspecified | 0 | n/a |
| Disaster.DiseaseOutbreak.Unspecified | 0 | n/a |
| Contact.RequestCommand.Meet | 0 | n/a |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 4 | 4,435 | 0.09% | 4 | 0 | 0 | 0 | 4,431 | 0 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

**train by role.**

| role | inside | placed | inside share | unmeasured |
| --- | --- | --- | --- | --- |
| Victim | 1 | 712 | 0.14% | 0 |
| Place | 2 | 538 | 0.37% | 0 |
| Target | 0 | 344 | 0.00% | 0 |
| Participant | 0 | 292 | 0.00% | 0 |
| Attacker | 0 | 290 | 0.00% | 0 |
| Defendant | 0 | 234 | 0.00% | 0 |
| ExplosiveDevice | 1 | 193 | 0.52% | 0 |
| Communicator | 0 | 171 | 0.00% | 0 |
| Detainee | 0 | 164 | 0.00% | 0 |
| Artifact | 0 | 160 | 0.00% | 0 |
| PassengerArtifact | 0 | 122 | 0.00% | 0 |
| Destination | 0 | 114 | 0.00% | 0 |
| Instrument | 0 | 109 | 0.00% | 0 |
| Killer | 0 | 108 | 0.00% | 0 |
| Recipient | 0 | 71 | 0.00% | 0 |
| Investigator | 0 | 70 | 0.00% | 0 |
| Topic | 0 | 69 | 0.00% | 0 |
| Components | 0 | 63 | 0.00% | 0 |
| Vehicle | 0 | 47 | 0.00% | 0 |
| Jailer | 0 | 45 | 0.00% | 0 |
| JudgeCourt | 0 | 42 | 0.00% | 0 |
| Transporter | 0 | 41 | 0.00% | 0 |
| Origin | 0 | 35 | 0.00% | 0 |
| IdentifiedObject | 0 | 34 | 0.00% | 0 |
| Injurer | 0 | 32 | 0.00% | 0 |
| Identifier | 0 | 29 | 0.00% | 0 |
| AcquiredEntity | 0 | 28 | 0.00% | 0 |
| IdentifiedRole | 0 | 27 | 0.00% | 0 |
| Perpetrator | 0 | 23 | 0.00% | 0 |
| Prosecutor | 0 | 23 | 0.00% | 0 |
| ObservedEntity | 0 | 22 | 0.00% | 0 |
| Observer | 0 | 21 | 0.00% | 0 |
| ManufacturerAssembler | 0 | 20 | 0.00% | 0 |
| Patient | 0 | 20 | 0.00% | 0 |
| Demonstrator | 0 | 17 | 0.00% | 0 |
| Giver | 0 | 14 | 0.00% | 0 |
| PaymentBarter | 0 | 9 | 0.00% | 0 |
| Researcher | 0 | 9 | 0.00% | 0 |
| BodyPart | 0 | 8 | 0.00% | 0 |
| Destroyer | 0 | 8 | 0.00% | 0 |
| Subject | 0 | 8 | 0.00% | 0 |
| Treater | 0 | 7 | 0.00% | 0 |
| Victor | 0 | 6 | 0.00% | 0 |
| Employee | 0 | 5 | 0.00% | 0 |
| Impeder | 0 | 5 | 0.00% | 0 |
| Defeated | 0 | 4 | 0.00% | 0 |
| Learner | 0 | 4 | 0.00% | 0 |
| Preventer | 0 | 4 | 0.00% | 0 |
| CrashObject | 0 | 3 | 0.00% | 0 |
| PlaceOfEmployment | 0 | 3 | 0.00% | 0 |
| DamagerDestroyer | 0 | 2 | 0.00% | 0 |
| TeacherTrainer | 0 | 2 | 0.00% | 0 |
| Damager | 0 | 1 | 0.00% | 0 |
| Disabler | 0 | 1 | 0.00% | 0 |
| Disease | 0 | 1 | 0.00% | 0 |
| Dismantler | 0 | 1 | 0.00% | 0 |

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3.30 | 56.39% | 4.54 | 64.41% | 0.31% | 4,435 | 0 | 3,191 | 0 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3,191 | 0.00% | 8,292 | 10.94% | 10.14% | 7.25% | 4.40% | 70.16% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

| split | docs | clusters | mentions | multi-mention | singletons | mention pairs | pronoun mentions | pronoun share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 194 | 4,609 | 20,941 | 4,609 | 0 | 124,879 | 4,500 | 21.49% |

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 0 | 0 | 8,292 |

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

| split | mentions the string rule adds | arguments skipped, 2+ clusters | arguments in one multi-mention cluster | other mentions of that cluster | argument spans in 2+ clusters |
| --- | --- | --- | --- | --- | --- |
| train | 17,352 | 656 | 3,073 | 31,210 | 0 |

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
