# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rams_merged | train | 6,322 | 6,322 | 0 | 0 | 0 | 0 | 6,864 | 16,555 |

## rams_merged

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 6,322 | 6,322 | 133 | 195 | 557 | 0 | 0 | 0 | 0 | 6,864 | 16,555 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | records | share |
| --- | --- | --- | --- |
| eng | 6,320 | 6,320 | 99.97% |
| und | 2 | 2 | 0.03% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| train_records | 6,322 |
| train_unique_inputs | 6,322 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

No label marginal has gold on both sides of a split pair.


### Events

**train.** 6,864 instances in 6,322 records, 0 triggerless. Trigger gold 6,864. Argument gold 16,555. Arguments per instance, mean 2.41. Trigger length in whitespace tokens, mean 1.04, single-token share 96.7%.

#### train event types

| label | count | share |
| --- | --- | --- |
| life.die.deathcausedbyviolentevents | 278 | 4.05% |
| conflict.attack.selfdirectedbattle | 234 | 3.41% |
| life.injure.injurycausedbyviolentevents | 206 | 3.00% |
| conflict.attack.n/a | 194 | 2.83% |
| transaction.transfermoney.n/a | 135 | 1.97% |
| contact.discussion.meet | 129 | 1.88% |
| movement.transportperson.preventexit | 128 | 1.86% |
| justice.arrestjaildetain.arrestjaildetain | 127 | 1.85% |
| life.die.n/a | 127 | 1.85% |
| life.injure.n/a | 107 | 1.56% |
| transaction.transfermoney.payforservice | 101 | 1.47% |
| movement.transportperson.n/a | 100 | 1.46% |
| transaction.transferownership.n/a | 83 | 1.21% |
| contact.requestadvise.broadcast | 79 | 1.15% |
| transaction.transfermoney.purchase | 76 | 1.11% |
| contact.commitmentpromiseexpressintent.meet | 75 | 1.09% |
| transaction.transferownership.purchase | 75 | 1.09% |
| movement.transportartifact.n/a | 73 | 1.06% |
| conflict.attack.bombing | 73 | 1.06% |
| transaction.transaction.embargosanction | 72 | 1.05% |
| artifactexistence.damagedestroy.damage | 71 | 1.03% |
| contact.discussion.correspondence | 69 | 1.01% |
| government.agreements.acceptagreementcontractceasefire | 68 | 0.99% |
| contact.requestadvise.correspondence | 67 | 0.98% |
| government.legislate.legislate | 67 | 0.98% |
| personnel.startposition.hiring | 67 | 0.98% |
| contact.publicstatementinperson.broadcast | 65 | 0.95% |
| movement.transportperson.evacuationrescue | 64 | 0.93% |
| contact.discussion.n/a | 63 | 0.92% |
| contact.commitmentpromiseexpressintent.broadcast | 62 | 0.90% |
| transaction.transfermoney.giftgrantprovideaid | 62 | 0.90% |
| life.injure.illnessdegradationphysical | 61 | 0.89% |
| personnel.endposition.quitretire | 61 | 0.89% |
| transaction.transferownership.embargosanction | 61 | 0.89% |
| transaction.transaction.transfercontrol | 59 | 0.86% |
| transaction.transfermoney.embargosanction | 59 | 0.86% |
| justice.initiatejudicialprocess.n/a | 58 | 0.84% |
| disaster.fireexplosion.fireexplosion | 58 | 0.84% |
| conflict.attack.stealrobhijack | 57 | 0.83% |
| justice.judicialconsequences.n/a | 57 | 0.83% |
| contact.requestadvise.meet | 56 | 0.82% |
| justice.initiatejudicialprocess.chargeindict | 56 | 0.82% |
| contact.requestadvise.n/a | 56 | 0.82% |
| movement.transportartifact.preventexit | 55 | 0.80% |
| transaction.transferownership.giftgrantprovideaid | 55 | 0.80% |
| transaction.transfermoney.borrowlend | 55 | 0.80% |
| transaction.transferownership.borrowlend | 54 | 0.79% |
| personnel.endposition.n/a | 53 | 0.77% |
| life.injure.illnessdegradationhungerthirst | 53 | 0.77% |
| artifactexistence.damagedestroy.destroy | 53 | 0.77% |
| justice.initiatejudicialprocess.trialhearing | 51 | 0.74% |
| transaction.transaction.giftgrantprovideaid | 50 | 0.73% |
| justice.judicialconsequences.execute | 50 | 0.73% |
| contact.collaborate.correspondence | 49 | 0.71% |
| justice.judicialconsequences.extradite | 49 | 0.71% |
| personnel.endposition.firinglayoff | 49 | 0.71% |
| government.agreements.violateagreement | 47 | 0.68% |
| contact.commitmentpromiseexpressintent.correspondence | 47 | 0.68% |
| inspection.sensoryobserve.physicalinvestigateinspect | 47 | 0.68% |
| conflict.attack.stabbing | 45 | 0.66% |
| personnel.elect.winelection | 44 | 0.64% |
| disaster.accidentcrash.accidentcrash | 44 | 0.64% |
| contact.publicstatementinperson.n/a | 43 | 0.63% |
| life.die.nonviolentdeath | 43 | 0.63% |
| artifactexistence.damagedestroy.n/a | 43 | 0.63% |
| contact.prevarication.broadcast | 42 | 0.61% |
| justice.judicialconsequences.convict | 42 | 0.61% |
| contact.threatencoerce.meet | 41 | 0.60% |
| justice.investigate.investigatecrime | 40 | 0.58% |
| conflict.attack.airstrikemissilestrike | 39 | 0.57% |
| contact.negotiate.meet | 38 | 0.55% |
| government.agreements.n/a | 38 | 0.55% |
| contact.threatencoerce.broadcast | 37 | 0.54% |
| inspection.sensoryobserve.inspectpeopleorganization | 37 | 0.54% |
| movement.transportperson.prevententry | 36 | 0.52% |
| contact.commandorder.broadcast | 36 | 0.52% |
| contact.mediastatement.broadcast | 36 | 0.52% |
| movement.transportartifact.smuggleextract | 36 | 0.52% |
| movement.transportartifact.bringcarryunload | 34 | 0.50% |
| contact.threatencoerce.correspondence | 34 | 0.50% |
| personnel.startposition.n/a | 34 | 0.50% |
| manufacture.artifact.n/a | 34 | 0.50% |
| movement.transportperson.selfmotion | 34 | 0.50% |
| manufacture.artifact.build | 33 | 0.48% |
| inspection.sensoryobserve.n/a | 33 | 0.48% |
| government.spy.spy | 33 | 0.48% |
| contact.prevarication.correspondence | 33 | 0.48% |
| movement.transportperson.grantentryasylum | 33 | 0.48% |
| movement.transportperson.smuggleextract | 33 | 0.48% |
| contact.prevarication.meet | 33 | 0.48% |
| conflict.attack.invade | 33 | 0.48% |
| movement.transportartifact.disperseseparate | 31 | 0.45% |
| contact.funeralvigil.meet | 31 | 0.45% |
| movement.transportartifact.hide | 30 | 0.44% |
| government.vote.castvote | 30 | 0.44% |
| transaction.transaction.n/a | 30 | 0.44% |
| movement.transportartifact.sendsupplyexport | 29 | 0.42% |
| contact.commandorder.n/a | 29 | 0.42% |
| justice.investigate.n/a | 28 | 0.41% |
| government.agreements.rejectnullifyagreementcontractceasefire | 28 | 0.41% |
| movement.transportartifact.prevententry | 27 | 0.39% |
| conflict.attack.firearmattack | 27 | 0.39% |
| contact.commitmentpromiseexpressintent.n/a | 25 | 0.36% |
| contact.commandorder.correspondence | 25 | 0.36% |
| contact.collaborate.meet | 25 | 0.36% |
| conflict.yield.n/a | 24 | 0.35% |
| conflict.demonstrate.marchprotestpoliticalgathering | 23 | 0.34% |
| contact.threatencoerce.n/a | 22 | 0.32% |
| movement.transportperson.disperseseparate | 22 | 0.32% |
| conflict.demonstrate.n/a | 22 | 0.32% |
| contact.prevarication.n/a | 22 | 0.32% |
| movement.transportartifact.grantentry | 22 | 0.32% |
| contact.mediastatement.n/a | 22 | 0.32% |
| movement.transportperson.hide | 21 | 0.31% |
| government.formation.mergegpe | 20 | 0.29% |
| movement.transportartifact.receiveimport | 20 | 0.29% |
| movement.transportartifact.fall | 19 | 0.28% |
| contact.negotiate.correspondence | 19 | 0.28% |
| contact.commandorder.meet | 19 | 0.28% |
| contact.collaborate.n/a | 19 | 0.28% |
| government.vote.violationspreventvote | 18 | 0.26% |
| conflict.attack.setfire | 18 | 0.26% |
| manufacture.artifact.createmanufacture | 18 | 0.26% |
| inspection.sensoryobserve.monitorelection | 17 | 0.25% |
| movement.transportartifact.nonviolentthrowlaunch | 16 | 0.23% |
| government.formation.n/a | 16 | 0.23% |
| conflict.yield.surrender | 16 | 0.23% |
| manufacture.artifact.createintellectualproperty | 16 | 0.23% |
| government.formation.startgpe | 15 | 0.22% |
| conflict.yield.retreat | 14 | 0.20% |
| conflict.attack.biologicalchemicalpoisonattack | 12 | 0.17% |
| movement.transportperson.bringcarryunload | 11 | 0.16% |
| movement.transportperson.fall | 11 | 0.16% |
| contact.negotiate.n/a | 10 | 0.15% |
| contact.funeralvigil.n/a | 10 | 0.15% |
| personnel.elect.n/a | 9 | 0.13% |
| conflict.attack.hanging | 8 | 0.12% |
| government.vote.n/a | 6 | 0.09% |
| conflict.attack.strangling | 5 | 0.07% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 16,555 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.20 | 14.09% | 1.56 | 31.45% | 0.00% | 0 | 16,555 | 0 | 6,864 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 6,864 | 0.00% | 14 | 57.14% | 50.00% | 21.43% | 7.14% | 0.13% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

| split | events with > 1 trigger | positive pairs | same-type pairs across events |
| --- | --- | --- | --- |
| train | 0 | 0 | 14 |

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

| split | mentions the string rule adds | arguments skipped, 2+ clusters | arguments in one multi-mention cluster | other mentions of that cluster | argument spans in 2+ clusters |
| --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | 0 | 0 | 0 |

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
