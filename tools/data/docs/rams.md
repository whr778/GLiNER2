# Dataset census

Gold the training and scoring path sees. Label counts are after the label map when one was applied.

Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.

| dataset | split | records | unique inputs | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rams | train | 7,329 | 6,322 | 0 | 0 | 0 | 0 | 7,329 | 16,991 |
| rams | val | 439 | 385 | 0 | 0 | 0 | 0 | 439 | 1,004 |
| rams | dev | 924 | 800 | 0 | 0 | 0 | 0 | 924 | 2,182 |
| rams | test | 432 | 385 | 0 | 0 | 0 | 0 | 432 | 1,012 |

## rams

### Splits

Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.

| split | records | unique inputs | tokens median | tokens p90 | tokens max | entities | relations | classifications | structures | event triggers | event arguments |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 7,329 | 6,322 | 132 | 196 | 557 | 0 | 0 | 0 | 0 | 7,329 | 16,991 |
| val | 439 | 385 | 134 | 193 | 488 | 0 | 0 | 0 | 0 | 439 | 1,004 |
| dev | 924 | 800 | 129 | 188 | 416 | 0 | 0 | 0 | 0 | 924 | 2,182 |
| test | 432 | 385 | 128 | 186 | 319 | 0 | 0 | 0 | 0 | 432 | 1,012 |
### Language

One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code `eval_by_language` uses. `und` is undetermined. The count is records.

| language | train | val | dev | test | records | share |
| --- | --- | --- | --- | --- | --- | --- |
| eng | 7,327 | 439 | 924 | 432 | 9,122 | 99.98% |
| und | 2 | 0 | 0 | 0 | 2 | 0.02% |

### Cross-split uniqueness

The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.

| check | count |
| --- | --- |
| dev_records | 924 |
| dev_unique_inputs | 800 |
| test_records | 432 |
| test_unique_inputs | 385 |
| train_records | 7,329 |
| train_unique_inputs | 6,322 |
| val_records | 439 |
| val_unique_inputs | 385 |
| train&val | 0 |
| train&test | 0 |
| val&test | 0 |

### Label shift (TVD)

TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. 0 means the two splits match. 1 means they share no label. The data-health gates are TVD 0.15 (WARN) and 0.3 (STRONG). Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.

| category | splits | TVD | support | largest movers |
| --- | --- | --- | --- | --- |
| event types (per trigger mention) | train / val | 0.2131 WARN | 7,329 / 439 | conflict.attack.selfdirectedbattle 3%→2%, life.injure.injurycausedbyviolentevents 3%→2%, movement.transportperson.n/a 2%→3%, contact.funeralvigil.n/a 0%→1%, inspection.sensoryobserve.inspectpeopleorganization 1%→1% |
| event types (per trigger mention) | train / dev | 0.1817 WARN | 7,329 / 924 | conflict.attack.selfdirectedbattle 3%→4%, life.injure.n/a 2%→3%, life.die.deathcausedbyviolentevents 4%→5%, justice.arrestjaildetain.arrestjaildetain 2%→1%, movement.transportperson.n/a 2%→1% |
| event types (per trigger mention) | train / test | 0.2301 WARN | 7,329 / 432 | contact.requestadvise.meet 1%→2%, contact.prevarication.broadcast 1%→2%, contact.funeralvigil.meet 0%→1%, government.spy.spy 0%→1%, movement.transportperson.n/a 2%→1% |
| event types (per trigger mention) | val / test | 0.2692 WARN | 439 / 432 | movement.transportperson.n/a 3%→1%, conflict.attack.selfdirectedbattle 2%→3%, transaction.transferownership.purchase 2%→0%, justice.arrestjaildetain.arrestjaildetain 3%→1%, life.injure.injurycausedbyviolentevents 2%→3% |
| event types (per trigger mention) | dev / test | 0.2665 WARN | 924 / 432 | artifactexistence.damagedestroy.destroy 1%→0%, contact.funeralvigil.meet 0%→1%, transaction.transaction.embargosanction 2%→0%, conflict.attack.selfdirectedbattle 4%→3%, movement.transportperson.preventexit 1%→2% |
| event types (per trigger mention) | val / dev | 0.2746 WARN | 439 / 924 | conflict.attack.selfdirectedbattle 2%→4%, movement.transportperson.n/a 3%→1%, justice.arrestjaildetain.arrestjaildetain 3%→1%, life.injure.injurycausedbyviolentevents 2%→3%, life.injure.n/a 2%→3% |
| event roles (per argument) | train / val | 0.0927 | 17,026 / 1,006 | instrument 3%→2%, artifact 5%→6%, place 16%→15%, target 4%→3%, attacker 4%→2% |
| event roles (per argument) | train / dev | 0.0767 | 17,026 / 2,188 | recipient 9%→10%, transporter 3%→2%, place 16%→17%, giver 5%→5%, origin 3%→2% |
| event roles (per argument) | train / test | 0.0779 | 17,026 / 1,017 | participant 6%→8%, beneficiary 4%→3%, vehicle 1%→1%, giver 5%→4%, communicator 6%→6% |
| event roles (per argument) | val / test | 0.1117 | 1,006 / 1,017 | place 15%→16%, giver 5%→4%, beneficiary 4%→3%, target 3%→4%, instrument 2%→3% |
| event roles (per argument) | dev / test | 0.1124 | 2,188 / 1,017 | participant 6%→8%, recipient 10%→9%, giver 5%→4%, beneficiary 4%→3%, transporter 2%→3% |
| event roles (per argument) | val / dev | 0.1316 | 1,006 / 2,188 | place 15%→17%, transporter 4%→2%, instrument 2%→4%, artifact 6%→5%, origin 3%→2% |

Thin support in val, event types (per trigger mention): 110 label(s) under 10 gold. government.formation.n/a ×1, conflict.demonstrate.marchprotestpoliticalgathering ×1, contact.prevarication.correspondence ×1, justice.investigate.n/a ×1, government.agreements.rejectnullifyagreementcontractceasefire ×1, life.die.nonviolentdeath ×1, conflict.attack.airstrikemissilestrike ×1, movement.transportartifact.grantentry ×1, manufacture.artifact.createmanufacture ×1, contact.funeralvigil.meet ×1, contact.commandorder.n/a ×1, contact.commandorder.meet ×1, contact.threatencoerce.n/a ×1, movement.transportperson.fall ×1, contact.mediastatement.broadcast ×1, conflict.yield.surrender ×1, conflict.yield.retreat ×1, manufacture.artifact.build ×1, contact.prevarication.meet ×2, contact.commandorder.broadcast ×2, conflict.yield.n/a ×2, contact.threatencoerce.meet ×2, conflict.attack.biologicalchemicalpoisonattack ×2, personnel.elect.winelection ×2, government.agreements.acceptagreementcontractceasefire ×2, movement.transportperson.grantentryasylum ×2, contact.commitmentpromiseexpressintent.correspondence ×2, government.vote.n/a ×2, movement.transportartifact.bringcarryunload ×2, transaction.transfermoney.giftgrantprovideaid ×2, contact.negotiate.correspondence ×2, contact.commandorder.correspondence ×2, personnel.endposition.quitretire ×2, movement.transportartifact.smuggleextract ×2, movement.transportartifact.fall ×2, conflict.attack.stabbing ×2, contact.prevarication.n/a ×2, life.injure.illnessdegradationphysical ×2, movement.transportperson.selfmotion ×2, movement.transportartifact.disperseseparate ×2, transaction.transaction.giftgrantprovideaid ×2, conflict.attack.stealrobhijack ×2, movement.transportperson.hide ×3, contact.publicstatementinperson.broadcast ×3, contact.commitmentpromiseexpressintent.meet ×3, movement.transportperson.evacuationrescue ×3, transaction.transferownership.borrowlend ×3, justice.judicialconsequences.extradite ×3, disaster.accidentcrash.accidentcrash ×3, contact.negotiate.meet ×3, movement.transportperson.smuggleextract ×3, contact.threatencoerce.correspondence ×3, transaction.transaction.embargosanction ×3, government.spy.spy ×3, movement.transportartifact.receiveimport ×3, life.injure.illnessdegradationhungerthirst ×3, contact.discussion.correspondence ×3, contact.commitmentpromiseexpressintent.n/a ×3, contact.threatencoerce.broadcast ×3, contact.requestadvise.correspondence ×3, contact.requestadvise.broadcast ×3, movement.transportartifact.sendsupplyexport ×4, justice.initiatejudicialprocess.trialhearing ×4, manufacture.artifact.n/a ×4, contact.publicstatementinperson.n/a ×4, conflict.attack.bombing ×4, personnel.endposition.firinglayoff ×4, government.formation.mergegpe ×4, contact.requestadvise.n/a ×4, contact.mediastatement.n/a ×4, personnel.endposition.n/a ×4, justice.judicialconsequences.execute ×4, contact.requestadvise.meet ×4, movement.transportartifact.hide ×4, contact.commitmentpromiseexpressintent.broadcast ×4, contact.prevarication.broadcast ×4, inspection.sensoryobserve.n/a ×4, transaction.transfermoney.borrowlend ×4, artifactexistence.damagedestroy.damage ×4, justice.investigate.investigatecrime ×4, personnel.startposition.hiring ×4, personnel.startposition.n/a ×4, justice.initiatejudicialprocess.chargeindict ×4, inspection.sensoryobserve.physicalinvestigateinspect ×5, contact.funeralvigil.n/a ×5, transaction.transferownership.giftgrantprovideaid ×5, artifactexistence.damagedestroy.n/a ×5, transaction.transferownership.embargosanction ×5, transaction.transaction.n/a ×5, artifactexistence.damagedestroy.destroy ×5, inspection.sensoryobserve.inspectpeopleorganization ×6, contact.collaborate.correspondence ×6, transaction.transfermoney.embargosanction ×6, contact.discussion.n/a ×6, government.legislate.legislate ×6, movement.transportartifact.preventexit ×6, transaction.transaction.transfercontrol ×6, justice.judicialconsequences.n/a ×6, movement.transportartifact.n/a ×7, transaction.transferownership.n/a ×7, transaction.transfermoney.n/a ×7, contact.discussion.meet ×7, life.injure.injurycausedbyviolentevents ×7, transaction.transfermoney.payforservice ×7, transaction.transfermoney.purchase ×8, conflict.attack.selfdirectedbattle ×8, life.injure.n/a ×8, movement.transportperson.preventexit ×8, transaction.transferownership.purchase ×8, life.die.n/a ×9.
Thin support in val, event roles (per argument): 38 label(s) under 10 gold. driverpassenger ×1, gpe ×1, founder ×1, demonstrator ×1, executioner ×1, artifactmoney ×1, yielder ×1, rejecternullifier ×1, otherparticipant ×1, ballot ×1, result ×1, territoryorfacility ×1, surrenderer ×1, retreater ×1, crashobject ×2, voter ×2, granter ×2, hidingplace ×3, extraditer ×3, spy ×3, observer ×3, damagerdestroyer ×3, prosecutor ×4, deceased ×4, candidate ×4, investigator ×4, damager ×4, destroyer ×4, inspector ×5, manufacturer ×5, governmentbody ×5, jailer ×6, judgecourt ×6, law ×6, observedentity ×6, vehicle ×7, inspectedentity ×8, money ×9.
Thin support in dev, event types (per trigger mention): 107 label(s) under 10 gold. contact.negotiate.n/a ×1, inspection.sensoryobserve.n/a ×1, movement.transportartifact.receiveimport ×1, manufacture.artifact.createintellectualproperty ×1, contact.funeralvigil.meet ×1, conflict.attack.invade ×1, movement.transportartifact.nonviolentthrowlaunch ×1, government.formation.n/a ×1, conflict.attack.setfire ×1, movement.transportartifact.disperseseparate ×1, conflict.attack.stabbing ×1, movement.transportartifact.fall ×1, government.formation.startgpe ×1, contact.funeralvigil.n/a ×1, movement.transportperson.hide ×1, contact.commandorder.correspondence ×1, government.agreements.violateagreement ×2, movement.transportartifact.grantentry ×2, conflict.attack.firearmattack ×2, conflict.demonstrate.marchprotestpoliticalgathering ×2, contact.negotiate.correspondence ×2, movement.transportperson.smuggleextract ×2, justice.judicialconsequences.extradite ×2, contact.commitmentpromiseexpressintent.n/a ×2, government.vote.violationspreventvote ×2, justice.investigate.n/a ×2, movement.transportartifact.hide ×2, personnel.elect.winelection ×3, movement.transportartifact.bringcarryunload ×3, disaster.accidentcrash.accidentcrash ×3, contact.commandorder.meet ×3, personnel.endposition.quitretire ×3, manufacture.artifact.build ×3, contact.mediastatement.n/a ×3, conflict.yield.retreat ×3, contact.threatencoerce.meet ×3, movement.transportperson.disperseseparate ×3, disaster.fireexplosion.fireexplosion ×3, movement.transportartifact.prevententry ×3, personnel.elect.n/a ×3, movement.transportartifact.smuggleextract ×3, conflict.attack.biologicalchemicalpoisonattack ×3, government.agreements.n/a ×4, contact.threatencoerce.broadcast ×4, justice.initiatejudicialprocess.n/a ×4, transaction.transaction.n/a ×4, inspection.sensoryobserve.physicalinvestigateinspect ×4, government.vote.castvote ×4, conflict.attack.airstrikemissilestrike ×4, contact.discussion.n/a ×4, personnel.startposition.n/a ×4, inspection.sensoryobserve.monitorelection ×4, contact.prevarication.n/a ×4, contact.commandorder.n/a ×4, transaction.transfermoney.borrowlend ×4, movement.transportperson.grantentryasylum ×4, government.vote.n/a ×5, conflict.yield.surrender ×5, contact.threatencoerce.correspondence ×5, movement.transportartifact.preventexit ×5, contact.commitmentpromiseexpressintent.broadcast ×5, contact.publicstatementinperson.n/a ×5, conflict.attack.stealrobhijack ×5, movement.transportperson.prevententry ×5, government.agreements.acceptagreementcontractceasefire ×6, justice.judicialconsequences.n/a ×6, manufacture.artifact.createmanufacture ×6, contact.mediastatement.broadcast ×6, life.injure.illnessdegradationphysical ×6, contact.commitmentpromiseexpressintent.correspondence ×6, contact.threatencoerce.n/a ×6, contact.commandorder.broadcast ×6, transaction.transaction.giftgrantprovideaid ×6, justice.judicialconsequences.execute ×6, contact.requestadvise.n/a ×6, contact.publicstatementinperson.broadcast ×6, justice.initiatejudicialprocess.chargeindict ×6, inspection.sensoryobserve.inspectpeopleorganization ×6, manufacture.artifact.n/a ×6, movement.transportartifact.sendsupplyexport ×6, conflict.yield.n/a ×7, conflict.attack.bombing ×7, contact.collaborate.correspondence ×7, transaction.transaction.transfercontrol ×7, artifactexistence.damagedestroy.damage ×7, transaction.transferownership.giftgrantprovideaid ×7, contact.requestadvise.meet ×7, movement.transportperson.selfmotion ×8, personnel.endposition.n/a ×8, contact.prevarication.broadcast ×8, transaction.transfermoney.giftgrantprovideaid ×8, justice.judicialconsequences.convict ×8, justice.initiatejudicialprocess.trialhearing ×8, movement.transportartifact.n/a ×8, contact.negotiate.meet ×8, life.injure.illnessdegradationhungerthirst ×8, government.legislate.legislate ×8, artifactexistence.damagedestroy.n/a ×8, justice.investigate.investigatecrime ×8, transaction.transferownership.borrowlend ×8, justice.arrestjaildetain.arrestjaildetain ×8, movement.transportperson.preventexit ×9, contact.prevarication.correspondence ×9, movement.transportperson.n/a ×9, movement.transportperson.evacuationrescue ×9, contact.collaborate.n/a ×9, transaction.transfermoney.embargosanction ×9.
Thin support in dev, event roles (per argument): 34 label(s) under 10 gold. ballot ×1, gpe ×1, driverpassenger ×2, crashobject ×2, result ×2, violator ×2, otherparticipant ×2, demonstrator ×2, deceased ×2, founder ×2, extraditer ×2, retreater ×3, executioner ×3, fireexplosionobject ×3, monitor ×3, monitoredentity ×3, law ×4, granter ×4, surrenderer ×5, territoryorfacility ×5, jailer ×5, artifactmoney ×6, inspectedentity ×6, yielder ×7, vehicle ×7, inspector ×7, damager ×7, governmentbody ×7, voter ×8, damagerdestroyer ×8, detainee ×8, prosecutor ×9, candidate ×9, observedentity ×9.
Thin support in test, event types (per trigger mention): 114 label(s) under 10 gold. conflict.yield.retreat ×1, contact.collaborate.correspondence ×1, movement.transportperson.grantentryasylum ×1, conflict.attack.bombing ×1, government.vote.castvote ×1, inspection.sensoryobserve.n/a ×1, contact.negotiate.n/a ×1, conflict.yield.surrender ×1, movement.transportartifact.grantentry ×1, contact.mediastatement.n/a ×1, government.agreements.violateagreement ×1, contact.collaborate.meet ×1, contact.commandorder.broadcast ×1, government.agreements.rejectnullifyagreementcontractceasefire ×1, disaster.fireexplosion.fireexplosion ×1, justice.initiatejudicialprocess.trialhearing ×1, contact.requestadvise.broadcast ×1, transaction.transferownership.borrowlend ×1, life.die.nonviolentdeath ×1, contact.commitmentpromiseexpressintent.n/a ×1, personnel.endposition.n/a ×2, personnel.elect.n/a ×2, contact.funeralvigil.n/a ×2, contact.threatencoerce.meet ×2, movement.transportperson.selfmotion ×2, transaction.transaction.n/a ×2, contact.negotiate.correspondence ×2, contact.threatencoerce.broadcast ×2, transaction.transaction.embargosanction ×2, conflict.attack.airstrikemissilestrike ×2, personnel.startposition.hiring ×2, contact.commitmentpromiseexpressintent.meet ×2, government.formation.startgpe ×2, conflict.attack.biologicalchemicalpoisonattack ×2, contact.prevarication.n/a ×2, inspection.sensoryobserve.inspectpeopleorganization ×2, movement.transportartifact.fall ×2, contact.publicstatementinperson.broadcast ×2, conflict.demonstrate.n/a ×2, conflict.attack.hanging ×2, personnel.elect.winelection ×2, transaction.transferownership.purchase ×2, transaction.transferownership.giftgrantprovideaid ×2, justice.judicialconsequences.extradite ×2, conflict.attack.stabbing ×2, conflict.attack.invade ×2, life.injure.illnessdegradationhungerthirst ×2, movement.transportartifact.bringcarryunload ×2, contact.threatencoerce.n/a ×2, government.formation.n/a ×2, movement.transportperson.evacuationrescue ×2, contact.mediastatement.broadcast ×3, conflict.attack.stealrobhijack ×3, movement.transportartifact.disperseseparate ×3, contact.commandorder.correspondence ×3, contact.commandorder.meet ×3, personnel.endposition.quitretire ×3, contact.discussion.correspondence ×3, transaction.transaction.giftgrantprovideaid ×3, manufacture.artifact.build ×3, conflict.yield.n/a ×3, contact.prevarication.correspondence ×3, movement.transportartifact.sendsupplyexport ×3, government.formation.mergegpe ×3, contact.threatencoerce.correspondence ×3, movement.transportartifact.hide ×3, justice.judicialconsequences.n/a ×3, movement.transportperson.disperseseparate ×3, contact.requestadvise.n/a ×3, movement.transportperson.bringcarryunload ×3, transaction.transfermoney.borrowlend ×3, justice.investigate.investigatecrime ×3, contact.collaborate.n/a ×3, transaction.transfermoney.giftgrantprovideaid ×3, transaction.transfermoney.embargosanction ×3, movement.transportperson.smuggleextract ×3, artifactexistence.damagedestroy.n/a ×4, justice.initiatejudicialprocess.n/a ×4, contact.commitmentpromiseexpressintent.broadcast ×4, government.legislate.legislate ×4, inspection.sensoryobserve.physicalinvestigateinspect ×4, contact.discussion.n/a ×4, conflict.demonstrate.marchprotestpoliticalgathering ×4, contact.prevarication.meet ×4, movement.transportperson.n/a ×4, transaction.transferownership.n/a ×4, life.injure.illnessdegradationphysical ×4, transaction.transfermoney.payforservice ×4, government.agreements.n/a ×4, contact.negotiate.meet ×4, artifactexistence.damagedestroy.damage ×5, justice.initiatejudicialprocess.chargeindict ×5, movement.transportartifact.smuggleextract ×5, contact.requestadvise.correspondence ×5, transaction.transaction.transfercontrol ×5, manufacture.artifact.n/a ×5, justice.arrestjaildetain.arrestjaildetain ×5, disaster.accidentcrash.accidentcrash ×5, personnel.endposition.firinglayoff ×5, movement.transportartifact.n/a ×5, government.agreements.acceptagreementcontractceasefire ×6, contact.publicstatementinperson.n/a ×6, contact.funeralvigil.meet ×6, movement.transportartifact.preventexit ×6, government.spy.spy ×6, justice.judicialconsequences.execute ×6, transaction.transfermoney.n/a ×7, transaction.transferownership.embargosanction ×7, contact.prevarication.broadcast ×7, life.die.n/a ×8, contact.requestadvise.meet ×8, transaction.transfermoney.purchase ×8, movement.transportperson.preventexit ×9, contact.discussion.meet ×9.
Thin support in test, event roles (per argument): 35 label(s) under 10 gold. retreater ×1, crashobject ×1, granter ×1, observer ×1, voter ×1, surrenderer ×1, violator ×1, rejecternullifier ×1, fireexplosionobject ×1, territoryorfacility ×2, artifactmoney ×2, jailer ×2, yielder ×2, extraditer ×2, damagerdestroyer ×3, judgecourt ×3, candidate ×3, observedentity ×3, driverpassenger ×3, investigator ×3, damager ×4, governmentbody ×4, law ×4, demonstrator ×4, gpe ×4, founder ×4, executioner ×4, deceased ×5, inspector ×5, detainee ×5, manufacturer ×6, spy ×6, inspectedentity ×6, prosecutor ×8, placeofemployment ×9.

### Events

**train.** 7,329 instances in 7,329 records, 0 triggerless. Trigger gold 7,329. Argument gold 16,991. Arguments per instance, mean 2.32. Trigger length in whitespace tokens, mean 1.04, single-token share 96.1%.

#### train event types

| label | count | share |
| --- | --- | --- |
| life.die.deathcausedbyviolentevents | 280 | 3.82% |
| conflict.attack.n/a | 260 | 3.55% |
| conflict.attack.selfdirectedbattle | 234 | 3.19% |
| life.injure.injurycausedbyviolentevents | 207 | 2.82% |
| life.die.n/a | 197 | 2.69% |
| transaction.transfermoney.n/a | 160 | 2.18% |
| life.injure.n/a | 143 | 1.95% |
| movement.transportperson.n/a | 134 | 1.83% |
| contact.discussion.meet | 129 | 1.76% |
| movement.transportperson.preventexit | 128 | 1.75% |
| justice.arrestjaildetain.arrestjaildetain | 127 | 1.73% |
| movement.transportartifact.n/a | 104 | 1.42% |
| transaction.transfermoney.payforservice | 101 | 1.38% |
| transaction.transferownership.n/a | 94 | 1.28% |
| contact.requestadvise.broadcast | 79 | 1.08% |
| transaction.transfermoney.purchase | 76 | 1.04% |
| contact.commitmentpromiseexpressintent.meet | 75 | 1.02% |
| transaction.transferownership.purchase | 75 | 1.02% |
| transaction.transaction.embargosanction | 74 | 1.01% |
| contact.discussion.n/a | 73 | 1.00% |
| conflict.attack.bombing | 73 | 1.00% |
| artifactexistence.damagedestroy.damage | 71 | 0.97% |
| justice.judicialconsequences.n/a | 70 | 0.96% |
| contact.discussion.correspondence | 69 | 0.94% |
| government.legislate.legislate | 68 | 0.93% |
| government.agreements.acceptagreementcontractceasefire | 68 | 0.93% |
| artifactexistence.damagedestroy.n/a | 68 | 0.93% |
| contact.requestadvise.correspondence | 67 | 0.91% |
| personnel.startposition.hiring | 67 | 0.91% |
| personnel.endposition.n/a | 66 | 0.90% |
| contact.publicstatementinperson.broadcast | 65 | 0.89% |
| movement.transportperson.evacuationrescue | 64 | 0.87% |
| contact.commitmentpromiseexpressintent.broadcast | 62 | 0.85% |
| transaction.transfermoney.giftgrantprovideaid | 62 | 0.85% |
| life.injure.illnessdegradationphysical | 61 | 0.83% |
| contact.requestadvise.n/a | 61 | 0.83% |
| personnel.endposition.quitretire | 61 | 0.83% |
| transaction.transferownership.embargosanction | 61 | 0.83% |
| justice.initiatejudicialprocess.n/a | 60 | 0.82% |
| transaction.transaction.transfercontrol | 59 | 0.81% |
| transaction.transfermoney.embargosanction | 59 | 0.81% |
| disaster.fireexplosion.fireexplosion | 58 | 0.79% |
| conflict.attack.stealrobhijack | 57 | 0.78% |
| justice.initiatejudicialprocess.chargeindict | 56 | 0.76% |
| contact.requestadvise.meet | 56 | 0.76% |
| transaction.transfermoney.borrowlend | 55 | 0.75% |
| movement.transportartifact.preventexit | 55 | 0.75% |
| transaction.transferownership.giftgrantprovideaid | 55 | 0.75% |
| transaction.transferownership.borrowlend | 54 | 0.74% |
| life.injure.illnessdegradationhungerthirst | 53 | 0.72% |
| artifactexistence.damagedestroy.destroy | 53 | 0.72% |
| justice.initiatejudicialprocess.trialhearing | 51 | 0.70% |
| justice.judicialconsequences.execute | 50 | 0.68% |
| transaction.transaction.giftgrantprovideaid | 50 | 0.68% |
| justice.judicialconsequences.extradite | 49 | 0.67% |
| personnel.endposition.firinglayoff | 49 | 0.67% |
| contact.collaborate.correspondence | 49 | 0.67% |
| transaction.transaction.n/a | 48 | 0.65% |
| government.agreements.violateagreement | 47 | 0.64% |
| contact.commitmentpromiseexpressintent.correspondence | 47 | 0.64% |
| inspection.sensoryobserve.physicalinvestigateinspect | 47 | 0.64% |
| contact.publicstatementinperson.n/a | 46 | 0.63% |
| conflict.attack.stabbing | 45 | 0.61% |
| inspection.sensoryobserve.n/a | 44 | 0.60% |
| personnel.elect.winelection | 44 | 0.60% |
| life.die.nonviolentdeath | 44 | 0.60% |
| disaster.accidentcrash.accidentcrash | 44 | 0.60% |
| contact.prevarication.broadcast | 42 | 0.57% |
| justice.judicialconsequences.convict | 42 | 0.57% |
| contact.threatencoerce.meet | 41 | 0.56% |
| justice.investigate.investigatecrime | 40 | 0.55% |
| government.agreements.n/a | 40 | 0.55% |
| conflict.attack.airstrikemissilestrike | 39 | 0.53% |
| contact.negotiate.meet | 38 | 0.52% |
| manufacture.artifact.n/a | 38 | 0.52% |
| movement.transportartifact.smuggleextract | 37 | 0.50% |
| contact.threatencoerce.broadcast | 37 | 0.50% |
| inspection.sensoryobserve.inspectpeopleorganization | 37 | 0.50% |
| movement.transportperson.prevententry | 36 | 0.49% |
| contact.commandorder.broadcast | 36 | 0.49% |
| contact.mediastatement.broadcast | 36 | 0.49% |
| personnel.startposition.n/a | 35 | 0.48% |
| movement.transportartifact.bringcarryunload | 34 | 0.46% |
| movement.transportperson.grantentryasylum | 34 | 0.46% |
| conflict.attack.invade | 34 | 0.46% |
| contact.threatencoerce.correspondence | 34 | 0.46% |
| movement.transportperson.selfmotion | 34 | 0.46% |
| contact.collaborate.n/a | 33 | 0.45% |
| manufacture.artifact.build | 33 | 0.45% |
| government.spy.spy | 33 | 0.45% |
| contact.prevarication.correspondence | 33 | 0.45% |
| movement.transportperson.smuggleextract | 33 | 0.45% |
| contact.prevarication.meet | 33 | 0.45% |
| contact.threatencoerce.n/a | 32 | 0.44% |
| contact.commitmentpromiseexpressintent.n/a | 31 | 0.42% |
| movement.transportartifact.disperseseparate | 31 | 0.42% |
| contact.funeralvigil.meet | 31 | 0.42% |
| contact.commandorder.n/a | 31 | 0.42% |
| movement.transportartifact.hide | 30 | 0.41% |
| contact.prevarication.n/a | 30 | 0.41% |
| government.vote.castvote | 30 | 0.41% |
| movement.transportartifact.sendsupplyexport | 29 | 0.40% |
| justice.investigate.n/a | 28 | 0.38% |
| government.agreements.rejectnullifyagreementcontractceasefire | 28 | 0.38% |
| conflict.attack.firearmattack | 27 | 0.37% |
| movement.transportartifact.prevententry | 27 | 0.37% |
| contact.commandorder.correspondence | 25 | 0.34% |
| conflict.yield.n/a | 25 | 0.34% |
| contact.collaborate.meet | 25 | 0.34% |
| contact.mediastatement.n/a | 25 | 0.34% |
| conflict.demonstrate.marchprotestpoliticalgathering | 23 | 0.31% |
| movement.transportperson.disperseseparate | 22 | 0.30% |
| conflict.demonstrate.n/a | 22 | 0.30% |
| movement.transportartifact.grantentry | 22 | 0.30% |
| movement.transportperson.hide | 21 | 0.29% |
| personnel.elect.n/a | 20 | 0.27% |
| government.formation.mergegpe | 20 | 0.27% |
| movement.transportartifact.receiveimport | 20 | 0.27% |
| government.formation.n/a | 19 | 0.26% |
| contact.negotiate.correspondence | 19 | 0.26% |
| contact.commandorder.meet | 19 | 0.26% |
| movement.transportartifact.fall | 19 | 0.26% |
| government.vote.violationspreventvote | 18 | 0.25% |
| manufacture.artifact.createmanufacture | 18 | 0.25% |
| contact.funeralvigil.n/a | 18 | 0.25% |
| conflict.attack.setfire | 18 | 0.25% |
| inspection.sensoryobserve.monitorelection | 17 | 0.23% |
| movement.transportartifact.nonviolentthrowlaunch | 16 | 0.22% |
| conflict.yield.surrender | 16 | 0.22% |
| manufacture.artifact.createintellectualproperty | 16 | 0.22% |
| government.formation.startgpe | 15 | 0.20% |
| conflict.yield.retreat | 14 | 0.19% |
| contact.negotiate.n/a | 12 | 0.16% |
| conflict.attack.biologicalchemicalpoisonattack | 12 | 0.16% |
| government.vote.n/a | 11 | 0.15% |
| movement.transportperson.bringcarryunload | 11 | 0.15% |
| movement.transportperson.fall | 11 | 0.15% |
| conflict.attack.hanging | 10 | 0.14% |
| conflict.attack.strangling | 5 | 0.07% |

**val.** 439 instances in 439 records, 0 triggerless. Trigger gold 439. Argument gold 1,004. Arguments per instance, mean 2.29. Trigger length in whitespace tokens, mean 1.03, single-token share 97.9%.

#### val event types

| label | count | share |
| --- | --- | --- |
| life.die.deathcausedbyviolentevents | 18 | 4.10% |
| conflict.attack.n/a | 13 | 2.96% |
| movement.transportperson.n/a | 12 | 2.73% |
| justice.arrestjaildetain.arrestjaildetain | 11 | 2.51% |
| life.die.n/a | 9 | 2.05% |
| transaction.transfermoney.purchase | 8 | 1.82% |
| conflict.attack.selfdirectedbattle | 8 | 1.82% |
| life.injure.n/a | 8 | 1.82% |
| movement.transportperson.preventexit | 8 | 1.82% |
| transaction.transferownership.purchase | 8 | 1.82% |
| movement.transportartifact.n/a | 7 | 1.59% |
| transaction.transferownership.n/a | 7 | 1.59% |
| transaction.transfermoney.n/a | 7 | 1.59% |
| contact.discussion.meet | 7 | 1.59% |
| life.injure.injurycausedbyviolentevents | 7 | 1.59% |
| transaction.transfermoney.payforservice | 7 | 1.59% |
| inspection.sensoryobserve.inspectpeopleorganization | 6 | 1.37% |
| contact.collaborate.correspondence | 6 | 1.37% |
| transaction.transfermoney.embargosanction | 6 | 1.37% |
| contact.discussion.n/a | 6 | 1.37% |
| government.legislate.legislate | 6 | 1.37% |
| movement.transportartifact.preventexit | 6 | 1.37% |
| transaction.transaction.transfercontrol | 6 | 1.37% |
| justice.judicialconsequences.n/a | 6 | 1.37% |
| inspection.sensoryobserve.physicalinvestigateinspect | 5 | 1.14% |
| contact.funeralvigil.n/a | 5 | 1.14% |
| transaction.transferownership.giftgrantprovideaid | 5 | 1.14% |
| artifactexistence.damagedestroy.n/a | 5 | 1.14% |
| transaction.transferownership.embargosanction | 5 | 1.14% |
| transaction.transaction.n/a | 5 | 1.14% |
| artifactexistence.damagedestroy.destroy | 5 | 1.14% |
| movement.transportartifact.sendsupplyexport | 4 | 0.91% |
| justice.initiatejudicialprocess.trialhearing | 4 | 0.91% |
| manufacture.artifact.n/a | 4 | 0.91% |
| contact.publicstatementinperson.n/a | 4 | 0.91% |
| conflict.attack.bombing | 4 | 0.91% |
| personnel.endposition.firinglayoff | 4 | 0.91% |
| government.formation.mergegpe | 4 | 0.91% |
| contact.requestadvise.n/a | 4 | 0.91% |
| contact.mediastatement.n/a | 4 | 0.91% |
| personnel.endposition.n/a | 4 | 0.91% |
| justice.judicialconsequences.execute | 4 | 0.91% |
| contact.requestadvise.meet | 4 | 0.91% |
| movement.transportartifact.hide | 4 | 0.91% |
| contact.commitmentpromiseexpressintent.broadcast | 4 | 0.91% |
| contact.prevarication.broadcast | 4 | 0.91% |
| inspection.sensoryobserve.n/a | 4 | 0.91% |
| transaction.transfermoney.borrowlend | 4 | 0.91% |
| artifactexistence.damagedestroy.damage | 4 | 0.91% |
| justice.investigate.investigatecrime | 4 | 0.91% |
| personnel.startposition.hiring | 4 | 0.91% |
| personnel.startposition.n/a | 4 | 0.91% |
| justice.initiatejudicialprocess.chargeindict | 4 | 0.91% |
| movement.transportperson.hide | 3 | 0.68% |
| contact.publicstatementinperson.broadcast | 3 | 0.68% |
| contact.commitmentpromiseexpressintent.meet | 3 | 0.68% |
| movement.transportperson.evacuationrescue | 3 | 0.68% |
| transaction.transferownership.borrowlend | 3 | 0.68% |
| justice.judicialconsequences.extradite | 3 | 0.68% |
| disaster.accidentcrash.accidentcrash | 3 | 0.68% |
| contact.negotiate.meet | 3 | 0.68% |
| movement.transportperson.smuggleextract | 3 | 0.68% |
| contact.threatencoerce.correspondence | 3 | 0.68% |
| transaction.transaction.embargosanction | 3 | 0.68% |
| government.spy.spy | 3 | 0.68% |
| movement.transportartifact.receiveimport | 3 | 0.68% |
| life.injure.illnessdegradationhungerthirst | 3 | 0.68% |
| contact.discussion.correspondence | 3 | 0.68% |
| contact.commitmentpromiseexpressintent.n/a | 3 | 0.68% |
| contact.threatencoerce.broadcast | 3 | 0.68% |
| contact.requestadvise.correspondence | 3 | 0.68% |
| contact.requestadvise.broadcast | 3 | 0.68% |
| contact.prevarication.meet | 2 | 0.46% |
| contact.commandorder.broadcast | 2 | 0.46% |
| conflict.yield.n/a | 2 | 0.46% |
| contact.threatencoerce.meet | 2 | 0.46% |
| conflict.attack.biologicalchemicalpoisonattack | 2 | 0.46% |
| personnel.elect.winelection | 2 | 0.46% |
| government.agreements.acceptagreementcontractceasefire | 2 | 0.46% |
| movement.transportperson.grantentryasylum | 2 | 0.46% |
| contact.commitmentpromiseexpressintent.correspondence | 2 | 0.46% |
| government.vote.n/a | 2 | 0.46% |
| movement.transportartifact.bringcarryunload | 2 | 0.46% |
| transaction.transfermoney.giftgrantprovideaid | 2 | 0.46% |
| contact.negotiate.correspondence | 2 | 0.46% |
| contact.commandorder.correspondence | 2 | 0.46% |
| personnel.endposition.quitretire | 2 | 0.46% |
| movement.transportartifact.smuggleextract | 2 | 0.46% |
| movement.transportartifact.fall | 2 | 0.46% |
| conflict.attack.stabbing | 2 | 0.46% |
| contact.prevarication.n/a | 2 | 0.46% |
| life.injure.illnessdegradationphysical | 2 | 0.46% |
| movement.transportperson.selfmotion | 2 | 0.46% |
| movement.transportartifact.disperseseparate | 2 | 0.46% |
| transaction.transaction.giftgrantprovideaid | 2 | 0.46% |
| conflict.attack.stealrobhijack | 2 | 0.46% |
| government.formation.n/a | 1 | 0.23% |
| conflict.demonstrate.marchprotestpoliticalgathering | 1 | 0.23% |
| contact.prevarication.correspondence | 1 | 0.23% |
| justice.investigate.n/a | 1 | 0.23% |
| government.agreements.rejectnullifyagreementcontractceasefire | 1 | 0.23% |
| life.die.nonviolentdeath | 1 | 0.23% |
| conflict.attack.airstrikemissilestrike | 1 | 0.23% |
| movement.transportartifact.grantentry | 1 | 0.23% |
| manufacture.artifact.createmanufacture | 1 | 0.23% |
| contact.funeralvigil.meet | 1 | 0.23% |
| contact.commandorder.n/a | 1 | 0.23% |
| contact.commandorder.meet | 1 | 0.23% |
| contact.threatencoerce.n/a | 1 | 0.23% |
| movement.transportperson.fall | 1 | 0.23% |
| contact.mediastatement.broadcast | 1 | 0.23% |
| conflict.yield.surrender | 1 | 0.23% |
| conflict.yield.retreat | 1 | 0.23% |
| manufacture.artifact.build | 1 | 0.23% |

**dev.** 924 instances in 924 records, 0 triggerless. Trigger gold 924. Argument gold 2,182. Arguments per instance, mean 2.37. Trigger length in whitespace tokens, mean 1.02, single-token share 98.4%.

#### dev event types

| label | count | share |
| --- | --- | --- |
| life.die.deathcausedbyviolentevents | 45 | 4.87% |
| conflict.attack.selfdirectedbattle | 41 | 4.44% |
| conflict.attack.n/a | 37 | 4.00% |
| life.injure.n/a | 29 | 3.14% |
| life.injure.injurycausedbyviolentevents | 27 | 2.92% |
| life.die.n/a | 23 | 2.49% |
| contact.discussion.meet | 18 | 1.95% |
| transaction.transfermoney.n/a | 17 | 1.84% |
| transaction.transaction.embargosanction | 16 | 1.73% |
| transaction.transferownership.n/a | 15 | 1.62% |
| transaction.transferownership.embargosanction | 14 | 1.52% |
| transaction.transfermoney.purchase | 14 | 1.52% |
| transaction.transferownership.purchase | 13 | 1.41% |
| transaction.transfermoney.payforservice | 13 | 1.41% |
| artifactexistence.damagedestroy.destroy | 13 | 1.41% |
| contact.requestadvise.broadcast | 12 | 1.30% |
| contact.commitmentpromiseexpressintent.meet | 12 | 1.30% |
| personnel.endposition.firinglayoff | 12 | 1.30% |
| government.spy.spy | 12 | 1.30% |
| contact.discussion.correspondence | 11 | 1.19% |
| life.die.nonviolentdeath | 10 | 1.08% |
| personnel.startposition.hiring | 10 | 1.08% |
| contact.requestadvise.correspondence | 10 | 1.08% |
| contact.prevarication.meet | 10 | 1.08% |
| movement.transportperson.preventexit | 9 | 0.97% |
| contact.prevarication.correspondence | 9 | 0.97% |
| movement.transportperson.n/a | 9 | 0.97% |
| movement.transportperson.evacuationrescue | 9 | 0.97% |
| contact.collaborate.n/a | 9 | 0.97% |
| transaction.transfermoney.embargosanction | 9 | 0.97% |
| movement.transportperson.selfmotion | 8 | 0.87% |
| personnel.endposition.n/a | 8 | 0.87% |
| contact.prevarication.broadcast | 8 | 0.87% |
| transaction.transfermoney.giftgrantprovideaid | 8 | 0.87% |
| justice.judicialconsequences.convict | 8 | 0.87% |
| justice.initiatejudicialprocess.trialhearing | 8 | 0.87% |
| movement.transportartifact.n/a | 8 | 0.87% |
| contact.negotiate.meet | 8 | 0.87% |
| life.injure.illnessdegradationhungerthirst | 8 | 0.87% |
| government.legislate.legislate | 8 | 0.87% |
| artifactexistence.damagedestroy.n/a | 8 | 0.87% |
| justice.investigate.investigatecrime | 8 | 0.87% |
| transaction.transferownership.borrowlend | 8 | 0.87% |
| justice.arrestjaildetain.arrestjaildetain | 8 | 0.87% |
| conflict.yield.n/a | 7 | 0.76% |
| conflict.attack.bombing | 7 | 0.76% |
| contact.collaborate.correspondence | 7 | 0.76% |
| transaction.transaction.transfercontrol | 7 | 0.76% |
| artifactexistence.damagedestroy.damage | 7 | 0.76% |
| transaction.transferownership.giftgrantprovideaid | 7 | 0.76% |
| contact.requestadvise.meet | 7 | 0.76% |
| government.agreements.acceptagreementcontractceasefire | 6 | 0.65% |
| justice.judicialconsequences.n/a | 6 | 0.65% |
| manufacture.artifact.createmanufacture | 6 | 0.65% |
| contact.mediastatement.broadcast | 6 | 0.65% |
| life.injure.illnessdegradationphysical | 6 | 0.65% |
| contact.commitmentpromiseexpressintent.correspondence | 6 | 0.65% |
| contact.threatencoerce.n/a | 6 | 0.65% |
| contact.commandorder.broadcast | 6 | 0.65% |
| transaction.transaction.giftgrantprovideaid | 6 | 0.65% |
| justice.judicialconsequences.execute | 6 | 0.65% |
| contact.requestadvise.n/a | 6 | 0.65% |
| contact.publicstatementinperson.broadcast | 6 | 0.65% |
| justice.initiatejudicialprocess.chargeindict | 6 | 0.65% |
| inspection.sensoryobserve.inspectpeopleorganization | 6 | 0.65% |
| manufacture.artifact.n/a | 6 | 0.65% |
| movement.transportartifact.sendsupplyexport | 6 | 0.65% |
| government.vote.n/a | 5 | 0.54% |
| conflict.yield.surrender | 5 | 0.54% |
| contact.threatencoerce.correspondence | 5 | 0.54% |
| movement.transportartifact.preventexit | 5 | 0.54% |
| contact.commitmentpromiseexpressintent.broadcast | 5 | 0.54% |
| contact.publicstatementinperson.n/a | 5 | 0.54% |
| conflict.attack.stealrobhijack | 5 | 0.54% |
| movement.transportperson.prevententry | 5 | 0.54% |
| government.agreements.n/a | 4 | 0.43% |
| contact.threatencoerce.broadcast | 4 | 0.43% |
| justice.initiatejudicialprocess.n/a | 4 | 0.43% |
| transaction.transaction.n/a | 4 | 0.43% |
| inspection.sensoryobserve.physicalinvestigateinspect | 4 | 0.43% |
| government.vote.castvote | 4 | 0.43% |
| conflict.attack.airstrikemissilestrike | 4 | 0.43% |
| contact.discussion.n/a | 4 | 0.43% |
| personnel.startposition.n/a | 4 | 0.43% |
| inspection.sensoryobserve.monitorelection | 4 | 0.43% |
| contact.prevarication.n/a | 4 | 0.43% |
| contact.commandorder.n/a | 4 | 0.43% |
| transaction.transfermoney.borrowlend | 4 | 0.43% |
| movement.transportperson.grantentryasylum | 4 | 0.43% |
| personnel.elect.winelection | 3 | 0.32% |
| movement.transportartifact.bringcarryunload | 3 | 0.32% |
| disaster.accidentcrash.accidentcrash | 3 | 0.32% |
| contact.commandorder.meet | 3 | 0.32% |
| personnel.endposition.quitretire | 3 | 0.32% |
| manufacture.artifact.build | 3 | 0.32% |
| contact.mediastatement.n/a | 3 | 0.32% |
| conflict.yield.retreat | 3 | 0.32% |
| contact.threatencoerce.meet | 3 | 0.32% |
| movement.transportperson.disperseseparate | 3 | 0.32% |
| disaster.fireexplosion.fireexplosion | 3 | 0.32% |
| movement.transportartifact.prevententry | 3 | 0.32% |
| personnel.elect.n/a | 3 | 0.32% |
| movement.transportartifact.smuggleextract | 3 | 0.32% |
| conflict.attack.biologicalchemicalpoisonattack | 3 | 0.32% |
| government.agreements.violateagreement | 2 | 0.22% |
| movement.transportartifact.grantentry | 2 | 0.22% |
| conflict.attack.firearmattack | 2 | 0.22% |
| conflict.demonstrate.marchprotestpoliticalgathering | 2 | 0.22% |
| contact.negotiate.correspondence | 2 | 0.22% |
| movement.transportperson.smuggleextract | 2 | 0.22% |
| justice.judicialconsequences.extradite | 2 | 0.22% |
| contact.commitmentpromiseexpressintent.n/a | 2 | 0.22% |
| government.vote.violationspreventvote | 2 | 0.22% |
| justice.investigate.n/a | 2 | 0.22% |
| movement.transportartifact.hide | 2 | 0.22% |
| contact.negotiate.n/a | 1 | 0.11% |
| inspection.sensoryobserve.n/a | 1 | 0.11% |
| movement.transportartifact.receiveimport | 1 | 0.11% |
| manufacture.artifact.createintellectualproperty | 1 | 0.11% |
| contact.funeralvigil.meet | 1 | 0.11% |
| conflict.attack.invade | 1 | 0.11% |
| movement.transportartifact.nonviolentthrowlaunch | 1 | 0.11% |
| government.formation.n/a | 1 | 0.11% |
| conflict.attack.setfire | 1 | 0.11% |
| movement.transportartifact.disperseseparate | 1 | 0.11% |
| conflict.attack.stabbing | 1 | 0.11% |
| movement.transportartifact.fall | 1 | 0.11% |
| government.formation.startgpe | 1 | 0.11% |
| contact.funeralvigil.n/a | 1 | 0.11% |
| movement.transportperson.hide | 1 | 0.11% |
| contact.commandorder.correspondence | 1 | 0.11% |

**test.** 432 instances in 432 records, 0 triggerless. Trigger gold 432. Argument gold 1,012. Arguments per instance, mean 2.35. Trigger length in whitespace tokens, mean 1.03, single-token share 97.5%.

#### test event types

| label | count | share |
| --- | --- | --- |
| life.die.deathcausedbyviolentevents | 19 | 4.40% |
| conflict.attack.n/a | 15 | 3.47% |
| conflict.attack.selfdirectedbattle | 14 | 3.24% |
| life.injure.injurycausedbyviolentevents | 12 | 2.78% |
| life.injure.n/a | 10 | 2.31% |
| movement.transportperson.preventexit | 9 | 2.08% |
| contact.discussion.meet | 9 | 2.08% |
| life.die.n/a | 8 | 1.85% |
| contact.requestadvise.meet | 8 | 1.85% |
| transaction.transfermoney.purchase | 8 | 1.85% |
| transaction.transfermoney.n/a | 7 | 1.62% |
| transaction.transferownership.embargosanction | 7 | 1.62% |
| contact.prevarication.broadcast | 7 | 1.62% |
| government.agreements.acceptagreementcontractceasefire | 6 | 1.39% |
| contact.publicstatementinperson.n/a | 6 | 1.39% |
| contact.funeralvigil.meet | 6 | 1.39% |
| movement.transportartifact.preventexit | 6 | 1.39% |
| government.spy.spy | 6 | 1.39% |
| justice.judicialconsequences.execute | 6 | 1.39% |
| artifactexistence.damagedestroy.damage | 5 | 1.16% |
| justice.initiatejudicialprocess.chargeindict | 5 | 1.16% |
| movement.transportartifact.smuggleextract | 5 | 1.16% |
| contact.requestadvise.correspondence | 5 | 1.16% |
| transaction.transaction.transfercontrol | 5 | 1.16% |
| manufacture.artifact.n/a | 5 | 1.16% |
| justice.arrestjaildetain.arrestjaildetain | 5 | 1.16% |
| disaster.accidentcrash.accidentcrash | 5 | 1.16% |
| personnel.endposition.firinglayoff | 5 | 1.16% |
| movement.transportartifact.n/a | 5 | 1.16% |
| artifactexistence.damagedestroy.n/a | 4 | 0.93% |
| justice.initiatejudicialprocess.n/a | 4 | 0.93% |
| contact.commitmentpromiseexpressintent.broadcast | 4 | 0.93% |
| government.legislate.legislate | 4 | 0.93% |
| inspection.sensoryobserve.physicalinvestigateinspect | 4 | 0.93% |
| contact.discussion.n/a | 4 | 0.93% |
| conflict.demonstrate.marchprotestpoliticalgathering | 4 | 0.93% |
| contact.prevarication.meet | 4 | 0.93% |
| movement.transportperson.n/a | 4 | 0.93% |
| transaction.transferownership.n/a | 4 | 0.93% |
| life.injure.illnessdegradationphysical | 4 | 0.93% |
| transaction.transfermoney.payforservice | 4 | 0.93% |
| government.agreements.n/a | 4 | 0.93% |
| contact.negotiate.meet | 4 | 0.93% |
| contact.mediastatement.broadcast | 3 | 0.69% |
| conflict.attack.stealrobhijack | 3 | 0.69% |
| movement.transportartifact.disperseseparate | 3 | 0.69% |
| contact.commandorder.correspondence | 3 | 0.69% |
| contact.commandorder.meet | 3 | 0.69% |
| personnel.endposition.quitretire | 3 | 0.69% |
| contact.discussion.correspondence | 3 | 0.69% |
| transaction.transaction.giftgrantprovideaid | 3 | 0.69% |
| manufacture.artifact.build | 3 | 0.69% |
| conflict.yield.n/a | 3 | 0.69% |
| contact.prevarication.correspondence | 3 | 0.69% |
| movement.transportartifact.sendsupplyexport | 3 | 0.69% |
| government.formation.mergegpe | 3 | 0.69% |
| contact.threatencoerce.correspondence | 3 | 0.69% |
| movement.transportartifact.hide | 3 | 0.69% |
| justice.judicialconsequences.n/a | 3 | 0.69% |
| movement.transportperson.disperseseparate | 3 | 0.69% |
| contact.requestadvise.n/a | 3 | 0.69% |
| movement.transportperson.bringcarryunload | 3 | 0.69% |
| transaction.transfermoney.borrowlend | 3 | 0.69% |
| justice.investigate.investigatecrime | 3 | 0.69% |
| contact.collaborate.n/a | 3 | 0.69% |
| transaction.transfermoney.giftgrantprovideaid | 3 | 0.69% |
| transaction.transfermoney.embargosanction | 3 | 0.69% |
| movement.transportperson.smuggleextract | 3 | 0.69% |
| personnel.endposition.n/a | 2 | 0.46% |
| personnel.elect.n/a | 2 | 0.46% |
| contact.funeralvigil.n/a | 2 | 0.46% |
| contact.threatencoerce.meet | 2 | 0.46% |
| movement.transportperson.selfmotion | 2 | 0.46% |
| transaction.transaction.n/a | 2 | 0.46% |
| contact.negotiate.correspondence | 2 | 0.46% |
| contact.threatencoerce.broadcast | 2 | 0.46% |
| transaction.transaction.embargosanction | 2 | 0.46% |
| conflict.attack.airstrikemissilestrike | 2 | 0.46% |
| personnel.startposition.hiring | 2 | 0.46% |
| contact.commitmentpromiseexpressintent.meet | 2 | 0.46% |
| government.formation.startgpe | 2 | 0.46% |
| conflict.attack.biologicalchemicalpoisonattack | 2 | 0.46% |
| contact.prevarication.n/a | 2 | 0.46% |
| inspection.sensoryobserve.inspectpeopleorganization | 2 | 0.46% |
| movement.transportartifact.fall | 2 | 0.46% |
| contact.publicstatementinperson.broadcast | 2 | 0.46% |
| conflict.demonstrate.n/a | 2 | 0.46% |
| conflict.attack.hanging | 2 | 0.46% |
| personnel.elect.winelection | 2 | 0.46% |
| transaction.transferownership.purchase | 2 | 0.46% |
| transaction.transferownership.giftgrantprovideaid | 2 | 0.46% |
| justice.judicialconsequences.extradite | 2 | 0.46% |
| conflict.attack.stabbing | 2 | 0.46% |
| conflict.attack.invade | 2 | 0.46% |
| life.injure.illnessdegradationhungerthirst | 2 | 0.46% |
| movement.transportartifact.bringcarryunload | 2 | 0.46% |
| contact.threatencoerce.n/a | 2 | 0.46% |
| government.formation.n/a | 2 | 0.46% |
| movement.transportperson.evacuationrescue | 2 | 0.46% |
| conflict.yield.retreat | 1 | 0.23% |
| contact.collaborate.correspondence | 1 | 0.23% |
| movement.transportperson.grantentryasylum | 1 | 0.23% |
| conflict.attack.bombing | 1 | 0.23% |
| government.vote.castvote | 1 | 0.23% |
| inspection.sensoryobserve.n/a | 1 | 0.23% |
| contact.negotiate.n/a | 1 | 0.23% |
| conflict.yield.surrender | 1 | 0.23% |
| movement.transportartifact.grantentry | 1 | 0.23% |
| contact.mediastatement.n/a | 1 | 0.23% |
| government.agreements.violateagreement | 1 | 0.23% |
| contact.collaborate.meet | 1 | 0.23% |
| contact.commandorder.broadcast | 1 | 0.23% |
| government.agreements.rejectnullifyagreementcontractceasefire | 1 | 0.23% |
| disaster.fireexplosion.fireexplosion | 1 | 0.23% |
| justice.initiatejudicialprocess.trialhearing | 1 | 0.23% |
| contact.requestadvise.broadcast | 1 | 0.23% |
| transaction.transferownership.borrowlend | 1 | 0.23% |
| life.die.nonviolentdeath | 1 | 0.23% |
| contact.commitmentpromiseexpressintent.n/a | 1 | 0.23% |

### Argument inside a trigger

An argument is inside when it and a trigger of the same event both carry character offsets and the argument span is contained in the trigger span, edges included. An equal span counts as inside. A partial overlap, and a trigger contained in a longer argument, count as outside. String gold has no span, so it is unmeasured and stays out of the inside share. This is a span count, not an F1.

| split | inside | placed | inside share | equal | nested inside | trigger inside argument | partial overlap | disjoint | unmeasured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 17,026 |
| val | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 1,006 |
| dev | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 2,188 |
| test | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 | 1,017 |

Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.

### String occurrence

How many places a gold surface lands when placement is a lowercased whitespace-token run, which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. Unplaced is the share of surfaces that land nowhere. Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. Counted over every record.

| split | trigger mean occ | trigger share > 1 | argument mean occ | argument share > 1 | unplaced share | offset arguments | string arguments | offset triggers | string triggers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1.20 | 14.15% | 1.56 | 31.60% | 0.00% | 0 | 17,026 | 0 | 7,329 |
| val | 1.19 | 15.49% | 1.62 | 33.60% | 0.00% | 0 | 1,006 | 0 | 439 |
| dev | 1.24 | 17.53% | 1.53 | 31.63% | 0.00% | 0 | 2,188 | 0 | 924 |
| test | 1.28 | 20.37% | 1.55 | 31.76% | 0.00% | 0 | 1,017 | 0 | 432 |

### Event identity

The multi-trigger share is events whose trigger list has more than one surface. A same-type pair that shares an argument text is an upper bound on one real event split into two records: two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.

| split | events | share > 1 trigger | same-type pairs | share an argument | share role + argument | nested | identical | docs with a shared argument |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 7,329 | 0.00% | 0 | n/a | n/a | n/a | n/a | 0.00% |
| val | 439 | 0.00% | 0 | n/a | n/a | n/a | n/a | 0.00% |
| dev | 924 | 0.00% | 0 | n/a | n/a | n/a | n/a | 0.00% |
| test | 432 | 0.00% | 0 | n/a | n/a | n/a | n/a | 0.00% |

### Coreference

**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.

No `coref_clusters`.

**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. A same-type pair drawn from two events is a gold hard negative: different events. False triggers that are not in the gold are not in this count.

No event lists more than one trigger, and no document has two events of one type, so there is no trigger pair.

**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.

| split | mentions the string rule adds | arguments skipped, 2+ clusters | arguments in one multi-mention cluster | other mentions of that cluster | argument spans in 2+ clusters |
| --- | --- | --- | --- | --- | --- |
| train | 0 | 0 | 0 | 0 | 0 |
| val | 0 | 0 | 0 | 0 | 0 |
| dev | 0 | 0 | 0 | 0 | 0 |
| test | 0 | 0 | 0 | 0 | 0 |

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
