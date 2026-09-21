# Option 4 — remap event ROLES into the ENTITY space

**Status: A/B COMPLETE 2026-09-20 — NEGATIVE — and RE-RUNNING, because the first run's
REGIME was broken rather than the idea.**

> ## *** VERDICT 1 (2026-09-20): NEGATIVE, AND REGIME-BOUND ***
>
> | metric (strict micro F1) | control | treatment | delta |
> |---|---:|---:|---:|
> | **event_argument** | 0.3506 | 0.3206 | **-0.0300** |
> | **event_trigger** | 0.6057 | 0.5516 | **-0.0541** |
> | event | 0.5375 | 0.5124 | -0.0251 |
> | structure | 0.2288 | 0.2063 | -0.0225 |
> | entity | 0.5540 | 0.5358 | -0.0182 (inside floor) |
> | event_type | 0.8137 | 0.8314 | +0.0176 (inside floor) |
>
> **0 up, 10 down outside the ±0.02 floor.** Both arms calibrated to threshold **0.3** on
> their own val sets at the same commit `92c0605`, so the operating point is not the
> explanation.
>
> **IT IS NOT A PRECISION/RECALL TRADE -- BOTH HALVES FELL.** `event_argument` precision
> 0.4536 -> 0.4252 and recall 0.2858 -> 0.2573, with **694 MORE arguments never found**
> (FN 11,696 -> 12,390). The sweep lines suggested a trade; the blind test does not.
>
> **~~THE FIRST STEP OF THE MECHANISM WORKED, THE SECOND DID NOT.~~ RETRACTED 2026-09-21.**
> That diagnosis -- entity recall +0.0429, entity precision -0.0912, COR +12,365 against FP
> +18,373, "the pool got noisier not richer" -- rested ENTIRELY on the entity row, and **the
> entity row is VOID**: the arms were scored on different test sets. The treatment config
> adds `data/cmnee_roles_ner`, whose TEST split joins the eval, so entity support is 79,912
> on the control and **96,306** on the treatment. Comparing F1 across different gold is
> exactly what the standing rule forbids.
>
> **WHICH ROWS SURVIVE.** Support is IDENTICAL on all other heads, so they compare legally:
>
> | head | support (both) | delta | valid |
> |---|---:|---:|---|
> | event_argument | 20,827 | -0.0300 | yes |
> | event_trigger | 14,041 | -0.0541 | yes |
> | event | 44,730 | -0.0251 | yes |
> | structure | 4,167 | -0.0225 | yes |
> | event_type | 9,862 | +0.0176 | yes (inside floor) |
> | classification | 10,291 | -0.0002 | yes |
> | relation | 9,428 | +0.0025 | yes |
> | **entity** | **79,912 vs 96,306** | ~~-0.0182~~ | **VOID** |
>
> **So the VERDICT stands and its EXPLANATION does not.** The derived corpus cost the event
> heads, on matched gold. Why is now unknown: the over-proposal story was measured on
> incomparable data. It was only caught because `eval_provenance` -- added 2026-09-20 -- made
> `compare_runs.py` REFUSE the re-run outright; yesterday's files had no provenance, so the
> tool could only warn and I read the entity row as evidence.
>
> **WHY THE REGIME WAS BROKEN.** Nothing in that run could punish over-proposal:
> 1. the roles configs declared **no `negative_pools`**;
> 2. `partial_annotation: cmnee_roles_ner: [entities]` is read ONLY by
>    `build_negative_pools.py` -- **the trainer never reads it**, so it protected nothing at
>    training time;
> 3. measured across all 11 fetchable corpora in the mix, **0% of presented entity labels
>    carry an empty answer** -- the mix itself never teaches that an offered label can be
>    absent;
> 4. and per [[LABEL_NEGATIVES_PLAN]], negatives could not have applied even if asked for.
>
> The augmentation pushes the same way: `synthetic_entity_label_prob: 0.2` renames a fifth of
> entity presentations, and with no descriptions and no absent labels a renamed label still
> has gold -- teaching "whatever this label is called, these spans answer it".
>
> ## *** THE RE-RUN (`roles2`, 2026-09-21) IS VOID -- TWO CONFOUNDS ***
>
> 1. **Shipped checkpoints differ.** `metric_for_best: eval_loss` again: the control improved
>    on val loss ONCE and shipped its **epoch-2** checkpoint, the treatment improved three
>    times and shipped **epoch 5**. Same defect as the absneg pair; it was diagnosed on
>    2026-09-20 while this run was already four hours in, fixed in `absneg2`, and never
>    back-checked against the run in flight.
> 2. **Different test sets**, 20,602 vs 23,326 records, for the same reason as above.
>
> `compare_runs.py` REFUSED it, which is the provenance fix working. ~$40, and preventable.
>
> **ONLY THE CONTROL NEEDS RETRAINING.** Checked against the Hub: each arm pushed `best/`
> and nothing else. The TREATMENT's `best/` IS its epoch-5 (final) checkpoint, from a run
> whose training was never corrupted -- it trained on exactly its intended 17 corpora. The
> CONTROL's `best/` is epoch 2 and its later weights are gone. So: retrain the control
> (~$20) with `roles3-control.yaml`, re-score the existing treatment on the common 15-corpus
> test set (~$2), and compare. Code drift between the two commits is INERT -- the only
> changes are option 2's TypedRole paths, both gated on `role_type_map`, which neither config
> sets.

> **BEFORE ANY FURTHER RE-RUN, FIX BOTH:** `metric_for_best` on a shared task metric, and
> evaluate the treatment on the CONTROL's test set so the entity comparison is legal at all.
>
> **SO THE REFUTATION IS REAL BUT REGIME-BOUND**, and the re-run (`roles2-*`, launched
> 2026-09-20) puts both arms in a regime that injects: **~4.9 absent queries per forward
> against 0.37**. The derived corpus KEEPS `partial_annotation` -- measured, 42.3% (upper
> bound) of its records have an absent role whose surface is in the text (核潜艇 present while
> `EventMilitaryforce` is absent), so it is positives-only and must never SOURCE negatives.

**Status of the original build: BUILT 2026-09-18, A/B RAN 2026-09-19.** The step AFTER this one is
[[OPTION_2_TYPED_ROLE_CONSTRAINTS]], which is gated behind this result. `tools/data/roles_to_entities.py`
exists, the corpus is on the Hub, and both arms are training — see [[EXPERIMENT_CATALOG]].

**AND THE DESIGN CHANGED, which supersedes §5 below.** This document concludes that only four
of eleven roles survive and usable supply is 24.3% (15,188 of 62,573). That reasoning assumed
the derived label CLAIMS A TYPE. It does not have to. Under `--mode namespaced` every role
becomes `Event<Role>` — a label of its own, claiming a ROLE — and both objections dissolve:
the same-document conflict is ordinary multi-label role annotation, and none of the eleven
namespaced names collide with the base's 2,050-label entity vocabulary (9 of 11 PLAIN names
do). **Supply is 76,863 unique (label, surface) pairs with 0% lost**, including `Subject` at
31,071 — the role the entity head fails hardest on, 96% never proposed.

What namespacing buys is PROPOSAL, not typing: it does not transfer to a real NER taxonomy and
stays tautological for option 2. `--mode canonical` (§5's four roles as real labels) and
`--mode hybrid` are retained for when transfer is the goal. Read §5 as the adjudication of the
TYPE-CLAIMING variant, which it is, and which remains correct on its own terms.

---

Written 2026-09-17, after options 1 and 3 both measured
negative. Every number below is measured, not estimated; where something is an estimate it
says so.

---

## 1. Why this one, when two others just failed

`event_argument` relaxed recall is 0.42 and strict 0.30, so **roughly a third of the loss is
arguments that were NEVER PROPOSED.** Options 1–3 all constrain *binding* — which span fills
which role — and none of them can recover a span that was never put forward.

Both binding-half options have now been measured, and both are negative:

| option | mechanism | result |
|---|---|---|
| 1 — `candidate_pool: shared` | one document pool, relieve budget competition | entity **−0.089**, trigger −0.030, argument **null** (+0.015 rel / −0.009 strict, inside floor) |
| 3 — filter arguments by predicted entity types | OneIE-style typing at decode | best configuration **−0.074 F1**; no menu or filter beat the no-menu baseline |

Option 1 mattered more than its own number, because it was a **pre-registered rescue for
option 3**: option 3's sweep concluded the entity menu cost 30–35% of argument recall through
candidate-budget competition, and stated that `candidate_pool: shared` "is exactly what
addresses" it. It ran, with a matched control and a gate proving the treatment applied, and the
target metric did not move. **Budget competition was not the binding constraint.** That is the
strongest reason to stop attacking binding.

Option 4 reframes an argument as a typed span extracted by the ENTITY head, needing no trigger
link — so it adds **extraction supervision** rather than constraining binding, and is the only
one of the four that can move the never-proposed third.

**It is a complement, not a replacement.** A span extracted as an entity is not attached to an
event instance, so the record head still owns grouping. Option 4 buys recall of the span, not
its role assignment.

---

## 2. The supply, measured

| corpus | records | records with entity gold | argument mentions | distinct roles | surfaces present verbatim |
|---|---:|---:|---:|---:|---:|
| **cmnee** | 9,281 | **0** | 62,573 | 11 | **100%** |
| casie | 798 | 798 | 17,992 | 26 | 100% |

Project-wide the earlier survey found 484,424 argument mentions across 953 roles, with the
largest pools in corpora carrying **no entity gold at all** — casualty_events (124,561),
cmnee (62,573), duee (28,875), rams (17,026).

**cmnee is the pilot.** It manufactures entity supervision where the corpus has none, its
roles are already entity-shaped, and it is the corpus every event number in this programme is
quoted against.

cmnee roles: `Subject`(20,743) `Equipment`(10,496) `Date`(8,984) `Object`(6,289)
`Materials`(5,875) `Location`(5,476) `Militaryforce`(2,736) `Content`(644) `Result`(602)
`Quantity`(390) `Area`(338)

---

## 3. Feasibility gates — both already run

**Gate A — do the surfaces align?** This is load-bearing: `_find_sublist` matches token
SUBSEQUENCES, so a surface that cannot be located as one is unusable no matter how many there
are. Measured on cmnee's first 2,000 records through the real `WhitespaceTokenSplitter`:

**18,477 / 18,481 = 99.98% align.**

The four failures are annotation artifacts, not a mechanism problem — e.g. `巴西空军4`, where
the trailing `4` belongs to `43架` in the source text.

**Gate B — is the derived label deterministic?** See §5. Measured on casie: type-named roles
are (`Time`→Time 100%, `Vulnerability`→Vulnerability 99%); function-named roles are not
(`Victim`→Person 41% / Organization 35% / System 12%).

---

## 4. Build order — the flag comes FIRST, and nothing may precede it

**`build_negative_pools.py` infers annotation from the presence of gold:**

```python
"annotates": {d: bool(seen[d]) for d in DIMENSIONS},
```

A derived corpus carries entity gold, so it would **auto-qualify as an entity annotator** and
begin drawing entity negatives against documents where only argument spans were labelled and
every other entity is unlabelled. There is no override today.

That is the within-dimension rule — "absence means *not labelled*, not *not present*" —
violated deliberately and at 62,573-mention scale. **It would poison the negatives pool for
every model trained afterwards, and it would do so silently**, because a negatives pool has no
natural correctness signal.

So the order is not negotiable:

1. **Partial-corpus flag** in `build_negative_pools.py`: a corpus may be declared a source of
   POSITIVES for a dimension while being excluded as a source of NEGATIVES for it. With a test
   that fails without it.
2. **The converter** `tools/data/roles_to_entities.py`, modelled on the existing
   `events_to_entities.py` (which already does the TRIGGER half: `event_type → entity label`
   using the trigger surface, and is how `maven_ner` was built). Option 4 is the ARGUMENT half:
   `role → entity label` using the argument surface.
3. **Derived corpus for cmnee only**, flagged partial, ADDITIVE — alongside the original,
   never replacing it (see §6).
4. **A/B**: base vs base + derived-cmnee, matched control, same box, same precision.

---

## 5. Labelling policy — and the circularity trap it resolves

Two guards pull against each other:

| labelling choice | safe? | enables option 2? |
|---|---|---|
| keep the ROLE NAME as the label | yes — deterministic by construction | **no — tautological** |
| map onto a real NER taxonomy | **no** — `Victim` → Person 41% / Org 35% / System 12% | yes |

If the entity label is derived FROM the role, then "role `Victim` is filled by entity
`Victim`" is a tautology: option 2 would run and teach nothing.

**The casie measurement resolves it.** Roles whose NAME IS A TYPE NAME are deterministic;
roles naming a semantic FUNCTION are not. So:

- **Type-named roles → map onto the canonical taxonomy** via `labels_file`: cmnee's `Date`
  (8,984), `Location` (5,476), `Area` (338), `Quantity` (390); casie's `Time`, `Place`, `CVE`.
  These are also the high-volume cases project-wide (`location` 53,779, `Date` 41,812).
- **Function-named roles → keep the role name**: `Subject`, `Object`, `Victim`, `Attacker`,
  `Materials`, `Content`, `Result`.

Mapping happens in the **config**, never by rewriting corpora — the standing rule.

### MEASURED 2026-09-17: six of cmnee's eleven roles COLLIDE with the base vocabulary

The two-way split above was too coarse. The base's entity vocabulary is 1,943 labels, and
**six of cmnee's eleven roles already exist in it** — so "keep the role name" is not the safe
default it was assumed to be. Reading the surfaces each side actually tags, per the standing
rule never to merge on string similarity:

| role | base tags (text2json / paraloq_json) | cmnee tags | verdict |
|---|---|---|---|
| `Subject` | "Visual Arts", "Sacrifice of Isaac", "Phonics" — SUBJECT MATTER | 俄罗斯海军总司令部, 印度海军 — the ACTOR | **INCOMPATIBLE** |
| `Object` | "water-jar", "small bowl", "Louisiana", "Hartford" | the thing acted upon | **INCOMPATIBLE** |
| `Equipment` | "Pheromone dispensers", "kiln" | 海豹号核潜艇 (submarine) | doubtful — same word, different domain |
| `Quantity` | "153 countries", "21 billion" | "21", "20" | compatible |
| `Date` | dates | dates | compatible |
| `Location` | places | places | compatible |

Merging `Subject` would teach the model that "Visual Arts" and "Russian Navy Headquarters"
are the same type, at loss weight, across 20,743 mentions — the single largest role in the
corpus.

**So the policy is THREE-way, not two:**

1. **Type-named and compatible** → map onto the canonical taxonomy: `Date`, `Location`,
   `Quantity`.
2. **Function-named, no collision** → keep the role name: `Materials`, `Militaryforce`,
   `Content`, `Result`, `Area`.
3. **Colliding with a base label of DIFFERENT meaning** → must be renamed, never merged:
   `Subject`, `Object`, and probably `Equipment`. A namespaced label (`EventSubject`,
   or role-qualified) keeps the supervision without corrupting an existing type.

**The general rule this establishes:** a derived label set must be diffed against the BASE's
vocabulary and every collision adjudicated by reading surfaces, BEFORE generation. A collision
is not evidence of agreement — it is the most dangerous case precisely because it looks like
agreement, and nothing downstream will flag it.

### FULL ADJUDICATION, 2026-09-17 — and it cuts the usable supply to 24%

Two corrections to the section above before the verdict. First, the exact-match collision check
was insufficient: matched case-insensitively, **9 of 11 roles collide**, not 6 — `Materials`,
`Content` and `Result` collide with lowercase `materials`, `content`, `result` in
`paraloq_json`. Only `Militaryforce` and `Area` are genuinely new.

Second, and decisive: **ROLE IS NOT TYPE, and cmnee proves it internally.**

**19.5% of cmnee documents (1,810 / 9,281) tag one surface with more than one role in the SAME
document.** 猎豹号核潜艇 — a nuclear submarine — is `Equipment`, `Materials` AND `Subject` in
one document. 美国华盛顿号航母 (a US carrier) is `Location` + `Subject`. A submarine's TYPE does
not change within a document; its ROLE does. Deriving type from role therefore manufactures
contradictory type supervision, and the conflicts concentrate exactly where the volume is:

| conflicting pair, same document | surfaces |
|---|---:|
| Object + Subject | 802 |
| Equipment + Materials | 751 |
| Equipment + Militaryforce | 333 |

| role | mentions | base label tags | cmnee tags | verdict |
|---|---:|---|---|---|
| `Date` | 8,984 | "Sunday", "October" | 2008年11月, 周二 | **DERIVE** → `Date` |
| `Location` | 5,476 | "New York", "Tahrir Square" | 日本海, 日本群马县 | **DERIVE** → `Location` |
| `Area` | 338 | *(new)* | 阿富汗, 土耳其中部城市瑟瓦斯 | **DERIVE** → merge into `Location` (67 of its surfaces already conflict with it) |
| `Quantity` | 390 | "153 countries", "21 billion" | 21, 20名 | **DERIVE** → `Quantity` |
| `Subject` | 20,743 | "Visual Arts", "Phonics" — subject MATTER | the ACTOR | **REJECT** — incompatible, and 802 same-doc conflicts with Object |
| `Object` | 6,289 | "water-jar", "Louisiana" | entity acted upon | **REJECT** — incompatible |
| `Equipment` | 10,496 | "Pheromone dispensers", "kiln" | submarines, aircraft | **REJECT** — 751 same-doc conflicts with Materials |
| `Materials` | 5,875 | "Oil on canvas", "Enamel paint" | submarines, missiles | **REJECT** — incompatible and conflicting |
| `Militaryforce` | 2,736 | *(new)* | patrol ships, rescue personnel | **REJECT** — 333 conflicts with Equipment |
| `Content` | 644 | chat messages, document text | 飞行训练, 军事演习 (activities) | **REJECT** — incompatible |
| `Result` | 602 | "12.5", "200" | "20人牺牲、21人受伤" | **REJECT** — incompatible |

**Restricted to the four DERIVE roles, the derivation is nearly conflict-free: 71 conflicting
surfaces in 15,188 mentions (0.5%), and 67 of those are Area↔Location, which merge anyway.
Four remain.** Against 2,374 conflicts if all roles are used.

**THE COST OF THIS HONESTY: usable supply falls from 62,573 to 15,188 mentions — 24.3%.** The
single largest role, `Subject` at 20,743, is unusable as a type. Option 4's headline supply
figure of 484,424 project-wide should be read as roughly a quarter of that until each corpus
is adjudicated the same way, because nothing about cmnee is unusual here.

### Can we skip the role labels and map roles DIRECTLY to NER types?

Asked 2026-09-17. Two halves, and they answer differently.

**Implementation half — yes.** The `labels_file` indirection exists so SOURCE corpora are never
rewritten. A derived corpus is new data we author, so the converter may emit the canonical
label directly (`Location`, not `Area`-then-remap). No rule is bent by that.

**Substantive half — measured on 50 cmnee documents, 583 argument mentions, offering the base
model a 12-label English menu and asking which type it assigns to each gold argument surface:**

| role | n | model's verdict | agrees on type where it predicts |
|---|---:|---|---|
| `Subject` | 176 | **96.0% NOT PREDICTED AT ALL** | — |
| `Date` | 138 | 66.7% not predicted | `Date` ×46 |
| `Location` | 91 | 71.4% not predicted | `Location` ×25 |
| `Quantity` | 57 | 94.7% not predicted | `Quantity` ×3 |
| `Result`, `Object`, `Equipment`, `Materials` | 110 | 100% not predicted | — |

**A direct map is right for the type-like roles and impossible for the rest — but not for the
reason expected.** It is not that the model assigns a scattered type; it is that **the model
does not propose these spans at all**, so there is no predicted type to map onto. Where it does
predict, it agrees: `Date`→Date, `Location`→Location, `Quantity`→Quantity, which independently
confirms the four DERIVE decisions above.

**AND THIS IS THE STRONGEST EVIDENCE YET FOR OPTION 4'S THESIS.** 96% of `Subject` spans, 71%
of `Location` spans and 67% of `Date` spans are human-annotated gold that the entity head never
puts forward. That is the never-proposed third, measured directly rather than inferred from a
recall figure — and it is exactly the gap extraction supervision is supposed to close.

**Limits, stated because the number is quotable:** 50 documents; the offered menu is 12 English
labels chosen by hand, not the model's trained vocabulary, so absolute rates would move with a
better menu. `Date` and `Location` ARE canonical labels and still go unproposed two thirds of
the time, so the effect is not an artefact of menu choice. A Chinese-language menu returns
nothing at all, consistent with labels being an INPUT that must match the trained spelling.

**This does not kill option 4, but it resizes it.** Deriving `Date`, `Location` and `Quantity`
into a corpus with ZERO entity gold is still real extraction supervision, still attacks the
never-proposed third, and is now known to be clean. It is simply a quarter of the lever the
scope assumed — and that should be priced before the converter is written, not after.

> *Updated 2026-09-21: option 2's TRAINED margin is refuted (loss effect 0.28% mean -- the
> model already down-ranks type-incompatible fillers), so neither route rescues the trained
> arm. The decode-time route below is unaffected and remains a null on F1 with a real
> precision/recall trade. See [[OPTION_2_TYPED_ROLE_CONSTRAINTS]].*

**Option 2 does not depend on this anyway.** It has a second route: use the model's PREDICTED
entity types as the type signal instead of gold, which works on cmnee today with no data
derivation.

---

## 6. Guards, each from something already measured

1. **PARTIAL ANNOTATION — the serious one.** §4. Positives only, never negatives.
2. **ROLE IS NOT ENTITY TYPE.** §5.
3. **INVENTED-LABEL CORPORA MUST NOT DRIVE IT.** `zh_multitask` shows 791 distinct roles and
   `mix_natural` 156 — the "annotator invented labels per document" problem. Only taxonomy
   corpora contribute, measured first. cmnee's 11 and casie's 26 are taxonomies; 791 is not.
4. **ADDITIVE, NEVER DESTRUCTIVE.** docee, chfinann, docfee and turkish_event were converted
   destructively — events replaced by entities plus classifications — which is why they train
   **zero events** today. The derived corpus sits beside the original.
5. **SPLIT HYGIENE.** The derived corpus inherits cmnee's splits exactly. Deriving entities
   from train and test independently would put the same document's spans on both sides.

---

## 7. Instruments that must be able to fail

- **Composition line**: the derived corpus must report entity gold on records that previously
  had none — `entities=0 → entities>0` for cmnee. If it reads 0, the derivation did not apply.
- **Negatives gate**: the pools JSON must show the derived corpus with `annotates.entities`
  **false** despite carrying entity gold. That inversion IS the flag working; if it reads
  true, stop — the pool is being poisoned.
- **Never-proposed metric**: the whole thesis is recall of spans that were never proposed.
  Report `event_argument` relaxed RECALL beside F1, since F1 can rise on precision alone and
  would hide a failure on the actual target.

---

## 8. Cost

| step | estimate |
|---|---|
| partial flag + test | local, no GPU |
| converter + derived cmnee | local, no GPU |
| A/B (2 arms, matched, A100) | ~40 min each at the observed 17.0 / 10.5 samples/s → **~$2–3** |

The expensive part is not the GPU; it is that a wrong negatives pool is silent and durable.

---

## 9. What would kill it

- The derived entity head learns the spans but `event_argument` recall does not move — the
  spans were already proposed and binding really is the whole loss, contradicting §1.
- Entity F1 on the ORIGINAL entity corpora drops: the derived labels are polluting a taxonomy
  the base already knows. Watch `entity_strict` on the unchanged corpora, not just the target.
- Partial annotation leaks into negatives despite the flag — detectable by §7's inversion
  check, and the reason that check exists.
