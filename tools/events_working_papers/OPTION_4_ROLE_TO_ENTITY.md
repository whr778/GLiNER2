# Option 4 — remap event ROLES into the ENTITY space

**Status: scoped, not built.** Written 2026-09-17, after options 1 and 3 both measured
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
