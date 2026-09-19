# Option 2 — typed constraints on event role edges

**Status: PLAN, nothing implemented.** Written 2026-09-19 while the Option 4 A/B runs, and
**gated behind it**: option 4 must first show the lever moves on cmnee, and the derivation must
then be generalised to every event corpus with no entity gold. This document is the design for
the step after that.

Companion to [[EVENT_ARGUMENT_DIAGNOSIS]] §4k (where options 1–4 are stated),
[[OPTION_4_ROLE_TO_ENTITY]] (the step before this one), [[JOINT_IE_DESIGN_RECORD]] (the beam),
and [[EXPERIMENT_CATALOG]].

---

## 1. What option 2 actually is, having now read the code

§4k describes it as "extend `joint_ie`'s typed constraints to event roles … event roles are the
same shape and simply are not wired in." That is right in spirit and wrong in two details that
change the build.

**First, the joint engine does not decode events at all.** `grep event gliner2/joint_ie/engine.py`
returns nothing: that module builds entity nodes and *relation* edges. Events reach the beam by
a different route — `boundary/engine.py::_decode_joint` (line 571) turns record groups into role
edges via `boundary_record_groups_to_role_edges(groups, query_types, **gate)`, and under
`event_records: true` **events ARE records**, so event role edges are already in the beam today.

**Second, the constraint is therefore not missing machinery but a missing emission.**
`_decode_joint` builds its constraint list from `core["rel_specs"]` alone:

```python
constraints = [
    TypedEndpoints(entry["spec"].relation_type, head_types, tail_types)
    for entry in core["rel_specs"][sample_index]
]
```

Relations get typed endpoints; **event role edges travel the same beam with no constraint at
all.** `TypedEndpoints` (`joint_ie/constraints.py:105`) is a frozen dataclass with one method,
`allows(candidate, relations, entities) -> bool`, and there are ten sibling constraints, so the
extension point is established and small.

So option 2 is: **emit a typed constraint for event role edges, per role, and give it a type
signal that is not circular.** The second half is the whole difficulty.

---

## 2. The evidence, and the one trap it sets

From the 800-document sweep in [[EVENT_ARGUMENT_DIAGNOSIS]] §4k:

- The type signal is **real**: 88% of correct arguments carry a predicted entity type against
  68% of wrong ones.
- It is **role-dependent**: `Location` and `Date` discriminate; `Subject` does not — in military
  news the subject is an aircraft whether the binding is right or wrong, so type identity
  carries nothing there.
- Post-hoc filtering on it is **NEGATIVE**: −0.074 F1 at best, and no menu or filter beat the
  no-menu baseline.

**THE TRAP, and the reason this plan is not "wire it in and run it".** The same sweep found that
merely OFFERING an entity menu cost **30–35% of argument recall before any filter was applied**
(recall 0.2495 → 0.16–0.17). If option 2 obtains its type signal by offering entity queries at
decode, it inherits that loss and will lose for the same reason option 3 lost. Option 1
(`candidate_pool: shared`) was the pre-registered fix for exactly this cannibalisation and
measured negative, so that escape is closed.

**Therefore the type signal must not come from a decode-time entity menu.** Three routes, and
only one is viable:

| route | verdict |
|---|---|
| offer an entity menu at decode, constrain on it | **CLOSED** — this is option 3, measured −0.074 |
| use option 4's derived `Event<Role>` labels as the type | **CIRCULAR** — "role `Subject` is filled by entity `EventSubject`" is a tautology and teaches nothing |
| **train with the constraint so the type is internalised** | **THE ONE LEFT** — and it is what the sweep's own conclusion names as not ruled out |

---

## 3. Design

### 3.1 The constraint

A new sibling in `joint_ie/constraints.py`, alongside `TypedEndpoints`:

```python
@dataclass(frozen=True)
class TypedRole(Constraint):
    """An event role edge may only land on a span the model types compatibly.

    Applied PER ROLE, never globally: the sweep measured that `Location`/`Date`
    discriminate and `Subject` does not, so a global constraint spends its
    precision on the roles where type identity is uninformative.
    """
    event_type: Optional[str] = None
    role: Optional[str] = None
    allowed_types: tuple[str, ...] = ()

    def allows(self, candidate, relations=(), entities=()) -> bool: ...
```

Emitted from `_decode_joint` beside the relation constraints, from the event record specs
rather than `rel_specs`.

### 3.2 Where the allowed types come from

**Declared in the config, derived from data, never invented.** A generator
(`tools/data/build_role_type_map.py`) reads corpora carrying BOTH entity gold and event gold —
casie today — and emits, per `(event_type, role)`, the entity types its gold arguments actually
carry, with counts. A role whose distribution is flat or whose support is below a floor is
**omitted**, which is how "per role, not globally" is enforced mechanically rather than by
judgement. The map is a config artefact like `labels/unified.yaml`, regenerated by a tool,
never hand-edited.

### 3.3 Training with it — what "internalised" actually means

This is the load-bearing paragraph, so it is spelled out rather than asserted.

**The diagnosis.** A constraint at decode narrows the search space; it does not improve the
ranking WITHIN that space. The beam already decodes over candidate scores trained GREEDILY, so
nothing in training ever optimised for constraint satisfaction — the tracked beam-aware/
structured-loss item in `TODO.md` states this as a general train/test mismatch, and gives two
measurements of it:

- where the beam SUBSTITUTES rather than deletes, greedy-only is right 41.4% of the time and
  joint-only 37.5% — **a coin flip**, which is exactly what is seen when the scores carry no
  information about which constraint-consistent assignment is correct;
- on biored the beam ADDS 143 entity predictions of which **5 are correct — 3.5% precision on
  what it contributes**. Confident, constraint-consistent, and wrong.

That is why option 3 lost and why simply moving the same filter into the beam should be
expected to lose too (prediction 1 in §5). A hard post-hoc filter with an imperfect type signal
— 88% of CORRECT arguments carry a predicted type, so 12% do not — caps recall at 88% before
anything else goes wrong, and the measured menu cost made it far worse.

**"Internalised" therefore means: teach the scores to rank correctly INSIDE the constrained
space, rather than filtering after they have ranked without knowing about it.** Three designs,
cheapest first. Only the first is proposed for this plan; the others are named because they are
the same question at larger scale.

1. **A constraint-violation penalty — a consistency loss between two heads.** Penalise the
   model when its ARGUMENT head binds a span its ENTITY head types incompatibly. No
   differentiable beam is required. **This shape already exists in the codebase:**
   `marginal_pair_consistency_loss` (`boundary/losses.py:555`) forces the boundary marginals to
   agree with the candidate-level noisy-OR probabilities — two views of the same object made to
   agree, at `consistency_loss_weight: 0.1` after `consistency_warmup_steps: 2000`. Option 2's
   version is the same template over a different pair of views, and should reuse the weight and
   the warmup rather than inventing new ones.
2. **A structured hinge over the beam's assignment** — train against what the beam will
   actually choose. Addresses the mismatch directly and is the tracked item's first candidate.
3. **A differentiable relaxation (Sinkhorn / SparseMAX)** so the assignment itself carries
   gradient. Most principled, most expensive, and out of scope here.

**THE FAILURE MODE THIS MUST BE GATED AGAINST, and it is not hypothetical.** A penalty on
disagreement has a trivial solution: **both heads agreeing while both are wrong.** Agreement is
a FORM metric, and this programme's standing lesson is that a form gate must be paired with a
correctness companion — counting firings is not counting hits. So the gate is not "did
head-agreement rise" but **"did agreement rise AND did `event_argument` strict F1 rise with
it"**. If agreement climbs while F1 is flat or falls, the penalty has taught the two heads to
share a mistake and the arm is negative regardless of how clean the consistency curve looks.

Keep the decode constraint on during training so train and eval see the same rule — this
programme has already paid for a train/eval mismatch once, in the negatives menu dose (training
injected 1 absent label per dimension while eval offered 0 or 20).

---

## 4. Build order, each step gated

1. **The map, and its own test.** `build_role_type_map.py` over casie. Gate: the emitted map is
   non-empty, every entry has support above the floor, and `Subject`-like flat roles are
   ABSENT. A map that contains every role has failed, not succeeded.
2. **`TypedRole` + unit tests.** Gate: a test that FAILS without the constraint — an edge to a
   wrongly-typed span is admitted before and refused after.
3. **Emission from `_decode_joint`.** Gate: a deterministic line per run showing how many role
   edges the constraint actually refused. **Zero refusals means it did not apply**, which is the
   failure this programme has shipped three times; the line must be emitted from a point where
   the refusal has already happened.
4. **Decode-only A/B (cheap, and expected to be ~NULL).** One trained checkpoint, constraint on
   vs off. This is the honest replication of option 3 through the beam rather than a filter. If
   it is strongly positive, stop — the training step is unnecessary.
5. **Trained A/B (the real test).** Two arms, matched, constraint in training + decode vs
   neither.

---

## 5. Measurement, pre-registered

**Primary metric:** `event_argument` strict micro-F1 on the shared blind test, at a fixed
operating point, both arms at the same threshold and menu dose, `eval_provenance` matching.
**Floors, measured:** `event_argument` seed sd **0.0009–0.0020**, entity **0.0139**,
classification **0.0013–0.0178**. A delta inside the floor is a null and will be reported as one.

**Secondary, and watched for collateral:** entity, event_trigger, event_type, structure,
classification. The negatives run is the precedent — it hit its target and broke an untargeted
head, and the aggregate then read as a wash.

**Pre-registered predictions**, recorded now so the result can falsify them:

1. The decode-only arm is **NULL or slightly negative** (|Δ| < 0.01 on `event_argument` strict).
   It is option 3 through a different path, and option 3 lost.
2. The trained arm moves `event_argument` **precision** more than recall.
3. Any gain is **concentrated in type-named roles** — `Location`, `Date`, `Quantity` — and
   absent on `Subject`/`Object`. If a gain appears on `Subject`, the mechanism is not the one
   claimed and the result needs a different explanation.
4. **Bounded payoff.** Only corpora with both entity and event gold can supply the map, so the
   ceiling is set by casie's share of the mixture. This is measurable but not decisive, and
   that was true in §4k before any of it was built.

---

## 6. Cost, and why it is gated behind option 4

Decode-only A/B: one checkpoint, two eval passes, **~$3**. Trained A/B: two arms at the current
recipe, **~$65**, 16–17h each.

**It is gated behind option 4 for a reason that is about evidence, not sequencing.** Option 4
adds extraction supervision and option 2 constrains binding; if option 4 moves the
never-proposed third, the binding population option 2 operates on changes, and a constraint
measured against the old population would have to be re-measured anyway. Running them in the
other order wastes the more expensive one.

**And if option 4 is negative**, option 2 becomes more interesting rather than less — it would
mean the argument gap is binding after all, which is what option 2 addresses directly.
