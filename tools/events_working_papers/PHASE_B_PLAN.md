# Phase B — the beam in the loss

Status: **PLAN, not started.** Written 2026-09-15, after Phase A returned and after the
artefact that made Phase A's headline wrong was found and removed. Companion to
[[JOINT_IE_DESIGN_RECORD]] §7 (which this expands) and [[RESEARCH_PROGRAM]] §3.

---

## 1. Why this is now the interesting arm, and not a consolation prize

The programme's claim is that dense document-level extraction needs **global structured
inference over the boundary candidate scores**, not the greedy per-chunk decode. Phase A
tested the decode-time half of that and came back **null at parity**: all seven heads
inside the ±0.02 floor, at ~2.5× the wall clock.

That is a real negative and it is reported as one. But it is a null about a beam that has
**never been trained for**, and — as of 2026-09-08 — one that had **never been decoded with
its own constraint machinery live** either. Cardinality selects the joint beam's utility
(`logit - absent` against bare `logit`) and its exclusivity slot, and every `dtype: str`
field compiled `ZERO_OR_MORE` because a parameter no caller passed. The scalar half of
decision B was inert for the entire history of the measurement.

So the honest statement of where the thesis stands:

| version of the claim | status |
|---|---|
| decoding **with** the beam beats greedy decode | **tested, null at parity** (Phase A, corrected) |
| decoding with the beam, constraints actually live | **tested once**, same null |
| training **for** the beam beats training for greedy | **never tested** — this document |

Phase B is the only untouched version. It is also the version the literature would
recognise: Phase A compared two decoders over one model trained for neither.

**A null here would close the programme's central question properly.** That is worth
saying plainly: this plan is not written expecting a win. It is written because the
question is currently unanswered, and because a negative from a trained-for beam is a much
stronger result than a negative from a decode swap.

## 2. What Phase A settled, and what it did not

**Settled.** Over one trained model, swapping `decode_mode` moves nothing outside the
noise floor on any head, on two independent lineages (`eb16-rebuild-tr` and
`joint-boundary-mmbert-137k-clean`), and beam width is not a lever — a 4/16/64 sweep moves
structure by 0.0018 and *narrower* is marginally better.

**Not settled.** Whether a model whose loss is computed through the beam learns different
candidate scores. Phase A holds the model fixed by construction; it cannot answer this.

**A trap this document exists partly to prevent:** the width sweep's flatness is sometimes
read as "search capacity does not help, therefore joint inference does not help". It does
not support that. W=1 barely searches and wins, which says the working contrast is
**independent thresholding vs constrained joint selection**, not *greedy vs beam*. Phase B
tests the formulation, so it should be built and reported in those terms.

## 3. The mechanism — and why it is a training-loop change, not a new component

The boundary record head already uses the idiom Phase B needs:

> **assignment on DETACHED scores, loss RECOMPUTED differentiably on the matched pairs.**

In `records.py::compute_dense_batch_loss` the Hungarian matching runs on `cost.detach().cpu()`
— a hard, non-differentiable assignment — and the resulting index pairs are then used to
recompute a differentiable loss from the live logits. Gradient flows through the *scores*,
never through the *argmin*.

Phase B replaces **which assignment** that step produces:

```
                     today (Phase A + greedy training)     Phase B
  assignment         Hungarian over a per-group cost       joint_ie beam over the whole
                     matrix, detached                      JointProblem, detached
  loss               recomputed on matched pairs           recomputed on beam-selected
                                                           pairs, identically
  gradient path      through candidate/assign logits       unchanged
```

Two consequences worth stating because they cut the work down:

1. **No new autograd machinery.** The beam is already a pure scoring-and-selection pass
   over `candidate_scores`; it needs no gradient. The only requirement is that it return
   index pairs in the same shape the matcher returns today.
2. **One objective covers BOTH faces.** An event's role edges and a relation's plain edge
   are the same `EdgeCandidate` in the same `JointProblem` (`RESEARCH_PROGRAM` §1), so
   events and relations are trained by one mechanism rather than two.

**The open design question, and it is not small:** a structured-prediction objective needs
a *contrast*, not just a positive. The cheapest defensible form is a **structured hinge /
margin**: the loss pushes the gold structure's score above the beam's best wrong structure
by a margin, which is exactly what the beam already computes. The alternative — treating
the beam's selection as a soft target — needs a temperature and a second hyperparameter and
should not be the first thing tried.

## 4. Decide FILL vs REJECT before writing any code

Carried from `JOINT_IE_DESIGN_RECORD` §7 and repeated here because it is the one decision
that silently invalidates the comparison:

`decode_group`'s scalar path runs `if chosen is None or chosen == 0: continue` **before**
the cardinality check, so when `ABSENT` wins the argmax a `REQUIRED_ONE` field is simply
left empty — exactly as an optional one is. Greedy **fills**; it never rejects an instance
for an unfilled required role.

Building `RequiredRoles` as a **rejection** constraint would therefore make the beam
stricter than the decoder it is measured against, and the resulting precision/recall gap
would read as a property of global decoding rather than of the constraint. **Only `fill`
preserves comparability.** If a rejection variant is ever wanted, it is a third arm, not a
default.

## 5. Pre-registered bars

Written before the run, per the standing rule, and stated so a null is publishable.

**Primary.** `relation` and `event_argument` strict micro F1 on the shared blind test,
against a control trained identically with the greedy objective. The bar is **> ±0.02**
(the measured single-run floor). Relation's own floor is **±0.041** on `mix_natural`, so a
relation claim needs either that margin or a second seed — and the honest choice is the
second seed.

**Secondary, and read as a cost not a win.** Wall-clock per epoch and the record head's
throughput (§7). A treatment that wins by 0.03 and trains 5× slower is a result, not a
recommendation.

**Guard heads.** `entity`, `event_type`, `classification`. The structured objective touches
edges; if the entity head moves materially, the change has leaked and the run is
uninterpretable rather than positive.

**Two seeds per arm, minimum.** Every one-seed curve in this project is suspect
(`TODO.md`), a control re-run of a published recipe once scored +0.023 above it, and this
programme has already retracted findings for exactly this reason.

## 6. The instrument requirements, which are not optional here

Three boxes and ~$4.20 were spent in September 2026 on runs that measured nothing, and not
one failure was the mechanism under test. Phase B is a bigger spend and inherits the rules
those failures bought:

- **The run must PROVE the treatment executed**, in its own log, before any metric is read.
  For Phase B that means printing which assignment path produced the matched pairs, per
  arm, per epoch — not inferring it from the numbers the change was supposed to move.
- **The gate must be able to FAIL.** A gate that samples one batch is not a gate; it must
  be deterministic and per-arm. Both arms print it, because a line that appears only in the
  treatment proves nothing about the control.
- **Verify on the TRAINING path, never on inference.** Three separate defects in September
  were "verified on the inference path, assumed training matched". Phase B is a training
  change; every check belongs in `collate_fn_train` and the loss, not in `extract`.
- **`GLINER2_STRICT_ATTN=1`, and confirm it can still raise.** On bf16 ModernBERT an sdpa
  fallback is a correctness failure, and the guard was unreachable by construction until
  2026-09-15.

## 7. The risk most likely to kill this, and it is not the science

**The record head trains at ~4.6 samples/s against the curve's 22** on the same H100 and
the same model (`JOINT_IE_DESIGN_RECORD` §7, 2026-08-10, unresolved). The signature —
idle GPU, one pegged core, idle dataloader workers — is many small GPU kernels behind heavy
Python-side work, and the record path is the prime suspect by elimination.

Phase B *adds* a Python-side selection pass per batch. If the beam is run naively inside
the training loop, the sensible prior is that it lands somewhere between 5× and 10× the
greedy objective's wall clock, which prices a real comparison out of reach.

**So the first increment is a throughput probe, not an experiment:** run the beam over one
batch's `JointProblem` at training shapes and time it against the Hungarian matcher it
replaces. If it is worse than ~2× the matcher, the plan changes — batch the beam across
groups, or cap it — before any arm is trained. **Do not skip this to "just try it".**

## 8. Cost model

| step | what | cost |
|---|---|--:|
| B0 | Throughput probe, one batch, CPU or one GPU-minute | **~$0** |
| B1 | Wire beam-assignment into the loss behind a config flag; unit-test that it returns matcher-shaped pairs | $0 |
| B2 | Smoke: 200 steps, both arms, one seed — does loss descend, does throughput hold | ~$2 |
| B3 | Real arms: 2 seeds × 2 arms on a mid-size corpus | ~$25–40 |
| B4 | If positive: repeat on the second lineage before quoting | ~$25 |

**B0 through B2 are ~$2 and gate everything after.** Nothing beyond B2 should be booked
until B0's number is known.

## 9. What would falsify the thesis

Stated so the negative is as publishable as the positive:

> Trained-for-the-beam matches greedy-trained-and-greedily-decoded on relation and
> event_argument, within the noise floor, at two seeds — while costing materially more
> wall clock.

That result, combined with Phase A's decode-time null, would say the programme's claim is
**not supported at the scales and corpora tested**, and would localise the remaining
possibility to regimes this project has not reached (denser documents, larger candidate
sets). That is a legitimate contribution and should be written up as one rather than
buried.

## 10. Open questions this plan does not answer

- **Which corpus.** Re-DocRED is the dense-relation case the thesis is about; RAMS is the
  event case. Doing both doubles B3. Undecided.
- **Margin form.** Structured hinge is proposed above as the cheapest defensible contrast,
  not as a settled choice.
- ~~**Whether the cardinality regime change should land first.**~~ **ANSWERED 2026-09-15:
  it does not need to.** The A/B came back NULL -- structure strict 0.7388 -> 0.7326,
  -0.0061, inside the floor, with precision up 0.020 and recall down 0.022 cancelling. So
  the record head's loss is NOT a moving target underneath Phase B, and the sequencing
  question dissolves. One seed on one structures-only corpus, so this closes the
  *sequencing* worry rather than the regime question itself.

- **Whether `event_records` changes what Phase B is measuring.** NEW, and more live than
  the one it replaces. The event-capable base (`eb16-eventrecords-tr`, training 2026-09-15)
  routes events through the RECORD head instead of the mention path. Phase B's structured
  objective operates on role edges, and which head produces those edges is not a detail. If
  that base lands, Phase B should be built against it rather than against a mention-path
  base -- otherwise it optimises a decoder for a representation the shipping model no
  longer uses.
