# Review of `EKF_PIPELINE_VISION.md` — missing pieces and feasibility

Assessed 2026-08-20 against what this programme has measured. The short answer: **the shape
is right and better than what ships, the routing key is wrong in a way we have already
measured, and the front end currently emits nothing routable on real news.**

---

## 1. What the design gets right

Extraction → route → per-track filter is the standard multi-hypothesis decomposition, and it
is a real improvement on what ships today, which is hard assignment on a string key.

*(Updated 2026-08-25: what ships is now `scope_gate.hmm_gate`, a global decode over
{own, aggregate, reject}, which beats the string-key gate on all three events. The review
below predates it and is kept as written.)*

Two details are better than they may look:

- **`0..*` and `1..*` are the right cardinalities.** Zero matching filters is *track birth*;
  several is *multi-hypothesis*. Those are exactly the two mechanisms with a measured value:
  the association headroom on Helene is **+0.111 (18.8%)**, and it splits about evenly
  between reassignment and having a reject option at all.
- **Routing the *text* alongside the events** (added in the latest revision) is right. A
  filter that can re-read its own source can do things the extractor's output cannot express.

## 2. The blocking problem: the routing key is `event type`, and type carries no signal

This is the one measured, load-bearing objection.

**Helene, Katrina, Milton and Maria are all the same event type.** An Event Type Map routes
Katrina's 1,400 to the Helene filter *by construction*, because the field it routes on is
identical for both.

That is not speculation. Ranking by the model's own type energies scored **0/11** on
cross-event contamination, and the reason recorded at the time was that *"the type is RIGHT
there"* — Katrina's 1,400 scores `death toll` 0.95. Type separated unit errors 4/4 with 0/83
false positives and contributed nothing here.

**The map has to be keyed on event *instance*, not type.** Type is a useful coarse pre-filter
— it stops an earthquake figure reaching a hurricane filter — but the discrimination the
pipeline exists for is between two instances of the same type.

## 3. The router is not a component of the unbuilt half; it *is* the unbuilt half

The doc leaves the router open: *"Collaborative Filter maybe or maybe a multi-type classifier"*.

- A **multi-type classifier** is ruled out by §2 — it re-routes on the field that does not
  discriminate.
- **Collaborative filtering** is a mis-fit rather than a bad idea. It infers preferences from
  co-occurrence across a user × item matrix; here there is no user dimension, one event, and
  70 articles. The signal is *content-based* — does this figure belong to this incident —
  which is a matching problem, not a recommendation one.

What the slot actually needs is a **matching function**: score `(figure, context)` against
each live filter's identity and current state, then assign, split or birth. That is MHT's
cost matrix, it has a measured ceiling of +0.111, and **four mechanisms have now failed at
it** — nearest-named-event, only-competitor-named, record-head binding, and type energies.

So the plumbing around the router is a week of work. The router is the programme.

## 4. The front end is not incapable — it is mis-thresholded, and the model is the wrong one

**Corrected twice while writing this section; both corrections matter.**

First, the model. `gliner2-warmstart-natural-clean` (2026-08-15) is a *downstream* arm that has
not been rebuilt. The current base is `gliner2-joint-boundary-mmbert-137k-clean`, retrained
2026-08-19 on repaired data. Measuring the stale downstream model said nothing useful about
the intended front end.

Second, and more important: on the current base the record head looked silent — 6 records over
88 windows — but filled `event` on **100%** of the records it did produce. That is a firing
rate problem, not an attribution one, and it is a threshold artifact:

| `record_anchor_threshold` | records | with `event` | span matched (of 40 windows) |
|---:|--:|--:|--:|
| **0.50 (the default)** | 11 | 11 | **1** |
| 0.30 | 30 | 25 | 4 |
| 0.20 | 65 | 50 | 12 |
| **0.10** | 220 | 160 | **37** |
| 0.05 | 588 | 437 | 78 |

**At the default the model binds 1 window in 40; at 0.10 it binds 37 of 40.** Record decode
carries its own `record_anchor_threshold` and `record_field_threshold`, distinct from the
general decision threshold, and `threshold_sweep` never touches them. They also live on a
**frozen dataclass**, so they cannot be set by assignment — which is the likeliest reason
nobody has swept them, and it is worth fixing so calibration is reachable.

So the front end is viable and the earlier verdict here — "emits nothing routable" — was wrong.
What it needs is **calibration**, and calibration is not free: 0.10 yields 5.5 records per
window, so precision has to be traded back. The operating point is an open measurement, not a
known good.

The caveat that stands: any front-end model must be evaluated on **real wire copy**, because
this programme has repeatedly shipped things that scored well in-domain and at zero on real
news (Track B: 0.532 in-domain, zero on real news).

## 5. Stages the diagram is missing

**A role and plausibility stage, before routing.** The re-audit puts **11.3%** of extracted
`dead` figures at *not casualty numbers at all* — Asheville's population, FEMA flood-insurance
policies, power crews, wellness checks, years read as tolls. Routing a population into a
death-toll filter corrupts that filter no matter how good the router is. Part of this is
already free: the derived tail cut (median + k·MAD on log10) removes the distributional tail
and transfers across events without being told their scale.

**Filter birth.** The diagram reads *"No → continue"*, which drops the observation. It should
create a filter. Worth roughly half the association headroom — and note the caution: the one
implementation tried, innovation-gated track birth, **lost** because it judged each stream
against a reference its own accepted values define.

**Filter death or decay.** Not in the diagram. A stream with no support should age out,
otherwise every spurious birth is permanent.

**Provenance on every bucket entry.** Which extractor, which router decision, which
confidence. The programme has now lost one headline result to an artifact that recorded only
`associate`, and cannot regenerate it.

## 6. The evaluation problem, which gates all of the above

The labels this line has been scored against were audited today and were **27% correct** on
their positive class. Corrected, the whole Helene feed contains **6 cross-event instances**.

You cannot train, tune or validate a router on n=6 from one event. **The routing evaluation
set is itself a deliverable**, and it is the thing multi-source and multi-event feeds produce
as a side effect. It should be built before the router, not after.

## 7. Feasibility

| piece | verdict |
|---|---|
| registry, buckets, routing interface, per-filter reformatting | **feasible, days** — ordinary engineering |
| type-keyed routing | feasible to build, **measured not to work** (§2) |
| role / plausibility pre-stage | **feasible now**, partly already built |
| filter birth and death | feasible; the naive innovation-gated form is measured negative |
| instance-keyed router | **the open research problem**, four failures, +0.111 ceiling |
| mmBERT front end on real news | **viable, needs threshold calibration** (§4) — not blocked |
| evaluation at instance level | feasible, and should come first |

**So: yes, feasible — with one substitution and one reordering.**

Substitute **instance** for **type** as the routing key. Reorder so the evaluation set comes
before the router; the front end needs calibration rather than a rebuild-and-wait.

**The recommended shape is a pluggable router.** Build the architecture with the router behind
an interface and ship the dumb one first — string-key identity, plus the tail cut, plus track
birth. That is roughly what exists today, so it is a refactor rather than a research bet, and
it makes everything *around* the router verifiable while the router stays an open slot.

**And there is one measured lead worth spending on.** The base model retains binding that
casualty fine-tuning destroys — 4/6 catches and 61/82 bound, against 1–2/6 and near-total
abstention for every fine-tuned descendant. Every casualty corpus carries 8 event *types* and
no named identities, so fine-tuning optimises a schema in which the owning event is never
supervised. **A corpus carrying named event identities is the first thing to try**, because it
targets both §4 and §3 at once: it is what could make a boundary model routable at all.

---

## 8. On a purpose-built front-end model (raised 2026-08-20)

The proposal: stop testing the warm-start base as a front end and train one *for* the front
end — events-all plus a full complement of NER, relations, structures and classification.

**Agreed, and the reason is sharper than "it would be nice".** `137k-clean` deliberately
excludes event corpora (`real-synth`, `synth`, `rams`) so that preservation can be measured.
Judging it as an EKF front end judges a model built to answer a different question. Every
"the boundary model can't do X" result in this programme has been measured on either that
model or an unrebuilt descendant of it.

### The one design constraint the measurements actually impose

**Include corpora in which the event's IDENTITY is a supervised field, not only its type.**

Today's re-audit produced the clearest signal this line has: casualty fine-tuning *destroys*
event binding. The never-casualty-tuned base binds an event for 61 of 82 genuine observations
and catches 4 of 6 cross-event; every casualty-tuned descendant collapses to 1–2 of 6 with
near-total abstention. The cause is structural — every casualty corpus carries 8 DocEE event
*types* and no named identities, so `event` is never a supervised field and the ability to
fill it decays.

An events-all mix built from the same corpora will inherit that collapse. **This is the thing
to fix in the data, and it is what makes a front end routable at all** (§2, §3).

### What each task buys the pipeline, concretely

| task | why the EKF front end needs it |
|---|---|
| **structures / records** | the `(dead, location, event)` record **is** the observation. Load-bearing. |
| **NER** | supplies storm names and places — the instance keys a router needs. Signal A runs on it. |
| **events (trigger + args)** | the only formulation carrying `event_key`, which is the routing key |
| **classification** | the relevance gate is a separate `fastino` call today; folding it in removes a model *and* a domain mismatch |
| **relations** | weakest case. A relation can fix place-pairing at best; it cannot express which event *owns* a number. Include for generality, but do not let it drive the mix. |

The pipeline currently loads three models — gate, event, casualty — from two architectures.
One front-end model doing gate + extract removes both the extra load and the mismatch between
stages.

### Three things to build into it from the start

1. **Calibrated record thresholds, in the model card.** §4: at the default 0.5 the current
   base binds 1 window in 40. A front-end model shipped without a calibrated
   `record_anchor_threshold` will look broken to anyone who loads it. Also worth making the
   settings mutable, since the frozen dataclass is why this went unswept.
2. **Real wire copy in the eval, not only held-out synthetic.** Track B scored 0.532 in-domain
   and zero on real news. In-domain numbers do not predict this pipeline's behaviour.
3. **Preservation measured against the front-end job, not in the abstract.** The question is
   not "does adding events cost NER" but "does the mix still read AP copy". Those are
   different questions and only the second matters here.
