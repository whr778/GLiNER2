# HMM x Titans/MIRAS, assessed against what we measured

*Parts 1-2 of this note were pasted third-party output describing the idea. They were
retired 2026-09-07 into `PROJECT_HISTORY.md`; the assessment below is this project's own
and rebuts their premise. See the history for what was removed and why.*

---

# Part 3 — Assessment against this project's own measurements

*Added 2026-09-03. Parts 1 and 2 above are a chatbot answer and are kept verbatim as the
source material. This part is the read against what we have actually built and measured.
Titans (Behrouz et al.) and MIRAS are real work and the four pillars above are quoted
correctly; what follows is about whether the proposed hybrids buy us anything.*

## 1. The premise is real — we do have a standing HMM gate

Part 2 addresses "your standing HMM routing gate" as a hypothetical. It isn't one.
`tools/ekf_showcase/scope_gate.py::hmm_gate` is three states decided by **Viterbi over the
whole observation stream**, with hard-EM around the decode because the `own` level is
itself unknown, and it ships as the default in `run_pipeline.py`. Emissions are one-sided
and per-observation evidence from outside the magnitude channel (out-of-window date, place
outside the declared hierarchy, syndication marker) is added to the reject state so those
filters *argue rather than veto*.

## 2. Suggestion #1 was built here and lost

"Use a cheap probabilistic surprise signal to gate what enters memory" is the same shape as
track birth by innovation gating, which [[EKF_MHT_DESIGN]] §4.3 records as built and
**lost** to the fixed magnitude ratio it was meant to replace — nRMSE 0.608 against 0.591,
while degrading the national stream 6.7x. The EKF innovation is a *better*-grounded surprise
statistic than perplexity (a normalised residual with a covariance behind it) and it still
lost. The diagnosis was not signal quality:

> The missing property was not a better birth rule. It is that **the decision has to be
> made over the whole sequence at once.**

`hmm_gate` exists *because* per-observation surprise gating failed. A Titans surprise metric
is also local, so it inherits that failure rather than fixing it.

## 3. But that loss localises to MONOTONE counts — and does not generalise

This is the correction that matters, and it was nearly missed by treating "casualty" as one
regime. A death toll is cumulative and monotone non-decreasing, which is why the shipped
emissions are one-sided ("a rising toll may legitimately exceed the level established so
far, so only a reading far *below* it argues against `own`") and why §4.2 found the
innovation "uninformative about scope on a rising toll." A surprise statistic on a monotone
series is structurally half-dead: it can only fire downward, and downward is not where the
error is — §4.2 measures 63% of the error as stale-and-BELOW truth, which the gate is
"blind to by construction."

Any quantity that legitimately moves both ways restores a two-sided, informative innovation:
financial amounts revised down as well as up (ChFinAnn), displacement and evacuation counts
that fall as people return, active cases as distinct from cumulative cases, containment
percentage, outage counts, territory control (CMNEE), breach scope revised downward (CASIE).

**So the negative result is about monotone cumulative counts, not about gates.** This is
the theoretical structure of the innovation; §4 checks it against what real reporting
actually contains, and narrows it further.

## 4. The regime test was not free to design — it was already built, and already run

Before designing a new experiment, checked whether one already existed. It did.
`scope_gate.py::hmm_gate4` is a fourth Viterbi state, `REV`, for exactly the case §3
describes — a toll that legitimately falls. `revision_state_test.py` tests it on REAL
Helene data, on North Carolina's `dead` stream specifically, because that is documented
ground truth: the official toll really does fall four times as deaths are reclassified.
Reproduced live, 2026-09-03:

```
helene: shipped gate 29.3 | 3-state HMM 20.7 | oracle 17.6
        North Carolina nRMSE: gate 0.518  3-state 0.313

  revise_cost  REV fires   kept  drop ||   POOLED  NC nRMSE
          5.0          0     45    12 ||     20.4     0.313
          6.0          0     45    12 ||     20.4     0.313
          8.0          0     45    12 ||     20.4     0.313
         12.0          0     45    12 ||     20.4     0.313
         20.0          0     45    12 ||     20.4     0.313
```

**`REV` fires zero times at every cost tested, and every score is identical to the 3-state
model.** [[EKF_MHT_DESIGN]] §4.5 already recorded why: "the reports never follow it down.
After each revision the later readings are 230, 230, 230, 1400, 98, 250 while the official
toll falls to 84-123." Outlets do not publish the correction; they repeat the old number or
move on. A correctly-specified state finds nothing, because the text never states the thing
it is looking for. Verdict: **correct, and inert.**

**That result reframes what to test on `missing`, and checking what actually exists there
kills the plan as first stated — before any new emissions code was written.**

1. **No ground truth for `missing` exists on real data.** `datasets/helene2024/ground_truth.json`
   carries exactly one field, `deaths`, sourced from Wikipedia's casualty table. No RMSE
   comparison is possible for this role on Helene, full stop.

2. **The real reported `missing` stream is not a clean decline — it is noise.** North
   Carolina's real extracted observations, in time order:
   `1, 11, 13, 13, 57, 32, 32, 10, 11, 1, 90, 1, 2000`. Across all three roles on the same
   real corpus, consecutive-step decreases are `dead` 20/85 (23.5%), `injured` 19/43
   (44.2%), `missing` 11/28 (39.3%) — comparable NOISE levels, not a role that behaves
   qualitatively differently. The trailing 2000 is almost certainly a different scope
   entirely, which is exactly what the existing reject/scope machinery targets, not a
   two-sidedness question.

3. **The synthetic showcase feed's clean 73.6%-declining `missing` truth series
   (`datasets/ekf_showcase/feed.truth.jsonl`) is an artefact of how that corpus was
   generated**, not evidence real disaster reporting behaves that way. Given (1) and (2),
   testing emissions on synthetic `missing` would measure the corpus generator's prose
   style, and this project has already paid to learn that lesson once (Track B: 0.532
   in-domain, zero on real news, "the corpus is the bottleneck not the formulation").

**So the correction is narrower than "monotone counts break surprise gating, non-monotone
counts do not."** The evidence in hand says: **a correctly-specified state for a real
phenomenon is inert when the reporting itself never surfaces that phenomenon** — a
sourcing problem, not a decode problem, matching §4.5's own conclusion exactly. There is
no equivalent case that `missing` gives the decoder anything real reporting would let it
use; if anything, its real behaviour looks like a scope/contamination problem the existing
gate already addresses, not an argument for a new state.

## 5. Separate the two claims the document bundles

**Claim A — gate on a probabilistic surprise signal.** Plausible for non-monotone
quantities per §3, and cheap: it is a change to the emission model in `hmm_gate`, not an
architecture.

**Claim B — back the gate with long-context associative memory.** The benefit Titans sells
is millions of tokens. Our streams are about a hundred observations — Helene is 106 `dead`
observations — and `hmm_gate` already decodes the entire stream at once. **We do not have a
context-horizon problem at the state layer**, which is exactly where the document proposes
the memory. It would solve a problem we do not have.

## 6. Where a memory could genuinely bind — a different problem

Not the scope gate: **data association**, deciding which stream an observation belongs to.
That is cross-document by nature and it is the piece with measured headroom — §4.1 re-priced
the assignment ceiling from +0.055 to **+0.111 (18.8%)** once the oracle was given a reject
option. If the Titans line is worth anything here, that is the address. Worth being explicit
that it is not the problem Parts 1 and 2 are about.

## 7. Practical blockers for Titans as an encoder

Titans is a sequence-model architecture. Our encoder is mmBERT, chosen for its multilingual
pretraining, and there is no multilingual Titans checkpoint to warm-start from.
[[PAPER_0_FOUNDATION]] §10.7 found that heads trained from scratch at ~100K records are already
the binding constraint; introducing an encoder with no pretrained multilingual weights makes
the worst part of the stack worse, not better.

## 8. Two places the mapping table is loose

- **Attentional bias** in MIRAS is the inner objective the memory optimises *at test time*.
  The HMM's likelihood optimisation is a *training-time* EM objective. Mapping one to the
  other conflates two different loops.
- **Retention gate** in MIRAS is a regulariser on memory parameters, not a decay applied to
  state posteriors. An HMM's Markov property already discards history; adding decay is not
  the same mechanism.

Both are serviceable analogies and neither is an implementation path.

## 9. Where this leaves the state layer

§4 closes the loop this document opened rather than queuing more work on it: the two-sided
emissions idea has a real implementation (`hmm_gate4`), a real test
(`revision_state_test.py`), and a real, already-published result (inert, on real data, for
a documented reason). There is no untested regime left to check on the roles this pipeline
currently tracks and scores — `dead` has truth and was tested; `injured` and `missing` have
no truth on real data to test against at all.

The one genuinely open item this document surfaced is §6: data association, where the
oracle-repriced ceiling (+0.111, 18.8%) says real headroom exists and nothing here has
tested a memory-backed approach against it. That is a different problem from the gate, and
it is the only place left where "give it more context" is an untested claim rather than a
re-run of a measured one.

**The standing caution:** this project's measured pattern is that data changes move the
number and formulation changes do not (Track B: 0.532 in-domain, zero on real news, "the
corpus is the bottleneck not the formulation"; and now revision-state, inert for the same
class of reason — the text does not carry the signal). Nothing in this document should be
budgeted against the state layer. If the Titans line is worth spending on at all, §6 is
where, and even there the first question is whether a real cross-document stream with
ground truth exists to test against — the same question that closed off §4.
