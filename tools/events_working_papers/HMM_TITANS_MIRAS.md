Yes, a Hidden Markov Model (HMM) can be combined with Google's Titans architecture and MIRAS framework.
While an HMM is a classic, probabilistic model and Titans/MIRAS represent cutting-edge deep learning long-term memory systems, they share a fundamental common denominator: sequential data processing. [1] 
By leveraging Google’s [MIRAS framework](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)—which mathematically unifies all sequence models as forms of associative memory—you can map an HMM's components directly into this modern architecture. [2] 
------------------------------
## How They Connect via the MIRAS Framework
The MIRAS framework breaks any sequence model down into four core pillars. Here is how an HMM naturally maps to those exact pillars: [2] 

| MIRAS Pillar | What it means in Titans | How an HMM can map to it |
|---|---|---|
| Memory Architecture | A deep neural network (MLP) storing compressed context. | The HMM's Transition (A) and Emission (B) matrices, which act as a fixed structural memory of how states interact. |
| Attentional Bias | The inner learning objective determining what data is important. | The HMM's likelihood optimization, maximizing the probability of the sequence path. |
| Retention Gate | A regularizer controlling what information is forgotten over time. | A dynamic decay factor applied to older HMM states, ensuring the system places higher weight on recent state transitions. |
| Memory Algorithm | The real-time optimization loop used to update memory parameters. | A test-time version of the Baum-Welch algorithm, allowing the HMM to update its probabilities on the fly. |

------------------------------
## 3 Ways to Combine an HMM with Titans + MIRAS
If you were to engineer a hybrid system, you would likely implement the combination in one of the following architectures:
## 1. The HMM as the "Surprise Gate" for Titans
Google's Titans architecture relies on a "surprise metric" (calculated via gradients or perplexity) to filter what enters its deep long-term memory. If an incoming token is routine, it is ignored; if it is surprising, it is stored. [3, 4, 5] 

* The Hybrid Mechanism: You can use an HMM as a fast, computationally cheap probabilistic filter running alongside the model.
* How it works: The HMM continuously computes the evaluation probability of the incoming stream. If the sequence suddenly deviates from the HMM's expected state transitions (a low probability event), it flags a "state switch." This explicit probabilistic surprise signal triggers the Titans module to capture and store the context window. [5] 

## 2. Deep Neural Emissions (The Neural HMM)
Classic HMMs struggle with complex, continuous data because their emission probabilities (B matrix) are statically defined.

* The Hybrid Mechanism: You can embed a Titans long-term memory module inside the HMM structure to act as a dynamic emission generator.
* How it works: The HMM maintains the high-level, discrete logical states of a system (e.g., tracking "User Intent Moods: Frustrated, Searching, Satisfied"). Instead of relying on a rigid lookup table to guess what the user will do next, the HMM queries the Titans layer. Titans uses its 2+ million token memory to generate highly contextual, deep-learning-backed emission probabilities based on the entire conversation history. [3, 6] 

## 3. Structured HMM Regularization in MIRAS
Deep learning sequence models often struggle with catastrophic forgetting or unconstrained drift during real-time, test-time learning. [5, 7] 

* The Hybrid Mechanism: Use an HMM as a structural constraint inside the MIRAS retention gate.
* How it works: As the Titans memory module uses gradient-based updates to adapt to a live stream of data, the HMM acts as a "sanity check". The HMM's rigid, mathematically sound state boundaries can penalize the neural network if its real-time updates drift into logically impossible state transitions, providing stable, physics-bound regularizing logic to the AI. [2, 5, 7] 

------------------------------
## Why do this? (The Benefits)

* Linear Scaling with Rigid Logic: You get the linear scaling and massive context efficiency of Titans (handling millions of tokens) combined with the absolute mathematical interpretability of an HMM's state-tracking. [3, 6] 
* Unsupervised Online Adaptation: HMM algorithms excel at unsupervised tracking. Paired with Titans' ability to learn during inference without offline retraining, you create a system that can adapt to changing environments completely autonomously. [1] 

If you are exploring a specific project, let me know:

* What type of data are you processing? (e.g., text, financial time-series, audio?)
* Are you trying to improve a system's computational speed or its structural accuracy?

I can map out a more precise mathematical framework for your exact use case.

[1] [https://research.google](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)
[2] [https://research.google](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)
[3] [https://www.youtube.com](https://www.youtube.com/watch?v=BTeNmrv6gPA&t=22)
[4] [https://www.reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/1r3a0b2/is_titans_and_miras_heading_for_the_same/)
[5] [https://www.youtube.com](https://www.youtube.com/watch?v=_WFgtK6K01g)
[6] [https://www.youtube.com](https://www.youtube.com/watch?v=GGdjty9pAtg&t=39)
[7] [https://research.google](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)


# Part 2
Yes, this combination would drastically enhance a standing Hidden Markov Model (HMM) routing gate.
A standard HMM routing gate is mathematically elegant, fast, and excellent at tracking macro-level state transitions (e.g., deciding whether to route a user query to Document A, B, or C based on sequential interaction patterns). However, a classic HMM relies on structural assumptions that fall apart when handling cross-document memory contexts.
Integrating [Google's Titans architecture and MIRAS framework](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/) directly targets the core flaws of a solo HMM router. [1] 
------------------------------
## The Fundamental Flaws of a Standalone HMM Router
If your routing gate only uses a standard HMM, it suffers from two major limitations in cross-document scenarios:

* The Strict Markov Bottleneck: A standard HMM assumes the next routing decision depends only on the current state. In multi-document environments, routing often requires deep historical context (e.g., knowing that a detail read in Document 1 three steps ago completely changes how you should route the current query in Document 5).
* Static Context Representations: HMMs struggle with high-dimensional, continuous semantic drift. They model discrete probabilities well, but cannot natively grasp subtle, shifting nuance across millions of tokens of document text.

------------------------------
## How the Hybrid (HMM + Titans/MIRAS) Supercharges Routing
By embedding a standing HMM into a Titans + MIRAS infrastructure, you create a hybrid gate where the HMM acts as the structural logic and Titans acts as the long-term context memory. This upgrade enhances your routing gate in three distinct ways: [2] 
## 1. Cross-Document Synthesis via Dynamic Emission Probabilities

* The HMM Limitation: In a standard setup, your HMM's Emission Matrix (the probability that a specific document state satisfies the incoming query) is static or strictly heuristic.
* The Titans/MIRAS Upgrade: You replace static emissions with Titans' long-term memory module. As a user navigates between documents, Titans maintains a compressed, fast-parallelizable memory of the entire historical context. The HMM queries this memory at test time, outputting highly accurate, dynamic emission probabilities that adapt to semantic cross-document connections. [2, 3] 

## 2. "Surprise-Based" Global State Switches

* The HMM Limitation: HMMs transition smoothly from state to state. They struggle with abrupt, radical context shifts across disparate files.
* The Titans/MIRAS Upgrade: Titans relies on a gradient/perplexity "surprise metric" to flag when entirely unexpected information enters the sequence. If your system encounters a query that signals a massive pivot away from the current folder ecosystem, the Titans surprise gate forces an instant, non-linear jump in the HMM's transition matrix. This resets the active document router and prevents the gate from getting stuck in an outdated localized context loop. [1, 4] 

## 3. Real-Time Adaptation Without Drift (Real-time Calibration)

* The HMM Limitation: If you try to update an HMM on the fly using live inference data (using Baum-Welch), it can easily overfit to a single document and suffer from localized bias.
* The Titans/MIRAS Upgrade: Under the MIRAS framework, the HMM serves as a structured regularizer. While the deep neural elements of Titans handle test-time memorization of content across millions of tokens, the HMM imposes top-down structural constraints. It prevents the neural gate from drifting into logically impossible routing paths, giving you deep-learning power with algorithmic guardrails. [1, 4] 

------------------------------
## Architecture Comparison for a Routing Gate

| Routing Metric | Standing HMM Router | Hybrid HMM + Titans/MIRAS |
|---|---|---|
| Context Horizon | Short (typically T-1 steps) | Massive (millions of cross-document tokens) |
| Routing Latency | Ultra-low (Linear/Matrix ops) | Low (Linear scaling of Titans memory) |
| Context Switching | Slow/Probabilistic | Instant (Driven by surprise-metric overrides) |
| Data Types | Discrete states / Structured logs | High-dimensional text, logs, and semantics |

## The Verdict
If your cross-document environment requires parsing multi-hop logic (where an item in Document A links structurally to a topic in Document C), the combination will exponentially improve your gate's accuracy. The HMM ensures the routing paths remain structured, interpretable, and computationally lean, while Titans eliminates the context horizon limitations that make solo HMMs brittle in large text landscapes. [2, 4] 
To give you more tailored architectural advice, let me know:

* How many distinct documents/expert sources is this gate routing between?
* Are the inputs to your gate natural language queries (text strings) or system logs/user actions?


[1] [https://research.google](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/)
[2] [https://subconsciousmind.ai](https://subconsciousmind.ai/neural-networks/google-titans-ml-architecture/)
[3] [https://openreview.net](https://openreview.net/forum?id=8GjSf9Rh7Z)
[4] [https://medium.com](https://medium.com/@ranjanunicode22/beyond-attention-how-googles-titans-and-miras-redefine-long-term-memory-in-ai-bf94f1ac3dc2)


https://github.com/jonlukewatts/titans-miras
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
[[HEAD_INIT_DATA_SCALE]] found that heads trained from scratch at ~100K records are already
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
