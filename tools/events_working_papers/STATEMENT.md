# Statement of derivation and influence

This work is derived from and influenced by:

> Kozak, M. C. *Multiple Model Methods for Cost Function Based Multiple Hypothesis
> Trackers.* Air Force Institute of Technology, 2012.
> [Semantic Scholar](https://www.semanticscholar.org/paper/75221aef94167cb5797428d55cb01b826283e840)

It is the closest prior work to this design, and the debt is specific rather than general.
Two of the choices here are the same two choices Kozak makes, in the classical setting:

- a **cost-function hypothesis score**, which is what the tracker in [[EKF_MHT_DESIGN]] §3
  optimizes;
- a **bank of models arbitrated per hypothesis**, which is the classical form of the §4
  mixture-of-experts gate. The learned router over `{local read, tracked state}` experts in
  §15 is a learned instantiation of what the classical treatment does with fixed IMM
  mixing.

We also design against the negative Kozak reports rather than only the positive result.
During *deferred decision periods* — when the mixture mean drifts far from true target
position — the multiple-model structures accumulate **greater** RMS error than a single
filter. That is a direct prediction about our gate: a router is a liability exactly when
the mixture is bimodal and no expert is yet right. It is recorded in §22 as something to
probe before the gate is trusted, and it is **not yet tested**.

## On MHT itself

**MHT is not built here, deliberately.** Measured 2026-08-11 against an oracle that assigns
every observation to the scope it actually fits, perfect association is worth **+0.055**
(9.3% relative: 0.591 shipped against 0.537 oracle) — and on two states the shipped scope
gate already beats a perfect two-way assignment, because it has a third option the oracle
lacks: *drop*. Building a hypothesis tree, cost matrix, Hungarian assignment and track
management for a 9% residual is not yet justified. See [[TODO]] item 6.

We intend to revisit it. The regime where MHT would earn its keep is **multiple sources
that disagree and revise about one event**, which is genuine association ambiguity in a way
one wire service's copy is not. That is [[TODO]] item 7, and it is the precondition, not
the schedule.

## Why this work may build on it freely

Kozak's thesis is a work of the U.S. Government, produced at the Air Force Institute of
Technology, so no copyright subsists in it. The reprint edition states this on its back
cover:

> This work has been selected by scholars as being culturally important, and is part of the
> knowledge base of civilization as we know it. This work was reproduced from the original
> artifact, and remains as true to the original work as possible. Therefore, you will see
> the original copyright references, library stamps (as most of these works have been
> housed in our most important libraries around the world), and other notations in the
> work.
>
> This work is in the public domain in the United States of America, and possibly other
> nations. Within the United States, you may freely copy and distribute this work, as no
> entity (individual or corporate) has a copyright on the body of the work.

Public domain removes the legal question, not the intellectual one. This statement exists
because the influence should be known regardless.
