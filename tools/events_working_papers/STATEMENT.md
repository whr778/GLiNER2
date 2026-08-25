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

**MHT is still not built, and now it should not be — the residual it was aimed at has been
taken by something cheaper.** (Updated 2026-08-25; the paragraphs below are the position
that led there and are kept.)

The specified answer was a hypothesis tree with a Hungarian cost matrix and track
birth/death. What ships instead is a **global Viterbi decode over three states — own place,
aggregate, reject** — which improves every event we have at one setting: Helene 29.3 → 20.7
(−29.4%), Türkiye–Syria 11,581.5 → 10,695.5 (−7.6%), Aegean 2020 74.4 → 15.7 (−78.8%). The
oracle below predicted exactly the two properties that made it work, and neither of them is
a hypothesis tree: the decision has to be **global** (a greedy rule commits per observation,
and one large figure admitted early poisons a stream's scale for everything after), and it
has to be able to **reject** (assignment headroom is measured at zero, so a decoder without
a null hypothesis has nothing to win). Hungarian assignment solves the half worth nothing.

**MHT is not built here, deliberately — but the number that justified that has been
corrected, and one piece has since been built and lost.**

Measured 2026-08-11 against an oracle that assigns every observation to the scope it actually
fits, perfect association looked worth **+0.055** (9.3% relative: 0.591 shipped against 0.537
oracle). That oracle is *two-way*: it can only move an observation between its own place and
the national total, so it has no home for a figure belonging to no scope in the event, and it
scores Katrina's 1,400 as badly as the shipped gate does. The tell was visible and misread —
on two states the gate already beats the "perfect" assignment, because it has a third option
the oracle lacks: *drop*.

Track birth/death **is** a null hypothesis, so re-priced with a reject option the ceiling
moves to **+0.111 (18.8%)**, roughly half of which needs the null hypothesis and half of which
does not (2026-08-19). We then built the cheapest piece that provides one — track birth by
innovation gating — and **it loses to the fixed magnitude ratio it was meant to replace**,
because judging a stream against its own track is circular: every contaminant the track
accepts moves the reference the next test uses. See [[TODO]] item 6 and
[[EKF_MHT_DESIGN]] §7.1–7.2.

A hypothesis tree, cost matrix and Hungarian assignment for an 18.8% residual on a
single-source feed is still not justified. What that failure *did* produce is the first
evidence **for** deferred assignment, which addresses the circularity directly.

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
