# The plausibility ceiling supersedes the muting arm

Added and swept 2026-08-20, following the false-positive audit. A declared per-event ceiling
drops any observation above the largest credible toll for that event, before gating.
Implemented in `scope_gate_test.py` as `plausibility_filter` / `--max-plausible`.

Hurricane Helene killed on the order of 230 people, so a five- or six-figure figure in any of
its streams is not a casualty count. That is prior knowledge, declared per event — the same
kind of statement as the `hierarchy` block already in `rollup.json`.

## Swept, not picked

Ungated per-place mean nRMSE. All runs on the same pipeline settings.

| ceiling | docee | control 4ep | muted |
|---:|--:|--:|--:|
| off | 378.809 | 46.844 | 19.822 |
| 20,000 | **18.287** | 26.961 | 19.822 |
| 5,000 | 18.287 | 22.289 | 19.822 |
| **2,000** | 18.190 | **5.853** | **6.194** |
| 1,000 | 18.190 | 5.057 | 6.194 |
| 500 | 18.190 | 5.057 | 6.194 |
| 250 | 18.190 | 5.019 | 6.194 |

**One observation is worth 20x on the production model**: dropping the single 94,000 takes
docee from 378.809 to 18.287.

## The finding: the arm's advantage does not survive it

At a ceiling of 2,000 — roughly nine times Helene's true toll, so generous rather than tuned:

| | dead obs | ungated | gated @2.0 |
|---|--:|--:|--:|
| control 4ep | 187 | **5.853** | **3.336** |
| muted | 106 | 6.194 | 3.729 |

**The control is better on both.** Muting won the unprotected comparison 46.844 → 19.822, and
that entire gain is recovered — and exceeded — by a threshold that needs no model, no
training and no GPU.

Worse for the arm: **the control scores better while carrying 81 more observations.** The
ceiling removes only junk; muting removed genuine signal alongside it. Per state at the
ceiling, muting still wins North Carolina (0.640 vs 4.727) but loses Georgia (6.805 vs 2.819)
and South Carolina (3.689 vs 2.021).

And the gated column never moves at any ceiling, in any arm — the scope gate was already
removing these values by magnitude. The ceiling matters only where nothing else is defending.

## So guard 1 fails once the cheap baseline exists

Guard 1 was pre-registered as ungated per-place mean, want lower than control. Muting passes
against an *undefended* control and fails against a control with a one-line ceiling. Since
nobody would ship the undefended configuration, **the honest reading is that the muting arm
is superseded**, not that it works.

This does not retract what the arm demonstrably did: it suppressed 15 of 20 large false
positives, and the suppression is real and learned. It says that the cheapest possible
alternative does the same job better, which is the comparison that decides whether to keep it.

## What the ceiling still cannot do, unchanged

Set to 2,000 it leaves Katrina's 1,400, 1,500 troops and 8,000 crews untouched, because all
three are plausible magnitudes. Set low enough to catch Katrina it stops being a plausibility
test and becomes the magnitude gate again — rejecting a cross-event toll for being large
rather than for belonging to another storm. **Cross-event is still unsolved, by this and by
the muting arm alike.**


---

# How the threshold is decided — and the answer changed

## The hand-set ceiling was chosen from the answer

Everything above uses 2,000, justified as "about nine times Helene's true toll, generous
rather than tuned". That justification **requires knowing the true toll**, which is the
ground truth being scored against. It is the same post-hoc criticism this programme already
levelled at the scope-gate ratio, and it is worse than it looks in two ways.

**The plateau only exists downward.** Below 2,000 the score is flat; above it degrades fast.
An *uninformed* ceiling of 20,000 — "surely no event killed more than that" — recovers only
26.961 of the control's 46.844 → 5.853, about 42% of the gain. The good result depends on the
informed choice.

**And it does not transfer at all.** Held out on Türkiye–Syria, whose true toll is ~41,000:

| ceiling | dropped | kept | streams | ungated |
|---:|--:|--:|--:|--:|
| off | 0 | 91 | 2 | 1.815 |
| 2,000 | 80 | **11** | **1** | 0.703 |

The Helene ceiling deletes 80 of 91 observations *including every one of Turkey's true
41,000s*, empties Syria's stream entirely — and the reported mean **improves**, because it is
then an average over one stream where the baseline averaged two. The random-removal control
does **not** catch this; it reports "selecting". Only the kept-count and the stream-count do,
and both are now printed.

## The fix: derive the cut from the event, do not choose it

Suggested by the user: reject the distribution's tail instead of a fixed value. Implemented as
`tail_cut` / `tail_filter` — **median + k·MAD on log10, upper tail only**, pooled over the
event's own observations. Three choices, each measured:

- **log10**, because values span 1 … 129,933 and the false positives are orders of magnitude
  out, not a few sigma out;
- **median/MAD**, robust in principle — though measured against mean/stdev on this data the
  two are nearly identical, so the log transform is doing most of the work. A prediction that
  masking would matter here was wrong;
- **one-sided**, because contamination is documented as one-directional and a toll of 1 or 2
  is legitimate.

At **k = 1** the derived cuts reproduce every best hand-set result without being told any
event's scale:

| event / arm | derived cut | ungated | best hand-set ceiling |
|---|--:|--:|--:|
| Helene, archived | 351 | 4.283 | 4.283 |
| Helene, `casualty-docee` | 352 | 18.190 | 18.190 |
| Helene, control 4ep | 516 | 5.057 | 5.057 |
| Helene, muted | 337 | 6.194 | 6.194 |
| **Türkiye–Syria** | **47,622** | **1.815, unchanged — 0 dropped, both streams intact** | destroyed the event |

The cut sits above each event's true peak and below its junk, on two events whose scales differ
by two orders of magnitude, and neither was chosen.

**k is a genuine knob and k = 0.5 is over-trimming.** It gives a much better ungated score
(control 0.846) by cutting at 134 — below Helene's national total of 230 — and the tell is
that the *gated* score gets worse (1.584 against 3.336). Selecting k by "the largest cut at
which gating does not degrade" needs no knowledge of the toll.

## Streaming: the CLT intuition holds, but only with the self-reference removed

The cut should update as observations arrive. The naive version — recompute on the values
**accepted so far** — collapses:

    helene, cut recomputed on the accepted set:  n=8 -> 3, and it never recovers
                                                 88 of 106 rejected, including 30, 32, 44, 50

A death toll *starts small*, so the early sample is not a small sample of the final
distribution, it is a biased sample of its low end. The cut locks onto it and then rejects the
legitimate growth. **This is the same self-reference that defeated M5 track birth** — a
mechanism judged against a reference its own decisions define.

Pooling over every observation **seen**, accepted or not, fixes it and converges to the batch
answer exactly:

    helene archived   n=8 -> 3,    25 -> 84,    50 -> 272,  100 -> 339   final 351  (batch 351)
    helene control    n=8 -> 15,   25 -> 31,    50 -> 93,   200 -> 478   final 516  (batch 516)
    Türkiye           n=8 -> 2827, 25 -> 24404, 50 -> 38155              final 47622, keeps 41,020

Residual cost, stated: during warm-up the cut is genuinely too low, so the streaming form
rejects some real mid-range readings that the batch form keeps. Retrospective scoring should
use the batch cut; a live tracker cannot, and pays that price.

## What does not change

The muting arm is still superseded. Under the derived cut at k = 1 the control beats it
**5.057 against 6.194**, the same verdict the hand-set ceiling gave. And both cross-event
tolls survive the tail cut exactly as they survived the ceiling: at a cut of 516, Katrina's
1,400 is gone for being *large*, not for belonging to another storm — and on Türkiye, where
the cut is 47,622, an equivalent cross-event figure would sail through.


## It is anomaly detection, which is also the statement of its limit

`median + k·MAD` with the 1.4826 scaling is the **Hampel identifier**; applied per point it is
the **modified z-score** (Iglewicz & Hoaglin). Standard ground, not invented here.

It is the **third** anomaly detector this line has built, and the comparison is the useful
part:

| mechanism | form | outcome |
|---|---|---|
| scope gate (§5) | ratio against an external reference stream | works, 5.247 → 0.591 |
| M5 track birth | normalized innovation squared — the classical form | **lost**, self-referential |
| tail cut | Hampel on log10, pooled over the event | works, and transfers |

The two that work judge against something the contaminant cannot move — a *larger scope*, or
the *whole event's* distribution. The one that lost judged each stream against its own
history. That is the single lesson these three share.

**And anomaly detection cannot reach what is left, by construction.** Against the five audited
cross-event contaminants:

| cut | catches |
|---|---|
| Helene, archived (351) | **1 of 5** — Katrina's 1,400 only |
| Helene, control (516) | **1 of 5** |
| at Türkiye's derived scale (47,622) | **0 of 5** |

Milton's 230, the typhoon's 250, Bosnia's 16 and Mexico's 2 are *statistically ordinary*
Helene casualty figures. They are wrong because of **identity**, not magnitude, and no test on
the value's distribution can see that. The single catch is Katrina's 1,400, removed for being
large rather than for being Katrina — and at Türkiye's scale even that becomes invisible.

The same holds for the 1,500 troops and 8,000 power crews: plausible magnitudes, living people
in the affected area, wrong only in *role*.

So the honest scope of this mechanism: it cleans the distributionally weird tail, cheaply and
transferably, and is structurally incapable of the rest. Anything further needs a
**conditional** judgement — is this number a casualty count, of *this* event, in *this* place —
which is not anomaly detection on a value. It is the binding problem, and it is what the
programme's remaining two halves are about.
