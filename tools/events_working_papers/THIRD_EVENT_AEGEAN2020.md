# Pre-registration: the 2020 Aegean Sea earthquake as the third event

Written 2026-08-25, BEFORE building the feed, so the result can falsify the hypothesis
rather than be fitted to it.

## Why this event

The collapse (one HMM emission absorbing the date gate, the scope gate and page
furniture) is demonstrated on Helene and **cannot** be validated on Turkiye 2023. The
diagnosis is scale separation between an event and its contaminants:

| event | toll | contaminant | ratio | does rejection pay? |
|---|---|---|---|---|
| Helene | ~230 | Katrina 1,400 | 6x | yes -- false rejections 19.8% -> 9.9% |
| Turkiye 2023 | ~50,000 | Izmit 1999, 17,500 | 3x, and it CROSSES the trajectory | no -- 10695 -> 15763 |
| **Aegean 2020** | **119** | **Izmit 1999, 17,000** | **143x** | predicted yes |

Aegean 2020: 119 dead -- 117 in Turkiye (Izmir province) and 2 in Greece (Samos);
1,053 injured. Contaminants its coverage reliably carries, from the Wikipedia article
itself: Izmit 1999 (17,000), Smyrna 1688 (15,000-20,000), Samos 1904 (4).

## The property that makes it the sharp test, not just another event

**Izmir IS Smyrna.** So "the 1688 Smyrna earthquake killed 15,000-20,000" is:

* an **in-scope place** -- scope membership cannot reject it, by construction
* a **massively out-of-window date** -- only the date feature can
* a **~140x larger toll** -- so magnitude evidence AGREES with the date evidence

That is exactly the configuration the collapse currently FAILS on. Helene's equivalent is
"a pair of hurricanes within a week killed at least 80" keyed to North Carolina in 1916:
in scope, out of window, and the HMM keeps it because one weak date feature cannot
outvote a magnitude of 80 that is perfectly plausible for North Carolina. Here 15,000 is
not plausible for a 119-death event, so the two channels reinforce instead of conflict.

It also gives a genuine two-part cross-border hierarchy (Turkiye/Izmir against
Greece/Samos), the same shape as Turkiye/Syria and Helene's states.

## Pre-registered predictions

1. **The collapse shows a LARGE gain here**, unlike Turkiye's zero. Admitting one 17,000
   reading into a stream whose truth is 119 is catastrophic, so rejection should pay even
   though the feed is sparse. If it does not, the scale-separation hypothesis is WRONG and
   the Helene result needs another explanation.
2. **The date feature carries an in-scope rejection** -- the Smyrna/Izmit figures are
   caught, and caught by `_f_date` rather than `_f_scope`.
3. **Sparsity does not reverse it.** Turkiye's lesson was that dropping accidentally-useful
   readings costs more than impurity. That should not recur, because a 143x contaminant is
   never accidentally useful.

Prediction 1 is the one that matters. It is falsifiable and it is the reason to build.

## Known weaknesses, recorded now

* **Small tolls make the metric noisy.** 119 deaths total; pooled RMSE in deaths will be
  a small number with a coarse trajectory.
* **The Samos stream is ~constant at 2** and almost certainly unscoreable. Expect to score
  Izmir and the aggregate only, which makes this a weaker test of ASSOCIATION than of
  cross-event rejection. That is acceptable -- rejection is what is unproven.
* **Sparse coverage.** A 119-death quake draws far less wire copy than Helene or the 2023
  quake. This is the risk to prediction 3.

## Build plan

Same recipe as the two existing feeds:

1. Ground truth from the **Wikipedia revision history** (dense: 60 revisions in the first
   3.5 hours), taking the infobox casualty figures per timestamped revision. Public,
   versioned and citable -- a better source than the Wayback tracker used for Turkiye.
2. Article feed via the existing harvest path. **Text is NOT committed** -- only source
   URLs, extracted observations and the GT trajectory, per the standing policy.
3. `rollup.json` declaring the hierarchy (izmir / samos under an aggregate).
4. Register in `DATASETS` in `scope_gate_test.py`; everything downstream then works.
