# Is the bound figure the RIGHT figure?

Measured 2026-08-24, local CPU, free. 60 casualty-bearing Helene windows; the gold death
toll is located by character offset, and a `dead` argument counts as a hit when its span
overlaps that range. Both models at **matched** thresholds -- never best-over-range.

    uv run python tools/ekf_showcase/binding_accuracy.py <model> --device cpu

| | rebuild `ekf-frontend` | | | | incumbent `137k-clean` | | | |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **threshold** | fired | hit | prec | yield | fired | hit | prec | yield |
| 0.50 | 3 | 3 | **100.0%** | 5.0% | 0 | 0 | -- | 0.0% |
| 0.40 | 4 | 4 | **100.0%** | 6.7% | 0 | 0 | -- | 0.0% |
| 0.30 | 5 | 4 | **80.0%** | 6.7% | 4 | 0 | **0.0%** | 0.0% |
| 0.20 | 6 | 4 | **66.7%** | 6.7% | 10 | 0 | **0.0%** | 0.0% |
| 0.10 | 12 | 9 | **75.0%** | **15.0%** | 39 | 3 | **7.7%** | 5.0% |
| 0.05 | 20 | 16 | 80.0% | 26.7% | 55 | 16 | 29.1% | 26.7% |

`hit` is whether the figure was the right one. `fired` is gate-1-like but NOT identical:
gate 1 accepts *any* bound role, this counts `dead` only. They coincide for the incumbent
(39 = 39 at 0.1) and not for the rebuild (12 here against gate 1's 15), so read `fired` as
"committed to a death toll", not as gate 1's number.

## This reverses the gate 1 verdict

Gate 1 scored the incumbent at 65% and the rebuild at 25%, and that was read as "the
rebuild is worse on the copy it was built for". It is not. **The incumbent's 39 firings at
threshold 0.1 contain 3 correct death tolls.** What it binds to `dead` instead is
`"car Hurricane Helene"`, `"Mexico"`, `"Pacific coast"`, `"Carolinas"`, `"Florida"` --
locations and sentence fragments, not numbers.

At every matched threshold the rebuild's yield is >= the incumbent's, and at 0.1 it is
**3x** (15.0% vs 5.0%) off **one third** the firings. The two only converge at 0.05, where
the rebuild reaches the same 16 hits from 20 firings that cost the incumbent 55.

**Gate 1 counts form and cannot see this.** Scored best-over-a-range it actively rewards a
model for firing indiscriminately at a permissive threshold, which is exactly the
comparison error MODEL_LINEAGE's "matched thresholds" caution exists to prevent -- applied
to A/B arms but never to the gate itself.

## What is still true

- **Gate 2 still fails for both.** Neither model separates two same-type events; the decode
  pools them into one instance. That is the live blocker, and this measurement does not
  touch it.
- **Yield is still low in absolute terms.** 15% at 0.1, 26.7% at 0.05. The rebuild is the
  better extractor and still not yet a good enough front end.
## The remaining misses are NOT a normalisation gap (measured 2026-08-24)

A first reading of three miss lines called this a word-form normalisation gap. It is not.
Splitting the 60 windows by gold form at threshold 0.1, and asking whether ANY span of ANY
role overlaps the gold:

| gold form | n | correct | role-miss | recognition-miss |
|---|--:|--:|--:|--:|
| digit | 36 | 8 | **0** | 28 |
| word | 24 | 1 | **0** | 23 |

**Zero role misses.** The model never finds the figure and files it under the wrong role --
when it misses, it emits nothing overlapping the gold at all. So there is nothing for a
normaliser to normalise, in the results or in the gate. Word-form golds are worse (1/24 vs
8/36) but most misses are plain digits.

## It is an operating point, and then a ranking problem

Extending the sweep down (spans deduped across event types -- one span found under three
event types is one candidate):

| threshold | fired | hit | yield | `dead` candidates | span precision |
|---|--:|--:|--:|--:|--:|
| 0.100 | 12 | 9 | 15.0% | 15 | 60.0% |
| 0.050 | 20 | 16 | 26.7% | 25 | 64.0% |
| 0.020 | 39 | 30 | 50.0% | 52 | 57.7% |
| 0.010 | 46 | 34 | 56.7% | 62 | 54.8% |
| 0.005 | 58 | 43 | 71.7% | 81 | 53.1% |
| 0.001 | 60 | 49 | **81.7%** | 111 | 44.1% |

**The gates were reading this model at 100x too high a threshold.** Yield triples from 0.05
to 0.001 while span precision falls only 64% -> 44%, and the candidate list stays at **1.85
spans per window**. The right toll is present in 82% of windows.

So confidence barely separates the correct toll from its distractors -- span precision is
flat-ish across a 100x range, which means you cannot buy precision by thresholding. But with
under two candidates per window, that is a **selection** problem, and this pipeline already
owns a selector: the declared per-event plausibility ceiling that beat the trained muting
arm. Run the extractor wide open and let the ceiling choose.

**Caveat, and the next free measurement.** These 60 windows are built AROUND a gold `dead`
observation, so every one contains a real death toll. This says nothing about what the model
emits on windows with no toll in them, which the router will also see. Measure false-positive
behaviour on non-casualty windows before acting on the 0.001 operating point.
