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

`fired` is what gate 1 counts. `hit` is whether the figure was the right one.

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
- The failure mode has moved. The rebuild's misses are mostly word-form golds -- `six`,
  `three`, `dozens` -- where it binds a nearby numeral instead (`160`, `70s`, `11`). That is
  a normalisation gap, not a binding failure.
