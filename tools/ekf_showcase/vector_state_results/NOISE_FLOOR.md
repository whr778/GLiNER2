# The aggregate-constraint verdict, against a measured noise floor

The recorded table was one RNG stream (`Random(0)`, 40 trials) at one noise level (0.10).
Re-running it reproduces exactly -- which is determinism, not variance. This sweeps **10
independent streams x 40 trials** at three noise levels and reports the spread of the
delta across streams. A delta smaller than that spread is not a result.

`uv run python tools/ekf_showcase/vector_state_test.py --trials 40 --seeds 10 --noise N`

| noise | density | parts-only | vector | delta | spread | readable? |
|---|--:|--:|--:|--:|--:|---|
| 0.05 | 10% | 0.4107 | 0.7370 | +0.3263 | 0.1367 | **CLEARS** |
| 0.05 | 20% | 0.2699 | 0.4676 | +0.1977 | 0.1290 | **CLEARS** |
| 0.05 | 35% | 0.1878 | 0.2774 | +0.0896 | 0.0861 | **CLEARS** |
| 0.05 | 50% | 0.1475 | 0.1863 | +0.0388 | 0.0478 | within floor |
| 0.05 | 80% | 0.1065 | 0.1132 | +0.0067 | 0.0104 | within floor |
| 0.10 | 10% | 0.4460 | 0.6248 | +0.1787 | 0.0806 | **CLEARS** |
| 0.10 | 20% | 0.3114 | 0.4011 | +0.0897 | 0.0659 | **CLEARS** |
| 0.10 | 35% | 0.2325 | 0.2762 | +0.0437 | 0.0678 | within floor |
| 0.10 | 50% | 0.1944 | 0.2090 | +0.0146 | 0.0275 | within floor |
| 0.10 | 80% | 0.1561 | 0.1511 | −0.0050 | 0.0073 | within floor |
| 0.20 | 10% | 0.5279 | 0.6282 | +0.1003 | 0.0770 | **CLEARS** |
| 0.20 | 20% | 0.4126 | 0.4441 | +0.0315 | 0.0526 | within floor |
| 0.20 | 35% | 0.3376 | 0.3493 | +0.0117 | 0.0517 | within floor |
| 0.20 | 50% | 0.2966 | 0.2914 | −0.0052 | 0.0120 | within floor |
| 0.20 | 80% | 0.2514 | 0.2402 | **−0.0112** | 0.0069 | **CLEARS** |

## Three findings the single-stream table could not show

**1. The core rejection holds and is robust.** Where per-state reports are SPARSE (10-20%)
the vector arm loses decisively and clears the floor at every noise level. That is the
regime real feeds are in, and it is the claim the programme acted on.

**2. The middle of the recorded table was never readable.** The published +0.0568 at 35%
and +0.0203 at 50% sit inside floors of 0.0678 and 0.0275. They were quoted as measurements
and they are noise.

**3. At high density AND high report noise, the vector arm WINS, and it clears the floor.**
At noise 0.20 / 80% density: −0.0112 against a spread of 0.0069, 350/400 trials. The
recorded experiment fixed noise at 0.10 and never saw this.

That third result has a mechanism rather than being a curiosity. The aggregate is an
independent measurement of the sum. When per-state reports are precise, it adds little and
mostly injects allocation error -- it constrains the SUM and says nothing about the SPLIT.
When they are noisy, the extra sum-level information outweighs the allocation cost. So the
aggregate constraint is not simply wrong; it is **wrong in the regime measured and right in
a regime that was not.**

## A distributional caveat

Win counts and mean deltas disagree in the middle. At noise 0.10 / 80% the vector arm wins
302 of 400 trials while its mean delta is within the floor: it wins more often and loses
bigger when it loses. Neither statistic alone describes it, and the mean is the
conservative one to quote.
