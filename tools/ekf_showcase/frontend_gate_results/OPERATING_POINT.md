# The 0.001 operating point is dead, and the answer is ~0.02

`BINDING_ACCURACY.md` recommended running the extractor wide open at 0.001 (yield 81.7%,
1.85 candidates per window) and letting a downstream selector choose. That recommendation
was made on windows built AROUND a gold `dead` observation -- every one contains a real
toll -- so it measured recall and nothing else. Measured on toll-free windows, it fails.

## False positives on windows with no death toll

146 numeric 400-char windows from the 26 feed articles carrying no `dead` observation;
143 clean, 3 excluded as SUSPECT (a death word near a number, so the heuristic may have
missed a real toll rather than the model inventing one). Negative windows must contain a
number, or the test is "can it avoid inventing a figure" rather than "can it avoid calling
the wrong figure a death toll".

| threshold | clean FP% | spans/win | irrelevant-article FP% | numeric-only FP% |
|---|--:|--:|--:|--:|
| 0.100 | 4.2% | 0.05 | 4.7% | 4.2% |
| 0.050 | 8.4% | 0.09 | 9.4% | 7.7% |
| 0.020 | 25.9% | 0.31 | 23.4% | 23.8% |
| 0.010 | 46.9% | 0.57 | 39.1% | 44.8% |
| 0.005 | 59.4% | 0.81 | 53.1% | 56.6% |
| 0.001 | **83.9%** | 1.52 | 78.1% | 74.1% |

**At 0.001 the model puts a death toll on 84% of windows that have none.** It binds `dead`
to `"6 feet"`, `"sea water"`, `"15-foot"`, `"over"`, `"at"`.

**Dropping non-numeric spans does NOT rescue it** -- 83.9% -> 74.1% at 0.001 and 46.9% ->
44.8% at 0.010. Most false positives are numbers, just the wrong ones. (A first look at six
sample windows suggested otherwise; the sample was unrepresentative.)

## Both halves, at the feed's real composition

The test sets are 60 positive against 143 negative windows, but the feed is 106 `dead`
observations against 486 numeric windows -- a **3.6:1** negative:positive base rate, not
2.4:1. Reweighted to that:

| threshold | recall | FP rate | TP | FP | precision | F1 |
|---|--:|--:|--:|--:|--:|--:|
| 0.100 | 15.0% | 4.2% | 16 | 16 | 49.9% | 0.231 |
| 0.050 | 26.7% | 8.4% | 28 | 32 | 47.0% | 0.341 |
| **0.020** | **50.0%** | **25.9%** | **53** | **98** | **35.0%** | **0.412** |
| 0.010 | 56.7% | 46.9% | 60 | 178 | 25.2% | 0.349 |
| 0.005 | 71.7% | 59.4% | 76 | 226 | 25.2% | 0.373 |
| 0.001 | 81.7% | 83.9% | 87 | 319 | 21.4% | 0.339 |

**Best operating point is ~0.02**, not 0.001: half the tolls found, at 35% precision.
0.001 buys 32 more true tolls at the cost of 221 more false ones.

**Precision never exceeds 50% at any threshold.** Roughly half of everything this extractor
calls a death toll is wrong, everywhere on the curve. The selector downstream is not
optional polish -- the pipeline cannot work without one, and the per-event plausibility
ceiling now has a measured job description: reject ~2 wrong tolls for every right one at
0.02, or ~4 at 0.001.

## Caveats

- 3 of 146 negatives were excluded as suspect; at 0.001 their FP rate is 33.3% (1 of 3),
  which is noise at that n and cannot move the conclusion.
- Negatives come only from articles with NO `dead` observation anywhere. Toll-free windows
  inside toll-bearing articles are not represented, and those are plausibly harder -- the
  surrounding article is about deaths. The real FP rate is likely WORSE than this table.
- All observations here are `mode: heuristic`. There is no human gold in this feed.
