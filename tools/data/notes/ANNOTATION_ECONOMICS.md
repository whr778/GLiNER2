# Pricing an annotation purchase

Annotation is billed per DOCUMENT and valued per POSITIVE, so **the filter's precision is
the price**. This records the method and the numbers from the Turkish extractor purchase
(2026-08-29) so the next language does not rediscover them.

Reproduce with:

    uv run python tools/data/gate_purity_curve.py --model whr778/gliner2-gate2-mmbert-tr

## The method

1. **Get a labelled sample OF THE POOL.** Not of a similar corpus -- of the pool itself.
   The held-out eval set was drawn from the same shards and outlets, so its label
   distribution IS the pool's.
2. **Score that sample with every candidate filter**, and report purity, recall, and the
   resulting price side by side.
3. **Check pool feasibility, not just price.** A high cut can be unreachable: purity rises
   while recall falls, so the positives that survive may be fewer than the target however
   much you are willing to spend.
4. **Only then buy.**

## Two errors this caught, both would have been paid for

**Wrong base rate.** The first price used 42.3% positives, measured on the TRT Haber
pilot. The multi-outlet pool is **25.1%**. That alone moved the unfiltered cost of 30,000
positives from $95 to **$160**.

**Assumed filter purity.** The estimate assumed the gate would pre-filter to ~75% purity.
Measured, the gate reaches **37.3%** at a cut the pool can support. Nobody had checked;
the number was invented to make a total.

## The measured table, 3,597 adjudicated documents, 903 positive (25.1%)

Batch pricing, Haiku 4.5: $0.50/M in, $2.50/M out; ~1,373 input + 260 output tokens/doc.

| strategy | purity | recall | gate must score | $ for 30K positives |
|---|--:|--:|---|--:|
| no filter | 25.1% | 100% | none | $159.71 |
| regex only (free) | 40.8% | 83.7% | none | $98.22 |
| gate cut 0.9 | 37.3% | 91.9% | full pool | $107.58 |
| gate cut 0.9999 | 52.7% | 64.7% | full pool | $76.07 |
| gate cut 0.99999 | 65.3% | 42.0% | full pool | $61.36 |
| **regex then gate 0.9999** | **70.1%** | 56.3% | half pool | **$57.22** |
| **regex then gate 0.99999** | **78.8%** | 37.8% | half pool | **$50.91** |

## Why compose a free regex with the model

The regex is Turkish death/injury words within 60 characters of a numeral (allowing scale
words, so "29 bin 313" counts). It is in `gate_purity_curve.py` as `TOLL_NEAR`.

- **Alone it beats the gate's usable cut** -- 40.8% purity against 37.3% -- for nothing.
- **It halves what the model must score**, and scoring is the expensive half: 6 docs/s
  locally means a 300k pool is ~13 hours, or ~$4 of GPU.
- **Composed, they are better than either** -- 78.8% purity, and the cheapest column in
  the table.

The coarse filter should be cheap and high-recall; the model should be expensive and
high-precision, and should only ever see what survived the cheap one.

## Traps

- **Read purity beside recall, always.** A cut with excellent purity that discards 62% of
  positives needs a proportionally larger pool. `POOL TOO SMALL` in the tool means the
  target cannot be reached at that cut at any price.
- **Expanding the pool can be CHEAPER than buying at a lower cut.** Shards cost only
  download time, and a bigger pool lets a purer cut be used. Pulling 8 -> 18 shards moved
  30K positives from $107 to ~$51.
- **A saturated model has no usable threshold.** This gate's confidence saturates on
  English (identical Helene RMSE from 0.5 to 0.9999). Check that the purity column
  actually MOVES with the cut before treating threshold as a lever.
- **Batching is not automatically faster.** On MPS this workload runs 5.1 docs/s at
  batch 1 and 2.0 at batch 32, because documents vary from 466 to 6,000 characters and
  every batch pads to its longest member. Length-sorting recovers part of it (6.0 docs/s).

## Related

`tools/data/annotate_casualty.py` (what the purchase buys), `tools/data/build_turkish_pool.py`
(where the pool comes from), `tools/data/notes/TURKISH_BATCHES.md` (batch ids -- a killed
poller does not lose money, but only if the id survives).
