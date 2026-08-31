# Pricing an annotation purchase

Annotation is billed per DOCUMENT and valued per POSITIVE, so **the filter's precision is
the price**. This records the method and the numbers from the Turkish extractor purchase
(2026-08-29) so the next language does not rediscover them.

Reproduce with:

    uv run python tools/data/gate_purity_curve.py --model whr778/gliner2-gate2-mmbert-tr

This is the PRICING half. The end-to-end workflow — build the pool, price it, submit,
verify — is [TRAINING.md §3a-2](../../train/TRAINING.md).

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

## Choosing a source corpus (Simplified Chinese, 2026-08-31)

Four candidates evaluated before buying. The winner was not the first, and two of the
rejections were only visible on inspection, not from the dataset card.

| dataset | verdict | why |
|---|---|---|
| **`shaowenchen/news_zh`** | **CHOSEN** | 2.43M native full articles, median 794 chars, 7,409 publishers incl. Xinhua and China News Service, 93.9% Simplified / 0.03% Traditional, 3.57% cue-bearing (~68,000 articles) |
| `wuzimo2025/ChineseNewsSummaryDaily` | rejected | LLM **summaries**, median 284 chars, 15% English, 3.6% Traditional. Too short to carry a figure bound to a place, and LLM-summarised text annotated by an LLM compounds one model's errors into another's training data |
| `christykoh/ag_news_zh` | rejected | machine-translated 2004 AG News headlines, median 73 chars, visibly degraded (`路透社路透社(路透社)`, `AP(AP AP)`, hallucinated `USDODA.com`) with **mangled numerals** — fatal for a task about numbers |
| `CloverSearch/cc-news-mutlilingual` (zh) | rejected | real CC-News with per-article `date_publish` and 2016-2021 year partitions, but only **6-8% Simplified** on a representative stride — the Chinese CC-News crawl is dominated by Traditional outlets. Adds ~6,000 cue-bearing Simplified articles against news_zh's 68,000 |

**Two sampling traps, both of which gave the wrong answer first:**

- **Classify per DOCUMENT, not per character.** Counting simplified-only against
  traditional-only characters over a corpus reported "MIXED 5:1" for a corpus that is
  77% Simplified by document: a few Traditional documents outweighed many Simplified ones.
- **Stride the whole file, never the head.** CC-News year files are ordered with empty
  `maintext` records first, so reading the first 15,000 lines reported 4.7% Traditional
  for 2019 and 26% for 2021. Strided across the file, Simplified is the MINORITY at 6-8%.

**What the pool is not the constraint on.** news_zh offers ~68,000 cue-bearing candidates
and the budget buys ~25,000 annotations. Adding a second corpus for 6,000 more documents
changes nothing about what can be afforded; the money is the limit, not the data.

**Editorial regime is a caveat on the Chinese arm as a whole, not on one corpus.**
CommonCrawl reaches Chinese sites from outside, so CC-News captures the outward-facing
subset by construction -- a narrower and differently-selected slice than domestic
coverage. news_zh is not neutral either: its named publishers are state media
(新华网, 中国新闻网).

For the EXTRACTOR this is close to harmless -- it learns to bind a number to a place from
whatever the text says, and editorially constrained text is still ordinary Chinese
casualty prose. For the EKF it is not. The filter models a toll as a quantity revised
upward as information arrives, and it is tuned on the shape of that revision. Figures that
are delayed, floored, or revised late describe a REPORTING REGIME rather than an event,
and a filter fitted to them is fitting the regime. Do not pool Chinese streams with
Helene-style streams without checking that assumption; per-language tracking error is the
measurement that would expose it.

**Known limitation, to be stated in the model card:** news_zh is 2014-2016 and its `time`
field carries NO year (`MM-DD HH:MM` only), so articles cannot be individually dated and no
held-out-by-year split is possible. The Turkish arm spans 2016-2023 by comparison.
