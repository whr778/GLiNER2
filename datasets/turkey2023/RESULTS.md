# Turkiye-Syria 2023: first real-world EKF result

Run against the pre-registered plan in `PREREGISTRATION.md`. Read that first -- it was
committed before the pipeline was executed, and it predicted the main failure.

**Headline: this is a negative result, and a useful one.** Extraction recovers the real
Turkiye trajectory almost point-for-point. Association does not work on real text, the
EKF loses to a trivial baseline, and a death toll from a *different earthquake in 1999*
is bound into the 2023 stream in every configuration tried. The bottleneck in this
pipeline is not the filter and not the extractor -- it is attribution.

## What the ground truth is

16 daily points, 6-21 Feb 2023, one Al Jazeera live-tracker page sampled via the Wayback
Machine, one archive URL per point. Turkiye 1,014 -> 41,000; Syria 783 -> 5,800. Both
monotonic. Truncated at 21 Feb, where the page froze while the real toll went on to
53,537 -- staleness of the source, not a plateau of the event.

Known-answer, not blind: Feb 2023 precedes the model cutoff. That is why every figure is
sourced and cited rather than written from memory.

## Results

`dead` role, nRMSE against each national trajectory on the 6-hour grid. 1.0 is roughly
the score of predicting a constant.

| Run | Streams | nRMSE vs Turkiye | vs Syria | Izmit bound |
|---|---|---|---|---|
| **A.** Pre-registered config | 1 (`unknown`) | ekf 0.288 / last 0.343 | 3.377 | 12 / 20 obs |
| **B.** + stage 1 enabled | 1 (`Earthquakes\|turkey`) | ekf 0.208 / **last 0.136** | 3.349 | 16 / 91 obs |
| **C.** `--associate envelope` | 5 | turkey 0.228, syria 0.196 | 3.26, 3.40 | 16 / 91 obs |

### A. The pre-registered config was mis-specified -- my error

`--associate type+location` and `--window event` both depend on stage 1, which only runs
when `--event-model` is set. I froze a config without it, so stage 1 was skipped,
association degenerated to a single `unknown` key, and no envelopes were built. With no
envelope the extractor reads the whole ~6.2k-character article at once.

Its nRMSE of 0.288 is **not** a good result, and is worth dwelling on as a warning about
the metric. 12 of its 20 death readings are the number 17,500 -- the 1999 Izmit toll,
quoted in the article's historical round-up. That value happens to sit mid-range of a
1,014 -> 41,000 trajectory, so a badly wrong observation scores well. The `last_value`
endpoint gives the game away: it finishes at exactly 17,500 against a truth of 41,000.

### B. Extraction is good; the filter loses to "use the latest number"

With stage 1 on, the event type is correctly `Earthquakes` and the genuine Turkiye
series is read out of the text nearly point-for-point: 1,014, 2,316, 5,434, 9,057,
17,674, 20,213, 21,848, 29,605, 31,643, 35,418, 36,187, 38,000, 41,020.

But `est_last_value` (0.136) **beats** `est_ekf` (0.208), ending at 40,642 against a
truth of 41,000 while the EKF ends at 26,972. The filter is not broken -- it is being
fed a contaminated stream (16 copies of the 1999 figure, plus stray values of 7, 81 and
380) and its noise model assumes error roughly zero-mean about the truth. Heavy-tailed
contamination of that kind drags a smoother down while leaving "repeat the last reading"
untouched, because the last reading of each document is usually the correct standfirst.

A filter that cannot beat that baseline has, on this evidence, earned nothing here.

### C. Per-envelope association: predicted defect confirmed, fix insufficient

Prediction 1 held exactly. `run_pipeline.py` computed the association key once per
*document*, outside the envelope loop, while `casualty_windows` directly above built one
envelope per *incident*. Every document names Turkey before Syria, so all 16 keyed to
`Earthquakes|turkey` and the Syrian tolls were tracked as Turkish. Syria was detected --
123 location spans across the feed -- it simply never reached the key.

Keying per envelope, by the location span nearest the number, was applied **after** seeing
this and is reported separately for that reason. It ships as an opt-in `--associate
envelope` rather than as a change to `type+location`, because it does not actually fix
attribution and the default path has a validated synthetic result behind it. Turning it on
does split the feed into 5 streams. It does **not** solve attribution:

    Earthquakes|syria: 3,317  5,800  |  17,674  20,213  21,848  35,418  38,000  41,000
                       ^ Syrian         ^ Turkish figures, keyed to Syria

Both major streams track Turkiye (0.228, 0.196); neither tracks Syria (3.26, 3.40).
**Syria is never recovered.** Character distance is not syntactic attachment: in "At least
41,000 deaths have been reported in Turkey, while 5,800 people have died in Syria", the
binding of each number to its country is grammatical, and proximity only approximates it.

This is the case the file's own comment reserved for MHT -- "genuine ambiguity ... is the
data-association problem proper". The Turkiye-Syria feed says that case is not exotic. It
is what one ordinary news article looks like.

## The 1999 contamination

17,500 is bound as a 2023 casualty observation in every run (12/20, then 16/91). Nothing
in the pipeline checks that an extracted figure belongs to the event being tracked rather
than to a historical comparison in the same article. Retrieval-style relevance (the gate)
answers "is this article about a mass-casualty event" -- correctly, 16/16 -- and that is a
different question from "does this number belong to *that* event".

## Predictions, scored

Pre-registration is worth nothing unless the misses are scored as loudly as the hits.

1. **Hit.** Association pooled Turkiye and Syria into one stream, for exactly the stated
   reason -- the key is computed per document, outside the envelope loop.
2. **MISS.** I predicted pooling would push nRMSE toward or past 1.0. It came out 0.288
   and 0.208. The reason is a finding in its own right: range-normalized RMSE is
   insensitive to contamination that lands mid-range, and 17,500 sits mid-range of a
   1,014 -> 41,000 trajectory. The metric did not notice that 12 of 20 readings came
   from a different earthquake. **The prediction was wrong because the metric is weaker
   than I assumed, not because the pipeline did better than expected.**
3. **Hit.** The gate passed 16/16 documents as mass-casualty.
4. **Hit, and worse than feared.** The 1999 toll was bound in every configuration.

## What this changes

Ranked by what the evidence supports:

1. **Attribution is the bottleneck, not extraction or filtering.** Both of those work
   here. Every headline number is dominated by which event a figure was assigned to.
2. **Single-source dated series are a cheap and honest validation instrument.** One page
   sampled over time cost nothing and immediately exposed three defects that the
   synthetic feeds did not, because synthetic interference was built by concatenating
   snippets that never argue about the same event in the same sentence.
3. **Report the baseline every time.** Run B would have read as a success at 0.208 if
   `est_last_value` had not been carried alongside it.

## Limitations, stated plainly

- Known-answer, not blind. A genuinely blind test needs a post-cutoff event.
- One source, one page. This tests attribution and tracking, not source disagreement.
- The article body is ~95% identical day to day; the varying signal is the standfirst.
  This is a hard test of attribution and a weak test of extraction diversity.
- 16 documents. Small.
- "Extraction is good" is about the trajectory figures, not every span: the same run also
  read `magnitude 7.8` as 7 dead. Stray reads are visible in the value dump and are minor
  next to the 1999 contamination, but they are there.
- Observation counts jitter across identical configs (89-92 `dead`). The encoder resize
  is unseeded ("new embeddings initialized from a multivariate normal" on every load);
  nRMSE is stable at 0.208 regardless.
- Article text is not committed (it is Al Jazeera's); `harvest_turkey_gt.py` and
  `build_turkey_feed.py` regenerate the feed from the archive.

## Addendum (2026-08-10): the record head was never the problem — the window was

Follow-up to the attribution finding above, prompted by asking whether entity *logits*
could rank associations. Two results, one negative and one that changes the plan.

**Unary scores cannot rank pairings.** The confidence on `Turkey` is P(this span is a
Location); `Syria` scores highly too. Measured on the standfirst, the unary scores are
actively misleading — `Syria 0.49` outranks `Turkey 0.31` in the sentence where Turkey
owns the 41,000. Association needs an energy over *pairs*, which is what the `[C]`
record head and `[R]` relation head provide and what NER tags, of any granularity,
cannot. GPE-vs-LOC does not help here for the same reason: Turkey and Syria are both GPE.

**Record extraction has an inverted-U response to context volume** (`framing_experiment.py`,
gold sentence held identical in every condition, only the surrounding text varies):

| context | records/doc | Turkiye | Syria |
|---|--:|--:|--:|
| +0 .. +250 | 0.25 | 2/16 | 2/16 |
| +500 | 1.06 | 2/16 | 2/16 |
| **+1000 .. +1500** | **2.00** | **16/16** | **16/16** |
| +2500 .. +5000 | ~1.9 | 16/16 | 1-4/16 |
| full article (6.2k) | 1.38 | 16/16 | 0/16 |

Ragged versus word-boundary cuts made **no difference at any level**, so this is not a
truncation artefact. There are two distinct failure regimes: below ~750 characters the
record head does not fire at all; above ~2500 it fires normally (~1.9 records) but binds
the wrong pairs. Only the 1000-1500 band gets both right.

**At 1000-1500 characters the attribution failure disappears: 16/16 on both countries.**
No new machinery. The `[C]` head could bind number to country the whole time; it was
being handed 6,200 characters and asked to pick.

Which retires the earlier conclusion that this needs relation-based attachment as "the
real fix". It may still be the more robust mechanism, but it is no longer the cheapest
thing that works, and it should be justified against a windowed baseline rather than
against the full-document one.

**One honest caveat on why the window looks so good here.** The 1999 Izmit toll sits at
offset 4153-6171 in every document, so a 1500-character window excludes it *by layout*,
not by principle — the window silently gets credit for removing the temporal
contamination as well. A source that places its historical comparison earlier would break
that. The date filter stays the principled fix for temporal validity; the window is the
fix for binding. They address different failures and neither substitutes for the other.

## Addendum 2: attribution SOLVED — and the benchmark cannot score the filter

Wiring the two findings above into the pipeline (`--window lead --associate record`,
base model, location as a record field) resolves the attribution failure completely:

| stream | n | vs Turkiye | vs Syria |
|---|--:|--:|--:|
| `Earthquakes\|turkey` | 16 | **0.107** | 4.059 |
| `Earthquakes\|syria` | 14 | 0.607 | **0.095** |

Exactly two observations per document, values spanning 783 .. 41,000 (both countries'
true ranges), **zero Izmit contamination**, and Syria recovered as its own stream for the
first time. Compare the original run: Turkiye 0.208, Syria never recovered at all (3.3+).
No new model and no new training -- a window anchored on the article lead, plus asking the
record head for the location it already knew.

**But the headline finding of this document needs correcting.** On the now-clean streams:

| stream | EKF | `last_value` |
|---|--:|--:|
| turkey | 0.107 | **0.000** |
| syria | 0.095 | **0.017** |

`last_value` is not merely better, it is *exact*. It has to be: **the ground truth was
sourced from the same standfirst sentence the extractor reads.** One source, one figure
per day, and the truth IS that figure. "Repeat the last reading" is therefore a perfect
oracle by construction, and no filter can beat it -- smoothing an already-exact signal can
only add error.

So this benchmark **cannot evaluate the tracker**. It evaluates extraction and
attribution, and on those it is decisive. The earlier conclusion "the EKF loses to a
trivial baseline" was measuring two different things at once and must be split:

- In run B the EKF lost **because the observation stream was contaminated** (16 copies of
  a 1999 toll). That is a real finding and it stands.
- Here the EKF loses **because the baseline is an oracle**. That is an artefact of the
  benchmark design, not evidence about the filter.

A filter earns its keep only where observations are noisy, conflicting, lagged, or
revised -- i.e. where truth differs from any single report. Testing that needs **multiple
disagreeing sources**, which a single tracker page cannot provide by definition. That is
the next validation to build, and it is a different dataset, not a different parameter.

## Addendum 3: correcting the sequence-length explanation, and the mmBERT test

Addendum 1 attributed the upper arm of the inverted-U to the encoder's 512-token limit,
saying it "physically cannot see the whole document". **That is wrong**, and the
correction matters because it changes which fix is right.

Measured, not assumed:

- `fastino/gliner2-base-v1` is DeBERTa-v2/v3 with `position_biased_input=False`,
  `relative_attention=True`, `position_buckets=256`. It adds NO absolute position
  embeddings, so `max_position_embeddings: 512` is effectively vestigial and longer
  sequences are processed rather than rejected.
- GLiNER2 does not chunk. `processor.py` truncates only at an explicit `max_len` in WORD
  tokens, and `max_len=None` means no truncation at all.
- Demonstrated: on the full 6,211-character article the base model returns 25 location
  spans **including `Marmara` (offset 5985) and `Istanbul` (offset 6038)**, roughly 1,250
  subword tokens in. Nothing is being cut off.

So sequence length is still implicated -- the record-binding cliff does sit where inputs
cross ~512 tokens -- but the mechanism is **degradation past the trained length**, not
truncation. `position_buckets=256` is the more likely culprit: relative positions saturate
beyond the bucket range, so *local* work (entity detection) survives at long range while
*binding* (which is a relative-position relation between a number and a place) decays.
That also explains why entity recall stayed fine while record binding collapsed.

### The 137K joint-boundary models cannot replace stage 2

Tested directly, since 8192-token mmBERT looked like the obvious fix:

| | result |
|---|---|
| context | **8192** confirmed (ModernBERT, local_attention 128, global every 3) |
| classification | works (`earthquake`) |
| entities | works, but **2** location spans on the full article versus DeBERTa's 25 |
| `[C]` records | **None at every threshold down to 0.01**, anchor/field thresholds swept to 0.01 |

Not a threshold problem and not a wiring problem -- `enable_records: true` and
`record_decoder` is a real module. It is a **training-data** problem. The config states
the mix plainly: *100,080 event + 36,707 relation records*, with the comment
`enable_records: true  # load-bearing: events decode as trigger + role edges`. The record
head was switched on so EVENTS could decode as trigger/role edges; no JSON-structure data
was in the mixture, so the head was never trained on `casualty_report`-shaped tasks.

Consequences for the plan:

1. These checkpoints **cannot** serve as the casualty structure extractor. They also
   under-perform the DeBERTa base at zero-shot entity extraction.
2. They remain usable for stage 0/1 (gate and event classification), where `rams-137k`
   scores event_type 0.963.
3. Long-context record extraction needs a boundary model **trained with structure data**.
   That is the corpus rebuild, with a stronger rationale than "add a location field": it
   would buy 8192-token context AND record extraction in one model.

One practical gotcha: these checkpoints pin `attn_implementation:
kernels-community/flash-attn2`, which raises `KeyError` at forward time on a CPU box.
Load with `attn_implementation='sdpa'` off-GPU.

## Addendum 4: `extract_long` already does this, and does it better

`--window lead` (Addendum 2) worked but rested on a layout accident: Al Jazeera puts the
1999 comparison at the bottom, so reading the head both captured the current tolls and
skipped the contamination. A publication that leads with historical context breaks it.

GLiNER2 already ships the right mechanism -- `extract_long` / `batch_extract_long`
(`inference/runtime.py`), which splits a document into overlapping WORD chunks and merges
the results. Setting its chunk size to the band the framing experiment identified, over
the FULL 6,211-character article:

| chunk_size | overlap | Turkiye | Syria | Izmit bound |
|---|--:|--:|--:|--:|
| 384 (default) | 64 | 16/16 | 15/16 | 15 |
| **200** | **50** | **16/16** | **16/16** | 15 |
| 160 | 40 | 16/16 | 16/16 | 25 |

Perfect attribution on both countries with no lead assumption and no clipping of the
document. The default 384 is slightly too coarse -- consistent with the framing curve,
where binding starts to decay once a window crosses the model's comfortable range.

Contamination returns (15 Izmit bindings) precisely because the whole document is now
read, which is the honest state of affairs: the lead window was suppressing that error by
never looking at the offending paragraph. The three failures need three mechanisms:

  coverage + binding   `extract_long`, chunk_size ~200 words
  attribution          location as a record FIELD (`--associate record`)
  temporal validity    the date filter (13/15 caught, 0 false positives)

`--window lead` should be treated as superseded: it is a special case that happens to work
on one publication's layout.
