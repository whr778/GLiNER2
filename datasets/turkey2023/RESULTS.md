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
- Article text is not committed (it is Al Jazeera's); `harvest_turkey_gt.py` and
  `build_turkey_feed.py` regenerate the feed from the archive.
