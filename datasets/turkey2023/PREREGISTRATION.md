# Turkiye-Syria 2023: pre-registered validation

Written **before** running the pipeline on this feed. Committed first so the result
cannot be retrofitted to whatever comes out.

## Why this event, and why it is not blind

The EKF/MHT line has so far been validated only on synthetic streams whose ground truth
we generated. This is the first run against a real event with an externally sourced
trajectory.

It is a **known-answer** test, not a blind one. The February 2023 earthquakes precede the
assistant's training cutoff, so the outcome is already known to the model driving the
pipeline. That is a real limitation and is why the ground truth was sourced and cited
rather than written from memory (`ground_truth.json` carries a Wayback URL per point),
and why this file is committed before the run. A genuinely blind test needs a
post-cutoff event; that remains the Venezuela case.

## Ground truth

16 daily points, 6-21 Feb 2023, from one Al Jazeera live-tracker page sampled via the
Wayback Machine. Turkiye 1,014 -> 41,000; Syria 783 -> 5,800. Both monotonic. The series
is truncated at 21 Feb because the page froze there while the real toll went on to
53,537 -- staleness of the source, not a plateau of the event.

## Frozen configuration

    uv run python tools/ekf_showcase/run_pipeline.py \
        --feed datasets/turkey2023/_cache/feed.jsonl \
        --gate-model fastino/gliner2-base-v1 \
        --casualty-model whr778/gliner2-base-v1-casualty-multievent \
        --normalizer heuristic \
        --associate type+location \
        --window event \
        --gate-threshold 0.5 --event-threshold 0.3 --grid-step 6.0 \
        --device cpu

Every threshold is the existing default. `--casualty-model` is the multi-event model
because this feed is multi-event, which is the condition it was trained for. Nothing here
is tuned on this feed, and it will not be: any change made after seeing output is
reported as a separate, clearly-labelled result.

## Metric

Per-stream normalized RMSE of the `dead` role against that stream's own trajectory on the
6-hour grid, `est_ekf` versus the `est_last_value` baseline. Normalized by the truth's
range, so 1.0 is the score of predicting a constant.

## Predictions

Recorded now, in order of confidence.

1. **Association will pool Turkiye and Syria into ONE stream.** `run_pipeline.py:531`
   computes the association key once per *document*, outside the envelope and record
   loops, while `casualty_windows` immediately above it builds one envelope per
   *incident*. Every document here describes both countries, so a document-level key
   cannot separate them. This contradicts the stated design intent at line 520 ("a
   multi-event article carries one casualty span per incident").
2. Pooling will make the tracked stream chase whichever figure was read last, so nRMSE
   against the Turkiye trajectory will be poor -- worse than 1.0 is plausible, since
   Syrian figures sit ~7x below Turkish ones over the same window.
3. The gate will pass all 16 documents as mass-casualty.
4. The 1999 Izmit toll (17,500), present in 15/16 documents, is the sharpest distractor.
   Whether it is bound is a pass/fail check reported regardless of the headline number.

If prediction 1 holds, the fix is to compute the key per envelope rather than per
document. That would be a defect corrected against the design intent already written in
the file -- not tuning -- but it happens after the fact, so both numbers get reported.
