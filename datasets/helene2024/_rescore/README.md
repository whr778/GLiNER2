# Helene re-scored: do the disabled gates reach this event's contamination?

Offline from `_cache/tracked_rollup.json`, no model, no GPU
(`tools/ekf_showcase/rescore_helene.py`). Companion to the Turkiye addendum in
`datasets/turkey2023/RESULTS.md`.

**EXPLORATORY, not pre-registered**, and more tuned than the Turkiye run: `MAX_RATE` was
swept and the best value read off the answer. Treat the settings as illustrative.

## The prediction going in was wrong, which is why it was worth running

Turkiye's contamination is a stale LOW constant -- the 1999 Izmit 17,500, which the real
toll eventually overtakes -- so the one-sided rise gate rejects it by construction. Helene's
is the mirror image: 1400, 1400, 3000, 300 against a total reaching ~250. **The same gate
admits every one of them**, as its own docstring says. Expecting it to generalise directly
was backwards.

## After the scope filter, what is left

The scope filter rejects 17 of 106 as out of scope -- `puerto rico` (3,000) and
`reading pennsylvania` (Katrina's 1,400) among them. It cannot reach the in-scope
outliers: **North Carolina 1,400 and 250, Florida 300**. Those are precisely the residual
`EKF_MHT_DESIGN` 7.3 names -- "cross-event figures whose place is IN scope".

## Per stream, in deaths (final estimate / RMSE)

| stream | truth | n_obs | default | innovation gate 3σ | rate filter 5/h |
|---|--:|--:|--:|--:|--:|
| `__aggregate__` | 233 | 23 | 67 / 80 | **203 / 34** | 67 / 81 |
| north carolina | 102 | 26 | 268 / 645 | 268 / 645 | **248 / 92** |
| florida | 26 | 16 | 222 / 173 | 228 / 182 | 222 / 172 |
| georgia | 33 | 5 | 147 / 105 | 147 / 105 | 147 / 105 |
| south carolina | 51 | 9 | 192 / 127 | 192 / 127 | 185 / 120 |
| tennessee | 18 | 10 | 8 / 14 | 11 / 24 | 8 / 14 |
| virginia | 2 | 0 | 0 / 2 | 0 / 2 | 0 / 2 |

## Three findings

**1. The two gates fix different streams, and neither is redundant.** The innovation gate
more than halves error on the national stream (80 -> 34 deaths, final 67 -> 203 against a
truth of 233) by refusing low readings that were dragging it down. The rate filter cuts
North Carolina by 7x (645 -> 92) by removing the in-scope 1,400. Each is inert where the
other works.

**2. The dominant per-state error is not filtering at all.** Florida finishes at 222
against a truth of 26; Georgia 147 against 33; South Carolina 192 against 51. Those are
national-total-shaped numbers filed under individual states -- the aggregate-SCOPE failure
that TODO item 10 calls the sharpened target. **No gate setting touches it**, which is what
this run is best evidence for.

**3. Virginia has zero observations** across the whole feed, so its "estimate" is the
absence of one. Any per-state metric that macro-averages will be dominated by streams like
this; the table above is in deaths for that reason.

## Caveat on the tuning

Sweeping `MAX_RATE` and reporting the best is the same post-hoc move that inflated gate 1
on the extractor. At 1/h the mean looks far better still (RMSE 37.6) but Florida
overshoots to 12 against a truth of 26 -- the "win" is partly the correction landing on the
other side. Helene's national toll rose ~0.33/h on average, so a 1/h cap sits close to the
event's own dynamics and would not transfer.
