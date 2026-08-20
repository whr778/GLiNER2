# The archived Helene baseline is not reproducible from git

Investigated 2026-08-20, after the muting arm's primary guard could not be scored against
the published 5.247 / 0.591.

## What was asked

`datasets/helene2024/_cache/tracked_rollup.json` (gitignored, written 2026-08-10 18:42) is
the observation set behind every published Helene number: the scope gate's **5.247 → 0.591**,
the two-way oracle **0.537**, the three-way oracle **0.480**, and the M5 track-birth table.
Running a new model through `run_pipeline.py` should place it on the same scale. It does not,
so the question was which flags the archived run used.

## What was found

**No committed state of the repository can produce that file.**

| check | result |
|---|---|
| model changed on the Hub since? | **no** — `casualty-docee` last touched 2026-08-10 00:06, before the run |
| CLI defaults changed since? | **no** — model, chunk size 200/50, thresholds, normalizer all identical |
| `--grid-step` | archived grid is 92 points at step **12.0**; the default is 6.0, so the run passed `--grid-step 12` |
| code regression after the run? | **no** — a worktree at `bc24c4c` gives 91 `dead` against current code's 88 |
| **`--rollup` implemented at 18:42?** | **NO.** `bb3ca39` (18:32) is the last commit before the file and contains no `--rollup` at all |
| **`rollup.json` in the tree at 18:42?** | **NO.** First committed in `3852d0b` at 19:14, then edited again at 19:21 ("rollup tail") |

There are no commits between 18:32 and 19:14. The archived artifact carries rollup-mapped
keys (`north carolina`, `__aggregate__`) produced by code and data that existed only as
uncommitted working-tree state, and that state changed before it was committed — the
committed version does not reproduce it.

## The signature of the mismatch

| run | `dead` | distinct keys | most common keys |
|---|--:|--:|---|
| **archived** | 106 | 21 | `north carolina` 26, `__aggregate__` 23, `florida` 16, `tennessee` 10 — **no `unknown`** |
| docee, current code | 88 | 15 | **`unknown` 49**, `north carolina` 12, `__aggregate__` 7 |
| docee, code at `bc24c4c` | 91 | 14 | **`unknown` 50** |
| multievent (the other pre-08-10 model) | 76 | 8 | **`unknown` 68**, plus `two`, `nine`, `more than 200` |

The archived run assigns every observation to a scope; every reproduction attempt leaves
over half unassigned because the record head returns no `location`. `multievent` additionally
shows the numeric-field collapse `_locate_place` documents — asked for a location it returns
the number.

A second, unrelated symptom: reproductions put **94,000 into Tennessee**, where the archived
run's largest `dead` value anywhere is 3,000. Present on both old and current code, so it is
not a regression introduced since.

## What this does and does not invalidate

**Does not.** Every published number derived from this file is *mutually* consistent, because
they all read the same frozen artifact: the gate's 5.247 → 0.591, the random-removal control
4.427, both oracle ceilings, and the M5 sweep. Comparisons *among* them stand.

**Does.** The file cannot be regenerated, and no newly trained model can be placed on its
scale. That is precisely what blocked the muting arm's primary guard, which was
pre-registered against 5.247. Any future arm faces the same wall.

**The fix is a fresh baseline, not an archaeology project.** Re-run `run_pipeline.py` for
`casualty-docee` and every arm under one recorded invocation, and treat *that* as the
reference. The three muting-arm runs already share settings, so they only need docee added
on the same footing. The published figures should then be labelled as belonging to a
superseded, unreproducible observation set rather than silently compared against.

**And record the invocation in the artifact.** `tracked_rollup.json` stores `associate` and
nothing else. Writing the full argument vector into the output would have made this a
one-minute check. This is the second time provenance has blocked this programme — the
Türkiye–Syria re-extraction is stalled for the same reason (`EKF_MHT_BUILD_RECORD.md` §25.5,
"the original run's `--event-model` is not recorded").
