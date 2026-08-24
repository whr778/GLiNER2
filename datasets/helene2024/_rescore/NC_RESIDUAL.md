# The North Carolina residual: half of it is not reachable

After the ratio scope gate (now wired into `run_pipeline`), the Helene residual sits almost
entirely in two streams -- the national total at 62.9 deaths RMSE and North Carolina at
59.3. This is what is in them.

## The scope gate is one-sided in the hierarchy

It rejects a figure too LARGE for a place (a national total filed under Florida) and is
blind to one too SMALL. North Carolina's 16 surviving observations:

| t_hours | value | NC truth | note |
|--:|--:|--:|---|
| 83.8 | 91 | 46 | high, but the toll is rising |
| 83.8 | 3, 3 | 46 | `"three"` -- a county or a single incident |
| 133.7-173.6 | 57, 57, 80, 57, 70, 70, 70, 61, 72, 72 | 73-99 | plausible, lagging |
| 192.2 | 32 | 99 | `"dozens"` |
| 569.6 | **1** | **123** | `"one"` -- a single death, filed as the state total |
| 726.9 | 98 | 123 | plausible |

The small values are the same hierarchy error one level DOWN: a county figure keyed to the
state. The scope gate cannot see them, because it only asks whether a value is too big.

## The innovation gate covers the other end, unevenly

"Too small for a rising toll" is what `REJECT_SIGMA` rejects -- the mechanism that fixed
Turkiye. Applied on top of the scope gate, in deaths:

| stream | truth | scope only | + innovation 3σ | + innovation 2σ |
|---|--:|--:|--:|--:|
| `__aggregate__` | 233 | 62.9 | **26.7** | 26.7 |
| north carolina | 102 | 59.3 | 54.1 | **52.4** |
| tennessee | 18 | 14.4 | **24.7** | 24.7 |
| florida | 26 | 12.5 | 12.5 | 12.5 |
| georgia | 33 | 10.2 | 10.1 | 10.1 |
| south carolina | 51 | 12.1 | 12.1 | 12.1 |
| **mean** | | **28.6** | 23.4 | **23.1** |

The national stream more than halves. **Tennessee gets worse** -- the gate assumes a
monotone rise, so an early over-estimate can never be corrected downward, and Tennessee's
truth is small enough that one bad early reading dominates. A one-sided gate is not free.

## Price the ceiling before spending on North Carolina

`stream_ceiling.py` takes, at each grid point, the observation closest to truth among those
seen so far. No method beats that from these observations.

| stream | achieved | oracle floor | addressable | coverage |
|---|--:|--:|--:|---|
| North Carolina | 52.4 | **23.0** | 29.4 | largest reported figure **98** vs truth **123** |
| Total | 26.7 | 16.0 | 10.7 | 250 vs 250 -- complete |

**About half of North Carolina's residual is not reachable from this feed.** 25 deaths never
appear in it. There is still 29.4 deaths of method headroom -- more than the national
stream's entire remaining error -- but a perfect tracker still finishes ~23 deaths out.

The national stream, by contrast, has complete coverage and is within 10.7 deaths of its own
ceiling. It is close to solved, and further association work aimed at it will not return
much.

## What this changes

- Rank association work in DEATHS and against the per-stream ceiling, not against nRMSE.
  Tennessee has the worst nRMSE (0.817) and the smallest absolute error; North Carolina has
  a middling nRMSE and the largest addressable gap.
- The next mechanism is a LOWER scope bound -- rejecting a figure too small to be the
  place's own total -- which is the mirror of the gate that already works. The innovation
  gate is a crude proxy for it and costs Tennessee to get it.
- Do not chase the national stream further without re-pricing: 10.7 deaths of headroom.
