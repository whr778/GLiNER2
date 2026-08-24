# The aggregate constraint was rejected by the metric, not by the data

`EKF_MHT_DESIGN` §6.2 was titled "Aggregates cannot constrain their parts" and TODO item 10
recorded "it was tried and it lost -- do not revisit without a new reason". Both rest on one
number, `nrmse`, and that number does not measure what the aggregate is for.

## What nrmse is

```python
per_state = sqrt(((est - x_true) ** 2).mean(axis=0))   # RMSE per state, in deaths
rng_      = x_true.max(axis=0) - x_true.min(axis=0)    # that state's OWN range
return mean(per_state / rng_)                          # macro-average across states
```

Each state is normalised by its own range and then all states are weighted **equally**.
Virginia ranges 1 -> 2, so its denominator is 1 and a 1.4-death error reads as **1.4
nRMSE**. North Carolina ranges 6 -> 123; the same 1.4 deaths reads as 0.012. Virginia was
carrying **110.5% of the vector arm's total excess error** while every reported state
improved.

## The same runs, three metrics (default sweep, seed 0, 40 trials)

| density | nRMSE Δ | per-state RMSE Δ (deaths) | national total RMSE Δ |
|---|--:|--:|--:|
| 10% | +0.1737 ✗ | +0.35 ✗ | 87.6 -> 28.5 = **−59.1** ✓ |
| 20% | +0.0805 ✗ | −0.35 ✓ | 65.5 -> 28.8 = **−36.8** ✓ |
| 35% | +0.0568 ✗ | −0.61 ✓ | 49.6 -> 29.8 = **−19.8** ✓ |
| 50% | +0.0203 ✗ | −0.85 ✓ | 42.1 -> 27.9 = **−14.1** ✓ |
| 80% | −0.0036 ✓ | −0.69 ✓ | 30.9 -> 24.3 = **−6.6** ✓ |

**The aggregate constraint improves the national total estimate at every density**, by the
largest margin exactly where §6.2 says it "loses worst": 10% per-state reporting, a 67%
reduction in total error. The margin shrinks as per-state reports get dense -- which is the
predicted behaviour, since the aggregate has less left to add.

Under the real skewed arrival pattern (NC 84%, FL 52%, VA 0%) the same holds: nRMSE +0.2677
against deaths −0.49 and total −16.3.

## The mechanism in §6.2 was RIGHT; the verdict drawn from it was wrong

§6.2 says: "an aggregate constrains the SUM and says nothing about the SPLIT". That is
exactly right, and the three columns above are that sentence made measurable -- the split
(nRMSE) does not improve at sparse density, the sum improves enormously. The error was
concluding "therefore do not use it" while scoring only the split.

For a casualty tracker the national total is a headline number, not a diagnostic. Rejecting
a mechanism that halves its error because a 1-death state's relative accuracy got worse is
the metric choosing the result.

## What is actually open

- **Do not read this as "the vector state ships".** It is one simulated reporting process
  over 31 real snapshots and 6 states, and per-state relative accuracy genuinely does not
  improve at sparse density.
- The right target depends on the consumer. If the EKF's output is a national toll, use the
  aggregate. If it is per-state, the aggregate buys little until parts are dense.
- `--drop-unobserved` removes the never-reported state and flips nRMSE positive too
  (−0.0183, clears the floor). That is treating the symptom: the real issue is that
  macro-averaged range-normalisation is the wrong scorer for this question.
