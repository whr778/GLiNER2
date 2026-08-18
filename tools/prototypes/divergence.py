"""Classify an MLM loss trace as trained / diverged / flat, robustly enough to trust.

Split out and unit-tested because two earlier in-line criteria were both wrong, in
opposite directions, and each wasted a run before the mistake was visible:

1. ``loss > first_loss * 2`` never fired. MLM starts at ~ln(vocab), the ceiling, and
   falls from there, so a threshold ABOVE the starting point is unreachable. Every arm
   was reported stable regardless of what happened.
2. ``loss > running_min + 1.0`` fired on step 51 of a run that finished at 6.79 against
   a 10.33 random baseline. Per-step MLM loss is noisy at batch 32, and early on the
   running minimum drops fast, so one unlucky batch trips it.

The signal that actually separates divergence from noise is the SMOOTHED trend: a
diverging run's moving average rises and stays up, while a noisy healthy run's moving
average keeps falling. So compare moving averages, and require the rise to persist.

Run ``python divergence.py`` to execute the self-check.
"""
from __future__ import annotations

import math
from collections import deque
from typing import List, Optional, Sequence


def moving_average(values: Sequence[float], window: int) -> List[float]:
    """Trailing mean over `window`, emitted once the window is full."""
    buf: deque = deque(maxlen=window)
    out = []
    for v in values:
        buf.append(v)
        out.append(sum(buf) / len(buf) if len(buf) == window else math.nan)
    return out


def classify(losses: Sequence[float], vocab: int, window: int = 50,
             margin: float = 1.0, persist: int = 100) -> dict:
    """Classify a loss trace.

    Args:
        losses: per-step loss values.
        vocab: vocabulary size; ``ln(vocab)`` is the random-guess loss.
        window: moving-average width, to average out per-batch noise.
        margin: how far the moving average must rise above its best to count. A
            transient spike of height h lasting k steps lifts a `window`-wide average
            by at most h*k/window, so margin must exceed that for the spikes worth
            tolerating: the 15-step, +2.5 bump in the self-check lifts it 0.75.
        persist: how many consecutive steps that rise must hold. A spike's influence
            decays out of the average after `window` steps, so persist > window means
            no transient can qualify however tall it is.

    Returns a dict with `diverged_at`, `learned`, `nonfinite`, `final`, `best_ma`.
    """
    random_loss = math.log(vocab)
    nonfinite_at = next((i for i, v in enumerate(losses) if not math.isfinite(v)), None)

    ma = moving_average(losses, window)
    best = math.inf
    above = 0
    diverged_at: Optional[int] = None
    for i, m in enumerate(ma):
        if not math.isfinite(m):
            continue
        if m < best:
            best = m
            above = 0
            continue
        if m > best + margin:
            above += 1
            if above >= persist and diverged_at is None:
                diverged_at = i
                break
        else:
            above = 0

    # A non-finite loss is divergence outright, and earlier than any trend test.
    if nonfinite_at is not None:
        diverged_at = nonfinite_at if diverged_at is None else min(diverged_at, nonfinite_at)

    finite = [v for v in losses if math.isfinite(v)]
    tail = finite[-window:] if finite else []
    final = sum(tail) / len(tail) if tail else math.nan
    return {
        "diverged_at": diverged_at,
        "learned": math.isfinite(final) and final < random_loss - 2.0,
        "nonfinite": nonfinite_at is not None,
        "final": final,
        "best_ma": best if math.isfinite(best) else math.nan,
        "random_baseline": random_loss,
    }


def _selfcheck() -> int:
    """Trace generators mimicking the four outcomes, with the real noise level."""
    import random

    rng = random.Random(0)
    V = 30522
    hi = math.log(V)
    n = 1500

    def healthy():                       # noisy but descending -- must NOT flag
        return [max(2.0, hi - 3.5 * (i / n) ** 0.5) + rng.gauss(0, 0.35) for i in range(n)]

    def spike():                         # one transient bump -- must NOT flag
        out = healthy()
        for i in range(700, 715):
            out[i] += 2.5
        return out

    def tall_spike():                    # taller transient -- margin alone would fail
        out = healthy()
        for i in range(700, 730):
            out[i] += 6.0
        return out

    def diverging():                     # descends then climbs and stays -- MUST flag
        out = []
        for i in range(n):
            base = hi - 3.0 * (i / 400) if i < 400 else hi - 3.0 + 2.5 * ((i - 400) / 400)
            out.append(min(hi + 1.0, base) + rng.gauss(0, 0.35))
        return out

    def nan_blowup():                    # goes non-finite -- MUST flag
        return healthy()[:600] + [math.nan] * (n - 600)

    def flat():                          # never learns -- not diverged, but not learned
        return [hi + rng.gauss(0, 0.2) for _ in range(n)]

    cases = [
        ("healthy   (noisy, descending)", healthy,   False, True),
        ("spike     (transient bump)",    spike,     False, True),
        ("tallspike (30 steps, +6.0)",     tall_spike, False, True),
        ("diverging (climbs and stays)",  diverging, True,  None),
        ("nan       (non-finite)",        nan_blowup, True, None),
        ("flat      (never learns)",      flat,      False, False),
    ]
    print(f"{'case':34s} {'diverged':>9s} {'expect':>8s} {'learned':>8s} {'expect':>8s}  result")
    failures = 0
    for label, gen, want_div, want_learned in cases:
        r = classify(gen(), V)
        got_div = r["diverged_at"] is not None
        ok = got_div == want_div and (want_learned is None or r["learned"] == want_learned)
        failures += not ok
        print(f"{label:34s} {str(got_div):>9s} {str(want_div):>8s} "
              f"{str(r['learned']):>8s} {str(want_learned):>8s}  {'PASS' if ok else 'FAIL'}")
    print("\nall passed" if not failures else f"\n{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_selfcheck())
