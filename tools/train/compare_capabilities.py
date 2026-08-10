"""Per-capability diff between two blind-test metric files.

Built for the warm-start question, which has two halves and one trap. The halves: did the
ADDED capability appear (structure/records, which `mmbert-137k` cannot do at all), and did
the OLD capabilities survive 30% replay. The trap is that a single headline number answers
neither -- the casualty fine-tune earlier in this project improved the thing it was trained
on while silently destroying a field type nobody was watching, and it took a targeted probe
to notice.

So this prints every task family side by side, flags regressions explicitly, and refuses to
average them into a score.

It also checks the thing that has twice changed a conclusion in this project: whether the
two runs were calibrated at the SAME threshold. Numbers read at different operating points
are not comparable, and `test_metrics.json` does not record the threshold, so the sweep file
has to be consulted separately.

    uv run python tools/train/compare_capabilities.py \
        --baseline .../joint-boundary-mmbert-137k/best \
        --candidate .../joint-boundary-warmstart-struct/best
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FAMILIES = ("entity", "relation", "event_type", "event_trigger", "event_argument",
            "event", "classification", "json")
VARIANTS = ("strict", "relaxed")


def load(d: Path):
    m = json.loads((d / "test_metrics.json").read_text(encoding="utf-8"))
    thr = None
    sweep = d / "threshold_sweep.json"
    if sweep.is_file():
        thr = json.loads(sweep.read_text(encoding="utf-8")).get("chosen_threshold")
    return m, thr


def val(m, fam, variant, stat="micro_f1"):
    return m.get(f"eval_{fam}_{variant}_{stat}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--variant", default="strict", choices=VARIANTS)
    args = ap.parse_args()

    base, base_thr = load(Path(args.baseline))
    cand, cand_thr = load(Path(args.candidate))

    print(f"baseline  {args.baseline}   threshold={base_thr}")
    print(f"candidate {args.candidate}   threshold={cand_thr}")
    if base_thr != cand_thr:
        print("\n  *** THRESHOLDS DIFFER -- these numbers are read at different operating\n"
              "      points and are NOT directly comparable. This has changed a conclusion\n"
              "      twice in this project; fix it before drawing one. ***")

    print(f"\n{'capability':<20}{'baseline':>10}{'candidate':>11}{'delta':>9}   verdict")
    regressions, gains = [], []
    for fam in FAMILIES:
        b, c = val(base, fam, args.variant), val(cand, fam, args.variant)
        if b is None and c is None:
            continue
        bs = "--" if b is None else f"{b:.3f}"
        cs = "--" if c is None else f"{c:.3f}"
        if b is None or c is None:
            verdict = "NEW capability" if b is None else "LOST capability"
            delta = "--"
        else:
            d = c - b
            delta = f"{d:+.3f}"
            # 0.005 is noise on a single seed; anything larger is worth a sentence.
            verdict = "regressed" if d < -0.005 else ("improved" if d > 0.005 else "flat")
            (regressions if d < -0.005 else gains if d > 0.005 else []).append((fam, d))
        print(f"{fam:<20}{bs:>10}{cs:>11}{delta:>9}   {verdict}")

    print("\nsupport (baseline / candidate), so a moved metric is not a moved test set:")
    for fam in FAMILIES:
        b, c = val(base, fam, args.variant, "support"), val(cand, fam, args.variant, "support")
        if b is not None or c is not None:
            flag = "  <-- SUPPORT CHANGED" if b != c else ""
            print(f"   {fam:<20}{b} / {c}{flag}")

    print()
    if regressions:
        print("REGRESSIONS (report these, do not average them away):")
        for fam, d in sorted(regressions, key=lambda x: x[1]):
            print(f"   {fam:<20}{d:+.3f}")
    else:
        print("No capability regressed by more than 0.005.")


if __name__ == "__main__":
    main()
