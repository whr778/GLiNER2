"""Compare two runs' test metrics, with the noise floor made explicit.

A delta is not a result without a floor beside it. This programme has measured a +/-0.041
run-to-run floor on relations and roughly +/-0.02 on a single run elsewhere, so anything
smaller is reported as INSIDE FLOOR rather than as a win or a loss.

It also REFUSES to compare runs whose eval data differs, because a delta across a changed test
set is confounded and this project has already retracted one finding for exactly that.

    uv run python tools/train/compare_runs.py --baseline a.json --candidate b.json \
        [--baseline-config x.yaml --candidate-config y.yaml] [--floor 0.02]
"""

import argparse
import json
from pathlib import Path

import yaml


def _eval_identity(path: str) -> tuple:
    """The parts of a config that decide WHAT was scored."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data = cfg.get("data") or {}
    return (
        tuple(data.get("corpora") or []),
        tuple(sorted((data.get("event_files") or {}).items(), key=lambda kv: kv[0])),
        cfg.get("labels_file"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline-config")
    ap.add_argument("--candidate-config")
    ap.add_argument("--floor", type=float, default=0.02,
                    help="deltas smaller than this are reported as inside the floor")
    ap.add_argument("--label-baseline", default="baseline")
    ap.add_argument("--label-candidate", default="candidate")
    args = ap.parse_args()

    # The comparison is void if the two runs did not score the same thing. Refuse rather
    # than print a confounded table someone will quote.
    if args.baseline_config and args.candidate_config:
        a, b = _eval_identity(args.baseline_config), _eval_identity(args.candidate_config)
        if a != b:
            fields = ("corpora", "event_files", "labels_file")
            differing = [f for f, x, y in zip(fields, a, b) if x != y]
            raise SystemExit(
                f"[compare] REFUSING: the runs scored different eval data ({differing}). "
                f"A delta across a changed test set is confounded, not a result."
            )
        print("[compare] eval data identical (corpora, event_files, labels_file) — "
              "the comparison is valid")

    base = json.load(open(args.baseline, encoding="utf-8"))
    cand = json.load(open(args.candidate, encoding="utf-8"))
    keys = sorted(k for k in set(base) | set(cand)
                  if k.endswith("micro_f1") and "fullmenu" not in k)

    print(f"\n{'metric (micro F1)':36s} {args.label_baseline:>12s} "
          f"{args.label_candidate:>12s} {'delta':>9s}")
    print("-" * 76)
    moved_up = moved_down = inside = 0
    for k in keys:
        x, y = base.get(k), cand.get(k)
        if x is None or y is None:
            print(f"{k.replace('eval_','').replace('_micro_f1',''):36s} "
                  f"{'-' if x is None else f'{x:.4f}':>12s} "
                  f"{'-' if y is None else f'{y:.4f}':>12s} {'n/a':>9s}  (missing one side)")
            continue
        d = y - x
        if abs(d) < args.floor:
            tag, inside = "  inside floor", inside + 1
        elif d > 0:
            tag, moved_up = "  UP", moved_up + 1
        else:
            tag, moved_down = "  DOWN", moved_down + 1
        print(f"{k.replace('eval_','').replace('_micro_f1',''):36s} "
              f"{x:12.4f} {y:12.4f} {d:+9.4f}{tag}")

    print("-" * 76)
    print(f"outside the +/-{args.floor} floor: {moved_up} up, {moved_down} down; "
          f"{inside} inside it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
