"""Compare the label marginals of generated corpora against a real-text reference.

The question this answers: does a generated corpus DISTRIBUTE its labels the way real
documents do, or the way a generator prefers to? A corpus can be well-formed, verbatim-
clean and fully parsed while every document lands on the same three labels -- which is
label-space collapse measured on the generator side rather than the annotator side
(`tools/events_working_papers/LABEL_SPACE_COLLAPSE.md`).

METRIC: TVD, total variation distance, ``0.5 * sum_k |p(k) - q(k)|`` over one task's
label marginal. 0 means the two distributions are identical; 1 means disjoint. It is
bounded, symmetric, needs no smoothing for zero counts, and is readable as "the largest
share of probability that would have to move".

TWO THINGS IT COUNTS, AND ONE IT REFUSES TO:

  * `true_label` -- the ANSWER. Never `labels`, which is the MENU the annotator was
    offered. Measuring the menu was a real defect in this project: it reports the
    sampler's uniform choice and reads as a healthy distribution no matter what the
    annotator answered.
  * Support, printed beside every TVD. A TVD over 12 records is not comparable to one
    over 500 and must not be read as if it were.
  * A task a record does not carry contributes NOTHING to that task's marginal. A task
    that was never offered is not evidence of anything about its distribution.

    uv run python tools/data/compare_label_distributions.py \\
        --reference data/cc_news_haiku45 --arm onecall=data/lg_onecall \\
        --arm twostage=data/lg_twostage
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")


def _read(base: str):
    """Yield records from base.{train,val,test}.jsonl, or from base itself if it is a file.

    All splits, because a generated arm's split assignment is an artifact of SplitWriter's
    seed and carries no meaning for a distribution question.
    """
    p = Path(base)
    paths = [p] if p.is_file() else [p.with_name(f"{p.name}.{s}.jsonl") for s in SPLITS]
    found = False
    for path in paths:
        if not path.is_file():
            continue
        found = True
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
    if not found:
        raise SystemExit(f"no corpus at {base} (tried {[str(x) for x in paths]})")


def marginals(base: str) -> dict[str, Counter]:
    """task -> Counter over true_label uses, plus a `_records` count per task."""
    out: dict[str, Counter] = defaultdict(Counter)
    seen: Counter = Counter()
    for rec in _read(base):
        for c in (rec.get("output") or {}).get("classifications") or []:
            task = c.get("task")
            if not task:
                continue
            seen[task] += 1
            for lab in c.get("true_label") or []:
                out[task][lab] += 1
    for task in out:
        out[task]["__records__"] = seen[task]
    return out


def tvd(p: Counter, q: Counter) -> float:
    """0.5 * sum |p(k) - q(k)| over the union of labels, each normalised to a marginal."""
    pt, qt = sum(p.values()), sum(q.values())
    if not pt or not qt:
        return float("nan")
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p[k] / pt - q[k] / qt) for k in keys)


def top(c: Counter, n: int = 2) -> str:
    tot = sum(c.values())
    return ", ".join(f"{k} {v / tot:.0%}" for k, v in c.most_common(n)) if tot else "-"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True,
                    help="corpus basename the arms are compared TO (real text)")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=BASE",
                    help="an arm to score; repeatable")
    ap.add_argument("--json", type=Path, help="also write the table here")
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        name, _, base = spec.partition("=")
        if not base:
            raise SystemExit(f"--arm needs NAME=BASE, got {spec!r}")
        arms[name] = base

    ref = marginals(args.reference)
    got = {name: marginals(base) for name, base in arms.items()}
    names = list(arms)

    print(f"reference: {args.reference}")
    for name, base in arms.items():
        print(f"arm {name}: {base}")
    print("\nTVD (total variation distance) of each arm's true_label marginal against "
          "the reference.\n0 = identical distribution, 1 = disjoint. Lower is closer to "
          "real text.\n")

    head = f"{'task':<22}{'ref n':>8}"
    for n in names:
        head += f"{n + ' n':>12}{n + ' TVD':>14}"
    print(head)
    print("-" * len(head))

    rows = {}
    wins = Counter()
    for task in sorted(ref):
        r = Counter({k: v for k, v in ref[task].items() if k != "__records__"})
        line = f"{task:<22}{ref[task]['__records__']:>8,}"
        row = {"reference_records": ref[task]["__records__"], "reference_top": top(r)}
        scores = {}
        for n in names:
            a = got[n].get(task, Counter())
            an = a.get("__records__", 0)
            ac = Counter({k: v for k, v in a.items() if k != "__records__"})
            d = tvd(r, ac) if an else float("nan")
            scores[n] = d
            line += f"{an:>12,}{d:>14.4f}" if an else f"{0:>12}{'-':>14}"
            row[n] = {"records": an, "tvd": None if an == 0 else round(d, 4),
                      "top": top(ac)}
        live = {n: d for n, d in scores.items() if d == d}
        if len(live) == len(names) and len(names) > 1:
            best = min(live.values())
            winners = [n for n, d in live.items() if d - best < 1e-9]
            wins["tie" if len(winners) > 1 else winners[0]] += 1
        rows[task] = row
        print(line)

    if len(names) > 1:
        scored = sum(wins.values())
        parts = [f"{n} {wins[n]} of {scored}" for n in names]
        if wins["tie"]:
            parts.append(f"tie {wins['tie']}")
        print(f"\ncloser to the reference, per task (of {scored} both arms scored): " +
              ", ".join(parts))
        # The count throws away magnitude: 7-of-12 by 0.001 each is not 7-of-12 by 0.2.
        for n in names:
            ds = [rows[t][n]["tvd"] for t in rows if rows[t][n]["tvd"] is not None]
            if ds:
                print(f"  mean TVD {n}: {sum(ds) / len(ds):.4f} over {len(ds)} tasks")
    print("\ntop labels (reference | " + " | ".join(names) + ")")
    for task in sorted(rows):
        parts = [rows[task]["reference_top"]] + [rows[task][n]["top"] for n in names]
        print(f"  {task:<22}" + "   |   ".join(parts))

    if args.json:
        args.json.write_text(json.dumps(
            {"metric": "tvd", "reference": args.reference, "arms": arms, "tasks": rows,
             "wins": dict(wins)}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
