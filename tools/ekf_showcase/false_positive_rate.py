"""What does the extractor emit on windows that contain NO death toll?

`binding_accuracy.py` builds every window AROUND a gold `dead` observation, so each one
contains a real toll by construction. It therefore measures recall and says nothing about
false positives -- and the wide-open 0.001 operating point it recommends lives or dies on
that. The router sees every window, not just the ones with an answer in them.

Negatives are drawn from the 26 feed articles carrying NO `dead` observation. Two controls
make that honest:

FAIRNESS -- a negative window must contain at least one NUMBER. Without that the test is
    "can the model avoid inventing a figure", which is far easier than "can it avoid
    labelling the wrong figure a death toll". Every positive window contains a number.

VALIDITY -- every observation in this feed is `mode: heuristic`; there is no human gold.
    "No toll found" may be the heuristic missing one. Windows whose text puts a death word
    near a number are reported as SUSPECT and kept out of the clean rate, because a model
    firing there may well be right and the label wrong.

    uv run python tools/ekf_showcase/false_positive_rate.py <model> [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from gliner2 import AutoExtractor, Schema

REPO = Path(__file__).resolve().parents[2]
TRACKED = REPO / "datasets/helene2024/_cache/tracked_rollup.json"

EVENT_TYPES = ["Floods", "Storm", "Hurricane", "Earthquakes", "Tropical Storm"]
ROLES = {"dead": "number of people killed", "location": "where the deaths occurred",
         "event_name": "the name of the storm or disaster"}
THRESHOLDS = (0.1, 0.05, 0.02, 0.01, 0.005, 0.001)

HAS_NUMBER = re.compile(r"\d")
DEATH_WORD = re.compile(
    r"\b(kill|killed|dead|death|deaths|died|dying|fatal|fatalities|toll|"
    r"victim|victims|perish|perished|deceased|casualt\w*|body|bodies|corpse)\b", re.I)


def negative_windows(size: int = 400):
    """(text, kind, suspect) for numeric windows of articles with no `dead` observation."""
    d = json.loads(TRACKED.read_text(encoding="utf-8"))
    out = []
    for a in d["articles"]:
        if any(o["role"] == "dead" for o in a["observations"]):
            continue
        kind = "irrelevant" if not a.get("relevant") else "relevant-no-toll"
        t = a["text"]
        for i in range(0, len(t), size):
            w = t[i:i + size]
            if len(w) < size // 2 or not HAS_NUMBER.search(w):
                continue
            out.append((w, kind, bool(DEATH_WORD.search(w))))
    return out


def dead_spans(model, text: str, th: float):
    """Deduped `dead` spans, matching binding_accuracy.py's counting."""
    found = []
    for etype in EVENT_TYPES:
        sch = Schema().events(
            {etype: {"roles": list(ROLES), "role_descriptions": ROLES}},
            trigger_threshold=th, argument_threshold=th)
        res = model.extract(text, sch, threshold=th, include_spans=True)
        for ev in (res.get("event_extraction") or {}).get(etype, []):
            for a in ev.get("arguments") or []:
                e = a.get("entity")
                if a.get("role") == "dead" and isinstance(e, dict) and "start" in e:
                    found.append((e["start"], e["end"], e["text"]))
    return sorted({s for s in found})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--thresholds", type=float, nargs="+", default=list(THRESHOLDS))
    args = ap.parse_args()

    wins = negative_windows()
    clean = [w for w in wins if not w[2]]
    suspect = [w for w in wins if w[2]]
    irrel = [w for w in clean if w[1] == "irrelevant"]
    print(f"{len(wins)} numeric windows from articles with no `dead` observation")
    print(f"  {len(clean)} clean (no death word), of which {len(irrel)} from irrelevant articles")
    print(f"  {len(suspect)} SUSPECT (death word present -- the heuristic may have missed a real toll)\n")

    model = AutoExtractor.from_pretrained(args.model, map_location=args.device)
    model.eval()
    print(f"model: {args.model}")
    print(f"{'thresh':>7}{'clean FP%':>11}{'spans/win':>11}{'irrel FP%':>11}{'suspect FP%':>13}"
          f"{'numFP%':>13}{'numspans':>11}")
    for th in args.thresholds:
        # Score each window ONCE; `irrel` is a subset of `clean`, so scoring the groups
        # separately would re-run 64 windows per threshold.
        scored = [(kind, susp, dead_spans(model, text, th)) for text, kind, susp in wins]

        def agg(rows, numeric_only=False):
            if not rows:
                return None, 0.0
            sel = [[x for x in s if HAS_NUMBER.search(x[2])] if numeric_only else s
                   for _, _, s in rows]
            return (sum(1 for s in sel if s) / len(rows),
                    sum(len(s) for s in sel) / len(rows))

        cl = [r for r in scored if not r[1]]
        c_rate, c_spans = agg(cl)
        n_rate, n_spans = agg(cl, numeric_only=True)
        i_rate, _ = agg([r for r in cl if r[0] == "irrelevant"])
        s_rate, _ = agg([r for r in scored if r[1]])
        fmt = lambda v: f"{v:.1%}" if v is not None else "--"
        print(f"{th:>7.3f}{fmt(c_rate):>11}{c_spans:>11.2f}{fmt(i_rate):>11}{fmt(s_rate):>13}"
              f"{fmt(n_rate):>13}{n_spans:>11.2f}")

    print("\nSample of what it emits on CLEAN negatives at the lowest threshold:")
    th = min(args.thresholds)
    shown = 0
    for text, kind, _ in clean:
        s = dead_spans(model, text, th)
        if s and shown < 6:
            print(f"  [{kind}] {[t for _, _, t in s][:4]}")
            shown += 1


if __name__ == "__main__":
    main()
