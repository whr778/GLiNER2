"""Is the bound figure the RIGHT figure? The measurement neither front-end gate makes.

Gate 1 counts windows where the model emits a trigger and >=1 bound argument. It scores
FORM, so a model that fires on everything at a permissive threshold wins it -- which is
what happened: `137k-clean` takes gate 1 with 65% at threshold 0.1, while gate 2 shows its
bindings there are nonsense (`dead` bound to "Helene decimated"). Scoring "best over a
threshold range" then rewards exactly that, against this project's own rule that arms be
compared at ONE decision threshold.

So: for each casualty-bearing Helene window we know where the gold death toll sits, by
character offset. A `dead` argument is CORRECT when its span overlaps that range. Two
models are reported side by side at each MATCHED threshold, never best-over-range.

    uv run python tools/ekf_showcase/binding_accuracy.py <model> [--device cpu]

ONE model per invocation. It accepts several, but loading a second boundary checkpoint in
the same process raises `KeyError: 'kernels-community/flash-attn2'` -- the first load leaves
an attn implementation registered that the second cannot resolve without `kernels`
installed. Run the tool twice and compare the two tables.

fired  -- windows with >=1 `dead` argument (what gate 1 counts)
hit    -- windows where a `dead` argument actually overlaps the gold figure
prec   -- hit / fired: when it commits to a toll, how often is it the right one
yield  -- hit / windows: end-to-end, what the EKF router would actually receive
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gliner2 import AutoExtractor, Schema

REPO = Path(__file__).resolve().parents[2]
TRACKED = REPO / "datasets/helene2024/_cache/tracked_rollup.json"
FEED = REPO / "datasets/helene2024/_cache/feed.jsonl"

# Identical to frontend_gates.py so the two are directly comparable.
EVENT_TYPES = ["Floods", "Storm", "Hurricane", "Earthquakes", "Tropical Storm"]
ROLES = {"dead": "number of people killed", "location": "where the deaths occurred",
         "event_name": "the name of the storm or disaster"}
THRESHOLDS = (0.5, 0.4, 0.3, 0.2, 0.1, 0.05)


def windows(limit: int = 60):
    """(text, gold_start, gold_end, gold_span) per casualty-bearing window.

    Offsets are into the WINDOW, not the article, so a predicted span can be compared
    directly against them.
    """
    d = json.loads(TRACKED.read_text(encoding="utf-8"))
    feed = {round(r["t_hours"], 2): r["text"]
            for r in (json.loads(l) for l in FEED.open(encoding="utf-8"))}
    out = []
    for a in d["articles"]:
        for o in a["observations"]:
            if o["role"] != "dead" or o["mode"] != "heuristic":
                continue
            t = feed.get(round(a["t_hours"], 2), "")
            span = str(o["span"])
            i = t.find(span)
            if i < 0:
                continue
            lo = max(0, i - 200)
            out.append((t[lo: i + len(span) + 200], i - lo, i - lo + len(span), span))
    return out[:limit]


def dead_spans(model, text: str, th: float):
    """Every `dead` argument the model binds, as (start, end, text)."""
    found = []
    for etype in EVENT_TYPES:
        schema = Schema().events(
            {etype: {"roles": list(ROLES), "role_descriptions": ROLES}},
            trigger_threshold=th, argument_threshold=th)
        res = model.extract(text, schema, threshold=th, include_spans=True)
        for ev in (res.get("event_extraction") or {}).get(etype, []):
            for a in ev.get("arguments") or []:
                if a.get("role") != "dead":
                    continue
                e = a.get("entity")
                if isinstance(e, dict) and "start" in e:
                    found.append((e["start"], e["end"], e["text"]))
    # One span found under three event types is ONE candidate, not three. Without this
    # the span counts are inflated ~5x by the EVENT_TYPES loop and `spans` reads as a
    # candidate-list length it is not.
    return sorted({(a, b, t) for a, b, t in found})


def score(model, wins, th):
    """Window-level and span-level accuracy.

    Two precisions, because they answer different questions and diverge badly as the
    threshold drops. `w-prec` is per WINDOW -- of windows where the model committed to a
    toll, how often was one of its `dead` spans right. That number inflates as the model
    emits more candidates per window, and degenerates to `yield` once it fires everywhere:
    a model tagging every numeral `dead` scores 100%. `s-prec` is per SPAN -- of every
    `dead` span emitted, how many hit. That is what a router picking ONE value faces.
    """
    fired = hit = n_spans = span_hit = 0
    misses = []
    for text, gs, ge, gold in wins:
        spans = dead_spans(model, text, th)
        n_spans += len(spans)
        span_hit += sum(1 for s, e, _ in spans if s < ge and e > gs)
        if not spans:
            continue
        fired += 1
        if any(s < ge and e > gs for s, e, _ in spans):
            hit += 1
        elif len(misses) < 3:
            misses.append((gold, [t for _, _, t in spans][:3]))
    return fired, hit, misses, n_spans, span_hit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--thresholds", type=float, nargs="+", default=list(THRESHOLDS),
                    help="Decision thresholds to score, matched across models.")
    args = ap.parse_args()

    wins = windows(args.limit)
    print(f"{len(wins)} casualty-bearing Helene windows, gold death toll located by offset\n")

    for name in args.models:
        model = AutoExtractor.from_pretrained(name, map_location=args.device)
        model.eval()
        print(f"=== {name}")
        print(f"{'thresh':>7}{'fired':>8}{'hit':>7}{'w-prec':>9}{'yield':>9}"
              f"{'spans':>8}{'s-prec':>10}")
        for th in args.thresholds:
            fired, hit, misses, n_spans, span_hit = score(model, wins, th)
            prec = f"{hit / fired:.1%}" if fired else "--"
            print(f"{th:>7.3f}{fired:>8}{hit:>7}{prec:>9}{hit / len(wins):>9.1%}"
                  f"{n_spans:>8}{(f'{span_hit / n_spans:.1%}' if n_spans else '--'):>10}")
            for gold, got in misses:
                print(f"         miss: gold {gold!r} -> bound {got}")
        print()


if __name__ == "__main__":
    main()
