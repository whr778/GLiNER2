"""How ambiguous is string-anchored event gold, per corpus?

Gold triggers and arguments are STRINGS, and training places a string at EVERY occurrence of its
word run (lowercased word split + token-run search, `processor._find_sublist`; occurrence policy
`latent_all`). A surface that repeats in its document therefore becomes several positive targets,
some of them the same word in an unrelated sentence. Per corpus, over a seeded sample of train
records that carry `output.events`:

    chars        median document length
    ev/doc       events per document
    trig occ     mean places a placed gold trigger lands (1.00 = unambiguous)
    trig>1       share of placed triggers that land in more than one place
    arg occ/>1   the same for arguments
    unplaced     share of surfaces training cannot place at all (e.g. inside a hyphenated word)

    uv run python tools/data/measure_occurrence_ambiguity.py
    uv run python tools/data/measure_occurrence_ambiguity.py --corpora wikievents duee_ner --n 1000

EVENT_ARGUMENT_DIAGNOSIS.md s4l; TODO #26.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from gliner2.processing.word_splitter import WhitespaceTokenSplitter

EB21_EVENT_CORPORA = [
    "maven", "casie", "events_biotech", "turkish_event", "text2json", "chfinann", "duee_ner", "cmnee_ner",
    "docee", "docee_zh", "docfee", "rams_merged", "wikievents", "cc_news_events_haiku45",
    "cc_news_events_sonnet55_v2", "synthetic_events_capped",
]
_split = WhitespaceTokenSplitter()


def occurrences(text: str, surface: str) -> int:
    """How many places training marks ``surface`` gold in ``text``."""
    words = [t for t, _, _ in _split(text, lower=True)]
    sub = [t for t, _, _ in _split(surface, lower=True)]
    n = len(sub)
    return sum(words[i:i + n] == sub for i in range(len(words) - n + 1)) if n else 0


def _surface(x) -> str:
    return (x.get("text") if isinstance(x, dict) else x or "").strip()


def measure(rows: list) -> dict:
    """The table's columns over ``rows`` (records with ``output.events``)."""
    lengths, events, trig, arg, unplaced, total = [], [], [], [], 0, 0
    for r in rows:
        text, evs = r["input"], r["output"]["events"]
        lengths.append(len(text))
        events.append(len(evs))
        for ev in evs:
            surfaces = [(trig, _surface(t)) for t in ev.get("triggers") or []]
            surfaces += [(arg, _surface(a.get("entity"))) for a in ev.get("arguments") or []]
            for bucket, s in surfaces:
                if not s:
                    continue
                n = occurrences(text, s)
                total += 1
                unplaced += n == 0
                if n:
                    bucket.append(n)
    mean = lambda x: statistics.mean(x) if x else 0.0
    many = lambda x: sum(v > 1 for v in x) / len(x) if x else 0.0
    return {"chars": statistics.median(lengths), "ev_per_doc": mean(events), "trig_occ": mean(trig),
            "trig_gt1": many(trig), "arg_occ": mean(arg), "arg_gt1": many(arg),
            "unplaced": unplaced / max(total, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpora", nargs="+", default=EB21_EVENT_CORPORA)
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=300, help="records sampled per corpus")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    print(f"{'corpus':28} {'chars':>7} {'ev/doc':>6} {'trig occ':>8} {'trig>1':>6} {'arg occ':>7} {'arg>1':>6} {'unplaced':>8}")
    for c in args.corpora:
        path = Path(args.data) / f"{c}.train.jsonl"
        if not path.is_file():
            print(f"{c:28} missing ({path})")
            continue
        with open(path, encoding="utf-8") as f:
            rows = [r for r in map(json.loads, f) if (r.get("output") or {}).get("events")]
        if not rows:
            print(f"{c:28} no output.events records (events stored in another shape; not measured)")
            continue
        m = measure(random.sample(rows, min(args.n, len(rows))))
        print(f"{c:28} {m['chars']:7.0f} {m['ev_per_doc']:6.1f} {m['trig_occ']:8.2f} {m['trig_gt1']:6.1%} "
              f"{m['arg_occ']:7.2f} {m['arg_gt1']:6.1%} {m['unplaced']:8.1%}")


if __name__ == "__main__":
    main()
