"""Is our event gold trigger-keyed or event-keyed? Two measurements that decide design C's payoff.

Design C (normalised events) separates an event's identity from its trigger mentions: several
mentions -> one event, arguments attached to the event. Its distinctive benefit only exists if
the data has events with several mentions, or one real event annotated as several trigger-keyed
events. Per corpus, from the gold alone (after the training label transform):

  (1) share of gold events carrying MORE THAN ONE trigger mention;
  (2) share of same-type event PAIRS in a document that share >= 1 argument -- by text, and
      stricter by (role, text) -- an UPPER bound on one real event split into several (two real
      events can share a participant); then NESTED (one event's argument set contained in the
      other's) and IDENTICAL (equal sets), the tighter signatures of a split event.

    uv run python tools/data/measure_event_normalisation.py [--split train]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/train"))
import train as T  # noqa: E402


def arg_sets(event: dict):
    """(texts, (role, text) pairs) of an event's arguments, normalised to lower case."""
    texts, pairs = set(), set()
    for a in event.get("arguments") or []:
        text = (a.get("entity") or a.get("text") or "")
        if isinstance(text, str) and text.strip():
            texts.add(text.strip().lower())
            pairs.add((a.get("role"), text.strip().lower()))
    return texts, pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="tools/train/config/base/eb18-balanced.yaml")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    fns = T._category_fns(T.load_labels_cfg(cfg, args.config))
    data = cfg["data"]
    files = T._dedupe_paths(T._split_files(data.get("corpora") or [], args.split,
                                           set(data.get("train_only") or ()) if args.split != "train" else set())
                            + T._event_split(data.get("event_files") or {}, args.split), args.split)
    print(f"[norm] {'corpus':28s} {'docs':>6s} {'events':>7s} {'>1 trigger':>11s} | {'same-type pairs':>15s} "
          f"{'share arg':>10s} {'share role+arg':>14s} {'nested':>8s} {'identical':>9s} | {'docs w/ split?':>14s}")
    for f in files:
        if not Path(f).is_file():
            continue
        c = Counter()
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            events = [e for e in (T.transform_record(json.loads(line), fns)["output"].get("events") or [])
                      if isinstance(e, dict) and e.get("event_type")]
            if not events:
                continue
            c["docs"] += 1
            c["events"] += len(events)
            c["multi_trigger"] += sum(len(e.get("triggers") or []) > 1 for e in events)
            split_doc = False
            for a, b in combinations(events, 2):
                if a["event_type"] != b["event_type"]:
                    continue
                c["pairs"] += 1
                (ta, pa), (tb, pb) = arg_sets(a), arg_sets(b)
                if ta & tb:
                    c["share_text"] += 1
                    split_doc = True
                if pa & pb:
                    c["share_role"] += 1
                if ta and tb and (ta <= tb or tb <= ta):
                    c["nested"] += 1           # one event's arguments contained in the other's
                if ta and ta == tb:
                    c["identical"] += 1
            c["split_docs"] += split_doc
        if not c["events"]:
            continue
        pct = lambda k, d: f"{100 * c[k] / c[d]:5.1f}%" if c[d] else "    -"
        print(f"[norm] {Path(f).name.split('.')[0]:28s} {c['docs']:6d} {c['events']:7d} {pct('multi_trigger', 'events'):>11s} | "
              f"{c['pairs']:15d} {pct('share_text', 'pairs'):>10s} {pct('share_role', 'pairs'):>14s} {pct('nested', 'pairs'):>8s} {pct('identical', 'pairs'):>9s} | {pct('split_docs', 'docs'):>14s}")


if __name__ == "__main__":
    main()
