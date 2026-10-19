"""Target structure of every event and relation corpus in a config: what the model is actually asked to learn.

Three properties decided whether the relation head could fit its own training data (TODO #40), and the same three
apply to event arguments. Measured per corpus, on the train split, through the real collator where it matters:

  EVENTS
    inflation      argument VALUES in the collated record targets per text-level argument (each occurrence of an
                   argument's text becomes a separate required value of a list role)
    same-type      share of events that share their document with another event of the SAME type (pairing load
                   on the trigger x argument junction)
    shared args    in those documents, share of arguments whose text also argues a sibling event of that type
    null labels    event types / roles named like none / other / no_event
  RELATIONS
    inflation      gold mention-pair targets per text-level edge, `relation_gold_pairs` all vs nearest
    pairing        same-type head x tail combinations per true edge (1.0 = unambiguous)
    null labels    relation types named like no relation / none

Prints a markdown section for tools/data/TRAINING_DATA.md.

    uv run tools/data/measure_target_structure.py --config tools/train/config/base/eb22b.yaml --docs 200
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/train"))
import train as T  # noqa: E402
import gliner2.processing.boundary_preprocessing as BP  # noqa: E402
from gliner2.training.trainer import ExtractorCollator  # noqa: E402

NULL = re.compile(r"(?i)^(none|other|null|n/?a|no[ _-]?(event|relation))$")
CAPTURED: list = []
_build = BP._build_sample_records


def _capture(*a, **k):
    out = _build(*a, **k)
    CAPTURED.append(out)
    return out


BP._build_sample_records = _capture


def _arg_text(arg) -> str | None:
    e = arg.get("entity")
    return e if isinstance(e, str) else (e or {}).get("text") if isinstance(e, dict) else None


def event_row(recs, coll) -> dict:
    """Inflation through the collator; pairing and null labels from the records."""
    text_args = values = 0
    for r in recs:
        text_args += sum(len(e.get("arguments") or []) for e in r["output"]["events"])
        CAPTURED.clear()
        random.seed(0)
        coll([(r["input"], r["output"])])
        for rec in (CAPTURED[0] if CAPTURED else []):
            values += sum(len(f.values) for f in rec.fields if f.query_id != rec.anchor_query_id)
    events = in_multi = args_multi = shared = 0
    labels = set()
    for r in recs:
        by = defaultdict(list)
        for e in r["output"]["events"]:
            by[str(e.get("event_type"))].append(e)
            labels.add(str(e.get("event_type")))
            labels.update(str(a.get("role")) for a in e.get("arguments") or [])
        for es in by.values():
            events += len(es)
            if len(es) < 2:
                continue
            in_multi += len(es)
            owners = defaultdict(set)
            for k, e in enumerate(es):
                for a in e.get("arguments") or []:
                    if _arg_text(a):
                        owners[_arg_text(a)].add(k)
            for e in es:
                for a in e.get("arguments") or []:
                    if _arg_text(a):
                        args_multi += 1
                        shared += len(owners[_arg_text(a)]) > 1
    return {"inflation": values / max(text_args, 1), "same_type": in_multi / max(events, 1),
            "shared": shared / max(args_multi, 1), "null": sorted(x for x in labels if NULL.match(x.strip()))}


def relation_row(recs, colls) -> dict:
    """Inflation (all vs nearest) through the collator; pairing and null labels from the records."""
    infl = {}
    for mode, coll in colls.items():
        edges = targets = 0
        for r in recs:
            edges += sum(len(x) for x in r["output"]["relations"])
            random.seed(0)
            et = getattr(coll([(r["input"], r["output"])]).targets, "edge_targets", None)
            targets += int(et[1].sum()) if isinstance(et, tuple) else 0
        infl[mode] = targets / max(edges, 1)
    true_edges = combos = 0
    labels = set()
    for r in recs:
        by = defaultdict(lambda: [set(), set(), set()])
        for x in r["output"]["relations"]:
            for t, v in x.items():
                labels.add(t)
                if isinstance(v, dict):
                    by[t][0].add(v.get("head")); by[t][1].add(v.get("tail")); by[t][2].add((v.get("head"), v.get("tail")))
        for hs, ts, es in by.values():
            true_edges += len(es); combos += len(hs) * len(ts)
    return {"all": infl["all"], "nearest": infl["nearest"], "pairing": combos / max(true_edges, 1),
            "null": sorted(x for x in labels if NULL.match(x.strip().replace("_", " ")))}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="tools/train/config/base/eb22b.yaml")
    ap.add_argument("--docs", type=int, default=200, help="train records per corpus (first N that carry the task)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    head = cfg["model"]["boundary_head"]
    # RAW labels for the null scan (a v4 map already drops `no relation`); the collator sees the config's map.
    fns, passthrough = T._category_fns(T.load_labels_cfg(cfg, args.config)), T.labels_passthrough(cfg)
    from gliner2 import AutoExtractor
    proc = AutoExtractor.from_pretrained("whr778/gliner2-eb21", map_location="cpu").processor
    mk = lambda mode: ExtractorCollator(proc, is_training=True, max_len=cfg["training"].get("max_len"), architecture="boundary",
                                       max_gold_per_query=int(head.get("max_gold_per_query", 32)),
                                       on_capacity_exceeded="truncate_with_warning", error_policy="skip",
                                       event_records=bool(head.get("event_records")), relation_gold_pairs=mode)
    colls = {"all": mk("all"), "nearest": mk("nearest")}
    ev_rows, rel_rows = [], []
    for c in [Path(p).name for p in cfg["data"]["corpora"]]:
        path = f"data/{c}.train.jsonl"
        if not Path(path).is_file():
            continue
        raw = [json.loads(line) for _, line in zip(range(5000), open(path, encoding="utf-8"))]
        xfm = T.read_transformed([path], fns, passthrough)[:5000]
        ev = [r for r in xfm if r["output"].get("events")][:args.docs]
        rel = [r for r in xfm if r["output"].get("relations")][:args.docs]
        if ev:
            row = event_row(ev, colls["all"])
            row["null"] = sorted(set(row["null"]) | {x for r in raw for e in (r.get("output") or {}).get("events") or []
                                                      for x in [str(e.get("event_type"))] + [str(a.get("role")) for a in e.get("arguments") or []]
                                                      if NULL.match(x.strip())})
            ev_rows.append((c, len(ev), row)); print(f"[events] {c}: {row}", file=sys.stderr, flush=True)
        if rel:
            row = relation_row(rel, colls)
            row["null"] = sorted(set(row["null"]) | {t for r in raw for x in (r.get("output") or {}).get("relations") or []
                                                      if isinstance(x, dict) for t in x if NULL.match(t.strip().replace("_", " "))})
            rel_rows.append((c, len(rel), row)); print(f"[relations] {c}: {row}", file=sys.stderr, flush=True)
    print(f"Measured by `tools/data/measure_target_structure.py --config {args.config} --docs {args.docs}` (first {args.docs} train "
          "records per corpus that carry the task; inflation through eb21's collator).\n")
    print("**Event corpora**\n\n| corpus | docs | argument inflation | events sharing a doc with a same-type event | of their arguments, shared with a sibling | null-like labels |")
    print("|---|--:|--:|--:|--:|---|")
    for c, n, r in ev_rows:
        print(f"| {c} | {n} | x{r['inflation']:.2f} | {100 * r['same_type']:.0f}% | {100 * r['shared']:.0f}% | {', '.join(r['null']) or 'none'} |")
    print("\n**Relation corpora**\n\n| corpus | docs | pair inflation, all | pair inflation, nearest | pairing ambiguity | null-like labels |")
    print("|---|--:|--:|--:|--:|---|")
    for c, n, r in rel_rows:
        print(f"| {c} | {n} | x{r['all']:.2f} | x{r['nearest']:.2f} | {r['pairing']:.2f} | {', '.join(r['null']) or 'none'} |")


if __name__ == "__main__":
    main()
