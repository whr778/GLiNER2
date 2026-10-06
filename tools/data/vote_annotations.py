"""Self-consistency gold: k annotation runs of the SAME documents -> one voted record per document.

An event in different runs is the SAME event when its type matches and the two share a trigger
mention (equal, or one inside the other, case-folded) -- coreferent trigger lists differ between
runs, so a shared mention, not identical lists, is the identity. Voting (``--min-votes``, default 2):

  events     kept when found in >= min runs; triggers = mentions listed by >= min of them (at
             least the most-listed one); arguments = (role, entity) given by >= min of them
  entities, relations, classification labels: kept when given by >= min runs
  json_structures: an instance is the same across runs when its name and ANCHOR value (from
             record_metadata) match; kept when found in >= min runs, each field value kept when
             >= min of them give it; record_metadata carried for the kept names

``--compare A B`` scores two voted files against each other at the EVENT level (the spec's
stability bar, ENGLISH_ANNOTATION_SPEC section 5) and for arguments.

    uv run python tools/data/vote_annotations.py --runs r1.jsonl r2.jsonl r3.jsonl --out voted.jsonl
    uv run python tools/data/vote_annotations.py --compare votedA.jsonl votedB.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, List

f = lambda s: str(s).strip().casefold()


def same_event(a: dict, b: dict) -> bool:
    """Same type and at least one shared trigger mention (equal or nested)."""
    if a["event_type"] != b["event_type"]:
        return False
    ta, tb = {f(t) for t in a.get("triggers") or []}, {f(t) for t in b.get("triggers") or []}
    return any(x == y or x in y or y in x for x in ta for y in tb)


def vote_events(runs: List[List[dict]], need: int) -> List[dict]:
    """Cluster events across runs (one per run per cluster) and keep the clusters with >= need runs."""
    clusters: List[Dict[int, dict]] = []
    for r, events in enumerate(runs):
        for ev in events:
            home = next((c for c in clusters if r not in c and any(same_event(ev, e) for e in c.values())), None)
            if home is None:
                clusters.append({r: ev})
            else:
                home[r] = ev
    out = []
    for c in clusters:
        if len(c) < need:
            continue
        evs = list(c.values())
        tcount = Counter(t for e in evs for t in dict.fromkeys(e.get("triggers") or []))
        triggers = [t for t, n in tcount.items() if n >= need] or [tcount.most_common(1)[0][0]]
        acount = Counter((a["role"], a["entity"]) for e in evs for a in {(x["role"], x["entity"]): x for x in e.get("arguments") or []}.values())
        args = [{"role": ro, "entity": en} for (ro, en), n in acount.items() if n >= need]
        out.append({"event_type": evs[0]["event_type"], "triggers": triggers, "arguments": args, "votes": len(c)})
    return out


def vote_structures(outputs: List[dict], need: int):
    """(json_structures, record_metadata) voted across runs, keyed by (name, anchor value)."""
    meta: Dict[str, dict] = {}
    for o in outputs:
        meta.update(o.get("record_metadata") or {})
    groups: Dict[tuple, List[dict]] = {}
    for o in outputs:
        seen = set()
        for inst in o.get("json_structures") or []:
            for name, fields in inst.items():
                anchor = (meta.get(name) or {}).get("anchor")
                key = (name, f(fields.get(anchor)) if anchor else json.dumps(fields, sort_keys=True))
                if key not in seen:
                    seen.add(key)
                    groups.setdefault(key, []).append(fields)
    out, names = [], set()
    for (name, _), insts in groups.items():
        if len(insts) < need:
            continue
        vals = Counter((k, json.dumps(v, sort_keys=True)) for fs in insts for k, v in fs.items())
        out.append({name: {k: json.loads(v) for (k, v), n in vals.items() if n >= need}})
        names.add(name)
    return out, {n: meta[n] for n in names if n in meta}


def vote_record(outputs: List[dict], need: int) -> dict:
    voted: dict = {"events": vote_events([o.get("events") or [] for o in outputs], need)}
    ents = Counter((t, s) for o in outputs for t, ss in (o.get("entities") or {}).items() for s in set(ss or []))
    voted["entities"] = {}
    for (t, s), n in ents.items():
        if n >= need:
            voted["entities"].setdefault(t, []).append(s)
    rels = Counter(json.dumps(r, sort_keys=True) for o in outputs for r in {json.dumps(x, sort_keys=True): x for x in o.get("relations") or []}.values())
    voted["relations"] = [json.loads(k) for k, n in rels.items() if n >= need]
    labs = Counter((c.get("task"), lab) for o in outputs for c in o.get("classifications") or []
                   for lab in set(c.get("true_label") or c.get("labels") or []))
    tasks: Dict[str, list] = {}
    for (task, lab), n in labs.items():
        if n >= need:
            tasks.setdefault(task, []).append(lab)
    voted["classifications"] = [{"task": t, "labels": ls} for t, ls in tasks.items()]
    voted["json_structures"], voted["record_metadata"] = vote_structures(outputs, need)
    return voted


def event_agreement(a_docs: List[dict], b_docs: List[dict]) -> dict:
    """Event and argument F1 between two annotations of the same docs (event identity = same_event)."""
    tp = na = nb = atp = ana = anb = 0
    for a, b in zip(a_docs, b_docs):
        ea, eb = list(a.get("events") or []), list(b.get("events") or [])
        na, nb = na + len(ea), nb + len(eb)
        used = set()
        for x in ea:
            j = next((j for j, y in enumerate(eb) if j not in used and same_event(x, y)), None)
            xa = {(r["role"], f(r["entity"])) for r in x.get("arguments") or []}
            ana += len(xa)
            if j is not None:
                used.add(j); tp += 1
                ya = {(r["role"], f(r["entity"])) for r in eb[j].get("arguments") or []}
                atp += len(xa & ya)
        anb += sum(len(y.get("arguments") or []) for y in eb)
    f1 = lambda t, p, g: 2 * t / (p + g) if p + g else float("nan")
    return {"events_f1": f1(tp, na, nb), "events": (na, nb, tp), "arguments_f1": f1(atp, ana, anb), "arguments": (ana, anb, atp)}


def load(path: str) -> Dict[str, dict]:
    return {r["input"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        A, B = load(args.compare[0]), load(args.compare[1])
        docs = sorted(set(A) & set(B))
        m = event_agreement([A[d]["output"] for d in docs], [B[d]["output"] for d in docs])
        print(f"[agree] {len(docs)} docs | events F1 {m['events_f1']:.3f} (A {m['events'][0]}, B {m['events'][1]}, matched {m['events'][2]}) "
              f"| arguments F1 {m['arguments_f1']:.3f} (A {m['arguments'][0]}, B {m['arguments'][1]}, agree {m['arguments'][2]})")
        return
    runs = [load(p) for p in args.runs]
    docs = sorted(set.intersection(*(set(r) for r in runs)))
    kept = Counter()
    with open(args.out, "w", encoding="utf-8") as fh:
        for d in docs:
            voted = vote_record([r[d]["output"] for r in runs], args.min_votes)
            kept["events"] += len(voted["events"])
            fh.write(json.dumps({"input": d, "output": voted}, ensure_ascii=False) + "\n")
    print(f"[vote] {len(runs)} runs x {len(docs)} docs, min {args.min_votes} -> {kept['events']} voted events -> {args.out}")


if __name__ == "__main__":
    main()
