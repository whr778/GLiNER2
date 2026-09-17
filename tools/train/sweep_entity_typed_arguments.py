"""Sweep: can the model's OWN predicted entity types improve event-argument precision?

OneIE conditions role classification on entity type, because an argument IS an entity node.
GLiNER2 makes an argument its own role query with no link to the entity head. This measures
whether that link can be recovered at DECODE, using predicted types rather than gold -- so it
runs on corpora with no entity annotation at all, which is where the event mass actually is
(cmnee is 85.7% of the blind test and has zero entity gold).

A 120-document pilot found the signal is real but uneven: 88% of CORRECT arguments carry a
predicted entity type against 68% of WRONG ones, and the per-role detail split exactly as
casie's GOLD did -- TYPE-NAMED roles (`Date`, `Location`) discriminate, FUNCTION-NAMED roles
(`Subject`) do not. This sweeps the two things that pilot fixed arbitrarily: the entity menu,
and whether the constraint is applied globally or only to type-named roles.

Runs the EVAL path (`_schema_from_gold` schemas through `model.batch_extract`) so the numbers
are comparable to every argument figure on file. One decode pass per menu serves every filter
variant, because the filters are post-hoc over the same predictions.

    uv run python tools/train/sweep_entity_typed_arguments.py --checkpoint <ckpt> --n 800
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

MENUS = {
    "none": [],
    "minimal": ["Person", "Organization", "Location", "Date", "Time", "Number"],
    "curated14": ["Person", "Organization", "Location", "Country", "Date", "Time",
                  "Weapon", "Equipment", "Vehicle", "Aircraft", "Ship", "Facility",
                  "Military Unit", "Number"],
    "wide28": ["Person", "Organization", "Location", "Country", "City", "Region",
               "Date", "Time", "Duration", "Number", "Quantity", "Money",
               "Weapon", "Equipment", "Vehicle", "Aircraft", "Ship", "Submarine",
               "Missile", "Facility", "Military Unit", "Government", "Company",
               "Event", "Product", "Technology", "Document", "Title"],
}


def gold_triples(output: dict) -> set:
    out = set()
    for ev in output.get("events") or []:
        if isinstance(ev, dict):
            for a in ev.get("arguments") or []:
                if isinstance(a, dict) and a.get("entity"):
                    out.add((ev.get("event_type"), a.get("role"), str(a["entity"]).strip()))
    return out


def pred_triples_and_types(pred: dict):
    """(argument triples, surface -> predicted entity types) from one prediction."""
    ents = defaultdict(set)
    for lab, items in ((pred.get("entities") or {}) if isinstance(pred, dict) else {}).items():
        for it in (items if isinstance(items, list) else [items]):
            sfc = it.get("text") if isinstance(it, dict) else it
            if isinstance(sfc, str):
                ents[sfc.strip()].add(lab)
    args = set()
    for t, insts in ((pred.get("event_extraction") or {}) if isinstance(pred, dict) else {}).items():
        for inst in (insts if isinstance(insts, list) else [insts]):
            if not isinstance(inst, dict):
                continue
            for a in inst.get("arguments") or []:
                if isinstance(a, dict) and a.get("entity"):
                    args.add((t, a.get("role"), str(a["entity"]).strip()))
    return args, ents


def type_named(role: str, menu: list) -> bool:
    """A role is TYPE-NAMED if its name matches a menu label -- data-driven, not a hand list.

    `Date` matches `Date`; `Location` matches `Location`; `Subject` matches nothing. This is
    the split both the casie gold and the cmnee predictions independently produced.
    """
    if not role:
        return False
    r = role.lower().replace("_", " ").strip()
    return any(r == lab.lower() or r in lab.lower() or lab.lower() in r for lab in menu)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test", default="data/cmnee.test.jsonl")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from gliner2 import AutoExtractor
    from gliner2.training.eval_metrics import _schema_from_gold

    model = AutoExtractor.from_pretrained(args.checkpoint, map_location=args.device)
    getattr(model, "model", model).to(torch.device(args.device)).eval()

    recs = [json.loads(l) for l in Path(args.test).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    recs = [r for r in recs if (r.get("output") or {}).get("events")][:args.n]
    print(f"[sweep] {len(recs)} documents from {args.test}, threshold {args.threshold}\n")

    print(f"{'menu':11s}{'filter':12s}{'P':>9}{'R':>9}{'F1':>9}{'TP':>8}{'FP':>8}{'dropped':>9}")
    baseline_f1 = None
    for menu_name, menu in MENUS.items():
        texts, schemas, golds = [], [], []
        for r in recs:
            sch = _schema_from_gold(r["output"])
            if not sch:
                continue
            sch = dict(sch)
            if menu:
                sch["entities"] = {e: "" for e in menu}
            texts.append(r["input"]); schemas.append(sch); golds.append(r["output"])
        preds = model.batch_extract(texts, schemas, batch_size=args.batch_size,
                                    threshold=args.threshold)

        variants = ["none"] if not menu else ["none", "global", "per-role"]
        for variant in variants:
            tp = fp = fn = dropped = 0
            for pred, gold in zip(preds, golds):
                g = gold_triples(gold)
                a, ents = pred_triples_and_types(pred)
                if variant == "global":
                    kept = {x for x in a if ents.get(x[2])}
                elif variant == "per-role":
                    # constrain ONLY where the role is type-named; leave function-named alone
                    kept = {x for x in a
                            if (not type_named(x[1], menu)) or ents.get(x[2])}
                else:
                    kept = a
                dropped += len(a) - len(kept)
                tp += len(kept & g); fp += len(kept - g); fn += len(g - kept)
            p, r, f = prf(tp, fp, fn)
            if baseline_f1 is None:
                baseline_f1 = f
            print(f"{menu_name:11s}{variant:12s}{p:>9.4f}{r:>9.4f}{f:>9.4f}"
                  f"{tp:>8}{fp:>8}{dropped:>9}")
    print("\nbaseline is menu=none: the eval schema as it ships, with the entity head never "
          "queried on a corpus that has no entity gold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
