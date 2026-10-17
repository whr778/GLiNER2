"""Find POISON labels in an inference menu: labels whose mere presence suppresses other outputs.

    uv run tools/train/menu_poison_check.py --model whr778/gliner2-eb21 \\
        --menu tools/train/config/schemas/casie.json --data data/casie.val.jsonl --docs 40 \\
        --labels-file tools/train/config/labels/unified-full-v3.yaml --out casie_poison.json

Runs the model on the first ``--docs`` records three ways and counts events fired and exact
(event type, trigger) hits against the gold:
  1. ``gold``: each document offered only its own gold labels (eval's menu), the reference;
  2. ``menu``: the whole menu, as an application would send it;
  3. ``menu minus X``: the whole menu with ONE label X removed, for every event role and event type
     (and every entity label with ``--entities``).
A label is a SUSPECT when removing it raises hits by at least ``--min-gain`` (default 5) AND by at
least half of the gold-menu hits the menu is missing. Suspects are then CONFIRMED greedily: the worst one
is removed, the remaining suspects are re-tried on that cleaner menu, and only those that still add
``--min-gain`` hits are poison too -- a partial recovery seen while a worse label is present is often a
side effect of it. Found this way on 2026-10-17: CASIE's role
``Damage_Amount`` zeroes every CASIE event in eb21 and corefC-treatment (full menu 1 hit vs 48 without
it, of 305 gold triggers on 40 docs). Training lists only argued roles on gold event types but the full
role list on injected absent ones, so a rarely argued role (26 gold vs 1,543 absent listings) became an
"absent" cue; the absent-type dose amplifies it. ``probe_poison_mechanism.py`` traces it.

Scan VAL, not test: choosing which labels to drop by reading test is fitting the test set.
The menu is mapped through the checkpoint's ``label_map`` exactly as ``infer.py --schema-json`` maps it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import infer as I

from gliner2.inference.label_map import apply_label_map


def gold_triggers(gold: dict) -> set:
    """(event type, casefolded trigger text) for every gold trigger."""
    return {(e["event_type"], t.casefold()) for e in gold.get("events") or []
            for t in (e.get("triggers") or [e.get("trigger")]) if t}


def pred_triggers(out: dict) -> set:
    return {(t, (tr["text"] if isinstance(tr, dict) else tr).casefold())
            for t, insts in (out.get("event_extraction") or {}).items() for i in insts for tr in i.get("triggers") or []}


def without(menu: dict, kind: str, label: str) -> dict:
    """The menu minus one entity label, event type, or event role (from every type that has it)."""
    if kind == "entity":
        ents = menu["entities"]
        return {**menu, "entities": {k: v for k, v in ents.items() if k != label} if isinstance(ents, dict)
                else [e for e in ents if e != label]}
    if kind == "type":
        return {**menu, "events": {t: r for t, r in menu["events"].items() if t != label}}
    return {**menu, "events": {t: [x for x in r if x != label] for t, r in menu["events"].items()}}


def candidates(menu: dict, entities: bool) -> list[tuple]:
    roles = sorted({r for rs in menu.get("events", {}).values() for r in rs})
    out = [("role", r) for r in roles] + [("type", t) for t in menu.get("events", {})]
    return out + ([("entity", e) for e in menu.get("entities", [])] if entities else [])


def run(model, recs, schemas, decode) -> tuple[int, list]:
    """(events fired, per-doc predicted trigger sets)."""
    out = model.batch_extract_long([r["input"] for r in recs], schemas, batch_size=1, **decode)
    return sum(len(v) for o in out for v in (o.get("event_extraction") or {}).values()), [pred_triggers(o) for o in out]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--menu", required=True, help="a schema or app-menu JSON (build_schema.py writes one)")
    ap.add_argument("--data", required=True, help="a .jsonl with gold (use the VAL split)")
    ap.add_argument("--docs", type=int, default=40)
    ap.add_argument("--labels-file", help="the unified labels YAML the model trained with (gold in model spellings)")
    ap.add_argument("--entities", action="store_true", help="also try removing each entity label (slower)")
    ap.add_argument("--min-gain", type=int, default=5)
    ap.add_argument("--threshold", type=float, help="span threshold; default the checkpoint's inference_defaults")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    from gliner2 import AutoExtractor
    model = AutoExtractor.from_pretrained(args.model, map_location=args.device)
    stored = getattr(model.config, "inference_defaults", None) or {}
    decode = {**{k: stored[k] for k in ("threshold", "chunk_size", "chunk_overlap", "global_decode") if k in stored},
              **({"threshold": args.threshold} if args.threshold is not None else {})}
    recs = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line][:args.docs]
    if args.labels_file:
        fns, _, transform = I._labels_file(args.labels_file)
        golds = [transform(r, fns)["output"] for r in recs]
    else:
        golds = [r["output"] for r in recs]
    raw = json.loads(Path(args.menu).read_text(encoding="utf-8"))
    menu, _ = apply_label_map(raw.get("schema", raw), getattr(model.config, "label_map", None),
                              getattr(model.config, "label_style", None))
    gt = [gold_triggers(g) for g in golds]
    n_gold = sum(map(len, gt))
    print(f"[poison] {args.model} | {len(recs)} docs of {args.data} | {n_gold} gold triggers | decode {decode}")

    def score(name, schemas):
        fired, preds = run(model, recs, schemas, decode)
        hits = sum(len(p & g) for p, g in zip(preds, gt))
        print(f"[poison] {name:45s} events fired {fired:5d} | exact type+trigger hits {hits:4d} / {n_gold}", flush=True)
        return {"fired": fired, "hits": hits}

    result = {"model": args.model, "menu_file": args.menu, "data": args.data, "docs": len(recs), "gold_triggers": n_gold,
              "gold": score("gold (each doc's own labels)", [I._gold_schema({"output": g}) for g in golds]),
              "menu": score("menu", [menu] * len(recs)), "minus": {}}
    for kind, label in candidates(menu, args.entities):
        result["minus"][f"{kind}:{label}"] = score(f"menu minus {kind} {label}", [without(menu, kind, label)] * len(recs))
    base, ref = result["menu"]["hits"], result["gold"]["hits"]
    suspects = sorted((k for k, v in result["minus"].items() if v["hits"] - base >= max(args.min_gain, (ref - base) / 2)),
                      key=lambda k: -result["minus"][k]["hits"])
    print(f"[poison] suspects (removal adds >= max({args.min_gain}, half the {ref - base} hits the menu loses vs gold)): "
          f"{suspects or 'none'}")
    poison, clean, hits = [], menu, base
    while suspects:
        top = suspects.pop(0)
        clean = without(clean, *top.split(":", 1))
        poison.append(top)
        hits = score(f"confirmed so far {poison}", [clean] * len(recs))["hits"]
        retry = {k: score(f"  + minus {k}", [without(clean, *k.split(":", 1))] * len(recs))["hits"] for k in suspects}
        suspects = sorted((k for k, h in retry.items() if h - hits >= args.min_gain), key=lambda k: -retry[k])
    result["poison"], result["clean_menu_hits"] = poison, hits
    print(f"[poison] POISON labels: {poison or 'none'} | menu {base} -> {hits} hits without them (gold menu {ref})")
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
