"""Which menu axis silences CASIE events? One model, the first N CASIE test docs, per-doc menus varying ONE axis.

    uv run python tools/train/probe_menu_axes.py whr778/gliner2-eb21 40

Scratch probe kept for the record (catalog 2026-10-17): edit `runs` to pick the variants.
"""
import json, sys
sys.path.insert(0, "tools"); sys.path.insert(0, "tools/train")
import infer as I
from gliner2 import AutoExtractor
from gliner2.inference.label_map import apply_label_map

model_id, n = sys.argv[1], int(sys.argv[2])
model = AutoExtractor.from_pretrained(model_id, map_location="cpu")
fns, _, transform = I._labels_file("tools/train/config/labels/unified-full-v3.yaml")
recs = [json.loads(l) for l in open("data/casie.test.jsonl", encoding="utf-8")][:n]
golds = [transform(r, fns)["output"] for r in recs]
menu = json.load(open("tools/train/config/schemas/casie.json", encoding="utf-8"))["schema"]
app, _ = apply_label_map(menu, model.config.label_map, getattr(model.config, "label_style", None))
r0 = I._gold_schema({"output": golds[0]})["events"]
print("doc0 gold roles", r0, "| menu roles", {t: app["events"][t] for t in r0})
print("app menu types", list(app["events"]), "| entities", len(app["entities"]))
variants = {}
for g in golds:
    gs = I._gold_schema({"output": g})
    variants.setdefault("gold", []).append(gs)
    variants.setdefault("gold_ents+all5_events", []).append({**gs, "events": app["events"]})
    variants.setdefault("all21_ents+gold_events", []).append({**gs, "entities": app["entities"]})
    variants.setdefault("app_menu", []).append({"entities": app["entities"], "events": app["events"]})
    variants.setdefault("gold_types+menu_roles", []).append({**gs, "events": {t: app["events"][t] for t in gs.get("events", {}) if t in app["events"]}})
    ge = gs.get("events", {})
    variants.setdefault("gold_roles_sorted", []).append({**gs, "events": {t: sorted(r) for t, r in ge.items()}})
    variants.setdefault("gold_order+extra_menu_roles", []).append({**gs, "events": {t: list(r) + [x for x in app["events"].get(t, []) if x not in r] for t, r in ge.items()}})
    nod = {t: [x for x in rs if x != "Damage_Amount"] for t, rs in app["events"].items()}
    variants.setdefault("app_menu minus Damage_Amount", []).append({"entities": app["entities"], "events": nod})
    variants.setdefault("gold_types+menu_roles minus Damage_Amount", []).append({**gs, "events": {t: nod[t] for t in ge if t in nod}})
    for k in ():
        variants.setdefault(f"gold+{k} extra roles", []).append({**gs, "events": {t: list(r) + [x for x in app["events"].get(t, []) if x not in r and x != "Damage_Amount"][:k] for t, r in ge.items()}})
    variants.setdefault("roles per type", []).append([len(r) for r in ge.values()])
    for x in []:
        variants.setdefault(f"gold+role {x}", []).append({**gs, "events": {t: list(r) + ([x] if x in app["events"].get(t, []) and x not in r else []) for t, r in ge.items()}})
    absent = [t for t in app["events"] if t not in gs.get("events", {})]
    variants.setdefault("gold+1_absent_type", []).append({**gs, "events": {**gs.get("events", {}), **({absent[0]: app["events"][absent[0]]} if absent else {})}})
decode = dict(threshold=0.3, chunk_size=4096, chunk_overlap=0, global_decode=True)
gold_trig = [{(e["event_type"], t.casefold()) for e in g["events"] for t in (e.get("triggers") or [e.get("trigger")]) if t} for g in golds]
print("gold roles per type, first docs:", variants["roles per type"][:8])
runs = [(k, variants[k], 0.3) for k in ("gold", "app_menu", "app_menu minus Damage_Amount")]
print("types with Damage_Amount:", [t for t, rs in app["events"].items() if "Damage_Amount" in rs])
for name, scs, th in runs:
    out = model.batch_extract_long([r["input"] for r in recs], scs, batch_size=1, **{**decode, "threshold": th})
    name = f"{name} @{th}"
    fired = hit = 0
    for o, gt in zip(out, gold_trig):
        ev = o.get("event_extraction") or {}
        pred = {(t, (tr["text"] if isinstance(tr, dict) else tr).casefold()) for t, insts in ev.items() for i in insts
                for tr in (i.get("triggers") or [])}
        fired += sum(len(v) for v in ev.values()); hit += len(pred & gt)
    print(f"[axis] {name:26s} events fired {fired:4d} | gold triggers {sum(map(len, gold_trig))} | exact type+trigger hits {hit}", flush=True)
