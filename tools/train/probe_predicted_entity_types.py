"""Does the model's OWN predicted entity type carry signal about argument correctness?

This is option 3 feeding option 2, and it needs no entity GOLD -- only an entity-capable
model, which ours is. Uses the eval path exactly: `_schema_from_gold` schemas through
`model.batch_extract`, with an entity menu ADDED so the entity head is actually queried
(cmnee has no entity gold, so the eval schema normally contains no entities at all and the
head is never asked).
"""
import json, torch
from collections import Counter, defaultdict
from pathlib import Path
from gliner2 import AutoExtractor
from gliner2.training.eval_metrics import _schema_from_gold

CK = "/Volumes/Development/tmp/ck_eb16-eventrecords-tr"
N = 120
ENTITY_MENU = ["Person", "Organization", "Location", "Country", "Date", "Time",
               "Weapon", "Equipment", "Vehicle", "Aircraft", "Ship", "Facility",
               "Military Unit", "Number"]

m = AutoExtractor.from_pretrained(CK, map_location="mps")
getattr(m, "model", m).to(torch.device("mps")).eval()

recs = [json.loads(l) for l in open("data/cmnee.test.jsonl", encoding="utf-8") if l.strip()]
recs = [r for r in recs if (r.get("output") or {}).get("events")][:N]
texts, schemas, golds = [], [], []
for r in recs:
    sch = _schema_from_gold(r["output"])
    if not sch: continue
    sch = dict(sch); sch["entities"] = {e: "" for e in ENTITY_MENU}   # the intervention
    texts.append(r["input"]); schemas.append(sch); golds.append(r["output"])

preds = m.batch_extract(texts, schemas, batch_size=8, threshold=0.3)

stat = {"correct": Counter(), "wrong": Counter()}
per_role = defaultdict(lambda: {"correct": Counter(), "wrong": Counter()})
n_corr = n_wrong = 0
for pred, gold in zip(preds, golds):
    g = set()
    for ev in gold.get("events") or []:
        if isinstance(ev, dict):
            for a in ev.get("arguments") or []:
                if isinstance(a, dict) and a.get("entity"):
                    g.add((ev.get("event_type"), a.get("role"), str(a["entity"]).strip()))
    # surface -> predicted entity type(s)
    ent = defaultdict(set)
    for lab, items in ((pred.get("entities") or {}) if isinstance(pred, dict) else {}).items():
        for it in (items if isinstance(items, list) else [items]):
            sfc = it.get("text") if isinstance(it, dict) else it
            if isinstance(sfc, str): ent[sfc.strip()].add(lab)
    for t, insts in ((pred.get("event_extraction") or {}) if isinstance(pred, dict) else {}).items():
        for inst in (insts if isinstance(insts, list) else [insts]):
            if not isinstance(inst, dict): continue
            for a in inst.get("arguments") or []:
                if not isinstance(a, dict) or not a.get("entity"): continue
                key = (t, a.get("role"), str(a["entity"]).strip())
                types = ent.get(key[2], set())
                bucket = "correct" if key in g else "wrong"
                if bucket == "correct": n_corr += 1
                else: n_wrong += 1
                stat[bucket]["ANY entity type predicted" if types else "no entity type"] += 1
                for ty in types: per_role[key[1]][bucket][ty] += 1

print(f"cmnee, {len(texts)} documents, entity menu of {len(ENTITY_MENU)} types\n")
print(f"predicted arguments: {n_corr} correct, {n_wrong} wrong\n")
print(f"{'':28s}{'correct':>10}{'wrong':>10}")
for k in ("ANY entity type predicted", "no entity type"):
    c, w = stat['correct'][k], stat['wrong'][k]
    print(f"{k:28s}{c:>10}{w:>10}"
          f"   ({100*c/n_corr if n_corr else 0:.0f}% vs {100*w/n_wrong if n_wrong else 0:.0f}%)")
print("\nper-role predicted entity types (correct | wrong):")
for role, d in sorted(per_role.items(), key=lambda kv: -sum(kv[1]['correct'].values()))[:6]:
    c = ", ".join(f"{t}:{n}" for t, n in d["correct"].most_common(3)) or "-"
    w = ", ".join(f"{t}:{n}" for t, n in d["wrong"].most_common(3)) or "-"
    print(f"   {str(role):16s} correct[{c}]  wrong[{w}]")
