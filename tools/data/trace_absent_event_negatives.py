"""Trace absent-event negatives exactly as training injects them, and test whether "absent" is true.

Reproduces ExtractorDataset.__getitem__ -> NegativeLabels.inject on a config's real training records
(real build_pools, seed, partial_annotation), then flags an injected absent event type whose
known trigger surface occurs in the document: an UPPER bound, and a TIGHTENED rate counting only
unambiguous surfaces (>=80% of the surface's trigger uses are that type, >=5 uses). Both are
detector rates, not contradiction rates: read the printed sentences.

    uv run python tools/data/trace_absent_event_negatives.py
"""
import importlib.util, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, "tools/train"); import train as T, yaml
from gliner2.training.negatives import NegativeLabels
cfg_path = "tools/train/config/base/eb18-balanced.yaml"; cfg = yaml.safe_load(open(cfg_path))
fns = T._category_fns(T.load_labels_cfg(cfg, cfg_path))
spec = importlib.util.spec_from_file_location("bnp", "tools/data/build_negative_pools.py"); bnp = importlib.util.module_from_spec(spec); spec.loader.exec_module(bnp)
pools = bnp.build_pools(cfg, Path(cfg_path))
inj = NegativeLabels(pools, cfg["training"]["negative_labels_per_dim"], seed=42, partial=cfg["data"].get("partial_annotation"))
CJK = re.compile(r"[㐀-鿿]")
paths = bnp._corpus_train_paths(cfg)
event_corpora = [n for n, p in paths.items() if (pools.get(n) or {}).get("annotates", {}).get("events")]
print(f"[trace] corpora with event pools: {event_corpora}")
recs = {}
lex = defaultdict(Counter)                       # event type -> trigger surface counts (all corpora)
for n in event_corpora:
    rows = [T.transform_record(json.loads(l), fns) for l in open(paths[n], encoding="utf-8") if l.strip()]
    recs[n] = rows
    for r in rows:
        for e in r["output"].get("events") or []:
            for t in e.get("triggers") or []:
                lex[e["event_type"]][t.lower()] += 1
def present(text, surf):
    if CJK.search(surf):
        return len(surf) >= 2 and surf in text
    return len(surf) >= 3 and re.search(rf"(?<!\w){re.escape(surf)}(?!\w)", text.lower()) is not None
for n in event_corpora:
    rng = random.Random(0); sample = rng.sample(recs[n], min(1500, len(recs[n])))
    c, ex, src = Counter(), [], Counter()
    for i, r in enumerate(sample):
        out = inj.inject(r["output"], i, n)
        absent = out.get("absent_events") or {}
        if not absent: c["no_injection"] += 1; continue
        gold_trig = {t.lower() for e in r["output"].get("events") or [] for t in e.get("triggers") or []}
        for etype in absent:
            c["injected"] += 1
            src["own_taxonomy" if etype in pools[n].get("events", {}) else "OTHER_corpus"] += 1
            hits = [s for s, k in lex[etype].most_common() if k >= 3 and s not in gold_trig and present(r["input"], s)]
            if hits:
                c["contains_known_trigger"] += 1
                if len(ex) < 3: ex.append((etype, hits[:3], [e["event_type"] for e in r["output"]["events"]][:3]))
    inj_n = c["injected"]
    print(f"[trace] {n:24s} docs {len(sample):5d} | injected {inj_n:5d} ({dict(src)}) | absent type's known trigger IN the text: "
          f"{c['contains_known_trigger']:5d} = {100*c['contains_known_trigger']/max(inj_n,1):5.1f}% (upper bound)")
    for e in ex: print(f"[trace]      e.g. absent {e[0]!r} but text has {e[1]} | doc gold types {e[2]}")

# ---- TIGHTENED: unambiguous trigger surfaces only ----
by_surface = defaultdict(Counter)
for etype, c in lex.items():
    for s, k in c.items():
        by_surface[s][etype] += k
def unambiguous(etype, s):
    tot = sum(by_surface[s].values())
    return tot >= 5 and by_surface[s][etype] / tot >= 0.8
def sentence(text, surf):
    i = text.lower().find(surf) if not CJK.search(surf) else text.find(surf)
    return text[max(0, i - 70): i + len(surf) + 50].replace("\n", " ")
print("\n[tight] ---- unambiguous surfaces only (>=80% of the surface's trigger uses are this type, >=5 uses) ----")
for n in event_corpora:
    rng = random.Random(0); sample = rng.sample(recs[n], min(1500, len(recs[n])))
    inj.stats = {k: 0 for k in inj.stats}
    c, ex = Counter(), []
    for i, r in enumerate(sample):
        absent = inj.inject(r["output"], i, n).get("absent_events") or {}
        gold_trig = {t.lower() for e in r["output"].get("events") or [] for t in e.get("triggers") or []}
        for etype in absent:
            c["injected"] += 1
            hits = [s for s in lex[etype] if s not in gold_trig and unambiguous(etype, s) and present(r["input"], s)]
            if hits:
                c["hit"] += 1
                if len(ex) < 4: ex.append((etype, hits[0], sentence(r["input"], hits[0])))
    if c["injected"]:
        print(f"[tight] {n:24s} injected {c['injected']:5d} | unambiguous trigger of the 'absent' type in text: {c['hit']:4d} = {100*c['hit']/c['injected']:5.1f}%")
        for e in ex: print(f"[tight]      absent {e[0]!r} / {e[1]!r}: ...{e[2]}...")
