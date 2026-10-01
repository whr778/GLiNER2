"""Trace casualty_docee paragraphs to their source snippet's ground truth and classify gold gaps."""
import json, re, sys, glob
from collections import Counter, defaultdict
sys.path.insert(0, "datasets/disaster_streams")
from build_multievent_corpus import _standalone, _variants, _locate_in_slice
src = defaultdict(dict); origin = {}
for f in glob.glob("datasets/*/*/observations.jsonl") + glob.glob("datasets/*/observations.jsonl"):
    for line in open(f, encoding="utf-8"):
        o = json.loads(line)
        if o.get("text"):
            src[o["text"]][o["role"]] = o["value"]; origin.setdefault(o["text"], f)
print(f"[index] {len(src):,} distinct source snippet texts")
c, ex, files = Counter(), defaultdict(list), Counter()
for line in open("data/casualty_docee.train.jsonl", encoding="utf-8"):
    r = json.loads(line); doc = r["input"]; recs = r["output"].get("json_structures") or []
    gold_vals = set(v for s in recs for k, v in s["casualty_report"].items() if k != "location")
    pos = 0
    for i, p in enumerate(doc.split("\n\n")):
        lo = doc.index(p, pos); hi = lo + len(p); pos = hi
        gt = src.get(p)
        if gt is None: c["UNMATCHED"] += 1; continue
        files[origin[p].split("/")[1]] += 1
        located = {role: _locate_in_slice(v, doc, lo, hi) for role, v in gt.items()}
        in_text = {role: any(re.search(rf"(?<![\d,]){re.escape(cand)}(?![\d]|,\d)", p) for cand in _variants(v)) for role, v in gt.items()}
        covered = any(str(v) in gold_vals or f"{v:,}" in gold_vals for v in gt.values())
        if covered:
            c["covered"] += 1
            if any(located[k] is None for k in gt) and not all(located[k] is None for k in gt): c["covered_partially"] += 1
            continue
        if all(located[k] is not None for k in gt): why = "UNEXPLAINED_all_locatable"
        elif any(not in_text[k] for k in gt) and all((located[k] is None) for k in gt) and not any(in_text[k] and located[k] is None for k in gt): why = "realizer_changed_number"
        elif all(located[k] is None for k in gt) and all(in_text[k] for k in gt): why = "collision"
        elif all(located[k] is None for k in gt): why = "mixed_changed_and_collision"
        else: why = "UNEXPLAINED_some_locatable"
        c[why] += 1; c["first_para:" + why] += (i == 0)
        if len(ex[why]) < 3: ex[why].append((i, gt, p[:220]))
print("[sources]", dict(files))
for k in sorted(c): print(f"  {k}: {c[k]:,}")
for why, xs in ex.items():
    for i, gt, p in xs: print(f"\n[{why}] para {i} gt={gt}\n   {p!r}")
