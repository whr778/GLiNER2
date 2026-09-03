"""Two-stage screen for casualty-annotation candidates: free regex, then the extractor.

REPRODUCES data/cas_ann_ccnews.jsonl. Pipeline, all deterministic:

    fetch_cc_news.py --count 490000 --seed 1 --exclude data/cc_news_parts/*_raw.jsonl
    screen_ccnews_candidates.py <fetched> <candidates> mps 34000
    annotate_casualty.py --corpora <candidates> --batch --out data/cas_ann_ccnews

Measured on 488,710 CC-News documents: the regex keeps 24.7%, the extractor keeps 28.0%
of those, giving 33,832 candidates -> 33,516 unique -> 12,491 annotated rows (37.5%).

Original docstring follows.

Two-stage screen: free regex first, then the casualty extractor on what survives.

The regex is free and fires on 22.59% of CC-News; of those, 30.8% also carry a model
toll. Running the model only on regex survivors therefore does ~4x less GPU work for the
same candidate set -- the composition the annotation economics measured as the cheapest
column, applied to selection rather than to pricing.
"""
import json, re, sys, time
from pathlib import Path
sys.path.insert(0, "tools/data")
from gate_purity_curve import PREFILTERS
from gliner2 import AutoExtractor, Schema

SRC   = Path(sys.argv[1])
OUT   = Path(sys.argv[2])
DEV   = sys.argv[3] if len(sys.argv) > 3 else "mps"
TARGET= int(sys.argv[4]) if len(sys.argv) > 4 else 0
MODEL = "whr778/gliner2-base-v1-casualty-docee"

rx = PREFILTERS["en"]; NUM = re.compile(r"\d")
survivors = []
seen = 0
for line in SRC.open(encoding="utf-8"):
    r = json.loads(line); seen += 1
    t = (r.get("input") or "").strip()
    if t and rx.search(t):
        survivors.append(t[:2000])
print(f"  regex: {len(survivors)}/{seen} survive ({100*len(survivors)/max(seen,1):.1f}%)", flush=True)

model = AutoExtractor.from_pretrained(MODEL, map_location=DEV)
schema = (Schema().structure("casualty_report", mode="natural", anchor="dead")
          .field("dead", dtype="str", description="number of people killed or confirmed dead")
          .field("injured", dtype="str", description="number of people injured")
          .field("missing", dtype="str", description="number of people missing")
          .field("location", dtype="str", description="where the casualties occurred")).to_dict()

kept, t0 = [], time.time()
B = 200
with OUT.open("w", encoding="utf-8") as fh:
    for i in range(0, len(survivors), B):
        chunk = survivors[i:i+B]
        for text, p in zip(chunk, model.batch_extract(chunk, [schema]*len(chunk),
                                                      batch_size=8, threshold=0.3)):
            recs = (p or {}).get("casualty_report") or []
            if any(NUM.search(str(r.get(k) or "")) for r in recs for k in ("dead","injured","missing")):
                kept.append(text)
                fh.write(json.dumps({"input": text, "source": "cc_news-screened"},
                                    ensure_ascii=False) + "\n")
        done = min(i+B, len(survivors)); el = time.time()-t0
        print(f"  scored {done}/{len(survivors)}  kept {len(kept)}  "
              f"{done/el:.1f}/s  eta {(len(survivors)-done)/(done/el)/60:.0f}m", flush=True)
        if TARGET and len(kept) >= TARGET:
            print(f"  target {TARGET} reached, stopping early", flush=True)
            break
print(f"DONE screened {len(kept)} candidates -> {OUT}")
