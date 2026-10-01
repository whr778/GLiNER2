"""Per paragraph of casualty_docee: is it covered by gold, and if not, why (collision vs unique toll)."""
import json, re, sys
from collections import Counter
sys.path.insert(0, "datasets/disaster_streams")
from build_multievent_corpus import _standalone
NUM = re.compile(r"\d+(?:,\d{3})*")                      # never swallows a trailing comma
TOLL = re.compile(r"(\d+(?:,\d{3})*)(?=[^.\d]{0,40}?\b(killed|dead|died|deaths?|death toll|injured|hurt|wounded|missing|unaccounted)\b)")
YEAR = re.compile(r"^(19|20)\d\d$")
def hits(doc, v): return sum(_standalone(doc, m.start(), m.end()) for m in re.finditer(re.escape(v), doc))
def scan(path, examples=0):
    c, ex = Counter(), []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line); doc = r["input"]; recs = r["output"].get("json_structures") or []
        gold = set(v for s in recs for k, v in s["casualty_report"].items() if k != "location")
        pos = 0
        for i, p in enumerate(doc.split("\n\n")):
            lo = doc.index(p, pos); pos = lo + len(p)
            c["paras"] += 1
            if gold & {m.group() for m in NUM.finditer(p)}:
                c["covered"] += 1; continue
            tolls = [m.group(1) for m in TOLL.finditer(p) if not YEAR.match(m.group(1))]
            if not tolls: c["no_numeric_toll"] += 1; continue
            if any(hits(doc, t) == 1 for t in tolls):
                key = "UNIQUE_toll_unlabelled_" + ("first" if i == 0 else "later"); c[key] += 1
                if len(ex) < examples: ex.append((tolls, p[:230], sorted(gold)))
            else: c["tolls_all_collide"] += 1
    return c, ex
if __name__ == "__main__":
    for split in ("train", "val", "test"):
        c, ex = scan(f"data/casualty_docee.{split}.jsonl", examples=4 if split == "train" else 0)
        print(split, dict(c))
        for t, p, g in ex: print(f"   tolls {t} | {p!r}\n      doc gold values: {g}")
