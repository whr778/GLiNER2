import json, re, sys
from pathlib import Path
from functools import lru_cache
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure import WORDS
def word(w):
    w = w.lower()
    return w in WORDS or any(w.endswith(s) and w[:-len(s)] in WORDS for s in ("s", "es", "ed", "d", "ing", "ies")) or (w.endswith("ies") and w[:-3] + "y" in WORDS)
@lru_cache(None)
def seg(s):
    """Fewest-words split of s into dictionary words (min word len 2, except 'a'); None if impossible."""
    if not s: return ()
    best = None
    for i in range(len(s), 0, -1):
        h = s[:i]
        if (len(h) >= 2 or h == "a") and word(h):
            rest = seg(s[i:])
            if rest is not None and (best is None or len(rest) + 1 < len(best)):
                best = (h,) + rest
    return best
if __name__ == "__main__":
    r = json.load(open(Path(__file__).resolve().parent / "report.json"))
    for cat, d in r.items():
        real = []
        for group in d["squashed_only"]:
            lab = max(group, key=len)
            segs = [s for s in lab.split(".") if s.isalpha() and len(s) >= 8 and not word(s)]
            if segs: real.append((lab, [seg(s.lower()) for s in segs]))
        print(f"\n{cat}: {len(real)} squashed-only labels with a non-word segment")
        ok = sum(all(x is not None for x in sg) for _, sg in real)
        print(f"  dictionary segmenter splits {ok} of {len(real)}")
        for lab, sg in real[:14]:
            print(f"   {lab:55} {['_'.join(x) if x else '???' for x in sg]}")
