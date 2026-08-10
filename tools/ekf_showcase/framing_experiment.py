"""Why does record extraction depend on how the input is framed?

Observed, unexplained: the standfirst sentence ALONE yields 0 records; the same sentence
plus one more yields 2 correctly-bound records; the full 6.2k-char article yields 1.
Before building anything on the record head, characterise that.

Design. The gold-bearing sentence is held FIXED and identical in every condition -- the
only thing that varies is what surrounds it. So any change in output is attributable to
framing, not to the sentence.

Two factors, crossed:
  - context volume: 0, 40, 100, 250, 500, 1000, 2000 chars of the real following text,
    plus the full article.
  - cut cleanliness: context truncated at a word boundary vs mid-word, to separate
    "not enough context" from "ragged input confuses it".

Measured per condition: how many records, and whether each country's gold figure is bound
to the right country. Run over several documents so nothing rests on one example.
"""
import json, re, sys
from collections import defaultdict
from gliner2 import AutoExtractor, Schema

ROWS = [json.loads(l) for l in open("datasets/turkey2023/_cache/feed.jsonl", encoding="utf-8")]
GT = {p["snapshot"][:10]: (p["turkey"], p["syria"])
      for p in json.loads(open("datasets/turkey2023/ground_truth.json", encoding="utf-8").read())["points"]}
CONTEXTS = [0, 40, 100, 250, 500, 1000, 2000]


def schema():
    return (Schema().structure("casualty_report")
            .field("dead", dtype="str", description="number of people killed or confirmed dead")
            .field("location", dtype="str", description="the country or place these deaths occurred in"))


def num(x):
    d = re.sub(r"[^\d]", "", str(x or ""))
    return int(d) if d else None


def records(m, text):
    r = m.extract(text, schema())
    out = []
    for rec in r.get("casualty_report") or []:
        out.append({k: (v.get("text") if isinstance(v, dict)
                        else (v[0] if isinstance(v, list) and v else v)) for k, v in rec.items()})
    return out


def score(recs, t, s):
    hit_t = any(num(x.get("dead")) == t and "turk" in str(x.get("location", "")).lower() for x in recs)
    hit_s = any(num(x.get("dead")) == s and "syri" in str(x.get("location", "")).lower() for x in recs)
    return hit_t, hit_s


def main():
    m = AutoExtractor.from_pretrained("fastino/gliner2-base-v1", map_location="cpu")
    agg = defaultdict(lambda: [0, 0, 0, 0])          # cond -> [nrec, turkey, syria, docs]

    for row in ROWS:
        text = row["text"]
        end = text.find(". ")                        # end of the standfirst sentence
        if end < 0:
            continue
        end += 1
        core = text[:end]
        t, s = GT[row["date"]]

        for k in CONTEXTS:
            for clean in (True, False):
                if k == 0 and not clean:
                    continue                          # no context to cut raggedly
                ctx = text[end:end + k]
                if clean and k:
                    ctx = ctx[:ctx.rfind(" ")] if " " in ctx else ctx
                cond = f"+{k:<5}{'clean' if clean else 'ragged'}"
                recs = records(m, core + ctx)
                ht, hs = score(recs, t, s)
                a = agg[cond]
                a[0] += len(recs); a[1] += ht; a[2] += hs; a[3] += 1

        recs = records(m, text)
        ht, hs = score(recs, t, s)
        a = agg["FULL article"]
        a[0] += len(recs); a[1] += ht; a[2] += hs; a[3] += 1

    print(f"{'condition':<18} {'recs/doc':>9} {'Turkiye':>9} {'Syria':>7}")
    for cond, (n, ht, hs, d) in agg.items():
        print(f"{cond:<18} {n / d:>9.2f} {ht:>6}/{d:<3} {hs:>4}/{d:<3}")


if __name__ == "__main__":
    main()
