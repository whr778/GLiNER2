"""Ingest the Venezuela reporting stream -> observations (real double-blind test).

Reads `sources.jsonl` (url, outlet, published, tier), loads each article's text from a
GIT-IGNORED cache (`data/venezuela_text_cache/<sha1(url)>.txt`, populated separately so
copyrighted text never enters git), runs the fine-tuned casualty extractor, and writes
`observations.jsonl` -- extracted figures only. The tracker `source` is the outlet TIER
(provenance), not the model's weak source field; values/qualifiers come from the model.
Reuses `datasets/disaster_streams/model_arm.py` so the extraction path matches the
synthetic eval exactly (the whole point of a double-blind).

  # 1. cache article text (copyright: stays gitignored) -- e.g. via WebFetch
  # 2. uv run python datasets/venezuela_2026/ingest.py --model out/casualty-finetune/best
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, "datasets/disaster_streams")
import extract  # noqa: E402  (normalizer; imported for parity/side-effect-free reuse)
import model_arm  # noqa: E402

ORIGIN = date(2026, 6, 24)  # mainshock day; t_hours origin
CACHE = Path("data/venezuela_text_cache")
ROLES = ("dead", "injured", "missing")


def cache_path(url: str) -> Path:
    return CACHE / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".txt")


def t_hours(published: str) -> int:
    y, m, d = (int(x) for x in published.split("-")[:3])
    return (date(y, m, d) - ORIGIN).days * 24


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/venezuela_2026")
    ap.add_argument("--model", default="out/casualty-finetune/best")
    ap.add_argument("--threshold", type=float, default=0.9)
    args = ap.parse_args(argv)

    root = Path(args.data)
    sources = [json.loads(l) for l in (root / "sources.jsonl").open(encoding="utf-8")]

    from gliner2 import GLiNER2
    ex = GLiNER2.from_pretrained(args.model, map_location="cpu")
    schema = model_arm.build_schema()

    rows, missing = [], []
    for s in sources:
        cp = cache_path(s["url"])
        if not cp.exists():
            missing.append(s)
            continue
        pred, _ = model_arm.extract_one(ex, schema, cp.read_text(encoding="utf-8"), args.threshold)
        th = t_hours(s["published"])
        for role in ROLES:
            if role in pred:
                rows.append({
                    "stream_id": "venezuela_2026", "t_hours": th, "role": role,
                    "value": pred[role]["value"], "qualifier": pred[role]["qualifier"],
                    "source": s["tier"], "confidence": pred[role]["confidence"], "url": s["url"],
                })

    with (root / "observations.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[ingest] {len(sources)} sources, {len(missing)} missing cached text; "
          f"wrote {len(rows)} observations -> {root / 'observations.jsonl'}")
    if missing:
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"[ingest] cache article text into {CACHE}/ first (copyright: gitignored):")
        for s in missing:
            print(f"   {cache_path(s['url']).name}  <-  {s['url']}")


if __name__ == "__main__":
    main()
