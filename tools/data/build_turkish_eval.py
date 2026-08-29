"""A HELD-OUT Turkish evaluation slice, from outlets the gate corpus never saw.

The Turkish training data is TRT Haber alone (2024). A model trained on it that scores well
on TRT proves nothing about Turkish -- it may have learned one outlet's register, which is
the confound that invalidated the v1 gate corpus. This builds the set that can tell those
apart: same language, different publishers, different years.

Source: umutertugrul/turkish-news-1.8M-tokenized, which carries raw `text` and a `url`, so
the publisher is recoverable. Sampled across the full index rather than contiguously,
because the dataset is ORDERED BY SOURCE IN BLOCKS -- a contiguous slice is single-outlet
and would reproduce the very confound this set exists to detect.

Outlets seen: aa.com.tr (state agency), sozcu.com.tr (opposition), t24.com.tr
(independent), haberler.com (aggregator). URLs date it 2016-2023, against the training
corpus's 2024, so this tests generalisation across time as well as publisher.

    uv run python tools/data/build_turkish_eval.py --per-outlet 400
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record, normalize_group_key  # noqa: E402
from build_turkish_candidates import CUE, MAX_CHARS  # noqa: E402

ROWS = ("https://datasets-server.huggingface.co/rows?dataset=umutertugrul%2F"
        "turkish-news-1.8M-tokenized&config=default&split=train&offset={}&length=100")
TOTAL = 1_790_000


def fetch(offset: int) -> list[dict]:
    req = urllib.request.Request(ROWS.format(offset), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as fh:
        return [r["row"] for r in json.load(fh).get("rows", [])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-outlet", type=int, default=400)
    ap.add_argument("--calls", type=int, default=140, help="rows-API calls to spread")
    ap.add_argument("--out", default="data/turkish_eval_candidates.jsonl")
    a = ap.parse_args()

    by_outlet: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    step = TOTAL // a.calls
    for i in range(a.calls):
        try:
            rows = fetch(i * step)
        except Exception as exc:  # the public API rate-limits; skip rather than abort
            print(f"  offset {i*step}: {type(exc).__name__}", flush=True)
            continue
        for row in rows:
            text = (row.get("text") or "").strip()[:MAX_CHARS]
            outlet = urlparse(row.get("url") or "").netloc.replace("www.", "")
            if len(text) < 400 or not outlet or not CUE.search(text):
                continue
            if len(by_outlet[outlet]) >= a.per_outlet:
                continue
            key = normalize_group_key(text)[:300]
            if key in seen:
                continue
            seen.add(key)
            by_outlet[outlet].append(text)
        if i % 20 == 0:
            print(f"  call {i}/{a.calls}: {dict(Counter({k: len(v) for k, v in by_outlet.items()}))}",
                  flush=True)
        if all(len(v) >= a.per_outlet for v in by_outlet.values()) and len(by_outlet) >= 4:
            break

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for outlet, texts in sorted(by_outlet.items()):
            for text in texts:
                fh.write(dumps_record({"input": text, "source": f"tr_{outlet}"}) + "\n")
                n += 1
    print(f"[turkish-eval] {n} cue-bearing candidates -> {out}")
    print(f"[turkish-eval] by outlet: {dict({k: len(v) for k, v in sorted(by_outlet.items())})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
