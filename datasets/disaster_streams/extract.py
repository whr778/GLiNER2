"""Text -> observation extraction + normalization (design sec 9, decision #9).

The tracker consumes structured observations (role, value, qualifier, source); real
input is news *text*. This module inverts that: a surface extractor (integers + hedge/
source keyword cues -- general, not template-specific) feeds a normalization layer that
maps the surface form to the tracker's obs schema (incl. bucket words -> a representative
value with wide uncertainty).

Fail-cheap validation ($0): render each structured obs back to text with the generator's
own templater, extract it, and compare recovered fields to the known truth. This measures
the near-perfect-extraction ceiling before spending on realistic (sonnet-5) text.

  uv run python datasets/disaster_streams/extract.py --split val
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))
import generate  # noqa: E402  (sibling script: _templated_text for validation rendering)

# Bucket word -> representative value. dozens/hundreds match the generator's geometric-mean
# midpoints; thousands/few are inherently lossy from text alone (coarse reporting) -> a
# nominal stand-in, which the tracker treats as high-variance via QUAL_FACTOR["interval"].
_BUCKET_VALUE = {"dozens": 32, "hundreds": 316, "thousands": 2000, "few": 5}


def _detect_source(t: str) -> str:
    tl = t.lower()
    if "early report" in tl:
        return "preliminary"
    if "authorit" in tl or "official" in tl:
        return "official"
    return "major_outlet"


def _detect_role(t: str) -> Optional[str]:
    tl = t.lower()
    for r in ("injured", "missing", "dead"):
        if r in tl:
            return r
    return None


def _detect_qualifier(t: str) -> str:
    tl = t.lower()
    if "at least" in tl:
        return "at_least"
    if "about" in tl or "roughly" in tl or "around" in tl:
        return "about"
    if "feared" in tl:
        return "feared"
    if any(b in tl for b in _BUCKET_VALUE):
        return "interval"
    return "point"


def value_qualifier(text: str):
    """Normalize a span/snippet to (value, qualifier). The normalization layer (design
    #9): reused both by the surface parser and to parse a model-bound field span."""
    qual = _detect_qualifier(text)
    if qual == "interval":
        tl = text.lower()
        bucket = next((b for b in _BUCKET_VALUE if b in tl), "few")
        return _BUCKET_VALUE[bucket], qual
    m = re.search(r"\d[\d,]*", text)
    return (int(m.group(0).replace(",", "")) if m else 0), qual


def qualifier_near(text: str, span: str, window: int = 45) -> str:
    """Detect the hedge local to a model-bound number. The model located the digits; the
    qualifier ('at least'/'feared'/'about'/a bucket word) sits right beside them, usually
    outside the extracted span -- so read the qualifier from the number's context, not the
    bare span. Falls back to the span if the number can't be located."""
    m = re.search(r"\d[\d,]*", span)
    key = m.group(0) if m else span
    i = text.find(key)
    if i < 0:
        return _detect_qualifier(span)
    # the hedge usually precedes the number ("at least 40", "about 300"); a small right
    # margin avoids pulling in an adjacent fact's hedge in multi-fact prose.
    return _detect_qualifier(text[max(0, i - window): i + len(key) + 8])


def extract_obs(text: str) -> Dict:
    """Surface extraction + normalization: text -> {role, value, qualifier, source}."""
    value, qual = value_qualifier(text)
    return {"role": _detect_role(text), "value": value,
            "qualifier": qual, "source": _detect_source(text)}


def _render(o: Dict) -> str:
    return o["text"] if "text" in o else generate._templated_text(o)


def evaluate_extraction(split_dir: Path) -> None:
    total = 0
    hit = {"role": 0, "qualifier": 0, "source": 0}
    by_qual = defaultdict(lambda: {"n": 0, "val_exact": 0, "val_relabs": 0.0})
    for line in (split_dir / "observations.jsonl").open(encoding="utf-8"):
        o = json.loads(line)
        r = extract_obs(_render(o))
        total += 1
        for f in hit:
            hit[f] += int(r[f] == o[f])
        q = o["qualifier"]; bq = by_qual[q]; bq["n"] += 1
        bq["val_exact"] += int(r["value"] == o["value"])
        bq["val_relabs"] += abs(r["value"] - o["value"]) / max(o["value"], 1)

    print(f"\n== extraction on {split_dir} ({total} observations) ==\n")
    for f in ("role", "qualifier", "source"):
        print(f"  {f:10s} accuracy: {hit[f] / total:.4f}")
    print("\n  value recovery by qualifier (exact-match, mean rel-abs error):")
    for q in sorted(by_qual):
        b = by_qual[q]
        print(f"    {q:9s} n={b['n']:5d}  exact={b['val_exact'] / b['n']:.3f}  "
              f"rel_err={b['val_relabs'] / b['n']:.3f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/disaster_streams")
    ap.add_argument("--split", default="val")
    args = ap.parse_args(argv)
    evaluate_extraction(Path(args.data) / args.split)


if __name__ == "__main__":
    main()
