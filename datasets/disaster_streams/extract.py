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


# Spelled-out numbers. News prose writes small tolls as words -- "two people were
# killed" -- and reading only digits turned every one of them into a FABRICATED ZERO.
# Measured on the Helene feed once extract_long read whole articles rather than the lead:
# 30 of 114 `dead` observations (26%) were zeros produced this way, from spans like
# 'two', 'three', 'One', 'Nine', 'six'. A real extraction became a report of no deaths,
# which is worse than missing it -- the filter cannot tell the difference.
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}

# Turkish writes EVERY number >= 1000 as digits around a scale WORD -- "3 bin 549" is
# 3,549, not 3 -- so `\d[\d,]*` reads the leading group and stops. Measured on the
# Turkish Turkiye-2023 feed: the whole extracted trajectory topped out at 1,602 against
# a ground truth reaching 41,000, because "3 bin 549" -> 3, "22 bin 168" -> 22,
# "41 bin 20" -> 41. The extractor was often binding the CORRECT full span; this layer
# destroyed it afterwards, which is why it read as a model failure.
#
# Handled here rather than in word_number(): that path is only reached when a span has
# NO digits, and these spans are digit-bearing.
_TR_SCALES = {"bin": 1_000, "milyon": 1_000_000, "milyar": 1_000_000_000}
# Turkish agglutinates onto the scale word ("20 bini geçti" = passed 20,000), so match a
# prefix and allow a suffix rather than requiring a bare word.
_TR_NUM = re.compile(
    r"(?:(\d[\d.]*)\s*)?\b(bin|milyon|milyar)\w*(?:\s*(\d[\d.]*))?", re.I)


def _plain_int(tok: str) -> Optional[int]:
    """Digits with Turkish/German '.' thousands separators, or English ','.

    A dot is a thousands separator only when it splits into 3-digit groups; "1.5"
    stays a decimal so English "1.5 million" is untouched.
    """
    tok = tok.strip().rstrip(".")
    if not tok:
        return None
    if "." in tok:
        head, *rest = tok.split(".")
        if rest and all(len(g) == 3 and g.isdigit() for g in rest) and head.isdigit():
            return int(head + "".join(rest))
        return None
    return int(tok.replace(",", "")) if tok.replace(",", "").isdigit() else None


def turkish_number(text: str) -> Optional[int]:
    """`3 bin 549` -> 3549, `20 bini` -> 20000, `1.014` -> 1014. None if no match."""
    m = _TR_NUM.search(text)
    if m:
        head, scale, tail = m.group(1), m.group(2).lower(), m.group(3)
        mult = _TR_SCALES[scale]
        n = (_plain_int(head) if head else None) or 1      # bare "bin kisi" = 1000
        rest = _plain_int(tail) if tail else 0
        return n * mult + (rest or 0)
    m = re.search(r"\d[\d.]*\d|\d", text)                  # dot-separated, no scale word
    return _plain_int(m.group(0)) if m else None


def word_number(text: str) -> Optional[int]:
    """Spelled-out cardinal in `text`, or None. Handles 'twenty-three', 'two hundred'.

    Returns None rather than 0 when nothing parses, so "no number here" stays
    distinguishable from "the number is zero" -- collapsing those is the defect this
    exists to fix.
    """
    tokens = re.findall(r"[a-z]+", text.lower().replace("-", " "))
    total = current = 0
    seen = False
    for tok in tokens:
        if tok in _UNITS:
            current += _UNITS[tok]; seen = True
        elif tok in _TENS:
            current += _TENS[tok]; seen = True
        elif tok in _SCALES:
            if not seen:                      # "hundreds of" is a BUCKET, not a count
                return None
            scale = _SCALES[tok]
            if scale == 100:
                current *= scale
            else:
                total += max(current, 1) * scale
                current = 0
        elif tok in ("and",) and seen:
            continue
        elif seen:
            break                              # the number ended; do not run on
    return (total + current) if seen else None


def value_qualifier(text: str):
    """Normalize a span/snippet to (value, qualifier). The normalization layer (design
    #9): reused both by the surface parser and to parse a model-bound field span.

    A value of 0 now means the text really said zero. When nothing parses at all the
    value is None, and callers drop the observation instead of recording a phantom
    report of no casualties.
    """
    qual = _detect_qualifier(text)
    if qual == "interval":
        tl = text.lower()
        bucket = next((b for b in _BUCKET_VALUE if b in tl), "few")
        return _BUCKET_VALUE[bucket], qual
    # Turkish scale words FIRST: they are digit-bearing, so the plain digit branch below
    # would match their leading group and stop ("3 bin 549" -> 3).
    if re.search(r"\b(bin|milyon|milyar)\w*", text, re.I):
        tr = turkish_number(text)
        if tr is not None:
            return tr, qual
    m = re.search(r"\d[\d,]*(?:\.\d{3})*", text)
    if m:
        got = _plain_int(m.group(0))
        if got is not None:
            return got, qual
    return word_number(text), qual


def qualifier_near(text: str, span: str, left: int = 45, right: int = 8) -> str:
    """Detect the hedge local to a model-bound number. The model located the digits; the
    qualifier sits right beside them, usually outside the extracted span -- and usually
    PRECEDES it ('at least 40', 'about 300'). A wide right window catches a trailing 'feared'
    but hurts end-to-end (a sparse decay filter needs feared readings as usable anchors, not
    down-weighted), so keep the right margin small. Falls back to the span if not located."""
    m = re.search(r"\d[\d,]*", span)
    key = m.group(0) if m else span
    i = text.find(key)
    if i < 0:
        return _detect_qualifier(span)
    return _detect_qualifier(text[max(0, i - left): i + len(key) + right])


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
        # value is None when nothing parsed. Score it as a MISS rather than crashing or
        # coercing to 0 -- coercion is exactly the confusion this change removes.
        bq["val_exact"] += int(r["value"] is not None and r["value"] == o["value"])
        bq["val_relabs"] += (abs(r["value"] - o["value"]) / max(o["value"], 1)
                             if r["value"] is not None else 1.0)

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
