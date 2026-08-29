"""Does the casualty extractor actually READ a language, or just emit numbers?

A stage-0 gate that admits a new language feeds documents into a stage-2 extractor
trained almost entirely on English and Chinese (121 of 152,578 training rows look
Turkish = 0.079%). The dangerous failure is not silence -- it is a confident WRONG
figure, because the EKF consumes whatever number arrives as a measurement and has no
way to reject it.

Two gold-free signals, so no annotation error enters the measurement:
  A. `location` contains a digit. Locations are place names; a digit there means the
     model is binding numerals to whatever field is open. Language-neutral, so it also
     catches Turkish numerals like "644 bin 439".
  B. One identical value smeared across >=3 casualty fields of a single record
     (dead == missing == source), which no real report does.

Both are compared against an English control drawn from the real pipeline feed, using
the production parameters in run_pipeline.py (threshold 0.3, chunk 200/50). Without the
control these rates mean nothing: they could be how this extractor behaves everywhere.

Measured 2026-08-29 on whr778/gliner2-base-v1-casualty-docee, 60 docs per side:
    location contains a digit  TR 78.2%  EN  5.8%  OR 58.5  p=1.5e-39
    value smeared >=3 fields   TR 11.8%  EN  3.0%  OR  4.3  p=1.5e-04
The extractor does not read Turkish. It emits fluent-looking garbage.

Usage:
    uv run tools/ekf_showcase/extractor_language_probe.py [model] [n_docs]
"""
import json
import re
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gliner2 import AutoExtractor  # noqa: E402
from run_pipeline import build_casualty_schema  # noqa: E402

TURKISH = Path("data/turkish_gate/gate_ann_tr_heldout.jsonl")
ENGLISH = Path("datasets/helene2024/_cache/feed.jsonl")
HAS_DIGIT = re.compile(r"\d")


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Exact two-sided Fisher p for the 2x2 table [[a, b], [c, d]]."""
    n, row1, col1 = a + b + c + d, a + b, a + c

    def prob(x):
        return comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1)

    p_obs = prob(a)
    lo, hi = max(0, col1 - (n - row1)), min(row1, col1)
    return sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * 1.0000001)


def measure(model, schema, texts: list[str], tag: str) -> tuple[int, int, int, int]:
    """Return (digit_locations, total_locations, smeared_records, total_records)."""
    n_loc = loc_digit = n_rec = smeared = 0
    for text in texts:
        found = model.extract_long(text, schema, threshold=0.3, chunk_size=200,
                                   chunk_overlap=50).get("casualty_report") or []
        for rec in found:
            n_rec += 1
            location = str(rec.get("location") or "").strip()
            if location:
                n_loc += 1
                loc_digit += bool(HAS_DIGIT.search(location))
            values = [v for v in (str(rec.get(f) or "").strip()
                      for f in ("dead", "injured", "missing", "source")) if v]
            if values and max(values.count(v) for v in set(values)) >= 3:
                smeared += 1
    print(f"\n{tag}: {len(texts)} docs -> {n_rec} records")
    print(f"  location contains a DIGIT : {loc_digit}/{n_loc} = {loc_digit / n_loc:.1%}")
    print(f"  one value in >=3 fields   : {smeared}/{n_rec} = {smeared / n_rec:.1%}")
    return loc_digit, n_loc, smeared, n_rec


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "whr778/gliner2-base-v1-casualty-docee"
    n_docs = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    turkish = [json.loads(line)["input"]
               for line in TURKISH.open(encoding="utf-8")
               if json.loads(line)["label"] == "current_toll"][:n_docs]
    english = [json.loads(line)["text"]
               for line in ENGLISH.open(encoding="utf-8")][:n_docs]

    model = AutoExtractor.from_pretrained(model_id, map_location="cpu")
    schema = build_casualty_schema(with_location=True)

    tr_ld, tr_l, tr_sm, tr_r = measure(model, schema, turkish, "TURKISH (adjudicated current_toll)")
    en_ld, en_l, en_sm, en_r = measure(model, schema, english, "ENGLISH control (Helene feed)")

    print("\nExact Fisher, Turkish vs English:")
    for name, (a, b, c, d) in {
        "location contains a digit": (tr_ld, tr_l - tr_ld, en_ld, en_l - en_ld),
        "value smeared >=3 fields": (tr_sm, tr_r - tr_sm, en_sm, en_r - en_sm),
    }.items():
        odds = (a * d) / (b * c) if b and c else float("inf")
        print(f"  {name:26s} OR {odds:5.1f}  p={fisher_two_sided(a, b, c, d):.3g}")


if __name__ == "__main__":
    main()
