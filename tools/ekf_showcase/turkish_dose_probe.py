"""BAR 1 of the Turkish dose curve: does the extractor read Turkish now?

GOLD-FREE, so no annotation error enters. `location` is a place name, so a digit there
means the model is binding numerals to whatever field is open rather than reading the
text. Reference points measured on the shipped extractor: 78.2% on Turkish against 5.8%
on real English news.

Scored on dw.com and t24.com.tr, which were reserved WHOLE from the annotation pool, so
this is cross-publisher as well as document-held-out: no arm trained on either outlet.

Run OFFLINE once the weights are cached. `from_pretrained` contacts the Hub to check for
updates even when every file is local, and on a hung network that sleeps forever instead
of failing -- it cost 11 hours of a probe sitting at 0.0% CPU on a dead socket:

    HF_HUB_OFFLINE=1 uv run python tools/ekf_showcase/turkish_dose_probe.py blind.jsonl <model>...
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gliner2 import AutoExtractor  # noqa: E402
from run_pipeline import build_casualty_schema  # noqa: E402

HAS_DIGIT = re.compile(r"\d")


def main() -> int:
    docs = [json.loads(l) for l in Path(sys.argv[1]).open(encoding="utf-8")]
    schema = build_casualty_schema(with_location=True)
    print(f"{len(docs)} blind Turkish documents (dw.com + t24.com.tr)\n")
    print(f"{'model':34s}{'recs':>7s}{'locs':>7s}{'digit-in-loc':>15s}{'smeared':>9s}")

    for model_id in sys.argv[2:]:
        model = AutoExtractor.from_pretrained(model_id, map_location="cpu")
        n_rec = n_loc = loc_digit = smeared = 0
        for doc in docs:
            found = model.extract_long(doc["input"], schema, threshold=0.3,
                                       chunk_size=200, chunk_overlap=50)
            for rec in found.get("casualty_report") or []:
                n_rec += 1
                location = str(rec.get("location") or "").strip()
                if location:
                    n_loc += 1
                    loc_digit += bool(HAS_DIGIT.search(location))
                vals = [v for v in (str(rec.get(f) or "").strip()
                        for f in ("dead", "injured", "missing")) if v]
                if vals and max(vals.count(v) for v in set(vals)) >= 3:
                    smeared += 1
        rate = f"{loc_digit}/{n_loc} = {loc_digit / n_loc:.1%}" if n_loc else "no locations"
        sm = f"{smeared / n_rec:.1%}" if n_rec else "-"
        print(f"{model_id.split('/')[-1]:34s}{n_rec:>7d}{n_loc:>7d}{rate:>15s}{sm:>9s}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
