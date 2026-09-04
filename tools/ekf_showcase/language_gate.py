"""The relevance gate reads exactly the languages ITS OWN MODEL was trained on. Say so
at ingest, because "multilingual" is a property of a checkpoint, not of mmBERT.

Turkish was excluded here from 2026-08-29 until 2026-09-04 because the DEFAULT gate
checkpoint at the time, `gate2-mmbert-v2`, could not read it: trained on
`data/gate2.train.jsonl` (n=11,781) at English 95.9%, Chinese 4.1% (duee), Turkish 0.19%
and incidental, it scored AUC 0.4733 on Turkish -- below chance, negatives' median above
positives' (`gate_turkish_fp.py --sweep`) -- while still admitting 22% of clean Turkish
articles at threshold 0.5, silently, indistinguishable from what it did to genuine
Turkish casualty reports. Detecting the language first turned that silent 22%
false-positive rate into an explicit, countable exclusion.

`gate2-mmbert-tr` (trained on Turkish news specifically, `data/gate2_tr`) closed the gap
Turkish AUC 0.4980 (the SAME chance-level failure, confirmed independently on a second,
larger held-out set) -> 0.8105 across 9 outlets, with English unregressed (Helene pooled
RMSE identical at 17.5 on the streams both gates cover) -- and it is now
`run_pipeline.py`'s default `--gate-model`. Turkish moved from excluded to supported
because the MODEL changed, not because this file's language detection got smarter. Passing
an older gate checkpoint here (e.g. `--gate-model whr778/gliner2-gate2-mmbert-v2`) would
let Turkish text back through this allowlist to a model that still cannot read it -- the
allowlist governs which languages CAN reach the gate, not what any given checkpoint does
with them once there.

Two measured details drive the rule:

* **Han override.** All 18 rows of `data/gate2.test.jsonl` that lumi_language_id calls
  neither `en` nor `zh` are CHINESE -- short headlines ("无锡一小吃店爆炸6死9伤", 24 chars)
  that fall to `und` at prob 0.36-0.50, or to `ja`. Length, not language. Script settles
  it where the classifier cannot.
* **Fail closed otherwise.** Costs nothing here: after the override, 1441/1441 test rows
  are supported, and all 300 Turkish articles are `tr` at high confidence. Uncertainty
  about a language we cannot handle should not resolve in favour of running the gate.
"""
from __future__ import annotations

from lumi_language_id import detect_language

SUPPORTED = frozenset({"en", "zh", "tr"})
MAX_CHARS = 1500          # matches fetch_cc_news.py; detection needs a sample, not the document
HAN_FRACTION = 0.10       # of the letters, not the characters -- CJK text is punctuation-light


def _han_fraction(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    han = sum(1 for c in letters if 0x3400 <= ord(c) <= 0x9FFF)
    return han / len(letters)


def language_of(text: str) -> tuple[str, float]:
    """Detected language, with Han script overriding a low-confidence guess."""
    sample = text[:MAX_CHARS]
    lang, prob = detect_language(sample)
    if lang not in SUPPORTED and _han_fraction(sample) >= HAN_FRACTION:
        return "zh", prob
    return lang, float(prob)


def is_supported(text: str) -> tuple[bool, str, float]:
    """(gate can read this, detected language, detector confidence)."""
    lang, prob = language_of(text)
    return lang in SUPPORTED, lang, prob
