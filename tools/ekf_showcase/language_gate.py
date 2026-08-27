"""The relevance gate is an English+Chinese instrument. Say so at ingest.

The gate is named multilingual because mmBERT is. Its TRAINING DATA is not: measured on
`data/gate2.train.jsonl` (n=11,781), English 95.9% (docee + cc_news), Chinese 4.1%
(duee), Turkish 0.19% and incidental. On Turkish news the consequence is not a degraded
score, it is no signal at all -- AUC 0.4733, below chance, with the negatives' median
score above the positives' (`gate_turkish_fp.py --sweep`). It still admits 22% of clean
Turkish articles at threshold 0.5, silently, and that admission rate is indistinguishable
from what it does to Turkish casualty reports.

Detecting the language first turns a silent 22% false-positive rate into an explicit,
countable exclusion. It does not make the gate multilingual; it stops it pretending.

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

SUPPORTED = frozenset({"en", "zh"})
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
