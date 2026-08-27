"""Is the Turkish false-positive rate a LANGUAGE problem or a LABEL problem?

The multilingual gate false-admits 28% of clean Turkish news. Two hypotheses, and they
call for completely different fixes:

  A. CROSS-LINGUAL TRANSFER. The label descriptions are fine; the model just reads Turkish
     worse than English. Fix = translate at ingest, or fine-tune the encoder.
  B. THE LABEL DESCRIPTIONS. They were rewritten (v1 -> v2) around the negatives an ENGLISH
     feed meets. Turkish news has a different negative distribution and the descriptions do
     not cover it. Fix = rewrite the descriptions; translation would change nothing.

RESULT 2026-08-25 (fastino models, negatives only): hypothesis B. English translations of
the articles the multilingual model got wrong failed the same way -- the ENGLISH model
admitted 17/60 of them.

WHY IT IS BEING RUN AGAIN. That verdict is about multi-v1, which READS Turkish and
over-admits. `gate2-mmbert-v2` has the opposite failure: AUC 0.4733 on Turkish, below
chance, because its corpus is 95.9% English and 4.1% Chinese with no Turkish at all
([[gate-does-not-read-turkish]]). For THAT failure, translation is what the diagnosis
predicts should work -- so B does not transfer, and the question is open again.

BOTH CLASSES, because false positives alone cannot answer it. Scoring only the negatives
measures how often the gate wrongly fires; a gate that stops firing entirely scores
perfectly on that and is useless. Translating the positives too gives an AUC on English
that is directly comparable to the Turkish AUC from `gate_turkish_fp.py --sweep`.

Sampling is `gate_turkish_fp.sample`, same seed and definitions, so the Turkish numbers
here line up with the ones recorded there rather than being a fresh draw.

Translation is by Claude Haiku 4.5 -- cheap, and the fidelity bar is low because we only
need the topic to survive, not the prose. ~300 articles is roughly $1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_turkish_fp import sample, sweep                        # noqa: E402

SYSTEM = ("Translate the Turkish news article into natural English. Preserve every number, "
          "place name and factual claim exactly. Do not summarise, editorialise or add "
          "anything. Return only the translation.")


def translate(texts, cache: Path):
    import anthropic
    cache.parent.mkdir(parents=True, exist_ok=True)
    have = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
    client = anthropic.Anthropic()
    out = []
    for t in texts:
        # sha1, not hash(): Python randomizes string hashing per process, so the old
        # key never hit across runs and every invocation re-paid for every translation
        # (120 cached entries for a 60-article experiment).
        k = hashlib.sha1(t.encode("utf-8")).hexdigest()
        if k not in have:
            r = client.messages.create(
                model="claude-haiku-4-5", max_tokens=2048, system=SYSTEM,
                messages=[{"role": "user", "content": t}])
            have[k] = next((b.text for b in r.content if b.type == "text"), "")
            cache.write_text(json.dumps(have, ensure_ascii=False), encoding="utf-8")
        out.append(have[k])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-neg", type=int, default=200)
    ap.add_argument("--n-pos", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-chars", type=int, default=1800)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--cache", default="data/turkish_translation_cache.json")
    ap.add_argument("--models", nargs="+", default=["whr778/gliner2-gate2-mmbert-v2"],
                    help="gates to score in Turkish and again in English")
    a = ap.parse_args()

    neg, pos = sample(a.n_neg, a.n_pos, a.seed, a.max_chars)
    print(f"sampled {len(neg)} negatives, {len(pos)} heuristic positives "
          f"({a.max_chars} chars max)")

    cache = Path(a.cache)
    neg_en = translate(neg, cache)
    pos_en = translate(pos, cache)
    print(f"translated {len(neg_en) + len(pos_en)} articles to English\n")

    for model_id in a.models:
        print(f"=== {model_id} on TURKISH")
        turkish = sweep(model_id, neg, pos, a.device)
        print(f"\n=== {model_id} on the SAME articles in ENGLISH")
        english = sweep(model_id, neg_en, pos_en, a.device)
        print(f"\n  AUC {turkish:.4f} (Turkish) -> {english:.4f} (English), "
              f"delta {english - turkish:+.4f}")
        print("  Translation restores discrimination." if english - turkish > 0.10 else
              "  Translation does NOT restore discrimination -- the label descriptions,\n"
              "  not the language, are what this gate is failing on.")


if __name__ == "__main__":
    main()
