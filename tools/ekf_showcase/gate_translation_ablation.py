"""Is the Turkish false-positive rate a LANGUAGE problem or a LABEL problem?

The multilingual gate false-admits 28% of clean Turkish news. Two hypotheses, and they
call for completely different fixes:

  A. CROSS-LINGUAL TRANSFER. The label descriptions are fine; the model just reads Turkish
     worse than English. Fix = translate at ingest, or fine-tune the encoder.
  B. THE LABEL DESCRIPTIONS. They were rewritten (v1 -> v2) around the negatives an ENGLISH
     feed meets. Turkish news has a different negative distribution and the descriptions do
     not cover it. Fix = rewrite the descriptions; translation would change nothing.

The ablation separates them. Take the articles the multilingual model actually got WRONG,
translate them to English, and re-run. If the English versions pass, it is A. If they fail
the same way, it is B, and no amount of translation helps.

Translation is by Claude Haiku 4.5 -- cheap, and the fidelity bar is low because we only
need the topic to survive, not the prose.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline import build_gate_schema                       # noqa: E402
from gate_turkish_fp import DIS, TOLL                            # noqa: E402

SYSTEM = ("Translate the Turkish news article into natural English. Preserve every number, "
          "place name and factual claim exactly. Do not summarise, editorialise or add "
          "anything. Return only the translation.")


def load_negatives(n, seed, max_chars):
    from datasets import load_dataset
    ds = load_dataset("denizzhansahin/Turkish_News-2024", split="train")
    neg = []
    for r in ds:
        t = ((r["Baslik"] or "") + ". " + (r["Icerik"] or "")).strip()
        if len(t) >= 400 and not (DIS.search(t) or TOLL.search(t)):
            neg.append((r["Kategori"], t[:max_chars]))
    rng = random.Random(seed)
    rng.shuffle(neg)
    return neg[:n]


def gate_labels(model_id, texts, threshold, device):
    from gliner2 import AutoExtractor
    model = AutoExtractor.from_pretrained(model_id, map_location=device)
    schema = build_gate_schema(model)
    out = []
    for t in texts:
        r = model.extract(t, schema, include_confidence=True)
        rel = r.get("relevance")
        lab = rel.get("label") if isinstance(rel, dict) else rel
        conf = float(rel.get("confidence", 1.0)) if isinstance(rel, dict) else 1.0
        out.append(bool(lab == "mass_casualty" and conf >= threshold))
    return out


def translate(texts, cache: Path):
    import anthropic
    cache.parent.mkdir(parents=True, exist_ok=True)
    have = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
    client = anthropic.Anthropic()
    out = []
    for t in texts:
        k = str(abs(hash(t)))
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
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--cache", default="data/turkish_translation_cache.json")
    a = ap.parse_args()

    rows = load_negatives(a.n, a.seed, a.max_chars)
    cats = [c for c, _ in rows]
    tr = [t for _, t in rows]
    print(f"{len(tr)} clean Turkish negatives sampled")

    multi_tr = gate_labels("fastino/gliner2-multi-v1", tr, a.threshold, a.device)
    print(f"  multilingual on TURKISH   : {sum(multi_tr):>3}/{len(tr)} false-admitted")

    en = translate(tr, Path(a.cache))
    print(f"  translated {len(en)} to English")

    multi_en = gate_labels("fastino/gliner2-multi-v1", en, a.threshold, a.device)
    base_en = gate_labels("fastino/gliner2-base-v1", en, a.threshold, a.device)
    print(f"  multilingual on ENGLISH   : {sum(multi_en):>3}/{len(en)} false-admitted")
    print(f"  english model on ENGLISH  : {sum(base_en):>3}/{len(en)} false-admitted"
          f"   <- the gate in its designed setting")

    print("\n  verdict:")
    if sum(base_en) / max(len(en), 1) < 0.10:
        print("    Translation FIXES it -> hypothesis A, cross-lingual transfer.")
    else:
        print("    English versions fail too -> hypothesis B, the LABEL DESCRIPTIONS.")
        print("    Translation would not help; the descriptions need rewriting.")

    bad = [(c, e[:150]) for c, e, f in zip(cats, en, base_en) if f]
    if bad:
        print("\n  English translations the ENGLISH gate still admits:")
        for c, e in bad[:6]:
            print(f"    [{c}] {re.sub(chr(10), ' ', e)}")


if __name__ == "__main__":
    main()
