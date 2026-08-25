"""Does the relevance gate hold up on non-English input? MEASURED on Turkish news.

The gate has only ever been tested on English. That matters now: the third event is
Turkish, and real coverage of it is in Turkish, so extending the feed beyond
English-language outlets depends on an assumption nobody has checked.

This is the same shape of test that caught v1. That gate described its negative class as
topically distant filler; benchmarked against real annotated messages it admitted 58.5% of
definitively non-disaster text at high confidence. v2 rewrote the descriptions around the
negatives it actually meets. Neither version was ever run on another language.

Two models, because the shipped default cannot do this by construction:

    fastino/gliner2-base-v1    DeBERTa-v3, vocab 128,011  -- ENGLISH ONLY, the gate default
    fastino/gliner2-multi-v1   mDeBERTa-v3, vocab 250,112 -- multilingual

Running only the default would produce a foregone conclusion rather than a measurement.
The label descriptions stay in ENGLISH against Turkish text, which is the cross-lingual
zero-shot setting the pipeline would actually be in.

Source: denizzhansahin/Turkish_News-2024 (TRT Haber, 19,170 articles, 14 categories).
Negatives are articles with no disaster word AND no casualty word; positives have both.
Those are heuristic labels, not gold -- so the positive rate is a sanity check on whether
the gate reads Turkish at all, and the NEGATIVE rate is the number that matters.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline import build_gate_schema                     # noqa: E402

DIS = re.compile(r"\b(deprem|sel\b|yangın|patlama|çöktü|göçük|tsunami|heyelan)", re.I)
TOLL = re.compile(r"\b(öldü|ölü|can kaybı|hayatını kaybet|yaralı|yaraland)", re.I)
MODELS = ("fastino/gliner2-base-v1", "fastino/gliner2-multi-v1")


def sample(n_neg, n_pos, seed, max_chars):
    from datasets import load_dataset
    ds = load_dataset("denizzhansahin/Turkish_News-2024", split="train")
    neg, pos = [], []
    for r in ds:
        t = ((r["Baslik"] or "") + ". " + (r["Icerik"] or "")).strip()
        if len(t) < 400:
            continue
        d, c = bool(DIS.search(t)), bool(TOLL.search(t))
        (pos if (d and c) else neg if not (d or c) else []).append(t[:max_chars])
    rng = random.Random(seed)
    rng.shuffle(neg); rng.shuffle(pos)
    return neg[:n_neg], pos[:n_pos]


def run(model_id, texts, threshold, device):
    from gliner2 import AutoExtractor
    model = AutoExtractor.from_pretrained(model_id, map_location=device)
    schema = build_gate_schema(model)
    admitted, labels, t0 = 0, [], time.time()
    for t in texts:
        r = model.extract(t, schema, include_confidence=True)
        rel = r.get("relevance")
        lab = rel.get("label") if isinstance(rel, dict) else rel
        conf = float(rel.get("confidence", 1.0)) if isinstance(rel, dict) else 1.0
        labels.append((lab, conf))
        admitted += bool(lab == "mass_casualty" and conf >= threshold)
    return admitted, labels, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-neg", type=int, default=200)
    ap.add_argument("--n-pos", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-chars", type=int, default=1800)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=20260825)
    a = ap.parse_args()

    neg, pos = sample(a.n_neg, a.n_pos, a.seed, a.max_chars)
    print(f"sampled {len(neg)} negatives, {len(pos)} heuristic positives "
          f"(threshold {a.threshold}, {a.max_chars} chars max)\n")
    print(f"  {'model':30}{'FP on negatives':>18}{'admitted positives':>21}{'sec':>8}")
    for m in MODELS:
        fp, _, t1 = run(m, neg, a.threshold, a.device)
        tp, _, t2 = run(m, pos, a.threshold, a.device)
        tag = "  <- shipped default, ENGLISH-only" if "base" in m else "  <- multilingual"
        print(f"  {m:30}{fp:>8}/{len(neg):<9}{tp:>11}/{len(pos):<9}{t1 + t2:>8.0f}{tag}")
    print("\n  The FP column is the number that matters. v1's failure was 58.5%.")


if __name__ == "__main__":
    main()
