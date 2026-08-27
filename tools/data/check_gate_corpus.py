"""Pre-flight: can this corpus be separated WITHOUT reading the words?

Run before spending anything on a gate fine-tune. A corpus that fails here cannot
teach the task -- whatever a model scores on it is a measurement of the shortcut.

History this exists to prevent. The first gate corpus drew negatives from Haiti-era SMS
and a length-only rule scored 98.5%. Length-matching it by TRUNCATING the negatives left
100% of positives ending on sentence punctuation against 33.5% of negatives. Under both
of those sat the real one: positives were 99.9% SYNTHETIC and every negative was real, so
the classes were separable on provenance and the trained gate rejected all 590 SMS and
all 71 Aegean news articles while scoring F1 1.0000 on its own test split.

So this checks two different things, because neither sees the other:

  SURFACE  -- length, punctuation, script, digits, case. Catches the first two.
  PROVENANCE -- generated-text markers per class. A surface check CANNOT catch that;
                a synthetic-vs-real corpus with matched lengths passes it cleanly.

    uv run python tools/data/check_gate_corpus.py data/gate2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# Phrasing that betrays generated text. Every one of these was measured at 86.5% of the
# positive class in casualty_events and 0% in the real corpora.
SYNTH = re.compile(
    r"(A major news outlet reported|An official government source|"
    r"A government spokesperson (?:said|confirmed)|Early, unconfirmed reports)", re.I)

_SENT_END = re.compile(r"[.!?。！？\"'\)\]]\s*$")
_CJK = re.compile(r"[一-鿿぀-ヿ]")
_CYR = re.compile(r"[Ѐ-ӿ]")
_ARAB = re.compile(r"[؀-ۿ]")

FEATURES = [
    "chars", "words", "mean_word_len", "ends_on_punct", "digit_frac", "punct_frac",
    "upper_frac", "space_frac", "newline_frac", "cjk_frac", "cyr_frac", "arab_frac",
    "comma_per_kchar", "period_per_kchar",
]


def featurize(text: str) -> list[float]:
    n = max(len(text), 1)
    words = text.split()
    return [
        len(text),
        len(words),
        sum(len(w) for w in words) / max(len(words), 1),
        1.0 if _SENT_END.search(text) else 0.0,
        sum(c.isdigit() for c in text) / n,
        sum(not c.isalnum() and not c.isspace() for c in text) / n,
        sum(c.isupper() for c in text) / n,
        sum(c.isspace() for c in text) / n,
        text.count("\n") / n,
        len(_CJK.findall(text)) / n,
        len(_CYR.findall(text)) / n,
        len(_ARAB.findall(text)) / n,
        text.count(",") * 1000 / n,
        text.count(".") * 1000 / n,
    ]


def load(prefix: str) -> tuple[list[str], np.ndarray]:
    """Texts and binary labels from a GLiNER2 classification corpus."""
    texts, labels = [], []
    for split in ("train", "val", "test"):
        path = Path(f"{prefix}.{split}.jsonl")
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            cls = ((rec.get("output") or {}).get("classifications") or [{}])[0]
            true = (cls.get("true_label") or [None])[0]
            pool = cls.get("labels") or []
            if true is None or len(pool) != 2:
                continue
            texts.append(rec["input"])
            labels.append(1 if true == pool[0] else 0)
    return texts, np.array(labels)


def stump(x: np.ndarray, y: np.ndarray) -> float:
    """Best accuracy from one threshold on one feature."""
    order = np.argsort(x)
    ys = y[order]
    pos = np.cumsum(ys)
    neg = np.cumsum(1 - ys)
    total_pos, total_neg = pos[-1], neg[-1]
    # predict 0 below the split and 1 above, and the reverse
    below = (neg + (total_pos - pos)) / len(y)
    above = (pos + (total_neg - neg)) / len(y)
    return float(max(below.max(), above.max()))


def logistic(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Held-out accuracy of a logistic regression on the surface features."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(len(y) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    w = np.zeros(Xtr.shape[1])
    for _ in range(3000):
        p = 1 / (1 + np.exp(-Xtr @ w))
        w -= 0.5 * (Xtr.T @ (p - y[tr])) / len(tr)
    pred = (Xte @ w > 0).astype(int)
    return float((pred == y[te]).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prefix", help="corpus prefix, e.g. data/gate2")
    ap.add_argument("--max-surface", type=float, default=0.60,
                    help="fail above this accuracy from surface features alone")
    ap.add_argument("--max-synth-gap", type=float, default=0.05,
                    help="fail above this per-class gap in generated-text markers")
    args = ap.parse_args()

    texts, y = load(args.prefix)
    if not texts:
        print(f"[check] no rows at {args.prefix}.*.jsonl")
        return 2
    X = np.array([featurize(t) for t in texts], dtype=float)
    base = max(y.mean(), 1 - y.mean())
    print(f"[check] {len(texts)} rows, {int(y.sum())} positive, majority baseline {base:.1%}\n")

    print("[check] SURFACE -- single feature, best threshold")
    worst_name, worst = "", 0.0
    for i, name in enumerate(FEATURES):
        acc = stump(X[:, i], y)
        flag = "  <-- SEPARATES" if acc >= args.max_surface else ""
        print(f"          {name:18s} {acc:6.1%}{flag}")
        if acc > worst:
            worst_name, worst = name, acc
    multi = logistic(X, y)
    print(f"\n[check] SURFACE -- all features, held out: {multi:.1%}")

    print("\n[check] PROVENANCE -- generated-text markers (surface CANNOT see this)")
    rates = []
    for cls in (1, 0):
        sel = [t for t, lab in zip(texts, y) if lab == cls]
        rate = sum(bool(SYNTH.search(t)) for t in sel) / max(len(sel), 1)
        rates.append(rate)
        print(f"          class {cls}: {rate:.1%} of {len(sel)}")
    gap = abs(rates[0] - rates[1])
    print(f"          gap: {gap:.1%}")

    fails = []
    if worst >= args.max_surface:
        fails.append(f"{worst_name} alone separates at {worst:.1%}")
    if multi >= args.max_surface:
        fails.append(f"surface features together reach {multi:.1%}")
    if gap > args.max_synth_gap:
        fails.append(f"generated-text markers differ by {gap:.1%} across classes")
    if fails:
        print("\n[check] FAIL -- do not train on this corpus:")
        for f in fails:
            print(f"          - {f}")
        return 1
    print(f"\n[check] PASS -- no shortcut above {args.max_surface:.0%}, provenance balanced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
