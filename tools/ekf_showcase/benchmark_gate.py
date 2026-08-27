"""Benchmark the relevance gate on REAL disaster messages.

The showcase gate is currently scored against distractors written by hand for the demo
feed -- a self-graded exam. `community-datasets/disaster_response_messages` gives 21,046
expert-annotated real messages (Haiti-era; direct/news/social; en/es/fr/ht/ur), which is
an independent test.

**What can and cannot be measured here.** No label in that dataset matches the gate's
question ("is this a mass-casualty REPORT?"):

* ``related=1`` means *disaster-related*, which is broader -- "Delmas 33 in Silo, need
  water" is disaster-related and carries no casualties, so the gate SHOULD reject it.
  Scoring recall against ``related`` would penalise correct behaviour.
* ``death=1`` means the message *mentions* death, which is also broader -- "the kids are
  starving to death" mentions death and reports no toll.

So the headline number here is the one that IS unambiguous: the **false-positive rate on
``related=0``**. Those messages ("I thank you for the good work you are doing", "please
add some minutes on my phone") are definitively not disaster reports, and any gate that
admits them is wrong with no room for interpretation. Recall against ``death |
missing_people`` is reported alongside as indicative, explicitly not as ground truth.

    uv run python tools/ekf_showcase/benchmark_gate.py --limit 600
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "ekf_showcase"))

DATASET = "community-datasets/disaster_response_messages"


def load(split: str, limit: int, seed: int):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, f"data/{split}-00000-of-00001.parquet",
                           repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    df = pd.read_parquet(path)
    df = df[df["related"].isin([0, 1])]           # 2 = untranslatable/ambiguous, drop
    if limit and len(df) > limit:
        # Stratified: keep every negative we can afford, since negatives are the
        # unambiguous half of this benchmark and are only 16% of the split.
        neg = df[df.related == 0]
        pos = df[df.related == 1]
        n_neg = min(len(neg), limit // 2)
        neg = neg.sample(n=n_neg, random_state=seed)
        pos = pos.sample(n=min(len(pos), limit - n_neg), random_state=seed)
        df = pd.concat([neg, pos]).sample(frac=1.0, random_state=seed)
    return df.reset_index(drop=True)


SWEEP_THRESHOLDS = (0.5, 0.9, 0.95, 0.99, 0.998, 0.9999)


def sweep(decisions, df) -> None:
    """The whole operating curve from ONE inference pass.

    Every external number this project has recorded for the gate was taken at threshold
    0.5, and on the gate's own test split moving to 0.998 changed accuracy more than any
    training intervention did. A single point cannot distinguish a gate that discriminates
    from one that has simply stopped firing, so the FP column is only interpretable beside
    the recall column.

    `relevance` is single-label, so runtime.py softmaxes over the two labels and
    1 - confidence is P(mass_casualty) exactly on messages answered `other`.
    """
    # Stage 0 rejects unsupported languages BEFORE the model runs, reporting
    # relevance="unsupported_language" at confidence 0.0. Those rows carry no model
    # score, and `1 - confidence` turns each one into a 1.0 -- a maximum-confidence false
    # positive that never happened. This corpus is multilingual (en/es/fr/ht/ur), so that
    # bug put an identical ~13-message floor under every model's FP count and made the
    # curves look model-independent. Drop them and say how many.
    dropped = sum(1 for d in decisions if d["relevance"] == "unsupported_language")
    usable = [d["relevance"] != "unsupported_language" for d in decisions]
    if dropped:
        print(f"\n  {dropped}/{len(decisions)} messages dropped by the language gate and "
              f"excluded below\n  (they never reached the model, so they are neither its "
              f"hits nor its false positives)")

    scores = [d["relevance_confidence"] if d["relevance"] == "mass_casualty"
              else 1.0 - d["relevance_confidence"] for d in decisions]
    neg = [s for s, n, u in zip(scores, df["related"] == 0, usable) if n and u]
    cas = ((df.get("death", 0) == 1) | (df.get("missing_people", 0) == 1))
    pos = [s for s, c, u in zip(scores, cas, usable) if c and u]

    print(f"\n  {'threshold':>10s}{'FP related=0':>15s}{'recall death|missing':>23s}")
    for t in SWEEP_THRESHOLDS:
        fp = sum(1 for x in neg if x >= t)
        hit = sum(1 for x in pos if x >= t)
        print(f"  {t:>10.4f}{fp:>9d}/{len(neg):<5d}{hit:>16d}/{len(pos):<6d}")

    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    auc = wins / max(len(pos) * len(neg), 1)
    print(f"    AUC = {auc:.4f}   (related=0 vs death|missing; the positive label is"
          f" INDICATIVE,\n    so read this as a comparison between models, not as an"
          f" absolute)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=600, help="messages to score (0 = all)")
    ap.add_argument("--gate-model", default="fastino/gliner2-base-v1")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sweep", action="store_true",
                    help="report the whole operating curve instead of one threshold")
    args = ap.parse_args()

    from gliner2 import AutoExtractor
    from run_pipeline import gate

    df = load(args.split, args.limit, args.seed)
    texts = df["message"].astype(str).tolist()
    print(f"[gate-bench] {len(texts)} messages from {DATASET} ({args.split})")
    print(f"[gate-bench] model {args.gate_model}, threshold {args.threshold}\n")

    model = AutoExtractor.from_pretrained(args.gate_model, map_location=args.device)
    # with_type=False: the sweep needs only the relevance pass, and disaster_type costs a
    # second encoder pass per message.
    decisions = gate(model, texts, args.threshold, with_type=not args.sweep)
    if args.sweep:
        sweep(decisions, df)
        return
    kept = [d["relevant"] for d in decisions]

    # 1. THE headline: false positives on definitively non-disaster messages.
    neg = df["related"] == 0
    fp = sum(1 for k, n in zip(kept, neg) if n and k)
    n_neg = int(neg.sum())
    print(f"  false-positive rate on related=0   {fp}/{n_neg} = {fp / max(n_neg, 1):.3f}")
    print("     (unambiguous: these messages are not disaster reports at all)")

    # 2. Indicative only -- the label is broader than the gate's question.
    cas = ((df.get("death", 0) == 1) | (df.get("missing_people", 0) == 1))
    n_cas = int(cas.sum())
    hit = sum(1 for k, c in zip(kept, cas) if c and k)
    print(f"\n  recall on death|missing_people     {hit}/{n_cas} = {hit / max(n_cas, 1):.3f}")
    print("     (INDICATIVE: the label means 'mentions death', not 'reports a toll')")

    rel = df["related"] == 1
    kept_rel = sum(1 for k, r in zip(kept, rel) if r and k)
    print(f"\n  kept among related=1               {kept_rel}/{int(rel.sum())} = "
          f"{kept_rel / max(int(rel.sum()), 1):.3f}")
    print("     (expected to be WELL below 1.0: most disaster messages are aid requests,")
    print("      not casualty reports, and the gate is right to drop them)")

    by_genre: dict = defaultdict(lambda: [0, 0])
    for k, g in zip(kept, df["genre"].astype(str)):
        by_genre[g][1] += 1
        by_genre[g][0] += int(k)
    print("\n  kept by genre:")
    for g, (k, n) in sorted(by_genre.items()):
        print(f"     {g:8} {k:4}/{n:<4} = {k / max(n, 1):.3f}")


if __name__ == "__main__":
    main()
