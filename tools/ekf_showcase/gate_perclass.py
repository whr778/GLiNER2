"""Per-class gate accuracy, scored PAIRED so the difference between two gates can be tested.

Two instrument failures this exists to prevent, both of which happened:

1. **A corpus-weighted metric hid the verdict.** gate2 v2 scored `relevance` 0.8341
   against v1's 0.8368 on the blind test and was written off as no improvement. On rows
   stratified toward the hard classes it wins 37 rows to 17, exact McNemar p = 0.0091.
   The aggregate was flat because easy `current_toll` rows dominate it.

2. **A 16-row cell produced a baseline that was mostly noise.** `exposure_only` was
   recorded at 0.250 from a random sample; on all 72 rows in the split it is 0.431. Part
   of the gap being chased did not exist. So this takes EVERY row of the scarce classes
   and caps only the plentiful ones.

Both models see identical rows, which is what licenses McNemar: it looks only at rows
where exactly one model is right, so per-row difficulty cancels. Per-class p-values are
EXPLORATORY -- five comparisons, uncorrected. The ALL row is the one test that stands
alone.

The four-way label comes from the Haiku adjudication in data/gate_ann.jsonl, joined to
the test split on the text itself.

    uv run python tools/ekf_showcase/gate_perclass.py \
        --models whr778/gliner2-gate2-mmbert-real whr778/gliner2-gate2-mmbert-v2

Device: cpu is the default because the recorded numbers were measured there. mps is
~28% faster warm on this workload (188 vs 265 ms/row on mmBERT), after a ~380 ms/row
first round of Metal warm-up; predictions have not been checked for device parity, so
do not mix devices within one comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline import build_gate_schema                     # noqa: E402

# Cap the classes that are plentiful; take ALL of the ones that are not. The uncapped
# classes are the open EKF failures -- "in 1999 ... deaths of over 17,000" (historical)
# and "220,000 victims have been served meals" (exposure).
CAP = {"current_toll": 150, "no_toll": 120, "(duee/filler)": 40}
FILLER = "(duee/filler)"


def stratified_rows(test_path, ann_path, seed):
    """Test rows tagged with their four-way label, hard classes complete."""
    fourway = {}
    for line in Path(ann_path).open(encoding="utf-8"):
        record = json.loads(line)
        fourway[record["input"]] = record["label"]

    buckets = defaultdict(list)
    for line in Path(test_path).open(encoding="utf-8"):
        record = json.loads(line)
        buckets[fourway.get(record["input"], FILLER)].append(record)

    rng = random.Random(seed)
    rows = []
    for label, records in buckets.items():
        rng.shuffle(records)
        for record in records[:CAP.get(label, len(records))]:
            classification = next(c for c in record["output"]["classifications"]
                                  if c["task"] == "relevance")
            rows.append({"text": record["input"], "label": label,
                         "gold": classification["true_label"][0] == classification["labels"][0]})
    return rows


def score(model_id, rows, device):
    """Store P(mass_casualty) on every row, in place.

    The score, not the decision, because the decision is what gets swept. `relevance` is a
    single-label task, so runtime.py takes a SOFTMAX over the two labels and reports the
    argmax with its probability -- which makes 1 - confidence the other label's probability
    exactly, not an approximation. At threshold 0.5 this reproduces the argmax decision.
    """
    from gliner2 import AutoExtractor
    model = AutoExtractor.from_pretrained(model_id, map_location=device)
    schema = build_gate_schema(model)
    start = time.time()
    for row in rows:
        relevance = model.extract(row["text"], schema, include_confidence=True).get("relevance")
        label = relevance.get("label") if isinstance(relevance, dict) else relevance
        confidence = float(relevance.get("confidence", 1.0)) if isinstance(relevance, dict) else 1.0
        row[model_id] = confidence if label == "mass_casualty" else 1.0 - confidence
    return (time.time() - start) / len(rows) * 1000


def admits(row, model, threshold):
    return row[model] >= threshold


def mcnemar(rows, first, second, threshold):
    """Exact two-sided McNemar: discordant pairs are binomial(n, 0.5) under the null.

    Exact rather than the normal approximation because the cells are small -- with 3
    discordant pairs each way the continuity-corrected z goes negative and reports
    p = 1.32, which is not a probability.
    """
    only_first = sum(1 for r in rows if admits(r, first, threshold) == r["gold"]
                     and admits(r, second, threshold) != r["gold"])
    only_second = sum(1 for r in rows if admits(r, second, threshold) == r["gold"]
                      and admits(r, first, threshold) != r["gold"])
    n = only_first + only_second
    if n == 0:
        return only_first, only_second, 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_first, only_second) + 1))
    return only_first, only_second, min(1.0, 2 * tail / 2 ** n)


RECALL_TARGETS = (0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50)


def threshold_for_recall(positives, model, target):
    """Lowest threshold that still admits `target` of the true tolls."""
    scores = sorted((r[model] for r in positives), reverse=True)
    return scores[min(len(scores), max(1, math.ceil(target * len(scores)))) - 1]


def sweep(rows, models):
    """Compare gates at MATCHED recall, because argmax puts them at different operating points.

    Reading raw argmax numbers across models compares two things at once -- how well a gate
    separates the classes, and where its decision happens to sit. v2 admits 282 of 541 rows
    and casualty-docee 116, so v2 looking better on true tolls and worse on exposure text is
    what any pair of thresholds on one curve would produce. Fixing recall isolates the part
    that is a property of the model.

    Every negative class here has gold `other` throughout, so its column is a false-positive
    rate: lower is better, and 0.000 would be a gate that admits none of it.
    """
    positives = [r for r in rows if r["gold"]]
    negatives = {}
    for label in {r["label"] for r in rows}:
        subset = [r for r in rows if r["label"] == label]
        if not any(r["gold"] for r in subset):
            negatives[label] = subset
    order = sorted(negatives, key=lambda k: -len(negatives[k]))

    print(f"\n  Matched-recall sweep. Recall is on the {len(positives)} true tolls; "
          f"the rest are false-positive rates (lower is better).")
    for tag, model in zip("ABCDEFGH", models):
        print(f"\n  {tag}: {model}")
        print(f"    {'recall':>7s}{'thresh':>8s}" + "".join(f"{k[:15]:>17s}" for k in order)
              + f"{'overall acc':>13s}")
        for target in RECALL_TARGETS:
            t = threshold_for_recall(positives, model, target)
            recall = sum(1 for r in positives if admits(r, model, t)) / len(positives)
            fps = "".join(f"{sum(1 for r in negatives[k] if admits(r, model, t)) / len(negatives[k]):>17.3f}"
                          for k in order)
            accuracy = sum(1 for r in rows if admits(r, model, t) == r["gold"]) / len(rows)
            print(f"    {recall:>7.3f}{t:>8.3f}{fps}{accuracy:>13.3f}")


def report(rows, models, threshold):
    accuracy = lambda rs, m: sum(1 for r in rs if admits(r, m, threshold) == r["gold"]) / len(rs)
    labels = sorted({r["label"] for r in rows},
                    key=lambda k: -sum(1 for r in rows if r["label"] == k))
    first, second = models[0], models[-1]
    paired = len(models) == 2

    # Tag the columns A/B rather than printing model names: the ids under comparison
    # differ in their LAST characters (…-real vs …-v2), so a truncated name column shows
    # two headers that look identical.
    tags = [chr(ord("A") + i) for i in range(len(models))]
    head = f"\n  {'label':22s}" + "".join(f"{t:>10s}" for t in tags)
    print(head + (f"{'B - A':>9s}{'only A':>8s}{'only B':>8s}{'p exact':>9s}" if paired else "")
          + f"{'n':>7s}")
    for label in labels + ["ALL"]:
        subset = rows if label == "ALL" else [r for r in rows if r["label"] == label]
        line = f"  {label:22s}" + "".join(f"{accuracy(subset, m):>10.3f}" for m in models)
        if paired:
            a, b, p = mcnemar(subset, first, second, threshold)
            line += (f"{accuracy(subset, second) - accuracy(subset, first):>+9.3f}"
                     f"{a:>8d}{b:>8d}{p:>9.4f}")
        print(line + f"{len(subset):>7d}")
    if paired:
        print("\n  Per-class p-values are exploratory (five uncorrected comparisons). "
              "The ALL row is the test.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="one gate to profile, or two to compare paired")
    ap.add_argument("--test", default="data/gate2.test.jsonl")
    ap.add_argument("--annotations", default="data/gate_ann.jsonl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="admit when P(mass_casualty) >= this (0.5 reproduces argmax)")
    ap.add_argument("--sweep", action="store_true",
                    help="also compare the models at matched recall")
    ap.add_argument("--out", type=Path, help="write per-row scores here")
    a = ap.parse_args()

    rows = stratified_rows(a.test, a.annotations, a.seed)
    composition = {k: sum(1 for r in rows if r["label"] == k) for k in {r["label"] for r in rows}}
    print(f"evaluation set: {composition}", flush=True)
    for tag, model_id in zip("ABCDEFGH", a.models):
        print(f"  {tag}: {model_id} at {score(model_id, rows, a.device):.0f} ms/row", flush=True)
    report(rows, a.models, a.threshold)
    if a.sweep:
        sweep(rows, a.models)
    if a.out:
        a.out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(f"\n  per-row scores -> {a.out}")


if __name__ == "__main__":
    main()
