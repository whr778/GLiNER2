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
    """Add this model's boolean prediction to every row, in place."""
    from gliner2 import AutoExtractor
    model = AutoExtractor.from_pretrained(model_id, map_location=device)
    schema = build_gate_schema(model)
    start = time.time()
    for row in rows:
        relevance = model.extract(row["text"], schema, include_confidence=True).get("relevance")
        label = relevance.get("label") if isinstance(relevance, dict) else relevance
        row[model_id] = label == "mass_casualty"
    return (time.time() - start) / len(rows) * 1000


def mcnemar(rows, first, second):
    """Exact two-sided McNemar: discordant pairs are binomial(n, 0.5) under the null.

    Exact rather than the normal approximation because the cells are small -- with 3
    discordant pairs each way the continuity-corrected z goes negative and reports
    p = 1.32, which is not a probability.
    """
    only_first = sum(1 for r in rows if r[first] == r["gold"] and r[second] != r["gold"])
    only_second = sum(1 for r in rows if r[second] == r["gold"] and r[first] != r["gold"])
    n = only_first + only_second
    if n == 0:
        return only_first, only_second, 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_first, only_second) + 1))
    return only_first, only_second, min(1.0, 2 * tail / 2 ** n)


def report(rows, models):
    accuracy = lambda rs, m: sum(1 for r in rs if r[m] == r["gold"]) / len(rs)
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
            a, b, p = mcnemar(subset, first, second)
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
    ap.add_argument("--out", type=Path, help="write per-row predictions here")
    a = ap.parse_args()

    rows = stratified_rows(a.test, a.annotations, a.seed)
    composition = {k: sum(1 for r in rows if r["label"] == k) for k in {r["label"] for r in rows}}
    print(f"evaluation set: {composition}", flush=True)
    for tag, model_id in zip("ABCDEFGH", a.models):
        print(f"  {tag}: {model_id} at {score(model_id, rows, a.device):.0f} ms/row", flush=True)
    report(rows, a.models)
    if a.out:
        a.out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(f"\n  per-row predictions -> {a.out}")


if __name__ == "__main__":
    main()
