"""Did the gate learn TURKISH, or did it learn TRT Haber's register?

The Turkish training data is one outlet in one year. That makes a good score on TRT
uninterpretable on its own -- register is a confound this project has been burned by
before. This scores nine outlets the training corpus never saw (state agency, opposition,
independents, foreign broadcasters' Turkish services; 2016-2023 against training's 2024),
with ADJUDICATED four-way labels rather than the regex heuristics used by
gate_turkish_fp.py.

That difference matters for reading the numbers: the 0.4733 baseline on record is AUC
against heuristic labels on TRT text. It is NOT comparable to anything here. Both arms are
therefore scored on this set, so the comparison is like-for-like.

Positives are `current_toll`; `no_toll`, `exposure_only` and `historical_toll` are all
negatives, which is exactly the gate's binary question. AUC is threshold-free on purpose --
a single operating point cannot separate discrimination from a gate that has stopped
firing.

    uv run python tools/ekf_showcase/gate_turkish_heldout.py \
        --models whr778/gliner2-gate2-mmbert-v2 whr778/gliner2-gate2-mmbert-tr
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline import build_gate_schema  # noqa: E402

HELDOUT = "data/turkish_gate/gate_ann_tr_heldout.jsonl"


def auc(pos: list[float], neg: list[float]) -> float:
    """P(a random positive outscores a random negative); ties count half."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def score(model_id: str, rows: list[dict], device: str, every: int = 250) -> list[float]:
    """Gate scores for every row. Progress goes to stderr so stdout stays parseable.

    A full pass is thousands of documents and several minutes with no output otherwise,
    which is indistinguishable from a hang.
    """
    import time

    from gliner2 import AutoExtractor
    model = AutoExtractor.from_pretrained(model_id, map_location=device)
    schema = build_gate_schema(model)
    out = []
    start = time.time()
    for i, row in enumerate(rows, 1):
        r = model.extract(row["input"], schema, include_confidence=True).get("relevance")
        label = r.get("label") if isinstance(r, dict) else r
        conf = float(r.get("confidence", 1.0)) if isinstance(r, dict) else 1.0
        out.append(conf if label == "mass_casualty" else 1.0 - conf)
        if every and (i % every == 0 or i == len(rows)):
            rate = i / (time.time() - start)
            eta = (len(rows) - i) / rate
            print(f"    {i}/{len(rows)} ({i / len(rows):.0%})  {rate:.1f} doc/s  "
                  f"eta {eta / 60:.1f} min", file=sys.stderr, flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--heldout", default=HELDOUT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--progress-every", type=int, default=250,
                    help="documents between progress lines on stderr; 0 to silence")
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.heldout).open(encoding="utf-8")]
    gold = [r["label"] == "current_toll" for r in rows]
    print(f"{len(rows)} held-out documents, {sum(gold)} positive "
          f"({sum(gold)/len(rows):.1%}), {len({r['source'] for r in rows})} outlets\n")

    for model_id in a.models:
        print(f"{model_id} ...", file=sys.stderr, flush=True)
        s = score(model_id, rows, a.device, a.progress_every)
        overall = auc([x for x, g in zip(s, gold) if g], [x for x, g in zip(s, gold) if not g])
        print(f"{model_id}")
        print(f"    OVERALL AUC {overall:.4f}   (0.5 = cannot separate them at any threshold)")
        per = defaultdict(lambda: ([], []))
        for x, g, r in zip(s, gold, rows):
            per[r["source"]][0 if g else 1].append(x)
        print(f"    {'outlet':22s}{'n':>5s}{'pos':>5s}{'AUC':>9s}")
        for src, (p, n) in sorted(per.items(), key=lambda kv: -(len(kv[1][0]) + len(kv[1][1]))):
            print(f"      {src:20s}{len(p)+len(n):>5d}{len(p):>5d}{auc(p, n):>9.4f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
