"""What purity can the gate deliver on the annotation pool, and what does that cost?

Buying field-level annotation is priced per DOCUMENT but valued per POSITIVE, so the
gate's precision at a cut point IS the price. This measures that curve on the adjudicated
held-out set, which is drawn from the same pool and the same outlets, so its label
distribution is the pool's.

Two things this exists to stop:
  - Pricing off the wrong base rate. The first estimate used 42.3%, measured on the TRT
    pilot, a DIFFERENT source. The multi-outlet pool is 25.1% positive, which nearly
    doubles the unfiltered cost of a fixed number of positives.
  - Assuming a threshold is a lever. On English this gate's confidence saturates, so a
    higher cut may buy no purity at all. If the purity column is flat, thresholding is
    inert and the honest move is to pay the unfiltered price or buy fewer positives.

Recall matters as much as purity: a cut that admits 90% positives while discarding most
of the positives in the pool needs a far larger pool to hit the same target, and the pool
is finite.

    uv run python tools/data/gate_purity_curve.py --model whr778/gliner2-gate2-mmbert-tr
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ekf_showcase"))
from run_pipeline import build_gate_schema  # noqa: E402

HELDOUT = "data/turkish_gate/gate_ann_tr_heldout_full.jsonl"

# A FREE coarse filter, run before the gate ever loads. Turkish death/injury words within
# 60 characters of a numeral, allowing Turkish scale words so "29 bin 313" counts. On the
# adjudicated set this alone reaches 40.8% purity at 83.7% recall -- better purity than
# the gate at cut 0.9 -- and it halves what the gate has to score, which is the expensive
# part. The two compose: coarse regex, then the model on what survives.
_DEATH = (r"(?:öl[düü]|ölü|ölüm|can kayb|hayat[ıi]n[ıi] kaybet|hayat[ıi]n[ıi] yitir|"
          r"yaşam[ıi]n[ıi] yitir|yaral[ıi]|yaraland|kay[ıi]p|cenaze|cesed)")
_NUM = r"\d[\d.,]*(?:\s*(?:bin|milyon))?"
TOLL_NEAR = re.compile(_NUM + r".{0,60}?" + _DEATH + r"|" + _DEATH + r".{0,60}?" + _NUM,
                       re.I | re.S)
CUTS = (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999)
INPUT_TOKENS, OUTPUT_TOKENS = 1023 + 350, 260
IN_RATE, OUT_RATE = 0.50, 2.50  # Haiku 4.5 batch, per million


def batch_cost(n_docs: float) -> float:
    return n_docs * INPUT_TOKENS / 1e6 * IN_RATE + n_docs * OUTPUT_TOKENS / 1e6 * OUT_RATE


def gate_scores(model_id: str, rows: list[dict], device: str, cache: Path) -> list[float]:
    """Confidence that each row is a current toll. Cached -- a pass is thousands of docs."""
    if cache.is_file():
        scores = json.loads(cache.read_text(encoding="utf-8"))
        if len(scores) == len(rows):
            print(f"[purity] reusing cached scores from {cache}")
            return scores
    from gliner2 import AutoExtractor
    model = AutoExtractor.from_pretrained(model_id, map_location=device)
    schema = build_gate_schema(model)
    out = []
    for i, row in enumerate(rows, 1):
        r = model.extract(row["input"], schema, include_confidence=True).get("relevance")
        label = r.get("label") if isinstance(r, dict) else r
        conf = float(r.get("confidence", 1.0)) if isinstance(r, dict) else 1.0
        out.append(conf if label == "mass_casualty" else 1.0 - conf)
        if i % 250 == 0 or i == len(rows):
            print(f"    {i}/{len(rows)}", file=sys.stderr, flush=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="whr778/gliner2-gate2-mmbert-tr")
    ap.add_argument("--heldout", default=HELDOUT)
    ap.add_argument("--pool", default="/Volumes/Development/data/turkish_pool.jsonl")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--target", type=int, default=30000, help="positives wanted")
    ap.add_argument("--cache", default="out/gate_purity_scores.json")
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.heldout).open(encoding="utf-8")]
    gold = [r["label"] == "current_toll" for r in rows]
    pool_n = sum(1 for _ in Path(a.pool).open(encoding="utf-8")) if Path(a.pool).is_file() else 0
    print(f"{len(rows)} adjudicated docs, {sum(gold)} positive ({sum(gold)/len(rows):.1%}) "
          f"-- the pool's own rate\npool: {pool_n} documents\n")

    cache = Path(a.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    scores = gate_scores(a.model, rows, a.device, cache)

    prefilter = [bool(TOLL_NEAR.search(r["input"])) for r in rows]
    pre_keep = sum(prefilter) / len(rows)
    pre_pur = sum(g for pre, g in zip(prefilter, gold) if pre) / max(sum(prefilter), 1)
    print(f"regex prefilter alone: keeps {pre_keep:.1%} of the pool, purity {pre_pur:.1%}, "
          f"recall {sum(g for pre, g in zip(prefilter, gold) if pre) / sum(gold):.1%} -- free\n")

    for label, mask in (("gate only (scores the FULL pool)", [True] * len(rows)),
                        ("regex THEN gate (scores HALF the pool)", prefilter)):
        print(label)
        print(f"{'cut':>9s}{'admit%':>9s}{'purity':>9s}{'recall':>9s}"
              f"{'pool docs':>11s}{'docs to buy':>13s}{'cost':>10s}")
        _curve(scores, gold, mask, pool_n, a.target)
        print()
    print("Read purity beside recall: a high cut that discards most positives needs a "
          "bigger pool,\nand the pool is finite. 'POOL TOO SMALL' means the target is "
          "unreachable at that cut.")
    return 0


def _curve(scores, gold, mask, pool_n, target):
    for cut in CUTS:
        kept = [(s, g) for s, g, m in zip(scores, gold, mask) if m and s >= cut]
        if not kept:
            print(f"{cut:>9.5f}{'-':>9s}{'admits nothing':>9s}")
            continue
        admit = len(kept) / len(scores)
        purity = sum(g for _, g in kept) / len(kept)
        recall = sum(g for _, g in kept) / sum(gold)
        # Positives actually REACHABLE in the pool at this cut, which caps the buy.
        reachable = pool_n * admit * purity
        need = target / purity if purity else float("inf")
        feasible = reachable >= target
        note = f"${batch_cost(need):,.2f}" if feasible else "POOL TOO SMALL"
        print(f"{cut:>9.5f}{admit:>8.1%}{purity:>9.1%}{recall:>9.1%}"
              f"{pool_n * admit:>11,.0f}{need:>13,.0f}{note:>10s}")



if __name__ == "__main__":
    raise SystemExit(main())
