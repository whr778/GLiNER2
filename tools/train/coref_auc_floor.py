"""Document-bootstrap noise floor for the coref readout's pair AUC, paired across arms.

    uv run python tools/train/coref_auc_floor.py \
        --arms control=whr778/gliner2-corefC-control treatment=whr778/gliner2-corefC-treatment \
        --contrasts treatment.link-control.cosine treatment.link-control.link

Pairs come from coref_readout.doc_pairs, so the point estimates must equal coref_readout's. Mentions are gold,
so every arm scores the SAME pairs; each resample draws documents with replacement and scores every arm and
contrast on that one draw (paired). This is VAL-SAMPLE noise -- which documents the val set holds -- not
run-to-run (seed) noise, which needs retraining.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coref_readout import VAL, doc_pairs, roc_auc

CACHE = Path("/Volumes/Development/tmp/coref_floor")
COLS = {"cosine": 3, "link": 4}


def pairs_per_doc(model_id: str, corpus: str, limit: int, device: str) -> list[np.ndarray]:
    """[docs] arrays of (label, string_match, pronoun, cosine, link); cached per model and corpus."""
    path = CACHE / f"{model_id.replace('/', '__')}__{corpus}__{limit}.npz"
    if not path.exists():
        from gliner2 import AutoExtractor
        model = AutoExtractor.from_pretrained(model_id).to(device).eval()
        with open(VAL[corpus], encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh]
        rows = [r for r in rows if r.get("coref_clusters")][:limit]
        docs = [np.array([[np.nan if v is None else float(v) for v in p] for p in doc_pairs(model, r, None)]).reshape(-1, 5)
                for r in rows]
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(path, *docs)
    with np.load(path) as z:
        return [z[f"arr_{i}"] for i in range(len(z.files))]


def subset_auc(docs: list[np.ndarray], col: int, subset: str) -> float:
    a = np.concatenate(docs)
    a = a[a[:, 2] == 1] if subset == "pronoun" else a[a[:, 1] == 0] if subset == "different_surface" else a
    return roc_auc(a[:, 0], a[:, col])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", nargs="+", required=True, help="name=model_id")
    ap.add_argument("--contrasts", nargs="+", default=[], help="armA.col-armB.col, col in cosine|link")
    ap.add_argument("--corpus", default="wikievents", choices=list(VAL))
    ap.add_argument("--subsets", nargs="+", default=["pronoun", "all"])
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    arms = dict(a.split("=", 1) for a in args.arms)
    data = {name: pairs_per_doc(mid, args.corpus, args.limit, args.device) for name, mid in arms.items()}
    n = len(next(iter(data.values())))
    labels = [d[:, 0] for d in next(iter(data.values()))]
    assert all(np.array_equal(d[:, 0], y) for docs in data.values() for d, y in zip(docs, labels)), "arms score different pairs"
    stats = [f"{a}.{c}" for a in arms for c in COLS] + args.contrasts

    def value(stat: str, idx, subset: str) -> float:
        if "-" in stat:
            left, right = stat.split("-")
            return value(left, idx, subset) - value(right, idx, subset)
        arm, col = stat.split(".")
        return subset_auc([data[arm][i] for i in idx], COLS[col], subset)

    rng = np.random.default_rng(args.seed)
    draws = [rng.integers(0, n, n) for _ in range(args.resamples)]
    out = {"corpus": args.corpus, "docs": n, "resamples": args.resamples, "arms": arms, "results": {}}
    for subset in args.subsets:
        res = {}
        for stat in stats:
            point = value(stat, range(n), subset)
            if np.isnan(point):
                continue
            boot = np.array([value(stat, idx, subset) for idx in draws])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            res[stat] = {"point": round(point, 4), "sd": round(float(boot.std()), 4),
                         "ci95": [round(float(lo), 4), round(float(hi), 4)]}
            print(f"{subset:10s} {stat:40s} {point:.4f}  sd {boot.std():.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]", flush=True)
        out["results"][subset] = res
    if args.out:
        args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
