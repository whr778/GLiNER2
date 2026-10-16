"""Pick the entity-coref cut-off on VAL, then score the blind test ONCE at it.

    uv run python tools/train/coref_threshold.py --model whr778/gliner2-corefC-treatment --split val
    uv run python tools/train/coref_threshold.py --model whr778/gliner2-corefC-treatment --split test --threshold 0.1

Mentions are the GOLD mentions of the *_coref_llm eval files (LLM silver, 3 draws voted), so this measures
linking, not mention detection. Clusters come from gliner2.inference.entity_coref.best_antecedent, the
inference path. Scored as pair P/R/F1 over every placed mention pair (singletons count as negatives), and
over pairs involving a pronoun. Link logits are cached per model and file, so a sweep re-reads them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from gliner2.inference.entity_coref import PRONOUNS, best_antecedent, link_logits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import _fetch_if_missing

CACHE = Path(os.environ.get("COREF_CACHE", "/Volumes/Development/tmp/coref_threshold"))
CORPORA = ["cc_news_events_sonnet55_v2_coref_llm", "cc_news_events_haiku45_coref_llm", "cc_news_haiku45_coref_llm"]


def doc_logits(model_id: str, path: str, limit: int, device: str) -> list:
    """Per doc (logits [P, P], gold cluster per placed mention, casefolded text per placed mention); cached."""
    cache = CACHE / f"{model_id.replace('/', '__')}__{Path(path).stem}__{limit}.pt"
    if cache.exists():
        return torch.load(cache)
    from gliner2 import AutoExtractor
    model = AutoExtractor.from_pretrained(model_id).to(device).eval()
    _fetch_if_missing(path)
    out = []
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh][:limit]
    for row in rows:
        ms = sorted(({**m, "c": ci} for ci, c in enumerate(row["coref_clusters"]) for m in c), key=lambda m: m["start"])
        logits, idx = link_logits(model, row["input"], ms)
        out.append((logits, [ms[i]["c"] for i in idx], [ms[i]["text"].casefold() for i in idx]))
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    return out


def string_match(texts: list) -> np.ndarray:
    """Baseline clusters: mentions with the same casefolded text are one entity."""
    first = {}
    return np.array([first.setdefault(t, i) for i, t in enumerate(texts)])


def pair_prf(docs: list, threshold: float | None) -> dict:
    """Pair precision/recall/F1 of best-antecedent clusters vs gold, all pairs and pronoun pairs.
    ``threshold`` None scores the string-match baseline instead."""
    tp = {"all": 0, "pronoun": 0}
    npred = dict(tp)
    ngold = dict(tp)
    for logits, gold, texts in docs:
        if threshold is None:
            pred = string_match(texts)
        else:
            pred = np.array(best_antecedent(logits, threshold)) if len(gold) > 1 else np.arange(len(gold))
        g = np.array(gold)
        p = np.array([t in PRONOUNS for t in texts], dtype=bool)
        upper = np.triu(np.ones((len(g), len(g)), dtype=bool), 1)
        same_p, same_g = (pred[:, None] == pred[None, :]) & upper, (g[:, None] == g[None, :]) & upper
        for k, mask in (("all", upper), ("pronoun", upper & (p[:, None] | p[None, :]))):
            tp[k] += int((same_p & same_g & mask).sum())
            npred[k] += int((same_p & mask).sum())
            ngold[k] += int((same_g & mask).sum())
    res = {}
    for k, hits in tp.items():
        pr, rc = hits / max(npred[k], 1), hits / max(ngold[k], 1)
        res[k] = {"precision": round(pr, 4), "recall": round(rc, 4), "f1": round(2 * pr * rc / max(pr + rc, 1e-9), 4)}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--corpora", nargs="+", default=CORPORA)
    ap.add_argument("--threshold", type=float, help="test: the cut-off picked on val (required for test)")
    ap.add_argument("--limit", type=int, default=100000)
    # CPU, not MPS: MPS keeps per-input-shape state, so every new document length costs ~100 MB that is
    # never returned (proven: one doc x30 flat, 30 distinct docs +2.9 GB driver; the full val run was
    # SIGKILLed). CPU is flat after warm-up and about as fast for this encode-only job.
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.split == "test" and args.threshold is None:
        raise SystemExit("the blind test is scored once, at the cut-off picked on val: pass --threshold")

    per = {c: doc_logits(args.model, f"data/{c}.{args.split}.jsonl", args.limit, args.device) for c in args.corpora}
    pooled = [d for docs in per.values() for d in docs]
    grid = [args.threshold] if args.split == "test" else [round(x, 2) for x in np.arange(-1.0, 2.01, 0.1)]
    result = {"model": args.model, "split": args.split, "docs": {c: len(d) for c, d in per.items()}, "sweep": {},
              "string_match": {"pooled": pair_prf(pooled, None), **{c: pair_prf(d, None) for c, d in per.items()}}}
    r = result["string_match"]["pooled"]
    print(f"string-match baseline all P {r['all']['precision']:.3f} R {r['all']['recall']:.3f} F1 {r['all']['f1']:.3f}"
          f"  | pronoun P {r['pronoun']['precision']:.3f} R {r['pronoun']['recall']:.3f} F1 {r['pronoun']['f1']:.3f}")
    for th in grid:
        result["sweep"][th] = {"pooled": pair_prf(pooled, th), **{c: pair_prf(d, th) for c, d in per.items()}}
        r = result["sweep"][th]["pooled"]
        print(f"threshold {th:+.2f}  all P {r['all']['precision']:.3f} R {r['all']['recall']:.3f} F1 {r['all']['f1']:.3f}"
              f"  | pronoun P {r['pronoun']['precision']:.3f} R {r['pronoun']['recall']:.3f} F1 {r['pronoun']['f1']:.3f}", flush=True)
    if args.split == "val":
        best = max(grid, key=lambda th: result["sweep"][th]["pooled"]["all"]["f1"])
        result["picked"] = best
        print(f"PICKED on val (max pooled all-pairs F1): {best:+.2f}")
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
