"""Entity-coreference readout: pair AUC of a checkpoint's mention link on VAL docs, beside string match and cosine.

    uv run python tools/train/coref_readout.py --model whr778/gliner2-eb21 --out readout.json

Mentions are represented exactly as training sees them (the inference collator over the record's own
schema, `_encode_core`, mean word states -- gliner2/models/boundary/mention_link.py). Every mention pair
inside a document is scored; positive = same gold `coref_clusters` cluster. Per corpus it reports AUC
over all pairs, pairs whose surfaces differ, and pairs involving an English pronoun. `link` is absent
when the checkpoint has no trained link (eb21, a control arm): the cosine column is then the baseline.
VAL only; the test splits stay blind.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gliner2 import AutoExtractor
from gliner2.models.boundary.mention_link import mention_states
from gliner2.processing import coref

PRONOUNS = {"he", "she", "they", "him", "her", "them", "his", "their", "hers", "theirs", "its", "it", "we",
            "us", "our", "i", "me", "my", "you", "your", "who", "whom", "whose", "himself", "herself",
            "themselves", "itself"}
VAL = {"wikievents": "data/wikievents.dev.jsonl", "gum_coref": "data/gum_coref.val.jsonl",
       "litbank_coref": "data/litbank_coref.val.jsonl", "gap_coref": "data/gap_coref.val.jsonl",
       "cmnee": "data/cmnee.val.jsonl"}


def doc_pairs(model, rec, max_docs_words: int):
    """[(label, string_match, pronoun, cosine, link)] for every mention pair of one record."""
    batch = model.processor.collate_fn_inference(
        [(rec["input"], coref.attach(rec["output"], rec["coref_clusters"]))], architecture="boundary",
        max_len=max_docs_words, event_records=True, ignore_missing_entities=True)
    ms = batch.coref_mentions[0] if batch.coref_mentions else []
    if len(ms) < 2:
        return []
    with torch.no_grad():
        core = model._encode_core(batch)
        states, mid = mention_states(core["text_states"][0], ms)
        z = torch.nn.functional.normalize(states.float(), dim=-1)
        cos = (z @ z.t()).cpu()
        link = model.mention_link(states, mid).float().cpu() if model.mention_link is not None else None
    words = batch.text_tokens[0]
    surf = [" ".join(words[s:e + 1]).casefold() for s, e, _ in ms]
    out = []
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            out.append((float(ms[i][2] == ms[j][2]), surf[i] == surf[j], surf[i] in PRONOUNS or surf[j] in PRONOUNS,
                        float(cos[i, j]), None if link is None else float(link[i, j])))
    return out


def roc_auc(y, s) -> float:
    """ROC AUC as the Mann-Whitney statistic, tied scores at their average rank."""
    y, s = np.asarray(y, dtype=bool), np.asarray(s, dtype=float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_s)) + 1]
    ends = np.r_[starts[1:], len(s)]
    for a, b in zip(starts, ends):
        ranks[order[a:b]] = (a + b + 1) / 2
    n_pos, n_neg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auc(pairs, col):
    y = [p[0] for p in pairs]
    return round(roc_auc(y, [p[col] for p in pairs]), 4) if len(set(y)) == 2 else None


def readout(model, corpora, limit: int, max_words: int):
    res = {}
    for name in corpora:
        rows = [json.loads(l) for l in open(VAL[name], encoding="utf-8")]
        rows = [r for r in rows if r.get("coref_clusters")][:limit]
        pairs = [p for r in rows for p in doc_pairs(model, r, max_words)]
        subsets = {"all": pairs, "different_surface": [p for p in pairs if not p[1]],
                   "pronoun": [p for p in pairs if p[2]]}
        res[name] = {"docs": len(rows), **{k: {"pairs": len(v), "string": auc([(p[0], 0, 0, float(p[1])) for p in v], 3),
                                                 "cosine": auc(v, 3), "link": auc(v, 4) if model.mention_link is not None else None}
                                             for k, v in subsets.items()}}
        print(name, json.dumps(res[name]), flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpora", nargs="+", default=list(VAL))
    ap.add_argument("--limit", type=int, default=500, help="Max val docs per corpus.")
    ap.add_argument("--max-words", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    model = AutoExtractor.from_pretrained(args.model).to(args.device).eval()
    res = readout(model, args.corpora, args.limit, args.max_words)
    if args.out:
        args.out.write_text(json.dumps({"model": args.model, "readout": res}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
