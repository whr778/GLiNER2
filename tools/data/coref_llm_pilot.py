"""Pilot: can an LLM CLUSTER English news mentions well enough to train the coref link on?

    uv run python tools/data/coref_llm_pilot.py --docs 40 --draws 3 --out /Volumes/Development/tmp/coref_pilot

The LLM clusters NUMBERED candidate mentions; it does not find them. That is what a purchase on
cc_news would buy (candidates = annotated entities/arguments + pronouns), and what the link head
consumes (it is handed its mentions). Candidates here are WikiEvents TRAIN gold coref mentions plus
DISTRACTORS -- occurrences of entity surfaces and pronouns outside every gold cluster, gold singletons
-- because gold clusters carry no singletons, and gold-only candidates would tell the LLM every
mention has a partner. Docs with overlapping gold mentions are skipped (inline tags cannot nest).

Scored against gold as pairwise link precision / recall / F1 over pairs of GOLD-CLUSTERED mentions
(all, and those touching a pronoun); distractor pairs are unscored, since the gold is partial, and
the share of predicted links touching a distractor is reported beside it. Draws of one model are
scored against each other for the redraw floor. Real cost comes from the batch's own token usage. Test and dev stay blind: TRAIN docs only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotate_coref import MODELS, PRONOUNS, SYSTEM, USER, parse, render  # noqa: E402,F401
from synthetic.providers import AnthropicProvider, ProviderConfig  # noqa: E402

def candidates(rec: dict) -> list[dict]:
    """Gold cluster mentions (cluster id) plus non-overlapping distractors (cluster None), by offset."""
    gold = [{"start": m["start"], "end": m["end"], "text": m["text"], "cluster": i}
            for i, c in enumerate(rec["coref_clusters"]) for m in c]
    taken = [(m["start"], m["end"]) for m in gold]
    surfaces = {s for v in (rec["output"].get("entities") or {}).values() for s in v}
    words = "|".join(sorted((re.escape(s) for s in surfaces | PRONOUNS), key=len, reverse=True))
    extra = []
    for m in re.finditer(rf"(?<!\w)(?:{words})(?!\w)", rec["input"]):
        span = (m.start(), m.end())
        if any(a < span[1] and span[0] < b for a, b in taken):
            continue
        if m.group(0).casefold() in PRONOUNS or m.group(0) in surfaces:
            extra.append({"start": span[0], "end": span[1], "text": m.group(0), "cluster": None})
            taken.append(span)
    return sorted(gold + extra, key=lambda m: m["start"])


def overlapping(rec: dict) -> bool:
    ms = sorted((m["start"], m["end"]) for c in rec["coref_clusters"] for m in c)
    return any(b[0] < a[1] for a, b in zip(ms, ms[1:]))


def pair_counts(pred: list[int], gold: list, pron: list[bool], subset: str) -> tuple[int, int, int]:
    """(true positive, predicted, gold) coreferent pairs; subset 'all' or 'pronoun'. Pairs touching a
    distractor (gold None) are unscored: WikiEvents' coref gold is partial, so a distractor can truly
    corefer ("explosive devices" ... "the devices" are unclustered in gold)."""
    tp = p = g = 0
    for i, j in itertools.combinations(range(len(pred)), 2):
        if gold[i] is None or gold[j] is None or (subset == "pronoun" and not (pron[i] or pron[j])):
            continue
        same_p, same_g = pred[i] == pred[j], gold[i] == gold[j]
        tp, p, g = tp + (same_p and same_g), p + same_p, g + same_g
    return tp, p, g


def prf(tp: int, p: int, g: int) -> dict:
    pr, rc = tp / p if p else 0.0, tp / g if g else 0.0
    return {"precision": round(pr, 4), "recall": round(rc, 4), "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0}


def score(preds: dict, docs: list[dict], against) -> dict:
    """Micro pairwise P/R/F1 of preds {doc index: labels} against against(doc index) labels."""
    out = {}
    for subset in ("all", "pronoun"):
        tot = [0, 0, 0]
        for d, pred in preds.items():
            c = docs[d]["cands"]
            pron = [m["text"].casefold() in PRONOUNS for m in c]
            tot = [a + b for a, b in zip(tot, pair_counts(pred, against(d), pron, subset))]
        out[subset] = prf(*tot)
    return out


def distractor_links(preds: dict, docs: list[dict]) -> float:
    """Share of predicted coreferent pairs with at least one distractor (unscored) mention."""
    hit = tot = 0
    for d, pred in preds.items():
        gold = [m["cluster"] for m in docs[d]["cands"]]
        for i, j in itertools.combinations(range(len(pred)), 2):
            if pred[i] == pred[j]:
                tot += 1
                hit += gold[i] is None or gold[j] is None
    return round(hit / tot, 4) if tot else 0.0


def doc_f1(preds: dict, docs: list[dict]) -> float:
    """Mean per-document pairwise F1: pair counts grow with the SQUARE of a doc's mentions, so one
    2,000-mention doc held 47% of the pilot's gold pairs and set the micro figure alone."""
    f = [prf(*pair_counts(p, [m["cluster"] for m in docs[d]["cands"]], [False] * len(p), "all"))["f1"]
         for d, p in preds.items()]
    return round(sum(f) / len(f), 4) if f else 0.0


def evaluate(replies: dict, docs: list[dict], n_draws: int, max_chars: int) -> dict:
    """Scores per draw against gold and between draws, over all docs and over docs <= max_chars."""
    draws = {k: {} for k in range(n_draws)}
    unusable = 0
    for cid, raw in replies.items():
        d, k = map(int, cid.split("_"))
        labels = parse(raw, len(docs[d]["cands"]))
        if labels is None:
            unusable += 1
        else:
            draws[k][d] = labels
    out = {"replies": len(replies), "unusable": unusable}
    short = {d for d, doc in enumerate(docs) if len(doc["rec"]["input"]) <= max_chars}
    for scope, keep in (("all_docs", set(range(len(docs)))), (f"docs_le_{max_chars}_chars", short)):
        dr = {k: {d: v for d, v in p.items() if d in keep} for k, p in draws.items()}
        gold = lambda d: [m["cluster"] for m in docs[d]["cands"]]  # noqa: E731
        out[scope] = {
            "docs": len(keep),
            "vs_gold": {k: {**score(p, docs, gold), "doc_mean_f1": doc_f1(p, docs),
                            "links_touching_distractor": distractor_links(p, docs)} for k, p in dr.items()},
            "redraw": {f"{a}v{b}": score({d: dr[a][d] for d in dr[a] if d in dr[b]}, docs, lambda d, b=b: dr[b][d])
                       for a, b in itertools.combinations(range(n_draws), 2)}}
    return out


def usage_cost(provider: AnthropicProvider, batch_id: str, price_in: float, price_out: float) -> dict:
    tin = tout = 0
    for r in provider._client.messages.batches.results(batch_id):
        if r.result.type == "succeeded":
            tin += r.result.message.usage.input_tokens
            tout += r.result.message.usage.output_tokens
    return {"input_tokens": tin, "output_tokens": tout, "usd": round(tin / 1e6 * price_in + tout / 1e6 * price_out, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", default="data/wikievents.train.jsonl")
    ap.add_argument("--docs", type=int, default=40)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="build and show prompts, send nothing")
    ap.add_argument("--rescore", action="store_true", help="score the saved replies in --out, send nothing")
    ap.add_argument("--max-chars", type=int, default=12000,
                    help="also score docs up to this length: the purchase corpora's own maximum")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.train, encoding="utf-8")]
    pool = [r for r in recs if r.get("coref_clusters") and not overlapping(r)]
    docs = [{"rec": r, "cands": candidates(r)} for r in random.Random(args.seed).sample(pool, args.docs)]
    n_gold = sum(m["cluster"] is not None for d in docs for m in d["cands"])
    n_all = sum(len(d["cands"]) for d in docs)
    print(f"[pilot] {len(docs)} of {len(pool)} usable TRAIN docs; {n_all} candidates, {n_gold} gold-clustered, "
          f"{n_all - n_gold} distractors")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(USER.format(text=render(docs[0]["rec"]["input"], docs[0]["cands"]))[:3000])
        return 0

    result = {"docs": len(docs), "candidates": n_all, "gold_clustered": n_gold, "models": {}}
    if args.rescore:
        for name in args.models:
            replies = json.loads((args.out / f"{name}.replies.json").read_text(encoding="utf-8"))
            result["models"][name] = {"model": MODELS[name][0], **evaluate(replies, docs, args.draws, args.max_chars)}
        (args.out / "rescore.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0
    for name in args.models:
        model, pin, pout = MODELS[name]
        provider = AnthropicProvider(ProviderConfig(provider="anthropic", model=model, max_tokens=4000))
        items = [(f"{d}_{k}", SYSTEM, USER.format(text=render(doc["rec"]["input"], doc["cands"])))
                 for d, doc in enumerate(docs) for k in range(args.draws)]
        replies = provider.complete_batch(items, id_path=args.out / f"{name}.batch_id")
        batch_id = (args.out / f"{name}.batch_id").read_text().strip()
        (args.out / f"{name}.replies.json").write_text(json.dumps(replies, ensure_ascii=False), encoding="utf-8")
        result["models"][name] = {"model": model, "batch_id": batch_id,
                                  **evaluate(replies, docs, args.draws, args.max_chars),
                                  "cost": usage_cost(provider, batch_id, pin, pout)}
        print(f"[pilot] {name}", json.dumps(result["models"][name]), flush=True)
    (args.out / "pilot.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
