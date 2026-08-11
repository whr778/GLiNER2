"""Precompute the frozen guide's scores once, so GIST costs no forward pass per step.

A guide forward on every training step would roughly double step cost. It is also pure
waste: the guide is frozen and the corpus is fixed, so its scores are constant across
epochs. The only question is what has to be cached, and the answer is much smaller than it
first appears.

**Only gold spans matter, exactly.** The veto is

    vetoed = selected & has_positive & (guide > pos_guide + margin) & ~positive

and ``has_positive`` gates it, so the veto can only ever fire on a candidate that is
POSITIVE for some query -- a gold span. The live model's proposed candidates drift as it
trains, but a cell on a non-gold candidate is never vetoed, so it never needs a guide score.
The cache is therefore exact, not an approximation: gold spans x the queries that record
declares, both fixed.

That turns an O(steps) cost into O(records), and the cached object into roughly
``n_records x n_gold_spans x n_queries`` floats -- megabytes, not gigabytes.

**Rivals must come from OTHER records.** A first version scored gold spans only against the
types the record itself declares, and that is the one case where a guide is not merely
unnecessary but harmful: within a record, gold is authoritative. If the schema declares both
``ReleasedDate`` and ``StartDate`` and gold assigns the span to ``ReleasedDate``, then
``StartDate`` is *definitively* wrong there, and vetoing it would delete a correct hard
negative. Measured on 200 records, a same-record rival outscores the gold owner 23.5% of the
time -- all of it material a veto must NOT touch.

GIST's negatives are the in-batch ones: type queries belonging to OTHER records, where gold
says nothing because the corpus declares a per-record vocabulary out of 17,128 types. Those
are what the guide has to adjudicate, so those are what is cached. Own-record types are
cached too, but flagged, so the training side can trust gold there and the guide only where
gold is silent.

**Sample WIDE, then keep the top scorers.** A hard negative is by definition a type that
scores the span highly, so no embedder or similarity model is needed -- the guide's own
scores rank them. What is needed is a pool wide enough to contain one. Measured on a real
casualty span, `killed a man and his 14-year-old daughter`:

    pool  12   person.name 0.02, health condition 0.00, Commodity 0.00      -- noise
    pool 200   Person/Entity 0.56, Human being 0.30, Family Relationship 0.09

Twelve random draws from 17,128 types essentially never contain a confusable one; two
hundred do, and the survivors are coherent. That example is also GIST behaving correctly:
the span genuinely IS a person reference, so `Person/Entity` outscoring the gold casualty
type is a true false negative, and penalizing it would be wrong.

Note the descriptions cannot help here. The types that matter carry none -- `Casualties and
Losses` appears 623 times with zero real descriptions -- and where descriptions do exist they
are instance-specific (`Location` -> "A city in England"), so similarity over them would
compare document blurbs rather than type definitions.

**Cost is the open problem, and filtering does not solve it.** At pool=100, 4.55s/record on
CPU. Restricting to records with a numeric gold span keeps 66.6% (47,475 of 71,327); a
tighter filter on count-ish TYPE NAMES keeps 37.1%, still ~37 CPU-hours. Only 3 of 8 sampled
records yield a coherent rival at all, so most of the compute produces nothing cacheable and
there is no cheap way to know which in advance. **This wants a GPU** -- roughly 2 hours there
against ~55 on CPU, for a one-time cost against a guide forward on every training step.

    uv run python tools/train/precompute_guide_scores.py \
        --corpus data/mix_natural.train.jsonl --out data/guide_scores.mix_natural.jsonl \
        --numeric-only
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from gliner2 import AutoExtractor, Schema


def record_queries_and_spans(rec: dict):
    """(query name -> description, gold surface spans) as the record declares them.

    Covers both shapes the mixes use: `entities` as ``{type: [spans]}`` with optional
    `entity_descriptions`, and `json_structures` as ``[{task: {field: value}}]``.
    """
    out = rec.get("output") or {}
    queries: dict[str, str] = {}
    spans: set[str] = set()

    ents = out.get("entities")
    if isinstance(ents, dict):
        descs = out.get("entity_descriptions") or {}
        for t, vals in ents.items():
            queries[t] = descs.get(t, t) if isinstance(descs, dict) else t
            for v in (vals if isinstance(vals, list) else [vals]):
                if isinstance(v, str) and v.strip():
                    spans.add(v.strip())

    for block in (out.get("json_structures") or []):
        if not isinstance(block, dict):
            continue
        for _, fields in block.items():
            if not isinstance(fields, dict):
                continue
            for fname, v in fields.items():
                queries.setdefault(fname, fname)
                for x in (v if isinstance(v, list) else [v]):
                    if isinstance(x, str) and x.strip():
                        spans.add(x.strip())
    return queries, spans


def _entities(out) -> dict:
    e = out.get("entities") or {}
    return e[0] if isinstance(e, list) else e


def score_record(model, text: str, queries: dict, spans: set) -> dict:
    """{span: {query: score}} -- one guide forward for the whole record."""
    if not queries or not spans:
        return {}
    out = model.extract(text, Schema().entities(queries), threshold=0.0,
                        include_confidence=True)
    per_query = _entities(out)
    scored: dict = {}
    for span in spans:
        row = {}
        for q, items in per_query.items():
            best = 0.0
            for it in (items or []):
                txt = it["text"] if isinstance(it, dict) else str(it)
                conf = float(it.get("confidence", 0.0)) if isinstance(it, dict) else 0.0
                if span == txt or span in txt or txt in span:
                    best = max(best, conf)
            if best > 0.0:
                row[q] = round(best, 4)
        if row:
            scored[span] = row
    return scored


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--guide", default="fastino/gliner2-base-v1",
                    help="FROZEN guide checkpoint. Must not be the model being trained: a "
                         "cell is mined because the live model scores it highly, so a live "
                         "self-guide would veto exactly the negatives it should select.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--numeric-only", action="store_true",
                    help="skip records with no digit-bearing gold span. The count-type "
                         "boundary lives there, and C++ docs and baseball contribute nothing "
                         "to it -- but it is a WEAK filter, keeping 66.6%% of records, "
                         "because dates, IDs and share counts are numeric too.")
    ap.add_argument("--rival-pool", type=int, default=12,
                    help="cross-record rivals RETAINED per record, after ranking by guide "
                         "score. These are the in-batch negatives a guide has to adjudicate; "
                         "own-record types are settled by gold and need no guide.")
    ap.add_argument("--pool-sample", type=int, default=100,
                    help="types scored per record before keeping the top --rival-pool. "
                         "Wide enough to contain a confusable rival, and cost is linear in "
                         "it. Swept on 8 numeric gold spans -- 25: 1.44s/rec, mean top rival "
                         "0.077; 50: 2.38s, 0.062; 100: 4.55s, 0.296; 200: 9.11s, 0.309. The "
                         "knee is at 100, which buys 200's quality for half the compute.")
    ap.add_argument("--pool-seed", type=int, default=0)
    ap.add_argument("--report-every", type=int, default=50)
    args = ap.parse_args()

    src, dst = Path(args.corpus), Path(args.out)

    # Build the cross-record rival pool first: every distinct type name in the corpus with
    # its description. This is the vocabulary in-batch negatives are drawn from.
    pool: dict[str, str] = {}
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            q, _ = record_queries_and_spans(json.loads(line))
            for name, desc in q.items():
                pool.setdefault(name, desc)
    names = sorted(pool)
    print(f"[pool] {len(names)} distinct type queries across the corpus")

    rng = random.Random(args.pool_seed)
    model = AutoExtractor.from_pretrained(args.guide, map_location=args.device)
    model.eval()
    n = kept = cells = 0
    t0 = time.time()
    with src.open(encoding="utf-8") as fh, dst.open("w", encoding="utf-8") as out:
        for i, line in enumerate(fh):
            if args.limit and n >= args.limit:
                break
            rec = json.loads(line)
            n += 1
            queries, spans = record_queries_and_spans(rec)
            if not queries or not spans:
                continue
            if args.numeric_only and not any(any(c.isdigit() for c in s) for s in spans):
                continue
            own = set(queries)
            sample = {n: pool[n] for n in rng.sample(names, min(args.pool_sample, len(names)))
                      if n not in own}
            scored = score_record(model, rec.get("input", ""),
                                  {**queries, **sample}, spans)
            if not scored:
                continue
            # Keep only the top-scoring cross-record rivals: the rest are noise, and caching
            # them would bloat the file with cells the veto can never act on.
            best: dict[str, float] = {}
            for row in scored.values():
                for q, v in row.items():
                    if q not in own:
                        best[q] = max(best.get(q, 0.0), v)
            keep = {q for q, _ in sorted(best.items(), key=lambda kv: -kv[1])[:args.rival_pool]}
            scored = {sp: {q: v for q, v in row.items() if q in own or q in keep}
                      for sp, row in scored.items()}
            scored = {"own": sorted(own), "rival": sorted(keep), "s": scored}
            kept += 1
            cells += sum(len(v) for v in scored["s"].values())
            out.write(json.dumps({"i": i, "scores": scored}, ensure_ascii=False) + "\n")
            if args.report_every and n % args.report_every == 0:
                rate = n / max(time.time() - t0, 1e-9)
                print(f"  {n} records  {rate:.1f}/s  {cells} cached cells", flush=True)

    dt = time.time() - t0
    print(f"\n{n} records read, {kept} with gold spans, {cells} cached (span, query) cells")
    print(f"{dt:.1f}s at {n/max(dt,1e-9):.1f} records/s -> {dst} "
          f"({dst.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
