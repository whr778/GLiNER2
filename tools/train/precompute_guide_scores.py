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

**Cache format**, one JSON object per record that has any scored gold span::

    {"i": 12,                       # corpus position, for debugging only
     "sha1": "...",                 # sha1 of the record's `input` -- the LOOKUP KEY
     "own": ["missing", "location"],          # types the record itself declares
     "rival": {"Person/Entity": "<description>"},   # cross-record types, with descriptions
     "s": {"66": {"missing": 0.91, "Person/Entity": 0.42}}}   # span -> type -> guide score

Rivals carry descriptions because training injects them as real schema queries
(:mod:`gliner2.training.guide_scores`); a bare type name is not a query.

    uv run python tools/train/precompute_guide_scores.py \
        --corpus data/mix_natural.train.jsonl --out data/guide_scores.mix_natural.jsonl \
        --numeric-only --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
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


def score_batch(model, pending: list, batch_size: int, threshold: float) -> list:
    """Score a buffer of records in one call: [(index, record, queries, spans, scored)].

    One record per forward leaves a GPU almost idle -- the work is a single encoder pass
    over a few hundred tokens. Batching is what makes the GPU run worth doing at all.
    """
    outputs = model.batch_extract(
        [item[1].get("input", "") for item in pending],
        [Schema().entities(item[2]) for item in pending],
        batch_size=batch_size, threshold=threshold, include_confidence=True,
    )
    return [item + (rows_for_spans(out, item[3]),)
            for item, out in zip(pending, outputs)]


def rows_for_spans(out, spans: set) -> dict:
    """{span: {query: score}} from one record's extraction output."""
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
    ap.add_argument("--score-threshold", type=float, default=0.01,
                    help="minimum guide score to decode, and therefore the minimum score the "
                         "cache can hold -- this is cache SEMANTICS, not just speed. It is "
                         "also the dominant cost: asking ~100 type queries at 0.0 makes the "
                         "guide decode EVERY candidate for EVERY query and the cache then "
                         "throws nearly all of it away. Measured 0.0 -> 0.01 on 96 records: "
                         "186.3s -> 105.5s (1.77x), and NOT free -- 48/52 shared records keep "
                         "an identical top-3 rival list and 41/52 an identical own-query "
                         "reference. The records that change are ones where the guide scored "
                         "everything under 0.01, i.e. where it has no opinion and the veto "
                         "would not have fired; losing them costs coverage, not correctness.")
    ap.add_argument("--shards", type=int, default=1,
                    help="split the corpus across N processes by record index. The job is "
                         "Python-bound, not forward-bound -- a GPU measured NO faster than a "
                         "laptop CPU (3.9 vs 4.55 s/record, 4%% GPU utilisation) -- so cores, "
                         "not accelerators, are what shorten it. Concatenate the shard "
                         "outputs; each line is independent and keyed by text.")
    ap.add_argument("--shard", type=int, default=0, help="this process's shard, 0-based")
    ap.add_argument("--pool-cache", default="",
                    help="JSON file holding the corpus type pool. Built on first use and "
                         "reused after. Essentially REQUIRED with --shards: without it each "
                         "shard re-parses the whole corpus, and four concurrent builds "
                         "peaked at 3.7-5.2GB each on a 32GB machine and swapped it to a "
                         "halt.")
    ap.add_argument("--report-every", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="records per guide forward. Defaults to 1 because batching MEASURED "
                         "SLOWER on CPU -- 0.68s/record unbatched, 0.76 at 4, 1.20 at 16 -- "
                         "since every sample pads to the longest document in its batch and "
                         "CPU has no parallelism left to win that back. A GPU trades that off "
                         "differently, but not without limit: batch 32 on a 40GB A100 asked "
                         "for a 32.9 GiB attention matrix and died, because one long document "
                         "sets the padded length for all 32. Sweep before trusting it.")
    args = ap.parse_args()

    src, dst = Path(args.corpus), Path(args.out)

    # The cross-record rival pool: every distinct type name in the corpus with its
    # description. This is the vocabulary in-batch negatives are drawn from.
    #
    # CACHED ON DISK because building it costs a full parse of the corpus, and under
    # --shards every shard would otherwise redo the identical parse. That is not just
    # duplicated work: measured on a 32GB machine, four concurrent pool builds peaked at
    # 3.7-5.2GB each and drove the box into swap, which stalled every shard on page faults.
    # The pool itself is small -- a few MB of names and descriptions.
    cache = Path(args.pool_cache) if args.pool_cache else None
    if cache is not None and cache.exists():
        pool = json.loads(cache.read_text(encoding="utf-8"))
    else:
        pool = {}
        with src.open(encoding="utf-8") as fh:
            for line in fh:
                q, _ = record_queries_and_spans(json.loads(line))
                for name, desc in q.items():
                    pool.setdefault(name, desc)
        if cache is not None:
            cache.write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
    names = sorted(pool)
    print(f"[pool] {len(names)} distinct type queries across the corpus", flush=True)

    rng = random.Random(args.pool_seed)
    model = AutoExtractor.from_pretrained(args.guide, map_location=args.device)
    model.eval()
    n = kept = cells = 0
    t0 = time.time()

    def write(scored_batch, out) -> None:
        nonlocal kept, cells
        for i, rec, _, _, own, scored in scored_batch:
            if not scored:
                continue
            # Keep only the top-scoring cross-record rivals, and only ones the guide
            # actually scored: a rival at 0.0 cannot clear the veto's floor, so caching or
            # injecting it just widens the query axis for nothing. Written BEST FIRST --
            # training injects a prefix of this list, and the hardest rival is the one
            # worth spending a query slot on.
            best: dict[str, float] = {}
            for row in scored.values():
                for q, v in row.items():
                    if q not in own and v > 0.0:
                        best[q] = max(best.get(q, 0.0), v)
            ranked = [q for q, _ in sorted(best.items(), key=lambda kv: -kv[1])][:args.rival_pool]
            if not ranked:
                continue
            keep = set(ranked)
            scored = {sp: {q: v for q, v in row.items() if q in own or q in keep}
                      for sp, row in scored.items()}
            kept += 1
            cells += sum(len(v) for v in scored.values())
            text = rec.get("input", "")
            out.write(json.dumps({
                "i": i,
                # Keyed by TEXT, not corpus position: training filters, shuffles and
                # re-splits its inputs, and a positional key would then silently point at
                # another record's spans -- a failure that looks exactly like no cache.
                "sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
                "own": sorted(own),
                # Rivals carry their DESCRIPTIONS because training has to inject them as
                # real schema queries; a bare name is not a query the model can score.
                "rival": {q: pool[q] for q in ranked},
                "s": scored,
            }, ensure_ascii=False) + "\n")

    pending: list = []
    with src.open(encoding="utf-8") as fh, dst.open("w", encoding="utf-8") as out:
        for i, line in enumerate(fh):
            if args.limit and n >= args.limit:
                break
            if args.shards > 1 and i % args.shards != args.shard:
                continue
            rec = json.loads(line)
            n += 1
            queries, spans = record_queries_and_spans(rec)
            if not queries or not spans:
                continue
            if args.numeric_only and not any(any(c.isdigit() for c in s) for s in spans):
                continue
            own = set(queries)
            sample = {name: pool[name]
                      for name in rng.sample(names, min(args.pool_sample, len(names)))
                      if name not in own}
            pending.append((i, rec, {**queries, **sample}, spans, own))
            if len(pending) >= args.batch_size:
                write(score_batch(model, pending, args.batch_size, args.score_threshold), out)
                pending = []
                if args.report_every and n % args.report_every < args.batch_size:
                    rate = n / max(time.time() - t0, 1e-9)
                    print(f"  {n} records  {rate:.1f}/s  {cells} cached cells", flush=True)
        if pending:
            write(score_batch(model, pending, args.batch_size, args.score_threshold), out)

    dt = time.time() - t0
    print(f"\n{n} records read, {kept} with gold spans, {cells} cached (span, query) cells")
    print(f"{dt:.1f}s at {n/max(dt,1e-9):.1f} records/s -> {dst} "
          f"({dst.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
