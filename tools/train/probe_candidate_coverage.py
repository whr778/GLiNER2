"""Why does emission stop scaling with the window? Measure gold COVERAGE by candidates.

`candidate_budget` (128 by default, and eb17 leaves it there) caps how many candidate
spans are enumerated per query -- INDEPENDENT of how much text the window holds. The
config sizes it from corpora whose median document is 457 tokens. On a 4096-token window
the span space is far larger while the budget is unchanged, so gold spans outside the top
`candidate_budget` never enter the candidate set and are UNREACHABLE AT ANY THRESHOLD.

READ `gold_hit_without_injection`, NOT THE LOSS LABELS. The first version of this probe
counted positives in `candidate_pair_loss` and reported 100% coverage at 256, 512 and 1024
tokens -- because `gold_injection_prob` defaults to **1.0**, so training INJECTS gold into
the candidate set and the answer was guaranteed before the model ran. `ProposalStats`
already carries the honest number: `gold_hit_without_injection` is computed against
`pre_keys`/`pre_valid`, the proposals BEFORE injection, which is the inference-time
question. It needs `collect_diagnostics`.

A coverage that falls with window size proves the cap. A flat coverage refutes it and
sends the search elsewhere (threshold behaviour, attention dilution).

    uv run python tools/train/probe_candidate_coverage.py \
        --config tools/train/config/base/eb17-best.yaml \
        --checkpoint whr778/gliner2-eb17-best --data data/cc_news_long.test.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def apply_pool_overrides(model, args) -> bool:
    """Switch the candidate pool and size the shared pool, printing what took effect.

    `candidate_pool` is read from the head's settings on every forward, but the shared
    builder's sizes are fixed at construction, so they are set on the builder itself.
    """
    import dataclasses
    import torch
    from gliner2.configuration import BoundaryHeadSettings
    heads = [m for m in model.modules() if isinstance(getattr(m, "settings", None), BoundaryHeadSettings)]
    builders = [m for m in model.modules() if type(m).__name__ == "DocumentCandidatePool"]
    if not heads or not builders:
        print(f"[cov] *** found {len(heads)} head(s), {len(builders)} pool builder(s) -- "
              "the override would be DECORATIVE. Refusing. ***")
        return False
    if args.pool:
        for h in heads:
            h.settings = dataclasses.replace(h.settings, candidate_pool=args.pool)
    for b in builders:
        if args.pool_boundary_top_k:
            b.pool_boundary_top_k = args.pool_boundary_top_k
        if args.pool_size:
            b.pool_size = args.pool_size
        if args.zero_compat:
            with torch.no_grad():
                for proj in (b.start_projection, b.end_projection):
                    proj.weight.zero_()
                    proj.bias.zero_()
    b = builders[0]
    print(f"[cov] POOL: candidate_pool={heads[0].settings.candidate_pool} "
          f"pool_boundary_top_k={b.pool_boundary_top_k} pool_size={b.pool_size} "
          f"min_pool_per_query={b.min_pool_per_query} "
          f"compat_weight_abs_sum={float(b.start_projection.weight.abs().sum()):.3f}", flush=True)
    return True


def main() -> int:
    import torch
    import yaml
    from torch.utils.data import DataLoader

    import train as T
    from gliner2 import AutoExtractor
    from gliner2.models.boundary import model as BM
    from gliner2.training.chunking import chunk_records
    from gliner2.training.trainer import ExtractorCollator, ExtractorDataset

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/cc_news_long.test.jsonl")
    ap.add_argument("--windows", default="256,512,1024,2048,4096")
    ap.add_argument("--max-records", type=int, default=40)
    ap.add_argument("--max-batches", type=int, default=60)
    ap.add_argument("--candidate-budget", type=int, default=0, help="0 = leave as built")
    ap.add_argument("--start-top-k", type=int, default=0)
    ap.add_argument("--end-top-k", type=int, default=0)
    ap.add_argument("--top-k-alpha", type=float, default=-1.0,
                    help="length-adaptive boundary top-k: ceil(alpha*(n_boundaries-1)), "
                         "clamped to [base_k, k_max]. The mechanism EXISTS and ships "
                         "DISABLED at alpha=0.0. -1 leaves it alone.")
    ap.add_argument("--top-k-max", type=int, default=0)
    ap.add_argument("--pool", choices=("per_query", "shared"), default=None,
                    help="switch candidate_pool at runtime (the head reads it every forward)")
    ap.add_argument("--pool-boundary-top-k", type=int, default=0)
    ap.add_argument("--pool-size", type=int, default=0)
    ap.add_argument("--zero-compat", action="store_true",
                    help="zero the shared pool's pairing projections, which are untrained in "
                         "per_query checkpoints, so ranking uses trained boundary scores only")
    ap.add_argument("--max-gold", type=int, default=0,
                    help="gold cap for the probe only (0 = config value); a sample over the "
                         "cap would raise, and dropping it would bias coverage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu",
                    help="cuda for checkpoints on flash_attention_2: the kernels FA2 op is CUDA-only "
                         "and raises NotImplementedError on CPU (eb18, 2026-10-02)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tc = cfg.get("training") or {}
    head = (cfg.get("model") or {}).get("boundary_head") or {}
    budget = int(head.get("candidate_budget", 128))
    fns = T._category_fns(T.load_labels_cfg(cfg, config_path=args.config))

    raw = []
    for line in Path(args.data).open(encoding="utf-8"):
        if line.strip():
            raw.append(T.transform_record(json.loads(line), fns))
        if len(raw) >= args.max_records:
            break
    model = AutoExtractor.from_pretrained(args.checkpoint, map_location="cpu").to(args.device).train()
    # train() is needed for the gold-coverage stats, but it also turns dropout ON, which
    # randomly perturbs the scores and understates what inference can reach.
    n_dropout = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
            n_dropout += 1
    print(f"[cov] dropout disabled in {n_dropout} module(s)", flush=True)
    proc = model.processor
    print(f"[cov] {len(raw)} docs from {args.data}; candidate_budget={budget}", flush=True)

    # Diagnostics off means ProposalStats is None and this probe silently measures
    # nothing, so turn it on wherever the flag lives and fail loudly if it never appears.
    holders = [m for m in model.modules() if hasattr(m, "collect_diagnostics")]
    for m in holders:
        m.collect_diagnostics = True
    if not holders:
        print("[cov] *** no module exposes collect_diagnostics -- UNMEASURED ***")
        return 1

    # PROPOSAL SETTINGS ARE BAKED AT CONSTRUCTION (model.py:246 builds them in __init__),
    # so setting BoundaryHeadSettings after from_pretrained lands too late and changes
    # NOTHING -- the decorative-override trap. Replace the proposer's own frozen settings,
    # which its forward re-reads each call, and PRINT what actually took effect.
    import dataclasses
    changes = {}
    if args.candidate_budget:
        changes["candidate_budget"] = args.candidate_budget
    if args.start_top_k:
        changes["start_top_k"] = args.start_top_k
    if args.end_top_k:
        changes["end_top_k"] = args.end_top_k
    if args.top_k_alpha >= 0:
        changes["boundary_top_k_alpha"] = args.top_k_alpha
    if args.top_k_max:
        changes["boundary_top_k_max"] = args.top_k_max
    proposers = [m for m in model.modules() if hasattr(m, "settings")
                 and hasattr(getattr(m, "settings"), "candidate_budget")]
    if changes:
        if not proposers:
            print("[cov] *** asked to change the budget but found no proposer -- the "
                  "override would be DECORATIVE. Refusing. ***")
            return 1
        for m in proposers:
            m.settings = dataclasses.replace(m.settings, **changes)
        eff = proposers[0].settings
        print(f"[cov] OVERRIDE APPLIED to {len(proposers)} proposer(s): {changes}")
        print(f"[cov] effective: start_top_k={eff.start_top_k} end_top_k={eff.end_top_k} "
              f"candidate_budget={eff.candidate_budget} alpha={eff.boundary_top_k_alpha} "
              f"k_max={eff.boundary_top_k_max}", flush=True)
    else:
        eff = proposers[0].settings if proposers else None
        if eff is not None:
            print(f"[cov] BASELINE as built: start_top_k={eff.start_top_k} "
                  f"end_top_k={eff.end_top_k} candidate_budget={eff.candidate_budget} "
                  f"alpha={eff.boundary_top_k_alpha} k_max={eff.boundary_top_k_max}", flush=True)

    if args.pool or args.pool_boundary_top_k or args.pool_size or args.zero_compat:
        if not apply_pool_overrides(model, args):
            return 1

    rows = []
    print(f"\n  {'window':>8} {'gold':>10} {'in candidates':>15} {'coverage':>10} "
          f"{'gold/window':>12}   boundary coverage")
    for win in [int(w) for w in args.windows.split(",")]:
        # Training-mode preprocessing randomly drops entities, labels and structure fields
        # (processor.py `remove_entity_prob` and friends), so an unseeded run scores a
        # different set of queries each time: gold totals drifted 8,620-8,921 on identical
        # data. Seeding per window gives every arm the same queries.
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        recs = chunk_records(raw, tokenizer=proc.tokenizer, window_size=win,
                             stride=win, show_progress=False)
        coll = ExtractorCollator(
            proc, is_training=True, max_len=win, architecture="boundary",
            max_gold_per_query=args.max_gold or int(head.get("max_gold_per_query", 32)),
            error_policy="skip", event_records=bool(head.get("event_records", False)))
        dl = DataLoader(ExtractorDataset(data=recs, max_samples=-1, shuffle=False,
                                         seed=42, validate=False),
                        batch_size=1, shuffle=False, num_workers=0, collate_fn=coll)
        hit = gold_total = n_win = 0
        s_hit = e_hit = b_total = 0
        for i, batch in enumerate(dl):
            if i >= args.max_batches:
                break
            with torch.no_grad():
                model(batch.to(args.device))
            for m in holders:
                st = getattr(m, "_last_proposal_stats", None)
                if st is None or st.gold_total is None:
                    continue
                hit += int(st.gold_hit_without_injection)
                gold_total += int(st.gold_total)
                # SEPARATE THE TWO FAILURES. High start/end hit with low PAIR coverage means
                # the boundaries were found and the PAIRING lost them (ends_per_start).
                # Low start/end hit means the scorer never ranked them in -- a model
                # problem no budget fixes.
                if st.start_hit is not None:
                    s_hit += int(st.start_hit)
                    e_hit += int(st.end_hit)
                    b_total += int(st.boundary_total)
                m._last_proposal_stats = None      # don't double count the next window
            n_win += 1
        stats = {"pos": hit}
        if not gold_total:
            print(f"  {win:>8} {'-':>10} {'-':>15}   no gold reached the loop")
            continue
        cov = stats["pos"] / gold_total
        rows.append({"window": win, "gold": gold_total, "in_candidates": stats["pos"],
                     "coverage": cov, "windows": n_win})
        sc = (s_hit / b_total * 100) if b_total else float("nan")
        ec = (e_hit / b_total * 100) if b_total else float("nan")
        rows[-1].update({"start_cov": sc, "end_cov": ec})
        print(f"  {win:>8} {gold_total:>10,} {stats['pos']:>15,} {cov*100:>9.1f}% "
              f"{gold_total/max(n_win,1):>12.1f}   start {sc:>5.1f}%  end {ec:>5.1f}%",
              flush=True)

    print()
    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        drop = first["coverage"] - last["coverage"]
        print(f"[cov] coverage {first['window']} -> {last['window']} tokens: "
              f"{first['coverage']*100:.1f}% -> {last['coverage']*100:.1f}%  "
              f"({-drop*100:+.1f} pts)")
        if drop > 0.05:
            print(f"[cov] CAP CONFIRMED: gold leaves the candidate set as the window grows.")
            print(f"[cov] Those mentions are unreachable at ANY threshold -- raising recall")
            print(f"[cov] needs candidate_budget (currently {budget}) to scale with the window.")
        else:
            print("[cov] Coverage is FLAT, so the budget is not the cap and the loss is")
            print("[cov] downstream -- threshold behaviour or scoring, not enumeration.")
    else:
        print("[cov] *** fewer than two windows measured -- UNMEASURED, not flat. ***")
        return 1
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
