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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


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
    model = AutoExtractor.from_pretrained(args.checkpoint, map_location="cpu").train()
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

    rows = []
    print(f"\n  {'window':>8} {'gold':>10} {'in candidates':>15} {'coverage':>10} {'gold/window':>12}")
    for win in [int(w) for w in args.windows.split(",")]:
        recs = chunk_records(raw, tokenizer=proc.tokenizer, window_size=win,
                             stride=win, show_progress=False)
        coll = ExtractorCollator(
            proc, is_training=True, max_len=win, architecture="boundary",
            max_gold_per_query=int(head.get("max_gold_per_query", 32)),
            error_policy="skip", event_records=bool(head.get("event_records", False)))
        dl = DataLoader(ExtractorDataset(data=recs, max_samples=-1, shuffle=False,
                                         seed=42, validate=False),
                        batch_size=1, shuffle=False, num_workers=0, collate_fn=coll)
        hit = gold_total = n_win = 0
        for i, batch in enumerate(dl):
            if i >= args.max_batches:
                break
            with torch.no_grad():
                model(batch)
            for m in holders:
                st = getattr(m, "_last_proposal_stats", None)
                if st is None or st.gold_total is None:
                    continue
                hit += int(st.gold_hit_without_injection)
                gold_total += int(st.gold_total)
                m._last_proposal_stats = None      # don't double count the next window
            n_win += 1
        stats = {"pos": hit}
        if not gold_total:
            print(f"  {win:>8} {'-':>10} {'-':>15}   no gold reached the loop")
            continue
        cov = stats["pos"] / gold_total
        rows.append({"window": win, "gold": gold_total, "in_candidates": stats["pos"],
                     "coverage": cov, "windows": n_win})
        print(f"  {win:>8} {gold_total:>10,} {stats['pos']:>15,} {cov*100:>9.1f}% "
              f"{gold_total/max(n_win,1):>12.1f}", flush=True)

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
