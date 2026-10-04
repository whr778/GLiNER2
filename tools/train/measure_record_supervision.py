"""How many trigger candidates does the record head SUPERVISE? Counted on real training batches.

Natural mode (events): every anchor (trigger) candidate becomes an instance hypothesis, but
`compute_group_loss` trains fields only for the instance each GOLD record's trigger seeds, and
returns object_loss 0 (records.py). Every other instance gets no argument loss and no existence
loss -- it is never told "these are not your arguments". This counts, per event group: instance
hypotheses, gold records, records trained (gold trigger found among the candidates), so the
supervised share of instances is a measured number, at the injection rates a run actually uses.

    uv run python tools/train/measure_record_supervision.py --batches 8
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/train"))
import measure_absent_dilution as D  # noqa: E402

import gliner2.models.boundary.records as R  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="tools/train/config/base/eb18-balanced.yaml")
    ap.add_argument("--checkpoint", default="whr778/gliner2-eb18-balanced")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tr, head = cfg["training"], cfg["model"]["boundary_head"]
    from gliner2 import AutoExtractor
    from gliner2.training.chunking import chunk_records
    from gliner2.training.negatives import NegativeLabels
    from gliner2.training.trainer import ExtractorCollator, ExtractorDataset

    model = AutoExtractor.from_pretrained(args.checkpoint, map_location="cpu")
    proc = model.processor
    spec = importlib.util.spec_from_file_location("bnp", ROOT / "tools/data/build_negative_pools.py")
    bnp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bnp)
    neg = NegativeLabels(bnp.build_pools(cfg, Path(args.config), log=lambda *a: None), tr["negative_labels_per_dim"],
                         seed=int(tr.get("negative_label_seed", 42)), partial=cfg["data"].get("partial_annotation"))
    sampled, totals = D.load_records(cfg, args.config, 40, args.seed)
    event_corpora = [n for n in sorted(sampled) if any((r["output"].get("events")) for r in sampled[n])]
    coll = ExtractorCollator(proc, is_training=True, max_len=tr.get("max_len"), architecture="boundary",
                             max_gold_per_query=int(head.get("max_gold_per_query", 32)),
                             on_capacity_exceeded=tr.get("on_capacity_exceeded", "raise"), error_policy="skip",
                             event_records=bool(head.get("event_records", False)))
    counts = Counter()
    real = R.compute_group_loss

    def counted(group, records):
        if group.spec.mode == "natural" and group.spec.task_type == "events":
            counts["groups"] += 1
            counts["instances"] += int(group.num_instances)
            counts["gold_records"] += len(records)
            out = real(group, records)
            counts["trained"] += out["field_count"] // max(len(group.field_specs), 1)
            return out
        return real(group, records)

    R.compute_group_loss = counted
    import gliner2.models.boundary.model as M
    if hasattr(M, "compute_group_loss"):
        M.compute_group_loss = counted
    print(f"[supervision] event corpora sampled: {event_corpora}")
    for p_inj in (1.0, 0.25):
        counts.clear()
        model.boundary_head.set_gold_injection_prob(p_inj)
        rng = random.Random(args.seed)
        for b in range(args.batches):
            corpora = rng.choices(event_corpora, [totals[n] for n in event_corpora], k=int(tr["batch_size"]))
            recs = [rng.choice([r for r in sampled[c] if r["output"].get("events")]) for c in corpora]
            chunks = chunk_records(recs, tokenizer=proc.tokenizer, window_size=int(tr["max_len"]),
                                   stride=int(tr["window_stride"]), show_progress=False)
            ds = ExtractorDataset(data=chunks, max_samples=-1, shuffle=False, seed=42, validate=False, negatives=neg)
            random.seed(args.seed + b)
            batch = coll([ds[i] for i in range(min(len(ds), int(tr["batch_size"])))])
            model.train()
            torch.manual_seed(args.seed + b)
            with torch.no_grad():
                model(batch)
        g, i, gr, t = counts["groups"], counts["instances"], counts["gold_records"], counts["trained"]
        print(f"[supervision] p_inj {p_inj}: {g} event groups | instance hypotheses {i} | gold records {gr} | "
              f"records trained {t} ({100 * t / max(gr, 1):.1f}% of gold) | instances with ANY argument loss "
              f"{t} of {i} = {100 * t / max(i, 1):.2f}% | instances never supervised {i - t}")


if __name__ == "__main__":
    main()
