"""Which TASKS does the junction's column loss train? Counted per task type on real training batches.

`RecordHead.forward_group` adds the junction link to EVERY natural-mode record group
(`spec.mode == "natural"`, no task check), and `compute_group_loss` runs the column loss on any
such group with gold. The junction was designed, A/B'd and gated on EVENTS; this counts what else
it touches -- per corpus and task type: natural groups, groups with gold, column-loss terms.

    uv run python tools/train/measure_link_by_task.py --checkpoint <junction checkpoint>
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="tools/train/config/base/eb19.yaml")
    ap.add_argument("--docs", type=int, default=16, help="training docs per corpus")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tr, head = cfg["training"], cfg["model"]["boundary_head"]
    from gliner2 import AutoExtractor
    from gliner2.training.chunking import chunk_records
    from gliner2.training.trainer import ExtractorCollator, ExtractorDataset

    model = AutoExtractor.from_pretrained(args.checkpoint, map_location="cpu")
    s = model.boundary_settings
    print(f"[link] record_link_mode={s.record_link_mode} column_weight={s.record_link_column_weight} "
          f"column_negatives={s.record_link_column_negatives}")
    coll = ExtractorCollator(model.processor, is_training=True, max_len=tr.get("max_len"), architecture="boundary",
                             max_gold_per_query=int(head.get("max_gold_per_query", 32)),
                             on_capacity_exceeded=tr.get("on_capacity_exceeded", "raise"), error_policy="skip",
                             event_records=True)
    model.processor.sampling_config.remove_events_prob = 0.0
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()

    tally, corpus = Counter(), {"name": ""}
    real_loss, real_col = R.compute_group_loss, R._column_loss

    def loss_spy(group, records, *a, **k):
        key = (corpus["name"], group.spec.task_type, group.spec.mode)
        tally[key + ("groups",)] += 1
        tally[key + ("with_gold",)] += bool(records)
        tally[key + ("linked",)] += model.record_decoder.link is not None and group.spec.mode == "natural"
        return real_loss(group, records, *a, **k)

    def col_spy(group, records, *a, **k):
        out = real_col(group, records, *a, **k)
        tally[(corpus["name"], group.spec.task_type, group.spec.mode, "column_terms")] += out[1]
        return out

    R.compute_group_loss, R._column_loss = loss_spy, col_spy
    sampled, _ = D.load_records(cfg, args.config, args.docs, args.seed, split="train")
    bs = int(tr["batch_size"])
    for name in sorted(sampled):
        corpus["name"] = name
        chunks = chunk_records(sampled[name], tokenizer=model.processor.tokenizer, window_size=int(tr["max_len"]),
                               stride=int(tr["window_stride"]), show_progress=False)
        ds = ExtractorDataset(data=chunks, max_samples=-1, shuffle=False, seed=42, validate=False, negatives=None)
        for b in range(0, len(ds), bs):
            random.seed(args.seed + b)
            torch.manual_seed(args.seed + b)
            with torch.no_grad():
                model(coll([ds[i] for i in range(b, min(len(ds), b + bs))]))
    report(tally)


def report(tally: Counter) -> None:
    rows = sorted({k[:3] for k in tally})
    print(f"{'corpus':28s} {'task':18s} {'mode':11s} {'groups':>7s} {'w/gold':>7s} {'linked':>7s} {'col terms':>10s}")
    by_task = Counter()
    for c, t, m in rows:
        g = [tally[(c, t, m, x)] for x in ("groups", "with_gold", "linked", "column_terms")]
        print(f"{c:28s} {t:18s} {m:11s} {g[0]:7d} {g[1]:7d} {g[2]:7d} {g[3]:10d}")
        for x, v in zip(("groups", "with_gold", "linked", "column_terms"), g):
            by_task[(t, m, x)] += v
    print("\n[link] totals by task type:")
    for t, m in sorted({k[:2] for k in by_task}):
        print(f"  {t:18s} {m:11s} groups {by_task[(t, m, 'groups')]:5d} | linked {by_task[(t, m, 'linked')]:5d} | "
              f"column-loss terms {by_task[(t, m, 'column_terms')]:6d}")


if __name__ == "__main__":
    main()
