"""Reproduce a record-spec raise on the REAL data path, with no GPU and no training.

WHY THIS EXISTS. `eb17-best` and `eb17-smoke` both died in a DataLoader worker with
``record 'record' declares anchor 'type' but no matching field query was found in the
layout``. Six local traces failed to reproduce it -- on-disk data (0 of 202,211 records
have an anchor missing from its group), schema sampling (0 of 155,080 draws),
`_transform_record` (0 of 62,032), two full-mixture collates of 203,736 chunks, and a
full mixture through four real worker processes (50,934 batches).

The one condition left is FORK. Linux workers inherit the parent's live memory -- the
processor's caches, the `random` state at fork time -- where macOS spawns and unpickles a
fresh copy. `trainer.py` pins Linux to `fork` deliberately, so the failing path exists
ONLY on a Linux box and the evidence has to be collected there.

This runs the same chunking, label map, negatives injector, collator and DataLoader the
trainer builds, and NOTHING ELSE: no model, no forward pass, no optimiser. A crash that
takes ~1.7h of training to reach is reached here in minutes on the cheapest card, because
collation is all that is being exercised.

    uv run python tools/train/trace_anchor_raise.py --config tools/train/config/base/eb17-smoke.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    import yaml
    from torch.utils.data import DataLoader

    import train as T
    from gliner2 import AutoExtractor
    from gliner2.training.chunking import chunk_records
    from gliner2.training.negatives import NegativeLabels
    from gliner2.training.trainer import ExtractorCollator, ExtractorDataset

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="only its processor is used; defaults to the config's model_name")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2,
                    help="passes over the data; the crash is order-dependent")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data = cfg.get("data") or {}
    train_cfg = cfg.get("training") or {}

    files = T._dedupe_paths(
        T._split_files(data.get("corpora") or [], "train",
                       set(data.get("train_only") or ()))
        + T._event_split(data.get("event_files") or {}, "train"),
        "train")
    fns = T._category_fns(T.load_labels_cfg(cfg, config_path=args.config))

    records = []
    for f in files:
        p = Path(f)
        if not p.is_file():
            print(f"[trace] MISSING {f}", flush=True)
            continue
        for line in p.open(encoding="utf-8"):
            if line.strip():
                rec = T.transform_record(json.loads(line), fns)
                rec.setdefault("_corpus", p.name.split(".")[0])
                records.append(rec)
    print(f"[trace] loaded {len(records):,} records from {len(files)} files", flush=True)

    ckpt = args.checkpoint or train_cfg.get("model_name") or cfg.get("model_name")
    proc = AutoExtractor.from_pretrained(ckpt, map_location="cpu").processor

    if train_cfg.get("sliding_window"):
        records = chunk_records(records, tokenizer=proc.tokenizer,
                                window_size=int(train_cfg.get("max_len") or 512),
                                stride=int(train_cfg.get("window_stride") or 512),
                                show_progress=False)
        print(f"[trace] chunked -> {len(records):,}", flush=True)

    negatives = None
    pools = train_cfg.get("negative_pools")
    if pools and train_cfg.get("negative_labels_per_dim"):
        if str(pools).strip().lower() == "auto":
            pools = T._auto_negative_pools(cfg, args.config, train_cfg.get("output_dir", "./out/trace"))
        negatives = NegativeLabels.load(
            pools, train_cfg["negative_labels_per_dim"],
            seed=int(train_cfg.get("negative_label_seed", 42)),
            partial=data.get("partial_annotation"))
        print(f"[trace] negatives ON: {train_cfg['negative_labels_per_dim']}", flush=True)

    ds = ExtractorDataset(data=records, max_samples=-1, shuffle=False,
                          seed=int(train_cfg.get("seed", 42)),
                          validate=bool(train_cfg.get("validate_data")),
                          negatives=negatives)
    head = (cfg.get("model") or {}).get("boundary_head") or {}
    coll = ExtractorCollator(
        proc, is_training=True,
        max_len=train_cfg.get("max_len"),
        architecture="boundary",
        max_gold_per_query=int(train_cfg.get("max_gold_per_query", 32)),
        on_capacity_exceeded=train_cfg.get("on_capacity_exceeded", "raise"),
        error_policy=train_cfg.get("error_policy", "raise"),
        event_records=bool(head.get("event_records", False)),
    )
    print(f"[trace] workers={args.num_workers} batch={args.batch_size} "
          f"event_records={coll.event_records} fork={sys.platform.startswith('linux')}",
          flush=True)

    for epoch in range(args.epochs):
        if negatives is not None:
            negatives.set_epoch(epoch)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=coll,
                        persistent_workers=args.num_workers > 0,
                        multiprocessing_context="fork" if args.num_workers > 0
                        and sys.platform.startswith("linux") else None)
        try:
            for i, _ in enumerate(dl):
                if i % 2000 == 0:
                    print(f"[trace] epoch {epoch} batch {i:,}", flush=True)
        except Exception:
            print(f"\n[trace] *** RAISED in epoch {epoch} ***\n", flush=True)
            traceback.print_exc()
            print("\n[trace] the [diag] lines above name the sample. Done.", flush=True)
            return 1
        print(f"[trace] epoch {epoch} completed clean", flush=True)

    print("[trace] NO RAISE over every epoch -- not reproduced here either", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
