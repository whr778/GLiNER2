"""How much gold never trains, because the model did not propose its anchor.

THE QUESTION. `compute_group_loss` in natural mode resolves each gold record's anchor
against the MODEL'S OWN candidate spans and skips the record on a miss -- no object loss,
no field loss, no gradient. Events compile as natural mode with the TRIGGER as the anchor
(`processing/records.py:70`), so a trigger the model did not propose discards that
instance's ENTIRE argument supervision. This measures the share.

IT IS A MEASUREMENT, NOT A VERDICT. EVENT_ARGUMENT_DIAGNOSIS 4f cleared three candidates
for the recall floor and concluded "undertraining, nothing structural capping it". This is
a fourth -- a supervision gate rather than a capacity cap -- and 4f did not examine it.
Whether it explains anything depends entirely on the number this prints. A few percent is a
footnote. Tens of percent is the recall story.

USE A FULLY-TRAINED CHECKPOINT. An undertrained model proposes fewer spans and will
overstate the miss rate, which is the direction that flatters the hypothesis.

    uv run python tools/train/probe_anchor_gate.py \
        --config tools/train/config/base/eb17-best.yaml \
        --checkpoint whr778/gliner2-eb17-best --max-batches 200
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
    from gliner2.models.boundary import records as R
    from gliner2.training.chunking import chunk_records
    from gliner2.training.trainer import ExtractorCollator, ExtractorDataset

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max-batches", type=int, default=200)
    ap.add_argument("--max-records", type=int, default=0,
                    help="subsample BEFORE chunking; 0 = every record. The probe only "
                         "consumes --max-batches x --batch-size samples, so loading and "
                         "chunking 201k records to score 800 of them is pure setup cost.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default=None, help="default: the model's own choice")
    ap.add_argument("--out", default=None, help="write the stats as JSON here too")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data, train_cfg = cfg.get("data") or {}, cfg.get("training") or {}

    files = T._dedupe_paths(
        T._split_files(data.get("corpora") or [], "train", set(data.get("train_only") or ()))
        + T._event_split(data.get("event_files") or {}, "train"), "train")
    fns = T._category_fns(T.load_labels_cfg(cfg, config_path=args.config))

    records = []
    for f in files:
        p = Path(f)
        if not p.is_file():
            continue
        for line in p.open(encoding="utf-8"):
            if line.strip():
                rec = T.transform_record(json.loads(line), fns)
                rec.setdefault("_corpus", p.name.split(".")[0])
                records.append(rec)
    print(f"[gate] {len(records):,} train records from {len(files)} files", flush=True)
    if args.max_records and len(records) > args.max_records:
        # SUBSAMPLE WITH A SEED, and shuffle first: the file order is by corpus, so a head
        # slice would measure one corpus and report it as the mix.
        import random
        random.Random(int(train_cfg.get("seed", 42))).shuffle(records)
        records = records[: args.max_records]
        print(f"[gate] subsampled -> {len(records):,} (seeded shuffle, not a head slice)",
              flush=True)

    model = AutoExtractor.from_pretrained(args.checkpoint, map_location=args.device)
    proc = model.processor

    if train_cfg.get("sliding_window"):
        records = chunk_records(records, tokenizer=proc.tokenizer,
                                window_size=int(train_cfg.get("max_len") or 512),
                                stride=int(train_cfg.get("window_stride") or 512),
                                show_progress=False)
        print(f"[gate] chunked -> {len(records):,}", flush=True)

    head = (cfg.get("model") or {}).get("boundary_head") or {}
    coll = ExtractorCollator(
        proc, is_training=True, max_len=train_cfg.get("max_len"), architecture="boundary",
        max_gold_per_query=int(head.get("max_gold_per_query", 32)),
        error_policy="skip", event_records=bool(head.get("event_records", False)),
    )
    ds = ExtractorDataset(data=records, max_samples=-1, shuffle=True,
                          seed=int(train_cfg.get("seed", 42)), validate=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                    collate_fn=coll)

    # TRAIN MODE is load-bearing: `compute_group_loss` only runs on the loss path, and the
    # anchor gate is inside it. no_grad keeps it cheap without changing which branch runs.
    model.train()
    R.reset_anchor_gate()
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= args.max_batches:
                break
            try:
                model(batch)
            except Exception as exc:                      # a bad batch must not end the probe
                print(f"[gate] batch {i} raised {type(exc).__name__}: {exc}", flush=True)
            if i and i % 50 == 0:
                print(f"[gate] {i} batches", flush=True)

    st = R.anchor_gate_stats()
    if not st["seen"]:
        print("\n[gate] *** THE GATE WAS NEVER REACHED. This run measured NOTHING -- "
              "check that the config compiles record groups (event_records / structures) "
              "and that the model ran in train mode. ***")
        return 1

    print(f"\n[gate] {st['seen']:,} gold records reached the anchor gate")
    for k, label in (("trained", "TRAINED"),
                     ("anchor_not_proposed", "anchor NOT PROPOSED by the model"),
                     ("anchor_not_seeded", "proposed, but not seeded as an instance"),
                     ("no_gold_anchor", "no gold anchor value at all")):
        print(f"        {label:44} {st[k + '_n']:>9,}  {st[k + '_share']:6.1%}")
    lost = st["anchor_not_proposed_share"] + st["anchor_not_seeded_share"]
    print(f"\n[gate] {lost:.1%} of gold records contributed NO supervision of any kind.")
    print("[gate] For events the anchor is the TRIGGER, so that share of gold event "
          "instances trained no arguments either.")
    if args.out:
        Path(args.out).write_text(json.dumps(st, indent=2), encoding="utf-8")
        print(f"[gate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
