"""Does the UNTRAINED object head already separate good instances from spurious ones?

THE SETUP. `decode_group` selects event instances by `sigmoid(object_logits)` thresholded
at `record_anchor_threshold` (records.py:731-736) -- so that head IS the existence gate at
decode. But in natural mode it receives NO gradient: `compute_group_loss` returns
`object_loss = zero` (records.py:1453), and the Hungarian branch that trains it is
anchorless-only. We threshold an untrained score.

THE QUESTION. Lowering that threshold to 0.1 on the blind test recovered 2,293 gold
triggers and emitted 8,658 spurious ones, 3.78 spurious per gold recovered. If the
existing object score already ranks the good above the spurious, training it is promising.
If it is at chance, the head is starting from nothing and this is the head-init question.

WHAT IT REPORTS. AUC of `sigmoid(object_logit)` for separating decoded instances whose
anchor span matches a GOLD anchor from those that match none. AUC 0.5 is chance. Gold
anchors are resolved exactly as `compute_group_loss` resolves them, against the model's
own candidate spans.

    uv run python tools/train/probe_object_score_separation.py \
        --config tools/train/config/base/eb17-best.yaml \
        --checkpoint whr778/gliner2-eb17-best --max-records 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _auc(pos, neg):
    """Rank-based AUC; ties get half credit. No sklearn dependency."""
    if not pos or not neg:
        return None
    merged = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank, i, total = {}, 0, 0.0
    ranks = [0.0] * len(merged)
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rsum = sum(r for r, (_, lab) in zip(ranks, merged) if lab == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (rsum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


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
    ap.add_argument("--split", default="val", choices=("train", "val"))
    ap.add_argument("--max-records", type=int, default=400)
    ap.add_argument("--max-batches", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--anchor-threshold", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data, tc = cfg.get("data") or {}, cfg.get("training") or {}
    files = T._dedupe_paths(T._event_split(data.get("event_files") or {}, args.split),
                            args.split)
    fns = T._category_fns(T.load_labels_cfg(cfg, config_path=args.config))
    records = []
    for f in files:
        p = Path(f)
        if not p.is_file():
            continue
        for line in p.open(encoding="utf-8"):
            if line.strip():
                records.append(T.transform_record(json.loads(line), fns))
    import random
    random.Random(int(tc.get("seed", 42))).shuffle(records)
    records = records[: args.max_records]
    print(f"[obj] {len(records):,} {args.split} event records from {len(files)} file(s)",
          flush=True)

    # train() not eval(): the record loss path (and so compute_group_loss, which is
    # where the gold anchors are available) only runs in training mode. Decode itself is
    # unaffected -- we call decode_group directly on the group.
    model = AutoExtractor.from_pretrained(args.checkpoint, map_location="cpu").train()
    proc = model.processor
    if tc.get("sliding_window"):
        records = chunk_records(records, tokenizer=proc.tokenizer,
                                window_size=int(tc.get("max_len") or 512),
                                stride=int(tc.get("window_stride") or 512),
                                show_progress=False)
    head = (cfg.get("model") or {}).get("boundary_head") or {}
    coll = ExtractorCollator(proc, is_training=True, max_len=tc.get("max_len"),
                             architecture="boundary",
                             max_gold_per_query=int(head.get("max_gold_per_query", 32)),
                             error_policy="skip",
                             event_records=bool(head.get("event_records", False)))
    dl = DataLoader(ExtractorDataset(data=records, max_samples=-1, shuffle=False,
                                     seed=int(tc.get("seed", 42)), validate=False),
                    batch_size=args.batch_size, shuffle=False, num_workers=0,
                    collate_fn=coll)

    good, spurious, groups_seen = [], [], 0
    real_loss = R.compute_group_loss

    def loss_spy(group, recs):
        """compute_group_loss is called with the group AND its gold; decode there."""
        nonlocal groups_seen
        if group.spec.task_type == "events" and group.spec.mode == "natural":
            groups_seen += 1
            afi = group.field_query_ids.index(group.spec.anchor_query_id)
            span_idx = R._span_index(group.field_spans[afi])
            gold_cols = set()
            for rec in recs:
                aft = rec.field_for_query(group.spec.anchor_query_id)
                if aft is None or not aft.values:
                    continue
                for v in aft.values:
                    for col in R._resolve_value_cols(v, span_idx):
                        gold_cols.add(col - 1)          # candidate column index
            # field_spans is a LIST of LongTensor [Cf, 2]; decode_group sets anchor_span
            # to a tuple of python ints. Comparing a tuple against a set of TENSORS is
            # never true, which is how this first reported 0 good out of 92.
            gold_spans = {
                (int(group.field_spans[afi][c][0]), int(group.field_spans[afi][c][1]))
                for c in gold_cols if 0 <= c < int(group.field_spans[afi].shape[0])
            }
            decoded = R.decode_group(group, anchor_threshold=args.anchor_threshold,
                                     field_threshold=0.5,
                                     object_threshold=args.anchor_threshold)
            for d in decoded:
                (good if d.anchor_span in gold_spans else spurious).append(float(d.score))
        return real_loss(group, recs)

    R.compute_group_loss = loss_spy
    try:
        for i, batch in enumerate(dl):
            if i >= args.max_batches:
                break
            with torch.no_grad():
                model(batch)
            if i and i % 25 == 0:
                print(f"[obj] {i} batches  good={len(good)} spurious={len(spurious)}",
                      flush=True)
    finally:
        R.compute_group_loss = real_loss

    print(f"\n[obj] event groups decoded: {groups_seen}")
    if not good or not spurious:
        print("[obj] *** need BOTH classes to score separation; "
              f"good={len(good)} spurious={len(spurious)}. UNMEASURED, not chance. ***")
        return 1

    auc = _auc(good, spurious)
    mg, ms = sum(good) / len(good), sum(spurious) / len(spurious)
    print(f"[obj] instances matching a GOLD anchor : {len(good):>7,}  mean score {mg:.4f}")
    print(f"[obj] instances matching NO gold anchor: {len(spurious):>7,}  mean score {ms:.4f}")
    print(f"[obj] ratio spurious:good = {len(spurious)/len(good):.2f} : 1")
    print(f"\n[obj] AUC of sigmoid(object_logit) = {auc:.4f}")
    print("[obj] 0.5 is chance. Well above -> the untrained head already ranks good above")
    print("[obj] spurious and training it is promising. At chance -> head-init question.")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"auc": auc, "n_good": len(good), "n_spurious": len(spurious),
             "mean_good": mg, "mean_spurious": ms,
             "anchor_threshold": args.anchor_threshold}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
