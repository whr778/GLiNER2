"""The per-task negative:positive ratio the boundary BCE actually sees.

WHY. `pos_weight` in BCEWithLogitsLoss scales the POSITIVE term, and the textbook setting
is n_neg/n_pos per class. eb17 sets `struct_pos_weight: 4.0`, which reaches STRUCTURES
only; it sets no `task_pos_weights`, and `_query_task_table` returns None when every value
is 1.0 -- so `relation`, `event_argument`, `event_trigger` and `entity` all train at an
effective pos_weight of 1.0, with NO correction for imbalance. Those include the two worst
recall heads on the blind test (relation 0.0517, event_argument 0.0589).

WHY A FORWARD PASS AND NOT THE COLLATED BATCH. The first version of this script read
`targets.start_targets` and counted nothing, because on this architecture those are `None`:
the negatives are not text POSITIONS, they are CANDIDATE spans, and `PaddedTargetBatch`
says why they cannot be read off the batch -- "candidate columns do not exist until the
head runs". So this runs the real model on real batches and counts inside the loss, which
is the only place the ratio exists.

WHAT IT COUNTS. Both weighted terms, separately, because `query_pos_weights` is passed to
both: the `pair` term over candidate spans (`candidate_pair_loss`) and the `marginal` term
over start/end positions (`balanced_multilabel_bce`). It counts under each term's OWN
effective mask -- hard-negative restriction and the sampled `pair_query_mask` included --
so the number is what the BCE sees, not what the tensor could hold. The soft-IoU call is
skipped by testing its targets for non-binary values, since a fractional overlap has no
positive class to weight.

WHAT THE NUMBER IS NOT. A ratio is a STARTING POINT, not a setting. Raising pos_weight
buys recall by making false negatives dearer and it WILL cost precision; whether F1 moves
is empirical, and this programme has a standing lesson that recall bought by indiscriminate
firing is not a win. Read it beside the reliability curve: if the model is under-confident,
pos_weight is the training-time fix for what the threshold patches at decode; if it is well
calibrated, pos_weight is a genuine trade.

    uv run python tools/train/measure_pos_neg_ratio.py \
        --config tools/train/config/base/eb17-best.yaml \
        --checkpoint whr778/gliner2-eb17-best --max-records 3000 --max-batches 60
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    import torch
    import yaml
    from torch.utils.data import DataLoader

    import train as T
    from gliner2 import AutoExtractor
    from gliner2.models.boundary import model as BM
    from gliner2.models.boundary.losses import _to_query_candidate
    from gliner2.training.chunking import chunk_records
    from gliner2.training.trainer import ExtractorCollator, ExtractorDataset

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max-records", type=int, default=3000,
                    help="subsample BEFORE chunking; the loop only consumes "
                         "--max-batches x --batch-size samples.")
    ap.add_argument("--max-batches", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cpu",
                    help="cpu by default: this counts integers, and MPS int64 gather is "
                         "known to corrupt on this repo.")
    ap.add_argument("--events-only", action="store_true",
                    help="load ONLY the event files. The record head's list-field term is "
                         "where event arguments train, and a random draw over the whole "
                         "corpus barely reaches it -- 5 batches hit it zero times.")
    ap.add_argument("--out", default=None, help="write the table as JSON here too")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data, tc = cfg.get("data") or {}, cfg.get("training") or {}

    event_files = T._event_split(data.get("event_files") or {}, "train")
    files = (T._dedupe_paths(event_files, "train") if args.events_only
             else T._dedupe_paths((data.get("train_files") or []) + event_files, "train"))
    print(f"[ratio] {len(files)} source file(s)"
          f"{' (EVENTS ONLY)' if args.events_only else ''}", flush=True)
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
    print(f"[ratio] {len(records):,} train records (seeded subsample)", flush=True)

    model = AutoExtractor.from_pretrained(args.checkpoint, map_location="cpu")
    # train() and not eval(): the question is what the loss sees DURING TRAINING, and
    # the candidate set is produced by the proposal head. Gold counts do not depend on
    # dropout; the candidate count barely does, but the faithful path is the real one.
    model = model.to(args.device).train()
    proc = model.processor
    if tc.get("sliding_window"):
        records = chunk_records(records, tokenizer=proc.tokenizer,
                                window_size=int(tc.get("max_len") or 512),
                                stride=int(tc.get("window_stride") or 512),
                                show_progress=False)
        print(f"[ratio] chunked -> {len(records):,}", flush=True)

    head = (cfg.get("model") or {}).get("boundary_head") or {}
    coll = ExtractorCollator(
        proc, is_training=True, max_len=tc.get("max_len"), architecture="boundary",
        max_gold_per_query=int(head.get("max_gold_per_query", 32)),
        error_policy="skip", event_records=bool(head.get("event_records", False)),
    )
    dl = DataLoader(ExtractorDataset(data=records, max_samples=-1, shuffle=True,
                                     seed=int(tc.get("seed", 42)), validate=False),
                    batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=coll)

    # --- the counter: intercept the two terms that carry a positive class -------------
    seen: list[tuple] = []          # (term, positive[B,Q,*], effective[B,Q,*])
    fired: dict = defaultdict(int)  # THE GATE: a term that never fires counts nothing,
                                    # and a silent zero would read as "balanced".

    def _binary(t):
        return bool(((t == 0) | (t == 1)).all())

    real_pair, real_marginal = BM.candidate_pair_loss, BM.balanced_multilabel_bce

    def pair_spy(logits, labels, valid_mask, hard_negative_mask=None, **kw):
        # Soft-IoU reuses this function with FRACTIONAL targets; there is no positive
        # class to weight there, so it is excluded rather than counted as noise.
        if _binary(labels):
            qa, ca = kw.get("query_axis", 1), kw.get("candidate_axis", 2)
            lab = _to_query_candidate(labels, qa, ca)
            val = _to_query_candidate(valid_mask, qa, ca)
            if hard_negative_mask is not None:
                hard = _to_query_candidate(hard_negative_mask, qa, ca)
                eff = val & ((lab > 0.5) | hard)
            else:
                eff = val
            qm = kw.get("query_mask")
            if qm is not None:
                eff = eff & qm.unsqueeze(-1)
            fired["pair"] += 1
            seen.append(("pair", lab > 0.5, eff))
        return real_pair(logits, labels, valid_mask,
                         hard_negative_mask=hard_negative_mask, **kw)

    def marginal_spy(logits, targets, valid_mask, **kw):
        if _binary(targets):
            eff = valid_mask
            qm = kw.get("query_mask")
            if qm is not None:
                eff = eff & qm.unsqueeze(-1)
            fired["marginal"] += 1
            seen.append(("marginal", targets > 0.5, eff))
        return real_marginal(logits, targets, valid_mask, **kw)

    # The RECORD head is a separate accounting. Its list-field term is a bare
    # F.binary_cross_entropy_with_logits inside records.py, so neither spy above sees it --
    # and that term is where EVENT ARGUMENTS train (they compile as list fields). Wrap the
    # functional and keep only calls whose caller frame is records.py, reading the group's
    # own task_type out of the frame locals rather than guessing it.
    import torch.nn.functional as F
    real_bcel = F.binary_cross_entropy_with_logits
    rec_pos: dict = defaultdict(int)
    rec_neg: dict = defaultdict(int)
    rec_calls: dict = defaultdict(int)

    def bcel_spy(input, target, weight=None, size_average=None, reduce=None,
                 reduction="mean", pos_weight=None):
        frame = sys._getframe(1)
        if frame.f_code.co_filename.endswith("records.py") and target.dtype.is_floating_point:
            group, walk = None, frame
            for _ in range(6):        # the BCE is several frames below the group
                if walk is None:
                    break
                group = walk.f_locals.get("group") or walk.f_locals.get("output")
                if group is not None:
                    break
                walk = walk.f_back
            task = getattr(getattr(group, "spec", None), "task_type", "?")
            site = f"records.py:{frame.f_lineno} [{task}]"
            rec_calls[site] += 1
            rec_pos[site] += int((target > 0.5).sum())
            rec_neg[site] += int((target <= 0.5).sum())
        return real_bcel(input, target, weight, size_average, reduce, reduction, pos_weight)

    F.binary_cross_entropy_with_logits = bcel_spy
    BM.candidate_pair_loss, BM.balanced_multilabel_bce = pair_spy, marginal_spy

    pos = defaultdict(int)
    neg = defaultdict(int)
    queries = defaultdict(int)
    batches = 0
    try:
        for i, batch in enumerate(dl):
            if i >= args.max_batches:
                break
            seen.clear()
            with torch.no_grad():
                model(batch.to(args.device) if hasattr(batch, "to") else batch)
            layouts = getattr(batch, "query_layouts", None)
            if not seen or layouts is None:
                continue
            batches += 1
            # [B, Q] task name per query, built the way _query_task_ids builds it:
            # layout.queries holds the QuerySpecs, and spec.query_id indexes the Q axis.
            width = max(t.shape[1] for _, t, _ in seen)
            names = [[None] * width for _ in layouts]
            for b, layout in enumerate(layouts):
                for spec in layout.queries:
                    if spec.query_id < width:
                        names[b][spec.query_id] = spec.task_type
                        queries[spec.task_type] += 1
            for term, positive, eff in seen:
                p_eff = (positive & eff).sum(-1).cpu()
                n_eff = ((~positive) & eff).sum(-1).cpu()
                for b in range(p_eff.shape[0]):
                    for q in range(min(p_eff.shape[1], width)):
                        t = names[b][q]
                        if t is None:
                            continue
                        pos[(t, term)] += int(p_eff[b, q])
                        neg[(t, term)] += int(n_eff[b, q])
            if batches % 20 == 0:
                print(f"[ratio] {batches} batches", flush=True)
    finally:
        BM.candidate_pair_loss, BM.balanced_multilabel_bce = real_pair, real_marginal
        F.binary_cross_entropy_with_logits = real_bcel

    if not pos and not neg:
        print("\n[ratio] *** NOTHING COUNTED -- the loss terms never fired. "
              "The ratio is not zero, it is unmeasured. ***")
        return 1

    print(f"\n[ratio] counted over {batches} batches; loss-term calls intercepted: "
          f"{dict(fired)}")
    for term in ("pair", "marginal"):
        if not fired[term]:
            print(f"[ratio] WARNING: the {term!r} term NEVER fired -- it is unmeasured, "
                  "not balanced.")
    print()
    print(f"  {'task_type':20} {'term':10} {'queries':>9} {'pos':>10} {'neg':>13} {'neg/pos':>10}")
    rows = []
    for key in sorted(set(pos) | set(neg), key=lambda k: (k[0], k[1])):
        t, term = key
        p, n = pos[key], neg[key]
        ratio = (n / p) if p else None
        rows.append({"task_type": t, "term": term, "queries": queries[t],
                     "pos": p, "neg": n,
                     "neg_per_pos": None if ratio is None else round(ratio, 1)})
        print(f"  {t:20} {term:10} {queries[t]:>9,} {p:>10,} {n:>13,} "
              f"{'n/a' if ratio is None else f'{ratio:>10.1f}'}")

    print("\n[ratio] === RECORD HEAD (separate accounting; no pos_weight exists here) ===")
    if not rec_pos:
        print("[ratio] the record head never fired -- unmeasured, not balanced.")
    else:
        print(f"  {'call site [task]':44} {'calls':>7} {'pos':>10} {'neg':>13} {'neg/pos':>10}")
        for site in sorted(rec_pos, key=lambda k: -rec_neg[k]):
            pr, nr = rec_pos[site], rec_neg[site]
            rr = (nr / pr) if pr else None
            rows.append({"call_site": site, "calls": rec_calls[site], "pos": pr, "neg": nr,
                         "neg_per_pos": None if rr is None else round(rr, 1)})
            print(f"  {site:44} {rec_calls[site]:>7,} {pr:>10,} {nr:>13,} "
                  f"{'n/a' if rr is None else f'{rr:>10.1f}'}")

    print("\n[ratio] n_neg/n_pos is the TEXTBOOK pos_weight. eb17 trains every one of these")
    print("[ratio] at an effective 1.0 -- ALL of them. `struct_pos_weight: 4.0` in the config")
    print("[ratio] is DEAD on the boundary architecture: its only reader is _struct_loss_term")
    print("[ratio] in models/span/model.py, a method BoundaryExtractor does not inherit.")
    print("[ratio] It is a starting point, not a setting: raising it buys recall and SPENDS")
    print("[ratio] precision, and whether F1 moves is empirical.")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"[ratio] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
