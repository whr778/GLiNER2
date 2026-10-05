"""Junction OWNERSHIP and DETECTION per corpus and split: is the trigger -> argument link overfitting?

Two questions, measured on the decode's own scores (`records.decode_group`, list fields):

  ownership  which trigger owns a gold argument? Per column of one role field: the gold trigger's
             score against the K hardest false triggers on the SAME argument (per-column AUC, the
             role term cancels), and top-1 = the gold trigger holds the column max over the
             instances decode SELECTS (anchor score >= the gate), when a gold owner is selected.
  detection  does the argument survive the gate? Decode keeps candidate j when the column max of
             sigmoid(assign logit) over instances clears the field threshold. Positive = j is a
             gold argument of some gold trigger. Pooled over columns ON PURPOSE: decode applies one
             threshold to these pooled scores. Reported as a DET curve (miss vs false-alarm rate,
             probit axes) with the operating point marked, plus EER and correct-at-the-gate (column
             max clears the gate AND its owner is the gold trigger).

Overfitting reads as TRAIN improving while VAL stalls: run both splits on each epoch checkpoint.
Forwards are training-shaped (`model.train()` with dropout OFF, no_grad) because only that path
builds record groups with gold; injection stays at the model's default 1.0, so every number is
CONDITIONAL on the gold trigger being an instance.

    uv run python tools/train/measure_junction_det.py --checkpoint whr778/gliner2-eb19-checkpoint-epoch-1 \\
        --out /Volumes/Development/tmp/junction_det/epoch1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/train"))
import measure_absent_dilution as D  # noqa: E402

import gliner2.models.boundary.records as R  # noqa: E402

FA_POINTS = (0.01, 0.05, 0.10, 0.20)


def has_args(rec) -> bool:
    return any(e.get("arguments") for e in rec["output"].get("events") or [])


def train_inputs(cfg) -> set:
    """Every training document's text, so a val doc that is also trained on can be refused."""
    data = cfg["data"]
    files = D.T._split_files(data.get("corpora") or [], "train", set()) + D.T._event_split(data.get("event_files") or {}, "train")
    seen = set()
    for f in set(files):
        if Path(f).is_file():
            seen.update(json.loads(line)["input"] for line in open(f, encoding="utf-8") if line.strip())
    return seen


def family(name: str) -> str:
    """Pair a corpus across splits: CMNEE/DuEE events train from `*_ner` and validate from `scaling_joint/`."""
    return name.split("/")[-1].removesuffix("_ner")


def collect(group, records, k_false, gate, out) -> None:
    """Append this natural group's ownership wins and detection (score, label) pairs to ``out``.

    Decode-faithful: only instances whose anchor score clears ``gate`` can own an argument
    (`decode_group` selects them the same way), so detection and top-1 use the selected rows."""
    aq = group.spec.anchor_query_id
    af = group.field_query_ids.index(aq)
    spans = [R._span_index(s) for s in group.field_spans]
    seed = {s[1]: i for i, s in enumerate(group.instance_seed) if s is not None and s[0] == af}
    gold = {}
    for rec in records:
        a = rec.field_for_query(aq)
        cols = R._resolve_value_cols(a.values[0], spans[af]) if a is not None and a.values else []
        if cols and (inst := seed.get(cols[0] - 1)) is not None:
            gold[inst] = rec
    if not gold:
        return
    false = R._negative_instances(group, records, aq, set(gold), k_false)
    selected = [i for i, p in enumerate(torch.sigmoid(group.object_logits.detach()).tolist()) if p >= gate]
    out["gold_selected"].extend(1.0 if inst in selected else 0.0 for inst in gold)
    for f, fs in enumerate(group.field_specs):
        if f == af or fs.cardinality.is_scalar:
            continue
        full = group.assign_logits[f].detach()[:, 1:]
        if full.numel() == 0:
            continue
        owners = defaultdict(set)
        for inst, rec in gold.items():
            cols, _ = R._field_target_cols(fs, rec, spans[f])
            for c in cols:
                if 0 <= c - 1 < full.shape[1]:
                    owners[c - 1].add(inst)
        if selected:
            col_max, arg = torch.sigmoid(full[selected]).max(dim=0)
            col_arg = [selected[a] for a in arg.tolist()]
        else:
            col_max, col_arg = full.new_zeros(full.shape[1]), [-1] * full.shape[1]
        all_max, all_arg = torch.sigmoid(full).max(dim=0)
        for j in range(full.shape[1]):
            out["det"].append((float(col_max[j]), j in owners, col_arg[j] in owners.get(j, ())))
            out["det_scorer"].append((float(all_max[j]), j in owners, int(all_arg[j]) in owners.get(j, ())))
        for j, insts in owners.items():
            if insts & set(selected):
                out["top1"].append(1.0 if col_arg[j] in insts else 0.0)
            for inst in insts:
                g = float(full[inst, j])
                for i in false:
                    s = float(full[i, j])
                    out["own"].append(1.0 if g > s else 0.5 if g == s else 0.0)


def det(points, op: float) -> dict:
    """DET summary: miss / false-alarm curve, EER, the rates at ``op``, and correct-at-the-gate.

    The curve is evaluated at EVERY positive score (miss only changes there), at 400 negative
    quantiles, and at ``op`` -- uniform subsampling of all scores starved the low-false-alarm end,
    where the few positives live, and left the operating-point dots off their own curves."""
    pos = sorted(s for s, y, _ in points if y)
    neg = sorted(s for s, y, _ in points if not y)
    if not pos or not neg:
        return {}
    miss = lambda t: bisect_left(pos, t) / len(pos)
    fa = lambda t: (len(neg) - bisect_left(neg, t)) / len(neg)
    thr = sorted(set(pos) | {neg[int(q * (len(neg) - 1) / 400)] for q in range(401)} | {op})
    curve = [(t, miss(t), fa(t)) for t in thr]
    eer = min(curve, key=lambda c: abs(c[1] - c[2]))
    at_fa = {str(p): min((c[1] for c in curve if c[2] <= p), default=1.0) for p in FA_POINTS}
    correct = sum(1 for s, y, own in points if y and s >= op and own) / len(pos)
    return {"n_pos": len(pos), "n_neg": len(neg), "eer": (eer[1] + eer[2]) / 2,
            "miss_at_op": miss(op), "fa_at_op": fa(op), "correct_at_op": correct,
            "miss_at_fa": at_fa, "curve": curve}


def pairs(results: dict) -> None:
    """TRAIN vs VAL per corpus family (pooled over files): the overfitting readout."""
    fams = defaultdict(dict)
    for (corpus, split), r in results.items():
        fams[r["family"]].setdefault(split, []).append(r)
    for fam, by in sorted(fams.items()):
        if not {"train", "val"} <= set(by):
            continue
        cell = lambda split, key: sum(r[key] for r in by[split]) / len(by[split])
        det_cell = lambda split, key, view="det": sum(r[view].get(key, float("nan")) for r in by[split]) / len(by[split])
        print(f"[pair] {fam:10s} TRAIN vs VAL | ownership AUC {cell('train', 'own_auc'):.3f} vs {cell('val', 'own_auc'):.3f} | "
              f"top-1 {cell('train', 'top1'):.3f} vs {cell('val', 'top1'):.3f} | "
              f"scorer EER {det_cell('train', 'eer', 'det_scorer'):.3f} vs {det_cell('val', 'eer', 'det_scorer'):.3f} | "
              f"correct@gate {det_cell('train', 'correct_at_op'):.3f} vs {det_cell('val', 'correct_at_op'):.3f}")


def plot(results: dict, op: float, path: Path) -> None:
    """DET panels: rows = scorer / decode view, columns = split; a line per corpus, dot = the gate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = NormalDist().inv_cdf
    clip = lambda p: min(max(p, 1e-4), 1 - 1e-4)
    ticks = [0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95]
    splits = sorted({s for _, s in results})
    views = (("det_scorer", "scorer (all instances)"), ("det", "decode (selected instances)"))
    fig, axes = plt.subplots(len(views), len(splits), figsize=(6 * len(splits), 5.5 * len(views)), squeeze=False)
    for row, (view, title) in enumerate(views):
        for ax, split in zip(axes[row], splits):
            panel(ax, results, view, split, z, clip)
            ax.set_xticks([z(t) for t in ticks], [f"{t:g}" for t in ticks])
            ax.set_yticks([z(t) for t in ticks], [f"{t:g}" for t in ticks])
            ax.set_xlabel("false-alarm rate (non-argument candidates kept)")
            ax.set_ylabel("miss rate (gold arguments dropped)")
            ax.set_title(f"{split}, {title}: dots = threshold {op}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def panel(ax, results, view, split, z, clip) -> None:
    for (corpus, s), r in sorted(results.items()):
        if s != split or not r.get(view):
            continue
        c = r[view]["curve"]
        line, = ax.plot([z(clip(x[2])) for x in c], [z(clip(x[1])) for x in c], label=corpus)
        ax.plot(z(clip(r[view]["fa_at_op"])), z(clip(r[view]["miss_at_op"])), "o", color=line.get_color())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="tools/train/config/base/eb19.yaml")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--docs", type=int, default=48, help="argument-bearing docs per corpus and split")
    ap.add_argument("--false-triggers", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=None, help="default: the checkpoint's inference_defaults")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="directory for det.json and det.png")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tr, head = cfg["training"], cfg["model"]["boundary_head"]
    from gliner2 import AutoExtractor
    from gliner2.training.chunking import chunk_records
    from gliner2.training.trainer import ExtractorCollator, ExtractorDataset

    model = AutoExtractor.from_pretrained(args.checkpoint, map_location="cpu")
    op = args.threshold if args.threshold is not None else (model.config.inference_defaults or {}).get("threshold", 0.5)
    coll = ExtractorCollator(model.processor, is_training=True, max_len=tr.get("max_len"), architecture="boundary",
                             max_gold_per_query=int(head.get("max_gold_per_query", 32)),
                             on_capacity_exceeded=tr.get("on_capacity_exceeded", "raise"), error_policy="skip",
                             event_records=True)
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.eval()
    current = {}
    real_loss = R.compute_group_loss

    def loss_spy(group, records, *a, **k):
        if group.spec.mode == "natural" and records:
            collect(group, records, args.false_triggers, op, current)
        return real_loss(group, records, *a, **k)

    R.compute_group_loss = loss_spy
    results = {}
    trained = train_inputs(cfg)
    for split in args.splits.split(","):
        sampled, _ = D.load_records(cfg, args.config, 400, args.seed, split=split)
        # Record-level hygiene, as the trainer's check_and_clean does: casie.val and
        # scaling_joint/casie.val are byte-identical, and a val doc seen in training is not held out.
        seen, dup, leak = set(), 0, 0
        for corpus in sorted(sampled):
            docs = []
            for r in sampled[corpus]:
                if not has_args(r) or len(docs) >= args.docs:
                    continue
                if r["input"] in seen:
                    dup += 1
                    continue
                if split != "train" and r["input"] in trained:
                    leak += 1
                    continue
                seen.add(r["input"])
                docs.append(r)
            if not docs:
                continue
            chunks = chunk_records(docs, tokenizer=model.processor.tokenizer, window_size=int(tr["max_len"]),
                                   stride=int(tr["window_stride"]), show_progress=False)
            ds = ExtractorDataset(data=chunks, max_samples=-1, shuffle=False, seed=42, validate=False, negatives=None)
            current.clear()
            current.update(own=[], top1=[], det=[], det_scorer=[], gold_selected=[])
            bs = int(tr["batch_size"])
            for b in range(0, len(ds), bs):
                random.seed(args.seed + b)
                torch.manual_seed(args.seed + b)
                with torch.no_grad():
                    model(coll([ds[i] for i in range(b, min(len(ds), b + bs))]))
            mean = lambda v: sum(v) / len(v) if v else float("nan")
            r = {"family": family(corpus), "docs": len(docs), "gold_selected": mean(current["gold_selected"]),
                 "own_auc": mean(current["own"]), "own_pairs": len(current["own"]),
                 "top1": mean(current["top1"]), "top1_cols": len(current["top1"]), "det": det(current["det"], op),
                 "det_scorer": det(current["det_scorer"], op)}
            results[(corpus, split)] = r
            d = r["det"] or {}
            print(f"[det] {split:5s} {corpus:28s} docs {len(docs):3d} | gold trigger selected {r['gold_selected']:.3f} | "
                  f"ownership AUC {r['own_auc']:.3f} "
                  f"({r['own_pairs']} pairs) top-1 {r['top1']:.3f} ({r['top1_cols']} cols) | "
                  f"scorer EER {r['det_scorer'].get('eer', float('nan')):.3f} | decode EER {d.get('eer', float('nan')):.3f} | at {op}: miss {d.get('miss_at_op', float('nan')):.3f} "
                  f"FA {d.get('fa_at_op', float('nan')):.4f} correct {d.get('correct_at_op', float('nan')):.3f} "
                  f"| pos {d.get('n_pos', 0)} neg {d.get('n_neg', 0)}", flush=True)
        print(f"[det] {split} hygiene: refused {dup} duplicate and {leak} trained-on documents", flush=True)
    pairs(results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "det.json").write_text(json.dumps({"checkpoint": args.checkpoint, "threshold": op,
                                              "results": {f"{c}|{s}": r for (c, s), r in results.items()}},
                                             ensure_ascii=False, indent=1), encoding="utf-8")
    plot(results, op, out / "det.png")
    print(f"[det] wrote {out / 'det.json'} and {out / 'det.png'}")


if __name__ == "__main__":
    main()
