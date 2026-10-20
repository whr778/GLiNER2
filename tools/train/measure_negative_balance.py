"""How many POSITIVE vs NEGATIVE event queries does a config's training mix carry? Counted after the real injector.

eb22c's events collapse to 0.0 on one GPU at steps ~4-8k under BCE and focal, and removing either its 8 structural
flags or its 8 negative/calibration knobs does not stop it (2026-10-20). Two remaining suspects shift the balance
toward "no event": `negative_gold_full_roles` (gold types list every role, so unargued roles become all-negative
queries) and the coref_llm corpora (~31k records with no event gold that may still receive absent event types). The
dose-20 and full-roles choices were validated warm-started on one or a few corpora, never on this from-scratch mix.

This reads a seeded sample of the config's REAL train mix (same files, label map and corpus tags as train.py), runs it
through the real NegativeLabels injector via ExtractorDataset, and counts per record:
  event-type queries   gold (present) vs injected absent
  role queries         argued (gold value) vs unargued on a gold type vs on an absent type
  all-negative records records with no gold event that still carry absent event queries
Variants: --no-full-roles, --drop-corpora NAME... (e.g. the coref_llm corpora).

    uv run tools/train/measure_negative_balance.py --config tools/train/config/base/eb22c.yaml --records 4000
"""
from __future__ import annotations

import argparse
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
from gliner2.training.negatives import NegativeLabels  # noqa: E402
from gliner2.training.trainer import ExtractorDataset  # noqa: E402


def train_records(cfg: dict, cfg_path: str) -> list[dict]:
    """The config's train mix exactly as train.py resolves it, label-mapped and corpus-tagged."""
    d = cfg["data"]
    corpora, train_only = d["corpora"], set(d.get("train_only") or [])
    paths = T._dedupe_paths(T._split_files(corpora, "train", train_only) + T._event_split(d.get("event_files") or {}, "train"),
                            "train")
    fns, passthrough = T._category_fns(T.load_labels_cfg(cfg, cfg_path)), T.labels_passthrough(cfg)
    return T.read_transformed(paths, fns, passthrough)


def negatives(cfg: dict, cfg_path: str, full_roles: bool) -> NegativeLabels:
    tr = cfg["training"]
    pools = tr["negative_pools"]
    if pools == "auto":
        pools = T._auto_negative_pools(cfg, cfg_path, tempfile.mkdtemp())
    return NegativeLabels.load(pools, tr["negative_labels_per_dim"], seed=int(tr.get("negative_label_seed", 42)),
                               partial=cfg["data"].get("partial_annotation"), gold_full_roles=full_roles)


def count(items) -> Counter:
    c = Counter()
    for _, out in items:
        events = out.get("events") or []
        gold_types = {e.get("event_type") for e in events}
        absent = out.get("absent_events") or {}
        roles = out.get("event_roles") or {}
        argued = {(e.get("event_type"), a.get("role")) for e in events for a in e.get("arguments") or []}
        c["records"] += 1
        c["records_with_gold_event"] += bool(events)
        c["type_present"] += len(gold_types)
        c["type_absent"] += len(absent)
        c["all_negative_records"] += (not events) and bool(absent)
        for t in gold_types:
            offered = set(roles.get(t) or []) | {r for (tt, r) in argued if tt == t}
            c["role_argued"] += len({r for (tt, r) in argued if tt == t})
            c["role_unargued_gold_type"] += len(offered - {r for (tt, r) in argued if tt == t})
        c["role_on_absent_type"] += sum(len(v or []) for v in absent.values())
    return c


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--records", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-full-roles", action="store_true", help="force negative_gold_full_roles off")
    ap.add_argument("--drop-corpora", nargs="*", default=[], help="remove these corpora from the mix before sampling")
    ap.add_argument("--dose", nargs="*", default=[], metavar="DIM=N",
                    help="override negative_labels_per_dim, e.g. events=3 entities=3")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    for kv in args.dose:
        dim, n = kv.split("=")
        cfg["training"]["negative_labels_per_dim"] = dict(cfg["training"]["negative_labels_per_dim"], **{dim: int(n)})
    records = [r for r in train_records(cfg, args.config) if r["_corpus"] not in set(args.drop_corpora)]
    sample = random.Random(args.seed).sample(records, min(args.records, len(records)))
    print(f"[balance] {args.config}: negative_labels_per_dim {cfg['training']['negative_labels_per_dim']}", flush=True)
    full = bool(cfg["training"].get("negative_gold_full_roles")) and not args.no_full_roles
    neg = negatives(cfg, args.config, full)
    neg.set_epoch(0)
    ds = ExtractorDataset(data=sample, max_samples=-1, shuffle=False, seed=42, validate=False, negatives=neg)
    c = count(ds[i] for i in range(len(ds)))
    label = (f"{Path(args.config).stem}" + (" --no-full-roles" if args.no_full_roles else "")
             + (f" --drop {len(args.drop_corpora)} corpora" if args.drop_corpora else "")
             + (f" dose {' '.join(args.dose)}" if args.dose else ""))
    roles_pos = c["role_argued"]
    roles_neg = c["role_unargued_gold_type"] + c["role_on_absent_type"]
    print(f"| {label} | {len(records):,} | {c['records']:,} | {100 * c['records_with_gold_event'] / c['records']:.1f}% | "
          f"{c['type_present']:,} / {c['type_absent']:,} ({c['type_absent'] / max(c['type_present'], 1):.1f}x) | "
          f"{roles_pos:,} / {c['role_unargued_gold_type']:,} + {c['role_on_absent_type']:,} ({roles_neg / max(roles_pos, 1):.1f}x) | "
          f"{c['all_negative_records']:,} ({100 * c['all_negative_records'] / c['records']:.1f}%) | full_roles={full} |", flush=True)


if __name__ == "__main__":
    main()
