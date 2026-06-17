"""Greedy multi-label stratified split shared across event-corpus converters.

The algorithm (described in detail in ``convert_ace2005.py``) places each
record into ``train`` / ``test`` / ``val`` so that every category seen in
the corpus appears in train first, then test, then val, then fills up
toward the requested ratios. Categories are sourced from a record's
``output.entities`` keys, ``output.relations[*]`` keys, and
``output.events[*].event_type`` values.

Public helpers:

* :func:`stratified_split` — the algorithm itself.
* :func:`per_type_targets` — per-type target counts under rule (b).
* :func:`derive_split_paths` — turn a ``--out`` base into the three
  ``<base>.{train,test,val}.jsonl`` paths.
* :func:`parse_ratios` — argparse-friendly ratio parser.
* :func:`record_categories` — the ``ent:`` / ``rel:`` / ``evt:`` prefixed
  category set used for stratification (exported so converters can
  share the same definition for coverage reports).
* :func:`coverage_summary` — human-readable per-split category coverage.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


SPLIT_NAMES: Tuple[str, str, str] = ("train", "test", "val")


def record_categories(record: Dict[str, Any]) -> Set[str]:
    """Return the prefixed category strings the record's ``output`` contains.

    Format: ``ent:<entity_type>``, ``rel:<relation_type>``,
    ``evt:<event_type>``. The prefixes keep entity / relation / event
    namespaces apart in the stratification bookkeeping even if a label
    string is reused across task types.
    """
    cats: Set[str] = set()
    out = record.get("output") or {}
    for et in (out.get("entities") or {}).keys():
        cats.add(f"ent:{et}")
    for rel in out.get("relations") or []:
        if isinstance(rel, dict):
            for rt in rel.keys():
                cats.add(f"rel:{rt}")
    for ev in out.get("events") or []:
        et = ev.get("event_type") if isinstance(ev, dict) else None
        if isinstance(et, str):
            cats.add(f"evt:{et}")
    return cats


def per_type_targets(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Return ``(train, test, val)`` targets summing to ``n`` under rule (b).

    * ``n=1`` -> ``(1, 0, 0)``
    * ``n=2`` -> ``(1, 1, 0)``
    * ``n=3`` -> ``(1, 1, 1)``
    * ``n>=4`` -> rounded ratios, each bucket >= 1.
    """
    if n <= 0:
        return (0, 0, 0)
    if n == 1:
        return (1, 0, 0)
    if n == 2:
        return (1, 1, 0)
    if n == 3:
        return (1, 1, 1)

    train_t = max(1, round(n * ratios[0]))
    test_t = max(1, round(n * ratios[1]))
    val_t = max(1, n - train_t - test_t)
    total = train_t + test_t + val_t
    if total != n:
        train_t = max(1, n - test_t - val_t)
    if train_t + test_t + val_t != n:
        train_t = n - test_t - val_t
    return (train_t, test_t, val_t)


def stratified_split(
    records: List[Dict[str, Any]],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Greedy multi-label stratification.

    Returns three lists in ``(train, test, val)`` order whose total length
    equals ``len(records)``.
    """
    rng = random.Random(seed)
    n_records = len(records)
    if n_records == 0:
        return [], [], []

    cats_per_record: List[Set[str]] = [record_categories(r) for r in records]

    total_counts: Counter = Counter()
    for cs in cats_per_record:
        for c in cs:
            total_counts[c] += 1
    targets: Dict[str, Tuple[int, int, int]] = {
        c: per_type_targets(n, ratios) for c, n in total_counts.items()
    }

    remaining: Counter = Counter(total_counts)
    placed_per_split: Dict[str, List[int]] = {c: [0, 0, 0] for c in total_counts}

    type_queue: Dict[str, List[int]] = defaultdict(list)
    for i, cs in enumerate(cats_per_record):
        for c in cs:
            type_queue[c].append(i)
    for c in type_queue:
        rng.shuffle(type_queue[c])

    unplaced: Set[int] = set(range(n_records))
    splits: List[List[Dict[str, Any]]] = [[], [], []]

    while unplaced:
        candidates = [
            c for c in total_counts
            if remaining[c] > 0 and any(idx in unplaced for idx in type_queue[c])
        ]
        if not candidates:
            # Records that carry no categories — distribute round-robin to
            # whichever split is most under-filled overall.
            for idx in sorted(unplaced):
                target_idx = min(range(3), key=lambda k: len(splits[k]))
                splits[target_idx].append(records[idx])
            break

        rare = min(candidates, key=lambda c: (remaining[c], total_counts[c], c))

        while type_queue[rare] and type_queue[rare][0] not in unplaced:
            type_queue[rare].pop(0)
        if not type_queue[rare]:
            continue
        rec_idx = type_queue[rare][0]

        placed_total = sum(placed_per_split[rare])
        if placed_total == 0:
            split_idx = 0
        elif placed_total == 1:
            split_idx = 1
        elif placed_total == 2:
            split_idx = 2
        else:
            tgt = targets[rare]
            cur = placed_per_split[rare]
            gaps = [tgt[k] - cur[k] for k in range(3)]
            split_idx = max(range(3), key=lambda k: (gaps[k], -k))
            if gaps[split_idx] <= 0:
                split_idx = min(range(3), key=lambda k: (len(splits[k]), k))

        splits[split_idx].append(records[rec_idx])
        unplaced.discard(rec_idx)
        for c in cats_per_record[rec_idx]:
            placed_per_split[c][split_idx] += 1
            remaining[c] = max(0, remaining[c] - 1)

    return splits[0], splits[1], splits[2]


def parse_ratios(spec: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--split-ratios needs 3 comma-separated values, got {spec!r}"
        )
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--split-ratios must be numeric, got {spec!r}"
        ) from e
    if any(v < 0 for v in vals):
        raise argparse.ArgumentTypeError(
            f"--split-ratios cannot contain negative values, got {vals}"
        )
    total = sum(vals)
    if abs(total - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError(
            f"--split-ratios must sum to 1.0, got {total:.4f}"
        )
    return vals  # type: ignore[return-value]


def derive_split_paths(base: Path) -> Dict[str, Path]:
    """Return ``{"train": ..., "test": ..., "val": ...}`` sibling paths."""
    if base.suffix == ".jsonl":
        stem = base.with_suffix("")
    else:
        stem = base
    return {s: Path(f"{stem}.{s}.jsonl") for s in SPLIT_NAMES}


def coverage_summary(
    records: List[Dict[str, Any]],
    train: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    val: List[Dict[str, Any]],
) -> str:
    """Format a one-line "types coverage train=N/N test=N/N val=N/N" string."""
    cats_all: Counter = Counter()
    for r in records:
        for c in record_categories(r):
            cats_all[c] += 1

    def _cov(slice_records: List[Dict[str, Any]]) -> Set[str]:
        out: Set[str] = set()
        for r in slice_records:
            out |= record_categories(r)
        return out

    cov_train, cov_test, cov_val = _cov(train), _cov(test), _cov(val)
    return (
        f"types coverage train={len(cov_train)}/{len(cats_all)} "
        f"test={len(cov_test)}/{len(cats_all)} val={len(cov_val)}/{len(cats_all)}"
    )
