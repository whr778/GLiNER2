"""Helpers for partitioning GLiNER2 JSONL output into train/val/test splits.

Each converter writes three sibling files based on the user-supplied ``--out``
base path. If the user passes ``--out data/foo.jsonl`` the writer produces:

    data/foo.train.jsonl
    data/foo.val.jsonl
    data/foo.test.jsonl

Split assignment is deterministic: a seeded RNG draws one ``random()`` per
written record and routes it according to the cumulative ratio. Running the
same converter twice with the same seed produces the same partition.

Usage::

    from _split import SplitWriter

    with SplitWriter(args.out, ratios=(0.8, 0.1, 0.1), seed=42) as writer:
        for record in records:
            writer.write(record)
    print(writer.summary())     # "train=8123 val=1014 test=1003"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import IO, Dict, List, Sequence, Tuple


SPLIT_NAMES = ("train", "val", "test")


def parse_ratios(spec: str) -> Tuple[float, float, float]:
    """Parse a 'train,val,test' string and validate it sums to ~1.0."""
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


def add_split_args(parser: argparse.ArgumentParser) -> None:
    """Attach the standard --split-ratios / --split-seed flags to a parser."""
    parser.add_argument(
        "--split-ratios", type=parse_ratios, default=(0.8, 0.1, 0.1),
        help="Comma-separated train,val,test ratios (default: 0.8,0.1,0.1).",
    )
    parser.add_argument(
        "--split-seed", type=int, default=42,
        help="Random seed for the train/val/test partition (default: 42).",
    )


def derive_split_paths(base: Path) -> Dict[str, Path]:
    """Return {split: path} for the three sibling files.

    If ``base`` ends in ``.jsonl`` the suffix is stripped before appending
    the per-split suffix; otherwise ``base`` is used as-is.
    """
    if base.suffix == ".jsonl":
        stem = base.with_suffix("")
    else:
        stem = base
    return {s: Path(f"{stem}.{s}.jsonl") for s in SPLIT_NAMES}


class SplitWriter:
    """JSONL writer that routes each record into train/val/test deterministically.

    Args:
        base: Output base path (``data/foo.jsonl`` or ``data/foo``).
        ratios: Three-tuple summing to 1.0. Default ``(0.8, 0.1, 0.1)``.
        seed: Seed for the per-record routing RNG.
    """

    def __init__(
        self,
        base: Path,
        ratios: Sequence[float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ) -> None:
        if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(
                f"ratios must be a 3-tuple summing to 1.0, got {ratios!r}"
            )
        self._paths: Dict[str, Path] = derive_split_paths(base)
        self._files: Dict[str, IO[str]] = {}
        self._counts: Dict[str, int] = {s: 0 for s in SPLIT_NAMES}
        # Cumulative thresholds, e.g. (0.8, 0.9, 1.0).
        self._cum = []
        acc = 0.0
        for r in ratios:
            acc += r
            self._cum.append(acc)
        self._rng = random.Random(seed)

    def __enter__(self) -> "SplitWriter":
        for name, path in self._paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            self._files[name] = path.open("w")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for fh in self._files.values():
            fh.close()
        self._files = {}

    def _route(self) -> str:
        x = self._rng.random()
        for i, threshold in enumerate(self._cum):
            if x < threshold:
                return SPLIT_NAMES[i]
        return SPLIT_NAMES[-1]

    def write(self, record: dict) -> str:
        """Write ``record`` to the chosen split and return the split name."""
        split = self._route()
        fh = self._files[split]
        fh.write(json.dumps(record) + "\n")
        self._counts[split] += 1
        return split

    @property
    def counts(self) -> Dict[str, int]:
        return dict(self._counts)

    @property
    def paths(self) -> Dict[str, Path]:
        return dict(self._paths)

    @property
    def total(self) -> int:
        return sum(self._counts.values())

    def summary(self) -> str:
        c = self._counts
        return (
            f"train={c['train']} val={c['val']} test={c['test']} "
            f"-> {self._paths['train']}, {self._paths['val']}, {self._paths['test']}"
        )


if __name__ == "__main__":
    print("This module is a helper for the converters under tools/data/.",
          file=sys.stderr)
    raise SystemExit(1)
