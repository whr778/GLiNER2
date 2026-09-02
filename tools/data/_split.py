"""Helpers for partitioning GLiNER2 JSONL output into train/val/test splits.

Each converter writes three sibling files based on the user-supplied ``--out``
base path. If the user passes ``--out data/foo.jsonl`` the writer produces:

    data/foo.train.jsonl
    data/foo.val.jsonl
    data/foo.test.jsonl

Split assignment is deterministic: a seeded RNG draws one ``random()`` per
written record and routes it according to the cumulative ratio. Running the
same converter twice with the same seed produces the same partition.

**Pass ``group`` whenever one source document yields several records.** Per-row
routing sends copies of the same document to different splits, so its own eval
scores memorisation. Measured across ``data/``: text2json's val is 99.0%
contained in its train (9,737 rows over 2,093 unique documents), and the same
pattern appears in gliclass_logic (38%), knowledgator_gliner (27%),
events_biotech (22%) and klue_re (17%).

Usage::

    from _split import SplitWriter

    with SplitWriter(args.out, ratios=(0.8, 0.1, 0.1), seed=42) as writer:
        for record in records:
            writer.write(record, group=record["input"])
    print(writer.summary())     # "train=8123 val=1014 test=1003"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from pathlib import Path
from typing import IO, Dict, List, Optional, Sequence, Tuple


SPLIT_NAMES = ("train", "val", "test")

# Distinguishes "caller said nothing" (group on the input text) from an explicit
# group=None (route per row, the pre-2026-08-15 behaviour).
_USE_INPUT = object()

_WHITESPACE = re.compile(r"\s+")


# Stray Unicode line/paragraph separators that json.dumps writes literally
# (they are >= U+0020, so not escaped) yet str.splitlines() treats as line
# breaks. Left in place they fragment a JSONL record across physical lines,
# breaking any splitlines()-based reader. Map them to a plain space.
_LINE_SEPARATORS = str.maketrans({"\x85": " ", " ": " ", " ": " "})


# Invisible characters NFKC does NOT touch. NFKC already folds every space variant
# (NBSP, narrow NBSP, figure/hair space) to a plain space, but leaves the zero-width and
# bidi FORMAT characters intact -- and those break span matching: an LLM annotator copies
# the visible characters and drops the invisible ones, so its surface is no longer a
# substring of the text it came from and the mention is skipped.
#
# ZWNJ (U+200C) and ZWJ (U+200D) are DELIBERATELY KEPT. They are not decoration: ZWNJ
# separates words in Persian and Arabic, ZWJ forms conjuncts in Indic scripts and joins
# emoji sequences. Removing them changes the text.
_INVISIBLES = str.maketrans({
    "\u00ad": None,  # soft hyphen -- a discretionary line break, never rendered
    "\u180e": None,  # Mongolian vowel separator
    "\u200b": None,  # zero-width space
    "\u2060": None,  # word joiner
    "\ufeff": None,  # BOM / zero-width no-break space
    "\u061c": None,  # Arabic letter mark
    "\u200e": None, "\u200f": None,                    # LRM, RLM
    "\u202a": None, "\u202b": None, "\u202c": None,    # LRE, RLE, PDF
    "\u202d": None, "\u202e": None,                    # LRO, RLO
    "\u2066": None, "\u2067": None, "\u2068": None, "\u2069": None,  # isolates
})


def clean_text(s: str) -> str:
    """NFKC-normalize, strip stray line separators, and drop invisible formatting."""
    return (unicodedata.normalize("NFKC", s)
            .translate(_LINE_SEPARATORS)
            .translate(_INVISIBLES))


def normalize_group_key(s: str) -> str:
    """Return the document key two rows must share to land in the same split.

    The SAME rule the contamination checks use (``tools/data/check_leakage.py``,
    ``gliner2.training.split_hygiene``). Grouping on the raw string instead left
    texts differing only in case or whitespace with different group keys but the
    same document key -- they scattered across splits and were then flagged as
    contamination. Measured: events_biotech still leaked 123 documents
    train->val after grouping was added, purely from this mismatch.
    """
    normalized = unicodedata.normalize("NFKC", str(s)).casefold()
    return _WHITESPACE.sub(" ", normalized).strip()


def normalize_record(obj):
    """Recursively normalize every string in a nested JSON-like structure.

    Each string is NFKC-normalized and has stray Unicode line separators
    (NEL, U+2028, U+2029) replaced with a space; non-string scalars pass
    through unchanged. When two dict keys normalize to the same string and
    both values are lists, the lists are concatenated (otherwise last-wins)
    so a key collision never silently drops data.
    """
    if isinstance(obj, str):
        return clean_text(obj)
    if isinstance(obj, dict):
        out: dict = {}
        for k, v in obj.items():
            nk = normalize_record(k)
            nv = normalize_record(v)
            if nk in out and isinstance(out[nk], list) and isinstance(nv, list):
                out[nk] = out[nk] + nv
            else:
                out[nk] = nv
        return out
    if isinstance(obj, list):
        return [normalize_record(x) for x in obj]
    return obj


def dumps_record(record: dict) -> str:
    """Serialize a record to one JSONL line: normalized, non-ASCII kept."""
    return json.dumps(normalize_record(record), ensure_ascii=False)


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
        self._seed = seed
        self._rng = random.Random(seed)

    def __enter__(self) -> "SplitWriter":
        for name, path in self._paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            self._files[name] = path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for fh in self._files.values():
            fh.close()
        self._files = {}

    def _route(self, group: Optional[str] = None) -> str:
        """Pick a split, deterministically per ``group`` when one is given.

        Without a group every ROW draws independently, so a document appearing
        more than once scatters across splits and its own eval is contaminated.
        Measured on text2json: 9,737 rows over 2,093 unique documents, giving a
        val 99.0% contained in train. Routing on a stable hash of the group key
        keeps every row of a document together.

        ``hashlib`` rather than ``hash()``: the built-in is salted per process,
        so it would reshuffle the split on every run.
        """
        if group is None:
            x = self._rng.random()
        else:
            # Normalize with the SAME rule the contamination checks use
            # (tools/data/check_leakage.py, gliner2.training.split_hygiene).
            # Grouping on the raw string instead left texts differing only in case
            # or whitespace with different group keys but the same document key --
            # they scattered across splits and were then flagged as contamination.
            # Measured: events_biotech still leaked 123 documents train->val after
            # grouping was added, purely from this mismatch.
            digest = hashlib.sha1(
                f"{self._seed}:{normalize_group_key(group)}".encode("utf-8")
            ).digest()
            x = int.from_bytes(digest[:8], "big") / float(1 << 64)
        for i, threshold in enumerate(self._cum):
            if x < threshold:
                return SPLIT_NAMES[i]
        return SPLIT_NAMES[-1]

    def write(self, record: dict, group: Optional[str] = _USE_INPUT) -> str:
        """Write ``record`` to the chosen split and return the split name.

        **Grouped on ``record["input"]`` by default.** A document belongs in
        exactly one split, always; making that opt-in meant 20 of 21 converter
        write sites silently did not do it, and six corpora leaked into their own
        blind tests. Pass an explicit ``group`` when the grouping key is not the
        input text, or ``group=None`` to restore per-row routing.
        """
        if group is _USE_INPUT:
            group = record.get("input")
        split = self._route(group)
        fh = self._files[split]
        fh.write(dumps_record(record) + "\n")
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
