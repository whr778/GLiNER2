"""Make train/val/test mutually disjoint before a single step is taken.

A blind test measured partly on trained documents is not a measurement, so this
runs as a GATE rather than a diagnostic. Both live configs failed it when it was
written: `joint-boundary-mmbert-137k` shared 1,080 documents between train and
test (7.03% of test) and `warmstart-natural` shared 299 (1.95%).

Two different notions of "duplicate", and conflating them loses real supervision:

WITHIN a split, only an exact repeat -- same text AND same target -- is waste.
    text2json legitimately emits one document up to 10 times with 8 distinct
    extraction schemas, which is the schema-conditioning signal that corpus
    exists to teach. Dropping those on text alone would throw it away.

ACROSS splits, any shared TEXT is contamination, whatever the targets say.
    The model has seen the document; scoring it again measures memorisation.

Priority is fixed by what the splits are for: **test is authoritative** (it is the
blind set), val yields to test, and train yields to both. So removal always comes
out of the lower-priority split and the blind test is never modified.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")

# test first: everything yields to the blind set.
PRIORITY = ("test", "val", "train")


def document_key(text: str) -> str:
    """Identity of a document, insensitive to case and whitespace."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return hashlib.sha1(_WHITESPACE.sub(" ", normalized).strip().encode("utf-8")).hexdigest()


def _record_key(record: dict) -> str:
    """Identity of a full training example: document plus its target."""
    target = json.dumps(record.get("output"), sort_keys=True, ensure_ascii=False)
    return document_key(str(record.get("input", "")) + "\x00" + target)


@dataclass
class SplitReport:
    """What the gate found and what it removed."""

    kept: Dict[str, int] = field(default_factory=dict)
    exact_duplicates: Dict[str, int] = field(default_factory=dict)
    contaminated: Dict[Tuple[str, str], int] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not self.contaminated and not any(self.exact_duplicates.values())

    def format(self) -> str:
        lines = ["[split hygiene] " + ("CLEAN" if self.clean else "REPAIRED")]
        for split in ("train", "val", "test"):
            if split not in self.kept:
                continue
            dropped = self.exact_duplicates.get(split, 0)
            note = f"  (dropped {dropped} exact duplicate(s))" if dropped else ""
            lines.append(f"    {split:<6} {self.kept[split]:>8} records{note}")
        for (loser, winner), n in sorted(self.contaminated.items()):
            lines.append(
                f"    removed {n} document(s) from {loser}: also present in {winner}"
            )
        return "\n".join(lines)


def enforce_disjoint_splits(
    splits: Dict[str, Optional[Sequence[dict]]],
    *,
    drop_exact_duplicates: bool = True,
) -> Tuple[Dict[str, List[dict]], SplitReport]:
    """Return splits with contamination removed, plus a report of what changed.

    ``splits`` maps ``"train"``/``"val"``/``"test"`` to record lists; a missing or
    ``None`` entry is skipped. Records are compared by their ``input`` text.
    """
    report = SplitReport()
    cleaned: Dict[str, List[dict]] = {}
    claimed: Dict[str, str] = {}  # document key -> the split that owns it

    for split in PRIORITY:
        records = splits.get(split)
        if records is None:
            continue

        seen_records: set = set()
        kept: List[dict] = []
        duplicates = 0
        stolen: Dict[str, int] = {}

        for record in records:
            text = record.get("input")
            if not text:
                continue
            doc = document_key(str(text))

            owner = claimed.get(doc)
            if owner is not None and owner != split:
                stolen[owner] = stolen.get(owner, 0) + 1
                continue

            if drop_exact_duplicates:
                exact = _record_key(record)
                if exact in seen_records:
                    duplicates += 1
                    continue
                seen_records.add(exact)

            claimed[doc] = split
            kept.append(record)

        cleaned[split] = kept
        report.kept[split] = len(kept)
        if duplicates:
            report.exact_duplicates[split] = duplicates
        for winner, n in stolen.items():
            report.contaminated[(split, winner)] = n

    return cleaned, report


def check_and_clean(
    train: Optional[Sequence[dict]],
    val: Optional[Sequence[dict]] = None,
    test: Optional[Sequence[dict]] = None,
    *,
    policy: str = "drop",
) -> Tuple[Optional[List[dict]], Optional[List[dict]], Optional[List[dict]], SplitReport]:
    """Gate the three splits according to ``policy``.

    ``drop``   remove the offending records and log what went (default)
    ``raise``  refuse to train on contaminated splits
    ``warn``   report only, change nothing -- for reproducing an older run
    """
    if policy not in ("drop", "raise", "warn"):
        raise ValueError(f"policy must be drop|raise|warn, got {policy!r}")

    cleaned, report = enforce_disjoint_splits({"train": train, "val": val, "test": test})
    logger.info("%s", report.format())

    if not report.clean and policy == "raise":
        raise ValueError(
            "train/val/test are not disjoint and split_hygiene='raise':\n"
            + report.format()
        )
    if policy == "warn":
        return (
            list(train) if train is not None else None,
            list(val) if val is not None else None,
            list(test) if test is not None else None,
            report,
        )
    return (
        cleaned.get("train") if train is not None else None,
        cleaned.get("val") if val is not None else None,
        cleaned.get("test") if test is not None else None,
        report,
    )
