"""Load a frozen guide's cached scores and inject the rival queries GIST needs.

GIST removes a mined negative from the loss when a frozen guide judges it to be a
positive for that instance. Two things have to be true before that rule can do anything,
and both are handled here.

**The rival has to be in the schema at all.** A training sample's query axis carries only
the types its own record declares. Cross-record rivals -- the in-batch negatives GIST is
about -- never appear, and only 0.23% of records happen to declare a competing count type
natively. So the veto without injection is a no-op by construction: the cells it would act
on do not exist. :meth:`GuideScores.inject` adds the record's cached rivals to its schema
as absent entity queries, which is what puts those cells on the tensor.

**Own-record types must be left alone.** Within a record gold is authoritative: if the
schema declares both ``ReleasedDate`` and ``StartDate`` and gold assigns the span to
``ReleasedDate``, then ``StartDate`` is definitively wrong there and is a correct hard
negative. A same-record rival outscores the gold owner 23.5% of the time (measured on 200
records), so vetoing on own-record types would delete that material wholesale. The split is
enforced downstream by *which cells get a guide value at all*: only a span's own gold query
and the injected rivals are filled, everything else stays at 0.0 and cannot clear the
veto's ``floor``.

Lookup is by the sha1 of the record's text rather than its corpus position, because
training filters, shuffles and re-splits its inputs; a positional key would quietly point
at a different record's spans.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

_WHITESPACE = re.compile(r"\s+")


def surface_key(text: str) -> str:
    """Whitespace-free, case-folded span key.

    Both sides of the lookup have to agree on one string. The cache holds raw gold values
    (``"1,400"``) while training reconstructs a span by joining word tokens, which may have
    split it (``"1 , 400"``). Dropping whitespace entirely makes the two identical; within
    a single record, collisions do not matter.
    """
    return _WHITESPACE.sub("", text).casefold()


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GuideEntry:
    """One record's cached guide scores."""

    own: frozenset
    rival: Mapping[str, str]           # type name -> description
    spans: Mapping[str, Mapping[str, float]]   # surface_key -> type name -> score


class GuideScores:
    """Cached guide scores for a corpus, keyed by record text."""

    def __init__(self, entries: Dict[str, GuideEntry]):
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def load(cls, path) -> "GuideScores":
        """Read a JSONL cache written by ``tools/train/precompute_guide_scores.py``."""
        entries: Dict[str, GuideEntry] = {}
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                entries[row["sha1"]] = GuideEntry(
                    own=frozenset(row["own"]),
                    rival=dict(row["rival"]),
                    spans={surface_key(span): scores
                           for span, scores in row["s"].items()},
                )
        return cls(entries)

    def get(self, text: str) -> Optional[GuideEntry]:
        return self.entries.get(text_key(text))

    def inject(self, text: str, schema: Any, count: int) -> Any:
        """Add the ``count`` hardest cached rivals to ``schema`` as absent entity queries.

        The cache stores rivals best-first, so this takes a prefix rather than a sample:
        the point of spending a query slot is the rival that scores the span highest, and
        a lower-ranked one is a negative the model already gets right. Deterministic, so a
        record sees the same rivals in every epoch and on every rank.
        """
        entry = self.get(text)
        if entry is None or not entry.rival or not isinstance(schema, dict):
            return schema
        chosen = [name for name in list(entry.rival)[:count]
                  if name not in (schema.get("entities") or {})]
        if not chosen:
            return schema
        out = dict(schema)
        out["entities"] = {**(out.get("entities") or {}), **{name: [] for name in chosen}}
        out["entity_descriptions"] = {
            **(out.get("entity_descriptions") or {}),
            **{name: entry.rival[name] for name in chosen},
        }
        return out


__all__ = ["GuideEntry", "GuideScores", "surface_key", "text_key"]
