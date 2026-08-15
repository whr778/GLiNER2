"""A document that yields several records must not be split across splits.

`SplitWriter._route` drew one random per ROW, so copies of one document scattered
into train/val/test and the corpus scored its own memorisation. Measured across
`data/` before the fix: text2json's val was 99.0% contained in its train (9,737
rows over 2,093 unique documents), gliclass_logic 38.3%, knowledgator_gliner
27.1%, events_biotech 21.6%, klue_re 17.3%.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "data"))

from _split import SplitWriter  # noqa: E402


def _written(base: Path) -> dict[str, set[str]]:
    """{document text: set of splits it appears in}."""
    where: dict[str, set[str]] = defaultdict(set)
    for split, path in SplitWriter(base).paths.items():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                where[json.loads(line)["input"]].add(split)
    return where


def test_grouped_rows_land_in_one_split(tmp_path):
    base = tmp_path / "corpus.jsonl"
    with SplitWriter(base, ratios=(0.6, 0.2, 0.2), seed=7) as writer:
        for doc in range(40):
            for schema in range(5):
                writer.write(
                    {"input": f"document {doc}", "output": {"schema": schema}},
                    group=f"document {doc}",
                )

    where = _written(base)
    assert len(where) == 40
    scattered = {d: s for d, s in where.items() if len(s) > 1}
    assert not scattered, f"documents split across splits: {scattered}"


def test_ungrouped_rows_still_scatter(tmp_path):
    """The old behaviour is intact for callers that pass no group -- the fix is
    opt-in, so converters emitting one record per document are unchanged."""
    base = tmp_path / "corpus.jsonl"
    with SplitWriter(base, ratios=(0.6, 0.2, 0.2), seed=7) as writer:
        for doc in range(40):
            for schema in range(5):
                writer.write({"input": f"document {doc}", "output": {"schema": schema}})

    where = _written(base)
    assert any(len(s) > 1 for s in where.values()), (
        "expected row-wise routing to scatter duplicates; if this now passes, the "
        "default changed and every converter's partition moved with it"
    )


def test_grouping_is_stable_across_runs(tmp_path):
    """Built-in hash() is salted per process; a converter re-run must reproduce
    its partition or the split silently changes under you."""
    placements = []
    for run in range(2):
        base = tmp_path / f"run{run}.jsonl"
        with SplitWriter(base, ratios=(0.8, 0.1, 0.1), seed=42) as writer:
            for doc in range(60):
                writer.write({"input": f"doc {doc}", "output": {}}, group=f"doc {doc}")
        placements.append({d: sorted(s) for d, s in _written(base).items()})

    assert placements[0] == placements[1]


def test_seed_still_changes_the_partition(tmp_path):
    """Grouping must not accidentally make the seed inert."""
    placements = []
    for seed in (1, 2):
        base = tmp_path / f"seed{seed}.jsonl"
        with SplitWriter(base, ratios=(0.8, 0.1, 0.1), seed=seed) as writer:
            for doc in range(60):
                writer.write({"input": f"doc {doc}", "output": {}}, group=f"doc {doc}")
        placements.append({d: sorted(s) for d, s in _written(base).items()})

    assert placements[0] != placements[1]
