"""Every base converter is wired into `run_all_converters.sh`, or this fails.

WHY THIS EXISTS. The runner's git history is three consecutive drift incidents:

    9d6d14d  run_all_converters.sh has not run since 2026-08-06; repair it and gate it
    ca775e1  run_all_converters.sh rebuilt the Chinese labels it was supposed to remove
    c6f4ac6  the rebuild path had not learned the last four days

The commit that said "gate it" did not add a gate, so coverage was a property someone
re-measured by hand and never a property anything maintained. A rebuild path that has
quietly stopped covering a corpus does not fail -- it succeeds, and writes a `data/`
that is missing something nobody notices until a run is scored against it.

HOW IT CLASSIFIES. A converter is a script on the mandated write path: CLAUDE.md requires
every converter to emit through `_split.dumps_record` / `SplitWriter`, so that import IS
the definition rather than a filename convention that drifts. Everything the runner's own
header excludes is then subtracted by prefix, and whatever remains must be referenced.

WHEN THIS FAILS, READ IT AS A QUESTION, NOT A CHORE. A new script on the SplitWriter path
is either a base converter -- wire it into the runner -- or one of the excluded kinds, in
which case add its prefix to EXCLUDED_PREFIXES here and say why. Silently widening the
exclusion list to make the test pass is how the runner drifted three times already.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA_TOOLS = ROOT / "tools" / "data"
RUNNER = DATA_TOOLS / "run_all_converters.sh"

# Each prefix is a KIND the runner's header excludes, with the reason it gives.
EXCLUDED_PREFIXES = {
    "_": "shared library module, not a script",
    "annotate_": "costs real money; the header says these are never run automatically",
    "synthetic": "costs real money (LLM generation)",
    "build_": "derived corpora -- scaling slices, mixes, dose arms; excluded by scope",
    "repair_": "repairs an existing corpus in place; not a pipeline step",
    "stamp_": "repairs corpora whose build invocation is unrecorded and cannot be re-derived",
    "push_": "publishes to the Hub",
    "restore_": "downloads from the Hub; the inverse of building",
    "audit_": "analysis",
    "check_": "analysis",
    "compare_": "analysis",
    "measure_": "analysis",
    "event_multiplicity": "analysis",
    "gate_purity_curve": "analysis",
    "screen_": "screening a candidate pool, not converting a corpus",
    "prefilter_": "screening a candidate pool, not converting a corpus",
    "fetch_": "acquires raw source material",
    "harvest_": "acquires raw source material",
    "hf_stream": "helper for streaming Hub datasets",
    "normalize_gate_source": "prepares one gate corpus's source",
    "interleave_": "repairs split ordering in an existing corpus",
    "translate_": "label-language repair on an existing corpus",
    "unify_": "label unification on an existing corpus",
    "merge_entity_types": "label surgery on an existing corpus",
    "augment_": "derives an augmented corpus from an existing one",
    "split_": "re-splits an existing corpus",
    "events_to_": "derives a corpus from an existing one",
    "roles_to_": "derives a corpus from an existing one",
}

WRITE_PATH = re.compile(r"SplitWriter|from _split|import _split|dumps_record")


def _referenced() -> set[str]:
    src = RUNNER.read_text(encoding="utf-8")
    return set(re.findall(r"tools/data/([A-Za-z0-9_]+)\.py", src))


def _on_write_path() -> list[str]:
    out = []
    for f in sorted(DATA_TOOLS.glob("*.py")):
        if WRITE_PATH.search(f.read_text(encoding="utf-8", errors="replace")):
            out.append(f.stem)
    return out


def _excluded(stem: str) -> bool:
    return any(stem.startswith(p) for p in EXCLUDED_PREFIXES)


def test_the_runner_exists_and_names_converters():
    """The premise. If the runner stopped naming scripts this way, the rest proves nothing."""
    assert RUNNER.is_file()
    assert len(_referenced()) > 20, "the runner suddenly references almost nothing"


def test_every_base_converter_is_in_the_runner():
    referenced = _referenced()
    missing = sorted(s for s in _on_write_path() if not _excluded(s) and s not in referenced)
    assert not missing, (
        "these scripts write corpora through SplitWriter but `run_all_converters.sh` never "
        f"runs them, so a rebuild silently omits them: {missing}. Wire each into the runner, "
        "or -- if it is not a base converter -- add its prefix to EXCLUDED_PREFIXES with the "
        "reason. Do not widen the exclusions just to get green."
    )


def test_the_runner_does_not_name_a_script_that_is_gone():
    """The other direction: a rebuild that invokes a deleted file fails halfway through."""
    absent = sorted(n for n in _referenced() if not (DATA_TOOLS / f"{n}.py").is_file())
    assert not absent, f"run_all_converters.sh invokes scripts that no longer exist: {absent}"


@pytest.mark.parametrize("prefix", sorted(EXCLUDED_PREFIXES))
def test_each_exclusion_still_matches_something(prefix):
    """An exclusion that matches nothing is dead, and dead exclusions hide the next one.

    If this fails the file was renamed or removed -- drop the prefix rather than leaving a
    rule nobody can evaluate.
    """
    stems = [f.stem for f in DATA_TOOLS.glob("*.py")] + [
        d.name for d in DATA_TOOLS.iterdir() if d.is_dir()
    ]
    assert any(s.startswith(prefix) for s in stems), (
        f"EXCLUDED_PREFIXES has {prefix!r} ({EXCLUDED_PREFIXES[prefix]}) but nothing matches it"
    )
