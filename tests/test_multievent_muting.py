"""Unit tests for interference muting in the multi-event casualty corpus.

Muting withholds an interference snippet's record while keeping its text, so its figures
become negatives for the same queries. Two properties decide whether the resulting A/B
means anything, and neither raises when broken:

1. **The control arm must still reproduce the historical corpus.** Drawing the muting
   decision from the shared ``rng`` advanced it once per interference snippet, which
   shifted every later ``randint``/``choice`` and rebuilt the corpus -- 4064 documents
   against the pre-change script's 4065, at ``mute_interference_prob=0.0``.

   Note what does NOT catch this: comparing arms to each other. The buggy draw fired once
   per interference snippet *regardless of probability*, so every arm shifted together and
   stayed mutually identical. Only a comparison against a builder with no muting concept
   at all exposes it, which is why the control is pinned by hash below. The hash was fixed
   after confirming, with ``cmp``, that the current control is byte-identical to the
   pre-change script's output on the real 4,106-snippet corpus.
2. **The focal record is never withheld.** Muting the lead event would teach suppression
   of a genuine toll rather than cross-event discrimination.
"""

import hashlib
import importlib.util
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "datasets" / "disaster_streams"
_BUILD = _SRC / "build_multievent_corpus.py"


def _load():
    sys.path.insert(0, str(_SRC))
    spec = importlib.util.spec_from_file_location("build_multievent_corpus", _BUILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load()


def _snippets(n_streams=5, per_stream=3):
    """Distinct values throughout, so no gold is ever dropped as a collision.

    Collisions are a real behaviour of the builder but a confound here: they remove
    records for reasons unrelated to muting, which would blur every assertion below.
    """
    out, value = [], 100
    for s in range(n_streams):
        for j in range(per_stream):
            value += 7
            out.append({
                "stream": f"s{s}",
                "text": f"Officials in district {s}-{j} said {value} people died.",
                "gt": {"dead": value},
            })
    return out


def _gold_values(example):
    return {v for st in example.structures for v in st.to_dict()["casualty_report"].values()}


def _build(prob, seed=1):
    return mod.build(_snippets(), max_interference=3, seed=seed,
                     mute_interference_prob=prob)


def test_control_arm_matches_the_pre_change_corpus():
    """Pins the untreated corpus so the muting feature cannot perturb it.

    This is the assertion the shared-RNG bug fails. Do NOT re-baseline this hash to make a
    failure go away: a change here means the control arm moved, so every earlier result
    measured against it is no longer comparable.
    """
    examples, stats = _build(0.0)
    blob = "\n".join(e.to_json() for e in examples)
    assert stats["muted_snippets"] == 0
    assert len(examples) == 15
    assert hashlib.sha1(blob.encode()).hexdigest() == (
        "dc56a240cee1b8e530f8535b143ebbafc8b1eff8"
    )


def test_documents_are_identical_across_mute_probabilities():
    """The arms must differ in labels only -- same seed, same documents, same order.

    Necessary but not sufficient on its own; see the module docstring.
    """
    base, _ = _build(0.0)
    for prob in (0.35, 0.5, 1.0):
        treated, _ = _build(prob)
        assert [e.text for e in treated] == [e.text for e in base], (
            f"mute_interference_prob={prob} changed document composition"
        )


def test_muting_withholds_records_without_touching_text():
    base, base_stats = _build(0.0)
    treated, stats = _build(0.5)

    assert base_stats["muted_snippets"] == 0
    assert stats["muted_snippets"] > 0
    assert stats["instances"] < base_stats["instances"]
    # Every withheld figure is still readable in the document it was withheld from.
    assert stats["unlabelled_figures"] > 0


def test_focal_record_is_never_muted():
    """The first snippet's value stays gold at every probability, including 1.0."""
    for prob in (0.5, 1.0):
        examples, _ = _build(prob)
        for e in examples:
            focal_value = re.search(r"said (\d+) people died", e.text).group(1)
            assert focal_value in _gold_values(e)


def test_full_muting_leaves_exactly_one_record_per_document():
    examples, stats = _build(1.0)
    assert examples
    assert all(len(e.structures) == 1 for e in examples)
    assert stats["instances"] == len(examples)


def test_muted_figures_are_present_in_text_but_absent_from_gold():
    """The point of the corpus change: a figure the model must learn to leave alone."""
    examples, _ = _build(1.0)
    unlabelled = 0
    for e in examples:
        in_text = set(re.findall(r"said (\d+) people died", e.text))
        unlabelled += len(in_text - _gold_values(e))
    assert unlabelled > 0


# --- focal position: the control that separates event identity from "read paragraph 1" ---

def _build_at(prob, position, seed=1):
    return mod.build(_snippets(), max_interference=3, seed=seed,
                     mute_interference_prob=prob, focal_position=position)


def test_focal_last_reorders_the_same_snippets():
    """Same document content, focal moved to the end -- not a different draw."""
    first, _ = _build_at(0.0, "first")
    last, _ = _build_at(0.0, "last")
    assert len(first) == len(last)
    for a, b in zip(first, last):
        assert sorted(a.text.split("\n\n")) == sorted(b.text.split("\n\n"))
        assert a.text != b.text or len(a.text.split("\n\n")) <= 2


def test_focal_last_carries_identical_gold():
    """The probe pairs document for document, so a score drop is position and nothing else."""
    for prob in (0.0, 0.5, 1.0):
        first, _ = _build_at(prob, "first")
        last, _ = _build_at(prob, "last")
        for a, b in zip(first, last):
            assert _gold_values(a) == _gold_values(b)


def test_focal_is_exempt_from_muting_at_either_position():
    """Muting keys on identity, not index, so the last-placed focal keeps its record."""
    for prob in (0.5, 1.0):
        for position in ("first", "last"):
            examples, _ = _build_at(prob, position)
            for e in examples:
                values = re.findall(r"said (\d+) people died", e.text)
                focal = values[0] if position == "first" else values[-1]
                assert focal in _gold_values(e)
