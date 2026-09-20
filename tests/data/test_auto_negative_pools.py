"""`negative_pools: auto` must cover every corpus a config names.

The committed pools JSON is keyed by corpus name and silently omits anything added since it
was generated. `cmnee_roles_ner` was absent from it, so `NegativeLabels._candidates` matched
no pool and 300 of 300 of its records were skipped as `no_candidate` -- option 4's treatment
arm trained on pure positive entity supervision and over-proposed. A static list is the bug.
"""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _builder():
    spec = importlib.util.spec_from_file_location(
        "_pb_test", ROOT / "tools" / "data" / "build_negative_pools.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CFG = ROOT / "tools" / "train" / "config" / "ab" / "roles-treatment.yaml"


@pytest.mark.skipif(not CFG.is_file(), reason="config not present")
def test_auto_pools_cover_every_corpus_in_the_config():
    mod = _builder()
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    wanted = set(mod._corpus_train_paths(cfg))
    pools = mod.build_pools(cfg, CFG, limit=50)
    assert not (wanted - set(pools)), f"corpora with no derived pool: {sorted(wanted - set(pools))}"


@pytest.mark.skipif(not CFG.is_file(), reason="config not present")
def test_the_committed_file_is_the_thing_this_replaces():
    """Documents the gap rather than asserting it stays: the static file omits a corpus the
    config trains on, which is the whole reason auto-derivation exists."""
    mod = _builder()
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    static_path = ROOT / "tools" / "train" / "config" / "labels" / "negative_pools.json"
    if not static_path.is_file():
        pytest.skip("no committed pools file")
    static = set(json.load(open(static_path, encoding="utf-8"))["pools"])
    derived = set(mod.build_pools(cfg, CFG, limit=50))
    assert derived >= set(mod._corpus_train_paths(cfg)) - set()
    # the derived set is a superset of the static one for this config's corpora
    assert not (set(mod._corpus_train_paths(cfg)) & static) - derived


@pytest.mark.skipif(not CFG.is_file(), reason="config not present")
def test_partial_annotation_turns_a_dimension_off_and_is_recorded():
    """PARTIAL may only ever turn annotation OFF, and must say so in the pool."""
    mod = _builder()
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    partial = mod._partial_annotation(cfg)
    if not partial:
        pytest.skip("this config declares no partial_annotation")
    pools = mod.build_pools(cfg, CFG, limit=50)
    for name, dims in partial.items():
        for dim in dims:
            assert pools[name]["annotates"][dim] is False
            assert dim in pools[name].get("partial", []), "the downgrade must be recorded"


def test_partial_annotation_rejects_a_corpus_the_config_does_not_train_on():
    """A typo here would protect nothing, silently."""
    mod = _builder()
    cfg = {"data": {"corpora": ["data/casie"], "partial_annotation": {"nosuch": ["entities"]}}}
    with pytest.raises(SystemExit, match="not in this config"):
        mod.build_pools(cfg, CFG, limit=1)


def test_partial_annotation_rejects_an_unknown_dimension():
    mod = _builder()
    cfg = {"data": {"corpora": ["data/casie"], "partial_annotation": {"casie": ["entites"]}}}
    with pytest.raises(SystemExit, match="unknown dimension"):
        mod.build_pools(cfg, CFG, limit=1)
