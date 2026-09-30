"""Tests for tools/data/build_capped_slice.py."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "build_capped_slice",
    Path(__file__).resolve().parent.parent / "tools" / "data" / "build_capped_slice.py")
bcs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bcs)


def _doc(i, n_args):
    args = [{"role": f"R{k}", "entity": f"e{i}_{k}"} for k in range(n_args)]
    return ("src.jsonl", {"input": f"doc {i}", "output": {"events": [
        {"event_type": "Attack", "triggers": [f"t{i}"], "arguments": args}]}})


def test_stops_once_the_cap_is_reached_with_whole_documents():
    records = [_doc(i, 3) for i in range(20)]
    picked, total = bcs.take_until(records, cap=10, seed=0)
    assert total >= 10 and total - 3 < 10          # the last whole document crosses the cap
    assert len(picked) == 4                         # 3 + 3 + 3 + 3 = 12


def test_same_seed_same_slice():
    records = [_doc(i, i % 4) for i in range(50)]
    a, _ = bcs.take_until(records, cap=20, seed=7)
    b, _ = bcs.take_until(records, cap=20, seed=7)
    assert [r[1]["input"] for r in a] == [r[1]["input"] for r in b]
