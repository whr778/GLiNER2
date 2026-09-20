"""Every metrics file must carry its operating point, from BOTH writers.

`eval_provenance` was added to the `eval` subcommand's writer only. The TRAINING path --
the one the Lambda runners publish and therefore the one every A/B verdict is read from --
kept writing 314 bare numbers. The roles A/B on 2026-09-20 had to reconstruct its operating
point by diffing configs, and `compare_runs.py` only warns when provenance is missing, so
it passed quietly.
"""
import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "tools" / "train" / "train.py"


def _tree():
    return ast.parse(SRC.read_text())


def test_exactly_one_provenance_builder():
    fns = [n.name for n in ast.walk(_tree())
           if isinstance(n, ast.FunctionDef) and n.name == "_eval_provenance"]
    assert fns == ["_eval_provenance"], "provenance must be built in ONE place"


def test_every_metrics_writer_sets_provenance():
    """Both assignments to `eval_provenance` must come from the shared helper."""
    calls = 0
    for node in ast.walk(_tree()):
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if (isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                    and t.slice.value == "eval_provenance"):
                assert isinstance(node.value, ast.Call), "must call the helper, not inline a dict"
                assert node.value.func.id == "_eval_provenance"
                calls += 1
    assert calls == 2, f"expected both writers to set provenance, found {calls}"


def test_provenance_carries_the_fields_a_comparison_needs():
    src = SRC.read_text()
    body = src[src.index("def _eval_provenance"):src.index("def _run_blind_test")]
    for field in ("commit", "threshold", "full_menu", "split", "records", "checkpoint"):
        assert f'"{field}"' in body, f"provenance is missing {field}"
