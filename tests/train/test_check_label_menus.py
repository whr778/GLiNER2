"""The classification-menu gate, and proof it can fail.

Labels are an INPUT at inference. `unified.yaml` protects the ENTITY space with 95 map
entries; its `classifications.map` is EMPTY, so nothing but convention keeps a task's menu
identical across the corpora that declare it. That convention is currently maintained by
hand -- `unify_docee_menus.py` was run once to put docee/docee_zh/turkish_event on the same
60-label union -- and adding docee_zh to a base made `docee_event` a THREE-corpus task.

A gate that only ever passes proves nothing, so the failing case is tested first.
"""

import json
import subprocess
import sys

import yaml

GATE = "tools/train/check_label_menus.py"


def _corpus(tmp_path, name, labels, task="shared_task"):
    p = tmp_path / f"{name}.train.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for i in range(3):
            fh.write(json.dumps({
                "input": f"doc {name} {i}",
                "output": {"classifications": [
                    {"task": task, "labels": list(labels), "true_label": [labels[0]]}]},
            }, ensure_ascii=False) + "\n")
    return str(tmp_path / name)


def _config(tmp_path, prefixes, name="cfg.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump({"data": {"corpora": prefixes}}), encoding="utf-8")
    return str(p)


def _run(cfg):
    return subprocess.run([sys.executable, GATE, "--config", cfg],
                          capture_output=True, text=True)


def test_gate_FAILS_when_a_menu_diverges(tmp_path):
    """THE POINT OF THE GATE. One corpus omits a label the other offers."""
    a = _corpus(tmp_path, "a", ["x", "y", "z"])
    b = _corpus(tmp_path, "b", ["x", "y"])
    r = _run(_config(tmp_path, [a, b]))
    assert r.returncode == 1, r.stdout
    assert "DISAGREE" in r.stdout
    assert "'z'" in r.stdout, "the gate must name the missing label, not just complain"


def test_gate_passes_when_menus_agree(tmp_path):
    a = _corpus(tmp_path, "a", ["x", "y", "z"])
    b = _corpus(tmp_path, "b", ["z", "y", "x"])      # order must not matter
    r = _run(_config(tmp_path, [a, b]))
    assert r.returncode == 0, r.stdout


def test_a_task_in_one_corpus_only_is_fine(tmp_path):
    """Distinct tasks are separate namespaces; only a SHARED task can diverge."""
    a = _corpus(tmp_path, "a", ["x", "y"], task="task_a")
    b = _corpus(tmp_path, "b", ["p", "q", "r"], task="task_b")
    r = _run(_config(tmp_path, [a, b]))
    assert r.returncode == 0, r.stdout


def test_three_corpora_one_diverging(tmp_path):
    """The docee_event shape: adding a third corpus is where this breaks quietly."""
    a = _corpus(tmp_path, "a", ["x", "y", "z"])
    b = _corpus(tmp_path, "b", ["x", "y", "z"])
    c = _corpus(tmp_path, "c", ["x", "y"])
    r = _run(_config(tmp_path, [a, b, c]))
    assert r.returncode == 1, r.stdout
    assert "sizes=[2, 3, 3]" in r.stdout
