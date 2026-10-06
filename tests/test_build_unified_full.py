"""build_unified_full.py --output: the map goes where it is told, and the committed files stay put."""
import runpy
import sys
from pathlib import Path

SCRIPT = "tools/train/build_unified_full.py"
V2 = Path("tools/train/config/labels/unified-full-v2.yaml")


def test_v2_writes_to_output_and_matches_the_committed_file(tmp_path, monkeypatch):
    out = tmp_path / "v2.yaml"
    before = V2.read_bytes()
    monkeypatch.setattr(sys, "argv", [SCRIPT, "--v2", "--output", str(out)])
    runpy.run_path(SCRIPT, run_name="__main__")
    assert out.read_bytes() == before
    assert V2.read_bytes() == before
