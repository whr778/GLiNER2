"""Update the auto-managed event-sweep table in tools/train/PAPER.md with one
config's blind-test metrics. Called by scripts/train_all_events.sh after each
training finishes.

Idempotent: re-running a config replaces its row (keyed by config name). Rows
live between the ``<!-- SWEEP_START -->`` / ``<!-- SWEEP_END -->`` markers.

    uv run python scripts/update_paper_metrics.py <config_name>
"""

import json
import sys
from pathlib import Path

import yaml

PAPER = Path("tools/train/PAPER.md")
START, END = "<!-- SWEEP_START -->", "<!-- SWEEP_END -->"


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "—"


def _metrics_path(cfg_name):
    cfg = yaml.safe_load(Path(f"tools/train/config/{cfg_name}.yaml").read_text())
    out = Path((cfg.get("training") or {}).get("output_dir", ""))
    for p in (out / "test_metrics.json", out / "best" / "test_metrics.json"):
        if p.exists():
            return p
    return None


def _row(cfg_name):
    mp = _metrics_path(cfg_name)
    if not mp:
        return f"| `{cfg_name}` | — | — | — | — | — | (no blind test) |"
    m = json.loads(mp.read_text())

    def g(cat, regime="strict"):
        return m.get(f"eval_{cat}_{regime}_micro_f1")

    arg = (f"{_fmt(g('event_argument'))} / {_fmt(g('event_argument','relaxed'))} / "
           f"{_fmt(g('event_argument','fair'))}")
    support = m.get("eval_event_strict_support", "—")
    return (f"| `{cfg_name}` | {_fmt(g('entity'))} | {_fmt(g('event_type'))} | "
            f"{_fmt(g('event_trigger'))} | {arg} | {_fmt(g('event'))} | {support} |")


def main(cfg_name):
    text = PAPER.read_text()
    if START not in text or END not in text:
        sys.exit(f"sweep markers not found in {PAPER}")
    head, rest = text.split(START, 1)
    mid, tail = rest.split(END, 1)

    rows = {}
    for line in mid.splitlines():
        if line.strip().startswith("| `"):
            rows[line.split("`")[1]] = line.rstrip()
    rows[cfg_name] = _row(cfg_name)

    body = "\n".join(rows[k] for k in sorted(rows))
    PAPER.write_text(f"{head}{START}\n{body}\n{END}{tail}")
    print(f"[paper] sweep row updated: {cfg_name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: update_paper_metrics.py <config_name>")
    main(sys.argv[1])
