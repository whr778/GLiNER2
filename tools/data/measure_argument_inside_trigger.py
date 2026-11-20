"""How many gold event arguments sit inside a gold trigger of the same event.

An argument is inside when both it and a trigger of that event carry character
offsets and the argument span is contained in the trigger span, including the
case where the two spans are equal. An argument contained in any one of the
event's triggers counts as inside. String gold has no span, so it is counted
as unmeasured and left out of the rate.

    uv run python tools/data/measure_argument_inside_trigger.py data/wojood_hadath.train.jsonl
    uv run python tools/data/measure_argument_inside_trigger.py data/wojood_hadath
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

Span = Tuple[int, int]


def span_of(value) -> Optional[Span]:
    """``(start, end)`` when ``value`` is an offset dict with a non-empty span."""
    if not isinstance(value, dict):
        return None
    start, end = value.get("start"), value.get("end")
    if isinstance(start, int) and isinstance(end, int) and start < end:
        return start, end
    return None


def contained(argument: Span, trigger: Span) -> bool:
    """True when ``argument`` lies inside ``trigger``, edges included."""
    return trigger[0] <= argument[0] and argument[1] <= trigger[1]


def measure_record(record: dict) -> Tuple[Counter, Dict[str, Counter]]:
    """Return ``(totals, by_role)`` for one record.

    Totals and each role counter use the keys ``inside``, ``outside``, and
    ``unmeasured``.
    """
    totals: Counter = Counter()
    by_role: Dict[str, Counter] = {}
    events = (record.get("output") or {}).get("events") or []
    for event in events:
        if not isinstance(event, dict):
            continue
        triggers = [s for s in (span_of(t) for t in event.get("triggers") or []) if s]
        for argument in event.get("arguments") or []:
            if not isinstance(argument, dict):
                continue
            role = argument.get("role") or ""
            argument_span = span_of(argument.get("entity"))
            role_counts = by_role.setdefault(role, Counter())
            if argument_span is None or not triggers:
                totals["unmeasured"] += 1
                role_counts["unmeasured"] += 1
                continue
            key = "inside" if any(contained(argument_span, trigger) for trigger in triggers) else "outside"
            totals[key] += 1
            role_counts[key] += 1
    return totals, by_role


def measure_paths(paths: Iterable[Path]) -> Tuple[Counter, Dict[str, Counter], int]:
    """Totals, per-role counts, and record count over ``paths``."""
    totals: Counter = Counter()
    by_role: Dict[str, Counter] = {}
    records = 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                records += 1
                part, roles = measure_record(json.loads(line))
                totals.update(part)
                for role, counts in roles.items():
                    by_role.setdefault(role, Counter()).update(counts)
    return totals, by_role, records


def resolve_paths(names: List[str]) -> List[Path]:
    """JSONL files, or a stem such as ``data/wojood_hadath`` expanded to its splits."""
    found: List[Path] = []
    for name in names:
        path = Path(name)
        if path.is_file():
            found.append(path)
            continue
        matches = sorted(path.parent.glob(path.name + ".jsonl"))
        matches += sorted(path.parent.glob(path.name + ".*.jsonl"))
        if not matches:
            raise SystemExit(f"no jsonl for {name}")
        found.extend(matches)
    return found


def _rate(counts: Counter) -> str:
    placed = counts["inside"] + counts["outside"]
    share = counts["inside"] / placed if placed else 0.0
    return (
        f"inside {counts['inside']}/{placed} ({share:.2%})  "
        f"outside {counts['outside']}  unmeasured {counts['unmeasured']}"
    )


def report(paths: List[Path]) -> str:
    """The text report for ``paths``."""
    totals, by_role, records = measure_paths(paths)
    lines = [f"{len(paths)} file(s), {records} records", f"  {_rate(totals)}"]
    placed = [
        role for role in by_role
        if by_role[role]["inside"] + by_role[role]["outside"]
    ]
    for role in sorted(placed, key=lambda role: (-(by_role[role]["inside"] + by_role[role]["outside"]), role)):
        lines.append(f"  {role or '(no role)'}  {_rate(by_role[role])}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+", help="JSONL files, or a stem like data/wojood_hadath")
    args = parser.parse_args()
    paths = resolve_paths(args.datasets)
    print(report(paths))


if __name__ == "__main__":
    main()
