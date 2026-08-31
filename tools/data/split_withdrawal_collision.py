"""Split the one many-to-one collision the DuEE label translation introduced.

`竞赛行为-退赛` (pulling out of a match) and `组织关系-退出` (leaving an organization) are
distinct DuEE event types that both translated to `withdrawal`, merging two classes into
one. Their role sets are disjoint, so the original type is recoverable from the arguments:

    exit_party / Original Organization  -> 组织关系-退出   -> departure
    withdrawal_party / withdrawn_event  -> 竞赛行为-退赛   -> competition_withdrawal

    uv run python tools/data/split_withdrawal_collision.py data/duee.train.jsonl ...
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402

ORG_ROLES = {"exit_party", "Original Organization"}
COMPETITION_ROLES = {"withdrawal_party", "withdrawn_event"}


def retype(event):
    """Return the disambiguated event type, or None if the roles do not decide it."""
    roles = {a.get("role") for a in event.get("arguments") or []}
    if roles & ORG_ROLES and not roles & COMPETITION_ROLES:
        return "departure"
    if roles & COMPETITION_ROLES and not roles & ORG_ROLES:
        return "competition_withdrawal"
    return None


def main(paths):
    tally = Counter()
    for path in paths:
        src = Path(path)
        rows = []
        for line in src.open(encoding="utf-8"):
            record = json.loads(line)
            for event in (record.get("output") or record).get("events") or []:
                if event.get("event_type") != "withdrawal":
                    continue
                new_type = retype(event)
                if new_type is None:
                    tally["UNDECIDED"] += 1
                    continue
                event["event_type"] = new_type
                tally[new_type] += 1
            rows.append(record)
        with src.open("w", encoding="utf-8") as out:
            for record in rows:
                out.write(dumps_record(record) + "\n")
        print(f"{src}: {len(rows)} records")
    print("retyped:", dict(tally))
    assert tally["UNDECIDED"] == 0, "roles did not decide every collision"


if __name__ == "__main__":
    main(sys.argv[1:])
