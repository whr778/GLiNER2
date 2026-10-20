"""Which TRAIN files a config resolves to on THIS machine, and how many of their records carry each task.

train.py drops an event_files path it cannot find with one printed line and trains on without it, so a machine
whose data restore missed the event corpora trains an event-blind model that scores 0.0 on every event head
(eb22c at the office, 2026-10-20). Nothing is fetched here: this reports what is on disk now.

    uv run tools/train/check_train_files.py --config tools/train/config/base/eb22c.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def train_paths(cfg: dict) -> list[str]:
    """Corpora train files (train_only included) then event_files train paths, de-duplicated in order."""
    d = cfg["data"]
    paths = [f"data/{Path(c).name}.train.jsonl" for c in d["corpora"]]
    paths += [f["train"] for f in (d.get("event_files") or {}).values() if isinstance(f, dict) and f.get("train")]
    return list(dict.fromkeys(paths))


def task_counts(path: str) -> dict:
    """Records, and records carrying entities / relations / events (at any nesting under `output`)."""
    n = {"records": 0, "entities": 0, "relations": 0, "events": 0}
    for line in open(path, encoding="utf-8"):
        out = (json.loads(line).get("output") or {})
        n["records"] += 1
        for k in ("entities", "relations", "events"):
            n[k] += bool(out.get(k))
    return n


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    total, missing = {"records": 0, "entities": 0, "relations": 0, "events": 0}, []
    for p in train_paths(cfg):
        if not Path(p).is_file():
            missing.append(p)
            print(f"MISSING  {p}")
            continue
        n = task_counts(p)
        total = {k: total[k] + n[k] for k in total}
        print(f"ok       {p:60} records {n['records']:>7}  events {n['events']:>7}  entities {n['entities']:>7}  "
              f"relations {n['relations']:>7}")
    print(f"\nTOTAL    records {total['records']}  with events {total['events']}  entities {total['entities']}  "
          f"relations {total['relations']}  |  MISSING {len(missing)} file(s)")


if __name__ == "__main__":
    main()
