"""Refuse a config whose classification MENUS disagree between corpora.

Labels are an INPUT at inference, so a task offered as 60 labels in one corpus and 58 in
another has never been asked to consider the missing two for those documents, and cannot
decline them either. The entity label space is protected by `unified.yaml`'s 95-entry map;
the CLASSIFICATION map is EMPTY, so nothing protects this but convention.

It currently holds by hand: `unify_docee_menus.py` was run once to bring docee (59),
docee_zh (58) and turkish_event (60) onto the same 60-label union, rewriting only `labels`
and never `true_label`. Nothing verified it would stay true, and adding docee_zh to a base
made `docee_event` a THREE-corpus task -- exactly the shape that breaks quietly.

    uv run python tools/train/check_label_menus.py --config <config.yaml>

Exits non-zero when a task's menu differs between the corpora that declare it.
"""

import argparse
import collections
import json
import os
import sys

import yaml


def scan(paths, limit=6000):
    """``task -> corpus -> frozenset(labels)`` over each FILE the trainer will load."""
    menus = collections.defaultdict(dict)
    for path in paths:
        prefix = path[: -len(".jsonl")] if path.endswith(".jsonl") else path
        for suffix in (".train", ".val", ".test"):
            if prefix.endswith(suffix):
                prefix = prefix[: -len(suffix)]
                break
        if not os.path.exists(path):
            continue
        seen = collections.defaultdict(set)
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= limit:
                    break
                for item in (json.loads(line).get("output") or {}).get("classifications") or []:
                    task = item.get("task")
                    if task:
                        seen[task].update(item.get("labels") or [])
        for task, labels in seen.items():
            menus[task][os.path.basename(prefix)] = frozenset(labels)
    return menus


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--records", type=int, default=6000,
                    help="records scanned per corpus")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    # Resolve the files the TRAINER will actually load, with the trainer's own helpers.
    # Reading `data.corpora` and appending ".train.jsonl" by hand missed two things and so
    # could not fail on either: a corpus named ONLY in `event_files` was never scanned at
    # all (mendeley_ed), and an `event_files` entry pointing its split at a different path
    # was scanned at the wrong one.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import train as _train

    data = cfg.get("data") or {}
    corpora = data.get("corpora") or []
    train_only = set(data.get("train_only") or ())
    event_files = data.get("event_files") or {}
    paths = _train._dedupe_paths(
        _train._split_files(corpora, "train", train_only)
        + _train._event_split(event_files, "train"),
        "train",
    )
    menus = scan(paths, args.records)

    failed = []
    for task, per_corpus in sorted(menus.items()):
        sizes = sorted(len(m) for m in per_corpus.values())
        agree = len(set(per_corpus.values())) == 1
        print(f"[menus] {task:22s} {len(per_corpus)} corpora  sizes={sizes}  "
              f"{'OK' if agree else '*** DISAGREE ***'}")
        if not agree:
            failed.append(task)
            union = set().union(*per_corpus.values())
            for corpus, menu in sorted(per_corpus.items()):
                missing = sorted(union - menu)
                if missing:
                    print(f"          {corpus}: missing {len(missing)} -> {missing[:8]}")

    if failed:
        print(f"\n[menus] *** REFUSING: {len(failed)} task(s) offer different menus in "
              f"different corpora: {failed}. Labels are an INPUT -- unify the menus "
              f"(see tools/data/unify_docee_menus.py) before training. ***")
        return 1
    print(f"\n[menus] every classification task offers ONE menu across "
          f"{len(paths)} train files")
    print("[menus] NOTE: this scans the LOCAL data/ tree. A fresh box fetches every corpus "
          "from HF, so a local pass does NOT prove the RUN's menus agree -- eb17-best "
          "trained docee at 59 labels against docee_zh and turkish_event at 60 while this "
          "gate read 60/60/60 on disk. Verify local and HF agree before launching.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
