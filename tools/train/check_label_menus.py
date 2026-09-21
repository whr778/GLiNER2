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


def scan(corpora, limit=6000):
    """``task -> corpus -> frozenset(labels)`` over each corpus's train split."""
    menus = collections.defaultdict(dict)
    for prefix in corpora:
        path = f"{prefix}.train.jsonl"
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
    corpora = (cfg.get("data") or {}).get("corpora") or []
    menus = scan(corpora, args.records)

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
          f"{len(corpora)} corpora")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
