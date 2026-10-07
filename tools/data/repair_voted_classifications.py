"""Re-vote the classifications of a voted eval set from its raw runs, changing nothing else.

    uv run python tools/data/repair_voted_classifications.py \\
        --runs /Volumes/Development/tmp/english_v2_buy/eval_run{1,2,3}.train.jsonl \\
        --files data/cc_news_events_sonnet55_v2.val.jsonl data/cc_news_events_sonnet55_v2.test.jsonl

The first vote wrote the winning labels as the task's ``labels`` and no ``true_label`` (fixed in
``vote_annotations.vote_classifications``). Each file's records are matched to the raw runs by NFKC
input text; only ``output.classifications`` is replaced, with the same-menu vote. The script refuses
unless every other byte of every record is unchanged and every kept task's answer is on its menu.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402
from vote_annotations import vote_classifications  # noqa: E402


def key(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True, help="the raw annotation runs, one JSONL each")
    ap.add_argument("--files", nargs="+", required=True, help="the voted split files to repair in place")
    ap.add_argument("--min-votes", type=int, default=2)
    args = ap.parse_args(argv)

    runs = [{key(json.loads(l)["input"]): json.loads(l)["output"] for l in open(p, encoding="utf-8")} for p in args.runs]
    for path in args.files:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        out, stats = [], Counter()
        for line in lines:
            rec = json.loads(line)
            outputs = [r[key(rec["input"])] for r in runs]
            asked = Counter(c["task"] for o in outputs for c in o.get("classifications") or [] if c.get("task"))
            new = vote_classifications(outputs, args.min_votes)
            stats["records"] += 1
            stats["tasks_asked"] += len(asked)
            stats["tasks_kept"] += len(new)
            for c in new:
                assert set(c["true_label"]) <= set(c["labels"]), (path, c)
            repaired = json.loads(dumps_record(dict(rec, output=dict(rec["output"], classifications=new))))
            before = {k: v for k, v in rec["output"].items() if k != "classifications"}
            after = {k: v for k, v in repaired["output"].items() if k != "classifications"}
            if before != after or repaired["input"] != rec["input"]:
                raise SystemExit(f"[repair] {path}: a field other than classifications would change -- refusing")
            out.append(dumps_record(repaired))
        Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"[repair] {path}: {stats['records']} records, classification tasks asked {stats['tasks_asked']}, "
              f"kept {stats['tasks_kept']} (dropped {stats['tasks_asked'] - stats['tasks_kept']}: no same-menu majority)")


if __name__ == "__main__":
    main()
