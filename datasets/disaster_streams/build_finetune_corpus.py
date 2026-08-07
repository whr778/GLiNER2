"""Build a GLiNER2 structure-training corpus from realized disaster streams.

Each report snippet -> one InputExample whose casualty_report structure carries the GT
figure for each role, as the number appears in the text (verbatim substring -- comma
form matters, so '2,265' not '2265'). Trains the extractor to bind numbers to roles +
raise precision/recall (the bottleneck the probes identified). Output feeds
tools/train/train.py as a json_structures corpus.

  uv run python datasets/disaster_streams/build_finetune_corpus.py \
      --data datasets/disaster_streams_sonnet5 --split val --out data/casualty_ft.val.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from gliner2.training.data import InputExample, Structure, TrainingDataset

ROLES = ("dead", "injured", "missing")


def _find_span(value: int, text: str):
    """The number as it appears in the text (plain or comma-grouped), or None."""
    for cand in (str(value), f"{value:,}"):
        if cand in text:
            return cand
    return None


def build(split_dir: Path):
    groups = defaultdict(lambda: {"text": "", "gt": {}})
    for line in (split_dir / "observations.jsonl").open(encoding="utf-8"):
        o = json.loads(line)
        g = groups[(o["stream_id"], o["t_hours"])]
        g["text"] = o.get("text", ""); g["gt"][o["role"]] = o["value"]

    examples, located, missed = [], 0, 0
    for g in groups.values():
        text = g["text"]
        fields = {}
        for role, val in g["gt"].items():
            span = _find_span(val, text)
            if span is None:
                missed += 1
                continue
            located += 1
            fields[role] = span
        if fields and text:
            examples.append(InputExample(text=text, structures=[Structure("casualty_report", **fields)]))
    return examples, located, missed


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/disaster_streams_sonnet5")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="data/casualty_ft.val.jsonl")
    args = ap.parse_args(argv)

    examples, located, missed = build(Path(args.data) / args.split)
    ds = TrainingDataset(examples)
    report = ds.validate(raise_on_error=False)
    print(f"[build] {len(examples)} examples; fields located={located} missed={missed} "
          f"({located / max(located + missed, 1):.1%})")
    print(f"[build] validate report: {report}")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    ds.save(str(out))
    print(f"[build] wrote {out}")


if __name__ == "__main__":
    main()
