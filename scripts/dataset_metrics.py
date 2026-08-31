"""Dataset metrics for the GLiNER2 training corpora.

Scans ``data/*.jsonl``, groups files by dataset basename and split
(train / val / dev / test, or ``all`` for an unsplit ``<name>.jsonl``), and
reports for each (dataset, split): record count, input-length stats, and the
full label distributions for every task present -- entity labels, event types
(trigger + per-role counts), relation types, and classification tasks/labels.

Usage:
  uv run python scripts/dataset_metrics.py                 # every dataset
  uv run python scripts/dataset_metrics.py casie redocred  # a subset
  uv run python scripts/dataset_metrics.py --top 15        # cap long tails
  uv run python scripts/dataset_metrics.py --json m.json   # machine-readable
  uv run python scripts/dataset_metrics.py --data-dir data

Input length is counted with GLiNER2's ``WhitespaceTokenSplitter`` (the same
tokenizer the model uses), so counts match the model's view of the text and CJK
corpora split per-character rather than collapsing to ~1 word.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from gliner2.processor import WhitespaceTokenSplitter

SPLITS = ("train", "val", "dev", "test")
SPLIT_ORDER = {"train": 0, "val": 1, "dev": 2, "test": 3, "all": 4}

# GLiNER2's own tokenizer, so word counts match the model and CJK corpora
# (Chinese/Japanese/Korean) split per-character instead of collapsing to 1.
_tokenize = WhitespaceTokenSplitter()


def parse_name(path: Path) -> Tuple[str, str]:
    """``casie.train.jsonl`` -> ``(casie, train)``; ``docred.jsonl`` -> ``(docred, all)``."""
    stem = path.name[: -len(".jsonl")]
    dataset, _, split = stem.rpartition(".")
    if dataset and split in SPLITS:
        return dataset, split
    return stem, "all"


def scan_file(path: Path) -> Dict[str, Any]:
    """Accumulate record count, input-length stats, and per-task distributions."""
    records = 0
    words: List[int] = []
    ent: Counter = Counter()
    ev_trig: Counter = Counter()
    ev_roles: Dict[str, Counter] = defaultdict(Counter)
    rel: Counter = Counter()
    cls: Dict[str, Counter] = defaultdict(Counter)
    # Structures were not counted at all until 2026-08-31, which is how a corpus with
    # ZERO location supervision (casualty_docee: 25,154 records, not one `location`
    # field) was trained on four times without anyone seeing it. Per-FIELD coverage is
    # the number that matters: a record type present in every row still teaches nothing
    # about a field it never carries.
    struct: Dict[str, Counter] = defaultdict(Counter)
    struct_records: Counter = Counter()
    struct_anchors: Dict[str, Counter] = defaultdict(Counter)

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records += 1
            rec = json.loads(line)
            words.append(sum(1 for _ in _tokenize(rec.get("input") or "", lower=False)))
            out = rec.get("output") or {}

            for label, surfaces in (out.get("entities") or {}).items():
                ent[label] += len(surfaces or [])
            for ev in out.get("events") or []:
                etype = ev.get("event_type")
                if not etype:
                    continue
                ev_trig[etype] += len(ev.get("triggers") or [])
                for arg in ev.get("arguments") or []:
                    if arg.get("role"):
                        ev_roles[etype][arg["role"]] += 1
            for rel_d in out.get("relations") or []:
                for name in rel_d or {}:
                    rel[name] += 1
            for s in out.get("json_structures") or []:
                for name, fields in (s or {}).items():
                    struct_records[name] += 1
                    for field, value in (fields or {}).items():
                        if value not in (None, ""):
                            struct[name][field] += 1
            for name, meta in (out.get("record_metadata") or {}).items():
                struct_anchors[name][f"{meta.get('mode')}/{meta.get('anchor')}"] += 1

            for c in out.get("classifications") or []:
                task = c.get("task")
                if not task:
                    continue
                cls[task]  # touch so a task with no gold label still registers
                tl = c.get("true_label")
                if isinstance(tl, str):
                    tl = [tl]
                cls[task].update(tl or [])

    return {
        "records": records,
        "words": words,
        "entities": ent,
        "event_triggers": ev_trig,
        "event_roles": ev_roles,
        "relations": rel,
        "classifications": cls,
        "structures": struct,
        "structure_records": struct_records,
        "structure_anchors": struct_anchors,
    }


def total_annotations(m: Dict[str, Any]) -> int:
    return (
        sum(m["entities"].values())
        + sum(m["event_triggers"].values())
        + sum(sum(r.values()) for r in m["event_roles"].values())
        + sum(m["relations"].values())
        + sum(sum(c.values()) for c in m["classifications"].values())
    )


def word_stats(words: List[int]) -> Dict[str, float]:
    if not words:
        return {"avg": 0.0, "median": 0, "min": 0, "max": 0}
    return {
        "avg": round(statistics.mean(words), 1),
        "median": int(statistics.median(words)),
        "min": min(words),
        "max": max(words),
    }


def collect(data_dir: Path, wanted: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return ``{dataset: {split: metrics}}`` for every matching file."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for path in sorted(data_dir.glob("*.jsonl")):
        dataset, split = parse_name(path)
        if wanted and dataset not in wanted:
            continue
        out[dataset][split] = scan_file(path)
    return out


# ----- text report -----------------------------------------------------------


def _dist_lines(title: str, counts: Counter, top: int, indent: str = "    ") -> List[str]:
    items = counts.most_common()
    total = sum(counts.values())
    lines = [f"{indent}{title} ({len(counts)} labels, {total} mentions):"]
    shown = items[:top] if top else items
    width = max((len(k) for k, _ in shown), default=0)
    for name, n in shown:
        lines.append(f"{indent}  {name.ljust(width)}  {n}")
    if top and len(items) > top:
        lines.append(f"{indent}  ... (+{len(items) - top} more)")
    return lines


def report_dataset(dataset: str, splits: Dict[str, Dict[str, Any]], top: int) -> List[str]:
    lines = [f"=== {dataset} ==="]
    for split in sorted(splits, key=lambda s: SPLIT_ORDER.get(s, 9)):
        m = splits[split]
        w = word_stats(m["words"])
        lines.append(
            f"[{split}] {m['records']} records | words avg {w['avg']} "
            f"median {w['median']} (min {w['min']}, max {w['max']}) "
            f"| {total_annotations(m)} annotations"
        )
        if m["entities"]:
            lines += _dist_lines("entities", m["entities"], top)
        if m["event_triggers"]:
            lines.append(f"    events ({len(m['event_triggers'])} types):")
            order = m["event_triggers"].most_common()
            for etype, trig in (order[:top] if top else order):
                roles = m["event_roles"].get(etype, Counter())
                role_str = ", ".join(f"{r}={n}" for r, n in roles.most_common())
                lines.append(f"      {etype}  triggers={trig}  roles: {role_str or '-'}")
            if top and len(order) > top:
                lines.append(f"      ... (+{len(order) - top} more)")
        if m["relations"]:
            lines += _dist_lines("relations", m["relations"], top)
        if m["classifications"]:
            lines.append(f"    classifications ({len(m['classifications'])} tasks):")
            for task, labels in sorted(m["classifications"].items()):
                lines += _dist_lines(f"task '{task}'", labels, top, indent="      ")
        if m.get("structure_records"):
            lines.append(f"    structures ({len(m['structure_records'])} record types):")
            for name, n in sorted(m["structure_records"].items()):
                lines.append(f"      {name}: {n:,} records")
                # Per-field COVERAGE, not just presence. A field at 0% is a field the
                # model cannot learn, and it is invisible in a record-level count.
                for field, c in sorted(m["structures"].get(name, {}).items(),
                                       key=lambda kv: -kv[1]):
                    lines.append(f"        {field:<14} {c:>8,}  {c / n:6.1%}")
                missing = [f for f in ("location", "dead", "injured", "missing")
                           if name == "casualty_report" and f not in m["structures"].get(name, {})]
                for f in missing:
                    lines.append(f"        {f:<14} {0:>8,}  {0.0:6.1%}   <-- NO SUPERVISION")
                anchors = m.get("structure_anchors", {}).get(name)
                if anchors:
                    shown = ", ".join(f"{k} x{v:,}" for k, v in anchors.most_common(4))
                    lines.append(f"        record_metadata: {shown}")
                else:
                    lines.append("        record_metadata: NONE  <-- boundary models "
                                 "cannot decode this; extraction returns {} silently")
    return lines


def summary_table(data: Dict[str, Dict[str, Dict[str, Any]]]) -> List[str]:
    header = ["dataset", "split", "recs", "words", "ent", "ev", "rel", "cls", "annots"]
    rows = [header]
    for dataset in sorted(data):
        for split in sorted(data[dataset], key=lambda s: SPLIT_ORDER.get(s, 9)):
            m = data[dataset][split]
            rows.append([
                dataset, split, str(m["records"]),
                str(word_stats(m["words"])["median"]),
                str(len(m["entities"])), str(len(m["event_triggers"])),
                str(len(m["relations"])), str(len(m["classifications"])),
                str(total_annotations(m)),
            ])
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    out = []
    for i, r in enumerate(rows):
        out.append("  ".join(c.ljust(widths[j]) for j, c in enumerate(r)))
        if i == 0:
            out.append("  ".join("-" * widths[j] for j in range(len(header))))
    return out


# ----- json report -----------------------------------------------------------


def to_json(data: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for dataset, splits in data.items():
        out[dataset] = {}
        for split, m in splits.items():
            out[dataset][split] = {
                "records": m["records"],
                "words": word_stats(m["words"]),
                "annotations": total_annotations(m),
                "entities": dict(m["entities"].most_common()),
                "events": {
                    et: {"triggers": trig, "roles": dict(m["event_roles"].get(et, Counter()))}
                    for et, trig in m["event_triggers"].most_common()
                },
                "relations": dict(m["relations"].most_common()),
                "classifications": {t: dict(c.most_common()) for t, c in m["classifications"].items()},
            }
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Report label distributions for the data/ corpora.")
    p.add_argument("datasets", nargs="*", help="Limit to these dataset basenames (default: all).")
    p.add_argument("--data-dir", default="data", help="Directory of *.jsonl corpora (default: data).")
    p.add_argument("--top", type=int, default=0, help="Cap each distribution to its top-N labels (0 = all).")
    p.add_argument("--json", dest="json_out", help="Write metrics as JSON here instead of a text report.")
    args = p.parse_args(argv)

    data = collect(Path(args.data_dir), args.datasets)
    if not data:
        raise SystemExit(f"No matching *.jsonl in {args.data_dir}.")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(to_json(data), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[out] wrote {args.json_out} ({len(data)} datasets)")
        return

    print("\n".join(summary_table(data)))
    print()
    for dataset in sorted(data):
        print("\n".join(report_dataset(dataset, data[dataset], args.top)))
        print()


if __name__ == "__main__":
    main()
