"""Dataset metrics for the GLiNER2 training corpora.

Two modes:

* glob mode scans ``data/*.jsonl`` and groups files by dataset basename and split
  (train / val / dev / test, or ``all`` for an unsplit ``<name>.jsonl``);
* ``--config`` mode resolves train / val / test EXACTLY as ``tools/train/train.py`` does
  (``corpora`` + ``event_files`` + ``train_only``, deduplicated) and applies the config's
  label map first, so every count is what the model is trained and scored on.

Per (corpus, split) it reports records, input length, and for each head:

* NER -- mentions, labels, singleton labels, records carrying any;
* relations -- instances, types, records carrying any;
* classification -- per task, label counts and the majority-class share;
* structures -- per-field coverage and the record_metadata anchors;
* events -- instances, triggerless instances (they can never score an argument),
  trigger and argument gold counted with the EVALUATOR'S OWN key sets
  (``_gold_event_trigger_set`` / ``_gold_event_argument_set``), arguments per instance,
  and trigger length.

``--lang`` adds the per-record language (``train._detect_lang``, what ``eval_by_language``
uses, ~1 ms per record). Config mode also prints a rollup of the TRAIN split by language
and by provenance -- who wrote the gold, from ``dataset_registry.yaml`` -- and a
cross-split uniqueness check on the input text.

Usage:
  uv run python scripts/dataset_metrics.py                 # every dataset
  uv run python scripts/dataset_metrics.py casie redocred  # a subset
  uv run python scripts/dataset_metrics.py --top 15        # cap long tails
  uv run python scripts/dataset_metrics.py --json m.json   # machine-readable
  uv run python scripts/dataset_metrics.py --config tools/train/config/base/eb17-best.yaml --lang

Input length is counted with GLiNER2's ``WhitespaceTokenSplitter`` (the same
tokenizer the model uses), so counts match the model's view of the text and CJK
corpora split per-character rather than collapsing to ~1 word.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import yaml

from gliner2.processor import WhitespaceTokenSplitter
from gliner2.training.eval_metrics import _gold_event_argument_set, _gold_event_trigger_set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "train"))
import train as T  # noqa: E402
from model_card import load_registry  # noqa: E402

SPLITS = ("train", "val", "dev", "test")
SPLIT_ORDER = {"train": 0, "val": 1, "dev": 2, "test": 3, "all": 4}
HEADS = ("entities", "relations", "classifications", "structures", "events")

# GLiNER2's own tokenizer, so word counts match the model and CJK corpora
# (Chinese/Japanese/Korean) split per-character instead of collapsing to 1.
_tokenize = WhitespaceTokenSplitter()


def n_words(text: str) -> int:
    return sum(1 for _ in _tokenize(text or "", lower=False))


def parse_name(path: Path) -> Tuple[str, str]:
    """``casie.train.jsonl`` -> ``(casie, train)``; ``docred.jsonl`` -> ``(docred, all)``."""
    stem = path.name[: -len(".jsonl")]
    dataset, _, split = stem.rpartition(".")
    if dataset and split in SPLITS:
        return dataset, split
    return stem, "all"


def corpus_key(path: Path, data_dir: Path = Path("data")) -> str:
    """Dataset name, prefixed by its subdirectory so ``scaling_joint/cmnee`` != ``cmnee``."""
    name = parse_name(path)[0]
    parent = path.parent
    return name if parent == data_dir else f"{parent.relative_to(data_dir)}/{name}"


def read_records(path: Path, fns: Optional[Dict] = None) -> Iterable[Dict]:
    """Yield the records of one JSONL file, label-mapped when ``fns`` is given."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                yield T.transform_record(rec, fns) if fns else rec


# ----- scanning --------------------------------------------------------------


def new_metrics() -> Dict[str, Any]:
    return {
        "records": 0, "words": [], "hashes": set(), "languages": Counter(),
        "entities": Counter(), "ent_records": 0,
        "relations": Counter(), "rel_records": 0,
        "classifications": defaultdict(Counter),
        "structures": defaultdict(Counter), "structure_records": Counter(),
        "structure_anchors": defaultdict(Counter),
        "event_triggers": Counter(), "event_roles": defaultdict(Counter),
        "ev_records": 0, "ev_instances": 0, "ev_triggerless": 0,
        "trigger_gold": 0, "argument_gold": 0, "args_per_instance": [],
        "trigger_words": [],
    }


def add_entities(m: Dict, out: Dict) -> None:
    ents = out.get("entities") or {}
    for label, surfaces in ents.items():
        m["entities"][label] += len(surfaces or [])
    m["ent_records"] += any(ents.values())


def add_relations(m: Dict, out: Dict) -> None:
    rels = out.get("relations") or []
    for rel_d in rels:
        for name in rel_d or {}:
            m["relations"][name] += 1
    m["rel_records"] += bool(rels)


def add_classifications(m: Dict, out: Dict) -> None:
    for c in out.get("classifications") or []:
        task = c.get("task")
        if not task:
            continue
        m["classifications"][task]  # touch so a task with no gold label still registers
        tl = c.get("true_label")
        m["classifications"][task].update([tl] if isinstance(tl, str) else tl or [])


def add_structures(m: Dict, out: Dict) -> None:
    # Per-FIELD coverage is the number that matters: casualty_docee trained four times
    # with ZERO `location` supervision, invisible in a record-level count.
    for s in out.get("json_structures") or []:
        for name, fields in (s or {}).items():
            m["structure_records"][name] += 1
            for field, value in (fields or {}).items():
                if value not in (None, ""):
                    m["structures"][name][field] += 1
    for name, meta in (out.get("record_metadata") or {}).items():
        m["structure_anchors"][name][f"{meta.get('mode')}/{meta.get('anchor')}"] += 1


def add_events(m: Dict, out: Dict) -> None:
    """Count event gold with the evaluator's own key sets, so the tool and eval agree."""
    events = [ev for ev in out.get("events") or [] if ev.get("event_type")]
    for ev in events:
        triggers = [t for t in ev.get("triggers") or [] if isinstance(t, str) and t.strip()]
        m["event_triggers"][ev["event_type"]] += len(triggers)
        m["ev_triggerless"] += not triggers
        m["args_per_instance"].append(len(ev.get("arguments") or []))
        m["trigger_words"].extend(n_words(t) for t in triggers)
        for arg in ev.get("arguments") or []:
            if arg.get("role"):
                m["event_roles"][ev["event_type"]][arg["role"]] += 1
    m["ev_instances"] += len(events)
    m["ev_records"] += bool(events)
    m["trigger_gold"] += len(_gold_event_trigger_set(out))
    m["argument_gold"] += len(_gold_event_argument_set(out))


def add_record(m: Dict, rec: Dict, lang: Optional[Callable[[str], str]]) -> None:
    text = rec.get("input") or ""
    m["records"] += 1
    m["words"].append(n_words(text))
    m["hashes"].add(hashlib.sha1(text.strip().encode("utf-8")).hexdigest())
    if lang:
        m["languages"][lang(text.strip()) if text.strip() else "und"] += 1
    out = rec.get("output") or {}
    for add in (add_entities, add_relations, add_classifications, add_structures, add_events):
        add(m, out)


def scan_file(path: Path, fns: Optional[Dict] = None,
              lang: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    """Accumulate record count, input length, per-head distributions and gold counts."""
    m = new_metrics()
    for rec in read_records(path, fns):
        add_record(m, rec, lang)
    return m


def collect(data_dir: Path, wanted: List[str], lang=None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return ``{dataset: {split: metrics}}`` for every matching file."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for path in sorted(data_dir.glob("*.jsonl")):
        dataset, split = parse_name(path)
        if wanted and dataset not in wanted:
            continue
        out[dataset][split] = scan_file(path, lang=lang)
    return out


def config_paths(cfg: Dict) -> Dict[str, List[str]]:
    """Resolve train / val / test exactly as ``train.py`` main() does."""
    d = cfg.get("data") or {}
    corpora, event_files = d.get("corpora") or [], d.get("event_files") or {}
    train_only = set(d.get("train_only") or ())
    return {
        s: T._dedupe_paths(T._split_files(corpora, s, train_only)
                           + T._event_split(event_files, s), s)
        for s in ("train", "val", "test")
    }


def collect_config(cfg_path: str, lang=None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return ``{corpus: {split: metrics}}`` for a training config, label map applied."""
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    fns = T._category_fns(T.load_labels_cfg(cfg, cfg_path))
    passthrough = T.labels_passthrough(cfg)
    out: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for split, paths in config_paths(cfg).items():
        for p in paths:
            mapped = None if T._corpus_of(p) in passthrough else fns
            out[corpus_key(Path(p))][split] = scan_file(Path(p), mapped, lang)
    return out


# ----- derived numbers -------------------------------------------------------


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


def head_counts(m: Dict[str, Any]) -> Dict[str, int]:
    """One gold count per head, the unit each head's F1 is scored on."""
    return {
        "entities": sum(m["entities"].values()),
        "relations": sum(m["relations"].values()),
        "classifications": sum(sum(c.values()) for c in m["classifications"].values()),
        "structures": sum(m["structure_records"].values()),
        "event_triggers": m["trigger_gold"],
        "event_arguments": m["argument_gold"],
    }


def provenance(registry: Dict, corpus: str, head: str) -> str:
    """Who wrote this corpus's gold for this head, or ``unknown`` if unverified."""
    value = (registry.get(corpus.split("/")[-1]) or {}).get("provenance")
    if isinstance(value, dict):
        return value.get(head, "unknown")
    return value or "unknown"


def dominant_language(m: Dict[str, Any]) -> str:
    """The corpus's most common DETERMINED language.

    ``und`` is skipped because it names no language. lumi 2.0.0 returned it on
    plainly Chinese text and filed 24,859 Chinese argument gold under ``und``; 3.0.0
    fixed that, but short text can still be legitimately ``und``.
    """
    known = [k for k, _ in m["languages"].most_common() if k != "und"]
    return known[0] if known else "?"


def uniqueness(data: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, int]:
    """Count records, distinct inputs per split, and the overlaps between splits.

    records > unique inputs means the same text appears more than once in a split,
    within or across its files.
    """
    by_split: Dict[str, set] = defaultdict(set)
    records: Counter = Counter()
    for splits in data.values():
        for split, m in splits.items():
            by_split[split] |= m["hashes"]
            records[split] += m["records"]
    out: Dict[str, int] = {}
    for s, h in by_split.items():
        out[f"{s}_records"] = records[s]
        out[f"{s}_unique_inputs"] = len(h)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        out[f"{a}&{b}"] = len(by_split.get(a, set()) & by_split.get(b, set()))
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


def _ner_lines(m: Dict, top: int) -> List[str]:
    singletons = sum(1 for n in m["entities"].values() if n == 1)
    lines = [f"    NER: {m['ent_records']:,} of {m['records']:,} records carry entities; "
             f"{singletons} of {len(m['entities'])} labels used once"]
    return lines + _dist_lines("entities", m["entities"], top)


def _relation_lines(m: Dict, top: int) -> List[str]:
    lines = [f"    relations: {m['rel_records']:,} of {m['records']:,} records carry relations"]
    return lines + _dist_lines("relations", m["relations"], top)


def _classification_lines(m: Dict, top: int) -> List[str]:
    lines = [f"    classifications ({len(m['classifications'])} tasks):"]
    for task, labels in sorted(m["classifications"].items()):
        total = sum(labels.values())
        major = labels.most_common(1)[0][1] / total if total else 0.0
        lines.append(f"      task '{task}': majority class {major:.1%} of {total:,}")
        lines += _dist_lines(f"task '{task}'", labels, top, indent="      ")
    return lines


def _structure_lines(m: Dict) -> List[str]:
    lines = [f"    structures ({len(m['structure_records'])} record types):"]
    for name, n in sorted(m["structure_records"].items()):
        lines.append(f"      {name}: {n:,} records")
        for field, c in sorted(m["structures"].get(name, {}).items(), key=lambda kv: -kv[1]):
            lines.append(f"        {field:<14} {c:>8,}  {c / n:6.1%}")
        missing = [f for f in ("location", "dead", "injured", "missing")
                   if name == "casualty_report" and f not in m["structures"].get(name, {})]
        for f in missing:
            lines.append(f"        {f:<14} {0:>8,}  {0.0:6.1%}   <-- NO SUPERVISION")
        anchors = m["structure_anchors"].get(name)
        if anchors:
            shown = ", ".join(f"{k} x{v:,}" for k, v in anchors.most_common(4))
            lines.append(f"        record_metadata: {shown}")
        else:
            lines.append("        record_metadata: NONE  <-- boundary models "
                         "cannot decode this; extraction returns {} silently")
    return lines


def _event_lines(m: Dict, top: int) -> List[str]:
    tw, api = m["trigger_words"], m["args_per_instance"]
    single = sum(1 for w in tw if w == 1) / len(tw) if tw else 0.0
    lines = [
        f"    events: {m['ev_records']:,} of {m['records']:,} records, "
        f"{m['ev_instances']:,} instances ({m['ev_triggerless']:,} triggerless), "
        f"{len(m['event_triggers'])} types",
        f"      gold as scored: trigger {m['trigger_gold']:,}  argument {m['argument_gold']:,}  "
        f"| args/instance mean {statistics.mean(api) if api else 0:.2f}  "
        f"| trigger words mean {statistics.mean(tw) if tw else 0:.2f}, single-word {single:.1%}",
    ]
    if m["argument_gold"] == 0:
        lines.append("      TRIGGER-ONLY: this corpus cannot teach or score event arguments")
    order = m["event_triggers"].most_common()
    for etype, trig in (order[:top] if top else order):
        roles = m["event_roles"].get(etype, Counter())
        role_str = ", ".join(f"{r}={n}" for r, n in roles.most_common())
        lines.append(f"      {etype}  triggers={trig}  roles: {role_str or '-'}")
    if top and len(order) > top:
        lines.append(f"      ... (+{len(order) - top} more)")
    return lines


def report_dataset(dataset: str, splits: Dict[str, Dict[str, Any]], top: int) -> List[str]:
    lines = [f"=== {dataset} ==="]
    for split in sorted(splits, key=lambda s: SPLIT_ORDER.get(s, 9)):
        m = splits[split]
        w = word_stats(m["words"])
        langs = ", ".join(f"{k} {v / m['records']:.1%}" for k, v in m["languages"].most_common(3))
        lines.append(
            f"[{split}] {m['records']} records | words avg {w['avg']} "
            f"median {w['median']} (min {w['min']}, max {w['max']}) "
            f"| {total_annotations(m)} annotations" + (f" | lang {langs}" if langs else "")
        )
        if m["entities"]:
            lines += _ner_lines(m, top)
        if m["ev_instances"]:
            lines += _event_lines(m, top)
        if m["relations"]:
            lines += _relation_lines(m, top)
        if m["classifications"]:
            lines += _classification_lines(m, top)
        if m["structure_records"]:
            lines += _structure_lines(m)
    return lines


def _table(rows: List[List[str]]) -> List[str]:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for i, r in enumerate(rows):
        out.append("  ".join(c.ljust(widths[j]) for j, c in enumerate(r)))
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return out


def summary_table(data: Dict[str, Dict[str, Dict[str, Any]]]) -> List[str]:
    rows = [["dataset", "split", "lang", "recs", "words", "ent", "rel", "cls",
             "struct", "ev_trig", "ev_arg"]]
    for dataset in sorted(data):
        for split in sorted(data[dataset], key=lambda s: SPLIT_ORDER.get(s, 9)):
            m = data[dataset][split]
            h = head_counts(m)
            rows.append([
                dataset, split, dominant_language(m), f"{m['records']:,}",
                str(word_stats(m["words"])["median"]), f"{h['entities']:,}",
                f"{h['relations']:,}", f"{h['classifications']:,}", f"{h['structures']:,}",
                f"{h['event_triggers']:,}", f"{h['event_arguments']:,}",
            ])
    return _table(rows)


def rollup(data: Dict[str, Dict[str, Dict[str, Any]]], registry: Dict,
           split: str = "train") -> Dict[str, Dict[str, Counter]]:
    """Gold per head in one split, summed by language and by provenance.

    Each corpus counts under its dominant language (every corpus here is monolingual
    by construction; per-record detection is too noisy to apportion). Provenance is per
    corpus and per head, from the registry.
    """
    by_lang: Dict[str, Counter] = defaultdict(Counter)
    by_prov: Dict[str, Counter] = defaultdict(Counter)
    for corpus, splits in data.items():
        m = splits.get(split)
        if not m:
            continue
        for head, n in head_counts(m).items():
            reg_head = "events" if head.startswith("event_") else head
            by_prov[head][provenance(registry, corpus, reg_head)] += n
            by_lang[head][dominant_language(m)] += n
    return {"language": by_lang, "provenance": by_prov}


def rollup_lines(roll: Dict[str, Dict[str, Counter]], split: str) -> List[str]:
    lines = []
    for axis, by_head in roll.items():
        lines.append(f"=== {split} gold by {axis} (share of each head) ===")
        keys = sorted({k for c in by_head.values() for k in c})
        rows = [["head"] + keys]
        for head, c in by_head.items():
            total = sum(c.values()) or 1
            rows.append([head] + [f"{c[k]:,.0f} ({c[k] / total:.0%})" for k in keys])
        lines += _table(rows) + [""]
    return lines


# ----- json report -----------------------------------------------------------


def to_json(data: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for dataset, splits in data.items():
        out[dataset] = {}
        for split, m in splits.items():
            out[dataset][split] = {
                "records": m["records"],
                "words": word_stats(m["words"]),
                "languages": dict(m["languages"]),
                "annotations": total_annotations(m),
                "gold": head_counts(m),
                "events_detail": {
                    "records": m["ev_records"], "instances": m["ev_instances"],
                    "triggerless": m["ev_triggerless"],
                },
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
    p.add_argument("--config", help="Measure a training config's resolved splits, label map applied.")
    p.add_argument("--lang", action="store_true", help="Detect each record's language (~1 ms/record).")
    p.add_argument("--top", type=int, default=0, help="Cap each distribution to its top-N labels (0 = all).")
    p.add_argument("--json", dest="json_out", help="Write metrics as JSON here instead of a text report.")
    args = p.parse_args(argv)

    lang = T._detect_lang if args.lang else None
    if args.config:
        data = collect_config(args.config, lang)
    else:
        data = collect(Path(args.data_dir), args.datasets, lang)
    if not data:
        raise SystemExit(f"No matching *.jsonl in {args.data_dir}.")

    if args.json_out:
        report = to_json(data)
        if args.config:
            roll = rollup(data, load_registry()["datasets"])
            report["_train_rollup"] = {a: {h: dict(c) for h, c in r.items()} for a, r in roll.items()}
            report["_uniqueness"] = uniqueness(data)
        Path(args.json_out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        print(f"[out] wrote {args.json_out} ({len(data)} datasets)")
        return

    print("\n".join(summary_table(data)))
    print()
    if args.config:
        print("\n".join(rollup_lines(rollup(data, load_registry()["datasets"]), "train")))
        print("=== cross-split uniqueness (sha1 of input) ===")
        for k, v in uniqueness(data).items():
            print(f"  {k:24} {v:,}")
        print()
    for dataset in sorted(data):
        print("\n".join(report_dataset(dataset, data[dataset], args.top)))
        print()


if __name__ == "__main__":
    main()
