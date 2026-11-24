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
  uv run python scripts/dataset_metrics.py --markdown census.md casie
  uv run python scripts/dataset_metrics.py --config tools/train/config/base/eb17-best.yaml --lang --markdown census.md

``--markdown`` writes one file: the census above, plus the dataset diagnostics that
used to live in their own commands. Label shift is TVD (total variation distance)
between splits. Argument-inside-trigger is the span count from
``tools/data/measure_argument_inside_trigger.py``, with equal spans, partial
overlaps, and a trigger sitting inside a longer argument broken out. String
occurrence is ``measure_occurrence_ambiguity.py`` over every record, not a sample.
Event identity is ``measure_event_normalisation.py``: multi-trigger events, and
same-type pairs that share an argument. Coreference is three counts from the gold:
mention clusters on ``coref_clusters``, trigger pairs inside one event (TriggerLink's
positives) against same-type pairs across events, and the argument mentions
``expand_argument_coreferents`` would add. Checkpoint measures (surface alignment
against the model tokenizer, gradient share, the junction) stay in their own
commands: they describe a model, not the gold.

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
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import yaml

from gliner2.inference.entity_coref import PRONOUNS
from gliner2.processing.argument_coref import expand_argument_coreferents
from gliner2.processor import WhitespaceTokenSplitter
from gliner2.training.eval_metrics import _gold_event_argument_set, _gold_event_trigger_set

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tools" / "train"))
sys.path.insert(0, str(_ROOT / "tools" / "data"))
import train as T  # noqa: E402
from model_card import load_registry  # noqa: E402
from adapt_ace2005_office import adapt_record, is_office_record  # noqa: E402
from compare_label_distributions import tvd  # noqa: E402
from measure_argument_inside_trigger import contained, measure_record, span_of  # noqa: E402

# Same gates as tools/derive/data_health.py.
TVD_WARN, TVD_STRONG = 0.15, 0.30
THIN_SUPPORT = 10

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
        # Span geometry of an argument against its own event's triggers. ``arg_place``
        # is the measure_argument_inside_trigger total (inside / outside / unmeasured).
        # The five counts under it partition the placed arguments.
        "arg_place": Counter(), "arg_place_role": defaultdict(Counter),
        "arg_equal": 0, "arg_nested": 0, "arg_trigger_inside": 0,
        "arg_partial": 0, "arg_disjoint": 0, "office_adapted": 0,
        # String occurrence under the whitespace splitter. Offset gold is still counted,
        # and ``occ_*_offset`` says how many of those surfaces carry a span.
        "occ_trig_sum": 0, "occ_trig_placed": 0, "occ_trig_gt1": 0, "occ_trig_unplaced": 0,
        "occ_trig_offset": 0, "occ_trig_string": 0,
        "occ_arg_sum": 0, "occ_arg_placed": 0, "occ_arg_gt1": 0, "occ_arg_unplaced": 0,
        "occ_arg_offset": 0, "occ_arg_string": 0,
        # Event identity. Shared-argument pairs are an upper bound: two real events can
        # share a participant.
        "id_docs": 0, "id_events": 0, "id_multi": 0, "id_pairs": 0,
        "id_share_text": 0, "id_share_role": 0, "id_nested": 0,
        "id_identical": 0, "id_split_docs": 0,
        # coref_clusters, trigger pairs, and what argument expansion would add.
        "coref_docs": 0, "coref_clusters": 0, "coref_mentions": 0,
        "coref_multi": 0, "coref_singletons": 0, "coref_mention_pairs": 0,
        "coref_pronouns": 0,
        "trig_coref_events": 0, "trig_coref_pos": 0, "trig_coref_cross": 0,
        "arg_coref_added": 0, "arg_coref_skipped": 0,
        "arg_span_coref": 0, "arg_span_others": 0, "arg_span_ambiguous": 0,
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


def _surface(value) -> str:
    if isinstance(value, dict):
        return (value.get("text") or "").strip()
    return (value or "").strip() if isinstance(value, str) else ""


def _occurrence(words: List[str], surface: str) -> int:
    """Places ``surface`` lands in an already-tokenized document. Same match as
    ``measure_occurrence_ambiguity.occurrences``: lowercased whitespace tokens."""
    sub = [t for t, _, _ in _tokenize(surface, lower=True)]
    n = len(sub)
    if not n:
        return 0
    return sum(words[i:i + n] == sub for i in range(len(words) - n + 1))


def _tally_surface(m: Dict, kind: str, words: List[str], value) -> None:
    surface = _surface(value)
    if not surface:
        return
    offset = span_of(value) is not None
    m[f"occ_{kind}_offset" if offset else f"occ_{kind}_string"] += 1
    n = _occurrence(words, surface)
    if n == 0:
        m[f"occ_{kind}_unplaced"] += 1
        return
    m[f"occ_{kind}_placed"] += 1
    m[f"occ_{kind}_sum"] += n
    m[f"occ_{kind}_gt1"] += n > 1


def _arg_text(arg: Dict) -> str:
    return _surface(arg.get("entity") if isinstance(arg, dict) else None).lower()


def _arg_pair(arg: Dict) -> Tuple[str, str]:
    role = arg.get("role") if isinstance(arg, dict) else None
    return (role or "", _arg_text(arg))


def _pairs(n: int) -> int:
    return n * (n - 1) // 2


def add_coref(m: Dict, rec: Dict, events: List[Dict]) -> None:
    """Mention clusters, trigger pairs, and argument mentions expansion would add.

    TriggerLink's positive is two triggers of one event. A same-type pair drawn from
    two events is a gold hard negative. ``expand_argument_coreferents`` is the string
    rule training uses; a span in exactly one multi-mention cluster is the offset rule,
    which still sees a pronoun that sits in two clusters under its surface.
    """
    clusters = [c for c in (rec.get("coref_clusters") or []) if isinstance(c, list)]
    if clusters:
        m["coref_docs"] += 1
    span_at: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        mentions = [x for x in cluster if isinstance(x, dict)]
        if not mentions:
            continue
        m["coref_clusters"] += 1
        m["coref_mentions"] += len(mentions)
        m["coref_multi"] += len(mentions) > 1
        m["coref_singletons"] += len(mentions) == 1
        m["coref_mention_pairs"] += _pairs(len(mentions))
        for mention in mentions:
            if (mention.get("text") or "").strip().casefold() in PRONOUNS:
                m["coref_pronouns"] += 1
            span = span_of(mention)
            if span:
                span_at[span].append(index)
    by_type: Dict[str, List[int]] = defaultdict(list)
    for event in events:
        triggers = [t for t in event.get("triggers") or [] if _surface(t)]
        by_type[event["event_type"]].append(len(triggers))
        if len(triggers) > 1:
            m["trig_coref_events"] += 1
            m["trig_coref_pos"] += _pairs(len(triggers))
    for counts in by_type.values():
        m["trig_coref_cross"] += _pairs(sum(counts)) - sum(_pairs(k) for k in counts)
    for event in events:
        for argument in event.get("arguments") or []:
            if not isinstance(argument, dict):
                continue
            span = span_of(argument.get("entity"))
            hits = set(span_at.get(span, ())) if span else set()
            if len(hits) > 1:
                m["arg_span_ambiguous"] += 1
                continue
            if len(hits) != 1:
                continue
            cluster = clusters[next(iter(hits))]
            others = [x for x in cluster if isinstance(x, dict) and span_of(x) != span]
            if others:
                m["arg_span_coref"] += 1
                m["arg_span_others"] += len(others)
    output = rec.get("output") if isinstance(rec.get("output"), dict) else None
    text = rec.get("input")
    if clusters and events and isinstance(output, dict) and isinstance(text, str):
        expanded = expand_argument_coreferents(output, clusters, text)
        m["arg_coref_added"] += expanded.get("_argument_coref_added") or 0
        m["arg_coref_skipped"] += expanded.get("_argument_coref_skipped") or 0


def add_diagnostics(m: Dict, rec: Dict) -> None:
    """Span geometry, string occurrence, and event identity for one record.

    Span geometry follows ``measure_argument_inside_trigger``: an office ACE record
    is adapted first, and an equal span counts as inside. The five placed counts
    partition inside (equal, nested) and outside (trigger inside the argument,
    partial overlap, disjoint).
    """
    source = rec
    if is_office_record(rec):
        source = adapt_record(rec)
        m["office_adapted"] += 1
    part, roles = measure_record(source)
    m["arg_place"].update(part)
    for role, counts in roles.items():
        m["arg_place_role"][role].update(counts)
    for event in (source.get("output") or {}).get("events") or []:
        if not isinstance(event, dict):
            continue
        triggers = [s for s in (span_of(t) for t in event.get("triggers") or []) if s]
        for argument in event.get("arguments") or []:
            if not isinstance(argument, dict):
                continue
            argument_span = span_of(argument.get("entity"))
            if argument_span is None or not triggers:
                continue
            if any(argument_span == trigger for trigger in triggers):
                m["arg_equal"] += 1
            elif any(contained(argument_span, trigger) for trigger in triggers):
                m["arg_nested"] += 1
            elif any(contained(trigger, argument_span) for trigger in triggers):
                m["arg_trigger_inside"] += 1
            elif any(argument_span[0] < trigger[1] and trigger[0] < argument_span[1]
                     for trigger in triggers):
                m["arg_partial"] += 1
            else:
                m["arg_disjoint"] += 1

    text = rec.get("input") or ""
    events = [e for e in (rec.get("output") or {}).get("events") or []
              if isinstance(e, dict) and e.get("event_type")]
    add_coref(m, rec, events)
    if not events:
        return
    words = [t for t, _, _ in _tokenize(text, lower=True)]
    for event in events:
        for trigger in event.get("triggers") or []:
            _tally_surface(m, "trig", words, trigger)
        for argument in event.get("arguments") or []:
            _tally_surface(m, "arg", words, (argument or {}).get("entity") if isinstance(argument, dict) else None)
    m["id_docs"] += 1
    m["id_events"] += len(events)
    m["id_multi"] += sum(
        len([t for t in event.get("triggers") or [] if _surface(t)]) > 1 for event in events)
    split_doc = False
    for left, right in combinations(events, 2):
        if left["event_type"] != right["event_type"]:
            continue
        m["id_pairs"] += 1
        left_text = {_arg_text(a) for a in left.get("arguments") or [] if _arg_text(a)}
        right_text = {_arg_text(a) for a in right.get("arguments") or [] if _arg_text(a)}
        left_pairs = {_arg_pair(a) for a in left.get("arguments") or [] if _arg_text(a)}
        right_pairs = {_arg_pair(a) for a in right.get("arguments") or [] if _arg_text(a)}
        if left_text & right_text:
            m["id_share_text"] += 1
            split_doc = True
        if left_pairs & right_pairs:
            m["id_share_role"] += 1
        if left_text and right_text and (left_text <= right_text or right_text <= left_text):
            m["id_nested"] += 1
        if left_text and left_text == right_text:
            m["id_identical"] += 1
    m["id_split_docs"] += split_doc


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
    add_diagnostics(m, rec)


def scan_file(path: Path, fns: Optional[Dict] = None,
              lang: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    """Accumulate record count, input length, per-head distributions and gold counts."""
    m = new_metrics()
    for rec in read_records(path, fns):
        add_record(m, rec, lang)
    return m


def load_label_fns(path: str) -> Dict:
    """Label transforms from a unified-map file (``labels:`` block, or the bare categories)."""
    block = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return T._category_fns(block.get("labels", block))


def collect(data_dir: Path, wanted: List[str], lang=None,
            fns: Optional[Dict] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return ``{dataset: {split: metrics}}`` for every matching file, label-mapped if ``fns``."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for path in sorted(data_dir.glob("*.jsonl")):
        dataset, split = parse_name(path)
        if wanted and dataset not in wanted:
            continue
        out[dataset][split] = scan_file(path, fns, lang)
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


def collect_config(cfg_path: str, lang=None, labels: Optional[str] = None
                   ) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return ``{corpus: {split: metrics}}`` for a training config.

    Labels are mapped with the config's own map (as training sees them) unless ``labels``
    names another map file, or is ``"none"`` for the raw labels.
    """
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    if labels == "none":
        fns = None
    elif labels:
        fns = load_label_fns(labels)
    else:
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


def _md(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _md_table(headers: List[str], rows: List[List]) -> List[str]:
    head = "| " + " | ".join(_md(h) for h in headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_md(c) for c in row) + " |" for row in rows]
    return [head, rule, *body]


def _share(n: int, d: int) -> str:
    return f"{n / d:.2%}" if d else "n/a"


def _percentile(words: List[int], p: float) -> int:
    if not words:
        return 0
    xs = sorted(words)
    index = min(len(xs) - 1, max(0, int(round(p / 100 * (len(xs) - 1)))))
    return xs[index]


def _marginals(m: Dict) -> List[Tuple[str, Counter]]:
    roles: Counter = Counter()
    for by_role in m["event_roles"].values():
        roles.update(by_role)
    classes: Counter = Counter()
    for task, labels in m["classifications"].items():
        for lab, n in labels.items():
            classes[f"{task}: {lab}"] += n
    return [
        ("entities (per mention)", m["entities"]),
        ("relations (per instance)", m["relations"]),
        ("event types (per trigger mention)", m["event_triggers"]),
        ("event roles (per argument)", roles),
        ("classifications (true_label)", classes),
    ]


def _tvd_cell(p: Counter, q: Counter) -> str:
    d = tvd(p, q)
    if d != d:
        return "n/a"
    mark = " STRONG" if d > TVD_STRONG else " WARN" if d > TVD_WARN else ""
    return f"{d:.4f}{mark}"


def _drivers(p: Counter, q: Counter, n: int = 5) -> str:
    pt, qt = sum(p.values()), sum(q.values())
    if not pt or not qt:
        return ""
    ranked = sorted(set(p) | set(q), key=lambda k: -abs(p[k] / pt - q[k] / qt))[:n]
    return ", ".join(f"{k} {p[k] / pt:.0%}→{q[k] / qt:.0%}" for k in ranked)


def _split_pairs(splits: Dict) -> List[Tuple[str, str]]:
    pairs = []
    for a, b in (("train", "val"), ("train", "dev"), ("train", "test"),
                 ("val", "test"), ("dev", "test"), ("val", "dev")):
        if a in splits and b in splits:
            pairs.append((a, b))
    return pairs


def _label_table(title: str, counts: Counter, top: int) -> List[str]:
    items = counts.most_common()
    shown = items[:top] if top else items
    total = sum(counts.values())
    lines = [f"#### {title}", ""]
    lines += _md_table(
        ["label", "count", "share"],
        [[name, f"{n:,}", _share(n, total)] for name, n in shown])
    if top and len(items) > top:
        lines.append("")
        lines.append(f"{len(items) - top} further labels omitted (`--top {top}`).")
    return lines


def _coref_section(splits: Dict[str, Dict], order: List[str]) -> List[str]:
    """Mention clusters, trigger pairs, and argument coreference. A zero is a measured zero."""
    lines = ["### Coreference", ""]
    any_clusters = any(m["coref_clusters"] for m in splits.values())
    lines.append("**Mention clusters.** `coref_clusters` on the record: a list of mentions `{text, start, end}`. A pair is two mentions in the same cluster. A pronoun is a mention whose text is in the inference pronoun list.")
    lines.append("")
    if not any_clusters:
        lines.append("No `coref_clusters`.")
        lines.append("")
    else:
        lines += _md_table(
            ["split", "docs", "clusters", "mentions", "multi-mention", "singletons",
             "mention pairs", "pronoun mentions", "pronoun share"],
            [[s, f"{splits[s]['coref_docs']:,}", f"{splits[s]['coref_clusters']:,}",
              f"{splits[s]['coref_mentions']:,}", f"{splits[s]['coref_multi']:,}",
              f"{splits[s]['coref_singletons']:,}", f"{splits[s]['coref_mention_pairs']:,}",
              f"{splits[s]['coref_pronouns']:,}",
              _share(splits[s]["coref_pronouns"], splits[s]["coref_mentions"])]
             for s in order])
        lines.append("")
    lines.append(
        "**Trigger coref.** A positive pair is two triggers listed on the same event. That is what TriggerLink trains as the same event. "
        "A same-type pair drawn from two events is a gold hard negative: different events. "
        "False triggers that are not in the gold are not in this count.")
    lines.append("")
    if not any(m["trig_coref_events"] or m["trig_coref_cross"] for m in splits.values()):
        lines.append("No event lists more than one trigger, and no document has two events of one type, so there is no trigger pair.")
        lines.append("")
    else:
        lines += _md_table(
            ["split", "events with > 1 trigger", "positive pairs", "same-type pairs across events"],
            [[s, f"{splits[s]['trig_coref_events']:,}", f"{splits[s]['trig_coref_pos']:,}",
              f"{splits[s]['trig_coref_cross']:,}"] for s in order])
        lines.append("")
    lines.append(
        "**Argument coref.** The string rule is `expand_argument_coreferents`: an argument surface that sits in exactly one cluster gains that cluster's other mentions, and a surface that sits in two clusters is skipped. "
        "The span rule matches the argument's offsets to one cluster. It still counts a pronoun whose surface is shared by two clusters.")
    lines.append("")
    if not any(m["argument_gold"] or m["arg_coref_added"] or m["arg_span_coref"] for m in splits.values()):
        lines.append("No event arguments.")
        lines.append("")
        return lines
    lines += _md_table(
        ["split", "mentions the string rule adds", "arguments skipped, 2+ clusters",
         "arguments in one multi-mention cluster", "other mentions of that cluster",
         "argument spans in 2+ clusters"],
        [[s, f"{splits[s]['arg_coref_added']:,}", f"{splits[s]['arg_coref_skipped']:,}",
          f"{splits[s]['arg_span_coref']:,}", f"{splits[s]['arg_span_others']:,}",
          f"{splits[s]['arg_span_ambiguous']:,}"] for s in order])
    lines.append("")
    return lines


def _language_section(splits: Dict[str, Dict], order: List[str]) -> List[str]:
    """Every detected language and how many records it covers. A missing counter is not a zero."""
    lines = ["### Language", ""]
    lines.append(
        "One language per record, from `train._detect_lang` (lumi, ISO 639-2), the same code "
        "`eval_by_language` uses. `und` is undetermined. The count is records.")
    lines.append("")
    if not any(m["languages"] for m in splits.values()):
        lines.append("Language was not measured.")
        lines.append("")
        return lines
    totals = Counter()
    for split in order:
        totals.update(splits[split]["languages"])
    grand = sum(m["records"] for m in splits.values())
    rows = []
    for lang, n in totals.most_common():
        rows.append([lang, *[f"{splits[s]['languages'][lang]:,}" for s in order],
                     f"{n:,}", _share(n, grand)])
    lines += _md_table(["language", *order, "records", "share"], rows)
    lines.append("")
    return lines


def _dataset_markdown(dataset: str, splits: Dict[str, Dict], registry: Dict, top: int) -> List[str]:
    lines = [f"## {dataset}", ""]
    order = sorted(splits, key=lambda s: SPLIT_ORDER.get(s, 9))
    rows = []
    for split in order:
        m = splits[split]
        h = head_counts(m)
        w = word_stats(m["words"])
        rows.append([
            split, f"{m['records']:,}", f"{len(m['hashes']):,}",
            w["median"], _percentile(m["words"], 90), w["max"],
            f"{h['entities']:,}", f"{h['relations']:,}", f"{h['classifications']:,}",
            f"{h['structures']:,}", f"{h['event_triggers']:,}", f"{h['event_arguments']:,}",
        ])
    lines.append("### Splits")
    lines.append("")
    lines.append("Lengths are whitespace tokens (`WhitespaceTokenSplitter`), the same count as the text report. They are not encoder subwords.")
    lines.append("")
    lines += _md_table(
        ["split", "records", "unique inputs", "tokens median", "tokens p90", "tokens max",
         "entities", "relations", "classifications", "structures", "event triggers", "event arguments"],
        rows)
    lines += _language_section(splits, order)

    lines.append("### Cross-split uniqueness")
    lines.append("")
    lines.append("The key is the sha1 of the stripped input text. `records` above `unique inputs` means the same text is repeated inside that split.")
    lines.append("")
    uniq = uniqueness({dataset: splits})
    lines += _md_table(["check", "count"], [[k, f"{v:,}"] for k, v in uniq.items()])
    lines.append("")

    lines.append("### Label shift (TVD)")
    lines.append("")
    lines.append(
        "TVD is total variation distance, `0.5 * sum |p - q|` over one category's label marginal. "
        "0 means the two splits match. 1 means they share no label. "
        f"The data-health gates are TVD {TVD_WARN} (WARN) and {TVD_STRONG} (STRONG). "
        "Support is the count each marginal is normalised over. A TVD on a small support is not comparable to one on a large support.")
    lines.append("")
    shift_rows = []
    for name, _ in _marginals(next(iter(splits.values()))):
        for a, b in _split_pairs(splits):
            pa = dict(_marginals(splits[a]))
            pb = dict(_marginals(splits[b]))
            p, q = pa[name], pb[name]
            if not sum(p.values()) and not sum(q.values()):
                continue
            shift_rows.append([
                name, f"{a} / {b}", _tvd_cell(p, q),
                f"{sum(p.values()):,} / {sum(q.values()):,}", _drivers(p, q),
            ])
    if shift_rows:
        lines += _md_table(["category", "splits", "TVD", "support", "largest movers"], shift_rows)
    else:
        lines.append("No label marginal has gold on both sides of a split pair.")
    lines.append("")
    train = splits.get("train")
    if train:
        for other in order:
            if other == "train":
                continue
            for name, counts in _marginals(splits[other]):
                train_counts = dict(_marginals(train))[name]
                unseen = {k: v for k, v in counts.items() if k not in train_counts and v}
                thin = {k: v for k, v in counts.items() if 0 < v < THIN_SUPPORT}
                if unseen:
                    shown = ", ".join(f"{k} ×{v}" for k, v in sorted(unseen.items(), key=lambda kv: -kv[1]))
                    lines.append(f"Unseen in train, present in {other}, {name}: {shown}.")
                if thin:
                    shown = ", ".join(f"{k} ×{v}" for k, v in sorted(thin.items(), key=lambda kv: kv[1]))
                    lines.append(
                        f"Thin support in {other}, {name}: {len(thin)} label(s) under {THIN_SUPPORT} gold. {shown}.")
    lines.append("")

    lines.append("### Events")
    lines.append("")
    if not any(m["ev_instances"] for m in splits.values()):
        lines.append("No event gold.")
        lines.append("")
    else:
        for split in order:
            m = splits[split]
            if not m["ev_instances"]:
                lines.append(f"**{split}.** No event instances.")
                lines.append("")
                continue
            tw = m["trigger_words"]
            single = sum(1 for w in tw if w == 1) / len(tw) if tw else 0.0
            lines.append(
                f"**{split}.** {m['ev_instances']:,} instances in {m['ev_records']:,} records, "
                f"{m['ev_triggerless']:,} triggerless. "
                f"Trigger gold {m['trigger_gold']:,}. Argument gold {m['argument_gold']:,}. "
                f"Arguments per instance, mean {statistics.mean(m['args_per_instance']) if m['args_per_instance'] else 0:.2f}. "
                f"Trigger length in whitespace tokens, mean {statistics.mean(tw) if tw else 0:.2f}, "
                f"single-token share {single:.1%}.")
            if m["argument_gold"] == 0:
                lines.append("Trigger-only: this split cannot teach or score event arguments.")
            lines.append("")
            lines += _label_table(f"{split} event types", m["event_triggers"], top)
            lines.append("")

    lines.append("### Argument inside a trigger")
    lines.append("")
    lines.append(
        "An argument is inside when it and a trigger of the same event both carry character offsets "
        "and the argument span is contained in the trigger span, edges included. An equal span counts as inside. "
        "A partial overlap, and a trigger contained in a longer argument, count as outside. "
        "String gold has no span, so it is unmeasured and stays out of the inside share. "
        "This is a span count, not an F1.")
    lines.append("")
    geo_rows = []
    any_args = False
    for split in order:
        m = splits[split]
        placed = m["arg_place"]["inside"] + m["arg_place"]["outside"]
        if placed or m["arg_place"]["unmeasured"]:
            any_args = True
        geo_rows.append([
            split, f"{m['arg_place']['inside']:,}", f"{placed:,}",
            _share(m["arg_place"]["inside"], placed),
            f"{m['arg_equal']:,}", f"{m['arg_nested']:,}", f"{m['arg_trigger_inside']:,}",
            f"{m['arg_partial']:,}", f"{m['arg_disjoint']:,}", f"{m['arg_place']['unmeasured']:,}",
        ])
    if any_args:
        lines += _md_table(
            ["split", "inside", "placed", "inside share", "equal", "nested inside",
             "trigger inside argument", "partial overlap", "disjoint", "unmeasured"],
            geo_rows)
        lines.append("")
        lines.append("Equal and nested inside sum to inside. Trigger-inside-argument, partial overlap, and disjoint sum to outside.")
        adapted = sum(m["office_adapted"] for m in splits.values())
        if adapted:
            lines.append(f"{adapted:,} office ACE records were adapted before this count, the same way `measure_argument_inside_trigger.py` adapts them.")
        for split in order:
            roles = splits[split]["arg_place_role"]
            placed_roles = [r for r in roles if roles[r]["inside"] + roles[r]["outside"]]
            if not placed_roles:
                continue
            lines.append("")
            lines.append(f"**{split} by role.**")
            lines.append("")
            lines += _md_table(
                ["role", "inside", "placed", "inside share", "unmeasured"],
                [[r or "(no role)", f"{roles[r]['inside']:,}",
                  f"{roles[r]['inside'] + roles[r]['outside']:,}",
                  _share(roles[r]["inside"], roles[r]["inside"] + roles[r]["outside"]),
                  f"{roles[r]['unmeasured']:,}"]
                 for r in sorted(placed_roles, key=lambda r: (-(roles[r]["inside"] + roles[r]["outside"]), r))])
    else:
        lines.append("No event arguments.")
    lines.append("")

    lines.append("### String occurrence")
    lines.append("")
    lines.append(
        "How many places a gold surface lands when placement is a lowercased whitespace-token run, "
        "which is what string gold does (`latent_all`). Mean 1.00 and a `share > 1` of 0% means every placed surface is unambiguous. "
        "Unplaced is the share of surfaces that land nowhere. "
        "Offset gold is placed at its span, so a repeat here is a repeated string, not an extra training target. "
        "Counted over every record.")
    lines.append("")
    occ_rows = []
    any_occ = False
    for split in order:
        m = splits[split]
        surfaces = m["occ_trig_placed"] + m["occ_trig_unplaced"] + m["occ_arg_placed"] + m["occ_arg_unplaced"]
        if surfaces:
            any_occ = True
        trig_mean = m["occ_trig_sum"] / m["occ_trig_placed"] if m["occ_trig_placed"] else 0
        arg_mean = m["occ_arg_sum"] / m["occ_arg_placed"] if m["occ_arg_placed"] else 0
        occ_rows.append([
            split,
            f"{trig_mean:.2f}", _share(m["occ_trig_gt1"], m["occ_trig_placed"]),
            f"{arg_mean:.2f}", _share(m["occ_arg_gt1"], m["occ_arg_placed"]),
            _share(m["occ_trig_unplaced"] + m["occ_arg_unplaced"], surfaces),
            f"{m['occ_arg_offset']:,}", f"{m['occ_arg_string']:,}",
            f"{m['occ_trig_offset']:,}", f"{m['occ_trig_string']:,}",
        ])
    if any_occ:
        lines += _md_table(
            ["split", "trigger mean occ", "trigger share > 1", "argument mean occ", "argument share > 1",
             "unplaced share", "offset arguments", "string arguments", "offset triggers", "string triggers"],
            occ_rows)
    else:
        lines.append("No trigger or argument surfaces.")
    lines.append("")

    lines.append("### Event identity")
    lines.append("")
    lines.append(
        "The multi-trigger share is events whose trigger list has more than one surface. "
        "A same-type pair that shares an argument text is an upper bound on one real event split into two records: "
        "two real events can share a participant. Nested means one argument set contains the other. Identical means the sets are equal.")
    lines.append("")
    if not any(m["id_events"] for m in splits.values()):
        lines.append("No event instances.")
    else:
        id_rows = []
        for split in order:
            m = splits[split]
            id_rows.append([
                split, f"{m['id_events']:,}", _share(m["id_multi"], m["id_events"]),
                f"{m['id_pairs']:,}", _share(m["id_share_text"], m["id_pairs"]),
                _share(m["id_share_role"], m["id_pairs"]),
                _share(m["id_nested"], m["id_pairs"]), _share(m["id_identical"], m["id_pairs"]),
                _share(m["id_split_docs"], m["id_docs"]),
            ])
        lines += _md_table(
            ["split", "events", "share > 1 trigger", "same-type pairs", "share an argument",
             "share role + argument", "nested", "identical", "docs with a shared argument"],
            id_rows)
    lines.append("")

    lines += _coref_section(splits, order)

    for title, key in (("Entities", "entities"), ("Relations", "relations")):
        lines.append(f"### {title}")
        lines.append("")
        any_head = False
        for split in order:
            counts = splits[split][key]
            if not counts:
                continue
            any_head = True
            lines += _label_table(f"{split}", counts, top)
            lines.append("")
        if not any_head:
            lines.append("None.")
            lines.append("")

    lines.append("### Classifications")
    lines.append("")
    if not any(m["classifications"] for m in splits.values()):
        lines.append("None.")
        lines.append("")
    else:
        for split in order:
            tasks = splits[split]["classifications"]
            if not tasks:
                continue
            for task, labels in sorted(tasks.items()):
                total = sum(labels.values())
                major = labels.most_common(1)[0][1] / total if total else 0.0
                lines.append(f"**{split}, {task}.** Majority class {major:.1%} of {total:,} answers. The marginal is `true_label`, not the menu.")
                lines.append("")
                lines += _label_table(f"{split} {task}", labels, top)
                lines.append("")

    lines.append("### Structures")
    lines.append("")
    if not any(m["structure_records"] for m in splits.values()):
        lines.append("None.")
        lines.append("")
    else:
        for split in order:
            m = splits[split]
            if not m["structure_records"]:
                continue
            lines.append(f"**{split}.**")
            lines.append("")
            rows = []
            for name, n in sorted(m["structure_records"].items()):
                fields = m["structures"].get(name, {})
                if not fields:
                    rows.append([name, "(no field values)", f"{n:,}", "0%"])
                for field, c in sorted(fields.items(), key=lambda kv: -kv[1]):
                    rows.append([name, field, f"{c:,}", _share(c, n)])
                anchors = m["structure_anchors"].get(name)
                anchor = ", ".join(f"{k} ×{v:,}" for k, v in anchors.most_common(4)) if anchors else "none"
                rows.append([name, "record_metadata", anchor, ""])
            lines += _md_table(["record type", "field", "count", "coverage of records"], rows)
            lines.append("")

    lines.append("### Provenance")
    lines.append("")
    prov_rows = []
    for head in ("entities", "relations", "events", "classifications", "structures"):
        prov_rows.append([head, provenance(registry, dataset, head)])
    lines += _md_table(["head", "who wrote the gold"], prov_rows)
    lines.append("")
    return lines


def markdown_report(data: Dict[str, Dict[str, Dict]], registry: Dict, top: int,
                    config: Optional[str] = None) -> str:
    """One markdown census. Each dataset is a section. Metric types are named where they are used."""
    lines = ["# Dataset census", ""]
    lines.append("Gold the training and scoring path sees. Label counts are after the label map when one was applied.")
    if config:
        lines.append(f"Config: `{config}`. Splits are resolved the same way `tools/train/train.py` resolves them.")
    lines.append("")
    lines.append(
        "Checkpoint measures stay in their own commands. Surface alignment needs the model tokenizer. "
        "Gradient share, the junction, and familiarity need a checkpoint. They describe a model, not this gold.")
    lines.append("")
    summary = [["dataset", "split", "records", "unique inputs", "entities", "relations",
                "classifications", "structures", "event triggers", "event arguments"]]
    for dataset in sorted(data):
        for split in sorted(data[dataset], key=lambda s: SPLIT_ORDER.get(s, 9)):
            m = data[dataset][split]
            h = head_counts(m)
            summary.append([
                dataset, split, f"{m['records']:,}", f"{len(m['hashes']):,}",
                f"{h['entities']:,}", f"{h['relations']:,}", f"{h['classifications']:,}",
                f"{h['structures']:,}", f"{h['event_triggers']:,}", f"{h['event_arguments']:,}",
            ])
    lines += _md_table(summary[0], summary[1:])
    lines.append("")
    if len(data) > 1:
        lines.append("## Across these datasets")
        lines.append("")
        lines.append("Uniqueness here unions every dataset in this run. An overlap can be two corpora sharing a document, not only a split of one corpus.")
        lines.append("")
        lines += _md_table(["check", "count"], [[k, f"{v:,}"] for k, v in uniqueness(data).items()])
        lines.append("")
    for dataset in sorted(data):
        lines += _dataset_markdown(dataset, data[dataset], registry, top)
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Report label distributions for the data/ corpora.")
    p.add_argument("datasets", nargs="*", help="Limit to these dataset basenames (default: all).")
    p.add_argument("--data-dir", default="data", help="Directory of *.jsonl corpora (default: data).")
    p.add_argument("--config", help="Measure a training config's resolved splits, label map applied.")
    p.add_argument("--lang", action="store_true", help="Detect each record's language (~1 ms/record).")
    p.add_argument("--labels", help="Label map to apply before counting: a unified-map YAML, or "
                                    "'none' for raw labels. Default: raw in glob mode, the "
                                    "config's own map in --config mode. Passthrough corpora stay raw.")
    p.add_argument("--top", type=int, default=0, help="Cap each distribution to its top-N labels (0 = all).")
    p.add_argument("--json", dest="json_out", help="Write metrics as JSON here instead of a text report.")
    p.add_argument("--markdown", dest="markdown_out",
                   help="Write one markdown census. Detects each record's language. Also includes "
                        "splits, TVD, argument-inside-trigger, string occurrence, event identity, "
                        "and trigger and argument coreference.")
    args = p.parse_args(argv)

    # The markdown census always names every language. Text and JSON stay opt-in via --lang.
    lang = T._detect_lang if args.lang or args.markdown_out else None
    if args.config:
        data = collect_config(args.config, lang, args.labels)
    else:
        fns = load_label_fns(args.labels) if args.labels and args.labels != "none" else None
        data = collect(Path(args.data_dir), args.datasets, lang, fns)
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
    if args.markdown_out:
        registry = load_registry().get("datasets") or {}
        Path(args.markdown_out).write_text(
            markdown_report(data, registry, args.top, args.config), encoding="utf-8")
        print(f"[out] wrote {args.markdown_out} ({len(data)} datasets)")
    if args.json_out or args.markdown_out:
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
