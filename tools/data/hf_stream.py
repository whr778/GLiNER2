"""Stream HuggingFace token-NER datasets into GLiNER2 training WITHOUT disk.

Unlike the tools/data/convert_*.py scripts (which materialize data/*.jsonl),
these functions yield GLiNER2 records lazily so the trainer can stream them
epoch-by-epoch (see gliner2.training.trainer.StreamingExtractorDataset and the
``data.hf_streaming`` config block wired in tools/train/train.py). Nothing is
written to disk; with ``streaming=True`` the parquet is not cached either.

Currently one source: WikiANN / PAN-X (``unimelb-nlp/wikiann``), token-BIO NER
over 176 languages, three entity types (PER/ORG/LOC -> person/organization/
location). Add more by registering a StreamSource in SOURCES.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_hf_token_ner import bio_to_entities  # noqa: E402

WIKIANN_ID = "unimelb-nlp/wikiann"
# Upstream is a (no-longer-supported) dataset script; load the auto-converted
# parquet export, which stores files as <lang>/<split>/*.parquet.
REVISION = "refs/convert/parquet"
# HF split name -> our canonical split name.
SPLIT_MAP = {"train": "train", "validation": "validation", "test": "test"}
# The fixed ner_tags ClassLabel ordering (same for every language).
WIKIANN_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
WIKIANN_LABEL_MAP = {"PER": "person", "ORG": "organization", "LOC": "location"}
WIKIANN_CLASSES = ["person", "organization", "location"]


# ----- repo file listing (cached) -----

_FILE_CACHE: Dict[tuple, List[str]] = {}


def _repo_files(hf_id: str, revision: str) -> List[str]:
    key = (hf_id, revision)
    if key not in _FILE_CACHE:
        from huggingface_hub import HfApi
        _FILE_CACHE[key] = HfApi().list_repo_files(
            hf_id, repo_type="dataset", revision=revision)
    return _FILE_CACHE[key]


def list_wikiann_languages(hf_id: str = WIKIANN_ID, revision: str = REVISION) -> List[str]:
    """All language codes (top-level parquet dirs) in the repo."""
    files = _repo_files(hf_id, revision)
    return sorted({f.split("/")[0] for f in files if f.endswith(".parquet") and "/" in f})


def resolve_langs(langs, hf_id: str = WIKIANN_ID, revision: str = REVISION) -> List[str]:
    """Resolve 'all' (or a list/CSV of codes) against the available languages."""
    if isinstance(langs, str):
        langs = langs.split(",") if langs.strip().lower() != "all" else "all"
    if langs == "all":
        return list_wikiann_languages(hf_id, revision)
    picked = [str(c).strip() for c in langs if str(c).strip()]
    available = set(list_wikiann_languages(hf_id, revision))
    unknown = [c for c in picked if c not in available]
    if unknown:
        raise ValueError(
            f"unknown wikiann language(s) {unknown}; {len(available)} available, e.g. "
            f"{sorted(available)[:10]} ... or 'all'")
    return picked


# ----- streaming records -----

def _shard_urls(hf_id: str, lang: str, hf_split: str, revision: str) -> List[str]:
    from huggingface_hub import hf_hub_url
    prefix = f"{lang}/{hf_split}/"
    shards = sorted(f for f in _repo_files(hf_id, revision)
                    if f.startswith(prefix) and f.endswith(".parquet"))
    return [hf_hub_url(repo_id=hf_id, filename=f, repo_type="dataset", revision=revision)
            for f in shards]


def _stream_split(hf_id: str, lang: str, hf_split: str, revision: str) -> Iterator[dict]:
    """Stream one language's split rows over HTTP (no local cache)."""
    from datasets import load_dataset
    urls = _shard_urls(hf_id, lang, hf_split, revision)
    ds = load_dataset("parquet", data_files={hf_split: urls}, split=hf_split, streaming=True)
    return iter(ds)


def _row_to_record(row: dict) -> Optional[dict]:
    """Fold a tokens+ner_tags row into a GLiNER2 entity record, or None if empty."""
    tokens = [str(t) for t in (row.get("tokens") or [])]
    tag_ids = row.get("ner_tags") or []
    tags = [WIKIANN_TAGS[t] if 0 <= t < len(WIKIANN_TAGS) else "O" for t in tag_ids]
    text, entities = bio_to_entities(tokens, tags)
    if not entities:
        return None
    entities = {WIKIANN_LABEL_MAP.get(k, k.lower()): v for k, v in entities.items()}
    return {"input": text, "output": {"entities": entities}}


def wikiann_records(langs, split: str, hf_id: str = WIKIANN_ID,
                    revision: str = REVISION) -> Iterator[dict]:
    """Yield GLiNER2 entity records for ``split``, round-robin across languages.

    Round-robin (one record per language per cycle) keeps a bounded eval sample
    language-diverse instead of dominated by alphabetically-first languages, and
    interleaves languages within each training epoch. NB: 'all' opens one stream
    per language; prefer a listed subset for the 176-language extreme.
    """
    langs = resolve_langs(langs, hf_id, revision)
    hf_split = SPLIT_MAP[split]
    iters = [_stream_split(hf_id, lang, hf_split, revision) for lang in langs]
    active = list(range(len(iters)))
    while active:
        still = []
        for i in active:
            try:
                row = next(iters[i])
            except StopIteration:
                continue
            still.append(i)
            rec = _row_to_record(row)
            if rec is not None:
                yield rec
        active = still


# ----- bounded eval sampling (cap by label class) -----

def entity_classes(record: dict) -> Iterable[str]:
    """The entity-type label classes present in a record."""
    return record.get("output", {}).get("entities", {}).keys()


def classification_classes(record: dict) -> Iterable[str]:
    """The true classification labels present in a record."""
    out = []
    for c in record.get("output", {}).get("classifications", []):
        out.extend(c.get("true_label") or [])
    return out


def cap_by_class(records: Iterable[dict], min_per_class: int,
                 classes_of: Callable[[dict], Iterable[str]] = entity_classes,
                 expected_classes: Optional[Iterable[str]] = None) -> Iterator[dict]:
    """Yield records until every class has >= ``min_per_class``, then stop.

    A record counts toward every class it contains, so frequent classes overshoot
    (e.g. min=3000 => each class has at least 3000, some more). ``expected_classes``
    is the class universe to wait for (so a rarer class discovered late is not
    skipped); if omitted, only classes seen so far gate the stop. min<=0 disables
    the cap (yield everything).
    """
    if not min_per_class or min_per_class <= 0:
        yield from records
        return
    counts: Dict[str, int] = {c: 0 for c in (expected_classes or ())}
    target = set(expected_classes) if expected_classes else None
    for rec in records:
        yield rec
        for c in classes_of(rec):
            counts[c] = counts.get(c, 0) + 1
        gate = target or set(counts)
        if gate and all(counts.get(c, 0) >= min_per_class for c in gate):
            return
    short = {c: n for c, n in counts.items() if n < min_per_class}
    if short:
        print(f"[hf_stream] cap_by_class: stream ended before reaching "
              f"min_per_class={min_per_class} for {short}", file=sys.stderr)


# ----- source registry -----

@dataclass
class StreamSource:
    """A streamable dataset: how to enumerate records and classes for the cap."""
    records: Callable[[object, str], Iterator[dict]]     # (langs, split) -> records
    classes: List[str]                                    # class universe for the eval cap
    classes_of: Callable[[dict], Iterable[str]]           # record -> its classes
    list_languages: Callable[[], List[str]]


SOURCES: Dict[str, StreamSource] = {
    "wikiann": StreamSource(
        records=wikiann_records,
        classes=WIKIANN_CLASSES,
        classes_of=entity_classes,
        list_languages=list_wikiann_languages,
    ),
}
