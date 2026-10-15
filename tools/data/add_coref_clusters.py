"""Add gold coreference clusters to corpus files IN PLACE: every other byte of every record unchanged.

    uv run python tools/data/add_coref_clusters.py --corpus cmnee
    uv run python tools/data/add_coref_clusters.py --corpus wikievents --raw /path/to/wikievents_raw

Each record gains a top-level ``coref_clusters``: a list of clusters, each a list of mentions
``{"text", "start", "end"}`` with code-point offsets into the record's ``input``. CMNEE ships
ARGUMENT clusters (``coref_arguments``), WikiEvents ENTITY clusters (``coref/*.jsonlines``).
Nothing in training reads the field yet.

WHY OFFSETS, NOT SURFACES: 11% of WikiEvents surfaces sit in two clusters of one document
(``he``, ``they``, ``his``), so a surface cannot say which cluster it belongs to.

WHY NOT A CONVERTER RE-RUN: the files on disk are NOT what the converters emit today. They
predate the invisible-character strip in ``_split.clean_text`` (16 CMNEE inputs keep a U+200B),
and they are the DEDUPED splits -- a fresh wikievents train carries 2 dev/test documents. So
this reads each disk line, finds its raw document by normalized text, maps the raw offsets
through that same normalization, and appends the key. It does not write through
``dumps_record``: that would strip the U+200B the disk inputs still carry.

A mention is kept only when ``input[start:end]`` equals its normalized surface; the rest are
counted and dropped. A ``.bak`` of each file is kept.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import _LINE_SEPARATORS, clean_text  # noqa: E402

KEY = "coref_clusters"
SPLITS = {"cmnee": {"train": "train", "val": "valid", "test": "test"},
          "wikievents": {"train": "train", "dev": "dev", "test": "test"}}


def _old_normalize(s: str) -> str:
    """The normalization the CMNEE files were written under: NFKC + line separators, no invisible strip."""
    return unicodedata.normalize("NFKC", s).translate(_LINE_SEPARATORS)


NORMALIZERS: Tuple[Callable[[str], str], ...] = (clean_text, _old_normalize)

Doc = Tuple[str, List[List[Tuple[str, int, int]]]]  # raw text, clusters of (surface, raw start, raw end)


def cmnee_docs(raw: Path, split: str) -> List[Doc]:
    rows = json.load(open(raw / f"{split}.json", encoding="utf-8"))
    return [(r["text"], [[(m["text"], m["offset"][0], m["offset"][1]) for m in c]
                         for c in r.get("coref_arguments") or []]) for r in rows]


def _char_offsets(text: str, tokens: List[str]) -> List[Tuple[int, int]]:
    out, pos = [], 0
    for t in tokens:
        i = text.index(t, pos)
        out.append((i, i + len(t)))
        pos = i + len(t)
    return out


def wikievents_docs(raw: Path, split: str) -> List[Doc]:
    clusters = {c["doc_key"]: c["clusters"]
                for c in map(json.loads, open(raw / f"coref_{split}.jsonlines", encoding="utf-8"))}
    docs = []
    for r in map(json.loads, open(raw / f"{split}.jsonl", encoding="utf-8")):
        offs = _char_offsets(r["text"], r["tokens"])
        ments = {m["id"]: (m["text"], offs[m["start"]][0], offs[m["end"] - 1][1]) for m in r["entity_mentions"]}
        docs.append((r["text"], [[ments[m] for m in c if m in ments] for c in clusters.get(r["doc_id"], [])]))
    return docs


def place(text: str, clusters: List[List[Tuple[str, int, int]]], norm: Callable[[str], str],
          raw: str, counts: Dict[str, int]) -> List[List[Dict]]:
    """Map raw offsets into the normalized ``text``; keep a mention only where its surface is."""
    out = []
    for c in clusters:
        kept = []
        for surface, s, e in c:
            ns, ne = len(norm(raw[:s])), len(norm(raw[:e]))
            want = norm(surface)
            ns, ne = ns + len(want) - len(want.lstrip()), ne - (len(want) - len(want.rstrip()))
            if text[ns:ne] == want.strip():
                kept.append({"text": text[ns:ne], "start": ns, "end": ne})
                counts["kept"] += 1
            else:
                counts["dropped"] += 1
        if kept:
            out.append(sorted(kept, key=lambda m: (m["start"], m["end"])))
    return sorted(out, key=lambda c: (c[0]["start"], c[0]["end"]))


def _match(text: str, docs: Dict[str, Doc]) -> Optional[Tuple[Doc, Callable[[str], str]]]:
    for norm in NORMALIZERS:
        doc = docs.get((norm.__name__, text))
        if doc is not None:
            return doc, norm
    return None


def augment(path: Path, docs: List[Doc]) -> Dict[str, int]:
    """Rewrite ``path`` with ``coref_clusters`` on every matched record; verify the rest unchanged."""
    index = {}
    for doc in docs:
        for norm in NORMALIZERS:
            index.setdefault((norm.__name__, norm(doc[0])), doc)
    counts = {"records": 0, "matched": 0, "with_clusters": 0, "clusters": 0, "kept": 0, "dropped": 0}
    old_lines = path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    for line in old_lines:
        rec = json.loads(line)
        if KEY in rec:
            raise SystemExit(f"{path} already carries {KEY}; restore the .bak to re-run")
        counts["records"] += 1
        hit = _match(rec["input"], index)
        if hit is not None:
            (raw, clusters), norm = hit
            counts["matched"] += 1
            rec[KEY] = place(rec["input"], clusters, norm, raw, counts)
            counts["with_clusters"] += bool(rec[KEY])
            counts["clusters"] += len(rec[KEY])
        new_lines.append(json.dumps(rec, ensure_ascii=False))
    for old, new in zip(old_lines, new_lines, strict=True):
        rec = json.loads(new)
        rec.pop(KEY, None)
        assert rec == json.loads(old), "a record changed beyond the added key"
    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", required=True, choices=sorted(SPLITS))
    p.add_argument("--raw", type=Path, help="Raw source dir (default data/cmnee/CMNEE for cmnee).")
    p.add_argument("--data", type=Path, default=Path("data"))
    args = p.parse_args()
    raw = args.raw or Path("data/cmnee/CMNEE")
    load = cmnee_docs if args.corpus == "cmnee" else wikievents_docs
    for split, raw_split in SPLITS[args.corpus].items():
        counts = augment(args.data / f"{args.corpus}.{split}.jsonl", load(raw, raw_split))
        print(f"{args.corpus}.{split}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
