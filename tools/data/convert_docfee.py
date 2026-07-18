"""Convert DocFEE (Chen et al., Scientific Data 2025) to GLiNER2 JSONL.

DocFEE is a document-level Chinese financial event-extraction dataset (9 event
types, ~19k announcements) built from EastMoney disclosures. It is
**trigger-free and offset-free**: an event is an event-type plus a table of
``{role -> string value}``, where some values are normalized (e.g. amounts) and
may not appear verbatim in the text. It therefore maps, like DocEE/ChFinAnn, to:

* ``output.entities``  — role-filler values bucketed by their (Chinese) role
  key, **best-effort**: only values that occur verbatim in the document are
  kept (normalized values that don't appear are skipped).
* ``output.classifications``  — one **multi-label** record per document with the
  document's event types as ``true_label`` and the 9-type vocabulary as
  ``labels``.

Classification-only records are kept (unlike ChFinAnn, which requires an entity)
because document event-type detection is DocFEE's primary, reliable signal. No
triggers are emitted; the training config should select on the classification
metric.

Raw record layout (JSONL, one object/line): ``content`` (text with ``<br>``
line separators), ``doc_id``, a top-level ``event_type`` that is an
annotation-BATCH label (ignored — the real type is per-event), and ``events``
(each ``{event_type, event_id, <role>: <value>, ...}``).

Data is CC-BY-4.0 (Figshare DOI 10.6084/m9.figshare.28632464). The converter
downloads the authors' GitHub ``DFREE_dataset.zip`` (train.jsonl + test.jsonl,
no account) -- more reliable than Figshare's S3 stream -- or reads them from a
dir via ``--input``. DocFEE ships no dev split, so val is carved from train with
a seeded shuffle; test.jsonl -> test.

Usage::

    uv run python tools/data/convert_docfee.py --out data/docfee.jsonl
    # or from a dir holding train.jsonl / test.jsonl:
    uv run python tools/data/convert_docfee.py --input /path/to/dir --out data/docfee.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402

# The DocFEE authors commit the data as a single zip (train.jsonl + test.jsonl)
# in their GitHub repo. Prefer it over Figshare -- Figshare's S3 redirect
# streams the 108 MB train file unreliably (it stalls mid-transfer), whereas the
# ~28 MB GitHub zip downloads in one shot with no account.
DOCFEE_ZIP_URL = "https://raw.githubusercontent.com/tongzhou21/DocFEE/main/data/DFREE_dataset.zip"
SPLIT_FILES = {"train": "train.jsonl", "test": "test.jsonl"}
# Keys that are event metadata rather than argument roles.
NON_ROLE_KEYS = frozenset({"event_id", "event_type"})


def _download_and_extract(url: str, dest: Path) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read()
    print(f"  downloaded {len(raw) / 1e6:.1f} MB, extracting...")
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        zf.extractall(dest)
    return dest


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _event_types(rec: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for ev in rec.get("events") or []:
        if isinstance(ev, dict):
            et = ev.get("event_type")
            if isinstance(et, str) and et.strip() and et.strip() not in out:
                out.append(et.strip())
    return out


def convert_record(rec: Dict[str, Any], class_vocab: List[str], sep: str) -> Optional[Dict[str, Any]]:
    content = rec.get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    # Strip HTML noise: <br> line breaks and literal/real non-breaking spaces
    # (the source announcements use &nbsp; as a field separator).
    text = content.replace("<br>", sep).replace("&nbsp;", " ").replace("\xa0", " ")

    entities: Dict[str, List[str]] = {}
    for ev in rec.get("events") or []:
        if not isinstance(ev, dict):
            continue
        for key, val in ev.items():
            if key in NON_ROLE_KEYS or not isinstance(key, str) or not isinstance(val, str):
                continue
            key, val = key.strip(), val.strip()
            if not key or not val or val not in text:
                continue
            bucket = entities.setdefault(key, [])
            if val not in bucket:
                bucket.append(val)

    types_present = _event_types(rec)
    if not types_present:
        return None

    output: Dict[str, Any] = {}
    if entities:
        output["entities"] = entities
    output["classifications"] = [{
        "task": "docfee_event",
        "labels": list(class_vocab),
        "true_label": types_present,
    }]
    return {"input": text, "output": output}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output base path; writes <base>.{train,val,test}.jsonl.")
    parser.add_argument("--input", type=Path, default=None,
                        help="Dir holding train.jsonl / test.jsonl. If omitted, "
                             "the GitHub DFREE_dataset.zip is downloaded + extracted.")
    parser.add_argument("--sep", default="",
                        help="Replacement for the <br> line separator in content "
                             "(default: '' = continuous text).")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of train.jsonl held out as val (default 0.1).")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Max records per source file (-1 = all).")
    args = parser.parse_args()

    if args.input is not None:
        data_dir = args.input
    else:
        import tempfile
        data_dir = _download_and_extract(DOCFEE_ZIP_URL, Path(tempfile.mkdtemp(prefix="docfee_")))
    src = {name: data_dir / fn for name, fn in SPLIT_FILES.items()}
    for p in src.values():
        if not p.is_file():
            raise SystemExit(f"missing {p}")

    raw = {name: _load_jsonl(p) for name, p in src.items()}
    print(f"Loaded train={len(raw['train'])} test={len(raw['test'])} docs")

    vocab_counts: Counter = Counter()
    for items in raw.values():
        for rec in items:
            for et in _event_types(rec):
                vocab_counts[et] += 1
    class_vocab = sorted(vocab_counts)
    print(f"  event-type vocab ({len(class_vocab)}): {class_vocab}")

    def _convert_all(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for rec in items:
            if 0 <= args.max_records <= len(out):
                break
            r = convert_record(rec, class_vocab, args.sep)
            if r is not None:
                out.append(r)
        return out

    train_all = _convert_all(raw["train"])
    test_records = _convert_all(raw["test"])

    rng = random.Random(args.split_seed)
    rng.shuffle(train_all)
    n_val = int(len(train_all) * args.val_ratio)
    val_records, train_records = train_all[:n_val], train_all[n_val:]

    stem = args.out.with_suffix("") if args.out.suffix == ".jsonl" else args.out
    out_paths = {s: Path(f"{stem}.{s}.jsonl") for s in ("train", "val", "test")}
    out_paths["train"].parent.mkdir(parents=True, exist_ok=True)
    for split, recs in (("train", train_records), ("val", val_records), ("test", test_records)):
        with out_paths[split].open("w", encoding="utf-8") as f:
            for rec in recs:
                f.write(dumps_record(rec) + "\n")
        print(f"  wrote {split}: {len(recs)} -> {out_paths[split]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
