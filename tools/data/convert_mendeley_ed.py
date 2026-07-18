"""Convert the Mendeley "Event Detection Dataset" (Maisonnave et al., 2020) to
GLiNER2 JSONL.

This is an English *ongoing-event detection* corpus over New York Times
economic/financial-crisis news: word-level **trigger identification only** —
no argument roles, no event-type ontology. A word is tagged ``<event>`` when it
refers to a fresh/ongoing real-world event at publication time. 2,000 training +
200 testing sentences (one sentence per XML file).

Because there are no types or arguments, each sentence maps to a single
generic event carrying its trigger words::

    output.events = [{"event_type": "Event", "triggers": [...words...],
                      "arguments": []}]

so GLiNER2's trigger head learns event-trigger detection. Sentences with no
trigger are dropped by default (pass ``--keep-negatives`` to emit them as
empty-event records). The trigger words are recovered verbatim from the clean
(tag-stripped) sentence text, so no character offsets are needed.

Source: Mendeley Data ``7d54rvzxkr`` (DOI 10.17632/7d54rvzxkr.1), CC-BY-4.0,
one account-free tar.gz. The converter downloads and extracts it in-process, or
reads a pre-extracted dir (holding ``training/`` and ``testing/``) via
``--input``. The training folder is split into train/val; testing -> test.

Usage::

    uv run python tools/data/convert_mendeley_ed.py --out data/mendeley_ed.jsonl
    # or from a pre-extracted dir:
    uv run python tools/data/convert_mendeley_ed.py \\
        --input /path/to/extracted --out data/mendeley_ed.jsonl
"""

from __future__ import annotations

import argparse
import io
import random
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402

MENDELEY_TARBALL_URL = (
    "https://data.mendeley.com/public-files/datasets/7d54rvzxkr/files/"
    "db5a05a7-b867-482d-b1b4-65b11dddbd21/file_downloaded"
)
EVENT_TYPE = "Event"


def _download_and_extract(url: str, dest: Path) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(req, timeout=180) as r:
        raw = r.read()
    print(f"  downloaded {len(raw) / 1e6:.1f} MB, extracting...")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        tf.extractall(dest)
    return dest


def _find_dir(root: Path, name: str) -> Optional[Path]:
    """Locate the training/ or testing/ folder anywhere under ``root``."""
    if (root / name).is_dir():
        return root / name
    for cand in root.rglob(name):
        if cand.is_dir():
            return cand
    return None


def parse_xml(path: Path) -> Optional[Tuple[str, List[str]]]:
    """Return ``(clean_sentence_text, trigger_words)`` for one XML file."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    sent = root.find("sentence")
    if sent is None:
        return None
    # Mixed content: sent.text, then each <event> child's .text (the trigger)
    # and .tail (following text). Concatenating reconstructs the tag-free text.
    parts: List[str] = [sent.text or ""]
    triggers: List[str] = []
    for child in sent:
        word = child.text or ""
        if child.tag == "event":
            w = word.strip()
            if w:
                triggers.append(w)
        parts.append(word)
        parts.append(child.tail or "")
    text = "".join(parts).strip()
    if not text:
        return None
    return text, triggers


def to_record(text: str, triggers: List[str], keep_negatives: bool) -> Optional[Dict[str, Any]]:
    if triggers:
        events = [{"event_type": EVENT_TYPE, "triggers": triggers, "arguments": []}]
        return {"input": text, "output": {"events": events}}
    if keep_negatives:
        return {"input": text, "output": {"events": []}}
    return None


def _convert_dir(folder: Path, keep_negatives: bool) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    empty = 0
    for xml_path in sorted(folder.glob("*.xml")):
        parsed = parse_xml(xml_path)
        if parsed is None:
            continue
        text, triggers = parsed
        if not triggers:
            empty += 1
        rec = to_record(text, triggers, keep_negatives)
        if rec is not None:
            records.append(rec)
    print(f"  {folder.name}: {len(records)} records (no-trigger sentences: {empty})")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output base path; writes <base>.{train,val,test}.jsonl.")
    parser.add_argument("--input", type=Path, default=None,
                        help="Pre-extracted dir holding training/ and testing/. "
                             "If omitted, the tar.gz is downloaded from Mendeley.")
    parser.add_argument("--url", default=MENDELEY_TARBALL_URL,
                        help="Tarball URL when --input is not provided.")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of the training folder held out as val "
                             "(default: 0.1). Testing folder is always the test split.")
    parser.add_argument("--keep-negatives", action="store_true",
                        help="Emit sentences with no trigger as empty-event records "
                             "(default: drop them).")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    if args.input is not None:
        root = args.input
    else:
        import tempfile
        root = _download_and_extract(args.url, Path(tempfile.mkdtemp(prefix="mendeley_ed_")))

    train_dir = _find_dir(root, "training")
    test_dir = _find_dir(root, "testing")
    if train_dir is None or test_dir is None:
        raise SystemExit(f"could not find training/ and testing/ under {root}")

    train_all = _convert_dir(train_dir, args.keep_negatives)
    test_records = _convert_dir(test_dir, args.keep_negatives)

    # Split the training folder into train/val (single event type, so a plain
    # seeded shuffle suffices; testing folder is the held-out test split).
    rng = random.Random(args.split_seed)
    rng.shuffle(train_all)
    n_val = int(len(train_all) * args.val_ratio)
    val_records = train_all[:n_val]
    train_records = train_all[n_val:]

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
