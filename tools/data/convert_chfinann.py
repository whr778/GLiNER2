"""Convert ChFinAnn (Zheng et al., EMNLP 2019 / Doc2EDAG) to GLiNER2 JSONL.

ChFinAnn is a large document-level Chinese financial event-extraction corpus:
32,040 announcements, 5 event types (EquityFreeze, EquityRepurchase,
EquityUnderweight, EquityOverweight, EquityPledge), and 24 role/field types.
It follows a **trigger-free** design — an event is an event-type label plus a
table of ``{role -> filler-mention}`` — so, like DocEE, it maps most faithfully
to entities + document classification rather than trigger-anchored events:

* ``output.entities``  — role-filler mentions bucketed by their field type
  (``EquityHolder``, ``Pledgee``, ``StockCode``, …); the catch-all
  ``OtherType`` bucket is dropped.
* ``output.classifications``  — one **multi-label** record per document with
  the document's event types as ``true_label`` (a doc may hold several) and the
  5-type vocabulary as ``labels``.

No triggers are emitted (there are none). Because this shape has no event
block, the matching training config should select checkpoints on the
classification metric (``eval_classification_strict_micro_f1``), not the event
metric.

Raw record layout (one big JSON array per split file, element = ``[recguid,
doc]``): ``doc.sentences`` (list[str]), ``doc.ann_mspan2guess_field``
(mention -> field type), ``doc.recguid_eventname_eventdict_list`` (events, each
``[idx, event_type, {role: filler_or_null}]``). Mentions occur verbatim inside
the sentences, so surface-string matching recovers them without offsets.

The data is one account-free zip on GitHub
(https://github.com/dolphin-zs/Doc2EDAG). The converter downloads and extracts
it in-process (no manual step), or reads a pre-extracted ``Data/`` directory
via ``--input``. Canonical train/dev/test splits are preserved (dev -> val).

Usage::

    uv run python tools/data/convert_chfinann.py --out data/chfinann.jsonl
    # or from a pre-extracted dir holding train.json/dev.json/test.json:
    uv run python tools/data/convert_chfinann.py \\
        --input /path/to/Data --out data/chfinann.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402

CHFINANN_ZIP_URL = "https://github.com/dolphin-zs/Doc2EDAG/raw/master/Data.zip"

# dev.json is the validation split; keep the canonical mapping explicit.
SPLIT_FILES = {"train": "train.json", "val": "dev.json", "test": "test.json"}


def _download_and_extract(url: str, dest: Path) -> Path:
    """Fetch Data.zip and extract it into ``dest``; return the dir with the splits."""
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(req, timeout=180) as r:
        raw = r.read()
    print(f"  downloaded {len(raw) / 1e6:.1f} MB, extracting...")
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        zf.extractall(dest)
    data_dir = dest / "Data"
    return data_dir if data_dir.is_dir() else dest


def _load_split(data_dir: Path, filename: str) -> List[Any]:
    path = data_dir / filename
    if not path.is_file():
        raise SystemExit(f"missing split file: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _event_types(doc: Dict[str, Any]) -> List[str]:
    """Distinct event types annotated on a document (order-preserving)."""
    out: List[str] = []
    for ev in doc.get("recguid_eventname_eventdict_list") or []:
        if isinstance(ev, list) and len(ev) >= 2 and isinstance(ev[1], str):
            et = ev[1].strip()
            if et and et not in out:
                out.append(et)
    return out


def convert_row(
    recguid: Any,
    doc: Dict[str, Any],
    class_vocab: List[str],
    sep: str,
) -> Optional[Dict[str, Any]]:
    sentences = [s for s in (doc.get("sentences") or []) if isinstance(s, str)]
    text = sep.join(sentences)
    if not text.strip():
        return None

    entities: Dict[str, List[str]] = {}
    for mention, field in (doc.get("ann_mspan2guess_field") or {}).items():
        if not isinstance(mention, str) or not isinstance(field, str):
            continue
        mention, field = mention.strip(), field.strip()
        if not mention or not field or field == "OtherType":
            continue
        if mention not in text:
            continue
        bucket = entities.setdefault(field, [])
        if mention not in bucket:
            bucket.append(mention)

    types_present = _event_types(doc)

    output: Dict[str, Any] = {}
    if entities:
        output["entities"] = entities
    if types_present:
        output["classifications"] = [{
            "task": "chfinann_event",
            "labels": list(class_vocab),
            "true_label": types_present,
        }]
    if not output.get("entities"):
        # Need at least one extraction signal beyond the doc label.
        return None
    return {"input": text, "output": output}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output base path; writes <base>.{train,val,test}.jsonl.")
    parser.add_argument("--input", type=Path, default=None,
                        help="Pre-extracted dir holding train.json/dev.json/test.json. "
                             "If omitted, Data.zip is downloaded from GitHub.")
    parser.add_argument("--url", default=CHFINANN_ZIP_URL,
                        help="Data.zip URL when --input is not provided.")
    parser.add_argument("--sep", default="",
                        help="Separator used to join sentences into the document "
                             "text (default: '' = continuous Chinese text).")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Max documents per split to emit (-1 = all).")
    args = parser.parse_args()

    if args.input is not None:
        data_dir = args.input / "Data" if (args.input / "Data").is_dir() else args.input
    else:
        import tempfile
        data_dir = _download_and_extract(args.url, Path(tempfile.mkdtemp(prefix="chfinann_")))

    raw = {name: _load_split(data_dir, fn) for name, fn in SPLIT_FILES.items()}
    print(f"Loaded train={len(raw['train'])} val={len(raw['val'])} test={len(raw['test'])} docs")

    # Event-type classification vocab derived from all splits (kept consistent).
    vocab_counts: Counter = Counter()
    for items in raw.values():
        for elem in items:
            if isinstance(elem, list) and len(elem) == 2:
                for et in _event_types(elem[1]):
                    vocab_counts[et] += 1
    class_vocab = sorted(vocab_counts)
    print(f"  event-type vocab ({len(class_vocab)}): {class_vocab}")

    out_paths = {s: Path(f"{args.out.with_suffix('') if args.out.suffix == '.jsonl' else args.out}.{s}.jsonl")
                 for s in SPLIT_FILES}
    out_paths["train"].parent.mkdir(parents=True, exist_ok=True)

    for split, items in raw.items():
        n = 0
        skipped = 0
        with out_paths[split].open("w", encoding="utf-8") as f:
            for elem in items:
                if 0 <= args.max_records <= n:
                    break
                if not (isinstance(elem, list) and len(elem) == 2):
                    skipped += 1
                    continue
                rec = convert_row(elem[0], elem[1], class_vocab, args.sep)
                if rec is None:
                    skipped += 1
                    continue
                f.write(dumps_record(rec) + "\n")
                n += 1
        print(f"  {split}: wrote {n} (skipped {skipped}) -> {out_paths[split]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
