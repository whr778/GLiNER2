"""Convert knowledgator/text2json-training-data to GLiNER2 JSONL.

The source repo holds many JSONL files with *inconsistent schemas* across
shards (some carry an ``_augmented`` column, others don't). ``datasets.load_dataset``
fails to merge them, so this converter downloads a single named file via
``huggingface_hub`` and iterates JSONL directly. Default file is
``augmented_train.jsonl`` (12.8k rows, clean ``{text, extracted}`` schema).

The ``extracted`` payload comes in two useful shapes (plus a long tail of
deeply nested objects we skip):

1. **Entity-list** — for entity extraction tasks::

       {"entities": [
           {"entity": "Sarah Cooley", "type": "Person",
            "description": "marine chemist ..."},
           ...
       ]}

   Converted to ``{type: [entity]}`` with descriptions kept in
   ``entity_descriptions``.

2. **Flat key->value** — for text2json proper::

       {"tournament_code": "ROL-2024", "winner": "Sofia Petrova",
        "aces": "15", "attendance": "23400"}

   Converted to ``{key: [str(value)]}``. List-of-strings values become a
   single bucket. Nested dicts and list-of-dicts values are skipped — they
   rarely round-trip verbatim into the source text.

A row is dropped entirely if no extracted surface appears verbatim in the
text (typical for synthetic / paraphrased extractions).

Usage::

    uv run python tools/data/convert_text2json.py \\
        --out data/text2json.jsonl

    # Use a different file from the same repo:
    uv run python tools/data/convert_text2json.py \\
        --file mixed_train.jsonl --out data/text2json_mixed.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, add_split_args

# Drop entity surfaces longer than this many whitespace tokens. text2json's
# flat key->value shape happily promotes things like plot_summary, abstract,
# or long-form description fields into "entities", which then balloon past
# 100 word-tokens and pollute the span head's max_width budget. The span
# head can only predict surfaces up to its configured max_width anyway, so
# anything beyond that is supervision the model can never match.
MAX_SURFACE_WORDS = 50


def _coerce_surface(value: Any) -> str | None:
    """Return a non-empty string surface for primitive scalars, else None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        v = value.strip()
        return v or None
    return None


def _add(entities: dict[str, list[str]], label: str, surface: str, text: str) -> None:
    """Append surface under label if it appears verbatim in text and isn't a dupe."""
    if surface not in text:
        return
    if len(surface.split()) > MAX_SURFACE_WORDS:
        return
    bucket = entities.setdefault(label, [])
    if surface not in bucket:
        bucket.append(surface)


def _ingest_entity_list(items: list, text: str,
                       entities: dict[str, list[str]],
                       descriptions: dict[str, str]) -> None:
    """Process a list of {entity, type, description} dicts."""
    for item in items:
        if not isinstance(item, dict):
            continue
        etype = item.get("type")
        surface = _coerce_surface(item.get("entity"))
        if not isinstance(etype, str) or not etype.strip() or not surface:
            continue
        etype = etype.strip()
        _add(entities, etype, surface, text)
        desc = item.get("description")
        if isinstance(desc, str) and desc.strip() and etype not in descriptions:
            descriptions[etype] = desc.strip()


def convert_row(row: dict) -> dict | None:
    """Convert one text2json row to a GLiNER2 record; None if no usable spans."""
    text = row.get("text")
    raw = row.get("extracted")
    if not isinstance(text, str) or not text.strip() or not raw:
        return None

    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    entities: dict[str, list[str]] = {}
    descriptions: dict[str, str] = {}

    for key, value in data.items():
        if key == "entities" and isinstance(value, list):
            _ingest_entity_list(value, text, entities, descriptions)
            continue
        if not isinstance(key, str) or not key.strip():
            continue
        label = key.strip()

        if isinstance(value, list):
            for item in value:
                surface = _coerce_surface(item)
                if surface is not None:
                    _add(entities, label, surface, text)
            continue

        surface = _coerce_surface(value)
        if surface is not None:
            _add(entities, label, surface, text)

    if not entities:
        return None
    output: dict[str, Any] = {"entities": entities}
    if descriptions:
        output["entity_descriptions"] = descriptions
    return {"input": text, "output": output}


def _iter_jsonl(path: Path):
    """Yield decoded JSON objects from a JSONL file; skip malformed lines."""
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path (writes <base>.train.jsonl, "
                             ".val.jsonl, .test.jsonl).")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum input records to read (-1 = all).")
    parser.add_argument("--repo", default="knowledgator/text2json-training-data",
                        help="HuggingFace dataset repo.")
    parser.add_argument("--file", default="augmented_train.jsonl",
                        help="JSONL file inside the repo to convert "
                             "(default: augmented_train.jsonl).")
    add_split_args(parser)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    print(f"Downloading {args.repo}/{args.file}...")
    src_path = Path(hf_hub_download(args.repo, args.file, repo_type="dataset"))

    emitted = 0
    skipped_empty = 0
    total_entities = 0
    all_types: set[str] = set()

    with SplitWriter(args.out, ratios=args.split_ratios, seed=args.split_seed) as writer:
        for idx, row in enumerate(_iter_jsonl(src_path)):
            if 0 <= args.max_records <= idx:
                break
            record = convert_row(row)
            if record is None:
                skipped_empty += 1
                continue
            writer.write(record)
            emitted += 1
            total_entities += sum(len(v) for v in record["output"]["entities"].values())
            all_types.update(record["output"]["entities"].keys())

            if emitted % 2000 == 0:
                print(f"  emitted={emitted}  skipped_empty={skipped_empty}  "
                      f"types={len(all_types)}")

    print(f"Done. emitted={emitted} skipped_empty={skipped_empty} "
          f"total_entities={total_entities} distinct_types={len(all_types)} "
          f"{writer.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
