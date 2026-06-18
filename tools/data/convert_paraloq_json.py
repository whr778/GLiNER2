"""Convert paraloq/json_data_extraction to GLiNER2 JSONL (schema-driven extraction).

Each source row is a ``(text, schema, item)`` triple: a natural-language
document, a JSON Schema, and the extracted structured object that conforms to
it. This converter walks the extracted ``item`` recursively and maps every leaf
scalar (and list-of-scalars) to GLiNER2's flat field->value entity shape --
``{field_name: [value, ...]}`` under ``output.entities`` -- keeping only values
that appear verbatim in the text (so the span head can match them). The leaf
field name is used as the label.

This mirrors ``convert_text2json.py``'s representation of schema-driven
extraction, generalized to the deeply nested schemas paraloq uses.

License: the source is Apache-2.0.

Usage::

    uv run python tools/data/convert_paraloq_json.py --out data/paraloq_json.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, add_split_args

# Drop surfaces longer than this many whitespace tokens (long free-text fields
# blow past the span head's max_width and can never be matched).
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
    if not surface or surface not in text:
        return
    if len(surface.split()) > MAX_SURFACE_WORDS:
        return
    bucket = entities.setdefault(label, [])
    if surface not in bucket:
        bucket.append(surface)


def _walk(value: Any, label: str | None, text: str, entities: dict[str, list[str]]) -> None:
    """Recurse into the extracted object; map leaf scalars to {label: [value]}.

    Dict keys become the label for their values; lists inherit the parent label;
    leaf scalars are added under the current label if present verbatim in text.
    """
    if isinstance(value, dict):
        for k, v in value.items():
            if isinstance(k, str) and k.strip():
                _walk(v, k.strip(), text, entities)
    elif isinstance(value, list):
        for item in value:
            _walk(item, label, text, entities)
    elif label is not None:
        surface = _coerce_surface(value)
        if surface is not None:
            _add(entities, label, surface, text)


def convert_row(row: dict) -> dict | None:
    """Convert one paraloq row to a GLiNER2 record; None if no usable spans."""
    text = row.get("text")
    item = row.get("item")
    if not isinstance(text, str) or not text.strip() or item is None:
        return None
    if isinstance(item, str):
        try:
            item = json.loads(item)
        except json.JSONDecodeError:
            return None
    if not isinstance(item, (dict, list)):
        return None

    entities: dict[str, list[str]] = {}
    _walk(item, None, text, entities)
    if not entities:
        return None
    return {"input": text, "output": {"entities": entities}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path (writes <base>.train.jsonl, "
                             ".val.jsonl, .test.jsonl).")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum input records to emit (-1 = all).")
    parser.add_argument("--repo", default="paraloq/json_data_extraction",
                        help="HuggingFace dataset repo.")
    parser.add_argument("--split", default="train", help="Dataset split to read.")
    add_split_args(parser)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Loading {args.repo} split={args.split}...")
    ds = load_dataset(args.repo, split=args.split)

    emitted = skipped_empty = total_entities = 0
    all_fields: set[str] = set()

    with SplitWriter(args.out, ratios=args.split_ratios, seed=args.split_seed) as writer:
        for row in ds:
            record = convert_row(row)
            if record is None:
                skipped_empty += 1
                continue
            writer.write(record)
            emitted += 1
            ents = record["output"]["entities"]
            total_entities += sum(len(v) for v in ents.values())
            all_fields.update(ents.keys())
            if 0 <= args.max_records <= emitted:
                break

    print(f"Done. emitted={emitted} skipped_empty={skipped_empty} "
          f"total_values={total_entities} distinct_fields={len(all_fields)} "
          f"{writer.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
