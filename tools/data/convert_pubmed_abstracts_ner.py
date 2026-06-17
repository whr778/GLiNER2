"""Convert knowledgator/PubMedAbstractsNER to GLiNER2 JSONL.

35,000 PubMed abstracts tokenized at the word level with entity spans
typed by UMLS-style biomedical concept names. The source ships a single
``train.json`` (a JSON array — not JSONL, not loadable via ``datasets``)
where each row has::

    tokenized_text: ["In", "this", "article", ",", "the", "development", ...]
    ner:            [[9, 9, "Body Regions - Anatomical areas of the body."],
                     [18, 18, "Hemic and Immune Systems - ..."],
                     [86, 87, "Risk - The probability that an event will occur..."],
                     ...]

Token offsets are inclusive on both ends. Each label string encodes the
entity TYPE and its DESCRIPTION joined by ``" - "`` (the type is the
prefix before the first ``" - "``). This converter splits that into the
type (used as the entity bucket) and the description (collected into
``entity_descriptions`` so the model can condition on it).

The HuggingFace ``datasets`` library fails to parse the repo's metadata
(``KeyError: 'feature'``), so the converter downloads ``train.json``
directly via ``huggingface_hub`` and parses it with the stdlib.

Usage::

    uv run python tools/data/convert_pubmed_abstracts_ner.py \\
        --out data/pubmed_abstracts_ner.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, add_split_args


def split_label(raw: str) -> tuple[str, str | None]:
    """Split ``"TypeName - long description"`` into ``(type, description)``.

    The first ``" - "`` is the separator. Returns ``(stripped, None)`` if
    no separator is present.
    """
    if not isinstance(raw, str):
        return "", None
    sep = " - "
    idx = raw.find(sep)
    if idx < 0:
        return raw.strip(), None
    return raw[:idx].strip(), raw[idx + len(sep):].strip() or None


def convert_row(row: dict) -> dict | None:
    """Convert one PubMed row to a GLiNER2 NER record; None if unusable."""
    tokens = row.get("tokenized_text") or []
    ner = row.get("ner") or []
    if not tokens or not ner:
        return None

    text = " ".join(t for t in tokens if isinstance(t, str))
    entities: dict[str, list[str]] = {}
    descriptions: dict[str, str] = {}

    for span in ner:
        if not isinstance(span, (list, tuple)) or len(span) != 3:
            continue
        try:
            start = int(span[0])
            end = int(span[1])
        except (TypeError, ValueError):
            continue
        etype, desc = split_label(span[2])
        if not etype:
            continue
        if start < 0 or end >= len(tokens) or end < start:
            continue

        surface = " ".join(tokens[start:end + 1])
        if surface not in text:
            continue
        bucket = entities.setdefault(etype, [])
        if surface not in bucket:
            bucket.append(surface)
        if desc and etype not in descriptions:
            descriptions[etype] = desc

    if not entities:
        return None
    output: dict = {"entities": entities}
    if descriptions:
        output["entity_descriptions"] = descriptions
    return {"input": text, "output": output}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path (writes <base>.train.jsonl, "
                             ".val.jsonl, .test.jsonl).")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum input records to read (-1 = all).")
    parser.add_argument("--repo", default="knowledgator/PubMedAbstractsNER",
                        help="HuggingFace dataset repo.")
    parser.add_argument("--file", default="train.json",
                        help="JSON array file inside the repo to convert "
                             "(default: train.json).")
    add_split_args(parser)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    print(f"Downloading {args.repo}/{args.file}...")
    src_path = Path(hf_hub_download(args.repo, args.file, repo_type="dataset"))
    with src_path.open() as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        raise SystemExit(f"expected a JSON array in {args.file}, got {type(rows).__name__}")
    print(f"Loaded {len(rows):,} rows")

    emitted = 0
    skipped_empty = 0
    total_entities = 0
    all_types: set[str] = set()
    all_described: set[str] = set()

    with SplitWriter(args.out, ratios=args.split_ratios, seed=args.split_seed) as writer:
        for idx, row in enumerate(rows):
            if 0 <= args.max_records <= idx:
                break
            record = convert_row(row)
            if record is None:
                skipped_empty += 1
                continue
            writer.write(record)
            emitted += 1
            ents = record["output"]["entities"]
            total_entities += sum(len(v) for v in ents.values())
            all_types.update(ents.keys())
            all_described.update(record["output"].get("entity_descriptions") or {})

            if emitted % 2000 == 0:
                print(f"  emitted={emitted}  skipped_empty={skipped_empty}  "
                      f"types={len(all_types)}")

    print(f"Done. emitted={emitted} skipped_empty={skipped_empty} "
          f"total_entities={total_entities} distinct_types={len(all_types)} "
          f"types_with_descriptions={len(all_described)} {writer.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
