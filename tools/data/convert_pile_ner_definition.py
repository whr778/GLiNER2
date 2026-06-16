"""Convert Universal-NER/Pile-NER-definition to GLiNER2 JSONL.

Each record in Pile-NER-definition is a ShareGPT-style conversation:

    turns[0]: human  -> "Text: <passage>"
    turns[1]: gpt    -> "I've read this text."
    turns[2k+0]: human -> "What describes <DEFINITION> in the text?"
    turns[2k+1]: gpt   -> '["surface_1", "surface_2", ...]'   (or "[]")

Each pair of (human, gpt) turns after the first two is one entity-type
query whose "type" is a natural-language definition. To keep schema tokens
short and let the model exploit GLiNER2's description channel, we mint
synthetic per-record keys (`e_0`, `e_1`, ...) and put the definition itself
in `entity_descriptions`.

Empty answers (`[]`) are negative samples for the source GPT prompt and are
dropped here, since GLiNER2's training format only emits positive entities.

Usage::

    uv run python tools/data/convert_pile_ner_definition.py --out pile_ner.jsonl
    uv run python tools/data/convert_pile_ner_definition.py --out pile_ner.jsonl --max-records 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, add_split_args


def strip_definition(human_value: str) -> str | None:
    """Extract the definition from a 'What describes X in the text?' turn."""
    prefix = "What describes "
    suffix = " in the text?"
    if not (human_value.startswith(prefix) and human_value.endswith(suffix)):
        return None
    return human_value[len(prefix):-len(suffix)].strip()


def convert_record(record: dict) -> dict | None:
    """Convert one Pile-NER record to a GLiNER2 record; None if no positive entities."""
    turns = record.get("conversations") or []
    if len(turns) < 3:
        return None
    text_turn = turns[0].get("value", "")
    if not text_turn.startswith("Text: "):
        return None
    text = text_turn[len("Text: "):]

    entities: dict[str, list[str]] = {}
    descriptions: dict[str, str] = {}
    next_key = 0

    qa_turns = turns[2:]
    for i in range(0, len(qa_turns) - 1, 2):
        human = qa_turns[i].get("value", "")
        gpt = qa_turns[i + 1].get("value", "")
        definition = strip_definition(human)
        if not definition:
            continue
        try:
            spans = json.loads(gpt)
        except json.JSONDecodeError:
            continue
        if not isinstance(spans, list) or not spans:
            continue

        # Verbatim filter
        spans = [s for s in spans if isinstance(s, str) and s in text]
        # Dedupe preserving order
        seen: set[str] = set()
        spans = [s for s in spans if not (s in seen or seen.add(s))]
        if not spans:
            continue

        key = f"e_{next_key}"
        next_key += 1
        entities[key] = spans
        descriptions[key] = definition

    if not entities:
        return None

    return {
        "input": text,
        "output": {"entities": entities, "entity_descriptions": descriptions},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path (writes <base>.train.jsonl, "
                             ".val.jsonl, .test.jsonl).")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum input records to read (-1 = all)")
    parser.add_argument("--repo", default="Universal-NER/Pile-NER-definition",
                        help="HuggingFace dataset repo")
    add_split_args(parser)
    args = parser.parse_args()

    from datasets import load_dataset

    print(f"Streaming {args.repo}...")
    ds = load_dataset(args.repo, split="train", streaming=True)

    emitted = 0
    skipped_empty = 0
    total_entities = 0

    with SplitWriter(args.out, ratios=args.split_ratios, seed=args.split_seed) as writer:
        for idx, record in enumerate(ds):
            if 0 <= args.max_records <= idx:
                break
            converted = convert_record(record)
            if converted is None:
                skipped_empty += 1
                continue
            writer.write(converted)
            emitted += 1
            total_entities += sum(len(v) for v in converted["output"]["entities"].values())

            if emitted % 2000 == 0:
                print(f"  emitted={emitted}  skipped_empty={skipped_empty}  entities={total_entities}")

    print(f"Done. emitted={emitted} skipped_empty={skipped_empty} "
          f"total_entities={total_entities} {writer.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
