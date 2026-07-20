"""Command-line inference for GLiNER2, with optional document-level global decoding.

Runs extraction over one or more texts through the long-document path, so long
inputs are windowed automatically. Pass --global-decode to reconnect events
across windows (OneIE-style, see tools/events_working_papers/DOCUMENT_EXTRACTION_PLAN.md).

Examples:
  uv run python tools/infer.py --model fastino/gliner2-base-v1 \
      --input document.txt --entities person,organization,location

  uv run python tools/infer.py --model out/fastino/gliner2-base-v1-wikievents/best \
      --input data/wikievents.test.jsonl \
      --events '{"Attack": ["Attacker", "Target", "Place"]}' \
      --global-decode --chunk-size 384 --chunk-overlap 128 --include-spans
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_texts(inp: str) -> List[str]:
    """A literal string, a ``.txt`` file (one document), or a ``.jsonl`` file
    (one document per line, read from the ``input`` field)."""
    path = Path(inp)
    if path.is_file():
        if path.suffix == ".jsonl":
            return [
                json.loads(line)["input"]
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return [path.read_text(encoding="utf-8")]
    return [inp]


def _build_schema(args: argparse.Namespace) -> Dict[str, Any]:
    """Assemble a raw schema dict from CLI options."""
    if args.schema_json:
        return json.loads(Path(args.schema_json).read_text(encoding="utf-8"))
    schema: Dict[str, Any] = {}
    if args.entities:
        schema["entities"] = [e.strip() for e in args.entities.split(",") if e.strip()]
    if args.events:
        schema["events"] = json.loads(args.events)
    if not schema:
        raise SystemExit("Provide --entities, --events, or --schema-json.")
    return schema


def _parse_args(argv: List[str] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GLiNER2 inference with optional global event decoding.")
    p.add_argument("--model", required=True, help="HF repo id or local checkpoint path.")
    p.add_argument("--input", required=True,
                   help="Literal text, a .txt file, or a .jsonl with an 'input' field per line.")
    p.add_argument("--entities", help="Comma-separated entity types.")
    p.add_argument("--events", help='JSON mapping event type -> [roles], e.g. \'{"Attack":["Target"]}\'.')
    p.add_argument("--schema-json", help="Path to a full schema JSON (overrides --entities/--events).")
    p.add_argument("--global-decode", action="store_true",
                   help="OneIE-style document-level event assembly across windows.")
    p.add_argument("--chunk-size", type=int, default=384, help="Word window length for long docs.")
    p.add_argument("--chunk-overlap", type=int, default=128, help="Word overlap between windows.")
    p.add_argument("--beam-width", type=int, default=8, help="Global-decode beam width.")
    p.add_argument("--include-spans", action="store_true")
    p.add_argument("--include-confidence", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args(argv)


def main(argv: List[str] = None) -> None:
    args = _parse_args(argv)

    from gliner2 import GLiNER2
    from gliner2.inference.global_decode import GlobalDecodeConfig

    texts = _read_texts(args.input)
    schema = _build_schema(args)
    model = GLiNER2.from_pretrained(args.model)

    results = model.batch_extract_long(
        texts, schema,
        batch_size=args.batch_size, threshold=args.threshold,
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
        include_spans=args.include_spans, include_confidence=args.include_confidence,
        global_decode=args.global_decode,
        global_decode_config=GlobalDecodeConfig(beam_width=args.beam_width),
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
