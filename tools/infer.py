"""Command-line inference for GLiNER2, with optional document-level global decoding.

Runs extraction over one or more texts through the long-document path, so long
inputs are windowed automatically. Pass --global-decode to reconnect events
across windows (OneIE-style, see tools/events_working_papers/PAPER_0_FOUNDATION.md sec 9).

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
    p.add_argument("--chunk-size", type=int, default=0,
                   help="Word window for long docs. 0 = the MODEL's own configured window, "
                        "which is the default: one window everywhere.")
    p.add_argument("--chunk-overlap", type=int, default=0,
                   help="Word overlap. 0 by default: the stride is a TRAINING device and "
                        "double-counts spans anywhere else.")
    p.add_argument("--beam-width", type=int, default=8, help="Global-decode beam width.")
    p.add_argument("--include-spans", action="store_true")
    p.add_argument("--include-confidence", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args(argv)



def resolve_model_window(model, requested=None, *, tokens_per_word: float = 1.5):
    """Words per window for a model whose window is configured in TOKENS.

    ONE WINDOW EVERYWHERE, STRIDE ONLY IN TRAINING. Training, eval, blind test and
    inference all read the model's configured sliding-window SIZE; only training uses the
    STRIDE, whose overlapping views are a training device and which double-counts spans
    anywhere else.

    THE UNIT DIFFERS AND THAT IS NOT COSMETIC. The model's window is `max_len` TOKENS;
    `extract_long` chunks by WORDS. Sizing words at the token count would silently
    truncate whatever did not fit, so the count is divided by a deliberately pessimistic
    tokens-per-word factor. 1.5 is above English subword rates and well above the
    whitespace-poor scripts in this corpus, so the window under-fills rather than
    truncates. It is a CONVERSION, not a measurement -- a corpus that tokenises harder
    than 1.5 tokens/word will still under-fill, which is the safe direction.
    """
    if requested:
        return int(requested)
    max_len = getattr(getattr(model, "config", None), "max_len", None)
    if not max_len:
        return 384
    return max(64, int(int(max_len) / tokens_per_word))

def main(argv: List[str] = None) -> None:
    args = _parse_args(argv)

    from gliner2 import GLiNER2
    from gliner2.inference.global_decode import GlobalDecodeConfig

    texts = _read_texts(args.input)
    schema = _build_schema(args)
    model = GLiNER2.from_pretrained(args.model)
    chunk_size = resolve_model_window(model, args.chunk_size)
    print(f"[infer] window {chunk_size} words (model max_len "
          f"{getattr(model.config, 'max_len', '?')} tokens), overlap {args.chunk_overlap}")

    results = model.batch_extract_long(
        texts, schema,
        batch_size=args.batch_size, threshold=args.threshold,
        chunk_size=chunk_size, chunk_overlap=args.chunk_overlap,
        include_spans=args.include_spans, include_confidence=args.include_confidence,
        global_decode=args.global_decode,
        global_decode_config=GlobalDecodeConfig(beam_width=args.beam_width),
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
