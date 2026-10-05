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

  uv run python tools/infer.py --model whr778/gliner2-eb18-balanced \
      --input document.txt --model-schema --tasks events --entities Person,Location

Like the viewer: --model-schema starts from the schema the checkpoint ships (config.json
default_schema), every decode setting not passed falls back to the checkpoint's
inference_defaults, and the checkpoint's label_map is applied to the schema. See
tools/train/INFER.md.
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


def _model_schema(config, tasks: str = None) -> Dict[str, Any]:
    """The checkpoint's shipped schema, minus the ``open_vocab`` marker, optionally narrowed
    to ``tasks`` (comma-separated: entities, relations, events, classifications, structures)."""
    shipped = dict(getattr(config, "default_schema", None) or {})
    if not shipped:
        raise SystemExit("--model-schema: this checkpoint ships no default_schema.")
    shipped.pop("open_vocab", None)
    if tasks:
        keep = {t.strip() for t in tasks.split(",") if t.strip()}
        shipped = {k: v for k, v in shipped.items() if k in keep}
    return shipped


def _build_schema(args: argparse.Namespace, config=None) -> Dict[str, Any]:
    """Assemble a raw schema dict from CLI options; --entities/--events add to --model-schema."""
    if args.schema_json:
        return json.loads(Path(args.schema_json).read_text(encoding="utf-8"))
    schema: Dict[str, Any] = _model_schema(config, args.tasks) if args.model_schema else {}
    if args.entities:
        schema["entities"] = [e.strip() for e in args.entities.split(",") if e.strip()]
    if args.events:
        schema["events"] = json.loads(args.events)
    if not schema:
        raise SystemExit("Provide --entities, --events, --schema-json, or --model-schema.")
    return schema


def _decode_settings(args: argparse.Namespace, model) -> Dict[str, Any]:
    """Each setting: the flag if passed, else the checkpoint's inference_defaults, else the
    script's fallback (threshold 0.5, the model's own window, overlap 0, no global decode)."""
    stored = getattr(model.config, "inference_defaults", None) or {}
    pick = lambda name, fallback: (getattr(args, name) if getattr(args, name) is not None
                                   else stored.get(name, fallback))
    return {"threshold": pick("threshold", 0.5),
            "chunk_size": resolve_model_window(model, pick("chunk_size", 0)),
            "chunk_overlap": pick("chunk_overlap", 0),
            "global_decode": bool(pick("global_decode", False))}


def _parse_args(argv: List[str] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GLiNER2 inference with optional global event decoding.")
    p.add_argument("--model", required=True, help="HF repo id or local checkpoint path.")
    p.add_argument("--input", required=True,
                   help="Literal text, a .txt file, or a .jsonl with an 'input' field per line.")
    p.add_argument("--entities", help="Comma-separated entity types.")
    p.add_argument("--events", help='JSON mapping event type -> [roles], e.g. \'{"Attack":["Target"]}\'.')
    p.add_argument("--schema-json", help="Path to a full schema JSON (overrides every other schema option).")
    p.add_argument("--model-schema", action="store_true",
                   help="Start from the schema the checkpoint ships (config.json default_schema).")
    p.add_argument("--tasks", help="With --model-schema: keep only these task types, comma-separated "
                                   "(entities, relations, events, classifications, structures).")
    p.add_argument("--no-label-map", action="store_true",
                   help="Send labels as typed instead of through the checkpoint's label_map.")
    p.add_argument("--global-decode", action=argparse.BooleanOptionalAction, default=None,
                   help="OneIE-style document-level event assembly across windows. "
                        "Default: the checkpoint's inference_defaults, else off.")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="Word window for long docs. Default: the checkpoint's inference_defaults, "
                        "else 0 = the MODEL's own configured window: one window everywhere.")
    p.add_argument("--chunk-overlap", type=int, default=None,
                   help="Word overlap. Default: the checkpoint's inference_defaults, else 0: the "
                        "stride is a TRAINING device and double-counts spans anywhere else.")
    p.add_argument("--beam-width", type=int, default=8, help="Global-decode beam width.")
    p.add_argument("--include-spans", action="store_true")
    p.add_argument("--include-confidence", action="store_true")
    p.add_argument("--threshold", type=float, default=None,
                   help="Span threshold. Default: the checkpoint's inference_defaults, else 0.5.")
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

    # AutoExtractor, not GLiNER2: GLiNER2 IS the span class, so a boundary checkpoint
    # died on `config.max_width` -- the same defect the viewer and eval already fixed.
    from gliner2 import AutoExtractor
    from gliner2.inference.global_decode import GlobalDecodeConfig
    from gliner2.inference.label_map import apply_label_map

    texts = _read_texts(args.input)
    model = AutoExtractor.from_pretrained(args.model)
    schema = _build_schema(args, model.config)
    if not args.no_label_map:
        schema, rewrites = apply_label_map(schema, getattr(model.config, "label_map", None))
        print(f"[infer] label_map rewrites: {json.dumps(rewrites, ensure_ascii=False) if rewrites else 'none'}")
    decode = _decode_settings(args, model)
    print(f"[infer] schema {', '.join(f'{k}={len(v)}' for k, v in schema.items())} | decode {decode} "
          f"(model max_len {getattr(model.config, 'max_len', '?')} tokens; "
          f"checkpoint inference_defaults {getattr(model.config, 'inference_defaults', None)})")

    results = model.batch_extract_long(
        texts, schema,
        batch_size=args.batch_size, **decode,
        include_spans=args.include_spans, include_confidence=args.include_confidence,
        global_decode_config=GlobalDecodeConfig(beam_width=args.beam_width),
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
