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

  uv run python tools/infer.py --model whr778/gliner2-eb18-balanced \
      --input data/casie.test.jsonl --gold-schema --output preds.jsonl

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


def _read_records(inp: str) -> List[Dict[str, Any]]:
    """A literal string, a ``.txt`` file (one document), or a ``.jsonl`` file (one record per
    line; its ``input`` is the text and its ``output``, if any, the gold)."""
    path = Path(inp)
    if path.is_file():
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [{"input": path.read_text(encoding="utf-8")}]
    return [{"input": inp}]


def _read_texts(inp: str) -> List[str]:
    """The document texts of ``_read_records``."""
    return [r["input"] for r in _read_records(inp)]


def _gold_schema(record: Dict[str, Any]) -> Dict[str, Any]:
    """The schema eval offers this record: only its OWN gold labels (`eval_metrics._schema_from_gold`)."""
    from gliner2.training.eval_metrics import _schema_from_gold
    return _schema_from_gold(record.get("output") or {})


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
    p.add_argument("--gold-schema", action="store_true",
                   help="Each .jsonl record gets the schema of its OWN gold labels, as eval scores it. "
                        "Excludes every other schema option.")
    p.add_argument("--output", help="Write predictions here as JSONL, one line per input record: "
                                    "{input, output: prediction, gold: the record's output}. "
                                    "Written as it goes. Default: print a JSON array.")
    p.add_argument("--docs-per-write", type=int, default=64,
                   help="With --output: documents decoded per flush to the file.")
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

    records = _read_records(args.input)
    model = AutoExtractor.from_pretrained(args.model)
    label_map = None if args.no_label_map else getattr(model.config, "label_map", None)
    rewrites: Dict[str, Dict[str, str]] = {}

    def mapped(schema):
        out, applied = apply_label_map(schema, label_map)
        for cat, m in applied.items():
            rewrites.setdefault(cat, {}).update(m)
        return out

    if args.gold_schema:
        if args.model_schema or args.schema_json or args.entities or args.events:
            raise SystemExit("--gold-schema excludes --model-schema, --schema-json, --entities and --events.")
        schemas = [mapped(_gold_schema(r)) for r in records]
        empty = sum(1 for sc in schemas if not sc)
        print(f"[infer] gold schema per record: {len(records)} records, {empty} with no gold labels (predicted as {{}})")
    else:
        schema = mapped(_build_schema(args, model.config))
        schemas = [schema] * len(records)
        print(f"[infer] schema {', '.join(f'{k}={len(v)}' for k, v in schema.items())}")
    print(f"[infer] label_map rewrites: {json.dumps(rewrites, ensure_ascii=False) if rewrites else 'none'}")
    decode = _decode_settings(args, model)
    print(f"[infer] decode {decode} (model max_len {getattr(model.config, 'max_len', '?')} tokens; "
          f"checkpoint inference_defaults {getattr(model.config, 'inference_defaults', None)})")

    def predict(recs, scs):
        keep = [i for i, sc in enumerate(scs) if sc]
        got = model.batch_extract_long(
            [recs[i]["input"] for i in keep], [scs[i] for i in keep],
            batch_size=args.batch_size, **decode,
            include_spans=args.include_spans, include_confidence=args.include_confidence,
            global_decode_config=GlobalDecodeConfig(beam_width=args.beam_width),
        ) if keep else []
        out = [{} for _ in recs]
        for i, r in zip(keep, got):
            out[i] = r
        return out

    if not args.output:
        print(json.dumps(predict(records, schemas), ensure_ascii=False, indent=2))
        return
    step = max(1, args.docs_per_write)
    with open(args.output, "w", encoding="utf-8") as f:
        for start in range(0, len(records), step):
            chunk = records[start:start + step]
            for rec, pred in zip(chunk, predict(chunk, schemas[start:start + step])):
                f.write(json.dumps({"input": rec["input"], "output": pred, "gold": rec.get("output")},
                                   ensure_ascii=False) + "\n")
            f.flush()
            print(f"[infer] {min(start + step, len(records))}/{len(records)} records -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
