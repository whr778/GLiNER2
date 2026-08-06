"""Run a GLiNER2 model over a text file and extract everything it supports.

GLiNER2 is schema-driven / open-vocabulary: a checkpoint does not store its own
label set, so "everything it supports" is defined by the ontology it was trained
on. This script recovers that ontology from a training/eval JSONL file (every
task type present -> entities, events with their roles, relations,
classifications) and runs all of it over the input document in one pass, using
the long-document path so long inputs are windowed automatically (with
OneIE-style global event decoding on by default).

Usage:
  uv run python scripts/infer_file.py --model <ckpt> --input <text_file> \
      [--schema-from <data.jsonl>] [--schema-json <schema.json>] [--out out.json]

The ontology source is resolved in order: --schema-json, then --schema-from,
then auto-discovery of data/<name>.train.jsonl matching the model directory
name. Example:
  uv run python scripts/infer_file.py \
      --model out/fastino/gliner2-base-v1-wikievents/best \
      --input data/inference/eng_1.txt
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _derive_schema(jsonl_path: str) -> Dict[str, Any]:
    """Build a full multi-task schema from the gold ``output`` of a JSONL corpus:
    every entity label, every event type with its roles, every relation name,
    and every classification task+labels the model was trained on. Thin wrapper
    over ``gliner2.inference.schema.derive_schema`` (the shared union logic)."""
    from gliner2.inference.schema import derive_schema

    with open(jsonl_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return derive_schema(records)


def _autodiscover_schema_file(model_path: str) -> Optional[str]:
    """Find data/<key>.train.jsonl whose <key> appears in the model dir name."""
    p = Path(model_path)
    names = {p.name, p.parent.name}
    for f in sorted(glob.glob("data/*.train.jsonl")):
        key = Path(f).name[: -len(".train.jsonl")]
        if any(key in n for n in names):
            return f
    return None


def _resolve_schema(args: argparse.Namespace) -> Dict[str, Any]:
    if args.schema_json:
        return json.loads(Path(args.schema_json).read_text(encoding="utf-8"))
    src = args.schema_from or _autodiscover_schema_file(args.model)
    if not src:
        raise SystemExit(
            "No ontology source. GLiNER2 is open-vocabulary, so extraction needs a "
            "schema. Pass --schema-from <data.jsonl> (the corpus the model was "
            "trained on) or --schema-json <schema.json>."
        )
    print(f"[schema] deriving ontology from {src}")
    return _derive_schema(src)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Extract everything a GLiNER2 model supports from a text file.")
    p.add_argument("--model", required=True, help="HF repo id or local checkpoint path.")
    p.add_argument("--input", required=True, help="Path to a UTF-8 text file (one document).")
    p.add_argument("--schema-from", help="JSONL corpus to derive the full ontology from.")
    p.add_argument("--schema-json", help="Explicit schema JSON (overrides --schema-from).")
    p.add_argument("--out", help="Write JSON here instead of stdout.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--chunk-size", type=int, default=384)
    p.add_argument("--chunk-overlap", type=int, default=128)
    p.add_argument("--no-global-decode", action="store_true", help="Disable cross-window event assembly.")
    p.add_argument("--no-spans", action="store_true")
    p.add_argument("--no-confidence", action="store_true")
    args = p.parse_args(argv)

    from gliner2 import GLiNER2

    schema = _resolve_schema(args)
    counts = {k: (len(v) if isinstance(v, (list, dict)) else 1) for k, v in schema.items()}
    print(f"[schema] tasks: {counts}")

    text = Path(args.input).read_text(encoding="utf-8")
    print(f"[input] {args.input}: {len(text.split())} words")

    model = GLiNER2.from_pretrained(args.model)
    result = model.batch_extract_long(
        [text], schema,
        batch_size=1, threshold=args.threshold,
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
        include_spans=not args.no_spans, include_confidence=not args.no_confidence,
        global_decode=not args.no_global_decode,
    )[0]

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
        print(f"[out] wrote {args.out}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
