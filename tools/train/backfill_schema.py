"""Backfill ``default_schema`` into models trained before schema co-location.

For each training yaml we resolve its ``output_dir`` and reproduce EXACTLY the
schema a fresh run would now co-locate: read the same train corpora/event files,
apply the same optional label transforms, then ``derive_schema``. The result is
written into the model's ``config.json`` (best/ and final/) via ``ExtractorConfig``
so the serialization is identical to training's own ``save_pretrained``.

Dry-run by default (prints what it would write). Pass ``--apply`` to write.
``--filter <substr>`` limits to matching yaml/output-dir names.

  uv run python tools/train/backfill_schema.py            # dry run, all
  uv run python tools/train/backfill_schema.py --filter casie
  uv run python tools/train/backfill_schema.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import (  # noqa: E402  (path set above)
    _category_fns,
    _event_split,
    _read_records,
    _split_files,
    transform_record,
)

from gliner2.inference.schema import derive_schema  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _derive_for_yaml(cfg: dict) -> dict | None:
    """Reproduce train.py's co-located schema for one parsed yaml config."""
    data = cfg.get("data") or {}
    if data.get("hf_streaming"):
        return None  # streaming runs co-locate no schema (no bounded record set)
    corpora = data.get("corpora") or []
    event_files = data.get("event_files") or {}
    train_files = _split_files(corpora, "train") + _event_split(event_files, "train")
    train_files = [f for f in train_files if (REPO / f).exists()]
    if not train_files:
        return None
    recs = _read_records([str(REPO / f) for f in train_files])
    fns = _category_fns(cfg.get("labels") or {})
    if fns:
        recs = [transform_record(r, fns) for r in recs]
    return derive_schema(recs) or None


def _summary(schema: dict | None) -> str:
    s = schema or {}
    return (f"{len(s.get('entities') or [])} ent, {len(s.get('events') or {})} evt, "
            f"{len(s.get('relations') or [])} rel, {len(s.get('classifications') or [])} cls")


def _write(model_dir: Path, schema: dict | None) -> None:
    """Additively set ``default_schema`` in config.json, touching nothing else.

    A minimal JSON patch, NOT an ``ExtractorConfig`` round-trip: re-saving through
    the config object would stamp current defaults (span_head/max_width, etc.) over
    fields these older checkpoints never stored, which can mismatch their weights.
    """
    p = model_dir / "config.json"
    cfg = json.loads(p.read_text(encoding="utf-8"))
    if schema is None:
        cfg.pop("default_schema", None)
    else:
        cfg["default_schema"] = schema
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                 encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write configs (default: dry run)")
    ap.add_argument("--filter", default="", help="only yamls/dirs whose name contains this")
    args = ap.parse_args()

    yamls = sorted((REPO / "tools/train/config").glob("*.yaml"))
    touched = skipped = 0
    for y in yamls:
        cfg = yaml.safe_load(y.read_text()) or {}
        out = (cfg.get("training") or {}).get("output_dir", "")
        if not out or (args.filter and args.filter not in y.name and args.filter not in out):
            continue
        out_dir = REPO / out.lstrip("./")
        model_dirs = [d for d in (out_dir / "best", out_dir / "final")
                      if (d / "config.json").exists()]
        if not model_dirs:
            continue

        schema = _derive_for_yaml(cfg)
        tag = "streaming/no-data" if schema is None else _summary(schema)
        print(f"{y.name:40s} {out:40s} -> {tag}")
        for d in model_dirs:
            rel = d.relative_to(REPO)
            if schema is None:
                print(f"    skip {rel} (no schema to write)")
                skipped += 1
                continue
            if args.apply:
                _write(d, schema)
                print(f"    wrote default_schema -> {rel}/config.json")
            else:
                print(f"    would write -> {rel}/config.json")
            touched += 1

    verb = "wrote" if args.apply else "would write"
    print(f"\n{verb} {touched} config(s); {skipped} skipped (no schema).")
    if not args.apply:
        print("Dry run -- re-run with --apply to write.")


if __name__ == "__main__":
    main()
