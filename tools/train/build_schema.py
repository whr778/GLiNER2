"""Build a common inference schema from corpora, for ``infer.py --schema-json``.

    uv run tools/train/build_schema.py data/casie --exhaustive --out tools/train/config/schemas/casie.json
    uv run tools/infer.py --model <model> --input data/casie.test.jsonl --schema-json tools/train/config/schemas/casie.json ...
    uv run tools/train/score_predictions.py --predictions <preds> --menu app:tools/train/config/schemas/casie.json --corpus casie

Every model then runs the SAME menu, whatever its config.json ``default_schema`` holds, so a full-menu
model and a single-corpus model are compared on equal terms. The schema is the union of the corpora's
TRAIN gold (``gliner2.inference.schema.derive_schema``, the function training co-locates with), in the
corpora's own spellings: ``infer.py`` sends them through each checkpoint's ``label_map``. Labels the val
or test gold uses but train never does are reported, and added only with ``--include-eval``.

Written as an application menu (``{name, source, exhaustive_for, schema}``, as config/menus/), which
``infer.py --schema-json`` reads directly. ``--exhaustive`` lists the corpora under ``exhaustive_for``: a
CLAIM that every document was annotated against the whole ontology (true of CASIE, not of a union of
corpora), which lets ``score_predictions --menu app:`` count a fired absent label as a false positive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from _split import derive_split_paths  # noqa: E402

from gliner2.inference.schema import derive_schema  # noqa: E402


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def split_records(bases: list[Path], split: str) -> list[dict]:
    """Records of one split across corpora; a missing val split falls back to ``.dev.jsonl``."""
    out = []
    for base in bases:
        p = derive_split_paths(base)[split]
        if not p.exists() and split == "val":
            p = Path(str(p).replace(".val.jsonl", ".dev.jsonl"))
        if p.exists():
            out += read(p)
    return out


def labels(schema: dict) -> set[tuple]:
    """Every (task, label[, role]) the schema offers."""
    out = {("entities", e) for e in schema.get("entities") or []}
    out |= {("relations", r) for r in schema.get("relations") or []}
    for t, roles in (schema.get("events") or {}).items():
        out |= {("events", t)} | {("events", t, r) for r in roles}
    for c in schema.get("classifications") or []:
        out |= {("classifications", c["task"], lab) for lab in c["labels"]}
    return out


def summary(schema: dict) -> str:
    return ", ".join(f"{k}={len(v)}" for k, v in schema.items())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("corpora", nargs="+", type=Path, help="split bases, e.g. data/casie -> data/casie.train.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--include-eval", action="store_true", help="also take labels from the val and test gold")
    ap.add_argument("--exhaustive", action="store_true",
                    help="claim every document was annotated against the whole ontology (exhaustive_for)")
    args = ap.parse_args()

    train = split_records(args.corpora, "train")
    if not train:
        raise SystemExit(f"no train split under {[str(c) for c in args.corpora]}")
    evals = split_records(args.corpora, "val") + split_records(args.corpora, "test")
    schema = derive_schema(train + (evals if args.include_eval else []))
    missing = labels(derive_schema(evals)) - labels(schema) if evals else set()
    print(f"[schema] {len(train)} train records -> {summary(schema)}")
    print(f"[schema] val/test gold labels not offered: {len(missing)}" + (f" {sorted(missing)[:10]}" if missing else ""))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    names = [c.name for c in args.corpora]
    menu = {"name": args.out.stem, "source": f"build_schema.py {' '.join(map(str, args.corpora))}: train gold"
            + (" + val/test gold" if args.include_eval else ""), "exhaustive_for": names if args.exhaustive else [],
            "schema": schema}
    args.out.write_text(json.dumps(menu, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[schema] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
