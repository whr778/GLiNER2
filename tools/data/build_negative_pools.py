"""Derive per-corpus NEGATIVE LABEL POOLS — the labels a document must learn to reject.

WHY POOLS ARE PER-CORPUS, AND WHY THAT IS THE WHOLE DESIGN. The corpora annotate different
things: cmnee carries zero entity gold, biored zero events. Sampling ``Organization`` as a
negative into a cmnee document would teach the model that a real organisation is not one --
at loss weight, thousands of times. A global taxonomy is therefore unusable as a negative
pool. Two rules make it safe, and both are encoded here:

    within-corpus    sample only from labels THIS corpus annotates
    within-dimension only offer entity negatives to records that carry entity gold, etc.

POOLS ARE DERIVED AFTER LABEL UNIFICATION. If a pool held a pre-map alias (``LOC``) of a
label present under its canonical name (``Location``), the injected negative would contradict
the gold directly. This reuses the training pipeline's own transforms --
``load_labels_cfg`` + ``_category_fns`` + ``transform_record`` from ``tools/train/train.py``
-- so the pool is built from exactly the records training sees.

AND FROM RECORDS, NOT FROM ``default_schema``. The persisted schema is capped:
``_OPEN_VOCAB_LIMIT = 1000`` drops a whole dimension past the limit, which is why the
event-records checkpoint carries ``open_vocab: ["entities"]`` and no entity list at all.
That copy is for inference and the viewer; training needs the real thing.

    uv run python tools/data/build_negative_pools.py \\
        --config tools/train/config/base/eb16-eventrecords-tr.yaml \\
        --out tools/train/config/labels/negative_pools.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import yaml

DIMENSIONS = ("entities", "events", "relations", "structures")


def _train_helpers():
    """Load `tools/train/train.py` by PATH, under a name that shadows nothing.

    `sys.path.insert(0, "tools/")` plus `from train.train import ...` made the bare name
    `train` resolve to the PACKAGE `tools/train/`, which broke every sibling test that does
    `from train import ...` after adding `tools/train` to the path -- two collection errors,
    and only in a full-suite run where this module is imported first. Loading by file
    location under a unique module name has no such side effect.
    """
    path = Path(__file__).resolve().parents[1] / "train" / "train.py"
    spec = importlib.util.spec_from_file_location("_gliner2_train_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._category_fns, mod.load_labels_cfg, mod.transform_record


def _corpus_train_paths(cfg: dict) -> dict:
    """``{corpus_name: train_path}`` from a config's corpora and event_files."""
    data = cfg.get("data") or {}
    out: dict = {}
    for prefix in data.get("corpora") or []:
        out.setdefault(Path(prefix).name, f"{prefix}.train.jsonl")
    for name, spec in (data.get("event_files") or {}).items():
        if isinstance(spec, dict) and spec.get("train"):
            out.setdefault(name, spec["train"])
    return out


def scan(path: Path, fns: dict, limit: int, transform=None) -> dict:
    """Label sets per dimension, plus which dimensions this corpus annotates at all."""
    entities: set = set()
    events: dict = {}
    relations: set = set()
    structures: dict = {}
    seen = Counter()
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if limit and n >= limit:
                break
            if not line.strip():
                continue
            n += 1
            rec = json.loads(line)
            if fns:
                rec = transform(rec, fns)
            out = rec.get("output") or {}
            ents = out.get("entities")
            if isinstance(ents, dict) and ents:
                seen["entities"] += 1
                entities.update(ents.keys())
            evs = out.get("events")
            if isinstance(evs, list) and evs:
                seen["events"] += 1
                for ev in evs:
                    if isinstance(ev, dict) and ev.get("event_type"):
                        roles = events.setdefault(ev["event_type"], set())
                        for a in ev.get("arguments") or []:
                            if isinstance(a, dict) and a.get("role"):
                                roles.add(a["role"])
            rels = out.get("relations")
            if isinstance(rels, list) and rels:
                seen["relations"] += 1
                for r in rels:
                    if isinstance(r, dict):
                        relations.update(r.keys())
            structs = out.get("json_structures")
            if isinstance(structs, list) and structs:
                seen["structures"] += 1
                for inst in structs:
                    if isinstance(inst, dict):
                        for sname, body in inst.items():
                            fields = structures.setdefault(sname, set())
                            if isinstance(body, dict):
                                fields.update(body.keys())
    return {
        "records_scanned": n,
        # A dimension is ANNOTATED only if this corpus actually carries gold for it. This is
        # the within-dimension rule: a corpus that never annotates entities must never be
        # offered an entity negative, because absence there means "not labelled", not "not
        # present".
        "annotates": {d: bool(seen[d]) for d in DIMENSIONS},
        "records_with": {d: seen[d] for d in DIMENSIONS},
        "entities": sorted(entities),
        "events": {t: sorted(r) for t, r in sorted(events.items())},
        "relations": sorted(relations),
        "structures": {k: sorted(v) for k, v in sorted(structures.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="tools/train/config/labels/negative_pools.json")
    ap.add_argument("--limit", type=int, default=0, help="records per corpus (0 = all)")
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _category_fns, load_labels_cfg, transform_record = _train_helpers()
    fns = _category_fns(load_labels_cfg(cfg, config_path))
    print(f"[pools] label transforms active for: {sorted(fns) or 'NONE'}")

    pools = {}
    print(f"\n{'corpus':22s}{'records':>9}{'ent':>7}{'evt':>6}{'rel':>6}{'struct':>8}   annotates")
    for name, path in sorted(_corpus_train_paths(cfg).items()):
        p = Path(path)
        if not p.is_file():
            print(f"{name:22s}{'ABSENT':>9}")
            continue
        info = scan(p, fns, args.limit, transform_record)
        pools[name] = info
        ann = ",".join(d for d, v in info["annotates"].items() if v) or "-"
        print(f"{name:22s}{info['records_scanned']:>9,}{len(info['entities']):>7}"
              f"{len(info['events']):>6}{len(info['relations']):>6}"
              f"{len(info['structures']):>8}   {ann}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": str(config_path), "pools": pools},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[pools] wrote {out} ({out.stat().st_size:,} bytes, {len(pools)} corpora)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
