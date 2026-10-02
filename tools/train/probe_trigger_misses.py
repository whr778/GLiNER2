"""Where do gold triggers go missing? Split every gold trigger into four outcomes, per corpus and language.

Two decodes of the same model on the same validation docs, gold event menu only:
  normal  the checkpoint's own inference_defaults (eb18: threshold 0.3)
  open    record (trigger) gate 0, span threshold 0, ARGUMENT gate 1.0 -- every proposed
          trigger survives and no argument is decoded, which keeps the open pass affordable

Each gold (type, trigger) is then exactly one of:
  found          decoded under its type by the normal pass
  wrong_type     normal pass decoded the surface, but only under another type
  below_gate     open pass decoded it under its type; the normal gate removed it
  never_decoded  not even the open pass decoded it -- the proposal top-k missed it, or the
                 decoded boundary differs from gold (strict surface match)

The lever follows the bucket: below_gate -> threshold / scorer; never_decoded -> proposal
reach (start_top_k) or boundaries; wrong_type -> type discrimination.
Language is a script heuristic (any CJK character -> zh), adequate for this en/zh mix.

    uv run python tools/train/probe_trigger_misses.py --checkpoint whr778/gliner2-eb18-balanced \
        --config tools/train/config/base/eb18-balanced.yaml --device cuda --per-corpus 200
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/train"))
import train as T  # noqa: E402
import yaml  # noqa: E402
from gliner2.training.eval_metrics import (  # noqa: E402
    _gold_event_trigger_set, _pred_event_trigger_set, _schema_from_gold, apply_boundary_overrides,
    load_with_overrides)

OPEN = {"record_anchor_threshold": 0.0, "record_anchor_proposal_threshold": 0.0,
        "record_anchor_threshold_wins": True,
        "record_field_threshold": 1.0, "record_field_threshold_wins": True}
CJK = re.compile(r"[㐀-鿿]")


def val_docs(cfg: dict, config: str, per_corpus: int, seed: int):
    """{corpus: [(text, events-only output)]} from each corpus's val split, events only."""
    fns = T._category_fns(T.load_labels_cfg(cfg, config))
    data = cfg["data"]
    paths = {Path(p).name: f"{p}.val.jsonl" for p in data.get("corpora") or []}
    paths.update({n: s["val"] for n, s in (data.get("event_files") or {}).items() if isinstance(s, dict) and s.get("val")})
    out = {}
    for name, path in sorted(paths.items()):
        if not Path(path).is_file():
            continue
        recs = [T.transform_record(json.loads(l), fns) for l in open(path, encoding="utf-8") if l.strip()]
        recs = [(r["input"], {"events": r["output"]["events"]}) for r in recs if r["output"].get("events")]
        if recs:
            out[name] = random.Random(seed).sample(recs, min(per_corpus, len(recs)))
    return out


def decode(model, docs, defaults, threshold):
    texts = [t for t, _ in docs]
    schemas = [_schema_from_gold(o) for _, o in docs]
    return model.batch_extract_long(texts, schemas, batch_size=2, threshold=threshold,
                                    chunk_size=defaults["chunk_size"], chunk_overlap=defaults["chunk_overlap"],
                                    global_decode=defaults["global_decode"])


def classify(gold, normal, opened):
    """Outcome per gold (type, trigger)."""
    normal_surfaces = {s for _, s in normal}
    for key in gold:
        if key in normal:
            yield key, "found"
        elif key[1] in normal_surfaces:
            yield key, "wrong_type"
        elif key in opened:
            yield key, "below_gate"
        else:
            yield key, "never_decoded"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--per-corpus", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    model = load_with_overrides(args.checkpoint, map_location=args.device).to(args.device).eval()
    defaults = model.config.inference_defaults
    base = {k: model.config.boundary_head.get(k) for k in OPEN}
    print(f"[probe] inference_defaults {defaults} | normal gates {base}")
    rows = Counter()
    for corpus, docs in val_docs(cfg, args.config, args.per_corpus, args.seed).items():
        apply_boundary_overrides(model, base)
        normal = decode(model, docs, defaults, defaults["threshold"])
        apply_boundary_overrides(model, OPEN)
        opened = decode(model, docs, defaults, 0.0)
        for (text, gold), pn, po in zip(docs, normal, opened):
            lang = "zh" if CJK.search(text) else "en"
            for _, outcome in classify(_gold_event_trigger_set(gold), _pred_event_trigger_set(pn),
                                       _pred_event_trigger_set(po)):
                rows[(corpus, lang, outcome)] += 1
        print(f"[probe] {corpus}: {len(docs)} docs done", flush=True)
    report(rows)
    if args.out:
        Path(args.out).write_text(json.dumps({"|".join(k): v for k, v in rows.items()}, ensure_ascii=False, indent=1),
                                  encoding="utf-8")


def report(rows: Counter) -> None:
    outcomes = ("found", "below_gate", "wrong_type", "never_decoded")
    groups = defaultdict(Counter)
    for (corpus, lang, outcome), n in rows.items():
        for g in (f"{corpus}/{lang}", f"ALL/{lang}", "ALL/all"):
            groups[g][outcome] += n
    print(f"[probe] {'group':34s} {'gold':>6s} " + " ".join(f"{o:>14s}" for o in outcomes))
    for g in sorted(groups):
        c = groups[g]
        n = sum(c.values())
        print(f"[probe] {g:34s} {n:6d} " + " ".join(f"{c[o]:5d} ({100 * c[o] / n:5.1f}%)" for o in outcomes))


if __name__ == "__main__":
    main()
