"""Per-corpus, per-threshold strict F1 of ONE checkpoint on a config's VAL files: is a bad head unlearned or mis-thresholded?

A headline number pools corpora (event_argument is ~88% CMNEE) and is read at one threshold. This scores each val
source separately -- the config's event_files val and every evaluated corpus's data/<c>.val.jsonl -- on its first
--records records, gold menu, global decode, at every threshold in --grid. A head that recovers at another threshold
is a calibration problem; a head at zero everywhere did not learn (or its data/decode path is broken).

    uv run tools/train/diagnose_checkpoint.py --config tools/train/config/base/eb22c.yaml \\
        --checkpoint out/eb22c/checkpoint-epoch-3 --device cuda --out eb22c-epoch3.diag.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
from gliner2 import AutoExtractor  # noqa: E402
from gliner2.training.eval_metrics import compute_metrics  # noqa: E402
from gliner2.training.trainer import ExtractorDataset  # noqa: E402

HEADS = ("entity", "relation", "event_type", "event_trigger", "event_argument")


def val_sources(cfg: dict) -> dict:
    """name -> val path: event_files val entries, then evaluated corpora with a local val file."""
    d = cfg["data"]
    out = {name: files["val"] for name, files in d["event_files"].items() if files.get("val")}
    for c in d["corpora"]:
        name = Path(c).name
        if name not in d["train_only"] and name not in out:
            out[name] = f"data/{name}.val.jsonl"
    return out


def score(model, records, threshold: float) -> dict:
    """Strict micro F1 and support per head (the in-run evaluator, gold menu, global decode)."""
    with torch.no_grad():
        m = compute_metrics(model, ExtractorDataset(records, shuffle=False, validate=False), batch_size=2,
                            threshold=threshold, report=False, chunk_size=4096, chunk_overlap=0, global_decode=True) or {}
    return {h: [round(float(m[f"eval_{h}_strict_micro_f1"]), 4), int(m.get(f"eval_{h}_strict_support", 0) or 0)]
            for h in HEADS if f"eval_{h}_strict_micro_f1" in m}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="local dir or Hub repo id")
    ap.add_argument("--records", type=int, default=100, help="first N val records per source")
    ap.add_argument("--grid", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
    ap.add_argument("--only", nargs="+", help="score only these sources")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    fns, passthrough = T._category_fns(T.load_labels_cfg(cfg, args.config)), T.labels_passthrough(cfg)
    model = AutoExtractor.from_pretrained(args.checkpoint, map_location="cpu").to(args.device).eval()
    print(f"[diag] {args.checkpoint} on {args.device} | labels {cfg.get('labels_file')} | grid {args.grid}", flush=True)
    result = {"args": vars(args), "sources": {}}
    for name, path in val_sources(cfg).items():
        if args.only and name not in args.only:
            continue
        T._fetch_if_missing(path)
        if not Path(path).is_file():
            print(f"[diag] {name}: no val file at {path}, skipped", flush=True)
            continue
        records = T.read_transformed([path], fns, passthrough)[:args.records]
        t0, rows = time.time(), {}
        for thr in args.grid:
            rows[str(thr)] = score(model, records, thr)
        result["sources"][name] = rows
        best = {h: max((rows[t][h][0], t) for t in rows if h in rows[t]) for h in HEADS if h in rows[str(args.grid[0])]}
        print(f"[diag] {name} ({len(records)} records, {time.time() - t0:.0f}s): "
              + " | ".join(f"{h} @0.3 {rows.get('0.3', {}).get(h, ['-'])[0]} best {b[0]}@{b[1]} (n={rows[b[1]][h][1]})"
                           for h, b in best.items()), flush=True)
        Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[diag] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
