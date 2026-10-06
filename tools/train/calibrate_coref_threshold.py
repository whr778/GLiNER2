"""Calibrate a TRAINED checkpoint's coreference merge threshold on val; train.py does this itself
when the config sets ``eval.coref_calibration``.

    uv run tools/train/calibrate_coref_threshold.py --config tools/train/config/base/eb20.yaml \\
        --checkpoint out/eb20/best                     # dry run: prints the sweep and the choice
    ... --write                                        # writes config.json + coref_threshold_sweep.json

The calibration corpora come from ``eval.coref_calibration`` in the config, or ``--corpora``.
Decode uses the checkpoint's own ``inference_defaults``. See gliner2/training/coref_calibration.py
for the grid and the gates; the blind test is NOT touched.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train as T  # noqa: E402
from infer import resolve_model_window  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="the training config the checkpoint was built from")
    ap.add_argument("--checkpoint", help="default: <output_dir>/best")
    ap.add_argument("--corpora", nargs="+", help="override eval.coref_calibration, e.g. data/cc_news_events_sonnet55_v2")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--write", action="store_true", help="write the choice into the checkpoint")
    args = ap.parse_args(argv)

    from gliner2 import AutoExtractor
    from gliner2.training import coref_calibration as CC

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.corpora:
        cfg.setdefault("eval", {})["coref_calibration"] = args.corpora
    best = Path(args.checkpoint or Path(cfg["training"]["output_dir"]) / "best")
    files, records = T.coref_calibration_data(cfg, args.config)
    if not records:
        raise SystemExit("[coref sweep] no calibration records: set eval.coref_calibration or --corpora")
    model = AutoExtractor.from_pretrained(str(best))
    d = model.config.inference_defaults or {}
    decode = {"threshold": d.get("threshold", 0.5), "chunk_size": resolve_model_window(model, d.get("chunk_size")),
              "chunk_overlap": d.get("chunk_overlap", 0), "global_decode": bool(d.get("global_decode", False))}
    print(f"[coref sweep] {best} over {len(records)} records from {files}; decode {decode}")
    rows = CC.sweep(model, records, decode, batch_size=args.batch_size)
    threshold, why = CC.choose(rows)
    print(f"[coref sweep] choice: {threshold} -- {why}")
    if args.write:
        CC.write(best, rows, threshold, why, source=files)
    else:
        print("[coref sweep] dry run: nothing written (pass --write)")


if __name__ == "__main__":
    main()
