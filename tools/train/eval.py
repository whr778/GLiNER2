"""Evaluate a saved GLiNER2 checkpoint against a config's val or test split,
without retraining. Reuses train.py's blind-test path; flags override the
config's ``eval:`` settings, which is handy for ablations (e.g. whole-doc vs.
windowed vs. global decode on a fixed checkpoint).

Examples:
  # blind test the config's checkpoint with its own eval settings
  uv run python tools/train/eval.py --config tools/train/config/gliner2-base-v1-wikievents.yaml

  # score the val split instead
  uv run python tools/train/eval.py --config <cfg> --split val

  # windowing/global-decode ablation on a fixed checkpoint
  uv run python tools/train/eval.py --config <cfg> --chunk-size 0                     # whole-doc
  uv run python tools/train/eval.py --config <cfg> --chunk-size 384 --no-global-decode  # chunk + simple merge
  uv run python tools/train/eval.py --config <cfg> --chunk-size 384 --global-decode     # chunk + beam
"""

import argparse

from train import evaluate_config


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a GLiNER2 checkpoint against a config split (no retraining).")
    p.add_argument("--config", required=True, help="Training config YAML.")
    p.add_argument("--split", choices=["val", "test"], default="test", help="Which split to score.")
    p.add_argument("--checkpoint", help="Checkpoint dir (default: <output_dir>/best).")
    p.add_argument("--threshold", type=float, help="Override eval.threshold.")
    p.add_argument("--chunk-size", type=int, dest="chunk_size",
                   help="Override eval.chunk_size (word window); 0 = whole-doc (no chunking).")
    p.add_argument("--chunk-overlap", type=int, dest="chunk_overlap", help="Override eval.chunk_overlap.")
    gd = p.add_mutually_exclusive_group()
    gd.add_argument("--global-decode", dest="global_decode", action="store_true", default=None,
                    help="Force OneIE-style global event decoding on.")
    gd.add_argument("--no-global-decode", dest="global_decode", action="store_false",
                    help="Force it off (chunk + simple merge).")
    args = p.parse_args()

    overrides = {}
    if args.threshold is not None:
        overrides["threshold"] = args.threshold
    if args.chunk_size is not None:
        overrides["chunk_size"] = None if args.chunk_size == 0 else args.chunk_size
    if args.chunk_overlap is not None:
        overrides["chunk_overlap"] = args.chunk_overlap
    if args.global_decode is not None:
        overrides["global_decode"] = args.global_decode

    evaluate_config(args.config, split=args.split, checkpoint=args.checkpoint, overrides=overrides)


if __name__ == "__main__":
    main()
