"""Add label_map + inference_defaults to a published checkpoint's config.json.

Checkpoints saved before 2026-10-01 carry neither, so consumers send unmapped labels and
decode with their own defaults. This derives both from the model's TRAINING config through
the same `checkpoint_fields` training uses, and rewrites only those two keys.

    uv run python tools/train/backfill_model_config.py --repo whr778/gliner2-eb17-best \\
        --config tools/train/config/base/eb17-best.yaml            # dry run: prints the change
    ... --upload                                                   # writes it to the Hub
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402


def fields_for(config_path: str):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    ev = T._parse_eval_settings(cfg, config_path, corpus_data=[])
    return T.checkpoint_fields(T.load_labels_cfg(cfg, config_path), ev)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True, help="the training config the model was built from")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    token = os.environ["HF_TOKEN"]
    current = json.load(open(hf_hub_download(args.repo, "config.json", token=token), encoding="utf-8"))
    label_map, defaults = fields_for(args.config)
    print(f"[backfill] {args.repo} <- {args.config}")
    print(f"  label_map: {current.get('label_map') is not None and 'present' or 'absent'} -> "
          f"{ {c: len(b.get('map') or {}) for c, b in (label_map or {}).items()} } map entries")
    print(f"  inference_defaults: {current.get('inference_defaults')} -> {defaults}")
    if not args.upload:
        print("[backfill] dry run; pass --upload to write")
        return 0
    updated = dict(current, label_map=label_map, inference_defaults=defaults)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
        HfApi(token=token).upload_file(path_or_fileobj=str(path), path_in_repo="config.json",
                                       repo_id=args.repo,
                                       commit_message="config.json: add label_map + inference_defaults")
    back = json.load(open(hf_hub_download(args.repo, "config.json", token=token, force_download=True),
                          encoding="utf-8"))
    ok = back.get("label_map") == label_map and back.get("inference_defaults") == defaults
    print(f"[backfill] verified on the Hub: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
