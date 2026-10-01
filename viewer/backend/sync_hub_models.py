"""Add an owner's published GLiNER2 checkpoints to the viewer's models.json.

The list is how models are chosen in the viewer, so a checkpoint missing from it is
invisible rather than broken (TODO 18: 50 behind on 2026-09-29). Only repos whose
config.json is a GLiNER2 extractor are added; existing entries and their hand-written
labels are kept; entries no longer on the Hub are REPORTED, never deleted.

    uv run python sync_hub_models.py            # dry run
    uv run python sync_hub_models.py --write
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

STORE = Path(__file__).resolve().parent / "models.json"


def extractor_arch(repo: str, token: str):
    """The checkpoint's architecture if it is a GLiNER2 extractor, else None."""
    try:
        cfg = json.load(open(hf_hub_download(repo, "config.json", token=token), encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a repo without a readable config is not a checkpoint
        return None
    return (cfg.get("architecture") or "span") if cfg.get("model_type") == "extractor" else None


def label_for(repo: str, arch: str) -> str:
    name = repo.split("/", 1)[1]
    return f"{arch}: {name.removeprefix('gliner2-')}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--owner", default="whr778")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    token = os.environ["HF_TOKEN"]

    saved = json.loads(STORE.read_text(encoding="utf-8")) if STORE.exists() else []
    listed = {e["path"] for e in saved}
    on_hub = sorted(m.id for m in HfApi(token=token).list_models(author=args.owner))
    added, skipped = [], []
    for repo in on_hub:
        if repo in listed:
            continue
        arch = extractor_arch(repo, token)
        (added.append({"path": repo, "label": label_for(repo, arch)}) if arch else skipped.append(repo))
    gone = sorted(p for p in listed if p.startswith(f"{args.owner}/") and p not in set(on_hub))

    print(f"[sync] {len(on_hub)} {args.owner}/ models on the Hub; {len(listed)} listed; "
          f"{len(added)} to add; {len(skipped)} not GLiNER2 extractors; {len(gone)} listed but gone")
    for e in added:
        print(f"  + {e['path']}  ({e['label']})")
    for r in skipped:
        print(f"  skip {r}")
    for p in gone:
        print(f"  GONE {p}  (kept; remove by hand if intended)")
    if args.write and added:
        merged = sorted(saved + added, key=lambda e: e["path"].lower())
        STORE.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[sync] wrote {STORE} ({len(merged)} entries)")


if __name__ == "__main__":
    main()
