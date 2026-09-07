"""Get a trained checkpoint OFF the machine, or refuse to let the machine die quietly.

Every runner in this project ends with `trap shutdown EXIT INT TERM`, so the box
terminates on every exit path. That is right -- it is why nothing bills idle. But it
means a failed push is unrecoverable the instant the trap fires, and twice on
2026-09-07 it was:

  * the realsynth re-run  -- 3h34m, lost to one transient 400 with no retry;
  * Phase 0 (eb16-composed) -- ~15h of A100, lost to an upload that returned WITHOUT
    raising and wrote nothing. The repo holds one file: .gitattributes.

`push_to_hub.py` now verifies the files are really on the Hub rather than trusting a
clean return. This is the second line: if the model repo still cannot be written, put
the weights somewhere that HAS been working, and say so loudly.

    uv run python tools/train/save_or_die.py --checkpoint out/run/best --repo-id ns/name

Exit 0 only if the weights are verifiably off the machine.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

FALLBACK_REPO = "whr778/gliner2-run-logs"   # a dataset repo, proven writable all day


def _required(checkpoint: Path) -> list:
    return sorted(f.name for f in checkpoint.iterdir()
                  if f.is_file() and (f.suffix in (".safetensors", ".bin")
                                      or f.name == "config.json"))


def verify(api, repo_id: str, required: list, repo_type: str = "model") -> list:
    """Return the required files the Hub does NOT have."""
    try:
        there = set(api.list_repo_files(repo_id=repo_id, repo_type=repo_type))
    except Exception as e:  # noqa: BLE001
        print(f"[save] cannot list {repo_id}: {type(e).__name__}: {e}")
        return list(required)
    return [f for f in required if f not in there]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--fallback-repo", default=FALLBACK_REPO)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    ckpt = Path(args.checkpoint)
    if not ckpt.is_dir():
        print(f"[save] no checkpoint at {ckpt} -- nothing to save")
        return 0

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    required = _required(ckpt)
    if not required:
        print(f"[save] {ckpt} holds no weights or config; nothing to save")
        return 0

    missing = verify(api, args.repo_id, required)
    if not missing:
        print(f"[save] already on the Hub at {args.repo_id}: {', '.join(required)}")
        return 0

    print(f"[save] {args.repo_id} is MISSING {', '.join(missing)} -- trying the fallback")
    prefix = f"rescued/{ckpt.parent.name or 'run'}"
    try:
        api.upload_folder(folder_path=str(ckpt), repo_id=args.fallback_repo,
                          repo_type="dataset", path_in_repo=prefix,
                          commit_message=f"rescue weights for {args.repo_id}")
    except Exception as e:  # noqa: BLE001
        print(f"[save] FALLBACK FAILED too: {type(e).__name__}: {e}")
        print(f"[save] *** THE WEIGHTS ARE ONLY ON THIS MACHINE: {ckpt} ***")
        return 2

    still = verify(api, args.fallback_repo, [f"{prefix}/{f}" for f in required],
                   repo_type="dataset")
    if still:
        print(f"[save] fallback upload reported success but {still} are absent")
        print(f"[save] *** THE WEIGHTS ARE ONLY ON THIS MACHINE: {ckpt} ***")
        return 2
    print(f"[save] RESCUED to {args.fallback_repo}/{prefix} -- {', '.join(required)}")
    print(f"[save] the model repo {args.repo_id} is still empty; move them when convenient")
    return 0


if __name__ == "__main__":
    sys.exit(main())
