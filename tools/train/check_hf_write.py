"""Prove an HF token can actually WRITE, before a run spends hours earning something to push.

`create_repo(..., exist_ok=True)` is NOT a write check. On a repo that already exists it
returns successfully for a READ-scoped token, so the obvious pre-flight reports success and
the run discovers the truth only at the push, hours later:

    403 Forbidden: you must use a write token to upload to a repository.

Measured 2026-09-05: a 6h47m casualty run trained cleanly, then failed to publish its model,
its metrics and its log for exactly this reason. The token was read-scoped and the pre-flight
had said "write OK".

So this uploads a real file to a real path and deletes it again. That is the only check that
distinguishes the two cases.

    uv run python tools/train/check_hf_write.py --repo whr778/gliner2-run-logs --repo-type dataset
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--repo-type", default="dataset", choices=("dataset", "model"))
    ap.add_argument("--keep", action="store_true",
                    help="leave the probe file behind instead of deleting it")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token)

    who = api.whoami()
    role = (who.get("auth") or {}).get("accessToken", {}).get("role")
    print(f"[hf-write] user={who.get('name')} token_role={role}")

    api.create_repo(args.repo, repo_type=args.repo_type, private=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"_write_probe/{stamp}.txt"
    api.upload_file(path_or_fileobj=f"write probe {stamp}\n".encode(),
                    path_in_repo=path, repo_id=args.repo, repo_type=args.repo_type)
    print(f"[hf-write] uploaded {path}")

    if not args.keep:
        api.delete_file(path_in_repo=path, repo_id=args.repo, repo_type=args.repo_type)
        print("[hf-write] probe deleted")
    print(f"[hf-write] WRITE CONFIRMED on {args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
