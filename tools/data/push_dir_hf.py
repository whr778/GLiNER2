"""Archive a directory of research artifacts to a private Hub dataset repo.

`push_corpus_hf.py` handles a corpus's train/val/test split files. This handles
everything that is not shaped like that: harvested article caches, precomputed
scores, raw pulls -- the things that are expensive or impossible to regenerate.

What belongs here is anything whose loss costs real time or money:

  * `datasets/{aegean2020,helene2024,turkey2023}` -- harvested publisher article
    text. News URLs rot, so a re-harvest does NOT return the same corpus, and the
    frozen tracked_*.json baselines could not be reproduced at all.
  * `data/guide_scores.*` -- 21.2 hours of precompute.
  * `data/cc_news_parts/*_raw.jsonl` -- the raw pulls behind a paid annotation run;
    re-pulling needs --exclude and the yield degrades as the corpus grows.

Repos are PRIVATE by default and these carry publisher copyright; keep them private.

    uv run python tools/data/push_dir_hf.py datasets/aegean2020 whr778/ekf-feed-caches \\
        --path-in-repo aegean2020
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def push(local: Path, repo_id: str, path_in_repo: str, private: bool,
         patterns: list[str] | None) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    # Count what will ACTUALLY be sent. Globbing everything and reporting that total
    # while upload_folder filters by allow_patterns prints a wildly wrong number --
    # it claimed 11 GB for a 249 MB upload.
    import fnmatch
    files = [f for f in local.rglob("*") if f.is_file()
             and (not patterns
                  or any(fnmatch.fnmatch(str(f.relative_to(local)), pat) for pat in patterns))]
    size = sum(f.stat().st_size for f in files) / 1e6
    print(f"{local} -> {repo_id}/{path_in_repo}  ({len(files)} files, {size:,.1f} MB)")
    api.upload_folder(folder_path=str(local), path_in_repo=path_in_repo,
                      repo_id=repo_id, repo_type="dataset",
                      allow_patterns=patterns,
                      ignore_patterns=["*.DS_Store", "__pycache__/*", "*.pyc"])
    print(f"done: https://huggingface.co/datasets/{repo_id}/tree/main/{path_in_repo}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("local", type=Path, help="directory to archive")
    ap.add_argument("repo_id", help="target dataset repo, e.g. whr778/ekf-feed-caches")
    ap.add_argument("--path-in-repo", default=None,
                    help="subdirectory in the repo (default: the local dir name)")
    ap.add_argument("--allow", nargs="+", default=None,
                    help="glob patterns to include (default: everything)")
    ap.add_argument("--public", action="store_true",
                    help="create a PUBLIC repo; these carry publisher copyright")
    args = ap.parse_args()

    if not args.local.is_dir():
        print(f"not a directory: {args.local}")
        return 2
    push(args.local, args.repo_id, args.path_in_repo or args.local.name,
         not args.public, args.allow)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
