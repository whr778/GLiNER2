"""Publish a corpus's split files to a private Hub dataset repo, and verify they landed.

Written after a finished 500-document corpus was lost with the one disk it lived on, and
after `upload_folder` returned cleanly having written nothing (PROJECT_HISTORY Phase 31).
So this does the two things that episode showed were missing: it publishes per stage
rather than at the end, and it CHECKS the Hub's own file list afterwards instead of
trusting a clean return.

    uv run python tools/data/push_corpus.py --repo whr778/gliner2-generation-ab \\
        data/lg_onecall data/lg_twostage
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bases", nargs="+", help="corpus basenames, e.g. data/lg_onecall")
    ap.add_argument("--repo", required=True, help="Hub dataset repo id")
    ap.add_argument("--public", action="store_true", help="default is private")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=not args.public,
                    exist_ok=True)

    sent = []
    for base in args.bases:
        p = Path(base)
        # ONLY the three split files. A `base.*` glob also matches `.prerepair`, `.bak`
        # and `.batch_id` siblings -- for cc_news_haiku45 that is 190MB of PRE-REPAIR
        # data uploaded beside the repaired corpus, where nothing at training time
        # distinguishes them. The same shape (a rehearsal's mock output published to the
        # corpus repo) was cleaned out of gliner2-generation-ab on 2026-09-08; this is
        # that defect caught before it fired, at twenty times the size.
        files = [f for f in (p.with_name(f"{p.name}.{s}.jsonl")
                             for s in ("train", "val", "test")) if f.is_file()]
        if not files:
            raise SystemExit(f"nothing to push for {base}")
        for f in files:
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                            repo_id=args.repo, repo_type="dataset")
            print(f"[push] {f.name} {f.stat().st_size:,}B")
            sent.append(f.name)

    on_hub = set(api.list_repo_files(args.repo, repo_type="dataset"))
    missing = [f for f in sent if f not in on_hub]
    if missing:
        raise SystemExit(f"*** NOT SAVED *** uploaded without error but absent from "
                         f"{args.repo}: {missing}")
    print(f"[push] verified {len(sent)} file(s) present in {args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
