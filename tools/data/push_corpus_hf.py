"""Push a GLiNER2 corpus's train/val/test JSONL to a private Hub dataset repo.

`tools/train/push_to_hub.py` is for MODELS; this is the corpus equivalent, and
until now each corpus was uploaded by hand.

**Files keep their exact local basename at the repo root** -- `<corpus>.train.jsonl`
and friends. That is not cosmetic: `_fetch_if_missing` in `tools/train/train.py`
downloads by `Path(path).name`, so a renamed or subdirectory-nested file is
invisible to the trainer. Set the corpus's `hf_jsonl` in `dataset_registry.yaml`
to the repo id afterwards, which is what makes the fetch happen at all.

Repos are created **private** by default. Several of these corpora carry text whose
license does not permit redistribution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import derive_split_paths  # noqa: E402


def push(base: Path, repo_id: str, card: Path | None, private: bool) -> None:
    """Create the repo if needed and upload each split under its local name."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset",
                    private=private, exist_ok=True)
    print(f"repo {repo_id} (private={private})")

    uploaded = 0
    for split, path in derive_split_paths(base).items():
        if not path.exists():
            print(f"  skip {split}: {path} absent")
            continue
        uploaded += 1
        size = path.stat().st_size / 1e6
        print(f"  uploading {path.name} ({size:,.1f} MB)...", flush=True)
        api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name,
                        repo_id=repo_id, repo_type="dataset")

    if card:
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="dataset")
        print(f"  uploaded README.md from {card}")
    if not uploaded:
        # Printing "done" after uploading nothing is how a backup silently fails to
        # exist. `data/x.jsonl` has no split suffix, so derive_split_paths finds
        # x.train/x.val/x.test and skips all three -- and the repo stays empty.
        raise SystemExit(
            f"push_corpus_hf: NOTHING was uploaded -- no {base}.train/.val/.test.jsonl "
            f"exists. For a single unsplit file use --dir on its directory, or split it "
            f"first.")
    print(f"done: {uploaded} file(s) -> https://huggingface.co/datasets/{repo_id}")


def push_dir(directory: Path, repo_id: str, private: bool) -> None:
    """Upload every .jsonl in a directory, basenames at the repo root.

    A data SUBDIRECTORY needs its own repo, because `_fetch_if_missing` resolves a repo
    from the file basename and two directories can share one. Registered under
    `jsonl_dirs` in dataset_registry.yaml, which wins over the per-corpus `hf_jsonl`.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    files = sorted(directory.glob("*.jsonl"))
    total = sum(f.stat().st_size for f in files) / 1e6
    print(f"repo {repo_id} (private={private}): {len(files)} files, {total:,.1f} MB")
    api.upload_folder(folder_path=str(directory), repo_id=repo_id, repo_type="dataset",
                      allow_patterns=["*.jsonl"])
    print(f"done: https://huggingface.co/datasets/{repo_id}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path,
                    help="split base, e.g. data/cc_news_haiku45")
    ap.add_argument("repo_id", help="target repo, e.g. whr778/cc_news_haiku45")
    ap.add_argument("--card", type=Path, help="README.md to upload as the dataset card")
    ap.add_argument("--public", action="store_true",
                    help="create a PUBLIC repo; check the corpus license first")
    ap.add_argument("--dir", action="store_true",
                    help="treat `base` as a DIRECTORY and upload every .jsonl in it")
    args = ap.parse_args()

    if args.dir:
        push_dir(args.base, args.repo_id, private=not args.public)
    else:
        push(args.base, args.repo_id, args.card, private=not args.public)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
