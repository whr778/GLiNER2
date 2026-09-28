"""Rebuild data/ from the private Hub mirrors.

`data/` is gitignored and holds 12 GB that took converters, LLM annotation and several
label-unification passes to produce. This restores it from the `hf_jsonl` repos recorded
in dataset_registry.yaml.

Coverage is NOT total, and the script says so rather than finishing quietly: a corpus with
no `hf_jsonl` cannot be restored and is listed under UNRECOVERABLE. Treat that list as a
backup gap to close, not as noise.

    uv run python tools/data/restore_from_hf.py --all --dry-run
    uv run python tools/data/restore_from_hf.py --config tools/train/config/base/joint-boundary-mmbert-137k.yaml
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from model_card import load_registry  # noqa: E402

DATA = Path("data")
SLICE = re.compile(r"\.[js]\d+k\.")


def repo_files(api, repo):
    """The .jsonl files a dataset repo holds, or None when the repo is absent.

    A missing mirror must not abort the restore: partial coverage plus an explicit list
    of what is missing is useful, a traceback on the first gap is not.
    """
    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        return [f for f in api.list_repo_files(repo, repo_type="dataset")
                if f.endswith(".jsonl")]
    except RepositoryNotFoundError:
        return None


def wanted_from_config(config_path):
    """The exact split files one training config reads."""
    import yaml
    data = (yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}).get("data") or {}
    files = []
    for base in data.get("corpora") or []:
        files += [f"{base}.{split}.jsonl" for split in ("train", "val", "test")]
    for entry in (data.get("event_files") or {}).values():
        if isinstance(entry, dict):
            files += [v for v in entry.values() if isinstance(v, str)]
    return sorted(set(files))


def plan(registry):
    """Return [(repo, path_in_repo, local_path)] for everything the registry can restore."""
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    jobs, unrecoverable = [], []
    for directory, repo in (registry.get("jsonl_dirs") or {}).items():
        names = repo_files(api, repo)
        if names is None:
            unrecoverable.append(f"{directory}/ (repo {repo} does not exist)")
            continue
        for name in names:
            jobs.append((repo, name, DATA / directory / name))
    # Scaling SLICES live in a subdirectory, but several parent corpus repos also carry
    # copies at their root. Those copies are stale -- whr778/docfee still serves a
    # docfee.j100k.test.jsonl with 1,983 Chinese labels, fixed long ago in
    # whr778/scaling-joint -- and restoring one would write it to data/ rather than the
    # subdirectory it belongs in, under a name nothing reads.
    #
    # Only SLICES are skipped. A base split shares its basename across repos while
    # mapping to a DIFFERENT local path (data/x.val.jsonl vs data/scaling_joint/x.val.jsonl),
    # so suppressing those would silently drop files a config needs.
    shadowed = []
    for key, entry in (registry.get("datasets") or {}).items():
        repo = (entry or {}).get("hf_jsonl")
        if not repo:
            unrecoverable.append(key)
            continue
        names = repo_files(api, repo)
        if names is None:
            unrecoverable.append(f"{key} (repo {repo} does not exist)")
            continue
        for name in names:
            if SLICE.search(name):
                shadowed.append(f"{repo}::{name}")
                continue
            jobs.append((repo, name, DATA / name))
    if shadowed:
        print(f"NOTE: {len(shadowed)} duplicate copies in parent repos ignored; the "
              f"directory mirror is authoritative. e.g. {shadowed[:2]}")
    return api, jobs, sorted(unrecoverable)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="every corpus the registry can restore")
    parser.add_argument("--config", help="restore only the splits this training config reads")
    parser.add_argument("--force", action="store_true", help="re-download files already present")
    parser.add_argument("--dry-run", action="store_true", help="report without downloading")
    args = parser.parse_args()
    if not (args.all or args.config):
        parser.error("pass --all or --config")

    registry = load_registry()
    api, jobs, unrecoverable = plan(registry)
    if args.config:
        keep = set(wanted_from_config(args.config))
        jobs = [j for j in jobs if str(j[2]) in keep]
        absent = keep - {str(j[2]) for j in jobs}
        # A FILE THE HUB DOES NOT HAVE IS ONLY A BACKUP GAP IF WE HOLD IT LOCALLY.
        # `wanted_from_config` enumerates train/val/test for every corpus, because a
        # corpus is restored in full whether or not this config reads every split --
        # `train_only` must NOT narrow the download, or a train-only corpus would lose
        # its val and test copies on a restore. The cost of enumerating is that splits
        # which never existed get asked for: duee_ner has no test split anywhere, because
        # DuEE's upstream mirror ships train + validation only (convert_duee.py documents
        # it and the model card records an em-dash).
        #
        # Reporting those as "NOT RESTORABLE ... these die with the disk" put a
        # non-problem in the one list whose whole job is to be believed. Split it: on
        # disk with no Hub copy is a REAL gap and stays loud; absent from both never
        # existed and is a note.
        gaps = sorted(a for a in absent if Path(a).exists())
        never = sorted(a for a in absent if not Path(a).exists())
        if gaps:
            print(f"*** NOT BACKED UP ({len(gaps)}) -- on disk, no Hub copy, these die "
                  f"with the disk:")
            for a in gaps:
                print(f"   {a}")
        if never:
            print(f"not on the Hub, and not on disk ({len(never)}) -- nothing to restore; "
                  f"expected where a corpus has no such split:")
            for a in never:
                print(f"   {a}")

    todo = [j for j in jobs if args.force or not j[2].exists()]
    print(f"{len(jobs)} files mirrored | {len(jobs) - len(todo)} already local | "
          f"{len(todo)} to download")
    if args.dry_run:
        for repo, name, local in todo[:40]:
            print(f"   {repo} :: {name} -> {local}")
        if len(todo) > 40:
            print(f"   ... and {len(todo) - 40} more")
    else:
        from huggingface_hub import hf_hub_download
        for i, (repo, name, local) in enumerate(todo, 1):
            local.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(repo_id=repo, filename=name, repo_type="dataset",
                            local_dir=str(local.parent), token=os.environ.get("HF_TOKEN"))
            print(f"  [{i}/{len(todo)}] {local}")

    if args.all and unrecoverable:
        print(f"\nUNRECOVERABLE -- no hf_jsonl, these die with the disk ({len(unrecoverable)}):")
        for key in unrecoverable:
            print(f"   {key}")


if __name__ == "__main__":
    main()
