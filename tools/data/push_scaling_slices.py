"""Host the joint_ie scaling slices alongside their parent corpora on the Hub.

The `data/scaling_joint/*.j{10,40,100}k.*.jsonl` slices are the actual training data
for the boundary scaling curve. They are deterministic (seed 42, nested) and therefore
regenerable by `tools/train/build_joint_scaling_mix.py`, but hosting them makes the
curve reproducible from the Hub alone and -- more usefully -- makes the trainer's
fallback work.

**Why they go INSIDE the per-corpus repos rather than a dedicated one.**
`_fetch_if_missing` in `tools/train/train.py` resolves a missing file by taking its
basename, stripping the last two dot-fields, mapping that through
`canonical_dataset_key`, and looking up `hf_jsonl` for the resulting corpus. So
`bio_ner_relations.j10k.train.jsonl` resolves to the `bio_ner_relations` repo and is
fetched by exact basename. Put the slices anywhere else and the fallback silently does
not work -- which is exactly how a launch failed: the slices were absent locally
(a cwd bug), the fallback could not supply them, and the run died at step 0.

Some parent corpora are not hosted at all yet because they were never contaminated and
so were skipped by `push_repaired_corpora.py`. Those get their base splits uploaded
too, with the registry's `license` string reproduced verbatim -- never upgraded to a
named SPDX id.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from model_card import canonical_dataset_key, load_registry  # noqa: E402

SLICE_DIR = Path("data/scaling_joint")


def group_slices() -> dict[str, list[Path]]:
    """Map each corpus key to the slice files that belong to it."""
    out: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(SLICE_DIR.glob("*.jsonl")):
        out[canonical_dataset_key(p.name.rsplit(".", 2)[0])].append(p)
    return out


def base_splits(corpus: str) -> list[Path]:
    return sorted(p for p in Path("data").glob(f"{corpus}.*.jsonl") if p.is_file())


def build_card(corpus: str, meta: dict, base: list[Path], slices: list[Path]) -> str:
    lic = str(meta.get("license", "unknown"))
    restricted = any(w in lic.lower() for w in
                     ("unknown", "none declared", "see source", "see card", "nc-", "ldc"))
    langs = meta.get("language") or ["en"]
    rows = "\n".join(f"| `{p.name}` | {sum(1 for _ in p.open(encoding='utf-8')):,} |"
                     for p in base)
    warn = ("\n## Not redistributable\n\nUpstream license is **`" + lic + "`**. Private "
            "research cache: do not make public, and do not describe a model trained on "
            "it as having used redistributable data. The `license: other` tag is a "
            "placeholder required by the card format, not a grant.\n") if restricted else ""
    return f"""---
license: other
language:
{chr(10).join(f'  - {l}' for l in langs)}
tags:
  - gliner2
---

# {corpus} (GLiNER2 format)

{meta.get('description', '')}

| split | records |
|---|--:|
{rows}

## Scaling slices

This repo also carries **{len(slices)} `j10k` / `j40k` / `j100k` slice files** — the
nested subsamples used by the joint_ie boundary scaling curve
(`tools/train/config/joint-boundary-mmbert-*.yaml`). They are deterministic (seed 42)
and nested: each corpus's 10K slice is a prefix of its 40K slice, which is a prefix of
its 100K slice. Regenerable with `tools/train/build_joint_scaling_mix.py`.

They live here rather than in a separate repo because the trainer's `_fetch_if_missing`
resolves a slice back to its parent corpus and fetches by exact basename; anywhere else
and the fallback does not work.

## Source

{meta.get('source_url', 'not recorded')}

Upstream license as recorded in `tools/train/dataset_registry.yaml`: **`{lic}`**
{warn}"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--owner", default="whr778")
    ap.add_argument("--with-base", action="store_true",
                    help="also upload base splits for corpora that are not hosted yet")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    registry = load_registry().get("datasets", {})
    api = HfApi()
    groups = group_slices()
    done, failed = [], []

    for corpus, slices in sorted(groups.items()):
        meta = registry.get(corpus) or {}
        repo = meta.get("hf_jsonl") or f"{args.owner}/{corpus}"
        hosted = bool(meta.get("hf_jsonl"))
        uploads = list(slices)
        if not hosted and args.with_base:
            uploads = base_splits(corpus) + uploads
        mb = sum(p.stat().st_size for p in uploads) / 1e6
        tag = "existing" if hosted else "NEW repo"
        print(f"{repo:34s} {tag:9s} {len(uploads):3d} files {mb:7.1f} MB", flush=True)
        if args.dry_run:
            done.append(corpus)
            continue
        try:
            api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
            for p in uploads:
                api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                                repo_id=repo, repo_type="dataset")
            if not hosted:
                card = Path(f"/tmp/{corpus}_card.md")
                card.write_text(build_card(corpus, meta, base_splits(corpus), slices),
                                encoding="utf-8")
                api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                                repo_id=repo, repo_type="dataset")
            done.append(corpus)
        except Exception as exc:
            print(f"      FAILED: {exc}", flush=True)
            failed.append((corpus, str(exc)[:110]))

    print(f"\npushed {len(done)}/{len(groups)}")
    for c, why in failed:
        print(f"  FAILED {c}: {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
