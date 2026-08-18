"""Push split-repaired corpora to private Hub dataset repos, one per corpus.

Every corpus here failed the within-split gate before repair: its train/val/test
overlapped each other, which makes a blind test partly a re-read of the selection set.
`dedupe_splits.py` fixed them in place with precedence test > val > train, so the blind
test keeps its documents and train gives them up.

Each generated card carries **that corpus's own pre-repair overlap counts** and its
before/after record counts. The point is that the contamination history travels with
the data: a future consumer sees what was wrong and what was done, instead of finding a
suspiciously clean corpus with no explanation.

Repos are private. Several of these carry text whose license does not permit
redistribution, and the registry's `license` field is reproduced verbatim rather than
guessed at.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import derive_split_paths  # noqa: E402


def load_leaks(path: Path) -> Dict[str, List[str]]:
    """Group `LEAK <corpus>: <pair> = n (pct)` lines by corpus."""
    out: Dict[str, List[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("LEAK "):
            continue
        corpus, _, detail = line[5:].partition(":")
        out.setdefault(corpus.strip(), []).append(detail.strip())
    return out


def load_registry(path: Path) -> dict:
    import yaml
    reg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return reg.get("datasets") or reg


def build_card(corpus: str, leaks: List[str], counts: dict, meta: dict) -> str:
    lic = str(meta.get("license", "unknown"))
    src = meta.get("source_url") or "not recorded"
    langs = meta.get("language") or ["en"]
    desc = meta.get("description") or ""
    restricted = any(w in lic.lower() for w in ("unknown", "none declared", "see source", "nc-"))

    leak_rows = "\n".join(f"| {d.split('=')[0].strip()} | {d.split('=')[1].strip()} |"
                          for d in leaks) or "| — | — |"
    count_rows = "\n".join(
        f"| {s} | {c['before']:,} | {c['after']:,} | {c['before'] - c['after']:,} |"
        for s, c in counts.items())
    total_before = sum(c["before"] for c in counts.values())
    total_after = sum(c["after"] for c in counts.values())

    warn = ""
    if restricted:
        warn = (
            "\n## Not redistributable\n\n"
            f"Upstream license is **`{lic}`**. Treat this repo as a **private research "
            "cache**: do not make it public, and do not describe a model trained on it "
            "as having used redistributable data. The `license: other` tag above is a "
            "placeholder required by the card format, not a grant.\n")

    return f"""---
license: other
language:
{chr(10).join(f'  - {l}' for l in langs)}
tags:
  - gliner2
size_categories:
  - {'100K<n<1M' if total_after > 100_000 else ('10K<n<100K' if total_after > 10_000 else '1K<n<10K')}
---

# {corpus} (GLiNER2 format, splits repaired)

{desc}

## Splits were overlapping and have been repaired

This corpus **failed the within-split contamination gate**. Measured on the document key
(NFKC + casefold + whitespace collapse), before repair:

| overlapping pair | shared documents |
|---|--:|
{leak_rows}

That matters because validation selects the checkpoint that test then scores, so a
val/test overlap makes the blind test partly a re-read of the selection set.

Repaired with `tools/data/dedupe_splits.py`, **precedence test > val > train** — the
blind test keeps every document it has and the training side gives up the duplicate.

| split | before | after | dropped |
|---|--:|--:|--:|
{count_rows}
| **total** | **{total_before:,}** | **{total_after:,}** | **{total_before - total_after:,}** |

Verified after repair: the splits are mutually disjoint.

## Source

{src}

Upstream license as recorded in `tools/train/dataset_registry.yaml`: **`{lic}`**
{warn}
## Provenance

Converted by the GLiNER2 converters in `tools/data/`, repaired by
`tools/data/dedupe_splits.py`. The repair is wired into
`tools/data/run_all_converters.sh` so a fresh build cannot silently regenerate the
contaminated version.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpora", type=Path, required=True, help="file, one corpus per line")
    ap.add_argument("--leaks", type=Path, required=True, help="pre-repair LEAK lines")
    ap.add_argument("--summary", type=Path, required=True, help="repair_summary.json")
    ap.add_argument("--registry", type=Path,
                    default=Path("tools/train/dataset_registry.yaml"))
    ap.add_argument("--owner", default="whr778")
    ap.add_argument("--cards-dir", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    names = args.corpora.read_text(encoding="utf-8").split()
    leaks = load_leaks(args.leaks)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    registry = load_registry(args.registry)
    args.cards_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    done, failed = [], []
    for i, corpus in enumerate(names, 1):
        paths = {s: p for s, p in derive_split_paths(Path(f"data/{corpus}")).items()
                 if p.exists()}
        # Pick up `.dev.jsonl` whenever `val` is absent, NOT only when every split is
        # missing. wikievents has train+test+dev, so an all-missing guard never fired
        # and its validation split was silently left off the upload.
        if "val" not in paths:
            dev = Path(f"data/{corpus}.dev.jsonl")
            if dev.exists():
                paths["dev"] = dev
        if not paths:
            failed.append((corpus, "no split files")); continue

        counts = {}
        for s, p in paths.items():
            after = sum(1 for _ in p.open(encoding="utf-8"))
            bk = Path("data/_backup_pre_dedupe_20260818") / p.name
            before = sum(1 for _ in bk.open(encoding="utf-8")) if bk.exists() else after
            counts[s] = {"before": before, "after": after}

        card = build_card(corpus, leaks.get(corpus, []), counts,
                          registry.get(corpus, {}))
        card_path = args.cards_dir / f"{corpus}.md"
        card_path.write_text(card, encoding="utf-8")

        repo_id = f"{args.owner}/{corpus}"
        mb = sum(p.stat().st_size for p in paths.values()) / 1e6
        print(f"[{i}/{len(names)}] {repo_id} ({mb:,.0f} MB, {len(paths)} files)", flush=True)
        if args.dry_run:
            done.append(corpus); continue
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=True,
                            exist_ok=True)
            for p in paths.values():
                api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                                repo_id=repo_id, repo_type="dataset")
            api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md",
                            repo_id=repo_id, repo_type="dataset")
            done.append(corpus)
        except Exception as exc:                      # keep going; report at the end
            print(f"      FAILED: {exc}", flush=True)
            failed.append((corpus, str(exc)[:120]))

    print(f"\npushed {len(done)}/{len(names)}")
    for c, why in failed:
        print(f"  FAILED {c}: {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
