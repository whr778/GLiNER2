"""Refuse to launch a box whose config names a corpus the box cannot fetch.

WHY IT EXISTS. On 2026-09-18 the Option 4 A/B launched two A100s and BOTH arms died in the
data-loading phase, ~12 minutes in: the control on `data/scierc.train.jsonl`, the treatment on
`data/cmnee_roles_ner.train.jsonl`. Neither corpus had an `hf_jsonl` entry in
`dataset_registry.yaml`, and `_fetch_corpus` RETURNS SILENTLY when a corpus is unregistered --
so the file was never fetched and `open()` failed. Both files were present on this laptop,
which is exactly why the configs looked fine.

A fresh box has NO `data/`. The only thing that matters is whether every split the config
needs can be fetched from the Hub, and that is answerable locally, in seconds, for free.

    uv run python tools/train/check_corpora_fetchable.py --config <config.yaml>

Exit 0 when every referenced split is fetchable; non-zero, naming the corpus and the fix,
otherwise. `--offline` checks only that a repo is REGISTERED, skipping the Hub round trip.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import yaml  # noqa: E402
from model_card import canonical_dataset_key, load_registry  # noqa: E402

SPLITS = ("train", "val", "test")


def referenced(cfg: dict) -> list:
    """Corpus base names a config's `corpora` and `event_files` refer to."""
    data = cfg.get("data") or {}
    names = []
    for c in data.get("corpora") or []:
        names.append(c if isinstance(c, str) else (c.get("name") or ""))
    for v in (data.get("event_files") or {}).values():
        if isinstance(v, str):
            names.append(v)
        elif isinstance(v, dict):
            names.extend(x for x in v.values() if isinstance(x, str))
    out, seen = [], set()
    for n in names:
        base = os.path.basename(str(n)).split(".")[0]
        if base and base not in seen:
            seen.add(base)
            out.append(base)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--offline", action="store_true",
                    help="only check the corpus is REGISTERED, skip the Hub round trip")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    reg = load_registry().get("datasets", {})
    api = None
    if not args.offline:
        from huggingface_hub import HfApi
        tok = (Path.home() / ".hf_token")
        api = HfApi(token=os.environ.get("HF_TOKEN") or
                    (tok.read_text().strip() if tok.exists() else None))

    names = referenced(cfg)
    print(f"[check] {args.config}: {len(names)} corpora referenced")
    unregistered, absent = [], []
    for base in names:
        repo = (reg.get(canonical_dataset_key(base)) or {}).get("hf_jsonl")
        if not repo:
            # THE EXACT DEFECT THAT BURNED TWO BOXES: _fetch_corpus returns silently here,
            # so the failure surfaces later as a FileNotFoundError on a box that is billing.
            unregistered.append(base)
            print(f"  {base:24s} *** NO hf_jsonl ENTRY -- a box cannot fetch this ***")
            continue
        if api is None:
            print(f"  {base:24s} registered -> {repo}")
            continue
        found = [s for s in SPLITS
                 if _exists(api, repo, f"{base}.{s}.jsonl")]
        if not found:
            absent.append((base, repo))
            print(f"  {base:24s} registered -> {repo}  *** NO SPLIT FILES FOUND ***")
        else:
            print(f"  {base:24s} -> {repo}  [{', '.join(found)}]")

    if unregistered or absent:
        print("\n[check] REFUSING: a box would die in the data phase, minutes into a paid run.")
        for b in unregistered:
            print(f"  add to tools/train/dataset_registry.yaml under datasets:\n"
                  f"    {b}:\n      hf_jsonl: whr778/{b}")
        for b, repo in absent:
            print(f"  {b}: registered to {repo} but no {b}.<split>.jsonl there -- push it")
        return 1
    print("\n[check] every referenced corpus is fetchable by a fresh box")
    return 0


def _exists(api, repo: str, fname: str) -> bool:
    try:
        return bool(api.file_exists(repo, fname, repo_type="dataset"))
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
