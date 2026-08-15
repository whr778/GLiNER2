"""Does any corpus share input text with any other, or across its own splits?

Two questions, both of which have changed a conclusion in this project before:

  WITHIN  a corpus, does train overlap val/test? Then its own eval is inflated.
  ACROSS  corpora, does one corpus's train overlap another's test? Then any blind
          test that mixes them is contaminated -- and the derived corpora are the
          prime suspects, since `casualty_docee` is built FROM DocEE while
          `docee.test.jsonl` sits in the blind test.

Texts are compared by sha1 of an NFKC-normalized, case-folded, whitespace-collapsed
form, so formatting differences do not hide a duplicate. Exact matching only: this
answers "is this the same document", not "is this a paraphrase".

    uv run python tools/data/check_leakage.py --pattern 'data/*.jsonl'
    uv run python tools/data/check_leakage.py --focus casualty
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

WHITESPACE = re.compile(r"\s+")


def text_key(text: str) -> str:
    """Stable identity for a document, insensitive to case and whitespace."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return hashlib.sha1(WHITESPACE.sub(" ", normalized).strip().encode("utf-8")).hexdigest()


def corpus_of(path: str) -> str:
    """`data/docee.train.jsonl` -> `docee`; `data/foo.jsonl` -> `foo`."""
    stem = Path(path).name[: -len(".jsonl")]
    for split in (".train", ".val", ".test"):
        if stem.endswith(split):
            return stem[: -len(split)]
    return stem


def split_of(path: str) -> str:
    stem = Path(path).name[: -len(".jsonl")]
    for split in ("train", "val", "test"):
        if stem.endswith("." + split):
            return split
    return "all"


def load(paths):
    """{path: set(text_key)} plus a count of unparsable/empty lines."""
    keys, counts = {}, {}
    for path in paths:
        seen = set()
        total = 0
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get("input")
                if not text:
                    continue
                total += 1
                seen.add(text_key(text))
        keys[path] = seen
        counts[path] = total
    return keys, counts


def aggregate_gate(config_path: str, min_overlap: int) -> int:
    """The gate: are a CONFIG's aggregated train/val/test mutually disjoint?

    Per-corpus checks are not sufficient. A training mix pools many corpora, and
    corpus A's train can contain a document sitting in corpus B's test -- neither
    file overlaps itself, yet the blind test is contaminated. This resolves the
    exact file lists the trainer would load and compares the pooled sets.

    Returns the number of contaminated pairs, so callers can treat it as a gate.
    """
    import yaml
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
    from train import _event_split, _split_files  # noqa: E402

    cfg = yaml.safe_load(Path(config_path).read_text())
    data = cfg.get("data") or {}
    corpora = data.get("corpora") or []
    events = data.get("event_files") or {}

    pooled, files_for = {}, {}
    for split in ("train", "val", "test"):
        files = [
            f for f in _split_files(corpora, split) + _event_split(events, split)
            if Path(f).exists() and Path(f).stat().st_size > 0
        ]
        keys, _ = load(files)
        pooled[split] = set().union(*keys.values()) if keys else set()
        files_for[split] = files

    print(f"=== AGGREGATE GATE for {config_path}")
    for split in ("train", "val", "test"):
        print(f"  {split:<6} {len(files_for[split]):>3} files, "
              f"{len(pooled[split]):>7} unique documents")

    per_file = {s: load(files_for[s])[0] for s in ("train", "val", "test")}
    bad = 0
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = pooled[a] & pooled[b]
        if len(shared) >= min_overlap:
            bad += 1
            smaller = min(len(pooled[a]), len(pooled[b])) or 1
            print(f"  CONTAMINATED  {a} n {b} = {len(shared)} documents "
                  f"({len(shared) / smaller:.2%} of the smaller)")
            # Attribute it: which file pair actually shares documents. Without
            # this the verdict is unactionable -- you cannot regenerate "the mix".
            culprits = []
            for fa, ka in per_file[a].items():
                for fb, kb in per_file[b].items():
                    n = len(ka & kb & shared)
                    if n >= min_overlap:
                        culprits.append((n, fa, fb))
            # Show the parent directory, not just the basename: configs pair a
            # regenerated data/<name>.train.jsonl with a FROZEN
            # data/scaling_joint/<name>.val.jsonl, and printing bare basenames made
            # that read as a corpus leaking into itself.
            def label(p: str) -> str:
                path = Path(p)
                return f"{path.parent.name}/{path.name}" if path.parent.name else path.name

            for n, fa, fb in sorted(culprits, reverse=True)[:8]:
                print(f"        {n:>6}  {label(fa)}  ->  {label(fb)}")
        else:
            print(f"  clean         {a} n {b} = 0")
    print("  VERDICT:", "CLEAN -- splits are mutually disjoint" if not bad
          else f"CONTAMINATED on {bad} pair(s); results from this config are invalid")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", default="data/*.jsonl")
    ap.add_argument("--focus", default=None,
                    help="only report pairs involving corpora whose name contains this")
    ap.add_argument("--config", default=None,
                    help="training config YAML: gate its AGGREGATED train/val/test "
                         "against each other and exit non-zero if contaminated")
    ap.add_argument("--min-overlap", type=int, default=1)
    args = ap.parse_args()

    if args.config:
        raise SystemExit(1 if aggregate_gate(args.config, args.min_overlap) else 0)

    paths = [p for p in sorted(glob.glob(args.pattern)) if Path(p).stat().st_size > 0]
    print(f"scanning {len(paths)} non-empty files matching {args.pattern!r}\n")
    keys, counts = load(paths)

    print("=== WITHIN corpus: train vs val/test")
    by_corpus = defaultdict(dict)
    for path in paths:
        by_corpus[corpus_of(path)][split_of(path)] = path
    dirty_within = 0
    for corpus, splits in sorted(by_corpus.items()):
        train = splits.get("train")
        if not train:
            continue
        for other in ("val", "test"):
            if other not in splits:
                continue
            shared = keys[train] & keys[splits[other]]
            if len(shared) >= args.min_overlap:
                dirty_within += 1
                n_other = len(keys[splits[other]])
                print(f"  LEAK {corpus}: train n {other} = {len(shared)} "
                      f"({len(shared) / max(n_other, 1):.1%} of {other})")
    if not dirty_within:
        print("  none")

    print("\n=== ACROSS corpora: any shared document")
    pairs = []
    names = sorted(by_corpus)
    merged = {c: set().union(*(keys[p] for p in s.values())) for c, s in by_corpus.items()}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if args.focus and args.focus not in a and args.focus not in b:
                continue
            shared = merged[a] & merged[b]
            if len(shared) >= args.min_overlap:
                pairs.append((len(shared), a, b, len(merged[a]), len(merged[b])))
    pairs.sort(reverse=True)
    if not pairs:
        print("  none")
    for n, a, b, na, nb in pairs:
        print(f"  {n:>7} shared   {a} ({na}) n {b} ({nb})   "
              f"= {n / min(na, nb):.1%} of the smaller")

    print("\n=== duplicate documents INSIDE a single file")
    any_dup = False
    for path in paths:
        unique, total = len(keys[path]), counts[path]
        if total - unique >= args.min_overlap:
            any_dup = True
            print(f"  {path}: {total - unique} duplicate(s) of {total} "
                  f"({(total - unique) / max(total, 1):.1%})")
    if not any_dup:
        print("  none")


if __name__ == "__main__":
    main()
