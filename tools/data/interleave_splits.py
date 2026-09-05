"""Interleave a corpus whose splits were written in blocks, without moving any row.

A split written language-by-language (or source-by-source) is correct in aggregate and
wrong in every prefix. Training is unaffected -- LengthGroupedSampler randperms before it
sorts -- but a language-mix, length or label measurement read off the head of the file is
meaningless, and that is exactly how `casualty_ml` produced a 89.5%-English reading for a
32.9%-English corpus.

Shuffles WITHIN each split with a fixed seed. Split membership is never changed, so
split hygiene is preserved by construction and no row crosses a boundary.

    uv run python tools/data/interleave_splits.py data/casualty_ml
"""
from __future__ import annotations
import argparse, json, random, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402


def blockiness(langs: list[str]) -> float:
    """Mean |decile share - overall share|. 0 = interleaved, high = blocked."""
    n = len(langs)
    if n < 20:
        return 0.0
    overall, step, dev, k = Counter(langs), max(n // 10, 1), 0.0, 0
    for d in range(10):
        c = Counter(langs[d * step:(d + 1) * step])
        t = sum(c.values()) or 1
        for g in set(overall):
            dev += abs(c[g] / t - overall[g] / n)
            k += 1
    return dev / max(k, 1)


def lang(t: str) -> str:
    if any("一" <= c <= "鿿" for c in t[:600]):
        return "zh"
    if any(c in "ğüşıöçĞÜŞİÖÇ" for c in t[:1000]):
        return "tr"
    return "en"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix", help="e.g. data/casualty_ml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    a = ap.parse_args()

    for s in a.splits:
        path = Path(f"{a.prefix}.{s}.jsonl")
        if not path.exists():
            print(f"[interleave] {path} missing, skipped")
            continue
        rows = [json.loads(l) for l in path.open(encoding="utf-8")]
        before = blockiness([lang(r.get("input") or "") for r in rows])
        random.Random(a.seed).shuffle(rows)
        after = blockiness([lang(r.get("input") or "") for r in rows])
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(dumps_record(r) + "\n")
        print(f"[interleave] {path.name:28s} {len(rows):7,} rows  "
              f"blockiness {before:.3f} -> {after:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
