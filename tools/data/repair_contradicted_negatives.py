"""Remove seeded entity negatives that the corpus's own usage contradicts.

`mint_entity_negatives` seeds absent types with `[]`, asserting "the annotator saw this
type and declined to use it, so it is absent". That inference is wrong whenever the
annotator was merely unsure, and until `uncertain_types` existed there was no way to tell
the two apart -- roughly 1 in 9 uncertain omissions became an explicit false negative
(12 seeded from ~113 absent, against a 125-type ontology).

The annotator's doubt was never recorded and CANNOT be recovered from the output: the
record shows what was chosen, never what was weighed. So this does not restore the missing
signal. What it does is audit the assertion, which IS checkable:

    a record asserts "type T is absent" while containing a surface that the corpus
    labels T elsewhere.

That record is claiming absence for something it demonstrably contains, so the negative is
removed. Positives are never touched, no label is added, and nothing is re-annotated.

    audit : uv run python tools/data/repair_contradicted_negatives.py --corpus cc_news_haiku45
    repair: uv run python tools/data/repair_contradicted_negatives.py --corpus cc_news_haiku45 --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402

SPLITS = ("train", "val", "test")


def load(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def surface_types(path: Path, min_uses: int, min_share: float) -> dict:
    """surface -> {type: count}, built from the TRAIN split only.

    Train-only so that a repair applied to val/test is never informed by val/test
    content. The map describes label USAGE, not documents, but keeping the provenance
    one-directional costs nothing and removes the question.
    """
    counts: dict = defaultdict(Counter)
    for rec in load(path):
        ents = (rec.get("output") or {}).get("entities")
        if not isinstance(ents, dict):
            continue
        for t, vs in ents.items():
            for v in vs or []:
                if isinstance(v, str):
                    counts[v][t] += 1
    out = {}
    for surf, c in counts.items():
        tot = sum(c.values())
        # A type must be a SUBSTANTIAL reading of this surface, not merely an attested
        # one. Raw counts alone are far too weak: the corpus calls "26" an ordinal twice
        # out of 164 uses, which is a rare minority reading, not evidence that a record
        # containing "26" must hold an ordinal. Auditing on counts alone flagged 3.8% of
        # all seeded negatives, and the sampled cases were overwhelmingly legitimate.
        keep = {t: n for t, n in c.items() if n >= min_uses and n / tot >= min_share}
        if keep:
            out[surf] = keep
    return out


def repair_split(path: Path, smap: dict, apply: bool) -> Counter:
    stats = Counter()
    out_lines = []
    for rec in load(path):
        ents = (rec.get("output") or {}).get("entities")
        stats["records"] += 1
        if isinstance(ents, dict):
            present = {v for vs in ents.values() for v in (vs or []) if isinstance(v, str)}
            negatives = [t for t, vs in ents.items() if not vs]
            stats["negatives"] += len(negatives)
            drop = [t for t in negatives
                    if any(t in smap.get(v, ()) for v in present)]
            if drop:
                stats["contradicted"] += len(drop)
                stats["records_touched"] += 1
                for t in drop:
                    del ents[t]
        out_lines.append(dumps_record(rec))
    if apply and stats["contradicted"]:
        backup = path.with_suffix(path.suffix + ".prerepair")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="basename, e.g. cc_news_haiku45")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--min-uses", type=int, default=2,
                    help="times the corpus must label a surface T before that counts as "
                         "evidence against a negative (default 2; 1 would let a single "
                         "annotation error delete real negatives)")
    ap.add_argument("--min-share", type=float, default=0.30,
                    help="fraction of a surface's uses that must carry type T before a "
                         "record containing it is said to contradict 'no T here' "
                         "(default 0.30; counts alone are far too weak -- see the note "
                         "in surface_types)")
    ap.add_argument("--apply", action="store_true",
                    help="write the repair; without it this only reports")
    args = ap.parse_args()

    d = Path(args.data_dir)
    train = d / f"{args.corpus}.train.jsonl"
    if not train.is_file():
        raise SystemExit(f"no train split at {train}")

    smap = surface_types(train, args.min_uses, args.min_share)
    print(f"[map] {len(smap):,} surfaces from {train.name} "
          f"(type needs >= {args.min_uses} uses AND >= {args.min_share:.0%} of that "
          f"surface's uses)")

    total = Counter()
    for split in SPLITS:
        p = d / f"{args.corpus}.{split}.jsonl"
        if not p.is_file():
            continue
        s = repair_split(p, smap, args.apply)
        total.update(s)
        pct = 100 * s["contradicted"] / max(s["negatives"], 1)
        print(f"  {split:5s} {s['records']:7,d} records  {s['negatives']:8,d} negatives  "
              f"{s['contradicted']:6,d} contradicted ({pct:.1f}%)  "
              f"in {s['records_touched']:,} records")

    pct = 100 * total["contradicted"] / max(total["negatives"], 1)
    print(f"\n{'APPLIED' if args.apply else 'AUDIT ONLY'}: "
          f"{total['contradicted']:,} of {total['negatives']:,} seeded negatives "
          f"({pct:.1f}%) assert absence for a type the record demonstrably contains.")
    if not args.apply:
        print("re-run with --apply to remove them (originals kept as *.prerepair)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
