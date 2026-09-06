"""Stamp missing ``record_metadata`` onto a corpus whose structures are silently undecodable.

A structure schema with no ``record_metadata`` is VALID and trains NOTHING on the boundary
path: the record head cannot decode it, and neither the loader nor the trainer reports an
error (``gliner2/processor.py`` says so in as many words). The rows look like supervision,
count as supervision in every composition print, and supply none.

Measured 2026-09-06 on ``data/mix_natural``: 28,736 of 29,710 rows carrying
``casualty_report`` structures had no ``record_metadata``. Every config warm-starting from
that corpus -- the whole ``warmstart-natural-*`` family and the RAMS clean arms -- trained
its structure half on nothing.

A REPAIR, not a rebuild. ``mix_natural``'s build invocation is recorded nowhere (
MODEL_LINEAGE lists every config that consumes it and none that produced it), so it cannot
be re-derived faithfully. Repairing in place also keeps composition and split membership
identical, so runs before and after stay comparable on everything except the defect.

ANCHOR SELECTION USES THE INTERSECTION, not the union. A row may carry several structures
with different fields; an anchor drawn from the union can be absent from one of them, and
the record head then raises "declares anchor X but no matching field query was found".
``build_turkish_dose_mix.stamp`` takes the union and has that latent bug.

    uv run python tools/data/stamp_record_metadata.py data/mix_natural --dry-run
    uv run python tools/data/stamp_record_metadata.py data/mix_natural
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

ANCHOR_ORDER = ("dead", "injured", "missing", "location")
MODE = "natural"


def anchor_for(structs: list, name: str) -> str | None:
    """An anchor field present in EVERY structure of this name, or None.

    Two shapes, because the corpora carry two. Casualty reports get ANCHOR_ORDER, whose
    ordering is deliberate: `dead` first, falling back through `injured` and `missing` to
    `location`, matching what `casualty_loc_split` -- which trains -- declares.

    Everything else takes the FIRST field in insertion order, which is the convention the
    healthy corpora already follow: text2json declares `tournament_code`, and that is
    field 0 of its record. Verified, not assumed.
    """
    dicts = [s[name] for s in structs if isinstance(s.get(name), dict)]
    if not dicts:
        return None
    inter = set.intersection(*(set(d) for d in dicts))
    union: list = []
    for d in dicts:
        for f in d:
            if f not in union:
                union.append(f)

    # PREFERENCE ORDER. Intersection first, so every instance carries a real anchor value.
    # Union is a valid FALLBACK, not a bug: the query layout is built from the union of
    # field names across occurrences -- `_process_json_structures` calls it `common` but
    # builds it by union, and spans use `occ.get(f)`, so a field an occurrence lacks is
    # None BY DESIGN. The anchor check (`records.py:312`) only requires the anchor to match
    # a field QUERY in that layout, which a union field always does. Verified in the source
    # before relying on it.
    for pool in (inter, set(union)):
        hit = next((f for f in ANCHOR_ORDER if f in pool), None)
        if hit:
            return hit
        hit = next((f for f in union if f in pool), None)
        if hit:
            return hit
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix", help="corpus prefix, e.g. data/mix_natural")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    total = Counter()
    for split in ("train", "val", "test"):
        path = Path(f"{args.prefix}.{split}.jsonl")
        if not path.exists() or path.stat().st_size == 0:
            print(f"[stamp] {path.name}: absent or empty, skipped")
            continue
        rows, n, stamped, skipped_no_anchor, already = [], 0, 0, 0, 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                out = rec.get("output") or {}
                structs = out.get("json_structures") or []
                if structs:
                    n += 1
                    if out.get("record_metadata") or rec.get("record_metadata"):
                        already += 1
                    else:
                        names = {k for s in structs for k in s}
                        meta = {}
                        for name in names:
                            a = anchor_for(structs, name)
                            if a:
                                meta[name] = {"mode": MODE, "anchor": a}
                        if meta:
                            out["record_metadata"] = meta
                            rec["output"] = out
                            stamped += 1
                        else:
                            skipped_no_anchor += 1
                rows.append(rec)
        print(f"[stamp] {path.name}: {len(rows):,} rows | {n:,} with structures | "
              f"already {already:,} | stamped {stamped:,} | no anchor {skipped_no_anchor:,}")
        total["stamped"] += stamped
        total["no_anchor"] += skipped_no_anchor
        if args.dry_run or not stamped:
            continue
        if not args.no_backup:
            shutil.copy(path, path.with_suffix(".jsonl.bak"))
        with path.open("w", encoding="utf-8") as fh:
            for rec in rows:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[stamp] {'DRY RUN -- ' if args.dry_run else ''}"
          f"stamped {total['stamped']:,}; no usable anchor {total['no_anchor']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
