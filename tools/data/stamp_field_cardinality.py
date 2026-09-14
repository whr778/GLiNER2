"""Declare per-field CARDINALITY in a corpus's ``record_metadata``, from its own usage.

WHAT THIS CHANGES, AND WHY IT IS NOT A BUG FIX. Cardinality selects the record head's
training loss: a SCALAR field is supervised with ``_scalar_field_nll`` -- a softmax over
candidates plus an explicit ``ABSENT`` -- and a LIST field with ``_list_field_bce``.
Corpora declare ``mode`` and ``anchor`` and nothing else, so `_default_cardinality` sees no
dtype and returns ``ZERO_OR_MORE`` for every non-anchor field. Measured across all of
``data/``: **99.8% of structure-field occurrences hold exactly one filler**, and only 1.9%
of distinct (structure, field) pairs ever hold more than one. So very nearly every
structure field is trained as multi-label when the data says it is single-valued.

Fixing that moves every future model's structure numbers, which makes it a DECISION with an
A/B in front of it, not a repair to apply everywhere. This tool exists to build the
treatment arm.

THE RULE IS THE CORPUS'S OWN EVIDENCE, not a template and not a guess: a field is scalar
iff it never holds more than one filler anywhere in TRAIN. Train only, so a decision baked
into val/test schemas is never informed by val/test content -- the same one-directional
provenance `repair_contradicted_negatives.py` keeps.

THE ANCHOR IS NEVER DECLARED. `compile_record_specs` uses a declared cardinality in
preference to its default, so declaring one for the anchor would demote it from
``REQUIRED_ONE`` -- the one field whose cardinality is structural rather than observed.

    audit : uv run python tools/data/stamp_field_cardinality.py data/mix_natural --dry-run
    apply : uv run python tools/data/stamp_field_cardinality.py data/mix_natural
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")
SCALAR, LIST = "optional_one", "zero_or_more"


def _fillers(value) -> int:
    if isinstance(value, list):
        return len(value)
    return 0 if value in (None, "") else 1


def observed_cardinality(train: Path) -> dict:
    """``{structure: {field: "optional_one" | "zero_or_more"}}`` from TRAIN usage."""
    seen: dict = defaultdict(Counter)
    for line in train.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        out = (json.loads(line).get("output") or {})
        for inst in out.get("json_structures") or []:
            if not isinstance(inst, dict):
                continue
            for name, body in inst.items():
                if not isinstance(body, dict):
                    continue
                for field, value in body.items():
                    n = _fillers(value)
                    seen[name][field] = max(seen[name][field], n)
    return {name: {f: (SCALAR if n <= 1 else LIST) for f, n in fields.items()}
            for name, fields in seen.items()}


def stamp(path: Path, cards: dict, apply: bool) -> Counter:
    stats = Counter()
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        rec = json.loads(raw)
        out = rec.get("output") or {}
        meta = out.get("record_metadata")
        if isinstance(meta, dict):
            for name, cfg in meta.items():
                if not isinstance(cfg, dict) or name not in cards:
                    continue
                anchor = cfg.get("anchor")
                # ONLY the fields THIS record uses. The corpus-wide map can be enormous --
                # text2json carries one structure name with 7,250 distinct field names,
                # because it invents fields per document -- and stamping all of them onto
                # every record turned a 35MB corpus into 3.8GB, a 108x blow-up, before
                # this line existed. A declaration for a field a record does not contain
                # is also meaningless: no query is built for it.
                used = {f for inst in out.get("json_structures") or []
                        if isinstance(inst, dict)
                        for n2, body in inst.items() if n2 == name and isinstance(body, dict)
                        for f in body}
                fields = {f: {"cardinality": c} for f, c in cards[name].items()
                          if f != anchor and f in used}
                if not fields:
                    continue
                cfg["fields"] = fields
                stats["records_stamped"] += 1
                stats["scalar"] += sum(1 for v in fields.values()
                                       if v["cardinality"] == SCALAR)
                stats["list"] += sum(1 for v in fields.values()
                                     if v["cardinality"] == LIST)
                break
        stats["records"] += 1
        lines.append(json.dumps(rec, ensure_ascii=False))
    if apply and stats["records_stamped"]:
        backup = path.with_suffix(path.suffix + ".precardinality")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix", help="corpus prefix, e.g. data/mix_natural")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    train = Path(f"{args.prefix}.train.jsonl")
    if not train.is_file():
        raise SystemExit(f"no train split at {train}")

    cards = observed_cardinality(train)
    if not cards:
        raise SystemExit(f"{train.name} carries no structures; nothing to declare")

    n_scalar = sum(1 for fs in cards.values() for c in fs.values() if c == SCALAR)
    n_list = sum(1 for fs in cards.values() for c in fs.values() if c == LIST)
    print(f"[cardinality] {len(cards)} structure type(s) from {train.name}: "
          f"{n_scalar} scalar field(s), {n_list} list field(s)")
    for name, fields in sorted(cards.items())[:8]:
        multi = [f for f, c in fields.items() if c == LIST]
        print(f"   {name:28s} {len(fields):3d} fields"
              + (f"  list: {', '.join(sorted(multi)[:5])}" if multi else "  all scalar"))

    total = Counter()
    for split in SPLITS:
        p = Path(f"{args.prefix}.{split}.jsonl")
        if not p.is_file() or p.stat().st_size == 0:
            continue
        st = stamp(p, cards, not args.dry_run)
        total.update(st)
        print(f"  {split:5s} {st['records']:8,d} records  "
              f"{st['records_stamped']:8,d} stamped")

    print(f"\n{'DRY RUN -- ' if args.dry_run else 'APPLIED: '}"
          f"{total['records_stamped']:,} of {total['records']:,} records now declare "
          f"per-field cardinality ({total['scalar']:,} scalar / {total['list']:,} list "
          f"declarations)")
    if args.dry_run:
        print("re-run without --dry-run to write (originals kept as *.precardinality)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
