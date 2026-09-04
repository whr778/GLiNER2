"""Make all three DocEE corpora offer the SAME classification menu.

Labels are an INPUT at inference, so a model trained where English offers 59 labels,
Chinese 58 and Turkish 60 has never been asked to consider `Armed Conflict` for a Chinese
document, and cannot answer `none` for an English one. The menus must be the union.

    docee          59
    docee_zh       58   (no Armed Conflict -- zh genuinely has no such documents)
    turkish_event  60   (59 + `none`)

Only the `labels` MENU is rewritten. `true_label` is never touched, so no annotation
changes and no label is invented -- a corpus that never uses `Armed Conflict` simply now
declines it explicitly instead of never being offered it.

    uv run python tools/data/unify_docee_menus.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402

CORPORA = ("docee", "docee_zh", "turkish_event")
SPLITS = ("train", "val", "test")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    union: set[str] = set()
    for c in CORPORA:
        for s in SPLITS:
            p = a.data / f"{c}.{s}.jsonl"
            if not p.exists():
                continue
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    for cl in (json.loads(line).get("output") or {}).get("classifications") or []:
                        union.update(str(x) for x in (cl.get("labels") or []))
    menu = sorted(union)
    print(f"[menu] union = {len(menu)} labels")

    for c in CORPORA:
        for s in SPLITS:
            p = a.data / f"{c}.{s}.jsonl"
            if not p.exists():
                continue
            rows, changed = [], 0
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    for cl in (r.get("output") or {}).get("classifications") or []:
                        if list(cl.get("labels") or []) != menu:
                            cl["labels"] = list(menu)
                            changed += 1
                    rows.append(r)
            print(f"[menu] {c}.{s}: {len(rows):,} rows, {changed:,} menus rewritten")
            if not a.dry_run:
                with p.open("w", encoding="utf-8") as fh:
                    for r in rows:
                        fh.write(dumps_record(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
