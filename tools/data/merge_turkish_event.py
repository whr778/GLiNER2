"""Join the Turkish TYPE pass and ENTITY pass into DocEE's converted shape.

Two passes bought over the SAME documents -- `annotate_event_type.py` for the event type
and `annotate_event_entities.py` for role-typed spans -- have to become one corpus in the
shape `docee` and `docee_zh` already use, or the three languages cannot train together:

    {"input": ..., "output": {"classifications": [...], "entities": {role: [spans]}}}

THE JOIN IS VERIFIED, NOT ASSUMED. Both passes read the same pool and the entity pass was
seeded from the type pass's own output, so the key sets should be identical. A silent
partial join would look like sparse annotation rather than a bug -- entities simply
missing on some rows, which is exactly what an under-trained entity head also looks like.
So orphans on BOTH sides are counted and reported, and `--require-exact` fails the build
rather than emitting a corpus whose sparsity has an unknown cause.

Keyed on `normalize_group_key`, the same key the split-hygiene checks and `SplitWriter`
use, so a document cannot join under one rule and split under another.

`none`-labelled documents (~16.9%) are kept by default. They are real negatives for a
59-way classifier -- "this Turkish news article reports no DocEE event" -- and dropping
them would leave a model that has never seen one. `--drop-none` removes them; that is a
decision about the label space, so it is explicit rather than a default.

    uv run python tools/data/merge_turkish_event.py --out-prefix data/turkish_event
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, normalize_group_key  # noqa: E402

TASK = "docee_event"
NONE_LABEL = "none"


def load(paths: list[str], field: str) -> dict[str, dict]:
    """Key -> record, for whichever pass. Later files do not overwrite earlier ones."""
    out: dict[str, dict] = {}
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"[merge] MISSING {path}, skipped")
            continue
        n = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if field not in r:
                    continue
                k = normalize_group_key(r.get("input") or "")[:300]
                if k and k not in out:
                    out[k] = r
                    n += 1
        print(f"[merge]   {path.name}: {n:,} rows")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type-files", nargs="+", default=[
        "data/turkish_gate/ev_ann_tr.jsonl",
        "data/turkish_gate/ev_ann_tr_pilot.jsonl",
        "data/turkish_gate/ev_ann_tr_pilot2.jsonl",
    ])
    ap.add_argument("--entity-files", nargs="+", default=[
        "data/turkish_gate/ev_ent_tr.jsonl",
        "data/turkish_gate/ev_ent_tr_pilot.jsonl",
    ])
    ap.add_argument("--en-corpus", type=Path, default=Path("data/docee.train.jsonl"),
                    help="source of the canonical label MENU offered on every record")
    ap.add_argument("--out-prefix", default="data/turkish_event")
    ap.add_argument("--drop-none", action="store_true",
                    help="drop documents the type pass labelled `none`. They are real "
                         "negatives for a 59-way classifier, so this is a decision about "
                         "the label space, not a cleanup.")
    ap.add_argument("--require-exact", action="store_true",
                    help="fail if either side has orphans, rather than emitting a corpus "
                         "whose sparsity has an unknown cause")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("[merge] TYPE pass:")
    types = load(args.type_files, "label")
    print("[merge] ENTITY pass:")
    ents = load(args.entity_files, "entities")

    kt, ke = set(types), set(ents)
    only_type, only_ent = kt - ke, ke - kt
    print(f"[merge] type {len(kt):,} | entity {len(ke):,} | joined {len(kt & ke):,}")
    print(f"[merge] orphans: type-only {len(only_type):,}, entity-only {len(only_ent):,}")
    if args.require_exact and (only_type or only_ent):
        raise SystemExit("--require-exact: orphans present, refusing to emit")

    menu = set()
    with args.en_corpus.open(encoding="utf-8") as fh:
        for line in fh:
            for c in (json.loads(line).get("output") or {}).get("classifications") or []:
                menu.update(str(x) for x in (c.get("labels") or []))
            if len(menu) >= 59:
                break
    # UNION, not append. `none` began as the one label outside DocEE's 59, so appending
    # it was safe -- until unify_docee_menus.py rewrote docee.train's menu to the union
    # of all three arms, which CONTAINS `none`. A rebuild then emitted it twice: 61
    # entries, 60 unique. A duplicated label is a menu the model is shown twice, and
    # nothing downstream errors on it. Order-independent this way.
    labels = sorted(menu | {NONE_LABEL})
    print(f"[merge] label menu: {len(labels)} ({len(menu - {NONE_LABEL})} DocEE + `{NONE_LABEL}`)")

    kept, dropped_none, with_ents = 0, 0, 0
    lab_counts, role_counts = Counter(), Counter()
    with SplitWriter(Path(args.out_prefix), seed=args.seed) as w:
        for k in sorted(kt):
            rec = types[k]
            label = rec.get("label")
            if label == NONE_LABEL and args.drop_none:
                dropped_none += 1
                continue
            if label not in menu and label != NONE_LABEL:
                continue                      # never emit a label off the menu
            entities = (ents.get(k) or {}).get("entities") or {}
            out: dict = {"classifications": [{"task": TASK, "labels": labels,
                                              "true_label": [label]}]}
            if entities:
                out["entities"] = entities
                with_ents += 1
                for r_ in entities:
                    role_counts[r_] += 1
            lab_counts[label] += 1
            kept += 1
            w.write({"input": rec["input"], "output": out},
                    group=normalize_group_key(rec["input"])[:300])
    print(f"[merge] {w.summary()}")
    print(f"[merge] kept {kept:,} ({with_ents:,} with entities = "
          f"{with_ents / max(kept, 1):.1%}); dropped `{NONE_LABEL}`: {dropped_none:,}")
    print(f"[merge] distinct labels {len(lab_counts)}; top: {lab_counts.most_common(5)}")
    if role_counts:
        print(f"[merge] roles {len(role_counts)}; top: {role_counts.most_common(8)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
