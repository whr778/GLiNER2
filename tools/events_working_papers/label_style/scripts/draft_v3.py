"""DRAFT unified-full-v3.yaml: Title_Snake over every training label (LABEL_STYLE_SPEC.md).

    uv run python tools/events_working_papers/label_style/scripts/draft_v3.py

Reads the four review files in label_style/; a filled `decision` overrides the default, so rerunning
after review produces the reviewed map. Defaults while a decision is blank:
- squashed segments: split in events and structures unless flagged CHECK; one word elsewhere;
- suspect groups: merge, except the pinned KEEP_APART below;
- structure dot pairs: fewest dots wins.
Writes tools/train/config/labels/unified-full-v3.yaml and prints the gates.
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "tools/train")
import build_label_maps as B  # noqa: E402
import build_unified_full as U  # noqa: E402
from measure import has_boundary, latin, words  # noqa: E402
from review_files import acronym  # noqa: E402

REVIEW = HERE.parent
OUT = Path("tools/train/config/labels/unified-full-v3.yaml")
SPLIT_BY_DEFAULT = {"events", "structures"}
# (category, exact spelling) -> target. Default keep-apart: DuEE's event TYPE 求婚 is not the cc_news ROLE.
KEEP_APART = {("events", "proposal"): "Marriage_Proposal"}


def rows(name):
    with open(REVIEW / name, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def squashed_table():
    """{category: {segment: styled form}} from squashed_segments.tsv, decisions first."""
    table = defaultdict(dict)
    for r in rows("squashed_segments.tsv"):
        d = r["decision"].strip()
        if d:
            table[r["category"]][r["segment"]] = r["segment"].capitalize() if d.lower() == "one word" else d
        elif r["category"] in SPLIT_BY_DEFAULT and r["flag"] != "CHECK" and r["proposed"] != "???":
            table[r["category"]][r["segment"]] = r["proposed"]
    return table


def acronym_overrides():
    """{TOKEN: kept?} where the reviewer corrected the verdict."""
    out = {}
    for r in rows("all_caps_tokens.tsv"):
        d = r["decision"].strip().lower()
        if d:
            out[r["token"]] = d.startswith("acronym")
    return out


def keep_apart():
    pins = dict(KEEP_APART)
    for r in rows("suspect_groups.tsv"):
        d = r["decision"].strip()
        if d.lower().startswith("keep apart:"):
            pins[(r["category"], r["spelling"])] = d.split(":", 1)[1].strip()
    return pins


def styler(category, table, overrides):
    def is_acronym(tok):
        return overrides.get(tok, acronym(tok))

    def style(label):
        if not latin(label):
            return label
        if "." in label:
            return ".".join(style(p) for p in label.split("."))
        if label.lower() in table.get(category, {}):
            return table[category][label.lower()]
        return "_".join(t if is_acronym(t) else t.capitalize() for t in words(label)) or label
    return style


def dot_pair_winners():
    """{spelling: winner} from structure_dot_pairs.tsv decisions (blank: fewest dots, as `source`)."""
    out = {}
    for r in rows("structure_dot_pairs.tsv"):
        d = r["decision"].strip()
        if d:
            for v in r["group"].split(" | "):
                out[v] = d
    return out


def source(variants, uses, category, winners=None):
    """The spelling a group is styled from: a reviewed dot-pair winner, else a boundary-bearing one
    if any (structures: fewest dots)."""
    for v in variants:
        if winners and v in winners:
            return winners[v]
    pool = [v for v in variants if has_boundary(v)] or variants
    if category == "structures":
        return min(pool, key=lambda v: B.style_rank(v, uses[v], True))
    return max(pool, key=lambda v: (uses[v], v))


def main():
    voters, _ = U.classify()
    names = voters + sorted(U.FOLLOWERS)
    uses = {c: Counter() for c in B.CATEGORIES}
    for name in names:
        for f in U.split_files(name):
            for line in open(f, encoding="utf-8"):
                import json
                for cat, lab in B.labels_by_category(json.loads(line)):
                    uses[cat][lab] += 1
    table, overrides, pins, winners = squashed_table(), acronym_overrides(), keep_apart(), dot_pair_winners()
    v2 = yaml.safe_load(open("tools/train/config/labels/unified-full-v2.yaml", encoding="utf-8"))["labels"]
    merges = {c: (U.MERGES.get(c) or {}) for c in B.CATEGORIES}
    out, report = {}, {}
    for cat in B.CATEGORIES:
        style = styler(cat, table, overrides)
        groups = defaultdict(list)
        for lab in uses[cat]:
            groups[B.fold(lab)].append(lab)
        m = {}
        for key, vs in groups.items():
            if not key:
                continue
            pinned = {B.SYNONYMS[v] for v in vs if cat == "entities" and v in B.SYNONYMS}
            target = style(pinned.pop() if len(pinned) == 1 else source(vs, uses[cat], cat, winners if cat == "structures" else None))
            for v in vs:
                t = pins.get((cat, v), target)
                if v != t:
                    m[v] = t
        for src, english in (U.TRANSLATIONS.get(cat) or {}).items():     # translated non-Latin labels
            if src in uses[cat]:
                m[src] = style(english)
        for old, new in merges[cat].items():                             # v2's pinned merges, styled
            for v in [k for k in uses[cat] if B.fold(k) == B.fold(old)]:
                m[v] = style(new)
        # GATES
        targets = set(m.values())
        bad_style = sorted(t for t in targets if style(t) != t)
        not_closed = {k: v for k, v in m.items() if v in m}
        fold_clash = defaultdict(set)
        for k, t in m.items():
            fold_clash[t].add(B.fold(k))
        merged = {t: f for t, f in fold_clash.items() if len(f) > 1}
        unstyled = sum(n for lab, n in uses[cat].items() if lab not in m and style(lab) != lab and latin(lab))
        report[cat] = {"labels": len(uses[cat]), "map entries": len(m), "targets": len(targets),
                       "targets failing style": len(bad_style), "map not closed": len(not_closed),
                       "targets joining several fold groups": len(merged), "uses left unstyled": unstyled}
        print(f"\n[{cat}] {report[cat]}")
        if bad_style: print("  BAD STYLE e.g.", bad_style[:5])
        if not_closed: print("  NOT CLOSED e.g.", dict(list(not_closed.items())[:5]))
        if merged: print("  JOINS GROUPS (merges / keep-apart / translations) e.g.", {t: sorted(f)[:4] for t, f in list(merged.items())[:8]})
        old = ((v2.get(cat) or {}).get("map") or {})
        moved = sum(1 for k in set(old) | set(m) if m.get(k, style(k) if latin(k) else k) != old.get(k, k))
        print(f"  differs from v2 for {moved} spellings")
        out[cat] = {"rollup": False, "separator": ".",
                    "map": dict(sorted(m.items(), key=lambda kv: (-uses[cat][kv[0]], kv[0])))}
    header = ("# Unified label space v3 -- DRAFT for review, DO NOT TRAIN. Generated by\n"
              "# tools/events_working_papers/label_style/scripts/draft_v3.py (LABEL_STYLE_SPEC.md).\n"
              "# Title_Snake over EVERY training label of the unified-full voters + followers: words capitalised and\n"
              "# joined by `_`, acronyms kept (<= 3 letters or not an English word), dots kept, non-Latin unchanged.\n"
              "# Each fold group is styled from its boundary-bearing spelling (PlaceOfEmployment, not placeofemployment).\n"
              "# Defaults where label_style/*.tsv decisions are blank: squashed segments split in events/structures\n"
              "# unless CHECK; suspects merged except `proposal` (DuEE marriage-proposal TYPE) -> Marriage_Proposal.\n"
              "# Open-vocabulary corpora are not mapped (data.labels_passthrough). Includes v2's pinned merges.\n")
    OUT.write_text(header + yaml.safe_dump({"labels": out}, allow_unicode=True, sort_keys=False, default_flow_style=False),
                   encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
