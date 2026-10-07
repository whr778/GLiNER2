"""Measure what Title_Snake would do to the unified-full training vocabulary (voters + followers).

Reports per category: labels changed, squashed-only labels (no spelling carries word boundaries),
fold clusters holding an ALL-CAPS member beside other casings (acronym or shouted word?), events
clusters mixing an event TYPE with a ROLE, and the corpora behind each, for reading.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "tools/train")
import build_label_maps as B
import build_unified_full as U

WORDS = {w.strip().lower() for w in open("/usr/share/dict/words") if w.strip()}
TOKEN = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def latin(label):
    return not re.search(r"[^\x00-\x7f]", label)


def words(label):
    return [t for part in re.split(r"[\s_\-/]+", label) for t in TOKEN.findall(part)]


def title_snake(label):
    if not latin(label):
        return label
    if "." in label:
        return ".".join(title_snake(p) for p in label.split("."))
    return "_".join(t if (t.isupper() and len(t) > 1) else t.capitalize() for t in words(label)) or label


def has_boundary(label):
    return bool(re.search(r"[\s_\-]", label) or re.search(r"[a-z][A-Z]", label))


def squashed_multiword(label):
    """Lowercase or ALL-CAPS, no boundary, not a dictionary word, long enough to be several words."""
    w = label.replace(".", "")
    return (w.isalpha() and (w.islower() or w.isupper()) and len(w) >= 8 and w.lower() not in WORDS)


def main():
    voters, _ = U.classify()
    followers = sorted(U.FOLLOWERS)
    names = voters + followers
    uses = {c: Counter() for c in B.CATEGORIES}
    where = {c: defaultdict(Counter) for c in B.CATEGORIES}
    kind = defaultdict(set)                       # events label -> {"type", "role"}
    for name in names:
        for f in U.split_files(name):
            for line in open(f, encoding="utf-8"):
                rec = json.loads(line)
                for cat, lab in B.labels_by_category(rec):
                    uses[cat][lab] += 1
                    where[cat][lab][name] += 1
                for e in (rec.get("output") or {}).get("events") or []:
                    if isinstance(e, dict):
                        if isinstance(e.get("event_type"), str):
                            kind[e["event_type"]].add("type")
                        for a in e.get("arguments") or []:
                            if isinstance(a, dict) and isinstance(a.get("role"), str):
                                kind[a["role"]].add("role")
    report = {}
    for cat in B.CATEGORIES:
        labs = uses[cat]
        clusters = defaultdict(list)
        for lab in labs:
            clusters[B.fold(lab)].append(lab)
        changed = [l for l in labs if title_snake(l) != l]
        nonlatin = [l for l in labs if not latin(l)]
        squashed_only = sorted((k for k, vs in clusters.items() if k and not any(has_boundary(v) for v in vs)
                                and any(squashed_multiword(v) for v in vs)),
                               key=lambda k: -sum(labs[v] for v in clusters[k]))
        caps_mixed = [vs for k, vs in clusters.items() if k and len(vs) > 1
                      and any(v.isupper() and len(v) > 1 for v in vs) and any(not v.isupper() for v in vs)]
        styles_split = [vs for k, vs in clusters.items() if k and len({title_snake(v) for v in vs}) > 1]
        type_role = ([vs for k, vs in clusters.items() if k and {"type", "role"} <= set().union(*(kind[v] for v in vs))]
                     if cat == "events" else [])
        print(f"\n===== {cat}: {len(labs):,} distinct labels, {len(clusters):,} fold groups, "
              f"{len(changed):,} change under Title_Snake, {len(nonlatin):,} non-Latin (unchanged)")
        print(f"  squashed-only groups (no spelling has word boundaries, likely multi-word): {len(squashed_only)}")
        for k in squashed_only[:25]:
            print(f"    {clusters[k]}  uses {sum(labs[v] for v in clusters[k])}  corpora {dict(sum((where[cat][v] for v in clusters[k]), Counter()).most_common(3))}")
        print(f"  groups with an ALL-CAPS member beside other casings: {len(caps_mixed)}")
        for vs in sorted(caps_mixed, key=lambda vs: -sum(labs[v] for v in vs))[:25]:
            print("    " + " | ".join(f"{v} {labs[v]} {dict(where[cat][v].most_common(2))}" for v in vs))
        print(f"  groups whose spellings STYLE DIFFERENTLY (need a pick): {len(styles_split)}")
        for vs in styles_split[:12]:
            print("    " + " | ".join(f"{v}->{title_snake(v)}" for v in vs))
        if cat == "events":
            print(f"  groups mixing an event TYPE and a ROLE: {len(type_role)}")
            for vs in type_role[:25]:
                print("    " + " | ".join(f"{v} {sorted(kind[v])} {dict(where[cat][v].most_common(2))}" for v in vs))
        report[cat] = {"labels": len(labs), "changed": len(changed), "squashed_only": [clusters[k] for k in squashed_only],
                       "caps_mixed": caps_mixed, "styles_split": styles_split, "type_role": type_role}
    Path(__file__).resolve().parent / "report.json".write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
