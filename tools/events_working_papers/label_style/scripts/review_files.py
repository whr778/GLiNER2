"""Write the LABEL_STYLE_SPEC review files: squashed labels, suspect groups, all-caps verdicts, dot/underscore pairs."""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "tools/train")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_label_maps as B
import build_unified_full as U
from measure import WORDS, words, latin, has_boundary
from squashed import seg, word

OUT = Path("tools/events_working_papers/label_style")
FUNCTION = {"of", "by", "for", "per", "in", "on", "to", "id", "at", "or", "and", "a", "the"}


def acronym(tok):
    """ALL-CAPS token kept as-is when it has <= 3 letters (PER, LOC, GPE) or is not an English word
    (NORP); a longer English word is shouted (PERSON -> Person)."""
    return tok.isupper() and len(tok) > 1 and (len(tok) <= 3 or not word(tok))


def styled(label, pinned=None):
    if not latin(label):
        return label
    if "." in label:
        return ".".join(styled(p, pinned) for p in label.split("."))
    if pinned and label.lower() in pinned:
        return pinned[label.lower()]
    return "_".join(t if acronym(t) else t.capitalize() for t in words(label)) or label


def surfaces(rec, cat, lab):
    out = rec.get("output") or {}
    if cat == "entities":
        v = (out.get("entities") or {}).get(lab)
        return [str(x)[:50] for x in v[:2]] if isinstance(v, list) else []
    if cat == "events":
        got = []
        for e in out.get("events") or []:
            if not isinstance(e, dict):
                continue
            if e.get("event_type") == lab:
                got.append(f"type, trigger {str(e.get('triggers') or e.get('trigger'))[:40]}")
            for a in e.get("arguments") or []:
                if isinstance(a, dict) and a.get("role") == lab:
                    got.append(f"role, {str(a.get('entity'))[:40]} in {e.get('event_type')}")
        return got[:2]
    if cat == "relations":
        return [str(r[lab])[:60] for r in out.get("relations") or [] if isinstance(r, dict) and lab in r][:2]
    if cat == "structures":
        got = []
        for s in out.get("json_structures") or []:
            for name, fields in (s.items() if isinstance(s, dict) else []):
                if name == lab:
                    got.append("structure name")
                if isinstance(fields, dict) and lab in fields:
                    got.append(f"field of {name}: {str(fields[lab])[:40]}")
        return got[:2]
    return []


def main():
    voters, _ = U.classify()
    names = voters + sorted(U.FOLLOWERS)
    uses = {c: Counter() for c in B.CATEGORIES}
    where = {c: defaultdict(Counter) for c in B.CATEGORIES}
    kind = defaultdict(set)
    ex = defaultdict(list)
    for name in names:
        for f in U.split_files(name):
            for line in open(f, encoding="utf-8"):
                rec = json.loads(line)
                for cat, lab in B.labels_by_category(rec):
                    uses[cat][lab] += 1
                    where[cat][lab][name] += 1
                    key = (cat, lab, name)
                    if len(ex[key]) < 2:
                        ex[key].extend(surfaces(rec, cat, lab)[: 2 - len(ex[key])])
                for e in (rec.get("output") or {}).get("events") or []:
                    if isinstance(e, dict):
                        if isinstance(e.get("event_type"), str):
                            kind[e["event_type"]].add("type")
                        for a in e.get("arguments") or []:
                            if isinstance(a, dict) and isinstance(a.get("role"), str):
                                kind[a["role"]].add("role")
    OUT.mkdir(parents=True, exist_ok=True)

    def corp(cat, lab):
        return "; ".join(f"{c} {n}" for c, n in where[cat][lab].most_common(3))

    def sample(cat, lab):
        c = where[cat][lab].most_common(1)[0][0]
        return f"{c}: " + " | ".join(ex[(cat, lab, c)])

    # 1. squashed: every dotted segment with no boundary in any spelling, not a word, segmentable or not
    rows = []
    for cat in B.CATEGORIES:
        groups = defaultdict(list)
        for lab in uses[cat]:
            groups[B.fold(lab)].append(lab)
        for key, vs in groups.items():
            if not key or any(has_boundary(v) for v in vs):
                continue
            lab = max(vs, key=lambda v: uses[cat][v])
            # A Capitalized token with no inner capitals is ONE word by its author's own convention
            # (Eukaryota); only all-lowercase or ALL-CAPS spellings can hide a word boundary.
            if not any(v.replace(".", "").islower() or v.replace(".", "").isupper() for v in vs):
                continue
            for s in lab.split("."):
                if not (s.isalpha() and len(s) >= 8 and latin(s) and not word(s)):
                    continue
                parts = seg(s.lower())
                proposed = "_".join(p.capitalize() for p in parts) if parts else ""
                doubt = (not parts) or any(len(p) <= 3 and p not in FUNCTION for p in parts)
                rows.append({"category": cat, "segment": s.lower(), "in_label": lab, "uses": sum(uses[cat][v] for v in vs),
                             "corpora": corp(cat, lab), "proposed": proposed or "???",
                             "flag": "CHECK" if doubt else "", "example": sample(cat, lab), "decision": ""})
    seen, uniq = set(), []
    for r in sorted(rows, key=lambda r: (r["category"], -r["uses"])):
        if (r["category"], r["segment"]) not in seen:
            seen.add((r["category"], r["segment"]))
            uniq.append(r)
    write(OUT / "squashed_segments.tsv", uniq)

    # 2. suspects: role/type mixes, and ALL-CAPS beside other casings, with text behind every spelling
    rows = []
    for cat in B.CATEGORIES:
        groups = defaultdict(list)
        for lab in uses[cat]:
            groups[B.fold(lab)].append(lab)
        for key, vs in groups.items():
            if not key:
                continue
            kinds = set().union(*(kind[v] for v in vs)) if cat == "events" else set()
            caps = len(vs) > 1 and any(v.isupper() and len(v) > 1 for v in vs) and any(not v.isupper() for v in vs)
            reason = "role+type" if {"type", "role"} <= kinds else ("ALL-CAPS beside other casing" if caps else "")
            if not reason:
                continue
            for v in sorted(vs, key=lambda v: -uses[cat][v]):
                rows.append({"category": cat, "group": key, "reason": reason, "spelling": v,
                             "kind": "+".join(sorted(kind[v])) if cat == "events" else "", "uses": uses[cat][v],
                             "corpora": corp(cat, v), "proposed": styled(v), "example": sample(cat, v), "decision": ""})
    write(OUT / "suspect_groups.tsv", rows)

    # 3. every ALL-CAPS token: acronym (kept) or shouted word (Title-cased)
    caps = Counter()
    for cat in B.CATEGORIES:
        for lab, n in uses[cat].items():
            if latin(lab):
                for t in words(lab.replace(".", " ")):
                    if t.isupper() and len(t) > 1:
                        caps[t] += n
    write(OUT / "all_caps_tokens.tsv", [{"token": t, "uses": n, "verdict": "acronym (kept)" if acronym(t) else "word -> " + t.capitalize(),
                                         "decision": ""} for t, n in caps.most_common()])

    # 4. structures: groups differing only by dots vs underscores -- fewest dots wins, then styled
    rows = []
    groups = defaultdict(list)
    for lab in uses["structures"]:
        groups[B.fold(lab)].append(lab)
    for key, vs in groups.items():
        if key and len({styled(v) for v in vs}) > 1:
            win = min(vs, key=lambda v: B.style_rank(v, uses["structures"][v], True))
            rows.append({"group": " | ".join(vs), "winner": win, "proposed": styled(win), "uses": sum(uses["structures"][v] for v in vs),
                         "decision": ""})
    write(OUT / "structure_dot_pairs.tsv", rows)


def write(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"{path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
