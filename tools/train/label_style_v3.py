"""Derive labels/unified-full-v3.yaml: Title_Snake over every training label (LABEL_STYLE_SPEC.md).

Called by ``build_unified_full.py --v3``. Inputs are the unified-full voters + followers and the
four reviewed files in tools/events_working_papers/label_style/. The style itself is
``gliner2.inference.label_style.title_snake`` -- the SAME function training and inference apply --
plus the reviewed squashed-segment splits, which only the map can carry. The file also gets a
``style:`` block (name, acronyms, short words), so training styles every label after the map and a
checkpoint can style what a user types. Refuses on a blank review decision or a failed gate.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

import yaml

import build_label_maps as B
import build_unified_full as U
from gliner2.inference.label_style import TITLE_SNAKE, title_snake, words

REVIEW = Path("tools/events_working_papers/label_style")
DICTIONARY = Path("/usr/share/dict/words")
OUT = Path("tools/train/config/labels/unified-full-v3.yaml")
REQUIRED = ("squashed_segments.tsv", "suspect_groups.tsv")   # every row must carry a decision
# Single words the camelCase splitter breaks (found by scanning the training labels, 2026-10-07):
# kept verbatim. ATPases by the user; GTPase, CoA (coenzyme A) and dL (decilitre) are the same defect;
# mg/dL keeps its slash, which means "per" (user, 2026-10-07).
PRESERVE = ["ATPases", "CoA", "GTPase", "dL", "mg/dL"]


def rows(name: str):
    with open(REVIEW / name, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def english():
    vocab = {w.strip().lower() for w in DICTIONARY.open() if w.strip()}

    def word(w):
        w = w.lower()
        return (w in vocab or any(w.endswith(s) and w[:-len(s)] in vocab for s in ("s", "es", "ed", "d", "ing"))
                or (w.endswith("ies") and w[:-3] + "y" in vocab))
    return word


def has_boundary(label: str) -> bool:
    return bool(re.search(r"[\s_\-]", label) or re.search(r"[a-z][A-Z]", label))


def review():
    """(squashed {cat: {segment: form}}, keep-apart {(cat, spelling): name}, acronym verdicts, dot-pair winners)."""
    for name in REQUIRED:
        blank = [r for r in rows(name) if not r["decision"].strip()]
        if blank:
            raise SystemExit(f"[v3] {name}: {len(blank)} rows without a decision, e.g. {blank[0]}")
    squashed = defaultdict(dict)
    for r in rows("squashed_segments.tsv"):
        d = r["decision"].strip()
        squashed[r["category"]][r["segment"]] = r["segment"].capitalize() if d.lower() == "one word" else d
    apart = {(r["category"], r["spelling"]): r["decision"].split(":", 1)[1].strip()
             for r in rows("suspect_groups.tsv") if r["decision"].lower().startswith("keep apart:")}
    verdicts = {r["token"]: r["decision"].strip().lower().startswith("acronym")
                for r in rows("all_caps_tokens.tsv") if r["decision"].strip()}
    winners = {v: r["decision"].strip() for r in rows("structure_dot_pairs.tsv") if r["decision"].strip()
               for v in r["group"].split(" | ")}
    return squashed, apart, verdicts, winners


def style_lists(uses: Dict[str, Counter], verdicts: Dict[str, bool], word) -> Dict:
    """The ``style:`` block: every ALL-CAPS token of the training labels that is an acronym (not an
    English word, or reviewed as one) -- short ones too, so a shouted phrase keeps `SRS` in `SRS-A` --
    and the short ones a reviewer judged words."""
    caps = {t for cat in uses for lab in uses[cat] for t in words(lab.replace(".", " ")) if t.isupper() and len(t) > 1}
    acronyms = sorted(t for t in caps if verdicts.get(t, not word(t)))
    short_words = sorted(t for t in caps if len(t) <= 3 and verdicts.get(t) is False)
    return {"name": TITLE_SNAKE, "acronyms": acronyms, "words": short_words, "preserve": PRESERVE}


def styler(category, squashed, style):
    def fn(label):
        if "." in label:
            return ".".join(fn(p) for p in label.split("."))
        return (squashed.get(category, {}).get(label.lower())
                or title_snake(label, style["acronyms"], style["words"], style["preserve"]))
    return fn


def source(variants, uses, category, winners):
    """The spelling a group is styled from: a reviewed dot-pair winner, else a boundary-bearing one."""
    for v in variants:
        if category == "structures" and v in winners:
            return winners[v]
    pool = [v for v in variants if has_boundary(v)] or variants
    if category == "structures":
        return min(pool, key=lambda v: B.style_rank(v, uses[v], True))
    return max(pool, key=lambda v: (uses[v], v))


def derive_v3(out: Path = OUT) -> dict:
    voters, _ = U.classify()
    uses = {c: Counter() for c in B.CATEGORIES}
    for name in voters + sorted(U.FOLLOWERS):
        for f in U.split_files(name):
            for line in open(f, encoding="utf-8"):
                for cat, lab in B.labels_by_category(json.loads(line)):
                    uses[cat][lab] += 1
    squashed, apart, verdicts, winners = review()
    style = style_lists(uses, verdicts, english())
    maps = {}
    for cat in B.CATEGORIES:
        fn = styler(cat, squashed, style)
        groups = defaultdict(list)
        for lab in uses[cat]:
            groups[B.fold(lab)].append(lab)
        m = {}
        for key, vs in groups.items():
            if not key:
                continue
            pinned = {B.SYNONYMS[v] for v in vs if cat == "entities" and v in B.SYNONYMS}
            target = fn(pinned.pop() if len(pinned) == 1 else source(vs, uses[cat], cat, winners))
            for v in vs:
                t = apart.get((cat, v), target)
                if v != t:
                    m[v] = t
        for src, eng in (U.TRANSLATIONS.get(cat) or {}).items():
            if src in uses[cat]:
                m[src] = fn(eng)
        for old, new in (U.MERGES.get(cat) or {}).items():
            for v in [k for k in uses[cat] if B.fold(k) == B.fold(old)]:
                m[v] = fn(new)
        gate(cat, m, uses[cat], style)
        maps[cat] = {"rollup": False, "separator": ".",
                     "map": dict(sorted(m.items(), key=lambda kv: (-uses[cat][kv[0]], kv[0])))}
    header = ("# Unified label space v3 -- GENERATED by tools/train/build_unified_full.py --v3 (LABEL_STYLE_SPEC.md).\n"
              "# Title_Snake over EVERY training label of the unified-full voters + followers; `style:` makes training\n"
              "# and inference style every other label the same way. Reviewed decisions: tools/events_working_papers/\n"
              "# label_style/*.tsv. Open-vocabulary corpora stay raw (data.labels_passthrough). For a NEW base only.\n")
    doc = {"style": style, "labels": maps}
    out.write_text(header + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, default_flow_style=False),
                   encoding="utf-8")
    print(f"wrote {out}: style acronyms {len(style['acronyms'])}, short words {len(style['words'])}")
    return doc


def gate(cat, m, uses, style):
    """Refuse a map whose targets are not styled, that is not closed, or that leaves a label unstyled."""
    ts = lambda x: title_snake(x, style["acronyms"], style["words"], style.get("preserve", ()))
    bad = sorted(t for t in set(m.values()) if ts(t) != t)
    open_ = {k: v for k, v in m.items() if v in m}
    unstyled = [lab for lab in uses if lab not in m and ts(lab) != lab]
    print(f"[v3 {cat}] {len(uses)} labels, {len(m)} map entries, {len(set(m.values()))} targets | "
          f"targets not styled {len(bad)}, not closed {len(open_)}, labels left unstyled {len(unstyled)}")
    if bad or open_ or unstyled:
        raise SystemExit(f"[v3 {cat}] gate failed: not styled {bad[:5]}, not closed {list(open_.items())[:5]}, "
                         f"unstyled {unstyled[:5]}")
