"""Build labels/unified-full.yaml: one label map over every TAXONOMY corpus in data/.

Measure before classifying (the standing rule): each split corpus in data/ is scored on its
label health, and only taxonomies may VOTE on spellings. A corpus is excluded from voting if
its labels are placeholders (> 5% `e_N`), mostly singletons (> 40%), or an open vocabulary
(> 10,000 distinct labels) -- measured 2026-09-30, the populations separate cleanly: taxonomy
corpora sit at <= 29% singletons, the rest at >= 43% or carry placeholders. Singleton share
alone is NOT enough: NuNER reads 0.2% singletons by this counter with 242,711 labels.

Roles, passed to `build_label_maps.build`:

* voters     -- taxonomy corpora; they form clusters and pick winners;
* canonical  -- eb17-best's taxonomy corpora; the base's spellings win (standing rule);
* followers  -- non-taxonomy corpora a config TRAINS under this map (FOLLOWERS); a spelling
               folding onto a voter label maps to it, and a cluster found only among
               followers collapses to its majority spelling (fold-only);
* open-vocab -- Pile-NER, NuNER, gliner_multilingual, knowledgator_gliner (ZERO_SHOT); left
               out entirely, kept raw by configs through `data.labels_passthrough`;
* translated -- a taxonomy corpus's non-English labels (TRANSLATIONS), mapped to English
               and kept distinct within their corpus.

    uv run python tools/train/build_unified_full.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_label_maps as B  # noqa: E402

# Open-vocabulary corpora: their free-text (and multilingual) labels ARE the lesson, so the
# map never touches them and configs keep them raw via data.labels_passthrough. The two
# multilingual ones were followers until 2026-09-30; their own spelling clusters put
# thousands of non-English entries (`eigenschap`, `meio de comunicação`) into the map.
ZERO_SHOT = {"pile_ner_def", "nuner_full", "replay_pile30",
             "gliner_multilingual", "knowledgator_gliner"}
# Non-taxonomy corpora a config actually TRAINS under this map (eb18 trains paraloq_json).
# Other excluded corpora are neither mapped nor mapping: including them only bloated the
# map (gliclass alone added ~44k classification entries).
FOLLOWERS = {"paraloq_json"}
# Labels a taxonomy corpus carries in another language, translated to English and kept
# DISTINCT within their corpus (translation must never merge two of its labels).
TRANSLATIONS = {"entities": {
    "人名": "Person", "法人名": "Company", "地名": "Location", "イベント名": "Event",
    "製品名": "Product", "施設名": "Facility", "政治的組織名": "Political Organization",
    "その他の組織名": "Organization",
}}
PLACEHOLDER = re.compile(r"^e_\d+$")
BASE_CONFIG = "tools/train/config/base/eb17-best.yaml"
OUT = Path("tools/train/config/labels/unified-full.yaml")


def split_files(name):
    return [f"data/{name}.{s}.jsonl" for s in ("train", "val", "test")
            if Path(f"data/{name}.{s}.jsonl").exists()]


def health(name):
    """(distinct labels, placeholder share of uses, singleton share of labels)."""
    uses = Counter()
    for f in split_files(name):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    uses.update(lab for _, lab in B.labels_by_category(json.loads(line)))
    if not uses:
        return None
    ph = sum(n for lab, n in uses.items() if PLACEHOLDER.match(lab)) / sum(uses.values())
    return len(uses), ph, sum(1 for n in uses.values() if n == 1) / len(uses)


def classify():
    """Return (voters, excluded {name: reasons}) over every split corpus in data/."""
    names = sorted({p.name.split(".")[0] for p in Path("data").glob("*.train.jsonl")})
    voters, excluded = [], {}
    for name in names:
        h = health(name)
        if h is None:
            continue
        n, ph, sg = h
        reasons = [r for r, hit in (("zero-shot", name in ZERO_SHOT), (f"placeholder {ph:.0%}", ph > 0.05),
                                    (f"singletons {sg:.0%}", sg > 0.40), (f"{n:,} labels", n > 10000)) if hit]
        (excluded.__setitem__(name, reasons) if reasons else voters.append(name))
    return voters, excluded


def main():
    voters, excluded = classify()
    base = {f.split("/")[-1].split(".")[0] for f in B.config_files(BASE_CONFIG)}
    canonical = sorted(base & set(voters))
    followers = sorted(FOLLOWERS & set(excluded))
    files = lambda names: [f for n in names for f in split_files(n)]
    maps, _ = B.build(files(voters), files(canonical), files(followers))
    for category, table in TRANSLATIONS.items():
        m = maps[category]
        for source, english in table.items():
            m[source] = m.get(english, english)     # the English label's FINAL spelling
    print(f"voters {len(voters)} | canonical {len(canonical)} | followers {len(followers)} | "
          f"excluded {len(excluded)}")
    for name, reasons in sorted(excluded.items()):
        print(f"  excluded {name}: {', '.join(reasons)}")
    for c, m in maps.items():
        print(f"  {c}: {len(m)} spellings collapse")
    header = ("# Unified label space, FULL -- GENERATED by tools/train/build_unified_full.py.\n"
              f"# voters = {len(voters)} TAXONOMY corpora of data/; canonical = eb17-best's taxonomy\n"
              "# corpora (the base's spellings win); followers = trained non-taxonomy corpora, mapped\n"
              "# onto a voter spelling of the same fold, and their own fold-only clusters collapsed.\n"
              "# Open-vocabulary corpora (Pile-NER, NuNER, gliner_multilingual, knowledgator) are not mapped: configs\n"
              "# keep them raw with data.labels_passthrough. Supersedes labels/unified.yaml (built\n"
              "# 2026-09-02 from 19 corpora) for NEW lines; the configs on unified.yaml are untouched.\n")
    block = {"labels": {c: {"rollup": False, "separator": ".", "map": m} for c, m in maps.items()}}
    OUT.write_text(header + yaml.safe_dump(block, allow_unicode=True, sort_keys=False,
                                           default_flow_style=False), encoding="utf-8")
    print(f"wrote {OUT}")


# PINNED MERGES for the NEXT base, applied on top of a committed map -- never by regenerating it, which
# would also pull in every corpus added since and move far more than the merge. The base's own map
# file stays byte-identical, so models warm-started from it keep its vocabulary (labels rule).
# Government.Protest -> Conflict.Demonstrate: one concept under two names, used near-equally for it in
# the Haiku gold (102 vs 111) and merged in the annotation ontology on 2026-10-06 (user).
MERGES = {"events": {"Government.Protest": "Conflict.Demonstrate"}}
V2_FROM = Path("tools/train/config/labels/unified-full.yaml")
V2_OUT = Path("tools/train/config/labels/unified-full-v2.yaml")


def derive_v2(src: Path = V2_FROM, out: Path = V2_OUT) -> dict:
    """Write ``out`` = ``src`` + MERGES, keeping the map CLOSED: an existing entry whose target is a
    merged-away label is re-pointed at the merge's target, and no target may itself be a key."""
    doc = yaml.safe_load(src.read_text(encoding="utf-8"))
    for cat, merges in MERGES.items():
        block = doc["labels"].setdefault(cat, {"rollup": False, "separator": ".", "map": {}})
        m = block.setdefault("map", {}) or {}
        for k, v in list(m.items()):
            if v in merges:
                m[k] = merges[v]
        m.update(merges)
        block["map"] = m
        stray = sorted({v for v in m.values() if v in m})
        if stray:
            raise SystemExit(f"[v2] {cat} map not closed: targets that are keys {stray}")
    header = (f"# Unified label space v2 -- DERIVED by tools/train/build_unified_full.py --v2 from {src.name}\n"
              f"# plus the pinned MERGES only: {MERGES}. {src.name} is unchanged; eb19 and its warm\n"
              "# starts keep it. Use this file for the NEXT base.\n")
    out.write_text(header + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, default_flow_style=False),
                   encoding="utf-8")
    print(f"wrote {out}")
    return doc


if __name__ == "__main__":
    derive_v2() if "--v2" in sys.argv else main()
