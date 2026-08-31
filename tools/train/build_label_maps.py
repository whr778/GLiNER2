"""Emit `labels:` maps that collapse spelling variants of the same label.

GLiNER2 takes the label as an INPUT string, so `CompanyName` and `Company Name` are two
different queries for one field of one taxonomy. This reports, per category, the clusters
that case/separator-fold together and the map that unifies them.

Only same-category clusters merge, and EXCLUDE lists pairs that fold together but mean
different things -- docee's `Target` is who an attack hit, bio_ner_relations' `target` is
the substrate of a kinase. The winning spelling is the most frequent, so the rewrite moves
as few uses as possible.

    uv run python tools/train/build_label_maps.py <config.yaml>
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Fold together but are not the same concept. Keyed by folded form.
EXCLUDE = {"target"}

# Abbreviations and spellings of the SAME word, which case-folding cannot reach: `LOC`
# folds to "loc" and `Location` to "location", so they never cluster. Hand-pinned because
# it is a judgement about meaning, not a string operation.
#
# Deliberately NOT merged, each checked by reading the surfaces they tag:
#   GPE   wikievents tags Thailand/Ontario/Iraq -- geo-political entities, a real
#         OntoNotes distinction from LOC (Indian Ocean, Ninoy Aquino Airport).
#   FAC   kept as Facility, not folded into Location: shrine/house/Murrah Building.
#   NORP  kaznerd tags nationalities (Kyrgyz, Kazakhstani), not organizations.
#   People/Place  synonyms rather than abbreviations, and docee's People tags groups
#         ("Children with kidney cancers"), not individuals.
#   TIME/Date  NOT cross-merged: redocred's TIME holds dates (2013, August 16 2013)
#         while casie's Time holds clock/relative time. Folding them would bury that.
SYNONYMS = {
    "ORG": "Organization", "ORGANIZATION": "Organization", "ORGANISATION": "Organization",
    "Organisation": "Organization", "organisation": "Organization",
    "PER": "Person", "PERSON": "Person",
    "LOC": "Location", "Loc": "Location", "LOCATION": "Location",
    "DATE": "Date", "TIME": "Time", "EVENT": "Event", "PRODUCT": "Product",
    "MONEY": "Money", "LANGUAGE": "Language", "FAC": "Facility",
}

CATEGORIES = ("entities", "relations", "events", "classifications", "structures")


def style_rank(label: str, count: int, prefer_snake: bool = False):
    """Rank a spelling so one convention wins every cluster, not whichever is commoner.

    Spaces lose first -- a space reads as two labels. Dots lose next, and by COUNT, for a
    functional reason rather than taste: "." is the rollup separator, so `asteroid.name`
    would split to `asteroid` the moment rollup is enabled. Counting rather than testing
    presence also settles nested paths, where every variant has dots and the fewest wins
    (`asteroid.discovery_year` over `asteroid.discovery.year`). Only then does frequency
    decide, so CASE still follows each corpus's own convention -- bio_ner_relations keeps
    lowercase `protein`, docee keeps `Economic Loss` -- while separator style is uniform.
    Alphabetical last, so the result is deterministic.

    ``prefer_snake`` is set for structure FIELD names only, where `areas_for_improvement`
    beating `areasforimprovement` is plainly better. It is deliberately NOT set for the
    taxonomies: chfinann names its fields `CompanyName`/`StartDate`/`EquityFreeze`, and
    forcing snake there would fight a convention the corpus applies on purpose.
    """
    snake = -(("_" in label) and " " not in label and "." not in label) if prefer_snake else 0
    return (label.count(" "), label.count("."), snake, -count, label)


def fold(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower())


def labels_by_category(record: dict) -> list[tuple[str, str]]:
    """Yield (category, label) for every label the trainer can transform."""
    out = []
    body = record.get("output") or record.get("schema") or {}
    for label in (body.get("entities") or {}):
        out.append(("entities", label))
    for label in (body.get("entity_descriptions") or {}):
        out.append(("entities", label))
    for rel in body.get("relations") or []:
        if isinstance(rel, dict):
            out.extend(("relations", name) for name in rel)
    for event in body.get("events") or []:
        if not isinstance(event, dict):
            continue
        if isinstance(event.get("event_type"), str):
            out.append(("events", event["event_type"]))
        for arg in event.get("arguments") or []:
            if isinstance(arg, dict) and isinstance(arg.get("role"), str):
                out.append(("events", arg["role"]))
    for block in body.get("classifications") or []:
        if not isinstance(block, dict):
            continue
        out.extend(("classifications", x) for x in block.get("labels") or [])
        true = block.get("true_label")
        out.extend(("classifications", x) for x in ([true] if isinstance(true, str) else true or []))
    for entry in body.get("json_structures") or []:
        if not isinstance(entry, dict):
            continue
        for name, fields in entry.items():
            out.append(("structures", name))
            if isinstance(fields, dict):
                out.extend(("structures", f) for f in fields)
    return [(c, l) for c, l in out if isinstance(l, str)]


def inputs_to_files(args: list[str]) -> list[str]:
    """Accept configs (expand their data section) and/or raw .jsonl paths."""
    files = []
    for arg in args:
        files += config_files(arg) if arg.endswith((".yaml", ".yml")) else [arg]
    return [f for f in sorted(set(files)) if Path(f).exists()]


def config_files(config_path: str) -> list[str]:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))["data"]
    files = []
    for base in cfg.get("corpora") or []:
        files += [f"{base}.{split}.jsonl" for split in ("train", "val", "test")]
    for entry in (cfg.get("event_files") or {}).values():
        files += list(entry.values())
    return [f for f in sorted(set(files)) if Path(f).exists()]


def scan(paths):
    """Count label uses per category over a set of files."""
    import json
    uses = {c: Counter() for c in CATEGORIES}
    for path in paths:
        for line in Path(path).open(encoding="utf-8"):
            for category, label in labels_by_category(json.loads(line)):
                uses[category][label] += 1
    return uses


def build(inputs, canonical=None):
    """Return {category: {variant: winner}}.

    ``canonical`` names the corpora whose spellings WIN. The base's label space is the
    one every warm-started model has to present, so a downstream corpus maps onto the
    base's form and never the reverse -- without this, cc_news_haiku45's 427k lowercase
    `location` outvotes the base and flips the winner to a spelling the base never learned.
    """
    import json
    if isinstance(inputs, str):
        inputs = [inputs]
    canon = (scan(inputs_to_files(canonical)) if canonical
             else {c: Counter() for c in CATEGORIES})
    uses = {c: Counter() for c in CATEGORIES}
    for path in inputs_to_files(inputs):
        for line in Path(path).open(encoding="utf-8"):
            for category, label in labels_by_category(json.loads(line)):
                uses[category][label] += 1

    maps = {}
    for category in CATEGORIES:
        clusters = defaultdict(list)
        for label in uses[category]:
            clusters[fold(label)].append(label)
        mapping = {}
        if category == "entities":
            # Canonical form first, so a later case-cluster winner cannot override it.
            for variant, winner in SYNONYMS.items():
                if variant in uses[category]:
                    mapping[variant] = winner
        for key, variants in clusters.items():
            # An empty fold means the label is pure punctuation: the POS tags ',', '.',
            # '-LRB-' and '-RRB-' are four distinct labels that all fold to "".
            if not key or len(variants) < 2 or key in EXCLUDE:
                continue
            # If any variant is pinned by SYNONYMS, that target wins the whole cluster.
            pinned = {SYNONYMS[v] for v in variants if v in SYNONYMS}
            in_base = [v for v in variants if v in canon[category]]
            if len(pinned) == 1:
                winner = pinned.pop()
            elif in_base:
                # The base's spelling wins, so warm-started models present its vocabulary.
                # Ranked on the BASE's own counts: using the global ones lets a downstream
                # corpus outvote the base among its own labels and flip the winner.
                winner = min(in_base, key=lambda v: style_rank(v, canon[category][v],
                                                               category == "structures"))
            else:
                winner = min(variants, key=lambda v: style_rank(v, uses[category][v],
                                                                category == "structures"))
            for variant in variants:
                if variant != winner:
                    mapping[variant] = winner
        # The map is applied ONCE, so a target that is itself a key leaves two spellings
        # alive: `LOC -> Location` beside `Location -> location` unifies nothing.
        unresolved = {k: v for k, v in mapping.items() if v in mapping}
        if unresolved:
            raise ValueError(
                f"{category}: {len(unresolved)} entries whose target is itself remapped, "
                f"e.g. {dict(list(unresolved.items())[:3])} -- the map is not closed")
        maps[category] = dict(sorted(mapping.items(), key=lambda kv: -uses[category][kv[0]]))
    return maps, uses


def main():
    argv = list(sys.argv[1:])
    out_path = canonical = None
    if "--out" in argv:
        i = argv.index("--out")
        out_path = argv[i + 1]
        del argv[i:i + 2]
    if "--canonical" in argv:
        i = argv.index("--canonical")
        canonical = [argv[i + 1]]
        del argv[i:i + 2]
    maps, uses = build(argv, canonical)
    for category, mapping in maps.items():
        moved = sum(uses[category][v] for v in mapping)
        print(f"\n# {category}: {len(uses[category])} distinct labels, "
              f"{len(mapping)} spellings collapse ({moved} uses moved)")
        if not mapping:
            print("  map: {}")
            continue
        for variant, winner in mapping.items():
            print(f"    {variant!r}: {winner!r}    # x{uses[category][variant]}")
    block = {"labels": {c: {"rollup": False, "separator": ".", "map": m}
                        for c, m in maps.items()}}
    if out_path:
        header = (
            "# Unified label space -- GENERATED by tools/train/build_label_maps.py.\n"
            "# Regenerate after any data change:\n"
            "#   uv run python tools/train/build_label_maps.py <configs/jsonl...> --out "
            f"{out_path}\n"
            "#\n"
            "# GLiNER2 takes the label as an INPUT string, so 'CompanyName' and 'Company Name'\n"
            "# are two queries for one field. Every model warm-started from a base must present\n"
            "# that base's label space, so configs reference this ONE file via `labels_file:`\n"
            "# rather than carrying copies that drift.\n"
            "#\n"
            "# Same-category clusters only. 'target' is excluded on purpose: docee's Target is\n"
            "# who an attack hit, bio_ner_relations' is a kinase substrate. Labels folding to an\n"
            "# empty string (the POS tags ',', '.', '-LRB-') are left alone.\n")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(
            header + yaml.safe_dump(block, allow_unicode=True, sort_keys=False,
                                    default_flow_style=False), encoding="utf-8")
        print(f"\nwrote {out_path}")
    else:
        print("\n--- YAML ---")
        print(yaml.safe_dump(block, allow_unicode=True, sort_keys=False,
                             default_flow_style=False))


if __name__ == "__main__":
    main()
