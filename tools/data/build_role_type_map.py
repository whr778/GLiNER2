"""Derive `(event_type, role) -> allowed entity types` from corpora carrying BOTH golds.

Option 2 (see OPTION_2_TYPED_ROLE_CONSTRAINTS.md) constrains an event role edge to land only
on a span the model types compatibly. The constraint is worthless without a map saying WHICH
types a role admits, and that map must be DERIVED, never invented.

APPLIED PER ROLE, NOT GLOBALLY, and this tool is where that is enforced mechanically rather
than by judgement. The 800-document sweep in EVENT_ARGUMENT_DIAGNOSIS measured that the type
signal is real but role-dependent: `Location` and `Date` discriminate, `Subject` does not --
in military news the subject is an aircraft whether the binding is right or wrong. A role
whose entity-type distribution is FLAT carries no constraint, so it is omitted. **A map
containing every role has failed, not succeeded.**

ONLY TAXONOMY CORPORA DRIVE IT. `zh_multitask` is 70% singletons because the annotator
invented labels per document, and the synthetic sets are ~99.9% synthetic; feeding either to
a map manufactures agreement. Default sources are casie and wikievents, both human taxonomies
carrying entity AND event gold on the same documents.

    uv run python tools/data/build_role_type_map.py \
        --corpora data/casie data/wikievents --out tools/train/config/labels/role_types.json
"""

import argparse
import json
import yaml
from collections import Counter, defaultdict
from pathlib import Path


def surface_types(output: dict) -> dict:
    """surface -> the entity type(s) this document gives it."""
    out = defaultdict(set)
    for etype, surfaces in (output.get("entities") or {}).items():
        for s in surfaces or []:
            if isinstance(s, str) and s.strip():
                out[s.strip()].add(str(etype))
    return out


def _train_helpers():
    """Label-transform helpers, loaded by file location under a unique module name.

    Same approach as build_negative_pools: importing `tools/train/train.py` by package path
    has import side effects that break a full-suite run.
    """
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "train" / "train.py"
    spec = importlib.util.spec_from_file_location("_gliner2_roletype_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._category_fns, mod.load_labels_cfg, mod.transform_record


def _doc_key(rec: dict) -> str:
    return str(rec.get("input") or rec.get("text") or "").strip()


def entity_index(path: Path, limit: int, fns=None, transform=None) -> dict:
    """``document text -> {surface: {entity types}}`` from a corpus carrying entity gold.

    cmnee and duee carry event gold; their entity gold was BOUGHT separately and lives in
    `cmnee_ner` / `duee_ner`, keyed by the same documents. Joining on document text lifts
    option 2's reach from casie alone (35.7% of arguments) to ~99%.
    """
    index: dict = {}
    for split in ("train", "val"):
        f = path.with_name(f"{path.name}.{split}.jsonl")
        if not f.is_file():
            continue
        with f.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if limit and i >= limit:
                    break
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if fns:
                    rec = transform(rec, fns)
                key = _doc_key(rec)
                if key:
                    index[key] = surface_types(rec.get("output") or {})
    return index


def scan(paths: list[Path], limit: int, joins_by_corpus: dict = None,
         fns=None, transform=None) -> dict:
    """(event_type, role) -> Counter(entity_type), counted only where the SAME document
    types the argument's surface. An argument whose surface carries no entity gold in its
    own document contributes nothing -- inferring a type from elsewhere would be exactly
    the cross-document leap the within-dimension rule forbids."""
    tally: dict = defaultdict(Counter)
    seen = Counter()
    joins = Counter()
    for path in paths:
        joined = (joins_by_corpus or {}).get(path.name)
        for split in ("train", "val"):
            f = path.with_name(f"{path.name}.{split}.jsonl")
            if not f.is_file():
                continue
            with f.open(encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if limit and i >= limit:
                        break
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if fns:
                        rec = transform(rec, fns)
                    o = rec.get("output") or {}
                    if joined is not None:
                        # JOINED SOURCE: the entity gold lives in another corpus, keyed by
                        # document. A document that does not match contributes NOTHING --
                        # typing a role from a different document is exactly the
                        # cross-document leap this tool refuses everywhere else.
                        types = joined.get(_doc_key(rec))
                        joins[f"{path.name}:{'hit' if types else 'miss'}"] += 1
                        types = types or {}
                    else:
                        types = surface_types(o)
                    if not types:
                        continue
                    for ev in (o.get("events") or []):
                        et = str(ev.get("event_type") or "").strip()
                        for a in (ev.get("arguments") or []):
                            if not isinstance(a, dict):
                                continue
                            role = str(a.get("role") or "").strip()
                            ent = str(a.get("entity") or "").strip()
                            if not (et and role and ent):
                                continue
                            seen[(et, role)] += 1
                            for t in types.get(ent, ()):
                                tally[(et, role)][t] += 1
    return tally, seen, joins


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpora", nargs="+",
                    default=["data/casie", "data/wikievents"],
                    help="TAXONOMY corpora only -- synthetic and invented-label sets "
                         "manufacture agreement")
    ap.add_argument("--out", default="tools/train/config/labels/role_types.json")
    ap.add_argument("--min-support", type=int, default=20,
                    help="typed argument mentions a (event_type, role) needs to be trusted")
    ap.add_argument("--min-purity", type=float, default=0.80,
                    help="share the admitted types must cover. Below this the role is FLAT "
                         "and carries no constraint, so it is omitted -- that omission is "
                         "the point, not a shortfall")
    ap.add_argument("--min-dominance", type=float, default=0.80,
                    help="share the SINGLE most common type must carry. Purity alone is not "
                         "enough: with --max-types 3 a role spread evenly over three types "
                         "covers 100%% and passes, which is how `Victim` (Person 41 / Org 35 "
                         "/ System 12) slipped through on the first run. A constraint that "
                         "admits three heterogeneous types constrains nothing.")
    ap.add_argument("--max-types", type=int, default=2,
                    help="a role admitting many types is not a constraint")
    ap.add_argument("--min-type-share", type=float, default=0.05,
                    help="drop an admitted type carrying less than this. A tail type at 0-1%% "
                         "is noise and admitting it only widens the constraint: casie's "
                         "`Patch` is Patch 100%% / System 0%%, and letting System in makes the "
                         "rule allow a span it should refuse.")
    ap.add_argument("--labels-config",
                    help="a training config whose labels_file unifies the entity vocabulary. "
                         "WITHOUT IT, MIXING CORPORA IS A BUG: casie types `Person` where "
                         "wikievents types `PER`, so a role tallied across both looks flat "
                         "when each corpus alone is clean -- `Vulnerability` read 99%% in "
                         "casie and three-way split once wikievents joined. Same reason "
                         "build_negative_pools derives pools AFTER label unification.")
    ap.add_argument("--join", nargs="*", default=[], metavar="EVENTS:ENTITIES",
                    help="pair an event corpus with a corpus carrying its entity gold, e.g. "
                         "`data/cmnee:data/cmnee_ner`. cmnee and duee carry no entity gold of "
                         "their own -- it was bought separately and keyed to the same "
                         "documents. Joining on document text lifts this map's reach from "
                         "casie alone (35.7%% of arguments) to ~99%%. A document that fails to "
                         "match contributes NOTHING.")
    ap.add_argument("--min-join-rate", type=float, default=0.90,
                    help="refuse a join that matches fewer than this share of documents -- a "
                         "silent 5%% join would build the map from a biased remnant")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if len(args.corpora) > 1 and not args.labels_config:
        raise SystemExit(
            "[roletypes] REFUSING: more than one corpus without --labels-config. Entity "
            "vocabularies differ across corpora (casie `Person` vs wikievents `PER`), so an "
            "un-unified tally makes a clean role look flat. Pass --labels-config, or build "
            "one corpus at a time."
        )
    fns, transform = None, None
    if args.labels_config:
        _category_fns, load_labels_cfg, transform_record = _train_helpers()
        cfg_path = Path(args.labels_config)
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        fns = _category_fns(load_labels_cfg(cfg, cfg_path))
        transform = transform_record
        # The flag USED to gate the multi-corpus refusal and never be applied, so a map built
        # "with unification" was built without it.
        print(f"[roletypes] label transforms active for: {sorted(fns) or 'NONE'}")

    joins_by_corpus = {}
    for pair in args.join:
        ev, _, ent = pair.partition(":")
        if not (ev and ent):
            raise SystemExit(f"[roletypes] --join wants EVENTS:ENTITIES, got {pair!r}")
        idx = entity_index(Path(ent), args.limit, fns, transform)
        if not idx:
            raise SystemExit(f"[roletypes] --join {pair}: no entity documents read from {ent}")
        joins_by_corpus[Path(ev).name] = idx
        print(f"[roletypes] join {Path(ev).name} <- {Path(ent).name}: {len(idx):,} documents indexed")
        if Path(ev).as_posix() not in [Path(c).as_posix() for c in args.corpora]:
            raise SystemExit(f"[roletypes] --join names {ev}, which is not in --corpora")

    tally, seen, joins = scan([Path(p) for p in args.corpora], args.limit,
                              joins_by_corpus, fns, transform)

    for name in sorted(joins_by_corpus):
        hit, miss = joins[f"{name}:hit"], joins[f"{name}:miss"]
        rate = hit / max(hit + miss, 1)
        print(f"[roletypes] join {name}: {hit:,} matched, {miss:,} unmatched ({rate:.1%})")
        if rate < args.min_join_rate:
            raise SystemExit(
                f"[roletypes] REFUSING: {name} joined only {rate:.1%} of its documents "
                f"(floor {args.min_join_rate:.0%}). A map built from the matched remnant is "
                f"a map of whatever happened to match.")
    emitted, omitted = {}, []
    for key, counter in sorted(tally.items(), key=lambda kv: -sum(kv[1].values())):
        et, role = key
        total = sum(counter.values())
        if total < args.min_support:
            omitted.append((et, role, total, "support", ""))
            continue
        dominance = counter.most_common(1)[0][1] / total
        if dominance < args.min_dominance:
            top = ", ".join(f"{t} {n/total:.0%}" for t, n in counter.most_common(3))
            omitted.append((et, role, total, f"flat, top only {dominance:.0%}", top))
            continue
        chosen, covered = [], 0
        for t, n in counter.most_common(args.max_types):
            if n / total < args.min_type_share:
                continue
            chosen.append(t); covered += n
        purity = covered / total
        if purity < args.min_purity:
            top = ", ".join(f"{t} {n/total:.0%}" for t, n in counter.most_common(3))
            omitted.append((et, role, total, f"tail too heavy {purity:.0%}", top))
            continue
        emitted.setdefault(et, {})[role] = sorted(chosen)

    n_roles = sum(len(v) for v in emitted.values())
    print(f"[roletypes] {len(seen)} (event_type, role) pairs seen")
    print(f"[roletypes] EMITTED {n_roles} across {len(emitted)} event types")
    print(f"[roletypes] OMITTED {len(omitted)}")
    for et, role, total, why, detail in omitted[:14]:
        print(f"    {et}/{role:22s} n={total:5d}  {why}{'  ' + detail if detail else ''}")

    # THE GATE. A map that admits every role is not a constraint, and a map that admits none
    # cannot be applied. Both are failures and neither announces itself.
    if not emitted:
        raise SystemExit("[roletypes] REFUSING: nothing emitted -- the map would constrain "
                         "nothing")
    if not omitted:
        raise SystemExit("[roletypes] REFUSING: every role passed. A role whose types are "
                         "FLAT carries no constraint, and a map that omits none has not "
                         "discriminated -- check --min-purity")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "role_types": emitted},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[roletypes] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
