"""Unify DocEE-zh into DocEE-en's label space, then hand it to convert_docee.py.

226 MB of Chinese document-level event data has sat in `data/DocEE/DocEE-zh/` unconverted:
`convert_docee.py` has no Chinese path, and the built corpus is 53/20,000 rows with any
CJK -- incidental names in English articles, not Chinese documents. This closes that.

A PREPROCESSOR, not a second converter. It emits the 4-element
`[title, text, event_type, annotations]` shape `convert_docee._normalise_record` already
accepts, so the tested conversion, stratification and split logic is reused rather than
duplicated. Two upstream shapes block that directly:

  * records are wrapped one-per-list -- `[[{...}], [{...}], ...]` -- and a 1-element list
    fails the `len(raw) >= 4` check, returning None for every record;
  * the key is `event_argument` SINGULAR, while `ANNOTATIONS_KEYS` carries
    `event_arguments` plural, so even unwrapped the arguments come back empty and the
    corpus converts to classification-only WITHOUT ERROR.

## Two label spaces, both needing work, and they are NOT the same job

**Event types (58 zh vs 59 en).** 32 match exactly; 26 do not, covering 19,773 records --
54% of the corpus. Left alone, more than half the Chinese data trains against label strings
the model is never shown at inference. Several are upstream misspellings (`Earthqueakes`,
`Volcano Eruptio`, `Famous Person - Sicks`), several are case (`air crash`,
`Gas explosion`), and twelve are genuine renamings where the zh arm uses a different
hierarchy (`Government Job change - Election` -> `Election`; `Famous Person - CommitCrime -
Arrest` -> `CommitCrime - Arrest`).

EN's spelling is canonical even where EN is itself wrong -- `Military Excercise`,
`Join in an Orgnization` -- because the base's spelling wins and a downstream corpus must
never outvote it.

**Argument roles (344 zh, ALL CJK, vs 356 en, all latin).** Disjoint by language, 216,496
instances. This is the trap the label rule names: an English event-type MENU sitting above
Chinese role ANSWERS. Full unification means proving 344 mappings by reading surfaces, and
that is a separate job.

Only the FOUR roles stage 1 consumes are mapped here, each verified by reading what it
actually tags rather than by string similarity:

    人员伤亡  "造成2人死亡、1人受伤" / "至少19人死亡，406受伤"  -> Casualties and Losses
    地点      "美国威斯康星州基诺沙" / "巴黎"                    -> Location
    日期      "10月10日晚上" / "当地时间10日上午"                -> Date
    起因于    "为了帮助乌克兰..." / "人道主义援助"                -> Cause

`人员伤亡` is the same multi-fact string shape as the English role -- the string the
casualty model binds numbers from -- so no reconstruction is needed.

Unmapped roles are DROPPED by default, not passed through under Chinese names: an entities
head trained on 340 unverified Chinese labels would violate the one-concept-one-spelling
rule wholesale, and the raw data is not lost -- it stays in the source file. `--roles all`
keeps them for anyone doing the full 344-role unification later.

    uv run python tools/data/prepare_docee_zh.py --out /tmp/docee_zh_prepared.json
    uv run python tools/data/convert_docee.py --input /tmp/docee_zh_prepared.json \
        --out data/docee_zh.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# zh spelling -> en canonical. Built by diffing the two vocabularies; every target is a
# verbatim member of DocEE-en's 59, asserted at runtime below.
EVENT_TYPE_MAP = {
    # upstream misspellings / truncations
    "Earthqueakes": "Earthquakes",
    "Volcano Eruptio": "Volcano Eruption",
    "Famous Person - Sicks": "Famous Person - Sick",
    "Military Excerciser": "Military Excercise",
    # case only
    "air crash": "Air Crash",
    "Gas explosion": "Gas Explosion",
    "Train collisions": "Train Collisions",
    "Awards ceremony": "Awards Ceremony",
    "Break historical records": "Break Historical Records",
    "New achievements in aerospace": "New Achievements in Aerospace",
    "New wonders in nature": "New Wonders in Nature",
    "New archeological discoveries": "New Archeological Discoveries",
    "Famous Person - Give a speech": "Famous Person - Give a Speech",
    "Famous Person - divorce": "Famous Person - Divorce",
    # renamings: the zh arm uses a deeper hierarchy than en
    "sport competition": "Sports Competition",
    "Hurricanes-Tornado-Storm-Blizzard": "Storm",
    "Protest or Online Condemnation": "Protest",
    "Diplomatic Talks - Diplomatic  Negotiation - Summit Meetin": "Diplomatic Talks",
    "Government Job change - Appoint or Inauguration": "Appoint_Inauguration",
    "Government Job change - Resignation or Dismissal": "Resignation_Dismissal",
    "Government Job change - Election": "Election",
    "Famous Person - CommitCrime - Arrest": "CommitCrime - Arrest",
    "Famous Person - CommitCrime - Investigate": "CommitCrime - Investigate",
    "Famous Person - CommitCrime - Release": "CommitCrime - Release",
    "Famous Person - Sentence": "CommitCrime - Sentence",
    "Famous Person - Accuse": "CommitCrime - Accuse",
}

# Verified by reading surfaces, not by string similarity. See the module docstring.
ROLE_MAP = {
    "人员伤亡": "Casualties and Losses",
    "地点": "Location",
    "日期": "Date",
    "起因于": "Cause",
}


def english_label_space(path: Path) -> set[str]:
    """DocEE-en's 59, read from the converted corpus -- the canonical spelling."""
    out: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            for c in (json.loads(line).get("output") or {}).get("classifications") or []:
                out.update(str(x) for x in (c.get("labels") or []))
            if len(out) >= 59:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path,
                    default=Path("data/DocEE/DocEE-zh/DocEE-zh-20230105.json"))
    ap.add_argument("--en-corpus", type=Path, default=Path("data/docee.train.jsonl"),
                    help="source of the canonical 59-label spelling")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--roles", choices=("mapped", "all"), default="mapped",
                    help="mapped: only the 4 surface-verified stage-1 roles. all: keep "
                         "the other ~340 under their Chinese names -- for the full "
                         "unification job, NOT for training.")
    args = ap.parse_args()

    canon = english_label_space(args.en_corpus)
    print(f"[zh] English label space: {len(canon)} types")
    bad = {k: v for k, v in EVENT_TYPE_MAP.items() if v not in canon}
    if bad:
        raise SystemExit(f"map targets absent from the English space: {bad}")
    # The map must be CLOSED: no target may itself be a key, or applying it once leaves
    # both spellings alive.
    collide = set(EVENT_TYPE_MAP.values()) & set(EVENT_TYPE_MAP)
    if collide:
        raise SystemExit(f"map is not closed, target is also a key: {collide}")
    print(f"[zh] event-type map: {len(EVENT_TYPE_MAP)} entries, all targets canonical, closed")

    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows, mapped, unmapped_types, kept_roles, dropped_roles = [], Counter(), Counter(), Counter(), Counter()
    for group in data:
        for rec in (group if isinstance(group, list) else [group]):
            et = (rec.get("event_type") or "").strip()
            if et in EVENT_TYPE_MAP:
                mapped[et] += 1
                et = EVENT_TYPE_MAP[et]
            elif et not in canon:
                unmapped_types[et] += 1
                continue                      # never emit a label the model won't be shown
            raw = rec.get("event_argument")
            try:
                args_list = json.loads(raw) if isinstance(raw, str) else (raw or [])
            except json.JSONDecodeError:
                args_list = []
            anns = []
            for a in args_list:
                if not isinstance(a, dict) or not a.get("type"):
                    continue
                role = a["type"]
                if role in ROLE_MAP:
                    kept_roles[role] += 1
                    anns.append({**a, "type": ROLE_MAP[role]})
                elif args.roles == "all":
                    kept_roles[role] += 1
                    anns.append(a)
                else:
                    dropped_roles[role] += 1
            rows.append([rec.get("title"), rec.get("content"), et, anns])

    args.out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"[zh] wrote {args.out}: {len(rows):,} records")
    print(f"[zh] event types remapped: {sum(mapped.values()):,} records over {len(mapped)} spellings")
    if unmapped_types:
        print(f"[zh] DROPPED {sum(unmapped_types.values()):,} records with unmappable types: "
              f"{dict(unmapped_types.most_common(5))}")
    print(f"[zh] argument roles kept: {sum(kept_roles.values()):,} over {len(kept_roles)} roles")
    if dropped_roles:
        print(f"[zh] roles dropped (unverified, --roles all keeps them): "
              f"{sum(dropped_roles.values()):,} over {len(dropped_roles)} roles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
