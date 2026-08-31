"""Translate a corpus's LABEL vocabulary to English, leaving the spans untouched.

GLiNER2 takes the label set as an INPUT at inference, so a model's labels must live in one
language regardless of the content language. The 137k base learned English labels -- even
on its Chinese corpora, where DuEE rows carry `event_type: "Exhibit"` with the trigger
`演示`. A corpus annotated with Chinese labels trains a DISJOINT label space: `人物` and
`person` become unrelated queries, and asking the model for `person` on a Chinese document
returns nothing.

This rewrites label KEYS only. Entity surfaces, triggers, argument entities, relation
heads and tails, and structure field VALUES are all left exactly as annotated -- they are
verbatim spans of Chinese text and must stay Chinese.

The closed entity vocabulary is hand-mapped rather than translated: those seven types
cover 99.8% of entity uses, and a translator returning "Persons"/"Person"/"People" across
different calls would fragment the dominant labels.

    uv run python tools/data/translate_labels.py --in <corpus.jsonl> --out <corpus_en.jsonl>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _split import dumps_record  # noqa: E402
from data.synthetic.providers import (  # noqa: E402
    REFUSAL_MARK, AnthropicProvider, ProviderConfig,
)

CJK = re.compile(r"[一-鿿]")
BATCH = 40

# Closed vocabulary from annotate_multitask.py's prompt: 99.8% of entity uses.
FIXED = {
    "人物": "Person", "组织": "Organization", "地点": "Location", "时间": "Date",
    "数量": "Quantity", "事件": "Event", "产品": "Product",
    # Common roles and relations, pinned for the same reason.
    "主体": "Subject", "客体": "Object", "参与": "Participant",
    "位于": "located_in", "任职于": "works_for", "隶属于": "part_of",
    "发生于": "occurred_at", "拥有": "owns",
}

SYSTEM = ("You translate short Chinese annotation labels into concise English labels for "
          "an information-extraction schema. Reply with a single JSON object and nothing else.")

USER = """Translate each Chinese label into a short English label suitable for an information-extraction schema.

Rules:
- Keep it SHORT: one to three words, no articles, no punctuation.
- Entity and structure-field labels: noun phrases, e.g. "Company Name", "Death Toll".
- Relation and role labels: snake_case verbs or roles, e.g. "works_for", "located_in", "Subject".
- Preserve the distinction between labels: do NOT map two different inputs to the same output.
- If a label is already English or a proper noun, return it unchanged.

Return {{"<chinese>": "<english>", ...}} covering EVERY input label.

LABELS:
{labels}"""


def collect(rows: list[dict]) -> set[str]:
    out: set[str] = set()
    for r in rows:
        o = r["output"]
        out |= set(o.get("entities") or {})
        for rel in o.get("relations") or []:
            out |= set(rel)
        for e in o.get("events") or []:
            out.add(e["event_type"])
            out |= {a["role"] for a in e.get("arguments") or []}
        for s in o.get("json_structures") or []:
            for name, fields in s.items():
                out.add(name)
                out |= set(fields)
    return {x for x in out if CJK.search(x)}


def apply_map(rows: list[dict], m: dict) -> tuple[list[dict], int, int]:
    """Rewrite label keys. Values -- the verbatim spans -- are never touched."""
    hit = miss = 0

    def tr(label: str) -> str:
        nonlocal hit, miss
        if label in m:
            hit += 1
            return m[label]
        miss += 1
        return label

    for r in rows:
        o = r["output"]
        if o.get("entities"):
            o["entities"] = {tr(k): v for k, v in o["entities"].items()}
        if o.get("relations"):
            o["relations"] = [{tr(k): v for k, v in rel.items()} for rel in o["relations"]]
        for e in o.get("events") or []:
            e["event_type"] = tr(e["event_type"])
            for a in e.get("arguments") or []:
                a["role"] = tr(a["role"])
        if o.get("json_structures"):
            new_structs, new_meta = [], {}
            for s in o["json_structures"]:
                for name, fields in s.items():
                    en = tr(name)
                    ef = {tr(f): v for f, v in fields.items()}
                    new_structs.append({en: ef})
                    old = (o.get("record_metadata") or {}).get(name, {})
                    anchor = old.get("anchor")
                    new_meta[en] = {"mode": old.get("mode", "natural"),
                                    "anchor": m.get(anchor, anchor) if anchor else
                                    next(iter(ef), None)}
            o["json_structures"] = new_structs
            o["record_metadata"] = new_meta
    return rows, hit, miss


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", nargs="+", required=True,
                    help="one or more corpora. ALL are translated with ONE shared map -- "
                         "a per-file map would give train and val different labels for "
                         "the same thing, which is worse than leaving them in Chinese")
    ap.add_argument("--out-suffix", default="",
                    help="write <stem><suffix>.jsonl; empty means overwrite in place")
    ap.add_argument("--map-out", default="data/chinese_gate/label_map_zh_en.json")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--model", default="claude-haiku-4-5")
    a = ap.parse_args()

    per_file = {f: [json.loads(l) for l in Path(f).open(encoding="utf-8")] for f in a.inp}
    labels = sorted(set().union(*(collect(r) for r in per_file.values())))
    todo = [x for x in labels if x not in FIXED]
    print(f"{len(labels):,} Chinese labels | {len(FIXED)} hand-pinned | {len(todo):,} to translate")

    provider = AnthropicProvider(ProviderConfig(
        provider="anthropic", model=a.model, max_tokens=4000, json_object=True))
    chunks = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    items = [(str(i), SYSTEM, USER.format(labels="\n".join(c))) for i, c in enumerate(chunks)]
    replies = provider.complete_batch(items) if a.batch else {
        cid: provider.complete(s, u) for cid, s, u in items}

    mapping = dict(FIXED)
    bad = 0
    for cid, raw in replies.items():
        if not raw or raw.startswith(REFUSAL_MARK):
            bad += 1
            continue
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            bad += 1
            continue
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            bad += 1
            continue
        for k, v in obj.items():
            v = str(v).strip()
            # Reject a "translation" that is still Chinese -- it would silently keep the
            # disjoint label space this whole pass exists to remove.
            if k in todo and v and not CJK.search(v):
                mapping[k] = v
    print(f"translated {len(mapping) - len(FIXED):,}/{len(todo):,} ({bad} unusable replies)")

    Path(a.map_out).write_text(json.dumps(mapping, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    total_hit = total_miss = 0
    for f, rows in per_file.items():
        rows, hit, miss = apply_map(rows, mapping)
        total_hit += hit
        total_miss += miss
        src = Path(f)
        out = src if not a.out_suffix else src.with_name(
            src.name.replace(".jsonl", f"{a.out_suffix}.jsonl"))
        out.write_text("".join(dumps_record(r) + "\n" for r in rows), encoding="utf-8")
        left = collect(rows)
        print(f"  {out.name:34s} rewritten {hit:6,} | zh left {len(left):3,}")
    print(f"label keys rewritten {total_hit:,} | untranslated {total_miss:,}")
    print(f"map -> {a.map_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
