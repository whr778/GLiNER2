"""Annotate Chinese news for ALL FIVE GLiNER2 tasks in one pass, with Haiku.

Entities, relations, events, classifications and structures are asked for in a SINGLE
call per document rather than five. The document is then read once instead of five times,
and since input dominates the token bill for 1,200-token articles that is the difference
between $48.34 and $30.22 for 6,000 documents. It also mirrors how a human annotator
works: read the article once, mark everything.

THE HARD CONSTRAINT, same as annotate_casualty.py: every surface must be a VERBATIM
substring of the article. The boundary head locates spans in the text, so a paraphrased or
normalised surface produces a row that trains nothing and reports no error. Values failing
it are dropped here and the rate printed -- a silent drop would let a bad prompt spend the
whole batch.

    smoke  : uv run python tools/data/annotate_multitask.py --corpora <f> --limit 10 --out data/zh_mt_smoke
    batch  : uv run python tools/data/annotate_multitask.py --corpora <f> --batch --out data/zh_multitask
    recover: uv run python tools/data/annotate_multitask.py --fetch-batch msgbatch_... --corpora <f> --out <same>
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
from _split import dumps_record, normalize_group_key  # noqa: E402
from data.synthetic.providers import (  # noqa: E402
    REFUSAL_MARK, AnthropicProvider, ProviderConfig,
)

MAX_CHARS = 4000

# One fixed task for classification. A SECOND classification task collapses the first to
# a single label at confidence 1.0 on boundary models -- measured on the gate, where it
# cost two runs' verdicts -- so the schema this data trains carries exactly one.
#
# LABELS ARE ENGLISH, SPANS ARE CHINESE. GLiNER2 takes the label set as an INPUT at
# inference, so labels must live in ONE language whatever the content language is. The
# 137k base learned English labels even on its Chinese corpora -- DuEE rows carry
# event_type "Exhibit" with the trigger 演示. A first run of this annotator asked for
# Chinese labels and produced a DISJOINT label space, where `人物` and `person` are
# unrelated queries and asking the model for `person` on a Chinese document returns
# nothing. Recovering it cost a translation pass over 24,826 labels.
# Broadened after a smoke run returned 教育, which was not in the list and was silently
# dropped -- a label set narrower than the corpus discards valid annotations.
TOPIC_LABELS = ["Politics", "Economy", "Society", "International", "Technology",
                "Sports", "Culture", "Health", "Law", "Disaster", "Education",
                "Military", "Environment", "Entertainment", "Transport", "Other"]

SYSTEM = (
    "You annotate Chinese news articles for an information-extraction dataset. "
    "Every span you output must be copied EXACTLY from the article, character for "
    "character. Reply with a single JSON object and nothing else."
)

USER = """Annotate this Chinese news article for five tasks at once.

TWO LANGUAGES, DELIBERATELY. Every LABEL you invent — entity types, relation names, event types, roles, structure and field names — must be in ENGLISH. Every SPAN you copy out of the article — entity surfaces, triggers, argument entities, relation heads and tails, field values — must stay in the article's own Chinese, character for character. The labels are the schema; the spans are the text.

CRITICAL RULE — copy spans verbatim. Every entity, trigger, argument, head, tail and field value must appear in the article EXACTLY as you write it. Do not translate, normalise, reformat, or add words. If you cannot find an exact span, omit that item.

1. "entities" — a map of ENGLISH entity type to the list of Chinese surfaces of that type. Use these types: Person, Organization, Location, Date, Quantity, Event, Product. Omit a type with no instances.

2. "relations" — a list of objects, each with exactly three keys: "relation", "head", "tail". "relation" is an ENGLISH snake_case name such as works_for, located_in, part_of, participated_in, occurred_at, owns. Head and tail are verbatim Chinese spans.
   Example: [{{"relation": "located_in", "head": "飞来峡水利枢纽", "tail": "清远"}}]

3. "events" — a list of {{"event_type": "...", "triggers": ["..."], "arguments": [{{"role": "...", "entity": "..."}}]}}. "event_type" and "role" are ENGLISH (e.g. "Earthquake", and roles Subject, Object, Date, Location, Quantity). The trigger and entity are verbatim Chinese spans.
   Example: {{"event_type": "Marital Status", "triggers": ["结婚"], "arguments": [{{"role": "Subject", "entity": "司马相如"}}]}}

4. "topic" — exactly one label from this list, choosing the CLOSEST if none is exact: {labels}

5. "structures" — a list of records, each {{"record_name": {{"field": "value"}}}}. The record name and every field name are ENGLISH; the values are verbatim Chinese spans. Emit a record ONLY when the article states a compact set of facts that belong together (a company report, a person profile, an incident). If nothing fits, use an empty list.
   Example: [{{"Dam": {{"name": "飞来峡水利枢纽", "location": "清远"}}}}]

Reply as:
{{"entities": {{"人物": ["..."]}}, "relations": [{{"relation": "...", "head": "...", "tail": "..."}}], "events": [], "topic": "...", "structures": []}}

ARTICLE:
{text}"""


def load_candidates(paths: list[Path], limit: int, exclude: set[str]) -> list[dict]:
    out, seen = [], set(exclude)
    for path in paths:
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            text = (rec.get("input") or rec.get("text") or "")[:MAX_CHARS]
            if not text.strip():
                continue
            key = normalize_group_key(text)[:300]
            if key in seen:
                continue
            seen.add(key)
            out.append({"text": text, "source": rec.get("source") or path.stem})
            if limit and len(out) >= limit:
                return out
    return out


def parse(raw: str) -> dict | None:
    if not raw or raw.startswith(REFUSAL_MARK):
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def clean(obj: dict, text: str) -> tuple[dict, int, int]:
    """Keep only VERBATIM spans. Returns (output, kept, dropped)."""
    kept = dropped = 0
    out: dict = {}

    ents = {}
    for etype, surfaces in (obj.get("entities") or {}).items():
        if not isinstance(surfaces, list):
            continue
        good = []
        for s in surfaces:
            s = str(s).strip()
            if not s:
                continue
            if s in text:
                good.append(s); kept += 1
            else:
                dropped += 1
        if good:
            ents[str(etype)] = sorted(set(good), key=good.index)
    if ents:
        out["entities"] = ents

    rels = []
    for r in (obj.get("relations") or []):
        if not isinstance(r, dict):
            continue
        pairs = []
        if "head" in r and "tail" in r:                    # flat {relation, head, tail}
            name = str(r.get("relation") or r.get("relation_name") or "").strip()
            pairs.append((name, r.get("head"), r.get("tail")))
        else:                                              # nested {name: {head, tail}}
            for name, ht in r.items():
                if isinstance(ht, dict):
                    pairs.append((str(name), ht.get("head"), ht.get("tail")))
        for name, h, t2 in pairs:
            h, t2 = str(h or "").strip(), str(t2 or "").strip()
            if name and h and t2 and h in text and t2 in text:
                rels.append({name: {"head": h, "tail": t2}}); kept += 2
            else:
                dropped += 2
    if rels:
        out["relations"] = rels

    evs = []
    for e in (obj.get("events") or []):
        if not isinstance(e, dict):
            continue
        etype = str(e.get("event_type") or "").strip()
        trigs = [str(t).strip() for t in (e.get("triggers") or []) if str(t).strip() in text]
        if not etype or not trigs:
            dropped += 1
            continue
        kept += len(trigs)
        args = []
        for a in (e.get("arguments") or []):
            if not isinstance(a, dict):
                continue
            role, ent = str(a.get("role") or "").strip(), str(a.get("entity") or "").strip()
            if role and ent and ent in text:
                args.append({"role": role, "entity": ent}); kept += 1
            else:
                dropped += 1
        evs.append({"event_type": etype, "triggers": trigs, "arguments": args})
    if evs:
        out["events"] = evs

    topic = str(obj.get("topic") or "").strip()
    if topic in TOPIC_LABELS:
        out["classifications"] = [{"task": "zh_topic", "labels": TOPIC_LABELS,
                                   "true_label": topic}]
        kept += 1

    structs, meta = [], {}
    for s in (obj.get("structures") or []):
        if not isinstance(s, dict):
            continue
        for name, fields in s.items():
            if not isinstance(fields, dict):
                continue
            good = {}
            for f, v in fields.items():
                v = str(v).strip()
                if v and v in text:
                    good[str(f)] = v; kept += 1
                else:
                    dropped += 1
            if good:
                structs.append({str(name): good})
                # The BOUNDARY record head cannot decode a structure without metadata,
                # and the failure is silent. Anchor on a field the record actually has.
                meta.setdefault(str(name), {"mode": "natural", "anchor": next(iter(good))})
    if structs:
        out["json_structures"] = structs
        out["record_metadata"] = meta
    return out, kept, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="+", required=True)
    ap.add_argument("--out", default="data/zh_multitask")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", action="store_true", help="Batch API, -50%% pricing")
    ap.add_argument("--fetch-batch")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--exclude", nargs="+", default=[])
    args = ap.parse_args()

    done: set = set()
    for p in args.exclude:
        for line in Path(p).open(encoding="utf-8"):
            done.add(normalize_group_key(json.loads(line)["input"])[:300])
    cands = load_candidates([Path(p) for p in args.corpora], args.limit, done)
    print(f"[mt-ann] {len(cands)} candidates")

    provider = AnthropicProvider(ProviderConfig(
        provider="anthropic", model=args.model, max_tokens=3000, json_object=True))
    items = [(str(i), SYSTEM, USER.format(text=c["text"], labels=", ".join(TOPIC_LABELS)))
             for i, c in enumerate(cands)]

    if args.fetch_batch:
        replies = provider.fetch_batch(args.fetch_batch)
    elif args.batch:
        replies = provider.complete_batch(items)
    else:
        replies = {cid: provider.complete(s, u) for cid, s, u in items}

    rows, unusable, empty, kept, dropped = [], 0, 0, 0, 0
    tasks: Counter = Counter()
    for cid, raw in replies.items():
        obj = parse(raw)
        if obj is None:
            unusable += 1
            continue
        text = cands[int(cid)]["text"]
        out, k, d = clean(obj, text)
        kept, dropped = kept + k, dropped + d
        if not out:
            empty += 1
            continue
        for key in out:
            tasks[key] += 1
        rows.append({"input": text, "output": out})

    path = Path(f"{args.out}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(dumps_record(r) + "\n")
    total = kept + dropped
    print(f"[mt-ann] {len(rows)} rows, {empty} empty, {unusable} unusable")
    if total:
        print(f"[mt-ann]   VERBATIM spans kept {kept}/{total} = {kept/total:.1%}, dropped {dropped}")
    print(f"[mt-ann]   task coverage: {dict(tasks)}")
    print(f"[mt-ann] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
