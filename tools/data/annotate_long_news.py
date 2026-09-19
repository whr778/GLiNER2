"""Annotate LONG English news for all five GLiNER2 tasks in one pass, with Haiku.

Sibling of `annotate_multitask.py` (which is Chinese-only) and built to the same contract:
the shared provider, `annotation.rules()` from GUIDELINES.md as the prompt's rule source,
`_split.dumps_record` on the write path, and a `clean()` that keeps only VERBATIM spans.

WHY A SEPARATE ANNOTATOR. Every existing annotator truncates its input --
`MAX_CHARS` is 4,000 in `annotate_multitask.py` and 6,000 in the casualty, gate and
event-entity ones -- which is correct for their corpora and fatal here. `cc_news_long` is 764
documents of 28,014 characters or more, median 9,274 TOKENS, max 38,412, harvested precisely
because nothing in the existing corpora reaches the model's sliding window (it engages past
4,096 subwords; the old collector capped at 12,000 characters). Annotating the first 6,000
characters of a 40,000-character article would defeat the entire point.

LABELS MATCH `cc_news_haiku45`, NOT A NEW INVENTION. Labels are an INPUT to GLiNER2, so one
concept must have one spelling across every corpus a model trains on. The seeds below and the
twelve classification tasks were read out of the shipped corpus rather than chosen: entity
types lowercase (`person`, `job title`), relations snake_case (`works_for`), event types
dotted (`Conflict.Attack`), structure records snake_case (`incident_report`).

    # submit (batch is -50%)
    uv run python tools/data/annotate_long_news.py \
        --corpora data/cc_news_long.pool.jsonl --out data/cc_news_long --batch
    # later, with the id it printed
    uv run python tools/data/annotate_long_news.py \
        --corpora data/cc_news_long.pool.jsonl --out data/cc_news_long --fetch-batch msgbatch_...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from _split import dumps_record, normalize_group_key  # noqa: E402
from annotation import rules  # noqa: E402
from data.synthetic.providers import AnthropicProvider, ProviderConfig  # noqa: E402

REFUSAL_MARK = "__REFUSAL__"

# NO TRUNCATION BY DEFAULT. 0 means send the whole document; the point of this corpus is the
# part every other annotator would have cut off.
MAX_CHARS = 0

# Read out of data/cc_news_haiku45.train.jsonl, not invented. Seeds, not a closed set: the
# shipped corpus carries 125 entity types and 66 relations because the prompt was open, and a
# closed list here would silently fork the label space of a corpus meant to sit beside it.
ENTITY_SEEDS = ("person", "location", "date", "organization", "job title", "media outlet",
                "cardinal", "money", "geopolitical entity", "company", "facility", "age",
                "occupation", "time")
RELATION_SEEDS = ("works_for", "member_of", "located_in", "based_in", "affiliated_with",
                  "employer_of", "part_of", "leader_of", "colleague_of", "participated_in",
                  "operates_in", "headquartered_in")
EVENT_SEEDS = ("Conflict.Attack", "Life.Die", "Life.Injure", "Contact.Meet",
               "Contact.Communicate", "Justice.Arrest", "Justice.TrialHearing",
               "Personnel.StartPosition", "Business.LaunchProduct", "Sport.Compete",
               "Disaster.NaturalDisaster", "Disaster.Accident")
STRUCTURE_SEEDS = ("person_profile", "event_listing", "incident_report", "company_profile",
                   "product", "transaction", "legal_case", "clinical_finding")

# CLOSED, unlike the four above: a classification task's label set IS its ontology, and a
# drifting one makes the task unscoreable across corpora.
CLASSIFICATION_TASKS = {
    "actionability": ["advisory", "informational", "requires action"],
    "audience": ["expert", "general public", "internal", "regulatory"],
    "certainty": ["asserted", "denied", "hedged", "speculative"],
    "document_genre": ["advertisement", "announcement", "correspondence", "legal filing",
                       "news report", "opinion", "review", "technical documentation"],
    "formality": ["formal", "informal"],
    "language_register": ["colloquial", "legal", "plain", "technical"],
    "risk_level": ["high", "low", "moderate", "none"],
    "sentiment": ["negative", "neutral", "positive"],
    "subjectivity": ["objective", "subjective"],
    "temporal_orientation": ["future", "past", "present", "timeless"],
    "topic": ["agriculture", "business", "defense", "education", "energy", "entertainment",
              "environment", "finance", "health", "legal", "politics", "science", "sports",
              "technology", "travel", "world"],
    "urgency": ["critical", "elevated", "routine", "urgent"],
}

SYSTEM = (
    "You annotate long English news articles for an information-extraction dataset."
) + " " + rules("json_only", "verbatim", "no_inference", "minority", "ambiguity",
                "uncertain_field")

USER = """Annotate this English news article for five tasks at once.

THE ARTICLE IS LONG AND YOU MUST COVER ALL OF IT. Annotate from the first paragraph to the
last, not only the opening. Coverage of the later sections matters as much as the lead.

CRITICAL RULE — copy spans verbatim. Every entity surface, trigger, argument, relation head
and tail, and structure field value must appear in the article EXACTLY as you write it,
character for character. Do not translate, normalise, reformat, correct spelling, or expand
abbreviations. If you cannot find an exact span, omit that item.

1. "entities" — a map of entity type to the list of surfaces of that type. Types are
   lowercase. Use these where they fit and add others in the same style when they do not:
   {entity_seeds}

2. "relations" — a list of {{"relation": "...", "head": "...", "tail": "..."}}. The relation
   name is snake_case; head and tail are verbatim spans. Common ones: {relation_seeds}

3. "events" — a list of {{"event_type": "...", "triggers": ["..."], "arguments":
   [{{"role": "...", "entity": "..."}}]}}. Event types are dotted, e.g. {event_seeds}.
   Roles are capitalised words such as Agent, Victim, Place, Time, Instrument.

4. "classifications" — an object mapping EACH task below to exactly one of its labels.
   Answer every task. The label sets are closed; do not invent labels.
{classification_block}

5. "structures" — a list of records, each {{"record_name": {{"field": "value"}}}}. Record and
   field names are snake_case; values are verbatim spans. Emit a record only when the article
   states a compact set of facts that belong together. Common records: {structure_seeds}

Reply as one JSON object:
{{"entities": {{}}, "relations": [], "events": [], "classifications": {{}}, "structures": []}}

ARTICLE:
{text}"""


def load_candidates(paths: list[Path], limit: int, max_chars: int) -> list[dict]:
    out, seen = [], set()
    for path in paths:
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("input") or rec.get("text") or ""
            if max_chars:
                text = text[:max_chars]
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
    return _parse_why(raw)[0]


def _parse_why(raw: str) -> tuple[dict | None, str]:
    """Parse, and SAY WHY when it fails.

    An unusable reply was an anonymous counter until 2026-09-19, when a 10-document pilot
    returned 2 of them and there was no way to tell a refusal from a truncated JSON -- which
    have opposite fixes. A long document's annotation is the case most likely to hit the
    output cap, so the reason has to be visible before a 743-document batch is bought.
    """
    if not raw:
        return None, "empty"
    if raw.startswith(REFUSAL_MARK):
        return None, "refusal"
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        # An opening brace with no close is the signature of hitting max_tokens.
        return None, "truncated_no_close" if "{" in raw else "no_json"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return None, f"bad_json:{str(exc)[:40]}"
    return (obj, "ok") if isinstance(obj, dict) else (None, "not_an_object")


def clean(obj: dict, text: str) -> tuple[dict, int, int]:
    """Keep only VERBATIM spans and CLOSED classification labels. -> (output, kept, dropped)."""
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
        name = str(r.get("relation") or "").strip()
        h, t = str(r.get("head") or "").strip(), str(r.get("tail") or "").strip()
        if name and h and t and h in text and t in text:
            rels.append({name: {"head": h, "tail": t}}); kept += 2
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

    cls = []
    raw_cls = obj.get("classifications") or {}
    if isinstance(raw_cls, dict):
        for task, labels in CLASSIFICATION_TASKS.items():
            answer = str(raw_cls.get(task) or "").strip()
            # A label outside the closed set is DROPPED, not coerced: a task whose ontology
            # drifts silently is unscoreable against the corpus it sits beside.
            if answer in labels:
                cls.append({"task": task, "labels": labels, "true_label": answer}); kept += 1
            elif answer:
                dropped += 1
    if cls:
        out["classifications"] = cls

    structs = []
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
    if structs:
        out["json_structures"] = structs

    return out, kept, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpora", nargs="+", required=True)
    ap.add_argument("--out", default="data/cc_news_long")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS,
                    help="0 = send the whole document, which is the point of this corpus")
    ap.add_argument("--batch", action="store_true", help="Batch API, -50%% pricing")
    ap.add_argument("--fetch-batch")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--max-tokens", type=int, default=32000,
                    help="OUTPUT budget. A truncated reply is unparseable JSON -- a silently "
                         "lost document that was still paid for. MEASURED 2026-09-19: at "
                         "16000 the two LONGEST documents of a 10-doc pilot (77,891 and "
                         "71,035 input chars) both truncated, their replies landing at 48,450 "
                         "and 46,777 chars. JSON runs about 3 chars/token, not prose's 4.08, "
                         "so 16000 tokens IS ~48,000 characters -- they hit the cap exactly. "
                         "32000 clears the longest document in this corpus with margin.")
    ap.add_argument("--batch-id-file",
                    default="/Volumes/Development/tmp/rescore/long_batch_id.txt",
                    help="Where to persist the batch id. Printing it is NOT enough: a poller "
                         "killed before its stdout is read leaves a PAID batch whose id "
                         "survives only in the API's list, and resubmitting pays twice. This "
                         "path is on the volume that survives a reboot.")
    ap.add_argument("--dry-run", action="store_true", help="price it, send nothing")
    args = ap.parse_args()

    cands = load_candidates([Path(p) for p in args.corpora], args.limit, args.max_chars)
    chars = sum(len(c["text"]) for c in cands)
    print(f"[long-ann] {len(cands)} documents, {chars:,} characters "
          f"(~{chars / 4.08:,.0f} tokens at this corpus's measured 4.08 chars/token)")

    block = "\n".join(f"   - {t}: {', '.join(v)}" for t, v in CLASSIFICATION_TASKS.items())
    items = [(str(i), SYSTEM, USER.format(
        text=c["text"],
        entity_seeds=", ".join(ENTITY_SEEDS),
        relation_seeds=", ".join(RELATION_SEEDS),
        event_seeds=", ".join(EVENT_SEEDS),
        structure_seeds=", ".join(STRUCTURE_SEEDS),
        classification_block=block,
    )) for i, c in enumerate(cands)]

    if args.dry_run:
        prompt_chars = sum(len(u) for _, _, u in items)
        tin = prompt_chars / 4.08
        # MEASURED, not a fraction of the cap. A 10-document pilot on 2026-09-19 gave a mean
        # cleaned-annotation/input ratio of 0.315 at ~3 chars/token for JSON. Pricing off
        # `max_tokens` instead over-estimated by 2.4x, which is the wrong direction to be
        # wrong in when the number decides whether a batch is bought.
        tout = sum(len(c["text"]) for c in cands) * 0.315 / 3.0 * 1.35   # 1.35: raw > cleaned
        rate_in, rate_out = (0.50, 2.50) if args.batch else (1.00, 5.00)
        print(f"[long-ann] DRY RUN -- nothing submitted")
        print(f"[long-ann]   input  ~{tin:,.0f} tok -> ${tin / 1e6 * rate_in:.2f}")
        print(f"[long-ann]   output ~{tout:,.0f} tok -> ${tout / 1e6 * rate_out:.2f}")
        print(f"[long-ann]   TOTAL  ~${tin / 1e6 * rate_in + tout / 1e6 * rate_out:.2f}"
              f"  ({'batch' if args.batch else 'standard'} pricing)")
        return 0

    provider = AnthropicProvider(ProviderConfig(
        provider="anthropic", model=args.model, max_tokens=args.max_tokens,
        json_object=True))

    if args.fetch_batch:
        replies = provider.fetch_batch(args.fetch_batch)
    elif args.batch:
        replies = provider.complete_batch(items, id_path=args.batch_id_file)
    else:
        replies = {cid: provider.complete(s, u) for cid, s, u in items}

    rows, unusable, empty, kept, dropped = [], 0, 0, 0, 0
    tasks: Counter = Counter()
    why: Counter = Counter()
    for cid, raw in replies.items():
        obj, reason = _parse_why(raw)
        if obj is None:
            unusable += 1
            why[reason] += 1
            print(f"[long-ann]   UNUSABLE doc {cid}: {reason} "
                  f"(input {len(cands[int(cid)]['text']):,} chars, reply {len(raw or ''):,})")
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
    print(f"[long-ann] {len(rows)} rows, {empty} empty, {unusable} unusable")
    if why:
        print(f"[long-ann]   unusable BY REASON: {dict(why)}")
    if total:
        print(f"[long-ann]   VERBATIM kept {kept}/{total} = {kept / total:.1%}, dropped {dropped}")
    print(f"[long-ann]   task coverage: {dict(tasks)}")
    print(f"[long-ann] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
