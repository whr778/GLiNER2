"""Annotate ENTITY gold on Chinese event corpora that have none (cmnee, duee).

WHY. A role-type map can only be derived from -- or applied to -- documents carrying BOTH
entity and event gold. Measured over the eb16 mix: cmnee 25,679 event arguments at 0%
entity-bearing, duee 6,757 at 0%, casie 17,992 at 100%. So 35.7% of arguments are reachable
and casie is the only source. This buys the other 64%.

OPTION 4 DOES NOT SUBSTITUTE: its labels are `Event<Role>`, derived from the roles, so typing
a role by them is circular, and the canonical mode (`Date`->Date) is tautological the same
way. Only INDEPENDENT entity annotation unlocks it -- which is what this is.

LABELS IN ENGLISH, SPANS IN CHINESE, and the label space is the canonical one already in use
(`Person`, `Organization`, `Location`, `Date`, `Quantity`, ... -- the targets of
`labels/unified.yaml`, and exactly what `zh_multitask` carries). A new corpus that invents
its own spelling forks the label space of every model trained beside it.

BATCHING SHORT DOCUMENTS. duee is SENTENCES -- median 32 characters, mean 54. One per request
spends ~400 prompt tokens on a 40-token input and costs $6.14 instead of $3.79, so several
documents ride one request and the reply is keyed by document number.

    uv run python tools/data/annotate_entities_zh.py --corpus data/cmnee --batch
    uv run python tools/data/annotate_entities_zh.py --corpus data/duee --docs-per-request 10 --batch
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
from _split import dumps_record  # noqa: E402
from annotation import rules  # noqa: E402
from data.synthetic.providers import AnthropicProvider, ProviderConfig  # noqa: E402

REFUSAL_MARK = "__REFUSAL__"

# The canonical targets of labels/unified.yaml, which zh_multitask already uses. Seeds, not a
# closed set -- a closed list would drop the domain types military text needs.
SEEDS = ("Person", "Organization", "Location", "Date", "Time", "Quantity", "Facility",
         "Product", "Event", "Money", "Nationality", "Occupation", "Region", "Measurement")

SYSTEM = ("You annotate Chinese text for a named-entity dataset."
          ) + " " + rules("json_only", "verbatim", "no_inference", "minority", "ambiguity")

USER = """Label the named entities in each numbered Chinese passage below.

TWO LANGUAGES, DELIBERATELY. Every entity TYPE you write must be in ENGLISH. Every SPAN you
copy out must stay in the passage's own Chinese, character for character. The types are the
schema; the spans are the text.

CRITICAL RULE — copy spans verbatim. Every surface must appear in its OWN passage EXACTLY as
you write it. Do not translate, normalise, or add words. If you cannot find an exact span,
omit it. Never copy a span from one passage into another's answer.

Use these types where they fit, and add others in the same capitalised style when they do not:
{seeds}

Reply as ONE JSON object keyed by passage number, each value a map of type to the list of
Chinese surfaces of that type. A passage with no entities gets an empty object:

{{"1": {{"Organization": ["..."], "Date": ["..."]}}, "2": {{}}}}

PASSAGES:
{passages}"""


def load(path: Path, limit: int) -> list[dict]:
    out = []
    for split in ("train", "val", "test"):
        f = path.with_name(f"{path.name}.{split}.jsonl")
        if not f.is_file():
            continue
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if (rec.get("output") or {}).get("entities"):
                    continue          # already has entity gold -- never overwrite it
                text = rec.get("input") or ""
                if text.strip():
                    out.append({"text": text, "split": split, "rec": rec})
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


def clean(ents: dict, text: str) -> tuple[dict, int, int]:
    """Keep only spans VERBATIM in THIS passage. Returns (entities, kept, dropped)."""
    kept = dropped = 0
    out = {}
    if not isinstance(ents, dict):
        return out, 0, 0
    for etype, surfaces in ents.items():
        if not isinstance(surfaces, list):
            continue
        good = []
        for s in surfaces:
            s = str(s).strip()
            if not s:
                continue
            # The cross-passage check that batching makes necessary: a surface belonging to
            # a NEIGHBOUR in the same request would otherwise be accepted silently.
            if s in text:
                good.append(s); kept += 1
            else:
                dropped += 1
        if good:
            out[str(etype)] = sorted(set(good), key=good.index)
    return out, kept, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=True, help="e.g. data/cmnee")
    ap.add_argument("--out", help="default: <corpus>_ner")
    ap.add_argument("--docs-per-request", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", action="store_true", help="Batch API, -50%% pricing")
    ap.add_argument("--fetch-batch")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--batch-id-file")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    docs = load(corpus, args.limit)
    groups = [docs[i:i + args.docs_per_request]
              for i in range(0, len(docs), args.docs_per_request)]
    chars = sum(len(d["text"]) for d in docs)
    print(f"[zh-ner] {len(docs):,} documents -> {len(groups):,} requests "
          f"({args.docs_per_request}/request), {chars:,} chars")

    items = []
    for gi, g in enumerate(groups):
        passages = "\n\n".join(f"[{i+1}]\n{d['text']}" for i, d in enumerate(g))
        items.append((str(gi), SYSTEM,
                      USER.format(seeds=", ".join(SEEDS), passages=passages)))

    if args.dry_run:
        tin = sum(len(u) for _, _, u in items) / 1.39
        # Output scales with the DOCUMENT, not per-document flat. duee is 32-character
        # sentences: a flat 250 tokens of entity JSON for one of those is nonsense and
        # inflated its quote from ~$2 to $8.16. Measured shape: an entity map runs roughly
        # a third of the passage's own token count, floored so a tiny passage still pays
        # for its JSON envelope.
        tout = sum(max(25, len(d["text"]) / 1.39 * 0.33) for d in docs)
        ri, ro = (0.50, 2.50) if args.batch else (1.00, 5.00)
        print(f"[zh-ner] DRY RUN  input ~{tin:,.0f} tok ${tin/1e6*ri:.2f}  "
              f"output ~{tout:,.0f} tok ${tout/1e6*ro:.2f}  "
              f"TOTAL ~${tin/1e6*ri + tout/1e6*ro:.2f}")
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

    rows = {"train": [], "val": [], "test": []}
    kept = dropped = unusable = 0
    types: Counter = Counter()
    for cid, raw in replies.items():
        obj = parse(raw)
        g = groups[int(cid)]
        if obj is None:
            unusable += len(g)
            continue
        for i, d in enumerate(g):
            ents, k, dr = clean(obj.get(str(i + 1)) or {}, d["text"])
            kept, dropped = kept + k, dropped + dr
            if not ents:
                continue
            for t in ents:
                types[t] += 1
            rec = dict(d["rec"])
            out = dict(rec.get("output") or {})
            out["entities"] = ents
            rec["output"] = out
            rows[d["split"]].append(rec)

    base = Path(args.out or f"{corpus}_ner")
    total_rows = 0
    for split, rs in rows.items():
        if not rs:
            continue
        p = base.with_name(f"{base.name}.{split}.jsonl")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for r in rs:
                fh.write(dumps_record(r) + "\n")
        total_rows += len(rs)
        print(f"[zh-ner] {p} -- {len(rs):,} records")
    tot = kept + dropped
    print(f"[zh-ner] {total_rows:,} annotated, {unusable:,} unusable")
    if tot:
        print(f"[zh-ner]   VERBATIM kept {kept:,}/{tot:,} = {kept/tot:.1%}")
    print(f"[zh-ner]   types: {dict(types.most_common(12))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
