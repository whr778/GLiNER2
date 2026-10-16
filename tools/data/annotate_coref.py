"""Buy English entity COREFERENCE clusters from an LLM, as a coref-only corpus (option 4).

    uv run python tools/data/annotate_coref.py --corpus cc_news_events_sonnet55_v2 --dry-run
    uv run python tools/data/annotate_coref.py --corpus cc_news_events_sonnet55_v2 --batch
    recover: ... --corpus cc_news_events_sonnet55_v2 --fetch-batch msgbatch_...

The prompt, renderer and parser live here and coref_llm_pilot.py imports them: the ones the pilot validated (Sonnet 5.5:
pronoun-pair F1 0.76 at precision 0.90 on news up to 12,000 chars; Haiku failed). The LLM CLUSTERS
numbered candidate mentions and never writes a span, so every mention is a real offset.

Candidates: every occurrence of the record's entity and event-argument surfaces, plus pronouns in
lowercase or Capitalized form only ("US" and the "Us" of Toys "R" Us are not pronouns), longest
surface first. WikiEvents also offers its gold coref mentions, so the LLM can complete its partial gold.
Texts over 12,000 chars are cut into windows, and each window is its own record: a cluster split by
a cut would otherwise train two mentions of one entity as different entities.

Writes data/<corpus>_coref_llm.train.jsonl, records {input, output: {}, coref_clusters, provenance}
like gum_coref. TRAIN splits only: dev and test stay blind and gold.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record
from annotation import rules
from synthetic.providers import AnthropicProvider, ProviderConfig

PRONOUNS = {"he", "she", "they", "him", "her", "them", "his", "their", "hers", "theirs", "its", "it", "we",
            "us", "our", "i", "me", "my", "you", "your", "who", "whom", "whose", "himself", "herself",
            "themselves", "itself"}
MODELS = {"haiku": ("claude-haiku-4-5-20251001", 0.50, 2.50), "sonnet": ("claude-sonnet-5-5", 1.50, 7.50)}
# The rules are annotation/GUIDELINES.md's `coreference` block -- reviewing that IS reviewing this prompt.
SYSTEM = (
    "You resolve entity coreference in English news text."
) + " " + rules("json_only", "coreference")
USER = """In the text below, candidate mentions are tagged <m id=N>...</m>. Group the ids that refer
to the same entity.

Reply as {{"clusters": [[id, id, ...], ...]}} listing only groups of 2 or more ids; every id not
listed is its own entity. An id appears in at most one group.

Text:
{text}"""
MAX_CHARS = 12000
REPLIES = Path("/Volumes/Development/tmp/coref_buy")


def render(text: str, cands: list[dict]) -> str:
    out, pos = [], 0
    for i, m in enumerate(cands):
        out += [text[pos:m["start"]], f"<m id={i}>{text[m['start']:m['end']]}</m>"]
        pos = m["end"]
    return "".join(out + [text[pos:]])


def parse(raw: str, n: int) -> list[int] | None:
    """Predicted cluster id per candidate (singletons get unique ids), or None if unusable.
    The LAST JSON object with "clusters" wins: Sonnet sometimes answers, writes "Wait, ... Corrected:"
    and answers again, and the correction is its answer."""
    dec, groups, i = json.JSONDecoder(), None, raw.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(raw, i)
            if isinstance(obj, dict) and isinstance(obj.get("clusters"), list):
                groups = obj["clusters"]
            i = raw.find("{", end)
        except ValueError:
            i = raw.find("{", i + 1)
    if groups is None:
        return None
    label = list(range(n, 2 * n))
    for g, ids in enumerate(groups):
        for i in ids if isinstance(ids, list) else []:
            if isinstance(i, int) and 0 <= i < n and label[i] >= n:
                label[i] = g
    return label


def surfaces(rec: dict) -> set[str]:
    out = rec.get("output") or {}
    found = {s for v in (out.get("entities") or {}).values() for s in v if s and s.strip()}
    return found | {a["entity"] for e in (out.get("events") or []) for a in e.get("arguments", [])
                    if a.get("entity") and a["entity"].strip()}


def mention_spans(text: str, names: set[str], taken: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Non-overlapping occurrences of names and pronouns, avoiding taken spans."""
    forms = names | PRONOUNS | {p.capitalize() for p in PRONOUNS}
    pattern = "|".join(sorted((re.escape(f) for f in forms), key=len, reverse=True))
    spans = []
    for m in re.finditer(rf"(?<!\w)(?:{pattern})(?!\w)", text):
        if not any(a < m.end() and m.start() < b for a, b in taken):
            spans.append((m.start(), m.end()))
    return spans


def windows(text: str, max_chars: int) -> list[tuple[int, int]]:
    """Cut points at the last sentence end (". ") or newline before max_chars."""
    out, start = [], 0
    while len(text) - start > max_chars:
        cut = max(text.rfind(". ", start, start + max_chars) + 2, text.rfind("\n", start, start + max_chars) + 1)
        cut = cut if cut > start + 1 else start + max_chars
        out.append((start, cut))
        start = cut
    return out + [(start, len(text))]


def units(rec: dict, max_chars: int) -> list[dict]:
    """One {text, cands} per window; cands are window-relative {start, end, text}."""
    text = rec["input"]
    gold = [(m["start"], m["end"]) for c in rec.get("coref_clusters") or [] for m in c]
    spans = sorted(gold + mention_spans(text, surfaces(rec), gold))
    out = []
    for ws, we in windows(text, max_chars):
        t = text[ws:we]
        cands = [{"start": s - ws, "end": e - ws, "text": text[s:e]} for s, e in spans if ws <= s and e <= we]
        out.append({"text": t, "cands": cands, "offset": ws})
    return out


def clusters(labels: list[int], cands: list[dict]) -> list[list[dict]]:
    """Every candidate's cluster, SINGLETONS KEPT (as gum_coref keeps them): an id the LLM left
    ungrouped is its judgment that the mention is its own entity, a negative for every other mention.
    Dropping them threw away 33-50% of candidates and left 2.8-4.6% of records with no negatives."""
    groups: dict[int, list[dict]] = {}
    for lab, m in zip(labels, cands):
        groups.setdefault(lab, []).append(m)
    return list(groups.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=True, help="reads data/<corpus>.train.jsonl")
    ap.add_argument("--model", default="sonnet", choices=list(MODELS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--dry-run", action="store_true", help="build, count and show one prompt; send nothing")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--fetch-batch")
    ap.add_argument("--from-replies", action="store_true", help="rebuild from the saved replies; send nothing")
    args = ap.parse_args()

    with open(f"data/{args.corpus}.train.jsonl", encoding="utf-8") as fh:
        recs = [json.loads(l) for l in fh][:args.limit]
    work = [(i, w, u) for i, r in enumerate(recs) for w, u in enumerate(units(r, args.max_chars)) if len(u["cands"]) >= 2]
    n_c = sum(len(u["cands"]) for _, _, u in work)
    n_p = sum(m["text"].casefold() in PRONOUNS for _, _, u in work for m in u["cands"])
    print(f"[coref-buy] {args.corpus}: {len(recs):,} docs -> {len(work):,} units; {n_c:,} candidates "
          f"({n_p / max(n_c, 1):.1%} pronouns)")
    if args.dry_run:
        print(USER.format(text=render(work[0][2]["text"], work[0][2]["cands"]))[:2500])
        return 0

    model = MODELS[args.model][0]
    provider = None if args.from_replies else AnthropicProvider(
        ProviderConfig(provider="anthropic", model=model, max_tokens=4000))
    REPLIES.mkdir(parents=True, exist_ok=True)
    items = [(f"{i}_{w}", SYSTEM, USER.format(text=render(u["text"], u["cands"]))) for i, w, u in work]
    if args.from_replies:
        replies = json.loads((REPLIES / f"{args.corpus}.replies.json").read_text(encoding="utf-8"))
        batch_id = (REPLIES / f"{args.corpus}.batch_id").read_text().strip()
    elif args.fetch_batch:
        replies = provider.fetch_batch(args.fetch_batch)
        batch_id = args.fetch_batch
    elif args.batch:
        id_path = REPLIES / f"{args.corpus}.batch_id"
        replies = provider.complete_batch(items, id_path=id_path)
        batch_id = id_path.read_text().strip()
    else:
        replies = {cid: provider.complete(s, u) for cid, s, u in items}
        batch_id = "sync"
    if not args.from_replies:
        (REPLIES / f"{args.corpus}.replies.json").write_text(json.dumps(replies, ensure_ascii=False), encoding="utf-8")

    rows, unusable = [], 0
    for i, w, u in work:
        raw = replies.get(f"{i}_{w}")
        labels = parse(raw, len(u["cands"])) if raw else None
        if labels is None:
            unusable += 1
            continue
        cl = clusters(labels, u["cands"])
        assert all(u["text"][m["start"]:m["end"]] == m["text"] for c in cl for m in c)
        rows.append({"input": u["text"], "output": {}, "coref_clusters": cl,
                     "provenance": {"source": args.corpus, "doc_index": i, "window": w, "char_offset": u["offset"],
                                    "annotator": model, "batch_id": batch_id}})
    out = Path(f"data/{args.corpus}_coref_llm.train.jsonl")
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(dumps_record(r) + "\n")
    n_cl = sum(len(r["coref_clusters"]) for r in rows)
    n_m = sum(len(c) for r in rows for c in r["coref_clusters"])
    n_s = sum(len(c) == 1 for r in rows for c in r["coref_clusters"])
    print(f"[coref-buy] {len(rows):,} records -> {out}; {unusable} unusable / missing; {n_cl:,} clusters "
          f"({n_s:,} singletons), {n_m:,} mentions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
