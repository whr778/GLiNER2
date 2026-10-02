"""Adjudicate a RANDOM sample of absent-event negatives with Claude: is the event really absent?

The trace (trace_absent_event_negatives.py) gives DETECTOR rates -- a known trigger surface in
the text -- which over-count (equipment nouns, compounds) and under-count (paraphrases). This
draws a random sample of real injections per corpus, flagged or not, and asks a model whether
the document reports a specific occurrence of the injected type. The result is the TRUE
contradiction rate per corpus, plus the detector's precision and recall against it.

    uv run python tools/data/adjudicate_absent_events.py --per-corpus 50          # build + submit, waits
    uv run python tools/data/adjudicate_absent_events.py --fetch <batch_id>       # resume a batch
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trace_absent_event_negatives as TR  # noqa: E402
from synthetic.providers import ProviderConfig, build_provider  # noqa: E402

OUT = Path("/Volumes/Development/tmp/absent_adjudication")
SYSTEM = (
    "You audit training labels for event extraction. You are given a document and an event "
    "type that the label set says is ABSENT from the document. Decide whether the document "
    "reports at least one specific occurrence of that event type (something that happened, is "
    "happening, or is reported as having happened). Do NOT count: a mere mention of a word "
    "associated with the type, a capability, equipment or organisation name, a hypothetical, "
    "a plan that did not happen, or a general statement. Reply with JSON only: "
    '{"occurs": true|false, "evidence": "<shortest quote from the document, or empty>"}'
)


def sample(per_corpus: int, config: str, seed: int = 7):
    recs, inj, lex = TR.load(config)
    unambiguous = TR.unambiguous_fn(lex)
    rows = []
    for corpus in recs:
        pool = list(TR.injections(corpus, recs, inj))
        for k, (r, etype) in enumerate(random.Random(seed).sample(pool, min(per_corpus, len(pool)))):
            loose, tight = TR.flags(r, etype, lex, unambiguous)
            examples = [s for s, _ in lex[etype].most_common(8)]
            rows.append({"id": f"{corpus}-{k}", "corpus": corpus, "event_type": etype, "text": r["input"],
                         "examples": examples, "loose": bool(loose), "tight": bool(tight)})
    return rows


def user_prompt(row) -> str:
    return (f"Event type: {row['event_type']}\nExample trigger words for this type in the corpus: "
            f"{', '.join(row['examples'])}\n\nDocument:\n{row['text']}")


def parse(raw: str):
    """The LAST flat JSON object carrying `occurs`: the model sometimes reconsiders after its first answer."""
    found = [o for o in re.findall(r"\{[^{}]*\}", raw or "") if '"occurs"' in o]
    return json.loads(found[-1]).get("occurs") if found else None


def report(rows, answers) -> None:
    by = defaultdict(Counter)
    for row in rows:
        occurs = parse(answers.get(row["id"], ""))
        c = by[row["corpus"]]
        c["n"] += 1
        c["unparsed"] += occurs is None
        c["occurs"] += occurs is True
        for d in ("loose", "tight"):
            c[f"{d}_flag"] += row[d]
            c[f"{d}_hit"] += row[d] and occurs is True
    print(f"{'corpus':24s} {'n':>4s} {'TRUE contradiction':>19s} | tight flag: precision recall | loose flag: precision recall | unparsed")
    for corpus, c in by.items():
        def pr(d):
            p = c[f"{d}_hit"] / c[f"{d}_flag"] if c[f"{d}_flag"] else float("nan")
            r = c[f"{d}_hit"] / c["occurs"] if c["occurs"] else float("nan")
            return f"{p:9.2f} {r:6.2f}"
        print(f"{corpus:24s} {c['n']:4d} {c['occurs']:6d} ({100 * c['occurs'] / c['n']:5.1f}%)      | "
              f"{pr('tight')}             | {pr('loose')}             | {c['unparsed']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="tools/train/config/base/eb18-balanced.yaml")
    ap.add_argument("--per-corpus", type=int, default=50)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--fetch", help="batch id to resume instead of submitting")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows_path = OUT / "rows.jsonl"
    if args.fetch:
        rows = [json.loads(l) for l in open(rows_path, encoding="utf-8")]
    else:
        rows = sample(args.per_corpus, args.config)
        with open(rows_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        print(f"[adjudicate] {len(rows)} injections sampled -> {rows_path}")
    provider = build_provider(ProviderConfig(provider="anthropic", model=args.model, max_tokens=300))
    answers = (provider.fetch_batch(args.fetch) if args.fetch else
               provider.complete_batch([(r["id"], SYSTEM, user_prompt(r)) for r in rows], id_path=OUT / "batch_id"))
    with open(OUT / "answers.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False)
    report(rows, answers)


if __name__ == "__main__":
    main()
