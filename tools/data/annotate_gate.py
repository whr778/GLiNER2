"""Adjudicate the stage-0 gate question on REAL news text, with Haiku.

The gate asks: does this article REPORT A CURRENT TOLL for a group of people?

Every previous attempt labelled that with a pattern, and a pattern cannot answer it.
These three all match "N dead/injured" and must all be REJECTED:

    "220,000 earthquake victims have been served meals"   -- exposure, not a toll
    "in 1999 ... deaths of over 17,000"                   -- a historical event
    "cholera symptoms ... can lead to death if untreated" -- no toll at all

Those are the measured false positives of the shipped gate, so labelling positives by
regex trains the deployment failure straight in. This asks the model to separate them.

Only the AMBIGUOUS region is worth paying for: documents carrying a casualty cue. Text
with no cue at all is a free negative and never reaches the API.

    smoke : uv run python tools/data/annotate_gate.py --limit 10 --out data/gate_ann_smoke
    batch : uv run python tools/data/annotate_gate.py --batch --out data/gate_ann
    recover: uv run python tools/data/annotate_gate.py --fetch-batch msgbatch_... --out data/gate_ann

A killed poller does NOT mean lost money -- the batch completes server-side. Recover it
with --fetch-batch; resubmitting pays twice for identical output.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _split import dumps_record, normalize_group_key  # noqa: E402
from data.synthetic.providers import (  # noqa: E402
    REFUSAL_MARK, AnthropicProvider, ProviderConfig,
)

LABELS = ["current_toll", "historical_toll", "exposure_only", "no_toll"]

# Broad on purpose. The cue decides only what is worth LOOKING at; the model decides
# the label. A narrow cue here would pre-judge the very cases being bought.
CUE = re.compile(
    r"\b(dead|died|die|dies|killed|kill|fatal|fatalities|casualt\w*|injur\w*|wounded|"
    r"hurt|missing|perished|death|deaths|toll|survivors?|bodies|victims?)\b", re.I)

# Applied to every row here AND by the corpus builder to every row it writes, so the
# cap is uniform across classes. Above the encoder's 2048-token window either way.
MAX_CHARS = 6000

SYSTEM = (
    "You label news documents for a disaster-monitoring filter. Answer only about the "
    "text given; never infer beyond it. Reply with a single JSON object and nothing else."
)

USER = """Does this article report a CURRENT casualty toll -- a count of people killed, injured or missing in the event the article is reporting on?

Choose exactly one label:

- current_toll: reports how many people were killed, injured or missing in the event this article is covering. The count may be approximate ("dozens", "at least 30", "scores"). This is the only positive label.
- historical_toll: the only counts given are for a DIFFERENT event than the one this article is about -- cited as background, an anniversary, or a comparison.
- exposure_only: gives counts of people AFFECTED but not harmed -- displaced, evacuated, rescued, sheltered, fed, left without power, at risk, or living in the area.
- no_toll: no count of people in any of the senses above.

Work through these in order and stop at the first that applies:

1. Does it state how many people were killed, injured or missing in the event it is covering? -> current_toll. This includes coverage of the ongoing aftermath, rescue or investigation of that event.
2. Are the only such counts for a DIFFERENT event, cited as background or comparison? -> historical_toll. Judge this by which event the article is about, not by how long ago it happened -- a full account of a past disaster still reports that disaster's toll.
3. Does it give a count of people displaced, evacuated, rescued, sheltered or otherwise affected but not harmed, and NO casualty count? -> exposure_only. Choose this over no_toll whenever such a count is present, even an approximate one like "hundreds were evacuated".
4. Otherwise -> no_toll.

Also:
- The count must cover TWO OR MORE people harmed in one event. A single named person dying is no_toll.
- If it reports a current toll AND cites a historical one, choose current_toll.
- An article merely on a disaster topic, with no count at all, is no_toll.

Reply as: {{"label": "<one label>", "evidence": "<short verbatim quote, or empty>"}}

ARTICLE:
{text}"""


# A numeric toll PATTERN. Not a label -- it only says which stratum a document sits in,
# so the quotas below can buy adjudication where the answer is actually in doubt.
TOLLNUM = re.compile(
    r"\b(?:at least\s+|about\s+|some\s+|more than\s+)?\d[\d,]*\s+"
    r"(?:people\s+|persons\s+|others\s+|workers\s+|passengers\s+)?"
    r"(?:were\s+|are\s+|have\s+been\s+|been\s+)?"
    r"(?:dead|died|killed|injured|wounded|missing|hurt)\b|\bdeath toll\b", re.I)

# The whole cue-bearing region is bought (~18.7k, ~$12 batched); the strata exist to be
# REPORTED, and to cap one if it ever dominates. The two off-diagonal ones are the point:
# without casualty-topic-WITHOUT-toll the model learns topic instead of toll, and without
# other-topic-WITH-toll it learns that death words alone are enough. Both registers
# (docee's news+encyclopedia, cc_news wire copy) appear in every stratum that has them,
# so register cannot separate the classes either.
#
# Note what more money CANNOT buy: positives are capped by the data, not the budget --
# only 2,706 of these documents carry a numeric toll at all. Raising the positive count
# means pulling more raw cc_news (fetch_cc_news.py --exclude), not a bigger batch.
STRATA = ["docee_casualty_toll", "docee_casualty_notoll", "docee_other_toll",
          "docee_other_notoll", "ccnews_toll", "ccnews_notoll"]
CAPS: dict[str, int] = {}


def _stratum(rec: dict, text: str, source: str, casualty: set) -> str:
    toll = "toll" if TOLLNUM.search(text) else "notoll"
    if not source.startswith("docee"):
        return f"ccnews_{toll}"
    cls = ((rec.get("output") or {}).get("classifications") or [{}])[0]
    topic = "casualty" if (cls.get("true_label") or [None])[0] in casualty else "other"
    return f"docee_{topic}_{toll}"


def load_candidates(paths: list[Path], limit: int, seed: int = 42) -> list[dict]:
    """Cue-bearing real documents, deduped, sampled to the per-stratum QUOTAS."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ekf_showcase"))
    from run_pipeline import DOCEE_CASUALTY_TYPES  # noqa: E402
    casualty = set(DOCEE_CASUALTY_TYPES)

    seen: set[str] = set()
    pools: dict[str, list[dict]] = {}
    for path in paths:
        if not path.exists():
            print(f"[gate-ann] skip missing {path}")
            continue
        source = path.stem.split(".")[0]
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            text = (rec.get("input") or "").strip()[:MAX_CHARS]
            if not text or not CUE.search(text):
                continue
            # Syndication republishes the same wire story with a different tail, so exact
            # dedup misses it -- the smoke bought the same Christchurch article twice.
            # The shared LEAD is what identifies it.
            key = normalize_group_key(text)[:300]
            if key in seen:
                continue
            seen.add(key)
            stratum = _stratum(rec, text, source, casualty)
            pools.setdefault(stratum, []).append(
                {"text": text, "source": source, "stratum": stratum})

    rng = random.Random(seed)
    out = []
    for stratum in STRATA:
        pool = pools.get(stratum, [])
        rng.shuffle(pool)
        # A smoke run takes a few from EVERY stratum -- the labels only mean anything
        # if the hard cells are represented, not just the easy ones.
        take = pool[:-(-limit // len(STRATA))] if limit else pool[:CAPS.get(stratum)]
        print(f"[gate-ann]   {stratum:24s} pool={len(pool):6d} take={len(take)}")
        out += take
    rng.shuffle(out)
    return out[:limit] if limit else out


_LABEL_FIELD = re.compile(r'"label"\s*:\s*"([a-z_]+)"')
_EVIDENCE_FIELD = re.compile(r'"evidence"\s*:\s*"(.*?)"\s*\}?\s*$', re.S)


def parse(raw: str) -> tuple[str, str] | None:
    """(label, evidence), or None when the reply is a refusal or unusable.

    The evidence field is a VERBATIM quote from a news article, so it regularly contains
    a double quote the model does not escape, and `json.loads` rejects the whole reply.
    The label is the part that matters and it is a bare enum, so read it directly rather
    than lose a paid annotation to a broken quote in a field we only keep for auditing.
    """
    if not raw or raw.startswith(REFUSAL_MARK):
        return None
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        label, evidence = obj.get("label"), str(obj.get("evidence") or "")
    except json.JSONDecodeError:
        found = _LABEL_FIELD.search(match.group(0))
        if not found:
            return None
        label = found.group(1)
        quoted = _EVIDENCE_FIELD.search(match.group(0))
        evidence = quoted.group(1) if quoted else ""
    return (label, evidence) if label in LABELS else None


def write(rows: list[dict], out_prefix: str) -> None:
    path = Path(f"{out_prefix}.jsonl")
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(dumps_record(row) + "\n")
    counts: dict[str, int] = {}
    cross: dict[str, dict[str, int]] = {}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
        cell = cross.setdefault(row["stratum"], {})
        cell[row["label"]] = cell.get(row["label"], 0) + 1
    print(f"[gate-ann] {path}: {len(rows)} rows")
    print(f"[gate-ann]   labels: {counts}")
    # The cross-tab is the check that the money bought adjudication rather than the
    # regex back: if every *_toll row is current_toll the model is echoing the pattern.
    for stratum in STRATA:
        if stratum in cross:
            print(f"[gate-ann]   {stratum:24s} {cross[stratum]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="+", default=[
        "data/docee.train.jsonl",
        "data/cc_news_parts/cc_news_10k_raw.jsonl",
        "data/cc_news_parts/cc_news_10k_b_raw.jsonl",
    ])
    ap.add_argument("--out", default="data/gate_ann")
    ap.add_argument("--limit", type=int, default=0, help="cap candidates (smoke run)")
    ap.add_argument("--batch", action="store_true", help="Batch API, -50%% pricing")
    ap.add_argument("--fetch-batch", help="recover an already-submitted batch id")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = ap.parse_args()

    cands = load_candidates([Path(p) for p in args.corpora], args.limit)
    print(f"[gate-ann] {len(cands)} cue-bearing candidates from {len(args.corpora)} corpora")

    provider = AnthropicProvider(ProviderConfig(
        provider="anthropic", model=args.model, max_tokens=200, json_object=True))

    items = [(str(i), SYSTEM, USER.format(text=c["text"])) for i, c in enumerate(cands)]
    if args.fetch_batch:
        replies = provider.fetch_batch(args.fetch_batch)
    elif args.batch:
        replies = provider.complete_batch(items)
    else:
        replies = {cid: provider.complete(sys_, usr) for cid, sys_, usr in items}

    rows, unusable = [], 0
    for cid, raw in replies.items():
        parsed = parse(raw)
        if parsed is None:
            unusable += 1
            continue
        label, evidence = parsed
        cand = cands[int(cid)]
        rows.append({"input": cand["text"], "label": label, "evidence": evidence,
                     "source": cand["source"], "stratum": cand["stratum"]})
    print(f"[gate-ann] {len(rows)} labelled, {unusable} unusable")
    write(rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
