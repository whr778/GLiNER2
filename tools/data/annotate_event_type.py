"""Adjudicate the DocEE EVENT TYPE on real Turkish news, with Haiku.

Stage 1 of the EKF pipeline is its only English-only stage. It supplies the event type
(the association key), plus Location and Date -- and Date is the only thing that can
reject a historical toll, so the gap is not cosmetic. Measured 2026-09-04 on 48 real
Turkish documents with hand-adjudicated gold, the shipped span model types Turkish at
2/20 on genuine earthquake reporting; both mmBERT candidates score 0/20. Turkish
event-type supervision does not exist in any corpus in this repo, so it has to be bought.

STRATIFIED, AND THE STRATIFICATION IS THE POINT. `turkish_pool18` is 100% cue-bearing by
construction (20,000/20,000 measured) because it was built to feed CASUALTY annotation.
Buying only from it would produce a Turkish arm with no `Sports Competition`,
`Organization Fine` or `Government Job change` at all -- the model would learn the ~20
disaster types and never learn to say "not a disaster", which is the worst possible skew
for a stage whose job includes REJECTING non-disasters. So half comes from the cued pool
(depth on the disaster types) and half from `turkish_pool_general`, its verified 0.00%
cue-bearing complement (breadth on the other ~39). The two arms are disjoint by
construction, not by deduplication.

LABELS ARE DocEE's ENGLISH SPELLINGS, canonical from `data/docee.train.jsonl`. Turkish
text, English label -- exactly the shape DocEE-zh already ships (Chinese text, English
labels), so all three arms land in one label space with no translation step. The base's
spelling is canonical even where DocEE itself is misspelled (`Military Excercise`,
`Join in an Orgnization`): a downstream corpus must never outvote the base.

    pilot : uv run python tools/data/annotate_event_type.py --limit 500 --batch \
                --out data/turkish_gate/ev_ann_tr_pilot
    buy   : uv run python tools/data/annotate_event_type.py --batch \
                --out data/turkish_gate/ev_ann_tr
    recover: uv run python tools/data/annotate_event_type.py \
                --fetch-batch msgbatch_... --out data/turkish_gate/ev_ann_tr

A killed poller does NOT lose the batch -- it completes server-side. Recover it with
--fetch-batch; resubmitting pays twice for identical output.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _split import dumps_record, normalize_group_key  # noqa: E402
from data.synthetic.providers import (  # noqa: E402
    REFUSAL_MARK, AnthropicProvider, ProviderConfig,
)

MAX_CHARS = 6000

# DocEE's own 59, read from data/docee.train.jsonl. Kept verbatim, typos included.
LABELS = [
    "Air Crash", "Appoint_Inauguration", "Armed Conflict", "Awards Ceremony",
    "Bank Robbery", "Break Historical Records", "CommitCrime - Accuse",
    "CommitCrime - Arrest", "CommitCrime - Investigate", "CommitCrime - Release",
    "CommitCrime - Sentence", "Diplomatic Talks", "Diplomatic Visit",
    "Disease Outbreaks", "Droughts", "Earthquakes", "Election", "Environment Pollution",
    "Famine", "Famous Person - Death", "Famous Person - Divorce",
    "Famous Person - Give a Speech", "Famous Person - Marriage",
    "Famous Person - Recovered", "Famous Person - Sick", "Financial Aid",
    "Financial Crisis", "Fire", "Floods", "Gas Explosion", "Government Policy Changes",
    "Insect Disaster", "Join in an Orgnization", "Mass Poisoning", "Military Excercise",
    "Mine Collapses", "Mudslides", "New Achievements in Aerospace",
    "New Archeological Discoveries", "New Wonders in Nature", "Organization Closed",
    "Organization Established", "Organization Fine", "Organization Merge", "Protest",
    "Regime Change", "Resignation_Dismissal", "Riot", "Road Crash", "Shipwreck",
    "Sign Agreement", "Sports Competition", "Storm", "Strike", "Tear Up Agreement",
    "Train Collisions", "Tsunamis", "Volcano Eruption", "Withdraw from an Orgnization",
]
NONE_LABEL = "none"          # not in DocEE; lets the annotator decline rather than guess

SYSTEM = (
    "You label Turkish news documents with a document-level event type, for an "
    "information-extraction training set. Answer only about the text given; never infer "
    "beyond it. Reply with a single JSON object and nothing else."
)

USER = """What is the PRIMARY event this Turkish news article reports?

Choose exactly one label from this list, copied VERBATIM (they are English labels for Turkish text -- this is intentional):

{labels}

Or "{none}" if no label above fits the primary event.

Rules:
- Judge the PRIMARY event -- what the article is actually reporting. An article about an election that merely mentions an earthquake is "Election", not "Earthquakes".
- Do not choose a disaster label just because casualties are mentioned. A road accident that killed people is "Road Crash", not "Armed Conflict".
- "{none}" is a real answer. Prefer it over forcing a poor fit; a wrong label is worse than an absent one.
- Copy the label exactly as written above, including any spelling it uses.

Reply as: {{"label": "<one label>", "evidence": "<short verbatim Turkish quote naming the event, or empty>"}}

ARTICLE:
{text}"""


def load_candidates(cued: Path, uncued: Path, per_arm: int, seed: int,
                    exclude: set[str]) -> list[dict]:
    """Half from the cue-bearing pool, half from its complement. Disjoint by construction."""
    out: list[dict] = []
    for path, arm in ((cued, "tr_cued"), (uncued, "tr_uncued")):
        pool, seen = [], set()
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                text = (json.loads(line)["input"] or "")[:MAX_CHARS]
                key = normalize_group_key(text)[:300]
                if key in seen or key in exclude:
                    continue
                seen.add(key)
                pool.append({"text": text, "source": "turkish_news", "stratum": arm})
        random.Random(seed).shuffle(pool)
        take = pool[:per_arm]
        print(f"[ev-ann]   {arm:12s} pool={len(pool):7d} take={len(take)}")
        out.extend(take)
    random.Random(seed).shuffle(out)
    return out


def parse(raw: str) -> tuple[str, str] | None:
    """The label must be one we asked for; a near-miss is dropped, never repaired."""
    if not raw or REFUSAL_MARK in raw:
        return None
    try:
        obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except Exception:
        return None
    label = str(obj.get("label", "")).strip()
    if label not in LABELS and label != NONE_LABEL:
        return None
    return label, str(obj.get("evidence", "")).strip()


def write(rows: list[dict], out_prefix: str) -> None:
    path = Path(f"{out_prefix}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(dumps_record(r) + "\n")
    by_label = Counter(r["label"] for r in rows)
    by_arm = Counter(r["stratum"] for r in rows)
    print(f"[ev-ann] wrote {path} ({len(rows)} rows)")
    print(f"[ev-ann] per arm: {dict(by_arm)}")
    print(f"[ev-ann] {len(by_label)} distinct labels; top 15:")
    for lab, n in by_label.most_common(15):
        print(f"[ev-ann]   {lab:34s} {n}")
    none_n = by_label.get(NONE_LABEL, 0)
    print(f"[ev-ann] '{NONE_LABEL}' rate: {none_n}/{len(rows)} = "
          f"{none_n / max(len(rows), 1):.1%}  (high means the taxonomy fits Turkish news badly)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cued", default="/Volumes/Development/data/turkish_pool18.jsonl")
    ap.add_argument("--uncued", default="/Volumes/Development/data/turkish_pool_general.jsonl")
    ap.add_argument("--out", default="data/turkish_gate/ev_ann_tr")
    ap.add_argument("--limit", type=int, default=30000,
                    help="TOTAL documents; split evenly across the two arms")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", action="store_true", help="Batch API, -50%% pricing")
    ap.add_argument("--fetch-batch", help="recover an already-submitted batch id")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--exclude", nargs="+", default=[],
                    help="already-annotated jsonl(s); those documents are not re-bought")
    args = ap.parse_args()

    done: set[str] = set()
    for path in args.exclude:
        for line in Path(path).open(encoding="utf-8"):
            done.add(normalize_group_key(json.loads(line)["input"])[:300])
    if done:
        print(f"[ev-ann] {len(done)} already annotated, not re-buying")

    cands = load_candidates(Path(args.cued), Path(args.uncued),
                            args.limit // 2, args.seed, done)
    print(f"[ev-ann] {len(cands)} candidates total")

    provider = AnthropicProvider(ProviderConfig(
        provider="anthropic", model=args.model, max_tokens=200, json_object=True))
    labels_block = "\n".join(f"- {l}" for l in LABELS)
    items = [(str(i), SYSTEM,
              USER.format(labels=labels_block, none=NONE_LABEL, text=c["text"]))
             for i, c in enumerate(cands)]

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
    print(f"[ev-ann] {len(rows)} labelled, {unusable} unusable")
    write(rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
