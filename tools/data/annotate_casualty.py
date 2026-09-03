"""Annotate FIELD-LEVEL casualty records on real news, with Haiku, for extractor training.

`annotate_gate.py` buys the stage-0 question (does this report a toll?). This buys the
stage-2 one: WHICH numbers, bound to WHICH place and reporter. It exists because the
extractor cannot read Turkish -- `location` holds a digit 78.2% of the time on Turkish
against 5.8% on English (`extractor_language_probe.py`), so it emits confident wrong
figures the EKF cannot reject.

THE HARD CONSTRAINT: every field value must be a VERBATIM substring of the article. The
boundary head locates fields as spans, so a paraphrased or normalised value ("644,439"
where the text says "644 bin 439") produces a row that trains nothing and reports no
error. The existing casualty corpus is 100% verbatim (127,359/127,359 values checked), so
that is the bar. Values failing it are dropped here and the rate is printed -- a silent
drop would let a bad prompt spend the whole batch.

Output matches data/casualty_loc_split.train.jsonl exactly:

    {"input": ..., "output": {"json_structures": [{"casualty_report": {...}}],
                              "record_metadata": {"casualty_report": {"mode": "natural",
                                                                      "anchor": <a field
                                                                      the row HAS>}}}}

    smoke  : uv run python tools/data/annotate_casualty.py --limit 10 --out data/cas_ann_smoke
    batch  : uv run python tools/data/annotate_casualty.py --batch --out data/cas_ann_tr
    recover: uv run python tools/data/annotate_casualty.py --fetch-batch msgbatch_... --out data/cas_ann_tr

A killed poller does NOT mean lost money -- the batch completes server-side. Recover it
with --fetch-batch; resubmitting pays twice for identical output.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _split import dumps_record, normalize_group_key  # noqa: E402
from data.synthetic.providers import (  # noqa: E402
    REFUSAL_MARK, AnthropicProvider, ProviderConfig,
)

FIELDS = ("location", "dead", "injured", "missing", "source")
COUNTS = ("dead", "injured", "missing")

# What actually goes into a training record. `source` is deliberately NOT among them:
# it is in `build_casualty_schema` and asked for at inference, but supervised in no
# corpus on disk (casualty_loc_split, mix_natural and mix_anchorless all carry exactly
# these four, 0 source values in 131,730 records) -- which is why the extractor answers
# it with a digit. Training it for Turkish alone would make the field set a second
# variable in what is meant to be a language replication, so the annotations are kept
# beside the row under "sources" and left for a deliberate all-language experiment.
TRAIN_FIELDS = ("location", "dead", "injured", "missing")
MAX_CHARS = 6000

# A count must be a bare numeral, optionally using the article's own scale words -- Turkish
# writes 29,313 as "29 bin 313". The existing casualty corpus is 100% numeric on these
# three fields (0 of 72,454 values non-numeric), so admitting "onlarcasi" ("dozens") here
# would teach one convention for Turkish and another for English and Chinese, and the EKF
# needs a figure it can parse either way.
NUMERAL = re.compile(
    r"^\d[\d.,]*(?:\s+(?:bin|milyon|milyar|thousand|million|billion)(?:\s+\d[\d.,]*)?)*$",
    re.I)

SYSTEM = (
    "You extract casualty figures from news articles for a disaster-monitoring system. "
    "Copy values EXACTLY as they appear in the article, character for character. "
    "Reply with a single JSON object and nothing else."
)

USER = """Extract every casualty figure this article reports, as a list of records.

One record per (place, reporting) pairing. If the article gives separate tolls for separate places, emit one record each. If it gives one combined toll, emit one record.

Each record may contain these fields, all optional -- omit a field entirely if the article does not give it:

- "location": the place these casualties occurred in
- "dead": how many people were killed or confirmed dead (not injured, missing or displaced)
- "injured": how many people were injured or hurt (not killed, missing or displaced)
- "missing": how many people are missing or unaccounted for (not killed or injured)
- "source": who reported these figures

"dead", "injured" and "missing" must be a NUMBER and nothing else. Write only the numeral, with no verb, no suffix and no surrounding words. Keep the article's own way of writing large numbers.

- "olu sayisi 130'u asti" -> write "130", not "130'u asti" and not "130'u".
- "29 bin 313 kisi" -> write "29 bin 313", which is how the article writes 29,313.
- "2 million displaced" -> write "2 million", which is how the article writes 2,000,000.
- Vague quantities are NOT numbers. If the article says only "onlarcasi", "cok sayida", "yuzlerce", "dozens", "scores", "several" or "hundreds of" with no figure, OMIT that field.

CRITICAL RULE -- copy verbatim. Every value must appear in the article EXACTLY as you write it, character for character. Do not translate it, do not convert it, do not reformat it, do not add or remove words.

- If the article says "644 bin 439", write "644 bin 439". Do NOT write "644439" or "644,439".
- If the article says "22'ye", the number is "22".
- If the article says "en az 30", you may write "30" or "en az 30" -- both appear in the text. Do not write "at least 30".
- Copy place names in the article's own language and spelling, with its own accents.

Do NOT include:
- counts of people displaced, evacuated, rescued, sheltered or merely affected
- casualty figures for a DIFFERENT event cited as background or comparison
- a single named individual dying

If the article reports no current casualty figures at all, reply {{"records": []}}.

Reply as: {{"records": [{{"location": "...", "dead": "...", "source": "..."}}]}}

ARTICLE:
{text}"""


def load_candidates(paths: list[Path], limit: int, exclude: set[str]) -> list[dict]:
    """Documents to annotate, deduped against what has already been bought."""
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


def parse(raw: str) -> list[dict] | None:
    """The records list, or None when the reply is a refusal or unusable."""
    if not raw or raw.startswith(REFUSAL_MARK):
        return None
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    records = obj.get("records")
    return records if isinstance(records, list) else None


def clean(records: list[dict], text: str) -> tuple[list[dict], list[str], int, int]:
    """Keep only VERBATIM, well-formed field values. Returns (records, kept, dropped).

    A value the boundary head cannot locate in the text is worse than a missing one: it
    trains the model toward output it can never ground. Counts must also be bare numerals,
    matching the existing corpus. Drop the field, keep the record.
    """
    out, sources, kept, dropped = [], [], 0, 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        good = {}
        for field in FIELDS:
            value = rec.get(field)
            if value is None or not str(value).strip():
                continue
            value = str(value).strip()
            if field in COUNTS and not NUMERAL.match(value):
                dropped += 1
                continue
            if value not in text:
                dropped += 1
                continue
            kept += 1
            if field in TRAIN_FIELDS:
                good[field] = value
            else:
                sources.append(value)
        if good:
            out.append({"casualty_report": good})
    return out, sources, kept, dropped


def write(rows: list[dict], out_prefix: str) -> None:
    path = Path(f"{out_prefix}.jsonl")
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(dumps_record(row) + "\n")
    fields: dict[str, int] = {}
    per_doc: dict[int, int] = {}
    for row in rows:
        structs = row["output"]["json_structures"]
        per_doc[len(structs)] = per_doc.get(len(structs), 0) + 1
        for struct in structs:
            for field in struct["casualty_report"]:
                fields[field] = fields.get(field, 0) + 1
    print(f"[cas-ann] {path}: {len(rows)} rows")
    print(f"[cas-ann]   field frequency: {sorted(fields.items(), key=lambda x: -x[1])}")
    print(f"[cas-ann]   records per doc: {sorted(per_doc.items())}")


ANCHOR_ORDER = ("dead", "injured", "missing", "location")


def pick_anchor(structs: list) -> str:
    """Return the anchor field for a row: the first ANCHOR_ORDER field the row actually has.

    In ``natural`` mode the anchor field's mentions delimit record instances, so declaring
    one the row does not carry makes the record head raise "declares anchor 'dead' but no
    matching field query was found in the layout" on the first batch that contains such a
    row. A fixed ``dead`` is wrong for the 20-33%% of real news that reports only injured,
    only missing, or only a place -- and it shipped that way in cas_ann_en, cas_ann_ccnews,
    cas_ann_tr and cas_ann_zh, which repair_casualty_anchors.py fixes on disk.
    """
    present = {f for s in structs for f in (s.get("casualty_report") or {})}
    return next((f for f in ANCHOR_ORDER if f in present), ANCHOR_ORDER[0])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="+", required=True)
    ap.add_argument("--out", default="data/cas_ann_tr")
    ap.add_argument("--limit", type=int, default=0, help="cap candidates (smoke run)")
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
        print(f"[cas-ann] {len(done)} already annotated, not re-buying")

    cands = load_candidates([Path(p) for p in args.corpora], args.limit, done)
    print(f"[cas-ann] {len(cands)} candidates from {len(args.corpora)} corpora")

    provider = AnthropicProvider(ProviderConfig(
        provider="anthropic", model=args.model, max_tokens=900, json_object=True))
    items = [(str(i), SYSTEM, USER.format(text=c["text"])) for i, c in enumerate(cands)]

    if args.fetch_batch:
        replies = provider.fetch_batch(args.fetch_batch)
    elif args.batch:
        replies = provider.complete_batch(items, id_path=f"{args.out}.batch_id")
    else:
        replies = {cid: provider.complete(sys_, usr) for cid, sys_, usr in items}

    rows, unusable, empty, kept, dropped = [], 0, 0, 0, 0
    for cid, raw in replies.items():
        records = parse(raw)
        if records is None:
            unusable += 1
            continue
        text = cands[int(cid)]["text"]
        structs, sources, k, d = clean(records, text)
        kept, dropped = kept + k, dropped + d
        if not structs:
            empty += 1
            continue
        rows.append({"input": text, "output": {
            "json_structures": structs,
            "record_metadata": {"casualty_report": {"mode": "natural",
                                                    "anchor": pick_anchor(structs)}},
        }, "sources": sources})

    total = kept + dropped
    print(f"[cas-ann] {len(rows)} rows, {empty} no-figures, {unusable} unusable")
    print(f"[cas-ann]   VERBATIM values kept {kept}/{total} = "
          f"{kept / total:.1%}, dropped {dropped}" if total else "[cas-ann]   no values")
    write(rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
