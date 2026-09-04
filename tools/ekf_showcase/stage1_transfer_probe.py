"""Does stage 1 transfer to Turkish and Chinese without any Turkish event supervision?

Stage 1 (`build_docee_schema`) is the pipeline's only English-only stage: gate (0) and
casualty extraction (2) are both multilingual now, stage 1 is a span DeBERTa. It supplies
three things the tracker depends on:

    event type   the association key -- which stream a figure joins
    Location     separates co-occurring events (Turkiye vs Syria share a type)
    Date         rejects HISTORICAL tolls. The Turkish feed quotes Haiti 2010 (316 bin),
                 Sichuan 2008 (87 bin 900) and Antakya 115/525 AD as death tolls, all
                 correctly bound as `dead`; only a date can reject them. This is the
                 Turkish twin of the 1999 Izmit 17,500 contaminant.

So a stage-1 gap is not cosmetic: it is why Turkish contaminants are currently
unrejectable.

WHY THIS IS TESTABLE FOR FREE. All three feeds cover the SAME event, so the gold type of
a genuinely earthquake-related document is `earthquake` in every language, with no
annotation. Language-specific keywords (deprem / 地震 / earthquake) decide relevance; the
Turkish feed deliberately carries unrelated politics and sport, which become a precision
check rather than being discarded.

THE LABEL SET IS NOT DocEE's. DocEE's 27-label vocabulary was an early placeholder. The
reconciled measurement in the project memory says the boundary/mmBERT classification head
collapses on LARGE, effectively zero-shot label sets (59 -> 0.0%, 32 -> 5.0%) while
scoring 97.5-100% on small IN-MIX ones -- and two of those wins, chfinann (5) and docfee
(9), are CHINESE. So the question is not "can mmBERT classify" but "does a small
disaster-shaped label set transfer", which is what this asks. 8 casualty types + 3 decoys
= 11, inside the range where the curve is flat.

    uv run python tools/ekf_showcase/stage1_transfer_probe.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TASK = "event_type"
TYPES = ["earthquake", "flood", "storm", "fire", "explosion",
         "armed conflict", "transport accident", "disease outbreak"]
DECOYS = ["election", "sports competition", "diplomatic talks"]

ENTITIES = {
    "Casualties and Losses": "people killed, injured, missing or otherwise harmed, with counts",
    "Location": "where the event happened",
    "Date": "a date, year or time reference such as 1999, August 1999, Monday, February 6",
}

FEEDS = {
    "en": "datasets/turkey2023/_cache/feed.jsonl",
    "tr": "datasets/turkey2023/_cache/feed_tr.jsonl",
    "zh": "datasets/turkey2023/_cache/feed_zh.jsonl",
}
# Relevance keywords per language. Deliberately narrow: this decides GOLD, so a false
# positive here would manufacture an error the model did not make.
QUAKE = {"en": ["earthquake", "quake"], "tr": ["deprem"], "zh": ["地震"]}

MODELS = [
    ("casualty-docee (incumbent, en-only span)", "whr778/gliner2-base-v1-casualty-docee"),
    ("137k-v2-eb16 (mmBERT base)", "whr778/gliner2-joint-boundary-mmbert-137k-v2-eb16"),
    ("casualty-multilingual (mmBERT, reads 3)", "whr778/gliner2-casualty-multilingual"),
]


def load(lang: str) -> list[dict]:
    rows = [json.loads(l) for l in Path(FEEDS[lang]).open(encoding="utf-8") if l.strip()]
    for r in rows:
        r["_quake"] = any(k in r["text"].lower() for k in QUAKE[lang])
    return rows


def probe(model, rows: list[dict], threshold: float) -> dict:
    schema = (model.create_schema()
              .classification(TASK, TYPES + DECOYS)
              .entities(ENTITIES))
    hit = n_rel = 0
    off_rel = 0
    loc = date = cas = 0
    preds: dict[str, int] = {}
    for r in rows:
        try:
            out = model.extract(r["text"], schema, threshold=threshold,
                                include_confidence=True, include_spans=True)
        except Exception as exc:                    # a stage-1 crash is itself a result
            preds[f"_ERROR:{type(exc).__name__}"] = preds.get(f"_ERROR", 0) + 1
            continue
        ev = out.get(TASK)
        label = ev.get("label") if isinstance(ev, dict) else ev
        preds[str(label)] = preds.get(str(label), 0) + 1
        ents = out.get("entities") or {}
        if isinstance(ents, list):
            ents = ents[0] if ents else {}
        if r["_quake"]:
            n_rel += 1
            hit += (label == "earthquake")
            loc += bool(ents.get("Location"))
            date += bool(ents.get("Date"))
            cas += bool(ents.get("Casualties and Losses"))
        else:
            off_rel += (label == "earthquake")      # precision: should NOT be earthquake
    return {"n_rel": n_rel, "type_acc": hit / n_rel if n_rel else None,
            "n_irrel": len(rows) - n_rel, "false_quake": off_rel,
            "loc": loc / n_rel if n_rel else None,
            "date": date / n_rel if n_rel else None,
            "cas": cas / n_rel if n_rel else None,
            "preds": dict(sorted(preds.items(), key=lambda kv: -kv[1])[:4])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from gliner2 import AutoExtractor

    feeds = {lang: load(lang) for lang in FEEDS}
    for lang, rows in feeds.items():
        print(f"[feed] {lang}: {len(rows)} docs, {sum(r['_quake'] for r in rows)} quake-relevant")

    results = {}
    for label, repo in MODELS:
        print(f"\n=== {label} ===")
        model = AutoExtractor.from_pretrained(repo)
        model.eval()
        for lang, rows in feeds.items():
            r = probe(model, rows, args.threshold)
            results[f"{repo}|{lang}"] = r
            acc = "n/a" if r["type_acc"] is None else f"{r['type_acc']:.3f}"
            print(f"  {lang}: type={acc} ({r['n_rel']} rel)  "
                  f"false_quake={r['false_quake']}/{r['n_irrel']}  "
                  f"Location={r['loc']:.2f} Date={r['date']:.2f} Casualties={r['cas']:.2f}"
                  if r["type_acc"] is not None else f"  {lang}: no relevant docs")
            print(f"       top preds: {r['preds']}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
