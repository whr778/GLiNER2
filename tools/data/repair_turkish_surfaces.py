"""Repair LEMMATISED gold surfaces in an agglutinative corpus, refusing unsafe repairs.

turkish_event's annotator produced dictionary forms while the text carries inflected ones:
gold ``mesafe`` where the document reads ``mesafenin``. 7.5% of its surfaces fail to align,
and 234 of 235 failures are the gold surface being a PREFIX of a longer word. Every failing
position sits well inside the window, so the long-standing `max_len`/windowing diagnosis was
wrong.

**This is worse than lost supervision.** The model predicts spans over the text's OWN words,
so it can never emit ``mesafe`` when the text reads ``mesafenin`` -- the gold is unlearnable
AND unscorable. And ``on_missing_surface=skip`` drops the mention while its LABEL may survive
on other surfaces, turning the dropped span into a false negative.

**THE REPAIR IS NOT UNCONDITIONALLY SAFE, which is why the guard exists.** Extending a gold
prefix to the word boundary is harmless for a case affix and WRONG for a derivational one:

    mesafe    -> mesafenin      genitive; same entity                      REPAIR
    futbol    -> futbolcunun    football -> of the footballer              REFUSE
    Bosna     -> bosnalı        Bosnia -> Bosnian                          REFUSE

Measured over 1,200 documents: 98.0% of repairs add case/number only, 1.2% are derivational,
0.8% have no prefix run. The derivational suffixes are a CLOSED SET, so the unsafe cases are
detected rather than merely rare -- a refused surface is left exactly as it was, for
``on_missing_surface=skip`` to drop as it does today.

    uv run python tools/data/repair_turkish_surfaces.py \
        --inputs data/turkish_event.train.jsonl --out-dir data --apply
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _split import dumps_record  # noqa: E402

# Turkish DERIVATIONAL suffixes: these build a new word from the stem, so extending a gold
# surface across one changes what the annotation refers to. Longest-first at match time.
DERIVATIONAL = (
    "cılık", "cilik", "culuk", "çülük", "lık", "lik", "luk", "lük",
    "sız", "siz", "suz", "süz", "cu", "cı", "ci", "cü", "çi", "çu", "çı",
    "lı", "li", "lu", "lü", "sal", "sel",
)


def is_derivational(suffix: str) -> bool:
    return any(suffix.startswith(s) for s in sorted(DERIVATIONAL, key=len, reverse=True))


def find_prefix_run(sub, words):
    """First index where every text word STARTS WITH the corresponding gold word."""
    n = len(sub)
    if not n:
        return -1
    for i in range(len(words) - n + 1):
        if all(words[i + j].lower().startswith(sub[j].lower()) for j in range(n)):
            return i
    return -1


def repair_record(record, splitter, exact_match, stats, examples):
    """Return (record, changed). Only entity surfaces are touched.

    THE REPAIRED TEXT IS SLICED FROM THE RAW INPUT, never rebuilt from tokens. The word
    splitter lowercases, and Turkish dotted capital I does not round-trip: `İ` folds to TWO
    codepoints (U+0069 U+0307). Rebuilding from tokens turned
    `'İnsan kaynaklı kimyasallar'` into `'i̇nsan kaynaklı kimyasalların'` -- a repair that
    silently lowercased the corpus and injected a combining dot. The splitter yields
    (token, start, end), so the exact original substring is available.
    """
    out = record.get("output")
    entities = (out or {}).get("entities")
    if not isinstance(entities, dict):
        return record, False
    text = record.get("input") or ""
    spans = list(splitter(text, lower=True))
    words = [tok for tok, _, _ in spans]
    changed = False
    for label, surfaces in entities.items():
        if not isinstance(surfaces, list):
            continue
        for idx, surface in enumerate(surfaces):
            if not isinstance(surface, str) or not surface.strip():
                continue
            sub = [t for t, _, _ in splitter(surface, lower=True)]
            if exact_match(sub, words)[0] != (-1, -1):
                stats["already_aligned"] += 1
                continue
            start = find_prefix_run(sub, words)
            if start < 0:
                stats["no_prefix_run"] += 1
                continue
            found = words[start:start + len(sub)]
            suffixes = [w[len(g):].lower() for w, g in zip(found, sub)]
            # Slice the RAW text from the first token's start to the last token's end.
            repaired = text[spans[start][1]:spans[start + len(sub) - 1][2]]
            if any(is_derivational(s) for s in suffixes if s):
                stats["refused_derivational"] += 1
                if len(examples["refused"]) < 12:
                    examples["refused"].append((surface, repaired))
                continue
            if repaired == surface:
                stats["no_prefix_run"] += 1
                continue
            surfaces[idx] = repaired
            changed = True
            stats["repaired"] += 1
            for s in suffixes:
                if s:
                    stats["suffixes"][s] += 1
            if len(examples["repaired"]) < 12:
                examples["repaired"].append((surface, repaired))
    return record, changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="write repaired copies here (default: report only)")
    ap.add_argument("--apply", action="store_true",
                    help="actually write; without it this is a dry run")
    ap.add_argument("--splitter", default="whitespace")
    args = ap.parse_args()

    from gliner2.processor import SchemaTransformer
    proc = SchemaTransformer(model_name="jhu-clsp/mmBERT-base",
                             word_splitter=args.splitter)

    stats = collections.Counter()
    stats["suffixes"] = collections.Counter()
    examples = {"repaired": [], "refused": []}

    for path in args.inputs:
        src = Path(path)
        records, changed_any = [], 0
        with src.open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                rec, changed = repair_record(
                    rec, proc.word_splitter, proc._find_sublist, stats, examples)
                records.append(rec)
                changed_any += bool(changed)
        stats["records"] += len(records)
        stats["records_changed"] += changed_any
        if args.apply and args.out_dir:
            dst = Path(args.out_dir) / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("w", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(dumps_record(rec) + "\n")
            print(f"[repair] wrote {dst} ({changed_any} records changed)")

    total = stats["already_aligned"] + stats["repaired"] + stats["refused_derivational"] + stats["no_prefix_run"]
    print(f"\n[repair] {stats['records']:,} records, {total:,} entity surfaces")
    for key, label in (("already_aligned", "already aligned, untouched"),
                       ("repaired", "REPAIRED to the word boundary"),
                       ("refused_derivational", "REFUSED (derivational: would change the head noun)"),
                       ("no_prefix_run", "no prefix run; left for on_missing_surface=skip")):
        print(f"    {stats[key]:>7,}  {100 * stats[key] / max(total, 1):5.1f}%  {label}")
    print(f"\n  most common added suffixes:")
    for suf, n in stats["suffixes"].most_common(8):
        print(f"    {n:>5}  -{suf}")
    for name, title in (("repaired", "REPAIRED"), ("refused", "REFUSED")):
        if examples[name]:
            print(f"\n  {title} examples:")
            for a, b in examples[name][:6]:
                print(f"    {a!r} -> {b!r}")
    if not args.apply:
        print("\n[repair] DRY RUN -- nothing written. Pass --apply --out-dir to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
