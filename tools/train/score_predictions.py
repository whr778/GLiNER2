"""Score a ``tools/infer.py --output`` predictions file exactly as the blind test scores.

    uv run tools/train/score_predictions.py --predictions test.preds.jsonl --by-language \\
        --out test.scored.json --card

Same scorer (``eval_metrics.score_predictions``, the scoring half of ``compute_metrics``), same
record filter (records with no gold labels are skipped), same exact-duplicate drop, and the same
language buckets (``train.language_groups``). Gold is ``gold_mapped`` when every line carries it
(``infer.py --labels-file``), else ``gold``.

The numbers equal the blind test's only when the predictions were made the same way:
``infer.py --gold-schema``, the checkpoint's own decode settings, on CPU or ``--batch-size 1``
on MPS. The blind test also drops test documents shared with train/val; this file cannot see
those splits, so that drop is the one thing it does not reproduce.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
from model_card import _metrics_table, _oneie_table  # noqa: E402

from gliner2.training.eval_metrics import _print_micro_report, _schema_from_gold, score_predictions  # noqa: E402
from gliner2.training.split_hygiene import _record_key  # noqa: E402


def read_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gold_key(rows: List[Dict[str, Any]], which: str) -> str:
    """``gold_mapped`` when every row has it (auto), else ``gold``; refuse a file that mixes them."""
    if which != "auto":
        return which
    mapped = sum("gold_mapped" in r for r in rows)
    if mapped and mapped != len(rows):
        raise SystemExit(f"[score] {mapped} of {len(rows)} lines carry gold_mapped: one infer.py run per file")
    return "gold_mapped" if mapped else "gold"


def prepare(rows: List[Dict[str, Any]], key: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Records ``{input, output: gold, pred}`` the blind test would score, and what was dropped."""
    seen, out, dropped = set(), [], {"no_gold": 0, "duplicate": 0}
    for r in rows:
        gold = r.get(key) or {}
        if not _schema_from_gold(gold):
            dropped["no_gold"] += 1
            continue
        rec = {"input": r["input"], "output": gold}
        k = _record_key(rec)
        if k in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(k)
        out.append(dict(rec, pred=r.get("output") or {}))
    return out, dropped


def off_menu(records: List[Dict[str, Any]]) -> int:
    """Records whose prediction names an entity, relation or event type outside their own gold menu:
    the file was not made with --gold-schema, so this is not the blind test's question."""
    n = 0
    for r in records:
        menu, p = _schema_from_gold(r["output"]), r["pred"]
        relations = {n for r in menu.get("relations") or [] for n in (r if isinstance(r, dict) else [r])}
        offered = set(menu.get("entities") or {}) | relations | set(menu.get("events") or {})
        named = set(p.get("entities") or {}) | set(p.get("relation_extraction") or {}) | set(p.get("event_extraction") or {})
        n += bool(named - offered)
    return n


def score(records: List[Dict[str, Any]], by_language: bool) -> Dict[str, Any]:
    golds, preds = [r["output"] for r in records], [r["pred"] for r in records]
    metrics = score_predictions(golds, preds, report=False)
    if by_language:
        groups, tiny = T.language_groups(records)
        per_lang = {lang: score_predictions([r["output"] for r in rs], [r["pred"] for r in rs], report=False)
                    for lang, rs in sorted(groups.items())}
        print("\n===== Blind test summary by language =====")
        for lang, m in per_lang.items():
            _print_micro_report(m, label=lang)
        metrics.update(T.by_language_block(per_lang, tiny))
    _print_micro_report(metrics, label="all")
    return metrics


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", required=True, help="an infer.py --output JSONL")
    ap.add_argument("--gold", choices=["auto", "gold", "gold_mapped"], default="auto")
    ap.add_argument("--by-language", action="store_true", help="per-language buckets, as eval.by_language")
    ap.add_argument("--out", help="write the metrics JSON (test_metrics.json shape) here")
    ap.add_argument("--card", action="store_true", help="print the model card's blind-test tables")
    args = ap.parse_args(argv)

    rows = read_rows(args.predictions)
    key = gold_key(rows, args.gold)
    records, dropped = prepare(rows, key)
    print(f"[score] {args.predictions}: {len(rows)} lines, scored {len(records)} against `{key}` "
          f"(dropped {dropped['no_gold']} with no gold, {dropped['duplicate']} exact duplicates)")
    stray = off_menu(records)
    if stray:
        print(f"[score] WARNING: {stray} records predict labels outside their own gold menu -- these predictions "
              f"were not made with --gold-schema, so this is NOT the blind test's number")
    metrics = score(records, args.by_language)
    metrics["scored_from"] = {"predictions": args.predictions, "gold": key, "lines": len(rows),
                              "records": len(records), "dropped": dropped, "off_menu_records": stray}
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[score] wrote {args.out}")
    if args.card:
        print("\n" + _metrics_table(metrics, "Blind test (held-out test splits)"))
        print(_oneie_table(metrics, "OneIE criteria (blind test)"))
        for lang, m in (metrics.get("by_language") or {}).items():
            print(_metrics_table(m, f"Blind test, {lang}"))


if __name__ == "__main__":
    main()
