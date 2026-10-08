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
from model_card import CATEGORIES, ONEIE_CRITERIA, _metric  # noqa: E402

from gliner2.inference.label_map import apply_label_map  # noqa: E402
from gliner2.training.eval_metrics import (  # noqa: E402
    _print_micro_report, _schema_from_gold, parse_menu, project_gold, score_predictions)
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


def load_menu(path: str, labels_file: str = None) -> Dict[str, Any]:
    """An application menu file, its schema mapped into the gold's spellings when ``labels_file`` is
    given -- the same transform ``infer.py --labels-file`` applied to make ``gold_mapped``."""
    menu = json.loads(Path(path).read_text(encoding="utf-8"))
    if labels_file:
        import yaml
        doc = yaml.safe_load(Path(labels_file).read_text(encoding="utf-8")) or {}
        menu["schema"], _ = apply_label_map(menu["schema"], doc.get("labels", doc), doc.get("style"))
    return menu


def prepare(rows: List[Dict[str, Any]], key: str, menu: Dict[str, Any] = None
            ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Records ``{input, output: gold, pred}`` the blind test would score, and what was dropped.

    Under an application ``menu`` the gold is projected onto the menu, and a document with no gold is
    KEPT: it was asked the whole menu, so anything fired on it is a measured false positive."""
    seen, out, dropped = set(), [], {"no_gold": 0, "duplicate": 0}
    for r in rows:
        gold = r.get(key) or {}
        if menu is not None:
            gold = project_gold(gold, menu["schema"])
        elif not _schema_from_gold(gold):
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


def off_menu(records: List[Dict[str, Any]], app: Dict[str, Any] = None) -> int:
    """Records whose prediction names an entity, relation or event type outside the menu they were scored
    under (their own gold, or the application menu): the predictions were made with a different menu."""
    n = 0
    for r in records:
        menu, p = (app["schema"] if app else _schema_from_gold(r["output"])), r["pred"]
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


def metrics_table(metrics: Dict[str, Any], title: str) -> str:
    """The blind test's metrics as a plain table: micro P / R / F1 (strict -> relaxed), support, and the
    OneIE criteria. No model-card prose: the card describes how ITS numbers were made, not this file's."""
    rows = [f"### {title}", "", "| Metric | Precision | Recall | F1 | Support |", "|---|--:|--:|--:|--:|"]
    for c in (c for c in CATEGORIES if _metric(metrics, c, "strict", "f1") is not None):
        def cell(m):
            s, r = _metric(metrics, c, "strict", m), _metric(metrics, c, "relaxed", m)
            return f"{s:.3f} -> {r:.3f}" if r is not None else f"{s:.3f}"
        rows.append(f"| {c} (strict -> relaxed) | {cell('precision')} | {cell('recall')} | {cell('f1')} | "
                    f"{metrics.get(f'eval_{c}_strict_support', '-')} |")
    for key, label, _ in ONEIE_CRITERIA:
        g = lambda m: metrics.get(f"eval_{key}_external_micro_{m}")
        if g("f1") is not None:
            rows.append(f"| {label} (OneIE) | {g('precision'):.3f} | {g('recall'):.3f} | {g('f1'):.3f} | "
                        f"{metrics.get(f'eval_{key}_external_support', '-')} |")
    return "\n".join(rows) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--predictions", required=True, help="an infer.py --output JSONL")
    ap.add_argument("--gold", choices=["auto", "gold", "gold_mapped"], default="auto")
    ap.add_argument("--by-language", action="store_true", help="per-language buckets, as eval.by_language")
    ap.add_argument("--out", help="write the metrics JSON (test_metrics.json shape) here")
    ap.add_argument("--card", action="store_true", help="print the metrics as tables (the blind test's metrics)")
    ap.add_argument("--menu", default="gold", help="gold (default) or app:<menu file>, the menu the predictions "
                                                   "were made with (infer.py --schema-json <menu file>)")
    ap.add_argument("--corpus", help="with --menu app: the corpus this file covers; refused unless the menu "
                                     "is exhaustive for it")
    ap.add_argument("--labels-file", help="with --menu app and gold_mapped: map the menu into the gold's spellings")
    args = ap.parse_args(argv)

    rows = read_rows(args.predictions)
    key = gold_key(rows, args.gold)
    kind, path = parse_menu(args.menu)
    menu = None
    if kind == "app":
        menu = load_menu(path, args.labels_file if key == "gold_mapped" else None)
        if args.corpus not in menu["exhaustive_for"]:
            raise SystemExit(f"[score] menu {menu['name']} is exhaustive for {menu['exhaustive_for']}, not "
                             f"{args.corpus!r}: its gold was never asked this menu's question (pass --corpus)")
        if key == "gold_mapped" and not args.labels_file:
            raise SystemExit("[score] gold_mapped needs --labels-file to map the menu into the same spellings")
    elif kind != "gold":
        raise SystemExit(f"[score] --menu {args.menu}: only gold and app:<file> can be scored from a file")
    records, dropped = prepare(rows, key, menu)
    print(f"[score] {args.predictions}: {len(rows)} lines, scored {len(records)} against `{key}` under menu "
          f"{args.menu} (dropped {dropped['no_gold']} with no gold, {dropped['duplicate']} exact duplicates)")
    stray = off_menu(records, menu)
    if stray:
        print(f"[score] WARNING: {stray} records predict labels outside the menu they are scored under -- the "
              f"predictions were made with a different menu, so this is NOT the {args.menu} number")
    metrics = score(records, args.by_language)
    metrics["scored_from"] = {"predictions": args.predictions, "gold": key, "menu": args.menu, "lines": len(rows),
                              "records": len(records), "dropped": dropped, "off_menu_records": stray}
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[score] wrote {args.out}")
    if args.card:
        src = f"{args.predictions}, gold `{key}`, menu {args.menu}"
        print("\n" + metrics_table(metrics, f"All records ({src})"))
        for lang, m in (metrics.get("by_language") or {}).items():
            print(metrics_table(m, f"{lang} ({src})"))


if __name__ == "__main__":
    main()
