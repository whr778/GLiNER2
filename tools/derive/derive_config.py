"""Measure a corpus against a pretrained base and write the calibration behind a config.

    uv run python tools/derive/derive_config.py --corpus data/ace2005_v3 \\
        --model whr778/gliner2-eb18-balanced --out derived/

Writes, per (corpus, model):
  <name>.calibration.json   every measurement, with the model facts it was measured against
  <name>.labels_review.md   labels a human must confirm before training
See tools/derive/README.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_probe import probe_corpus  # noqa: E402
from label_review import review  # noqa: E402
from model_probe import probe_model  # noqa: E402


def run_name(corpus: str, model: str) -> str:
    return f"{Path(corpus).name}__{Path(model.rstrip('/')).name}"


def write_review(path: Path, name: str, rows_by_cat: dict) -> None:
    lines = [f"# Label review: {name}", "",
             "Confirm every `fold_match` by reading the surfaces it tags; decide a natural-language",
             "name for every `open_vocab_code_like` label. `known`/`mapped` need no action.", ""]
    for cat, rows in rows_by_cat.items():
        if not rows:
            continue
        lines += [f"## {cat}  ({dict(Counter(r['status'] for r in rows))})", "",
                  "| label | uses | status | target |", "|---|---|---|---|"]
        lines += [f"| `{r['label']}` | {r['uses']} | {r['status']} | {r.get('target', '')} |" for r in rows]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=True, help="split base: <base>.{train,val,test}.jsonl")
    ap.add_argument("--model", required=True, help="Hub id or local checkpoint dir")
    ap.add_argument("--out", type=Path, default=Path("derived"))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    model_info = probe_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer_source"])
    event_records = bool(model_info["boundary_head"].get("event_records"))
    corpus = probe_corpus(args.corpus, tokenizer, event_records)
    labels = review(corpus["labels"], model_info)

    name = run_name(args.corpus, args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    calib = {"model": {k: v for k, v in model_info.items() if k not in ("label_map", "boundary_head")},
             "corpus": corpus, "label_review": labels}
    (args.out / f"{name}.calibration.json").write_text(
        json.dumps(calib, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    write_review(args.out / f"{name}.labels_review.md", name, labels)

    u = corpus["uniqueness"]
    dups = {s: u[f"{s}_records"] - u[f"{s}_unique_inputs"] for s in ("train", "val", "test")}
    overlap = {k: u[k] for k in ("train&val", "train&test", "val&test")}
    print(f"[derive] {name}: arch={model_info['architecture']} encoder={model_info['encoder']} "
          f"max_len={model_info['max_len']}")
    print(f"[derive] split hygiene: duplicates {dups} overlap {overlap}"
          + ("  *** NOT CLEAN ***" if any(dups.values()) or any(overlap.values()) else "  clean"))
    print(f"[derive] gold capacity: {corpus['gold_capacity']}")
    print(f"[derive] wrote {args.out / (name + '.calibration.json')} and {name}.labels_review.md")


if __name__ == "__main__":
    main()
