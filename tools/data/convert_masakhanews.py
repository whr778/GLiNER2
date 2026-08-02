"""Convert MasakhaNEWS (masakhane/masakhanews) to GLiNER2 classification JSONL.

MasakhaNEWS (Adelani et al., IJCNLP-AACL 2023) is a news-topic classification
benchmark covering 16 African languages, each a HuggingFace config with OFFICIAL
train/validation/test splits. Every article carries a single ``category`` topic
label; the union across languages is a 7-topic ontology (business,
entertainment, health, politics, religion, sports, technology), so this maps to
GLiNER2's single-label classification head::

    {"input": <headline + text>,
     "output": {"classifications": [
         {"task": "news topic", "labels": <union topic set>, "true_label": [<category>]}
     ]}}

The candidate ``labels`` set is the union of categories over the SELECTED
languages, so every record shares one consistent schema (a language whose data
lacks a topic still offers it as a candidate negative).

Pick languages with ``--langs`` (comma-separated ISO 639-3 codes) or ``all``
(the default). For each selected language the official splits are written to
    data/masakhanews_<lang>.{train,val,test}.jsonl
and every selected language is also merged into
    data/masakhanews.{train,val,test}.jsonl

Usage::

    uv run python tools/data/convert_masakhanews.py --out data/masakhanews.jsonl
    uv run python tools/data/convert_masakhanews.py --out data/masakhanews.jsonl \\
        --langs swa,hau,yor
    uv run python tools/data/convert_masakhanews.py --out data/masakhanews.jsonl \\
        --text-field headline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402

HF_ID = "masakhane/masakhanews"
# Upstream is a (no-longer-supported) dataset script; load the auto-converted
# parquet export instead.
REVISION = "refs/convert/parquet"
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}

# The 16 languages, ISO 639-3 code -> English name.
LANGUAGES: Dict[str, str] = {
    "amh": "Amharic", "eng": "English", "fra": "French", "hau": "Hausa",
    "ibo": "Igbo", "lin": "Lingala", "lug": "Luganda", "orm": "Oromo",
    "pcm": "Nigerian-Pidgin", "run": "Rundi", "sna": "Shona", "som": "Somali",
    "swa": "Swahili", "tir": "Tigrinya", "xho": "isiXhosa", "yor": "Yoruba",
}


def parse_langs(spec: str) -> List[str]:
    """Resolve a --langs value ('all' or a comma-separated code list)."""
    if spec.strip().lower() == "all":
        return list(LANGUAGES)
    picked = [c.strip() for c in spec.split(",") if c.strip()]
    unknown = [c for c in picked if c not in LANGUAGES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown language code(s) {unknown}; choose from {sorted(LANGUAGES)} or 'all'"
        )
    return picked


def load_lang(hf_id: str, lang: str, revision: str):
    """Load one language's official train/validation/test splits as a DatasetDict."""
    from huggingface_hub import hf_hub_download
    from datasets import load_dataset
    files = {sp: hf_hub_download(hf_id, f"{lang}/{sp}/0000.parquet",
                                 repo_type="dataset", revision=revision)
             for sp in SPLIT_MAP}
    return load_dataset("parquet", data_files=files)


def build_text(row: Dict[str, Any], text_field: str) -> str:
    """Assemble the classification input from headline and/or body."""
    headline = (row.get("headline") or "").strip()
    body = (row.get("text") or "").strip()
    if text_field == "headline":
        return headline
    if text_field == "text":
        return body
    return f"{headline}\n\n{body}".strip() if headline and body else (headline or body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Combined output base path; also derives per-language "
                             "<stem>_<lang>.{train,val,test}.jsonl.")
    parser.add_argument("--langs", type=parse_langs, default="all",
                        help="Comma-separated ISO 639-3 codes or 'all' (default).")
    parser.add_argument("--text-field", choices=["both", "text", "headline"],
                        default="both",
                        help="Which field(s) form the input (default: both).")
    parser.add_argument("--task-name", default="news topic",
                        help="Classification task name written into each record.")
    parser.add_argument("--hf-id", default=HF_ID,
                        help=f"HuggingFace dataset id (default: {HF_ID}).")
    parser.add_argument("--revision", default=REVISION,
                        help=f"Dataset revision (default: {REVISION}).")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Max records per language-split to keep (-1 = all).")
    args = parser.parse_args()
    langs = args.langs if isinstance(args.langs, list) else parse_langs(args.langs)

    stem = args.out.with_suffix("") if args.out.suffix == ".jsonl" else args.out
    stem.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: load selected languages and collect the union label set.
    per_lang: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
    label_set: set = set()
    for lang in langs:
        print(f"Loading {args.hf_id} [{lang}] ({LANGUAGES[lang]}) ...")
        ds = load_lang(args.hf_id, lang, args.revision)
        per_lang[lang] = {}
        for hf_split, out_split in SPLIT_MAP.items():
            rows: List[Tuple[str, str]] = []
            for row in ds[hf_split]:
                if 0 <= args.max_records <= len(rows):
                    break
                text = build_text(row, args.text_field)
                cat = (row.get("category") or "").strip()
                if not text or not cat:
                    continue
                rows.append((text, cat))
                label_set.add(cat)
            per_lang[lang][out_split] = rows

    labels = sorted(label_set)
    print(f"\nLabel set ({len(labels)}): {labels}")

    # Pass 2: write per-language + combined with the shared label schema.
    combined = {s: Path(f"{stem}.{s}.jsonl").open("w", encoding="utf-8")
                for s in SPLIT_MAP.values()}
    totals = {s: 0 for s in SPLIT_MAP.values()}
    try:
        for lang in langs:
            per_counts = {s: 0 for s in SPLIT_MAP.values()}
            for out_split, rows in per_lang[lang].items():
                out_path = Path(f"{stem}_{lang}.{out_split}.jsonl")
                with out_path.open("w", encoding="utf-8") as f:
                    for text, cat in rows:
                        rec = {"input": text, "output": {"classifications": [
                            {"task": args.task_name, "labels": labels,
                             "true_label": [cat]}]}}
                        line = dumps_record(rec) + "\n"
                        f.write(line)
                        combined[out_split].write(line)
                per_counts[out_split] = len(rows)
                totals[out_split] += len(rows)
            print(f"  {lang}: train={per_counts['train']} "
                  f"val={per_counts['val']} test={per_counts['test']}")
    finally:
        for f in combined.values():
            f.close()

    print(f"\nDone. {len(langs)} language(s): {', '.join(langs)}")
    print(f"Combined -> {stem}.{{train,val,test}}.jsonl : "
          f"train={totals['train']} val={totals['val']} test={totals['test']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
