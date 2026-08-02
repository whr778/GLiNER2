"""Convert MasakhaNER 2.0 (masakhane/masakhaner2) to GLiNER2 entity JSONL.

MasakhaNER 2.0 (Adelani et al., EMNLP 2022) is a human-annotated NER benchmark
covering 20 African languages, each a HuggingFace config with OFFICIAL
train/validation/test splits and four entity types: PER, ORG, LOC, DATE.

Pick languages with ``--langs`` (comma-separated ISO 639-3 codes) or ``all``
(the default). Each row's ``tokens`` + ``ner_tags`` fold into GLiNER2 entities
with ``bio_to_entities`` (shared with convert_hf_token_ner). Terse source tags
map to natural-language type names by default (PER -> person, ORG ->
organization, LOC -> location, DATE -> date); pass ``--raw-labels`` to keep the
source tags.

For each selected language the official splits are written to
    data/masakhaner_<lang>.{train,val,test}.jsonl
and every selected language is also merged into
    data/masakhaner.{train,val,test}.jsonl

so a training config can pick one language (``data/masakhaner_hau``), a subset,
or all of them at once (``data/masakhaner``).

Usage::

    uv run python tools/data/convert_masakhaner.py --out data/masakhaner.jsonl
    uv run python tools/data/convert_masakhaner.py --out data/masakhaner.jsonl \\
        --langs hau,yor,swa
    uv run python tools/data/convert_masakhaner.py --out data/masakhaner.jsonl \\
        --langs all --raw-labels
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402
from convert_hf_token_ner import bio_to_entities  # noqa: E402

HF_ID = "masakhane/masakhaner2"
# The upstream repo is a (no-longer-supported) dataset script; load the
# datasets-server auto-converted parquet export instead.
REVISION = "refs/convert/parquet"
# HF split name -> our output split suffix.
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}

# The fixed ner_tags ClassLabel ordering (same for every language), used as a
# fallback if the parquet features do not carry the ClassLabel names.
TAG_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG",
             "B-LOC", "I-LOC", "B-DATE", "I-DATE"]

# The 20 languages, ISO 639-3 code -> English name (order = the paper's).
LANGUAGES: Dict[str, str] = {
    "bam": "Bambara", "bbj": "Ghomala", "ewe": "Ewe", "fon": "Fon",
    "hau": "Hausa", "ibo": "Igbo", "kin": "Kinyarwanda", "lug": "Luganda",
    "luo": "Luo", "mos": "Mossi", "nya": "Chichewa", "pcm": "Nigerian-Pidgin",
    "sna": "Shona", "swa": "Swahili", "tsn": "Setswana", "twi": "Twi",
    "wol": "Wolof", "xho": "isiXhosa", "yor": "Yoruba", "zul": "isiZulu",
}

# Terse source tags -> GLiNER2 natural-language type names.
LABEL_MAP = {"PER": "person", "ORG": "organization",
             "LOC": "location", "DATE": "date"}


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


def _id2name(features) -> List[str]:
    """id -> tag-name list from the ner_tags ClassLabel feature, else TAG_NAMES."""
    inner = getattr(features["ner_tags"], "feature", None)
    return getattr(inner, "names", None) or TAG_NAMES


def load_lang(hf_id: str, lang: str, revision: str):
    """Load one language's official train/validation/test splits as a DatasetDict."""
    from huggingface_hub import hf_hub_download
    from datasets import load_dataset
    files = {sp: hf_hub_download(hf_id, f"{lang}/{sp}/0000.parquet",
                                 repo_type="dataset", revision=revision)
             for sp in SPLIT_MAP}
    return load_dataset("parquet", data_files=files)


def row_to_record(row: Dict[str, Any], id2name: List[str], raw_labels: bool):
    """Fold one tokens+ner_tags row into a GLiNER2 entity record, or None."""
    tokens = [str(t) for t in (row.get("tokens") or [])]
    tag_ids = row.get("ner_tags") or []
    tags = [id2name[t] if 0 <= t < len(id2name) else "O" for t in tag_ids]
    text, entities = bio_to_entities(tokens, tags)
    if not entities:
        return None
    if not raw_labels:
        entities = {LABEL_MAP.get(k, k.lower()): v for k, v in entities.items()}
    return {"input": text, "output": {"entities": entities}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Combined output base path; also derives per-language "
                             "<stem>_<lang>.{train,val,test}.jsonl.")
    parser.add_argument("--langs", type=parse_langs, default="all",
                        help="Comma-separated ISO 639-3 codes or 'all' (default).")
    parser.add_argument("--raw-labels", action="store_true",
                        help="Keep source tags (PER/ORG/LOC/DATE) instead of mapping "
                             "them to person/organization/location/date.")
    parser.add_argument("--hf-id", default=HF_ID,
                        help=f"HuggingFace dataset id (default: {HF_ID}).")
    parser.add_argument("--revision", default=REVISION,
                        help=f"Dataset revision (default: {REVISION}).")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Max records per language-split to emit (-1 = all).")
    args = parser.parse_args()
    langs = args.langs if isinstance(args.langs, list) else parse_langs(args.langs)

    stem = args.out.with_suffix("") if args.out.suffix == ".jsonl" else args.out
    stem.parent.mkdir(parents=True, exist_ok=True)

    combined = {s: Path(f"{stem}.{s}.jsonl").open("w", encoding="utf-8")
                for s in SPLIT_MAP.values()}
    totals = {s: 0 for s in SPLIT_MAP.values()}
    try:
        for lang in langs:
            print(f"Loading {args.hf_id} [{lang}] ({LANGUAGES[lang]}) ...")
            ds = load_lang(args.hf_id, lang, args.revision)
            id2name = _id2name(ds["train"].features)
            per_counts = {s: 0 for s in SPLIT_MAP.values()}
            for hf_split, out_split in SPLIT_MAP.items():
                if hf_split not in ds:
                    continue
                out_path = Path(f"{stem}_{lang}.{out_split}.jsonl")
                with out_path.open("w", encoding="utf-8") as f:
                    n = 0
                    for row in ds[hf_split]:
                        if 0 <= args.max_records <= n:
                            break
                        rec = row_to_record(row, id2name, args.raw_labels)
                        if rec is None:
                            continue
                        line = dumps_record(rec) + "\n"
                        f.write(line)
                        combined[out_split].write(line)
                        n += 1
                per_counts[out_split] = n
                totals[out_split] += n
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
