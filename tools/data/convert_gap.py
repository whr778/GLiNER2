"""Convert GAP (Gendered Ambiguous Pronouns, Webster et al. 2018) to GLiNER2 JSONL coreference records.

    uv run python tools/data/convert_gap.py --raw-dir data/coref_raw/GAP/data

GAP (Hub: coref-data/gap_raw, Apache-2.0) gives a Wikipedia passage, one ambiguous pronoun and two
candidate names A and B, each labelled coreferent with the pronoun or not. Each row becomes:

    {"input": text, "output": {}, "coref_clusters": [[pronoun, antecedent], [other candidate]],
     "provenance": {...}}

so the pronoun-antecedent pair is a positive and the pronoun-distractor pair a hard negative. When
neither name is coreferent, all three are singletons. Only the labelled mentions are clusters: other
mentions of A or B in the passage are not annotated. Offsets are code points into the cleaned text.
Splits are GAP's own: development -> train, validation -> val, test -> test.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import clean_text, dumps_record  # noqa: E402
from add_coref_clusters import place  # noqa: E402

SPLITS = {"train": "train", "val": "validation", "test": "test"}


def to_record(row, counts: Dict[str, int]) -> Dict:
    raw = row["Text"]
    p = (row["Pronoun"], row["Pronoun-offset"], row["Pronoun-offset"] + len(row["Pronoun"]))
    a = (row["A"], row["A-offset"], row["A-offset"] + len(row["A"]))
    b = (row["B"], row["B-offset"], row["B-offset"] + len(row["B"]))
    if row["A-coref"]:
        clusters = [[p, a], [b]]
    elif row["B-coref"]:
        clusters = [[p, b], [a]]
    else:
        clusters = [[p], [a], [b]]
    return {"input": raw, "output": {}, "coref_clusters": place(clean_text(raw), clusters, clean_text, raw, counts),
            "provenance": {"source": "GAP (coref-data/gap_raw)", "doc_id": row["ID"], "url": row["URL"],
                           "license": "Apache-2.0"}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", type=Path, default=Path("data/coref_raw/GAP/data"))
    ap.add_argument("--name", default="gap_coref")
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    args = ap.parse_args()
    if not any(args.raw_dir.glob("*.parquet")):
        from huggingface_hub import snapshot_download
        snapshot_download("coref-data/gap_raw", repo_type="dataset", local_dir=args.raw_dir.parent)
    for split, src in SPLITS.items():
        df = pd.read_parquet(next(args.raw_dir.glob(f"{src}-*.parquet")))
        counts = {"kept": 0, "dropped": 0}
        path = args.out_dir / f"{args.name}.{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(dumps_record(to_record(row, counts)) + "\n")
        print(f"{path}: rows {len(df)}, mentions kept {counts['kept']}, dropped {counts['dropped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
