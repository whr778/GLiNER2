"""Control arm for base-word augmentation: duplicate WITHOUT lemmatizing.

`augment_baseword.py` emits a second copy of each record in base-word form, so the
augmented corpus is 1.81x the size of the original. A gain measured against the
un-augmented baseline therefore has two possible causes, and they are not separable:
the model saw those records **twice**, or it saw them **lemmatized**.

This builds the arm that separates them. For every record the augmentation actually
duplicated, emit the original twice instead of original + lemma. The result has the same
record count, the same records duplicated, and the same whitespace normalization -- the
only remaining difference from the augmented corpus is lemmatization itself.

    uv run python tools/data/build_duplicate_control.py \\
        --in data/rams.train.jsonl --backend simplemma --lang en \\
        --out datasets/rams_baseword/train.duplicate_control.jsonl

The copy is whitespace-normalized (`" ".join(text.split())`) because the augmented copy is
a join of its token list. Leaving the original spacing here would reintroduce a second
difference between the arms, which is what this file exists to remove.

**Which records to duplicate is RECOMPUTED, not read off the augmented file.** Reading it
back cannot work: 794 of RAMS's lemma copies are textually identical to their originals
(nothing lemmatized, whitespace already normalized), so they are indistinguishable from an
original in the output and the count comes out 13,389 against the true 13,291. Calling
`augment()` reproduces the emit decision exactly, so the arms match by construction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _split import dumps_record
from augment_baseword import augment, mock_lemma, simplemma_lemma


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=("mock", "simplemma"), default="simplemma",
                    help="must match the backend the augmented arm was built with, or the "
                         "two arms duplicate different records")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args(argv)

    lemma = simplemma_lemma if args.backend == "simplemma" else mock_lemma
    n = duplicated = 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.src).open(encoding="utf-8") as fh, out.open("w", encoding="utf-8") as sink:
        for line in fh:
            record = json.loads(line)
            n += 1
            sink.write(dumps_record(record) + "\n")
            if augment(record, lemma, args.lang) is None:
                continue          # refused there, so not duplicated here either
            copy = dict(record)
            copy["input"] = " ".join(record["input"].split())
            sink.write(dumps_record(copy) + "\n")
            duplicated += 1

    print(f"[control] {n} originals, {duplicated} duplicated -> {n + duplicated} written")
    print(f"[control] wrote {out}")


if __name__ == "__main__":
    main()
