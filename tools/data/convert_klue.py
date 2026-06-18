"""Convert KLUE (Korean) NER / RE to GLiNER2 JSONL from the canonical KLUE-benchmark GitHub.

KLUE's HuggingFace loader is broken, so this reads the official release files
directly. License: CC-BY-SA-4.0.

``--task ner``: char-level BIO TSV. The text is the concatenated CHAR column;
each ``B-/I-<type>`` char run becomes an entity surface (types PS, LC, OG, DT,
TI, QT) under ``output.entities``.

``--task re``: JSON records with ``subject_entity`` / ``object_entity`` (word +
type) and a relation ``label``. Emits the two typed entities plus, when the label
is not ``no_relation``, a ``{label: {head, tail}}`` relation.

Usage::

    uv run python tools/data/convert_klue.py --task ner --out data/klue_ner.jsonl
    uv run python tools/data/convert_klue.py --task re  --out data/klue_re.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, add_split_args

_BASE = "https://raw.githubusercontent.com/KLUE-benchmark/KLUE/main/klue_benchmark"
URLS = {
    "ner": f"{_BASE}/klue-ner-v1.1/klue-ner-v1.1_train.tsv",
    "re": f"{_BASE}/klue-re-v1.1/klue-re-v1.1_train.json",
}


def _fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=180) as resp:
        return resp.read().decode("utf-8")


def _char_bio_entities(chars: list, tags: list) -> tuple[str, dict]:
    """Char-level BIO -> (text, {type: [surface]}); surfaces are char-joined (no spaces)."""
    text = "".join(chars)
    entities: dict[str, list[str]] = {}
    spans: list[tuple[str, int, int]] = []
    cur: str | None = None
    start = 0
    for i, tag in enumerate(tags):
        if not tag or tag == "O":
            if cur is not None:
                spans.append((cur, start, i))
                cur = None
            continue
        prefix, _, typ = tag.partition("-") if "-" in tag else ("B", "", tag)
        typ = typ or prefix
        if prefix == "B" or typ != cur:
            if cur is not None:
                spans.append((cur, start, i))
            cur, start = typ, i
    if cur is not None:
        spans.append((cur, start, len(chars)))
    for typ, s, e in spans:
        surface = "".join(chars[s:e]).strip()
        if not surface:
            continue
        bucket = entities.setdefault(typ, [])
        if surface not in bucket:
            bucket.append(surface)
    return text, entities


def iter_ner(tsv: str):
    chars: list[str] = []
    tags: list[str] = []
    for line in tsv.split("\n"):
        if line.startswith("##"):
            continue
        if line.strip() == "" and "\t" not in line:
            if chars:
                yield _char_bio_entities(chars, tags)
                chars, tags = [], []
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        chars.append(parts[0])
        tags.append(parts[1])
    if chars:
        yield _char_bio_entities(chars, tags)


def iter_re(jtext: str):
    for rec in json.loads(jtext):
        sent = rec.get("sentence")
        subj = rec.get("subject_entity") or {}
        obj = rec.get("object_entity") or {}
        sw, ow = subj.get("word"), obj.get("word")
        if not isinstance(sent, str) or not sw or not ow:
            continue
        entities: dict[str, list[str]] = {}
        for word, typ in ((sw, subj.get("type")), (ow, obj.get("type"))):
            if isinstance(typ, str) and isinstance(word, str) and word in sent:
                bucket = entities.setdefault(typ, [])
                if word not in bucket:
                    bucket.append(word)
        output: dict = {}
        if entities:
            output["entities"] = entities
        label = rec.get("label")
        if isinstance(label, str) and label and label != "no_relation" and sw in sent and ow in sent:
            output["relations"] = [{label: {"head": sw, "tail": ow}}]
        if output:
            yield sent, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, choices=["ner", "re"])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path (writes <base>.{train,val,test}.jsonl).")
    parser.add_argument("--url", default=None, help="Override the source URL (else the canonical train file).")
    parser.add_argument("--max-records", type=int, default=-1, help="Max records to emit.")
    add_split_args(parser)
    args = parser.parse_args()

    url = args.url or URLS[args.task]
    print(f"Fetching {url} ...")
    raw = _fetch(url)
    it = iter_ner(raw) if args.task == "ner" else iter_re(raw)

    emitted = total_entities = total_relations = 0
    types: set[str] = set()
    with SplitWriter(args.out, ratios=args.split_ratios, seed=args.split_seed) as writer:
        for text, output in it:
            if args.task == "ner" and not output:
                continue
            ents = output.get("entities", {}) if args.task == "re" else output
            out = {"entities": ents} if args.task == "ner" else output
            writer.write({"input": text, "output": out})
            emitted += 1
            total_entities += sum(len(v) for v in (out.get("entities") or {}).values())
            total_relations += len(out.get("relations") or [])
            types.update((out.get("entities") or {}).keys())
            if 0 <= args.max_records <= emitted:
                break

    print(f"Done. task={args.task} emitted={emitted} total_entities={total_entities} "
          f"total_relations={total_relations} entity_types={sorted(types)} {writer.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
