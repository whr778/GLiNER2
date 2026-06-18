"""Convert BioRED (biomedical NER + relations) to GLiNER2 JSONL from the NCBI release.

Canonical source: https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip (~2 MB,
read from the BioC.JSON inside). Pass ``--zip`` to use a local copy.

Entity types: GeneOrGeneProduct, DiseaseOrPhenotypicFeature, ChemicalEntity,
SequenceVariant, CellLine, OrganismTaxon. Document-level relations link normalized
entity *identifiers* (Association, Positive_Correlation, ...), so each relation's
head/tail is a representative mention of that identifier. Maps to
``{type: [mention]}`` entities + ``{rel_type: {head, tail}}`` relations.

License: NCBI / U.S. National Library of Medicine (see the bundled README.txt).

Usage::

    uv run python tools/data/convert_biored.py --out data/biored.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, add_split_args

URL = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip"
MEMBER = {"train": "BioRED/Train.BioC.JSON",
          "dev": "BioRED/Dev.BioC.JSON",
          "test": "BioRED/Test.BioC.JSON"}


def convert_doc(doc: dict) -> dict | None:
    text_parts: list[str] = []
    entities: dict[str, list[str]] = {}
    id2surface: dict[str, str] = {}  # normalized identifier -> representative mention

    for passage in doc.get("passages", []):
        txt = passage.get("text") or ""
        if txt:
            text_parts.append(txt)
        for ann in passage.get("annotations", []):
            infons = ann.get("infons") or {}
            typ = infons.get("type")
            ident = infons.get("identifier")
            surface = (ann.get("text") or "").strip()
            if not isinstance(typ, str) or not surface:
                continue
            bucket = entities.setdefault(typ, [])
            if surface not in bucket:
                bucket.append(surface)
            if isinstance(ident, str) and ident and ident not in id2surface:
                id2surface[ident] = surface

    if not entities:
        return None
    text = " ".join(text_parts)

    relations = []
    for rel in doc.get("relations", []):
        infons = rel.get("infons") or {}
        typ = infons.get("type")
        head = id2surface.get(infons.get("entity1"))
        tail = id2surface.get(infons.get("entity2"))
        if isinstance(typ, str) and head and tail and head != tail:
            relations.append({typ: {"head": head, "tail": tail}})

    output: dict = {"entities": entities}
    if relations:
        output["relations"] = relations
    return {"input": text, "output": output}


def _load_documents(args) -> list:
    if args.zip:
        data = Path(args.zip).read_bytes()
    else:
        print(f"Downloading {URL} ...")
        with urllib.request.urlopen(URL, timeout=180) as resp:
            data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open(MEMBER[args.split]) as f:
            return json.load(f)["documents"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path (writes <base>.{train,val,test}.jsonl).")
    parser.add_argument("--zip", default=None, help="Local BIORED.zip (skips the download).")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--max-records", type=int, default=-1, help="Max docs to emit.")
    add_split_args(parser)
    args = parser.parse_args()

    docs = _load_documents(args)
    emitted = total_entities = total_relations = 0
    ent_types: set[str] = set()
    rel_types: set[str] = set()
    with SplitWriter(args.out, ratios=args.split_ratios, seed=args.split_seed) as writer:
        for doc in docs:
            record = convert_doc(doc)
            if record is None:
                continue
            writer.write(record)
            emitted += 1
            ents = record["output"].get("entities") or {}
            rels = record["output"].get("relations") or []
            total_entities += sum(len(v) for v in ents.values())
            total_relations += len(rels)
            ent_types.update(ents.keys())
            rel_types.update(next(iter(r)) for r in rels)
            if 0 <= args.max_records <= emitted:
                break

    print(f"Done. emitted={emitted} total_entities={total_entities} total_relations={total_relations} "
          f"entity_types={sorted(ent_types)} relation_types={sorted(rel_types)} {writer.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
