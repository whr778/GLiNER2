"""Convert a CorefUD CoNLL-U corpus (coreference in Universal Dependencies) to GLiNER2 JSONL.

    uv run python tools/data/convert_corefud.py --corefud English-GUM --name gum_coref

CorefUD (https://ufal.mff.cuni.cz/corefud) puts many languages' coreference corpora in one format;
the 1.2 mirror is FrancophonIA/CorefUD_1.2 on the Hub. Each document becomes one record:

    {"input": text, "output": {}, "coref_clusters": [[{"text", "start", "end"}, ...], ...],
     "provenance": {"source", "doc_id", "genre", "license"}}

Clusters keep every mention, pronouns and singletons included, with code-point offsets into the
CLEANED text ``dumps_record`` writes (offsets mapped through the cleaning, as add_coref_clusters.py
does). ``output`` is empty: these corpora supervise coreference only.

Mentions on empty nodes (zero pronouns) have no surface and are skipped; discontinuous mentions
(``e5[1/2]``) are skipped and counted. Only train and dev are public, so dev becomes TEST and
``--val-fraction`` of train documents (seeded) becomes VAL.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import clean_text, dumps_record  # noqa: E402
from add_coref_clusters import place  # noqa: E402

BRACKET = re.compile(r"\(([^()]+)\)|\(([^()]+)|([^()]+)\)")


MIRROR = "FrancophonIA/CorefUD_1.2"


def fetch(corpus: str, dest: Path) -> Path:
    """Download one CorefUD corpus (train/dev .conllu, LICENSE, README) from the Hub mirror into ``dest``."""
    import shutil
    from huggingface_hub import hf_hub_download, list_repo_files
    out = dest / corpus
    out.mkdir(parents=True, exist_ok=True)
    for f in list_repo_files(MIRROR, repo_type="dataset"):
        if f.startswith(f"data/CorefUD_{corpus}/"):
            shutil.copy(hf_hub_download(MIRROR, f, repo_type="dataset"), out / Path(f).name)
    return out


def _misc(field: str) -> Dict[str, str]:
    return dict(kv.split("=", 1) for kv in field.split("|") if "=" in kv) if field != "_" else {}


def _eid(spec: str) -> str:
    return spec.split("-", 1)[0]


def read_docs(path: Path) -> List[Dict]:
    """[{doc_id, genre, text, mentions: [(eid, start, end)], skipped}] from one .conllu file."""
    docs, doc = [], None
    pieces: List[str] = []
    open_: Dict[str, List[int]] = {}
    mwt_end, mwt_space = 0, True
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# newdoc"):
            doc = {"doc_id": line.split("=", 1)[1].strip(), "genre": None, "mentions": [], "skipped": 0}
            docs.append(doc)
            pieces, open_ = [], {}
            doc["pieces"] = pieces
            continue
        if line.startswith("# meta::genre"):
            doc["genre"] = line.split("=", 1)[1].strip()
            continue
        if line.startswith("# newpar") and pieces and not line.startswith("# newpar_block"):
            pieces.append("\n")
            continue
        if not line or line.startswith("#"):
            if not line and pieces and pieces[-1] not in (" ", "\n"):
                pieces.append(" ")
            continue
        cols = line.split("\t")
        tid, form, misc = cols[0], cols[1], _misc(cols[9])
        if "." in tid:
            doc["skipped"] += sum(1 for m in BRACKET.finditer(misc.get("Entity", "")) if m.group(1) or m.group(2))
            continue
        if "-" in tid:
            a, b = (int(x) for x in tid.split("-"))
            mwt_end, mwt_space = b, misc.get("SpaceAfter") != "No"
            start = sum(len(p) for p in pieces)
            pieces.append(form)
            doc["_mwt"] = (start, start + len(form), form, 0)
            continue
        n = int(tid)
        if n <= mwt_end and "_mwt" in doc:
            ms, me, mform, cur = doc["_mwt"]
            if mform[cur:].lower().startswith(form.lower()):   # "It's" -> "It" + "'s": place the word inside the token
                s, e = ms + cur, ms + cur + len(form)
                doc["_mwt"] = (ms, me, mform, cur + len(form))
            else:
                s, e = ms, me
            if n == mwt_end:
                if mwt_space:
                    pieces.append(" ")
                doc.pop("_mwt")
        else:
            s = sum(len(p) for p in pieces)
            pieces.append(form)
            e = s + len(form)
            if misc.get("SpaceAfter") != "No":
                pieces.append(" ")
        for m in BRACKET.finditer(misc.get("Entity", "")):
            single, opening, closing = m.groups()
            spec = single or opening or closing
            if "[" in _eid(spec):
                doc["skipped"] += bool(single or opening)
                continue
            if single:
                doc["mentions"].append((_eid(spec), s, e))
            elif opening:
                open_.setdefault(_eid(spec), []).append(s)
            elif open_.get(_eid(spec)):
                doc["mentions"].append((_eid(spec), open_[_eid(spec)].pop(), e))
    for d in docs:
        d["text"] = "".join(d.pop("pieces"))
    return docs


def to_record(doc: Dict, source: str, licenses: Dict[str, str], counts: Dict[str, int]) -> Dict:
    raw = doc["text"]
    by_eid: Dict[str, List[Tuple[str, int, int]]] = {}
    for eid, s, e in sorted(doc["mentions"], key=lambda m: (m[1], m[2])):
        by_eid.setdefault(eid, []).append((raw[s:e], s, e))
    clusters = place(clean_text(raw), list(by_eid.values()), clean_text, raw, counts)
    lic = licenses.get(doc["genre"] or "", licenses.get("*", "see source"))
    return {"input": raw, "output": {}, "coref_clusters": clusters,
            "provenance": {"source": source, "doc_id": doc["doc_id"], "genre": doc["genre"], "license": lic}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--conllu-dir", type=Path, help="A local CorefUD corpus directory.")
    src.add_argument("--corefud", help=f"A corpus name on {MIRROR} (e.g. English-GUM), downloaded to --raw-dir.")
    p.add_argument("--raw-dir", type=Path, default=Path("data/coref_raw"))
    p.add_argument("--name", required=True, help="Output corpus name: data/<name>.{train,val,test}.jsonl")
    p.add_argument("--source", default=None, help="Provenance source label (default: the directory name).")
    p.add_argument("--license", action="append", default=[],
                   help="genre=license (repeatable); '*=license' for every document.")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    args = p.parse_args()
    licenses = dict(x.split("=", 1) for x in args.license)
    if args.corefud:
        args.conllu_dir = fetch(args.corefud, args.raw_dir)
    source = args.source or f"CorefUD 1.2 {args.conllu_dir.name}"
    train = read_docs(next(args.conllu_dir.glob("*-train.conllu")))
    test = read_docs(next(args.conllu_dir.glob("*-dev.conllu")))
    random.Random(0).shuffle(train)
    k = max(1, round(len(train) * args.val_fraction))
    for split, docs in (("train", train[k:]), ("val", train[:k]), ("test", test)):
        counts = {"kept": 0, "dropped": 0}
        recs = [to_record(d, source, licenses, counts) for d in docs]
        path = args.out_dir / f"{args.name}.{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(dumps_record(r) + "\n")
        skipped = sum(d["skipped"] for d in docs)
        print(f"{path}: docs {len(recs)}, clusters {sum(len(r['coref_clusters']) for r in recs)}, "
              f"mentions kept {counts['kept']}, dropped {counts['dropped']}, skipped (empty node / discontinuous) {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
