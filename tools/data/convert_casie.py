"""Convert CASIE (Satyapanich et al., AAAI 2020) to GLiNER2 JSONL.

CASIE is a 1,000-document cybersecurity-event corpus with 5 event
subtypes (``Databreach``, ``Phishing``, ``Ransom``,
``Vulnerability-Discover``, ``Vulnerability-Patch``) and typed
arguments. Each argument carries **both** an event-role label
(``Compromised-Data``, ``Attacker``, ``Place``, …) and an entity
``type`` (``PII``, ``Person``, ``Organization``, ``Device``, …) — so
the corpus naturally supports co-training of entity NER and event
extraction.

Source layout (https://github.com/Ebiquity/CASIE)::

    data/source/<doc_id>.txt        — title/source/date markup wrapping <text>
    data/annotation/<doc_id>.json   — annotation:
        info: {title, date, link}
        content: clean document body (matched to char offsets)
        cyberevent.hopper[].events[]:
            nugget: {startOffset, endOffset, text}      — trigger
            subtype: "Databreach" | "Phishing" | ...    — event type
            argument: [
                {startOffset, endOffset, text, type, role:{type}}
            ]

The corpus has no canonical train/dev/test split, so the converter
emits a stratified 80/10/10 split by default using the same greedy
multi-label algorithm as ``convert_ace2005.py`` (rule (b) — rare
categories first to cover train/test/val, then fill toward ratios).
Categories combine entity types (``ent:PII``) and event subtypes
(``evt:Databreach``).

Pass ``--no-stratify`` for single-file output. ``--no-prefix-event``
keeps subtypes bare (``Databreach``); the default prefixes them as
``Cyber.Databreach`` to namespace them apart from other event corpora.

Source code on GitHub. ~10 MB tarball; the converter downloads it once
to a temp directory and parses in-process — no manual download step.

Usage::

    uv run python tools/data/convert_casie.py --out data/casie.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stratify import (  # noqa: E402
    coverage_summary,
    derive_split_paths,
    parse_ratios,
    stratified_split,
)


CASIE_TARBALL_URL = "https://github.com/Ebiquity/CASIE/archive/refs/heads/master.tar.gz"


def _download_tarball(url: str, dest: Path) -> None:
    """Fetch the CASIE tarball into ``dest`` and extract it."""
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    print(f"  downloaded {len(raw) / 1e6:.1f} MB, extracting...")
    tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
    tf.extractall(dest)


def _find_data_root(extract_root: Path) -> Optional[Path]:
    """Locate the ``data/`` directory inside the extracted tarball."""
    # GitHub tarballs extract into a top-level <repo>-<branch> folder.
    for child in extract_root.iterdir():
        candidate = child / "data"
        if candidate.is_dir():
            return candidate
    return None


def parse_annotation(
    annotation_path: Path,
    prefix_event: bool,
) -> Optional[Dict[str, Any]]:
    """Parse one CASIE annotation JSON into a GLiNER2 record; None if unusable."""
    try:
        with annotation_path.open() as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None

    text = data.get("content")
    if not isinstance(text, str) or not text.strip():
        return None

    entities_by_type: Dict[str, List[str]] = {}
    events_out: List[Dict[str, Any]] = []

    cyberevent = data.get("cyberevent") or {}
    hoppers = cyberevent.get("hopper") or []
    for hopper in hoppers:
        if not isinstance(hopper, dict):
            continue
        for ev in hopper.get("events") or []:
            if not isinstance(ev, dict):
                continue
            subtype = ev.get("subtype")
            nugget = ev.get("nugget") or {}
            trigger = nugget.get("text") if isinstance(nugget, dict) else None
            if not isinstance(subtype, str) or not isinstance(trigger, str):
                continue
            subtype = subtype.strip()
            trigger = trigger.strip()
            if not subtype or not trigger or trigger not in text:
                continue
            event_type = f"Cyber.{subtype}" if prefix_event else subtype

            arguments: List[Dict[str, str]] = []
            seen_args: set = set()
            for arg in ev.get("argument") or []:
                if not isinstance(arg, dict):
                    continue
                surface = arg.get("text")
                if not isinstance(surface, str):
                    continue
                surface = surface.strip()
                if not surface or surface not in text:
                    continue
                # Role lives in a nested dict: argument[i].role.type.
                role_obj = arg.get("role") or {}
                role = role_obj.get("type") if isinstance(role_obj, dict) else None
                if not isinstance(role, str) or not role.strip():
                    continue
                role = role.strip()
                key = (role, surface)
                if key in seen_args:
                    continue
                seen_args.add(key)
                arguments.append({"role": role, "entity": surface})

                # The argument's own ``type`` field is the entity type.
                ent_type = arg.get("type")
                if isinstance(ent_type, str) and ent_type.strip():
                    ent_type = ent_type.strip()
                    bucket = entities_by_type.setdefault(ent_type, [])
                    if surface not in bucket:
                        bucket.append(surface)

            events_out.append({
                "event_type": event_type,
                "trigger": trigger,
                "arguments": arguments,
            })

    output: Dict[str, Any] = {}
    if entities_by_type:
        output["entities"] = entities_by_type
    if events_out:
        output["events"] = events_out
    if not output:
        return None
    return {"input": text, "output": output}


def _stats(records: List[Dict[str, Any]]) -> str:
    ent_types: Counter = Counter()
    ev_types: Counter = Counter()
    n_events = 0
    n_args = 0
    for r in records:
        out = r["output"]
        for t, ms in (out.get("entities") or {}).items():
            ent_types[t] += len(ms)
        for ev in out.get("events") or []:
            n_events += 1
            ev_types[ev["event_type"]] += 1
            n_args += len(ev.get("arguments") or [])
    return (
        f"docs={len(records)} entity_types={len(ent_types)} "
        f"entity_surfaces={sum(ent_types.values())} "
        f"events={n_events} event_types={len(ev_types)} "
        f"arguments={n_args}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL base path. Stratified mode writes "
                             "<base>.train.jsonl / .test.jsonl / .val.jsonl; "
                             "with --no-stratify, --out is the single output file.")
    parser.add_argument("--input", type=Path, default=None,
                        help="Path to a pre-extracted CASIE repo root "
                             "(containing data/annotation/ and data/source/). "
                             "If omitted, the tarball is downloaded from GitHub.")
    parser.add_argument("--url", default=CASIE_TARBALL_URL,
                        help="Tarball URL when --input is not provided.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum documents to emit (-1 = all).")
    parser.add_argument("--no-stratify", action="store_true",
                        help="Disable stratified split; write a single file at --out.")
    parser.add_argument("--no-prefix-event", action="store_true",
                        help="Keep event types as bare 'Databreach' / 'Phishing' / "
                             "...; default prefixes them as 'Cyber.<Subtype>' so "
                             "they don't collide with event types from other corpora.")
    parser.add_argument("--split-ratios", type=parse_ratios, default=(0.8, 0.1, 0.1),
                        help="Comma-separated train,test,val ratios (default 0.8,0.1,0.1).")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Seed for the deterministic stratified placement.")
    args = parser.parse_args()

    prefix_event = not args.no_prefix_event

    # ----- locate the annotation directory -----
    if args.input is not None:
        if not args.input.is_dir():
            raise SystemExit(f"--input not a directory: {args.input}")
        data_root = args.input / "data" if (args.input / "data").is_dir() else args.input
        annotation_dir = data_root / "annotation"
        if not annotation_dir.is_dir():
            raise SystemExit(
                f"could not find data/annotation/ under {args.input}"
            )
        tmp_root = None
    else:
        tmp_root = Path(tempfile.mkdtemp(prefix="casie_"))
        _download_tarball(args.url, tmp_root)
        data_root = _find_data_root(tmp_root)
        if data_root is None:
            raise SystemExit(f"no data/ directory found under {tmp_root}")
        annotation_dir = data_root / "annotation"
        if not annotation_dir.is_dir():
            raise SystemExit(f"no data/annotation/ found under {data_root}")

    # ----- collect all records -----
    records: List[Dict[str, Any]] = []
    skipped = 0
    for ann_path in sorted(annotation_dir.glob("*.json")):
        if 0 <= args.max_records <= len(records):
            break
        rec = parse_annotation(ann_path, prefix_event=prefix_event)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)

    print(f"Parsed: {_stats(records)} skipped_no_content={skipped}")

    if tmp_root is not None:
        # The extracted tarball is throwaway; data_root references files
        # inside tmp_root so we leave cleanup to the OS for /tmp.
        pass

    # ----- single-file mode -----
    if args.no_stratify:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote {len(records)} records -> {args.out}")
        return 0

    # ----- stratified mode -----
    train, test, val = stratified_split(
        records, ratios=args.split_ratios, seed=args.split_seed,
    )
    paths = derive_split_paths(args.out)
    paths["train"].parent.mkdir(parents=True, exist_ok=True)
    for split_name, slice_records in (("train", train), ("test", test), ("val", val)):
        with paths[split_name].open("w") as f:
            for rec in slice_records:
                f.write(json.dumps(rec) + "\n")
    print(
        f"Stratified split (ratios={args.split_ratios}): "
        f"train={len(train)} test={len(test)} val={len(val)}\n"
        f"  {coverage_summary(records, train, test, val)}\n"
        f"  -> {paths['train']}, {paths['test']}, {paths['val']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
