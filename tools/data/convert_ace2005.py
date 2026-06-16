"""Convert ACE 2005 English event annotations to GLiNER2 events JSONL.

ACE 2005 (LDC2006T06) is the canonical event-extraction benchmark — 33
event subtypes under 8 main types, with typed arguments. The corpus is
licensed; this script never downloads it. You provide the path to your
local copy and the converter walks the directory tree extracting events.

Source layout — typical LDC delivery::

    ace_2005_td_v7/
      data/
        English/                 (or any locale code)
          bc/ bn/ nw/ ... /      (genre dirs)
            adj/ timex2norm/     (annotation dirs)
              CNN_CF_*.sgm       (text)
              CNN_CF_*.apf.xml   (annotations)

Each ``.apf.xml`` file pairs with a ``.sgm`` of the same stem (sans
suffix). For each event mention, we record:

* ``event_type``: ``<TYPE>.<SUBTYPE>`` from the parent ``<event>`` element
  (e.g. ``Conflict.Attack``). Use ``--no-subtypes`` to keep only the top
  level (``Conflict``).
* ``trigger``: text of the ``<anchor><charseq>...`` element.
* ``arguments``: one entry per ``<event_mention_argument>``; ``role``
  comes from the ``ROLE`` attribute, ``entity`` from the argument's own
  ``<extent><charseq>...`` element.

Surface strings are pulled directly out of the SGM body so trigger and
argument text round-trip into the converted ``input`` (which is also
sourced from the SGM body, with SGML tags stripped).

Usage::

    uv run python tools/data/convert_ace2005.py \\
        --input /path/to/ace_2005_td_v7/data/English \\
        --out data/ace2005.jsonl

    # Top-level event types only (Conflict/Life/Movement/Justice/...)
    uv run python tools/data/convert_ace2005.py \\
        --input /path/to/ace_2005_td_v7/data/English \\
        --out data/ace2005_toplevel.jsonl --no-subtypes
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple


TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def _strip_sgml(sgm_text: str) -> str:
    """Return the plain text body of a .sgm file (tags removed, ws normalised)."""
    # ACE .sgm files wrap their body in <TEXT>...</TEXT>. Pull that out if
    # present; otherwise fall back to the full file with tags stripped.
    m = re.search(r"<TEXT>(.*?)</TEXT>", sgm_text, re.DOTALL | re.IGNORECASE)
    body = m.group(1) if m else sgm_text
    stripped = TAG_RE.sub(" ", body)
    return WS_RE.sub(" ", stripped).strip()


def _first_charseq_text(parent: ET.Element, sub_tag: str) -> str | None:
    """Find ``parent/<sub_tag>/<charseq>`` and return its text content."""
    sub = parent.find(sub_tag)
    if sub is None:
        return None
    cs = sub.find("charseq")
    if cs is None or cs.text is None:
        return None
    return WS_RE.sub(" ", cs.text).strip() or None


def _pair_sgm(apf_path: Path) -> Path | None:
    """Return the .sgm sibling of an .apf.xml path, or None if missing."""
    stem = apf_path.name
    if stem.endswith(".apf.xml"):
        sgm = apf_path.with_name(stem[:-len(".apf.xml")] + ".sgm")
    else:
        sgm = apf_path.with_suffix(".sgm")
    return sgm if sgm.is_file() else None


def parse_apf(
    apf_path: Path,
    keep_subtypes: bool,
) -> List[Dict[str, Any]] | None:
    """Parse one .apf.xml file and return a list of GLiNER2 event records.

    One record per source document — events from all ``<event_mention>``
    occurrences in the file are collected into a single ``events`` list.
    """
    sgm_path = _pair_sgm(apf_path)
    if sgm_path is None:
        return None
    try:
        tree = ET.parse(apf_path)
    except ET.ParseError:
        return None
    root = tree.getroot()

    try:
        sgm_text = sgm_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    text = _strip_sgml(sgm_text)
    if not text:
        return None

    events_out: List[Dict[str, Any]] = []
    for evt in root.iter("event"):
        etype = evt.get("TYPE", "").strip()
        esub = evt.get("SUBTYPE", "").strip()
        if not etype:
            continue
        event_type = f"{etype}.{esub}" if keep_subtypes and esub else etype

        for emention in evt.iter("event_mention"):
            anchor_text = _first_charseq_text(emention, "anchor")
            if not anchor_text or anchor_text not in text:
                continue
            arguments: List[Dict[str, Any]] = []
            seen: set = set()
            for arg in emention.iter("event_mention_argument"):
                role = (arg.get("ROLE") or "").strip()
                if not role:
                    continue
                arg_text = _first_charseq_text(arg, "extent")
                if not arg_text or arg_text not in text:
                    continue
                key = (role, arg_text)
                if key in seen:
                    continue
                seen.add(key)
                arguments.append({"role": role, "entity": arg_text})

            events_out.append({
                "event_type": event_type,
                "trigger": anchor_text,
                "arguments": arguments,
            })

    if not events_out:
        return None
    return [{"input": text, "output": {"events": events_out}}]


def iter_apf_files(root: Path):
    """Yield every .apf.xml under ``root`` (recursively)."""
    yield from root.rglob("*.apf.xml")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path,
                        help="Root directory of the ACE 2005 corpus (typically "
                             "ace_2005_td_v7/data/English).")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output GLiNER2 events JSONL path.")
    parser.add_argument("--max-records", type=int, default=-1,
                        help="Maximum documents to emit (-1 = all).")
    parser.add_argument("--no-subtypes", action="store_true",
                        help="Use only top-level event types (drop SUBTYPE).")
    args = parser.parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"input directory not found: {args.input}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    keep_subtypes = not args.no_subtypes

    emitted = 0
    skipped = 0
    total_events = 0
    total_args = 0
    all_types: set[str] = set()
    all_roles: set[str] = set()

    with args.out.open("w") as f:
        for apf_path in iter_apf_files(args.input):
            if 0 <= args.max_records <= emitted:
                break
            records = parse_apf(apf_path, keep_subtypes=keep_subtypes)
            if records is None:
                skipped += 1
                continue
            for rec in records:
                f.write(json.dumps(rec) + "\n")
                emitted += 1
                for ev in rec["output"]["events"]:
                    total_events += 1
                    total_args += len(ev["arguments"])
                    all_types.add(ev["event_type"])
                    for a in ev["arguments"]:
                        all_roles.add(a["role"])
            if emitted % 100 == 0:
                print(f"  emitted={emitted}  events={total_events}  "
                      f"types={len(all_types)}  roles={len(all_roles)}")

    print(f"Done. emitted={emitted} skipped_no_events={skipped} "
          f"total_events={total_events} total_arguments={total_args} "
          f"event_types={len(all_types)} roles={len(all_roles)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
