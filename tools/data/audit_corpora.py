"""Structural validity audit over every corpus in data/.

    uv run python tools/data/audit_corpora.py

Run it after any corpus build or repair. Every check here corresponds to a defect that
actually shipped in this project and was invisible until someone looked.

Cheap checks this project has actually been bitten by:
  EMPTY      zero-byte or zero-row split files (mix_natural.test was 0 bytes)
  JSON       unparseable lines
  NOINPUT    records with no `input` text
  SPANS      entity spans NOT verbatim in the text -- trains nothing, reports nothing
  STRUCT     structures with no record_metadata (silently undecodable on the boundary path)
  MENU       a classification true_label absent from that record's OWN label menu --
             the model cannot answer a label it is never shown
  SEPS       stray line separators (NEL U+0085, U+2028, U+2029) that fragment records
"""
import glob
import json
import os

SEPS = ("", " ", " ")


def audit(path):
    r = dict(rows=0, bad_json=0, no_input=0, nonverbatim=0, spans=0,
             struct=0, struct_nometa=0, menu_miss=0, cls=0, seps=0)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r["rows"] += 1
            try:
                rec = json.loads(line)
            except Exception:
                r["bad_json"] += 1
                continue
            text = rec.get("input") or rec.get("text") or ""
            if not text:
                r["no_input"] += 1
            if any(c in text for c in SEPS):
                r["seps"] += 1
            out = rec.get("output") or {}
            for _role, spans in (out.get("entities") or {}).items():
                for s in (spans if isinstance(spans, list) else [spans]):
                    if isinstance(s, str):
                        r["spans"] += 1
                        if s and s not in text:
                            r["nonverbatim"] += 1
            for c in (out.get("classifications") or []):
                menu = set(map(str, c.get("labels") or []))
                r["cls"] += 1
                # true_label is EITHER a scalar string or a list. Iterating a string
                # yields characters, which reported every letter as an off-menu label --
                # 38,358 phantom findings on zh_multitask. Scalar is a legitimate shape;
                # gliner2 7e67232 fixed the processor for it.
                tl = c.get("true_label")
                tl = [] if tl is None else ([tl] if isinstance(tl, str) else tl)
                for t in tl:
                    if menu and str(t) not in menu:
                        r["menu_miss"] += 1
            st = out.get("json_structures") or out.get("structures")
            if st:
                r["struct"] += 1
                if not (out.get("record_metadata") or rec.get("record_metadata")):
                    r["struct_nometa"] += 1
    return r


def main():
    files = [f for f in sorted(glob.glob("data/*.jsonl")) + sorted(glob.glob("data/*/*.jsonl"))
             if "guide_scores" not in os.path.basename(f)]   # score caches, not corpora
    problems = []
    for path in files:
        if os.path.getsize(path) == 0:
            problems.append((path, "EMPTY (0 bytes)"))
            continue
        try:
            a = audit(path)
        except Exception as e:
            problems.append((path, f"UNREADABLE {type(e).__name__}"))
            continue
        if a["rows"] == 0:
            problems.append((path, "ZERO ROWS"))
            continue
        flags = []
        if a["bad_json"]:
            flags.append(f"bad_json={a['bad_json']}")
        if a["no_input"]:
            flags.append(f"no_input={a['no_input']}")
        if a["seps"]:
            flags.append(f"line_separators={a['seps']}")
        if a["nonverbatim"]:
            flags.append(f"nonverbatim_spans={a['nonverbatim']}/{a['spans']}")
        if a["struct_nometa"]:
            flags.append(f"struct_no_metadata={a['struct_nometa']}/{a['struct']}")
        if a["menu_miss"]:
            flags.append(f"label_off_menu={a['menu_miss']}/{a['cls']}")
        if flags:
            problems.append((path, "; ".join(flags)))

    print(f"files scanned: {len(files):,}")
    print(f"files with findings: {len(problems)}\n")
    for p, f in problems:
        print(f"  {os.path.relpath(p):58s} {f}")


if __name__ == "__main__":
    main()
