"""How much gold never reaches the model, and WHY, for any corpus.

A mention is supervision only if its surface aligns to the tokenized text. The training
path is `word_splitter(text, lower=True)` -> `_tokenize_text(surface)` -> `_find_sublist`,
and a miss decodes to (-1, -1) and is silently skipped -- no error, no counter in the run
log. `surface in text` is NOT the same test and will report a corpus as clean when the
pipeline is losing mentions: measured on docee, substring says 100.00% and the real path
says 99.19%.

The loss is split by WHAT WOULD FIX IT, because the three cases need different answers:

  EXTENDABLE   expanding the surface to its enclosing word boundaries aligns, and the
               surface is multi-word or long enough that the expansion is completing a
               truncated span rather than renaming it. A gold defect worth repairing.
               docee: 'he child’s skeleton' -> 'the child’s skeleton',
               'ood services' -> 'food services', 'New Delh' -> 'New Delhi'.

  SUBTOKEN     the surface sits inside a longer word, and extending it CHANGES THE
               REFERENT. Report, never repair -- this is the biored hyphenation case,
               where 'mannose' inside 'mannose-binding' is a chemical inside a binding
               property. docee: 'artist' -> 'artists', '66' -> '66th'.

  RUNON        expanding to a word boundary swallows a large blob, because the text has
               no boundary there -- CJK, or concatenated table text. Measured on
               bio_ner_relations: 'TGF \u03b2' sits inside 'hESCsNameProteinmRNATGF \u03b2Stat1'.
               Neither repair applies.

  TOKENIZATION boundaries are clean and the splitter still disagrees, usually adjacent
               punctuation. Fixable in the splitter, not the data.

  ABSTRACTIVE  the surface is nowhere in the text. Inferred annotation; no repair exists.

Numbers depend on the checkpoint's word splitter, which is language-specific -- the CJK
splitter differs from the Latin one -- so quote the checkpoint beside any figure.

    uv run python tools/data/measure_surface_alignment.py --checkpoint <ckpt> \
        --config tools/train/config/base/eb17-best.yaml
    uv run python tools/data/measure_surface_alignment.py --checkpoint <ckpt> \
        --files data/docee.test.jsonl --fail-over 0.5
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

CATEGORIES = (
    "OFFSET_MISMATCH",
    "EXTENDABLE",
    "SUBTOKEN",
    "RUNON",
    "TOKENIZATION",
    "ABSTRACTIVE",
)

# A real truncation adds a character or two -- 'ood services' -> 'food services',
# 'New Delh' -> 'New Delhi'. When expanding to the enclosing word boundary adds far
# more than that, the text simply HAS no boundary there: CJK, or a concatenated
# table run-on. Neither repair applies, so it gets its own bucket instead of being
# reported as a fixable gold defect.
MAX_EXPANSION_CHARS = 4


def iter_gold_items(output):
    """Yield ``(dimension, label, surface, span_or_none)`` for every gold span.

    When a trigger or argument carries ``{"text", "start", "end"}``, ``span`` is
    ``(start, end)`` so alignment checks ``SchemaTransformer._resolve_span``'s
    exact word-edge rule rather than an unanchored substring search.
    """
    for label, vals in (output.get("entities") or {}).items():
        for v in vals or []:
            if isinstance(v, dict) and isinstance(v.get("text"), str) and v["text"]:
                span = (v["start"], v["end"]) if {"start", "end"} <= v.keys() else None
                yield "entities", label, v["text"], span
            elif isinstance(v, str) and v:
                yield "entities", label, v, None
    for ev in output.get("events") or []:
        if not isinstance(ev, dict):
            continue
        if "event_type" in ev:
            etype = ev.get("event_type")
            if not isinstance(etype, str) or not etype:
                continue
            for trig in ev.get("triggers") or []:
                if isinstance(trig, dict):
                    s = trig.get("text")
                    span = (trig["start"], trig["end"]) if {"start", "end"} <= trig.keys() else None
                else:
                    s, span = trig, None
                if isinstance(s, str) and s:
                    yield "events", f"{etype}.trigger", s, span
            for arg in ev.get("arguments") or []:
                if not isinstance(arg, dict):
                    continue
                role = arg.get("role")
                ent = arg.get("entity")
                if isinstance(ent, dict):
                    s = ent.get("text")
                    span = (ent["start"], ent["end"]) if {"start", "end"} <= ent.keys() else None
                else:
                    s, span = ent, None
                if isinstance(role, str) and role and isinstance(s, str) and s:
                    yield "events", f"{etype}.{role}", s, span
            continue
        for etype, body in ev.items():
            if not isinstance(body, dict):
                continue
            for role, vals in body.items():
                vals = vals if isinstance(vals, list) else [vals]
                for v in vals:
                    if isinstance(v, str) and v:
                        yield "events", f"{etype}.{role}", v, None
    for item in output.get("json_structures") or []:
        if not isinstance(item, dict):
            continue
        for parent, fields in item.items():
            if not isinstance(fields, dict):
                continue
            for field, v in fields.items():
                vals = v if isinstance(v, list) else [v]
                for x in vals:
                    if isinstance(x, str) and x:
                        yield "json_structures", f"{parent}.{field}", x, None


def iter_surfaces(output):
    """Yield ``(dimension, label, surface)`` for every gold span a corpus declares."""
    for dim, label, surface, _span in iter_gold_items(output):
        yield dim, label, surface


class Aligner:
    """The real training alignment path, and the reason a surface missed it."""

    def __init__(self, processor):
        self.p = processor

    def word_spans(self, text):
        return list(self.p.word_splitter(text, lower=True))

    def tokens(self, text):
        return [t for t, _, _ in self.p.word_splitter(text, lower=True)]

    def aligns(self, surface, text_tokens):
        pos = self.p._find_sublist(self.p._tokenize_text(surface), text_tokens)
        return bool(pos) and tuple(pos[0]) != (-1, -1)

    def span_aligns(self, surface, span, text, starts, ends):
        s, e = span
        return 0 <= s < e <= len(text) and text[s:e] == surface and s in starts and e in ends

    def _classify_range(self, i, j, surface, text, w_spans, text_tokens):
        overlapping = [(ws, we) for _, ws, we in w_spans if ws < j and we > i]
        if not overlapping:
            return "TOKENIZATION", ""
        a, b = overlapping[0][0], overlapping[-1][1]
        if a == i and b == j:
            return "TOKENIZATION", ""
        expansion = text[a:b]
        if len(expansion) - len(surface) > MAX_EXPANSION_CHARS:
            return "RUNON", expansion
        if self.aligns(expansion, text_tokens):
            if " " in surface or len(surface) > 12:
                return "EXTENDABLE", expansion
            return "SUBTOKEN", expansion
        return "SUBTOKEN", expansion

    def classify(self, surface, text, text_tokens, span=None, w_spans=None):
        """Return ``(category, expansion)``; expansion is '' when there is none."""
        if w_spans is None:
            w_spans = self.word_spans(text)
        if span is not None:
            s, e = span
            actual = text[max(0, s):max(0, e)] if 0 <= s < len(text) else ""
            if s < 0 or e > len(text) or e <= s or text[s:e] != surface:
                return "OFFSET_MISMATCH", actual
            return self._classify_range(s, e, surface, text, w_spans, text_tokens)

        tl, vl = text.lower(), surface.lower()
        fallback = None
        for m in re.finditer(re.escape(vl), tl):
            cat, exp = self._classify_range(m.start(), m.end(), surface, text, w_spans, text_tokens)
            if cat in ("TOKENIZATION", "EXTENDABLE"):
                return cat, exp
            fallback = fallback or (cat, exp)
        return fallback or ("ABSTRACTIVE", "")


def scan(path, aligner, max_records=0, examples_per=4):
    counts = collections.Counter()
    per_label = collections.defaultdict(collections.Counter)
    per_dim = collections.Counter()
    per_dim_bad = collections.Counter()
    examples = collections.defaultdict(list)
    total = 0
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_records and i >= max_records:
                break
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("input") or rec.get("text") or ""
            if not text:
                continue
            output = rec.get("output") or rec.get("schema") or {}
            w_spans = aligner.word_spans(text)
            text_tokens = [t for t, _, _ in w_spans]
            starts = {s for _, s, _ in w_spans}
            ends = {e for _, _, e in w_spans}
            for dim, label, surface, span in iter_gold_items(output):
                total += 1
                per_dim[dim] += 1
                ok = (
                    aligner.span_aligns(surface, span, text, starts, ends)
                    if span is not None
                    else aligner.aligns(surface, text_tokens)
                )
                if ok:
                    continue
                per_dim_bad[dim] += 1
                cat, expansion = aligner.classify(
                    surface, text, text_tokens, span=span, w_spans=w_spans
                )
                counts[cat] += 1
                per_label[cat][f"{dim}:{label}"] += 1
                if len(examples[cat]) < examples_per:
                    examples[cat].append((label, surface[:70], expansion[:70]))
    return total, counts, per_label, per_dim, per_dim_bad, examples


def resolve_files(args):
    if args.files:
        return list(args.files)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
    import yaml
    import train as trainer

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data = cfg.get("data") or {}
    return trainer._dedupe_paths(
        trainer._split_files(data.get("corpora") or [], args.split,
                             set(data.get("train_only") or ()))
        + trainer._event_split(data.get("event_files") or {}, args.split),
        args.split,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="any checkpoint; only its word splitter is used")
    ap.add_argument("--config", help="training config -- scans the files it would load")
    ap.add_argument("--files", nargs="+", help="explicit jsonl files instead of a config")
    ap.add_argument("--split", default="train", choices=("train", "val", "test"))
    ap.add_argument("--max-records", type=int, default=0, help="0 = every record")
    ap.add_argument("--fail-over", type=float, default=None,
                    help="exit 1 when unaligned%% exceeds this, so it can gate a launch")
    args = ap.parse_args()
    if not args.config and not args.files:
        ap.error("pass --config or --files")

    from gliner2 import AutoExtractor

    aligner = Aligner(AutoExtractor.from_pretrained(args.checkpoint,
                                                    map_location="cpu").processor)
    files = [f for f in resolve_files(args) if os.path.exists(f)]
    print(f"[align] checkpoint {args.checkpoint}  files {len(files)}\n")

    grand_total = 0
    grand = collections.Counter()
    grand_labels = collections.defaultdict(collections.Counter)
    worst = []
    for path in files:
        total, counts, per_label, per_dim, per_dim_bad, examples = scan(
            path, aligner, args.max_records
        )
        if not total:
            continue
        bad = sum(counts.values())
        pct = 100.0 * bad / total
        grand_total += total
        grand.update(counts)
        for cat, c in per_label.items():
            grand_labels[cat].update(c)
        worst.append((pct, Path(path).name, total, bad))
        dims = " ".join(
            f"{d}={per_dim_bad[d]:,}/{n:,} ({100.0 * per_dim_bad[d] / n:.1f}%)"
            for d, n in per_dim.most_common()
        )
        print(f"{Path(path).name:34} surfaces={total:>8,} unaligned={bad:>6,} "
              f"({pct:5.2f}%)  [{dims}]")
        if bad:
            print("        " + "  ".join(f"{c}={counts[c]:,}" for c in CATEGORIES
                                         if counts[c]))
            for cat in CATEGORIES:
                for label, surface, expansion in examples.get(cat, [])[:2]:
                    arrow = f" -> {expansion!r}" if expansion else ""
                    print(f"        {cat:12} {label}: {surface!r}{arrow}")

    bad = sum(grand.values())
    pct = 100.0 * bad / grand_total if grand_total else 0.0
    print(f"\n[align] TOTAL {bad:,} unaligned of {grand_total:,} surfaces ({pct:.2f}%)")
    for cat in CATEGORIES:
        if not grand[cat]:
            continue
        share = 100.0 * grand[cat] / bad
        top = ", ".join(f"{l}={n}" for l, n in grand_labels[cat].most_common(4))
        print(f"        {cat:12} {grand[cat]:>6,} ({share:4.1f}%)  top: {top}")
    print("\n[align] EXTENDABLE is a gold defect worth repairing. SUBTOKEN is NOT -- "
          "extending it changes the referent. RUNON means the text has no word boundary "
          "there (CJK, concatenated tables), TOKENIZATION is a splitter question, and "
          "ABSTRACTIVE has no repair.")

    if worst:
        worst.sort(reverse=True)
        print("[align] worst files: " + ", ".join(f"{n} {p:.2f}%" for p, n, _, _ in worst[:3]))
    if args.fail_over is not None and pct > args.fail_over:
        print(f"\n[align] *** REFUSING: {pct:.2f}% of gold surfaces never align, over the "
              f"{args.fail_over}% bar. That supervision is dropped silently. ***")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
