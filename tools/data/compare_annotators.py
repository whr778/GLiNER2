"""Agreement between two annotators on the SAME documents, and the real price of one batch.

Prices an annotation purchase from a labelled sample (tools/data/notes/ANNOTATION_ECONOMICS.md):
the reference annotator (e.g. Sonnet, candidate EVAL gold) against the other (e.g. Haiku, whose
labels the training data carries). Reports, per unit, how many each finds and the F1 of the other
against the reference -- an upper bound on how well a model scored against the weaker labels can
be judged. Units: events (type, trigger), arguments keyed (type, trigger, role, entity) and
unkeyed (type, role, entity), entities (type, surface); surfaces compared case-folded.

    uv run python tools/data/compare_annotators.py --reference sonnet.train.jsonl \\
        --other haiku_gold.jsonl --batch-id msgbatch_... --price-in 1.5 --price-out 7.5
"""
from __future__ import annotations

import argparse
import json
import os


def units(rec: dict) -> dict:
    """{unit: set of keys} for one record's output."""
    out = rec.get("output") or {}
    f = lambda s: str(s).strip().casefold()
    ev, key, unkey = set(), set(), set()
    for e in out.get("events") or []:
        t = e.get("event_type")
        for trig in e.get("triggers") or [None]:
            ev.add((t, f(trig)))
            for a in e.get("arguments") or []:
                key.add((t, f(trig), a.get("role"), f(a.get("entity"))))
        for a in e.get("arguments") or []:
            unkey.add((t, a.get("role"), f(a.get("entity"))))
    ents = out.get("entities") or {}
    if isinstance(ents, list):
        ents = {k: v for d in ents for k, v in d.items()}
    en = {(t, f(s)) for t, ss in ents.items() for s in (ss or []) if isinstance(s, str)}
    return {"events": ev, "arguments (keyed)": key, "arguments (unkeyed)": unkey, "entities": en}


def compare(ref: dict, other: dict) -> None:
    shared = sorted(set(ref) & set(other))
    print(f"[compare] {len(shared)} shared documents (reference {len(ref)}, other {len(other)})")
    for unit in ("events", "arguments (keyed)", "arguments (unkeyed)", "entities"):
        tp = n_ref = n_oth = 0
        for doc in shared:
            r, o = units(ref[doc])[unit], units(other[doc])[unit]
            tp += len(r & o); n_ref += len(r); n_oth += len(o)
        p = tp / n_oth if n_oth else float("nan")
        rc = tp / n_ref if n_ref else float("nan")
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        print(f"[compare] {unit:20s} reference {n_ref:5d} | other {n_oth:5d} | agree {tp:5d} | "
              f"other vs reference: P {p:.3f} R {rc:.3f} F1 {f1:.3f}")


def batch_cost(batch_id: str, price_in: float, price_out: float, docs: int) -> None:
    """Real token usage from the batch results; prices are per MTok at BATCH rates."""
    from anthropic import Anthropic
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    client = Anthropic(default_headers={"anthropic-workspace-id": ws} if ws else None)
    n = tin = tout = 0
    for r in client.messages.batches.results(batch_id):
        if r.result.type == "succeeded":
            u = r.result.message.usage
            n += 1; tin += u.input_tokens; tout += u.output_tokens
    cost = tin / 1e6 * price_in + tout / 1e6 * price_out
    print(f"[cost] {n} succeeded | {tin / max(n, 1):.0f} in / {tout / max(n, 1):.0f} out tokens per doc | "
          f"${cost:.4f} total, ${cost / max(n, 1):.5f}/doc | {docs} docs -> ${cost / max(n, 1) * docs:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--batch-id", help="price the reference's batch from its real usage")
    ap.add_argument("--price-in", type=float, help="USD per MTok input, batch rate")
    ap.add_argument("--price-out", type=float, help="USD per MTok output, batch rate")
    ap.add_argument("--project-docs", type=int, default=1698, help="docs to price the full purchase at")
    args = ap.parse_args()
    load = lambda p: {r["input"]: r for r in map(json.loads, open(p, encoding="utf-8")) if r}
    compare(load(args.reference), load(args.other))
    if args.batch_id:
        batch_cost(args.batch_id, args.price_in, args.price_out, args.project_docs)


if __name__ == "__main__":
    main()
