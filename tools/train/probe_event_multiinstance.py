"""Does this checkpoint emit MORE THAN ONE event instance of the same type per document?

THE QUESTION THIS ANSWERS, and why it is not an F1 question. The mention path keys event
instances by TYPE, so N events of one type in a document pool into ONE instance -- measured
at 69.7% of gold instances on the blind test (`tools/data/event_multiplicity.py`). No amount
of training or threshold moves that: it is the decoder's addressing scheme. `event_records:
true` routes events through the RECORD head instead, which allocates a slot per instance and
is therefore the only thing that CAN emit two.

So for a mid-training checkpoint the useful question is not "is F1 higher" -- a model at
epoch 1 of 4 loses to a trained one regardless -- but "is the mechanism firing at all".
If it never emits two instances of a type, the line is dead and ten more GPU-hours will not
revive it. That verdict is available on twelve short documents and a CPU.

    uv run python tools/train/probe_event_multiinstance.py \
        --checkpoint /path/to/best --docs multi_docs.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def cmnee_schema(train: Path) -> dict:
    """{event_type: [role, ...]} from the corpus's own train split -- not from gold, so the
    probe does not hand the model the answer sheet."""
    roles = defaultdict(set)
    for line in train.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for ev in ((json.loads(line).get("output") or {}).get("events") or []):
            if isinstance(ev, dict) and ev.get("event_type"):
                for a in ev.get("arguments") or []:
                    if isinstance(a, dict) and a.get("role"):
                        roles[ev["event_type"]].add(a["role"])
    return {t: sorted(r) for t, r in sorted(roles.items())}


def emitted_counts(pred: dict) -> tuple:
    """(instances per type, triggers per type) from an extract_events return.

    THE KEY IS `event_extraction`, not `events`. Reading the wrong key made both a trained
    incumbent and a mid-training checkpoint look like they emitted nothing at all -- the
    control is what caught it, which is the only reason this probe has one.

    Triggers are counted separately because POOLING SHOWS UP THERE FIRST: the mention path
    returns ONE instance carrying every trigger it found, e.g.
    `Experiment: [{triggers: ["下水", "测试"], arguments: [...unioned...]}]` where the gold
    is two separate Experiment events. Counting instances alone would score that as 1 and
    miss that the model DID find both triggers and merged them.
    """
    inst, trig = Counter(), Counter()
    evs = pred.get("event_extraction") if isinstance(pred, dict) else None
    if isinstance(evs, dict):
        for t, v in evs.items():
            items = v if isinstance(v, list) else [v]
            inst[t] = len(items)
            trig[t] = sum(len(i.get("triggers") or []) for i in items if isinstance(i, dict))
    elif isinstance(evs, list):
        for e in evs:
            if isinstance(e, dict):
                t = e.get("event_type") or e.get("type") or "?"
                inst[t] += 1
                trig[t] += len(e.get("triggers") or [])
    return inst, trig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--docs", required=True, help="JSON from the doc-picker")
    ap.add_argument("--train", default="data/cmnee.train.jsonl")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    import torch
    from gliner2 import AutoExtractor

    docs = json.loads(Path(args.docs).read_text(encoding="utf-8"))
    schema = cmnee_schema(Path(args.train))

    model = AutoExtractor.from_pretrained(args.checkpoint, map_location=args.device)
    # The extractor IS the nn.Module here (BoundaryExtractor); older shapes wrapped one.
    module = getattr(model, "model", model)
    module.to(torch.device(args.device)).eval()
    print(f"[probe] {args.label or args.checkpoint}  device={args.device}  "
          f"threshold={args.threshold}  schema={len(schema)} event types")

    gold_multi = pred_multi = pooled_n = 0
    rows = []
    for d in docs:
        rec = d["record"]
        text = rec["input"] if isinstance(rec["input"], str) else str(rec["input"])
        out = model.extract_events(text, schema, threshold=args.threshold)
        got, trig = emitted_counts(out)
        want = Counter(e.get("event_type") for e in (rec["output"].get("events") or []))
        dup_t = {t for t, n in want.items() if n > 1}
        gold_multi += 1
        # ANY type emitted twice counts as the mechanism firing, even a wrong one: the
        # question is whether the decoder CAN address two slots at all.
        got = {t: n for t, n in got.items() if n}      # drop types with zero instances
        trig = {t: n for t, n in trig.items() if n}
        # Score the type the GOLD actually duplicates, not just any type emitted twice --
        # two instances of a WRONG type would otherwise read as the mechanism working.
        on_gold = max((got.get(t, 0) for t in dup_t), default=0)
        emitted_two_any = on_gold > 1
        # POOLED = found several triggers for a type but returned them in ONE instance.
        pooled = any(trig.get(t, 0) > 1 and got.get(t, 0) == 1 for t in got)
        pred_multi += 1 if emitted_two_any else 0
        pooled_n += 1 if pooled else 0
        rows.append((d["corpus"], d["idx"], dict(want), dict(got), dict(trig),
                     emitted_two_any, pooled, sorted(dup_t)[0] if dup_t else "", on_gold))

    print(f"\n{'idx':>6}  {'gold type':12s}{'gold n':>7}{'pred n':>8}{'pred trig':>11}   "
          f"{'all predicted (type:inst)':38s}{'2 inst?':9s}{'pooled?'}")
    for corpus, idx, want, got, trig, any_two, pooled, gt, on_gold in rows:
        gn = want.get(gt, 0)
        tn = trig.get(gt, 0)
        g = ", ".join(f"{k}:{v}" for k, v in sorted(got.items()))[:36] or "(none)"
        print(f"{idx:>6}  {gt[:12]:12s}{gn:>7}{on_gold:>8}{tn:>11}   {g:38s}"
              f"{'YES' if any_two else 'no':9s}{'YES' if pooled else 'no'}")

    print(f"\ndocuments where it emitted >1 instance OF THE GOLD-DUPLICATED TYPE: "
          f"{pred_multi}/{gold_multi}")
    print(f"documents where it found >1 trigger but POOLED them into one instance: "
          f"{pooled_n}/{gold_multi}")
    print("every one of these documents contains >=2 gold events of one type, so a decoder")
    print("that never emits two is structurally incapable of scoring them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
