"""Measure the listwise logit scale per corpus, and calibrate the typed margin's k.

The typed margin adds ``k * sd`` to type-incompatible candidates, so `k` is only meaningful
against the scale of the logits it is added to -- and those logits are NOT normalised. This
reports, per corpus and per loss path:

  * the raw per-batch sd and its coefficient of variation;
  * the sd an EMA (beta, bias-corrected, ratio-clamped) would actually hand the margin;
  * the `k` that yields a target probability multiplier, since a margin of `d` multiplies a
    candidate's weight by `e^d`.

WHY AN INSTRUMENT AND NOT A ONE-OFF. The scales drift as the model trains, and they differ
more by DOMAIN than by language -- measured: Chinese/English 1.12x on the proposal path,
while scierc/casie, both English, differ 1.8x. Any k chosen once on one checkpoint is a
guess about every other one. Re-run this whenever the checkpoint or the mix changes.

MEASURED 2026-09-21 on eb16-eventrecords-tr, 72 mixed batches: raw per-batch CV 0.555
(proposal) and 0.338 (rerank). A plain EMA at beta=0.99 gives 0.511 and 0.548 -- the rerank
figure being WORSE than not smoothing at all, because the estimate starts at zero and the
climb dominates. Bias correction takes the same beta to 0.061 / 0.078.

    uv run python tools/train/measure_logit_scale.py \
        --checkpoint whr778/gliner2-eb16-eventrecords-tr \
        --corpora data/cmnee_typed data/casie_typed data/scierc --batches 16
"""

import argparse
import json
import math
import statistics as st
from pathlib import Path

import torch


def _cv(xs):
    med = st.median(xs)
    return st.pstdev(xs) / max(med, 1e-9)


def _ema(xs, beta, clamp):
    """Bias-corrected EMA with a ratio clamp -- the form the loss actually uses."""
    out, s, t = [], 0.0, 0
    for x in xs:
        if t and clamp > 1.0:
            ref = s / (1.0 - beta ** t)
            x = min(max(x, ref / clamp), ref * clamp)
        t += 1
        s = beta * s + (1.0 - beta) * x
        out.append(s / (1.0 - beta ** t))
    return out


def _interleave(report, beta, clamp):
    """Replay the corpora as ONE round-robin stream -- what a mixed sampler actually feeds.

    The margin keeps a SINGLE EMA for the whole model, so in a mixed stream every corpus is
    dosed with the MIX's scale, not its own. This reports the ratio between the two: 1.0 means
    a corpus gets the dose its own logits imply, 2.0 means twice that. It also counts how
    often the ratio clamp fires, because across corpora the clamp sees real scale changes
    rather than the outliers it was added to reject.
    """
    for pathname in ("proposal", "rerank"):
        streams = {c: list(d[pathname]["raw"]) for c, d in report.items()
                   if isinstance(d.get(pathname), dict)}
        if len(streams) < 2:
            continue
        order, cur = [], {c: 0 for c in streams}
        while any(cur[c] < len(streams[c]) for c in streams):
            for c in streams:
                if cur[c] < len(streams[c]):
                    order.append((c, streams[c][cur[c]]))
                    cur[c] += 1
        xs = [v for _, v in order]
        mixed = _ema(xs, beta, clamp)
        clipped = sum(1 for x, m in zip(xs, _ema(xs, beta, 0.0)) if abs(x - m) > 1e-12) \
            if clamp > 1.0 else 0
        print(f"\n=== mixed stream, {pathname} (round-robin, {len(order)} batches) ===")
        for c in streams:
            got = [m for (cc, _), m in zip(order, mixed) if cc == c]
            own = st.median(streams[c])
            ratio = st.median(got) / max(own, 1e-9)
            report[c][pathname]["mixed_dose_ratio"] = ratio
            print(f"   {c:16s} own sd={own:5.2f}  dosed with={st.median(got):5.2f}  "
                  f"ratio={ratio:.2f}x")
        n_clip = _count_clipped(xs, beta, clamp)
        print(f"   clamp={clamp}: {n_clip}/{len(xs)} batches clipped "
              f"({100.0 * n_clip / len(xs):.0f}%)")


def _count_clipped(xs, beta, clamp):
    """How many batches the ratio clamp actually modified."""
    if clamp <= 1.0:
        return 0
    n, s, t = 0, 0.0, 0
    for x in xs:
        if t:
            ref = s / (1.0 - beta ** t)
            if x > ref * clamp or x < ref / clamp:
                n += 1
                x = min(max(x, ref / clamp), ref * clamp)
        t += 1
        s = beta * s + (1.0 - beta) * x
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, help="HF id or local path")
    ap.add_argument("--corpora", nargs="+", required=True, help="corpus prefixes")
    ap.add_argument("--batches", type=int, default=16, help="records per corpus")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--beta", type=float, default=0.99)
    ap.add_argument("--clamp", type=float, default=1.5)
    ap.add_argument("--targets", nargs="+", type=float, default=[2.0, 3.0],
                    help="probability multipliers to solve k for")
    ap.add_argument("--device", default="cpu",
                    help="cpu by default: MPS int64 gather corrupts packed candidate keys "
                         "on long Chinese documents")
    ap.add_argument("--out", help="write the measurements as JSON")
    args = ap.parse_args()

    import gliner2.models.boundary.losses as L
    from gliner2.models.boundary.constants import MASK_LOGIT
    from gliner2 import AutoExtractor
    from gliner2.processor import SchemaTransformer

    seq = {"proposal": [], "rerank": []}

    def spy(name, orig):
        def f(logits, *a, **k):
            v = logits.detach().float()
            live = v[v > MASK_LOGIT + 1.0]
            if live.numel() > 32:
                seq[name].append(float(live.std()))
            return orig(logits, *a, **k)
        return f

    # PATCH ONLY THE MODEL'S NAMESPACE. `reranker_listwise_loss` DELEGATES to
    # `proposal_listwise_loss`, so patching the name in `losses` counts every rerank call a
    # second time as a proposal call -- measured as exactly 2x (n=16 proposal against n=8
    # rerank on 8 batches), which silently mixes rerank logits into the proposal statistics.
    # model.py imports both names into its own module, so binding them here intercepts the
    # model's call sites and leaves the internal delegation untouched.
    import gliner2.models.boundary.model as M
    M.proposal_listwise_loss = spy("proposal", L.proposal_listwise_loss)
    M.reranker_listwise_loss = spy("rerank", L.reranker_listwise_loss)

    model = AutoExtractor.from_pretrained(args.checkpoint).to(args.device)
    model.train()

    report = {}
    for prefix in args.corpora:
        path = Path(f"{prefix}.train.jsonl")
        if not path.is_file():
            print(f"[scale] {prefix}: no train split; skipping")
            continue
        # CJK needs the char splitter; whitespace elsewhere. Guessed from the data, so a
        # mis-guess shows up as a wildly different scale rather than silently skewing it.
        sample = json.loads(path.open(encoding="utf-8").readline())
        text = sample.get("input") or sample.get("text") or ""
        cjk = sum(1 for c in text[:400] if "一" <= c <= "鿿")
        splitter = "char" if cjk > 40 else "whitespace"

        recs = []
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= args.batches * args.batch_size:
                    break
                r = json.loads(line)
                recs.append(((r.get("input") or r.get("text")),
                             r.get("output") or r.get("schema")))
        proc = SchemaTransformer(model_name="jhu-clsp/mmBERT-base", word_splitter=splitter)
        proc.change_mode(is_training=True)
        for k in seq:
            seq[k].clear()
        ok = fail = 0
        for lo in range(0, len(recs), args.batch_size):
            try:
                b = proc.collate_fn_train(recs[lo:lo + args.batch_size],
                                          architecture="boundary", event_records=True)
                with torch.no_grad():
                    model(b)
                ok += 1
            except Exception:
                fail += 1
        name = Path(prefix).name
        report[name] = {"splitter": splitter, "batches_ok": ok, "batches_failed": fail}
        print(f"\n=== {name}  ({splitter} splitter, {ok} batches ok, {fail} failed) ===")
        for pathname in ("proposal", "rerank"):
            xs = list(seq[pathname])
            if len(xs) < 3:
                print(f"   {pathname:9s} too few samples ({len(xs)})")
                continue
            sm = _ema(xs, args.beta, args.clamp)
            med = st.median(sm)
            ks = {f"{m:g}x": round(math.log(m) / max(med, 1e-9), 4) for m in args.targets}
            report[name][pathname] = {
                "n": len(xs), "raw_median": st.median(xs), "raw_cv": _cv(xs),
                "ema_median": med, "ema_cv": _cv(sm), "k": ks, "raw": xs,
            }
            print(f"   {pathname:9s} n={len(xs):3d}  raw median={st.median(xs):6.2f} "
                  f"CV={_cv(xs):.3f}  ->  EMA median={med:6.2f} CV={_cv(sm):.3f}   "
                  f"k: " + "  ".join(f"{t}={v}" for t, v in ks.items()))
    _interleave(report, args.beta, args.clamp)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[scale] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
