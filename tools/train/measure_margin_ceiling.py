"""How much can the typed margin actually move? Measure the ceiling before spending on it.

Adding margin ``d`` to a disallowed set S changes the listwise loss's view of gold by

    p_gold' / p_gold  =  1 / (1 + (e^d - 1) * w_S)        w_S = probability mass on S

Only ``d`` and ``w_S``. The logit sd does not appear, so no amount of k-calibration matters if
w_S is small. This measures w_S directly, per corpus and per loss path.

THREE BRACKETS, BECAUSE THE REAL DISALLOWED SET IS NOT RANDOM. Until ``typed_margin_mask`` is
built, the type-disallowed candidates cannot be identified, so their position in the score
order is bracketed instead:

    random : a random ``--frac`` of non-gold candidates -- the NEUTRAL assumption
    bottom : the lowest-scoring ``--frac``              -- the model already down-ranks them
    top    : the highest-scoring ``--frac``             -- THE CASE THE MARGIN EXISTS FOR

Measured on held-out val at d=ln2: random moves gold 0.0-0.7%, bottom 0.000%, top 20.0%
(cmnee) to 0.0% (casie). The verdict is therefore CONDITIONAL on where the disallowed fillers
sit, and the brackets cannot tell. When the real mask exists, replace the brackets with it and
this script returns one number instead of a range.

Run it on the SPLIT YOU MEAN. A trained checkpoint is far more confident on its own training
data; train and val happen to differ little here, but that is a finding, not an assumption.

    uv run python tools/train/measure_margin_ceiling.py \
        --checkpoint /path/to/ckpt --corpora data/cmnee_typed data/scierc --split val
"""

import argparse
import json
import math
import statistics as st
from pathlib import Path

import torch


def _seed_everything(seed: int) -> None:
    """Seed every RNG this path touches. torch alone is not enough."""
    import os
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    if os.environ.get("PYTHONHASHSEED") is None:
        print("[ceiling] WARNING: PYTHONHASHSEED unset; query counts may vary between runs")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpora", nargs="+", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--records", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--frac", type=float, default=0.034,
                    help="share of non-gold candidates treated as disallowed "
                         "(0.034 = the measured containment dose, 0.012 = exact matching)")
    ap.add_argument("--delta", type=float, default=math.log(2.0), help="margin, in logits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu",
                    help="cpu: MPS int64 gather corrupts packed candidate keys on long "
                         "Chinese documents")
    ap.add_argument("--out")
    args = ap.parse_args()

    _seed_everything(args.seed)

    import gliner2.models.boundary.losses as L
    import gliner2.models.boundary.model as M
    from gliner2.models.boundary.constants import MASK_LOGIT
    from gliner2 import AutoExtractor
    from gliner2.processor import SchemaTransformer

    rows = {"proposal": [], "rerank": []}

    def spy(name, orig, labels_are_float):
        """Read the candidate distribution on the way through.

        `reranker_listwise_loss` passes FLOAT `labels` where `proposal_listwise_loss` passes a
        BOOL `gold_mask`, so the conversion is not optional -- `&` on a float tensor raises.
        """
        def f(logits, gold, valid, query_mask, *a, **k):
            with torch.no_grad():
                qa, ca = k.get("query_axis", 1), k.get("candidate_axis", 2)
                lg = L._to_query_candidate(logits, qa, ca).float()
                gm = L._to_query_candidate(gold, qa, ca)
                vm = L._to_query_candidate(valid, qa, ca)
                gm = (gm > 0.5) if labels_are_float else gm.bool()
                live = vm.bool() & (lg > MASK_LOGIT + 1.0)
                gen = torch.Generator().manual_seed(args.seed)
                for q in range(lg.shape[0]):
                    m = live[q]
                    gold_here = gm[q] & m
                    if int(m.sum()) < 8 or not bool(gold_here.any()):
                        continue
                    s = lg[q][m]
                    p = torch.softmax(s, -1)
                    entropy = float(-(p * p.clamp_min(1e-12).log()).sum())
                    nongold = (~gold_here[m]).nonzero().flatten()
                    if len(nongold) < 2:
                        continue
                    n = max(1, int(round(args.frac * len(nongold))))
                    order = torch.argsort(s[nongold])
                    rnd = nongold[torch.randperm(len(nongold), generator=gen)[:n]]
                    rows[name].append((
                        entropy,
                        float(p[rnd].sum()),
                        float(p[nongold[order[:n]]].sum()),
                        float(p[nongold[order[-n:]]].sum()),
                    ))
            return orig(logits, gold, valid, query_mask, *a, **k)
        return f

    # Patch the MODEL's namespace only: reranker_listwise_loss delegates to
    # proposal_listwise_loss, so patching the loss module counts every rerank call twice.
    M.proposal_listwise_loss = spy("proposal", L.proposal_listwise_loss, False)
    M.reranker_listwise_loss = spy("rerank", L.reranker_listwise_loss, True)

    model = AutoExtractor.from_pretrained(args.checkpoint).to(args.device)
    # TRAIN MODE IS REQUIRED, not incidental: in eval mode the proposal loss is never
    # called at all (0 queries captured), and the rerank distribution shifts hard -- scierc
    # w_top read 0.375 against 0.084. So dropout is live, and REPRODUCIBILITY MUST BE BOUGHT
    # EXPLICITLY. Seed torch, Python's `random` AND numpy: the processor's label handling
    # uses Python's random, which torch.manual_seed does not touch. Run with PYTHONHASHSEED
    # set as well -- label order derives from sets, and without it the query COUNT moves
    # between processes (measured 15 against 14 on the same data).
    model.train()

    def keeps(w):
        """p_gold' / p_gold. 1.0 means the margin did nothing."""
        return 1.0 / (1.0 + (math.exp(args.delta) - 1.0) * w)

    report = {}
    for prefix in args.corpora:
        path = Path(f"{prefix}.{args.split}.jsonl")
        if not path.is_file():
            print(f"[ceiling] {prefix}: no {args.split} split; skipping")
            continue
        first = json.loads(path.open(encoding="utf-8").readline())
        cjk = sum(1 for c in (first.get("input") or "")[:400] if "一" <= c <= "鿿")
        proc = SchemaTransformer(model_name="jhu-clsp/mmBERT-base",
                                 word_splitter="char" if cjk > 40 else "whitespace")
        proc.change_mode(is_training=True)
        recs = []
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= args.records:
                    break
                r = json.loads(line)
                recs.append((r.get("input"), r.get("output")))
        for key in rows:
            rows[key].clear()
        ok = bad = 0
        for lo in range(0, len(recs), args.batch_size):
            try:
                b = proc.collate_fn_train(recs[lo:lo + args.batch_size],
                                          architecture="boundary", event_records=True)
                with torch.no_grad():
                    model(b)
                ok += 1
            except Exception as exc:
                bad += 1
                if bad == 1:
                    print(f"   [{Path(prefix).name}] first batch failure: "
                          f"{type(exc).__name__}: {exc}")
        report[Path(prefix).name] = {k: list(v) for k, v in rows.items()}
        report[Path(prefix).name]["_batches"] = {"ok": ok, "failed": bad}

    out = {}
    for pathname in ("proposal", "rerank"):
        print(f"\n===== {pathname}  (split={args.split}, delta={args.delta:.3f}, "
              f"frac={args.frac}, seed={args.seed}) =====")
        print(f"{'corpus':16s} {'n':>4} {'H nats':>7} {'w rand':>8} {'w bot':>9} {'w TOP':>8}"
              f" | {'keeps rand':>11} {'keeps TOP':>10}")
        for corpus, d in report.items():
            r = d[pathname]
            if len(r) < 10:
                print(f"{corpus:16s} too few queries ({len(r)}); "
                      f"batches {d['_batches']}")
                continue
            entropy = st.median([x[0] for x in r])
            w_rand = st.median([x[1] for x in r])
            w_bot = st.median([x[2] for x in r])
            w_top = st.median([x[3] for x in r])
            out.setdefault(corpus, {})[pathname] = {
                "n": len(r), "entropy_nats": entropy, "effective_candidates": math.exp(entropy),
                "w_random": w_rand, "w_bottom": w_bot, "w_top": w_top,
                "keeps_random": keeps(w_rand), "keeps_top": keeps(w_top),
            }
            print(f"{corpus:16s} {len(r):>4} {entropy:>7.2f} {w_rand:>8.4f} {w_bot:>9.5f} "
                  f"{w_top:>8.4f} | {keeps(w_rand):>11.4f} {keeps(w_top):>10.4f}")
        print("   keeps = p_gold' / p_gold; 1.0000 means the margin did nothing.")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n[ceiling] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
