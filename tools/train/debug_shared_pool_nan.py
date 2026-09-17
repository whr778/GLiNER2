"""Localize the non-finite that kills `candidate_pool: shared` at micro-batch 1.

The pool A/B's treatment arms died with `196 non-finite micro-batch loss(es)` -- 49 optimizer
steps x 4 accumulation, i.e. EVERY micro-batch from the first. Every CPU-reachable variable was
then ruled out and all stayed finite: random and trained weights, fp32 and bf16, `.to()` and
autocast, real cmnee records through the real collator with `event_records`, gold injection at
its step-0 value of 1.0, and the full backward. So the fault is CUDA-side, and this script is
the instrument for it.

WHAT IT REPORTS, and why that shape: a hook on EVERY submodule records the first module whose
OUTPUT is non-finite while ALL ITS INPUTS WERE FINITE. That conjunction is the point -- once a
NaN exists it propagates, so "contains a NaN" names dozens of modules and the culprit is the
one that MANUFACTURED it. Run both arms: `per_query` is the control that proves the harness
itself is clean.

    python tools/train/debug_shared_pool_nan.py --pool shared --steps 2
"""

import argparse
import json

import torch

from gliner2 import AutoExtractor
from gliner2.configuration import BoundaryHeadSettings, validate_boundary_head

CKPT = "whr778/gliner2-eb16-eventrecords-tr"
DATA = "data/cmnee.train.jsonl"


def _stats(t):
    """Compact finite/range summary for a tensor, or a short tag for anything else."""
    if not isinstance(t, torch.Tensor) or not t.dtype.is_floating_point:
        return type(t).__name__
    f = torch.isfinite(t)
    finite = bool(f.all())
    absmax = float(t[f].abs().max()) if bool(f.any()) else float("nan")
    return (f"shape={tuple(t.shape)} dtype={t.dtype} finite={finite} "
            f"nonfinite={int((~f).sum())}/{t.numel()} absmax={absmax:.4g}")


def _tensors(obj, depth=0):
    """Yield tensors from arbitrarily nested outputs (tuples, dataclasses, dicts)."""
    if depth > 3:
        return
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            yield from _tensors(o, depth + 1)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from _tensors(o, depth + 1)
    elif hasattr(obj, "__dict__"):
        for o in vars(obj).values():
            yield from _tensors(o, depth + 1)


def _all_finite(obj):
    ts = [t for t in _tensors(obj) if t.dtype.is_floating_point]
    return all(bool(torch.isfinite(t).all()) for t in ts)


def install_hooks(model, found):
    """Flag every module that turns finite inputs into a non-finite output."""
    def make(name, mod):
        def hook(_m, inp, out):
            if _all_finite(inp) and not _all_finite(out):
                found.append((name, type(mod).__name__, inp, out))
        return hook

    handles = []
    for name, mod in model.named_modules():
        if name:
            handles.append(mod.register_forward_hook(make(name, mod)))
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="shared", choices=["per_query", "shared"])
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    # `action="store_true", default=True` is unturnoffable; bf16 autocast is the
    # variable under test here, so it needs a real off switch.
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dbg] device={dev} pool={args.pool} bf16={args.bf16}")

    model = AutoExtractor.from_pretrained(CKPT, architecture="boundary").to(dev)
    bh = dict(model.config.boundary_head)
    bh["candidate_pool"] = args.pool
    settings = BoundaryHeadSettings(**validate_boundary_head(bh))
    model.boundary_settings = settings
    model.boundary_head.settings = settings
    model.train()
    # Its value at step 0 under the trainer's schedule (gold_injection_start = 1.0).
    model.boundary_head._gold_injection_prob = 1.0

    rows = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= args.batch * args.steps:
                break

    ckw = dict(max_len=1024, architecture="boundary", max_gold_per_query=256,
               on_capacity_exceeded="skip_sample", event_records=True, error_policy="skip")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for step in range(args.steps):
        chunk = rows[step * args.batch:(step + 1) * args.batch]
        batch = [(r["input"], r["output"]) for r in chunk]
        collated = model.processor.collate_fn_train(batch, **ckw)

        found = []
        handles = install_hooks(model, found)
        torch.manual_seed(step)
        if args.bf16 and dev == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(collated)
                loss = out.total_loss
        else:
            out = model(collated)
            loss = out.total_loss
        for h in handles:
            h.remove()

        finite = bool(torch.isfinite(loss))
        print(f"\n[dbg] === step {step}: loss={float(loss.detach()):.5f} finite={finite} ===")
        if found:
            print(f"[dbg] {len(found)} module(s) MANUFACTURED a non-finite value. First:")
            for name, cls, inp, o in found[:5]:
                print(f"[dbg]   MODULE {name}  ({cls})")
                for i, t in enumerate(t for t in _tensors(inp) if isinstance(t, torch.Tensor)):
                    print(f"[dbg]      in[{i}]  {_stats(t)}")
                for i, t in enumerate(t for t in _tensors(o) if isinstance(t, torch.Tensor)):
                    print(f"[dbg]      out[{i}] {_stats(t)}")
        elif not finite:
            print("[dbg] loss is non-finite but NO module turned finite inputs into a "
                  "non-finite output -- it is arising in the LOSS arithmetic, not a module.")
        else:
            print("[dbg] everything finite at this step.")

        if finite:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            bad = [n for n, p in model.named_parameters()
                   if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
            sp = sum(float(p.grad.detach().float().norm() ** 2)
                     for n, p in model.named_parameters()
                     if "shared_pool" in n and p.grad is not None) ** 0.5
            print(f"[dbg] backward: shared-pool grad norm {sp:.4e}; "
                  f"{len(bad)} parameter(s) with non-finite grad")
            for n in bad[:10]:
                print(f"[dbg]    non-finite grad: {n}")
            opt.step()


if __name__ == "__main__":
    main()
