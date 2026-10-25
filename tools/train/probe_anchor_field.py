"""Code review Issue 2: what does the ANCHOR field's own assignment row cost the record head?

In natural mode (events) every trigger candidate seeds one instance, and `_instance_field_loss` supervises EVERY field
of the record, the anchor (trigger) field included: instance i must point at its own trigger among the trigger
candidates, and the junction adds a link term to that row. Decode never reads the anchor row (`decode_group`
copies the instance's own span and `continue`s; the coreferent merge and the column loss read list fields only).
This measures, on the TRAINING path (gradient_conflict's real data preparation, negatives and schedules):

  forward   per anchor row: its NLL, the probability on its target, whether its argmax is a target, the field count
            n it shares the 1/n mean with, and the junction's |bilinear| vs |geometry| on anchor vs role rows
  --grads   three replays per batch under one seed -- all terms, anchor term zeroed, role terms zeroed (the 1/n
            divisor unchanged) -- so the anchor term's exact loss and gradient are all - no_anchor and the role
            terms' are all - no_roles; per parameter group: |g_anchor|, |g_role|, cos(g_anchor, g_role)

Gates: a replay of the same batch must reproduce the loss, and on the first batch all - no_anchor - no_roles +
none = 0 (the split is linear and exact), or the tool refuses to report.

    uv run tools/train/probe_anchor_field.py --config tools/train/config/base/eb22d.yaml \\
        --checkpoint /Volumes/Development/tmp/eb22d_epochs/checkpoint-epoch-2 --step 33172 \\
        --only-corpora casie cmnee --records 40 --batches 12 --grads --out anchor_field.json
"""
from __future__ import annotations

import argparse
import json
import random
import resource
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gradient_conflict as G  # noqa: E402
from gliner2.models.boundary import records as R  # noqa: E402

MODE = ["all"]          # all | no_anchor | no_roles | none
LOG = [False]           # rows and link terms are logged on the first forward of a batch only
ROWS: list[dict] = []   # one per anchor row trained (forward stats)
LINK: list[tuple] = []  # (is_anchor, |bilinear|, |geometry|) per junction field term


def _is_anchor(group, fspec) -> bool:
    return group.spec.mode == "natural" and fspec.query_id == group.spec.anchor_query_id


def field_loss_split(group, inst, record, span_indices, hard_k=0, drop_anchor=False):
    """`records._instance_field_loss` with the anchor and role terms separable (MODE), divisor unchanged."""
    assert not drop_anchor, "this probe measures the anchor term: run it with record_drop_anchor_field_loss off"
    total, n_fields = group.object_logits.new_zeros(()), 0
    for f_idx, fspec in enumerate(group.field_specs):
        cols, is_scalar = R._field_target_cols(fspec, record, span_indices[f_idx])
        row = group.assign_logits[f_idx][inst]
        term = R._scalar_field_nll(row, cols) if is_scalar else R._list_field_bce(row, cols, hard_k)
        anchor = _is_anchor(group, fspec)
        if anchor and LOG[0]:
            p = torch.softmax(row.detach().float(), -1)
            ROWS.append({"nll": float(term), "p_target": float(p[cols].sum()) if cols else 0.0,
                         "argmax_is_target": int(p.argmax()) in cols, "self_in_target": inst + 1 in cols,
                         "scalar": is_scalar, "targets": len(cols), "candidates": row.shape[0] - 1,
                         "n_fields": len(group.field_specs)})
        keep = {"all": True, "no_anchor": not anchor, "no_roles": anchor, "none": False}[MODE[0]]
        if keep:
            total = total + term
        n_fields += 1
    return total / max(n_fields, 1)


def spy_link(link):
    real = link.forward

    def forward(inst_states, fq, cands, inst_spans, field_spans):
        out = real(inst_states, fq, cands, inst_spans, field_spans)
        if LOG[0]:
            for f, cand in enumerate(cands):
                if cand.shape[0] == 0:
                    continue
                geo = link.geom(link._geometry(inst_spans, field_spans[f]).to(cand.dtype))
                geo = (F.gelu(geo) if link.geom_activation == "gelu" else geo) @ link.geom_q(fq[f])
                LINK.append((field_spans[f] is inst_spans, float((out[f] - geo).abs().mean()),
                             float(geo.abs().mean())))
        return out
    link.forward = forward


def replay(model, batch, seed: int, mode: str, ps, log: bool = False):
    MODE[0], LOG[0] = mode, log
    torch.manual_seed(seed)
    random.seed(seed)
    out = model(batch)
    grads = torch.autograd.grad(out.loss, ps, allow_unused=True) if ps is not None else None
    loss = float(out.loss)
    del out
    return loss, grads


def diff(a, b, ps):
    return [(x if x is not None else torch.zeros_like(p)) - (y if y is not None else torch.zeros_like(p))
            for x, y, p in zip(a, b, ps)]


def summarize_rows() -> dict:
    n = len(ROWS)
    if not n:
        return {"rows": 0}
    mean = lambda k: sum(float(r[k]) for r in ROWS) / n
    nll = sorted(r["nll"] for r in ROWS)
    return {"rows": n, "nll_mean": mean("nll"), "nll_median": nll[n // 2], "nll_p90": nll[int(0.9 * (n - 1))],
            "p_target_mean": mean("p_target"), "argmax_is_target": mean("argmax_is_target"),
            "self_in_target": mean("self_in_target"), "scalar": mean("scalar"), "targets_mean": mean("targets"),
            "candidates_mean": mean("candidates"), "n_fields_mean": mean("n_fields"),
            "anchor_weight_in_mean": sum(1.0 / r["n_fields"] for r in ROWS) / n}


def summarize_link() -> dict:
    out = {}
    for name, sel in (("anchor", True), ("roles", False)):
        t = [x for x in LINK if x[0] is sel]
        if t:
            b, g = sum(x[1] for x in t) / len(t), sum(x[2] for x in t) / len(t)
            out[name] = {"terms": len(t), "bilinear": b, "geometry": g, "geometry_share": g / (b + g) if b + g else None}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="local dir or repo:subfolder")
    ap.add_argument("--step", type=int, required=True, help="the checkpoint's global step (schedules)")
    ap.add_argument("--planned-steps", type=int, default=82930)
    ap.add_argument("--only-corpora", nargs="+", required=True)
    ap.add_argument("--records", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=6000, help="skip longer records (laptop memory)")
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grads", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    pool = G.training_records(cfg, args.config, 10 ** 9, args.seed, set(args.only_corpora))
    pool = [r for r in pool if len(r.get("input") or r.get("text") or "") <= args.max_chars
            and (r.get("output") or {}).get("events")]
    records = random.Random(args.seed).sample(pool, min(args.records, len(pool)))
    print(f"[anchor] {len(records)} records with events from {sorted({r.get('_corpus') for r in records})}", flush=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model, trainer, loader = G.build(cfg, args.config, args.checkpoint, [dict(r) for r in records], args.batch_size)
    R._instance_field_loss = field_loss_split
    spy_link(model.record_decoder.link)
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    G.set_schedules(trainer, model, args.step, args.planned_steps)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.embeddings")]
    ps = [p for _, p in params] if args.grads else None
    sums = {k: [torch.zeros_like(p) for _, p in params] for k in ("anchor", "roles")} if args.grads else None
    losses = {"all": 0.0, "anchor": 0.0, "roles": 0.0}
    used = 0
    for batch in loader:
        if used >= args.batches:
            break
        L_all, g_all = replay(model, batch, used, "all", ps, log=True)
        if args.grads:
            L_again, _ = replay(model, batch, used, "all", None)
            assert abs(L_again - L_all) <= 1e-5 * max(1.0, abs(L_all)), f"replay not deterministic: {L_all} vs {L_again}"
            L_noA, g_noA = replay(model, batch, used, "no_anchor", ps)
            L_noR, g_noR = replay(model, batch, used, "no_roles", ps)
            if used == 0:
                L_none, _ = replay(model, batch, used, "none", None)
                resid = L_all - L_noA - L_noR + L_none
                assert abs(resid) <= 1e-5 * max(1.0, abs(L_all)), f"split not linear: residual {resid:.3e}"
                print(f"[anchor] gates: replay exact, linear split residual {resid:+.2e} (all {L_all:.6f}, no_anchor {L_noA:.6f}, "
                      f"no_roles {L_noR:.6f}, none {L_none:.6f})", flush=True)
            for k, other in (("anchor", g_noA), ("roles", g_noR)):
                for acc, v in zip(sums[k], diff(g_all, other, ps)):
                    acc += v
            losses["anchor"] += L_all - L_noA
            losses["roles"] += L_all - L_noR
        losses["all"] += L_all
        used += 1
        print(f"[anchor] batch {used}: input {tuple(batch.input_ids.shape)} | loss {L_all:.4f} | anchor rows so far "
              f"{len(ROWS)} | peak rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30:.1f} GB", flush=True)
    result = {"args": vars(args), "batches": used, "loss": losses, "anchor_rows": summarize_rows(), "link": summarize_link()}
    if args.grads:
        groups = {}
        for i, (n, _) in enumerate(params):
            groups.setdefault(G.group_of(n), []).append(i)
        rows = {}
        for g, idx in sorted(groups.items()):
            a = torch.cat([sums["anchor"][i].flatten() for i in idx])
            r = torch.cat([sums["roles"][i].flatten() for i in idx])
            if a.norm() > 0 or r.norm() > 0:
                rows[g] = {"anchor_norm": float(a.norm()), "roles_norm": float(r.norm()), "cos": G.cos(a, r)}
        result["groups"] = rows
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("batches", "loss", "anchor_rows", "link")}, indent=1), flush=True)
    for g, r in result.get("groups", {}).items():
        print(f"[anchor]   {g:36s} |g_anchor| {r['anchor_norm']:.4g}  |g_roles| {r['roles_norm']:.4g}  "
              f"cos {'n/a' if r['cos'] is None else format(r['cos'], '+.3f')}", flush=True)
    print(f"[anchor] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
