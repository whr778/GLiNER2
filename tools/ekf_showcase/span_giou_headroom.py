"""Does a GIoU-shaped span target have anything to bite on? MEASURED: essentially no.

GIoU's whole contribution over IoU is distinguishing a NEAR miss from a FAR miss:
both score IoU exactly 0 today. If the proposer's zero-IoU candidates are nearly all
far from gold, a GIoU-shaped target changes almost nothing and the $35 run is wasted.
Hooks build_candidate_labels so we measure exactly what the training loss sees.
"""
import json, sys, torch, collections
from gliner2 import AutoExtractor
from gliner2.training.trainer import ExtractorDataset
import gliner2.models.boundary.losses as L
import gliner2.models.boundary.model as M

SNAP = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 64

model = AutoExtractor.from_pretrained(SNAP)
model.eval()

recs = [json.loads(l) for l in
        open("data/casualty_events.train.jsonl", encoding="utf-8").read().splitlines()[:N]]
ds = ExtractorDataset(recs, shuffle=False, validate=False)
items = [ds[i] for i in range(len(ds))]

cap = []
orig = L.build_candidate_labels
def spy(cand_idx, cand_mask, gold_pairs, gold_mask, **kw):
    cap.append((cand_idx.detach().cpu(), cand_mask.detach().cpu(),
                gold_pairs.detach().cpu(), gold_mask.detach().cpu(), dict(kw)))
    return orig(cand_idx, cand_mask, gold_pairs, gold_mask, **kw)
L.build_candidate_labels = spy
M.build_candidate_labels = spy

bs = 8
for i in range(0, len(items), bs):
    batch = model.processor.collate_fn_train(
        items[i:i + bs], max_len=512, architecture="boundary",
        max_gold_per_query=64, on_capacity_exceeded="truncate_with_warning", error_policy="fallback",
        event_records=True,
    )
    with torch.no_grad():
        model(batch)

print(f"captured {len(cap)} calls to build_candidate_labels")

iou_hist = collections.Counter(); gap_hist = collections.Counter()
gio_hist = collections.Counter()
tot = zero = 0
for cand_idx, cand_mask, gold_pairs, gold_mask, kw in cap:
    pooled = cand_idx.dim() == 3
    if pooled:
        c = cand_idx.unsqueeze(2).unsqueeze(3); g = gold_pairs.unsqueeze(1)
        cm = cand_mask.unsqueeze(-1)
    else:
        qa, ca = kw.get("query_axis", 1), kw.get("candidate_axis", 2)
        ci = torch.movedim(cand_idx, (qa, ca), (1, 2))
        cm = torch.movedim(cand_mask, (qa, ca), (1, 2)) if cand_mask.dim() == 3 else cand_mask
        c = ci.unsqueeze(3); g = gold_pairs.unsqueeze(2)
    cs, ce = c[..., 0], c[..., 1]
    gs, ge = g[..., 0], g[..., 1]
    gm = (gold_mask.unsqueeze(1) if pooled else gold_mask.unsqueeze(2))
    inter = (torch.minimum(ce, ge) - torch.maximum(cs, gs)).clamp_min(0)
    union = (ce - cs) + (ge - gs) - inter
    iou = inter.float() / union.clamp_min(1).float()
    iou = iou * gm.float()
    # token gap between intervals (0 when they touch/overlap)
    gap = torch.maximum(gs - ce, cs - ge).clamp_min(0)
    gap = torch.where(gm, gap, torch.full_like(gap, 10_000))
    has_gold = gm.any(-1)
    best_iou = iou.amax(-1)
    best_gap = gap.amin(-1)
    valid = has_gold & (cm.bool() if cm.dim() == best_iou.dim() else cm.bool().unsqueeze(-1))
    v = valid.reshape(-1); bi = best_iou.reshape(-1)[v]; bg = best_gap.reshape(-1)[v]
    tot += bi.numel()
    iou_hist["exact_0"] += int((bi == 0).sum())
    iou_hist["0<iou<0.5"] += int(((bi > 0) & (bi < 0.5)).sum())
    iou_hist["0.5<=iou<1"] += int(((bi >= 0.5) & (bi < 1)).sum())
    iou_hist["iou==1"] += int((bi == 1).sum())
    # what target would each scheme assign to the zero-IoU candidates?
    enclose = (torch.maximum(ce, ge) - torch.minimum(cs, gs)).clamp_min(1)
    giou = iou - (enclose - union).clamp_min(0).float() / enclose.float()
    giou = torch.where(gm, giou, torch.full_like(giou, -1.0))
    best_giou = giou.amax(-1).reshape(-1)[v]
    rescaled = (best_giou + 1.0) * 0.5
    rz = rescaled[bi == 0]
    for lo, hi, name in [(0, .02, "giou_t <0.02"), (.02, .10, "giou_t 0.02-0.10"),
                         (.10, .25, "giou_t 0.10-0.25"), (.25, 1.01, "giou_t >0.25")]:
        gio_hist[name] += int(((rz >= lo) & (rz < hi)).sum())
    z = bg[bi == 0]
    zero += z.numel()
    for lo, hi, name in [(0, 1, "gap 0"), (1, 4, "gap 1-3"), (4, 11, "gap 4-10"),
                         (11, 51, "gap 11-50"), (51, 9999, "gap >50")]:
        gap_hist[name] += int(((z >= lo) & (z < hi)).sum())

print(f"\ncandidate-query pairs scored: {tot:,}")
for k in ("exact_0", "0<iou<0.5", "0.5<=iou<1", "iou==1"):
    print(f"  {k:12} {iou_hist[k]:>10,} ({iou_hist[k]/tot:6.2%})")
print(f"\nof the {zero:,} EXACT-ZERO-IoU candidates, distance to nearest gold span:")
for k in ("gap 0", "gap 1-3", "gap 4-10", "gap 11-50", "gap >50"):
    print(f"  {k:12} {gap_hist[k]:>10,} ({gap_hist[k]/max(zero,1):6.2%})")
print(f"\nrescaled-GIoU target these zero-IoU candidates would receive (BCE needs [0,1]):")
for k in ("giou_t <0.02", "giou_t 0.02-0.10", "giou_t 0.10-0.25", "giou_t >0.25"):
    print(f"  {k:18} {gio_hist[k]:>10,} ({gio_hist[k]/max(zero,1):6.2%})")
moved = zero - gio_hist["giou_t <0.02"]
print(f"  -> moved off ~zero: {moved:,} = {moved/tot:.2%} of ALL candidates")
import math
floor = sum(gap_hist[k] for k in ("gap 0", "gap 1-3"))
print(f"\ndecayed floor 0.1*exp(-gap/2) would move: gap0->0.100, gap1-3->0.022-0.061,")
print(f"  gap4-10->0.0007-0.014, gap>10-> ~0   => {floor:,} = {floor/tot:.2%} of ALL candidates")
near = gap_hist["gap 0"] + gap_hist["gap 1-3"]
print(f"\nHEADLINE: zero-IoU candidates within 3 tokens of gold = {near:,}")
print(f"          share of ALL candidates: {near/tot:.3%}")
print(f"          share of zero-IoU ones : {near/max(zero,1):.2%}")
