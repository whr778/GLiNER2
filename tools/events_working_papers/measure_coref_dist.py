"""FREE CHECK on a TRAINED checkpoint: how much of TriggerLink is its scalar distance bias? Per natural group with gold
owners: mean |bilinear| vs mean |dist term| over the gold x (gold + false) pairs the coref loss ranks, and the coref pair
AUC (coreferent pairs outscore non-coreferent, the loss's own accounting) with the real logits vs with the distance
term removed. Usage: measure_coref_dist.py <repo:checkpoint-N> <n records per corpus>"""
import json, random, sys
import torch
sys.path.insert(0, "/Users/williamroe/development/git/GLiNER2"); sys.path.insert(0, "/Users/williamroe/development/git/GLiNER2/tools/train")
import gliner2.models.boundary.records as R
from gliner2.training import ExtractorCollator
import probe_event_collapse as P
ref, n = sys.argv[1], int(sys.argv[2])
model = P.load_checkpoint(ref); link = model.record_decoder.coref
if isinstance(link.dist, torch.nn.Sequential): print(f"TriggerLink dist: MLP width {link.dist[0].weight.shape[0]} | |W| mean {float(link.w.weight.abs().mean()):.4f}")
else: print(f"TriggerLink dist: weight {float(link.dist.weight):.4f} bias {float(link.dist.bias):.4f} | |W| mean {float(link.w.weight.abs().mean()):.4f}")
coll = ExtractorCollator(model.processor, is_training=True, architecture="boundary", event_records=True)
stats = {}
real_cgl = R.compute_group_loss
def cgl_spy(g, rs, *a, **k):
    if g.spec.mode == "natural" and rs and g.coref_logits is not None:
        spans = [R._span_index(s) for s in g.field_spans]
        _, npairs, w, r = R._coref_link_loss(g, rs, g.spec.anchor_query_id, spans, 4)
        if npairs:
            inst_spans = torch.tensor(g.instance_spans, dtype=torch.float32)
            mid = (inst_spans[:, 0] + inst_spans[:, 1]) / 2
            dist = torch.log1p((mid[:, None] - mid[None, :]).abs()).unsqueeze(-1).to(g.coref_logits.dtype)
            dterm = link.dist(dist).squeeze(-1)
            g2 = R.RecordGroupOutput(**{**g.__dict__, "coref_logits": g.coref_logits - dterm})
            _, _, w2, r2 = R._coref_link_loss(g2, rs, g2.spec.anchor_query_id, spans, 4)
            s = stats.setdefault(corpus, [0, 0, 0, 0, 0, 0.0, 0.0, 0]); s[0] += npairs; s[1] += w; s[2] += r; s[3] += w2; s[4] += r2
            bil = (g.coref_logits - dterm).abs().mean(); s[5] += float(bil); s[6] += float(dterm.abs().mean()); s[7] += 1
    return real_cgl(g, rs, *a, **k)
R.compute_group_loss = cgl_spy
model.train(); torch.manual_seed(0); random.seed(0)
for corpus, path in (("casie", "data/scaling_joint/casie.val.jsonl"), ("cmnee", "data/cmnee.val.jsonl"), ("wikievents_coref", "data/wikievents_coref_llm.val.jsonl")):
    try: recs = [json.loads(l) for l in open(path, encoding="utf-8")][:n]
    except FileNotFoundError: print(corpus, "no local val file"); continue
    for rec in recs:
        with torch.no_grad(): model(coll([(rec["input"], rec["output"])]))
    print(corpus, "done", flush=True)
for c, (npairs, w, r, w2, r2, b, d, ng) in stats.items():
    print(f"{c}: {ng} groups, {npairs} ranked pairs | mean |bilinear| {b/ng:.3f} vs |dist term| {d/ng:.3f} (share {d/(b+d):.3f}) | "
          f"coref AUC with dist {w/max(r,1):.3f} of {r} | WITHOUT dist {w2/max(r2,1):.3f} of {r2}")
