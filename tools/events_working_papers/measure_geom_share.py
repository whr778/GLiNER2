"""FREE CHECK on a TRAINED checkpoint: how much of the junction is the geometry channel? For every natural group
with gold owners on real val records (train-mode forward, no grad), per list field: mean |bilinear| vs mean |geometry|
of the junction term, and the column-loss win rate (gold trigger beats false trigger) with the real logits vs with the
geometry term removed. Usage: measure_geom_share.py <repo:checkpoint-N> <n records per corpus>"""
import json, os, random, resource, sys
import torch
sys.path.insert(0, "/Users/williamroe/development/git/GLiNER2"); sys.path.insert(0, "/Users/williamroe/development/git/GLiNER2/tools/train")
import gliner2.models.boundary.records as R
from gliner2.training import ExtractorCollator
import probe_event_collapse as P
ref, n = sys.argv[1], int(sys.argv[2])
model = P.load_checkpoint(ref); link = model.record_decoder.link
print("junction geom width:", link.geom.weight.shape[0], "| |geom.W| mean", float(link.geom.weight.abs().mean()), "| |geom_q.W| mean", float(link.geom_q.weight.abs().mean()), "| |v.W| mean", float(link.v.weight.abs().mean()))
coll = ExtractorCollator(model.processor, is_training=True, architecture="boundary", event_records=True)
terms = []  # per (field): (|B|, |G|, |B+G|)
geo_cache = []
real_fwd = link.forward
def spy(inst_states, fq, cands, inst_spans, field_spans):
    out = real_fwd(inst_states, fq, cands, inst_spans, field_spans)
    geos = []
    for f, cand in enumerate(cands):
        if cand.shape[0] == 0: geos.append(None); continue
        geo = link.geom(link._geometry(inst_spans, field_spans[f]).to(cand.dtype)) @ link.geom_q(fq[f])
        geos.append(geo); terms.append((float((out[f] - geo).abs().mean()), float(geo.abs().mean()), float(out[f].abs().mean())))
    geo_cache.append(geos); return out
link.forward = spy
stats = {}
real_cgl = R.compute_group_loss
def cgl_spy(g, rs, *a, **k):
    if g.spec.mode == "natural" and rs and geo_cache:
        geos = geo_cache[-1]; spans = [R._span_index(s) for s in g.field_spans]
        _, t, w, p, *_ = R._column_loss(g, rs, g.spec.anchor_query_id, spans, 8)
        if t:
            g2 = R.RecordGroupOutput(**{**g.__dict__, "assign_logits": [a if ge is None else torch.cat([a[:, :1], a[:, 1:] - ge], dim=-1) for a, ge in zip(g.assign_logits, geos)]})
            _, t2, w2, p2, *_ = R._column_loss(g2, rs, g2.spec.anchor_query_id, spans, 8)
            s = stats.setdefault(corpus, [0, 0, 0, 0, 0]); s[0] += t; s[1] += w; s[2] += p; s[3] += w2; s[4] += p2
    return real_cgl(g, rs, *a, **k)
R.compute_group_loss = cgl_spy
model.train(); torch.manual_seed(0); random.seed(0)
for corpus, path in (("casie", "data/scaling_joint/casie.val.jsonl"), ("cmnee", "data/cmnee.val.jsonl")):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")][:n]
    for i, rec in enumerate(recs):
        geo_cache.clear()
        with torch.no_grad():
            model(coll([(rec["input"], rec["output"])]))
        if i % 5 == 4: print(f"  {corpus} {i+1}/{n} done, maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.1f} GB", flush=True)
    if terms:
        b = sum(t[0] for t in terms) / len(terms); g = sum(t[1] for t in terms) / len(terms)
        print(f"{corpus}: {len(terms)} field terms | mean |bilinear| {b:.4f} | mean |geometry| {g:.4f} | geometry share {g/(b+g):.3f}")
    terms.clear()
for c, (t, w, p, w2, p2) in stats.items():
    print(f"{c}: column terms {t} | gold-beats-false with geometry {w/max(p,1):.3f} of {p} pairs | WITHOUT geometry {w2/max(p2,1):.3f} of {p2}")
