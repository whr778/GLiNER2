"""Baseline: do trigger-instance states separate coreferent mention pairs from DIFFERENT same-type events?

Pairwise AUC of three scores (raw instance state, inst_proj state, argument-score rows) on real
training-shaped batches. COREFERENT_LINK_SPEC.md section 1: rows 0.468 (below chance), state 0.685,
proj 0.709 on p5link-junc_w03 x 40 cc_news_events_sonnet55_v2 docs.
"""
import json, glob, random, sys
import torch, yaml
import torch.nn.functional as F
import gliner2.models.boundary.records as R
from gliner2 import AutoExtractor
from gliner2.training import ExtractorCollator
cfg = yaml.safe_load(open("tools/train/config/base/eb19.yaml")); tr = cfg["training"]; head = cfg["model"]["boundary_head"]
M = glob.glob("/Users/williamroe/.cache/huggingface/hub/models--whr778--gliner2-p5link-junc_w03/snapshots/*/")[0]
model = AutoExtractor.from_pretrained(M, map_location="cpu"); dec = model.record_decoder
model.processor.sampling_config.remove_events_prob = 0.0
recs = [json.loads(l) for l in open("data/cc_news_events_sonnet55_v2.train.jsonl", encoding="utf-8")]
def useful(r):
    evs = r["output"].get("events") or []
    from collections import Counter
    return any(len(e["triggers"]) > 1 for e in evs) and any(n > 1 for n in Counter(e["event_type"] for e in evs).values()) and len(r["input"]) < 2500
pick = random.Random(0).sample([r for r in recs if useful(r)], 40)
coll = ExtractorCollator(model.processor, is_training=True, max_len=tr.get("max_len"), architecture="boundary",
                         max_gold_per_query=int(head.get("max_gold_per_query", 32)), on_capacity_exceeded="truncate_with_warning",
                         error_policy="skip", event_records=True)
captured, scores = [], {"state": ([], []), "proj": ([], []), "rows": ([], [])}
real_assign = dec._assign_logits
def spy(inst_states, fq, fc):
    captured.append(inst_states.detach()); return real_assign(inst_states, fq, fc)
dec._assign_logits = spy
real_loss = R.compute_group_loss
def loss_spy(group, records, *a, **k):
    st = captured.pop(0) if captured else None
    if st is None or group.spec.mode != "natural" or group.spec.task_type != "events" or len(records) < 2:
        return real_loss(group, records, *a, **k)
    aq = group.spec.anchor_query_id; af = group.field_query_ids.index(aq)
    spans = R._span_index(group.field_spans[af])
    seed = {s[1]: i for i, s in enumerate(group.instance_seed) if s is not None and s[0] == af}
    owner = {}
    for ri, rec in enumerate(records):
        ft = rec.field_for_query(aq)
        for c in R._resolve_value_cols(list(ft.values[0]) if ft is not None and ft.values else [], spans):
            if (c - 1) in seed: owner.setdefault(seed[c - 1], ri)
    if len(set(owner.values())) < 2: return real_loss(group, records, *a, **k)
    insts = sorted(owner)
    proj = dec.inst_proj(st)
    rows = torch.cat([torch.sigmoid(group.assign_logits[f][:, 1:].detach()) for f, fs in enumerate(group.field_specs) if not fs.cardinality.is_scalar], dim=1)
    for x in range(len(insts)):
        for y in range(x + 1, len(insts)):
            i, j = insts[x], insts[y]; same = owner[i] == owner[j]
            for name, mat in (("state", st), ("proj", proj), ("rows", rows)):
                scores[name][0 if same else 1].append(float(F.cosine_similarity(mat[i], mat[j], dim=0)))
    return real_loss(group, records, *a, **k)
R.compute_group_loss = loss_spy
model.train()
for m in model.modules():
    if isinstance(m, torch.nn.Dropout): m.eval()
for b in range(0, len(pick), 4):
    random.seed(b); torch.manual_seed(b); captured.clear()
    with torch.no_grad(): model(coll([(r["input"], r["output"]) for r in pick[b:b+4]]))
def auc(pos, neg):
    p, n = torch.tensor(pos), torch.tensor(neg)
    return float(((p[:, None] > n[None, :]).float() + 0.5 * (p[:, None] == n[None, :]).float()).mean())
for name, (pos, neg) in scores.items():
    print(f"{name:6s} coreferent pairs {len(pos):4d} vs different same-type events {len(neg):5d} | AUC {auc(pos, neg):.3f} | mean cos {sum(pos)/len(pos):.3f} vs {sum(neg)/len(neg):.3f}")
