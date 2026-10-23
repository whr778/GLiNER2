"""HELD-OUT junction read for Issue 4: on val records (train-mode forward, no grad, gold-seeded instances), the real
_column_loss replayed with siblings=True -- gold trigger beats false trigger / beats same-type sibling trigger, per corpus.
The in-run gate measures this on TRAINING batches; this is the same quantity on VAL. Usage: junction_heldout.py <repo:checkpoint-N> [n_rams]"""
import json, random, sys
import torch
sys.path.insert(0, "/Users/williamroe/development/git/GLiNER2"); sys.path.insert(0, "/Users/williamroe/development/git/GLiNER2/tools/train")
import gliner2.models.boundary.records as R
from gliner2.training import ExtractorCollator
import probe_event_collapse as P
ref = sys.argv[1]; n_rams = int(sys.argv[2]) if len(sys.argv) > 2 else 100
import train as T
from gliner2.processing.targets import TargetCapacityError
model = P.load_checkpoint(ref)
T._apply_boundary_head_overrides(model, dict(max_gold_per_query=768, training_candidate_budget=768))   # the TRAINING caps (eb22c config), not the checkpoint's inference caps
skipped = 0
coll = ExtractorCollator(model.processor, is_training=True, architecture="boundary", event_records=True)
stats = {}
real = R.compute_group_loss
def spy(g, rs, *a, **k):
    if g.spec.mode == "natural" and rs:
        spans = [R._span_index(s) for s in g.field_spans]
        _, terms, wins, pairs, sw, sp = R._column_loss(g, rs, g.spec.anchor_query_id, spans, 8, siblings=True)
        s = stats.setdefault(corpus, [0, 0, 0, 0, 0, 0]); s[0] += 1; s[1] += terms; s[2] += wins; s[3] += pairs; s[4] += sw; s[5] += sp
    return real(g, rs, *a, **k)
R.compute_group_loss = spy
model.train(); torch.manual_seed(0); random.seed(0)
for corpus, path, n in (("casie", "data/scaling_joint/casie.val.jsonl", 95), ("rams", "data/rams.val.jsonl", n_rams)):
    recs = [json.loads(l) for l in open(path, encoding="utf-8")][:n]
    for rec in recs:
        try:
            with torch.no_grad(): model(coll([(rec["input"], rec["output"])]))
        except TargetCapacityError:
            skipped += 1
print("records skipped (capacity):", skipped)
for c, (ng, t, w, p, sw, sp) in stats.items():
    print(f"{ref.split(':')[-1]} {c}: {ng} groups, {t} gold-argument columns | gold>false {w/max(p,1):.3f} of {p} pairs | gold>sibling {sw/max(sp,1):.3f} of {sp} sibling pairs")
