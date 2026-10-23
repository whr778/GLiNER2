"""TRACE: which gold relation pairs reach the relation loss, on a real training batch of the 30k control.

For every gold (head, tail) pair the collator emits (relation_gold_pairs: nearest), follow it through the pair
generator: is each endpoint span in the candidate pool at all, what is its best pair probability over the member
queries, does it clear relation_argument_proposal_threshold, does it survive the per-relation head/tail top-k, and is
the pair among the pair_cap kept pairs (i.e. labelled positive in the loss). Prints one line per gold pair and totals.
Usage: trace_relation_gold_coverage.py <corpus> <n records> [checkpoint dir]
"""
import random
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, "/Users/williamroe/development/git/GLiNER2"); sys.path.insert(0, "/Users/williamroe/development/git/GLiNER2/tools/train")
import train as T  # noqa: E402
from gliner2 import AutoExtractor  # noqa: E402
from gliner2.training import ExtractorCollator  # noqa: E402
from gliner2.training.trainer import ExtractorDataset  # noqa: E402
import gliner2.models.boundary.model as M  # noqa: E402

corpus, n = sys.argv[1], int(sys.argv[2])
import os
if os.environ.get("SEED") is not None:
    torch.manual_seed(int(os.environ["SEED"])); random.seed(int(os.environ["SEED"])); print(f"[trace] seeded {os.environ['SEED']}")
ck = sys.argv[3] if len(sys.argv) > 3 else "/Volumes/Development/tmp/autopsy_ckpts/whr778__gliner2-eb22c-collapse-trace/checkpoint-30000"
cfg_path = "tools/train/config/base/eb22c-ddp-trace.yaml"
cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
tr, head = cfg["training"], cfg["model"]["boundary_head"]
fns, passthrough = T._category_fns(T.load_labels_cfg(cfg, cfg_path)), T.labels_passthrough(cfg)
records = [r for r in T.read_transformed([f"data/{corpus}.train.jsonl"], fns, passthrough) if r["output"].get("relations")][:n]
for r in records:
    r["_corpus"] = corpus
model = AutoExtractor.from_pretrained(ck, map_location="cpu")
s = model.boundary_settings
import os
gs = model.relation_pair_generator.settings
print(f"[trace] generator settings: argument_threshold {gs.argument_threshold} pair_cap {gs.pair_cap} (boundary_settings.relation_argument_proposal_threshold {s.relation_argument_proposal_threshold})")
if os.environ.get("KEEP") == "1":
    import dataclasses, logging
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    model.boundary_settings = dataclasses.replace(model.boundary_settings, relation_keep_gold_pairs=True)
    print(f"[trace] OVERRIDE relation_keep_gold_pairs -> {model.boundary_settings.relation_keep_gold_pairs}")
if os.environ.get("THR") is not None:
    import dataclasses
    gs = dataclasses.replace(gs, argument_threshold=float(os.environ["THR"])); model.relation_pair_generator.settings = gs
    s = dataclasses.replace(s, relation_argument_proposal_threshold=gs.argument_threshold) if dataclasses.is_dataclass(s) else s
    print(f"[trace] OVERRIDE argument_threshold -> {gs.argument_threshold}")
print(f"[trace] {corpus}: {len(records)} records, {sum(len(r['output']['relations']) for r in records)} gold relations | "
      f"argument_threshold {s.relation_argument_proposal_threshold} pair_cap {s.relation_pair_cap} "
      f"heads/tails per relation {model.relation_pair_generator.settings.heads_per_relation}/{model.relation_pair_generator.settings.tails_per_relation} "
      f"| relation_gold_pairs {tr.get('relation_gold_pairs')}")
ds = ExtractorDataset(data=records, max_samples=-1, shuffle=False, seed=42, validate=False)
coll = ExtractorCollator(model.processor, is_training=True, max_len=tr.get("max_len"), architecture="boundary",
                         max_gold_per_query=int(head.get("max_gold_per_query", 32)), on_capacity_exceeded="truncate_with_warning",
                         error_policy=tr.get("error_policy", "skip"), event_records=bool(head.get("event_records", False)),
                         relation_gold_pairs=tr.get("relation_gold_pairs", "all"))
batch = coll([ds[i] for i in range(len(ds))])

cap = {}
orig_gen = model.relation_pair_generator.generate_batched
def gen_spy(candidates, layouts, schemas, compact=False, routing=None):
    pairs = orig_gen(candidates, layouts, schemas, compact=compact, routing=routing)
    cap.update(candidates=candidates, routing=routing, pairs=pairs); return pairs
model.relation_pair_generator.generate_batched = gen_spy
orig_loss = M.BoundaryExtractorModel._relation_loss
def loss_spy(self, batch, core, candidates, targets=None):
    cap["edge_targets"] = getattr(targets, "edge_targets", None); return orig_loss(self, batch, core, candidates, targets)
M.BoundaryExtractorModel._relation_loss = loss_spy
model.train()
with torch.no_grad():
    out = model(batch)
print(f"[trace] relation_loss {float(out.losses['relation_loss']):.4f}")

cands, pairs, et = cap["candidates"], cap["pairs"], cap["edge_targets"]
gold_pairs, gold_mask = et[0], et[1]
head_member, tail_member = et[2], et[3]
probs = torch.sigmoid(cands.pair_logits)                       # [B, Q, C]
kept = torch.stack((pairs.head_start, pairs.head_end, pairs.tail_start, pairs.tail_end), -1)   # [P, 4]
kept_valid = pairs.pair_mask if pairs.pair_mask is not None else torch.ones(len(pairs.batch_index), dtype=torch.bool)
thr = s.relation_argument_proposal_threshold
tot = dict(gold=0, head_in_pool=0, tail_in_pool=0, head_ok=0, tail_ok=0, kept=0)

def best_prob(b, span, member):
    """Best pair probability of this span over the relation's member queries, and whether it is in the pool at all."""
    hit = (cands.indices[b] == span).all(-1) & cands.valid_mask[b]            # [Q, C]
    hit &= member[:, None]
    if not hit.any():
        return None
    return float(probs[b][hit].max())

for b in range(gold_pairs.shape[0]):
    for r in range(gold_pairs.shape[1]):
        for k in range(gold_pairs.shape[2]):
            if not gold_mask[b, r, k]:
                continue
            g = gold_pairs[b, r, k]; hspan, tspan = g[:2], g[2:]
            hp = best_prob(b, hspan, head_member[b, r]); tp_ = best_prob(b, tspan, tail_member[b, r])
            sel = (pairs.batch_index == b) & (pairs.relation_index == r) & kept_valid
            in_kept = bool(((kept[sel] == g).all(-1)).any()) if sel.any() else False
            tot["gold"] += 1; tot["head_in_pool"] += hp is not None; tot["tail_in_pool"] += tp_ is not None
            tot["head_ok"] += (hp or 0) >= thr; tot["tail_ok"] += (tp_ or 0) >= thr; tot["kept"] += in_kept
            print(f"[gold] sample {b} rel {r}: head {hspan.tolist()} p={hp if hp is None else round(hp, 3)} | tail {tspan.tolist()} "
                  f"p={tp_ if tp_ is None else round(tp_, 3)} | >= {thr}: {(hp or 0) >= thr and (tp_ or 0) >= thr} | kept as positive: {in_kept}")
print(f"[trace] TOTALS {tot} | kept pairs {int(kept_valid.sum())} of {len(kept_valid)} slots | positives/kept {tot['kept'] / max(int(kept_valid.sum()), 1):.3f}")
