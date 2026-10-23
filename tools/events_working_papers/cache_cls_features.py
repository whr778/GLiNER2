"""Cache the classifier's inputs from the frozen eb22c control encoder: for every record of a classification corpus
and split, the per-label choice states the classifier MLP scores ([n_labels, 768]) and the multi-hot gold.

The TRAINING collate is used (it carries the gold as structure_labels) with every classification sampling knob off:
no label removal, no synthetic names, no shuffling, so the cached menu is the record's own menu in its own order.
Usage: cache_cls_features.py <corpus> <max_train> <max_val> <max_test> <out_dir>
"""
import json, random, sys, time
from pathlib import Path

import torch

sys.path.insert(0, "/Users/williamroe/development/git/GLiNER2"); sys.path.insert(0, "/Users/williamroe/development/git/GLiNER2/tools/train")
import train as T  # noqa: E402
from gliner2 import AutoExtractor  # noqa: E402
from gliner2.training import ExtractorCollator  # noqa: E402
from gliner2.training.trainer import ExtractorDataset, apply_schema_dropout  # noqa: E402

corpus, mx_tr, mx_va, mx_te, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
out.mkdir(parents=True, exist_ok=True)
CK = "/Volumes/Development/tmp/autopsy_ckpts/whr778__gliner2-eb22c-collapse-trace/checkpoint-30000"
model = AutoExtractor.from_pretrained(CK, map_location="cpu").eval()
sc = model.processor.sampling_config
changed = apply_schema_dropout(sc, {"remove_classification_prob": 0.0, "remove_classification_label_prob": 0.0, "synthetic_label_prob": 0.0, "include_true_label_prob": 0.0})
sc.shuffle_classification_labels = False
print(f"[cache] {corpus}: sampling off for classification {changed}, shuffle_classification_labels={sc.shuffle_classification_labels}", flush=True)
coll = ExtractorCollator(model.processor, is_training=True, max_len=4096, architecture="boundary", max_gold_per_query=768,
                         on_capacity_exceeded="truncate_with_warning", error_policy="skip", event_records=True)
for split, mx in (("train", mx_tr), ("val", mx_va), ("test", mx_te)):
    rows = [json.loads(l) for l in open(f"data/{corpus}.{split}.jsonl", encoding="utf-8")]
    random.Random(0).shuffle(rows); rows = rows[:mx]
    for r in rows: r["_corpus"] = corpus
    ds = ExtractorDataset(data=rows, max_samples=-1, shuffle=False, seed=0, validate=False)
    feats, golds, menus, t0, skipped = [], [], [], time.time(), 0
    for b0 in range(0, len(ds), 8):
        items = [ds[i] for i in range(b0, min(b0 + 8, len(ds)))]
        batch = coll(items)
        with torch.no_grad():
            core = model._encode_core(batch)
        for i, item in enumerate(items):
            specs = core["cls_specs"][i]
            if not specs:
                skipped += 1; continue
            cls = specs[0]
            states = cls["choice_states"] if "choice_states" in cls else cls["group_embs"][1:]
            labels_raw = batch.structure_labels[i][cls["group_index"]]
            menu = rows[b0 + i]["output"]["classifications"][0]["labels"]
            if len(labels_raw) != states.shape[0] or len(menu) != states.shape[0]:
                skipped += 1; continue
            feats.append(states.detach().float().cpu()); golds.append([float(x) for x in labels_raw]); menus.append(menu)
        if b0 % 400 == 0:
            print(f"[cache] {corpus} {split}: {b0 + len(items)}/{len(ds)} records, {time.time() - t0:.0f} s", flush=True)
    torch.save({"feats": feats, "gold": golds, "menus": menus}, out / f"{corpus}.{split}.pt")
    print(f"[cache] {corpus} {split}: saved {len(feats)} records ({skipped} skipped) in {time.time() - t0:.0f} s | widths {sorted({f.shape[0] for f in feats})} x {feats[0].shape[1]}", flush=True)
