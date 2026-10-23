"""Classifier-head A/B on cached frozen-encoder features: ReLU (upstream's head) vs GELU vs fg-GEGLU.

Mirrors the real head: create_mlp(768 -> 1536 -> 1, dropout 0.1), one logit per label state, BCE per label
(the model's _classification_loss), argmax over the record's menu for the prediction. Trains to a fixed budget with
AdamW, picks the epoch on VAL accuracy, scores the blind TEST once at that epoch. Reports accuracy and macro F1 on
val and test, and the val-test gap, mean over seeds. Usage: cls_head_ab.py <cache_dir> <corpus> [epochs] [seeds]
"""
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, "/Users/williamroe/development/git/GLiNER2")
from gliner2.models.boundary.flipped_gate import FlippedGateGELU  # noqa: E402

cache, corpus = sys.argv[1], sys.argv[2]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
HID = 1536


def load(split):
    d = torch.load(f"{cache}/{corpus}.{split}.pt")
    return d["feats"], [torch.tensor(g) for g in d["gold"]]


def head(act):
    a = {"relu": nn.ReLU(), "gelu": nn.GELU(), "fg12": FlippedGateGELU(12, 3, 3, eval_mode="mc", mc_samples=8),
         "fg24": FlippedGateGELU(24, 4, 4, eval_mode="mc", mc_samples=8)}[act]
    return nn.Sequential(nn.Linear(768, HID), a, nn.Dropout(0.1), nn.Linear(HID, 1))     # = create_mlp's layout


def batches(feats, golds, bs, g):
    order = torch.randperm(len(feats), generator=g).tolist()
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        yield torch.cat([feats[j] for j in idx]), torch.cat([golds[j] for j in idx]), [feats[j].shape[0] for j in idx]


def evaluate(m, feats, golds):
    m.eval(); correct = 0; tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    with torch.no_grad():
        torch.manual_seed(999)
        for f, g in zip(feats, golds):
            logits = m(f).squeeze(-1); pred = int(logits.argmax()); gold = int(g.argmax())
            correct += pred == gold
            if pred == gold: tp[gold] += 1
            else: fp[pred] += 1; fn[gold] += 1
    labels = set(tp) | set(fp) | set(fn)
    f1s = [2 * tp[l] / max(2 * tp[l] + fp[l] + fn[l], 1) for l in labels]
    return correct / len(feats), sum(f1s) / max(len(f1s), 1)


tr_f, tr_g = load("train"); va_f, va_g = load("val"); te_f, te_g = load("test")
print(f"{corpus}: train {len(tr_f)} val {len(va_f)} test {len(te_f)} | menu {tr_f[0].shape[0]} labels x {tr_f[0].shape[1]} | {EPOCHS} epochs, {SEEDS} seeds", flush=True)
print(f"{'act':5s} {'val acc':>8s} {'test acc':>9s} {'gap':>7s} | {'val mF1':>8s} {'test mF1':>9s} {'gap':>7s} | best epoch (per seed) | per-seed test acc", flush=True)
for act in ("relu", "gelu", "fg12", "fg24"):
    rows = []
    for seed in range(SEEDS):
        torch.manual_seed(seed); g = torch.Generator().manual_seed(100 + seed)
        m = head(act); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
        best = (-1.0, None, None)
        for ep in range(EPOCHS):
            m.train()
            for x, y, _ in batches(tr_f, tr_g, 32, g):
                opt.zero_grad(); F.binary_cross_entropy_with_logits(m(x).squeeze(-1), y).backward(); opt.step()
            va = evaluate(m, va_f, va_g)
            if va[0] > best[0]:
                best = (va[0], ep, {k: v.clone() for k, v in m.state_dict().items()}, va)
        m.load_state_dict(best[2]); te = evaluate(m, te_f, te_g)
        rows.append((best[3], te, best[1]))
    va_acc = sum(r[0][0] for r in rows) / SEEDS; te_acc = sum(r[1][0] for r in rows) / SEEDS
    va_f1 = sum(r[0][1] for r in rows) / SEEDS; te_f1 = sum(r[1][1] for r in rows) / SEEDS
    print(f"{act:5s} {va_acc:8.4f} {te_acc:9.4f} {va_acc - te_acc:+7.4f} | {va_f1:8.4f} {te_f1:9.4f} {va_f1 - te_f1:+7.4f} | {[r[2] for r in rows]} | {[round(r[1][0], 3) for r in rows]}", flush=True)
