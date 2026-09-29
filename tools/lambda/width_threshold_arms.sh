#!/bin/bash
# DOES THE WIDER PROPOSAL HELP ONCE THE THRESHOLD MOVES WITH IT?
#
# WHY THIS EXISTS. Lifting the proposal cap raises gold COVERAGE at a 4096-token window
# from 8.1% to 18.7%. Scored at the SHIPPED threshold of 0.3 that produced 183,031 entity
# predictions against 2,942 correct -- precision 0.353 -> 0.0128, F1 0.0668 -> 0.0198. That
# is not evidence the width is useless: 0.3 was tuned for a decoder offering 8x FEWER
# candidates, so the operating point necessarily moved and holding it fixed guarantees a
# false-positive flood. This programme has made that exact mistake before -- the record gate
# shipped at 0.5 and its real operating point was 0.1, worth +0.0579 event_argument strict.
#
# ARMS. One variable per comparison, all with global_decode ON and the model's own window:
#   S03   SHIPPED width, threshold 0.3   <- the incumbent, the reference for everything
#   W03   WIDE width,    threshold 0.3   <- reproduces the flood, proves the width is live
#   W05 W07 W09  WIDE width, rising threshold
# W* minus S03 is the width effect AT ITS OWN OPERATING POINT. If no W beats S03 on F1, the
# width does not pay on long documents and item 1 is a coverage result with no metric behind
# it. If some W does, the shipped threshold was the problem, not the cap.
#
# FEWER DOCS ON PURPOSE. An arm costs ~56 min at WIDE on 300 docs, so five arms would be
# ~5h. At 60 docs an arm is ~11 min and the grid is ~1h. 60 long docs still carry ~12,000
# gold entities, which is ample for a precision/recall trade; it is the DELTA that matters
# here, not a shippable absolute.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=./.venv/bin/python
CFG=${CFG:-tools/train/config/base/eb17-best.yaml}
HFCKPT=${HFCKPT:-whr778/gliner2-eb17-best}
CKPT=${CKPT:-$HOME/ckpt/eb17-best}
LONGSET=${LONGSET:-data/cc_news_long.test.jsonl}
MAXDOCS=${MAXDOCS:-60}
BATCH=${BATCH:-1}
DEST=${DEST:-width_threshold_arms}
OUT=$HOME/wtarms
mkdir -p "$OUT"
source tools/lambda/_publish.sh
trap 'cp ~/box.log "$OUT/box.log" 2>/dev/null && publish "$DEST" "$OUT/box.log" >/dev/null 2>&1 || true' EXIT

echo "[wt] ===== START $(date -u) ====="
[ -f "$CKPT/model.safetensors" ] || $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$HFCKPT', local_dir='$CKPT')" || exit 2
[ -f "$LONGSET" ] || $PY -c "
from huggingface_hub import hf_hub_download
import shutil
shutil.copy(hf_hub_download('whr778/cc_news_long','cc_news_long.test.jsonl',repo_type='dataset'), '$LONGSET')" || exit 2

$PY -u - "$CFG" "$CKPT" "$LONGSET" "$MAXDOCS" "$BATCH" "$OUT" "$DEST" <<'PYEOF' 2>&1 | tee "$OUT/arms.log"
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, "tools/train")
import yaml
import train as T
from gliner2.training.eval_metrics import evaluate_checkpoint

cfg_path, ckpt, longset, maxdocs, batch, out, dest = sys.argv[1:8]
cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
tc = cfg.get("training") or {}
WIN_WORDS = max(64, int(int(tc.get("max_len") or 4096) / 1.5))
fns = T._category_fns(T.load_labels_cfg(cfg, config_path=cfg_path))

rows = []
for line in Path(longset).open(encoding="utf-8"):
    if line.strip():
        rows.append(T.transform_record(json.loads(line), fns))
    if len(rows) >= int(maxdocs):
        break
print(f"[wt] {len(rows)} long docs; window {WIN_WORDS} words; global_decode ON", flush=True)

WIDE = {"start_top_k": 128, "end_top_k": 128, "candidate_budget": 384,
        "boundary_top_k_alpha": 0.05, "boundary_top_k_max": 512}
ARMS = [("S03_shipped_t03", {}, 0.3),
        ("W03_wide_t03", WIDE, 0.3),
        ("W05_wide_t05", WIDE, 0.5),
        ("W07_wide_t07", WIDE, 0.7),
        ("W09_wide_t09", WIDE, 0.9)]
HEADS = ("entity", "event_trigger", "event_argument", "event_type", "structure")

res = {}
for name, wide, thr in ARMS:
    print(f"[wt] arm {name}: width={'WIDE' if wide else 'SHIPPED'} threshold={thr}", flush=True)
    m = evaluate_checkpoint(ckpt, rows, batch_size=int(batch), threshold=thr,
                            chunk_size=WIN_WORDS, chunk_overlap=0, global_decode=True,
                            boundary_overrides=dict(wide) or None) or {}
    res[name] = m
    f = f"{out}/{name}.json"
    Path(f).write_text(json.dumps(m, indent=2), encoding="utf-8")
    # Publish per arm: an overrun must not take completed arms with the box.
    subprocess.run(["bash", "-c",
                    f'source tools/lambda/_publish.sh && publish "{dest}" "{f}"'],
                   check=False, timeout=600)
    e = m.get("eval_entity_error_COR", 0); fp = m.get("eval_entity_error_FP", 0)
    print(f"[wt]   entity COR {e:.0f} FP {fp:.0f} -> predictions {e+fp:.0f}", flush=True)

print(f"\n[wt] === strict micro F1 (entity P/R shown, it drives the trade) ===")
print(f"  {'arm':18} {'entity P':>9} {'entity R':>9} {'entity F1':>10} "
      + "".join(f"{h[:9]:>10}" for h in HEADS[1:]))
best, best_f1 = None, -1.0
for name, _, _ in ARMS:
    m = res[name]
    p = m.get("eval_entity_strict_micro_precision"); r = m.get("eval_entity_strict_micro_recall")
    f1 = m.get("eval_entity_strict_micro_f1")
    if f1 is None:
        print(f"  {name:18}  (no entity metrics)"); continue
    if f1 > best_f1:
        best, best_f1 = name, f1
    rest = "".join(f"{(m.get(f'eval_{h}_strict_micro_f1') or 0.0):>10.4f}" for h in HEADS[1:])
    print(f"  {name:18} {p:>9.4f} {r:>9.4f} {f1:>10.4f} {rest}")

ref = res.get("S03_shipped_t03", {}).get("eval_entity_strict_micro_f1")
print()
if ref is None:
    print("[wt] *** the SHIPPED reference arm produced no entity metric -- every delta below "
          "is unanchored and this grid decides NOTHING. ***")
    raise SystemExit(3)
print(f"[wt] reference S03 (incumbent) entity F1 = {ref:.4f}")
print(f"[wt] best arm = {best} at {best_f1:.4f}  ({best_f1-ref:+.4f} vs incumbent)")
if best == "S03_shipped_t03":
    print("[wt] VERDICT: no WIDE arm beats the incumbent at ANY threshold tried, so the")
    print("[wt] proposal width does not pay on long documents -- item 1 is a coverage")
    print("[wt] result with no metric behind it, and the cap was protecting precision.")
else:
    print("[wt] VERDICT: WIDE wins once the threshold moves with it. The shipped 0.3 was")
    print("[wt] the problem, not the cap. PICKED ON THIS SET -- re-score before shipping.")
PYEOF
RC=${PIPESTATUS[0]}
publish "$DEST" "$OUT/arms.log" || echo "[wt] *** PUBLISH FAILED ***"
echo "[wt] ===== END $(date -u) rc=$RC ====="
exit $RC
