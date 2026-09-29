#!/bin/bash
# DOES THE WINDOW/OVERLAP CHOICE MATTER? Measured where it CAN matter.
#
# Our own corpora cannot answer this: median 406-596 tokens, ~1% over the 4096 window,
# 13 hard cuts across 4,000 test documents. Every setting scores the same there, which is
# silence, not support. `cc_news_long` can answer it: 733 documents, median 9,697 tokens,
# max 42,635, 100% multi-window (mean 3.07, max 11), 36% of gold in the second half and
# 2,556 gold surfaces (4.1%) within ~128 tokens of a boundary.
#
# THREE ARMS, one variable:
#   A  model window, overlap 0      -- the uniform-window policy as currently committed
#   B  model window, overlap 64     -- the library's inference overlap: boundary coverage
#   C  200 words / 50 overlap       -- what the EKF pipeline ships today, the measured band
#
# OVERLAP DOES NOT DOUBLE-COUNT HERE. compute_metrics scores through batch_extract_long,
# whose merge concatenates and DEDUPES, so B and C are not inflated by construction --
# that was checked in code before this run, not assumed.
#
# THE CONTROL IS THE REGULAR TEST SPLIT. It barely splits at any of these settings, so the
# three arms must agree there. If they do not, the effect is not windowing and this run is
# void -- that is the check that can fail.
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
MAXDOCS=${MAXDOCS:-300}
BATCH=${BATCH:-1}
DEST=${DEST:-window_overlap_arms}
OUT=$HOME/winarms
mkdir -p "$OUT"
source tools/lambda/_publish.sh
trap 'cp ~/box.log "$OUT/box.log" 2>/dev/null && publish "$DEST" "$OUT/box.log" >/dev/null 2>&1 || true' EXIT

echo "[win] ===== START $(date -u) ====="
if [ ! -f "$CKPT/model.safetensors" ]; then
  echo "[win] fetching $HFCKPT"
  $PY -c "
from huggingface_hub import snapshot_download
snapshot_download('$HFCKPT', local_dir='$CKPT')" || exit 2
fi
if [ ! -f "$LONGSET" ]; then
  echo "[win] fetching cc_news_long from the Hub"
  $PY -c "
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download('whr778/cc_news_long', 'cc_news_long.test.jsonl', repo_type='dataset')
shutil.copy(p, '$LONGSET')" || { echo '[win] FATAL: no long set'; exit 2; }
fi

$PY - "$CFG" "$CKPT" "$LONGSET" "$MAXDOCS" "$BATCH" "$OUT" <<'PYEOF' 2>&1 | tee "$OUT/arms.log"
import json, sys
from pathlib import Path
sys.path.insert(0, "tools/train")
import yaml
import train as T
from gliner2.training.eval_metrics import evaluate_checkpoint

cfg_path, ckpt, longset, maxdocs, batch, out = sys.argv[1:7]
cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
tc = cfg.get("training") or {}
WIN_TOK = int(tc.get("max_len") or 4096)
WIN_WORDS = max(64, int(WIN_TOK / 1.5))          # extract_long chunks by WORDS
fns = T._category_fns(T.load_labels_cfg(cfg, config_path=cfg_path))

def load(path, n):
    rows = []
    for line in Path(path).open(encoding="utf-8"):
        if line.strip():
            rows.append(T.transform_record(json.loads(line), fns))
        if len(rows) >= n:
            break
    return rows

long_rows = load(longset, int(maxdocs))
print(f"[win] {len(long_rows)} long docs; window {WIN_TOK} tokens = {WIN_WORDS} words", flush=True)

# THE CONVERSION IS THE WEAK POINT, SO MEASURE IT RATHER THAN ARGUE ABOUT IT.
# extract_long chunks by WORDS; the window is configured in TOKENS. At a fixed word
# count the realised token window moves with tokenisation density, and past the MODEL's
# max_len (not the training window) text is silently truncated. Print both so the arms
# are readable, and say plainly how much would truncate.
import json as _json
from transformers import AutoTokenizer as _AT
_tok = _AT.from_pretrained(ckpt)
_model_max = int(_json.loads(Path(ckpt, "config.json").read_text()).get("max_len") or 8192)
_r = []
for _row in long_rows[:80]:
    _t = _row.get("input") or _row.get("text") or ""
    _w = len(_t.split())
    if _w:
        _r.append(len(_tok(_t, add_special_tokens=False)["input_ids"]) / _w)
_r.sort()
if _r:
    _med, _p90, _mx = _r[len(_r)//2], _r[int(len(_r)*.9)], _r[-1]
    _trunc = sum(1 for x in _r if WIN_WORDS * x > _model_max)
    print(f"[win] tokens/word: median {_med:.3f} p90 {_p90:.3f} max {_mx:.3f}")
    print(f"[win] realised window: {WIN_WORDS*_med:.0f} tok at median, {WIN_WORDS*_p90:.0f} at p90 "
          f"(training window {WIN_TOK}, MODEL cap {_model_max})")
    print(f"[win] chunks that would TRUNCATE against the model cap: {_trunc}/{len(_r)} "
          f"({_trunc/len(_r)*100:.1f}%)")
    print("[win] A vs B share this window exactly and differ ONLY in overlap, so that "
          "comparison is unaffected by the conversion. C differs in both.", flush=True)

ARMS = [("A_win_ov0", WIN_WORDS, 0), ("B_win_ov64", WIN_WORDS, 64), ("C_200_ov50", 200, 50)]
HEADS = ("entity", "event_trigger", "event_argument", "event_type", "structure", "relation")

def run(rows, label):
    res = {}
    for name, cs, ov in ARMS:
        print(f"[win] {label} arm {name}: chunk_size={cs} overlap={ov}", flush=True)
        m = evaluate_checkpoint(ckpt, rows, batch_size=int(batch), threshold=0.3,
                                chunk_size=cs, chunk_overlap=ov) or {}
        res[name] = m
        Path(f"{out}/{label}_{name}.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    return res

long_res = run(long_rows, "long")

print(f"\n[win] === cc_news_long: strict micro F1 by arm ===")
print(f"  {'head':18} " + "".join(f"{n:>14}" for n, _, _ in ARMS))
moved = 0
for h in HEADS:
    key = f"eval_{h}_strict_micro_f1"
    vals = [long_res[n].get(key) for n, _, _ in ARMS]
    if not any(isinstance(v, (int, float)) for v in vals):
        continue
    spread = max(v for v in vals if v is not None) - min(v for v in vals if v is not None)
    if spread > 1e-9:
        moved += 1
    print(f"  {h:18} " + "".join(f"{(v if v is not None else float('nan')):>14.4f}" for v in vals)
          + f"   spread {spread:+.4f}")
print()
if moved == 0:
    print("[win] *** EVERY ARM IDENTICAL on a corpus that is 100% multi-window. That is not")
    print("[win] *** 'windowing does not matter' -- it means the arms did not reach the")
    print("[win] *** decoder. Check chunk_size actually differed. MEASURED NOTHING. ***")
    raise SystemExit(3)
print(f"[win] {moved} of {len(HEADS)} heads moved -- the arms reached the decoder.")

# OneIE criteria, reported BESIDE the heads and never merged into them. These are the
# numbers comparable with published OneIE-style work; the corpus heads are ours.
ONEIE = (("trigi", "Trig-I"), ("trigc", "Trig-C"), ("argi", "Arg-I"), ("argc", "Arg-C"))
print(f"\n[win] === OneIE criteria (external), micro F1 by arm ===")
print(f"  {'metric':18} " + "".join(f"{n:>14}" for n, _, _ in ARMS))
for key, label in ONEIE:
    k = f"eval_{key}_external_micro_f1"
    vals = [long_res[n].get(k) for n, _, _ in ARMS]
    if not any(isinstance(v, (int, float)) for v in vals):
        print(f"  {label:18} " + "  not emitted on this corpus")
        continue
    spread = max(v for v in vals if v is not None) - min(v for v in vals if v is not None)
    print(f"  {label:18} " + "".join(f"{(v if v is not None else float('nan')):>14.4f}" for v in vals)
          + f"   spread {spread:+.4f}")

# THE BOUND IS A GATE, AND IT CAN FAIL. Arg-C is defined to sit between our strict and
# relaxed event_argument: event_argument_strict <= argc_external <= event_argument_relaxed.
# Outside that, the two accountings disagree and the OneIE column is not trustworthy.
print()
for name, _, _ in ARMS:
    m = long_res[name]
    lo = m.get("eval_event_argument_strict_micro_f1")
    hi = m.get("eval_event_argument_relaxed_micro_f1")
    ac = m.get("eval_argc_external_micro_f1")
    if None in (lo, hi, ac):
        print(f"[win] {name}: bound UNCHECKED (a term is missing)")
        continue
    ok = lo - 1e-9 <= ac <= hi + 1e-9
    print(f"[win] {name}: {lo:.4f} <= Arg-C {ac:.4f} <= {hi:.4f}  "
          f"{'OK' if ok else '*** OUT OF BOUND -- the accountings disagree ***'}")
PYEOF
RC=${PIPESTATUS[0]}
publish "$DEST" "$OUT/arms.log" $OUT/long_*.json || echo "[win] *** PUBLISH FAILED ***"
echo "[win] ===== END $(date -u) rc=$RC ====="
exit $RC
