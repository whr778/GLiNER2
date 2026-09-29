#!/bin/bash
# DOES THE ANCHOR OPTIMUM MOVE ONCE THE PROPOSAL CAP IS LIFTED?
#
# THE PICK UNDER TEST. A validation sweep chose record_anchor_threshold 0.1 and the blind
# test confirmed it: event_argument strict F1 0.1752 -> 0.2331, +0.0579, with the predicted
# delta reproducing. That was measured at the SHIPPED proposal cap, where gold coverage at a
# 4096-token window is 8.1%.
#
# WHY IT MIGHT BE AN ARTEFACT. A starved candidate set forces a LOW threshold to get
# anything through. Lift the cap and the same threshold floods: at WIDE width and the
# shipped span threshold, entity predictions went 6,893 -> 183,031 with precision
# 0.353 -> 0.0128. If the anchor's 0.1 was compensating for starvation the same way, its
# optimum should move UP at WIDE -- and "0.1 is the operating point" would be a statement
# about a capped decoder, not about the model.
#
# TWO SWEEPS, ONE VARIABLE. Same tool, same validation split, same grid, no global_decode
# (matching how the original pick was made). SHIPPED reproduces the published curve and is
# the control that proves the harness still behaves; WIDE answers the question. Compare
# WHERE the optimum sits, not just its height.
set -uo pipefail
cd ~/gliner2
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=$(cat ~/.hf_token)
export GLINER2_STRICT_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=./.venv/bin/python
CFG=${CFG:-tools/train/config/base/eb17-best.yaml}
CKPT=${CKPT:-whr778/gliner2-eb17-best}
MAXREC=${MAXREC:-4000}
BATCH=${BATCH:-4}
DEST=${DEST:-anchor_sweep_wide}
OUT=$HOME/anchorwide
mkdir -p "$OUT"
source tools/lambda/_publish.sh
trap 'cp ~/box.log "$OUT/box.log" 2>/dev/null && publish "$DEST" "$OUT/box.log" >/dev/null 2>&1 || true' EXIT

echo "[aw] ===== START $(date -u) ====="
for arm in shipped wide; do
  extra=""
  [ "$arm" = "wide" ] && extra="--wide"
  echo "[aw] --- anchor sweep, proposal width $arm $(date -u) ---"
  timeout 7200 $PY -u tools/train/sweep_record_anchor_threshold.py \
      --config "$CFG" --checkpoint "$CKPT" \
      --axes record_anchor_threshold --batch-size "$BATCH" \
      --max-records "$MAXREC" $extra \
      --out "$OUT/anchor_$arm.json" 2>&1 | tee "$OUT/anchor_$arm.log" | grep -E "^\[sweep\]|FLAT|best |PICKED"
  echo "[aw] $arm exit ${PIPESTATUS[0]}"
  # Publish per arm: an overrun must not take a completed sweep with the box.
  publish "$DEST" "$OUT/anchor_$arm.json" "$OUT/anchor_$arm.log" || echo "[aw] publish failed for $arm"
done

$PY - "$OUT" <<'PYEOF' 2>&1 | tee "$OUT/compare.txt"
import json, sys
from pathlib import Path
out = sys.argv[1]
def load(p):
    f = Path(out, p)
    return json.loads(f.read_text()) if f.is_file() else None
sh, wd = load("anchor_shipped.json"), load("anchor_wide.json")
if not sh or not wd:
    print(f"[aw] *** MISSING ARM: shipped={bool(sh)} wide={bool(wd)} -- no comparison, "
          "and this job decided NOTHING. ***")
    raise SystemExit(2)
KEY = "event_argument_strict"
def curve(rows):
    return {r["value"]: r.get(KEY) for r in rows if r.get("axis") == "record_anchor_threshold"}
cs, cw = curve(sh), curve(wd)
vals = sorted(set(cs) | set(cw), key=lambda v: -float(v))
print(f"  {'anchor':>8} {'SHIPPED':>10} {'WIDE':>10} {'delta':>10}")
for v in vals:
    a, b = cs.get(v), cw.get(v)
    d = (b - a) if (a is not None and b is not None) else None
    print(f"  {v:>8} {a if a is None else f'{a:>10.4f}'} "
          f"{b if b is None else f'{b:>10.4f}'} "
          f"{'' if d is None else f'{d:>+10.4f}'}")
def best(c):
    live = {k: v for k, v in c.items() if v is not None}
    return max(live, key=live.get) if live else None
bs, bw = best(cs), best(cw)
print()
print(f"[aw] SHIPPED optimum at anchor {bs} = {cs.get(bs)}")
print(f"[aw] WIDE    optimum at anchor {bw} = {cw.get(bw)}")
if bs is None or bw is None:
    print("[aw] *** an arm has no live point -- UNMEASURED ***"); raise SystemExit(3)
if float(bw) > float(bs):
    print("[aw] VERDICT: the optimum MOVED UP at WIDE, so the low anchor was compensating")
    print("[aw] for the cap. '0.1 is the operating point' describes a CAPPED decoder.")
elif float(bw) == float(bs):
    print("[aw] VERDICT: the optimum did NOT move. The anchor pick is independent of the")
    print("[aw] cap and the +0.0579 stands on its own terms.")
else:
    print("[aw] VERDICT: the optimum moved DOWN at WIDE -- unexpected; do not rationalise")
    print("[aw] it here, measure why before acting on it.")
PYEOF
publish "$DEST" "$OUT/compare.txt" || echo "[aw] *** PUBLISH FAILED ***"
echo "[aw] ===== END $(date -u) ====="
