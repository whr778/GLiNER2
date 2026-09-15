#!/bin/bash
# Bring a fresh Lambda GPU box from bare to "can run eval/training", and PROVE each step.
#
# THE cu128 TRAP, which TRAINING.md documents a fix for that does not work. The note says to
# pin `torch==2.11.0 --index-url .../cu128`, but `uv sync` has already installed a wheel that
# also calls itself `2.11.0` -- just `+cu130`. The requirement is satisfied, uv prints
# "Checked 1 package in 3ms", changes nothing, and `torch.cuda.is_available()` is False
# because the driver is 12.8. `--reinstall-package torch` is what actually swaps the wheel.
#
# FA2 IS PRE-FLIGHTED THROUGH THE REAL LOAD PATH, not `get_kernel`. A bare
# `get_kernel("kernels-community/flash-attn2")` now raises -- the API wants `version=` or
# `revision=` -- so it fails on a box where FA2 is perfectly healthy. What matters is what
# the encoder is BUILT with, and the model config reports the REQUEST, not the result.
#
#   CKPT=whr778/gliner2-eb16-rebuild-tr CFG=tools/train/config/base/eb16-rebuild-tr.yaml \
#     bash tools/lambda/bootstrap_box.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
CFG=${CFG:-}
CKPT=${CKPT:-}
fail() { echo "[bs] *** FATAL: $* ***" >&2; exit 1; }

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd ~/gliner2 || fail "no ~/gliner2"

echo "[bs] venv + sync $(date -u)"
uv venv --python 3.12 || fail "uv venv"
uv sync || fail "uv sync"

echo "[bs] torch cu128 $(date -u)"
uv pip install --reinstall-package torch "torch==2.11.0" \
   --index-url https://download.pytorch.org/whl/cu128 >/dev/null 2>&1 || fail "torch install"

# PROVE it, rather than trusting the installer's exit code.
./.venv/bin/python - <<'PY' || fail "CUDA is not available -- the cu128 swap did not take"
import sys, torch
print(f"[bs] torch {torch.__version__} | cuda {torch.cuda.is_available()} | built for {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit(1)
print("[bs] device:", torch.cuda.get_device_name(0))
PY

export HF_TOKEN=$(cat ~/.hf_token)
if [ -n "$CFG" ]; then
  echo "[bs] data for $CFG $(date -u)"
  ./.venv/bin/python tools/data/restore_from_hf.py --config "$CFG" 2>&1 | tail -3 \
    || fail "data restore"
fi

if [ -n "$CKPT" ]; then
  echo "[bs] checkpoint $CKPT $(date -u)"
  DEST=$HOME/ckpt/$(basename "$CKPT")
  ./.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('$CKPT', local_dir='$DEST')" >/dev/null 2>&1 || fail "checkpoint download"
  echo "[bs] FA2 pre-flight through the REAL load path $(date -u)"
  GLINER2_STRICT_ATTN=1 ./.venv/bin/python - <<PY || fail "attention pre-flight"
import sys
from gliner2 import AutoExtractor
m = AutoExtractor.from_pretrained("$DEST")
got = m.encoder.config._attn_implementation
print(f"[bs] requested {m.config.attn_implementation} -> loaded {got}")
# Anything that is not an FA2 spelling means every later run is on a different backend than
# the numbers it will be compared against.
if "flash" not in got:
    sys.exit(1)
PY
fi
echo "[bs] READY $(date -u)"
