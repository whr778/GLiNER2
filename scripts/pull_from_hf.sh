#!/usr/bin/env bash
#
# Download the pushed GLiNER2 models from the Hugging Face Hub to this laptop,
# into out/fastino/<name>/best/ (auto-discovered by the viewer's model registry).
# Uses Xet high-performance transfer (parallel, CDN-accelerated) -- far faster
# than a single rsync-over-SSH stream over a high-latency / per-flow-throttled
# path (a VPN). The repos are public (whr778/<name>), so no auth is needed.
#
#   bash scripts/pull_from_hf.sh                        # all default models
#   bash scripts/pull_from_hf.sh gliner2-base-v1-casie  # only these
#
# Requires `uv` on PATH; it fetches the `hf` CLI into an ephemeral env.
# Override the namespace with GLINER2_HF_NAMESPACE.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

NAMESPACE="${GLINER2_HF_NAMESPACE:-whr778}"

# The models that trained successfully in the event batch.
CONFIGS=(
  gliner2-base-v1-casie
  gliner2-base-v1-docee
  gliner2-base-v1-rams
  gliner2-multi-v1-cmnee
  gliner2-large-v1-casie
  gliner2-large-v1-docee
  gliner2-large-v1-rams
  gliner2-large-v1-wikievents
)
[ "$#" -gt 0 ] && CONFIGS=("$@")

export HF_XET_HIGH_PERFORMANCE=1
hf_cli() { uv run --no-project --with 'huggingface_hub[hf_xet]' hf "$@"; }

OK=(); FAIL=()
for name in "${CONFIGS[@]}"; do
  dest="out/fastino/$name/best"
  mkdir -p "$dest"
  echo ""
  echo "=== $NAMESPACE/$name -> $dest ==="
  if hf_cli download "$NAMESPACE/$name" --local-dir "$dest"; then
    OK+=("$name")
  else
    FAIL+=("$name")
  fi
done

echo ""
echo "============================================================"
echo "Summary: ${#OK[@]} ok, ${#FAIL[@]} failed"
[ "${#OK[@]}"   -gt 0 ] && printf '  ok:   %s\n' "${OK[@]}"
[ "${#FAIL[@]}" -gt 0 ] && printf '  fail: %s\n' "${FAIL[@]}"
echo "============================================================"

[ "${#FAIL[@]}" -eq 0 ]
