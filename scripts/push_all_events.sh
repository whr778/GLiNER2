#!/usr/bin/env bash
#
# Push each trained event model's checkpoint to the Hugging Face Hub under
# whr778/<config-name> as a PUBLIC repo.
#
# Unlike tools/train/push_to_hub.py (which re-serializes via save_pretrained and
# so drops the model card + metrics), this uploads the checkpoint folder directly
# -- weights, config, tokenizer, encoder_config/, and the *_metrics.json files --
# and renders the training-generated MODEL_CARD.md as the repo README.md so it
# shows on the model page.
#
# Run this on whatever machine holds the checkpoints (the AWS training box):
#   export HF_TOKEN=hf_...                              # write-scoped token
#   bash scripts/push_all_events.sh                     # all default configs
#   bash scripts/push_all_events.sh gliner2-base-v1-casie   # only these
#   DRY_RUN=1 bash scripts/push_all_events.sh           # preview, no upload
#
# Auth: HF_TOKEN env var, or `uv run huggingface-cli login` with a write token.
# A config with no checkpoint on disk (e.g. maven has no val split -> no best/)
# is skipped with a message, not treated as a failure.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

NAMESPACE="whr778"
OUT_ROOT="out/fastino"

# Default: the per-dataset event models (matches the trimmed training batch).
CONFIGS=(
  gliner2-base-v1-casie
  gliner2-base-v1-docee
  gliner2-base-v1-rams
  gliner2-base-v1-maven
  gliner2-multi-v1-cmnee
  gliner2-large-v1-casie
  gliner2-large-v1-docee
  gliner2-large-v1-rams
  gliner2-large-v1-maven
  gliner2-large-v1-wikievents
)

# If config names are passed as arguments, push only those instead.
if [ "$#" -gt 0 ]; then
  CONFIGS=("$@")
fi

DRY_RUN="${DRY_RUN:-}"

# Fail fast if there is no auth (unless just previewing).
if [ -z "$DRY_RUN" ] && [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HOME/.cache/huggingface/token" ]; then
  echo "ERROR: no HuggingFace auth. Set HF_TOKEN (write scope) or run" >&2
  echo "       'uv run huggingface-cli login'. Aborting." >&2
  exit 1
fi

# Pick a config's checkpoint dir: prefer best/, then last/, final/, or the root.
pick_ckpt() {
  local base="$1" sub d
  for sub in best last final ""; do
    d="$base${sub:+/$sub}"
    if [ -f "$d/model.safetensors" ]; then echo "$d"; return 0; fi
  done
  return 1
}

OK=(); FAIL=(); SKIP=()

for name in "${CONFIGS[@]}"; do
  base="$OUT_ROOT/$name"
  repo="$NAMESPACE/$name"
  if ! ckpt="$(pick_ckpt "$base")"; then
    echo "[skip] $name: no checkpoint under $base"
    SKIP+=("$name")
    continue
  fi
  size="$(du -sh "$ckpt" 2>/dev/null | cut -f1)"

  echo ""
  echo "=== $name -> https://huggingface.co/$repo  (from $ckpt, $size) ==="

  if [ -n "$DRY_RUN" ]; then
    echo "[dry-run] would push $ckpt to $repo (public)"
    OK+=("$name (dry-run)")
    continue
  fi

  if uv run python - "$repo" "$ckpt" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import HfApi

repo_id, folder = sys.argv[1], Path(sys.argv[2])
api = HfApi()
api.create_repo(repo_id=repo_id, private=False, exist_ok=True)
card = folder / "MODEL_CARD.md"
api.upload_folder(
    folder_path=str(folder),
    repo_id=repo_id,
    ignore_patterns=["MODEL_CARD.md"] if card.exists() else None,
    commit_message="Upload GLiNER2 event checkpoint",
)
if card.exists():
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add model card as README",
    )
print(f"pushed {repo_id}")
PY
  then
    echo "[ok] $repo"
    OK+=("$name")
  else
    echo "[fail] $repo (see output above)"
    FAIL+=("$name")
  fi
done

echo ""
echo "============================================================"
echo "Summary: ${#OK[@]} ok, ${#SKIP[@]} skipped, ${#FAIL[@]} failed"
[ "${#OK[@]}"   -gt 0 ] && printf '  ok:   %s\n' "${OK[@]}"
[ "${#SKIP[@]}" -gt 0 ] && printf '  skip: %s\n' "${SKIP[@]}"
[ "${#FAIL[@]}" -gt 0 ] && printf '  fail: %s\n' "${FAIL[@]}"
echo "============================================================"

# Non-zero exit if anything failed, so callers can detect it.
[ "${#FAIL[@]}" -eq 0 ]
