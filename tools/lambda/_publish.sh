# Sourceable publish helper. Both the A/B runner and the throughput smoke use this.
#
# WHY IT IS A SHARED FILE. This retry-and-verify block was written for cardinality_ab.sh,
# then NOT written for throughput_smoke.sh -- which self-terminated on its normal path and
# took its only copy of the result with it (2026-09-15, ~$0.30 for nothing). The lesson had
# been applied to one file instead of to the practice. One definition, called by both.
#
#   source tools/lambda/_publish.sh
#   publish <dest-folder> <file>...        # retries 6x over ~30 min, verifies on the Hub
#
# Returns non-zero if the artefacts are not visible on the Hub afterwards, so a caller can
# hold the box rather than terminating into a loss. A clean return is NOT proof: upload_folder
# has returned successfully having written nothing, and it cost ~15h of A100 once.
LOGREPO=${LOGREPO:-whr778/gliner2-run-logs}
PUBPY=${PUBPY:-./.venv/bin/python}

_publish_once() {
  local dest=$1; shift
  "$PUBPY" - "$LOGREPO" "$dest" "$@" <<'PY'
import os, sys
from huggingface_hub import HfApi
repo, dest, files = sys.argv[1], sys.argv[2], sys.argv[3:]
api = HfApi(); api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
sent = []
for p in files:
    if os.path.exists(p) and os.path.getsize(p) > 0:
        name = f"{dest}/{os.path.basename(p)}"
        api.upload_file(path_or_fileobj=p, path_in_repo=name, repo_id=repo,
                        repo_type="dataset")
        sent.append(name)
if not sent:
    print("[publish] nothing to send"); raise SystemExit(0)
on_hub = set(api.list_repo_files(repo, repo_type="dataset"))
missing = [f for f in sent if f not in on_hub]
print("[publish] sent:", ", ".join(sent))
raise SystemExit(f"*** NOT SAVED *** {missing}" if missing else 0)
PY
}

publish() {
  local dest=$1; shift
  local n
  for n in 1 2 3 4 5 6; do
    if _publish_once "$dest" "$@"; then
      [ "$n" -gt 1 ] && echo "[publish] succeeded on attempt $n"
      return 0
    fi
    echo "[publish] FAILED (attempt $n/6)"
    [ "$n" -lt 6 ] && sleep $((n * 120))
  done
  echo "[publish] *** FAILED AFTER 6 ATTEMPTS: $dest $* ***"
  return 1
}
