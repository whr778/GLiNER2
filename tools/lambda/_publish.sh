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
#
# IT IS ALSO NON-ZERO WHEN A NAMED FILE DOES NOT EXIST. Publishing is where a dead job is
# noticed, so `nothing to send` must never be a success: the threshold sweep asked to
# publish a test result its eval had never written, got a clean return, and terminated.
LOGREPO=${LOGREPO:-whr778/gliner2-run-logs}
PUBPY=${PUBPY:-./.venv/bin/python}

_publish_once() {
  local dest=$1; shift
  "$PUBPY" - "$LOGREPO" "$dest" "$@" <<'PY'
import os, sys
from huggingface_hub import HfApi
repo, dest, files = sys.argv[1], sys.argv[2], sys.argv[3:]
api = HfApi(); api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
sent, absent = [], []
for p in files:
    if os.path.exists(p) and os.path.getsize(p) > 0:
        name = f"{dest}/{os.path.basename(p)}"
        api.upload_file(path_or_fileobj=p, path_in_repo=name, repo_id=repo,
                        repo_type="dataset")
        sent.append(name)
    else:
        absent.append(p)
# Save what DOES exist first -- a partial result beats none -- then verify, then complain.
if sent:
    on_hub = set(api.list_repo_files(repo, repo_type="dataset"))
    missing = [f for f in sent if f not in on_hub]
    print("[publish] sent:", ", ".join(sent))
    if missing:
        raise SystemExit(f"*** NOT SAVED *** {missing}")
if absent:
    # A file the CALLER NAMED but the job never wrote means the job failed. Returning 0
    # here made a dead test pass look like a successful publish on 2026-09-15: the sweep
    # terminated on its normal path having saved five validation points and no test point.
    print(f"[publish] *** NEVER WRITTEN *** {absent}")
    raise SystemExit(3)
raise SystemExit(0)
PY
}

publish() {
  local dest=$1; shift
  local n
  local rc
  for n in 1 2 3 4 5 6; do
    _publish_once "$dest" "$@"; rc=$?
    if [ "$rc" -eq 0 ]; then
      [ "$n" -gt 1 ] && echo "[publish] succeeded on attempt $n"
      return 0
    fi
    # 3 = the job never wrote the artefact. Backing off six times cannot conjure it, and
    # on an A100 that is ~30 minutes of billed sleep before the same answer.
    if [ "$rc" -eq 3 ]; then
      echo "[publish] *** NOT RETRYING -- the job never produced it ***"
      return 1
    fi
    echo "[publish] FAILED (attempt $n/6)"
    [ "$n" -lt 6 ] && sleep $((n * 120))
  done
  echo "[publish] *** FAILED AFTER 6 ATTEMPTS: $dest $* ***"
  return 1
}
