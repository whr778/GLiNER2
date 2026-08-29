"""Score an annotation pool with the stage-0 gate, checkpointing as it goes.

Runs ON the Lambda box. The laptop may lose its network at any moment, so this must be
able to finish, publish and be torn down with nobody watching:

  - **Checkpoints every --every documents.** A crash or a spot reclaim at 90% costs the
    last chunk, not the run. Restarting resumes from the checkpoint.
  - **Pushes to a private Hub repo when done**, because the box is about to be destroyed
    and its disk with it.
  - Exits non-zero on any failure, so the caller's watchdog can tell "finished" from
    "died" and still terminate either way.

Length-sorted batching: documents run 466-6,000 characters and every batch pads to its
longest member, so sorting by length is what makes batching a win rather than a loss
(measured on MPS: 5.1 docs/s at batch 1 unsorted, 2.0 at batch 32 unsorted).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# The gate schema is inlined rather than imported from run_pipeline, which pulls in
# datasets/disaster_streams/extract.py and a long chain this job does not need. Copied
# verbatim from run_pipeline.GATE_LABELS_V2 / build_gate_schema.
#
# ONE classification task ONLY. On a BOUNDARY model a second task collapses `relevance`
# to `other` at confidence 1.0 for every input -- and a gate that admits nothing has a
# perfect false-positive rate, so the collapse reads as success. Do not add to this.
GATE_LABELS_V2 = {
    "mass_casualty": (
        "a report of how many people were killed, injured, or are missing in a specific "
        "disaster, accident or attack -- it states or estimates a TOLL for a group of people"
    ),
    "other": (
        "anything else, including: personal messages, greetings, thanks and requests for "
        "help, money, jobs or travel; notes from a translator about the message itself; "
        "politics, elections, government policy, diplomacy and development or NGO reports; "
        "aid logistics and supply inventories, even with very large quantities; business, "
        "finance, science and environment writing; one individual's illness, injury or "
        "death; and metaphorical use of disaster words such as an 'explosion' in an industry"
    ),
}


def build_gate_schema(model):
    return model.create_schema().classification("relevance", GATE_LABELS_V2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="whr778/gliner2-gate2-mmbert-tr")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--every", type=int, default=5000, help="checkpoint interval")
    ap.add_argument("--repo", default="", help="HF repo to publish to (private)")
    a = ap.parse_args()

    import torch
    assert torch.cuda.is_available(), "no CUDA -- refusing to run at CPU speed on a GPU bill"
    print(f"[score] CUDA {torch.version.cuda}, {torch.cuda.get_device_name(0)}", flush=True)

    from gliner2 import AutoExtractor

    rows = [json.loads(l) for l in Path(a.pool).open(encoding="utf-8")]
    print(f"[score] {len(rows)} documents", flush=True)

    out_path = Path(a.out)
    done: dict[str, float] = {}
    if out_path.is_file():
        done = {r["url"]: r["score"] for r in
                (json.loads(l) for l in out_path.open(encoding="utf-8"))}
        print(f"[score] resuming: {len(done)} already scored", flush=True)

    todo = [r for r in rows if r.get("url") not in done]
    todo.sort(key=lambda r: len(r["input"]))   # near-uniform batches, minimal padding
    print(f"[score] {len(todo)} to score", flush=True)

    model = AutoExtractor.from_pretrained(a.model, map_location="cuda")
    schema = build_gate_schema(model)

    start = time.time()
    fh = out_path.open("a", encoding="utf-8")
    for i in range(0, len(todo), a.batch_size):
        chunk = todo[i:i + a.batch_size]
        res = model.batch_extract([c["input"] for c in chunk], schema,
                                  a.batch_size, 0.5, 0, True, True)
        for row, r in zip(chunk, res):
            rel = r.get("relevance")
            label = rel.get("label") if isinstance(rel, dict) else rel
            conf = float(rel.get("confidence", 1.0)) if isinstance(rel, dict) else 1.0
            fh.write(json.dumps({"url": row.get("url"), "outlet": row.get("outlet"),
                                 "score": conf if label == "mass_casualty" else 1.0 - conf},
                                ensure_ascii=False) + "\n")
        if (i + a.batch_size) % a.every < a.batch_size:
            fh.flush()
            os.fsync(fh.fileno())
            n = i + len(chunk)
            rate = n / (time.time() - start)
            print(f"[score] {n}/{len(todo)}  {rate:.1f} doc/s  "
                  f"eta {(len(todo) - n) / rate / 60:.0f} min", flush=True)
    fh.close()
    print(f"[score] DONE in {(time.time() - start) / 60:.1f} min", flush=True)

    if a.repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=str(out_path), path_in_repo=out_path.name,
                        repo_id=a.repo, repo_type="dataset")
        print(f"[score] published to {a.repo}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
