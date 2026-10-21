"""Follow each gold event through EVENT INFERENCE, stage by stage, for one or more checkpoints: where do events vanish?

eb22c's events collapse right after the 5-epoch schedule's LR peak (event_type 0.925 -> 0.000 between steps 4k and 5k,
2026-10-20) while every training loss stays smooth. On the records path (event_records) the count and null heads gate
nothing (engine `_decode_records`): a gold event survives or dies at
  REACH   is the gold trigger span among the trigger query's proposed candidates (top-k, no threshold)?
  SCORE   its raw pair logit, its rank among the candidates, and the best competing candidate
  GATE    sigmoid(pair_logit / record_temperature) >= threshold (the anchor gate in records.decode_group)
  EMIT    does the final batch_extract_long output carry an event of that type with that trigger?
Hooks `_forward_batch` (candidates, pair logits) around the real `batch_extract_long`, so it measures what inference does.
Single-chunk documents only (chunk start 0). String gold: every occurrence of the trigger is tried, the best kept.

    uv run tools/train/probe_event_collapse.py --config tools/train/config/base/eb22c-ddp-trace.yaml \\
        --checkpoints whr778/gliner2-eb22c-collapse-trace:checkpoint-4000 whr778/gliner2-eb22c-collapse-trace:checkpoint-5000 \\
        --corpora cmnee duee --docs 40 --out collapse.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402
from gliner2 import AutoExtractor  # noqa: E402
from gliner2.processing.targets import char_span_to_word_boundaries  # noqa: E402
from gliner2.training.eval_metrics import _schema_from_gold  # noqa: E402


def load_checkpoint(ref: str):
    """A local directory, or `repo:subfolder` on the Hub (downloaded once)."""
    if ":" in ref and not Path(ref).exists():
        from huggingface_hub import snapshot_download
        repo, sub = ref.split(":", 1)
        root = snapshot_download(repo, allow_patterns=[f"{sub}/*"],
                                 local_dir=f"/Volumes/Development/tmp/autopsy_ckpts/{repo.replace('/', '__')}")
        ref = str(Path(root) / sub)
    return AutoExtractor.from_pretrained(ref, map_location="cpu").eval()


def occurrences(text: str, needle: str):
    i = text.find(needle)
    while i >= 0:
        yield i, i + len(needle)
        i = text.find(needle, i + 1)


def probe_doc(model, rec: dict, threshold: float) -> list[dict]:
    """One row per gold event: reach, gold logit and rank, best competitor, gate, emitted."""
    cap = {}
    orig = model._forward_batch

    def hook(batch, *a, **k):
        fwd = orig(batch, *a, **k)
        cap.setdefault("calls", []).append((batch, fwd))
        return fwd

    model._forward_batch = hook
    try:
        schema = _schema_from_gold(rec["output"])
        with torch.no_grad():
            pred = model.batch_extract_long([rec["input"]], [schema], threshold=threshold, chunk_size=4096,
                                            chunk_overlap=0, global_decode=True)[0]
    finally:
        model._forward_batch = orig
    calls = cap.get("calls", [])
    if len(calls) != 1 or len(calls[0][0].record_specs) != 1:
        return []                                              # multi-chunk document: skipped (chunk offsets)
    batch, fwd = calls[0]
    cand = fwd["out"].candidates
    offset = int(fwd["core"]["word_offsets"][0])
    temp = float(model.boundary_settings.record_temperature)
    starts, ends = batch.start_mappings[0], batch.end_mappings[0]
    anchor = {s.task_name: s.anchor_query_id for s in batch.record_specs[0].values()}
    emitted = (pred.get("event_extraction") or {})
    rows = []
    for ev in rec["output"]["events"]:
        etype, trig = ev.get("event_type"), (ev.get("triggers") or [None])[0]
        trig = trig["text"] if isinstance(trig, dict) else trig
        if etype not in anchor or not trig:
            continue
        qid = anchor[etype]
        valid = cand.valid_mask[0, qid]
        logits = cand.pair_logits[0, qid].float()
        idx = cand.indices[0, qid]
        vlogits = logits[valid]
        best, best_c = None, None
        for cs, ce in occurrences(rec["input"], trig):
            try:
                ws, we = char_span_to_word_boundaries(cs, ce, starts, ends)
            except ValueError:
                continue
            hit = (valid & (idx[:, 0] == ws + offset) & (idx[:, 1] == we + offset)).nonzero().flatten()
            if hit.numel():
                lg = float(logits[hit[0]])
                if best is None or lg > best:
                    best, best_c = lg, int(hit[0])
        top = float(vlogits.max()) if vlogits.numel() else float("nan")
        n_gate = int((torch.sigmoid(vlogits / temp) >= threshold).sum())
        out_trigs = {(t if isinstance(t, str) else (t or {}).get("text"))
                     for inst in emitted.get(etype) or [] for t in (inst.get("triggers") or [])}
        rows.append({
            "type": etype, "trigger": trig, "reached": best is not None, "query_id": qid, "gold_candidate": best_c,
            "gold_logit": best, "gold_rank": (int((vlogits > best).sum()) if best is not None else None),
            "gold_gate_prob": (1 / (1 + math.exp(-best / temp)) if best is not None else None),
            "top_logit": top, "n_candidates": int(valid.sum()), "n_above_gate": n_gate,
            "emitted_type": bool(emitted.get(etype)), "emitted_trigger": trig in out_trigs,
        })
    return rows


def summarize(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    reached = [r for r in rows if r["reached"]]
    med = lambda xs: round(statistics.median(xs), 3) if xs else None
    s = {"checkpoint": name, "gold_events": n,
         "reach": round(len(reached) / n, 3) if n else None,
         "gold_rank_median": med([r["gold_rank"] for r in reached]),
         "gold_logit_median": med([r["gold_logit"] for r in reached]),
         "top_logit_median": med([r["top_logit"] for r in rows if not math.isnan(r["top_logit"])]),
         "gold_above_gate": round(sum(1 for r in reached if r["gold_gate_prob"] >= THRESHOLD) / n, 3) if n else None,
         "candidates_above_gate_median": med([r["n_above_gate"] for r in rows]),
         "emitted_type": round(sum(r["emitted_type"] for r in rows) / n, 3) if n else None,
         "emitted_trigger": round(sum(r["emitted_trigger"] for r in rows) / n, 3) if n else None}
    print(f"[probe] {name}: " + " | ".join(f"{k} {v}" for k, v in s.items() if k != "checkpoint"), flush=True)
    return s


THRESHOLD = 0.3


def main(argv=None) -> None:
    global THRESHOLD
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="the training config (label map, event_files val)")
    ap.add_argument("--checkpoints", nargs="+", required=True, help="local dirs or repo:subfolder")
    ap.add_argument("--corpora", nargs="+", default=["cmnee"], help="event_files names whose val split is probed")
    ap.add_argument("--docs", type=int, default=40, help="first N val records with gold events, per corpus")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    THRESHOLD = args.threshold
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    fns, passthrough = T._category_fns(T.load_labels_cfg(cfg, args.config)), T.labels_passthrough(cfg)
    docs = {}
    for c in args.corpora:
        path = cfg["data"]["event_files"][c]["val"]
        T._fetch_if_missing(path)
        docs[c] = [r for r in T.read_transformed([path], fns, passthrough) if r["output"].get("events")][:args.docs]
    result = {"args": vars(args), "checkpoints": {}}
    for ref in args.checkpoints:
        model = load_checkpoint(ref)
        per = {}
        for c, recs in docs.items():
            rows = [row for r in recs for row in probe_doc(model, r, args.threshold)]
            per[c] = {"summary": summarize(f"{ref} [{c}]", rows), "rows": rows}
        result["checkpoints"][ref] = per
        Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
        del model
    print(f"[probe] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
