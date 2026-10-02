"""GPU stages of config derivation: reachability, zero-shot baseline, operating points.

All three run on the VALIDATION split, with the corpus labels rewritten through the base's
own `label_map` so the base is asked in the spellings it trained on. Every curve is kept, not
only the pick: an operating point measured on a base is a STARTING point -- calibration moves
as a model trains, so the fine-tuned model must be re-swept on validation.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/train"))
import train as T  # noqa: E402

SPAN_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
RECORD_GRID = (0.5, 0.3, 0.2, 0.1, 0.05)
HEADS = ("entity", "relation", "event_type", "event_trigger", "event_argument",
         "classification", "structure")
RECORD_HEADS = ("event_type", "event_trigger", "event_argument", "structure")


def val_records(base: str, label_map: Dict, max_records: int) -> List[Dict]:
    """Validation records, labels rewritten through the base's label_map."""
    fns = T._category_fns(label_map or {})
    out = []
    for line in open(f"{base}.val.jsonl", encoding="utf-8"):
        if line.strip():
            out.append(T.transform_record(json.loads(line), fns))
        if max_records and len(out) >= max_records:
            break
    return out


def reachability(model: str, base: str, label_map: Dict, event_records: bool, cap: int,
                 window: int, ks: tuple, out_dir: Path, max_records: int, device: str) -> Dict[int, Dict]:
    """Share of gold in the candidate set before injection, per start/end_top_k."""
    cfg_path = out_dir / "probe_config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "labels": label_map or {},
        "model": {"boundary_head": {"event_records": event_records, "max_gold_per_query": cap}},
    }, allow_unicode=True), encoding="utf-8")
    result = {}
    for k in ks:
        out = out_dir / f"reach_k{k}.json"
        subprocess.run([sys.executable, str(ROOT / "tools/train/probe_candidate_coverage.py"),
                        "--config", str(cfg_path), "--checkpoint", model,
                        "--data", f"{base}.val.jsonl", "--windows", f"{window // 2},{window}",
                        "--max-records", str(max_records), "--start-top-k", str(k),
                        "--end-top-k", str(k), "--max-gold", str(cap), "--device", device, "--out", str(out)],
                       check=True)
        rows = json.loads(out.read_text(encoding="utf-8"))
        result[k] = next(r for r in rows if r["window"] == window)
    return result


def at_edge(value: float, grid: tuple) -> bool:
    """True when a pick sits on the grid boundary: the optimum may lie beyond the grid."""
    return value in (min(grid), max(grid))


def head_scores(m: Dict[str, Any]) -> Dict[str, float]:
    return {h: round(m[f"eval_{h}_strict_micro_f1"], 4) for h in HEADS
            if f"eval_{h}_strict_micro_f1" in m}


def operating_points(model, records: List[Dict], window: int, global_decode: bool,
                     default_threshold: float, records_present: bool, batch_size: int) -> Dict:
    """Baseline at the base's own threshold, the span sweep, and the record-gate sweep."""
    from gliner2.training.eval_metrics import apply_boundary_overrides, compute_metrics, sweep_thresholds
    from gliner2.training.trainer import ExtractorDataset

    ds = ExtractorDataset(records, shuffle=False, validate=False)
    kw = dict(batch_size=batch_size, chunk_size=window, chunk_overlap=0, global_decode=global_decode)
    baseline = compute_metrics(model, ds, threshold=default_threshold, report=False, **kw)
    best_t, _, curve = sweep_thresholds(model, ds, thresholds=SPAN_GRID, **kw)
    out = {"baseline": {"threshold": default_threshold, "strict_f1": head_scores(baseline)},
           "span_sweep": {"picked": best_t, "at_grid_edge": at_edge(best_t, SPAN_GRID),
                          "rule": "support-weighted strict micro-F1 (sweep_thresholds)",
                          "curve": {str(t): head_scores(m) for t, m in curve.items()}}}
    if records_present:
        rows = {}
        for v in RECORD_GRID:
            apply_boundary_overrides(model, {"record_anchor_threshold": v,
                                             "record_anchor_threshold_wins": True,
                                             "record_anchor_proposal_threshold": v})
            rows[str(v)] = head_scores(compute_metrics(model, ds, threshold=best_t, report=False, **kw))
        # Score the heads the record gate actually moves. A grid that is all-zero on them is
        # NO SIGNAL, and max() over ties would silently return the first row -- 0.5, the worst
        # point -- as if it had been chosen (caught tracing eb17 on CASIE, 2026-10-02).
        score = lambda v: sum(rows[v].get(h, 0) for h in RECORD_HEADS)
        pick = max(rows, key=score) if any(score(v) > 0 for v in rows) else None
        out["record_sweep"] = {"span_threshold": best_t, "picked": float(pick) if pick else None,
                               "at_grid_edge": at_edge(float(pick), RECORD_GRID) if pick else None,
                               "rule": f"max sum of strict F1 over {RECORD_HEADS}"
                                       + ("" if pick else " -- NO SIGNAL: all zero, nothing picked"),
                               "curve": rows}
    return out
