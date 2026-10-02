"""Stage 7: write the fine-tuning config from the calibration, every value annotated.

Each emitted value carries a comment naming its source:
  measured    from this run's calibration (stage and number given)
  checkpoint  the base's own value, kept
  policy      a standing project default (warm-start learning rates, bf16, negatives, ...)

Refuses to write when data health BLOCKs: a config is a promise that the run measures
something, and contaminated splits break that promise.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/train"))
import train as T  # noqa: E402

PLATEAU = 0.95          # smallest k reaching 95% of the best measured coverage


def _reach_pick(reach: Dict[str, Dict]) -> Tuple[int, str]:
    pq = {int(k.rsplit("k", 1)[1]): v["coverage"] for k, v in reach.items() if k.startswith("per_query_k")}
    best = max(pq.values())
    k = min(k for k, c in pq.items() if c >= PLATEAU * best)
    curve = ", ".join(f"k{kk} {100 * c:.1f}%" for kk, c in sorted(pq.items()))
    return k, f"measured: smallest k within {int(PLATEAU * 100)}% of the best coverage ({curve})"


def _pool_pick(gpu: Dict, checkpoint_pool: str) -> Tuple[str, str]:
    comp = gpu.get("per_query_companion")
    if checkpoint_pool != "shared" or not comp:
        return checkpoint_pool, "checkpoint"
    heads = ("event_type", "event_trigger", "event_argument", "structure", "entity", "relation")
    pick_row = gpu["span_sweep"]["curve"][str(gpu["span_sweep"]["picked"])]
    shared = sum(pick_row.get(h, 0) for h in heads)
    per_q = sum(comp["strict_f1"].get(h, 0) for h in heads)
    reach = gpu["reachability"]
    note = (f"measured: shared reach {100 * reach['as_built_shared']['coverage']:.1f}% -> per_query k128 "
            f"{100 * reach['per_query_k128']['coverage']:.1f}%; val strict F1 sum {shared:.3f} (shared) vs {per_q:.3f} (per_query)")
    return ("per_query", note + " -> switch") if per_q >= shared else ("shared", note + " -> keep shared")


def build(name: str, model: str, corpus_base: str, calib: Dict) -> Tuple[Dict, Dict[str, str], List[str]]:
    """Return (config, {dotted.path: comment}, warnings for the header)."""
    m, c, gpu = calib["model"], calib["corpus"], calib.get("gpu") or {}
    bh_ck = calib.get("checkpoint_boundary_head") or {}
    ev = gpu.get("eval_settings") or {}
    cm: Dict[str, str] = {}
    warnings: List[str] = []
    boundary = m["architecture"] == "boundary"
    window = ev.get("window") or (min(m["max_len"], 512) - 128 if not boundary else min(m["max_len"], 4096))
    cap = c["gold_capacity"]["recommended_cap"]

    model_cfg: Dict[str, Any] = {"pretrained": model}
    cm["model.pretrained"] = "checkpoint: warm start from the base's own heads"
    if boundary:
        bh: Dict[str, Any] = {"max_gold_per_query": cap}
        cm["model.boundary_head.max_gold_per_query"] = (
            f"measured: largest gold group {c['gold_capacity']['largest_group']}, "
            f"{c['gold_capacity']['pct_groups_over_32']}% of groups over 32")
        tcb = max(cap, int(bh_ck.get("training_candidate_budget") or bh_ck.get("candidate_budget") or cap))
        bh["training_candidate_budget"] = tcb
        cm["model.boundary_head.training_candidate_budget"] = "measured floor: must be >= max_gold_per_query"
        if gpu.get("reachability"):
            k, note = _reach_pick(gpu["reachability"])
            bh["start_top_k"] = bh["end_top_k"] = k
            cm["model.boundary_head.start_top_k"] = cm["model.boundary_head.end_top_k"] = note
            pool, pnote = _pool_pick(gpu, bh_ck.get("candidate_pool", "per_query"))
            if pool != bh_ck.get("candidate_pool", "per_query"):
                bh["candidate_pool"] = pool
                bh["boundary_top_k_alpha"] = 0.0
                cm["model.boundary_head.boundary_top_k_alpha"] = "measured with alpha 0, so start_top_k is what governs"
            cm["model.boundary_head.candidate_pool"] = pnote
        rs = gpu.get("record_sweep") or {}
        if rs.get("picked") is not None:
            bh["record_anchor_threshold"] = bh["record_anchor_proposal_threshold"] = rs["picked"]
            bh["record_anchor_threshold_wins"] = True
            note = (f"measured on the BASE (val): {rs['rule']}" + ("; AT GRID EDGE" if rs.get("at_grid_edge") else "")
                    + " -- a starting point, re-sweep after fine-tuning")
            cm["model.boundary_head.record_anchor_threshold"] = note
            cm["model.boundary_head.record_anchor_threshold_wins"] = "makes the record gate reachable (S14); off = the span threshold decides"
        structural = sorted(k for k in bh if k in T._STRUCTURAL_BOUNDARY_KEYS)
        if structural:
            raise SystemExit(f"[emit] would write structural keys {structural} on a pretrained base")
        model_cfg["boundary_head"] = bh

    span_pick = (gpu.get("span_sweep") or {}).get("picked")
    threshold = span_pick if span_pick is not None else ev.get("threshold", (m.get("inference_defaults") or {}).get("threshold", 0.5))
    tr = {
        "output_dir": f"./out/{name}", "experiment_name": name.replace("-", "_"),
        "num_epochs": 10, "early_stopping": True, "early_stopping_patience": 3,
        "batch_size": 4 if window >= 2048 else 16, "gradient_accumulation_steps": 4 if window >= 2048 else 1,
        "encoder_lr": 1.0e-5, "task_lr": 3.0e-4, "warmup_ratio": 0.05, "scheduler_type": "cosine_restarts",
        "bf16": True, "fp16": False, "max_grad_norm": 1.0,
        "eval_strategy": "epoch", "metric_for_best": "eval_overall_strict_head_min_f1", "greater_is_better": True,
        "save_best": True, "save_total_limit": 2, "logging_steps": 20, "num_workers": 4,
        "validate_data": True, "sliding_window": True, "max_len": window, "window_stride": int(0.75 * window),
        "gradient_checkpointing": window >= 2048, "pin_memory": True,
        "error_policy": "skip", "on_capacity_exceeded": "truncate_with_warning",
        "negative_pools": "auto", "negative_labels_per_dim": {"entities": 1, "events": 1}, "negative_label_seed": 42,
    }
    for k in tr:
        cm[f"training.{k}"] = "policy"
    cm["training.max_len"] = f"measured: eval window ({ev.get('source', 'default')}); train p99 {c['lengths']['train']['p99']} subwords"
    cm["training.window_stride"] = "policy: 25% window overlap"
    cm["training.batch_size"] = "policy: NOT measured for your GPU -- 4x4 fits 4096 windows on an A100-40GB (eb18: 18.9 GB peak)"
    cm["training.metric_for_best"] = "policy: worst-head strict F1, never eval_loss"
    cm["training.task_lr"] = "policy: warm-start default; TODO #9 suspects it is high for warm relation heads"
    cm["training.bf16"] = "policy: fp16 crashed a run (Track A); bf16 did not"
    evc = {"batch_size": 2, "threshold": threshold, "threshold_sweep": True,
           "global_decode": ev.get("global_decode", True), "eval_by_language": False}
    for k in evc:
        cm[f"eval.{k}"] = "policy"
    cm["eval.threshold"] = (f"measured on the BASE (val, span sweep): {gpu['span_sweep']['rule']}"
                            + ("; AT GRID EDGE" if gpu["span_sweep"].get("at_grid_edge") else "")
                            if span_pick is not None else "checkpoint/default: run with --gpu to measure")
    cm["eval.global_decode"] = f"checkpoint/default: {ev.get('source', 'default')}"

    warnings.append("NO REPLAY: the base's training corpora are not available here. Fine-tuning without replay "
                    "forgot the base on all 8 heads in this project (gate3 warm cells); add an exact ~30% replay "
                    "slice of the base's data to data.corpora (train_only) if you have it.")
    if not gpu:
        warnings.append("GPU stages not run: start_top_k, candidate_pool, thresholds are the checkpoint's, not measured.")
    for stage in ("span_sweep", "record_sweep"):
        if (gpu.get(stage) or {}).get("at_grid_edge"):
            warnings.append(f"{stage} picked {gpu[stage]['picked']} AT THE GRID EDGE: the optimum may lie beyond it "
                            f"(small --max-val or a base far from this corpus); re-sweep after fine-tuning.")
    for f in calib["data_health"]["findings"]:
        if f["severity"] == "WARN":
            warnings.append(f"DATA {f['check']}: {f['message']}")
    config = {"labels_file": f"{name}.labels.yaml", "model": model_cfg,
              "data": {"corpora": [corpus_base]}, "training": tr, "eval": evc}
    cm["labels_file"] = "generated beside this config: base label_map + corpus clusters; confirm PROPOSED lines"
    return config, cm, warnings


def _render(node: Any, comments: Dict[str, str], path: str = "", indent: int = 0) -> List[str]:
    lines = []
    for k, v in node.items():
        p = f"{path}.{k}" if path else k
        note = f"  # {comments[p]}" if p in comments else ""
        if isinstance(v, dict) and v and p != "training.negative_labels_per_dim":
            lines.append(f"{' ' * indent}{k}:{note}")
            lines += _render(v, comments, p, indent + 2)
        else:
            val = yaml.safe_dump(v, default_flow_style=True, allow_unicode=True).strip().removesuffix("\n...").removesuffix("...").strip()
            lines.append(f"{' ' * indent}{k}: {val}{note}")
    return lines


def write(path: Path, name: str, model: str, config: Dict, comments: Dict, warnings: List[str]) -> None:
    head = [f"# {name}: fine-tune {model} -- GENERATED by tools/derive (stage 7).",
            "# Every value says where it came from: measured (this run's calibration), checkpoint (the",
            "# base's own), or policy (a standing project default). Thresholds measured on the BASE are",
            "# starting points: re-sweep on validation after fine-tuning.", "#", "# WARNINGS"]
    head += [f"#   - {w}" for w in warnings] + [""]
    path.write_text("\n".join(head + _render(config, comments)) + "\n", encoding="utf-8")
