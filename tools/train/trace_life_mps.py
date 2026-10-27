"""One WikiEvents document through the real training path, on MPS, fresh mDeBERTa heads.

Two forwards of the same batch: gold injection 1.0 (step 0 of the schedule) and 0.0
(the annealed end-state). Prints each stage and appends one JSON line per stage to
the debug log. Random heads: the numbers are the code path, not a trained model.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/train"))
import train as T  # noqa: E402

LOG = ROOT / ".cursor/debug-834c72.log"
DEV = ROOT / "data/wikievents.dev.jsonl"
NEEDLE = "7 militants killed"


def dlog(hypothesis: str, message: str, data: dict, location: str) -> None:
    # #region agent log
    line = json.dumps({
        "sessionId": "834c72",
        "runId": "life1",
        "hypothesisId": hypothesis,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }, ensure_ascii=False)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    # #endregion


def stage(name: str) -> None:
    print(f"\n[life] ===== {name} =====", flush=True)


def arg_count(schema: dict) -> int:
    n = 0
    for ev in (schema or {}).get("events") or []:
        n += len(ev.get("arguments") or [])
    return n


def pick() -> dict:
    for line in DEV.open(encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("input", "").startswith(NEEDLE) and (rec.get("output") or {}).get("events"):
            return rec
    raise SystemExit(f"no record starting {NEEDLE!r} in {DEV}")


def main() -> None:
    cfg_path = ROOT / "tools/train/config/trace/life-mdeberta-wikievents.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    tr, head_cfg = cfg["training"], cfg["model"]["boundary_head"]
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available")

    from gliner2 import AutoExtractor
    from gliner2.processing.argument_coref import expand_argument_coreferents
    from gliner2.training.chunking import chunk_records
    from gliner2.training.negatives import NegativeLabels
    from gliner2.training.trainer import ExtractorCollator, ExtractorDataset
    import gliner2.models.boundary.records as records_mod
    import gliner2.processing.coref as coref

    spec = importlib.util.spec_from_file_location("bnp", ROOT / "tools/data/build_negative_pools.py")
    bnp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bnp)

    stage("S0 raw record")
    raw = pick()
    events = raw["output"]["events"]
    print(f"[life] chars {len(raw['input'])} events {len(events)} clusters {len(raw.get('coref_clusters') or [])}")
    print(f"[life] text: {raw['input'][:240]!r}")
    for ev in events:
        args = [a.get("entity") if isinstance(a, dict) else a for a in ev.get("arguments") or []]
        print(f"[life]   {ev.get('event_type')} triggers={ev.get('triggers')} args={args}")
    dlog("C", "raw record", {
        "chars": len(raw["input"]),
        "events": len(events),
        "arguments": arg_count(raw["output"]),
        "clusters": len(raw.get("coref_clusters") or []),
        "trigger_types": [type(t).__name__ for ev in events for t in (ev.get("triggers") or [])],
        "entity_types": [type(a.get("entity")).__name__ for ev in events for a in (ev.get("arguments") or []) if isinstance(a, dict)],
    }, "trace_life_mps.py:S0")

    stage("S1 label transform")
    fns = T._category_fns(T.load_labels_cfg(cfg, str(cfg_path)))
    rec = T.transform_record(raw, fns)
    rec["_corpus"] = "wikievents"
    before = [(ev.get("event_type"), ev.get("triggers")) for ev in events]
    after = [(ev.get("event_type"), ev.get("triggers")) for ev in rec["output"]["events"]]
    print(f"[life] event types before {before}")
    print(f"[life] event types after  {after}")
    dlog("D", "label transform", {"before": before, "after": after, "changed": before != after}, "trace_life_mps.py:S1")

    stage("S2 chunking")
    # Tokenizer only, so chunking does not wait on the encoder weights twice.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["encoder"])
    n_tok = len(tok(rec["input"])["input_ids"])
    chunks = chunk_records([rec], tokenizer=tok, window_size=int(tr["max_len"]),
                           stride=int(tr["window_stride"]), show_progress=False)
    print(f"[life] subwords {n_tok} window {tr['max_len']} -> {len(chunks)} chunk(s)")
    dlog("E", "chunking", {"subwords": n_tok, "chunks": len(chunks), "max_len": int(tr["max_len"])}, "trace_life_mps.py:S2")

    stage("S3 negatives + argument-coref expansion")
    pools = bnp.build_pools(cfg, cfg_path, log=lambda *a: None)
    neg = NegativeLabels(pools, tr["negative_labels_per_dim"], seed=int(tr.get("negative_label_seed", 42)),
                         partial=cfg["data"].get("partial_annotation"),
                         gold_full_roles=bool(tr.get("negative_gold_full_roles", False)))
    bare_args = arg_count(chunks[0]["output"])
    expanded = expand_argument_coreferents(chunks[0]["output"], chunks[0].get("coref_clusters"), chunks[0]["input"])
    print(f"[life] arguments before expansion {bare_args} after {arg_count(expanded)}")
    ds = ExtractorDataset(data=chunks, max_samples=-1, shuffle=False, seed=42, validate=False,
                          negatives=neg, expand_argument_coref=True)
    text, schema = ds[0]
    print(f"[life] schema events {len(schema.get('events') or [])} arguments {arg_count(schema)}")
    print(f"[life] injector: {neg.composition_line()}")
    dlog("C", "argument expansion", {
        "args_before": bare_args,
        "args_after_direct": arg_count(expanded),
        "args_after_getitem": arg_count(schema),
        "schema_has_coref_key": "_coref_clusters" in schema,
        "injector": neg.composition_line(),
    }, "trace_life_mps.py:S3")

    stage("load mDeBERTa on MPS")
    model = AutoExtractor.from_encoder(
        cfg["model"]["encoder"],
        architecture="boundary",
        max_len=int(cfg["model"]["max_len"]),
        attn_implementation=cfg["model"]["attn_implementation"],
        boundary_head=head_cfg,
        map_location="mps",
    )
    model.train()
    proc = model.processor

    stage("S4 collator")
    coll = ExtractorCollator(
        proc, is_training=True, max_len=tr.get("max_len"), architecture="boundary",
        max_gold_per_query=int(head_cfg.get("max_gold_per_query", 32)),
        on_capacity_exceeded="raise", error_policy="raise",
        event_records=bool(head_cfg.get("event_records", False)),
    )
    placed_before = dict(coref.COUNTS)
    batch = coll([(text, schema)])
    placed = {k: coref.COUNTS[k] - placed_before.get(k, 0) for k in coref.COUNTS}
    layout = batch.query_layouts[0]
    gold = batch.targets.mention_mask[0].sum(-1)
    n_absent = 0
    by_task = {}
    for q in layout.queries:
        g = int(gold[q.query_id])
        by_task[q.task_type] = by_task.get(q.task_type, 0) + 1
        if g == 0:
            n_absent += 1
        print(f"[life]   q{q.query_id:<3d} {q.task_type:9s} {str(q.task_name)[:28]:28s} {str(q.role_name)[:16]:16s} gold {g}")
    raw_records = batch.targets.records
    records = raw_records[0] if raw_records else []
    if not records:
        print(f"[life] record targets missing ({type(raw_records).__name__})")
    anchors = []
    for rec_t in records:
        field = rec_t.field_for_query(rec_t.anchor_query_id) if rec_t.anchor_query_id is not None else None
        n_spans = 0 if field is None else sum(len(v) for v in field.values)
        anchors.append({"instance": rec_t.instance_id, "anchor_qid": rec_t.anchor_query_id, "anchor_spans": n_spans, "fields": len(rec_t.fields)})
        print(f"[life]   record {rec_t.instance_id} anchor_qid={rec_t.anchor_query_id} anchor_spans={n_spans} fields={len(rec_t.fields)}")
    print(f"[life] queries {len(layout.queries)} absent {n_absent} by_task {by_task} coref_placed {placed}")
    dlog("E", "collator", {
        "queries": len(layout.queries),
        "absent_queries": n_absent,
        "by_task": by_task,
        "records": len(records),
        "anchors": anchors,
        "coref_counts": placed,
        "words": int(batch.text_word_counts[0]),
    }, "trace_life_mps.py:S4")

    def schedules(step: int) -> dict:
        """The three scales ExtractorTrainer._backward_one writes before the forward.

        Planned steps are eb22d's 82,930, not this one-corpus config's. Consistency
        warmup is 2,000 steps. Soft-IoU anneals over 20,000. Gold injection holds
        at 1.0 for the first 15 percent.
        """
        from gliner2.training.trainer import ExtractorTrainer
        planned = 82930
        settings = model.boundary_head.settings
        warm = int(settings.consistency_warmup_steps)
        injection = ExtractorTrainer._gold_injection_probability(
            step / planned, 1.0, 0.25, 0.15)
        consistency = 1.0 if warm <= 0 else min(step / warm, 1.0)
        soft = ExtractorTrainer._soft_iou_anneal_scale(step, int(settings.soft_iou_anneal_steps))
        return {
            "step": step, "planned": planned, "injection": injection,
            "consistency_scale": consistency, "soft_iou_scale": soft,
            "consistency_warmup_steps": warm,
            "soft_iou_anneal_steps": int(settings.soft_iou_anneal_steps),
        }

    def forward(step: int, tag: str, hypothesis: str) -> None:
        sched = schedules(step)
        stage(f"S5/S6 {tag} step={step}")
        head = model.boundary_head
        head.set_gold_injection_prob(sched["injection"])
        head.set_consistency_scale(sched["consistency_scale"])
        head.set_soft_iou_scale(sched["soft_iou_scale"])
        before_gate = dict(records_mod._ANCHOR_GATE)
        torch.manual_seed(0)
        out = model(batch)
        gate = {k: records_mod._ANCHOR_GATE[k] - before_gate.get(k, 0) for k in records_mod._ANCHOR_GATE}
        L = out.losses
        obj = L["record_object_loss"]
        field = L["record_field_loss"]
        cons = L["consistency_loss"]
        iou = L["soft_iou_loss"]
        payload = {
            **sched,
            "record_object_loss": float(obj.detach()),
            "object_requires_grad": bool(obj.requires_grad),
            "record_field_loss": float(field.detach()),
            "field_requires_grad": bool(field.requires_grad),
            "consistency_loss": float(cons.detach()),
            "consistency_contrib": float(cons.detach()) * head.settings.consistency_loss_weight * sched["consistency_scale"],
            "soft_iou_loss": float(iou.detach()),
            "soft_iou_contrib": float(iou.detach()) * head.settings.soft_iou_aux_weight * sched["soft_iou_scale"],
            "total_loss": float(out.loss.detach()),
            "anchor_gate_delta": gate,
        }
        print(f"[life] {tag} {json.dumps(payload, ensure_ascii=False)}")
        dlog(hypothesis, tag, payload, "trace_life_mps.py:forward")

    # Step 0: consistency warmup is on (scale 0). Step 2000: that warmup is off (scale 1).
    forward(0, "warmup-on", "A")
    forward(2000, "warmup-off", "B")

    # record_merge_coreferent is read by decode_group only. Three eval forwards
    # with injection 1.0 must match if the loss ignores the setting. Three
    # extracts then show off / args / link on the same document.
    from gliner2.training.eval_metrics import apply_boundary_overrides, _schema_from_gold

    def event_summary(pred: dict) -> dict:
        # extract() nests events under event_extraction, not at the top level.
        events = (pred or {}).get("event_extraction") or {}
        instances = multi = 0
        by_type = {}
        for task, rows in events.items() if isinstance(events, dict) else []:
            if not isinstance(rows, list):
                continue
            by_type[task] = len(rows)
            for inst in rows:
                trig = inst.get("triggers") if isinstance(inst, dict) else None
                if not isinstance(trig, list):
                    continue
                instances += 1
                n = len(trig)
                if n > 1:
                    multi += 1
        return {"event_instances": instances, "multi_trigger": multi, "by_type": by_type}

    stage("S7 merge modes")
    # Record loss is computed only while training (model.py). Eval mode skips it,
    # so the comparison stays in train mode with a fixed seed: dropout then draws
    # the same noise, and a setting the loss reads would move the numbers.
    model.train()
    head = model.boundary_head
    head.set_gold_injection_prob(1.0)
    head.set_consistency_scale(1.0)
    head.set_soft_iou_scale(1.0)
    for mode in ("off", "args", "link"):
        apply_boundary_overrides(model, {"record_merge_coreferent": mode})
        torch.manual_seed(0)
        out = model(batch)
        if "record_object_loss" not in (out.losses or {}):
            missing = {"merge": mode, "loss_keys": sorted((out.losses or {}).keys())}
            print(f"[life] loss missing record term {json.dumps(missing)}")
            dlog("A", "record loss absent", missing, "trace_life_mps.py:merge-loss")
            continue
        obj = out.losses["record_object_loss"]
        loss_payload = {
            "merge": mode,
            "yaml": head_cfg.get("record_merge_coreferent"),
            "live": model.boundary_head.settings.record_merge_coreferent,
            "record_object_loss": float(obj.detach()),
            "object_requires_grad": bool(obj.requires_grad),
            "record_field_loss": float(out.losses["record_field_loss"].detach()),
            "total_loss": float(out.loss.detach()),
        }
        print(f"[life] loss {json.dumps(loss_payload, ensure_ascii=False)}")
        dlog("A", f"loss under {mode}", loss_payload, "trace_life_mps.py:merge-loss")
    model.eval()
    infer_schema = _schema_from_gold(raw["output"])
    print(f"[life] infer schema {json.dumps(infer_schema, ensure_ascii=False)[:500]}")
    for mode in ("off", "args", "link"):
        apply_boundary_overrides(model, {"record_merge_coreferent": mode})
        for threshold in (0.3, 0.0):
            pred = model.extract(text, infer_schema, threshold=threshold, include_spans=True, include_confidence=True)
            summary = {
                "merge": mode, "threshold": threshold,
                "keys": {k: (type(v).__name__ if not isinstance(v, list) else len(v)) for k, v in (pred or {}).items()},
                **event_summary(pred),
            }
            print(f"[life] decode {json.dumps(summary, ensure_ascii=False)}")
            dlog("B", f"decode {mode} @{threshold}", summary, "trace_life_mps.py:merge-decode")
    print("\n[life] done", flush=True)


if __name__ == "__main__":
    main()
