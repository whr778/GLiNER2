"""Calibrate the coreference-link merge threshold on validation, once, after training.

COREFERENT_LINK_SPEC.md section 6. Each grid value decodes the calibration set with
``record_merge_coreferent: link`` at that threshold, beside an ``off`` row. A threshold is
eligible only if it makes at least MIN_MERGES merges (a gate that admits nothing scores a
perfect precision), its merges pass gates 4-5 (merge precision >= 0.8; fusion -- the share of
merges joining DIFFERENT gold events -- < 5%), AND neither its ``event_cluster`` F1 nor its
``event_cluster_argument`` F1 falls below ``off`` -- on the calibration set AND, when given, on a
CONTROL set of one-trigger-per-event corpora (CMNEE, MAVEN), where also fewer than 5% of merges may
join different gold events. The calibration gold is the only coreferent gold, but the merge ships to
every corpus. F1 alone cannot guard the control: the cluster metric matches on ANY shared trigger, so a
fused event collects the pooled arguments of several gold events and F1 can RISE -- traced at 0.5 on
the init link, control argument F1 0.2134 -> 0.2417 while 30 of 62 control merges were fusions.
The spec's pair-based fusion (gold hard-negative pairs merged) is reported as ``pair_fusion``
with its support but not gated: traced on 40 val docs it had ONE pair, so one merge swung it
between 0 and 1. The best eligible one is written into
the checkpoint's config.json as the default decode; if none is eligible the merge stays
``off``. Either way the whole table goes to ``coref_threshold_sweep.json``.
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

COREF_THRESHOLD_GRID: Tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
MIN_MERGE_PRECISION = 0.8
MAX_FUSION = 0.05
MIN_MERGES = 20


def merge_quality(gold: list, pred: list, mentions: List[int]) -> Counter:
    """Count one document's merges against its gold clusters (``_*_event_clusters`` form).

    ``mentions[i]`` is predicted event i's RAW trigger count: a merge of "meeting" + "meeting"
    is one distinct casefolded mention but still a merge. A predicted event with >= 2 raw
    trigger mentions is a MERGE: ``right`` when every mention is a
    trigger of ONE same-type gold event, ``fused`` when its mentions hit >= 2 gold events,
    ``absorbed`` otherwise (it pulled in a non-gold trigger). ``hardneg_pairs`` counts pairs of
    predicted mentions belonging to DIFFERENT same-type gold events; ``hardneg_merged`` those
    of them a single predicted event holds.
    """
    c = Counter()
    for (etype, ptrig, _), n in zip(pred, mentions):
        if n < 2:
            continue
        hits = [j for j, (gt, gtrig, _) in enumerate(gold) if gt == etype and gtrig & ptrig]
        c["merged"] += 1
        if len(hits) >= 2:
            c["fused"] += 1
        elif len(hits) == 1 and ptrig <= gold[hits[0]][1]:
            c["right"] += 1
        else:
            c["absorbed"] += 1
    for etype in {g[0] for g in gold}:
        owner = {}
        for j, (gt, gtrig, _) in enumerate(gold):
            if gt == etype:
                for m in gtrig:
                    owner[m] = j if m not in owner else None
        held = {}
        for i, (pt, ptrig, _) in enumerate(pred):
            if pt == etype:
                for m in ptrig:
                    if owner.get(m) is not None:
                        held.setdefault(m, set()).add(i)
        for a, b in combinations(sorted(held), 2):
            if owner[a] != owner[b]:
                c["hardneg_pairs"] += 1
                c["hardneg_merged"] += bool(held[a] & held[b])
    return c


def _pred_clusters_with_mentions(pred: Dict) -> Tuple[list, List[int]]:
    """``_pred_event_clusters`` plus each event's raw trigger count, in the same order."""
    from gliner2.training.eval_metrics import _event_cluster
    clusters, counts = [], []
    block = pred.get("event_extraction") or {}
    for etype, events in (block.items() if isinstance(block, dict) else []):
        for ev in events if isinstance(events, list) and isinstance(etype, str) else []:
            c = _event_cluster(etype, ev.get("triggers"), ev.get("arguments")) if isinstance(ev, dict) else None
            if c:
                clusters.append(c)
                counts.append(len(ev.get("triggers") or []))
    return clusters, counts


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def score(golds: List[Dict], preds: List[Dict]) -> Dict[str, Any]:
    """event_cluster F1, event_cluster_argument F1 and merge quality over a set of documents."""
    from gliner2.training.eval_metrics import _gold_event_clusters, _tally_event_clusters
    ev, arg = (Counter(), Counter(), Counter()), (Counter(), Counter(), Counter())
    q = Counter()
    for gold, pred in zip(golds, preds):
        g, (p, n) = _gold_event_clusters(gold), _pred_clusters_with_mentions(pred)
        _tally_event_clusters(g, p, ev, arg)
        q += merge_quality(g, p, n)
    return {
        "event_cluster_f1": _f1(*(sum(x.values()) for x in ev)),
        "event_cluster_argument_f1": _f1(*(sum(x.values()) for x in arg)),
        "events_predicted": sum(ev[0].values()) + sum(ev[1].values()),
        "merged": q["merged"], "right": q["right"], "fused": q["fused"], "absorbed": q["absorbed"],
        "merge_precision": q["right"] / q["merged"] if q["merged"] else None,
        "fusion": q["fused"] / q["merged"] if q["merged"] else 0.0,
        "hardneg_pairs": q["hardneg_pairs"], "hardneg_merged": q["hardneg_merged"],
        "pair_fusion": q["hardneg_merged"] / q["hardneg_pairs"] if q["hardneg_pairs"] else None,
    }


def _predict(model, records: List[Dict], decode: Dict[str, Any], batch_size: int) -> Tuple[list, list]:
    """(golds, preds) for records with event gold, each offered its own gold schema as eval does.
    Event-free documents are skipped, so a false merge on one cannot be seen; on the calibration and
    control sets every document the merge can touch carries event gold."""
    from gliner2.training.eval_metrics import _schema_from_gold
    keep = [r for r in records if (r.get("output") or {}).get("events")]
    schemas = [_schema_from_gold(r["output"]) for r in keep]
    preds = model.batch_extract_long([r["input"] for r in keep], schemas, batch_size=batch_size, **decode)
    return [r["output"] for r in keep], preds


def multi_trigger_events(records: List[Dict]) -> int:
    """Gold events listing more than one trigger; a CONTROL corpus must have none."""
    return sum(1 for r in records for e in (r.get("output") or {}).get("events") or []
               if isinstance(e, dict) and len(e.get("triggers") or []) > 1)


def sweep(model, records: List[Dict], decode: Dict[str, Any], grid=COREF_THRESHOLD_GRID,
          batch_size: int = 8, control: Optional[List[Dict]] = None) -> Dict[str, Dict[str, Any]]:
    """{"off": row, "<t>": row} -- the calibration set decoded without the merge and at each threshold.
    With ``control`` (one-trigger-per-event gold), each row also carries ``control``: the same scores on
    corpora where the merge can only cost, which the calibration set alone cannot see."""
    from gliner2.training.eval_metrics import apply_boundary_overrides
    rows = {}
    for t in ("off", *grid):
        mode = {"record_merge_coreferent": "off"} if t == "off" else {
            "record_merge_coreferent": "link", "record_coref_link_threshold": float(t)}
        apply_boundary_overrides(model, mode)
        row = score(*_predict(model, records, decode, batch_size))
        if control:
            row["control"] = score(*_predict(model, control, decode, batch_size))
        rows[str(t)] = row
        print(f"[coref sweep] {t}: {json.dumps(row)}", flush=True)
    return rows


_F1 = ("event_cluster_f1", "event_cluster_argument_f1")


def _no_worse(row: Dict[str, Any], off: Dict[str, Any]) -> bool:
    return all(row[k] >= off[k] for k in _F1)


def _eligible(row: Dict[str, Any], off: Dict[str, Any]) -> bool:
    return (row["merged"] >= MIN_MERGES and row["merge_precision"] >= MIN_MERGE_PRECISION
            and row["fusion"] < MAX_FUSION and _no_worse(row, off)
            and ("control" not in off or (_no_worse(row["control"], off["control"])
                                          and row["control"]["fusion"] < MAX_FUSION)))


def choose(rows: Dict[str, Dict[str, Any]]) -> Tuple[Optional[float], str]:
    """(threshold, why) -- the best eligible threshold by event_cluster F1, or (None, why not)."""
    off = rows["off"]
    eligible = {t: r for t, r in rows.items() if t != "off" and _eligible(r, off)}
    gates = (f">= {MIN_MERGES} merges, precision >= {MIN_MERGE_PRECISION}, fusion < {MAX_FUSION}, cluster and "
             f"argument F1 >= off" + (f" on calibration AND control (control fusion < {MAX_FUSION})"
                                      if "control" in off else ""))
    if not eligible:
        return None, f"no threshold passes {gates}; merge stays off"
    t = max(eligible, key=lambda k: eligible[k]["event_cluster_f1"])
    r = eligible[t]
    ctrl = (f"; control cluster F1 {r['control']['event_cluster_f1']:.4f} vs off "
            f"{off['control']['event_cluster_f1']:.4f}" if "control" in r else "")
    return float(t), (f"{r['merged']} merges, precision {r['merge_precision']:.3f}, fusion {r['fusion']:.3f}, "
                      f"event_cluster F1 {r['event_cluster_f1']:.4f} vs off {off['event_cluster_f1']:.4f}{ctrl}")


def write(checkpoint: Path, rows: Dict[str, Dict[str, Any]], threshold: Optional[float], why: str,
          source: List[str], control: Optional[List[str]] = None) -> None:
    """Record the sweep beside the checkpoint and set config.json's default merge from it."""
    checkpoint = Path(checkpoint)
    (checkpoint / "coref_threshold_sweep.json").write_text(json.dumps(
        {"chosen_threshold": threshold, "why": why, "calibration_files": source,
         "control_files": control or [], "by_threshold": rows},
        indent=2, ensure_ascii=False), encoding="utf-8")
    path = checkpoint / "config.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    bh = cfg.setdefault("boundary_head", {})
    bh["record_merge_coreferent"] = "link" if threshold is not None else "off"
    if threshold is not None:
        bh["record_coref_link_threshold"] = threshold
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[coref sweep] {checkpoint}: record_merge_coreferent={bh['record_merge_coreferent']}"
          + (f" threshold={threshold}" if threshold is not None else "") + f" -- {why}")
