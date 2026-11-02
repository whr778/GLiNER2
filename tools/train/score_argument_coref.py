"""Score argument-to-cluster links. This is not a training head and does not move head-min F1.

Without ``--model``, each JSONL row is an infer.py line (``output`` is the prediction,
``gold`` plus ``gold_coref`` is the reference) or a data row (``output`` is the gold and
there is no prediction). With ``--model``, the checkpoint decodes spans, clusters
mentions when it has a mention link, and the join is scored against the row's gold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from gliner2.inference.argument_coref import score_argument_clusters


def _rows(path: str) -> List[Dict[str, Any]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _gold(row: Dict[str, Any]) -> Dict[str, Any]:
    if "gold" in row:
        gold = dict(row.get("gold") or {})
        if row.get("gold_coref"):
            gold["coref_clusters"] = row["gold_coref"]
        return gold
    gold = dict(row.get("output") or {})
    if row.get("coref_clusters"):
        gold["coref_clusters"] = row["coref_clusters"]
    return gold


def _stored_pred(row: Dict[str, Any]) -> Dict[str, Any]:
    output = row.get("output") if "gold" in row else {}
    return output if isinstance(output, dict) else {}


def _decode(rows: List[Dict[str, Any]], model_name: str, threshold: float, coref_threshold: float,
            device: str) -> List[Dict[str, Any]]:
    from gliner2 import AutoExtractor
    from gliner2.inference.entity_coref import candidate_mentions, resolve
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import infer

    model = AutoExtractor.from_pretrained(model_name, map_location=None if device == "auto" else device)
    defaults = getattr(model.config, "inference_defaults", None) or {}
    cut = threshold if threshold is not None else float(defaults.get("threshold") or 0.5)
    schemas = [infer._gold_schema(row) for row in rows]
    keep = [i for i, schema in enumerate(schemas) if schema]
    preds: List[Dict[str, Any]] = [{} for _ in rows]
    if keep:
        got = model.batch_extract_long(
            [rows[i]["input"] for i in keep],
            [schemas[i] for i in keep],
            batch_size=4,
            threshold=cut,
            include_spans=True,
            chunk_size=int(defaults.get("chunk_size") or 384),
            chunk_overlap=int(defaults.get("chunk_overlap") or 0),
            global_decode=bool(defaults.get("global_decode") or False),
        )
        for index, pred in zip(keep, got):
            preds[index] = pred
    if getattr(model, "mention_link", None) is None:
        print("[argument-coref] checkpoint has no mention link; clusters are absent", flush=True)
        return preds
    for row, pred in zip(rows, preds):
        mentions = candidate_mentions(row.get("input") or "", pred.get("entities"))
        pred["coref_clusters"] = [
            cluster for cluster in resolve(model, row.get("input") or "", mentions, coref_threshold)
            if len(cluster) > 1
        ]
    return preds


def _print(metrics: Dict[str, Any]) -> None:
    linked = metrics["eval_argument_cluster_link_linked"]
    ambiguous = metrics["eval_argument_cluster_link_ambiguous"]
    none = metrics["eval_argument_cluster_link_none"]
    print(f"[argument-coref] linked={linked} ambiguous={ambiguous} none={none} "
          f"gold_documents={metrics['eval_argument_cluster_link_gold_documents']} "
          f"ungraded_documents={metrics['eval_argument_cluster_link_ungraded_documents']}")
    if "eval_argument_cluster_link_strict_micro_f1" not in metrics:
        print("[argument-coref] no gold clusters; strict micro F1 omitted")
        return
    print(f"[argument-coref] strict micro "
          f"P={metrics['eval_argument_cluster_link_strict_micro_precision']:.4f} "
          f"R={metrics['eval_argument_cluster_link_strict_micro_recall']:.4f} "
          f"F1={metrics['eval_argument_cluster_link_strict_micro_f1']:.4f} "
          f"support={metrics['eval_argument_cluster_link_support']}")


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Score argument spans against coreference clusters.")
    parser.add_argument("data", help="JSONL. Data rows carry gold output and coref_clusters. "
                        "Infer rows carry output, gold and gold_coref.")
    parser.add_argument("--model", default=None, help="Decode this checkpoint. Without it, stored predictions are scored.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--entity-coref", type=float, default=0.5, help="Mention-link cut-off when --model has a link.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args(argv)
    rows = _rows(args.data)
    golds = [_gold(row) for row in rows]
    preds = _decode(rows, args.model, args.threshold, args.entity_coref, args.device) if args.model else [
        _stored_pred(row) for row in rows]
    _print(score_argument_clusters(golds, preds))


if __name__ == "__main__":
    main()
