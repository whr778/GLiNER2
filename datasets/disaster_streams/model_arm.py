"""Model extraction arm: a zero-shot GLiNER2 model binds numbers to roles.

For each report (one snippet) the model fills a casualty_report structure
{dead, injured, missing, source}; the normalizer (extract.value_qualifier) turns each
BOUND field span into (value, qualifier) -- so the model solves the number-to-role binding
the surface parser can't (design sec 17). Reports role precision/recall + value/qualifier
accuracy on true positives, and writes a model-extracted observations.jsonl for end-to-end
tracking with evaluate.py.

  uv run python datasets/disaster_streams/model_arm.py --data datasets/disaster_streams_sonnet5 \
      --split val --model fastino/gliner2-base-v1 --out datasets/disaster_streams_model
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import extract  # noqa: E402
from gliner2 import GLiNER2, Schema  # noqa: E402

ROLES = ("dead", "injured", "missing")


def build_schema() -> Schema:
    return (Schema()
            .structure("casualty_report")
            .field("dead", dtype="str", description="number of people killed or confirmed dead")
            .field("injured", dtype="str", description="number of people injured or hurt")
            .field("missing", dtype="str", description="number of people missing or unaccounted for")
            .field("source", dtype="str", description="who reported these figures"))


def _cell(v):
    """extract() with include_confidence returns {text, confidence} per field (or a bare
    string without it); normalize to (span, confidence)."""
    if isinstance(v, dict):
        return v.get("text", ""), float(v.get("confidence", 1.0))
    return (v or ""), 1.0


def model_extract(ex, schema, text: str, thr: float):
    """text -> ({role: {value, qualifier, confidence}}, source). Keep a role only if its
    bound span holds a digit and its confidence clears thr (post-filter -> precision)."""
    recs = ex.extract(text, schema, include_confidence=True).get("casualty_report") or [{}]
    rec = recs[0] if recs else {}
    out = {}
    for role in ROLES:
        span, conf = _cell(rec.get(role))
        if conf >= thr and any(c.isdigit() for c in span):
            v, q = extract.value_qualifier(span)
            out[role] = {"value": v, "qualifier": q, "confidence": round(conf, 4)}
    src_span, _ = _cell(rec.get("source"))
    return out, (extract._detect_source(src_span) if src_span else None)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/disaster_streams_sonnet5")
    ap.add_argument("--split", default="val")
    ap.add_argument("--model", default="fastino/gliner2-base-v1")
    ap.add_argument("--out", default="datasets/disaster_streams_model")
    ap.add_argument("--streams", type=int, default=0, help="0 = all streams in the subset")
    ap.add_argument("--threshold", type=float, default=None,
                    help="per-field confidence cutoff to suppress spurious fills (raise precision)")
    args = ap.parse_args(argv)

    src_dir = Path(args.data) / args.split
    groups: dict = defaultdict(lambda: {"text": "", "gt": {}})
    for line in (src_dir / "observations.jsonl").open(encoding="utf-8"):
        o = json.loads(line)
        g = groups[(o["stream_id"], o["t_hours"])]
        g["text"] = o.get("text", ""); g["gt"][o["role"]] = o
    if args.streams:
        keep = sorted({k[0] for k in groups})[:args.streams]
        groups = {k: v for k, v in groups.items() if k[0] in keep}

    ex = GLiNER2.from_pretrained(args.model, map_location="cpu")
    schema = build_schema()
    thr = args.threshold if args.threshold is not None else 0.0

    tp = fp = fn = val_n = val_exact = qual_ok = 0
    rows = []
    for (sid, t), g in sorted(groups.items()):
        pred, src = model_extract(ex, schema, g["text"], thr)
        gt = g["gt"]
        for role in ROLES:
            if role in gt and role in pred:
                tp += 1; val_n += 1
                val_exact += int(pred[role]["value"] == gt[role]["value"])
                qual_ok += int(pred[role]["qualifier"] == gt[role]["qualifier"])
            elif role in pred:
                fp += 1
            elif role in gt:
                fn += 1
            if role in pred:
                rows.append({"stream_id": sid, "t_hours": t, "role": role,
                             "value": pred[role]["value"],
                             "qualifier": pred[role]["qualifier"],
                             "source": src or "preliminary",
                             "confidence": pred[role]["confidence"]})

    out_dir = Path(args.out) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "observations.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sids = {k[0] for k in groups}
    with (out_dir / "trajectory.jsonl").open("w", encoding="utf-8") as tf:
        for line in (src_dir / "trajectory.jsonl").open(encoding="utf-8"):
            if json.loads(line)["stream_id"] in sids:
                tf.write(line)

    prec = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"[model_arm] {args.model}: {len(groups)} reports; "
          f"role P={prec:.3f} R={recall:.3f} (tp={tp} fp={fp} fn={fn}); "
          f"value exact={val_exact / max(val_n, 1):.3f} "
          f"qualifier acc={qual_ok / max(val_n, 1):.3f} (on {val_n} TP) -> {out_dir}")


if __name__ == "__main__":
    main()
