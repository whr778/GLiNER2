"""Model extraction arm: a zero-shot GLiNER2 model binds numbers to roles.

For each report (one snippet) the model fills a casualty_report structure
{dead, injured, missing, source}; the normalizer turns each BOUND span into
(value via extract.value_qualifier, qualifier via extract.qualifier_near = the hedge
beside the located number) -- so the model solves the number-to-role binding the surface
parser can't (design sec 17-18). Reports role precision/recall + value/qualifier accuracy
on true positives, and writes a model-extracted observations.jsonl for end-to-end tracking.

The model pass is cached to raw.jsonl (bound span + confidence + text per role); --from-raw
re-normalizes with no model reload, so normalization tweaks are free.

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

ROLES = ("dead", "injured", "missing")


def build_schema():
    from gliner2 import Schema
    return (Schema()
            .structure("casualty_report")
            .field("dead", dtype="str",
                   description="number of people killed or confirmed dead, not injured/missing/displaced")
            .field("injured", dtype="str",
                   description="number of people injured or hurt, not killed/missing/displaced/homeless")
            .field("missing", dtype="str",
                   description="number of people missing or unaccounted for, not killed/injured/displaced")
            .field("source", dtype="str", description="who reported these figures"))


def _cell(v):
    """extract() with include_confidence returns {text, confidence} per field (or a bare
    string); normalize to (span, confidence)."""
    if isinstance(v, dict):
        return v.get("text", ""), float(v.get("confidence", 1.0))
    return (v or ""), 1.0


def run_model(model: str, groups: dict) -> dict:
    """Model pass -> {key: {text, spans:{role:{span,confidence}}, source_span}}."""
    from gliner2 import GLiNER2
    ex = GLiNER2.from_pretrained(model, map_location="cpu")
    schema = build_schema()
    raw = {}
    for key, g in sorted(groups.items()):
        rec = (ex.extract(g["text"], schema, include_confidence=True)
               .get("casualty_report") or [{}])[0]
        spans = {}
        for role in ROLES:
            span, conf = _cell(rec.get(role))
            spans[role] = {"span": span, "confidence": round(conf, 4)}
        raw[key] = {"text": g["text"], "spans": spans, "source_span": _cell(rec.get("source"))[0]}
    return raw


def normalize(text: str, spans: dict, source_span: str, thr: float):
    """Bound spans -> {role: {value, qualifier, confidence}}, source. Value from the span;
    qualifier from the number's local context; keep only digit spans clearing thr."""
    out = {}
    for role in ROLES:
        span, conf = spans[role]["span"], spans[role]["confidence"]
        if conf >= thr and any(c.isdigit() for c in span):
            v, _ = extract.value_qualifier(span)
            out[role] = {"value": v, "qualifier": extract.qualifier_near(text, span),
                         "confidence": conf}
    src = extract._detect_source(source_span) if source_span else None
    return out, src


def _load_groups(src_dir: Path, n_streams: int):
    groups = defaultdict(lambda: {"text": "", "gt": {}})
    for line in (src_dir / "observations.jsonl").open(encoding="utf-8"):
        o = json.loads(line)
        g = groups[(o["stream_id"], o["t_hours"])]
        g["text"] = o.get("text", ""); g["gt"][o["role"]] = o
    if n_streams:
        keep = sorted({k[0] for k in groups})[:n_streams]
        groups = {k: v for k, v in groups.items() if k[0] in keep}
    return dict(groups)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/disaster_streams_sonnet5")
    ap.add_argument("--split", default="val")
    ap.add_argument("--model", default="fastino/gliner2-base-v1")
    ap.add_argument("--out", default="datasets/disaster_streams_model")
    ap.add_argument("--streams", type=int, default=0, help="0 = all streams in the subset")
    ap.add_argument("--threshold", type=float, default=0.0, help="drop fills below this confidence")
    ap.add_argument("--from-raw", action="store_true", help="re-normalize raw.jsonl; skip the model")
    args = ap.parse_args(argv)

    src_dir = Path(args.data) / args.split
    out_dir = Path(args.out) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = _load_groups(src_dir, args.streams)
    raw_path = out_dir / "raw.jsonl"

    if args.from_raw:
        raw = {(r["stream_id"], r["t_hours"]): r for r in
               (json.loads(l) for l in raw_path.open(encoding="utf-8"))}
    else:
        raw = run_model(args.model, groups)
        with raw_path.open("w", encoding="utf-8") as f:
            for (sid, t), r in raw.items():
                f.write(json.dumps({"stream_id": sid, "t_hours": t, **r}, ensure_ascii=False) + "\n")

    tp = fp = fn = val_n = val_exact = qual_ok = 0
    rows = []
    for (sid, t), g in sorted(groups.items()):
        r = raw.get((sid, t))
        if r is None:
            continue
        pred, src = normalize(r["text"], r["spans"], r["source_span"], args.threshold)
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
                             "value": pred[role]["value"], "qualifier": pred[role]["qualifier"],
                             "source": src or "preliminary", "confidence": pred[role]["confidence"]})

    with (out_dir / "observations.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    sids = {k[0] for k in groups}
    with (out_dir / "trajectory.jsonl").open("w", encoding="utf-8") as tf:
        for line in (src_dir / "trajectory.jsonl").open(encoding="utf-8"):
            if json.loads(line)["stream_id"] in sids:
                tf.write(line)

    prec = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"[model_arm] {args.model}{' (from-raw)' if args.from_raw else ''}: {len(groups)} reports; "
          f"role P={prec:.3f} R={recall:.3f} (tp={tp} fp={fp} fn={fn}); "
          f"value exact={val_exact / max(val_n, 1):.3f} "
          f"qualifier acc={qual_ok / max(val_n, 1):.3f} (on {val_n} TP) -> {out_dir}")


if __name__ == "__main__":
    main()
