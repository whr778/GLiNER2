"""Measure a corpus against a pretrained base and write the calibration behind a config.

    uv run python tools/derive/derive_config.py --corpus data/ace2005_v3 \\
        --model whr778/gliner2-eb18-balanced --out derived/

Writes, per (corpus, model):
  <name>.calibration.json   every measurement, with the model facts it was measured against
  <name>.labels_review.md   labels a human must confirm before training
See tools/derive/README.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_probe import probe_corpus, split_paths  # noqa: E402
from labels_file import build_labels, write_labels_file  # noqa: E402
from data_health import run_checks, write_report  # noqa: E402
from label_review import review  # noqa: E402
from model_probe import probe_model  # noqa: E402


def run_name(corpus: str, model: str) -> str:
    return f"{Path(corpus).name}__{Path(model.rstrip('/')).name}"


def write_review(path: Path, name: str, rows_by_cat: dict) -> None:
    lines = [f"# Label review: {name}", "",
             "Confirm every `fold_match` by reading the surfaces it tags; decide a natural-language",
             "name for every `open_vocab_code_like` label. `known`/`mapped` need no action.", ""]
    for cat, rows in rows_by_cat.items():
        if not rows:
            continue
        lines += [f"## {cat}  ({dict(Counter(r['status'] for r in rows))})", "",
                  "| label | uses | status | target |", "|---|---|---|---|"]
        lines += [f"| `{r['label']}` | {r['uses']} | {r['status']} | {r.get('target', '')} |" for r in rows]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def eval_settings(model_info: dict) -> dict:
    """Window, global_decode and default threshold, each with where it came from."""
    d = model_info.get("inference_defaults") or {}
    span = model_info["architecture"] == "span"
    window = d.get("chunk_size") or (min(model_info["max_len"], 512) - 128 if span else min(model_info["max_len"], 4096))
    return {"window": window, "global_decode": d.get("global_decode", True),
            "threshold": d.get("threshold", 0.5),
            "source": "checkpoint inference_defaults" if d else
                      "default: window=max_len-128 (span) or min(max_len,4096); global_decode on; threshold 0.5"}


def run_gpu_stages(args, model_info: dict, corpus: dict, event_records: bool) -> dict:
    import gpu_stages as G
    from gliner2.training.eval_metrics import load_with_overrides
    ev = eval_settings(model_info)
    boundary = model_info["architecture"] == "boundary"
    pool = model_info["boundary_head"].get("candidate_pool", "per_query")
    cap = corpus["gold_capacity"]["recommended_cap"]
    out = {"eval_settings": ev}
    if boundary:
        out["reachability"] = G.reachability(args.model, args.corpus, model_info["label_map"], event_records,
                                             cap, ev["window"], (16, 32, 64, 128), args.out, args.reach_records,
                                             args.device, pool)
    records = G.val_records(args.corpus, model_info["label_map"], args.max_val)
    model = load_with_overrides(args.model).to(args.device).eval()
    has_records = boundary and (event_records or corpus["heads"]["val"]["structure_records"] > 0)
    out.update(G.operating_points(model, records, ev["window"], ev["global_decode"], ev["threshold"],
                                  has_records, args.batch_size, pool))
    out["val_records_used"] = len(records)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=True, help="split base: <base>.{train,val,test}.jsonl")
    ap.add_argument("--model", required=True, help="Hub id or local checkpoint dir")
    ap.add_argument("--out", type=Path, default=Path("derived"))
    ap.add_argument("--gpu", action="store_true", help="also run stages 4-6 (reachability, baseline, sweeps)")
    ap.add_argument("--device", default="cuda", help="device for stages 5-6")
    ap.add_argument("--max-val", type=int, default=0, help="cap validation records (0 = all)")
    ap.add_argument("--reach-records", type=int, default=40, help="val records for the reachability probe")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--siblings", nargs="*", default=[],
                    help="extra split bases to check for shared documents (same-directory corpora are always checked)")
    ap.add_argument("--align-records", type=int, default=2000, help="records per split for the alignment check (0 = all)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    model_info = probe_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer_source"])
    event_records = bool(model_info["boundary_head"].get("event_records"))
    corpus = probe_corpus(args.corpus, tokenizer, event_records)
    labels = review(corpus["labels"], model_info)

    name = run_name(args.corpus, args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    calib = {"model": {k: v for k, v in model_info.items() if k not in ("label_map", "boundary_head")},
             "corpus": corpus, "label_review": labels}
    health = run_checks(args.corpus, split_paths(args.corpus), corpus, args.model,
                        eval_settings(model_info)["window"], args.siblings, args.align_records)
    calib["data_health"] = health
    write_report(args.out / f"{name}.data_health.md", name, health)
    if args.gpu:
        calib["gpu"] = run_gpu_stages(args, model_info, corpus, event_records)
    (args.out / f"{name}.calibration.json").write_text(
        json.dumps(calib, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    write_review(args.out / f"{name}.labels_review.md", name, labels)
    block, proposals = build_labels([str(p) for p in split_paths(args.corpus).values()], model_info, labels)
    write_labels_file(args.out / f"{name}.labels.yaml", name, args.model, block, proposals)

    u = corpus["uniqueness"]
    dups = {s: u[f"{s}_records"] - u[f"{s}_unique_inputs"] for s in ("train", "val", "test")}
    overlap = {k: u[k] for k in ("train&val", "train&test", "val&test")}
    print(f"[derive] {name}: arch={model_info['architecture']} encoder={model_info['encoder']} "
          f"max_len={model_info['max_len']}")
    print(f"[derive] split hygiene: duplicates {dups} overlap {overlap}"
          + ("  *** NOT CLEAN ***" if any(dups.values()) or any(overlap.values()) else "  clean"))
    print(f"[derive] gold capacity: {corpus['gold_capacity']}")
    sev = Counter(f["severity"] for f in health["findings"])
    print(f"[derive] data health: {dict(sev) or 'no findings'}"
          + ("  *** BLOCKED -- DO NOT TRAIN; see " + name + ".data_health.md ***" if health["blocked"] else ""))
    m = health["measured"]
    tv = {k: v for k, v in m.get("tvd", {}).items() if v is not None}
    print(f"[derive]   measured: max TVD {max(tv.values()) if tv else None} over {len(tv)} split pairs; "
          f"unaligned % {m.get('unaligned_pct')}; siblings checked {len(m['siblings_checked'])}")
    for f in health["findings"]:
        if f["severity"] != "INFO":
            print(f"  {f['severity']:5} {f['check']}: {f['message']}")
            if f["fix"]:
                print(f"        fix: {f['fix']}")
    n_map = {c: len(b["map"]) for c, b in block.items()}
    print(f"[derive] labels_file: map entries {n_map}, proposals awaiting review "
          f"{ {c: len(v) for c, v in proposals.items()} }")
    print(f"[derive] wrote {args.out / (name + '.calibration.json')}, {name}.labels_review.md, {name}.labels.yaml")


if __name__ == "__main__":
    main()
