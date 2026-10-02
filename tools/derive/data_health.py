"""Data-health checks run while a config is derived: each finding carries its severity, the
evidence, and the exact command that fixes it.

  BLOCK  the split cannot produce a trustworthy measurement (val/test contaminated by train,
         duplicates inside the blind test). The YAML writer refuses to emit a config.
  WARN   the numbers will be biased or noisy (label shift between splits, unseen labels,
         thin support, gold that never aligns, train-side duplicates, sibling overlap).
  INFO   worth knowing (no-gold share, documents longer than the window).

Every check composes an existing tool so a finding here reads the same there:
`_split.normalize_group_key` (the key dedupe_splits/check_leakage use), the TVD of
compare_label_distributions, and the real tokenizer alignment of measure_surface_alignment.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools/data")]
from _split import normalize_group_key  # noqa: E402
from compare_label_distributions import tvd  # noqa: E402

SPLITS = ("train", "val", "test")
TVD_WARN, TVD_STRONG = 0.15, 0.30
THIN_SUPPORT = 10
UNALIGNED_WARN = 0.01
LONG_DOC_INFO = 0.05


def _finding(sev: str, check: str, msg: str, evidence=None, fix: str = "") -> Dict:
    return {"severity": sev, "check": check, "message": msg, "evidence": evidence, "fix": fix}


def _keys(path: Path) -> List[str]:
    return [normalize_group_key(json.loads(l)["input"]) for l in open(path, encoding="utf-8") if l.strip()]


def contamination(base: str, paths: Dict[str, Path]) -> List[Dict]:
    keys = {s: _keys(p) for s, p in paths.items()}
    out = []
    fix = f"uv run python tools/data/dedupe_splits.py {base} --dry-run   # then without --dry-run (test > val > train)"
    for s, k in keys.items():
        dups = len(k) - len(set(k))
        if dups:
            sev = "WARN" if s == "train" else "BLOCK"
            out.append(_finding(sev, "duplicates", f"{dups} duplicate input(s) inside {s}", {"split": s, "count": dups}, fix))
    sets = {s: set(k) for s, k in keys.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        n = len(sets[a] & sets[b])
        if n:
            out.append(_finding("BLOCK", "cross_split", f"{n} document(s) in both {a} and {b}", {"pair": f"{a}&{b}", "count": n}, fix))
    return out


def siblings(base: str, extra: List[str]) -> List[str]:
    """Other split corpora in the same directory (e.g. the other ACE permutations), plus extras."""
    me = Path(base)
    found = {str(p)[: -len(".train.jsonl")] for p in me.parent.glob("*.train.jsonl")}
    found |= set(extra or [])
    found.discard(str(me))
    return sorted(b for b in found if all(Path(f"{b}.{s}.jsonl").exists() for s in SPLITS))


def cross_corpus(base: str, paths: Dict[str, Path], others: List[str]) -> List[Dict]:
    mine = {s: set(_keys(p)) for s, p in paths.items()}
    out = []
    for other in others:
        theirs = {s: set(_keys(Path(f"{other}.{s}.jsonl"))) for s in SPLITS}
        ev = {"our_eval_in_their_train": len((mine["val"] | mine["test"]) & theirs["train"]),
              "our_train_in_their_eval": len(mine["train"] & (theirs["val"] | theirs["test"]))}
        if any(ev.values()):
            out.append(_finding(
                "WARN", "cross_corpus",
                f"shares documents with {Path(other).name}: never train on both, and do not compare "
                f"a model trained on one against the other's test", {"other": other, **ev},
                f"uv run python tools/data/check_leakage.py --pattern '{Path(base).parent}/*.jsonl' --focus {Path(base).name}"))
    return out


def _flat(d: Dict) -> Counter:
    """Classification labels are nested (task -> label -> n); flatten to `task:label`."""
    out = Counter()
    for k, v in d.items():
        if isinstance(v, dict):
            out.update({f"{k}:{lab}": n for lab, n in v.items()})
        else:
            out[k] += v
    return out


def label_shift(labels_by_split: Dict[str, Dict[str, Dict[str, int]]], measured: Dict) -> List[Dict]:
    out = []
    cats = labels_by_split["train"].keys()
    for cat in cats:
        c = {s: _flat(labels_by_split[s].get(cat) or {}) for s in SPLITS}
        if not sum(c["train"].values()):
            continue
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            d = tvd(c[a], c[b])
            measured.setdefault("tvd", {})[f"{cat}:{a}/{b}"] = round(d, 4) if d == d else None
            if d == d and d > TVD_WARN:
                pa, pb = sum(c[a].values()), sum(c[b].values())
                drivers = sorted(set(c[a]) | set(c[b]), key=lambda k: -abs(c[a][k] / pa - c[b][k] / pb))[:5]
                out.append(_finding(
                    "WARN", "label_shift",
                    f"{cat}: TVD {d:.3f} between {a} and {b}" + (" (STRONG)" if d > TVD_STRONG else ""),
                    {"category": cat, "pair": f"{a}/{b}", "tvd": round(d, 4), "drivers": drivers},
                    "re-split with the multi-label stratifier (tools/data/_stratify.py, used by "
                    "convert_ace2005.py's default 80/10/10) instead of a random or document-order split"))
        for s in ("val", "test"):
            unseen = {k: v for k, v in c[s].items() if k not in c["train"]}
            if unseen:
                out.append(_finding("WARN", "unseen_labels",
                                    f"{cat}: {len(unseen)} label(s) in {s} never seen in train -- scored zero-shot",
                                    {"category": cat, "split": s, "labels": dict(sorted(unseen.items(), key=lambda x: -x[1])[:10])},
                                    "re-split so every label has train support, or report these labels separately"))
            thin = {k: v for k, v in c[s].items() if v < THIN_SUPPORT}
            if thin:
                out.append(_finding("WARN", "thin_support",
                                    f"{cat}: {len(thin)} label(s) with < {THIN_SUPPORT} gold in {s} -- their per-label F1 is noise",
                                    {"category": cat, "split": s, "labels": dict(sorted(thin.items(), key=lambda x: x[1])[:10])},
                                    "read micro F1 only for these, or merge rare labels in the labels_file"))
    return out


def alignment(model: str, paths: Dict[str, Path], max_records: int, measured: Dict) -> List[Dict]:
    from gliner2 import AutoExtractor
    from measure_surface_alignment import Aligner, scan
    aligner = Aligner(AutoExtractor.from_pretrained(model, map_location="cpu").processor)
    out = []
    for s, p in paths.items():
        total, counts, _per_label, per_dim, examples = scan(str(p), aligner, max_records)
        bad = sum(counts.values())
        measured.setdefault("unaligned_pct", {})[s] = round(100 * bad / total, 3) if total else None
        if total and bad / total > UNALIGNED_WARN:
            out.append(_finding(
                "WARN", "unaligned_gold",
                f"{s}: {bad:,} of {total:,} gold surfaces ({100 * bad / total:.2f}%) never align to the "
                f"tokenized text -- silently dropped from training and unscorable",
                {"split": s, "by_cause": dict(counts), "examples": examples},
                f"uv run python tools/data/measure_surface_alignment.py --checkpoint {model} --files {p}"))
    return out


def shape(heads: Dict, lengths: Dict, window: int) -> List[Dict]:
    out = []
    for s in SPLITS:
        h = heads[s]
        gold = h["entity_mentions"] + h["relations"] + h["event_instances"] + h["structure_records"]
        if not gold:
            out.append(_finding("INFO", "no_gold", f"{s} carries no gold at all", {"split": s}))
    if window and lengths["train"]["p90"] > window:
        out.append(_finding("INFO", "long_documents",
                            f"train p90 length {lengths['train']['p90']} subwords exceeds the {window} window: "
                            f"documents will be windowed", {"p90": lengths["train"]["p90"], "window": window}))
    return out


def run_checks(base: str, paths: Dict[str, Path], corpus: Dict, model: str, window: int,
               extra_siblings: List[str], align_records: int) -> Dict:
    # Every check also records what it MEASURED, so a check that found nothing is visibly a
    # check that ran -- a findings list alone cannot tell "clean" from "never ran".
    sibs = siblings(base, extra_siblings)
    measured: Dict = {"siblings_checked": sibs}
    findings = contamination(base, paths)
    findings += cross_corpus(base, paths, sibs)
    findings += label_shift(corpus["labels_by_split"], measured)
    findings += alignment(model, paths, align_records, measured)
    findings += shape(corpus["heads"], corpus["lengths"], window)
    order = {"BLOCK": 0, "WARN": 1, "INFO": 2}
    findings.sort(key=lambda f: order[f["severity"]])
    return {"blocked": any(f["severity"] == "BLOCK" for f in findings), "findings": findings,
            "measured": measured}


def write_report(path: Path, name: str, health: Dict) -> None:
    lines = [f"# Data health: {name}", "",
             "**BLOCKED -- do not train on this split until the BLOCK findings are fixed.**" if health["blocked"]
             else "No blocking findings.", ""]
    for f in health["findings"]:
        lines += [f"## {f['severity']} -- {f['check']}", "", f["message"], ""]
        if f["evidence"]:
            lines += ["```json", json.dumps(f["evidence"], ensure_ascii=False, indent=1, default=str)[:1500], "```", ""]
        if f["fix"]:
            lines += ["Fix:", "", "```bash", f["fix"], "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
