"""Build the 2-label relevance-gate corpus for a multilingual boundary fine-tune.

The gate asks one question: does this text REPORT A TOLL for a group of people?

Positives come from `casualty_events`, whose events carry `dead`/`injured`/`missing`
arguments with numeric entities -- an actual toll, which is the concept the gate needs.

Negatives are drawn from three pools and LENGTH-MATCHED to the positives:

  * `docee` records whose gold class is not a casualty type -- real news articles about
    sports, awards, finance. Long, English, and definitively not casualty reports.
  * `chfinann` and `cmnee` -- Chinese news, which carries the multilingual signal at
    news register rather than SMS register.
  * `disaster_response_messages` at `related=0`, using the `original` column where
    non-empty (French, Haitian Creole).

Length matching is not cosmetic. A first version drew negatives only from the SMS pool,
and a classifier using LENGTH ALONE scored 98.5% on it (positives median 1,104 chars,
negatives 93). Trained on that a model learns "long text = casualty report" and then
admits every long foreign news article -- which is precisely the failure being fixed
(the English-only gate admitted 199 of 200 clean Turkish articles).

Deliberately NOT used: that dataset's `death` / `missing_people` labels as positives.
They mean the message MENTIONS death, not that it reports a toll -- "the kids are
starving to death" carries `death=1`. Training positives on them would teach exactly
the confusion that makes the incumbent gate admit "cholera symptoms ... can lead to
death if untreated".
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import dumps_record  # noqa: E402

TASK = "relevance"
POSITIVE, NEGATIVE = "mass_casualty", "other"
LABELS = [POSITIVE, NEGATIVE]
TOLL_ROLES = {"dead", "injured", "missing", "killed"}
_NUM = re.compile(r"\d")


def _has_toll(record: dict) -> bool:
    """True when an event carries a casualty role whose entity contains a number."""
    for event in (record.get("output") or {}).get("events") or []:
        for arg in event.get("arguments") or []:
            if str(arg.get("role", "")).lower() in TOLL_ROLES and _NUM.search(
                str(arg.get("entity", ""))
            ):
                return True
    return False


def _row(text: str, label: str) -> dict:
    return {
        "input": text,
        "output": {"classifications": [
            {"task": TASK, "labels": list(LABELS), "true_label": [label]}
        ]},
    }


def positives(path: Path, limit: int) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            text = (rec.get("input") or "").strip()
            if text and _has_toll(rec):
                out.append(_row(text, POSITIVE))
            if len(out) >= limit:
                break
    return out


def _pool_from_jsonl(path: Path, exclude_casualty: bool, cap: int) -> list[tuple[str, str]]:
    """(text, source) pairs; optionally skip records whose gold class is a casualty type."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ekf_showcase"))
    from run_pipeline import DOCEE_CASUALTY_TYPES  # noqa: E402

    casualty = set(DOCEE_CASUALTY_TYPES)
    out = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if exclude_casualty:
                cls = ((rec.get("output") or {}).get("classifications") or [{}])[0]
                if (cls.get("true_label") or [None])[0] in casualty:
                    continue
            text = (rec.get("input") or "").strip()
            if text:
                out.append((text, path.stem.split(".")[0]))
            if len(out) >= cap:
                break
    return out


def _sms_pool(seed: int, cap: int) -> list[tuple[str, str]]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ekf_showcase"))
    from benchmark_gate import load  # noqa: E402

    df = load("train", 0, seed)
    out = []
    for _, r in df.iterrows():
        if int(r.get("related", 1)) != 0:
            continue
        original = str(r.get("original") or "").strip()
        message = str(r.get("message") or "").strip()
        text = original if len(original) > 3 else message
        if text:
            out.append((text, "sms_original" if len(original) > 3 else "sms_en"))
        if len(out) >= cap:
            break
    return out


def negatives(pos_lengths: list[int], seed: int) -> tuple[list[dict], dict]:
    """Length-MATCHED negatives, so length alone carries no signal."""
    pool: list[tuple[str, str]] = []
    pool += _pool_from_jsonl(Path("data/docee.train.jsonl"), True, 6000)
    pool += _pool_from_jsonl(Path("data/chfinann.train.jsonl"), False, 4000)
    pool += _pool_from_jsonl(Path("data/cmnee.train.jsonl"), False, 4000)
    pool += _sms_pool(seed, 4000)

    rng = random.Random(seed)
    rng.shuffle(pool)
    by_len = sorted(pool, key=lambda t: len(t[0]))
    lens = [len(t[0]) for t in by_len]
    used, out, sources = set(), [], {}
    import bisect
    for target in pos_lengths:
        i = bisect.bisect_left(lens, target)
        pick = None
        for off in range(0, len(by_len)):
            for j in (i - off, i + off):
                if 0 <= j < len(by_len) and j not in used:
                    pick = j
                    break
            if pick is not None:
                break
        if pick is None:
            break
        used.add(pick)
        text, src = by_len[pick]
        # trim only when the negative is far longer, so the match is on length not topic
        out.append(_row(text[:max(target, 200)], NEGATIVE))
        sources[src] = sources.get(src, 0) + 1
    return out, sources


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--casualty", default="data/casualty_events.train.jsonl")
    ap.add_argument("--out-prefix", default="data/gate2")
    ap.add_argument("--max-per-class", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    pos = positives(Path(args.casualty), args.max_per_class)
    pos_lengths = [len(r["input"]) for r in pos]
    neg, sources = negatives(pos_lengths, args.seed)
    print(f"[gate2] positives (toll reported) {len(pos)}")
    print(f"[gate2] negatives (length-matched) {len(neg)}  by source: {sources}")

    rows = pos + neg
    random.Random(args.seed).shuffle(rows)
    cut = int(len(rows) * (1 - args.val_frac))
    for split, part in (("train", rows[:cut]), ("val", rows[cut:])):
        path = Path(f"{args.out_prefix}.{split}.jsonl")
        with path.open("w", encoding="utf-8") as fh:
            for rec in part:
                fh.write(dumps_record(rec) + "\n")
        n_pos = sum(1 for r in part
                    if r["output"]["classifications"][0]["true_label"] == [POSITIVE])
        print(f"[gate2] {path}: {len(part)} rows ({n_pos} positive, "
              f"{len(part) - n_pos} negative)")


if __name__ == "__main__":
    main()
