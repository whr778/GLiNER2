"""Build the 2-label relevance-gate corpus. ALL TEXT IS REAL.

The gate asks one question: does this text REPORT A CURRENT TOLL for a group of people?

## Why the previous version was thrown away

It drew positives from `casualty_events`, which is SYNTHETIC -- 99.9% of those documents
are dated 2026 and 86.5% carry generation templates ("A major news outlet reported...").
Every negative was real. So the two classes were separable on PROVENANCE, the trained
gate learned "generated disaster template -> mass_casualty", and it then rejected all 590
benchmark messages and all 71 Aegean news articles while scoring F1 1.0000 on its own
test split. Length, sentence-ending punctuation and CJK script were all downstream
symptoms of that one split; fixing them one at a time never had a chance.

Two label traps found the same way, by reading samples rather than trusting counts:

  * `casualty_events` records with no toll ARGUMENT still report tolls in their text
    ("27 people died and 159 were injured") -- they annotate `location` alone. They are
    not negatives.
  * `cmnee` `Injure` marks a casualty MENTIONED anywhere in a document, so ship-naming
    and procurement articles carrying a historical toll are labelled Injure (~45%
    precise). Dropped. `duee` disaster events verified clean 16/16 and ARE used.

## How this one is built

Positives and negatives both come from real news, adjudicated by
`tools/data/annotate_gate.py` -- because a regex cannot tell a current toll from
"220,000 victims have been served meals" or "in 1999 ... deaths of over 17,000", which
are the measured false positives of the shipped gate.

Balance is by CONSTRUCTION, not by patching. Within every (source, length-decile) cell
the two classes are equalised, so neither source nor length carries any signal. Negatives
prefer the ADJUDICATED hard cases (historical / exposure) over cue-free filler, and
cue-free filler is capped, so "contains a casualty word" cannot be the rule either.

Verify before training, and before booking a GPU:

    uv run python tools/data/build_gate_corpus.py
    uv run python tools/data/check_gate_corpus.py data/gate2
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _split import SplitWriter, normalize_group_key  # noqa: E402
from annotate_gate import CUE, MAX_CHARS  # noqa: E402

TASK = "relevance"
POSITIVE, NEGATIVE = "mass_casualty", "other"
LABELS = [POSITIVE, NEGATIVE]

# The SECOND task, and it is not decoration. Measured on the first model, the binary
# corpus taught "does this mention a toll" (93.9% on positives) and NOT the two
# distinctions the adjudication was bought for: exposure_only scored 0.250 and
# historical_toll 0.471, i.e. the "220,000 served meals" and "1999 toll of 17,000"
# failures survived training. Supervising the four-way label directly makes that
# distinction an objective instead of hoping the binary boundary captures it.
#
# It also fixes a second problem for free. A model trained with exactly ONE
# classification task per record is off-distribution as soon as a schema carries
# another one -- relevance degrades from 0.999 to 0.862 with a second task and
# collapses with a third. Two tasks per record, in RANDOMISED order, is the
# augmentation that removes that brittleness.
KIND_TASK = "toll_kind"
KIND_LABELS = ["current_toll", "historical_toll", "exposure_only", "no_toll"]

# Rows that never went to the annotator still have a sound four-way label.
KIND_FROM_SOURCE = {
    "duee_toll": "current_toll",
    "duee_other": "no_toll",
    "cue_free": "no_toll",
}

# Which negatives fill a cell's quota. exposure and historical are the ones the model
# actually fails, and they are scarce (691 and 1,686), so they go first and cue-free
# filler goes last. This changes the MIX without touching the per-cell balance that
# keeps source and length uninformative.
NEGATIVE_PRIORITY = {"exposure_only": 0, "historical_toll": 1, "duee_other": 2,
                     "no_toll": 3, "cue_free": 4}

# Cue-free filler is real and free, but if the negative class were mostly filler the
# model could pass by keyword-matching "died". Capped as a FRACTION of negatives.
FILLER_FRAC = 0.25

CN_NUM = re.compile(r"[0-9０-９一二两三四五六七八九十百千万余多]")
DUEE_TOLL = {"死亡人数", "受伤人数", "失踪人数"}


def _row(text: str, label: str, kind: str, rng: random.Random) -> dict:
    """Both tasks, in randomised order so neither position is the trained one."""
    tasks = [
        {"task": TASK, "labels": list(LABELS), "true_label": [label]},
        {"task": KIND_TASK, "labels": list(KIND_LABELS), "true_label": [kind]},
    ]
    rng.shuffle(tasks)
    return {"input": text, "output": {"classifications": tasks}}


def _source(name: str) -> str:
    """Both cc_news pulls are one distribution; splitting them only thins the cells."""
    return "cc_news" if name.startswith("cc_news") else name


def load_annotated(path: Path) -> list[dict]:
    rows = []
    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        label = rec["label"]
        rows.append({
            "text": rec["input"],
            "positive": label == "current_toll",
            "source": _source(rec["source"]),
            "kind": label,
        })
    return rows


def load_filler(paths: list[Path], seen: set, cap: int, seed: int) -> list[dict]:
    """Cue-free real documents. No casualty word at all, so definitionally no toll."""
    out = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            text = (json.loads(line).get("input") or "").strip()[:MAX_CHARS]
            if not text or CUE.search(text):
                continue
            key = normalize_group_key(text)[:300]
            if key in seen:
                continue
            seen.add(key)
            out.append({"text": text, "positive": False,
                        "source": _source(path.stem.split(".")[0]), "kind": "cue_free"})
    random.Random(seed).shuffle(out)
    return out[:cap]


def load_duee() -> list[dict]:
    """The Chinese stratum: same corpus, same register, both classes."""
    out = []
    for split in ("train", "val"):
        path = Path(f"data/duee.{split}.jsonl")
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            text = (rec.get("input") or "").strip()
            if not text:
                continue
            events = (rec.get("output") or {}).get("events") or []
            toll = any(a.get("role") in DUEE_TOLL and CN_NUM.search(str(a.get("entity", "")))
                       for ev in events for a in ev.get("arguments") or [])
            disaster = any(str(ev.get("event_type", "")).startswith("灾害/意外")
                           for ev in events)
            if toll:
                out.append({"text": text, "positive": True, "source": "duee",
                            "kind": "duee_toll"})
            elif not disaster:
                out.append({"text": text, "positive": False, "source": "duee",
                            "kind": "duee_other"})
    return out


def balance(rows: list[dict], seed: int, deciles: int = 10) -> list[dict]:
    """Equalise the classes inside every (source, length-decile) cell.

    This is the whole design. Matching on aggregate statistics leaves the model free to
    use source or length WITHIN a stratum; equalising per cell means P(positive | source,
    length band) is 0.5 by construction, so neither can carry signal at all.
    """
    rng = random.Random(seed)
    pos = [r for r in rows if r["positive"]]
    if not pos:
        return []
    edges = _decile_edges([len(r["text"]) for r in pos], deciles)

    cells: dict[tuple, dict[bool, list]] = defaultdict(lambda: {True: [], False: []})
    for row in rows:
        cell = (row["source"], _bucket(len(row["text"]), edges))
        cells[cell][row["positive"]].append(row)

    out, dropped = [], Counter()
    for cell, sides in sorted(cells.items(), key=lambda kv: str(kv[0])):
        take = min(len(sides[True]), len(sides[False]))
        if not take:
            dropped[cell[0]] += len(sides[True]) + len(sides[False])
            continue
        rng.shuffle(sides[True])
        # Scarce, hard, adjudicated negatives first; cue-free filler last.
        sides[False].sort(key=lambda r: (NEGATIVE_PRIORITY.get(r["kind"], 9), rng.random()))
        out += sides[True][:take] + sides[False][:take]
    if dropped:
        print(f"[gate2] cells with no counterpart, dropped: {dict(dropped)}")
    rng.shuffle(out)
    return out


def enforce_shares(rows: list[dict], targets: dict[str, float], seed: int) -> list[dict]:
    """Downsample sources so each hits its TARGET share of the corpus.

    `balance` equalises the classes inside every (source, length-decile) cell, which fixes
    P(positive | source, length) at 0.5 -- but it says nothing about how much of the corpus
    each source IS. That gap has now cost two models, in opposite directions:

        gate2_tr  EN 74.8%  TR 22.0%  ZH  3.1%   Chinese admission 97% -> 83%
        gate3     EN 65.5%  TR 19.3%  ZH 15.2%   Chinese repaired, ENGLISH regressed
                                                 (Helene pooled RMSE 132.6 -> 175.7)

    Adding a language silently taxes the others, and being three-way does not escape it --
    a fourth would hit it again. So the shares become an input, chosen and asserted, rather
    than whatever the pools happened to contain.

    The positive/negative balance `balance` established is preserved: each source is cut to
    an equal number of positives and negatives. Sources with no target are passed through.
    """
    rng = random.Random(seed)
    by_source: dict[str, dict[bool, list]] = defaultdict(lambda: {True: [], False: []})
    for row in rows:
        by_source[row["source"]][row["positive"]].append(row)

    # The binding source is the one furthest short of its target; scale everything to it.
    total = min(
        (min(len(sides[True]), len(sides[False])) * 2) / targets[src]
        for src, sides in by_source.items() if src in targets and targets[src] > 0
    )
    out: list[dict] = []
    for src, sides in sorted(by_source.items()):
        if src not in targets:
            out += sides[True] + sides[False]
            continue
        per_class = int(total * targets[src] / 2)
        for positive in (True, False):
            pool = sides[positive]
            rng.shuffle(pool)
            out += pool[:per_class]
    rng.shuffle(out)

    got = Counter(r["source"] for r in out)
    n = len(out)
    print(f"[gate2] shares enforced -> {n} rows")
    for src in sorted(got):
        want = targets.get(src)
        note = f"  (target {want:.1%})" if want else "  (untargeted)"
        print(f"[gate2]   {src:16s} {got[src]:6d}  {got[src]/n:6.1%}{note}")
    return out


def _decile_edges(lengths: list[int], deciles: int) -> list[int]:
    ordered = sorted(lengths)
    return [ordered[int(len(ordered) * i / deciles)] for i in range(1, deciles)]


def _bucket(length: int, edges: list[int]) -> int:
    for i, edge in enumerate(edges):
        if length < edge:
            return i
    return len(edges)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # Several adjudicated pools can be combined. Each keeps its own `source`, which is
    # what makes adding a language safe: `balance` equalises the classes inside every
    # (source, length-decile) cell, so Turkish rows form their own cells and
    # P(positive | source=turkish_news, length band) is 0.5 by construction. The model
    # therefore cannot learn "Turkish => positive", which is the failure that would
    # otherwise make a monolingual-source corpus worse than useless.
    ap.add_argument("--annotated", nargs="+", default=["data/gate_ann.jsonl"])
    ap.add_argument("--out-prefix", default="data/gate2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source-share", nargs="+", metavar="SOURCE=FRACTION",
                    help="fix each source's share of the corpus BY CONSTRUCTION, e.g. "
                         "'docee=0.55 turkish_news=0.22 zh_news=0.15 cc_news=0.08'. Must "
                         "sum to 1.0. Without this the shares are whatever the pools "
                         "contain, which has now regressed two models in opposite "
                         "directions -- see enforce_shares().")
    args = ap.parse_args()

    rows = []
    for path in args.annotated:
        pool = load_annotated(Path(path))
        print(f"[gate2] adjudicated {len(pool):6d} from {path}: "
              f"{dict(Counter(r['kind'] for r in pool))}")
        rows += pool

    rows += load_duee()
    n_pos = sum(r["positive"] for r in rows)
    seen = {normalize_group_key(r["text"])[:300] for r in rows}
    filler_cap = int(n_pos * FILLER_FRAC / (1 - FILLER_FRAC))
    rows += load_filler([Path("data/docee.train.jsonl"),
                         Path("data/cc_news_parts/cc_news_10k_raw.jsonl"),
                         Path("data/cc_news_parts/cc_news_10k_b_raw.jsonl")],
                        seen, filler_cap, args.seed)

    final = balance(rows, args.seed)
    if args.source_share:
        targets = {}
        for spec in args.source_share:
            name, _, frac = spec.partition("=")
            targets[name] = float(frac)
        total_share = sum(targets.values())
        if abs(total_share - 1.0) > 1e-6:
            raise SystemExit(f"--source-share must sum to 1.0, got {total_share}")
        final = enforce_shares(final, targets, args.seed)
    kinds = Counter(r["kind"] for r in final)
    n_pos = sum(r["positive"] for r in final)
    print(f"[gate2] balanced {len(final)}: {n_pos} positive, {len(final) - n_pos} negative")
    print(f"[gate2]   by kind  : {dict(kinds)}")
    print(f"[gate2]   by source: {dict(Counter(r['source'] for r in final))}")

    # Grouped on the normalized LEAD, the same key the near-dup dedup uses, so a
    # syndicated retelling cannot land on the far side of the split from its twin.
    order_rng = random.Random(args.seed)
    with SplitWriter(Path(args.out_prefix), seed=args.seed) as writer:
        for row in final:
            kind = KIND_FROM_SOURCE.get(row["kind"], row["kind"])
            writer.write(_row(row["text"], POSITIVE if row["positive"] else NEGATIVE,
                              kind, order_rng),
                         group=normalize_group_key(row["text"])[:300])
    print(f"[gate2] {writer.summary()}")


if __name__ == "__main__":
    main()
