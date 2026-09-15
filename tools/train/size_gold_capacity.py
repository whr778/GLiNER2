"""Size `max_gold_per_query` per config, from that config's OWN training corpora.

WHY EACH CONFIG NEEDS ITS OWN NUMBER. `max_gold_per_query` defaults to 32, and a config
running `on_capacity_exceeded: skip_sample` does not fail when a query exceeds it -- it
clears gold for EVERY query in that document, and an empty mention_mask is the POSITIVE
target for abstention (`abstention_loss` trains on `~mention_mask.any(-1)`). So the model is
taught to emit nothing on the documents carrying the most gold, silently. A config training
on sentence-level classification never comes near 32; one training on CASIE carries 188 gold
spans in a single query. One global number is wrong in both directions.

WHAT IS MEASURED, and it is an APPROXIMATION -- stated because the number gets acted on.
A "query group" here is a (document, label) pair, counted as the number of distinct surface
occurrences the label's gold spans would produce:

  * entities            -> one group per entity label
  * events              -> one group per (event_type, role), plus one per event_type for
                           triggers, but ONLY when the config does not set `event_records`
                           (with it, events are supervised through the record head, which
                           `skip_sample` does not touch -- see tests/processing/
                           test_gold_capacity.py)

It counts `text.count(surface)`, which is what the span builder would find, so it can
OVERCOUNT where a short surface recurs incidentally. That biases toward a larger cap, which
is the safe direction: an oversized cap costs a little padding memory, an undersized one
silently teaches abstention.

    report : uv run python tools/train/size_gold_capacity.py tools/train/config/base/*.yaml
    apply  : uv run python tools/train/size_gold_capacity.py --apply <configs...>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

LADDER = (128, 256, 384, 512)
SAMPLE = 3000          # documents per corpus; the tail is what matters, not the mean


def corpora_for(cfg: dict) -> list[Path]:
    data = cfg.get("data") or {}
    paths: list[Path] = []
    for prefix in data.get("corpora") or []:
        paths.append(Path(f"{prefix}.train.jsonl"))
    for spec in (data.get("event_files") or {}).values():
        if isinstance(spec, dict) and spec.get("train"):
            paths.append(Path(spec["train"]))
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


def measure(paths: list[Path], count_events: bool) -> tuple[Counter, int]:
    counts: Counter = Counter()
    missing = 0
    for p in paths:
        if not p.is_file():
            missing += 1
            continue
        n = 0
        with p.open(encoding="utf-8") as f:
            for line in f:
                if n >= SAMPLE:
                    break
                if not line.strip():
                    continue
                n += 1
                rec = json.loads(line)
                text = rec.get("input")
                if not isinstance(text, str):
                    continue
                out = rec.get("output") or {}
                groups: Counter = Counter()
                ents = out.get("entities")
                if isinstance(ents, dict):
                    for label, surfaces in ents.items():
                        if isinstance(surfaces, list):
                            groups[("ent", label)] += sum(
                                max(1, text.count(s))
                                for s in surfaces if isinstance(s, str) and s)
                if count_events:
                    for ev in out.get("events") or []:
                        if not isinstance(ev, dict):
                            continue
                        et = ev.get("event_type")
                        for t in ev.get("triggers") or []:
                            if isinstance(t, str) and t:
                                groups[("trig", et)] += max(1, text.count(t))
                        for a in ev.get("arguments") or []:
                            if isinstance(a, dict) and isinstance(a.get("entity"), str):
                                groups[("arg", et, a.get("role"))] += max(
                                    1, text.count(a["entity"]))
                for v in groups.values():
                    if v:
                        counts[v] += 1
    return counts, missing


def recommend(counts: Counter) -> tuple[int | None, float, int]:
    """(cap or None if 32 suffices, % over 32, largest group)."""
    total = sum(counts.values())
    if not total:
        return None, 0.0, 0
    over32 = sum(v for k, v in counts.items() if k > 32)
    biggest = max(counts)
    if not over32:
        return None, 0.0, biggest
    # smallest ladder rung leaving <= 0.02% overflowing
    for cap in LADDER:
        if sum(v for k, v in counts.items() if k > cap) / total <= 0.0002:
            return cap, 100.0 * over32 / total, biggest
    return LADDER[-1], 100.0 * over32 / total, biggest


def _child_indent(lines: list[str], idx: int, fallback: int) -> int:
    """Indent of the block's children, read from the next non-blank line."""
    own = len(lines[idx]) - len(lines[idx].lstrip())
    for nxt in lines[idx + 1:]:
        if nxt.strip():
            got = len(nxt) - len(nxt.lstrip())
            if got > own:
                return got
            break
    return own + fallback


def apply_to(path: Path, cap: int, budget: int, pct: float, biggest: int) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    idx = next((i for i, l in enumerate(lines) if l.strip() == "boundary_head:"), None)
    if idx is None:
        # Many configs never declare boundary_head at all and run entirely on defaults --
        # which is exactly how the 32 cap reached them. Create the block under `model:`.
        midx = next(i for i, l in enumerate(lines) if l.rstrip("\n") == "model:")
        mchild = _child_indent(lines, midx, 2)
        lines.insert(midx + 1, " " * mchild + "boundary_head:\n")
        idx = midx + 1
    child = _child_indent(lines, idx, 2)
    note = [
        "# Sized by tools/train/size_gold_capacity.py from THIS config's own corpora.",
        f"# {pct:.3f}% of (doc, label) query groups exceed the default 32; largest seen {biggest}.",
        "# The default matters because `on_capacity_exceeded: skip_sample` does not fail on",
        "# overflow -- it clears gold for EVERY query in the document, and an empty",
        "# mention_mask is the POSITIVE target for abstention, so the model is taught to emit",
        "# nothing on its richest documents. See tests/processing/test_gold_capacity.py.",
        f"max_gold_per_query: {cap}",
        f"training_candidate_budget: {budget}",
    ]
    block = [" " * child + l + "\n" for l in note]
    path.write_text("".join(lines[:idx + 1] + block + lines[idx + 1:]), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    print(f"{'config':52s}{'groups':>9}{'>32':>9}{'largest':>9}{'cap':>7}  action")
    changed = 0
    for spec in sorted(args.configs):
        path = Path(spec)
        if "archive" in path.parts:
            print(f"{path.name:52s}{'':>9}{'':>9}{'':>9}{'':>7}  SKIP (archived run record)")
            continue
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        bh = ((cfg.get("model") or {}).get("boundary_head") or {})
        if "max_gold_per_query" in bh:
            print(f"{path.name:52s}{'':>9}{'':>9}{'':>9}{bh['max_gold_per_query']:>7}  already set")
            continue
        if ((cfg.get("training") or {}).get("on_capacity_exceeded")) != "skip_sample":
            print(f"{path.name:52s}{'':>9}{'':>9}{'':>9}{'':>7}  SKIP (not skip_sample)")
            continue
        count_events = not bh.get("event_records")
        counts, missing = measure(corpora_for(cfg), count_events)
        total = sum(counts.values())
        if not total:
            print(f"{path.name:52s}{0:>9}{'':>9}{'':>9}{'':>7}  "
                  f"SKIP (no measurable gold{'; ' + str(missing) + ' files absent' if missing else ''})")
            continue
        cap, pct, biggest = recommend(counts)
        if cap is None:
            print(f"{path.name:52s}{total:>9,}{0:>9}{biggest:>9}{'':>7}  ok at 32")
            continue
        budget = max(160, cap + 128)
        action = "would set"
        if args.apply:
            # Write FIRST, report after. Printing "APPLIED" before the write made five
            # configs report success and then raise on the very next line.
            apply_to(path, cap, budget, pct, biggest)
            changed += 1
            action = "APPLIED"
        print(f"{path.name:52s}{total:>9,}{pct:>8.3f}%{biggest:>9}{cap:>7}  {action} "
              f"(budget {budget})")
    if args.apply:
        print(f"\n{changed} config(s) changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
