"""Score a candidate EKF front-end model against the gates fixed in its config.

The gates are pre-registered in `tools/train/config/ekf-frontend-mmbert.yaml` and are
deliberately on AP wire copy, not on held-out DocEE -- the incumbent scores well on DocEE
(trigger 0.710 / argument 0.506) while emitting nonsense on Helene, so held-out corpus
metrics do not predict what this model is for.

    uv run python tools/ekf_showcase/frontend_gates.py <model-id> [--device cpu]

GATE 1  usable events-form on the Helene feed: a trigger AND >=1 bound argument on >= 50%
        of casualty-bearing windows, at a swept threshold -- the bar for "the router has
        an input at all".

        CORRECTED 2026-08-23. This used to read "the incumbent is ~0 at every threshold".
        That was an artifact of this harness: it set only the Schema's per-event
        thresholds, which a boundary model never reads, so every sweep row actually ran
        at the default 0.5 -- where 0.0% is indeed what the incumbent scores. Measured
        properly it is 0.0% at 0.5/0.4, 8.3% at 0.3, 20.0% at 0.2 and 65.0% at 0.1, so
        the incumbent PASSES this gate on the registered range. Gate 1 measures FORM,
        not content: at 0.1 the incumbent's bindings are the documented nonsense, which
        is what gate 2 is for.

GATE 2  the span block is LOCAL. On the Katrina passage, min(start)..max(end) over the
        event containing the 1,400 must CONTAIN "1,400" and must NOT contain "Helene".
        The span architecture fails this at every threshold -- the block is either the
        bare name, missing the figure it should bind, or it swallows the competitor. It
        is the single most diagnostic case in the feed, because it is the one the whole
        span-embedding router turns on.

The threshold that matters here is ``extract(threshold=...)``. The per-event
``trigger_threshold``/``argument_threshold`` carried on the Schema are read only by the
span engine (``inference/runtime.py``); the boundary greedy path (``_decode_events``)
gates candidates on the single global threshold and never consults them, so setting only
the Schema values sweeps nothing on a boundary model. Both are set below.

Gates 3 and 4 (no regression on event_trigger/event_argument; the other heads survive)
are read from the run's own test_metrics.json and are not scored here.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from gliner2 import AutoExtractor, Schema

REPO = Path(__file__).resolve().parents[2]
TRACKED = REPO / "datasets/helene2024/_cache/tracked_rollup.json"
FEED = REPO / "datasets/helene2024/_cache/feed.jsonl"

# Verbatim from the feed. The Katrina passage is gate 2; the other two are controls --
# a genuine Helene window, and the 1916 case that has no storm name at all and that no
# span-content method is expected to reach.
KATRINA = ("At a recent rally in Reading, Pennsylvania, Trump said the response has been "
           "worse than during 2005's Hurricane Katrina, which left nearly 1,400 people "
           "dead and caused $200 billion in damages.")
HELENE = ("Helene decimated remote towns throughout Appalachia, left millions without "
          "power, knocked out cellular service and killed at least 246 people. It was the "
          "deadliest hurricane to hit the U.S. mainland since Katrina in 2005.")

# The sweep fixed before spending. EXTENDED is diagnostic only -- added after seeing
# that this architecture emits nothing above 0.05, so it cannot score the gate.
REGISTERED = (0.5, 0.4, 0.3, 0.2, 0.1)
EXTENDED = (0.05, 0.01)

EVENT_TYPES = ["Floods", "Storm", "Hurricane", "Earthquakes", "Tropical Storm"]
ROLES = {"dead": "number of people killed", "location": "where the deaths occurred",
         "event_name": "the name of the storm or disaster"}


def schema(etype: str, th: float) -> Schema:
    return Schema().events(
        {etype: {"roles": list(ROLES), "role_descriptions": dict(ROLES)}},
        trigger_threshold=th, argument_threshold=th)


def blocks(model, text: str, th: float):
    """Every event's min(start)..max(end) over its own trigger + argument spans."""
    out = []
    for etype in EVENT_TYPES:
        res = model.extract(text, schema(etype, th), threshold=th, include_spans=True)
        for ev in (res.get("event_extraction") or {}).get(etype, []):
            pts = []
            for t in ev.get("triggers") or []:
                if isinstance(t, dict) and "start" in t:
                    pts += [t["start"], t["end"]]
            for a in ev.get("arguments") or []:
                e = a.get("entity") if isinstance(a, dict) else None
                if isinstance(e, dict) and "start" in e:
                    pts += [e["start"], e["end"]]
            if pts:
                out.append({"type": etype, "lo": min(pts), "hi": max(pts),
                            "n_args": len(ev.get("arguments") or []),
                            "text": text[min(pts):max(pts)]})
    return out


def helene_windows(limit: int = 60):
    """Windows around each 'dead' observation -- the casualty-bearing text."""
    d = json.loads(TRACKED.read_text(encoding="utf-8"))
    feed = {round(r["t_hours"], 2): r["text"]
            for r in (json.loads(l) for l in FEED.open(encoding="utf-8"))}
    wins = []
    for a in d["articles"]:
        for o in a["observations"]:
            if o["role"] != "dead" or o["mode"] != "heuristic":
                continue
            t = feed.get(round(a["t_hours"], 2), "")
            i = t.find(str(o["span"]))
            if i >= 0:
                wins.append(t[max(0, i - 200): i + len(str(o["span"])) + 200])
    return wins[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = AutoExtractor.from_pretrained(args.model, map_location=args.device)
    model.eval()
    wins = helene_windows()
    print(f"model: {args.model}\n{len(wins)} casualty-bearing Helene windows\n")

    print("GATE 1 -- usable events-form on real wire copy (want >= 50%)")
    print(f"{'thresh':>7}{'windows w/ trigger+arg':>25}{'share':>9}")
    best, best_ext = (0.0, 0.0), (0.0, 0.0)
    for th in REGISTERED + EXTENDED:
        hits = sum(1 for w in wins if any(b["n_args"] for b in blocks(model, w, th)))
        share = hits / max(len(wins), 1)
        if th in REGISTERED and share > best[1]:
            best = (th, share)
        if share > best_ext[1]:
            best_ext = (th, share)
        tag = "" if th in REGISTERED else "   (diagnostic, not pre-registered)"
        print(f"{th:>7.2f}{hits:>25}{share:>9.1%}{tag}")
    print(f"  -> gate 1 {'PASS' if best[1] >= 0.50 else 'FAIL'} "
          f"-- best {best[1]:.1%} at threshold {best[0]:.2f} over the REGISTERED range "
          f"{REGISTERED[-1]}-{REGISTERED[0]}")
    if best_ext[1] >= 0.50 > best[1]:
        print(f"     (reaches {best_ext[1]:.1%} at {best_ext[0]:.2f}, below the registered "
              f"range -- check WHAT it binds there before reading that as progress)")
    print()

    print("GATE 2 -- the span block is LOCAL (Katrina block must hold '1,400', not 'Helene')")
    print(f"{'thresh':>7}  case      block                                    verdict")
    g2 = g2_ext = False
    for th in REGISTERED + EXTENDED:
        for name, text, must, forbid in (("KATRINA", KATRINA, "1,400", "Helene"),
                                         ("HELENE ", HELENE, "246", "Katrina")):
            bs = blocks(model, text, th)
            hit = next((b for b in bs if must in b["text"]), None)
            if hit is None:
                v = "no block contains the figure"
            elif forbid in hit["text"]:
                v = f"SWALLOWS {forbid}"
            else:
                v = "LOCAL -- ok"
                # Same rule as gate 1: only the pre-registered range can score it.
                if name == "KATRINA" and th in REGISTERED:
                    g2 = True
                elif name == "KATRINA":
                    g2_ext = True
            frag = (hit["text"][:38] + "...") if hit else "-"
            print(f"{th:>7.2f}  {name}  {frag:<40} {v}")
    print(f"  -> gate 2 {'PASS' if g2 else 'FAIL'} (scored on the REGISTERED range)")
    if g2_ext and not g2:
        print("     (the Katrina block is local only BELOW the registered range)")
    print()

    print("Gates 3 and 4 are NOT scored here, and NOT from the run's own test_metrics.json.")
    print("  That file is a different test composition at a different threshold, so its")
    print("  numbers are not comparable to the reference. Score the candidate on the 137k")
    print("  blind test at a PINNED 0.5:")
    print("    uv run python tools/train/eval.py \\")
    print("      --config tools/train/config/joint-boundary-mmbert-137k.yaml \\")
    print("      --checkpoint <ckpt> --threshold 0.5")
    print()
    print("  3. event_trigger / event_argument vs the incumbent, LIKE FOR LIKE:")
    print("       strict  0.7447 / 0.1014      fair (Ortmann)  0.7480 / 0.5629")
    print("     The 0.710 / 0.506 this line used to quote are RELAXED own-test numbers;")
    print("     comparing a strict score against them is a category error that made a")
    print("     2x improvement in strict argument F1 read as a halving.")
    print("  4. entity 0.6306 / relation 0.1573 strict, and structure 0.1119 -- the last")
    print("     one SWEPT to record threshold 0.1 (record_sweep_results/137k.json), so it")
    print("     is only comparable to a candidate swept the same way, not to a default-")
    print("     cutoff number: the record head's max object probability is 0.178, under")
    print("     the 0.5 default.")


if __name__ == "__main__":
    main()
