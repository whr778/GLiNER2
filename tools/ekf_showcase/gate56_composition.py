"""Should the date gate [5] and the scope gate [6] be merged into one LEARNED gate?

Both answer the same question -- is this figure about THIS event? -- from different
evidence: [5] from the nearest date, [6] from the declared place hierarchy. Scored
against the 86 hand-audited Helene occurrence labels (helene | cross-event |
non-casualty | unclear), on the 106 'dead' observations of the stored run.

The stored run has event_year unset, so out_of_window never fired in it. The 10 genuine
observations it would reject are therefore still present and this is a clean
counterfactual rather than a double-count.
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline import out_of_window                      # noqa: E402
from event_binding_probe import _ctx_key, window, load_labels  # noqa: E402

roll = json.load(open("datasets/helene2024/rollup.json", encoding="utf-8"))
IN = set(roll["aliases"]) | set(roll["aliases"].values())
h = roll["hierarchy"]
IN |= {h.get("aggregate")} | set(h.get("parts") or [])
IN = {str(x).lower() for x in IN if x}

# Document structure, not world knowledge: a syndication marker between the article
# body and the figure. Available on day one of a new event, unlike rollup.json.
BOILER = re.compile(r"RELATED COVERAGE|MORE COVERAGE|READ MORE|SEE ALSO|Related:", re.I)

labels = load_labels()
res = json.loads(Path("datasets/helene2024/_cache/tracked_rollup.json")
                 .read_text(encoding="utf-8"))
rows = []
for a in res["articles"]:
    for o in a["observations"]:
        if o["role"] != "dead":
            continue
        ctx, _off = window(a["text"], o.get("span") or "")
        if ctx is None:
            continue
        lab = labels.get(_ctx_key(ctx, int(o["value"])))
        lab = (lab or {}).get("label") if isinstance(lab, dict) else lab
        pre = ctx[:_off]
        rows.append((str(o.get("event_key", "")).lower() not in IN,
                     out_of_window(a["text"], o.get("span") or "",
                                   a.get("events") or {}, 2024),
                     lab or "unlabelled",
                     bool(BOILER.search(pre))))

n_h = sum(1 for r in rows if r[2] == "helene")
n_c = sum(1 for r in rows if r[2] == "cross-event")
print(f"declared in-scope keys: {len(IN)}")
print(f"population {len(rows)}: {n_h} genuine helene, {n_c} cross-event\n")
print(f"  {'rule':22} {'caught':>10}   false-rejects of genuine")
for name, pred in (("gate6 scope only", lambda a, b, c: a),
                   ("gate5 date only", lambda a, b, c: bool(b)),
                   ("UNION (series = OR)", lambda a, b, c: a or bool(b)),
                   ("INTERSECTION (AND)", lambda a, b, c: a and bool(b)),
                   ("boilerplate marker", lambda a, b, c: c)):
    tp = sum(1 for g6, g5, l, bp in rows if pred(g6, g5, bp) and l == "cross-event")
    fp = sum(1 for g6, g5, l, bp in rows if pred(g6, g5, bp) and l == "helene")
    print(f"  {name:22} {tp:>5}/{n_c}      {fp:2}/{n_h} = {fp / n_h:5.1%}")
print("\n  gate5's marginal contribution over gate6 alone: +1 catch, +10 false rejects")
