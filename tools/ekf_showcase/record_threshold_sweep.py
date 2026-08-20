"""Is a boundary model's silence on real news a capability gap or a threshold gap?

    uv run python tools/ekf_showcase/record_threshold_sweep.py <model-id>

On whr778/gliner2-joint-boundary-mmbert-137k-clean the answer is threshold, decisively: at
the default 0.5 one window in 40 yields a record whose `dead` matches the target span; at
0.10 it is 37 of 40. Reported 2026-08-20.


Record decode has its OWN thresholds (record_anchor_threshold / record_field_threshold),
default 0.5, which the ordinary threshold sweep never touches. They are read from
``model.boundary_settings`` -- NOT from ``model.config``. Setting the config attributes
does nothing and yields a flat sweep, which is a bug in the test, not a property of the
model.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gliner2 import AutoExtractor                       # noqa: E402
from event_binding_probe import binding_schema, window  # noqa: E402

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    sys.exit(__doc__)

d = json.loads((REPO / "datasets/helene2024/_cache/tracked_rollup.json").read_text(encoding="utf-8"))
feed = {round(r["t_hours"], 2): r["text"]
        for r in (json.loads(l) for l in (REPO / "datasets/helene2024/_cache/feed.jsonl").open(encoding="utf-8"))}

pairs = []
for a in d["articles"]:
    for o in a["observations"]:
        if o["role"] == "dead" and o["mode"] == "heuristic":
            c, _ = window(feed.get(round(a["t_hours"], 2), ""), str(o["span"]))
            if c:
                pairs.append((c, str(o["span"])))
pairs = pairs[:40]
print(f"{len(pairs)} windows\n")

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    sys.exit(__doc__)
model = AutoExtractor.from_pretrained(sys.argv[1], map_location="cpu")
model.eval()
sch = binding_schema()
import dataclasses
base = model.boundary_settings
assert hasattr(base, "record_anchor_threshold"), "wrong settings object"
# boundary_settings is a FROZEN dataclass, so it cannot be mutated in place -- which is
# very likely why these two thresholds have never been swept.

print(f"{'anchor':>8}{'field':>8}{'records':>9}{'w/ dead':>9}{'w/ event':>10}{'span matched':>14}")
for anc in (0.5, 0.3, 0.2, 0.1, 0.05, 0.01):
    model.boundary_settings = dataclasses.replace(
        base, record_anchor_threshold=anc, record_field_threshold=min(anc, 0.5))
    n = dd = ev = matched = 0
    for ctx, span in pairs:
        for r in (model.extract(ctx, sch).get("casualty_report") or []):
            n += 1
            dead = str(r.get("dead") or "").strip()
            dd += bool(dead)
            ev += bool(str(r.get("event") or "").strip())
            matched += bool(dead and span in dead)
    print(f"{anc:>8.2f}{min(anc,0.5):>8.2f}{n:>9}{dd:>9}{ev:>10}{matched:>14}")
