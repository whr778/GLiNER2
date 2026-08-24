"""Can the mmBERT/boundary front end do stage 2 (the casualty RECORD)? Yes, at its own threshold.

Measured 2026-08-24 over 12 Helene articles:

    model                         threshold        records   scope
    casualty-docee (span)         default 0.5           12   unclear x12
    ekf-frontend  (boundary)      default 0.5            2   unclear x2
    ekf-frontend  (boundary)      anchor 0.15            8   unclear x8
    ekf-frontend  (boundary)      anchor 0.05           15   unclear x14, sub-place x1

The boundary record head does NOT fail silently here -- the predicted empty result did not
happen -- but at the 0.5 default it under-fires 6x, exactly the direction the record sweep
implies: its max object probability is 0.178, so 0.5 is a cutoff it can barely reach. At
0.05 it produces MORE records than the span model (15 vs 12) and without the span model's
`None` cells.

`boundary_settings` is a FROZEN dataclass; use `dataclasses.replace`, not attribute
assignment.

Original question:

Prediction: empty, because the boundary record head's max object probability is 0.178
against a default record_anchor_threshold of 0.5. Test it, and test it again with the
threshold lowered, so an empty result is attributed rather than assumed.
"""
import json, sys, collections
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/ekf_showcase"))
from run_pipeline import build_casualty_schema, normalize_scope
from gliner2 import AutoExtractor

feed = [json.loads(l) for l in open(REPO / "datasets/helene2024/_cache/feed.jsonl", encoding="utf-8")]
sch = build_casualty_schema(with_location=True, record_mode="natural", with_scope=True)

for name in ("whr778/gliner2-base-v1-casualty-docee", "whr778/gliner2-ekf-frontend-mmbert"):
    m = AutoExtractor.from_pretrained(name, map_location="cpu"); m.eval()
    arch = type(m).__mro__[0].__name__
    for anchor_th in (None, 0.15, 0.05):
        if anchor_th is not None:
            import dataclasses
            bs = getattr(m, "boundary_settings", None)
            if bs is None:
                print(f"  {name.split('/')[-1]:<34} (span arch, no record head) skipped")
                continue
            # frozen dataclass -> replace, do not mutate
            m.boundary_settings = dataclasses.replace(
                bs, record_anchor_threshold=anchor_th, record_field_threshold=anchor_th)
        n_rec, scopes, deads = 0, [], []
        for row in feed[:12]:
            res = m.extract(row["text"], sch, include_confidence=True)
            for rec in (res.get("casualty_report") or []):
                n_rec += 1
                scopes.append(normalize_scope(rec.get("scope")))
                d = rec.get("dead")
                deads.append(str(d.get("text") if isinstance(d, dict) else d)[:12])
        tag = "default(0.5)" if anchor_th is None else f"anchor={anchor_th}"
        print(f"  {name.split('/')[-1]:<34} {arch:<22} {tag:<14} records={n_rec:>3} "
              f"scope={dict(collections.Counter(scopes))}")
        if deads[:5]:
            print(f"      dead cells: {deads[:5]}")
