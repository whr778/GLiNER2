"""News feed -> EKF tracked casualty timeline. The whole research line in one pass.

Four stages, each swappable, so the demo is also an ablation:

  0. GATE          classification (multi-task)  is this article a mass-casualty report?
  1. EVENT         boundary / joint_ie          event type + trigger + argument spans
  2. EXTRACT       casualty structure model     bind NUMBERS to roles {dead,injured,missing}
  3. NORMALIZE     heuristic | classification   span -> (value, qualifier, source)
  4. TRACK         EKF (+ last_value baseline)  observations -> a state over time

**Division of labour, which is the point of the demo.** An EKF observation is
``(t, role, value, qualifier, source)``. Classification is the right tool for exactly
two of those fields plus the gate, because they are closed sets:

    qualifier  point | at_least | about | feared | interval
    source     official | major_outlet | preliminary

Both are currently decided by keyword heuristics (``extract._detect_qualifier`` /
``_detect_source``), and qualifier accuracy is the pipeline's weakest normalized field
(0.724 zero-shot, 0.691 after fine-tuning -- EKF_MHT_DESIGN sec 20). So ``--normalizer
classify`` is not a toy alternative; it targets a measured weak point, and
``--normalizer both`` scores them against each other on the same feed.

Classification CANNOT produce ``value`` (an open-vocabulary number) or bind a number to
a role on multi-fact text -- that is precisely the binding collapse of sec 17 that the
structure model exists to solve. Hence stage 2 stays a structure extractor.

    uv run python tools/ekf_showcase/run_pipeline.py \
        --feed datasets/ekf_showcase/feed.jsonl \
        --truth datasets/ekf_showcase/feed.truth.jsonl \
        --out datasets/ekf_showcase/tracked.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "datasets" / "disaster_streams"))

import extract as surface  # noqa: E402  (the shared normalizer)

ROLES = ("dead", "injured", "missing")
QUALIFIERS = {
    "point": "an exact figure stated plainly, with no hedge",
    "at_least": "a floor: the true number is at least this, phrased 'at least'",
    "about": "an approximation, phrased 'about', 'around' or 'roughly'",
    "feared": "a fear or expectation rather than a count, phrased 'feared'",
    "interval": "a vague bucket word with no digits, such as 'dozens' or 'thousands'",
}
SOURCES = {
    "official": "attributed to authorities, government or officials",
    "major_outlet": "reported by a news organisation with no official attribution",
    "preliminary": "explicitly early, initial or unconfirmed",
}


# --------------------------------------------------------------------------- #
# Stage 0 - gate
# --------------------------------------------------------------------------- #
def build_gate_schema(model):
    """Multiple classification tasks in one pass (tutorial 1, 'Multiple Tasks').

    Relevance is the load-bearing one; disaster_type rides along free because
    classification tasks share the encoder pass.
    """
    return (model.create_schema()
            .classification("relevance", {
                "mass_casualty": "reports deaths, injuries or missing people from a disaster or attack",
                "other": "any other topic: sport, markets, technology, weather, policy",
            })
            .classification("disaster_type", {
                "earthquake": "an earthquake or its aftermath",
                "flood": "flooding or storm surge",
                "fire": "a fire or explosion",
                "attack": "a deliberate attack",
                "other": "any other or unclear cause",
            }))


def gate(model, texts: List[str], threshold: float) -> List[Dict[str, Any]]:
    schema = build_gate_schema(model)
    out = []
    for text in texts:
        r = model.extract(text, schema, include_confidence=True)
        rel = r.get("relevance")
        label = rel.get("label") if isinstance(rel, dict) else rel
        conf = float(rel.get("confidence", 1.0)) if isinstance(rel, dict) else 1.0
        dis = r.get("disaster_type")
        out.append({
            "relevant": bool(label == "mass_casualty" and conf >= threshold),
            "relevance": label, "relevance_confidence": conf,
            "disaster_type": dis.get("label") if isinstance(dis, dict) else dis,
        })
    return out


# --------------------------------------------------------------------------- #
# Stage 1 - event extraction (boundary)
# --------------------------------------------------------------------------- #
def build_event_schema() -> Dict[str, Any]:
    """A boundary/joint_ie event schema. Roles are chosen to WINDOW the report -- the
    trigger and the casualty-bearing arguments -- so stage 1 does real work rather than
    only labelling."""
    return {"events": {"MassCasualtyIncident": ["casualties", "location", "cause"]}}


def extract_events(model, texts: List[str], threshold: float,
                   schema: Optional[Dict] = None) -> List[Dict[str, Any]]:
    schema = schema or build_event_schema()
    out = []
    for text in texts:
        try:
            r = model.batch_extract([text], [schema], threshold=threshold,
                                    include_spans=True)[0]
            out.append(r.get("event_extraction") or {})
        except Exception as exc:                      # a research checkpoint may not cope
            out.append({"_error": f"{type(exc).__name__}: {exc}"})
    return out


def event_envelopes(text: str, block: Dict[str, Any], margin: int = 40) -> List[Dict[str, Any]]:
    """`min(start) .. max(end)` over each event's trigger and argument spans.

    The envelope is the slice of article the event actually occupies, so downstream
    stages see one event's text instead of the whole article. That matters because
    qualifier and source are per-reading attributes: a whole-article view has to emit
    one label for readings that legitimately disagree, which is why the whole-text
    keyword scan scores ~0.49 on source. An event-derived envelope is semantically
    bounded rather than a fixed character window.

    ``margin`` pads each side, since the hedge ("officials said", "feared") often sits
    just outside the annotated spans.
    """
    out = []
    for etype, instances in (block or {}).items():
        if etype.startswith("_") or not isinstance(instances, list):
            continue
        for inst in instances:
            offsets = []
            for t in inst.get("triggers") or []:
                if isinstance(t, dict) and "start" in t:
                    offsets.append((t["start"], t["end"]))
            for a in inst.get("arguments") or []:
                e = a.get("entity")
                if isinstance(e, dict) and "start" in e:
                    offsets.append((e["start"], e["end"]))
            if not offsets:
                continue
            lo = max(0, min(o[0] for o in offsets) - margin)
            hi = min(len(text), max(o[1] for o in offsets) + margin)
            out.append({"event_type": etype, "start": lo, "end": hi, "text": text[lo:hi]})
    return out


# --------------------------------------------------------------------------- #
# Stage 2 - casualty structure extraction
# --------------------------------------------------------------------------- #
def build_casualty_schema():
    """Identical to datasets/disaster_streams/model_arm.py, so numbers from this demo
    are comparable with the measured EKF results."""
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
    if isinstance(v, dict):
        return v.get("text", ""), float(v.get("confidence", 1.0))
    return (v or ""), 1.0


# --------------------------------------------------------------------------- #
# Stage 3 - normalization
# --------------------------------------------------------------------------- #
def build_normalizer_schema(model):
    return (model.create_schema()
            .classification("qualifier", QUALIFIERS)
            .classification("source", SOURCES))


def role_window(text: str, span: str, left: int = 120, right: int = 60) -> str:
    """Text around the model-bound number.

    Qualifier and source are attributes of an INDIVIDUAL reading, not of the article:
    one report routinely carries three roles with different hedges and different
    attributions ("officials confirmed 5 dead ... early reports say dozens injured").
    Classifying the whole article therefore cannot be right for more than one of them --
    the same granularity error that makes ``_detect_source``'s whole-text keyword scan
    score ~0.49 on a 3-way task. So the classifier sees the neighbourhood of the number
    it is describing, the same premise as ``qualifier_near``'s character window.
    """
    m = surface.re.search(r"\d[\d,]*", span)
    key = m.group(0) if m else span
    i = text.find(key)
    if i < 0:
        return span
    return text[max(0, i - left): i + len(key) + right]


def normalize(text: str, span: str, mode: str, cls_model=None, cls_schema=None):
    """span -> (value, qualifier, source).

    ``value`` always comes from the digits in the model-bound span: no classifier can
    emit an open-vocabulary number. Only qualifier/source vary by mode.
    """
    value, _ = surface.value_qualifier(span)
    heur_qual = surface.qualifier_near(text, span)
    heur_src = surface._detect_source(text)
    if mode == "heuristic" or cls_model is None:
        return value, heur_qual, heur_src

    window = role_window(text, span)
    r = cls_model.extract(window, cls_schema, include_confidence=True)
    q, s = r.get("qualifier"), r.get("source")
    q = q.get("label") if isinstance(q, dict) else q
    s = s.get("label") if isinstance(s, dict) else s
    q = q if q in QUALIFIERS else heur_qual
    s = s if s in SOURCES else heur_src
    if mode == "hybrid":
        # Measured on this feed (81 matched obs): the keyword window beats zero-shot
        # classification on QUALIFIER (0.654 vs 0.395) because a hedge is a literal
        # lexical cue, while classification beats it on SOURCE (0.605 vs 0.494) because
        # attribution is semantic. Take each field from whichever actually wins.
        return value, heur_qual, s
    return value, q, s


# --------------------------------------------------------------------------- #
# Stage 4 - tracking
# --------------------------------------------------------------------------- #
def track(observations: List[Dict], grid: List[float]) -> Dict[str, Any]:
    import evaluate as ekf
    series: Dict[str, Any] = {}
    for role in ROLES:
        obs = sorted((o for o in observations if o["role"] == role), key=lambda o: o["t_hours"])
        series[role] = {
            "n_obs": len(obs),
            "ekf": ekf.est_ekf(obs, grid, role) if obs else [0.0] * len(grid),
            "last_value": ekf.est_last_value(obs, grid) if obs else [0.0] * len(grid),
        }
    return series


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feed", required=True)
    ap.add_argument("--truth", default=None)
    ap.add_argument("--out", default="datasets/ekf_showcase/tracked.json")
    ap.add_argument("--gate-model", default="fastino/gliner2-base-v1",
                    help="general model for the relevance gate + normalizer classification")
    ap.add_argument("--casualty-model", default="whr778/gliner2-base-v1-casualty")
    ap.add_argument("--event-model", default=None,
                    help="boundary checkpoint for stage 1; omit to skip event extraction")
    ap.add_argument("--normalizer", choices=("heuristic", "classify", "hybrid", "both"), default="heuristic")
    ap.add_argument("--gate-threshold", type=float, default=0.5)
    ap.add_argument("--event-threshold", type=float, default=0.3)
    ap.add_argument("--grid-step", type=float, default=6.0, help="hours between grid points")
    # Event models are 3-4x SLOWER on MPS than CPU (per-op sync overhead), so cpu is the
    # default rather than "best available".
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--window", choices=("article", "event"), default="article",
                    help="event: pass each event's min(start)..max(end) envelope to "
                         "stages 2-3 instead of the whole article (needs --event-model)")
    ap.add_argument("--limit", type=int, default=0, help="first N articles only (smoke test)")
    args = ap.parse_args()

    from gliner2 import AutoExtractor

    feed = [json.loads(l) for l in Path(args.feed).open(encoding="utf-8") if l.strip()]
    feed.sort(key=lambda r: r["t_hours"])
    if args.limit:
        feed = feed[: args.limit]
    texts = [r["text"] for r in feed]
    print(f"[feed] {len(feed)} articles, t {feed[0]['t_hours']}h .. {feed[-1]['t_hours']}h")

    print(f"[stage 0] gate            {args.gate_model}")
    gate_model = AutoExtractor.from_pretrained(args.gate_model, map_location=args.device)
    gates = gate(gate_model, texts, args.gate_threshold)
    kept = [i for i, g in enumerate(gates) if g["relevant"]]
    print(f"           kept {len(kept)}/{len(feed)} articles as mass-casualty")

    events: List[Dict[str, Any]] = [{} for _ in feed]
    if args.event_model:
        print(f"[stage 1] events          {args.event_model}")
        ev_model = AutoExtractor.from_pretrained(args.event_model, map_location=args.device)
        found = extract_events(ev_model, [texts[i] for i in kept], args.event_threshold)
        for i, e in zip(kept, found):
            events[i] = e
        n = sum(1 for e in events if e and "_error" not in e)
        print(f"           event blocks on {n}/{len(kept)} kept articles")
        del ev_model
    else:
        print("[stage 1] events          SKIPPED (--event-model not set)")

    print(f"[stage 2] casualty        {args.casualty_model}")
    cas_model = AutoExtractor.from_pretrained(args.casualty_model, map_location=args.device)
    cas_schema = build_casualty_schema()

    modes = ["heuristic", "classify", "hybrid"] if args.normalizer == "both" else [args.normalizer]
    cls_schema = (build_normalizer_schema(gate_model)
                  if any(m in modes for m in ("classify", "hybrid")) else None)

    per_mode: Dict[str, List[Dict]] = {m: [] for m in modes}
    articles: List[Dict[str, Any]] = []
    for i, row in enumerate(feed):
        entry = {"t_hours": row["t_hours"], "text": row["text"], **gates[i],
                 "events": events[i], "observations": []}
        if gates[i]["relevant"]:
            # --window event: hand stage 2/3 the event's own envelope instead of the whole
            # article, so per-reading attributes are judged on per-event text.
            envelopes = event_envelopes(row["text"], events[i]) if args.window == "event" else []
            entry["envelopes"] = envelopes
            read_text = envelopes[0]["text"] if envelopes else row["text"]
            rec = (cas_model.extract(read_text, cas_schema, include_confidence=True)
                   .get("casualty_report") or [{}])[0]
            for role in ROLES:
                span, conf = _cell(rec.get(role))
                if not span:
                    continue
                for mode in modes:
                    value, qual, src = normalize(read_text, span, mode, gate_model, cls_schema)
                    o = {"t_hours": row["t_hours"], "role": role, "value": value,
                         "qualifier": qual, "source": src, "confidence": conf,
                         "span": span, "mode": mode}
                    per_mode[mode].append(o)
                    # Keep EVERY mode on the article: retaining only the first made
                    # `--normalizer both` unscoreable, which is the whole point of it.
                    entry["observations"].append(o)
        articles.append(entry)
        if (i + 1) % 20 == 0:
            print(f"           {i + 1}/{len(feed)} articles")

    t0, t1 = feed[0]["t_hours"], feed[-1]["t_hours"]
    grid = [t0 + k * args.grid_step for k in range(int((t1 - t0) / args.grid_step) + 1)]

    result: Dict[str, Any] = {
        "feed": args.feed, "grid": grid, "articles": articles,
        "n_articles": len(feed), "n_relevant": len(kept),
        "tracked": {m: track(per_mode[m], grid) for m in modes},
        "n_observations": {m: len(per_mode[m]) for m in modes},
    }

    if args.truth and Path(args.truth).is_file():
        truth = [json.loads(l) for l in Path(args.truth).open(encoding="utf-8") if l.strip()]
        result["truth"] = {
            role: [_truth_at(truth, role, t) for t in grid] for role in ROLES
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] {out}")
    for m in modes:
        print(f"   {m:9} observations={len(per_mode[m]):3}  "
              + "  ".join(f"{r}:{result['tracked'][m][r]['n_obs']}" for r in ROLES))


def _truth_at(truth: List[Dict], role: str, t: float) -> Optional[float]:
    """Last trajectory value for ``role`` at or before ``t``."""
    best = None
    for rec in truth:
        if rec.get("t_hours", 0) <= t and role in rec:
            best = rec[role]
    return best


if __name__ == "__main__":
    main()
