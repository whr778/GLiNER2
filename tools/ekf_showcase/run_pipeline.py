"""News feed -> EKF tracked casualty timeline. The whole research line in one pass.

Four stages, each swappable, so the demo is also an ablation:

  0. GATE          classification (multi-task)  is this article a mass-casualty report?
  1. EVENT         DocEE classification         event type + 'Casualties and Losses' spans
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

**Measured on the 102-article demo feed** (81 observations matched to gold):

    window            n   value  qualifier  source
    whole article    81   1.000      0.654   0.494   <- best overall here
    DocEE window     46   0.848      0.652   0.565

Windowing improves SOURCE (+0.071) exactly as intended, but costs value binding and
43% of observations -- because every article in this feed is a SINGLE-event snippet,
so the article already is the envelope and the window only clips context.

**Multi-event feed** (``make_demo_feed.py --interference 2``: each article also carries
two OTHER disasters' snippets, so their numbers compete), DocEE-large stage 1:

    window            n  correct  wrong-event  qualifier  source
    whole article    84    0.369        0.226      0.613   0.645
    event envelope  351    0.399        0.177      0.636   0.479

Two readings, and the second matters more than the first:

1. The envelope does what it was designed to do -- **cross-event misbinding falls 0.226
   -> 0.177 (-22% relative)** and correct binding rises -- but it emits 4x the
   observations (each envelope read separately) and clips the attribution context that
   ``source`` needs.
2. Far larger: **value binding collapses from 1.000 on single-event text to 0.369 once
   articles are multi-event.** That is sec 17's number-to-role binding collapse
   reappearing at DOCUMENT level. The casualty model was fine-tuned on single-fact
   snippets and does not transfer to multi-event documents -- a training-data gap, not
   a decode gap, and the envelope only mitigates it.

Default stays ``--window article``; use ``--window event`` when articles are genuinely
multi-event and cross-event contamination costs more than attribution precision.

    uv run python tools/ekf_showcase/run_pipeline.py \
        --feed datasets/ekf_showcase/feed.jsonl \
        --truth datasets/ekf_showcase/feed.truth.jsonl \
        --out datasets/ekf_showcase/tracked.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
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
# Label descriptions for the relevance gate, v2. v1 described the negative class as
# "sport, markets, technology, weather, policy" -- topically distant filler invented for
# the demo feed. Benchmarked against 21k REAL annotated disaster messages, that gate
# admitted 58.5% of definitively non-disaster text at high confidence ("Your voice is
# sweet like a morning kiss", 0.98). The negatives it actually meets are close-domain:
# personal requests, translator notes, governance and policy news, aid-logistics
# inventories carrying huge numbers (5,400,000 kg of rice), science and finance writing,
# and single-casualty medical items. Two traps are explicit because they appear verbatim
# in the data: a lone death ("the corpse arrives today") is not a mass-casualty event,
# and disaster words are used metaphorically ("explosion in crowdfunding").
GATE_LABELS_V2 = {
    "mass_casualty": (
        "a report of how many people were killed, injured, or are missing in a specific "
        "disaster, accident or attack -- it states or estimates a TOLL for a group of people"
    ),
    "other": (
        "anything else, including: personal messages, greetings, thanks and requests for "
        "help, money, jobs or travel; notes from a translator about the message itself; "
        "politics, elections, government policy, diplomacy and development or NGO reports; "
        "aid logistics and supply inventories, even with very large quantities; business, "
        "finance, science and environment writing; one individual's illness, injury or "
        "death; and metaphorical use of disaster words such as an 'explosion' in an industry"
    ),
}


def build_gate_schema(model):
    """Multiple classification tasks in one pass (tutorial 1, 'Multiple Tasks').

    Relevance is the load-bearing one; disaster_type rides along free because
    classification tasks share the encoder pass.
    """
    return (model.create_schema()
            .classification("relevance", GATE_LABELS_V2)
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
# DocEE's own vocabulary. Not an invented ontology: `docee_event` is a real 59-label
# task in data/docee.*.jsonl, and models in this repo have trained on it. Restricted to
# the mass-casualty subset plus a handful of decoys so the classifier is choosing among
# plausible alternatives rather than being handed only the answer.
DOCEE_TASK = "docee_event"
DOCEE_CASUALTY_TYPES = [
    "Air Crash", "Armed Conflict", "Disease Outbreaks", "Droughts", "Earthquakes",
    "Famine", "Fire", "Floods", "Gas Explosion", "Insect Disaster", "Mass Poisoning",
    "Mine Collapses", "Mudslides", "Riot", "Road Crash", "Shipwreck", "Storm",
    "Train Collisions", "Tsunamis", "Volcano Eruption",
]
DOCEE_DECOYS = [
    "Sports Competition", "Election", "Financial Crisis", "Diplomatic Talks",
    "Awards Ceremony", "Organization Merge", "Government Policy Changes",
]
# The annotated casualty envelope. Values look like "35 dead and 24 injured" or
# "killing 15 children" -- i.e. already the multi-fact string the casualty model binds
# numbers from, so no min(start)/max(end) reconstruction is needed.
DOCEE_ENTITIES = {
    "Casualties and Losses": "people killed, injured, missing or otherwise harmed, with counts",
    "Location": "where the event happened",
    "Cause": "what caused the event",
    # Asked for so a figure can be checked against the event's own time window. Articles
    # quote HISTORICAL tolls: the Turkiye standfirst sits above a round-up of past quakes,
    # and the 1999 Izmit figure of 17,500 was tracked as a 2023 casualty count in every
    # configuration until this existed. Zero-shot extraction already finds the date --
    # 13/15 Izmit envelopes resolve to "August 1999" and NO genuine observation resolves
    # to an old date, so the check is high precision. It was simply never asked for.
    "Date": "a date, year or time reference such as 1999, August 1999, Monday, February 6",
}

# Four-digit years only; a bare "Monday" carries no year and cannot falsify anything.
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


def build_docee_schema(model):
    """Stage 1 in DocEE's own shape: a 59-way event-type CLASSIFICATION plus entities.

    DocEE does not annotate triggers and arguments -- it annotates a document-level event
    type and entity roles. So the natural stage 1 here is exactly the "multiple
    classification tasks" mechanism, and ``Casualties and Losses`` is the casualty
    envelope, annotated rather than derived.
    """
    return (model.create_schema()
            .classification(DOCEE_TASK, DOCEE_CASUALTY_TYPES + DOCEE_DECOYS)
            .entities(DOCEE_ENTITIES))


def extract_stage1(model, texts: List[str], threshold: float) -> List[Dict[str, Any]]:
    """Per article: the DocEE event type plus its casualty/location/cause spans."""
    schema = build_docee_schema(model)
    out = []
    for text in texts:
        try:
            r = model.extract(text, schema, threshold=threshold,
                              include_confidence=True, include_spans=True)
            ev = r.get(DOCEE_TASK)
            ents = r.get("entities") or {}
            if isinstance(ents, list):
                ents = ents[0] if ents else {}
            out.append({
                "event_type": ev.get("label") if isinstance(ev, dict) else ev,
                "event_confidence": float(ev.get("confidence", 1.0)) if isinstance(ev, dict) else None,
                "casualties": ents.get("Casualties and Losses") or [],
                "location": ents.get("Location") or [],
                "date": ents.get("Date") or [],
                "cause": ents.get("Cause") or [],
            })
        except Exception as exc:
            out.append({"_error": f"{type(exc).__name__}: {exc}"})
    return out


def _span_text(v) -> str:
    return v.get("text", "") if isinstance(v, dict) else (v or "")


def casualty_windows(text: str, block: Dict[str, Any], margin: int = 60) -> List[Dict[str, Any]]:
    """Context windows around each ``Casualties and Losses`` span.

    The span itself carries the numbers; the surrounding text carries the hedge and the
    attribution ("officials said", "early reports"), which qualifier/source need. So the
    window is the span plus a margin, not the span alone.
    """
    out = []
    for item in block.get("casualties") or []:
        span = _span_text(item)
        if not span:
            continue
        if isinstance(item, dict) and "start" in item:
            lo, hi = item["start"], item["end"]
        else:
            i = text.find(span)
            if i < 0:
                out.append({"span": span, "text": span})
                continue
            lo, hi = i, i + len(span)
        out.append({"span": span,
                    "start": max(0, lo - margin), "end": min(len(text), hi + margin),
                    "text": text[max(0, lo - margin): min(len(text), hi + margin)]})
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
def build_casualty_schema(with_location: bool = False, record_mode: str = "natural",
                          anchor: str = "dead"):
    """Identical to datasets/disaster_streams/model_arm.py, so numbers from this demo
    are comparable with the measured EKF results.

    ``with_location`` adds a location FIELD to the record. That makes attribution the
    model's job rather than a proximity heuristic's: the number and the place it belongs
    to are bound inside one record instance. Proximity cannot do this -- measured on the
    Turkiye-Syria standfirst, both countries sit within 26 characters of both numbers, so
    the nearest-location rule is a coin flip.

    ``record_mode`` declares Instance Formation, and omitting it is a silent trap on the
    BOUNDARY architecture: that path needs ``record_metadata``, a plain
    ``Schema().structure(name)`` emits none, and the extraction then returns nothing with
    no error. Every span model here is unaffected -- verified, plain and declared schemas
    give byte-identical output on `fastino/gliner2-base-v1`, including on an injured-only
    report where the ``dead`` anchor might have been expected to bite. So declaring it
    costs nothing today and is what lets a boundary model be pointed at this pipeline at
    all. Pass ``record_mode=None`` for the legacy form.
    """
    from gliner2 import Schema
    s = Schema()
    s = (s.structure("casualty_report", mode=record_mode, anchor=anchor)
         if record_mode else s.structure("casualty_report"))
    if with_location:
        s = s.field("location", dtype="str",
                    description="the country or place these deaths occurred in")
    return (s
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


def _emit(rec, read_text, row, entry, modes, per_mode, cls_model, cls_schema,
          event_key: str = "all", events_block: Optional[Dict] = None,
          event_year: int = 0, rejected: Optional[List] = None) -> None:
    """Turn one casualty_report reading into observations, one per role per mode.

    ``event_year`` enables the temporal check: a figure whose nearest date predates the
    event is a historical comparison, not a report of it.
    """
    rejected = rejected if rejected is not None else []
    for role in ROLES:
        span, conf = _cell(rec.get(role))
        if not span:
            continue
        stale = (out_of_window(read_text, span, events_block, event_year)
                 if event_year else None)
        if stale is not None:
            rejected.append({"value": span, "year": stale})
            continue
        for mode in modes:
            value, qual, src = normalize(read_text, span, mode, cls_model, cls_schema)
            if value is None:
                # No number in the span at all. Emitting 0 here is what turned 30 of 114
                # Helene observations into fabricated reports of no deaths; a missing
                # observation is recoverable, an invented zero is not.
                continue
            o = {"t_hours": row["t_hours"], "role": role, "value": value,
                 "qualifier": qual, "source": src, "confidence": conf,
                 "span": span, "mode": mode, "event_key": event_key}
            per_mode[mode].append(o)
            # Keep EVERY mode on the article: retaining only the first made
            # `--normalizer both` unscoreable, which is the whole point of it.
            entry["observations"].append(o)


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
_GENERIC_PLACE = {
    "the region", "the area", "temporary camps", "temporary shelters",
    "hardest-hit areas", "the country", "the city", "the district",
}


def out_of_window(text: str, span: str, events: Dict[str, Any], event_year: int,
                  slack: int = 1) -> Optional[int]:
    """The year of the nearest date to `span`, when that year predates the event.

    Returns None when the figure is temporally plausible, so the caller keeps it. A
    number is rejected only on POSITIVE evidence that it belongs to another time -- an
    absent or unparsable date is not evidence, and treating it as such would discard most
    of the feed.

    Nearest-date-by-character-distance is a weak proxy for attachment, and the same proxy
    failed for LOCATION (both countries within 26 chars of both numbers on the Turkiye
    standfirst). It is defensible here for a reason that does not apply there: dates are
    sparse and clustered, so the competing hypotheses are far apart rather than adjacent.
    Measured: 13/15 Izmit envelopes resolve to "August 1999" and 0 genuine observations
    resolve to an old year.
    """
    dates = [d for d in (events or {}).get("date") or [] if isinstance(d, dict) and "start" in d]
    if not dates:
        return None
    i = text.find(span)
    if i < 0:
        return None
    mid = i + len(span) / 2
    near = min(dates, key=lambda d: abs((d["start"] + d["end"]) / 2 - mid))
    m = _YEAR_RE.search(_span_text(near))
    if not m:
        return None
    year = int(m.group(0))
    return year if year < event_year - slack else None


def record_key(events: Dict[str, Any], rec: Dict[str, Any]) -> str:
    """Association key taken from the record's OWN location field.

    The strongest available signal, because it is the only one produced by the same
    decoding step that produced the number: the model emitted this place and this count
    as one instance. Every other strategy here infers the pairing after the fact from
    character offsets, which the Turkiye-Syria standfirst shows is close to a coin flip.
    """
    etype = (events or {}).get("event_type") or "unknown"
    span, _ = _cell(rec.get("location"))
    place = re.sub(r"[^\w\s-]", "", str(span or "")).strip().lower()
    if not place or place in _GENERIC_PLACE or place[0].isdigit():
        return etype
    return f"{etype}|{place}"


def _nearest_place(events: Dict[str, Any], envelope: Dict[str, Any]) -> str:
    """The location span closest to this envelope, by character distance.

    Taking the document's FIRST location instead is wrong whenever one document covers
    more than one place, and it fails loudly on the Turkiye-Syria feed: every article
    names Turkey before Syria, so all 16 documents keyed to `Earthquakes|turkey` and the
    Syrian tolls (5,800 among them) were tracked as if they were Turkish. Syria was
    detected -- 123 location spans across the feed -- it just never reached the key.

    Distance to the envelope is the available signal: the number and the place it belongs
    to are written near each other. It is a heuristic, not syntax, and it inherits the
    caveat in ``association_key`` -- genuinely ambiguous attachment is MHT's problem.
    """
    if not envelope or "start" not in envelope:
        return ""
    mid = (envelope["start"] + envelope["end"]) / 2
    best, best_d = "", None
    for item in events.get("location") or []:
        if not isinstance(item, dict) or "start" not in item:
            continue
        place = _span_text(item).strip().lower()
        if not place or place in _GENERIC_PLACE or place[0].isdigit():
            continue
        d = abs((item["start"] + item["end"]) / 2 - mid)
        if best_d is None or d < best_d:
            best, best_d = place, d
    return best


def association_key(events: Dict[str, Any], strategy: str,
                    envelope: Optional[Dict[str, Any]] = None) -> str:
    """Which event stream an observation belongs to.

    The tracker is single-stream: it assumes every observation describes the SAME
    incident. Pooling a whole feed violates that, and the violation dominates
    everything else -- on the multi-event feed ~half the extracted values came from
    other disasters (injured up to 18,334 against a truth peaking at 316), driving
    normalized RMSE to 102 while the same tracker scores 0.313 on clean observations.

    So associate before tracking. This uses the OBSERVABLE label (DocEE event type,
    plus a location when one is specific enough to identify a place) rather than
    enumerating hypotheses: association is given here, not latent. Genuine ambiguity --
    two earthquakes, or a feed with no event labels -- is the data-association problem
    proper, and belongs to MHT (see EKF_MHT_DESIGN); reaching for that first would
    build hypothesis machinery for hypotheses we do not yet need to enumerate.
    """
    if strategy == "none":
        return "all"
    etype = (events or {}).get("event_type") or "unknown"
    if strategy == "type":
        return etype
    # type+location: a place name only helps if it names a place. DocEE's generic
    # spans ("the region", "temporary camps") identify nothing and would split one
    # event into several streams, which is worse than pooling it.
    if strategy == "envelope":
        near = _nearest_place(events or {}, envelope or {})
        if near:
            return f"{etype}|{near}"
    places = [_span_text(p).strip().lower() for p in (events or {}).get("location") or []]
    places = [p for p in places if p and p not in _GENERIC_PLACE and not p[0].isdigit()]
    return f"{etype}|{places[0]}" if places else etype


def track(observations: List[Dict], grid: List[float]) -> Dict[str, Any]:
    """Track one stream (all observations assumed to be the same event)."""
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


def apply_rollup(observations: List[Dict], rollup: Dict[str, Any]) -> None:
    """Fold city/county/region keys up to the level the ground truth is keyed on.

    Two axes fragment independently on a real event, and both are handled:

    **Type.** The same event is classified differently across articles -- Helene appears as
    Floods, Storm and Mudslides -- so `Floods|florida` and `Storm|florida` are one stream
    described twice. `collapse_type` drops the type from the key.

    **Place.** `asheville north carolina`, `buncombe county` and `western north carolina`
    are all North Carolina; `southeastern us`, `appalachia` and `six states` are none of the
    six states -- they are the national aggregate, which is a SUM over states rather than a
    seventh region, and is keyed `__aggregate__` so downstream work can treat it that way
    instead of filing it under whichever state is nearest.

    A place with no alias is left ALONE rather than guessed at. An unmapped key is visible
    fragmentation; a wrongly-mapped one is a silent error in the wrong stream.
    """
    aliases = {k.lower(): v for k, v in (rollup.get("aliases") or {}).items()}
    collapse = bool(rollup.get("collapse_type"))
    for o in observations:
        key = o.get("event_key", "all")
        etype, _, place = key.partition("|")
        place = aliases.get(place.strip().lower(), place)
        o["event_key"] = place if (collapse and place) else (
            f"{etype}|{place}" if place else etype)


def merge_prefix_keys(observations: List[Dict]) -> None:
    """Fold `Earthquakes|syr` into `Earthquakes|syria`, in place.

    A clipped span is the same place under a shorter name, and splitting on it costs
    observations to a stream too small to track: on the Turkiye-Syria feed 2 of Syria's
    16 readings landed under `syr`. Merge only when one key is a strict PREFIX of a
    longer one with the same event type -- that catches truncation without asserting
    that similar-looking places are the same place.
    """
    keys = {o.get("event_key", "all") for o in observations}
    canon = {}
    for k in keys:
        longer = [c for c in keys
                  if c != k and c.startswith(k) and "|" in k and len(c) > len(k)]
        if longer:
            canon[k] = min(longer, key=len)
    for o in observations:
        k = o.get("event_key", "all")
        if k in canon:
            o["event_key"] = canon[k]


def track_by_event(observations: List[Dict], grid: List[float],
                   rollup: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One tracked stream per association key, largest first."""
    if rollup:
        apply_rollup(observations, rollup)
    merge_prefix_keys(observations)
    streams: Dict[str, List[Dict]] = {}
    for o in observations:
        streams.setdefault(o.get("event_key", "all"), []).append(o)
    order = sorted(streams, key=lambda k: -len(streams[k]))
    return {k: {"n_obs": len(streams[k]), **track(streams[k], grid)} for k in order}


# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    """Short HEAD, marked dirty when the tree has uncommitted changes.

    The dirty flag is the part that matters: the archived Helene baseline was produced by a
    tree that did not match any commit, and nothing in the artifact said so.
    """
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).resolve().parents[2])
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, cwd=Path(__file__).resolve().parents[2])
        if head.returncode:
            return "unknown"
        return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except OSError:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feed", required=True)
    ap.add_argument("--truth", default=None)
    ap.add_argument("--out", default="datasets/ekf_showcase/tracked.json")
    ap.add_argument("--gate-model", default="fastino/gliner2-base-v1",
                    help="general model for the relevance gate + normalizer classification")
    ap.add_argument("--casualty-model", default="whr778/gliner2-base-v1-casualty-docee",
                    help="best on all three showcase gates: multi-event correct 0.450, "
                         "wrong-event 0.181, single-event 1.000")
    ap.add_argument("--event-model", default=None,
                    help="boundary checkpoint for stage 1; omit to skip event extraction")
    ap.add_argument("--normalizer", choices=("heuristic", "classify", "hybrid", "both"), default="heuristic")
    ap.add_argument("--gate-threshold", type=float, default=0.5)
    ap.add_argument("--event-threshold", type=float, default=0.3)
    ap.add_argument("--grid-step", type=float, default=6.0, help="hours between grid points")
    # Event models are 3-4x SLOWER on MPS than CPU (per-op sync overhead), so cpu is the
    # default rather than "best available".
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--associate",
                    choices=("none", "type", "type+location", "envelope", "record"),
                    default="none", help="group observations into per-event streams "
                         "before tracking (needs --event-model for a real key)")
    ap.add_argument("--record-mode", default="natural",
                    choices=("natural", "anchorless", "latent", ""),
                    help="Instance Formation mode on the casualty schema. Required for "
                         "BOUNDARY models -- without it the record path has no metadata "
                         "and returns nothing silently. No effect on span models. Pass "
                         "'' for the legacy plain form")
    ap.add_argument("--envelope-margin", type=int, default=60,
                    help="chars of context each side of a casualty span. Record extraction has "
                         "an inverted-U response to this (see framing_experiment.py): starve it "
                         "and the record head never fires, flood it and it binds the wrong "
                         "pairs. The default 60 sits in the starved regime")
    ap.add_argument("--lead-chars", type=int, default=1100,
                    help="with --window lead: how much of the article head to read")
    ap.add_argument("--rollup", default="",
                    help="json map folding city/county/region keys up to the level the "
                         "ground truth is keyed on, plus __aggregate__ for multi-state "
                         "phrases. Unmapped places are left alone, never guessed")
    ap.add_argument("--event-year", type=int, default=0,
                    help="reject casualty figures whose nearest date predates this year "
                         "(0 = off). The 1999 Izmit toll was tracked as a 2023 figure in "
                         "every Turkiye configuration until this existed")
    ap.add_argument("--chunk-size", type=int, default=200,
                    help="with --window long: words per chunk. 200 is the measured band; "
                         "the library default 384 loses one binding on Turkiye")
    ap.add_argument("--chunk-overlap", type=int, default=50)
    ap.add_argument("--window", choices=("article", "event", "lead", "long"), default="article",
                    help="event: pass each DocEE 'Casualties and Losses' window to stages "
                         "2-3 instead of the whole article (needs --event-model)")
    ap.add_argument("--limit", type=int, default=0, help="first N articles only (smoke test)")
    args = ap.parse_args()

    from gliner2 import AutoExtractor

    rollup = (json.loads(Path(args.rollup).read_text(encoding="utf-8"))
              if args.rollup else None)
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
        found = extract_stage1(ev_model, [texts[i] for i in kept], args.event_threshold)
        for i, e in zip(kept, found):
            events[i] = e
        n = sum(1 for e in events if e and "_error" not in e)
        print(f"           event blocks on {n}/{len(kept)} kept articles")
        del ev_model
    else:
        print("[stage 1] events          SKIPPED (--event-model not set)")

    print(f"[stage 2] casualty        {args.casualty_model}")
    cas_model = AutoExtractor.from_pretrained(args.casualty_model, map_location=args.device)
    cas_schema = build_casualty_schema(with_location=args.associate == "record",
                                       record_mode=args.record_mode or None)

    modes = ["heuristic", "classify", "hybrid"] if args.normalizer == "both" else [args.normalizer]
    cls_schema = (build_normalizer_schema(gate_model)
                  if any(m in modes for m in ("classify", "hybrid")) else None)

    per_mode: Dict[str, List[Dict]] = {m: [] for m in modes}
    stale_hits: List[Dict] = []
    articles: List[Dict[str, Any]] = []
    for i, row in enumerate(feed):
        entry = {"t_hours": row["t_hours"], "text": row["text"], **gates[i],
                 "events": events[i], "observations": []}
        if gates[i]["relevant"]:
            # --window event: hand stage 2/3 the event's own envelope instead of the whole
            # article, so per-reading attributes are judged on per-event text.
            if args.window == "long":
                # extract_long chunks the WHOLE document with overlap and merges, so it
                # reads text the lead window never sees. Proven on Turkiye: 16/16 on both
                # countries over the full article at chunk_size 200, versus 15/16 at the
                # default 384 -- the framing curve's band, not a guess.
                records = (cas_model.extract_long(
                    row["text"], cas_schema, threshold=args.event_threshold,
                    chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
                ).get("casualty_report") or [])
                entry["envelopes"] = [{"text": f"<extract_long chunks of {args.chunk_size} words>"}]
                for rec in records:
                    key = (record_key(events[i], rec) if args.associate == "record"
                           else association_key(events[i], args.associate, {}))
                    _emit(rec, row["text"], row, entry, modes, per_mode,
                          gate_model, cls_schema, event_key=key,
                          events_block=events[i], event_year=args.event_year,
                          rejected=stale_hits)
                articles.append(entry)
                if (i + 1) % 20 == 0:
                    print(f"           {i + 1}/{len(feed)} articles")
                continue
            if args.window == "lead":
                # The article LEAD, not envelopes scattered through the whole piece.
                # Measured on Turkiye-Syria: reading standfirst + ~1000 chars with a
                # location-bearing record schema binds BOTH countries 16/16, while the
                # same model on the full 6.2k article gets Syria 0/16. News puts the
                # current toll up top and the historical comparison at the bottom.
                envelopes = [{"text": row["text"][: args.lead_chars]}]
            else:
                envelopes = (casualty_windows(row["text"], events[i], args.envelope_margin)
                             if args.window == "event" and events[i] else [])
            entry["envelopes"] = envelopes
            # EVERY envelope is read, not just the first. A multi-event article carries
            # one casualty span per incident, and reading only the first would discard
            # the rest -- which is precisely the case the envelope exists to handle.
            reads = envelopes or [{"text": row["text"]}]
            for env in reads:
                read_text = env["text"]
                # EVERY casualty_report instance, not just [0]. A multi-event-trained
                # model emits one record per incident, and reading only the first would
                # make it look identical to the single-instance model it replaces --
                # the same defect class as envelopes[0] above.
                records = (cas_model.extract(read_text, cas_schema, include_confidence=True)
                           .get("casualty_report") or [])
                # Per ENVELOPE, not per document: the envelope is the incident, and
                # keying at document level pooled every event an article mentioned.
                for rec in (records or [{}]):
                    key = (record_key(events[i], rec) if args.associate == "record"
                           else association_key(events[i], args.associate, env))
                    _emit(rec, read_text, row, entry, modes, per_mode,
                          gate_model, cls_schema, event_key=key)
        articles.append(entry)
        if (i + 1) % 20 == 0:
            print(f"           {i + 1}/{len(feed)} articles")

    t0, t1 = feed[0]["t_hours"], feed[-1]["t_hours"]
    grid = [t0 + k * args.grid_step for k in range(int((t1 - t0) / args.grid_step) + 1)]

    result: Dict[str, Any] = {
        "feed": args.feed, "grid": grid, "articles": articles,
        "n_articles": len(feed), "n_relevant": len(kept),
        "tracked": {m: track(per_mode[m], grid) for m in modes},
        "tracked_by_event": {m: track_by_event(per_mode[m], grid, rollup) for m in modes},
        "associate": args.associate,
        "n_observations": {m: len(per_mode[m]) for m in modes},
        # EVERY argument, not just `associate`. The 2026-08-10 Helene run recorded only
        # `associate` and its baseline (5.247 -> 0.591) is now unreproducible: the flag and
        # the rollup file it used existed as uncommitted working-tree state and changed
        # before being committed, so no git state regenerates it. That blocked the muting
        # arm's pre-registered guard. See muting_arm_results/PROVENANCE.md.
        "invocation": {
            "args": vars(args),
            "argv": sys.argv,
            "git_commit": _git_commit(),
        },
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
    if args.event_year:
        print(f"   [temporal] rejected {len(stale_hits)} figure(s) dated before "
              f"{args.event_year}: {[(h['value'][:14], h['year']) for h in stale_hits[:6]]}")
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
