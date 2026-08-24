"""Dual-label a sample of casualty records and measure whether Haiku is good enough.

The scope field (``run_pipeline.build_casualty_schema(with_scope=True)``) is a zero-shot
no-op, so it needs supervision. Before paying to label the whole corpus this probe answers
two questions, and the SECOND one matters more than the first:

1. Do a cheap labeller (Haiku 4.5) and an expensive one (Opus 5) agree?
2. Is the corpus worth labelling at all -- i.e. does it actually CONTAIN scope variety?

Question 2 exists because ``casualty_events`` is synthetic. If ~all figures label `place`,
the supervision teaches the model to answer `place` unconditionally, which silently asserts
the very thing the ratio gate exists to measure (see ``run_pipeline.normalize_scope``). That
is a corpus verdict, not a model verdict, and no amount of spend on a better labeller fixes
it.

Agreement is reported with Cohen's kappa as well as raw percent, because a corpus dominated
by one class makes raw agreement meaningless -- two labellers that always say `place` agree
100% of the time and have learned nothing.

Two strata are drawn and reported separately:
  random -- uniform over count-bearing records; gives the UNBIASED class distribution
  hard   -- >=2 count figures AND >=2 distinct locations; where scope ambiguity actually lives

Spend is reported from the API's own ``usage``, never estimated.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline import SCOPE_CLASSES, normalize_scope  # noqa: E402

COUNT_ROLES = ("dead", "injured", "missing")

# $ per 1M tokens (input, output), Anthropic first-party sync rates.
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
}

SYSTEM = """You label the SCOPE of casualty figures in disaster reports.

Each figure is bound to an event that names one or more locations. Decide what area the
figure's number counts for, RELATIVE to those locations:

  place      - the named location itself (e.g. location "North Carolina", "12 died" there)
  national   - a whole country or nation, wider than the named location (e.g. the national
               toll, "the countrywide total rose to 225")
  sub-place  - a narrower area inside the named location (one county, town, city, district)
  unclear    - the text genuinely does not settle it

`unclear` is a real answer, not a dustbin. Use it when the text does not say. Do NOT guess
`place` to avoid answering -- a wrong `place` is worse than an honest `unclear`.

Proximity to a place name does not decide it. In "The toll stood at 225 Friday; two more
were recorded in South Carolina", 225 is national and 2 is place, though both sit beside
place names. Read the sentence the figure is in, and the ones around it.

Return one JSON object with one entry per figure id you were given."""

SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "scope": {"type": "string", "enum": list(SCOPE_CLASSES)},
                },
                "required": ["id", "scope"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}


def count_figures(rec):
    """Every count-bearing argument, with the locations its own event names."""
    out = []
    for ev in (rec.get("output") or {}).get("events") or []:
        args = ev.get("arguments") or []
        locs = [a.get("entity", "") for a in args
                if (a.get("role") or "").lower() == "location"]
        for a in args:
            role = (a.get("role") or "").lower()
            # Scope is defined RELATIVE to a location. An event with no location
            # argument has nothing to be relative to, so asking for a scope label
            # there measures nothing (6% of figures, and 18% of the disagreements
            # in the first run).
            if role in COUNT_ROLES and locs:
                out.append({"role": role, "entity": a.get("entity", ""),
                            "event_type": ev.get("event_type", ""), "locations": locs})
    return out


def n_locations(rec):
    locs = set()
    for ev in (rec.get("output") or {}).get("events") or []:
        for a in ev.get("arguments") or []:
            if (a.get("role") or "").lower() == "location":
                locs.add(a.get("entity"))
    return len(locs)


def _standalone_spans(text, entity):
    """Offsets where `entity` occurs as a whole number, not inside a longer one.

    Plain ``str.find`` is wrong for short figures: find("6") matches the 6 inside
    "2026", so the window lands on a date rather than the casualty figure. Measured
    at 22% of figures before this guard.
    """
    pat = re.compile(r"(?<![\d,.])" + re.escape(entity) + r"(?![\d,.])")
    return [m.start() for m in pat.finditer(text)]


def window_for(text, entity, locations=(), pad=200):
    """The passage a figure actually lives in.

    ``casualty_events`` records concatenate several unrelated disaster snippets, so
    handing a labeller the whole record forces it to first locate the figure among
    three other disasters -- a needle-in-haystack task layered on top of the scope
    judgement, which is not what we are trying to measure and not what the pipeline
    does at inference. Among repeated occurrences prefer the one nearest this event's
    own location, then return the containing paragraph, falling back to the +/-200
    char window ``event_binding_probe.py`` and ``helene_audit_labels.json`` use.
    """
    hits = _standalone_spans(text, entity) or [text.find(entity)]
    hits = [h for h in hits if h >= 0] or [0]
    loc_at = [m for l in locations for m in _standalone_spans(text, l) or [text.find(l)]
              if m >= 0]
    i = min(hits, key=lambda h: min((abs(h - m) for m in loc_at), default=h))
    for para in re.split(r"\n\s*\n", text):
        j = text.find(para)
        if j <= i < j + len(para):
            return para.strip()
    return text[max(0, i - pad): i + len(entity) + pad].strip()


def user_msg(rec, figs, window=True):
    lines, blocks = [], []
    for i, f in enumerate(figs):
        loc = ", ".join(f["locations"]) or "(none named)"
        lines.append(f'{i}. "{f["entity"]}" role={f["role"]} '
                     f'event={f["event_type"]} locations=[{loc}]')
        if window:
            blocks.append(f'--- passage for figure {i} ---\n'
                          f'{window_for(rec["input"], f["entity"], f["locations"])}')
    body = "\n\n".join(blocks) if window else f"TEXT:\n{rec['input']}"
    return f"{body}\n\nFIGURES TO LABEL:\n" + "\n".join(lines)


def label(client, model, rec, figs, effort=None, max_tokens=4096, window=True):
    """One labelling call. Returns (labels_by_id, usage) or (None, usage) on refusal."""
    output_config = {"format": {"type": "json_schema", "schema": SCHEMA}}
    if effort:
        output_config["effort"] = effort
    resp = client.messages.create(
        model=model, max_tokens=max_tokens, system=SYSTEM,
        output_config=output_config,
        messages=[{"role": "user", "content": user_msg(rec, figs, window)}],
    )
    usage = (resp.usage.input_tokens, resp.usage.output_tokens)
    if resp.stop_reason in ("refusal", "max_tokens"):
        return None, usage
    # With thinking on, content[0] is a thinking block -- find the text block.
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        return None, usage
    got = {e["id"]: normalize_scope(e["scope"]) for e in json.loads(text)["labels"]}
    return {i: got.get(i, "unclear") for i in range(len(figs))}, usage


def kappa(a, b):
    """Cohen's kappa. Raw agreement is meaningless when one class dominates."""
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] / n * cb[c] / n for c in set(a) | set(b))
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def sample(path, n_random, n_hard, seed):
    """Two strata: `random` for an unbiased class distribution, `hard` for agreement."""
    recs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]
    pool, hard = [], []
    for r in recs:
        figs = count_figures(r)
        if not figs:
            continue
        pool.append(r)
        if len(figs) >= 2 and n_locations(r) >= 2:
            hard.append(r)
    rng = random.Random(seed)
    drawn = [("random", r) for r in rng.sample(pool, min(n_random, len(pool)))]
    picked = {id(r) for _, r in drawn}
    rest = [r for r in hard if id(r) not in picked]
    drawn += [("hard", r) for r in rng.sample(rest, min(n_hard, len(rest)))]
    return drawn, len(pool), len(hard)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/casualty_events.train.jsonl")
    p.add_argument("--candidate", default="claude-haiku-4-5")
    p.add_argument("--reference", default="claude-opus-5")
    p.add_argument("--effort", default="medium",
                   help="effort for the REFERENCE only; Haiku 4.5 rejects this param")
    p.add_argument("--n-random", type=int, default=80)
    p.add_argument("--n-hard", type=int, default=120)
    p.add_argument("--limit", type=int, default=0, help="pilot: cap total records")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--whole-record", action="store_true",
                   help="label against the FULL record instead of the figure's passage")
    p.add_argument("--seed", type=int, default=20260824)
    p.add_argument("--out", default="tools/ekf_showcase/scope_field_results")
    a = p.parse_args()

    drawn, n_pool, n_hard_pool = sample(a.data, a.n_random, a.n_hard, a.seed)
    if a.limit:
        drawn = drawn[:a.limit]
    print(f"corpus {a.data}: {n_pool} count-bearing records, {n_hard_pool} in the hard cell")
    print(f"sampled {len(drawn)} "
          f"({sum(s == 'random' for s, _ in drawn)} random / "
          f"{sum(s == 'hard' for s, _ in drawn)} hard)")
    print(f"candidate={a.candidate}  reference={a.reference} (effort={a.effort})\n")

    client = anthropic.Anthropic()
    items = [(i, stratum, r, count_figures(r)) for i, (stratum, r) in enumerate(drawn)]

    def run(model, effort):
        res, spend = {}, [0, 0]
        def one(it):
            i, _, rec, figs = it
            labs, (ti, to) = label(client, model, rec, figs, effort=effort,
                                   window=not a.whole_record)
            spend[0] += ti; spend[1] += to
            return i, labs
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for i, labs in ex.map(one, items):
                res[i] = labs
        return res, spend

    print("labelling with candidate ...")
    cand, cand_spend = run(a.candidate, None)
    print("labelling with reference ...")
    ref, ref_spend = run(a.reference, a.effort)

    total_cost = 0.0
    for name, model, sp in [("candidate", a.candidate, cand_spend),
                            ("reference", a.reference, ref_spend)]:
        pin, pout = PRICES.get(model, (0.0, 0.0))
        c = sp[0] / 1e6 * pin + sp[1] / 1e6 * pout
        total_cost += c
        print(f"  {name:9} {model:18} {sp[0]:>8,} in / {sp[1]:>8,} out  ${c:6.3f}")
    print(f"  {'TOTAL':9} {'':18} {'':>8}      {'':>8}      ${total_cost:6.3f}\n")
    report(items, cand, ref, a, total_cost)


def report(items, cand, ref, a, total_cost):
    rows, skipped = [], 0
    for i, stratum, rec, figs in items:
        c, r = cand.get(i), ref.get(i)
        if c is None or r is None:
            skipped += 1
            continue
        for k, f in enumerate(figs):
            rows.append({"rec": i, "stratum": stratum, "fig": k, "role": f["role"],
                         "entity": f["entity"], "locations": f["locations"],
                         "passage": window_for(rec["input"], f["entity"], f["locations"]),
                         "cand": c[k], "ref": r[k]})
    if skipped:
        print(f"WARNING: {skipped} records dropped (refusal / max_tokens / no text block)")
    if not rows:
        print("no labelled figures -- nothing to report")
        return

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines = []
    def say(s=""):
        print(s); lines.append(s)

    say("=" * 72)
    say("QUESTION 2 (decides the CORPUS): does the data contain scope variety?")
    say("=" * 72)
    rnd = [x for x in rows if x["stratum"] == "random"]
    for label_src in ("ref", "cand"):
        for name, sub in (("random", rnd), ("hard", [x for x in rows if x["stratum"] == "hard"])):
            if not sub:
                continue
            dist = Counter(x[label_src] for x in sub)
            n = len(sub)
            share = "  ".join(f"{c}={dist[c] / n:5.1%}" for c in SCOPE_CLASSES)
            say(f"  {label_src:4} {name:6} n={n:4}  {share}")
    if rnd:
        d = Counter(x["ref"] for x in rnd)
        top, cnt = d.most_common(1)[0]
        dom = cnt / len(rnd)
        say(f"\n  dominant class on the unbiased sample: {top} at {dom:.1%}")
        say("  VERDICT: " + (
            f"corpus is DEGENERATE for scope -- {dom:.0%} is one class. Labelling it teaches\n"
            "           the model to answer unconditionally. Fix the corpus, not the labeller."
            if dom >= 0.90 else
            f"corpus carries real scope variety ({1 - dom:.0%} non-{top}). Worth labelling."))

    say()
    say("=" * 72)
    say("QUESTION 1 (decides the MODEL): candidate vs reference")
    say("=" * 72)
    for name, sub in (("overall", rows),
                      ("random", rnd),
                      ("hard", [x for x in rows if x["stratum"] == "hard"])):
        if not sub:
            continue
        agree = sum(x["cand"] == x["ref"] for x in sub) / len(sub)
        k = kappa([x["cand"] for x in sub], [x["ref"] for x in sub])
        say(f"  {name:8} n={len(sub):4}  agreement {agree:6.1%}   kappa {k:5.3f}")

    say("\n  unclear rate (a labeller that never says unclear is failing):")
    for src, model in (("cand", a.candidate), ("ref", a.reference)):
        u = sum(x[src] == "unclear" for x in rows) / len(rows)
        say(f"    {model:18} {u:5.1%}")

    say("\n  confusion (rows = reference, cols = candidate):")
    say("    " + " " * 11 + "".join(f"{c:>11}" for c in SCOPE_CLASSES))
    for rc in SCOPE_CLASSES:
        cells = [sum(1 for x in rows if x["ref"] == rc and x["cand"] == cc)
                 for cc in SCOPE_CLASSES]
        say(f"    {rc:>11}" + "".join(f"{v:>11}" for v in cells))

    disagree = [x for x in rows if x["cand"] != x["ref"]]
    rng = random.Random(a.seed)
    spot = rng.sample([x for x in rows if x["cand"] == x["ref"]],
                      min(30, len(rows) - len(disagree)))
    (out / "scope_probe_disagreements.json").write_text(
        json.dumps(disagree, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "scope_probe_agreement_spotcheck.json").write_text(
        json.dumps(spot, ensure_ascii=False, indent=2), encoding="utf-8")
    say(f"\n  {len(disagree)} disagreements -> scope_probe_disagreements.json (adjudicate ALL)")
    say(f"  {len(spot)} agreements    -> scope_probe_agreement_spotcheck.json (spot-check;")
    say("                                    correlated error is invisible to agreement)")
    say(f"\n  probe spend: ${total_cost:.3f}")
    (out / "scope_probe.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out / 'scope_probe.txt'}")


if __name__ == "__main__":
    main()
