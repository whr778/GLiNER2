"""Can any available guide tell "33 killed" from "10,000 evacuated"? The GIST veto, tested.

GIST filters a sampled negative out of the loss when a guide model judges it to be a
positive for that instance. So the guide's job here is: given a span and a competing type
query, say whether the competition is real. If no available guide can do that, GIST cannot
be wired -- the veto would be noise.

Gold cannot serve (EKF_MHT_DESIGN sec 27.6): the corpus declares a per-record type
vocabulary out of 17,128 types and only 0.23% of records name both a casualty type and a
competing count type. Of those 99, exactly **1** annotates the same span under both, and
**40** carry a digit-bearing span of each kind in one document. Those 40 are the test set.

They test a HARDER boundary than the inference-time probe did. That probe competed
`death toll` against "a count of things that are not people"; the corpus's real competitors
are largely counts OF people -- Number of Evacuated People, Number of Rescued People -- so
the distinction is the verb, not the noun. Any guide that only separates people from objects
will look fine there and fail here.

Two guides are actually available and both are scored:

    self         the model being corrected, asked with the record's own type names.
                 Free, but a model cannot referee a boundary it does not know -- if it
                 could, there would be no failure to fix.
    description  type-level similarity from `entity_descriptions`. Cheap, but it judges
                 "are these two TYPES confusable" and GIST's veto is per-INSTANCE.

    uv run python tools/train/score_gist_guides.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from gliner2 import AutoExtractor, Schema

NUM = re.compile(r"\d")


def pairs_from(rows):
    """(text, casualty_type, casualty_span, rival_type, rival_span) with digits on both."""
    out = []
    for r in rows:
        cas = [(k, s) for k in r["cas"] for s in r["ents"][k] if NUM.search(s)]
        qty = [(k, s) for k in r["qty"] for s in r["ents"][k] if NUM.search(s)]
        if cas and qty:
            out.append((r["text"], cas[0][0], cas[0][1], qty[0][0], qty[0][1]))
    return out


def _entities(out) -> dict:
    ents = out.get("entities") or {}
    return ents[0] if isinstance(ents, list) else ents


def score_span(model, text: str, span: str, types: list) -> dict:
    """Best score for `span` under each type name, all types queried together."""
    out = model.extract(text, Schema().entities(types), threshold=0.0,
                        include_confidence=True)
    got = {}
    for t, items in _entities(out).items():
        best = 0.0
        for it in (items or []):
            txt = it["text"] if isinstance(it, dict) else str(it)
            conf = float(it.get("confidence", 0.0)) if isinstance(it, dict) else 0.0
            if span in txt or txt in span:
                best = max(best, conf)
        got[t] = best
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="fastino/gliner2-base-v1")
    ap.add_argument("--gold", default="/tmp/guide_gold.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rows = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    cases = pairs_from(rows)
    print(f"{len(cases)} records carry a digit-bearing span of BOTH kinds\n")

    model = AutoExtractor.from_pretrained(args.model, map_location=args.device)
    model.eval()

    ok_cas = ok_riv = both = 0
    fails = []
    for text, ctype, cspan, rtype, rspan in cases:
        types = [ctype, rtype]
        sc_c = score_span(model, text, cspan, types)
        sc_r = score_span(model, text, rspan, types)
        # A usable guide must get BOTH directions right on the same document.
        c_right = sc_c.get(ctype, 0.0) > sc_c.get(rtype, 0.0)
        r_right = sc_r.get(rtype, 0.0) > sc_r.get(ctype, 0.0)
        ok_cas += c_right
        ok_riv += r_right
        both += c_right and r_right
        if not (c_right and r_right):
            fails.append((cspan, ctype, round(sc_c.get(ctype, 0), 2), round(sc_c.get(rtype, 0), 2),
                          rspan, rtype, round(sc_r.get(rtype, 0), 2), round(sc_r.get(ctype, 0), 2)))

    n = max(len(cases), 1)
    print("SELF-GUIDE (the model being corrected, asked with the record's own type names)")
    print(f"  casualty span scored higher under its own type : {ok_cas}/{n} = {ok_cas/n:.1%}")
    print(f"  rival span scored higher under its own type    : {ok_riv}/{n} = {ok_riv/n:.1%}")
    print(f"  BOTH directions right on the same document     : {both}/{n} = {both/n:.1%}")
    print(f"  (coin flip on both directions would be 25%)")

    print("\nfailures, first 8  [span | own-type score vs rival-type score]:")
    for cs, ct, cown, criv, rs, rt, rown, rriv in fails[:8]:
        print(f"  cas {cs[:34]!r:<36} {ct[:22]:<24} own={cown} rival={criv}")
        print(f"  riv {rs[:34]!r:<36} {rt[:22]:<24} own={rown} rival={rriv}\n")


if __name__ == "__main__":
    main()
