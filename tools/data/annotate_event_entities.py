"""Second pass over the Turkish buy: role-typed ARGUMENT SPANS, conditioned on the type.

The 30,134-document type buy (annotate_event_type.py) purchased `{label, evidence}` --
one event type and a free-text quote. It bought no role-typed spans, so the Turkish arm
has ZERO supervision for `Location`, `Date`, `Casualties and Losses` or `Cause`. That is
the half of stage 1 the tracker leans on hardest: `Date` is the only field that can reject
a historical toll (the Turkish feed quotes Haiti 2010's 316,000 and Antakya 115 AD's
260,000 as figures the extractor binds CORRECTLY as `dead`), and `Location` drives
association, the documented bottleneck.

THE MENU IS CONDITIONED ON THE TYPE WE ALREADY BOUGHT. DocEE is not a flat role space --
it carries per-event-type role sets, 59 types at a median of 9 roles each. Because the
first pass already labelled every document's type, this pass can show an `Earthquakes`
article only its 15 roles rather than all 356. That keeps prompts tight, matches
DocEE-en's and DocEE-zh's structure exactly so the three languages stay parallel, and
picks up profile-bearing fields a 4-role menu would miss -- `Magnitude`, `Epicenter`,
`Number of Evacuated People`. Magnitude in particular is a stronger regional-profile
feature than toll alone: M7.8 in Turkey is ordinary, M7.8 in Fiji is not.

Menus are derived from `data/docee.train.jsonl` at run time rather than frozen into a
constant, so they cannot drift from the English corpus they must match.

SPANS MUST BE VERBATIM. The boundary head locates fields as spans, so a paraphrase trains
nothing while reporting no error -- the same reason `zh_multitask` dropped its 2.2%
non-verbatim spans rather than repairing them. Non-verbatim returns are DROPPED and
counted here, never edited into place.

    pilot : uv run python tools/data/annotate_event_entities.py --limit 500 --batch \
                --out data/turkish_gate/ev_ent_tr_pilot
    buy   : uv run python tools/data/annotate_event_entities.py --batch \
                --out data/turkish_gate/ev_ent_tr
    recover: uv run python tools/data/annotate_event_entities.py \
                --fetch-batch msgbatch_... --out data/turkish_gate/ev_ent_tr

Needs ANTHROPIC_WORKSPACE_ID; the key lives in ~/.zshrc, which non-interactive shells do
NOT source -- use `zsh -ic` AND pass the workspace id. Getting only one right still 400s
before a batch is created. A killed poller does NOT lose the batch: use --fetch-batch.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _split import dumps_record, normalize_group_key  # noqa: E402
from data.synthetic.providers import (  # noqa: E402
    REFUSAL_MARK, AnthropicProvider, ProviderConfig,
)

MAX_CHARS = 6000
NONE_LABEL = "none"
# Roles that are meaningful for ANY event, used for documents the type pass labelled
# `none` (16.9%). They still carry places, dates and casualty counts.
GENERIC_ROLES = ["Date", "Location", "Casualties and Losses", "Cause"]

SYSTEM = (
    "You extract role-typed argument spans from Turkish news documents, for an "
    "information-extraction training set. Every span you return must be copied VERBATIM "
    "from the article. Reply with a single JSON object and nothing else."
)

USER = """This Turkish news article reports a {etype} event.

Extract the spans that fill each role below. Roles are English labels for Turkish text -- this is intentional.

{roles}

Rules:
- Copy each span VERBATIM from the article, character for character. A paraphrase, a translation, or a corrected form is unusable and will be discarded.
- Omit a role entirely if the article does not state it. Do not guess, and do not return an empty string.
- A role may have more than one span; return them as a list.
- Keep spans tight -- the phrase that fills the role, not the whole sentence. For "Casualties and Losses" the natural span is the figure with its wording, e.g. "3 bin 549 kisi hayatini kaybetti".

Reply as: {{"<role>": ["<verbatim span>", ...], ...}} using only the roles listed above. Return {{}} if none apply.

ARTICLE:
{text}"""


def role_menus(en_corpus: Path, min_share: float, cap: int) -> dict[str, list[str]]:
    """Per-event-type role sets, from the English corpus these must stay parallel to.

    Kept to roles used by at least `min_share` of that type's documents: DocEE's role
    space has a long tail, and a menu carrying one-off roles invites the annotator to
    manufacture fillers for them.
    """
    per_type: dict[str, Counter] = defaultdict(Counter)
    doc_counts: Counter = Counter()
    with en_corpus.open(encoding="utf-8") as fh:
        for line in fh:
            out = json.loads(line).get("output") or {}
            etype = None
            for c in out.get("classifications") or []:
                for x in (c.get("true_label") or []):
                    etype = str(x)
            if not etype:
                continue
            doc_counts[etype] += 1
            for role in (out.get("entities") or {}):
                per_type[etype][role] += 1
    menus: dict[str, list[str]] = {}
    for etype, roles in per_type.items():
        n = max(doc_counts[etype], 1)
        keep = [r for r, c in roles.most_common() if c / n >= min_share][:cap]
        if keep:
            menus[etype] = keep
    return menus


def load_typed(paths: list[Path], limit: int, skip_none: bool) -> list[dict]:
    """Documents the type pass already labelled, deduped by the split-hygiene key."""
    out, seen = [], set()
    for p in paths:
        if not p.exists():
            print(f"[ent-ann] MISSING {p}, skipped")
            continue
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                text = (r.get("input") or "")[:MAX_CHARS]
                key = normalize_group_key(text)[:300]
                if not text or key in seen:
                    continue
                if skip_none and r.get("label") == NONE_LABEL:
                    continue
                seen.add(key)
                out.append({"text": text, "label": r.get("label"),
                            "source": r.get("source", "turkish_news"),
                            "stratum": r.get("stratum", "")})
    return out[:limit] if limit else out


def parse(raw: str, allowed: set[str], text: str) -> tuple[dict, int] | None:
    """Role -> verbatim spans. Returns (entities, n_dropped) or None if unusable."""
    if not raw or REFUSAL_MARK in raw:
        return None
    try:
        obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    ents: dict[str, list[str]] = {}
    dropped = 0
    for role, spans in obj.items():
        if role not in allowed:
            dropped += len(spans) if isinstance(spans, list) else 1
            continue
        for s in (spans if isinstance(spans, list) else [spans]):
            if not isinstance(s, str) or not s.strip():
                continue
            s = s.strip()
            if s not in text:            # non-verbatim: dropped, never repaired
                dropped += 1
                continue
            ents.setdefault(role, [])
            if s not in ents[role]:
                ents[role].append(s)
    return ents, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--typed", nargs="+", default=[
        "data/turkish_gate/ev_ann_tr.jsonl",
        "data/turkish_gate/ev_ann_tr_pilot.jsonl",
        "data/turkish_gate/ev_ann_tr_pilot2.jsonl",
    ], help="output of the TYPE pass; the menu is conditioned on its label")
    ap.add_argument("--en-corpus", type=Path, default=Path("data/docee.train.jsonl"))
    ap.add_argument("--out", default="data/turkish_gate/ev_ent_tr")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-role-share", type=float, default=0.02,
                    help="drop roles used by fewer than this share of a type's documents")
    ap.add_argument("--max-roles", type=int, default=14)
    ap.add_argument("--skip-none", action="store_true",
                    help="skip documents the type pass labelled `none` (~16.9%%). Off by "
                         "default: they still carry dates, places and tolls, and GENERIC_ROLES "
                         "covers them.")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--fetch-batch")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = ap.parse_args()

    menus = role_menus(args.en_corpus, args.min_role_share, args.max_roles)
    sizes = sorted(len(v) for v in menus.values())
    print(f"[ent-ann] role menus for {len(menus)} event types; "
          f"roles/type min {sizes[0]} median {sizes[len(sizes)//2]} max {sizes[-1]}")

    cands = load_typed([Path(p) for p in args.typed], args.limit, args.skip_none)
    by_label = Counter(c["label"] for c in cands)
    n_none = by_label.get(NONE_LABEL, 0)
    print(f"[ent-ann] {len(cands):,} typed documents "
          f"({n_none:,} labelled `{NONE_LABEL}`, using GENERIC_ROLES)")

    provider = AnthropicProvider(ProviderConfig(
        provider="anthropic", model=args.model, max_tokens=900, json_object=True))
    items, allowed_by_id = [], {}
    for i, c in enumerate(cands):
        roles = menus.get(c["label"]) or GENERIC_ROLES
        allowed_by_id[str(i)] = set(roles)
        items.append((str(i), SYSTEM, USER.format(
            etype=c["label"] if c["label"] != NONE_LABEL else "news",
            roles="\n".join(f"- {r}" for r in roles), text=c["text"])))

    if args.fetch_batch:
        replies = provider.fetch_batch(args.fetch_batch)
    elif args.batch:
        replies = provider.complete_batch(items)
    else:
        replies = {cid: provider.complete(s, u) for cid, s, u in items}

    rows, unusable, dropped_spans, empty = [], 0, 0, 0
    for cid, raw in replies.items():
        c = cands[int(cid)]
        got = parse(raw, allowed_by_id[cid], c["text"])
        if got is None:
            unusable += 1
            continue
        ents, dropped = got
        dropped_spans += dropped
        if not ents:
            empty += 1
        rows.append({"input": c["text"], "label": c["label"], "entities": ents,
                     "source": c["source"], "stratum": c["stratum"]})

    out = Path(f"{args.out}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(dumps_record(r) + "\n")

    role_counts = Counter(k for r in rows for k in r["entities"])
    span_total = sum(len(v) for r in rows for v in r["entities"].values())
    print(f"[ent-ann] {len(rows):,} rows -> {out} ({unusable} unusable)")
    print(f"[ent-ann] {span_total:,} verbatim spans over {len(role_counts)} roles; "
          f"{dropped_spans:,} dropped as non-verbatim or off-menu")
    print(f"[ent-ann] documents with NO extractable role: {empty:,} "
          f"({empty / max(len(rows), 1):.1%})")
    for role, n in role_counts.most_common(12):
        print(f"[ent-ann]   {role:34s} {n:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
