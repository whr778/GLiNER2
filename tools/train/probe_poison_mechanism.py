"""Where a poison role acts: trigger max p per event type under menu variants, plus a query/text state swap.

    uv run tools/train/build_schema.py data/casie --exhaustive --out casie_full.json   # the menu WITH Damage-Amount
    uv run tools/train/probe_poison_mechanism.py whr778/gliner2-eb21 casie_full.json --role Damage_Amount \\
        --types Cyber.Phishing Cyber.Ransom --add Damage Ballot Vehicle Amount --swap

Runs CASIE val docs 1..``--docs`` through the real decode (global decode, threshold 0.3) and prints, per variant,
the best trigger-span probability of each ``--types`` event type:
  ``no role``    the menu without ``--role`` (matched across label styles: ``Damage-Amount`` == ``Damage_Amount``);
  ``original``   the menu as built;
  ``+X``         the menu without ``--role``, plus role X on ``--types`` only.
``--swap`` re-scores the boundary head with the text states of one arm and the query states of the other, which
separates "the role changes how the text is read" from "the role changes the trigger query".
Measured 2026-10-17: the role acts through the trigger QUERY of its own event type only; the dose of absent
event types in training (``negative_labels_per_dim``) scales it -- see the catalog row of that date.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gliner2 import AutoExtractor
from gliner2.inference.label_map import apply_label_map
from gliner2.models.boundary import engine as E

CAP: list = []


def _capture() -> None:
    """Keep each forward of ``_extract_from_batch`` so the probe reads the real query/text states and probs."""
    orig = E.BoundaryExtractor._extract_from_batch

    def hooked(self, batch, threshold, metadata_list, *a, **k):
        k["forward"] = k.get("forward") or self._forward_batch(batch, threshold, metadata_list, k.get("hidden_states"))
        CAP.append(k["forward"])
        return orig(self, batch, threshold, metadata_list, *a, **k)
    E.BoundaryExtractor._extract_from_batch = hooked


def forward(model, text: str, menu: dict) -> dict:
    CAP.clear()
    with torch.no_grad():
        model.batch_extract_long([text], [menu], batch_size=1, threshold=0.3, chunk_size=4096,
                                 chunk_overlap=0, global_decode=True)
    return CAP[0]


def triggers(fwd: dict, types: list[str]) -> dict:
    """Best trigger-span probability per event type."""
    specs, probs = fwd["core"]["ext_specs"][0], fwd["probs"][0]
    return {s["task_name"]: probs[q].max().item() for q, s in enumerate(specs)
            if s["task_type"] == "events" and s["field_name"] == "trigger" and s["task_name"] in types}


def swap(model, a: dict, b: dict, types: list[str]) -> dict:
    """Trigger p from arm a's TEXT states with arm b's QUERY states (b's queries must be a subset of a's)."""
    sa, sb = a["core"]["ext_specs"][0], b["core"]["ext_specs"][0]
    keys_b = [(s["task_type"], s["task_name"], s["field_name"]) for s in sb]
    with torch.no_grad():
        out = model.boundary_head(a["core"]["text_states"], a["core"]["text_mask"],
                                  b["core"]["query_states"], b["core"]["query_mask"], return_candidates=True)
    probs = torch.sigmoid(out.candidates.pair_logits)[0]
    assert {(s["task_type"], s["task_name"], s["field_name"]) for s in sa} >= set(keys_b)
    return {k[1]: probs[q].max().item() for q, k in enumerate(keys_b)
            if k[0] == "events" and k[2] == "trigger" and k[1] in types}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model")
    ap.add_argument("menu", help="app-menu JSON that still holds --role")
    ap.add_argument("--role", required=True, help="the poison role, any label style")
    ap.add_argument("--types", nargs="+", required=True, help="event types to read and to add --add roles to")
    ap.add_argument("--add", nargs="*", default=[], help="roles to add, one variant each")
    ap.add_argument("--data", default="data/casie.val.jsonl")
    ap.add_argument("--docs", type=int, default=5)
    ap.add_argument("--swap", action="store_true")
    args = ap.parse_args()

    model = AutoExtractor.from_pretrained(args.model, map_location="cpu")
    _capture()
    raw = json.loads(Path(args.menu).read_text(encoding="utf-8"))
    menu, _ = apply_label_map(raw.get("schema", raw), getattr(model.config, "label_map", None),
                              getattr(model.config, "label_style", None))
    norm = args.role.replace("-", "_").casefold()
    ev0 = {t: [r for r in rs if r.replace("-", "_").casefold() != norm] for t, rs in menu["events"].items()}
    assert ev0 != menu["events"], f"{args.role} not in the mapped menu"
    variants = {"no role": {**menu, "events": ev0}, "original": menu}
    variants |= {f"+{r}": {**menu, "events": {t: rs + [r] if t in args.types else rs for t, rs in ev0.items()}}
                 for r in args.add}
    lines = Path(args.data).read_text(encoding="utf-8").splitlines()
    texts = [json.loads(line)["input"] for line in lines[1:args.docs + 1]]
    print(f"[probe] {args.model} | {len(texts)} docs of {args.data} | role {args.role} | trigger max p per doc")
    kept = {}
    for name, m in variants.items():
        rows = {t: [] for t in args.types}
        for i, text in enumerate(texts):
            fwd = forward(model, text, m)
            if args.swap and name in ("no role", "original"):
                kept.setdefault(name, []).append(fwd)
            for t, p in triggers(fwd, args.types).items():
                rows[t].append(p)
        print(f"[probe] {name:18s} " + " | ".join(f"{t} " + " ".join(f"{p:.3f}" for p in ps) for t, ps in rows.items()),
              flush=True)
    if args.swap:
        rows = {t: [] for t in args.types}
        for fa, fb in zip(kept["original"], kept["no role"]):
            for t, p in swap(model, fa, fb, args.types).items():
                rows[t].append(p)
        print("[probe] swap: text of 'original' + queries of 'no role' "
              + " | ".join(f"{t} " + " ".join(f"{p:.3f}" for p in ps) for t, ps in rows.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
