"""Sonnet-5 text realizer: turn structured observations into realistic news snippets.

Groups each stream's observations by report time -> ONE multi-fact snippet per report
(several roles + distractor numbers), so extraction must BIND each figure to the right
role amid dates/magnitudes/other figures. The conditioning tuple (role, value, qualifier,
source) stays the known ground truth: the snippet states the EXACT digits with the hedge,
so a correct extractor recovers the tuple. This is the discriminating test the templated
text can't give (design sec 16). Reuses tools/data/synthetic providers + batch pricing.

Safe by default (--provider mock = no spend). Real run:
  ANTHROPIC_API_KEY=... uv run --with anthropic python datasets/disaster_streams/realize.py \
      --split val --provider anthropic --model claude-sonnet-5 --out datasets/disaster_streams_sonnet5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, "tools/data/synthetic")
import cost  # noqa: E402
import providers  # noqa: E402

QUAKE_DAY = date(2026, 6, 24)  # Venezuela 2026 anchor (design sec 12)
_SRC_DESC = {"official": "an official government or UN source",
             "major_outlet": "a major news outlet",
             "preliminary": "early, unconfirmed reports"}
_HEDGE = {"point": "a confirmed, exact count", "at_least": "a lower bound ('at least')",
          "about": "an approximate figure ('about'/'roughly')",
          "feared": "an alarming estimate ('feared')",
          "interval": "a rough figure stated with a bucket word (dozens/hundreds/thousands)"}

SYSTEM = (
    "You are a newswire editor covering an ongoing earthquake disaster. Given the "
    "structured facts of ONE report, write a single realistic news snippet of 2-4 "
    "sentences that states each casualty figure naturally with its hedge, attributed to "
    "the source. Rules: (1) use the EXACT digits given for each figure -- you may add "
    "hedge words but never change a number; (2) weave in the listed distractor details "
    "(date, magnitude, displaced count) as a real report would; (3) do NOT invent extra "
    "casualty figures for the given roles. Reply with a single JSON object: {\"text\": \"...\"}."
)


def _load_groups(split_dir: Path, n_streams: int) -> List[Tuple[str, str, float, List[Dict]]]:
    by_stream: Dict[str, Dict[float, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for line in (split_dir / "observations.jsonl").open(encoding="utf-8"):
        o = json.loads(line)
        by_stream[o["stream_id"]][o["t_hours"]].append(o)
    groups = []
    for sid in sorted(by_stream)[:n_streams] if n_streams else sorted(by_stream):
        for t in sorted(by_stream[sid]):
            groups.append((f"g{len(groups)}", sid, t, by_stream[sid][t]))  # batch-safe id
    return groups


def _user_prompt(t_hours: float, obs: List[Dict]) -> str:
    d = QUAKE_DAY + timedelta(days=int(t_hours // 24))
    src = obs[0]["source"]
    facts = [{"role": o["role"], "value": o["value"], "hedge": o["qualifier"],
              "hedge_meaning": _HEDGE[o["qualifier"]]} for o in obs]
    displaced = int(max(o["value"] for o in obs) * 2.5) + 500  # a distractor, not a role value
    return json.dumps({
        "date": d.isoformat(),
        "source": _SRC_DESC[src],
        "casualty_facts": facts,
        "distractors_to_include": {"date": d.isoformat(), "magnitude": 7.5,
                                   "displaced_people": displaced},
    }, ensure_ascii=False)


def _parse_text(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").split("\n", 1)[-1]
    try:
        return json.loads(raw).get("text", "").strip()
    except json.JSONDecodeError:
        return ""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/disaster_streams")
    ap.add_argument("--split", default="val")
    ap.add_argument("--streams", type=int, default=0, help="0 = all streams in the split")
    ap.add_argument("--provider", default="mock", help="mock (no spend) | anthropic")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--out", default="datasets/disaster_streams_sonnet5")
    ap.add_argument("--estimate", action="store_true", help="print cost estimate and exit")
    args = ap.parse_args(argv)

    src_dir = Path(args.data) / args.split
    groups = _load_groups(src_dir, args.streams)
    items = [(cid, SYSTEM, _user_prompt(t, obs)) for cid, _, t, obs in groups]
    in_tok = int(sum(len(s) + len(u) for _, s, u in items) / 4 / len(items)) if items else 0
    est = cost.estimate(args.model, len(items), input_tokens=in_tok,
                        output_tokens=args.max_tokens, batch=True)
    print(f"[realize] {len(groups)} report-groups from {args.split} "
          f"(~{in_tok} in / {args.max_tokens} out tok each); "
          f"batch est ${est.usd:.2f} on {args.model}")
    if args.estimate:
        return

    cfg = providers.ProviderConfig(provider=args.provider, model=args.model,
                                   max_tokens=args.max_tokens, json_object=False)
    provider = providers.build_provider(cfg)
    texts = provider.complete_batch(items)  # {custom_id: raw_json}

    out_dir = Path(args.out) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    written = miss = 0
    with (out_dir / "observations.jsonl").open("w", encoding="utf-8") as f:
        for cid, sid, t, obs in groups:
            text = _parse_text(texts.get(cid, ""))
            if not text:
                miss += 1
                continue
            for o in obs:
                f.write(json.dumps({**o, "text": text}, ensure_ascii=False) + "\n")
                written += 1
    # ground truth is unchanged -- copy the trajectory for the realized streams only
    sids = {sid for _, sid, _, _ in groups}
    with (out_dir / "trajectory.jsonl").open("w", encoding="utf-8") as tf:
        for line in (src_dir / "trajectory.jsonl").open(encoding="utf-8"):
            if json.loads(line)["stream_id"] in sids:
                tf.write(line)
    print(f"[realize] wrote {written} obs ({miss} groups missing text) -> {out_dir}")


if __name__ == "__main__":
    main()
