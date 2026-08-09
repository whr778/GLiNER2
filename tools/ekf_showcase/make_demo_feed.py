"""Build a mixed-topic, time-ordered news feed for the EKF showcase.

The feed interleaves:

* **on-topic** snippets from one `disaster_streams_sonnet5` stream — realized
  multi-fact disaster text that states figures with hedges and *distractor numbers*
  (dates, magnitudes, displaced counts), each carrying its ground-truth
  ``(role, value, qualifier, source)`` so the chart can show truth beside estimates;
* **off-topic** synthetic articles (sports, markets, tech, weather) that a real feed
  would carry and the pipeline must reject.

Ground truth rides along under ``_gt`` (and ``_truth`` for the trajectory) so the
pipeline can be scored without a second file. A production feed would simply omit it.

    uv run python tools/ekf_showcase/make_demo_feed.py \
        --split test --stream test-00000 \
        --out datasets/ekf_showcase/feed.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Off-topic filler. Deliberately includes NUMBERS and a couple of casualty-adjacent
# words ("crash", "victims of a hoax") so the relevance gate has to do real work rather
# than keying on the presence of digits.
DISTRACTORS = [
    "Markets closed higher on Tuesday, with the index up 142 points after stronger than expected retail figures.",
    "The home side won 3-1, their fourth straight victory, in front of a crowd of 41,200.",
    "A new smartphone launched today with a 200 megapixel camera and a price tag of 899 dollars.",
    "Rainfall of 38 millimetres is forecast for the weekend, with temperatures near 21 degrees.",
    "Regulators fined the firm 4.2 million dollars over billing practices affecting 30,000 customers.",
    "The airline said a computer crash delayed 87 flights, though no injuries were reported.",
    "Police warned that victims of a phone hoax had lost some 12,000 dollars in total.",
    "Researchers surveyed 1,500 households about commuting habits across the metropolitan area.",
    "A vintage car sold at auction for 3.1 million dollars, a record for the model.",
    "The council approved 250 new housing units near the eastern rail corridor.",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="datasets/disaster_streams_sonnet5")
    ap.add_argument("--split", default="test")
    ap.add_argument("--stream", default=None, help="stream_id; default = first found")
    ap.add_argument("--out", default="datasets/ekf_showcase/feed.jsonl")
    ap.add_argument("--distractor-ratio", type=float, default=1.0,
                    help="off-topic articles per on-topic one (1.0 = half the feed is noise)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    split = Path(args.data) / args.split
    obs = [json.loads(l) for l in (split / "observations.jsonl").open(encoding="utf-8") if l.strip()]
    obs = [o for o in obs if o.get("text")]
    if not obs:
        raise SystemExit(f"no realized text in {split}/observations.jsonl")

    stream = args.stream or obs[0]["stream_id"]
    on_topic = sorted((o for o in obs if o["stream_id"] == stream), key=lambda o: o["t_hours"])
    if not on_topic:
        raise SystemExit(f"stream {stream!r} not found")

    traj = []
    traj_path = split / "trajectory.jsonl"
    if traj_path.is_file():
        traj = [json.loads(l) for l in traj_path.open(encoding="utf-8") if l.strip()]
        traj = sorted((t for t in traj if t["stream_id"] == stream), key=lambda t: t["t_hours"])

    rng = random.Random(args.seed)
    lines = []
    for o in on_topic:
        lines.append({
            "t_hours": o["t_hours"],
            "text": o["text"],
            # Ground truth for scoring; a real feed would not have this.
            "_gt": {k: o[k] for k in ("role", "value", "qualifier", "source") if k in o},
        })

    span = (on_topic[-1]["t_hours"] - on_topic[0]["t_hours"]) or 1.0
    for i in range(int(len(on_topic) * args.distractor_ratio)):
        lines.append({
            "t_hours": round(on_topic[0]["t_hours"] + rng.random() * span, 2),
            "text": DISTRACTORS[i % len(DISTRACTORS)],
            "_gt": None,
        })

    lines.sort(key=lambda r: r["t_hours"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in lines:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # The reference trajectory travels beside the feed, not inside it.
    if traj:
        tp = out.with_name(out.stem + ".truth.jsonl")
        with tp.open("w", encoding="utf-8") as fh:
            for t in traj:
                fh.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"wrote {tp}  ({len(traj)} trajectory points)")

    n_on = len(on_topic)
    print(f"wrote {out}")
    print(f"  stream      : {stream}")
    print(f"  articles    : {len(lines)}  ({n_on} on-topic, {len(lines) - n_on} distractors)")
    print(f"  time span   : {lines[0]['t_hours']}h .. {lines[-1]['t_hours']}h")


if __name__ == "__main__":
    main()
