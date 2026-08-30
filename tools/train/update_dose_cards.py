"""Append the dose-curve context and the honest evaluation caveat to each arm's card.

The generated card advertises "entity, relation, event and classification extraction"
because the BASE does those, while this run evaluated only json_structures -- the val set
was built from the casualty corpora alone. A reader takes the tags at face value.
"""
import os
from huggingface_hub import HfApi, hf_hub_download

CURVE = {
    0:     ("control", 0, 18109, 0.912, 0.227, 0.363),
    5000:  ("dose",    5000, 25218, 0.851, 0.549, 0.667),
    15000: ("dose",   15000, 39431, 0.814, 0.609, 0.697),
    31263: ("dose",   29700, 60283, 0.842, 0.631, 0.722),
}

TABLE = "\n".join(
    f"| {'**control**' if d == 0 else f'{tr:,}'} | {tr:,} | {p:.3f} | {r:.3f} | **{f:.3f}** |"
    for d, (_, tr, _, p, r, f) in CURVE.items()
)


def section(dose: int) -> str:
    kind, tr, total, p, r, f = CURVE[dose]
    role = ("the **CONTROL** arm: zero Turkish, otherwise identical to every dose arm"
            if dose == 0 else f"a dose arm carrying **{tr:,} Turkish** casualty records")
    return f"""

---

## What this model is

One arm of a **Turkish dose curve** for the EKF casualty extractor. This is {role}.

Held constant across all four arms: 12,691 English casualty records, plus 30% *exact*
replay sampled from this base's own 137k training pool. Only the Turkish count varies, so
a difference between arms is attributable to Turkish data and nothing else.

The control exists because every dose arm adds replay **and** Turkish at once. Without it,
an improvement over the shipped extractor could not be attributed to either.

| arm | Turkish rows | precision | recall | F1 (strict) |
|---|--:|--:|--:|--:|
{TABLE}

Turkish data moves **recall** (0.227 → 0.631) far more than precision (0.912 → 0.842):
the untrained model was not wrong about Turkish so much as blind to it. Returns diminish
sharply — the first 5,000 records buy +0.304 F1, the remaining 24,700 buy +0.055.

## ⚠️ What was NOT measured

**Only `json_structures` was evaluated.** The val and test splits were built from the
casualty corpora alone, so they contain 1,115 structure rows and nothing else. The card's
task list and tags above are inherited from the base model, which does entity, relation,
event and classification extraction — **this run measured none of them.**

That matters most for the replay. Its purpose is to stop the narrow fine-tune destroying
capabilities it is not training on — the failure that made the previous casualty model
return a digit when asked for a `location`. Whether the replay succeeded here is
**unmeasured**: the eval could not observe forgetting. Treat non-structure capability as
inherited-but-unverified until it is scored directly.

Fixed for later runs in `tools/train/build_turkish_dose_mix.py`, which now puts a replay
slice in val so all five categories are reported.
"""


def main() -> int:
    api = HfApi()
    token = os.environ["HF_TOKEN"]
    for dose in CURVE:
        repo = f"whr778/gliner2-tr-dose-{dose}"
        path = hf_hub_download(repo, "README.md", token=token)
        text = open(path, encoding="utf-8").read()
        marker = "## What this model is"
        if marker in text:
            text = text[: text.index("\n---\n\n" + marker)]
        out = text.rstrip() + section(dose)
        api.upload_file(path_or_fileobj=out.encode("utf-8"), path_in_repo="README.md",
                        repo_id=repo, commit_message="card: dose-curve context and evaluation caveat")
        print(f"  updated {repo} ({len(out)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
