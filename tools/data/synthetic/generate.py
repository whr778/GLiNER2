"""Generate synthetic GLiNER2 base-training data with a configurable LLM.

One document per call, annotated for all requested task types (entities,
relations, document-level events with triggers+arguments, classifications,
structures) using the broad ontology in ``schema_spec.py``. Every span is
verbatim-validated; the output is written straight into train/val/test JSONL
via the shared ``SplitWriter`` and is directly trainable by ``GLiNER2Trainer``.

Examples
--------
    # Estimate cost only -- no API calls, no keys needed:
    uv run python tools/data/synthetic/generate.py --config default.yaml \
        --count 254334 --estimate

    # Dry run the full pipeline with the keyless mock provider:
    uv run python tools/data/synthetic/generate.py --config default.yaml \
        --out data/synthetic.jsonl --count 5 --dry-run

    # Real generation (needs OPENAI_API_KEY or ANTHROPIC_API_KEY in env):
    uv run --with openai python tools/data/synthetic/generate.py \
        --config default.yaml --out data/synthetic.jsonl --count 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/data (for _split)

from _split import SplitWriter, add_split_args  # noqa: E402
from collections import Counter  # noqa: E402

import json  # noqa: E402
import cost as cost_mod  # noqa: E402
from prompts import (  # noqa: E402
    ANNOTATE_SYSTEM, SYSTEM, build_annotate_prompt, build_user_prompt,
)
from providers import ProviderConfig, build_provider  # noqa: E402
from schema_spec import ALL_TASKS, DOMAINS  # noqa: E402
from validate import build_record, parse_reply  # noqa: E402


def _iter_corpus(path: Path):
    """Yield (input_text, gold_output) from an existing GLiNER2 JSONL corpus."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("input")
            if isinstance(text, str) and text.strip():
                yield text, (rec.get("output") or {})


def load_config(path: Path) -> dict:
    """Load the YAML config; resolve a bare name against the config/ dir."""
    if not path.exists() and path.parent == Path("."):
        alt = Path(__file__).resolve().parent / "config" / path.name
        if alt.exists():
            path = alt
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _print_estimate(model: str, count: int, cfg_cost: dict) -> None:
    it = cfg_cost.get("est_input_tokens", cost_mod.DEFAULT_INPUT_TOKENS)
    ot = cfg_cost.get("est_output_tokens", cost_mod.DEFAULT_OUTPUT_TOKENS)
    print(f"Cost estimate for {count:,} records with {model} "
          f"({it} in / {ot} out tokens each):")
    for batch in (False, True):
        est = cost_mod.estimate(model, count, it, ot, batch=batch)
        tier = "batch (-50%)" if batch else "standard"
        usd = est.usd
        amount = "unknown model price" if usd is None else f"${usd:,.2f}"
        print(f"  {tier:<14} {amount}  "
              f"({est.total_input/1e6:.1f}M in + {est.total_output/1e6:.1f}M out tokens)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("default.yaml"),
                    help="YAML config (bare name resolves to synthetic/config/).")
    ap.add_argument("--out", type=Path, help="Output JSONL base path (writes .train/.val/.test).")
    ap.add_argument("--count", type=int, help="Number of documents to generate.")
    ap.add_argument("--provider", help="Override provider (openai|anthropic|mock).")
    ap.add_argument("--model", help="Override model name.")
    ap.add_argument("--tasks", help="Comma-separated task subset (default: all).")
    ap.add_argument("--limit", type=int, help="Alias for a small --count (smoke run).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Use the keyless mock provider (no API calls, no spend).")
    ap.add_argument("--batch", action="store_true",
                    help="Submit all documents via the provider's Batch API (-50%% "
                         "pricing, async). Anthropic only; mock supports it for dry runs.")
    ap.add_argument("--estimate", action="store_true",
                    help="Print the cost estimate and exit (no generation).")
    ap.add_argument("--annotate-from", type=Path,
                    help="Annotate the real text in this existing GLiNER2 JSONL "
                         "instead of generating new documents (paper's real-text "
                         "half). --count caps how many rows to annotate.")
    ap.add_argument("--annotate-replace", action="store_true",
                    help="Replace existing gold annotations instead of merging.")
    add_split_args(ap)
    args = ap.parse_args()

    cfg = load_config(args.config)
    gen_cfg = cfg.get("generation", {})
    prov_cfg = cfg.get("provider", {})
    cost_cfg = cfg.get("cost", {})

    count = args.limit or args.count or gen_cfg.get("count", 100)
    model = args.model or prov_cfg.get("model", ProviderConfig.model)
    provider_name = args.provider or prov_cfg.get("provider", ProviderConfig.provider)
    if args.dry_run:
        provider_name = "mock"

    tasks = (args.tasks.split(",") if args.tasks else gen_cfg.get("tasks", ALL_TASKS))
    tasks = [t.strip() for t in tasks if t.strip() in ALL_TASKS]
    min_words = gen_cfg.get("min_words", 180)
    max_words = gen_cfg.get("max_words", 320)
    domains = gen_cfg.get("domains") or DOMAINS

    if args.estimate:
        _print_estimate(model, count, cost_cfg)
        return 0

    if not args.out:
        ap.error("--out is required unless --estimate is set")

    pcfg = ProviderConfig(
        provider=provider_name, model=model,
        temperature=prov_cfg.get("temperature", ProviderConfig.temperature),
        max_tokens=prov_cfg.get("max_tokens", ProviderConfig.max_tokens),
    )
    provider = build_provider(pcfg)
    annotate = args.annotate_from is not None
    mode = f"annotate-from {args.annotate_from}" if annotate else "generate"
    print(f"Provider={provider_name} model={model} tasks={tasks} count={count} mode={mode}")
    _print_estimate(model, count, cost_cfg)

    def _jobs():
        """Yield (system, user_prompt, text_override, base_output) per document."""
        if annotate:
            for i, (text, gold) in enumerate(_iter_corpus(args.annotate_from)):
                if i >= count:
                    break
                base = None if args.annotate_replace else gold
                yield ANNOTATE_SYSTEM, build_annotate_prompt(text, tasks), text, base
        else:
            for i in range(count):
                domain = domains[i % len(domains)]
                yield SYSTEM, build_user_prompt(domain, tasks, min_words, max_words), None, None

    stats: Counter = Counter()
    written = 0
    failed = 0

    def _record_from(raw, text_override, base_output):
        """Parse a raw reply into a GLiNER2 record (or None); updates stats."""
        reply = parse_reply(raw)
        if reply is None:
            stats["parse_error"] += 1
            return None
        return build_record(reply, tasks, stats,
                            text_override=text_override, base_output=base_output)

    # SplitWriter writes normalized UTF-8 JSONL (dumps_record: NFKC,
    # ensure_ascii=False, encoding=utf-8) for both the sync and batch paths.
    with SplitWriter(args.out, ratios=args.split_ratios, seed=args.split_seed) as writer:
        if args.batch:
            jobs = list(_jobs())
            meta = {f"doc-{i}": (t, b) for i, (_, _, t, b) in enumerate(jobs)}
            items = [(f"doc-{i}", system, user)
                     for i, (system, user, _, _) in enumerate(jobs)]
            replies = provider.complete_batch(items)
            failed = len(items) - len(replies)  # requests that errored/expired
            for cid, raw in replies.items():
                text_override, base_output = meta[cid]
                record = _record_from(raw, text_override, base_output)
                if record is None:
                    failed += 1
                    continue
                writer.write(record)
                written += 1
        else:
            for i, (system, user, text_override, base_output) in enumerate(_jobs()):
                try:
                    raw = provider.complete(system, user)
                except Exception as e:  # network / API errors: log and continue
                    failed += 1
                    print(f"[{i}] API error: {e}", file=sys.stderr)
                    continue
                record = _record_from(raw, text_override, base_output)
                if record is None:
                    failed += 1
                    continue
                writer.write(record)
                written += 1
                if written % 50 == 0:
                    print(f"  ...{written} written ({failed} failed)")
        summary = writer.summary()

    print(f"\nDone. written={written} failed={failed}")
    print("Kept/dropped per task:")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
