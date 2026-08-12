"""Acceptance gate for `augment_baseword.py`: augmentation must not change gold.

Run this before training any augmented arm. It compares each augmented copy against the
record it was derived from and requires an EXACT match in the number of gold mentions
collation builds. See TODO.md item 12.

    uv run python tools/data/check_augment_alignment.py \
        --original data/rams.train.jsonl --augmented data/rams_aug.train.jsonl

Three things decide whether this measures anything, and each cost a wrong answer first:

1. **`missing_surface_counts()` cannot serve as the gate.** It increments only for
   `task_type == "entities"` (`boundary_preprocessing.py:443`). RAMS supervises `events`,
   and for non-entity types an unlocatable surface is treated as legitimately absent and
   skipped with NO counter at all (`:465`) -- so a copy could lose every argument and the
   counter would still read 0. What IS observable is the target graph:
   `targets.mention_mask.sum()` is the gold the collator actually built.

2. **Sampling must be off.** `collate_fn_train` sets `is_training=True`, and the default
   `remove_events_prob=0.2` drops the whole event group a fifth of the time -- one record
   collated ten times gives [5,0,0,5,5,5,5,5,5,0]. Comparing under that measures noise.

3. **Gold can be INVENTED, not just lost.** Lemmatization collapses surface forms, so a
   label starts matching positions that were never annotated. This gate is two-sided for
   that reason: any change fails, in either direction.

Exit status is 0 on PASS and 1 on FAIL, so it can gate a pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gliner2.processor import SamplingConfig, SchemaTransformer

# Every stochastic knob off: this is a measurement, not a training pass.
DETERMINISTIC = SamplingConfig(
    remove_json_structure_prob=0.0, shuffle_json_fields=False, remove_json_field_prob=0.0,
    remove_entities_prob=0.0, shuffle_entities=False, remove_entity_prob=0.0,
    synthetic_entity_label_prob=0.0,
    remove_relations_prob=0.0, swap_head_tail_prob=0.0,
    remove_events_prob=0.0, shuffle_event_roles=False,
    remove_classification_prob=0.0, shuffle_classification_labels=False,
    remove_classification_label_prob=0.0, synthetic_label_prob=0.0,
    include_true_label_prob=1.0,
)


def load(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def gold_mentions(proc: SchemaTransformer, record: dict) -> int:
    """How many gold mentions collation builds for this record."""
    batch = proc.collate_fn_train(
        [(record["input"], record["output"])],
        error_policy="skip", architecture="boundary",
    )
    return 0 if batch.targets is None else int(batch.targets.mention_mask.sum())


def compare(proc: SchemaTransformer, originals: list[dict], augmented: list[dict]):
    """Pair each augmented copy with its source and total the gold on both sides.

    The augmented file interleaves `original, [copy], original, [copy], ...` and omits the
    copy for refused records, so pairing is by membership in the source set rather than by
    index -- an index walk silently mis-pairs from the first refusal onward.
    """
    by_text = {r["input"]: r for r in originals}
    source, pairs, from_source, from_copy, changed = None, 0, 0, 0, []
    for record in augmented:
        if record["input"] in by_text:
            source = record
            continue
        if source is None:
            continue
        before, after = gold_mentions(proc, source), gold_mentions(proc, record)
        pairs += 1
        from_source += before
        from_copy += after
        if before != after:
            changed.append((source["input"][:60], before, after))
    return pairs, from_source, from_copy, changed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", required=True, help="the un-augmented corpus")
    ap.add_argument("--augmented", required=True, help="output of augment_baseword.py")
    ap.add_argument("--tokenizer", default="fastino/gliner2-base-v1")
    ap.add_argument("--show", type=int, default=5, help="how many changed records to print")
    args = ap.parse_args(argv)

    proc = SchemaTransformer(model_name=args.tokenizer, sampling_config=DETERMINISTIC)
    pairs, before, after, changed = compare(
        proc, load(args.original), load(args.augmented))

    print(f"[gate] augmented pairs checked : {pairs}")
    print(f"[gate] gold mentions, originals: {before}")
    print(f"[gate] gold mentions, copies   : {after}")
    print(f"[gate] delta                   : {after - before}")
    if changed:
        print(f"\n[gate] records whose gold changed: {len(changed)}")
        for text, a, b in changed[:args.show]:
            print(f"  {a} -> {b}  {text!r}")
    ok = before == after and not changed
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
