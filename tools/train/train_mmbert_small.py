"""Train a fresh GLiNER2 on top of jhu-clsp/mmBERT-small.

Run::

    uv run python tools/train/train_mmbert_small.py

Expects each corpus split into three JSONL files under ``data/`` by the
``tools/data/`` converters — ``<name>.train.jsonl``, ``<name>.val.jsonl``,
``<name>.test.jsonl``. The train splits feed ``trainer.train``, the val
splits drive end-of-epoch eval with the F1 hook, and the test splits are
held out for a blind pass on the best checkpoint after training finishes.

Edit the ``CORPORA`` list below to drop a corpus or point at smaller subsets.

Event-extraction corpora (ACE 2005, MAVEN, RAMS) are listed separately in
``EVENT_FILES`` because they ship canonical train/dev/test splits and
require manual download (see TRAINING.md §2). Each entry is included only
if the file exists on disk, so the script runs cleanly with any subset of
event corpora present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from gliner2 import GLiNER2
from gliner2.training import estimate_eta, evaluate_checkpoint, make_compute_metrics
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


CORPORA: List[str] = [
    "data/nuner_full",
    "data/pile_ner_def",
    "data/knowledgator_gliner",
    "data/text2json",
    "data/gliner_multilingual",
    "data/gliclass_logic",
    "data/scientific_text",
    "data/biomed_ner",
    "data/events_biotech",
    "data/sentence_rex",
    "data/bio_ner_relations",
    "data/pubmed_abstracts_ner",
]

# Event-extraction corpora — emitted by tools/data/convert_{ace2005,maven,rams}.py.
# Files are picked up only if they exist on disk; absent ones are silently
# skipped so the script works with any subset (or none) of these.
EVENT_FILES: Dict[str, Dict[str, str]] = {
    "rams":    {"train": "data/rams.train.jsonl",
                "val":   "data/rams.dev.jsonl",
                "test":  "data/rams.test.jsonl"},
    "maven":   {"train": "data/maven.train.jsonl",
                "val":   "data/maven.valid.jsonl"},
    # ACE 2005: convert_ace2005.py now emits a stratified 80/10/10
    # train/test/val split by default (greedy multi-label rule covering
    # entity types, relation types, and event types). Each split is
    # picked up here only if its file is present on disk.
    "ace2005": {"train": "data/ace2005.train.jsonl",
                "val":   "data/ace2005.val.jsonl",
                "test":  "data/ace2005.test.jsonl"},
}


def _split_files(corpora: List[str], suffix: str) -> List[str]:
    return [f"{c}.{suffix}.jsonl" for c in corpora]


def _event_split(suffix: str) -> List[str]:
    paths: List[str] = []
    for corpus, by_split in EVENT_FILES.items():
        p = by_split.get(suffix)
        if p and Path(p).is_file():
            paths.append(p)
    return paths


TRAIN_DATA = _split_files(CORPORA, "train") + _event_split("train")
EVAL_DATA = _split_files(CORPORA, "val") + _event_split("val")
TEST_DATA = _split_files(CORPORA, "test") + _event_split("test")


def main() -> None:
    model = GLiNER2.from_encoder("jhu-clsp/mmBERT-small", max_width=12, max_len=8192)

    config = TrainingConfig(
        output_dir="./out/mmbert-small",
        experiment_name="mmbert_small_multi_corpus",
        num_epochs=3,
        batch_size=2,
        gradient_accumulation_steps=2,
        encoder_lr=2e-5,
        task_lr=5e-4,
        warmup_ratio=0.05,
        scheduler_type="cosine_restarts",
        bf16=True,
        fp16=False,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        metric_for_best="eval_loss",
        greater_is_better=False,
        save_best=True,
        save_total_limit=3,
        logging_steps=50,
        num_workers=0,        # was 4; per-worker tokenizer copies + MPS unified
                              # memory triggered OOM-kills of DataLoader workers on
                              # macOS (single leaked semaphore at shutdown).
        validate_data=False,
        max_len=2048,         # was 8192; most records are well under 2048 word-
                              # tokens, 8192 was ~4x padding cost for no signal.
                              # `from_encoder(max_len=8192)` above keeps the
                              # positional embeddings, so inference still handles 8k.
        pin_memory=False,     # CUDA-only hint; no-op on MPS.
    )

    trainer = GLiNER2Trainer(
        model, config,
        eval_data=EVAL_DATA,
        compute_metrics=make_compute_metrics(batch_size=8, threshold=0.5),
    )
    estimate_eta(model, TRAIN_DATA, config)
    trainer.train(train_data=TRAIN_DATA)

    best = Path(config.output_dir) / "best"
    if not best.is_dir():
        print(f"\n[blind test] No 'best' checkpoint at {best}; skipping.")
        return

    print(f"\n[blind test] Loading {best} and scoring against {len(TEST_DATA)} held-out splits...")
    test_metrics = evaluate_checkpoint(best, TEST_DATA, batch_size=8, threshold=0.5)
    if not test_metrics:
        print("[blind test] No metrics produced (empty test set?).")
        return

    print("\n===== Blind test metrics =====")
    for key in sorted(test_metrics):
        val = test_metrics[key]
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        elif isinstance(val, int):
            print(f"  {key}: {val}")

    for category in ("entity", "relation", "classification", "event_trigger", "event_argument"):
        report_key = f"eval_{category}_classification_report"
        if report_key in test_metrics:
            print(f"\n--- {category} classification report ---")
            print(test_metrics[report_key])


if __name__ == "__main__":
    main()
