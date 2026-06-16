"""Train a fresh GLiNER2 on top of jhu-clsp/mmBERT-base.

Run::

    uv run python tools/train/train_mmbert_base.py

Expects each corpus split into three JSONL files under ``data/`` by the
``tools/data/`` converters — ``<name>.train.jsonl``, ``<name>.val.jsonl``,
``<name>.test.jsonl``. The train splits feed ``trainer.train``, the val
splits drive end-of-epoch eval with the F1 hook, and the test splits are
held out for a blind pass on the best checkpoint after training finishes.

Edit the ``CORPORA`` list below to drop a corpus or point at smaller subsets.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from gliner2 import GLiNER2
from gliner2.training import evaluate_checkpoint, make_compute_metrics
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
]


def _split_files(corpora: List[str], suffix: str) -> List[str]:
    return [f"{c}.{suffix}.jsonl" for c in corpora]


TRAIN_DATA = _split_files(CORPORA, "train")
EVAL_DATA = _split_files(CORPORA, "val")
TEST_DATA = _split_files(CORPORA, "test")


def main() -> None:
    model = GLiNER2.from_encoder("jhu-clsp/mmBERT-base", max_width=12, max_len=8192)

    config = TrainingConfig(
        output_dir="./out/mmbert-base",
        experiment_name="mmbert_base_multi_corpus",
        num_epochs=2,
        batch_size=4,
        gradient_accumulation_steps=2,
        encoder_lr=1e-5,
        task_lr=3e-4,
        warmup_ratio=0.05,
        scheduler_type="cosine",
        bf16=True,
        fp16=False,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        metric_for_best="eval_loss",
        greater_is_better=False,
        save_best=True,
        save_total_limit=3,
        logging_steps=50,
        num_workers=4,
        validate_data=False,
        max_len=8192,
    )

    trainer = GLiNER2Trainer(
        model, config,
        eval_data=EVAL_DATA,
        compute_metrics=make_compute_metrics(batch_size=8, threshold=0.5),
    )
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

    for category in ("entity", "relation", "classification"):
        report_key = f"eval_{category}_classification_report"
        if report_key in test_metrics:
            print(f"\n--- {category} classification report ---")
            print(test_metrics[report_key])


if __name__ == "__main__":
    main()
