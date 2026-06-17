"""Train a fresh GLiNER2 on top of jhu-clsp/mmBERT-base.

Run::

    uv run python tools/train/train_mmbert_base.py

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
    "data/gliclass_rac",
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
    "wikievents": {"train": "data/wikievents.train.jsonl",
                   "val":   "data/wikievents.dev.jsonl",
                   "test":  "data/wikievents.test.jsonl"},
    "casie":      {"train": "data/casie.train.jsonl",
                   "val":   "data/casie.val.jsonl",
                   "test":  "data/casie.test.jsonl"},
    # DocEE — manual Google Drive download; entity- and classification-shaped
    # rather than events-shaped (no triggers in the source).
    "docee":      {"train": "data/docee.train.jsonl",
                   "val":   "data/docee.val.jsonl",
                   "test":  "data/docee.test.jsonl"},
    # CMNEE — Chinese military event extraction (triggers + typed args).
    # Manual Google Drive download required.
    "cmnee":      {"train": "data/cmnee.train.jsonl",
                   "val":   "data/cmnee.val.jsonl",
                   "test":  "data/cmnee.test.jsonl"},
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
    # Structure-loss variant (events/relations/entities span scoring):
    #   "bce"           - plain BCE-with-logits (default, original behavior)
    #   "bce_posweight" - BCE up-weighting positive spans by struct_pos_weight
    #   "focal"         - focal loss (focal_gamma / focal_alpha)
    # focal / bce_posweight handle class imbalance via the loss itself, so the
    # random negative masking in compute_struct_loss becomes partly redundant.
    model = GLiNER2.from_encoder(
        "jhu-clsp/mmBERT-base",
        max_width=20,
        max_len=8192,
        struct_loss="focal",
        struct_pos_weight=8.0,
        focal_gamma=2.0,
        focal_alpha=0.25,
    )

    config = TrainingConfig(
        output_dir="./out/mmbert-base",
        experiment_name="mmbert_base_multi_corpus",
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=2,
        encoder_lr=1e-5,
        task_lr=3e-4,
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
        num_workers=0,
        # num_workers=4,
        validate_data=False,
        max_len=8192,
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
