"""Train a fresh GLiNER2 on top of jhu-clsp/mmBERT-base.

Run::

    uv run python tools/train/train_mmbert_base.py

Expects these JSONLs under ``data/`` (see ``tools/data/`` converters):
    - nuner_full.jsonl                   (numind/NuNER)
    - pile_ner_def.jsonl                 (Universal-NER/Pile-NER-definition)
    - knowledgator_gliner.jsonl          (knowledgator/GLINER-multi-task-synthetic-data)
    - text2json.jsonl                    (knowledgator/text2json-training-data)

Edit ``TRAIN_DATA`` below to drop a corpus or to point at smaller subsets.
"""

from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


TRAIN_DATA = [
    "data/nuner_full.jsonl",
    "data/pile_ner_def.jsonl",
    "data/knowledgator_gliner.jsonl",
    "data/text2json.jsonl",
]


def main() -> None:
    model = GLiNER2.from_encoder("jhu-clsp/mmBERT-base", max_width=12, max_len=8192)

    config = TrainingConfig(
        output_dir="./out/mmbert-base",
        experiment_name="mmbert_base_multi_corpus",
        num_epochs=2,
        batch_size=8,
        gradient_accumulation_steps=2,
        encoder_lr=1e-5,
        task_lr=3e-4,
        warmup_ratio=0.05,
        scheduler_type="cosine",
        bf16=True,
        max_grad_norm=1.0,
        eval_strategy="no",
        save_total_limit=3,
        logging_steps=50,
        num_workers=4,
        validate_data=False,
        max_len=512,
    )

    trainer = GLiNER2Trainer(model, config)
    trainer.train(train_data=TRAIN_DATA)


if __name__ == "__main__":
    main()
