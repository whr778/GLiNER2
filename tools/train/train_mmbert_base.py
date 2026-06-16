"""Train a fresh GLiNER2 on top of jhu-clsp/mmBERT-base.

Run::

    uv run python tools/train/train_mmbert_base.py

Expects ``data/nuner_full.jsonl`` and ``data/pile_ner_def.jsonl`` to already
exist (see tools/data/convert_nuner.py and tools/data/convert_pile_ner_definition.py).
"""

from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


def main() -> None:
    model = GLiNER2.from_encoder("jhu-clsp/mmBERT-base", max_width=12, max_len=8192)

    config = TrainingConfig(
        output_dir="./out/mmbert-base",
        experiment_name="mmbert_base_nuner_pile",
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
    trainer.train(train_data=["data/nuner_full.jsonl", "data/pile_ner_def.jsonl"])


if __name__ == "__main__":
    main()
