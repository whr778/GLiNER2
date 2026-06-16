"""Train a fresh GLiNER2 on top of jhu-clsp/mmBERT-small.

Run::

    uv run python tools/train/train_mmbert_small.py

Expects ``data/nuner_full.jsonl`` and ``data/pile_ner_def.jsonl`` to already
exist (see tools/data/convert_nuner.py and tools/data/convert_pile_ner_definition.py).
"""

from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


def main() -> None:
    model = GLiNER2.from_encoder("jhu-clsp/mmBERT-small", max_width=12, max_len=8192)

    config = TrainingConfig(
        output_dir="./out/mmbert-small",
        experiment_name="mmbert_small_nuner_pile",
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=1,
        encoder_lr=2e-5,
        task_lr=5e-4,
        warmup_ratio=0.05,
        scheduler_type="cosine",
        fp16=True,
        max_grad_norm=1.0,
        eval_strategy="no",
        save_total_limit=3,
        logging_steps=50,
        num_workers=4,
        validate_data=False,
        max_len=8192,
    )

    trainer = GLiNER2Trainer(model, config)
    trainer.train(train_data=["data/nuner_full.jsonl", "data/pile_ner_def.jsonl"])


if __name__ == "__main__":
    main()
