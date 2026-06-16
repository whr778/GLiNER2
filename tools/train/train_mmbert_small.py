"""Train a fresh GLiNER2 on top of jhu-clsp/mmBERT-small.

Run::

    uv run python tools/train/train_mmbert_small.py

Expects these JSONLs under ``data/`` (see ``tools/data/`` converters):
    - nuner_full.jsonl                   (numind/NuNER)
    - pile_ner_def.jsonl                 (Universal-NER/Pile-NER-definition)
    - knowledgator_gliner.jsonl          (knowledgator/GLINER-multi-task-synthetic-data)
    - text2json.jsonl                    (knowledgator/text2json-training-data)
    - gliner_multilingual.jsonl          (knowledgator/gliner-multilingual-synthetic)
    - gliclass_logic.jsonl               (knowledgator/gliclass-v3-logic-dataset; classification)
    - scientific_text.jsonl              (knowledgator/Scientific-text-classification; classification)
    - biomed_ner.jsonl                   (knowledgator/biomed_NER; biomedical NER, 35 classes)
    - events_biotech.jsonl               (knowledgator/events_classification_biotech; multi-label classification)
    - sentence_rex.jsonl                 (knowledgator/sentence_rex; sentence-level relation extraction)
    - bio_ner_relations.jsonl            (knowledgator/bio-NER-relations; biomedical NER + RE)

Edit ``TRAIN_DATA`` below to drop a corpus or to point at smaller subsets.
"""

from gliner2 import GLiNER2
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig


TRAIN_DATA = [
    "data/nuner_full.jsonl",
    "data/pile_ner_def.jsonl",
    "data/knowledgator_gliner.jsonl",
    "data/text2json.jsonl",
    "data/gliner_multilingual.jsonl",
    "data/gliclass_logic.jsonl",
    "data/scientific_text.jsonl",
    "data/biomed_ner.jsonl",
    "data/events_biotech.jsonl",
    "data/sentence_rex.jsonl",
    "data/bio_ner_relations.jsonl",
]


def main() -> None:
    model = GLiNER2.from_encoder("jhu-clsp/mmBERT-small", max_width=12, max_len=8192)

    config = TrainingConfig(
        output_dir="./out/mmbert-small",
        experiment_name="mmbert_small_multi_corpus",
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=1,
        encoder_lr=2e-5,
        task_lr=5e-4,
        warmup_ratio=0.05,
        scheduler_type="cosine",
        bf16=True,
        max_grad_norm=1.0,
        eval_strategy="no",
        save_total_limit=3,
        logging_steps=50,
        num_workers=4,
        validate_data=False,
        max_len=8192,
    )

    trainer = GLiNER2Trainer(model, config)
    trainer.train(train_data=TRAIN_DATA)


if __name__ == "__main__":
    main()
