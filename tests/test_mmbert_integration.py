"""Smoke test: mmBERT backbone integrates with GLiNER2 trainer.

Validates that the embedding-resize and tie-weights fixes in `Extractor` let
mmBERT (ModernBERT-style multilingual encoder) initialize from scratch, accept
the GLiNER2 schema-special-tokens, and run one forward+backward without error.

Marked `slow` because it downloads ~550 MB on first run.

Run::

    uv run pytest tests/test_mmbert_integration.py -m slow -s
"""

from __future__ import annotations

import pytest
import torch

from gliner2.model import Extractor, ExtractorConfig
from gliner2.training.data import InputExample
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

MMBERT_SMALL = "jhu-clsp/mmBERT-small"


def _available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.mark.slow
@pytest.mark.parametrize("device", _available_devices())
def test_mmbert_loads_and_trains_one_step(tmp_path, device):
    from transformers import AutoConfig, AutoTokenizer

    encoder_config = AutoConfig.from_pretrained(MMBERT_SMALL, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MMBERT_SMALL)
    config = ExtractorConfig(model_name=MMBERT_SMALL, max_width=8)

    model = Extractor(config, encoder_config=encoder_config, tokenizer=tokenizer)
    model.to(device)

    examples = [
        InputExample(
            text="Marie Curie discovered radium in Paris.",
            entities={"scientist": ["Marie Curie"], "element": ["radium"], "city": ["Paris"]},
        ),
        InputExample(
            text="Mount Fuji is the highest peak in Japan.",
            entities={"mountain": ["Mount Fuji"], "country": ["Japan"]},
        ),
    ]

    cfg = TrainingConfig(
        output_dir=str(tmp_path),
        num_epochs=1,
        batch_size=2,
        fp16=False,
        bf16=False,
        eval_strategy="no",
        save_total_limit=1,
        num_workers=0,
        pin_memory=False,
        validate_data=False,
        report_to_wandb=False,
    )
    trainer = GLiNER2Trainer(model, cfg)
    result = trainer.train(train_data=examples)

    assert result["total_steps"] >= 1
    assert result["train_metrics_history"], "no metrics were logged"
    last = result["train_metrics_history"][-1]
    assert torch.isfinite(torch.tensor(last["loss"])), f"non-finite loss: {last['loss']}"
