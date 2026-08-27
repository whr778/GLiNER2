"""Off CUDA, a flash-attention checkpoint must degrade to sdpa AT LOAD.

Measured on `whr778/gliner2-gate2-mmbert-v2`, whose config stores the plain name:
transformers accepts `flash_attention_2` on a CPU box, normalizes it to the hub repo
id, and the FIRST FORWARD raises KeyError('kernels-community/flash-attn2') from
ALL_ATTENTION_FUNCTIONS. Construction alone looks healthy, so these tests assert on
the kwarg the encoder is actually built with.
"""

import pytest
import torch

from gliner2.models import base


class ModernBertConfig:
    """Config whose class name does NOT trigger the FlashDeBERTa path."""


@pytest.fixture
def built_with(monkeypatch):
    """Return the attn_implementation the encoder was constructed with."""
    seen = []

    def fake_from_config(config, **kwargs):
        seen.append(kwargs.get("attn_implementation"))
        return torch.nn.Linear(2, 2)

    monkeypatch.setattr(base.AutoModel, "from_config", fake_from_config)
    return seen


@pytest.mark.parametrize(
    "requested", ["flash_attention_2", base._HUB_FLASH_ATTN_2]
)
def test_flash_attention_degrades_to_sdpa_without_cuda(monkeypatch, built_with, requested):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    base.BaseExtractorModel._load_encoder(
        "unused", encoder_config=ModernBertConfig(), attn_implementation=requested
    )

    assert built_with == ["sdpa"]


def test_flash_attention_is_kept_on_cuda(monkeypatch, built_with):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    base.BaseExtractorModel._load_encoder(
        "unused", encoder_config=ModernBertConfig(),
        attn_implementation="flash_attention_2",
    )

    assert built_with == ["flash_attention_2"]
