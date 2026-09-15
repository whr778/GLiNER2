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


def test_hub_flash_attn_is_registered_in_the_MASK_interface_too():
    """There are TWO registries and the Hub kernel only lands in one of them.

    Loading FA2 from the Hub sets `config._attn_implementation` to the repo id, and
    transformers registers the kernel in `ALL_ATTENTION_FUNCTIONS` under that key. It does
    NOT register it in `ALL_MASK_ATTENTION_FUNCTIONS`, which is keyed by the six built-in
    names, so every `create_*_mask` raises KeyError(repo id).

    That failure is INVISIBLE TO INFERENCE on this architecture -- eval never reaches a
    `create_*_mask` call. Measured 2026-09-14 on an A100: three decode-arm eval runs passed
    on kernels 0.12.3 / transformers 5.6.2, and the first TRAINING run on the same box and
    the same locked versions aborted at step 0 with the GPU at 0%. So a GPU is not needed
    to test this -- only the registry is.
    """
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

    from gliner2.models.base import _HUB_FLASH_ATTN_2, _register_hub_flash_attn_mask

    _register_hub_flash_attn_mask()
    assert _HUB_FLASH_ATTN_2 in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping, \
        "the Hub repo id must be a mask key, or training raises KeyError on the first forward"
    assert ALL_MASK_ATTENTION_FUNCTIONS[_HUB_FLASH_ATTN_2] is \
        ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"], \
        "the repo id is the SAME FlashAttention 2 and must take the same mask builder"

    _register_hub_flash_attn_mask()   # idempotent: loaders may call it per model
    assert _HUB_FLASH_ATTN_2 in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping


def test_strict_attn_refuses_the_downgrade_when_the_HUB_KERNEL_fails(monkeypatch):
    """GLINER2_STRICT_ATTN must refuse at ANY index, not only the first.

    The guard read `strict and index == 0`, which made it unreachable on the only path
    that matters. Candidates are [flash_attention_2, <hub id>, sdpa, eager]: index 0
    failing is a REPAIR and `continue`s, so the real downgrade always happens at index 1 --
    where `index == 0` is false and strict mode was skipped entirely.

    Measured 2026-09-14, on a box with the variable set: the hub kernel was rejected for a
    version mismatch (kernels 0.17 against transformers 5.16, which wants 0.16.x), the
    loader fell back to sdpa with a warning, and BOTH ARMS trained 42 minutes each on a
    backend that is a correctness failure on bf16 ModernBERT. The guard written to stop
    exactly that watched it happen and said nothing an operator would act on.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("GLINER2_STRICT_ATTN", "1")

    # Every FA2 form fails, as it did when the kernel version was incompatible; sdpa
    # would succeed, which is precisely the downgrade that must be refused.
    def fake_from_config(config, **kwargs):
        impl = kwargs.get("attn_implementation")
        if impl in ("flash_attention_2", base._HUB_FLASH_ATTN_2):
            raise ValueError(f"kernels version incompatible for {impl}")
        return torch.nn.Linear(2, 2)

    monkeypatch.setattr(base.AutoModel, "from_config", fake_from_config)

    with pytest.raises(RuntimeError, match="GLINER2_STRICT_ATTN"):
        base.BaseExtractorModel._load_encoder(
            "unused", encoder_config=ModernBertConfig(),
            attn_implementation="flash_attention_2",
        )


def test_strict_attn_still_allows_sdpa_to_degrade_to_eager(monkeypatch):
    """Strict mode is about FALLING OUT OF FA2, not about every downgrade.

    deberta-v2 supports neither FA2 nor sdpa and must reach eager, so a request for sdpa
    degrading is still permitted even under the variable.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("GLINER2_STRICT_ATTN", "1")

    seen = []

    def fake_from_config(config, **kwargs):
        impl = kwargs.get("attn_implementation")
        seen.append(impl)
        if impl == "sdpa":
            raise ValueError("sdpa unsupported by this encoder")
        return torch.nn.Linear(2, 2)

    monkeypatch.setattr(base.AutoModel, "from_config", fake_from_config)

    base.BaseExtractorModel._load_encoder(
        "unused", encoder_config=ModernBertConfig(), attn_implementation="sdpa",
    )
    assert seen[-1] == "eager", f"expected a degrade to eager, saw {seen}"
