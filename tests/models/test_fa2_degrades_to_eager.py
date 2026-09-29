"""A fall out of FlashAttention 2 goes to EAGER, never to sdpa, and a KeyError is caught.

TWO DEFECTS, ONE PATH, both hit on a real box on 2026-09-28.

KeyError ESCAPED THE DEGRADE CHAIN. `_load_encoder` caught
`(TypeError, ValueError, ImportError)`. When the Hub kernel cannot be FETCHED -- a
kernels/transformers version mismatch, no Hub access, or `kernels-community/flash-attn2`
returning 404 to a valid token, which it has done -- the config still names it and the
first forward raises `KeyError('kernels-community/flash-attn2')` out of
`ALL_ATTENTION_FUNCTIONS`. That is not in the tuple, so no fallback ran and the caller got
a bare KeyError instead of the warning naming the pin to check.

sdpa WAS ON THE DEGRADE PATH. It must not be. sdpa + bf16 on a ModernBERT encoder is a
CORRECTNESS failure and not a slowdown -- finite forward, NaN backward -- so the
middle rung was the one that silently breaks a run, while eager is merely slow. A
42-minute arm trained on that path on 2026-09-14 with a guard in place that was written to
prevent exactly it.

The control matters as much as the treatment: a request for sdpa ITSELF must still reach
eager, because deberta-v2 supports neither FA2 nor sdpa.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from gliner2.models import base


class PlainConfig:
    """Config that does not trigger the optional FlashDeBERTa backend."""


def _recorder(monkeypatch, exc, *, on_cuda=False):
    """Fail every non-eager implementation with `exc`, recording what was attempted."""
    attempted: list[str] = []
    encoder = torch.nn.Linear(2, 2)

    def fake_from_config(config, **kwargs):
        impl = kwargs.get("attn_implementation", "eager")
        attempted.append(impl)
        if impl != "eager":
            raise exc(impl)
        return encoder

    monkeypatch.setattr(base.AutoModel, "from_config", fake_from_config)
    if on_cuda:
        # The hub-kernel rung is CUDA-only, so the FA2 path only exists here.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    return attempted, encoder


@pytest.mark.parametrize("exc", [KeyError, ValueError, ImportError, TypeError])
def test_fa2_never_degrades_through_sdpa(monkeypatch, exc):
    """Whatever the kernel fails with, the landing spot is eager and sdpa is untouched."""
    attempted, encoder = _recorder(monkeypatch, exc, on_cuda=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = base.BaseExtractorModel._load_encoder(
            "unused", encoder_config=PlainConfig(),
            attn_implementation="flash_attention_2", use_flashdeberta=False)
    assert got is encoder
    assert "sdpa" not in attempted, (
        f"sdpa must not be on the FA2 degrade path (attempted {attempted}); on bf16 "
        "ModernBERT it is a correctness failure, and eager is merely slow"
    )
    assert attempted[-1] == "eager"


def test_a_keyerror_is_caught_rather_than_raised(monkeypatch):
    """The specific escape: KeyError used to propagate and kill the run."""
    attempted, encoder = _recorder(monkeypatch, KeyError, on_cuda=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = base.BaseExtractorModel._load_encoder(
            "unused", encoder_config=PlainConfig(),
            attn_implementation="flash_attention_2", use_flashdeberta=False)
    assert got is encoder


def test_the_hub_repo_id_is_still_tried_before_degrading(monkeypatch):
    """Retrying the SAME kernel under its Hub repo id is a REPAIR, not a downgrade -- it
    is what makes every checkpoint published with the plain string still get FA2."""
    attempted, _ = _recorder(monkeypatch, KeyError, on_cuda=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base.BaseExtractorModel._load_encoder(
            "unused", encoder_config=PlainConfig(),
            attn_implementation="flash_attention_2", use_flashdeberta=False)
    assert attempted[:2] == ["flash_attention_2", base._HUB_FLASH_ATTN_2]


def test_a_plain_sdpa_request_still_reaches_eager(monkeypatch):
    """The control. deberta-v2 supports neither FA2 nor sdpa and must get there."""
    attempted, encoder = _recorder(monkeypatch, ValueError)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = base.BaseExtractorModel._load_encoder(
            "unused", encoder_config=PlainConfig(),
            attn_implementation="sdpa", use_flashdeberta=False)
    assert got is encoder
    assert attempted == ["sdpa", "eager"]


def test_the_warning_names_flash_attention_2(monkeypatch):
    """A silent degrade out of FA2 is how a run gets lost; the warning must say so."""
    _recorder(monkeypatch, KeyError, on_cuda=True)
    with pytest.warns(RuntimeWarning, match="flash_attention_2"):
        base.BaseExtractorModel._load_encoder(
            "unused", encoder_config=PlainConfig(),
            attn_implementation="flash_attention_2", use_flashdeberta=False)


def test_strict_mode_still_refuses_the_degrade(monkeypatch):
    """GLINER2_STRICT_ATTN must keep failing at load, before GPU hours are spent."""
    _recorder(monkeypatch, KeyError, on_cuda=True)
    monkeypatch.setenv("GLINER2_STRICT_ATTN", "1")
    with pytest.raises(RuntimeError, match="GLINER2_STRICT_ATTN"):
        base.BaseExtractorModel._load_encoder(
            "unused", encoder_config=PlainConfig(),
            attn_implementation="flash_attention_2", use_flashdeberta=False)
