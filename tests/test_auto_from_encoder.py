"""Architecture dispatch when bootstrapping from a raw encoder.

`GLiNER2` is the span class and its `from_encoder` hardcodes architecture="span",
so a training config had no way to ask for the boundary head. `AutoExtractor.
from_encoder` is the dispatching counterpart (JOINT_IE_SCALING sec 3c).
"""

from __future__ import annotations

import pytest

from gliner2 import AutoExtractor


@pytest.fixture
def encoder_dir(tmp_path, tiny_tokenizer, tiny_encoder_config):
    """A raw encoder on disk: config + tokenizer, no task heads."""
    path = tmp_path / "tiny-encoder"
    tiny_encoder_config.save_pretrained(str(path))
    tiny_tokenizer.save_pretrained(str(path))
    return str(path)


def test_from_encoder_builds_a_boundary_model(encoder_dir):
    """The point of the change: a config can now ask for the boundary head."""
    from tests.fixtures.tiny_boundary_checkpoint import TINY_BOUNDARY_HEAD

    model = AutoExtractor.from_encoder(
        encoder_dir, architecture="boundary",
        boundary_head=dict(TINY_BOUNDARY_HEAD), token_pooling="first",
    )
    assert type(model).__name__ == "BoundaryExtractor"
    assert model.config.architecture == "boundary"


def test_from_encoder_defaults_to_span(encoder_dir):
    """Existing configs declare no architecture and must be unaffected."""
    model = AutoExtractor.from_encoder(encoder_dir, max_width=8)
    assert type(model).__name__ == "SpanExtractor"
    assert model.config.architecture == "span"


def test_unknown_architecture_is_rejected(encoder_dir):
    """A typo in the config must fail loudly, not fall back to span."""
    with pytest.raises(ValueError, match="nonsense"):
        AutoExtractor.from_encoder(encoder_dir, architecture="nonsense")
