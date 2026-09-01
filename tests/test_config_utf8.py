"""config.json must ship readable labels, not \\uXXXX escapes.

transformers' PretrainedConfig.to_json_string calls json.dumps without
ensure_ascii=False. The label set is an INPUT to this model, so an unreadable vocabulary
is one nobody audits -- a DocFEE checkpoint shipped 1,054 escapes and its Chinese labels
went unnoticed for two runs.
"""

import json

from gliner2.models.base import rewrite_config_as_utf8


def test_escapes_become_real_characters(tmp_path):
    labels = ["股东减持", "Person", "한국어", "عربي", "日本語"]
    (tmp_path / "config.json").write_text(
        json.dumps({"labels": labels}, indent=2, sort_keys=True), encoding="utf-8")
    assert "\\u" in (tmp_path / "config.json").read_text(encoding="utf-8")

    rewrite_config_as_utf8(tmp_path)

    text = (tmp_path / "config.json").read_text(encoding="utf-8")
    assert "\\u" not in text
    for label in labels:
        assert label in text
    assert json.loads(text)["labels"] == labels   # value is unchanged, only the encoding


def test_missing_config_is_a_no_op(tmp_path):
    rewrite_config_as_utf8(tmp_path)   # nothing to rewrite, must not raise


def test_ascii_only_config_is_unchanged_semantically(tmp_path):
    payload = {"architecture": "boundary", "labels": ["Person", "Location"]}
    (tmp_path / "config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    rewrite_config_as_utf8(tmp_path)
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8")) == payload


def test_real_boundary_save_pretrained_emits_readable_labels(
        tmp_path, tiny_tokenizer, tiny_encoder_config):
    """End-to-end: the hook must fire from save_pretrained, not just in isolation."""
    from gliner2.auto import AutoExtractor
    from tests.fixtures.tiny_boundary_checkpoint import TINY_BOUNDARY_HEAD

    encoder = tmp_path / "tiny-encoder"
    tiny_encoder_config.save_pretrained(str(encoder))
    tiny_tokenizer.save_pretrained(str(encoder))

    model = AutoExtractor.from_encoder(
        str(encoder), architecture="boundary",
        boundary_head=dict(TINY_BOUNDARY_HEAD), token_pooling="first",
    )
    # A label set is an INPUT to this model; park a CJK one on the config and save.
    model.config.default_entity_labels = ["股东减持", "한국어", "Person"]

    out = tmp_path / "saved"
    model.save_pretrained(str(out))

    text = (out / "config.json").read_text(encoding="utf-8")
    assert "\\u" not in text, "config.json still ships escaped labels"
    assert "股东减持" in text and "한국어" in text
