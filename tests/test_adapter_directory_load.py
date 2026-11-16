"""An adapter directory loads through AutoExtractor.

The CMNEE LoRA run saved best/ as adapter_config.json plus
adapter_model.safetensors. The post-train sweep called
AutoExtractor.from_pretrained on that directory, checkpoint_file joined a
config.json that was never written, and transformers treated the missing
path as a Hub repo id.
"""

from __future__ import annotations

import json

import pytest
import torch
from peft import PeftModel

from gliner2 import AutoExtractor
from gliner2.training.lora import save_adapter_checkpoint
from tests.fixtures.tiny_boundary_checkpoint import (
    build_tiny_boundary_model,
    save_tiny_boundary_checkpoint,
)


def _lora_b(model):
    name, param = next(
        (n, p) for n, p in model.named_parameters() if "lora_B" in n
    )
    return name, param


def test_adapter_directory_loads_with_and_without_extractor_config(tmp_path):
    base_dir = save_tiny_boundary_checkpoint(tmp_path / "base")
    model = build_tiny_boundary_model()
    model.name_or_path = str(base_dir)
    adapted = model.apply_lora(r=4, alpha=8, targets=["encoder"])
    name, param = _lora_b(adapted)
    param.data.fill_(0.25)
    adapted.get_base_model().config.label_map = {"entities": {"map": {"Foo": "Bar"}}}

    adapter_dir = tmp_path / "adapter"
    save_adapter_checkpoint(adapted, adapter_dir)
    written = sorted(p.name for p in adapter_dir.iterdir())
    assert "adapter_config.json" in written
    assert "adapter_model.safetensors" in written
    assert "config.json" in written
    saved_config = json.loads((adapter_dir / "config.json").read_text(encoding="utf-8"))
    assert saved_config["label_map"] == {"entities": {"map": {"Foo": "Bar"}}}
    assert saved_config["architecture"] == "boundary"

    loaded = AutoExtractor.from_pretrained(str(adapter_dir), map_location="cpu")
    assert isinstance(loaded, PeftModel)
    assert loaded.config is loaded.get_base_model().config
    assert loaded.config.boundary_head
    assert loaded.processor is loaded.get_base_model().processor
    assert loaded.config.label_map == {"entities": {"map": {"Foo": "Bar"}}}
    torch.testing.assert_close(
        dict(loaded.named_parameters())[name].detach().cpu(), param.detach().cpu()
    )

    loaded.eval()
    adapted.eval()
    text, labels = "john works at google", ["person"]
    schema = {"entities": labels}
    assert loaded.batch_extract([text], schema, threshold=0.0) == adapted.batch_extract(
        [text], schema, threshold=0.0
    )
    from gliner2.training.eval_metrics import apply_boundary_overrides

    apply_boundary_overrides(loaded, {"candidate_budget": 7})
    assert loaded.get_base_model().boundary_settings.candidate_budget == 7

    # The directory the CMNEE run actually wrote: adapter files, no config.json.
    (adapter_dir / "config.json").unlink()
    bare = AutoExtractor.from_pretrained(str(adapter_dir), map_location="cpu")
    assert isinstance(bare, PeftModel)
    assert bare.config is bare.get_base_model().config
    assert bare.config.boundary_head
    assert bare.config.label_map is None
    torch.testing.assert_close(
        dict(bare.named_parameters())[name].detach().cpu(), param.detach().cpu()
    )


def test_directory_with_no_checkpoint_raises_file_not_found(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        AutoExtractor.from_pretrained(str(empty))


def test_adapter_directory_without_base_id_raises(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": ""}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="base_model_name_or_path"):
        AutoExtractor.from_pretrained(str(adapter))
