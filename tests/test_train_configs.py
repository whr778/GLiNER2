"""Validate the YAML training configs under ``tools/train/config/``.

Builds an :class:`ExtractorConfig` and a :class:`TrainingConfig` from each
config exactly the way ``tools/train/train.py`` does, without downloading any
encoder. Guards against:

* a config that no longer matches the ``TrainingConfig`` / ``ExtractorConfig``
  API (a stray or renamed field raises here),
* a ``struct_loss`` typo that would silently fall through to plain BCE,
* the YAML scalar trap where ``1e-5`` (no dot) parses as a string, not a float.
"""

from pathlib import Path

import pytest
import yaml

from gliner2.model import ExtractorConfig
from gliner2.training.trainer import TrainingConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "tools" / "train" / "config"

# struct_loss values dispatched in gliner2/model.py (compute_struct_loss +
# _struct_loss_term). Anything else silently trains as plain BCE, so keep this
# in sync when adding a variant.
RECOGNIZED_STRUCT_LOSS = {"bce", "bce_posweight", "focal", "asl", "dice", "bce_dice"}

CONFIG_FILES = sorted(CONFIG_DIR.glob("*.yaml"))
CONFIG_IDS = [p.name for p in CONFIG_FILES]


def test_config_dir_has_yaml_files():
    assert CONFIG_FILES, f"no YAML configs found under {CONFIG_DIR}"


@pytest.mark.parametrize("path", CONFIG_FILES, ids=CONFIG_IDS)
def test_config_builds(path):
    cfg = yaml.safe_load(path.read_text())
    assert {"model", "training", "data"} <= set(cfg), (
        f"{path.name}: missing one of the model/training/data sections"
    )

    model = dict(cfg["model"])
    if "pretrained" in model:
        # from_pretrained path: the encoder/heads/struct_loss come from the saved
        # checkpoint (can't build without downloading it). Validate the override
        # shape: a non-empty repo id, and a recognized struct_loss if overridden.
        assert isinstance(model["pretrained"], str) and model["pretrained"], (
            f"{path.name}: empty 'pretrained' repo id"
        )
        assert "encoder" not in model, f"{path.name}: set either 'pretrained' or 'encoder', not both"
        if "struct_loss" in model:
            assert model["struct_loss"] in RECOGNIZED_STRUCT_LOSS, (
                f"{path.name}: struct_loss override {model['struct_loss']!r} is not recognized"
            )
    else:
        # from_encoder path: mirror train.py, model.* (minus encoder) -> ExtractorConfig.
        encoder = model.pop("encoder")
        assert isinstance(encoder, str) and encoder, f"{path.name}: empty encoder"
        ec = ExtractorConfig(
            model_name=encoder,
            max_width=model.pop("max_width"),
            max_len=model.pop("max_len"),
            **model,
        )
        assert ec.struct_loss in RECOGNIZED_STRUCT_LOSS, (
            f"{path.name}: struct_loss={ec.struct_loss!r} is not recognized; "
            f"it would silently train as plain BCE"
        )

    # training.* -> TrainingConfig (a dataclass; raises on an unknown field).
    tc = TrainingConfig(**cfg["training"])
    assert isinstance(tc.encoder_lr, float), f"{path.name}: encoder_lr is not a float"
    assert isinstance(tc.task_lr, float), f"{path.name}: task_lr is not a float"


LABEL_CATEGORIES = {"entities", "relations", "events", "classifications"}


@pytest.mark.parametrize("path", CONFIG_FILES, ids=CONFIG_IDS)
def test_config_labels_section_shape(path):
    labels = yaml.safe_load(path.read_text()).get("labels")
    if labels is None:
        return  # the section is optional
    # Nested per-category form only; the removed flat form must not reappear.
    assert not any(k in labels for k in ("rollup", "separator", "map")), (
        f"{path.name}: labels uses the removed flat form; nest under a category"
    )
    for cat, block in labels.items():
        assert cat in LABEL_CATEGORIES, f"{path.name}: unknown labels category {cat!r}"
        assert isinstance(block, dict), f"{path.name}: labels.{cat} must be a mapping"
        assert isinstance(block.get("rollup", False), bool), f"{path.name}: {cat}.rollup must be bool"
        assert isinstance(block.get("separator", "."), str), f"{path.name}: {cat}.separator must be str"
        mapping = block.get("map") or {}
        assert isinstance(mapping, dict), f"{path.name}: {cat}.map must be a mapping"
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()), (
            f"{path.name}: {cat}.map keys and values must be strings"
        )


@pytest.mark.parametrize("path", CONFIG_FILES, ids=CONFIG_IDS)
def test_config_data_section_shape(path):
    data = yaml.safe_load(path.read_text()).get("data") or {}

    corpora = data.get("corpora") or []
    assert isinstance(corpora, list), f"{path.name}: data.corpora must be a list"

    event_files = data.get("event_files") or {}
    assert isinstance(event_files, dict), f"{path.name}: data.event_files must be a map"
    for name, splits in event_files.items():
        assert isinstance(splits, dict) and splits, f"{path.name}: event_files[{name}] empty"
        for split, fpath in splits.items():
            assert split in {"train", "val", "test"}, f"{path.name}: bad split {split!r}"
            assert isinstance(fpath, str) and fpath.endswith(".jsonl"), (
                f"{path.name}: event_files[{name}][{split}] is not a .jsonl path"
            )
