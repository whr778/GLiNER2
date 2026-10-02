"""Describe a pretrained GLiNER2 checkpoint from its config.json alone (no weights).

Bases differ in ways a derived config must follow: fastino/gliner2-base-v1 is a legacy SPAN
model with no `architecture` key, no `max_len` and no tokenizer file; gliner2.5 models are
boundary models on DeBERTa; our eb* bases are boundary models on mmBERT that may carry a
`label_map` and `default_schema`. Nothing here assumes one of them.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

from huggingface_hub import hf_hub_download, list_repo_files


def _read_json(model: str, name: str) -> Dict[str, Any]:
    if os.path.isdir(model):
        return json.load(open(os.path.join(model, name), encoding="utf-8"))
    return json.load(open(hf_hub_download(model, name, token=os.environ.get("HF_TOKEN")),
                          encoding="utf-8"))


def _has_file(model: str, name: str) -> bool:
    if os.path.isdir(model):
        return os.path.exists(os.path.join(model, name))
    return name in list_repo_files(model, token=os.environ.get("HF_TOKEN"))


def probe_model(model: str) -> Dict[str, Any]:
    """Architecture, encoder, tokenizer source, length limit and label inventory."""
    cfg = _read_json(model, "config.json")
    encoder = cfg.get("model_name")
    enc_cfg = _read_json(model, "encoder_config/config.json")
    max_len = cfg.get("max_len") or enc_cfg.get("max_position_embeddings")
    schema = cfg.get("default_schema") or {}
    return {
        "model": model,
        "architecture": cfg.get("architecture") or "span",
        "encoder": encoder,
        "encoder_type": enc_cfg.get("model_type"),
        "max_len": max_len,
        "max_len_source": "config.max_len" if cfg.get("max_len") else "encoder max_position_embeddings",
        "attn_implementation": cfg.get("attn_implementation"),
        "tokenizer_source": model if _has_file(model, "tokenizer.json") else encoder,
        "label_map": cfg.get("label_map"),
        "inference_defaults": cfg.get("inference_defaults"),
        "default_schema": schema,
        "has_label_inventory": bool(cfg.get("label_map") or schema),
        "boundary_head": cfg.get("boundary_head") or {},
    }
