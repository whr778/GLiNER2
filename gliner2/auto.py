"""Architecture-dispatching loader for GLiNER2 extractors.

``AutoExtractor`` resolves the architecture from a checkpoint's config and
returns the matching public model class. Legacy checkpoints without an
``architecture`` field load as ``"span"``.
"""

from __future__ import annotations

import json
import os
from typing import Any, ClassVar, Dict, Optional, Type

from gliner2.configuration import (
    ExtractorConfig,
    architecture_from_config,
    normalize_architecture,
)
from gliner2.models.loading import (
    HUB_LOAD_OPTIONS,
    checkpoint_file,
    split_load_kwargs,
)


class UnknownArchitectureError(ValueError):
    pass


class ArchitectureMismatchError(ValueError):
    pass


class ArchitectureRegistrationError(ValueError):
    pass


# Backward-compatible private alias; the shared loader owns the option set.
_HUB_LOAD_KEYS = HUB_LOAD_OPTIONS


def _ensure_registered() -> None:
    """Import the engine module so built-in architectures self-register."""
    if set(AutoExtractor._registry) >= {"span", "boundary"}:
        return
    # Importing the engine registers SpanExtractor and BoundaryExtractor.
    import gliner2.inference.engine  # noqa: F401


class AutoExtractor:
    """Registry-based dispatcher over extractor architectures."""

    _registry: ClassVar[Dict[str, Type]] = {}

    # -- registry -----------------------------------------------------------
    @classmethod
    def register(
        cls,
        architecture: str,
        model_class: Type,
        *,
        exist_ok: bool = False,
    ) -> None:
        name = normalize_architecture(architecture)
        if name in cls._registry and not exist_ok:
            if cls._registry[name] is model_class:
                return
            raise ArchitectureRegistrationError(
                f"Architecture {name!r} is already registered to "
                f"{cls._registry[name]!r}. Pass exist_ok=True to override."
            )
        cls._registry[name] = model_class

    @classmethod
    def _resolve_class(cls, architecture: str) -> Type:
        _ensure_registered()
        name = normalize_architecture(architecture)
        if name not in cls._registry:
            raise UnknownArchitectureError(
                f"No model class registered for architecture {name!r}. "
                f"Registered: {sorted(cls._registry)}"
            )
        return cls._registry[name]

    # -- loading ------------------------------------------------------------
    @classmethod
    def from_config(cls, config: ExtractorConfig, **kwargs: Any):
        architecture = architecture_from_config(config)
        model_class = cls._resolve_class(architecture)
        return model_class(config, **kwargs)

    @classmethod
    def from_encoder(
        cls,
        encoder_name_or_path: str,
        *,
        architecture: str = "span",
        max_len: Optional[int] = None,
        map_location: Optional[str] = None,
        trust_remote_code: bool = True,
        **config_kwargs: Any,
    ):
        """Bootstrap a fresh extractor of any architecture on a raw HF encoder.

        The architecture-dispatching counterpart to the per-class ``from_encoder``:
        encoder weights are pretrained, task heads are randomly initialized. This
        is what lets a training config choose ``boundary`` -- ``GLiNER2`` is the
        span class and hardcodes ``architecture="span"``.

        Args:
            encoder_name_or_path: HF repo id or local path of the base encoder.
            architecture: Registered architecture name (``"span"`` default keeps
                existing callers unchanged; ``"boundary"`` for the boundary head).
            max_len: Optional max sequence length override.
            map_location: Device to place the model on.
            trust_remote_code: Forwarded to ``AutoConfig.from_pretrained``.
            **config_kwargs: Extra fields forwarded to the architecture's config
                (e.g. ``span_head``/``boundary_head`` blocks, ``token_pooling``).
        """
        from transformers import AutoConfig, AutoTokenizer

        from gliner2.models.base import resolve_device

        name = normalize_architecture(architecture)
        model_class = cls._resolve_class(name)
        encoder_config = AutoConfig.from_pretrained(
            encoder_name_or_path, trust_remote_code=trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(encoder_name_or_path)
        config = model_class.config_class(
            model_name=encoder_name_or_path,
            architecture=name,
            max_len=max_len,
            **config_kwargs,
        )
        model = model_class(
            config, encoder_config=encoder_config, tokenizer=tokenizer,
        )
        return model.to(resolve_device(map_location))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args: Any,
        architecture: Optional[str] = None,
        config: Optional[ExtractorConfig] = None,
        allow_architecture_override: bool = False,
        **kwargs: Any,
    ):
        model_kwargs, hub_kwargs = split_load_kwargs(
            kwargs, context="AutoExtractor.from_pretrained"
        )
        source = str(pretrained_model_name_or_path)
        if _is_local_adapter_dir(source):
            return _load_saved_adapter(
                cls, source, model_args, architecture, config,
                allow_architecture_override, model_kwargs, hub_kwargs,
            )

        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

        try:
            return cls._from_full_checkpoint(
                source, model_args, architecture, config,
                allow_architecture_override, model_kwargs, hub_kwargs,
            )
        except (FileNotFoundError, EntryNotFoundError, LocalEntryNotFoundError):
            # A Hub adapter repo has adapter_config.json and no
            # model.safetensors. A local adapter directory is handled above;
            # a local directory that is neither is a missing checkpoint.
            if os.path.isdir(source) or _hub_adapter_config(source, hub_kwargs) is None:
                raise
            return _load_saved_adapter(
                cls, source, model_args, architecture, config,
                allow_architecture_override, model_kwargs, hub_kwargs,
            )

    @classmethod
    def _from_full_checkpoint(
        cls,
        source,
        model_args,
        architecture,
        config,
        allow_architecture_override,
        model_kwargs,
        hub_kwargs,
    ):
        if config is None:
            config = _load_config(source, hub_kwargs)

        saved_architecture = architecture_from_config(config)
        requested_architecture = (
            normalize_architecture(architecture)
            if architecture is not None
            else saved_architecture
        )

        if requested_architecture != saved_architecture and not allow_architecture_override:
            raise ArchitectureMismatchError(
                f"Checkpoint architecture is {saved_architecture!r}, "
                f"but {requested_architecture!r} was requested. "
                "Span and boundary heads are not checkpoint-compatible, and "
                "automatic architecture conversion is not supported. Load the "
                "checkpoint with its saved architecture or initialize and train "
                "a separate boundary checkpoint."
            )

        model_class = cls._resolve_class(requested_architecture)
        # revision and token stay on the config fetch. subfolder has to follow the
        # weights: they live beside config.json, not at the Hub repo root.
        if "subfolder" in hub_kwargs:
            model_kwargs = {**model_kwargs, "subfolder": hub_kwargs["subfolder"]}
        return model_class.from_pretrained(
            source,
            *model_args,
            config=config,
            **model_kwargs,
        )


def _is_local_adapter_dir(path: str) -> bool:
    """True when ``path`` is a PEFT adapter directory, not a full checkpoint.

    A full checkpoint that also contains an adapter file is loaded as the
    full model. An adapter directory has ``adapter_config.json`` and no
    ``model.safetensors``.
    """
    if not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "adapter_config.json")):
        return False
    if os.path.isfile(os.path.join(path, "model.safetensors")):
        return False
    if os.path.isfile(os.path.join(path, "pytorch_model.bin")):
        return False
    return True


def _adapter_base_id(adapter_config_path: str) -> str:
    with open(adapter_config_path, encoding="utf-8") as handle:
        data = json.load(handle)
    base = data.get("base_model_name_or_path")
    if not isinstance(base, str) or not base.strip():
        raise ValueError(
            f"{adapter_config_path} has no base_model_name_or_path; "
            "the adapter cannot be loaded without the model it was trained on"
        )
    return base


def _extractor_config_beside_adapter(source: str, hub_kwargs: Dict[str, Any]):
    """The extractor config saved next to an adapter, if the save wrote one."""
    if os.path.isdir(source):
        path = os.path.join(source, "config.json")
        if not os.path.isfile(path):
            return None
        return ExtractorConfig.from_pretrained(path)
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=source, filename="config.json", **dict(hub_kwargs or {})
        )
    except (EntryNotFoundError, LocalEntryNotFoundError):
        return None
    return ExtractorConfig.from_pretrained(path)


def _hub_adapter_config(repo_id: str, hub_kwargs: Dict[str, Any]):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    try:
        return hf_hub_download(
            repo_id=repo_id, filename="adapter_config.json", **dict(hub_kwargs or {})
        )
    except (EntryNotFoundError, LocalEntryNotFoundError):
        return None


def _load_saved_adapter(
    cls,
    source,
    model_args,
    architecture,
    config,
    allow_architecture_override,
    model_kwargs,
    hub_kwargs,
):
    """Load the base named by the adapter, then apply the adapter weights.

    The sweep reads ``model.config.boundary_head`` and ``model.processor``.
    Those live on the extractor, so the returned ``PeftModel`` keeps the
    extractor config object.
    """
    if os.path.isdir(source):
        adapter_config = os.path.join(source, "adapter_config.json")
    else:
        adapter_config = _hub_adapter_config(source, hub_kwargs)
        if adapter_config is None:
            raise FileNotFoundError(
                f"{source} has no adapter_config.json"
            )
    base_id = _adapter_base_id(adapter_config)
    if config is None:
        config = _extractor_config_beside_adapter(source, hub_kwargs)
    base = cls.from_pretrained(
        base_id,
        *model_args,
        architecture=architecture,
        config=config,
        allow_architecture_override=allow_architecture_override,
        **model_kwargs,
        **hub_kwargs,
    )
    from peft import PeftModel

    loaded = PeftModel.from_pretrained(base, source)
    loaded.config = base.config
    return loaded


def _load_config(path, hub_kwargs: Dict[str, Any]) -> ExtractorConfig:
    """Load an ``ExtractorConfig`` from a local dir or the Hub."""
    config_file = checkpoint_file(str(path), "config.json", hub_kwargs)
    return ExtractorConfig.from_pretrained(config_file)
