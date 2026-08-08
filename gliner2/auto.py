"""Architecture-dispatching loader for GLiNER2 extractors.

``AutoExtractor`` resolves the architecture from a checkpoint's config and
returns the matching public model class. Legacy checkpoints without an
``architecture`` field load as ``"span"``.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar, Dict, Optional, Type

from gliner2.configuration import (
    ExtractorConfig,
    architecture_from_config,
    normalize_architecture,
)


class UnknownArchitectureError(ValueError):
    pass


class ArchitectureMismatchError(ValueError):
    pass


class ArchitectureRegistrationError(ValueError):
    pass


# Keys that address the Hub loader rather than the model constructor.
_HUB_LOAD_KEYS = {
    "cache_dir",
    "force_download",
    "local_files_only",
    "token",
    "revision",
    "subfolder",
    "proxies",
}


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
        hub_kwargs = {k: kwargs[k] for k in _HUB_LOAD_KEYS if k in kwargs}

        if config is None:
            config = _load_config(pretrained_model_name_or_path, hub_kwargs)

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
                "Use BoundaryExtractor.from_span_checkpoint() to transfer "
                "a span checkpoint into a boundary model."
            )

        model_class = cls._resolve_class(requested_architecture)
        return model_class.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            **kwargs,
        )


def _load_config(path, hub_kwargs: Dict[str, Any]) -> ExtractorConfig:
    """Load an ``ExtractorConfig`` from a local dir or the Hub."""
    if os.path.isdir(str(path)):
        config_file = os.path.join(str(path), "config.json")
        return ExtractorConfig.from_pretrained(config_file)
    return ExtractorConfig.from_pretrained(str(path), **hub_kwargs)
