"""Architecture-neutral query metadata, encoded-batch container, and base model.

``QuerySpec``/``QueryLayout`` describe the extractive/classification queries of
a sample independent of architecture. ``EncodedBatch`` is the vectorized
encoder output both architectures can consume. ``BaseExtractorModel`` provides
shared encoder loading and architecture-stamping save; the span model keeps its
own encode path until parity is proven (per the blueprint).
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import os
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

from gliner2.configuration import ExtractorConfig

logger = logging.getLogger(__name__)

# transformers reads "flash_attention_2" as the PIP PACKAGE; the same kernel is
# served from the Hub registry under this repo id, which is what actually loads here.
_HUB_FLASH_ATTN_2 = "kernels-community/flash-attn2"


def _register_hub_flash_attn_mask() -> None:
    """Teach the MASK registry the Hub kernel's repo id. Idempotent.

    THERE ARE TWO REGISTRIES AND THE KERNEL ONLY LANDS IN ONE. Loading FA2 from the Hub
    sets ``config._attn_implementation`` to the repo id, and transformers registers the
    kernel in ``ALL_ATTENTION_FUNCTIONS`` under that key -- but ``masking_utils`` keys
    ``ALL_MASK_ATTENTION_FUNCTIONS`` by the six built-in names only. Every
    ``create_*_mask`` then does ``ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]``
    and raises ``KeyError('kernels-community/flash-attn2')``.

    THIS IS A LATENT GAP, NOT THE CAUSE OF THE 2026-09-14 TRAINING ABORT. I registered it
    believing it was, and I was wrong: that traceback names ``ALL_ATTENTION_FUNCTIONS`` at
    ``modeling_modernbert.py:293``, a different registry. The real cause was external --
    ``kernels-community/flash-attn2`` began returning 404 to a valid token, so the kernel
    could not be fetched at all. Kept because the gap is real: if the Hub kernel ever does
    load, every ``create_*_mask`` would raise on the repo id. Registering it costs a dict
    insert.

    The lesson is the misread, not the gap: "KeyError from a registry" matched a registry
    I had just been reading, and I did not check which one the traceback named. That cost
    a box launch.

    The repo id is the same FlashAttention 2, so it takes the same mask builder.
    """
    from transformers.masking_utils import (
        ALL_MASK_ATTENTION_FUNCTIONS,
        flash_attention_mask,
    )

    if _HUB_FLASH_ATTN_2 not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        ALL_MASK_ATTENTION_FUNCTIONS.register(_HUB_FLASH_ATTN_2, flash_attention_mask)


def resolve_device(map_location: Optional[str] = None) -> str:
    """Resolve where to place a model, defaulting to the best available device.

    An explicit ``map_location`` is honored as-is; otherwise pick CUDA, then MPS
    (Apple Silicon GPU), then CPU. Shared by the model ``from_pretrained`` /
    ``from_encoder`` loaders so inference (the extractor and the viewer) uses the
    GPU without the caller having to ask.
    """
    if map_location is not None:
        device = map_location
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    if str(device).startswith("mps"):
        _enable_mps_flash_attention()
    return device


_MPS_FLASH_ENABLED = False


def _enable_mps_flash_attention() -> None:
    """Swap in Metal FlashAttention for SDPA -- OPT-IN (GLINER2_MPS_FLASH_ATTN=1). MPS only, once.

    OFF BY DEFAULT since 2026-10-05: it does not apply padding exactly. On 5 real CASIE docs,
    the 3 PADDED ones decoded differently alone than batched beside longer docs; CPU, and MPS
    with stock SDPA, were identical either way. So with it on, a batched prediction depends on
    which documents share its batch. The speed it buys (75.7 -> 70.7 ms per extract on mmBERT;
    ~28% on few-label classification) is not worth a result that moves with the batch.
    The import lives here because the package is Mac-only -- CUDA and CPU never touch it.
    """
    global _MPS_FLASH_ENABLED
    if _MPS_FLASH_ENABLED:
        return
    _MPS_FLASH_ENABLED = True
    if os.environ.get("GLINER2_MPS_FLASH_ATTN") != "1":
        return
    try:
        from mps_flash_attn import replace_sdpa
    except ImportError as exc:
        warnings.warn(f"mps-flash-attn unavailable, MPS keeps stock SDPA ({exc})",
                      RuntimeWarning, stacklevel=2)
        return
    replace_sdpa()


def load_extractor_tokenizer(repo_or_dir: str, **kwargs):
    """Load checkpoint tokenizers across Transformers metadata versions.

    Older GLiNER2 checkpoints serialized ``extra_special_tokens`` as a list.
    Newer Transformers releases reserve that field for a name-to-token mapping
    and fail before ``SchemaTransformer`` can register GLiNER2's special tokens.
    Retry only that known incompatibility while leaving all other load failures
    untouched. ``kwargs`` is the Hub load (``subfolder``, and nothing else here).
    """
    try:
        return AutoTokenizer.from_pretrained(repo_or_dir, **kwargs)
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        warnings.warn(
            "Checkpoint uses legacy list-valued extra_special_tokens metadata; "
            "loading with compatibility normalization.",
            UserWarning,
            stacklevel=2,
        )
        return AutoTokenizer.from_pretrained(
            repo_or_dir, **kwargs, extra_special_tokens={}
        )


# =============================================================================
# Query metadata
# =============================================================================

@dataclass(frozen=True)
class QuerySpec:
    """Metadata for one query (an extractive field/entity or a classification)."""
    query_id: int
    task_index: int
    task_type: str
    task_name: str
    role_index: int = 0
    role_name: str = ""
    field_path: Tuple[str, ...] = ()
    extractive: bool = True


QUERY_TYPE_SEP = "::"


def qualified_query_type(query_id: int, role_name: str) -> str:
    """The joint-IE node type for one query: unique per query, not per field name.

    Field names are arbitrary and scoped to their own relation type (tutorial 8), so two
    relation types routinely declare the same names -- the Standard Format's ``head``/
    ``tail`` being the common case. Keying joint-IE mentions on the bare field name then
    gives two relation types identical node ids for the same span, and ``JointProblem``
    rejects the problem. The query id is used as the qualifier rather than the task name
    because it is unique *structurally*, with no assumption that task names are distinct.

    Internal only: `display_query_type` strips it before anything is emitted. Splitting
    from the left is safe whatever the field name contains, since the prefix is an integer.
    """
    return f"{int(query_id)}{QUERY_TYPE_SEP}{role_name}"


def display_query_type(qualified: str) -> str:
    """The user-facing field name behind a `qualified_query_type`."""
    head, sep, name = qualified.partition(QUERY_TYPE_SEP)
    return name if sep and head.isdigit() else qualified


@dataclass(frozen=True)
class QueryLayout:
    """Ordered queries for a single sample, with fast id lookup."""
    queries: Tuple[QuerySpec, ...]
    classification_query_ids: Tuple[int, ...] = ()
    extractive_query_ids: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        # Derive id groupings if not provided.
        if not self.classification_query_ids and not self.extractive_query_ids and self.queries:
            cls_ids = tuple(q.query_id for q in self.queries if not q.extractive)
            ext_ids = tuple(q.query_id for q in self.queries if q.extractive)
            object.__setattr__(self, "classification_query_ids", cls_ids)
            object.__setattr__(self, "extractive_query_ids", ext_ids)

    def __len__(self) -> int:
        return len(self.queries)

    def query(self, query_id: int) -> QuerySpec:
        for q in self.queries:
            if q.query_id == query_id:
                return q
        raise KeyError(f"no query with id {query_id}")

    def extractive_count(self) -> int:
        return len(self.extractive_query_ids)

    def classification_count(self) -> int:
        return len(self.classification_query_ids)


# =============================================================================
# Encoded batch
# =============================================================================

@dataclass
class EncodedBatch:
    """Vectorized encoder output shared by both architectures."""
    text_states: torch.Tensor            # [B, L, H]
    text_mask: torch.BoolTensor          # [B, L]
    text_lengths: torch.LongTensor       # [B]
    query_states: torch.Tensor           # [B, Q, H]
    query_mask: torch.BoolTensor         # [B, Q]
    query_layouts: Tuple[QueryLayout, ...]
    classification_states: Optional[torch.Tensor] = None

    def to(self, device) -> "EncodedBatch":
        return EncodedBatch(
            text_states=self.text_states.to(device),
            text_mask=self.text_mask.to(device),
            text_lengths=self.text_lengths.to(device),
            query_states=self.query_states.to(device),
            query_mask=self.query_mask.to(device),
            query_layouts=self.query_layouts,
            classification_states=(
                self.classification_states.to(device)
                if self.classification_states is not None else None
            ),
        )


# =============================================================================
# Base model
# =============================================================================

def rewrite_config_as_utf8(save_directory) -> None:
    """Re-emit config.json with real characters instead of \\uXXXX escapes.

    transformers' PretrainedConfig.to_json_string calls json.dumps WITHOUT
    ensure_ascii=False, so every non-ASCII label is escaped: a DocFEE checkpoint ships
    1,054 escapes and its label list reads as "\\u80a1\\u4e1c\\u51cf\\u6301" rather than
    股东减持. The two parse identically -- this is readability, not correctness -- but the
    label set is an INPUT to this model, and a vocabulary nobody can read is a vocabulary
    nobody checks. Chinese labels hid in exactly that way for two runs.
    """
    import json
    from pathlib import Path as _Path

    path = _Path(save_directory) / "config.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def _pad_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    *,
    hidden_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Right-pad per-row encoder states to ``attention_mask``'s shape.

    Args:
        hidden_states: One ``(tokens_i, hidden)`` tensor per batch row.
        attention_mask: The collated ``[B, L]`` mask; rows are right-padded.
        hidden_size: Expected hidden width.
        dtype: Output dtype, matching the model's parameters.

    Returns:
        A ``[B, L, hidden_size]`` tensor on ``attention_mask``'s device.

    Raises:
        ValueError: On a row count, row length or hidden width mismatch.
    """
    rows, width = attention_mask.shape
    if len(hidden_states) != rows:
        raise ValueError(
            f"hidden_states has {len(hidden_states)} rows; the batch has {rows}"
        )
    lengths = attention_mask.sum(dim=-1).tolist()
    padded = torch.zeros(
        (rows, width, hidden_size), dtype=dtype, device=attention_mask.device
    )
    for index, (states, length) in enumerate(zip(hidden_states, lengths, strict=True)):
        if states.dim() != 2 or states.shape != (length, hidden_size):
            raise ValueError(
                f"hidden_states[{index}] has shape {tuple(states.shape)}; "
                f"expected ({length}, {hidden_size})"
            )
        padded[index, :length] = states.to(device=padded.device, dtype=dtype)
    return padded


class BaseExtractorModel(PreTrainedModel):
    """Shared base for extractor architectures.

    Provides encoder construction and architecture-aware serialization. It does
    not impose an ``encode()`` contract on the span model; the boundary model
    uses ``encode()`` while the span model retains its legacy path.
    """
    config_class = ExtractorConfig

    # packed="auto" thresholds, measured on an H200 with
    # benchmarks/benchmark_packed_threshold.py.
    DISENTANGLED_FLASH_PACKED_MIN_PADDING = 0.15
    DISENTANGLED_FLASH_PACKED_MIN_WORK = 2**21  # B * L**2

    @staticmethod
    def _disentangled_flash_backend_for_device(device: torch.device) -> str:
        """Select Triton on NVIDIA CUDA and PyTorch everywhere else."""
        if device.type == "cuda" and torch.version.hip is None:
            return "triton"
        if device.type in {"cpu", "mps", "cuda"}:
            return "torch"
        raise RuntimeError(
            "DisentangledFlash supports NVIDIA CUDA through Triton and "
            "CPU, MPS, and AMD ROCm through its PyTorch backend; got "
            f"device {device}."
        )

    @staticmethod
    def _disentangled_flash_packing_layout(
        attention_mask: Optional[torch.Tensor],
        packed: bool | str,
        min_padding: float,
        min_work: int = 0,
    ) -> Optional[Tuple[torch.Tensor, Tuple[int, ...]]]:
        """Return ``(mask, lengths)`` to run the batch packed, else ``None``."""
        forced = packed is True

        def reject(reason: str) -> None:
            if forced:
                raise ValueError(f"DisentangledFlash packed mode: {reason}")
            return None

        if attention_mask is None or attention_mask.ndim != 2:
            return reject("requires a [B, L] attention_mask")
        batch_size, sequence_length = attention_mask.shape
        if batch_size == 0 or sequence_length == 0:
            return reject("requires a non-empty batch")
        if not forced and batch_size * sequence_length**2 < min_work:
            return None

        mask = attention_mask.bool()
        lengths_tensor = mask.sum(dim=1)
        lengths = tuple(int(length) for length in lengths_tensor.tolist())
        if min(lengths) <= 0:
            return reject("empty sequences are unsupported")
        padding = 1.0 - sum(lengths) / (batch_size * sequence_length)
        if not forced and padding < min_padding:
            return None

        positions = torch.arange(sequence_length, device=mask.device)
        if not torch.equal(mask, positions[None, :] < lengths_tensor[:, None]):
            return reject("only right-padded batches can be packed")
        return mask, lengths

    def enable_disentangled_flash(
        self,
        *,
        backend: str = "auto",
        inference: bool | None = None,
        packed: bool | str = "auto",
        packed_min_padding: float | None = None,
        packed_min_work: int | None = None,
        tuning: Any = None,
    ) -> "BaseExtractorModel":
        """Enable DisentangledFlash for DeBERTa-v2/v3 attention.

        Move the model to its final device and dtype first. For training, call
        it after :meth:`train` and before building the optimizer. Packed
        batches return zeros at padded positions.

        Args:
            backend: ``"auto"``, ``"triton"`` or ``"torch"``. ``"auto"`` uses
                Triton on NVIDIA CUDA and PyTorch elsewhere.
            inference: Defaults to ``not self.training``.
            packed: ``"auto"`` packs when both thresholds below are met,
                ``True`` always packs, ``False`` never does.
            packed_min_padding: Minimum padded fraction of ``B * L``.
            packed_min_work: Minimum ``B * L**2``.
            tuning: Kernel tuning mode name or ``KernelTuningOptions``;
                ``None`` keeps the DisentangledFlash default.
        """
        if packed not in (True, False, "auto"):
            raise ValueError("packed must be True, False, or 'auto'")
        if packed_min_padding is None:
            packed_min_padding = (
                BaseExtractorModel.DISENTANGLED_FLASH_PACKED_MIN_PADDING
            )
        if isinstance(packed_min_padding, bool) or not isinstance(
            packed_min_padding, (int, float)
        ):
            raise TypeError("packed_min_padding must be a number")
        packed_min_padding = float(packed_min_padding)
        if not 0.0 <= packed_min_padding <= 1.0:
            raise ValueError("packed_min_padding must be between 0 and 1")
        if packed_min_work is None:
            packed_min_work = BaseExtractorModel.DISENTANGLED_FLASH_PACKED_MIN_WORK
        if isinstance(packed_min_work, bool) or not isinstance(packed_min_work, int):
            raise TypeError("packed_min_work must be an int")
        if packed_min_work < 0:
            raise ValueError("packed_min_work must be non-negative")
        if inference is None:
            inference = not self.training
        if not isinstance(inference, bool):
            raise TypeError(
                f"inference must be a bool or None, got {type(inference).__name__}"
            )
        if inference and self.training:
            raise RuntimeError(
                "DisentangledFlash inference requires eval mode; call "
                "model.eval() before enable_disentangled_flash()."
            )
        if not inference and not self.training:
            raise RuntimeError(
                "DisentangledFlash training requires training mode; call "
                "model.train() before enable_disentangled_flash(inference=False)."
            )
        if backend not in {"auto", "torch", "triton"}:
            raise ValueError("backend must be 'auto', 'torch', or 'triton'")

        encoder_config = getattr(self.encoder, "config", None)
        if (
            encoder_config is None
            or encoder_config.__class__.__name__ != "DebertaV2Config"
        ):
            raise TypeError(
                "DisentangledFlash supports Hugging Face DeBERTa-v2/v3 "
                "backbones only."
            )

        try:
            from disentangled_flash import (
                DebertaV2OptimizedEncoder,
                KernelTuningOptions,
                optimize_deberta,
            )
        except ImportError as error:
            raise ImportError(
                "DisentangledFlash is optional; install it with "
                "'pip install gliner2[disentangled-flash]'."
            ) from error

        if isinstance(tuning, str):
            tuning = KernelTuningOptions(mode=tuning)
        elif tuning is not None and not isinstance(tuning, KernelTuningOptions):
            raise TypeError(
                "tuning must be None, a mode name, or KernelTuningOptions"
            )

        existing_mode = getattr(self, "_disentangled_flash_mode", None)
        requested_mode = "inference" if inference else "training"
        if existing_mode == requested_mode:
            return self
        if existing_mode is not None:
            raise RuntimeError(
                "DisentangledFlash is already enabled in "
                f"{existing_mode} mode; create a fresh model to use "
                f"{requested_mode} mode."
            )

        if isinstance(
            getattr(self.encoder, "encoder", None),
            DebertaV2OptimizedEncoder,
        ):
            raise RuntimeError(
                "The DeBERTa encoder is already optimized outside GLiNER2; "
                "refusing to install a second lifecycle hook."
            )

        parameter = next(self.encoder.parameters())
        if parameter.dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
        }:
            raise TypeError(
                "DisentangledFlash supports FP16, BF16, and FP32; encoder "
                f"dtype is {parameter.dtype}."
            )
        enabled_device = parameter.device
        selected_backend = (
            BaseExtractorModel._disentangled_flash_backend_for_device(
                enabled_device
            )
            if backend == "auto"
            else backend
        )

        optimize_deberta(
            self.encoder,
            backend=selected_backend,
            inference=inference,
            sequence_lengths=None,
            fp32_precision="strict",
            tuning=tuning,
        )
        optimized_encoder = self.encoder.encoder
        if not isinstance(optimized_encoder, DebertaV2OptimizedEncoder):
            raise RuntimeError(
                "DisentangledFlash did not install its optimized DeBERTa "
                "encoder."
            )

        self._disentangled_flash_backend = optimized_encoder.backend
        self._disentangled_flash_mode = requested_mode
        self._disentangled_flash_packed = packed
        self._disentangled_flash_packed_min_padding = packed_min_padding
        self._disentangled_flash_packed_min_work = packed_min_work
        self._disentangled_flash_batches = {"packed": 0, "padded": 0}

        # Set by the backbone pre-hook, consumed by the encoder forward.
        pending: dict = {}

        def select_layout(module, args, kwargs) -> bool:
            pending.clear()
            if packed is False or kwargs.get("output_attentions"):
                self._disentangled_flash_batches["padded"] += 1
                return False
            if inference and torch.is_grad_enabled():
                if packed is True:
                    raise RuntimeError(
                        "DisentangledFlash packed inference requires "
                        "torch.no_grad() or torch.inference_mode()."
                    )
                self._disentangled_flash_batches["padded"] += 1
                return False
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None and len(args) > 1:
                attention_mask = args[1]
            layout = BaseExtractorModel._disentangled_flash_packing_layout(
                attention_mask, packed, packed_min_padding, packed_min_work
            )
            if layout is None:
                self._disentangled_flash_batches["padded"] += 1
                return False
            output_hidden_states = kwargs.get("output_hidden_states")
            if output_hidden_states is None:
                output_hidden_states = module.config.output_hidden_states
            pending["layout"] = layout
            pending["all_hidden_states"] = bool(output_hidden_states)
            self._disentangled_flash_batches["packed"] += 1
            return True

        if packed is not False:
            from disentangled_flash import PackedSequenceInfo
            from transformers.modeling_outputs import BaseModelOutput

            padded_forward = optimized_encoder.forward

            def forward(hidden_states, attention_mask, *args, **kwargs):
                layout = pending.pop("layout", None)
                if layout is None:
                    return padded_forward(
                        hidden_states, attention_mask, *args, **kwargs
                    )
                mask, lengths = layout
                if mask.shape != hidden_states.shape[:2]:
                    raise RuntimeError(
                        "DisentangledFlash packed layout does not match the "
                        "encoder input."
                    )
                offsets = [0]
                for length in lengths:
                    offsets.append(offsets[-1] + length)
                info = PackedSequenceInfo(tuple(offsets), lengths, max(lengths))
                cu_seqlens = torch.tensor(
                    offsets, dtype=torch.int32, device=hidden_states.device
                )
                outputs = optimized_encoder.forward_packed(
                    hidden_states[mask],
                    cu_seqlens,
                    info.max_seqlen,
                    output_hidden_states=pending.pop("all_hidden_states"),
                    return_dict=True,
                    packed_info=info,
                )

                def unpack(states: torch.Tensor) -> torch.Tensor:
                    padded = states.new_zeros(
                        (*mask.shape, *states.shape[1:])
                    )
                    padded[mask] = states
                    return padded

                last_hidden_state = unpack(outputs.last_hidden_state)
                if outputs.hidden_states is None:
                    # DebertaV2Model reads encoder_outputs[1][-1].
                    all_hidden_states = (last_hidden_state,)
                else:
                    all_hidden_states = tuple(
                        unpack(states) for states in outputs.hidden_states[:-1]
                    ) + (last_hidden_state,)
                if not kwargs.get("return_dict", True):
                    return (last_hidden_state, all_hidden_states)
                return BaseModelOutput(
                    last_hidden_state=last_hidden_state,
                    hidden_states=all_hidden_states,
                    attentions=None,
                )

            optimized_encoder.forward = forward

        if not inference:
            if packed is not False:
                def select_training_layout(module, args, kwargs):
                    select_layout(module, args, kwargs)

                self._disentangled_flash_hook_handle = (
                    self.encoder.register_forward_pre_hook(
                        select_training_layout,
                        with_kwargs=True,
                    )
                )
            logger.info(
                "Enabled DisentangledFlash training backend %s (packed=%s)",
                optimized_encoder.backend,
                packed,
            )
            return self

        def prepare_active_length(module, args, kwargs):
            if self.training or module.training:
                raise RuntimeError(
                    "DisentangledFlash inference cannot run in training mode."
                )
            if getattr(module, "encoder", None) is not optimized_encoder:
                raise RuntimeError(
                    "The DisentangledFlash encoder replacement is no longer "
                    "active."
                )

            current_parameter = next(module.parameters())
            if current_parameter.device != enabled_device:
                raise RuntimeError(
                    "The model device changed after DisentangledFlash was "
                    "enabled. Move the model to its final device first, then "
                    "enable the backend."
                )
            if current_parameter.dtype not in {
                torch.float16,
                torch.bfloat16,
                torch.float32,
            }:
                raise TypeError(
                    "DisentangledFlash supports FP16, BF16, and FP32; encoder "
                    f"dtype is {current_parameter.dtype}."
                )

            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if input_ids is None or input_ids.ndim != 2:
                raise ValueError(
                    "DisentangledFlash requires input_ids with shape [B, L] "
                    "to select a prepared sequence-length plan."
                )

            if select_layout(module, args, kwargs):
                return

            sequence_length = int(input_ids.shape[1])
            prepared_lengths = tuple(
                sorted(int(length) for length in optimized_encoder._prepared_plans)
            )
            if sequence_length not in prepared_lengths:
                optimized_encoder.prepare_for_inference(
                    tuple(sorted(set(prepared_lengths) | {sequence_length}))
                )
            optimized_encoder.activate_shape(sequence_length)

        self._disentangled_flash_hook_handle = (
            self.encoder.register_forward_pre_hook(
                prepare_active_length,
                with_kwargs=True,
            )
        )
        logger.info(
            "Enabled DisentangledFlash inference backend %s",
            optimized_encoder.backend,
        )
        return self

    @staticmethod
    def _load_encoder(
        model_name: str,
        encoder_config: Optional[PretrainedConfig] = None,
        attn_implementation: Optional[str] = "sdpa",
        use_flashdeberta: Optional[bool] = None,
    ) -> nn.Module:
        """Load a shared optimized encoder with a safe eager fallback."""
        config = encoder_config
        if config is None:
            config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=True
            )

        if use_flashdeberta is None:
            use_flashdeberta = bool(os.environ.get("USE_FLASHDEBERTA"))
        flashdeberta_available = (
            importlib.util.find_spec("flashdeberta") is not None
        )
        if (
            config.__class__.__name__ == "DebertaV2Config"
            and use_flashdeberta
            and flashdeberta_available
        ):
            try:
                from flashdeberta import FlashDebertaV2Model

                logger.info("Using FlashDeBERTa backend")
                if encoder_config is not None:
                    return FlashDebertaV2Model(config).float()
                return FlashDebertaV2Model.from_pretrained(model_name).float()
            except Exception as error:  # noqa: BLE001 - optional backend boundary
                warnings.warn(
                    "FlashDeBERTa could not initialize; falling back to the "
                    f"standard Transformers encoder ({error})",
                    RuntimeWarning,
                    stacklevel=2,
                )

        def load(implementation: Optional[str]) -> nn.Module:
            # Load the encoder in fp32 to match the fp32 task heads. transformers>=5
            # otherwise loads a checkpoint in its stored dtype (e.g. deberta-v3 ships
            # fp16), which then mismatches the fp32 heads at F.linear. Reduced-precision
            # training is applied at runtime via autocast, not the loaded weights.
            kwargs = {"trust_remote_code": True, "dtype": torch.float32}
            if implementation:
                kwargs["attn_implementation"] = implementation
            with warnings.catch_warnings():
                # Transformers' DeBERTa module decorates helpers with
                # ``torch.jit.script`` at import time, which PyTorch flags as
                # unsupported on Python 3.14+. The helpers still run eagerly,
                # so the warning is not actionable for GLiNER2 users.
                warnings.filterwarnings(
                    "ignore",
                    message=r"`torch\.jit\.script` is not supported",
                    category=FutureWarning,
                )
                if encoder_config is not None:
                    encoder = AutoModel.from_config(config, **kwargs)
                else:
                    encoder = AutoModel.from_pretrained(model_name, **kwargs)
            # Transformers 5 honors a serialized encoder dtype during
            # ``from_config``. GLiNER2 task heads are initialized in FP32, so a
            # half-precision encoder would emit activations that cannot enter
            # those heads. Build the complete model consistently in FP32;
            # callers can still cast it atomically with ``model.half()`` or
            # ``model.to(dtype=...)`` afterwards.
            return encoder.float()

        # Degrade in speed order -- requested, then sdpa, then eager -- instead of
        # dropping straight to eager, which is the slowest of the three and used to
        # be the silent cost of a missing optional backend. eager stays the last
        # rung because deberta-v2 supports neither FlashAttention 2 nor sdpa.
        # "flash_attention_2" means the PIP PACKAGE to transformers, and prebuilt wheels
        # stop at cp313/torch2.9 -- so on this stack it is always rejected and the loader
        # silently drops to sdpa, which on a bf16 ModernBERT is a correctness failure, not
        # a slowdown. The Hub kernel registry provides the same kernel under a repo id and
        # is accepted on both transformers 5.6 (kernels 0.12) and 5.13 (kernels 0.15.2),
        # verified on each.
        #
        # Trying the repo form BEFORE degrading repairs every checkpoint already published
        # with the plain string -- twelve of ours -- without re-uploading any of them.
        requested = attn_implementation
        if requested in ("flash_attention_2", _HUB_FLASH_ATTN_2) and not torch.cuda.is_available():
            # A checkpoint trained on GPU saves flash attention into its config, under
            # either the plain name or the hub repo id. Off CUDA BOTH must fall through to
            # sdpa rather than accept-then-explode: transformers normalizes the plain name
            # to the hub id, load succeeds, and the first forward raises
            # KeyError('kernels-community/flash-attn2') from ALL_ATTENTION_FUNCTIONS.
            # Measured on gate2-mmbert-v2 (trained attn_implementation=flash_attention_2)
            # on Apple Silicon -- the hub-id-only guard did not catch it.
            requested = "sdpa"
        candidates = [requested]
        # CUDA-ONLY, and the guard is load-bearing: the Hub kernel is ACCEPTED at load time
        # on a CPU box and then raises KeyError at the first forward, which is worse than
        # the silent sdpa fallback it was added to fix. Verified by running a forward, not
        # just a load -- checking construction alone is what missed this.
        if requested in ("flash_attention_2", _HUB_FLASH_ATTN_2) and torch.cuda.is_available():
            candidates.append(_HUB_FLASH_ATTN_2)
            # Register the repo id in the MASK registry before anything can load under it
            # -- otherwise the load succeeds and the first TRAINING forward raises
            # KeyError(repo id). See _register_hub_flash_attn_mask.
            _register_hub_flash_attn_mask()
        # A FALL OUT OF FA2 GOES TO EAGER, NEVER TO SDPA. sdpa + bf16 on a ModernBERT
        # encoder is a CORRECTNESS failure and not a slowdown -- finite forward, NaN
        # backward -- so the "safe" middle rung is the one that silently breaks a run,
        # while eager is merely slow. Leaving sdpa in this chain is how a 42-minute arm
        # trained on a numerically broken path on 2026-09-14.
        #
        # A request for sdpa ITSELF still degrades to eager as before: deberta-v2 supports
        # neither FA2 nor sdpa and must reach eager.
        if requested:
            tail = (("eager",) if requested in ("flash_attention_2", _HUB_FLASH_ATTN_2)
                    else ("sdpa", "eager"))
            candidates += [c for c in tail if c not in candidates]

        last_error: Exception
        for index, implementation in enumerate(candidates):
            try:
                return load(implementation)
            # KeyError IS LOAD-BEARING HERE. When the Hub kernel cannot be fetched --
            # a kernels/transformers version mismatch, no Hub access, or the repo 404ing
            # as kernels-community/flash-attn2 has done to a valid token before -- the
            # config still names it and the first forward raises
            # KeyError('kernels-community/flash-attn2') out of ALL_ATTENTION_FUNCTIONS.
            # That escaped this tuple, so the degrade chain never ran and the caller got
            # a bare KeyError instead of the warning below naming the pin to check.
            except (TypeError, ValueError, ImportError, KeyError) as error:
                last_error = error
                if index + 1 < len(candidates):
                    # A silent downgrade is how a whole training run gets lost. Measured
                    # 2026-08-10: a stale `kernels` pin (0.12 against transformers 5.13,
                    # which needs >=0.15.2) made FA2 unavailable, the loader fell back to
                    # sdpa, and the run died at step ~50 with non-finite losses after
                    # running 11x slow -- 4.6 samples/s where FA2 gives 51.4. The only
                    # evidence beforehand was this one warning in a log full of them.
                    #
                    # So callers that require FA2 can demand it: GLINER2_STRICT_ATTN=1
                    # turns the downgrade into a failure at load time, before any GPU
                    # hours are spent. Default stays permissive because deberta-v2
                    # supports neither FA2 nor sdpa and must reach eager.
                    # Retrying the SAME kernel under its Hub repo id is a repair, not a
                    # downgrade: FA2 is still what ends up loaded. Say so quietly and do
                    # not let strict mode reject it, or every checkpoint published with
                    # the plain string would fail under GLINER2_STRICT_ATTN.
                    if candidates[index + 1] == _HUB_FLASH_ATTN_2:
                        logger.info(
                            "attn_implementation=%r resolves to the Hub kernel %r "
                            "(transformers reads the plain name as the pip package). "
                            "FlashAttention 2 is still in use.",
                            implementation, _HUB_FLASH_ATTN_2,
                        )
                        continue
                    # `and index == 0` stood here and made this guard UNREACHABLE on the
                    # only path that matters. Candidates are
                    # [flash_attention_2, <hub id>, sdpa, eager]: index 0 failing is a
                    # REPAIR and `continue`s above, so the real downgrade always happens at
                    # index 1 -- where `index == 0` is false and strict mode was skipped.
                    # Measured 2026-09-14: GLINER2_STRICT_ATTN=1 was set, the hub kernel was
                    # rejected for a version mismatch, and both arms trained 42 minutes each
                    # on sdpa, which on bf16 ModernBERT is a correctness failure. The guard
                    # written to prevent exactly that watched it happen.
                    #
                    # The condition is about WHAT WAS ASKED FOR, not where in the list we
                    # are: refuse any fall out of FA2, at whatever index it occurs. A
                    # request for sdpa degrading to eager is still permitted, because
                    # deberta-v2 supports neither FA2 nor sdpa and must reach eager.
                    if os.environ.get("GLINER2_STRICT_ATTN") and requested in (
                        "flash_attention_2", _HUB_FLASH_ATTN_2
                    ):
                        raise RuntimeError(
                            f"attn_implementation={implementation!r} unavailable and "
                            f"GLINER2_STRICT_ATTN is set, so the {candidates[index + 1]!r} "
                            f"fallback is refused: on a bf16 ModernBERT that fallback is a "
                            f"CORRECTNESS failure (non-finite losses), not a slowdown. "
                            f"Underlying error: {error}"
                        ) from error
                    # WARN ABOUT WHAT WAS ASKED FOR, and only then. A request for sdpa
                    # degrading to eager is EXPECTED -- deberta-v2 supports neither FA2
                    # nor sdpa and must reach eager -- so warning on it fires on every
                    # ordinary deberta load and trains people to ignore the one warning
                    # that means a run is about to be lost. The correctness failure this
                    # exists to catch (FA2 -> sdpa on a bf16 ModernBERT) can only happen
                    # when FA2 was requested, which is exactly the condition below.
                    #
                    # `attn_implementation`, not `requested`: off CUDA `requested` was
                    # rewritten to "sdpa" above, so using it would both suppress the
                    # warning on the path that needs it and drop the name the caller
                    # actually passed out of the message.
                    if attn_implementation in ("flash_attention_2", _HUB_FLASH_ATTN_2):
                        warnings.warn(
                            f"Encoder rejected attn_implementation={implementation!r}; "
                            f"falling back to {candidates[index + 1]!r} ({error}). "
                            f"flash_attention_2 was requested and is NOT in use. On a "
                            f"ModernBERT encoder trained in bf16 this is a correctness "
                            f"issue, not just speed: sdpa+bf16 there produces non-finite "
                            f"losses (measured), so install 'kernels' to keep FA2. Set "
                            f"GLINER2_STRICT_ATTN=1 to make this an error instead.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
        raise last_error

    def _encoder_autocast(self):
        """Autocast context the encoder needs, or a no-op.

        FlashAttention 2 kernels accept only fp16/bf16, but ``_load_encoder``
        deliberately loads weights in fp32 to match the fp32 task heads -- reduced
        precision is applied at runtime. Training wraps the forward in autocast, so
        FA2 sees bf16 there; INFERENCE does not, so an FA2 encoder raises
        "FlashAttention only support fp16 and bf16 data type" on every call. That
        breaks eval metrics (compute_metrics goes through batch_extract) and the
        public extract path alike. Supply the missing wrapper at the encoder call.

        A no-op unless the encoder actually uses flash attention on CUDA, and safe
        to nest inside the training autocast already in effect.
        """
        implementation = str(getattr(self.config, "attn_implementation", "") or "")
        device_type = next(self.parameters()).device.type
        if "flash" not in implementation or device_type != "cuda":
            return contextlib.nullcontext()
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    def encode_tokens(
        self,
        batch,
        hidden_states: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Return padded ``[B, L, H]`` token states for a collated batch.

        This is the only inference call into ``self.encoder``. A serving engine
        that already ran the encoder passes its per-row output instead.

        Args:
            batch: A ``PreprocessedBatch`` on the model's device.
            hidden_states: Optional precomputed encoder output, one
                ``(tokens_i, hidden)`` tensor per row, where ``tokens_i`` is
                that row's unpadded length in ``batch.attention_mask``.

        Returns:
            Token states shaped like ``batch.input_ids`` plus a hidden axis.

        Raises:
            ValueError: If ``hidden_states`` does not match the batch, or the
                model has no encoder and ``hidden_states`` is ``None``.
        """
        if hidden_states is None:
            if getattr(self, "encoder", None) is None:
                raise ValueError(
                    f"{type(self).__name__} was built without an encoder; "
                    "pass hidden_states"
                )
            with self._encoder_autocast():
                return self.encoder(
                    input_ids=batch.input_ids, attention_mask=batch.attention_mask
                ).last_hidden_state
        return _pad_hidden_states(
            hidden_states,
            batch.attention_mask,
            hidden_size=self.hidden_size,
            dtype=next(self.parameters()).dtype,
        )

    def task_module_names(self) -> Tuple[str, ...]:
        raise NotImplementedError

    def default_schema(self) -> Optional[dict]:
        """The extraction schema this model was trained on, if one shipped in the
        config (entities / events / relations / classifications). A recommended
        default -- the model is open-vocabulary and not limited to it. ``None`` for
        checkpoints saved before schemas were co-located."""
        return getattr(self.config, "default_schema", None)

    def save_pretrained(self, *args, **kwargs):
        self.config.architecture = getattr(self, "architecture", self.config.architecture)
        self.config.architectures = [type(self).__name__]
        result = super().save_pretrained(*args, **kwargs)
        if args:
            rewrite_config_as_utf8(args[0])
        elif "save_directory" in kwargs:
            rewrite_config_as_utf8(kwargs["save_directory"])
        return result

    def set_word_splitter(self, word_splitter):
        """Replace the word splitter used at inference and training collation.

        ``word_splitter`` may be a built-in name (``"whitespace"`` or
        ``"char"``) or a callable yielding ``(token, start, end)`` with
        exclusive-end offsets into the original text. The default
        ``"whitespace"`` strategy is the one used to train public checkpoints.

        ``"char"`` is suitable for languages without whitespace-delimited
        words, such as Chinese. Changing a pretrained model's word boundaries
        can affect quality unless training used the same splitter.
        """
        from gliner2.processing.word_splitter import resolve_word_splitter

        processor = getattr(self, "processor", None)
        if processor is None:
            raise AttributeError(
                f"{type(self).__name__} has no processor to attach a word splitter to"
            )
        processor.word_splitter = resolve_word_splitter(word_splitter)
        return self
