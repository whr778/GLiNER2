"""
GLiNER2 World-Class Trainer
===========================

Production-grade training infrastructure with flexible data input.

Supported Data Formats:
-----------------------
1. Single JSONL file path (str or Path)
2. List of JSONL file paths
3. List of InputExample objects
4. TrainingDataset object
5. List of raw dict records ({"input": ..., "output": ...} format)

Basic Examples:
--------------
    >>> from gliner2.training.data import InputExample, TrainingDataset
    >>> from gliner2.training.trainer import TrainingConfig, GLiNER2Trainer
    >>>
>>> # 1. From list of InputExample
    >>> examples = [
    ...     InputExample(text="John works at Google.", entities={"person": ["John"], "company": ["Google"]}),
    ...     InputExample(text="Apple released iPhone.", entities={"company": ["Apple"], "product": ["iPhone"]}),
    ... ]
    >>> trainer = GLiNER2Trainer(model, config)
    >>> trainer.train(train_data=examples)
    >>>
>>> # 2. From JSONL file(s)
    >>> trainer.train(train_data="train.jsonl")
    >>> trainer.train(train_data=["train1.jsonl", "train2.jsonl"])
>>>
>>> # 3. From TrainingDataset
>>> dataset = TrainingDataset.load("train.jsonl")
>>> trainer.train(train_data=dataset)
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import random
import re
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import autocast
from torch.amp import GradScaler  # device-typed; we instantiate it conditionally
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm.auto import tqdm

from gliner2.training.parallel import BatchDataParallel, _AutocastModule
from gliner2.training.eta import _fmt_seconds

from gliner2.processor import SchemaTransformer, SamplingConfig

# Import training data classes
from gliner2.training.data import (
    InputExample, TrainingDataset, ValidationError,
    DataFormat, detect_data_format, DataLoader_Factory, TrainDataInput
)

from peft import PeftModel
from peft.tuners.lora.layer import LoraLayer as _PeftLoraLayer

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Complete training configuration.
    
    Parameters
    ----------
    output_dir : str
        Directory for saving checkpoints and logs.
    experiment_name : str
        Name of the experiment (used for logging).
    num_epochs : int
        Number of training epochs.
    max_steps : int
        Maximum training steps (-1 = determined by epochs).
    batch_size : int
        Training batch size per device.
    eval_batch_size : int
        Evaluation batch size.
    gradient_accumulation_steps : int
        Number of gradient accumulation steps.
    encoder_lr : float
        Learning rate for encoder parameters.
    task_lr : float
        Learning rate for task-specific parameters.
    weight_decay : float
        Weight decay for AdamW optimizer.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    scheduler_type : str
        LR scheduler type: "linear", "cosine", "cosine_restarts", "constant".
    warmup_ratio : float
        Warmup ratio (portion of total steps).
    warmup_steps : int
        Explicit warmup steps (overrides warmup_ratio if > 0).
    fp16 : bool
        Use FP16 mixed precision.
    bf16 : bool
        Use BF16 mixed precision.
    eval_strategy : str
        When to evaluate and save: "epoch", "steps", or "no".
    eval_steps : int
        Evaluate and save every N steps (if eval_strategy="steps").
    save_total_limit : int
        Maximum checkpoints to keep.
    save_best : bool
        Save best model based on metric.
    metric_for_best : str
        Metric to use for best model selection.
    greater_is_better : bool
        Whether higher metric is better.
    logging_steps : int
        Log every N steps (updates progress bar metrics).
    report_to_wandb : bool
        Enable Weights & Biases logging.
    wandb_project : str, optional
        W&B project name.
    early_stopping : bool
        Enable early stopping.
    early_stopping_patience : int
        Patience for early stopping.
    num_workers : int
        DataLoader workers.
    seed : int
        Random seed.
    validate_data : bool
        Validate training data before training.
    use_lora : bool
        Enable LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
    lora_r : int
        LoRA rank (bottleneck dimension). Higher = more parameters but better approximation.
        Typical values: 4, 8, 16, 32, 64.
    lora_alpha : float
        LoRA scaling factor. Final scaling is alpha/r. Typical: 2*r.
    lora_dropout : float
        Dropout probability for LoRA layers.
    lora_target_modules : List[str]
        Module groups to apply LoRA to. Options:
        - "encoder": All encoder layers (query, key, value, dense)
        - "encoder.query": Only query layers in encoder
        - "encoder.key": Only key layers in encoder
        - "encoder.value": Only value layers in encoder
        - "encoder.dense": Only dense (FFN) layers in encoder
        - "span_rep": All linear layers in span representation
        - "classifier": All linear layers in classifier head
        - "count_embed": All linear layers in count embedding
        - "count_pred": All linear layers in count prediction
        Default: All modules for maximum adaptation.
    save_adapter_only : bool
        When use_lora=True, save only adapter weights (not full model).
    """
    output_dir: str = "./output"
    experiment_name: str = "gliner2"
    num_epochs: int = 10
    max_steps: int = -1
    batch_size: int = 2
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    encoder_lr: float = 1e-5
    task_lr: float = 5e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    num_cycles: float = 0.5
    fp16: bool = True
    bf16: bool = False
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_total_limit: int = 3
    save_best: bool = True
    metric_for_best: str = "eval_loss"
    greater_is_better: bool = False
    logging_steps: int = 1
    logging_first_step: bool = True
    report_to_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    gradient_checkpointing: bool = False  # trade compute for activation memory
    checkpoint_restart: Optional[str] = None  # None | "highest" | "last" -- resume mid-run
    seed: int = 42
    deterministic: bool = False
    local_rank: int = -1
    # Multi-GPU via nn.DataParallel (single process, one thread per GPU).
    # CUDA-only: ignored on CPU/MPS, with a single GPU, or when the DDP path
    # (local_rank >= 0) is active. The DataLoader batch is split across GPUs,
    # so per-GPU work is batch_size / num_gpus. device_ids=None uses all
    # visible CUDA devices. autocast (bf16/fp16) is preserved on the replica
    # threads via a wrapper -- see gliner2.training.parallel.
    data_parallel: bool = False
    data_parallel_device_ids: Optional[List[int]] = None
    debug: bool = False
    max_train_samples: int = -1
    max_eval_samples: int = -1
    validate_data: bool = True
    max_len: Optional[int] = None
    # Sliding-window chunking. When sliding_window=True, max_len is
    # interpreted as the window size in SUBWORD tokens (the encoder's
    # tokenizer); records longer than the window are expanded into
    # overlapping window_stride-spaced chunks at dataset-load time.
    # See gliner2.training.chunking for the per-task annotation filters.
    sliding_window: bool = False
    window_stride: int = 256

    # LoRA Configuration (Parameter-Efficient Fine-Tuning)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_use_dora: bool = False
    lora_target_modules: List[str] = field(default_factory=lambda: ["encoder", "span_rep", "classifier", "count_embed", "count_pred"])
    save_adapter_only: bool = True  # Only applies when use_lora=True

    def __post_init__(self):
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16")
        if self.bf16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            logger.warning("bf16 not supported on this CUDA device, falling back to fp16")
            self.bf16 = False
            self.fp16 = True
        
        # Validate logging_steps
        if self.logging_steps <= 0:
            raise ValueError(f"logging_steps must be > 0, got {self.logging_steps}")
        
        # Validate batch_size
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        
        if self.eval_batch_size <= 0:
            raise ValueError(f"eval_batch_size must be > 0, got {self.eval_batch_size}")
        
        # Validate gradient_accumulation_steps
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be > 0, got {self.gradient_accumulation_steps}")
        
        # Validate LoRA configuration
        if self.use_lora:
            if self.lora_r <= 0:
                raise ValueError(f"lora_r must be > 0, got {self.lora_r}")
            if self.lora_alpha <= 0:
                raise ValueError(f"lora_alpha must be > 0, got {self.lora_alpha}")
            if not 0 <= self.lora_dropout < 1:
                raise ValueError(f"lora_dropout must be in [0, 1), got {self.lora_dropout}")
            if not self.lora_target_modules:
                raise ValueError("lora_target_modules cannot be empty when use_lora=True")

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        with open(path) as f:
            return cls(**json.load(f))


# =============================================================================
# Dataset
# =============================================================================

class ExtractorDataset(Dataset):
    """
    Dataset for GLiNER2 training with multi-format support.

    Supports all formats through DataLoader_Factory:
    - JSONL file path(s)
    - List of InputExample objects
    - TrainingDataset object
    - List of raw dict records
    
    Examples
    --------
    >>> # From JSONL
    >>> dataset = ExtractorDataset("train.jsonl")
    
    >>> # From multiple JSONL files
    >>> dataset = ExtractorDataset(["train1.jsonl", "train2.jsonl"])
    
    >>> # From InputExample list
    >>> dataset = ExtractorDataset(examples)
    """

    def __init__(
            self,
            data: TrainDataInput,
            max_samples: int = -1,
            shuffle: bool = True,
            seed: int = 42,
            validate: bool = False,
    ):
        """
        Initialize dataset from various input formats.

        Parameters
        ----------
        data : TrainDataInput
            Training data in any supported format.
        max_samples : int, default=-1
            Maximum samples to use (-1 = all).
        shuffle : bool, default=True
            Whether to shuffle the data.
        seed : int, default=42
            Random seed for shuffling.
        validate : bool, default=False
            Whether to validate the data. Validation is always strict:
            checks that entity spans, relation values, and structure
            field values exist in the text.
        """
        self.data = DataLoader_Factory.load(
            data=data,
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed,
            validate=validate,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, Dict]:
        record = self.data[idx]
        # Handle both formats
        if "input" in record:
            return record["input"], record["output"]
        else:
            return record["text"], record["schema"]

    # Factory methods for explicit creation
    @classmethod
    def from_jsonl(cls, paths: Union[str, Path, List], **kwargs) -> 'ExtractorDataset':
        """Create from JSONL file(s)."""
        return cls(paths, **kwargs)

    @classmethod
    def from_examples(cls, examples: List[InputExample], **kwargs) -> 'ExtractorDataset':
        """Create from list of InputExample."""
        return cls(examples, **kwargs)

    @classmethod
    def from_training_dataset(cls, dataset: TrainingDataset, **kwargs) -> 'ExtractorDataset':
        """Create from TrainingDataset."""
        return cls(dataset, **kwargs)

    @classmethod
    def from_dicts(cls, dicts: List[Dict], **kwargs) -> 'ExtractorDataset':
        """Create from list of dicts."""
        return cls(dicts, **kwargs)


# =============================================================================
# Collator
# =============================================================================

class ExtractorCollator:
    """Data collator that converts raw records to model inputs."""

    def __init__(self, processor: SchemaTransformer, is_training: bool = True, max_len=None):
        self.processor = processor
        self.is_training = is_training
        self.max_len = max_len

    def __call__(self, batch: List[Tuple[str, Dict]]):
        """
        Convert batch of (text, schema) tuples to PreprocessedBatch.

        Args:
            batch: List of (text, schema) tuples from dataset

        Returns:
            PreprocessedBatch ready for model.forward()
        """
        if self.is_training:
            return self.processor.collate_fn_train(batch, max_len=self.max_len)
        else:
            return self.processor.collate_fn_inference(batch, max_len=self.max_len)


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    loss: float = 0.0
    classification_loss: float = 0.0
    structure_loss: float = 0.0
    count_loss: float = 0.0
    learning_rate: float = 0.0
    epoch: float = 0.0
    step: int = 0
    samples_seen: int = 0
    throughput: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# =============================================================================
# Scheduler Factory
# =============================================================================

def get_scheduler(optimizer, scheduler_type, num_training_steps, num_warmup_steps, num_cycles=0.5):
    """Create learning rate scheduler."""
    def lr_lambda_linear(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - step) / float(max(1, num_training_steps - num_warmup_steps)))

    def lr_lambda_cosine(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def lr_lambda_cosine_restarts(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((num_cycles * progress) % 1.0))))

    def lr_lambda_constant(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        return 1.0

    schedulers = {
        "linear": lr_lambda_linear,
        "cosine": lr_lambda_cosine,
        "cosine_restarts": lr_lambda_cosine_restarts,
        "constant": lr_lambda_constant,
    }

    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

    return LambdaLR(optimizer, schedulers[scheduler_type])


# =============================================================================
# Main Trainer
# =============================================================================

class GLiNER2Trainer:
    """
    World-class trainer for GLiNER2 with flexible multi-format data input.

    Parameters
    ----------
    model : nn.Module
        The GLiNER2 model to train.
    config : TrainingConfig
        Training configuration.
    processor : SchemaTransformer, optional
        Schema processor. If None, uses model.processor.
    train_data : TrainDataInput, optional
        Training data (can be provided here or in train()).
    eval_data : TrainDataInput, optional
        Evaluation data.
    compute_metrics : Callable, optional
        Custom metrics function.

    Supported Data Formats
    ----------------------
    - Single JSONL file path (str or Path)
    - List of JSONL file paths
    - List of InputExample objects
    - TrainingDataset object
    - List of raw dict records

    Examples
    --------
    >>> # With InputExample list
    >>> examples = [InputExample(...), InputExample(...)]
    >>> trainer = GLiNER2Trainer(model, config)
    >>> trainer.train(train_data=examples)

    >>> # With JSONL file
    >>> trainer.train(train_data="train.jsonl")

    >>> # With multiple JSONL files
    >>> trainer.train(train_data=["train1.jsonl", "train2.jsonl"])

    >>> # With TrainingDataset
    >>> dataset = TrainingDataset.load("train.jsonl")
    >>> trainer.train(train_data=dataset)
    """

    def __init__(
            self,
            model: nn.Module,
            config: TrainingConfig,
            processor: SchemaTransformer = None,
            train_data: TrainDataInput = None,
            eval_data: TrainDataInput = None,
            compute_metrics: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.processor = processor or getattr(model, 'processor', None)
        if self.processor is None:
            raise ValueError("Processor must be provided or model must have .processor attribute")

        self.train_data = train_data
        self.eval_data = eval_data
        self.compute_metrics = compute_metrics

        self._setup_seed()
        self._setup_device()
        self._setup_gradient_checkpointing()
        self._setup_output_dir()
        self._setup_logging()

        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf') if not config.greater_is_better else float('-inf')
        self.patience_counter = 0
        self.train_metrics_history = []
        self.eval_metrics_history = []

        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.wandb_run = None
        self.progress_bar = None
        
        # LoRA state
        self.lora_layers = {}
        self._setup_lora()

        # Multi-GPU forward wrapper (DDP / DataParallel; no-op on single device).
        self._setup_parallel()

    def _setup_seed(self):
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if self.config.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                torch.backends.cudnn.benchmark = True

    def _setup_device(self):
        # The cuDNN fused-attention (SDPA) backend intermittently fails to
        # execute its graph on certain sequence shapes -- with variable-length
        # mmBERT/ModernBERT inputs (up to 8192 tokens) this surfaces mid-run as
        # "mha_graph.execute().is_good() ... got False" and aborts training.
        # Disabling it lets SDPA fall back to the stable flash / mem-efficient
        # kernels. Global flag, so set it on every rank before the first forward.
        if torch.cuda.is_available():
            torch.backends.cuda.enable_cudnn_sdp(False)

        # DDP: torchrun sets LOCAL_RANK/RANK/WORLD_SIZE. Initialize the process
        # group once (nccl on CUDA, gloo for CPU runs) and pin this rank's device.
        local_rank = int(os.environ.get("LOCAL_RANK", self.config.local_rank))
        self.config.local_rank = local_rank
        if local_rank >= 0:
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                self.device = torch.device("cuda", local_rank)
                backend = "nccl"
            else:
                self.device = torch.device("cpu")
                backend = "gloo"
                self.config.fp16 = self.config.bf16 = False
            if not dist.is_initialized():
                dist.init_process_group(backend=backend)
            self.is_distributed = True
            self.model.to(self.device)
            logger.info(f"DDP rank {dist.get_rank()}/{dist.get_world_size()} on {self.device}")
            return
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.is_distributed = False
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.is_distributed = False
            if self.config.fp16 or self.config.bf16:
                logger.warning(
                    "Mixed precision disabled on MPS (fp16/bf16 with GradScaler is "
                    "CUDA-only; the model can still be quantized post-training via "
                    "model.quantize()).")
                self.config.fp16 = False
                self.config.bf16 = False
            if self.config.pin_memory:
                logger.info("Disabling pin_memory on MPS (not supported).")
                self.config.pin_memory = False
        else:
            self.device = torch.device("cpu")
            self.is_distributed = False
            if self.config.fp16 or self.config.bf16:
                logger.warning("Mixed precision disabled on CPU")
                self.config.fp16 = False
                self.config.bf16 = False
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

    def _setup_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on the encoder when configured.

        Recomputes activations during backward instead of storing them, cutting
        training-activation memory at the cost of ~20-30% more compute. This is
        the main lever for fitting a large encoder on constrained memory (e.g.
        MPS unified memory), where it lowers the high-water mark that the eval
        pass inherits.
        """
        if not getattr(self.config, "gradient_checkpointing", False):
            return
        enc = getattr(self.model, "encoder", None)
        if enc is None or not hasattr(enc, "gradient_checkpointing_enable"):
            logger.warning(
                "gradient_checkpointing requested but the encoder does not "
                "support it; skipping."
            )
            return
        if hasattr(enc, "config"):
            enc.config.use_cache = False  # KV cache is incompatible with checkpointing
        try:
            enc.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:  # older transformers without the kwargs argument
            enc.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing on the encoder (use_reentrant=False).")

    def _setup_output_dir(self):
        self.output_dir = Path(self.config.output_dir)
        self.logs_dir = self.output_dir / "logs"
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logs_dir.mkdir(exist_ok=True)
            self.config.save(str(self.output_dir / "training_config.json"))

    def _setup_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO if self.is_main_process else logging.WARNING,
        )
        
        # W&B setup (HuggingFace style)
        self.wandb_run = None
        if self.config.report_to_wandb and self.is_main_process:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.config.wandb_project or self.config.experiment_name,
                    entity=self.config.wandb_entity,
                    name=self.config.wandb_run_name or f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=asdict(self.config),
                    tags=self.config.wandb_tags,
                    notes=self.config.wandb_notes,
                    dir=str(self.output_dir),
                )
                logger.info(f"W&B run: {self.wandb_run.url}")
            except ImportError:
                logger.warning("wandb not installed. Run: pip install wandb")
                self.config.report_to_wandb = False

    def _setup_lora(self):
        """Setup LoRA for parameter-efficient fine-tuning if enabled."""
        if not self.config.use_lora:
            logger.info("LoRA is disabled")
            return

        for p in self.model.parameters():
            p.requires_grad = False
        logger.info("Froze all model parameters for LoRA training")

        self.model = self.model.apply_lora(
            r=self.config.lora_r, alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
            targets=self.config.lora_target_modules,
            use_dora=self.config.lora_use_dora,
        )
        self.lora_layers = {n: m for n, m in self.model.named_modules() if isinstance(m, _PeftLoraLayer)}
        self.model._lora_layers = self.lora_layers

        lora_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        pct = (lora_params / total_params * 100) if total_params > 0 else 0.0
        logger.info(f"LoRA setup complete: {lora_params:,} trainable / {total_params:,} total ({pct:.2f}%)")

    def _setup_parallel(self):
        """Wrap the model for multi-GPU forward (DDP or nn.DataParallel).

        ``self.model`` stays the raw module (optimizer, grad-clipping, and
        checkpoint saving all use it); only the forward pass goes through
        ``self._fwd_model``. DDP (launched via torchrun, i.e. local_rank >= 0)
        takes precedence; otherwise DataParallel is used when ``data_parallel``
        is set and >=2 CUDA devices are available; otherwise it's a no-op.
        """
        self._fwd_model = self.model
        if self.is_distributed:
            device_ids = [self.config.local_rank] if self.device.type == "cuda" else None
            self._fwd_model = DistributedDataParallel(
                self.model, device_ids=device_ids,
                # the model runs different task heads per sample, so some params
                # get no gradient on a given batch -- DDP must tolerate that.
                find_unused_parameters=True,
            )
            logger.info(f"DDP wrapping model on {self.device}")
            return
        if not self.config.data_parallel:
            return
        if self.device.type != "cuda" or torch.cuda.device_count() < 2:
            logger.info("data_parallel requested but <2 CUDA devices available; "
                        "running on a single device.")
            return
        device_ids = self.config.data_parallel_device_ids or list(range(torch.cuda.device_count()))
        if len(device_ids) < 2:
            return
        # nn.DataParallel requires the model to live on device_ids[0] (its
        # source/output device); honour a custom device_ids list that doesn't
        # start at 0 by moving the model there.
        primary = torch.device("cuda", device_ids[0])
        if self.device != primary:
            self.device = primary
            self.model.to(self.device)
        # parallel_apply re-enables autocast on replica threads but drops the
        # dtype (thread-local, CUDA default fp16), so a bf16 run would silently
        # execute replicas in fp16 and overflow to NaN. Wrap the model so each
        # replica re-opens autocast with the configured dtype.
        inner = self.model
        if self.config.bf16 or self.config.fp16:
            amp_dtype = torch.bfloat16 if self.config.bf16 else torch.float16
            inner = _AutocastModule(self.model, self.device.type, amp_dtype)
        self._fwd_model = BatchDataParallel(inner, device_ids=device_ids)
        logger.info(f"DataParallel enabled across CUDA devices {device_ids}")

    @property
    def is_main_process(self) -> bool:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return self.config.local_rank <= 0

    def _barrier(self) -> None:
        """Synchronize all DDP ranks (no-op when not distributed)."""
        if self.is_distributed and dist.is_initialized():
            dist.barrier()

    def _sync_flag(self, flag: bool) -> bool:
        """Broadcast a boolean decision (e.g. early-stop) from rank 0 to all ranks."""
        if not (self.is_distributed and dist.is_initialized()):
            return flag
        t = torch.tensor([1 if flag else 0], device=self.device)
        dist.broadcast(t, src=0)
        return bool(t.item())

    @staticmethod
    def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safely divide two numbers, returning default if denominator is zero."""
        if denominator == 0:
            return default
        return numerator / denominator
    
    def _validate_training_setup(self, train_dataset: ExtractorDataset, eval_dataset: Optional[ExtractorDataset]):
        """Validate training setup and raise informative errors for edge cases."""
        # Check if dataset is empty
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty. Please provide at least one training example.")
        
        # Check if dataset is smaller than batch size
        if len(train_dataset) < self.config.batch_size:
            logger.warning(
                f"Training dataset size ({len(train_dataset)}) is smaller than batch_size "
                f"({self.config.batch_size}). Adjusting batch_size to {len(train_dataset)}."
            )
            # We'll handle this in _create_dataloader by adjusting drop_last
        
        # Check early stopping configuration
        if self.config.early_stopping:
            if eval_dataset is None:
                raise ValueError(
                    "early_stopping is enabled but no eval_data provided. "
                    "Please provide eval_data or disable early_stopping."
                )
            if len(eval_dataset) == 0:
                raise ValueError("Evaluation dataset is empty but early_stopping is enabled.")
        
        # Check eval strategy configuration
        if self.config.eval_strategy == "steps" and eval_dataset is None:
            logger.warning(
                "eval_strategy='steps' but no eval_data provided. "
                "Evaluation will be skipped."
            )
        
        # Warn about very small datasets
        if len(train_dataset) < self.config.gradient_accumulation_steps:
            logger.warning(
                f"Training dataset size ({len(train_dataset)}) is smaller than "
                f"gradient_accumulation_steps ({self.config.gradient_accumulation_steps}). "
                f"Training may not work as expected."
            )
    
    def _flush_gradients(self) -> Optional[float]:
        """Flush accumulated gradients at the end of epoch if incomplete cycle exists."""
        # Check if there are accumulated gradients
        has_gradients = False
        for param in self.model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        if not has_gradients:
            return None
        
        # Apply the accumulated gradients
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1
        
        logger.info(f"Flushed incomplete gradient accumulation cycle at end of epoch (grad_norm: {grad_norm:.2f})")
        return grad_norm

    def _prepare_data(self, data: TrainDataInput, is_train: bool = True) -> ExtractorDataset:
        """Convert any supported data format to ExtractorDataset.

        Two behaviours are layered on top of the basic load:

        * **Sliding-window chunking** (when ``self.config.sliding_window`` is
          set) expands each record's ``input`` into overlapping subword-
          token windows. ``max_len`` is the window size and
          ``window_stride`` is the step, both in subword tokens. Each chunk
          inherits annotations whose surfaces appear inside it; the
          per-task filter rules live in
          :mod:`gliner2.training.chunking`.

        * **Deterministic shuffle** runs over every split (train, eval,
          and — via :func:`gliner2.training.metrics.evaluate_checkpoint` —
          test) so corpus-aggregation order can't bias the model. The
          training DataLoader still shuffles per-epoch on top.
        """
        if data is None:
            return None

        if isinstance(data, ExtractorDataset):
            return data

        max_samples = self.config.max_train_samples if is_train else self.config.max_eval_samples

        # Load records once via the factory so we can chunk and re-shuffle
        # before handing them to ExtractorDataset.
        records = DataLoader_Factory.load(
            data=data,
            max_samples=max_samples,
            shuffle=False,  # we shuffle ourselves after chunking
            seed=self.config.seed,
            validate=self.config.validate_data if is_train else False,
        )

        if self.config.sliding_window:
            from gliner2.training.chunking import chunk_records
            window_size = self.config.max_len or 512
            stride = self.config.window_stride or window_size
            split = "train" if is_train else "eval"
            n_in = len(records)
            logger.info(
                "[sliding-window] %s: chunking %d records "
                "(window_size=%d subwords, stride=%d)",
                split, n_in, int(window_size), int(stride),
            )
            records = chunk_records(
                records,
                tokenizer=self.processor.tokenizer,
                window_size=int(window_size),
                stride=int(stride),
                desc=f"chunking {split}",
                show_progress=self.is_main_process,
            )
            n_out = len(records)
            ratio = (n_out / n_in) if n_in else 0.0
            logger.info(
                "[sliding-window] %s: %d records -> %d chunks (%.2fx)",
                split, n_in, n_out, ratio,
            )

        rng = random.Random(self.config.seed)
        rng.shuffle(records)

        return ExtractorDataset(
            data=records,
            max_samples=-1,         # already applied above
            shuffle=False,          # already shuffled above
            seed=self.config.seed,
            validate=False,         # already validated above
        )

    def _create_optimizer(self) -> AdamW:
        """Create optimizer with appropriate parameters based on LoRA configuration."""
        if self.config.use_lora:
            lora_params = [p for p in self.model.parameters() if p.requires_grad]
            if not lora_params:
                raise ValueError("No LoRA parameters found. Check LoRA configuration.")
            logger.info("Optimizer: LoRA params only = %d, LR=%s", len(lora_params), self.config.task_lr)
            return AdamW(
                [{"params": lora_params, "lr": self.config.task_lr, "weight_decay": self.config.weight_decay}],
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
            )
        else:
            # Normal training: separate LRs for encoder and task-specific layers
            encoder_params = []
            task_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if "encoder" in name:
                    encoder_params.append(param)
                else:
                    task_params.append(param)

            return AdamW(
                [
                    {"params": encoder_params, "lr": self.config.encoder_lr, "weight_decay": self.config.weight_decay},
                    {"params": task_params, "lr": self.config.task_lr, "weight_decay": self.config.weight_decay},
                ],
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
            )

    def _create_dataloader(self, dataset: ExtractorDataset, batch_size: int, shuffle: bool = True, is_training: bool = True) -> DataLoader:
        sampler = None
        # Shard only the training data across ranks. Evaluation runs on rank 0
        # over the full set (see _evaluate), so eval loaders stay unsharded.
        if self.is_distributed and is_training:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False

        # When sliding_window is on, records are already chunked to fit
        # the encoder. Don't apply word-level truncation on top — that
        # would re-truncate by word count while the chunks are sized by
        # subword count.
        if self.config.sliding_window:
            max_len = None
        else:
            max_len = self.config.max_len or getattr(self.model.config, "max_len", None)
        collator = ExtractorCollator(self.processor, is_training=is_training, max_len=max_len)

        # Fix Bug #1 & #9: Handle small datasets
        # If dataset is smaller than batch_size, adjust to prevent empty dataloader
        effective_batch_size = min(batch_size, len(dataset))
        drop_last = is_training and len(dataset) > batch_size
        
        # Adjust num_workers for small datasets
        effective_num_workers = self.config.num_workers if len(dataset) > self.config.num_workers else 0

        # pin_memory is a no-op on MPS/CPU; only enable when targeting CUDA.
        pin_memory = self.config.pin_memory and self.device.type == "cuda"

        return DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=effective_num_workers,
            pin_memory=pin_memory,
            prefetch_factor=self.config.prefetch_factor if effective_num_workers > 0 else None,
            collate_fn=collator,
            drop_last=drop_last,
            persistent_workers=effective_num_workers > 0,
        )

    def train(
            self,
            train_data: TrainDataInput = None,
            eval_data: TrainDataInput = None,
    ) -> Dict[str, Any]:
        """
        Main training loop.

        Parameters
        ----------
        train_data : TrainDataInput, optional
            Training data. Supports all formats:
            - str/Path: JSONL file path
            - List[str/Path]: Multiple JSONL files
            - List[InputExample]: List of examples
            - TrainingDataset: Dataset object
            - List[Dict]: Raw records

        eval_data : TrainDataInput, optional
            Evaluation data (same formats supported).

        Returns
        -------
        Dict[str, Any]
            Training summary with metrics history.
        """
        # Prepare datasets
        train_data = train_data or self.train_data
        eval_data = eval_data or self.eval_data

        if train_data is None:
            raise ValueError("No training data provided")

        train_dataset = self._prepare_data(train_data, is_train=True)
        eval_dataset = self._prepare_data(eval_data, is_train=False) if eval_data else None

        # Fix Bug #7: Validate training setup
        self._validate_training_setup(train_dataset, eval_dataset)

        train_loader = self._create_dataloader(train_dataset, self.config.batch_size, shuffle=True, is_training=True)

        # Fix Bug #1: Check if dataloader is empty
        if len(train_loader) == 0:
            raise ValueError(
                f"Training dataloader is empty. Dataset size: {len(train_dataset)}, "
                f"Batch size: {self.config.batch_size}. Please reduce batch_size or add more data."
            )

        # Calculate steps
        num_update_steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        
        # Fix Bug #1: Handle case where num_update_steps_per_epoch is 0
        if num_update_steps_per_epoch == 0:
            # If gradient accumulation is larger than dataloader, we have at least the batches we can process
            num_update_steps_per_epoch = 1
            logger.warning(
                f"gradient_accumulation_steps ({self.config.gradient_accumulation_steps}) is larger than "
                f"batches per epoch ({len(train_loader)}). Setting to 1 update step per epoch."
            )
        
        if self.config.max_steps > 0:
            max_steps = self.config.max_steps
            num_epochs = math.ceil(max_steps / num_update_steps_per_epoch)
        else:
            max_steps = num_update_steps_per_epoch * self.config.num_epochs
            num_epochs = self.config.num_epochs

        warmup_steps = self.config.warmup_steps or int(max_steps * self.config.warmup_ratio)

        # Resume from a prior checkpoint if requested. Load weights BEFORE the
        # optimizer is created so it tracks the restored model's parameters.
        resume_state = None
        resume_dir = self._find_resume_checkpoint()
        if resume_dir is not None:
            logger.info("checkpoint_restart=%s: resuming from %s",
                        self.config.checkpoint_restart, resume_dir)
            self.load_checkpoint(str(resume_dir))
            resume_state = torch.load(
                resume_dir / "training_state.pt", map_location="cpu", weights_only=False,
            )

        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = get_scheduler(self.optimizer, self.config.scheduler_type, max_steps, warmup_steps, self.config.num_cycles)

        start_epoch = 0
        if resume_state is not None:
            start_epoch = self._restore_training_state(resume_state)
            logger.info(
                "Resumed: continuing at epoch %d/%d (global_step %d, best_metric %.4f)",
                start_epoch + 1, num_epochs, self.global_step, self.best_metric,
            )

        # Mixed precision (CUDA-only; on MPS/CPU these were already disabled in
        # _setup_device, so use_amp is False).
        use_amp = self.config.fp16 or self.config.bf16
        amp_dtype = torch.bfloat16 if self.config.bf16 else torch.float16
        if self.device.type == "cuda" and self.config.fp16:
            self.scaler = GradScaler("cuda", enabled=True)
        else:
            self.scaler = None

        # Logging
        logger.info("***** Running Training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num epochs = {num_epochs}")
        logger.info(f"  Batch size = {self.config.batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size = {self.config.effective_batch_size}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(f"  Warmup steps = {warmup_steps}")
        
        # Log trainable parameters. Uses the same manual-count expression in
        # both branches; the LoRA branch just labels the output differently.
        # (Previously called ``count_lora_parameters`` from ``gliner2.training.lora``, 
        # which was dropped from the import list during the PEFT migration but
        # left behind here, producing a NameError at the start of every LoRA run.)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        percentage = (trainable_params / total_params * 100) if total_params > 0 else 0.0
        if self.config.use_lora:
            logger.info(f"  LoRA enabled: {trainable_params:,} trainable / {total_params:,} total ({percentage:.2f}%)")
        else:
            logger.info(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({percentage:.2f}%)")

        # Training state
        self.model.train()
        self.processor.change_mode(is_training=True)
        self.global_step = 0
        self.epoch = 0
        tr_loss = 0.0

        start_time = time.time()
        samples_seen = 0
        # Stashed so _evaluate can report remaining ETA from measured progress.
        self._train_start_time = start_time
        self._train_max_steps = max_steps

        self.progress_bar = tqdm(total=max_steps, desc="Training", disable=not self.is_main_process)

        should_stop = False
        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            # Release the previous epoch's cached blocks at the boundary so the
            # MPS/CUDA high-water mark does not climb across epochs.
            self._free_memory()

            if self.is_distributed:
                train_loader.sampler.set_epoch(epoch)

            epoch_loss = 0.0
            epoch_steps = 0

            for step, batch in enumerate(train_loader):
                samples_seen += len(batch)

                try:
                    with autocast(device_type=self.device.type, enabled=use_amp, dtype=amp_dtype):
                        outputs = self._fwd_model(batch)
                        loss = outputs["total_loss"]

                        if self.config.gradient_accumulation_steps > 1:
                            loss = loss / self.config.gradient_accumulation_steps

                    # Skip batches where loss doesn't require grad (edge cases in data)
                    if not loss.requires_grad:
                        logger.warning(
                            f"Skipping batch {step}: loss doesn't require grad "
                            f"(loss={loss.item():.4f}). This may indicate edge cases in your data."
                        )
                        continue

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    tr_loss += loss.item()
                    epoch_loss += loss.item()
                    epoch_steps += 1

                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    logger.warning(
                        f"OOM at step {step}, batch skipped. "
                        f"Consider reducing batch_size or max sequence length."
                    )
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    elif self.device.type == "mps":
                        torch.mps.empty_cache()
                    gc.collect()
                    self.optimizer.zero_grad()
                    continue

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)

                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % self.config.logging_steps == 0:
                        elapsed = time.time() - start_time
                        # Fix Bug #2: Safe division for metrics
                        avg_loss = self._safe_divide(tr_loss, self.config.logging_steps, default=tr_loss)
                        # Fix Bug #5: Safe division for epoch progress
                        epoch_progress = self._safe_divide(step, len(train_loader), default=0.0)
                        metrics = TrainingMetrics(
                            loss=avg_loss,
                            classification_loss=outputs.get("classification_loss", torch.tensor(0)).item(),
                            structure_loss=outputs.get("structure_loss", torch.tensor(0)).item(),
                            count_loss=outputs.get("count_loss", torch.tensor(0)).item(),
                            learning_rate=self.scheduler.get_last_lr()[0],
                            epoch=epoch + epoch_progress,
                            step=self.global_step,
                            samples_seen=samples_seen,
                            throughput=self._safe_divide(samples_seen, elapsed, default=0.0),
                        )
                        self._log_metrics(metrics, prefix="train")
                        tr_loss = 0.0

                    if self.config.eval_strategy == "steps" and self.global_step % self.config.eval_steps == 0:
                        # Under DDP, eval/early-stop run on rank 0 only; the
                        # decision is then broadcast so all ranks stop together.
                        if eval_dataset and self.is_main_process:
                            prev_best = self.best_metric
                            eval_metrics = self._evaluate(eval_dataset)
                            self.model.train()
                            self.processor.change_mode(is_training=True)
                            if self.config.early_stopping and self._check_early_stopping(eval_metrics, prev_best):
                                logger.info(f"Early stopping triggered at step {self.global_step}")
                                should_stop = True
                        self._save_checkpoint(f"checkpoint-{self.global_step}")
                        should_stop = self._sync_flag(should_stop)
                        self._barrier()
                        if should_stop:
                            break

                    self.progress_bar.update(1)

                    if self.global_step >= max_steps:
                        break
            
            if should_stop:
                break

            # Fix Bug #6: Flush incomplete gradient accumulation at end of epoch
            if epoch_steps % self.config.gradient_accumulation_steps != 0:
                grad_norm = self._flush_gradients()
                if grad_norm is not None:
                    logger.info(f"Applied incomplete gradient accumulation at end of epoch {epoch + 1}")

            # Fix Bug #3: Safe division for epoch loss
            avg_epoch_loss = self._safe_divide(epoch_loss, epoch_steps, default=0.0)
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_epoch_loss:.4f}")

            if self.config.eval_strategy == "epoch":
                epoch_stop = False
                if eval_dataset and self.is_main_process:
                    prev_best = self.best_metric
                    eval_metrics = self._evaluate(eval_dataset)
                    self.model.train()
                    self.processor.change_mode(is_training=True)
                    if self.config.early_stopping and self._check_early_stopping(eval_metrics, prev_best):
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        epoch_stop = True
                self._save_checkpoint(f"checkpoint-epoch-{epoch + 1}")
                epoch_stop = self._sync_flag(epoch_stop)
                self._barrier()
                if epoch_stop:
                    break

            if self.global_step >= max_steps:
                break

        self.progress_bar.close()
        self.progress_bar = None

        self._barrier()  # all ranks finish training before rank 0 writes the final checkpoint
        if self.is_main_process:
            self._save_checkpoint("final")
            if self.config.report_to_wandb:
                import wandb
                wandb.summary["best_metric"] = self.best_metric
                wandb.summary["total_steps"] = self.global_step
                wandb.finish()

        if self.is_distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

        total_time = time.time() - start_time
        return {
            "total_steps": self.global_step,
            "total_epochs": self.epoch + 1,
            "total_time_seconds": total_time,
            "samples_per_second": samples_seen / total_time,
            "best_metric": self.best_metric,
            "train_metrics_history": self.train_metrics_history,
            "eval_metrics_history": self.eval_metrics_history,
        }

    def _log_remaining_eta(self) -> None:
        """Log estimated remaining training time, projected from progress so far.

        Uses the measured step rate since training started (steps done / elapsed)
        rather than the warmup-based startup estimate, so it sharpens over time.
        """
        start = getattr(self, "_train_start_time", None)
        max_steps = getattr(self, "_train_max_steps", 0)
        if start is None or self.global_step <= 0 or max_steps <= 0:
            return
        elapsed = time.time() - start
        remaining_steps = max(0, max_steps - self.global_step)
        eta = remaining_steps * elapsed / self.global_step
        logger.info(
            f"Training ETA: ~{_fmt_seconds(eta)} remaining "
            f"(step {self.global_step}/{max_steps}, elapsed {_fmt_seconds(elapsed)})"
        )

    def _free_memory(self) -> None:
        """Release cached allocator memory so the high-water mark does not grow
        across epochs. The MPS/CUDA caching allocators retain freed blocks; on
        MPS (unified memory) an unbounded high-water mark ends in an uncatchable
        OS OOM kill, typically first hit during the memory-heavy eval pass.
        Cheap and safe to call between phases (it frees only unused blocks)."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

    def _evaluate(self, eval_dataset: ExtractorDataset) -> Dict[str, float]:
        logger.info("Running evaluation...")
        self._log_remaining_eta()
        self.model.eval()
        self.processor.change_mode(is_training=False)
        # Free the training step's cached blocks so eval has headroom.
        self._free_memory()

        eval_loader = self._create_dataloader(eval_dataset, self.config.eval_batch_size, shuffle=False, is_training=False)

        # Fix Bug #4: Check if eval dataloader is empty
        if len(eval_loader) == 0:
            logger.warning(
                f"Evaluation dataloader is empty. Dataset size: {len(eval_dataset)}, "
                f"Batch size: {self.config.eval_batch_size}. Skipping evaluation."
            )
            return {
                "eval_loss": 0.0,
                "eval_classification_loss": 0.0,
                "eval_structure_loss": 0.0,
                "eval_count_loss": 0.0,
                "step": self.global_step,
                "epoch": self.epoch,
            }

        total_loss = 0.0
        total_cls_loss = 0.0
        total_struct_loss = 0.0
        total_count_loss = 0.0
        num_batches = 0

        use_amp = self.config.fp16 or self.config.bf16
        amp_dtype = torch.bfloat16 if self.config.bf16 else torch.float16

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", disable=not self.is_main_process):
                with autocast(device_type=self.device.type, enabled=use_amp, dtype=amp_dtype):
                    # Raw model, not the DDP/DataParallel wrapper: eval runs on
                    # rank 0 only, so the wrapper's collectives would deadlock.
                    outputs = self.model(batch)

                # Fix Bug #10: Move tensors to CPU to prevent memory leak
                total_loss += outputs["total_loss"].detach().cpu().item()
                total_cls_loss += outputs.get("classification_loss", torch.tensor(0)).detach().cpu().item()
                total_struct_loss += outputs.get("structure_loss", torch.tensor(0)).detach().cpu().item()
                total_count_loss += outputs.get("count_loss", torch.tensor(0)).detach().cpu().item()
                num_batches += 1

        # Fix Bug #4: Safe division for evaluation metrics
        metrics = {
            "eval_loss": self._safe_divide(total_loss, num_batches, default=0.0),
            "eval_classification_loss": self._safe_divide(total_cls_loss, num_batches, default=0.0),
            "eval_structure_loss": self._safe_divide(total_struct_loss, num_batches, default=0.0),
            "eval_count_loss": self._safe_divide(total_count_loss, num_batches, default=0.0),
            "step": self.global_step,
            "epoch": self.epoch,
        }

        # The eval-loss pass is done; release its activations before the second
        # forward pass that compute_metrics runs (batch_extract over the val set).
        self._free_memory()
        if self.compute_metrics:
            metrics.update(self.compute_metrics(self.model, eval_dataset))

        self._log_metrics(metrics, prefix="eval")
        self.eval_metrics_history.append(metrics)

        metric_value = metrics.get(self.config.metric_for_best, metrics["eval_loss"])
        is_best = (
            (self.config.greater_is_better and metric_value > self.best_metric) or
            (not self.config.greater_is_better and metric_value < self.best_metric)
        )

        if is_best:
            self.best_metric = metric_value
            if self.config.save_best:
                self._save_checkpoint("best")
            if self.is_main_process:
                self._write_eval_metrics(metrics)
            logger.info(f"New best {self.config.metric_for_best}: {self.best_metric:.4f}")

        # Release eval activations before training resumes.
        self._free_memory()
        return metrics

    def _check_early_stopping(self, metrics: Dict[str, float], prev_best: Optional[float] = None) -> bool:
        metric_value = metrics.get(self.config.metric_for_best, metrics["eval_loss"])
        compare_against = prev_best if prev_best is not None else self.best_metric
        if self.config.greater_is_better:
            improved = metric_value > compare_against + self.config.early_stopping_threshold
        else:
            improved = metric_value < compare_against - self.config.early_stopping_threshold

        if improved:
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.config.early_stopping_patience

    def _log_metrics(self, metrics: Union[Dict, TrainingMetrics], prefix: str = ""):
        """Log metrics with safe handling of edge cases."""
        if isinstance(metrics, TrainingMetrics):
            metrics = metrics.to_dict()
        
        # Handle empty metrics gracefully
        if not metrics:
            logger.warning("Attempted to log empty metrics")
            return

        # Update progress bar with key metrics
        if self.is_main_process and self.progress_bar is not None:
            postfix = {}
            for key, value in metrics.items():
                if key in ["loss", "learning_rate", "throughput"]:
                    if isinstance(value, float):
                        if math.isnan(value):
                            postfix[key] = "NaN"
                        elif math.isinf(value):
                            postfix[key] = "Inf"
                        elif key == "learning_rate":
                            postfix["lr"] = f"{value:.2e}"
                        elif key == "throughput":
                            postfix["samples/s"] = f"{value:.1f}"
                        else:
                            postfix[key] = f"{value:.4f}"
            
            # Add epoch info if available
            if "epoch" in metrics:
                postfix["epoch"] = f"{metrics['epoch']:.1f}"
            
            if postfix:
                self.progress_bar.set_postfix(postfix)

        # W&B logging
        if self.config.report_to_wandb and self.is_main_process:
            try:
                import wandb
                # Filter out NaN and Inf values for wandb
                wandb_metrics = {
                    k: v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float)) and not (math.isnan(v) or math.isinf(v))
                }
                if wandb_metrics:
                    wandb.log(wandb_metrics, step=self.global_step)
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")

        if prefix == "train":
            self.train_metrics_history.append(metrics)

    def _write_eval_metrics(self, metrics: Dict) -> None:
        """Persist the best checkpoint's eval metrics as JSON.

        Writes ``eval_metrics.json`` to ``output_dir`` (always reflecting the
        current best) and, when the ``best/`` checkpoint folder exists, into it
        too — so the metrics travel with the saved model. Caller guards on
        ``is_main_process``.
        """
        payload = json.dumps(metrics, indent=2, default=str)
        (self.output_dir / "eval_metrics.json").write_text(payload, encoding="utf-8")
        best_dir = self.output_dir / "best"
        if best_dir.is_dir():
            (best_dir / "eval_metrics.json").write_text(payload, encoding="utf-8")

    def _save_training_state(self, checkpoint_dir: Path) -> None:
        """Save optimizer/scheduler/epoch state for mid-run resume.

        Written only into numbered ``checkpoint-*`` dirs (not ``best``/``final``,
        which stay model-only and publishable). For Adam this roughly doubles a
        checkpoint's size (two moments per parameter).
        """
        if self.optimizer is None:
            return
        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "patience_counter": self.patience_counter,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(state, checkpoint_dir / "training_state.pt")

    def _find_resume_checkpoint(self) -> Optional[Path]:
        """Pick a resumable checkpoint per ``config.checkpoint_restart``.

        Considers numbered ``checkpoint-*`` dirs carrying a ``training_state.pt``
        (``best``/``final`` are excluded). ``highest`` = largest trailing number,
        ``last`` = newest by mtime. Returns None (with a warning) when none exist.
        """
        mode = getattr(self.config, "checkpoint_restart", None)
        if not mode:
            return None
        candidates = [
            d for d in self.output_dir.glob("checkpoint-*")
            if d.is_dir() and (d / "training_state.pt").is_file()
        ]
        if not candidates:
            logger.warning(
                "checkpoint_restart=%r but no resumable checkpoint (with "
                "training_state.pt) found in %s; starting a fresh run.",
                mode, self.output_dir,
            )
            return None
        if mode == "last":
            return max(candidates, key=lambda d: d.stat().st_mtime)
        if mode == "highest":
            def _num(d: Path) -> int:
                m = re.search(r"(\d+)$", d.name)
                return int(m.group(1)) if m else -1
            return max(candidates, key=_num)
        logger.warning(
            "Unknown checkpoint_restart=%r (use 'highest' or 'last'); starting fresh.", mode
        )
        return None

    def _restore_training_state(self, state: dict) -> int:
        """Restore optimizer/scheduler/counters/RNG; return the epoch to resume at."""
        self.optimizer.load_state_dict(state["optimizer"])
        # Optimizer tensors were loaded on CPU; move them to the training device.
        for opt_state in self.optimizer.state.values():
            for k, v in opt_state.items():
                if isinstance(v, torch.Tensor):
                    opt_state[k] = v.to(self.device)
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = int(state.get("global_step", 0))
        self.best_metric = state.get("best_metric", self.best_metric)
        self.patience_counter = int(state.get("patience_counter", 0))
        if state.get("torch_rng") is not None:
            torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return int(state.get("epoch", -1)) + 1

    def _save_checkpoint(self, name: str):
        if not self.is_main_process:
            return

        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(exist_ok=True)

        save_start = time.time()

        if self.config.use_lora and self.config.save_adapter_only:
            # PEFT-native format only: ``self.model`` is a ``PeftModel``, so
            # ``save_pretrained`` writes ``adapter_config.json`` (with
            # ``peft_type: "LORA"``) + ``adapter_model.safetensors``, which is
            # what ``PeftModel.from_pretrained`` and every downstream PEFT
            # consumer expect. The pre-migration code also invoked
            # ``gliner2.training.lora.save_lora_adapter`` here to emit the
            # legacy ``adapter_weights.safetensors`` + gliner2 ``LoRAAdapterConfig``
            # alongside the PEFT files, but ``LoRAAdapterConfig.save`` writes
            # to the same ``adapter_config.json`` path and **clobbers** the
            # PEFT config (strips ``peft_type``), so any PEFT reader then blew
            # up with ``KeyError: 'peft_type'`` at ``PeftConfig._get_peft_type``.
            # Legacy callers that still need the gliner2-native directory
            # shape can invoke ``save_lora_adapter`` directly on a
            # checkpoint dir after training — the shim is preserved with
            # ``PendingDeprecationWarning`` in ``gliner2/training/lora.py``.
            self.model.save_pretrained(str(checkpoint_dir))
            checkpoint_type = "adapter"
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            if self.config.use_lora and isinstance(self.model, PeftModel):
                self.model.merge_adapter()
                self.model.get_base_model().save_pretrained(str(checkpoint_dir))
                self.model.unmerge_adapter()
            else:
                self.model.save_pretrained(str(checkpoint_dir))
            checkpoint_type = "full"
            trainable_params = sum(p.numel() for p in self.model.parameters())

        # Numbered checkpoints carry resume state; best/final stay model-only.
        if name.startswith("checkpoint-"):
            self._save_training_state(checkpoint_dir)

        save_time = time.time() - save_start
        checkpoint_size_mb = sum(f.stat().st_size for f in checkpoint_dir.rglob('*') if f.is_file()) / (1024 * 1024)
        
        # World-class logging
        logger.info(
            f"💾 Saved {checkpoint_type} checkpoint '{name}' | "
            f"step {self.global_step} | epoch {self.epoch + 1:.1f} | "
            f"{trainable_params:,} params | {checkpoint_size_mb:.1f}MB | {save_time:.1f}s"
        )

        # Save model artifacts to W&B for best and final checkpoints
        if self.config.report_to_wandb and name in ["best", "final"]:
            try:
                import wandb
                artifact = wandb.Artifact(
                    name=f"model-{self.config.experiment_name}-{name}",
                    type="model",
                    metadata={
                        "step": self.global_step,
                        "epoch": self.epoch,
                        "checkpoint_type": checkpoint_type,
                        "params": trainable_params,
                        "size_mb": checkpoint_size_mb,
                    }
                )
                artifact.add_dir(str(checkpoint_dir))
                wandb.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"W&B artifact upload failed: {e}")

        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        if self.config.save_total_limit <= 0:
            return

        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: x.stat().st_mtime,
        )
        protected = {"best", "final"}
        checkpoints = [c for c in checkpoints if c.name not in protected]

        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            shutil.rmtree(oldest)
            logger.info(f"Removed old checkpoint: {oldest.name}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from a checkpoint (adapter-only or full)."""
        checkpoint_dir = Path(checkpoint_path)
        is_adapter = (checkpoint_dir / "adapter_config.json").exists()

        if is_adapter:
            logger.info("Loading LoRA adapter from %s", checkpoint_path)
            base = self.model.get_base_model() if isinstance(self.model, PeftModel) else self.model
            self.model = PeftModel.from_pretrained(base, str(checkpoint_dir))
            self.model.to(self.device)
            self.lora_layers = {n: m for n, m in self.model.named_modules() if isinstance(m, _PeftLoraLayer)}
            self.model._lora_layers = self.lora_layers
        else:
            self.model = self.model.__class__.from_pretrained(str(checkpoint_dir))
            self.model.to(self.device)
            if self.config.use_lora:
                logger.info("Applying LoRA to loaded model...")
                self.lora_layers = {}
                self._setup_lora()

        # self.model was reassigned; rebuild the forward wrapper to track it and
        # re-apply gradient checkpointing (a runtime flag from_pretrained drops),
        # else a resume silently loses the activation-memory fix and re-OOMs.
        self._setup_parallel()
        self._setup_gradient_checkpointing()
        logger.info("Loaded checkpoint: %s", checkpoint_path)


# =============================================================================
# Convenience Functions
# =============================================================================

def train_gliner2(
        model_path: str,
        train_data: TrainDataInput,
        output_dir: str = "./output",
        eval_data: TrainDataInput = None,
        **config_kwargs,
) -> Dict[str, Any]:
    """
    Convenience function for training GLiNER2.

    Parameters
    ----------
    model_path : str
        Path to pretrained model.
    train_data : TrainDataInput
        Training data in any supported format:
        - JSONL path(s)
        - List of InputExample
        - TrainingDataset
        - List of dicts
    output_dir : str
        Output directory for checkpoints.
    eval_data : TrainDataInput, optional
        Evaluation data.
    **config_kwargs
        Additional TrainingConfig parameters.

    Returns
    -------
    Dict[str, Any]
        Training results.

    Examples
    --------
    >>> # Train with JSONL file
    >>> results = train_gliner2("model-path", "train.jsonl", num_epochs=10)

    >>> # Train with multiple JSONL files
    >>> results = train_gliner2("model-path", ["train1.jsonl", "train2.jsonl"])

    >>> # Train with InputExample list
    >>> examples = [InputExample(...), ...]
    >>> results = train_gliner2("model-path", examples)

    >>> # Train with TrainingDataset
    >>> dataset = TrainingDataset.load("train.jsonl")
    >>> results = train_gliner2("model-path", dataset)
    """
    from gliner2 import GLiNER2

    model = GLiNER2.from_pretrained(model_path)
    config = TrainingConfig(output_dir=output_dir, **config_kwargs)

    trainer = GLiNER2Trainer(model=model, config=config)
    return trainer.train(train_data=train_data, eval_data=eval_data)