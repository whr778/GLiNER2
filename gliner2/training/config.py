"""Torch-free training configuration shared by planning and the native engine."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Union


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
    default_schema_mode : str
        How ``config.json`` ``default_schema`` is built during training:
        ``"dataset"`` (target dataset schema excluding replay corpora and ungraded
        partial-annotation dimensions), ``"train"`` (union over all training records),
        or ``"union_base"`` (union of the loaded base model's ``default_schema`` with
        the training records).
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
    # False (upstream's rule): every parameter whose NAME contains "encoder" gets encoder_lr -- which also catches
    # the HEAD's boundary_head.boundary_encoder and boundary_head.candidate_encoder (660,736 randomly initialised
    # params in eb22c; they moved 0.085/0.095 relative over 3.5k-30k steps vs 0.94-6.0 for the rest of the head,
    # 2026-10-22). True: only the BACKBONE (`encoder.*`) gets encoder_lr; the head encoders get task_lr.
    encoder_lr_backbone_only: bool = False
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    # 1.0 = one full cosine that anneals to ZERO. This was 0.5 until 2026-09-22, and 0.5 is
    # a trap: `cosine_restarts` computes `cos(pi * ((num_cycles * progress) % 1.0))`, so with
    # num_cycles=0.5 the argument only reaches 0.5, NEVER WRAPS, and the schedule degenerates
    # to a half-cosine that performs no restarts and ends at 50% of base LR. Every one of the
    # 158 configs in this repo names `cosine_restarts` and none sets `num_cycles`, so every
    # model this project has trained -- eb16, the 137k curve, every A/B -- finished its last
    # epoch taking half-size steps instead of annealing. Measured LR multipliers:
    #     num_cycles=0.5   p=0.50 -> 0.854   p=1.00 -> 0.500   (no restart, no anneal)
    #     num_cycles=1.0   p=0.50 -> 0.500   p=1.00 -> 0.000   (identical to `cosine`)
    num_cycles: float = 1.0
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_total_limit: int = 3
    save_best: bool = True
    # Also write checkpoint-epoch-N (with training_state.pt) at each epoch end when eval_strategy is "steps". Resume
    # always restarts at the NEXT epoch (no mid-epoch batch skip), so only an epoch-end checkpoint resumes without
    # silently skipping the rest of an epoch's data.
    save_epoch_checkpoint: bool = False
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
    seed: int = 42
    deterministic: bool = False
    local_rank: int = -1
    debug: bool = False
    max_train_samples: int = -1
    max_eval_samples: int = -1
    max_samples_seed: int = 42   # seed of the RANDOM draw behind the two caps (sliding-window path)
    validate_data: bool = True
    max_len: Optional[int] = None
    # GIST. `guide_scores` is a JSONL cache from tools/train/precompute_guide_scores.py;
    # setting it does two things at once, and both are needed for either to matter:
    # each training record's cached rivals are injected into its schema, and the frozen
    # guide's scores are handed to the veto. Train only -- eval must not be perturbed.
    guide_scores: Optional[str] = None
    rivals_per_record: int = 3
    # LABEL NEGATIVES. `negative_pools` is the JSON from tools/data/build_negative_pools.py;
    # `negative_labels_per_dim` is how many ABSENT labels to offer per dimension, e.g.
    # {"entities": 2, "events": 1}. Empty = OFF, so every existing config reproduces. Train
    # only: eval must not be perturbed, and the full-menu eval is a separate mode.
    negative_pools: Optional[str] = None
    negative_labels_per_dim: Optional[dict] = None
    negative_label_seed: int = 42
    # MENU_SPEC s4 refinements, all OFF by default. `negative_gold_free_dims`: absents also for a
    # dimension the record has no gold in, from its OWN corpus pool. `negative_k_sampling`:
    # "fixed" | "loguniform" (K per record drawn from 1..negative_labels_per_dim). `negative_max_per_record`:
    # a whole-record cap on injected labels (the window budget).
    negative_gold_free_dims: bool = False
    # Role-list symmetry: gold event types also list their FULL pool role list (unargued roles get empty
    # targets), as injected absent types always have. Off = historical; see NegativeLabels.gold_full_roles.
    negative_gold_full_roles: bool = False
    negative_k_sampling: str = "fixed"
    negative_max_per_record: Optional[int] = None
    # Per-epoch dose, {dim: [epoch 0, epoch 1, ...]} (the last value holds): an annealing-style ramp such as
    # [1, 5, 20], or [1, 1, 20] for a final rejection phase. Overrides negative_labels_per_dim for the dims it
    # names; negative_labels_per_dim must still be set (it switches negatives on).
    negative_labels_schedule: Optional[dict] = None
    # SCHEMA DROPOUT. The processor's SamplingConfig randomly REMOVES gold from a training
    # schema -- remove_events_prob and remove_relations_prob default to 0.2,
    # remove_classification_label_prob to 0.5 -- and was previously unreachable from config
    # (SamplingConfig is imported into this module and never used). Sensible regularisation
    # at 174k records and 5 epochs; at A/B scale it withholds a fifth of the signal from the
    # very heads under test, and it has produced three false readings in tests by removing
    # the only gold item in a small batch. `schema_dropout: false` zeroes every *_prob; a dict
    # ({remove_relations_prob: 0.0, ...}) sets only the named probabilities and leaves the rest
    # at their defaults (2026-10-23: the relation drop is per gold INSTANCE and flips labels to
    # negatives, the event drop is per TYPE; they deserve separate decisions).
    schema_dropout: Union[bool, Dict[str, float]] = True

    # Restored feature knobs (pre-boundary; consumed by tools/train/train.py).
    # checkpoint_restart: resume selection ('last' | 'highest' | None).
    checkpoint_restart: Optional[str] = None
    # sliding_window/window_stride: window long training docs (word-level) so a
    # short-context encoder (e.g. DeBERTa-v3 512-cap) doesn't overflow; each window
    # carries its own gold. False -> single pass with max_len truncation.
    # Entity labels whose gold is entirely unalignable are offered with nothing to
    # find, so they train as ABSENT while the entity is in the text -- measured at
    # 1.41% of label queries. On, the query is dropped instead. Training only.
    suppress_orphaned_queries: bool = False
    # `{corpus: [dimension, ...]}` -- gold that is NOT exhaustive for that dimension.
    # Such a corpus is refused as a SOURCE of negatives by build_negative_pools.py and,
    # from here, as a TARGET of them: absent-from-gold is not absent-from-text.
    partial_annotation: Optional[Dict[str, list]] = None
    sliding_window: bool = False
    window_stride: int = 256
    # data_parallel: DEPRECATED no-op. The trainer now uses DistributedDataParallel
    # (torchrun / local_rank); the old single-process nn.DataParallel path is gone.
    data_parallel: bool = False

    # Strict training invariants (boundary architecture; harmless for span).
    # When strict_training is True: processor/model exceptions propagate, a
    # non-finite loss raises (unless ignore_nonfinite_losses is enabled), and
    # batch-size discrepancies raise. ``skip_step_errors`` is an explicit
    # single-process recovery override for faulty micro-batches.
    strict_training: bool = True
    ignore_nonfinite_losses: bool = False
    skip_step_errors: bool = False
    allow_invalid_samples: bool = False
    log_proposal_metrics: bool = True
    gold_injection_start: float = 1.0
    gold_injection_end: float = 0.25
    gold_injection_hold_frac: float = 0.15
    diagnostics_every_n_steps: int = 0
    profile_first_n_steps: int = 0
    # Dump complete CPU-side preprocessed batches before these global steps.
    debug_global_steps: List[int] = field(default_factory=list)
    ddp_consensus_check: bool = True
    ddp_find_unused_parameters: bool = False
    # False: DDP(static_graph=True) never averaged GLiNER2's gradients -- its backward sink is attached through
    # pytree, which could not see into the ExtractorOutput dataclass -- so 2-GPU ranks drifted silently (25 of 33
    # module groups, 2x H100, 2026-10-20); and once the sink does fire it cannot accumulate under no_sync(). Plain
    # DDP plus _ddp_full_touch averages every step (tools/lambda/ddp_trace.py, 26 of 26 sync checks identical).
    ddp_static_graph: bool = False
    # >0: compare per-module GRADIENT (pre-clip) and WEIGHT checksums across ranks on the first N optimizer
    # steps -- localizes where 2-GPU ranks diverge (ddp_sync_probe, 2026-10-20). 0 = off.
    ddp_sync_probe_steps: int = 0
    dry_run_recall_steps: int = 0
    gate_recall: float = 0.97
    gate_long_recall: float = 0.93
    # Gold-capacity overflow policy for boundary targets: "raise" (default,
    # no silent loss), "truncate_with_warning", or "skip_sample".
    on_capacity_exceeded: str = "raise"
    # Malformed-record policy for boundary targets: "raise" (default, no silent
    # loss), "skip" (drop the record and COUNT it), or "fallback". Needed because
    # boundary word-alignment cannot always map a verbatim-present surface to word
    # token boundaries -- notably for languages without whitespace delimiters --
    # and "raise" then aborts a whole run over ~10% of records.
    error_policy: str = "raise"
    # Drop an entity mention whose surface is not a whole word, and keep the
    # record. The span is not widened onto the enclosing word. Malformed records
    # still follow error_policy. Pair with suppress_orphaned_queries so a type
    # whose every surface missed is not trained as absent.
    ignore_missing_entities: bool = False


    # Filter an entity query's independently re-searched listed-value matches
    # down to the maximal, non-contained spans before they become gold
    # mentions (see boundary_preprocessing._dedupe_contained_spans). Defaults
    # to the corrected behavior; set False to restore the pre-fix behavior.
    dedupe_contained_entity_spans: bool = True
    # "all": a text-level relation labels every head-occurrence x tail-occurrence pair; "nearest":
    # only the closest pair (see boundary_preprocessing._pair_gap).
    relation_gold_pairs: str = "all"
    # Expand gold event arguments with coreferent mentions from coref_clusters (argument_coref.py).
    expand_argument_coref: bool = True
    group_by_length: bool = True
    length_group_window_batches: int = 50
    # "charclass" (default) approximates tokens from character classes and works for
    # scripts without whitespace. "words" is the pre-2026-09-05 len(text.split()),
    # kept ONLY so the proxy can be A/B'd against the models trained under it --
    # it reads every Chinese document as ~1 word. Do not select it for new work.
    length_proxy: str = "charclass"
    compile_model: bool = False
    gradient_checkpointing: bool = False
    fused_optimizer: bool = True
    allow_tf32: bool = True
    float32_matmul_precision: str = "high"

    # LoRA Configuration (Parameter-Efficient Fine-Tuning)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_use_dora: bool = False
    lora_target_modules: List[str] = field(default_factory=lambda: ["encoder", "span_rep", "classifier", "count_embed", "count_pred"])
    save_adapter_only: bool = True  # Only applies when use_lora=True
    default_schema_mode: str = "dataset"

    def __post_init__(self):
        self._precision_explicit = self.fp16 is not None or self.bf16 is not None
        if self.fp16 is None:
            self.fp16 = True
        if self.bf16 is None:
            self.bf16 = False
        if self.fp16 and self.bf16:
            raise ValueError("Cannot use both fp16 and bf16")
        if not 0.0 <= self.gold_injection_start <= 1.0:
            raise ValueError("gold_injection_start must be in [0, 1]")
        if not 0.0 <= self.gold_injection_end <= 1.0:
            raise ValueError("gold_injection_end must be in [0, 1]")
        if not 0.0 <= self.gold_injection_hold_frac <= 1.0:
            raise ValueError("gold_injection_hold_frac must be in [0, 1]")
        if self.diagnostics_every_n_steps < 0:
            raise ValueError("diagnostics_every_n_steps must be >= 0")
        if self.profile_first_n_steps < 0:
            raise ValueError("profile_first_n_steps must be >= 0")
        if self.dry_run_recall_steps < 0:
            raise ValueError("dry_run_recall_steps must be >= 0")
        if not 0.0 <= self.gate_recall <= 1.0:
            raise ValueError("gate_recall must be in [0, 1]")
        if not 0.0 <= self.gate_long_recall <= 1.0:
            raise ValueError("gate_long_recall must be in [0, 1]")
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

        if self.on_capacity_exceeded not in ("raise", "truncate_with_warning", "skip_sample"):
            raise ValueError(
                "on_capacity_exceeded must be 'raise', 'truncate_with_warning', or "
                f"'skip_sample', got {self.on_capacity_exceeded!r}"
            )
        if self.length_group_window_batches <= 0:
            raise ValueError("length_group_window_batches must be > 0")
        if self.float32_matmul_precision not in ("highest", "high", "medium"):
            raise ValueError(
                "float32_matmul_precision must be 'highest', 'high', or 'medium'"
            )
        if self.default_schema_mode not in ("dataset", "train", "union_base"):
            raise ValueError(
                "default_schema_mode must be 'dataset', 'train', or 'union_base', "
                f"got {self.default_schema_mode!r}"
            )
        
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
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        with open(path) as f:
            return cls(**json.load(f))
