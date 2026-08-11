"""GLiNER2 training utilities (architecture-neutral)."""

from gliner2.training.trainer import (
    ExtractorTrainer,
    GLiNER2Trainer,
    TrainingConfig,
    TrainingMetrics,
    ExtractorDataset,
    ExtractorCollator,
    train_gliner2,
)
from gliner2.training.eta import estimate_eta
from gliner2.training.guide_scores import GuideScores
from gliner2.training.stopwords import build_stopwords
from gliner2.training.metrics import (
    compute_metrics,
    evaluate_checkpoint,
    make_compute_metrics,
    make_sweeping_compute_metrics,
    sweep_thresholds,
    sweep_global_decode,
)

__all__ = [
    "ExtractorTrainer",
    "GLiNER2Trainer",
    "TrainingConfig",
    "TrainingMetrics",
    "ExtractorDataset",
    "ExtractorCollator",
    "train_gliner2",
    "estimate_eta",
    "GuideScores",
    "build_stopwords",
    "compute_metrics",
    "evaluate_checkpoint",
    "make_compute_metrics",
    "make_sweeping_compute_metrics",
    "sweep_thresholds",
    "sweep_global_decode",
]
