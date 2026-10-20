"""Building a trainer disables the cuDNN SDPA backend.

cuDNN SDPA compiles a kernel through NVRTC for every new sequence length; on an H100 one boundary-attention
forward+backward cost 1.2-1.5 s per new length vs 0.001 s without it, and eb22c trained at 2.9 samples/s with the
GPU 2% busy (2026-10-20).
"""
import tempfile

import torch

from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model


def test_trainer_turns_cudnn_sdpa_off_and_says_so(caplog):
    torch.backends.cuda.enable_cudnn_sdp(True)
    try:
        tc = TrainingConfig(output_dir=tempfile.mkdtemp(), bf16=False, fp16=False, validate_data=False)
        with caplog.at_level("INFO"):
            ExtractorTrainer(build_tiny_boundary_model(), tc)
        assert torch.backends.cuda.cudnn_sdp_enabled() is False
        assert any("[sdpa] cuDNN SDPA backend disabled" in r.getMessage() for r in caplog.records)
    finally:
        torch.backends.cuda.enable_cudnn_sdp(True)
