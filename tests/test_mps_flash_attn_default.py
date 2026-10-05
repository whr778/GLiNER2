"""The mps-flash-attn SDPA patch is OPT-IN: by default stock SDPA is kept.

With the patch, padded documents decode differently when batched (3 of 5 CASIE docs,
2026-10-05); stock SDPA on MPS is batch-invariant. A fresh process, because the patch is
applied once per process and replaces a global.
"""
import importlib.util
import os
import subprocess
import sys

import pytest

PROBE = ("import torch.nn.functional as F; from gliner2.models.base import _enable_mps_flash_attention; "
         "_enable_mps_flash_attention(); print(F.scaled_dot_product_attention.__module__)")


def _sdpa_module(env_value):
    env = {k: v for k, v in os.environ.items() if k != "GLINER2_MPS_FLASH_ATTN"}
    if env_value is not None:
        env["GLINER2_MPS_FLASH_ATTN"] = env_value
    out = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def test_default_keeps_stock_sdpa():
    assert _sdpa_module(None) != "mps_flash_attn"


@pytest.mark.skipif(importlib.util.find_spec("mps_flash_attn") is None, reason="mps-flash-attn is Mac-only")
def test_opt_in_applies_the_patch():
    assert _sdpa_module("1") == "mps_flash_attn"
