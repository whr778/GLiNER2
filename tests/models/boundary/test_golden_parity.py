"""Deliberately updateable seeded forward-output parity contract."""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


GOLDEN = Path(__file__).parent / "_golden" / "boundary_head.safetensors"


def _forward(batch):
    with torch.no_grad():
        output = batch["head"](
            batch["token_states"],
            batch["text_mask"],
            batch["query_states"],
            batch["query_mask"],
            batch["targets"],
        )
    tensors = {
        "start_logits": output.start_logits.contiguous(),
        "end_logits": output.end_logits.contiguous(),
        "inside_logits": output.inside_logits.contiguous(),
        "candidates.indices": output.candidates.indices.contiguous(),
        "candidates.valid_mask": output.candidates.valid_mask.contiguous(),
        "candidates.pair_logits": output.candidates.pair_logits.contiguous(),
    }
    tensors.update(
        {
            f"losses.{name}": value.detach().contiguous()
            for name, value in output.losses.items()
            if name not in {
                "soft_iou_loss",
                "rerank_listwise_loss",
                "count_loss",
            }
        }
    )
    return tensors


def test_seeded_boundary_golden_parity(golden_batch, request):
    actual = _forward(golden_batch)
    if request.config.getoption("--update-golden"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        save_file(actual, str(GOLDEN))
    assert GOLDEN.exists(), "run pytest with --update-golden deliberately"
    expected = load_file(str(GOLDEN))
    assert actual.keys() == expected.keys()
    for name in actual:
        # Small tolerance rather than exact equality: float32 matmul reductions are
        # not bit-reproducible across hardware/torch builds (~5e-7 drift), so atol=0
        # fails everywhere but the exact capture machine. 1e-4 still catches any real
        # forward change (logit shifts are O(0.1+)).
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-4, atol=1e-4)
