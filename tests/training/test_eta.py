"""Tests for gliner2.training.eta.estimate_eta."""

import json
import random

import torch

from gliner2.models.boundary import model as b_mod
from gliner2.training.config import TrainingConfig
from gliner2.training.eta import _warmup_records, estimate_eta
from gliner2.training.trainer import GLiNER2Trainer
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model


def _make_records(n: int = 8):
    return [
        {
            "input": "apple released iphone in california today .",
            "output": {
                "entities": {"company": ["apple"], "product": ["iphone"]},
            },
        }
        for _ in range(n)
    ]


def test_estimate_eta_scales_with_epochs_includes_eval_and_restores_gates(tmp_path, capsys):
    pool_path = tmp_path / "pools.json"
    pool_path.write_text(
        json.dumps({
            "pools": {
                "tiny": {
                    "entities": ["company", "product"] + [f"neg_ent_{i}" for i in range(10)],
                }
            }
        }),
        encoding="utf-8",
    )
    train_recs = _make_records(8)
    val_recs = _make_records(4)

    model = build_tiny_boundary_model()
    cfg = TrainingConfig(
        output_dir=str(tmp_path / "out"),
        batch_size=2,
        gradient_accumulation_steps=2,
        num_epochs=3,
        eval_strategy="epoch",
        negative_pools=str(pool_path),
        negative_labels_per_dim={"entities": 4},
        fp16=False,
        bf16=False,
        num_workers=0,
        local_rank=-1,
    )
    trainer = GLiNER2Trainer(model, cfg, eval_data=val_recs)
    neg = trainer._negative_labels()

    calls_before = b_mod._NEGATIVE_QUERY_CALLS
    absent_before = b_mod._ABSENT_CELL_CALLS
    random.seed(123)
    torch.manual_seed(123)
    py_rng_before = random.getstate()
    th_rng_before = torch.get_rng_state()

    est = estimate_eta(
        model,
        train_recs,
        cfg,
        n_warmup=2,
        trainer=trainer,
        eval_data=val_recs,
        eval_batch_size=2,
        eval_decode_passes=5,
        post_eval_passes=3.0,
    )
    printed = capsys.readouterr().out

    assert est is not None
    # 8 examples, batch_size=2 -> 4 micro-batches/epoch, grad_accum=2 -> 2 opt steps/epoch * 3 epochs = 6
    assert est["total_opt_steps"] == 6.0
    assert est["train_seconds"] > 0.0
    assert est["eval_seconds"] > 0.0
    assert est["eta_seconds"] == est["train_seconds"] + est["eval_seconds"]

    # Header has no emoji and displays both train and eval/sweep rows.
    assert "Training Time Estimate" in printed
    assert "\U0001f4ca" not in printed
    assert "Train steps ETA" in printed
    assert "Eval / sweep ETA" in printed
    assert "(6 in-loop" in printed or "(15 in-loop" in printed

    # In-run gate counters, NegativeLabels cache, and RNGs are untouched.
    assert b_mod._NEGATIVE_QUERY_CALLS == calls_before
    assert b_mod._ABSENT_CELL_CALLS == absent_before
    assert neg.stats["records"] == 0
    assert neg._epoch_records.value == 0
    assert random.getstate() == py_rng_before
    assert torch.equal(torch.get_rng_state(), th_rng_before)


def test_warmup_records_strides_across_multiple_files(tmp_path):
    f1 = tmp_path / "a.jsonl"
    f2 = tmp_path / "b.jsonl"
    f1.write_text(
        "\n".join(json.dumps({"input": f"a_{i}", "output": {}}) for i in range(10)) + "\n",
        encoding="utf-8",
    )
    f2.write_text(
        "\n".join(json.dumps({"input": f"b_{i}", "output": {}}) for i in range(10)) + "\n",
        encoding="utf-8",
    )
    sampled = _warmup_records([f1, f2], n=4)
    inputs = [r["input"] for r in sampled]
    assert len(inputs) == 4
    assert any(s.startswith("a_") for s in inputs)
    assert any(s.startswith("b_") for s in inputs)
