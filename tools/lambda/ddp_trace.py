"""TRACE where 2-GPU DDP stops averaging gradients: the real eb22c model and trainer, one block at a time.

ddp_sync_probe arm E (2026-10-20) found every group that receives a real gradient DIFFERS across ranks right after
backward (step 0, pre-clip), encoder included; only unused groups (zero grads) matched. This walks the path:
  T0  weights right after the DDP wrap           (did DDP broadcast rank 0's weights?)
  T1  DDP state: static_graph, sync flag, registered / ignored parameters
  T2  grads after a RAW ddp(batch) + loss.backward()   (DDP alone, no trainer code)
  T3  grads after trainer._backward_one(is_last_micro=True)   (the trainer's path)
Each rank draws a DIFFERENT batch (DistributedSampler), so an absent average cannot hide. Rank 0 prints.

    torchrun --standalone --nproc_per_node=2 tools/lambda/ddp_trace.py tools/train/config/base/eb22c.yaml
"""
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import train as T  # noqa: E402
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig  # noqa: E402


def say(msg: str) -> None:
    if dist.get_rank() == 0:
        print(f"[trace] {msg}", flush=True)


def compare(label: str, model, kind: str) -> None:
    """Per-group checksum (sum, sum of squares) gathered from both ranks; names the groups that differ."""
    groups = {}
    for name, p in model.named_parameters():
        t = p.grad if kind == "grads" else p
        if t is None:
            continue
        g = ".".join(name.split(".")[:2])
        v = t.detach().float()
        s = torch.stack([v.sum(), (v * v).sum()])
        groups[g] = groups[g] + s if g in groups else s
    names = sorted(groups)
    local = torch.stack([groups[n] for n in names])
    out = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(out, local)
    other = out[1] if len(out) > 1 else out[0]   # world 1 (a local dry run) compares a rank with itself
    diff = [n for n, a, b in zip(names, out[0], other) if not torch.equal(a, b)]
    say(f"{label}: {kind} -- {len(names) - len(diff)} of {len(names)} groups IDENTICAL; differ: {diff[:12]}")


def main() -> None:
    cfg_path = sys.argv[1]
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")   # the trainer reuses an existing group
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    cfg["training"].update(output_dir="./out/ddp-trace", batch_size=2, gradient_accumulation_steps=1,
                           negative_pools=None, negative_labels_per_dim=None, max_train_samples=64,
                           eval_strategy="no", save_best=False, local_rank=int(os.environ["LOCAL_RANK"]))
    # ONE axis per run: DDP_STATIC=0|1, DDP_BUCKET_VIEW=0|1 (default = the trainer's own settings)
    if "DDP_STATIC" in os.environ:
        cfg["training"]["ddp_static_graph"] = os.environ["DDP_STATIC"] == "1"
    if "TRACE_LOSS" in os.environ:   # ONE head change: boundary_marginal_loss (bce | asymmetric_focal)
        cfg["model"]["boundary_head"]["boundary_marginal_loss"] = os.environ["TRACE_LOSS"]
    if "TRACE_GC" in os.environ:
        cfg["training"]["gradient_checkpointing"] = os.environ["TRACE_GC"] == "1"
    if "DDP_BUCKET_VIEW" in os.environ:
        import torch.nn.parallel as P
        import gliner2.training.trainer as TR
        view = os.environ["DDP_BUCKET_VIEW"] == "1"
        real = P.DistributedDataParallel
        TR.DistributedDataParallel = lambda *a, **k: real(*a, **{**k, "gradient_as_bucket_view": view})
    cuda = torch.cuda.is_available()
    if not cuda:   # a local dry run: same path, fp32
        cfg["training"].update(bf16=False, fp16=False)
    config = TrainingConfig(**cfg["training"])
    torch.manual_seed(0)
    model = T._build_model(cfg["model"])
    fns, passthrough = T._category_fns(T.load_labels_cfg(cfg, cfg_path)), T.labels_passthrough(cfg)
    path = "data/cmnee_ner.train.jsonl"
    T._fetch_if_missing(path)
    records = T.read_transformed([path], fns, passthrough)[:64]
    # CONTROL: a plain Linear under DDP, different input per rank. Its grads MUST come back identical, or this
    # comparison (or the environment) cannot tell an averaged gradient from an unaveraged one.
    ctl = torch.nn.Linear(4, 3)
    ctl_ddp = torch.nn.parallel.DistributedDataParallel(ctl)
    ctl_ddp(torch.full((2, 4), float(dist.get_rank() + 1))).sum().backward()
    compare("CONTROL Linear under DDP", ctl, "grads")
    trainer = ExtractorTrainer(model, config)
    if os.environ.get("TRACE_NOHOOKS") == "1":   # remove the trainer's finite-grad tensor hooks
        for h in trainer._finite_grad_hook_handles:
            h.remove()
        say(f"removed {len(trainer._finite_grad_hook_handles)} finite-grad hooks")
    ddp = trainer.model
    say(f"wrapped: {type(ddp).__name__} | world {dist.get_world_size()} | device {trainer.device}")

    compare("T0 after the DDP wrap", ddp.module, "weights")
    trainable = [p for p in ddp.module.parameters() if p.requires_grad]
    say(f"T1 static_graph={getattr(ddp, 'static_graph', '?')} require_backward_grad_sync={ddp.require_backward_grad_sync} "
        f"find_unused={ddp.find_unused_parameters} | trainable tensors {len(trainable)} | "
        f"ignored {len(getattr(ddp, 'parameters_to_ignore', []))} | gradient_as_bucket_view {ddp.gradient_as_bucket_view}")

    ds = trainer._prepare_data(records, True)
    batch = next(iter(trainer._create_dataloader(ds, 2, shuffle=True, is_training=True))).to(trainer.device)
    ddp.train()

    only = os.environ.get("TRACE_ONLY", "")   # T3: skip T2, whose incomplete reduction leaves DDP mid-iteration
    ddp.zero_grad(set_to_none=True)
    if only != "T3":
     with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
        loss = ddp(batch).get("total_loss")
     loss.backward()
     compare("T2 raw ddp(batch) + backward", ddp.module, "grads")
     none = [n for n, q in ddp.module.named_parameters() if q.requires_grad and q.grad is None]
     say(f"T2 trainable tensors with NO gradient: {len(none)} -- {none[:8]}")

    ddp.zero_grad(set_to_none=True)
    trainer._backward_one(batch, 0, cuda, torch.bfloat16, is_last_micro=True)
    compare("T3 trainer._backward_one", ddp.module, "grads")
    none = [n for n, q in ddp.module.named_parameters() if q.requires_grad and q.grad is None]
    say(f"T3 trainable tensors with NO gradient: {len(none)} -- {none[:8]}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
