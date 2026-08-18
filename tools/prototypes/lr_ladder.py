"""Stage 0: does fixed partial gating survive a learning rate that destabilises GeGLU?

This is the cheap, high-information half of the crammed-BERT plan in
`CRAMMING_EXPERIMENT.md`. It needs no published baseline and no GLUE finetuning,
because the comparison is internal: three FFN variants, identical everything else,
learning rate escalated until something breaks.

**Why this and not a quality comparison.** The toy study (`PARTIAL_GATING.md`) found
hybrid4:fixed TIED with plain GELU on fit -- a quality win is not the hypothesis. What it
did measure is a lower gradient ceiling (13.4 vs plain GELU 17.2 vs GeGLU 23.2), and
section 4 showed the multiplicative blowup is scale-dependent. So the falsifiable claim
is about STABILITY, and the way to test it is to push the learning rate until the
unbounded path bites. If all three arms behave identically here, the ceiling difference
does not matter at depth and the expensive stages are not worth running.

Divergence is recorded three ways, because they fail differently: a non-finite loss
(hard failure), a loss that climbs well above its own starting point (slow blowup), and
the max gradient norm reached (the quantity the hypothesis is actually about).

Runs on CPU, CUDA or MPS -- see `resolve_device`. Nothing here is CUDA-specific.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ffn_variants import build_ffn


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick the best available device, or honour an explicit request.

    Mirrors `gliner2.models.base.resolve_device` rather than importing it, so this
    directory stays standalone. CUDA first, then MPS (Apple GPU), then CPU.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer with a swappable FFN."""

    def __init__(self, kind: str, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = build_ffn(kind, d_model, d_ff)

    def forward(self, x, pad_mask):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


class Encoder(nn.Module):
    """Small MLM encoder. Weight-tied output head to keep the parameter count honest."""

    def __init__(self, kind: str, vocab: int, d_model: int = 256, n_layers: int = 6,
                 n_heads: int = 4, d_ff: int = 1024, max_len: int = 128):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(kind, d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, ids, pad_mask):
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.tok(ids) + self.pos(pos)[None]
        for layer in self.layers:
            x = layer(x, pad_mask)
        return self.head(self.norm(x))


# Namespaced repo id. The bare "wikitext" name no longer resolves -- current
# huggingface_hub requires 'namespace/name' and raises HfUriError otherwise.
WIKITEXT_REPO = "Salesforce/wikitext"


def load_batches(tokenizer, n_tokens: int, seq_len: int, repo: str = WIKITEXT_REPO):
    """Tokenise wikitext-2 into a flat id tensor, trimmed to whole sequences."""
    from datasets import load_dataset

    ds = load_dataset(repo, "wikitext-2-raw-v1", split="train")
    ids: list[int] = []
    for row in ds:
        text = row["text"].strip()
        if text:
            ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(ids) >= n_tokens:
            break
    usable = (len(ids) // seq_len) * seq_len
    return torch.tensor(ids[:usable], dtype=torch.long).view(-1, seq_len)


def mask_tokens(batch, mask_id: int, vocab: int, prob: float, generator):
    """Standard 80/10/10 MLM corruption. Returns (inputs, labels) with -100 elsewhere."""
    labels = batch.clone()
    selected = torch.rand(batch.shape, generator=generator) < prob
    labels[~selected] = -100
    inputs = batch.clone()
    roll = torch.rand(batch.shape, generator=generator)
    inputs[selected & (roll < 0.8)] = mask_id
    randomize = selected & (roll >= 0.8) & (roll < 0.9)
    inputs[randomize] = torch.randint(vocab, (int(randomize.sum()),), generator=generator)
    return inputs, labels


def run_arm(kind, lr, data, tokenizer, device, steps, batch_size, seed, mlm_prob,
            arch, warmup_frac=0.06):
    """Train one (variant, lr) pair and report how it failed, if it did.

    `warmup_frac` is a real LINEAR LR WARMUP, not a reporting window. Without it every
    arm takes full-magnitude updates from a random init, which is the regime warmup
    exists to avoid -- and crammed-BERT, the baseline this feeds into, uses warmup. A
    no-warmup ladder measures which variant best survives an init shock nobody trains
    through, which is not the hypothesis.

    Distinct from `measure_from` below, which excludes early steps from the reported
    gradient maximum and never gates an update. Backprop runs on every step.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    vocab = len(tokenizer)
    model = Encoder(kind, vocab, **arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    warmup_steps = max(1, int(steps * warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup_steps))

    # Reporting window only: the step-0 gradient is an initialisation artifact and is
    # LR-INDEPENDENT, so including it made the reported maximum identical across
    # different learning rates and measured nothing.
    measure_from = min(50, steps // 10)
    losses, grad_max, grad_max_late, diverged_at = [], 0.0, 0.0, None
    running_min = float("inf")
    for step in range(steps):
        idx = torch.randint(data.shape[0], (batch_size,), generator=gen)
        inputs, labels = mask_tokens(data[idx], tokenizer.mask_token_id, vocab,
                                     mlm_prob, gen)
        inputs, labels = inputs.to(device), labels.to(device)
        pad = torch.zeros(inputs.shape, dtype=torch.bool, device=device)

        logits = model(inputs, pad)
        loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        # Unclipped: the norm IS the measurement. 1e9 makes clip_grad_norm_ a no-op.
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
        opt.step()
        sched.step()

        value = loss.item()
        losses.append(value)
        if math.isfinite(gnorm):
            grad_max = max(grad_max, gnorm)
            if step >= measure_from:
                grad_max_late = max(grad_max_late, gnorm)
        # Divergence for MLM is a RISE OFF THE RUNNING MINIMUM, not a multiple of the
        # first loss. The initial loss is already ~ln(vocab), the ceiling, and loss falls
        # from there -- a "2x first loss" test can essentially never fire, so it reported
        # every arm stable regardless of what happened.
        if math.isfinite(value):
            running_min = min(running_min, value)
        if diverged_at is None and (not math.isfinite(value)
                                    or (step > measure_from and value > running_min + 1.0)):
            diverged_at = step

    tail = [v for v in losses[-50:] if math.isfinite(v)]
    final = sum(tail) / len(tail) if tail else float("nan")
    random_loss = math.log(vocab)
    return {
        "variant": kind,
        "lr": lr,
        "params": sum(p.numel() for p in model.parameters()),
        "final_loss": final,
        "min_loss": min((v for v in losses if math.isfinite(v)), default=float("nan")),
        "random_baseline": random_loss,
        # A run that never beats ln(vocab) by a clear margin has not trained, and a
        # stability comparison over untrained models is meaningless. Surfaced as a
        # first-class flag so it cannot be mistaken for "stable".
        "learned": math.isfinite(final) and final < random_loss - 2.0,
        "grad_max": grad_max,
        "grad_max_after_warmup": grad_max_late,
        "warmup_steps": warmup_steps,
        "diverged_at": diverged_at,
        "nonfinite": not all(math.isfinite(v) for v in losses),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", action="append", default=None,
                    help="Repeatable: gelu | geglu | hybrid4. Default: all three.")
    ap.add_argument("--lr", type=float, action="append", default=None,
                    help="Repeatable. Default ladder: 3e-4 1e-3 3e-3 1e-2.")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=1024,
                    help="Plain-GELU inner width; each variant is matched to it.")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--mlm-prob", type=float, default=0.15)
    ap.add_argument("--warmup-frac", type=float, default=0.06,
                    help="Linear LR warmup as a fraction of steps. 0 disables it, which "
                         "is NOT a realistic schedule -- crammed-BERT uses warmup.")
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--tokenizer", default="bert-base-uncased")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--out", type=Path, default=Path("lr_ladder_results.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = resolve_device(args.device)
    variants = args.variant or ["gelu", "geglu", "hybrid4"]
    ladder = args.lr or [3e-4, 1e-3, 3e-3, 1e-2]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    data = load_batches(tokenizer, args.tokens, args.seq_len)
    print(f"[ladder] device={device} data={tuple(data.shape)} "
          f"variants={variants} lrs={ladder} steps={args.steps} seeds={args.seeds}\n"
          f"[ladder] arch d_model={args.d_model} layers={args.n_layers} "
          f"heads={args.n_heads} d_ff={args.d_ff} seq={args.seq_len}", flush=True)

    arch = dict(d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                d_ff=args.d_ff, max_len=args.seq_len)

    results = []
    for seed in range(args.seeds):
        for kind in variants:
            for lr in ladder:
                r = run_arm(kind, lr, data, tokenizer, device, args.steps,
                            args.batch_size, seed, args.mlm_prob, arch,
                            args.warmup_frac)
                r["seed"] = seed
                results.append(r)
                if r["diverged_at"] is not None:
                    status = f"DIVERGED@{r['diverged_at']}"
                elif not r["learned"]:
                    status = "NOT-LEARNED"
                else:
                    status = "ok"
                print(f"  {kind:8s} lr={lr:<8.1e} seed={seed} loss={r['final_loss']:8.4f} "
                      f"grad_late={r['grad_max_after_warmup']:9.1f} {status}", flush=True)

    args.out.write_text(json.dumps(
        {"device": str(device), "steps": args.steps, "batch_size": args.batch_size,
         "seq_len": args.seq_len, "arch": arch, "results": results},
        indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\n[ladder] wrote {args.out}")

    print(f"\nfinal loss; DIV = diverged, FLAT = never beat ln(vocab)-2 so nothing "
          f"was learned")
    print(f"{'variant':10s} " + " ".join(f"{lr:>11.0e}" for lr in ladder))
    for kind in variants:
        cells = []
        for lr in ladder:
            rows = [r for r in results if r["variant"] == kind and r["lr"] == lr]
            n_div = sum(1 for r in rows if r["diverged_at"] is not None)
            n_flat = sum(1 for r in rows if not r["learned"])
            if n_div:
                cell = f"DIV {n_div}/{len(rows)}"
            elif n_flat:
                cell = f"FLAT {n_flat}/{len(rows)}"
            else:
                cell = f"{sum(r['final_loss'] for r in rows) / len(rows):.3f}"
            cells.append(f"{cell:>11s}")
        print(f"{kind:10s} " + " ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
