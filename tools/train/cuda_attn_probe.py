"""Pre-probe: what attention path actually loads on CUDA, and does it produce NaN.

Two questions, both cheap, both of which must be answered before spending GPU
hours on training runs:

1. **Which implementation actually loads.** ``base.py`` falls back to ``eager``
   when an implementation is rejected, so a config asking for FlashAttention 2
   can silently train on the SLOWEST path. Requesting an implementation is not
   evidence of getting it -- this prints what the encoder ended up with.

2. **Do fully-masked sliding-window rows produce NaN on CUDA in bf16.** mmBERT is
   ModernBERT: 2 of every 3 layers use a +/-64 sliding window, so a short document
   padded against a long one in the same batch has query rows attending to nothing.
   transformers' guard for that is gated ``not _is_torch_greater_or_equal_than_2_5``
   and so is NOT applied on torch 2.11. This does not reproduce on CPU, and
   ``torch.autocast("cpu", bfloat16)`` is a no-op for this model, so CUDA bf16 is
   the only place the question can actually be asked.

Run on the GPU box::

    .venv/bin/python tools/train/cuda_attn_probe.py
"""
import torch
from transformers import AutoConfig, AutoModel

ENCODER = "jhu-clsp/mmBERT-base"
CANDIDATES = ["sdpa", "eager", "flash_attention_2", "kernels-community/flash-attn"]


def load(impl):
    """Load the encoder exactly as gliner2 does: fp32 weights, reduced precision
    applied at runtime by autocast."""
    return AutoModel.from_pretrained(
        ENCODER, trust_remote_code=True, dtype=torch.float32, attn_implementation=impl
    )


def padded_batch(device, long_len=4096, short_len=150):
    """One full-length window plus one median-length document -- the real batch
    shape at max_len 4096, where medians are RAMS 141 / Re-DocRED 209 subwords."""
    ids = torch.randint(5, 1000, (2, long_len), device=device)
    mask = torch.zeros(2, long_len, dtype=torch.long, device=device)
    mask[0, :long_len] = 1
    mask[1, :short_len] = 1
    ids[1, short_len:] = 0
    return ids, mask, short_len


def main() -> None:
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} "
          f"| {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}")
    cfg = AutoConfig.from_pretrained(ENCODER, trust_remote_code=True)
    print(f"encoder: {cfg.model_type} local_attention={getattr(cfg, 'local_attention', None)} "
          f"global_every={getattr(cfg, 'global_attn_every_n_layers', None)} "
          f"layers={getattr(cfg, 'num_hidden_layers', None)}\n")

    print("=== 1. which implementation actually loads ===")
    working = []
    for impl in CANDIDATES:
        try:
            model = load(impl)
            actual = model.config._attn_implementation
            ok = "OK " if actual == impl else "MISMATCH"
            print(f"  requested {impl:32} -> got {actual:32} {ok}")
            if actual == impl:
                working.append(impl)
            del model
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"  requested {impl:32} -> FAILED: {type(exc).__name__}: {str(exc)[:110]}")

    if not torch.cuda.is_available():
        return

    print("\n=== 2. fully-masked sliding-window rows on CUDA, bf16 ===")
    for impl in working:
        model = load(impl).cuda().eval()
        ids, mask, short = padded_batch("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(input_ids=ids, attention_mask=mask).last_hidden_state
        real = h[1, :short]                      # REAL tokens of the short document
        print(f"  {impl:32} dtype={str(h.dtype):16} "
              f"NaN_anywhere={str(h.isnan().any().item()):5} "
              f"NaN_in_real_tokens={str(real.isnan().any().item()):5} "
              f"nan_count={h.isnan().sum().item()}")
        del model, h
        torch.cuda.empty_cache()

    print("\n=== 3. same, but every sequence the same length (no padding) ===")
    for impl in working:
        model = load(impl).cuda().eval()
        ids = torch.randint(5, 1000, (2, 512), device="cuda")
        mask = torch.ones(2, 512, dtype=torch.long, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(input_ids=ids, attention_mask=mask).last_hidden_state
        print(f"  {impl:32} NaN={h.isnan().any().item()}")
        del model, h
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
