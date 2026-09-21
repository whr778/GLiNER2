"""Measure how well the BASE encoder knows each corpus's (language, domain) cell.

Masked pseudo-perplexity of the base encoder on raw corpus text. This is a prior on how far
to trust the model's own ranking: a cell the base never learned gives a flat, unreliable
ranking, and a structural prior (the typed role-constraint margin) should carry more weight
there. The point of measuring the ENCODER rather than the task head is that pseudo-perplexity
is independent of candidate count, while candidate entropy is not -- a long document with 300
spans has high entropy for structural reasons that say nothing about familiarity.

LANGUAGE AND DOMAIN ARE NOT SEPARABLE. Measured on this base: two Chinese corpora differ by
1.31x and two ENGLISH corpora by 1.68x, so neither factor alone predicts the scale. A base can
know Chinese medical and not English medical; only the CELL can express that.

MASK, OR THE NUMBER IS MEANINGLESS. Scoring with ``labels=input_ids`` and no masking lets a
masked LM see the token it is predicting: measured NLL 0.001 on English clinical text and
0.213 on pure gibberish, i.e. it measures copy fidelity, not familiarity. With real masking
the same two read 1.84 and 7.47 (pseudo-perplexity 6.3 against 1748).

    uv run python tools/train/measure_corpus_familiarity.py \
        --corpora data/bc5cdr:en:biomedical data/cmnee_typed:zh:news --docs 30
"""

import argparse
import json
import math
import statistics as st
from pathlib import Path

import torch


def _pll(model, tok, text, draws, frac, max_length, gen):
    """Mean NLL over MASKED positions only -- the real pseudo-log-likelihood."""
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    ids = enc["input_ids"]
    special = torch.tensor(
        tok.get_special_tokens_mask(ids[0].tolist(), already_has_special_tokens=True)
    ).bool()
    cand = (~special).nonzero().flatten()
    if len(cand) < 8:
        return None
    total, n = 0.0, 0
    for _ in range(draws):
        k = max(1, int(frac * len(cand)))
        sel = cand[torch.randperm(len(cand), generator=gen)[:k]]
        x, lab = ids.clone(), torch.full_like(ids, -100)
        lab[0, sel] = ids[0, sel]
        x[0, sel] = tok.mask_token_id
        with torch.no_grad():
            out = model(input_ids=x, attention_mask=enc["attention_mask"], labels=lab)
        total += out.loss.item() * k
        n += k
    return total / n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--encoder", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--corpora", nargs="+", required=True,
                    help="prefix[:lang[:domain]], e.g. data/bc5cdr:en:biomedical")
    ap.add_argument("--docs", type=int, default=30)
    ap.add_argument("--draws", type=int, default=4, help="mask draws per document")
    ap.add_argument("--frac", type=float, default=0.15, help="fraction of tokens masked")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out")
    args = ap.parse_args()

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.encoder)
    model = AutoModelForMaskedLM.from_pretrained(args.encoder)
    model.eval()

    report, gen = {}, torch.Generator().manual_seed(0)
    for spec in args.corpora:
        parts = spec.split(":")
        prefix, lang, domain = parts[0], (parts[1] if len(parts) > 1 else "?"), \
            (parts[2] if len(parts) > 2 else "?")
        path = Path(f"{prefix}.{args.split}.jsonl")
        if not path.is_file():
            print(f"[fam] {prefix}: no {args.split} split; skipping")
            continue
        vals = []
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if len(vals) >= args.docs:
                    break
                text = (json.loads(line).get("input") or "").strip()
                if len(text) < 40:
                    continue
                v = _pll(model, tok, text, args.draws, args.frac, args.max_length, gen)
                if v is not None:
                    vals.append(v)
        if len(vals) < 5:
            print(f"[fam] {prefix}: too few usable docs ({len(vals)})")
            continue
        nll = st.median(vals)
        report[Path(prefix).name] = {
            "lang": lang, "domain": domain, "n": len(vals),
            "nll": nll, "pseudo_ppl": math.exp(nll),
        }
        print(f"[fam] {Path(prefix).name:20s} {lang:3s} {domain:12s} n={len(vals):3d} "
              f"NLL={nll:5.3f}  pseudo_ppl={math.exp(nll):8.1f}")

    print(f"\n{'corpus':20s} {'lang':5s} {'domain':13s} {'NLL':>6} {'pseudo_ppl':>11}")
    for c, d in sorted(report.items(), key=lambda kv: kv[1]["pseudo_ppl"]):
        print(f"{c:20s} {d['lang']:5s} {d['domain']:13s} {d['nll']:>6.3f} "
              f"{d['pseudo_ppl']:>11.1f}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"\n[fam] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
