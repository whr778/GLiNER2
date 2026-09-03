"""Per-language pseudo-perplexity for an encoder, before committing supervised data.

Supervised data does not rescue a language the base cannot model. If English sits at 2.2
and Lingala at 75, the Lingala fix is unsupervised continued pretraining, not more labels
-- and forcing supervised training on a poorly-modelled language distorts the embedding
space for the languages that WERE modelled well.

This reports masked-LM pseudo-perplexity per language on held-out text from our own
corpora, so the number describes the text we will actually train on rather than a
benchmark. Numbers are comparable ACROSS LANGUAGES for one encoder; comparing across
encoders is only meaningful when they share a tokenizer, because perplexity is per-token
and tokenizers differ in how many tokens a language costs. `tokens/char` is reported
beside it for exactly that reason: a language can look cheap per token while needing three
times the tokens.

    uv run python tools/train/base_model_perplexity.py --encoder jhu-clsp/mmBERT-base
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

# Held-out text per language, drawn from corpora already on disk.
SOURCES = {
    "en": ("data/cas_ann_en.jsonl", "input"),
    "tr": ("data/turkish_gate/cas_ann_tr.jsonl", "input"),
    "zh": ("data/chinese_gate/cas_ann_zh.jsonl", "input"),
    "ko": ("data/klue_ner.train.jsonl", "input"),
    "ja": ("data/stockmark_jpn.train.jsonl", "input"),
    "kk": ("data/kaznerd.train.jsonl", "input"),
    "afr": ("data/masakhaner.train.jsonl", "input"),
}

# The SAME question on a second axis. A language table is not enough: an encoder trained on
# book-and-web text can carry a language well and still not model the register it will be
# deployed on. Held-out ENGLISH text, varied by domain, so language is held fixed.
DOMAINS = {
    "news (real)":        ("data/cas_ann_en.jsonl", "input"),
    "news (synthetic)":   ("data/casualty_natural.train.jsonl", "input"),
    "news (CC-News)":     ("data/cc_news_haiku45.train.jsonl", "input"),
    "biomedical":         ("data/biored.train.jsonl", "input"),
    "financial filings":  ("data/chfinann.train.jsonl", "input"),
    "scientific":         ("data/scierc.train.jsonl", "input"),
    "encyclopedic":       ("data/redocred.train.jsonl", "input"),
    "synthetic registers": ("data/synthetic_haiku45_5k.train.jsonl", "input"),
    # Short-form, unedited. Every domain above is edited prose, which does not test the
    # register question at all -- a model can carry a language and still not model tweets.
    # 400 messages over 60 characters from the train split of the public HF dataset
    # `community-datasets/disaster_response_messages`. Committed rather than fetched so the
    # published 29.52 is reproducible from this repo; it lived only on a temp volume until
    # 2026-09-03, which would have made that number unrepeatable.
    "short-form messages": ("data/short_form_probe.jsonl", "input"),
}


def load_texts(path, field, n, seed=42):
    rows = []
    p = Path(path)
    if not p.is_file():
        return rows
    for line in p.open(encoding="utf-8"):
        t = (json.loads(line).get(field) or "").strip()
        if len(t) > 60:
            rows.append(t)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def pseudo_perplexity(model, tok, texts, device, max_len=256, mask_prob=0.15, seed=42):
    """Mean MLM cross-entropy over randomly masked positions, exponentiated."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    total, count = 0.0, 0
    for text in texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
        ids = enc["input_ids"]
        special = torch.tensor(tok.get_special_tokens_mask(ids[0].tolist(),
                                                           already_has_special_tokens=True),
                               dtype=torch.bool).unsqueeze(0)
        pick = (torch.rand(ids.shape, generator=g) < mask_prob) & ~special
        if not pick.any():
            continue
        labels = ids.clone()
        labels[~pick] = -100
        masked = ids.clone()
        masked[pick] = tok.mask_token_id
        with torch.no_grad():
            out = model(input_ids=masked.to(device),
                        attention_mask=enc["attention_mask"].to(device),
                        labels=labels.to(device))
        total += float(out.loss) * int(pick.sum())
        count += int(pick.sum())
    return math.exp(total / count) if count else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--encoder", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", help="write the table as JSON")
    ap.add_argument("--domains", action="store_true",
                    help="score DOMAINS at fixed language instead of languages")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.encoder)
    model = AutoModelForMaskedLM.from_pretrained(args.encoder).to(args.device).eval()

    print(f"  encoder: {args.encoder}   samples/language: {args.samples}\n")
    axis = DOMAINS if args.domains else SOURCES
    label = "domain" if args.domains else "lang"
    print(f"  {label:20s} {'docs':>5s} {'pseudo-PPL':>11s} {'tokens/char':>12s}  reading")
    table = {}
    for lang, (path, field) in axis.items():
        texts = load_texts(path, field, args.samples)
        if not texts:
            print(f"  {lang:20s} {'-':>5s} {'no text':>11s}")
            continue
        ppl = pseudo_perplexity(model, tok, texts, args.device)
        tpc = sum(len(tok(t, truncation=True, max_length=256)["input_ids"]) for t in texts) \
            / max(sum(len(t[:1000]) for t in texts), 1)
        verdict = ("well modelled" if ppl < 10 else
                   "usable, watch it" if ppl < 30 else
                   "POOR -- supervised data will not fix this")
        table[lang] = {"pseudo_ppl": round(ppl, 2), "tokens_per_char": round(tpc, 4),
                       "docs": len(texts)}
        print(f"  {lang:20s} {len(texts):5d} {ppl:11.2f} {tpc:12.4f}  {verdict}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"encoder": args.encoder, "samples": args.samples, "languages": table},
            indent=2), encoding="utf-8")
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
