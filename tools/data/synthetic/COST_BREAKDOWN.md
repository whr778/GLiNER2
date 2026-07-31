# Synthetic Data Generation — Cost Breakdown

Estimated API cost to generate synthetic GLiNER2 base-training data with
`generate.py`. One document (all five task types, verbatim-validated) per API
call. Reproduce any figure with:

```bash
uv run python tools/data/synthetic/generate.py --model gpt-4o --count 254334 --estimate
```

Prices move — verify before budgeting a large run. Figures below are USD,
standard (non-batch) tier unless noted, as published **July 2026**.

## Token assumptions (per document)

| Component | Tokens | What it is |
|---|---:|---|
| Input | 1,600 | System prompt + the full broad ontology (entity/relation/event/role/class/structure label lists) + per-document instructions |
| Output | 1,400 | The generated ~180–320-word document + JSON annotations for all five tasks |

These are the defaults in `cost.py` (`DEFAULT_INPUT_TOKENS` / `DEFAULT_OUTPUT_TOKENS`)
and `config/default.yaml` (`cost:` block). Adjust them there if you change the
document length, the task subset, or the ontology size. Fewer tasks or shorter
docs → fewer output tokens; a trimmed ontology → fewer input tokens.

**Annotate mode (`--annotate-from`)** has a different profile: the model does
not write the document, so **output is smaller** (annotations only, ~600–1,000
tokens) but **input is larger** (ontology prompt + the corpus document text). For
abstract-length inputs (~200–500 tokens) the two roughly cancel, so the tables
below are a reasonable proxy; for long documents, input tokens dominate — re-run
`--estimate` with `est_input_tokens` raised to the ontology (~1,600) plus the
document length. Output-heavy models (5× input) get relatively cheaper in this
mode; input-heavy long-doc runs get pricier.

## Model prices (USD per 1M tokens)

| Provider | Model | Input | Output |
|---|---|---:|---:|
| OpenAI | gpt-4o-mini | 0.15 | 0.60 |
| OpenAI | gpt-4.1-nano | 0.10 | 0.40 |
| OpenAI | gpt-4.1-mini | 0.40 | 1.60 |
| OpenAI | gpt-4o | 2.50 | 10.00 |
| OpenAI | gpt-4.1 | 5.00 | 15.00 |
| Anthropic | claude-haiku-4.5 | 1.00 | 5.00 |
| Anthropic | claude-sonnet-5 | 2.00 | 10.00 |
| Anthropic | claude-opus-5 | 5.00 | 25.00 |

## Estimated total cost by corpus size

Columns are document counts. **10k**/**50k** are common starter sizes;
**118,636** matches the synthetic half of the GLiNER2 paper; **254,334** matches
its full training set. Each model shows the standard tier then the Batch API tier
(−50%).

| Model | 10,000 | 50,000 | 118,636 | 254,334 |
|---|---:|---:|---:|---:|
| gpt-4o-mini | $11 | $54 | $128 | $275 |
| &nbsp;&nbsp;+ batch −50% | $5 | $27 | $64 | $137 |
| gpt-4.1-nano | $7 | $36 | $85 | $183 |
| &nbsp;&nbsp;+ batch −50% | $4 | $18 | $43 | $92 |
| gpt-4.1-mini | $29 | $144 | $342 | $732 |
| &nbsp;&nbsp;+ batch −50% | $14 | $72 | $171 | $366 |
| gpt-4o | $180 | $900 | $2,135 | $4,578 |
| &nbsp;&nbsp;+ batch −50% | $90 | $450 | $1,068 | $2,289 |
| gpt-4.1 | $290 | $1,450 | $3,440 | $7,376 |
| &nbsp;&nbsp;+ batch −50% | $145 | $725 | $1,720 | $3,688 |
| claude-haiku-4.5 | $86 | $430 | $1,020 | $2,187 |
| &nbsp;&nbsp;+ batch −50% | $43 | $215 | $510 | $1,094 |
| claude-sonnet-5 | $172 | $860 | $2,041 | $4,375 |
| &nbsp;&nbsp;+ batch −50% | $86 | $430 | $1,020 | $2,187 |
| claude-opus-5 | $430 | $2,150 | $5,101 | $10,936 |
| &nbsp;&nbsp;+ batch −50% | $215 | $1,075 | $2,551 | $5,468 |

## Cost per 1,000 documents (quick planning number)

| Model | $/1k docs (standard) | $/1k docs (batch) |
|---|---:|---:|
| gpt-4.1-nano | $0.72 | $0.36 |
| gpt-4o-mini | $1.08 | $0.54 |
| gpt-4.1-mini | $2.88 | $1.44 |
| claude-haiku-4.5 | $8.60 | $4.30 |
| claude-sonnet-5 | $17.20 | $8.60 |
| gpt-4o | $18.00 | $9.00 |
| gpt-4.1 | $29.00 | $14.50 |
| claude-opus-5 | $43.00 | $21.50 |

## Ways to spend less

- **Batch API (−50%).** Both providers halve input+output for asynchronous
  batch jobs — the single biggest lever for a bulk corpus. Not wired into
  `generate.py` (it does live calls); export prompts and submit a batch job for
  large runs, or accept the live-call cost for smaller ones.
- **Prompt caching (~−90% on repeated input).** The ~1,600-token ontology prompt
  is identical every call. With provider prompt caching the cached portion bills
  at ~10%, cutting input cost roughly to a tenth (input is the smaller half here,
  so the net saving is moderate — a few percent to ~15%).
- **Cheaper model for the bulk, strong model for a slice.** e.g. generate the
  mass with `gpt-4.1-mini`/`gpt-4o-mini` and a quality-anchor slice with
  `gpt-4o`/`claude-sonnet-5`. Mirrors the paper's mixed strategy.
- **Fewer output tokens.** Drop task types you don't need (`--tasks`) or shorten
  documents (`min_words`/`max_words`); output is the dominant cost term.

## What this buys

A `254,334`-document corpus at these settings is the same scale as the GLiNER2
paper's full training set (its own split was ~53% real-text/GPT-4o-annotated +
~47% fully synthetic; this generator produces the fully-synthetic style for all
records). For a first base-training corpus, **`gpt-4.1-mini` at 50k docs (~$144,
or ~$72 batched)** is a sensible quality/price starting point before committing
to a six-figure-document run.
