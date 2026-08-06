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
| Output | 2,500 | The generated ~180–320-word document + JSON annotations for all five tasks |

These figures are **measured**, not guessed: a `claude-sonnet-5` all-five-task run
averages ~1,614 input and ~2,438 output tokens per document (a full document plus
five task blocks is verbose — events with typed arguments dominate the output).
The 1,600 input figure is `cost.py`'s `DEFAULT_INPUT_TOKENS`; the 2,500 output
figure is set in `config/default.yaml`'s `cost:` block (`est_output_tokens`),
which the estimator reads and which reproduces the tables below (`cost.py`'s own
`DEFAULT_OUTPUT_TOKENS` is 1,400 — the config overrides it). Adjust them there if
you change the document length, the task subset, or the ontology size. Fewer tasks or shorter
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

`claude-sonnet-5` uses **introductory pricing ($2/$10 per 1M through 2026-08-31)**;
standard is $3/$15, which raises its figures below by ~1.5×. Verify the current
rate before a large run.

## Estimated total cost by corpus size

Columns are document counts. **10k**/**50k** are common starter sizes;
**118,636** matches the synthetic half of the GLiNER2 paper; **254,334** matches
its full training set. Each model shows the standard tier then the Batch API tier
(−50%).

| Model | 10,000 | 50,000 | 118,636 | 254,334 |
|---|---:|---:|---:|---:|
| gpt-4o-mini | $17 | $87 | $206 | $443 |
| &nbsp;&nbsp;+ batch −50% | $9 | $44 | $103 | $221 |
| gpt-4.1-nano | $12 | $58 | $138 | $295 |
| &nbsp;&nbsp;+ batch −50% | $6 | $29 | $69 | $147 |
| gpt-4.1-mini | $46 | $232 | $550 | $1,180 |
| &nbsp;&nbsp;+ batch −50% | $23 | $116 | $275 | $590 |
| gpt-4o | $290 | $1,450 | $3,440 | $7,376 |
| &nbsp;&nbsp;+ batch −50% | $145 | $725 | $1,720 | $3,688 |
| gpt-4.1 | $455 | $2,275 | $5,398 | $11,572 |
| &nbsp;&nbsp;+ batch −50% | $228 | $1,138 | $2,699 | $5,786 |
| claude-haiku-4.5 | $141 | $705 | $1,673 | $3,586 |
| &nbsp;&nbsp;+ batch −50% | $71 | $353 | $836 | $1,793 |
| claude-sonnet-5 | $282 | $1,410 | $3,346 | $7,172 |
| &nbsp;&nbsp;+ batch −50% | $141 | $705 | $1,673 | $3,586 |
| claude-opus-5 | $705 | $3,525 | $8,364 | $17,930 |
| &nbsp;&nbsp;+ batch −50% | $353 | $1,763 | $4,182 | $8,965 |

## Cost per 1,000 documents (quick planning number)

| Model | $/1k docs (standard) | $/1k docs (batch) |
|---|---:|---:|
| gpt-4.1-nano | $1.16 | $0.58 |
| gpt-4o-mini | $1.74 | $0.87 |
| gpt-4.1-mini | $4.64 | $2.32 |
| claude-haiku-4.5 | $14.10 | $7.05 |
| claude-sonnet-5 | $28.20 | $14.10 |
| gpt-4o | $29.00 | $14.50 |
| gpt-4.1 | $45.50 | $22.75 |
| claude-opus-5 | $70.50 | $35.25 |

Measured: two `claude-sonnet-5` batches of 1,000 docs each cost ~$14 apiece
(batch tier), matching the $14.10/1k line.

## Ways to spend less

- **Batch API (−50%).** Both providers halve input+output for asynchronous
  batch jobs — the single biggest lever for a bulk corpus. **Wired into
  `generate.py` via `--batch`** (Anthropic today): it submits every document as
  one batch, polls, and writes the results. Prefer it for any run over a few
  hundred docs.
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
records). For a first base-training corpus, **`gpt-4.1-mini` at 50k docs (~$232,
or ~$116 batched)** is a sensible quality/price starting point before committing
to a six-figure-document run.
