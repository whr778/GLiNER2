# Synthetic base-training data

Generate broadly-labeled GLiNER2 training data with a configurable LLM. Each
document is written and annotated in one pass for all five task types, every
span is verbatim-validated, and output goes straight into train/val/test JSONL
that `GLiNER2Trainer` reads directly. Style follows the GLiNER2 paper's
synthetic recipe; event annotation is document-level with triggers + typed
arguments (ACE 2005 / WikiEvents style).

## Quick start

```bash
# 1. Estimate cost first — no API calls, no keys needed:
uv run python tools/data/synthetic/generate.py --model gpt-4o --count 50000 --estimate

# 2. Dry-run the whole pipeline with the keyless mock provider (no spend):
uv run python tools/data/synthetic/generate.py --out data/synthetic.jsonl --count 5 --dry-run

# 3. Real generation (needs a key in the environment):
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY for --provider anthropic
uv add openai                    # or: uv add anthropic
uv run python tools/data/synthetic/generate.py \
    --config default.yaml --out data/synthetic.jsonl --count 2000
```

Writes `data/synthetic.train.jsonl`, `.val.jsonl`, `.test.jsonl` (80/10/10 by
default). Train with the mmBERT config or any other:

```bash
uv run python tools/train/train.py --config tools/train/config/mmbert-base-rams.yaml
```

## Two modes

**Generate (default).** The model writes a document and annotates it — the
paper's fully-synthetic half.

**Annotate existing text (`--annotate-from`).** The model annotates the real
text of an existing GLiNER2 JSONL corpus instead of inventing text — the paper's
"real text + synthetic annotations" half. This turns single-task corpora into
multi-task base data: it keeps whatever gold a corpus already has and *merges in*
the missing task types (pass `--annotate-replace` to overwrite gold instead).

```bash
# Re-annotate real ArXiv abstracts (already have a classification label) with
# entities + relations, keeping the existing classification gold:
uv run python tools/data/synthetic/generate.py \
    --annotate-from data/scientific_text.train.jsonl \
    --tasks entities,relations \
    --out data/scientific_reannot.jsonl --count 5000
```

Any `{"input", "output"}` JSONL works: `pubmed_abstracts_ner` (adds relations/
events on NER gold), `docred` (adds events on NER+relations), `docee`/`nuner`,
etc. `--count` caps how many rows to annotate. **Ontology fit matters**: the
ACE/ERE event ontology fires on news/legal/bio text but rarely on math/CS
abstracts — non-fitting tasks simply come back empty (validated out), so pick
`--tasks` per corpus or run all five and let the misfits drop.

## Configuration

Everything is set in `config/default.yaml` and overridable at the CLI:

| Config | CLI | Meaning |
|---|---|---|
| `provider.provider` | `--provider` | `openai` \| `anthropic` \| `mock` |
| `provider.model` | `--model` | e.g. `gpt-4o`, `gpt-4.1-mini`, `claude-sonnet-5` |
| `generation.count` | `--count` / `--limit` | documents to generate |
| `generation.tasks` | `--tasks` | subset of `entities,relations,events,classifications,structures` |
| `generation.min_words`/`max_words` | — | document length |
| `generation.domains` | — | `[]` rotates all built-in registers, or list a subset |
| `split.ratios`/`seed` | `--split-ratios`/`--split-seed` | train/val/test partition |

API keys are read from the environment only — never written to disk or logged.

## Labels

Broad, general-domain ontology lives in `schema_spec.py` (entity types, relation
types, an ACE/ERE/KAIROS-style event ontology with roles, classification tasks +
vocabularies, and structure templates). Edit those lists to broaden or retarget;
no other module hard-codes labels.

## Files

| File | Role |
|---|---|
| `generate.py` | CLI orchestration → `SplitWriter` JSONL |
| `providers.py` | Configurable LLM clients (openai / anthropic / mock) |
| `prompts.py` | Prompt construction + the LLM JSON output contract |
| `schema_spec.py` | The broad label ontology and domains |
| `validate.py` | Verbatim-substring + ontology validation → strict record |
| `cost.py` | Token/price model for `--estimate` |
| `COST_BREAKDOWN.md` | Cost tables across models × corpus sizes |

## How validation keeps the data clean

Like the other `tools/data` converters, anything that isn't a **verbatim
substring** of the document — or whose label/role/task isn't in the ontology —
is **dropped**, never repaired. A document keeps only its clean supervision; if
nothing survives, the document is skipped. The run prints per-task kept/dropped
counts.
