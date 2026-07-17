# Recommendations

_Verified against the codebase on 2026-07-17: every option below is still
unimplemented and valid. Both "works today" starting points are confirmed —
`include_spans=True` returns character spans for entities and event arguments
(`gliner2/inference/engine.py`), and a single schema can hold both `.entities()`
and `.events()` (`gliner2/inference/schema.py`). The two sections below assume
sentence- or window-level events; see [Document-Level Events](#document-level-events)
for how they scope to long documents and what to add for cross-sentence events._

## Connecting Event Arguments to NER Entities

Three options, ordered by complexity.

### Option 1: Post-processing span overlap (recommended starting point)

Extract entities and events in separate calls with `include_spans=True`. After extraction, match each event argument to any entity whose `(start, end)` overlaps the argument's `(start, end)`.

Works today with no model changes. The two extractions are independent — the model has no shared representation across them — but for most use cases position overlap is sufficient: it is exact when both extractors agree on boundaries, and still resolves to the containing entity when they differ (e.g. an argument `Cook` inside the entity `Tim Cook`).

Start here. Validate match quality against your data before investing in a learned approach.

### Option 2: Joint schema

Put entity types and event roles in the same schema. The model processes all fields in one forward pass, so entity spans and argument spans share the same internal span representation. After extraction, match by position as in option 1.

No model changes required. Adds schema complexity but gives richer joint context during inference.

### Option 3: Learned linking head in `gliner2/model.py`

Add a linking head that scores (event argument span, entity span) pairs — a bilinear or dot-product scorer over the shared span representations. Closest to how models like DEGREE and OneIE work. Gives learned linking rather than heuristic position matching.

Requires training data with explicit argument-to-entity coreference annotations and adds parameters and training complexity. Only worth pursuing once option 1 is validated and you have the supervision signal to train it.

---

## Event Arguments as Coreferenced Entity Clusters

The goal here is broader: group all mentions — both NER entity mentions and event argument mentions — that refer to the same real-world entity into coreference clusters. Four options, ordered by complexity.

### Option 1: Post-processing — span overlap + text normalization (recommended starting point)

Extract entities and events with `include_spans=True`. For each event argument, find the NER entity with the most overlapping character span. Group all arguments that map to the same entity under that entity as the canonical cluster head.

To handle surface variation (e.g. "Tim Cook" vs "Cook"), normalize by stripping titles, lowercasing, and checking substring containment before falling back to span overlap. Works today, no model changes. Breaks on pronouns and cross-sentence aliases.

### Option 2: Post-processing — embedding similarity clustering

Instead of matching by span position, extract contextual embeddings for each span and cluster by cosine similarity. GLiNER2 computes internal span representations during inference (`compute_span_rep` in `gliner2/model.py`); exposing them (a small addition there, surfaced through `gliner2/inference/engine.py`) lets you run agglomerative or k-means clustering over both entity and argument spans without any new training.

Handles aliases and some cross-sentence coreference that text overlap misses. Requires choosing a similarity threshold and adds inference latency proportional to the number of spans.

### Option 3: Expose span representations from `gliner2/model.py`

Add a `extract_with_span_embeddings()` method (or a flag on `batch_extract`) that returns the raw span representation tensors alongside the normal extraction output. Your own clustering (option 2) can then run directly on those tensors; separately, off-the-shelf text-based coref models like `fastcoref` or `coreferee` run on the extracted text spans — they do their own mention detection, so they consume text, not the tensors.

Moderate model.py change, no new training. Keeps coreference logic decoupled and swappable.

### Option 4: Learned pairwise mention-scoring head in `gliner2/model.py`

Add a mention-pair scoring head (bilinear scorer over span representation pairs, as in SpanBERT-coref or DEGREE) that explicitly scores whether two spans — regardless of whether they came from the entity task or an event argument task — are coreferent. Training targets would be coreference cluster annotations (e.g. from OntoNotes or a domain-specific annotation pass).

Full end-to-end solution; the model learns to resolve aliases, pronouns, and cross-sentence references. Requires coreference-annotated training data and significant additional training complexity. Pursue after validating that options 1–3 leave a measurable gap on your evaluation set.

---

## Document-Level Events

Both sections above assume an event's trigger and arguments were extracted **together, in one window**. That holds for sentence- or paragraph-level events but breaks for **document-level** events (e.g. WikiEvents), where an argument often sits several sentences from its trigger.

**Why the options above are not enough.** Trigger↔argument association is **window-bounded**. Training uses `sliding_window`/`max_len` (384 tokens in the WikiEvents configs) and inference chunks the text via `split_text_into_chunks`, so the span event head (`_extract_events` in `gliner2/inference/engine.py`) only links a trigger to arguments in the *same* window. `merge_chunk_results` (`gliner2/inference/chunking.py`) concatenates and de-dupes mentions across chunks — it does **not** stitch a trigger in one chunk to an argument in another. So a cross-window argument is an **upstream extraction gap**: the model never emits it for that event, and the argument→entity linking and coreference options above (which run *after* extraction) cannot recover it. Relatedly, linking Option 2's "one forward pass, shared representation" premise holds only *within* a window, and the span-overlap options need both entity and event extraction run through the long-doc path (`batch_extract_long`) so spans share one global coordinate system.

To extract document-level events, close the window bound first. Options, ordered by complexity.

### Option 1: Widen the context window (recommended starting point)

This is a **train-time** change, not an inference knob. The head only associates trigger–argument pairs of the kind it saw in training, so widening the window at inference alone does nothing — the current checkpoints were trained with `sliding_window` at `max_len` 384 and never saw longer-range pairs (and the DeBERTa-v3 fastino checkpoints cap at ~512 positions regardless). **Retrain on a long-context backbone** — mmBERT / ModernBERT is long-context (thousands of tokens) — with a larger `max_len` (or `sliding_window` off) so the trigger and its distant arguments fall in one forward pass; the existing span event head then learns to associate them with **no architecture change** (no new head, but yes retraining). Costs: attention is quadratic in length, and the prepended schema eats into the budget. Confirm the encoder's real position limit before raising `max_len`. Targets the root cause — validate it before anything below.

### Option 2: Trigger-anchored windows

If a document still exceeds the backbone's context, stop using fixed sliding windows for events. First detect triggers document-wide (cheap, local), then run argument extraction in a window **centered on each trigger** so its arguments fall in range. Needs a change to the inference chunker (`split_text_into_chunks` is position-based, not trigger-aware) and, ideally, trigger-centered training windows to match. Cheaper than a new head, but still misses arguments beyond the window radius.

### Option 3: Two-stage document-level argument reader

Split detection from argument extraction. Stage 1: local trigger + event-type detection (already a strength). Stage 2: for each trigger, a document-level argument pass that takes the **whole document plus the trigger as the query** and scores role spans anywhere in the doc — the PAIE / EEQA / TabEAE paradigm, or conditional generation as in the original WikiEvents model (Li et al. 2021). Largest change: a new inference mode and training data conditioned on the trigger, but it is how document-level argument-extraction SOTA is built. Pursue once Options 1–2 leave a measurable cross-sentence gap.

### Option 4: Graph- or knowledge-augmented linking

Add a document-level graph over candidate mentions and let graph *structure* — not linear proximity — carry signal between a trigger and a distant argument. Two flavors, in increasing weight:

- **Mention-graph augmentation (no external data).** Build edges among spans for coreference, co-occurrence, and co-type, then pass messages over that graph (graph attention or a GNN) over the span representations GLiNER2 already computes, before scoring argument roles. This is the directly applicable variant — it needs only those span reps plus a coreference signal (the [coreference options](#event-arguments-as-coreferenced-entity-clusters) above), no external knowledge base. See *A Semantic Mention Graph Augmented Model for Document-Level Event Argument Extraction* ([arXiv 2403.09721](https://arxiv.org/abs/2403.09721)).
- **External knowledge-graph injection (heaviest, domain-gated).** When entities can be linked to a knowledge graph, inject KG relation embeddings (e.g. employment, shareholding) so long-range roles can be resolved from known entity relations — the event-enhanced KGE line, *EventKE* ([Findings of EMNLP 2021](https://aclanthology.org/2021.findings-emnlp.120/)) and *EventKGE*. Caveat: these are knowledge-*embedding* methods that enrich representations, not extractors on their own; they require an entity linker plus a maintained KG and pay off mostly in relation-dense domains like finance.

Largest change in this section: a graph-construction step and a GNN / graph-attention component (plus, for the external-KG flavor, an entity linker and KG embeddings). Builds directly on the coreference infrastructure above. Pursue only once Options 1–3 leave a residual long-range gap — the reported gains for graph/KG augmentation are incremental over a strong long-context baseline, not a substitute for one.

### Option 5: Global graph decoding over windowed candidates (OneIE-style)

Rather than change how the model *links*, change how its outputs are **assembled**. Slide overlapping windows over the whole document with `include_confidence`/`include_spans`, remap to global offsets, treat every trigger, argument, and entity as a scored node in one document-level graph, then search that graph for the globally best set of event structures — the OneIE paradigm of local node/edge scoring plus a beam decoder under global constraints ([Lin et al., ACL 2020](https://aclanthology.org/2020.acl-main.713/)). Per event the structure is a tree (trigger root → role → argument); across events, shared arguments and coreference make it a graph.

Pipeline: (1) windowed candidate generation — native: spans, confidences, and global offsets via `remap_result_spans`; (2) consolidate nodes detected across overlapping windows by span overlap, pooling confidence — this replaces the naive concatenate-and-dedupe in `merge_chunk_results`; (3) score edges — within-window trigger→argument edges come from the model, argument→entity edges from overlap/coref; (4) decode the best event assemblies under global constraints (valid roles per type, one-vs-multi per role, coref-merged arguments).

Two honest limits. **Cross-window trigger→argument edges have no native score** — the model only scores an argument against a trigger in the *same* window, so overlapping windows recover only pairs within ~one window and the decoder cannot invent an edge that was never scored. This option therefore **consumes** the long-range edges produced by Options 2–3 rather than producing them itself; its recall ceiling is set by that upstream candidate/edge recall. And **beam search is the general form** — for largely independent events a greedy per-trigger threshold assignment suffices; reserve beam/ILP for global interactions like shared arguments.

The appeal: a first version is **inference-only** — pure post-processing over outputs you already produce — so it is far cheaper to prototype than the Option 1 retrain, and it can reuse the confidence calibration from `sweep_thresholds` (`gliner2/training/metrics.py`) to make cross-window scores comparable. Best paired with Option 2 feeding it trigger-anchored long-range edges.

**Why graph beam search and not a chain HMM/Viterbi?** Viterbi is exact MAP decoding for a *linear chain* with Markov transitions, so it fits genuinely sequential sub-problems — within-window span (BIO) tagging, or event temporal ordering across sentences. It does **not** fit argument binding: an event is a tree/graph where a trigger binds a *set* of arguments scattered across the document, so the decisive dependency ("this argument belongs to that distant trigger") is long-range and non-sequential, violating the Markov assumption. Forcing it into a chain needs states that remember which event is being filled across intervening spans — which breaks the Markov property or explodes the state space. Viterbi and this beam decoder are the same idea (MAP inference over structured output) at two ends of a spectrum: exact for chains, approximate for graphs; the event structure is a graph.

### Argument coreference and evaluation

Document-level benchmarks score an argument as correct when it matches **any coreferential gold mention** (WikiEvents "Arg-C" with coref), not only the exact surface the model emitted. So (a) resolve extracted arguments to entity clusters via the [coreference options](#event-arguments-as-coreferenced-entity-clusters) above, and (b) make the argument metric coref-aware — otherwise a correct argument phrased as a different mention of the right entity scores as a miss. The strict/relaxed/fair regimes in [METRICS.md](../../METRICS.md) soften *surface* mismatches but do **not** resolve coreference, so document-level argument F1 measured here is a lower bound until coref is added.
