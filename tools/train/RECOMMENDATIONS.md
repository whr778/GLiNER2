# Recommendations

## Connecting Event Arguments to NER Entities

Three options, ordered by complexity.

### Option 1: Post-processing span overlap (recommended starting point)

Extract entities and events in separate calls with `include_spans=True`. After extraction, match each event argument to any entity whose `(start, end)` overlaps the argument's `(start, end)`.

Works today with no model changes. The two extractions are independent — the model has no shared representation across them — but for most use cases position overlap is sufficient and provably correct (same surface text = same span).

Start here. Validate match quality against your data before investing in a learned approach.

### Option 2: Joint schema

Put entity types and event roles in the same schema. The model processes all fields in one forward pass, so entity spans and argument spans share the same internal span representation. After extraction, match by position as in option 1.

No model changes required. Adds schema complexity but gives richer joint context during inference.

### Option 3: Learned linking head in model.py

Add a linking head that scores (event argument span, entity span) pairs — a bilinear or dot-product scorer over the shared span representations. Closest to how models like DEGREE and OneIE work. Gives learned linking rather than heuristic position matching.

Requires training data with explicit argument-to-entity coreference annotations and adds parameters and training complexity. Only worth pursuing once option 1 is validated and you have the supervision signal to train it.

---

## Event Arguments as Coreferenced Entity Clusters

The goal here is broader: group all mentions — both NER entity mentions and event argument mentions — that refer to the same real-world entity into coreference clusters. Four options, ordered by complexity.

### Option 1: Post-processing — span overlap + text normalization (recommended starting point)

Extract entities and events with `include_spans=True`. For each event argument, find the NER entity with the most overlapping character span. Group all arguments that map to the same entity under that entity as the canonical cluster head.

To handle surface variation (e.g. "Tim Cook" vs "Cook"), normalize by stripping titles, lowercasing, and checking substring containment before falling back to span overlap. Works today, no model changes. Breaks on pronouns and cross-sentence aliases.

### Option 2: Post-processing — embedding similarity clustering

Instead of matching by span position, extract contextual embeddings for each span and cluster by cosine similarity. GLiNER2 computes internal span representations during inference; exposing them (a small addition to `engine.py`) lets you run agglomerative or k-means clustering over both entity and argument spans without any new training.

Handles aliases and some cross-sentence coreference that text overlap misses. Requires choosing a similarity threshold and adds inference latency proportional to the number of spans.

### Option 3: Expose span representations from model.py

Add a `extract_with_span_embeddings()` method (or a flag on `batch_extract`) that returns the raw span representation tensors alongside the normal extraction output. External coreference algorithms — including off-the-shelf coref models like `fastcoref` or `coreferee` — can then operate on those embeddings or on the extracted text spans.

Moderate model.py change, no new training. Keeps coreference logic decoupled and swappable.

### Option 4: Learned pairwise mention-scoring head in model.py

Add a mention-pair scoring head (bilinear scorer over span representation pairs, as in SpanBERT-coref or DEGREE) that explicitly scores whether two spans — regardless of whether they came from the entity task or an event argument task — are coreferent. Training targets would be coreference cluster annotations (e.g. from OntoNotes or a domain-specific annotation pass).

Full end-to-end solution; the model learns to resolve aliases, pronouns, and cross-sentence references. Requires coreference-annotated training data and significant additional training complexity. Pursue after validating that options 1–3 leave a measurable gap on your evaluation set.
