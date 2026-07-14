# Relation Extraction: GLiNER vs GLiNER2

GLiNER (v1, `urchade/GLiNER`) and GLiNER2 take fundamentally different
architectural approaches to relation extraction. GLiNER treats RE as a
**graph problem over pre-detected entities**. GLiNER2 treats it as
**structured slot-filling** — the same span-selection mechanism it uses for
entities and events, though (in this fork) events are trained against a
separate, independently-tunable loss term — see
["Relations vs. events: a shared mechanism, a split loss"](#relations-vs-events-a-shared-mechanism-a-split-loss)
below.

## GLiNER (v1) — dedicated RelEx architecture, two-stage graph pipeline

Selected via `UniEncoderSpanRelexConfig` / `UniEncoderTokenRelexConfig`
(`gliner/config.py:196-273`) — RE is a separate architecture variant, not a
mode of the base model. Pipeline:

1. **Entity detection** — spans/tokens are classified into entity types
   first, exactly as in plain NER, producing a set of entity "nodes."
2. **Adjacency prediction** (`gliner/modeling/multitask/relations_layers.py`)
   — a `RelationsRepLayer` decides *which entity pairs are connected at all*,
   pluggable across 6 graph methods: dot-product/cosine similarity, MLP over
   concatenated pairs, multi-head attention weights, bilinear projection, or
   a GCN/GAT that first message-passes over an initial graph. Output is a
   dense `(B, E, E)` adjacency matrix, trained with `adjacency_loss_coef`.
3. **Triple scoring** (`gliner/modeling/multitask/triples_layers.py`,
   `TriplesScoreLayer`) — for each connected pair, the relation *type* is
   scored against a zero-shot relation-label embedding (via a `<<REL>>`
   marker token / label encoder) using classical **knowledge-graph-embedding
   interaction functions** — TransE, TransH, TransF, PairRE, TripleRE,
   DistMult, SimplE, ComplEx, QuatE, HolE, TuckER, ConvE, ConvKB, etc. (18
   modes total, `interaction_mode` config). This directly reuses the KGE
   literature's `score(h, r, t)` formulations.
4. **Decoding** (`SpanRelexDecoder` / `TokenRelexDecoder` in
   `gliner/decoding/decoder.py:910+`) — entities are decoded first, then
   relations are mapped back onto the decoded entity-pair indices.

There's also an older, separate escape hatch:
`gliner/multitask/relation_extraction.py`'s `GLiNERRelationExtractor` — no
RelEx model at all, just prompt-hacks the plain NER head by prepending
"Extract relationships between entities from the text: " and reusing entity
labels. This predates RelEx and is a workaround, not the real mechanism.

Training also supports data augmentation specific to graphs:
`augment_ent_drop_prob`, `augment_rel_drop_prob`, `augment_add_other_prob`
(randomly dropping entities/relations, adding "other"-relation negatives).

## GLiNER2 — no separate RE module; relations are structured extraction

GLiNER2 has **no adjacency matrix and no KGE scoring at all**. Relations are
just one more instance of its unified schema mechanism, identical to how it
handles `events` and multi-field `structures`:

- `.relations()` on a schema (`gliner2/inference/schema.py:228`) registers
  each relation type with `{"head": "", "tail": ""}` — literally a struct
  with two named field slots.
- At training-data-processing time (`gliner2/processor.py:772`
  `_process_relations`), each relation type becomes a schema entry whose
  fields are `[head, tail]` (optionally swapped for augmentation via
  `swap_head_tail_prob`), and every occurrence in the text becomes a tuple
  of two spans. The label is `[count, [[head_span, tail_span], ...]]` — the
  same shape used for event trigger+role tuples.
- At the model level (`gliner2/model.py:415-506`), there is one shared
  *scoring* function (`compute_struct_loss`) for `entities` / `relations` /
  `structures` / `events`: a count head (`CountLSTM`) predicts how many
  instances of that relation exist, and for each instance a projected schema
  embedding is scored against every candidate span via
  `einsum('lkd,bpd->bplk', span_rep, struct_proj)` — i.e. "for field-slot k
  (head/tail) of instance p, which span l fills it." No entity-pair graph,
  no relation-type embedding scored against a fixed KG-style triple
  function — the relation *type itself* is just another prompt token
  embedding, and "head"/"tail" are extracted as two independent
  span-selection slots conditioned on that prompt, the same way an event's
  trigger and argument roles are.
- The two task types diverge, however, in how that scoring function's output
  is *aggregated into a loss* — see the next section.

### Relations vs. events: a shared mechanism, a split loss

`_compute_sample_loss` (`gliner2/model.py:415-506`) calls the same
`compute_struct_loss` for every non-classification task type, but routes the
result into one of **two separate accumulators** depending on `task_type`
(`gliner2/model.py:467-488`):

```python
if task_type == "events":
    ev_variant = getattr(self.config, "event_struct_loss", None) or getattr(self.config, "struct_loss", "bce")
    ev_pos_weight = getattr(self.config, "event_struct_pos_weight", None)
    span_loss = self.compute_struct_loss(..., variant=ev_variant, pos_weight=ev_pos_weight)
    event_struct_loss = event_struct_loss + span_loss
else:
    span_loss = self.compute_struct_loss(...)   # uses config.struct_loss / struct_pos_weight
    struct_loss = struct_loss + span_loss
```

So `relations` and `entities` (and plain `structures`) share one bucket
(`structure_loss`, tuned by `struct_loss` / `struct_pos_weight`), while
`events` get their own bucket (`event_structure_loss`), which falls back to
the same variant/weight by default but can be **independently overridden**
via `event_struct_loss` / `event_struct_pos_weight` on `ExtractorConfig`
(`gliner2/model.py:56-57, 76-79`). Both terms are summed into `total_loss`
(`gliner2/model.py:344`) but tracked, logged, and eval-aggregated as
distinct components — `classification_loss` / `structure_loss` /
`event_structure_loss` / `count_loss` — all the way through
`compute_losses()`, `_empty_loss_dict()`, and the trainer's step/eval metrics
(`gliner2/training/trainer.py:432, 1273, 1415-1450`).

This split is a deliberate enhancement in this fork, not present upstream
(see `CORE_CHANGES.md`): events tend to have much sparser positive labels
than entities/relations, so giving them their own loss variant/positive
weight lets `struct_loss` be tuned for entities/relations independently of
events, without forcing one loss configuration on both. Relations do **not**
get this same independence from entities — only events are split out.

## The key difference

| | GLiNER (RelEx) | GLiNER2 |
|---|---|---|
| Paradigm | Two-stage: detect entity nodes -> predict graph edges (adjacency) -> score edge relation type via KGE interaction functions | Single-stage: relation type is a schema prompt; head/tail are two span-slots filled directly, like event roles |
| Entities vs. relations | Entities must be predicted first as a separate task; relations link *already-decoded* entity spans | Head/tail spans are extracted fresh per relation instance -- not required to coincide with a separately-run entity-extraction pass |
| Relation-type representation | Zero-shot label embedding scored against entity pairs via 18 pluggable KGE interaction functions (TransE, ComplEx, TuckER, ...) | Relation type is just a schema/prompt token, scored against spans the same way entity-type or event-trigger prompts are |
| Architecture surface | Dedicated config classes, dedicated adjacency + triples modules, dedicated decoder | No dedicated module -- reuses the same `compute_struct_loss` / count-and-span-select path as entities/events/structures |
| Graph structure | Explicit, learned adjacency matrix over entity pairs (optionally refined with GCN/GAT message passing) | None -- no explicit pairwise graph is ever formed |
| Loss tuning vs. events | N/A -- no unified structure loss to compare against | Relations share `struct_loss`/`struct_pos_weight` with entities; events get their own independently-tunable `event_struct_loss`/`event_struct_pos_weight` and a separate `event_structure_loss` total (fork-specific, see `CORE_CHANGES.md`) |

In short: GLiNER v1 borrows from the knowledge-graph-embedding literature
and models RE as **link prediction over an entity graph**; GLiNER2 folds RE
into its general schema-driven **slot-filling** formulation. Architecturally
relations are indistinguishable from event/structure extraction beyond the
field names being `head`/`tail` -- but for loss *tuning*, relations are
grouped with entities, not with events, since events alone get a
split-out, independently-weighted loss term in this fork.
