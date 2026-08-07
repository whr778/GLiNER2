# Counting Layer — Implementation Reference

Status: reference note (how/where the count head is implemented). Date: 2026-08-07.
File:line references are against the code as of this date and may drift.

## TL;DR

- There are **two separate count mechanisms**, one per architecture. `counting_layer`
  (`count_lstm` / `count_lstm_v2` / `count_lstm_moe`) governs **only the span model**
  (`from_encoder` training; the released `gliner2-*-v1` checkpoints). The boundary
  model has its own, unrelated count head.
- In the span model the count head is **not structure-only**. It is the **shared
  multi-instance decoder** for relations, events, and structures (and `count_embed`
  also runs for entities). It is trained on the event corpora and used in event /
  argument decoding. The separate `_extract_events` method can mislead — it consumes
  span scores that were produced by the count head upstream.

## Two components (span model)

Built in the span model `__init__`:

- **`count_pred`** (`gliner2/models/span/model.py:121`) — MLP `hidden -> 2*hidden -> 20`.
  A 20-way classifier that predicts **how many instances** (0-19) a schema has, from
  the schema's anchor embedding.
- **`count_embed`** (`gliner2/models/span/model.py:131-146`) — the module selected by
  `counting_layer`. Given the field embeddings and a count `L`, it unrolls `L`
  count-aware "instance slots" and returns `[L, num_fields, hidden]`.

`max_count = 20` everywhere (counts clamped to 0-19). An unknown `counting_layer`
raises `ValueError` (`model.py:142-146`).

## The three `count_embed` variants (`gliner2/layers.py`)

All share the same skeleton: a learned per-slot positional embedding
(`nn.Embedding(20, hidden)`) fed through a `CompileSafeGRU` conditioned on the field
embeddings. They differ only in the refinement head:

| `counting_layer` | class | refinement | notes |
|---|---|---|---|
| `count_lstm` (default) | `CountLSTM` (`layers.py:138`) | MLP projector over `concat(GRU_out, field_emb)` (`h*2 -> h*4 -> h`) | original |
| `count_lstm_v2` | `CountLSTMv2` (`layers.py:183`) | `DownscaledTransformer` (dim 128, 4 heads, 2 layers) over `GRU_out + field_emb` | ONNX/compile-friendly (slices a full index vector); what the v1 checkpoints ship |
| `count_lstm_moe` | `CountLSTMoE` (`layers.py:219`) | Mixture-of-Experts FFN (4 experts, `ffn_mult=2`) + softmax router | heaviest |

`forward(field_embs [M,D], count L) -> [L, M, D]` for all three.

## Training (`gliner2/models/span/model.py`)

Per-sample loss routing (`model.py:372-411`):

- **classifications** -> `self.classifier` (BCE); count head not involved.
- **entities / relations / events / structures** -> `compute_struct_loss(...)`
  (`model.py:395`). Inside (`model.py:590-592`): `gold_count = min(structure[0], 19)`,
  `struct_proj = self.count_embed(schema_emb[1:], gold_count)`, then score every span
  against each `(slot, field)` via `einsum('lkd,bpd->bplk', span_rep, struct_proj)`.
  This is **teacher-forced on the gold count**. Labels mark the gold span for each
  `(slot, field)` (`model.py:595-614`); the term is masked BCE / bce_posweight /
  focal / asl / dice per `struct_loss` (`model.py:616-634`).
- **count loss** (`model.py:403-411`): for **relations, events, structures**
  (entities are skipped), `count_loss = F.cross_entropy(self.count_pred(anchor_embs),
  gold_counts)`.
- **Aggregate** (`model.py:271-273`): `total_loss = cls + struct + count`, summed
  **1:1**. The span model does NOT apply `count_loss_weight` (that field belongs to
  the boundary arch — see below).

Net effect: on the event corpora, `count_pred` learns the number of event mentions
and `count_embed` learns per-instance argument-field projections. The count head
therefore **does** train on an event-only pool.

## Inference (`gliner2/inference/runtime.py`, shared decode ~`:434-482`)

For every schema, regardless of task type:

1. `count_logits = self.count_pred(embs[0]...)` -> `pred_count = argmax`
   (`runtime.py:434-435`). If `pred_count <= 0`, emit empty (`:437-444`).
2. `struct_proj = self.count_embed(embs[1:], pred_count)` (`:446`);
   `span_scores = sigmoid(einsum('lkd,bpd->bplk', span_rep, struct_proj))` (`:447-450`).
3. Dispatch to `_extract_entities` / `_extract_relations` / `_extract_events` /
   `_extract_structures` (`:452-482`), **each passed `span_scores` + `pred_count`**.

In `_extract_events` (`runtime.py:771-849`): `for inst in range(count):`
(`:817`) -> `scores = span_scores[inst]` (`:818`) -> trigger spans from field 0,
argument spans per role from fields 1..N (`:820-845`). So the count head sets both
the **number of event mentions** and the **per-instance span/argument scores**.

## The other mechanism — boundary model (`gliner2/models/boundary/model.py`)

Unrelated to `counting_layer`. `count_head = nn.Linear(query_dim, 1)` (zeros-init,
`boundary/model.py:198-203`), gated by `enable_count_head`, producing a per-query
count **log-rate** (`count_log_rates = self.count_head(query_states)`, `:409`). This
is where `enable_count_head` / `count_loss_weight` (`configuration.py:105-106`) apply.
`GLiNER2.from_pretrained` is span-only, so the span `counting_layer` work never
touches this path.

## Plumbing

- **Config:** `counting_layer` on `ExtractorConfig` (`configuration.py:692`, default
  `count_lstm` at `:729`), serialized into `config.json`, forwarded YAML `model:` ->
  `from_encoder(**model_cfg)` -> `ExtractorConfig`.
- **Task module** (saved weights, LoRA-targetable): `count_embed`, `count_pred` in
  `task_module_names` (`model.py:68`), `TASK_MODULES` (`training/lora.py`), and the
  default LoRA targets (`training/trainer.py:276`).
- **compile:** `count_embed` is `torch.compile`d with `dynamic=True` (`model.py:914`)
  — the reason for v2's ONNX/compile-safe slicing.

## Consequence for the count_lstm_v2 vs count_lstm question

Because the count head shapes per-instance argument scoring (and the number of event
instances), `count_lstm_v2` vs `count_lstm` **can** affect event-argument extraction,
and the event pool already trains the count head. Whether v2 empirically beats the
default on downstream RAMS argument F1 remains an open experiment, but it is a
mechanistically valid lever — not orthogonal to event arguments.

## Limitation: the 19-instance cap (why this line is a dead end for doc-level events)

`count_pred` is a 20-way classifier and gold counts are clamped `min(count, 19)` in
training, so the span model emits **at most 19 instances per schema type per
document** — the 20th+ event mention / relation / structure record is structurally
unrecoverable (no slot, no class). Properties:

- **Training clamp** means "19" is a saturation bucket; the model can't distinguish
  19 from 60.
- **Native single-pass (mmBERT-8192) is worst-exposed:** it skips windowing/global
  decode, so the whole document's instances of a type must fit under 19 — there is no
  cross-window union to exceed it. A windowed + global-decode path (DeBERTa-v3 512)
  can aggregate past 19 across chunks.
- **Hard cap for high-cardinality tasks.** Re-DocRED routinely has 30-50+ relation
  triples/doc -> `gliner2-base-v1-redocred` truncates at 19. Dense multi-event docs
  (DocEE/CMNEE/MAVEN) hit it too.
- **Entities are exempt** (single slot, multi-valued type-fields; they skip the count
  loss). **RAMS is unaffected** (short, ~single-event -> counts ~1), which is why the
  head-init scaling metric was valid despite the cap.

**No `count_lstm` variant fixes this** — it is a property of the 20-way `count_pred`,
not of the refinement head. The structural fix is the **boundary architecture's
continuous per-query count log-rate** (`boundary/model.py:409`), which has no ceiling.
Decision (2026-08-07): the span `count_lstm_v2` A/B was shelved for this reason;
document-level and beyond-document event work moves to the boundary head + `joint_ie`
constrained decode (+ an EKF/MHT optimizer for streaming/temporal tracking).
