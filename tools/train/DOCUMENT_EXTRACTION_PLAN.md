# Plan: Optional global graph decoding over windowed candidates (OneIE-style)

## Context

`RECOMMENDATIONS.md` (Document-Level Events, Option 5) proposes assembling
document-level events by sliding overlapping windows over the whole document,
collecting every trigger/argument/entity candidate with confidences and global
offsets, then decoding the best event structures from that candidate graph
(OneIE-style, Lin et al. 2020). Today the long-doc path (`batch_extract_long`)
instead uses `merge_chunk_results`, which concatenates and de-dupes per-window
mentions as opaque whole-event dicts and **never stitches a trigger in one
window to an argument in another**.

This implements it as an **opt-in** mode so short inputs (news headlines) keep
the cheap path with no overhead, while document-level inputs can turn on global
decoding. Scope: expose the toggle on **(a) the inference API, (b) the
training/eval yaml, and (c) a new inference CLI**, and build the **full beam
decoder with global constraints**. Default is off everywhere, so existing
behavior is unchanged.

**Windowing:** eval, blind test, and inference all chunk with **overlapping**
windows — `chunk_size` = training `max_len` (384), `chunk_overlap` = 128 → step
256 = training's `window_stride`, so eval/inference windowing matches training.
Non-overlapping tiling ("no stride") was rejected: it leaves the decoder nothing
to reconnect (each event lives in one window) and drops boundary-straddling
arguments with no overlap to rescue them.

**Honesty caveat (in code + docs):** OneIE *learns* its global-feature weights
during training. We have no such training signal, so this is an OneIE-*shaped*
decoder — beam search over the candidate graph with **heuristic/config-set**
feature weights, not learned ones. Its recall is still bounded by within-window
candidate recall (arguments more than ~one window from their trigger are never
emitted, so cannot be recovered); the beam's value over greedy is cross-event
span-conflict resolution and constraint enforcement.

## Problem verification (measured 2026-07-17)

The eval-yaml surface was gated on proving eval actually loses long-range
content. Confirmed against the real WikiEvents data and runs:

- **Records are whole documents.** `data/wikievents.{train,dev,test}.jsonl`:
  median ~460-553 words, and **50-70% of dev/test records exceed 512 words**
  (subword tokens more), max 8296.
- **Backbone caps at 512.** The wikievents configs are DeBERTa-v3 (fastino
  base/large); training already works around it with `sliding_window`/`max_len:
  384`.
- **Train/eval mismatch.** Training windows to 384 words, but `compute_metrics`
  feeds the **whole** doc to `batch_extract` single-pass with `max_len=None`
  (no chunk, no processor cap) - so eval runs the model out of its trained
  window and past its position cap.
- **Measured headroom is in arguments.** Existing blind-test artifacts
  (`out/fastino/gliner2-*-v1-wikievents/test_metrics.json`): event_type strict
  F1 0.82-0.93 and trigger 0.51-0.52, but **event_argument strict F1 is
  0.068 (base) / 0.28 (large)** - the document-level argument-dispersal
  bottleneck. These files are the A/B baseline for increment 4.

So the eval surface targets a real, measured gap *for this DeBERTa-v3 setup*.
**Caveat to document:** it helps only when docs exceed the model's one-pass
window; on a future long-context (mmBERT/ModernBERT) WikiEvents run the model
sees the whole doc already, and chunk+reassemble would be strictly worse - so
`eval.global_decode` must stay opt-in, never default.

## Key facts from exploration (reuse points)

- **Single hook point:** `merge_chunk_results` (`gliner2/inference/chunking.py:110`),
  called once per doc from `batch_extract_long` (`engine.py:1186`). After
  `remap_result_spans` (chunking.py:121-124) `remapped_results` is a list of
  per-chunk dicts already in **global char offsets with `{text,confidence,start,end}`
  intact** (`batch_extract_long` forces confidence+spans on at engine.py:1176-1177).
  Both paths funnel through `_strip_span_metadata` (chunking.py:126) so output
  flags keep applying.
- **Candidate shape:** `event_extraction = {event_type: [{"triggers":[span],
  "arguments":[{"role", "entity":span}]}]}`; span = `{text,confidence,start,end}`.
  Arguments are grouped per **mention** and implicitly share that mention's
  triggers (`_extract_events`, engine.py:676-713). Entities: `{label:[span]}`.
- **Reusable dedup/overlap:** `_dedupe_items`, `_spans_overlap`, `_span_key`,
  `_is_span_dict` (chunking.py:234-322) - reuse for node clustering.
- **Schema role constraints:** valid roles per event type are available in
  `batch_extract_long` via each schema's `event_role_orders` / `_event_metadata`
  (engine.py:138-147); thread them to the decoder for the role-validity constraint.
- **Eval does NOT chunk today:** `compute_metrics` runs whole-text through
  `model.batch_extract` single-pass (`metrics.py:126-128`). Enabling global
  decode in eval means switching that path to `batch_extract_long(...,
  global_decode=True)` when the flag is set.
- **No inference CLI exists** - only `tools/train/train.py`'s `--config`. The
  CLI is net-new.

## Design

### New module: `gliner2/inference/global_decode.py` (the decoder, model-free)

Pure post-processing over `remapped_results` - no model, fully unit-testable.

- `@dataclass GlobalDecodeConfig`: `beam_width=8`, `trigger_iou=0.5`,
  `arg_iou=0.5`, `conflict_penalty=0.5`, `single_filler_roles: frozenset=()`,
  `min_trigger_conf=0.0`. Sensible defaults; overridable from API/yaml/CLI.
- `collect_candidates(remapped_results)` -> per-doc pools of trigger nodes,
  argument (role, span) nodes tagged by mention, and entity nodes; each carries
  text/span/confidence/source-chunk.
- `cluster_by_overlap(nodes, iou)` -> clusters of same-span nodes across windows
  (reuse `_spans_overlap`); pooled confidence = max, representative = argmax.
- `build_event_graph(...)` -> candidate **events** = clusters of mentions whose
  triggers overlap (same event_type); each event holds pooled triggers and a
  pool of `(role, arg-cluster, conf)` edges. (Entity nodes kept for a later
  arg->entity extension; not emitted in v1 to preserve output shape.)
- `beam_decode(events, event_roles, cfg)` -> beam search assembling the final
  event set. State = partial assignment (events kept + args bound to roles).
  Score = sum log-confidences + global features: **role validity** (hard: role in
  `event_roles[type]`), **cardinality** (single-filler roles keep top-1),
  **cross-event span-uniqueness** (`conflict_penalty` when one span fills
  multiple slots), **trigger floor** (`min_trigger_conf`). Beam keeps top
  `beam_width` partials.
- `assemble_events_global(remapped_results, event_roles, cfg)` -> rebuilds the
  `event_extraction` dict in the **exact normal shape** (`{event_type:
  [mention_dict]}`, spans still carrying confidence) so `_strip_span_metadata`
  and all downstream metrics work unchanged.

### Wire into the long path

- `chunking.py: merge_chunk_results(..., global_decode=False, event_roles=None,
  global_decode_config=None)`. When `global_decode`, replace only the
  `event_extraction` block with `assemble_events_global(remapped_results, ...)`;
  all other blocks keep `_merge_result_dicts`. Still ends with `_strip_span_metadata`.
- `engine.py: batch_extract_long`/`extract_long` gain `global_decode: bool=False`
  and optional `global_decode_config`; pass them plus the per-doc schema's
  `event_role_orders` into `merge_chunk_results` (engine.py:1186-1192). The
  `*_entities_long` variants are unaffected (no events).

### Eval / blind-test / inference windowing (overlapping) + decoder

Windowing and the decoder are **separable knobs** (keeps the A/B measurable):

- `metrics.py`: `compute_metrics(..., chunk_size=None, chunk_overlap=128,
  global_decode=False)`:
  - `chunk_size=None` -> today's whole-doc single pass (baseline, unchanged).
  - `chunk_size` set, `global_decode=False` -> `batch_extract_long(...,
    global_decode=False)` - chunk + existing simple merge.
  - `chunk_size` set, `global_decode=True` -> `batch_extract_long(...,
    global_decode=True)` - chunk + OneIE beam.
  Output shape is identical after `_strip_span_metadata`, so tallying is
  unchanged. Thread the three kwargs through `make_compute_metrics`,
  `evaluate_checkpoint`, `_blind_test_by_language`, `sweep_thresholds`.
- `tools/train/train.py`: read `chunk_size`/`chunk_overlap`/`global_decode` from
  `eval_cfg` alongside train.py:587-593 (`global_decode`: `true` = defaults or a
  dict of overrides, mirroring `threshold_sweep`). Thread to the eval call sites
  (train.py:543, 549, 602-605, 636-638, 651, 654). WikiEvents eval config:
  `chunk_size: 384`, `chunk_overlap: 128`, `global_decode: true`.

### New inference CLI: `tools/infer.py`

`argparse` entry point: `--model`, `--input` (text or JSONL path), `--schema-json`
(a JSON describing entities/events/...) and/or `--entities a,b,c`, `--events`,
`--global-decode`, `--chunk-size`, `--chunk-overlap`, `--include-spans`,
`--include-confidence`, `--beam-width`. Loads `GLiNER2.from_pretrained`, builds a
schema, runs `extract_long`/`batch_extract_long` (global_decode per flag), prints
`json.dumps(..., ensure_ascii=False, indent=2)`.

## Build increments (each validated before the next)

The end state is the full beam decoder. Greedy assembly is built and shipped
first as its own increment because the beam layer *sits on top of it*, and
because it lets us A/B-measure whether the (heuristic-weighted) beam adds
anything over greedy before we tune weights - per the repo's "prove it /
incremental" rules. This is sequencing, not scope reduction.

1. **Greedy assembly + unit tests** - `global_decode.py` core: candidate
   collection, cross-window trigger clustering, argument union (dedupe by
   role+span via `_dedupe_items`), confidence pooling -> `assemble_events_global`
   emitting the exact normal event shape. `tests/test_global_decode.py` on
   synthetic `remapped_results`: arguments split across two overlapping windows
   get unioned into one event; invalid-role dropped; empty/no-event input.
2. **Beam layer + global features** - add `beam_decode` on top of the greedy
   substrate: role-validity (hard), single-filler cardinality, cross-event
   span-conflict penalty, trigger floor; `GlobalDecodeConfig`. Tests: span
   conflict prefers higher-total-score assignment; cardinality; beam vs greedy
   differ only where a conflict exists.
3. **Long-path wiring** - thread `global_decode`+`event_roles`+config through
   `merge_chunk_results`/`batch_extract_long`; end-to-end test via a fake model
   returning per-chunk event blocks: `global_decode=True` unions cross-window
   args; `global_decode=False` is byte-identical to today (regression).
4. **Eval integration** - `compute_metrics` windowed path (separable
   chunk/decoder) + `train.py` yaml. Validate with a **3-point A/B** on the
   wikievents blind test: (i) whole-doc baseline (recorded 0.068 base / 0.28
   large), (ii) chunk + simple merge (`global_decode: false`), (iii) chunk +
   beam (`global_decode: true`) - isolating how much the beam adds over just
   windowing correctly.
5. **CLI** - `tools/infer.py`; smoke-run on a sample document. Lowest leverage /
   net-new; first to cut if scope bites.

Docs last: mark Option 5 implemented in `RECOMMENDATIONS.md`, add a
`tutorial/12-long_context.md` section, note the heuristic-weights caveat.

## Known limitations (document, don't fix in v1)

- **Trigger clustering by span overlap can false-merge** two distinct same-type
  events with adjacent/overlapping triggers into one. Acceptable for v1.
- **Recall ceiling unchanged**: arguments beyond ~one window from their trigger
  are never emitted by the model, so the decoder cannot recover them (pair with
  RECOMMENDATIONS Option 2 trigger-anchored windows later).
- **Beam weights are heuristic**, not learned (no global-feature training signal).

## Verification

- `uv run pytest tests/test_global_decode.py` + the existing chunking/long tests
  - global_decode defaults off, so no regression.
- End-to-end fake-model test through `batch_extract_long` (both flag states).
- `compute_metrics(global_decode=True)` on a synthetic 2-window doc where an
  event's arguments are split across windows - assert the merged event unions
  them and metrics improve vs off.
- Drive `tools/infer.py --global-decode` on a sample long document; eyeball JSON.
- Confirm short-input path untouched: `extract`/`batch_extract` and
  `global_decode=False` unchanged.
