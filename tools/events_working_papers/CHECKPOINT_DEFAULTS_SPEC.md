# Checkpoint defaults: the library applies what config.json already stores

**Status:** SPEC 2026-10-05, not built. Fixes the library side of 63eb320, which taught only the viewer.

## 1. The gap (measured)

Since 63eb320, `train.py` writes two fields into every checkpoint's `config.json` (`checkpoint_fields`, `train.py:604`):

- `inference_defaults`: the decode settings the eval measured with. For eb18 and the p5link arms these are threshold 0.3, chunk_size 4096, chunk_overlap 0, global_decode true. eb17 has global_decode **false**, so the values really do differ by model.
- `label_map`: the training label transform per category (rollup, separator, map), so a user's `person` reaches the model as the `Person` it learned.

Only the viewer reads them (`viewer/backend/app.py:211`, `:295`). The library does not:

| Call | What it uses today | What the checkpoint says |
|---|---|---|
| `extract`, `batch_extract`, every `extract_*` / `classify_*` (`runtime.py`) | threshold **0.5** | 0.3 |
| `extract_long`, `batch_extract_long` and every `*_long` | chunk_size **384**, overlap **64**, global_decode **False** | 4096 / 0 / True |
| any schema | labels sent as typed | `label_map` never applied |

A library user therefore runs a different operating point and a different label vocabulary from the one every reported score was measured at.

## 2. Survey of the stored fields (every cached checkpoint, 2026-10-05)

- 9 checkpoints carry both fields: eb17-best, eb18-balanced, p3arg x2, p5link x5.
- Every other checkpoint carries **neither**: upstream-shaped, base-v1, pre-63eb320. For those this spec must change nothing.
- Every stored map is **closed**: in all 9, no map target is itself a key. Applying the map twice is therefore a no-op. That makes the viewer's existing mapping harmless, and it makes eval schemas, which `train.py` has already mapped, pass through unchanged.

## 3. Design

**Defaults: `None` means "the checkpoint's value, else today's".**

- `threshold`, `chunk_size`, `chunk_overlap` and `global_decode` default to `None` in every public signature in `runtime.py`. That is about 25 methods; every convenience method funnels into `batch_extract` or `batch_extract_long`.
- One resolver, `_decode_setting(name, value)`, resolves in this order: an explicit argument, then `config.inference_defaults[name]`, then today's literal (0.5, 384, 64, False).
- An explicit argument always wins. A checkpoint without the field behaves exactly as today.

**Label map: applied once at each public entry point.**

- `batch_extract` and `batch_extract_long` map their schemas before anything else reads them.
- `batch_extract_long` must map at its top, because its merge step reads `_schema_event_roles(schema)` and `_scalar_entity_labels(schema)`. Those must see the mapped spellings the model's output carries. Its inner `batch_extract` call maps again, which is a no-op on a closed map.
- **Dict schemas** go through `apply_label_map` directly.
- **`Schema` objects** are what every convenience method builds (`create_schema().entities(...)`). They go `to_dict()`, then `apply_label_map`, then `from_dict()`. Only when the map actually rewrote something is the object replaced; otherwise it is untouched.
- Structures are left alone, as `apply_label_map` already does: their keys are flattened field paths.
- **Opt-out:** `apply_label_map: bool = True` on the two core methods, for a caller who wants raw labels.
- **Visibility:** one INFO line per call that rewrote labels, listing `{category: {sent: model spelling}}`. This is the same report the viewer shows, so a rewrite is never silent.

**Not touched:**
- `api_client.py`: the remote client takes its defaults from the server.
- `joint_ie/engine.py`: a different architecture with no such fields.
- `global_decode_config` and `overlap_policy`: not stored.

## 4. Gates (each must be able to fail)

1. **No-field checkpoint is bit-identical.** Decode the same documents with an upstream-shaped checkpoint before and after the change: identical outputs.
2. **Eval is bit-identical.** `eval_metrics` passes every decode setting explicitly, and its schemas are already mapped. Rerun the p5link-junc_w03 per-corpus eval on a fixed slice before and after: identical scores. It fails if the resolver overrides an explicit argument, or if the map is not closed.
3. **The defaults reach the decode.** `model.extract_long(text, schema)` with no arguments on junc_w03 must equal the same call with 0.3 / 4096 / 0 / True passed explicitly, and must **differ** from 0.5 / 384 / 64 / False on at least one document. Without that second half the gate cannot fail.
4. **The label map reaches the model.** On eb18-balanced, `extract_entities(text, ["person"])` must equal `extract_entities(text, ["Person"])`, provided the map has `person -> Person`; check that before trusting the gate. With `apply_label_map=False` it must equal today's output.
5. **The `Schema` round trip is lossless.** For schemas built through each builder method (entities with descriptions and thresholds, relations, events with role descriptions, classifications, structures), `from_dict(to_dict(s)).build()` must equal `s.build()`, and the metadata must match. Trace this before the code depends on it. If it is lossy, the round trip is replaced by mapping the label lists before `create_schema()`, and this spec is amended.
6. **The merge sees mapped labels.** A long event document with `global_decode` on, queried with an unmapped event-type spelling: the output must carry the arguments, which the role lookup can only find under the mapped spelling.

## 5. Build order (trace, code, trace, test)

1. **Trace before:** junc_w03 and eb18-balanced through `extract_long` and `extract_entities` with no arguments. Print the threshold, chunking and schema labels that actually reach `batch_extract`.
2. **Trace** the `Schema` round trip (gate 5), on real builder output.
3. **Code** the resolver, the `None` defaults, and the mapping at the two entry points.
4. **Trace after:** the same calls; the printed values must now be the checkpoint's.
5. **Tests:** gates 1-6, each mutation-checked:
   - drop the resolver: gate 3 fails;
   - drop the map: gate 4 fails;
   - map only in `batch_extract`: gate 6 fails.
6. **Viewer:** keep its explicit mapping (it reports rewrites to the user). It is idempotent with the library's.

## 6. Risks

- **Upstream API behaviour changes**, but only for checkpoints that carry the fields, which are all ours. Upstream checkpoints are unaffected (gate 1).
- **The threshold is the eval setting, not the sweep.** `inference_defaults.threshold` is written from the config's eval threshold. If a run's validation sweep settles elsewhere, `backfill_model_config.py` must update it, as eb18 was backfilled. eb19 needs this check after its sweep.
- **Double snapshots.** `eb18-balanced` exists on the Hub both with and without the fields (before and after backfill). A pinned old revision gets today's behaviour.
