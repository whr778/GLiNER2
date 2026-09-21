# eb17-best crash -- diagnosis state 2026-09-21

## What happened
- Crashed 22:03:18 UTC, rc=1, at optimizer step 6,930/63,270 (epoch 0.50), ~1h32m in.
- `ValueError: record 'record' declares anchor 'type' but no matching field query was
  found in the layout` -- records.py:349, raised inside a DataLoader WORKER, on the
  TRAIN loader (trainer.py:2227), via collate_fn_train -> _add_boundary_metadata ->
  build_boundary_batch_metadata -> compile_record_specs.
- NOT an OOM, NOT the forked-worker [composition] artefact, NOT eval.
- No checkpoint written (`NO CHECKPOINT AT ./out/eb17-best/best`). Logs + config were
  published to whr778/gliner2-run-logs (eb17_best/). test_metrics.json never written.
- Box terminated 22:36 UTC after ~33 min idle billing. ~2h05 billed on H100 @ $3.29/hr.

## Proven, by trace on real data
1. Only `text2json` carries `record_metadata` among the 18 train files. Verified by
   scanning every file the config actually loads.
2. The box's text2json is BYTE-IDENTICAL to local (md5 1067b01e3adb191c8cb2e1c229824de6).
   The 7,976 -> 7,233 count gap in the composition line is the `validate_data: true`
   split-uniqueness dedup, not different data.
3. `anchor == "type"` occurs in exactly 4 of 7,754 text2json train records
   (lines 570, 2279, 6158, 6671). 2279 and 6671 are duplicates of each other.
4. STATIC CHECK over all 202,211 train records in the real 18-file list, with the real
   unified label map applied: **0 anchors missing from their record's gold field keys.**
5. Those 4 samples do NOT raise in isolation: 0/40 seeds each.
6. Full-fidelity repro on text2json (label map -> chunk_records(4096/3072) -> batched
   collate of 8, training-mode sampling ON): **0 failures over 6 full passes (~50k samples).**

## Why the obvious mechanisms are refuted
- `_process_json_structures` guards record structures: `is_record` disables both
  `remove_json_field_prob` and synthetic relabelling, so a record's anchor field is
  always in `chosen`.
- `normalize_record_metadata` does `if mode is None: continue` -- it never invents a
  mode, so it agrees with the processor's `bool(block.get("mode"))` guard.
- The negatives injector shallow-copies (`out = dict(schema)`) and its structures branch
  is gated on `per_dim["structures"]`, which eb17 does not set (entities/events only).
- `_collate_batch` skip keeps `original_schemas` and `schema_tokens_list` aligned (both
  built from the same filtered `transformed_records`), so no index misalignment.
- `_build_outputs` emits exactly one result per schema; it never drops a task.

## Separate REAL defect found while tracing (not the crash)
`gliner2/training/chunking.py::_filter_record_output` builds a fresh `new_out` and copies
entities / entity_descriptions / classifications / relations / events / json_structures /
json_descriptions -- but **never `record_metadata`**. So every CHUNKED long document
silently loses its record supervision: the structure becomes undecodable on the boundary
record path and trains nothing, with no error. Measured locally on text2json:
7,754 records with metadata -> 8,303 chunks of which only 7,656 carry metadata.
On the box: 201,073 records -> 202,470 chunks.
This is the same failure shape as the already-priced record_metadata repair
(+0.182 structure F1), reintroduced at the chunking stage.

## Still unexplained
The raise itself. Data is clean statically and the single-corpus pipeline does not
reproduce it. Remaining untested condition: the full 18-corpus MIXTURE (batches mixing
text2json with other corpora), which is the only thing the local repro has not run.

## Final state of the hunt, 2026-09-21 end of session

**The full mixture does NOT reproduce it.** Faithful repro -- all 18 train files, unified
label map, `chunk_records(4096/3072)`, `Random(42)` shuffle, the real `NegativeLabels`
injector, the real `ExtractorCollator` (`max_len=4096`, `error_policy=skip`,
`on_capacity_exceeded=skip_sample`, `event_records=True`, `max_gold_per_query=256`):
**203,736 chunks, 0 anchor raises.** The box failed after ~110,880 samples; this run went
past that with nothing.

**The only task named `record` anywhere in the train set is text2json's structure** (7,754
of them). No record-metadata record co-occurs with events or relations, so no event or
relation group can collide with the `record` name.

Also refuted tonight:
- The injector's structures branch is genuinely off: `self.per_dim` is built as
  `{d: int(n) for d, n in per_dim.items() if int(n) > 0}` from the config's
  `{entities: 1, events: 1}`. The gate is real, not a gate that cannot fail. (The
  text2json pool DOES carry a `structures['record']` entry, and its "fields" are garbage --
  `-RRB-`, `.`, `Amazon <> introduced`. Worth fixing before structures negatives are ever
  turned on, but it did not fire here.)
- `_wrap_classification_fields` rewrites field VALUES, never field KEYS.
- `_record_meta` is a plain `schema.get("record_metadata")` -- it synthesizes nothing.
- `validate_data` DROPS whole records (`records = [records[i] for i in valid_indices]`);
  it never strips a field. On the box it removed exactly 1 record of 201,074.
- The `is_record` guard holds: if it did not, `remove_json_field_prob=0.2` would drop the
  anchor on ~20% of draws, and the 4 anchor-`type` samples raised on 0 of 40 seeds.

## THE LEAD TO CLOSE FIRST

**Local and HF corpora have diverged.** The box loaded **201,074** train records; the same
18-file list locally yields **202,211** -- I hold **1,137 records the box did not**. A fresh
box fetches every corpus from HF (`_fetch_if_missing`), so the BOX's corpora are the HF
copies and mine are whatever is on disk. `text2json` is byte-identical (md5
`1067b01e3adb191c8cb2e1c229824de6`), but at least one other corpus is not.

This matters beyond the crash: **every local repro of a GPU run is unfaithful until this is
closed.** Next action is a per-file record-count diff of local vs HF across all 18 train
files, not another repro.

Local per-file counts to diff against HF:
sentence_rex 34,314 | bio_ner_relations 2,084 | biored 308 | chfinann 25,632 |
docee 21,842 | docfee 16,384 | duee_ner 11,323 | cmnee_ner 9,281 | text2json 7,976 |
maven 2,913 | events_biotech 2,217 | casie 798 | turkish_event 24,028 |
professorbob_re 13,926 | scierc 265 | paraloq_json 378 | docee_zh 27,122 |
mendeley_ed 1,420 = 202,211.

## Shipped tonight
`compile_record_specs` now names the fields it actually saw in the raise. The old message
gave the record and the anchor but not the layout, which is why this cost a session rather
than minutes -- the next occurrence identifies its own corpus. 117 record tests pass.

## NOT done, deliberately
No relaunch. There is no fix yet, and $30 on a guess would most likely buy a second crash.
