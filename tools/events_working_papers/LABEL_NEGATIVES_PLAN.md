# Label negatives: implementation plan

**Status: PLAN, nothing implemented.** Living checklist — tick items as they land and record
the measurement that proved each one. Companion to [[EVENT_ARGUMENT_DIAGNOSIS]] §4h, §4i.

---

## 0. The evidence this exists to fix

| measurement | value | source |
|---|---:|---|
| incumbent fires on a schema of **only absent** event types | **63 / 100 docs** | `record_sweep_results/absent_type_firing.txt` |
| event-records epoch 2, same test | **54 / 100 docs** | same |
| incumbent real `event_type` precision, full 8-type menu | **0.5521** | `probe_event_type_fp.py` |
| ...as the blind test reports it | **1.0000** | pinned by construction |
| event types invented, 150 docs | **142 of 317** predictions | same |
| explicit label negatives anywhere in `data/` | **0** | 14 corpora scanned |
| absent queries in a training batch | **0 of 574** | `probe_absent_queries.py` |

`event_type` F1 = `2R/(1+R)` in **12 of 12** readings on file, because precision is pinned.
Every `event_type` F1 this project has quoted is a reparameterisation of recall.

---

## 1. What ALREADY EXISTS — do not rebuild any of this

The loss side is largely done. **This is a menu problem, not a loss problem**, and new loss
code should be treated as a smell everywhere except the record path (§4, item R).

- [x] **Taxonomy derivation** — `derive_schema()` (`inference/schema.py:31`) unions every
      entity label, event type + roles, relation name and classification task from gold
      records. Already called at `train.py:1324` and persisted as `default_schema` in
      `config.json` (incumbent: 858 entities, 78 event types, 827 relations, 4 tasks).
- [x] **Negative-query selection** — `negative_query_ratio` (**0.5**) and
      `max_negative_queries_per_batch` (**64**) at `models/boundary/model.py:845-864` sample
      `absent_queries = query_mask & ~positive_queries` into the pair loss. **Live in every
      run ever trained, and it has always selected from an empty set.**
- [x] **Abstention head** — `abstention_loss` (weight **0.2**), a per-query gate whose target
      is 1 for an absent query (`losses.py:601`). **Consumed at decode**
      (`engine.py:225-226`, `abstention_threshold` at 390 and 965), so training it reaches
      inference rather than being a training-only ornament.
- [x] **Count head** — `count_log_rate_loss` (weight **0.2**) supervises a count of zero.
- [x] **Span-axis hard negatives** — `hard_negatives_per_positive: 5`,
      `select_hard_negative_candidates`. These teach *"this span is not a `Person`"*. They are
      a different axis and stay as they are.
- [x] **Classification already has real negatives** — `InputExample.from_dict` carries
      `labels=cls_data["labels"]` (the full menu) with `true_label` separate. This is why its
      precision (0.5862) is real while `event_type`'s is 1.0000. **Classification is a
      VERIFY-ONLY item below, not a build item.**

**The gap is one line of intent:** `entities = output.get("entities")` and one `Event` per
gold event (`training/data.py:1143-1195`). The menu is the answer key.

### The representation of an absent query, per dimension — established by review

- **Entities: SOLVED, already supported end to end.** An absent entity query is
  `{"label": []}` — the label mapped to an EMPTY LIST. `GuideScores.inject` emits exactly
  that (`guide_scores.py:107`), so the collator, target builder and losses already accept it.
  This is also why "zero empty entity labels in `data/`" was the right thing to measure.
- **Events: NOT SUPPORTED — this is real implementation, not injection.** The training path
  skips any event whose triggers are empty
  (`processor.py:1183`, `if ... not triggers: continue`), so an `Event(type=X, triggers=[],
  arguments=[])` produces **no query at all** — measured, the query count does not move. But
  `_process_events` already accepts a second shape: the **inference schema**
  `dict[event_type, list[role]]`, which appends `labels.append([0, []])` — a menu entry with
  empty gold, i.e. precisely an absent query. The work is to let a training record carry
  menu-only event types alongside its gold list.
- **Relations / structures: to be established** the same way (Phase 3).
- **Classification: already has it** via `labels` + `true_label`.

### Two augmentations that are NOT negatives — do not confuse them

`SamplingConfig` (`processor.py:271`) is live in training (`sampling = self.sampling_config
if self.is_training else None`, line 909; `collate_fn_train` sets `is_training = True` at
line 418). It contains:

- `synthetic_entity_label_prob: 0.2` — **RENAMES** real labels to `entity 1`, `entity 2`, ...
  (`processor.py:1049-1057`). Label ANONYMISATION, forcing reliance on descriptions. It adds
  nothing absent.
- `remove_entities_prob` / `remove_entity_prob` / `remove_events_prob` / `remove_relations_prob`
  — **REMOVE positives**, shrinking the menu. The opposite direction.
- `max_num_labels: 1000` — an existing per-schema budget knob worth reusing for the token
  budget below rather than inventing another.

Neither adds a label the model must reject. The measurement stands: **0 absent queries of
574** in a real training batch.

---

## 2. Design decisions, made

- **Inject in `ExtractorDataset.__getitem__`, beside GIST — REVISED after code review.** The
  plan first said "the collator". The codebase already has this exact hook: `__getitem__`
  calls `self.guide_scores.inject(text, schema, n)` to add **absent entity queries**
  (`trainer.py:582`). Mirroring it costs three lines and inherits a proven path;
  a parallel collator mechanism would be the duplication this review exists to avoid.
  `__getitem__` runs per item per epoch in the worker, so seed as
  `f(seed, epoch, record_index)` — the trainer sets the epoch on the dataset, as
  `DistributedSampler.set_epoch` does.
- **Data is NEVER rewritten.** Runtime injection only. No stamping — the 108x blow-up from
  `stamp_field_cardinality` is on file. The only data-side artifact is a derived, cached
  per-corpus pool.
- **Phase 1 random, phase 2 hard.** Confusable negatives are more valuable and much easier to
  get wrong; earn them after random ones prove the plumbing.
- **Eval gets a NEW MODE with NEW METRIC KEYS** (`*_fullmenu_*`), never a silent change to
  `_schema_from_gold`. Every historical number is gold-menu; the menu is part of the metric's
  identity, which extends this project's rule that a metric is quoted with its type.
- **Token budget is a first-class config parameter.** Every injected label is schema-marker
  tokens: it costs throughput and eats the 8192 input budget. `k_negatives` is capped per
  dimension, not "add the taxonomy".

---

## 3. THE TRAPS — design for these or the run is worse than useless

1. **FALSE NEGATIVES FROM INCOMPLETE ANNOTATION. This is the one that can silently sink it.**
   The corpora annotate different things: cmnee has **zero** entity gold, biored **zero**
   events. Sampling `Organization` as a negative into a cmnee document teaches the model that
   a real organisation is not one, at loss weight, thousands of times. Two rules make phase 1
   safe:
   - **within-corpus**: sample only from labels that corpus itself annotates;
   - **within-dimension**: inject entity negatives only into records that carry entity gold,
     event negatives only into records with event gold, and so on.
   `tools/data/repair_contradicted_negatives.py` existing at all is evidence this class of
   contradiction has bitten this project before.
2. **DERIVE THE POOL AFTER LABEL UNIFICATION.** If the pool holds a pre-map alias (`LOC`) of a
   present canonical label (`Location`), the injected negative directly contradicts gold.
   Derive from the same post-transform records `derive_schema` sees at `train.py:1324`, and
   assert `negative ∉ gold labels` for that record at injection time.
3. **DERIVE FROM RECORDS, NOT FROM `default_schema`.** `_OPEN_VOCAB_LIMIT = 1000` drops a
   dimension past the cap — epoch-2's config carries `open_vocab: ["entities"]` with no entity
   list at all. The persisted copy is for inference and the viewer; the training pool must
   come from the corpus.
4. **PROVENANCE FOR MIXED CORPORA.** `mix_natural`, `warmstart_mix` and the replay files merge
   sources. Rule 1 needs to know which corpus a record came from — the loader knows the source
   file at load time; if a record cannot be attributed, fall back to a co-occurrence pool
   (labels seen alongside this record's labels) rather than the global taxonomy.
5. **OVERCORRECTION IS THE FAILURE MODE ON THE OTHER SIDE.** An abstention gate trained too
   hard collapses recall. Every gate below pairs the rejection metric with a recall guard.

---

## 4. Checklist

### Phase 0 — instrument first
- [x] `tools/train/probe_absent_queries.py` — counts absent queries in a real training batch.
      Reads **0 of 574** today. This is the before/after gate for the whole feature.
- [x] `tools/train/probe_event_type_fp.py` — real precision against a full menu (0.5521).
- [x] Absent-type firing probe — 63% / 54%. Promote to a committed tool alongside the above.
- [ ] Decide and document the acceptance targets (§5).

### Phase 1 — the pool ✅ DONE
- [x] `tools/data/build_negative_pools.py`: per-corpus, post-label-map pools for entities,
      events (types **and** roles), relations, structures →
      `tools/train/config/labels/negative_pools.json`.
- [x] Records **which dimensions each corpus annotates**, with per-dimension record counts so
      a thin dimension is visible rather than flipped by one stray record.
- [x] Reuses the training pipeline's own transforms (`load_labels_cfg`, `_category_fns`,
      `transform_record`) rather than re-implementing unification.
- [x] `tests/data/test_negative_pools.py` — 5 passing: empty entity pool for an events-only
      corpus, empty event pool for an entities-only corpus, post-label-map canonicalisation
      (`LOC` never survives alongside `Location`), record counts, sampling limit.

**The output validates the design empirically.** Scanned over `eb16-eventrecords-tr`:

| corpus | entity labels | event types | relations | annotates |
|---|---:|---:|---:|---|
| cmnee | **0** | 8 | 0 | events only |
| maven | **0** | 168 | 0 | events only |
| duee | **0** | 38 | 0 | events only |
| biored | 6 | **0** | 8 | entities, relations |
| docee | 61 | **0** | 0 | entities only |
| sentence_rex | 0 | 0 | 450 | relations only |
| paraloq_json | 1,349 | 0 | 0 | entities only |

A global taxonomy would have offered entity negatives to cmnee — which annotates no entities
at all — and event negatives to biored. **That is the contradiction §3.1 predicted, and the
per-corpus pool is what prevents it.** Note `docee` carries 61 entity labels and zero events
despite being an event corpus: its events were converted to entities + classifications
(EVENT_ARGUMENT_DIAGNOSIS §6).

### Phase 2 — the injector ✅ DONE for entities and events
- [x] Config knobs on `TrainerConfig`, beside GIST: `negative_pools`,
      `negative_labels_per_dim`, `negative_label_seed`. **Empty = OFF**, so every existing
      config reproduces bit-for-bit.
- [x] `gliner2/training/negatives.py` — `NegativeLabels`, mirroring `GuideScores`.
- [x] Injected at `ExtractorDataset.__getitem__` beside the GIST hook.
- [x] Entities: `{"label": []}`, the representation GIST already uses.
- [x] Events: `_process_events` now reads an `absent_events` key and emits
      `labels.append([0, []])` — the same menu-entry-with-empty-gold the inference path emits.
      Needed because the training list SKIPS an event with no triggers.
- [x] Deterministic `sha256(seed, epoch, index)` — not `hash()`, which is salted per process
      and would give DDP ranks different menus. Trainer calls `set_epoch` in the epoch loop
      beside `DistributedSampler.set_epoch`, so negatives resample per epoch.
- [x] Asserted at injection: no injected label appears in that record's gold, either dimension.
- [x] `composition_line()` that can fail an A/B gate; logged once on the first epoch.
- [x] **CORPUS IDENTIFICATION WITHOUT PROVENANCE.** Records carry only `{"input", "output"}`.
      Candidate corpora are those whose pool contains every gold label the record uses, and
      the usable pool is their **intersection**; a dimension is vetoed if any candidate does
      not annotate it; no candidate means no negatives. Every fallback is conservative.
- [x] 13 tests: off-by-default, empty-list representation, never-injects-gold (50 draws),
      within-dimension both ways, no-candidate, determinism, per-epoch resampling, pool
      exhaustion, composition line, and **three end-to-end** through the real collator.

**THE GATE MOVED.** 100 real documents, `{"entities": 2, "events": 1}`:

| corpus | docs | queries before | after | docs with an absent query |
|---|---:|---:|---:|---:|
| cmnee | 51 | 340 | 585 | 42 |
| casie | 1 | 11 | 23 | 1 |
| docee | 2 | 7 | 11 | 2 |
| sentence_rex | 45 | 50 | 60 | 9 |
| **ALL** | **100** | **415** | **686** | **54** |

**271 absent queries created where there were 0.** `negative_query_ratio` and
`abstention_loss` finally have a positive class.

- [x] **Relations**: `absent_relations` as a NAME LIST. The inference shape
      `{name: {"head": "", "tail": ""}}` is unusable — the training loop checks
      `all(f in occ for f in field_names)` and appends `occ[f]`, so head/tail
      present-but-empty becomes a GOLD pair of empty surfaces rather than an absence.
      Verified: +2 absent relations = +4 queries.
- [x] **Structures**: `absent_structures` as `{name: [field, ...]}`, **and the injector emits
      `record_metadata` for them**. Without it `compile_record_specs` builds no spec, the
      record head never sees the negative, and it is a schema entry nobody decodes — measured,
      the query count did not move until the metadata was added.
- [x] Token budget: `max_per_record`, a single number across dimensions. Every injected label
      is schema-marker tokens, so it costs throughput and eats the input budget.

**Regression caught and fixed while doing this:** `build_negative_pools.py` originally did
`sys.path.insert(0, "tools/")` + `from train.train import ...`, which made the bare name
`train` resolve to the PACKAGE `tools/train/` and broke two sibling test modules that do
`from train import ...`. Only visible in a full-suite run. It loads `train.py` by file path
under a unique module name now.

### Phase 3 — per-head verification (mostly verify, not build)
- [x] **Absent queries reach the loss.** Measured through the real collator with targets
      built, 60 documents: control **0 of 559 (0.0%)**, negatives on **306 of 865 (35.4%)**.
      That is simultaneously the abstention gate's positive class (`target = 1` for an absent
      query) and `negative_query_ratio`'s selection pool. Both had been live and starved.
- [x] Selected-negative count logged in-band, once:
      `negative queries: 2 absent available, 1 selected into the pair loss`. An arm printing
      `available=0` has not applied the treatment, whatever its config says. This block ran in
      every model ever trained and always saw 0.
- [x] **Relations**: `_relation_loss` handles a zero-gold spec — a real training step with two
      absent relations is finite and puts gradient on the head.
- [x] **Structures**: an absent structure needs `record_metadata` or nothing decodes it;
      the injector emits it (Phase 2).
- [x] **Classification**: VERIFY ONLY, and verified — the injector leaves `classifications`
      byte-identical and adds no `absent_classifications`, while still injecting other
      dimensions. It already has a real menu, which is why its precision is real.
- [x] Real training steps with absent entity, relation and event queries: all finite, all with
      gradient, and the loss **changes** when negatives are added rather than ignoring them.
- [ ] Confirm the decode path rejects. `abstention_threshold` is READ at decode
      (`engine.py:225`, `390`, `965`), so the wiring is confirmed; the behavioural
      confirmation is the A/B acceptance metric (absent-type firing, 63% → lower).
- [ ] **R. RECORD PATH under `event_records: true`** — an absent event type as a record spec
      with zero gold instances. Not yet exercised; the event-records base is the run that
      would hit it.

### Phase 4 — eval and inference
- [x] `_widen_with_absent` + a `full_menu` parameter on `compute_metrics`, and
      `evaluate_checkpoint` runs a SECOND pass merging `eval_fullmenu_*` keys beside the
      originals. `_schema_from_gold` is untouched, so every published number still means what
      it meant.
- [x] Menu source: the model's own `default_schema` (the taxonomy it was trained on).
- [x] `uv run python tools/train/eval.py --full-menu`.
- [x] 5 tests + end-to-end on a tiny model: entity precision **0.25 → 0.105** when the menu
      widens from 2 labels to 5, which is the whole point — an untrained head fires
      indiscriminately and the gold menu cannot show it.
- [x] Widening never invents a dimension the record did not carry (that would change WHICH
      documents are scored for a head, not just the menu).
- [ ] One-time re-baseline of **both** models under the new mode (needs a GPU).
- [ ] Fixed-seed negative menu for the per-epoch eval, so rejection has a training **curve**.
- [x] `model_card.py` states the menu next to the metric table, so a published card cannot be
      read as if its `event_type` precision were a measurement.

### Phase 5 — the run: RAN 2026-09-16, and the answer is NOT YET

**The mechanism is confirmed. The A/B is underpowered and the treatment overcorrected.**

GATE 2 PASSED and discriminates cleanly — cumulative over 601 training batches:

| | absent available | selected into the pair loss |
|---|---:|---:|
| control | 2,683 | 2,649 |
| treatment | **23,421** | **13,341** |

8.7x more absent queries. The treatment applied. (The control is not zero because some absence
arises naturally when a gold surface fails to align — the gate still separates the arms
unambiguously.)

**GATE 1 CANNOT WORK AS BUILT.** With `num_workers: 2`, `__getitem__` runs in FORKED WORKER
PROCESSES, so the injector's counters increment in the worker's copy and the parent's
`composition_line()` always reads `0/0`. Gate 2 lives in the training process, which is why it
works. Either aggregate across workers or drop the line; do not "fix" it by moving it again.

**THE RESULT: the treatment stopped emitting.** Acceptance metric (absent-type firing, casie,
full menu):

| | precision | recall | predictions |
|---|---:|---:|---:|
| control | 0.3333 | 0.2857 | 6 |
| treatment | **0.0000** | **0.0000** | **0** |

Zero predictions is the failure mode this plan pre-registered: *"an overcorrecting abstention
gate is the failure mode on the other side, and a gate that admits nothing has a perfect FP
rate."* The blind test agrees — `event_type` recall 0.1405 → 0.0595, entity F1 0.0205 → 0.0119.

**BUT THE A/B IS UNDERPOWERED AND THIS IS NOT A VERDICT ON THE MECHANISM.** Both arms are
barely trained: 2 epochs over ~1,400 records leaves relation, trigger and argument F1 at
exactly 0.0000 in BOTH arms, and entity F1 at 0.02. Comparing 0.0205 against 0.0119 on models
that extract almost nothing measures which one is closer to silent, not which one discriminates
better. A treatment that suppresses an already-near-silent model is the expected result at this
scale.

**THROUGHPUT, measured, and it is the number that prices the real run:** 12.3 → 8.4 samples/s,
**32% slower**. The 16h02m event-base rebuild becomes **~23.5h** (~$47 on an A100) at
`{entities: 2, events: 1, relations: 1}`.

- [x] A/B: control vs negatives-on, one recipe-level variable, same data, same seed.
- [x] Gates read BEFORE any metric — and gate 1's defect is recorded above rather than papered over.
- [ ] **DO NOT launch the A100 on this evidence.** Three cheaper things first:
      1. **Is it calibration, not training?** The treatment now has a TRAINED abstention gate and
         `abstention_threshold` ships at **0.5**. A gate that learned to fire may simply need a
         different operating point — sweep it on the existing treatment checkpoint before
         retraining anything. Cheap, and it is the same lesson as the stage-0 gate that needed
         0.998.
      2. **Lower the dose.** `{entities: 1}` alone, and/or `abstention_loss_weight` below 0.2.
      3. **Re-run the A/B with enough training that both arms are non-degenerate** — more epochs
         or a larger slice — so the comparison measures discrimination rather than silence.

### Phase 6 — documentation
- [x] `METRICS.md` — "The MENU is part of a metric's identity", extending the existing rule
      that a metric is quoted with its type.
- [x] **Events tutorial** — `docs/events_tutorial.md`, written around the measured traps
      rather than the API. Linked from the README.
- [x] `model_card.py` states the menu beside the metric table.
- [x] `EVENT_ARGUMENT_DIAGNOSIS.md` opens by pointing at this plan and says its numbers are
      expected to be superseded.
- [ ] `RESEARCH_PROGRAM.md`, `EVENT_LINE.md`, `TODO.md` — after the A/B result.
- [ ] Supersede §4h/§4i with the measured outcome — after the A/B result.
- [ ] Model cards for anything published.

---

## 5. Gates and acceptance

**Gates, read before any metric** (a gate that cannot fail is not a gate — this project has
inverted a verdict for exactly that):
1. `probe_absent_queries.py` prints **0%** for control and the configured rate for treatment.
2. The in-band log shows `negative_query_ratio` selecting **> 0** queries in treatment.
3. The `[composition]` negatives line differs between arms.

**Acceptance:**
- Primary: absent-type firing **63% → materially lower** (target to be fixed in Phase 0).
- Primary: full-menu `event_type` precision **0.5521 → materially higher**.
- **Guard (the overcorrection side):** gold-menu recall must not collapse. `event_argument`
  relaxed recall and `entity` strict F1 are the tripwires.
- Neutral-or-better on the eight strict heads.

---

## 6. Sequencing

The event-base blind test lands first: model report, then the comparable 18,786-record eval
through `config/ab/eventrecords-ep1-eval.yaml`. Those results are **expected to be superseded
by this work** and should be written up saying so. Then Phase 1 onward, in order.
