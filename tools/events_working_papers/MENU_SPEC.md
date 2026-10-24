# Menus: score, calibrate and train against the question production asks

**Status:** STEP 1 BUILT and STEP 3 (dose A/B) RUN 2026-10-07 -- POSITIVE (section 8). Step 2 skipped by the user (the control arm is the baseline). Was SPEC 2026-10-07. Decisions so far are the user's (section 3). This is
TODO #23 (full menu over-fires) turned into a plan. The plan is cheapest-first: step 2 is free,
and the GPU A/B (step 3) runs only if step 2 shows a material gap.

## 1. Why (measured or read from code, 2026-10-07)

A MENU is the list of labels offered to the model for one document. Production never knows which
labels a document holds. If it did, it would not need the model. Every stage today offers a menu
built from the document's own gold:

| stage | menu today | consequence |
|---|---|---|
| training | gold **+ 1 absent label** per entity/event dimension (`negative_labels_per_dim: {entities: 1, events: 1}`), drawn from per-corpus negative pools | the model never practises rejecting many absent types |
| checkpoint selection | gold | selection rewards recall; precision under a real menu is never seen |
| threshold sweep | gold | eb19 picked 0.5 for a question production never asks |
| coref calibration | gold (`coref_calibration._predict`) | the merge threshold is set without the false triggers a real menu produces |
| blind test | gold | `event_type` precision is pinned at 1.0000, so its F1 is recall alone |
| `eval.py --full-menu` | gold + up to **20** absent per dimension, separate `eval_fullmenu_*` keys | not the application's menu |
| production (viewer default) | the whole shipped schema: 320 event types, ~860 entity types | one 743-char CASIE article fired **225 types / 773 instances**; a one-type menu fired 0 |

**Why "full" cannot mean the union.** Gold is exhaustive only for its own corpus's taxonomy. A CASIE document
was never annotated for MAVEN types, so a MAVEN type fired on it cannot be scored as right or wrong. A
union menu would count unannotated truths as false positives. The negative pools already encode this for
training: within-corpus, within-dimension, and partial corpora refused as a negative source
(`tools/data/build_negative_pools.py`).

**Template sizes** (per-corpus pools, an eb17 pool file; eb21's is rebuilt at launch):

| dimension | corpus: pool size |
|---|---|
| events | CASIE 5, DuEE 65, MAVEN 168 |
| entities | docfee 30, docee 352, paraloq 1,349 |
| relations | sentence_rex 815 |

## 2. Design

**Menu modes**, one vocabulary for every stage:

| mode | menu per document | scorable on |
|---|---|---|
| `gold` | its own gold labels (today) | every corpus |
| `widened:K` | gold + K absent labels from its corpus pool (deterministic per document) | every corpus |
| `corpus_full` | gold + its corpus's whole pool (capped, section 4) | every corpus, within its own taxonomy |
| `app:<menu file>` | the APPLICATION's curated menu, the same for every document | ONLY corpora declared exhaustive for that menu |

**Application menus are curated, one per application (user, 2026-10-07).** A menu file is a schema (entities
list, events `{type: [roles]}`) plus `exhaustive_for:`, the corpora annotated against exactly this menu.
- **English news:** the 55-type sonnet55 ontology (`schema_spec.EVENT_DEFINITIONS`). It is exhaustive for
  `cc_news_events_sonnet55_v2`, whose val/test are 2-of-3 voted.
- **EKF disaster pipeline:** its casualty/disaster menu, exhaustive for its own corpora (casualty_docee).

An `app:` mode refuses any record from a corpus outside `exhaustive_for`. It never scores against gold that
was not asked the menu's question. The menu reaches the model through the checkpoint's own `label_map` and
`label_style` (`apply_label_map`), so v1 and v3 models are both asked in their own spellings.

**Stage by stage:**
- **Blind test:** ALWAYS emits the `gold` keys (every published number is gold-menu; they stay comparable)
  PLUS the configured mode under its own prefix (`eval_menu_<mode>_*`). The non-gold figure is the headline
  going forward.
- **Threshold sweep and coref calibration:** run under the configured mode. For news production this is
  `app:` on sonnet55 val.
- **Checkpoint selection:** `widened:K` (K ~ 20-50), a tractable proxy that makes precision real without
  scoring a whole taxonomy every epoch.
- **`score_predictions.py`:** scores an `infer.py` file under the menu it was made with, and refuses `app:`
  rows from non-exhaustive corpora.
- **Config:** `eval.menu` (default `gold`, so nothing changes until asked), `eval.selection_menu`,
  `eval.calibration_menu`, and `eval.app_menus: {name: path}`.

## 3. Decisions

**Made by the user, 2026-10-07:**
- Calibration and the blind test must be harder than a gold menu, because production has no gold menu.
- Production menus are CURATED PER APPLICATION, not the 320-type union.
- Cheapest-first: build the metric, measure, and run the A/B only if the gap is material.

**Proposed here, awaiting the user** (2026-10-07; decided since -- the vocabulary is built (s7), dose 20 shipped in eb21 after the s8 A/B, the news55 app menu is eb21's and eb22's eval menu, and `widened:K` is a scoring mode, not the selection metric):
- The mode vocabulary.
- Always keeping the gold keys.
- `widened:K` for checkpoint selection.
- The section 4 dose arms.

## 4. Training: menu dose (step 3, only if step 2 shows a gap)

"Merge gold into the corpus template" is the existing negatives machinery at a higher DOSE: today 1 absent
label per dimension; the template is the whole pool.

**Constraints:**
- **Window budget.** Menu labels are encoded as text beside the document. 168 MAVEN types plus their roles
  take a real slice of the 4,096-token window, and paraloq's 1,349 entity labels would not fit at all. Cap
  the menu (e.g. 64 labels per dimension).
- **Menu size varies in production.** Draw K per example, log-uniform from 1 to the cap, so small application
  menus and large ones are both trained, and the model does not learn a fixed menu.
- **Hard negatives first within K.** Confusable types (`Conflict.Attack` / `Conflict.Demonstrate`), drawn from
  the label confusions eval already records. Random absents are mostly easy. Hard role negatives were the one
  negatives lever that clearly paid off for arguments (+0.023 strict F1, p3arg).
- **Label noise grows with dose.** Each absent label asserts the type is NOT present. LLM gold misses
  peripheral events (Haiku vs Sonnet events F1 ~0.40; peripheral-event selection is 12.3% of disagreement),
  so low-agreement corpora get a low dose or are declared partial for events. sonnet55 (single-run ~0.80) is
  the safest full-template corpus.
- **Loss balance.** Hundreds of absent queries against a few positives shift the loss toward rejection.
  Price it with `measure_absent_dilution.py` and set `absent_loss_weight` before the run.

**Arms** (warm from eb19, p5link design: one shared 30k draw, one pass, fixed threshold; labels stay eb19's,
so the dose is the ONLY difference):
- `dose1` (control), run twice for the noise floor;
- `logK64+hard`: K log-uniform 1-64, hard negatives first;
- `full64`: the corpus template capped at 64.

**Readouts:**
- the `app:news55` sonnet55 test, as the DECISION metric;
- the gold-menu blind test, for the recall cost;
- throughput and peak memory.

Lambda ~$20-25, or free at the user's office.

## 5. Gates (each must be able to fail)

1. **Off is unchanged:** with `eval.menu: gold` (the default), every metric is byte-identical to today on the
   same checkpoint and records.
2. **The menu is what it claims:** a printed per-mode line with menu sizes (min / median / max per dimension)
   from inside the eval, after logging exists.
3. **`app:` refuses non-exhaustive corpora:** a CASIE record under `app:news55` is refused, not scored.
4. **The mode can fail:** under a menu that includes absent types, `event_type` precision is a measurement --
   below 1.0000 on eb19 -- not an identity.
5. **Dose applied (step 3):** a per-arm in-run line counting the absent labels actually offered (mean K per
   dimension), emitted where the menu is built, not where it is configured.

## 6. Build order (trace, code, trace, test)

1. **Trace** today's eval on real sonnet55 val docs, printing the menu offered per document.
2. **Code** the modes in `eval_metrics` (menu builder beside `_schema_from_gold`), the `eval.*` config keys,
   the blind test's second key set, and `score_predictions.py --menu`. Then gates 1-4.
3. **Measure (free, local CPU):** eb19 on sonnet55 test under `gold` and under `app:news55`, plus the threshold
   sweep under each.
   - **Material gap** (event precision falls, or the chosen threshold moves): step 4.
   - **No gap:** stop. The training change is not needed for news.
4. **The dose A/B** (section 4), with gate 5.

## 7. Step 1 results, 2026-10-07

**Built:**
- **`eval_metrics`:**
  - `parse_menu`;
  - `widen_from_pool`, which offers every dimension the CORPUS annotates, so an event-free document is
    asked about events (the old `_widen_with_absent`, kept unchanged for `--full-menu`, widens only
    dimensions the gold already has);
  - `project_gold` (app gold restricted to the menu's labels and roles);
  - `build_menus` (app menus refuse non-exhaustive corpora);
  - `menu_sizes`;
  - `compute_metrics(menus=)`, with an in-run line of the menu sizes offered;
  - `sweep_thresholds(menus=)`.
- **`train.py`:**
  - `corpus_pools`: per-corpus templates from transformed TRAIN records; role-less event types are kept
    (MAVEN), partial dimensions are skipped;
  - `load_app_menu`: mapped through the run's labels config, style included;
  - `menu_split`: an app menu reads its OWN exhaustive corpora's split, because they are often train_only
    in the config (sonnet55 is);
  - `score_under_menu`;
  - a `check_menus` startup refusal.
- **Config keys:**
  - `eval.menu`: an extra blind-test pass under `by_menu`, with the gold keys untouched;
  - `eval.calibration_menu`: the threshold sweep AND the coref calibration;
  - `eval.app_menus`.
- **CLIs:** `eval.py --menu`; `score_predictions.py --menu app:<file> --corpus --labels-file`;
  `infer.py --schema-json` accepts a menu file.
- **Menu file:** `config/menus/news55.json` -- 125 entity types + 55 event types with roles, exactly the menu
  `ccnews_english_v2.yaml` offered every document. Relations, classifications and structures were sampled
  per document, so they are not exhaustive and are left out.
  - Exhaustiveness checked: all 9,939 entity, 2,944 event and 5,385 argument gold uses in sonnet55 val/test
    fall inside the menu.

**Gates:**

| gate | result |
|---|---|
| 1 off unchanged | `compute_metrics` without menus: the 40-record 426-key output is BYTE-IDENTICAL to before |
| 2 menu is what it claims | in-run `[menu] offered per document (min, median, max)` line; app:news55 prints 125 / 55 for every document |
| 3 app refuses | `score_predictions --menu app:news55 --corpus casie` refuses; `build_menus` refuses and counts non-exhaustive records |
| 4 mode can fail | under app:news55, `event_type` precision is a measurement (0.088 on 12 docs), not 1.0000 |
| equivalence | on the same 12 sonnet55 test docs, the eval path (`menu_split` + `compute_metrics`) and the file path (`infer.py --schema-json` + `score_predictions --menu`) agree on all 330 keys |
| end to end | a tiny real `train.py` run with `eval.menu` + `eval.calibration_menu`: startup check, a threshold sweep under the app menu, and a `by_menu` blind-test block beside intact gold keys |

`menu_split` on eb21's real config: `app:news55` scores sonnet55's 500 test docs; `widened:20` and
`corpus_full` score exactly 20,602 test records (= the blind test's count), offering up to 20 / 64 absents.

**A first look (p5link, an older model, 12 sonnet55 test docs, threshold 0.3) -- NOT step 2:**

| head | precision, gold menu | precision, app:news55 |
|---|---|---|
| entity | 0.657 | 0.0055 |
| event_type | 1.000 (pinned) | 0.088 |
| event_trigger | 0.474 | 0.049 |

On one 2,286-char document with 21 gold entity mentions, 3,141 entity mentions were predicted across 96
types: `60` got 82 types and `Timothy Fiore` 71. The spellings agree (every gold label is among the
predicted keys), so this is over-firing, not a measurement artefact. Step 2 measures eb19.

**Deferred:** `eval.selection_menu` (per-epoch checkpoint selection). The trainer's eval is WINDOWED -- records
become chunks -- so per-record menus (and the corpus each needs) must be carried through chunking first.
It is a separate change.

**eb21:** carries `eval.app_menus` and `eval.menu: app:news55` (an extra `by_menu` blind-test pass only). It
does NOT set `calibration_menu`: that changes the operating point, and step 2 decides it.

## 8. Step 3 result: the dose A/B, 2026-10-07

Warm from eb19, one pass over `cc_news_events_sonnet55_v2` train, a fixed threshold of 0.5, and a single
control (user). The treatment was proven in the loss: 96,310 absent queries available over 200 batches
against the control's 14,011.

**Strict P / R / F1 on the 500 voted test docs, control -> treatment:**

| head | NEWS menu (production) | GOLD menu |
|---|---|---|
| event_type | P 0.118 -> **0.917**, R 0.783 -> 0.409, F1 0.205 -> **0.566** | F1 0.798 -> 0.785 |
| event_trigger | P 0.096 -> **0.712**, R 0.457 -> 0.304, F1 0.158 -> **0.426** | F1 0.521 -> 0.511 |
| entity | P 0.322 -> **0.553**, R 0.493 -> 0.425, F1 0.389 -> **0.481** | F1 0.476 -> 0.533 |
| event_cluster | F1 0.142 -> 0.373 | F1 0.425 -> 0.428 |
| event_argument | P 0.073 -> 0.307, R 0.124 -> 0.054, F1 0.092 -> 0.092 | F1 0.102 -> 0.121 |

**Reading:**
- The training dose is the over-firing lever. The news-menu gains are 10-20x the +/-0.02 single-run floor.
  There is no cost under the gold menu, and entity even improves.
- Recall falls under the news menu at a threshold chosen for an over-firing model. **Next:** re-pick the
  threshold with `eval.calibration_menu: app:news55` on sonnet55 val, then score the test once.
- Arguments: precision up, recall down, F1 flat.
- The pair loss samples a capped number of absent queries (~6,460 in both arms), so the extra negatives act
  mainly through the boundary loss.
- Still open (section 4 build):
  - event-free documents get no event negatives in either arm;
  - the menu size is fixed at 20, not variable;
  - there is no per-document cap;
  - there are no hard negatives.

  The next base can take the dose as is, and the rest can be A/B'd on top.


## 9. After the dose A/B, 2026-10-07 to 2026-10-14

Details and costs are in `EXPERIMENT_CATALOG.md`. All numbers are strict micro F1 on the voted sonnet55 test
(500 docs), single runs, floor +/-0.02.

**Thresholds (calibrated under app:news55 on val).** Dose 20 is already best at 0.5. The dose-1 control is best
at 0.7, and calibrated it still loses: news event_type 0.345 vs 0.566, trigger 0.264 vs 0.426. The dose wins on
its own, not through a threshold.

**Section 4 refinements: rejected.** Gold-free event absents and the full recipe (log-uniform K, per-record
cap) were each scored at their own best threshold. Under the news menu they lose to plain dose 20 on entity and
arguments (full: entity 0.311 vs 0.480, argument 0.035 vs 0.092). Under the gold menu the full recipe was best.
The size ladder below locates that flip: the full recipe leads at every widened size up to corpus_full (the
corpus's own 74 entity / 55 event types) and loses only under news55's 125 curated entity types.

**The entity position leak (fixed be581a1).** Every widened ladder rung collapsed entities (widened:20 entity
F1 0.000). Root cause: the injector appended absent entity labels after the gold, and `shuffle_entities`
defaulted False, so in training the gold entity labels were always first; synthetic `entity 1` was gold 100% of
the time. The same model on the same 28 labels scores 0.000 absent-first, 0.646 gold-first, 0.231 shuffled.
Fix: shuffle before synthetic naming, default True. Every entity score from a negatives-trained model before
be581a1 carries the shortcut. Event types, relations and structures were never affected: each label is its own
task, and task order is shuffled. Measured at inference with `tools/train/probe_label_order.py`: event_type
0.847 / 0.847 / 0.853, relations 0.059 / 0.061 / 0.057, post-fix entities 0.622 / 0.613 / 0.604.

**Negatives never resampled across epochs (fixed 0d4f248).** Persistent forked DataLoader workers never saw
`set_epoch`, so every multi-epoch run before 2026-10-08 showed the same absent labels every epoch. The epoch and
per-epoch counters are now shared memory, traced 1 / 5 / 20 in real forked workers; epoch 0 is byte-identical to
before.

**Dose over epochs (3 epochs from eb19, threshold 0.5).** `negative_labels_schedule` sets a per-epoch dose.
`training.abstention_phase: {epochs, main_dose}` (off by default) trains the main phase at `main_dose`, then
runs the phase at the configured dose from the main phase's `best/`, with a fresh optimizer and LR.

| arm | dose per epoch | news event_type | news argument | gold entity | gold event_type |
|---|---|---|---|---|---|
| const20 | 20, 20, 20 | 0.621 | 0.100 | **0.575** | **0.826** |
| last | 1, 1, 20 | 0.626 | 0.114 | 0.507 | 0.776 |
| abst | 1, 1, 1 + phase 20 | 0.634 | 0.124 | 0.511 | 0.790 |
| abst5 | 5, 5, 5 + phase 20 | **0.656** | **0.136** | 0.530 | 0.812 |

ramp (1, 5, 20) is confounded: head_min selection (~0.01, noise) shipped its dose-5 epoch.

- A final dose-20 stage reaches const20's news-menu precision from a low dose.
- Gold-menu recall is learned during the main run at a dose above 1. abst5 keeps the event heads inside the
  floor and is the best arm on events: news event_type and argument each gain +0.035, about 1.75x the floor.
- Entity recall under the gold menu still costs abst5 -0.045.

**Decision for eb21:** dose 20 throughout, unchanged. abst5's event result needs a second seed before it can
change the recipe.

## 10. Common menus and menu poison, 2026-10-17

Details, numbers and costs are in `EXPERIMENT_CATALOG.md`, 2026-10-17.

**Common inference menus.** `tools/train/build_schema.py` builds one app menu from corpora's train gold
(`config/schemas/casie.json`, `wikievents.json`). It reports val/test labels the menu does not offer.
`infer.py --schema-json` runs every model on that one menu, overriding each checkpoint's `default_schema`. eb21's
`default_schema` ships no entity labels: more than 1,000 go to `open_vocab`, unlisted. Score with
`score_predictions --menu app:<file> --corpus <name>`.

**The checkpoint's full menu cannot run yet.** Its 598 relations make the relation decode allocate an int64
[B, R, Q*K] tensor of 5.47 GiB at batch 1, which runs a 40 GB A100 out of memory. The tensor grows with the
menu, not the batch. CPU runs it at ~130 s/doc and 14 GB, about 100 h per arm for the blind test. The fix is to
chunk over R in `TypedRelationPairGenerator.select` (TODO #37). **Fixed a27cc6b:** the relation-independent
sorts now run once and each relation takes a blocked `topk`; output byte-identical, selection 62.0 -> 1.59 s and
+13.3 -> +0.8 GB on a real full-menu doc, which now takes 59.6 s on CPU instead of 122.8 s (the encoder now dominates).

**Menu poison.**
- Under CASIE's own menu, eb21 and both corefC arms fire no CASIE events: 1 hit vs 37 under the gold menu, on
  40 test docs.
- One role, `Damage_Amount`, causes it. Without it the corpus menu beats the gold menu (48 hits).
- `tools/train/menu_poison_check.py` finds such labels on VAL: drop-one removal, then greedy confirmation.
- `build_schema.py --drop` removes them.

**The mechanism** (`tools/train/probe_poison_mechanism.py`, CASIE val):
- The role acts through the trigger QUERY of the event type that lists it. Text states carry nothing.
- Other types are untouched, and span ranks are kept while scores fall 10-100x.
- The cue is the token `▁Damage`, which reaches eb21 only through this role.
- **Cause: training lists roles asymmetrically.**
  - A gold event type lists only the roles argued in its document (`processor.py`).
  - An injected absent type lists its full pool role list (`negatives.py`).
  - Damage_Amount was listed 26 times on gold types vs 1,543 on absent ones.
- **The dose amplifies it.** The menudose arms share eb19's warm start and data and differ only in dose. In
  their doc-1 trigger p for Ransom, the role cuts 0.696 to 0.314 at dose 1, and 0.271 to 0.001 at dose 20.
- Section 8's dose gain therefore came with a menu-shaped cost that no gold-menu eval could see. Any rarely
  argued role can poison a production menu.

**The fix, under test.**
- `training.negative_gold_full_roles: true` lists each gold type's full pool role set; unargued roles get
  empty targets. Gold and absent types then list the same roles.
- Listing gate: roles more than 0.05 above the absent base rate fall from 20 to 2. Damage_Amount's absent share
  falls from 0.983 to 0.645.
- **Result, rolesym A/B, 2026-10-18 (POSITIVE):** news55 (test, once per arm) event_type F1 0.153 -> 0.541 with precision UP (0.414 -> 0.460), trigger 0.069 -> 0.397, argument 0.007 -> 0.115; CASIE val full menu 2 -> 72 hits (gold menu 60 -> 47); WikiEvents dev full menu 48 -> 97 hits but 3.5x the firing; blind-test heads all inside the floor. Gold-menu prompts (argued roles only) are now out of
  distribution for the treatment, so gold-menu evals understate it. EXPERIMENT_CATALOG has the numbers.
- The rolesym A/B (`config/ab/rolesym-*.yaml`) differs only in this flag. Reads were pre-registered on VAL:
  1. poison check under the original full CASIE menu, then WikiEvents and RAMS;
  2. app:news55 over-firing, the companion, since the role list may have been doing some of the dose's work;
  3. eb21's heads against the floor.
