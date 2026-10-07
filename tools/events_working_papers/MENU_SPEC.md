# Menus: score, calibrate and train against the question production asks

**Status:** SPEC 2026-10-07, not built. Decisions so far are the user's (section 3). This is
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

**Proposed here, awaiting the user:**
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
