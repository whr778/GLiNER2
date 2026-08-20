# Open items — resume list

Completed work is removed rather than struck through; history lives in
`PROJECT_JOURNAL.md` and the commit log. Everything below is a defect with evidence
attached, or a decision with a stated next test.

State at 2026-08-15 close. **One 2xH100 live** (`clean-rebaseline-2`): the 137k
re-baseline on decontaminated data is done, and four warm-start arms are running (two
control seeds for a fresh noise floor, then `evwide2`/`evwide4`). Items 11 and 12 are
closed; the phase-3 loss work has a result and a redirect.

**Read this before quoting any historical number.** The blind test leaked.
`SplitWriter` drew one random per ROW, so a document emitted more than once scattered
across train/val/test — **1,080 documents, 7.03% of the cold-start blind test, were in
train**. Fixed (grouped splits are now the default, `check_leakage.py --config` gates a
config, and `train.py` repairs before every run); contamination is down to 21 documents.
The 137k reference `entity 0.586 / relation 0.170 / event_type 0.956 /
event_argument 0.098` is **superseded** by `0.6306 / 0.1573 / 0.9365 / 0.1014`, and the
delta is not a contamination estimate because the test was also recomposed.

**Closed since 12 Aug.**

- **Item 11 (GIST query-axis veto) — DONE, negative.** Negative on 7 of 8 metrics.
  Entity −0.025 survives the measured noise floor; relation −0.033 does **not**.
- **Item 12 (base-word arms) — the provisional +0.0119 remains provisional**, and now
  also rests on a contaminated blind test.
- **Phase-3 event loss — the flat weight is a null lever, for a mechanical reason.**
  `task_loss_weights` reaches only start/end/pair, 18.5% of the loss, so `w=4` moved
  events 6.6% → 10.6% of the gradient. `task_loss_weight_scope: all` raises reach to
  94.3%; that is what the live arms test. `pos_weight` belongs in the **cold-start**
  rebuild, not here: event positive fraction is 0.052 at init but 0.562 at convergence.
- **A measured noise floor exists.** Two seeds of one control on `mix_natural`:
  relation strict **±0.041**, event_type ±0.014, event_trigger ±0.013, entity fair
  ±0.010, event strict ±0.008. Anything smaller is unreadable on one seed. Being
  re-measured on the rebuilt mixture.

**New open items from 15 Aug.**

- **Every one-seed curve in this project is now suspect.** A control re-run of the
  published RAMS 137K recipe scored **+0.023** above it, so single-run variance on that
  metric is >=±0.02. The head-init boundary curve (0.177/0.191/0.202/0.192) is **flat
  within noise** and its shape must not be cited; the "turns at 100K" reading was
  retracted the same day it was made. The span-vs-boundary verdict survives (3.5× at 10K).
  **Any curve claim needs >=2 seeds per point before it is quoted.**
- **`scope: all` is a real lever and wants a proper sweep.** +0.013 event strict at both
  w=2 and w=4, above the floor, entities unharmed; cost is event_type −0.019 at w=4.
  A dose sweep with 2 seeds per point would locate the knee.
- **An intermediate `mix_natural` stage is a wash on RAMS** (arguments span 0.005 across
  three arms), which also means warm-starting through a broad multi-task corpus does NOT
  cost event capability. `event_type` +0.021/+0.018 in both treatment arms is the one
  delta worth a second seed.

- **Four converters still split row-wise** (`docee`, `docfee`, `cmnee`, `mendeley_ed`) —
  no `SplitWriter`. 21 residual contaminated documents, gate-removed each run, unfixed
  at source.
- **`data/scaling_joint/` val files are frozen from 8 Aug**, pre-fix. Rebuilding also
  rebuilds the j10k/j40k/j100k slices the scaling curve rests on — deferred.
- **Structures are never scored by the blind test.** `_schema_from_gold` builds no schema
  for `json_structures`; structure-only records are skipped — 35.1% of `mix_natural`'s
  val, 97.3% of the reframed text2json's. Use `probe_records.py`.
- **The wider contaminated corpora are unregenerated**: gliclass_logic 38%,
  knowledgator_gliner 27%, klue_re 17%, finer_ord 14%, MasakhaNER 12-14%, nuner_full,
  pubmed_abstracts_ner. Not in a live config, so not blocking.
- **`rams` has 13.7% duplicate documents inside train alone** — a different problem from
  cross-split leakage, and RAMS is a key downstream.

**Where the line stands.** The scope gate took per-state error 5.247 → 0.591 (item 10) and
largely closed the attachment blocker. Two candidate next steps were then *closed by
measurement rather than argument*:

- **MHT is not the bottleneck — but the number that said so was wrong, and is now corrected.**
  Perfect *two-way* association is worth +0.055; that oracle cannot reject, so it never priced
  a null hypothesis. With a reject option the ceiling is **+0.111 (18.8%)**. The cheapest
  piece that delivers one — M5 track birth by innovation gating — was then **built and lost**
  (0.608 against the magnitude gate's 0.591), because judging a stream against its own track
  is circular. `PIPELINES.md` §4/§4.1, `tools/ekf_showcase/mht_associate.py`. Item 6.
- **Extraction recall is not the bottleneck either**, and the claim that it was rested on a
  stale count from a superseded run. `extract_long` had already fixed it **4.2x**.

The live bottleneck is **cross-event contamination** (item 2, quantified at 4.7% and
resistant to all three signals tried) and underneath it the query-axis training gap that
GIST was built for (item 11 — **RAN 2026-08-14 and it is NEGATIVE**).

**Update 2026-08-20 — item 2's data-side fix RAN and is SUPERSEDED.** The muting arm trained
and the suppression is real, but a declared per-event **plausibility ceiling** — one
threshold, no model — beats it outright, and the large false positives it removed were never
other storms' tolls: populations, insurance policies, power crews, years. Both genuine
cross-event figures survive both mechanisms. So **cross-event is still the live bottleneck,
now with two mechanisms measured against it and neither touching it.** What did change:
**item 3 (non-casualty numbers) is promoted** — at 3.8% of observations it carries the largest
values and dominates nRMSE, and it splits into classes needing different fixes (entity typing
reaches insurance policies and churches; it does not reach 1,500 troops or 8,000 power crews,
which are living people and need casualty-role semantics).

**Why events and not relations, when a fix is proposed.** The programme's priority is event
extraction; NER, relations and structures are carried along with it. That is why the losses
were separated in the first place — lumping structures, relations and events into one loss
made the event signal invisible, and phase 2 split them (re-implemented for the boundary head
in [[EVENT_LOSS_PHASE3_PLAN]]). So when a defect admits both a relation-shaped and an
event-shaped fix, the event-shaped one is the one that serves the programme *and* the one
that carries the right information: only the event formulation has an `event_key`, which is
the field an EKF observation needs. See item 1 for the worked case.

### FRONT-END REBUILD IN FLIGHT — `ekf-frontend-mmbert` (2026-08-20)

The EKF pipeline's router work is blocked on an extractor that emits, per event, a trigger
and arguments bound to *that* trigger. Nothing in the line does.

**Span arch** emits a bag of triggers, all sharing one role; no threshold fixes it — the
Katrina block is either the bare name (missing its own 1,400) or swallows Helene.
**Boundary `137k-clean`** yields nothing above threshold 0.3 on English disaster copy and
nonsense at 0.1, despite trigger 0.710 / argument 0.506 on its own test set.

**The arithmetic behind it.** Counting only corpora that bind arguments to a trigger — DocEE,
ChFinAnn and DocFEE do *not*, they are `entities` + `classifications`:

| | English | Chinese | English share |
|---|--:|--:|--:|
| `137k-clean` as built | **798** (CASIE alone) | 20,884 | **3.7%** |
| every available corpus | 39,783 | 20,884 | 65.6% |

MAVEN and Mendeley are trigger-only. So argument F1 0.506 is very nearly a Chinese-only
number and English trigger→argument rests on 798 examples.

`tools/train/config/ekf-frontend-mmbert.yaml` — cold start, 189,284 records, 50× the English
trigger→argument supervision, Chinese kept. Split gate CLEAN (180,660 / 11,486 / 20,571).
`rams` gained val+test by carving its 871-row test **by document**, which found 101 duplicate
rows in test alone — the same hazard this file records for rams train.

**Declared risk:** 72% of the new English trigger+argument data is synthetic against 20%
human-annotated real news, on a line whose recurring failure is in-domain-good /
real-news-zero. Gates are on AP prose for that reason, not held-out DocEE.

**Smoke (1× A100-40GB):** 18.4 samples/s, 34% faster than the extrapolation every earlier cost
estimate used. `num_workers` is NOT the bottleneck — 18.4 at 0 workers, 18.5 at 4 — so do not
re-try it; utilisation swings 28–81% on variable sequence length while memory stays flat at
10.6 GB of 40. **batch_size is the untested lever.**

---

State at **2026-08-20**. **A smoke GPU is running** (self-terminating watchdog on the box) (the muting-arm A10 is terminated, verified).
A programme-wide caveat landed with it: the cached Helene observation set behind every
published Helene figure **cannot be regenerated from any committed state** — see
`tools/ekf_showcase/muting_arm_results/PROVENANCE.md`. Comparisons among the published
numbers stand; placing a new model on their scale does not. Earlier state at **2026-08-17**: Two eval-side defects fixed on 08-14 (`c0ab89c`,
`7586411`); see "What the metrics fixes did and did not touch" below before re-reading any
number in this file. A third is now open and unfixed — **item 0**, the cross-event probe's
scoring — and it invalidates the readout item 2 would be measured with. Nothing is
mid-flight.

---

## P0 — blocks the next experiment

### 0. CLOSED 2026-08-19 -- the record head was fine; `runtime.py` dropped `record_metadata`
The five zeros were one line of plumbing. `runtime.py` rebuilt each schema through
`Schema.from_dict(...).build()` (which produces `record_metadata`) and then copied only
`json_structures` and `json_descriptions` out. With no metadata `compile_record_specs`
returns `{}`, no `RecordSpec` compiles, the head decodes nothing, and **nothing raises**.
Fixed in `d754132`, regression-tested in `tests/test_record_metadata_roundtrip.py`.

Re-scored from the same checkpoints, the head shows an ordinary monotone curve:
0.0238 / 0.0552 / 0.1043 / **0.1119** strict F1 at 10k / 40k / 100k / 137k. Full write-up,
including why the four points are not yet on one test set, in `JOINT_IE_SCALING.md` §0c.

The "+45% structure supervision" in the warm-start was real in record count and **zero in
effect**: those 3,494 cc_news/synthetic records carry `json_structures` with no
`record_metadata`, so they supervise the record head with nothing. See item 0.1 -- that is the part still open.

### 0.1. cc_news and synthetic converters emit `json_structures` with NO `record_metadata`
**Successor to item 0, and it needs a data-design decision, not a patch.** On the training
path `Structure.get_record_metadata()` returns `None` unless `mode` is set, so a structure
record without metadata produces no `RecordSpec` and no record targets. Counted across the
warm-start mix:

| source | structure records | reaching the record head |
|---|--:|--:|
| cc_news_haiku45 | 1,033 | 0 |
| synthetic_haiku45_5k | 996 | 0 |
| synthetic_sonnet5_1k | 1,465 | 0 |
| replay_137k30 | 519 | 519 |
| *137k base pool (text2json)* | *7,754* | *7,754* |

So the warm-start **cut** record-head supervision from 7,754 to 519 (-93%) while appearing
to add structure capability, and still only lost 0.0059 -- read that as replay working, not
as the head failing.

**Any future arm claiming structure capability from these corpora will add none.** Note the
2026-08-19 builder default (item 0.3) does **not** fix this: it changed the inference/eval
path, while training reads `record_metadata` out of the corpus through
`Structure.get_record_metadata()`, which still returns `None` without a stored `mode`. The
symmetric change on the training side is exactly the decision below, and it is bigger than
the inference one because it alters what every future run learns rather than what a decode
emits. Fixing it means assigning `mode` and `anchor` per structure type in the converters. `natural` +
first-field anchor is the obvious default but is a real modelling choice about what anchors
a `person_profile` or a `transaction`, so it wants a decision before it is written.

### 0.3. CLOSED 2026-08-19 -- the fluent builder now emits `record_metadata` by default
Item 0 fixed `runtime.py` dropping the key. This was a different drop, upstream of it:
the fluent builder never produced the key at all unless the caller passed `mode=`, so
anyone following the obvious API got an empty extraction and no error. What it looked
like:

    Schema().structure("casualty_report").field("dead", dtype="str")
      -> build()["record_metadata"] is None        # head decodes NOTHING, silently

    Schema().structure("casualty_report", mode="natural", anchor="dead") \
            .field("dead", dtype="str", cardinality="required_one")
      -> {"casualty_report": {"mode": "natural", "anchor": "dead", ...}}

**Fixed.** `StructureBuilder._auto_finish` now defaults to `mode="natural"` anchored on the
first declared field -- the same choice `_store_record_metadata` already made for a caller
who set mode and omitted anchor, so this is not a new convention. A structure with no
declared fields still emits nothing (no anchor is possible), and `mode="latent"` is
unchanged. Verified end-to-end on the 137k checkpoint: plain and declared forms return
identical records where plain previously returned `None`.

**Scope of the change: inference and eval only.** It does NOT touch training supervision --
see item 0.1, which stays open. One behaviour change to expect: `Schema.from_dict` defaults
too, so `_schema_from_gold` now compiles specs for metadata-less gold, and in-loop eval on
corpora like cc_news/synthetic will start reporting nonzero `structure` where it reported
0.0000 before. Published curve numbers are unaffected -- all 148/856-record test sets behind
them are text2json, 100% of which carries explicit metadata.

This also invalidates, in the other direction, the earlier "records return None"
measurements recorded in `JOINT_IE_SCALING.md`: they were taken against a schema that could
never have worked and can now be retaken.

### 0.2. `record_anchor_threshold` defaults to 0.5, which no model can use well
Separate from the decode bug and still live. Nothing calibrates the record cutoffs --
`threshold_sweep` moves the general decision threshold only. At the 0.5 default the models
score 0.0000 / 0.0052 / 0.0654 / 0.0760 versus 0.0238 / 0.0552 / 0.1043 / 0.1119 at their
swept thresholds: a 32-100% loss depending on scale. Either fold the record cutoffs into
`threshold_sweep` or change the default; `tools/train/sweep_record_thresholds.py` is the
sweep in the meantime.


### -1. `eval_metrics.py` CANNOT SCORE `json_structures` — every casualty model is affected

Found 2026-08-17 during the Track A run, and it is the most consequential thing that run
produced. `gliner2/training/eval_metrics.py` builds gold/pred sets for exactly six families:

```
_gold_entity_set  _gold_relation_set  _gold_event_trigger_set
_gold_event_type_set  _gold_event_argument_set  _gold_classification_pairs
```

There is **no structure/record scorer**. `casualty_loc_split` is 100% `json_structures`, so
the evaluator found nothing it could score and the entire 36-minute run emitted **one**
metric key: `eval_loss`. Verified by scanning the log for `eval_[a-z_]+` — one match.

**Consequence, and it is not confined to Track A.** With no F1 available, `metric_for_best`
can only be `eval_loss`, and on this corpus it *latched at epoch 1*: one "New best
eval_loss: 82.1840" and never again, while train loss fell **35.16 → 4.50** over six
epochs. `best/` was therefore the epoch-1 model — confirmed by mtime, written 6 minutes
into a 36-minute run. Measured cost of trusting it (40 Helene windows):

| checkpoint | location filled | event filled |
|---|--:|--:|
| `best/` (epoch 1, what the selector chose) | 43/48 | 17/25 |
| `checkpoint-epoch-6` (what training actually produced) | **51/54** | **30/36** |

Every casualty model ever trained — `casualty_ft`, `casualty_multi`, `casualty_docee`, and
the `casualty-docee` checkpoint the EKF pipeline runs in production — used the same selector
on a corpus the evaluator cannot score. Whether they latched as early is unknown; their logs
are gone. This is the same failure class as the MAVEN Tier 2 "+0.049 win".

**Fix:** add a structure/record scorer emitting `eval_structure_strict_micro_f1`, then
re-select both Track A and its `casualty-docee` baseline on it. Until then, no casualty
checkpoint selection can be trusted, and the readouts in item 0 and item 1 below are the
missing metric computed by hand.

### -1b. CLEAN RE-RUN DONE 2026-08-17 — both arms, bf16 + structure-metric selection

Item 1 and item 3 of the follow-up list, one A100, ~$3.00, instance terminated.
Commit `1b6e0f6`; scorer from `fb456e4`.

**bf16 fixed the crash.** Both arms ran 8/8 epochs with no non-finite loss, where the fp16
arm died at 79%.

**The metric now drives selection — where there is anything to select.** `casualty-docee`
re-selected **six times** (0.9563 → 0.9794 on val). `casualty-loc-split` selected **once,
at epoch 1**, and never improved through epoch 8 while train loss fell 34.65 → 3.42.

That second result is REAL, not another selector artifact — the same recipe on the other
corpus tracked fine, so the metric works. Field extraction on `casualty_loc_split`
saturates after one epoch. **Practical consequence: epochs 2-8 buy nothing measurable
there, so ~85% of that arm's GPU cost is waste.** Cut `num_epochs` before re-running it.

**Item 3 — the published models rescored on the metric nobody could compute before.**
Same 400-record blind slice per corpus:

| model | corpus | strict F1 | relaxed | support |
|---|---|--:|--:|--:|
| published `casualty (ft)` | casualty_ft | 0.9959 | 0.9959 | 610 |
| published `casualty-docee` (production EKF extractor) | casualty_docee | 0.9784 | 0.9784 | 920 |
| **new** `casualty-docee` clean | casualty_docee | **0.9822** | 0.9822 | 920 |
| published `loc-split` (fp16 run) | casualty_loc_split | 0.7693 | 0.9159 | 2,155 |
| **new** `loc-split` clean | casualty_loc_split | **0.8351** | 0.9374 | 2,155 |

**Read this within a corpus, never across one.** `casualty_docee` has three numeric fields;
`casualty_loc_split` adds free-text `location`, which is far harder and carries 2.3x the
support. The 0.98-vs-0.84 gap is task difficulty, not model quality.

Within-corpus, the clean re-runs win on both: **+0.004** on docee and **+0.066** on
loc-split. So the old `eval_loss` selection cost little on the numeric-only corpora — the
production `casualty-docee` extractor was not badly damaged — but cost a lot on the corpus
with a hard field.

**What item 3 could NOT answer:** whether each published model latched early during its own
training. Those per-epoch checkpoints are gone; only a re-run would show it.
`casualty-multievent` was skipped entirely — no local `casualty_multi` test split, and the
rescore script checks the filesystem directly instead of going through `_fetch_if_missing`.

Blind tests: docee **0.9766** (P 0.9655 / R 0.9879, support 4,619); loc-split **0.8485**
strict / 0.9543 relaxed (support 15,187).

Models: `whr778/gliner2-base-v1-casualty-{docee,loc-split}-clean` (private).

### 0c. TRACK B RAN 2026-08-17 — NEGATIVE, and the corpus is the bottleneck, not the formulation

`casualty-events-boundary.yaml`, A100, ~$2.30, terminated. The same documents, splits and
figures as Track A, re-emitted as trigger + typed arguments so the loss reaches the event
path — the formulation `EKF_MHT_BUILD_RECORD.md` §27.2 says is missing.

**The run itself was clean.** 8/8 epochs, zero non-finite losses, FA2 confirmed active (0
sdpa fallbacks — on mmBERT that is correctness, not speed). Selection re-selected
0.3494 → 0.4973, where the Track A structure arm froze after one epoch. So the event
formulation was still learning where the structure one had stopped.

**In-domain it works.** Blind test on 2,852 documents / 14,614 argument instances:

| metric | strict | relaxed |
|---|--:|--:|
| event_argument (the binding) | **0.5320** (P 0.673 / R 0.440) | 0.7521 |
| event_trigger | 0.9610 | 0.9612 |
| event_type | 0.9897 | 0.9897 |
| event (combined) | 0.7566 | 0.8652 |

Trigger and type are near-ceiling and largely **circular**: triggers were derived by
matching a fixed per-type surface list, so the model mostly learns that list back.

**On real news it produces NOTHING.** All 104 Helene windows unbound — with the eight
trained types, and zero-shot with a `Hurricane` type. Verified against a working in-domain
extraction on the same checkpoint in the same session, so this is transfer failure, not a
broken probe. (An earlier reading of "zero" *was* a probe bug — the event type was given as
`casualty_report` instead of the trained DocEE types — and was caught before being believed.)

**This was predicted.** Item 0b, written before the run: *"the boundary base fills ~0 on
real wire copy … any events-form arm trained on a boundary base inherits that domain gap
and will be unmeasurable for the same reason."* It did, and it is.

**So the Track A vs Track B comparison cannot be made.** Track A scores 3/11 @ 9.6% FP on
these windows; Track B scores nothing at all. Taken together with Track A's positive, the
reading is: **the formulation is not the bottleneck — the corpus is.** Synthetic-realized
prose does not transfer to AP wire copy on the boundary/mmBERT path, while the span path
fine-tuned from `fastino/gliner2-base-v1` does. Effort belongs in data, not architecture.

Model: `whr778/gliner2-casualty-events-boundary` (private).

### 0. The cross-event readout was unsound — FIXED 2026-08-17 (`63249ed`), and C is dead

**Fixed and re-measured.** Read this section for what the numbers now are; the diagnosis
below is kept because it explains why the published ones cannot be reused.

Three arms, 104 observations, current code, CPU. `C` is now validated; `C-raw` is the old
unsound scoring, retained as a diagnostic:

| model | arch | A | B | **C** | C-raw | cross-event binding coverage |
|---|---|--:|--:|--:|--:|---|
| `fastino/gliner2-base-v1` | span | 3/11 @ 53.0% | 3/11 @ 50.6% | **3/11 @ 30.1%** | 3/11 @ 36.1% | ours 7, competitor 3, unbound 1 |
| `gliner2-base-v1-casualty-docee` | span | 3/11 @ 19.3% | 3/11 @ 15.7% | **0/11 @ 2.4%** | 7/11 @ 37.3% | **rejected 7**, ours 2, unbound 2 |
| `gliner2-warmstart-natural-clean` | boundary | 6/11 @ 77.1% | 4/11 @ 63.9% | **0/11 @ 0.0%** | 0/11 @ 1.2% | **unbound 11** |

**Three readings, none of them good for signal C:**

1. **The casualty-docee "9/11 win" was 100% artifact** — 0/11 once bindings must name an
   event. Its 7 rejected cross-event cases are `raw='230'`, `raw='1,400'`, `raw='250'`.
2. **The boundary arm binds nothing** — 11/11 unbound. Note it scores `0/11 @ 0.0% FP`,
   which under the old table reads as *perfect precision*. This is exactly why the coverage
   table exists.
3. **The only model that binds sanely gets it confidently wrong.** `fastino` binds *ours*
   (Helene) for **7 of 11** cross-event cases. The failure is not uncertainty that a
   threshold could recover — it asserts the wrong event.

**So C via the structure/record path is dead on every model available.** Best is 3/11 at
30.1% FP, unshippable, which agrees with §27.2's conclusion even though its numbers do not
reproduce. What this does *not* refute is the item-1 event-formulation hypothesis: none of
these three was trained events-form, so the training arm remains untested — but it now has
an honest floor to beat, and a working instrument to beat it with.

**New prerequisite this surfaced.** The boundary base fills fields at ~0 on real wire copy
(separately measured: `location` 1/14, `event` 0/11 over 40 Helene windows, versus 26/33 and
20/21 for the domain-adapted span model). Its record head was trained on synthetic templated
casualty text. Any events-form arm trained on a boundary base inherits that domain gap and
will be unmeasurable for the same reason this arm was — so the conversion has to carry real
contexts, `casualty-docee` style, not just a reformat of `casualty_multi_loc`.

---

**The diagnosis, kept for provenance.** `event_binding_probe.py` scored signal C as:

```python
"C bound event is a competitor": bool(r["bound"]) and not OURS.search(r["bound"])
#  OURS = re.compile(r"\bhelene\b", re.I)
```

C therefore fires on **any non-empty string that does not contain "helene"** — including a
string that is not an event mention at all. Three arms on the same 104 observations, all on
current code, CPU:

| model | arch | C catches | C false pos | what `bound` actually holds |
|---|---|--:|--:|---|
| `fastino/gliner2-base-v1` (§27.2's model) | span | 3/11 | 44.6% | event names — Helene, Katrina, John. Sane |
| `whr778/gliner2-warmstart-natural-clean` | boundary | 0/11 | 3.6% | **nothing** — the record head fills no field |
| `whr778/gliner2-base-v1-casualty-docee` | span | **9/11** | 50.6% | **the casualty number** — `'230'`, `'1,400'`, `'250'` |

The 9/11 is an **artifact**, not a result: `casualty-docee` was trained on
`casualty_report{dead, injured, missing, location}`, so an `event` field is out of
distribution and it copies the anchor number into it. `'230'` contains no "helene", so C
scores it as a caught cross-event. The 50.6% false-positive rate is the same artifact firing
on genuine observations.

Two more defects in the same function:

- **§27.2 does not reproduce.** Published C was 2/11 at 26.5% FP; the same model on current
  code gives 3/11 at **44.6%**. Something in the eval path moved. Do not compare any new arm
  against the published numbers — re-run the control.
- **The fallback is unsound.** `if bound is None and recs: bound = recs[0]["event"]` attributes
  the *first* record's event to a span that did not match any record.

**All three fixed in `63249ed`:** `validate_binding()` requires the bound string to name an
event the event schema also found (and rejects purely numeric strings); the `recs[0]`
fallback is gone; and a coverage table separates competitor / ours / rejected / unbound.
`validate_binding` is unit-tested against all three observed artifacts. The two fixes moved
the false-positive rate independently — on `fastino`, dropping the fallback took C-raw
44.6% → 36.1%, and validation took it to 30.1%, while catches never left 3/11.

### 0b. TRACK A RAN 2026-08-17 — location supervision WORKS, and Track B is warranted

`casualty-loc-split.yaml` on a Lambda GH200, `fastino/gliner2-base-v1` + `casualty_loc_split`,
one variable against `casualty-docee.yaml` (the corpus). **Crashed at 79%** — step 4,719 of
5,936, epoch 6.3 — with `FloatingPointError: 1 non-finite micro-batch loss(es) were zeroed`
(`trainer.py:1280`). That is `fp16` overflowing. `checkpoint-epoch-6` was on disk and train
loss had plateaued (5.81 → 4.74 → 4.50), so the readout uses epoch 6. **The arm is therefore
6 epochs against the baseline's 8 — a real if small confound, stated not hidden.**

**Binding, via the fixed probe, 104 observations:**

| model | C catches | C false pos | what it binds |
|---|--:|--:|---|
| `fastino/gliner2-base-v1` | 3/11 | 30.1% | event names |
| `casualty-docee` (no location supervision) | 0/11 | 2.4% | **the casualty number** |
| **Track A, epoch 6** | **3/11** | **9.6%** | **event names** |

**The finding: location supervision fixes the field-semantics collapse.** `casualty-docee`
answers an `event` query with a number — the numeric-field collapse `_locate_place`'s
docstring predicted. The Track A model binds `Hurricane Helene`, `Hurricane Katrina`,
`Georgia`. Same catch rate as the base model at **a third of the false positives** (30.1%
→ 9.6%), and it correctly binds Katrina's 1,400 to `Hurricane Katrina`.

Location fill also passed its pre-registered gate comfortably: **51/54** against the
baseline's 26/33, on more records (54 vs 33).

**What it does NOT show.** Catches are still 3/11, unshippable as a detector. And the 11 is
contaminated: §27.2 established that six of the `230`s are mislabelled — they bind
`Hurricane Helene` because Helene's toll genuinely reached 230, so they are *correct*
predictions counted as misses. The real denominator is nearer 5. The model also binds
non-events (`the election`), so precision is better, not good.

**Verdict: positive, so Track B is warranted** — supervision changed binding behaviour in
the right direction in the cheapest possible setting. Before spending on Track B, fix
item -1: a 6-epoch-vs-8-epoch comparison selected on an unscoreable metric is not a
foundation to build the expensive arm on.

Model: `whr778/gliner2-base-v1-casualty-loc-split` (private, epoch 6).

### 1. Number-to-place attachment — DEMOTED to P2 on 2026-08-17; see item 2 for the live defect

**Nothing is blocked on this, and the two routes below should not be run as written.** Both
were re-examined against the code on 2026-08-17. Kept here rather than deleted because the
*question* survives; only the proposed answers do not.

**The routes are genuinely unrun** — verified, not assumed. `deaths_in` appears only in this
file, [[PROJECT_JOURNAL]], and unit tests: no training config exists. `casualty_multi_loc` is
wired into training, but as `STRUCTURE_DEFAULT` in `build_warmstart_mix.py`, i.e. as
`json_structures` — never as a relation or an event. And `run_pipeline.py` has no
joint/beam/decode-mode flag at all, so the `TypedEndpoints` arm has never touched this task.

**Why not to run them anyway:**

- **The beam arm is predicted-negative for the live defect, by our own measurement.**
  `EKF_MHT_BUILD_RECORD.md` §27.2 found the type signal catches **0/11** on cross-event *"because the
  type is RIGHT there"* — Katrina is a storm too. `TypedEndpoints` is a type constraint, so it
  targets the scope problem the magnitude gate already mitigates, not the one that is live.
- **The relation arm trains the wrong head.** A supervised `deaths_in(value, place)` is
  *satisfied by the contaminating pair*: Katrina's 1400 beside "North Carolina" is a
  well-typed `(value, place)` edge. A relation can fix place-pairing at best; it cannot
  express which event owns the number. Only the event formulation carries `event_key`, which
  is the field the EKF observation needs.

**The unstated prerequisite, and the real reason "runnable" was misleading.** Every mechanism
named above — `TypedEndpoints`, joint decode, `event_records`, the record head — is
**boundary architecture**. The casualty extractor is a **span** model: all three
`casualty-*.yaml` sit on `fastino/gliner2-base-v1` with no `architecture:` key, and a live run
loads `gliner2/models/span/model.py`. Any of these routes first requires moving the casualty
extractor onto a boundary base.

**What survives, and it is the finding not the fix.** Proximity, GPE tags, record-internal
location and admin rollup have all been tried and all failed. Rollup did what it was supposed
to (58 keys → 21, 84% of observations in six clean streams) and per-state tracking is *still*
catastrophic: North Carolina 5.637, Georgia with 0 of 5 values in plausible range. The reason
is not fragmentation — national totals get filed under whichever state the article happens to
be about. Zero-shot is close but fragile, and the fragility is still the argument for training
over prompt-tuning: `explicit-scope` phrasing got the hard aggregate case exactly right
(120 → North Carolina, 17 → Tennessee, correctly *excluding* the national 227) while two other
phrasings of the same request, same model, same text, got it wrong.

Note also that attachment is **not** "solved" by the gate: `EKF_MHT_DESIGN.md` §5 records a
9x win on Helene but a **2.3x loss on the clean held-out stream**. It is a stopgap with a
measured cost, which is an argument for trained binding rather than for complacency.

---

## P1 — known-wrong

### 2. Cross-event contamination — now the top real defect
> **Gated by item 0 (2026-08-17).** This is the live bottleneck, but the probe that would
> score any fix is unsound — it counts a copied casualty number as a caught cross-event.
> Fix the readout first; otherwise a training arm cannot be told from an artifact.

Whole-article reading via `extract_long` surfaced streams for `poland`, `bosnia`,
`afghanistan`, `iran`, `japan`, `ukraine`, `cameroon` — casualty figures lifted from unrelated
stories sharing an article body.

The gate answers "is this article about a mass-casualty event". It never answers "does this
number belong to *that* event". The date filter is the temporal version of that check and it
worked (Izmit 15 → 3 false bindings, zero genuine losses). **The spatial version does not
exist.** This is not cleanup — it is the same research question as item 1 seen from the other
end, and it should probably be solved once, for both.

Left deliberately unmapped in `datasets/helene2024/rollup.json`: mapping the foreign places
would hide this problem rather than fix it.

**Quantified 2026-08-11**, context audit of all 106 'dead' observations: 82.1% genuine Helene
casualties, **4.7% cross-event**, 3.8% non-casualty numbers, 9.4% unclear. The five are
Katrina 1400, a Typhoon's 250, Milton's 230, Bosnia's 16, and Hurricane John's 2 in Mexico —
they carry the *large* values, so the most damage per instance.

**Three signals tried, all failed** (`EKF_MHT_BUILD_RECORD.md` §27.2): nearest named event 3/11 at 32.5%
false positives, only-competitor-named 3/11 at 31.3%, record-head binding 2/11 at 26.5%.
Helene articles routinely name Milton and Katrina for comparison. Bosnia's 16 is structurally
invisible — Bosnia is a *place*, not a named storm.

Note the scope gate removes Katrina's 1400 **for the wrong reason** — because it is large,
not because it belongs to another event — so it keeps any *small* cross-event figure, as it
does with Bosnia's 16 and Mexico's 2.

#### Proposed next experiment: negative documents (data, not decode)

**Do not reach for sharper type boundaries.** The standard mitigations for event
cross-contamination — span-based boundaries, contrastive/hard-negative objectives — target
the failure this project already solved. Type energies separated unit errors 4/4 with 0/83
false positives and scored **0/11** on cross-event, because in every cross-event case the
type is *right*: Katrina's 1,400 scores `death toll` 0.95. The boundary architecture and the
GIST veto (item 11) both sharpen `death toll` vs `people evacuated`; neither can separate
Helene's dead from Katrina's dead.

**The gap is negative supervision on event identity.** Measured on 20,000 records of each
corpus: **0.0% of training documents have zero records.**

| corpus | records/doc | zero-record docs |
|---|---|---|
| `casualty_ft` | all 1 | **0.0%** |
| `casualty_multi` | mean 2.35, `{1,2,3,4}` | **0.0%** |

`build_multievent_corpus.py` already concatenates *k* interference snippets from other
streams — but gives **every** one its own record. The model is therefore never once shown a
figure it is supposed to leave alone. Practitioner experience puts the healthy share of
negative documents at **30–40% of the mix** (not measured here — a prior to test, not a
result).

Note `remove_json_structure_prob: 0.2` does **not** provide this. It drops the structure from
the *schema*, so no query is emitted at all; the model never sees the `casualty_report` query
answered with nothing.

**The change is small and local:** in `build_multievent_corpus.py`, keep a fraction of
interference snippets in the document text while *withholding their records*, so the gold for
that document covers the focal event only. Per-snippet span location (already implemented, to
avoid labelling one event with another's number) is exactly the machinery needed to know
which spans to leave unlabelled.

#### RAN 2026-08-20 — TRAINED, and SUPERSEDED by a one-line threshold

> **Verdict first.** Both arms trained (4 epochs, A10, ~$2.35, terminated). The treatment is
> real — blind-test precision up / recall down, and 15 of the control's 20 large Helene false
> positives removed, cutting ungated per-place error 46.844 → 19.822. Then a **declared
> per-event plausibility ceiling** — no model, no training, no GPU — recovered and exceeded
> that gain: at a ceiling of 2,000 the CONTROL wins both ungated (5.853 vs 6.194) and gated
> (3.336 vs 3.729), while carrying 81 *more* observations. The arm's pre-registered guard
> passes only against an undefended control. **Do not cite this as a success.**
>
> **And it fixed the wrong class.** The large false positives were never other storms' tolls:
> they were Asheville's population (94,000), Boone's (19,000), FEMA flood-insurance *policies*
> (129,933), wellness checks (15,000), power crews (8,000), troops (1,500), churches (1,100)
> and years read as tolls (1,916, 2,004). Both genuine cross-event figures — Katrina's 1,400
> and Maria's 3,000 — survive muting **and** the ceiling. **Cross-event is still open.**
>
> Two things this promoted out of the footnotes. **Item 3 (non-casualty numbers) is worth more
> than its 3.8% billing** — it carries the largest values, so it dominates nRMSE, and one
> figure destroyed one state's stream. And it splits into classes needing different
> mechanisms: entity typing reaches policies and churches but *not* troops and crews, which
> are living people in the affected area and need casualty-role semantics.
>
> Full write-up, raw probe output and the ceiling sweep:
> `tools/ekf_showcase/muting_arm_results/` (`README.md`, `FALSE_POSITIVES.md`,
> `PLAUSIBILITY_CEILING.md`, `PROVENANCE.md`).

#### BUILT 2026-08-12 — `--mute-interference-prob`, control proven

`build_multievent_corpus.py` + `tests/test_multievent_muting.py` (6 tests). Four things
the implementation settled, two of which change what the experiment can claim:

**1. No loss or model change is needed — the architecture already carries this.** A muted
snippet emits no record, so its figures are spans with no gold. `build_candidate_labels`
scores a candidate 1.0 only on an *exact* match with a gold pair; everything else takes
0.0 at full candidate weight, and the mask encodes validity rather than goldness — there
is **no ignore path**. Measured on the two-candidate case (focal gold, interference
muted): scoring the muted span high costs **1.1269 against 0.1269**, an 8.9x penalty.

The corpus has always depended on this — documents are full of unlabelled displaced
counts, magnitudes and dates — and guard 2's collision-drop only makes sense if
unlabelled-vs-labelled matters in both directions. Muting extends it to the figures that
actually confuse the model, and additionally drops the gold instance count from k+1 to the
unmuted count, supervising instance formation toward focal-only.

*Not* the mechanism, though it exists: an all-empty record yields `count = 0`
(`processor.py:968`) with its queries still counted (`model.py:1394`). That is the
fully-negative document, which finding 3 rules out here. It does confirm at code level why
`remove_json_structure_prob` is no substitute for either — it hits `continue` at
`processor.py:911` *before* `schemas.append`, so no query is emitted at all.

**2. The focal snippet is always `parts[0]`, so muting is learnable from POSITION.**
"Extract from the first paragraph" scores perfectly on this corpus without representing
event identity at all. Real articles do lead with their focal event, so the prior is not
pure artifact — but the corpus cannot distinguish the shortcut from the intended
behaviour. **Required control before any gain is read as event identity:** a held-out
probe with the focal placed last. Without it this arm cannot answer the Bosnia question,
which is the reason it was proposed.

**3. A true zero-record document is not constructible from this corpus.** Every snippet
reports a toll, so a document with the `casualty_report` query answered empty would teach
suppression of a *genuine* lead-event toll. What muting produces is the **partial**
negative — focal record kept, interference figures unlabelled. The measured "0.0% of
training documents have zero records" is real, but closing it needs negative *snippets*
(disaster text carrying no casualty figure) drawn from another source; it is a separate
lever, not this one.

**4. The control arm nearly moved silently.** Drawing the muting decision from the shared
`rng` advanced it once per interference snippet, shifting every later `randint`/`choice`
and rebuilding the corpus — 4,064 documents against the pre-change 4,065 **at
`mute_interference_prob=0.0`**. Fixed with a dedicated `mute_rng`, so the arms now differ
in labels only. Note the obvious test does *not* catch this: the buggy draw fired
regardless of probability, so all arms shifted together and stayed mutually identical.
Only comparison against a builder with no muting concept exposes it, so the control is
pinned by hash.

Measured on 40 streams (4,106 snippets), `--mute-interference-prob 0.35`:

| | control | muted |
|---|--:|--:|
| documents | 4,065 | 4,033 |
| instances | 9,869 | 7,763 |
| documents with a muted snippet | 0 | **1,688 (41.9%)** |
| unlabelled figures delivered | 0 | **3,075** |

Read the 41.9% as documents with a muted *snippet*, not as documents whose gold actually
changed: 69 of them lost nothing, because every value in the muted snippet had already
collided and so carried no record in the control either. Gold differs on **1,619**.

`0.0` reproduces the pre-change corpus **byte-identically** (`cmp`). The 32 missing
documents are focal-collision cases where every interference record was also muted; they
are dropped rather than emitted empty, for the reason in (3). The reported counter is
`dropped_empty` = 41 control / **73** muted — 41 of those are collision drops the control
makes too, so the muting-attributable loss is the 32-document difference, not the 73.

**Build the val split at `--mute-interference-prob 0.0`.** Nothing in the flag enforces
it, and a muted val is not comparable with the control arm or with any historical number.

**Why this and not another association signal.** It is the only candidate that would reach
**Bosnia's 16**, which is structurally invisible to every signal tried so far — Bosnia is a
place, not a named storm, so nothing keyed on storm names can see it. And the evidence says
this is a training-data gap rather than a decode gap: binding collapses 1.000 → 0.369 the
moment documents become multi-event, which no decoder change has moved.

**Pass/fail as pre-registered — and note the readout it named could not be scored as written.**
The 106-observation reference set turned out to be a cached artifact reproducible from no
committed state of the repo (`muting_arm_results/PROVENANCE.md`), so the arm was read against
a fresh baseline under one recorded invocation instead. Original text: cross-event share below
4.7% on the same 106-observation
audit; single-event binding stays ~1.000; the §20 harness unchanged.

#### Second lever, same data side: base-word positive/negative samples

A *different* granularity from the above, aimed at a different failure — **noun-phrase
routing**, where the head latches onto whatever salient noun phrase is nearby rather than a
filler of the requested type. Two reproducible instances:

```
"Rebels attacked the convoy near Aleppo on Tuesday, killing three soldiers."
  schema: victim = "a person harmed"
  gliner2-joint-boundary-rams-137k  ->  victim: ["convoy"]      # not a person
```

and, from the guide-score cache, `Person/Entity` at **0.56** outscoring the gold casualty
type on *"killed a man and his 14-year-old daughter"* — the span genuinely *is* a person
reference, so a generic person type wins on a casualty query.

Both are the same mechanism: the model routes to the syntactically salient NP, and the type
query only re-ranks among NPs rather than deciding whether the head word can fill the role at
all. Supervision at the **base-word** level — positives for head words that can fill a role,
negatives for words that cannot (`convoy`, `homes`, `customers` for a person role) — attacks
that directly, where a span-level objective does not: every candidate the span objective sees
is already a plausible NP.

Note this is orthogonal to the negative-document work above. Negative *documents* teach
**which event** a figure belongs to; base-word negatives teach **whether a word can head a
filler** at all. Item 11's GIST veto sits between them, on the query axis, and does neither.

**Untested here.** No measurement in this repo yet supports or refutes it; the two examples
above establish the failure exists, not that word-level supervision fixes it.

---

## P2 — research direction

### 2b. Does a fine-tune need an explicit regularizer, or is early stopping enough?
Raised 2026-08-18 while launching the real-vs-synthetic arms. The synthetic fine-tune
gained hugely in-distribution (entity fair 0.7946 → 0.9134) and **lost 22% relative on
real general NER** (entity strict 0.5320 → 0.4136, swept best-vs-best). FairEval says it
is not a boundary regression — BES and BEL both fell while FN rose 27,267 → 36,244. The
model stopped proposing spans and mislabelled more of what it proposed: the 125-type
synthetic label space overwrote what the base knew.

**No regularizer was added to the real arms, deliberately.** The synthetic control was
trained without one, so a regularizer in the new arms would confound the comparison —
any preservation difference could be the penalty rather than the text source. There is
also a real hypothesis that real news needs less of one: the damage came from training
on out-of-distribution prose, and cc_news is far closer to `pile_ner_def` than generated
passages are.

What WAS changed is retention only: `save_total_limit` 3 → 10 on both arms, so every
epoch checkpoint survives (`best`/`final` are exempt from rotation — `trainer.py:2332`).
Selection is on in-domain val, so if forgetting grows with epochs the best in-domain
checkpoint is the worst preserving one. Keeping all ten lets the preservation curve be
scored per epoch and the knee found post hoc. **Early stopping is the cheapest
regularizer and needs no change to the loss.** It doubles as crash insurance.

Order of levers if the curve shows real text still degrades: early stopping first (free,
already instrumented), then replay of base-distribution data in the mixture, then
parameter-space constraints (LoRA / L2-SP / EWC). Do not start at the expensive end.

### 2c. Chunking distorts the real-news arm — measured, and not where expected
`window_size`/`stride` are **subword tokens, not words**; the configs' "word window"
comments are wrong and `gliner2/training/chunking.py` is authoritative. At 384/256 on
1,500 cc_news train docs (deberta-v3 tokenizer), 2.25 chunks per document.

The hypothesis was that cross-window structure is lost and that this argues for mmBERT's
8,192 context. Measured per document — supervision that survives in **no** chunk:

    relations   1,794 in source,  84 lost = 4.7%
    events        383 in source,   6 lost = 1.6%

So the long-context argument is real but modest, not decisive. **The larger distortion is
classification inheritance**: doc-level labels are copied to every chunk (+124.5%, exactly
the 2.25x expansion), so most classification training examples are fragments asserting a
document label the fragment may not support. That is injected label noise, and it lands
hardest on the arm with the longest documents. Quantify its effect before adding more
classification data to a real-news mixture.

Raw annotation counts are useless for this — they rise across the board under chunking
(entities +42.4%) because overlap duplicates them. Measure survival per document.


### 2d. What breaks a stochastic activation is FUNCTION-CLASS churn, not randomness
Explored 2026-08-18, prompted by the observation that GeGLU's gate reintroduces an
unbounded gradient path. **Answered -- and the motivating premise was then REFUTED in a
real transformer. CLOSED.**

**Read this first.** `tools/prototypes/lr_ladder.py`, 6-layer MLM encoder on wikitext-2,
LR escalated to divergence: plain GELU dies at 1e-1, **GeGLU and hybrid4:fixed both
survive it and both die at 3e-1**. So GeGLU is the MOST stable of the three, not the
least, and hybrid4's toy-measured gradient-ceiling advantage (13.4 vs 23.2) does not
transfer -- in the encoder its grad norms track GeGLU's and it breaks at GeGLU's
threshold. The gradient-bound analysis below is correct as measurement and wrong as
prediction: a bound on a layer's local derivative is not a bound on training dynamics,
which depend on normalisation, depth, the optimiser, and the network's ability to adapt
its own gates. Full write-up in `tools/prototypes/PARTIAL_GATING.md` section 11.

The function-class-churn result (below) is unaffected -- it is about stochastic masking,
not about gating, and it still holds. Cost of the refutation: $0, ~90 min local MPS,
against a ~$379 staged plan that the stop rule correctly cancelled.

The starting diagnosis is correct and worth recording. A pointwise activation has a
bounded derivative -- ReLU exactly [0,1], GELU [-0.1289, +1.1289] -- so it cannot
amplify a gradient. GeGLU's `y = v * gelu(g)` gives `dy/dv = gelu(g)` and
`dy/dg = v * gelu'(g)`: each branch scales with the OTHER branch, and neither is
bounded. Measured max |grad| through the activation, by input scale:

    scale      gelu    geglu   stoch-gelu   1-gated-chunk-of-6
        1      1.13     4.42         1.13                 4.36
        2      1.13     9.80         1.13                 8.69
        4      1.13    19.48         1.13                17.41
        8      1.13    37.77         1.13                32.68

**Partial gating does not partially protect.** In the hybrid only 2.6% of channels
exceed 1.13, yet the MAX is within 15% of full GeGLU. Explosion risk is set by the
worst channel, not the mean, so gating one chunk in six buys ~1/6 the exposure and
~6/6 the tail. Note also that GELU never fixed explosion over ReLU -- both are
bounded. What GELU fixed was dead units.

**The negative: randomising WHICH CHANNELS GET THE NONLINEARITY costs ~10x.** Five
chunk layouts, param-matched 4-block MLP, test MSE (lower better):

    layout            fixed    random
    2-of-4           0.0567    0.7267
    6-of-12          0.0702    0.8010
    8-of-12          0.0492    0.6239
    10-of-12              -    0.4326
    HYBRID4 8-chunk  0.0655    0.6555

No overlap: deterministic 0.049-0.070, random 0.43-0.80. Per-sample masks (0.7783)
and finer chunks (0.7439) do not help; the random arms improve with the activated
fraction only because that dilutes the randomness (at p=1 it IS plain GELU). The
random arms also have the TIGHTEST seed spread (HYBRID4 random +/-0.0089) -- they
converge reliably to a bad solution, which is a method-level floor, not bad luck.

**But randomness itself is not the problem, and HYBRID5 is the control that proves
it.** HYBRID5 keeps 6 slots always-GELU and gives each of the 2 linear slots a
randomly drawn GELU'd partner to multiply. Same harness, same target, also randomized:

    randomized variant   what the draw changes                      test MSE
    HYBRID4              whether a slot is GELU or identity          0.6555
                         -> the slot's FUNCTION CLASS moves
    HYBRID5              which chunk partners a gated slot           0.0623
                         -> function class fixed, only the operand moves

**So the rule is: a draw that changes what KIND of function a slot computes is fatal;
a draw inside a stable function form is free.** `fc2` reads a fixed slot, and one
weight cannot be correct for both GELU output and identity output. **Dropout escapes
this only because it is linear in the mask** -- `E[mask*x] = p*x`, so one scalar
corrects it. Swapping a nonlinearity has no scalar correction, hence on
HYBRID4-random weights: expectation blend 0.8203, sampled 0.8622, **plain GELU at
eval 7.2860** -- the intuitive "stochastic at train, clean at eval" rule is the worst
of the three. HYBRID5 has no such problem: `c6` is independent of the draw, so
`E[c6 * g_j] = c6 * mean(g)` is an EXACT eval rule, not an approximation.

**The positive, worth trying in a real model.** HYBRID4 with a FIXED assignment --
8 chunks, 6 GELU, the two linear slots holding `a*b` and a passthrough, every chunk
staying in its own slot -- is parameter-identical to a plain GELU FFN (100,481 both),
preserves width exactly (no 2x up-projection, so none of GeGLU's +50% or the 2/3-d_ff
workaround), scores 0.0655 against plain GELU's 0.0636, and has the LOWEST gradient
max in the study because only 1/8 of channels carries a product.

**One gated slot in eight is the sweet spot; two is worse.** Whole-network gradient
max, every row statistically tied on MSE:

    HYBRID4 fixed   13.4    1 gated slot of 8      MSE 0.0655
    plain GELU      17.2    none                       0.0636
    GeGLU (2/3)     23.2    all channels gated         0.0483   (+50% params raw)
    HYBRID5 fixed   26.1    2 gated slots of 8         0.0601
    HYBRID5 random  34.0    2 gated, random partner    0.0623

HYBRID5 is the better result scientifically and the worse design: it tolerates
randomization but moves the gradient ceiling the WRONG way, above GeGLU, which is the
thing this whole line of work set out to avoid.

**Prior art -- the mechanism is Shazeer's, the fractional application is what is not
covered.** Noam Shazeer, *GLU Variants Improve Transformer*, arXiv:2002.05202 (2020),
defines the family this work sits in: GLU with sigmoid, **GEGLU** with GELU on the gate
(what mmBERT/ModernBERT use), SwiGLU with Swish, and **Bilinear** -- the variant that
omits the nonlinearity entirely and is just the component-wise product of two
projections. **HYBRID4's product chunk IS Bilinear**, applied to 1/8 of the channels
instead of all of them. That paper is also the origin of the two-thirds `d_ff` rule this
entry quotes for parameter matching, and it reports GEGLU/SwiGLU as the best variants.

Every variant there gates ALL hidden units. A search over the obvious phrasings found no
published study of gating only a FRACTION of FFN channels with the rest left pointwise.
**Treat that as weak evidence, not a novelty claim** -- web search is not a systematic
review, the construction is simple enough to be sitting unremarked in someone's ablation
appendix, and a negative like the randomisation result is exactly the kind of thing that
never gets written up. Adjacent but not the same: Highway Networks (Srivastava et al.,
2015) mix a transformed and a carried path under a learned gate; arXiv:2410.08417 studies
bilinear MLPs for weight-based interpretability.

**Do not over-read the small gaps.** Harness is a 4-block residual MLP, D=64, H=192,
AdamW 3e-3, 3000 steps, 3-5 seeds, synthetic regression target with multiplicative
interactions. Plain GELU alone ranged 0.043-0.072 across runs. This harness separates
0.06 from 0.65 reliably and cannot separate 0.048 from 0.066 at all. Every claim above
rests on the first kind of gap. Scripts were scratchpad-only; the recipe here is the
record.

One harness bug, corrected mid-study and worth not repeating: applying the expectation
blend at eval to a DETERMINISTIC mask is a train/eval mismatch, not a calibration. It
reported FIXED variants at 7.54 and 81.1 before the fix; the real numbers are 0.0492
and 0.0702.

### 3. §10's crux is reopened; §14 does not reproduce
The harder-regime ablation concluded the EKF's edge *widens* under unreliability. On real
Helene trajectories the gain is flat and *shrinks* at the hardest setting (+1.8% → +0.8%).
The likely reason: §14 measured synthetic streams generated by the same rise/decay dynamics
that `est_ekf` models. **A dynamics model validated on data generated from it is not
validated.** Either re-derive on real trajectories or drop the claim from the paper. Do not
leave it standing as written.

### 4. "Boundary beats span at 10K" is unverified
It compares against 0.158 from a different experiment whose blind-test support was never
checked. A support mismatch (3,527 vs 20,845) already invalidated the cold-base row of this
same curve once today. Re-derive on a shared test set before this goes anywhere near Paper 0.

### 5. The relation regression in the warm start (−0.037, −22% relative)
`task_lr` is 5.0e-4, tuned in the curve for **cold** heads. In the warm start the relation head
is already warm and sees only 8% of the mixture — few gradients at a high rate. Test a lower
`task_lr`, or a per-head rate. This targets the regression more directly than `encoder_lr`,
which acts on the shared trunk. One run, one variable.

### 6. MHT — ANSWERED: not the bottleneck, do not build it yet
§3 specifies gate → Hungarian → top-K hypotheses → track birth/death; none is built.
Measured 2026-08-11 by assigning every observation to the scope it actually fits using
ground truth — a ceiling, not a method:

    shipped scope gate      0.591
    oracle association      0.537
    headroom               +0.055     (9.3% relative)

MHT is a hypothesis tree, cost matrix, Hungarian assignment and track management, competing
for a 9% residual. Sharper still, **the gate already beats a perfect two-way assignment** on
Florida (0.704 vs 0.734) and South Carolina (0.365 vs 0.558) — it has a third option the
oracle lacks: *drop*. Florida's 300 and North Carolina's 1400 are not misassigned; they
belong to no Helene scope at all.

Tennessee is the one genuine association gap (0.817 vs 0.320), and it is diagnostic: its
contaminants are 32, 32, 32, 36, 50 against a truth of 18 — **too large for the state, too
small to look national**, exactly what a magnitude rule cannot catch.

Revisit when multi-source feeds land (item 7): sources disagreeing about one event is real
association ambiguity in a way one wire service's copy is not.

### 7. Still no benchmark that can score the filter
Turkiye's baseline was an oracle by construction — truth read from the sentence the extractor
reads, so `est_last_value` scored 0.000. Helene's per-state streams were mis-bound until the
scope gate. A real filter benchmark needs **multiple sources that disagree and revise** about
one event, which is also the regime where MHT would finally earn its keep.

### 8. Beam vs greedy — RAN, and the result is "the beam is not the story"
Ran 2026-08-10 on Re-DocRED (`joint-boundary-redocred-137k`, 96 relation types, the schema
that raised before the qualified-key fix). Same checkpoint both arms, eval-time
`decode_mode` switch, threshold 0.5, full 500-doc test:

| | greedy | joint (W=16) |
|---|---|---|
| relation strict F1 | 0.0740 | 0.1803 |
| entity strict F1 | 0.6960 | 0.6786 |

**Do not quote that +0.106 as a beam win.** It is largely a threshold artifact — 0.5 is
near the worst operating point for greedy, which reaches 0.2082 at 0.1 in its own shipped
sweep. Three real findings did come out of it:

**(a) Beam width should be 1.** Sweep over W ∈ {1,2,4,8,16,32,64} on a 20-doc slice, relation
strict F1: 0.2406 / 0.2290 / 0.2260 / 0.2211 / 0.2170 / 0.2152 / 0.2058. **Monotonically
decreasing.** Widening drops predictions 157 → 117, of which 18 were correct (45% precision
on the dropped set, below the 61% overall), so precision rises and F1 falls. Entity metrics
are byte-identical at every width — `_finish_nodes` sweeps in every positive-score node
regardless of beam state, so width touches only edges. Classic score-vs-F1 divergence: the
wider beam maximizes the objective better, and the objective is not F1.

**(b) The gain is the formulation, not the search.** W=1 barely searches and wins. The
working contrast is *independent thresholding vs constrained joint selection*, not
*greedy vs beam*. Phase A's framing is mis-specified and the papers should say so.

**(c) It exposed the hard-wired threshold** — see item 9, which was the actual bug.

**Best-vs-best, settled on the slice after item 9 was fixed:** both arms peak at threshold
0.2 — greedy **0.2835**, joint W=1 **0.3357**. **Joint wins by +0.052 (+18% relative)** and
beats greedy at every threshold on the grid. Real, but a third of what the fixed-0.5
comparison implied. Remaining: confirm on the full 500-doc test. Wall clock 1.5x greedy on
a clean slice (the 2.0x full-run figure was CPU-contended).

### 9. Joint decode ignored `--threshold` for edge selection — FIXED 2026-08-10
`joint_decode` filtered mentions by `mention_threshold` but never passed
`decision_threshold`, so it stayed at its 0.5 default and every node/edge utility was
centered on 0.5. `gain > 0` therefore demanded p > 0.5 for edges no matter what threshold
was requested. Nothing raised; the decode simply stopped responding to `--threshold`, which
reads as a model insensitive to calibration rather than as a plumbing bug.

Measured before the fix, relation recall across thresholds 0.5 → 0.1:

| arm | R @ 0.5 | R @ 0.1 |
|---|---|---|
| greedy | 0.0461 | **0.4134** |
| joint W=1 | 0.1498 | 0.1591 |

Fixed by threading `decision_threshold` from the eval threshold through `joint_decode`.
Record **role edges bypass** it via a new `pre_scored_edges` path: a scalar role's utility
is the ABSENT-relative log-odds `logit_c - logit_ABSENT`, a comparison against the record
head's own ABSENT class rather than a probability cutoff, so shifting it would move scalar
roles against a baseline they do not have. That was documented at `candidate_scores.py:223`
and is now enforced by a test rather than by a comment.

**Consequence for anything already measured:** every joint-arm number produced before this
fix — including the 12-arm curve's joint rows, if any were run — was measured at 0.5
regardless of the threshold requested.

### 10. Aggregate SCOPE (not the aggregate constraint) — the sharpened target
Two different things wear the word "aggregate" and only one of them is open.

**The constraint direction is measured and it LOSES.** `vector_state_test.py` feeds the
national total in as a sum row over the six state components. Against `parts-only`, on real
Wikipedia trajectories with `--q-prop 0.15`:

| per-state report rate | parts-only | vector | delta | vector wins |
|---|--:|--:|--:|--:|
| 10% | 0.4348 | 0.6085 | +0.174 | 4/40 |
| 50% | 0.2030 | 0.2234 | +0.020 | 22/40 |
| 80% | 0.1556 | **0.1520** | **−0.004** | 30/40 |

It loses everywhere except 80% density, and loses **worst exactly where it was predicted to
win**. An aggregate constrains the SUM and says nothing about the SPLIT, so when parts are
sparse the filter must guess the division and the total injects error. Do not revisit this
without a new reason; it is not "deferred pending recall", it was tried and it lost.
(Isotropic `Q` makes it 7.7x worse still — proportional process noise is a precondition,
not a tuning knob, since Virginia ranges 1→2 while North Carolina ranges 6→123.)

**The scope direction is open and is where the remaining error lives.** The failure is
filing a national total under a state — measured on real text: "The number of deaths stood
at 225 on Friday; two more were recorded in South Carolina" binds **225 → south carolina**.
That is not a rival claim about South Carolina, and a wrong state silently poisons a state
stream where an unbound total is recoverable.

**Measured contamination.** Every state stream receives larger-scope numbers, and the leak
is always UPWARD — never once downward:

| stream | truth (final) | contaminants received |
|---|--:|---|
| Florida | 26 | 64, 150, 150, 160, 180, 230, 230, 300 |
| North Carolina | 96 | 200, 215, 215, 227, 230×3, 250, **1400** |
| South Carolina | 51 | 72, 200, 227 |
| Georgia | 34 | 178 |

**Sub-part 1 (unlocated → `__aggregate__`) is a NO-OP: 4 of 106 observations.** It was
proposed first on the reasoning that it had no bootstrap dependency; measurement says it is
not worth doing on its own. Multi-state scope phrases (sub-part 2) are already handled by
`rollup.json`'s 38 aliases.

**Sub-part 3 — the scope gate — WORKS** (`scope_gate_test.py`, 2026-08-10). Judge each state
observation against the running **national** total rather than against the state's own scale
(a state's early history legitimately jumps 6 → 25, faster than any ratio tolerates), and
classify three ways: keep / reroute to `__aggregate__` / drop as exceeding the whole.

| ratio | Total | per-state mean |
|---|--:|--:|
| off | 0.402 | 5.247 |
| 2.5 | **0.316** | 0.592 |
| 2.0 | **0.316** | **0.591** |
| 1.5 | 0.317 | 0.591 |

Per-state **5.247 → 0.591 (8.9x)** and the national stream *improves* too. Flat from 1.5 to
2.5, so it is not a knife-edge setting. **Control:** removing the same 25 observations at
random over 40 trials gives 4.427, so the gate is selecting rather than thinning.

Three-way classification is load-bearing. A two-way version that rerouted every reject wrecked
the national stream (0.402 → 2.110), because North Carolina's **1400** is not a national
total — it is not a casualty count at all, and it poisoned `__aggregate__`.

**Held out on Turkiye-Syria (2026-08-10), ratio fixed at 2.0, not retuned. Partly transfers,
and the failure is the informative half.**

As validated it **cannot run**: the gate judges against the `__aggregate__` stream and
Turkiye-Syria has none — turkey and syria are siblings with no declared parent, and the
combined toll never got its own stream. With `--reference aggregate` the gate is a no-op at
every ratio.

Generalizing the reference to the running max across all streams (`global-max`) makes it run:

| | turkey | syria | mean |
|---|--:|--:|--:|
| off | **0.228** | 3.401 | 1.815 |
| gate @2.0 | 0.522 | **0.923** | 0.723 |

Syria — the contaminated small stream, 11 of 17 values were Turkey's tolls — improves 3.7x.
But **Turkey, which was clean, degrades 2.3x**, because `global-max` is dominated by Turkey's
own values, so Turkey is judged against a reference it defines itself. It rerouted 1,014 at
t=12.5h, which is Turkey's *true* value at that time. Circular by construction.

Mean still improves 2.5x with the control at 1.440 vs 0.723, so the mechanism does transfer.
The **reference definition does not generalize for free**.

**The finding: the gate needs a declared scope hierarchy, not just a magnitude.** Helene has
one (`__aggregate__` in `rollup.json` declares states ⊂ national). Without it, a magnitude
test cannot separate "this is a larger scope" from "this is the largest part". Next step is
to declare the hierarchy per event rather than infer it — cheap, and it is the same
information `rollup.json` already carries.

Other caveats: the ratio was chosen after seeing Helene's contaminated values, so the 1.5–2.5
plateau mitigates but does not remove the post-hoc problem. And 0.591 is 9x better than
catastrophic, not good in absolute terms.


### 11. GIST query-axis hard negatives — RAN 2026-08-14, and it is NEGATIVE

**The A/B is done and the veto lost.** Two arms on one 2xH100, one per GPU, differing in
exactly four keys, `mix_natural` 84,280 records x 3 epochs (15,804 steps, 1h44m / 1h43m).
Both arms swept to threshold 0.3 on val, so this is best-vs-best:

| metric | control | gist | delta |
|---|--:|--:|--:|
| entity strict | 0.5858 | 0.5610 | **−0.0248** |
| entity fair | 0.6248 | 0.6041 | −0.0207 |
| relation strict | 0.1439 | 0.1108 | **−0.0331** |
| classification | 0.6301 | 0.6292 | −0.0009 |
| event_type | 0.9531 | 0.9515 | −0.0016 |
| event_trigger fair | 0.7527 | 0.7399 | −0.0128 |
| event_argument fair | 0.5786 | 0.5702 | −0.0084 |
| event strict | 0.3433 | 0.3443 | +0.0010 |

It loses on every metric but one, and that one is +0.0010. The sharpest reading is that it
is **down on `event_argument`** — the axis the query veto was built to sharpen — so this is
not "right idea, wrong dosage" on the evidence available.

**The control validates the harness rather than the conclusion resting on it:** retrained
from scratch it reproduces the historical `warmstart-natural` reference (relation 0.1439
vs 0.154, entity 0.5858 vs 0.580), so the gap is the treatment.

**The veto was live, not inert** — `[gist] loaded 46149 cached guide records` in the
training log, which is the failure mode the wiring notes below warn about.

Caveats before this is called settled: one seed, and no variance estimate on this corpus,
so anything under ~0.005 is unreadable. −0.025 and −0.033 are well outside that. Artifacts
(both `best/` checkpoints, sha256-verified off the box, plus metrics and sweeps) are local
under `out/gist-ab/`, so a re-probe needs no retrain.

#### The original specification, kept because the cache and wiring are still sound
The measured gap: with specific rival types, `people evacuated` outscores `death toll` on
**11.2% of genuine death tolls**. "N people killed" vs "N people evacuated" — both counts of
people, separated only by the verb. No type description fixes it (`EKF_MHT_BUILD_RECORD.md` §27.8); it
is a training-time boundary the model has never been taught.

Hard negatives are mined on the **span** axis only — `select_hard_negative_candidates` picks
negative *spans* per query. The missing axis is **query**: for a span, which sibling type
queries score it highly.

Wired 2026-08-11. Set `guide_scores: <cache.jsonl>` in a training config and the veto is
live; leave it unset and nothing in training changes.

| piece | state |
|---|---|
| `apply_guide_veto` + abstention `floor` | `losses.py`; takes an explicit `reference` |
| guide choice | self-guide validated **82.5% vs 25%** chance on 40 gold records |
| rival selection | wide-pool top-k; **no embedder needed** |
| rival **injection** | `GuideScores.inject` — dataset-side, hardest-first |
| cache -> `[B,Q,C]` | `models/boundary/guide.py`, with hit-rate counters |
| `precompute_guide_scores.py` | batched; format frozen (`sha1` key + rival descriptions) |
| **the cache itself** | **BUILT 2026-08-12** — `data/guide_scores.mix_natural.dedup.jsonl` |
| **a RAMS cache too** | **BUILT 2026-08-12** — `data/guide_scores.rams_baseword.dedup.jsonl` |

#### Running the arm — four things checked 2026-08-14, before any spend

1. **The A/B is TWO training runs, not one.** Five commits touched the training path after
   the control trained on 08-10 (`ca3e362`, `e189362`, `bf2c9b4`, `3a83c8d`, and `210af17`,
   the GIST wiring itself). `mix_natural` is 7.6% events (379 of the first 5,000 train
   records), so the Tier 2 event-record changes are **not** inert here and the existing
   control checkpoint is not a valid arm against a fresh GIST run.
2. **Neither checkpoint is local.** `out/joint-boundary-mmbert-137k/best` — the GIST
   config's `pretrained` — and `out/warmstart-natural/best` are both absent. Both are on the
   Hub privately (`whr778/gliner2-joint-boundary-mmbert-137k`,
   `whr778/gliner2-joint-boundary-warmstart-natural`).
3. **Both warmstart configs select on `metric_for_best: eval_loss`.** Kept deliberately for
   this arm: with 3 epochs there are 3 candidates, so selection is a small lever, and the
   decision that matters is the swept-threshold comparison between arms. Revisit if the arm
   is ever run longer.
4. Pull the checkpoints off the box **before** terminating it, and sweep thresholds on both
   arms before reading the comparison. Item 12 is what skipping either costs.

#### The cache — built 2026-08-12, and the merge needed a fix

Four local shards, **21.2 hours**, each verified at exactly `21070 records read`
(4 x 21,070 = 84,280, the whole corpus). 46,581 cached records, 0 malformed. Hit rate on
the corpus is **54.5%**, which is right: only records with gold spans are cached
(46,581 / 84,280 = 55.3%). Loads in 1.1s. Train with
`tools/train/config/warmstart-natural-gist.yaml`, which differs from the
`warmstart-natural` control in exactly four keys — `guide_scores`, `rivals_per_record`,
`output_dir`, `experiment_name` — with the data section identical.

**Use the DEDUP file, not the raw concatenation.** 194 of 46,343 sha1 keys collide, and
all 194 **conflict**: the same text appears twice in `mix_natural` declaring *different
entity type sets* (indices 104 and 46250 share text but declare `{name, severity}` versus
`{address, symptom}`). That is a property of the corpus, not a sharding bug — the shards
partition by index, so a repeated text lands in different shards.

It matters because `GuideScores.load` does a plain `entries[row["sha1"]] = ...`
(`guide_scores.py:80`): **last wins, no warning**. Those records would be vetoed against
another record's own-types — and own-record types must never be vetoed, since gold is
authoritative within a record. Dropping the colliding keys costs 0.42% of the cache and
leaves the veto explicitly *inactive* for them rather than quietly wrong. The four shard
files are retained, so any variant rebuilds in seconds.

**Two things the wiring turned up, both of which would have made it silently inert:**

1. **Injection is not optional.** A sample's query axis carries only the types its own
   record declares, and only 0.23% of records name a competing count type natively. The
   cross-record rivals GIST exists for are *never* on the tensor unless something puts
   them there. Without injection the veto is a no-op by construction.
2. **`apply_guide_veto` could not fire under the default candidate pool.** It derived each
   candidate's own positive by taking a max down the query axis at a fixed column — which
   assumes column *c* is the same span for every query. True for `candidate_pool="shared"`,
   **false for the default `"per_query"`**, where each query proposes its own list. The
   reference is now resolved by span identity and passed in explicitly.

Own-record types are deliberately never vetoed: within a record gold is authoritative, and a
same-record rival outscores the gold owner 23.5% of the time — all of it correct hard
negatives. Enforced structurally, by only ever filling injected-rival cells: everything else
sits at exactly 0.0 and cannot clear `floor`.

**Still to run: the precompute — and it is a LOCAL, SHARDED job, not a GPU one.** Renting an
A100 to find out was worth the $3: same 96 records, byte-identical output, **376.0s on the
A100 (3.9 s/record, 4-13% GPU utilisation) against 186.3s on a laptop (1.94 s/record)**. The
accelerator was half the speed, because the cost is Python post-processing rather than the
forward pass — ~100 type queries at `threshold=0.0` decode every candidate for every query
and the cache then throws nearly all of it away. So `--score-threshold` (now default 0.01)
is the real knob, and `--shards` across cores is how the job gets shorter.

Filtering does not close the cost either — a numeric-gold filter keeps 66.6%, a
count-type-name filter 37.1% — because only 3 of 8 records yield a coherent rival at all and
there is no cheap way to know which in advance.

**Nor does renting a bigger box — run it locally.** A 240-vCPU / 1771GB instance ($22.32/h,
120 shards × 2 threads) cached **zero** records in 15 minutes: >33 s/record per shard against
**3.3 s/record on a laptop**, ~3.6 rec/s aggregate versus 1.2. Workers were at 142% CPU with
1.4TB RAM free while load stalled at 172 of 240 — the ceiling is **memory bandwidth**, not
cores (~30 concurrent processes is about where a mid-size server's bus saturates). Choosing
the box on `$/vCPU-hour` assumed throughput scales with cores; it does not. **Measure one
shard's s/record on the target box before renting.**

Local shape that works: **4 shards × 2 threads with `--pool-cache`**, ~1.2 rec/s, ~19h for
the full mix. More shards than that exhausts a 32GB machine and swaps it to a standstill.

**Do not** use the live model as the guide. A cell is mined *because* the live model scores
it highly, so a live self-guide vetoes exactly the negatives it should select. The guide must
be a frozen checkpoint.

### 12. Base-word (lemmatized) duplicate samples — BUILT, alignment proven; not yet trained on

`tools/data/augment_baseword.py` + `tests/test_augment_baseword.py` (5 tests).
Measured on 300 RAMS records with the deterministic `mock` backend:

| | |
|---|---|
| augmentation rate | **91.7%** (275/300) |
| texts actually rewritten | 275/275 — not a silent no-op |
| labels no longer verbatim | **0** |
| extra mentions lost vs original, through the real collator | **0** |

The 8.3% that are refused are labels covering only *part* of a token — `Armenian` inside
`Armenians` — which cannot survive lemmatization of their host token. Those records are
dropped whole rather than emitted with a broken span; partial augmentation is precisely the
silent-supervision-loss failure this is guarding against.

Example (mock backend, so `urging`→`urg` is crude on purpose — it tests alignment, not lemma
quality):

```
ORIG : Transportation officials are urging carpool ... death of Freddie Gray
LEMMA: transportation official are urg carpool ... death of freddie gray
args : ('victim', 'Freddie Gray')  ->  ('victim', 'freddie gray')
```

#### RUN 2026-08-12 on `--backend simplemma` — and the stated gate was VACUOUS

Full RAMS train, simplemma 1.2.0, `--lang en`: **7,329 → 13,291 (5,962 augmented, 81.3%)**,
3 seconds. The gate passes — gold mentions 27,599 against 27,599, zero records changed —
but only after two corrections, both of which the arm would otherwise have been trained
under.

**1. `missing_surface_counts()` cannot serve as this gate.** It increments only for
`task_type == "entities"` (`boundary_preprocessing.py:443`). RAMS supervises **events**,
and for non-entity types an unlocatable surface is treated as legitimately absent and
skipped with **no counter at all** (`:465`). A lemmatized copy could lose every argument
and the counter would still read 0. What is observable is the target graph:
`targets.mention_mask.sum()` is the gold the collator actually built, and each lemma copy
must produce exactly as many as its source record.

Collate with sampling OFF when measuring this. `collate_fn_train` sets `is_training=True`
and the default `remove_events_prob=0.2` drops the whole event group a fifth of the time —
one record collated ten times gives `[5,0,0,5,5,5,5,5,5,0]`. The first version of this
measurement was reading that noise.

**2. The real failure is INVENTED gold, not lost gold.** Lemmatization *collapses* surface
forms, so a label starts matching positions that were never annotated. Gold `guns` occurs
once in its source; as `gun` it occurs **three times** in the lemmatized text, so collation
builds three mentions where one was annotated. Before the guard: **+1,085 mentions on
31,773 (3.4%), in 718 of 6,680 augmented records** — every one a silent false positive, and
invisible to any missing-surface check by construction.

Guarded by refusing any record where a label's occurrence count changes, in the same
tokenization collation uses. That ruler is load-bearing: `WhitespaceTokenSplitter` is a
regex tokenizer that splits trailing punctuation and lower-cases, so `they,` contains the
token `they` while `str.split()` sees only `they,`. A `str.split()` guard still let four
records through, netting to a delta of 0 by coincidence — two gaining, two losing.

Cost of the guard: augmentation rate **91.1% → 81.3%**. Those are refusals, not losses;
the un-augmented original is always emitted.

**Still to do:** train the arm. `simplemma` is installed to a scratch dir and used via
`PYTHONPATH`, deliberately not `uv add`, which re-locks and syncs the whole environment and
could rewrite packages the four precompute workers have mmap'd. Make it a real dependency
once they exit.

#### Arms A/B/C RAN 2026-08-12 — PROVISIONAL, and unreadable until thresholds are swept

Three arms, configs differing from `gliner2-base-v1-rams.yaml` in three lines each (train
file, `output_dir`, `experiment_name`), val and test un-augmented. Trained on a Lambda A10,
which is terminated; `test_metrics.json` for all three plus trimmed logs are local under
`out/gliner2-base-v1-rams{,-baseword,-dupcontrol}/`. **The checkpoints went with the box**,
so the sweep below cannot be run without retraining.

Confirmed 2026-08-14: those three directories contain `test_metrics.json` and nothing else —
no `best/`, no `threshold_sweep.json`, no per-epoch checkpoints. Recovering the sweep is
three full retrains (~4h, ~$5 on an A10), and this file's own "higher-value uses of the same
GPU hour" note argues against spending it here.

| arm | train file | records |
|---|---|--:|
| A baseline | `data/rams.train.jsonl` | 7,329 |
| B lemma | `datasets/rams_baseword/train.jsonl` | 13,291 |
| C duplicate control | `datasets/rams_baseword/train.duplicate_control.jsonl` | 13,291 |

Blind test on the un-augmented RAMS test set, support 2,016 arguments / 848 triggers on
every arm:

| metric | A base | B lemma | C dup | B − C |
|---|--:|--:|--:|--:|
| **argument strict** | 0.4474 | **0.4582** | 0.4463 | **+0.0119** |
| argument fair | **0.6192** | 0.6124 | 0.6072 | +0.0052 |
| argument relaxed | **0.6873** | 0.6805 | 0.6781 | +0.0024 |
| event strict | 0.6797 | **0.6892** | 0.6885 | +0.0007 |
| trigger strict | 0.9127 | 0.9313 | **0.9369** | −0.0056 |
| event type strict | **0.9970** | 0.9887 | 0.9941 | −0.0054 |

**The one number that carries the item: C sits at baseline.** 0.4463 against A's 0.4474 —
duplicating those 5,962 records verbatim bought **nothing** on strict argument F1, while B
beats both by +0.0119. On the metric base-word supervision actually targets, the gain is
therefore attributable to **lemmatization, not duplication**. That is the B > C row of the
decision table below, so arm D's dosage question becomes legitimate.

**Do not act on it yet. Three reasons, in order of severity:**

1. **Every arm sits at the config's fixed threshold 0.5, unswept.** The standing rule in
   this project — promoted from lesson to rule *because it changed a conclusion three
   times* — is that no arm or curve comparison is readable until every arm sits at its own
   swept threshold. A +0.0119 gap is comfortably inside the range that rule exists to
   protect against. **This result is provisional until swept, and the checkpoints needed
   to sweep it no longer exist.**
2. **One seed, no variance estimate.** +0.0119 is ~2.7% relative on a single run.
3. **The picture is mixed, not clean.** B wins strict argument but *loses* to C on triggers
   (C best at 0.9369) and to plain A on argument fair/relaxed and event type. Winning
   strict while losing relaxed means the spans land more exactly without more of them being
   found — sharper boundaries, not better role routing. That is the opposite of the
   noun-phrase-routing failure item 12 was proposed to fix, and it should temper any claim
   that base-word supervision addresses `convoy` as a `victim`.

The validation curves said the opposite, which is itself worth recording: C tracked B
closely at every shared epoch (C ahead at 1–2, within ~0.006 thereafter), implying the gain
was duplication. The blind test reversed it. Validation and blind test also diverged on
arm A alone (val 0.6797, blind-test strict argument 0.4474) — read the blind test.

#### On a combined arm D — HOLD, and make it conditional on B vs C

Do not plan D now. It is worth running in exactly one of three outcomes:

| B vs C | reading | D worth it? |
|---|---|---|
| B > C | lemmatization adds something beyond duplication | yes — the dosage question is real |
| B ≈ C | the gain is duplication, not lemma | no — D adds more of what did not help |
| B < C | lemmatization actively hurts | no — D would hurt more |

**And D would need its own control or it reproduces the confound C exists to remove.** A
combined arm is originals + lemma + verbatim = **19,253** records against B and C's 13,291,
so D-vs-B differs in record COUNT as well as composition and a gain is again
unattributable. The matched control is arm E: originals + *two* verbatim copies, also
19,253. That is two runs (~2.5h, ~$3.30) to answer a dosage question, and only after
B > C is established.

Note also that B and C are not two treatments to combine. **C is a control — the null
version of B.** Combining a treatment with its own control is "double the augmentation",
not a factorial design; a genuine 2x2 needs a second *factor*, not a second dose.

**Higher-value uses of the same GPU hour**, both of which are the confluence rather than
more dosage:

- **base-word applied to the casualty/event line** — tests whether the lever generalizes
  off RAMS, which is what would justify it in the papers.
- **GIST on RAMS** — the literal river-join with item 11. The RAMS guide cache is being
  built (see item 11). It carries a concrete prediction to test rather than assume: the
  veto drops mined negatives the guide judges positive, and base-word negatives ARE mined
  negatives, so a frozen guide carrying the same NP-routing bug will score `convoy` highly
  for a `victim` query and **veto exactly the negative item 12 exists to add**. That says
  the merge order must be: measure base-word alone, then add GIST, then check whether the
  base-word gain survives. Merging first gives a null nobody can attribute.

The remainder of this item is the original specification, kept because it states the
constraints the implementation had to satisfy.



**Proposal.** For each training sample, emit a **second** sample in which every surface word
in *both* the text and the labelled spans is reduced to its base form. Surface and normalized
variants both stay in the mix (1:1 duplication, not replacement). Reported from prior
practice as helping training substantially. *Not measured in this repo.*

Prior art, if replicating: PURE (Princeton) is recalled as doing a **partial** version of
this **in its code rather than its paper** — reportedly inherited from the DyGIE/DyGIE++
preprocessing it reuses. Recollection is several years old and unverified here; do not go
looking in the PURE paper's method section for it, which is where this note originally went
wrong.

**Why it is plausible here specifically.** The event corpora are small — RAMS 7,329 train,
CASIE 795, WikiEvents 206 — while role fillers and triggers inflect freely (`killed` /
`killing` / `kills`). Normalizing collapses those into one form, so a trigger–role
association is learned once instead of three times under-powered. It is also a second angle
on the noun-phrase routing in item 2: normalization strips the morphological cue the model
may be latching onto instead of the role semantics.

**The constraint that decides whether this works: spans must stay verbatim.** Boundary
collation locates each gold surface inside the text; a mention that cannot be aligned is
**silently dropped** under `on_missing_surface="skip"` (counted in
`missing_surface_counts()`, `boundary_preprocessing.py`). So the failure mode is not an
exception — it is quietly reduced supervision, which looks like "augmentation didn't help".

The rule that avoids it: **lemmatize the token sequence ONCE, then re-derive every label from
its token offsets.** Never lemmatize the text and the label string independently — lemmas are
context-sensitive (`left` → `leave` or `left`), so the two passes diverge and the label stops
matching. Verified today that `text_tokens[start:end]` reconstructs gold surfaces exactly
(69/69, and cleanly under truncation), which is the property an offset-based rewrite must
preserve.

**Acceptance gate, cheap and decisive:** run the augmented corpus through the collator and
assert `missing_surface_counts()` gains **zero** entries relative to the un-augmented run. If
it gains any, the alignment is broken and the measurement that follows is meaningless.

**Language gating.** The mix is multilingual (mmBERT; CMNEE/DuEE/ChFinAnn Chinese, KLUE
Korean, MasakhaNER across 20 African languages). Lemmatization is a no-op for Chinese and a
different operation for agglutinative languages, so this must be opt-in per corpus rather
than applied across `data/`. No lemmatizer is currently a dependency — a dictionary-based,
token-wise one (no per-language model download, deterministic) is the right shape, because
token-wise is exactly what the alignment rule above requires.

**Write path.** Any new emitter must route through `_split.dumps_record`, per the repo rule —
NFKC plus line-separator stripping, `ensure_ascii=False`.

---

## What the metrics fixes did and did not touch (2026-08-14)

Two eval-side defects were fixed. Neither reaches training, and the blast radius was
measured rather than assumed, so **no number in this file needs redoing**.

**`c0ab89c` — `metric_for_best` silently fell back to `eval_loss`.** A run configured to
maximize an F1 maximized loss instead. Now raises.

**`7586411` — `_schema_from_gold` dropped `entity_descriptions`.** Corpora that name types
`e_0`/`e_1` and carry the meaning in a parallel map were scored by asking for the empty
label. On 100 `pile_ner_def` val records against pristine `fastino/gliner2-base-v1`, strict
entity F1 0.0174 without the map against 0.5381 with it; recall 0.0092 → 0.4771.

(`bbacce6` claimed this fix and was a **no-op** — it put the map under `schema["entities"]`
as the values, which are label targets, not prompt text. Cite `7586411`, not `bbacce6`.)

**Selection was never affected.** Every training config except `eval-preservation-ner.yaml`
selects on `eval_loss`, which the trainer computes from the forward pass
(`trainer.py:2109`) and which never passes through `_schema_from_gold`.

**Blind-test reach**, as share of each config's test records carrying `entities` **and**
`entity_descriptions`:

| config | affected |
|---|--:|
| `eval-preservation-ner` | 78.5% (4,715/6,003) — built 08-13, never had a valid number before |
| `mmbert-base` | **49.3%** (106,657/216,154) — its blind-test entity row is understated |
| `joint-boundary-mmbert-{10k,40k,100k}` | 0.5% |
| `warmstart-{natural,anchorless,struct}`, `mmbert-137k`, `natural-gist` | 0.2% |

The record-mode A/B (item 11's control, `0ca9447`) is **0.2%** and stands:
`data/mix_natural.test.jsonl` is 0 bytes, so `mix_natural` contributes a val split only and
its 35.5% description share never reached a blind test. No working paper quotes an entity
number from `pile_ner_def`, `nuner_full` or `pubmed_abstracts_ner` — the headline numbers
are event metrics on RAMS, which carries no descriptions.

---

## Notes for whoever picks this up

- **RESOLVED (2026-08-17): the "~49% stall" is not a hang — it is one 6.5-minute test.**
  `test_public_api_e2e_real_deberta.py::test_boundary_public_api_lifecycle_real_deberta`
  takes **390.5s standalone and passes** (`--durations`), and it lands at the 52% mark in the
  combined run. There is no deadlock and no cross-test pollution: the suite was simply sitting
  in a slow test with no timeout, on a machine also holding a 15GB job.

  Four tests carry `@pytest.mark.slow`, all real-DeBERTa; three exceed 120s. They are slow
  because `DebertaV2Model` rejects `sdpa` and falls back to **eager** attention (the loader
  warns), so a real training loop runs unaccelerated on CPU.

  **Run this and the problem disappears** — the whole suite in ONE process, no chunking:

  ```
  uv run pytest tests/models/boundary tests/processing -m "not slow" --timeout=120
  ```

  → **328 passed, 4 skipped, 4 deselected in ~30s.** With the slow tests included it is
  454s and the three time out. Keep `--timeout` on in CI so a slow test reports as a failure
  with a stack instead of looking like a hang.

  Diagnosis cost one command; `pytest-timeout>=2.1` was already in the dev group. The earlier
  note that this "needs a machine not already holding a 15GB job" was wrong — you never needed
  the suite to *finish*, only to hang, which it already did reliably.

  Two things fixed on the way: `pythonpath = ["."]` is now set in `pyproject.toml`, because
  `tests/` is not a package while `tests/conftest.py` imports `tests.fixtures` — plain
  `pytest` used to die at *collection* with `ModuleNotFoundError: No module named 'tests'`
  and only `python -m pytest` worked. Both invocations work now.

- **Summarizer-as-segmenter was tested and is not the answer** (`bullet_premise_test.py`).
  Hand-written bullets on 5 real Helene sentences, rollup-aware scoring: raw text 3/5 with
  1 false positive; *free* bullets 2/5 with 3 FP and **2 fabricated figures**; *extractive*
  bullets (every digit copied from source) 3/5 with 1 FP and 0 fabrications. Restructuring
  does not improve attachment on this corpus. The free variant actively harms — its most
  useful act, turning "they died together" into "2 people died", is exactly what a
  verbatim-number guard must reject, so guard and summarizer are in direct tension. Also
  note the corpus does NOT contain the tidy "120 NC / 17 TN / 227 total" sentence everyone
  reaches for; the real numbers are distances, populations, years and rainfall.
- **Everything new is off by default.** `--rollup`, `--event-year`, `--record-mode` and
  `--associate envelope` all have to be passed explicitly on `run_pipeline.py`. The defaults
  reproduce the older numbers, on purpose.
- **`probe_records.py` is the record-extraction check, not the blind test.** The blind test
  scores tasks; it does not tell you whether record mode is emitting the fields you think.
- `datasets/helene2024/_cache/` and `datasets/turkey2023/_cache/` hold harvested article text,
  are gitignored by design, and both harvesters regenerate from the Wayback archive.
- The anchorless arm is deliberately **not** published: it learned nothing (1 of 9 instances),
  so it is evidence for the papers rather than an artifact worth shipping. The natural arm is
  on the Hub as `whr778/gliner2-joint-boundary-warmstart-natural`, private.
