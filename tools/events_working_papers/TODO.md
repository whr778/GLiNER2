# Open items

**Only outstanding work lives here.** Everything resolved, refuted or superseded is in
[[EXPERIMENT_CATALOG]] (what was run, newest first) and [[PROJECT_HISTORY]]
(narrative, including [the full pre-2026-09-21 TODO](PROJECT_HISTORY.md#todo-archive-2026-09-21)
preserved verbatim). Last rewritten 2026-09-21.

## Outstanding, at a glance

| # | item | why it matters | state | next action |
|---|---|---|---|---|
| 1 | **absneg4: absent pool scoped to event roles** | the only lever that has ever moved `event_argument` outside the floor (+0.0376), minus the collateral | **RUNNING** since 16:38 UTC 2026-09-21, ~19h, A100 `ceeee75126414…` | read it against BOTH absneg2 arms |
| 2 | **eb17-best: the warm-start base** | folds in every measured lesson; adds label negatives, which four dead mechanisms have waited on | BUILT, not launched, ~14h ~$30 | launch once (1) lands |
| 3 | **classification collapses under interventions it never targets** | −0.1977 on absneg2, −0.403 on warm cells, −0.1492 before that; blocks shipping negatives | ROOT CAUSE NARROWED: 96% is `docee_event` alone | see whether (1) spares it; else isolate docee |
| 4 | **`record_anchor_threshold` defaults to 0.5** | nothing calibrates record cutoffs; no model can use 0.5 well | open, unstarted | sweep on validation like the record threshold was |
| 5 | **Cross-event contamination** | 6 of 86 audited Helene `dead` observations belong to OTHER events | **keying root cause FIXED at source; both anchors turn out to ALREADY EXIST as `--scope-filter`/`--event-year`; the cached artefact is IRREPRODUCIBLE** | do NOT overwrite the cache; re-score `--scope-filter` on corrected labels |
| 6 | **`candidate_pool: shared` A/B** | `per_query` makes entity and event queries compete for candidate budget; adding an entity menu costs 30-35% of argument recall | **RUNNING**, arm 1 of 3 on epoch 2/2. **GATE PASSED: shared-pool grad norm 3.853e+00** (per_query reads 0.000e+00), so the treatment is live. ETA ~20:55 UTC, ~$6 | compare the three arms |
| 7 | ~~Purchased NER gold idle~~ **DONE** | swapped into eb17-best; train entity mentions 707,007 → 905,691 (**+28.1%**) with the blind test byte-identical | **DONE 2026-09-21** | — |
| 8 | ~~`turkish_event` lemmatised gold~~ **DONE** | train+val 93.02% → **99.83%** aligned, **9,648 mentions recovered**, 88 unsafe repairs refused | **DONE 2026-09-21** | re-upload the HF mirror when convenient |
| 9 | **Relation warm-start regression (−0.037, −22%)** | `task_lr: 5.0e-4` was tuned for COLD heads | hypothesis unverified | try a lower task_lr on the warm stage |
| 10 | **EKF: §10 crux reopened, §14 does not reproduce** | the EKF's claimed edge under unreliability | open | re-derive on real streams |
| 11 | **EKF: exposure counts are not casualties** | 12 of Helene's 106 `dead` are audited non-casualty; schema has nowhere to put them | open | extend the schema |
| 12 | **No benchmark that can score the filter** | Türkiye's baseline was an oracle by construction | open | build a non-oracle benchmark |

---

## 1. In flight

**absneg4 (`absneg4-roles.yaml`)** — `absent_negatives_scope: roles` excludes the event ANCHOR
query from the absent pool. absneg2 showed the two halves of the `events` bucket move in
opposite directions: `event_argument` recall +0.0329 against `event_type` recall −0.0818 and
`event_trigger` −0.0468. The Ortmann decomposition says the argument gain is **754 gold
arguments leaving "never found"**, which happened DESPITE fewer instances — so it does not
depend on the suppression.

Read it against **both** absneg2 arms (same operating point, same 20,602-record test set):
vs `absneg2-control`, is pooling better than none? vs `absneg2-treatment`, did scoping remove
the collateral? **A null is informative** — it would mean the argument gain did depend on
instance suppression, and the lever is dead.

## 2. Ready to launch

**eb17-best** (`tools/train/config/base/eb17-best.yaml`) — vanilla mmBERT → the warm-start
base. Four changes against `eb16-eventrecords-tr`, each forced by a measurement and all
documented in the file: task-metric selection instead of `eval_loss`, threshold 0.3, the split
gate on, and **label negatives added**. That last is the substantive one: no base this project
has trained has ever had a single absent query (0 in 574 measured), so `null_loss`,
`count_loss`, `negative_query_ratio` and `abstention_loss`'s entire positive class have been
supervised on an empty set in every model to date.

Hold until absneg4 lands — `absent_negatives_in_denominator` is exactly a base-level decision.

**Selection now has an aggregate.** `eval_overall_{strict,relaxed}_{micro_f1, head_macro_f1,
head_min_f1}` landed 2026-09-21. `head_min_f1` is the interesting one for a base: it cannot be
improved by trading one head away, which is the failure mode this programme keeps hitting.
Consider it for `metric_for_best` in place of entity F1.

## 3. P0 — blocks the next experiment

**`record_anchor_threshold` defaults to 0.5 and nothing calibrates it.** The record threshold
itself was swept and moved the 137k structure reference to 0.1119 at 0.1; the anchor cutoff
never got the same treatment.

### Cross-event contamination — not blocked, and the readout is already fixed

Whole-article reading via `extract_long` binds casualty figures lifted from unrelated stories
sharing an article body.

**TWO STALE BLOCKERS, both cleared.** The entry read "Gated by item 0 (2026-08-17)" -- item 0
CLOSED 2026-08-19. The 2026-09-21 rewrite additionally mis-cited it as "item 4". And "fix the
unsound readout first" was done twice over, in `tools/ekf_showcase/event_binding_probe.py`:

- **2026-08-17, the scorer.** Signal C scored `bool(bound) and not OURS.search(bound)`, so it
  fired on any non-empty string lacking "helene". A casualty-trained record head has no
  `event` field in distribution and copies the anchor number in, so `'230'` counted as a
  caught cross-event and C read 9/11 on pure artifact. Now `validate_binding` keeps a binding
  only if it names an event the schema ALSO found, `_NUMERIC` rejects copied numbers, the
  `recs[0]` fallback is gone, and "bound nothing" is reported separately from "bound ours".
- **2026-08-20, the LABELS.** The ground truth itself was **27% correct on its own positive
  class** (3 of 11). All six `'230'`s were marked cross-event (Milton) when every one is
  Helene's OWN national total (truth 228, windows read "Helene death toll hits 230"); `'250'`
  likewise; one `'1,400'` is "1,400 LANDSLIDES", not people. It MISSED Maria's 3,000, the 1916
  hurricanes' 80 and a Taiwan typhoon's "dozens".

**CONSEQUENCE: every published score on this instrument is uninterpretable.** A detector
reading 3/11 might have found the three real cases or three of the eight false ones. That
includes the three signal results still quoted in `EKF_MHT_BUILD_RECORD.md` §27.2 (nearest
3/11 at 32.5% FP, only-competitor 3/11 at 31.3%, bound 2/11 at 26.5%) and the 2026-08-11
"4.7% cross-event of 106" figure. **Do not reuse either.**

**THE CORRECTED GROUND TRUTH** lives in `tools/ekf_showcase/helene_audit_labels.json`, keyed
by `sha1(value|normalised context)` so a label survives re-extraction. 86 occurrences:
64 helene, 11 non-casualty, **6 cross-event**, 5 unclear.

| span | belongs to |
|---|---|
| 3,000 | Hurricane Maria, Puerto Rico |
| 1,400 | Hurricane Katrina |
| 80 | the **1916** Appalachian hurricanes |
| dozens | Typhoon heading to Taiwan (injuries, not deaths) |
| at least 16 | Bosnia floods/landslides |
| at least two | Hurricane John, Mexico |

**EXTERNAL ANCHORS LOOK RIGHT, and the two are complementary.** Against the corrected six:
spatial (outside Helene's six-state footprint) catches 5/6, temporal (outside Sept 2024)
catches 3/6, **composed 6/6** -- the one spatial misses is the 1916 Appalachian hurricanes,
IN the footprint and 108 years early, which is exactly what temporal is for. This matches the
pattern that external-anchor checks work where text-proximity inference does not: the temporal
filter already took Izmit from 15 false bindings to 3 with zero genuine losses, while all
three text signals infer ownership from the words near the number and Helene articles
routinely name Milton and Katrina for comparison.

**CAVEAT, and it is the one that matters.** That 6/6 is a feasibility analysis over the places
and dates recorded in the audit's own `why` fields -- NOT an end-to-end run. A real filter must
EXTRACT place and date per observation, and that extraction can fail. **The false-positive rate
against the 64 genuine observations is unmeasured, and FP is precisely where all three prior
signals died** (3/11 recall at 26-32% FP). Recall on six cases proves nothing on its own.

### RE-RUN 2026-09-21, against the corrected labels -- the text signals are dominated

`whr778/gliner2-base-v1-casualty-docee`, 104 observations, audit classes
{cross-event 6, helene 82, non-casualty 12, unclear 4}:

| signal | catches cross-event | FP on genuine Helene |
|---|---|---|
| A nearest is a competitor | 3/6 | 19/82 = **23.2%** |
| B only a competitor named | 3/6 | 16/82 = **19.5%** |
| **C bound event is a competitor** | **0/6** | 1/82 = 1.2% |
| C-raw (unsound, diagnostic) | 3/6 | 32/82 = 39.0% |

**C IS DEAD.** Honestly scored it catches NOTHING -- for all six cross-event cases it either
bound nothing (3) or named something the schema never found (3). Its old 9/11 was pure
artifact, and the fix is confirmed working by the fact that C-raw still reads 3/6 at 39% FP
beside it.

**A AND B CAP AT 3/6, and the per-case detail says why:**

    '1,400'        nearest='Hurricane Katrina'    caught
    '3,000'        nearest='Hurricane Maria'      caught
    'at least two' nearest='John'                 caught
    'at least 16'  nearest='Hurricane Helene'     Bosnia is a PLACE, not a storm
    'dozens'       nearest='Hurricane Helene'     Taiwan typhoon, not named nearby
    '80'           nearest='None'  events=[]      1916 -- names no event at all

**THE DECIDING RESULT: A/B's three catches are a STRICT SUBSET of spatial's five.**
Katrina -> Louisiana, Maria -> Puerto Rico, John -> Mexico are all outside the six-state
footprint, so spatial gets those three PLUS Bosnia and Taiwan, and temporal gets the 1916
case. The text signals find less, at 19.5-23.2% false positives -- roughly one genuine
observation in five discarded. **They are dominated and should not be shipped.**

### SPATIAL ANCHOR BUILT 2026-09-21 -- `tools/ekf_showcase/spatial_anchor.py`

The pipeline ALREADY binds every observation to a place; that is what `event_key` is. So the
anchor needs no extraction and no model call: is that place inside the event's footprint? The
footprint and its 38 aliases come from the event's own `rollup.json`, not an invented
gazetteer -- `hierarchy.parts` is the six states, aliases map `asheville` to `north carolina`
and `carolinas` to `__aggregate__`.

| signal | catches cross-event | FP on genuine | model call |
|---|---|---|---|
| A nearest is a competitor | 3/6 | 23.2% | no |
| B only a competitor named | 3/6 | 19.5% | no |
| C bound event is a competitor | 0/6 | 1.2% | yes |
| **SPATIAL** | **4/6** | **1.2%** | **no** |

**Strictly better than A and B on BOTH axes**, and cheaper than C, which catches nothing.

    at least two   event_key=mexico                 FLAG
    3,000          event_key=puerto rico            FLAG
    at least 16    event_key=bosnia                 FLAG
    1,400          event_key=reading pennsylvania   FLAG   (Katrina)
    dozens         event_key=tennessee              pass   <- MIS-ASSOCIATED upstream
    80             event_key=north carolina         pass   <- 1916, TEMPORAL's job

**ABSTAINING IS LOAD-BEARING.** `event_key` is sometimes a TYPE (`Storm`, `Floods`) rather
than a place -- `collapse_type` territory. Those carry no spatial evidence and are passed,
never flagged; 5 genuine observations abstain this way. Flagging them would manufacture
exactly the false positives that make A and B unshippable.

**THE ONE FALSE POSITIVE IS ALSO AN UPSTREAM BUG**, not a filter error: a genuine Helene
figure of 180 is keyed `scotland`. Both remaining misses and the single FP are association
errors, so the ceiling here is set by the keying, not by the test over it.

### TEMPORAL ANCHOR ADDED -- one case, and the naive version does NOT work

| anchor | catches | FP on genuine |
|---|---|---|
| SPATIAL | 4/6 | 1/81 = 1.2% |
| TEMPORAL (year < 1950) | 1/6 | 0/81 = 0.0% |
| **COMBINED** | **5/6** | **1/81 = 1.2%** |

**The permissive version reproduces the A/B failure exactly.** 10 of 81 GENUINE Helene
observations carry a non-2024 year, essentially all in a comparative clause -- "Helene is
already the deadliest hurricane to hit the mainland U.S. since Katrina in 2005", "Helene
passed the 35 killed after Hurricane Hugo" (1989). **The year is attached to the COMPARISON,
not to the figure**, which is the same proximity-is-not-attachment problem that caps A and B:

    a year < 2010 in context    2/6    9/81 = 11.1% FP
    a year < 2000               1/6    2/81
    a year < 1950               1/6    0/81

Only an ANCIENT year survives. `temporal_flag` therefore returns True or ABSTAINS and never
False -- absence of an old year is not evidence a figure is current, so the signal can only
ADD to spatial, never overrule it.

**STATE THE EVIDENCE HONESTLY: this rests on ONE positive** (`'80'`, the 1916 Appalachian
hurricanes). The sharpness of the threshold -- 11.1% FP at 2010 against 0% at 1950 -- is
itself a fragility signal. It is a conservative complement to the spatial anchor, not a
validated signal in its own right, and it adds exactly one case.

**WHAT REMAINED WAS NOT A DETECTION PROBLEM -- and the cause is now found and fixed.**
The single miss and the single false positive were both upstream association errors, and they
share ONE mechanism: **AP embeds a rail of unrelated headlines INSIDE the story body**, and it
is flattened into the text with no separator.

    dozens   keyed `tennessee`  -- "RELATED COVERAGE Typhoon headed to Taiwan injures
                                    dozens ... 11 workers at a Tennessee factory ..."
    180      keyed `scotland`   -- "... paired at pro-am event in Scotland More than 180
                                    people have been killed from Hurricane Helene ..."

A genuine Helene figure took its place from a GOLF headline. The boundary is unrecoverable
downstream -- the headlines run together with no punctuation -- so the fix belongs at
HTML->text time, where the rail is a DOM node (`build_helene_feed.plain`).

**TWO TRAPS, both measured before committing:**

- **Strip by text and you lose more than you gain.** 74% of articles carry the marker, and
  11 of 106 `dead` observations sit within 400 chars after one -- 7 of them GENUINE. Dropping
  everything after the marker would discard 7 real observations to remove 2 cross-event ones.
- **`contains(@class,"Enhancement")` is too broad.** It also matches `LinkEnhancement`, an
  inline link inside the prose; stripping those deletes real article words ("Broadway",
  "assassination attempts"). Measured 26-48 nodes per article against 1-2 for the block
  selector. The committed xpath targets `@data-gtm-region="RELATED COVERAGE"` and
  `div[contains(@class,"PageListEnhancement")]` only.

**A SECOND LATENT DEFECT, found in passing: `lxml` was never declared.** Without it `plain()`
fell back to stripping every tag with NO structural selection, returning navigation, rails
and footer as article text -- a 26,598-character median document against 5,100, which is how
an article about a four-day workweek comes to "name" Florida and Georgia. It now RAISES
instead of silently producing a feed worth nothing, and lxml is a declared dependency.

**VERIFIED on the real article**: `RELATED COVERAGE`, `Scotland` and `Typhoon headed to
Taiwan` are all gone, and the 180 figure now sits in clean body prose.

### THE REBUILD WAS ATTEMPTED AND MUST NOT BE INSTALLED

The feed rebuilt cleanly and offline from cached HTML: 70 -> 65 rows, 375,609 -> 332,419
chars, and `RELATED COVERAGE` present in 52 rows -> **0**. **The 5 dropped articles are a
CORRECTION**, every one off-topic and qualifying only because the rail injected Helene states
and toll words into it -- "Georgia Supreme Court restores near-ban on abortions"
(has_toll=True, states=[NC, GA]), "Elon Musk makes first appearance at Trump rally"
(has_toll=True, states=[NC, SC]).

**But the pipeline re-run cannot be installed, and a CONTROL is what showed why:**

| run | dead obs | joined to audit labels | cross-event |
|---|---|---|---|
| committed artefact | 106 | 102 | **6** |
| control: OLD feed, current config | 47 | 27 | **0** |
| rebuilt: new feed, current config | 44 | 22 | **0** |

**The current configuration extracts NONE of the six cross-event observations -- from the old
feed either.** So the text change costs 3 observations (47 -> 44), not 62; the other 59 and
all six positives are lost to a configuration difference. Overwriting the cache would destroy
the basis for every cross-event measurement and orphan 80 of 86 hand-assigned audit labels.

**The committed `tracked_rollup.json` is IRREPRODUCIBLE.** It records only `feed`,
`associate` and counts -- no models, thresholds or flags. `run_pipeline` now writes an
`invocation` block (args, argv, git_commit) and the artefact I produced carries it, but the
committed one predates that. Same class as the `eval_provenance` defect, in a different
pipeline, and already fixed going forward.

### BOTH ANCHORS ALREADY EXISTED -- I duplicated them

    --scope-filter   "drop observations keyed outside the rollup's declared hierarchy.
                      Needs --rollup. 4/6 cross-event at 7.3% FP on Helene, no model."
    --event-year     "reject casualty figures whose nearest date predates this year ...
                      The 1999 Izmit toll was tracked as a 2023 figure in every Turkiye
                      configuration until this existed"

`--scope-filter` IS the spatial anchor; `--event-year` IS the temporal one, and better
engineered (nearest date to the span, with slack). I missed them by searching the probe files
and "association" rather than the pipeline's flag list.

**What survives as new work:** the `--scope-filter` figure of **7.3% FP predates the
2026-08-20 label correction**; re-scored against the corrected labels it is **4/6 at 1.2%**.
That is the same uninterpretability that voided the three text signals. `spatial_anchor.py`
is therefore worth keeping as a SCORER against the audit labels -- which no flag does -- but
its filter logic should defer to `--scope-filter` rather than reimplement it.

**NEXT:** re-score `--scope-filter` and `--event-year` in place, on an artefact that still
carries the six positives. Do NOT regenerate that artefact until the configuration which
produced 106 observations is identified.

31 tests.

## 4. Open experiments, cheap

- **`candidate_pool: shared`** — RUNNING since 2026-09-21 17:55 UTC on one A100.
  **The gate has PASSED on arm 1: `shared-pool grad norm 3.853e+00`.** That matters more than
  it looks: `shared_pool_builder` exists in every checkpoint and receives NO gradient under
  `per_query`, so an arm that failed to switch is indistinguishable from one that switched and
  did nothing — attempt one died on exactly that confusion. A non-zero reading proves the
  treatment is live before any result is believed.

  Three arms (shared 2ep, perquery 2ep, shared-long 4ep), 1,260 steps per epoch at ~20 min,
  so ~2.7h and **~$6** — not the ~$4/1.5h first estimated. The control is re-run on the SAME
  box deliberately: attempt one's control was bf16 on an A10, so reusing it would confound
  card with treatment. Note the script's header comment claims it runs only the two `shared`
  arms; `ARMS` actually defaults to three. The code is right, the comment is stale.
- **Negative documents** (data, not decode) for cross-event contamination.
- **Base-word positive/negative samples** — a different granularity, aimed at noun-phrase
  boundaries rather than event interference.
- **A lower `task_lr` on warm stages** — 5.0e-4 was tuned for cold heads and the relation head
  regresses −0.037 in the warm start.

## 5. Data debt

- **The purchased NER gold: both reasons for leaving it idle turned out to be wrong**
  (checked 2026-09-21).

  **1. The flag already exists and is already in use.** `partial_annotation` is documented in
  `build_negative_pools.py` and live in roles2/roles3: a corpus declared partial for a
  dimension "contributes POSITIVES to the dimension while being refused as a source of
  NEGATIVES for it". Nothing needs building.

  **2. It is a corpus SWAP, not an addition, and the exhaustiveness bar is already met.**
  `cmnee_ner` IS `cmnee` plus entities -- the same 9,281 documents (99.9% overlap), the same
  19,422 events -- and `cmnee` currently contributes **ZERO** entity mentions. Same for
  `duee_ner`/`duee`. Swapping adds **167,315 entity mentions, +23.7%** of the base's entity
  supervision, already paid for at $6.92.

  Probed for exhaustiveness *within an offered label*, which is the only risk that matters
  because the menu is built from the record's own gold keys:

  | corpus | annotated/doc | missed/doc | missed as % of annotated |
  |---|---|---|---|
  | cmnee_ner | 13.0 | 4.23 | **32.4%** |
  | duee_ner | 4.1 | 0.29 | **7.1%** |
  | **docee (already used as supervision)** | 8.7 | 2.87 | **33.0%** |

  `cmnee_ner` is exactly as exhaustive as `docee`, which supplies 26.9% of the base's entity
  mentions today; `duee_ner` is 4.6x cleaner than both. So the bar excluding them is one the
  incumbent mix does not clear either. The probe is an UPPER BOUND -- a string match is not
  proof of the same entity in context, and this methodology once read 35.8% loose against
  0.25-0.6% tightened -- but the comparison between corpora is fair, being the same probe at
  the same settings.

  **Note the `entity_types` decision still stands and is different.** `merge_entity_types.py`
  writes a typing SIGNAL rather than an `entities` block because it attaches types to
  argument spans for the typed margin; that is not the same as using a corpus's own entity
  gold as supervision.
- **`turkish_event` REPAIRED 2026-09-21** (`tools/data/repair_turkish_surfaces.py`). The
  `max_len` diagnosis was wrong: the annotator lemmatised the LABELS but not the TEXT, so
  gold read `mesafe` where the document reads `mesafenin`. 234 of 235 failures were the gold
  surface being a prefix of a longer word, every failing position well inside the window,
  0 case-only, 0 genuinely absent.

  Train+val went **93.02% -> 99.83% aligned, 9,648 mentions recovered**; 88 derivational
  repairs were REFUSED and left for `skip` (`futbol` -> `futbolcunun` is football -> of the
  footballer). TEST DELIBERATELY UNTOUCHED and verified byte-identical -- repairing it would
  move the blind set, and would buy nothing, since those surfaces score zero either way.

  **`tools/data/augment_baseword.py` had already written down this exact failure**: "never
  lemmatize text and label strings in separate passes ... the passes diverge and the label
  stops matching", and "a mention it cannot find is silently dropped -- it quietly shrinks
  supervision and reads as 'augmentation did not help'." The RAMS base-word corpus avoids it
  by rebuilding text and labels from the SAME token list. That is also the dup-control
  experiment (lemma beat dup-control +0.0119 strict argument, PROVISIONAL).

  REMAINING: `data/turkish_event.{train,val}.jsonl` now differ from the HF mirror
  `whr778/turkish-event`, which still holds the unrepaired originals. Re-upload when convenient.

- **`biored` aligns at 97.4%** and only 8 of its 126 failures are the prefix pattern, so it has
  a DIFFERENT cause. Not yet diagnosed.

## 6. EKF / casualty line

- **§10's crux is reopened and §14 does not reproduce.** The harder-regime ablation concluded
  the EKF's edge *widens* under unreliability; that does not hold on real streams.
- **Exposure counts are not casualties** — 12 of Helene's 106 `dead` observations are audited
  non-casualty (six are exposure figures), and the schema has nowhere to put them.
- **No benchmark can score the filter.** Türkiye's baseline was an oracle by construction:
  truth read from the sentence the extractor read.
- **Regional disaster profiles as a plausibility prior** — an operator observation, recorded
  as an idea. The flag-not-veto mechanism it needs already exists twice and is switchable.
- **Do NOT buy scope labels under the current scheme** — settled by a $2.25 dual-label probe.

## 7. Research direction

- **Does a fine-tune need an explicit regularizer, or is early stopping enough?**
- **Chunking distorts the real-news arm** — `window_size`/`stride` are **subword tokens, not
  words**, and the configs' comments say "word window". Measured, and not where expected.
- **We cannot compare event arguments to the literature** because we do not compute the
  metric the literature reports.

## 8. Tracked, not dropped — the beam-aware / structured loss (Phase B)

**The argument that made this interesting no longer holds, and the entry used to claim the
opposite.** It read: *"Phase B is now MORE interesting, not less — every measurement of the
beam to date ran with its scalar constraints disconnected."* Two things have happened since:

1. **The disconnected constraints were fixed**, and the decode-arms run then measured that
   **joint matches greedy on all seven heads**. So the null is no longer "a null about a beam
   with its constraints off" — it is a null about a beam with them on.
2. **The strongest candidate constraint was refuted.** TypedRole has almost nothing to act on:
   the probability mass on type-disallowed candidates is median 0.00003 / mean 0.00284, so the
   model already ranks them near zero. See [[OPTION_2_TYPED_ROLE_CONSTRAINTS]].

**What survives is narrower and still real:** the beam decodes JOINTLY over candidate scores
trained GREEDILY, so nothing in training ever optimises for constraint satisfaction. That
train/test mismatch is a genuine structural argument for putting the beam in the loss. But it
is now the *only* argument, and the constraint it would enforce has been measured to be nearly
inert. Treat Phase B as **lower priority than it was**, not higher.
