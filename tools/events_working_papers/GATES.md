# Every gate in the EKF pipeline: what it does, what type it is, and whether it is on

Written 2026-08-25. Three different things in this project are called a "gate" and they
live at different layers. Conflating them has already inverted one verdict for a day
(see the gate-1 lesson at the bottom), so they are separated here by name and by type.

## Type legend

| Type | Meaning |
|---|---|
| **NEURAL** | a forward pass decides it (GLiNER2 head output) |
| **LEARNED** | parameters fitted by gradient descent on a training objective, but not the extractor |
| **THRESHOLD** | a cut applied to a neural score. The score is learned; the cut is hand-set |
| **DECLARED** | reads knowledge a human wrote down (a hierarchy, an ontology). No model, no statistics |
| **HEURISTIC** | hand-written rule over text (keywords, character distance, nearest-match) |
| **ARITHMETIC** | a closed-form inequality on numbers already in the stream |
| **STATISTICAL** | derived from the event's own distribution at run time |
| **DECODE** | one decision over the WHOLE sequence, not per observation. Can revisit |
| **ORACLE** | consults ground truth. Prices a ceiling; can never ship |

---

## Metrics quoted on this page

| metric | units | meaning |
|---|---|---|
| **pooled RMSE** | deaths | one RMSE over every (stream, time-grid) point. The headline |
| **nRMSE** | dimensionless | per-stream RMSE normalised by that stream's range, then averaged |
| **caught / false positives** | counts, % | cross-event figures rejected, against genuine ones wrongly rejected |
| **FP rate** | % | share of clean negatives a gate wrongly admits |

Thresholds and knobs (`--gate-threshold 0.5`, σ=0.3, `reject_cost`) are settings, not
measurements, and are named by their parameter.

## The flow

Article text in, trajectory out. Each stage can only remove or re-route what the stage
above it passed.

```
  raw article feed
        |
  [0] relevance gate ............ NEURAL      ON      drop non-mass-casualty articles
        |
  [1] stage-1 event extraction .. NEURAL      ON      events, envelopes, places
        |
  [2] extract(threshold=) ....... THRESHOLD   ON      candidate span score cut
      abstention head ........... LEARNED     ON      "does this label appear at all?"
      record_*_threshold ........ THRESHOLD   ON      anchor / field selection
        |
  [3] stage-2 casualty records .. NEURAL      ON      value + role + location per record
        |
  [4] normalize() ............... HEURISTIC   ON      qualifier + source from keywords
        |
  [5] out_of_window() ........... HEURISTIC   ON      reject figures dated before the event
      association_key() ......... HEURISTIC   ON      bind a figure to a place
        |
  [6] rollup .................... DECLARED    ON      map places into the event hierarchy
      scope_filter() ............ DECLARED    opt-in  drop places outside the event
        |
  [7] apply_extracted_scope() ... NEURAL      opt-in  route on the model's `scope` field
      hmm_gate() ................ DECODE      SHIP    own / aggregate / reject, globally
      viterbi_gate() ............ DECODE      -       hmm_gate without the extra features
      gate() upward ............. ARITHMETIC  legacy  keep / reroute / drop by magnitude
      gate() downward ........... ARITHMETIC  OFF     drop stale readings (down_ratio)
      hmm_gate4() ............... DECODE      OFF     + a downward-revision state (inert)
      plausibility_filter() ..... HEURISTIC   opt-in  hand-set per-event ceiling
      tail_filter() ............. STATISTICAL opt-in  median + k*MAD on log10, upper tail
        |
  [8] EKF update ................ STATISTICAL ON      relative-noise Kalman filter
      STUDENT_T_NU .............. STATISTICAL OFF     robust one-sided measurement model
      REJECT_SIGMA .............. STATISTICAL OFF     one-sided innovation gate
      MAX_RATE .................. ARITHMETIC  OFF     impossible-accrual-rate filter
      CONF_R .................... STATISTICAL OFF     fold extractor confidence into R
      CENSOR_AT_LEAST ........... ARITHMETIC  ON      `at_least` below estimate is a no-op
      --learn-gate router ....... LEARNED     opt-in  logistic blend of EKF vs last_value
        |
  trajectory per (place, role)
```

---

## Stage by stage

### [0] Relevance gate — NEURAL, on
`run_pipeline.build_gate_schema` -> `.classification("relevance", GATE_LABELS_V2)`.
Zero-shot classification, two labels: `mass_casualty` vs `other`. v1 described the
negative class as topically distant filler and admitted **58.5%** of definitively
non-disaster text at high confidence when benchmarked against 21k real annotated
messages. v2 describes the negatives it actually meets -- personal requests, policy
news, aid-logistics inventories carrying huge numbers, single-casualty medical items --
and names two traps verbatim: a lone death is not a mass-casualty event, and disaster
words used metaphorically ("explosion in crowdfunding").

**The shipped operating point is wrong, and that mattered more than any training change**
*(measured 2026-08-27)*. `--gate-threshold` defaults to 0.5. The trained gate's softmax is
saturated, so 0.5 sits deep inside its positive region: it needs **0.998** to sit at this
project's stated recall bar of ~0.676. Choosing the threshold on the validation split and
scoring the blind test once:

| class | accuracy @0.5 | accuracy @0.998 | n |
|---|---|---|---|
| exposure_only | 0.444 | **0.903** | 72 |
| historical_toll | 0.528 | **0.874** | 159 |
| no_toll | 0.792 | 0.942 | 120 |
| current_toll | 0.933 | 0.673 | 150 |
| **all** | **0.719** | **0.847** | 541 |
| recall on true tolls | 0.938 | 0.696 | 161 |

Paired on the same model, moving the threshold fixes 108 rows and breaks 39
(exact McNemar p = 1.1e-08). `exposure_only` — the open failure two GPU runs and a
four-way auxiliary label were bought to fix — went 0.444 → 0.903 by moving a number. It was
never a capability gap.

**A corollary that retires a model comparison.** Scored at each model's own validation
threshold, `gate2-mmbert-v2` and its predecessor are indistinguishable: 18 rows to 18,
p = 1.0000. The apparent gain from the four-way rebuild was a calibration difference.

**And the gate-model switch does not survive a sweep.** `benchmark_gate.py --sweep` reports
the whole curve from one inference pass, on 1,000 real annotated messages (397 usable
negatives after language-gate drops):

| model | AUC | FP on related=0 @0.5 | recall death\|missing @0.5 | best point |
|---|---|---|---|---|
| `casualty-docee` (span, SHIPPED) | 0.9241 | 0/397 | 20/36 | 0 FP, recall 20 |
| `gate2-mmbert-v2` (boundary) | 0.9472 | 2/397 | 14/36 | 0 FP @0.9, recall 8 |
| **`fastino/gliner2-base-v1`** (span) | **0.9635** | 8/397 | **28/36** | **1/397 @0.998, recall 27** |

The switch to `casualty-docee` (b607fae) was decided on false positives at a single
threshold, 34/410 → 1/410, with no recall column. The false-positive half reproduces; the
conclusion does not. `recall death|missing` is INDICATIVE — the label means "mentions
death", not "reports a toll" — so read the table as a comparison between identically-scored
models, not as absolutes.

**Every external gate number recorded before 2026-08-27 was taken at threshold 0.5** and
means only what one arbitrary operating point means.

**It does not filter non-English at all, and the failure is silent.** Measured on 200 clean
Turkish news articles, English label descriptions against Turkish text:

    fastino/gliner2-base-v1   199/200 = 99.5% false admits   <- THE SHIPPED DEFAULT
    fastino/gliner2-multi-v1   56/200 = 28.0%

The default is DeBERTa-v3, vocab 128,011, English-only. It cannot read the text, so it
answers `mass_casualty` to nearly everything — worse than the 58.5% that forced the v1→v2
rewrite, and nothing in the pipeline reports that the gate has stopped discriminating.

**Superseded 2026-08-27 — a stage-0 language gate now makes that failure loud.**
`tools/ekf_showcase/language_gate.py` (shipped a5fc6a0) detects the language before the
model runs and rejects what the gate cannot read, counted by language in the run log.
Supported set is `{en, zh}`, which is what the corpus contains, plus a Han-script override:
all 18 rows of `data/gate2.test.jsonl` that `lumi_language_id` calls neither `en` nor `zh`
are short Chinese headlines falling to `und` at probability 0.36–0.50. Detection is clean —
1,441/1,441 gate2 test rows supported, 0/300 Turkish articles admitted.

**Also superseded: "translation does not fix it".** That was measured on `multi-v1`, which
reads Turkish and over-admits, so its residual fault genuinely was the label descriptions.
The trained gate `gliner2-gate2-mmbert-v2` has the opposite failure — it cannot read the
language at all — and for that failure translation is decisive. Same 300 articles, both
classes, scored in each language (`gate_translation_ablation.py`):

| condition | AUC | admits pos @0.9 | FP neg @0.9 |
|---|---|---|---|
| Turkish (original) | **0.4733** | 7/100 | 12/200 |
| English (Haiku 4.5 translation) | **0.8359** | 59/100 | 17/200 |

AUC here is the probability a random positive outscores a random negative on the heuristic
Turkish labels; 0.5 is chance, and 0.4733 is below it. **A stored verdict is only valid for
the model it was measured on** — the same experiment gave opposite diagnoses for two models
with opposite failures.

The root cause is the corpus, measured on `data/gate2.train.jsonl` (n=11,781): English
95.9%, Chinese 4.1%, Turkish 0.19% and incidental. mmBERT's *encoder* is multilingual; the
fine-tuning signal never was. A 989-article Turkish adjudication pilot ($0.72, Haiku 4.5
batched) found 428 `current_toll` positives — 43.3% — so a Turkish corpus is not
positive-limited, and the full cue-bearing region is ~2,100 positives for ~$2.84 more.

**A multilingual gate is necessary but not sufficient: stage 2 cannot read Turkish either.**
Fixing stage 0 admits documents into a casualty extractor whose training rows are 121 of
152,578 Turkish (0.079%). It is not silent on them -- it emits confident, wrong figures,
which is the worse failure, because the EKF consumes any arriving number as a measurement
and has no way to reject it. Measured on 60 adjudicated Turkish `current_toll` articles
against 60 real English Helene feed articles, same model, production parameters (threshold
0.3, chunk 200/50), with two gold-free signals so no annotation error enters
(`extractor_language_probe.py`):

| signal | Turkish | English | odds ratio | p (exact Fisher) |
|---|---|---|---|---|
| `location` contains a digit | **78.2%** | 5.8% | 58.5 | 1.5e-39 |
| one value smeared across >=3 fields | **11.8%** | 3.0% | 4.3 | 1.5e-04 |

Evidence "1 kisi hayatini kaybetti, 3 kisi yaralandi" (1 dead, 3 injured) returns
`{'location': '644 bin 439', 'dead': '644 bin 439'}`. Locations are place names; a digit
there means numerals are being bound to whatever field is open. **The English control is
what makes this a language result** rather than a general defect -- and it must stay
attached to the claim, because digit-in-location is independently the known collapse
signature of narrow no-replay fine-tuning (see the replay dosing rule). The collapse mode
pre-exists; Turkish drives the model into it 13x more often than English. So the remedy is
Turkish supervision *plus* 30% exact replay, not Turkish data alone.

A corollary for how gate work is scored: a gate win measured on gate metrics alone can be
net-negative end-to-end, because its reward is admitting documents that stage 2 then
fabricates figures for. This is the same shape as the finding that a form gate scored
best-over-range rewards indiscriminate firing.

**Why a third language, and why Turkish specifically.** The case is typological, not
orthographic. Turkish is Latin script, so three languages here is **two scripts, not
three**; the claim that survives scrutiny is that English (Germanic, fusional-analytic),
Chinese (Sinitic, analytic, logographic) and Turkish (Turkic, agglutinative) span three
families and three morphological types. A recipe that transfers across that is a stronger
generalisation claim than one that adds a second SVO Latin-script language.

The sharper reason is that Turkish is a *controlled* test of the recipe rather than a
confound, and this was checked rather than assumed. A first pass at this argument claimed
agglutinative morphology fragments tolls at the tokenizer, since Turkish declines the
numeral itself (`22'ye yuk` + `seldi`, "rose to 22"). **That is wrong.** mmBERT tokenizes
Turkish cleanly -- `kisi`, `kisinin` and `bin` are single tokens, `22'ye` splits as
`['2','2',"'",'ye']` with the suffix intact, and digits split per character in English
exactly as in Turkish. There is no fragmentation penalty. Tokenizer and encoder already
handle the language; the *only* variable left is supervision. That is what makes Turkish a
clean replication test: a success isolates "the recipe transfers" from "the backbone
happened to know the language", which a language the encoder handled less evenly could not.

**Translation is a diagnostic here, not a shipping path.** The AUC 0.4733 -> 0.8359 result
above proves the signal is present in the text and that the fault is the fine-tuning
corpus. It is not a proposal: document translation is excluded from this pipeline, on the
grounds that a mistranslated toll is indistinguishable downstream from a misread one.
Multilingual capability has to come from the models. What ships is the language gate, which
drops what it cannot read rather than rewriting it.

**Sizing, and the trap in it.** The requirement is positive *extraction examples*, not
documents: at the 42.3% positive rate of the cue-bearing region, 18K documents yields ~7.6K
positives, which lands below the knee of every scaling curve measured here. The relevant
evidence is the RAMS curve, where 10K already delivers ~88% of the 137K result and
single-run variance of +/-0.02 makes nothing past 10K interpretable -- so the knee is at or
below 10K, and the clean tokenizer above predicts transfer should be data-efficient rather
than expensive. Annotation is a one-time cost and subsets are free, whereas each training
run costs GPU; buying enough to train several doses turns the spend into a *curve*, and a
curve -- how much data language N+1 needs -- is what answers whether the second language
was a fluke. A single point only says it worked once.

Check the encoder *and the training corpus* before pointing this gate at any non-English
feed.

**A trained, bar-passing model does not make it into the pipeline by itself (2026-09-01).**
`gliner2-gate2-mmbert-tr` was trained on the Turkish-augmented corpus above and passed its
pre-registered bars 2026-08-29 (Turkish AUC 0.4980 → 0.8105, English pooled-RMSE unchanged
at 17.5, like-for-like). It then sat unwired for three days, because `language_gate.py`'s
`SUPPORTED = frozenset({"en", "zh"})` runs *before* any gate model and hard-rejects Turkish
regardless of which model is behind it — a trained model and a stage-0 allowlist are two
independent things, and fixing one does not touch the other. `--gate-model` also still
defaulted to `gate2-mmbert-v2`, which was never trained to read Turkish at all. Both had to
move together (`2d048ba`): `SUPPORTED` gained `"tr"`, and the default flipped to
`gate2-mmbert-tr`. Neither change alone would have produced a working Turkish feed —
un-blocking the allowlist in front of `v2` would have admitted Turkish text into a model
that cannot read it, reproducing the silent-wrong-figure failure documented above for
stage 2.

**Verified on 60 real, held-out Turkish and 60 real, held-out Chinese documents from
`casualty_ml`'s blind test** (`pipeline_language_test.py`, gate called directly, bypassing
the allowlist so the two models are compared on identical text):

| gate model | Turkish admitted | Chinese admitted |
|---|---|---|
| `gate2-mmbert-v2` (old default) | 21/60 (35%) | 58/60 (97%) |
| `gate2-mmbert-tr` (new default) | **58/60 (97%)** | 50/60 (83%) |

Turkish went from largely unreadable to on par with the corpus's own Turkish test-set AUC.
**Chinese paid for it** — 97% → 83%, an 8-document regression on the same held-out set.
`build_gate_corpus.balance()` only equalises classes *within* a source; it has no term for
the relative weight *across* sources, so adding 4,220 Turkish rows without growing Chinese's
share diluted it. Not a new failure mode — the exact one `balance()` exists to prevent, just
one level up from where it currently operates.

**Stage 2 also improved for both languages independent of the gate**, on the same 60+60
documents, old vs. new casualty extractor (`3e2b357`, matched-instance F1 against gold
`casualty_report`):

| extractor | Turkish F1 | Chinese F1 |
|---|---|---|
| `casualty-docee` (old default) | 0.0847 | 0.1900 |
| `casualty-multilingual` (new default) | **0.2919** | **0.3592** |

Both extraction numbers are still low in absolute terms — precision sits at 0.21–0.26 on
both languages, so this is "reads the language" evidence, not "production-ready" evidence —
but the direction and the fact that BOTH languages move together rules out one language
being carried at the other's expense.

**Closing the Chinese regression: a genuinely three-way corpus, not a bigger Turkish one
(`f066c0b`, 2026-09-04, training in progress).** `gate3-mmbert.yaml` adds
`data/chinese_gate/zh_gate_sample.jsonl` — 4,994 rows, already adjudicated by the same
four-way schema as the Turkish pool, paid for and sitting unused since an earlier phase —
rather than reusing `duee`, whose toll/no-toll split was a heuristic over event-role
arguments, not adjudicated, and which `balance()` drops entirely once three languages'
length deciles shift (13,056 rows, no same-decile counterpart). Its `source` field carried
1,909 distinct per-outlet names, which `balance()`'s per-(source, decile) stratification
would have read as 1,909 near-singleton cells and silently dropped almost all of; collapsed
to one value (`tools/data/normalize_gate_source.py`, mirroring `turkish_news`'s own
convention) before it went anywhere near training. Final train balance: `docee`+`cc_news`
65.5% (en), `turkish_news` 19.3%, `zh_news` 15.2% — 22,164 rows, 11,082/11,082
positive/negative by construction, 0 duplicates, 0 cross-split overlap. Pre-registered
bars: Turkish AUC must not regress below 0.8105; Chinese admission on the same 60 documents
must recover toward 97%; Helene end-to-end must not regress. **RESULT 2026-09-04: both admission bars PASS, and gate3 strictly dominates both
predecessors.** All three gates scored by ONE command on ONE machine over the same 60+60
real held-out `casualty_ml` documents, so no historical number is reused:

| gate | Turkish admitted | Chinese admitted |
|---|---|---|
| `gate2-mmbert-v2` | 21/60 (35%) | 58/60 (97%) |
| `gate2-mmbert-tr` | 58/60 (97%) | 50/60 (83%) |
| **`gate3-mmbert`** | **58/60 (97%)** | **59/60 (98%)** |

Turkish is IDENTICAL to gate2_tr, so the gap that model closed stays closed. Chinese
recovers to 98%, marginally past v2's own 97% -- the three-way rebalance bought Turkish
support at NO Chinese cost, which is precisely what `balance()`'s blindness to relative
weight ACROSS sources had made impossible.

Held-out test set: relevance 0.8084, toll_kind 0.7393, micro 0.7739 (n=2,156). Do NOT read
that against gate2_tr's 0.7920 -- the test set changed with the corpus, and that is the
confounded cross-version comparison this project has already been burned by. The admission
table above is the matched comparison.

Selected at epoch 4 of 8 (val 0.7964); epochs 5-8 never beat it while train loss kept
falling 0.1423 -> 0.0613. The 8-epoch schedule is longer than this corpus needs, and
`save_best` is what made that harmless.

**THE THIRD BAR FAILS, and gate3 is therefore NOT shipped as the default.** Helene, real
wire copy, same feed and truth, only the gate model varying:

| gate | admitted | dead obs | pooled RMSE | final estimate (truth 233) |
|---|---|---|---|---|
| `gate2-mmbert-tr` | 59/70 | 40 | **132.59** | 97.8 |
| `gate3-mmbert` | 53/70 | 37 | **175.66** | **1.1** |

RMSE 32% worse and the track effectively dies -- the final estimate collapses to 1.1
because gate3 drops the LAST observation and is left standing on a run of spurious 1s.

The dropped documents were read, not assumed. Of the 8 gate3 rejects, at least two carry a
GENUINE current toll:

    t=185.4h  "The death toll has topped 200 after the Category 4 storm rolled through
               the southeast last week"                                       -> 200
    t=726.9h  "There have been 98 reported deaths in North Carolina from the storm,
               according to state officials"                                  -> 98

Both are recovery-framed pieces (a sports relief drive; a $600m funding approval) that
nonetheless report a current toll, which `GATE_LABELS_V2`'s own `current_toll` definition
explicitly covers -- "includes coverage of the ongoing aftermath, rescue or investigation
of that event". So these are recall MISSES, not correct rejections of aid filler. It would
have been easy, and wrong, to read them as the gate improving.

**The cause is the same defect that gate3 was built to fix, one language over.**

| corpus | EN share | TR | ZH |
|---|---|---|---|
| `gate2_tr` | **74.8%** | 22.0% | 3.1% |
| `gate3` | **65.5%** | 19.3% | 15.2% |

gate2_tr diluted Chinese to 3.1% and Chinese admission fell 97% -> 83%. gate3 repaired
Chinese by diluting ENGLISH 74.8% -> 65.5%, and English recall fell. `balance()` equalises
classes WITHIN a source and has no term for relative weight ACROSS sources, so **every
language added silently taxes the others**, and a three-way corpus does not escape this by
being three-way. The fix is a corpus whose per-language shares are fixed by construction
against a target, not whatever the pools happen to contain -- and a fourth language would
hit it again.

**REPLICATION 2026-09-04: the English regression does NOT reproduce, and the verdict above
is withdrawn.** Helene was one feed resting on two documents, so it was run against the two
other English feeds with committed ground truth. Pooled RMSE in deaths:

| feed | gate2-mmbert-tr | gate3-mmbert | admitted | |
|---|---|---|---|---|
| Helene | 132.59 | 175.66 | 59 -> 53 | gate3 worse |
| Aegean | 518.17 | **518.13** | 34 -> **38** | tie |
| Turkiye-EN | 7444.67 | **7444.67** | 16 -> 16 | identical |

Turkiye-EN is the sharpest: both gates admit all 16 articles and emit 58 IDENTICAL
observations. On Aegean gate3 admits MORE than the incumbent and matches its RMSE to 0.04.
So gate3 does not systematically under-admit English, and "gate3 regresses English" is not
established -- one feed out of three, on a 2-document margin, with no seed replication and
no variance estimate for this metric.

**The lesson is about the instrument, not the model.** A single-feed end-to-end number was
treated as a pre-registered bar; it moved 32% on two documents. Bars of that kind need
either several feeds or a variance estimate before a verdict is read off them, and this one
had neither. The share-rebalancing arm that this verdict was about to justify would have
been chasing a signal that does not replicate.

**Standing consequence: a gate must be scored on ALL of its languages before shipping.**
gate3 passes both admission bars and still regresses the pipeline. Two of three bars is not
a pass.

### Does language-MIXED batching train a better multilingual gate? Measured: NO.

**A pre-registered A/B, and the pre-registration is what stopped a bad ship.** Single
variable: `gate3-mmbert-mixed.yaml` is `gate3-mmbert.yaml` with `group_by_length: false`,
verified by diff and by a runtime assertion that the flag reaches `TrainingConfig`. Same
corpus bytes, same seed, same A10. The control was already trained, so this cost one run.

`group_by_length` defaults TRUE and, for a boundary model, replaces shuffle entirely with
`LengthGroupedSampler`. Length tracks language here, so length-grouped batches are
de-facto language-grouped: **14.6% of batches are 100% one language against 3.7% under
random shuffling.**

**Mixed wins BOTH F1 instruments and loses the job.**

| instrument | grouped (control) | mixed | winner |
|---|---|---|---|
| validation micro F1 (selection) | 0.7964 | **0.8095** | mixed +0.0131 |
| blind TEST micro F1, n=4,312 | 0.7739 | **0.7778** | mixed +0.0039 |
| Turkish admission, 60 held-out | **58/60** | 57/60 | grouped |
| **Chinese admission, 60 held-out** | **59/60** | **50/60** | **grouped, decisively** |
| Helene pooled RMSE | 175.66 (n=37) | **138.17** (n=28) | mixed |
| **Aegean pooled RMSE** | **518.13** (n=62) | **1094.71** (n=54) | **grouped, 2.1x** |
| Turkiye-EN pooled RMSE | 7444.67 | 7444.67 | identical |

Mixed drops Chinese admission to **exactly where `gate2-mmbert-tr` sat (50/60)** -- it
reopens the regression gate3 was built to close -- and doubles Aegean RMSE. It admits
fewer documents everywhere (Helene 53->30, Aegean 38->30), so its Helene RMSE gain is a
smaller, luckier sample rather than better tracking.

**gate3-mmbert-mixed does NOT ship.** The pre-registration named this failure mode in
advance -- "a validation win that loses the downstream bars is a LOSS" -- and it fired.
Without those bars a model that beat the control on every F1 number would have shipped.

**The epoch-7 prediction was FALSIFIED.** Peak was epoch 5 (0.8095), with 6-8 all failing
to beat it while training loss fell to 0.0569. The proposed mechanism -- noisier gradients
producing a later, higher peak -- is not what happened.

**And a mechanism offered here was WRONG, retracted rather than defended.** It was claimed
that Chinese suffers because English holds 77.5% of the gradient. That used CHARACTER
share as a proxy for token share. Measured with mmBERT's own tokenizer:

| | rows | chars | TOKENS | chars/token |
|---|---|---|---|---|
| en | 65.0% | 77.6% | **63.7%** | 4.55 |
| tr | 19.4% | 14.7% | **16.2%** | 3.40 |
| zh | 15.6% | **7.6%** | **20.1%** | **1.42** |

Chinese packs 3.2x more tokens per character, so by TOKENS it is 20.1% -- *higher* than
its row share, not starved. **There is currently no validated explanation for the 59->50
drop**, and with n=1 per arm and no seed replication the admission difference cannot be
separated from run-to-run variance either. Recorded as unexplained rather than given a
story the data does not support.

**Consequences for anyone tuning this knob.** Leave `group_by_length` ON. Its measured
cost is ~22% throughput (12.8 vs 16.4 samples/s), NOT the ~2x predicted from padding waste
going 2.5% -> 50.9% -- waste does not translate linearly to wall-clock because `max_len`
caps it and the GPU is not purely FLOP-bound. And balancing batches by CODEPOINTS would
over-correct: it hands Chinese more documents to reach equal character count, pushing its
token share above 20.1%. The principled unit is TOKENS. A sampler that stratifies by
language, length-groups WITHIN each language, then composes batches to a per-language
token budget would get both properties -- that is real work, not a config flag, and it is
unbuilt.

### [2] Extraction thresholds — THRESHOLD, on
`extract(threshold=)` is the single global cut the boundary greedy path gates on.
**Trap:** `Schema().events(trigger_threshold=, argument_threshold=)` is read only by the
*span* engine. On the boundary path those values are silently ignored -- an entire
threshold sweep once ran at the default 0.5 without moving. `record_anchor_threshold`
and `record_field_threshold` default to 0.5 and are separate cuts on the record head.

### [2] Abstention head — LEARNED, on
`enable_abstention=True`, `abstention_threshold=0.5`, trained by `abstention_loss` at
weight 0.2. Per **query**, not per mention: the target is 1 when a label has no mentions
at all. It answers "does `dead` appear in this text?", **not** "should this particular
figure be rejected" -- so it is not a per-observation reject and cannot be used as one.

### [4] normalize() — HEURISTIC, on
`_detect_qualifier` / `_detect_source` keyword rules produce `qualifier`
(point/about/interval/feared/at_least) and `source` (official/major_outlet/preliminary).
This is the pipeline's weakest normalized field: accuracy 0.724 zero-shot, 0.691
after fine-tuning. `--normalizer classify` swaps in a NEURAL alternative and `--normalizer
both` scores them against each other on the same feed.

### [5] out_of_window() — HEURISTIC, on
Takes the year of the nearest date by character distance and rejects the figure when
that year predates the event. Rejects only on **positive** evidence -- an absent or
unparsable date returns None and the figure is kept, because treating absence as
evidence would discard most of the feed. Nearest-by-character-distance is a weak proxy
for attachment and it *failed* for location (both countries within 26 chars of both
numbers on the Turkiye standfirst). It is defensible for dates only because dates are
sparse and clustered, so competing hypotheses sit far apart. Measured: 13/15 Izmit
envelopes resolve to "August 1999", and no genuine observation resolves to an old year.

**Two later fixes, both from feeds that broke an unstated assumption.**

*A feed with no dates at all.* The gate reads date spans from the events block and returns
None on its first guard when there are none — so on the Türkiye 2023 feed, whose events
block carries only event_type/confidence/casualties/location/cause, it was structurally
unable to fire while the 1999 İzmit toll of 17,500 sat in the 2023 observation stream. Now
falls back to scanning the raw text for bare years.

*Competing dates that are NOT far apart.* On the Aegean feed the nearest year to İzmit's
17,000 is **2020 at +117 chars**, beating **1999 at −152**, so nearest-by-distance returns
the current year and misses a 143× contaminant. 117 against 152 is not "far apart", and
this docstring's own justification does not hold there. `mode="any"` takes any
out-of-window year within a radius instead — lower precision, and the right shape for
additive evidence in the decode's emission rather than for a veto.

### [6] scope_filter() — DECLARED, opt-in
Rejects observations keyed to a place outside the event's declared hierarchy. Run after
the rollup. **The cheapest cross-event filter available, and it beats every learned
signal tried:**

| | cross-event caught | false positives |
|---|---|---|
| scope membership (no model) | 4/6 | **7.3%** |
| best learned signal, one call/obs | 4/6 | 31.7% |

It works because the contaminating events happened *somewhere else* -- Mexico, Puerto
Rico, Bosnia, Reading PA -- which is declared knowledge, not a statistical property.

### [7] hmm_gate() — DECODE, the recommended replacement for the ratio gate

`scope_gate.hmm_gate` decides the whole stream at once over three states — `own`,
`aggregate`, `reject` — instead of committing per observation. It is `viterbi_gate` plus
per-observation REJECT evidence from outside the magnitude channel (an out-of-window date,
a place outside the declared hierarchy, a syndication marker), so those gates ARGUE rather
than veto. Recommended σ=0.3, reject_cost=4.0, stay=0.1, **warmup=0**.

Measured at that one setting on every event we have:

*Pooled RMSE, in deaths — lower is better.*

| event | ratio gate | decode | change |
|---|---|---|---|
| Helene | 29.3 | **20.7** | −29.4% |
| Türkiye–Syria | 11,581.5 | **10,695.5** | −7.6% |
| Aegean 2020 | 74.4 | **15.7** | −78.8% |

Two properties are load-bearing and both were predicted by the three-way oracle:
**global** (a greedy rule commits per observation, and one large figure admitted early
poisons a stream's running scale for everything after) and **able to reject** (assignment
headroom is measured at zero; the entire residual is the null hypothesis).

**Design rule, measured:** keep every feature weight BELOW `reject_cost`, so no single
feature can force a reject on its own — it can only tip a case magnitude has already made
marginal. The sweep shows a cliff exactly at that boundary.

**Two traps that cost real time, both from importing assumptions:** `warmup`, copied from
the greedy gate, pins the first readings to `own` and reintroduces exactly the commitment
the decode exists to remove — it alone flipped Türkiye from a loss to a win. And a MISSING
reference must not be read as evidence a value is too large: on a feed with no aggregate
stream `natl=0` made every value score above the whole event and dropped 52 of 53
observations at zero feature weight.

### [7] gate() — ARITHMETIC, legacy; superseded by hmm_gate
Three outcomes against the running larger-scope reference, per stream in time order:

```
keep     v < natl / ratio        plausibly this place's own toll
reroute  natl/ratio <= v <= natl*ratio    it IS the national figure -> __aggregate__
drop     v > natl * ratio        no scope in this event can exceed the whole
```

Reclassify, do not discard: rerouting a rejected figure to `__aggregate__` keeps the
national signal, which is the project's one honest measurement. `reference_for` chooses
what "natl" means -- `aggregate` (works while a part is small relative to the whole),
`global-max` (needed where no aggregate stream exists, but circular for the dominant
stream), or `implied`.

Helene at ratio 2.0: pooled RMSE **314.5 -> 29.3 deaths**, a 10.7x reduction.

`down_ratio` adds the second side (drop a reading far below the stream's own running
max, since a toll does not fall). **Measured 1-for-2 and left OFF** (pooled RMSE, deaths):
Helene 29.3 -> 21.7, Türkiye 14,765 -> 15,349 (worse). See `scope_field_results/two_sided_gate.txt`. The
global decode reaches the same place on Helene (20.7) and wins on Turkiye too, which is
why this knob is not needed.

### [7] plausibility_filter vs tail_filter — HEURISTIC vs STATISTICAL, both opt-in
`plausibility_filter` is a hand-set ceiling and has to be *told* the event's scale --
which it gets from the answer, so it is not a method. `tail_cut` derives the cut from
the event's own observations (median + k*MAD on log10, upper tail only), which is
scale-free and uses no ground truth.

**Every magnitude prior on this page is language-specific, and the Chinese arm is
measurably shifted (recorded 2026-08-31, surfaced here 2026-09-04).** Measured on 4,994
adjudicated `shaowenchen/news_zh` articles -- same corpus, same outlets, same period --
Chinese DOMESTIC casualty reports carry smaller figures than the same outlets' FOREIGN
reporting: median toll 17 against 26, and tolls >= 100 at 15.3% against 28.4% (odds ratio
0.46, exact Fisher p = 1.1e-04, Mann-Whitney z = -5.85). Full measurement and its
limitations in `tools/data/notes/CHINESE_TOLL_DISTRIBUTION.md`.

This does NOT establish editorial suppression -- newsworthiness selection is a sufficient
alternative (a three-fatality crash in Sichuan is routine local news; the same crash in
Brazil is not covered at all) and the corpus cannot separate the two. **The consequence
holds under either explanation**, because both leave the same distribution in the data.

For the EXTRACTOR it is close to harmless: it binds a number to a place from whatever the
text says, and a distribution shifted low does not change what `dead` means. For
everything in this section it is not. `plausibility_filter` is an explicit magnitude prior
(2,000 on Helene, chosen because 94,000 was Asheville's population); the EKF's `R` scales
with the reference value; `hmm_gate`'s whole magnitude channel is relative to a running
scale. All three encode magnitude expectations, and a Chinese arm carrying a
domestically-shifted distribution pushes them downward.

**Standing rule: do not pool Chinese streams with English ones without checking.
Per-language tracking error is the measurement that would expose it.**

A second, independent provenance caveat on the same arm: CommonCrawl reaches Chinese sites
from outside, so any CC-News-derived Chinese slice is the OUTWARD-FACING subset by
construction -- a narrower and differently-selected sample than domestic coverage.
`news_zh` avoids that particular selection by being native domestic text, and is not
neutral either: its named publishers are state media (新华网, 中国新闻网). Both bear on how
far any Chinese result generalises. The same caveat applies to the Wayback-sourced Chinese
feed added 2026-09-04 (`build_turkey_feed_zh.py`), which is additionally a FOREIGN event --
the 28.4% column, the favourable half of the table above.

**What would settle it, and it is now unblocked.** The cross-sectional comparison cannot
separate selection from reporting control, but a TRAJECTORY comparison can: selection
explains *which* events get covered and has no mechanism to act on *how a covered event's
toll evolves*, whereas reporting control does. Time-to-plateau, count of upward revisions
and first/final ratio all discriminate where magnitude alone cannot. That test was
deferred because it needs reliable (t, toll) extraction from Chinese text -- a regex would
confound measurement error with the effect being measured. **That gate has lifted:**
`gliner2-casualty-multilingual` scores 0.3592 matched-instance F1 on real held-out Chinese
(against 0.1900 for the model it replaced). Three events with independent tolls already sit
in the 5k sample -- Shenzhen 2015 landslide (47 articles, 77 dead), Tianjin 2015 explosions
(19, 173), Nepal 2015 earthquake (4, ~8,964) -- so the cost is analysis time, no annotation
spend and no GPU.

### [8] The EKF and its gates — STATISTICAL, mostly off
Measurement noise is **relative**: `R = (sig * max(ref,1))^2`, with `sig` from
`SRC_REL_SIGMA` (official 0.06 / major_outlet 0.12 / preliminary 0.25) scaled by
`QUAL_FACTOR` (point 1.0 ... feared 2.5). Process noise grows as
`q_rel * max(mu,1) * dt`, so real jumps stay admissible between reports.

- **STUDENT_T_NU** (OFF): a Student-t measurement model, applied as one-step IRLS
  reweighting (`R` inflated by `w = (nu+1)/(nu+d^2)`) and ONE-SIDED, because the physics is
  — a rising toll may legitimately surge above the estimate, only a reading far below it is
  implausible. The symmetric textbook form is much worse on both events. Measured 1-for-2
  (Helene −1.7 deaths, Türkiye +651) and it retires none of the thresholds below, which was
  the reason to want it: with the scope gate off, Helene is 314.5 under every nu tested.
  Whatever the gate catches, it is not a fat tail.
- **REJECT_SIGMA** (OFF): one-sided innovation gate. A rising toll is non-decreasing, so
  it rejects only readings implausibly *below* the estimate; the decay roles invert it.
- **MAX_RATE** (OFF): drops an observation whose upward accrual rate exceeds the limit --
  the impossible jump the one-sided gate *admits* by construction, since that gate only
  rejects lows.
- **CONF_R** (OFF): folds extractor confidence into R, so low confidence widens the noise
  instead of hard-dropping the reading.
- **CENSOR_AT_LEAST** (ON): treats an `at_least` reading below the estimate as
  uninformative. Logically correct for a strictly rising toll -- "at least 96" is
  consistent with 123 -- and **measured wrong on Helene**, where ground truth falls four
  times in North Carolina and three times nationally as deaths are reclassified. The
  measured negative is recorded at the flag definition.
- **--learn-gate** (opt-in): the only genuinely LEARNED component outside the extractor.
  A logistic router `alpha = sigmoid(w.x)` over 8 features (staleness, source
  unreliability, qualifier coarseness, censored-bound flag, decay-vs-rise role,
  EKF-vs-last_value disagreement, reports seen so far, bias), fitted by gradient descent
  on peak-normalized blend MSE. It replaces the hand-set `SRC_TRUST` / `QUAL_TRUST` /
  `GATE_TAU` tables with fitted weights.

---

## The oracles — ORACLE, never shippable

All three consult ground truth. They price ceilings so we know whether a component is
worth building before building it.

| Oracle | What it does | What it prices |
|---|---|---|
| `oracle_gate` | sends every observation to its own place or to Total, whichever is closer to truth | perfect **assignment**, no reject option |
| `oracle_gate_three_way(tol)` | adds a third outcome: reject when relative error against **both** scopes exceeds `tol` | perfect assignment **plus** a reject option |
| `stream_ceiling` | at each grid point takes the observation closest to truth | method quality vs **coverage** |
| `random_control(n, trials)` | removes n observations at random | the null: does dropping *any* n help? |

**The measurement that matters.** On Helene the two-way oracle scores 29.3 -- identical
to the shipped gate. So **assignment headroom is zero**; there is nothing to win by
associating better. The three-way oracle reaches 17.6 at tol 0.25, and it gets there
purely by dropping more (106 kept -> 76). All ~11.7 remaining deaths are in the reject
option, and 63% of those rejects are stale readings *below* truth, which the upward-only
gate cannot see. See `scope_field_results/reject_headroom.txt`.

---

## What these gates are actually filtering

Eleven percent of Helene's `dead` observations are hand-audited **non-casualty**, and they
split into two kinds that need different fixes:

| kind | examples | fix |
|---|---|---|
| **exposure counts** | 300 rescued, 50 patients rescued, 32 evacuated, 11 swept away | a schema role |
| **unit confusion** | a two-day period, six states, dozens of vehicles, 1,400 landslides | a schema role |

A rescued person is a **counterfactual casualty** — averted harm, not realized harm. And
exposure counts run systematically LARGER than casualty counts in disaster copy (505
displaced against 6 dead; 300 rescued against Florida's true peak of 26), so a mis-bind is
the same magnitude as cross-event contamination and points the same way, upward. The gates
therefore catch some of them — for the wrong reason, magnitude rather than category, which
is the gate-1 lesson again.

**The root cause is upstream of every gate on this page.** `casualty_events` has exactly
four roles — `location`, `injured`, `missing`, `dead` — and no role for exposure, while the
prose is full of it. A number with no correct home lands in a wrong one. Adding `displaced`
and `rescued` removes the ambiguity at source instead of filtering it downstream. Full
taxonomy in `ekf_showcase/gate_results/EXPOSURE_VS_CASUALTY.md`.

## Not pipeline stages: the pre-registered gates 1-4

`frontend_gates.py` uses "gate" for **acceptance tests on a model**. They gate a spending
decision, not an observation. Sweep fixed before spending: thresholds 0.5-0.1 registered,
0.05/0.01 diagnostic only.

| | Test | Bar |
|---|---|---|
| 1 | usable events-form on the Helene feed: a trigger AND >=1 bound argument | >= 50% of windows |
| 2 | the span block is LOCAL: on the Katrina passage it must hold 1,400, not Helene's figure | pass/fail |
| 3 | event_trigger / event_argument F1 on the shared blind test | >= the incumbent's |
| 4 | entity / relation / structure F1 | not below the 137k-clean reference |

**The standing lesson.** Gate 1 counts *firings*, not correct ones. Scored best-over-a-
range it rewards indiscriminate firing, and it inverted a verdict for a day: the
incumbent's "65%" was 39 firings carrying **three** correct tolls, while the rebuild it
rejected binds the right figure 67-100% of the time. Pair every form gate with a
correctness companion -- that is what `binding_accuracy.py` is for.

---

## Should [5] and [6] be merged into one LEARNED gate?

Measured 2026-08-25 (`gate56_composition.py`), against the 86 hand-audited Helene
occurrence labels. **No -- but the current composition is the worst term in the system
and should be fixed.**

They are genuinely complementary. All six cross-event figures, and what catches each:

| value | keyed to | in declared scope | nearest-date year | caught by |
|---|---|---|---|---|
| 2 | mexico | no | - | [6] |
| 3000 | puerto rico | no | 2020 | [6] + [5] |
| 32 | tennessee | **yes** | - | **neither** |
| 80 | north carolina | **yes** | **1916** | **[5] only** |
| 16 | bosnia | no | - | [6] |
| 1400 | reading pennsylvania | no | 2005 | [6] + [5] |

So [5] catches one figure [6] structurally *cannot* -- the 1916 hurricanes' "80", keyed
to North Carolina, which is legitimately in scope. Series composition is already OR, and
it reaches 5/6.

But the trade is bad:

| rule | caught | false-rejects of genuine |
|---|---|---|
| [6] scope only | 4/6 | 6/81 = **7.4%** |
| [5] date only | 3/6 | 10/81 = 12.3% |
| UNION (as shipped) | 5/6 | 16/81 = **19.8%** |
| INTERSECTION | 2/6 | 0/81 = **0.0%** |

**[5]'s marginal contribution over [6] alone is +1 catch for +10 false rejections.** It
is a hard reject (`continue`), not a flag, so that cost is realised.

Three reasons not to make the merge *learned*:

1. **It was already tried and lost.** Best learned signal, one call per observation:
   4/6 at 31.7% FP, against declared scope's 4/6 at 7.4%. Same recall, 4.3x the false
   positives. Replacing knowledge that is written down and correct with a statistical
   estimate of it is strictly worse.
2. **The one figure neither gate catches is not a filter failure.** "Typhoon headed to
   Taiwan injures dozens" was bound to Tennessee because "11 workers at a Tennessee f..."
   follows it. That is `association_key`'s nearest-place heuristic failing at [5] -- the
   same failure mode `out_of_window`'s own docstring cites for the Turkiye standfirst. No
   filter downstream of a bad binding can repair it.
3. **There are six positive examples.** A learned gate cannot be fitted on that, and the
   audit set is 86 rows total.

**What to do instead:** the operating point between AND (0% FP, 2/6) and OR (19.8% FP,
5/6) is unexplored, and [5] is the miscalibrated half. Either require the old date to be
nearer the span than any current-year date before rejecting, or demote [5] from hard
reject to a feature and let [6] keep the veto. Both are arithmetic.
