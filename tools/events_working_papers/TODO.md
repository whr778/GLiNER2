# Open items — resume list

Completed work is removed rather than struck through; history lives in
`PROJECT_HISTORY.md` and the commit log. Everything below is a defect with evidence
attached, or a decision with a stated next test.

**State at 2026-08-28 close. No GPU running, nothing billing.** The phase that just closed
produced no new model: every result came from measuring shipped components at more than one
operating point. See `PROJECT_HISTORY.md` Phase 25.

### Open after 2026-08-28

- **Which gate ships is now an open decision, and the current default is not the leader.**
  Swept on 1,000 annotated messages, `fastino/gliner2-base-v1` leads on AUC (0.9635) and on
  recall at every operating point; the shipped `casualty-docee` is 0.9241 and the trained
  `gate2-mmbert-v2` is 0.9472. The switch away from fastino (b607fae) was decided on false
  positives at one threshold with no recall column. **Next test:** re-run the pipeline's
  end-to-end event metrics under each gate at its own swept threshold, rather than choosing
  on the gate benchmark alone — the benchmark's recall label is indicative, and what matters
  is pooled RMSE in deaths on the three events.
- **`--gate-threshold` still defaults to 0.5 in `run_pipeline.py`.** Every recorded gate
  number predates the sweep. The trained gate needs ~0.998 on its own distribution; the
  right per-model default is whatever its validation split chooses, and no default has been
  changed yet. **Next test:** the same end-to-end run as above decides it.
- **Turkish is affordable but not bought.** The pilot ($0.72) found 43.3% positives in 989
  articles; the remaining cue-bearing region is ~2,100 more positives for ~$2.84, plus one
  ~$2 training run. Two caveats stand: the source is one outlet in one year, and its
  positives skew to conflict rather than natural disasters. **Decision, not a test:**
  buy the region and retrain, or ship translation-at-ingest (measured, AUC 0.4733 → 0.8359)
  and spend nothing.
- **M2 — the cost matrix — remains unbuilt and unmeasured**, and with M4 superseded by the
  global decode there is currently no reason to build it. Recorded so its absence stays
  visible in the divergence table rather than being inferred.
- **`exposure_only` is closed as a training problem.** 0.444 → 0.903 by moving a threshold.
  Do not spend another run on it.

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
in [[PROJECT_HISTORY.md]]). So when a defect admits both a relation-shaped and an
event-shaped fix, the event-shaped one is the one that serves the programme *and* the one
that carries the right information: only the event formulation has an `event_key`, which is
the field an EKF observation needs. See item 1 for the worked case.

### The next move is NOT downsampling casualty_events

The config pre-registers that remedy for a gate-1 failure, but two measurements taken since
say it cannot work:

1. **Gate 2 scores sparsity, not binding, so a mix change cannot reliably move it.**
   `_decode_events` emits ONE instance per event type and pools every trigger and argument
   into it -- its own docstring says the mention path "carries no instance dimension".
   Measured on the incumbent with two hurricanes in one passage: `n_event_instances=1`, with
   Helene's 246 and Katrina's 1,400 both filed as `dead` on the same event, at 0.1 and at
   0.01. Gate 2 takes min(start)..max(end) over that single pooled instance, so it passes
   only when the model happens to emit *nothing but* Katrina-local spans.
   **Not impossible -- the incumbent does produce a LOCAL Katrina block at 0.01** (the
   passage carries one Hurricane-typed event; Helene is a mention, not a second extracted
   event). But what the gate rewards there is sparsity, which is why it only lands at the
   threshold where the rest of the output is nonsense. Pooling makes it fragile, and no
   corpus teaches a model to emit fewer spans on demand.
2. **`casualty_events` cannot teach the missing capability anyway.** It carries 8 event types
   and no named identities, so it has no same-type discrimination (Helene vs Katrina) in it
   -- which is the live defect in item 2.

The reasoning changed on 2026-08-24 but the conclusion did not. It is NOT "the mix failed";
the mix succeeded. It is that binding correctness is now good and the router still has no
per-event input, because the decode cannot emit two events of one type. **The next move is
the instance dimension -- the record head -- not another corpus.**

That path has one prior measurement, and it is not a clean negative: the CASIE Tier 2 arm
produced REAL multi-instance output (2-9 instances of one type on 17 of 39 probed docs,
structurally impossible before) and scored 0.0036 against a 0.2998 control -- diagnosed as
head-init, the record head having never been supervised on events and given 375 steps to
learn instance formation cold. The fair test it names: warm-start the record head on MAVEN,
then fine-tune on CASIE, with far more steps. The head is also miscalibrated (max object
probability 0.178 against a 0.5 default).

**Open and worth cheap work first:** yield is still only 15% at 0.1 and 26.7% at 0.05. The
rebuild's remaining misses are mostly word-form golds (`six`, `three`, `dozens`) where it
binds a nearby numeral -- a normalisation gap, not a binding failure, and probably fixable
without training.

`tools/data/split_rams_test.py` gave rams a val split by carving its 871-row test **by
document**, which found 101 duplicate rows in test alone.

State at **2026-08-23**. **No GPU running** (the front-end A100 self-terminated 2026-08-21, verified 0 instances).
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

### 0.2. `record_anchor_threshold` defaults to 0.5, which no model can use well
Separate from the decode bug and still live. Nothing calibrates the record cutoffs --
`threshold_sweep` moves the general decision threshold only. At the 0.5 default the models
score 0.0000 / 0.0052 / 0.0654 / 0.0760 versus 0.0238 / 0.0552 / 0.1043 / 0.1119 at their
swept thresholds: a 32-100% loss depending on scale. Either fold the record cutoffs into
`threshold_sweep` or change the default; `tools/train/sweep_record_thresholds.py` is the
sweep in the meantime.


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

### 7. Still no benchmark that can score the filter
Turkiye's baseline was an oracle by construction — truth read from the sentence the extractor
reads, so `est_last_value` scored 0.000. Helene's per-state streams were mis-bound until the
scope gate. A real filter benchmark needs **multiple sources that disagree and revise** about
one event, which is also the regime where MHT would finally earn its keep.

### 10. Aggregate SCOPE (not the aggregate constraint) — the sharpened target
Two different things wear the word "aggregate" and only one of them is open.

**REOPENED 2026-08-24 — the constraint direction was rejected by the METRIC.** `nrmse`
normalises each state by its own range and macro-averages, so Virginia (1→2) counts like
North Carolina (6→123) and carried 110.5% of the vector arm's excess error. On the same
runs the aggregate cuts **national total RMSE at every density** — 87.6→28.5 at 10%
reporting, the density where this section says it "loses worst". The mechanism recorded
below ("constrains the SUM, says nothing about the SPLIT") is correct; concluding "so it
does not pay off" while scoring only the split is not. See
`tools/ekf_showcase/vector_state_results/METRIC_INVERTED_THE_VERDICT.md`. The text below is
kept as measured, and its verdict is withdrawn.

**The constraint direction is measured and it LOSES *on nRMSE*.** `vector_state_test.py` feeds the
national total in as a sum row over the six state components. Against `parts-only`, on real
Wikipedia trajectories with `--q-prop 0.15`:

| per-state report rate | parts-only | vector | delta | vector wins |
|---|--:|--:|--:|--:|
| 10% | 0.4348 | 0.6085 | +0.174 | 4/40 |
| 50% | 0.2030 | 0.2234 | +0.020 | 22/40 |
| 80% | 0.1556 | **0.1520** | **−0.004** | 30/40 |

It loses everywhere except 80% density, and loses **worst exactly where it was predicted to
win**. An aggregate constrains the SUM and says nothing about the SPLIT, so when parts are
sparse the filter must guess the division and the total injects error.

**QUALIFIED 2026-08-24 against a measured noise floor** (10 RNG streams x 40 trials, three
noise levels -- `vector_state_results/NOISE_FLOOR.md`). The table above is ONE stream at
noise 0.10; re-running it reproduces, which is determinism, not variance.

- The rejection **holds where it matters**: at 10-20% density it clears the floor at every
  noise level. Real feeds are in that regime.
- The **middle rows were never readable**: +0.0568 at 35% and +0.0203 at 50% sit inside
  floors of 0.0678 and 0.0275. They were quoted as measurements and they are noise.
- At **80% density AND noise 0.20 the vector arm WINS and clears the floor** (-0.0112,
  spread 0.0069, 350/400). The recorded experiment fixed noise at 0.10 and never looked.

So the constraint is not wrong in general -- it is wrong in the regime we measured and
right in one we did not. It stays closed for THIS pipeline because sparse per-state
reporting is what the feed provides, but "it was tried and it lost" overstates it: the
correct statement is that it loses when parts are sparse, and pays when parts are dense and
individually noisy.
(Isotropic `Q` makes it 7.7x worse still — proportional process noise is a precondition,
not a tuning knob, since Virginia ranges 1→2 while North Carolina ranges 6→123. As of
2026-08-24 `--q-prop 0.15` is the tool's DEFAULT, because the documented trap was the
default behaviour and the journal records someone mistaking the isotropic run for a new
catastrophic result.)

**A candidate reason to revisit, tested and eliminated (2026-08-24).** Both Q options are
strictly DIAGONAL, which asserts state tolls accrue independently — and the aggregate row
`H = [1,1,1,1,1,1]` is exactly where that bites, since `Var(sum) = Σᵢⱼ Pᵢⱼ` gives the
off-diagonals a direct say. So the constraint may have been rejected on a process model
that cannot represent what the aggregate observes. `--q-rho` adds uniform correlation with
the marginals preserved. It does not rescue it — the vector arm gets monotonically worse:

| rho | 10% | 35% | 80% |
|---|--:|--:|--:|
| 0.0 | +0.174 (4/40) | +0.057 (10/40) | **−0.004 (30/40)** |
| 0.3 | +0.256 (3/40) | +0.091 (1/40) | +0.011 (9/40) |
| 0.6 | +0.286 (5/40) | +0.096 (3/40) | +0.016 (4/40) |
| 0.9 | +0.308 (5/40) | +0.098 (5/40) | +0.024 (2/40) |

Correlation degrades **parts-only** too (0.4348 → 0.5280 at ρ=0.9), so ρ>0 is simply a
worse fit for these trajectories: the real revisions are idiosyncratic — North Carolina's
123→102→96 reclassification is about North Carolina — not common-mode. The rejection now
survives isotropic, proportional-diagonal AND correlated Q.

### 16. Exposure counts are not casualties, and the schema has nowhere to put them

12 of Helene's 106 `dead` observations are audited non-casualty: six are EXPOSURE (300
rescued, 50 patients rescued, 32 evacuated, 11 swept away) and six are UNIT CONFUSION (a
two-day period, six states, dozens of vehicles, 1,400 landslides). A rescued person is a
counterfactual casualty -- averted harm, not realized harm -- and exposure counts run
systematically LARGER than casualty counts, so a mis-bind is the same magnitude as
cross-event contamination and points the same way, upward.

Root cause is the schema: `casualty_events` has location/injured/missing/dead and no role
for exposure while the prose is full of it. Adding `displaced` and `rescued` fixes it at
source. Same diagnosis as 10a reached for `scope`: a data-modelling problem being handled
by a gate. Full taxonomy in `ekf_showcase/gate_results/EXPOSURE_VS_CASUALTY.md`.

### 14. Scope supervision — do NOT buy labels under the current scheme (2026-08-25)

Closes the open question in item 10a. A $2.25 dual-label probe (Haiku 4.5 vs Opus 5, 200
records, `scope_label_probe.py`) says:

* **Haiku is not cleared** -- 83.0% agreement but **kappa 0.121**, which is the number that
  matters when 86% of the corpus is one class.
* **The model is not the variable.** Hand-adjudicating 14 place<->sub-place disagreements
  came out **7-7**. `sub-place` conflates a genuinely narrower counting unit ("three
  districts IN Iraq") with a narrower incident SITE whose number is still the bound place's
  full toll ("across 2 districts OF Baghdad", "a mine collapse in Xinjiang"). Case two
  mislabelled makes `apply_extracted_scope` DROP a valid observation.
* **The corpus cannot teach the class that matters.** `national` is 0.0% of the unbiased
  sample and 0.5% of the hard stratum. `casualty_events` can teach place-vs-sub-place and
  not place-vs-national -- which is the only distinction the gate exists to arbitrate.

Next: rewrite the question to be about the COUNTING UNIT rather than the incident site,
re-probe, and beat kappa 0.121. Separately, find a source of place-vs-national supervision;
the 86 hand-audited Helene occurrences are the existing precedent.

### Regional disaster profiles as a plausibility prior (idea, 2026-09-04)

**Operator's observation, recorded because it converts a measured limitation into an
instrument.** Every region has a disaster profile. A tsunami in Turkey, or major snowfall
in Fiji, is an outlier -- and the outlier is informative in BOTH directions: either a
genuinely notable event, or an extraction error. Both are things the pipeline should
surface rather than silently pass to a filter that consumes any arriving number as a
measurement.

**This reframes a result already in hand.** The Turkish event-type pilots were reported as
having a coverage gap -- 0 Tsunamis, 0 Volcano Eruption, ~199 Earthquakes projected at
30K. That distribution IS Turkey's profile. Turkey has earthquakes, floods, fires, mine
collapses and road crashes; it does not have volcanic eruptions. The pilot did not fail to
find those types, it measured that they do not occur.

**It would make a filter shippable that this page currently says is not a method.** Of
`plausibility_filter`, GATES.md says: "a hand-set ceiling [that] has to be *told* the
event's scale -- which it gets from the answer, so it is not a method." A learned
`P(event_type | region)` and `P(magnitude | event_type, region)` takes the scale from the
REGION instead of from the answer. That is the difference between an oracle and a filter.

**It would catch contaminants nothing currently can.** The Turkish feed quotes Haiti 2010
(316 bin = 316,000 dead) and Antakya 115/525 AD (260,000 / 250,000) as bare figures that
the extractor binds CORRECTLY as `dead` -- they are death tolls. Today only `Date` can
reject them, and Date is the weakest stage-1 field in Turkish (0.26 non-empty). A region
prior says 316,000 deaths in a Turkish-datelined earthquake is ~3 orders of magnitude
outside profile. Same shape as the 1999 Izmit 17,500 contaminant on the English feed.

**Related measurement already recorded.** `tools/data/notes/CHINESE_TOLL_DISTRIBUTION.md`
found Chinese DOMESTIC tolls run smaller than the same outlets' FOREIGN reporting (median
17 vs 26; >=100 at 15.3% vs 28.4%). That is a reporting-regime profile, and treating it as
a prior is more honest than treating it as noise -- the note already warns that the EKF and
`max_plausible` both encode magnitude expectations and will be pushed downward by it.

**The cheap version needs no new data.** DocEE-en (21,842), DocEE-zh (36,729, still
unconverted) and the ~30K Turkish buy are all in ONE label space and all carry Location.
Three regional profiles fall out of data already bought. Cost is analysis time.

**Cautions before building it.** (1) A profile learned from news measures REPORTING, not
occurrence -- the Chinese note's own selection-vs-suppression ambiguity applies here too.
(2) It must flag, not veto: the whole point is that a real outlier and an extraction error
look identical to the prior, so a hard reject would discard exactly the rare events the
tracker exists for. (3) Scored on catch rate AND false-reject rate together, per the
standing gate-1 lesson that a form gate scored on firings alone rewards indiscriminate
firing.

#### The flag-not-veto MECHANISM already exists, twice, and both are switched off

Caution (2) is not an intention to be remembered -- the plumbing is built. Nothing new has
to be invented, and the two mechanisms fail differently, so use both.

**(a) `CONF_R`, per observation, continuous.** The filter already implements
`R /= confidence**2`, so a weight of 0.1 inflates that reading's noise **100x**
(`imm_gate_sweep.py`). That IS flag-not-veto: a veto DROPS the observation and destroys
the information; inflating `R` keeps it in the update while shrinking the Kalman gain
`K = P/(P+R)`, so it moves the estimate less.

The property that makes this right for outliers: **a real rare event REPEATS, an
extraction error does not.** With `R` inflated 100x a single Haiti-shaped 316,000 barely
nudges the filter, but a genuine Turkish tsunami reported five times applies five
(down-weighted) pulls and the filter converges there anyway. The prior does not decide --
it buys time for corroboration to decide. A veto forecloses that on the first sighting.

`CONF_R` is OFF because the only signal available to feed it was extractor confidence,
which was measured unreliable -- confidence "did not filter the mis-binds" on Venezuela.
A regional prior is a better-grounded number for the SAME plumbing:

    R  <-  R * exp(lambda * surprisal),   surprisal = -log P(value | event_type, region)

**(b) `hmm_gate` feature, global, able to revisit.** The decode already enforces this
structurally rather than by convention -- the measured design rule is "keep every feature
weight BELOW `reject_cost`, so no single feature can force a reject on its own; it can
only tip a case magnitude has already made marginal. The sweep shows a cliff exactly at
that boundary." A regional-profile feature enters beside out-of-window dates and
out-of-hierarchy places, weight < 4.0, and ARGUES.

**Why both.** `hmm_gate` is a global decode that can revisit and routes to
`own`/`aggregate`/`reject`; `CONF_R` is per-observation and continuous. A figure that
belongs to a DIFFERENT STREAM is a gate problem. A figure that is right-stream but
wrong-MAGNITUDE is an `R` problem. One does not cover the other.

**The failure this avoids has been paid for here twice.** `value_qualifier` returned `0`
for an unparsable span and manufactured 30 fabricated zeros out of 114 Helene
observations; the fix was returning `None` so "no number here" stays distinguishable from
"the number is zero". A hard veto on prior-implausibility is the same mistake wearing a
Bayesian hat -- it erases the tsunami-in-Turkey the tracker exists to catch, and leaves no
trace that it did.
