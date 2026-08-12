# Open items — resume list

State at 2026-08-11 close. Completed work is removed rather than struck through; history
lives in `PROJECT_JOURNAL.md` and the commit log. Everything below is a defect with evidence
attached, or a decision with a stated next test.

State at 2026-08-12 close. **No GPUs running; both Lambda A10s terminated.** The GIST cache
is built for `mix_natural` and for RAMS (item 11). The base-word arms A/B/C ran — the result
is **provisional**, since every arm sits at a fixed unswept threshold and the checkpoints
went with the box (item 12). Tier 2 multi-instance events are wired and shipped behind
`event_records`, and its first measurement on CASIE is strongly negative for head-init
reasons rather than mechanism ones (JOINT_IE_SCALING). Nothing is mid-flight.

**Where the line stands.** The scope gate took per-state error 5.247 → 0.591 (item 10) and
largely closed the attachment blocker. Two candidate next steps were then *closed by
measurement rather than argument*:

- **MHT is not the bottleneck.** Perfect association is worth **+0.055** — and the gate
  already beats a perfect two-way assignment on two states, because it can *drop*. Item 6.
- **Extraction recall is not the bottleneck either**, and the claim that it was rested on a
  stale count from a superseded run. `extract_long` had already fixed it **4.2x**.

The live bottleneck is **cross-event contamination** (item 2, quantified at 4.7% and
resistant to all three signals tried) and underneath it the query-axis training gap that
GIST is being built for (item 11, the active work).

---

## P0 — blocks the next experiment

### 1. Number-to-place attachment — the actual research blocker
Proximity, GPE tags, record-internal location and admin rollup have all been tried and all
failed. Rollup did what it was supposed to (58 keys → 21, 84% of observations in six clean
streams) and per-state tracking is *still* catastrophic: North Carolina 5.637, Georgia with
0 of 5 values in plausible range. The reason is not fragmentation. It is that national totals
get filed under whichever state the article happens to be about.

Two routes, not mutually exclusive:

- **Train the relation.** `casualty_multi_loc` already carries gold `(value, place)` pairs, so
  a `deaths_in` relation can be supervised directly instead of relied on zero-shot.
- **Run the beam arm** with `TypedEndpoints`, which makes `('storm', Florida)` structurally
  unrepresentable rather than merely unlikely. Unblocked 2026-08-10 by the qualified-key fix
  (see item 8) — this is now runnable and unrun, where before it was unrunnable.

Zero-shot is close but fragile, and the fragility is the argument for training over
prompt-tuning: `explicit-scope` phrasing got the hard aggregate case exactly right
(120 → North Carolina, 17 → Tennessee, correctly *excluding* the national 227) while two other
phrasings of the same request, same model, same text, got it wrong.

---

## P1 — known-wrong

### 2. Cross-event contamination — now the top real defect
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

**Three signals tried, all failed** (EKF_MHT_DESIGN §27.2): nearest named event 3/11 at 32.5%
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

#### BUILT 2026-08-12 — `--mute-interference-prob`, control proven; not yet trained on

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

**Pass/fail, fixed before spending:** cross-event share below 4.7% on the same 106-observation
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


### 11. GIST query-axis hard negatives — CACHE BUILT; ready to train
The measured gap: with specific rival types, `people evacuated` outscores `death toll` on
**11.2% of genuine death tolls**. "N people killed" vs "N people evacuated" — both counts of
people, separated only by the verb. No type description fixes it (EKF_MHT_DESIGN §27.8); it
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

## Notes for whoever picks this up

- **`pytest tests/models/boundary tests/processing` as ONE process stalls at ~49%, and it
  predates this work.** Every file passes individually and in chunks (239 boundary + 55
  processing + 29 trainer). Attribution was confirmed by running the suite from a worktree at
  `f0925d1`: it stalls at the same point there. Import precedence is **cwd → PYTHONPATH →
  editable finder**, so `cd <worktree>` is sufficient to load the pre-change library —
  verified by printing `gliner2.__file__` and confirming `guide.py` is absent. *Which* test
  hangs is still undiagnosed; that needs the full suite in one process, which wants a machine
  not already holding a 15GB job. Run tests in chunks meanwhile.

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
