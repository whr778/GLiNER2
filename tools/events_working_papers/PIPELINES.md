# Pipeline maps — as-built and as-designed

Two pipelines, drawn separately because they are two programmes that share an extractor, not
two halves of one:

| | unit | span | ends when |
|---|---|---|---|
| **standard event pipeline** | one document | within-document | the document ends |
| **EKF pipeline** | one real-world event | across documents and across time | the event stops being reported |

The boundary between them is a single record type — the **observation**
`(t, role, value, qualifier, source, event_key)`. Everything upstream produces one;
everything downstream consumes one. That seam is also where every hard problem of the last
week lives: attachment, scope, and cross-event contamination are all failures to build a
correct observation, not failures of either pipeline's internals.

`✅` shipped and measured `⚠` shipped with a known defect `⛔` specified, not built

---

## 1. Standard event pipeline — AS BUILT

`gliner2/` — one document in, structured records out. No time, no tracking, no state.

```
  text (str)  +  Schema{entities, relations, json_structures, events, classifications}
        |
        v
  [A] SchemaTransformer                                     processor.py
        |   schema dict -> per-task schema token sequences
        |   markers: [P] prompt  [E] entity  [C] class  [R] relation  [V] value  [L] label
        |   record_metadata declares mode: natural | latent | anchorless
        v
  [B] ExtractorCollator -> PreprocessedBatch                training/trainer.py
        |   tokenize, pad, build start/end word->token maps
        v
  [C] BaseExtractorModel._encode_core                       models/boundary/model.py
        |   encoder (DeBERTa-v3 / mmBERT) -> text_states, query_states
        |   ext_specs: one query per (task, field);  QueryLayout types them by role_name
        v
  [D] BoundaryHead -> CandidateTensorBatch                  models/boundary/heads.py
        |   start/end boundary scores -> sparse (query, candidate) spans
        |   NOT fixed-width enumeration: this is what removes the span 20-instance cap
        v
        +----------------------------+----------------------------+
        |                            |                            |
        v decode_mode=greedy         v decode_mode=joint           v
  [E1] per-query threshold      [E2] joint_ie                 [E3] classifications
       _decode_records               boundary_candidates_to_             pooled logits
       _decode_relations             candidate_score_set                 -> labels
       null-abstention ✅            + relation pairs -> edges
       adaptive threshold ✅         + record groups -> role edges
                                     -> JointProblem
                                     -> BeamOptimizer + TypedEndpoints
        |                            |
        +----------------------------+
        v
  [F] span -> char offsets, format_results                  models/boundary/engine.py
        v
   {entities: [...], relations: [...], <task>: [instances], classifications: {...}}
```

**Long documents** bypass nothing — they re-enter at [A] per chunk:

```
  document --> chunk (words, overlapping) --> [A..F] per chunk --> merge by char offset
              extract_long / batch_extract_long, chunk_size + chunk_overlap
```

### As-built notes that matter when working in this code

- **[E2] beam width should be 1.** Relation F1 falls monotonically 0.2406 (W=1) to 0.2058
  (W=64); entity metrics do not move at all, because `_finish_nodes` admits every
  positive-score node regardless of beam state. Default is still 16. *(JOINT_IE_SCALING §4c)*
- **[E2] gained the eval threshold on 2026-08-10.** Before that, `decision_threshold` was
  pinned at 0.5 and the joint decode ignored `--threshold` for edge selection entirely.
- ⚠ **[E1] vs [E2] are not threshold-identical.** Null-abstention and adaptive thresholding
  are greedy-only. Any arm comparison inherits that asymmetry.
- **Record role edges bypass threshold centering** — a scalar role's utility is the
  ABSENT-relative `logit_c - logit_ABSENT`, which has no probability cutoff to centre on.
- ⚠ **A `json_structures` schema without `record_metadata` silently yields no records** on
  the boundary architecture. The processor warns; it does not fail.

---

## 2. EKF pipeline — AS BUILT

`tools/ekf_showcase/run_pipeline.py` — a feed of articles in, a tracked state over time out.
Stage 2 calls the *whole* of pipeline 1 as a subroutine.

```
  feed.jsonl   [{t_hours, text, url}, ...]        one row per archived article
        |
        v
  [0] GATE                        classification, gliner2-base-v1              ✅
        |   "is this article a mass-casualty report?"  --gate-threshold 0.5
        |   ⚠ answers "is this article ABOUT one", never "does this number
        |     belong to THAT one" -- the source of cross-event contamination
        v
  [1] EVENT (optional)            DocEE classification, --event-model          ✅
        |   event_type + 'Casualties and Losses' spans
        |   SKIPPED when --event-model is unset; then event_type = 'unknown'
        |   and --associate envelope / type+location degrade to no key
        v
  [2] EXTRACT                     casualty structure model  === PIPELINE 1 === ✅
        |   Schema().structure("casualty_report")
        |     .field(dead|injured|missing) .field(location)
        |   --record-mode natural   (REQUIRED for boundary models)
        |   --window whole | event | long   (long = extract_long)
        v
  [3] NORMALIZE                   span -> (value, qualifier, source)           ✅
        |   --normalizer heuristic | classify | hybrid | both
        |   word_number() parses spelled-out cardinals; unparsable -> None,
        |     never 0 (that bug fabricated 30 zeros)
        |   qualifier: point|at_least|about|feared|interval   source: official|...
        v
  [3b] TEMPORAL FILTER            --event-year                                 ✅
        |   drop figures dated before the event: Izmit 1999's 17,500 bound as
        |   a 2023 observation 15 times -> 3, with zero genuine losses
        v
  [3c] ASSOCIATE                  --associate none|type|type+location|envelope|record
        |   record   : location from the record's OWN field   (strongest signal)
        |   envelope : nearest location span by char distance (heuristic)
        |   -> event_key,  e.g. "Floods|north carolina"
        v
  [3d] ROLLUP                     --rollup datasets/<event>/rollup.json        ✅
        |   collapse_type: Floods|florida == Storm|florida
        |   aliases:   asheville -> north carolina;  six states -> __aggregate__
        |   hierarchy: {aggregate: __aggregate__, parts: [...]}   <- declares scope
        |   unmapped places are left ALONE, never guessed
        v
  [3e] SCOPE GATE                 ratio 2.0 vs the __aggregate__ series        ✅ NEW
        |   keep     value <  natl/ratio          plausibly this place's own
        |   reroute  value <= natl*ratio          it IS the national figure
        |   drop     otherwise                    exceeds the whole; not a toll
        |   per-state 5.247 -> 0.591; the aggregate improves too (0.402 -> 0.316)
        |   ⚠ needs an __aggregate__ stream. No aggregate => no gate.
        v
  [4] TRACK                       per event_key                                ✅
        |   HARD assignment on the string key -> one single-stream EKF per key
        |   est_ekf(observations, grid, role)   vs   est_last_value (baseline)
        v
   tracked.json  {grid, tracked_by_event: {key: {role: {ekf, last_value, n_obs}}}}
        |
        v
   scoring: score_helene.py | revision_test.py | vector_state_test.py | scope_gate_test.py
```

### As-built notes

- ⚠ **[3c] is the research blocker**, not [4]. Proximity, GPE tags, record-location and
  admin rollup have all been tried; the scope gate at [3e] is the first thing that moved it.
- **[3e] is where MHT would have gone.** What ships is hard assignment plus a magnitude rule.
- ⛔ **No MHT.** §3 specifies gate -> Hungarian -> top-K hypotheses -> track birth/death.
  None of it is built. See §4 below for whether it should be.
- **Everything from [3b] down is OFF BY DEFAULT.** `--event-year`, `--rollup`,
  `--record-mode`, `--associate` all have to be passed. Defaults reproduce the older numbers
  on purpose.

---

## 3. EKF pipeline — AS DESIGNED

`EKF_MHT_DESIGN.md` §3. Stages 0-3 are the same; everything from association onward differs.
The design's claim is *diarization*: track AND assign, jointly, with decisions deferred.

```
  ... [0][1][2][3] as built ...
        |
        v
  observations  (t, role, value, qualifier, source)   -- NO event_key yet
        |                                                the design never commits
        v                                                to a key at extraction time
  ============================ MHT ============================        ⛔ NOT BUILT
        |
  [M1] GATE          which existing tracks could this observation belong to?
        |   validation gate on (predicted value, innovation covariance)
        |   scope hierarchy makes part-vs-whole a TYPED question, not a magnitude one
        v
  [M2] COST MATRIX   -log likelihood of observation i under track j
        |   + track-birth cost (a new event) and missed-detection cost
        v
  [M3] ASSIGNMENT    Hungarian -> the best single assignment
        |            ... then the top-K assignments, not just the best
        v
  [M4] HYPOTHESIS    each assignment spawns a child hypothesis; score it;
       TREE          prune to N-best; DEFER the choice
        |   this is the whole point: a decision made now is revised when
        |   later evidence arrives, instead of being locked in by a threshold
        v
  [M5] TRACK         birth: an unassignable observation starts a stream
       MANAGEMENT    death: a stream with no support decays out
        v
  ============================================================
        v
  [4'] PER-TRACK FILTER   one EKF per surviving track, as built
        |   + aggregate as a SUM ROW over components (vector state)     ⛔ MEASURED
        v                                                                  NEGATIVE
   tracked state, with per-hypothesis weights rather than one hard answer
```

### Where as-designed and as-built diverge, and what the measurements say

| design element | status | evidence |
|---|---|---|
| deferred assignment (M4) | ⛔ not built | headroom **+0.111** once the null hypothesis is priced — see §4 |
| track birth/death (M5) | ✅ built, ⛔ **measured negative** | 0.608 best against the magnitude gate's 0.591 — see §4.1 |
| aggregate as sum row | ⛔ **do not build** | loses at every density except 80%, worst where predicted to win (§23) |
| implied-max reference | ⛔ **do not build** | 2.590 vs 0.591 on Helene (§25.5) |
| hard key assignment | ✅ built | what ships; the scope gate patches its worst failure |
| negative supervision (data-side) | ✅ built, ⛔ **superseded** | a plausibility ceiling beats it — see §4.2 |
| per-event plausibility ceiling | ✅ built | 378.809 → 18.287 on the production model — §4.2 |
| scope membership | ✅ built | 4/6 cross-event at 7.3% FP, no model — 21 filters → 6 |
| span-embedding router | ⛔ blocked on the front end | no current model emits trigger+argument spans on wire copy — §4.4 |
| EKF front-end model | 🔄 in build | cold-start mmBERT; English trigger→argument 798 → ~39,800 — §4.4 |
| scope hierarchy declared | ✅ built | `rollup.json` `hierarchy` block |

---

## 4. Is the line far enough along for MHT? — measured, and the answer is no

The question is not "is MHT good" but "how much is left on the table for *any* better
association". That is measurable without building anything: assign every observation to the
scope it actually fits, using ground truth. It is a ceiling, not a method.

    shipped scope gate            0.591
    oracle hard association       0.537
    headroom                     +0.055     (9.3% relative)

**Perfect association buys 0.055.** MHT is a hypothesis tree, a cost matrix, Hungarian
assignment and track management — a large subsystem — to compete for a 9% residual.

**Corrected 2026-08-19: that oracle prices the wrong thing.** It is *two-way* — every
observation goes to its own place or to Total — so a figure belonging to no Helene scope at
all has no correct home, and it scores Katrina's 1,400 as badly as the shipped gate does.
The tell was already in the per-place table below and was read as a curiosity: the gate
**beats** the perfect oracle on Florida and South Carolina, because the gate can *drop* and
the oracle cannot. MHT's track birth/death **is** a null hypothesis, so the two-way oracle
never priced the version of MHT worth building. Adding a reject option
(`oracle_gate_three_way`, swept over tolerance rather than fixed at one lucky value):

    tol   kept    per-place mean
   2.00    100          0.533
   1.00    100          0.533
   0.50     85          0.480
   0.25     76          0.499

    shipped scope gate            0.591
    three-way oracle              0.480
    corrected headroom           +0.111     (18.8% relative)

Still a ceiling that uses ground truth, still one event, and the tolerance is tuned and
non-monotone — 0.25 is worse than 0.50, so there is no plateau to hide behind. But the prize
for association is **double** what MHT was rejected on, and it splits almost evenly: 0.591 →
0.537 is reassignment (+0.055, mostly Tennessee) and 0.537 → 0.480 is the reject option
(+0.057). **About half the prize needs a null hypothesis; the other half does not.**

The per-place breakdown says something sharper:

| | shipped gate | oracle |
|---|--:|--:|
| Florida | **0.704** | 0.734 |
| Georgia | 0.553 | 0.553 |
| South Carolina | **0.365** | 0.558 |
| North Carolina | 0.518 | 0.518 |
| Tennessee | 0.817 | **0.320** |

The gate already **beats** a perfect two-way assignment on Florida and South Carolina,
because it has a third option the oracle lacks: *drop*. Florida's 300 and North Carolina's
1400 are not misassigned: 300 is not a casualty figure and 1400 is **Hurricane Katrina's**
toll quoted inside a Helene article. Neither belongs to any Helene scope, so no association
scheme can place them correctly.

Tennessee is the one genuine association gap (0.817 vs 0.320), and it is instructive: its
contaminants are 32, 32, 32, 36, 50 against a truth of 18 — **too large for the state, too
small to look national**, so a magnitude test cannot catch them. That is the case for richer
association, and it is worth 0.055 across the event.

**And 0.537 of the 0.591 is not association error at all.** Context audit of all 106
observations puts that residual at cross-event contamination (4.7%), non-casualty numbers
(3.8%), a 9.4% unclear tail, and the filter itself — *not* at starvation. See the conclusion
below; the recall claim that used to sit here was stale.

### Conclusion

Not yet, and MHT is probably not what is missing. In order:

1. ~~**Extraction recall**~~ — **already solved, and the claim behind this was stale.**
   `extract_long` took the Helene feed from 25 dead observations to 106 (4.2x) and article
   coverage to 44 of the 45 articles that contain a casualty-bearing sentence. The feed has
   ~86 sentences carrying both casualty language and a digit, so at 106 the pipeline
   over-extracts rather than starves. The "25 observations from 70 articles" figure quoted
   here and in `EKF_MHT_DESIGN.md` §6.2 came from the superseded `tracked_lead` run.
2. **Cross-event contamination — the real top item.** Context audit of all 106 observations:
   82.1% are genuine Helene casualties, **4.7% belong to another event** (Katrina 1400,
   Typhoon 250, Milton 230, Bosnia 16, Mexico's Hurricane John 2), 3.8% are non-casualty
   numbers (mph, inches), 9.4% unclear. Cross-event carries the *large* values, so it does
   the most damage per instance.
3. **Non-casualty number rejection** — smaller and easier: speeds, durations, rainfall.
   Pure extraction typing; no tracker can help.
3. **Multi-source feeds** — the one thing that would make a filter benchmark meaningful, and
   the one regime where MHT's deferred assignment genuinely earns its keep, because
   *sources disagreeing about the same event* is real association ambiguity in a way that
   one wire service's copy is not.

MHT becomes the right move at step 3, not before — with two amendments. Its prize is 18.8%
rather than 9%, and the cheapest piece of it has since been built and lost (§4.1). The
"starved" clause in the earlier version of this sentence was also stale: the audit above
shows the pipeline over-extracts rather than starves.

### 4.1 M5 track birth: built, and it loses to the fixed ratio it replaces

`tools/ekf_showcase/mht_associate.py`. Tracks advance jointly in time order; each observation
is tested by normalized innovation against its candidate tracks, and one that gates out of
all of them is born into its own and leaves these streams. No ground truth anywhere.

    arm                                          Total   per-place   (sigma 4.0)
    no gate                                      0.402       5.247
    symmetric birth, own+aggregate               0.555       1.059   q_rel 0.20 (the filter's)
    symmetric birth, tuned                       2.115       0.636   q_rel 2.00
    one-sided birth, tuned                       2.115       0.608   q_rel 2.00
    aggregate-only reference                     0.387       0.624   q_rel 0.20
    SHIPPED magnitude scope gate                 0.316       0.591
    three-way oracle (ground truth)              0.308       0.480

**Read the pair, and the pair is what makes this an unambiguous loss.** The two tuned arms
buy their per-place improvement by dumping junk into the aggregate: Total 2.115 against the
gate's 0.316, a 6.7x degradation of the one stream this project calls its honest measurement.
That is precisely the failure the shipped gate's three-outcome design exists to prevent —
`gate()`'s docstring records that an earlier two-way version rerouted every reject to
`__aggregate__` and destroyed the national stream. The associator reproduced a solved bug.
So "0.608 against 0.591" understates it; the honest comparison is 2.115/0.608 against
0.316/0.591.

Two causes, both measured:

1. **Judging a stream against its own track is circular.** Every contaminant the track
   accepts moves the reference the next test uses. Removing the self-reference is worth more
   than every other knob combined — 1.059 → 0.624 at native dynamics. This is the same
   failure the implied-maximum reference hit on Türkiye, where Turkey was judged against a
   reference Turkey itself defines, so it is now **two independent mechanisms defeated by one
   cause**.
2. **The innovation is not informative about scope on a rising toll.** At the filter's own
   `q_rel = 0.20` the tracks are far too tight to admit real growth — Georgia keeps only
   `[2, 3]` against a truth peak of 34 — and the sweep must reach `q_rel = 2.00` before real
   rises survive, by which point only 1–4 observations are ever born. Birth is never the
   lever; the rerouting is.

**This strengthens rather than kills the case for M4.** Deferred assignment is the one piece
that addresses cause (1) directly: hard assignment commits early and poisons its own
reference, which is exactly what keeping rival hypotheses alive exists to prevent. Before
this run that was a design preference; it is now the mechanism a measurement implicates.

### 4.2 The data-side route: built, and beaten by a threshold

The other route to cross-event contamination is negative supervision — withhold an
interfering event's records while keeping its text, so its figures become negatives for the
same queries. `casualty_loc_muted`, two arms trained identically for four epochs.

**It works as a treatment.** Blind-test precision up and recall down (0.8119/0.8182 →
0.8273/0.7754), and on Helene it removes 15 of the control's 20 large false positives, cutting
ungated per-place error 46.844 → 19.822.

**Then a declared per-event plausibility ceiling beats it.** Auditing what those large values
actually are found that none is a Helene death toll: Asheville's population (94,000), Boone's
(19,000), FEMA flood-insurance *policies* (129,933), wellness checks (15,000), power crews
(8,000), active-duty troops (1,500), churches (1,100), and two years read as tolls (1,916,
2,004). Only Katrina's 1,400 and Maria's 3,000 are genuinely cross-event.

Ungated per-place mean, ceiling swept:

    ceiling   production   control 4ep   muted
    off          378.809        46.844  19.822
    20000         18.287        26.961  19.822
    2000          18.190         5.853   6.194
    1000          18.190         5.057   6.194

At a ceiling of 2,000 — nine times Helene's true toll — **the control beats the muted arm both
ungated (5.853 vs 6.194) and gated (3.336 vs 3.729)**, while carrying 81 *more* observations.
The ceiling removes only junk; muting removed genuine signal too. Dropping the single 94,000
is worth 20× on the production model.

**Verdict: superseded.** The arm's pre-registered guard passes only against an undefended
control. The suppression is real and learned; it is not worth having.

**And neither mechanism touches cross-event.** Both true cross-event tolls survive muting and
the ceiling alike — at 2,000 Katrina's 1,400 is a plausible magnitude, and a ceiling low
enough to catch it is the magnitude gate again. The same holds for the troops and crews:
living people in the affected area, wrong for a reason no ceiling sees and no entity-type
check reaches. That residue — living people miscounted as dead, and another storm's dead — is
what still needs the model to represent who a figure is *about*.

### 4.3 The Helene reference is a cached artifact and cannot be regenerated

Everything in §4, §4.1 and §4.2 above that cites 5.247 / 0.591 / 0.537 / 0.480 reads one
cached file written 2026-08-10, which **no committed state of the repository reproduces**: the
`--rollup` flag did not exist in any commit before it was written, and the rollup file was not
in the tree either. Comparisons among those figures stand — one frozen artifact — but no new
model can be placed on their scale, which is why §4.2 uses a fresh baseline. Full evidence in
`../ekf_showcase/muting_arm_results/PROVENANCE.md`; `run_pipeline.py` now records its full
invocation and a `-dirty` git marker in every output.


### 4.4 The front end was rebuilt, and the rebuild did not fix the router's input

The association work above all assumes an extractor that produces, per event, a trigger and
arguments bound to *that* trigger. Measured 2026-08-20, nothing in the line does.

**Span architecture** emits a *bag* of triggers and labels every one with the same role. No
threshold works: at 0.4 and below the Katrina block is `[95:119]`, the bare name, missing the
1,400 it should bind, while the Helene block runs `[0:212]` and swallows Katrina; at 0.5+
Katrina yields nothing.

**Boundary mmBERT `137k-clean`** yields nothing at trigger/argument threshold 0.3 and above on
English disaster text, and nonsense at 0.1 — trigger `"remote"`, with both `dead` and
`location` bound to `"Helene decimated"`. A textbook earthquake sentence returns empty even at
0.1, despite the model scoring trigger 0.710 / argument 0.506 on its own test set.

> **Corrected 2026-08-23.** Two things here were wrong. (a) 0.710 / 0.506 are **relaxed**
> numbers on the model's own test set; like-for-like on the shared blind test at a pinned
> 0.5 the base is strict 0.7487 / 0.0913, fair 0.7523 / 0.4939. (b) The "no events above
> 0.3, nonsense at 0.1" reading came from a harness whose threshold sweep was inert — it
> set a Schema value the boundary decode never reads, so every row ran at 0.5. Re-measured
> properly the base forms usable events on 0.0 / 0.0 / 8.3 / 20.0 / 65.0% of Helene windows
> across 0.5→0.1. The qualitative premise stands; "~0 at every threshold" did not.

**The cause is the mix, and it is arithmetic.** Counting only corpora that bind arguments to a
trigger — DocEE, ChFinAnn and DocFEE do not; they are stored as `entities` + `classifications`:

| | English | Chinese | English share |
|---|--:|--:|--:|
| `137k-clean` as built | **798** | 20,884 | **3.7%** |
| every available corpus | 39,783 | 20,884 | 65.6% |

The English side is CASIE alone. MAVEN and Mendeley are trigger-only. So an argument F1 of
0.506 is very nearly a Chinese-only number, and English trigger→argument rests on 798 examples
— which no threshold reaches.

**The rebuild was run, and it refutes the inference drawn from this table.** 50× more
English trigger→argument supervision produced a model that beats the base on all eight
held-out heads and forms usable events on **25%** of Helene windows against the base's
**65%**. The English-share arithmetic below is correct; the conclusion that it was the
binding constraint is not. See EKF_MHT_DESIGN §7.6.

`tools/train/config/ekf-frontend-mmbert.yaml` is the cold-start rebuild: 189,284 records, a
50× increase in English trigger→argument, the Chinese corpora kept because they are why the
argument head works at all. Split gate clean at 180,660 / 11,486 / 20,571.

**The risk is in the config, stated before spending:** 72% of the new English trigger+argument
data is synthetic, against 20% human-annotated real news, on a line whose recurring failure is
in-domain-good / real-news-zero. The gates are on AP prose for that reason.

**Smoke, 1× A100-SXM4-40GB:** 18.4 samples/s — 34% faster than the extrapolation every earlier
cost estimate used. `num_workers` is *not* the bottleneck (18.4 at 0 workers, 18.5 at 4);
utilisation swings 28–81% on variable sequence length while memory stays flat at 10.6 GB of 40.
