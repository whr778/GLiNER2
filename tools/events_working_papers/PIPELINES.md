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
| deferred assignment (M4) | ⛔ not built | headroom measured at **+0.055** — see §4 |
| track birth/death (M5) | ⛔ not built | untested; nearest real need is cross-event contamination |
| aggregate as sum row | ⛔ **do not build** | loses at every density except 80%, worst where predicted to win (§23) |
| implied-max reference | ⛔ **do not build** | 2.590 vs 0.591 on Helene (§25.5) |
| hard key assignment | ✅ built | what ships; the scope gate patches its worst failure |
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
   here and in EKF_MHT_DESIGN §23 came from the superseded `tracked_lead` run.
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

MHT becomes the right move at step 3, not before. Building it now would optimize a 9%
residual on a single-source feed whose real problem is that it is starved.
