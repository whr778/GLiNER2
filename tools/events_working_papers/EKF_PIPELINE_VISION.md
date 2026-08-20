An EKF/MHT pipeline

Input: News text ==> Joint Boundary Event Model (The JOINT_IE_SCALING models)
Output: Extracted Events

ROLE / PLAUSIBILITY stage -- before any routing

  A figure has to be a casualty count before it is worth asking whose it is.
  Re-audit of the Helene feed, by context: 11.3% of extracted `dead` figures are not
  casualty numbers at all -- Asheville's POPULATION (94,000), FEMA flood-insurance
  POLICIES (129,933), power crews (8,000), wellness checks (15,000), advocacy groups,
  landslides, and years read as tolls (1916, 2004).
  Routing a population into a death-toll filter corrupts that filter no matter how
  good the router is.

  Part of this is already free -- the derived tail cut: median + k*MAD on log10 of the
  event's own pooled values, upper tail only, k=1. It derives its own threshold per
  event (Helene 516, Turkiye 47,622) so it transfers without being told the scale, and
  it obeys the RULE below (pooled over the event, never over what a stream accepted).

  LIMIT, so it is not mistaken for attribution: it is anomaly detection, and it reaches
  only the distributionally weird tail. Against the audited cross-event cases it catches
  1 of 5 on Helene and 0 of 5 at Turkiye's scale. Milton's 230 and Bosnia's 16 are
  statistically ORDINARY Helene figures -- wrong by identity, not magnitude. So is
  "1,500 troops": people, in the affected area, wrong only in ROLE.

Extracted Events evaluation via Event Instance Map

  Keyed on INSTANCE, not on type.
  Type carries no signal for this decision: Helene, Katrina, Milton and Maria are all
  the same event type, so a type-keyed map routes Katrina's 1,400 into the Helene
  filter by construction. Measured -- ranking by the model's own type energies caught
  0/11 on cross-event contamination, "because the type is RIGHT there" (Katrina's
  1,400 scores death toll 0.95).
  Type stays useful one level up, as a coarse pre-filter: keep an earthquake figure
  out of a hurricane filter. It just cannot be the routing key.

  instance key = named event + time/place  -- Helene-2024, Katrina-2005
  the hard case is the instance with no name: Bosnia's 16 is a PLACE, not a storm,
  so place+time has to carry it

Event Instance Map where each event INSTANCE has a registered filters list 0..*

  SCOPE MEMBERSHIP -- BUILT, run_pipeline.py --scope-filter (tests/test_scope_filter.py)

  A free first cut, and it beats every learned signal tried.
  The declared hierarchy already says what is in scope (six states + __aggregate__ for
  Helene). Anything keyed OUTSIDE it is, by construction, not this event. Measured on
  the 106-observation feed with the corrected labels:

      scope membership (zero model calls)   4/6 cross-event caught,  7.3% FP
      fastino signal C (one call per obs)   4/6 cross-event caught, 31.7% FP

  Same recall, a quarter of the false positives, no model. It works because the
  contaminating events happened SOMEWHERE ELSE -- Mexico, Puerto Rico, Bosnia, Reading
  PA. That is declared knowledge, not a statistical property to be discovered, and
  rollup.json already records it: out-of-scope places are "deliberately NOT mapped:
  they are other events leaking through the gate". The pipeline records the signal
  today and does not act on it.

  Of the 6 remaining false positives, 4 are TYPE keys (Storm, Floods) that survive
  because no place was extracted. Those are "location unknown", a different disposition
  -- hold, or send to __aggregate__ -- not "out of scope". Handle them separately and
  the false-positive rate is ~2%.

  WHAT IT CANNOT DO, and this is the residue the router exists for: it misses the two
  cross-event cases whose places are IN scope -- a Taiwan typhoon's 32 keyed to
  tennessee, the 1916 hurricanes' 80 keyed to north carolina. Same place, different
  incident. Nothing about location reaches those.

  AND IT DOES NOT MOVE nRMSE -- measured, so nobody expects it to. On the control-4ep
  run, ungated per-place error is 46.850 with the stage against 46.844 without it.
  The reason is structural: out-of-scope observations form their own streams (fema,
  poweroutageus, scotland, gulf of mexico), and those streams were never among the six
  scored places, so the metric never saw them. What the stage removes was already
  invisible to the number.

  ITS VALUE IS EXACTLY THE THING THIS ARCHITECTURE INTRODUCES. Every distinct key
  spawns a FILTER here. On the archived feed that is 21 filters without the stage and
  6 with it; on a fresh control-4ep run, 8 streams out (six states + __aggregate__ +
  unknown) from 291 rejected observations. Junk filters are not free -- they are state
  to advance, buckets to route into, and candidates the matching function must score
  against. The stage is cheap upstream hygiene for the router, not a tracking gain.

Do we have any filters mapped to that instance?
No  -- track BIRTH: open a filter for this instance.
       "continue" drops the observation. The reject/birth option is worth about half
       the +0.111 association headroom, so it should not be dropped.

       CAUTION -- birth has been built once and it LOST.
       M5 track birth by innovation gating: each observation tested by normalized
       innovation against its candidate tracks, born into its own if it gated out of
       all of them. Best setting 0.608 per-place against the shipped magnitude ratio's
       0.591 -- and the per-place number was the flattering half. With the national
       stream included it is 2.115 against 0.316, a 6.7x degradation of the one
       measurement this project calls honest, because rejected junk was being assigned
       into the aggregate. It had reproduced a bug the scope gate's three-outcome
       design was written to prevent.

       WHY it lost, and the rule that follows:
       it judged each stream against ITS OWN track history, so every contaminant the
       track accepted moved the reference the next test used. Removing that
       self-reference was worth more than every other knob combined (1.059 -> 0.624).
       The same cause has now killed three mechanisms -- M5 birth, the implied-maximum
       scope reference on Turkiye, and the naive streaming form of the tail cut (which
       locked onto the small early values of a rising toll and rejected 88 of 106).

       RULE: judge against something the contaminant cannot move.
       A larger scope, or the whole event's pooled distribution -- never the accepting
       stream's own history. The two mechanisms that work both obey it.
Yes -- Discern which filter or filters apply

  Which filter? -- a MATCHING function, not a classifier.
  Score (figure, context) against each live filter's identity and current state, then
  assign / split / birth. This is MHT's cost matrix.
  Not a multi-type classifier: that re-routes on type, ruled out above.
  Not collaborative filtering: no user dimension, one event, 70 articles, and the
  signal is content-based rather than co-occurrence.
  THIS IS THE OPEN PIECE -- ceiling +0.111, four mechanisms tried and failed
  (nearest-named-event, only-competitor-named, record-head binding, type energies).

  CLUSTERING (agglomerative, mean shift) does not solve it, and the reason is not
  the algorithm. Clustering needs a metric in which instances separate, and measured
  on this feed they do not:
      value  -- 4 of 6 cross-event values fall INSIDE the genuine range (1..250)
      time   -- 5 of 6 fall inside the genuine span
  The feature that would separate them is a representation of "which event is this
  figure predicated of", and constructing that IS the open problem. Clustering
  CONSUMES a good representation; it does not produce one. Imbalance finishes the
  argument: 6 positives in 106, so the dominant structure is Helene and the six are
  singletons indistinguishable from noise.
  One thing in its favour, worth keeping: clustering pools globally, so it obeys the
  RULE above. If a representation ever exists, clustering over it is a sound way to
  form instances -- it is just not a way to get the representation.
  And scope membership above already does most of what clustering was hoped to do,
  with no metric to learn.

  CANDIDATE ROUTER: span-embedding retrieval (proposed 2026-08-20)
  Take min(start)..max(end) over an event's OWN trigger and arguments, vectorize that
  block, attach it to (doc, event), index it, and match a new observation against the
  live filters. Clustering forms filters; retrieval routes into them.

  This is the first proposal that PRODUCES a representation instead of assuming one,
  and the span construct is the right one -- it is per-event and local, so it is doing
  the discourse-attachment job that proximity and type both failed at. Verified on the
  hard case: in the Katrina passage the only named event the extractor finds is
  `Hurricane Katrina` at (102,119), so the block is Katrina-local, not Helene-local.
  (`include_spans=True` returns the offsets.)

  Three things decide whether it works.

  1. IT REQUIRES EVENTS-FORM OUTPUT, not records. min/max over trigger+arguments needs
     per-event argument bindings; a `casualty_report` record carries no event span. So
     this and the purpose-built events-all front end are the same decision, not two.

  2. A SINGLE POOLED VECTOR IS THE WRONG RETRIEVAL. Lexical cosine as a proxy, on hand-
     picked spans: cross-event scores 0.063-0.140 against a Helene centroid and genuine
     spans score 0.195-0.289 -- but the WITHIN-Helene spread is 0.127-0.316, so a
     genuine pair (0.127) is less similar than Katrina is (0.140). The classes overlap.
     And stripping the storm name barely moves it (0.140 -> 0.149), so what separation
     exists is incidental vocabulary, not the discriminating token.
     That is the argument for COLBERT-style late interaction over a single dense vector:
     MaxSim lets one token ("Katrina") dominate the match, which mean pooling averages
     away. FAISS still earns its place underneath, for candidate generation.
     (Caveat: TF-IDF on six spans is a proxy, not a measurement of a dense embedder.)

  3. THE SELF-REFERENCE TRAP APPLIES. If a filter's centroid updates with every
     observation it accepts, one contaminant moves the reference and the next match is
     wrong -- the cause that killed M5, the implied-maximum reference and the naive
     streaming tail cut. Anchor a filter on its DECLARED identity (name + place + date),
     or on periodically re-pooled exemplars. Never an incrementally drifting centroid.

  WHAT IT WILL NOT REACH: the 1916 case. The extractor finds only `hurricane` and
  `hurricanes` there -- no name at all -- so the block is generic hurricane-casualty
  language and the discriminating signal is temporal ("since 1916", "That year"). Same
  shape as Bosnia's 16. The vector should be PAIRED with date and place features rather
  than asked to carry identity alone.

  ONE REAL BONUS: it improves the evaluation problem instead of inheriting it. A
  classifier has 6 positives to learn from; a retrieval formulation has every
  (observation x live filter) pair -- with 106 observations and 8 filters that is ~848
  labelled comparisons from the same annotation. Ranking metrics are measurable where a
  6-positive classification is not.

Route text and extracted events to the 1..* filters
Events from the text are reformatted for each filter and added to the filter's bucket

  PROVENANCE on every bucket entry: which extractor and revision, which router
  decision and score, which thresholds, which commit (and whether the tree was dirty).
  Not bookkeeping -- this programme has already lost a headline result to it. The
  archived Helene artifact behind 5.247 -> 0.591 recorded only `associate`, and no
  committed state of the repo reproduces it, so no new model can be put on its scale.

EKF/MHT Filters reprocess and update

Filter DEATH / decay
  A stream with no support ages out. Without it every spurious birth is permanent, and
  birth is exactly the stage most likely to over-fire on a noisy feed.


=== BUILD THIS FIRST: the routing evaluation set ===

  Everything above is gated on it, including the claim that four mechanisms failed.

  The labels this line has been scored against were a (span, value) string match. Audited
  per occurrence 2026-08-20 they were 27% CORRECT on their own positive class -- 3 of 11.
  All six '230's were marked cross-event (Milton) when every one is Helene's OWN national
  total (truth 228). '250' likewise. One of the two '1,400's is "1,400 LANDSLIDES". And
  three genuine cases were missed entirely: Maria's 3,000, the 1916 hurricanes' 80, a
  Taiwan typhoon's "dozens".

  Corrected, the ENTIRE Helene feed contains 6 cross-event instances.

  You cannot train, tune or validate a matching function on n=6 from one event. That is
  also why "four mechanisms failed" is weaker than it sounds: a detector scoring 3/11 on
  the old set might have found exactly the three real cases or three of the eight false
  ones, and nothing distinguished those outcomes. Re-measured on the corrected labels the
  base model reads 4/6, not 3/11 -- the same model, a different instrument.

  What the set has to be:
    - labelled per OCCURRENCE, by reading context, never by value
    - keyed on a context hash so labels survive re-extraction
      (helene_audit_labels.json does this -- the feed text is fixed, so the same figure
       in the same passage keys identically in any pipeline run)
    - across SEVERAL events, so the cross-event cases are not all one value
    - across SEVERAL sources, which is also the only regime where deferred assignment
      earns its keep and where the filter's own 1-2% edge over last-value becomes testable

  Multi-source and multi-event feeds produce this set as a side effect. That is the
  argument for building them before the router, not after.


=== NEXT EXPERIMENT: the span-embedding router ===

  HYPOTHESIS
  A per-event span block, retrieved with late interaction, routes an observation to the
  right event INSTANCE better than the hard string key -- and better than the free
  scope-membership test.

  WHY IT IS WORTH A RUN. It is the first candidate that builds a representation rather
  than assuming one, it needs NO new training for a first pass, and it is cheap to
  falsify. Design and its three conditions are above, under "Which filter?".

  BUILD (first pass, all local, no GPU)
    1. events-form extraction over the Helene feed with include_spans=True
    2. per event: block = text[min(start) .. max(end)] over its trigger + arguments
    3. embed each block; index with FAISS
    4. score a new observation's block against each live filter; argmax, or birth
       below threshold
    5. filters anchored on DECLARED identity (name + place + date), not on a centroid
       that drifts with whatever it accepts

  THE BAR IS NOT THE STRING KEY -- IT IS SCOPE MEMBERSHIP
  A free, model-free test already gets 4/6 cross-event at 7.3% FP. Beating the string
  key proves nothing; beating a test that costs one dict lookup is the real bar.

  PASS / FAIL, fixed before spending

    PRIMARY -- and it is deliberately the hardest thing to fake:
      catch at least ONE of the two cases scope membership STRUCTURALLY CANNOT --
      the Taiwan typhoon's 32 keyed to `tennessee`, the 1916 hurricanes' 80 keyed to
      `north carolina`. Both are cross-event with an IN-SCOPE place. A router that
      recovers only the other four has reproduced a free test with a model attached,
      and should be reported as negative however good its aggregate score looks.

    GUARD  cross-event overall  >= 4/6 at <= 7.3% FP     (do not regress on the free test)
    GUARD  genuine observations stay on their own stream -- report per-place assignment
           accuracy, not just cross-event catches. A router that routes everything to
           __aggregate__ scores well on the primary and is useless.
    REPORT MRR and precision@1 over all (observation x live filter) pairs. With 106
           observations and ~8 filters that is ~848 labelled comparisons out of the same
           annotation -- the one formulation so far that is not stuck at n=6.

  ABLATIONS, each testing a stated condition rather than tuning
    late interaction vs mean-pooled single vector   -- tests the ColBERT claim. If they
        tie, the token-level argument is wrong and a dense vector is enough.
    anchored identity vs drifting centroid          -- tests the self-reference caution.
        If they tie, the trap does not bite here and the simpler form wins.
    span block vs whole document                    -- tests that LOCALITY is what helps,
        not just "embeddings help".
    + date and place features alongside the vector  -- the 1916 case has no name at all,
        so this is the arm that could reach it.

  BLOCKING PREREQUISITE: NO CURRENT MODEL CAN PRODUCE THE INPUT (measured 2026-08-20)

    Calibration was attempted and the answer is that calibration is not the problem.

    Span arch (fastino, kept only for posterity): emits a BAG of triggers and labels
    every one with the same role. No threshold works -- at 0.4 and below Katrina's block
    is [95:119], the name alone, missing the 1,400 it is supposed to bind; Helene's block
    becomes [0:212] and swallows Katrina. At 0.5+ Katrina yields no spans at all. The
    architecture does not bind arguments to their own trigger, so the block cannot be
    formed.

    Boundary mmBERT 137k-clean: on English disaster text the event head yields NOTHING at
    threshold 0.3 and above, and nonsense at 0.1 -- trigger "remote", with both `dead` and
    `location` bound to "Helene decimated". A textbook earthquake sentence returns empty
    even at 0.1.

    WHY, and it is not a mystery: 68% of that model's event supervision is Chinese --
    62,900 rows (ChFinAnn, DocFEE, DuEE, CMNEE) against 29,190 English. Its argument F1 of
    0.506 is carried by Chinese doc-level financial events. RAMS, the main English
    trigger-to-argument corpus, was deliberately excluded, as were the real-synth and
    synth arms, because the model was built to WARM-START other training and to support
    preservation tests -- not to be a front end.

    SO THE ORDER IS: rebuild -> calibrate -> run. Not calibrate -> run.

  WHAT THE REBUILD HAS TO CORRECT, from the evidence rather than from taste
    - English trigger-to-argument supervision. RAMS is the specific gap; the current
      argument ability is real and in the wrong language.
    - Event IDENTITY as a supervised field, not only type. Casualty fine-tuning
      demonstrably destroys binding, and the router keys on instance.
    - A REAL-WIRE-COPY GATE, not held-out DocEE. Note the caution: DocEE is ALREADY
      21,842 English disaster events and this model still does nothing on Helene copy, so
      "more event data of the same kind" is not automatically the fix. Gate on AP prose.

  ALSO STILL REQUIRED
    - the corrected per-occurrence labels (helene_audit_labels.json) are the only valid
      scoring set. Do not use the old string-matched labels; they were 27% correct.

  WHAT WOULD RETIRE THE IDEA
    It fails the primary -- catches neither in-scope case -- while scope membership sits
    there getting 4/6 for free. Then instance identity is not recoverable from the span's
    content on this data, and the next move is the multi-source feed, not a better encoder.

---
Review, open questions and feasibility: EKF_PIPELINE_VISION_REVIEW.md
