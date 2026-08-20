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

  SCOPE MEMBERSHIP is a free first cut, and it beats every learned signal tried.
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

---
Review, open questions and feasibility: EKF_PIPELINE_VISION_REVIEW.md
