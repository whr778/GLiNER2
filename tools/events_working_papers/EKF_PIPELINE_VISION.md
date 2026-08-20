An EKF/MHT pipeline

Input: News text ==> Joint Boundary Event Model (The JOINT_IE_SCALING models)
Output: Extracted Events

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

Do we have any filters mapped to that instance?
No  -- track BIRTH: open a filter for this instance.
       ("continue" drops the observation; the reject/birth option is worth about half
        the +0.111 association headroom, so it should not be dropped)
Yes -- Discern which filter or filters apply

  Which filter? -- a MATCHING function, not a classifier.
  Score (figure, context) against each live filter's identity and current state, then
  assign / split / birth. This is MHT's cost matrix.
  Not a multi-type classifier: that re-routes on type, ruled out above.
  Not collaborative filtering: no user dimension, one event, 70 articles, and the
  signal is content-based rather than co-occurrence.
  THIS IS THE OPEN PIECE -- ceiling +0.111, four mechanisms tried and failed
  (nearest-named-event, only-competitor-named, record-head binding, type energies).

Route text and extracted events to the 1..* filters
Events from the text are reformatted for each filter and added to the filter's bucket

EKF/MHT Filters reprocess and update

---
Review, open questions and feasibility: EKF_PIPELINE_VISION_REVIEW.md
