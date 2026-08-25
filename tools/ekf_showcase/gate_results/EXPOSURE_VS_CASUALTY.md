# Exposure counts are not casualties, and the schema has nowhere to put them

Measured 2026-08-25 against the 86 hand-audited Helene occurrence labels.

Twelve of the 106 `dead` observations (11%) are audited **non-casualty**. They split into
two failure modes that need different fixes:

## 1. Exposure counts mis-bound as deaths (6 of 12)

    300   "the sheriff's office RESCUED more than 300 people overnight"
     50   "more than 50 patients had to be RESCUED"
     32   "tried to EVACUATE 11 patients and dozens of others"
     11   "the raging waters SWEPT 11 people away, and only five..."   (x3 captures)

A rescued person is a **counterfactual casualty** -- someone who would have become one.
It measures averted harm, not realized harm, and belongs in a different stream.

## 2. Unit confusion -- counting things that are not people (6 of 12)

      2   "a TWO-day period"                    (a time span)
      6   "across SIX Southeastern states"      (states)
     32   "DOZENS of vehicles"                  (vehicles)
     40   "at least 40 advocacy groups"         (organisations)
   1400   "causing 1,400 LANDSLIDES"            (landslides)

Note 1400 appears twice in this feed as two different traps: Katrina's death toll
(cross-event) and Helene's landslide count (unit confusion). Same number, same stream,
different fixes.

## Why this matters more than its 11% suggests

Exposure counts are **systematically larger** than casualty counts in disaster copy --
505 displaced against 6 dead in the synthetic corpus, 300 rescued against Florida's true
peak of 26. So a mis-bound exposure number is the same magnitude of error as cross-event
contamination and points the SAME direction, upward. The scope gate therefore catches some
of them, but for the wrong reason (magnitude, not category) -- the gate-1 lesson again.

## The root cause is the schema, not the gate

`casualty_events` has exactly four roles:

    location 53,777 | injured 26,341 | missing 22,373 | dead 22,070

There is **no role for exposure**, yet the prose is full of it ("roughly 505 residents
were displaced"). A number with no correct home lands in a wrong one. Adding `displaced`
and `rescued` as explicit roles is a data-modelling fix and it removes the ambiguity at
source rather than filtering downstream.

## Taxonomy

| class | roles | tracked by the EKF? |
|---|---|---|
| realized harm | dead, injured, missing | yes |
| averted / exposure | rescued, evacuated, displaced, stranded, homeless | no -- separate streams |
| non-human units | vehicles, buildings, landslides, states, days | never a casualty observation |

**Armed conflict** adds an axis rather than a class: KIA/WIA are realized harm, non-battle
casualties (vehicle accidents, disease) are realized but have different dynamics and
deserve their own stream, and BELLIGERENT SIDE is the scope problem again -- "300,000
Russian losses" is scoped to a side, not a place, with the same failure modes (one side's
claim about the other, totals aggregated across both). Captured/POW is exposure.

Caveat worth stating before anyone builds it: disaster tolls can be scored against an
official figure. Conflict casualty ground truth is contested by construction, so the
taxonomy could be built but not validated the way Helene and Turkiye are.
