# Chasing the 94,000: what the large `dead` values actually are

The Helene reproduction put **94,000 into Tennessee**, wrecking that stream (nRMSE 2,163
against a truth peak of 18). Chased 2026-08-20. It is not one bug; it is a taxonomy, and
tracing it corrected a claim made earlier the same day.

## The 94,000

> "The hub of tourism and arts, **home to about 94,000 people**, was unusually still after
> floodwaters swamped neighborhoods..."

**Asheville's population.** Two independent failures stack:

1. **Typing** — a population is read as a casualty count. It is emitted as `dead` *and*
   `injured` *and* `missing`, same span, confidence 1.0. Filling every numeric field with
   one number is the field-collapse signature `_locate_place` documents.
2. **Attribution** — filed under `tennessee`, because the nearest place mention is a
   headline fragment, "dozens stranded on Tennessee hospital roof". The passage is about
   Asheville, North Carolina.

## Every large value is a false positive, in four classes

All eight `dead` values >= 400 from the production model. **None is a Helene death toll.**

| value | source text | class |
|---:|---|---|
| 94,000 | "home to about 94,000 people" (Asheville) | population |
| 3,100 | "a town of around 3,100" (Saluda) | population |
| 19,000 | "a town of about 19,000 people" (Boone) | population |
| 129,933 | "North Carolina has 129,933 such **policies** in force" (FEMA flood insurance) | count of non-people |
| 15,000 | "his office said it had done 15,000 **wellness checks**" | count of non-people |
| 8,000 | "8,000 **crews** are out working to restore power" | count of non-people |
| 1,500 | "total number of active-duty **forces** to about 1,500" | count of non-people |
| 1,100 | "of the convention's 3,000 churches, **1,100** are in communities affected" | count of non-people |
| 2,004 | "**In 2004**, for example, four people were killed" | year-as-toll |
| 1,916 | "one of the most significant weather events to happen since **1916**" | year-as-toll |
| 1,400 | Hurricane **Katrina**, "left nearly 1,400 people dead" | cross-event |
| 3,000 | Hurricane **Maria**, "which killed 3,000 people" | cross-event |

The 15,000 deserves its own note: the sentence exists *to warn against this exact error* —
"that was mistakenly interpreted as meaning 15,000 people were missing." The extractor made
the mistake the article is correcting. And 2,004 sits in "In 2004... **four** people were
killed", so the true value, 4, is in the same clause as the number that displaced it.

## The temporal filter does not catch any of it

Re-run with `--event-year 2024`: **`rejected 0 figure(s) dated before 2024`.** The filter
that took Izmit contamination 15 -> 3 on Türkiye fires zero times on Helene, including on
the two year-as-toll cases that are *literally* dates. Worth knowing before it is relied on
again.

## What muting actually fixed, and it is NOT what it was built for

Large `dead` values >= 400, same pipeline settings:

| arm | dead obs | values >= 400 |
|---|--:|---|
| `casualty-docee` | 88 | 8 values |
| control 4ep | 205 | **20 values**, incl. 129,933 · 83,000 · 19,000 · 15,000 · 8,000 · 7,600 · 7,000 |
| **muted** | 109 | **5 values**: 1,100 · 1,400 · 1,500 · 1,916 · 3,000 |

**Muting eliminates 15 of the control's 20**, including every value above 3,000 and five of
seven year-as-toll cases (1848, 1968, 2000, 2005, 2023 gone; 1916 survives).

**But both cross-event tolls survive — Katrina's 1,400 and Maria's 3,000.** That is the class
the arm was designed against. What it removed instead is the *non-casualty number* class,
which `TODO.md` lists as a separate and smaller item (3.8%, "pure extraction typing; no
tracker can help").

So the arm works, for a different reason than intended.

## Correction to this directory's README

The earlier write-up said North Carolina's 127.504 -> 0.640 was driven by "Katrina's 1,400 —
the single largest cross-event contaminant". Half right, and the important half was wrong.

North Carolina, control, truth peak 123:

    1,1,1,1,1,1,1,1,2,3,4,6,13,16,26,27,30,50,57,57,57,61,72,90,91,100,100,180,200,230,
    400,1400,1400,1500,2000,8000,15000,19000,129933,129933

Muted, same stream: `1,1,16,16,57,57,57,61,61,80,80,91,98,100,100,180,180,200,230,230` —
max 230.

Katrina's 1,400 **is** removed from North Carolina (it survives in the muted run only under
`reading pennsylvania`, the dateline of the speech that quotes it, where it corrupts no
scored stream). But the error magnitude was dominated by **129,933 flood-insurance
policies**, not by Katrina. The win is real; the mechanism attributed to it was not.

Maria's 3,000 tells the same story: control files it under `georgia` twice, muted once.
Re-attribution away from scored streams, not suppression.

## What follows

- **Non-casualty typing is worth more than its 3.8% billing.** It carries the largest values,
  so it dominates nRMSE, and one figure destroyed one state's stream.
- **A cheap sanity filter is available and is not implemented**: no US state lost six-figure
  numbers of people to this storm. A per-event plausibility ceiling would have killed
  129,933, 94,000, 83,000 and 19,000 without any model change. It would not touch Katrina.
- **Cross-event remains unsolved**, and this run is evidence *against* reading the muting
  arm's Helene win as progress on it.
