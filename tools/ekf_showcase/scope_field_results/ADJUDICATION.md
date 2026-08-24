# Adjudication of the scope-probe disagreements

Adjudicated by hand, 2026-08-24, on the fixed-instrument run (kappa 0.121, 119
disagreements, 104 of them place<->sub-place).

## Verdict: 7-7. Neither labeller is systematically right.

Fourteen place<->sub-place disagreements were read in full and judged:

| # | figure | bound location | opus | haiku | correct | why |
|---|---|---|---|---|---|---|
| 1 | 6 dead | Belle Harbor neighborhood of Queens | place | sub-place | **opus** | deaths are IN Belle Harbor, the bound location |
| 2 | 2,089 dead | West Texas cities of Midland and Odessa | sub-place | place | **haiku** | "across three districts IN the cities" = the cities' total |
| 3 | 12,744 injured | Baghdad | sub-place | place | **haiku** | "across 2 districts OF the capital" = Baghdad's total |
| 4 | 665 missing | Baghdad | sub-place | place | **haiku** | not district-scoped at all |
| 5 | 3,671 missing | Iraq | sub-place | place | **opus** | "three districts in Iraq" is a proper subset of Iraq |
| 6 | 14,326 injured | town of Sopore | place | sub-place | **opus** | the toll is for the fighting in Sopore itself |
| 7 | 12,788 injured | near Damascus | place | sub-place | **opus** | "across the seven affected districts" = the bound area |
| 8 | 9 missing | Xinjiang Uygur AR | sub-place | place | **haiku** | a mine is a SITE, not a smaller area |
| 9 | 1,502 dead | town of Sopore | place | sub-place | **opus** | as #6 |
| 10 | 9 dead | Myanmar | sub-place | place | **opus** | "the jade mining region in Myanmar" is a proper subset |
| 11 | 832 dead | eastern Syrian province of Hasakah | sub-place | place | **haiku** | "across two districts OF the province" = province total |
| 12 | 2,727 missing | town of Sopore | place | sub-place | **opus** | as #6 |
| 13 | 67 injured | MEXICO CITY | sub-place | place | **haiku** | neighbourhood is the incident SITE; toll is the event's |
| 14 | 84 injured | MEXICO CITY | sub-place | place | **haiku** | as #13 |

opus 7, haiku 7.

## The label scheme is the defect, not the model

`sub-place` conflates two different things:

(a) **The counting unit is genuinely narrower than the bound location.**
    "three districts IN Iraq", "the jade mining region IN Myanmar". Here `sub-place`
    is correct and load-bearing: the figure must not be read as Iraq's total.

(b) **The incident SITE is narrower, but the number is the event's full toll.**
    "a mine collapse in Xinjiang", "a gas explosion in a Mexico City neighbourhood",
    "fighting across 2 districts OF Baghdad". The toll IS the bound place's toll for
    that event. Labelling it `sub-place` makes `apply_extracted_scope` DROP a valid
    observation -- the pipeline loses a real casualty figure.

Both labellers flip between (a) and (b), in both directions. The linguistic tell is
nearly mechanical and neither model uses it reliably:

    "across N districts OF X"  -> aggregate over X      -> place
    "N districts IN X"         -> a proper subset of X  -> sub-place

## Consequence

Do not buy labels under this scheme. Rewrite the question to be about the COUNTING
UNIT rather than the incident site, e.g.

> Does this number count ALL casualties the report attributes to the named location
> for this event (place), only a part of that location's toll (sub-place), or a wider
> area such as a whole country (national)?

Then re-run the probe. Kappa 0.121 is the number to beat, and the pre-registered bar
stands.

Independently: `national` is 0.0% of both strata in this run. Even a fixed scheme
cannot teach place-vs-national from this corpus, which remains the blocking issue for
the EKF gate.
