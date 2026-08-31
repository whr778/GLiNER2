# Chinese domestic tolls are reported smaller than foreign ones

Measured 2026-08-31 on 4,994 Haiku-adjudicated articles from `shaowenchen/news_zh`
(2014-2016), 1,775 of them `current_toll`. The figure compared is the largest number in
the adjudicator's verbatim evidence quote.

## The measurement

Same corpus, same outlets, same period. Domestic articles are those naming China or a
Chinese province/major city and no foreign country; foreign articles the reverse.

| | domestic (China) | foreign |
|---|--:|--:|
| tolls | 588 | 194 |
| median | **17** | 26 |
| p90 | 259 | 356 |
| tolls >= 100 | **15.3%** | **28.4%** |

    odds ratio 0.46, exact Fisher p = 1.1e-04
    Mann-Whitney z = -5.85 (domestic ranks lower)

The difference is real and strongly significant.

## What it does NOT establish

**Newsworthiness selection is a sufficient alternative explanation.** Foreign disasters
reach Chinese domestic news mainly when they are large: a three-fatality road accident in
Brazil is not covered, while the same accident in Sichuan is routine local news. That
selection effect predicts exactly this result with no editorial suppression required.

The data cannot separate the two explanations, and framing effects make the comparison
worse rather than better -- coverage of a foreign disaster can serve domestic narrative
purposes, which is another reason foreign-vs-domestic is not a like-for-like baseline.

**What would disambiguate:** comparing reported figures against an external ground truth
for the SAME events, which is what the EKF work already does with Wikipedia casualty
tables for Helene, Turkiye and Aegean. A Chinese event with an independent toll record
would settle it; the corpus alone cannot.

## Why it matters anyway

The practical consequence holds under either explanation, because both leave the same
distribution in the training data.

**For the EXTRACTOR: harmless.** It learns to bind a number to a place from whatever the
text says. A distribution shifted low does not change what `dead` means.

**For the EKF and for `max_plausible`: not harmless.** The filter models a toll as a
quantity revised upward as information arrives and is tuned on the shape of that revision;
the plausibility ceiling is an explicit magnitude prior (2,000 on Helene, chosen because
94,000 was Asheville's population). Both encode magnitude expectations, and a Chinese arm
carrying a domestic distribution shifted low will push both downward.

**Do not pool Chinese streams with English ones without checking.** Per-language tracking
error is the measurement that would expose it, and it costs nothing once a Chinese model
exists.

## Also true of the corpus

CommonCrawl reaches Chinese sites from outside, so CC-News captures the outward-facing
subset by construction. `news_zh`'s named publishers are state media (Xinhua, China News
Service). Neither observation is needed for the measurement above -- it is within-corpus --
but both bear on how far it generalises.
