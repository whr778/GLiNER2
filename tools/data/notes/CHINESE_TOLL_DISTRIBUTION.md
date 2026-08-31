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

---

# TODO -- OWED AFTER THE CHINESE EXTRACTOR EXISTS

**Status: NOT DONE. Deferred deliberately, not forgotten.** Deferred because step 2 below
needs reliable (t, toll) extraction from Chinese text; doing it with a regex would confound
measurement error with the effect being measured.

## Why this must actually happen

The EKF **is** a trajectory model: it estimates a toll being revised upward as reports
arrive, and `max_plausible` is an explicit magnitude prior (2,000 on Helene, chosen because
94,000 was Asheville's population). If Chinese domestic events plateau earlier or revise
less than the English events the filter was tuned on, it will systematically mis-track
them and the ceiling will be set wrong. This is filter validation for a multilingual
pipeline, not media studies.

## The design, and why it settles what the cross-sectional test cannot

Selection explains WHICH events get covered. It has no mechanism to act on how a covered
event's toll evolves. Suppression does. So the trajectory discriminates where the
magnitude comparison cannot:

| signal | selection predicts | reporting control predicts |
|---|---|---|
| cross-sectional magnitude | domestic lower | domestic lower |
| time to plateau | no difference | domestic plateaus EARLIER |
| count of upward revisions | no difference | domestic FEWER |
| first-report / final ratio | no difference | domestic CLOSER TO 1 |
| downward revisions | no difference | domestic MORE likely |

## Steps

1. **Group articles by event** -- cluster on entity plus a date window within `news_zh`.
2. **Extract (t, toll) pairs** with the Chinese extractor. NOT with a regex.
3. **Compare trajectory shape**, domestic against foreign, controlling for final magnitude.
4. **Anchor on independent ground truth**, the same way the EKF work uses Wikipedia
   casualty tables for Helene, Turkiye and Aegean.

## The data is already in hand

A 5,000-document sample -- 0.2% of the corpus -- already contains multi-article coverage
of single events, with published official tolls to anchor against:

| event | articles in the 5k sample | independent toll | |
|---|--:|--:|---|
| Shenzhen 2015 landslide | 47 | 77 | domestic |
| Tianjin 2015 explosions | 19 | 173 | domestic |
| Nepal 2015 earthquake | 4 | ~8,964 | foreign |

Cost is analysis time: no annotation spend, no GPU.

## Known obstacle

`news_zh`'s `time` field carries NO year (`MM-DD HH:MM`), so trajectory timing must come
from dates inside the article text. Workable for major events, which are heavily
date-stamped, but a reason to run this on a few well-covered events rather than
corpus-wide.
