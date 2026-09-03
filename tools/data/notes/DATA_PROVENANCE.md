# Real text vs synthetic text, when the labels are synthetic either way

Both halves of a training corpus can be synthetic -- the TEXT and the ANNOTATION over it.
They are separable and they behave differently. **The two axes disagree, so "is synthetic
data good" has no single answer and asking it that way produces the wrong decision.**

## Axis 1 -- preservation: SYNTHETIC TEXT WINS

2x2 fine-tune over `base-v1`, common held-out set (`pile_ner_def` + `knowledgator_gliner`
val, stride-6 slice, best-vs-best), entity strict micro F1:

| arm | text | annotation | strict F1 | delta vs base |
|---|---|---|--:|--:|
| base-v1 | -- | -- | 0.5322 | -- |
| `synthetic_haiku45_5k` | synthetic | synthetic | **0.4099** | -0.1223 (-23.0%) |
| `cc_news_haiku45` | **real** | synthetic | 0.3596 | -0.1726 (-32.4%) |
| real + synthetic mix | both | synthetic | 0.3267 | -0.2055 (**-38.6%**) |

Real text forgot MORE. The prediction was the opposite -- real news is closer to
`pile_ner_def` than generated passages are, so it "should" have preserved better.

**Volume confound, do not quote -32.4% without it:** the real arm ran 10,960 optimizer
steps over 15,839 docs against the synthetic arm's ~5,080 over 4,018. It does not explain
the result away -- step-matched at epoch 5 (~5,480 steps) real text still preserves
0.3653 against 0.4099.

**75% of the forgetting happens in EPOCH 1** (-0.1296 of -0.1726). That is a step change
at first contact with a new label space, not gradual drift, so early stopping cannot
rescue most of it and replay is the only lever. See [[replay-dose-for-forgetting]].

## Axis 2 -- transfer to real deployment text: REAL TEXT WINS, decisively

| system | training positives | in-distribution | on real news |
|---|---|--:|--:|
| stage-0 relevance gate | 99.9% synthetic | F1 **1.0000** | admits **0 of 71** articles, 0 of 590 SMS |
| EKF front end | 71.4% synthetic English trigger+arg | correct on its own corpus | forms events on FEWER real wire-copy windows than the model it replaced |
| casualty extractor | 100% synthetic English | strong on its own corpus | `location` holds a digit 78.2% of the time on Turkish vs 5.8% English (p=1.5e-39) |

A gate at F1 = 1.0000 that admits nothing real has a trivially perfect false-positive rate
because it never fires.

## The corollary that costs the most

**When positives and negatives come from different provenance, an in-distribution metric
measures provenance, not the task.** The gate above was first diagnosed as a truncation
and CJK problem; the real cause was that its positive class was synthetic and its negative
class was real, so it learned to separate registers. Balance provenance across classes, or
report the in-distribution score as uninterpretable.

## How to decide

- Must the model KEEP prior capability? Synthetic text is the safer regulariser -- but use
  replay regardless: 5-10% is the minimum that prevents catastrophic forgetting, ~30% is
  better, and EXACT replay from the base's own pool beats a proxy.
- Must the model WORK on real input? Real text, at whatever preservation cost, and buy
  replay to pay it back. Synthetic text does not buy transfer at any volume.
- Never mix them and expect the average: the mix arm preserved WORST of all three.


## It is domain adaptation tangled with forgetting, not "synthetic is bad"

The two axes above are the same mechanism seen twice: distance between distributions.
Preservation measures distance from what the base ALREADY KNEW; transfer measures distance
from where the model will be DEPLOYED. Synthetic text is broad and close to a general
pretraining distribution, so it forgets less and generalises zero-shot. Real text carries
a domain's own bias -- news, email, tweets -- which is precisely the signal that transfers,
and precisely what makes it look far from a general eval set.

### Domain-prompted synthetic captures the REGISTER, not the DISTRIBUTION

`casualty_natural` is already domain-prompted -- it generates disaster news with datelines
and attribution ("A major news outlet reported that dozens of residents remain missing
following the June 24, 2026 crash of Flight MH370 into Villa St. Louis"). It is not a
missing experiment; it is the corpus the EKF line has trained on all along, and the line
is still real-news-poor.

Measured against `cas_ann_en` (real DocEE news), same task, same schema, same language,
9,000 documents each:

| | synthetic "news" | real news | ratio |
|---|--:|--:|--:|
| distinct tokens (first 400/doc) | 10,538 | **155,251** | **14.7x** |
| median length | 1,025 ch | 2,785 ch | 2.7x |
| casualty records per doc | 2.48 | 1.56 | -- |
| median position of the first toll | 29.4% | **15.9%** | -- |

Prompting fixes the surface: datelines, wire phrasing, "officials confirmed". It does not
fix the lexical tail, the document length, or the inverted pyramid -- real wire copy leads
with the number, synthetic buries it mid-document, and most of a real article is about
something other than the toll. **You can prompt for register. You cannot prompt for the
145,000 tokens the generator never uses.**

### The result is a property of a TRIPLE, not of synthetic data

The preservation table was measured on `base-v1` (fastino DeBERTa) against
`pile_ner_def`. "Real text forgets more" is a statement about distance from what THAT base
knew, on THAT eval set, at THAT replay dose. The 137k/eb16 base already contains real news
(docee, cc_news_haiku45), so the same fine-tune is no longer novel to it and the ordering
may not hold. Do not carry the ordering across a base change without re-measuring; carry
the mechanism.

### The A/B that is now possible for the first time

`cas_ann_en` (real English news) and `casualty_natural` (domain-prompted synthetic
English) share a schema, a task and a language. Same base, same replay, one variable:
text provenance. That isolates domain adaptation from forgetting and from the label space,
and it is the experiment the EKF line has needed since the front-end rebuild scored well
in-domain and formed events on fewer real wire-copy windows than the model it replaced.

## Sources

`tools/train/preservation_results/`, [[lambda-real-vs-synth-run]],
[[lambda-gate2-multilingual-run]], [[extractor-cannot-read-turkish]],
[[lambda-synthetic-sanity-run]], PAPER_0_FOUNDATION.md section 7.2.
