# Domain-adapting mmBERT: plan, constraints, and what would make it worth doing

**Status: NOT STARTED. Deliberately held.** The gate is the `casualty_ml` run finishing and
the EKF gates being re-tested against it. Nothing here should be executed before that,
because the adapted encoder invalidates every downstream comparison until the current
baseline is measured. This document exists so the design is settled before the budget is.

Related: [[PAPER_0_FOUNDATION]] §7.3 (the measurement that motivates this), [[GATES]],
[[HEAD_INIT_DATA_SCALE]], [[MODEL_LINEAGE]].

---

## 1. What the measurement actually licenses

Continued pretraining is expensive and it is not free of risk, so the first question is
which cells of the perplexity table justify it. mmBERT-base, 120 held-out documents per
cell, from the corpora we train on (`tools/train/base_model_perplexity.py`):

| language | pseudo-PPL | | English domain | pseudo-PPL |
|---|--:|---|---|--:|
| Kazakh | 2.26 | | financial filings | 3.13 |
| Turkish | 2.46 | | biomedical | 4.15 |
| English | 4.93 | | **news (real)** | **4.67** |
| Chinese | 5.90 | | news (synthetic) | 5.42 |
| Japanese | 10.36 | | scientific | 8.56 |
| African langs (MasakhaNER) | **114.11** | | **short-form messages** | **29.52** |

Read as a decision table, this says three separate things.

**Adaptation is NOT the lever for the main line.** mmBERT models real news at 4.67, as well
as anything we measured. The EKF front end's weakness on real news is therefore supervised
signal, not representation, and continued pretraining would buy nothing there. This is the
single most useful line in the table, because it is the one that says *don't spend*.

**Adaptation IS the lever for register.** Short-form, unedited text sits at 29.52 — a 6.3x
gap against news, at fixed language. Our stage-0 relevance gate admitted 0 of 590 disaster
SMS messages, and we published a provenance confound (synthetic positives against real
negatives) as the cause. That was real but incomplete: part of that failure is
representational, and no quantity of supervised SMS labels closes it.

**Adaptation IS the lever for the low-resource languages**, at 114.11. But nothing in the
current pipeline consumes them, so this is a capability question, not a delivery one.

The honest summary: the case for domain adaptation rests on **register**, not on domain in
the topical sense, and not on the languages we currently ship (Turkish 2.46, Chinese 5.90,
English 4.93 are all firmly modelled).

---

## 2. The governing constraint: replay, per language

Continued pretraining on new-domain text will degrade the domains and languages the encoder
already modelled well unless original-domain text is mixed back in. That much is standard.
The part that is specific to a multilingual encoder, and the part that is easy to get
wrong, is that **the replay ratio has to hold per language, not globally.**

The failure mode is arithmetic, not subtle. Suppose the adaptation corpus is 60% English
short-form, 25% Turkish, 15% Chinese, and replay is set at a global 30%. If the replay pool
is drawn without a language constraint it will follow whatever is abundant — in this repo,
English. Chinese then receives new-domain gradient with almost no Chinese replay, and its
representation drifts while the global ratio looks correct on the dashboard. Every measured
quantity would look fine except the one that matters.

There is a second, sharper reason. A multilingual encoder holds one shared representation
space, and cross-lingual transfer is a property of how the languages sit in it relative to
one another. Adaptation weighted toward one language does not merely degrade the others; it
can pull the shared space toward the dominant language and damage the alignment that makes
transfer work at all. Per-language perplexity is a *necessary* instrument here but not a
*sufficient* one — a language can hold its own perplexity while its position relative to the
others moves.

**Rule for this work: the mix is specified per language, and forgetting is measured per
language, on held-out sets fixed before training starts.**

### 2.1 We cannot do exact replay, and that changes the design

This project's standing finding on replay is that 5–10% is the minimum that prevents
catastrophic forgetting, ~30% is the best of the doses measured, and **exact replay beats a
proxy** — sampling the base's literal training pool rather than something that resembles it
(see [[replay-dose-for-forgetting]] in the project memory, and `build_137k_replay.py`, which
implements exact replay for the supervised stage).

That finding does not transfer here, because mmBERT's pretraining corpus is not available to
us. Whatever we mix back in is a **proxy** by construction. Three consequences follow, and
they should be stated up front rather than discovered:

1. The measured 30% dose is not directly portable. It was measured for supervised fine-tuning
   against an exactly-reconstructable pool. Treat it as a starting point, not a result.
2. Because the replay is a proxy, the *measurement* has to carry more weight than the ratio.
   The bar is the per-language held-out perplexity, not the mix percentage.
3. A proxy replay pool that is itself narrow is worse than a smaller, broader one. General
   web/encyclopedic text per language is the target, not more of our own corpora — our
   corpora are exactly the distribution we are trying not to overfit to.

---

## 3. Design

**Objective.** Masked-language-model continued pretraining on the encoder only. The task
heads are not involved and are re-initialised downstream; this stage produces a new encoder
checkpoint, nothing more.

**Languages in scope.** English, Turkish, Simplified Chinese — the three the pipeline
ships. Low-resource languages are explicitly out of scope for the first pass: they need a
different corpus, and mixing that question in makes the result unattributable.

**Corpus, per language, two halves.**

| half | content | purpose |
|---|---|---|
| target | short-form / unedited register, plus in-domain news | the thing we are buying |
| replay | broad general-domain text in the SAME language | the thing we are protecting |

Starting ratio **70 target / 30 replay within each language**, and the three languages
balanced against one another the way `casualty_ml` is balanced — downsampled to the smallest,
so no language wins on volume. `build_casualty_multilingual.py` already implements that
pattern and its `check_anchors`-style gate discipline should be copied.

**What we have and what is missing.** News pools exist for all three languages. Broad
general-domain replay text is obtainable per language. **Short-form text in Turkish and
Chinese is the real gap** — the 400-message English probe (`data/short_form_probe.jsonl`,
mirrored at `whr778/short_form_probe`) is a measurement instrument, not a training corpus.
Sourcing that is the first concrete task, and its cost is the first thing to price.

**Schedule.** One pass, short. Continued pretraining that runs long is how a well-modelled
language gets lost; the perplexity instrument should be run at checkpoints, not only at the
end, so the run can be stopped at the point the target improves and the protected languages
have not yet moved.

---

## 4. Instrumentation and pre-registered bars

The instrument already exists and is the same one that produced the table in §1, which is
the point — the before and after are directly comparable.

```
uv run python tools/train/base_model_perplexity.py --encoder <ckpt> --domains \
    --out tools/train/preservation_results/<name>.json
```

Held-out sets are fixed before the run and not touched afterwards.

**Bars, to be committed before training, not chosen after:**

- **Target.** Short-form register improves materially from 29.52. A run that does not move
  this number has bought nothing and should be discarded regardless of what else it did.
- **Protection, per language.** No shipped language (en, tr, zh) regresses beyond the
  measurement's own noise. This bar is per language and any single failure fails the run —
  an average across languages would hide exactly the skew §2 describes.
- **Protection, per domain.** Real news does not regress. It is 4.67 today and it is what
  the EKF front end consumes; this stage must not pay for register with news.
- **Cross-lingual alignment.** Per-language perplexity alone cannot see the shared-space
  drift described in §2, so a downstream check is required, not optional: the rebuilt 137k
  base's per-language extraction scores, compared against the current base. That is the only
  measurement in this plan that directly tests transfer.

---

## 5. Downstream, and the comparison that has to stay valid

The stated intent is to rebuild the 137k base and the gates on the adapted encoder. That
comparison is only interpretable if the encoder is the **single** variable:

- the same `labels_file` (a warm start against a different vocabulary is one the base never
  learned — see the label rule in `CLAUDE.md`),
- the same corpora and the same splits,
- the same recipe, including the per-GPU batch trap: `batch_size` is per GPU, so the same
  config file is two different recipes under different `--nproc_per_node`,
- the same evaluation, including the record-threshold sweep, since the shipped structure
  reference is a swept number.

Anything less and a rebuild that scores differently tells us nothing about the encoder.

**Order of operations:**

1. `casualty_ml` run completes; EKF gates re-tested against it. *(the gate on all of this)*
2. Price and source short-form text in tr/zh. Decide go/no-go on that cost alone.
3. Build the per-language adaptation mix; verify balance and split hygiene the same way
   `casualty_ml` was verified.
4. Continued pretraining, with perplexity checkpoints.
5. Apply the §4 bars. A failed protection bar ends it here.
6. Rebuild the 137k base on the adapted encoder, single variable.
7. Rebuild the gates.

---

## 6. What would falsify this, and the standing caution

The plan is falsified cheaply at step 5: if the protected languages regress, the adapted
encoder is worse for everything we currently ship, and the correct response is to stop
rather than to look for a downstream metric that survived.

It is also worth recording what this plan is *not* a response to. Our register problem was
first attributed entirely to a data-provenance confound; the perplexity table showed that
diagnosis was incomplete, not wrong. Continued pretraining addresses the representational
half. **It does not fix the synthetic-positive / real-negative confound in the gate's
training set, and running it must not be allowed to look like a fix for that.** Both need
doing, and they are independent.

Finally, the standing constraint holds throughout: multilingual support in this pipeline
comes from the models, not from a translation step. Nothing in this plan introduces one.
