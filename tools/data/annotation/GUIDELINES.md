# Annotator guidelines

**This file is the source, not a description of the source.** `annotation/__init__.py`
reads the blocks below and injects them verbatim into every annotator's system prompt. What
you review here is byte-for-byte what the model is sent — there is no second copy in Python
to drift away from it. Edit this file to change annotator behaviour.

Rationale, evidence and the review policy are in
`tools/events_working_papers/LABEL_SPACE_COLLAPSE.md`. Kept here: the rules themselves and
the minimum needed to review them.

Each injectable block sits between `<!-- rule: name -->` and `<!-- end -->`. The
surrounding prose is for humans and is never sent.

---

## Reply format

Every annotator returns parsed JSON, so this is non-negotiable and identical everywhere.

<!-- rule: json_only -->
Reply with a single JSON object and nothing else. Do not wrap it in prose, explanation, or
code fences.
<!-- end -->

---

## Verbatim spans

For any task that produces a span the model must later locate in the text. The boundary head
finds spans by matching them against the source, so a paraphrased, translated, normalised or
re-cased surface produces a training row that **teaches nothing and reports no error**. The
converters drop such values and print the rate; a silent drop would let one bad prompt spend
a whole batch.

<!-- rule: verbatim -->
Every span you output must be copied from the source text EXACTLY, character for character.
Do not translate, normalise, reformat, correct, or add words. If you cannot find an exact
span for something you believe is present, omit that item rather than approximating it.
<!-- end -->

---

## No inference beyond the text

For classification and typing tasks, where a model asked to be helpful will happily supply
what the article implies rather than what it says.

<!-- rule: no_inference -->
Answer only about the text you are given. Never infer beyond it, and never use knowledge of
the wider world to fill a gap the text leaves open.
<!-- end -->

---

## Minority labels

**This is the newest rule and the one with the clearest evidence behind it.** An LLM
annotating against a closed label set drifts toward fallback categories — `Other`,
`Neutral`, `No`, `Unknown`. MultiSoc-4D (arXiv 2605.06940) measured frontier models missing
**79% of hateful and 75% of sarcastic** content this way, and — the part that matters for
any agreement check — several models *agreed with each other* at high raw rates while
Fleiss' κ ≈ −0.001, because they had all defaulted to the same safe label.

Our own corpora do **not** currently show this (fallback share 0–1.8% on the 16-label topic
tasks; see the working paper). This rule is preventive, and cheap.

<!-- rule: minority -->
Do not retreat to a catch-all label. Categories such as "other", "none", "neutral",
"unknown" or "no" are for cases that genuinely fit nothing else, not for cases that are
merely hard, ambiguous, or subtle. If a specific category applies even partially, prefer it
over the catch-all. Rare and understated instances are the ones this dataset most needs you
to find.
<!-- end -->

---

## Genuine ambiguity

The anti-fallback rule above says where *not* to put a hard case. It does not say where to
put it, and "don't use the catch-all" is useless advice on its own when a drone is
plausibly an aircraft and plausibly a weapon.

**The first question is whether the case is ambiguous at all, or simply plural.** Most of
what feels ambiguous is genuinely both, and our record format carries a surface under more
than one type at no cost — `entities` maps type to surfaces, so the same span may appear
twice. **Annotators already do this unprompted: 5.1% of `cc_news_haiku45` records and 7.7%
of `zh_multitask` records contain a surface filed under two types.** For a drone that is
being flown as a weapon, both labels are correct and emitting both is not a compromise.

**Only when the categories are genuinely exclusive does the hard choice arise**, and there
the ranking below is reasoning from this pipeline's mechanics, **not measurement** — we have
never run the experiment that would settle it, and the note in the working paper says so.

<!-- rule: ambiguity -->
When a span or document is genuinely ambiguous, prefer these in order. First, if more than
one label is actually correct, emit all of them rather than choosing — overlapping types are
expected and supported. Second, if the labels are mutually exclusive but you cannot tell
which applies, choose the more specific one you believe most likely and label it; a
considered guess is more useful than an omission. Third, omit the item only when you cannot
support any label from the text. Never resolve an ambiguity by reaching for a catch-all
category — that is the one option that destroys the distinction for every future reader.
<!-- end -->

**Why omission ranks below a considered guess here, specifically:**

1. **Omission is not neutral in this pipeline — it can become an assertion.**
   `mint_entity_negatives` seeds 12 absent types per record with `[]`, on the reasoning that
   a type the model saw and declined to use is strong evidence of absence. With a 125-type
   ontology that is **12 of ~113 absent types, so roughly a 1-in-9 chance that any given
   omission is converted into an explicit "this type is not present"** — a confident
   falsehood rather than a silence. (Applies to `cc_news_haiku45` and
   `synthetic_haiku45_5k`, which carry 12.0 seeded negatives per record;
   `synthetic_sonnet5_1k` and `zh_multitask` carry none.)
2. **Our models' measured failure mode is under-proposing, not mis-proposing.** Label-space
   collapse shows up as false negatives rising *while boundary errors fall*. Omission feeds
   the error direction the architecture is already prone to.
3. **A coin-flip's damage is symmetric; a fallback's is systematic.** Guessing wrong on a
   two-way ambiguity is 50% label noise scattered across two real classes, which training
   tolerates. A catch-all instead collects every hard case into one attractor with no
   semantic centre — the shape the working paper calls a Tulula.

**The honest caveat:** the right answer is none of the three. It is to record the
uncertainty and *exclude that item from negative seeding* — an abstain that is routed rather
than trained. That is the direction MultiSoc-4D proposes as future work, and it is not built
here yet: the annotators currently have no way to express uncertainty, so
`mint_entity_negatives` cannot know to skip a type the annotator was unsure about. Until it
exists, the ranking above is the best available, and it is reasoning rather than evidence.

---

## What is deliberately NOT here

- **Task ontologies and label lists.** They belong with the task, in its own annotator, and
  differ per corpus.
- **Output schemas.** Same reason.
- **Language rules** (e.g. "labels in English, spans in the source language"). Currently
  specific to `annotate_multitask`; promote it here only if a second annotator needs it.

A rule earns its place in this file by being needed by **two or more** annotators, or by
guarding a failure that has actually been observed. Otherwise it lives in its own annotator
and stays legible there.

---

## Review policy for machine-annotated batches

Not yet implemented — recorded here so the gate is defined before it is built, and so the
next increment has a specification rather than an intention.

A batch is **not** fit to train on merely because it parsed. Before a machine-annotated
corpus is used:

1. **Overlap sample.** Re-annotate ≥500 documents (or 10%, whichever is smaller) with a
   second annotator from a **different model family**. Full double annotation is not needed
   and not affordable.
2. **Chance-corrected agreement.** Report Cohen's κ (two annotators) or Krippendorff's α.
   **Never raw agreement** — it passes exactly the failure being tested for, and fails in
   the direction that looks like success.
3. **Marginal distribution, reported beside κ.** Per task: label count, top-label share,
   fallback share. This pairs a form check with a correctness companion, the standing lesson
   from `gate1-counts-firings-not-hits`.
4. **Adjudicate disagreements**, do not discard them: send them to a third annotator or a
   human. Systematic disagreement is a prompt defect, and discarding it hides the defect.
5. **Both numbers are needed, and neither is sufficient.** A skewed marginal is consistent
   with a lazy annotator *and* with an accurate one on genuinely skewed data — our
   `cc_news_haiku45.audience` sits at 88% "general public", which may be simply true. High κ
   on a collapsed marginal is the MultiSoc-4D illusion. Read them together or not at all.
