# Adding a language: the recipe, and the traps that cost us a weekend

Written 2026-09-05, from the Turkish build. Every step here exists because skipping it
cost something measurable. The order matters: each step is a gate on the next.

Related: [[GATES]], [[MODEL_LINEAGE]], `tools/data/notes/CHINESE_TOLL_DISTRIBUTION.md`,
`tools/train/preregistration/COLLATOR_ARM_20260904.md`.

---

## 0. Before anything: know what your instrument measures

Two numbers disagreed all weekend, and the disagreement is the single most useful thing
learned:

| instrument | what it rewards |
|---|---|
| classification F1 on a balanced test set | matching the corpus you built |
| **admission / extraction on real held-out documents** | **doing the job** |

`gate3-mixed` won BOTH F1 numbers and lost Chinese admission 59/60 -> 50/60. The
length-proxy fix improves Turkish extraction F1 (+0.050) and degrades Chinese (-0.028).
**Never ship on F1 alone.** Bars on real documents decide.

## 1. Source the pool, and check what the pool selects FOR

`turkish_pool18` is 100% casualty-cue-bearing by construction, because it was built to
feed casualty annotation. Buying event-type labels from it would have produced a Turkish
arm with no `Sports Competition` and no `Organization Fine` at all -- the model learns the
disaster types and never learns to say "not a disaster".

Measure the pool's own selection before buying from it. `--cue exclude` on
`build_turkish_pool.py` collects the complement.

## 2. Pilot at ~500 documents. Always. It is ~$0.50

Both Turkish pilots paid for themselves several times over:

- pilot 1 proved the cued/uncued split (29.0% vs 4.5% casualty types) AND exposed that the
  cue matches DEATH, not DISASTER -- six `Earthquakes` in 238 documents. Without it we
  would have spent $27 on a corpus led by `Armed Conflict` and `Famous Person - Death`.
- pilot 2 validated the disaster-noun arm at 7.7x, and its projections held within ~30% of
  the full buy on most types.

Fix the pass criteria BEFORE seeing results. Ours: unparseable <5%, non-verbatim drops
<20%, documents with no extractable role <40%, and the target roles present in volume.

## 3. Condition the second pass on the first

The entity pass showed each document only ITS event type's roles (DocEE carries per-type
role sets, 59 types at a median of 9) rather than all 356. That recovered 340 of 356 roles
and picked up `Magnitude`, `Epicenter`, `Number of Evacuated People` -- profile-bearing
fields a 4-role menu misses.

Spans must be VERBATIM or dropped, never repaired: the boundary head locates fields as
spans, so a paraphrase trains nothing while reporting no error.

## 4. Unify the label MENUS, not just the labels

Menus were 59 (en) / 58 (zh) / 60 (tr). A model whose Chinese menu lacks `Armed Conflict`
is never asked to consider it; one whose English menu lacks `none` cannot answer it. Rewrite
all corpora to the UNION; never touch `true_label`.

Separately, run the label MAP: 5 of Turkish's 338 roles are `labels/unified.yaml` keys
(`Start_Date` and `Start Date` both -> `StartDate`). Without `labels_file` those train as
different roles.

## 5. Interleave before writing

A builder that accumulates per-language and writes in order produces blocked splits.
Training never sees it (the sampler randperms first) but every PREFIX measurement is wrong:
`casualty_ml`'s first 20,000 rows read 89.5% English for a 32.9% English corpus. Use
`interleave_splits.py`, or `rng.shuffle(rows)` before the write loop.

## 6. Check the length proxy against the SCRIPT

`len(text.split())` returns ~1 for Chinese and understates agglutinative Turkish by 1.80x.
Under it, `casualty_ml` trained with **75.7%** of batches 100% one language against a 33.5%
random floor; gate3 with 39.2% against 3.7%. Character classes cut cross-language spread to
1.37x. Any new script needs this checked, not assumed.

## 7. Shares are an INPUT, not an outcome

`balance()` equalises classes within a source and has no term for weight ACROSS sources, so
adding a language taxes the others. It cost Chinese in `gate2_tr` (0.4% of TOKENS, not the
3.1% row share suggested) and English in `gate3`. Use `--source-share`.

And measure share in TOKENS, not rows or characters: Chinese is 15.6% of gate3 rows, 7.6%
of characters, and 20.1% of tokens. Characters mislead by 2.6x.

## 8. Establish the variance floor BEFORE the A/B

Three seeds of one config: test F1 spread **0.0095**, Chinese admission spread **5 of 60
documents**. Two conclusions died on contact with that floor -- a "+0.0039 win" for mixed
batching, and a Chinese admission verdict I stated with far too much confidence.

Use five seeds, and prefer TrGLUE's (1, 4, 21, 40, 124) over 42/1337/7: 42 is the default
of half the ecosystem and the least independent draw available.

## 9. Replicate across feeds before believing an end-to-end number

Helene said gate3 regressed English 132.59 -> 175.66. It did not reproduce on Aegean (tie)
or Turkiye-EN (identical). One feed, two documents, no seed replication.

---

## The standing lesson

Four recorded conclusions did not survive re-examination this weekend: LEVEN's "108-label"
OOM (MAVEN has 168 and trained fine), Helene's English regression, the mixed-batching win,
and my own gradient-share explanation for the Chinese drop. All four were single
observations written down confidently. **A number you have not replicated is a hypothesis.**
