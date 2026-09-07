# Label-Space Collapse: Three Failures Sharing One Name, and Which One We Have

**William Roe**¹ (whr778@gmail.com) and **Claude**² (noreply@anthropic.com)

¹ Project author and maintainer  ·  ² AI assistant (Anthropic, Claude Opus 5) — measurement and drafting

**Supporting note to the research programme in `RESEARCH_PROGRAM.md`.** Revision of 2026-09-07.

*Companion to `DOMAIN_ADAPTATION_PLAN.md` (which sets the replay dose this note explains)
and `PAPER_0_FOUNDATION.md` (which reports the zero-replay arms). The operational rules
that follow from it live in `tools/data/annotation/GUIDELINES.md`, which is loaded into
every annotator prompt rather than restated here.*

---

## Abstract

"Label-space collapse" names at least three distinct failures in the literature, and this
project can suffer all three by different routes. They have different causes, different
diagnostic signatures, and different fixes; conflating them leads to applying the wrong
remedy, which is how this note began. We separate them, state which one we have measured
(the training-time one, twice, on two different bases), and report a new measurement:
the **fallback-label rate in our own LLM-annotated corpora is 0–1.8%**, so the
annotation-time failure described by MultiSoc-4D is *not* present in our data — but two
low-cardinality tasks are near-degenerate in a way the marginal distribution alone cannot
adjudicate. That last point is the argument for inter-annotator agreement, and it is why
a skewed histogram is not by itself evidence of anything.

---

## 1. Three failures, one name

### 1a. Instruction-induced label collapse — *annotation time*

An LLM asked to annotate against a closed label set drifts toward fallback categories:
`Other`, `Neutral`, `No`, `Unknown`. The failure enters the **data**, before any training
happens.

The reference result is **MultiSoc-4D** (arXiv 2605.06940), a 58K-comment Bengali social
media benchmark annotated on four dimensions by ChatGPT, Gemini, Claude and Grok. The
models **failed to detect 79% of hateful and 75% of sarcastic instances**, defaulting to
neutral categories instead.

Its important contribution is not the miss rate but the **agreement illusion**. Cross-
verifying with several LLMs produced high *raw* agreement while **Fleiss' κ ≈ −0.001** —
chance-level. The models agreed because they had all defaulted to the same safe label, not
because they had converged on a reading of the text. Raw agreement is therefore not merely
a weak check here; it is actively misleading, and it fails in the direction that looks like
success.

**Consequence for us:** an annotator-agreement gate that reports raw agreement would have
passed MultiSoc-4D's collapsed annotations. Agreement must be chance-corrected, and must be
read beside the marginal distribution — see §5.

### 1b. Label-space collapse under narrow fine-tuning — *training time*

A trained extractor fine-tuned on a narrow ontology stops proposing labels outside it. The
failure is in the **model**, and the data was fine.

This is the one we have measured, and the signature is specific:

> **False negatives rise while boundary errors fall.**

The model does not get worse at locating span edges. It stops proposing spans at all. It
has narrowed what it believes exists.

This matters architecturally because GLiNER2 takes the label set as an **input** rather
than a fixed head. A large part of its capability lives in having been conditioned on a
broad, varied label distribution. Fine-tuning on a narrow ontology teaches it that the
world contains only those types — which is why breadth of training labels is the defence,
and why any narrow fine-tune erodes it.

**It also explains why EWC is the wrong instrument here**, independent of EWC's general
fragility. Elastic Weight Consolidation penalises movement in parameters a Fisher matrix
says mattered for the old task. That is the right shape when capability lives in the
weights. Here a large part of it lives in the *input conditioning*, and a diagonal Fisher
penalty on weights does not address an input-distribution problem. Replay does, directly:
it keeps showing the model wide label menus. This is also why a round, unswept 30% works
as well as it does — it is not a delicate regularisation constant, it is just keeping the
input distribution broad.

### 1c. Representation / embedding collapse — *geometric*

Distinct semantic concepts map to the same coordinates in embedding space (complete
collapse), or the network flattens data into a narrow subspace (dimensional collapse). Its
remedies are architectural — anti-collapse losses built on coding-rate maximisation
(arXiv 2407.03106), label smoothing, and related regularisers.

**This is not what we have measured**, and it is worth stating explicitly because the name
collides. Our failure is in *which labels get proposed*, not in *where concepts sit in
embedding space*. We have never measured our embedding geometry. Applying an anti-collapse
loss to our problem would be treating a symptom we have not observed.

**Do not confuse any of the above with Neural Collapse**, which is *desirable*: in the
terminal phase of training, intra-class samples converge to their class mean while class
means maximally separate. Healthy geometry, same word.

### 1d. The sink label is the bridge between 1a and 1c

The three failures above are distinct, but a **sink label** — `Misc`, `Other`, `Unknown` —
connects the first to the third, and the connection explains why deleting such labels helps
more than their frequency suggests.

Plotted in embedding space, a sink class does not form a cluster. It forms an **octopus**:
no coherent centre, with tentacles reaching into every real cluster on the map. That shape
is the direct consequence of how the class is populated — each `Misc` instance is really a
member of some *other* class that the annotator declined to name, so the members are
scattered wherever those real classes live.

*We call this shape a **Tulula**, after the octopus at the Baltimore aquarium — the name a
former intern of the project author gave the plot that first showed it. It is used here as
a working term because a named failure gets discussed and an unnamed one gets rediscovered;
it is not standard terminology.*

Three consequences follow, and only the first is obvious:

1. The sink class is unlearnable, because it has no defining feature. Expected.
2. **Its class mean is a point in the middle of nowhere** that is near everything and
   belongs to nothing. Under the Neural-Collapse dynamic, training pulls genuinely
   unrelated regions of the space toward that meaningless centroid. A sink label is
   therefore a *supervised force toward* the geometric collapse of §1c — not merely a
   category that fails to learn, but one that actively deforms the space around it.
3. **The damage is not confined to the sink.** Every real class acquires sink-labelled
   neighbours inside or adjacent to it, so its decision boundary is contested by a class
   with no definition. The real classes get worse.

**For GLiNER2 this is worse than for a fixed-head classifier**, because the label is an
*input*. A sink label's embedding must serve as a query that matches arbitrary spans, so it
is trained toward a "matches everything a little" direction — a high-norm, low-information
query vector. The poison is on the query side, not just the target side, and the query side
is the part shared with every other label.

**Practical consequence, and it is the strongest recommendation in this note:** do not put a
sink label in a training label set. Note that removing it is *not* the same as forbidding
abstention — see §5, item 0.

*Status of this argument: mechanism, not measurement. We have never inspected our embedding
geometry (§7.4). What we can state as fact is §3b: our entity label space contains no sink
types at all.*

---

## 2. How 1a and 1b compose, which is the real risk

The two failures chain. A collapsed annotator produces a corpus whose label distribution is
narrow; training on that corpus then teaches the model the same narrowness. Neither step
raises an error. The corpus looks fine — every record is well-formed, every span is
verbatim — and the model trains to convergence.

This is not hypothetical for us: `cc_news_haiku45`, `synthetic_haiku45_5k`,
`synthetic_sonnet5_1k` and `zh_multitask` are all machine-annotated, by one or two model
families. Buying more from the same annotator inherits whatever narrowing that annotator
has. **Annotator diversity is therefore a design variable, not a cost line.**

---

## 3. What we have measured

### 3a. Training-time collapse, measured twice

**Three zero-replay arms, 2026-08-18** (`fastino/gliner2-base-v1`, real vs synthetic 2×2):
lost **23%, 32% and 39%** of general-domain entity F1, monotonic in training volume
(~5,080 / 10,960 / 12,690 steps). FairEval attributed it to label-space collapse rather
than boundary degradation: false negatives rose **27,267 → 36,244** while boundary errors
**fell**.

**Three gate3 warm cells, 2026-09-06** (`eb16-rebuild-tr`, fine-tuned on an 18K single-task
corpus, zero replay, scored on the base's own 18,786-record blind test):

| head | s1 | s4 | s21 | mean | relative |
|---|--:|--:|--:|--:|--:|
| classification | −0.413 | −0.378 | −0.418 | **−0.403** | **−69%** |
| event_trigger | −0.225 | −0.205 | −0.188 | −0.206 | −34% |
| entity | −0.198 | −0.169 | −0.168 | −0.178 | **−31%** |
| event | −0.088 | −0.072 | −0.080 | −0.080 | −20% |
| relation | −0.052 | −0.040 | −0.025 | −0.039 | *at the ±0.041 floor* |
| structure | −0.027 | −0.021 | −0.023 | −0.024 | −20% |

Every head negative, three seeds agreeing tightly, for **+0.024 on the target task**. The
−31% entity loss lands inside the 23–39% band measured three weeks earlier on a different
base, a different corpus and a different task.

**The dose rule that follows** (`replay-dose-for-forgetting`, and implemented in
`build_warmstart_mix.py`): **5–10% replay is the minimum that prevents catastrophic
forgetting; ~30% is the operating point; exact replay beats a proxy.** Note honestly that
30% has never been swept — it is experience plus one confirmation, not a measured optimum,
and the only dose-variant corpora in the repo (`tr_dose0/5000/15000/31263`) vary *Turkish
volume*, not replay fraction.

### 3b. Annotation-time collapse: measured, and largely absent

New measurement, 2026-09-07, over `true_label` only. **Counting the label menu instead of
the answer is the trap here** — the menu is offered, not chosen, and including it produced
a spuriously uniform ~15% across every gate label on the first attempt.

| corpus | task | labels | top label | fallback share |
|---|---|--:|---|--:|
| zh_multitask | zh_topic | 16 | Economy 32.9% | **1.8%** |
| cc_news_haiku45 | topic | 16 | politics 15.8% | **0.0%** |
| cc_news_haiku45 | audience | 4 | *general public 88.0%* | 0.0% |
| synthetic_haiku45_5k | topic | 16 | business 23.0% | 0.0% |
| synthetic_sonnet5_1k | topic | 12 | business 35.0% | 0.0% |
| synthetic_sonnet5_1k | sentiment | 3 | positive 64.0% | 22.0% (`neutral`) |
| synthetic_sonnet5_1k | formality | 2 | *formal 87.2%* | 0.0% |
| gate3 | relevance | 2 | mass_casualty 50.1% | 49.9% (balanced *by construction*) |

**The MultiSoc-4D failure is not present in our data.** Fallback rates of 0–1.8% on the
16-label topic tasks are nothing like a model retreating to `Other`.

**But two tasks are near-degenerate** — `cc_news_haiku45.audience` at 88% "general public"
and `synthetic_sonnet5_1k.formality` at 87.2% "formal" — and **the marginal alone cannot
tell us whether that is collapse or correctness.** Most news genuinely is written for a
general audience; the synthetic documents genuinely are mostly formal. A skewed histogram
is consistent with both a lazy annotator and an accurate one.

That is the whole argument for §5. You cannot resolve it by staring at the distribution.
You need a second, independent annotator and a chance-corrected statistic.

### 3c. A third, related narrowing: task-composition collapse

Measured 2026-09-06 on the `eb16-rebuild-tr` mix: it presents **8 of 31 possible task
combinations**, and **11 of 13 corpora emit exactly one combination at ~100%**. A
combination is therefore a *corpus signature*, and the model can route to the right heads
from a document's style without reading the requested schema. 54.7% of rows carry two
tasks; **0.0% carry three or more**.

This is the same disease one level up — narrowing of the *menu of task combinations* rather
than of labels within a task — and it is what the Phase 0 experiment
(`config/base/eb16-composed.yaml`) tests.

---

## 4. Detection

Each failure has a signature. Test for the one you have.

| failure | signature | what to compute |
|---|---|---|
| annotation-time (1a) | high raw agreement, chance-level κ; mass on fallback labels | Fleiss'/Cohen's κ or Krippendorff's α **plus** the marginal histogram |
| training-time (1b) | **false negatives up, boundary errors down**; losses scale with step count | FairEval error decomposition against the base's own test set |
| geometric (1c) | embedding rank / eigenvalue spectrum flattening | singular values of the representation matrix |

Two traps, both hit in this project:

- **Score preservation on out-of-replay data.** Replaying corpus X's train split and
  scoring on X's val split tells you the replay worked, not that the capability survived.
- **A target-task gain is not evidence a warm start worked.** The gate3 cells gained +0.024
  where they trained and lost on all eight heads elsewhere. Always score the base's own
  test set.

---

## 5. Mitigation, in the order it should be applied

**At annotation time**

0. **Delete sink labels from the trained label set — the first and largest lever.**
   `Misc`, `Other`, `Unknown` as *entity types* should not exist (§1d). This is long-
   standing practice for the project author, applied since ~2019, and our entity space is
   clean: **zero sink entity types across all 120 corpora** (§3b).

   **MultiSoc-4D does not test this.** The paper is diagnostic, not prescriptive: it runs no
   ablation on the label set, and its *future work* proposes the opposite direction —
   adding explicit `NaN`/`Uncertain` labels for open-set labelling. Both positions can be
   right, because the failure is not the existence of an abstain option but **training a
   class on it**. Deleting the sink and recording an abstain that is then *excluded from
   training* both avoid teaching a garbage category. What must not happen is a trained
   `Other` class that absorbs the annotator's uncertainty.

   **The distinction that decides each case: is the fallback a real semantic class, or a
   sink for uncertainty?** `no_toll` in `annotate_gate` is real — "this article states no
   casualty count" — is balanced by construction at ~50%, and is the class the gate exists
   to identify. `Other` in a 16-way topic task is a sink. Delete sinks; keep real negatives
   and never discourage them.

1. **Instruct against the fallback explicitly — but only where the fallback is a sink.**
   None of our five annotators previously said anything about minority labels, and three
   offer an `other`/`none` class. Now applied via the `minority` rule, and **deliberately
   withheld from `annotate_gate`**: its fallback is a real class and its measured failure
   mode is false positives, so discouraging the negative would make the gate worse. A
   blanket rule would have been a regression there.
2. **Few-shot the minority classes**, not the majority ones.
3. **Diversify annotators across model families**, and treat that as design, not cost.
4. **Gate on chance-corrected agreement plus the marginal** — never raw agreement, which
   passes exactly the case you are trying to catch. This mirrors the project's standing
   lesson that a form gate must be paired with a correctness companion
   (`gate1-counts-firings-not-hits`).

**At training time**

5. **30% exact replay** in any fine-tune of a trained checkpoint; 5–10% is a floor for when
   budget forbids 30%, not a target.
6. **Keep the label menu broad.** Breadth of training labels is the defence for an
   input-conditioned architecture.
7. **Do not reach for EWC** — see §1b.

---

## 6. Two analogies, offered as intuition only

**Groupthink.** The structural parallel is real and narrow: both narrow *what gets
considered*, not *what can be done*. Members of a group under groupthink do not become less
capable; the range of alternatives raised collapses — which is exactly false-negatives-up,
boundary-errors-down. More striking, the classic remedies (devil's advocate, outside
experts, independent subgroups, leader withholding preference) are all *inject diversity
into the input distribution*, and none of them try to make individuals hold prior beliefs
more firmly. That is the same conclusion as replay-over-EWC, reached in a different field.

The analogy breaks on mechanism: ours is gradient descent on a shifted distribution, with
no conformity pressure, status, or self-censorship; groupthink is multi-agent and involves
motivated reasoning and pluralistic ignorance. Janis's model is also empirically contested.
Use it as an intuition pump, not as evidence.

**Model collapse on recursive data** is the tighter analogue, and the more uncomfortable
one: a population trained on model-generated output loses the distribution's tails first,
converges, and no individual step is detectably wrong. That is multi-agent, self-
reinforcing and tail-destroying — and §2 is precisely that risk in our pipeline.

---

## 7. Open questions

1. **Is 30% replay actually optimal?** Unswept. A 10/20/30/40/50% sweep on the 137k base is
   five short warm starts, ~$25. Expected shape: a broad plateau from ~20–40% with no sharp
   peak. Low priority — the operating point works.
2. **Are `audience` at 88% and `formality` at 87.2% collapse or correctness?** Unresolvable
   from the marginal. Needs a second annotator on an overlap sample (§5.4).
3. **Does composed supervision reverse §3c?** Phase 0 (`eb16-composed`) is the test;
   pre-registered bar is structure F1 +0.02 over the measured baseline.
4. **Have we ever measured our embedding geometry?** No. §1c may or may not apply to us and
   we have no evidence either way.

---

## References

Verified 2026-09-07 by fetching the arXiv abstract pages.

- **arXiv 2605.06940** — *MultiSoc-4D: A Benchmark for Diagnosing Instruction-Induced Label
  Collapse in Closed-Set LLM Annotation of Bengali Social Media.* Pramanik, Antu, Abyad,
  Khalil, Hussain. Source of the 79%/75% miss rates and the Fleiss' κ ≈ −0.001 agreement
  illusion.
- **arXiv 2407.03106** — *Anti-Collapse Loss for Deep Metric Learning Based on Coding Rate
  Metric.* Source for the geometric-collapse remedy in §1c.

Internal:

- `PAPER_0_FOUNDATION.md` — the 2026-08-18 zero-replay arms and the FairEval decomposition.
- `DOMAIN_ADAPTATION_PLAN.md` — replay dosing under multilingual adaptation.
- `JOINT_IE_SCALING.md` — the 30%-replay warm-start results.
- `tools/data/annotation/GUIDELINES.md` — the operational annotator contract.
- Project memory: `replay-dose-for-forgetting`, `gate3-warm-cells-catastrophic-forgetting`,
  `gate1-counts-firings-not-hits`.
