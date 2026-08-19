# Constrained Joint Decoding over Boundary-Head Candidates: Data Scaling, Replay, and Three Silent Defects

**William Roe**¹ (whr778@gmail.com) and **Claude**² (noreply@anthropic.com)

¹ Project author and maintainer  ·  ² AI assistant (Anthropic, Claude Opus 5) — design, implementation, and drafting

*Revision of 2026-08-19. Companion to `PAPER_0_FOUNDATION.md` (the substrate) and
`EKF_MHT_DESIGN.md` (the temporal instantiation of the same thesis). The build record —
decision log, wiring map, increment status, cost model, deferred Phase B plan — is kept
verbatim in `JOINT_IE_DESIGN_RECORD.md`.*

---

## Abstract

A boundary-head extractor scores span, mention and pair candidates per chunk, then a
**greedy per-query decode** independently thresholds each one into a final record set. We
replace that decode with **constrained joint selection** over the same candidate scores
and measure what it buys, on a multilingual multi-task base trained from a raw mmBERT
encoder at four data scales (10K–137K records). Three results. First, on document-level
relation extraction (Re-DocRED, 500-document blind test) joint selection beats greedy by
**+0.052 strict F1 (+18% relative)** when both arms are read at their own swept
threshold — and **beam width 1 is optimal**, monotonically beating every wider beam, so
the gain comes from the *formulation* rather than from search. Second, warm-starting the
137K base on new-domain data with **30% exact replay** reverses catastrophic forgetting
into a net gain, holding the original capability while nearly tripling on the new
distribution. Third, and the reason we report the negative space as carefully as the
positive: three separate silent defects — a decode path that dropped one schema key, a
per-row split randomiser, and an uncalibrated decode threshold — each produced results
that looked like clean scientific findings and were not. We report the corrected numbers,
the wrong conclusions we drew before correcting them, and the standing rules adopted so
they do not recur.

---

## 1. Thesis

The boundary head emits candidate scores per chunk — span boundaries, mention logits,
pair logits, and contextual candidate states. A greedy per-query decode then selects a
record set by thresholding each query independently. That is adequate for sparse,
single-document extraction. The claim tested here is that it is not adequate for dense
structure:

> **Global structured inference over the boundary candidate scores — not the greedy
> per-chunk decode — is what dense document-level extraction requires.**

Events and relations are two faces of one mechanism, which is why a single decode covers
both: in the joint problem an event is a *trigger node plus role edges* and a relation is
a *plain edge*. A win on only one face is a weaker but still reportable result, and the
honest negative — that global decoding helps relations and not events, or the reverse —
localises where greedy per-query decoding actually costs you.

The same top-K hypothesis machinery, applied *across* documents rather than within one,
is the EKF/MHT tracker of the companion paper. The boundary head is the shared substrate:
it removed the span architecture's 19-instances-per-type cap that had made either
formulation impossible to express.

## 2. Method

**Base models.** mmBERT-base trained `from_encoder` (fresh extraction heads) on a
multi-task pool of events, relations, entities, classification and structures, at nested
sizes of 10K / 40K / 100K / 137K records. The 137K point is reached by adding ~37K
non-leaking relation records to the 100K event pool, which warms the relation head at no
generation cost. `docred` is **excluded** from the pool because Re-DocRED re-annotates the
same documents and including it would leak the downstream evaluation; §3.2 verifies the
exclusion held.

**Decode arms.** Both arms are an eval-time switch on one trained model, so no training
difference can confound the comparison:

- *greedy* — each query thresholded independently, the shipped default.
- *joint* — constrained selection over the candidate scores, maximising summed edge score
  under typed constraints, with a beam of width W. The implementation keeps greedy as a
  floor, so the objective score is monotone in W by construction.

**Metrics.** Strict micro-F1 (exact match on type and span) unless stated. Relaxed
variants and per-category supports are reported where they change the reading.

## 3. Experimental integrity

Three defects in this project produced results that looked clean and were not. Each is
reported here with what it cost, because the failure modes generalise beyond this work.

### 3.1 A per-row split randomiser (found 2026-08-18)

The split writer drew a random number **per row** rather than per document, so a document
emitted more than once — normal in these corpora, where one document yields several task
records — had its copies scattered independently across train, validation and test.
Nothing crashed; no metric looked anomalous.

A repo-wide gate found **45 corpora** shipping overlapping splits; repair dropped
**21,553 records of 2.3M (0.94%)** under the precedence *test > val > train*. This
configuration additionally paired regenerated training files against frozen validation
slices, adding 252 train-in-val and 22 val-in-test documents.

**Every number in this paper post-dates that repair.** The scaling curve was re-run from
scratch on repaired data; all four points now gate clean. Numbers from before the repair
are not quoted here, and any earlier version of this document that quoted them is
superseded.

### 3.2 Verifying the downstream is genuinely blind

Because the headline result is read on Re-DocRED, the `docred` exclusion was verified
rather than assumed. Over the 137K training pool (129,155 distinct documents) against the
Re-DocRED evaluation splits, matched on normalised document text:

| split | documents | overlap with the 137K training pool |
|---|--:|--:|
| `redocred.test` | 500 | **0 (0.0%)** |
| `redocred.val` | 500 | **0 (0.0%)** |

### 3.3 A decode path that silently dropped a schema key

`structure` read exactly 0.0000 on all four scaling points and on the warm start — five
independent measurements, all exactly zero. The natural reading was that the record head
was broken or untrainable, and this document previously said so and recommended
abandoning structure data.

That reading was wrong. The inference runtime rebuilt each schema through a builder that
*does* produce `record_metadata`, then copied only two other keys out of the result. With
the metadata missing, the spec compiler returns an empty mapping, the record head decodes
nothing, **and no error is raised**. One dropped key produced five clean-looking zeros and
one confidently wrong conclusion. §4.4 reports what the head actually does.

Two further instances of the same failure were found and are recorded because the pattern,
not the instance, is the lesson: the fluent schema builder never emitted the key at all
unless the caller passed a mode argument, so a caller following the obvious API also got
silence; and the corpora used for the warm start emit structures without the metadata the
training path needs, so records that appear to supply structure supervision supply none.

### 3.4 The matched-threshold rule

**No arm comparison is readable until both arms are at their own swept threshold.** This
changed a conclusion three separate times in this experiment before being adopted as a
standing rule. The largest case: at a fixed threshold of 0.5, joint decoding appeared to
beat greedy by +0.106 strict F1 — but 0.5 is near greedy's *worst* operating point, and
at best-vs-best the real gap is +0.052. The fixed-threshold comparison overstated the
finding by a factor of two.

The corollary is that a metric read at an uncalibrated operating point is not a
measurement. The record head's own decode thresholds had never been swept by any tool,
and at the 0.5 default the head scores 0.0000 to 0.0760 depending on scale against
0.0238 to 0.1119 at its swept threshold.

## 4. Results

### 4.1 Joint decoding beats greedy, and the beam should be width 1

Re-DocRED, 96 relation types, one trained model, eval-time decode switch. Both arms swept
over the threshold grid; each read at its own optimum. Both peak at 0.2.

| threshold | greedy | joint (W=1) |
|---|--:|--:|
| 0.1 | 0.2785 | 0.3270 |
| **0.2** | **0.2835** | **0.3357** |
| 0.3 | 0.2270 | 0.3264 |
| 0.4 | 0.1648 | 0.2828 |
| 0.5 | 0.0980 | 0.2406 |

**Joint selection wins by +0.052 strict F1 (+18% relative) at best-vs-best, and beats
greedy at every threshold on the grid.**

The beam-width sweep is the more interesting result:

| W | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|--:|--:|--:|--:|--:|--:|--:|
| relation strict F1 | **0.2406** | 0.2290 | 0.2260 | 0.2211 | 0.2170 | 0.2152 | 0.2058 |

Monotonically decreasing. Widening the beam drops predictions 157 → 117, of which **18
were correct** — 45% precision on the dropped set against 61% overall — so precision rises
while F1 falls. Entity metrics are byte-identical at every width, because node admission
does not consult beam state; width touches only edges.

This is **score-versus-F1 divergence**. The beam maximises its objective *better* as it
widens — the implementation keeps greedy as a floor, so the score is monotone — and the
objective is not F1. A better search on a mis-specified objective produces worse output.

**So the gain is the formulation, not the search.** W=1 barely searches and wins outright.
The working contrast is *independent thresholding versus constrained joint selection*, not
*greedy versus beam*, and the original framing of this experiment as a beam-width study
was mis-specified.

For context, OneIE (Lin et al., 2020) used θ=10 with per-step branching capped at 2, and
its released package defaults to 5. Two independent global-IE decoders landed at 5–10
where our optimum is 1. The mechanism differs — their cap is on the label dimension, and
their beam has no greedy floor, so a narrow width there risks falling *below* the local
baseline — but the direction of the finding agrees.

A label-dimension cap was considered as the next move and **rejected**: the joint arm sits
at P=0.61 / R=0.15, so precision is four times recall, and a pruner targets the axis
already being won. Compute is not binding either — W=64 ran in the same wall clock as W=1.

### 4.2 The data-scaling curve

Four base sizes, re-run from scratch on repaired data. Strict micro-F1 on the held-out
blind test; the pool is 136,772 records at the largest point.

| head | 10K | 40K | 100K | 137K |
|---|--:|--:|--:|--:|
| event_type | 0.8138 | 0.7503 | 0.7979 | **0.9841** |
| event_trigger | 0.2259 | 0.3285 | 0.4032 | **0.6953** |
| classification | 0.0974 | 0.4838 | 0.5518 | **0.6336** |
| entity | 0.2779 | 0.4323 | 0.4795 | **0.5158** |
| event | 0.1475 | 0.2232 | 0.2643 | **0.2688** |
| relation | 0.0058 | 0.0175 | 0.0486 | **0.2071** |
| event_argument | 0.0130 | 0.0502 | 0.0815 | 0.0692 |
| structure † | 0.0294 | 0.0568 | 0.1102 | **0.1119** |

† The structure row is measured on the full test splits at each model's own swept record
threshold; see §4.4. Every other row is at the shared decision threshold.

**The 100K → 137K jump is recall unlocking, not new capability.** Only 1.37× the data, but
event_trigger gains +0.29, relation +0.16 and event_type +0.19. The precision/recall split
gives it away: event_trigger at 100K was P=0.58 / R=0.31, and at 137K P=0.62 / R=0.79. The
smaller models were not bad at the task — they were *withholding predictions*. Read as
smooth capability scaling, this curve supports the wrong conclusion about what more data
buys.

`event_argument` strict *falls* 0.0815 → 0.0692 while its relaxed score rises 0.2758 →
0.5064: the same inversion. More arguments proposed, approximately right far more often,
exact spans lagging. Calibration, not regression.

### 4.3 Warm start with 30% exact replay

The 137K base was warm-started on new-domain data (real news plus synthetic, 21,354
records) with 9,151 replay records drawn from **the base's own training pool**, then
scored on the base's own blind test:

| head | base 137K | warm start | delta |
|---|--:|--:|--:|
| entity | 0.5158 | **0.6326** | **+0.1168** |
| event | 0.2688 | **0.3644** | **+0.0956** |
| event_trigger | 0.6953 | **0.7490** | **+0.0537** |
| event_argument | 0.0692 | 0.0975 | +0.0283 |
| classification | 0.6336 | 0.6394 | +0.0058 |
| structure | 0.1119 | 0.1060 | −0.0059 |
| event_type | 0.9841 | 0.9447 | −0.0394 |
| relation | 0.2071 | 0.1245 | −0.0826 |

**With 30% replay the model gained on the original task while learning a new one.** For
context, three zero-replay fine-tunes of a comparable base on the same corpora lost 23%,
32% and 39% of general-domain entity F1, in monotonic order of training volume.

This arm is stronger evidence than those because we own both the base and its pool, so
the replay is **exact** rather than a proxy drawn from a stand-in corpus.

Qualifications that must travel with the result: it is a different architecture from the
zero-replay arms, so it is not a controlled contrast; part of the gain is genuinely new
learning, since the new corpora supply real entity signal; and **protection is not
uniform** — relation −0.083 and event_type −0.039 both declined, which is the open thread.

### 4.4 The structure head: what it does once it is asked

With the decode defect of §3.3 fixed, the same checkpoints were re-scored — nothing
retrained — at the record head's own swept thresholds:

| | 10K | 40K | 100K | 137K |
|---|--:|--:|--:|--:|
| structure strict F1 | 0.0294 | 0.0568 | 0.1102 | **0.1119** |
| at record threshold | 0.03 | 0.05 | 0.10 | 0.10 |

A monotone curve that flattens hard between 100K and 137K (+0.0017), where 100K is
actually the more precise model (0.303 versus 0.237). This is the ordinary shape of a head
learning from 7,754 supervised records. There was never an anomaly to explain — only a
question the head was never asked.

Recall is the binding constraint throughout: even the best model recovers 7.3% of gold
triples.

**The warm start is the sharper result.** Scoring both checkpoints on both test sets gives
the full 2×2, which is the only readable form:

| scored on | base 137K | warm start | delta |
|---|--:|--:|--:|
| 137K test set (856 records) | **0.1119** | 0.1060 | −0.0059 |
| warm-start test set (452 records) | 0.0755 | **0.2179** | **+0.1424** |

Read down the columns, never across the rows — the two test sets differ in difficulty and
distribution. The warm start **held the old capability (−5% relative) and nearly tripled on
the new distribution (+189%)**. The single-test-set reading, that it merely survived, could
not see the gain.

The mechanism matters more than the number. The new corpora contributed **zero** record-head
targets, because they emit structures without the metadata the training path requires
(§3.3): the record head saw 519 replay records against the base's 7,754, a 93% *reduction*.
The +0.1424 is therefore not learned record structure but **transfer from the span
representations the record head reads its field fillers out of** — entity F1 moved 0.5158 →
0.6326 over the same run. Supplying real metadata for those corpora is a measurable next
experiment rather than a speculative cleanup.

## 5. Limitations

- **The relation result is one model on one corpus.** Joint-versus-greedy is measured on
  Re-DocRED only. The event face of the same decode is not yet measured, so the thesis is
  supported on one of its two faces.
- **Single-run variance on these metrics is ±0.02 or worse.** A control re-run of a
  published recipe scored +0.023 above it from the re-run alone. The +0.052 joint-decode
  gain clears that floor; smaller deltas quoted anywhere in this paper do not, and are
  flagged where they appear.
- **The scaling curve is one seed per point.** Its *shape* is informative and its
  point-to-point differences at the 100K → 137K step are not.
- **`event_argument` at 137K sits below its 100K value** on strict, and the calibration
  reading of that inversion is an interpretation, not a measurement.
- **The warm start's protection is not uniform.** Relation and event_type both declined;
  no mechanism is established for why those two and not the others.
- **Structure supervision is not reaching the record head from most corpora**, so any
  future arm claiming structure capability from them will add none. Fixing it requires
  assigning a record mode and anchor per structure type — a data-design decision, not a
  patch.

## 6. Reproducibility

Configurations are in `tools/train/config/`: `joint-boundary-mmbert-{10k,40k,100k,137k}.yaml`
for the curve, `warmstart-137k-realsynth-replay30.yaml` for the replay arm. Replay slices
are built by `tools/train/build_137k_replay.py` (proportional across all 13 pool corpora,
seed 42). Record-head thresholds are swept by `tools/train/sweep_record_thresholds.py`,
which no earlier tool did — the trainer's threshold sweep moves the shared decision
threshold and never touches the record cutoffs. Split integrity is gated by
`tools/data/check_leakage.py --config`, wired into `tools/train/train.py` so it runs before
every training run.

Models are published privately as `whr778/gliner2-joint-boundary-mmbert-{10k,40k,100k,137k}-clean`
and `whr778/gliner2-warmstart-137k-realsynth-replay30`. Each model card reports the
structure metric at its own swept record threshold alongside what the 0.5 default would
have scored.

The build record — decision log, wiring map, increment status, data survey, cost model and
the deferred Phase B plan — is in `JOINT_IE_DESIGN_RECORD.md`.

## 7. References

- Lin, Y., Ji, H., Huang, F., Wu, L. (2020). *A Joint Neural Model for Information
  Extraction with Global Features* (OneIE). ACL.
  https://aclanthology.org/2020.acl-main.713/
- Zaratiana, U., Pasternak, G., Boyd, O., Hurn-Maloney, G., Lewis, A. (2025). *GLiNER2:
  Schema-Driven Multi-Task Learning for Structured Information Extraction.* EMNLP 2025
  System Demonstrations. https://aclanthology.org/2025.emnlp-demos.10/
- Marone, M., Weller, O., Fleshman, W., Yang, E., Lawrie, D., Van Durme, B. (2025).
  *mmBERT: A Modern Multilingual Encoder with Annealed Language Learning.*
  arXiv:2509.06888. https://arxiv.org/abs/2509.06888
- Tan, Q., Xu, L., Bing, L., Ng, H.T., Aljunied, S.M. (2022). *Revisiting DocRED —
  Addressing the False Negative Problem in Relation Extraction* (Re-DocRED). EMNLP.
  https://aclanthology.org/2022.emnlp-main.580/
