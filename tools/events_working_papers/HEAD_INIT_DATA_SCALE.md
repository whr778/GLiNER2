# How Many Samples to Train a Warm-Start Head? — An Estimate

Status: exploratory estimate (reasoned bracket, not a measured result). Date: 2026-08-03.
Question: is fastino's **254K** the minimum to build a good warm-start GLiNER2 head, or
would a smaller corpus be good enough? Companion to
[`FASTINO_GLINER2_TRAINING.md`](FASTINO_GLINER2_TRAINING.md) and PAPER.md §10.6/§10.7.

**Bottom line: 254K is well above the minimum.** It is fastino's choice for broad,
three-task, many-domain, *zero-shot-generalizing* coverage — not the floor for a usable
warm-start head. My central estimate for "good enough to warm the span/structure/argument
heads, then fine-tune downstream" is **~30-80K diverse, structure-dense examples (~50K as
a single-number guess)**. This is an estimate; the definitive answer is a data-scaling
curve (Section 4).

---

## 1. The evidence bracket

Three data points bracket the from-scratch threshold:

| Examples | Setup | Outcome |
|--:|---|---|
| **1,497** | fresh heads from raw encoder (`deberta-base-fromenc-synthetic`) | **fails** — entity 0.14, relation 0.00, event-arg 0.00, trigger 0.22 (only coarse event-type 0.998) |
| **254,334** | fastino `gliner2-base-v1`, fresh heads | **works** — RAMS arg-strict 0.462 after fine-tune |
| 1,497 | *fine-tuning existing* fastino heads (`gliner2-base-v1-synthetic`) | works (entity 0.90, arg 0.70) — but adaptation, not head-building |

So the "build a head from scratch" threshold sits in the ~170x gap between 1.5K and 254K.
External anchors from the GLiNER lineage: the **original GLiNER** reached strong zero-shot
NER heads from *tens of thousands* of LLM-annotated passages (Pile-NER scale, ~45K), and
GLiREL / GLiClass reached usable relation / classification heads at similar scale.

## 2. The estimate

- **Floor (heads do not form): below ~5-10K.** Our 1,497 is deep in the failure zone;
  even 5-10K is likely too few for the span + count + occurrence-ID machinery.
- **"Good enough" warm-start head: ~30-80K** diverse, structure-dense examples, central
  guess **~50K** — where the span-matching + count + occurrence-ID heads should become
  competent enough for a downstream RAMS/WikiEvents fine-tune to take over.
- **254K buys the extra ~3-5x** for *broad zero-shot generalization* across many label
  types and domains — which we may not need if the goal is "warm the argument head, then
  fine-tune a specific event ontology."

## 3. What actually drives the number ("254K" is the wrong unit)

1. **Diversity / density, not raw doc count.** The span head learns "(span, type) ->
   match?" decisions; 50K examples spanning many structures and argument-role types beats
   100K near-duplicates of one schema. Our 1.5K had all five task types but far too few
   examples *per type/role* — starvation, not just small N.
2. **Zero-shot vs downstream-only changes it a lot.** Fastino needs 254K to generalize to
   *unseen* schemas at inference. To merely *initialize before fine-tuning* on a known
   ontology, far less is needed — the downstream data supplies the ontology.
3. **The encoder does the heavy lifting.** On a strong pretrained DeBERTa-v3, the head
   learns a fairly simple matching function, so moderate data suffices. On a cold /
   multilingual encoder you need more (part of why the mmBERT-from-encoder runs struggled).

## 4. How to actually know: a data-scaling curve

An estimate is a guess; the real answer is a scaling curve. Train the from-encoder
structure/argument head at several corpus sizes and plot **downstream RAMS arg-strict F1
vs base-corpus size**; the knee is the practical minimum.

- **Design:** 3 points is enough to find the knee — e.g. **~10K / ~40K / ~120K** — fresh
  heads (`from_encoder`), identical recipe, only the base-corpus size varies, each then
  fine-tuned on RAMS (or WikiEvents) under a fixed recipe. Prediction: **knee at ~40-60K**.
- **Caveat A — corpus must be structure/argument-dense**, not the NER-heavy public sets
  (GLiNER multilingual/multi-task), or it will not exercise the head that matters. Our
  own `synthetic_sonnet5` shape (events + structures + relations) is right, but only 1.5K.
- **Caveat B — data sourcing / cost.** To hit 10K-120K we either generate more synthetic
  (Sonnet-5 batch per `tools/data/synthetic/COST_BREAKDOWN.md`, ~a few $ for tens of K) or
  assemble a structure/argument corpus from existing event/structure datasets. Generation
  is the cleaner control (uniform shape, known quality) but is the cost driver.
- **Cheapest informative version:** the 3-point curve above on an A10 (each run is minutes;
  the pole is data generation, not compute).

## 5. Practical recommendation

If the goal is a better *event* model rather than a general zero-shot model, do **not**
target 254K event docs (wrong unit; see [`FASTINO_GLINER2_TRAINING.md`](FASTINO_GLINER2_TRAINING.md)
- the transferable competence is the *structure* head). Instead build a **structure/argument
curriculum at ~40-80K** in the fastino mold (mixed real+synthetic, LLM-annotated and
validated), warm the heads on it, then fine-tune. Run the 3-point scaling curve first to
confirm the knee before committing to the full generation cost.

## 6. Related
- [`FASTINO_GLINER2_TRAINING.md`](FASTINO_GLINER2_TRAINING.md) — fastino's actual recipe
  (supervised LLM-distillation, shared span/count/occurrence heads, differential LR).
- PAPER.md §10.6 (head-init bottleneck), §10.7 (combined null + synthetic sanity +
  from-encoder from-scratch failure).
- Memories: `mmbert-head-init-finding`, `lambda-base-v1-synthetic-sanity`.
