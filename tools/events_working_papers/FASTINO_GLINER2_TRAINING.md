# How `fastino/gliner2-base-v1` Was Actually Trained — and Why It Warms Our Event Heads

Status: reference note (verified against the source paper). Date: 2026-08-03.
Source: Zaratiana, Pasternak, Boyd, Hurn-Maloney, Lewis (Fastino AI), *GLiNER2: An
Efficient Multi-Task Information Extraction System with Schema-Driven Interface*,
EMNLP 2025 System Demonstrations, pp. 130-140. https://aclanthology.org/2025.emnlp-demos.10/

This documents the fastino base model's real training recipe (data, paradigm, heads)
because our head-initialization thesis (PAPER.md §10.6/§10.7) depends on it, and our
initial understanding was incomplete. All figures below are quoted from the paper's
§3.1, Appendix A, Appendix B, and Tables 5-6.

---

## 1. Training data — 254,334 examples, ~47% synthetic / ~53% real (Table 6)

| | Examples | Share |
|---|--:|--:|
| **Real-world data** | **135,698** | **53.4%** |
| — News | 74,456 | |
| — Law | 19,798 | |
| — Wikipedia | 17,909 | |
| — PubMed | 16,400 | |
| — ArXiv | 7,135 | |
| **Synthetic data** (mixed domains) | **118,636** | **46.7%** |
| **Total** | **254,334** | |

- **Real-world half** = real *text* (news/Wikipedia/legal/PubMed/ArXiv) + **GPT-4o
  annotations**: *"All documents were automatically annotated with GPT-4o using
  task-specific prompts and validated for quality."*
- **Synthetic half** = GPT-4o-generated *text and* labels: *"generated 118,636
  synthetic examples with GPT-4o ... each synthetic example includes complete
  annotations for all tasks."* Covers emails, texts, resumes, social posts,
  e-commerce orders, banking records, sports commentary.
- **100% of the annotation is GPT-4o.** The two halves differ only in whether the
  *text* is real or generated.

## 2. Training paradigm — fully supervised on LLM-distilled labels

- **Supervised**, not semi-supervised (no unlabeled pool) and not self-supervised:
  every example carries complete task labels.
- The supervision signal is **GPT-4o knowledge-distillation / weak (distant)
  supervision** — the GLiNER lineage's signature recipe (the paper cites NuNER and
  UniversalNER, both "distill labels from an LLM teacher"). Mechanically it is
  standard supervised training; the novelty is the label *source*.
- **Config (Table 5):** 5 epochs, AdamW, **differential LR — encoder backbone 1e-5,
  task heads 2e-5**, weight decay 0.01, 1,000 warmup steps, gradient clipping 1.0.
  I.e. heads adapt ~2x faster than the encoder, over only 5 epochs on 254K.

## 3. The heads (Appendix A) — a shared span head + a structure head; NO event/relation head

Backbone: DeBERTa-v3 encoder. Five learned special tokens, randomly initialized:
`[P]` prompt, `[E]` entity type, `[C]` child/component field, `[L]` label, `[SEP]`.

1. **Span-matching head (bilinear dot-product), shared by NER *and* structure fields.**
   Enumerate all spans up to a max width; score each span against a type embedding:
   `score(s_i, e_j) = sim(h_{s_i}, h_{e_j})` = dot product -> **sigmoid**, keep > 0.5.
   -> BCE-style per span-type pair.
2. **Count-prediction head** (hierarchical structures): an MLP on the `[P]` token doing
   **20-way classification of the instance count (0-19)**.
3. **Occurrence-ID conditioning:** base `[C]` field embeddings combined with learned
   per-instance embeddings -> K distinct field representations, each matched to spans
   via head #1 (so K instances keep separate attribute values).
4. **Classification head:** `logit_i = MLP(h_{L_i})` -> softmax (single-label) or
   sigmoid (multi-label).

Objective = multi-task mix of **BCE (span-type) + softmax-CE (count head) +
softmax/sigmoid-CE (classification)** — which maps onto our codebase's
`structure_loss` / `count_loss` / `classification_loss` split.

**Training tasks were exactly three: entity recognition, hierarchical structure
extraction, and text classification. No events. No relations.**

## 4. Why this explains our results

Our event extraction reuses heads #1-#3, with no dedicated event head:
- **trigger / argument spans** = the span-matching head (#1),
- **multiple event mentions** = the count head (#2),
- **per-instance arguments** = occurrence-ID conditioning (#3).

The fastino hierarchical-structure task (parent structure + `[C]` child fields +
count + span-attribute matching) is *architecturally the same machinery* as event
trigger->argument extraction. Therefore:

- **Warm-start from `fastino/gliner2-base-v1` lifts RAMS/WikiEvents arguments even
  though fastino trained on zero events** — the structure->span-attribute head is
  already trained (supervised, 254K, 5 epochs, differential LR). This is the
  head-initialization effect of PAPER.md §10.6.
- **From-encoder on ~1.5K synthetic records fails** (PAPER.md §10.7 / our
  `deberta-base-fromenc-synthetic` run: entity 0.14, relation 0.00, arg 0.00): the
  span + count + occurrence-ID heads cannot be trained from random init at that
  scale. The warm-start counterpart (`gliner2-base-v1-synthetic`) works (entity 0.90,
  arg 0.70) because it inherits those heads.
- **Ceiling caveat:** labels are GPT-4o-distilled, so head quality is bounded by
  GPT-4o's IE quality on the 254K — which is why the paper validates its real-world
  annotations, and why our synthetic pipeline's valid/verifiable-JSON push matters.

## 5. Implication for the head-init roadmap

- **The ~250K scale (or a warm-start from it) is the lever, not optional** — our
  from-encoder failure is ~170x under it.
- **But it need not be 250K *event* examples.** Since the *structure-extraction* head
  is what transfers, the efficient path to better events is a large **hierarchical-
  structure** synthetic curriculum (fastino-scale, mixed real+synthetic, LLM-annotated
  and verified), which warms the same head — a sharper version of §10.7's "heavier
  head-init pass." Building 250K event docs is the wrong target; building a
  structure/argument curriculum at scale is the right one.

## 6. Related
- PAPER.md §10.6 (head-init bottleneck), §10.7 (combined A/B null + synthetic sanity +
  from-encoder from-scratch failure).
- Our synthetic pipeline: `tools/data/synthetic/`.
- Prior head-init evidence and the two synthetic runs: memories
  `mmbert-head-init-finding`, `lambda-combined-experiment`,
  `lambda-base-v1-synthetic-sanity`.
