# English event annotation with Sonnet 5.5: spec and price

**Status:** SPEC 2026-10-06, not bought. Budget remaining for annotation: ~$98.

## 1. Why

English is the measured gap. In eb19, CASIE is the only English corpus with both train and val
splits, and its in-run validation junction AUC drifted 0.744 -> 0.715 over epochs 2-5 while
training held. The blind test's argument score is 88.4% Chinese CMNEE gold.

The existing English LLM gold is noisy:

| comparison (50 cc_news docs) | events F1 |
|---|---|
| Haiku 4.5 vs Sonnet 5.5 | ~0.40 |
| Sonnet 5.5 vs Sonnet 5.5, same prompt | 0.679 |

Given an event both annotators found, their arguments agree at ~0.70. The disagreement is
**which events exist, and their type** (EXPERIMENT_CATALOG, 2026-10-05).

So the goal is not more of the same labels. It is better-defined labels, stabilised.

## 2. Source

`data/cc_news_parts/cc_news_pool_c_raw.jsonl` has 59,996 English news docs. 8,300 are already
annotated (the pilot plus main8k), which leaves **51,696 unused**. The unused docs are a median
of 2,050 characters.

Before purchase:
- deduplicate the selection against every corpus in `data/` (`check_leakage.py`);
- record the selection seed.

## 3. Recipe

| component | state |
|---|---|
| Tasks | **DECIDED: recipe D, all five tasks**: entities (whole pool), the full 56-type event menu, relations, classifications, structures |
| Event definitions | `schema_spec.EVENT_DEFINITIONS` (56 types, one line each plus a "Not: ..." boundary). **DRAFT, awaiting review.** |
| Events section in GUIDELINES.md | what counts as an event, one record per occurrence, trigger extent, argument extent. **Blocked on 4 user decisions.** |
| GUIDELINES blocks | `json_only, verbatim, no_inference, minority, ambiguity`. Measured null on agreement but harmless; kept for conduct. |
| Self-consistency | k runs per doc; keep an event (type, trigger) found in at least 2 of 3 runs, and keep an argument found in at least 2 of the runs that kept its event |
| Model | `claude-sonnet-5-5`, Batch API. Thinking off means `between_tools`; `providers.py` already maps this. |

## 4. Price per 1,000 documents (Batch API: $1 in / $5 out per MTok, verified on the pricing page 2026-10-05)

| recipe | input tok/doc | output tok/doc | **$ per 1,000** | basis |
|---|---|---|---|---|
| A. entities+events, 1 run | 3,953 | 602 | **$6.96** | input counted on the unused pool; output MEASURED (batch usage, 50 docs) |
| B. A + event definitions | 6,791 | ~620 | **$9.89** | definitions counted at +2,838 tok; output estimated |
| C. B + GUIDELINES blocks + events section | ~7,200 | ~620 | **~$10.30** | estimate |
| C x 3 runs (self-consistency) | | | **~$30.90** | 3 x C |
| D. C + relations, classifications, structures | ~7,600 | ~1,000 | **~$12.60** | estimate. Output is the unknown, so price it in the pilot. |

What $98 buys:

| plan | docs |
|---|---|
| C x 1 | ~9,500 |
| C x 3 | ~3,100 |
| **Split** (recommended): held-out val/test at C x 3, train at C x 1 | 1,000 eval docs ($31) + ~6,400 train docs ($66) |

The recommended split puts the voting where measurement needs it, so that **eval gold is the most
stable gold**, and puts volume where training needs it.

**Prompt caching (not counted above):** the instructions, menu, definitions and guidelines form a
static prefix of ~5-6k tokens, if per-doc label sampling is off. A cache read on Sonnet 5.5 is
0.1x input, and the batch discount stacks on it, so input could fall by roughly half. But cache
hits inside a batch are best-effort. Measure the hit rate in the pilot before counting on it.

## 5. Pilot: gates before buying (~$3.50, 50 docs)

Run C six times on the same 50 cc_news docs (not from the existing set). Measure three things:

1. **Do definitions help?** Compare single-run self-agreement for C against the 0.679 baseline,
   using a redraw of the same prompt.
2. **Is the voted gold stable?** Compare the events F1 of two INDEPENDENT 3-run votes, runs 1-3
   against runs 4-6. That number is the reproducibility of the gold the eval would score against.
3. **Price:** real output tokens and the cache hit rate.

Stop rule. **The bar is FIXED at 0.85 events F1 (user, 2026-10-06), before any pilot result:**

| result | decision |
|---|---|
| (2) at or above the agreed bar, proposed 0.85 events F1 | buy |
| (2) below the bar | do not buy; re-think the menu (e.g. merge overlapping types) |
| (1) no gain | drop the definitions and save ~$2.90 per 1k per run |

A gate must be able to fail: report the bar BEFORE seeing the result.

## 6. After purchase

The same discipline as `cc_news_events_haiku45`:
- 80/10/10 split, test blind;
- uniqueness and cross-corpus leakage verified;
- labels through `unified-full.yaml`, map closed;
- private Hub repo, registry entry (`provenance: llm_real`), restorable;
- `compare_annotators.py` agreement report, plus label marginals, in the catalog row.

## 7. Decisions

**Decided (user, 2026-10-06):**
- Recipe D, all five tasks.
- Stability bar 0.85 events F1.
- Reporting verbs ('said', 'told', 'talked', 'chatted') ARE Contact.Communicate events.

Measured consequence of the reporting-verb decision, for the pilot to watch:
- the unused pool averages 4.0 reporting verbs per doc (79% of docs have one);
- Haiku's existing gold had Communicate at 8.2% of events, ~1.7 events per doc;
- Communicate could therefore become ~70% of events, at about +$1 per 1k output.

The pilot reports Communicate's share. A middle option, if it swamps the other types: count a
reporting verb only when a Recipient or Topic is named.

**Open:**
1. Demonstrate/Protest merge.
2. Whether announced or planned events count.
3. The events-section questions:
   - event status;
   - repeated mentions;
   - argument extent;
   - partial type fit.
