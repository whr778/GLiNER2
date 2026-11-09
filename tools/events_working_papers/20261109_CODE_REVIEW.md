# Code review: argument width, head spans, and per-head thresholds

**Date**: 2026-11-09
**Target**: The decode and inference changes from `0607b29` through `8157633` on `merge/upstream-20261014`. Doc-only paper updates are out of this review.
**Entrypoint**: [gliner2/models/boundary/records.py](../../gliner2/models/boundary/records.py) `RecordHead.forward_group`, `decode_group`, and `compute_group_loss`; [gliner2/inference/head_spans.py](../../gliner2/inference/head_spans.py); [gliner2/inference/runtime.py](../../gliner2/inference/runtime.py).
**Prior review**: [20261025_CODE_REVIEW.md](20261025_CODE_REVIEW.md)
**This paper changes no code.** The snippets under "Recommended patch" are proposals.

Traced 2026-11-09 against the functions as they stand. `strip_leading_stopwords_text` and `expand_entity_wildcards` were called on the strings quoted below. The 40-document precision numbers are the check already run on the first 20 lines of `data/casie.val.jsonl` and the first 20 lines of `data/cmnee.val.jsonl`; this paper did not re-run that model.

---

## 1. Verdict

Two defaults are live on every checkpoint whose `config.json` does not contain the new keys, including `whr778/gliner2-eb22g` as published. `record_field_max_tokens` (`24`) filters wide argument candidates inside `RecordHead.forward_group` (both training and inference), while `record_field_min_pair_logit` (`-10.0`) is applied in `decode_group` (inference only). Filtering wide chimeras in `forward_group` is desirable for training because 100-token chimeras otherwise dominate the top-$K$ (`role_hard_negatives: 8`) negative pool in `_list_field_bce` and push down valid start/end boundary representations. However, `record_field_max_tokens` is currently listed under `_EVAL_TIME_BOUNDARY_KEYS` without a gold-drop counter in `compute_group_loss`, and `record_field_min_pair_logit` is not applied in `forward_group`, leaving a train/decode asymmetry for shorter reranker-vetoed spans.

`head_spans` is off unless the flag is set. When it is on, it strips `"El Paso"` to `"Paso"`, `"de Gaulle"` to `"Gaulle"`, and `"as many as 40 people"` to `"many as 40 people"`. That last one is a quantifier phrase the EKF path needs intact.

`per_head_thresholds.classification` is stored on the card and is not read by `_decode_classification_probabilities`. `"*"` expands to a bounded 125-type menu (`DEFAULT_OPEN_VOCAB_ENTITIES`), which is why an open-vocab model now returns entities without overflowing the prompt window, though its deduplication in `expand_entity_wildcards` is currently case-sensitive.

The seven pipeline fixes in `0607b29` do what they were written to do. Section 8 records that and stops.

---

## 2. `record_field_max_tokens` in `forward_group`: why it helps training, and the missing gold-drop gate

`BoundaryHeadSettings` defaults both candidate filters on:

```python
# gliner2/configuration.py
record_field_max_tokens: int = 24
record_field_min_pair_logit: float = -10.0
```

`validate_boundary_head` populates those defaults whenever a loaded `config.json` omits the keys (`configuration.py:710-715`). A checkpoint saved before this change therefore decodes with `24` and `-10.0` under current code. The published `eb22g` Hub metrics were scored before that decode existed.

In `tools/train/train.py`, both keys were added to `_EVAL_TIME_BOUNDARY_KEYS`:

```python
# tools/train/train.py
# The ARGUMENT gate (TODO 22): read at decode, sizes nothing.
"record_field_threshold", "record_field_threshold_wins",
"record_field_max_tokens", "record_field_min_pair_logit",
```

`record_field_min_pair_logit` is purely decode-time: it is passed into `decode_group` from `BoundaryExtractor._decode_records` and is not read in `forward_group`.

`record_field_max_tokens`, by contrast, is also active inside `RecordHead.forward_group`:

```python
# gliner2/models/boundary/records.py, forward_group
max_toks = getattr(self, "max_field_tokens", 0) or 0
for f_idx, (qid, query_valid) in enumerate(zip(safe_field_query_ids, valid_query_ids)):
    row_spans = spans_rows[f_idx]
    cap_width = max_toks > 0 and f_idx != anchor_f_idx
    keep_list = (
        [
            c for c, ok in enumerate(valid_rows[f_idx])
            if ok and (not cap_width or (row_spans[c][1] - row_spans[c][0]) <= max_toks)
        ]
        if query_valid else []
    )
```

### Why filtering in `forward_group` is right for training

In `RecordHead`, candidate representations are built by `candidate_encoder` from `[g_start, g_end]` alone (no inside-token pooling or span-width penalty). When a valid argument start (e.g. `"San'a"`) and a valid argument end 99 tokens later (e.g. `"Hormuz"`) are paired by the proposer, their bilinear assignment score in `RecordHead` is a linear combination of the start and end compatibilities. During training, `_list_field_bce` (`records.py:1326`) with `role_hard_negatives = 8` (`hard_k = 8`) mines the 8 highest-scoring wrong candidates per role. If 100-token chimeric pairs remain in `group.field_spans` during `forward_group`:

1. Chimeras with high endpoint compatibility occupy the `hard_k = 8` slots, crowding out local same-sentence false entities that the record head actually needs to learn to separate.
2. Pushing down the assignment score of `[g_start("San'a"), g_end("Hormuz")]` applies negative gradients directly to the valid endpoint embeddings of `"San'a"` and `"Hormuz"`, fighting the positive supervision on `"San'a, Yemen"`.

Keeping `record_field_max_tokens` inside `forward_group` prevents both pathologies.

### What is missing: a gold-drop counter in `compute_group_loss`

`compute_group_loss` resolves gold argument spans against the filtered `group.field_spans`:

```python
# gliner2/models/boundary/records.py
def _resolve_value_cols(value_alternatives, span_to_idx):
    return [
        idx + 1 for span in value_alternatives
        if (idx := span_to_idx.get((int(span[0]), int(span[1])))) is not None
    ]
```

If a gold argument span is wider than `max_field_tokens` (24 subword tokens, which is narrower than 24 whitespace words), `forward_group` drops it from `group.field_spans` before `compute_group_loss` runs. `_resolve_value_cols` then returns `[]` for that gold value, and nothing logs that a gold argument was dropped by the width cap. By our gate rules, a filter that can silently drop gold supervision during training must carry a counter that can print a non-zero.

Second, because `record_field_max_tokens` alters the training forward pass when `max_field_tokens > 0`, documenting it in `_EVAL_TIME_BOUNDARY_KEYS` under `# read at decode, sizes nothing` is inaccurate—it is a parameter-free *both-time* setting (safe to override on a warm start because it adds/reshapes zero tensors, but active in both `forward_group` and `decode_group`).

Finally, note the precision check already run on `casie.val[:20] + cmnee.val[:20]` at threshold `0.7`: `event_argument` strict micro-precision moved from **`0.3193 -> 0.3346`** (`+0.0153`) and strict micro-F1 from **`0.2073 -> 0.2104`** (`+0.0031`), with zero recall loss on that 40-document slice. That check masked logits after `forward_group`, so it is a decode measurement, and its F1 delta sits inside the single-run noise floor (`±0.02`).

### Recommended patch

Keep `max_field_tokens` in `forward_group`, and record any gold argument span whose token width exceeds the cap inside `compute_group_loss` (where gold `records: Sequence[RecordTarget]` are in scope):

```python
# gliner2/models/boundary/records.py, inside compute_group_loss ( or _instance_field_loss )
for f_idx, fspec in enumerate(group.field_specs):
    if f_idx == anchor_f_idx:
        continue
    target = record.field_for_query(fspec.query_id)
    if target is None:
        continue
    for alternatives in target.values:
        if alternatives and not _resolve_value_cols(alternatives, span_indices[f_idx]):
            if any(( int(s[1]) - int(s[0]) ) > max_toks for s in alternatives):
                _note_anchor_gate("arg_exceeds_max_tokens", group.spec.task_type)
```

Verify the counter with a unit test that passes a 25-token gold argument into `compute_group_loss` and asserts that `anchor_gate_stats()[f"{task_type}/arg_exceeds_max_tokens"] == 1`. Also update the comment in `tools/train/train.py` above `_EVAL_TIME_BOUNDARY_KEYS` to note that `record_field_max_tokens` is a tensor-free candidate width cap applied in both `forward_group` and `decode_group`.

---

## 3. `head_spans` strips name particles and one quantifier the EKF needs

The flag is off by default. These traces are `strip_leading_stopwords_text` with the live stopword set, including `tools/train/config/stopwords.yaml`, run 2026-11-09.

| input | returned text | `start` delta | `text[delta:] == returned` |
|---|---|---:|---|
| `El Paso` | `Paso` | 3 | yes |
| `La Paz` | `Paz` | 3 | yes |
| `de Gaulle` | `Gaulle` | 3 | yes |
| `da Silva` | `Silva` | 3 | yes |
| `Al Qaeda` | `Qaeda` | 3 | yes |
| `as many as 40 people` | `many as 40 people` | 3 | yes |
| `as few as 3` | `few as 3` | 3 | yes |
| `Los Angeles` | `Los Angeles` | 0 | yes |
| `New York` | `New York` | 0 | yes |
| `at least 246 people` | `at least 246 people` | 0 | yes |
| `a dozen people` | `a dozen people` | 0 | yes |
| `the port ` | `port` | 4 | **no** (`text[4:]` is `port `) |
| `in progress` | `progress` | 3 | yes |

Four concrete bugs show up in this trace:

1. **Missing `"as many as"` / `"as few as"` in `_QUANTIFIER_GUARD_RE`:** `"as"` is in `_EN_LEADING_STOPWORDS`. `_QUANTIFIER_GUARD_RE` matches `at least`, `more than`, `up to`, `a dozen`, `nearly`, etc., but omits `as many as` and `as few as`. The loop strips `"as "`, then stops because `"many"` / `"few"` are in `_PROTECTED_WORDS`, leaving `"many as 40 people"`.
2. **Proper-noun particles (`El Paso`, `La Paz`, `de Gaulle`, `da Silva`, `Al Qaeda`):** `Los`, `Las`, and `San` are in `_PROTECTED_WORDS`, whereas `el`, `la`, `de`, `da`, and `al` are in `_MULTILINGUAL_ARTICLES` (or `stopwords.yaml`) and are stripped unconditionally. Note that naively refusing to strip `el`/`la` whenever the next word is capitalized would fail on sentence-initial Spanish/French spans before a title or proper noun (e.g. `"El Presidente..."`, `"La Ministra..."`). Instead, we should protect known multi-word proper-noun prefixes and lowercase nobiliary particles (`de`, `da`, `van`, `von`, `bin`, `al`, `del`, `di`) when followed by a capitalized surname, while keeping `El` / `La` protected specifically for capitalized place/organization names (or when `El`/`La` occurs mid-sentence with a capital letter).
3. **Trailing whitespace (`"the port "`):** `s = lstripped.rstrip()` trims trailing whitespace before the loop, while `delta` only measures leading characters removed and `_strip_span_item` never decrements `end`. If a span carries trailing whitespace, `doc[start:end]` no longer equals `span["text"]`.
4. **Confidence-blind deduplication (`_dedupe_span_list`) and bare-string stripping:**
   Traced on a formatted result:
   ```text
   entities Location:
     {text: "the Red Sea", start: 4, end: 15, confidence: 0.41}
     {text: "Red Sea",     start: 8, end: 15, confidence: 0.91}
   -> [{text: "Red Sea", start: 8, end: 15, confidence: 0.41}]
   ```
   When `"the Red Sea"` (`0.41`) strips to `"Red Sea"` (`8..15`), its key collides with the existing `"Red Sea"` (`0.91`), and `_dedupe_span_list` keeps the first item (`0.41`) and drops `0.91`.
   In addition, on structure instances (`{"status": "in progress"}` when `include_spans=False`), `_strip_span_item` strips bare strings that have no `start`/`end`, turning enum choice `"in progress"` into `"progress"`.

### Redundant calls on the long-document path

`batch_extract_long` (`gliner2/inference/runtime.py:2027-2060`) passes `head_spans=head_spans` into `batch_extract` (stripping each chunk), then `remap_result_spans` (`gliner2/inference/chunking.py:100-105`) overwrites `remapped["text"] = original_text[start:end]` using the shifted `start`, and `batch_extract_long` calls `strip_leading_stopwords_from_results(merged)` a second time without passing `classification_tasks`. In `tools/infer.py:395`, when `--entity-coref` is active, `strip_leading_stopwords_from_results` is called a third time on the full result dict rather than only on `coref_clusters`.

### Recommended patch

1. **Add `as many as` / `as few as` to `_QUANTIFIER_GUARD_RE`:**
```python
_QUANTIFIER_GUARD_RE = re.compile(
    r"^(?:at\s+least|more\s+than|less\s+than|fewer\s+than|up\s+to|"
    r"as\s+many\s+as|as\s+few\s+as|"
    r"a\s+(?:dozen|hundred|thousand|million|billion|few|couple)|"
    r"about|around|roughly|nearly|almost|approximately|over|under|"
    r"feared|estimated)\b",
    re.I,
)
```

2. **Protect nobiliary particles and `El`/`La`/`Al` proper names without breaking sentence-initial articles:**
```python
# Particles that almost exclusively attach to proper surnames/orgs in English/multilingual NER
_SURNAME_OR_ORG_PARTICLES = frozenset({
    "de", "da", "del", "di", "van", "von", "al",
})
# Known proper-name prefixes starting with El / La
_PROPER_PREFIX_RE = re.compile(
    r"^(?:el\s+(?:paso|salvador|nino|niño|dorado|cajon|monte|centro|segundo|cerrito)|"
    r"la\s+(?:paz|habana|jolla|verne|crosse|mesa|plata|rioja|coruña|salle))\b",
    re.I,
)
```
Inside the `while True` loop in `strip_leading_stopwords_text`:
```python
if _PROPER_PREFIX_RE.match(s):
    break
next_word = rest.lstrip().split(" ", 1)[0]
if first_clean in _SURNAME_OR_ORG_PARTICLES and next_word[:1].isupper():
    break
```

3. **Preserve exact `end` offsets on trailing whitespace (or do not `.rstrip()` inside `strip_leading_stopwords_text`):**
```python
# In strip_leading_stopwords_text: do not rstrip `s` so len(text) == delta + len(s)
s = lstripped
```
If trailing whitespace is not trimmed, `text[delta:] == new_text` holds unconditionally and `end` never needs adjustment.

4. **Keep the highest-confidence span in `_dedupe_span_list` and event arguments:**
```python
def _dedupe_span_list(items: List[Any]) -> List[Any]:
    out: List[Any] = []
    key_to_idx: Dict[ Any, int ] = {}
    for item in items:
        if isinstance(item, str):
            key = (item.lower(), None, None)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            key = (item["text"].lower(), item.get("start"), item.get("end"))
        else:
            out.append(item)
            continue
        if key not in key_to_idx:
            key_to_idx[key] = len(out)
            out.append(item)
        else:
            prev = out[key_to_idx[key]]
            if (isinstance(item, dict) and isinstance(prev, dict)
                    and (item.get("confidence") or 0.0) > (prev.get("confidence") or 0.0)):
                out[key_to_idx[key]] = item
    return out
```

5. **Run `strip_leading_stopwords_from_results` once in `batch_extract_long`:**
Pass `head_spans=False` to the inner `batch_extract` call in `batch_extract_long`, and run `strip_leading_stopwords_from_results(merged)` once after `merge_chunk_results`. In `tools/infer.py`, strip only `r["coref_clusters"]` after `resolve(...)` when `--entity-coref` is used.

---

## 4. `per_head_thresholds.classification` is stored and never read

Entity, relation, event, and structure thresholds are wired and active:
- `SpanExtractor._decode_span_probabilities` reads `entity`, `relation`, `event`, and `structure`.
- `BoundaryExtractor._query_thresholds` maps `spec["task_type"]` through `_TASK_TO_HEAD` for extractive queries.
- `BoundaryExtractor._decode_records`, `_decode_relations`, and `_decode_legacy_structures` read `event`, `structure`, and `relation`.

Classification, however, does not use an extractive query and is decoded by `_decode_classification_probabilities` (`gliner2/inference/runtime.py:1118-1145`), a `@staticmethod` that never receives `metadata`:

```python
# gliner2/inference/runtime.py
@staticmethod
def _decode_classification_probabilities(
    results: Dict,
    schema_name: str,
    cls_config: Dict,
    probs: torch.Tensor,
) -> None:
    labels = cls_config["labels"]
    cls_threshold = cls_config.get("cls_threshold", 0.5)
    is_multi = cls_config.get("multi_label", False)

    if is_multi:
        chosen = [
            (labels[j], probs[j].item())
            for j in range(len(labels))
            if probs[j].item() >= cls_threshold
        ]
        ...
    else:
        best = int(torch.argmax(probs).item())
        results[schema_name] = (labels[best], probs[best].item())
```

Consequently:
1. For **multi-label** classification (`multi_label=True`), `per_head_thresholds["classification"]` is ignored; only `cls_config.get("cls_threshold", 0.5)` is used.
2. For **single-label** classification (`multi_label=False`), decoding is always `argmax(probs)` regardless of threshold (which is why `eb22g`'s validation sweep was flat at `0.7219` across `{0.1, 0.3, 0.5, 0.7, 0.9}` and why the sweep picker defaulted to the top of the grid, `0.9`).

### Recommended patch

Pass `metadata` (or the resolved `per_head` classification threshold) into `_decode_classification_probabilities` so multi-label tasks honour `per_head_thresholds["classification"]` when `cls_config` does not set an explicit task-level `cls_threshold`:

```python
# gliner2/inference/runtime.py
@staticmethod
def _decode_classification_probabilities(
    results: Dict,
    schema_name: str,
    cls_config: Dict,
    probs: torch.Tensor,
    default_threshold: float = 0.5,
) -> None:
    labels = cls_config["labels"]
    cls_threshold = float(cls_config.get("cls_threshold", default_threshold))
```

And in `BoundaryExtractor._decode_classifications` (`engine.py:623-631`) and `SpanExtractor._extract_classification_result` (`runtime.py:1065-1083`), pass `per_head.get("classification", threshold)` as `default_threshold`.

---

## 5. `min_field_pair_logit` in `decode_group`: `widths <= 1` vs prefix choice offset

In `decode_group` (`gliner2/models/boundary/records.py:1052-1066`), the candidate filter is:

```python
# gliner2/models/boundary/records.py, decode_group
spans_t = group.field_spans[f_idx]
widths = spans_t[:, 1] - spans_t[:, 0]
ok = torch.ones_like(widths, dtype=torch.bool)
if max_field_tokens > 0:
    ok = ok & (widths <= max_field_tokens)
if group.field_cand_logits and f_idx < len(group.field_cand_logits):
    # Prefix choice tokens are single-token literals embedded before the document
    # whose pair score is ~0 (-20); only apply min_field_pair_logit to multi-token
    # spans where the reranker's inside/length terms veto multi-sentence spans.
    ok = ok & ((widths <= 1) | (group.field_cand_logits[f_idx] >= min_field_pair_logit))
```

Using `widths <= 1` as a proxy for schema-prefix choice tokens has two edge cases:
1. Every 1-token document span bypasses `min_field_pair_logit`, even if the boundary pair scorer strongly vetoed it (`pair_logit < -10.0`).
2. A schema-prefix `choices=` literal that tokenizes into 2 or more tokens has `widths > 1`, so its low `pair_logit` is not exempted by `widths <= 1` and can be masked out before `_format_field` resolves prefix choices.

Note that `decode_group` does not currently receive `offset` (the schema prefix length in tokens, which lives in `BoundaryExtractor._decode_records` in `engine.py:1956`).

### Recommended patch

Thread `doc_offset: int = 0` from `BoundaryExtractor._decode_records` (`offset`) into `decode_group`, and exempt prefix choice candidates by their token position (`spans_t[:, 1] <= doc_offset`) rather than by `widths <= 1`:

```python
# gliner2/models/boundary/records.py, decode_group(..., doc_offset: int = 0)
if group.field_cand_logits and f_idx < len(group.field_cand_logits):
    is_prefix_choice = spans_t[:, 1] <= doc_offset
    ok = ok & (is_prefix_choice | (group.field_cand_logits[f_idx] >= min_field_pair_logit))
```

In `BoundaryExtractor._decode_records` (`engine.py:2257`), pass `doc_offset=offset` to `decode_group`.

---

## 6. `"*"` wildcard expansion: case-sensitive deduplication

`expand_entity_wildcards` (`gliner2/inference/schema.py:78-115`) expands `"*"` and `"any"` to the 125-type `DEFAULT_OPEN_VOCAB_ENTITIES` tuple (sourced from the `app:news55` taxonomy).

Why a bounded 125-type list is the right design here: models with `"open_vocab": ["entities"]` in `default_schema` were trained on >1,000 entity types across open-domain corpora (`mix_natural`, `zh_multitask`, etc.). Stuffing >1,000 entity queries or all keys of `labels/unified.yaml` into the schema prefix would overflow `max_len` and slow inference by an order of magnitude. Expanding `"*"` to the 125 general types and then passing the schema through `apply_label_map` (which both `tools/infer.py` and `viewer/backend/models.py` do) maps those 125 types to the checkpoint's canonical spelling (`Title_Snake` / `unified.yaml`).

The one bug in `expand_entity_wildcards` is that its deduplication set `seen` is case-sensitive:

```python
# gliner2/inference/schema.py
if isinstance(entities, (list, tuple)):
    out: List[Any] = []
    seen: set = set()
    for item in entities:
        if isinstance(item, str) and item.strip().lower() in ENTITY_WILDCARD_TOKENS:
            for default_ent in DEFAULT_OPEN_VOCAB_ENTITIES:
                if default_ent not in seen:
                    seen.add(default_ent)
                    out.append(default_ent)
```

Traced: `expand_entity_wildcards(["Person", "*"])` emits both `"Person"` and `"person"`. If a checkpoint has no `label_style` or `label_map`, both queries run separately.

### Recommended patch

Key `seen` (and dict wildcard expansion) by `.casefold()`, preserving the user's explicit spelling when both appear:

```python
# gliner2/inference/schema.py
if isinstance(entities, (list, tuple)):
    out: List[Any] = []
    seen: set = set()
    for item in entities:
        if isinstance(item, str) and item.strip().lower() in ENTITY_WILDCARD_TOKENS:
            for default_ent in DEFAULT_OPEN_VOCAB_ENTITIES:
                cf = default_ent.casefold()
                if cf not in seen:
                    seen.add(cf)
                    out.append(default_ent)
        else:
            key = item.casefold() if isinstance(item, str) else None
            if key is not None:
                if key not in seen:
                    seen.add(key)
                    out.append(item)
            else:
                out.append(item)
    return out
```

---

## 7. Smaller notes

**Viewer event filter (`viewer/frontend/components/ResultView.tsx`).**
The checkbox `"Only events with both triggers and arguments"` filters `result.event_extraction` using `hasTriggerAndArguments` (`triggers.length > 0 && arguments.length > 0`) and feeds the filtered `displayResult` to both the Events panel and `collectMarks`, keeping `evt-N` highlight IDs synchronized. Two minor UX details:
1. Toggling the checkbox re-indexes `evt-0, evt-1, ...`, so if `pinnedEid` was set before toggling, it can point at a different mention after toggling. Clearing `pinnedEid` in a `useEffect` when `requireBothTriggerAndArgs` changes (or keying `eid` by `${etype}:${mi}` on the unfiltered index) avoids stale pins.
2. When all mentions of an event type are filtered out by the checkbox, `filtered[etype] = []` moves that event type into the `"No matches: ..."` footer (`EmptyNote`), which is reasonable, though worth knowing it includes types that had trigger-only predictions.

**Run-logs fallback (`tools/train/format_metrics.py`).**
`_read_hub` checks `_run_logs_fallback(repo)` (`<owner>/gliner2-run-logs/<exp>`) when `not any(name in found for name in METRIC_FILES) and subfolder is None`. If a model repo has `val_metrics.json` but is missing `test_metrics.json`, `any(...)` is True and the fallback is skipped. Checking `if not all(name in found for name in METRIC_FILES)` (or checking whenever `missing` is non-empty) would also heal partial sidecar uploads. Also update the comment at line 186 to note that `push_to_hub.py` now uploads sidecar JSONs directly for new pushes, while the fallback heals older repos pushed before `bca2bff`.

---

## 8. The `0607b29` pipeline fixes, checked and left as they are

These are the seven findings from the October reviews. The code matches the specification:

- **Schema dropout stays on the training list only (`gliner2/processor.py:1431-1467`).** `_process_events` no longer applies `remove_events_prob` in the inference `dict` branch; it applies only in the training `list` branch, and `eb22g.yaml` sets `remove_events_prob: 0.0`.
- **Offset-dict gold is unwrapped in string evaluation (`gliner2/training/eval_metrics.py:914-1122`).** `_gold_event_trigger_set`, `_gold_event_argument_set`, and `_gold_oneie_sets` unwrap `{"text": ..., "start": ..., "end": ...}` to `text` before constructing the evaluation tuple sets.
- **Coreferent arguments are grouped as alternatives (`gliner2/processing/argument_coref.py:86-116`, `gliner2/processor.py:1501-1520`, `gliner2/processing/boundary_preprocessing.py:257-263`).** `expand_argument_coreferents` tags coreferent expansions with `_coref_group: arg_idx`, `SchemaTransformer` packs them as a sub-list on that role slot, and `_build_record_field_target` compiles a sub-list into alternatives of a single `RecordFieldTarget` value rather than separate required role fillers.
- **`split_positive_negative` in `candidate_pair_loss` (`gliner2/models/boundary/losses.py:451-481`).** Reduces positive and negative pair BCE over their own denominators and sums them so positive span gradients are not diluted by the negative pool size.

---

## 9. Missing test cases to add alongside the patches

Existing tests in `tests/test_infer_cli.py` and `tests/models/boundary/test_record_head_pipeline.py` cover basic stopword stripping (`"the hackers"`, `"In the northern province"`), quantifier guards (`"at least 246 people"`, `"a dozen people"`), `"New York"`, `"May 15"`, wildcard expansion (`"*"` / `"any"`), and the 40-token / `-25.0` pair-logit filter in `forward_group` + `decode_group`.

The following 6 test assertions should be added when the patches above are applied:

1. **Quantifier and particle preservation in `head_spans`:** `strip_leading_stopwords_text` leaves `"as many as 40 people"`, `"as few as 3"`, `"El Paso"`, `"La Paz"`, `"de Gaulle"`, `"da Silva"`, and `"Al Qaeda"` unchanged with `delta == 0`.
2. **Trailing-whitespace offset invariant:** On a span dict with trailing whitespace (or across any stripped span), `doc[span["start"]:span["end"]] == span["text"]` holds after `strip_leading_stopwords_from_results`.
3. **Max-confidence retention on deduplication:** When `"the Red Sea"` (`confidence: 0.41`) and `"Red Sea"` (`confidence: 0.91`) collapse to the same `(text, start, end)` key in `_dedupe_span_list`, the surviving span has `confidence == 0.91`.
4. **Single-pass stripping in `batch_extract_long`:** `batch_extract_long(..., head_spans=True)` invokes `strip_leading_stopwords_from_results` once after chunk merging and preserves `doc[start:end] == text`.
5. **Multi-label classification `per_head_thresholds`:** `_decode_classification_probabilities` with `multi_label=True` and `per_head_thresholds={"classification": 0.9}` filters out a class with probability `0.6`, while `multi_label=False` continues to emit the `argmax` class.
6. **Gold argument width-drop counter and case-insensitive wildcard dedupe:**
   - A 25-token gold argument passed to `compute_group_loss` with `max_field_tokens=24` increments `anchor_gate_stats()[f"{task_type}/arg_exceeds_max_tokens"]`.
   - `expand_entity_wildcards(["Person", "*"])` emits `"Person"` and does not also emit lowercase `"person"`.

---

## 10. Priority order for implementation

1. **Fix `head_spans` edge cases (§3):** Add `"as many as"` / `"as few as"` to `_QUANTIFIER_GUARD_RE`, add the surname/org particle and `El`/`La` proper-prefix guards, preserve max confidence in `_dedupe_span_list`, stop `.rstrip()` from desyncing `end`, and strip only once in `batch_extract_long`.
2. **Thread `doc_offset` into `decode_group` (§5):** Replace `widths <= 1` with `spans_t[:, 1] <= doc_offset` so 1-token document spans are subject to `min_field_pair_logit` and multi-token prefix choices are not dropped.
3. **Wire `per_head_thresholds["classification"]` into multi-label classification (§4)** and casefold `seen` in `expand_entity_wildcards` (§6).
4. **Add the `arg_exceeds_max_tokens` counter in `compute_group_loss` (§2)** while keeping `max_field_tokens` in `forward_group`, and add the 6 test cases in §9.
