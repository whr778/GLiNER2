# Evaluation Plan: Per-Language Blind-Test

Target file: `tools/train/train.py`
Reference: `tools/train/EVALUATION.md`

---

## Step 1 — Micro summary after blind-test detail [DONE]

`_print_micro_report(metrics)` in `gliner2/training/metrics.py` already produces the compact
"micro P/R/F1 strict→relaxed" table shown at the end of each training epoch.  It is called
inside `compute_metrics`, so it runs during the blind-test pass — but before `main()` prints
the detailed classification reports.

**Change:** after the detailed classification report loop in `main()`, import
`_print_micro_report` from `gliner2.training.metrics` and call it again on `test_metrics`:

```python
print("\n===== Blind test summary =====")
_print_micro_report(test_metrics)
```

No changes to `metrics.py` needed.

---

## Step 2 — Language detection during record loading [DONE]

### Dependencies

Add to `pyproject.toml` under a new `[project.optional-dependencies]` group (e.g. `train`):

```toml
train = [
    "lumi-language-id-2",
    "langcodes",
]
```

### API

```python
from lumi_language_id import detect_language   # returns (lang_2char, confidence)
import langcodes

def _to_iso3(lang2: str) -> str:
    """Convert a 2-char BCP-47 tag (or 'und') to a 3-char ISO 639-2 code."""
    return langcodes.Language.get(lang2).to_alpha3()
```

`detect_language(text)` returns `(code, probability)` where `code` is a 2-char BCP-47 tag
or `'und'` when confidence < 0.5.  `langcodes.Language.get('en').to_alpha3()` → `'eng'`,
`langcodes.Language.get('und').to_alpha3()` → `'und'`.

### Storage

Store the 3-char code directly on each record dict as `"_lang"`.  Keeping it in the record
means it survives any shuffle performed by `ExtractorDataset` without a parallel list.

### Helpers to add in `train.py`

```python
def _detect_lang(text: str) -> str:
    from lumi_language_id import detect_language as _lid
    import langcodes
    lang2, _ = _lid(text)
    return langcodes.Language.get(lang2).to_alpha3()

def _annotate_languages(records: List[Dict]) -> List[Dict]:
    """Add '_lang' field (ISO 639-2) to each record in-place; returns the list."""
    for rec in records:
        text = rec.get("input") or rec.get("text") or ""
        rec["_lang"] = _detect_lang(text) if text.strip() else "und"
    return records
```

Language detection is only triggered when `eval_by_language: true` to avoid LID startup
cost on every training run.

---

## Step 3 — `eval_by_language` config option [DONE]

In the `eval` YAML section (alongside `batch_size` / `threshold`):

```yaml
eval:
  batch_size: 8
  threshold: 0.5
  eval_by_language: false   # default
```

In `main()`:

```python
eval_by_language = eval_cfg.get("eval_by_language", False)
```

---

## Step 4 — Per-language blind-test pass [DONE]

### When `eval_by_language: false` (default)

Unchanged: call `evaluate_checkpoint(best, test_data, ...)` as today, then print detail +
summary.

### When `eval_by_language: true`

1. **Materialise test records** — if `test_data` is still a list of file paths (no label
   transforms were active), call `_read_records(test_data)` now.  Then call
   `_annotate_languages(test_data)`.

2. **Load model once** from the best checkpoint:
   ```python
   from gliner2 import GLiNER2
   from gliner2.training.trainer import ExtractorDataset
   from gliner2.training.metrics import compute_metrics, _print_micro_report
   model = GLiNER2.from_pretrained(str(best))
   ```

3. **Group by language** (alphabetical):
   ```python
   from collections import defaultdict
   by_lang = defaultdict(list)
   for rec in test_data:
       by_lang[rec.get("_lang", "und")].append(rec)
   ```

4. **Per-language loop**:
   ```python
   for lang in sorted(by_lang):
       subset = by_lang[lang]
       print(f"\n[blind test] Language: {lang}  ({len(subset)} samples)")
       ds = ExtractorDataset(subset, shuffle=False, validate=False)
       lang_metrics = compute_metrics(model, ds, batch_size=eval_bs, threshold=eval_thr)
       # print detail reports ...
       _print_micro_report(lang_metrics)
   ```

5. **Full-dataset pass** (same as current blind test):
   ```python
   print(f"\n[blind test] All languages ({len(test_data)} samples)")
   ds_all = ExtractorDataset(test_data, shuffle=False, validate=False)
   test_metrics = compute_metrics(model, ds_all, batch_size=eval_bs, threshold=eval_thr)
   # print detail reports + summary, write JSON files ...
   ```

   The JSON files (`test_metrics.json`, `best/test_metrics.json`) continue to hold only the
   aggregate metrics, preserving backward compatibility.

---

## Implementation order

1. [DONE] Add `_detect_lang` / `_annotate_languages` helpers.
2. [DONE] Read `eval_by_language` from config.
3. [DONE] Add per-language branch to the blind-test block.
4. [DONE] Add summary call (`_print_micro_report`) after the detail block in both branches.
5. [DONE] Add optional deps to `pyproject.toml`.
6. [DONE] Tests in `tests/test_eval_language.py` (30 tests, all passing).
