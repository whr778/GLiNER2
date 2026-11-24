# Single-dataset event configs

One fine-tune per event corpus, warm-started from `whr778/gliner2-eb23a`, and a control from `fastino/gliner2.5-multi-v1`. The pair shares data, negative dose, and schedule. The pretrained checkpoint is the difference, except Mendeley: that corpus is multi-trigger, so the eb23a arm keeps TriggerLink weight `0.3` and the multi-v1 arm stays at `0.0` (that checkpoint has no TriggerLink).

Both checkpoints are boundary models with `max_len: 4096`. Labels are `labels/unified-full-v4.yaml`. Both arms set `loss_reduction: global`. The multi-v1 checkpoint ships `sum`, and the per-task loss report rejects `sum`. `sum` also grows the span terms with document length. Relation negatives are off: none of these corpora have relation gold, and the relneg A/B (2026-10-19) was negative. `expand_argument_coref: false` and `record_argument_coref_copy: none`. Every event corpus here has one trigger per event except Mendeley, so TriggerLink weight is `0.0` on the rest.

Doses are the train-split ontology, counted 2026-11-24. A dose larger than the pool injects the whole pool.

| Config pair | Train | Docs | Epochs | Eff. batch | Negatives | Best metric |
|---|---|---:|---:|---:|---|---|
| `wikievents-*` | `wikievents_off.train`, val `wikievents.dev` | 199 | 10 | 8 | entities 17, events 48 | `event_argument` |
| `casie-*` | `casie_off` | 798 | 8 | 16 | entities 21, events 5 | `event_argument` |
| `wojood-*` | `wojood_hadath` | 1,096 | 6 | 16 | events 1 | `event_argument` |
| `mendeley-*` | `mendeley_ed` | 1,420 | 6 | 16 | events 1 | `event_trigger` |
| `maven-*` | `maven` | 2,913 | 6 | 16 | events 168 | `event_trigger` |
| `rams-*` | `rams_merged.train`, val `rams.dev` | 6,322 | 4 | 16 | events 139 | `event_argument` |
| `cmnee-*` | `cmnee_ner_off.train`, val/test `cmnee` | 9,281 | 4 | 16 | events 8 | `event_argument` |
| `duee-*` | `duee` (no test split) | 11,603 | 4 | 16 | events 65 | `event_argument` |
| `docfee-*` | `docfee` | 16,384 | 3 | 16 | entities 30 | head macro |
| `docee-*` | `docee` | 21,842 | 3 | 16 | entities 356 | head macro |
| `chfinann-*` | `chfinann` | 25,632 | 3 | 16 | entities 23 | head macro |

CMNEE train rows carry partial LLM entity labels. `partial_annotation` withholds entity negatives. DocEE, DocFEE, and ChFinAnn are trigger-free: the event type is a classification and the roles are entities.

ACE 2005 stays in `tools/train/config/ace2005/`. Not here: `events_biotech` (classification only), the `*_ner` reframes, and the synthetic news mixes.

Score the zero-shot checkpoint on the same config before comparing a fine-tune. Pick on validation. Score the blind test once. Floor ±0.02 strict micro-F1.

```bash
uv run python tools/train/train.py --config tools/train/config/events/wikievents-eb23a.yaml
uv run python tools/train/eval.py \
  --checkpoint whr778/gliner2-eb23a \
  --config tools/train/config/events/wikievents-eb23a.yaml \
  --split val --model-schema
```
