---
license: other
language:
- en
task_categories:
- token-classification
tags:
- information-extraction
- structured-extraction
- evaluation
- synthetic
size_categories:
- 1K<n<10K
---

# casualty_loc_probe_focal_last

**A control, not training data.** Test split only. Never put it in a training mixture.

The `whr778/casualty_loc_muted` arm teaches a model to report the figures of the event a
document leads with and leave the others alone. But the focal snippet in that corpus is
always **first**, so the behaviour is learnable from position alone -- "extract from the
first paragraph" scores perfectly without representing event identity at all.

This corpus separates the two. It is the same test streams rebuilt with the focal snippet at
the **end** of each document, and nothing else changed.

| | |
|---|--:|
| documents | 2,873 |
| instances | 7,058 |
| paired with | `casualty_loc_split` / `casualty_loc_muted` test split |

It pairs **document for document** with the ordinary test split: same documents, same
instances, identical gold, only the snippet order differs. A model that learned to read the
first paragraph scores near zero here; a model that learned event identity does not.

## Score the 2,168 documents that actually reordered

705 of the 2,873 are single-snippet (k = 0) and are byte-identical to the ordinary test
split, because there is nothing to reorder. They carry no position signal, so including them
dilutes the contrast. Score the remaining 2,168.

## Rebuild

```bash
uv run python datasets/disaster_streams/build_multievent_corpus.py \
  --data datasets/disaster_streams_docee250 --split train \
  --contexts datasets/disaster_streams/contexts250.json \
  --streams-file datasets/disaster_streams_docee250/splits/test.streams.txt \
  --out data/casualty_loc_probe_focal_last.test.jsonl \
  --max-interference 3 --record-mode natural --mute-interference-prob 0.0 \
  --focal-position last
```
