# casualty_multi_muted — negative documents for cross-event contamination

Multi-event casualty documents where a fraction of the interference snippets keep their
**text** but lose their **record**, so their figures become negatives for the same queries.
Built for [[TODO]] item 2; the failure it targets is 4.7% cross-event contamination, where
a figure belonging to another storm is bound to the focal event.

| file | mute prob | documents | instances |
|---|--:|--:|--:|
| `train.jsonl` | **0.35** | 28,951 | 55,593 |
| `train.control.jsonl` | 0.0 | 29,155 | 70,213 |
| `val.jsonl` | 0.0 | 1,301 | 3,128 |
| `test.jsonl` | 0.0 | 1,028 | 2,496 |

At 0.35, **11,947 documents (41.3%) carry a muted snippet and 21,377 figures are delivered
unlabelled**. The 204 documents the muted arm lacks are focal-collision cases where every
interference record was also muted; they are dropped rather than emitted with empty gold,
because the focal toll is real but ambiguous and empty gold would teach suppression of a
genuine figure.

## Use `train.control.jsonl` as the control, NOT `data/casualty_multi.train.jsonl`

The shipped `data/casualty_multi.train.jsonl` **is not a valid control for this arm.** It
carries the same document text but **no `record_metadata`** — it predates `--record-mode`,
so `compile_record_specs` returns nothing and the record head is never supervised. Its
document count also differs (29,030 against 29,155), so other builder changes landed after
it too. An A/B against it would be measuring the builder, not the muting.

`train.control.jsonl` is built by the *current* builder at `--mute-interference-prob 0.0`,
which consumes no randomness for muting, so the two arms hold **byte-identical documents**
and differ only in which records are withheld. Verified: every muted document's text is
present in the control.

Val and test are built at 0.0 deliberately. A muted val is not comparable with the control
arm or with any historical number.

## The caveat that bounds what a gain here means

The focal snippet is always `parts[0]`, so muting is learnable from **position** —
"extract from the first paragraph" scores perfectly on this corpus without representing
event identity at all. Real articles do lead with their focal event, so the prior is not
pure artifact, but the corpus cannot distinguish the shortcut from the intended behaviour.
**Run the held-out probe with the focal placed last before reading any gain as event
identity.** Without it this arm cannot speak to Bosnia's 16, which is the case that
motivated it.

## Rebuild

```bash
uv run python datasets/disaster_streams/build_multievent_corpus.py \
  --data datasets/disaster_streams_sonnet5 --split train \
  --out datasets/casualty_multi_muted/train.jsonl \
  --max-interference 3 --record-mode natural --mute-interference-prob 0.35

# control and the held-out splits: prob 0.0
uv run python datasets/disaster_streams/build_multievent_corpus.py \
  --data datasets/disaster_streams_sonnet5 --split train \
  --out datasets/casualty_multi_muted/train.control.jsonl \
  --max-interference 3 --record-mode natural --mute-interference-prob 0.0
```

Repeat with `--split val` / `--split test` at prob 0.0. Train streams only — the showcase
feeds come from `test`, and that separation is what keeps the evaluation uncontaminated.
