# Turkish annotation batches

A killed poller does NOT lose money -- the batch completes server-side. Recover with
`--fetch-batch <id>`; resubmitting pays twice for identical output.

| date | batch id | n | what | recover with |
|---|---|---|---|---|
| 2026-08-29 | `msgbatch_01HiLuc6HmxizZzPBvFo1WfD` | 2,469 | held-out gate eval top-up to 400/outlet (~$2) | `uv run python tools/data/annotate_gate.py --fetch-batch msgbatch_01HiLuc6HmxizZzPBvFo1WfD --out data/turkish_gate/gate_ann_tr_heldout_topup --corpora data/turkish_gate/tr_eval_topup_candidates.jsonl` |
