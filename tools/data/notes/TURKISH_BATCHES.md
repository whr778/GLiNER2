# Annotation batch ids (Turkish, Simplified Chinese)

Submitting is [TRAINING.md §3a-2](../../train/TRAINING.md); pricing is
[ANNOTATION_ECONOMICS.md](ANNOTATION_ECONOMICS.md).

A killed poller does NOT lose money -- the batch completes server-side. Recover with
`--fetch-batch <id>`; resubmitting pays twice for identical output.

| date | batch id | n | what | recover with |
|---|---|---|---|---|
| 2026-08-29 | `msgbatch_01HiLuc6HmxizZzPBvFo1WfD` | 2,469 | held-out gate eval top-up to 400/outlet (~$2) | `uv run python tools/data/annotate_gate.py --fetch-batch msgbatch_01HiLuc6HmxizZzPBvFo1WfD --out data/turkish_gate/gate_ann_tr_heldout_topup --corpora data/turkish_gate/tr_eval_topup_candidates.jsonl` |
| 2026-08-29 | `msgbatch_013jcgFy6gWHdq4G44qPPnTx` | 38,094 | field-level casualty annotation for extractor training, top 38,094 of the pool at gate cut 0.99999 (~$50.91, est. 78.8% purity -> ~30K positives) | `uv run python tools/data/annotate_casualty.py --fetch-batch msgbatch_013jcgFy6gWHdq4G44qPPnTx --corpora data/turkish_gate/cas_candidates_top38k.jsonl --out data/turkish_gate/cas_ann_tr` |
| 2026-08-31 | `msgbatch_01QwD9SeHV8Uzqxdcu7H4h2F` | 4,998 | Chinese gate-label measurement sample from shaowenchen/news_zh (~$4.56) -- measures the real positive rate and what the existing gate's 559 Chinese rows deliver as a pre-filter | `uv run python tools/data/annotate_gate.py --fetch-batch msgbatch_01QwD9SeHV8Uzqxdcu7H4h2F --out data/chinese_gate/zh_gate_sample --corpora data/chinese_gate/zh_candidates.jsonl` |
| 2026-08-31 | `msgbatch_01M7RNoD7BXyXCqtTNmu75E9` | 26,835 | Simplified Chinese FIELD-LEVEL casualty records (~$38.55): 25,060 gate-selected at cut 0.99999 (purity 59.1%) plus 1,775 already-adjudicated current_toll documents at 100% purity | `uv run python tools/data/annotate_casualty.py --fetch-batch msgbatch_01M7RNoD7BXyXCqtTNmu75E9 --corpora data/chinese_gate/zh_cas_candidates.jsonl --out data/chinese_gate/cas_ann_zh` |
| 2026-08-31 | `msgbatch_01UqfwJKi5t4eiANZPYh9Zrt` | 31,721 | Simplified Chinese field-level, tranche 2 -- EVERYTHING not already annotated, down to gate score 0.0 (~$45.57). Bought for corpus BALANCE (~21k Chinese against Turkish 31,263 / English 29,324) rather than cost-per-positive, and the low-scoring tail supplies negative supervision: a model trained only on documents that contain tolls learns to always emit one, which is the EKF's worst failure mode | `uv run python tools/data/annotate_casualty.py --fetch-batch msgbatch_01UqfwJKi5t4eiANZPYh9Zrt --corpora data/chinese_gate/zh_cas_candidates_2.jsonl --out data/chinese_gate/cas_ann_zh2` |
