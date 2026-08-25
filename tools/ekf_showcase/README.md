# EKF showcase

A news feed in, a tracked casualty timeline out. This is the runnable half of Paper 1
(`EKF_MHT_DESIGN.md`); the scripts here built the results it reports, and the build
detail for them is in `EKF_MHT_BUILD_RECORD.md`.

The seam the whole directory turns on is one record — the **observation**
`(t, role, value, qualifier, source, event_key)`. Everything upstream produces one,
everything downstream consumes one, and every hard problem (attachment, scope,
cross-event contamination) is a failure to build a correct one. See
`../events_working_papers/PIPELINES.md` §2 for the as-built stage map.

## Quickstart

```bash
uv run python tools/ekf_showcase/run_pipeline.py --feed datasets/ekf_showcase/feed.jsonl
```

`run_pipeline.py` is the whole line in one pass, and each of its four stages is swappable,
so the demo doubles as an ablation:

| stage | what it does | key flag |
|---|---|---|
| 0. gate | is this article a mass-casualty report? | `--gate-model`, `--gate-threshold` |
| 1. event | event type + "Casualties and Losses" spans | `--event-model`, `--event-threshold` |
| 2. extract | bind numbers to roles `{dead, injured, missing}` | `--casualty-model` |
| 3. normalize | span to `(value, qualifier, source)` | `--normalizer {heuristic,classify,hybrid,both}` |
| 4. track | EKF plus a `last_value` baseline | `--associate`, `--record-mode` |

`--help` is the authority on flags; it carries the full rationale.

## The four feeds

| feed | built by | what it is |
|---|---|---|
| synthetic | `make_demo_feed.py` | mixed-topic, time-ordered; on-topic disaster snippets interleaved with distractors. Free |
| DocEE | `make_docee_feed.py` | real DocEE articles, multi-event — the synthetic feed's only "place" is the region, so type+location association has nothing to bite on |
| real events | `build_turkey_feed.py`, `build_helene_feed.py`, `build_aegean_feed.py` | archived wire copy for Turkiye-Syria 2023, Hurricane Helene 2024, and the 2020 Aegean earthquake |

The Aegean feed is the only one with genuinely independent sources on both sides: ground
truth from the Wikipedia REVISION HISTORY (55 timestamped points, Izmir 12 -> 116 including
a real downward reclassification), documents from Hurriyet Daily News and Daily Sabah. The
Turkiye feed reads truth from the same sentence the extractor reads, which is why
`est_last_value` scores 0.000 there by construction.

Real-event runs are three steps: harvest ground truth, build the feed, score the run.

```bash
uv run python tools/ekf_showcase/harvest_helene_gt.py     # -> datasets/helene2024/ground_truth.json
uv run python tools/ekf_showcase/build_helene_feed.py     # -> datasets/helene2024/_cache/feed.jsonl
uv run python tools/ekf_showcase/run_pipeline.py --feed datasets/helene2024/_cache/feed.jsonl --out run.jsonl
uv run python tools/ekf_showcase/score_helene.py --run run.jsonl
```

Scoring is kept out of `run_pipeline.py` on purpose: the pipeline's built-in `--truth`
pools truth across streams, which is exactly the assumption the real-event tests exist to
challenge. It also expects JSONL, not the `{"points": [...]}` shape the harvesters emit —
pass `--truth` only to the scorers.

**What ships for association, as of 2026-08-25: `scope_gate.hmm_gate`**, a global decode
over {own, aggregate, reject} at σ=0.3, reject_cost=4.0, stay=0.1, **warmup=0**. It beats
the ratio gate on every event (Helene −29.4%, Türkiye −7.6%, Aegean −78.8%) and replaces
the date gate and the scope gate as separate stages. `GATES.md` in the working papers has
the full flow and every knob's default.

## Script index

**Pipeline and feeds**

| script | what it does |
|---|---|
| `run_pipeline.py` | the four-stage pipeline; feed to tracked timeline |
| `make_demo_feed.py` | synthetic mixed-topic feed |
| `make_docee_feed.py` | real multi-event feed from DocEE |
| `build_turkey_feed.py` | Turkiye-Syria 2023, one document per day, 6-21 Feb |
| `build_helene_feed.py` | Hurricane Helene 2024 from archived AP wire copy |
| `harvest_turkey_gt.py` | death-toll trajectory from archived pages |
| `harvest_helene_gt.py` | per-state Helene tolls from archived Wikipedia |
| `build_aegean_feed.py` | 2020 Aegean earthquake from archived Turkish English-language wire copy |
| `harvest_aegean_gt.py` | Aegean trajectory from the Wikipedia revision history |
| `score_turkey.py` | score tracked streams against the sourced trajectory |
| `score_helene.py` | score per-state streams; aggregate-vs-parts aware |

**Probes and experiments** — each answers one question, and each was written to be able to
kill an idea before it got built.

| script | the question it answers | paper § |
|---|---|---|
| `benchmark_gate.py` | how does the gate do on *real* disaster messages, not hand-written distractors? | §15 |
| `revision_test.py` | does the EKF absorb a genuine downward revision better than repeating the last reading? | §24 |
| `vector_state_test.py` | does a national total constrain the per-state estimates, or is it just noise? | §23 |
| `scope_gate_test.py` | can a magnitude gate tell a state's own toll from a larger-scope number filed under it? | §25 |
| `energy_probe.py` | do type/trigger energies separate the two measured extraction error classes? | §27.1 |
| `event_binding_probe.py` | which storm does this number belong to — the cross-event half, measured three ways | §27.2 |
| `framing_experiment.py` | why does record extraction depend on how the input is framed? | §17 |
| `bullet_premise_test.py` | does restructuring text into self-contained bullets fix number-to-place attachment? | §26 |

**The association rebuild (2026-08-25)** — five mechanisms measured against every event.
One ships; the rest are kept because the negatives localize why.

| script | the question it answers | verdict |
|---|---|---|
| `reject_headroom.py` | what is the reject option actually rejecting? | 63% STALE-LOW, not cross-scope-high — the gate is one-sided |
| `viterbi_gate_sweep.py` | does a GLOBAL decode beat the greedy gate? | **yes, both events** — Helene −29.4%, Türkiye −9.8% |
| `robust_filter_sweep.py` | does a Student-t model retire the hard thresholds? | no — 1-for-2, retires nothing |
| `imm_gate_sweep.py` | does SOFT (PDA) association beat hard? | no — soft cannot remove, and nested hypotheses double-count |
| `revision_state_test.py` | does a 4th state for downward revision earn its place? | correct but inert — reports never follow a revision down |
| `two_sided_gate_sweep.py` | should the ratio gate reject downward too? | 1-for-2, superseded by the decode |
| `gate56_composition.py` | should the date and scope gates merge? | complementary, but OR costs +10 false rejects for +1 catch |
| `hmm_collapse_test.py` | can one emission absorb date + scope + page furniture? | yes — false rejections 19.8% → 9.9% |
| `turkey_collapse_check.py` | does the collapse hold on Türkiye? | features never fire there; safe, unproven |
| `aegean_collapse_test.py` | the pre-registered third-event test | decode −78.8%; **prediction 1 falsified** |
| `span_giou_headroom.py` | is a GIoU-shaped span target worth a training run? | no — 11× blast radius for a 6% signal |
| `scope_label_probe.py` | is Haiku good enough to label the `scope` field? | not cleared (kappa 0.121) — and the SCHEME is the defect |
| `gate_turkish_fp.py` | does the relevance gate work on non-English? | **no** — 199/200 false admits on the shipped model |
| `gate_translation_ablation.py` | would translating at ingest fix that? | no — it is the label descriptions |

Section numbers below refer to the ORIGINAL working-paper numbering, which is preserved
in `../events_working_papers/EKF_MHT_BUILD_RECORD.md`; the paper itself was renumbered on
2026-08-19. **The mapping is by
topic, not by citation** — only `vector_state_test.py` is named in that document (§23). If
you renumber its sections, this table does not move with them.

## Data

Feeds and ground truth live under the committable `datasets/` tree, not the git-ignored
`data/` — see `EKF_MHT_DESIGN.md` §13. The models the pipeline defaults to
(`whr778/gliner2-base-v1-casualty-docee` and friends) are on the Hub; the viewer's
`backend/models.json` carries the full list if you want to swap one in.

### On the Hub

Everything here is mirrored to **private** Hub dataset repos (needs `HF_TOKEN`).
Real-event evaluations are **split into two repos each** — derived artifacts separate from
archived source pages, so the derived half can be shared without the news copy riding along
on the same visibility toggle.

| local | Hub repo | contents |
|---|---|---|
| `datasets/disaster_streams*` | `whr778/ekf-disaster-streams[-sonnet5\|-docee\|-docee250\|-scaled\|-scaled250\|-hard\|-model\|-model-ft]` | synthetic streams, all variants |
| `datasets/casualty_multi_muted` | `whr778/ekf-casualty-multi-muted` | muted-interference control |
| `datasets/ekf_showcase` | `whr778/ekf-showcase-feeds` | the feeds and tracker outputs |
| `datasets/rams_baseword` | `whr778/rams-baseword` | RAMS lemma variants + duplicate control |
| `datasets/venezuela_2026` | `whr778/ekf-venezuela-2026` | the blind holdout; text-free by design |
| `datasets/helene2024` | `whr778/ekf-helene2024` | ground truth, rollup, tracker output — **article text removed** |
| `datasets/helene2024/_cache` | `whr778/ekf-helene2024-raw` | archived source pages |
| `datasets/turkey2023` | `whr778/ekf-turkey2023` | pre-registration, results, ground truth, 12 tracker arms — **article text removed** |
| `datasets/turkey2023/_cache` | `whr778/ekf-turkey2023-raw` | archived source pages |

The `-raw` repos hold third-party news pages retrieved from the Wayback Machine, kept so the
measurements reproduce. Copyright stays with the publishers; they are a private research
cache, not for redistribution. The derived repos carry `text_chars` where a `text` field was
removed, so the omission is explicit rather than silent.

Training corpora (`data/*.jsonl`) are separate and **fetch themselves**: the registry entry
carries `hf_jsonl: whr778/<corpus>` and `tools/train/train.py::_fetch_if_missing` pulls any
missing split by basename. Wired for `casualty_loc_split`, `casualty_docee`,
`casualty_multi_loc`, `casualty_multi`, `casualty_ft`, `casualty_natural` and
`casualty_anchorless` — so a fresh GPU box needs `HF_TOKEN` and nothing else.

## Related

- `../events_working_papers/EKF_MHT_DESIGN.md` — the paper: results and what they mean
- `../events_working_papers/EKF_MHT_BUILD_RECORD.md` — the design, decisions and build detail
- `../events_working_papers/PIPELINES.md` §2 — the as-built stage map
- `../events_working_papers/MODEL_LINEAGE.md` — which config trained the casualty models
- `../../viewer/` — the same pipeline behind a UI, via the EKF panel
