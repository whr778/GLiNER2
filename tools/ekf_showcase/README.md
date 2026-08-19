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

## The three feeds

| feed | built by | what it is |
|---|---|---|
| synthetic | `make_demo_feed.py` | mixed-topic, time-ordered; on-topic disaster snippets interleaved with distractors. Free |
| DocEE | `make_docee_feed.py` | real DocEE articles, multi-event — the synthetic feed's only "place" is the region, so type+location association has nothing to bite on |
| real events | `build_turkey_feed.py`, `build_helene_feed.py` | archived wire copy for Turkiye-Syria 2023 and Hurricane Helene 2024 |

Real-event runs are three steps: harvest ground truth, build the feed, score the run.

```bash
uv run python tools/ekf_showcase/harvest_helene_gt.py     # -> datasets/helene2024/ground_truth.json
uv run python tools/ekf_showcase/build_helene_feed.py     # -> datasets/helene2024/_cache/feed.jsonl
uv run python tools/ekf_showcase/run_pipeline.py --feed datasets/helene2024/_cache/feed.jsonl --out run.jsonl
uv run python tools/ekf_showcase/score_helene.py --run run.jsonl
```

Scoring is kept out of `run_pipeline.py` on purpose: the pipeline's built-in `--truth`
pools truth across streams, which is exactly the assumption the real-event tests exist to
challenge.

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
