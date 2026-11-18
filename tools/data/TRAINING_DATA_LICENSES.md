# Training Data: Research Use vs. Redistribution

Cross-referenced from [`TRAINING_DATA.md`](TRAINING_DATA.md), [`../train/dataset_registry.yaml`](../train/dataset_registry.yaml), [`../train/TRAINING.md`](../train/TRAINING.md) (§3a), and the license classifier in [`../train/model_card.py`](../train/model_card.py).

**"Usable for research"** and **"not redistributable"** overlap: several academic benchmarks and news-derived corpora permit internal research use (or institutionally licensed research use in the case of LDC) while prohibiting public republication of the underlying text.

---

## 1. Training Data Usable for Research Purposes

All corpora listed in `TRAINING_DATA.md` can be used for **internal / non-commercial research**, subject to holding an institutional LDC license for `ace2005`:

### Permissive Open Licenses (Research + Commercial + Redistributable with Attribution)
- `nuner_full` (`NuNER`) — `MIT` ([`numind/NuNER`](https://huggingface.co/datasets/numind/NuNER))
- `docred` (`DocRED`) — `MIT` ([`thunlp/docred`](https://huggingface.co/datasets/thunlp/docred))
- `chfinann` (`ChFinAnn`) — `MIT` in `dataset_registry.yaml` / `see source` in `TRAINING_DATA.md` ([Doc2EDAG](https://github.com/dolphin-zs/Doc2EDAG))
- `knowledgator_gliner` (`GLiNER multi-task synthetic`) — `Apache-2.0` ([`knowledgator/GLINER-multi-task-synthetic-data`](https://huggingface.co/datasets/knowledgator/GLINER-multi-task-synthetic-data))
- `biomed_ner` (`biomed_NER`) — `Apache-2.0` ([`knowledgator/biomed_NER`](https://huggingface.co/datasets/knowledgator/biomed_NER))
- `pubmed_abstracts_ner` (`PubMedAbstractsNER`) — `Apache-2.0` ([`knowledgator/PubMedAbstractsNER`](https://huggingface.co/datasets/knowledgator/PubMedAbstractsNER))
- `sentence_rex` — `Apache-2.0` ([`knowledgator/sentence_rex`](https://huggingface.co/datasets/knowledgator/sentence_rex))
- `gliclass_logic` (`GLiClass v3 logic`) — `Apache-2.0` ([`knowledgator/gliclass-v3-logic-dataset`](https://huggingface.co/datasets/knowledgator/gliclass-v3-logic-dataset))
- `gliclass_rac` (`GLiClass v2.0-RAC`) — `Apache-2.0` ([`knowledgator/gliclass-v2.0-RAC`](https://huggingface.co/datasets/knowledgator/gliclass-v2.0-RAC))
- `paraloq_json` (`json_data_extraction`) — `Apache-2.0` ([`paraloq/json_data_extraction`](https://huggingface.co/datasets/paraloq/json_data_extraction))
- `gap_coref` (`GAP`) — `Apache-2.0` ([`coref-data/gap_raw`](https://huggingface.co/datasets/coref-data/gap_raw))
- `kaznerd` (`KazNERD`) — `cc-by-4.0` ([`yeshpanovrustem/kaznerd`](https://huggingface.co/datasets/yeshpanovrustem/kaznerd))
- `docfee` (`DocFEE`) — `cc-by-4.0` ([`tongzhou21/DocFEE`](https://github.com/tongzhou21/DocFEE))
- `mendeley_ed` / `mendeley_ner` (`Mendeley-ED`) — `cc-by-4.0` ([Mendeley `7d54rvzxkr`](https://doi.org/10.17632/7d54rvzxkr.1))
- `litbank_coref` (`LitBank` via CorefUD 1.2) — `CC-BY-4.0`
- `events_biotech` (`events_classification_biotech`) — `ODC-BY` ([`knowledgator/events_classification_biotech`](https://huggingface.co/datasets/knowledgator/events_classification_biotech))
- `masakhaner` (`MasakhaNER 2.0` + per-language `masakhaner_*`) — `afl-3.0` ([`masakhane/masakhaner2`](https://huggingface.co/datasets/masakhane/masakhaner2))
- `masakhanews` (`MasakhaNEWS` + per-language `masakhanews_*`) — `afl-3.0` ([`masakhane/masakhanews`](https://huggingface.co/datasets/masakhane/masakhanews))
- `biored` (`BioRED`) — `NLM / NCBI` ([NCBI BioRED](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/))
- Purely synthetic corpora generated in-repo: `casualty_ft`, `casualty_multi`, `synthetic_haiku45_5k`, `synthetic_haiku45_5k_coerced`, `synthetic_sonnet5_1k`, `synthetic_events_capped`, `lg_onecall`, `lg_twostage` — `generated here`

### Share-Alike & Non-Commercial Open Licenses (Research Permitted)
- `stockmark_jpn` (`stockmark-jpn`) — `cc-by-sa-3.0` ([`stockmark/ner-wikipedia-dataset`](https://huggingface.co/datasets/stockmark/ner-wikipedia-dataset))
- `klue_ner` (`KLUE-NER`) — `cc-by-sa-4.0` ([KLUE GitHub](https://github.com/KLUE-benchmark/KLUE))
- `klue_re` (`KLUE-RE`) — `cc-by-sa-4.0` ([KLUE GitHub](https://github.com/KLUE-benchmark/KLUE))
- `finer_ord` (`FiNER-ORD`) — `cc-by-nc-4.0` (research / non-commercial permitted; commercial prohibited; [`gtfintechlab/finer-ord`](https://huggingface.co/datasets/gtfintechlab/finer-ord))
- `gum_coref` (`GUM` via CorefUD 1.2) — per-genre `CC-BY` / `CC-BY-SA` / `CC-BY-NC-SA` (4 genres unverified; annotations `CC-BY-4.0`)

### Research-Only / Licensed / Publisher-Text Caches (Research Permitted, NOT Redistributable)
- `ace2005` (`ace2005_eng`, `ace2005_ara`, `ace2005_zho`, `ace2005_off`) — `LDC (LDC2006T06)` (**usable for research only by organizations/seats holding the LDC2006T06 license**; [LDC2006T06](https://catalog.ldc.upenn.edu/LDC2006T06))
- `scierc` (`SciERC`) — `research use (AI2)` ([AI2 SciIE](http://nlp.cs.washington.edu/sciIE/))
- `docee` & `docee_zh` (`DocEE`) — academic benchmark over third-party publisher-copyright news text ([`tongmeihan1995/docee`](https://github.com/tongmeihan1995/docee))
- `turkish_event` (`turkish_event_type`, `turkish_pool_general`) — LLM-annotated Turkish news ([`umutertugrul/turkish-news-1.8M-tokenized`](https://huggingface.co/datasets/umutertugrul/turkish-news-1.8M-tokenized))
- `cc_news_haiku45`, `cc_news_events_haiku45`, `cc_news_events_sonnet55_v2` (and `*_coref_llm`) — private research caches over [`vblagoje/cc_news`](https://huggingface.co/datasets/vblagoje/cc_news)
- `gate2` (`gate2_tr`, `gate3`, `gate3_eb22g`) & `casualty_docee`, `casualty_events`, `casualty_ml` (`casualty_ml_eb22g`, `casualty_full_ml`) — real-news casualty/gate corpora derived from `docee`, `cc_news`, `duee`, and Turkish/Chinese news

### Public Academic / Hugging Face Benchmarks with Unspecified Redistribution Terms (Research Permitted Under Upstream Release Terms)
- **NER**: `pile_ner_def` (`Universal-NER/Pile-NER-definition`), `gliner_multilingual` (`knowledgator/gliner-multilingual-synthetic`), `bc4chemd` (`chintagunta85/bc4chemd`), `bc5cdr` (`tner/bc5cdr`), `wikiann` (`unimelb-nlp/wikiann`, streamed)
- **MTL-Bioinformatics-2016** (`anatem`, `bc2gm`, `bionlp09`, `bionlp11epi`, `bionlp11id`, `bionlp13cg`, `bionlp13ge`, `bionlp13pc`, `craft`, `ex_ptm`, `jnlpba`, `ncbi_disease`, `linnaeus`) — wrapper repository is `cc-by-4.0`, while `TRAINING.md` §3a and `dataset_registry.yaml` keep individual shared-task sets as `see source` / research-use
- **Relations / Classification / Structured**: `bio_ner_relations`, `redocred` (`tonytan48/Re-DocRED`), `professorbob_re` (`ProfessorBob/relation_extraction`), `scientific_text` (`knowledgator/Scientific-text-classification`), `text2json` (`knowledgator/text2json-training-data`)
- **Events & Derived Event Views**: `wikievents` (`wikievents_off`, `wikievents_coref_llm`), `rams` (`rams_merged`), `maven` (`maven_ner`), `casie` (`casie_off`), `cmnee` (`cmnee_ner`, `cmnee_ner_off`), `duee` (`duee_ner`)

---

## 2. Not Redistributable (Do Not Republish Publicly)

### A. Explicitly Non-Redistributable (Restricted by LDC, Research-Only Terms, or Publisher Copyright)
- `ace2005` (`ace2005_eng`, `ace2005_ara`, `ace2005_zho`, `ace2005_off`) — `LDC (LDC2006T06)`; explicitly barred from redistribution and kept off Hugging Face entirely (`TRAINING.md` §3a).
- `scierc` (`SciERC`) — `research use (AI2)`; explicitly marked non-redistributable in `TRAINING.md` §3a (any Hub mirror must remain private for internal recovery only).
- `turkish_event` (plus `turkish_event_type`, `turkish_event_type_pilots`, `turkish_pool_general`) — `not redistributable`; underlying news text remains publisher copyright.
- `cc_news_haiku45`, `cc_news_events_haiku45`, `cc_news_events_sonnet55_v2` (and `*_coref_llm`) — `unknown` upstream card license + publisher-copyright news text; documented in `TRAINING_DATA.md` as *"a private research cache and not redistributable"*.
- `docee` & `docee_zh` (`DocEE`) — `none declared (see source; not redistributable)` in `dataset_registry.yaml`; underlying news text is third-party publisher copyright.
- `gate2` (`gate2_tr`, `gate3`, `gate3_eb22g`) — `none declared (derived; not redistributable)`; inherits `docee`, `cc_news`, `duee`, and Turkish news text.
- `casualty_docee`, `casualty_events`, `casualty_loc_muted`, `casualty_loc_split`, `casualty_loc_probe_focal_last`, `casualty_multi_loc`, `casualty_natural`, `casualty_anchorless`, `casualty_ml` (`casualty_ml_eb22g`, `casualty_full_ml`, `tr_dose*`, `loc_control`) — inherit `docee`, `cc_news`, and Turkish/Chinese publisher news text (`TRAINING.md` §3a).
- `zh_multitask` & `zh_multitask_noreplay` — inherit `shaowenchen/news_zh` publisher news text.
- `bionlp09`, `bionlp11epi`, `bionlp11id`, `bionlp13cg`, `bionlp13ge`, `bionlp13pc`, `anatem`, `ex_ptm` — explicitly flagged in `TRAINING.md` §3a alongside `ace2005` and `casualty_full_ml` as *"none of them may be REDISTRIBUTED"* due to upstream shared-task source restrictions.
- Mixed replay / warm-start slices (`replay_137k30`, `replay_eb21_*`, `replay_eb22g_*`, `warmstart_mix`, `mix_natural`, `mix_anchorless`) — inherit the most restrictive non-redistributable corpus in their pool (`docee`, `cc_news_*`, `turkish_event`, `scierc`).

### B. Unverified License — Treat as Non-Redistributable Until Verified Upstream (`TRAINING_DATA.md` Footnote `‡`)
Per `TRAINING_DATA.md` (`‡` *"verify before redistribution"*) and `model_card.py` (never upgrade an unverified license to a named SPDX license):
- **Hugging Face `"see card"` / `"other"` (no explicit SPDX license declared on card)**:
  - `pile_ner_def` (and `replay_pile30`), `gliner_multilingual`, `bc4chemd`, `bc5cdr` (`other`), `wikiann`, `bio_ner_relations`, `redocred`, `professorbob_re`, `scientific_text`, `text2json`
- **Academic manual/scripted downloads (`"see source"`)**:
  - `wikievents` (`wikievents_off`, `wikievents_coref_llm`), `rams` (`rams_merged`), `maven` (`maven_ner`), `casie` (`casie_off`, `casie_typed`), `cmnee` (`cmnee_ner`, `cmnee_ner_off`, `cmnee_typed`, `cmnee_roles_ner`), `duee` (`duee_ner`, `duee_typed`), and the remaining MTL-Bioinformatics-2016 corpora (`bc2gm`, `craft`, `jnlpba`, `ncbi_disease`, `linnaeus`, recorded as `see source` in `dataset_registry.yaml`)
- **Partially unverified per-genre corpora**:
  - `gum_coref` (`conversation`, `speech`, `textbook`, `vlog` genres carry `UNVERIFIED` text provenance in `run_all_converters.sh`)
