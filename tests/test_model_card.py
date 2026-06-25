"""Model-card generation: license engine, registry completeness, rendering."""

import glob
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "train"))
from model_card import (  # noqa: E402
    build_model_card,
    classify_license,
    load_registry,
    summarize_licenses,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "tools" / "train" / "config"


# --- license classification ------------------------------------------------

def test_classify_license_classes():
    assert classify_license("MIT").klass == "permissive"
    assert classify_license("Apache-2.0").klass == "permissive"
    assert classify_license("cc-by-4.0").klass == "permissive"
    assert classify_license("ODC-BY").klass == "permissive"
    assert classify_license("cc-by-sa-4.0").klass == "share-alike"
    assert classify_license("cc-by-nc-4.0").klass == "non-commercial"
    assert classify_license("research use (AI2)").klass == "research-only"
    assert classify_license("LDC (LDC2006T06)").klass == "research-only"


def test_unverified_is_never_upgraded():
    for raw in ("see card", "see source", "other", "NLM / NCBI", "", None):
        lic = classify_license(raw)
        assert lic.klass == "unverified" and lic.verified is False


def test_most_restrictive_wins():
    # non-commercial dominates everything
    v = summarize_licenses([("A", "MIT"), ("B", "cc-by-nc-4.0"), ("C", "see card")])
    assert v.headline == "Non-commercial"
    assert v.commercial == "Not permitted"
    assert v.all_verified is False
    # all permissive -> permissive, commercial permitted
    v2 = summarize_licenses([("A", "MIT"), ("B", "Apache-2.0")])
    assert v2.headline == "Permissive" and v2.commercial == "Permitted" and v2.all_verified


# --- registry completeness (single source of truth) ------------------------

def test_every_config_corpus_has_a_registry_entry():
    reg = load_registry()
    ds, base = reg["datasets"], reg["base_models"]
    missing = set()
    for f in glob.glob(str(CONFIG_DIR / "*.yaml")):
        cfg = yaml.safe_load(Path(f).read_text()) or {}
        data = cfg.get("data") or {}
        for c in data.get("corpora") or []:
            key = c.split("/")[-1]
            if key not in ds:
                missing.add(key)
        for name in (data.get("event_files") or {}):
            if name not in ds:
                missing.add(name)
        model = cfg.get("model") or {}
        bm = model.get("encoder") or model.get("pretrained")
        if bm and bm not in base:
            missing.add(bm)
    assert not missing, f"registry missing entries for: {sorted(missing)}"


# --- acceptance: the real mmbert-base mix is non-commercial + unverified ----

def test_mmbert_base_verdict_is_non_commercial_and_unverified():
    cfg = yaml.safe_load((CONFIG_DIR / "mmbert-base.yaml").read_text())
    reg = load_registry()
    corpora = [c.split("/")[-1] for c in cfg["data"]["corpora"]]
    named = [(reg["datasets"][k]["name"], reg["datasets"][k]["license"]) for k in corpora]
    v = summarize_licenses(named)
    assert v.headline == "Non-commercial", v.headline
    assert any("FiNER-ORD" in r for r in v.reasons["non-commercial"])
    assert len(v.reasons.get("unverified", [])) >= 3


# --- rendering -------------------------------------------------------------

def _metrics(prefix):
    return {f"eval_{prefix}_strict_micro_f1": 0.79, f"eval_{prefix}_relaxed_micro_f1": 0.86,
            f"eval_{prefix}_strict_micro_precision": 0.81, f"eval_{prefix}_relaxed_micro_precision": 0.88,
            f"eval_{prefix}_strict_micro_recall": 0.78, f"eval_{prefix}_relaxed_micro_recall": 0.85,
            f"eval_{prefix}_strict_support": 1234}


def test_build_model_card_structure_and_honesty():
    card = build_model_card(
        model_name="demo-model", base_model="jhu-clsp/mmBERT-base", cfg={"model": {}},
        config=type("C", (), {"experiment_name": "demo", "num_epochs": 3, "batch_size": 4,
                              "encoder_lr": 1e-5, "task_lr": 3e-4, "seed": 42, "bf16": True})(),
        dataset_keys=["nuner_full", "finer_ord", "ace2005"],
        results={"total_time_seconds": 3661.0},
        eval_metrics=_metrics("entity"), test_metrics={"eval_loss": 0.2, **_metrics("entity")},
        generated_at="2026-06-25",
    )
    # frontmatter never asserts a clean SPDX for a non-commercial mix
    assert card.startswith("---\n")
    assert "license: other" in card
    assert "license: apache-2.0" not in card
    # required sections
    for heading in ("# demo-model", "## Training data", "## Training procedure",
                    "## Evaluation", "## License", "Trained on", "2026-06-25"):
        assert heading in card, heading
    # honest license determination surfaces the restrictive datasets
    assert "Effective license: Non-commercial" in card
    assert "FiNER-ORD (cc-by-nc-4.0)" in card
    assert "ACE 2005" in card


def test_unknown_dataset_is_flagged_not_silently_dropped():
    card = build_model_card(
        model_name="m", base_model="jhu-clsp/mmBERT-base", cfg={"model": {}},
        config=type("C", (), {"experiment_name": "m"})(),
        dataset_keys=["nuner_full", "totally_unknown_corpus"],
        results={}, eval_metrics=None, test_metrics=None, generated_at="2026-06-25",
    )
    assert "totally_unknown_corpus" in card
    assert "UNKNOWN" in card
