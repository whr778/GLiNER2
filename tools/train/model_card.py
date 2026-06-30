"""Generate a MODEL_CARD.md for a trained GLiNER2 model.

``build_model_card(...)`` is a pure function: it takes the resolved training
inputs (config, hyperparameters, the datasets actually used, and the metrics)
and returns a Markdown string. The train.py hook gathers the inputs and writes
the result to ``<output_dir>/best/MODEL_CARD.md``.

License handling is deliberately conservative. Dataset license strings are
copied verbatim from ``dataset_registry.yaml`` and classified; the effective
license is the MOST RESTRICTIVE constraint across every dataset and the base
model. An unverified license ("see card" / "see source" / "other" / anything
unrecognized) is NEVER upgraded to a named license -- it stays unverified and is
flagged. A model card that asserts the wrong license is worse than none.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REGISTRY_PATH = Path(__file__).resolve().parent / "dataset_registry.yaml"

CATEGORIES = (
    "entity", "relation", "classification",
    "event_type", "event_trigger", "event_argument", "event",
)


def load_registry(path: Path = REGISTRY_PATH) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------
# License engine
# ---------------------------------------------------------------------------

@dataclass
class License:
    raw: str                 # verbatim license string
    klass: str               # permissive | share-alike | non-commercial | research-only | unverified
    commercial: Optional[bool]   # True / False / None(unknown)
    share_alike: bool
    verified: bool


def classify_license(raw: Optional[str]) -> License:
    """Classify a verbatim license string. Never upgrades unknown to named."""
    s = (raw or "").strip()
    low = s.lower()
    parts = low.replace("/", "-").replace(" ", "-").split("-")

    if "nc" in parts or "noncommercial" in low or "non-commercial" in low:
        return License(s or "unknown", "non-commercial", False, False, True)
    if "research" in low or "ldc" in low:
        return License(s or "unknown", "research-only", False, False, True)
    if not s or low.startswith("see ") or low in {"other", "unknown"}:
        return License(s or "unspecified", "unverified", None, False, False)
    if "sa" in parts or "share" in low:
        return License(s, "share-alike", True, True, True)
    permissive = (
        low in {"mit", "bsd", "cc0", "odc-by", "isc", "unlicense", "zlib"}
        or low.startswith("apache")
        or low.startswith("cc-by-4")
        or low.startswith("cc-by-3")
        or low == "cc-by"
        or low.startswith("bsd-")
    )
    if permissive:
        return License(s, "permissive", True, False, True)
    # Unrecognized named terms (e.g. "NLM / NCBI"): do not guess.
    return License(s, "unverified", None, False, False)


@dataclass
class LicenseVerdict:
    headline: str
    commercial: str          # "Permitted" | "Not permitted" | "Unverified"
    share_alike: bool
    all_verified: bool
    reasons: Dict[str, List[str]]   # klass -> ["<name> (<license>)", ...]


def summarize_licenses(named: List[tuple]) -> LicenseVerdict:
    """``named`` is a list of (display_name, License). Returns the verdict."""
    lic_by = [(n, classify_license(l) if isinstance(l, str) else l) for n, l in named]
    reasons: Dict[str, List[str]] = {}
    for name, lic in lic_by:
        reasons.setdefault(lic.klass, []).append(f"{name} ({lic.raw})")

    any_nc = "non-commercial" in reasons
    any_research = "research-only" in reasons
    any_unverified = "unverified" in reasons
    any_sa = "share-alike" in reasons

    if any_nc or any_research:
        commercial = "Not permitted"
    elif any_unverified:
        commercial = "Unverified"
    else:
        commercial = "Permitted"

    if any_nc:
        headline = "Non-commercial"
    elif any_research:
        headline = "Research use only"
    elif any_unverified:
        headline = "Unverified — review required"
    elif any_sa:
        headline = "Share-alike (copyleft)"
    else:
        headline = "Permissive"
    if any_unverified and headline not in ("Non-commercial", "Research use only", "Unverified — review required"):
        headline += " (with unverified datasets)"

    return LicenseVerdict(headline, commercial, any_sa, not any_unverified, reasons)


# ---------------------------------------------------------------------------
# Small rendering helpers
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: Optional[float]) -> str:
    if not seconds or seconds <= 0:
        return "—"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def _resolve_datasets(registry: Dict[str, Any], keys: List[str]) -> List[tuple]:
    """Return [(key, entry_or_None)] preserving order, deduped."""
    ds = registry.get("datasets") or {}
    seen, out = set(), []
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        out.append((k, ds.get(k)))
    return out


def _metric(metrics: Dict[str, Any], cat: str, regime: str, m: str) -> Optional[float]:
    return metrics.get(f"eval_{cat}_{regime}_micro_{m}")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _frontmatter(model_name, base_model, lang_codes, hf_ids, verdict) -> str:
    lines = ["---", "library_name: gliner2", "license: other",
             f"license_name: {verdict.headline.lower()}",
             "license_link: \"#license\""]
    if base_model:
        lines.append(f"base_model: {base_model}")
    if lang_codes:
        lines.append("language:")
        lines += [f"  - {c}" for c in lang_codes]
    if hf_ids:
        lines.append("datasets:")
        lines += [f"  - {d}" for d in hf_ids]
    lines += [
        "tags:",
        "  - gliner2",
        "  - information-extraction",
        "  - named-entity-recognition",
        "  - relation-extraction",
        "  - event-extraction",
        "  - text-classification",
        "metrics:",
        "  - f1",
        "  - precision",
        "  - recall",
        "pipeline_tag: token-classification",
        "---",
    ]
    return "\n".join(lines)


def _data_section(datasets: List[tuple], counts: Optional[Dict[str, Dict[str, int]]] = None) -> str:
    has_counts = bool(counts)
    n = len(datasets)

    summary = f"**{n}** dataset{'s' if n != 1 else ''} used for this run."
    if has_counts:
        total_train = sum(c.get("train", 0) for c in counts.values())
        total_val = sum(c.get("val", 0) for c in counts.values())
        total_test = sum(c.get("test", 0) for c in counts.values())
        if total_train:
            extras = []
            if total_val:
                extras.append(f"val: {total_val:,}")
            if total_test:
                extras.append(f"test: {total_test:,}")
            detail = f"{total_train:,} training records"
            if extras:
                detail += f" ({', '.join(extras)})"
            summary += f" {detail}."

    if has_counts:
        header = ("| Dataset | Task(s) | Train | Val | Test | Language | License | Source |",
                  "|---|---|--:|--:|--:|---|---|---|")
    else:
        header = ("| Dataset | Task(s) | Language | License | Source |",
                  "|---|---|---|---|---|")

    rows = ["## Training data", "", summary, ""] + list(header)

    for key, entry in datasets:
        if entry is None:
            if has_counts:
                rows.append(f"| ⚠️ `{key}` | unknown | — | — | — | — | **UNKNOWN — not in registry** | — |")
            else:
                rows.append(f"| ⚠️ `{key}` | unknown | — | **UNKNOWN — not in registry** | — |")
            continue
        lang = ", ".join(entry.get("language") or []) or "—"
        src = entry.get("source_url")
        src_md = f"[link]({src})" if src else "—"
        name = entry.get("name", key)
        tasks = entry.get("tasks", "—")
        lic = entry.get("license", "—")
        if has_counts:
            c = counts.get(key, {})
            tr = f"{c['train']:,}" if c.get("train") else "—"
            vl = f"{c['val']:,}" if c.get("val") else "—"
            te = f"{c['test']:,}" if c.get("test") else "—"
            rows.append(f"| {name} | {tasks} | {tr} | {vl} | {te} | {lang} | {lic} | {src_md} |")
        else:
            rows.append(f"| {name} | {tasks} | {lang} | {lic} | {src_md} |")

    notes = [
        (entry.get("name", key), entry["description"])
        for key, entry in datasets
        if entry and entry.get("description")
    ]
    if notes:
        rows += ["", "**Dataset notes**", ""]
        for name, desc in notes:
            rows.append(f"- **{name}** — {desc}")

    return "\n".join(rows)


def _license_section(verdict: LicenseVerdict, base_name, base_license) -> str:
    order = ["non-commercial", "research-only", "unverified", "share-alike", "permissive"]
    titles = {
        "non-commercial": "Non-commercial (no commercial use)",
        "research-only": "Research use only",
        "unverified": "Unverified — verify the upstream terms before redistribution",
        "share-alike": "Share-alike / copyleft (derivatives must keep a compatible license)",
        "permissive": "Permissive",
    }
    out = ["## License", "",
           f"**Effective license: {verdict.headline}.** This model is a derivative of "
           "its base model and every training dataset, so the most restrictive term "
           "across all of them governs the whole model.", "",
           f"- **Commercial use:** {verdict.commercial}",
           f"- **Share-alike obligation:** {'Yes' if verdict.share_alike else 'No'}",
           f"- **All licenses verified:** {'Yes' if verdict.all_verified else 'No'}",
           f"- **Base model:** {base_name} — {base_license}", ""]
    for klass in order:
        names = verdict.reasons.get(klass)
        if not names:
            continue
        out.append(f"**{titles[klass]}**")
        out += [f"- {n}" for n in sorted(names)]
        out.append("")
    out.append(
        "> License strings are copied verbatim from each dataset's card/source and "
        "from `tools/train/dataset_registry.yaml`. \"see card\"/\"see source\"/\"other\" "
        "mean the upstream declares no clear license — treat as unverified. This summary "
        "is informational, not legal advice; confirm terms before redistribution or "
        "commercial use."
    )
    return "\n".join(out)


def _metrics_table(metrics: Dict[str, Any], title: str) -> str:
    present = [c for c in CATEGORIES if f"eval_{c}_strict_micro_f1" in metrics]
    if not present:
        return ""
    rows = [f"### {title}", "",
            "Micro precision / recall / F1, strict → relaxed.", "",
            "| Category | Precision | Recall | F1 | Support |",
            "|---|--:|--:|--:|--:|"]
    for c in present:
        def cell(m):
            s = _metric(metrics, c, "strict", m)
            r = _metric(metrics, c, "relaxed", m)
            if s is None:
                return "—"
            return f"{s:.3f} → {r:.3f}" if r is not None else f"{s:.3f}"
        sup = metrics.get(f"eval_{c}_strict_support", "—")
        rows.append(f"| {c} | {cell('precision')} | {cell('recall')} | {cell('f1')} | {sup} |")
    return "\n".join(rows)


def _training_section(config, cfg, results, generated_at) -> str:
    m = cfg.get("model") or {}
    arch = []
    for k in ("max_width", "max_len", "struct_loss", "struct_pos_weight",
              "focal_gamma", "focal_alpha"):
        if k in m:
            arch.append(f"`{k}={m[k]}`")
    rt = (results or {}).get("total_time_seconds")
    sps = (results or {}).get("samples_per_second")
    rows = ["## Training procedure", "",
            "| Setting | Value |", "|---|---|",
            f"| Trained on | {generated_at} |",
            f"| Duration | {_fmt_duration(rt)} |"]
    if sps:
        rows.append(f"| Throughput | {sps:.1f} samples/s |")
    rows += [
        f"| Epochs | {getattr(config, 'num_epochs', '—')} |",
        f"| Batch size | {getattr(config, 'batch_size', '—')} (× {getattr(config, 'gradient_accumulation_steps', 1)} grad-accum) |",
        f"| Encoder LR | {getattr(config, 'encoder_lr', '—')} |",
        f"| Task-head LR | {getattr(config, 'task_lr', '—')} |",
        f"| Weight decay | {getattr(config, 'weight_decay', '—')} |",
        f"| Scheduler | {getattr(config, 'scheduler_type', '—')} (warmup {getattr(config, 'warmup_ratio', '—')}) |",
        f"| Precision | {'bf16' if getattr(config, 'bf16', False) else 'fp16' if getattr(config, 'fp16', False) else 'fp32'} |",
        f"| Max grad norm | {getattr(config, 'max_grad_norm', '—')} |",
        f"| Best-checkpoint metric | {getattr(config, 'metric_for_best', '—')} |",
        f"| Seed | {getattr(config, 'seed', '—')} |",
    ]
    if arch:
        rows.append(f"| Architecture | {', '.join(arch)} |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_model_card(
    *,
    model_name: str,
    base_model: Optional[str],
    cfg: Dict[str, Any],
    config: Any,
    dataset_keys: List[str],
    results: Optional[Dict[str, Any]],
    eval_metrics: Optional[Dict[str, Any]],
    test_metrics: Optional[Dict[str, Any]],
    generated_at: str,
    registry: Optional[Dict[str, Any]] = None,
    dataset_counts: Optional[Dict[str, Dict[str, int]]] = None,
) -> str:
    """Render a complete MODEL_CARD.md as a Markdown string."""
    registry = registry or load_registry()
    datasets = _resolve_datasets(registry, dataset_keys)

    base_entry = (registry.get("base_models") or {}).get(base_model or "", {})
    base_name = base_entry.get("name", base_model or "—")
    base_license = base_entry.get("license", "see model card")

    # License verdict over base model + every dataset.
    named = [(base_name, base_license)]
    for key, entry in datasets:
        name = entry.get("name", key) if entry else f"{key} (unknown)"
        named.append((name, entry.get("license", "") if entry else ""))
    verdict = summarize_licenses(named)

    # Frontmatter inputs.
    lang_codes, hf_ids = [], []
    for _key, entry in datasets:
        if not entry:
            continue
        for c in entry.get("language") or []:
            if c not in lang_codes:
                lang_codes.append(c)
        if entry.get("hf_id"):
            hf_ids.append(entry["hf_id"])

    parts: List[str] = [
        _frontmatter(model_name, base_model, lang_codes, hf_ids, verdict),
        "",
        f"# {model_name}",
        "",
        "A [GLiNER2](https://github.com/fastino-ai/GLiNER2) multi-task "
        "information-extraction model (entities, relations, events, and "
        "classification) fine-tuned from "
        f"`{base_model}`.",
        "",
        "## ⚠️ License at a glance",
        "",
        f"- **Effective license:** {verdict.headline}",
        f"- **Commercial use:** {verdict.commercial}",
        f"- **All dataset licenses verified:** {'Yes' if verdict.all_verified else 'No'}",
        "",
        "See [License](#license) for the full determination and per-dataset terms.",
        "",
        "## Model details",
        "",
        f"- **Base model:** [`{base_model}`]({base_entry.get('source_url', '')})"
        if base_entry.get("source_url") else f"- **Base model:** `{base_model}`",
        "- **Library:** `gliner2`",
        f"- **Tasks:** entity, relation, event, and classification extraction",
        f"- **Experiment:** `{getattr(config, 'experiment_name', '—')}`",
        "",
        _data_section(datasets, dataset_counts),
        "",
        _training_section(config, cfg, results, generated_at),
        "",
        "## Evaluation",
        "",
    ]

    blind = _metrics_table(test_metrics or {}, "Blind test (held-out test splits)")
    val = _metrics_table(eval_metrics or {}, "Best checkpoint (validation)")
    if blind:
        parts += [blind, ""]
    if val:
        parts += [val, ""]
    if not blind and not val:
        parts += ["_No evaluation metrics were produced for this run._", ""]

    parts += [
        _license_section(verdict, base_name, base_license),
        "",
        "## Citation",
        "",
        "If you use this model, please cite GLiNER2 and the underlying datasets "
        "(linked in [Training data](#training-data)).",
        "",
        "---",
        f"_Model card generated automatically at the end of training ({generated_at})._",
        "",
    ]
    return "\n".join(parts)
