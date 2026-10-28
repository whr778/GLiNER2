"""Print the metrics card stored with a checkpoint.

Train and eval write ``val_metrics.json``, ``test_metrics.json``, and
``eval_metrics.json``. Each category is ``eval_<category>_strict_micro_{precision,recall,f1}``.
The operating point is ``eval_provenance`` (threshold, split, records). Decode settings
are ``inference_defaults`` in ``config.json``. This module reads those files and does
not load weights.

    uv run python tools/train/format_metrics.py <checkpoint>
    uv run python tools/train/format_metrics.py owner/repo
    uv run python tools/train/format_metrics.py owner/repo@subfolder
    uv run python tools/train/format_metrics.py owner/repo --subfolder checkpoint-epoch-2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

# Display order. Same categories as model_card.CATEGORIES. Any other
# eval_*_strict_micro_f1 key in the file is printed after these.
CATEGORIES = (
    "entity", "relation", "classification", "structure",
    "event_type", "event_trigger", "event_argument", "event",
)

METRIC_FILES = ("val_metrics.json", "test_metrics.json", "eval_metrics.json")
JSON_FILES = METRIC_FILES + ("train_results.json", "config.json")
PROVENANCE_KEYS = (
    "split", "records", "threshold", "full_menu", "menu_negatives",
    "commit", "config", "checkpoint", "per_head_thresholds",
)


def _is_hub_id(name: str) -> bool:
    """True for ``owner/model``. Same rule as ``model_card._is_hub_id``."""
    if not name or name.startswith((".", "/", "~")) or "\\" in name:
        return False
    parts = name.split("/")
    return len(parts) == 2 and all(parts) and not any(p.startswith(".") for p in parts)


def _fmt(value: Any) -> str:
    if isinstance(value, dict):
        if not value:
            return "none"
        return ", ".join(f"{k}={_fmt(v)}" for k, v in value.items())
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return f"{value:,}" if abs(value) >= 1000 else str(value)
    return str(value)


def _dirs(root: Path) -> list[Path]:
    """The directory itself, its ``best/`` child, and the parent of a ``best/`` dir."""
    dirs = [root]
    best = root / "best"
    if best.is_dir():
        dirs.append(best)
    if root.name == "best":
        dirs.append(root.parent)
    return dirs


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _last_jsonl(path: Path) -> Optional[dict]:
    last = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = line
    if not last:
        return None
    row = json.loads(last)
    return row if isinstance(row, dict) else None


def _read_local(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"No checkpoint directory: {root}")
    found: dict[str, Any] = {}
    for name in JSON_FILES:
        for directory in _dirs(root):
            path = directory / name
            if path.is_file():
                found[name] = _load_json(path)
                break
    if not any(name in found for name in METRIC_FILES):
        for directory in _dirs(root):
            path = directory / "eval_history.jsonl"
            if path.is_file():
                row = _last_jsonl(path)
                if row:
                    found["eval_history.jsonl"] = row
                break
    return found


def _hub_file(repo: str, filename: str, subfolder: Optional[str], repo_type: str):
    """One JSON file. ``absent`` means the repo exists and the file does not.

    ``local-miss`` is an offline cache miss: the repo may still exist as the
    other repo type (run logs are datasets). ``no-repo`` means the Hub has no
    such repo.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        EntryNotFoundError, LocalEntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError,
    )

    try:
        return hf_hub_download(
            repo_id=repo, filename=filename, subfolder=subfolder, repo_type=repo_type,
        )
    except LocalEntryNotFoundError:
        return "local-miss"
    except (EntryNotFoundError, RevisionNotFoundError):
        return "absent"
    except RepositoryNotFoundError:
        return "no-repo"


def _scan_hub(repo: str, subfolder: Optional[str], repo_type: str, names: tuple[str, ...]) -> tuple[dict[str, Any], bool]:
    """``(files, repo_was_reached)``. A local cache miss does not count as reached."""
    found: dict[str, Any] = {}
    reached = False
    for name in names:
        got = _hub_file(repo, name, subfolder, repo_type)
        if got in ("no-repo", "local-miss"):
            continue
        if got == "absent":
            reached = True
            continue
        found[name] = _load_json(Path(got))
        reached = True
    return found, reached


def _read_hub(repo: str, subfolder: Optional[str]) -> dict[str, Any]:
    found, reached = _scan_hub(repo, subfolder, "model", JSON_FILES)
    if not found and not reached:
        found, reached = _scan_hub(repo, subfolder, "dataset", JSON_FILES)
    if not found and not reached:
        where = f"{repo}@{subfolder}" if subfolder else repo
        raise FileNotFoundError(f"Hub repo not found: {where}")
    if not any(name in found for name in METRIC_FILES):
        for repo_type in ("model", "dataset"):
            got = _hub_file(repo, "eval_history.jsonl", subfolder, repo_type)
            if got in ("no-repo", "local-miss"):
                continue
            if got == "absent":
                break
            row = _last_jsonl(Path(got))
            if row:
                found["eval_history.jsonl"] = row
            break
    return found


def resolve(spec: str, subfolder: Optional[str] = None) -> tuple[str, str, Optional[str]]:
    """``(kind, location, subfolder)``. ``kind`` is ``local`` or ``hub``."""
    raw = (spec or "").strip()
    if not raw:
        raise FileNotFoundError("checkpoint is empty")
    path = Path(raw).expanduser()
    if path.is_dir():
        if subfolder:
            path = path / subfolder
        if not path.is_dir():
            raise FileNotFoundError(f"No checkpoint directory: {path}")
        return "local", str(path), None
    repo, inline = raw, None
    if "@" in raw:
        repo, _, inline = raw.partition("@")
        inline = inline.strip() or None
        if not _is_hub_id(repo):
            raise FileNotFoundError(f"Not a local directory or Hub id: {raw}")
    elif not _is_hub_id(raw):
        raise FileNotFoundError(f"Not a local directory or Hub id: {raw}")
    return "hub", repo, (subfolder or inline)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _micro_rows(metrics: dict, prefix: str) -> list[dict[str, Any]]:
    """One row per category with a numeric strict micro F1.

    Row keys are ``category`` plus whichever of ``precision``, ``recall``,
    ``f1``, ``support``, and ``relaxed_f1`` are present.
    """
    head = "eval_" + prefix
    tail = "_strict_micro_f1"
    cats: list[str] = []
    for key, value in metrics.items():
        if not (isinstance(key, str) and key.startswith(head) and key.endswith(tail)):
            continue
        if not _is_number(value):
            continue
        cat = key[len(head):-len(tail)]
        if prefix == "" and cat.startswith("fullmenu_"):
            continue
        if cat not in cats:
            cats.append(cat)
    ordered = [c for c in CATEGORIES if c in cats] + [c for c in cats if c not in CATEGORIES]
    rows = []
    for cat in ordered:
        base = f"{head}{cat}_strict_micro_"
        row: dict[str, Any] = {"category": cat}
        for metric in ("precision", "recall", "f1"):
            value = metrics.get(base + metric)
            if _is_number(value):
                row[metric] = value
        support = metrics.get(f"{head}{cat}_strict_support")
        if isinstance(support, int) and not isinstance(support, bool):
            row["support"] = support
        relaxed = metrics.get(f"{head}{cat}_relaxed_micro_f1")
        if _is_number(relaxed):
            row["relaxed_f1"] = relaxed
        if any(key in row for key in ("precision", "recall", "f1", "support", "relaxed_f1")):
            rows.append(row)
    return rows


def _micro_lines(metrics: dict, prefix: str, indent: str) -> list[str]:
    lines = []
    for row in _micro_rows(metrics, prefix):
        bits = []
        for metric, label in (
            ("precision", "strict micro precision"),
            ("recall", "strict micro recall"),
            ("f1", "strict micro F1"),
        ):
            if metric in row:
                bits.append(f"{label} {_fmt(row[metric])}")
        if "support" in row:
            bits.append(f"support {_fmt(row['support'])}")
        if "relaxed_f1" in row:
            bits.append(f"relaxed micro F1 {_fmt(row['relaxed_f1'])}")
        if bits:
            lines.append(f"{indent}{row['category']}  " + "  ".join(bits))
    return lines


def _section(name: str, metrics: dict) -> list[str]:
    lines = [name]
    provenance = metrics.get("eval_provenance")
    if isinstance(provenance, dict):
        for key in PROVENANCE_KEYS:
            if key in provenance:
                lines.append(f"  {key}: {_fmt(provenance[key])}")
    else:
        lines.append("  eval_provenance: none")
    for key in ("eval_epoch", "eval_step", "is_best"):
        if key in metrics:
            lines.append(f"  {key}: {_fmt(metrics[key])}")
    lines.extend(_micro_lines(metrics, "", "  "))
    full = _micro_lines(metrics, "fullmenu_", "    ")
    if full:
        lines.append("  full menu")
        lines.extend(full)
    for key, label in (
        ("eval_overall_strict_head_min_f1", "overall strict head-min F1"),
        ("eval_fullmenu_overall_strict_head_min_f1", "full menu overall strict head-min F1"),
    ):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            lines.append(f"  {label}: {_fmt(value)}")
    record_threshold = metrics.get("structure_record_threshold")
    if isinstance(record_threshold, (int, float)) and not isinstance(record_threshold, bool):
        lines.append(f"  structure record threshold: {_fmt(record_threshold)}")
    default_f1 = metrics.get("structure_f1_at_default_threshold")
    if isinstance(default_f1, (int, float)) and not isinstance(default_f1, bool):
        lines.append(f"  structure strict micro F1 at the 0.5 record-threshold default: {_fmt(default_f1)}")
    per_head = metrics.get("per_head")
    if isinstance(per_head, dict):
        lines.append("  per_head")
        if per_head.get("identical_to_global"):
            lines.append("    identical to the global threshold")
        if isinstance(per_head.get("thresholds"), dict):
            lines.append(f"    thresholds: {_fmt(per_head['thresholds'])}")
        if isinstance(per_head.get("metrics"), dict):
            lines.extend(_micro_lines(per_head["metrics"], "", "    "))
    off = metrics.get("coref_merge_off")
    if isinstance(off, dict):
        nested = _micro_lines(off, "", "    ")
        if nested:
            lines.append("  coref_merge_off")
            lines.extend(nested)
    by_menu = metrics.get("by_menu")
    if isinstance(by_menu, dict):
        for menu, block in by_menu.items():
            if not isinstance(block, dict):
                continue
            nested = _micro_lines(block, "", "    ")
            if nested:
                lines.append(f"  by_menu {menu}")
                lines.extend(nested)
    languages = metrics.get("by_language")
    if isinstance(languages, dict) and languages:
        lines.append("  by_language: " + ", ".join(str(k) for k in languages) + " (not expanded)")
    return lines


def format_card(title: str, files: dict[str, Any]) -> str:
    """Render one card from already-loaded JSON objects, keyed by filename."""
    lines = [title, ""]
    config = files.get("config.json")
    defaults = config.get("inference_defaults") if isinstance(config, dict) else None
    lines.append("inference_defaults")
    if isinstance(defaults, dict) and defaults:
        for key, value in defaults.items():
            lines.append(f"  {key}: {_fmt(value)}")
    else:
        lines.append("  none")
    lines.append("")
    results = files.get("train_results.json")
    if isinstance(results, dict):
        picked = [k for k in ("best_epoch", "best_step", "best_metric") if k in results]
        if picked:
            lines.append("train_results.json")
            for key in picked:
                lines.append(f"  {key}: {_fmt(results[key])}")
            lines.append("")
    shown = False
    for name in (*METRIC_FILES, "eval_history.jsonl"):
        metrics = files.get(name)
        if isinstance(metrics, dict):
            shown = True
            lines.extend(_section(name, metrics))
            lines.append("")
    if not shown:
        lines.append("No metrics file (val_metrics.json, test_metrics.json, eval_metrics.json).")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load(spec: str, subfolder: Optional[str] = None) -> tuple[str, dict[str, Any]]:
    """``(title, files)`` for a local directory or a Hub id."""
    kind, location, sub = resolve(spec, subfolder)
    if kind == "local":
        return location, _read_local(Path(location))
    files = _read_hub(location, sub)
    title = f"{location}@{sub}" if sub else location
    return title, files


_ROW_KEYS = ("precision", "recall", "f1", "support", "relaxed_f1")


def _display_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    shown = []
    for row in rows:
        item = {"category": row["category"]}
        for key in _ROW_KEYS:
            if key in row:
                item[key] = _fmt(row[key])
        shown.append(item)
    return shown


def _pairs(mapping: dict, keys: Optional[tuple[str, ...]] = None) -> list[dict[str, str]]:
    if keys is None:
        items = list(mapping.items())
    else:
        items = [(key, mapping[key]) for key in keys if key in mapping]
    return [{"key": str(key), "value": _fmt(value)} for key, value in items]


def _section_view(name: str, metrics: dict) -> dict[str, Any]:
    provenance = metrics.get("eval_provenance")
    prov = _pairs(provenance, PROVENANCE_KEYS) if isinstance(provenance, dict) else None
    fields = _pairs({key: metrics[key] for key in ("eval_epoch", "eval_step", "is_best") if key in metrics})
    notes = []
    for key, label in (
        ("eval_overall_strict_head_min_f1", "overall strict head-min F1"),
        ("eval_fullmenu_overall_strict_head_min_f1", "full menu overall strict head-min F1"),
    ):
        value = metrics.get(key)
        if _is_number(value):
            notes.append({"key": label, "value": _fmt(value)})
    record_threshold = metrics.get("structure_record_threshold")
    if _is_number(record_threshold):
        notes.append({"key": "structure record threshold", "value": _fmt(record_threshold)})
    default_f1 = metrics.get("structure_f1_at_default_threshold")
    if _is_number(default_f1):
        notes.append({
            "key": "structure strict micro F1 at the 0.5 record-threshold default",
            "value": _fmt(default_f1),
        })
    per_head_view = None
    per_head = metrics.get("per_head")
    if isinstance(per_head, dict):
        thresholds = per_head.get("thresholds")
        nested = per_head.get("metrics")
        per_head_view = {
            "identical_to_global": bool(per_head.get("identical_to_global")),
            "thresholds": _fmt(thresholds) if isinstance(thresholds, dict) else None,
            "rows": _display_rows(_micro_rows(nested, "")) if isinstance(nested, dict) else [],
        }
    coref = None
    off = metrics.get("coref_merge_off")
    if isinstance(off, dict):
        rows = _display_rows(_micro_rows(off, ""))
        if rows:
            coref = rows
    menus = []
    by_menu = metrics.get("by_menu")
    if isinstance(by_menu, dict):
        for menu, block in by_menu.items():
            if not isinstance(block, dict):
                continue
            rows = _display_rows(_micro_rows(block, ""))
            if rows:
                menus.append({"menu": str(menu), "rows": rows})
    languages = metrics.get("by_language")
    by_language = None
    if isinstance(languages, dict) and languages:
        by_language = ", ".join(str(key) for key in languages)
    return {
        "name": name,
        "provenance": prov,
        "fields": fields,
        "rows": _display_rows(_micro_rows(metrics, "")),
        "full_menu": _display_rows(_micro_rows(metrics, "fullmenu_")),
        "notes": notes,
        "per_head": per_head_view,
        "coref_merge_off": coref,
        "by_menu": menus,
        "by_language": by_language,
    }


def structure_card(title: str, files: dict[str, Any]) -> dict[str, Any]:
    """The same card as ``format_card``, as fields the viewer lays out.

    Metric cells are display strings (four decimal places, grouped counts)
    so the popup matches the CLI text.
    """
    config = files.get("config.json")
    defaults = config.get("inference_defaults") if isinstance(config, dict) else None
    inference = _pairs(defaults) if isinstance(defaults, dict) and defaults else None
    train = None
    results = files.get("train_results.json")
    if isinstance(results, dict):
        picked = _pairs(results, ("best_epoch", "best_step", "best_metric"))
        if picked:
            train = picked
    sections = []
    for name in (*METRIC_FILES, "eval_history.jsonl"):
        metrics = files.get(name)
        if isinstance(metrics, dict):
            sections.append(_section_view(name, metrics))
    return {
        "title": title,
        "inference_defaults": inference,
        "train_results": train,
        "sections": sections,
        "empty": not sections,
    }


def render(spec: str, subfolder: Optional[str] = None) -> str:
    """Load a local directory or a Hub id and return the card text."""
    title, files = load(spec, subfolder)
    return format_card(title, files)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Print the metrics card stored with a checkpoint.")
    parser.add_argument("checkpoint", help="local directory, Hub id, or repo@subfolder")
    parser.add_argument("--subfolder", help="subfolder inside a Hub repo (same as repo@subfolder)")
    args = parser.parse_args(argv)
    try:
        sys.stdout.write(render(args.checkpoint, args.subfolder))
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
