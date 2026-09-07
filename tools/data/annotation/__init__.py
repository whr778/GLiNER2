"""Load shared annotator rules from GUIDELINES.md at prompt-build time.

THE MARKDOWN IS THE SOURCE. The rules are not defined in Python and rendered to a document,
nor described by a document kept alongside the code -- both of those let the reviewed text
and the sent text drift apart, and the drift is invisible because a prompt has no type
error. Here the bytes between the markers in GUIDELINES.md are the bytes the model
receives, so reviewing the file IS reviewing the prompt.

    from annotation import rules
    SYSTEM = rules("json_only", "verbatim") + " You annotate Chinese news articles..."
"""
from __future__ import annotations

import re
from pathlib import Path

GUIDELINES = Path(__file__).resolve().parent / "GUIDELINES.md"

_BLOCK = re.compile(r"<!--\s*rule:\s*(\w+)\s*-->\n(.*?)\n<!--\s*end\s*-->", re.DOTALL)


def _blocks() -> dict[str, str]:
    text = GUIDELINES.read_text(encoding="utf-8")
    found = {m.group(1): " ".join(m.group(2).split()) for m in _BLOCK.finditer(text)}
    if not found:
        raise ValueError(f"no rule blocks found in {GUIDELINES}; the markers are the "
                         f"contract between the document and the prompt")
    return found


def rules(*names: str) -> str:
    """Return the named rules as one paragraph, in the order asked for.

    An unknown name raises rather than silently contributing nothing: a prompt missing a
    rule it believes it has is exactly the defect this module exists to prevent.
    """
    found = _blocks()
    missing = [n for n in names if n not in found]
    if missing:
        raise KeyError(f"no such rule(s) in {GUIDELINES.name}: {', '.join(missing)}; "
                       f"available: {', '.join(sorted(found))}")
    return " ".join(found[n] for n in names)


def available() -> list[str]:
    """Rule names defined in GUIDELINES.md."""
    return sorted(_blocks())
