"""One label style: Title_Snake (LABEL_STYLE_SPEC.md).

Labels are an INPUT to GLiNER2, so every spelling of one concept must reach the model as one
string. Words are capitalised and joined by ``_``; dots stay as the hierarchy separator and each
dotted segment is styled on its own; non-Latin labels are unchanged. An ALL-CAPS token is kept as
an acronym when it has at most 3 letters (``GPE``, ``PER``) and is not listed in ``words``, or is
listed in ``acronyms`` -- the generator lists the longer ones it judged acronyms (``NORP``) and the
short ones a reviewer judged words (``LAW``), so inference needs no dictionary.
A longer English word in capitals is shouted, not an acronym: ``PERSON`` -> ``Person``. So is a
short FUNCTION word inside a longer label: ``PART-OF`` -> ``Part_Of`` (a lone ``PER`` stays), and every
word of a SHOUTED phrase -- a multi-word segment written all in capitals -- unless it is a listed acronym:
``REGULATION OR LAW`` -> ``Regulation_Or_Law``. Words in ``preserve`` are kept verbatim: the camelCase
splitter would break ``ATPases`` into ``AT_Pases``, and a unit ratio keeps its slash (``mg/dL``).
"""
from __future__ import annotations

import re
from typing import Callable, Collection, Optional

TITLE_SNAKE = "title_snake"
_TOKEN = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_SPLIT = re.compile(r"[\s_\-/]+")
_SPLIT_KEEP_SLASH = re.compile(r"[\s_\-]+")
# Short ALL-CAPS tokens that are English function words, never acronyms, inside a multi-word label.
_FUNCTION = frozenset({"A", "AN", "AND", "AS", "AT", "BY", "FOR", "FROM", "IN", "OF", "ON", "OR", "PER",
                       "THE", "TO", "VS", "WITH"})


def _latin(label: str) -> bool:
    return not re.search(r"[^\x00-\x7f]", label)


def words(label: str) -> list:
    """Word tokens of one dotted segment: split on spaces, ``_``, ``-``, ``/`` and camelCase."""
    return [t for part in _SPLIT.split(label) for t in _TOKEN.findall(part)]


def is_acronym(tok: str, acronyms: Collection[str] = (), short_words: Collection[str] = ()) -> bool:
    """ALL-CAPS and either short (<= 3 letters, not a listed word) or a listed acronym."""
    return tok.isupper() and len(tok) > 1 and ((len(tok) <= 3 and tok not in short_words) or tok in acronyms)


def title_snake(label: str, acronyms: Collection[str] = (), short_words: Collection[str] = (),
                preserve: Collection[str] = ()) -> str:
    """``street address`` / ``streetAddress`` / ``STREET-ADDRESS`` -> ``Street_Address``."""
    if not _latin(label):
        return label
    if "." in label:
        return ".".join(title_snake(p, acronyms, short_words, preserve) for p in label.split("."))
    ws = [t for part in _SPLIT_KEEP_SLASH.split(label)
          for sub in ([part] if part in preserve else part.split("/"))
          for t in ([sub] if sub in preserve else _TOKEN.findall(sub))]
    shouted = len(ws) > 1 and all(t.isupper() or t.isdigit() for t in ws)
    toks = [t if t in preserve or (t in acronyms if shouted else is_acronym(t, acronyms, short_words)
                                   and not (len(ws) > 1 and t in _FUNCTION)) else t.capitalize() for t in ws]
    return "_".join(toks) or label


def styler(style: Optional[dict]) -> Optional[Callable[[str], str]]:
    """The label function a config's ``label_style`` names, or None when there is none."""
    if not style:
        return None
    if style.get("name") != TITLE_SNAKE:
        raise ValueError(f"unknown label_style {style.get('name')!r}; known: {TITLE_SNAKE}")
    acronyms, short_words = frozenset(style.get("acronyms") or ()), frozenset(style.get("words") or ())
    preserve = frozenset(style.get("preserve") or ())
    return lambda label: title_snake(label, acronyms, short_words, preserve)
