"""Post-processing to strip leading function words / stopwords from extracted spans.

Converts extent-style span predictions (e.g. ``"the health system"``,
``"an attacker"``, ``"in the northern province"``) into head-aligned spans
(``"health system"``, ``"attacker"``, ``"northern province"``) while preserving:

1. Exact character offsets (``start`` advances by the stripped prefix length so
   ``doc_text[start:end] == text`` holds exactly);
2. Quantifier and number phrases needed by downstream EKF casualty tracking
   (e.g. ``"at least 246 people"``, ``"more than 100,000"``, ``"nearly 1,400"``);
3. Proper-noun prefixes that generic IR stopword lists mistakenly include
   (e.g. ``"New York"``, ``"United States"``, ``"May 15"``).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

_EN_LEADING_STOPWORDS: FrozenSet[str] = frozenset({
    "the", "a", "an",
    "this", "that", "these", "those",
    "its", "their", "his", "her", "our", "my", "your",
    "of", "in", "on", "at", "to", "for", "by", "with", "from", "as",
    "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "being",
})

# Multilingual articles, determiners, and clitic prepositions for major languages
# (used even if stopwordsiso is not installed).
_MULTILINGUAL_ARTICLES: FrozenSet[str] = frozenset({
    # French
    "le", "la", "les", "un", "une", "des", "du", "de", "au", "aux", "ce", "cet", "cette", "ces",
    # Spanish
    "el", "los", "las", "una", "unos", "unas", "del", "al", "este", "esta", "estos", "estas",
    # Portuguese
    "o", "os", "as", "um", "uma", "uns", "umas", "do", "da", "dos", "das", "no", "na", "nos", "nas",
    # German
    "der", "die", "das", "den", "dem", "ein", "eine", "einer", "einem", "einen", "eines",
    # Italian
    "il", "lo", "gli", "uno", "della", "dello", "dei", "degli", "delle", "nel", "nella",
    # Dutch
    "het", "een",
    # Turkish determiners / function words (excluding "bir" when used as number 1)
    "bu", "şu", "o", "ve", "ile", "için",
    # Russian prepositions / conjunctions
    "в", "во", "на", "с", "со", "к", "ко", "по", "из", "у", "от", "за", "о", "об", "и", "а", "но",
    # Arabic standalone particles
    "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه",
})

# Words that must never be stripped even if present in a language's stopword list:
# numerals, scale words, bucket words, and common proper-noun adjectives/months.
_PROTECTED_WORDS: FrozenSet[str] = frozenset({
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million", "billion", "trillion",
    "first", "second", "third", "half", "quarter",
    "few", "couple", "dozen", "several", "many", "dozens", "hundreds", "thousands", "millions",
    "bin", "milyon", "milyar", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz", "on",
    "new", "united", "states", "general", "national", "world", "state",
    "north", "south", "east", "west", "central", "saint", "san", "los", "las",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
})

_QUANTIFIER_GUARD_RE = re.compile(
    r"^(?:at\s+least|more\s+than|less\s+than|fewer\s+than|up\s+to|"
    r"a\s+(?:dozen|hundred|thousand|million|billion|few|couple)|"
    r"about|around|roughly|nearly|almost|approximately|over|under|feared|estimated)\b",
    re.I,
)

_HAS_NUMBER_RE = re.compile(
    r"\b(?:\d[\d.,]*|zero|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion|trillion|"
    r"dozen|dozens|hundreds|thousands|few|couple|several|many|"
    r"bin|milyon|milyar)\b",
    re.I,
)

_LEADING_TOKEN_RE = re.compile(r"^([^\s]+)(\s+)(.+)$", flags=re.DOTALL)


@lru_cache(maxsize=1)
def default_leading_stopwords() -> FrozenSet[str]:
    """Return the combined multilingual leading-stopword set (cached)."""
    words = set(_EN_LEADING_STOPWORDS) | set(_MULTILINGUAL_ARTICLES)
    repo_root = Path(__file__).resolve().parents[2]
    yaml_path = repo_root / "tools" / "train" / "config" / "stopwords.yaml"
    if yaml_path.is_file():
        try:
            import yaml
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            for lang_words in data.values():
                if lang_words:
                    words.update(str(w).strip().lower() for w in lang_words if w)
        except Exception:
            pass
    words -= _PROTECTED_WORDS
    return frozenset(words)


def strip_leading_stopwords_text(
    text: str,
    stopwords: Optional[FrozenSet[str]] = None,
) -> Tuple[str, int]:
    """Strip leading stopword tokens from ``text``.

    Returns ``(new_text, offset_delta)`` where ``offset_delta`` is the number of
    characters removed from the front of ``text`` so ``text[offset_delta:] == new_text``
    (up to any trailing whitespace trim). Never strips if doing so would empty the
    span or break a numerical quantifier phrase (e.g. ``"at least 246"``).
    """
    if not isinstance(text, str) or not text:
        return text, 0
    sw = stopwords if stopwords is not None else default_leading_stopwords()
    lstripped = text.lstrip()
    if not lstripped:
        return text, 0
    delta = len(text) - len(lstripped)
    s = lstripped.rstrip()

    while True:
        if _QUANTIFIER_GUARD_RE.match(s) and _HAS_NUMBER_RE.search(s):
            break
        m = _LEADING_TOKEN_RE.match(s)
        if not m:
            break
        first, ws, rest = m.group(1), m.group(2), m.group(3)
        first_clean = first.lower()
        if any(ch.isdigit() for ch in first_clean) or first_clean in _PROTECTED_WORDS:
            break
        if first_clean in sw and rest.strip():
            delta += len(first) + len(ws)
            s = rest
        else:
            break

    return s, delta


def _strip_span_item(item: Any, stopwords: FrozenSet[str]) -> Any:
    """Strip leading stopwords from a bare string or a span dict ``{text, start, end, ...}``."""
    if isinstance(item, str):
        new_text, _ = strip_leading_stopwords_text(item, stopwords)
        return new_text
    if isinstance(item, dict) and isinstance(item.get("text"), str):
        new_text, delta = strip_leading_stopwords_text(item["text"], stopwords)
        if delta == 0 and new_text == item["text"]:
            return item
        updated = dict(item)
        updated["text"] = new_text
        if isinstance(updated.get("start"), int):
            updated["start"] = updated["start"] + delta
        return updated
    return item


def _dedupe_span_list(items: List[Any]) -> List[Any]:
    """Deduplicate formatted span items after stopword stripping."""
    out: List[Any] = []
    seen = set()
    for item in items:
        if isinstance(item, str):
            key = (item.lower(), None, None)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            key = (item["text"].lower(), item.get("start"), item.get("end"))
        else:
            out.append(item)
            continue
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def strip_leading_stopwords_from_results(
    results: Dict[str, Any],
    stopwords: Optional[FrozenSet[str]] = None,
    classification_tasks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Post-process a formatted extraction result dict to strip leading stopwords."""
    if not isinstance(results, dict) or not results:
        return results
    sw = stopwords if stopwords is not None else default_leading_stopwords()
    cls_set = set(classification_tasks or ())
    out: Dict[str, Any] = {}

    for key, value in results.items():
        if key in cls_set:
            out[key] = value
        elif key == "entities" and isinstance(value, dict):
            ent_out: Dict[str, Any] = {}
            for label, spans in value.items():
                if isinstance(spans, list):
                    ent_out[label] = _dedupe_span_list([_strip_span_item(s, sw) for s in spans])
                else:
                    ent_out[label] = _strip_span_item(spans, sw)
            out[key] = ent_out
        elif key == "relation_extraction" and isinstance(value, dict):
            rel_out: Dict[str, Any] = {}
            for rtype, edges in value.items():
                if not isinstance(edges, list):
                    rel_out[rtype] = edges
                    continue
                new_edges = []
                for edge in edges:
                    if isinstance(edge, tuple) and len(edge) == 2:
                        h, _ = strip_leading_stopwords_text(edge[0], sw)
                        t, _ = strip_leading_stopwords_text(edge[1], sw)
                        new_edges.append((h, t))
                    elif isinstance(edge, dict) and "head" in edge and "tail" in edge:
                        updated = dict(edge)
                        updated["head"] = _strip_span_item(edge["head"], sw)
                        updated["tail"] = _strip_span_item(edge["tail"], sw)
                        new_edges.append(updated)
                    else:
                        new_edges.append(edge)
                rel_out[rtype] = new_edges
            out[key] = rel_out
        elif key == "event_extraction" and isinstance(value, dict):
            ev_out: Dict[str, Any] = {}
            for etype, instances in value.items():
                if not isinstance(instances, list):
                    ev_out[etype] = instances
                    continue
                new_instances = []
                for inst in instances:
                    if not isinstance(inst, dict):
                        new_instances.append(inst)
                        continue
                    new_inst = dict(inst)
                    if isinstance(inst.get("triggers"), list):
                        new_inst["triggers"] = _dedupe_span_list(
                            [_strip_span_item(t, sw) for t in inst["triggers"]]
                        )
                    if isinstance(inst.get("arguments"), list):
                        new_args = []
                        seen_args = set()
                        for arg in inst["arguments"]:
                            if isinstance(arg, dict) and "entity" in arg:
                                new_arg = dict(arg)
                                new_arg["entity"] = _strip_span_item(arg["entity"], sw)
                                ent_val = new_arg["entity"]
                                ent_key = (
                                    (ent_val.get("text", "").lower(), ent_val.get("start"), ent_val.get("end"))
                                    if isinstance(ent_val, dict)
                                    else (str(ent_val).lower(), None, None)
                                )
                                dedupe_key = (new_arg.get("role"), ent_key)
                                if dedupe_key not in seen_args:
                                    seen_args.add(dedupe_key)
                                    new_args.append(new_arg)
                            else:
                                new_args.append(arg)
                        new_inst["arguments"] = new_args
                    new_instances.append(new_inst)
                ev_out[etype] = new_instances
            out[key] = ev_out
        elif key == "coref_clusters" and isinstance(value, list):
            out[key] = [
                [_strip_span_item(m, sw) for m in cluster] if isinstance(cluster, list) else cluster
                for cluster in value
            ]
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            # Structure instances
            new_structs = []
            for inst in value:
                new_inst = {}
                for fname, fval in inst.items():
                    if fname == "confidence":
                        new_inst[fname] = fval
                    elif isinstance(fval, list):
                        new_inst[fname] = _dedupe_span_list([_strip_span_item(v, sw) for v in fval])
                    else:
                        new_inst[fname] = _strip_span_item(fval, sw)
                new_structs.append(new_inst)
            out[key] = new_structs
        else:
            out[key] = value

    return out
