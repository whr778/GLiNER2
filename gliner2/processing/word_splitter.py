"""Word-boundary splitters used before subword tokenization.

A splitter is any callable with signature
``(text: str, lower: bool = True) -> Iterator[tuple[str, int, int]]``.
Each yield is ``(token, start, end)`` where ``start``/``end`` are exclusive-end
character offsets into the **original** ``text``. Lower-case only the token
value when ``lower=True``; never mutate the source string first, because
Unicode case folding can change length and corrupt offsets.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterator, Optional, Tuple, Union

WordSplitter = Callable[[str, bool], Iterator[Tuple[str, int, int]]]
WordSplitterSpec = Union[str, WordSplitter, type]


# Scripts written without whitespace word delimiters. Each character is its own token,
# because a run of them would otherwise become ONE token and no entity span inside it could
# ever align -- `_find_sublist` matches token SUBSEQUENCES, so `美国` cannot be found inside
# a single `中国国家主席访问美国` token. Defined once and interpolated; the ranges used to be
# repeated three times inside one pattern, which is how the extension blocks below came to be
# missing from it.
_CJK_RANGES = (
    "\u4e00-\u9fff"      # CJK Unified Ideographs
    "\u3400-\u4dbf"      # CJK Unified Ideographs Extension A
    "\uf900-\ufaff"      # CJK Compatibility Ideographs
    "\U00020000-\U0002a6df"  # Extension B -- outside the BMP
    "\U0002a700-\U0002ebef"  # Extensions C-F
    "\u3040-\u309f"      # Hiragana
    "\u30a0-\u30ff"      # Katakana
    "\u31f0-\u31ff"      # Katakana Phonetic Extensions
    "\uff66-\uff9d"      # Halfwidth Katakana
    "\u3100-\u312f"      # Bopomofo
    "\uac00-\ud7af"      # Hangul Syllables
    "\u1100-\u11ff"      # Hangul Jamo
    "\u3130-\u318f"      # Hangul Compatibility Jamo
)

# Combining marks (Unicode Mn/Mc) are NOT word characters to `re`, so a bare `\w+` run
# breaks at every one of them and leaves the mark stranded as its own token: `สวัสดี`
# tokenized to ['สว', 'ั', 'สด', 'ี'], which is worse than a coarse split because the mark
# is severed from the base character it modifies. Listing the ranges for the scripts that
# actually carry them keeps a word and its marks in one token.
_MARK_RANGES = (
    "\u0300-\u036f"      # Combining Diacritical Marks (NFD Latin/Greek/Cyrillic)
    "\u0483-\u0489"      # Cyrillic
    "\u0591-\u05bd\u05bf\u05c1-\u05c2\u05c4-\u05c5\u05c7"  # Hebrew
    "\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc"          # Arabic
    "\u0900-\u0903\u093a-\u094f\u0951-\u0957\u0962-\u0963"  # Devanagari
    "\u0e31\u0e34-\u0e3a\u0e47-\u0e4e"                          # Thai
    "\u0eb1\u0eb4-\u0ebc\u0ec8-\u0ecd"                          # Lao
    "\u17b4-\u17d3"      # Khmer
    "\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20f0\ufe20-\ufe2f"  # further combining blocks
)


class WhitespaceTokenSplitter:
    """Fast regex-based tokenizer for text splitting.

    Characters from scripts without whitespace delimiters (Chinese, Japanese, Korean) are
    each matched as an individual token, so a span inside them can align. The non-CJK word
    alternation explicitly EXCLUDES those ranges, so adjacent Latin and CJK do not merge --
    ``Apple公司`` is ``['Apple', '公', '司']``, not one token.

    Combining marks stay attached to the character they modify rather than becoming separate
    tokens. ``CharLevelSplitter`` below remains available for fully character-level use, but
    note it fragments URLs and emails, which this splitter keeps intact.

    KNOWN LIMIT: Thai, Lao and Khmer are written without spaces but their words are
    multi-character, so a run still becomes a single token here. Correct segmentation needs a
    dictionary, not a regex. Marks are no longer severed from their base, which is the part a
    regex can get right.
    """

    __slots__ = ()

    _PATTERN = re.compile(
        rf"""(?:https?://[^\s]+|www\.[^\s]+)
        |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{{2,}}
        |@[a-z0-9_]+
        |[{_CJK_RANGES}][{_MARK_RANGES}]*
        |(?:[^\W{_CJK_RANGES}]|[{_MARK_RANGES}])+
         (?:[-_](?:[^\W{_CJK_RANGES}]|[{_MARK_RANGES}])+)*
        |\S""",
        re.VERBOSE | re.IGNORECASE,
    )

    def __call__(self, text: str, lower: bool = True) -> Iterator[Tuple[str, int, int]]:
        # Match against the original text (the pattern is already case-insensitive)
        # so offsets index the caller's string, then lower-case only the token
        # value. Lower-casing the text first is unsafe because Unicode case
        # folding can change length (e.g. "İ".lower() -> "i\u0307"), which would
        # corrupt the recorded start/end offsets.
        for m in self._PATTERN.finditer(text):
            token = m.group()
            yield (token.lower() if lower else token), m.start(), m.end()


class CharLevelSplitter:
    """Keep Latin words together and treat every other non-space as a token.

    Suitable for languages without whitespace-delimited words, such as Chinese.
    ASCII words, emails, and similar ``[A-Za-z0-9@._-+]`` runs stay intact.
    """

    __slots__ = ()

    _PATTERN = re.compile(r"[A-Za-z0-9@._\-+]+|\S")

    def __call__(self, text: str, lower: bool = True) -> Iterator[Tuple[str, int, int]]:
        for m in self._PATTERN.finditer(text):
            token = m.group()
            yield (token.lower() if lower else token), m.start(), m.end()


WORD_SPLITTERS = {
    "whitespace": WhitespaceTokenSplitter,
    "char": CharLevelSplitter,
}
DEFAULT_WORD_SPLITTER = "whitespace"


def resolve_word_splitter(
    word_splitter: Optional[WordSplitterSpec] = None,
) -> WordSplitter:
    """Resolve a built-in name, class, or callable to a splitter instance."""
    if word_splitter is None:
        word_splitter = DEFAULT_WORD_SPLITTER
    if isinstance(word_splitter, str):
        try:
            word_splitter = WORD_SPLITTERS[word_splitter]
        except KeyError:
            supported = ", ".join(repr(name) for name in sorted(WORD_SPLITTERS))
            raise ValueError(
                f"Unknown word_splitter {word_splitter!r}. "
                f"Supported names: {supported}."
            ) from None
    if isinstance(word_splitter, type):
        try:
            word_splitter = word_splitter()
        except TypeError as exc:
            raise TypeError(
                "word_splitter class must be instantiable with no arguments"
            ) from exc
    if callable(word_splitter):
        return word_splitter
    raise TypeError(
        "word_splitter must be a built-in name "
        f"{sorted(WORD_SPLITTERS)} or a callable, "
        f"got {type(word_splitter).__name__}"
    )


def word_splitter_from(owner: Any) -> Optional[WordSplitter]:
    """Return the active splitter on a model or facade, if one is attached."""
    processor = getattr(owner, "processor", None)
    if processor is None:
        model = getattr(owner, "model", None)
        processor = getattr(model, "processor", None)
    return getattr(processor, "word_splitter", None)
