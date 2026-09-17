"""The default splitter must not make a span unalignable.

`SchemaTransformer._find_sublist` matches token SUBSEQUENCES, so any run of characters that
collapses into ONE token makes every entity inside it impossible to align -- training then
aborts with "entity ... was not found" for a surface that is present verbatim. That is
upstream issue #156; these tests pin our side of it.
"""

import pytest

from gliner2.processing.word_splitter import WhitespaceTokenSplitter


@pytest.fixture
def split():
    sp = WhitespaceTokenSplitter()
    return lambda text: [w for w, _s, _e in sp(text, lower=False)]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("中国", ["中", "国"]),                      # CJK Unified
        ("㐀㐁", ["㐀", "㐁"]),      # Extension A
        # ESCAPES, NOT LITERALS. U+F900 renders identically to the Unified U+8C48 it
        # folds to, so a pasted literal silently tests the wrong block -- this case
        # did exactly that and passed against the unfixed pattern.
        ("\uf900\uf901", ["\uf900", "\uf901"]),      # Compatibility Ideographs
        ("\U00020000\U00020001", ["\U00020000", "\U00020001"]),  # Extension B, non-BMP
        ("ひらがな", ["ひ", "ら", "が", "な"]),          # Hiragana
        ("カタカナ", ["カ", "タ", "カ", "ナ"]),          # Katakana
        ("ｱｲｳ", ["ｱ", "ｲ", "ｳ"]),                    # Halfwidth Katakana
        ("ㄅㄆ", ["ㄅ", "ㄆ"]),                        # Bopomofo
        ("한국", ["한", "국"]),                        # Hangul Syllables
        ("ᄀᄁ", ["ᄀ", "ᄁ"]),      # Hangul Jamo
    ],
)
def test_whitespaceless_scripts_split_per_character(split, text, expected):
    assert split(text) == expected


def test_latin_and_cjk_do_not_merge(split):
    """The adjacency case: a shared token would make both sides unalignable."""
    assert split("Apple公司发布iPhone") == ["Apple", "公", "司", "发", "布", "iPhone"]


def test_an_entity_inside_a_cjk_sentence_is_alignable(split):
    """The actual failure from issue #156, expressed as the property that matters."""
    text, entity = "中国国家主席访问美国", "美国"
    text_tokens, entity_tokens = split(text), split(entity)
    n = len(entity_tokens)
    positions = [
        i for i in range(len(text_tokens) - n + 1)
        if text_tokens[i:i + n] == entity_tokens
    ]
    assert positions, "entity present verbatim must be locatable as a token subsequence"


@pytest.mark.parametrize(
    "text",
    ["https://example.com/a?b=1", "www.example.com/x", "bob.smith@example.com",
     "state-of-the-art", "@mention"],
)
def test_urls_emails_and_hyphenation_stay_intact(split, text):
    """CharLevelSplitter fragments these; the default must not start doing so."""
    assert split(text) == [text]


@pytest.mark.parametrize(
    "text", ["สวัสดีครับ", "ສະບາຍດີ", "ខ្ញុំ", "café", "résumé"]
)
def test_combining_marks_stay_attached_to_their_base(split, text):
    """Mn/Mc marks are not \\w, so a bare \\w+ run severs them into their own tokens.

    Thai tokenized to ['สว', 'ั', 'สด', 'ี'] -- the mark divorced from the character it
    modifies, which is worse than a coarse split because it corrupts the surface itself.
    """
    tokens = split(text)
    assert tokens == [text], f"expected one intact token, got {tokens}"


def test_offsets_index_the_original_string(split):
    """Offsets are used to slice the caller's text; drift silently mislabels spans."""
    sp = WhitespaceTokenSplitter()
    text = "华为 announced https://huawei.com/新闻 today"
    for token, start, end in sp(text, lower=False):
        assert text[start:end] == token


def test_lowercasing_does_not_move_offsets():
    """Lower-casing the text first is unsafe: Unicode case changes length."""
    sp = WhitespaceTokenSplitter()
    text = "İstanbul 中国"
    for token, start, end in sp(text, lower=True):
        assert text[start:end].lower() == token


def test_compatibility_ideographs_are_folded_by_our_own_converters():
    """Context for the range above: our corpora never carry U+F900 in the first place.

    `_split.dumps_record` NFKC-normalizes every record on write, and NFKC folds
    U+F900 to U+8C48. The range is kept because the splitter is public API and a caller
    can pass un-normalized text straight to it.
    """
    import unicodedata

    assert unicodedata.normalize("NFKC", "\uf900") == "\u8c48"
