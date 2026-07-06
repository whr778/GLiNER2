r"""Regression tests for CJK tokenization in WhitespaceTokenSplitter.

Bug: ``\w+`` in the regex consumed consecutive CJK characters as a single
token, making it impossible to locate entity spans within Asian-language text
via _find_sublist. Each CJK character must be its own token.

Covered Unicode ranges:
  一-鿿  (U+4E00–U+9FFF)  CJK Unified Ideographs (main Chinese block)
  㐀-䶿  (U+3400–U+4DBF)  CJK Extension A
  ぀-ゟ  (U+3040–U+309F)  Hiragana
  ゠-ヿ  (U+30A0–U+30FF)  Katakana
  가-힯  (U+AC00–U+D7AF)  Hangul Syllables
"""

import pytest
from gliner2.processor import WhitespaceTokenSplitter


@pytest.fixture(scope="module")
def splitter():
    return WhitespaceTokenSplitter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokens(splitter, text):
    """Return (token, start, end) triples with lowercasing."""
    return list(splitter(text, lower=True))


def _token_strings(splitter, text):
    return [t for t, _, _ in _tokens(splitter, text)]


def _find_span(text_tokens, span_tokens):
    """Return True if span_tokens appears contiguously inside text_tokens."""
    n = len(span_tokens)
    return any(text_tokens[i:i + n] == span_tokens for i in range(len(text_tokens) - n + 1))


# ===========================================================================
# Core regression: CJK chars are individual tokens
# ===========================================================================

class TestCJKCharactersAreIndividualTokens:
    """Each CJK character must produce exactly one token."""

    def test_chinese_sentence_token_count_equals_char_count(self, splitter):
        text = "李明是一名工程师"
        toks = _token_strings(splitter, text)
        assert len(toks) == len(text), (
            f"Expected {len(text)} tokens, got {len(toks)}: {toks}"
        )

    def test_chinese_each_token_is_one_char(self, splitter):
        text = "李明是一名工程师"
        for tok, _, _ in _tokens(splitter, text):
            assert len(tok) == 1, f"Token {tok!r} has length > 1"

    def test_japanese_hiragana_individual_tokens(self, splitter):
        text = "きのうはいいてんきでした"
        toks = _token_strings(splitter, text)
        assert len(toks) == len(text)
        for t in toks:
            assert len(t) == 1

    def test_japanese_katakana_individual_tokens(self, splitter):
        text = "トウキョウ"
        toks = _token_strings(splitter, text)
        assert len(toks) == len(text)
        for t in toks:
            assert len(t) == 1

    def test_japanese_kanji_individual_tokens(self, splitter):
        text = "東京日本"
        toks = _token_strings(splitter, text)
        assert len(toks) == len(text)
        for t in toks:
            assert len(t) == 1

    def test_korean_hangul_individual_tokens(self, splitter):
        text = "서울한국"
        toks = _token_strings(splitter, text)
        assert len(toks) == len(text)
        for t in toks:
            assert len(t) == 1

    def test_cjk_extension_a_individual_tokens(self, splitter):
        # U+3400–U+4DBF: pick two Extension-A characters
        text = "㐀㐁"  # 㐀㐁
        toks = _token_strings(splitter, text)
        assert len(toks) == 2, f"Expected 2 tokens, got {len(toks)}: {toks}"


# ===========================================================================
# Character position correctness
# ===========================================================================

class TestCJKCharacterPositions:
    """Start/end offsets must round-trip back to the original string."""

    def test_chinese_positions_cover_every_char(self, splitter):
        text = "李明是工程师"
        triples = _tokens(splitter, text)
        for tok, start, end in triples:
            assert text[start:end] == tok, (
                f"text[{start}:{end}] = {text[start:end]!r} != {tok!r}"
            )

    def test_chinese_positions_are_contiguous(self, splitter):
        text = "你好世界"
        triples = _tokens(splitter, text)
        prev_end = 0
        for tok, start, end in triples:
            assert start == prev_end, (
                f"Gap between {prev_end} and {start} in {text!r}"
            )
            prev_end = end

    def test_korean_positions_cover_every_char(self, splitter):
        text = "서울은한국"
        for tok, start, end in _tokens(splitter, text):
            assert text[start:end] == tok

    def test_japanese_mixed_kana_positions(self, splitter):
        # Hiragana + Katakana interleaved
        text = "きトきト"
        for tok, start, end in _tokens(splitter, text):
            assert text[start:end] == tok

    def test_no_overlapping_spans(self, splitter):
        text = "北京上海广州"
        triples = _tokens(splitter, text)
        for i in range(len(triples) - 1):
            _, _, end_i = triples[i]
            _, start_j, _ = triples[i + 1]
            assert end_i <= start_j, "Overlapping token spans"


# ===========================================================================
# Span findability (the core NER use case)
# ===========================================================================

class TestCJKSpanFindability:
    """Entity spans must be locatable inside tokenised text."""

    def test_chinese_entity_at_start(self, splitter):
        text = "李明是一名工程师"
        text_toks = _token_strings(splitter, text)
        span_toks = _token_strings(splitter, "李明")
        assert _find_span(text_toks, span_toks), (
            f"Span {span_toks} not found in {text_toks}"
        )

    def test_chinese_entity_at_end(self, splitter):
        text = "这位工程师叫李明"
        text_toks = _token_strings(splitter, text)
        span_toks = _token_strings(splitter, "李明")
        assert _find_span(text_toks, span_toks)

    def test_chinese_entity_in_middle(self, splitter):
        text = "他在北京工作"
        text_toks = _token_strings(splitter, text)
        span_toks = _token_strings(splitter, "北京")
        assert _find_span(text_toks, span_toks)

    def test_single_cjk_char_entity(self, splitter):
        text = "他来自京"
        text_toks = _token_strings(splitter, text)
        span_toks = _token_strings(splitter, "京")
        assert _find_span(text_toks, span_toks)

    def test_japanese_entity_findable(self, splitter):
        text = "東京は日本の首都です"
        text_toks = _token_strings(splitter, text)
        span_toks = _token_strings(splitter, "東京")
        assert _find_span(text_toks, span_toks)

    def test_korean_entity_findable(self, splitter):
        text = "서울은한국의수도입니다"
        text_toks = _token_strings(splitter, text)
        span_toks = _token_strings(splitter, "서울")
        assert _find_span(text_toks, span_toks)

    def test_absent_span_not_found(self, splitter):
        text = "李明是工程师"
        text_toks = _token_strings(splitter, text)
        span_toks = _token_strings(splitter, "北京")
        assert not _find_span(text_toks, span_toks)

    def test_full_text_as_span(self, splitter):
        """The entire CJK text must be findable as a span."""
        text = "上海"
        text_toks = _token_strings(splitter, text)
        span_toks = _token_strings(splitter, text)
        assert _find_span(text_toks, span_toks)


# ===========================================================================
# Non-regression: Latin / URL / email / compound words still work
# ===========================================================================

class TestNonCJKPreservation:
    """Existing Latin-script tokenisation must be unaffected."""

    def test_english_single_words(self, splitter):
        assert _token_strings(splitter, "hello world") == ["hello", "world"]

    def test_english_multiple_words(self, splitter):
        toks = _token_strings(splitter, "Apple CEO Tim Cook")
        assert toks == ["apple", "ceo", "tim", "cook"]

    def test_hyphenated_word_stays_whole(self, splitter):
        assert _token_strings(splitter, "state-of-the-art") == ["state-of-the-art"]

    def test_underscored_word_stays_whole(self, splitter):
        assert _token_strings(splitter, "co_worker") == ["co_worker"]

    def test_url_is_single_token(self, splitter):
        toks = _token_strings(splitter, "visit https://example.com now")
        assert "https://example.com" in toks

    def test_email_is_single_token(self, splitter):
        toks = _token_strings(splitter, "mail user@example.com please")
        assert "user@example.com" in toks

    def test_punctuation_is_separate_token(self, splitter):
        toks = _token_strings(splitter, "end.")
        assert toks == ["end", "."]

    def test_numbers_are_single_token(self, splitter):
        assert _token_strings(splitter, "42") == ["42"]


# ===========================================================================
# Mixed CJK + Latin text
# ===========================================================================

class TestMixedScript:
    """Strings combining CJK and Latin must split each script correctly."""

    def test_latin_then_cjk(self, splitter):
        text = "CEO李明"
        toks = _token_strings(splitter, text)
        # "ceo" is one Latin token; 李, 明 are two individual CJK tokens
        assert toks == ["ceo", "李", "明"]

    def test_cjk_then_latin(self, splitter):
        text = "李明CEO"
        toks = _token_strings(splitter, text)
        assert toks == ["李", "明", "ceo"]

    def test_cjk_surrounded_by_latin(self, splitter):
        text = "hello北京world"
        toks = _token_strings(splitter, text)
        assert toks == ["hello", "北", "京", "world"]

    def test_latin_surrounded_by_cjk(self, splitter):
        text = "北京CEO上海"
        toks = _token_strings(splitter, text)
        assert toks == ["北", "京", "ceo", "上", "海"]

    def test_latin_entity_findable_in_mixed_text(self, splitter):
        text = "在Apple公司工作"
        text_toks = _token_strings(splitter, text)
        span_toks = _token_strings(splitter, "Apple")
        assert _find_span(text_toks, span_toks)

    def test_cjk_entity_findable_in_mixed_text(self, splitter):
        text = "Apple在北京发布"
        text_toks = _token_strings(splitter, text)
        span_toks = _token_strings(splitter, "北京")
        assert _find_span(text_toks, span_toks)

    def test_numbers_between_cjk(self, splitter):
        text = "第3章"
        toks = _token_strings(splitter, text)
        assert "第" in toks and "章" in toks
        assert "3" in toks

    def test_spaces_between_cjk_produce_same_tokens(self, splitter):
        """Spaces are just whitespace; CJK chars on both sides are individual tokens."""
        toks_nospace = _token_strings(splitter, "你好")
        toks_space = _token_strings(splitter, "你 好")
        assert toks_nospace == toks_space


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge cases that could trip up a naïve regex fix."""

    def test_empty_string(self, splitter):
        assert _token_strings(splitter, "") == []

    def test_single_cjk_char(self, splitter):
        assert _token_strings(splitter, "京") == ["京"]

    def test_single_latin_char(self, splitter):
        assert _token_strings(splitter, "a") == ["a"]

    def test_cjk_with_only_punctuation(self, splitter):
        text = "你好！"
        toks = _token_strings(splitter, text)
        assert "你" in toks and "好" in toks

    def test_repeated_cjk_char(self, splitter):
        text = "哈哈哈"
        toks = _token_strings(splitter, text)
        assert toks == ["哈", "哈", "哈"]

    def test_lowercasing_does_not_break_cjk(self, splitter):
        """CJK chars have no case; lower=True must not corrupt them."""
        toks_lower = list(splitter("李明", lower=True))
        toks_nolower = list(splitter("李明", lower=False))
        assert [t for t, _, _ in toks_lower] == [t for t, _, _ in toks_nolower]

    def test_cjk_extension_a_boundary_chars(self, splitter):
        """First and last chars of CJK Extension A are each one token."""
        first = "㐀"  # 㐀
        last = "䶿"   # 䶿
        assert _token_strings(splitter, first) == [first]
        assert _token_strings(splitter, last) == [last]

    def test_main_cjk_block_boundary_chars(self, splitter):
        """First and last chars of the main CJK block are each one token."""
        first = "一"  # 一
        last = "鿿"   # 鿿
        assert _token_strings(splitter, first) == [first]
        assert _token_strings(splitter, last) == [last]

    def test_hiragana_boundary_chars(self, splitter):
        first = "ぁ"  # ぁ (first assigned hiragana)
        last = "ゟ"   # ゟ
        assert _token_strings(splitter, first) == [first]
        assert _token_strings(splitter, last) == [last]

    def test_katakana_boundary_chars(self, splitter):
        first = "ァ"  # ァ
        last = "ヿ"   # ヿ
        assert _token_strings(splitter, first) == [first]
        assert _token_strings(splitter, last) == [last]

    def test_hangul_boundary_chars(self, splitter):
        first = "가"  # 가
        last = "힯"   # 힯
        assert _token_strings(splitter, first) == [first]
        assert _token_strings(splitter, last) == [last]
