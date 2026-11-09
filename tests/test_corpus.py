"""Tests for corpus preprocessing utilities."""

from __future__ import annotations

from interpretant.corpus.preprocessing import filter_empty, preprocess_text

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TestPreprocessText:
    def test_lowercases_input(self) -> None:
        assert preprocess_text("Hello World") == "hello world"

    def test_removes_inline_math(self) -> None:
        result = preprocess_text("The formula $E=mc^2$ is famous.")
        assert "$" not in result
        assert "e" in result  # "the" and "is" and "famous" should survive

    def test_removes_display_math(self) -> None:
        result = preprocess_text("See $$\\int_0^1 f(x) dx$$ for details.")
        assert "$$" not in result

    def test_removes_punctuation(self) -> None:
        result = preprocess_text("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_collapses_whitespace(self) -> None:
        result = preprocess_text("  multiple   spaces   here  ")
        assert "  " not in result
        assert result == result.strip()

    def test_unicode_normalisation(self) -> None:
        # Ligature 'ﬁ' → 'fi' after NFKC
        result = preprocess_text("ﬁeld")
        assert "ﬁ" not in result

    def test_empty_string(self) -> None:
        assert preprocess_text("") == ""

    def test_preserves_hyphenated_words(self) -> None:
        result = preprocess_text("state-of-the-art method")
        assert "state-of-the-art" in result

    def test_maps_target_plurals_and_british_spellings_to_lemmas(self) -> None:
        result = preprocess_text("Behaviour of Networks and their Representations")
        assert result == "behavior of network and their representation"

    def test_removes_urls_and_dois(self) -> None:
        result = preprocess_text("Code at https://github.com/x/y and doi 10.5281/zenodo.123 here")
        assert result == "code at and doi here"

    def test_leaves_ambiguous_and_non_target_forms_alone(self) -> None:
        assert preprocess_text("the media controls the networking") == (
            "the media controls the networking"
        )


class TestFilterEmpty:
    def test_removes_empty_strings(self) -> None:
        assert filter_empty(["a", "", "b"]) == ["a", "b"]

    def test_removes_whitespace_only(self) -> None:
        assert filter_empty(["a", "   ", "\t", "b"]) == ["a", "b"]

    def test_empty_list(self) -> None:
        assert filter_empty([]) == []

    def test_all_empty(self) -> None:
        assert filter_empty(["", " ", "\n"]) == []

    def test_preserves_order(self) -> None:
        texts = ["c", "a", "b"]
        assert filter_empty(texts) == texts
