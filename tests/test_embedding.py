"""Tests for the embedding helpers."""


def test_training_sentences_split_long_documents_so_gensim_sees_every_token() -> None:
    from interpretant.embedding.base import MAX_SENTENCE_TOKENS, training_sentences

    book = " ".join(f"w{index}" for index in range(MAX_SENTENCE_TOKENS * 2 + 5))

    sentences = list(training_sentences(book))

    assert [len(sentence) for sentence in sentences] == [
        MAX_SENTENCE_TOKENS,
        MAX_SENTENCE_TOKENS,
        5,
    ]
    assert sentences[-1][-1] == f"w{MAX_SENTENCE_TOKENS * 2 + 4}"
    assert list(training_sentences("   ")) == []
